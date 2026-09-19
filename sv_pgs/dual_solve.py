"""Stage 2's Gaussian E-step in dual form: every model, fold, probe and draw rides one pass.

A model is a trait on a sample mask. It has likelihood weights w (n,), zero off the mask (1/sigma^2
for a quantitative trait; the logistic curvature or EP likelihood-site precision, positive because
the logistic likelihood is log-concave, for a binary one), Gaussian site variances D = 1/Pi (p,)
>= 0, and the covariates C profiled out in its own weighted metric:

    H  = W^1/2 C (C'WC)^+ C' W^1/2,   Xt = (I - H) W^1/2 X,
    A  = Xt'Xt + diag(1/D)            (the posterior precision)
    S  = I + Xt D Xt'                 (n x n, every eigenvalue >= 1)

The mean is mu = m + D Xt' z with S z = b, b = (I - H) W^1/2 (y~ - X m), by the push-through
identity A^-1 Xt' = D Xt' S^-1. For any iterate with residual r = b - S z_hat,

    ||mu_hat - mu||_A^2 = r'(I - S^-1) r <= ||r||^2

(e = -D Xt' S^-1 r and Xt (D A D) Xt' = G S with G = S - I), so a column's exact residual norm is
its certificate against ep_eb.md's target ||e||_A^2 <= p_eff / K. Rows outside the training metric
(held-out folds, target people) satisfy ||X_out e||^2 <= lambda_max(X_out D X_out') ||r||^2 / 4.
H is the projector onto the column space of W^1/2 C, which can lose rank on a model's rows while C
has full rank on the cohort (`covariate_whitener`).

One pass reads each tile once and serves every column of every model:

    T = W^1/2 (I - H) V;  per tile: U = X_b' T;  U *= D_b[:, model];  Y += X_b U;  S V = V + (I - H) W^1/2 Y.

Nothing is p x columns except per-tile products. A binary model is a per-column weight, and a
fold is a mask, so neither needs a Gram or a factor. Site precisions that are not positive cannot
enter S (with q of them and A > 0, S has exactly q negative eigenvalues, by Sylvester's law of
inertia); they are eliminated exactly by the caller's split.

Tiles are code_products.CodeBlockTile (or DenseDualTile): a read prepares its sample operand once,
with a normwise relative error per column ||delta_k|| <= relative_error ||operand_k|| (the int8
digit split, exact zeros kept); relative_error 0 means the exact products.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Iterator, Protocol

import numpy as np

from sv_pgs.marginal_variances import BlockGrams, BulkSolve, WindowCross

DIGIT_BITS = 7
"""Bits per balanced base-128 operand digit of the int8 split."""


class DualTile(Protocol):
    """One LD block's standardized genotype columns X_b (n x p_b): code_products.CodeBlockTile's interface.

    `sample_operand(left, relative_error)` prepares a read's sample-side operand once for every
    block (relative_error > 0); `rmatmat` takes that operand or a plain array (exact);
    `accumulate_matmat(right, image, relative_error)` adds X_b R~ into the read's image; `matmat` is
    exact. The error budget is normwise per column: ||L~_k - L_k||_2 <= relative_error ||L_k||_2.
    """

    def sample_operand(self, left: Any, relative_error: float) -> Any:
        ...

    def rmatmat(self, left: Any) -> Any:
        """X_b' left for left (n, K) or its prepared operand: (p_b, K)."""
        ...

    def matmat(self, right: Any) -> Any:
        """X_b right for right (p_b, K): (n, K), exact."""
        ...

    def accumulate_matmat(self, right: Any, image: Any, relative_error: float) -> None:
        """image += X_b R~ with ||R~_k - right_k||_2 <= relative_error ||right_k||_2."""
        ...

    def weighted_column_squares(self, weights: Any) -> Any:
        """(X_b * X_b)' weights for weights (n, M): (p_b, M)."""
        ...

    def columns(self, local: Any) -> Any:
        """X_b[:, local] as dense float64: (n, len(local))."""
        ...


class DualTileSource(Protocol):
    array_module: Any
    sample_count: int
    variant_count: int
    block_bounds: list[tuple[int, int]]

    def blocks(self) -> Iterator[tuple[int, int, DualTile]]:
        """(start, stop, tile) for every block, in variant order, covering 0..p-1."""
        ...

    def map_reduce(self, work: Callable[..., None], shared: dict, rows: dict, image_shape: tuple[int, ...]) -> Any:
        """sum over blocks of what work(start, stop, tile, shared, rows_b, image) adds into a zero image.

        `shared` holds sample-side arrays every block reads (work may cache per-device state in it),
        `rows` variant-side arrays of which each block gets its rows. A source over several devices
        runs its blocks on all of them at once and sums the devices' images in a fixed order.
        """
        ...


def rounded_operand(values: Any, relative_error: float, array_module: Any) -> Any:
    """The operand the int8 digit split multiplies: the fewest digits meeting the normwise bound.

    m digits keep each entry within 2^-(7m - 2) of its column's maximum and leave zeros zero, so a
    column's rounding is at most sqrt(nnz) 2^-(7m - 2) max|v|. m is the least count meeting
    relative_error * ||v|| for every column; once 2^-(7m - 2) reaches float64's resolution the
    operand is exact.
    """
    if relative_error <= 0.0:
        return values
    magnitude = array_module.max(array_module.abs(values), axis=0)
    norm = array_module.linalg.norm(values, axis=0)
    support = array_module.sum(values != 0.0, axis=0)
    needed = array_module.sqrt(support) * magnitude / array_module.maximum(relative_error * norm, np.finfo(np.float64).tiny)
    exponent = float(array_module.max(array_module.log2(array_module.maximum(needed, 1.0))))
    digits = int(np.ceil((exponent + 2.0) / DIGIT_BITS))
    if DIGIT_BITS * digits - 2 >= np.finfo(np.float64).nmant:
        return values
    scale = array_module.exp2((DIGIT_BITS * digits - 2) - array_module.ceil(array_module.log2(array_module.where(magnitude > 0, magnitude, 1.0))))
    return array_module.rint(values * scale[None, :]) / scale[None, :]


class DenseDualTile:
    """A block held as dense standardized columns; its products round the operand as the int8 split does."""

    def __init__(self, values: Any, array_module: Any) -> None:
        self.values = values
        self.array_module = array_module

    def sample_operand(self, left: Any, relative_error: float) -> Any:
        return rounded_operand(left, relative_error, self.array_module)

    def rmatmat(self, left: Any) -> Any:
        return self.values.T @ left

    def matmat(self, right: Any) -> Any:
        return self.values @ right

    def accumulate_matmat(self, right: Any, image: Any, relative_error: float) -> None:
        image += self.values @ rounded_operand(right, relative_error, self.array_module)

    def weighted_column_squares(self, weights: Any) -> Any:
        return (self.values * self.values).T @ weights

    def columns(self, local: Any) -> Any:
        return self.values[:, local]


class DenseDualSource:
    """Dense standardized genotypes (n x p) cut into contiguous blocks."""

    def __init__(self, genotypes: Any, block_bounds: list[tuple[int, int]], array_module: Any = np) -> None:
        self.array_module = array_module
        self.genotypes = array_module.asarray(genotypes, dtype=array_module.float64)
        self.sample_count, self.variant_count = (int(extent) for extent in self.genotypes.shape)
        self.block_bounds = _checked_bounds(block_bounds, self.variant_count)

    def blocks(self) -> Iterator[tuple[int, int, DualTile]]:
        for start, stop in self.block_bounds:
            yield start, stop, DenseDualTile(self.genotypes[:, start:stop], self.array_module)

    def map_reduce(self, work: Callable[..., None], shared: dict, rows: dict, image_shape: tuple[int, ...]) -> Any:
        return sequential_map_reduce(self, work, shared, rows, image_shape)


def _checked_bounds(block_bounds: list[tuple[int, int]], variant_count: int) -> list[tuple[int, int]]:
    bounds = [(int(start), int(stop)) for start, stop in block_bounds]
    if not bounds or bounds[0][0] != 0 or bounds[-1][1] != variant_count or any(
        previous[1] != following[0] or following[1] <= following[0] for previous, following in zip(bounds, bounds[1:])
    ):
        raise ValueError("blocks must cover every variant once, in order.")
    return bounds


def sequential_map_reduce(source: DualTileSource, work: Callable[..., None], shared: dict, rows: dict, image_shape: tuple[int, ...]) -> Any:
    """map_reduce on one device: every block in order into one image."""
    image = source.array_module.zeros(image_shape)
    device_shared = dict(shared)
    for start, stop, tile in source.blocks():
        work(start, stop, tile, device_shared, {name: values[start:stop] for name, values in rows.items()}, image)
    return image


class StreamedDualSource:
    """A streamed block source (store_block_source.StoreGenotypeBlockSource) as a DualTileSource.

    Its tiles are code_products.CodeBlockTile, which already has the DualTile interface; the blocks
    must be contiguous runs of the model's variant axis, in order.
    """

    def __init__(self, source: Any) -> None:
        self.source = source
        self.array_module = source.array_module
        self.sample_count = int(source.sample_count)
        indices = [np.asarray(block, dtype=np.int64) for block in source.block_variant_indices]
        if any(not np.array_equal(block, np.arange(block[0], block[-1] + 1)) for block in indices):
            raise ValueError("every block must be a contiguous run of variants.")
        self.variant_count = int(indices[-1][-1] + 1)
        self.block_bounds = _checked_bounds([(int(block[0]), int(block[-1]) + 1) for block in indices], self.variant_count)

    def blocks(self) -> Iterator[tuple[int, int, DualTile]]:
        for block_index, tile in self.source.iter_tiles():
            start, stop = self.block_bounds[block_index]
            yield start, stop, tile

    def map_reduce(self, work: Callable[..., None], shared: dict, rows: dict, image_shape: tuple[int, ...]) -> Any:
        return sequential_map_reduce(self, work, shared, rows, image_shape)


def _host(values: Any) -> np.ndarray:
    return np.asarray(values.get() if hasattr(values, "get") else values)


def covariate_whitener(array_module: Any, weighted_covariates: Any) -> Any:
    """F (k x k) with F F' = (A'A)^+ on the numerical column space of A = W^1/2 C (n x k).

    The covariates are shared by every trait, so a model's rows can take their rank away: a
    sex-restricted disease (prostate cancer: men only) makes the sex indicator equal the intercept and
    age x female zero, and a fold can hold no member of a rare level. Singular values of A at or below
    the numerical-rank tolerance max(n, k) eps s_max (Golub and Van Loan 5.4.1) span no direction of
    its column space; the others give F = V diag(1/s) from A's own SVD, zero columns for the rest, so
    the Gram's squared condition number never enters. A F has orthonormal (or zero) columns, so
    H = A F F' A' is the projector onto A's column space and F F' A'y the minimum-norm coefficients.
    The SVD is taken of R in A = QR (Householder, backward stable), which has A's singular values and
    right vectors without forming the n x k left ones.
    """
    triangle = array_module.linalg.qr(weighted_covariates, mode="r")
    _, singular_values, right_vectors = array_module.linalg.svd(triangle, full_matrices=False)
    kept = singular_values > max(weighted_covariates.shape) * np.finfo(np.float64).eps * singular_values[0]
    inverse = array_module.where(kept, 1.0 / array_module.where(kept, singular_values, 1.0), 0.0)
    return right_vectors.T * inverse[None, :]


class DualModels:
    """Weights (n, M), site variances (p, M) and shared covariates (n, k) of every model.

    `covariate_factor` (M, k, k) holds each model's `covariate_whitener`: F_m F_m' = (C'W_m C)^+.
    """

    def __init__(self, weights: Any, variances: Any, covariates: Any, array_module: Any = np) -> None:
        self.array_module = array_module
        self.weights = array_module.asarray(weights, dtype=array_module.float64)
        self.variances = array_module.asarray(variances, dtype=array_module.float64)
        self.covariates = array_module.asarray(covariates, dtype=array_module.float64)
        if bool(array_module.any(self.variances < 0.0)):
            raise ValueError("site variances must be non-negative; negative sites need the exact split.")
        self.root_weights = array_module.sqrt(self.weights)
        self.covariate_factor = array_module.stack(
            [covariate_whitener(array_module, self.root_weights[:, model : model + 1] * self.covariates) for model in range(self.model_count)]
        )

    @property
    def model_count(self) -> int:
        return int(self.weights.shape[1])

    def covariate_solve(self, right: Any, column_models: Any) -> Any:
        """(C'W_m C)^+ r for each column r of `right` (k x columns), with m the column's model."""
        factor = self.covariate_factor[column_models]
        whitened = self.array_module.einsum("cab,ac->cb", factor, right)
        return self.array_module.einsum("cab,cb->ac", factor, whitened)

    def complement(self, values: Any, column_models: Any) -> Any:
        """(I - H_m) v per column, with H_m the weighted covariate projector of the column's model."""
        root = self.root_weights[:, column_models]
        return values - root * (self.covariates @ self.covariate_solve(self.covariates.T @ (root * values), column_models))

    def sample_to_design(self, values: Any, column_models: Any) -> Any:
        """The sample-side operand of X_b': Xt'v = X' [W^1/2 (I - H) v]."""
        return self.root_weights[:, column_models] * self.complement(values, column_models)

    def design_to_sample(self, image: Any, column_models: Any) -> Any:
        """Xt u = (I - H) W^1/2 [X u]."""
        return self.complement(self.root_weights[:, column_models] * image, column_models)


@dataclass
class PassCount:
    """Reads of the store, and the columns and operand error each carried."""

    passes: int = 0
    column_passes: int = 0
    records: list = field(default_factory=list)

    def note(self, columns: int, relative_error: float, label: str) -> None:
        self.passes += 1
        self.column_passes += columns
        self.records.append((label, columns, relative_error))


def _accumulate(tile: DualTile, right: Any, image: Any, relative_error: float) -> None:
    if relative_error <= 0.0:
        image += tile.matmat(right)
    else:
        tile.accumulate_matmat(right, image, relative_error)


def _operator_block(relative_error: float) -> Callable[..., None]:
    """One block of S V's read: X_b' L, scaled by D_b per column, back through X_b into the image.

    With a relaxed error the read's sample operand is prepared at a device's first block and reused
    by its other blocks (code_products.CodeBlockTile.sample_operand)."""

    def work(_start: int, _stop: int, tile: DualTile, shared: dict, rows: dict, image: Any) -> None:
        if relative_error > 0.0 and "operand" not in shared:
            shared["operand"] = tile.sample_operand(shared["left"], relative_error)
        products = tile.rmatmat(shared["operand"] if relative_error > 0.0 else shared["left"])
        scaled = rows["variances"][:, shared["column_models"]] * products
        _accumulate(tile, scaled if scaled.flags.c_contiguous else scaled.copy(order="C"), image, relative_error)

    return work


def apply_operator(source: DualTileSource, models: DualModels, values: Any, column_models: Any, relative_error: float, count: PassCount, label: str) -> Any:
    """S V for every column, one read over every device of the source."""
    left = models.sample_to_design(values, column_models)
    image = source.map_reduce(
        _operator_block(relative_error), {"left": left, "column_models": column_models}, {"variances": models.variances}, tuple(values.shape)
    )
    count.note(int(values.shape[1]), relative_error, label)
    return values + models.design_to_sample(image, column_models)


def mean_right_hand_side(models: DualModels, working_response: Any, prior_image: Any) -> Any:
    """b_m = (I - H_m) W_m^1/2 (y~_m - X m_m); prior_image is X m per model (n, M)."""
    model_columns = models.array_module.arange(models.model_count)
    return models.complement(models.root_weights * (working_response - prior_image), model_columns)


def _orthonormal_block(array_module: Any, values: Any) -> Any:
    """An orthonormal basis of a model's columns (SVQB), dropping directions below float64 resolution."""
    gram = values.T @ values
    eigenvalues, eigenvectors = array_module.linalg.eigh(0.5 * (gram + gram.T))
    largest = float(eigenvalues[-1]) if eigenvalues.shape[0] else 0.0
    resolvable = eigenvalues > largest * values.shape[0] * np.finfo(np.float64).eps
    if not bool(array_module.any(resolvable)):
        return values[:, :0]
    return values @ (eigenvectors[:, resolvable] / array_module.sqrt(eigenvalues[resolvable]))


def _cholesky_solve(array_module: Any, factor: Any, right: Any) -> Any:
    return array_module.linalg.solve(factor.T, array_module.linalg.solve(factor, right))


def column_squares(source: DualTileSource, models: DualModels, count: PassCount) -> Any:
    """||xt_k||^2 for every variant and model, one read: x_k'W x_k - ||F'(C'Wx_k)||^2, F F' = (C'WC)^+."""
    array_module = source.array_module
    squares = array_module.zeros((source.variant_count, models.model_count))
    covariate_count = int(models.covariates.shape[1])
    weighted_covariates = array_module.concatenate(
        [models.weights[:, model : model + 1] * models.covariates for model in range(models.model_count)], axis=1
    )
    for start, stop, tile in source.blocks():
        cross = tile.rmatmat(weighted_covariates).reshape(stop - start, models.model_count, covariate_count)
        whitened = array_module.einsum("vma,mab->vmb", cross, models.covariate_factor)
        squares[start:stop] = tile.weighted_column_squares(models.weights) - array_module.sum(whitened * whitened, axis=2)
    count.note(models.model_count * (1 + covariate_count), 0.0, "column-squares")
    return squares


def resolved_spikes(spikes: Any, sample_count: int, array_module: Any) -> np.ndarray:
    """The variants whose spike D_k ||xt_k||^2 exceeds 1 plus the bulk's mean eigenvalue (a fixed point)."""
    chosen = array_module.zeros(spikes.shape[0], dtype=bool)
    while True:
        level = float(array_module.sum(array_module.where(chosen, 0.0, spikes))) / sample_count
        updated = spikes > 1.0 + level
        if bool(array_module.all(updated == chosen)):
            return _host(array_module.flatnonzero(chosen))
        chosen = updated


def deflation_factor(array_module: Any, basis: Any, image: Any) -> Any:
    """F with F F' = (W'SW)^+ on the numerical column space of a spike basis W (n x k), from S W = `image`.

    Spikes tied on a model's rows (variants that differ only off them) give W equal columns, and
    W'SW, whose null space is W's since S >= I, is singular: a Cholesky factorization fails, or passes
    only on a pivot that rounding left positive. The computed Gram W'(SW) is within
    gamma_n |W|'|SW| of the exact one (Higham 2002, 3.5; gamma_n = n u / (1 - n u) <= n eps for
    u = eps / 2), so its error is below n eps ||W||_F ||SW||_F in norm and, by Weyl, an eigenvalue
    at or below that spans no direction.
    The others give F = V diag(lambda^-1/2): W F F' W' S is the S-orthogonal projector onto span(W),
    so the deflated iteration is the one on the independent spike directions.
    """
    gram = basis.T @ image
    eigenvalues, eigenvectors = array_module.linalg.eigh(0.5 * (gram + gram.T))
    rounding = basis.shape[0] * np.finfo(np.float64).eps * array_module.linalg.norm(basis) * array_module.linalg.norm(image)
    kept = eigenvalues > rounding
    return eigenvectors * array_module.where(kept, 1.0 / array_module.sqrt(array_module.where(kept, eigenvalues, 1.0)), 0.0)[None, :]


@dataclass
class Deflation:
    """Per model, a basis W (n x k) of its spike directions, S W, F with F F' = (W'SW)^+
    (`deflation_factor`), and the variants W holds.

    Deflated CG (Saad, Yeung, Erhel & Guyomarc'h 2000) keeps every residual orthogonal to W and
    every direction S-orthogonal to it: the spikes W spans are solved exactly in the k x k system
    W'SW, and CG sees only the rest of the spectrum.
    """

    bases: dict
    images: dict
    factors: dict
    indices: dict

    def project_start(self, array_module: Any, model: int, solution: Any, residual: Any) -> tuple[Any, Any]:
        factor = self.factors[model]
        coefficients = factor @ (factor.T @ (self.bases[model].T @ residual))
        return solution + self.bases[model] @ coefficients, residual - self.images[model] @ coefficients

    def project_direction(self, array_module: Any, model: int, values: Any) -> Any:
        factor = self.factors[model]
        return values - self.bases[model] @ (factor @ (factor.T @ (self.images[model].T @ values)))


def spike_deflation(
    source: DualTileSource, models: DualModels, count: PassCount, edges: dict | None = None, column_budget: int | None = None
) -> tuple[Deflation, dict]:
    """Each model's spikes and their deflation basis: one pass for ||xt_k||^2, one for S W.

    Variant k adds D_k ||xt_k||^2 to G = Xt D Xt' along Xt e_k. Without `edges`, k is resolved when
    that spike exceeds 1 plus the bulk's mean eigenvalue tr(G_bulk)/n (iterated to its fixed point,
    since removing spikes lowers the level). With `edges` (model -> the largest Ritz value of the
    model's deflated operator in an earlier solve), only spikes above the bulk's measured edge are
    resolved: a spike below it does not raise the condition number.
    ||xt_k||^2 = x_k'W x_k - (x_k'WC)(C'WC)^-1(C'Wx_k) for every model comes from the first pass.
    `column_budget` caps the basis columns over all models, from the memory the caller has for W,
    S W and the S W pass: the largest spikes, which cost CG the most iterations, are kept.
    """
    array_module = source.array_module
    spikes = models.variances * column_squares(source, models, count)
    resolved: dict[int, np.ndarray] = {}
    for model in range(models.model_count):
        if edges is not None and model in edges:
            resolved[model] = _host(array_module.flatnonzero(1.0 + spikes[:, model] > edges[model]))
        else:
            resolved[model] = resolved_spikes(spikes[:, model], source.sample_count, array_module)
    if column_budget is not None and sum(indices.size for indices in resolved.values()) > column_budget:
        ranked = sorted(
            ((float(spikes[int(index), model]), model, int(index)) for model, indices in resolved.items() for index in indices),
            reverse=True,
        )[:column_budget]
        resolved = {model: np.sort(np.asarray([index for _spike, owner, index in ranked if owner == model], dtype=np.int64)) for model in resolved}
    bases: dict[int, Any] = {}
    for model, indices in resolved.items():
        if indices.size == 0:
            continue
        genotype_columns = array_module.concatenate(
            [tile.columns(indices[(indices >= start) & (indices < stop)] - start) for start, stop, tile in source.blocks()], axis=1
        )
        bases[model] = models.design_to_sample(genotype_columns, array_module.full(indices.size, model))
    images: dict[int, Any] = {}
    factors: dict[int, Any] = {}
    if bases:
        order = sorted(bases)
        stacked_models = np.concatenate([np.full(int(bases[model].shape[1]), model) for model in order])
        image = apply_operator(source, models, array_module.concatenate([bases[model] for model in order], axis=1), array_module.asarray(stacked_models), 0.0, count, "deflation")
        offset = 0
        for model in order:
            width = int(bases[model].shape[1])
            images[model] = image[:, offset : offset + width]
            offset += width
            factors[model] = deflation_factor(array_module, bases[model], images[model])
    return Deflation(bases, images, factors, resolved), {model: int(indices.size) for model, indices in resolved.items()}


@dataclass
class SolveResult:
    solution: Any
    residual: Any
    residual_norm: Any
    iterations: int
    restarts: int
    relative_errors: list
    operator_scale: float
    model_scales: dict


def _conjugate_gradient_rate(operator_scale: float) -> float:
    """-ln of CG's worst-case contraction per iteration on an operator with spectrum in [1, operator_scale]."""
    root = np.sqrt(operator_scale)
    return float(-np.log((root - 1.0) / (root + 1.0))) if operator_scale > 1.0 else np.inf


def recursive_share(log_ratio: float, rate: float, operator_scale: float, sample_count: int) -> float:
    """The share s of a residual bound B given to the recursive residual; the rest bounds the product drift.

    A cycle that reduces the residual from r_0 to s B takes N(s) = (L + ln(1/s)) / c iterations
    (L = ln(r_0 / B) > 0, c the per-iteration log contraction). Spreading the drift budget (1 - s) B
    over them needs operands of relative error eps = (1 - s) B / (2 lambda r N), with r ~ sqrt(r_0 s B)
    over the cycle, and the int8 split writes such an operand in d(s) = log_128(sqrt(n) / eps) digits
    (a continuous relaxation of the integer count). With D = d ln 128 = const + ln N + (1/2) ln s -
    ln(1 - s), the stationarity of the cycle's digit-passes N(s) d(s) in s is

        (1 + s)(L + ln(1/s)) = 2 (1 - s)(D(s) + 1).

    The left side minus the right is negative as s -> 0 and tends to 2L > 0 as s -> 1, so a root
    exists and bisection finds one, to float64 resolution.
    """

    def excess(share: float) -> float:
        iterations = (log_ratio + np.log(1.0 / share)) / rate
        inverse_error = 2 * operator_scale * iterations * np.sqrt(np.exp(log_ratio) * share) / (1.0 - share)
        scaled_digits = max(0.0, float(np.log(np.sqrt(sample_count) * inverse_error)))
        return (1.0 + share) * (log_ratio + np.log(1.0 / share)) - 2 * (1.0 - share) * (scaled_digits + 1.0)

    low, high = np.finfo(np.float64).tiny, 1.0 - np.finfo(np.float64).epsneg
    while high - low > np.finfo(np.float64).eps * high:
        middle = 0.5 * (low + high)
        if excess(middle) < 0.0:
            low = middle
        else:
            high = middle
    return float(high)


def certified_block_cg(
    source: DualTileSource,
    models: DualModels,
    right_hand_side: Any,
    start: Any,
    column_models: Any,
    residual_bound: Any,
    count: PassCount,
    operator_scale: float = 1.0,
    deflation: Deflation | None = None,
    label: str = "cg",
) -> SolveResult:
    """Solve S_m z = b for every column; each model's columns share one block Krylov space.

    A column is done when its exact residual meets ||b - S z|| <= residual_bound (the certificate).
    Iterations run with relaxed operand error (van den Eshof and Sleijpen 2004): iteration k's
    product error F_k moves the true residual away from the recursive one by at most
    ||F_k|| ||r_k|| (S >= I and orthonormal directions give ||alpha_k|| <= ||r_k||), and
    ||F_k|| <= 2 lambda eps_k for two operands of relative error eps_k. Each restart cycle splits a
    column's bound B into s B for the recursive residual and (1 - s) B for the drift, with s from
    recursive_share, and each iteration spends at most the drift budget still left over the
    iterations still expected, eps_k = (budget left) / (2 lambda ||r_k|| N_left), so the spent drift
    never exceeds (1 - s) B whatever N_left turns out to be. A column stops when its recursive
    residual plus its spent drift is below B. lambda is the largest Ritz value seen, a lower bound on
    lambda_max, so the drift bound is an estimate: the certificate is the exact residual each
    restart begins with, and a column that misses it continues.

    With no scale known (operator_scale 1) the first iteration runs exact to find one. The error is
    relaxed only on a deflated operator: with its spikes still in S, relaxed products delayed CG by
    3-22 iterations (finite-precision CG re-converges its outlying Ritz values), and on the deflated
    operator they cost none (measured on the s1M_100k real-haplotype subset). A restart from a zero
    start needs no product, since then r = b. `deflation` solves each model's spikes exactly and CG
    the rest.

    There is no iteration cap. A bound below what float64 can attain for a column,
    (n + p) eps lambda_max ||z||, the rounding of one exact product S z, is refused up front, and a
    restart whose exact residual did not fall below the previous one fails loudly instead of looping.
    So does a residual that is not finite: NaN compares false against every bound, so it would
    otherwise pass as certified.
    """
    array_module = source.array_module
    host_models = _host(column_models)
    solution = start.copy()
    bound = array_module.asarray(residual_bound, dtype=array_module.float64)
    relative_errors: list = []
    model_scales: dict[int, float] = {}
    scale_known = operator_scale > 1.0
    relaxed = deflation is not None
    iterations = 0
    restarts = 0
    zero_start = not bool(array_module.any(solution != 0.0))
    while True:
        if zero_start and restarts == 0:
            residual = right_hand_side.copy()
        else:
            residual = right_hand_side - apply_operator(source, models, solution, column_models, 0.0, count, f"{label}:exact")
        norms = array_module.linalg.norm(residual, axis=0)
        if not bool(array_module.all(array_module.isfinite(norms))):
            raise FloatingPointError("an exact residual is not finite: the right-hand side or the operator holds NaN or inf.")
        open_mask = _host(norms > bound)
        if not open_mask.any():
            return SolveResult(solution, residual, norms, iterations, restarts, relative_errors, operator_scale, model_scales)
        attainable = (source.sample_count + source.variant_count) * np.finfo(np.float64).eps * operator_scale * array_module.linalg.norm(solution, axis=0)
        if bool(array_module.any((bound <= attainable)[array_module.asarray(np.flatnonzero(open_mask))])):
            raise ValueError("a residual bound lies below the accuracy float64 can attain for S z.")
        if restarts and bool(array_module.any((norms >= previous_norms)[array_module.asarray(np.flatnonzero(open_mask))])):
            raise FloatingPointError("an exact residual did not fall across a restart; the solve stagnated.")
        previous_norms = norms
        restarts += 1
        open_columns = array_module.asarray(np.flatnonzero(open_mask))
        # This cycle's split of each bound between the recursive residual and the product drift.
        drift_budget = array_module.zeros_like(bound)
        spent = array_module.zeros_like(bound)
        cycle_start_worst = float(array_module.max(norms[open_columns] / bound[open_columns]))
        rate = _conjugate_gradient_rate(operator_scale)
        if relaxed and scale_known:
            share = recursive_share(float(np.log(cycle_start_worst)), rate, operator_scale, source.sample_count)
            drift_budget = (1.0 - share) * bound
        directions: dict[int, tuple[np.ndarray, Any]] = {}
        for model in np.unique(host_models[open_mask]):
            model = int(model)
            columns = np.flatnonzero((host_models == model) & open_mask)
            seed = residual[:, columns]
            if deflation is not None and model in deflation.bases:
                solution[:, columns], residual[:, columns] = deflation.project_start(array_module, model, solution[:, columns], residual[:, columns])
                seed = deflation.project_direction(array_module, model, residual[:, columns])
            block = _orthonormal_block(array_module, seed)
            if block.shape[1]:
                directions[model] = (columns, block)
        cycle_iterations = 0
        while directions:
            order = sorted(directions)
            block_models = array_module.asarray(np.concatenate([np.full(directions[model][1].shape[1], model) for model in order]))
            stacked = array_module.concatenate([directions[model][1] for model in order], axis=1)
            residual_norms = array_module.linalg.norm(residual, axis=0)
            relative_error = 0.0
            if relaxed and scale_known:
                live_columns = np.concatenate([directions[model][0] for model in order])
                live = array_module.asarray(live_columns)
                recursive_target = bound[live] - drift_budget[live]
                left = array_module.maximum(array_module.log(residual_norms[live] / recursive_target) / rate, 1.0)
                budget_left = array_module.maximum(drift_budget[live] - spent[live], 0.0)
                relative_error = float(array_module.min(budget_left / (2 * operator_scale * residual_norms[live] * left)))
            image = apply_operator(source, models, stacked, block_models, relative_error, count, f"{label}:{iterations}")
            iterations += 1
            cycle_iterations += 1
            relative_errors.append(relative_error)
            next_directions: dict[int, tuple[np.ndarray, Any]] = {}
            offset = 0
            for model in order:
                columns, block = directions[model]
                width = int(block.shape[1])
                applied = image[:, offset : offset + width]
                offset += width
                curvature = 0.5 * (block.T @ applied + applied.T @ block)
                ritz = float(array_module.linalg.eigvalsh(curvature)[-1])
                model_scales[model] = max(model_scales.get(model, 1.0), ritz)
                device_columns = array_module.asarray(columns)
                spent[device_columns] += 2 * operator_scale * relative_error * residual_norms[device_columns]
                operator_scale = max(operator_scale, ritz)
                live = columns[_host(residual_norms[device_columns] + spent[device_columns] > bound[device_columns])]
                if live.size == 0:
                    continue
                step = array_module.linalg.solve(curvature, block.T @ residual[:, live])
                solution[:, live] += block @ step
                residual[:, live] -= applied @ step
                live_device = array_module.asarray(live)
                still = live[_host(array_module.linalg.norm(residual[:, live], axis=0) + spent[live_device] > bound[live_device])]
                if still.size == 0:
                    continue
                candidate = residual[:, still]
                if deflation is not None and model in deflation.bases:
                    candidate = deflation.project_direction(array_module, model, candidate)
                conjugation = -array_module.linalg.solve(curvature, applied.T @ candidate)
                orthonormal = _orthonormal_block(array_module, candidate + block @ conjugation)
                if orthonormal.shape[1]:
                    next_directions[model] = (still, orthonormal)
            if not scale_known:
                scale_known = True
                rate = _conjugate_gradient_rate(operator_scale)
                if relaxed:
                    share = recursive_share(float(np.log(cycle_start_worst)), rate, operator_scale, source.sample_count)
                    drift_budget = (1.0 - share) * bound
            worst = float(array_module.max(array_module.linalg.norm(residual[:, open_columns], axis=0) / bound[open_columns]))
            if worst < cycle_start_worst:
                # The measured average contraction replaces the worst-case one once the cycle has made progress.
                rate = float(np.log(cycle_start_worst / worst)) / cycle_iterations
            directions = next_directions


def refresh_pass(source: DualTileSource, models: DualModels, duals: Any, column_models: Any, block_update: Callable[[int, int, Any], tuple[Any, Any]], count: PassCount) -> tuple[Any, Any]:
    """A refresh with no pass of its own: lagged products, the block-local step and the new product.

    Per tile: U_b = Xt_b' z for every column (the previous solve's products, which the local step
    consumes); (D_b, m_b) = block_update(start, stop, U_b) writes block b's new sites; then
    Y += X_b (D_b U_b) and X m += X_b m_b under the new sites. Each block's second product uses
    only its own D_b, so after the read S_new z and X m_new are exact. Returns (S_new z, X m_new).
    """
    array_module = source.array_module
    left = models.sample_to_design(duals, column_models)
    image = array_module.zeros_like(duals)
    prior_image = array_module.zeros((source.sample_count, models.model_count))
    column_count = int(duals.shape[1])
    for start, stop, tile in source.blocks():
        products = tile.rmatmat(left)
        variances, means = block_update(start, stop, products)
        if bool(array_module.any(variances < 0.0)):
            raise ValueError("a block update returned a negative site variance; such sites need the exact split.")
        models.variances[start:stop] = variances
        combined = tile.matmat(array_module.concatenate([variances[:, column_models] * products, means], axis=1))
        image += combined[:, :column_count]
        prior_image += combined[:, column_count:]
    count.note(column_count + models.model_count, 0.0, "refresh")
    return duals + models.design_to_sample(image, column_models), prior_image


def mean_from_dual(source: DualTileSource, models: DualModels, prior_mean: Any, duals: Any, count: PassCount) -> Any:
    """mu = m + D Xt' z per model, one read."""
    array_module = source.array_module
    left = models.sample_to_design(duals, array_module.arange(models.model_count))
    mean = prior_mean.copy()
    for start, stop, tile in source.blocks():
        mean[start:stop] += models.variances[start:stop] * tile.rmatmat(left)
    count.note(int(duals.shape[1]), 0.0, "mean")
    return mean


def draw_right_hand_side(source: DualTileSource, models: DualModels, draw_models: Any, prior_noise: Any, sample_noise: Any, count: PassCount) -> Any:
    """Matheron's dual right-hand side e2 - Xt D^1/2 e1 per draw, one read.

    beta* = mu + D^1/2 e1 + D Xt' S^-1 (e2 - Xt D^1/2 e1) has covariance A^-1 exactly: the map from
    (e1, e2) has Gram D - D Xt' S^-1 Xt D.
    """
    array_module = source.array_module
    image = array_module.zeros((source.sample_count, int(draw_models.shape[0])))
    for start, stop, tile in source.blocks():
        image += tile.matmat(array_module.sqrt(models.variances[start:stop][:, draw_models]) * prior_noise[start:stop])
    count.note(int(draw_models.shape[0]), 0.0, "draw-rhs")
    return sample_noise - models.design_to_sample(image, draw_models)


def fused_final_pass(
    source: DualTileSource,
    models: DualModels,
    prior_mean: Any,
    duals: Any,
    column_models: Any,
    right_hand_side: Any,
    draw_columns: Any,
    prior_noise: Any,
    score_rows: Any,
    count: PassCount,
    design_columns: Any = None,
) -> tuple[Any, Any, Any, Any]:
    """Weights, scores, exact certificates and chosen Xt'z, one read.

    Per tile: U_b = Xt_b' z for every column; the model columns give mu_b = m_b + D_b U_b, the draw
    columns beta*_b = mu_b + D_b^1/2 e1_b + D_b U_b; one product with [D_b U_b, weights] gives both
    S z (hence each column's exact residual) and the scores of `score_rows`.
    Returns (weights (p, columns), scores (score rows, columns), exact residual norms, Xt'z of
    `design_columns` or None).
    """
    array_module = source.array_module
    left = models.sample_to_design(duals, column_models)
    column_count = int(duals.shape[1])
    image = array_module.zeros_like(duals)
    weights = array_module.empty((source.variant_count, column_count))
    scores = array_module.zeros((int(score_rows.shape[0]), column_count))
    model_columns = array_module.arange(models.model_count)
    design_products = None if design_columns is None else array_module.empty((source.variant_count, int(len(design_columns))))
    for start, stop, tile in source.blocks():
        products = tile.rmatmat(left)
        if design_products is not None:
            design_products[start:stop] = products[:, design_columns]
        variances = models.variances[start:stop][:, column_models]
        shifted = variances * products
        block_weights = shifted.copy()
        block_weights[:, model_columns] += prior_mean[start:stop]
        block_weights[:, draw_columns] += block_weights[:, column_models[draw_columns]] + array_module.sqrt(variances[:, draw_columns]) * prior_noise[start:stop]
        weights[start:stop] = block_weights
        combined = tile.matmat(array_module.concatenate([shifted, block_weights], axis=1))
        image += combined[:, :column_count]
        scores += combined[score_rows, column_count:]
    count.note(column_count, 0.0, "fused-final")
    residual = right_hand_side - (duals + models.design_to_sample(image, column_models))
    return weights, scores, array_module.linalg.norm(residual, axis=0), design_products


@dataclass
class ResolvedSites:
    """Per model, the variants whose site precision is not positive, and their sites (Pi_L, h_L).

    They cannot enter S (Sylvester), so they are eliminated exactly: with the bulk operator
    S_S = I + Xt_S D_S Xt_S' (their D set to 0 in the bulk models) and Z = S_S^-1 [b, Xt_L],

        core = Pi_L + Xt_L' Z_L      (the Schur complement of A's bulk block: PD exactly when A is)
        mu_L = core^-1 (h_L + Xt_L' z_b),   mean dual z = z_b - Z_L mu_L,

    so mu_S = m_S + D_S Xt_S' z. A joint draw exact for PD A takes beta_L* = mu_L + core^-1/2 eps_L
    from the marginal of beta_L, then beta_S* from its conditional by Matheron on the bulk
    operator: its dual is z_e - Z_L (beta_L* - mu_L), with z_e = S_S^-1 (e2 - Xt_S D_S^1/2 e1).
    """

    indices: dict
    precision: dict
    shift: dict


def resolved_design(source: DualTileSource, models: DualModels, resolved: ResolvedSites) -> dict:
    """Xt_L (n x |L|) of every model with resolved sites, its columns in the order of the model's indices.

    The blocks yield the columns in variant order, so they are put back in the order the sites
    (precision, shift) are given in; the core pairs column k with site k.
    """
    array_module = source.array_module
    designs: dict[int, Any] = {}
    for model, indices in resolved.indices.items():
        indices = np.asarray(indices, dtype=np.int64)
        if indices.size == 0:
            continue
        if np.unique(indices).size != indices.size:
            raise ValueError("a model's resolved indices must be distinct.")
        order = np.argsort(indices, kind="stable")
        ascending = indices[order]
        columns = array_module.concatenate(
            [tile.columns(ascending[(ascending >= start) & (ascending < stop)] - start) for start, stop, tile in source.blocks()], axis=1
        )
        given_order = np.empty_like(order)
        given_order[order] = np.arange(order.size)
        designs[model] = models.design_to_sample(columns[:, array_module.asarray(given_order)], array_module.full(indices.size, model))
    return designs


def split_draw_duals(
    core_factors: dict, resolved_mean: dict, resolved_duals: dict, draw_models: np.ndarray, perturbation_duals: Any, resolved_noise: dict, array_module: Any
) -> tuple[Any, dict]:
    """Each draw's dual and its resolved block, from the bulk Matheron duals z_e.

    beta_L* - mu_L = core^-1/2 eps_L (L^-T eps_L with core = L L'), and the draw's dual relative to
    the mean is z_e - Z_L (beta_L* - mu_L). `resolved_noise[model]` is eps_L (|L| x the model's
    draws), in the order of that model's draw columns.
    """
    draw_duals = perturbation_duals.copy()
    resolved_draws: dict[int, Any] = {}
    for model, factor in core_factors.items():
        columns = np.flatnonzero(np.asarray(draw_models) == model)
        if columns.size == 0:
            continue
        offset = array_module.linalg.solve(factor.T, array_module.asarray(resolved_noise[model]))
        resolved_draws[model] = resolved_mean[model][:, None] + offset
        draw_duals[:, columns] -= resolved_duals[model] @ offset
    return draw_duals, resolved_draws


@dataclass
class ResolvedBlock:
    """One model's eliminated sites after a solve: Xt_L, Pi_L, Z_L = S_S^-1 Xt_L (computed), the exact
    residuals R_L = Xt_L - S_S Z_L, and the computed core Pi_L + Xt_L'Z_L with its Cholesky factor."""

    design: Any
    precision: Any
    duals: Any
    residual: Any
    core: Any
    factor: Any


def resolved_block(array_module: Any, design: Any, precision: Any, duals: Any, residual: Any) -> ResolvedBlock:
    """The core of a model's resolved sites; LinAlgError when it, hence the global precision, is not PD."""
    core = array_module.diag(precision) + design.T @ duals
    core = 0.5 * (core + core.T)
    factor = array_module.linalg.cholesky(core)
    if not bool(array_module.all(array_module.isfinite(factor))):
        raise np.linalg.LinAlgError("the resolved sites' core is not positive definite: the global precision is not.")
    return ResolvedBlock(design, precision, duals, residual, core, factor)


def split_columns(array_module: Any, block: ResolvedBlock, shift: Any, duals: Any, residual: Any) -> tuple[Any, Any, Any]:
    """Columns sharing one model's split: their resolved block, their mean duals and their A-norm certificate.

    For right-hand sides whose resolved rows are `shift` (|L| x c) and whose bulk duals `duals` (n x c)
    have exact residuals `residual`: mu_L = core^-1 (shift + Xt_L'z_b) and z = z_b - Z_L mu_L.

    The certificate bounds ||x_hat - x||_A through Schur's identity (DualCertificate), accounting for
    the inexact Z_L. With E = Z_L - Z_hat = S_S^-1 R_L (R_L the exact residuals of the Z_L columns),
    - the core is off by Delta = sym(Z_hat'R_L) + R_L'S_S^-1 R_L, so with core_hat = L L' its relative
      size is at most delta = ||L^-1 sym(Z_hat'R_L) L^-T|| + ||R_L||^2 / lambda_min(core_hat), and
      core^-1 <= core_hat^-1 / (1 - delta) once delta < 1. Block CG keeps each residual orthogonal to its
      Krylov space, where Z_hat lies, so the first term is at rounding level and delta is second order;
    - Z_L'r_S is off from Z_hat'r_S by R_L'S_S^-1 r_S, at most ||R_L|| ||r_S|| (S_S >= I).
    All norms are spectral. The certificate is infinite when delta is not below 1.
    """
    resolved_mean = _cholesky_solve(array_module, block.factor, shift + block.design.T @ duals)
    mean_duals = duals - block.duals @ resolved_mean
    bulk_residual = residual - block.residual @ resolved_mean
    stationarity = shift + block.design.T @ mean_duals - block.precision[:, None] * resolved_mean
    lowest = float(array_module.linalg.eigvalsh(block.core)[0])
    # ||L^-1||_2^2 = 1 / lambda_min(core_hat).
    inverse_factor_norm = float(np.sqrt(1.0 / lowest))
    residual_gram = block.residual.T @ block.residual
    residual_norm = float(np.sqrt(max(float(array_module.linalg.eigvalsh(0.5 * (residual_gram + residual_gram.T))[-1]), 0.0)))
    coupling = block.duals.T @ block.residual
    whitened = array_module.linalg.solve(block.factor, array_module.linalg.solve(block.factor, 0.5 * (coupling + coupling.T)).T)
    core_error = float(array_module.max(array_module.abs(array_module.linalg.eigvalsh(0.5 * (whitened + whitened.T))))) + residual_norm**2 / lowest
    bulk_norms = array_module.linalg.norm(bulk_residual, axis=0)
    if core_error >= 1.0:
        return resolved_mean, mean_duals, array_module.full(bulk_norms.shape, np.inf)
    projected = array_module.linalg.solve(block.factor, stationarity + block.duals.T @ bulk_residual)
    quadratic = (array_module.linalg.norm(projected, axis=0) + inverse_factor_norm * residual_norm * bulk_norms) ** 2 / (1.0 - core_error)
    return resolved_mean, mean_duals, array_module.sqrt(bulk_norms * bulk_norms + quadratic)


@dataclass(frozen=True)
class DualCertificate:
    """Per model, the certified bound on the mean's error ||mu_hat - mu||_A, and the bound asked for.

    With the resolved sites L eliminated (bulk dual residual r_S = r_b - R_L mu_L, from the exact
    residuals of the solve; L stationarity residual r_L; Z_L = S_S^-1 Xt_L),

        ||mu_hat - mu||_A^2 = r_S'(I - S_S^-1) r_S + (r_L + Z_L'r_S)' core^-1 (r_L + Z_L'r_S)

    exactly (Schur's complement of A's bulk block). The solve's Z_L and core are inexact; split_columns
    bounds what that adds, to second order in the Z_L residuals.
    """

    error_bound: Any
    residual_bound: Any
    resolved_counts: np.ndarray
    iterations: int
    restarts: int


class _WindowLayout:
    """Which rows of C = Xt'Z_L the marginal-variance maps read: block b's variants against the resolved
    sites whose block lies in W(b) = {b-1, b, b+1} (b alone without cross-Grams), as marginal_variances
    does. The blocks partition the variants; each must lie inside one tile of the source."""

    def __init__(self, grams: BlockGrams, source: DualTileSource) -> None:
        self.blocks = [np.asarray(members, dtype=np.int64) for members in grams.blocks]
        self.block_count = len(self.blocks)
        self.adjacent = bool(grams.next_cross)
        self.block_of_variant = np.empty(source.variant_count, dtype=np.int64)
        self.block_of_variant[np.concatenate(self.blocks)] = np.repeat(np.arange(self.block_count), [members.size for members in self.blocks])
        if np.concatenate(self.blocks).size != source.variant_count or np.unique(np.concatenate(self.blocks)).size != source.variant_count:
            raise ValueError("the window blocks must partition the variants.")
        starts = np.asarray([start for start, _stop in source.block_bounds], dtype=np.int64)
        stops = np.asarray([stop for _start, stop in source.block_bounds], dtype=np.int64)
        self.tile_blocks: dict[int, list[int]] = {}
        for block, members in enumerate(self.blocks):
            if members.size == 0:
                continue
            tile = int(np.searchsorted(starts, members.min(), side="right")) - 1
            if members.max() >= stops[tile]:
                raise ValueError("a window block must lie inside one tile of the source.")
            self.tile_blocks.setdefault(int(starts[tile]), []).append(block)
        self.empty_blocks = [block for block, members in enumerate(self.blocks) if members.size == 0]

    def positions(self, resolved: np.ndarray) -> list[np.ndarray]:
        """Per block, the indices into `resolved` of the sites in the block's window, ascending."""
        blocks_of_resolved = self.block_of_variant[resolved]
        order = np.argsort(blocks_of_resolved, kind="stable")
        ordered = blocks_of_resolved[order]
        reach = 1 if self.adjacent else 0
        out = []
        for block in range(self.block_count):
            low = np.searchsorted(ordered, block - reach, side="left")
            high = np.searchsorted(ordered, block + reach, side="right")
            out.append(np.sort(order[low:high]).astype(np.int64))
        return out

    def gather(self, array_module: Any, products: Any, start: int, column_offsets: dict, positions: dict, values: dict) -> None:
        """Copy each window's entries of one tile's products to the host, in one gather."""
        total = int(products.shape[1])
        pieces, shapes = [], []
        for model, offset in column_offsets.items():
            for block in self.tile_blocks.get(start, []):
                rows = self.blocks[block] - start
                columns = offset + positions[model][block]
                shapes.append((model, block, rows.size, columns.size))
                pieces.append((rows[:, None] * total + columns[None, :]).ravel())
        if pieces:
            flat = np.concatenate(pieces)
            gathered = _host(products.ravel()[array_module.asarray(flat)])
            cursor = 0
            for model, block, row_count, column_count in shapes:
                size = row_count * column_count
                values[model][block] = gathered[cursor : cursor + size].reshape(row_count, column_count)
                cursor += size
        for model in column_offsets:
            for block in self.empty_blocks:
                values[model][block] = np.zeros((0, positions[model][block].size))

    def empty(self) -> WindowCross:
        """The window layout of a model with no resolved sites."""
        return WindowCross(
            positions=tuple(np.zeros(0, dtype=np.int64) for _block in self.blocks),
            values=tuple(np.zeros((members.size, 0)) for members in self.blocks),
        )


class DualGaussian:
    """Stage 2's Gaussian q(beta) for every model in dual form: the mean, its certificate, the refresh
    quantities for the leave-block-out marginal variances, the covariates and posterior draws.

    A model is a quantitative trait on a training set: `training` (n, M) is 1 on its training rows,
    `targets` and `offsets` are (n, M) (a target off its model's rows is never read, so it may be
    missing: NaN, as in cohort.Cohort.targets), and the covariates are profiled out in its metric
    W = training / sigma^2. The sites (tau, nu) of `iterate` give D = 1/tau and m = nu / tau on the
    bulk; the resolved set L of each model, its non-positive sites and its spikes (resolved_spikes),
    is eliminated exactly, and the bulk probes (Rademacher on the training rows) give
    tr(S_S^-1)/n, tr(S_S^-2)/n and tr(Q^2)/n for marginal_variances.BulkSolve. `grams` gives the LD
    blocks and windows the marginal-variance maps use (only its blocks and whether it has cross-Grams
    are read), so C = Xt'Z_L is kept only where the maps read it (marginal_variances.WindowCross).
    """

    def __init__(
        self, *, source: DualTileSource, training: Any, targets: Any, offsets: Any, covariates: Any, grams: BlockGrams, probe_count: int, seed: int
    ) -> None:
        array_module = source.array_module
        self.source = source
        self.windows = _WindowLayout(grams, source)
        self.array_module = array_module
        self.training = array_module.asarray(training, dtype=array_module.float64)
        targets = array_module.asarray(targets, dtype=array_module.float64)
        if not bool(array_module.all(array_module.isfinite(targets) | (self.training == 0.0))):
            raise ValueError("every target on a model's training rows must be finite.")
        # Zero weight times a missing target is NaN, not zero: the rows off the mask take target 0.
        self.targets = array_module.where(self.training == 0.0, 0.0, targets)
        self.offsets = array_module.asarray(offsets, dtype=array_module.float64)
        self.covariates = array_module.asarray(covariates, dtype=array_module.float64)
        self.model_count = int(self.training.shape[1])
        self.training_counts = _host(self.training.sum(axis=0))
        self.count = PassCount()
        self.unit_squares = column_squares(source, DualModels(self.training, array_module.zeros((source.variant_count, self.model_count)), self.covariates, array_module), self.count)
        generator = np.random.default_rng(seed)
        signs = generator.choice(np.array([-1.0, 1.0]), size=(source.sample_count, self.model_count * probe_count))
        self.probe_models = np.repeat(np.arange(self.model_count), probe_count)
        self.probes = array_module.asarray(signs) * self.training[:, array_module.asarray(self.probe_models)]
        self.probe_count = int(probe_count)
        self.mean = array_module.zeros((source.variant_count, self.model_count))
        self.genetic_image = array_module.zeros((source.sample_count, self.model_count))
        self.alpha = array_module.zeros((int(self.covariates.shape[1]), self.model_count))
        self.noise_variance = np.ones(self.model_count)
        self._duals: Any = None
        self._resolved: dict = {}
        self.bulk_solves: list = []
        self._state: dict = {}

    def _models(self, noise_variance: np.ndarray, variances: Any) -> DualModels:
        weights = self.training / self.array_module.asarray(noise_variance)[None, :]
        return DualModels(weights, variances, self.covariates, self.array_module)

    def iterate(self, *, site_precision: Any, site_shift: Any, noise_variance: np.ndarray, error_bound: Any, probe_residual_ratio: float) -> DualCertificate:
        """The exact mean at the sites, certified to ||mu_hat - mu||_A <= error_bound per model.

        `probe_residual_ratio` is the accuracy the refresh quantities need (the probes and the Z_L
        columns solved to that share of their norm), from marginal_variances' certificate tolerance.
        """
        array_module = self.array_module
        source = self.source
        precision = array_module.asarray(site_precision, dtype=array_module.float64)
        shift = array_module.asarray(site_shift, dtype=array_module.float64)
        self.noise_variance = np.asarray(noise_variance, dtype=np.float64).copy()
        positive = precision > 0.0
        variances = array_module.where(positive, 1.0 / array_module.where(positive, precision, 1.0), 0.0)
        spikes = variances * self.unit_squares / array_module.asarray(self.noise_variance)[None, :]
        resolved: dict[int, np.ndarray] = {}
        for model in range(self.model_count):
            nonpositive = _host(array_module.flatnonzero(~positive[:, model]))
            candidate = array_module.where(positive[:, model], spikes[:, model], 0.0)
            resolved[model] = np.union1d(nonpositive, resolved_spikes(candidate, int(self.training_counts[model]), array_module)).astype(np.int64)
        bulk_variances = variances.copy()
        for model, indices in resolved.items():
            bulk_variances[array_module.asarray(indices), model] = 0.0
        bulk_mean = bulk_variances * shift
        models = self._models(self.noise_variance, bulk_variances)
        resolved_sites = ResolvedSites(
            {model: indices for model, indices in resolved.items() if indices.size},
            {model: precision[array_module.asarray(indices), model] for model, indices in resolved.items() if indices.size},
            {model: shift[array_module.asarray(indices), model] for model, indices in resolved.items() if indices.size},
        )
        designs = resolved_design(source, models, resolved_sites)
        prior_image = array_module.zeros((source.sample_count, self.model_count))
        for start, stop, tile in source.blocks():
            prior_image += tile.matmat(bulk_mean[start:stop])
        self.count.note(self.model_count, 0.0, "prior-image")
        right = mean_right_hand_side(models, self.targets - self.offsets, prior_image)
        order = sorted(designs)
        blocks = [right] + [designs[model] for model in order] + [self.probes]
        column_models = np.concatenate(
            [np.arange(self.model_count)] + [np.full(int(designs[model].shape[1]), model) for model in order] + [self.probe_models]
        )
        stacked = array_module.concatenate(blocks, axis=1)
        widths = [self.model_count] + [int(designs[model].shape[1]) for model in order] + [int(self.probes.shape[1])]
        offsets = np.concatenate([[0], np.cumsum(widths)])
        target = array_module.asarray(error_bound, dtype=array_module.float64)
        column_norms = array_module.linalg.norm(stacked, axis=0)
        bound = probe_residual_ratio * column_norms
        bound[: self.model_count] = target
        for position, model in enumerate(order):
            relative = min(probe_residual_ratio, float(target[model]) / max(float(column_norms[model]), np.finfo(np.float64).tiny))
            bound[offsets[position + 1] : offsets[position + 2]] = relative * column_norms[offsets[position + 1] : offsets[position + 2]]
        start = array_module.zeros_like(stacked)
        same_resolved = self._resolved.keys() == resolved.keys() and all(np.array_equal(self._resolved[model], resolved[model]) for model in resolved)
        if self._duals is not None and self._duals.shape == stacked.shape and same_resolved:
            start = self._duals
        # The split removes every spike from the bulk operator, which is what the relaxed operand error
        # needs (certified_block_cg): an empty deflation says so.
        spike_free = Deflation({}, {}, {}, resolved)
        iterations = 0
        restarts = 0
        while True:
            result = certified_block_cg(source, models, stacked, start, array_module.asarray(column_models), bound, self.count, deflation=spike_free, label="gaussian")
            iterations += result.iterations
            restarts += result.restarts
            residual = result.residual
            mean_duals = result.solution[:, : self.model_count].copy()
            certificate = array_module.linalg.norm(residual[:, : self.model_count], axis=0)
            blocks_by_model: dict[int, ResolvedBlock] = {}
            resolved_means: dict[int, Any] = {}
            for position, model in enumerate(order):
                columns = slice(int(offsets[position + 1]), int(offsets[position + 2]))
                block = resolved_block(array_module, designs[model], resolved_sites.precision[model], result.solution[:, columns], residual[:, columns])
                resolved_mean, model_duals, model_certificate = split_columns(
                    array_module, block, resolved_sites.shift[model][:, None], result.solution[:, model : model + 1], residual[:, model : model + 1]
                )
                mean_duals[:, model] = model_duals[:, 0]
                certificate[model] = model_certificate[0]
                blocks_by_model[model] = block
                resolved_means[model] = resolved_mean[:, 0]
            state: dict = {"designs": designs, "order": order, "offsets": offsets, "blocks": blocks_by_model, "resolved_mean": resolved_means}
            open_models = _host(certificate > target)
            if not open_models.any():
                break
            # Tighten the open models' mean and Z_L columns by the measured shortfall and continue.
            for model in np.flatnonzero(open_models):
                shortfall = float(target[model] / certificate[model]) if bool(array_module.isfinite(certificate[model])) else float(target[model] / max(float(column_norms[model]), np.finfo(np.float64).tiny))
                bound[model] *= shortfall
                if model in order:
                    position = order.index(model)
                    bound[offsets[position + 1] : offsets[position + 2]] *= shortfall
            start = result.solution
        self._duals = result.solution
        self._resolved = resolved
        self._state = state | {"models": models, "bulk_mean": bulk_mean, "bulk_variances": bulk_variances, "precision": precision}
        self._finish(result, mean_duals, bulk_mean, bulk_variances, precision, models, state, resolved)
        return DualCertificate(certificate, target, np.array([resolved[model].size for model in range(self.model_count)]), iterations, restarts)

    def _finish(self, result: SolveResult, mean_duals: Any, bulk_mean: Any, bulk_variances: Any, precision: Any, models: DualModels, state: dict, resolved: dict) -> None:
        """One read: the bulk mean m + D Xt'z, its genetic image X mu, and C = Xt' Z_L on the LD windows."""
        array_module = self.array_module
        source = self.source
        order = state["order"]
        resolved_blocks = [state["blocks"][model].duals for model in order]
        resolved_models = [np.full(int(block.shape[1]), model) for model, block in zip(order, resolved_blocks)]
        duals = array_module.concatenate([mean_duals] + resolved_blocks, axis=1)
        column_models = array_module.asarray(np.concatenate([np.arange(self.model_count)] + resolved_models))
        left = models.sample_to_design(duals, column_models)
        mean = bulk_mean.copy()
        widths = {model: int(block.shape[1]) for model, block in zip(order, resolved_blocks)}
        column_offsets = dict(zip(order, (self.model_count + np.concatenate([[0], np.cumsum([widths[model] for model in order])]))[:-1].astype(np.int64)))
        positions = {model: self.windows.positions(resolved[model]) for model in order}
        window_values: dict = {model: [None] * self.windows.block_count for model in order}
        for start, stop, tile in source.blocks():
            products = tile.rmatmat(left)
            mean[start:stop] += bulk_variances[start:stop] * products[:, : self.model_count]
            self.windows.gather(array_module, products, start, column_offsets, positions, window_values)
        for model in order:
            mean[array_module.asarray(resolved[model]), model] = state["resolved_mean"][model]
        image = array_module.zeros((source.sample_count, self.model_count))
        for start, stop, tile in source.blocks():
            image += tile.matmat(mean[start:stop])
        self.count.note(int(duals.shape[1]), 0.0, "finish")
        self.mean = mean
        self.genetic_image = image
        remainder = self.targets - self.offsets - image
        self.alpha = models.covariate_solve(self.covariates.T @ (models.weights * remainder), array_module.arange(self.model_count))
        self.linear_predictor = self.offsets + image + self.covariates @ self.alpha
        probe_solutions = result.solution[:, int(state["offsets"][-2]) :]
        self.bulk_solves = []
        for model in range(self.model_count):
            probe_columns = array_module.asarray(np.flatnonzero(self.probe_models == model))
            probes = self.probes[:, probe_columns]
            solved = probe_solutions[:, probe_columns]
            count = float(self.training_counts[model])
            kernel = solved
            core = array_module.zeros((0, 0))
            model_cross = self.windows.empty()
            if model in order:
                block = state["blocks"][model]
                design = block.design
                kernel = solved - block.duals @ _cholesky_solve(array_module, block.factor, design.T @ solved)
                core = block.core
                model_cross = WindowCross(positions=tuple(positions[model]), values=tuple(window_values[model]))
            self.bulk_solves.append(BulkSolve(
                site_precision=_host(precision[:, model]),
                resolved=resolved[model],
                resolved_core=_host(core),
                resolved_cross=model_cross,
                bulk_trace=float(array_module.sum(probes * solved)) / (self.probe_count * count),
                bulk_square_trace=float(array_module.sum(solved * solved)) / (self.probe_count * count),
                kernel_square_trace=float(array_module.sum(kernel * kernel)) / (self.probe_count * count),
                sample_count=int(count),
            ))

    def _bulk_image(self, right: Any, model: int) -> tuple[Any, Any, Any, Any, Any]:
        """(v, v with its resolved rows zeroed, the resolved rows, the column models, u = Xt D_S v), one read."""
        array_module = self.array_module
        state = self._state
        bulk_variances = state["bulk_variances"][:, model]
        values = array_module.asarray(right, dtype=array_module.float64)
        columns = int(values.shape[1])
        resolved = array_module.asarray(self._resolved[model])
        bulk_values = values.copy()
        bulk_values[resolved] = 0.0
        column_models = array_module.full(columns, model)
        image = array_module.zeros((self.source.sample_count, columns))
        for start, stop, tile in self.source.blocks():
            image += tile.matmat(bulk_variances[start:stop, None] * bulk_values[start:stop])
        self.count.note(columns, 0.0, "bulk-image")
        return values, bulk_values, resolved, column_models, state["models"].design_to_sample(image, column_models)

    def information_solve(self, probes: Any, model: int, residual_tolerance: Any) -> tuple[Any, Any, Any]:
        """The products the cavity certificate needs for variant-side probes v (p x k), at the last iterate's sites.

        With u = Xt D_S v, t = Z_L'u and w = K_S^-1 u - Z_L core^-1 (t - v_L) (K_S = S_S, the bulk
        operator), returns (Xt'w (p x k), t (|L| x k), the exact relative residuals ||u - K_S y|| / ||u||
        of the bulk solve (k,)). The solve runs to `residual_tolerance`, a relative bound on that
        residual (marginal_variances' information_solve_tolerance), applied to each column's own u.
        w is minus the dual posterior_solve forms for the shift v, before D_S v_S is added. Two reads
        plus the CG passes.
        """
        array_module = self.array_module
        state = self._state
        models = state["models"]
        values, _bulk_values, resolved, column_models, image = self._bulk_image(probes, model)
        columns = int(values.shape[1])
        image_norms = array_module.linalg.norm(image, axis=0)
        bound = array_module.broadcast_to(array_module.asarray(residual_tolerance, dtype=array_module.float64), (columns,)) * image_norms
        spike_free = Deflation({}, {}, {}, self._resolved)
        result = certified_block_cg(self.source, models, image, array_module.zeros_like(image), column_models, bound, self.count, deflation=spike_free, label="information")
        block = state["blocks"].get(model)
        duals = result.solution
        coupling = array_module.zeros((0, columns))
        if block is not None:
            coupling = block.duals.T @ image
            duals = duals - block.duals @ _cholesky_solve(array_module, block.factor, coupling - values[resolved])
        left = models.sample_to_design(duals, column_models)
        back_products = array_module.empty((self.source.variant_count, columns))
        for start, stop, tile in self.source.blocks():
            back_products[start:stop] = tile.rmatmat(left)
        self.count.note(columns, 0.0, "information-products")
        return back_products, coupling, result.residual_norm / array_module.maximum(image_norms, np.finfo(np.float64).tiny)

    def posterior_solve(self, right: Any, model: int, error_bound: Any) -> Any:
        """A_m^-1 right (p x r) at the last iterate's sites, each column certified to ||x_hat - x||_A <= error_bound.

        It is the split mean with no data and shift `right`: the bulk dual solves
        S_S z_b = -Xt_S D_S v_S, the resolved rows are x_L = core^-1 (v_L + Xt_L'z_b), and
        x_S = D_S v_S + D_S Xt_S'(z_b - Z_L x_L), with split_columns' certificate. The last iterate's
        Z_L is reused; when the certificate needs it, the columns and Z_L are tightened by the
        measured shortfall. Two reads plus the CG passes.
        """
        array_module = self.array_module
        source = self.source
        state = self._state
        models = state["models"]
        bulk_variances = state["bulk_variances"][:, model]
        values, bulk_values, resolved, column_models, image = self._bulk_image(right, model)
        columns = int(values.shape[1])
        rhs = -image
        target = array_module.broadcast_to(array_module.asarray(error_bound, dtype=array_module.float64), (columns,)).copy()
        bound = target.copy()
        spike_free = Deflation({}, {}, {}, self._resolved)
        block = state["blocks"].get(model)
        start = array_module.zeros_like(rhs)
        while True:
            result = certified_block_cg(source, models, rhs, start, column_models, bound, self.count, deflation=spike_free, label="posterior")
            if block is None:
                duals, certificate, resolved_values = result.solution, result.residual_norm, None
            else:
                resolved_values, duals, certificate = split_columns(array_module, block, values[resolved], result.solution, result.residual)
            open_mask = _host(certificate > target)
            if not open_mask.any():
                break
            finite = array_module.isfinite(certificate)
            fallback = target / array_module.maximum(array_module.linalg.norm(rhs, axis=0), np.finfo(np.float64).tiny)
            shortfall = array_module.where(finite, target / array_module.where(finite, certificate, 1.0), fallback)
            open_columns = array_module.asarray(np.flatnonzero(open_mask))
            bound[open_columns] *= shortfall[open_columns]
            if block is not None:
                tightening = float(array_module.min(shortfall[open_columns]))
                resolved_columns = array_module.full(int(block.design.shape[1]), model)
                resolved_bound = tightening * array_module.linalg.norm(block.residual, axis=0)
                refined = certified_block_cg(source, models, block.design, block.duals, resolved_columns, resolved_bound, self.count, deflation=spike_free, label="posterior-resolved")
                block = resolved_block(array_module, block.design, block.precision, refined.solution, refined.residual)
                state["blocks"][model] = block
            start = result.solution
        left = models.sample_to_design(duals, column_models)
        solution = bulk_variances[:, None] * bulk_values
        for start, stop, tile in source.blocks():
            solution[start:stop] += bulk_variances[start:stop, None] * tile.rmatmat(left)
        self.count.note(columns, 0.0, "posterior-solution")
        if resolved_values is not None:
            solution[resolved] = resolved_values
        return solution

    def residual_sum_of_squares(self) -> np.ndarray:
        """(M,): each model's training sum of (y - linear predictor)^2 at the current mean."""
        residual = self.targets - self.linear_predictor
        return _host(self.array_module.sum(self.training * residual * residual, axis=0))

    def draws_from_noise(self, *, prior_noise: Any, sample_noise: Any, resolved_noise: dict, draw_models: np.ndarray, error_bound: Any) -> tuple[Any, Any]:
        """Posterior draws for given noise: the bulk by Matheron on S_S, L from its marginal (split_draw_duals).

        prior_noise (p, K) and sample_noise (n, K) are e1 and e2 of each draw column, resolved_noise[m]
        is eps_L (|L_m| x the model's draws); each draw's dual is certified to error_bound (K,).
        Returns (draws (p, K), exact bulk dual residual norms (K,)).
        """
        array_module = self.array_module
        source = self.source
        state = self._state
        models = state["models"]
        device_models = array_module.asarray(draw_models)
        rhs = draw_right_hand_side(source, models, device_models, prior_noise, sample_noise, self.count)
        result = certified_block_cg(source, models, rhs, array_module.zeros_like(rhs), device_models, array_module.asarray(error_bound), self.count, label="draws")
        draw_duals, resolved_draws = split_draw_duals(
            {model: block.factor for model, block in state["blocks"].items()}, state["resolved_mean"],
            {model: block.duals for model, block in state["blocks"].items()}, draw_models, result.solution, resolved_noise, array_module,
        )
        left = models.sample_to_design(draw_duals, device_models)
        bulk_variances = state["bulk_variances"][:, device_models]
        draws = self.mean[:, device_models] + array_module.sqrt(bulk_variances) * prior_noise
        for start, stop, tile in source.blocks():
            draws[start:stop] += bulk_variances[start:stop] * tile.rmatmat(left)
        self.count.note(int(draw_duals.shape[1]), 0.0, "draws")
        for model, values in resolved_draws.items():
            columns = np.flatnonzero(draw_models == model)
            draws[array_module.asarray(self._resolved[model])[:, None], array_module.asarray(columns)[None, :]] = values
        return draws, result.residual_norm

    def draws(self, *, draw_count: int, error_bound: Any, seed: int) -> Any:
        """(p, M, draw_count) exact posterior draws of every model, each certified to its model's error_bound."""
        array_module = self.array_module
        generator = np.random.default_rng(seed)
        draw_models = np.repeat(np.arange(self.model_count), draw_count)
        prior_noise = array_module.asarray(generator.standard_normal((self.source.variant_count, draw_models.size)))
        sample_noise = array_module.asarray(generator.standard_normal((self.source.sample_count, draw_models.size)))
        resolved_noise = {
            model: array_module.asarray(generator.standard_normal((self._resolved[model].size, draw_count)))
            for model in range(self.model_count) if self._resolved[model].size
        }
        bound = array_module.asarray(error_bound, dtype=array_module.float64)[array_module.asarray(draw_models)]
        draws, _norms = self.draws_from_noise(prior_noise=prior_noise, sample_noise=sample_noise, resolved_noise=resolved_noise, draw_models=draw_models, error_bound=bound)
        return draws.reshape(self.source.variant_count, self.model_count, draw_count)
