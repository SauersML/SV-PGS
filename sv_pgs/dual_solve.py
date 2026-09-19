"""Stage 2's Gaussian E-step in dual form: every model, fold, probe and draw rides one pass.

A model is a trait on a sample mask. It has likelihood weights w (n,), zero off the mask (1/sigma^2
for a quantitative trait; the logistic curvature or EP likelihood-site precision, positive because
the logistic likelihood is log-concave, for a binary one), Gaussian site variances D = 1/Pi (p,)
>= 0, and the covariates C profiled out in its own weighted metric:

    H  = W^1/2 C (C'WC)^-1 C' W^1/2,   Xt = (I - H) W^1/2 X,
    A  = Xt'Xt + diag(1/D)            (the posterior precision)
    S  = I + Xt D Xt'                 (n x n, every eigenvalue >= 1)

The mean is mu = m + D Xt' z with S z = b, b = (I - H) W^1/2 (y~ - X m), by the push-through
identity A^-1 Xt' = D Xt' S^-1. For any iterate with residual r = b - S z_hat,

    ||mu_hat - mu||_A^2 = r'(I - S^-1) r <= ||r||^2

(e = -D Xt' S^-1 r and Xt (D A D) Xt' = G S with G = S - I), so a column's exact residual norm is
its certificate against ep_eb.md's target ||e||_A^2 <= p_eff / K. Rows outside the training metric
(held-out folds, target people) satisfy ||X_out e||^2 <= lambda_max(X_out D X_out') ||r||^2 / 4.

One pass reads each tile once and serves every column of every model:

    T = W^1/2 (I - H) V;  per tile: U = X_b' T;  U *= D_b[:, model];  Y += X_b U;  S V = V + (I - H) W^1/2 Y.

Nothing is p x columns except per-tile products. A binary model is a per-column weight, and a
fold is a mask, so neither needs a Gram or a factor. Site precisions that are not positive cannot
enter S (with q of them and A > 0, S has exactly q negative eigenvalues, by Sylvester's law of
inertia); they are eliminated exactly by the caller's split.

Tiles multiply with a normwise relative operand error per column (the int8 digit split of
code_products): ||delta_k|| <= relative_error ||operand_k||, exact zeros kept, 0 meaning exact.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Iterator, Protocol

import numpy as np

DIGIT_BITS = 7
"""Bits per balanced base-128 operand digit of the int8 split."""


class DualTile(Protocol):
    """One LD block's standardized genotype columns X_b (n x p_b)."""

    def rmatmat(self, left: Any, relative_error: float) -> Any:
        """X_b' left for left (n, K): (p_b, K)."""
        ...

    def matmat(self, right: Any, relative_error: float) -> Any:
        """X_b right for right (p_b, K): (n, K)."""
        ...

    def column_squares(self, weights: Any) -> Any:
        """(X_b * X_b)' weights for weights (n, M): (p_b, M)."""
        ...

    def columns(self, local: np.ndarray) -> Any:
        """X_b[:, local] as dense float64: (n, len(local))."""
        ...


class DualTileSource(Protocol):
    array_module: Any
    sample_count: int
    variant_count: int

    def blocks(self) -> Iterator[tuple[int, int, DualTile]]:
        """(start, stop, tile) for every block, in variant order, covering 0..p-1."""
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
    """A block held as dense standardized columns; products round the operand as the int8 split does."""

    def __init__(self, values: Any, array_module: Any) -> None:
        self.values = values
        self.array_module = array_module

    def rmatmat(self, left: Any, relative_error: float) -> Any:
        return self.values.T @ rounded_operand(left, relative_error, self.array_module)

    def matmat(self, right: Any, relative_error: float) -> Any:
        return self.values @ rounded_operand(right, relative_error, self.array_module)

    def column_squares(self, weights: Any) -> Any:
        return (self.values * self.values).T @ weights

    def columns(self, local: np.ndarray) -> Any:
        return self.values[:, local]


class DenseDualSource:
    """Dense standardized genotypes (n x p) cut into contiguous blocks."""

    def __init__(self, genotypes: Any, block_bounds: list[tuple[int, int]], array_module: Any = np) -> None:
        self.array_module = array_module
        self.genotypes = array_module.asarray(genotypes, dtype=array_module.float64)
        self.sample_count, self.variant_count = (int(extent) for extent in self.genotypes.shape)
        self.block_bounds = list(block_bounds)
        if not self.block_bounds or self.block_bounds[0][0] != 0 or self.block_bounds[-1][1] != self.variant_count or any(
            previous[1] != following[0] for previous, following in zip(self.block_bounds, self.block_bounds[1:])
        ):
            raise ValueError("block_bounds must cover every variant once, in order.")

    def blocks(self) -> Iterator[tuple[int, int, DualTile]]:
        for start, stop in self.block_bounds:
            yield start, stop, DenseDualTile(self.genotypes[:, start:stop], self.array_module)


def _host(values: Any) -> np.ndarray:
    return np.asarray(values.get() if hasattr(values, "get") else values)


class DualModels:
    """Weights (n, M), site variances (p, M) and shared covariates (n, k) of every model."""

    def __init__(self, weights: Any, variances: Any, covariates: Any, array_module: Any = np) -> None:
        self.array_module = array_module
        self.weights = array_module.asarray(weights, dtype=array_module.float64)
        self.variances = array_module.asarray(variances, dtype=array_module.float64)
        self.covariates = array_module.asarray(covariates, dtype=array_module.float64)
        if bool(array_module.any(self.variances < 0.0)):
            raise ValueError("site variances must be non-negative; negative sites need the exact split.")
        self.root_weights = array_module.sqrt(self.weights)
        normal = array_module.einsum("na,nm,nb->mab", self.covariates, self.weights, self.covariates)
        self.covariate_factor = array_module.linalg.cholesky(normal)

    @property
    def model_count(self) -> int:
        return int(self.weights.shape[1])

    def complement(self, values: Any, column_models: Any) -> Any:
        """(I - H_m) v per column, with H_m the weighted covariate projector of the column's model."""
        array_module = self.array_module
        root = self.root_weights[:, column_models]
        projected = (self.covariates.T @ (root * values)).T[:, :, None]
        factor = self.covariate_factor[column_models]
        solved = array_module.linalg.solve(array_module.swapaxes(factor, 1, 2), array_module.linalg.solve(factor, projected))[:, :, 0].T
        return values - root * (self.covariates @ solved)

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


def apply_operator(source: DualTileSource, models: DualModels, values: Any, column_models: Any, relative_error: float, count: PassCount, label: str) -> Any:
    """S V for every column, one read."""
    array_module = source.array_module
    left = models.sample_to_design(values, column_models)
    image = array_module.zeros_like(values)
    for start, stop, tile in source.blocks():
        products = tile.rmatmat(left, relative_error)
        image += tile.matmat(models.variances[start:stop][:, column_models] * products, relative_error)
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


@dataclass
class Deflation:
    """Per model, a basis W (n x k) of its spike directions, S W, and the factor of W'SW.

    Deflated CG (Saad, Yeung, Erhel & Guyomarc'h 2000) keeps every residual orthogonal to W and
    every direction S-orthogonal to it: the spikes W spans are solved exactly in the k x k system
    W'SW, and CG sees only the rest of the spectrum.
    """

    bases: dict
    images: dict
    factors: dict

    def project_start(self, array_module: Any, model: int, solution: Any, residual: Any) -> tuple[Any, Any]:
        coefficients = _cholesky_solve(array_module, self.factors[model], self.bases[model].T @ residual)
        return solution + self.bases[model] @ coefficients, residual - self.images[model] @ coefficients

    def project_direction(self, array_module: Any, model: int, values: Any) -> Any:
        return values - self.bases[model] @ _cholesky_solve(array_module, self.factors[model], self.images[model].T @ values)


def spike_deflation(source: DualTileSource, models: DualModels, count: PassCount, edges: dict | None = None) -> tuple[Deflation, dict]:
    """Each model's spikes and their deflation basis: one pass for ||xt_k||^2, one for S W.

    Variant k adds D_k ||xt_k||^2 to G = Xt D Xt' along Xt e_k. Without `edges`, k is resolved when
    that spike exceeds 1 plus the bulk's mean eigenvalue tr(G_bulk)/n (iterated to its fixed point,
    since removing spikes lowers the level). With `edges` (model -> the largest Ritz value of the
    model's deflated operator in an earlier solve), only spikes above the bulk's measured edge are
    resolved: a spike below it does not raise the condition number.
    ||xt_k||^2 = x_k'W x_k - (x_k'WC)(C'WC)^-1(C'Wx_k) for every model comes from the first pass.
    """
    array_module = source.array_module
    squares = array_module.zeros_like(models.variances)
    covariate_count = int(models.covariates.shape[1])
    weighted_covariates = array_module.concatenate(
        [models.weights[:, model : model + 1] * models.covariates for model in range(models.model_count)], axis=1
    )
    for start, stop, tile in source.blocks():
        cross = tile.rmatmat(weighted_covariates, 0.0).reshape(stop - start, models.model_count, covariate_count)
        factor = array_module.broadcast_to(models.covariate_factor[None], (stop - start,) + tuple(models.covariate_factor.shape))
        solved = array_module.linalg.solve(array_module.swapaxes(factor, 2, 3), array_module.linalg.solve(factor, cross[..., None]))[..., 0]
        squares[start:stop] = tile.column_squares(models.weights) - array_module.sum(cross * solved, axis=2)
    count.note(models.model_count * (1 + covariate_count), 0.0, "column-squares")
    spikes = models.variances * squares
    resolved: dict[int, np.ndarray] = {}
    for model in range(models.model_count):
        if edges is not None and model in edges:
            resolved[model] = _host(array_module.flatnonzero(1.0 + spikes[:, model] > edges[model]))
            continue
        chosen = array_module.zeros(spikes.shape[0], dtype=bool)
        while True:
            level = float(array_module.sum(array_module.where(chosen, 0.0, spikes[:, model]))) / source.sample_count
            updated = spikes[:, model] > 1.0 + level
            if bool(array_module.all(updated == chosen)):
                break
            chosen = updated
        resolved[model] = _host(array_module.flatnonzero(chosen))
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
            gram = bases[model].T @ images[model]
            factors[model] = array_module.linalg.cholesky(0.5 * (gram + gram.T))
    return Deflation(bases, images, factors), {model: int(indices.size) for model, indices in resolved.items()}


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
    Iterations run with a relaxed operand error: iteration k's product error F_k moves the true
    residual by at most ||F_k|| ||r_k|| (S >= I, orthonormal directions), and ||F_k|| <=
    lambda_max(S) (eps_left + eps_right); keeping the drift left over a residual contracting by rho
    below half the bound gives each operand eps = bound (1 - rho) / (4 lambda_max ||r_k||). The
    recursive residual takes the other half. lambda_max is the largest Ritz value seen; with none
    known (operator_scale 1) the first iteration runs exact to find it. Every restart begins with
    an exact residual (none when the start is zero, since then r = b), so the returned residual is
    exact. `deflation` solves each model's spikes exactly and CG the rest.

    There is no iteration cap. A bound below what float64 can attain for a column,
    (n + p) eps lambda_max ||z||, the rounding of one exact product S z, is refused up front, and a
    restart whose exact residual did not fall below the previous one fails loudly instead of looping.
    """
    array_module = source.array_module
    host_models = _host(column_models)
    solution = start.copy()
    bound = array_module.asarray(residual_bound, dtype=array_module.float64)
    relative_errors: list = []
    model_scales: dict[int, float] = {}
    scale_known = operator_scale > 1.0
    iterations = 0
    restarts = 0
    zero_start = not bool(array_module.any(solution != 0.0))
    while True:
        if zero_start and restarts == 0:
            residual = right_hand_side.copy()
        else:
            residual = right_hand_side - apply_operator(source, models, solution, column_models, 0.0, count, f"{label}:exact")
        norms = array_module.linalg.norm(residual, axis=0)
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
        contraction = 0.0
        previous_worst = None
        while directions:
            order = sorted(directions)
            block_models = array_module.asarray(np.concatenate([np.full(directions[model][1].shape[1], model) for model in order]))
            stacked = array_module.concatenate([directions[model][1] for model in order], axis=1)
            residual_norms = array_module.linalg.norm(residual[:, open_columns], axis=0)
            relative_error = 0.0
            if scale_known:
                relative_error = float(array_module.min(bound[open_columns] * (1.0 - contraction) / (4.0 * operator_scale * residual_norms)))
            image = apply_operator(source, models, stacked, block_models, relative_error, count, f"{label}:{iterations}")
            iterations += 1
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
                operator_scale = max(operator_scale, ritz)
                model_scales[model] = max(model_scales.get(model, 1.0), ritz)
                scale_known = True
                live = columns[_host(array_module.linalg.norm(residual[:, columns], axis=0) > 0.5 * bound[columns])]
                if live.size == 0:
                    continue
                step = array_module.linalg.solve(curvature, block.T @ residual[:, live])
                solution[:, live] += block @ step
                residual[:, live] -= applied @ step
                still = live[_host(array_module.linalg.norm(residual[:, live], axis=0) > 0.5 * bound[live])]
                if still.size == 0:
                    continue
                candidate = residual[:, still]
                if deflation is not None and model in deflation.bases:
                    candidate = deflation.project_direction(array_module, model, candidate)
                conjugation = -array_module.linalg.solve(curvature, applied.T @ candidate)
                orthonormal = _orthonormal_block(array_module, candidate + block @ conjugation)
                if orthonormal.shape[1]:
                    next_directions[model] = (still, orthonormal)
            worst = float(array_module.max(array_module.linalg.norm(residual[:, open_columns], axis=0) / bound[open_columns]))
            if previous_worst is not None and previous_worst > 0.0:
                contraction = min(max(worst / previous_worst, contraction), 1.0 - np.finfo(np.float64).eps)
            previous_worst = worst
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
        products = tile.rmatmat(left, 0.0)
        variances, means = block_update(start, stop, products)
        if bool(array_module.any(variances < 0.0)):
            raise ValueError("a block update returned a negative site variance; such sites need the exact split.")
        models.variances[start:stop] = variances
        combined = tile.matmat(array_module.concatenate([variances[:, column_models] * products, means], axis=1), 0.0)
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
        mean[start:stop] += models.variances[start:stop] * tile.rmatmat(left, 0.0)
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
        image += tile.matmat(array_module.sqrt(models.variances[start:stop][:, draw_models]) * prior_noise[start:stop], 0.0)
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
        products = tile.rmatmat(left, 0.0)
        if design_products is not None:
            design_products[start:stop] = products[:, design_columns]
        variances = models.variances[start:stop][:, column_models]
        shifted = variances * products
        block_weights = shifted.copy()
        block_weights[:, model_columns] += prior_mean[start:stop]
        block_weights[:, draw_columns] += block_weights[:, column_models[draw_columns]] + array_module.sqrt(variances[:, draw_columns]) * prior_noise[start:stop]
        weights[start:stop] = block_weights
        combined = tile.matmat(array_module.concatenate([shifted, block_weights], axis=1), 0.0)
        image += combined[:, :column_count]
        scores += combined[score_rows, column_count:]
    count.note(column_count, 0.0, "fused-final")
    residual = right_hand_side - (duals + models.design_to_sample(image, column_models))
    return weights, scores, array_module.linalg.norm(residual, axis=0), design_products
