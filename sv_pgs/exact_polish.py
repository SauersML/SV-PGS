"""Stage 2: the exact full-data Gaussian E-step of the one joint model.

Stage 1 fits the model in LD space with the LD between blocks taken to be zero;
that is a warm start only (SPEC). Stage 2 iterates the same expectation
propagation fixed point with the exact full-covariance Gaussian of all variants
on the individual-level data. This module holds that Gaussian: its mean, its
marginal variances and exact draws, for every model (traits x folds) at once.

Model. With the covariates C profiled out (Frisch-Waugh-Lovell, flat prior on
their effects alpha) and Gaussian sites exp(-tau_j beta_j^2 / 2 + nu_j beta_j)
standing in for the prior of every variant, q(beta) is Gaussian with precision

    A = X' P_W X + diag(tau),   P_W = W - W C (C' W C)^-1 C' W,

where W is the likelihood curvature on the training samples: 1 / sigma_e^2 for
a quantitative trait, and the Laplace curvature mu (1 - mu) of the logistic
likelihood for a binary one. An iteration takes the Newton step A delta = g,
g the full-data gradient of the log-likelihood plus the log sites; for a
quantitative trait one step gives the exact mean, and a binary step backtracks
on the penalized log-likelihood (strictly concave, so Newton converges).

Solve. Conjugate gradients is preconditioned by the block-Jacobi inverse M^-1:
each LD block's restricted precision X_b' P_W X_b + diag(tau_b), at the model's
exact curvature or at its mean curvature on the mask (one Gram per mask serves
every model on it). Every model, probe and draw is a column of the same reads,
and every model's columns (its mean step and probes, or its draws) share one
block-Krylov space, so the probes' directions also serve the mean. One read
serves one block-CG iteration: with X P known, a read forms
A P = X' P_W (X P) + diag(tau) P and M^-1 A P block by block and accumulates
X M^-1 A P, so the next block's image follows by linearity. The
iteration's first read also factors the blocks, evaluates the start residuals
and starts CG, so an iteration costs 1 + (CG iterations) reads. The answer
depends only on the exact operator; the blocks set the rate.

Variances. diag(A^-1) is estimated by the control-variate Hutchinson identity

    diag(A^-1) = diag(B) + E_z[z * (A^-1 z - B z)],  z Rademacher,

with B the block-Jacobi inverse, and projected onto its exact bounds
1 / A_jj <= (A^-1)_jj <= 1 / tau_j (A >= diag(tau), and Cauchy-Schwarz). The
probe systems A x = z are extra columns of the same reads, warm-started across
iterations. With one block and the exact curvature the correction vanishes and
the variances are exact.

Certificate. The first read of every iteration evaluates, at the state the
iteration starts from, the exact gradient g, the covariate gradient and the
residual z - A x of every probe system, and diag(A) for the variance bounds.
The iteration then solves the corrections from zero.

Draws. beta + A^-1 (X' s + sqrt(tau) e) with s = W^1/2 e_n minus its weighted
projection onto the covariates, e and e_n standard normal, is an exact draw of
q(beta): its covariance is A^-1 (X' P_W X + diag(tau)) A^-1 = A^-1.

Sample-side arrays (n x columns) live on the source's compute device;
variant-side arrays live on the host and move per block.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Iterator, Protocol, Sequence

import numpy as np
from numpy.typing import NDArray

from sv_pgs.config import TraitType

_MAXIMUM_CONJUGATE_GRADIENT_ITERATIONS = 400
# SVQB drops a model's direction whose squared norm in its block falls below this
# fraction of the largest (a converged or dependent column), so block CG cannot break down.
_DIRECTION_DROP = 1e-20
# Posterior draws solved per model at a time (one block-CG block).
_DRAW_BLOCK_COLUMNS = 64
# Backtracking of a binary model's Newton step on its penalized log-likelihood.
_MINIMUM_NEWTON_STEP = 2.0**-30
# Newton iterations and relative step tolerance of the covariate-only fit at the start.
_COVARIATE_NEWTON_ITERATIONS = 64
_COVARIATE_NEWTON_TOLERANCE = 1e-12
# The covariate normal matrix gets this ridge relative to its mean diagonal,
# which keeps the explicit (C' W C)^-1 accurate when a covariate is nearly
# constant on a training mask.
_COVARIATE_RIDGE = 1e-10


class GenotypeBlockTile(Protocol):
    """One LD block's standardized genotype columns (samples x variants)."""

    def matmat(self, right: Any) -> Any:
        """X_b right: (p_b, c) -> (n, c)."""
        ...

    def rmatmat(self, left: Any) -> Any:
        """X_b' left: (n, c) -> (p_b, c)."""
        ...

    def weighted_gram(self, weights: Any) -> Any:
        """X_b' diag(weights) X_b for an (n,) weight vector."""
        ...

    def weighted_cross(self, weights: Any, covariates: Any) -> Any:
        """X_b' diag(weights) C for an (n,) weight vector."""
        ...

    def weighted_column_squares(self, weights: Any) -> Any:
        """(X_b * X_b)' weights for an (n, c) weight array: (p_b, c)."""
        ...


class GenotypeBlockSource(Protocol):
    """The genotypes as LD blocks, in the order of the variant axis."""

    @property
    def sample_count(self) -> int: ...

    @property
    def array_module(self) -> Any:
        """NumPy, or CuPy when the tiles live on a CUDA device."""
        ...

    @property
    def block_variant_indices(self) -> Sequence[NDArray]:
        """Variant indices of every block; together they are 0..p-1 in order."""
        ...

    def iter_tiles(self) -> Iterator[tuple[int, GenotypeBlockTile]]: ...


class DenseGenotypeTile:
    """A block held as a dense (n, p_b) array."""

    def __init__(self, values: Any) -> None:
        self.values = values

    def matmat(self, right: Any) -> Any:
        return self.values @ right

    def rmatmat(self, left: Any) -> Any:
        return self.values.T @ left

    def weighted_gram(self, weights: Any) -> Any:
        return self.values.T @ (weights[:, None] * self.values)

    def weighted_cross(self, weights: Any, covariates: Any) -> Any:
        return self.values.T @ (weights[:, None] * covariates)

    def weighted_column_squares(self, weights: Any) -> Any:
        return (self.values * self.values).T @ weights


class DenseGenotypeBlockSource:
    """Dense standardized genotypes cut into contiguous LD blocks."""

    def __init__(self, genotypes: Any, block_variant_indices: Sequence[NDArray], array_module: Any = np) -> None:
        self._genotypes = array_module.asarray(genotypes, dtype=array_module.float64)
        self._block_variant_indices = [np.asarray(indices, dtype=np.int64) for indices in block_variant_indices]
        self._array_module = array_module
        if not np.array_equal(np.concatenate(self._block_variant_indices), np.arange(self._genotypes.shape[1])):
            raise ValueError("block_variant_indices must cover every variant once, in order.")

    @property
    def sample_count(self) -> int:
        return int(self._genotypes.shape[0])

    @property
    def array_module(self) -> Any:
        return self._array_module

    @property
    def block_variant_indices(self) -> Sequence[NDArray]:
        return self._block_variant_indices

    def iter_tiles(self) -> Iterator[tuple[int, GenotypeBlockTile]]:
        for block_index, indices in enumerate(self._block_variant_indices):
            yield block_index, DenseGenotypeTile(self._genotypes[:, int(indices[0]) : int(indices[-1]) + 1])


@dataclass(frozen=True)
class GaussianModel:
    """The sample side of one model: a trait on one training sample mask."""

    trait_type: TraitType
    targets: NDArray
    sample_mask_index: int
    predictor_offset: NDArray


@dataclass(frozen=True)
class StartCertificate:
    """Exactness of the E-step at the state an iteration started from, per model.

    ``gradient_relative_norm`` is ||g|| / (||X' u~|| + ||tau beta - nu||), u~ the
    projected log-likelihood score; ``covariate_gradient_relative_norm`` is
    ||C' u|| / ||C'|u| ||; ``probe_residual`` is the largest ||z - A x|| / ||z||.
    """

    gradient_relative_norm: NDArray
    covariate_gradient_relative_norm: NDArray
    probe_residual: NDArray


@dataclass(frozen=True)
class MarginalVariances:
    """diag(A^-1) per model, its Hutchinson standard error and the share clamped to its bounds.

    ``relative_standard_error`` is the standard error of sum_j Sigma_jj tau_j
    relative to its value.
    """

    variance: NDArray
    relative_standard_error: NDArray
    clamped_fraction: NDArray


class _Device:
    def __init__(self, array_module: Any) -> None:
        self.array_module = array_module

    def to_host(self, values: Any) -> NDArray:
        if self.array_module is np:
            return np.asarray(values)
        return self.array_module.asnumpy(values)


class _SampleSystem:
    """Targets, masks, covariates and curvature of every model; the weighted covariate projection.

    Columns of sample-side arrays are models; ``column_models`` maps the columns
    of a right-hand side (means, probes, draws) to their model.
    """

    def __init__(self, *, device: _Device, models: Sequence[GaussianModel], covariates: NDArray, sample_masks: NDArray) -> None:
        array_module = device.array_module
        host_masks = np.asarray(sample_masks, dtype=np.float64)
        self.device = device
        self.covariates = array_module.asarray(covariates, dtype=array_module.float64)
        self.covariate_count = int(self.covariates.shape[1])
        self.mask_index = np.asarray([model.sample_mask_index for model in models], dtype=np.int64)
        self.sample_masks = array_module.asarray(host_masks)
        self.training = array_module.asarray(host_masks[self.mask_index].T)
        self.training_count = host_masks[self.mask_index].sum(axis=1)
        self.is_binary = np.asarray([model.trait_type == TraitType.BINARY for model in models], dtype=bool)
        self.targets = array_module.asarray(np.column_stack([np.asarray(model.targets, dtype=np.float64) for model in models]))
        self.offsets = array_module.asarray(
            np.column_stack([np.asarray(model.predictor_offset, dtype=np.float64) for model in models])
        )
        if self.covariates.shape[0] != self.targets.shape[0]:
            raise ValueError("covariates must have one row per sample.")

    def fitted_probability(self, linear_predictor: Any) -> Any:
        return 0.5 * (1.0 + self.device.array_module.tanh(0.5 * linear_predictor))

    def curvature(self, linear_predictor: Any, noise_variance: NDArray) -> Any:
        """W per model: training mask / sigma_e^2, or mu (1 - mu) on the training mask for a binary model."""
        array_module = self.device.array_module
        probability = self.fitted_probability(linear_predictor)
        gaussian = self.training / array_module.asarray(noise_variance)[None, :]
        binary = self.training * probability * (1.0 - probability)
        return array_module.where(array_module.asarray(self.is_binary)[None, :], binary, gaussian)

    def score(self, linear_predictor: Any, noise_variance: NDArray) -> Any:
        """d loglik / d eta on the training samples."""
        array_module = self.device.array_module
        gaussian = self.training * (self.targets - linear_predictor) / array_module.asarray(noise_variance)[None, :]
        binary = self.training * (self.targets - self.fitted_probability(linear_predictor))
        return array_module.where(array_module.asarray(self.is_binary)[None, :], binary, gaussian)

    def binary_log_likelihood(self, linear_predictor: Any) -> NDArray:
        array_module = self.device.array_module
        softplus = array_module.logaddexp(0.0, linear_predictor)
        return self.device.to_host(array_module.sum(self.training * (self.targets * linear_predictor - softplus), axis=0))

    def covariate_inverses(self, curvature: Any) -> Any:
        """(C' W_m C + ridge)^-1 per model: (K, k, k)."""
        array_module = self.device.array_module
        weighted = array_module.einsum("sa,sm,sb->mab", self.covariates, curvature, self.covariates)
        ridge = _COVARIATE_RIDGE * array_module.trace(weighted, axis1=1, axis2=2) / self.covariate_count
        return array_module.linalg.inv(weighted + ridge[:, None, None] * array_module.eye(self.covariate_count)[None, :, :])

    def solve_covariates(self, inverses: Any, right_hand_side: Any, column_models: NDArray) -> Any:
        array_module = self.device.array_module
        return array_module.einsum("cab,bc->ac", inverses[array_module.asarray(column_models)], right_hand_side)

    def project(self, inverses: Any, curvature: Any, values: Any, column_models: NDArray) -> Any:
        """P_W v per column: W v - W C (C' W C)^-1 C' W v, with W the column's model's curvature."""
        column_curvature = curvature[:, self.device.array_module.asarray(column_models)]
        weighted = column_curvature * values
        coefficients = self.solve_covariates(inverses, self.covariates.T @ weighted, column_models)
        return weighted - column_curvature * (self.covariates @ coefficients)

    def complement_noise(self, inverses: Any, curvature: Any, noise: Any, column_models: NDArray) -> Any:
        """s = W^1/2 e - W C (C' W C)^-1 C' W^1/2 e, so that Cov(X' s) = X' P_W X."""
        column_curvature = curvature[:, self.device.array_module.asarray(column_models)]
        root_weighted = self.device.array_module.sqrt(column_curvature) * noise
        coefficients = self.solve_covariates(inverses, self.covariates.T @ root_weighted, column_models)
        return root_weighted - column_curvature * (self.covariates @ coefficients)


class _Reads:
    """One read of the genotype blocks, counted: X_b' left per block, then X_b right accumulated."""

    def __init__(self, source: GenotypeBlockSource, device: _Device) -> None:
        self.source = source
        self.device = device
        self.count = 0

    def sweep(self, left: Any, local: Callable[[int, NDArray, GenotypeBlockTile, Any], Any], image_columns: int) -> Any:
        """For every block b: right_b = local(b, variants, tile, X_b' left); returns sum_b X_b right_b (n, image_columns).

        ``left`` may be None (no transposed product); ``local`` may return None (no image).
        """
        array_module = self.device.array_module
        self.count += 1
        image = array_module.zeros((self.source.sample_count, image_columns), dtype=array_module.float64)
        for block_index, tile in self.source.iter_tiles():
            variants = self.source.block_variant_indices[block_index]
            products = None if left is None else tile.rmatmat(left)
            right = local(block_index, variants, tile, products)
            if right is not None:
                image += tile.matmat(right)
        return image


def _spd_inverse(array_module: Any, matrices: Any) -> tuple[Any, Any]:
    """Inverses and their diagonals of a stack of SPD matrices, through Jacobi-scaled Cholesky factors."""
    scale = array_module.sqrt(array_module.diagonal(matrices, axis1=1, axis2=2))
    factor = array_module.linalg.cholesky(matrices / (scale[:, :, None] * scale[:, None, :]))
    inverse_factor = array_module.linalg.inv(factor)
    pair_scale = scale[:, :, None] * scale[:, None, :]
    inverse = array_module.matmul(array_module.swapaxes(inverse_factor, 1, 2), inverse_factor) / pair_scale
    inverse_diagonal = array_module.sum(inverse_factor * inverse_factor, axis=1) / (scale * scale)
    return inverse, inverse_diagonal


def _restricted_gram(array_module: Any, tile: GenotypeBlockTile, weights: Any, covariates: Any) -> Any:
    """X_b' P_W X_b for an (n,) weight vector, with the module's covariate ridge."""
    cross = tile.weighted_cross(weights, covariates)
    covariate_gram = covariates.T @ (weights[:, None] * covariates)
    covariate_count = int(covariates.shape[1])
    ridge = _COVARIATE_RIDGE * array_module.trace(covariate_gram) / covariate_count
    normal = covariate_gram + ridge * array_module.eye(covariate_count)
    return tile.weighted_gram(weights) - cross @ array_module.linalg.solve(normal, cross.T)


class _BlockJacobi:
    """Per block and model, the inverse of the block's restricted precision, factored block by block in a read.

    A model marked in ``exact_curvature`` uses X_b' P_W X_b at its own
    curvature; the others use mean(w_m) X_b' P_mask X_b, which is exact for a
    quantitative model (constant curvature on its mask).
    """

    def __init__(self, *, system: _SampleSystem, curvature: Any, site_precision: NDArray, exact_curvature: NDArray) -> None:
        array_module = system.device.array_module
        self.system = system
        self.curvature = curvature
        self.site_precision = site_precision
        self.exact_curvature = np.asarray(exact_curvature, dtype=bool)
        self.mean_curvature = curvature.sum(axis=0) / array_module.asarray(system.training_count)
        self.model_count = int(system.mask_index.shape[0])
        self.inverse: dict[int, Any] = {}
        self.inverse_diagonal = np.empty_like(site_precision)
        self.block_variant_indices: dict[int, NDArray] = {}

    def factor_block(self, block_index: int, variants: NDArray, tile: GenotypeBlockTile) -> None:
        system = self.system
        array_module = system.device.array_module
        mask_grams = {
            int(mask_index): _restricted_gram(array_module, tile, system.sample_masks[int(mask_index)], system.covariates)
            for mask_index in np.unique(system.mask_index[~self.exact_curvature])
        }
        model_grams = [
            _restricted_gram(array_module, tile, self.curvature[:, model_index], system.covariates)
            if self.exact_curvature[model_index]
            else self.mean_curvature[model_index] * mask_grams[int(mask_index)]
            for model_index, mask_index in enumerate(system.mask_index)
        ]
        matrices = array_module.stack(model_grams)
        diagonal = array_module.arange(int(matrices.shape[1]))
        matrices[:, diagonal, diagonal] += array_module.asarray(self.site_precision[variants]).T
        inverse, inverse_diagonal = _spd_inverse(array_module, matrices)
        self.inverse[block_index] = inverse
        self.inverse_diagonal[variants] = system.device.to_host(inverse_diagonal).T
        self.block_variant_indices[block_index] = variants

    def apply_block(self, block_index: int, block_values: Any, column_models: NDArray) -> Any:
        """B_b applied to a (p_b, c) device block, one product per model."""
        array_module = self.system.device.array_module
        result = array_module.empty_like(block_values)
        for model_index in range(self.model_count):
            columns = np.flatnonzero(column_models == model_index)
            if columns.size:
                result[:, columns] = self.inverse[block_index][model_index] @ block_values[:, columns]
        return result

    def apply(self, values: NDArray, column_models: NDArray) -> NDArray:
        """B applied to every column of a (p, c) host array (no read)."""
        device = self.system.device
        result = np.empty_like(values)
        for block_index, variants in self.block_variant_indices.items():
            result[variants] = device.to_host(self.apply_block(block_index, device.array_module.asarray(values[variants]), column_models))
        return result


def _orthonormal_directions(
    array_module: Any, values: NDArray, values_image: Any, column_models: NDArray, capacity: NDArray
) -> tuple[NDArray, Any, NDArray]:
    """Per model, an orthonormal basis of its columns' span (SVQB), dropping dependent directions.

    Returns (basis, X basis, model of every basis column). The drop keeps block
    CG breakdown-free when some of a model's columns have converged, and a model
    keeps at most ``capacity`` directions, its unexplored Krylov dimension.
    """
    bases = []
    images = []
    models = []
    for model_index in range(capacity.shape[0]):
        columns = np.flatnonzero(column_models == model_index)
        if columns.size == 0 or capacity[model_index] <= 0:
            continue
        block = values[:, columns]
        gram = block.T @ block
        if not np.all(np.isfinite(gram)):
            raise FloatingPointError("A block-CG direction is not finite.")
        eigenvalues, eigenvectors = np.linalg.eigh(0.5 * (gram + gram.T))
        kept = eigenvalues > max(_DIRECTION_DROP * float(eigenvalues[-1]), np.finfo(np.float64).tiny)
        kept[: max(eigenvalues.shape[0] - int(capacity[model_index]), 0)] = False
        if not np.any(kept):
            continue
        transform = eigenvectors[:, kept] / np.sqrt(eigenvalues[kept])
        bases.append(block @ transform)
        images.append(values_image[:, array_module.asarray(columns)] @ array_module.asarray(transform))
        models.append(np.full(int(kept.sum()), model_index))
    if not bases:
        return np.zeros((values.shape[0], 0)), array_module.zeros((values_image.shape[0], 0)), np.zeros(0, dtype=np.int64)
    return np.concatenate(bases, axis=1), array_module.concatenate(images, axis=1), np.concatenate(models)


def _block_conjugate_gradient(
    *,
    reads: _Reads,
    system: _SampleSystem,
    curvature: Any,
    inverses: Any,
    preconditioner: _BlockJacobi,
    site_precision: NDArray,
    residual: NDArray,
    preconditioned: NDArray,
    preconditioned_image: Any,
    column_models: NDArray,
    tolerance: float,
) -> tuple[NDArray, Any, NDArray, int]:
    """Solve A x = r0 from x = 0 by block CG per model, one read per iteration.

    Every model's columns (its mean step, probes or draws) share one Krylov
    space: each iteration's direction block is an orthonormal basis of the
    model's preconditioned residuals plus the conjugated previous block. The
    read forms A P = X' P_W (X P) + diag(tau) P, M^-1 A P and X M^-1 A P, so
    X of the next block follows by linearity. Given r0, z0 = M^-1 r0 and X z0;
    returns (x, X x, final relative residual per column, iterations).
    """
    device = system.device
    array_module = device.array_module
    model_count = int(system.mask_index.shape[0])
    residual = residual.copy()
    preconditioned = preconditioned.copy()
    preconditioned_image = preconditioned_image.copy()
    solution = np.zeros_like(residual)
    solution_image = array_module.zeros_like(preconditioned_image)
    right_hand_side_norm = np.linalg.norm(residual, axis=0)
    safe_norm = np.where(right_hand_side_norm > 0.0, right_hand_side_norm, 1.0)
    relative_residual = np.where(right_hand_side_norm > 0.0, 1.0, 0.0)
    active_models = np.unique(column_models[relative_residual > tolerance])
    # Block CG explores at most p directions per model; past that its space is complete.
    capacity = np.full(model_count, residual.shape[0])
    direction, direction_image, direction_models = _orthonormal_directions(
        array_module, preconditioned, preconditioned_image, np.where(np.isin(column_models, active_models), column_models, -1), capacity
    )
    capacity -= np.bincount(direction_models, minlength=model_count)
    iterations = 0
    for iterations in range(_MAXIMUM_CONJUGATE_GRADIENT_ITERATIONS + 1):
        if direction.shape[1] == 0 or iterations == _MAXIMUM_CONJUGATE_GRADIENT_ITERATIONS:
            break
        operator_direction = np.empty_like(direction)
        preconditioned_operator = np.empty_like(direction)

        def local(block_index: int, variants: NDArray, tile: GenotypeBlockTile, products: Any) -> Any:
            applied = device.to_host(products) + site_precision[variants][:, direction_models] * direction[variants]
            operator_direction[variants] = applied
            block_preconditioned = preconditioner.apply_block(block_index, array_module.asarray(applied), direction_models)
            preconditioned_operator[variants] = device.to_host(block_preconditioned)
            return block_preconditioned

        projected = system.project(inverses, curvature, direction_image, direction_models)
        preconditioned_operator_image = reads.sweep(projected, local, int(direction.shape[1]))
        next_candidates = []
        next_candidate_images = []
        next_models = []
        for model_index in np.unique(direction_models):
            block_columns = np.flatnonzero(direction_models == model_index)
            model_columns = np.flatnonzero(column_models == model_index)
            block = direction[:, block_columns]
            applied = operator_direction[:, block_columns]
            curvature_matrix = block.T @ applied
            curvature_matrix = 0.5 * (curvature_matrix + curvature_matrix.T)
            if not np.all(np.isfinite(curvature_matrix)):
                raise FloatingPointError("A block-CG curvature is not finite.")
            step = np.linalg.solve(curvature_matrix, block.T @ residual[:, model_columns])
            device_step = array_module.asarray(step)
            device_block_columns = array_module.asarray(block_columns)
            device_model_columns = array_module.asarray(model_columns)
            solution[:, model_columns] += block @ step
            solution_image[:, device_model_columns] += direction_image[:, device_block_columns] @ device_step
            residual[:, model_columns] -= applied @ step
            preconditioned[:, model_columns] -= preconditioned_operator[:, block_columns] @ step
            preconditioned_image[:, device_model_columns] -= preconditioned_operator_image[:, device_block_columns] @ device_step
            relative_residual[model_columns] = np.linalg.norm(residual[:, model_columns], axis=0) / safe_norm[model_columns]
            if float(np.max(relative_residual[model_columns])) <= tolerance:
                continue
            conjugate = -np.linalg.solve(curvature_matrix, applied.T @ preconditioned[:, model_columns])
            next_candidates.append(preconditioned[:, model_columns] + block @ conjugate)
            next_candidate_images.append(
                preconditioned_image[:, device_model_columns] + direction_image[:, device_block_columns] @ array_module.asarray(conjugate)
            )
            next_models.append(np.full(model_columns.size, model_index))
        if not next_candidates:
            direction = np.zeros((residual.shape[0], 0))
            continue
        direction, direction_image, direction_models = _orthonormal_directions(
            array_module,
            np.concatenate(next_candidates, axis=1),
            array_module.concatenate(next_candidate_images, axis=1),
            np.concatenate(next_models),
            capacity,
        )
        capacity -= np.bincount(direction_models, minlength=model_count)
    return solution, solution_image, relative_residual, iterations


class FullDataGaussian:
    """The mean, probes and draws of q(beta) for every model, updated one Newton step at a time.

    ``iterate`` takes the sites (tau, nu) and noise variances of this iteration.
    Its first read certifies the state it starts from (exact gradient,
    covariate gradient, probe residuals), refactors the blocks when asked and
    starts CG; the Newton step and the probe corrections then take one read per
    CG iteration.
    """

    def __init__(
        self,
        *,
        source: GenotypeBlockSource,
        models: Sequence[GaussianModel],
        covariates: NDArray,
        sample_masks: NDArray,
        initial_mean: NDArray,
        probe_count: int,
        seed: int,
    ) -> None:
        self.device = _Device(source.array_module)
        self.reads = _Reads(source, self.device)
        self.system = _SampleSystem(device=self.device, models=models, covariates=covariates, sample_masks=sample_masks)
        array_module = self.device.array_module
        self.variant_count = int(sum(indices.shape[0] for indices in source.block_variant_indices))
        model_count = len(models)
        if initial_mean.shape != (self.variant_count, model_count):
            raise ValueError("initial_mean must be (variants, models).")
        self.model_count = model_count
        self.probe_count = int(probe_count)
        self.generator = np.random.default_rng(seed)
        self.mean = np.asarray(initial_mean, dtype=np.float64).copy()
        self.probes = self.generator.choice(np.array([-1.0, 1.0]), size=(self.variant_count, self.probe_count))
        self.probe_models = np.repeat(np.arange(model_count), self.probe_count)
        self.probe_solution = np.zeros((self.variant_count, model_count * self.probe_count))
        self.probe_image = array_module.zeros((source.sample_count, model_count * self.probe_count), dtype=array_module.float64)
        self.alpha = array_module.zeros((self.system.covariate_count, model_count), dtype=array_module.float64)
        self.genetic_image = self.reads.sweep(None, lambda _block, variants, _tile, _products: array_module.asarray(self.mean[variants]), model_count)
        self.preconditioner: _BlockJacobi | None = None
        self.noise_variance = np.ones(model_count)
        # diag(X' P_W X), and W, at the curvature the probes were last solved for.
        self.data_diagonal = np.zeros((self.variant_count, model_count))
        self.probe_curvature = np.zeros((source.sample_count, model_count))
        # CG iterations (one read each) of the last iterate.
        self.conjugate_gradient_iterations = 0
        self._fit_covariates()

    def _fit_covariates(self) -> None:
        """alpha maximizing the likelihood at the current beta: exact for Gaussian, Newton to convergence for binary."""
        array_module = self.device.array_module
        mean_models = np.arange(self.model_count)
        for _newton in range(_COVARIATE_NEWTON_ITERATIONS):
            linear_predictor = self.system.offsets + self.genetic_image + self.system.covariates @ self.alpha
            curvature = self.system.curvature(linear_predictor, self.noise_variance)
            inverses = self.system.covariate_inverses(curvature)
            step = self.system.solve_covariates(
                inverses, self.system.covariates.T @ self.system.score(linear_predictor, self.noise_variance), mean_models
            )
            self.alpha = self.alpha + step
            scale = float(array_module.max(array_module.abs(self.alpha))) + 1.0
            if float(array_module.max(array_module.abs(step))) <= _COVARIATE_NEWTON_TOLERANCE * scale:
                break
        self.linear_predictor = self.system.offsets + self.genetic_image + self.system.covariates @ self.alpha

    def _penalized_objective(self, linear_predictor: Any, mean: NDArray, site_precision: NDArray, site_shift: NDArray) -> NDArray:
        """Binary penalized log-likelihood per model: loglik - tau beta^2 / 2 + nu beta."""
        penalty = np.sum(-0.5 * site_precision * mean * mean + site_shift * mean, axis=0)
        return self.system.binary_log_likelihood(linear_predictor) + penalty

    def iterate(
        self,
        *,
        site_precision: NDArray,
        site_shift: NDArray,
        noise_variance: NDArray,
        tolerance: float,
        refactor: bool,
        exact_curvature: NDArray,
    ) -> StartCertificate:
        """One certifying read at the start state, then one Newton step and the probe corrections.

        ``refactor`` rebuilds the block-Jacobi inverse at this state within the
        first read, with the exact curvature for the models in ``exact_curvature``.
        """
        device = self.device
        array_module = device.array_module
        system = self.system
        model_count = self.model_count
        self.noise_variance = np.asarray(noise_variance, dtype=np.float64).copy()
        mean_models = np.arange(model_count)
        columns = np.concatenate([mean_models, self.probe_models])
        curvature = system.curvature(self.linear_predictor, self.noise_variance)
        inverses = system.covariate_inverses(curvature)
        refactor = refactor or self.preconditioner is None
        if refactor:
            self.preconditioner = _BlockJacobi(
                system=system, curvature=curvature, site_precision=site_precision, exact_curvature=exact_curvature
            )
        preconditioner = self.preconditioner
        score = system.score(self.linear_predictor, self.noise_variance)
        projected_score = score - curvature * (
            system.covariates @ system.solve_covariates(inverses, system.covariates.T @ score, mean_models)
        )
        weighted_covariates = array_module.concatenate(
            [curvature[:, model_index : model_index + 1] * system.covariates for model_index in range(model_count)], axis=1
        )
        left = array_module.concatenate(
            [projected_score, system.project(inverses, curvature, self.probe_image, self.probe_models), weighted_covariates],
            axis=1,
        )
        probe_column_count = model_count * self.probe_count
        host_inverses = device.to_host(inverses)
        residual = np.empty((self.variant_count, model_count + probe_column_count))
        preconditioned = np.empty_like(residual)
        score_square_sum = np.zeros(model_count)
        penalty_square_sum = np.zeros(model_count)

        def start(block_index: int, variants: NDArray, tile: GenotypeBlockTile, products: Any) -> Any:
            if refactor:
                preconditioner.factor_block(block_index, variants, tile)
            host_products = device.to_host(products)
            score_products = host_products[:, :model_count]
            probe_products = host_products[:, model_count : model_count + probe_column_count]
            cross = host_products[:, model_count + probe_column_count :].reshape(variants.shape[0], model_count, system.covariate_count)
            squares = device.to_host(tile.weighted_column_squares(curvature))
            self.data_diagonal[variants] = squares - np.einsum("jma,mab,jmb->jm", cross, host_inverses, cross)
            penalty = site_precision[variants] * self.mean[variants] - site_shift[variants]
            score_square_sum[:] += np.sum(score_products * score_products, axis=0)
            penalty_square_sum[:] += np.sum(penalty * penalty, axis=0)
            residual[variants, :model_count] = score_products - penalty
            residual[variants, model_count:] = (
                np.tile(self.probes[variants], (1, model_count))
                - probe_products
                - site_precision[variants][:, self.probe_models] * self.probe_solution[variants]
            )
            block_preconditioned = preconditioner.apply_block(block_index, array_module.asarray(residual[variants]), columns)
            preconditioned[variants] = device.to_host(block_preconditioned)
            return block_preconditioned

        preconditioned_image = self.reads.sweep(left, start, int(columns.size))
        self.probe_curvature = device.to_host(curvature)
        gradient_norm = np.linalg.norm(residual[:, :model_count], axis=0)
        probe_norm = np.linalg.norm(residual[:, model_count:], axis=0) / np.sqrt(self.variant_count)
        certificate = StartCertificate(
            gradient_relative_norm=gradient_norm
            / np.maximum(np.sqrt(score_square_sum) + np.sqrt(penalty_square_sum), np.finfo(np.float64).tiny),
            covariate_gradient_relative_norm=device.to_host(
                array_module.linalg.norm(system.covariates.T @ score, axis=0)
                / array_module.maximum(
                    array_module.linalg.norm(array_module.abs(system.covariates).T @ array_module.abs(score), axis=0),
                    np.finfo(np.float64).tiny,
                )
            ),
            probe_residual=np.max(probe_norm.reshape(model_count, self.probe_count), axis=1),
        )
        correction, correction_image, _residual, self.conjugate_gradient_iterations = _block_conjugate_gradient(
            reads=self.reads,
            system=system,
            curvature=curvature,
            inverses=inverses,
            preconditioner=preconditioner,
            site_precision=site_precision,
            residual=residual,
            preconditioned=preconditioned,
            preconditioned_image=preconditioned_image,
            column_models=columns,
            tolerance=tolerance,
        )
        self.probe_solution += correction[:, model_count:]
        self.probe_image += correction_image[:, model_count:]
        step = correction[:, :model_count]
        step_image = correction_image[:, :model_count]
        # The covariates move with beta along the profiled Newton direction.
        alpha_step = system.solve_covariates(inverses, system.covariates.T @ (score - curvature * step_image), mean_models)
        sample_step = step_image + system.covariates @ alpha_step
        step_length = np.ones(model_count)
        binary = np.flatnonzero(system.is_binary)
        if binary.size:
            start_value = self._penalized_objective(self.linear_predictor, self.mean, site_precision, site_shift)
            pending = binary.copy()
            while pending.size and float(np.min(step_length[pending])) >= _MINIMUM_NEWTON_STEP:
                trial_mean = self.mean + step_length[None, :] * step
                trial_predictor = self.linear_predictor + array_module.asarray(step_length)[None, :] * sample_step
                trial_value = self._penalized_objective(trial_predictor, trial_mean, site_precision, site_shift)
                pending = pending[trial_value[pending] < start_value[pending]]
                step_length[pending] *= 0.5
            step_length[pending] = 0.0
        device_length = array_module.asarray(step_length)[None, :]
        self.mean += step_length[None, :] * step
        self.alpha = self.alpha + device_length * alpha_step
        self.genetic_image = self.genetic_image + device_length * step_image
        self.linear_predictor = self.linear_predictor + device_length * sample_step
        return certificate

    def marginal_variances(self, site_precision: NDArray) -> MarginalVariances:
        """diag(A^-1) of the operator the probes were last solved for, projected onto [1/A_jj, 1/tau_j]."""
        assert self.preconditioner is not None
        repeated = np.tile(self.probes, (1, self.model_count))
        control_images = self.preconditioner.apply(repeated, self.probe_models)
        correction = (repeated * (self.probe_solution - control_images)).reshape(
            self.variant_count, self.model_count, self.probe_count
        )
        estimate = self.preconditioner.inverse_diagonal + correction.mean(axis=2)
        lower = 1.0 / (self.data_diagonal + site_precision)
        upper = np.where(site_precision > 0.0, 1.0 / np.where(site_precision > 0.0, site_precision, 1.0), np.inf)
        variance = np.clip(estimate, lower, upper)
        leverage = np.sum(correction * site_precision[:, :, None], axis=0)
        total = np.maximum(np.sum(variance * site_precision, axis=0), np.finfo(np.float64).tiny)
        return MarginalVariances(
            variance=variance,
            relative_standard_error=np.std(leverage, axis=1, ddof=1) / np.sqrt(self.probe_count) / total,
            clamped_fraction=np.mean((estimate < lower) | (estimate > upper), axis=0),
        )

    def draws(self, *, site_precision: NDArray, draw_count: int, tolerance: float) -> NDArray:
        """Exact draws of q(beta) per model: (p, K, draw_count), by perturb-and-solve.

        Draws are solved _DRAW_BLOCK_COLUMNS per model at a time, which bounds the
        block-CG work per iteration.
        """
        chunks = [
            self._draw_chunk(site_precision=site_precision, draw_count=min(_DRAW_BLOCK_COLUMNS, draw_count - first), tolerance=tolerance)
            for first in range(0, draw_count, _DRAW_BLOCK_COLUMNS)
        ]
        return np.concatenate(chunks, axis=2)

    def _draw_chunk(self, *, site_precision: NDArray, draw_count: int, tolerance: float) -> NDArray:
        device = self.device
        array_module = device.array_module
        system = self.system
        assert self.preconditioner is not None
        preconditioner = self.preconditioner
        draw_models = np.repeat(np.arange(self.model_count), draw_count)
        curvature = system.curvature(self.linear_predictor, self.noise_variance)
        inverses = system.covariate_inverses(curvature)
        sample_noise = array_module.asarray(self.generator.standard_normal((self.reads.source.sample_count, draw_models.size)))
        complement = system.complement_noise(inverses, curvature, sample_noise, draw_models)
        prior_part = np.sqrt(site_precision[:, draw_models]) * self.generator.standard_normal((self.variant_count, draw_models.size))
        residual = np.empty((self.variant_count, draw_models.size))
        preconditioned = np.empty_like(residual)

        def start(block_index: int, variants: NDArray, tile: GenotypeBlockTile, products: Any) -> Any:
            residual[variants] = device.to_host(products) + prior_part[variants]
            block_preconditioned = preconditioner.apply_block(block_index, array_module.asarray(residual[variants]), draw_models)
            preconditioned[variants] = device.to_host(block_preconditioned)
            return block_preconditioned

        preconditioned_image = self.reads.sweep(complement, start, int(draw_models.size))
        perturbation, _image, _residual, _iterations = _block_conjugate_gradient(
            reads=self.reads,
            system=system,
            curvature=curvature,
            inverses=inverses,
            preconditioner=preconditioner,
            site_precision=site_precision,
            residual=residual,
            preconditioned=preconditioned,
            preconditioned_image=preconditioned_image,
            column_models=draw_models,
            tolerance=tolerance,
        )
        return (self.mean[:, draw_models] + perturbation).reshape(self.variant_count, self.model_count, draw_count)
