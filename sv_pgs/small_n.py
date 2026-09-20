"""The small-n route: Stage 2's model (``full_data_fit``) with q(beta)'s algebra in the n x n kernel form.

Where the training samples are far fewer than the columns (a cis window: n ~ 600, p ~ 10^4), every quantity the EP-EB
fit asks of q(beta) = N(mu, Sigma) is exact from one n x n Cholesky factor per site update, so none needs a Krylov
solve or a probe certificate:

- q's precision is A = Xp' Xp / sigma^2 + diag tau with Xp = (I - H_C) X~ on the training rows. With the scaled sites
  t = sigma^2 tau it is A = A' / sigma^2, A' = Xp' Xp + diag t, and Sigma = sigma^2 A'^-1.
- The columns split into the bulk P (t > 0) and the rest N (t <= 0; EP is unclipped). On P, Woodbury gives
  A'_PP^-1 = T^-1 - T^-1 Xp_P' K^-1 Xp_P T^-1 with the kernel K = I_n + Xp_P T^-1 Xp_P' (always positive definite).
  N enters exactly through its Schur complement S = T_N + Xp_N' K^-1 Xp_N, and A' is positive definite exactly when
  S is (``np.linalg.LinAlgError`` otherwise).
- So Sigma = sigma^2 (Delta - Phi'Phi + Psi'Psi): Delta = diag(1/t) on P and 0 on N, Phi = U^-T Xp_P T^-1 (n x |P|,
  zero on N, K = U'U), and Psi = [S_U^-T E' | -S_U^-T] (|N| x p, S = S_U'S_U, E = T^-1 Xp_P' K^-1 Xp_N). The marginal
  variances, solves, the variance JVP -(Sigma o Sigma) W and exact posterior draws all come from these factors.

The fixed-point iteration and the outer loop are Stage 2's (``full_data_fit._FullDataFixedPoints`` and
``scale_mixture_ep.fit_hyperparameters``), with every certified quantity replaced by its exact value: the mean move
r' Sigma r, the variances diag Sigma, and no cavity-information certificate (the variances are exact).

The genotype side is Stage 0's (``genotype_statistics``) on a dense training matrix: signed codes (code - 127), their
training means and population SDs, every polymorphic record active (no rarity or other threshold filter: SPEC), and
exact ties (equal or negated standardized columns) merged by the same integer test, over the whole window rather than
within an LD block. The dense route's cost per site update is O(n^2 p) in BLAS-3; the variance JVP is formed as
Sigma o Sigma once per fixed point when it fits the working memory (one GEMM per call after that), else from the
factors at O(n^2 p) per column (``DENSE_JVP_BYTES`` is what decides; see ``_DensePosterior``).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, Sequence

import numpy as np
from scipy import linalg

from sv_pgs._typing import F64Array, I64Array
from sv_pgs.config import TraitType
from sv_pgs.data import TieMap
from sv_pgs.fast_scoring import SIGNED_CODE_OFFSET, ScoringModel
from sv_pgs.full_data_fit import FitCertificate, NoFixedPoint
from sv_pgs.genotype_statistics import _covariate_gram_pseudo_inverse
from sv_pgs.scale_mixture_ep import (
    Cavity,
    FixedPoint,
    GaussianPosterior,
    MixtureHyperparameters,
    ScaleMixturePrior,
    derived_lattice,
    fit_hyperparameters,
    initial_hyperparameters,
    moment_matched_prior_sites,
    noise_gain,
    noise_variance,
    prior_second_moment,
    scale_mixture_prior,
    site_targets,
    tilted_moments,
)
from sv_pgs.tie_map import _compact_identity_tie_map, tie_map_from_groups

_EPSILON = float(np.finfo(np.float64).eps)
_FLOAT_BYTES = np.dtype(np.float64).itemsize


# ------------------------------------------------------------------ the genotype side (Stage 0, dense)


@dataclass(frozen=True)
class DenseStatistics:
    """Stage 0 of one training set on a dense code matrix.

    ``active_rows`` are the input columns with training variance (the store rows of ``fast_scoring``), ``means`` and
    ``scales`` their signed-code means and population SDs, ``tie_map`` their ties; ``projected`` (n x reduced,
    Fortran order) is (I - H_C) X~ of the reduced columns, ``loading`` = C' X~ (k x reduced), and ``projected_target``
    = (I - H_C) y.
    """

    active_rows: I64Array
    means: F64Array
    scales: F64Array
    tie_map: TieMap
    projected: F64Array
    loading: F64Array
    covariates: F64Array
    covariate_pseudo_inverse: F64Array
    target: F64Array
    projected_target: F64Array

    @property
    def sample_count(self) -> int:
        return int(self.projected.shape[0])

    @property
    def reduced_rows(self) -> I64Array:
        """The input columns of the reduced columns (each tie group's representative)."""
        return self.active_rows[np.asarray(self.tie_map.kept_indices, dtype=np.int64)]


def _ties(signed: np.ndarray, sums: np.ndarray, count: int) -> TieMap:
    """Exact ties among the columns of ``signed`` (int64, n x a): columns whose centred integer vectors n x - sum x
    are proportional, i.e. N_jk^2 = N_jj N_kk with N = n S - u u' (Stage 0's test), found by each column's canonical
    form (divided by its gcd, first nonzero entry positive) rather than a correlation screen. The representative is
    the lowest index; a member's sign is + for a copy and - for a negated copy."""
    width = signed.shape[1]
    centred = count * signed - sums[None, :]
    divisor = np.gcd.reduce(np.abs(centred), axis=0)
    canonical = (centred // divisor[None, :]).T.copy()
    first = canonical[np.arange(width), np.argmax(canonical != 0, axis=1)]
    orientation = np.sign(first).astype(np.int64)
    canonical *= orientation[:, None]
    groups: dict[bytes, list[int]] = {}
    for column in range(width):
        groups.setdefault(canonical[column].tobytes(), []).append(column)
    if len(groups) == width:
        return _compact_identity_tie_map(width)
    members = {
        columns[0]: [(column, float(orientation[column] * orientation[columns[0]])) for column in columns]
        for columns in groups.values()
    }
    return tie_map_from_groups(width, members)


def dense_statistics(codes: np.ndarray, covariates: F64Array, target: F64Array) -> DenseStatistics:
    """Stage 0 of ``codes`` (n x records, store codes 0..254 = dosage x 127) with ``covariates`` (n x k, intercept
    included) and ``target`` (n,)."""
    values = np.asarray(codes)
    if values.ndim != 2 or values.min(initial=0) < 0 or values.max(initial=0) > 2 * SIGNED_CODE_OFFSET:
        raise ValueError("codes must be store codes 0..254 of shape [samples, records].")
    count = int(values.shape[0])
    signed = values.astype(np.int64) - int(SIGNED_CODE_OFFSET)
    sums = signed.sum(axis=0)
    numerator = count * np.einsum("ij,ij->j", signed, signed) - sums * sums
    active = np.flatnonzero(numerator > 0)
    signed = signed[:, active]
    sums = sums[active]
    means = sums / count
    scales = np.sqrt(numerator[active].astype(np.float64)) / count
    tie_map = _ties(signed, sums, count)
    kept = np.asarray(tie_map.kept_indices, dtype=np.int64)
    standardized = (signed[:, kept] - means[kept][None, :]) / scales[kept][None, :]
    covariate_matrix = np.asarray(covariates, dtype=np.float64)
    pseudo_inverse = _covariate_gram_pseudo_inverse(covariate_matrix)
    loading = covariate_matrix.T @ standardized
    projected = np.asfortranarray(standardized - covariate_matrix @ (pseudo_inverse @ loading))
    target_values = np.asarray(target, dtype=np.float64)
    projected_target = target_values - covariate_matrix @ (pseudo_inverse @ (covariate_matrix.T @ target_values))
    return DenseStatistics(
        active_rows=active.astype(np.int64),
        means=means,
        scales=scales,
        tie_map=tie_map,
        projected=projected,
        loading=loading,
        covariates=covariate_matrix,
        covariate_pseudo_inverse=pseudo_inverse,
        target=target_values,
        projected_target=projected_target,
    )


# ------------------------------------------------------------------ q(beta) in kernel form


class _Kernel:
    """A' = Xp' Xp + diag t at the scaled sites t (see the module docstring), factored once.

    ``solve(R)`` is A'^-1 R, ``variances()`` diag A'^-1, and ``factors()`` (Delta, Phi, Psi) with
    A'^-1 = Delta - Phi'Phi + Psi'Psi. Raises ``np.linalg.LinAlgError`` when A' is not positive definite.
    """

    def __init__(self, design: F64Array, scaled_precision: F64Array) -> None:
        self.design = design
        self.precision = scaled_precision
        self.variant_count = int(design.shape[1])
        positive = scaled_precision > 0.0
        self.bulk = np.flatnonzero(positive)
        self.rest = np.flatnonzero(~positive)
        self.inverse = 1.0 / scaled_precision[self.bulk]
        # Xp_P as an F-ordered (n x |P|) block: BLAS reads it without a copy.
        self.bulk_design = np.asfortranarray(design[:, self.bulk])
        scaled = self.bulk_design * np.sqrt(self.inverse)[None, :]
        kernel = linalg.blas.dsyrk(1.0, scaled)
        kernel[np.diag_indices_from(kernel)] += 1.0
        self.upper = linalg.cholesky(kernel, lower=False, check_finite=False, overwrite_a=True)
        self._whitened: F64Array | None = None
        self._factors: tuple[F64Array, F64Array, F64Array] | None = None
        if self.rest.size:
            self.rest_design = np.asfortranarray(design[:, self.rest])
            self.rest_whitened = linalg.solve_triangular(self.upper, self.rest_design, trans="T", lower=False, check_finite=False)
            schur = self.rest_whitened.T @ self.rest_whitened
            schur[np.diag_indices_from(schur)] += scaled_precision[self.rest]
            self.schur_upper = linalg.cholesky(schur, lower=False, check_finite=False)

    def _kernel_solve(self, values: F64Array) -> F64Array:
        return linalg.cho_solve((self.upper, False), values, check_finite=False)

    def _bulk_apply(self, values: F64Array) -> tuple[F64Array, F64Array]:
        """(A'_PP^-1 v, Xp_P T^-1 v) for v (|P| x q)."""
        scaled = self.inverse[:, None] * values
        image = self.bulk_design @ scaled
        return scaled - self.inverse[:, None] * (self.bulk_design.T @ self._kernel_solve(image)), image

    def solve(self, right: F64Array) -> F64Array:
        values = np.asarray(right, dtype=np.float64)
        column = values.ndim == 1
        values = values[:, None] if column else values
        solution = np.empty_like(values)
        bulk_part, image = self._bulk_apply(values[self.bulk])
        if self.rest.size:
            # x_N = S^-1 (r_N - Xp_N' K^-1 Xp_P T^-1 r_P), x_P = A'_PP^-1 (r_P - Xp_P' Xp_N x_N).
            whitened_image = linalg.solve_triangular(self.upper, image, trans="T", lower=False, check_finite=False)
            rest_part = linalg.cho_solve((self.schur_upper, False), values[self.rest] - self.rest_whitened.T @ whitened_image, check_finite=False)
            correction, _ = self._bulk_apply(self.bulk_design.T @ (self.rest_design @ rest_part))
            bulk_part = bulk_part - correction
            solution[self.rest] = rest_part
        solution[self.bulk] = bulk_part
        return solution[:, 0] if column else solution

    def whitened(self) -> F64Array:
        """U^-T Xp_P (n x |P|), for the variances and the factors."""
        if self._whitened is None:
            self._whitened = linalg.solve_triangular(self.upper, self.bulk_design, trans="T", lower=False, check_finite=False)
        return self._whitened

    def factors(self) -> tuple[F64Array, F64Array, F64Array]:
        """(delta (p,), Phi (n x p), Psi (|N| x p)) with A'^-1 = diag(delta) - Phi'Phi + Psi'Psi."""
        if self._factors is not None:
            return self._factors
        delta = np.zeros(self.variant_count)
        delta[self.bulk] = self.inverse
        phi = np.zeros((self.upper.shape[0], self.variant_count), order="F")
        whitened = self.whitened()
        phi[:, self.bulk] = whitened * self.inverse[None, :]
        psi = np.zeros((self.rest.size, self.variant_count))
        if self.rest.size:
            # E' = Xp_N' K^-1 Xp_P T^-1 = (U^-T Xp_N)'(U^-T Xp_P) T^-1; Psi_P = S_U^-T E', Psi_N = -S_U^-T.
            cross = (self.rest_whitened.T @ whitened) * self.inverse[None, :]
            psi[:, self.bulk] = linalg.solve_triangular(self.schur_upper, cross, trans="T", lower=False, check_finite=False)
            psi[:, self.rest] = -linalg.solve_triangular(self.schur_upper, np.eye(self.rest.size), trans="T", lower=False, check_finite=False)
        self._factors = (delta, phi, psi)
        return self._factors

    def variances(self) -> F64Array:
        delta, phi, psi = self.factors()
        return delta - np.einsum("ij,ij->j", phi, phi) + np.einsum("ij,ij->j", psi, psi)

    def cavity(self) -> tuple[F64Array, F64Array, F64Array]:
        """(z', 1 - t z', 1/z' - t): the variances of A'^-1, each site's share of its variance the data remove, and its
        cavity precision, without cancellation. For a bulk column z' = d - d^2 q + f (d = 1/t, q = ||U^-T x_j||^2,
        f = ||Psi_j||^2), so 1 - t z' = d q - t f exactly; forming 1/z' - t from z' instead loses every digit where the
        data inform a column far less than its site does (d q below eps), which is most rare variants at the start."""
        delta, phi, psi = self.factors()
        extra = np.einsum("ij,ij->j", psi, psi)
        variances = delta - np.einsum("ij,ij->j", phi, phi) + extra
        removed = np.empty(self.variant_count)
        whitened = self.whitened()
        removed[self.bulk] = self.inverse * np.einsum("ij,ij->j", whitened, whitened) - self.precision[self.bulk] * extra[self.bulk]
        removed[self.rest] = 1.0 - self.precision[self.rest] * variances[self.rest]
        return variances, removed, removed / variances

    def draws(self, generator: np.random.Generator, draw_count: int) -> F64Array:
        """(p x draws) exact draws of N(0, A'^-1): beta_N from its marginal N(0, S^-1), then beta_P | beta_N with
        precision A'_PP (Bhattacharya et al. 2016: u ~ N(0, T^-1), e ~ N(0, I_n), u - T^-1 Xp_P' K^-1 (Xp_P u + e))."""
        draws = np.zeros((self.variant_count, draw_count))
        prior_noise = generator.standard_normal((self.bulk.size, draw_count))
        sample_noise = generator.standard_normal((self.upper.shape[0], draw_count))
        spread = np.sqrt(self.inverse)[:, None] * prior_noise
        image = self.bulk_design @ spread + sample_noise
        bulk_draw = spread - self.inverse[:, None] * (self.bulk_design.T @ self._kernel_solve(image))
        if self.rest.size:
            rest_noise = generator.standard_normal((self.rest.size, draw_count))
            rest_offset = linalg.solve_triangular(self.schur_upper, rest_noise, lower=False, check_finite=False)
            draws[self.rest] += rest_offset
            # E (beta_N - mean_N) with E = A'_PP^-1 A'_PN = A'_PP^-1 Xp_P' Xp_N.
            conditional, _ = self._bulk_apply(self.bulk_design.T @ (self.rest_design @ rest_offset))
            bulk_draw = bulk_draw - conditional
        draws[self.bulk] += bulk_draw
        return draws


def _symmetrize_upper(matrix: F64Array) -> None:
    """Copy the upper triangle over the lower one in place, in row panels (no p x p temporary)."""
    size = matrix.shape[0]
    step = max(1, int(np.sqrt(size)))
    for start in range(0, size, step):
        stop = min(start + step, size)
        matrix[stop:, start:stop] = matrix[start:stop, stop:].T
        block = matrix[start:stop, start:stop]
        block[...] = np.triu(block) + np.triu(block, 1).T


def _hadamard_gram_product(left: F64Array, right: F64Array, weights: F64Array) -> F64Array:
    """((L'L) o (R'R)) W, column by column: out_jc = l_j' (L diag(w_c) R') r_j."""
    result = np.empty_like(weights)
    for column in range(weights.shape[1]):
        core = (left * weights[:, column][None, :]) @ right.T
        result[:, column] = np.einsum("ij,ij->j", left, core @ right)
    return result


class _DensePosterior:
    """q's responses at one fixed point (``scale_mixture_ep.GaussianPosterior``): Sigma R exactly, and
    -(Sigma o Sigma) W exactly. Sigma o Sigma is formed once (O(n p^2) and 8 p^2 bytes) when that fits
    ``jvp_bytes``, and each call is then one GEMM; otherwise each column costs O(n^2 p) from the factors.
    Forming it pays as soon as a call has more than p / (2 n) columns, which the total curvature's (p x D) GMRES
    iterates always have, so memory alone decides."""

    def __init__(self, kernel: _Kernel, noise: float, jvp_bytes: int, profile: dict) -> None:
        self.kernel = kernel
        self.noise = float(noise)
        self.jvp_bytes = int(jvp_bytes)
        self.profile = profile
        self._squared: F64Array | None = None

    def solve(self, right: F64Array, _relative_tolerance: float) -> F64Array:
        started = time.perf_counter()
        result = self.noise * self.kernel.solve(right)
        self.profile["solve_seconds"] += time.perf_counter() - started
        self.profile["solve_columns"] += int(np.asarray(right).shape[1]) if np.ndim(right) == 2 else 1
        return result

    def _explicit(self) -> F64Array:
        if self._squared is None:
            started = time.perf_counter()
            delta, phi, psi = self.kernel.factors()
            covariance = linalg.blas.dsyrk(-1.0, phi, trans=1)
            if psi.shape[0]:
                covariance = linalg.blas.dsyrk(1.0, psi, beta=1.0, c=covariance, trans=1, overwrite_c=1)
            _symmetrize_upper(covariance)
            covariance[np.diag_indices_from(covariance)] += delta
            covariance *= self.noise
            np.square(covariance, out=covariance)
            self._squared = covariance
            self.profile["form_seconds"] += time.perf_counter() - started
        return self._squared

    def variance_jvp(self, weights: F64Array) -> F64Array:
        started = time.perf_counter()
        values = np.asarray(weights, dtype=np.float64)
        column = values.ndim == 1
        values = values[:, None] if column else values
        p = self.kernel.variant_count
        if _FLOAT_BYTES * p * p <= self.jvp_bytes:
            result = -(self._explicit() @ values)
        else:
            delta, phi, psi = self.kernel.factors()
            phi_squares = np.einsum("ij,ij->j", phi, phi)
            psi_squares = np.einsum("ij,ij->j", psi, psi)
            local = (delta * delta - 2.0 * delta * phi_squares + 2.0 * delta * psi_squares)[:, None] * values
            nonlocal_part = _hadamard_gram_product(phi, phi, values)
            if psi.shape[0]:
                nonlocal_part += _hadamard_gram_product(psi, psi, values) - 2.0 * _hadamard_gram_product(phi, psi, values)
            result = -(self.noise * self.noise) * (local + nonlocal_part)
        self.profile["jvp_seconds"] += time.perf_counter() - started
        self.profile["jvp_columns"] += int(values.shape[1])
        return result[:, 0] if column else result

    def gaussian_posterior(self) -> GaussianPosterior:
        return GaussianPosterior(solve=self.solve, variance_jvp=self.variance_jvp)


# ------------------------------------------------------------------ the EP fixed points (Stage 2's, exact)


def _new_profile() -> dict:
    return {name: 0 for name in (
        "factorizations", "refreshes", "passes", "fixed_point_calls", "solve_columns", "jvp_columns",
    )} | {name: 0.0 for name in ("factor_seconds", "variance_seconds", "solve_seconds", "jvp_seconds", "form_seconds", "tilted_seconds")}


class _DenseFixedPoints:
    """``scale_mixture_ep.FixedPoints`` for one model on a dense training matrix: ``full_data_fit``'s fixed-point
    iteration (refresh, the check, mean-only EP with frozen cavity precisions, the noise update), each quantity exact."""

    def __init__(self, statistics: DenseStatistics, prior: ScaleMixturePrior, draw_count: int, working_bytes: int) -> None:
        self.statistics = statistics
        self.prior = prior
        self.draw_count = int(draw_count)
        self.working_bytes = int(working_bytes)
        self.design = statistics.projected
        self.data_score = self.design.T @ statistics.target
        self.sample_count = statistics.sample_count
        self.covariate_count = int(statistics.covariates.shape[1])
        precision, shift = moment_matched_prior_sites(prior, initial_hyperparameters(prior))
        self.site_precision = precision.copy()
        self.site_shift = shift.copy()
        residual = statistics.projected_target
        self.noise = float(residual @ residual) / (self.sample_count - self.covariate_count)
        self.effective = float(prior.variant_count)
        self.mean_move = np.inf
        self.noise_gain = np.inf
        self.refusals: list[str] = []
        self.profile = _new_profile()
        self.kernel: _Kernel | None = None
        self.mean = np.zeros(prior.variant_count)

    # the Gaussian at the current sites

    def _iterate(self, site_precision: F64Array, site_shift: F64Array) -> None:
        """The exact mean mu = A'^-1 (Xp' y + sigma^2 nu) at the sites; ``LinAlgError`` when A' is not positive definite."""
        started = time.perf_counter()
        kernel = _Kernel(self.design, self.noise * site_precision)
        self.profile["factorizations"] += 1
        self.mean = kernel.solve(self.data_score + self.noise * site_shift)
        self.kernel = kernel
        self.profile["passes"] += 1
        self.profile["factor_seconds"] += time.perf_counter() - started

    def _cavity(self) -> tuple[F64Array, F64Array, F64Array]:
        """(z, 1 - tau z, P = 1/z - tau) at the current sites, from the kernel without cancellation (z = sigma^2 z',
        tau = t / sigma^2, so 1 - tau z = 1 - t z' and P = (1/z' - t) / sigma^2)."""
        started = time.perf_counter()
        variances, removed, precision = self.kernel.cavity()
        self.profile["variance_seconds"] += time.perf_counter() - started
        return self.noise * variances, removed, precision / self.noise

    def _residual_sum_of_squares(self) -> float:
        residual = self.statistics.projected_target - self.design @ self.mean
        return float(residual @ residual)

    def _noise(self, variances: F64Array) -> float:
        return noise_variance(
            residual_sum_of_squares=self._residual_sum_of_squares(), sample_count=self.sample_count, covariate_count=self.covariate_count,
            site_precision=self.site_precision, posterior_variance=variances, noise=self.noise,
        )

    def _refresh(self) -> tuple[F64Array, F64Array]:
        """The mean, the exact marginal variances and the cavity precisions at the current sites; negative sites halve
        while the precision is not positive definite or a cavity is not proper (``full_data_fit``'s refresh)."""
        while True:
            try:
                self._iterate(self.site_precision, self.site_shift)
            except np.linalg.LinAlgError:
                failure = "the precision is not positive definite with non-negative sites"
            else:
                variances, removed, cavity_precision = self._cavity()
                if not np.any(cavity_precision <= 0.0):
                    self.profile["refreshes"] += 1
                    count = self.prior.variant_count
                    # p_eff = sum_j (1 - tau_j z_j), each term without cancellation, floored at its rounding p eps.
                    self.effective = max(float(np.sum(removed)), _EPSILON * count)
                    return variances, cavity_precision
                failure = "a cavity is improper (1/z - tau <= 0) with non-negative sites"
            negative = self.site_precision < 0.0
            if not np.any(negative):
                raise NoFixedPoint(failure)
            self.site_precision[negative] *= 0.5

    def _targets(self, hyperparameters: MixtureHyperparameters, cavity: Cavity) -> tuple[F64Array, F64Array]:
        started = time.perf_counter()
        targets = site_targets(tilted_moments(self.prior, hyperparameters, cavity, self.working_bytes), cavity)
        self.profile["tilted_seconds"] += time.perf_counter() - started
        return targets

    def _precision_norm(self) -> Callable[[F64Array], float]:
        design, precision, noise = self.design, self.site_precision.copy(), self.noise

        def norm(direction: F64Array) -> float:
            values = np.asarray(direction, dtype=np.float64)
            image = design @ values
            return float(image @ image) / noise + float(np.sum(precision * values * values))

        return norm

    def _snapshot(self) -> dict:
        return {
            "site_precision": self.site_precision.copy(), "site_shift": self.site_shift.copy(), "noise": self.noise,
            "effective": self.effective, "kernel": self.kernel, "mean": self.mean.copy(),
        }

    def _restore(self, snapshot: dict) -> None:
        self.site_precision, self.site_shift = snapshot["site_precision"].copy(), snapshot["site_shift"].copy()
        self.noise, self.effective = snapshot["noise"], snapshot["effective"]
        self.kernel, self.mean = snapshot["kernel"], snapshot["mean"].copy()

    def __call__(self, hyperparameters: Sequence[MixtureHyperparameters]) -> list[FixedPoint | None]:
        (model_hyperparameters,) = hyperparameters
        self.profile["fixed_point_calls"] += 1
        snapshot = self._snapshot()
        try:
            return [self._solve(model_hyperparameters)]
        except NoFixedPoint as error:
            self._restore(snapshot)
            self.refusals.append(str(error))
            return [None]

    def _solve(self, hyperparameters: MixtureHyperparameters) -> FixedPoint:
        tolerance = 0.5 / self.draw_count
        while True:
            variances, frozen = self._refresh()
            mean = self.mean.copy()
            cavity = Cavity(precision=frozen, shift=mean / variances - self.site_shift)
            target_precision, target_shift = self._targets(hyperparameters, cavity)
            # The undamped update moves the mean by Sigma (delta nu - delta tau o mu): its squared size in the
            # posterior metric is r' Sigma r, exactly.
            right = (target_shift - self.site_shift) - (target_precision - self.site_precision) * mean
            self.mean_move = float(right @ (self.noise * self.kernel.solve(right)))
            noise = self._noise(variances)
            self.noise_gain = noise_gain(noise, self.noise, self.sample_count, self.covariate_count)
            draw_tolerance = self.effective / self.draw_count
            if self.mean_move <= draw_tolerance and self.noise_gain <= tolerance:
                posterior = _DensePosterior(self.kernel, self.noise, self.working_bytes // 4, self.profile)
                return FixedPoint(
                    cavity=cavity, posterior=posterior.gaussian_posterior(), mean=mean, precision_norm=self._precision_norm(),
                    effective_effects=float(self.effective),
                )
            self._frozen_passes(hyperparameters, frozen, target_precision, target_shift)
            self.noise = self._noise(1.0 / (frozen + self.site_precision))

    def _frozen_passes(self, hyperparameters: MixtureHyperparameters, frozen: F64Array, target_precision: F64Array, target_shift: F64Array) -> None:
        """Mean-only EP with the cavity precisions frozen until the frozen move is below p_eff / K
        (``full_data_fit._FullDataFixedPoints._frozen_passes``)."""
        previous_move, damping = np.inf, 1.0
        while True:
            mean = self.mean.copy()
            fraction = damping
            move = max(float(np.max(np.abs(target_precision - self.site_precision))), float(np.max(np.abs(target_shift - self.site_shift))))
            scale = 1.0 + max(float(np.max(np.abs(self.site_precision))), float(np.max(np.abs(self.site_shift))))
            while True:
                if fraction * move <= _EPSILON * scale:
                    raise NoFixedPoint("no damped EP pass keeps the precision positive definite")
                trial_precision = self.site_precision + fraction * (target_precision - self.site_precision)
                trial_shift = self.site_shift + fraction * (target_shift - self.site_shift)
                try:
                    self._iterate(trial_precision, trial_shift)
                    break
                except np.linalg.LinAlgError:
                    fraction *= 0.5
            self.site_precision, self.site_shift = trial_precision, trial_shift
            marginal = 1.0 / (frozen + self.site_precision)
            mean_move = float(np.sum(np.square(self.mean - mean) / marginal)) / (fraction * fraction)
            if mean_move <= self.effective / self.draw_count:
                return
            if mean_move / previous_move >= 1.0:
                damping = min(damping, 1.0 / (1.0 + np.sqrt(mean_move / previous_move)))
            previous_move = mean_move
            cavity = Cavity(precision=frozen, shift=self.mean / marginal - self.site_shift)
            target_precision, target_shift = self._targets(hyperparameters, cavity)


# ------------------------------------------------------------------ the fit


@dataclass(frozen=True)
class SmallNFit:
    """One model's fit by the small-n route: its scoring model over the input columns (``store_rows`` index the
    columns of the code matrix), noise variance, hyperparameters, certificate and where the time went."""

    scoring: ScoringModel
    noise_variance: float
    hyperparameters: MixtureHyperparameters
    certificate: FitCertificate
    prior: ScaleMixturePrior
    statistics: DenseStatistics
    profile: dict = field(default_factory=dict)


def small_n_prior(statistics: DenseStatistics, variant_class: np.ndarray, log_variance_offset: F64Array, draw_count: int) -> ScaleMixturePrior:
    """The run wiring's prior on the reduced columns (``stage2_wiring._fit_one``): one class per variant class
    present, each representative's log reliability as its offset, no annotation groups, and the start lattice from
    the single-variant likelihoods at the covariate-only residual variance."""
    reduced = statistics.reduced_rows
    _classes, class_index = np.unique(np.asarray(variant_class)[reduced], return_inverse=True)
    offsets = np.asarray(log_variance_offset, dtype=np.float64)[reduced]
    residual = statistics.projected_target
    start_noise = float(residual @ residual) / (statistics.sample_count - statistics.covariates.shape[1])
    design = statistics.projected
    single_precision = np.einsum("ij,ij->j", design, design) / start_noise
    single_shift = (design.T @ statistics.target) / start_noise
    nodes, floor, top = derived_lattice(single_precision, single_shift, offsets, 0.5 / draw_count)
    return scale_mixture_prior(
        class_index=class_index.astype(np.int64), log_variance_offset=offsets, annotation_design=np.zeros((reduced.shape[0], 0)),
        annotation_groups=(), nodes=nodes, floor=floor, top=top,
    )


def fit_small_n(
    *,
    codes: np.ndarray,
    covariates: F64Array,
    target: F64Array,
    variant_class: np.ndarray,
    log_variance_offset: F64Array | None,
    draw_count: int,
    working_bytes: int,
    seed: int,
    trait_type: TraitType = TraitType.QUANTITATIVE,
) -> SmallNFit:
    """Fit one quantitative model on the dense training codes (n x records, store codes) with ``covariates`` (n x k,
    intercept first) and ``target`` (n,): Stage 0 dense, the prior, Stage 2's EP-EB with exact dense algebra, and
    ``draw_count`` exact posterior draws (``seed``)."""
    if trait_type != TraitType.QUANTITATIVE:
        raise NotImplementedError("the small-n route fits quantitative traits (Stage 2 has no binary likelihood yet).")
    started = time.perf_counter()
    statistics = dense_statistics(codes, covariates, target)
    offsets = np.zeros(np.asarray(codes).shape[1]) if log_variance_offset is None else np.asarray(log_variance_offset, dtype=np.float64)
    prior = small_n_prior(statistics, variant_class, offsets, draw_count)
    stage0_seconds = time.perf_counter() - started
    oracle = _DenseFixedPoints(statistics, prior, draw_count, working_bytes)
    tolerance = 0.5 / draw_count
    try:
        (outer,) = fit_hyperparameters(prior, [initial_hyperparameters(prior)], oracle, working_bytes, tolerance)
    except FloatingPointError as error:
        raise FloatingPointError(f"{error}; EP refusals: {oracle.refusals}") from error
    # Draws of N(mu, sigma^2 A'^-1): the kernel's N(0, A'^-1) draws, scaled by sigma, around the mean.
    draws = oracle.mean[:, None] + np.sqrt(oracle.noise) * oracle.kernel.draws(np.random.default_rng(seed), draw_count)
    alpha = statistics.covariate_pseudo_inverse @ (statistics.covariates.T @ statistics.target - statistics.loading @ oracle.mean)
    active_to_reduced = np.asarray(statistics.tie_map.original_to_reduced, dtype=np.int64)
    scoring = ScoringModel.from_reduced_fit(
        active_rows=statistics.active_rows,
        signed_means=statistics.means,
        signed_scales=statistics.scales,
        tie_map=statistics.tie_map,
        member_prior_variances=prior_second_moment(prior, outer.hyperparameters)[active_to_reduced],
        beta_reduced=oracle.mean,
        posterior_draws_reduced=draws,
        alpha=alpha,
        trait_type=trait_type,
        predictive_intercept_shift=0.0,
    )
    certificate = FitCertificate(
        remaining_gain=np.array([outer.remaining_gain]),
        newton_decrement=np.array([outer.newton_decrement]),
        smoothing_gradient=np.array([outer.step.smoothing_gradient]),
        stationarity_steps=(outer.step.stationarity_steps,),
        stationarity_errors=(outer.step.stationarity_errors,),
        mean_move=np.array([oracle.mean_move]),
        draw_tolerance=np.array([oracle.effective / draw_count]),
        noise_gain=np.array([oracle.noise_gain]),
        # Exact algebra: the mean and the variances carry rounding only.
        mean_error=np.zeros(1),
        information_bound=np.zeros(1),
        information_tolerance=np.zeros(1),
        undecided_blocks=0,
        negative_sites=np.array([int(np.sum(oracle.site_precision < 0.0))], dtype=np.int64),
        effective_effects=np.array([oracle.effective]),
        outer_iterations=np.array([outer.iterations], dtype=np.int64),
        halvings=np.array([outer.halvings], dtype=np.int64),
        prediction_move=np.array([outer.prediction_move]),
        prediction_tolerance=np.array([outer.prediction_tolerance]),
        unresolved=np.array([outer.unresolved], dtype=np.int64),
        refusals=tuple(oracle.refusals),
        outer_history=(outer.history,),
        refreshes=int(oracle.profile["refreshes"]),
        passes=int(oracle.profile["passes"]),
    )
    profile = dict(oracle.profile) | {
        "stage0_seconds": stage0_seconds,
        "total_seconds": time.perf_counter() - started,
        "samples": statistics.sample_count,
        "active": int(statistics.active_rows.shape[0]),
        "reduced": int(statistics.projected.shape[1]),
        "coefficients": int(prior.coefficient_size),
        "grid": int(prior.grid_size),
        "classes": int(prior.class_count),
        "outer_iterations": int(outer.iterations),
    }
    return SmallNFit(
        scoring=scoring, noise_variance=float(oracle.noise), hyperparameters=outer.hyperparameters, certificate=certificate, prior=prior,
        statistics=statistics, profile=profile,
    )
