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
- So A'^-1 = Delta - Phi'Phi + Psi'Psi: Delta = diag(1/t) on P and 0 on N, Phi = U^-T Xp_P T^-1 (K = U'U), and
  Psi = [S_U^-T E' | -S_U^-T] (|N| x p, S = S_U'S_U, E = T^-1 Xp_P' K^-1 Xp_N). The marginal variances, the cavities
  (without cancellation), solves, the variance JVP -(Sigma o Sigma) W and exact posterior draws all come from these.

The design is kept as Xp = (I - H_C) G, with H_C = Q Q' and G each reduced column's minor-allele codes over its SD,
signed so that (I - H_C) G equals (I - H_C) X~ exactly (H_C removes every column's constant, since the intercept is a
covariate). G is dense: its carriers are sparse (13-17x fewer kernel flops as sparse products on bench-real chr22
[real]), but scipy's sparse products lose to BLAS-3 on every kernel operation there (Gram 0.27 s vs 0.07 s, kernel
and cavity 0.40 s vs 0.20 s, formed Sigma 2.5 s vs 2.1 s; n = 534, p = 8,916, one core).

The fixed-point iteration and the outer loop are Stage 2's (``full_data_fit._FullDataFixedPoints`` and
``scale_mixture_ep.fit_hyperparameters``), with every certified quantity replaced by its exact value: the mean move
r' Sigma r, the variances diag Sigma, and no cavity-information certificate (the variances are exact).

The genotype side is Stage 0's (``genotype_statistics``) on a dense training matrix: signed codes (code - 127), their
training means and population SDs, every polymorphic record active (no rarity or other threshold filter: SPEC), and
exact ties (equal or negated standardized columns) merged by the same integer test, over the whole window rather than
within an LD block. The variance JVP forms Sigma o Sigma once per fixed point when it fits its memory share (one GEMM
per call after that), else works from the factors at O(n^2 p) per column (see ``_DensePosterior``).
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
    log_scale,
    moment_matched_prior_sites,
    moment_start,
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
# ``fit_hyperparameters`` holds each model's current fixed point and the trial's: two posteriors are alive at once.
_LIVE_FIXED_POINTS = 2
# A posterior on the exact response keeps two p x p matrices: Sigma o Sigma and the response's LU factor.
_EXACT_RESPONSE_MATRICES = 2


# ------------------------------------------------------------------ the design


class _Design:
    """Xp = P G with P = I - Q Q' (Q an orthonormal basis of the covariates' columns) and G ``carriers`` (n x p, dense,
    Fortran-ordered)."""

    def __init__(self, carriers: F64Array, basis: F64Array) -> None:
        self.carriers = np.asfortranarray(carriers, dtype=np.float64)
        self.basis = basis
        self.sample_count, self.variant_count = (int(size) for size in carriers.shape)

    @classmethod
    def dense(cls, projected: F64Array) -> _Design:
        """An already projected dense design (no covariates left to project)."""
        return cls(np.asfortranarray(projected, dtype=np.float64), np.zeros((projected.shape[0], 0)))

    def project(self, samples: F64Array) -> F64Array:
        return samples - self.basis @ (self.basis.T @ samples) if self.basis.shape[1] else samples

    def project_both(self, matrix: F64Array) -> F64Array:
        """P M P for a symmetric (n x n) M."""
        if not self.basis.shape[1]:
            return matrix
        left = self.basis.T @ matrix
        return matrix - self.basis @ left - left.T @ self.basis.T + self.basis @ (left @ self.basis) @ self.basis.T

    def image(self, values: F64Array) -> F64Array:
        """Xp v for v (p,) or (p, q)."""
        return self.project(self.carriers @ values)

    def back(self, samples: F64Array) -> F64Array:
        """Xp' u for u (n,) or (n, q)."""
        return self.carriers.T @ self.project(samples)

    def weighted_gram(self, weights: F64Array) -> F64Array:
        """Xp diag(w) Xp' (n x n) for w >= 0."""
        gram = linalg.blas.dsyrk(1.0, self.carriers * np.sqrt(weights)[None, :])
        return self.project_both(np.triu(gram) + np.triu(gram, 1).T)

    def quadratic_diagonal(self, core: F64Array) -> F64Array:
        """x_j' core x_j for every column, core (n x n) symmetric with P core P = core."""
        return np.einsum("ij,ij->j", self.carriers, core @ self.carriers)

    def quadratic(self, core: F64Array, columns: I64Array) -> F64Array:
        """Xp_c' core Xp_c (|c| x |c|) for core (n x n) with P core P = core."""
        block = self.carriers[:, columns]
        return block.T @ (core @ block)

    def columns(self, columns: I64Array) -> F64Array:
        """Xp's columns ``columns`` (n x |c|, F-ordered)."""
        return np.asfortranarray(self.project(self.carriers[:, columns]))

    def column_squares(self) -> F64Array:
        """||x_j||^2 = ||g_j||^2 - ||Q' g_j||^2."""
        squares = np.einsum("ij,ij->j", self.carriers, self.carriers)
        if not self.basis.shape[1]:
            return squares
        loading = self.carriers.T @ self.basis
        return squares - np.einsum("jk,jk->j", loading, loading)


# ------------------------------------------------------------------ the genotype side (Stage 0, dense)


@dataclass(frozen=True)
class DenseStatistics:
    """Stage 0 of one training set on a dense code matrix.

    ``active_rows`` are the input columns with training variance (the store rows of ``fast_scoring``), ``means`` and
    ``scales`` their signed-code means and population SDs, ``tie_map`` their ties; ``design`` is Xp = (I - H_C) X~ of
    the reduced columns, ``loading`` = C' X~ (k x reduced), and ``projected_target`` = (I - H_C) y.
    """

    active_rows: I64Array
    means: F64Array
    scales: F64Array
    tie_map: TieMap
    design: _Design
    loading: F64Array
    covariates: F64Array
    covariate_pseudo_inverse: F64Array
    target: F64Array
    projected_target: F64Array

    @property
    def sample_count(self) -> int:
        return self.design.sample_count

    @property
    def reduced_rows(self) -> I64Array:
        """The input columns of the reduced columns (each tie group's representative)."""
        return self.active_rows[np.asarray(self.tie_map.kept_indices, dtype=np.int64)]

    @property
    def projected(self) -> F64Array:
        """Xp, dense (n x reduced)."""
        return self.design.columns(np.arange(self.design.variant_count))


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


def _covariate_basis(covariates: F64Array) -> F64Array:
    """An orthonormal basis of the covariates' numerical column space, by Stage 0's rank rule (``max(n, k) eps s_max``,
    as ``_covariate_gram_pseudo_inverse``)."""
    if not covariates.shape[1]:
        return np.zeros((covariates.shape[0], 0))
    left, singular, _right = np.linalg.svd(covariates, full_matrices=False)
    return left[:, singular > max(covariates.shape) * _EPSILON * singular[0]]


def dense_statistics(codes: np.ndarray, covariates: F64Array, target: F64Array) -> DenseStatistics:
    """Stage 0 of ``codes`` (n x records, store codes 0..254 = dosage x 127) with ``covariates`` (n x k, intercept
    included) and ``target`` (n,)."""
    values = np.asarray(codes)
    full = int(2 * SIGNED_CODE_OFFSET)
    if values.ndim != 2 or values.min(initial=0) < 0 or values.max(initial=0) > full:
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
    # G: each reduced column's codes from its minor end (codes of the fewer carriers), over its SD, signed so that
    # code_j = offset_j + scale_j g_j; P g_j = P X~_j because P removes the constant (the intercept is a covariate).
    reduced_codes = signed[:, kept] + int(SIGNED_CODE_OFFSET)
    flip = np.count_nonzero(reduced_codes == full, axis=0) > np.count_nonzero(reduced_codes == 0, axis=0)
    minor = np.where(flip[None, :], full - reduced_codes, reduced_codes)
    column_factor = np.where(flip, -1.0, 1.0) / scales[kept]
    offsets = np.where(flip, full, 0.0)
    covariate_matrix = np.asarray(covariates, dtype=np.float64)
    design = _Design(minor * column_factor[None, :], _covariate_basis(covariate_matrix))
    # C' X~_j = C' g_j + C'1 (offset_j - 127 - mean_j) / scale_j.
    constants = (offsets - SIGNED_CODE_OFFSET - means[kept]) / scales[kept]
    loading = (design.carriers.T @ covariate_matrix).T + np.outer(covariate_matrix.sum(axis=0), constants)
    target_values = np.asarray(target, dtype=np.float64)
    return DenseStatistics(
        active_rows=active.astype(np.int64),
        means=means,
        scales=scales,
        tie_map=tie_map,
        design=design,
        loading=loading,
        covariates=covariate_matrix,
        covariate_pseudo_inverse=_covariate_gram_pseudo_inverse(covariate_matrix),
        target=target_values,
        projected_target=design.project(target_values),
    )


# ------------------------------------------------------------------ q(beta) in kernel form


class _Kernel:
    """A' = Xp' Xp + diag t at the scaled sites t (see the module docstring), factored once.

    ``solve(R)`` is A'^-1 R, ``cavity()`` the variances, removed shares and cavity precisions, ``covariance()``
    A'^-1 formed, and ``factors()`` (Delta, Phi, Psi) with A'^-1 = Delta - Phi'Phi + Psi'Psi. Raises
    ``np.linalg.LinAlgError`` when A' is not positive definite.
    """

    def __init__(self, design: _Design, scaled_precision: F64Array) -> None:
        self.design = design
        self.precision = scaled_precision
        self.variant_count = design.variant_count
        positive = scaled_precision > 0.0
        self.bulk = np.flatnonzero(positive)
        self.rest = np.flatnonzero(~positive)
        self.inverse = 1.0 / scaled_precision[self.bulk]
        weights = np.zeros(self.variant_count)
        weights[self.bulk] = self.inverse
        kernel = design.weighted_gram(weights)
        kernel[np.diag_indices_from(kernel)] += 1.0
        self.upper = linalg.cholesky(kernel, lower=False, check_finite=False, overwrite_a=True)
        self._core: F64Array | None = None
        self._rest_factors: tuple[F64Array, F64Array] | None = None
        self._factors: tuple[F64Array, F64Array, F64Array] | None = None
        if self.rest.size:
            self.rest_design = design.columns(self.rest)
            self.rest_whitened = linalg.solve_triangular(self.upper, self.rest_design, trans="T", lower=False, check_finite=False)
            schur = self.rest_whitened.T @ self.rest_whitened
            schur[np.diag_indices_from(schur)] += scaled_precision[self.rest]
            self.schur_upper = linalg.cholesky(schur, lower=False, check_finite=False)

    def _kernel_solve(self, values: F64Array) -> F64Array:
        return linalg.cho_solve((self.upper, False), values, check_finite=False)

    def _expand(self, bulk_values: F64Array) -> F64Array:
        full = np.zeros((self.variant_count,) + bulk_values.shape[1:])
        full[self.bulk] = bulk_values
        return full

    def _bulk_back(self, samples: F64Array) -> F64Array:
        """Xp_P' u."""
        return self.design.back(samples)[self.bulk]

    def _bulk_apply(self, values: F64Array) -> tuple[F64Array, F64Array]:
        """(A'_PP^-1 v, Xp_P T^-1 v) for v (|P| x q)."""
        scaled = self.inverse[:, None] * values
        image = self.design.image(self._expand(scaled))
        return scaled - self.inverse[:, None] * self._bulk_back(self._kernel_solve(image)), image

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
            correction, _ = self._bulk_apply(self._bulk_back(self.rest_design @ rest_part))
            bulk_part = bulk_part - correction
            solution[self.rest] = rest_part
        solution[self.bulk] = bulk_part
        return solution[:, 0] if column else solution

    def core(self) -> F64Array:
        """P K^-1 P (n x n), so x_j' K^-1 x_k = g_j' (P K^-1 P) g_k."""
        if self._core is None:
            self._core = self.design.project_both(self._kernel_solve(np.eye(self.upper.shape[0])))
        return self._core

    def rest_factors(self) -> tuple[F64Array, F64Array]:
        """(Psi_P = S_U^-T E' (|N| x |P|), Psi_N = -S_U^-T (|N| x |N|)), E' = Xp_N' K^-1 Xp_P T^-1."""
        if self._rest_factors is None:
            cross = self._bulk_back(self._kernel_solve(self.rest_design)).T * self.inverse[None, :]
            psi_bulk = linalg.solve_triangular(self.schur_upper, cross, trans="T", lower=False, check_finite=False)
            psi_rest = -linalg.solve_triangular(self.schur_upper, np.eye(self.rest.size), trans="T", lower=False, check_finite=False)
            self._rest_factors = (psi_bulk, psi_rest)
        return self._rest_factors

    def cavity(self) -> tuple[F64Array, F64Array, F64Array]:
        """(z', 1 - t z', 1/z' - t): the variances of A'^-1, each site's share of its variance the data remove, and its
        cavity precision, without cancellation. For a bulk column z' = d - d^2 q + f (d = 1/t, q = x_j' K^-1 x_j,
        f = ||Psi_j||^2), so 1 - t z' = d q - t f exactly; forming 1/z' - t from z' instead loses every digit where the
        data inform a column far less than its site does (d q below eps), which is most rare variants at the start."""
        quadratic = self.design.quadratic_diagonal(self.core())[self.bulk]
        variances = np.empty(self.variant_count)
        removed = np.empty(self.variant_count)
        extra = np.zeros(self.bulk.size)
        if self.rest.size:
            psi_bulk, psi_rest = self.rest_factors()
            extra = np.einsum("ij,ij->j", psi_bulk, psi_bulk)
            variances[self.rest] = np.einsum("ij,ij->j", psi_rest, psi_rest)
            removed[self.rest] = 1.0 - self.precision[self.rest] * variances[self.rest]
        variances[self.bulk] = self.inverse - self.inverse * self.inverse * quadratic + extra
        removed[self.bulk] = self.inverse * quadratic - self.precision[self.bulk] * extra
        return variances, removed, removed / variances

    def variances(self) -> F64Array:
        return self.cavity()[0]

    def covariance(self) -> F64Array:
        """A'^-1 formed (p x p): Delta - T^-1 Xp_P' K^-1 Xp_P T^-1 on P, plus Psi'Psi."""
        bulk_block = self.design.quadratic(self.core(), self.bulk)
        bulk_block *= -self.inverse[:, None]
        bulk_block *= self.inverse[None, :]
        bulk_block[np.diag_indices_from(bulk_block)] += self.inverse
        if not self.rest.size:
            return bulk_block
        covariance = np.zeros((self.variant_count, self.variant_count))
        covariance[np.ix_(self.bulk, self.bulk)] = bulk_block
        psi_bulk, psi_rest = self.rest_factors()
        psi = np.zeros((self.rest.size, self.variant_count))
        psi[:, self.bulk] = psi_bulk
        psi[:, self.rest] = psi_rest
        covariance += psi.T @ psi
        return covariance

    def factors(self) -> tuple[F64Array, F64Array, F64Array]:
        """(delta (p,), Phi (n x p), Psi (|N| x p)) with A'^-1 = diag(delta) - Phi'Phi + Psi'Psi (the O(n^2 p) route of
        the variance JVP, when Sigma o Sigma is too large to form)."""
        if self._factors is not None:
            return self._factors
        delta = np.zeros(self.variant_count)
        delta[self.bulk] = self.inverse
        phi = np.zeros((self.upper.shape[0], self.variant_count), order="F")
        whitened = linalg.solve_triangular(self.upper, self.design.columns(self.bulk), trans="T", lower=False, check_finite=False)
        phi[:, self.bulk] = whitened * self.inverse[None, :]
        psi = np.zeros((self.rest.size, self.variant_count))
        if self.rest.size:
            psi_bulk, psi_rest = self.rest_factors()
            psi[:, self.bulk] = psi_bulk
            psi[:, self.rest] = psi_rest
        self._factors = (delta, phi, psi)
        return self._factors

    def draws(self, generator: np.random.Generator, draw_count: int) -> F64Array:
        """(p x draws) exact draws of N(0, A'^-1): beta_N from its marginal N(0, S^-1), then beta_P | beta_N with
        precision A'_PP (Bhattacharya et al. 2016: u ~ N(0, T^-1), e ~ N(0, I_n), u - T^-1 Xp_P' K^-1 (Xp_P u + e))."""
        draws = np.zeros((self.variant_count, draw_count))
        prior_noise = generator.standard_normal((self.bulk.size, draw_count))
        sample_noise = generator.standard_normal((self.upper.shape[0], draw_count))
        spread = np.sqrt(self.inverse)[:, None] * prior_noise
        image = self.design.image(self._expand(spread)) + sample_noise
        bulk_draw = spread - self.inverse[:, None] * self._bulk_back(self._kernel_solve(image))
        if self.rest.size:
            rest_noise = generator.standard_normal((self.rest.size, draw_count))
            rest_offset = linalg.solve_triangular(self.schur_upper, rest_noise, lower=False, check_finite=False)
            draws[self.rest] += rest_offset
            # E (beta_N - mean_N) with E = A'_PP^-1 A'_PN = A'_PP^-1 Xp_P' Xp_N.
            conditional, _ = self._bulk_apply(self._bulk_back(self.rest_design @ rest_offset))
            bulk_draw = bulk_draw - conditional
        draws[self.bulk] += bulk_draw
        return draws


def _hadamard_gram_product(left: F64Array, right: F64Array, weights: F64Array) -> F64Array:
    """((L'L) o (R'R)) W, column by column: out_jc = l_j' (L diag(w_c) R') r_j."""
    result = np.empty_like(weights)
    for column in range(weights.shape[1]):
        core = (left * weights[:, column][None, :]) @ right.T
        result[:, column] = np.einsum("ij,ij->j", left, core @ right)
    return result


class _DensePosterior:
    """q's responses at one fixed point (``scale_mixture_ep.GaussianPosterior``): Sigma R exactly, and
    -(Sigma o Sigma) W exactly. Sigma o Sigma is formed once (8 p^2 bytes) when that fits ``jvp_bytes``, and each call
    is then one GEMM; otherwise each column costs O(n^2 p) from the factors. Forming it pays as soon as a call has more
    than p / (2 n) columns, which the total curvature's (p x D) GMRES iterates always have, so memory alone decides."""

    def __init__(self, kernel: _Kernel, noise: float, jvp_bytes: int, profile: dict) -> None:
        self.kernel = kernel
        self.noise = float(noise)
        self.jvp_bytes = int(jvp_bytes)
        self.profile = profile
        self._squared: F64Array | None = None
        self._response_key: tuple | None = None
        self._response_factor: tuple | None = None

    def solve(self, right: F64Array, _relative_tolerance: float) -> F64Array:
        started = time.perf_counter()
        result = self.noise * self.kernel.solve(right)
        self.profile["solve_seconds"] += time.perf_counter() - started
        self.profile["solve_columns"] += int(np.asarray(right).shape[1]) if np.ndim(right) == 2 else 1
        return result

    def _explicit(self) -> F64Array:
        if self._squared is None:
            started = time.perf_counter()
            covariance = self.kernel.covariance()
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

    def linear_response(self, left: F64Array, right: F64Array, diagonal: F64Array, weight: F64Array, rhs: F64Array) -> F64Array:
        """The exact X of (I - (I - diag(w) S2) M) X = B, S2 = Sigma o Sigma and M = diag(left) Sigma diag(right) +
        diag(diagonal) (``scale_mixture_ep.GaussianPosterior``), by one LU of that p x p matrix.

        Sigma = sigma^2 (diag(delta) - Phi'Phi + Psi'Psi) makes M = diag(d0) + U V' with rank r = n + |N|, so the matrix
        diag(1 - d0) - U V' + diag(w) (S2 diag(d0) + (S2 U) V') is formed in 3 p^2 r flops, and the LU costs 2 p^3 / 3,
        whatever the number of directions. The coefficients depend only on the fixed point and its hyperparameters,
        which one correction holds for every view it is asked (the lazy correction extends its directions as edges
        release), so the factor is kept and each later call costs 2 p^2 per direction."""
        started = time.perf_counter()
        key = (left, right, diagonal, weight)
        if self._response_key is None or not all(np.array_equal(a, b) for a, b in zip(self._response_key, key)):
            # The matrix is C-ordered: its transpose is the same buffer in Fortran order, which LAPACK factors in
            # place (a C-ordered one it would copy, a third p x p), and the transposed solve then gives M X = B.
            matrix = self._response_matrix(left, right, diagonal, weight)
            self._response_factor = linalg.lu_factor(matrix.T, overwrite_a=True, check_finite=False)
            self._response_key = tuple(np.array(value, copy=True) for value in key)
            self.profile["response_factorizations"] += 1
        solution = linalg.lu_solve(self._response_factor, np.asarray(rhs, dtype=np.float64), trans=1, check_finite=False)
        self.profile["response_seconds"] += time.perf_counter() - started
        self.profile["responses"] += 1
        return solution

    def _response_matrix(self, left: F64Array, right: F64Array, diagonal: F64Array, weight: F64Array) -> F64Array:
        squared = self._explicit()
        delta, phi, psi = self.kernel.factors()
        noise = self.noise
        d0 = diagonal + noise * left * delta * right
        factor_rows = np.vstack([phi, psi]) if psi.shape[0] else phi
        signs = np.concatenate([-np.ones(phi.shape[0]), np.ones(psi.shape[0])])
        low = (noise * left)[:, None] * (factor_rows.T * signs[None, :])  # U (p x r)
        high = factor_rows * right[None, :]  # V' (r x p)
        squared_low = squared @ low
        matrix = squared * d0[None, :]
        # Row panels of r rows keep every temporary the size of the factors.
        step = max(1, high.shape[0])
        for start in range(0, matrix.shape[0], step):
            rows = slice(start, min(start + step, matrix.shape[0]))
            matrix[rows] += squared_low[rows] @ high
            matrix[rows] *= weight[rows, None]
            matrix[rows] -= low[rows] @ high
        matrix[np.diag_indices_from(matrix)] += 1.0 - d0
        return matrix

    def gaussian_posterior(self) -> GaussianPosterior:
        """The responses; the exact linear response when this posterior's S2 and response factor fit its share."""
        p = self.kernel.variant_count
        exact = _EXACT_RESPONSE_MATRICES * _FLOAT_BYTES * p * p <= self.jvp_bytes
        return GaussianPosterior(solve=self.solve, variance_jvp=self.variance_jvp, linear_response=self.linear_response if exact else None)


# ------------------------------------------------------------------ the EP fixed points (Stage 2's, exact)


def _new_profile() -> dict:
    return {name: 0 for name in (
        "factorizations", "refreshes", "passes", "fixed_point_calls", "solve_columns", "jvp_columns", "responses", "response_factorizations",
    )} | {name: 0.0 for name in (
        "factor_seconds", "variance_seconds", "solve_seconds", "jvp_seconds", "form_seconds", "tilted_seconds", "response_seconds",
    )}


class _DenseFixedPoints:
    """``scale_mixture_ep.FixedPoints`` for one model on a dense training matrix: ``full_data_fit``'s fixed-point
    iteration (refresh, the check, mean-only EP with frozen cavity precisions, the noise update), each quantity exact."""

    def __init__(
        self, statistics: DenseStatistics, prior: ScaleMixturePrior, start: MixtureHyperparameters, start_noise: float, draw_count: int, working_bytes: int
    ) -> None:
        self.statistics = statistics
        self.prior = prior
        self.draw_count = int(draw_count)
        self.working_bytes = int(working_bytes)
        self.design = statistics.design
        self.data_score = self.design.back(statistics.target)
        self.sample_count = statistics.sample_count
        self.covariate_count = int(statistics.covariates.shape[1])
        precision, shift = moment_matched_prior_sites(prior, start)
        self.site_precision = precision.copy()
        self.site_shift = shift.copy()
        self.noise = float(start_noise)
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
        residual = self.statistics.projected_target - self.design.image(self.mean)
        return float(residual @ residual)

    def _noise(self, variances: F64Array) -> float:
        return noise_variance(
            residual_sum_of_squares=self._residual_sum_of_squares(), sample_count=self.sample_count, covariate_count=self.covariate_count,
            site_precision=self.site_precision, posterior_variance=variances, noise=self.noise,
        )

    def _largest_variances(self, hyperparameters: MixtureHyperparameters) -> F64Array:
        """Each variant's largest prior variance on the lattice, u_j exp(t_K): its tilted law is proper exactly when
        1 + v P > 0 at every node (``scale_mixture_ep._kernel_terms``'s own test), i.e. when its cavity precision
        exceeds -1 / this."""
        return np.exp(log_scale(self.prior, hyperparameters.coefficients) + self.prior.log_variance_grid[-1])

    def _refresh(self, hyperparameters: MixtureHyperparameters) -> tuple[F64Array, F64Array]:
        """The mean, the exact marginal variances and the cavity precisions at the current sites; negative sites halve
        while the precision is not positive definite or a cavity's tilted law is not proper (``full_data_fit``'s
        refresh).

        A cavity is proper when its tilted law is, 1 + v P > 0 at every lattice node, which admits a negative cavity
        precision down to -1 / v_max. ``full_data_fit`` asks P > 0 instead, which EP cannot reach where a column is
        an exact linear combination of columns whose sites are negative (a doubleton is the sum of two singletons):
        its cavity is then proportional to those sites, negative for every negative value, and exactly 0 only when
        they underflow, so halving runs until they do (bench-real chr22 gene 1 [real]: 7 negative sites halved past
        1e-250 over 870 refactorizations while one column's cavity stayed at -9e-16, its rounding of 0)."""
        largest = self._largest_variances(hyperparameters)
        while True:
            try:
                self._iterate(self.site_precision, self.site_shift)
            except np.linalg.LinAlgError:
                failure = "the precision is not positive definite with non-negative sites"
            else:
                variances, removed, cavity_precision = self._cavity()
                if np.all(1.0 + largest * cavity_precision > 0.0):
                    self.profile["refreshes"] += 1
                    count = self.prior.variant_count
                    # p_eff = sum_j (1 - tau_j z_j), each term without cancellation, floored at its rounding p eps.
                    self.effective = max(float(np.sum(removed)), _EPSILON * count)
                    return variances, cavity_precision
                failure = "a cavity's tilted law is improper (1 + v P <= 0 on the lattice) with non-negative sites"
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
            image = design.image(values)
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
            variances, frozen = self._refresh(hyperparameters)
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
                # Each fixed point alive at once (the outer loop holds the current one and one trial) gets an equal share
                # of the working memory for its posterior's p x p matrices. The exact response replaces the curvature's
                # GMRES, whose memory it takes; where it does not fit, GMRES has half (``fit_small_n``).
                posterior = _DensePosterior(self.kernel, self.noise, self.working_bytes // _LIVE_FIXED_POINTS, self.profile)
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
    single_precision = statistics.design.column_squares() / start_noise
    single_shift = statistics.design.back(statistics.target) / start_noise
    nodes, floor, top = derived_lattice(single_precision, single_shift, offsets, 0.5 / draw_count)
    return scale_mixture_prior(
        class_index=class_index.astype(np.int64), log_variance_offset=offsets, annotation_design=np.zeros((reduced.shape[0], 0)),
        annotation_groups=(), nodes=nodes, floor=floor, top=top,
    )


def small_n_start(statistics: DenseStatistics, prior: ScaleMixturePrior) -> tuple[MixtureHyperparameters, float, object]:
    """The EB start from Haseman-Elston moments (``scale_mixture_ep.moment_start``), exact from the n x n kernel
    K0 = Xp Xp': tr G = tr K0, sum_j u_j ||G e_j||^2 = sum_j u_j x_j' K0 x_j and ||G||_F^2 = tr(K0^2), one n^2 p pass."""
    design = statistics.design
    residual = statistics.projected_target
    weights = np.exp(prior.log_variance_offset)
    kernel = design.weighted_gram(np.ones(design.variant_count))
    squares = design.column_squares()
    score = design.back(statistics.target)
    moment = moment_start(
        target_square=float(residual @ residual),
        residual_dimension=float(statistics.sample_count - statistics.covariates.shape[1]),
        score_square=float(score @ score),
        gram_trace=float(np.trace(kernel)),
        weighted_diagonal=float(weights @ squares),
        weighted_square=float(weights @ design.quadratic_diagonal(kernel)),
        gram_square=float(np.sum(kernel * kernel)),
    )
    return initial_hyperparameters(prior, moment.mean_variance), float(moment.noise), moment


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
    start, start_noise, moment = small_n_start(statistics, prior)
    oracle = _DenseFixedPoints(statistics, prior, start, start_noise, draw_count, working_bytes)
    tolerance = 0.5 / draw_count
    try:
        (outer,) = fit_hyperparameters(prior, [start], oracle, working_bytes // 2, tolerance)
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
        "reduced": int(statistics.design.variant_count),
        "coefficients": int(prior.coefficient_size),
        "grid": int(prior.grid_size),
        "classes": int(prior.class_count),
        "outer_iterations": int(outer.iterations),
        "start_heritability": float(moment.heritability),
        "start_resolution": float(moment.resolution),
    }
    return SmallNFit(
        scoring=scoring, noise_variance=float(oracle.noise), hyperparameters=outer.hyperparameters, certificate=certificate, prior=prior,
        statistics=statistics, profile=profile,
    )
