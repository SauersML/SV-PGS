"""The small-n route: Stage 2's model (``full_data_fit``) with q(beta)'s algebra in the n x n kernel form.

Where the training samples are far fewer than the columns (a cis window: n ~ 600, p ~ 10^4), every quantity the EP-EB
fit asks of q(beta) = N(mu, Sigma) is exact from one n x n Cholesky factor per site update, so none needs a Krylov
solve or a probe certificate:

- q's precision is A = Xp' Xp / sigma^2 + diag tau with Xp = (I - H_C) X~ on the training rows. With the scaled sites
  t = sigma^2 tau it is A = A' / sigma^2, A' = Xp' Xp + diag t, and Sigma = sigma^2 A'^-1.
- The columns split into the bulk P (t >= ||x_j||^2, where Woodbury keeps every digit) and the rest N (every other
  site, negative ones included: EP is unclipped). On P, Woodbury gives
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
exact ties (equal or negated standardized columns) found by the same integer test, over the whole window rather than
within an LD block. Tie members keep their own effects, sites and class priors (review-mathbugs T1, lead ruling;
``tie_members.TieGroups`` is the shared representation): only ``_Design`` aggregates them, inside the kernel, so the
kernel costs what the tie groups do, and the posterior's p x p work runs on the units of members that share a group
and a site (``_Kernel.units``). The variance JVP forms Sigma o Sigma once per fixed point when it fits its memory share (one GEMM
per call after that), else works from the factors at O(n^2 p) per column (see ``_DensePosterior``).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, Sequence

import numpy as np
from scipy import linalg, sparse

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
    _components,
    class_log_density,
    initial_hyperparameters,
    log_scale,
    moment_matched_prior_sites,
    moment_start,
    noise_gain,
    noise_variance,
    prior_second_moment,
    scale_mixture_prior,
    site_targets,
    tilted_cumulants,
    tilted_moments,
)
from sv_pgs.tie_map import _compact_identity_tie_map, tie_map_from_groups
from sv_pgs.tie_members import TieGroups

_EPSILON = float(np.finfo(np.float64).eps)
_FLOAT_BYTES = np.dtype(np.float64).itemsize
# ``fit_hyperparameters`` holds each model's current fixed point and the trial's: two posteriors are alive at once.
_LIVE_FIXED_POINTS = 2
# A posterior on the exact response keeps two p x p matrices: Sigma o Sigma and the response's LU factor.
_EXACT_RESPONSE_MATRICES = 2


# ------------------------------------------------------------------ the design


class _Design:
    """Xp over the model's effects (review-mathbugs T1, lead ruling): exact-tie members keep their own effects, so the
    columns are the members, in oriented coordinates (member j's column is its group's x_g, its effect gamma_j = s_j
    beta_j, whose prior is beta_j's: every prior is symmetric). Duplicates are aggregated only here: Xp = P G A' with
    P = I - Q Q' (Q an orthonormal basis of the covariates' columns), G ``carriers`` (n x groups, dense, Fortran-ordered)
    and A the members' group indicator (``members[j]`` is member j's column of G; None: one member per column). So a
    member-level product costs a group-level one plus O(members).

    ``codes`` is (the groups' minor-allele codes, uint8 n x groups; each group's factor, G = codes * factor), when known."""

    def __init__(self, carriers: F64Array, basis: F64Array, members: I64Array | None = None, codes: tuple[np.ndarray, F64Array] | None = None) -> None:
        self.carriers = np.asfortranarray(carriers, dtype=np.float64)
        self.basis = basis
        self.codes = codes
        self.sample_count, self.group_count = (int(size) for size in carriers.shape)
        self.members = np.arange(self.group_count, dtype=np.int64) if members is None else np.asarray(members, dtype=np.int64)
        self.variant_count = int(self.members.shape[0])
        self.tied = not np.array_equal(self.members, np.arange(self.group_count))
        self._squares: F64Array | None = None
        self._sum = sparse.csr_matrix(
            (np.ones(self.variant_count), (self.members, np.arange(self.variant_count))), shape=(self.group_count, self.variant_count)
        ) if self.tied else None

    @classmethod
    def dense(cls, projected: F64Array) -> _Design:
        """An already projected dense design (no covariates left to project)."""
        return cls(np.asfortranarray(projected, dtype=np.float64), np.zeros((projected.shape[0], 0)))

    def group_sum(self, values: F64Array) -> F64Array:
        """A' v: each group's sum over its members, (groups,) or (groups, q)."""
        return np.asarray(self._sum @ values) if self.tied else values

    def spread(self, group_values: F64Array) -> F64Array:
        """A u: each member's group's value."""
        return group_values[self.members] if self.tied else group_values

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
        return self.project(self.carriers @ self.group_sum(values))

    def back(self, samples: F64Array) -> F64Array:
        """Xp' u for u (n,) or (n, q)."""
        return self.spread(self.carriers.T @ self.project(samples))

    def weighted_gram(self, weights: F64Array) -> F64Array:
        """Xp diag(w) Xp' (n x n) for w >= 0: G diag(A' w) G', the members' weights summed within their group."""
        gram = linalg.blas.dsyrk(1.0, self.carriers * np.sqrt(self.group_sum(weights))[None, :])
        return self.project_both(np.triu(gram) + np.triu(gram, 1).T)

    def group_quadratic_diagonal(self, core: F64Array) -> F64Array:
        """x_g' core x_g for every group, core (n x n) symmetric with P core P = core."""
        return np.einsum("ij,ij->j", self.carriers, core @ self.carriers)

    def quadratic_diagonal(self, core: F64Array) -> F64Array:
        """x_j' core x_j for every member."""
        return self.spread(self.group_quadratic_diagonal(core))

    def quadratic(self, core: F64Array, columns: I64Array) -> F64Array:
        """Xp_c' core Xp_c (|c| x |c|) for core (n x n) with P core P = core."""
        block = self.carriers[:, self.members[columns]]
        return block.T @ (core @ block)

    def group_columns(self, groups: I64Array) -> F64Array:
        """The groups' columns x_g (n x |g|, F-ordered)."""
        return np.asfortranarray(self.project(self.carriers[:, groups]))

    def columns(self, columns: I64Array) -> F64Array:
        """Xp's member columns ``columns`` (n x |c|, F-ordered)."""
        return self.group_columns(self.members[columns])

    @property
    def squares(self) -> F64Array:
        """``column_squares()``, computed once."""
        if self._squares is None:
            self._squares = self.column_squares()
        return self._squares

    def column_squares(self) -> F64Array:
        """||x_j||^2 = ||g_j||^2 - ||Q' g_j||^2."""
        squares = np.einsum("ij,ij->j", self.carriers, self.carriers)
        if self.basis.shape[1]:
            loading = self.carriers.T @ self.basis
            squares = squares - np.einsum("jk,jk->j", loading, loading)
        return self.spread(squares)


# ------------------------------------------------------------------ the genotype side (Stage 0, dense)


@dataclass(frozen=True)
class DenseStatistics:
    """Stage 0 of one training set on a dense code matrix.

    ``active_rows`` are the input columns with training variance (the store rows of ``fast_scoring``), ``means`` and
    ``scales`` their signed-code means and population SDs, ``tie_map`` their exact ties and ``ties`` the same as
    ``tie_members.TieGroups`` (each member's group and sign). Every effect is a member's (review-mathbugs T1): ``design``
    is Xp = (I - H_C) X~ over the members in oriented coordinates (``_Design``; member j's effect gamma_j = s_j beta_j),
    ``loading`` = C' X~ in the same coordinates (k x members), and ``projected_target`` = (I - H_C) y.
    """

    active_rows: I64Array
    means: F64Array
    scales: F64Array
    tie_map: TieMap
    ties: TieGroups
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
    def signs(self) -> F64Array:
        """Each member's sign: its standardized column is s_j x_g, and beta_j = s_j gamma_j."""
        return self.ties.sign

    @property
    def projected(self) -> F64Array:
        """Xp over the members, dense (n x members, oriented)."""
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
    ties = TieGroups.from_tie_map(tie_map)
    design = _Design(minor * column_factor[None, :], _covariate_basis(covariate_matrix), members=ties.group, codes=(minor.astype(np.uint8), column_factor))
    # C' X~_g = C' g_g + C'1 (offset_g - 127 - mean_g) / scale_g for each group's representative; a member's, in its
    # oriented coordinates, is its group's.
    constants = (offsets - SIGNED_CODE_OFFSET - means[kept]) / scales[kept]
    loading = ((design.carriers.T @ covariate_matrix).T + np.outer(covariate_matrix.sum(axis=0), constants))[:, ties.group]
    target_values = np.asarray(target, dtype=np.float64)
    return DenseStatistics(
        active_rows=active.astype(np.int64),
        means=means,
        scales=scales,
        tie_map=tie_map,
        ties=ties,
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
        # A positive site's variance by Woodbury is d (1 - d q) + f (d = 1/t, q = x_j' K^-1 x_j): 1 - d q carries d q's
        # rounding amplified by d q / (1 - d q) = x_j' K_-j^-1 x_j / t_j <= ||x_j||^2 / t_j (K_-j >= I). So a site stays
        # in the bulk only where that amplification is at most 1, t_j >= ||x_j||^2; every other one goes by the Schur
        # route, which never inverts t and is exact for any sign (review-mathbugs K1: at t_j = 1e-8 ||x_j||^2 the
        # Woodbury variance was 770% off an exact rational reference).
        bulk_mask = (scaled_precision > 0.0) & (scaled_precision >= design.squares)
        self.bulk = np.flatnonzero(bulk_mask)
        self.rest = np.flatnonzero(~bulk_mask)
        self.inverse = 1.0 / scaled_precision[self.bulk]
        weights = np.zeros(self.variant_count)
        weights[self.bulk] = self.inverse
        kernel = design.weighted_gram(weights)
        kernel[np.diag_indices_from(kernel)] += 1.0
        self.upper = linalg.cholesky(kernel, lower=False, check_finite=False, overwrite_a=True)
        self._core: F64Array | None = None
        self._rest_factors: tuple[F64Array, F64Array] | None = None
        self._factors: tuple[F64Array, F64Array, F64Array] | None = None
        self._units: _Units | None = None
        if self.rest.size:
            non_positive_groups = design.members[self.rest[scaled_precision[self.rest] <= 0.0]]
            if np.unique(non_positive_groups).shape[0] < non_positive_groups.shape[0]:
                # Two members of one group with t <= 0: A' along their difference (e_j - e_k, oriented; the data see
                # neither) is t_j + t_k <= 0, so A' is not positive definite (exactly).
                raise np.linalg.LinAlgError("two tie members with non-positive sites share a group: A' is not positive definite")
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

    def log_determinant(self) -> float:
        """log |A'| = log |T_P| + log |K| + log |S| (the determinant lemma on P, then N's Schur complement)."""
        value = float(np.sum(np.log(self.precision[self.bulk]))) + 2.0 * float(np.sum(np.log(np.diag(self.upper))))
        if self.rest.size:
            value += 2.0 * float(np.sum(np.log(np.diag(self.schur_upper))))
        return value

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

    def units(self) -> _Units:
        """Members whose columns of Phi and Psi coincide: the bulk members of one group with one site (their columns
        and t agree; tied rare variants under one prior reach one site by symmetry), and each member of the rest on
        its own. Then A'^-1 = E G E' + diag(delta) over the members, with G = -Phi_u'Phi_u + Psi_u'Psi_u on the units
        (E the members' unit indicator, delta = 1/t on the bulk, 0 on the rest): p_units ~ the tie groups, not the
        members (review-mathbugs T1)."""
        if self._units is None:
            groups = self.design.members
            key = np.column_stack([groups[self.bulk].astype(np.float64), self.precision[self.bulk]])
            _unique, representative, bulk_unit, count = np.unique(key, axis=0, return_index=True, return_inverse=True, return_counts=True)
            bulk_units = representative.shape[0]
            of_member = np.empty(self.variant_count, dtype=np.int64)
            of_member[self.bulk] = np.asarray(bulk_unit).ravel()
            of_member[self.rest] = bulk_units + np.arange(self.rest.size)
            self._units = _Units(
                of_member=of_member,
                bulk_representative=self.bulk[representative],
                count=np.concatenate([count, np.ones(self.rest.size, dtype=np.int64)]),
                delta=np.concatenate([1.0 / self.precision[self.bulk[representative]], np.zeros(self.rest.size)]),
                kernel=self,
            )
        return self._units

    def hadamard_quadratic(self, direction: F64Array) -> float:
        """d'(A'^-1 o A'^-1) d without forming it: with A'^-1 = E G E' + diag(delta) (``units``) and G = U' S U (U =
        [Phi_u; Psi_u], S = diag(-1 on Phi, +1 on Psi)), it is tr(S M S M) + sum_j (2 delta_j G_jj + delta_j^2) d_j^2,
        M = U diag(E' d) U' (r x r): O(p_u r^2)."""
        units = self.units()
        phi, psi = units.factors()
        unit_direction = units.sum(direction)
        factor_rows = np.vstack([phi, psi]) if psi.shape[0] else phi
        signs = np.concatenate([-np.ones(phi.shape[0]), np.ones(psi.shape[0])])
        middle = (factor_rows * unit_direction[None, :]) @ factor_rows.T
        unit_diagonal = -np.einsum("ij,ij->j", phi, phi) + np.einsum("ij,ij->j", psi, psi)
        delta = units.delta[units.of_member]
        local = (2.0 * delta * unit_diagonal[units.of_member] + delta * delta) @ np.square(direction)
        return float(np.sum(signs[:, None] * signs[None, :] * middle * middle) + local)

    def update_divergence(self, noise: float, precision_step: F64Array, shift_step: F64Array, mean: F64Array) -> float:
        """KL(q || q') to second order for the site change (d tau, d nu), in nats: the Fisher metric of q's natural
        parameters, 1/2 r' Sigma r + 1/4 d tau' (Sigma o Sigma) d tau with r = d nu - d tau o mu (the mean's and the
        variances' parts of q's covariance of the statistics (beta, -beta^2 / 2))."""
        right = shift_step - precision_step * mean
        return 0.5 * noise * float(right @ self.solve(right)) + 0.25 * noise * noise * self.hadamard_quadratic(precision_step)

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


@dataclass
class _Units:
    """``_Kernel.units``: each member's unit, each bulk unit's representative member, the units' member counts and
    delta (1/t, 0 on the rest), and (lazily) Phi_u (n x units) and Psi_u (|N| x units), the columns the unit's members
    share."""

    of_member: I64Array
    bulk_representative: I64Array
    count: I64Array
    delta: F64Array
    kernel: _Kernel
    _phi: F64Array | None = None
    _psi: F64Array | None = None
    _sum: object = None

    @property
    def size(self) -> int:
        return int(self.count.shape[0])

    def sum(self, values: F64Array) -> F64Array:
        """E' v: each unit's sum over its members."""
        if self._sum is None:
            members = self.of_member.shape[0]
            self._sum = sparse.csr_matrix((np.ones(members), (self.of_member, np.arange(members))), shape=(self.size, members))
        return np.asarray(self._sum @ values)

    def factors(self) -> tuple[F64Array, F64Array]:
        if self._phi is None:
            kernel = self.kernel
            bulk_units = self.bulk_representative.shape[0]
            groups = kernel.design.members[self.bulk_representative]
            phi = np.zeros((kernel.upper.shape[0], self.size), order="F")
            whitened = linalg.solve_triangular(kernel.upper, kernel.design.group_columns(groups), trans="T", lower=False, check_finite=False)
            phi[:, :bulk_units] = whitened * self.delta[:bulk_units][None, :]
            psi = np.zeros((kernel.rest.size, self.size))
            if kernel.rest.size:
                psi_bulk, psi_rest = kernel.rest_factors()
                psi[:, :bulk_units] = psi_bulk[:, np.searchsorted(kernel.bulk, self.bulk_representative)]
                psi[:, bulk_units:] = psi_rest
            self._phi, self._psi = phi, psi
        return self._phi, self._psi


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
    """q's responses at one fixed point (``scale_mixture_ep.GaussianPosterior``): Sigma R, -(Sigma o Sigma) W and the
    curvature's linear response, all exactly, over the members.

    Sigma = sigma^2 (E G E' + diag(delta)) (``_Kernel.units``), so Sigma o Sigma = E (G o G) E' sigma^4 + diag(a) with
    a = sigma^4 (2 delta G_uu + delta^2): G o G is formed once on the units (8 p_u^2 bytes) when that fits ``jvp_bytes``,
    and each JVP is then one p_u^2 GEMM; otherwise each column costs O(n^2 p) from the member factors. Forming it pays
    as soon as a call has more than p / (2 n) columns, which the curvature's direction blocks always have."""

    def __init__(self, kernel: _Kernel, noise: float, jvp_bytes: int, profile: dict) -> None:
        self.kernel = kernel
        self.noise = float(noise)
        self.jvp_bytes = int(jvp_bytes)
        self.profile = profile
        self._squared: F64Array | None = None
        self._local: F64Array | None = None
        self._response_key: tuple | None = None
        self._response_factor: tuple | None = None
        self._response_units: tuple | None = None

    def solve(self, right: F64Array, _relative_tolerance: float) -> F64Array:
        started = time.perf_counter()
        result = self.noise * self.kernel.solve(right)
        self.profile["solve_seconds"] += time.perf_counter() - started
        self.profile["solve_columns"] += int(np.asarray(right).shape[1]) if np.ndim(right) == 2 else 1
        return result

    def _explicit(self) -> tuple[F64Array, F64Array]:
        """(sigma^4 G o G on the units, a per unit)."""
        if self._squared is None:
            started = time.perf_counter()
            units = self.kernel.units()
            phi, psi = units.factors()
            gram = linalg.blas.dsyrk(-1.0, phi, trans=1)
            if psi.shape[0]:
                gram = linalg.blas.dsyrk(1.0, psi, beta=1.0, c=gram, trans=1, overwrite_c=1)
            _symmetrize_upper(gram)
            gram *= self.noise
            diagonal = np.diag(gram).copy()
            np.square(gram, out=gram)
            delta = self.noise * units.delta
            self._local = 2.0 * delta * diagonal + delta * delta
            self._squared = gram
            self.profile["form_seconds"] += time.perf_counter() - started
        return self._squared, self._local

    def variance_jvp(self, weights: F64Array) -> F64Array:
        started = time.perf_counter()
        values = np.asarray(weights, dtype=np.float64)
        column = values.ndim == 1
        values = values[:, None] if column else values
        units = self.kernel.units()
        if _FLOAT_BYTES * units.size * units.size <= self.jvp_bytes:
            squared, local = self._explicit()
            result = -((squared @ units.sum(values))[units.of_member] + local[units.of_member][:, None] * values)
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
        diag(diagonal) (``scale_mixture_ep.GaussianPosterior``), over the members.

        Response units are the members that share a unit and all four coefficients. On a vector constant within them
        the matrix is the p_r x p_r one below; on a vector that sums to zero within each, S2 acts as diag(a) and M as
        diag(m0) (E' of it vanishes), so the matrix is diag(1 - (1 - w a) m0) there. B splits into the two parts
        exactly, so X does. The reduced matrix: M_r = diag(d0) + U V' of rank r = n + |N| (G's factors, V scaled by the
        units' counts) and S2_r = (G o G)[u_r, u_r] diag(k_r) + diag(a_r); diag(1 - d0) - U V' + diag(w) (S2_r diag(d0)
        + (S2_r U) V') is formed in 3 p_r^2 r flops and its LU costs 2 p_r^3 / 3, whatever the number of directions.
        The coefficients depend only on the fixed point and its hyperparameters, which one correction holds for every
        view it is asked, so the factor is kept and each later call costs 2 p_r^2 per direction."""
        started = time.perf_counter()
        units = self.kernel.units()
        key = (left, right, diagonal, weight)
        if self._response_key is None or not all(np.array_equal(a, b) for a, b in zip(self._response_key, key)):
            rows = np.column_stack([units.of_member.astype(np.float64), left, right, diagonal, weight])
            _unique, representative, of_member, count = np.unique(rows, axis=0, return_index=True, return_inverse=True, return_counts=True)
            of_member = np.asarray(of_member).ravel()
            unit = units.of_member[representative]
            # The matrix is C-ordered: its transpose is the same buffer in Fortran order, which LAPACK factors in
            # place (a C-ordered one it would copy), and the transposed solve then gives M X = B.
            matrix = self._response_matrix(left[representative], right[representative], diagonal[representative], weight[representative], unit, count)
            self._response_factor = linalg.lu_factor(matrix.T, overwrite_a=True, check_finite=False)
            _squared, local = self._explicit()
            member_local = local[units.of_member]
            m0 = diagonal + self.noise * left * units.delta[units.of_member] * right
            self._response_units = (of_member, count, 1.0 - (1.0 - weight * member_local) * m0)
            self._response_key = tuple(np.array(value, copy=True) for value in key)
            self.profile["response_factorizations"] += 1
        of_member, count, within = self._response_units
        values = np.asarray(rhs, dtype=np.float64)
        column = values.ndim == 1
        values = values[:, None] if column else values
        summed = sparse.csr_matrix((np.ones(of_member.shape[0]), (of_member, np.arange(of_member.shape[0]))), shape=(count.shape[0], of_member.shape[0])) @ values
        mean = np.asarray(summed) / count[:, None]
        solution = linalg.lu_solve(self._response_factor, mean, trans=1, check_finite=False)[of_member]
        shared = count[of_member] > 1
        if np.any(shared):
            solution[shared] += (values[shared] - mean[of_member[shared]]) / within[shared, None]
        self.profile["response_seconds"] += time.perf_counter() - started
        self.profile["responses"] += 1
        return solution[:, 0] if column else solution

    def _response_matrix(self, left: F64Array, right: F64Array, diagonal: F64Array, weight: F64Array, unit: I64Array, count: I64Array) -> F64Array:
        """The reduced response matrix over response units (``linear_response``): ``unit`` each one's unit, ``count``
        its members."""
        squared, local = self._explicit()
        units = self.kernel.units()
        phi, psi = units.factors()
        noise = self.noise
        d0 = diagonal + noise * left * units.delta[unit] * right
        factor_rows = np.vstack([phi[:, unit], psi[:, unit]]) if psi.shape[0] else phi[:, unit]
        signs = np.concatenate([-np.ones(phi.shape[0]), np.ones(psi.shape[0])])
        low = (noise * left)[:, None] * (factor_rows.T * signs[None, :])  # U (p_r x r)
        high = factor_rows * (count * right)[None, :]  # V' (r x p_r)
        matrix = squared[np.ix_(unit, unit)]
        matrix *= count[None, :]
        matrix[np.diag_indices_from(matrix)] += local[unit]
        squared_low = matrix @ low
        matrix *= d0[None, :]
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
        """The responses; the exact linear response when this posterior's unit S2 and response factor fit its share."""
        size = self.kernel.units().size
        exact = _EXACT_RESPONSE_MATRICES * _FLOAT_BYTES * size * size <= self.jvp_bytes
        return GaussianPosterior(solve=self.solve, variance_jvp=self.variance_jvp, linear_response=self.linear_response if exact else None)


# ------------------------------------------------------------------ the convergent fallback (Opper-Winther)


Tilted = Callable[[F64Array, F64Array], tuple[F64Array, F64Array, F64Array, F64Array, F64Array]]
_TILTED_FIELDS = ("log_normalizer", "mean", "variance", "third_cumulant", "fourth_cumulant")
"""(cavity precision, cavity shift) -> each site's tilted (log normalizer, mean, variance, third and fourth cumulant)."""


@dataclass
class _LoopPoint:
    """q at sites (tau, nu) under fixed marginals (P_s, h_s): its kernel, mean, marginal variances, the cavities
    (P_s - tau, h_s - nu), their tilted moments, and Phi = log Z_q + log Z_r with its gradient in (nu, tau)."""

    site_precision: F64Array
    site_shift: F64Array
    kernel: _Kernel
    mean: F64Array
    variance: F64Array
    tilted_mean: F64Array
    tilted_variance: F64Array
    third: F64Array
    fourth: F64Array
    value: float
    gradient: F64Array


def _loop_point(
    design: _Design, noise: float, data_score: F64Array, site_precision: F64Array, site_shift: F64Array,
    marginal_precision: F64Array, marginal_shift: F64Array, tilted: Tilted, largest_variance: F64Array,
) -> _LoopPoint | None:
    """The double loop's inner objective at these sites (``tests/ep_eb_reference._double_loop_objective``), or None
    outside EP's domain: A' not positive definite, or a cavity whose tilted law is improper (1 + v_max P <= 0).

    Phi = 1/2 s' mu - 1/2 log |Lambda + diag tau| + sum_j log Z_j(P_s - tau, h_s - nu) with s = l + nu, Lambda + diag
    tau = A' / sigma^2 and mu = A'^-1 (Xp'y + sigma^2 nu), up to a constant in the sites."""
    cavity_precision = marginal_precision - site_precision
    if not np.all(1.0 + largest_variance * cavity_precision > 0.0):
        return None
    try:
        kernel = _Kernel(design, noise * site_precision)
    except np.linalg.LinAlgError:
        return None
    scaled_shift = data_score + noise * site_shift
    mean = kernel.solve(scaled_shift)
    variance = noise * kernel.variances()
    log_normalizer, tilted_mean, tilted_variance, third, fourth = tilted(cavity_precision, marginal_shift - site_shift)
    values = (log_normalizer, tilted_mean, tilted_variance, third, fourth)
    # A computed tilted variance of 0 (the engine's below-floor approximation, not the model: review-mathbugs N1) has no
    # finite site: outside the domain.
    if not (all(np.all(np.isfinite(value)) for value in values) and np.all(tilted_variance > 0.0)):
        return None
    value = 0.5 * float(scaled_shift @ mean) / noise - 0.5 * kernel.log_determinant() + float(np.sum(log_normalizer))
    tilted_second = tilted_variance + tilted_mean**2
    gradient = np.concatenate([mean - tilted_mean, -0.5 * (variance + mean**2 - tilted_second)])
    return _LoopPoint(
        site_precision=site_precision, site_shift=site_shift, kernel=kernel, mean=mean, variance=variance, tilted_mean=tilted_mean,
        tilted_variance=tilted_variance, third=third, fourth=fourth, value=value, gradient=gradient,
    )


def _positive_definite(design: _Design, noise: float, site_precision: F64Array) -> bool:
    """Whether A' = Xp'Xp + sigma^2 diag(tau) is positive definite (the double loop's only condition on its start)."""
    try:
        _Kernel(design, noise * site_precision)
    except np.linalg.LinAlgError:
        return False
    return True


def _in_domain(
    design: _Design, noise: float, data_score: F64Array, site_precision: F64Array, site_shift: F64Array, tilted: Tilted, largest_variance: F64Array
) -> bool:
    """Whether the sites lie in EP's domain: A' positive definite, and every cavity at q's own marginals proper."""
    try:
        kernel = _Kernel(design, noise * site_precision)
    except np.linalg.LinAlgError:
        return False
    mean = kernel.solve(data_score + noise * site_shift)
    variance = noise * kernel.variances()
    return _loop_point(design, noise, data_score, site_precision, site_shift, 1.0 / variance, mean / variance, tilted, largest_variance) is not None


def _site_blocks(point: _LoopPoint) -> tuple[F64Array, F64Array, F64Array]:
    """Each site's 2 x 2 block of Cov_r of the statistics (beta, -beta^2 / 2) under its tilted law: (a, b, c) =
    (v, -(k3 + 2 m v) / 2, (k4 + 2 v^2 + 4 m k3 + 4 m^2 v) / 4), from the central moments m, v, k3 and mu4 = k4 + 3 v^2."""
    mean, variance, third, fourth = point.tilted_mean, point.tilted_variance, point.third, point.fourth
    return variance, -0.5 * (third + 2.0 * mean * variance), 0.25 * (fourth + 2.0 * variance**2 + 4.0 * mean * third + 4.0 * mean**2 * variance)


def _hessian_product(point: _LoopPoint, noise: float, jvp_bytes: int, profile: dict) -> Callable[[F64Array], F64Array]:
    """v -> H v for Phi's Hessian H = Cov_q + Cov_r in (nu, tau) at ``point`` (``_newton_step``)."""
    size = point.mean.shape[0]
    mean = point.mean
    posterior = _DensePosterior(point.kernel, noise, jvp_bytes, profile)
    a_r, b_r, c_r = _site_blocks(point)

    def product(vector: F64Array) -> F64Array:
        shift_part, precision_part = vector[:size], vector[size:]
        weighted = posterior.solve(shift_part - mean * precision_part, 0.0)
        return np.concatenate([
            weighted + a_r * shift_part + b_r * precision_part,
            -mean * weighted - 0.5 * posterior.variance_jvp(precision_part) + b_r * shift_part + c_r * precision_part,
        ])

    return product


def _segment_start(
    design: _Design, noise: float, data_score: F64Array, precision: F64Array, shift: F64Array, marginal_precision: F64Array,
    marginal_shift: F64Array, cavity_precision: F64Array, tilted: Tilted, largest_variance: F64Array, jvp_bytes: int, profile: dict,
) -> _LoopPoint | None:
    """The inner problem's start when q's cavities at the current sites theta* leave its domain (theory-ep, THEORY_EP.md
    section 7): on the segment theta(s) = theta0 + s (theta* - theta0) from the sites equal to q's new marginals, theta0
    (every cavity zero, the precision definite), every cavity is s times q's cavity at theta*, and the precision stays
    definite (a convex combination of two definite ones). The domain is then s < s_j = 1 / (v_j (-P*_j)) for each
    negative cavity precision P*_j, and the start is the minimizer of the convex Phi on [0, min(1, min_j s_j)), where
    Phi is +inf at the boundary: safeguarded Newton on s, with its derivative g'd and curvature d'Hd, bracketed by the
    derivative's sign, until its decrement is below Phi's rounding. The inner problem's minimizer does not depend on
    the start, so this only shortens the inner solve. None where a computed tilted variance is 0 (N1)."""
    size = precision.shape[0]
    direction = np.concatenate([shift - marginal_shift, precision - marginal_precision])
    negative = cavity_precision < 0.0
    limits = np.where(negative, 1.0 / np.where(negative, largest_variance * -cavity_precision, 1.0), np.inf)
    upper = min(1.0, float(np.min(limits)) if limits.size else 1.0)
    profile["double_loop_projected_starts"] += 1

    def at(fraction: float) -> _LoopPoint | None:
        profile["segment_evaluations"] += 1
        return _loop_point(
            design, noise, data_score, marginal_precision + fraction * direction[size:], marginal_shift + fraction * direction[:size],
            marginal_precision, marginal_shift, tilted, largest_variance,
        )

    def slope_and_curvature(point: _LoopPoint) -> tuple[float, float]:
        return float(point.gradient @ direction), float(direction @ _hessian_product(point, noise, jvp_bytes, profile)(direction))

    lower, fraction = 0.0, 0.0
    point = at(0.0)
    if point is None:
        return None
    slope, curvature = slope_and_curvature(point)
    while slope < 0.0 and curvature > 0.0 and slope * slope / curvature > _EPSILON * abs(point.value):
        trial = fraction - slope / curvature
        if not lower < trial < upper:
            trial = 0.5 * (lower + upper)
        if trial in (lower, upper, fraction):
            break
        candidate = at(trial)
        if candidate is None:
            upper = trial
            continue
        candidate_slope, candidate_curvature = slope_and_curvature(candidate)
        if candidate_slope < 0.0:
            lower = trial
        else:
            upper = trial
        if candidate.value <= point.value:
            fraction, point, slope, curvature = trial, candidate, candidate_slope, candidate_curvature
    return point


def _newton_step(point: _LoopPoint, noise: float, jvp_bytes: int, profile: dict) -> tuple[F64Array, float]:
    """A Newton step s ~ -H^-1 g of Phi and its decrement -g's, by conjugate gradients on H = Cov_q + Cov_r.

    Cov_q is q's covariance of the statistics (beta, -beta^2 / 2): with w = Sigma (a - mu o b) its product with (a, b) is
    (w, -mu o w + (Sigma o Sigma) b / 2), two kernel solves and one variance JVP. Cov_r is block-diagonal over the
    sites, and H >= Cov_r, so the model decrease CG can still add, r' H^-1 r / 2, is at most r' Cov_r^-1 r / 2. CG stops
    once that bound is no more than the decrease it has achieved, so the step always takes at least half of Newton's
    model decrease (it stops at once only where g = 0), or at the dimension, where CG is exact. It is preconditioned
    by the 2 x 2 site blocks of H."""
    size = point.mean.shape[0]
    mean = point.mean
    a_r, b_r, c_r = _site_blocks(point)
    a_h, b_h, c_h = a_r + point.variance, b_r - point.variance * mean, c_r + 0.5 * point.variance**2 + mean**2 * point.variance
    product = _hessian_product(point, noise, jvp_bytes, profile)

    def block_solve(vector: F64Array, a: F64Array, b: F64Array, c: F64Array) -> F64Array:
        determinant = a * c - b * b
        first, second = vector[:size], vector[size:]
        return np.concatenate([(c * first - b * second) / determinant, (a * second - b * first) / determinant])

    with np.errstate(divide="ignore", invalid="ignore"):
        bounded = bool(np.all(a_r * c_r - b_r * b_r > 0.0))
    residual = -point.gradient
    step = np.zeros(2 * size)
    preconditioned = block_solve(residual, a_h, b_h, c_h)
    direction = preconditioned.copy()
    product_value = float(residual @ preconditioned)
    achieved = 0.0
    for _iteration in range(2 * size):
        if bounded and 0.5 * float(residual @ block_solve(residual, a_r, b_r, c_r)) <= achieved:
            break
        image = product(direction)
        curvature = float(direction @ image)
        if not curvature > 0.0:
            break
        length = product_value / curvature
        # Each PCG step lowers the quadratic model by length (r' M^-1 r) / 2.
        achieved += 0.5 * length * product_value
        step += length * direction
        residual -= length * image
        preconditioned = block_solve(residual, a_h, b_h, c_h)
        next_value = float(residual @ preconditioned)
        direction = preconditioned + (next_value / product_value) * direction
        product_value = next_value
        profile["double_loop_cg"] += 1
    return step, -float(point.gradient @ step)


def double_loop_sites(
    design: _Design, noise: float, data_score: F64Array, site_precision: F64Array, site_shift: F64Array, tilted: Tilted,
    largest_variance: F64Array, draw_count: int, jvp_bytes: int, profile: dict, trace: list | None = None,
) -> tuple[F64Array, F64Array]:
    """EP's sites at fixed hyperparameters and noise by the Opper-Winther double loop, which provably reaches a
    stationary point of the EP free energy (MODEL.md section 4's fallback; ``tests/ep_eb_reference.double_loop_sites``).

    The outer loop fixes (P_s, h_s) at q's marginals, which bounds the free energy's concave part linearly; the inner
    problem, the minimum of the convex Phi over the sites, is solved by Newton with sufficient-decrease halving until no
    representable step along Newton's direction lowers Phi by half its quadratic model's decrease; below Phi's rounding,
    where values cannot tell a decrease, full Newton steps continue while Newton's decrement (from the gradient) falls. Each outer step is
    majorize-minimize (the free energy is at most Phi plus a constant, with equality at q's current marginals), so every
    accepted inner step lowers the free energy. The loop ends at small_n's own EP check, the undamped
    update's KL 1/2 r' Sigma r at most 1/(2K) nats, or when an outer step leaves the sites unchanged, which is EP's fixed
    point: at an outer step's start (P_s, h_s) are q's own marginals, so Phi's gradient there, (mu - E_r[beta],
    -(z + mu^2 - E_r[beta^2]) / 2) at EP's own cavities, is exactly EP's moment-matching residual. An unchanged step
    means the first Newton step, which takes at least half of Newton's model decrease (``_newton_step``), found no
    representable fraction with sufficient decrease: Phi is stationary there to its rounding, and with it the moment-matching equations,
    whose solutions are EP's fixed points. Sites are never clipped.

    No tilted law is ever evaluated outside EP's domain, so no FloatingPointError is reachable from it: q's own
    cavities are evaluated (the fixed-point check) only when every one is proper; the inner problem's start is moved
    into its domain when they are not (the inner problem is convex, so its minimizer does not depend on the start, and
    the marginals' own sites, zero cavities with positive sites, are always inside it); every inner step is accepted
    only at a point ``_loop_point`` finds in the domain (definite precision, proper inner cavities, finite moments).
    The start's precision must be positive definite (ValueError otherwise: the prior's moment-matched sites are)."""
    precision = np.array(site_precision, dtype=np.float64, copy=True)
    shift = np.array(site_shift, dtype=np.float64, copy=True)
    size = precision.shape[0]
    if not _positive_definite(design, noise, precision):
        raise ValueError("the EP double loop's start leaves the precision not positive definite")
    while True:
        profile["double_loop_outer"] += 1
        if trace is not None:
            # Verification only (theory-ep's exact decrease check): the EC free energy at each outer step's start.
            trace.append(_ec_free_energy(design, noise, data_score, precision, shift, tilted, largest_variance))
        kernel = _Kernel(design, noise * precision)
        mean = kernel.solve(data_score + noise * shift)
        variances, _removed, cavity_scaled = kernel.cavity()
        variance = noise * variances
        cavity_precision = cavity_scaled / noise
        marginal_precision, marginal_shift = 1.0 / variance, mean / variance
        proper = 1.0 + largest_variance * cavity_precision > 0.0
        if np.all(proper):
            log_normalizer, tilted_mean, tilted_variance, _third, _fourth = tilted(cavity_precision, mean / variance - shift)
            if not (np.all(tilted_variance > 0.0) and np.all(np.isfinite(tilted_mean))):
                raise NoFixedPoint("a computed tilted variance is 0 at q's cavities: no finite EP site")
            # The EP check (``_DenseFixedPoints._solve``): the undamped update's KL in nats.
            target_precision = 1.0 / tilted_variance - cavity_precision
            target_shift = tilted_mean / tilted_variance - (mean / variance - shift)
            if kernel.update_divergence(noise, target_precision - precision, target_shift - shift, mean) <= 0.5 / draw_count:
                return precision, shift
            reseeded = False
        else:
            reseeded = True
        if reseeded:
            # The outer step sets the frozen marginals to q's (CCCP), and q's own cavity 1/z - tau can then be improper
            # (a tilted law can be wider than its cavity, so the inner optimum's marginals, the tilted moments at its
            # proper inner cavities, need not leave 1/z - tau proper; fit-api rwAMR [real]). Such sites are no fixed
            # point (every fixed point's cavities are proper) and are never evaluated; the inner problem starts on the
            # segment toward the sites equal to q's new marginals (``_segment_start``), and its convex minimizer does
            # not depend on the start.
            profile["double_loop_reseeds"] += 1
            point = _segment_start(
                design, noise, data_score, precision, shift, marginal_precision, marginal_shift, cavity_precision, tilted, largest_variance,
                jvp_bytes, profile,
            )
        else:
            point = _loop_point(design, noise, data_score, precision, shift, marginal_precision, marginal_shift, tilted, largest_variance)
        if point is None:
            # The start is in the inner domain (the precision is definite, and every inner cavity is q's proper cavity,
            # or on the segment a fraction of it inside the boundary), so only a tilted moment can fail: a computed
            # variance of 0 (the engine's below-floor approximation, N1), which no finite site matches.
            raise NoFixedPoint("a computed tilted variance is 0 at the double loop's cavities: no finite EP site")
        precision, shift = point.site_precision, point.site_shift
        start_precision, start_shift = precision.copy(), shift.copy()
        polish_decrement: float | None = None
        polish_origin = point
        while True:
            step, decrement = _newton_step(point, noise, jvp_bytes, profile)
            if not decrement > 0.0:
                break
            accepted = None
            fraction = 1.0 if decrement > _EPSILON * abs(point.value) else 0.0
            while fraction * float(np.max(np.abs(step))) > _EPSILON * (1.0 + max(float(np.max(np.abs(point.site_precision))), float(np.max(np.abs(point.site_shift))))):
                candidate = _loop_point(
                    design, noise, data_score, point.site_precision + fraction * step[size:], point.site_shift + fraction * step[:size],
                    marginal_precision, marginal_shift, tilted, largest_variance,
                )
                # Sufficient decrease: at least half of the quadratic model's own decrease along the step, which for a CG
                # iterate (d'Hd = -g'd = decrement) is fraction (1 - fraction / 2) decrement. Plain decrease can accept
                # vanishing decreases and stall short of the minimum (theory-ep); this is Armijo with c = 1/4 at
                # fraction <= 1, and it keeps every exact Newton step on a quadratic.
                model = fraction * (1.0 - 0.5 * fraction) * decrement
                if candidate is not None and candidate.value <= point.value - 0.5 * model:
                    accepted = candidate
                    break
                fraction *= 0.5
            if accepted is None:
                # Phi's values cannot tell a decrease along the step (it is below Phi's evaluation error), and a
                # value-based stop resolves the minimum only to sqrt(eps). Newton's decrement comes from the gradient,
                # not from differences of Phi: in this quadratic regime the full step is taken while that decrement
                # keeps falling; a step after which it does not fall is undone.
                if polish_decrement is not None and not decrement < polish_decrement:
                    point = polish_origin
                    break
                accepted = _loop_point(
                    design, noise, data_score, point.site_precision + step[size:], point.site_shift + step[:size],
                    marginal_precision, marginal_shift, tilted, largest_variance,
                )
                if accepted is None:
                    break
                polish_decrement, polish_origin = decrement, point
            point = accepted
            profile["double_loop_newton"] += 1
        precision, shift = point.site_precision, point.site_shift
        if not reseeded and np.array_equal(precision, start_precision) and np.array_equal(shift, start_shift):
            # Phi stationary at q's own marginals to its rounding, with q's cavities proper: EP's fixed point (see the
            # docstring). A reseeded start is not q's own marginals, so its outer step is taken and checked afresh.
            profile["double_loop_stationary"] += 1
            return precision, shift


def _log_evidence(design: _Design, noise: float, data_score: F64Array, site_precision: F64Array, site_shift: F64Array, tilted: Tilted) -> float:
    """log Z_EP at these sites, up to terms shared by every candidate at the same hyperparameters and noise (the
    fixed-point selection's criterion, lead ruling; ``tests/ep_eb_reference.site_state``'s log_evidence in small_n's
    units): 1/2 r' A'^-1 r / sigma^2 - 1/2 log |A'| + p/2 log sigma^2 + sum_j log Z~_j - sum_j (m_j^2 / (2 v_j) + log(v_j) / 2),
    r = Xp'y + sigma^2 nu, with every cavity from the kernel without cancellation."""
    kernel = _Kernel(design, noise * site_precision)
    right = data_score + noise * site_shift
    mean = kernel.solve(right)
    variances, _removed, cavity_scaled = kernel.cavity()
    variance = noise * variances
    log_normalizer = tilted(cavity_scaled / noise, mean / variance - site_shift)[0]
    return (
        0.5 * float(right @ mean) / noise - 0.5 * kernel.log_determinant() + 0.5 * mean.shape[0] * float(np.log(noise))
        + float(np.sum(log_normalizer)) - float(np.sum(0.5 * np.square(mean) / variance + 0.5 * np.log(variance)))
    )


def _ec_free_energy(
    design: _Design, noise: float, data_score: F64Array, site_precision: F64Array, site_shift: F64Array, tilted: Tilted,
    largest_variance: F64Array,
) -> float:
    """The EC free energy F(eta) = A_q*(eta) + A_s*(eta) - A_r*(eta) at q's diagonal moments eta (theory-ep, THEORY_EP.md
    section 7): the quantity the double loop's outer steps lower, so its decrease can be verified rather than assumed.
    At an EP fixed point it equals -log Z_EP (``_log_evidence``) exactly.

    A_q*(eta) = theta . eta - A_q(theta) at q's own sites; A_r*(eta) = -1/2 sum_j log(2 pi e z_j), the Gaussian's; and
    A_s*(eta) = sum_j sup_lambda (lambda . eta_j - log Z~_j(lambda)), each a 2-D convex problem over the cavity lambda =
    (h, P) whose tilted moments match eta_j: Newton on it with the tilted covariance of (beta, -beta^2 / 2) as its Hessian,
    sufficient-decrease halving inside the tilted law's domain (1 + v_max P > 0), from q's own cavity where proper and
    the zero cavity elsewhere, until no decrement exceeds its value's rounding."""
    kernel = _Kernel(design, noise * site_precision)
    right = data_score + noise * site_shift
    mean = kernel.solve(right)
    variances, _removed, cavity_scaled = kernel.cavity()
    variance = noise * variances
    count = mean.shape[0]
    second = variance + np.square(mean)
    theta_eta = float(site_shift @ mean) - 0.5 * float(site_precision @ second)
    log_partition_q = 0.5 * float(right @ mean) / noise - 0.5 * (kernel.log_determinant() - count * float(np.log(noise))) + 0.5 * count * float(np.log(2.0 * np.pi))
    entropy_r = -0.5 * float(np.sum(np.log(2.0 * np.pi * variance) + 1.0))
    # A_s*: per site, the cavity whose tilted law has q's marginal moments.
    cavity = cavity_scaled / noise
    precision = np.where(1.0 + largest_variance * cavity > 0.0, cavity, 0.0)
    shift = np.where(1.0 + largest_variance * cavity > 0.0, mean / variance - site_shift, 0.0)

    def objective(precision_values: F64Array, shift_values: F64Array) -> tuple[F64Array, tuple[F64Array, ...]]:
        values = tilted(precision_values, shift_values)
        log_normalizer = values[0]
        return log_normalizer - (shift_values * mean - 0.5 * precision_values * second), values

    value, moments = objective(precision, shift)
    live = np.ones(count, dtype=bool)
    while np.any(live):
        _log_normalizer, tilted_mean, tilted_variance, third, fourth = moments
        gradient_shift = tilted_mean - mean
        gradient_precision = -0.5 * (tilted_variance + np.square(tilted_mean) - second)
        a, b, c = tilted_variance, -0.5 * (third + 2.0 * tilted_mean * tilted_variance), 0.25 * (fourth + 2.0 * tilted_variance**2 + 4.0 * tilted_mean * third + 4.0 * tilted_mean**2 * tilted_variance)
        determinant = a * c - b * b
        step_shift = -(c * gradient_shift - b * gradient_precision) / determinant
        step_precision = -(a * gradient_precision - b * gradient_shift) / determinant
        decrement = -(gradient_shift * step_shift + gradient_precision * step_precision)
        live &= decrement > _EPSILON * np.maximum(np.abs(value), 1.0)
        if not np.any(live):
            break
        fraction = np.where(live, 1.0, 0.0)
        accepted = ~live
        while not np.all(accepted):
            trial_precision = np.where(accepted, precision, precision + fraction * step_precision)
            trial_shift = np.where(accepted, shift, shift + fraction * step_shift)
            inside = 1.0 + largest_variance * trial_precision > 0.0
            safe_precision = np.where(inside, trial_precision, precision)
            safe_shift = np.where(inside, trial_shift, shift)
            trial_value, trial_moments = objective(safe_precision, safe_shift)
            model = fraction * (1.0 - 0.5 * fraction) * decrement
            good = inside & (trial_value <= value - 0.5 * model) & ~accepted
            precision, shift = np.where(good, trial_precision, precision), np.where(good, trial_shift, shift)
            value = np.where(good, trial_value, value)
            moments = tuple(np.where(good, trial, current) for trial, current in zip(trial_moments, moments))
            accepted |= good
            fraction = np.where(accepted, fraction, 0.5 * fraction)
            stalled = ~accepted & (fraction * np.maximum(np.abs(step_precision), np.abs(step_shift)) <= _EPSILON * (1.0 + np.maximum(np.abs(precision), np.abs(shift))))
            live &= ~stalled
            accepted |= stalled
    conjugate_s = float(np.sum(-value))
    return theta_eta - log_partition_q + conjugate_s - entropy_r


def _best_double_loop(
    design: _Design, noise: float, data_score: F64Array, starts: Sequence[tuple[F64Array, F64Array]], tilted: Tilted, largest_variance: F64Array,
    draw_count: int, jvp_bytes: int, profile: dict,
) -> tuple[F64Array, F64Array]:
    """The double loop from every start whose precision is positive definite (its only condition: an improper cavity
    only reseeds the inner problem), keeping the fixed point with the highest log Z_EP (lead ruling: EP's fixed point
    is not unique in general, and the selection is by the evidence)."""
    best: tuple[float, F64Array, F64Array] | None = None
    for precision, shift in starts:
        if not _positive_definite(design, noise, precision):
            continue
        found = double_loop_sites(design, noise, data_score, precision, shift, tilted, largest_variance, draw_count, jvp_bytes, profile)
        evidence = _log_evidence(design, noise, data_score, found[0], found[1], tilted)
        profile["double_loop_candidates"] += 1
        if best is None or evidence > best[0]:
            best = (evidence, found[0], found[1])
    if best is None:
        raise NoFixedPoint("no start of the double loop leaves the precision positive definite (the moment-matched sites leave it singular)")
    return best[1], best[2]


# ------------------------------------------------------------------ the EP fixed points (Stage 2's, exact)


def _new_profile() -> dict:
    return {name: 0 for name in (
        "factorizations", "refreshes", "passes", "fixed_point_calls", "solve_columns", "jvp_columns", "responses", "response_factorizations",
        "double_loops", "double_loop_outer", "double_loop_newton", "double_loop_cg", "double_loop_stationary", "double_loop_candidates",
        "double_loop_reseeds", "double_loop_projected_starts", "segment_evaluations", "repairs", "repair_rounds", "repair_active_max",
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
        """The mean, the exact marginal variances and the cavity precisions at the current sites. Where the precision
        is not positive definite or a cavity's tilted law is not proper, the offending variants are solved back into
        EP's domain by the double loop (``_repair``, growing the set while the rest's cavities leave it) rather than by
        halving negative sites: each halving costs a full
        build, erases the step that led there and returns the same targets (theory-ep's fix (2): a cycle of 33
        halvings per refresh on svpgs-profiler's g3 [real]).

        A cavity is proper when its tilted law is, 1 + v P > 0 at every lattice node, which admits a negative cavity
        precision down to -1 / v_max. ``full_data_fit`` asks P > 0 instead, which EP cannot reach where a column is
        an exact linear combination of columns whose sites are negative (a doubleton is the sum of two singletons):
        its cavity is then proportional to those sites, negative for every negative value, and exactly 0 only when
        they underflow (bench-real chr22 gene 1 [real]: 7 negative sites halved past 1e-250 over 870 refactorizations
        while one column's cavity stayed at -9e-16, its rounding of 0)."""
        largest = self._largest_variances(hyperparameters)
        members = self.design.members
        offending = np.zeros(self.prior.variant_count, dtype=bool)
        while True:
            improper = None
            try:
                self._iterate(self.site_precision, self.site_shift)
            except np.linalg.LinAlgError:
                failure = "the precision is not positive definite with non-negative sites"
            else:
                variances, removed, cavity_precision = self._cavity()
                improper = ~(1.0 + largest * cavity_precision > 0.0)
                if not np.any(improper):
                    self.profile["refreshes"] += 1
                    count = self.prior.variant_count
                    # p_eff = sum_j (1 - tau_j z_j), each term without cancellation, floored at its rounding p eps.
                    self.effective = max(float(np.sum(removed)), _EPSILON * count)
                    return variances, cavity_precision
                failure = "a cavity's tilted law is improper (1 + v P <= 0 on the lattice) with non-negative sites"
            negative = self.site_precision <= 0.0
            if not np.any(negative):
                # Every site positive: A' is positive definite and every cavity precision non-negative, so this is
                # Cholesky's own breakdown, not EP's domain.
                raise NoFixedPoint(failure)
            # A repaired set's new sites, negative ones included, couple into the rest, so a cavity there can leave the
            # domain after a correct repair: A grows by it and is solved again (theory-ep). A only grows, so this ends;
            # at A = every variant it is the whole problem's double loop, which reaches a fixed point in the domain (the
            # EC existence argument, with the lattice's finite top), so a set that cannot grow is numerics.
            grown = offending | negative | (improper if improper is not None else False)
            grown = np.isin(members, np.unique(members[grown]))
            if np.array_equal(grown, offending):
                raise NoFixedPoint(f"{failure}, after the double loop solved every offending variant back into EP's domain")
            if not np.any(offending):
                self.profile["repairs"] += 1
            offending = grown
            self.profile["repair_rounds"] += 1
            self._repair(hyperparameters, offending)

    def _repair(self, hyperparameters: MixtureHyperparameters, offending: F64Array) -> None:
        """Solve the ``offending`` variants (improper cavities, non-positive sites) and their tie groups, A, back into
        EP's domain at the rest's sites by the double loop (theory-ep's fix (2)). At fixed bulk sites (all positive
        here) A's posterior is exact by the Schur complement (``_conditional_block``), A's problem is small-n EP with
        the whitened design, and the double loop's convex inner problem is +inf outside EP's domain, so every step it
        takes keeps every cavity proper; it starts from A's current sites where they lie in A's domain, and from the
        prior's moment-matched ones, keeping the fixed point with the highest log Z_EP (lead ruling). One bulk build
        and |A|-scale work, against a full build per halving."""
        members = self.design.members
        active = np.flatnonzero(np.isin(members, np.unique(members[np.flatnonzero(offending)])))
        self.profile["repair_active_max"] = max(self.profile["repair_active_max"], int(active.size))
        whitened, score = self._conditional_block(active, self.site_precision, self.site_shift)
        design, noise = _Design.dense(whitened), float(self.noise)
        tilted = self._tilted_rows(hyperparameters, active)
        largest = self._largest_variances(hyperparameters)[active]
        prior_precision, prior_shift = moment_matched_prior_sites(self.prior, hyperparameters)
        starts = [(self.site_precision[active], self.site_shift[active]), (prior_precision[active], prior_shift[active])]
        self.profile["double_loops"] += 1
        tau, nu = _best_double_loop(design, noise, score, starts, tilted, largest, self.draw_count, self.working_bytes // _LIVE_FIXED_POINTS, self.profile)
        self.site_precision, self.site_shift = self.site_precision.copy(), self.site_shift.copy()
        self.site_precision[active], self.site_shift[active] = tau, nu

    def _conditional_block(self, active: I64Array, site_precision: F64Array, site_shift: F64Array) -> tuple[F64Array, F64Array]:
        """The active set A's posterior given the rest's (positive) sites, exactly, by the Schur complement: A's
        problem is small-n EP with the whitened design W = U_B^-T X_A (n x |A|, K_B = U_B'U_B = I + X_B T_B^-1 X_B') and
        the score s_A = X_A'y - X_A' K_B^-1 X_B T_B^-1 r'_B (r' = Xp'y + sigma^2 nu). One bulk build."""
        design, noise, count = self.design, float(self.noise), self.prior.variant_count
        bulk = np.setdiff1d(np.arange(count), active, assume_unique=True)
        scaled_bulk = noise * site_precision[bulk]
        weights = np.zeros(count)
        weights[bulk] = 1.0 / scaled_bulk
        kernel = design.weighted_gram(weights)
        kernel[np.diag_indices_from(kernel)] += 1.0
        upper = linalg.cholesky(kernel, lower=False, check_finite=False, overwrite_a=True)
        self.profile["factorizations"] += 1
        right = self.data_score + noise * site_shift
        bulk_values = np.zeros(count)
        bulk_values[bulk] = right[bulk] / scaled_bulk
        solved = linalg.cho_solve((upper, False), design.image(bulk_values), check_finite=False)
        columns = design.columns(active)
        whitened = np.asfortranarray(linalg.solve_triangular(upper, columns, trans="T", lower=False, check_finite=False))
        return whitened, self.data_score[active] - columns.T @ solved

    def _tilted_rows(self, hyperparameters: MixtureHyperparameters, rows: I64Array) -> Tilted:
        """``Tilted`` for the variants ``rows`` alone (each row's own class density and scale)."""
        prior = self.prior
        log_density = class_log_density(prior, hyperparameters.coefficients)[prior.class_index[rows]]
        scales = log_scale(prior, hyperparameters.coefficients)[rows]

        def tilted(cavity_precision: F64Array, cavity_shift: F64Array) -> tuple[F64Array, F64Array, F64Array, F64Array, F64Array]:
            started = time.perf_counter()
            values = [np.empty(rows.shape[0]) for _field in _TILTED_FIELDS]
            classes = prior.class_index[rows]
            for density in np.unique(classes):
                members = np.flatnonzero(classes == density)
                terms = _components(
                    log_density[members[0]], scales[members], prior.log_variance_grid, prior.kernel_floor,
                    np.asarray(cavity_precision)[members], np.asarray(cavity_shift)[members],
                )
                shift = np.asarray(cavity_shift)[members][:, None]
                weights, conditional = terms.responsibility, terms.conditional_variance
                centre = shift * conditional
                mean = np.sum(weights * centre, axis=1)
                offset = centre - mean[:, None]
                second = np.sum(weights * (np.square(offset) + conditional), axis=1)
                values[0][members] = terms.log_normalizer
                values[1][members] = mean
                values[2][members] = second
                values[3][members] = np.sum(weights * (offset**3 + 3.0 * offset * conditional), axis=1)
                values[4][members] = np.sum(weights * (offset**4 + 6.0 * np.square(offset) * conditional + 3.0 * np.square(conditional)), axis=1) - 3.0 * np.square(second)
            self.profile["tilted_seconds"] += time.perf_counter() - started
            return values[0], values[1], values[2], values[3], values[4]

        return tilted

    def _targets(self, hyperparameters: MixtureHyperparameters, cavity: Cavity) -> tuple[F64Array, F64Array]:
        """The mean-matched sites; ``NoFixedPoint`` where a computed tilted variance is 0 or a moment is not finite
        (review-mathbugs N2). The model's tilted laws always have positive variance; a zero one is the engine's
        flat-kernel approximation below the lattice floor (v = 0 there, N1, e2e's) putting all of a far trial's mass on
        those nodes. No finite site matches it, so the trial is refused and the outer loop halves it."""
        started = time.perf_counter()
        moments = tilted_moments(self.prior, hyperparameters, cavity, self.working_bytes)
        self.profile["tilted_seconds"] += time.perf_counter() - started
        if not (np.all(moments.variance > 0.0) and np.all(np.isfinite(moments.mean)) and np.all(np.isfinite(moments.variance))):
            degenerate = int(np.sum(~(moments.variance > 0.0)))
            raise NoFixedPoint(f"{degenerate} computed tilted variances are 0 (point masses) at these hyperparameters: no finite EP site")
        return site_targets(moments, cavity)

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
            divergence = self.kernel.update_divergence(self.noise, target_precision - self.site_precision, target_shift - self.site_shift, mean)
            self.mean_move = 2.0 * divergence
            noise = self._noise(variances)
            self.noise_gain = noise_gain(noise, self.noise, self.sample_count, self.covariate_count)
            # The fixed point is certified in evidence units, as every other certificate: the undamped update moves q
            # by KL(q || q') nats (``_Kernel.update_divergence``: its mean and variance parts, to second order in the
            # site change), and the noise update gains ``noise_gain`` nats; both at most 1/(2K). A tolerance relative
            # to p_eff (the scorer's Monte Carlo resolution, p_eff / K) goes to rounding where the prior collapses
            # (p_eff -> 0 at an edge trial), and EP then ran for minutes on the move's own rounding before refusing;
            # there the KL is second order in the prior's scale and the check passes at the first refresh.
            if divergence <= tolerance and self.noise_gain <= tolerance:
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

    def _tilted(self, hyperparameters: MixtureHyperparameters) -> Tilted:
        def tilted(cavity_precision: F64Array, cavity_shift: F64Array) -> tuple[F64Array, F64Array, F64Array, F64Array, F64Array]:
            started = time.perf_counter()
            cavity = Cavity(precision=cavity_precision, shift=cavity_shift)
            moments = tilted_moments(self.prior, hyperparameters, cavity, self.working_bytes)
            third, fourth = tilted_cumulants(self.prior, hyperparameters, cavity, self.working_bytes)
            self.profile["tilted_seconds"] += time.perf_counter() - started
            return moments.log_normalizer, moments.mean, moments.variance, third, fourth

        return tilted

    def _double_loop(self, hyperparameters: MixtureHyperparameters) -> None:
        """EP's fixed point at these hyperparameters and noise by the double loop (``double_loop_sites``), from the
        current sites and from the prior's moment-matched ones (whichever lie in EP's domain), keeping the one with the
        highest log Z_EP (``_best_double_loop``)."""
        self.profile["double_loops"] += 1
        largest = self._largest_variances(hyperparameters)
        tilted = self._tilted(hyperparameters)
        starts = [(self.site_precision, self.site_shift), moment_matched_prior_sites(self.prior, hyperparameters)]
        precision, shift = _best_double_loop(
            self.design, self.noise, self.data_score, starts, tilted, largest, self.draw_count, self.working_bytes // _LIVE_FIXED_POINTS, self.profile,
        )
        self.site_precision, self.site_shift = precision, shift
        self._iterate(precision, shift)

    def _frozen_passes(self, hyperparameters: MixtureHyperparameters, frozen: F64Array, target_precision: F64Array, target_shift: F64Array) -> None:
        """Mean-only EP with the cavity precisions frozen until the frozen move's KL is below 1/(2K) nats
        (``full_data_fit._FullDataFixedPoints._frozen_passes``)."""
        previous_move, damping = np.inf, 1.0
        while True:
            mean = self.mean.copy()
            fraction = damping
            move = max(float(np.max(np.abs(target_precision - self.site_precision))), float(np.max(np.abs(target_shift - self.site_shift))))
            scale = 1.0 + max(float(np.max(np.abs(self.site_precision))), float(np.max(np.abs(self.site_shift))))
            while True:
                trial_precision = self.site_precision + fraction * (target_precision - self.site_precision)
                trial_shift = self.site_shift + fraction * (target_shift - self.site_shift)
                try:
                    self._iterate(trial_precision, trial_shift)
                    break
                except np.linalg.LinAlgError:
                    fraction *= 0.5
                # Written so that a non-finite step also ends the halving (review-mathbugs N2).
                if not fraction * move > _EPSILON * scale:
                    # PD failures halved the damped step to the sites' rounding: no damped EP pass keeps the precision
                    # positive definite from here, so EP falls back to the convergent double loop (MODEL.md section 4).
                    self._double_loop(hyperparameters)
                    return
            self.site_precision, self.site_shift = trial_precision, trial_shift
            marginal = 1.0 / (frozen + self.site_precision)
            mean_move = float(np.sum(np.square(self.mean - mean) / marginal)) / (fraction * fraction)
            if not np.isfinite(mean_move):
                raise NoFixedPoint("the frozen pass's move is not finite")
            if 0.5 * mean_move <= 0.5 / self.draw_count:
                return
            ratio = mean_move / previous_move
            if ratio >= 1.0:
                if damping < 1.0:
                    # A pass damped by 1/(1 + rho), exact for the map's eigenvalue at -rho^2, still does not contract:
                    # the frozen-cavity map has a mode damping cannot reach (an eigenvalue past +1, or complex), and
                    # mean-only EP does not converge here. EP falls back to the convergent double loop (MODEL.md
                    # section 4), as where no damped pass stays positive definite (svpgs-profiler's slow genes 6, 7).
                    self._double_loop(hyperparameters)
                    return
                damping = 1.0 / (1.0 + np.sqrt(ratio))
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
    """The run wiring's prior (``stage2_wiring._fit_one``) over every member (review-mathbugs T1: an exact-tie member
    keeps its own class and offset, so an SV tied to SNVs keeps the SV prior): one class per variant class present,
    each member's log reliability as its offset, no annotation groups, and the start lattice from the single-variant
    likelihoods at the covariate-only residual variance."""
    members = statistics.active_rows
    _classes, class_index = np.unique(np.asarray(variant_class)[members], return_inverse=True)
    offsets = np.asarray(log_variance_offset, dtype=np.float64)[members]
    residual = statistics.projected_target
    start_noise = float(residual @ residual) / (statistics.sample_count - statistics.covariates.shape[1])
    single_precision = statistics.design.column_squares() / start_noise
    single_shift = statistics.design.back(statistics.target) / start_noise
    nodes, floor, top = derived_lattice(single_precision, single_shift, offsets, 0.5 / draw_count)
    return scale_mixture_prior(
        class_index=class_index.astype(np.int64), log_variance_offset=offsets, annotation_design=np.zeros((members.shape[0], 0)),
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
    # Every member is its own effect (review-mathbugs T1): beta_j = s_j gamma_j on its own standardized column, with
    # no split of a group's effect; the identity map carries each member's own mean and draws.
    member_count = statistics.active_rows.shape[0]
    scoring = ScoringModel.from_reduced_fit(
        active_rows=statistics.active_rows,
        signed_means=statistics.means,
        signed_scales=statistics.scales,
        tie_map=_compact_identity_tie_map(member_count),
        member_prior_variances=prior_second_moment(prior, outer.hyperparameters),
        beta_reduced=statistics.signs * oracle.mean,
        posterior_draws_reduced=statistics.signs[:, None] * draws,
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
        # The move's tolerance: 1/2 r' Sigma r <= 1/(2K) nats.
        draw_tolerance=np.array([1.0 / draw_count]),
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
        "members": int(statistics.design.variant_count),
        "groups": int(statistics.design.group_count),
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
