"""Sequential EP at gene scale on the compiled training coordinates: one site per exact alias group's sum
(``alias_groups``), each updated in turn against the exact current Gaussian, with q's n' x n' kernel inverse carried by
rank-one updates, and every step kept inside EP's domain.

Why sequential. The small-n route's EP (``small_n._DenseFixedPoints``) moved every site at once (mean-only passes with
frozen cavity precisions), which does not contract on correlated designs, and fell back to the double loop, whose
Newton steps each re-form Sigma o Sigma over the tie units: on the first fixed point of a polygenic simulation on real
genotypes (ENSG00000105612.9, 27,424 members in 11,696 tie groups, n = 421) the fallback took 274 of the first 300 s,
176 s of it in 70 formations of that p_u x p_u matrix, and on ENSG00000138468.16 [real, loso/AFR snv] 216 of the first
300 s; neither fit reached its second fixed point in ten minutes. Each update below is EP's exact update of one site
given every other (Minka 2001; Seeger 2008), so the sweeps have EP's fixed points as their fixed points.

Why on the groups' sums. Members of an exact alias group share one column: the data see only their sum, and every
difference between them is data-free. With one site per member, updates one member at a time broke their symmetry and
EP reached a fixed point committed to one proxy (the brief's identical-proxy example: (1.4393, 0.0463), or its
mirror, against the exact (0.742691, 0.742691)); on real designs they turned sites negative across large groups
(sizes 11 to 323) and left the data-free directions nearly singular. Compiled to their sum with its exact induced
prior, a group is one coordinate of the likelihood, and its members are decoded from it (``alias_groups``).

EP's domain. The scale mixture reaches variances past any bound on its lattice (V_max = e^20 u on real genes), so a
cavity's tilted integral is finite only where 1 + V_max P > 0, in effect P >= 0; with p >> n' every column is a
combination of others, and one negative site makes the cavities of the columns that depend on it negative. A valid
multimodal factor can legitimately have a negative site, so no site is clipped: every step is checked against every
group's cavity after it (all of them, by the rank-one form below) and damped by halving until each stays finite, and a
step that no damping keeps valid is refused, which is the numerical contract's failure and is counted (``run``'s
return).

The algebra. With the covariates' complement in an orthonormal basis (n' = n - k rows), A' = X'X + diag t over the
groups (t = sigma^2 tau the scaled sites), Sigma = sigma^2 A'^-1 and mu = A'^-1 b with b = X'y + sigma^2 nu. The groups
split into the bulk P and the rest N: ``small_n._Kernel`` keeps a site in the bulk only where Woodbury keeps every
digit (t >= ||x||^2), while the sweep only moves the sites and every sweep is followed by that exact refresh, so the
sweep keeps half the digits (t >= eps^(1/2) ||x||^2): the kernel's own split put nearly every column in N at the
prior's moment-matched start and made N's Schur complement thousands wide (ENSG00000138468.16 [real]: 776 of the
first 900 s refactoring it). With M = (I + X_P T_P^-1 X_P')^-1, Q = M X_N, S = T_N + X_N' M X_N, u = X_P T_P^-1 b_P,
z = M u, mu_N = S^-1 (b_N - X_N' z), w = z + Q mu_N and W = M - Q S^-1 Q' = (I + X T^-1 X')^-1 over every group:

    for g in P:  A'^-1_gg = d_g - d_g^2 I_g,  mu_g = d_g (b_g - x_g' w),  I_g = x_g' W x_g, d_g = 1 / t_g;
    for g in N:  A'^-1_gg = (S^-1)_gg,  mu_g = (mu_N)_g,

and the cavity's scaled precision is d_g I_g / A'^-1_gg = I_g / (1 - d_g I_g) on P, without cancellation
(``small_n._Kernel.cavity``), 1 / A'^-1_gg - t_g on N. A site change moves W by a rank one: t_g -> t' on P (delta =
1/t' - 1/t) gives W' = W - delta w_g w_g' / (1 + delta I_g) with w_g = W x_g, so every I_k moves by -delta (x_k' w_g)^2 /
(1 + delta I_g), one n' x groups product; M, Q, S^-1, u and z follow by Sherman-Morrison (M <- M - c a a' with a = M x_g,
c = delta / (1 + delta x_g' a), and so on), two n'^2 products per step. On N, S_gg changes alone. A site crossing the
bulk boundary is moved between the sets outside the compiled loop (``SequentialSweep``).
"""

from __future__ import annotations

import numba
import numpy as np
from scipy import linalg

from sv_pgs._typing import F64Array, I64Array
from sv_pgs.alias_groups import GroupPriors, tilted_sum

_EPSILON = float(np.finfo(np.float64).eps)
_HALF_PRECISION = _EPSILON**0.5

# The compiled loop's events: the sweep ended; a group crosses into the rest; a group crosses into the bulk.
_DONE = 0
_TO_REST = 1
_TO_BULK = 2


@numba.njit(cache=True, error_model="numpy")
def _rest_mean(rest_count, rest_rows, rows, rest_score, solved, rest_images, schur_inverse, rest_mean, combined):
    """mu_N = S^-1 (b_N - X_N' z) and w = z + Q mu_N, in place."""
    dimension = solved.shape[0]
    residual = np.empty(rest_count)
    for slot in range(rest_count):
        residual[slot] = rest_score[slot] - np.dot(rows[rest_rows[slot]], solved)
    for slot in range(rest_count):
        value = 0.0
        for other in range(rest_count):
            value += schur_inverse[slot, other] * residual[other]
        rest_mean[slot] = value
    for index in range(dimension):
        combined[index] = solved[index]
    for slot in range(rest_count):
        for index in range(dimension):
            combined[index] += rest_images[slot, index] * rest_mean[slot]


@numba.njit(cache=True, error_model="numpy")
def _proper_everywhere(group, informed, projection, factor, precision, is_rest, rest_slot, schur_diagonal, largest, noise):
    """Whether every other group's cavity stays finite after the step (module docstring: I_k - factor (x_k' w)^2 on P,
    the new (S^-1)_kk on N, with 1 + V_max P > 0 at the group's largest variance)."""
    for k in range(informed.shape[0]):
        if k == group:
            continue
        if is_rest[k]:
            scaled = 1.0 / schur_diagonal[rest_slot[k]] - precision[k]
        else:
            new_informed = informed[k] - factor * projection[k] * projection[k]
            scaled = new_informed / (1.0 - new_informed / precision[k])
        cavity = scaled / noise
        if cavity < 0.0 and not 1.0 + largest[k] * cavity > 0.0:
            return False
        if not np.isfinite(cavity):
            return False
    return True


@numba.njit(cache=True, error_model="numpy")
def _proper_rest(group, precision, rest_rows, rest_count, schur_diagonal, largest, noise):
    """Whether every rest group's cavity stays finite after the step (the new (S^-1)_kk)."""
    for slot in range(rest_count):
        k = rest_rows[slot]
        if k == group:
            continue
        cavity = (1.0 / schur_diagonal[slot] - precision[k]) / noise
        if cavity < 0.0 and not 1.0 + largest[k] * cavity > 0.0:
            return False
        if not np.isfinite(cavity):
            return False
    return True


@numba.njit(cache=True, error_model="numpy")
def _bulk_proper(informed, t, largest, noise):
    """Whether a bulk group's cavity at I = ``informed`` and site t is finite (module docstring)."""
    cavity = informed / (1.0 - informed / t) / noise
    if cavity < 0.0 and not 1.0 + largest * cavity > 0.0:
        return False
    return np.isfinite(cavity)


@numba.njit(cache=True, error_model="numpy")
def _step(group, rows, squares, group_score, component_start, log_weight, log_variance, largest, noise, precision, shift_value, informed,
          is_rest, rest_slot, rest_rows, rest_count, inverse, rest_images, schur_inverse, image, solved, rest_mean, combined, target, half_precision,
          exact, w_out, own_residual):
    """One group's damped sequential update, in place. ``exact``: every other cavity is checked now, by its rank-one
    move, and ``informed`` follows; otherwise only the rest's are, the bulk's being checked by the caller from the
    step's w (written to ``w_out``) and the returned factor. Returns (status: 0 refused, 1 applied, 2 crossing of the
    bulk boundary with the new site in ``target`` and nothing applied, factor)."""
    dimension = image.shape[0]
    r = np.zeros(rest_count)
    solved_r = np.zeros(rest_count)
    schur_diagonal = np.empty(rest_count)
    t_old = precision[group]
    b_old = shift_value[group]
    x = rows[group]
    d_old = 0.0
    quadratic = 0.0
    explained = 0.0
    here = 0.0
    slot_g = -1
    a = solved
    column = np.zeros(rest_count)
    if not is_rest[group]:
        d_old = 1.0 / t_old
        a = np.dot(inverse, x)
        quadratic = np.dot(x, a)
        if rest_count:
            r = np.dot(rest_images, x)
        w = a.copy()
        for slot in range(rest_count):
            value = 0.0
            for other in range(rest_count):
                value += schur_inverse[slot, other] * r[other]
            solved_r[slot] = value
            explained += r[slot] * value
            for i in range(dimension):
                w[i] -= value * rest_images[slot, i]
        here = quadratic - explained
        marginal = d_old - d_old * d_old * here
        mean = d_old * (b_old - np.dot(x, combined))
        cavity_scaled = here / (1.0 - d_old * here)
    else:
        slot_g = rest_slot[group]
        marginal = schur_inverse[slot_g, slot_g]
        mean = rest_mean[slot_g]
        cavity_scaled = 1.0 / marginal - t_old
        column = schur_inverse[:, slot_g][:rest_count].copy()
        w = np.zeros(dimension)
        for slot in range(rest_count):
            for i in range(dimension):
                w[i] += column[slot] * rest_images[slot, i]
    cavity_precision = cavity_scaled / noise
    cavity_shift = mean / (noise * marginal) - (b_old - group_score[group]) / noise
    start_c, stop_c = component_start[group], component_start[group + 1]
    proper, _log_z, tilted_mean, tilted_variance = tilted_sum(log_weight[start_c:stop_c], log_variance[start_c:stop_c], cavity_precision, cavity_shift)
    if not proper or not tilted_variance > 0.0:
        own_residual[group] = np.inf
        return 0, 0.0
    target_t = noise * (1.0 / tilted_variance - cavity_precision)
    target_b = group_score[group] + noise * (tilted_mean / tilted_variance - cavity_shift)
    if not (np.isfinite(target_t) and np.isfinite(target_b)) or target_t == 0.0:
        own_residual[group] = np.inf
        return 0, 0.0
    projection = np.dot(rows, w) if exact else np.zeros(1)
    fraction = 1.0
    accepted = False
    factor = 0.0
    c = 0.0
    denominator = 1.0
    step = 0.0
    t_new = t_old
    b_new = b_old
    while fraction * abs(target_t - t_old) > half_precision * half_precision * abs(t_old) or fraction == 1.0:
        t_new = t_old + fraction * (target_t - t_old)
        b_new = b_old + fraction * (target_b - b_old)
        if t_new != 0.0:
            if not is_rest[group]:
                delta = 1.0 / t_new - d_old
                factor = delta / (1.0 + delta * here)
                c = delta / (1.0 + delta * quadratic)
                denominator = 1.0 - c * explained
                if denominator > 0.0:
                    for slot in range(rest_count):
                        schur_diagonal[slot] = schur_inverse[slot, slot] + c * solved_r[slot] * solved_r[slot] / denominator
                    if exact:
                        if _proper_everywhere(group, informed, projection, factor, precision, is_rest, rest_slot, schur_diagonal, largest, noise):
                            accepted = True
                            break
                    elif _proper_rest(group, precision, rest_rows, rest_count, schur_diagonal, largest, noise):
                        accepted = True
                        break
            else:
                step = t_new - t_old
                denominator = 1.0 + step * schur_inverse[slot_g, slot_g]
                if denominator > 0.0:
                    factor = -step / denominator
                    for slot in range(rest_count):
                        schur_diagonal[slot] = schur_inverse[slot, slot] - step * column[slot] * column[slot] / denominator
                    if exact:
                        if _proper_everywhere(group, informed, projection, factor, precision, is_rest, rest_slot, schur_diagonal, largest, noise):
                            accepted = True
                            break
                    elif _proper_rest(group, precision, rest_rows, rest_count, schur_diagonal, largest, noise):
                        accepted = True
                        break
        fraction *= 0.5
    if not accepted:
        own_residual[group] = np.inf
        return 0, 0.0
    # The residual this group's own step leaves: KL of its tilted law from q's new marginal (cavity times the new site),
    # zero where the full step was taken, and what no single site in the domain can remove where it was damped.
    q_variance = 1.0 / (cavity_precision + t_new / noise)
    q_mean = q_variance * (cavity_shift + (b_new - group_score[group]) / noise)
    own_residual[group] = 0.5 * (tilted_variance / q_variance + (q_mean - tilted_mean) ** 2 / q_variance - 1.0 + np.log(q_variance / tilted_variance))
    bulk_new = t_new > 0.0 and t_new >= half_precision * squares[group]
    if is_rest[group] == bulk_new:
        target[0] = t_new
        target[1] = b_new
        return 2, factor
    if exact:
        for k in range(informed.shape[0]):
            informed[k] -= factor * projection[k] * projection[k]
    else:
        for i in range(dimension):
            w_out[i] = w[i]
    if not is_rest[group]:
        d_new = 1.0 / t_new
        for slot in range(rest_count):
            for other in range(rest_count):
                schur_inverse[slot, other] += c * solved_r[slot] * solved_r[other] / denominator
        for slot in range(rest_count):
            for i in range(dimension):
                rest_images[slot, i] -= c * r[slot] * a[i]
        projection_u = np.dot(a, image)
        change = d_new * b_new - d_old * b_old
        for i in range(dimension):
            scaled = c * a[i]
            for k in range(dimension):
                inverse[i, k] -= scaled * a[k]
            image[i] += change * x[i]
            solved[i] += -c * a[i] * projection_u + change * (1.0 - c * quadratic) * a[i]
    else:
        for slot in range(rest_count):
            for other in range(rest_count):
                schur_inverse[slot, other] -= step * column[slot] * column[other] / denominator
    precision[group] = t_new
    shift_value[group] = b_new
    rest_score_update = np.empty(rest_count)
    for slot in range(rest_count):
        rest_score_update[slot] = shift_value[rest_rows[slot]]
    _rest_mean(rest_count, rest_rows, rows, rest_score_update, solved, rest_images, schur_inverse, rest_mean, combined)
    return 1, factor


@numba.njit(cache=True, error_model="numpy")
def _sweep(order, start, rows, squares, group_score, component_start, log_weight, log_variance, largest, noise, precision, shift_value,
           informed, is_rest, rest_slot, rest_rows, rest_count, inverse, rest_images, schur_inverse, image, solved, rest_mean, combined,
           target, half_precision, block_width, own_residual):
    """Sequential EP site updates over ``order`` from position ``start``, in place (module docstring), until the end or
    a crossing of the bulk boundary: returns (position, event, group, refused), with the group's new (t, b) in
    ``target`` for an event. ``precision`` holds each group's t, ``shift_value`` its b = X'y + sigma^2 nu, ``informed``
    I = x' W x per group.

    Blocked, with the per-step sweep's results. Checking every other group's cavity at each step is a G x n' product
    per step, memory-bound (32 s a sweep on 7,221 groups [real, ENSG00000144369.13]). So each block of
    ``block_width`` steps runs with the bulk's checks deferred (the rest's, O(|N|), stay per step), keeping each
    step's w and factor; at the block's end one G x n' x B product gives every bulk cavity after every step of the
    block exactly (I_k less the steps' factor (x_k' w)^2 in turn), and every intermediate state is checked as the
    per-step sweep checks it. A block with a state that is not proper is restored from its snapshot and run again
    step by step with the per-step checks, which is the sequential sweep itself; so is a step that crosses the bulk
    boundary."""
    dimension = image.shape[0]
    group_count = order.shape[0]
    refused = 0
    position = start
    w_block = np.zeros((block_width, dimension))
    factors = np.zeros(block_width)
    groups = np.zeros(block_width, dtype=np.int64)
    old_t = np.zeros(block_width)
    old_b = np.zeros(block_width)
    unused = np.zeros(dimension)
    while position < group_count:
        block_start = position
        saved_inverse = inverse.copy()
        saved_images = rest_images.copy()
        saved_schur = schur_inverse.copy()
        saved_image = image.copy()
        saved_solved = solved.copy()
        saved_rest_mean = rest_mean.copy()
        saved_combined = combined.copy()
        count = 0
        block_refused = 0
        crossing = False
        while position < group_count and count < block_width:
            group = order[position]
            t_before, b_before = precision[group], shift_value[group]
            status, factor = _step(group, rows, squares, group_score, component_start, log_weight, log_variance, largest, noise, precision,
                                   shift_value, informed, is_rest, rest_slot, rest_rows, rest_count, inverse, rest_images, schur_inverse, image,
                                   solved, rest_mean, combined, target, half_precision, False, w_block[count], own_residual)
            if status == 2:
                crossing = True
                break
            if status == 0:
                block_refused += 1
            else:
                groups[count] = group
                factors[count] = factor
                old_t[count] = t_before
                old_b[count] = b_before
                count += 1
            position += 1
        valid = True
        if count:
            projections = np.dot(rows, np.ascontiguousarray(w_block[:count].T))
            current = informed.copy()
            sites = precision.copy()
            for index in range(count):
                sites[groups[index]] = old_t[index]
            for index in range(count):
                group = groups[index]
                sites[group] = precision[group]
                for k in range(informed.shape[0]):
                    current[k] -= factors[index] * projections[k, index] * projections[k, index]
                for k in range(informed.shape[0]):
                    if k == group or is_rest[k]:
                        continue
                    if not _bulk_proper(current[k], sites[k], largest[k], noise):
                        valid = False
                        break
                if not valid:
                    break
            if valid:
                for k in range(informed.shape[0]):
                    informed[k] = current[k]
        if valid:
            refused += block_refused
            if not crossing:
                continue
            # The crossing step again with the per-step checks, from the verified state.
            group = order[position]
            status, factor = _step(group, rows, squares, group_score, component_start, log_weight, log_variance, largest, noise, precision,
                                   shift_value, informed, is_rest, rest_slot, rest_rows, rest_count, inverse, rest_images, schur_inverse, image,
                                   solved, rest_mean, combined, target, half_precision, True, unused, own_residual)
            if status == 2:
                return position, _TO_BULK if is_rest[group] else _TO_REST, group, refused
            if status == 0:
                refused += 1
            position += 1
            continue
        # Restore the block and run it step by step.
        inverse[:, :] = saved_inverse
        rest_images[:, :] = saved_images
        schur_inverse[:, :] = saved_schur
        image[:] = saved_image
        solved[:] = saved_solved
        rest_mean[:] = saved_rest_mean
        combined[:] = saved_combined
        for index in range(count):
            precision[groups[index]] = old_t[index]
            shift_value[groups[index]] = old_b[index]
        block_end = position
        position = block_start
        while position < block_end:
            group = order[position]
            status, factor = _step(group, rows, squares, group_score, component_start, log_weight, log_variance, largest, noise, precision,
                                   shift_value, informed, is_rest, rest_slot, rest_rows, rest_count, inverse, rest_images, schur_inverse, image,
                                   solved, rest_mean, combined, target, half_precision, True, unused, own_residual)
            if status == 2:
                return position, _TO_BULK if is_rest[group] else _TO_REST, group, refused
            if status == 0:
                refused += 1
            position += 1
    return group_count, _DONE, -1, refused


class SequentialSweep:
    """Sequential EP sweeps over the alias groups' sums at fixed priors and noise (module docstring).

    ``rows`` is the design over the groups in the covariates' complement (groups x n'), ``squares`` each group's
    ||x_g||^2, ``score`` each group's x_g'y, ``priors`` the groups' induced priors (``alias_groups.group_priors``).
    ``run`` updates ``precision`` (each group's scaled site t) and ``shift`` (nu) in place and returns how many updates
    were refused (module docstring), or None where the state at the sweep's start is not positive definite."""

    def __init__(self, rows: F64Array, squares: F64Array, score: F64Array, priors: GroupPriors, noise: float) -> None:
        self.rows = np.ascontiguousarray(rows, dtype=np.float64)
        self.squares = np.asarray(squares, dtype=np.float64)
        self.group_score = np.asarray(score, dtype=np.float64)
        self.priors = priors
        with np.errstate(over="ignore"):
            self.largest = np.exp(priors.largest_log_variance())
        self.noise = float(noise)
        self.dimension = int(self.rows.shape[1])
        # A block's deferred checks (``_sweep``) cost one G x n' x B product; B = sqrt(n') columns make it a matrix
        # product rather than B matrix-vector products, and a restored block repeats at most that many steps.
        self.block_width = max(1, int(np.sqrt(self.dimension)))
        self.clusters: list[I64Array] = []
        self.coupling: list[F64Array] = []
        self.clustered = np.zeros(self.rows.shape[0], dtype=bool)
        self.checked_largest = self.largest

    def set_clusters(self, clusters: list[I64Array], coupling: list[F64Array] | None = None) -> None:
        """Joint sites: each cluster's groups share one Gaussian site on their sums, its diagonal in the groups'
        ``precision`` (t) and its off-diagonal part (scaled, sigma^2 Lambda) in ``coupling`` (zero where None: the
        groups' own sites, the start of a newly joined cluster). A cluster's groups stay in the rest N, whose Schur
        complement S = T_N + X_N' M X_N takes the off-diagonal blocks as they are; the per-group sweep skips them and
        its domain check leaves their cavities to the cluster's own (``cluster_cavity``)."""
        self.clusters = [np.asarray(groups, dtype=np.int64) for groups in clusters]
        self.coupling = [np.zeros((groups.shape[0], groups.shape[0])) for groups in self.clusters] if coupling is None else [
            np.array(block, dtype=np.float64) for block in coupling]
        self.clustered = np.zeros(self.rows.shape[0], dtype=bool)
        for groups in self.clusters:
            self.clustered[groups] = True
        # 1 + 0 P > 0 always: a clustered group's scalar cavity is not its cavity.
        self.checked_largest = np.where(self.clustered, 0.0, self.largest)

    def _build(self, precision: F64Array, shift: F64Array) -> bool:
        """The state at these sites from scratch; False where it is not positive definite."""
        t = precision
        self.shift_value = self.group_score + self.noise * shift
        self.is_rest = ~((t > 0.0) & (t >= _HALF_PRECISION * self.squares)) | self.clustered
        bulk_inverse = np.where(self.is_rest, 0.0, 1.0 / np.where(self.is_rest, 1.0, t))
        kernel = self.rows.T @ (self.rows * bulk_inverse[:, None])
        kernel[np.diag_indices_from(kernel)] += 1.0
        upper = linalg.cholesky(0.5 * (kernel + kernel.T), lower=False, check_finite=False)
        inverse = linalg.cho_solve((upper, False), np.eye(self.dimension), check_finite=False)
        self.inverse = np.ascontiguousarray(0.5 * (inverse + inverse.T))
        self.image = self.rows.T @ (bulk_inverse * self.shift_value)
        self.solved = self.inverse @ self.image
        return self._set_rest(np.flatnonzero(self.is_rest), precision)

    def _set_rest(self, rest: I64Array, precision: F64Array) -> bool:
        """Q', S^-1, mu_N, w and every group's I = x' W x for the rest ``rest`` at the current M and z; False where S
        is not positive definite."""
        self.rest_rows = np.asarray(rest, dtype=np.int64)
        self.rest_slot = np.full(self.rows.shape[0], -1, dtype=np.int64)
        self.rest_slot[self.rest_rows] = np.arange(self.rest_rows.shape[0])
        columns = self.rows[self.rest_rows]
        self.rest_images = np.ascontiguousarray(columns @ self.inverse)
        schur = columns @ self.rest_images.T
        schur[np.diag_indices_from(schur)] += precision[self.rest_rows]
        for groups, block in zip(self.clusters, self.coupling):
            slots = self.rest_slot[groups]
            schur[np.ix_(slots, slots)] += block
        if schur.size:
            try:
                upper = linalg.cholesky(0.5 * (schur + schur.T), lower=False, check_finite=False)
            except np.linalg.LinAlgError:
                return False
            inverse = linalg.cho_solve((upper, False), np.eye(schur.shape[0]), check_finite=False)
            self.schur_inverse = np.ascontiguousarray(0.5 * (inverse + inverse.T))
        else:
            self.schur_inverse = np.zeros((0, 0))
        self.rest_mean = np.empty(self.rest_rows.shape[0])
        self.combined = np.empty(self.dimension)
        _rest_mean(self.rest_rows.shape[0], self.rest_rows, self.rows, self.shift_value[self.rest_rows].copy(), self.solved, self.rest_images,
                   self.schur_inverse, self.rest_mean, self.combined)
        # I_g = x_g' W x_g, W = M - Q S^-1 Q' (Q' = rest_images).
        whitened = self.rows @ self.inverse
        informed = np.einsum("ij,ij->i", whitened, self.rows)
        if self.rest_rows.size:
            coupling = self.rows @ self.rest_images.T
            informed -= np.einsum("ij,jk,ik->i", coupling, self.schur_inverse, coupling)
        self.informed = informed
        return True

    def _move_bulk(self, group: int, delta: float, change: float) -> None:
        """K <- K + delta x_g x_g' with u's term changed by ``change`` (d'b' - d b), on M, u and z."""
        x = self.rows[group]
        a = self.inverse @ x
        quadratic = float(x @ a)
        c = delta / (1.0 + delta * quadratic)
        projection = float(a @ self.image)
        self.inverse -= c * np.outer(a, a)
        self.image += change * x
        self.solved += -c * a * projection + change * (1.0 - c * quadratic) * a

    def run(self, precision: F64Array, shift: F64Array, order: I64Array) -> int | None:
        """One sweep over the unclustered groups in ``order``, updating ``precision`` (t) and ``shift`` (nu) in place."""
        order = np.ascontiguousarray(order[~self.clustered[order]], dtype=np.int64)
        # Each group's residual after its own step this sweep (``_step``): inf where refused, nan where not swept.
        self.own_residual = np.full(self.rows.shape[0], np.nan)
        if not self._build(precision, shift):
            return None
        target = np.empty(2)
        refused = 0
        position = 0
        priors = self.priors
        while True:
            position, event, group, count = _sweep(
                order, position, self.rows, self.squares, self.group_score, priors.component_start, priors.log_weight, priors.log_variance,
                self.checked_largest, self.noise, precision, self.shift_value, self.informed, self.is_rest, self.rest_slot, self.rest_rows,
                self.rest_rows.shape[0], self.inverse, self.rest_images, self.schur_inverse, self.image, self.solved, self.rest_mean,
                self.combined, target, _HALF_PRECISION, self.block_width, self.own_residual,
            )
            refused += count
            if event == _DONE:
                shift[:] = (self.shift_value - self.group_score) / self.noise
                return refused
            t_old, b_old = float(precision[group]), float(self.shift_value[group])
            t_new, b_new = float(target[0]), float(target[1])
            rest = list(self.rest_rows)
            if event == _TO_REST:
                self._move_bulk(group, -1.0 / t_old, -b_old / t_old)
                rest.append(group)
            else:
                rest.remove(group)
                self._move_bulk(group, 1.0 / t_new, b_new / t_new)
            precision[group], self.shift_value[group] = t_new, b_new
            self.is_rest[group] = event == _TO_REST
            if not self._set_rest(np.asarray(rest, dtype=np.int64), precision):
                # The crossing site leaves A' indefinite: it keeps its old site, and the state is rebuilt there.
                precision[group], self.shift_value[group] = t_old, b_old
                refused += 1
                shift[:] = (self.shift_value - self.group_score) / self.noise
                if not self._build(precision, shift):
                    return None
            position += 1

    def cluster_cavity(self, index: int, precision: F64Array, shift: F64Array) -> tuple[F64Array, F64Array, F64Array, F64Array] | None:
        """Cluster ``index``'s cavity on its sums (unscaled precision matrix, shift) and q's marginal there (mean,
        covariance), from a fresh build: Sigma_C = sigma^2 (S^-1)_CC, mu_C = mu_N[C], and the cavity is Sigma_C^-1
        less the cluster's site."""
        if not self._build(precision, shift):
            return None
        groups = self.clusters[index]
        slots = self.rest_slot[groups]
        covariance = self.noise * self.schur_inverse[np.ix_(slots, slots)]
        mean = self.rest_mean[slots]
        inverse = np.linalg.inv(covariance)
        site = (np.diag(precision[groups]) + self.coupling[index]) / self.noise
        return inverse - site, inverse @ mean - shift[groups], mean, covariance

    def current_cluster_cavity(self, index: int, precision: F64Array, shift: F64Array) -> tuple[F64Array, F64Array, F64Array, F64Array]:
        """``cluster_cavity`` at the state as it stands (after ``run`` or ``step_cluster``), with no rebuild."""
        groups = self.clusters[index]
        slots = self.rest_slot[groups]
        covariance = self.noise * self.schur_inverse[np.ix_(slots, slots)]
        mean = self.rest_mean[slots]
        inverse = np.linalg.inv(covariance)
        site = (np.diag(precision[groups]) + self.coupling[index]) / self.noise
        return inverse - site, inverse @ mean - shift[groups], mean, covariance

    def step_cluster(self, index: int, site_precision: F64Array, site_shift: F64Array, precision: F64Array, shift: F64Array) -> bool:
        """Cluster ``index``'s site moved to (``site_precision``, ``site_shift``), unscaled, where every cavity stays
        inside EP's domain, updating the state in place and returning True; the state is kept otherwise.

        The cluster's groups are in the rest N, so the move is S <- S + E D E' on their slots (D the scaled site's
        change, m x m): with U = S^-1 E and V = E'S^-1 E, S^-1 <- S^-1 - U (I + D V)^-1 D U' (Woodbury), and S stays
        positive definite where I + L'DL is (V = LL'). Every group's I = x'W x moves by y' (I + D V)^-1 D y with
        y = U'Q x (one G x n' x m product), which gives every bulk cavity; the rest's and the clusters' come from the
        new S^-1 itself."""
        groups = self.clusters[index]
        slots = self.rest_slot[groups]
        size = groups.shape[0]
        new_scaled = self.noise * np.asarray(site_precision, dtype=np.float64)
        old_scaled = np.diag(precision[groups]) + self.coupling[index]
        change = 0.5 * (new_scaled + new_scaled.T) - old_scaled
        images = self.schur_inverse[:, slots]
        block = images[slots]
        try:
            factor = np.linalg.cholesky(0.5 * (block + block.T))
        except np.linalg.LinAlgError:
            return False
        if not np.linalg.eigvalsh(np.eye(size) + factor.T @ change @ factor)[0] > 0.0:
            return False
        core = np.linalg.solve(np.eye(size) + change @ block, change)
        schur_inverse = self.schur_inverse - images @ core @ images.T
        schur_inverse = 0.5 * (schur_inverse + schur_inverse.T)
        projected = self.rows @ (self.rest_images.T @ images)
        informed = self.informed + np.einsum("ij,jk,ik->i", projected, core, projected)
        bulk = ~self.is_rest
        with np.errstate(divide="ignore", invalid="ignore"):
            scaled = informed[bulk] / (1.0 - informed[bulk] / precision[bulk])
        cavity = scaled / self.noise
        if not np.all(np.isfinite(cavity)) or np.any((cavity < 0.0) & ~(1.0 + self.largest[bulk] * cavity > 0.0)):
            return False
        trial_t = precision.copy()
        trial_t[groups] = np.diag(new_scaled)
        single_rest = self.rest_rows[~self.clustered[self.rest_rows]]
        if single_rest.size:
            with np.errstate(divide="ignore"):
                rest_cavity = (1.0 / np.diag(schur_inverse)[self.rest_slot[single_rest]] - trial_t[single_rest]) / self.noise
            if not np.all(np.isfinite(rest_cavity)) or np.any((rest_cavity < 0.0) & ~(1.0 + self.largest[single_rest] * rest_cavity > 0.0)):
                return False
        couplings = [block.copy() for block in self.coupling]
        couplings[index] = new_scaled - np.diag(np.diag(new_scaled))
        for other, members in enumerate(self.clusters):
            member_slots = self.rest_slot[members]
            try:
                inverse = np.linalg.inv(self.noise * schur_inverse[np.ix_(member_slots, member_slots)])
            except np.linalg.LinAlgError:
                return False
            cavity_precision = inverse - (np.diag(trial_t[members]) + couplings[other]) / self.noise
            with np.errstate(divide="ignore"):
                bound = np.diag(1.0 / self.largest[members])
            if not np.linalg.eigvalsh(0.5 * (cavity_precision + cavity_precision.T) + bound)[0] > 0.0:
                return False
        self.schur_inverse = np.ascontiguousarray(schur_inverse)
        self.informed = informed
        precision[groups] = np.diag(new_scaled)
        self.coupling[index] = couplings[index]
        shift[groups] = site_shift
        self.shift_value[groups] = self.group_score[groups] + self.noise * np.asarray(site_shift, dtype=np.float64)
        _rest_mean(self.rest_rows.shape[0], self.rest_rows, self.rows, self.shift_value[self.rest_rows].copy(), self.solved, self.rest_images,
                   self.schur_inverse, self.rest_mean, self.combined)
        return True

    def set_cluster_site(self, index: int, site_precision: F64Array, site_shift: F64Array, precision: F64Array, shift: F64Array) -> None:
        """Cluster ``index``'s site (unscaled precision matrix and shift on its sums), into ``precision``, ``shift``
        and the coupling."""
        groups = self.clusters[index]
        scaled = self.noise * np.asarray(site_precision, dtype=np.float64)
        precision[groups] = np.diag(scaled)
        self.coupling[index] = scaled - np.diag(np.diag(scaled))
        shift[groups] = site_shift

    def valid(self, precision: F64Array, shift: F64Array) -> bool:
        """Whether the state is inside EP's domain: q definite, every unclustered group's cavity with 1 + V_max P > 0,
        every cluster's cavity with Lambda + diag(1 / V_max) positive definite."""
        cavities = self.cavities(precision, shift)
        if cavities is None:
            return False
        cavity_precision, _cavity_shift = cavities
        alone = ~self.clustered
        if not np.all(np.isfinite(cavity_precision[alone])) or np.any((cavity_precision[alone] < 0.0) & ~(1.0 + self.largest[alone] * cavity_precision[alone] > 0.0)):
            return False
        for index, groups in enumerate(self.clusters):
            cavity = self.cluster_cavity(index, precision, shift)
            if cavity is None:
                return False
            with np.errstate(divide="ignore"):
                bound = np.diag(1.0 / self.largest[groups])
            if not np.linalg.eigvalsh(cavity[0] + bound)[0] > 0.0:
                return False
        return True

    def cavities(self, precision: F64Array, shift: F64Array) -> tuple[F64Array, F64Array] | None:
        """Every group's cavity (unscaled precision, shift) at these sites, from a fresh build."""
        if not self._build(precision, shift):
            return None
        cavity_precision = np.empty(precision.shape[0])
        cavity_shift = np.empty(precision.shape[0])
        bulk = ~self.is_rest
        d = np.where(bulk, 1.0 / np.where(bulk, precision, 1.0), 0.0)
        marginal = np.where(bulk, d - d * d * self.informed, 0.0)
        mean = np.where(bulk, d * (self.shift_value - self.rows @ self.combined), 0.0)
        scaled = np.where(bulk, self.informed / (1.0 - d * self.informed), 0.0)
        rest = self.rest_rows
        if rest.size:
            marginal[rest] = np.diag(self.schur_inverse)
            mean[rest] = self.rest_mean
            scaled[rest] = 1.0 / marginal[rest] - precision[rest]
        cavity_precision[:] = scaled / self.noise
        cavity_shift[:] = mean / (self.noise * marginal) - shift
        return cavity_precision, cavity_shift

    def mean(self, precision: F64Array, shift: F64Array) -> F64Array | None:
        """The Gaussian's mean over the groups' sums at these sites."""
        if not self._build(precision, shift):
            return None
        bulk = ~self.is_rest
        d = np.where(bulk, 1.0 / np.where(bulk, precision, 1.0), 0.0)
        values = np.where(bulk, d * (self.shift_value - self.rows @ self.combined), 0.0)
        values[self.rest_rows] = self.rest_mean
        return values
