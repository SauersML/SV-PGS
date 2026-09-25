"""The fused device kernels (``engine_kernels``) against the host engine, and against quadrature.

Every tolerance is a first-order forward-error bound in units u = eps/2 (Higham's gamma_n = n u / (1 - n u)), carried
from the inputs through each side's own operations: one unit for each rounded operation, and ``_TRANSCENDENTAL`` for
exp, log and log1p. ``_class_bounds`` bounds each node's quantities for both formulations at once (the device's
products e^(log u) e^t, q r and v r against the host's exp(log u + t), 1/(1 + 1/q) and 1/(1/v + P), and its weights
e^(log pi + a/2 - peak) sqrt(r) against the host's e^(log pi - log1p(q)/2 + a/2 - peak)); the reductions
add the larger of the two sides' summation errors (the device's running log-sum-exp and Welford updates, the host's
two-pass sums). Each side is within its bound of the exact value, so the two agree within twice it.
"""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
from scipy.special import log_softmax

from sv_pgs import engine_kernels
from sv_pgs.compute_budget import _try_import_cupy
from sv_pgs.scale_mixture_ep import (
    Cavity,
    _components,
    _data_objective,
    _data_value,
    _kernel_terms,
    class_log_density,
    log_scale,
    tilted_moments,
)
from tests.test_scale_mixture_ep import (
    _WORKING_BYTES,
    _hyperparameters,
    _problem,
    assert_tilted_moments_match_quadrature,
)

cupy = _try_import_cupy()
pytestmark = pytest.mark.skipif(cupy is None, reason="needs a CUDA device")

_UNIT = np.finfo(np.float64).eps / 2.0
# exp, log and log1p: within 4 ulp on the host (numpy's AVX-512 float64 loops, from SVML) and 1 ulp on the device
# (CUDA's double-precision math library); one ulp is 2 u.
_TRANSCENDENTAL = 8.0 * _UNIT
_SUBNORMAL = np.finfo(np.float64).smallest_subnormal
_LARGEST = np.finfo(np.float64).max


def _gamma(count: float) -> float:
    return count * _UNIT / (1.0 - count * _UNIT)


def _class_bounds(log_density, scales, grid, class_rows, precision, shift):
    """Per class: its rows and, per node, the host's values and bounds on either side's error in them.

    With q = vP, r = 1/(1 + q), qr = q r, c = v r and a = h^2 c, the relative errors are those of v (the argument's
    rounding amplified by exp, or the product of two exponentials, or a subnormal's spacing), of q, of 1 + q
    (u (1 + |q|) |r| from the sum, which a negative P amplifies, and q's), and of each product or reciprocal form.
    A node whose v overflowed at a finite log v takes both sides' limit, log sqrt(r) = -1/2 (log v + log P), c = 1/P
    and q r = 1 (``scale_mixture_ep._kernel_terms``, the audit's M18): its q and r are the limits 0 and 1 in the
    error terms below (no inf * 0), and its log(1 + q) error is the log-domain sum's. Every node takes its own
    variance v = u e^t (review-mathbugs N1: no flat kernel below the floor).
    """
    u, t = _UNIT, _TRANSCENDENTAL
    node_count = grid.shape[0]
    for class_position, rows in enumerate(class_rows):
        arguments = (log_density[class_position], scales[rows], grid, precision[rows], shift[rows])
        conditional, retained, ratio_retained, log_component, signal = _kernel_terms(*arguments)
        terms = _components(*arguments)
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            exponent = scales[rows][:, None] + grid[None, :]
            variance = np.exp(exponent)
            overflowed = ~np.isfinite(variance) & (precision[rows][:, None] > 0.0)
            q = np.where(overflowed, 0.0, np.abs(variance * precision[rows][:, None]))
            r, qr, a = np.abs(retained), np.abs(ratio_retained), np.abs(signal)
            log1p_signed = np.where(overflowed, exponent + np.log(precision[rows])[:, None], np.log1p(variance * precision[rows][:, None]))
            log1p_ratio = np.abs(log1p_signed)
            spacing = np.where(variance > 0.0, np.minimum(1.0, 4.0 * _SUBNORMAL / variance), 1.0)
            v_error = u * np.abs(exponent) + 2.0 * t + u + spacing
            q_error = v_error + u
            r_error = u * (1.0 + q) * r + qr * q_error + u + np.where(r > 0.0, np.minimum(1.0, 4.0 * _SUBNORMAL / r), 0.0)
            qr_error = np.maximum(q_error + r_error + u, r * (q_error + u) + 2.0 * u)
            c_error = np.maximum(v_error + r_error + u, r * (v_error + u) + u * (1.0 + q) * r + u)
            a_error = c_error + 2.0 * u
            component_error = (
                0.5 * (qr * q_error + t * log1p_ratio) + 0.5 * a * a_error
                + _gamma(2) * (np.abs(log_density[class_position])[None, :] + 0.5 * log1p_ratio + 0.5 * a)
            )
            ra = r * a
            first_error = 0.5 * (ra * (r_error + a_error + u) + qr * qr_error + u * np.abs(retained * signal - ratio_retained))
            twice_less_one = np.abs(2.0 * retained - 1.0)
            second_error = 0.5 * (
                ra * twice_less_one * (a_error + r_error + 2.0 * u) + ra * (2.0 * r * r_error + u * twice_less_one)
                + qr * r * (qr_error + r_error + u) + u * np.abs(signal * retained * (2.0 * retained - 1.0) - ratio_retained * retained)
            )
            # Below 1/max the host's 1/v and 1/q overflow, so its v r and q r are 0: wrong by all of their (subnormal) values.
            c_lost = np.where(variance * _LARGEST < 1.0, variance * r, 0.0)
            qr_lost = np.where(q * _LARGEST < 1.0, q * r, 0.0)
            a_lost = np.square(shift[rows])[:, None] * c_lost
            component_error = component_error + 0.5 * a_lost
            first_error = first_error + 0.5 * (r * a_lost + qr_lost)
            second_error = second_error + 0.5 * (r * twice_less_one * a_lost + r * qr_lost)
            finite = np.isfinite(log_component)
            component_error, first_error, second_error = (np.where(finite, error, 0.0) for error in (component_error, first_error, second_error))
            conditional_error = np.where(finite, np.abs(conditional) * c_error + c_lost, 0.0)
            log_normalizer = terms.log_normalizer
            responsibility = terms.responsibility
            # The device's exponent log pi + a/2 + log sqrt(r) (its peak taken over these): the log(1 + q) term's own
            # error (its argument's, or the log-domain sum's where v overflowed) and the sum's rounding.
            device_exponent = np.where(finite, log_density[class_position][None, :] + 0.5 * signal - 0.5 * log1p_signed, -np.inf)
            exponent_error = (
                0.5 * a * a_error + 0.5 * (qr * q_error + t * log1p_ratio)
                + u * (np.abs(log_density[class_position])[None, :] + 0.5 * a + 0.5 * log1p_ratio)
            )
            root_error = 0.0
            peak, device_peak = log_component.max(axis=1), device_exponent.max(axis=1)
            # A weight's error beyond the part common to a row, on either side: its log's, the subtraction of the peak, its
            # exp, and the device's rescales as the peak moves (an exp and a product each, at most one per node).
            host_weight_error = component_error + u * (peak[:, None] - log_component) + t
            device_weight_error = exponent_error + root_error + u * (device_peak[:, None] - device_exponent) + t
            weight_error = np.where(finite, np.maximum(host_weight_error, device_weight_error) + node_count * (t + u), 0.0)
            log_normalizer_error = (
                np.sum(responsibility * weight_error, axis=1) + _gamma(node_count)
                + t * np.maximum(np.abs(log_normalizer - peak), np.abs(log_normalizer - device_peak)) + u * np.abs(log_normalizer)
            )
            # A normalized responsibility's relative error, on either side (e^(log Z_k - log Z), or e^(exponent - log Z)
            # sqrt(r)), with a subnormal's spacing.
            responsibility_error = np.where(
                finite,
                weight_error + log_normalizer_error[:, None] + t
                + u * np.maximum(np.abs(log_component - log_normalizer[:, None]), np.abs(device_exponent - log_normalizer[:, None]))
                + np.minimum(1.0, 4.0 * _SUBNORMAL / np.maximum(responsibility, _SUBNORMAL)),
                0.0,
            )
        yield class_position, rows, SimpleNamespace(
            responsibility=responsibility, log_normalizer=log_normalizer, conditional=np.where(finite, np.abs(conditional), 0.0),
            first=np.where(finite, terms.first, 0.0), second=np.where(finite, terms.second, 0.0),
            conditional_error=conditional_error, first_error=first_error, second_error=second_error,
            log_normalizer_error=log_normalizer_error, responsibility_error=responsibility_error,
        )


def _assert_within(difference, bound, what: str) -> None:
    difference = np.abs(np.asarray(difference))
    bound = np.asarray(bound)
    ratio = np.max(np.divide(difference, bound, out=np.where(difference > 0.0, np.inf, 0.0), where=bound > 0.0))
    assert np.all(difference <= bound), f"{what}: |device - host| reaches {ratio:.3g} of the derived bound"


def _assert_tilted_parity(log_density, scales, grid, class_rows, precision, shift, device, host) -> None:
    """(log Z, mean, variance) from the device against the host's, row by row.

    The device's Welford mean of the conditional variances is within (K + 3) u m of its weights' mean: its steps are
    non-negative because c = v/(1 + vP) grows along the increasing lattice, so they sum to m; the running total adds
    gamma_2K. Its spread, Welford's M2 over the total, is within gamma_(8K + 8) (V + m^2)."""
    assert np.all(np.diff(grid) > 0.0)
    u = _UNIT
    node_count = grid.shape[0]
    for _class, rows, node in _class_bounds(log_density, scales, grid, class_rows, precision, shift):
        responsibility, conditional = node.responsibility, node.conditional
        first_moment = np.sum(responsibility * conditional, axis=1)
        centred = conditional - first_moment[:, None]
        spread = np.sum(responsibility * np.square(centred), axis=1)
        first_error = (
            np.sum(responsibility * (node.responsibility_error * (conditional + first_moment[:, None]) + node.conditional_error), axis=1)
            + _gamma(3 * node_count + 4) * first_moment
        )
        spread_error = (
            np.sum(responsibility * (node.responsibility_error * (np.square(centred) + spread[:, None]) + 2.0 * np.abs(centred) * node.conditional_error), axis=1)
            + _gamma(8 * node_count + 8) * (spread + np.square(first_moment))
        )
        row_shift = shift[rows]
        shift_square = np.square(row_shift)
        _assert_within(device[0][rows] - host[0][rows], 2.0 * node.log_normalizer_error, "log Z")
        _assert_within(device[1][rows] - host[1][rows], 2.0 * np.abs(row_shift) * (first_error + u * first_moment), "tilted mean")
        _assert_within(
            device[2][rows] - host[2][rows],
            2.0 * (first_error + shift_square * (spread_error + 3.0 * u * spread) + u * (first_moment + shift_square * spread)),
            "tilted variance",
        )


def _assert_tilted_moments_match(prior, hyperparameters, cavity, working_bytes=_WORKING_BYTES) -> None:
    host = tilted_moments(prior, hyperparameters, cavity, working_bytes)
    device = tilted_moments(prior, hyperparameters, cavity, working_bytes, cupy)
    _assert_tilted_parity(
        class_log_density(prior, hyperparameters.coefficients), log_scale(prior, hyperparameters.coefficients), prior.log_variance_grid,
        prior.class_rows, cavity.precision, cavity.shift,
        (device.log_normalizer, device.mean, device.variance), (host.log_normalizer, host.mean, host.variance),
    )


def _assert_objective_matches(prior, coefficients, cavity, working_bytes) -> None:
    """sum log Z, its gradient and Hessian in z, and sum |log Z| from the device against the host's.

    The device's density block is diag(sum D) - D'D - (sum D) pi' - pi (sum D)' with D = r - pi, the host's
    diag(sum r) - r'r - n (diag pi - pi pi'); with R = r + pi >= |D|, r both are bounded through R. The device's
    Welford mean of a non-monotone sequence x is within gamma_(8K + 4K^2) max |x| (each step's error, with the running
    total's, is a few units of a step no larger than 2 max |x|), and its M2 within gamma_(8K + 6) of itself plus four
    times that mean error times max |x|."""
    host = _data_objective(prior, coefficients, cavity, _WORKING_BYTES)
    device = _data_objective(prior, coefficients, cavity, working_bytes, cupy)
    u, t = _UNIT, _TRANSCENDENTAL
    node_count, variant_count = prior.grid_size, prior.variant_count
    welford = _gamma(8 * node_count + 4 * node_count**2)
    density = np.exp(class_log_density(prior, coefficients))
    scale_span = slice(prior.density_size, prior.density_size + prior.scale_size)
    gradient_bound = np.zeros_like(host.gradient)
    hessian_bound = np.zeros_like(host.hessian)
    value_bound = _gamma(variant_count) * host.magnitude
    for class_position, rows, node in _class_bounds(
        class_log_density(prior, coefficients), log_scale(prior, coefficients), prior.log_variance_grid, prior.class_rows, cavity.precision, cavity.shift,
    ):
        count = rows.shape[0]
        span = slice(class_position * node_count, (class_position + 1) * node_count)
        responsibility, weight_error = node.responsibility, node.responsibility_error
        mass = density[class_position]
        value_bound += float(np.sum(node.log_normalizer_error))
        with_mass = responsibility + mass[None, :]
        with_mass_sum = with_mass.sum(axis=0)
        summed_error = np.sum(responsibility * weight_error, axis=0) + _gamma(variant_count + 3) * responsibility.sum(axis=0) + (
            _gamma(variant_count + 3) + t
        ) * count * mass
        element_error = responsibility * weight_error + (_gamma(3) + t) * with_mass
        gradient_bound[span] = summed_error
        hessian_bound[span, span] = (
            np.diag(summed_error) + element_error.T @ with_mass + with_mass.T @ element_error
            + np.outer(summed_error, mass) + np.outer(mass, summed_error) + _gamma(variant_count + 4) * (with_mass.T @ with_mass)
            + (_gamma(4) + 2.0 * t) * (
                np.diag(with_mass_sum) + np.outer(with_mass_sum, mass) + np.outer(mass, with_mass_sum)
                + count * (np.diag(mass) + np.outer(mass, mass))
            )
        )
        first, second = node.first, node.second
        largest_first = np.abs(first).max(axis=1)
        mean_first = np.sum(responsibility * first, axis=1)
        mean_first_error = (
            np.sum(responsibility * (weight_error * (np.abs(first) + np.abs(mean_first)[:, None]) + node.first_error), axis=1)
            + welford * largest_first
        )
        design = np.abs(prior.scale_design[rows])
        gradient_bound[scale_span] += design.T @ mean_first_error + _gamma(variant_count) * (design.T @ np.abs(mean_first))
        centred = first - mean_first[:, None]
        centred_error = responsibility * ((weight_error + 2.0 * u) * np.abs(centred) + node.first_error + mean_first_error[:, None])
        cross_bound = centred_error.T @ design + _gamma(variant_count) * (np.abs(responsibility * centred).T @ design)
        hessian_bound[span, scale_span] = cross_bound
        hessian_bound[scale_span, span] = cross_bound.T
        first_spread = np.sum(responsibility * np.square(centred), axis=1)
        second_mean = np.sum(responsibility * second, axis=1)
        second_size = np.sum(responsibility * np.abs(second), axis=1)
        curvature_error = (
            np.sum(
                responsibility * (
                    weight_error * (np.square(centred) + first_spread[:, None] + np.abs(second) + np.abs(second_mean)[:, None])
                    + 2.0 * np.abs(centred) * (node.first_error + mean_first_error[:, None]) + node.second_error
                ),
                axis=1,
            )
            + _gamma(8 * node_count + 8) * (first_spread + second_size)
            + 4.0 * welford * np.square(largest_first) + welford * np.abs(second).max(axis=1)
        )
        curvature_size = curvature_error + _gamma(variant_count + 2) * (first_spread + second_size)
        hessian_bound[scale_span, scale_span] += design.T @ (curvature_size[:, None] * design)
    _assert_within(device.value - host.value, 2.0 * value_bound, "sum log Z")
    _assert_within(device.magnitude - host.magnitude, 2.0 * value_bound, "sum |log Z|")
    _assert_within(device.gradient - host.gradient, 2.0 * gradient_bound, "gradient")
    _assert_within(device.hessian - host.hessian, 2.0 * hessian_bound, "Hessian")
    _assert_within(_data_value(prior, coefficients, cavity, working_bytes, cupy) - host.value, 2.0 * value_bound, "sum log Z alone")


def _largest_variance(prior, hyperparameters) -> np.ndarray:
    """Each effect's largest kernel variance on the lattice (its top node's)."""
    return np.exp(log_scale(prior, hyperparameters.coefficients) + prior.log_variance_grid[-1])


def test_cuda_tilted_moments_match_quadrature():
    prior, cavity = _problem(variant_count=12, seed=1, node_count=9)
    hyperparameters = _hyperparameters(prior, 2)
    assert_tilted_moments_match_quadrature(prior, hyperparameters, cavity, tilted_moments(prior, hyperparameters, cavity, _WORKING_BYTES, cupy))


@pytest.mark.parametrize("seed", [5, 17])
def test_cuda_tilted_moments_match_the_host_to_rounding(seed):
    prior, cavity = _problem(variant_count=400, seed=seed)
    _assert_tilted_moments_match(prior, _hyperparameters(prior, seed + 1), cavity)


def test_cuda_chunking_leaves_every_row_unchanged():
    prior, cavity = _problem(variant_count=300, seed=7)
    hyperparameters = _hyperparameters(prior, 8)
    whole = tilted_moments(prior, hyperparameters, cavity, _WORKING_BYTES, cupy)
    # The smallest budget that holds a row: one row per launch.
    pieces = tilted_moments(prior, hyperparameters, cavity, engine_kernels._tilted_row_bytes(), cupy)
    for name in ("log_normalizer", "mean", "variance"):
        np.testing.assert_array_equal(getattr(pieces, name), getattr(whole, name))


@pytest.mark.parametrize("rows_per_chunk", [None, 7])
def test_cuda_objective_matches_the_host_to_rounding(rows_per_chunk):
    """With a whole class per chunk (batches of several rows and zero padding) and with seven rows per chunk (a batch each)."""
    prior, cavity = _problem(variant_count=500, seed=11)
    per_row = engine_kernels._objective_row_bytes(prior.grid_size, prior.scale_size) + engine_kernels._objective_batch_bytes(prior.grid_size, prior.scale_size)
    working_bytes = prior.variant_count * per_row if rows_per_chunk is None else rows_per_chunk * per_row
    _assert_objective_matches(prior, _hyperparameters(prior, 12).coefficients, cavity, working_bytes)


def test_cuda_negative_cavity_precisions_match_the_host_while_every_component_is_proper():
    prior, cavity = _problem(variant_count=240, seed=19)
    hyperparameters = _hyperparameters(prior, 20)
    # Every third effect at half the most negative precision that keeps 1 + v P > 0 at every node.
    precision = np.where(np.arange(prior.variant_count) % 3 == 0, -0.5 / _largest_variance(prior, hyperparameters), cavity.precision)
    negative = Cavity(precision=precision, shift=cavity.shift)
    _assert_tilted_moments_match(prior, hyperparameters, negative)
    _assert_objective_matches(prior, hyperparameters.coefficients, negative, _WORKING_BYTES)


def test_cuda_underflowing_and_overflowing_variances_take_the_hosts_values():
    generator = np.random.default_rng(29)
    rows, node_count = 90, 30
    grid = np.linspace(-5.0, 15.0, node_count)
    # log u where e^(log u) is subnormal, ordinary, and where e^(log u + t) overflows at the top nodes.
    scales = np.repeat(np.array([np.log(_SUBNORMAL) + 5.0, 0.0, np.log(np.finfo(np.float64).max) - 5.0]), rows // 3)
    precision = 1.0 + np.abs(generator.standard_normal(rows))
    shift = generator.standard_normal(rows) * np.sqrt(precision)
    log_density = log_softmax(generator.standard_normal((1, node_count)), axis=1)
    class_index = np.zeros(rows, dtype=np.int64)
    host_log_normalizer, host_mean, host_variance = np.empty(rows), np.empty(rows), np.empty(rows)
    everything = np.arange(rows)
    terms = _components(log_density[0], scales, grid, precision, shift)
    first_moment = np.sum(terms.responsibility * terms.conditional_variance, axis=1)
    host_log_normalizer[everything] = terms.log_normalizer
    host_mean[everything] = shift * first_moment
    host_variance[everything] = first_moment + np.square(shift) * np.sum(
        terms.responsibility * np.square(terms.conditional_variance - first_moment[:, None]), axis=1
    )
    device = engine_kernels.tilted_moments(cupy, class_index, log_density, scales, grid, precision, shift, _WORKING_BYTES)
    _assert_tilted_parity(
        log_density, scales, grid, [everything], precision, shift,
        tuple(cupy.asnumpy(values) for values in device), (host_log_normalizer, host_mean, host_variance),
    )


def test_cuda_an_improper_cavity_is_an_error_as_on_the_host():
    prior, cavity = _problem(variant_count=20, seed=23)
    hyperparameters = _hyperparameters(prior, 24)
    # Twice the most negative proper precision: 1 + v P = -1 at the top node.
    precision = np.where(np.arange(prior.variant_count) == 4, -2.0 / _largest_variance(prior, hyperparameters), cavity.precision)
    improper = Cavity(precision=precision, shift=cavity.shift)
    with pytest.raises(FloatingPointError):
        tilted_moments(prior, hyperparameters, improper, _WORKING_BYTES)
    with pytest.raises(FloatingPointError):
        tilted_moments(prior, hyperparameters, improper, _WORKING_BYTES, cupy)
    with pytest.raises(FloatingPointError):
        _data_objective(prior, hyperparameters.coefficients, improper, _WORKING_BYTES, cupy)


def test_cuda_kernel_allocations_inside_a_ledger_evict_its_caches_rather_than_run_out():
    """The EB kernels' device arrays are allocations like any other: inside a ledger scope (``memory_broker``) each
    goes through the ledger's allocator, which evicts an idle device cache (the resident genotype codes, the panel
    Grams) to make room before it would refuse. Here a cache fills the device's capacity to within a few kernel
    arrays, and the kernel call still runs, with the cache dropped (bench-sim scenario_004 [sim]: a 0.49 GB kernel
    allocation against 41.7 GB held on an A100-40GB, the resident codes among them, died before the ledger)."""
    from sv_pgs.compute_budget import ComputeBudget
    from sv_pgs.memory_broker import device_pool, memory_scope

    prior, cavity = _problem(variant_count=400, seed=9)
    hyperparameters = _hyperparameters(prior, 10)
    expected = tilted_moments(prior, hyperparameters, cavity, _WORKING_BYTES, cupy)
    pool = cupy.get_default_memory_pool()
    pool.free_all_blocks()
    device = int(cupy.cuda.runtime.getDevice())
    capacity = int(pool.total_bytes()) + (64 << 20)
    budget = ComputeBudget(
        device_kind="cuda", device_ids=(device,), device_names=("test",), device_bytes=(capacity,), device_compute_capabilities=((8, 0),),
        host_bytes=1 << 34, cpu_threads=1,
    )
    cache, dropped = {}, []
    with memory_scope(budget) as broker:
        cache["codes"] = cupy.zeros((64 << 20) - (1 << 12), dtype=cupy.uint8)
        assert broker.admit(device_pool(device), cache["codes"].nbytes, "stand-in resident codes", lambda: (dropped.append(1), cache.clear()), allocated=True)
        moments = tilted_moments(prior, hyperparameters, cavity, _WORKING_BYTES, cupy)
        assert dropped == [1] and broker.held(device_pool(device)) <= capacity
    for name in ("log_normalizer", "mean", "variance"):
        np.testing.assert_array_equal(getattr(moments, name), getattr(expected, name))


def test_the_line_values_stay_inside_a_tight_device_ledger():
    """``_line`` on the device under a ledger whose device pool leaves 32 MiB, with a host budget of 16 GiB: its chunks
    (fixed rows' kernels and the fused moving rows) are sized from the device's remainder, so no allocation is
    refused, and the values are the host's (bench-sim chr22: chunks sized from the host's budget were refused)."""
    from sv_pgs.compute_budget import ComputeBudget
    from sv_pgs.memory_broker import memory_scope
    from sv_pgs.scale_mixture_ep import _line, device_scope

    prior, cavity = _problem(variant_count=20000, seed=51, node_count=40)
    hyperparameters = _hyperparameters(prior, 52, log_smoothing=1.0)
    moving = 0.3 * np.random.default_rng(53).standard_normal(prior.coefficient_size)
    still = moving.copy()
    still[prior.coefficient_size - prior.scale_size :] = 0.0
    steps = np.array([-1.0, 0.0, 0.7])
    host_bytes = 1 << 34
    for direction in (moving, still):
        expected = _line(prior, hyperparameters.log_smoothing, hyperparameters.coefficients, direction, cavity, _WORKING_BYTES)(steps)
        pool = cupy.get_default_memory_pool()
        pool.free_all_blocks()
        held = int(pool.total_bytes())
        budget = ComputeBudget(
            device_kind="cuda", device_ids=(0,), device_names=("ledger test",), device_bytes=(held + (32 << 20),),
            device_compute_capabilities=((0, 0),), host_bytes=host_bytes, cpu_threads=1,
        )
        with memory_scope(budget), device_scope(cupy):
            got = _line(prior, hyperparameters.log_smoothing, hyperparameters.coefficients, direction, cavity, host_bytes)(steps)
        np.testing.assert_allclose(got, expected, rtol=1e-10, atol=1e-10 * float(np.max(np.abs(expected))))


def test_the_device_variant_derivatives_are_the_host_twins_within_a_tight_ledger():
    """``_variant_derivatives`` on the device (``engine_kernels.derivative_rows``), chunked from a ledger that leaves
    32 MiB on the device with a 16 GiB host budget, against its host twin."""
    from sv_pgs.compute_budget import ComputeBudget
    from sv_pgs.memory_broker import memory_scope
    from sv_pgs.scale_mixture_ep import _variant_derivatives, device_scope

    prior, cavity = _problem(variant_count=20000, seed=51, node_count=40)
    hyperparameters = _hyperparameters(prior, 52, log_smoothing=1.0)
    expected = _variant_derivatives(prior, hyperparameters.coefficients, cavity, _WORKING_BYTES)
    pool = cupy.get_default_memory_pool()
    pool.free_all_blocks()
    held = int(pool.total_bytes())
    budget = ComputeBudget(
        device_kind="cuda", device_ids=(0,), device_names=("ledger test",), device_bytes=(held + (32 << 20),),
        device_compute_capabilities=((0, 0),), host_bytes=1 << 34, cpu_threads=1,
    )
    with memory_scope(budget), device_scope(cupy):
        got = _variant_derivatives(prior, hyperparameters.coefficients, cavity, 1 << 34)
    for name in ("mean", "second", "variance", "variance_by_shift", "mean_by_precision", "variance_by_precision", "mean_by_log_scale",
                 "second_by_log_scale", "mean_by_density", "second_by_density"):
        values = getattr(expected, name)
        scale = float(np.max(np.abs(values)))
        np.testing.assert_allclose(getattr(got, name), values, rtol=1e-10, atol=1e-10 * scale, err_msg=name)
