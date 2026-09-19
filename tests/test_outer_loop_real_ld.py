"""The outer step's curvature on real LD, from the dense reference: why plain EP-EM is not the outer step.

Fixture ``data/pooled_chr22_w150_r12_v7.npz`` [semi-real: real 1kGP-haplotype LD, simulated effects]: 12 chr22 windows
of 150 variants x 50k people from bench-sim's v7 cohort (public 1kGP founder weights); its source, generator and
script hashes are in ``data/pooled_chr22_w150_r12_v7.provenance.json``. The windows are independent and share the
prior's coefficients x, so their union is one model with a block-diagonal likelihood; it is stated at c = 1 (the
genome-scaled objective c sum_r log Z_EP,r - x'S_w x / 2 is c times this one's with weights w / c: the same
maximizer and outer spectrum). The prior is the reference's (one class, no annotations, its Chebyshev mixing
quadrature). It carries the true prior, where plain EP-EM started, and plain EP-EM's stall with its EP sites.

What holds there, and what the engine's outer loop is built on:
1. At the true prior B + S is indefinite: the evidence has a saddle there, so the outer loop needs negative
   curvature handling from its start.
2. Plain EP-EM stalls where its M-step has nothing left to do (the fixed-cavity decrement is below the resolution)
   while A + S is not positive definite: the stall is not a maximum of the objective EP-EM maximizes. It is the null
   model (log Z_EP about 0, the prior's variance collapsed), where the data barely see the density's shape.
3. At both points B + S is not positive definite, so no certificate exists there; the engine's loop issues one
   only where B + S is.
At production signal this model's evidence peaks at the null boundary, so no evidence ordering is asserted there.

A second fixture, ``data/pooled_chr22_w150_r12_v7_x100.npz`` (same windows and provenance record beside it, 100x
production signal, penalty weights 1), has an interior maximum, and there the ordering is real: plain EP-EM rises
and then falls (it is not an ascent method), and the certified Newton-B point, where A + S and B + S are positive
definite and the decrement is below the resolution, is above every plain iterate by more than the resolution.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

import tests.ep_eb_reference as reference

_FIXTURE = Path(__file__).resolve().parent / "data" / "pooled_chr22_w150_r12_v7.npz"
# The evidence resolution a fit certifies for a scorer with 64 posterior draws, 1/(2K) nats.
_RESOLUTION = 1.0 / 128.0


def _windows(data):
    count = sum(1 for key in data.files if key.endswith("_likelihood_precision"))
    quadrature = reference.mixing_quadrature(
        float(data["mixing_lower"]), float(data["mixing_upper"]), int(data["mixing_degree"]), int(data["mixing_nodes_per_panel"])
    )
    windows = []
    for window in range(count):
        size = data[f"window{window}_linear_term"].shape[0]
        design = reference.ReferenceDesign(
            class_index=data[f"window{window}_class_index"].astype(np.int64),
            log_variance_offset=data[f"window{window}_log_variance_offset"],
            annotation_design=np.zeros((size, 0)),
            annotation_groups=(),
        )
        windows.append((reference.ReferencePrior(design, quadrature), data[f"window{window}_likelihood_precision"], data[f"window{window}_linear_term"]))
    return windows


def _curvatures(windows, point, states, weights):
    """A + S, B + S and the penalized gradient over the allowed coefficients, summed over the windows."""
    prior = windows[0][0]
    basis = reference.allowed_basis(prior, weights)
    penalty = reference.penalty_matrix(prior, weights)
    fixed = sum(-reference.cavity_log_marginal_hessian(window[0], point, state.cavity_precision, state.cavity_shift) for window, state in zip(windows, states))
    total = sum(reference.total_curvature(window[0], point, state) for window, state in zip(windows, states))
    gradient = sum(reference.cavity_log_marginal(window[0], point, state.cavity_precision, state.cavity_shift)[1] for window, state in zip(windows, states))
    return basis.T @ (fixed + penalty) @ basis, basis.T @ (total + penalty) @ basis, basis.T @ (gradient - penalty @ point)


def _not_positive_definite(matrix):
    """No Cholesky factor, with the smallest eigenvalue below minus the rounding scale n eps ||M||."""
    symmetric = 0.5 * (matrix + matrix.T)
    try:
        np.linalg.cholesky(symmetric)
        return False
    except np.linalg.LinAlgError:
        rounding = np.finfo(np.float64).eps * symmetric.shape[0] * float(np.linalg.norm(symmetric, 2))
        return bool(np.linalg.eigvalsh(symmetric)[0] < -rounding)


def _sites(windows, data, point, key):
    return [
        reference.site_state(prior, point, likelihood_precision, linear_term, data[f"window{window}_{key}site_precision"], data[f"window{window}_{key}site_shift"])
        for window, (prior, likelihood_precision, linear_term) in enumerate(windows)
    ]


def test_plain_ep_em_is_not_the_outer_step_on_real_ld():
    data = np.load(_FIXTURE)
    windows = _windows(data)
    weights = np.asarray(data["penalty_weights_at_c1"], dtype=np.float64)

    stall = np.asarray(data["plain_stall_point"], dtype=np.float64)
    stall_states = _sites(windows, data, stall, "stall_")
    assert all(state is not None for state in stall_states)
    stall_fixed, stall_total, stall_gradient = _curvatures(windows, stall, stall_states, weights)
    # (2) A stall: the M-step's own model (|A + S|) gains less than the resolution; yet A + S is not positive definite.
    values, vectors = np.linalg.eigh(0.5 * (stall_fixed + stall_fixed.T))
    magnitudes = np.maximum(np.abs(values), np.finfo(np.float64).eps * float(np.max(np.abs(values))))
    assert 0.5 * float(np.sum(np.square(vectors.T @ stall_gradient) / magnitudes)) <= _RESOLUTION
    assert _not_positive_definite(stall_fixed)
    # (3) No certificate at the stall.
    assert _not_positive_definite(stall_total)


    # (1) and (3) At the true prior, with EP solved there from the stall's sites, B + S is indefinite: a saddle.
    truth = np.asarray(data["truth_start"], dtype=np.float64)
    truth_states = []
    for prior, likelihood_precision, linear_term in windows:
        window = len(truth_states)
        start = reference.site_state(
            prior, truth, likelihood_precision, linear_term, data[f"window{window}_stall_site_precision"], data[f"window{window}_stall_site_shift"]
        )
        if start is None:
            start = reference.initial_sites(prior, truth, likelihood_precision, linear_term, np.ones(linear_term.shape[0]))
        truth_states.append(reference.solve_sites(prior, truth, likelihood_precision, linear_term, start))
    _truth_fixed, truth_total, _truth_gradient = _curvatures(windows, truth, truth_states, weights)
    assert _not_positive_definite(truth_total)


_STRONG_FIXTURE = Path(__file__).resolve().parent / "data" / "pooled_chr22_w150_r12_v7_x100.npz"


def _evidence(windows, point, states, weights):
    """E = sum_r log Z_EP,r - x'S x / 2, the pooled model's EP evidence at these sites."""
    penalty = reference.penalty_matrix(windows[0][0], weights)
    return sum(state.log_evidence for state in states) - 0.5 * float(point @ penalty @ point)


def test_plain_ep_em_is_not_monotone_and_the_certified_point_is_above_it_on_real_ld():
    data = np.load(_STRONG_FIXTURE)
    windows = _windows(data)
    weights = np.asarray(data["penalty_weights_at_c1"], dtype=np.float64)

    certified = np.asarray(data["certified_point"], dtype=np.float64)
    certified_states = _sites(windows, data, certified, "")
    assert all(state is not None for state in certified_states)
    certified_fixed, certified_total, certified_gradient = _curvatures(windows, certified, certified_states, weights)
    # A certified maximum: both curvatures positive definite, and the Newton-B decrement below the resolution.
    assert not _not_positive_definite(certified_fixed) and not _not_positive_definite(certified_total)
    np.linalg.cholesky(0.5 * (certified_total + certified_total.T))
    assert 0.5 * float(certified_gradient @ np.linalg.solve(certified_total, certified_gradient)) <= _RESOLUTION
    certified_value = _evidence(windows, certified, certified_states, weights)

    values = []
    for iterate in range(3):
        point = np.asarray(data[f"plain{iterate}_point"], dtype=np.float64)
        states = [
            reference.site_state(
                prior, point, likelihood_precision, linear_term,
                data[f"plain{iterate}_window{window}_site_precision"], data[f"plain{iterate}_window{window}_site_shift"],
            )
            for window, (prior, likelihood_precision, linear_term) in enumerate(windows)
        ]
        assert all(state is not None for state in states)
        values.append(_evidence(windows, point, states, weights))
    # Plain EP-EM is not an ascent method: its second step lowers the evidence by more than the resolution.
    assert values[2] < values[1] - _RESOLUTION
    # The certified point is above every plain iterate by more than the resolution.
    assert all(certified_value >= value + _RESOLUTION for value in values)
