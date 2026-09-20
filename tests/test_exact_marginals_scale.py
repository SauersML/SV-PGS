import numpy as np
import pytest

from sv_pgs.exact_marginals_scale import NotCertified, exact_dual_cost, exact_marginals


def _extended_inverse(matrix):
    """Gauss-Jordan with partial pivoting in extended precision: an independent reference for small cases."""
    size = matrix.shape[0]
    work = np.concatenate([matrix.astype(np.longdouble), np.eye(size, dtype=np.longdouble)], axis=1)
    for column in range(size):
        pivot = column + int(np.argmax(np.abs(work[column:, column])))
        work[[column, pivot]] = work[[pivot, column]]
        work[column] /= work[column, column]
        others = np.arange(size) != column
        work[others] -= np.outer(work[others, column], work[column])
    return work[:, size:]


def _reference(design, precision):
    """diag(A^-1) and the bulk diagonal diag(K_S^-1), K_S = I + Xt_S D_S Xt_S' over the sites with Pi > 0."""
    extended = design.astype(np.longdouble)
    values = precision.astype(np.longdouble)
    variances = np.diag(_extended_inverse(extended.T @ extended + np.diag(values)))
    bulk = values > 0
    kernel = np.eye(design.shape[0], dtype=np.longdouble) + (extended[:, bulk] / values[bulk]) @ extended[:, bulk].T
    return variances, np.diag(_extended_inverse(kernel))


def _blocks(design, widths):
    edges = np.cumsum([0, *widths])

    def blocks():
        return [(np.arange(start, stop), design[:, start:stop]) for start, stop in zip(edges[:-1], edges[1:])]

    return blocks


def _problem(seed, samples, variants, negative=0, zero=0):
    generator = np.random.default_rng(seed)
    design = generator.binomial(2, generator.uniform(0.05, 0.5, variants), size=(samples, variants)).astype(np.float64)
    design = (design - design.mean(axis=0)) * generator.uniform(0.2, 1.0, samples)[:, None]
    precision = np.exp(generator.normal(4.0, 1.5, variants))
    chosen = generator.choice(variants, negative + zero, replace=False)
    precision[chosen[:zero]] = 0.0
    gram_diagonal = (design * design).sum(axis=0)
    precision[chosen[zero:]] = -generator.uniform(0.005, 0.02, negative) * gram_diagonal[chosen[zero:]]
    return design, precision


@pytest.mark.parametrize("negative,zero", [(0, 0), (3, 2)])
def test_variances_and_bulk_diagonal_match_the_extended_precision_inverse_within_the_certificate(negative, zero):
    design, precision = _problem(1, 45, 70, negative, zero)
    result = exact_marginals(_blocks(design, [30, 25, 15]), precision, design.shape[0], bulk_diagonal=True, identity_block=17)
    variances, diagonal = _reference(design, precision)
    variance_error = np.abs(result.variances - variances.astype(np.float64))
    diagonal_error = np.abs(result.bulk_diagonal - diagonal.astype(np.float64))
    assert np.all(variance_error <= result.variance_bound)
    assert np.all(diagonal_error <= result.bulk_diagonal_bound)
    assert np.all(result.variance_bound <= np.abs(variances.astype(np.float64)) * 1e-6)
    assert np.all(result.bulk_diagonal_bound <= diagonal.astype(np.float64) * 1e-6)
    assert np.array_equal(result.resolved, np.flatnonzero(precision <= 0))


def test_named_resolved_sites_give_the_same_marginals():
    design, precision = _problem(2, 40, 60)
    plain = exact_marginals(_blocks(design, [60]), precision, design.shape[0])
    named = exact_marginals(_blocks(design, [60]), precision, design.shape[0], resolved=np.array([3, 17, 41]))
    assert np.all(np.abs(plain.variances - named.variances) <= plain.variance_bound + named.variance_bound)


def test_the_block_partition_does_not_change_the_answer_beyond_the_bound():
    design, precision = _problem(3, 50, 90, negative=2)
    one = exact_marginals(_blocks(design, [90]), precision, design.shape[0], bulk_diagonal=True)
    many = exact_marginals(_blocks(design, [7, 40, 1, 42]), precision, design.shape[0], bulk_diagonal=True, identity_block=11)
    assert np.all(np.abs(one.variances - many.variances) <= one.variance_bound + many.variance_bound)
    assert np.all(np.abs(one.bulk_diagonal - many.bulk_diagonal) <= one.bulk_diagonal_bound + many.bulk_diagonal_bound)


def test_a_core_that_is_not_positive_definite_is_refused():
    design, precision = _problem(4, 30, 40)
    precision[5] = -10 * (design[:, 5] ** 2).sum() - 1e3
    with pytest.raises(NotCertified):
        exact_marginals(_blocks(design, [40]), precision, design.shape[0])


def test_blocks_must_cover_every_column():
    design, precision = _problem(5, 20, 30)
    with pytest.raises(ValueError):
        exact_marginals(lambda: [(np.arange(20), design[:, :20])], precision, design.shape[0])


def test_cost_model_counts_the_two_passes_and_the_factor():
    cost = exact_dual_cost(50_000, 500_000, bulk_diagonal=True)
    assert cost["formation"] == cost["forward_solves"] == 50_000.0 ** 2 * 500_000
    assert cost["factor"] == cost["diagonal_of_inverse"] == 50_000.0 ** 3 / 3
    assert cost["resident_bytes"] == 8 * 50_000.0 ** 2
