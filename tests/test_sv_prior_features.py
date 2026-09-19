import numpy as np
import pytest
from scipy.special import digamma

from sv_pgs.sv_prior_features import block_tagging, ewens_theta, locus_frequency


def test_a_split_locus_carries_its_summed_frequency_on_every_rare_record():
    bubbles = np.array([4, 4, 4, 9, 9, 2])
    is_sv = np.array([True, True, True, True, True, False])
    frequency = np.array([2.0**-5, 2.0**-5, 2.0**-5, 0.25, 0.875, 0.5])
    locus = locus_frequency(bubbles, is_sv, frequency)
    np.testing.assert_array_equal(locus[:5], [3 * 2.0**-5, 3 * 2.0**-5, 3 * 2.0**-5, 1.0, 1.0])
    assert np.isnan(locus[5])
    with pytest.raises(ValueError, match="one bubble"):
        locus_frequency(bubbles[:2], is_sv, frequency)


def test_ewens_theta_is_the_float_root_of_the_expected_allele_count():
    haplotypes = 2_072  # the public bench-sim panel's haplotype count

    def expected(theta):
        return 1.0 + theta * (digamma(theta + haplotypes) - digamma(theta + 1.0))

    alleles = expected(np.array([0.05, 0.5, 3.0, 40.0]))
    estimate = ewens_theta(alleles, haplotypes)
    below, above = np.nextafter(estimate, 0.0), np.nextafter(estimate, np.inf)
    brackets_up = (expected(estimate) <= alleles) & (alleles < expected(above))
    brackets_down = (expected(below) <= alleles) & (alleles < expected(estimate))
    assert np.all(brackets_up | brackets_down)
    assert ewens_theta(np.array([1.0]), haplotypes)[0] == 0.0
    with pytest.raises(ValueError, match="allele counts"):
        ewens_theta(np.array([float(haplotypes)]), haplotypes)


def test_block_tagging_matches_brute_force_and_least_squares_predictability():
    generator = np.random.default_rng(5)
    samples = 4_000
    snvs = generator.normal(size=(samples, 6))
    snvs[:, 5] = snvs[:, 0] + snvs[:, 1]
    tagged = snvs[:, 0] - 0.5 * snvs[:, 2] + generator.normal(0.0, 0.7, samples)
    exact = snvs[:, 3] + snvs[:, 4]
    columns = np.column_stack([snvs, tagged, exact])
    columns = (columns - columns.mean(axis=0)) / columns.std(axis=0)
    correlation = columns.T @ columns / samples
    is_structural = np.array([False] * 6 + [True, True])
    tagging = block_tagging(correlation, is_structural)
    squared = correlation[:, 6:] ** 2
    squared[6, 0] = squared[7, 1] = 0.0
    np.testing.assert_allclose(tagging.largest_squared_correlation, squared.max(axis=1))
    np.testing.assert_allclose(tagging.summed_squared_correlation, squared.sum(axis=1))
    for column in (6, 7):
        coefficients, *_ = np.linalg.lstsq(columns[:, :6], columns[:, column], rcond=None)
        residual = columns[:, column] - columns[:, :6] @ coefficients
        assert tagging.structural_predictability[column] == pytest.approx(1.0 - residual.var(), abs=1e-8)
    assert tagging.structural_predictability[7] == pytest.approx(1.0, abs=1e-8)
    assert np.all(np.isnan(tagging.structural_predictability[:6]))


def test_block_tagging_without_snvs_or_svs():
    only_structural = block_tagging(np.eye(2), np.array([True, True]))
    np.testing.assert_array_equal(only_structural.structural_predictability, [0.0, 0.0])
    only_snv = block_tagging(np.eye(3), np.array([False, False, False]))
    np.testing.assert_array_equal(only_snv.largest_squared_correlation, [0.0, 0.0, 0.0])
    assert np.all(np.isnan(only_snv.structural_predictability))
    with pytest.raises(ValueError, match="square"):
        block_tagging(np.eye(3), np.array([True, False]))
