"""GATK-SV store rows: fused rows, no-call fills, copy-number codes, dropped records."""
from __future__ import annotations

import numpy as np
import pytest

from sv_pgs.config import VariantClass
from sv_pgs.gatksv_source import GatksvBlock
from sv_pgs.gatksv_store_rows import (
    CODES_PER_ALLELE,
    FalsePositiveRates,
    ImputedSvRecords,
    gatksv_sites,
    gatksv_store_rows,
)
from sv_pgs.sv_fusion import AnchorErrorModel, SvSites, calibrate_two_sources

_SAMPLE_COUNT = 20_000
# Stratum 0 trusts its anchors; stratum 1's anchor error swamps them, so its
# records keep the reliability model's prediction.
_ERROR_MODELS = (
    AnchorErrorModel(bias=0.0, excess_variance=0.01, berkson=False),
    AnchorErrorModel(bias=0.0, excess_variance=1e6, berkson=False),
)
_CONTRADICTED_RELIABILITY = 0.02


def _draw_and_calls(
    generator: np.random.Generator, frequency: float, accuracy: float, miss_rate: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Genotype, a draw-like imputed DS (each haplotype right with probability ``accuracy``,
    else a fresh draw from the frequency) and hard calls that miss alleles."""
    haplotypes = generator.random((_SAMPLE_COUNT, 2)) < frequency
    informed = generator.random(haplotypes.shape) < accuracy
    dosage = np.where(informed, haplotypes, generator.random(haplotypes.shape) < frequency).sum(axis=1)
    calls = (haplotypes & (generator.random(haplotypes.shape) > miss_rate)).sum(axis=1)
    return haplotypes.sum(axis=1), dosage.astype(np.float64), calls


def _codes(values: np.ndarray, codes_per_unit: int) -> np.ndarray:
    return np.floor(values * codes_per_unit + 0.5).astype(np.uint8)


def _scenario() -> tuple[GatksvBlock, ImputedSvRecords, dict[str, np.ndarray]]:
    generator = np.random.default_rng(17)
    no_call = generator.random((5, _SAMPLE_COUNT)) < 0.1
    no_call[4] = True
    # Record 0: a DEL both sources carry.
    genotype, dosage, calls = _draw_and_calls(generator, 0.15, 0.8, 0.15)
    # Record 1: an INS at the same place in both sources, but unrelated genotypes.
    _, unrelated_dosage, _ = _draw_and_calls(generator, 0.2, 0.8, 0.1)
    _, _, unrelated_calls = _draw_and_calls(generator, 0.2, 0.8, 0.1)
    # Record 2: a DEL that pairs strongly, but whose predicted reliability is
    # far below what the pair's correlation needs, so the calibration rejects it.
    _, sharp_dosage, accurate_calls = _draw_and_calls(generator, 0.2, 0.9, 0.05)
    copy_numbers = generator.choice([1, 2, 3], size=_SAMPLE_COUNT, p=[0.1, 0.7, 0.2])
    values = np.stack([calls, unrelated_calls, accurate_calls, copy_numbers, np.zeros(_SAMPLE_COUNT)])
    values = np.where(no_call, 0, values).astype(np.uint8)
    gatksv = GatksvBlock(
        chromosomes=("chr1",) * 5,
        positions=np.array([10_000, 30_000, 50_000, 10_000, 70_000], dtype=np.int64),
        ends=np.array([12_000, 30_001, 51_000, 12_000, 72_000], dtype=np.int64),
        lengths=np.array([2_000.0, 800.0, 1_000.0, 2_000.0, 2_000.0]),
        variant_ids=("gatksv_del", "gatksv_ins", "gatksv_del_contradicted", "gatksv_cnv", "gatksv_dup_uncalled"),
        svtypes=("DEL", "INS", "DEL", "CNV", "DUP"),
        variant_classes=(
            VariantClass.DELETION,
            VariantClass.INSERTION_MEI,
            VariantClass.DELETION,
            VariantClass.COPY_NUMBER,
            VariantClass.DUPLICATION,
        ),
        is_copy_number=np.array([False, False, False, True, False]),
        values=values,
        no_call=no_call,
    )
    imputed = ImputedSvRecords(
        sites=SvSites(
            chromosomes=np.array(["chr1", "chr1", "chr1"]),
            starts=np.array([10_001, 30_001, 50_001], dtype=np.int64),
            ends=np.array([12_001, 30_002, 51_001], dtype=np.int64),
            sizes=np.array([2_000, 800, 1_000], dtype=np.int64),
            kinds=np.array(["DEL", "INS", "DEL"]),
            duplications_are_insertions=True,
        ),
        codes=_codes(np.stack([dosage, unrelated_dosage, sharp_dosage]), CODES_PER_ALLELE),
        prior_log_reliabilities=np.log(np.array([0.64, 0.64, _CONTRADICTED_RELIABILITY])),
        strata=np.array([0, 0, 1], dtype=np.int64),
    )
    return gatksv, imputed, {"genotype": genotype}


_NO_FALSE_POSITIVES = FalsePositiveRates(means=np.zeros(5), variances=np.zeros(5))


def _squared_correlation(first: np.ndarray, second: np.ndarray) -> float:
    return float(np.corrcoef(first, second)[0, 1] ** 2)


def test_gatksv_sites_use_the_first_affected_base_and_an_exclusive_end() -> None:
    gatksv, _, _ = _scenario()

    sites = gatksv_sites(gatksv)

    assert sites.starts.tolist() == [10_001, 30_001, 50_001, 10_001, 70_001]
    assert sites.ends.tolist() == [12_001, 30_002, 51_001, 12_001, 72_001]
    assert sites.sizes.tolist() == [2_000, 800, 1_000, 2_000, 2_000]
    assert not sites.duplications_are_insertions


def test_store_rows_fuse_the_accepted_pair_and_fill_every_other_no_call() -> None:
    gatksv, imputed, truth = _scenario()
    observed = ~gatksv.no_call

    fused, rows = gatksv_store_rows(gatksv, imputed, _ERROR_MODELS, _NO_FALSE_POSITIVES)

    # Only the first DEL fuses; its row replaces imputed record 0.
    assert fused.imputed_records.tolist() == [0]
    assert fused.gatksv_records.tolist() == [0]
    calibration = fused.calibrations[0]
    assert calibration.accepted
    fused_dosage = fused.codes[0] / CODES_PER_ALLELE
    imputed_dosage = imputed.codes[0] / CODES_PER_ALLELE
    assert _squared_correlation(fused_dosage, truth["genotype"]) > _squared_correlation(imputed_dosage, truth["genotype"]) + 0.05
    # A GATK-SV no-call takes the recalibrated imputed dosage.
    recalibrated = calibration.missing_genotype_mean + calibration.missing_slope * (imputed_dosage - calibration.missing_first_mean)
    expected = np.floor(np.clip(recalibrated, 0.0, 2.0) * 127 + 0.5).astype(np.uint8)
    np.testing.assert_array_equal(fused.codes[0][gatksv.no_call[0]], expected[gatksv.no_call[0]])

    # The rest stay rows; the all-no-call DUP is dropped.
    assert rows.gatksv_records.tolist() == [1, 2, 3]
    assert rows.unobserved_records.tolist() == [4]
    assert rows.filled_from_imputed.tolist() == [False, True, False]
    np.testing.assert_allclose(rows.observed_fractions, observed[[1, 2, 3]].mean(axis=1))
    # Observed calls are stored exactly: 127 per allele, a copy number as itself.
    for row, record in enumerate([1, 2]):
        np.testing.assert_array_equal(rows.codes[row][observed[record]], gatksv.values[record][observed[record]] * 127)
    np.testing.assert_array_equal(rows.codes[2][observed[3]], gatksv.values[3][observed[3]])

    # Unrelated INS: its no-calls take the mean of its observed calls.
    unrelated_mean = gatksv.values[1][observed[1]].mean()
    assert set(rows.codes[0][gatksv.no_call[1]].tolist()) == {int(np.floor(unrelated_mean * 127 + 0.5))}
    # Contradicted DEL: significant pairing, rejected calibration, so its
    # no-calls are E[B | DS] from the paired imputed record.
    contradicted = calibrate_two_sources(
        imputed.codes[2] / CODES_PER_ALLELE, gatksv.values[2], observed[2], _CONTRADICTED_RELIABILITY
    )
    assert not contradicted.accepted and contradicted.pairing_z >= 5
    prediction = contradicted.observed_second_mean + contradicted.second_slope * (
        imputed.codes[2] / CODES_PER_ALLELE - contradicted.observed_first_mean
    )
    expected = np.floor(np.clip(prediction, 0.0, 2.0) * 127 + 0.5).astype(np.uint8)
    np.testing.assert_array_equal(rows.codes[1][gatksv.no_call[2]], expected[gatksv.no_call[2]])
    # Copy numbers never pair; their no-calls take the rounded observed mean.
    copy_number_mean = gatksv.values[3][observed[3]].mean()
    assert set(rows.codes[2][gatksv.no_call[3]].tolist()) == {int(np.floor(copy_number_mean + 0.5))}
    predicted = prediction[gatksv.no_call[2]]
    half_code = 0.5 / 127
    assert rows.clipped_counts.tolist() == [0, int(np.count_nonzero((predicted < -half_code) | (predicted >= 2 + half_code))), 0]


def test_store_rows_need_the_reliability_predictions_and_false_positive_rates() -> None:
    gatksv, imputed, _ = _scenario()
    missing_prediction = ImputedSvRecords(
        sites=imputed.sites,
        codes=imputed.codes,
        prior_log_reliabilities=np.array([np.nan, 0.0, 0.0]),
        strata=imputed.strata,
    )

    with pytest.raises(ValueError, match="reliability-model prediction"):
        gatksv_store_rows(gatksv, missing_prediction, _ERROR_MODELS, _NO_FALSE_POSITIVES)
    with pytest.raises(ValueError, match="one false-positive rate per GATK-SV record"):
        gatksv_store_rows(gatksv, imputed, _ERROR_MODELS, FalsePositiveRates(means=np.zeros(4), variances=np.zeros(4)))
