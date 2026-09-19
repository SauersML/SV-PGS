"""GATK-SV store rows: fused rows, no-call fills, copy-number codes, dropped records."""
from __future__ import annotations

import numpy as np

from sv_pgs.config import VariantClass
from sv_pgs.gatksv_source import GatksvBlock
from sv_pgs.gatksv_store_rows import (
    CODES_PER_ALLELE,
    GP2_CODES_PER_UNIT,
    ImputedSvRecords,
    gatksv_sites,
    gatksv_store_rows,
)
from sv_pgs.sv_fusion import SvSites, calibrate_two_sources

_SAMPLE_COUNT = 20_000


def _imputed_and_calls(
    generator: np.random.Generator, frequency: float, separation: float, miss_rate: float, odds_deflation: float = 1.0
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Genotype, imputed DS and GP2 from per-haplotype posteriors, and hard calls that miss alleles.

    ``odds_deflation`` below 1 shrinks the posterior log-odds: an underconfident DS.
    """
    haplotypes = generator.random((_SAMPLE_COUNT, 2)) < frequency
    evidence = generator.standard_normal(haplotypes.shape) + separation * haplotypes
    log_odds = separation * evidence - separation**2 / 2 + np.log(frequency / (1 - frequency))
    posteriors = 1 / (1 + np.exp(-odds_deflation * log_odds))
    calls = (haplotypes & (generator.random(haplotypes.shape) > miss_rate)).sum(axis=1)
    return haplotypes.sum(axis=1), posteriors.sum(axis=1), posteriors.prod(axis=1), calls


def _codes(values: np.ndarray, codes_per_unit: int) -> np.ndarray:
    return np.floor(values * codes_per_unit + 0.5).astype(np.uint8)


def _posterior_variance(imputed: ImputedSvRecords, record: int) -> np.ndarray:
    dosage = imputed.codes[record] / CODES_PER_ALLELE
    return dosage + 2.0 * imputed.gp2_codes[record] / GP2_CODES_PER_UNIT - dosage**2


def _scenario() -> tuple[GatksvBlock, ImputedSvRecords, dict[str, np.ndarray]]:
    generator = np.random.default_rng(17)
    no_call = generator.random((5, _SAMPLE_COUNT)) < 0.1
    no_call[4] = True
    # Record 0: a DEL both sources carry; the imputed DS is calibrated.
    genotype, dosage, gp2, calls = _imputed_and_calls(generator, 0.1, 2.0, 0.15)
    # Record 1: an INS at the same place in both sources, but unrelated genotypes.
    _, unrelated_dosage, unrelated_gp2, _ = _imputed_and_calls(generator, 0.2, 2.0, 0.1)
    _, _, _, unrelated_calls = _imputed_and_calls(generator, 0.2, 2.0, 0.1)
    # Record 2: one DEL whose DS is underconfident (deflated odds), so the
    # pairing is strong but the implied GATK-SV reliability exceeds 1.
    _, deflated_dosage, deflated_gp2, accurate_calls = _imputed_and_calls(generator, 0.2, 3.0, 0.05, odds_deflation=0.3)
    copy_numbers = generator.choice([1, 2, 3], size=_SAMPLE_COUNT, p=[0.1, 0.7, 0.2])
    values = np.stack([calls, unrelated_calls, accurate_calls, copy_numbers, np.zeros(_SAMPLE_COUNT)])
    values = np.where(no_call, 0, values).astype(np.uint8)
    gatksv = GatksvBlock(
        chromosomes=("chr1",) * 5,
        positions=np.array([10_000, 30_000, 50_000, 10_000, 70_000], dtype=np.int64),
        ends=np.array([12_000, 30_001, 51_000, 12_000, 72_000], dtype=np.int64),
        lengths=np.array([2_000.0, 800.0, 1_000.0, 2_000.0, 2_000.0]),
        variant_ids=("gatksv_del", "gatksv_ins", "gatksv_del_underconfident", "gatksv_cnv", "gatksv_dup_uncalled"),
        svtypes=("DEL", "INS", "DEL", "CNV", "DUP"),
        variant_classes=(
            VariantClass.DELETION_LONG,
            VariantClass.INSERTION_MEI,
            VariantClass.DELETION_SHORT,
            VariantClass.COPY_NUMBER,
            VariantClass.DUPLICATION_LONG,
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
        codes=_codes(np.stack([dosage, unrelated_dosage, deflated_dosage]), CODES_PER_ALLELE),
        gp2_codes=_codes(np.stack([gp2, unrelated_gp2, deflated_gp2]), GP2_CODES_PER_UNIT),
    )
    return gatksv, imputed, {"genotype": genotype}


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

    fused, rows = gatksv_store_rows(gatksv, imputed)

    # Only the calibrated DEL fuses; its row replaces imputed record 0.
    assert fused.imputed_records.tolist() == [0]
    assert fused.gatksv_records.tolist() == [0]
    assert fused.calibrations[0].accepted
    fused_dosage = fused.codes[0] / CODES_PER_ALLELE
    imputed_dosage = imputed.codes[0] / CODES_PER_ALLELE
    assert _squared_correlation(fused_dosage, truth["genotype"]) > _squared_correlation(imputed_dosage, truth["genotype"]) + 0.1
    # A GATK-SV no-call keeps the imputed DS.
    np.testing.assert_array_equal(fused.codes[0][gatksv.no_call[0]], imputed.codes[0][gatksv.no_call[0]])

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
    # Underconfident DEL: significant pairing, rejected calibration, so its
    # no-calls are E[B | DS] from the paired imputed record.
    calibration = calibrate_two_sources(
        imputed.codes[2] / CODES_PER_ALLELE, _posterior_variance(imputed, 2), gatksv.values[2], observed[2]
    )
    assert not calibration.accepted and calibration.pairing_z >= 5
    prediction = calibration.second_mean + calibration.second_slope * (imputed.codes[2] / CODES_PER_ALLELE - calibration.first_mean)
    expected = np.floor(np.clip(prediction, 0.0, 2.0) * 127 + 0.5).astype(np.uint8)
    np.testing.assert_array_equal(rows.codes[1][gatksv.no_call[2]], expected[gatksv.no_call[2]])
    # Copy numbers never pair; their no-calls take the rounded observed mean.
    copy_number_mean = gatksv.values[3][observed[3]].mean()
    assert set(rows.codes[2][gatksv.no_call[3]].tolist()) == {int(np.floor(copy_number_mean + 0.5))}
    predicted = prediction[gatksv.no_call[2]]
    half_code = 0.5 / 127
    assert rows.clipped_counts.tolist() == [0, int(np.count_nonzero((predicted < -half_code) | (predicted >= 2 + half_code))), 0]
