"""GATK-SV store rows: every called record a row, no-call fills, copy-number codes, fusion pairs, dropped records."""
from __future__ import annotations

import numpy as np
import pytest

from sv_pgs.config import VariantClass
from sv_pgs.dosage_store import CODES_PER_DOSAGE
from sv_pgs.gatksv_source import GatksvBlock
from sv_pgs.gatksv_store_rows import ImputedSvRecords, gatksv_sites, gatksv_store_rows
from sv_pgs.sv_fusion import MINIMUM_PAIRING_Z, SvSites

_SAMPLE_COUNT = 20_000


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
        codes=_codes(np.stack([dosage, unrelated_dosage, sharp_dosage]), CODES_PER_DOSAGE),
    )
    return gatksv, imputed, {"genotype": genotype}



def test_gatksv_sites_use_the_first_affected_base_and_an_exclusive_end() -> None:
    gatksv, _, _ = _scenario()

    sites = gatksv_sites(gatksv)

    assert sites.starts.tolist() == [10_001, 30_001, 50_001, 10_001, 70_001]
    assert sites.ends.tolist() == [12_001, 30_002, 51_001, 12_001, 72_001]
    assert sites.sizes.tolist() == [2_000, 800, 1_000, 2_000, 2_000]
    assert not sites.duplications_are_insertions


def _best_linear_prediction(gatksv: GatksvBlock, imputed: ImputedSvRecords, record: int, imputed_record: int) -> np.ndarray:
    dosage = imputed.codes[imputed_record] / CODES_PER_DOSAGE
    called = ~gatksv.no_call[record]
    covariance = np.cov(dosage[called], gatksv.values[record][called].astype(np.float64), bias=True)
    slope = covariance[0, 1] / covariance[0, 0]
    return gatksv.values[record][called].mean() + slope * (dosage - dosage[called].mean())


def test_every_called_record_is_a_row_and_the_strong_pairs_are_left_to_the_measurement_model() -> None:
    gatksv, imputed, truth = _scenario()
    observed = ~gatksv.no_call

    rows, fusion = gatksv_store_rows(gatksv, imputed)

    # The two DELs pair strongly with their imputed records; the unrelated INS does not.
    assert fusion.imputed_records.tolist() == [0, 2] and fusion.gatksv_rows.tolist() == [0, 2]
    assert np.all(fusion.pairing_z >= MINIMUM_PAIRING_Z)
    # Every called record stays a row, the all-no-call DUP is dropped, nothing is fused here.
    assert rows.gatksv_records.tolist() == [0, 1, 2, 3]
    assert rows.unobserved_records.tolist() == [4]
    np.testing.assert_array_equal(rows.lengths, gatksv.lengths[[0, 1, 2, 3]])
    assert rows.filled_from_imputed.tolist() == [True, True, True, False]
    np.testing.assert_allclose(rows.observed_fractions, observed[[0, 1, 2, 3]].mean(axis=1))
    # Observed calls are stored exactly: 127 per allele, and floor(254 / 3) = 84 per copy for
    # the copy-number record, whose calls reach 3 copies and whose modal call is 2.
    for record in (0, 1, 2):
        np.testing.assert_array_equal(rows.codes[record][observed[record]], gatksv.values[record][observed[record]] * 127)
    assert gatksv.values[3][observed[3]].max() == 3
    assert rows.codes_per_unit.tolist() == [127, 127, 127, 84]
    assert rows.value_origin.tolist() == [0, 0, 0, -2]
    np.testing.assert_array_equal(rows.codes[3][observed[3]], gatksv.values[3][observed[3]].astype(np.int64) * 84)
    # No-calls take E[B | DS] from the paired imputed DS, or the observed mean for a copy number.
    for record in (0, 1, 2):
        expected = np.floor(np.clip(_best_linear_prediction(gatksv, imputed, record, record), 0.0, 2.0) * 127 + 0.5).astype(np.uint8)
        np.testing.assert_array_equal(rows.codes[record][gatksv.no_call[record]], expected[gatksv.no_call[record]])
    copy_number_mean = gatksv.values[3][observed[3]].mean()
    assert set(rows.codes[3][gatksv.no_call[3]].tolist()) == {int(np.floor(copy_number_mean * 84 + 0.5))}
    half_code = 0.5 / 127
    for record in (0, 1, 2):
        predicted = _best_linear_prediction(gatksv, imputed, record, record)[gatksv.no_call[record]]
        assert rows.clipped_counts[record] == int(np.count_nonzero((predicted < -half_code) | (predicted >= 2 + half_code)))
    assert rows.clipped_counts[3] == 0
    assert truth["genotype"].shape == (gatksv.values.shape[1],)


def test_store_rows_need_the_store_samples() -> None:
    gatksv, imputed, _ = _scenario()
    narrow = ImputedSvRecords(sites=imputed.sites, codes=imputed.codes[:, :-1])
    with pytest.raises(ValueError, match="store's samples"):
        gatksv_store_rows(gatksv, narrow)
