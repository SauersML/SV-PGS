"""Copy-number store columns: modal copy number, code scale, and the per-record affine decode."""
from __future__ import annotations

import numpy as np
import pytest

from sv_pgs.copy_number import (
    allele_count_decode,
    copy_number_codes_per_unit,
    decode_values,
    encode_copy_numbers,
    modal_copy_numbers,
)
from sv_pgs.dosage_store import CODES_PER_DOSAGE, MAXIMUM_CODE


def test_the_modal_copy_number_is_the_most_frequent_call_and_the_smallest_among_ties() -> None:
    copy_numbers = np.array([[2, 2, 3, 1, 2], [1, 3, 3, 1, 4], [0, 5, 5, 7, 7]])
    called = np.array([[True] * 5, [True] * 5, [True, True, True, False, False]])
    np.testing.assert_array_equal(modal_copy_numbers(copy_numbers, called), [2, 1, 5])
    with pytest.raises(ValueError):
        modal_copy_numbers(copy_numbers, np.zeros_like(called))


def test_every_integer_copy_number_up_to_the_maximum_is_stored_exactly() -> None:
    rng = np.random.default_rng(3)
    maximum = np.array([2, 5, 12, 127, MAXIMUM_CODE])
    copy_numbers = np.stack([rng.integers(0, top + 1, size=400) for top in maximum])
    copy_numbers[np.arange(maximum.size), 0] = maximum
    scale = copy_number_codes_per_unit(copy_numbers.max(axis=1))
    np.testing.assert_array_equal(scale, MAXIMUM_CODE // maximum)
    modal = modal_copy_numbers(copy_numbers, np.ones_like(copy_numbers, dtype=bool))
    values = decode_values(encode_copy_numbers(copy_numbers, scale), scale, -modal)
    # CN * k / k is exact in fp64 for these small integers, and so is the shift.
    np.testing.assert_array_equal(values, copy_numbers - modal[:, None])


def test_a_copy_number_past_the_code_range_is_refused() -> None:
    with pytest.raises(ValueError):
        copy_number_codes_per_unit(np.array([MAXIMUM_CODE + 1]))
    with pytest.raises(ValueError):
        encode_copy_numbers(np.array([[3, 6]]), np.array([50], dtype=np.uint8))


def test_allele_count_records_decode_as_today() -> None:
    codes = np.array([[0, 127, 254, 63]], dtype=np.uint8)
    scale, origin = allele_count_decode(1)
    assert scale[0] == CODES_PER_DOSAGE and origin[0] == 0
    np.testing.assert_array_equal(decode_values(codes, scale, origin), codes / CODES_PER_DOSAGE)


def test_a_copy_number_column_passes_stage_0_as_its_decoded_copy_numbers(tmp_path) -> None:
    from sv_pgs.compute_budget import ComputeBudget
    from sv_pgs.config import ModelConfig
    from sv_pgs.genotype_statistics import compute_genotype_statistics
    from tests.phenotype_bounds import rounding_gamma
    from tests.stage0_support import InMemoryTileSource, bubble_groups, mosaic_codes

    rng = np.random.default_rng(29)
    samples, records, row = 400, 60, 7
    codes = mosaic_codes(rng, samples, records, hotspot_spacing=60, regular_hotspots=False)
    copy_numbers = rng.choice([1, 2, 3, 4], size=(1, samples), p=[0.15, 0.6, 0.2, 0.05])
    scale = copy_number_codes_per_unit(copy_numbers.max(axis=1))
    codes[row] = encode_copy_numbers(copy_numbers, scale)[0]
    source = InMemoryTileSource(codes={"chr2": codes}, groups={"chr2": bubble_groups(rng, records)})
    covariates = np.column_stack([np.ones(samples), rng.normal(size=samples)])
    targets = rng.normal(size=(samples, 1))
    budget = ComputeBudget(device_kind="cpu", device_ids=(), device_names=(), device_bytes=(),
                           device_compute_capabilities=(), host_bytes=8 * 10**9, cpu_threads=2)
    statistics = compute_genotype_statistics(
        source, np.arange(samples), covariates, targets, ModelConfig(), budget, 128, tmp_path / "ld"
    )
    active = {int(record): index for index, record in enumerate(statistics.active_rows.tolist())}
    reduced = statistics.tie_map.original_to_reduced[active[row]]
    block_index = int(statistics.block_of_reduced[reduced])
    block = statistics.ld.block(block_index)
    position = int(np.flatnonzero(block.reduced_columns == reduced)[0])

    # The reference works on the decoded values CN - modal CN, never on the codes.
    modal = modal_copy_numbers(copy_numbers, np.ones_like(copy_numbers, dtype=bool))
    values = decode_values(codes[[row]], scale, -modal)[0]
    standardized = (values - values.mean()) / values.std()
    hat = covariates @ np.linalg.solve(covariates.T @ covariates, covariates.T)
    projected_x = standardized - hat @ standardized
    projected_y = targets[:, 0] - hat @ targets[:, 0]
    expected = projected_x @ projected_y
    magnitude = np.abs(standardized) @ (np.abs(targets[:, 0]) + np.abs(hat) @ np.abs(targets[:, 0]))
    assert abs(block.projected_score[position, 0] - expected) <= rounding_gamma(16 * (samples + covariates.shape[1])) * magnitude
