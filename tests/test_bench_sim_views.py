"""Bounded reads of benchmark genotype views, including row ordering and repeated indices."""
from types import SimpleNamespace

import numpy as np
import pytest

from benchmarks.bench_sim import harness


@pytest.mark.parametrize("view", [harness.TrainData, harness.ScoreData])
@pytest.mark.parametrize("rows", [0, slice(None), [2, 0, 2], np.array([], dtype=int), np.array([True, False, True])])
def test_codes_select_samples_in_the_source_read(view, rows):
    selections = []

    class Observed(np.ndarray):
        def __getitem__(self, selection):
            selections.append(selection)
            return super().__getitem__(selection)

    original = np.arange(70, dtype=np.uint8).reshape(7, 10)
    records = np.array([5, 1, 4])
    columns = np.array([7, 2, 7, 0])
    source = SimpleNamespace(_observed=original.view(Observed), _records=records, _columns=columns)
    actual = view.codes(source, rows)
    np.testing.assert_array_equal(actual, original[records[rows]][..., columns])
    assert len(selections) == 1 and isinstance(selections[0], tuple)
    np.testing.assert_array_equal(selections[0][1], columns)
