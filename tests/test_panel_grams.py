"""``device_sweep.PanelGrams``: panels are named by their members, and the cache holds what its capacity allows."""

import numpy as np

from sv_pgs.device_sweep import PanelGrams


def _panel(members):
    return (0, np.ascontiguousarray(members, dtype=np.int64).tobytes())


def test_panels_of_other_members_never_read_each_others_gram():
    # Under the old key (a piece's first member plus the panel's offset) these two panels both named (0, 74): the
    # second read the first's Gram.
    first, second = np.arange(74, 138), np.array([74, *range(200, 263)])
    grams, built = PanelGrams(), []

    def build(members):
        def make():
            built.append(members[0])
            return (np.full((2, 2), float(members.sum())), np.zeros((1, 2)))
        return make

    one = grams.get(_panel(first), build(first))
    two = grams.get(_panel(second), build(second))
    assert len(built) == 2 and one[0][0, 0] != two[0][0, 0]
    assert grams.get(_panel(first), build(first)) is one and len(built) == 2


def test_the_cache_keeps_the_panels_that_fit_and_rebuilds_the_rest():
    part = (np.zeros((4, 4)), np.zeros((1, 4)))
    size = sum(value.nbytes for value in part)
    grams, calls = PanelGrams(capacity_bytes=2 * size), []

    def build():
        calls.append(1)
        return tuple(value.copy() for value in part)

    for sweep in range(2):
        for panel in range(3):
            grams.get(_panel([panel]), build)
    # Panels 0 and 1 fit and are kept; panel 2 is rebuilt on every sweep.
    assert len(calls) == 4
    grams.clear()
    grams.get(_panel([0]), build)
    assert len(calls) == 5
