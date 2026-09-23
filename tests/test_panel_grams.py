"""``device_sweep.PanelGrams``: panels are named by their members, and the cache holds what the shared ledger leaves
it, giving its panels back when a mandatory lease needs the bytes."""

import numpy as np
import pytest

from sv_pgs.device_sweep import PanelGrams
from sv_pgs.memory_broker import MemoryBroker


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


def _part():
    return (np.zeros((4, 4)), np.zeros((1, 4)))


def test_the_cache_keeps_the_panels_the_ledger_leaves_room_for_and_rebuilds_the_rest():
    size = sum(value.nbytes for value in _part())
    broker = MemoryBroker({"device0": 3 * size})
    mandatory = broker.reserve("device0", size, "a mandatory buffer")
    grams, calls = PanelGrams(broker, "device0"), []

    def build():
        calls.append(1)
        return _part()

    for _sweep in range(2):
        for panel in range(3):
            grams.get(_panel([panel]), build)
    # The mandatory buffer leaves room for two panels: 0 and 1 are kept, panel 2 is rebuilt on every sweep.
    assert len(calls) == 4 and broker.held("device0") == 3 * size
    grams.clear()
    assert broker.held("device0") == size
    grams.get(_panel([0]), build)
    assert len(calls) == 5
    mandatory.release()


def test_a_mandatory_lease_evicts_panels_and_never_fails_for_them():
    size = sum(value.nbytes for value in _part())
    broker = MemoryBroker({"device0": 2 * size})
    grams, calls = PanelGrams(broker, "device0"), []

    def build():
        calls.append(1)
        return _part()

    grams.get(_panel([0]), build)
    grams.get(_panel([1]), build)
    assert broker.remaining("device0") == 0
    # The whole pool is mandatory now: both panels are evicted, and the next sweep rebuilds them without keeping them.
    lease = broker.reserve("device0", 2 * size, "the fit's mandatory buffers")
    assert broker.held("device0") == 2 * size
    grams.get(_panel([0]), build)
    assert len(calls) == 3 and broker.held("device0") == 2 * size
    with pytest.raises(MemoryError):
        broker.reserve("device0", 1, "one byte more")
    lease.release()
