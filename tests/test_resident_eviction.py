"""Resident-code eviction releases unread blocks while a consumer holds its current tile."""
import numpy as np
from sv_pgs.store_block_source import StoreGenotypeBlockSource


def test_resident_eviction_frees_unread_blocks_while_the_iterator_is_paused(monkeypatch) -> None:
    """A mandatory allocation must see the eviction's bytes before the consumer resumes its tile iterator.
    NumPy arrays and a weak-reference meter exercise ownership without a device or a store read."""
    import weakref

    from sv_pgs.memory_broker import MemoryBroker, device_pool

    source = object.__new__(StoreGenotypeBlockSource)
    source._cupy = object()
    source._resident = [np.ones(8, dtype=np.int8) for _ in range(3)]
    references = [weakref.ref(codes) for codes in source._resident]
    block_bytes = source._resident[0].nbytes
    source._resident_complete = True
    source._evictions = 0
    source._pool = device_pool(0)
    source._block_rows = [np.arange(8) for _ in references]

    def held():
        return sum(codes.nbytes for reference in references if (codes := reference()) is not None)

    broker = MemoryBroker({source._pool: held()}, meters={source._pool: held})
    lease = broker.admit(source._pool, held(), "resident test codes", source._drop_resident, allocated=True)
    assert lease is not None
    monkeypatch.setattr(source, "_resident_tile", lambda block, codes: codes)
    streamed = []

    def stream(first):
        streamed.append(first)
        for block in range(first, len(references)):
            yield block, np.ones(8, dtype=np.int8)

    monkeypatch.setattr(source, "_iter_streamed", stream)
    iterator = source.iter_tiles()
    block, tile = next(iterator)
    assert block == 0 and tile is references[0]()
    broker.make_room(source._pool, 2 * block_bytes, "workspace while reading a tile")
    assert not lease.live and source._evictions == 1
    assert source._resident is None and not source._resident_complete
    assert references[0]() is tile and all(reference() is None for reference in references[1:])
    assert broker.held(source._pool) == block_bytes
    assert [block for block, _tile in iterator] == [1, 2]
    assert streamed == [1]
    np.testing.assert_array_equal(tile, np.ones(8, dtype=np.int8))


def test_resident_eviction_clears_a_keep_allocations_list_alias():
    import weakref

    source = object.__new__(StoreGenotypeBlockSource)
    source._resident = [np.ones(8, dtype=np.int8) for _ in range(3)]
    references = [weakref.ref(codes) for codes in source._resident]
    alias = source._resident
    source._resident_complete = True
    source._evictions = 0
    source._pool = "device0"
    source._block_rows = [np.arange(8) for _ in references]

    class Slot:
        def __getitem__(self, rows):
            return self

        def copy(self):
            source._drop_resident()
            # _keep still owns the list while the allocator retries this copy.
            assert alias == [] and all(reference() is None for reference in references)
            return np.ones(8, dtype=np.int8)

    source._signed = [Slot()]
    source._keep(0, 0)
    assert source._resident is None and not source._resident_complete
