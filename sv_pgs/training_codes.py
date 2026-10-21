"""Stage 2's genotype codes kept from Stage 0's own read of the store, so that no step after Stage 0 reads it again.

Stage 0's pass (``genotype_statistics.compute_genotype_statistics``) reads every sample of every candidate record
(``genotype_statistics.DosageStoreTileSource``), and every record Stage 2 reads is one of them: its blocks' rows are
Stage 0's tie representatives among its active rows. Stage 2 then read them all from the store a second time, to fill
the devices' resident genotype codes (``store_block_source.StoreGenotypeBlockSource``: the first read keeps every
block's signed codes on its device; ~20 minutes of a whole-genome arm at ~400 MB/s [real], after Stage 0's own 43-50),
and the far field's columns one record at a time (``gram_space.far_field_scale``). ``CodeTee`` hands each tile Stage 0
reads to ``TrainingCodes``, which keeps its training columns (candidates x training samples, int8 on the host: each
record's row is the resident set's row of it), and Stage 2's reads take their tiles from them while they are held: on a
device a host copy and an upload per block, every device at once (``device_plan.ShardedDualSource.fill``), so the
resident set fills in the time of its upload; on the host a copy per block.

The codes are the store's own. Stage 0's tile holds each record's codes c as the store decodes them, and the kept code
is c - SIGNED_CODE_OFFSET narrowed to int8 as the device's gather (``store_block_source``'s kernel) and the host's
subtraction narrow it, two's complement, so a tile built from them is the streamed tile bit for bit, and so is every
product, solve and fit made from it (tests/test_training_codes.py).

Memory. The codes are a cache of the host ledger (``memory_broker``): admitted before Stage 0 from what the host has
left, candidates x training samples bytes (17-44 GB at the whole genome's ~1.0 million HapMap3 records and 17,000-44,000
training samples, more with the structural variants), evicted whenever a mandatory lease needs the room, after which
every read is the store's as before (a read in progress takes the blocks it has not reached from the store). The
evaluation codes of the EB monitor (``eb_monitor.EvaluationCodes``) are admitted after them, so the ledger evicts those
first. ``stage2_wiring`` holds them for the group's fit: on the host, where no resident set exists, they serve every read;
on a device, a resident set a later peak of the device evicts (``store_block_source``) is filled again in seconds, where
the store's read took minutes. The lease holds the codes only weakly and is released when they are collected.

The restart store (``restart_store``). On a device the fill keeps each block's codes as its restart unit
(``StoreGenotypeBlockSource._iter_prepared``), whichever source it read them from, and a read whose every block has its
unit takes them from the units first: a restarted fit's Stage 0 is its unit's, which reads no tile and so keeps no
codes, and its devices fill from the units on the local disk. Codes are made only where Stage 0 runs (``_stage0``'s
``tiles``), so a restart admits none.

CODE_CACHE_SWITCH set to 0 keeps no codes: every read of Stage 2 is the store's, as before."""
from __future__ import annotations

import contextlib
import os
import weakref
from typing import Any, Sequence

import numba
import numpy as np

from sv_pgs._typing import I64Array, U8Array
from sv_pgs.genotype_buffers import SIGNED_CODE_OFFSET
from sv_pgs.genotype_statistics import DosageCodeStore, DosageStoreTileSource
from sv_pgs.memory_broker import HOST, current_broker
from sv_pgs.progress import log
from sv_pgs.store_block_source import consecutive_runs

CODE_CACHE_SWITCH = "SVPGS_CODE_CACHE"
"""The environment variable that, set to 0, keeps no codes (``code_cache_on``); unset or anything else, they are kept.
The fit is the same bit for bit either way, so no step's key reads it."""


def code_cache_on() -> bool:
    """Whether Stage 0's pass keeps the training codes (``CODE_CACHE_SWITCH``)."""
    return os.environ.get(CODE_CACHE_SWITCH, "").strip() != "0"


@numba.njit(cache=True, nogil=True)
def _signed_columns(tile, columns, offset, out):
    """out[i, k] = tile[i, columns[k]] - offset narrowed to int8 (two's complement, as the device's gather and the host's
    subtraction narrow it): a Stage 0 tile's training columns as signed codes, without the GIL, so each device's reader
    thread keeps its own tiles at once."""
    for row in range(out.shape[0]):
        for position in range(columns.shape[0]):
            out[row, position] = tile[row, columns[position]] - offset


class TrainingCodes:
    """The training samples' signed codes of Stage 0's candidate records [candidates, training samples], int8 on the
    host, kept from Stage 0's own tiles (``CodeTee``), with which candidates a tile has filled: a cache of the host
    ledger (the module docstring), admitted only from what the host has left and evicted whenever a lease of the fit
    needs the room. Where the ledger cannot admit them the constructor raises ``MemoryError``; a tile they cannot take
    (or any failure of theirs) turns them off, and never reaches Stage 0. ``columns`` are the training samples' store
    columns, ascending: a block source over the same samples reads from them (``serves``)."""

    def __init__(self, candidates: I64Array, columns: I64Array) -> None:
        self.candidates = np.asarray(candidates, dtype=np.int64)
        self.columns = np.asarray(columns, dtype=np.int64)
        nbytes = self.candidates.shape[0] * self.columns.shape[0]
        # Why the codes are off (evicted, released, or a tile they could not take), or None while they are kept.
        self.off: str | None = None
        broker = current_broker()
        self._lease = None
        if broker is not None:
            held = weakref.ref(self)

            def evict() -> None:
                codes = held()
                if codes is not None:
                    codes._drop("the ledger evicted them for the fit's own memory; Stage 2 reads the store")

            self._lease = broker.admit(HOST, nbytes, "Stage 0's training codes", evict)
            if self._lease is None:
                raise MemoryError(f"Stage 0's training codes ({nbytes / 1e9:.2f} GB) do not fit what the host ledger has left: {broker.describe()}")
            weakref.finalize(self, self._lease.release)
        self.codes = np.empty((self.candidates.shape[0], self.columns.shape[0]), dtype=np.int8)
        self.filled = np.zeros(self.candidates.shape[0], dtype=bool)
        log(f"training codes: {nbytes / 1e9:.2f} GB kept from Stage 0's pass for Stage 2's reads")

    def _drop(self, reason: str) -> None:
        if self.off is None:
            self.off = reason
            log(f"training codes: off: {reason}")
        self.codes = np.empty((0, self.columns.shape[0]), dtype=np.int8)

    def capture(self, rows: I64Array, tile: U8Array) -> None:
        """Keep the training columns of a Stage 0 tile: the codes [rows, every store sample] of the candidates ``rows``,
        consecutive among the candidates (one chromosome's tile)."""
        if self.off is not None:
            return
        try:
            codes, first = self.codes, int(np.searchsorted(self.candidates, rows[0]))
            target = slice(first, first + rows.shape[0])
            if not np.array_equal(self.candidates[target], rows):
                raise ValueError("a Stage 0 tile's rows are not consecutive candidates")
            _signed_columns(tile, self.columns, SIGNED_CODE_OFFSET, codes[target])
            self.filled[target] = True
        except Exception as error:  # the codes never fail Stage 0
            self._drop(f"a Stage 0 tile could not be kept ({type(error).__name__}); Stage 2 reads the store")

    def serves(self, columns: I64Array) -> bool:
        """Whether these are the codes of the samples ``columns`` (store columns, in a block source's order)."""
        return bool(np.array_equal(self.columns, np.asarray(columns, dtype=np.int64)))

    def positions(self, rows: I64Array) -> I64Array | None:
        """The positions among the candidates of the store rows ``rows``, where every one is a candidate a
        tile filled; None where one is not (a Stage 0 that did not read it), or the codes are off."""
        rows = np.asarray(rows, dtype=np.int64)
        if self.off is not None or not self.candidates.shape[0]:
            return None
        positions = np.searchsorted(self.candidates, rows)
        if np.any(positions >= self.candidates.shape[0]) or not np.array_equal(self.candidates[np.minimum(positions, self.candidates.shape[0] - 1)], rows):
            return None
        return positions if bool(np.all(self.filled[positions])) else None

    def in_use(self) -> Any:
        """While inside, the codes are not evicted (``memory_broker.Lease.in_use``)."""
        return contextlib.nullcontext() if self._lease is None else self._lease.in_use()

    def copy_rows(self, positions: I64Array, out: np.ndarray) -> bool:
        """out[k] = the signed codes of the candidate at ``positions[k]`` (ascending, from ``positions``), each run of
        consecutive ones one copy, the codes held meanwhile; False, with ``out`` untouched, where they are off."""
        with self.in_use():
            if self.off is not None:
                return False
            held = self.codes
            filled = 0
            for first, last in consecutive_runs(np.asarray(positions, dtype=np.int64)):
                out[filled : filled + last - first] = held[first:last]
                filled += last - first
        return True

    def unsigned(self, rows: I64Array) -> U8Array | None:
        """The store's own codes [rows, training samples] of the store rows ``rows``, as the store's ``read_codes``
        gives them on these columns: each kept code plus SIGNED_CODE_OFFSET in uint8, the narrowing undone; None where a
        row is not kept or the codes are off."""
        with self.in_use():
            positions = self.positions(rows)
            if positions is None:
                return None
            return self.codes[positions].view(np.uint8) + np.uint8(SIGNED_CODE_OFFSET)

    def release(self, reason: str = "released") -> None:
        """The codes dropped, for ``reason``, and their lease released."""
        self._drop(self.off or reason)
        if self._lease is not None:
            self._lease.release()


class CodeTee(DosageStoreTileSource):
    """Stage 0's tile source (``genotype_statistics.DosageStoreTileSource``) that hands every tile it reads to each of
    ``keepers`` (``TrainingCodes``, ``eb_monitor.EvaluationCodes``: each keeps its own columns of it, never failing the
    read); the tiles themselves are unchanged."""

    def __init__(self, store: DosageCodeStore, candidate_rows: I64Array, keepers: Sequence[Any]) -> None:
        super().__init__(store, candidate_rows)
        self._keepers = tuple(keepers)

    def read_rows(self, chromosome: str, start: int, stop: int, out: U8Array) -> None:
        super().read_rows(chromosome, start, stop, out)
        rows = self.store_rows(chromosome)[start:stop]
        for keeper in self._keepers:
            keeper.capture(rows, out[: stop - start])
