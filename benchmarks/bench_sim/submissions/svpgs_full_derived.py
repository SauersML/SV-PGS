"""bench-sim prototype arms: ``svpgs_full`` on a store where some columns are fixed linear combinations of records.

A derived column is c + sum_a w_a D_a over some of the arm's records (``Derived``: a sparse weight matrix over the
records, one row per derived column, an offset, and the records it replaces, which leave the store). Two arms use it:

- ``svpgs_full_trlocus``: each tandem-repeat locus's signed-length column, MODEL.md section 1;
- ``svpgs_full_refstate``: each non-TR multi-allelic site's REF state, the same section.

These measure a column before the engine computes it from the per-allele products. The store holds a derived column
as an ALT-count record: its training range [lo, hi] is mapped onto the codes 0..MAXIMUM_CODE and rounded, so its raw
unit is (hi - lo) / 2 of the column's own and its values are quantized to 1 / MAXIMUM_CODE of the range. Those two
departures are the prototype's; the unit contract reads a derived column's variance in that unit. The test samples'
values are the same combination of their records, unrounded, so the coefficients carry back to the records exactly.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy import sparse

from benchmarks.bench_sim.submissions import svpgs_full as full
from sv_pgs.config import TraitType, VariantClass
from sv_pgs.dosage_store import CODES_PER_DOSAGE, MAXIMUM_CODE, VARIANT_CLASSES, DosageStore, VariantTable, write_dosage_store
from sv_pgs.fast_scoring import SIGNED_CODE_OFFSET
from sv_pgs.fit_model import DRAW_COUNT
from sv_pgs.progress import log
from sv_pgs.scale_mixture_ep import device_scope
from sv_pgs.stage2_wiring import fit_models


@dataclass(frozen=True)
class Derived:
    """Derived columns over the arm's records: ``weights`` (derived x records, sparse), ``offset`` per derived column,
    ``replaced`` (the records they stand for, left out of the store), each column's ``variant_class`` (its
    ``VariantClass``), whether it is ``structural`` (its score counts in the harness's structural part) and a ``name``
    for the store cache and the log."""

    weights: sparse.csr_matrix
    offset: np.ndarray
    replaced: np.ndarray
    variant_class: tuple[VariantClass, ...]
    structural: np.ndarray
    name: str


def derived_values(data, derived: Derived) -> np.ndarray:
    """The derived columns' values [derived, samples] on ``data``'s samples, from the records they combine."""
    used = np.flatnonzero(np.asarray(derived.weights.getnnz(axis=0)) > 0)
    weights = derived.weights[:, used].tocsr()
    values = np.repeat(derived.offset[:, None].astype(np.float64), data.n_samples, axis=1)
    for first in range(0, used.shape[0], full.BLOCK_ROWS):
        dosage = np.asarray(data.codes(used[first:first + full.BLOCK_ROWS]), dtype=np.float64) / CODES_PER_DOSAGE
        values += weights[:, first:first + full.BLOCK_ROWS] @ dosage
    return values


def _layout(variants: dict, derived: Derived) -> tuple[np.ndarray, np.ndarray]:
    """The store's rows in position order: (kind, index) with kind 0 a record and 1 a derived column, which sits at the
    first position of the records it combines."""
    positions = np.asarray(variants["pos"], dtype=np.int64)
    records = np.setdiff1d(np.arange(positions.shape[0]), derived.replaced)
    weights = derived.weights.tocsr()
    first = np.array([positions[weights.indices[weights.indptr[row]:weights.indptr[row + 1]]].min() for row in range(weights.shape[0])], dtype=np.int64)
    kind = np.concatenate([np.zeros(records.shape[0], dtype=np.int64), np.ones(first.shape[0], dtype=np.int64)])
    index = np.concatenate([records, np.arange(first.shape[0])])
    order = np.argsort(np.concatenate([positions[records], first]), kind="stable")
    return kind[order], index[order]


def _combined(variants: dict, derived: Derived, kind: np.ndarray, index: np.ndarray) -> dict:
    """The public fields of every store row: a record's own; a derived column's from its records (the largest
    in_gene, in_exon and in_repeat, the smallest TSS distance, the lowest reported r^2, the first position and
    genetic position), with no length of its own (the missing state)."""
    weights = abs(derived.weights.tocsr())

    def reduce(name: str, how) -> np.ndarray:
        values = np.asarray(variants[name], dtype=np.float64)
        return np.array([how(values[weights.indices[weights.indptr[row]:weights.indptr[row + 1]]]) for row in range(weights.shape[0])])

    rows = {}
    for name, how in (("pos", np.min), ("cm", np.min), ("ref_len", np.max), ("alt_len", np.max), ("in_gene", np.max), ("in_exon", np.max),
                      ("in_repeat", np.max), ("log_tss_distance", np.min), ("imputation_info", np.min)):
        own = np.asarray(variants[name], dtype=np.float64)
        rows[name] = np.where(kind == 0, own[np.where(kind == 0, index, 0)], reduce(name, how)[np.where(kind == 1, index, 0)])
    for name in ("log_sv_length", "len_change"):
        own = np.asarray(variants[name], dtype=np.float64)
        rows[name] = np.where(kind == 0, own[np.where(kind == 0, index, 0)], np.nan)
    return rows


def build_store(train, work: Path, derived: Derived) -> tuple[Path, np.ndarray, np.ndarray, np.ndarray]:
    """The derived arm's one-half store; returns (path, kind, index, (lo, span) per derived column)."""
    variants = train.variants
    kind, index = _layout(variants, derived)
    values = derived_values(train, derived)
    low = values.min(axis=1)
    span = values.max(axis=1) - low
    codes_of_derived = np.round(np.divide(values - low[:, None], span[:, None], out=np.zeros_like(values), where=span[:, None] > 0) * MAXIMUM_CODE)
    codes_of_derived = codes_of_derived.astype(np.uint8)
    del values
    rows = _combined(variants, derived, kind, index)
    record_classes = full.variant_classes(np.asarray(variants["cls"]), np.asarray(variants["len_change"]))
    classes = np.where(kind == 0, record_classes[np.where(kind == 0, index, 0)], np.array([VARIANT_CLASSES.index(name) for name in derived.variant_class])[np.where(kind == 1, index, 0)]).astype(np.uint8)
    count = kind.shape[0]

    def blocks():
        for first in range(0, count, full.BLOCK_ROWS):
            span_kind, span_index = kind[first:first + full.BLOCK_ROWS], index[first:first + full.BLOCK_ROWS]
            block = np.empty((span_kind.shape[0], train.n_samples), dtype=np.uint8)
            own = span_kind == 0
            if own.any():
                block[own] = np.asarray(train.codes(span_index[own]), dtype=np.uint8)
            block[~own] = codes_of_derived[span_index[~own]]
            yield block

    sums = np.zeros(count, dtype=np.uint64)
    squares = np.zeros(count, dtype=np.uint64)
    for first, block in zip(range(0, count, full.BLOCK_ROWS), blocks()):
        wide = block.astype(np.uint64)
        sums[first:first + full.BLOCK_ROWS] = wide.sum(axis=1)
        squares[first:first + full.BLOCK_ROWS] = (wide * wide).sum(axis=1)
    ids = [f"v{row}" if row_kind == 0 else f"{derived.name}{row}" for row_kind, row in zip(kind.tolist(), index.tolist())]
    id_bytes = np.frombuffer("".join(ids).encode(), dtype=np.uint8)
    annotations = full.store_annotations(rows)
    table = VariantTable(
        chromosome=np.full(count, full.CHROMOSOME, dtype=np.int8),
        position=rows["pos"].astype(np.int64),
        genetic_position_cm=rows["cm"],
        ref_length=rows["ref_len"].astype(np.int32),
        alt_length=rows["alt_len"].astype(np.int32),
        variant_class=classes,
        codes_per_unit=np.full(count, CODES_PER_DOSAGE, dtype=np.uint8),
        value_origin=np.zeros(count, dtype=np.int64),
        group_first=np.arange(count, dtype=np.int64),
        sum_code=sums,
        sum_code2=squares,
        annotations=annotations,
        annotation_legends={},
        id_bytes=id_bytes,
        id_offsets=np.concatenate([[0], np.cumsum([len(name) for name in ids])]).astype(np.int64),
    )
    path = work / "store"
    started = time.time()
    write_dosage_store(path, train.n_samples, table, blocks(), codec="raw")
    log(f"svpgs_full_{derived.name}: {int((kind == 1).sum()):,} derived columns for {derived.replaced.shape[0]:,} records; store written in {time.time() - started:.0f} s")
    return path, kind, index, np.column_stack([low, span])


class Model:
    def __init__(self, scoring, kind, index, scaling, derived: Derived, structural_records: np.ndarray, profile: dict) -> None:
        self.scoring, self.kind, self.index, self.scaling, self.derived = scoring, kind, index, scaling, derived
        self.structural_records, self.profile = structural_records, profile

    def score(self, test) -> dict:
        scoring = self.scoring
        store_rows = np.asarray(scoring.store_rows)
        values = derived_values(test, self.derived)
        low, span = self.scaling[:, 0], self.scaling[:, 1]
        derived_codes = np.divide(values - low[:, None], span[:, None], out=np.zeros_like(values), where=span[:, None] > 0) * MAXIMUM_CODE
        total = np.zeros(test.n_samples)
        structural = np.zeros(test.n_samples)
        for first in range(0, store_rows.shape[0], full.BLOCK_ROWS):
            rows = store_rows[first:first + full.BLOCK_ROWS]
            kind, index = self.kind[rows], self.index[rows]
            codes = np.empty((rows.shape[0], test.n_samples))
            own = kind == 0
            if own.any():
                codes[own] = np.asarray(test.codes(index[own]), dtype=np.float64)
            codes[~own] = derived_codes[index[~own]]
            standardized = (codes - SIGNED_CODE_OFFSET - scoring.signed_means[first:first + full.BLOCK_ROWS, None]) / scoring.signed_scales[first:first + full.BLOCK_ROWS, None]
            weights = scoring.coefficients[first:first + full.BLOCK_ROWS]
            total += weights @ standardized
            members = np.where(own, self.structural_records[np.where(own, index, 0)], self.derived.structural[np.where(own, 0, index)])
            structural += weights[members] @ standardized[members]
        return {"total": total, "structural": structural}


def fit_derived(train, derived: Derived) -> Model:
    """``svpgs_full.fit`` on the derived arm's store (cached per arm as ``svpgs_full.cached_store`` does, the key
    naming the derivation)."""
    work = Path(tempfile.mkdtemp(prefix=f"svpgs_full_{derived.name}_", dir=os.environ.get("TMPDIR")))
    try:
        cache = os.environ.get(full.STORE_CACHE_VARIABLE)
        if cache:
            digest = hashlib.sha256(full._store_key(train).encode())
            for part in (derived.weights.indptr, derived.weights.indices, derived.weights.data, derived.offset, derived.replaced):
                digest.update(np.ascontiguousarray(part).tobytes())
            root = Path(cache) / f"{derived.name}-{digest.hexdigest()[:24]}"
            if not (root / "COMPLETE").exists():
                staging = Path(tempfile.mkdtemp(prefix=f"{root.name}.", dir=cache))
                _path, kind, index, scaling = build_store(train, staging, derived)
                np.savez(staging / "layout.npz", kind=kind, index=index, scaling=scaling)
                (staging / "COMPLETE").write_text("")
                try:
                    staging.rename(root)
                except OSError:
                    shutil.rmtree(staging, ignore_errors=True)
            shutil.copytree(root / "store", work / "store")
            layout = np.load(root / "layout.npz")
            store_path, kind, index, scaling = work / "store", layout["kind"], layout["index"], layout["scaling"]
        else:
            store_path, kind, index, scaling = build_store(train, work, derived)
        budget = full.task_budget()
        phenotype = np.asarray(train.phenotype, dtype=np.float64)
        covariates = np.column_stack([np.ones(train.n_samples), np.asarray(train.covariates, dtype=np.float64)])
        started = time.time()
        (work / "fit").mkdir()
        with DosageStore.open(store_path) as store, device_scope(full._array_module(budget)):
            fitted = fit_models(
                store=store, store_columns=np.arange(train.n_samples, dtype=np.int64), covariates=covariates,
                covariate_columns=np.ones((1, covariates.shape[1]), dtype=bool), targets=phenotype[:, None],
                training=np.ones((train.n_samples, 1), dtype=bool), trait_types=[TraitType.QUANTITATIVE], log_variance_offset=None,
                budget=budget, work_dir=work / "fit", seed=20260922, draw_count=DRAW_COUNT, inference="mean_field",
            )
        (scoring,) = fitted.scoring
        profile = {"fit_seconds": time.time() - started, "device": budget.device_kind, "rows": int(np.asarray(scoring.store_rows).shape[0]),
                   "derived_columns": int((kind == 1).sum()), "replaced_records": int(derived.replaced.shape[0])}
        log(f"svpgs_full_{derived.name}: {profile}")
        return Model(scoring, kind, index, scaling, derived, np.asarray(train.variants["cls"]) >= 2, profile)
    finally:
        shutil.rmtree(work, ignore_errors=True)
