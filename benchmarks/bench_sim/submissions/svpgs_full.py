"""bench-sim submission: SV-PGS by the full-data route (``stage2_wiring.fit_models``) on the arm's observed codes.

The training samples' observed codes become a one-half dosage store under the task's TMPDIR, with the public
per-record fields as its variant table and the arm's reported imputation r^2 as its ``quality`` column (a record
with no reported r^2, a called simple site on the Beagle arm, is measured exactly by the store's contract). The
store is then fitted exactly as a cohort store is (Stage 0 statistics, the LD blocks, the mean-field fixed points
of the learned scale-mixture prior with one mixing density per variant class), on the device the machine's budget
finds. The test samples are scored from the fit's standardized-column coefficients, and the structural part is the
score carried by TR and SV records.

A binary trait is fitted as a quantitative one on its 0/1 phenotype (``fit_models`` has no binary likelihood), and
the harness scores its total on the liability scale.
"""

from __future__ import annotations

import os
import shutil
import tempfile
import time
from pathlib import Path

import numpy as np

from benchmarks.svpgs_method import RUNQ_MEMORY_VARIABLE, _resident_bytes
from sv_pgs.compute_budget import ComputeBudget, detect_compute_budget
from sv_pgs.config import TraitType, VariantClass
from sv_pgs.dosage_store import CODES_PER_DOSAGE, VARIANT_CLASSES, DosageStore, VariantTable, write_dosage_store
from sv_pgs.fast_scoring import SIGNED_CODE_OFFSET, ScoringModel
from sv_pgs.fit_model import DRAW_COUNT
from sv_pgs.progress import log
from sv_pgs.scale_mixture_ep import device_scope
from sv_pgs.stage2_wiring import fit_models

BLOCK_ROWS = 4096
"""Rows per read of the harness's observed codes: 4096 x 40,000 training samples is a 164 MB block."""
CHROMOSOME = 22
"""bench-sim v1 is chr22 only (SUBMIT.md)."""


def variant_classes(cls: np.ndarray, len_change: np.ndarray) -> np.ndarray:
    """The store's class code per record: SNV; an INDEL or SV a deletion or insertion by the sign of its length
    change (an SV with no length change is a complex one); a TR a repeat locus. The prior learns one mixing
    density per class present."""
    codes = np.empty(cls.shape[0], dtype=np.uint8)
    for index, (name, members) in enumerate(
        (
            (VariantClass.SNV, cls == 0),
            (VariantClass.DELETION, (cls == 1) & (len_change < 0)),
            (VariantClass.INSERTION, (cls == 1) & (len_change >= 0)),
            (VariantClass.STR_VNTR_REPEAT, cls == 2),
            (VariantClass.DELETION, (cls == 3) & (len_change < 0)),
            (VariantClass.INSERTION, (cls == 3) & (len_change > 0)),
            (VariantClass.OTHER_COMPLEX_SV, (cls == 3) & (len_change == 0)),
        )
    ):
        codes[members] = VARIANT_CLASSES.index(name)
    return codes


def quality(imputation_info: np.ndarray) -> np.ndarray:
    """The store's reliability column: the arm's reported r^2 where it reports one, else exact measurement."""
    info = np.asarray(imputation_info, dtype=np.float64)
    return np.where(np.isfinite(info), np.clip(info, 0.0, 1.0), 1.0)


def store_annotations(variants) -> dict[str, np.ndarray]:
    """The store's sidecar columns: the arm's reliability as ``quality`` (the prior's offset) and the public
    per-record annotations that vary over the records (a constant one carries no prior information), which the prior
    reads as its annotation design (``annotation_design``)."""
    columns = {"quality": quality(variants["imputation_info"])}
    for name in ("in_gene", "in_exon", "in_repeat", "log_tss_distance", "log_sv_length", "len_change"):
        values = np.asarray(variants[name], dtype=np.float64)
        if np.unique(values[np.isfinite(values)]).shape[0] > 1:
            columns[name] = values
    return columns


def build_store(train, work: Path) -> tuple[Path, np.ndarray]:
    """Write the training samples' codes as a one-half store in position order; returns (path, order) with
    order[store row] the harness row."""
    variants = train.variants
    n_var, n_train = train.n_variants, train.n_samples
    order = np.argsort(variants["pos"], kind="stable").astype(np.int64)
    codes = np.lib.format.open_memmap(work / "codes.npy", mode="w+", dtype=np.uint8, shape=(n_var, n_train))
    sums = np.zeros(n_var, dtype=np.uint64)
    squares = np.zeros(n_var, dtype=np.uint64)
    started = time.time()
    for first in range(0, n_var, BLOCK_ROWS):
        rows = order[first:first + BLOCK_ROWS]
        block = np.asarray(train.codes(rows), dtype=np.uint8)
        codes[first:first + BLOCK_ROWS] = block
        wide = block.astype(np.uint64)
        sums[first:first + BLOCK_ROWS] = wide.sum(axis=1)
        squares[first:first + BLOCK_ROWS] = (wide * wide).sum(axis=1)
    codes.flush()
    log(f"svpgs_full: {n_var:,} records x {n_train:,} training samples read in {time.time() - started:.0f} s")
    ids = [f"v{row}" for row in order]
    id_bytes = np.frombuffer("".join(ids).encode(), dtype=np.uint8)
    table = VariantTable(
        chromosome=np.full(n_var, CHROMOSOME, dtype=np.int8),
        position=np.asarray(variants["pos"], dtype=np.int64)[order],
        genetic_position_cm=np.asarray(variants["cm"], dtype=np.float64)[order],
        ref_length=np.asarray(variants["ref_len"], dtype=np.int32)[order],
        alt_length=np.asarray(variants["alt_len"], dtype=np.int32)[order],
        variant_class=variant_classes(np.asarray(variants["cls"]), np.asarray(variants["len_change"]))[order],
        codes_per_unit=np.full(n_var, CODES_PER_DOSAGE, dtype=np.uint8),
        value_origin=np.zeros(n_var, dtype=np.int64),
        group_first=np.arange(n_var, dtype=np.int64),
        sum_code=sums,
        sum_code2=squares,
        annotations={name: values[order] for name, values in store_annotations(variants).items()},
        annotation_legends={},
        id_bytes=id_bytes,
        id_offsets=np.concatenate([[0], np.cumsum([len(name) for name in ids])]).astype(np.int64),
    )
    path = work / "store"
    started = time.time()
    write_dosage_store(path, n_train, table, (codes[first:first + BLOCK_ROWS] for first in range(0, n_var, BLOCK_ROWS)), codec="raw")
    del codes
    (work / "codes.npy").unlink()
    log(f"svpgs_full: store written in {time.time() - started:.0f} s")
    return path, order


def task_budget() -> ComputeBudget:
    """The machine's budget (its device where it has one), with the host share capped by the runner's allotment
    less what this process already holds, as ``svpgs_method.process_budget`` caps it."""
    machine = detect_compute_budget()
    allotment = os.environ.get(RUNQ_MEMORY_VARIABLE)
    host_bytes = machine.host_bytes
    if allotment is not None:
        host_bytes = min(host_bytes, int(allotment) - _resident_bytes())
    if host_bytes <= 0:
        raise MemoryError("the task's memory allotment is already spent by what the process holds.")
    return ComputeBudget(
        device_kind=machine.device_kind,
        device_ids=machine.device_ids,
        device_names=machine.device_names,
        device_bytes=machine.device_bytes,
        device_compute_capabilities=machine.device_compute_capabilities,
        host_bytes=host_bytes,
        cpu_threads=machine.cpu_threads,
    )


def _array_module(budget: ComputeBudget):
    if budget.device_kind != "cuda":
        return None
    import cupy  # noqa: PLC0415 - only where the budget found a device

    return cupy


class Model:
    def __init__(self, scoring: ScoringModel, order: np.ndarray, structural: np.ndarray, profile: dict) -> None:
        self.scoring, self.order, self.structural, self.profile = scoring, order, structural, profile

    def score(self, test) -> dict:
        scoring = self.scoring
        total = np.zeros(test.n_samples)
        structural = np.zeros(test.n_samples)
        store_rows = np.asarray(scoring.store_rows)
        for first in range(0, store_rows.shape[0], BLOCK_ROWS):
            rows = store_rows[first:first + BLOCK_ROWS]
            codes = np.asarray(test.codes(self.order[rows]), dtype=np.float64)
            standardized = (codes - SIGNED_CODE_OFFSET - scoring.signed_means[first:first + BLOCK_ROWS, None]) / scoring.signed_scales[first:first + BLOCK_ROWS, None]
            weights = scoring.coefficients[first:first + BLOCK_ROWS]
            total += weights @ standardized
            members = self.structural[self.order[rows]]
            structural += weights[members] @ standardized[members]
        return {"total": total, "structural": structural}


def fit(train) -> Model:
    work = Path(tempfile.mkdtemp(prefix="svpgs_full_", dir=os.environ.get("TMPDIR")))
    try:
        store_path, order = build_store(train, work)
        budget = task_budget()
        phenotype = np.asarray(train.phenotype, dtype=np.float64)
        covariates = np.column_stack([np.ones(train.n_samples), np.asarray(train.covariates, dtype=np.float64)])
        started = time.time()
        (work / "fit").mkdir()
        with DosageStore.open(store_path) as store, device_scope(_array_module(budget)):
            fitted = fit_models(
                store=store,
                store_columns=np.arange(train.n_samples, dtype=np.int64),
                covariates=covariates,
                covariate_columns=np.ones((1, covariates.shape[1]), dtype=bool),
                targets=phenotype[:, None],
                training=np.ones((train.n_samples, 1), dtype=bool),
                trait_types=[TraitType.QUANTITATIVE],
                log_variance_offset=None,
                budget=budget,
                work_dir=work / "fit",
                seed=20260922,
                draw_count=DRAW_COUNT,
            )
        (scoring,) = fitted.scoring
        certificate = fitted.certificate
        profile = {
            "fit_seconds": time.time() - started,
            "device": budget.device_kind,
            "rows": int(np.asarray(scoring.store_rows).shape[0]),
            "noise_variance": float(fitted.noise_variance[0]),
            "phenotype_variance": float(phenotype.var()),
            "certified": None if certificate.outer_criterion_met is None else bool(np.all(certificate.outer_criterion_met)),
            "remaining_gain": float(np.max(certificate.remaining_gain)) if np.asarray(certificate.remaining_gain).size else None,
        }
        log(f"svpgs_full: {profile}")
        return Model(scoring, order, np.asarray(train.variants["cls"]) >= 2, profile)
    finally:
        shutil.rmtree(work, ignore_errors=True)
