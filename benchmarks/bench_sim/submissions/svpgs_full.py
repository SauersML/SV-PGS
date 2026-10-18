"""bench-sim submission: SV-PGS by the full-data route (``stage2_wiring.fit_models``) on the arm's observed codes.

The training samples' observed codes become a one-half dosage store under the task's TMPDIR, with the public
per-record fields as its variant table and the arm's reported imputation r^2 as its ``quality`` column (a record
with no reported r^2, a called simple site on the Beagle arm, is measured exactly by the store's contract). The
store is then fitted exactly as a cohort store is (Stage 0 statistics, the LD blocks, the mean-field fixed points
of the learned scale-mixture prior with one mixing density per variant class), on the device the machine's budget
finds. The test samples are scored from the fit's standardized-column coefficients, and the structural part is the
score carried by TR and SV records.

A binary trait uses the Gaussian working likelihood on its 0/1 phenotype in this submission, and
the harness scores its total on the liability scale.
"""

from __future__ import annotations

import hashlib
import json
import os
import resource
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
from sv_pgs.model_acceptance import subset_refusals
from sv_pgs.progress import log
from sv_pgs.scale_mixture_ep import device_scope
from sv_pgs.stage2_wiring import fit_models
from sv_pgs.store_converter import overlap_group_first

BLOCK_ROWS = 4096
"""Rows per read of the harness's observed codes: 4096 x 40,000 training samples is a 164 MB block."""
CHROMOSOME = 22
"""bench-sim v1 is chr22 only (SUBMIT.md)."""
TR_CLASS = 2
"""bench-sim's TR class (``cls``): an indel in a repeat."""


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


def site_group_first(position: np.ndarray, reference_length: np.ndarray, cls: np.ndarray, repeat_locus: np.ndarray) -> np.ndarray:
    """Each store row's ``group_first`` (rows in position order): the records whose reference spans overlap, a
    same-POS set among them, and each public repeat_locus's TR-class records (class 2, bench-sim's indels in a
    repeat, the records its TR truths are drawn from) are one unbreakable group (``store_converter.overlap_group_first``),
    the store's site key as a converted store's bubble, same-POS set and TR locus are. A locus's other records join
    only by overlap: with them, records spanning several loci chained chr22 into groups of up to 18,610 records
    (937 kb), past any LD block cap; the TR records alone give at most 1,305 (55 kb)."""
    position = np.asarray(position, dtype=np.int64)
    tr_locus = np.where(np.asarray(cls) == TR_CLASS, np.asarray(repeat_locus, dtype=np.int64), -1)
    return overlap_group_first(position, position + np.asarray(reference_length, dtype=np.int64), tr_locus)


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
        # Every uint8 code's square fits uint16; only the accumulated totals need uint64.
        # At 40,000 people this avoids two 1.3 GB uint64 temporaries per block.
        wide = block.astype(np.uint16)
        sums[first:first + BLOCK_ROWS] = wide.sum(axis=1, dtype=np.uint64)
        squares[first:first + BLOCK_ROWS] = (wide * wide).sum(axis=1, dtype=np.uint64)
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
        group_first=site_group_first(*(np.asarray(variants[name])[order] for name in ("pos", "ref_len", "cls", "repeat_locus"))),
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


STORE_CACHE_VARIABLE = "SVPGS_STORE_CACHE"
"""A directory where the training store of an arm is kept between scenarios: the store is the arm's training codes
and public fields only (no phenotype), the same for every scenario on the arm."""


def _store_key(train) -> str:
    """Bind every public variant field and every training code.

    Sampled rows cannot identify a cache: a changed unsampled dosage or annotation
    must not silently reuse a previous simulation's training store. The codes are bound by the harness's identity of
    their source (the observed file's path, size and modification time, and the records and samples read), which any
    change to a code changes, so a cached store is found without a pass over the codes.
    """
    variants = train.variants
    digest = hashlib.sha256()
    digest.update(f"svpgs-simulation-store-v3:{train.n_samples}:{train.n_variants}".encode())
    for name in sorted(variants):
        values = np.ascontiguousarray(np.asarray(variants[name]))
        if values.dtype.hasobject:
            raise ValueError("Public variant metadata must use portable numeric or string arrays")
        digest.update(json.dumps([name, values.dtype.str, values.shape]).encode())
        digest.update(values.tobytes())
    digest.update(train.codes_identity().encode())
    return digest.hexdigest()


def cached_store(train, work: Path) -> tuple[Path, np.ndarray]:
    """The training store under ``work``: built once per arm into ``$SVPGS_STORE_CACHE`` (atomically, by rename),
    then copied to the task's local disk, where the fit streams it every sweep. Without the variable, built in place."""
    cache = os.environ.get(STORE_CACHE_VARIABLE)
    if not cache:
        return build_store(train, work)
    order = np.argsort(train.variants["pos"], kind="stable").astype(np.int64)
    root = Path(cache) / _store_key(train)
    if not (root / "store" / "COMPLETE").exists():
        staging = Path(tempfile.mkdtemp(prefix=f"{root.name}.", dir=cache))
        built, _order = build_store(train, staging)
        (built / "COMPLETE").write_text("")
        from sv_pgs.content_identity import tree_digest
        (built / "CACHE_COMPLETE.json").write_text(json.dumps({"sha256": tree_digest(built)}))
        try:
            staging.rename(root)
        except OSError:
            shutil.rmtree(staging, ignore_errors=True)
            if not (root / "store" / "COMPLETE").exists():
                raise
    started = time.time()
    shutil.copytree(root / "store", work / "store")
    log(f"svpgs_full: cached store {root.name} copied in {time.time() - started:.0f} s")
    from sv_pgs.content_identity import tree_digest
    receipt = json.loads((work / "store" / "CACHE_COMPLETE.json").read_text())
    if tree_digest(work / "store") != receipt["sha256"]:
        raise ValueError(f"the cached store {root} has changed since conversion")
    return work / "store", order


def task_budget() -> ComputeBudget:
    """The machine's budget (its device where it has one), with the host share capped by the runner's allotment
    less what this process already holds."""
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
    return fit_with(train, "mean_field")


def fit_with(train, inference: str, *, annotated_prior: bool = True) -> Model:
    """The fit by ``inference``'s fixed points (``stage2_wiring.fit_models``), with its profile: time, the process's
    peak resident set and the device pool's bytes at the end. A fit the acceptance check quarantines raises."""
    (outcome,) = fit_phenotypes(train, train.phenotype[:, None], inference, annotated_prior=annotated_prior)
    if isinstance(outcome, Exception):
        raise outcome
    return outcome


def fit_phenotypes(train, phenotypes: np.ndarray, inference: str = "mean_field", *, annotated_prior: bool = True) -> list:
    """One model per column of ``phenotypes`` (the training samples' phenotypes of several scenarios on the same
    genotypes), fitted together: ``fit_models`` fits the models that share training rows, covariates and likelihood as
    one group, so the store, Stage 0's LD blocks and every pass over the codes serve all of them. Each model is
    accepted or quarantined on its own; a quarantined model's entry is the ValueError ``require_accepted`` raises."""
    phenotypes = np.asarray(phenotypes, dtype=np.float64)
    count = phenotypes.shape[1]
    work = Path(tempfile.mkdtemp(prefix="svpgs_full_", dir=os.environ.get("TMPDIR")))
    try:
        store_path, order = cached_store(train, work)
        budget = task_budget()
        covariates = np.column_stack([np.ones(train.n_samples), np.asarray(train.covariates, dtype=np.float64)])
        started = time.time()
        (work / "fit").mkdir()
        with DosageStore.open(store_path) as store, device_scope(_array_module(budget)):
            fitted = fit_models(
                store=store,
                store_columns=np.arange(train.n_samples, dtype=np.int64),
                covariates=covariates,
                covariate_columns=np.ones((count, covariates.shape[1]), dtype=bool),
                targets=phenotypes,
                training=np.ones((train.n_samples, count), dtype=bool),
                trait_types=[TraitType.QUANTITATIVE] * count,
                log_variance_offset=None,
                budget=budget,
                work_dir=work / "fit",
                seed=20260922,
                draw_count=DRAW_COUNT,
                inference=inference,
                annotated_prior=annotated_prior,
            )
        seconds = time.time() - started
        certificate = fitted.certificate
        from types import SimpleNamespace
        from sv_pgs.fit_model import certificate_parts
        from sv_pgs.model_acceptance import acceptance, require_accepted
        parts = certificate_parts(certificate, count)
        structural = np.asarray(train.variants["cls"]) >= 2
        outcomes = []
        for index, scoring in enumerate(fitted.scoring):
            # the model's own certificate terms: every per-model array's row ``index``
            terms = {key: (np.asarray(value)[index:index + 1] if np.ndim(value) and np.shape(value)[0] == count else value)
                     for key, value in parts.terms.items()}
            refusals = subset_refusals(parts.refusals, [index])
            profile = {
                "fit_seconds": seconds,
                "models_fitted_together": count,
                "device": budget.device_kind,
                "rows": int(np.asarray(scoring.store_rows).shape[0]),
                "noise_variance": float(fitted.noise_variance[index]),
                "phenotype_variance": float(phenotypes[:, index].var()),
                "outer_criterion_met": None if certificate.outer_criterion_met is None else bool(np.asarray(certificate.outer_criterion_met)[index]),
                "remaining_gain": float(np.asarray(certificate.remaining_gain)[index]) if np.asarray(certificate.remaining_gain).size else None,
                "inference": inference,
                "refusals": len(refusals),
                # The outer loop's own record, so an uncertified fit shows how it ended: accepted steps, refused trials, those
                # with no fixed point, and the remaining gain at its last evaluations.
                "outer_iterations": int(np.asarray(certificate.outer_iterations)[index]),
                "halvings": int(np.asarray(certificate.halvings)[index]),
                "unresolved": int(np.asarray(certificate.unresolved)[index]),
                "outer_history_tail": list(map(float, certificate.outer_history[index][-8:])),
                "peak_host_bytes": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024,
                "device_pool_bytes": None if _array_module(budget) is None else int(_array_module(budget).get_default_memory_pool().total_bytes()),
            }
            state = SimpleNamespace(model_names=(f"simulation {index}",), certificate=terms, refusals=refusals, scoring=(scoring,))
            profile["acceptance"] = acceptance(state)
            profile["annotated_prior"] = annotated_prior
            log(f"svpgs_full: model {index}: {profile}")
            try:
                require_accepted(state)
            except ValueError as error:
                outcomes.append(error)
                continue
            outcomes.append(Model(scoring, order, structural, profile))
        return outcomes
    finally:
        shutil.rmtree(work, ignore_errors=True)
