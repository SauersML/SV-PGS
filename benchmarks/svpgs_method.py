"""SV-PGS as a benchmark method, for both harnesses.

- **bench-sim:** ``fit(train) -> model`` and ``model.score(test)``, per benchmarks/bench_sim/SUBMIT.md.
- **bench-real:** ``fit_expression(train) -> predictor`` and ``predictor.predict(genotypes)``, per
  benchmarks/bench_real/harness.py. The method spec is ``benchmarks/svpgs_method.py:fit_expression``.

Each training view becomes a one-chromosome dosage store holding the harness's public variant table as the store's
prior columns. Codes are dosage x 127, the store's own format, and bench-real's 0/1/2 calls map onto codes exactly.
``sv_pgs.fit_model.fit`` then fits one model on every training sample, and the test view is scored with that model's
standardized effects:
- bench-sim's test codes go through the production scorer (``fast_scoring.score_genetic``), read straight from the
  harness's code view;
- bench-real's test genotypes are dosages, not codes: its SV-credit arm sets SV columns to their training means. So
  they are scored in closed form from the same ``ScoringModel``, with no rounding to codes.

The prediction is the posterior-mean genetic score (plus the fitted intercept for bench-real, whose phenotype is
already residualized). Covariate effects are left out, as the harnesses adjust for covariates themselves.

Two bench-real ablation arms, benchmark adapters only (no production path), withhold prior information from the
store they fit on:
- ``fit_expression_no_sv_terms``: the SV-specific prior terms. Every record is classed by the small-variant rule
  from its allele lengths (so SNVs, deletions and insertions keep their classes, and no SV type is used), and the
  SV-type, source, length and length-change columns are withheld; the TSS distance stays.
- ``fit_expression_no_annotations``: every annotation. One class for every record and no annotation columns, so the
  prior sees only the genotypes, as mr.ash does.

Both harnesses load this file without registering it as a module, so it has no ``from __future__ import
annotations``: dataclasses would then look the module up to resolve its string annotations, and fail.
"""

import tempfile
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable, Iterator, Mapping, Sequence

import numpy as np

from sv_pgs import fit_model
from sv_pgs.compute_budget import ComputeBudget, detect_compute_budget
from sv_pgs.config import TraitType, VariantClass
from sv_pgs.dosage_store import (
    CODES_PER_DOSAGE,
    DEFAULT_INNER_CHUNK_ROWS,
    MAXIMUM_CODE,
    DosageStore,
    VariantTable,
    chromosome_number,
    write_dosage_store,
)
from sv_pgs.fast_scoring import SIGNED_CODE_OFFSET, ScoringModel, ScoringPlan, score_genetic
from sv_pgs.variant_typing import normalize_variant_token, structural_variant_class_from_token

_CLASS_CODES = {variant_class: index for index, variant_class in enumerate(VariantClass)}


def _length_class(reference_length: np.ndarray, alternate_length: np.ndarray) -> np.ndarray:
    """A small variant's class from its allele lengths, as synthetic_store types 1kGP indels."""
    classes = np.full(reference_length.shape[0], _CLASS_CODES[VariantClass.OTHER_COMPLEX_SV], dtype=np.uint8)
    classes[reference_length > alternate_length] = _CLASS_CODES[VariantClass.DELETION]
    classes[reference_length < alternate_length] = _CLASS_CODES[VariantClass.INSERTION]
    classes[(reference_length == 1) & (alternate_length == 1)] = _CLASS_CODES[VariantClass.SNV]
    return classes


def _same_position_groups(position: np.ndarray) -> np.ndarray:
    """Each row's first row with the same position: same-POS records form one unbreakable group."""
    rows = np.arange(position.shape[0], dtype=np.int64)
    starts = np.concatenate([[True], position[1:] != position[:-1]])
    return np.maximum.accumulate(np.where(starts, rows, 0))


def _informative_annotations(annotations: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    """The annotations that vary over the table; a constant one carries no prior information."""
    return {name: values for name, values in annotations.items() if np.unique(values).shape[0] > 1}


def _categorical(values: np.ndarray) -> tuple[np.ndarray, tuple[str, ...]]:
    legend, codes = np.unique(np.asarray(values, dtype=str), return_inverse=True)
    return codes.astype(np.int32), tuple(str(level) for level in legend)


def write_store(
    root: Path,
    *,
    chromosome: str,
    position: np.ndarray,
    genetic_position_cm: np.ndarray,
    reference_length: np.ndarray,
    alternate_length: np.ndarray,
    variant_class: np.ndarray,
    annotations: Mapping[str, np.ndarray],
    annotation_legends: Mapping[str, tuple[str, ...]],
    code_rows: Callable[[slice], np.ndarray],
    sample_count: int,
) -> DosageStore:
    """Write a one-half store of ``code_rows`` (uint8 codes [rows, samples] of a row slice) and open it."""
    count = position.shape[0]
    if np.any(np.diff(position) < 0):
        raise ValueError("store rows must be in position order.")
    blocks = [slice(start, min(start + DEFAULT_INNER_CHUNK_ROWS, count)) for start in range(0, count, DEFAULT_INNER_CHUNK_ROWS)]
    sums = np.zeros(count, dtype=np.uint64)
    squares = np.zeros(count, dtype=np.uint64)
    for block in blocks:
        codes = code_rows(block).astype(np.int64)
        if codes.shape != (block.stop - block.start, sample_count) or codes.min(initial=0) < 0 or codes.max(initial=0) > MAXIMUM_CODE:
            raise ValueError(f"code rows must be codes 0..{MAXIMUM_CODE} of shape [rows, {sample_count}].")
        sums[block] = codes.sum(axis=1)
        squares[block] = np.einsum("ij,ij->i", codes, codes)
    identifiers = "".join(f"{chromosome}-{row}" for row in range(count)).encode()
    lengths = np.array([len(f"{chromosome}-{row}") for row in range(count)], dtype=np.int64)
    table = VariantTable(
        chromosome=np.full(count, chromosome_number(chromosome), dtype=np.int8),
        position=np.asarray(position, dtype=np.int64),
        genetic_position_cm=np.asarray(genetic_position_cm, dtype=np.float64),
        ref_length=np.asarray(reference_length, dtype=np.int32),
        alt_length=np.asarray(alternate_length, dtype=np.int32),
        variant_class=np.asarray(variant_class, dtype=np.uint8),
        group_first=_same_position_groups(np.asarray(position)),
        sum_code=sums,
        sum_code2=squares,
        annotations=dict(annotations),
        annotation_legends=dict(annotation_legends),
        id_bytes=np.frombuffer(identifiers, dtype=np.uint8),
        id_offsets=np.concatenate([[0], np.cumsum(lengths)]).astype(np.int64),
    )
    write_dosage_store(root, sample_count, table, (code_rows(block) for block in blocks), codec="raw")
    return DosageStore.open(root)


def _fit_one(store: DosageStore, covariates: np.ndarray, covariate_names: Sequence[str], phenotype: np.ndarray, trait_type: TraitType, work_dir: Path, budget: ComputeBudget) -> ScoringModel:
    sample_count = phenotype.shape[0]
    research_ids = [f"train{sample}" for sample in range(sample_count)]
    model = fit_model.fit(
        store=store,
        store_columns=np.arange(sample_count, dtype=np.int64),
        covariates=covariates,
        covariate_names=covariate_names,
        covariate_columns=np.ones((1, len(covariate_names)), dtype=bool),
        targets=np.asarray(phenotype, dtype=np.float64)[:, None],
        training=np.ones((sample_count, 1), dtype=bool),
        model_names=("trait",),
        trait_types=(trait_type,),
        research_ids=research_ids,
        log_variance_offset=None,
        budget=budget,
        work_dir=work_dir,
        seed=fit_model.cohort_seed(store.root, research_ids),
    )
    return model.scoring[0]


def _restricted(scoring: ScoringModel, members: np.ndarray) -> ScoringModel:
    """The part of a model's genetic score carried by the rows ``members`` (a mask over its store rows)."""
    return replace(
        scoring,
        store_rows=scoring.store_rows[members],
        signed_means=scoring.signed_means[members],
        signed_scales=scoring.signed_scales[members],
        coefficients=scoring.coefficients[members],
        posterior_draws=scoring.posterior_draws[members],
    )


# ---------------------------------------------------------------------------
# bench-sim
# ---------------------------------------------------------------------------


def _bench_sim_class(variants: Mapping[str, np.ndarray], name: str) -> np.ndarray:
    """Rows of the public class ``name`` (SNV, INDEL, TR or SV, in the table's ``class_names``)."""
    return np.asarray(variants["cls"]) == [str(value) for value in variants["class_names"]].index(name)


def bench_sim_classes(variants: Mapping[str, np.ndarray]) -> np.ndarray:
    """SV-PGS classes of bench-sim's public classes: an INDEL or SV by its allele lengths, a TR as a repeat."""
    classes = _length_class(np.asarray(variants["ref_len"]), np.asarray(variants["alt_len"]))
    classes[_bench_sim_class(variants, "TR")] = _CLASS_CODES[VariantClass.STR_VNTR_REPEAT]
    return classes


def bench_sim_annotations(variants: Mapping[str, np.ndarray]) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """The store's prior columns and the rows kept.

    ``quality`` is the arm's imputation r^2 (GLIMPSE2 INFO or Beagle DR2), which the prior reads as the measurement
    reliability. A record at r^2 = 0 has prior effect variance r^2 x (its class's) = 0, so its effect is exactly zero
    and the record is left out.
    """
    quality = np.asarray(variants["imputation_info"], dtype=np.float64)
    if not np.all((quality >= 0.0) & (quality <= 1.0)):
        raise ValueError("imputation_info must be an r^2 in [0, 1] for every record.")
    kept = np.flatnonzero(quality > 0.0)
    annotations = {
        "quality": quality,
        "in_gene": np.asarray(variants["in_gene"], dtype=bool),
        "in_exon": np.asarray(variants["in_exon"], dtype=bool),
        "in_repeat": np.asarray(variants["in_repeat"], dtype=bool),
        "log_tss_distance": np.asarray(variants["log_tss_distance"], dtype=np.float64),
        "log_sv_length": np.asarray(variants["log_sv_length"], dtype=np.float64),
        "len_change": np.asarray(variants["len_change"], dtype=np.float64),
    }
    return _informative_annotations({name: values[kept] for name, values in annotations.items()}), kept


class _HarnessCodes:
    """A harness view's ``codes(rows)`` as ``fast_scoring.CodeBlockSource``, over the view's own samples."""

    def __init__(self, view: Any, records: np.ndarray) -> None:
        self.view = view
        self.records = records

    @property
    def sample_count(self) -> int:
        return int(self.view.n_samples)

    def iter_code_blocks(self, variant_ranges: Sequence[tuple[int, int]], buffers: Sequence[np.ndarray]) -> Iterator[tuple[int, int, np.ndarray]]:
        for start, stop in variant_ranges:
            yield start, stop, np.ascontiguousarray(self.view.codes(self.records[start:stop]))


@dataclass(frozen=True)
class BenchSimModel:
    total: ScoringModel
    structural: ScoringModel | None
    records: np.ndarray
    budget: ComputeBudget

    def score(self, test: Any) -> dict[str, np.ndarray]:
        models = [self.total] if self.structural is None else [self.total, self.structural]
        genetic = score_genetic(_HarnessCodes(test, self.records), ScoringPlan.from_models(models), self.budget)
        structural = genetic.means[:, 1] if self.structural is not None else np.zeros(test.n_samples)
        return {"total": genetic.means[:, 0], "structural": structural}


def fit(train: Any) -> BenchSimModel:
    """bench-sim: fit SV-PGS on the training view (SUBMIT.md)."""
    budget = detect_compute_budget()
    variants = train.variants
    annotations, kept = bench_sim_annotations(variants)
    with tempfile.TemporaryDirectory(prefix="svpgs-bench-sim.") as scratch:
        store = write_store(
            Path(scratch) / "store",
            chromosome="chr22",
            position=np.asarray(variants["pos"])[kept],
            genetic_position_cm=np.asarray(variants["cm"])[kept],
            reference_length=np.asarray(variants["ref_len"])[kept],
            alternate_length=np.asarray(variants["alt_len"])[kept],
            variant_class=bench_sim_classes(variants)[kept],
            annotations=annotations,
            annotation_legends={},
            code_rows=lambda rows: np.ascontiguousarray(train.codes(kept[rows])),
            sample_count=train.n_samples,
        )
        (Path(scratch) / "work").mkdir()
        trait_type = TraitType(train.trait_type)
        scoring = _fit_one(store, train.covariates, train.covariate_names, train.phenotype, trait_type, Path(scratch) / "work", budget)
    structural = (_bench_sim_class(variants, "TR") | _bench_sim_class(variants, "SV"))[kept][scoring.store_rows]
    return BenchSimModel(
        total=scoring,
        structural=_restricted(scoring, structural) if structural.any() else None,
        records=kept,
        budget=budget,
    )


# ---------------------------------------------------------------------------
# bench-real
# ---------------------------------------------------------------------------


def bench_real_classes(variants: Any) -> np.ndarray:
    """SV-PGS classes of bench-real rows: by the SVTYPE token where there is one (an indel's is its SV counterpart's),
    else by allele lengths."""
    reference_length, alternate_length = bench_real_allele_lengths(variants)
    classes = _length_class(reference_length, alternate_length)
    for token in np.unique(np.asarray(variants.sv_type, dtype=str)).tolist():
        normalized = normalize_variant_token(token)
        if normalized not in (None, "."):  # "." is VCF's missing value
            classes[np.asarray(variants.sv_type, dtype=str) == token] = _CLASS_CODES[structural_variant_class_from_token(normalized)]
    return classes


def bench_real_allele_lengths(variants: Any) -> tuple[np.ndarray, np.ndarray]:
    """REF length from the record's span (END = POS + len(REF) - 1) and ALT length from the length change; a symbolic
    ALT has change 0 in the harness, so its ALT length is recorded as the REF length."""
    reference_length = np.asarray(variants.end, dtype=np.int64) - np.asarray(variants.position, dtype=np.int64) + 1
    return reference_length, reference_length + np.asarray(variants.allele_length_change, dtype=np.int64)


# Each arm's annotation columns: the full model's, the SV-specific ones withheld, or none.
_ARM_ANNOTATIONS = {
    "full": ("log1p_tss_distance", "log1p_sv_length", "allele_length_change", "sv_type", "source"),
    "no_sv_terms": ("log1p_tss_distance",),
    "no_annotations": (),
}


def bench_real_classes_for_arm(variants: Any, arm: str) -> np.ndarray:
    """The full model's classes; the small-variant length rule for every record without the SV terms; one class
    without any annotation."""
    if arm == "full":
        return bench_real_classes(variants)
    if arm == "no_sv_terms":
        return _length_class(*bench_real_allele_lengths(variants))
    return np.full(np.asarray(variants.position).shape[0], _CLASS_CODES[VariantClass.SNV], dtype=np.uint8)


def bench_real_annotations(variants: Any, arm: str) -> tuple[dict[str, np.ndarray], dict[str, tuple[str, ...]]]:
    """The arm's prior columns; training allele frequencies stay out (Stage 0 computes them itself)."""
    sv_type, sv_type_legend = _categorical(variants.sv_type)
    source, source_legend = _categorical(variants.source)
    columns = {
        "log1p_tss_distance": np.log1p(np.abs(np.asarray(variants.distance_to_tss, dtype=np.float64))),
        "log1p_sv_length": np.log1p(np.abs(np.asarray(variants.sv_length, dtype=np.float64))),
        "allele_length_change": np.asarray(variants.allele_length_change, dtype=np.float64),
        "sv_type": sv_type,
        "source": source,
    }
    annotations = _informative_annotations({name: columns[name] for name in _ARM_ANNOTATIONS[arm]})
    legends = {name: legend for name, legend in (("sv_type", sv_type_legend), ("source", source_legend)) if name in annotations}
    return annotations, legends


@dataclass(frozen=True)
class BenchRealPredictor:
    scoring: ScoringModel
    columns: np.ndarray

    def predict(self, genotypes: np.ndarray) -> np.ndarray:
        """The genetic score plus the intercept, in closed form from dosages: x_j = (127 d_j - 127 - mu_j) / sigma_j."""
        dosages = np.asarray(genotypes, dtype=np.float64)[:, self.columns]
        signed = CODES_PER_DOSAGE * dosages - SIGNED_CODE_OFFSET
        standardized = (signed - self.scoring.signed_means) / self.scoring.signed_scales
        return standardized @ self.scoring.coefficients + self.scoring.alpha[0]


def one_core_budget() -> ComputeBudget:
    """One core's share of the machine: bench-real fits one gene per forked worker, one worker per core (its
    ``--workers`` defaults to the task's cores), so each fit gets one thread and its share of host memory, on the CPU
    (a device can't be split between the workers)."""
    machine = detect_compute_budget()
    return ComputeBudget(
        device_kind="cpu",
        device_ids=(),
        device_names=(),
        device_bytes=(),
        device_compute_capabilities=(),
        host_bytes=machine.host_bytes // machine.cpu_threads,
        cpu_threads=1,
    )


def fit_expression(train: Any) -> BenchRealPredictor:
    """bench-real: fit SV-PGS on one gene's cis window (harness.py)."""
    return _fit_expression(train, "full")


def fit_expression_no_sv_terms(train: Any) -> BenchRealPredictor:
    """bench-real ablation arm: ``fit_expression`` with the SV-specific prior terms withheld (module docstring)."""
    return _fit_expression(train, "no_sv_terms")


def fit_expression_no_annotations(train: Any) -> BenchRealPredictor:
    """bench-real ablation arm: ``fit_expression`` on the genotypes alone, every annotation withheld (module docstring)."""
    return _fit_expression(train, "no_annotations")


def _fit_expression(train: Any, arm: str) -> BenchRealPredictor:
    budget = one_core_budget()
    genotypes = np.asarray(train.genotypes)
    if not np.all(np.isin(genotypes, (0, 1, 2))):
        raise ValueError("bench-real training genotypes must be allele counts 0, 1 or 2.")
    variants = train.variants
    order = np.argsort(np.asarray(variants.position), kind="stable")
    reference_length, alternate_length = bench_real_allele_lengths(variants)
    annotations, legends = bench_real_annotations(variants, arm)
    classes = bench_real_classes_for_arm(variants, arm)
    codes = (np.asarray(genotypes, dtype=np.uint8).T * np.uint8(CODES_PER_DOSAGE))[order]
    with tempfile.TemporaryDirectory(prefix="svpgs-bench-real.") as scratch:
        store = write_store(
            Path(scratch) / "store",
            chromosome=f"chr{str(train.chrom).removeprefix('chr')}",
            position=np.asarray(variants.position)[order],
            # bench-real carries no genetic map and no fitting step reads one: the store records it as unknown.
            genetic_position_cm=np.full(order.shape[0], np.nan),
            reference_length=reference_length[order],
            alternate_length=alternate_length[order],
            variant_class=classes[order],
            annotations={name: values[order] for name, values in annotations.items()},
            annotation_legends=legends,
            code_rows=lambda rows: np.ascontiguousarray(codes[rows]),
            sample_count=genotypes.shape[0],
        )
        (Path(scratch) / "work").mkdir()
        scoring = _fit_one(store, np.empty((genotypes.shape[0], 0)), (), train.phenotype, TraitType.QUANTITATIVE, Path(scratch) / "work", budget)
    return BenchRealPredictor(scoring=scoring, columns=order[scoring.store_rows])
