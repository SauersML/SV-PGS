"""The in-workspace pipeline's synthetic dry run: every step on a synthetic store and synthetic OMOP.

The inputs mimic the imputation's deliverables:
popped GT:DS:GP batches in two imputation halves, long-read hard calls on the same sites, the
strata sidecar v2 with its chrK.json manifest of md5s and its _done marker, bubble.split path records, a repeat BED, the
CDR crosswalk, ancestry and relatedness tables, and OMOP rows through a fake BigQuery client. The
measurement model, fit, model files and prediction are bound to doubles with the production
signatures; they are other lanes' modules.
"""
from __future__ import annotations

import csv
import dataclasses
import datetime
import gzip
import hashlib
import inspect
import json
import pickle
import re
import shutil
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from sv_pgs.all_of_us import DISEASE_DEFINITIONS, MEASUREMENT_EXCLUSION_REASONS, build_all_of_us_measurement_sql
from sv_pgs.cohort import pipeline_half_levels
from sv_pgs.compute_budget import ComputeBudget
from sv_pgs.config import TraitType, VariantClass
from sv_pgs.dosage_store import VARIANT_CLASS_LEGEND, VARIANT_CLASSES, DosageStore, HalfSamples, VariantTable, encode_dosage_milli
from sv_pgs.fast_scoring import (
    ScoringModel,
    ScoringPlan,
    posterior_predictive_probability,
    score_genetic,
    score_linear_predictor,
)
from sv_pgs.fit_model import FitRequest
from tests.oracles import fit_at_exponent
from sv_pgs.ld_partition import cut_allowed_from_groups, validate_cut_allowed
from sv_pgs.store_converter import linear_recalibration, refalt_digest, tr_loci, value_matched_background
from sv_pgs import workspace_pipeline
from sv_pgs.workspace_pipeline import (
    ARMS,
    COMPLETE_FILE,
    STEP_NAMES,
    StrataSites,
    WorkspaceBindings,
    WorkspaceConfig,
    check_exportable,
    fit_filtered,
    path_counts,
    run_pipeline,
    store_group_first,
)

CHROMOSOMES = ("chr21", "chr22")
# (POS, REF, ALT, n_paths, n_paths_total, SVLEN) per popped record, the same layout on each chromosome.
RECORDS = (
    (1000, "A", "G", 1, 1, None),
    (2000, "ACGT", "A", 2, 3, None),
    (2000, "A", "AT", 2, 3, None),
    (3000, "GA", "G", 3, 12, None),
    (3000, "G", "GTT", 2, 12, None),
    (3000, "GAA", "G", 8, 12, None),
    (4000, "CA", "C", 1, 1, None),
    (5000, "C", "T", 1, 1, None),
    (6000, "T", "T" + "ACGT" * 13, 1, 1, 52),
)
# bubble.split: each bubble's paths as the records (indices into RECORDS) they carry; one path is
# written with ',' and one with ':' between its IDs.
BUBBLES = ((2000, ((1,), (1, 2), (2,))), (3000, ((3,), (4,)) + ((5,),) * 8 + ((3, 4), (3,))))
REPEATS = ((3995, 4010),)
# One GENCODE gene (1-based closed) over the 2000 and 3000 sites, with one exon over the 2000 site.
GENE = (1500, 3500)
EXON = (1990, 2010)
HALVES = (("A", (("001", 20), ("002", 20))), ("B", (("001", 24),)))
TRUTH = ("7001", "7002", "7003", "7004", "7005", "7006", "7099")
# Imputed samples the crosswalk maps to truth persons (three per ancestry group, by parity); everyone
# else is research ID 5000 + index. 7099 has only a truth row.
TRUTH_OF_SEQUENCING = {"SA001": "7001", "SA002": "7003", "SA003": "7005", "SB001": "7002", "SB002": "7004", "SB003": "7006"}
KINSHIP = (("7003", "5010", 0.45), ("5020", "5021", 0.25), ("5030", "5031", 0.05))
# The imputation reference panel: 7099 has no imputed row and 5030 is no truth person's relative, so every pair calibrates.
IMPUTATION_REFERENCE = ("7099", "5030")
FEMALE, MALE = 45878463, 45880669
# The GATK-SV call set per chromosome: (POS, ID, ALT, FILTER, INFO, kind). A deletion at an imputed record's POS, an
# insertion that is imputed record 8's own event (it pairs with it), a breakend and a HIGH_NCR deletion (both dropped), a
# multi-allelic copy number, and a duplication shorter than 50 bp (a direct row, so structural whatever its length).
GATKSV_RECORDS = (
    (1000, "gsv_del_1000", "<DEL>", "PASS", "SVTYPE=DEL;SVLEN=300;END=1300", "deletion"),
    (6000, "gsv_ins_6000", "<INS>", "PASS", "SVTYPE=INS;SVLEN=52;END=6001", "paired"),
    (6500, "gsv_bnd_6500", "N[chr21:90000[", "UNRESOLVED", "SVTYPE=BND", "deletion"),
    (7000, "gsv_cnv_7000", "<CN=0>,<CN=2>,<CN=3>,<CN=4>", "MULTIALLELIC", "SVTYPE=CNV;SVLEN=5000;END=12000", "copy_number"),
    (8000, "gsv_del_8000", "<DEL>", "HIGH_NCR", "SVTYPE=DEL;SVLEN=700;END=8700", "deletion"),
    (9000, "gsv_dup_9000", "<DUP>", "PASS", "SVTYPE=DUP;SVLEN=40;END=9040", "deletion"),
)
# People the call set does not carry: their store columns are no-calls, filled like any other.
GATKSV_ABSENT = ("5001", "5045")
GATKSV_QUALITY = 30
DIRECT_SV_NULL = {"gatksv_calls": None, "read_depth_cnv_calls": None, "ctyper_copy_numbers": None, "cn_estimator_copy_numbers": None, "truth_copy_numbers": None}
BUDGET = ComputeBudget(
    device_kind="cpu", device_ids=(), device_names=(), device_bytes=(), device_compute_capabilities=(), host_bytes=1 << 29, cpu_threads=1
)


def _identifier(chromosome: str, record: int) -> str:
    position, _ref, alt, *_rest = RECORDS[record]
    return f"{chromosome}-{position}-allele{record}-{len(alt)}"


def _sequencing_names() -> list[list[list[str]]]:
    return [
        [[f"S{label}{start + column + 1:03d}" for column in range(width)] for start, (_batch, width) in zip(np.cumsum([0] + [w for _b, w in batches]), batches)]
        for label, batches in HALVES
    ]


def _research_of(name: str) -> str:
    """Imputed samples of half A are 5000..5039 and of half B 5040..5063, except the truth persons'."""
    if name in TRUTH_OF_SEQUENCING:
        return TRUTH_OF_SEQUENCING[name]
    offset = 0 if name[1] == "A" else sum(width for _batch, width in HALVES[0][1])
    return str(5000 + offset + int(name[2:]) - 1)


def _vcf_header(samples: list[str], formats: tuple[str, ...]) -> list[str]:
    lines = [
        "##fileformat=VCFv4.2",
        '##INFO=<ID=ID,Number=A,Type=String,Description="atomic ids">',
        '##INFO=<ID=CM,Number=1,Type=Float,Description="genetic position">',
        '##INFO=<ID=INFO,Number=1,Type=Float,Description="imputation r2">',
        '##INFO=<ID=SVLEN,Number=A,Type=Integer,Description="SV length">',
        *(f'##FORMAT=<ID={name},Number={"G" if name == "GP" else 1},Type={"String" if name == "GT" else "Float"},Description="{name}">' for name in formats),
        *(f"##contig=<ID={chromosome},length=50000000>" for chromosome in CHROMOSOMES),
    ]
    return lines + ["\t".join(["#CHROM", "POS", "ID", "REF", "ALT", "QUAL", "FILTER", "INFO", *(["FORMAT"] if formats else []), *samples])]


def _reported_info(record: int) -> float:
    """The synthetic INFO/INFO of a record, the same in every batch."""
    return 0.5 + 0.05 * record


def _info(chromosome: str, record: int) -> str:
    fields = [f"ID={_identifier(chromosome, record)}", f"CM={0.001 * RECORDS[record][0]:.4f}", f"INFO={_reported_info(record):.3f}"]
    if RECORDS[record][5] is not None:
        fields.append(f"SVLEN={RECORDS[record][5]}")
    return ";".join(fields)


def _persons() -> list[str]:
    return sorted({_research_of(name) for half in _sequencing_names() for batch in half for name in batch} | set(TRUTH))


@dataclasses.dataclass
class _Synthetic:
    root: Path
    raw: dict
    genotypes: dict[tuple[str, str], np.ndarray]
    dosages: dict[tuple[str, str], np.ndarray]
    gatksv_calls: dict[tuple[str, str], tuple[int | None, ...]]
    gatksv_pattern: str


def _write_gatksv(root: Path, persons: list[str], genotypes: dict[tuple[str, str], np.ndarray], rng: np.random.Generator) -> dict:
    """The synthetic GATK-SV VCFs (GATKSV_RECORDS), samples named by research ID, and each (chromosome, person)'s calls
    per record: an ALT count or copy number, None at a no-call. FORMAT is AoU's GT:CN:CNQ, GT '.' on a copy number."""
    (root / "gatksv").mkdir()
    called = [person for person in persons if person not in GATKSV_ABSENT]
    calls: dict[tuple[str, str], tuple[int | None, ...]] = {}
    for chromosome in CHROMOSOMES:
        per_record = []
        for _position, _identifier, _alt, _filter, _info, kind in GATKSV_RECORDS:
            if kind == "paired":
                values = [int(genotypes[(chromosome, person)][8]) for person in called]
            elif kind == "copy_number":
                values = rng.choice([1, 2, 3, 4], size=len(called), p=[0.1, 0.6, 0.2, 0.1]).tolist()
            else:
                values = rng.binomial(2, 0.25, len(called)).tolist()
            missing = rng.random(len(called)) < 0.1
            per_record.append([None if gap else int(value) for value, gap in zip(values, missing)])
        for column, person in enumerate(called):
            calls[(chromosome, person)] = tuple(record[column] for record in per_record)
        lines = [
            "##fileformat=VCFv4.2",
            *(f"##contig=<ID={name},length=50000000>" for name in CHROMOSOMES),
            '##FILTER=<ID=UNRESOLVED,Description="unresolved">',
            '##FILTER=<ID=HIGH_NCR,Description="high no-call rate">',
            '##FILTER=<ID=MULTIALLELIC,Description="multi-allelic CNV">',
            '##INFO=<ID=SVTYPE,Number=1,Type=String,Description="type">',
            '##INFO=<ID=SVLEN,Number=1,Type=Integer,Description="length">',
            '##INFO=<ID=END,Number=1,Type=Integer,Description="end">',
            '##FORMAT=<ID=GT,Number=1,Type=String,Description="genotype">',
            '##FORMAT=<ID=CN,Number=1,Type=Integer,Description="copy number">',
            '##FORMAT=<ID=CNQ,Number=1,Type=Integer,Description="copy number quality">',
            "\t".join(["#CHROM", "POS", "ID", "REF", "ALT", "QUAL", "FILTER", "INFO", "FORMAT", *called]),
        ]
        for record, (position, identifier, alt, filter_value, info, kind) in enumerate(GATKSV_RECORDS):
            fields = []
            for person in called:
                value = calls[(chromosome, person)][record]
                if kind == "copy_number":
                    fields.append(".:.:." if value is None else f".:{value}:{GATKSV_QUALITY}")
                else:
                    fields.append("./.:.:." if value is None else f"{('0/0', '0/1', '1/1')[value]}:.:.")
            row_alt = alt.replace("chr21", chromosome)
            lines.append("\t".join([chromosome, str(position), f"{identifier}_{chromosome}", "N", row_alt, ".", filter_value, info, "GT:CN:CNQ", *fields]))
        (root / "gatksv" / f"{chromosome}.vcf").write_text("\n".join(lines) + "\n")
    return calls


def _write_inputs(root: Path) -> _Synthetic:
    """Every input file, and the genotypes and imputed dosages [records] per (chromosome, person) they were written from."""
    rng = np.random.default_rng(20260919)
    root.mkdir(parents=True)
    names = _sequencing_names()
    persons = _persons()
    frequencies = rng.uniform(0.15, 0.5, len(RECORDS))
    genotypes, dosages = {}, {}
    for chromosome in CHROMOSOMES:
        called = rng.binomial(2, frequencies[:, None], (len(RECORDS), len(persons)))
        called[:, persons.index("5010")] = called[:, persons.index("7003")]  # one genome under two research IDs
        noisy = np.round(np.clip(called + rng.normal(0.0, 0.2, called.shape), 0.0, 2.0), 3)
        single_path = np.array([record[4] == 1 for record in RECORDS])[:, None]
        # The single-path background q = w / (1 + w) rounds to 0.001: half of the zero dosages carry it.
        noisy[(noisy == 0.0) & single_path & (rng.random(noisy.shape) < 0.5)] = 0.001
        for column, person in enumerate(persons):
            genotypes[(chromosome, person)] = called[:, column]
            dosages[(chromosome, person)] = noisy[:, column]

    for half_index, (label, batches) in enumerate(HALVES):
        (root / label).mkdir()
        for (batch, _width), samples in zip(batches, names[half_index]):
            for chromosome in CHROMOSOMES:
                lines = _vcf_header(samples, ("GT", "DS", "GP"))
                for record, (position, ref, alt, *_rest) in enumerate(RECORDS):
                    fields = []
                    for sample in samples:
                        dosage = float(dosages[(chromosome, _research_of(sample))][record])
                        second = max(dosage - 1.0, 0.0)
                        first = dosage - 2.0 * second
                        fields.append(f"0|0:{dosage:.3f}:{1.0 - first - second:.3f},{first:.3f},{second:.3f}")
                    lines.append("\t".join([chromosome, str(position), ".", ref, alt, ".", "PASS", _info(chromosome, record), "GT:DS:GP", *fields]))
                (root / label / f"{chromosome}.{batch}.vcf").write_text("\n".join(lines) + "\n")
    (root / "truth").mkdir()
    for chromosome in CHROMOSOMES:
        lines = _vcf_header(list(TRUTH), ("GT",))
        for record, (position, ref, alt, *_rest) in enumerate(RECORDS):
            calls = [("0/0", "0/1", "1/1")[int(genotypes[(chromosome, person)][record])] for person in TRUTH]
            if chromosome == "chr21" and record == 0:
                calls[1] = "./."
            lines.append("\t".join([chromosome, str(position), ".", ref, alt, ".", "PASS", _info(chromosome, record), "GT", *calls]))
        (root / "truth" / f"{chromosome}.vcf").write_text("\n".join(lines) + "\n")

    strata = root / "strata"
    (strata / "_done").mkdir(parents=True)
    (root / "bubbles").mkdir()
    for chromosome in CHROMOSOMES:
        rows = ["#idx\tpos\tid\trefalt_md5\tref_len\talt_len\tclass\tn_paths\tn_paths_total\tcx"]
        sites, ids = hashlib.md5(), hashlib.md5()
        for record, (position, ref, alt, carried, total, _length) in enumerate(RECORDS):
            identifier = _identifier(chromosome, record)
            rows.append(f"{record}\t{position}\t{identifier}\t{refalt_digest(ref, alt):016x}\t{len(ref)}\t{len(alt)}\tsnv\t{carried}\t{total}\t{int(total > 1)}")
            sites.update(f"{chromosome}\t{position}\t{ref}\t{alt}\n".encode())
            ids.update(f"{identifier}\n".encode())
        with gzip.open(strata / f"{chromosome}.strata.tsv.gz", "wt") as handle:
            handle.write("\n".join(rows) + "\n")
        (strata / f"{chromosome}.json").write_text(json.dumps({"sites_md5": sites.hexdigest(), "ids_md5": ids.hexdigest()}))
        (strata / "_done" / f"{chromosome}.json").write_text(json.dumps({"chrom": chromosome}))
        lines = _vcf_header([], ())
        for bubble, (position, paths) in enumerate(BUBBLES):
            for path, carried in enumerate(paths):
                separator = "," if (bubble, path) == (1, 10) else ":"
                listed = separator.join(_identifier(chromosome, record) for record in carried)
                lines.append("\t".join([chromosome, str(position), ".", "A", "C", ".", "PASS", f"ID={listed}"]))
        (root / "bubbles" / f"{chromosome}.bubble.split.vcf").write_text("\n".join(lines) + "\n")
    with gzip.open(root / "repeats.bed.gz", "wt") as handle:
        handle.write("".join(f"{chromosome}\t{start}\t{end}\tTR\n" for chromosome in CHROMOSOMES for start, end in REPEATS))
    with gzip.open(root / "genes.gtf.gz", "wt") as handle:
        handle.write("##description: synthetic\n" + "".join(
            f"{chromosome}\tSYN\t{feature}\t{start}\t{end}\t.\t+\t.\tgene_id \"G1\";\n"
            for chromosome in CHROMOSOMES for feature, (start, end) in (("gene", GENE), ("transcript", GENE), ("exon", EXON))
        ))

    all_names = [name for half in names for batch in half for name in batch]
    (root / "crosswalk.tsv").write_text("research_id\tsequencing_id\n" + "".join(f"{_research_of(name)}\t{name}\n" for name in all_names))
    (root / "ancestry.tsv").write_text(
        "research_id\tancestry_pred\tpca_features\n"
        + "".join(f"{person}\t{('afr', 'eur')[int(person) % 2]}\t{json.dumps(rng.normal(size=3).round(4).tolist())}\n" for person in persons)
    )
    (root / "relatedness.tsv").write_text("i.s\tj.s\tkin\n" + "".join(f"{first}\t{second}\t{kinship}\n" for first, second, kinship in KINSHIP))
    (root / "reference_ids.tsv").write_text("research_id\n" + "".join(f"{person}\n" for person in IMPUTATION_REFERENCE))
    raw = {
        "run_directory": str(root.parent / "run"),
        "marker_directory": str(root.parent / "markers"),
        "chromosomes": list(CHROMOSOMES),
        "imputed_halves": [
            {"label": label, "batches": [batch for batch, _width in batches], "batch_path": str(root / label / "{chromosome}.{batch}.vcf")}
            for label, batches in HALVES
        ],
        "truth_calls": str(root / "truth" / "{chromosome}.vcf"),
        "imputation_reference_ids": str(root / "reference_ids.tsv"),
        "strata_directory": str(strata),
        "bubble_split": str(root / "bubbles" / "{chromosome}.bubble.split.vcf"),
        "genetic_map": None,
        "tandem_repeats": str(root / "repeats.bed.gz"),
        "gene_annotation": str(root / "genes.gtf.gz"),
        "crosswalk": str(root / "crosswalk.tsv"),
        "crosswalk_columns": ["research_id", "sequencing_id"],
        "ancestry": str(root / "ancestry.tsv"),
        "relatedness": str(root / "relatedness.tsv"),
        "relatedness_columns": ["i.s", "j.s", "kin"],
        "diseases": ["atrial_fibrillation"],
        "traits": ["total_bilirubin"],
        "fold_count": 2,
        "seed": 7,
        "codec": "zstd",
        "export": ["accuracy.tsv", "comparison.tsv", "effects_by_class.tsv", "genetic_variance.tsv"],
        "direct_sv": None,
    }
    gatksv_calls = _write_gatksv(root, persons, genotypes, rng)
    return _Synthetic(
        root=root, raw=raw, genotypes=genotypes, dosages=dosages, gatksv_calls=gatksv_calls, gatksv_pattern=str(root / "gatksv" / "{chromosome}.vcf"),
    )


def _omop_rows(persons: list[str]) -> tuple[list[dict], list[dict]]:
    """Disease-query and measurement-query rows: every person has EHR and 9001 has no genotypes. 5049 has one
    afib date (neither case nor control, so no disease row) and, like every person numbered 1 mod 8, no
    bilirubin occasion, so no table lists it."""
    rng = np.random.default_rng(3)
    disease, measurement = [], []
    for person in persons + ["9001"]:
        number = int(person)
        sex = FEMALE if number % 2 == 0 else MALE
        occurrences = 1 if person == "5049" else (3 if number % 3 == 0 else 0)
        disease.append({
            "sample_id": person, "person_id": person, "phenotype_occurrence_count": occurrences,
            "first_condition_date": "2020-01-01" if occurrences else None, "observation_start_date": "2015-01-01",
            "observation_end_date": "2024-01-01", "primary_consent_date": "2018-06-01", "year_of_birth": 1975,
            "age_at_first_condition": 40.0 + number % 30 if occurrences else None, "age_at_first_case_procedure": None,
            "age_at_observation_end": 45.0 + number % 30, "pre_landmark_condition_dates": 6 + number % 7, "condition_dates": 9 + number % 7,
            "has_control_exclusion_code": False, "has_ambiguous_code": False, "case_medication_dates": 0,
            "has_control_exclusion_medication": False, "case_procedure_dates": 0, "sex_at_birth_concept_id": sex,
            "sex_at_birth_name": "female" if sex == FEMALE else "male",
        })
        if number % 8 == 1:
            continue
        for day in range(int(rng.integers(2, 5))):
            measurement.append({
                "sample_id": person, "person_id": person, "measurement_date": datetime.date(2016, 1, 1) + datetime.timedelta(days=day),
                "age_at_occasion": float(rng.uniform(30.0, 70.0)), "treated": False, "row_count": 1,
                **{f"{reason}_row_count": 0 for reason in MEASUREMENT_EXCLUSION_REASONS}, "retained_row_count": 1,
                "occasion_value": round(float(1.0 + 0.2 * rng.normal()), 1), "unrecognized_unit_labels": [],
                "sex_at_birth_concept_id": sex, "sex_at_birth_name": "female" if sex == FEMALE else "male",
            })
    return disease, measurement


class _FakeBigQuery:
    project = "billing-project"

    def __init__(self, disease: list[dict], measurement: list[dict]) -> None:
        self.disease, self.measurement = disease, measurement

    def query(self, sql: str, job_config):
        rows = self.measurement if sql == build_all_of_us_measurement_sql() else self.disease
        return SimpleNamespace(result=lambda: [SimpleNamespace(items=row.items) for row in rows])


@dataclasses.dataclass(frozen=True)
class _Moments:
    pair_counts: np.ndarray
    dosage_mean: np.ndarray
    truth_mean: np.ndarray
    dosage_variance: np.ndarray
    truth_variance: np.ndarray
    covariance: np.ndarray


def _calibration_moments(dosage: np.ndarray, truth: np.ndarray) -> _Moments:
    complete = np.isfinite(dosage) & np.isfinite(truth)
    counts = complete.sum(axis=1)
    d, t = np.where(complete, dosage, 0.0), np.where(complete, truth, 0.0)
    with np.errstate(invalid="ignore", divide="ignore"):
        dm, tm = d.sum(axis=1) / counts, t.sum(axis=1) / counts
        return _Moments(
            counts, dm, tm, (np.where(complete, d - dm[:, None], 0.0) ** 2).sum(axis=1) / counts,
            (np.where(complete, t - tm[:, None], 0.0) ** 2).sum(axis=1) / counts,
            (np.where(complete, (d - dm[:, None]) * (t - tm[:, None]), 0.0)).sum(axis=1) / counts,
        )


def _fit_measurement_model(calibration, cohort_dosage_variance, strata, design=None, reported_reliability=None):
    """A stand-in with the production signature: kappa = Cov(T, D) / Var(D) where there are three pairs, else 1."""
    assert strata.shape == cohort_dosage_variance.shape and set(strata) <= set(VARIANT_CLASS_LEGEND)
    assert reported_reliability is not None and np.all((reported_reliability >= 0.0) & (reported_reliability <= 1.0))
    scales = np.ones_like(cohort_dosage_variance)
    if calibration is not None:
        moments = calibration.moments
        usable = (moments.pair_counts >= 3) & (moments.dosage_variance > 0.0) & (moments.covariance > 0.0)
        scales[usable] = moments.covariance[usable] / moments.dosage_variance[usable]
    certificate = {"recalibration": "not applied: no truth genotypes were supplied" if calibration is None else "applied"}
    return SimpleNamespace(scales=scales, residual_variance=np.zeros_like(scales), log_reliability=np.zeros_like(scales), certificate=certificate)


@dataclasses.dataclass(frozen=True)
class _MeasurementModel:
    """A stand-in for measurement_model.MeasurementModel's arrays, its maps and certificate, and its npz save and load."""

    scales: np.ndarray
    residual_variance: np.ndarray
    log_reliability: np.ndarray
    leakage_maps: tuple = ()
    certificate: dict = dataclasses.field(default_factory=dict)

    def save(self, path: Path) -> None:
        assert not self.leakage_maps
        np.savez(path, scales=self.scales, residual_variance=self.residual_variance, log_reliability=self.log_reliability,
                 certificate=json.dumps(self.certificate))

    @classmethod
    def load(cls, path: Path) -> _MeasurementModel:
        with np.load(path) as archive:
            return cls(archive["scales"], archive["residual_variance"], archive["log_reliability"], (), json.loads(str(archive["certificate"])))


def _pooled_measurement_model(models, cohort_dosage_mean, cohort_dosage_variance, group_counts) -> _MeasurementModel:
    """A stand-in with the production signature: log Var(D*) / (Var(D*) + v) of the stacked groups, -inf without variance."""
    scales = np.column_stack([model.scales for model in models])
    residual = np.column_stack([model.residual_variance for model in models])
    weights = group_counts / group_counts.sum()
    mean = cohort_dosage_mean @ weights
    variance = (scales**2 * cohort_dosage_variance + (cohort_dosage_mean - mean[:, None]) ** 2) @ weights
    total = variance + residual @ weights
    with np.errstate(divide="ignore", invalid="ignore"):
        log_reliability = np.where(total > 0.0, np.log(variance / total), -np.inf)
    return _MeasurementModel(np.ones_like(mean), residual @ weights, log_reliability)


@dataclasses.dataclass
class _Fitted:
    model_names: tuple[str, ...]
    covariate_names: tuple[str, ...]
    scoring: tuple[ScoringModel, ...]
    noise_variance: np.ndarray
    certificate: dict[str, np.ndarray]
    refusals: tuple[str, ...] = ()
    covariate_columns: np.ndarray | None = None
    hyperparameters: tuple = ()

    def __post_init__(self):
        if self.covariate_columns is None:
            self.covariate_columns = np.ones((len(self.model_names), len(self.covariate_names)), dtype=bool)
        if not self.hyperparameters:
            self.hyperparameters = (None,) * len(self.model_names)


class _Recorder:
    def __init__(self, fits_before_failing: int | None = None) -> None:
        self.fit_calls: list[FitRequest] = []
        self.fits_before_failing = fits_before_failing

    def fit(self, request: FitRequest) -> _Fitted:
        """A stand-in for the engine behind the real ``fit_model.FitRequest``: each model's own covariates by least
        squares, then marginal effects of the standardized codes on the training rows, over the records the request's
        offsets give any signal (a -inf offset leaves a record out, as the real fit does)."""
        if self.fits_before_failing is not None and len(self.fit_calls) >= self.fits_before_failing:
            raise RuntimeError("preempted")
        assert isinstance(request, FitRequest)
        self.fit_calls.append(request)
        store, store_columns, covariates = request.store, request.store_columns, request.covariates
        covariate_columns, targets, training = request.covariate_columns, request.targets, request.training
        model_names, trait_types, work_dir = request.model_names, request.trait_types, request.work_dir
        offset = request.log_variance_offset
        assert offset.shape == (store.n_variants,) and np.all(offset <= 0.0) and work_dir.is_dir()
        codes = store.read_codes(0, store.n_variants, np.asarray(store_columns)).astype(np.float64) - 127.0
        models = []
        for model, trait_type in enumerate(trait_types):
            rows, own = training[:, model], covariate_columns[model]
            signed = codes[:, rows]
            means, scales = signed.mean(axis=1), signed.std(axis=1)
            active = np.flatnonzero((scales > 0.0) & np.isfinite(offset))
            target = targets[rows, model]
            design = np.column_stack([np.ones(rows.sum()), covariates[rows][:, own]])
            coefficients = np.linalg.lstsq(design, target, rcond=None)[0]
            standardized = (signed[active] - means[active, None]) / scales[active, None]
            effects = standardized @ (target - design @ coefficients) / rows.sum() / len(active)
            draws = effects[:, None] + np.random.default_rng(model).normal(0.0, 1e-3, (len(active), 4))
            alpha = np.zeros(1 + covariates.shape[1])
            if trait_type != TraitType.BINARY:
                alpha[0], alpha[1:][own] = coefficients[0], coefficients[1:]
            models.append(ScoringModel(active.astype(np.int64), means[active], scales[active], effects, draws, alpha, trait_type, 0.0,
                                      np.repeat(alpha[:, None], draws.shape[1], axis=1), np.zeros((alpha.size, alpha.size)), True))
        return _Fitted(tuple(model_names), tuple(request.covariate_names), tuple(models), np.ones(len(models)), {"outer_criterion_met": np.ones(len(models), dtype=bool), "unresolved": np.zeros(len(models)), "budget_unresolved": np.zeros(len(models)), "restored_best": np.zeros(len(models)), "far_field_scale": np.zeros(len(models))}, covariate_columns=np.asarray(request.covariate_columns))


def _save_model(path: Path, model: _Fitted) -> None:
    path.mkdir()
    (path / "model.pickle").write_bytes(pickle.dumps(model))


def _load_model(path: Path) -> _Fitted:
    return pickle.loads((path / "model.pickle").read_bytes())


class _StoreBlocks:
    def __init__(self, store: DosageStore, budget: ComputeBudget) -> None:
        self.store, self.budget = store, budget

    @property
    def sample_count(self) -> int:
        return int(self.store.n_samples)

    def iter_code_blocks(self, variant_ranges, buffers):
        yield from self.store.iter_codes(variant_ranges, None, self.budget)


def _predict(models: list[_Fitted], store: DosageStore, sample_indices, covariates, budget):
    scorings = tuple(scoring for model in models for scoring in model.scoring)
    genetic = score_genetic(_StoreBlocks(store, budget), ScoringPlan.from_models(scorings), budget, np.asarray(sample_indices))
    linear = score_linear_predictor(genetic.means, covariates, scorings)
    mean = linear.copy()
    for index, scoring in enumerate(scorings):
        if scoring.trait_type == TraitType.BINARY:
            mean[:, index] = posterior_predictive_probability(linear[:, index], genetic.variances[:, index], 0.0)
    return SimpleNamespace(genetic=genetic, linear_predictor=linear, predictive_mean=mean, predictive_variance=np.ones_like(mean))


def _bindings(recorder: _Recorder, client: _FakeBigQuery) -> WorkspaceBindings:
    return WorkspaceBindings(
        bigquery_client=lambda: client,
        calibration_moments=_calibration_moments,
        concatenate_calibration_moments=lambda parts: _Moments(
            *(np.concatenate([getattr(part, field.name) for part in parts]) for field in dataclasses.fields(_Moments))
        ),
        calibration_pairs=lambda sample_ids, moments, blocks=(): SimpleNamespace(sample_ids=sample_ids, moments=moments, blocks=blocks),
        fit_measurement_model=_fit_measurement_model,
        pooled_measurement_model=_pooled_measurement_model,
        load_measurement_model=_MeasurementModel.load,
        fit=recorder.fit,
        save_model=_save_model,
        load_model=_load_model,
        predict=_predict,
    )


@pytest.fixture
def workspace(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("GOOGLE_PROJECT", "billing-project")
    monkeypatch.setenv("WORKSPACE_CDR", "aou_workspace.cdr_dataset")
    # The occasion model at the identity transform in a small budget; its exponent search is tested in test_all_of_us_phenotypes.
    monkeypatch.setattr("sv_pgs.all_of_us.fit_occasion_model", lambda occasions, working_bytes: fit_at_exponent(occasions, 1.0, min(working_bytes, 1 << 28)))
    return _write_inputs(tmp_path / "inputs"), _FakeBigQuery(*_omop_rows(_persons()))


def _completed_at(run: Path, step: str) -> float:
    return (run / step / "_COMPLETE.json").stat().st_mtime_ns


def test_the_production_bindings_hand_the_fit_step_s_request_to_the_real_fitter() -> None:
    """No stand-in: the modules the run imports are the real ones, and ``fit`` takes the ``FitRequest`` the fit step
    builds, so the step's call cannot drift from the fitter's input."""
    from sv_pgs import artifact, fit_model, measurement_model
    from sv_pgs.workspace_pipeline import workspace_bindings

    bindings = workspace_bindings()
    assert bindings.fit is fit_model.fit and bindings.save_model is artifact.save_model
    assert bindings.load_model is artifact.load_model and bindings.predict is artifact.predict_models
    assert bindings.load_measurement_model == measurement_model.MeasurementModel.load
    (parameter,) = inspect.signature(fit_model.fit).parameters.values()
    assert parameter.kind is inspect.Parameter.POSITIONAL_OR_KEYWORD and parameter.default is inspect.Parameter.empty
    # The request the fit step builds is the fitter's own class, not a copy of its fields.
    assert workspace_pipeline.FitRequest is fit_model.FitRequest is FitRequest


def test_a_dry_run_builds_every_step_from_synthetic_inputs(workspace) -> None:
    synthetic, client = workspace
    recorder = _Recorder()
    config = WorkspaceConfig.from_mapping(synthetic.raw)
    summaries = run_pipeline(config, _bindings(recorder, client), BUDGET)
    run = config.run_directory
    assert tuple(summaries) == STEP_NAMES and all((run / step / "_COMPLETE.json").is_file() for step in STEP_NAMES)
    assert {path.name.split("_", 1)[1] for path in config.marker_directory.iterdir()} >= {f"{step}_completed" for step in STEP_NAMES}

    # Cohort rows: each truth row wins over its imputed row; 5010, which duplicates 7003's genome, is dropped.
    rows = {row.split("\t")[0]: row.split("\t") for row in (run / "samples" / "rows.tsv").read_text().splitlines()[1:]}
    assert {person for person, row in rows.items() if row[1] == "long_read"} == set(TRUTH)
    assert "5010" not in rows and rows["5020"][5] == rows["5021"][5]
    # Second-degree relatives form one family; a pair below the KING second-degree bound does not.
    assert rows["5020"][6] == rows["5021"][6] and rows["5030"][6] != rows["5031"][6]
    assert summaries["samples"]["calibration_pairs"] == len(TRUTH_OF_SEQUENCING)
    assert summaries["samples"]["calibration"]["reference_members_left_out"] == summaries["samples"]["calibration"]["relatives_left_out"] == "0"

    # The store: typed half manifests, background-corrected imputed codes, long-read hard calls.
    with DosageStore.open(run / "store" / "store") as store:
        halves = store.half_samples()
        assert [half.namespace for half in halves] == ["dragen_sample", "dragen_sample", "research_id"]
        assert halves[2] == HalfSamples("research_id", TRUTH)
        names = [name for batch in _sequencing_names()[0] for name in batch]
        stored = store.read_codes(0, len(RECORDS), np.arange(len(names)))
        milli = np.rint(np.column_stack([synthetic.dosages[("chr21", _research_of(name))] for name in names]) * 1000).astype(np.uint16)
        kept = np.array([min(10, record[4]) for record in RECORDS])
        carrying = np.array([1, 2, 2, 1, 1, 8, 1, 1, 1])
        np.testing.assert_array_equal(stored, encode_dosage_milli(value_matched_background(milli, kept, carrying).dosage_milli))
        truth_codes = store.read_codes(0, len(RECORDS), np.arange(store.n_samples - len(TRUTH), store.n_samples))
        called = np.column_stack([synthetic.genotypes[("chr21", person)] for person in TRUTH]) * 127
        # 7002's no-call at the first record is filled with its group's measured mean; every call is its ALT count.
        assert truth_codes[0, 1] != 255
        np.testing.assert_array_equal(np.delete(truth_codes[0], 1), np.delete(called[0], 1))
        np.testing.assert_array_equal(truth_codes[1:], called[1:])
        table = store.variant_table
        classes = [VARIANT_CLASS_LEGEND[code] for code in table.variant_class[: len(RECORDS)]]
        assert classes == ["snv", "deletion", "insertion", "deletion", "insertion", "deletion", "str_vntr_repeat", "snv", "insertion"]
        assert table.group_first[: len(RECORDS)].tolist() == [0, 1, 1, 3, 3, 3, 6, 7, 8]
        assert table.annotations["n_paths_total"][: len(RECORDS)].tolist() == [record[4] for record in RECORDS]
        assert table.annotations["in_gene"][: len(RECORDS)].tolist() == [0, 1, 1, 1, 1, 1, 0, 0, 0]
        assert table.annotations["in_exon"][: len(RECORDS)].tolist() == [0, 1, 1, 0, 0, 0, 0, 0, 0]
    assert summaries["store"]["background_zeroed"] > 0 and summaries["store"]["no_calls"] == 1
    # The reported r^2 the measurement model falls back on is the batches' INFO/INFO, not a dosage-variance ratio.
    np.testing.assert_allclose(np.load(run / "store" / "reported" / "chr21.npy"), [_reported_info(record) for record in range(len(RECORDS))])

    # D*: the imputed halves were rewritten with the measurement's scales; the MANIFEST says so.
    measurement = np.load(run / "measurement" / "measurement.npz")
    assert summaries["measurement"]["recalibrated"] and np.any(measurement["scales"] != 1.0)
    with DosageStore.open(run / "store" / "store", half_indices=[0]) as before, DosageStore.open(run / "measurement" / "store", half_indices=[0]) as after:
        groups = np.load(run / "samples" / "groups_half0.npy")
        stored_milli = ((before.read_codes(0, before.n_variants).astype(np.int64) * 2000 + 127) // 254).astype(np.uint16)
        expected = linear_recalibration(stored_milli, groups, measurement["scales"][:, : groups.max() + 1]).dosage_milli
        np.testing.assert_array_equal(after.read_codes(0, after.n_variants), encode_dosage_milli(expected))
        assert after.manifest_attributes["recalibrated"] is True
    # direct_sv is null here: the step adds no row, and the fit reads the measurement step's store and model.
    assert summaries["direct_sv"]["direct_rows"] == 0 and summaries["direct_sv"]["store"] == "measurement/store"
    assert summaries["direct_sv"]["measurement_model"] == "measurement/measurement_model.npz" and "direct_sv is null" in summaries["direct_sv"]["gatksv_calls"]

    # The fit saw, per arm and trait, the trait's fold training sets and its all-rows set, with no held-out row, no
    # intercept column, and the trait's own covariates only.
    cohort = np.load(run / "cohort" / "cohort.npz")
    design = json.loads((run / "cohort" / "cohort.json").read_text())
    traits = ["atrial_fibrillation", "total_bilirubin"]
    assert design["model_names"] == [f"{trait}/{fold}" for trait in traits for fold in ("fold0", "fold1", "all")]
    # One fit per arm holds every trait's models, so the models that share training rows share the store's passes.
    assert [call.model_names for call in recorder.fit_calls] == [tuple(design["model_names"])] * len(ARMS)
    names = design["covariate_names"]
    structure = {"intercept", "pipeline_half=B", "genotype_source=long_read", "PC1", "PC2", "PC3"}
    own = {
        "atrial_fibrillation": {
            "atrial_fibrillation:age_at_observation_end", "atrial_fibrillation:age_at_observation_end_squared",
            "atrial_fibrillation:age_at_observation_end_x_female", "atrial_fibrillation:log1p_pre_landmark_condition_dates",
            "atrial_fibrillation:sex_at_birth_concept_id=45880669",
        },
        "total_bilirubin": {
            "total_bilirubin:age_at_measurement", "total_bilirubin:age_at_measurement_squared",
            "total_bilirubin:age_at_measurement_x_female", "total_bilirubin:sex_at_birth_concept_id=45880669",
        },
    }
    assert set(names) == structure | own["atrial_fibrillation"] | own["total_bilirubin"]
    assert summaries["cohort"]["covariates"] == {trait: [name for name in names if name in structure | columns] for trait, columns in own.items()}
    saved = _MeasurementModel.load(run / "measurement" / "measurement_model.npz")
    with DosageStore.open(run / "measurement" / "store") as store:
        structural = np.asarray(store.variant_table.annotations["sv_length"]) >= 50
    # The SNV/indel arm leaves out exactly the structural records (the 52 bp insertion), through a -inf offset.
    assert structural.sum() == len(CHROMOSOMES)
    fit_rows = cohort["training"].any(axis=1)
    for arm, call in zip(ARMS, recorder.fit_calls):
        assert call.covariates.shape[1] == len(call.covariate_names) == len(names) - 1
        assert len(call.research_ids) == len(call.store_columns) == fit_rows.sum()
        # Every model, a disease's included, is fitted on the Gaussian working likelihood (FIT_TRAIT_TYPE).
        assert call.trait_types == (TraitType.QUANTITATIVE,) * 6
        folds = cohort["folds"][fit_rows]
        for model, model_name in enumerate(call.model_names):
            trait, fold = model_name.split("/")
            observed = np.isfinite(cohort["targets"][fit_rows, traits.index(trait)])
            used = {name for name, used in zip(names[1:], call.covariate_columns[model]) if used}
            assert used == (structure | own[trait]) - {"intercept"}
            assert np.all(np.isnan(call.targets[~call.training[:, model], model]))
            # A fold model holds its fold out; the all-rows model trains on every row that observes its trait.
            held = folds == int(fold.removeprefix("fold")) if fold != "all" else np.zeros_like(observed)
            np.testing.assert_array_equal(call.training[:, model], observed & ~held)
        expected = np.where(structural, -np.inf, saved.log_reliability) if arm == "snv_indel" else saved.log_reliability
        np.testing.assert_array_equal(call.log_variance_offset, expected)
    # Both arms' fits have the run's seed and the same models in the same order: each group's probes and draws are
    # common random numbers across the arms.
    assert [call.seed for call in recorder.fit_calls] == [config.seed] * len(ARMS)
    # A trait's own columns are 0 on the rows it does not observe.
    bilirubin_age = cohort["covariates"][:, names.index("total_bilirubin:age_at_measurement")]
    assert np.all(bilirubin_age[~np.isfinite(cohort["targets"][:, 1])] == 0.0) and np.all(bilirubin_age[np.isfinite(cohort["targets"][:, 1])] > 0.0)
    families = cohort["families"]
    assert np.array_equal(np.unique(families), np.arange(families.max() + 1)) and len(set(families.tolist())) == len(families) - 1
    # 5049 is genotyped but in no table: a cohort row with no target and no fit row.
    untabled = list(cohort["research_ids"]).index("5049")
    assert np.all(np.isnan(cohort["targets"][untabled])) and not fit_rows[untabled] and "7099" in cohort["research_ids"]
    assert all(json.loads((run / "fit" / "models" / arm / f"{trait}.json").read_text())["outer_criterion_met"] == [True] * 3 for arm in ARMS for trait in traits)

    # Scores: in both arms every observed target has its own fold's held-out prediction, and the arms differ.
    predictions = np.load(run / "score" / "predictions.npz")
    assert tuple(predictions["arms"]) == ARMS and predictions["predictive_mean"].shape == (len(ARMS), len(cohort["research_ids"]), 6)
    for name in ("held_out_predictive_mean", "held_out_linear_predictor", "held_out_genetic"):
        assert all(np.array_equal(np.isfinite(predictions[name][arm]), np.isfinite(cohort["targets"])) for arm in range(len(ARMS)))
    assert not np.allclose(predictions["genetic_mean"][0], predictions["genetic_mean"][1])

    # Effects: each trait's all-rows model in both arms, record by record, and summaries by class and length.
    for arm in ARMS:
        for trait in traits:
            effects = pd.read_csv(run / "effects" / f"{trait}.{arm}.tsv.gz", sep="\t")
            assert (effects["length"] >= 50).any() == (arm == "with_sv")
            assert np.all((effects["probability_positive"] >= 0.0) & (effects["probability_positive"] <= 1.0))
    by_class = pd.read_csv(run / "effects" / "effects_by_class.tsv", sep="\t")
    assert set(by_class["length_bin"]) == {"<50", "50-499"} and set(by_class["arm"]) == set(ARMS)
    assert by_class[by_class["length_bin"] == "50-499"]["arm"].eq("with_sv").all()
    # The structural records' share of each with_sv score's variance over the trait's observed rows is the split of
    # beta' Sigma beta, Sigma the covariance of the standardized genotypes the all-rows model scores, LD terms included.
    variance = pd.read_csv(run / "effects" / "genetic_variance.tsv", sep="\t")
    assert list(variance.columns) == list(workspace_pipeline.GENETIC_VARIANCE_COLUMNS)
    with DosageStore.open(run / "measurement" / "store") as store:
        for trait_index, trait in enumerate(traits):
            observed = np.isfinite(cohort["targets"][:, trait_index])
            fitted = _load_model(run / "fit" / "models" / "with_sv" / trait)
            scoring = fitted.scoring[fitted.model_names.index(f"{trait}/all")]
            signed = store.read_codes(0, store.n_variants, cohort["store_columns"][observed]).astype(np.float64) - 127.0
            standardized = (signed[scoring.store_rows] - scoring.signed_means[:, None]) / scoring.signed_scales[:, None]
            covariance = np.cov(standardized, bias=True)
            in_part = structural[scoring.store_rows]
            assert in_part.any() and not in_part.all()
            whole = scoring.coefficients
            part, rest = np.where(in_part, whole, 0.0), np.where(in_part, 0.0, whole)
            expected = {
                "total_variance": whole @ covariance @ whole, "structural_variance": part @ covariance @ part,
                "small_variance": rest @ covariance @ rest, "covariance_term": 2.0 * part @ covariance @ rest,
                "structural_share": part @ covariance @ whole / (whole @ covariance @ whole),
            }
            rows = variance[variance["trait"] == trait].set_index("quantity")
            assert rows["n"].tolist() == [int(observed.sum())] * len(expected)
            np.testing.assert_allclose(rows.loc[list(expected), "value"], list(expected.values()), rtol=1e-5, atol=1e-5 * expected["total_variance"])

    # The report suppresses cells of 1-20 people; export stages the approved files and nothing is sent.
    report = [line.split("\t") for line in (run / "report" / "accuracy.tsv").read_text().splitlines()]
    assert report[0] == list(workspace_pipeline.ACCURACY_COLUMNS)
    long_read = [line for line in report if line[2:4] == ["half", "long_read"]]
    assert long_read and all(line[4] == "1-20 (suppressed)" and line[7] == "" for line in long_read)
    assert any(line[2] == "fold" and line[7] for line in report[1:])
    assert {line[1] for line in report[1:]} == set(ARMS)
    comparison = [line.split("\t") for line in (run / "report" / "comparison.tsv").read_text().splitlines()]
    assert comparison[0] == list(workspace_pipeline.COMPARISON_COLUMNS)
    assert any(line[0] == "total_bilirubin" and line[5] == "delta_pgs_partial_r2" and line[6] for line in comparison[1:])
    for name in synthetic.raw["export"]:
        source = run / ("report" if name in ("accuracy.tsv", "comparison.tsv") else "effects") / name
        assert (run / "export" / name).read_text() == source.read_text()


def test_a_rerun_skips_completed_steps_and_resumes_after_a_failure(workspace) -> None:
    synthetic, client = workspace
    config = WorkspaceConfig.from_mapping(synthetic.raw)
    with pytest.raises(RuntimeError, match="preempted"):
        run_pipeline(config, _bindings(_Recorder(fits_before_failing=1), client), BUDGET)
    run = config.run_directory
    assert (run / "fit.partial").is_dir() and any(path.name.endswith("_fit_failed-RuntimeError") for path in config.marker_directory.iterdir())
    # The first arm's fit finished before the preemption and every trait of it was saved; the rerun fits the other arm.
    assert all((run / "fit.partial" / "models" / ARMS[0] / f"{trait}.json").is_file() for trait in ("atrial_fibrillation", "total_bilirubin"))
    assert not (run / "fit.partial" / "models" / ARMS[1]).exists()
    finished = {step: _completed_at(run, step) for step in STEP_NAMES[:5]}
    recorder = _Recorder()
    run_pipeline(config, _bindings(recorder, client), BUDGET)
    assert {step: _completed_at(run, step) for step in STEP_NAMES[:5]} == finished
    assert len(recorder.fit_calls) == len(ARMS) - 1 and (run / "export" / "_COMPLETE.json").is_file()

    # A changed phenotype set re-runs the steps that read it and keeps the store.
    fewer = WorkspaceConfig.from_mapping({**synthetic.raw, "traits": []})
    run_pipeline(fewer, _bindings(_Recorder(), client), BUDGET)
    assert _completed_at(run, "store") == finished["store"] and _completed_at(run, "samples") == finished["samples"]
    assert json.loads((run / "cohort" / "cohort.json").read_text())["trait_names"] == ["atrial_fibrillation"]
    assert any(path.name.startswith("phenotypes.superseded-") for path in run.iterdir())


def test_an_uncertified_fit_is_scored_and_reported_and_named(workspace) -> None:
    """A model that misses certification must not withhold its trait: held-out accuracy measures it whatever its
    convergence, so the trait is scored and in every table, and uncertified.tsv names it with its arm and codes."""
    synthetic, client = workspace
    config = WorkspaceConfig.from_mapping(synthetic.raw)
    recorder = _Recorder()
    fit = recorder.fit

    def uncertified(request: FitRequest) -> _Fitted:
        fitted = fit(request)
        if len(recorder.fit_calls) == 1:
            fitted.certificate["outer_criterion_met"][0] = False
        return fitted

    recorder.fit = uncertified
    run_pipeline(config, _bindings(recorder, client), BUDGET)
    run = config.run_directory
    arms = json.loads((run / "fit" / "_COMPLETE.json").read_text())["summary"]["arms"]
    assert len(arms[ARMS[0]]["uncertified"]) == 1 and arms[ARMS[1]]["uncertified"] == []
    cohort = json.loads((run / "cohort" / "cohort.json").read_text())
    flagged = cohort["trait_names"][0]
    score = json.loads((run / "score" / "_COMPLETE.json").read_text())["summary"]
    assert score["quarantined"] == {} and list(score["uncertified"]) == [flagged]
    assert "outer_criterion_met" in score["uncertified"][flagged][0]
    predictions = np.load(run / "score" / "predictions.npz")
    assert np.any(np.isfinite(predictions["held_out_genetic"][:, :, 0])) and np.any(np.isfinite(predictions["held_out_genetic"][:, :, 1]))
    for table in ("accuracy.tsv", "comparison.tsv"):
        traits = {line.split("\t")[0] for line in (run / "report" / table).read_text().splitlines()[1:]}
        assert flagged in traits and cohort["trait_names"][1] in traits
    named = [line.split("\t") for line in (run / "report" / "uncertified.tsv").read_text().splitlines()]
    assert named[0] == list(workspace_pipeline.UNCERTIFIED_COLUMNS) and [line[0] for line in named[1:]] == [flagged]
    assert (run / "report" / "quarantined.tsv").read_text().splitlines() == ["\t".join(workspace_pipeline.QUARANTINE_COLUMNS)]
    assert list((run / "effects").glob(f"{flagged}.*.tsv.gz"))
    assert (run / "export" / "_COMPLETE.json").is_file()


def test_a_run_restored_from_its_mirror_on_a_new_disk_resumes_where_it_was_preempted(workspace) -> None:
    """Each Batch job has a new disk (run_mirror): a job preempted during the second arm's fit, whose run was pushed,
    resumes on an empty run directory restored from the mirror, fitting only that arm."""
    from sv_pgs.run_mirror import DirectoryRemote, push, restore

    synthetic, client = workspace
    config = WorkspaceConfig.from_mapping(synthetic.raw)
    remote = DirectoryRemote(config.run_directory.with_name("mirror"))
    with pytest.raises(RuntimeError, match="preempted"):
        run_pipeline(config, _bindings(_Recorder(fits_before_failing=1), client), BUDGET)
    push(config.run_directory, remote)
    shutil.rmtree(config.run_directory)
    restore(config.run_directory, remote)
    recorder = _Recorder()
    run_pipeline(config, _bindings(recorder, client), BUDGET)
    assert len(recorder.fit_calls) == len(ARMS) - 1 and (config.run_directory / "export" / COMPLETE_FILE).is_file()
    push(config.run_directory, remote)
    shutil.rmtree(config.run_directory)
    restore(config.run_directory, remote)
    recorder = _Recorder()
    run_pipeline(config, _bindings(recorder, client), BUDGET)
    assert recorder.fit_calls == []


def test_without_truth_calls_the_run_says_so_and_keeps_the_stored_dosages(workspace) -> None:
    synthetic, client = workspace
    two_threads = dataclasses.replace(BUDGET, cpu_threads=2)
    config = WorkspaceConfig.from_mapping({**synthetic.raw, "truth_calls": None, "imputation_reference_ids": None})
    summaries = run_pipeline(config, _bindings(_Recorder(), client), two_threads, through="measurement")
    run = config.run_directory
    assert summaries["samples"]["calibration_pairs"] == 0 and summaries["store"]["halves"] == ["imputed_dosage", "imputed_dosage"]
    assert not summaries["measurement"]["recalibrated"] and summaries["measurement"]["store"] == "store/store"
    certificate = json.loads((run / "measurement" / "certificate.json").read_text())
    assert all(entry["pairs_supplied"] == 0 and entry["recalibration"].startswith("not applied") for entry in certificate.values())
    rows = (run / "samples" / "rows.tsv").read_text()
    assert "long_read" not in rows and "\n5010\t" in rows


def test_the_calibration_pairs_leave_out_the_imputation_reference_and_its_members_close_relatives(workspace) -> None:
    """7001's genome was in the imputation reference, and 7004 is a first-degree relative of 5004, whose genome was too:
    both imputed rows are leave-in imputations, so neither calibrates. 7006, only third-degree to 5004, does."""
    synthetic, client = workspace
    (synthetic.root / "reference_ids.tsv").write_text("research_id\n7001\n5004\n")
    with (synthetic.root / "relatedness.tsv").open("a") as handle:
        handle.write("5004\t7004\t0.25\n7006\t5004\t0.05\n")
    summaries = run_pipeline(WorkspaceConfig.from_mapping(synthetic.raw), _bindings(_Recorder(), client), BUDGET, through="samples")
    record = json.loads((Path(synthetic.raw["run_directory"]) / "samples" / "samples.json").read_text())
    # The store still gathers every pair's codes; the measurement model is fitted on the others.
    assert len(record["calibration_pairs"]) == len(TRUTH_OF_SEQUENCING)
    assert {record["calibration_pairs"][position][0] for position in record["calibrating"]} == {"7002", "7003", "7005", "7006"}
    account = summaries["samples"]["calibration"]
    assert summaries["samples"]["calibration_pairs"] == 4 and account["reference_overlap"].startswith("checked")
    # The list states the panel; the run cannot see which genomes the imputation read, and says so.
    assert "cannot verify" in account["reference_overlap"]
    # One member and one relative: a count of 1 to 20 is never written.
    assert account["reference_members_left_out"] == account["relatives_left_out"] == "1-20 (suppressed)"


def test_with_truth_calls_and_no_reference_list_no_pair_calibrates_and_the_run_says_why(workspace) -> None:
    """Without the reference list, whether a truth person's imputed row is a leave-in imputation is unchecked, so the
    measurement model takes its no-truth form. The list, once given, recalibrates without rebuilding the store."""
    synthetic, client = workspace
    unchecked = WorkspaceConfig.from_mapping({**synthetic.raw, "imputation_reference_ids": None})
    summaries = run_pipeline(unchecked, _bindings(_Recorder(), client), BUDGET, through="measurement")
    run = unchecked.run_directory
    account = summaries["samples"]["calibration"]
    assert summaries["samples"]["calibration_pairs"] == 0 and account["reference_overlap"].startswith("unchecked")
    assert account["pairs_withheld"] == "1-20 (suppressed)" and summaries["store"]["halves"][-1] == "long_read_calls"
    assert not summaries["measurement"]["recalibrated"] and summaries["measurement"]["calibration"] == account
    certificate = json.loads((run / "measurement" / "certificate.json").read_text())
    assert all(entry["pairs_supplied"] == 0 and entry["calibration"] == account for entry in certificate.values())
    stored = _completed_at(run, "store")
    summaries = run_pipeline(WorkspaceConfig.from_mapping(synthetic.raw), _bindings(_Recorder(), client), BUDGET, through="measurement")
    assert summaries["measurement"]["recalibrated"] and _completed_at(run, "store") == stored


def _store_research_ids() -> list[str]:
    """Each store sample's research ID in store column order: the imputed halves' through the crosswalk, then TRUTH."""
    return [_research_of(name) for half in _sequencing_names() for batch in half for name in batch] + list(TRUTH)


def test_a_gatksv_call_set_adds_direct_rows_that_only_with_sv_fits(workspace) -> None:
    """The direct read-based channel on the dry-run cohort: every kept GATK-SV record a store row in coordinate order,
    filled no-calls, its reported r^2 from the hard-call rule or the CNQ bound, left out of snv_indel and fitted by
    with_sv, and carried by the effects tables with its class and length; a rerun skips every completed step."""
    synthetic, client = workspace
    config = WorkspaceConfig.from_mapping({**synthetic.raw, "direct_sv": {**DIRECT_SV_NULL, "gatksv_calls": synthetic.gatksv_pattern}})
    recorder = _Recorder()
    # Two threads: the chromosomes are merged in two processes.
    summaries = run_pipeline(config, _bindings(recorder, client), dataclasses.replace(BUDGET, cpu_threads=2), through="effects")
    run = config.run_directory
    direct = summaries["direct_sv"]
    kept = [record for record in GATKSV_RECORDS if record[3] in ("PASS", "MULTIALLELIC")]
    assert direct["store"] == "direct_sv/store" and direct["measurement_model"] == "direct_sv/measurement_model.npz"
    assert direct["direct_rows"] == len(kept) * len(CHROMOSOMES)
    assert direct["skipped_records"] == {"FILTER UNRESOLVED": len(CHROMOSOMES), "FILTER HIGH_NCR": len(CHROMOSOMES)}
    # The insertion is imputed record 8's own event: it pairs one to one and fills its no-calls from that DS.
    assert direct["pairs"] == len(CHROMOSOMES) and direct["filled_from_paired_imputed_ds"] == len(CHROMOSOMES)
    assert direct["reliability_rules"]["hard_call"] == 3 * len(CHROMOSOMES) and direct["reliability_rules"]["cnq_bound"] == len(CHROMOSOMES)
    assert all(value == "null: not implemented, not read" for value in direct["optional_sources"].values())
    certificate = json.loads((run / "direct_sv" / "certificate.json").read_text())
    assert certificate["uncalibrated_records"] == direct["direct_rows"] and certificate["direct_calls_fused"] == 0
    assert "hard-call ALT count" in certificate["reliability_source"] and "CNQ" in certificate["reliability_source"]

    store_ids = _store_research_ids()
    with DosageStore.open(run / "measurement" / "store") as before, DosageStore.open(run / "direct_sv" / "store") as after:
        table = after.variant_table
        assert after.n_variants == before.n_variants + direct["direct_rows"]
        assert after.manifest_attributes["direct_calls"]["rows"] == direct["direct_rows"]
        assert after.manifest_attributes["half_measurements"] == before.manifest_attributes["half_measurements"]
        assert table.annotation_legends["row_source"] == ("imputed", "direct")
        is_direct = np.asarray(table.annotations["row_source"]) == 1
        # Each direct row after the imputed records at its POS: on chr21, after record 0 (POS 1000) and record 8 (POS 6000).
        assert np.flatnonzero(is_direct[: len(RECORDS) + len(kept)]).tolist() == [1, 10, 11, 12]
        assert np.all(np.diff(table.position[: len(RECORDS) + len(kept)]) >= 0)
        # The imputed records keep their codes, classes and annotations.
        np.testing.assert_array_equal(after.read_codes(0, after.n_variants)[~is_direct], before.read_codes(0, before.n_variants))
        np.testing.assert_array_equal(table.variant_class[~is_direct], before.variant_table.variant_class)
        np.testing.assert_array_equal(table.annotations["sv_length"][~is_direct], before.variant_table.annotations["sv_length"])
        rows = np.flatnonzero(is_direct)[: len(kept)]
        assert [VARIANT_CLASS_LEGEND[code] for code in table.variant_class[rows]] == ["deletion", "insertion", "copy_number", "duplication"]
        assert table.annotations["sv_length"][rows].tolist() == [300.0, 52.0, 5000.0, 40.0]
        assert table.variant_ids(rows) == [f"{record[1]}_chr21" for record in kept]
        # The pair is one unbreakable group with imputed record 8.
        assert table.group_first[rows[1]] == table.group_first[9]
        codes = after.read_codes(0, after.n_variants)[rows]
        for row, (_position, _identifier, _alt, _filter, _info, kind) in enumerate(kept):
            record = GATKSV_RECORDS.index(kept[row])
            calls = [synthetic.gatksv_calls.get(("chr21", person), (None,) * len(GATKSV_RECORDS))[record] for person in store_ids]
            called = np.array([value is not None for value in calls])
            values = np.array([0 if value is None else value for value in calls])
            unit = int(table.codes_per_unit[rows[row]])
            # A called entry is stored exactly; a copy number at floor(254 / its largest called copy number) per copy.
            assert unit == (254 // values[called].max() if kind == "copy_number" else 127)
            np.testing.assert_array_equal(codes[row][called], values[called] * unit)
            if kind != "paired":
                # An unpaired record's no-calls (absent people's included) take its observed mean, never zero.
                assert set(codes[row][~called].tolist()) == {int(np.floor(values[called].mean() * unit + 0.5))}
        assert int(table.value_origin[rows[2]]) == -2
        # The copy number's reported r^2 is the CNQ bound over the store's called samples.
        record = GATKSV_RECORDS.index(kept[2])
        calls = np.array([value for person in store_ids if (value := synthetic.gatksv_calls.get(("chr21", person), (None,) * 6)[record]) is not None])
        reported = np.load(run / "direct_sv" / "rows" / "chr21.npz")["reported_reliability"]
        assert reported.tolist()[:2] == [1.0, 1.0] and reported[3] == 1.0
        assert reported[2] == pytest.approx(1.0 - 10.0 ** (-GATKSV_QUALITY / 10.0) / calls.var())

    # snv_indel leaves every direct row out, the 40 bp duplication included; with_sv fits them.
    model = _MeasurementModel.load(run / "direct_sv" / "measurement_model.npz")
    imputed_model = _MeasurementModel.load(run / "measurement" / "measurement_model.npz")
    np.testing.assert_array_equal(model.log_reliability[~is_direct], imputed_model.log_reliability)
    assert np.all(np.isfinite(model.log_reliability[is_direct]))
    arms = set()
    for call in recorder.fit_calls:
        assert call.log_variance_offset.shape == (is_direct.shape[0],)
        arm = "snv_indel" if np.all(np.isneginf(call.log_variance_offset[is_direct])) else "with_sv"
        arms.add(arm)
        if arm == "with_sv":
            np.testing.assert_array_equal(call.log_variance_offset, model.log_reliability)
    assert arms == set(ARMS)
    for trait in ("atrial_fibrillation", "total_bilirubin"):
        for arm in ARMS:
            effects = pd.read_csv(run / "effects" / f"{trait}.{arm}.tsv.gz", sep="\t")
            own = effects[effects["row_source"] == "direct"]
            if arm == "snv_indel":
                assert own.empty
            else:
                assert set(own["variant_class"]) == {"deletion", "insertion", "copy_number", "duplication"}
                assert set(own["length"]) == {300.0, 52.0, 5000.0, 40.0}

    # A rerun skips every completed step.
    finished = {step: _completed_at(run, step) for step in STEP_NAMES[: STEP_NAMES.index("effects") + 1]}
    rerun = _Recorder()
    run_pipeline(config, _bindings(rerun, client), BUDGET, through="effects")
    assert {step: _completed_at(run, step) for step in finished} == finished and not rerun.fit_calls


def test_a_null_gatksv_call_set_adds_no_row_and_says_so(workspace) -> None:
    synthetic, client = workspace
    config = WorkspaceConfig.from_mapping({**synthetic.raw, "direct_sv": DIRECT_SV_NULL})
    summaries = run_pipeline(config, _bindings(_Recorder(), client), BUDGET, through="direct_sv")
    direct = summaries["direct_sv"]
    assert direct["direct_rows"] == 0 and direct["store"] == summaries["measurement"]["store"]
    assert direct["measurement_model"] == "measurement/measurement_model.npz" and "gatksv_calls is null" in direct["gatksv_calls"]
    certificate = json.loads((config.run_directory / "direct_sv" / "certificate.json").read_text())
    assert certificate["gatksv_calls"] == direct["gatksv_calls"] and not (config.run_directory / "direct_sv" / "store").exists()


def test_direct_sv_refuses_a_source_it_does_not_read(workspace) -> None:
    synthetic, _client = workspace
    with pytest.raises(ValueError, match="ctyper_copy_numbers is not implemented"):
        WorkspaceConfig.from_mapping({**synthetic.raw, "direct_sv": {**DIRECT_SV_NULL, "ctyper_copy_numbers": "ctyper.tsv"}})
    with pytest.raises(ValueError, match="names exactly"):
        WorkspaceConfig.from_mapping({**synthetic.raw, "direct_sv": {"gatksv_calls": None}})
    with pytest.raises(ValueError, match=r"over \{chromosome\}"):
        WorkspaceConfig.from_mapping({**synthetic.raw, "direct_sv": {**DIRECT_SV_NULL, "gatksv_calls": "calls.vcf.gz"}})
    # A missing direct_sv says where the CDR keeps the call set.
    with pytest.raises(ValueError, match=r"AoU_srWGS_SV\.v8\.\{chromosome\}\.vcf\.gz"):
        WorkspaceConfig.from_mapping({key: value for key, value in synthetic.raw.items() if key != "direct_sv"})


def test_a_sidecar_that_disagrees_with_the_popped_records_stops_the_store(workspace) -> None:
    synthetic, client = workspace
    manifest = synthetic.root / "strata" / "chr22.json"
    manifest.write_text(json.dumps({**json.loads(manifest.read_text()), "ids_md5": "0" * 32}))
    with pytest.raises(ValueError, match="md5"):
        run_pipeline(WorkspaceConfig.from_mapping(synthetic.raw), _bindings(_Recorder(), client), BUDGET, through="store")


def test_store_shards_on_their_own_disks_meet_in_the_mirror_and_make_the_store_one_run_makes(workspace, monkeypatch) -> None:
    """A fan-out store job's tasks (store_shard) start on new disks from one mirror, each converts its chromosome and
    pushes it without completing the step; the next job, restored from the mirror, writes the store a single run writes
    and converts nothing again."""
    from sv_pgs.run_mirror import DirectoryRemote, push, restore

    synthetic, client = workspace
    config = WorkspaceConfig.from_mapping(synthetic.raw)
    run = config.run_directory
    whole = run_pipeline(config, _bindings(_Recorder(), client), BUDGET, through="store")
    # Every file but the completion record, whose time differs (its summary is compared).
    tree = lambda root: {path.relative_to(root).as_posix(): path.read_bytes() for path in root.rglob("*") if path.is_file() and path.name != COMPLETE_FILE}
    expected = tree(run / "store")
    shutil.rmtree(run)
    remote = DirectoryRemote(run.with_name("mirror"))
    run_pipeline(config, _bindings(_Recorder(), client), BUDGET, through="samples")
    push(run, remote)
    shutil.rmtree(run)
    shards = []
    for shard, chromosome in enumerate(CHROMOSOMES):
        restore(run, remote)
        summaries = run_pipeline(config, _bindings(_Recorder(), client), BUDGET, through="export", store_shard=shard)
        assert list(summaries) == ["samples"] and not (run / "store").exists()
        assert sorted(path.name for path in (run / "store.partial" / "chromosomes").iterdir()) == [f"{chromosome}.json"]
        shards.append(run.rename(run.with_name(f"shard{shard}")))
    events = {path.name.split("_", 1)[1] for path in config.marker_directory.iterdir()}
    assert {f"store_converted-{chromosome}" for chromosome in CHROMOSOMES} | {"store_part-done"} <= events
    for directory in shards:
        push(directory, remote)
    restore(run, remote)
    monkeypatch.setattr(workspace_pipeline, "_convert_chromosome", lambda *arguments: pytest.fail("a mirrored chromosome was converted again"))
    merged = run_pipeline(config, _bindings(_Recorder(), client), BUDGET, through="store")
    assert merged["store"] == whole["store"] and tree(run / "store") == expected
    with pytest.raises(ValueError, match="store_shard"):
        run_pipeline(config, _bindings(_Recorder(), client), BUDGET, store_shard=len(CHROMOSOMES))


def test_each_arm_can_be_fitted_by_its_own_run(workspace) -> None:
    """A run given fit_arms fits only those and leaves the fit step to the run that finds every arm on disk."""
    synthetic, client = workspace
    config = WorkspaceConfig.from_mapping(synthetic.raw)
    partial = config.run_directory / "fit.partial" / "models"
    first = _Recorder()
    summaries = run_pipeline(config, _bindings(first, client), BUDGET, through="fit", fit_arms=("with_sv",))
    assert "fit" not in summaries and len(first.fit_calls) == 1
    assert (partial / "with_sv").is_dir() and not (partial / "snv_indel").exists()
    assert any(path.name.endswith("_fit_part-done") for path in config.marker_directory.iterdir())
    second = _Recorder()
    summaries = run_pipeline(config, _bindings(second, client), BUDGET, through="fit", fit_arms=("snv_indel",))
    assert "fit" in summaries and len(second.fit_calls) == 1
    # The snv_indel arm leaves every structural record out; the with_sv arm fitted some.
    assert np.isinf(second.fit_calls[0].log_variance_offset).sum() > np.isinf(first.fit_calls[0].log_variance_offset).sum()
    assert set(summaries["fit"]["arms"]) == set(ARMS)
    with pytest.raises(ValueError, match="fit_arms"):
        run_pipeline(config, _bindings(_Recorder(), client), BUDGET, fit_arms=("all",))


def test_a_marker_that_cannot_be_written_does_not_stop_the_run(workspace, monkeypatch) -> None:
    """Markers are status signals: on the bucket mount, fan-out tasks touching one name in one second raised Stale file
    handle, which must not end the step."""
    synthetic, client = workspace
    config = WorkspaceConfig.from_mapping(synthetic.raw)

    def stale(self, *arguments, **keywords):
        raise OSError(116, "Stale file handle")

    monkeypatch.setattr(Path, "touch", stale)
    summaries = run_pipeline(config, _bindings(_Recorder(), client), BUDGET, through="samples")
    assert list(summaries) == ["samples"] and (config.run_directory / "samples" / COMPLETE_FILE).is_file()


def test_the_code_moments_split_over_threads_are_the_exact_moments_of_each_group_s_columns(workspace) -> None:
    synthetic, client = workspace
    config = WorkspaceConfig.from_mapping(synthetic.raw)
    run_pipeline(config, _bindings(_Recorder(), client), BUDGET, through="store")
    with DosageStore.open(config.run_directory / "store" / "store") as store:
        codes = store.read_codes(0, store.n_variants).astype(np.int64)
        columns = np.arange(1, store.n_samples, 2)
        groups = np.arange(columns.shape[0]) % 3
        for threads in (1, 3, 64):
            counts, sums, squares = workspace_pipeline._group_code_moments(store, columns, groups, 4, dataclasses.replace(BUDGET, cpu_threads=threads))
            assert counts.tolist() == [int((groups == group).sum()) for group in range(4)]
            for group in range(4):
                values = codes[:, columns[groups == group]]
                assert np.array_equal(sums[:, group], values.sum(axis=1)) and np.array_equal(squares[:, group], (values * values).sum(axis=1))


def test_popped_records_without_info_cm_take_their_genetic_positions_from_the_genetic_map(workspace) -> None:
    """AoU's popped batches declare INFO/CM but carry it on no record: the store interpolates every record's genetic
    position from genetic_map and says so, and stops naming genetic_map where it is null."""
    synthetic, client = workspace
    for label, batches in HALVES:
        for batch, _width in batches:
            for chromosome in CHROMOSOMES:
                path = synthetic.root / label / f"{chromosome}.{batch}.vcf"
                path.write_text(re.sub(r";CM=[0-9.]+", "", path.read_text()))
    with pytest.raises(ValueError, match=r"chr21: the batches' records lack INFO/CM.*genetic_map"):
        run_pipeline(WorkspaceConfig.from_mapping(synthetic.raw), _bindings(_Recorder(), client), BUDGET, through="store")
    (synthetic.root / "maps").mkdir()
    for chromosome in CHROMOSOMES:
        (synthetic.root / "maps" / f"plink.{chromosome}.GRCh38.map").write_text(f"{chromosome} . 0.0 1\n{chromosome} . 20.0 10001\n")
    config = WorkspaceConfig.from_mapping({**synthetic.raw, "genetic_map": str(synthetic.root / "maps" / "plink.{chromosome}.GRCh38.map")})
    summaries = run_pipeline(config, _bindings(_Recorder(), client), BUDGET, through="store")
    assert summaries["store"]["genetic_position"] == {
        chromosome: f"genetic_map plink.{chromosome}.GRCh38.map, as INFO/CM is on 0 of {len(RECORDS)} records" for chromosome in CHROMOSOMES
    }
    with DosageStore.open(config.run_directory / "store" / "store") as store:
        np.testing.assert_allclose(store.variant_table.genetic_position_cm, [0.002 * (position - 1) for position, *_rest in RECORDS] * len(CHROMOSOMES))


def _strata(identifiers, carried, totals) -> StrataSites:
    count = len(identifiers)
    zeros = np.zeros(count, dtype=np.int64)
    return StrataSites(zeros, tuple(identifiers), zeros.astype(np.uint64), zeros, zeros, np.array(carried), np.array(totals), zeros.astype(float), None, "", "")


def test_path_counts_count_only_the_first_kept_paths_and_check_the_sidecar() -> None:
    bubble = [frozenset({"x"})] + [frozenset({"y"})] * 10 + [frozenset({"x", "y"})]
    counts = path_counts(_strata(["x", "y", "z"], [2, 11, 1], [12, 12, 1]), [bubble])
    assert counts.kept.tolist() == [10, 10, 1] and counts.carrying.tolist() == [1, 9, 1] and counts.bubble.tolist() == [0, 0, -1]
    with pytest.raises(ValueError, match="sidecar says"):
        path_counts(_strata(["x"], [3], [12]), [bubble])
    with pytest.raises(ValueError, match="no bubble.split path"):
        path_counts(_strata(["w"], [1], [2]), [bubble])


def test_export_refuses_a_count_of_one_to_twenty(tmp_path: Path) -> None:
    table = tmp_path / "accuracy.tsv"
    table.write_text("trait\tcell\tlevel\tn\tmetric\tvalue\nldl\tfold\t0\t25\tr2\t0.1\nldl\thalf\tlong_read\t1-20 (suppressed)\t\t\n")
    check_exportable(table)
    table.write_text("trait\tcell\tlevel\tn\tmetric\tvalue\nldl\thalf\tlong_read\t7\tr2\t0.3\n")
    with pytest.raises(ValueError, match="below 21"):
        check_exportable(table)
    # A disease cell's case count is a participant count too.
    table.write_text("trait\tarm\tcell\tlevel\tn\tcases\tmetric\tvalue\naf\twith_sv\tall\tall\t400\t12\tauc\t0.7\n")
    with pytest.raises(ValueError, match="below 21"):
        check_exportable(table)


def test_a_case_count_is_shown_only_where_neither_it_nor_the_controls_it_implies_is_one_to_twenty() -> None:
    assert workspace_pipeline._cases_text(400, 30) == "30" and workspace_pipeline._cases_text(400, 0) == "0"
    # 25 cases of 30 imply 5 controls; 400 rows with 390 cases imply 10.
    assert workspace_pipeline._cases_text(30, 25) == "suppressed" and workspace_pipeline._cases_text(400, 390) == "suppressed"
    assert workspace_pipeline._cases_text(400, 12) == "suppressed" and workspace_pipeline._cases_text(50, None) == ""
    assert not workspace_pipeline._shown(30, 25) and workspace_pipeline._shown(60, 30) and workspace_pipeline._shown(21, None)


def test_a_long_read_row_takes_the_reference_half_so_the_indicators_differ() -> None:
    assert pipeline_half_levels(["B", "A", "long_read"], ["imputed", "imputed", "long_read"]) == ("B", "A", "A")


def test_the_config_names_every_key_and_canonical_phenotypes(workspace) -> None:
    synthetic, _client = workspace
    with pytest.raises(ValueError, match="every key is required"):
        WorkspaceConfig.from_mapping({key: value for key, value in synthetic.raw.items() if key != "truth_calls"})
    # A missing gene_annotation says which public file to point it at.
    with pytest.raises(ValueError, match=r"gencode\.v38\.basic\.annotation\.gtf\.gz"):
        WorkspaceConfig.from_mapping({key: value for key, value in synthetic.raw.items() if key != "gene_annotation"})
    with pytest.raises(ValueError, match="reference panel"):
        WorkspaceConfig.from_mapping({key: value for key, value in synthetic.raw.items() if key != "imputation_reference_ids"})
    # The reference list is read only for the truth calls' calibration pairs.
    with pytest.raises(ValueError, match="imputation_reference_ids"):
        WorkspaceConfig.from_mapping({**synthetic.raw, "truth_calls": None})
    with pytest.raises(ValueError, match="canonical name"):
        WorkspaceConfig.from_mapping({**synthetic.raw, "diseases": ["afib"]})


LAUNCHER = Path(__file__).resolve().parents[1] / "launcher" / "workspace"


def _filled(text: str) -> str:
    """A template with its numeric placeholders set to 1, the two half labels to A and B, and every other
    placeholder to its own name in lower case."""
    for name, value in (("FOLD_SEED", "1"), ("MAX_RETRIES", "1"), ("GPU_COUNT", "1"), ("RUN_DISK_GB", "1"), ("TASK_COUNT", "1"), ("STORE_SHARDS", ""), ("HALF_A_LABEL", "A"), ("HALF_B_LABEL", "B")):
        text = text.replace("${" + name + "}", value)
    return re.sub(r"\$\{([A-Z_]+)\}", lambda match: match.group(1).lower(), text)


def test_the_launcher_templates_parse_and_name_the_preregistered_panel() -> None:
    raw = json.loads(_filled((LAUNCHER / "run_config.template.json").read_text()))
    config = WorkspaceConfig.from_mapping(raw)
    # The study's panel is its diseases; the quantitative traits are left out of the template to keep the fit in budget.
    assert (len(config.diseases), len(config.traits), config.fold_count) == (15, 0, 5) and config.truth_calls is None
    # The direct channel reads the CDR's GATK-SV call set, one VCF per chromosome.
    assert config.direct_sv is not None and config.direct_sv.gatksv_calls == (
        "/mnt/disks/cdr/wgs/short_read/structural_variants/vcf/full/AoU_srWGS_SV.v8.{chromosome}.vcf.gz"
    )
    # Every catalogued disease of the panel, in the catalogue's order (the study abstract's five first, hypertension
    # leading); Alzheimer's disease, catalogued last for the APOE locus's regional run, is not one of it.
    assert config.diseases == tuple(definition.canonical_name for definition in DISEASE_DEFINITIONS[:-1])
    assert DISEASE_DEFINITIONS[-1].canonical_name == "alzheimers_disease"
    for job in ("job_store.json.template", "job_fit.json.template"):
        network = json.loads(_filled((LAUNCHER / job).read_text()))["allocationPolicy"]["network"]["networkInterfaces"][0]
        assert network["noExternalIpAddress"] is True


def test_the_launcher_sends_nothing_outside_the_workspace() -> None:
    text = "\n".join(path.read_text().lower() for path in LAUNCHER.iterdir() if path.is_file())
    # The service account's "email" key names an identity inside the workspace; it sends nothing.
    assert not re.search(r"https?://|curl|wget|webhook|notif|sendmail|smtp|\bmail\b|slack|pubsub|scp |rsync", text)


def test_a_record_spanning_several_repeats_chains_no_unsplittable_group_past_the_block_cap() -> None:
    """Three repeats of 40 TR indels each with SNVs between them, and one long deletion over all three: the deletion
    is inside no repeat (``tr_loci``), so it is a deletion, not a TR, and bridges nothing; each locus groups its TR
    records alone, and every unsplittable group fits a block cap of one repeat's records. Merging bridged intervals
    and grouping every record of a locus chained the public chr22 panel sites into a 34,180-record group."""
    repeats = (np.array([1_000, 3_000, 5_000]), np.array([1_200, 3_200, 5_200]))
    tr_positions = np.concatenate([start + 5 + 4 * np.arange(40) for start in repeats[0]])
    snv_positions = np.arange(1_500, 5_000, 500)
    positions = np.sort(np.concatenate([tr_positions, snv_positions, [900]]))
    is_tr = np.isin(positions, tr_positions)
    deletion = positions == 900
    core_starts = positions - 1
    core_ends = np.where(deletion, 5_500, core_starts + np.where(is_tr, 2, 1))
    changes = np.where(is_tr, -2, np.where(deletion, -4_600, 0))
    loci = tr_loci(*repeats, core_starts, core_ends, changes)
    assert loci.record_locus[deletion][0] == np.iinfo(np.uint32).max
    code = {variant_class: index for index, variant_class in enumerate(VARIANT_CLASSES)}
    classes = np.where(is_tr, code[VariantClass.STR_VNTR_REPEAT], np.where(deletion, code[VariantClass.DELETION], code[VariantClass.SNV]))
    group_first = store_group_first(np.full(positions.shape[0], -1), positions, classes, loci.record_locus)
    validate_cut_allowed(cut_allowed_from_groups(group_first), 40)
    assert np.unique(group_first[is_tr]).shape[0] == 3


def test_the_genetic_variance_decomposition_is_beta_sigma_beta_under_ld() -> None:
    """On synthetic genotypes where each structural column tags a small one, the decomposition of the scores' variance
    is beta' Sigma beta split by part, and the per-record sum of squared effects (no LD) is not the variance."""
    rng = np.random.default_rng(11)
    small = rng.binomial(2, 0.3, (4000, 6)).astype(np.float64)
    structural = np.clip(small[:, :2] + rng.binomial(1, 0.1, (4000, 2)), 0.0, 2.0)
    genotypes = np.column_stack([structural, small])
    standardized = (genotypes - genotypes.mean(axis=0)) / genotypes.std(axis=0)
    covariance = np.cov(standardized, rowvar=False, bias=True)
    whole = np.array([0.4, -0.3, 0.5, 0.2, -0.1, 0.3, 0.2, -0.4])
    in_part = np.arange(whole.shape[0]) < 2
    part, rest = np.where(in_part, whole, 0.0), np.where(in_part, 0.0, whole)
    result = workspace_pipeline.genetic_variance_decomposition(standardized @ whole, standardized @ part)
    assert result["total_variance"] == pytest.approx(whole @ covariance @ whole)
    assert result["structural_variance"] == pytest.approx(part @ covariance @ part)
    assert result["small_variance"] == pytest.approx(rest @ covariance @ rest)
    assert result["covariance_term"] == pytest.approx(2.0 * part @ covariance @ rest)
    assert result["structural_share"] == pytest.approx(part @ covariance @ whole / (whole @ covariance @ whole))
    assert result["total_variance"] == pytest.approx(result["structural_variance"] + result["small_variance"] + result["covariance_term"])
    # The structural columns' LD with the small ones moves the share away from their sum of squared effects.
    assert abs(result["covariance_term"]) > 0.1 * result["total_variance"]
    assert result["structural_share"] != pytest.approx(np.sum(part**2) / np.sum(whole**2), rel=0.05)
    assert np.isnan(workspace_pipeline.genetic_variance_decomposition(np.ones(30), np.ones(30))["structural_share"])


def test_scoring_holds_at_most_a_quarter_of_the_host_in_saved_models_per_store_read() -> None:
    # A quarter of 80 bytes is 20: two 10-byte models share a read; a model past the quarter is read alone.
    assert workspace_pipeline.score_batches([10, 10, 10], 80) == [range(0, 2), range(2, 3)]
    assert workspace_pipeline.score_batches([30, 5, 5], 80) == [range(0, 1), range(1, 3)]
    assert workspace_pipeline.score_batches([], 80) == []
    with pytest.raises(MemoryError, match="before it is loaded"):
        workspace_pipeline.score_batches([5], 1)


def test_the_simpler_prior_reaches_every_fit_call(workspace):
    synthetic, client = workspace
    config = WorkspaceConfig.from_mapping({**synthetic.raw, 'annotated_prior': False})
    recorder = _Recorder()
    run_pipeline(config, _bindings(recorder, client), BUDGET, through='fit')
    assert recorder.fit_calls and all(not call.annotated_prior for call in recorder.fit_calls)


def test_frozen_probability_calibration_uses_only_reserved_development_families(workspace):
    from sv_pgs.probability_calibration import FrozenCalibration
    synthetic, client = workspace
    config = WorkspaceConfig.from_mapping({**synthetic.raw, 'calibration_fraction': .3, 'annotated_prior': False})
    recorder = _Recorder()
    run_pipeline(config, _bindings(recorder, client), BUDGET, through='score')
    assert all(not call.annotated_prior for call in recorder.fit_calls)
    cohort = np.load(config.run_directory / 'cohort' / 'cohort.npz')
    predictions = np.load(config.run_directory / 'score' / 'predictions.npz')
    records = json.loads((config.run_directory / 'score' / 'calibrators.json').read_text())
    names = json.loads((config.run_directory / 'cohort' / 'cohort.json').read_text())['model_names']
    assert records
    assert not np.any(cohort['calibration'] & (cohort['training'] | cohort['held_out']))
    for entry in records:
        model = names.index(entry['model'])
        arm = ARMS.index(entry['arm'])
        rows = cohort['calibration'][:, model]
        trait = cohort['model_traits'][model]
        assert not set(cohort['families'][rows]) & set(cohort['families'][cohort['training'][:, model]])
        frozen = FrozenCalibration.fit(predictions['linear_predictor'][arm, rows, model], cohort['targets'][rows, trait])
        np.testing.assert_allclose([entry['intercept'], entry['slope']], [frozen.intercept, frozen.slope])
        np.testing.assert_allclose(predictions['calibrated_logits'][arm, :, model], frozen.logits(predictions['linear_predictor'][arm, :, model]))


def test_two_diseases_with_the_same_eligible_rows_reserve_the_same_families_and_are_fitted_together() -> None:
    from sv_pgs.stage2_wiring import model_groups

    generator = np.random.default_rng(5)
    rows, fold_count, diseases = 240, 2, 3
    families = generator.integers(0, 80, size=rows)
    folds = families % fold_count
    # The third disease observes fewer people than the other two.
    observed = np.ones((rows, diseases), dtype=bool)
    observed[families < 10, 2] = False
    model_traits = np.repeat(np.arange(diseases), fold_count + 1)
    model_folds = np.tile(np.arange(fold_count + 1), diseases)
    training = observed[:, model_traits] & (folds[:, None] != model_folds[None, :])
    calibration = workspace_pipeline.development_calibration(training, model_traits, model_folds, diseases, families, 0.2, 7)
    assert np.all(calibration.any(axis=0)) and not np.any(calibration & ~training)
    for fold in range(fold_count + 1):
        np.testing.assert_array_equal(calibration[:, fold], calibration[:, fold_count + 1 + fold])
    groups = model_groups(np.arange(rows), np.ones((model_traits.shape[0], 1), dtype=bool), training & ~calibration, (TraitType.QUANTITATIVE,) * model_traits.shape[0])
    assert [group.tolist() for group in groups] == [[0, 3], [1, 4], [2, 5], [6], [7], [8]]


def test_a_single_split_fits_one_model_per_trait_and_reports_its_test_fold_alone(workspace) -> None:
    """cross_fit false: each trait's one model trains on every fold but the test fold (fold 0 without a panel), is
    scored on that fold alone, and is the model the effects step reads; the report's cells hold the test fold only."""
    synthetic, client = workspace
    config = WorkspaceConfig.from_mapping({**synthetic.raw, "cross_fit": False, "panel": None})
    assert config.model_folds() == (0,)
    run_pipeline(config, _bindings(_Recorder(), client), BUDGET)
    run = config.run_directory
    cohort = np.load(run / "cohort" / "cohort.npz")
    design = json.loads((run / "cohort" / "cohort.json").read_text())
    traits = len(design["trait_names"])
    assert cohort["model_traits"].tolist() == list(range(traits)) and cohort["model_folds"].tolist() == [0] * traits
    assert design["model_names"] == [f"{name}/fold0" for name in design["trait_names"]]
    observed = np.isfinite(cohort["targets"])
    np.testing.assert_array_equal(cohort["held_out"], observed & (cohort["folds"] == 0)[:, None])
    assert not np.any(cohort["training"] & (cohort["folds"] == 0)[:, None])
    with (run / "report" / "accuracy.tsv").open() as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    assert {row["level"] for row in rows if row["cell"] == "fold"} == {"0"}
    assert all(row["value"] != "nan" for row in rows if row["cell"] == "all" and row["value"])
    # The cross-fitting default keeps its cohort key, so its runs are not redone.
    assert WorkspaceConfig.from_mapping(synthetic.raw).cross_fit and WorkspaceConfig.from_mapping(synthetic.raw).model_folds() == (0, 1, 2)


def test_fit_keeps_listed_substitutions_and_every_structural_record(tmp_path: Path, workspace) -> None:
    # (chromosome, POS, ID, REF length, ALT length, SVLEN): a listed A/G site's A>G (kept) and A>T (other alleles),
    # an rsID substitution at a listed position (kept on position), an unlisted substitution, an indel at a listed
    # position, a substitution listed on the other strand (kept), and an unlisted structural record (kept).
    records = (
        (2, 100, "chr2:100:A:G", 1, 1, 0.0), (2, 100, "chr2:100:A:T", 1, 1, 0.0), (2, 200, "rs7", 1, 1, 0.0), (2, 300, "chr2:300:C:T", 1, 1, 0.0),
        (2, 400, "chr2:400:AT:A", 2, 1, 1.0), (5, 100, "chr5:100:G:A", 1, 1, 0.0), (5, 150, "sv1", 1, 1, 5000.0),
    )
    chromosome, position, ids, ref_length, alt_length, sv_length = zip(*records)
    count = len(records)
    table = VariantTable(
        chromosome=np.array(chromosome, dtype=np.int8), position=np.array(position, dtype=np.int64), genetic_position_cm=np.zeros(count),
        ref_length=np.array(ref_length, dtype=np.int32), alt_length=np.array(alt_length, dtype=np.int32),
        variant_class=np.full(count, VARIANT_CLASSES.index(VariantClass.SNV), dtype=np.uint8), codes_per_unit=np.full(count, 254, dtype=np.uint8),
        value_origin=np.zeros(count, dtype=np.int64), group_first=np.arange(count, dtype=np.int64), sum_code=np.full(count, 254 * 40, dtype=np.uint64),
        sum_code2=np.zeros(count, dtype=np.uint64), annotations={"sv_length": np.array(sv_length)}, annotation_legends={},
        id_bytes=np.frombuffer("".join(ids).encode(), dtype=np.uint8), id_offsets=np.concatenate([[0], np.cumsum([len(name) for name in ids])]).astype(np.int64),
    )
    sites = tmp_path / "sites.tsv"
    sites.write_text("chromosome\tposition\tallele_a\tallele_b\n2\t100\tA\tG\n2\t200\tC\tT\n2\t400\tA\tT\n5\t100\tC\tT\n")
    log_reliability = np.log(np.linspace(0.9, 0.6, count))
    kept = fit_filtered(table, 100, log_reliability, 0.0, 0.0, sites)
    assert np.isfinite(kept).tolist() == [True, False, True, False, False, True, True]
    np.testing.assert_array_equal(kept[np.isfinite(kept)], log_reliability[[0, 2, 5, 6]])
    np.testing.assert_array_equal(fit_filtered(table, 100, log_reliability, 0.0, 0.0), log_reliability)
    synthetic, _client = workspace
    assert WorkspaceConfig.from_mapping({**synthetic.raw, "fit_variant_sites": str(sites)}).fit_variant_sites == str(sites)
    assert WorkspaceConfig.from_mapping(synthetic.raw).fit_variant_sites is None
    with pytest.raises(ValueError, match="fit_variant_sites"):
        WorkspaceConfig.from_mapping({**synthetic.raw, "fit_variant_sites": 7})


def test_a_fit_region_keeps_the_records_in_its_window_of_every_class(workspace, caplog) -> None:
    # (chromosome, POS, ID, REF length, ALT length, SVLEN, class): a substitution, a short deletion and a structural
    # deletion inside chr5:100-200 (its ends included), a substitution before it, after it and at its position on chr2.
    records = (
        (5, 100, "a", 1, 1, 0.0, VariantClass.SNV), (5, 150, "b", 3, 1, 2.0, VariantClass.DELETION), (5, 200, "c", 1, 1, 0.0, VariantClass.DELETION),
        (5, 160, "d", 900, 1, 899.0, VariantClass.DELETION), (5, 99, "e", 1, 1, 0.0, VariantClass.SNV), (5, 201, "f", 1, 1, 0.0, VariantClass.SNV),
        (2, 150, "g", 1, 1, 0.0, VariantClass.SNV),
    )
    chromosome, position, ids, ref_length, alt_length, sv_length, classes = zip(*records)
    count = len(records)
    table = VariantTable(
        chromosome=np.array(chromosome, dtype=np.int8), position=np.array(position, dtype=np.int64), genetic_position_cm=np.zeros(count),
        ref_length=np.array(ref_length, dtype=np.int32), alt_length=np.array(alt_length, dtype=np.int32),
        variant_class=np.array([VARIANT_CLASSES.index(name) for name in classes], dtype=np.uint8), codes_per_unit=np.full(count, 254, dtype=np.uint8),
        value_origin=np.zeros(count, dtype=np.int64), group_first=np.arange(count, dtype=np.int64), sum_code=np.full(count, 254 * 40, dtype=np.uint64),
        sum_code2=np.zeros(count, dtype=np.uint64), annotations={"sv_length": np.array(sv_length)}, annotation_legends={},
        id_bytes=np.frombuffer("".join(ids).encode(), dtype=np.uint8), id_offsets=np.arange(count + 1, dtype=np.int64),
    )
    log_reliability = np.log(np.linspace(0.9, 0.6, count))
    with caplog.at_level("INFO", logger="sv_pgs.workspace_pipeline"):
        kept = fit_filtered(table, 100, log_reliability, 0.0, 0.0, None, (5, 100, 200))
    assert np.isfinite(kept).tolist() == [True, True, True, True, False, False, False]
    np.testing.assert_array_equal(kept[:4], log_reliability[:4])
    # The window's records by class, short and structural apart: the snv_indel arm fits the short ones.
    assert "the fit region chr5:100-200 keeps 2 deletion short, 1 deletion structural, 1 snv short" in caplog.text
    # The floors still apply inside the window.
    assert np.isfinite(fit_filtered(table, 100, log_reliability, 0.0, 0.82, None, (5, 100, 200))).tolist() == [True, True, False, False, False, False, False]
    with pytest.raises(ValueError, match="holds no record"):
        fit_filtered(table, 100, log_reliability, 0.0, 0.0, None, (5, 300, 400))
    synthetic, _client = workspace
    config = WorkspaceConfig.from_mapping({**synthetic.raw, "fit_region": {"chromosome": 22, "start": 1500, "end": 6500}})
    assert config.fit_region == (22, 1500, 6500) and WorkspaceConfig.from_mapping(synthetic.raw).fit_region is None
    # The region is part of the fit step's key, and of no earlier step's.
    steps = {step.name: step for step in workspace_pipeline._STEPS}
    unbounded = WorkspaceConfig.from_mapping(synthetic.raw)
    assert steps["fit"].inputs(config) != steps["fit"].inputs(unbounded)
    assert all(steps[name].inputs(config) == steps[name].inputs(unbounded) for name in ("cohort", "measurement", "direct_sv"))
    for region in (
        {"chromosome": 19, "start": 1, "end": 2}, {"chromosome": 22, "start": 7, "end": 6}, {"chromosome": 22, "start": 0, "end": 6},
        {"chromosome": 22, "start": 1.0, "end": 6}, {"chromosome": True, "start": 1, "end": 6}, {"chromosome": 22, "start": 1}, [22, 1, 6],
        {"chromosome": "chr22", "start": 1, "end": 6},
    ):
        with pytest.raises(ValueError, match="fit_region"):
            WorkspaceConfig.from_mapping({**synthetic.raw, "fit_region": region})


def test_a_regional_alzheimers_run_fits_its_window_suppresses_its_few_cases_and_compares_ldpred2_on_its_people(
    workspace, monkeypatch, tmp_path: Path
) -> None:
    """The APOE run's shape: one disease with few cases (Alzheimer's: cases by their codes, controls 65 and over), a
    single split, both arms fitted on one window's records alone, every report table exportable, then LDpred2 on exactly
    the snv_indel arm's records and the arms and LDpred2 compared on the same held-out people."""
    from benchmarks.bench_real import aou_compare, aou_ldpred2

    synthetic, client = workspace
    region = (22, 1500, 6500)
    raw = {
        **synthetic.raw, "diseases": ["alzheimers_disease"], "traits": [], "cross_fit": False,
        "panel": {"primary_fold": 0, "weights": {"alzheimers_disease": 1.0}}, "fit_region": dict(zip(("chromosome", "start", "end"), region)),
        "export": ["accuracy.tsv", "comparison.tsv", "panel.tsv", "effects_by_class.tsv", "genetic_variance.tsv"],
    }
    config_path = tmp_path / "run_config.json"
    config_path.write_text(json.dumps(raw))
    config = WorkspaceConfig.read(config_path)
    recorder = _Recorder()
    # Saved models with their provenance, so the driver's handoff check recomputes the window's arm offsets.
    bindings = _artifact_bindings(recorder, client)
    run_pipeline(config, bindings, BUDGET)
    run = config.run_directory
    assert json.loads((run / "cohort" / "cohort.json").read_text())["model_names"] == ["alzheimers_disease/fold0"]
    # Each arm fits the window's records alone: snv_indel its short ones, with_sv those and its structural ones.
    snv_indel, with_sv = recorder.fit_calls
    table = snv_indel.store.variant_table
    inside, structural = workspace_pipeline.region_records(table, region), workspace_pipeline.structural_records(table)
    short_fitted, all_fitted = np.isfinite(snv_indel.log_variance_offset), np.isfinite(with_sv.log_variance_offset)
    assert short_fitted.any() and not np.any(short_fitted & ~inside) and not np.any(short_fitted & structural)
    assert not np.any(all_fitted & ~inside) and np.any(all_fitted & structural)
    np.testing.assert_array_equal(all_fitted & ~structural, short_fitted)
    # Every count of 1 to 20 is suppressed in every staged table (the export step refuses any other).
    assert sorted(path.name for path in (run / "export").glob("*.tsv")) == sorted(raw["export"])
    for name in raw["export"]:
        check_exportable(run / "export" / name)

    # LDpred2 reads exactly the snv_indel arm's records and scores exactly the model's held-out people.
    monkeypatch.setattr(aou_ldpred2, "workspace_bindings", lambda: bindings)
    monkeypatch.setattr(aou_compare, "workspace_bindings", lambda: bindings)
    monkeypatch.setattr(aou_ldpred2, "detect_compute_budget", lambda: BUDGET)
    monkeypatch.setattr(aou_compare, "detect_compute_budget", lambda: BUDGET)
    read: list[np.ndarray] = []

    def posterior_effects(train, rows):
        read.append(train.records.copy())
        return train.codes(rows).astype(np.float64).mean(axis=1) / 127.0, np.random.default_rng(3).normal(size=rows.shape[0])

    monkeypatch.setattr(aou_ldpred2.ldpred2_auto_allvar, "posterior_effects", posterior_effects)
    # The synthetic disease's held-out cases are too few to show, which would leave LDpred2 unfitted: fit it anyway here.
    monkeypatch.setattr(aou_ldpred2, "_shown", lambda count, cases: count > 0)
    scores_path = tmp_path / "ldpred2_held_out.npz"
    assert aou_ldpred2.main(["--config", str(config_path), "--out", str(tmp_path / "ldpred2.jsonl"), "--scores", str(scores_path)]) == 0
    [records] = read
    np.testing.assert_array_equal(records, np.flatnonzero(short_fitted))
    cohort = np.load(run / "cohort" / "cohort.npz")
    with np.load(scores_path) as saved:
        assert saved["research_ids"].tolist() == cohort["research_ids"].tolist() and saved["trait_names"].tolist() == ["alzheimers_disease"]
        # One region, unnamed: the run's own window.
        assert saved["regions"].tolist() == [""] and saved["variants"].tolist() == [records.shape[0]]
        np.testing.assert_array_equal(np.isfinite(saved["held_out_score"][0, :, 0]), cohort["held_out"][:, 0] & np.isfinite(cohort["targets"][:, 0]))
    [line] = (tmp_path / "ldpred2.jsonl").read_text().splitlines()
    assert json.loads(line)["variants"] == records.shape[0]
    compared = tmp_path / "compare.tsv"
    assert aou_compare.main(["--config", str(config_path), "--scores", str(scores_path), "--out", str(compared)]) == 0
    with compared.open() as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    assert {row["region"] for row in rows} == {""} and [(row["score"], row["metric"]) for row in rows] == [
        (score, metric) for metric in ("pgs_partial_r2", "pgs_auc") for score in ("snv_indel", "with_sv", "ldpred2", "with_sv - snv_indel", "snv_indel - ldpred2", "with_sv - ldpred2")
    ]
    # The synthetic disease's few held-out cases and controls are never shown, nor any value read on them.
    check_exportable(compared)
    assert all(row["estimate"] == "" for row in rows)


def test_report_cells_hide_the_levels_from_which_a_small_count_would_follow() -> None:
    """Each family of cells partitions the held-out rows, so a level's small count is hidden with the smallest others
    until the hidden ones hold at least MINIMUM_REPORTED_PARTICIPANTS rows, cases and controls."""
    from sv_pgs.workspace_pipeline import _HeldOutCell, _partition_hidden

    # (ancestry, rows, cases) and (half, rows): every level's rows and cases are its first ones.
    ancestry = (("eur", 300, 40), ("afr", 100, 25), ("amr", 30, 10), ("eas", 25, 0))
    halves = (("A", 440), ("B", 15))
    total = sum(rows for _level, rows, _cases in ancestry)
    targets = np.zeros(total)
    starts = np.cumsum([0] + [rows for _level, rows, _cases in ancestry])
    members = {}
    for (level, rows, cases), start in zip(ancestry, starts):
        targets[start : start + cases] = 1.0
        members[level] = np.arange(start, start + rows)
    cells = [("all", "all", _HeldOutCell((np.arange(total),)))]
    cells += [("ancestry", level, _HeldOutCell((rows,))) for level, rows in members.items()]
    cells += [("half", "A", _HeldOutCell((np.arange(440),))), ("half", "B", _HeldOutCell((np.arange(440, total),)))]
    # Cases: amr's 10 hides afr's 25 (the smallest other) with it; controls: amr's 20 hides eas's 25; rows: half B's 15
    # hides half A's.
    assert _partition_hidden(cells, targets, True) == {
        ("ancestry", "amr"): "cases", ("ancestry", "afr"): "cases", ("ancestry", "eas"): "cases", ("half", "A"): "rows", ("half", "B"): "rows",
    }
    assert _partition_hidden(cells, targets, False) == {("half", "A"): "rows", ("half", "B"): "rows"}
    # A total itself suppressed hides every level that is not empty.
    small = [("all", "all", _HeldOutCell((np.arange(15),))), ("ancestry", "eur", _HeldOutCell((np.arange(15),))), ("ancestry", "afr", _HeldOutCell((np.arange(0),)))]
    assert _partition_hidden(small, np.zeros(15), False) == {("ancestry", "eur"): "rows"}



def test_fit_regions_name_windows_each_fitted_as_a_run_of_its_own_on_the_runs_shared_steps(workspace) -> None:
    synthetic, _client = workspace
    regions = [{"name": "apoe", "chromosome": 22, "start": 1500, "end": 6500}, {"name": "lpa_kiv2", "chromosome": 21, "start": 1, "end": 9000}]
    config = WorkspaceConfig.from_mapping({**synthetic.raw, "fit_regions": regions})
    assert config.fit_regions == (("apoe", (22, 1500, 6500)), ("lpa_kiv2", (21, 1, 9000))) and config.fit_region is None
    runs = dict(config.region_runs())
    assert list(runs) == ["apoe", "lpa_kiv2"] and WorkspaceConfig.from_mapping(synthetic.raw).region_runs() == ()
    apoe = runs["apoe"]
    assert apoe.run_directory == config.run_directory / workspace_pipeline.REGIONS_DIRECTORY / "apoe"
    assert apoe.marker_directory == config.marker_directory / workspace_pipeline.REGIONS_DIRECTORY / "apoe"
    assert apoe.fit_region == (22, 1500, 6500) and apoe.fit_regions is None
    # A region's window keys its fit alone (the steps before it are the run's, which a region never makes again).
    steps = {step.name: step for step in workspace_pipeline._STEPS}
    assert all(steps[name].inputs(apoe) == steps[name].inputs(config) for name in ("cohort", "measurement", "direct_sv"))
    assert steps["fit"].inputs(apoe) != steps["fit"].inputs(runs["lpa_kiv2"])
    window = {key: value for key, value in regions[0].items() if key != "name"}
    for bad, match in (
        ([], "non-empty list"), (regions[0], "non-empty list"), (["apoe"], "non-empty list"), ([regions[0], regions[0]], "distinct name"),
        ([{**regions[0], "name": "APOE"}], "distinct name"), ([{**regions[0], "name": "a-b"}], "distinct name"),
        ([{**regions[0], "name": ""}], "distinct name"), ([window], "distinct name"), ([{**regions[0], "chromosome": 19}], r"fit_regions\[0\]"),
        ([regions[0], {**regions[1], "start": 0}], r"fit_regions\[1\]"), ([{**regions[0], "strand": "+"}], r"fit_regions\[0\]"),
    ):
        with pytest.raises(ValueError, match=match):
            WorkspaceConfig.from_mapping({**synthetic.raw, "fit_regions": bad})
    with pytest.raises(ValueError, match="never both"):
        WorkspaceConfig.from_mapping({**synthetic.raw, "fit_regions": regions, "fit_region": window})


def test_a_run_of_regions_fits_each_window_on_the_shared_steps_mirrors_each_once_and_compares_each_with_ldpred2(
    workspace, monkeypatch, tmp_path: Path
) -> None:
    """Two named windows on one store: the steps before the fit made once, each region's fit and later steps in its own
    directory on its window alone; the mirror carries each region's own steps and the shared ones once, and a fresh disk
    restored from it fits nothing again; LDpred2 and the comparison run region by region."""
    from benchmarks.bench_real import aou_compare, aou_ldpred2
    from sv_pgs.run_mirror import DirectoryRemote, push, restore

    synthetic, client = workspace
    windows = {"left": (21, 1, 4500), "right": (22, 2500, 9500)}
    raw = {**synthetic.raw, "fit_regions": [{"name": name, **dict(zip(("chromosome", "start", "end"), window))} for name, window in windows.items()]}
    config_path = tmp_path / "run_config.json"
    config_path.write_text(json.dumps(raw))
    config = WorkspaceConfig.read(config_path)
    recorder = _Recorder()
    bindings = _artifact_bindings(recorder, client)
    assert not workspace_pipeline.completed_through(config, "export")
    summaries = run_pipeline(config, bindings, BUDGET)
    run, shared = config.run_directory, STEP_NAMES[:STEP_NAMES.index("fit")]
    own = STEP_NAMES[STEP_NAMES.index("fit"):]
    assert list(summaries) == [*shared, *(f"{workspace_pipeline.REGIONS_DIRECTORY}/{name}/{step}" for name in windows for step in own)]
    assert not any((run / step).exists() for step in own)
    assert workspace_pipeline.completed_through(config, "export") and workspace_pipeline.completed_through(config, "direct_sv")
    keys = {}
    for name in windows:
        region = run / workspace_pipeline.REGIONS_DIRECTORY / name
        assert all((region / step).is_symlink() and (region / step).resolve() == (run / step).resolve() for step in shared)
        assert all((region / step / COMPLETE_FILE).is_file() and not (region / step).is_symlink() for step in own)
        keys[name] = json.loads((region / "fit" / COMPLETE_FILE).read_text())["key"]
        for table in raw["export"]:
            check_exportable(region / "export" / table)
    assert keys["left"] != keys["right"]
    # Each region's two arms fit its window alone, the regions in the config's order.
    assert len(recorder.fit_calls) == len(ARMS) * len(windows)
    for position, window in enumerate(windows.values()):
        calls = recorder.fit_calls[len(ARMS) * position:len(ARMS) * (position + 1)]
        inside = workspace_pipeline.region_records(calls[0].store.variant_table, window)
        assert all(np.isfinite(call.log_variance_offset).any() and not np.any(np.isfinite(call.log_variance_offset) & ~inside) for call in calls)

    mirror = DirectoryRemote(tmp_path / "mirror")
    push(run, mirror)
    held = mirror.sizes("")
    for name in windows:
        prefix = f"{workspace_pipeline.REGIONS_DIRECTORY}/{name}/"
        assert all(f"{prefix}{step}/{COMPLETE_FILE}" in held for step in own)
        assert not any(entry.startswith(f"{prefix}{step}") for entry in held for step in shared)
    assert all(f"{step}/{COMPLETE_FILE}" in held for step in shared)
    restored = dataclasses.replace(config, run_directory=tmp_path / "restored")
    restore(restored.run_directory, mirror)
    resumed = _Recorder(fits_before_failing=0)
    run_pipeline(restored, _artifact_bindings(resumed, client), BUDGET)
    assert not resumed.fit_calls
    for name in windows:
        relative = Path(workspace_pipeline.REGIONS_DIRECTORY) / name / "report" / "accuracy.tsv"
        assert (restored.run_directory / relative).read_bytes() == (run / relative).read_bytes()

    # LDpred2 region by region, each on its region's snv_indel records, then the comparison of each region's arms with it.
    for module in (aou_ldpred2, aou_compare):
        monkeypatch.setattr(module, "workspace_bindings", lambda: bindings)
        monkeypatch.setattr(module, "detect_compute_budget", lambda: BUDGET)
    read: list[np.ndarray] = []

    def posterior_effects(train, rows):
        read.append(train.records.copy())
        return train.codes(rows).astype(np.float64).mean(axis=1) / 127.0, np.random.default_rng(3).normal(size=rows.shape[0])

    monkeypatch.setattr(aou_ldpred2.ldpred2_auto_allvar, "posterior_effects", posterior_effects)
    monkeypatch.setattr(aou_ldpred2, "_shown", lambda count, cases: count > 0)
    scores_path, ldpred2_path = tmp_path / "ldpred2_held_out.npz", tmp_path / "ldpred2.jsonl"
    assert aou_ldpred2.main(["--config", str(config_path), "--out", str(ldpred2_path), "--scores", str(scores_path)]) == 0
    short = [np.flatnonzero(np.isfinite(recorder.fit_calls[len(ARMS) * position].log_variance_offset)) for position in range(len(windows))]
    assert read and all(any(np.array_equal(records, region) for region in short) for records in read)
    lines = [json.loads(line) for line in ldpred2_path.read_text().splitlines()]
    traits = [*config.diseases, *config.traits]
    assert [(line["region"], line["trait"]) for line in lines] == [(name, trait) for name in windows for trait in traits]
    assert [line["variants"] for line in lines] == [region.shape[0] for region in short for _trait in traits]
    with np.load(scores_path) as saved:
        assert saved["regions"].tolist() == list(windows) and saved["held_out_score"].shape[0] == len(windows)
    compared = tmp_path / "compare.tsv"
    assert aou_compare.main(["--config", str(config_path), "--scores", str(scores_path), "--out", str(compared)]) == 0
    with compared.open() as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    assert list(dict.fromkeys(row["region"] for row in rows)) == list(windows)
    check_exportable(compared)


def _single_model_batches(model_bytes, host_bytes):
    """Give the dry run several production-shaped score batches without reducing the scorer's working memory."""
    return [range(position, position + 1) for position in range(len(model_bytes))]


def _postfit_outputs(directory: Path) -> dict[str, bytes | dict[str, np.ndarray]]:
    """The step's public outputs, comparing gzip's table bytes, not its wall-clock header, and NPZ's arrays."""
    outputs = {}
    for path in directory.iterdir():
        if not path.is_file() or path.name in (COMPLETE_FILE, workspace_pipeline.KEY_FILE):
            continue
        if path.suffix == ".npz":
            with np.load(path, allow_pickle=False) as arrays:
                outputs[path.name] = {name: arrays[name] for name in arrays.files}
        else:
            outputs[path.name] = gzip.decompress(path.read_bytes()) if path.suffix == ".gz" else path.read_bytes()
    return outputs


def _assert_postfit_outputs(directory: Path, expected) -> None:
    actual = _postfit_outputs(directory)
    assert actual.keys() == expected.keys()
    for name, values in expected.items():
        if isinstance(values, dict):
            assert actual[name].keys() == values.keys()
            for column, array in values.items():
                np.testing.assert_array_equal(actual[name][column], array)
        else:
            assert actual[name] == values


@pytest.mark.parametrize("step", ["score", "effects"])
def test_postfit_resumes_after_a_mirrored_batch_and_keeps_every_output(workspace, monkeypatch, tmp_path, step) -> None:
    """A completed batch survives the loss of the VM; a failed batch is redone, and reassembly is exactly the same
    as an uninterrupted pass, including held-out/calibrated scores, structural parts and report tables."""
    from sv_pgs.run_mirror import DirectoryRemote, push, restore

    synthetic, client = workspace
    config = WorkspaceConfig.from_mapping({**synthetic.raw, "calibration_fraction": .3, "annotated_prior": False})
    recorder = _Recorder()
    bindings = _bindings(recorder, client)
    monkeypatch.setattr(workspace_pipeline, "score_batches", _single_model_batches)
    run_pipeline(config, bindings, BUDGET, through="report")
    run = config.run_directory
    expected = _postfit_outputs(run / step)
    report = _postfit_outputs(run / "report")
    total = len(list((run / step / "batches").glob("*/complete.json")))
    assert total > 1
    fit_time = _completed_at(run, "fit")
    for name in STEP_NAMES[STEP_NAMES.index(step):]:
        if (run / name).exists():
            shutil.rmtree(run / name)
    calls = []
    original = bindings.predict if step == "score" else workspace_pipeline.variant_effects

    def interrupted(*args, **kwargs):
        calls.append(True)
        if len(calls) == 2:
            raise RuntimeError("preempted batch")
        return original(*args, **kwargs)

    if step == "score":
        interrupted_bindings = dataclasses.replace(bindings, predict=interrupted)
    else:
        interrupted_bindings = bindings
        monkeypatch.setattr(workspace_pipeline, "variant_effects", interrupted)
    with pytest.raises(RuntimeError, match="preempted batch"):
        run_pipeline(config, interrupted_bindings, BUDGET, through=step)
    assert _completed_at(run, "fit") == fit_time
    partial = run / f"{step}.partial"
    assert len(list((partial / "batches").glob("*/complete.json"))) == 1
    remote = DirectoryRemote(tmp_path / "postfit-mirror")
    push(run, remote)
    shutil.rmtree(run)
    restore(run, remote)
    assert len(list((partial / "batches").glob("*/complete.json"))) == 1
    calls.clear()

    def resumed(*args, **kwargs):
        calls.append(True)
        return original(*args, **kwargs)

    if step == "score":
        bindings = dataclasses.replace(bindings, predict=resumed)
    else:
        monkeypatch.setattr(workspace_pipeline, "variant_effects", resumed)
    fit_calls = len(recorder.fit_calls)
    run_pipeline(config, bindings, BUDGET, through="report")
    assert len(recorder.fit_calls) == fit_calls
    assert len(calls) == total - 1
    _assert_postfit_outputs(run / step, expected)
    _assert_postfit_outputs(run / "report", report)

    # All batches were committed but final assembly was interrupted. No model should be opened on the next pass.
    partial = run / f"{step}.partial"
    (run / step).rename(partial)
    (partial / COMPLETE_FILE).unlink()
    for path in partial.iterdir():
        if path.is_file() and path.name != workspace_pipeline.KEY_FILE:
            path.unlink()
            path.write_bytes(b"interrupted assembly")

    def no_model(path):
        pytest.fail(f"a completed post-fit batch loaded {path}")

    run_pipeline(config, dataclasses.replace(bindings, load_model=no_model), BUDGET, through=step)
    _assert_postfit_outputs(run / step, expected)

    # A same-size damaged payload is a single batch to redo, even with every completion record still present.
    (run / step).rename(partial)
    (partial / COMPLETE_FILE).unlink()
    record = next((partial / "batches").glob("*/complete.json"))
    payload = record.parent / ("predictions.npz" if step == "score" else next(record.parent.glob("*.tsv.gz")).name)
    damaged = bytearray(payload.read_bytes())
    damaged[0] ^= 1
    payload.write_bytes(damaged)
    calls.clear()
    run_pipeline(config, bindings, BUDGET, through=step)
    assert len(calls) == 1
    _assert_postfit_outputs(run / step, expected)


def test_postfit_checkpoint_code_is_keyed_only_after_the_fit() -> None:
    """Changing the batch protocol invalidates score/effects partials without changing phenotypes, measurement or fit
    keys: the helper is a code dependency of those two steps and never an engine import."""
    from sv_pgs import postfit_checkpoint

    for step in workspace_pipeline._STEPS:
        assert (postfit_checkpoint in step.code) == (step.name in ("score", "effects"))
    assert postfit_checkpoint not in workspace_pipeline.module_closure(workspace_pipeline.fit_model)


def _artifact_bindings(recorder: _Recorder, client: _FakeBigQuery, *, flagged: str | None = None) -> WorkspaceBindings:
    """The synthetic fitter with production artifacts and prediction, so a dry run crosses the actual saved-model
    boundary. The certificate can miss convergence or refuse a model; the scorer's numerical parameters stay finite.
    Neither a failed nor an uncertified fit is simulated by corrupting an array that the artifact cannot load."""
    from sv_pgs import artifact, fit_model
    from sv_pgs.scale_mixture_ep import MixtureHyperparameters

    def fit(request: FitRequest):
        fitted = recorder.fit(request)
        if flagged == "uncertified":
            fitted.certificate["outer_criterion_met"][0] = False
        elif flagged == "quarantined":
            fitted.refusals = ("model 0: the fit failed: synthetic refusal",)
        problem = artifact.problem_digest(
            research_ids=request.research_ids, store_columns=request.store_columns, model_names=request.model_names,
            trait_types=request.trait_types, covariate_names=request.covariate_names, covariate_columns=request.covariate_columns,
            covariates=request.covariates, targets=request.targets, training=request.training, draw_count=fit_model.DRAW_COUNT, seed=request.seed,
        )
        return artifact.FittedModel(
            model_names=fitted.model_names, covariate_names=fitted.covariate_names, covariate_columns=request.covariate_columns,
            scoring=fitted.scoring, noise_variance=fitted.noise_variance, certificate=fitted.certificate, refusals=fitted.refusals,
            hyperparameters=tuple(MixtureHyperparameters(np.zeros(1), np.zeros(1)) for _ in fitted.scoring), fit_counts={},
            provenance=artifact.Provenance(
                code_digest="synthetic fitter", store_digest=artifact.store_digest(request.store.root), sites_digest=artifact.sites_digest(request.store.root),
                cohort_digest=artifact.cohort_digest(request.research_ids), offset_digest=artifact.offset_digest(request.log_variance_offset),
                problem_digest=problem, prior_digest="synthetic prior",
            ),
        )

    return dataclasses.replace(_bindings(recorder, client), fit=fit, save_model=artifact.save_model, load_model=artifact.load_model, predict=artifact.predict_models)


@pytest.mark.parametrize("likelihood,cross_fit,flagged", [
    ("gaussian_working", True, "uncertified"),
    ("bernoulli", False, "uncertified"),
    ("gaussian_working", False, "quarantined"),
])
def test_two_runs_merge_their_finished_arms_and_reach_the_report_without_fitting(workspace, likelihood, cross_fit, flagged) -> None:
    """The expensive handoff: independent arm runs with one fit key, their mirrors merged by copying a whole arm,
    then restored to a fresh disk. Real artifact loading and prediction take the result through effects and report,
    including a reordered saved fold axis, a binary and a quantitative trait, and an uncertified or refused arm."""
    from sv_pgs import artifact
    from sv_pgs.run_mirror import DirectoryRemote, push, restore

    synthetic, client = workspace
    config = WorkspaceConfig.from_mapping({**synthetic.raw, "disease_likelihood": likelihood, "cross_fit": cross_fit,
                                           "calibration_fraction": 0.25 if likelihood == "bernoulli" else 0.0})
    first, second = _Recorder(), _Recorder()
    run_pipeline(config, _artifact_bindings(first, client), BUDGET, through="fit", fit_arms=(ARMS[0],))
    mirror_a = config.run_directory.with_name("mirror-a")
    push(config.run_directory, DirectoryRemote(mirror_a))
    other = dataclasses.replace(config, run_directory=config.run_directory.with_name("run-b"))
    restore(other.run_directory, DirectoryRemote(mirror_a))
    # Each job starts with the same upstream steps but keeps only its own arm's output.
    shutil.rmtree(other.run_directory / "fit.partial")
    run_pipeline(other, _artifact_bindings(second, client, flagged=flagged), BUDGET, through="fit", fit_arms=(ARMS[1],))
    partial_a, partial_b = config.run_directory / "fit.partial", other.run_directory / "fit.partial"
    assert (partial_a / workspace_pipeline.KEY_FILE).read_bytes() == (partial_b / workspace_pipeline.KEY_FILE).read_bytes()
    assert len(first.fit_calls) == len(second.fit_calls) == 1
    trait = config.traits[0]
    folder = partial_b / "models" / ARMS[1] / trait
    fitted = artifact.load_model(folder)
    shutil.rmtree(folder)
    artifact.save_model(folder, artifact.model_subset(fitted, tuple(reversed(fitted.model_names))))
    mirror_b = config.run_directory.with_name("mirror-b")
    push(other.run_directory, DirectoryRemote(mirror_b))
    shutil.copytree(mirror_b / "fit.partial" / "models" / ARMS[1], mirror_a / "fit.partial" / "models" / ARMS[1])
    shutil.rmtree(config.run_directory)
    restore(config.run_directory, DirectoryRemote(mirror_a))
    resumed = _Recorder(fits_before_failing=0)
    summaries = run_pipeline(config, _artifact_bindings(resumed, client), BUDGET)
    assert not resumed.fit_calls and set(summaries) == set(STEP_NAMES)
    assert all((config.run_directory / step / COMPLETE_FILE).is_file() for step in ("fit", "score", "effects", "report", "export"))
    disease = config.diseases[0]
    assert summaries["score"][flagged][disease]
    assert disease not in summaries["score"]["quarantined" if flagged == "uncertified" else "uncertified"]
    cohort = workspace_pipeline._read_cohort(workspace_pipeline._Run(config, _artifact_bindings(resumed, client), BUDGET, {}))
    run = workspace_pipeline._Run(config, _artifact_bindings(resumed, client), BUDGET, {})
    with DosageStore.open(workspace_pipeline._final_store(run)) as store:
        signed = store.read_codes(0, store.n_variants, cohort.store_columns).astype(np.float64) - 127.0
    with np.load(config.run_directory / "score" / "predictions.npz") as scores:
        for arm_index, arm in enumerate(ARMS):
            for name in cohort.trait_names:
                if name == disease and flagged == "quarantined":
                    continue
                model = artifact.load_model(config.run_directory / "fit" / "models" / arm / name)
                for model_name, scoring in zip(model.model_names, model.scoring):
                    column = cohort.model_names.index(model_name)
                    expected = ((signed[scoring.store_rows] - scoring.signed_means[:, None]) / scoring.signed_scales[:, None]).T @ scoring.coefficients
                    np.testing.assert_allclose(scores["genetic_mean"][arm_index, :, column], expected, rtol=1e-12, atol=1e-12)
        for trait_index, name in enumerate(cohort.trait_names):
            for model in cohort.trait_models(trait_index):
                held = cohort.held_out[:, model]
                if name == disease and flagged == "quarantined":
                    assert np.all(np.isnan(scores["genetic_mean"][:, :, model]))
                else:
                    np.testing.assert_array_equal(scores["held_out_genetic"][:, held, trait_index], scores["genetic_mean"][:, held, model])
            assert set(cohort.evaluated_folds(trait_index, config.fold_count)) == (set(range(config.fold_count)) if cross_fit else {0})
    with (config.run_directory / "report" / "accuracy.tsv").open() as handle:
        accuracy = list(csv.DictReader(handle, delimiter="\t"))
    assert any(row["trait"] == trait and row["metric"] == "pgs_partial_r2" and row["value"] for row in accuracy)
    assert (disease in {row["trait"] for row in accuracy}) == (flagged == "uncertified")
    assert (config.run_directory / "effects" / f"{trait}.{ARMS[1]}.tsv.gz").is_file()
    assert bool(list((config.run_directory / "effects").glob(f"{disease}.*.tsv.gz"))) == (flagged == "uncertified")
    # Repeating the complete handoff is also a no-fit operation, including a mirror restore of the finalized run.
    push(config.run_directory, DirectoryRemote(mirror_a))
    shutil.rmtree(config.run_directory)
    restore(config.run_directory, DirectoryRemote(mirror_a))
    run_pipeline(config, _artifact_bindings(resumed, client), BUDGET)
    assert not resumed.fit_calls


@pytest.mark.parametrize("damage", ["missing_directory", "missing_trait", "wrong_key", "wrong_names", "wrong_covariates", "wrong_record", "truncated_file"])
def test_a_bad_arm_handoff_stops_before_any_fitter_is_called(workspace, damage) -> None:
    synthetic, client = workspace
    config = WorkspaceConfig.from_mapping(synthetic.raw)
    run_pipeline(config, _bindings(_Recorder(), client), BUDGET, through="fit", fit_arms=(ARMS[0],))
    root = config.run_directory / "fit.partial" / "models" / ARMS[0]
    folder, record = root / config.diseases[0], root / (config.diseases[0] + ".json")
    if damage in ("missing_directory", "missing_trait"):
        shutil.rmtree(folder)
        if damage == "missing_trait":
            record.unlink()
    elif damage in ("wrong_key", "wrong_record", "truncated_file"):
        content = json.loads(record.read_text())
        if damage == "wrong_key":
            content["fit_key"] = "other run"
        elif damage == "wrong_record":
            content["models"] = ["other_trait/fold0"]
        else:
            content["model_files"] = {"model.pickle": (folder / "model.pickle").stat().st_size + 1}
        record.write_text(json.dumps(content))
    else:
        model = _load_model(folder)
        if damage == "wrong_names":
            model.model_names = tuple("other/" + name for name in model.model_names)
        else:
            model.covariate_names = tuple(reversed(model.covariate_names))
        (folder / "model.pickle").write_bytes(pickle.dumps(model))
    recorder = _Recorder(fits_before_failing=0)
    with pytest.raises(ValueError):
        run_pipeline(config, _bindings(recorder, client), BUDGET, through="fit")
    assert not recorder.fit_calls and not (config.run_directory / "fit").exists()


@pytest.mark.parametrize("all_traits", [False, True])
def test_nonfinite_predictions_quarantine_both_arms_and_finish_the_remaining_tables(workspace, all_traits) -> None:
    synthetic, client = workspace
    config = WorkspaceConfig.from_mapping(synthetic.raw)
    bindings = _bindings(_Recorder(), client)
    calls = []

    def predict(models, store, sample_indices, covariates, budget):
        calls.append(tuple(name for model in models for name in model.model_names))
        prediction = _predict(models, store, sample_indices, covariates, budget)
        for index, name in enumerate(calls[-1]):
            if all_traits or name.startswith(config.diseases[0] + "/"):
                prediction.linear_predictor[:, index] = np.nan
        return prediction

    summary = run_pipeline(config, dataclasses.replace(bindings, predict=predict), BUDGET)
    expected = set((*config.diseases, *config.traits) if all_traits else config.diseases)
    assert set(summary["score"]["quarantined"]) == expected and not summary["score"]["uncertified"]
    assert len(calls) > 1 and (config.run_directory / "report" / COMPLETE_FILE).is_file()
    with (config.run_directory / "report" / "accuracy.tsv").open() as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    assert not ({row["trait"] for row in rows} & expected)
    with np.load(config.run_directory / "score" / "predictions.npz") as scores:
        assert np.all(np.isnan(scores["held_out_genetic"][:, :, 0]))
        if not all_traits:
            assert np.any(np.isfinite(scores["held_out_genetic"][:, :, 1]))


def test_empty_held_out_folds_keep_named_report_rows_and_do_not_enter_the_panel(workspace) -> None:
    synthetic, client = workspace
    config = WorkspaceConfig.from_mapping(synthetic.raw)
    bindings = _bindings(_Recorder(), client)
    run_pipeline(config, bindings, BUDGET, through="score")
    run = workspace_pipeline._Run(config, bindings, BUDGET, {})
    cohort = workspace_pipeline._read_cohort(run)
    cohort.held_out[:] = False
    # These are report inputs alone: no retraining, and no changes to a keyed cohort on disk.
    from unittest.mock import patch
    directory = config.run_directory / "empty-report"
    directory.mkdir()
    panel = workspace_pipeline.PanelInputs(0, tuple((name, 1.0) for name in cohort.trait_names))
    run.config = dataclasses.replace(config, panel=panel)
    with patch.object(workspace_pipeline, "_read_cohort", return_value=cohort):
        summary = workspace_pipeline._report_step(run, directory)
    assert summary["panel"]["left_out"] == list(cohort.trait_names)
    with (directory / "accuracy.tsv").open() as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    assert rows and all(row["n"] == "0" and row["metric"] and not row["value"] for row in rows)
    assert all(cell.count == 0 for _, _, cell in workspace_pipeline._cells(cohort, 0, config.fold_count))


def test_prediction_workspace_excludes_loaded_models_shared_arrays_and_results() -> None:
    values = np.zeros(128, dtype=np.float64)
    assert workspace_pipeline._array_bytes([values, values[::2], {"again": values}]) == values.nbytes
    cohort = SimpleNamespace(store_columns=np.arange(50), covariates=np.zeros((50, 3)))
    model = SimpleNamespace(scoring=[SimpleNamespace(store_rows=np.arange(100), draw_count=64)])
    budget = workspace_pipeline._prediction_budget(BUDGET, cohort, [model], {"output": values}, values)
    assert 0 < budget.host_bytes < BUDGET.host_bytes // 2
    with pytest.raises(MemoryError, match="no host memory"):
        workspace_pipeline._prediction_budget(dataclasses.replace(BUDGET, host_bytes=values.nbytes), cohort, [model], {"output": values}, values)


@pytest.mark.parametrize("damage", ["other_arm", "other_problem"])
def test_saved_provenance_rejects_an_arm_or_training_problem_swap(workspace, damage) -> None:
    from sv_pgs import artifact

    synthetic, client = workspace
    config = WorkspaceConfig.from_mapping(synthetic.raw)
    bindings = _artifact_bindings(_Recorder(), client)
    run_pipeline(config, bindings, BUDGET, through="fit", fit_arms=(ARMS[0],))
    root = config.run_directory / "fit.partial" / "models"
    if damage == "other_arm":
        shutil.copytree(root / ARMS[0], root / ARMS[1])
    else:
        folder = root / ARMS[0] / config.diseases[0]
        fitted = artifact.load_model(folder)
        shutil.rmtree(folder)
        artifact.save_model(folder, dataclasses.replace(fitted, provenance=dataclasses.replace(fitted.provenance, problem_digest="other training rows")))
    resumed = _Recorder(fits_before_failing=0)
    with pytest.raises(ValueError, match="different arm offsets|different cohort or training problem"):
        run_pipeline(config, _artifact_bindings(resumed, client), BUDGET, through="fit")
    assert not resumed.fit_calls


def test_legacy_records_are_checked_and_missing_verdicts_are_recovered_from_the_model(workspace) -> None:
    synthetic, client = workspace
    config = WorkspaceConfig.from_mapping(synthetic.raw)
    run_pipeline(config, _bindings(_Recorder(), client), BUDGET, through="fit", fit_arms=(ARMS[0],))
    root = config.run_directory / "fit.partial" / "models" / ARMS[0]
    for record in root.glob("*.json"):
        content = json.loads(record.read_text())
        content.pop("fit_key")
        content.pop("acceptance")
        record.write_text(json.dumps(content))
    recorder = _Recorder()
    run_pipeline(config, _bindings(recorder, client), BUDGET, through="report")
    assert len(recorder.fit_calls) == 1
    assert (config.run_directory / "report" / COMPLETE_FILE).is_file()


def test_an_entirely_unscorable_fit_finishes_with_only_quarantine_rows(workspace) -> None:
    synthetic, client = workspace
    config = WorkspaceConfig.from_mapping(synthetic.raw)
    recorder = _Recorder()
    bindings = _bindings(recorder, client)

    def fit(request):
        model = recorder.fit(request)
        model.refusals = tuple(f"model {index}: the fit failed: synthetic refusal" for index in range(len(model.model_names)))
        return model

    def unexpected_prediction(*args):
        pytest.fail("no quarantined trait is passed to prediction")

    summary = run_pipeline(config, dataclasses.replace(bindings, fit=fit, predict=unexpected_prediction), BUDGET)
    assert set(summary["score"]["quarantined"]) == set((*config.diseases, *config.traits))
    assert summary["score"]["prediction_batches"] == 0 and summary["score"]["models"] == 0
    assert summary["effects"]["variant_tables"].startswith("0 ")
    assert (config.run_directory / "report" / "accuracy.tsv").read_text().splitlines() == ["\t".join(workspace_pipeline.ACCURACY_COLUMNS)]
    assert (config.run_directory / "export" / COMPLETE_FILE).is_file()
