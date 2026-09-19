"""The cohort every fit shares: covariates, multi-trait targets and kinship-grouped folds.

All traits and arms share one covariate matrix C (intercept, person-level covariates,
indicators, genetic PCs, the pipeline-half and cohort indicators) so they share one
genotype pass. Targets are one column per trait, NaN where a participant has no value
for that trait. Folds are assigned once, before any phenotype is built: relatives stay
inside one fold, and every stratum (pipeline half x ancestry) is spread evenly over the
folds (docs/design/EVALUATION.md, "Folds").
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
import json
import logging
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from sv_pgs._typing import BoolArray, F64Array, I64Array
from sv_pgs.all_of_us import MINIMUM_REPORTED_PARTICIPANTS
from sv_pgs.sample_crosswalk import SampleCrosswalk
from sv_pgs.sample_ids import ResearchId, SequencingId

LOGGER = logging.getLogger(__name__)

SECOND_DEGREE_KINSHIP = 2.0 ** -3.5
"""KING's lower kinship bound for second-degree relatives (Manichaikul et al. 2010,
Bioinformatics 26:2867, Table 1). Pairs above it, duplicates included, share a fold."""

DUPLICATE_KINSHIP = 2.0 ** -1.5
"""KING's lower kinship bound for duplicate samples and monozygotic twins (Manichaikul et
al. 2010, Table 1). Such a pair across the two halves is one genome measured twice."""

LONG_READ_SOURCE = "long_read"
IMPUTED_SOURCE = "imputed"


@dataclass(frozen=True, slots=True)
class CohortRows:
    """One genotype row per person over the union of the long-read truth half and the imputed half.

    A person with a truth row and an imputed row (the crosswalk maps the imputed
    sample to a truth research ID, or a KING pair across the halves is at
    duplicate/MZ level) keeps the truth row: the imputed row is a leave-in imputation
    of a panel genome. Everyone else keeps the one row they have. ``kinship_pairs``
    are the input pairs with every sample in the research-ID namespace, for
    kinship_components.
    """

    research_ids: tuple[ResearchId, ...]
    genotype_source: tuple[str, ...]
    kinship_pairs: tuple[tuple[ResearchId, ResearchId, float], ...]
    imputed_rows_sharing_a_research_id: int
    imputed_rows_duplicating_a_truth_genome: int


def resolve_cohort_rows(
    truth_research_ids: Sequence[ResearchId],
    imputed_sequencing_ids: Sequence[SequencingId],
    crosswalk: SampleCrosswalk,
    kinship_pairs: Sequence[tuple[ResearchId | SequencingId, ResearchId | SequencingId, float]],
) -> CohortRows:
    """Resolve the two halves into one row per person.

    The truth half is named by research ID and the imputed half by sequencing ID;
    the two are matched only through the crosswalk, never by name.
    ``kinship_pairs`` are KING pairs over the union of both halves, each sample
    typed by the half that named it. A repeat within either half, or an imputed
    sample (in the half or in a pair) without a crosswalk row, is an error. The
    number of imputed rows given up is logged, with counts of 1 to 20 suppressed.
    """
    if not all(isinstance(research_id, ResearchId) for research_id in truth_research_ids):
        raise TypeError("the long-read truth half is named by ResearchId.")
    if not all(isinstance(sequencing_id, SequencingId) for sequencing_id in imputed_sequencing_ids):
        raise TypeError("the imputed half is named by SequencingId.")
    if len(set(truth_research_ids)) != len(truth_research_ids):
        raise ValueError("the long-read truth half repeats a research ID.")
    if len(set(imputed_sequencing_ids)) != len(imputed_sequencing_ids):
        raise ValueError("the imputed half repeats a sequencing ID.")
    research_of_sequencing = {
        SequencingId(sequencing_id): ResearchId(research_id)
        for sequencing_id, research_id in zip(crosswalk.sequencing_ids, crosswalk.research_ids, strict=True)
    }

    def as_research_id(sample: ResearchId | SequencingId) -> ResearchId:
        if isinstance(sample, ResearchId):
            return sample
        if not isinstance(sample, SequencingId):
            raise TypeError(f"a kinship pair names a {type(sample).__name__}, not a typed sample ID.")
        if sample not in research_of_sequencing:
            raise ValueError("a kinship pair names an imputed sample with no crosswalk row.")
        return research_of_sequencing[sample]

    unmapped = [sequencing_id for sequencing_id in imputed_sequencing_ids if sequencing_id not in research_of_sequencing]
    if unmapped:
        raise ValueError(f"{len(unmapped)} imputed samples have no crosswalk row.")
    imputed_research_ids = [research_of_sequencing[sequencing_id] for sequencing_id in imputed_sequencing_ids]
    truth = set(truth_research_ids)
    shared_research_ids = truth & set(imputed_research_ids)
    imputed_only = set(imputed_research_ids) - truth
    research_pairs = tuple(
        (as_research_id(first), as_research_id(second), float(coefficient))
        for first, second, coefficient in kinship_pairs
    )
    duplicated_genomes = {
        imputed_side
        for first, second, coefficient in research_pairs
        if coefficient > DUPLICATE_KINSHIP
        for truth_side, imputed_side in ((first, second), (second, first))
        if truth_side in truth and imputed_side in imputed_only
    }
    kept_imputed = [research_id for research_id in imputed_research_ids if research_id in imputed_only - duplicated_genomes]
    LOGGER.info(
        "cohort rows: imputed rows replaced by a truth row: %s sharing a research ID, %s duplicating a truth genome",
        _reportable_count(len(shared_research_ids)),
        _reportable_count(len(duplicated_genomes)),
    )
    return CohortRows(
        research_ids=(*truth_research_ids, *kept_imputed),
        genotype_source=(LONG_READ_SOURCE,) * len(truth_research_ids) + (IMPUTED_SOURCE,) * len(kept_imputed),
        kinship_pairs=research_pairs,
        imputed_rows_sharing_a_research_id=len(shared_research_ids),
        imputed_rows_duplicating_a_truth_genome=len(duplicated_genomes),
    )


def _research_id_values(research_ids: Sequence[ResearchId]) -> list[str]:
    """The strings of research IDs, refusing any ID of another namespace."""
    if not all(isinstance(research_id, ResearchId) for research_id in research_ids):
        raise TypeError("the cohort is keyed by ResearchId; map imputed samples through the crosswalk first.")
    return [research_id.value for research_id in research_ids]


def _reportable_count(count: int) -> str:
    """The All of Us dissemination rule: a count of 1 to 20 participants is never written out."""
    return str(count) if count == 0 or count >= MINIMUM_REPORTED_PARTICIPANTS else "1-20 (suppressed)"


@dataclass(frozen=True, slots=True)
class AncestryPcs:
    """Genetic PCs per research ID, from the All of Us ancestry predictions table."""

    research_ids: tuple[str, ...]
    components: F64Array

    @classmethod
    def read(cls, path: str | Path) -> AncestryPcs:
        """Every PC of the release's ``pca_features`` column (a JSON list per ``research_id``)."""
        with Path(path).open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle, delimiter="\t"))
        research_ids = tuple(row["research_id"] for row in rows)
        if len(set(research_ids)) != len(research_ids):
            raise ValueError("the ancestry table repeats a research_id.")
        components = np.array([json.loads(row["pca_features"]) for row in rows], dtype=np.float64)
        if components.ndim != 2 or not np.all(np.isfinite(components)):
            raise ValueError("every pca_features entry must be a finite list of the same length.")
        return cls(research_ids=research_ids, components=components)

    def for_samples(self, research_ids: Sequence[ResearchId]) -> F64Array:
        """The PC rows of ``research_ids``, in that order; every sample must have PCs."""
        values = _research_id_values(research_ids)
        row_of = {research_id: row for row, research_id in enumerate(self.research_ids)}
        missing = [research_id for research_id in values if research_id not in row_of]
        if missing:
            raise ValueError(f"{len(missing)} samples have no genetic PCs.")
        return self.components[[row_of[research_id] for research_id in values]]


def indicator_columns(levels: Sequence[str]) -> tuple[tuple[str, ...], F64Array]:
    """0/1 columns for every level but the first in sorted order.

    With an intercept in C, one level must be dropped or C is rank-deficient. The
    covariate projection is the same whichever level is dropped, so the choice only
    fixes the column set.
    """
    observed = sorted(set(levels))
    kept = tuple(observed[1:])
    values = np.array([[level == name for name in kept] for level in levels], dtype=np.float64)
    return kept, values.reshape(len(levels), len(kept))


def kinship_components(
    sample_ids: Sequence[str],
    first_ids: Sequence[str],
    second_ids: Sequence[str],
    kinship: Sequence[float],
) -> I64Array:
    """A component label per sample: connected components of the pairs above second degree.

    Pairs naming a sample outside ``sample_ids`` are ignored; they cannot link two samples.
    Components are numbered in the order of their smallest sample ID, so the labels, and
    the folds kinship_folds seeds from them, depend on who the samples are, not on the
    order they are listed in.
    """
    index_of = {sample_id: index for index, sample_id in enumerate(sample_ids)}
    if len(index_of) != len(sample_ids):
        raise ValueError("sample_ids repeats a sample.")
    parent = np.arange(len(sample_ids), dtype=np.int64)

    def root(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = int(parent[index])
        return index

    for first, second, coefficient in zip(first_ids, second_ids, kinship, strict=True):
        if coefficient > SECOND_DEGREE_KINSHIP and first in index_of and second in index_of:
            first_root, second_root = root(index_of[first]), root(index_of[second])
            parent[max(first_root, second_root)] = min(first_root, second_root)
    roots = np.array([root(index) for index in range(len(sample_ids))], dtype=np.int64)
    smallest_member: dict[int, str] = {}
    for sample_id, component_root in zip(sample_ids, roots.tolist(), strict=True):
        if component_root not in smallest_member or sample_id < smallest_member[component_root]:
            smallest_member[component_root] = sample_id
    label_of_root = {
        component_root: label
        for label, component_root in enumerate(sorted(smallest_member, key=smallest_member.__getitem__))
    }
    return np.array([label_of_root[component_root] for component_root in roots.tolist()], dtype=np.int64)


def kinship_folds(components: I64Array, strata: Sequence[str], fold_count: int, seed: int) -> I64Array:
    """A fold per sample: whole components, spread evenly over the folds within every stratum.

    Components go largest first (ties in a seeded random order), each to the fold whose
    placement least increases the sum of squared (fold, stratum) counts. For components
    inside one stratum this is list scheduling, so that stratum's fold counts differ by at
    most its largest component.
    """
    if fold_count < 2:
        raise ValueError("cross-fitting needs at least two folds.")
    component_labels = np.asarray(components, dtype=np.int64)
    stratum_names, stratum_of_sample = np.unique(np.asarray(strata), return_inverse=True)
    component_count = int(component_labels.max()) + 1
    members = np.zeros((component_count, stratum_names.shape[0]), dtype=np.int64)
    np.add.at(members, (component_labels, stratum_of_sample), 1)
    shuffled = np.random.default_rng(seed).permutation(component_count)
    order = shuffled[np.argsort(-members[shuffled].sum(axis=1), kind="stable")]
    fold_counts = np.zeros((fold_count, stratum_names.shape[0]), dtype=np.int64)
    fold_of_component = np.empty(component_count, dtype=np.int64)
    for component in order:
        fold = int(np.argmin(fold_counts @ members[component]))
        fold_of_component[component] = fold
        fold_counts[fold] += members[component]
    return fold_of_component[component_labels]


@dataclass(frozen=True, slots=True)
class Cohort:
    """The shared design: C with its column names, and one target column per trait."""

    research_ids: tuple[ResearchId, ...]
    covariate_names: tuple[str, ...]
    covariates: F64Array
    trait_names: tuple[str, ...]
    targets: F64Array

    @property
    def observed(self) -> BoolArray:
        """Which (sample, trait) targets exist. A training set uses only rows observed for its traits."""
        return np.isfinite(self.targets)


def build_cohort(
    research_ids: Sequence[ResearchId],
    person_covariates: Mapping[str, Sequence[float]],
    categorical_covariates: Mapping[str, Sequence[str]],
    ancestry: AncestryPcs,
    pipeline_half: Sequence[str],
    genotype_source: Sequence[str],
    trait_targets: Mapping[str, Mapping[str, float]],
) -> Cohort:
    """Assemble C and the target matrix for ``research_ids``.

    ``person_covariates`` are trait-agnostic numeric columns (one value per sample);
    ``categorical_covariates``, ``pipeline_half`` (the imputation half) and
    ``genotype_source`` (imputed or long-read-called rows) become indicators. Each
    trait's targets are keyed by research ID. C must have full column rank.
    """
    research_id_values = _research_id_values(research_ids)
    sample_count = len(research_ids)
    if len(set(research_id_values)) != sample_count:
        raise ValueError("research_ids repeats a participant; resolve the two halves with resolve_cohort_rows first.")
    names = ["intercept"]
    columns = [np.ones((sample_count, 1))]
    for name, values in person_covariates.items():
        column = np.asarray(values, dtype=np.float64).reshape(sample_count, 1)
        if not np.all(np.isfinite(column)):
            raise ValueError(f"covariate {name!r} has missing values.")
        names.append(name)
        columns.append(column)
    indicator_sources = dict(categorical_covariates)
    indicator_sources["pipeline_half"] = pipeline_half
    indicator_sources["genotype_source"] = genotype_source
    for name, levels in indicator_sources.items():
        if len(levels) != sample_count:
            raise ValueError(f"{name!r} needs one level per sample.")
        kept, values = indicator_columns(levels)
        names.extend(f"{name}={level}" for level in kept)
        columns.append(values)
    components = ancestry.for_samples(research_ids)
    names.extend(f"PC{index + 1}" for index in range(components.shape[1]))
    columns.append(components)
    covariates = np.concatenate(columns, axis=1)
    if np.linalg.matrix_rank(covariates) < covariates.shape[1]:
        raise ValueError("the covariate matrix is rank-deficient; remove the collinear covariate.")
    targets = np.full((sample_count, len(trait_targets)), np.nan)
    for trait_index, (trait, values_by_id) in enumerate(trait_targets.items()):
        if not all(np.isfinite(value) for value in values_by_id.values()):
            raise ValueError(f"trait {trait!r} has a non-finite target; leave a missing participant out instead.")
        for row, research_id in enumerate(research_id_values):
            if research_id in values_by_id:
                targets[row, trait_index] = values_by_id[research_id]
        if not np.any(np.isfinite(targets[:, trait_index])):
            raise ValueError(f"trait {trait!r} has no target for any cohort sample; key its targets by research ID.")
    return Cohort(
        research_ids=tuple(research_ids),
        covariate_names=tuple(names),
        covariates=covariates,
        trait_names=tuple(trait_targets),
        targets=targets,
    )
