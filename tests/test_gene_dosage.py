"""The gene-dosage term: exact exon overlaps, the shared unit column, its nesting, and the organismal smooth."""
from __future__ import annotations

import numpy as np
import pytest

from sv_pgs.gene_dosage import (
    dosage_sensitivity_burden,
    exon_overlap,
    gene_dosage_design,
    merged_exons,
    unit_columns,
    unit_loadings,
)
from sv_pgs.prior_terms import measurement_offset, stack_terms
from sv_pgs.scale_mixture_ep import AnnotationGroup, derived_lattice, scale_mixture_prior

_EPSILON = float(np.finfo(np.float64).eps)


def _rounding_bound(terms: int, magnitude: float) -> float:
    """The forward rounding bound n eps |x| of a sum of n products of size |x|."""
    return terms * _EPSILON * magnitude


def _random_annotation(seed: int, genes: int = 6, exons_per_gene: int = 5, svs: int = 40, extent: int = 4000):
    generator = np.random.default_rng(seed)
    exon_starts, exon_ends, exon_genes = [], [], []
    for gene in range(genes):
        origin = int(generator.integers(0, extent - 600))
        starts = np.sort(origin + generator.integers(0, 500, size=exons_per_gene))
        exon_starts.append(starts)
        exon_ends.append(starts + generator.integers(1, 80, size=exons_per_gene))
        exon_genes.append(np.full(exons_per_gene, gene))
    sv_starts = generator.integers(0, extent, size=svs)
    sv_ends = sv_starts + generator.integers(1, 900, size=svs)
    return (
        np.concatenate(exon_starts), np.concatenate(exon_ends), np.concatenate(exon_genes),
        sv_starts.astype(np.int64), sv_ends.astype(np.int64), genes,
    )


def _brute_force_fractions(exon_starts, exon_ends, exon_genes, sv_starts, sv_ends, genes):
    """phi by counting bases one at a time: the definition, with no interval arithmetic."""
    extent = int(max(exon_ends.max(), sv_ends.max()))
    fractions = {}
    for gene in range(genes):
        mask = np.zeros(extent, dtype=bool)
        for start, end in zip(exon_starts[exon_genes == gene], exon_ends[exon_genes == gene]):
            mask[start:end] = True
        for variant, (start, end) in enumerate(zip(sv_starts, sv_ends)):
            covered = int(mask[start:end].sum())
            if covered:
                fractions[(variant, gene)] = covered / int(mask.sum())
    return fractions


class TestExonOverlap:
    """phi_vg is the share of a gene's merged exonic bases inside the SV's span, exactly."""

    def test_merging_joins_a_gene_s_overlapping_exons_and_never_two_genes(self):
        merged = merged_exons(
            exon_starts=np.array([100, 150, 400, 120]),
            exon_ends=np.array([200, 250, 450, 130]),
            exon_genes=np.array([0, 0, 0, 1]),
            gene_count=2,
        )
        assert merged.exonic_bases.tolist() == [150 + 50, 10]
        assert merged.starts.tolist() == [100, 400, 120] and merged.ends.tolist() == [250, 450, 130]

    @pytest.mark.parametrize("seed", [0, 1, 2])
    def test_the_fractions_equal_a_base_by_base_count(self, seed):
        exon_starts, exon_ends, exon_genes, sv_starts, sv_ends, genes = _random_annotation(seed)
        merged = merged_exons(exon_starts=exon_starts, exon_ends=exon_ends, exon_genes=exon_genes, gene_count=genes)
        overlap = exon_overlap(sv_starts=sv_starts, sv_ends=sv_ends, merged=merged)
        found = {(int(v), int(g)): float(f) for v, g, f in zip(overlap.variant, overlap.gene, overlap.fraction)}
        expected = _brute_force_fractions(exon_starts, exon_ends, exon_genes, sv_starts, sv_ends, genes)
        assert found.keys() == expected.keys()
        for key, value in expected.items():
            assert found[key] == pytest.approx(value, rel=_rounding_bound(2, 1.0))

    def test_a_whole_gene_deletion_covers_it_and_an_intronic_one_does_not(self):
        merged = merged_exons(
            exon_starts=np.array([1000, 1500, 5000]), exon_ends=np.array([1100, 1600, 5200]),
            exon_genes=np.array([0, 0, 1]), gene_count=2,
        )
        overlap = exon_overlap(sv_starts=np.array([900, 1200, 1050]), sv_ends=np.array([1700, 1400, 5100]), merged=merged)
        found = {(int(v), int(g)): float(f) for v, g, f in zip(overlap.variant, overlap.gene, overlap.fraction)}
        assert found[(0, 0)] == 1.0
        assert (1, 0) not in found, "an SV inside an intron changes no exonic copies"
        # SV 2 reaches into both genes: half of gene 0's second exon is past it, and half of gene 1's exon.
        assert found[(2, 0)] == pytest.approx(150 / 200) and found[(2, 1)] == pytest.approx(100 / 200)

    def test_malformed_annotations_are_refused(self):
        with pytest.raises(ValueError):
            merged_exons(exon_starts=np.array([10]), exon_ends=np.array([10]), exon_genes=np.array([0]), gene_count=1)
        with pytest.raises(ValueError):
            merged_exons(exon_starts=np.array([10]), exon_ends=np.array([20]), exon_genes=np.array([0]), gene_count=2)
        merged = merged_exons(exon_starts=np.array([10]), exon_ends=np.array([20]), exon_genes=np.array([0]), gene_count=1)
        with pytest.raises(ValueError):
            exon_overlap(sv_starts=np.array([30]), sv_ends=np.array([30]), merged=merged)


def _one_gene_fit(seed: int):
    """A gene hit by a deletion, a duplication and an inversion, with SNV columns beside them."""
    generator = np.random.default_rng(seed)
    merged = merged_exons(exon_starts=np.array([100, 300]), exon_ends=np.array([200, 400]), exon_genes=np.array([0, 0]), gene_count=1)
    # Variants: two SNVs (spans of 1 bp between the exons), a DEL over exon 1, a DUP over the gene, an INV.
    sv_starts = np.array([250, 260, 90, 50, 150])
    sv_ends = np.array([251, 261, 210, 450, 350])
    copy_change = np.array([0.0, 0.0, -1.0, 1.0, 0.0])
    overlap = exon_overlap(sv_starts=sv_starts, sv_ends=sv_ends, merged=merged)
    dosages = generator.integers(0, 3, size=(60, 5)).astype(np.float64)
    return overlap, copy_change, dosages


class TestUnitColumn:
    """U = D C(rho)' is Theorem 5's shared prior Cov(beta) = tau^2 c c', and nests the current model."""

    def test_the_loadings_are_signed_exon_fractions(self):
        overlap, copy_change, _dosages = _one_gene_fit(0)
        loadings = unit_loadings(overlap=overlap, copy_change=copy_change, gene_count=1)
        assert loadings.loss.toarray().tolist() == [[0.0, 0.0, -0.5, 0.0, 0.0]]
        assert loadings.gain.toarray().tolist() == [[0.0, 0.0, 0.0, 1.0, 0.0]]

    def test_the_unit_column_is_the_shared_effect_on_every_variant(self):
        overlap, copy_change, dosages = _one_gene_fit(1)
        loadings = unit_loadings(overlap=overlap, copy_change=copy_change, gene_count=1)
        rho = 0.8
        unit = unit_columns(dosages, loadings, rho)[:, 0]
        loading = loadings.at(rho).toarray()[0]
        generator = np.random.default_rng(2)
        beta, theta = generator.normal(size=5), float(generator.normal())
        assert np.allclose(dosages @ beta + unit * theta, dosages @ (beta + loading * theta), atol=_rounding_bound(5, float(np.abs(dosages).max()) * 4.0))

    def test_the_column_is_linear_in_the_gain_factor(self):
        overlap, copy_change, dosages = _one_gene_fit(3)
        loadings = unit_loadings(overlap=overlap, copy_change=copy_change, gene_count=1)
        derivative = np.asarray(loadings.gain @ dosages.T).T
        difference = unit_columns(dosages, loadings, 1.7) - unit_columns(dosages, loadings, 0.4)
        assert np.allclose(difference, 1.3 * derivative, atol=_rounding_bound(5, float(np.abs(dosages).max()) * 2.0))

    def test_the_evidence_is_the_current_model_at_zero_variance_and_its_score_is_closed_form(self):
        # y ~ N(0, K + t a a'), a = U: log|K + t aa'| = log|K| + log(1 + t s) and the quadratic form
        # drops by t q^2 / (1 + t s), with s = a'K^-1 a and q = a'K^-1 y. So the evidence is exactly the
        # current model's at t = 0, and its derivative there is (q^2 - s) / 2.
        overlap, copy_change, dosages = _one_gene_fit(4)
        loadings = unit_loadings(overlap=overlap, copy_change=copy_change, gene_count=1)
        unit = unit_columns(dosages, loadings, 1.0)[:, 0]
        generator = np.random.default_rng(5)
        centred = dosages - dosages.mean(axis=0)
        kernel = centred @ np.diag(generator.uniform(0.01, 0.1, size=5)) @ centred.T + np.eye(dosages.shape[0])
        y = generator.normal(size=dosages.shape[0])
        a = unit - unit.mean()

        def evidence(t: float) -> float:
            covariance = kernel + t * np.outer(a, a)
            _sign, log_determinant = np.linalg.slogdet(covariance)
            return -0.5 * log_determinant - 0.5 * float(y @ np.linalg.solve(covariance, y))

        s = float(a @ np.linalg.solve(kernel, a))
        q = float(a @ np.linalg.solve(kernel, y))
        condition = float(np.linalg.cond(kernel))
        for t in (0.0, 0.01, 0.3, 2.0):
            closed = evidence(0.0) - 0.5 * np.log1p(t * s) + 0.5 * t * q * q / (1.0 + t * s)
            assert evidence(t) == pytest.approx(closed, abs=_rounding_bound(dosages.shape[0] ** 2, condition * abs(evidence(0.0))))
        score = 0.5 * (q * q - s)
        # The closed form's own derivative at 0, by its exact series: the error of a one-sided step h is h s^2 / 2 + O(h^2).
        step = 1e-6
        assert (evidence(step) - evidence(0.0)) / step == pytest.approx(score, abs=step * s * (s + q * q) + _rounding_bound(dosages.shape[0] ** 2, condition * abs(evidence(0.0))) / step)


class TestOrganismalTerm:
    """The burden sums squared loadings times the gene's dosage sensitivity, by direction."""

    def test_the_burden_is_the_squared_loadings_times_the_score(self):
        merged = merged_exons(exon_starts=np.array([100, 300, 1000]), exon_ends=np.array([200, 400, 1100]), exon_genes=np.array([0, 0, 1]), gene_count=2)
        # A DEL over half of gene 0, a two-copy DUP over all of gene 1, and an INV over gene 0.
        overlap = exon_overlap(sv_starts=np.array([90, 950, 50]), sv_ends=np.array([210, 1200, 450]), merged=merged)
        loss_burden, gain_burden = dosage_sensitivity_burden(
            overlap=overlap, copy_change=np.array([-1.0, 2.0, 0.0]),
            loss_score=np.array([0.9, 0.2]), gain_score=np.array([0.1, 0.7]),
        )
        assert loss_burden.tolist() == pytest.approx([0.25 * 0.9, 0.0, 0.0])
        assert gain_burden.tolist() == pytest.approx([0.0, 4.0 * 0.7, 0.0])

    def test_an_unscored_or_negative_score_is_refused(self):
        merged = merged_exons(exon_starts=np.array([100]), exon_ends=np.array([200]), exon_genes=np.array([0]), gene_count=1)
        overlap = exon_overlap(sv_starts=np.array([90]), sv_ends=np.array([210]), merged=merged)
        with pytest.raises(ValueError):
            dosage_sensitivity_burden(overlap=overlap, copy_change=np.array([-1.0]), loss_score=np.array([np.nan]), gain_score=np.array([0.1]))
        with pytest.raises(ValueError):
            dosage_sensitivity_burden(overlap=overlap, copy_change=np.array([-1.0]), loss_score=np.array([-0.1]), gain_score=np.array([0.1]))


def _genome_like(seed: int, variant_count: int = 600):
    """SNVs, deletions and duplications; a fifth of the CNVs cover exons, with varied fractions and scores."""
    generator = np.random.default_rng(seed)
    classes = generator.choice(3, size=variant_count, p=[0.6, 0.25, 0.15]).astype(np.int64)
    copy_change = np.where(classes == 1, -1.0, np.where(classes == 2, 1.0, 0.0))
    covering = (classes > 0) & (generator.uniform(size=variant_count) < 0.2)
    loss_burden = np.where(covering & (classes == 1), generator.uniform(0.0, 1.0, size=variant_count) ** 2 * generator.uniform(size=variant_count), 0.0)
    gain_burden = np.where(covering & (classes == 2), generator.uniform(0.0, 1.0, size=variant_count) ** 2 * generator.uniform(size=variant_count), 0.0)
    return classes, copy_change, covering, loss_burden, gain_burden


class TestGeneDosageDesign:
    """Each direction's smooth is switched off exactly by its own two weights."""

    def test_every_direction_is_penalized_so_the_term_switches_off_exactly(self):
        classes, _change, _covering, loss_burden, gain_burden = _genome_like(0)
        term = gene_dosage_design(loss_burden=loss_burden, gain_burden=gain_burden, spacing=0.25, class_index=classes)
        assert term.null_basis.shape[1] == 0
        assert [group.name for group in term.groups] == [
            "gene_dosage::loss linear", "gene_dosage::loss curvature", "gene_dosage::gain linear", "gene_dosage::gain curvature",
        ]

    def test_a_direction_no_sv_exercises_gives_no_columns(self):
        classes, _change, _covering, loss_burden, _gain = _genome_like(1)
        term = gene_dosage_design(loss_burden=loss_burden, gain_burden=np.zeros_like(loss_burden), spacing=0.25, class_index=classes)
        assert all(group.name.startswith("gene_dosage::loss") for group in term.groups)

    def test_the_burden_is_not_what_an_exon_overlap_indicator_already_carries(self):
        classes, _change, covering, loss_burden, _gain = _genome_like(2)
        indicators = np.column_stack([classes == value for value in range(3)] + [covering]).astype(np.float64)
        base = np.linalg.matrix_rank(indicators)
        assert np.linalg.matrix_rank(np.column_stack([indicators, loss_burden])) == base + 1

    def test_the_stacked_term_builds_a_prior_the_engine_accepts(self):
        classes, _change, _covering, loss_burden, gain_burden = _genome_like(3)
        stacked = stack_terms(
            gene_dosage_design(loss_burden=loss_burden, gain_burden=gain_burden, spacing=0.25, class_index=classes),
            class_index=classes,
        )
        generator = np.random.default_rng(6)
        offset = measurement_offset(reliability=generator.uniform(0.3, 1.0, size=classes.shape[0]), local_variance=generator.uniform(0.02, 0.5, size=classes.shape[0]))
        # Effects with signal above the noise, so the derived kernel range is not empty (see test_prior_terms).
        effects = generator.normal(scale=1.0, size=classes.shape[0])
        nodes, floor, top = derived_lattice(np.full(classes.shape[0], 4.0), effects * 4.0, offset, tolerance=1e-3)
        prior = scale_mixture_prior(
            class_index=classes,
            log_variance_offset=offset,
            annotation_design=stacked.design,
            annotation_groups=[AnnotationGroup(columns=columns, penalty=penalty) for columns, penalty in stacked.groups],
            nodes=nodes,
            floor=floor,
            top=top,
        )
        assert prior.scale_size == stacked.design.shape[1] > 0
