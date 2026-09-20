"""The gene-dosage term: an SV's effect through the copies of the genes whose exons it changes.

Derivation (novel-svfunction.md §6, Theorem 5; review-theory's Proposal 4.2). To
first order a variant's per-allele effect is

    beta_v = sum_g c_vg theta_g + epsilon_v,

where theta_g is the effect of one extra functional copy of gene g, epsilon_v the
variant's own non-dosage effect, and c_vg the change in g's functional copies that
one unit of the stored column carries. For a copy-number SV allele,

    c_vg = Delta_v phi_vg,

with Delta_v the allele's copy change per unit of its column (-1 for a deletion
allele, +k for an allele carrying k extra copies, +1 per copy for a copy-number
column) and phi_vg the fraction of g's exonic bases inside the SV's span. Gains
carry a learned factor rho on top (rho = 1 under linear dosage, Theorem 5), which
is kept separate so that it enters linearly.

The gene set. The mechanism defines it: the genes whose merged exons the SV
overlaps, {g : phi_vg > 0}. There is no nearest-gene rule and no window. An SV
that overlaps no exon changes no copies (c = 0), and whatever it does to a nearby
gene is regulatory, which the TSS-distance and SV-context terms carry.

Two uses, one annotation (``exon_overlap``):
- **Cis, one target gene** (``unit_loadings``, ``unit_columns``). The column
  U_g = sum_v c_vg D_v sits beside the variant columns with its own prior. As that
  prior's variance tends to zero the model is exactly the current one, and the
  identity X beta + U theta = X (beta + c theta) means the column is the shared
  prior Cov(beta) = tau^2 c c' of Theorem 5, not a new genotype.
- **Organismal, every gene** (``dosage_sensitivity_burden``, ``gene_dosage_design``).
  There is no single target, so the term is an annotation on the SV's prior scale.
  Independent gene effects add their variances with squared loadings,
  Var(beta_v | dosage) = sum_g c_vg^2 tau_g^2, and tau_g^2 scales with the gene's
  dosage sensitivity (Corollary 5a: tau_g^2 proportional to s_het,g), for which a
  public score stands in: pHaplo for losses, pTriplo for gains (Collins et al.
  2022). So the burden is w_v = sum_g c_vg^2 score_g, and log u_v gains a learned
  smooth of it, separately for losses and gains, each switched off exactly by its
  own two weights (``prior_terms.switchable_smooth``).
  This matches each SV's marginal variance only. The scale-mixture prior treats
  effects as independent given their scales, so Theorem 5's covariance
  tau^2 c c' between SVs of one gene is dropped here: a known approximation. The
  cis unit column carries that covariance exactly.

Not here: coding breaks by INV, INS or MEI breakpoints (c = -1 in Theorem 5), and a
gain/loss asymmetry within one copy-number column (review-theory's #6 smooth).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.sparse import csr_matrix

from sv_pgs._typing import F64Array, I64Array
from sv_pgs.prior_terms import PenaltyGroup, TermDesign, switchable_smooth


@dataclass(frozen=True)
class MergedExons:
    """Each gene's exons merged into disjoint sorted intervals, laid out gene by gene.

    Gene g's intervals are ``starts[offsets[g]:offsets[g + 1]]`` (0-based,
    half-open); ``before[i]`` is the exonic bases of interval i's gene that lie
    before it, and ``exonic_bases[g]`` the gene's total.
    """

    offsets: I64Array
    starts: I64Array
    ends: I64Array
    before: I64Array
    exonic_bases: I64Array

    @property
    def gene_count(self) -> int:
        return int(self.offsets.shape[0] - 1)


def merged_exons(*, exon_starts: I64Array, exon_ends: I64Array, exon_genes: I64Array, gene_count: int) -> MergedExons:
    """Merge every gene's exons over all its transcripts, one chromosome at a time.

    The union over transcripts avoids choosing a transcript: an exon any transcript
    uses is a copy of the gene's coding or processed sequence. Overlapping or
    touching exons of one gene merge; exons of different genes never do.
    """
    starts = np.asarray(exon_starts, dtype=np.int64)
    ends = np.asarray(exon_ends, dtype=np.int64)
    genes = np.asarray(exon_genes, dtype=np.int64)
    if not starts.shape == ends.shape == genes.shape or starts.ndim != 1:
        raise ValueError("every exon needs one start, end and gene")
    if np.any(ends <= starts) or np.any(starts < 0):
        raise ValueError("every exon must be a non-empty 0-based half-open interval")
    if np.any((genes < 0) | (genes >= gene_count)):
        raise ValueError("every exon's gene must be below gene_count")
    if np.any(np.bincount(genes, minlength=gene_count) == 0):
        raise ValueError("every gene needs at least one exon")
    order = np.lexsort((starts, genes))
    starts, ends, genes = starts[order], ends[order], genes[order]
    # A new merged interval opens where the gene changes or the exon starts past the gene's running end.
    # The running end restarts at each gene: offsetting every gene's ends by gene x (largest end + 1)
    # puts each gene's keys above all earlier genes', so one running maximum never crosses a gene.
    stride = int(ends.max()) + 1
    running_end = np.maximum.accumulate(genes * stride + ends) - genes * stride
    opens = np.ones(starts.shape[0], dtype=bool)
    opens[1:] = (genes[1:] != genes[:-1]) | (starts[1:] > running_end[:-1])
    group = np.cumsum(opens) - 1
    merged_starts = starts[opens]
    merged_ends = np.zeros(merged_starts.shape[0], dtype=np.int64)
    np.maximum.at(merged_ends, group, ends)
    merged_genes = genes[opens]
    lengths = merged_ends - merged_starts
    offsets = np.concatenate([[0], np.cumsum(np.bincount(merged_genes, minlength=gene_count))]).astype(np.int64)
    running = np.cumsum(lengths) - lengths
    before = running - running[offsets[merged_genes]]
    exonic_bases = np.add.reduceat(lengths, offsets[:-1]).astype(np.int64)
    return MergedExons(offsets=offsets, starts=merged_starts, ends=merged_ends, before=before.astype(np.int64), exonic_bases=exonic_bases)


def _exonic_bases_below(merged: MergedExons, gene: int, positions: I64Array) -> I64Array:
    """Gene ``gene``'s exonic bases below each position: its cumulative exonic length there."""
    first, last = int(merged.offsets[gene]), int(merged.offsets[gene + 1])
    starts, ends, before = merged.starts[first:last], merged.ends[first:last], merged.before[first:last]
    interval = np.searchsorted(starts, positions, side="right") - 1
    inside = interval >= 0
    safe = np.maximum(interval, 0)
    partial = np.clip(positions - starts[safe], 0, ends[safe] - starts[safe])
    return np.where(inside, before[safe] + partial, 0).astype(np.int64)


@dataclass(frozen=True)
class ExonOverlap:
    """The per-(SV, gene) annotation: one row per SV allele and gene whose exons it overlaps.

    ``fraction`` is phi, the share of the gene's merged exonic bases inside the
    SV's span, in (0, 1]; pairs with no exonic overlap are absent.
    """

    variant: I64Array
    gene: I64Array
    fraction: F64Array


def exon_overlap(*, sv_starts: I64Array, sv_ends: I64Array, merged: MergedExons) -> ExonOverlap:
    """phi_vg for every SV allele v and gene g of one chromosome, exactly, from base counts.

    The span is the region whose copies the allele changes: the deleted or the
    duplicated interval, 0-based half-open. A gene is visited once and only the
    SVs that can reach it are examined: those starting within the longest SV's
    length before the gene's first exon and before its last exon's end.
    """
    starts = np.asarray(sv_starts, dtype=np.int64)
    ends = np.asarray(sv_ends, dtype=np.int64)
    if starts.shape != ends.shape or starts.ndim != 1:
        raise ValueError("every SV needs one start and one end")
    if np.any(ends <= starts):
        raise ValueError("every SV span must be a non-empty 0-based half-open interval")
    empty = ExonOverlap(variant=np.zeros(0, np.int64), gene=np.zeros(0, np.int64), fraction=np.zeros(0))
    if starts.size == 0:
        return empty
    order = np.argsort(starts, kind="stable")
    sorted_starts = starts[order]
    longest = int((ends - starts).max())
    variants: list[I64Array] = []
    genes: list[I64Array] = []
    fractions: list[F64Array] = []
    for gene in range(merged.gene_count):
        first, last = int(merged.offsets[gene]), int(merged.offsets[gene + 1])
        gene_start, gene_end = int(merged.starts[first]), int(merged.ends[last - 1])
        low = np.searchsorted(sorted_starts, gene_start - longest, side="left")
        high = np.searchsorted(sorted_starts, gene_end, side="left")
        candidates = order[low:high]
        candidates = candidates[ends[candidates] > gene_start]
        if candidates.size == 0:
            continue
        covered = _exonic_bases_below(merged, gene, ends[candidates]) - _exonic_bases_below(merged, gene, starts[candidates])
        hit = covered > 0
        if not np.any(hit):
            continue
        variants.append(candidates[hit])
        genes.append(np.full(int(hit.sum()), gene, dtype=np.int64))
        fractions.append(covered[hit] / float(merged.exonic_bases[gene]))
    if not variants:
        return empty
    return ExonOverlap(variant=np.concatenate(variants), gene=np.concatenate(genes), fraction=np.concatenate(fractions))


def _checked_copy_change(copy_change: F64Array, overlap: ExonOverlap) -> F64Array:
    change = np.asarray(copy_change, dtype=np.float64)
    if change.ndim != 1 or np.any(~np.isfinite(change)):
        raise ValueError("copy_change needs one finite value per variant")
    if overlap.variant.size and int(overlap.variant.max()) >= change.shape[0]:
        raise ValueError("the overlap names a variant beyond copy_change")
    return change


@dataclass(frozen=True)
class UnitLoadings:
    """c_vg split by direction: ``loss`` (Delta < 0) and ``gain`` (Delta > 0), genes by variants.

    The unit column of gene g at gain factor rho is U_g = D (loss + rho gain)[g]',
    so dU/drho = D gain[g]' exactly, and rho is profiled in one dimension.
    """

    loss: csr_matrix
    gain: csr_matrix

    def at(self, rho: float) -> csr_matrix:
        """C(rho) = loss + rho gain: the loadings of every unit at one gain factor."""
        if not np.isfinite(rho):
            raise ValueError("the gain factor must be finite")
        return (self.loss + rho * self.gain).tocsr()


def unit_loadings(*, overlap: ExonOverlap, copy_change: F64Array, gene_count: int) -> UnitLoadings:
    """c_vg = Delta_v phi_vg, as sparse (genes x variants) matrices for losses and gains."""
    change = _checked_copy_change(copy_change, overlap)
    loading = change[overlap.variant] * overlap.fraction
    shape = (gene_count, change.shape[0])
    losses = loading < 0.0
    gains = loading > 0.0
    return UnitLoadings(
        loss=csr_matrix((loading[losses], (overlap.gene[losses], overlap.variant[losses])), shape=shape),
        gain=csr_matrix((loading[gains], (overlap.gene[gains], overlap.variant[gains])), shape=shape),
    )


def unit_columns(dosages: F64Array, loadings: UnitLoadings, rho: float) -> F64Array:
    """U = D C(rho)': one column per gene, from the stored columns D (samples x variants)."""
    columns = np.asarray(dosages, dtype=np.float64)
    if columns.ndim != 2 or columns.shape[1] != loadings.loss.shape[1]:
        raise ValueError("the dosages must be (samples, variants) over the loadings' variants")
    return np.asarray(loadings.at(rho) @ columns.T).T


def dosage_sensitivity_burden(
    *, overlap: ExonOverlap, copy_change: F64Array, loss_score: F64Array, gain_score: F64Array
) -> tuple[F64Array, F64Array]:
    """w_v = sum_g c_vg^2 score_g, for losses (pHaplo) and gains (pTriplo) separately.

    Squared loadings because independent gene effects add variances (Theorem 5's
    Cov(beta) = tau^2 c c'); the score stands in for tau_g^2, proportional to the
    gene's dosage sensitivity (Corollary 5a). A variant with no exonic overlap, or
    no copy change, has zero burden in both.
    """
    change = _checked_copy_change(copy_change, overlap)
    losses = np.asarray(loss_score, dtype=np.float64)
    gains = np.asarray(gain_score, dtype=np.float64)
    if losses.ndim != 1 or gains.shape != losses.shape:
        raise ValueError("the loss and gain scores need one value per gene")
    if overlap.gene.size and int(overlap.gene.max()) >= losses.shape[0]:
        raise ValueError("the overlap names a gene beyond the scores")
    if np.any(~np.isfinite(losses)) or np.any(~np.isfinite(gains)) or np.any(losses < 0.0) or np.any(gains < 0.0):
        raise ValueError("every dosage-sensitivity score must be finite and non-negative")
    delta = change[overlap.variant]
    squared = np.square(delta * overlap.fraction)
    loss_burden = np.bincount(overlap.variant, weights=np.where(delta < 0.0, squared * losses[overlap.gene], 0.0), minlength=change.shape[0])
    gain_burden = np.bincount(overlap.variant, weights=np.where(delta > 0.0, squared * gains[overlap.gene], 0.0), minlength=change.shape[0])
    return loss_burden, gain_burden


def gene_dosage_design(
    *, loss_burden: F64Array, gain_burden: F64Array, spacing: float, class_index: I64Array | None = None
) -> TermDesign:
    """The organismal gene-dosage term: a switchable smooth of each burden, side by side.

    Four learned weights: each direction's linear part and its curvature. At all
    four infinite the term is exactly zero, the model without it; the loss and gain
    halves switch off independently. A direction no SV in the fit exercises gives no
    columns.
    """
    parts = [
        switchable_smooth(loss_burden, spacing, name="gene_dosage::loss", class_index=class_index),
        switchable_smooth(gain_burden, spacing, name="gene_dosage::gain", class_index=class_index),
    ]
    design = np.hstack([part.design for part in parts])
    groups: list[PenaltyGroup] = []
    start = 0
    for part in parts:
        for group in part.groups:
            groups.append(PenaltyGroup(name=group.name, columns=start + group.columns, factor=group.factor))
        start += part.design.shape[1]
    names = tuple(name for part in parts for name in part.names)
    return TermDesign(design=design, groups=tuple(groups), names=names)
