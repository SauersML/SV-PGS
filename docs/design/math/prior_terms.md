# The scale model's terms as code: what is nested in what

Scope: `sv_pgs/prior_terms.py`, which turns measured quantities into the engine's
offset `o_j`, annotation design `d_j` and penalty groups. The derivations are in
[scale_model.md](scale_model.md) (§1, §5, §6, §8, §9) and
[novel-evoprior.md](novel-evoprior.md); this document records only what the code
guarantees, and which simpler model each term reduces to when its weight is
infinite. No number here is an accuracy claim: adoption is by ablation on
bench-real and bench-sim (EVIDENCE_RULE).

## The rule every term follows

A term is present in the design whatever the data say, and its own learned
weight decides how much of it survives. So:
- **nothing is switched on or off by hand**, and there is no option flag;
- **the infinite-weight limit is exactly the simpler model**, not an approximation
  of it, so a term can only help the evidence;
- **the limit is a penalty null space**, which the engine profiles as a fixed
  effect, so the simpler model's coefficients stay estimable.

`TermDesign.null_basis` returns that limit's directions, and the tests assert what
each one is.

## The offset: coefficient 1, never fitted

`measurement_offset` returns o_j = log r²_j + log H^loc_j.

| part | why its coefficient is 1 | what would break it |
|---|---|---|
| log r²_j | the prior is on the true genotype's effect; the map to the stored column is exactly r² (scale_model.md §1) | a non-affine column map, which changes r itself |
| log H^loc_j | the per-allele to per-SD unit conversion (scale_model.md §5) | using Var(D) instead of Var(D*)/r², which turns the offset's effective coefficient into −S |

Var(D*)/r² = Var(G) holds only for the recalibrated column, because the
recalibration makes Cov(G, D*) = Var(D*). For a raw draw-like DS, Var(DS) is near
Var(G), so dividing it by r² overstates H^loc by about 1/r² (review-mathbugs F6;
bench-sim v7's variance ratio is 0.73–0.97 [semi-real]). So `local_heterozygosity`
takes a `Recalibration`, the block `store_converter.linear_recalibration` returns,
and refuses anything else by type: a raw column cannot reach it.
`heterozygosity(p)` is the 2p(1−p) route, and it is safe from either column because
the recalibration leaves each group's mean unchanged.

## Frequency: the power law is the limit, not the model

`frequency_design` is a penalized cubic-spline smooth in log H^sel, where H^sel
comes from `ancestry_mixed_frequency` (p^sel = Σ_g w_g p_g, the weights profiled
on the evidence, never fitted as linear coefficients).

- **Penalty:** the exact integrated squared second derivative, by two-node
  Gauss-Legendre per knot interval, which is exact because a cubic B-spline's
  second derivative is linear.
- **Null space:** one direction, the straight line in log H. So at an infinite
  weight the term is exactly the power law H^(1+S) that BayesS, SBayesS and LDAK
  fit, with S the profiled null coefficient rather than a hand-set constant.
- **At a finite weight** it holds what a power law cannot: the low-frequency
  plateau where κHs ≪ 1 and the saturation toward −1 at common frequency, both
  derived in scale_model.md §5.
- **Identifiability:** the basis's constant is dropped (`_drop_constant`), since a
  B-spline basis is a partition of unity and its constant direction is exactly the
  class level. What remains cannot absorb ℓ_c.
- **Not this term:** the saturating variance map v = u H s/(1 + κHs) changes the
  density's *shape* with frequency, so it needs the engine to carry κ and n.
  This term is its learned residual, which novel-evoprior.md §8 keeps in that
  model too.

## Class pooling: already the engine's

The density's pooling is `scale_mixture_ep`'s own: η_c = η̄ + δ_c with one learned
roughness weight per deviation, a Gaussian pooling prior on each deviation's
location and width, and no separate class level, so a class's scale is its
deviation's location. The E_g[s] = 1 pin of scale_model.md §9 is what makes the
level identified; the engine takes it by letting g carry its own location. This
module adds nothing there, and must not: a second pooling of the same quantity
would be a second parametrization of one model.

## SV context: one kernel, deviations pooled away

`sv_context_design` takes `store_converter.sv_kernel_features` and lays out

- **pooled columns:** the features summed over classes, whose coefficient is the
  kernel every class shares;
- **deviation columns:** the same features against orthonormal sum-to-zero
  contrasts over classes.

Three weights are learned, each on its own columns:

| weight | its null space | its infinite limit |
|---|---|---|
| roughness | {1, x} for each length term, x = log(1 + gap) | the kernel is a power law in (1 + gap) for each length term |
| overlap | empty | nesting inside an SV allele has no effect |
| deviation size | empty | every class shares one kernel exactly |

Overlap is its own column: being nested inside an SV allele is a different relation
from being near one, so distance 0 would misstate it.

**The roughness is exact.** It is ∫f''(x)² dx over the covered range, the same
measure as the frequency smooth's. It is not a P-spline second difference of the
coefficients, which is a different measure at finite weight even though its null
space is the same (review-mathbugs F5). `sv_kernel_features` puts basis function m
on the knots h(m − 3), …, h(m + 1). That is `_cubic_design`'s convention, so
`_second_derivative_penalty` on those knots is exact, and a test checks it against
an independent per-interval Simpson integral of the store's own basis.

## Stacking: only what the engine can identify

The engine centres each design column within its class and requires full column
rank, with one weight per column. Raw term columns need not satisfy that. A kernel
basis function beyond the chromosome's largest gap is a zero column, and two terms
can share a direction. `stack_terms`:

- **partitions the columns:** each weight's `PenaltyGroup` owns its columns, so no
  column gets two weights and every column gets one;
- **keeps identified directions in order:** each group keeps the directions whose
  class-centred columns are not already spanned by what was kept before it. It takes
  the penalty's null space first, so an identified power-law or linear limit
  survives exactly;
- **finishes with the engine's own test:** while the kept Gram's smallest eigenvalue
  is at or under the engine's floor, it drops the column that eigenvector loads on
  most;
- **maps back:** it returns `loading`, with original = loading @ stacked, so a fitted
  kernel or smooth can be read in its own coordinates.

## Gene dosage: a CNV's effect through the copies of the genes it changes

`sv_pgs/gene_dosage.py` implements review-theory's Proposal 4.2, which is
novel-svfunction Theorem 5 (novel-svfunction.md §6). To first order a variant's
per-allele effect is

β_v = Σ_g c_vg θ_g + ε_v, with c_vg = Δ_v φ_vg,

- **θ_g** is the effect of one extra functional copy of gene g;
- **Δ_v** is the copy change per unit of the stored column: −1 for a deletion
  allele, +k for an allele with k extra copies, +1 per copy for a copy-number column;
- **φ_vg** is the share of g's merged exonic bases (the union over its transcripts,
  so no transcript is chosen) inside the SV's span, counted exactly
  (`exon_overlap`);
- **gains** carry a learned factor ρ (ρ = 1 under linear dosage), kept apart so that
  it enters linearly.

**The gene set** is what the mechanism defines: the genes whose merged exons the SV
overlaps, {g : φ_vg > 0}. There is no nearest-gene rule and no window. An SV that
overlaps no exon changes no copies. What it does to a nearby gene is regulatory,
and the TSS-distance and SV-context terms carry that.

**Cis, one target gene (bench-real).** The column U_g = Σ_v c_vg D_v sits beside
the variant columns with its own prior (`unit_loadings`, `unit_columns`).

- It is Theorem 5's shared prior, not a new genotype: X β + U θ = X(β + c θ), so it
  adds τ² (Xc)(Xc)' to the marginal covariance.
- **Nesting:** with a = U, s = a'K⁻¹a and q = a'K⁻¹y, the evidence is
  L(τ²) = L(0) − ½ log(1 + τ² s) + ½ τ² q²/(1 + τ² s). So at τ² = 0 it is exactly the
  current model, and its score there is (q² − s)/2. The tests check the closed form
  against dense slogdet/solve.
- **ρ:** U(ρ) = D(c_loss + ρ c_gain)', so dU/dρ = D c_gain' exactly, and ρ is
  profiled in one dimension.
- **Identifiability** is Theorem 5's: with one variant per unit only τ_θ² + τ_ε² is
  identified; units with two or more variants and the DEL/DUP sign constraint split
  them.

**Organismal, every gene (AoU).** There is no single target, so the term is an
annotation on the SV's prior scale.

- Independent gene effects add variances with squared loadings, so
  Var(β_v | dosage) = Σ_g c_vg² τ_g².
- By Corollary 5a, τ_g² ∝ s_het,g, for which a public score stands in: pHaplo for
  losses and pTriplo for gains (Collins et al. 2022, taken from a non-Google host).
- The burden is w_v = Σ_g c_vg² score_g, separately for losses and gains
  (`dosage_sensitivity_burden`). log u_v gains a `switchable_smooth` of each.
- Squared loadings, not first powers: with θ_g ~ N(0, τ² score_g) independent and
  β_v = Σ_g c_vg θ_g, Var(β_v) = τ² Σ_g c_vg² score_g exactly. The first-power form
  f(Σ overlap · pHaplo) of review-theory's §4.3 is its whole-gene special case
  (c² = c = 1). e2e concurs with this derivation; the lead's ruling is pending.
- **Known approximation:** the term matches each SV's marginal variance only. The
  scale-mixture prior treats effects as independent given their scales, so Theorem
  5's covariance τ² c c' between SVs of one gene is dropped. The cis unit column
  carries it exactly.

**Switched off by its own weights.** `switchable_smooth` splits the spline into the
linear direction, with a ridge weight, and the curvature directions, whose exact
∫f''² penalty is full rank on them. Every direction is penalized:

| weights at ∞ | the term |
|---|---|
| all four | exactly 0: the model without it |
| loss (or gain) linear and curvature | that direction switched off, the other kept |
| curvature only | exactly linear in the burden, the first-order dosage model |
| none | a learned monotone-or-not smooth |

The linear ridge introduces no scale: a Gaussian with a learned precision on one
coefficient is invariant to rescaling its column.

**Identified against what is already there.** The class levels absorb any per-class
constant, and the burden is not a function of an "overlaps an exon" indicator (a
test checks the rank increase). So the term is identified wherever φ or the score
varies among exon-covering CNVs of a class.

**Not built:**
- coding breaks by INV, INS or MEI breakpoints (c = −1 in Theorem 5);
- a gain/loss asymmetry inside one copy-number column (review-theory's #6 φ(CN)
  smooth);
- the score tables' fetch and provenance.

Evaluation is bench-sim's sealed sv_gene_dosage out-of-family scenarios and
bench-real's held-out stratum, pooled arm, run by the lanes that own them.

## Shape: tested, never assumed

`shape_functionals` returns the degree-2 and degree-3 polynomials in t = log(u s),
orthogonalized against {1, t} under the fitted weights. Those two directions are
exactly what a scale term reaches (normalization and translation), so what remains
is what only a shape change can produce. `shape_score` is the per-variant score of
scale_model.md §6,

    U_j = a_j (Σ_k q_jk φ_k − Σ_k w_k φ_k),

from the responsibilities the M-step already forms. A translated density scores
zero and a reweighted one does not, which the tests assert.

The nested generalization, if the test fires, is log g(t | d) = log g_0(t − m(d)) +
Σ_k d_k δ_k(t) with each δ_k penalized and constrained to that complement: at an
infinite weight it returns scale-only exactly, so the test and the model agree.
Its calibration needs the identified-subspace restriction of §6; the fitted-nuisance
forms failed without it.

## What this module does not do

- It does not fit anything. Every weight and coefficient is the engine's.
- It does not choose a knot spacing: the caller halves the spacing until the
  evidence stops moving, as the lattice does (SPEC 131b205).
- It does not decide whether a term is adopted. That is an ablation on the
  benchmarks, coordinated with lane `ablate`.
