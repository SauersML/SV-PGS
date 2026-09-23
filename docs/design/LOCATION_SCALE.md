# A location-scale mixing law for oriented alleles (design note, not implemented)

Status: the evidence gate below passed, so this note specifies the change. Implementation is held until the EP route
settles and goes to one owner of the per-node kernels (lead ruling, 2026-09-23): `scale_mixture_ep._kernel_terms`,
`_KernelRows.derivatives`, `tilted_moments`, `tilted_cumulants`, `quadrature_majorant_ratio`, the device kernel
`engine_kernels.tilted_moments`, and the mean-field sweep and draws (`mean_field._sweep`, `_sample_nodes`).

## Why: SV effects on their own gene are sign-skewed [real]

The current law, beta ~ sum_k w_k N(0, u e^{t_k}), is symmetric and has its maximum at zero. It cannot express "a
deletion of this gene's exon tends to lower its expression". An SV's ALT allele is the event's presence, so its
orientation is biological. For a SNV, REF/ALT orientation is arbitrary.

The test (`benchmarks/bench_real/sv_sign_skew.py <worktree>`; bench-real genes500, loso/AFR training people only, 534 people,
66,397 DEL/INS/DUP SVs in 500 cis windows) used each SV's marginal association z with its gene's
covariate-residualized expression. The null permutes expression across genes (2,000 derangements), which keeps each
window's LD. Mean z by class and overlap:

| SVs | n | mean z | null 95% | perm. p | gene-bootstrap mean z (95% CI) |
|---|---|---|---|---|---|
| DEL overlapping an exon | 78 (42 genes) | -1.33 | [-0.28, +0.25] | 0.0005 | -1.20 [-2.52, -0.21], 28 of 42 genes negative |
| INS/DUP overlapping an exon | 93 (75 genes) | +0.66 | [-0.22, +0.21] | 0.0005 | +0.72 [+0.25, +1.30] |
| DEL in the gene body, no exon | 866 | +0.05 | [-0.07, +0.08] | 0.22 | +0.16 [-0.15, +0.48] |
| INS/DUP in the body, no exon | 737 | +0.01 | [-0.08, +0.08] | 0.85 | -0.21 [-0.47, +0.06] |
| outside the gene | 64,623 | ~0 | | > 0.4 | |

p 0.0005 is the smallest value 2,000 permutations can give. The skew is confined to exon overlap, and its sign
depends on the class: removing exonic sequence lowers expression, adding it raises it. A symmetric law puts equal
prior mass on both signs there.

## The family

    beta_j ~ sum_{k,l} pi_{c,kl}(j) N(sqrt(u_j) mu_l, u_j e^{t_k})

- The (t, mu) lattice is K x L: the existing log-variance axis, plus a location axis mu_l = l delta, l = -M..M.
- The location scales with sqrt(u_j), so u stays the effect's scale. A component with mu_l = 0 is today's law exactly.
- Given the component, beta is still Gaussian. The scale sampler, alias convolution and every Gaussian solve apply
  unchanged, with the component carrying a mean.

**Class densities.** Write eta_c(k, l) = eta_sym(k, |l|) + a_c(k, l).
- eta_sym is shared by all classes, with the existing third-order roughness in t and a second-order roughness in mu.
- a_c is antisymmetric in l: a_c(k, -l) = -a_c(k, l).
- For a class whose orientation is arbitrary (SNV, and any REF/ALT-coded small variant), a_c = 0 by construction, so
  the reflection symmetry is exact and identified.
- For presence-oriented classes (DEL, INS, DUP, MEI, CNV gain/loss), a_c is a learned block. It carries roughness
  penalties plus a ridge toward 0 with a learned weight, and starts at the lambda = infinity edge, nested like the
  annotation groups: asymmetry enters only where the evidence rises.

**Annotation-driven skew.** The measured skew is carried by exon overlap. So the location axis also takes an
exponential tilt by the annotations:

    log pi_{c,kl}(j) = eta_c(k, l) + mu_l s_c(d_j) - log normalizer_j

- s_c(d) is a penalized linear/smooth function of variant j's annotations (exon overlap, distance, length), zero for
  arbitrary-orientation classes.
- The normalizer is per variant (one log-sum-exp over the K x L nodes).
- The tilt has the same form as today's theta, but acts on the location axis instead of log u.

**Allele flip.** Recoding a presence-oriented record's ALT/REF maps beta to -beta, which is l to -l. The scorer and
the store must carry each record's orientation so that a flipped record's law is the mirror image. For a symmetric
class the law is invariant under the flip.

## The per-node kernel (the derivation the implementation replaces)

Take a component N(m, v), with m = sqrt(u) mu_l and v = u e^{t_k}, and a cavity exp(-P beta^2 / 2 + h beta). With
q = vP and r = 1/(1 + q):

    log Z_kl = -1/2 log(1 + q) + 1/2 (h^2 v + 2 h m - P m^2) r
    E[beta | kl] = (v h + m) r,   Var[beta | kl] = v r

(complete the square: precision 1/v + P, linear term m/v + h).

The mixture's moments are the responsibility-weighted ones. Mean: sum w (vh + m) r. Variance: sum w v r plus the
responsibility-weighted variance of the component means. The third and fourth cumulants follow from
`tilted_cumulants`' formula with d_kl = E[beta | kl] - mean, since each component is still Gaussian.

**The derivative recursion in eta = log u.** This replaces `_components`' A_n / B_n recursion. Write
a = h^2 v r, d = P m^2 r, A = a - d and b = 2 h m r. With dq/deta = q, dm/deta = m/2 and dr/deta = -r(1 - r):

    dA/deta = A r,        db/deta = b (r - 1/2)

So every derivative has the form

    d^n log Z_kl / deta^n = A A_n(r) + b C_n(r) - B_n(r)

- A_n and B_n are today's (A replaces a).
- C_1 = (r - 1/2) / 2, and C_{n+1} = (r - 1/2) C_n - r(1 - r) C_n'.
- At mu_l = 0, b = d = 0 and this reduces to the current recursion exactly.

The derivatives in the location coefficients (eta_c(k, l), the tilt s_c) are the same log-sum-exp responsibilities
as for eta today, over K x L nodes instead of K.

**Rounding bound.** Each node's term, log pi - 1/2 log(1 + q) + 1/2 (a + b - d), rounds by about six ulps of the size
of its pieces (today four). Here b has either sign and can cancel against a - d, so the bound must use
|a| + |b| + |d|, not |a + b - d|.

**Quadrature on the location axis.**
- The t-axis floor (`kernel_floor`) bounds only the mu = 0 column. As v -> 0 a component with mu != 0 tends to a
  point at m, whose kernel exp(h m - P m^2 / 2) is not flat.
- The location range comes from the data: M delta sqrt(u_j) must cover the largest |h_j| / P_j, with the same
  a-posteriori extension check (`lattice_check`) at the fitted state.
- The spacing: the kernel in m is Gaussian with width sqrt(v + 1/P) >= 1/sqrt(P). The trapezoid error on the strip
  bound gives delta sqrt(u) <= pi sqrt(2 / (P_max ln(2 sum_j M_j / Z_j / tol))), the analogue of `spacing_bound`.

**Mean-field sweep and draws.**
- Each member's q_j is the K x L mixture tilted by its pseudo-likelihood (omega_j, shift_j). The sweep's per-member
  work grows from K to K L nodes.
- Draws pick a component (k, l) by its responsibility, then draw N((v h + m) r, v r).

## Cost

Per-node work scales by L. With the location axis resolved at the data's spacing, L is of order 2 ceil(max |z_j|) + 1
(tens at most). An arbitrary-orientation class needs the location axis only where its fitted symmetric density puts
mass off zero. The recursion adds one family (C_n) of the same cost as A_n.
