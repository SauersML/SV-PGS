# SV-PGS term ablations: pre-registration v1 (frozen before any ablation result exists)

**Purpose:** decide with evidence which terms of the one model (docs/design/MODEL.md) stay. Every adopt, keep or remove decision below follows from the rules in this file, which are fixed before any ablation is run. Amendments are appended and dated; they are never edited in place, and each amendment states whether any ablation result existed when it was made.

**Evidence labels:** bench-real results are [real] (MAGE expression, public 1kGP genotypes). bench-sim results are [semi-real] (real public haplotypes, real imputation, simulated phenotypes; v7 cohort). Nothing from All of Us is used.

## 1. How an ablation is built
- **The full model (F)** is the engine's fit/score entry point at a given main sha, run through each benchmark's own harness: `benchmarks/bench_real/harness.py` and `benchmarks/bench_sim/SUBMIT.md`.
- **An ablated arm (F − t)** removes one term t in the way its own structure defines, with no new constant:
  - by withholding the term's inputs (its columns or features), or
  - by fixing its learned penalty at the null the model already nests (λ = ∞, or Γ = 0).
  - The adapter doing this lives in `benchmarks/` and is evaluation-only. The production code gains no option (SPEC: one model).
- **Leave-one-term-out from F** is the primary design, because the terms interact. One forward design is added for SV columns only (a minimal base plus SV columns), since that is the project's headline question.
- **Matched compute:** each arm runs under its benchmark's fixed per-scenario or per-gene limits. Wall time and peak memory are reported for every arm.

## 2. The terms (t1–t10)
| id | term | ablated arm (F − t) | benchmarks |
|---|---|---|---|
| t1 | SV columns (SV and TR records as regressors) | SNV and indel columns only | both |
| t2 | learned mixing density g | g fixed at its λ = ∞ limit (the log-normal in log s, fitted globally) | both |
| t3 | class pooling of g (η + δ_c) | independent per-class densities, i.e. no shared η | both |
| t4 | frequency term in the scale model | frequency smooth removed | both |
| t5 | annotation-dependent shape (Γ) | Γ = 0 (scale-only annotations) | both |
| t6 | SV-context distance kernel | kernel features withheld | both |
| t7 | A-map leakage correction and D* recalibration | uncorrected stored dosage columns | bench-sim only (see §6) |
| t8 | TR smooth length function | the linear signed-length column (its λ = ∞ limit) | both, where TR loci exist |
| t9 | gene-dosage sharing (β_v = c_v θ_m + ε_v) | independent variant effects | both |
| t10 | fixed shape families, as reference only | g replaced by TPB-like and BayesR-like fixed shapes | both; descriptive only, never a decision |

## 3. Metrics
- **Primary, one per benchmark:**
  - bench-real: per-gene held-out r² within superpopulation, averaged over genes;
  - bench-sim: incremental held-out R² over covariates and PCs, on the liability scale for binary traits (Lee et al. 2012).
- **Secondary, reported for every arm and never used alone for a decision:**
  - calibration slope of y on the prediction, adjusted for covariates and PCs;
  - SV credit: bench-sim's error of the TR+SV genetic-variance share against the truth share; bench-real's share of held-out predictive covariance carried by SV columns;
  - AUC for binary traits;
  - R² per ancestry group;
  - wall time and peak memory.

## 4. Paired designs and standard errors
- **bench-sim:**
  - the 64 sealed test scenarios, each arm run once per method version;
  - per scenario, Δ = metric(F) − metric(F − t);
  - the estimate is the mean Δ over scenarios, with SE = sd(Δ)/√64, since scenarios are independent draws from the pre-registered family;
  - an exact sign test on the per-scenario Δ is reported alongside, as a robustness check.
- **bench-real:**
  - both sealed designs, random5 and loso, reported separately;
  - Δ per gene and superpopulation, averaged over genes;
  - SE by the delete-one-chromosome jackknife, which is the harness default for genome-wide runs;
  - loso is reported per held-out superpopulation.
  - Costly arms run on the sealed gene-order prefix (`--gene-prefix N`), and N is recorded.
- **The development rule:** all tuning and debugging happens on bench-sim's 24 dev scenarios and on bench-real's random5 chr22 genes. The sealed bench-sim tests run once per method version.
  - A re-run after a code change is a new version. It is reported as such, and it enlarges the multiplicity family in §5.

## 5. Multiplicity control
- **The decision family:** the primary one-sided tests of benefit (Δ > 0) for t1–t9 on each applicable benchmark (bench-real loso and random5 count as two members), plus the matching one-sided tests of harm (Δ < 0).
- **The level:** family-wise error rate 0.05, controlled by Holm's step-down procedure over the whole family. The 0.05 level is the pre-registered convention of this plan, an external standard, not a model constant.
- **Descriptive analyses,** reported without decisions: the secondary metrics, t10, and breakdowns by bench-sim truth mode (shape family, SV enrichment mode, frequency form, annotation mode, trait type).

## 6. Decision rules (fixed now)
- **ADOPT t:** its benefit test is significant after Holm on at least one benchmark, AND its harm test is not significant on any benchmark.
- **REMOVE t:** its benefit test is significant nowhere. Parsimony follows the project rule that a term is adopted only on a measured win.
- **REJECT t:** its harm test is significant on any benchmark. The term is removed, and the regime where it hurts is recorded.
- **Scope of the evidence:**
  - bench-real's small n and sparse cis architecture test prior shape, SV credit and portability in the large-effect regime; they say little about the polygenic tail;
  - bench-sim covers the polygenic range;
  - a decision records which benchmark carried it.
- **t7 needs a truth-labelled subset.** It is decided on bench-sim only, and only once the harness offers an arm that exposes true genotypes for a training subset, mirroring the long-read half. Until then t7 stays design-derived (MODEL.md §2) and untested here.
- **The final ruling is the lead's, given these results.** No term is adopted or removed without the evidence this file specifies.

## 7. What gets reported
For every arm and benchmark:
- mean Δ with its SE and Holm-adjusted p-value, and the sign test;
- every scenario's or gene set's Δ, losses included;
- the secondary metrics;
- the compute;
- the main sha of F and the adapter sha.

Results are written to `benchmarks/ablations/` with the benchmark's evidence label.

## Amendment 1 (2026-09-19, before any ablation result exists): the arm that tests t7
- **The arm:** t7 (the A-map and D* recalibration) is tested on bench-sim's `beagle_truthhalf` measurement arm, PREREG amendment 8, lane/bench-sim-truthhalf f235eda. Its flagged 20% of each group's training samples are observed at their true genotypes, and the harness exposes this as `train.truth_half`.
- **The 20%** is bench-sim's stated design choice, not a production quantity.
- **Only the ablation adapter reads `train.truth_half`,** to estimate the stored-to-true map Σ_DG. F − t7 ignores it and uses the uncorrected stored columns.
- **The t7 decision rule** is unchanged from §6, and applies to bench-sim only.

## Amendment 2 (2026-09-19, before any ablation result and before any sealed or confirmation run): versions, confirmation genes and the headline test
This amendment adopts review-stats' STATS_REVIEW §5. It replaces the sentences of §4–§5 named below, and everything else stands.

### Versions and alpha-spending
This replaces §4's sentence that a re-run "enlarges the multiplicity family", and §5's single level.
- **A version** v = 0, 1, 2, … is a frozen pair: the main sha of F and the adapter sha. Versions are numbered in the order their confirmatory runs start.
- **The level:** version v's decision family is the §5 family, fixed before its run. Holm controls it at FWER α_v = 0.05 · 2^−(v+1).
  - Σ_v α_v ≤ 0.05 however many versions run, so the error rate holds even though later versions are chosen after earlier results are seen.
  - The spending sequence is fixed here.
- **Each decision stands at its own version's level.** An earlier version's results are reported, never pooled with a later version's.

### bench-real decisions use the sealed confirmation genes
This replaces §4's bench-real design wherever a decision is made.
- **Which genes:** every bench-real ADOPT, REMOVE or REJECT test is computed once per version on the 3,504 sealed confirmation genes, `dataset/sealed_confirmation_genes.tsv` (sha256 5c0d5f1c…).
  - Those genes are scored only under `--confirmation`, and only when the lead calls it.
  - Every arm runs on all 3,504 genes, with no prefix or subset.
  - random5 and loso remain separate family members.
- **The SE** is the crossed bootstrap of `benchmarks/bench_real/robust.py` (Owen 2007): families within each held-out group, crossed with chromosomes. The delete-one-chromosome jackknife of §4 is kept for screening only.
- **The non-sealed genes,** meaning the random5 and loso development runs including the gene-order prefix, are for screening and development only. No decision rests on them.

### The headline test H: one primary test
- **The claim:** SV-PGS predicts held-out expression better than the primary competitor.
- **The test:** one-sided, Δ > 0, on the pooled paired Δr² of SV-PGS (F at version v) minus `mr_ashr_init`.
  - It runs on loso, over the sealed confirmation genes, with the crossed-bootstrap SE.
  - Pooled means the mean over genes of the mean over the five held-out groups, as in robust.py.
- **The columns:** both arms use the `snv_sv` feature set, so H compares methods on the same columns.
- **The competitor:** `mr_ashr_init` is mr-ash-workflow's mr.ash.init through mr.ashr 0.1-90 (3d65ce3), as `benchmarks/compete` specifies. That's on lane/compete-harness, not yet on main.
- **The level:** H is its own family of one per version, at α_v. It is a separate claim from the term decisions of §6, so it's outside their Holm family.
- **Everything else is secondary** and reported without a claim: mr_ashr_both and the other competitors, random5, other feature sets, results per group, and bench-sim.
