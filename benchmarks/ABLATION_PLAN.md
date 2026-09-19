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
