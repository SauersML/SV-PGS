# Pre-registration: the SV-PGS portability comparison on bench-real

critic-real, 2026-09-20. Written before any SV-PGS portability result exists: fit-api's `fit_expression` had not been delivered, and no SV-PGS prediction had been scored.

**Ratified by the lead** on 2026-09-20, with two edits, both applied below:
1. The comparator is compete's `mr_ashr_init` arm, the lasso-started mr-ash-workflow arm and the primary competitor. bench-real's numba mr.ash port is secondary.
2. This is a SECONDARY family, next to ablate's Amendment 2 headline H: the pooled Δr² of SV-PGS − mr_ashr_init, loso, on the confirmation genes. It takes its own α from the same alpha-spending scheme, set with ablate.

Changes need a dated amendment that says why, and nobody may read an SV-PGS portability number before the amendment is made.

## Primary metric
**AFR portability ratio,**
`P = Σ_g r²f(AFR | loso) / Σ_g r²f(AFR | random5_king)`,
where r²f is the floor-corrected r² in robust.py: `(r² − 1/(n−1)) / (1 − 1/(n−1))`, with n the number of AFR people.
- The numerator is each gene's accuracy in AFR when all of AFR is held out.
- The denominator is the same people's accuracy under the KING-cluster random folds, where the training set includes AFR.
- The sums run over genes, so a ratio of sums, as in Saitou et al. 2024.
- The denominator is the group's own within-ancestry accuracy, not the other groups', so it measures ancestry transfer and not between-group differences in heritability.

## Primary comparison of this family
**SV-PGS minus mr_ashr_init** in P, on the same genes and the same people. mr_ashr_init is compete's lasso-started mr-ash-workflow arm at the workflow's settings. SV-PGS minus bench-real's numba mr.ash (mr.ash.alpha published defaults) is reported as secondary.
- The 95% CI is the pigeonhole bootstrap (robust.py; families within held-out groups × chromosomes). It is conditional on training, and robust.py states this.
- The inflation factor from train_resample.py is reported alongside, but doesn't enter the primary interval.

## Arms and gene sets
- **Arms:** the snv and snv_sv feature sets, for both methods. The primary result uses snv_sv; snv is secondary.
- **Genes that support claims:** the ranked-list genes never scored before (bench-real's held-out stratum) and, on the lead's call, the sealed 3,504 confirmation genes.
- **Development genes:** the 2,000-gene prefix. Numbers on these are reported, labelled development.

## Secondary metrics (reported, not used to decide)
1. **Pooled floor-corrected r², by held-out group,** and the paired SV gain (snv_sv − snv) per group.
2. **The slope of per-population accuracy against genetic distance:** mean floor-corrected r² over genes against mean PC distance to the loso training centroid, across the 26 populations (portability.py).
3. **P within Saitou's driver-frequency classes** (MAF 5% boundary), with drivers taken from each method's own loso fit.
4. **Location and scale:** oos_r2_centred per group, plus the location split of raw out-of-sample R². Raw out-of-sample R² under loso is never used as a metric; it is dominated by the adjusted truth's group mean offset.

## Decision rule and error rate
- **This is a secondary family,** tested after ablate's headline H (pooled Δr² of SV-PGS − mr_ashr_init, loso, confirmation genes) in a fixed sequence.

**Amendment 1 (2026-09-20, before any SV-PGS portability result exists; set with ablate, confirmed by the lead):** the error rate follows ABLATION_PLAN Amendment 3 (lane/ablate-prereg-a3 7bd3875).
- **Version:** P shares H's version v, meaning the same fit and adapter shas. It takes no separate place in the alpha-spending order.
- **Level:** α_v = 0.05·2^−(v+1), per ABLATION_PLAN Amendment 2 (acb1e5b).
- **The claim rule:** SV-PGS is said to port better than mr_ashr_init only if H rejects at α_v in version v **and** the two-sided (1 − α_v) interval of the primary comparison excludes 0, on claim-supporting genes. Otherwise the result is reported as no detectable difference, with its interval.
- **Error rate:** the fixed sequence keeps FWER over {H, P} ≤ α_v and leaves H's level unchanged.
- **Timing:** P's confirmatory value is computed only at version v's confirmation run, together with H.
- **Other secondaries:** they spend no α.
