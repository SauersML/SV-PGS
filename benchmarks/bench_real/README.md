# bench-real: MAGE cis-expression prediction with 1kGP SNV/indel and SV genotypes

The real-data benchmark that no method lane controls (scratchpad EVIDENCE_RULE). Public data only; nothing from All of Us.

## Sources (all public; nothing from All of Us)
| Data | Version and location | Integrity |
|---|---|---|
| Expression, covariates, sample metadata | MAGE v1.0 (Taylor et al. 2024, Nature 631:610), Zenodo record 10535719, `MAGE.v1.0.data.zip`, CC-BY-4.0. 15 members are extracted by HTTP range reads, one request per member (`fetch_mage.py`). | zip md5 9b32d1e24aa883b3dc57359598420203; per-member CRC32 and sha256 in `data/mage/PROVENANCE.json` |
| Genotypes | 1kGP 3,202-sample phased SNV/INDEL/SV panel, NYGC high coverage, `20220422_3202_phased_SNV_INDEL_SV` (Byrska-Bishop et al. 2022, Cell 185:3426), from the EBI FTP over http | every file md5-checked against `20220804_manifest.txt`; `public/kgp_phased_panel_20220422/md5_checked.txt` |
| Pedigree, populations | `20130606_g1k_3202_samples_ped_population.txt`, from the same FTP directory | — |
| Second SV source | HGSVC2 PanGenie genotypes of the 3,202 samples (Ebert et al. 2021), `20201217_pangenie_merged_bi_nosnvs.vcf.gz`: short-read genotyping of long-read-discovered SVs | — |

On MSI the benchmark root is `/scratch.global/sauer354/svpgs-team/bench-real/`, and the shared panel copy is `/scratch.global/sauer354/svpgs-team/public/kgp_phased_panel_20220422/`.

## Samples
- **Individuals:** 730 of the 731 MAGE lymphoblastoid-line individuals, one library each: AFR 196, AMR 113, EAS 141, EUR 141, SAS 139, across 26 populations.
- **Library swap:** MAGE v1.0 carries a swap, per the mccoy-lab README of 2026-03-27. SRR19762530 is labelled HG00237 but is NA11919, and SRR19762653 is the reverse.
  - Only SRR19762653 is in the 731-library analysis set, and HG00237's own library SRR19762247 is in it too. So the column labelled NA11919 holds HG00237's expression, and NA11919 has none.
  - `build_dataset.py` drops NA11919 and records this in `dataset/BUILD_NOTES.txt`. (If both swapped libraries were present, it would swap their expression and PEER columns instead.)

## Phenotype
- **Expression:** inverse-normal TMM expression, as used for MAGE eQTL mapping (GENCODE v38 genes that passed MAGE expression filtering). Autosomes only; chrX is excluded because it's haploid in XY samples.
- **Covariates:** MAGE's own eQTL covariates: sex, 5 genotype PCs, 60 PEER factors.
- **Adjustment:** the covariates are regressed out by OLS fitted on the training samples only, and test expression is adjusted with that training fit.
- **Caveat:** MAGE computed the PEER factors and PCs on all samples. That use is unsupervised (expression-only or genotype-only) and identical for every method.

## Genotypes and variant annotation
- **Dosages:** alternate-allele counts 0/1/2, one row per ALT allele. A site is kept if it's polymorphic among the 731, and each fold then drops sites that are monomorphic in its training set.
- **SV definition:** a symbolic ALT, or an allele of at least 50 bp. That's the field-standard size definition (Mahmoud et al. 2019; HGSVC).
- **Annotations passed to methods:** position, end, signed distance to the TSS (0 when an SV overlaps it), is_sv, SV type, SV length, allele-length change, and training allele frequency.

## Window
A variant is included if its interval overlaps TSS ± 1 Mb, the cis window of MAGE's own mapping and of GTEx (GTEx Consortium 2020).
- **This is an external standard, not derived from the data.** A data-derived cut-off (e.g. where pooled excess χ² reaches zero) needs a significance threshold or a random-walk argmax, and neither is well posed.
- **What methods do instead:** they get distance to the TSS as an annotation, and can learn how signal decays.

## Splits (sealed; `dataset/splits.json` and its sha256 in `splits.sha256`)
- **random5:** 5 folds. Whole 1kGP families stay in one fold, and each superpopulation is spread evenly over the folds. The seed comes from sha256("bench-real/random5").
- **loso:** leave one superpopulation out (5 splits). Train on four superpopulations, test on the fifth. This is the portability test.

## Harness contract
- **The method:** `fit(train: TrainData) -> predictor`, where `predictor.predict(test_genotypes) -> ndarray`.
- **What TrainData holds:** training genotypes, the adjusted phenotype, variant annotations, and superpopulation and population labels.
- **Sealing:** test phenotypes never reach method code; the harness reads them only to score.
- **Feature sets:** `snv` (panel SNVs and indels under 50 bp), `snv_sv` (all panel rows), and `snv_pgsv` (panel SNVs/indels plus PanGenie SVs).
- **Submitting:** a method lane sends a file and callable (`path.py:callable`), and bench-real runs it on the sealed splits. Lanes don't run the benchmark themselves.
- **Costly methods:** they run on a sealed random gene sample, a prefix of `dataset/gene_order.tsv` (a seeded permutation of all genes), via `--gene-prefix N`.
- **Output:** out-of-fold predictions for every gene and sample, per design and feature set, plus per-fit CPU seconds and variant counts.

## Scoring
- **r²:** per gene and superpopulation, the squared Pearson correlation of prediction with adjusted held-out expression; 0 for a constant prediction.
- **Pairs:** paired differences between arms, averaged over genes.
- **SE:** the delete-one-chromosome jackknife, since genes on one chromosome share variants. On a single chromosome the SE is gene-level and labelled as such.

## Baselines
- **top_variant:** the lead marginal variant, fitted by OLS.
- **gblup_reml:** REML h², with the GLS intercept.
- **mr_ash:** a faithful port of mr.ash.alpha 8e257fd with its published defaults, cited in `baselines.py`.
- **The engine's current design** is added when it lands on main.

The baselines' math checks are in `tests/test_baselines.py`: REML optimality against direct REML, dual/primal BLUP identity, sparse recovery, and lead-variant choice.

## Leakage rules
- MAGE's published eQTL, fine-mapping and colocalization results were computed on all 731 samples, including every test fold. **They must never be used as features, priors or gene filters.**
- Gene strata may only come from training-fold quantities, e.g. GBLUP-REML h² on the training fold.

## Limits (what this benchmark is and is not)
- **Small n** (about 585 training samples in random5, 535–618 in loso) and a sparse cis architecture. It's closer to fine-mapping than to polygenic traits, so it tests prior shape, SV credit and portability in the large-effect, few-causal regime. It says little about the polygenic tail.
- **Genotypes are direct high-coverage calls,** not imputed. There's no imputation-reliability channel, so the r² offset and fusion terms of the model aren't exercised. The PanGenie arm (short-read genotyping of long-read SVs) is the nearest public proxy.
- **Tandem repeats aren't annotated** in the panel, so the TR signed-length term isn't tested.
- **Lymphoblastoid-line expression** is a molecular phenotype. Its architecture (strong cis, larger SV enrichment) differs from complex traits.

## Running it
The baselines need numba, which isn't a repository dependency. Run in a venv with numpy, scipy, pandas, cyvcf2 and numba, and bcftools on PATH. From the repository root:

    python benchmarks/bench_real/fetch_mage.py <root>/data/mage
    benchmarks/bench_real/fetch_genotypes.sh <root> <public-dir> <threads>
    python benchmarks/bench_real/build_dataset.py --root <root> --chromosomes 1 2 ... 22
    python benchmarks/bench_real/splits.py <root>/dataset
    python benchmarks/bench_real/harness.py --dataset <root>/dataset --method benchmarks/bench_real/baselines.py:gblup_reml         --name gblup_reml --design loso --chromosomes chr22 --out <root>/results --workers <n>
    python benchmarks/bench_real/report.py --results <root>/results --dataset <root>/dataset --methods gblup_reml mr_ash --out <report dir>

Set OMP_NUM_THREADS=1 (and the OpenBLAS, MKL and numba equivalents) when running several workers.

The math checks run with `python -m pytest benchmarks/bench_real/tests`. They sit outside the CI testpaths because of numba.
