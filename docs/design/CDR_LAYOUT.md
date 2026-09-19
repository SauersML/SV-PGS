# Where the All of Us inputs live (Controlled Tier CDR)

This was recorded from the deleted `aou_runner.py` (tag `archive/2026-09-19/old-path-final`). The source of truth is the "Controlled CDR Directory" article in the AoU User Support hub; re-check it against the attached CDR version.

- **Root:** `CDR_STORAGE_PATH`, which the workbench sets (v8: `gs://fc-aou-datasets-controlled/v8`).
- **Genetic PCs and predicted ancestry:** `<root>/wgs/short_read/snpindel/aux/ancestry/`.
  - The file is `echo_v4_r2.ancestry_preds.tsv`, or `ancestry_preds.tsv` in earlier releases.
  - Each `research_id` has its predicted continental ancestry and a 16-dimensional `pca_features` list. `cohort.AncestryPcs.read` reads this file.
- **Kinship pairs:** `<root>/wgs/short_read/snpindel/aux/relatedness/`. These feed `cohort.kinship_components`.
- **Other aux directories:**
  - `vat/` (variant annotations), `admixture_estimates/`, `pgx/`, `phasing/`;
  - `qc/` (sample QC flags and metrics).
- **GATK-SV short-read SVs:** `<root>/wgs/short_read/structural_variants/vcf/full/AoU_srWGS_SV.v8.chr{N}.vcf.gz` (+ `.tbi`).
- **srWGS SNP/indel call sets:** `<root>/wgs/short_read/snpindel/`. Each call set comes as Hail MT, VCF, PLINK, BGEN and PGEN:
  - `vds/`, the full sparse joint call set;
  - `acaf_threshold/`: AF > 1% or AC > 100 in some ancestry;
  - `exome/`, `clinvar/`, `cmrg/`. Don't intersect `cmrg/` with the others: it is called against a masked reference.
- **Manifests:**
  - CRAM: `<root>/wgs/cram/manifest.csv`;
  - array: `<root>/microarray/…`;
  - long-read: `<root>/wgs/long_read/manifest.csv`;
  - known-issue sample lists: `<root>/known_issues/`.
