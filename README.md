# SV-PGS

Bayesian polygenic scoring for structural variants. Fits a joint empirical-Bayes GLM on GPU via CuPy (cuBLAS) with JAX for element-wise ops.

## All of Us Quickstart

**First-time setup** (installs uv + Python 3.12 + GPU dependencies):

```bash
cd ~ && rm -rf SV-PGS && git clone https://github.com/SauersML/SV-PGS.git && cd SV-PGS \
  && curl -LsSf https://astral.sh/uv/install.sh | sh && export PATH="$HOME/.local/bin:$PATH" \
  && uv sync --python 3.12 --extra gpu
```

**Run a full analysis** (downloads VCFs, prepares phenotype, merges PCs, fits per-chromosome):

```bash
cd ~/SV-PGS && uv run sv-pgs run-all-of-us --disease hypertension --output-dir hypertension_results
```

That single command:
1. Downloads all 22 chromosome SV VCFs from the controlled-tier bucket (skips existing)
2. Downloads ancestry predictions and merges top 10 genomic PCs
3. Queries BigQuery for the disease phenotype (ICD-9/10 codes built-in)
4. Fits the Bayesian PGS model per chromosome on GPU
5. Covariates: age, age^2, sex at birth, race, ethnicity, PC1-PC10
6. Skips already-completed chromosomes (restartable)

**Available diseases:**

```bash
uv run sv-pgs list-all-of-us-diseases
```

asthma, atrial_fibrillation, chronic_kidney_disease, copd, coronary_artery_disease, depression, heart_failure, hypertension, stroke, type2_diabetes

**Options:**

```bash
# Single chromosome:
uv run sv-pgs run-all-of-us --disease type2_diabetes --chromosomes 22 --output-dir t2d_chr22

# Specific chromosomes:
uv run sv-pgs run-all-of-us --disease heart_failure --chromosomes 1,6,22 --output-dir hf_results

# More PCs:
uv run sv-pgs run-all-of-us --disease depression --n-pcs 20 --output-dir depression_results
```

## Generic usage (non-AoU)

```bash
uv run sv-pgs run \
  --genotypes input.vcf.gz \
  --sample-table phenotypes.tsv \
  --target-column target \
  --covariate-column age \
  --covariate-column sex \
  --output-dir results
```

## Verify GPU runtime

```bash
uv run python -c "import jax; print('backend', jax.default_backend()); print('devices', jax.devices())"
uv run python -c "import cupy as cp; print('cupy_devices', cp.cuda.runtime.getDeviceCount())"
```

## Data

SV VCFs are sharded by chromosome under `${CDR_STORAGE_PATH}/wgs/short_read/structural_variants/vcf/full/`. Ancestry predictions with PCs are at `${CDR_STORAGE_PATH}/wgs/short_read/auxiliary/ancestry/ancestry_preds.tsv`. The `run-all-of-us` command handles all downloads automatically.
