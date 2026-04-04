# SV-PGS

Single-device empirical-Bayes polygenic scoring with support for SNVs and structural variants.

SV-PGS is intended to run fully on CUDA. If `uv run python -c "import jax; print(jax.default_backend(), jax.devices())"` does not report a GPU backend, do not start training yet.

## All of Us Quickstart

All of Us terminals do not have `uv` pre-installed, and may default to Python 3.14 which lacks a compatible `jaxlib` wheel. The command below installs `uv`, creates a Python 3.12 environment, copies a single-chromosome SV VCF from the controlled-tier bucket, prepares a disease phenotype from BigQuery, and runs the PGS model.

```bash
cd ~ && rm -rf SV-PGS && git clone https://github.com/SauersML/SV-PGS.git && cd SV-PGS && curl -LsSf https://astral.sh/uv/install.sh | sh && export PATH="$HOME/.local/bin:$PATH" && uv venv --python 3.12 && uv sync --python 3.12 --extra gpu && source .venv/bin/activate && gsutil -u $GOOGLE_PROJECT cp "${CDR_STORAGE_PATH}/wgs/short_read/structural_variants/vcf/full/AoU_srWGS_SV.v8.chr22.vcf.gz" "${CDR_STORAGE_PATH}/wgs/short_read/structural_variants/vcf/full/AoU_srWGS_SV.v8.chr22.vcf.gz.tbi" . && uv run sv-pgs prepare-all-of-us-disease --disease type2_diabetes --output t2d.samples.tsv && uv run sv-pgs run --genotypes AoU_srWGS_SV.v8.chr22.vcf.gz --genotype-format vcf --sample-table t2d.samples.tsv --target-column target --covariate-column age_at_observation_start --covariate-column gender_concept_id --covariate-column race_concept_id --covariate-column ethnicity_concept_id --output-dir t2d_sv_chr22_results
```

Before the long `sv-pgs run`, verify the runtime:

```bash
uv run python -c "import jax; print('backend', jax.default_backend()); print('devices', jax.devices())"
uv run python -c "import cupy as cp; print('cupy_devices', cp.cuda.runtime.getDeviceCount())"
```

The SV VCFs are sharded by chromosome under `${CDR_STORAGE_PATH}/wgs/short_read/structural_variants/vcf/full/`. Chr22 (~1 GB compressed, ~26K variants, ~97K samples) is a good starting point. Replace `chr22` with another chromosome or copy multiple shards as needed.

The phenotype step uses the active All of Us workspace dataset from `WORKSPACE_CDR` and writes `sample_id = person_id`. The training step auto-detects `sample_id`, `research_id`, or `person_id` columns in the sample table.

To see available built-in disease names:

```bash
uv run sv-pgs list-all-of-us-diseases
```
