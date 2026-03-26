# SV-PGS

Single-device empirical-Bayes polygenic scoring with support for SNVs and structural variants.

## All of Us Quickstart

All of Us terminals may default to Python 3.14, which does not currently have a compatible `jaxlib` wheel. Create the environment with Python 3.12 explicitly.

```bash
cd ~ && rm -rf SV-PGS && git clone https://github.com/SauersML/SV-PGS.git && cd SV-PGS && uv venv --python 3.12 && uv sync --python 3.12 && source .venv/bin/activate && uv run sv-pgs prepare-all-of-us-disease --disease heart_failure --output heart_failure.samples.tsv && uv run sv-pgs run --genotypes ../arrays.bed --sample-table heart_failure.samples.tsv --target-column target --covariate-column age_at_observation_start --covariate-column gender_concept_id --covariate-column race_concept_id --covariate-column ethnicity_concept_id --output-dir heart_failure_results
```

This assumes your genotype files are in your home directory as `arrays.bed`, `arrays.bim`, and `arrays.fam`. The phenotype step uses the active All of Us workspace dataset from `WORKSPACE_CDR` and the official All of Us genomics manifests to map EHR `person_id` values onto the genotype sample IDs. The training step auto-detects `sample_id`, `research_id`, or `person_id` columns in the sample table.

To see available built-in disease names:

```bash
uv run sv-pgs list-all-of-us-diseases
```
