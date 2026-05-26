# SV-PGS

Bayesian polygenic scoring for structural variants. Fits a joint empirical-Bayes GLM on all visible GPUs via CuPy (cuBLAS) with JAX for element-wise ops.

## All of Us Quickstart

**First-time setup** (installs uv + Python 3.12 + GPU dependencies):

```bash
cd ~ && rm -rf SV-PGS && git clone https://github.com/SauersML/SV-PGS.git && cd SV-PGS \
  && curl -LsSf https://astral.sh/uv/install.sh | sh && export PATH="$HOME/.local/bin:$PATH" \
  && uv sync --python 3.12 --extra gpu
```

**Run a full analysis** (downloads VCFs, prepares phenotype, merges PCs, fits one unified genome-wide model):

```bash
cd ~/SV-PGS && uv run sv-pgs run-all-of-us --disease hypertension --output-dir hypertension_results
```

That single command defaults to `--variants snp+sv`, which fits one joint
model on BOTH genotype sources:
- AoU microarray PLINK SNPs (447k samples, ~700k variants) from the
  controlled-tier microarray PLINK trio (`.bed` / `.bim` / `.fam`)
- AoU srWGS structural variant VCFs (97k samples, ~1.7M variants) from the
  controlled-tier `structural_variants/vcf/full/` bucket

To restrict to one source pass `--variants sv` (SV VCFs only) or
`--variants snp` (microarray PLINK SNPs only).

That single command (with the default `--variants snp+sv`):
1. Downloads the microarray PLINK trio AND all 22 chromosome SV VCFs from the
   controlled-tier buckets (skips existing files); `--variants sv` skips the
   PLINK trio and `--variants snp` skips the SV VCFs
2. Downloads ancestry predictions and merges top 10 genomic PCs
3. Queries BigQuery for the disease phenotype (ICD-9/10 codes built-in)
4. Concatenates the requested chromosome data into one genome-wide training
   dataset, intersecting samples across the SNP and SV sources when both are
   loaded
5. Fits one Bayesian PGS model across all visible GPUs and requested chromosomes
6. Uses `--variant-metadata` annotations when supplied; it does not derive
   annotations from VCF INFO
7. Reuses an existing fit only when the full AoU run configuration (including
   the `--variants` choice) matches
8. Covariates: age, age^2, sex at birth, race, ethnicity, PC1-PC10

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
  --variant-metadata variant_metadata.tsv \
  --output-dir results
```

`--variant-metadata` is keyed by `variant_id` and drives the schema-based prior hypermodel. Apart from reserved model columns (`variant_id`, `variant_class`, `chromosome`, `position`, `length`, `allele_frequency`, `quality`, `training_support`, `is_repeat`, `is_copy_number`, `prior_class_members`, `prior_class_membership`), every column is treated as a user annotation. Column types are inferred from the values:

- boolean values (`true`, `false`, `1`, `0`, `yes`, `no`) become binary annotations
- numeric values become continuous annotations
- `level=weight` lists become weighted membership annotations
- `parent>child` values become nested annotations
- other strings become categorical annotations

Example:

```tsv
variant_id	coding	constraint	functional_state	regulatory_mix	gene_context
sv1	1	0.82	lof	enhancer=0.7,promoter=0.3	protein_coding>exon
sv2	0	0.15	missense	enhancer=0.2,promoter=0.8	protein_coding>intron
```

## Verify GPU runtime

```bash
uv run python -c "import jax; print('backend', jax.default_backend()); print('devices', jax.devices())"
uv run python -c "import cupy as cp; print('cupy_devices', cp.cuda.runtime.getDeviceCount())"
```

When two or more CUDA devices are visible, SV-PGS shards the resident genotype
cache by variant columns and runs CuPy matmul shards concurrently across the
devices. With two comparable GPUs, genotype matmul-heavy phases should approach
2x single-GPU throughput once the cache is resident.

## Quickstart

Once `uv sync --extra gpu` has finished, two short commands confirm that the
GPU pipeline is wired end-to-end.

```bash
# 1) Smoke check: builds a tiny synthetic BED and exercises screening / matvec
#    / rmatvec / gram_block against the CPU reference. Expects "BITPACKED
#    PIPELINE OK" + exit 0. Takes <5s on V100/T4.
uv run python -m sv_pgs.bitpacked.smoke

# 2) Bench harness: detects the active GPU, prints HBM, runs gemv_nt /
#    gemv_tn / gemm_gram / screen at three scales, emits a markdown table +
#    optional JSON report. --quick runs two scales for sub-30s CI.
uv run python -m sv_pgs.bitpacked.bench --output bench.json
uv run python -m sv_pgs.bitpacked.bench --quick   # faster CI invocation
```

The bench output looks like::

    === sv-pgs bitpacked benchmark on Tesla V100-SXM2-16GB sm_70 family=volta HBM=16.9 GB ===
    HBM total: 16.9 GB, free at start: 16.6 GB

    | Op        | n_samples | n_variants | bytes_GB | time_ms | GB/s   | TFLOPS |
    |-----------|-----------|------------|----------|---------|--------|--------|
    | gemv_nt   |     97000 |       4096 |    0.099 |    0.99 |  100.1 |      - |
    | gemv_tn   |     97000 |       4096 |    0.099 |    0.75 |  132.4 |      - |
    | gemm_gram |     97000 |       4096 |    0.099 |  344.25 |      - |   4.73 |
    | screen    |     97000 |       4096 |    0.099 |    6.08 |   16.3 |      - |

## Data

SV VCFs are sharded by chromosome under `${CDR_STORAGE_PATH}/wgs/short_read/structural_variants/vcf/full/`. Ancestry predictions with PCs are at `${CDR_STORAGE_PATH}/wgs/short_read/snpindel/aux/ancestry/ancestry_preds.tsv`. The `run-all-of-us` command handles all downloads automatically.

## Troubleshooting

### "Bitpacked path SKIPPED in logs"

The bitpacked GPU path is the fast path. If the training log shows
`bitpacked upgrade: SKIPPED`, scan for the parenthetical reason:

```
bitpacked upgrade: SKIPPED (reason: ...)
```

Common reasons:
- `config.genotype_backend=...` is not `'bitpacked'` — set it explicitly.
- `CuPy unavailable` — run `uv sync --extra gpu` and retry.
- `dataset.genotypes is not backed by a single PLINK BED` — only the AoU
  PLINK SNP source is bitpacked-eligible today; SV VCFs still go through
  the legacy int8 path.
- `variant-subset wrapper present` — disable `--variants` filtering or
  load the source without the wrapper.
- `loader failed: ...` — the loader logged a Python exception; see the
  lines immediately above the SKIPPED message for the traceback.

### "CuPy failed to load libnvrtc.so.12"

The NVIDIA `nvrtc` wheel is not on `LD_LIBRARY_PATH`. Fix:

```bash
uv sync --extra gpu
```

`run.sh` automatically prepends the `nvidia/*` wheel `lib` directories from
the active `.venv` to `LD_LIBRARY_PATH` before launching the CLI, so a
fresh `uv sync --extra gpu` followed by `bash run.sh ...` is enough.

### "Cold load very slow (>20 min)"

Cold PLINK loads can take 15-30 minutes on the AoU CDR trio. Watch for:

- `BED stream:` progress lines in the log — these show streaming
  throughput; a healthy load shows >100 MB/s sustained.
- `bitpacked upgrade: loading BED ... -> device via active-matrix cache
  (cache_dir=...)` — confirms the active-matrix cache is engaged. The
  SECOND run with the same sample intersection and variant axis is
  ~30 seconds (cache hit) instead of 20+ minutes.

If `BED stream:` lines stall, the underlying file is most likely on
gcsfuse and the workbench VM is IOPS-starved. Stage the BED to local
disk before launching.

### "EM iterations slow"

Grep the log for `model fit: matrix=BitpackedDeviceMatrix`. If the banner
shows a different type:

```
model fit: matrix=PlinkRawGenotypeMatrix ...
```

then the bitpacked upgrade was skipped — see the first troubleshooting
section to find the reason. EM iterations on the legacy int8 path can be
50-100x slower than the bitpacked GPU path at AoU scale (~10 minutes vs.
~10 seconds per outer iteration).

Same banner applies on the scoring path; iter 6 added `scoring:
matrix=BitpackedDeviceMatrix (single GPU matvec; ...)`. If scoring shows
`scoring: matrix=PlinkRawGenotypeMatrix (legacy iter_column_batches ...)`,
holdout / test predictions are being computed via host-streamed decode
and will be slow at AoU scale.

### "Out of memory" / "memory guardrail tripped"

The legacy int8 path materializes a (samples × variants) int8 mmap, which
at AoU scale (~78k × 700k) is ~55 GB. The memory guardrail
(`_legacy_int8_memory_guardrail` in `sv_pgs/io.py`) refuses to start a fit
that would OOM. Two remediations:

- Engage the bitpacked path (preferred): set `config.genotype_backend =
  "bitpacked"`. The bitpacked active matrix uses ~22 GB HBM at AoU scale
  (packed bytes plus per-variant mean/std side buffers).
- Reduce the variant axis: pass `--variants sv` or `--variants snp` rather
  than `snp+sv`, or supply a `--variants-file` subset.
