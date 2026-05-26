# SV-PGS Bitpacked GPU Pipeline — Shared Spec

This document is the single source of truth for the bitpacked-everywhere pipeline. All agents working on this refactor MUST read this spec before writing code, and MUST match the contracts here.

## Goal

Replace the host int8 `.npy` cache + host-side LUT decode in sv_pgs with a path that:
- Reads PLINK 1.9 BED bytes from disk
- Keeps them bitpacked (2 bits/sample) all the way to GPU HBM
- Decodes via constant-memory LUT inside CuPy RawKernels
- Runs three custom kernels: forward GEMV, transpose GEMV, gram GEMM
- Supports T4 (Turing), A100 (Ampere), H100 (Hopper) via per-arch dispatch

## PLINK 1.9 BED layout (canonical)

- File begins with 3-byte magic: `0x6c 0x1b 0x01`
- Then variant-major bitpacked region. For `N` samples, each variant occupies `bytes_per_variant = (N + 3) // 4` bytes.
- Inside each byte, 4 samples are packed low-bit-first:
  - `byte = (s3 << 6) | (s2 << 4) | (s1 << 2) | s0`
- Two-bit codes (under `count_A1=True`, which is sv-pgs's default per `plink.py:_DECODE_LOOKUP_A1`):
  - `0b00` → dosage **2** (homozygous A1)
  - `0b01` → **missing**
  - `0b10` → dosage **1** (heterozygous)
  - `0b11` → dosage **0** (homozygous A2)
- Under `count_A1=False`: 0b00→0, 0b01→missing, 0b10→1, 0b11→2.

Reference CPU decode: `sv_pgs/plink.py:_BYTE_DECODE_LUT_A1`, `_decode_payload`. New work MUST be bit-for-bit identical to this reference.

## Conventions

- Bitpacked matrices live as CuPy `uint8` arrays of shape `(n_variants, bytes_per_variant)`, contiguous C-order. **Variant-major** — row `v` holds variant `v`'s packed sample bytes.
- `n_samples` is passed separately and may be less than `4 * bytes_per_variant` (the last 0–3 trailing slots are padding to discard).
- `mean[v]`, `scale[v]` (preferred name; `std` is a legacy alias) are `float32` arrays precomputed by `sv_pgs.preprocessing._means_and_scales_with_floor`. The contract is REPO-EXACT and load-bearing:

```
N = total sample count
count[v]  = number of non-missing samples for variant v
sum[v]    = Σ_{i: observed} dosage[i,v]
sumsq[v]  = Σ_{i: observed} dosage[i,v] * dosage[i,v]
css[v]    = sumsq[v] - sum[v]^2 / max(count[v], 1)        # centered sum of squares, with missing mean-imputed to mean[v]

if count[v] > 0:
    mean[v]      = sum[v] / count[v]
    scale_raw[v] = sqrt(css[v] / max(N, 1))                # IMPORTANT: denominator is N (total), NOT count[v]
else:
    mean[v]      = 0.0
    scale_raw[v] = 0.0

if count[v] == 0:
    mean[v]  = 0.0
    scale[v] = 1.0
elif scale_raw[v] < minimum_scale:
    # low-variance / constant observed column
    mean[v]  = sum[v] / count[v]    # keep actual mean
    scale[v] = 1.0                  # floored; standardized output is 0 for the actual constant
else:
    scale[v] = scale_raw[v]
```

Standardized value:
- if code == 0b01 (missing): `z = 0` (mean-imputed; contributes 0 to every reduction)
- else: `z = (raw_dosage - mean[v]) / scale[v]`

Invariant: under no missingness and no floor, `Σ_i z[i,v]^2 == N` exactly (ddof=0). All kernel implementations and tests MUST match this contract — not a per-column ddof=1 std, not a non-missing-count denominator. The denominator is total `N`.

- Missing-value semantics for accumulation: a missing genotype contributes 0 to every reduction.
- All kernels are launched on a user-provided CuPy stream (default `None` → current stream).

## Module layout (NEW files only — no edits to existing files in wave 1)

```
sv_pgs/bitpacked/
  __init__.py            # re-exports public API
  lut.py                 # LUT generators (numpy)
  cpu_reference.py       # CPU implementations of GEMV/GEMV.T/GEMM/screen for tests
  launch.py              # GPU arch probe + per-arch launch configs
  gemv_nt.py             # forward GEMV kernel + Python wrapper
  gemv_tn.py             # transpose GEMV
  gemm_gram.py           # gram GEMM (dp4a / tf32 / fp16 variants)
  screening.py           # fused MAF + marginal-z screening kernel

sv_pgs/bitpacked_matrix.py    # BitpackedDeviceMatrix class (RawGenotypeMatrix-compatible)
sv_pgs/bitpacked_loader.py    # PLINK BED → bitpacked CuPy device buffer
sv_pgs/sv_transcoder.py       # SV VCF → PLINK BED transcoder (one-time per cohort)
sv_pgs/mmap_reader.py         # mmap reader with MADV_SEQUENTIAL fast path
sv_pgs/gds.py                 # GPUDirect Storage / cuFile probe + fast cold-load
sv_pgs/screening_pipeline.py  # end-to-end NVMe → screening → active gather

tests/test_bitpacked_lut.py
tests/test_bitpacked_cpu_reference.py
tests/test_bitpacked_gemv_nt.py
tests/test_bitpacked_gemv_tn.py
tests/test_bitpacked_gemm_gram.py
tests/test_bitpacked_screening.py
tests/test_bitpacked_matrix.py
tests/test_sv_transcoder.py
```

## Public API contracts

### `sv_pgs.bitpacked.lut`

```python
def make_decode_lut(count_a1: bool = True) -> np.ndarray:
    """Return (256, 4) int8 table mapping byte → (s0, s1, s2, s3) dosages.
    Missing slot encoded as -127 (PLINK_MISSING_INT8, matches plink.py)."""
```

### `sv_pgs.bitpacked.cpu_reference`

```python
def cpu_gemv_nt(packed, n_samples, x, mean, std, count_a1=True) -> np.ndarray:
    """Returns y of shape (n_samples,)."""

def cpu_gemv_tn(packed, n_samples, y, mean, std, count_a1=True) -> np.ndarray:
    """Returns g of shape (n_variants,)."""

def cpu_gemm_gram(packed, n_samples, mean, std, count_a1=True) -> np.ndarray:
    """Returns B = Z.T @ Z of shape (n_variants, n_variants), where Z is standardized.
    Missing slots zero. Use float64 internally for precision."""

def cpu_screen(packed, n_samples, y_resid=None, count_a1=True) -> dict:
    """Returns dict with keys:
        'count' (n_variants,) int32 — non-missing count
        'sum'   (n_variants,) float64 — sum of raw dosage
        'sumsq' (n_variants,) float64 — sum of squared raw dosage
        'y_dot' (n_variants,) float64 — inner product G_v · y_resid (if y_resid provided)
    """
```

### `sv_pgs.bitpacked.launch`

```python
def gpu_arch() -> Literal["t4", "ampere", "hopper", "unknown"]:
    """Probe compute capability via cupy/runtime. Cache."""

def gemv_nt_config(n_samples: int, n_variants: int, arch: str) -> dict:
    """Returns {'grid': ..., 'block': ..., 'shmem_bytes': ...}."""

# Similar for gemv_tn, gemm_gram, screening.
```

### `sv_pgs.bitpacked.gemv_nt`

```python
def gemv_nt(
    packed: cp.ndarray,      # (n_variants, bytes_per_variant) uint8
    n_samples: int,
    x: cp.ndarray,           # (n_variants,) float32
    mean: cp.ndarray,        # (n_variants,) float32
    std: cp.ndarray,         # (n_variants,) float32
    out: cp.ndarray,         # (n_samples,) float32 — accumulated INTO (not overwritten)
    count_a1: bool = True,
    stream: cp.cuda.Stream | None = None,
) -> None:
    """Compute y[i] += sum_v ((dosage[i,v] - mean[v]) / std[v]) * x[v]; missing -> 0.

    `out` must be initialized by the caller (e.g. cupy.zeros). Kernel adds INTO out.
    """
```

### `sv_pgs.bitpacked.gemv_tn`

```python
def gemv_tn(
    packed, n_samples, y, mean, std, out, count_a1=True, stream=None
) -> None:
    """Compute g[v] += sum_i ((dosage[i,v] - mean[v]) / std[v]) * y[i]; missing -> 0."""
```

### `sv_pgs.bitpacked.gemm_gram`

```python
def gemm_gram(
    packed, n_samples, mean, std, out, count_a1=True, stream=None
) -> None:
    """Compute B[u,v] = sum_i Z[i,u]*Z[i,v] with Z standardized.

    `out` is (n_variants, n_variants) float32, accumulated INTO.
    Implementation dispatches to per-arch kernel:
      - t4: DP4A int8 integer tensor cores
      - ampere: TF32 mma.sync m16n8k8
      - hopper: FP16 mma.sync m16n8k16
      - unknown: fp32 fallback
    """
```

### `sv_pgs.bitpacked.screening`

```python
def screen(
    packed, n_samples,
    out_count: cp.ndarray,        # (n_variants,) int32
    out_sum: cp.ndarray,          # (n_variants,) float64
    out_sumsq: cp.ndarray,        # (n_variants,) float64
    y_resid: cp.ndarray | None = None,
    out_y_dot: cp.ndarray | None = None,  # (n_variants,) float64 — required iff y_resid provided
    count_a1: bool = True,
    stream=None,
) -> None:
    """One-pass screening reduction. All outputs accumulated INTO."""
```

### `sv_pgs.bitpacked_matrix.BitpackedDeviceMatrix`

```python
class BitpackedDeviceMatrix:
    """Drop-in replacement for Int8RawGenotypeMatrix backed by bitpacked HBM bytes.

    Holds:
      packed: cp.ndarray (n_variants, bytes_per_variant) uint8
      mean:   cp.ndarray (n_variants,) float32
      std:    cp.ndarray (n_variants,) float32
      n_samples: int
      count_a1: bool
    """
    @property
    def shape(self) -> tuple[int, int]: ...  # (n_samples, n_variants)
    @property
    def dtype(self): return cp.dtype('float32')

    def matvec_numpy(self, x_np: np.ndarray) -> np.ndarray: ...    # G @ x
    def rmatvec_numpy(self, y_np: np.ndarray) -> np.ndarray: ...   # G.T @ y
    def matvec(self, x_dev: cp.ndarray) -> cp.ndarray: ...
    def rmatvec(self, y_dev: cp.ndarray) -> cp.ndarray: ...
    def matmat(self, X) -> ...
    def transpose_matmat_numpy(self, Y_np) -> ...
    def gram_block(self, variant_indices: cp.ndarray) -> cp.ndarray: ...  # subset gram
    def subset(self, variant_indices: np.ndarray) -> "BitpackedDeviceMatrix": ...
    def column_means(self) -> np.ndarray: ...
    def column_stds(self) -> np.ndarray: ...
    def to_host_int8(self) -> np.ndarray: ...  # (n_samples, n_variants) — for tests
```

Methods MUST satisfy the `RawGenotypeMatrix` ABC in `sv_pgs/genotype.py:702`. Look at `Int8RawGenotypeMatrix` (genotype.py:823) for the exact method set required.

### `sv_pgs.bitpacked_loader`

```python
def load_bed_to_bitpacked_device(
    bed_path: str | Path,
    n_samples: int,
    n_variants: int,
    variant_indices: np.ndarray | None = None,   # gather subset; None = all
    sample_indices: np.ndarray | None = None,    # gather subset; None = all
    mean: np.ndarray | None = None,              # precomputed; None = compute now
    std: np.ndarray | None = None,
    count_a1: bool = True,
    stream=None,
) -> "BitpackedDeviceMatrix":
    """Read PLINK BED → device-resident BitpackedDeviceMatrix.

    For variant_indices != None: gather only those variants (use coalesced reads,
    see plink.py:_pread_indexed_variant_payload for the pattern).

    For sample_indices != None: must rebitpack to a new sample stride. This is
    the cohort-intersection path. Done in CPU before upload because the rebitpack
    is a permutation, not a contiguous slice.

    If mean/std not provided, run the screening kernel after upload to compute them.
    """
```

### `sv_pgs.sv_transcoder`

```python
def transcode_sv_vcf_to_bed(
    vcf_paths: list[Path],
    bed_out_path: Path,
    sample_ids: list[str] | None = None,   # restrict to these
    sample_id_order: list[str] | None = None,  # final ordering in BED .fam
) -> dict:
    """Read AoU SV VCFs (cyvcf2 preferred), write a single PLINK 1.9 BED trio.

    Returns metadata dict with:
        n_variants, n_samples, variant_records (list of VariantRecord),
        svtype_per_variant (np.ndarray of str)
    Re-uses sv_pgs.plink.to_bed for the write path and sv_pgs.io's existing
    cyvcf2 helpers for the read path.
    """
```

### `sv_pgs.mmap_reader`

```python
class BedMmapReader:
    """mmap-based BED reader. Used for sequential cold loads (screening pass).
    Falls back to preadv on any OSError/SIGBUS-recovery path.
    """
    def __init__(self, path, n_samples, n_variants, count_a1=True): ...
    def read_all_packed(self) -> np.ndarray: ...  # entire file as (n_variants, bpv) uint8
    def read_packed_range(self, start, stop) -> np.ndarray: ...
    def read_packed_indexed(self, indices) -> np.ndarray: ...
    def close(self): ...
```

Uses `mmap.mmap(fd, length, access=mmap.ACCESS_READ)` + `madvise(MADV_SEQUENTIAL | MADV_WILLNEED)` for the screening pass.

### `sv_pgs.gds`

```python
def gpudirect_available() -> bool:
    """Probe cuFile / GPUDirect Storage by importing kvikio or running a tiny test."""

def cufile_read_to_device(path: Path, device_buffer: cp.ndarray, offset: int, count: int) -> None:
    """Read `count` bytes from `path` at `offset` directly into device_buffer.
    Raises RuntimeError if cuFile not available."""
```

If `kvikio` is installable in the env, use it; otherwise this module's `gpudirect_available()` returns False and callers fall back to pinned-RAM staging.

### `sv_pgs.screening_pipeline`

```python
def run_screening_pass(
    bed_paths: list[Path],            # one or more BED shards
    n_samples_per_path: list[int],
    n_variants_per_path: list[int],
    sample_intersect: np.ndarray | None = None,  # global sample subset
    y_resid: np.ndarray | None = None,
    count_a1: bool = True,
    stream=None,
) -> dict:
    """One pass over all BED files: stream → DMA → on-device screening kernel.

    Returns dict with concatenated per-variant arrays:
        count, sum, sumsq, y_dot (if y_resid given).
    """
```

## Reference quantities for tests

All tests should generate small synthetic BED files (e.g. 100 samples × 500 variants), validate against `sv_pgs.plink._decode_payload` (the canonical CPU LUT), and additionally validate kernel outputs against the CPU reference at fp32 tolerance `rtol=1e-4, atol=1e-3`, fp64 tolerance `rtol=1e-9, atol=1e-9`.

Important: synthetic data MUST exercise the missing code (0b01) and the trailing-padding samples (n_samples % 4 != 0).

## Existing-file integration (wave 2 — out of scope for wave 1)

Wave 1 agents do NOT edit existing files. Their new modules import nothing from `sv_pgs.io`, `sv_pgs.genotype`, `sv_pgs.aou_runner`, `sv_pgs.preprocessing`, `sv_pgs.model`, `sv_pgs.pipeline`, `sv_pgs.block_matvec`, `sv_pgs.linear_solvers`, `sv_pgs.plink` *except* for: importing `sv_pgs.plink.to_bed`, `sv_pgs.plink._BYTE_DECODE_LUT_A1`, `sv_pgs.plink._BYTE_DECODE_LUT_A2`, `sv_pgs.plink.PLINK1_MAGIC`, `sv_pgs.plink.PLINK_MISSING_INT8`, `sv_pgs.plink._bytes_per_variant` as read-only utilities.

Wave 1 agents MAY import `sv_pgs._jax` for GPU capability detection.

## Acceptance criteria for wave 1

- All new files compile / import without errors when `cupy` is absent (use lazy imports, no top-level `import cupy`).
- CPU reference implementations are exact (no approximations).
- Kernel implementations include the CUDA source as a Python triple-quoted string in the wrapper module.
- Tests pass on CPU (kernel tests should `pytest.skip` cleanly when cupy/CUDA absent).
- No edits to existing files in `sv_pgs/`.
- `ruff check` clean on new files.

## Performance targets (wave 1 — informational)

For 97k samples × 400k active variants (~10 GB bitpacked, microarray scale):
- Cold load: < 5 s (NVMe Gen4 bound)
- Screening pass: < 5 s
- GEMV forward: T4 < 35 ms, A100 < 7 ms, H100 < 5 ms per call
- GEMV transpose: same
- Gram block (B=10k, n=97k): T4 < 100 ms (DP4A), A100 < 8 ms (TF32), H100 < 3 ms (FP16)
