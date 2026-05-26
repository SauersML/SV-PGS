# Dead-Code Audit (Iter 6)

Catalogue of code paths that have been superseded by the bitpacked GPU stack
(`sv_pgs.bitpacked_matrix.BitpackedDeviceMatrix`, `sv_pgs.bitpacked_loader.*`)
but are still resident in the repo. This pass is **read-only** — nothing is
deleted here, only documented. A future pass can prune based on the
disposition flags below.

Disposition values:

- **KEEP**: still serves a live role (fallback when bitpacked not engaged,
  small-N path, CPU CI, etc.). Do not delete.
- **GATED**: dead behind a compile-time / module-level flag (`False` by
  default). Easy to delete if no caller flips the flag to `True`.
- **SHADOW**: superseded by a bitpacked equivalent, but still has live
  callers because the bitpacked upgrade is opt-in (`config.genotype_backend
  == "bitpacked"`). Cannot delete until bitpacked is the only supported
  backend.
- **DELETABLE**: no live callers; safe to delete pending a grep audit by
  the next pass.

| # | Location | What it does | Replacement | Disposition | Lines saved |
|---|----------|--------------|-------------|-------------|-------------|
| 1 | `sv_pgs/io.py:51` `_USE_INT8_NPY_CACHE: bool = False` | Module-level flag that gates writing a multi-GB int8 `.npy` mirror of the PLINK BED file. | Bitpacked path keeps genotypes in BED form and decodes on-GPU. | **GATED** | Flag itself: 1 line. Deletion would cascade into items 2 + 5. |
| 2 | `sv_pgs/io.py:2250-2376` `_build_plink_int8_cache_only` | Stream BED once and write the int8 cache without recomputing stats. First line of the function returns None when the gate flag is False, so the rest is unreachable for the bitpacked backend. | `sv_pgs.bitpacked_loader.load_bed_to_bitpacked_device` (used directly by `_maybe_upgrade_to_bitpacked` in `pipeline.py`). | **GATED** (the body is unreachable while `_USE_INT8_NPY_CACHE=False`; the function itself is still called by the int8 fallback path at io.py:2450, but only enters the no-op branch). | ~127 lines (function body); + ~12 lines at io.py:2440-2452 (the caller) once the fallback is removed. |
| 3 | `sv_pgs/io.py:2643-2790+` `_compute_variant_stats_teeing_int8` | Variant statistics with an optional tee-write of each decoded int8 batch into the streaming `.npy` cache. Called from io.py:2538. | Bitpacked loader computes mean/std directly on the device from packed bytes, no host int8 mirror. | **SHADOW** — still called when `int8_cache_writer is not None` (legacy/fallback int8 path). Keep until bitpacked is the only backend. | ~150 lines if removed wholesale. The non-teeing inner branch (line 2664: `compute_variant_statistics(...)`) is the actual replacement. |
| 4 | `sv_pgs/genotype.py:823-871` `class Int8RawGenotypeMatrix` | Pure-host int8 `RawGenotypeMatrix` backed by a numpy array / mmap. | `BitpackedDeviceMatrix` (device-resident, packed). | **SHADOW** — still used as a legacy fallback in genotype.py at lines 4835, 5027, 5034, 5314, 5394 (cache-rehydration and CPU-only test paths). Cannot delete without rewiring those. | ~50 lines for the class itself; ~120 lines of related glue in genotype.py if all references go away. |
| 5 | `sv_pgs/genotype.py:762, 836, 928, 1014, 1109, 1323` `iter_column_batches_i8` | int8-typed sibling of `iter_column_batches` defined across every `RawGenotypeMatrix` subclass for the legacy streaming pipeline. | Bitpacked path uses `matvec` / `rmatvec` / `gram_block` / `subset` directly. | **SHADOW** — still consumed by `_build_plink_int8_cache_only`, `_compute_variant_stats_teeing_int8`, and the CPU streaming PG-IRLS branch in `mixture_inference`. | ~40-60 lines per class × 6 classes ≈ 250-350 lines if the int8 streaming API is removed. |
| 6 | `sv_pgs/genotype.py:1781`, `1991`, `5343`, `5353` | int8 BED-decode → upload helpers (`raw_int8.iter_column_batches_i8(...)`) used to build the on-disk int8 mmap cache. | Bitpacked active-matrix cache (`SV_PGS_BITPACKED_ACTIVE_CACHE_DIR` + `load_bed_to_bitpacked_device_cached`). | **SHADOW** — the bitpacked active-matrix cache is the de-facto replacement for AoU runs, but the legacy int8 mmap cache is still triggered by `Int8RawGenotypeMatrix(existing_mmap)` in the cache-load fast path (genotype.py:5394). | ~25 lines per call site. |
| 7 | `sv_pgs/genotype.py:4392` `for raw_batch in raw.iter_column_batches_i8(...)` | The host-side decode-and-quantize loop that turns BED bytes into a streamed int8 column tile. | Bitpacked LUT decode happens on the device (`sv_pgs.bitpacked.lut`). | **SHADOW** — live caller in the streaming-residualization path. | Inside a single function (`_streaming_residualize_genotype_columns`); ~60-line body. |
| 8 | `sv_pgs/genotype.py:4835`, `5027`, `5034` `isinstance(raw_matrix, Int8RawGenotypeMatrix)` branches | Per-iter conditionals that select int8-mmap vs. dense-float behavior inside the EM hot loop. | Bitpacked path bypasses these branches entirely via the GPU matvec dispatch. | **SHADOW** — each branch is still reachable when bitpacked upgrade is skipped (no CuPy, non-BED source, variant-subset wrapper). | ~30 lines collectively. |
| 9 | `sv_pgs/model.py:2494-2512` `_raw_standardized_subset_matvec` | Scoring-time `Z @ beta` over the held-out set via streaming `iter_column_batches`. Used by `decision_components` when the matrix is not bitpacked. | `_bitpacked_scoring_fast_path` (added in iter 6, model.py:~2510) wraps a single GPU `matvec` over the nonzero-coefficient subset. | **KEEP** — explicit fallback for non-bitpacked matrices (dense numpy arrays in tests, CPU-only deployments, fallback when the upgrade skips). | 0 lines saved by deletion; the legacy path is the fallback contract. |
| 10 | `sv_pgs/io.py:2440-2452` int8-cache build wrapper | Conditional that decides whether to call `_build_plink_int8_cache_only`. | Bitpacked active-matrix cache. | **GATED** — body becomes unreachable once `_USE_INT8_NPY_CACHE=False` propagates. | ~12 lines. |
| 11 | `tests/test_*int8*.py` (16 files at last grep) | Unit tests pinning the legacy int8 cache writer / reader / progress-resume semantics. | The bitpacked stack has its own test suite (`tests/test_bitpacked_*.py`). | **SHADOW** — keep as long as the int8 fallback is callable; delete in the same pass as items 2-7. | ~2,000 lines if all dropped together. |
| 12 | `README.md` pre-bitpacked sections | Some sections describe the int8-decode pipeline as the default. | Bitpacked is the proven default; int8 is the named fallback. | **KEEP** but reword — see iter 6 README troubleshooting append. | Wording, not line count. |
| 13 | `sv_pgs/genotype.py:5343-5353` int8 cache rehydration after a `RowSubsetRawGenotypeMatrix` swap | Rebuild an `Int8RawGenotypeMatrix` from a freshly-written cache mmap and rebase `self.raw`. | Bitpacked active-matrix cache + `subset()` device gather. | **SHADOW**. | ~25 lines. |

## Aggregate disposition

- **GATED (3 items)**: items 1, 2, 10 — ~140 lines, behind a single
  module-level flag that has defaulted to `False` since iter 3.
- **SHADOW (8 items)**: items 3, 4, 5, 6, 7, 8, 11, 13 — ~2,400 lines of
  int8-fallback code. Cannot be deleted until the bitpacked backend is
  promoted from "opt-in fast path" to "only supported backend".
- **KEEP (2 items)**: item 9 (the legacy `_raw_standardized_subset_matvec`
  is the explicit fallback contract for non-bitpacked matrices) and item
  12 (a doc fix, not a code fix).

## Recommended pruning order (future pass)

1. Flip `_USE_INT8_NPY_CACHE` to `True` in CI for one full AoU shadow run.
   Confirm the bitpacked path produces an artifact bit-identical to the
   `_USE_INT8_NPY_CACHE=False` run. (Iter 5 already gated this with the
   e2e equivalence test.)
2. Delete items 1, 2, 10 (the GATED set; ~140 lines).
3. Delete items 4, 5, 7 (the int8 `RawGenotypeMatrix` class + its
   `iter_column_batches_i8` API surface; ~300 lines + 2,000 lines of
   pinned tests).
4. Delete item 3 (`_compute_variant_stats_teeing_int8`; ~150 lines).
5. Delete items 6, 8, 13 (call-site cleanup; ~80 lines).

Total: ~2,700 lines of code + tests can be pruned **once** bitpacked is
declared the sole supported backend. Iter 6 does not perform any
deletions — see `RELEASE_NOTES.md` for the cumulative scope of this pass.
