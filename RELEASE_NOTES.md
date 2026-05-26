# SV-PGS Release Notes

Cumulative summary of what shipped across the multi-iteration hardening
work (iters 1-6). Each section groups changes by theme rather than by
iteration so operators can see the current capability surface at a glance.

## Highlights

- **Bitpacked GPU pipeline** is the default fast path for AoU-scale fits
  and now also for scoring. The legacy int8 streaming path remains as an
  explicit fallback.
- **End-to-end equivalence** of the bitpacked and int8 backends is pinned
  by a numerical-tolerance test (`tests/test_e2e_equivalence.py`).
- **Operational robustness**: nvrtc env auto-wiring, preflight nvrtc
  probe, smoke canary, memory guardrail, active-matrix cache, atomic
  publish, signal handling (SIGTERM / SIGHUP / SIGTSTP).
- **Observability**: TIMING SUMMARY, ENGAGED/SKIPPED banners,
  `BED stream:` progress, hot-loop dtype banner, scoring-path banner,
  per-iteration EM checkpoints with warm-restart.

## Performance (V100)

Measured by `python -m sv_pgs.bitpacked.bench` at the AoU scale (n=97k):

| Op        | Throughput            |
|-----------|-----------------------|
| `gemv_nt` | 104 GB/s              |
| `gemv_tn` | 199 GB/s              |
| `gemm_gram` | 11.7+ TFLOPS (fp32) |
| `screen`  | (see bench output)    |

EM outer iteration time on the AoU CDR PLINK SNP trio (78k × 700k):
- Legacy int8 streaming: ~10 min/iter
- Bitpacked GPU: ~10 s/iter (60-90x speedup)

Scoring (`Z @ beta` over the held-out cohort, k ~ 10k nonzero
coefficients): a single GPU `matvec` over the subset rather than a host
streaming decode (iter 6 — `_bitpacked_scoring_fast_path` in
`sv_pgs/model.py`).

## Correctness

- 61 CPU + 39 GPU pytest cases pass.
- `tests/test_e2e_equivalence.py` pins bitpacked-vs-int8 numerical
  equivalence end-to-end (iter 5).
- `tests/test_scoring_bitpacked_fast_path.py` pins the scoring-path
  fallback contract and verifies wiring (iter 6).
- A pure-CPU reference (`sv_pgs/bitpacked/cpu_reference.py`) is used by
  the smoke canary (`python -m sv_pgs.bitpacked.smoke`) to validate every
  kernel against its numpy equivalent before any production fit runs.

## Robustness

- **nvrtc env auto-wiring**: `run.sh` prepends every `nvidia/*/lib`
  directory from the active `.venv/lib/python*/site-packages/nvidia/` to
  `LD_LIBRARY_PATH` before launching the CLI, so CuPy finds
  `libnvrtc.so.12` even on workbench VMs without the system CUDA toolkit.
- **Preflight nvrtc probe**: before EM, `sv_pgs/preflight.py` compiles a
  trivial CUDA kernel to surface CUDA-stack failures with a clear error.
- **Smoke canary**: `python -m sv_pgs.bitpacked.smoke` builds a tiny
  synthetic BED, exercises every kernel, and exits 0 + prints
  "BITPACKED PIPELINE OK".
- **Memory guardrail**: `_legacy_int8_memory_guardrail` refuses to start
  a legacy int8 fit that would OOM (host RAM > available); operators see
  a structured diagnostic instead of a Linux OOM kill.
- **Active-matrix cache**: `SV_PGS_BITPACKED_ACTIVE_CACHE_DIR` skips
  cold-load decode on the second and later invocations whose sample
  intersection + variant axis hash matches a prior run. Atomic publish
  via `.partial` + rename so a kill mid-write is safe.
- **Atomic publish for every cache writer**: variant stats, int8 npy,
  bitpacked active matrix, EM checkpoint — every cache file is published
  via a temp file + rename so a SIGKILL mid-write leaves the destination
  in a consistent prior state.
- **Signal handling**: SIGTERM, SIGHUP, and (iter 6) SIGTSTP are
  converted into `KeyboardInterrupt` so atexit + finally + GPU teardown
  + per-iter EM checkpoint persist on the AoU workbench when the parent
  shell exits (rc=148 / SIGTSTP).
- **Per-iteration EM checkpoint**: `_save_variational_checkpoint` and the
  binary-block partial-progress writer in `mixture_inference.py` persist
  after every outer iteration. A warm restart probes
  `_try_load_variational_checkpoint` and resumes from the last completed
  iteration (or last mid-iteration block for binary fits) instead of
  iter 0.

## Operational visibility

Every long-running phase emits a greppable banner so operators can audit
what engaged from the log:

- **Fit-time matrix dispatch**: `model fit: matrix=BitpackedDeviceMatrix`
  vs `model fit: matrix=PlinkRawGenotypeMatrix`.
- **Scoring matrix dispatch** (iter 6): `scoring:
  matrix=BitpackedDeviceMatrix (single GPU matvec; ...)` vs `scoring:
  matrix=<legacy> (legacy iter_column_batches; ...)`.
- **Bitpacked upgrade**: `bitpacked upgrade: ENGAGED (active=N variants,
  packed_bytes=..., HBM bytes=... GB)` vs `bitpacked upgrade: SKIPPED
  (reason: ...)`.
- **BED stream progress**: `BED stream: X / Y variants (Z MB/s)`.
- **TIMING SUMMARY**: emitted at end of each phase with wall-time
  breakdown by sub-step.
- **Heartbeat profiling**: per-iteration timing breakdown in the EM loop
  (added iter 5).
- **Hot-loop dtype banner**: confirms whether the inner GPU matmul ran
  in fp32 or fp16/bf16 (iter 4).

## New entrypoints

- `python -m sv_pgs.bitpacked.smoke` — tiny BED smoke canary;
  exits 0 + prints "BITPACKED PIPELINE OK".
- `python -m sv_pgs.bitpacked.bench [--quick] [--output bench.json]` —
  GPU benchmark over `gemv_nt` / `gemv_tn` / `gemm_gram` / `screen` at
  three scales, with a markdown table and optional JSON report.
- `run.sh --smoke` — wraps the smoke canary plus the nvrtc-env preflight
  in a single shell command.
- `run.sh --validate` — runs the smoke + the bench + the e2e equivalence
  test against the local checkout.

## Dead-code audit

See `DEAD_CODE_AUDIT.md` for the cumulative list of int8 paths that have
been superseded by the bitpacked stack. ~140 lines are GATED behind
`_USE_INT8_NPY_CACHE=False` (default); ~2,400 lines of SHADOW code are
still callable as the explicit int8 fallback. No code was deleted in
iter 6 — the audit is the input to a future pruning pass.
