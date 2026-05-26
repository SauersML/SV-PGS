# SV-PGS AoU Operational Contract

This document is the single source of truth for any sv-pgs run executing inside an All of Us (AoU) Researcher Workbench VM. It is a hard contract. Violations MUST cause the entrypoint to abort loudly with a single-line reason before any GPU init, before any large stage, before any model fit. There is no "warn and continue" mode.

Distilled from the external operational review. Wave-2 work that touches any AoU code path MUST conform.

## 1. Scope

Applies to every production AoU run: `aou_runner`, the AoU CLI entrypoint, batch reruns, resumed runs, and any helper script that touches Controlled Tier data. Local laptop / non-AoU GCE / CI runs are out of scope and MAY relax these rules via explicit flags.

sv-pgs aborts loudly on violation. No silent fallbacks. No "best effort". A run that cannot satisfy the contract is a run that does not start.

## 2. Storage classes

There are exactly three storage classes. They map 1:1 to `sv_pgs.path_policy.StorageClass`. No fourth class is invented at runtime.

- **`CONTROLLED_GCS_SOURCE`** — read-only, remote. Controlled Tier CDR objects: PLINK trios, SV VCFs, ancestry preds, sample manifests. Access is via `gs://` URI or a gcsfuse-mounted read-only path. Treated as a tape: high-latency, single-pass, never the hot side of a loop.
- **`LOCAL_HOT_CACHE`** — persistent disk on the VM (NVMe / pd-ssd). The only legal hot-path source for genotype matrices and any object that is read more than once. Lifetime: persists across runs; survives kernel restarts; does not survive VM delete.
- **`WORKSPACE_BUCKET`** — durable outputs only. Final model artifacts, scores, manifests, logs, checkpoints intended to survive VM termination. Writes are one-shot sequential. Never the source of a hot loop.

Anything that does not fit one of these three classes is a bug.

## 3. The gcsfuse rule

Verbatim from the external review:

> No production AoU run may perform repeated genotype-matrix reads from a gcsfuse-mounted path.

This is the load-bearing invariant. Everything else in this spec exists to enforce it.

**Forbidden uses of gcsfuse:**

- PLINK `.bed` opened directly off a gcsfuse mount for matvec / rmatvec / screening.
- Repeated `tabix` queries against `.vcf.gz` on a gcsfuse mount.
- `mmap` of any genotype file living on a gcsfuse mount.
- Checkpoint working directory on a gcsfuse mount.
- Using `Path.exists()` on a gcsfuse mount as a cache-validity check. Existence on a mount is not a cache hit; it is a network round trip with no integrity guarantee.

**Allowed uses of gcsfuse:**

- Small sidecar reads: `.fam`, `.bim`, `.psam`, ancestry preds, small TSVs (under a few MB), read once, copied to a local Python object.
- One-shot sequential copy via `sv_pgs.aou_storage.stage_gcs_object` (or the equivalent `gsutil cp` shellout in early bootstrap) to land an object in `LOCAL_HOT_CACHE`.
- Manifest-aware existence checks where the manifest is itself a tiny sidecar.

If in doubt: it is forbidden.

## 4. Cache manifest contract

Every object staged into `LOCAL_HOT_CACHE` MUST have a sibling `<path>.manifest.json`. No manifest, no cache hit. File existence is not evidence of anything.

Manifest schema (see `sv_pgs.aou_storage` for the canonical writer/reader):

- `schema_version` — int, bumped on incompatible changes.
- `source_uri` — `gs://...` URI of the canonical source.
- `source_size_bytes` — int, from `gs stat` at stage time.
- `source_generation` — GCS generation number, the only stable object identity.
- `source_crc32c` — base64 crc32c from GCS metadata.
- `local_size_bytes` — `os.stat(...).st_size` after publish.
- `content_kind` — `"plink_bed"`, `"plink_bim"`, `"plink_fam"`, `"sv_vcf"`, `"sv_tbi"`, `"ancestry_tsv"`, `"derived_bed"`, etc.
- `complete` — bool. ONLY set true after fsync + replace + manifest write succeeds.

Cache hit = manifest parses + `schema_version` matches + `source_generation` matches the current `gs stat` + `local_size_bytes` matches `os.stat` + `complete=true`. Anything else is a miss and triggers re-stage.

Point implementations at `sv_pgs.aou_storage`. Do not roll your own.

## 5. Atomic publish

Staging writes go to `<path>.partial.<pid>.<uuid8>`. After the byte stream completes:

1. `fsync` the partial file fd.
2. `os.replace(partial, final)` — atomic on the same filesystem.
3. Write `<path>.manifest.json` (also via partial + fsync + replace).
4. Release the lock.

Failure at any step leaves no manifest, therefore no cache hit, therefore a clean re-stage on next run. The partial file is garbage-collected by a `*.partial.*` sweep in preflight.

Concurrent workers serialize on `<path>.lock` via `fcntl.lockf(LOCK_EX)`. The lock file is allowed to be empty; it exists only as a lock target. Lock acquisition has a bounded timeout; timeout is a hard abort, not a fallthrough.

## 6. Preflight contract

Before any GPU init, before any large staging, before any model touches data, `sv_pgs.preflight.check_aou_preflight()` MUST be called and MUST pass. The checks are listed in the docstring of `sv_pgs/preflight.py` and are the source of truth. Summary:

- AoU environment detection (env vars, workspace bucket presence, CDR version probe).
- `LOCAL_HOT_CACHE` directory exists, is on local disk (not gcsfuse), has expected free bytes.
- `WORKSPACE_BUCKET` is writable (probe write + delete).
- `XLA_PYTHON_CLIENT_PREALLOCATE` is set (see Section 7).
- No `jax`/`cupy` imports have already happened at the time of the check.
- gcsfuse mount inventory: every mount is tagged read-only; no mount is in `LOCAL_HOT_CACHE`.
- Stale partial sweep in cache dir.
- GPU visible, compute capability probe matches a supported arch (`t4`, `ampere`, `hopper`).

Failure of any single check is a hard abort with a single-line reason.

## 7. GPU memory ownership

JAX preallocates 75% of device memory by default. On a T4 with CuPy kernels (`sv_pgs.bitpacked.*`) co-resident, this is fatal.

- AoU entrypoints MUST set `XLA_PYTHON_CLIENT_PREALLOCATE=false` (or a bounded `XLA_PYTHON_CLIENT_MEM_FRACTION`) **before** any `import jax` happens, anywhere in the process. Setting it after import is a no-op.
- The CuPy pinned memory pool and device memory pool MUST be explicitly bounded. Unbounded pools on a T4 cause OOM at second epoch, not first.
- Preflight logs: JAX preallocate setting, CuPy pool caps, free HBM at probe, free HBM after a 1 MB warm-up allocation.

If a process imports `jax` before `XLA_PYTHON_CLIENT_PREALLOCATE` is set, preflight aborts. There is no remediation in-process.

## 8. mmap policy

`mmap` of BED files is **off by default** on AoU.

`sv_pgs.mmap_reader.BedMmapReader` enforces this: construction requires `allow_gcsfuse=False` and a path that resolves into `LOCAL_HOT_CACHE`. Construction with a gcsfuse path raises. This is already wired in `sv_pgs.mmap_reader`; do not bypass it.

Experimental override: `SV_PGS_EXPERIMENTAL_MMAP=1` allows mmap of a local-hot path only (never gcsfuse). The override does not unlock gcsfuse mmap under any combination of flags. There is no flag to do that. Don't ask.

## 9. Pipeline phases

Every AoU run is exactly five phases. Phases B, C, and D never read gcsfuse.

- **A — Resolve.** Read CDR paths from env (`WORKSPACE_BUCKET`, `WGS_PLINK_PATH`, CDR version, etc.). Sidecar reads only (`.bim`, `.fam`, ancestry). Tiny, may touch gcsfuse.
- **B — Stage.** Copy CDR genotype sources to `LOCAL_HOT_CACHE` via `sv_pgs.aou_storage`. Atomic publish + manifest. One-shot sequential. May touch gcsfuse exactly once per object.
- **C — Derive.** Transcode / bitpack / compute stats. Reads ONLY from `LOCAL_HOT_CACHE`. Writes derived artifacts back to `LOCAL_HOT_CACHE` with manifests.
- **D — Fit.** Model fit. Reads ONLY from `LOCAL_HOT_CACHE`. Checkpoints to `LOCAL_HOT_CACHE`; periodic snapshot to `WORKSPACE_BUCKET`.
- **E — Persist.** Final compact outputs (weights, scores, manifest, log) to `WORKSPACE_BUCKET`. One-shot sequential writes.

A phase that violates its read-source class is a bug, not a config choice.

## 10. SV VCF special case

SV VCFs are never a repeated-read source. Per chromosome:

1. Stage `<chr>.vcf.gz` + `<chr>.vcf.gz.tbi` to `LOCAL_HOT_CACHE` once. Manifest each.
2. Transcode to PLINK BED via `sv_pgs.sv_transcoder.transcode_sv_vcf_to_bed` once. Manifest the BED with `content_kind="derived_bed"`.
3. All downstream runs (and all retries, all resumes, all reruns of phase D) consume the derived BED, never the VCF.

The derived BED's manifest records the source VCF generation. If the source VCF generation changes, the derived BED is invalidated.

## 11. Restartability

Any phase MUST be safely interruptible. The contract:

- Cache manifests + lock files make resume free. A killed process leaves no valid manifest for the in-flight object, so the next run re-stages that one object.
- `kill -9` mid-stage MUST leave no valid manifest, MUST leave no broken final file. (The partial is garbage; the final is absent because `os.replace` had not run.)
- `kill -9` mid-fit MUST leave the previous good checkpoint intact. Checkpoint writes use the same partial + fsync + replace pattern.
- No phase relies on Python `atexit` for correctness. `atexit` does not run on SIGKILL.

Test contract: a chaos test that SIGKILLs the runner at random offsets during stage and during fit MUST leave the cache directory in a state where the next run either resumes cleanly or re-stages the single affected object. No manual cleanup, ever.

## 12. AoU CLI defaults

Env vars consumed:

- `WORKSPACE_BUCKET` — durable output bucket, required.
- `WORKSPACE_CDR` — CDR version string, required.
- `SV_PGS_CACHE_DIR` — `LOCAL_HOT_CACHE` root, required.
- `XLA_PYTHON_CLIENT_PREALLOCATE` — must be `false` or a bounded fraction. Required.
- `SV_PGS_EXPERIMENTAL_MMAP` — optional, opt-in only.

CLI flags (defaults shown):

- `--cache-dir=$SV_PGS_CACHE_DIR`
- `--require-local-genotype-cache=true`
- `--no-gcsfuse-hot-path=true`
- `--resume=true`
- `--upload-checkpoints=true`

Emergency override, for a human who has read this document and accepted the consequences:

- `--allow-remote-hot-path-i-know-this-will-be-slow-and-fragile`

The flag is intentionally ugly. It disables the gcsfuse-rule abort and nothing else. It does not disable preflight. It does not disable manifests. It does not make a remote run fast. It exists because once a year someone needs to demo something on a workspace with no scratch disk attached.

## 13. Acceptance tests

Twelve numbered tests, transcribed from the external review section "AoU operational acceptance criteria". Each is a single hard assertion. All twelve MUST pass on a real AoU VM before any model fit is trusted in production.

1. Preflight aborts loudly when `XLA_PYTHON_CLIENT_PREALLOCATE` is unset.
2. Preflight aborts loudly when `jax` has already been imported before the check runs.
3. Preflight aborts loudly when `SV_PGS_CACHE_DIR` resolves into a gcsfuse mount.
4. Staging a CDR PLINK trio produces three final files plus three manifests; `complete=true` on all three; `source_generation` matches `gs stat`.
5. Killing the staging process with SIGKILL at a random offset leaves zero manifests for the in-flight object and zero non-partial files; the next run completes staging.
6. Opening a `.bed` directly off a gcsfuse mount via `BedMmapReader` raises before any I/O.
7. A second concurrent worker attempting to stage the same object blocks on the lock file and exits cleanly after the first publishes the manifest, with zero duplicate writes.
8. SV VCF transcode runs exactly once across N consecutive runs; runs 2..N consume the derived BED and never open the VCF.
9. Phase D fit reads only from `LOCAL_HOT_CACHE`; an `strace`-equivalent audit over the fit phase shows no opens against any gcsfuse-mounted path.
10. Final artifact upload to `WORKSPACE_BUCKET` is one-shot sequential and produces a sibling manifest in the bucket recording source local generation and crc32c.
11. A rerun after a successful run is a no-op for phases B, C, D, E (all cache hits, all manifests valid, no GPU work) and exits in under 30 seconds.
12. The `--allow-remote-hot-path-i-know-this-will-be-slow-and-fragile` override disables only the gcsfuse-rule abort; preflight, manifests, atomic publish, and locking all still run.

Anything less than twelve green is not a production run.
