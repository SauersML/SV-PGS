#!/usr/bin/env bash
set -euo pipefail

# Run every child in our own process group so a ^C / ^Z on this script
# tears down every descendant (uv run, sv-pgs, decode threads, GPU
# contexts) instead of leaving ghost processes pinned to the GPU and
# blocking the next run. Without this, hitting Ctrl-C on the parent
# leaves the foreground PG alive and the user has to manually find +
# kill stuck pids.
set -m
SV_PGS_RUN_PGID="$$"

_sv_pgs_run_cleanup() {
  local rc=$?
  trap - EXIT INT TERM HUP
  # Send SIGTERM to the whole process group (negative pid = group), wait
  # briefly for graceful Python shutdown, then SIGKILL anything left.
  # Best-effort; cleanup must never fail the cleanup.
  local pids pid
  pids=$(pgrep -g "$SV_PGS_RUN_PGID" 2>/dev/null || true)
  for pid in $pids; do
    [ "$pid" = "$$" ] && continue
    kill "$pid" 2>/dev/null || true
  done
  local waited=0
  while [ "$waited" -lt 10 ]; do
    local survivors=0
    pids=$(pgrep -g "$SV_PGS_RUN_PGID" 2>/dev/null || true)
    for pid in $pids; do
      [ "$pid" = "$$" ] && continue
      survivors=$((survivors + 1))
    done
    if [ "$survivors" -eq 0 ]; then
      break
    fi
    sleep 1
    waited=$((waited + 1))
  done
  pids=$(pgrep -g "$SV_PGS_RUN_PGID" 2>/dev/null || true)
  for pid in $pids; do
    [ "$pid" = "$$" ] && continue
    kill -SIGKILL "$pid" 2>/dev/null || true
  done
  exit "$rc"
}

# EXIT covers normal completion + uncaught errors under `set -e`.
# INT covers ^C. TERM covers external kill. HUP covers terminal close.
# (^Z sends TSTP which suspends; we don't trap it because the user may
# legitimately want to fg/bg the run later. The kill_stuck_or_running
# function at start of run_variant_pass auto-resumes-and-kills any
# stopped prior run.)
trap _sv_pgs_run_cleanup EXIT INT TERM HUP

# Usage:
#   ./run.sh                 # all 19 SNOMED diseases, both SNP-only and SNP+SV
#   ./run.sh hypertension    # one disease, both variant sets
#   ./run.sh all             # explicit "all"
#   ./run.sh --smoke         # ONLY run the bitpacked end-to-end smoke check and exit
SV_PGS_SMOKE_ONLY=0
SV_PGS_VALIDATE_ONLY=0
if [ "${1:-}" = "--smoke" ]; then
  SV_PGS_SMOKE_ONLY=1
  shift || true
elif [ "${1:-}" = "--validate" ]; then
  SV_PGS_VALIDATE_ONLY=1
  shift || true
fi
DISEASE="${1:-all}"

REPO_DIR="$HOME/SV-PGS"

if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
fi
export PATH="$HOME/.local/bin:$PATH"

cd "$REPO_DIR"
if [ -d .git ] && [ -z "${SV_PGS_SKIP_PULL:-}" ]; then
  if git diff --quiet && git diff --cached --quiet; then
    branch="$(git rev-parse --abbrev-ref HEAD)"
    echo "updating code: git pull --ff-only origin ${branch}"
    git pull --ff-only origin "$branch" || echo "  (git pull failed — continuing with local code)"
  else
    echo "skipping git pull: working tree has local changes"
  fi
fi
uv sync --python 3.12 --extra gpu

# If the host driver only speaks CUDA 11 (driver_version < 525), cupy-cuda12x
# will refuse to run with cudaErrorInsufficientDriver. Detect that case and
# swap in cupy-cuda11x so the GPU is actually usable.
if command -v nvidia-smi >/dev/null 2>&1; then
  drv=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d '[:space:]')
  if [ -n "$drv" ]; then
    drv_major=$(echo "$drv" | cut -d. -f1)
    if [ -n "$drv_major" ] && [ "$drv_major" -lt 525 ] 2>/dev/null; then
      echo "host NVIDIA driver $drv supports CUDA 11 only — installing cupy-cuda11x..."
      .venv/bin/python -m pip uninstall -y cupy-cuda12x >/dev/null 2>&1 || true
      .venv/bin/python -m pip install --quiet "cupy-cuda11x>=13.0"
    fi
  fi
fi

# Make the CUDA shared libraries shipped by the nvidia-* wheels under .venv
# discoverable by cupy's runtime dlopen. The AoU container has only
# cuda-cudart-12-9 in /usr/local/cuda/lib64 — no libnvrtc.so.12 — so cupy
# fails at first kernel compile with:
#   RuntimeError: CuPy failed to load libnvrtc.so.12
# The wheels install lib*.so.* into .venv/lib/python3.*/site-packages/nvidia/*/lib;
# prepending those dirs to LD_LIBRARY_PATH (before /usr/local/cuda/lib64) fixes it
# without bind-mounting anything into the container.
nvidia_libs=""
nvidia_lib_count=0
for lib_dir in $(ls -d .venv/lib/python3.*/site-packages/nvidia/*/lib 2>/dev/null); do
  abs_dir="$(cd "$lib_dir" 2>/dev/null && pwd)"
  if [ -n "$abs_dir" ]; then
    nvidia_libs="${abs_dir}:${nvidia_libs}"
    nvidia_lib_count=$((nvidia_lib_count + 1))
  fi
done
if [ "$nvidia_lib_count" -gt 0 ]; then
  export LD_LIBRARY_PATH="${nvidia_libs}${LD_LIBRARY_PATH:-}"
  echo "LD_LIBRARY_PATH augmented with .venv nvidia wheel libs: ${nvidia_lib_count} dirs"
fi

# Fast-fail bitpacked GPU smoke: catches libnvrtc / kernel-compile / kernel-
# correctness / integration breakage in ~5s, BEFORE any 90-minute production
# run. Skipped automatically when no /dev/nvidia* device nodes are present
# (CI / cpu-only dev shells).
run_bitpacked_smoke() {
  local require="${1:-soft}"  # "soft" | "strict"
  if [ "$require" != "strict" ]; then
    # In normal run mode, skip silently if there is no GPU on the host.
    for nvidia_node in /dev/nvidia[0-9]* /dev/nvidiactl /dev/nvidia-uvm; do
      [ -e "$nvidia_node" ] && break || true
    done
    if ! ls /dev/nvidia[0-9]* /dev/nvidiactl /dev/nvidia-uvm >/dev/null 2>&1; then
      echo "  bitpacked smoke: skipped (no /dev/nvidia* device nodes)"
      return 0
    fi
  fi
  echo "=== BITPACKED PIPELINE SMOKE ==="
  if .venv/bin/python -m sv_pgs.bitpacked.smoke; then
    echo "=== END BITPACKED PIPELINE SMOKE (OK) ==="
    return 0
  else
    local rc=$?
    echo "=== END BITPACKED PIPELINE SMOKE (FAIL rc=${rc}) ==="
    return "$rc"
  fi
}

if [ "$SV_PGS_SMOKE_ONLY" = "1" ]; then
  run_bitpacked_smoke strict
  exit $?
fi
# Run the smoke as a non-fatal canary before the heavy steps; if it fails
# the operator gets a clear error in the log, but we still try to continue
# so partial diagnostics (gpu info, cache state) get printed.
run_bitpacked_smoke soft || echo "  WARNING: bitpacked smoke failed; production run may also fail"

has_nvidia_device_nodes() {
  local nodes=()
  local node
  for node in /dev/nvidia[0-9]* /dev/nvidiactl /dev/nvidia-uvm; do
    [ -e "$node" ] && nodes+=("$node")
  done
  [ "${#nodes[@]}" -gt 0 ]
}

echo
echo "=== GPU RUNTIME DIAGNOSTICS ==="
echo "  device files:"
nvidia_nodes=()
for nvidia_node in /dev/nvidia[0-9]* /dev/nvidiactl /dev/nvidia-uvm; do
  [ -e "$nvidia_node" ] && nvidia_nodes+=("$nvidia_node")
done
if [ "${#nvidia_nodes[@]}" -gt 0 ]; then
  ls -l "${nvidia_nodes[@]}" 2>&1 | sed 's/^/    /' | head -20 || true
else
  echo "    unavailable (no /dev/nvidia* device nodes mounted)"
fi
echo "  lspci (nvidia):"
lspci 2>/dev/null | grep -i nvidia | sed 's/^/    /' || echo "    (no lspci output)"
echo "  /proc/driver/nvidia/version:"
cat /proc/driver/nvidia/version 2>/dev/null | sed 's/^/    /' || echo "    not loaded"
echo "  nvidia-smi locations on disk:"
find /usr /opt -maxdepth 6 -name "nvidia-smi" 2>/dev/null | sed 's/^/    /' || true
echo "  libcuda.so locations:"
find /usr /opt -maxdepth 6 -name "libcuda.so*" 2>/dev/null | sed 's/^/    /' || true
echo "  nvidia/cuda packages installed:"
dpkg -l 2>/dev/null | grep -iE "nvidia|cuda" | awk '{print "    "$1, $2, $3}' | head -20 || echo "    (none reported by dpkg)"
echo "  nvidia-smi:"
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi -L 2>&1 | sed 's/^/    /' || true
  nvidia-smi --query-gpu=index,name,driver_version,memory.total,memory.free --format=csv,noheader,nounits 2>&1 | sed 's/^/    /' || true
else
  echo "    unavailable (not on PATH)"
fi
echo "  python CUDA runtimes:"
if has_nvidia_device_nodes; then
  .venv/bin/python - <<'PYEOF' 2>&1 | sed 's/^/    /' || true
try:
    import cupy
    print(f"cupy={cupy.__version__}")
    try:
        count = int(cupy.cuda.runtime.getDeviceCount())
        print(f"cupy_cuda_devices={count}")
        for device_id in range(count):
            with cupy.cuda.Device(device_id):
                free, total = cupy.cuda.runtime.memGetInfo()
            print(f"cupy_device{device_id}_memory={free}/{total}")
    except BaseException as exc:
        print(f"cupy_runtime_error={exc.__class__.__name__}: {exc}")
except BaseException as exc:
    print(f"cupy_import_error={exc.__class__.__name__}: {exc}")
try:
    import jax
    import jaxlib
    print(f"jax={jax.__version__} jaxlib={jaxlib.__version__} backend={jax.default_backend()}")
    print("jax_devices=" + ", ".join(f"{d.platform}:{getattr(d, 'id', '?')}:{getattr(d, 'device_kind', '?')}" for d in jax.devices()))
except BaseException as exc:
    print(f"jax_runtime_error={exc.__class__.__name__}: {exc}")
PYEOF
else
  echo "    skipped (no NVIDIA device nodes; importing CUDA runtimes can crash in this container state)"
fi
echo "=== END GPU RUNTIME DIAGNOSTICS ==="

BASE_SNP="$HOME/sv_pgs_results_snp"
BASE_JOINT="$HOME/sv_pgs_results_snp_sv"
# Single canonical cache dir. ~/.sv_pgs_cache is where every previous AoU run
# already wrote, since aou_runner uses work_dir.parent/.sv_pgs_cache and the
# historical work_dir lived directly under $HOME (e.g. ~/hypertension_results
# → cache at ~/.sv_pgs_cache). Reusing this avoids any re-download.
SHARED_CACHE="$HOME/.sv_pgs_cache"
mkdir -p "$BASE_SNP" "$BASE_JOINT" "$SHARED_CACHE"

# If a previous version of this script created ~/.sv_pgs_shared_cache, fold it
# into the canonical dir. Anything that already exists in the canonical cache
# wins (it's the trusted full copy); duplicates from the stray dir get dropped
# since they may be partial-download fragments.
PRIOR_BAD_SHARED="$HOME/.sv_pgs_shared_cache"
if [ -d "$PRIOR_BAD_SHARED" ] && [ ! -L "$PRIOR_BAD_SHARED" ]; then
  echo "  cache: folding stray $PRIOR_BAD_SHARED into $SHARED_CACHE"
  shopt -s nullglob
  for subdir in "$PRIOR_BAD_SHARED"/*; do
    name="$(basename "$subdir")"
    target="$SHARED_CACHE/$name"
    if [ -e "$target" ]; then
      echo "    dropping duplicate: $subdir (canonical exists at $target)"
      rm -rf "$subdir"
    else
      echo "    moving: $subdir -> $target"
      mv "$subdir" "$target"
    fi
  done
  shopt -u nullglob
  rmdir "$PRIOR_BAD_SHARED" 2>/dev/null || true
fi

# Point each per-variant base's cache dir at the canonical shared cache.
# Idempotent: if the link is already correct, return; if it points elsewhere
# replace it; if it's a real directory, migrate its contents (dropping
# duplicates that already exist in the shared dir) and replace with a symlink.
link_cache_to_shared() {
  local link="$1"
  if [ -L "$link" ]; then
    if [ "$(readlink "$link")" = "$SHARED_CACHE" ]; then
      return 0
    fi
    echo "  cache: relinking $link -> $SHARED_CACHE"
    rm "$link"
  elif [ -d "$link" ]; then
    echo "  cache: folding $link into $SHARED_CACHE"
    shopt -s nullglob
    for subdir in "$link"/*; do
      name="$(basename "$subdir")"
      target="$SHARED_CACHE/$name"
      if [ -e "$target" ]; then
        echo "    dropping duplicate: $subdir (canonical exists at $target)"
        rm -rf "$subdir"
      else
        echo "    moving: $subdir -> $target"
        mv "$subdir" "$target"
      fi
    done
    shopt -u nullglob
    rm -rf "$link"
  fi
  ln -sfn "$SHARED_CACHE" "$link"
}

plink_triplet_exists() {
  local prefix="$1"
  [ -f "$prefix.bed" ] && [ -f "$prefix.bim" ] && [ -f "$prefix.fam" ]
}

mounted_array_plink_prefix() {
  local prefix
  prefix="$HOME/workspace/vwb-aou-datasets-controlled/v8/microarray/plink/arrays"
  if plink_triplet_exists "$prefix"; then
    printf '%s\n' "$prefix"
    return 0
  fi
  return 1
}

link_mounted_array_plink() {
  local prefix
  if ! prefix="$(mounted_array_plink_prefix)"; then
    return 1
  fi
  local plink_dir="$SHARED_CACHE/aou_array_plink"
  mkdir -p "$plink_dir"
  echo "  PLINK: using mounted workspace resource"
  echo "    source: $prefix.{bed,bim,fam}"
  echo "    cache:  $plink_dir/arrays.{bed,bim,fam}"
  local ext target
  for ext in bed bim fam; do
    target="$plink_dir/arrays.$ext"
    if [ -f "$target" ]; then
      continue
    fi
    rm -f "$target"
    ln -s "$prefix.$ext" "$target"
  done
  return 0
}

link_cache_to_shared "$BASE_SNP/.sv_pgs_cache"
link_cache_to_shared "$BASE_JOINT/.sv_pgs_cache"
if link_mounted_array_plink; then
  :
fi

echo
echo "=== CACHE DIAGNOSTICS ==="
echo "  disk usage:"
df -h "$HOME" 2>/dev/null | sed 's/^/    /'
echo
echo "  block device backing \$HOME:"
home_src="$(df --output=source "$HOME" 2>/dev/null | tail -1 || true)"
home_dev=""
if [ -n "$home_src" ]; then
  home_dev="$(basename "$home_src" 2>/dev/null || true)"
fi
if [ -n "$home_dev" ]; then
  # Trim partition suffix (sdb1 -> sdb, nvme0n1p1 -> nvme0n1) so /sys/block lookup works.
  case "$home_dev" in
    nvme*p*) home_block="${home_dev%p*}" ;;
    *[0-9])  home_block="$(echo "$home_dev" | sed 's/[0-9]*$//')" ;;
    *)       home_block="$home_dev" ;;
  esac
  echo "    $home_src  (block device: /dev/$home_block)"
  if command -v lsblk >/dev/null 2>&1; then
    lsblk -d -o NAME,ROTA,SIZE,MODEL,TYPE --nodeps "/dev/$home_block" 2>/dev/null | sed 's/^/    /' || true
  fi
  rota_path="/sys/block/$home_block/queue/rotational"
  if [ -r "$rota_path" ]; then
    rota=$(cat "$rota_path" 2>/dev/null || echo "?")
    case "$rota" in
      0) media="SSD-backed (rotational=0)" ;;
      1) media="HDD-backed (rotational=1)" ;;
      *) media="rotational=$rota" ;;
    esac
    echo "    media: $media"
  fi
  # Local NVMe shows up as /dev/nvme*; PDs show up as /dev/sd* with MODEL=PersistentDisk.
  # The MODEL string is the cheap test: local SSDs read "EphemeralDisk" / "nvme_card" /
  # "Local-SSD"; persistent disks read "PersistentDisk".
  model="$(lsblk -dn -o MODEL "/dev/$home_block" 2>/dev/null | tr -d ' ' || true)"
  case "$model" in
    *PersistentDisk*) echo "    interpretation: GCE persistent disk (NETWORK-ATTACHED; throughput scales with provisioned size)" ;;
    *EphemeralDisk*|*Local-SSD*|*nvme_card*) echo "    interpretation: local NVMe / local SSD (direct-attached, non-persistent)" ;;
    *) [ -n "$model" ] && echo "    interpretation: unrecognized model='$model'" ;;
  esac
fi
echo
echo "  canonical shared cache ($SHARED_CACHE):"
if [ -d "$SHARED_CACHE" ]; then
  ls -lah "$SHARED_CACHE" 2>/dev/null | sed 's/^/    /'
  echo "  shared cache top-level sizes:"
  shopt -s nullglob
  shared_cache_entries=("$SHARED_CACHE"/*)
  shopt -u nullglob
  if [ "${#shared_cache_entries[@]}" -gt 0 ]; then
    du -sh "${shared_cache_entries[@]}" 2>/dev/null | sed 's/^/    /' || true
  else
    echo "    (empty)"
  fi
else
  echo "    (missing)"
fi
echo
echo "  PLINK trio dir ($SHARED_CACHE/aou_array_plink):"
plink_dir="$SHARED_CACHE/aou_array_plink"
if [ -d "$plink_dir" ]; then
  ls -lah "$plink_dir" 2>/dev/null | sed 's/^/    /'
  for ext in bed bim fam; do
    f="$plink_dir/arrays.$ext"
    if [ -f "$f" ]; then
      sz=$(stat -c%s "$f" 2>/dev/null || stat -f%z "$f" 2>/dev/null)
      echo "    OK : arrays.$ext  (${sz} bytes)"
    else
      echo "    MISSING: arrays.$ext"
    fi
  done
else
  echo "    (missing — full PLINK trio download would be required)"
fi
echo
echo "  PLINK int8 / stats caches at shared root:"
ls -lah "$SHARED_CACHE"/plink_int8_*.npy "$SHARED_CACHE"/plink_stats_*.npy 2>/dev/null | sed 's/^/    /' || \
  echo "    (none)"
echo
echo "  symlinks resolve to:"
for link in "$BASE_SNP/.sv_pgs_cache" "$BASE_JOINT/.sv_pgs_cache"; do
  if [ -L "$link" ]; then
    echo "    $link -> $(readlink -f "$link" 2>/dev/null || readlink "$link")"
  elif [ -d "$link" ]; then
    echo "    $link (real directory)"
  else
    echo "    $link (missing)"
  fi
done
echo
echo "  legacy single-disease cache (~/<disease>_results/.sv_pgs_cache):"
shopt -s nullglob
legacy_found=0
for legacy in "$HOME"/*_results/.sv_pgs_cache; do
  [ -e "$legacy" ] || continue
  legacy_found=1
  echo "    $legacy"
  ls -lah "$legacy" 2>/dev/null | sed 's/^/      /'
done
shopt -u nullglob
[ "$legacy_found" = "0" ] && echo "    (none)"
echo "=== END DIAGNOSTICS ==="
echo

# Abort early if the PLINK trio would trigger a re-download — the user has
# already downloaded this and the workspace doesn't have headroom for it.
plink_dir="$SHARED_CACHE/aou_array_plink"
missing_plink=()
for ext in bed bim fam; do
  [ -f "$plink_dir/arrays.$ext" ] || missing_plink+=("arrays.$ext")
done
if [ "${#missing_plink[@]}" -gt 0 ]; then
  echo "ABORT: canonical PLINK trio at $plink_dir is incomplete (missing: ${missing_plink[*]})."
  echo "       run.sh refuses to download — find or restore the missing files first."
  echo "       hint: search for stray copies on disk:"
  echo "         find $HOME -maxdepth 6 -name 'arrays.bed' -o -name 'arrays.bim' -o -name 'arrays.fam' 2>/dev/null"
  exit 2
fi

is_all() {
  [ "$DISEASE" = "all" ] || [ "$DISEASE" = "top20" ]
}

list_diseases() {
  if is_all; then
    uv run python -c "from sv_pgs.all_of_us import DISEASE_DEFINITIONS; print('\n'.join(d.canonical_name for d in DISEASE_DEFINITIONS))"
  else
    echo "$DISEASE"
  fi
}

sweep_zombie_gpu_procs() {
  # Print GPU owners without killing anything. Earlier versions attempted to
  # reclaim stale CUDA contexts here, but in notebook/agent environments a
  # same-user Python process can be important even when it looks orphaned.
  command -v nvidia-smi >/dev/null 2>&1 || return 0
  local me
  me=$(id -un)
  local raw_pids
  raw_pids=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | tr -d ' ')
  [ -z "$raw_pids" ] && return 0
  for pid in $raw_pids; do
    [ -z "$pid" ] && continue
    local owner cmd
    owner=$(ps -o user= -p "$pid" 2>/dev/null | tr -d ' ' || true)
    [ "$owner" = "$me" ] || continue
    cmd=$(ps -o args= -p "$pid" 2>/dev/null || true)
    echo "  GPU owner: pid=$pid cmd=$(printf '%s' "$cmd" | head -c 100)"
  done
}

sweep_zombie_sv_pgs_procs() {
  # Diagnostics only. Do not kill same-user Python processes; notebooks and
  # agent sessions often use the same venv/cmdline markers as this run.
  local me
  me=$(id -un)
  local candidates
  candidates=$(pgrep -u "$me" -f "sv-pgs|run-all-of-us" 2>/dev/null || true)
  [ -z "$candidates" ] && return 0
  for pid in $candidates; do
    [ -z "$pid" ] && continue
    local cmd rss_kb stat
    cmd=$(ps -o args= -p "$pid" 2>/dev/null || true)
    rss_kb=$(ps -o rss= -p "$pid" 2>/dev/null | tr -d ' ' || echo 0)
    stat=$(ps -o stat= -p "$pid" 2>/dev/null | tr -d ' ' || true)
    [ -z "$cmd" ] && continue
    echo "  sv-pgs process present: pid=$pid rss=${rss_kb}kB stat=${stat:-?} cmd=$(printf '%s' "$cmd" | head -c 100)"
  done
}

kill_stuck_or_running_sv_pgs() {
  # Auto-tear-down: a prior AoU fit (running, stopped via ^Z, or stuck in
  # uninterruptible disk wait) holds >10 GB RSS and full GPU contexts. We
  # used to refuse to start when one was found; that left users stuck doing
  # "ps + kill -9" loops. Now we SIGCONT-then-SIGKILL the prior run + its
  # entire process tree (uv run -> python -> any background readers), wait
  # briefly for the kernel to reap, and then start fresh.
  if [ "${SV_PGS_SKIP_STARTUP_SWEEP:-}" = "1" ]; then
    echo "  startup sweep skipped because SV_PGS_SKIP_STARTUP_SWEEP=1"
    return 0
  fi
  local me
  me=$(id -un)
  local candidates
  candidates=$(pgrep -u "$me" -f "sv-pgs|run-all-of-us" 2>/dev/null || true)
  [ -z "$candidates" ] && return 0
  local victims=()
  for pid in $candidates; do
    [ -z "$pid" ] && continue
    [ "$pid" = "$$" ] && continue
    local cmd stat rss_kb
    cmd=$(ps -o args= -p "$pid" 2>/dev/null || true)
    stat=$(ps -o stat= -p "$pid" 2>/dev/null | tr -d ' ' || true)
    rss_kb=$(ps -o rss= -p "$pid" 2>/dev/null | tr -d ' ' || echo 0)
    [ -z "$cmd" ] && continue
    case "$stat" in
      *Z*) continue ;;  # zombie — kernel will reap when parent waits
    esac
    case "$cmd" in
      *"sv-pgs run-all-of-us"*|*"run-all-of-us --"*|*"sv-pgs"*"--disease"*)
        victims+=("$pid")
        echo "  killing stale sv-pgs: pid=$pid rss=${rss_kb}kB stat=$stat cmd=$(printf '%s' "$cmd" | head -c 120)"
        ;;
    esac
  done
  [ "${#victims[@]}" -eq 0 ] && return 0
  # Phase 1: SIGCONT in case the process was ^Z-stopped (otherwise SIGTERM
  # is queued but never delivered until it resumes). Then SIGTERM for a
  # graceful Python shutdown (atexit handlers, GPU contexts, ThreadPool
  # joins). Walk children/grandchildren too so we don't leave decode
  # workers running on the GPU.
  local all_pids=()
  for pid in "${victims[@]}"; do
    all_pids+=("$pid")
    # Children + grandchildren via pgrep -P (BFS one level at a time).
    local kids
    kids=$(pgrep -P "$pid" 2>/dev/null || true)
    for k in $kids; do
      all_pids+=("$k")
      local gk
      gk=$(pgrep -P "$k" 2>/dev/null || true)
      for g in $gk; do all_pids+=("$g"); done
    done
  done
  for pid in "${all_pids[@]}"; do
    kill -SIGCONT "$pid" 2>/dev/null || true
    kill -SIGTERM "$pid" 2>/dev/null || true
  done
  # Phase 2: wait briefly for graceful shutdown.
  local waited=0
  while [ "$waited" -lt 30 ]; do
    local still_alive=0
    for pid in "${all_pids[@]}"; do
      local stat_now
      stat_now=$(ps -o stat= -p "$pid" 2>/dev/null | tr -d ' ' || true)
      case "$stat_now" in
        ""|*Z*) continue ;;
      esac
      kill -0 "$pid" 2>/dev/null && still_alive=$((still_alive + 1))
    done
    [ "$still_alive" -eq 0 ] && break
    sleep 1
    waited=$((waited + 1))
  done
  # Phase 3: anyone still alive after the grace window gets SIGKILL.
  for pid in "${all_pids[@]}"; do
    local stat_now
    stat_now=$(ps -o stat= -p "$pid" 2>/dev/null | tr -d ' ' || true)
    case "$stat_now" in
      ""|*Z*) continue ;;
    esac
    if kill -0 "$pid" 2>/dev/null; then
      echo "  SIGKILL stale pid=$pid (did not exit within 30s of SIGTERM)"
      kill -SIGKILL "$pid" 2>/dev/null || true
    fi
  done
  sleep 1
  # Phase 4: report any holdouts (rare; usually disk-wait threads that the
  # kernel will reap on their next I/O return).
  local survivors=0
  for pid in "${all_pids[@]}"; do
    local stat_now
    stat_now=$(ps -o stat= -p "$pid" 2>/dev/null | tr -d ' ' || true)
    case "$stat_now" in
      ""|*Z*) continue ;;
    esac
    kill -0 "$pid" 2>/dev/null && survivors=$((survivors + 1))
  done
  if [ "$survivors" -gt 0 ]; then
    echo "  WARNING: $survivors process(es) still alive after SIGKILL; continuing anyway"
  fi
  return 0
}

# Backwards-compatible alias so any external caller / cron still works.
abort_if_sv_pgs_running() {
  kill_stuck_or_running_sv_pgs
}

postmortem_after_silent_death() {
  # Called when uv run sv-pgs exits non-zero without a Python traceback
  # surfacing in the terminal. Dumps everything that explains a silent
  # death: exit/signal, kernel OOM-killer log, memory state, GPU state.
  local rc="$1"
  local signal_name=""
  if [ "$rc" -gt 128 ]; then
    signal_name=" (signal $((rc - 128))$(kill -l $((rc - 128)) 2>/dev/null | sed 's/^/=SIG/'))"
  fi
  echo
  echo "=== POSTMORTEM: sv-pgs exited rc=${rc}${signal_name} ==="
  echo "  --- memory state now ---"
  awk '/^MemTotal|^MemFree|^MemAvailable|^Buffers|^Cached|^SwapTotal|^SwapFree/ {print "    "$0}' /proc/meminfo 2>/dev/null
  echo "  --- top RSS processes (own user) ---"
  ps -u "$(id -un)" -o pid,ppid,rss,vsz,stat,cmd --sort=-rss 2>/dev/null | head -10 | sed 's/^/    /'
  echo "  --- kernel OOM-killer hits (last 200 dmesg lines) ---"
  if dmesg -T 2>/dev/null | tail -200 | grep -iE "killed process|out of memory|oom-kill|invoked oom" | tail -20 | sed 's/^/    /'; then :; fi
  echo "  --- journalctl kernel oom (last 200 lines) ---"
  journalctl -k -n 200 --no-pager 2>/dev/null | grep -iE "killed|oom" | tail -20 | sed 's/^/    /' || echo "    (journalctl unavailable)"
  echo "  --- GPU state now ---"
  if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=index,memory.used,memory.free,utilization.gpu --format=csv,noheader 2>/dev/null | sed 's/^/    /'
    nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader 2>/dev/null | sed 's/^/    /' || true
  fi
  echo "  --- last 50 lines of fit checkpoint dir (if any artifacts written) ---"
  if [ -n "${SV_PGS_LAST_OUTDIR:-}" ] && [ -d "${SV_PGS_LAST_OUTDIR}" ]; then
    ls -laht "${SV_PGS_LAST_OUTDIR}" 2>/dev/null | head -10 | sed 's/^/    /'
  fi
  echo "=== END POSTMORTEM ==="
}

build_aou_env_args() {
  # Build an `env -i` argv that scrubs the inherited environment down to the
  # minimum the AoU fit actually needs. Stale exports from prior runs (e.g.
  # CUDA_VISIBLE_DEVICES=0 left over from a single-GPU pin, XLA_FLAGS that
  # disabled tensor cores, MALLOC_ARENA_MAX=1 fragmenting allocator, half a
  # JAX_PLATFORMS=cpu override forgotten in .bashrc) are the kind of thing
  # that silently make a run slower or kill it. Whitelisting is far safer
  # than blocklisting: if a future env var becomes load-bearing we'll see
  # it missing immediately rather than rotting silently behind a leftover.
  local keep=(
    # Shell context — without these uv / python can't even find HOME or the user's name.
    HOME USER LOGNAME SHELL TERM LANG LC_ALL LC_CTYPE TZ
    # PATH must include the venv launcher dir + uv install dir + system bins + nvidia tools.
    PATH
    # CUDA / NVIDIA — let the driver, runtime, and CuPy find each other.
    LD_LIBRARY_PATH CUDA_HOME CUDA_PATH NVIDIA_VISIBLE_DEVICES NVIDIA_DRIVER_CAPABILITIES
    # AoU / GCP credentials and workspace — BigQuery + GCS access for phenotype + cohort.
    GOOGLE_APPLICATION_CREDENTIALS GOOGLE_CLOUD_PROJECT GOOGLE_PROJECT
    WORKSPACE_BUCKET WORKSPACE_CDR WORKSPACE_NAMESPACE WORKSPACE_ID
    CDR_STORAGE_PATH WGS_GVCF_PATH WGS_VCF_PATH WGS_VDS_PATH WGS_HAIL_STORAGE_PATH
    MICROARRAY_HAIL_STORAGE_PATH MICROARRAY_VDS_PATH MICROARRAY_PLINK_PATH
    OWNER_EMAIL CLOUDSDK_CONFIG CLOUDSDK_CORE_PROJECT
    # Temp dir override (keeps /tmp off the small root fs if user moved it).
    TMPDIR
    # uv cache reuse — without this uv re-resolves and re-installs every call.
    UV_CACHE_DIR
    # Our own knobs — postmortem dir + sweep override.
    SV_PGS_LAST_OUTDIR SV_PGS_SKIP_STARTUP_SWEEP SV_PGS_SKIP_PULL
  )
  local env_args=()
  local name value
  for name in "${keep[@]}"; do
    if [ -n "${!name+x}" ]; then
      env_args+=("$name=${!name}")
    fi
  done
  printf '%s\n' "${env_args[@]}"
}

run_variant_pass() {
  local variants="$1"
  local base="$2"
  echo "=== FIT PASS: variants=${variants}  base=${base} ==="
  sweep_zombie_sv_pgs_procs
  sweep_zombie_gpu_procs
  abort_if_sv_pgs_running
  local rc=0
  # Scrub env down to the AoU allowlist so inherited junk (stale CUDA_VISIBLE_DEVICES,
  # forgotten XLA_FLAGS=..., MALLOC_ARENA_MAX, JAX_PLATFORMS=cpu, etc.) can't silently
  # change the run. mapfile reads the allowlist; `env -i` clears, then re-sets only those.
  local env_args=()
  mapfile -t env_args < <(build_aou_env_args)
  echo "  env scrub: passing ${#env_args[@]} whitelisted vars to sv-pgs (stripped $(env | wc -l) inherited)"
  # Run unfaulted so we can inspect rc; restore -e immediately after.
  set +e
  if is_all; then
    export SV_PGS_LAST_OUTDIR="$base"
    env -i "${env_args[@]}" SV_PGS_LAST_OUTDIR="$base" \
      uv run sv-pgs run-all-of-us --all-diseases --variants "$variants" --output-dir "$base"
    rc=$?
  else
    local out="$base/${DISEASE}_results"
    mkdir -p "$out"
    export SV_PGS_LAST_OUTDIR="$out"
    env -i "${env_args[@]}" SV_PGS_LAST_OUTDIR="$out" \
      uv run sv-pgs run-all-of-us --disease "$DISEASE" --variants "$variants" --output-dir "$out"
    rc=$?
  fi
  set -e
  if [ "$rc" -ne 0 ]; then
    postmortem_after_silent_death "$rc"
    return "$rc"
  fi
}

disease_outdir() {
  local base="$1"
  local d="$2"
  echo "$base/${d}_results"
}

run_variant_pass snp     "$BASE_SNP"
run_variant_pass snp+sv  "$BASE_JOINT"

echo
echo "=== EVALUATION PASS ==="
mapfile -t DISEASES < <(list_diseases)
for d in "${DISEASES[@]}"; do
  for base in "$BASE_SNP" "$BASE_JOINT"; do
    out="$(disease_outdir "$base" "$d")"
    if [ -f "$out/summary.json.gz" ]; then
      uv run sv-pgs evaluate-all-of-us --disease "$d" --output-dir "$out" || \
        echo "  (evaluation failed for ${d} in ${out})"
    else
      echo "  skipping evaluation: $out/summary.json.gz not found"
    fi
  done
done

echo
echo "=== SNP-only vs SNP+SV COMPARISON ==="
uv run python - "$BASE_SNP" "$BASE_JOINT" "${DISEASES[@]}" <<'PYEOF'
import json, sys
from pathlib import Path

base_snp = Path(sys.argv[1])
base_joint = Path(sys.argv[2])
diseases = sys.argv[3:]

def _load(base: Path, disease: str) -> dict:
    candidate = base / f"{disease}_results" / f"{disease}.evaluation.json"
    if not candidate.exists():
        return {}
    try:
        return json.loads(candidate.read_text())
    except Exception:
        return {}

def _fmt(v) -> str:
    if v is None:
        return "    NA"
    try:
        return f"{float(v):6.4f}"
    except (TypeError, ValueError):
        return "    NA"

header = f"{'disease':<26}  {'snp_AUC':>8}  {'snp+sv_AUC':>10}  {'Δ(sv-snp)':>10}  {'snp_gen':>8}  {'snp+sv_gen':>10}"
print(header)
print("-" * len(header))
for d in diseases:
    a = _load(base_snp, d)
    b = _load(base_joint, d)
    auc_a = a.get("test_holdout_auc_all")
    auc_b = b.get("test_holdout_auc_all")
    gen_a = a.get("test_holdout_auc_genetic_only")
    gen_b = b.get("test_holdout_auc_genetic_only")
    try:
        delta = f"{float(auc_b) - float(auc_a):+.4f}"
    except (TypeError, ValueError):
        delta = "    NA"
    print(f"{d:<26}  {_fmt(auc_a):>8}  {_fmt(auc_b):>10}  {delta:>10}  {_fmt(gen_a):>8}  {_fmt(gen_b):>10}")
PYEOF
