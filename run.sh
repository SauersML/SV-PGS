#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   ./run.sh                 # all 19 SNOMED diseases, both SNP-only and SNP+SV
#   ./run.sh hypertension    # one disease, both variant sets
#   ./run.sh all             # explicit "all"
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

echo
echo "=== GPU RUNTIME DIAGNOSTICS ==="
echo "  device files:"
ls -l /dev/nvidia* /dev/nvidiactl /dev/nvidia-uvm 2>&1 | sed 's/^/    /' | head -20
echo "  lspci (nvidia):"
lspci 2>/dev/null | grep -i nvidia | sed 's/^/    /' || echo "    (no lspci output)"
echo "  /proc/driver/nvidia/version:"
cat /proc/driver/nvidia/version 2>/dev/null | sed 's/^/    /' || echo "    not loaded"
echo "  nvidia-smi locations on disk:"
find /usr /opt -maxdepth 6 -name "nvidia-smi" 2>/dev/null | sed 's/^/    /' || true
echo "  libcuda.so locations:"
find /usr /opt -maxdepth 6 -name "libcuda.so*" 2>/dev/null | sed 's/^/    /' || true
echo "  nvidia/cuda packages installed:"
dpkg -l 2>/dev/null | grep -iE "nvidia|cuda" | awk '{print "    "$1, $2, $3}' | head -20
echo "  nvidia-smi:"
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi -L 2>&1 | sed 's/^/    /' || true
  nvidia-smi --query-gpu=index,name,driver_version,memory.total,memory.free --format=csv,noheader,nounits 2>&1 | sed 's/^/    /' || true
else
  echo "    unavailable (not on PATH)"
fi
echo "  python CUDA runtimes:"
.venv/bin/python - <<'PYEOF' 2>&1 | sed 's/^/    /'
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

link_cache_to_shared "$BASE_SNP/.sv_pgs_cache"
link_cache_to_shared "$BASE_JOINT/.sv_pgs_cache"

echo
echo "=== CACHE DIAGNOSTICS ==="
echo "  disk usage:"
df -h "$HOME" 2>/dev/null | sed 's/^/    /'
echo
echo "  block device backing \$HOME:"
home_src="$(df --output=source "$HOME" 2>/dev/null | tail -1)"
home_dev="$(basename "$home_src" 2>/dev/null)"
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
  du -sh "$SHARED_CACHE"/* 2>/dev/null | sed 's/^/    /'
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

abort_if_sv_pgs_running() {
  # Refuse overlap. A previous AoU fit can hold >10 GB RSS while this script
  # starts a second run; on 30 GB hosts that makes the new process look like it
  # "randomly" died with SIGKILL during the first GPU streaming pass.
  local me
  me=$(id -un)
  local candidates
  candidates=$(pgrep -u "$me" -f "sv-pgs|run-all-of-us" 2>/dev/null || true)
  [ -z "$candidates" ] && return 0
  local found=0
  for pid in $candidates; do
    [ -z "$pid" ] && continue
    [ "$pid" = "$$" ] && continue
    local cmd rss_kb stat
    cmd=$(ps -o args= -p "$pid" 2>/dev/null || true)
    stat=$(ps -o stat= -p "$pid" 2>/dev/null | tr -d ' ' || true)
    rss_kb=$(ps -o rss= -p "$pid" 2>/dev/null | tr -d ' ' || echo 0)
    [ -z "$cmd" ] && continue
    case "$stat" in
      *Z*) continue ;;
    esac
    case "$cmd" in
      *"sv-pgs run-all-of-us"*|*"run-all-of-us --"*)
        found=1
        echo "  blocking live sv-pgs run: pid=$pid rss=${rss_kb}kB stat=$stat cmd=$(printf '%s' "$cmd" | head -c 120)"
        ;;
    esac
  done
  if [ "$found" -ne 0 ]; then
    echo "ERROR: another sv-pgs AoU fit is already running. Stop it or wait for it before starting a new run."
    return 1
  fi
  return 0
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
