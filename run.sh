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
ls -l /dev/nvidia* 2>/dev/null | sed 's/^/    /' || echo "    (none)"
echo "  nvidia-smi:"
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi -L 2>&1 | sed 's/^/    /' || true
  nvidia-smi --query-gpu=index,name,driver_version,memory.total,memory.free --format=csv,noheader,nounits 2>&1 | sed 's/^/    /' || true
else
  echo "    unavailable"
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
  # Reclaim GPU memory held by stale CUDA contexts from prior killed runs.
  # A process is killed only if ALL of these hold:
  #   * it owns GPU memory now (nvidia-smi reports it)
  #   * it belongs to the current user
  #   * it is NOT an ancestor of this shell (so we never SIGKILL sshd / jupyter)
  #   * it is EITHER orphaned (ppid==1) — its launching shell died, leaving
  #     a stuck CUDA context — OR its cmdline clearly belongs to this project
  #     (matches sv-pgs / run-all-of-us / this repo's .venv path).
  # That last clause is the important one for shared boxes: another notebook
  # or another GPU job that the same user legitimately launched will not be
  # touched.
  command -v nvidia-smi >/dev/null 2>&1 || return 0
  local self_pid=$$
  local skip_pids=" $self_pid "
  local p="$self_pid"
  while [ -n "$p" ] && [ "$p" != "1" ] && [ "$p" != "0" ]; do
    p=$(ps -o ppid= -p "$p" 2>/dev/null | tr -d ' ' || true)
    [ -n "$p" ] && skip_pids+="$p "
  done
  local me
  me=$(id -un)
  local raw_pids
  raw_pids=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | tr -d ' ')
  [ -z "$raw_pids" ] && return 0
  local repo_marker="${REPO_DIR}/.venv"
  local killed=0
  local skipped_other=0
  for pid in $raw_pids; do
    [ -z "$pid" ] && continue
    case "$skip_pids" in *" $pid "*) continue ;; esac
    local owner ppid cmd reason=""
    owner=$(ps -o user= -p "$pid" 2>/dev/null | tr -d ' ' || true)
    [ "$owner" = "$me" ] || continue
    ppid=$(ps -o ppid= -p "$pid" 2>/dev/null | tr -d ' ' || true)
    cmd=$(ps -o args= -p "$pid" 2>/dev/null || true)
    if [ "$ppid" = "1" ]; then
      reason="orphaned (ppid=1)"
    elif printf '%s' "$cmd" | grep -qE "sv-pgs|run-all-of-us|${repo_marker}"; then
      reason="matches sv-pgs cmdline"
    else
      echo "  GPU sweep: leaving pid=$pid alone (foreign GPU job: $(printf '%s' "$cmd" | head -c 80))"
      skipped_other=1
      continue
    fi
    echo "  GPU sweep: killing stale GPU process pid=$pid reason='$reason' cmd=$(printf '%s' "$cmd" | head -c 80)"
    kill -9 "$pid" 2>/dev/null || true
    killed=1
  done
  if [ "$killed" = "1" ]; then
    # Driver needs a moment to reclaim the memory.
    sleep 3
    echo "  GPU sweep: state after reclaim:"
    nvidia-smi --query-gpu=memory.used,memory.free --format=csv,noheader 2>/dev/null | sed 's/^/    /'
  fi
  if [ "$skipped_other" = "1" ]; then
    echo "  GPU sweep: at least one foreign GPU process remains; expect reduced GPU memory budget."
  fi
}

run_variant_pass() {
  local variants="$1"
  local base="$2"
  echo "=== FIT PASS: variants=${variants}  base=${base} ==="
  sweep_zombie_gpu_procs
  if is_all; then
    uv run sv-pgs run-all-of-us --all-diseases --variants "$variants" --output-dir "$base"
  else
    local out="$base/${DISEASE}_results"
    mkdir -p "$out"
    uv run sv-pgs run-all-of-us --disease "$DISEASE" --variants "$variants" --output-dir "$out"
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
