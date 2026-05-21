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
if [ ! -x ".venv/bin/python" ]; then
  uv sync --python 3.12 --extra gpu
fi

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

run_variant_pass() {
  local variants="$1"
  local base="$2"
  echo "=== FIT PASS: variants=${variants}  base=${base} ==="
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
