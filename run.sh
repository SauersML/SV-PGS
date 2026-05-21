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
mkdir -p "$BASE_SNP" "$BASE_JOINT"

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
