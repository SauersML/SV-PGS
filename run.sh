#!/usr/bin/env bash
set -euo pipefail

DISEASE="${1:-hypertension}"
if [ "$#" -gt 0 ]; then
  shift
fi
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
OUTPUT_DIR="$HOME/${DISEASE}_results"
uv run sv-pgs run-all-of-us --disease "$DISEASE" --output-dir "$OUTPUT_DIR" "$@"
uv run sv-pgs evaluate-all-of-us --disease "$DISEASE" --output-dir "$OUTPUT_DIR"
