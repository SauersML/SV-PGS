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
if [ ! -x ".venv/bin/python" ]; then
  uv sync --python 3.12 --extra gpu
fi
uv run sv-pgs run-all-of-us --disease "$DISEASE" --output-dir "$HOME/${DISEASE}_results" "$@"
