#!/usr/bin/env bash
set -euo pipefail

DISEASE="${1:-hypertension}"
REPO_DIR="$HOME/SV-PGS"

if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
fi
export PATH="$HOME/.local/bin:$PATH"

if [ ! -d "$REPO_DIR/.git" ]; then
  git clone https://github.com/SauersML/SV-PGS.git "$REPO_DIR"
fi

cd "$REPO_DIR"
git pull
uv sync --python 3.12 --extra gpu
uv run sv-pgs run-all-of-us --disease "$DISEASE" --output-dir "$HOME/${DISEASE}_results"
