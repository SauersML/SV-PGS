#!/bin/bash
# Never run by an agent: staging is the user's step, from a Workbench app with internet access inside the workspace.
# Stages what every Batch job builds its environment from under gs://${OPS_BUCKET}/svpgs/${RUN_ID}/:
#   source/          this repository at HEAD (git archive)
#   bin/uv           the uv binary
#   python/          a uv-installed CPython 3.12
#   uv-cache/        every locked wheel, the gpu extra and the dev group (the GPU check's pytest)
#   run_config.json  the filled run_config.template.json, refused while a ${PLACEHOLDER} or a template key is missing
#
#   OPS_BUCKET=... RUN_ID=... launcher/workspace/stage.sh my_run_config.json
set -euo pipefail
: "${OPS_BUCKET:?}" "${RUN_ID:?}"
config=${1:?usage: stage.sh RUN_CONFIG.json}
here=$(cd "$(dirname "$0")" && pwd)
python3 - "$config" "$here/run_config.template.json" <<'PY'
import json, re, sys
filled_text = open(sys.argv[1]).read()
if "${" in filled_text:
    sys.exit(f"{sys.argv[1]} still holds placeholders: {sorted(set(re.findall(r'\$\{[A-Z_0-9]+\}', filled_text)))}")
filled = json.loads(filled_text)
template = json.loads(re.sub(r'\$\{[A-Z_0-9]+\}', "0", open(sys.argv[2]).read()))
missing = sorted(set(template) - set(filled))
if missing:
    sys.exit(f"{sys.argv[1]} lacks the template's keys {missing}")
PY
# The wheels and the interpreter must be the Batch VMs' platform.
[ "$(uname -sm)" = "Linux x86_64" ] || { echo "stage from Linux x86-64, the Batch VMs' platform" >&2; exit 2; }
root=$(git -C "$here" rev-parse --show-toplevel)
[ -z "$(git -C "$root" status --porcelain --untracked-files=no)" ] || { echo "commit or discard local changes: git archive stages HEAD only" >&2; exit 2; }
work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT
mkdir -p "$work/source" "$work/bin"
git -C "$root" archive HEAD | tar -x -C "$work/source"
cp "$(command -v uv)" "$work/bin/uv"
export UV_PYTHON_INSTALL_DIR="$work/python" UV_CACHE_DIR="$work/uv-cache"
uv python install 3.12
# Fills the cache; the venv it builds here is not staged (each job builds its own from the cache, offline).
(cd "$work/source" && UV_PROJECT_ENVIRONMENT="$work/venv" uv sync --frozen --extra gpu --python 3.12)
cp "$config" "$work/run_config.json"
destination="gs://${OPS_BUCKET}/svpgs/${RUN_ID}"
# each part removed before its copy, so a restage at a later commit leaves no file of the earlier one behind
for part in source bin python uv-cache; do
  if gcloud storage ls "$destination/$part" > /dev/null 2>&1; then gcloud storage rm -r "$destination/$part"; fi
  gcloud storage cp -r "$work/$part" "$destination/"
done
gcloud storage cp "$work/run_config.json" "$destination/run_config.json"
echo "staged $(git -C "$root" rev-parse --short HEAD) at $destination"
