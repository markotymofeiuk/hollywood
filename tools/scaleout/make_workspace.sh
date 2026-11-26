#!/usr/bin/env bash
# Copy the generator (code + shipped banks) into a fresh run workspace.
# The pipeline writes all outputs next to its own scripts, so runs must not
# happen inside the repository's generator/ directory.
#
# usage: tools/scaleout/make_workspace.sh <dest-dir>
set -euo pipefail
repo="$(cd "$(dirname "$0")/../.." && pwd)"
dest="${1:?usage: make_workspace.sh <dest-dir>}"
# An existing directory is fine as long as it holds no workspace yet (run_sf.sh
# --detach creates it first for its log file).
if [[ -e "$dest/run_pipeline.py" ]]; then
  echo "refusing to overwrite existing workspace $dest" >&2
  exit 1
fi
mkdir -p "$dest"
excludes=(
  __pycache__ entities graph _step100_resume imdb_schema reports
  decision_logs checkpoints critic '*.arrow' .pipeline_checkpoint.json
)
if command -v rsync >/dev/null 2>&1; then
  args=()
  for pattern in "${excludes[@]}"; do args+=(--exclude "$pattern"); done
  rsync -a "${args[@]}" "$repo/generator/" "$dest/"
else
  args=()
  for pattern in "${excludes[@]}"; do args+=(--exclude "$pattern"); done
  (cd "$repo/generator" && tar -cf - "${args[@]}" .) | (cd "$dest" && tar -xf -)
fi
echo "workspace ready: $dest"
