#!/usr/bin/env bash
# Real LLM run on the Albert API: fresh workspace, monitor, summary.
#
# usage: tools/scaleout/run_real.sh <workspace-dir> <profile.env> [extra run_pipeline args...]
# M0_TAG / M0_REUSE work as in run_m0.sh (resume: M0_REUSE=1, same dir).
set -euo pipefail
repo="$(cd "$(dirname "$0")/../.." && pwd)"
ws="${1:?workspace dir}"; profile="${2:?profile file}"; shift 2
py="$repo/.venv/bin/python"
tag="${M0_TAG:-run}"
if [[ ! -f "$repo/.env" ]]; then echo "missing $repo/.env with LOCAL_LLM_* settings" >&2; exit 1; fi
set -a
# shellcheck disable=SC1091
source "$repo/.env"
# shellcheck disable=SC1091
source "$repo/tools/scaleout/albert.env"
set +a
if [[ "${M0_REUSE:-0}" != "1" ]]; then
  "$repo/tools/scaleout/make_workspace.sh" "$ws"
fi
cd "$ws"
export PYTHONUNBUFFERED=1
export DATA_SYS_NEAR_DUPLICATE_LIMIT="${DATA_SYS_NEAR_DUPLICATE_LIMIT:-0}"
set +e
"$py" "$repo/tools/scaleout/monitor.py" --out "$ws/monitor_$tag" --interval 5 -- \
  "$py" run_pipeline.py --profile-file "$profile" "$@" > "$ws/$tag.log" 2>&1
status=$?
set -e
"$py" "$repo/tools/scaleout/summarize_run.py" "$ws" > "$ws/summary_$tag.md" || true
echo "exit=$status"
exit $status
