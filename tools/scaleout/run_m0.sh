#!/usr/bin/env bash
# Run one M0 measurement: fresh workspace, fake LLM, audits on, tree-RSS monitor.
#
# usage: tools/scaleout/run_m0.sh <workspace-dir> <profile.env> [extra run_pipeline args...]
# Set M0_TAG to name the log/monitor files (default: run) and M0_REUSE=1 to
# run more steps in an existing workspace.
set -euo pipefail
repo="$(cd "$(dirname "$0")/../.." && pwd)"
ws="${1:?workspace dir}"; profile="${2:?profile file}"; shift 2
py="$repo/.venv/bin/python"
tag="${M0_TAG:-run}"
if [[ "${M0_REUSE:-0}" != "1" ]]; then
  "$repo/tools/scaleout/make_workspace.sh" "$ws"
fi
cd "$ws"
export LLM_PROVIDER=fake
export DATA_SYS_LLM_MAX_ATTEMPTS=1
export DATA_SYS_SPEED_AUDIT=1
export DATA_SYS_SPEED_AUDIT_DIR="$ws/reports/speed_audit"
export DATA_SYS_MEMORY_AUDIT=1
export DATA_SYS_MEMORY_AUDIT_DIR="$ws/reports/memory_audit"
export PYTHONUNBUFFERED=1
# Step 90 runs a diagnostic O(n^2) near-duplicate name check below this size.
export DATA_SYS_NEAR_DUPLICATE_LIMIT="${DATA_SYS_NEAR_DUPLICATE_LIMIT:-0}"
set +e
"$py" "$repo/tools/scaleout/monitor.py" --out "$ws/m0_monitor_$tag" --interval 1 -- \
  "$py" run_pipeline.py --profile-file "$profile" "$@" > "$ws/$tag.log" 2>&1
status=$?
set -e
"$py" "$repo/tools/scaleout/summarize_run.py" "$ws" > "$ws/m0_summary_$tag.md" || true
echo "exit=$status unhandled=$(ls "$ws/fake_llm_unhandled" 2>/dev/null | wc -l | tr -d ' ')"
exit $status
