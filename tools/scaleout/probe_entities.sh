#!/usr/bin/env bash
# Capacity probe: run the LLM-free entity steps one by one at a target size.
# Each step is monitored separately; a failing step does not stop the others.
#
# usage: tools/scaleout/probe_entities.sh <workspace-dir> <profile.env>
set -uo pipefail
repo="$(cd "$(dirname "$0")/../.." && pwd)"
ws="${1:?workspace dir}"; profile="${2:?profile file}"
"$repo/tools/scaleout/make_workspace.sh" "$ws" || exit 1
for step in 10 20 30 50 60; do
  M0_REUSE=1 M0_TAG="step$step" "$repo/tools/scaleout/run_m0.sh" "$ws" "$profile" --only "$step"
  echo "step $step -> exit $?"
done
