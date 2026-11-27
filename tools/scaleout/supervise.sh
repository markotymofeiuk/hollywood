#!/usr/bin/env bash
# Keep a generation run going without anyone watching.
#
#   supervise.sh <profile> <workspace> [--start new|resume] [--max-restarts N] [--pause SEC]
#
# Waits for the run's driver (run_sf.sh --_inner) to end. If the pipeline
# exited with a non-zero code (network outage, API errors that outlasted the
# step's own retries, a killed process), it waits --pause seconds and resumes
# the run in the same workspace: finished steps are skipped, LLM steps keep
# their shards, step 100 continues from its last completed year. It stops
# when the pipeline succeeds, after --max-restarts resumes, or when the file
# <workspace>/STOP exists (create it before stopping a run on purpose).
#
# --start new|resume launches the first driver itself (run_sf.sh --detach uses
# this); without it the supervisor attaches to a driver that is already running.
# Log: <workspace>/supervisor.log
set -uo pipefail

here="$(cd "$(dirname "$0")" && pwd)"
profile="${1:?profile}"; ws="${2:?workspace}"; shift 2
start=""; max_restarts=8; pause=300
while [[ $# -gt 0 ]]; do
  case "$1" in
    --start) start="${2:?}"; shift ;;
    --max-restarts) max_restarts="${2:?}"; shift ;;
    --pause) pause="${2:?}"; shift ;;
    *) echo "unknown option: $1" >&2; exit 2 ;;
  esac
  shift
done

log() { echo "[$(date '+%F %T')] supervisor: $*" >> "$ws/supervisor.log"; }

driver_alive() { pgrep -f "[r]un_sf.sh $profile --_inner .*$ws" > /dev/null; }

# Exit code of the latest attempt: the "pipeline exit code N" line after the
# last "running pipeline" line in driver.log (empty if that attempt has none).
last_exit_code() {
  awk '/running pipeline/ {code = ""} /pipeline exit code/ {code = $NF} END {print code}' "$ws/driver.log" 2>/dev/null
}

run_driver() {  # $1 = --workspace or --resume
  RUN_SF_FAKE="${RUN_SF_FAKE:-0}" bash "$here/run_sf.sh" "$profile" --_inner "$1" "$ws" >> "$ws/driver.log" 2>&1 < /dev/null
}

log "started (pid $$, max restarts $max_restarts, pause ${pause}s)"
case "$start" in
  new) run_driver --workspace ;;
  resume) run_driver --resume ;;
  "") ;;
  *) log "bad --start value: $start"; exit 2 ;;
esac

restarts=0
while true; do
  while driver_alive; do sleep 30; done
  code="$(last_exit_code)"
  if [[ "$code" == "0" ]]; then
    log "pipeline finished successfully"
    exit 0
  fi
  if [[ -f "$ws/STOP" ]]; then
    log "STOP file present; not restarting (last exit ${code:-none})"
    exit 0
  fi
  if (( restarts >= max_restarts )); then
    log "giving up after $restarts restarts (last exit ${code:-none})"
    exit 1
  fi
  restarts=$((restarts + 1))
  log "pipeline ended with exit ${code:-unknown}; resume $restarts/$max_restarts in ${pause}s"
  sleep "$pause"
  if [[ -f "$ws/STOP" ]]; then
    log "STOP file present; not restarting"
    exit 0
  fi
  log "resuming"
  run_driver --resume
done
