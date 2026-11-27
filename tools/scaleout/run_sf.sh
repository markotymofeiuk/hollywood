#!/usr/bin/env bash
# One-command generation run on the Albert API (DeepSeek V4 Flash).
#
#   tools/scaleout/run_sf.sh <profile> [--detach] [--workspace DIR] [--resume DIR]
#
#   <profile>        name of tools/scaleout/profiles/<profile>.env, e.g. sf0_1
#   --detach         keep running after you log out (nohup), under supervise.sh,
#                    which resumes the run automatically if it stops with an
#                    error (up to 8 times, 5 min apart; `touch <ws>/STOP` to
#                    prevent that). Setup and the API preflight still run in
#                    the foreground so errors show up
#   --workspace DIR  where to generate (default: $HOME/hollywood-runs/<profile>-<timestamp>)
#   --resume DIR     continue an interrupted run in DIR: finished steps are skipped,
#                    LLM steps pick up their shards, step 100 resumes at the last year
#
# Needs bash, git, curl, uv (https://docs.astral.sh/uv/) and a repo-root .env with
# LOCAL_LLM_URL, LOCAL_LLM_MODEL and LOCAL_LLM_API_KEY or LOCAL_LLM_API_KEYS=k1,k2,k3
# (see .env.example). Python and all packages are installed by this script.
set -euo pipefail

repo="$(cd "$(dirname "$0")/../.." && pwd)"
here="$repo/tools/scaleout"
self="$here/$(basename "$0")"

profile="${1:-}"
if [[ -z "$profile" || "$profile" == -* ]]; then sed -n '2,16p' "$self"; exit 2; fi
shift
detach=0; ws=""; resume=0; inner=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --detach) detach=1 ;;
    --workspace) ws="${2:?--workspace needs a directory}"; shift ;;
    --resume) ws="${2:?--resume needs a directory}"; resume=1; shift ;;
    --_inner) inner=1 ;;
    *) echo "unknown option: $1" >&2; exit 2 ;;
  esac
  shift
done

profile_file="$here/profiles/$profile.env"
[[ -f "$profile_file" ]] || { echo "no profile $profile_file" >&2; exit 2; }
[[ -n "$ws" ]] || ws="$HOME/hollywood-runs/$profile-$(date +%Y%m%d-%H%M%S)"
mkdir -p "$(dirname "$ws")"
ws="$(cd "$(dirname "$ws")" && pwd)/$(basename "$ws")"
if [[ $resume -eq 1 && ! -f "$ws/run_pipeline.py" ]]; then echo "no workspace to resume at $ws" >&2; exit 1; fi
if [[ $resume -eq 0 && -f "$ws/run_pipeline.py" ]]; then echo "$ws already holds a run; use --resume $ws" >&2; exit 1; fi

log() { echo "[$(date '+%F %T')] $*"; }

# --- 1. Python environment via uv ------------------------------------------------
if ! command -v uv >/dev/null 2>&1; then
  echo "uv is not installed. Install it with:  curl -LsSf https://astral.sh/uv/install.sh | sh" >&2
  exit 1
fi
if [[ ! -x "$repo/.venv/bin/python" ]]; then
  log "creating $repo/.venv (Python 3.12) with uv"
  uv venv "$repo/.venv" --python 3.12 -q
fi
uv pip install -q --python "$repo/.venv/bin/python" -r "$repo/requirements.txt" -c "$repo/constraints.txt"
py="$repo/.venv/bin/python"

# --- 2. secrets and Albert settings ---------------------------------------------
if [[ ! -f "$repo/.env" ]]; then
  echo "missing $repo/.env. Create it with these lines (see .env.example):" >&2
  echo "  LOCAL_LLM_URL=https://albert.api.etalab.gouv.fr/v1" >&2
  echo "  LOCAL_LLM_MODEL=deepseek-v4-flash-0731" >&2
  echo "  LOCAL_LLM_API_KEYS=key1,key2,key3      (or LOCAL_LLM_API_KEY=key)" >&2
  exit 1
fi
chmod 600 "$repo/.env" 2>/dev/null || true
set -a
# shellcheck disable=SC1091
source "$repo/.env"
# shellcheck disable=SC1091
source "$here/albert.env"
set +a
# Test hook: RUN_SF_FAKE=1 runs the same orchestration on the offline fake LLM.
if [[ "${RUN_SF_FAKE:-0}" == "1" ]]; then
  export LLM_PROVIDER=fake DATA_SYS_LLM_RPM=0 DATA_SYS_LLM_TPM=0
  log "RUN_SF_FAKE=1: using the offline fake LLM"
fi

# --- 3. preflight: model available, every key answers ---------------------------
if [[ $inner -eq 0 && "${RUN_SF_FAKE:-0}" != "1" ]]; then
  log "preflight"
  "$py" "$here/preflight.py"
fi

# --- 4. detach: re-run this script under nohup and return ------------------------
if [[ $detach -eq 1 && $inner -eq 0 ]]; then
  mkdir -p "$ws"
  if [[ $resume -eq 1 ]]; then first=resume; else first=new; fi
  rm -f "$ws/STOP"
  nohup env RUN_SF_FAKE="${RUN_SF_FAKE:-0}" bash "$here/supervise.sh" "$profile" "$ws" --start "$first" \
    ${SUPERVISE_ARGS:-} > /dev/null 2>&1 < /dev/null &
  echo "started in the background under supervise.sh (pid $!)"
  echo "  workspace: $ws"
  echo "  follow:    tail -f $ws/driver.log $ws/run.log $ws/supervisor.log"
  echo "  stop:      touch $ws/STOP, then stop the processes (see README)"
  echo "  resume:    $self $profile --detach --resume $ws"
  exit 0
fi

# --- 5. workspace + manifest -----------------------------------------------------
if [[ $resume -eq 1 ]]; then
  log "resuming in $ws"
else
  "$here/make_workspace.sh" "$ws"
fi
REPO_DIR="$repo" "$py" - "$ws" "$profile" "$resume" <<'PY'
import json, os, platform, subprocess, sys, time
ws, profile, resume = sys.argv[1], sys.argv[2], sys.argv[3] == "1"
def git(*args):
    out = subprocess.run(["git", *args], capture_output=True, text=True, check=False, cwd=os.environ["REPO_DIR"])
    return out.stdout.strip()
keys = [k for k in os.environ.get("LOCAL_LLM_API_KEYS", "").split(",") if k.strip()]
entry = {
    "started": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    "resume": resume,
    "profile": profile,
    "git_commit": git("rev-parse", "HEAD"),
    "git_branch": git("rev-parse", "--abbrev-ref", "HEAD"),
    "git_dirty_files": len([l for l in git("status", "--porcelain").splitlines() if l.strip()]),
    "host": platform.node(),
    "os": platform.platform(),
    "cpus": os.cpu_count(),
    "model": os.environ.get("LOCAL_LLM_MODEL"),
    "endpoint": os.environ.get("LOCAL_LLM_URL"),
    "api_keys": len(keys) or (1 if os.environ.get("LOCAL_LLM_API_KEY") else 0),
    "settings": {k: os.environ[k] for k in sorted(os.environ)
                 if k.startswith(("DATA_SYS_LLM", "DATA_SYS_CRITIC", "DATA_SYS_OPENAI"))},
}
with open(os.path.join(ws, "run_manifest.jsonl"), "a", encoding="utf-8") as handle:
    handle.write(json.dumps(entry) + "\n")
print(f"[manifest] {entry['git_branch']}@{entry['git_commit'][:10]}, {entry['api_keys']} key(s), {entry['cpus']} CPUs")
PY

# --- 6. pipeline -----------------------------------------------------------------
extra=()
if [[ $resume -eq 1 && -f "$ws/_step100_resume/manifest.json" ]]; then
  s100=$("$py" -c 'import json,sys; print(json.load(open(sys.argv[1])).get("status",""))' "$ws/_step100_resume/manifest.json")
  if [[ "$s100" != "complete" ]]; then extra+=(--resume-step100); fi
fi
log "running pipeline ($profile); log: $ws/run.log"
set +e
(
  cd "$ws"
  export PYTHONUNBUFFERED=1
  export DATA_SYS_NEAR_DUPLICATE_LIMIT="${DATA_SYS_NEAR_DUPLICATE_LIMIT:-0}"
  "$py" "$here/monitor.py" --out "$ws/monitor" --interval 5 -- \
    "$py" run_pipeline.py --profile-file "$profile_file" ${extra[@]+"${extra[@]}"} >> "$ws/run.log" 2>&1
)
rc=$?
set -e
log "pipeline exit code $rc"

# --- 7. checks -------------------------------------------------------------------
"$py" "$here/post_run.py" "$ws" > /dev/null || true
log "summary: $ws/post/summary.md"
if [[ $rc -ne 0 ]]; then
  log "the pipeline stopped early; after fixing the cause, continue with:"
  log "  $self $profile --detach --resume $ws"
fi
exit $rc
