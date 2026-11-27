# Scale-out runs

Tools for generating Hollywood at larger scale factors with concurrent LLM
calls on the Albert API (DeepSeek V4 Flash).

## Start the 0.1 GB run (SF0.1) on a Linux machine

```bash
# 1. Get the code. The release data (Git LFS, ~1 GB) is not needed to generate.
GIT_LFS_SKIP_SMUDGE=1 git clone -b parallel-scaleout https://github.com/markotymofeiuk/hollywood.git
cd hollywood

# 2. Install uv once if missing.
curl -LsSf https://astral.sh/uv/install.sh | sh

# 3. Put the endpoint and key(s) in .env (gitignored).
cat > .env <<'EOF'
LOCAL_LLM_URL=https://albert.api.etalab.gouv.fr/v1
LOCAL_LLM_MODEL=deepseek-v4-flash-0731
LOCAL_LLM_API_KEYS=sk-key1,sk-key2,sk-key3
EOF

# 4. Start. Setup and a one-call-per-key check run in the foreground, then the
#    run detaches so you can log out.
tools/scaleout/run_sf0_1.sh --detach
```

The script prints the workspace (default `~/hollywood-runs/sf0_1-<timestamp>`)
and the commands to follow and resume it.

- Follow: `tail -f <workspace>/driver.log <workspace>/run.log`
- The run is detached from your session and runs under `supervise.sh`: if the
  pipeline stops with an error (network outage, API errors outlasting a step's
  retries, a killed process) it is resumed automatically after 5 minutes, up to
  8 times. `supervisor.log` in the workspace lists every restart.
- Resume by hand (after a reboot, or after the supervisor gave up):
  `tools/scaleout/run_sf0_1.sh --detach --resume <workspace>`. Finished steps are
  skipped, LLM steps continue from their saved shards and step 100 continues
  from its last completed year.
- Stop on purpose: `touch <workspace>/STOP` first (otherwise the supervisor
  resumes it), then
  `pkill -f "[r]un_pipeline.py --profile-file"; pkill -f "<workspace>/[a-z_]*[.]py"`.
  Remove `STOP` before resuming (`--resume` does that for you).
- On Grid'5000 deploy nodes, keep the workspace on local disk and note that
  Ubuntu empties `/tmp` at boot unless `/etc/tmpfiles.d/tmp.conf` contains
  `d /tmp 1777 root root -`.
- Another location: `--workspace /data/runs/sf0_1`.

## What it does

`run_sf.sh <profile>` (the `run_sf0_1.sh` wrapper passes `sf0_1`):

1. creates `.venv` with uv (Python 3.12) and installs the pinned requirements;
2. loads `.env` and `albert.env` (per-key limits 50 requests/min and 256K
   tokens/min, 128 concurrent calls, thinking off, model pinned);
3. checks that the model is available and that every key answers;
4. copies `generator/` into the workspace and records commit, host and
   settings in `run_manifest.jsonl`;
5. runs `run_pipeline.py` with `profiles/<profile>.env`, sampling memory per
   step into `monitor.csv`/`monitor.json`;
6. checks the export (keys, foreign keys, required fields), runs the 213
   queries on the generated DuckDB file and writes `post/summary.md`.

Outputs: strict CSVs and `imdb.duckdb` in `<workspace>/imdb_schema/`, LLM
usage per call in `<workspace>/decision_logs/llm_usage.jsonl`.

## Expected time for SF0.1

21,450 movies, about 19K LLM calls. At the rate limits that is roughly 8 h of
LLM time on one key and under 3 h on three, most of it TV summaries; step 100
adds about 40 min of compute plus ~20 min of serial yearly planning calls.
The query results in `post/summary.md` use the release's literals, so some
queries are expected to return no rows until literals are re-mined for the
new data.

## Other tools

- `run_m0.sh`, `monitor.py`, `summarize_run.py`, `probe_entities.sh`: offline
  measurements with the fake LLM (`LLM_PROVIDER=fake`, see `generator/fake_llm.py`).
- `RUN_SF_FAKE=1 tools/scaleout/run_sf.sh m0_smoke`: dry-run the whole
  orchestration offline.
- Profiles: `profiles/*.env` (`sf0_1`, `real_tiny`, M0 sizes).
