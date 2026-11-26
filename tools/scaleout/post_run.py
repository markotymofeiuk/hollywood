#!/usr/bin/env python3
"""Checks after a finished run: export integrity, the 213 queries on DuckDB,
and a short Markdown summary.

usage: post_run.py <workspace>
Writes <workspace>/post/{integrity/, duckdb/, summary.md}.
"""
from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
PY = sys.executable


def _run(args: list[str], log: Path) -> int:
    with log.open("w", encoding="utf-8") as handle:
        return subprocess.run(args, cwd=REPO, stdout=handle, stderr=subprocess.STDOUT, check=False).returncode


def _empty(note: str) -> bool:
    text = str(note or "").strip()
    return text.startswith("(") and all(part.strip() in {"None", ""} for part in text.strip("()").split(","))


def main() -> int:
    ws = Path(sys.argv[1]).resolve()
    db = ws / "imdb_schema" / "imdb.duckdb"
    post = ws / "post"
    post.mkdir(parents=True, exist_ok=True)
    lines = [f"# Run summary: `{ws.name}`", ""]
    if not db.exists():
        lines.append(f"No DuckDB export at `{db}`; the pipeline did not finish.")
        (post / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
        print("\n".join(lines))
        return 1

    _run([PY, "tools/check_export_integrity.py", "--db", str(db), "--out-dir", str(post / "integrity")], post / "integrity.log")
    integrity = json.loads((post / "integrity" / "export_integrity_summary.json").read_text(encoding="utf-8"))
    _run([PY, "scripts/run_duckdb_workload.py", "--db", str(db), "--out-dir", str(post / "duckdb")], post / "duckdb.log")

    manifest = ws / "imdb_schema" / "export_manifest.json"
    if manifest.exists():
        core = json.loads(manifest.read_text(encoding="utf-8")).get("core_tables", {})
        lines += ["## Export", "", "| Table | Rows |", "|---|---:|"]
        lines += [f"| {k} | {int(v):,} |" for k, v in sorted(core.items())]
        size = sum(p.stat().st_size for p in (ws / "imdb_schema").glob("*.csv"))
        lines += ["", f"Strict CSV size: {size / 1e9:.3f} GB", ""]

    lines += [
        "## Integrity",
        "",
        f"- duplicate primary keys: {integrity.get('duplicate_primary_key_violations')}",
        f"- null primary keys: {integrity.get('null_primary_key_violations')}",
        f"- foreign-key violations: {integrity.get('checked_fk_violations')}",
        f"- required-field failures: {integrity.get('checked_required_field_failures')}",
        "",
        "## Queries on DuckDB (release literals, before literal mining)",
        "",
        "| Workload | Queries | OK | Empty result | Errors |",
        "|---|---:|---:|---:|---:|",
    ]
    for name in ("job_light", "job", "job_complex"):
        path = post / "duckdb" / f"{name}_duckdb_results.csv"
        if not path.exists():
            lines.append(f"| {name} | – | – | – | missing |")
            continue
        rows = list(csv.DictReader(path.open(encoding="utf-8")))
        ok = [r for r in rows if r.get("status") == "OK"]
        empty = [r for r in ok if _empty(r.get("note", ""))]
        lines.append(f"| {name} | {len(rows)} | {len(ok)} | {len(empty)} | {len(rows) - len(ok)} |")

    summary = subprocess.run(
        [PY, str(REPO / "tools" / "scaleout" / "summarize_run.py"), str(ws)],
        capture_output=True, text=True, check=False,
    ).stdout
    lines += ["", "## Steps and LLM calls", "", summary]
    (post / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    sys.exit(main())
