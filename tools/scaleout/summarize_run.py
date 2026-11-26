#!/usr/bin/env python3
"""Summarise one generator workspace run: step times and LLM calls.

Reads <workspace>/pipeline_timing.jsonl and the latest LLM usage log in
<workspace>/decision_logs/, and prints a Markdown report (or JSON with
--json). Token counts from the fake provider are chars/4 estimates.

usage: uv run --no-project python tools/scaleout/summarize_run.py <workspace> [--json]
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


def _jsonl(path: Path) -> list[dict]:
    rows = []
    if not path.exists():
        return rows
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return rows


def summarize(workspace: Path) -> dict:
    timing = _jsonl(workspace / "pipeline_timing.jsonl")
    steps: dict[int, dict] = {}
    for row in timing:
        sid = int(row["step_id"])
        steps[sid] = {
            "step_id": sid,
            "step_name": row.get("step_name", ""),
            "ok": bool(row.get("ok")),
            "elapsed_sec": float(row.get("elapsed_sec", 0.0)),
        }

    usage_path = workspace / "decision_logs" / "llm_usage.jsonl"
    usage = _jsonl(usage_path)
    per_step: dict[int, dict] = defaultdict(lambda: defaultdict(float))
    per_kind: dict[tuple[int, str], dict] = defaultdict(lambda: defaultdict(float))
    for row in usage:
        try:
            sid = int(row.get("step_id") or 0)
        except ValueError:
            sid = 0
        kind = str(row.get("prompt_kind") or "")
        ok = row.get("event") == "llm_call"
        for bucket in (per_step[sid], per_kind[(sid, kind)]):
            bucket["calls"] += 1 if ok else 0
            bucket["errors"] += 0 if ok else 1
            bucket["prompt_chars"] += float(row.get("prompt_chars", 0) or 0)
            bucket["input_tokens"] += float(row.get("input_tokens", 0) or 0)
            bucket["output_tokens"] += float(row.get("output_tokens", 0) or 0)
            bucket["latency_sec"] += float(row.get("latency_sec", 0) or 0)

    for sid, agg in per_step.items():
        steps.setdefault(sid, {"step_id": sid, "step_name": "", "ok": None, "elapsed_sec": 0.0})
        steps[sid]["llm"] = {k: round(v, 3) for k, v in agg.items()}
    kinds = [
        {"step_id": sid, "prompt_kind": kind, **{k: round(v, 3) for k, v in agg.items()}}
        for (sid, kind), agg in sorted(per_kind.items())
    ]
    return {
        "workspace": str(workspace),
        "steps": [steps[k] for k in sorted(steps)],
        "prompt_kinds": kinds,
        "total_elapsed_sec": round(sum(s["elapsed_sec"] for s in steps.values()), 1),
        "total_llm_calls": int(sum(a["calls"] for a in per_step.values())),
    }


def to_markdown(report: dict) -> str:
    out = [f"Workspace: `{report['workspace']}`", ""]
    out.append("| Step | Name | OK | Seconds | LLM calls | Prompt chars | Est. in tok | Est. out tok |")
    out.append("|---:|---|:-:|---:|---:|---:|---:|---:|")
    for s in report["steps"]:
        llm = s.get("llm", {})
        out.append(
            f"| {s['step_id']} | {s['step_name']} | {'yes' if s['ok'] else ('' if s['ok'] is None else 'NO')} "
            f"| {s['elapsed_sec']:.1f} | {int(llm.get('calls', 0)):,} | {int(llm.get('prompt_chars', 0)):,} "
            f"| {int(llm.get('input_tokens', 0)):,} | {int(llm.get('output_tokens', 0)):,} |"
        )
    out.append("")
    out.append(f"Total step time: {report['total_elapsed_sec']:,} s; LLM calls: {report['total_llm_calls']:,}")
    out.append("")
    out.append("| Step | Prompt kind | Calls | Errors | Avg prompt chars | Avg est. out tok |")
    out.append("|---:|---|---:|---:|---:|---:|")
    for k in report["prompt_kinds"]:
        calls = max(1.0, k.get("calls", 0.0) + k.get("errors", 0.0))
        out.append(
            f"| {k['step_id']} | {k['prompt_kind'] or '(unclassified)'} | {int(k.get('calls', 0)):,} | {int(k.get('errors', 0))} "
            f"| {k.get('prompt_chars', 0) / calls:,.0f} | {k.get('output_tokens', 0) / calls:,.0f} |"
        )
    return "\n".join(out)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("workspace", type=Path)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    report = summarize(args.workspace.resolve())
    print(json.dumps(report, indent=2) if args.json else to_markdown(report))


if __name__ == "__main__":
    main()
