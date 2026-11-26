#!/usr/bin/env python3
"""Run a command and sample the memory of its whole process tree.

The pipeline runs each step as a child process, so the runner's own peak
RSS says nothing about the steps. This samples RSS of the full tree every
--interval seconds, attributes each sample to the step script that is
running, and writes:

  <out>.csv   timeline: t_sec, script, tree_rss_mb, step_rss_mb, cpu_percent
  <out>.json  per script: first/last seen, wall seconds, peak RSS

usage: monitor.py --out <prefix> [--interval 1.0] -- <command> [args...]
Needs psutil (present in the generator environment).
"""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from pathlib import Path

import psutil


_CACHE: dict[int, psutil.Process] = {}


def _tree(proc: psutil.Process) -> list[psutil.Process]:
    """Root plus descendants, reusing Process objects so cpu_percent works."""
    try:
        found = [proc] + proc.children(recursive=True)
    except psutil.Error:
        found = [proc]
    out = []
    for p in found:
        out.append(_CACHE.setdefault(p.pid, p))
    return out


def _script_of(p: psutil.Process) -> str:
    try:
        cmd = p.cmdline()
    except psutil.Error:
        return ""
    for arg in cmd[1:]:
        if arg.endswith(".py"):
            return Path(arg).name
    return ""


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--interval", type=float, default=1.0)
    parser.add_argument("cmd", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    cmd = args.cmd[1:] if args.cmd and args.cmd[0] == "--" else args.cmd
    if not cmd:
        parser.error("missing command")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    started = time.time()
    child = subprocess.Popen(cmd)
    root = psutil.Process(child.pid)
    per_script: dict[str, dict] = {}
    with open(f"{args.out}.csv", "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["t_sec", "script", "tree_rss_mb", "step_rss_mb", "cpu_percent"])
        while child.poll() is None:
            procs = _tree(root)
            tree_rss = 0
            step_rss = 0
            step_script = ""
            cpu = 0.0
            for p in procs:
                try:
                    rss = p.memory_info().rss
                    cpu += p.cpu_percent(interval=None)
                except psutil.Error:
                    continue
                tree_rss += rss
                name = _script_of(p)
                if name and name != "run_pipeline.py" and rss >= step_rss:
                    step_rss = rss
                    step_script = name
            t = time.time() - started
            writer.writerow([round(t, 2), step_script, round(tree_rss / 2**20, 1), round(step_rss / 2**20, 1), round(cpu, 1)])
            if step_script:
                info = per_script.setdefault(
                    step_script, {"first_sec": t, "last_sec": t, "peak_step_rss_mb": 0.0, "peak_tree_rss_mb": 0.0}
                )
                info["last_sec"] = t
                info["peak_step_rss_mb"] = max(info["peak_step_rss_mb"], step_rss / 2**20)
                info["peak_tree_rss_mb"] = max(info["peak_tree_rss_mb"], tree_rss / 2**20)
            handle.flush()
            time.sleep(args.interval)
    for info in per_script.values():
        info["wall_sec"] = round(info["last_sec"] - info["first_sec"], 1)
        for key in ("first_sec", "last_sec", "peak_step_rss_mb", "peak_tree_rss_mb"):
            info[key] = round(info[key], 1)
    summary = {
        "cmd": cmd,
        "exit_code": child.returncode,
        "wall_sec": round(time.time() - started, 1),
        "per_script": per_script,
    }
    Path(f"{args.out}.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return int(child.returncode or 0)


if __name__ == "__main__":
    sys.exit(main())
