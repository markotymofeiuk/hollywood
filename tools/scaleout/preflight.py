#!/usr/bin/env python3
"""Check the LLM endpoint before a long run: model listed, every key answers.

Reads LOCAL_LLM_URL, LOCAL_LLM_MODEL and LOCAL_LLM_API_KEY / LOCAL_LLM_API_KEYS
plus DATA_SYS_LLM_EXTRA_BODY from the environment (run_sf.sh exports them).
Makes one tiny chat call per key. Exit code 0 only if every key works.
"""
from __future__ import annotations

import json
import os
import sys
import time

import requests


def main() -> int:
    url = os.environ.get("LOCAL_LLM_URL", "").rstrip("/")
    model = os.environ.get("LOCAL_LLM_MODEL", "")
    keys = [k.strip() for k in os.environ.get("LOCAL_LLM_API_KEYS", "").split(",") if k.strip()]
    if not keys and os.environ.get("LOCAL_LLM_API_KEY"):
        keys = [os.environ["LOCAL_LLM_API_KEY"].strip()]
    if not url or not model or not keys:
        print("preflight: set LOCAL_LLM_URL, LOCAL_LLM_MODEL and LOCAL_LLM_API_KEY(S) in .env")
        return 2
    extra = json.loads(os.environ.get("DATA_SYS_LLM_EXTRA_BODY", "") or "{}")
    ok = True
    for i, key in enumerate(keys, start=1):
        headers = {"Authorization": f"Bearer {key}"}
        label = f"key {i}/{len(keys)} (...{key[-4:]})"
        try:
            models = requests.get(f"{url}/models", headers=headers, timeout=30)
            models.raise_for_status()
            ids = {m.get("id") for m in models.json().get("data", [])}
            aliases = {a for m in models.json().get("data", []) for a in (m.get("aliases") or [])}
            if model not in ids | aliases:
                print(f"preflight: {label}: model {model!r} not available to this key")
                ok = False
                continue
            body = {
                "model": model,
                "messages": [{"role": "user", "content": 'Reply with the JSON array [1, 2, 3] and nothing else.'}],
                "max_tokens": 20,
                "temperature": 0,
                **extra,
            }
            t0 = time.time()
            resp = requests.post(f"{url}/chat/completions", json=body, headers=headers, timeout=60)
            resp.raise_for_status()
            data = resp.json()
            text = (data.get("choices") or [{}])[0].get("message", {}).get("content", "")
            usage = data.get("usage", {})
            print(
                f"preflight: {label}: OK in {time.time() - t0:.1f}s, "
                f"{usage.get('completion_tokens', '?')} completion tokens, reply {text.strip()[:40]!r}"
            )
        except Exception as exc:  # report and continue with the other keys
            print(f"preflight: {label}: FAILED: {exc}")
            ok = False
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
