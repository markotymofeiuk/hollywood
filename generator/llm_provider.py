"""
Unified LLM Provider Layer (v20)
=================================
Single entry point for ALL LLM calls across the pipeline.

Supports two backends:
  1. **GeminiProvider** — Google Gemini API (production default)
  2. **OpenAICompatibleProvider** — HTTP client for vLLM, Ollama, TGI,
     LiteLLM, OpenAI, or another OpenAI-compatible server

Configuration via environment / .env::

    # --- Gemini (default) ---
    LLM_PROVIDER=gemini
    GOOGLE_API_KEY=<your-google-api-key>

    # --- OpenAI-compatible HTTP backend (started separately) ---
    LLM_PROVIDER=local
    LOCAL_LLM_URL=http://localhost:8000/v1
    LOCAL_LLM_MODEL=<model-served-by-your-backend>

Usage in any script::

    from llm_provider import get_llm_client
    client = get_llm_client()
    response = client.generate("Your prompt here", json_mode=True, temperature=0.5)
    print(response.text)
"""
from __future__ import annotations

import concurrent.futures
import json
import os
import random
import ssl
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, ClassVar, Dict, List, Optional

from dotenv import load_dotenv
from model_defaults import DEFAULT_GEMINI_MODEL, model_for_role

# Load .env from project root (one level up from script dir)
_script_dir = Path(__file__).resolve().parent
_env_path = _script_dir.parent / ".env"
if _env_path.exists():
    load_dotenv(_env_path)
# Also try local .env
_local_env = _script_dir / ".env"
if _local_env.exists():
    load_dotenv(_local_env, override=True)


# ═══════════════════════════════════════════════════════════════════════
# Response / Stats dataclasses
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class LLMResponse:
    """Unified response object returned by all providers."""
    text: str
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    model: str = ""
    cached: bool = False
    raw: Any = None  # original provider response for advanced usage


@dataclass
class RetryStats:
    """Statistics from a single generate call (for logging/debugging)."""
    attempts: int = 0
    retries: int = 0
    total_sleep_sec: float = 0.0
    errors: List[str] = field(default_factory=list)


# ═══════════════════════════════════════════════════════════════════════
# Retry / Resilience (provider-agnostic)
# ═══════════════════════════════════════════════════════════════════════

_RETRIABLE_TOKENS = (
    "503", "429", "UNAVAILABLE", "RESOURCE_EXHAUSTED",
    "DEADLINEEXCEEDED", "DEADLINE_EXCEEDED", "TIMEOUT",
    "TIMED OUT", "SERVICE UNAVAILABLE", "rate limit",
    "too many requests", "overloaded",
)


def _env_int(name: str, default: int, *, minimum: int | None = None) -> int:
    raw = os.getenv(name)
    if raw is None or str(raw).strip() == "":
        value = int(default)
    else:
        try:
            value = int(float(str(raw).strip()))
        except Exception:
            value = int(default)
    if minimum is not None:
        value = max(int(minimum), value)
    return value


def _env_float(name: str, default: float, *, minimum: float | None = None) -> float:
    raw = os.getenv(name)
    if raw is None or str(raw).strip() == "":
        value = float(default)
    else:
        try:
            value = float(str(raw).strip())
        except Exception:
            value = float(default)
    if minimum is not None:
        value = max(float(minimum), value)
    return value


def _env_flag(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return bool(default)
    return str(raw).strip().lower() in {"1", "true", "yes", "on"}


def _build_gemini_http_options() -> Any | None:
    """Build HTTP options for Gemini calls.

    On Windows we prefer the OS trust store over certifi/httpx defaults so
    research runs do not randomly fail behind enterprise TLS interception.
    """
    from google.genai import types  # type: ignore

    client_args: Dict[str, Any] = {"trust_env": True}

    disable_verify = _env_flag("DATA_SYS_GEMINI_DISABLE_SSL_VERIFY", False)
    custom_ca = (
        os.getenv("DATA_SYS_GEMINI_SSL_CERT_FILE")
        or os.getenv("SSL_CERT_FILE")
        or os.getenv("REQUESTS_CA_BUNDLE")
    )

    if disable_verify:
        client_args["verify"] = False
    elif custom_ca:
        client_args["verify"] = str(custom_ca)
    elif os.name == "nt":
        client_args["verify"] = ssl.create_default_context()

    if not client_args:
        return None
    return types.HttpOptions(client_args=client_args)


_USAGE_LOG_LOCK = threading.Lock()


def _write_usage_log(entry: Dict[str, Any]) -> None:
    targets = []
    primary = os.getenv("DATA_SYS_LLM_USAGE_LOG")
    latest = os.getenv("DATA_SYS_LLM_USAGE_LOG_LATEST")
    for raw in (primary, latest):
        if raw and str(raw).strip():
            targets.append(str(raw).strip())
    unique_targets: list[str] = []
    seen: set[str] = set()
    for raw in targets:
        key = str(Path(raw))
        if key in seen:
            continue
        seen.add(key)
        unique_targets.append(raw)
    if not unique_targets:
        return
    line = json.dumps(entry, ensure_ascii=True, default=str) + "\n"
    with _USAGE_LOG_LOCK:
        for raw in unique_targets:
            try:
                target = Path(raw)
                target.parent.mkdir(parents=True, exist_ok=True)
                with target.open("a", encoding="utf-8") as handle:
                    handle.write(line)
            except Exception:
                pass


def _prompt_kind(prompt: str) -> str:
    """Name of the prompt archetype (shared with the fake provider), or ''."""
    try:
        import fake_llm

        return fake_llm.classify(prompt) or ""
    except Exception:
        return ""


def _log_llm_success(
    *,
    response: "LLMResponse",
    prompt: str,
    json_mode: bool,
    temperature: float,
    max_tokens: int,
    stats: "RetryStats",
    latency_sec: float = 0.0,
) -> None:
    _write_usage_log(
        {
            "event": "llm_call",
            "prompt_kind": _prompt_kind(prompt),
            "latency_sec": round(float(latency_sec), 4),
            "step_id": os.getenv("DATA_SYS_STEP_ID", ""),
            "step_name": os.getenv("DATA_SYS_STEP_NAME", ""),
            "provider": os.getenv("LLM_PROVIDER", "gemini"),
            "model": response.model,
            "json_mode": bool(json_mode),
            "temperature": float(temperature),
            "max_tokens": int(max_tokens),
            "prompt_chars": len(prompt),
            "input_tokens": int(response.input_tokens or 0),
            "output_tokens": int(response.output_tokens or 0),
            "total_tokens": int(response.input_tokens or 0) + int(response.output_tokens or 0),
            "cached": bool(response.cached),
            "attempts": int(stats.attempts),
            "retries": int(stats.retries),
            "sleep_sec": round(float(stats.total_sleep_sec), 4),
            "timeout_sec": float(os.getenv("DATA_SYS_LLM_TIMEOUT_SEC", "0") or 0),
            "retry_budget": int(os.getenv("DATA_SYS_LLM_MAX_ATTEMPTS", "0") or 0),
            "cost_usd": float(response.cost_usd or 0.0),
            "timestamp": time.time(),
        }
    )


def _log_llm_failure(
    *,
    model: str,
    prompt: str,
    json_mode: bool,
    temperature: float,
    max_tokens: int,
    stats: "RetryStats",
    error: BaseException | None,
) -> None:
    _write_usage_log(
        {
            "event": "llm_error",
            "prompt_kind": _prompt_kind(prompt),
            "step_id": os.getenv("DATA_SYS_STEP_ID", ""),
            "step_name": os.getenv("DATA_SYS_STEP_NAME", ""),
            "provider": os.getenv("LLM_PROVIDER", "gemini"),
            "model": str(model or ""),
            "json_mode": bool(json_mode),
            "temperature": float(temperature),
            "max_tokens": int(max_tokens),
            "prompt_chars": len(prompt),
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
            "attempts": int(stats.attempts),
            "retries": int(stats.retries),
            "sleep_sec": round(float(stats.total_sleep_sec), 4),
            "timeout_sec": float(os.getenv("DATA_SYS_LLM_TIMEOUT_SEC", "0") or 0),
            "retry_budget": int(os.getenv("DATA_SYS_LLM_MAX_ATTEMPTS", "0") or 0),
            "error": str(error) if error is not None else "",
            "timestamp": time.time(),
        }
    )


def _is_retriable(exc: BaseException) -> bool:
    msg = str(exc).upper()
    return any(tok in msg for tok in _RETRIABLE_TOKENS)


def _call_with_timeout(fn: Callable[[], Any], timeout_sec: float) -> Any:
    result: dict[str, Any] = {"value": None, "error": None}

    def _runner() -> None:
        try:
            result["value"] = fn()
        except BaseException as exc:  # pragma: no cover - passthrough
            result["error"] = exc

    thread = threading.Thread(target=_runner, daemon=True)
    thread.start()
    thread.join(timeout=timeout_sec)

    if thread.is_alive():
        raise TimeoutError(f"LLM call exceeded {timeout_sec:.0f}s")
    if result["error"] is not None:
        raise result["error"]
    return result["value"]


# ═══════════════════════════════════════════════════════════════════════
# Rate limiting across API keys (requests/min and tokens/min per key)
# ═══════════════════════════════════════════════════════════════════════

class _Lease:
    """One admitted request on one key; settle() replaces the token estimate."""

    __slots__ = ("key", "entry", "limiter")

    def __init__(self, key: str, entry: list, limiter: "KeyPool") -> None:
        self.key = key
        self.entry = entry  # [timestamp, tokens]
        self.limiter = limiter

    def settle(self, tokens: int | None) -> None:
        if tokens is not None and tokens > 0:
            with self.limiter._cond:
                self.entry[1] = int(tokens)
                self.limiter._cond.notify_all()

    def penalize(self, seconds: float) -> None:
        self.limiter.block_key(self.key, seconds)


class KeyPool:
    """Sliding 60-second window limiter over one or more API keys.

    Each request is admitted on the key that can take it soonest, keeping
    both requests and tokens in the last 60 s under ``safety * limit``. The
    token count of a request is an estimate until settle() records the
    usage the server reports. Limits are per process: run one pipeline per
    key set at a time (or give concurrent runs disjoint keys).
    """

    WINDOW = 60.0
    # Short window that caps bursts at BURST x the average rate, so a cold
    # start does not fire a whole minute's allowance in one second.
    SHORT_WINDOW = 10.0
    BURST = 2.0

    def __init__(self, keys: List[str], *, rpm: float, tpm: float, safety: float = 0.9) -> None:
        if not keys:
            raise ValueError("KeyPool needs at least one key")
        self.keys = list(keys)
        self.rpm = max(1.0, float(rpm) * float(safety))
        self.tpm = max(1.0, float(tpm) * float(safety))
        self._events: Dict[str, Any] = {k: [] for k in self.keys}
        self._blocked_until: Dict[str, float] = {k: 0.0 for k in self.keys}
        self._cond = threading.Condition()

    def _admit_at(self, key: str, est_tokens: int, now: float) -> float:
        events = self._events[key]
        cutoff = now - self.WINDOW
        while events and events[0][0] <= cutoff:
            events.pop(0)
        t = max(now, self._blocked_until[key])
        if len(events) >= self.rpm:
            t = max(t, events[len(events) - int(self.rpm)][0] + self.WINDOW)
        short_cap = max(1, int(self.rpm * self.SHORT_WINDOW / self.WINDOW * self.BURST))
        short_start = now - self.SHORT_WINDOW
        recent = [e for e in events if e[0] > short_start]
        if len(recent) >= short_cap:
            t = max(t, recent[len(recent) - short_cap][0] + self.SHORT_WINDOW)
        short_tok_cap = self.tpm * self.SHORT_WINDOW / self.WINDOW * self.BURST
        recent_tok = sum(e[1] for e in recent)
        if recent and recent_tok + est_tokens > short_tok_cap:
            freed = 0
            for ts, tok in recent:
                freed += tok
                if recent_tok - freed + est_tokens <= short_tok_cap:
                    t = max(t, ts + self.SHORT_WINDOW)
                    break
            else:
                t = max(t, recent[-1][0] + self.SHORT_WINDOW)
        used = sum(e[1] for e in events)
        if events and used + est_tokens > self.tpm:
            freed = 0
            for ts, tok in events:
                freed += tok
                if used - freed + est_tokens <= self.tpm:
                    t = max(t, ts + self.WINDOW)
                    break
            else:
                t = max(t, events[-1][0] + self.WINDOW)
        return t

    def acquire(self, est_tokens: int) -> _Lease:
        est = max(1, int(est_tokens))
        with self._cond:
            while True:
                now = time.time()
                best_key, best_t, best_load = None, float("inf"), float("inf")
                for key in self.keys:
                    t = self._admit_at(key, est, now)
                    events = self._events[key]
                    load = max(len(events) / self.rpm, sum(e[1] for e in events) / self.tpm)
                    # Earliest admission first; among keys free now, the least loaded.
                    if (max(t, now), load) < (max(best_t, now), best_load):
                        best_key, best_t, best_load = key, t, load
                if best_t <= now:
                    entry = [now, est]
                    self._events[best_key].append(entry)
                    return _Lease(best_key, entry, self)
                self._cond.wait(timeout=min(1.0, max(0.01, best_t - now)))

    def block_key(self, key: str, seconds: float) -> None:
        with self._cond:
            self._blocked_until[key] = max(self._blocked_until[key], time.time() + float(seconds))


def _key_pool_from_env(keys: List[str], prefix: str = "DATA_SYS_LLM") -> Optional[KeyPool]:
    rpm = _env_float(f"{prefix}_RPM", 0.0)
    tpm = _env_float(f"{prefix}_TPM", 0.0)
    if rpm <= 0 and tpm <= 0:
        return None
    return KeyPool(
        keys,
        rpm=rpm if rpm > 0 else 1e9,
        tpm=tpm if tpm > 0 else 1e12,
        safety=_env_float(f"{prefix}_LIMIT_SAFETY", 0.9, minimum=0.1),
    )


_OUTPUT_ESTIMATES: Dict[str, list] = {}
_OUTPUT_ESTIMATES_LOCK = threading.Lock()


def _estimate_request_tokens(prompt: str, max_tokens: int) -> int:
    """Prompt tokens (chars/3.2, conservative) + running mean output for this prompt kind."""
    kind = _prompt_kind(prompt)
    with _OUTPUT_ESTIMATES_LOCK:
        total, n = _OUTPUT_ESTIMATES.get(kind, (0, 0))
    expected_out = (total / n) if n >= 3 else min(int(max_tokens or 4096), 4096)
    return int(len(prompt) / 3.2) + int(expected_out)


def _record_output_tokens(prompt: str, output_tokens: int) -> None:
    if output_tokens <= 0:
        return
    kind = _prompt_kind(prompt)
    with _OUTPUT_ESTIMATES_LOCK:
        total, n = _OUTPUT_ESTIMATES.get(kind, (0, 0))
        _OUTPUT_ESTIMATES[kind] = (total + int(output_tokens), n + 1)


def llm_concurrency() -> int:
    """Worker count for concurrent LLM maps (DATA_SYS_LLM_CONCURRENCY, default 1 = serial)."""
    return _env_int("DATA_SYS_LLM_CONCURRENCY", 1, minimum=1)


def ordered_map(
    fn: Callable[[Any], Any],
    items: List[Any],
    *,
    workers: Optional[int] = None,
    prepare: Optional[Callable[[Any], Any]] = None,
):
    """Yield (item, result) in input order while running fn concurrently.

    workers=1 runs fn inline, exactly like a plain loop. With more workers at
    most ``2 * workers`` items are in flight, so memory stays bounded. If fn
    raises, no new items start, in-flight ones finish, and the exception is
    raised when its item's turn comes (earlier results are yielded first).

    ``prepare(item)`` runs on the calling thread when the item is submitted
    and its return value is what fn receives. Item k is submitted after the
    caller has consumed items 0..k-window, so state captured in prepare is
    deterministic: it reflects exactly the first k-window+1 results
    (window = 2 * workers; with workers=1 it reflects all earlier results).
    """
    n_workers = llm_concurrency() if workers is None else max(1, int(workers))
    items = list(items)
    prep = prepare if prepare is not None else (lambda item: item)
    if n_workers <= 1 or len(items) <= 1:
        for item in items:
            yield item, fn(prep(item))
        return
    window = 2 * n_workers
    with concurrent.futures.ThreadPoolExecutor(max_workers=n_workers, thread_name_prefix="llm") as pool:
        futures: Dict[int, Any] = {}
        next_submit = 0
        failed = False
        for idx in range(len(items)):
            while not failed and next_submit < len(items) and next_submit < idx + window:
                futures[next_submit] = pool.submit(fn, prep(items[next_submit]))
                next_submit += 1
            future = futures.pop(idx)
            try:
                result = future.result()
            except BaseException:
                failed = True
                for other in futures.values():
                    other.cancel()
                raise
            yield items[idx], result


# ═══════════════════════════════════════════════════════════════════════
# Abstract Provider
# ═══════════════════════════════════════════════════════════════════════

class LLMProvider(ABC):
    """Abstract base for LLM backends."""

    # Set by providers that enforce timeouts themselves (no watchdog thread).
    native_timeout: bool = False
    # Optional KeyPool; when set, every attempt is admitted by the limiter.
    key_pool: Optional[KeyPool] = None

    @abstractmethod
    def _raw_generate(
        self,
        prompt: str,
        *,
        model: str,
        json_mode: bool,
        temperature: float,
        max_tokens: int,
        thinking_budget: int | None = None,
        api_key: str | None = None,
        timeout_sec: float | None = None,
    ) -> LLMResponse:
        """Execute a single LLM call (no retry). Subclasses implement this."""
        ...

    def generate(
        self,
        prompt: str,
        *,
        model: Optional[str] = None,
        json_mode: bool = False,
        temperature: float = 0.5,
        max_tokens: int = 4096,
        thinking_budget: int | None = None,
        timeout_sec: float = 70.0,
        max_attempts: int = 5,
        base_delay_sec: float = 4.0,
        max_delay_sec: float = 35.0,
        on_retry: Optional[Callable] = None,
    ) -> LLMResponse:
        """Generate with timeout + exponential backoff retry."""
        stats = RetryStats()
        last_err: Optional[BaseException] = None
        rng = random.Random()

        effective_model = self.resolve_model(model)
        effective_timeout_sec = _env_float("DATA_SYS_LLM_TIMEOUT_SEC", timeout_sec, minimum=5.0)
        effective_max_attempts = _env_int("DATA_SYS_LLM_MAX_ATTEMPTS", max_attempts, minimum=1)
        effective_base_delay_sec = _env_float("DATA_SYS_LLM_BASE_DELAY_SEC", base_delay_sec, minimum=0.0)
        effective_max_delay_sec = _env_float(
            "DATA_SYS_LLM_MAX_DELAY_SEC",
            max(max_delay_sec, effective_base_delay_sec),
            minimum=effective_base_delay_sec,
        )

        for attempt in range(1, effective_max_attempts + 1):
            stats.attempts += 1
            lease = None
            if self.key_pool is not None:
                lease = self.key_pool.acquire(_estimate_request_tokens(prompt, max_tokens))
            attempt_started = time.perf_counter()
            try:
                call = lambda: self._raw_generate(  # noqa: E731
                    prompt,
                    model=effective_model,
                    json_mode=json_mode,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    thinking_budget=thinking_budget,
                    api_key=lease.key if lease is not None else None,
                    timeout_sec=effective_timeout_sec,
                )
                if self.native_timeout:
                    resp = call()
                else:
                    resp = _call_with_timeout(call, timeout_sec=effective_timeout_sec)
                resp.model = effective_model
                if lease is not None:
                    lease.settle(int(resp.input_tokens or 0) + int(resp.output_tokens or 0))
                _record_output_tokens(prompt, int(resp.output_tokens or 0))
                _log_llm_success(
                    response=resp,
                    prompt=prompt,
                    json_mode=json_mode,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    stats=stats,
                    latency_sec=time.perf_counter() - attempt_started,
                )
                return resp
            except BaseException as exc:
                last_err = exc
                stats.errors.append(str(exc))
                if lease is not None and ("429" in str(exc) or "rate limit" in str(exc).lower()):
                    lease.penalize(_env_float("DATA_SYS_LLM_429_COOLDOWN_SEC", 20.0, minimum=0.0))

                retriable = isinstance(exc, TimeoutError) or _is_retriable(exc)

                if (not retriable) or attempt >= effective_max_attempts:
                    break

                stats.retries += 1
                delay = min(effective_max_delay_sec, effective_base_delay_sec * (2 ** (attempt - 1)))
                jitter = delay * 0.25 * rng.random()
                sleep_for = delay + jitter
                stats.total_sleep_sec += sleep_for
                if on_retry:
                    on_retry(attempt, effective_max_attempts, exc, sleep_for)
                time.sleep(sleep_for)

        _log_llm_failure(
            model=effective_model,
            prompt=prompt,
            json_mode=json_mode,
            temperature=temperature,
            max_tokens=max_tokens,
            stats=stats,
            error=last_err,
        )
        raise RuntimeError(
            f"LLM call failed after {stats.attempts} attempts: {last_err}"
        ) from last_err

    @property
    @abstractmethod
    def default_model(self) -> str:
        ...

    def resolve_model(self, requested: Optional[str]) -> str:
        """Model actually sent. With DATA_SYS_LLM_PIN_MODEL=1 every call uses
        the provider default, ignoring per-step names (the steps default to
        Gemini model ids via model_defaults.model_for_role)."""
        if _env_flag("DATA_SYS_LLM_PIN_MODEL", False):
            return self.default_model
        return requested or self.default_model


# ═══════════════════════════════════════════════════════════════════════
# Gemini Provider (Google AI Studio / Vertex)
# ═══════════════════════════════════════════════════════════════════════

class GeminiProvider(LLMProvider):
    """Google Gemini API backend."""

    def __init__(self, api_key: str, default_model: str = DEFAULT_GEMINI_MODEL):
        from google import genai  # type: ignore
        http_options = _build_gemini_http_options()
        client_kwargs: Dict[str, Any] = {"api_key": api_key}
        if http_options is not None:
            client_kwargs["http_options"] = http_options
        self._client = genai.Client(**client_kwargs)
        self._default_model = default_model

    @property
    def default_model(self) -> str:
        return self._default_model

    def _raw_generate(
        self,
        prompt: str,
        *,
        model: str,
        json_mode: bool,
        temperature: float,
        max_tokens: int,
        thinking_budget: int | None = None,
        api_key: str | None = None,
        timeout_sec: float | None = None,
    ) -> LLMResponse:
        config: Dict[str, Any] = {
            "temperature": temperature,
        }
        if thinking_budget is not None:
            budget = int(thinking_budget)
            if not (budget <= 0 and "pro" in str(model).lower()):
                config["thinking_config"] = {"thinking_budget": budget}
        if json_mode:
            config["response_mime_type"] = "application/json"
        if max_tokens:
            config["max_output_tokens"] = max_tokens

        resp = self._client.models.generate_content(
            model=model,
            contents=prompt,
            config=config,
        )

        text = str(getattr(resp, "text", "") or "")

        # Extract token usage
        usage = getattr(resp, "usage_metadata", None)
        inp_tokens = int(getattr(usage, "prompt_token_count", 0) or 0) if usage else 0
        out_tokens = int(getattr(usage, "candidates_token_count", 0) or 0) if usage else 0

        return LLMResponse(
            text=text,
            input_tokens=inp_tokens,
            output_tokens=out_tokens,
            model=model,
            raw=resp,
        )


# ═══════════════════════════════════════════════════════════════════════
# OpenAI-Compatible Provider (vLLM, Ollama, TGI, LiteLLM, etc.)
# ═══════════════════════════════════════════════════════════════════════

class OpenAICompatibleProvider(LLMProvider):
    """Any server exposing the OpenAI /v1/chat/completions API.

    Works with:
      - vLLM: ``python -m vllm.entrypoints.openai.api_server --model <model-served-by-the-lab>``
      - Ollama: ``ollama serve`` (default port 11434, endpoint /v1/chat/completions)
      - TGI: ``text-generation-launcher --model-id ...``
      - LiteLLM proxy: ``litellm --model ...``

    Configuration via environment:
      - LOCAL_LLM_URL: base URL (e.g. http://localhost:8000/v1)
      - LOCAL_LLM_MODEL: model name as registered on the server
      - LOCAL_LLM_API_KEY: optional API key (some servers require one)
    """

    native_timeout = True

    def __init__(
        self,
        base_url: str = "http://localhost:8000/v1",
        model: str = "served-model",
        api_key: str = "not-needed",
        api_keys: Optional[List[str]] = None,
    ):
        self._base_url = base_url.rstrip("/")
        self._default_model = model
        self._api_key = api_key
        self._api_keys = [k for k in (api_keys or []) if k] or [api_key]
        self.key_pool = _key_pool_from_env(self._api_keys)
        self._local = threading.local()
        # Optional JSON merged into every chat/completions body, e.g.
        # {"chat_template_kwargs": {"thinking": false}} for reasoning models.
        raw_extra = os.getenv("DATA_SYS_LLM_EXTRA_BODY", "").strip()
        self._extra_body: Dict[str, Any] = json.loads(raw_extra) if raw_extra else {}
        # "object" sends response_format=json_object (forces a top-level
        # object, which breaks prompts that ask for a JSON array); the
        # default "prompt" relies on the prompt and normalises the reply.
        self._json_mode = os.getenv("DATA_SYS_OPENAI_JSON_MODE", "prompt").strip().lower()

        # Lazy import — requests is almost always available
        try:
            import requests  # noqa: F401
        except ImportError:
            raise ImportError(
                "The 'requests' package is required for the OpenAI-compatible provider. "
                "Install with: pip install requests"
            )

    @property
    def default_model(self) -> str:
        return self._default_model

    def _ollama_native_base(self) -> str:
        base = self._base_url
        if base.endswith("/v1"):
            return base[:-3]
        return base

    def _looks_like_ollama(self) -> bool:
        url = self._base_url.lower()
        return "11434" in url or "ollama" in url

    def _use_ollama_native(self) -> bool:
        value = os.getenv("DATA_SYS_OLLAMA_NATIVE", "1").strip().lower()
        return value not in {"0", "false", "no", "off"}

    def _apply_ollama_think_option(self, body: Dict[str, Any]) -> None:
        value = os.getenv("DATA_SYS_OLLAMA_THINK", "0").strip().lower()
        if value in {"1", "true", "yes", "on"}:
            body["think"] = True
        elif value in {"0", "false", "no", "off"}:
            body["think"] = False

    def _parse_ollama_native_response(self, data: Dict[str, Any], *, model: str) -> LLMResponse:
        message = data.get("message", {}) if isinstance(data, dict) else {}
        text = ""
        if isinstance(message, dict):
            text = str(message.get("content") or "")
        if not text and isinstance(data, dict):
            text = str(data.get("response") or "")
        inp_tokens = 0
        out_tokens = 0
        if isinstance(data, dict):
            inp_tokens = int(data.get("prompt_eval_count") or 0)
            out_tokens = int(data.get("eval_count") or 0)
        return LLMResponse(
            text=text,
            input_tokens=inp_tokens,
            output_tokens=out_tokens,
            model=model,
            raw=data,
        )

    def _raw_generate_ollama_native(
        self,
        prompt: str,
        *,
        model: str,
        json_mode: bool,
        temperature: float,
        max_tokens: int,
        thinking_budget: int | None = None,
        **_ignored: Any,
    ) -> LLMResponse:
        import requests

        url = f"{self._ollama_native_base().rstrip('/')}/api/chat"
        body: Dict[str, Any] = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            },
        }
        if json_mode:
            body["format"] = "json"
        self._apply_ollama_think_option(body)

        resp = requests.post(url, json=body, timeout=120)
        if resp.status_code == 404:
            return self._raw_generate_ollama_generate(
                prompt,
                model=model,
                json_mode=json_mode,
                temperature=temperature,
                max_tokens=max_tokens,
            )
        resp.raise_for_status()
        data = resp.json()
        return self._parse_ollama_native_response(data, model=model)

    def _raw_generate_ollama_generate(
        self,
        prompt: str,
        *,
        model: str,
        json_mode: bool,
        temperature: float,
        max_tokens: int,
        thinking_budget: int | None = None,
        **_ignored: Any,
    ) -> LLMResponse:
        import requests

        url = f"{self._ollama_native_base().rstrip('/')}/api/generate"
        body: Dict[str, Any] = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            },
        }
        if json_mode:
            body["format"] = "json"
        self._apply_ollama_think_option(body)

        resp = requests.post(url, json=body, timeout=120)
        resp.raise_for_status()
        data = resp.json()
        return self._parse_ollama_native_response(data, model=model)

    def _session(self):
        import requests

        session = getattr(self._local, "session", None)
        if session is None:
            session = requests.Session()
            self._local.session = session
        return session

    def _raw_generate(
        self,
        prompt: str,
        *,
        model: str,
        json_mode: bool,
        temperature: float,
        max_tokens: int,
        thinking_budget: int | None = None,
        api_key: str | None = None,
        timeout_sec: float | None = None,
    ) -> LLMResponse:
        import requests

        if self._looks_like_ollama() and self._use_ollama_native():
            return self._raw_generate_ollama_native(
                prompt,
                model=model,
                json_mode=json_mode,
                temperature=temperature,
                max_tokens=max_tokens,
                thinking_budget=thinking_budget,
            )

        url = f"{self._base_url}/chat/completions"
        headers = {
            "Content-Type": "application/json",
        }
        key = api_key or self._api_key
        if key and key != "not-needed":
            headers["Authorization"] = f"Bearer {key}"

        body: Dict[str, Any] = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if json_mode and self._json_mode == "object":
            body["response_format"] = {"type": "json_object"}
        body.update(self._extra_body)

        read_timeout = float(timeout_sec) if timeout_sec else 120.0
        resp = self._session().post(url, json=body, headers=headers, timeout=(15.0, read_timeout))
        if resp.status_code == 404 and self._looks_like_ollama():
            return self._raw_generate_ollama_native(
                prompt,
                model=model,
                json_mode=json_mode,
                temperature=temperature,
                max_tokens=max_tokens,
                thinking_budget=thinking_budget,
            )
        resp.raise_for_status()
        data = resp.json()

        # Parse OpenAI-format response
        choices = data.get("choices", [])
        text = choices[0]["message"]["content"] if choices else ""
        if isinstance(text, list):
            parts: list[str] = []
            for item in text:
                if isinstance(item, dict):
                    parts.append(str(item.get("text") or item.get("content") or ""))
                else:
                    parts.append(str(item))
            text = "".join(parts)
        text = str(text or "")
        if json_mode and self._json_mode != "raw":
            # Strip fences/think blocks and re-serialise, so step parsers
            # that call json.loads directly see clean JSON.
            parsed = safe_json_parse(text)
            if parsed is not None:
                text = json.dumps(parsed, ensure_ascii=False)

        usage = data.get("usage", {}) or {}
        inp_tokens = int(usage.get("prompt_tokens", 0) or 0)
        out_tokens = int(usage.get("completion_tokens", 0) or 0)

        return LLMResponse(
            text=text,
            input_tokens=inp_tokens,
            output_tokens=out_tokens,
            model=model,
            raw=data,
        )


# ═══════════════════════════════════════════════════════════════════════
# Fake Provider (offline measurement and determinism tests)
# ═══════════════════════════════════════════════════════════════════════

class FakeProvider(LLMProvider):
    """Deterministic offline backend; see fake_llm.py.

    Responses depend only on the prompt text, so serial and parallel runs
    can be compared byte-for-byte. Token counts are chars/4 estimates.
    """

    def __init__(self, default_model: str = "fake-llm") -> None:
        self._default_model = default_model
        n_keys = _env_int("FAKE_LLM_KEYS", 1, minimum=1)
        self.key_pool = _key_pool_from_env([f"fake-key-{i}" for i in range(n_keys)])

    @property
    def default_model(self) -> str:
        return self._default_model

    def _raw_generate(
        self,
        prompt: str,
        *,
        model: str,
        json_mode: bool,
        temperature: float,
        max_tokens: int,
        thinking_budget: int | None = None,
        api_key: str | None = None,
        timeout_sec: float | None = None,
    ) -> LLMResponse:
        import fake_llm

        fake_llm.fake_latency()
        _handler, text = fake_llm.fake_response_text(prompt)
        return LLMResponse(
            text=text,
            input_tokens=fake_llm.estimate_tokens(prompt),
            output_tokens=fake_llm.estimate_tokens(text),
            model=model,
        )


# ═══════════════════════════════════════════════════════════════════════
# Factory / Singleton
# ═══════════════════════════════════════════════════════════════════════

_global_client: Optional[LLMProvider] = None


def get_llm_client(
    *,
    provider: Optional[str] = None,
    force_new: bool = False,
) -> LLMProvider:
    """Get or create the global LLM client.

    Provider is determined by (in priority order):
      1. ``provider`` argument
      2. ``LLM_PROVIDER`` env var
      3. Default: "gemini"

    Returns a configured LLMProvider with retry logic built in.
    """
    global _global_client

    if _global_client is not None and not force_new:
        return _global_client

    provider_name = (provider or os.getenv("LLM_PROVIDER", "gemini")).lower().strip()

    if provider_name in ("gemini", "google"):
        api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError(
                "LLM_PROVIDER=gemini but no API key found.\n"
                "Set GOOGLE_API_KEY in your .env file or environment."
            )
        default_model = model_for_role("entity_gen")

        _global_client = GeminiProvider(api_key=api_key, default_model=default_model)

    elif provider_name in ("local", "vllm", "ollama", "openai", "tgi", "litellm"):
        base_url = os.getenv("LOCAL_LLM_URL", "http://localhost:8000/v1")
        model = os.getenv("LOCAL_LLM_MODEL", "")
        if not model:
            raise RuntimeError(
                "LLM_PROVIDER=local requires LOCAL_LLM_MODEL.\n"
                "Set it to the model name exposed by the OpenAI-compatible backend."
            )
        api_key = os.getenv("LOCAL_LLM_API_KEY", "not-needed")
        # Several keys (comma-separated) share the load via the KeyPool.
        api_keys = [k.strip() for k in os.getenv("LOCAL_LLM_API_KEYS", "").split(",") if k.strip()]
        _global_client = OpenAICompatibleProvider(
            base_url=base_url,
            model=model,
            api_key=api_key,
            api_keys=api_keys or [api_key],
        )

    elif provider_name == "fake":
        _global_client = FakeProvider()

    else:
        raise ValueError(
            f"Unknown LLM_PROVIDER='{provider_name}'. "
            f"Supported: gemini, local, vllm, ollama, tgi, litellm, fake"
        )

    return _global_client


class LLMManager:
    """Compatibility wrapper for older critic code paths."""

    def __init__(self, workspace: Any | None = None) -> None:
        self.workspace = workspace
        self._client = get_llm_client()

    def generate(
        self,
        *,
        role: str,
        contents: str,
        schema_name: str | None = None,
        seed: int | None = None,
        model: str | None = None,
        thinking_budget: int | None = None,
    ) -> tuple[LLMResponse, int, int, float, bool]:
        del schema_name, seed
        response = self._client.generate(
            contents,
            model=model,
            json_mode=True,
            temperature=0.2 if str(role or "").lower() == "critic" else 0.35,
            max_tokens=4096 if str(role or "").lower() == "critic" else 3072,
            timeout_sec=90.0,
            max_attempts=4,
            thinking_budget=thinking_budget,
        )
        return (
            response,
            int(response.input_tokens or 0),
            int(response.output_tokens or 0),
            float(response.cost_usd or 0.0),
            bool(response.cached),
        )


# ═══════════════════════════════════════════════════════════════════════
# Convenience helpers (used by refactored scripts)
# ═══════════════════════════════════════════════════════════════════════

def llm_generate(
    prompt: str,
    *,
    model: Optional[str] = None,
    json_mode: bool = False,
    temperature: float = 0.5,
    max_tokens: int = 4096,
    thinking_budget: int | None = None,
    timeout_sec: float = 70.0,
    max_attempts: int = 5,
    on_retry: Optional[Callable] = None,
) -> LLMResponse:
    """One-liner for LLM generation. Gets the global client automatically.

    Usage::
        from llm_provider import llm_generate
        resp = llm_generate("Generate a movie concept", json_mode=True)
        data = json.loads(resp.text)
    """
    client = get_llm_client()
    return client.generate(
        prompt,
        model=model,
        json_mode=json_mode,
        temperature=temperature,
        max_tokens=max_tokens,
        thinking_budget=thinking_budget,
        timeout_sec=timeout_sec,
        max_attempts=max_attempts,
        on_retry=on_retry,
    )


def repair_truncated_json_object(text: str) -> Any:
    """Parse a JSON object whose only defect is its ending.

    DeepSeek V4 sometimes drops the closing brace after the last string
    value, or adds a stray quote after it. This tries exactly those two
    repairs and returns the parsed dict, or None. Anything malformed in the
    middle still fails, so the caller's retry logic is unchanged for it.
    """
    value = str(text or "").strip()
    if not value.startswith("{"):
        return None
    candidates = []
    if value.endswith('"'):
        candidates.append(value + "}")           # missing final brace
    if value.endswith('}"'):
        candidates.append(value[:-1])            # stray quote after brace
    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed
    return None


def safe_json_parse(text: str) -> Any:
    """Parse JSON from LLM response, handling common issues."""
    text = str(text or "").strip()
    if not text:
        return None
    while "<think>" in text and "</think>" in text:
        start = text.find("<think>")
        end = text.find("</think>", start)
        if end == -1:
            break
        text = (text[:start] + text[end + len("</think>"):]).strip()
    # Strip markdown code fences if present
    if text.startswith("```"):
        lines = text.split("\n")
        # Remove first and last lines (``` markers)
        lines = [l for l in lines if not l.strip().startswith("```")]
        text = "\n".join(lines).strip()
    if text.lower().startswith("json"):
        text = text[4:].strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    decoder = json.JSONDecoder()
    for idx, ch in enumerate(text):
        if ch not in "[{":
            continue
        try:
            parsed, _end = decoder.raw_decode(text[idx:])
            return parsed
        except json.JSONDecodeError:
            continue
    return None
