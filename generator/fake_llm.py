"""Offline fake LLM backend for measurement and parallelism tests.

Select it with ``LLM_PROVIDER=fake``. Every response is a pure function of the
prompt text: the handler that recognises the prompt seeds a private RNG with a
SHA-256 of the prompt, so the same prompt always yields the same answer,
whatever the call order or concurrency. That property is what lets a parallel
run be compared byte-for-byte with a serial one.

Handlers return JSON that the step's own parser accepts, with plausible sizes
so token estimates are meaningful. They make no attempt at semantic quality.

Token counts are estimates (characters / 4) because there is no tokenizer.

Environment:
  FAKE_LLM_LATENCY_SEC   optional sleep per call, to simulate network latency
  FAKE_LLM_UNHANDLED_DIR where to dump prompts no handler recognises
                         (default: ./fake_llm_unhandled)
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import random
import re
import time
from pathlib import Path
from typing import Any, Callable

from contracts import DIRECTOR_STYLES, GENRES, MARKETS, STYLE_TAGS

Handler = Callable[[str, random.Random], Any]

_WORDS = (
    "acclaimed restless luminous stubborn quiet fearless meticulous wry tender "
    "ambitious celebrated unlikely veteran gifted volatile patient precise bold "
    "film director actor writer producer composer studio festival debut career "
    "drama comedy thriller ensemble stage television screen role award nominee "
    "breakthrough collaboration independent mainstream international audience "
    "critics season premiere archive harbor city river border winter summer "
    "family secret promise memory frontier signal machine garden letter storm"
).split()


def _rng_for(prompt: str, salt: str) -> random.Random:
    digest = hashlib.sha256((salt + "\x00" + prompt).encode("utf-8")).digest()
    return random.Random(int.from_bytes(digest[:16], "big"))


def _sentence(rng: random.Random, n_words: int) -> str:
    words = [rng.choice(_WORDS) for _ in range(max(3, n_words))]
    words[0] = words[0].capitalize()
    return " ".join(words) + "."


def _prose(rng: random.Random, min_words: int, max_words: int) -> str:
    target = rng.randint(min_words, max_words)
    parts: list[str] = []
    used = 0
    while used < target:
        n = min(target - used, rng.randint(9, 18))
        n = max(n, 3)
        parts.append(_sentence(rng, n))
        used += n
    return " ".join(parts)


def _f3(rng: random.Random, lo: float = 0.0, hi: float = 1.0) -> float:
    return round(rng.uniform(lo, hi), 3)


def _ids(prompt: str) -> list[int]:
    return [int(m.group(1)) for m in re.finditer(r"^\[(\d+)\]", prompt, flags=re.MULTILINE)]


# ---------------------------------------------------------------------------
# Step 40: person enrichment
# ---------------------------------------------------------------------------

def _persons_enrich(prompt: str, rng: random.Random) -> Any:
    styles = list(STYLE_TAGS) + list(DIRECTOR_STYLES)
    out = []
    for pid in _ids(prompt):
        out.append(
            {
                "person_id": pid,
                "bio": _summary(rng, rng.choice(_OPENERS), (2, 3), (45, 95)),
                "style_tags": rng.sample(styles, rng.randint(2, 4)),
                "genre_affinity": rng.sample(list(GENRES), rng.randint(1, 3)),
                "market_fit": rng.sample(list(MARKETS), rng.randint(1, 2)),
            }
        )
    return out


# ---------------------------------------------------------------------------
# Step 70: latent variables
# ---------------------------------------------------------------------------

def _person_latents(prompt: str, rng: random.Random) -> Any:
    out = []
    for pid in _ids(prompt):
        out.append(
            {
                "person_id": pid,
                "creative_style_vector": [_f3(rng, -1.0, 1.0) for _ in range(8)],
                "risk_tolerance": _f3(rng),
                "collaboration_style": rng.choice(["solo", "ensemble", "chameleon", "mentorship"]),
                "controversy_score": _f3(rng),
                "public_reputation": _f3(rng),
                "budget_band_pref": [_f3(rng) for _ in range(5)],
                "artistic_ambition": _f3(rng),
                "volatility": _f3(rng),
                "avoid_genres": rng.sample(list(GENRES), rng.randint(1, 3)),
            }
        )
    return out


def _normalised(rng: random.Random, n: int) -> list[float]:
    raw = [rng.random() + 1e-3 for _ in range(n)]
    total = sum(raw)
    return [round(v / total, 3) for v in raw]


def _company_latents(prompt: str, rng: random.Random) -> Any:
    out = []
    for cid in _ids(prompt):
        out.append(
            {
                "company_id": cid,
                "risk_appetite": _f3(rng),
                "prestige_score": _f3(rng),
                "genre_portfolio": _normalised(rng, 12),
                "budget_tier_focus": _normalised(rng, 5),
                "market_trend_sensitivity": _f3(rng),
                "controversy_tolerance": _f3(rng),
            }
        )
    return out


# ---------------------------------------------------------------------------
# Step 95: keyword genres
# ---------------------------------------------------------------------------

def _json_after(prompt: str, marker: str) -> Any:
    idx = prompt.find(marker)
    if idx < 0:
        raise ValueError(f"marker {marker!r} not in prompt")
    decoder = json.JSONDecoder()
    rest = prompt[idx + len(marker):].lstrip()
    value, _ = decoder.raw_decode(rest)
    return value


def _keyword_genres(prompt: str, rng: random.Random) -> Any:
    genres = _json_after(prompt, "genre from this list:")
    keywords = _json_after(prompt, "Keywords:")
    return [
        {
            "keyword": kw,
            "topic_genre": rng.choice(genres),
            "pop_weight": round(math.exp(rng.uniform(math.log(0.005), math.log(0.12))), 4),
        }
        for kw in keywords
    ]


# ---------------------------------------------------------------------------
# Step 100: concept rerank, keyword rerank, yearly planner, critic
# ---------------------------------------------------------------------------

def _concept_rerank(prompt: str, rng: random.Random) -> Any:
    items = _json_after(prompt, "Candidates:")
    out = []
    for item in items:
        candidates = item.get("candidates") or []
        if not candidates:
            continue
        # Mimic the instruction "avoid mode collapse": prefer less-used packs.
        weights = [1.0 / (1.0 + float(c.get("usage_count", 0) or 0)) for c in candidates]
        choice = rng.choices(candidates, weights=weights, k=1)[0]
        out.append(
            {
                "movie_id": item.get("movie_id"),
                "selected_pack_id": choice.get("pack_id"),
                "selected_index": choice.get("index", 0),
            }
        )
    return out


def _keyword_rerank(prompt: str, rng: random.Random) -> Any:
    items = _json_after(prompt, "Items:")
    out = []
    for item in items:
        candidates = item.get("candidates") or []
        if not candidates:
            continue
        on_genre = [c for c in candidates if c.get("genre_match")]
        off_genre = [c for c in candidates if not c.get("genre_match")]
        want = min(len(candidates), rng.randint(4, 8))
        picked = rng.sample(on_genre, min(len(on_genre), want))
        if len(picked) < want and off_genre:
            picked += rng.sample(off_genre, min(len(off_genre), 1))
        out.append(
            {
                "movie_id": item.get("movie_id"),
                "selected_keyword_ids": [int(c["keyword_id"]) for c in picked],
            }
        )
    return out


_PLANNER_BLUEPRINTS = {
    "friendship_clustering": (("friendship", "create"), ("collaboration", "strengthen"), ("clique", "create")),
    "rivalry_wave": (("rivalry", "create"), ("avoid", "create")),
    "mentor_tree": (("mentorship", "create"), ("friendship", "create")),
    "director_camp": (("collaboration", "strengthen"), ("mentorship", "create")),
    "recruiting_wave": (("brand_fit", "create"), ("employment", "create")),
    "blacklist_cascade": (("blacklist", "create"), ("avoid", "create"), ("employment", "expire")),
    "prestige_alliance": (("co_production", "create"), ("brand_fit", "strengthen"), ("friendship", "create")),
    "bridge_building": (("friendship", "create"), ("collaboration", "create"), ("co_production", "create")),
    "company_competition_shift": (("market_rival", "create"), ("co_production", "expire"), ("brand_fit", "weaken")),
    "regional_migration_wave": (("friendship", "create"), ("collaboration", "create"), ("brand_fit", "create")),
    "scandal_fallout": (("avoid", "create"), ("rivalry", "create"), ("blacklist", "create")),
}
_PLANNER_SCOPES = ("local_frontier", "cross_community", "director_camp", "competition_frontier", "company_frontier")
_PLANNER_COHORTS = ("active_people", "rising_people", "volatile_people", "controversial_people", "cross_market_people")


def _year_plan(prompt: str, rng: random.Random) -> Any:
    world = _json_after(prompt, "World payload:")
    people = [int(p["person_id"]) for p in world.get("sample_people", []) if p.get("person_id")]
    companies = [
        int(c["company_id"])
        for c in world.get("sample_companies", []) or world.get("companies", []) or []
        if isinstance(c, dict) and c.get("company_id")
    ]
    genres = [g for g, _n in (world.get("summary", {}) or {}).get("top_genres", [])] or list(GENRES)
    programs = []
    for motif in rng.sample(sorted(_PLANNER_BLUEPRINTS), rng.randint(6, 10)):
        edge_type, mode = rng.choice(_PLANNER_BLUEPRINTS[motif])
        programs.append(
            {
                "motif": motif,
                "edge_type": edge_type,
                "mode": mode,
                "target_cohort": rng.choice(_PLANNER_COHORTS),
                "graph_scope": rng.choice(_PLANNER_SCOPES),
                "anchors_people": rng.sample(people, min(len(people), rng.randint(1, 3))),
                "anchors_companies": rng.sample(companies, min(len(companies), rng.randint(0, 2))),
                "intensity": _f3(rng, 0.3, 0.8),
                "reach": rng.choice(["local", "cohort", "regional", "industry"]),
                "fanout": rng.randint(2, 7),
                "max_depth": rng.randint(1, 3),
                "weight_mean": _f3(rng, 0.25, 0.7),
                "weight_spread": _f3(rng, 0.03, 0.15),
                "delta_weight": _f3(rng, -0.15, 0.2),
                "cross_community_bias": _f3(rng, 0.1, 1.0),
                "novelty_bias": _f3(rng, 0.1, 0.8),
                "durability_years": rng.randint(1, 6),
                "reason": _sentence(rng, 8),
            }
        )
    return {
        "world_event_narrative": _prose(rng, 18, 36),
        "market_regime_moves": [
            {"kind": "genre_delta", "genre": g, "delta": _f3(rng, -0.04, 0.05)}
            for g in rng.sample(genres, min(len(genres), rng.randint(1, 3)))
        ],
        "person_shocks": [
            {"person_id": pid, "reputation_delta": _f3(rng, -0.08, 0.08), "controversy_delta": _f3(rng, -0.05, 0.05)}
            for pid in rng.sample(people, min(len(people), rng.randint(1, 4)))
        ],
        "company_moves": [],
        "relationship_programs": programs,
    }


def _critic(prompt: str, rng: random.Random) -> Any:
    payload = json.loads(prompt)
    movies = payload.get("sampled_movies") or []
    actions = []
    flagged = []
    for movie in rng.sample(movies, min(len(movies), 10)):
        tid = int(movie["title_id"])
        genre = str(movie.get("genre") or "film")
        flagged.append({"title_id": tid, "issues": ["generic synopsis"]})
        actions.append(
            {
                "action": "rewrite_plot_summary",
                "params": {"title_id": tid, "plot_summary": _prose(rng, 45, 80)},
                "reason": "generic synopsis",
            }
        )
        actions.append(
            {
                "action": "rewrite_tagline",
                "params": {"title_id": tid, "tagline": f"{_sentence(rng, 5)[:-1]} {genre.lower()} {tid}"},
                "reason": "generic tagline",
            }
        )
    return {"critic_summary": _sentence(rng, 14), "flagged_titles": flagged, "actions": actions[:20]}


# ---------------------------------------------------------------------------
# Steps 110 and 120: plot, series and episode text
# ---------------------------------------------------------------------------

_OPENERS = (
    "Years", "When", "After", "Before", "Inside", "Across", "Under", "Beneath", "Between",
    "Every", "Once", "Months", "Halfway", "Nobody", "Somewhere", "Late", "Early", "Long",
    "Three", "Two", "Four", "Rumors", "Letters", "Debts", "Storms", "Borders", "Winters",
    "Summers", "Harbors", "Rivers", "Signals", "Gardens", "Secrets", "Promises", "Memories",
)


def _summary(rng: random.Random, opener: str, sentences: tuple[int, int], words: tuple[int, int]) -> str:
    n_sent = rng.randint(*sentences)
    total = rng.randint(*words)
    sizes = [total // n_sent] * n_sent
    sizes[-1] += total - sum(sizes)
    parts = []
    for i, size in enumerate(sizes):
        sent = _sentence(rng, size)
        if i == 0:
            sent = opener + " " + sent[0].lower() + sent[1:]
        parts.append(sent)
    return " ".join(parts)


def _openers(rng: random.Random, n: int) -> list[str]:
    pool = list(_OPENERS)
    rng.shuffle(pool)
    return [pool[i % len(pool)] + ("" if i < len(pool) else f" {i}") for i in range(n)]


def _plot_batch(prompt: str, rng: random.Random) -> Any:
    ids = [int(m.group(1)) for m in re.finditer(r'^\[(\d+)\] "', prompt, flags=re.MULTILINE)]
    return [
        {"title_id": tid, "plot_summary": _summary(rng, opener, (2, 3), (45, 80))}
        for tid, opener in zip(ids, _openers(rng, len(ids)))
    ]


def _plot_single(prompt: str, rng: random.Random) -> Any:
    return _summary(rng, rng.choice(_OPENERS), (2, 3), (45, 80))


def _series_overview(prompt: str, rng: random.Random) -> Any:
    return "\n\n".join(
        _summary(rng, rng.choice(_OPENERS), (s, s), w)
        for s, w in ((3, (60, 80)), (2, (40, 55)), (1, (22, 30)))
    )


def _series_retry(prompt: str, rng: random.Random) -> Any:
    return _summary(rng, rng.choice(_OPENERS), (4, 5), (95, 140))


def _season_episodes(prompt: str, rng: random.Random) -> Any:
    ids = [int(m.group(1)) for m in re.finditer(r"^\s*\[(\d+)\] Ep", prompt, flags=re.MULTILINE)]
    out: dict[str, str] = {}
    for eid, opener in zip(ids, _openers(rng, len(ids))):
        out[str(eid)] = _summary(rng, opener, (2, 3), (36, 58))
    out["_finale_line"] = _sentence(rng, 14)
    return out


def _episode_single(prompt: str, rng: random.Random) -> Any:
    return _summary(rng, rng.choice(_OPENERS), (2, 3), (36, 66))


def _episode_titles(prompt: str, rng: random.Random) -> Any:
    ids = [int(m.group(1)) for m in re.finditer(r"^\s*\[(\d+)\] Ep", prompt, flags=re.MULTILINE)]
    return {
        str(eid): " ".join(w.capitalize() for w in rng.sample(_WORDS, rng.randint(2, 4)))
        for eid in ids
    }


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------

HANDLERS: list[tuple[str, str, Handler]] = [
    ("persons_enrich", "You are enriching a synthetic film-industry person roster.", _persons_enrich),
    ("person_latents", "You are assigning numeric latent variables to movie industry persons", _person_latents),
    ("company_latents", "You are assigning numeric latent variables to movie production companies", _company_latents),
    ("keyword_genres", "For each keyword below, assign:\n1. \"topic_genre\"", _keyword_genres),
    ("concept_rerank", "You are choosing the strongest reusable movie concept candidate", _concept_rerank),
    ("keyword_rerank", "You are refining keyword bundles for synthetic IMDb-style movie rows.", _keyword_rerank),
    ("year_plan", "You are planning one year of synthetic film-industry evolution.", _year_plan),
    ("critic", "\"task\": \"post_generation_critic\"", _critic),
    ("plot_batch", "Write brief plot summaries for these", _plot_batch),
    ("plot_single", "Write one polished movie plot summary", _plot_single),
    ("series_overview", "Write a compelling 2-3 paragraph series overview", _series_overview),
    ("series_retry", "Write exactly one polished streaming-platform series overview paragraph", _series_retry),
    ("season_episodes", "Write one-paragraph episode descriptions for every episode", _season_episodes),
    ("episode_single", "Write one polished episode description", _episode_single),
    ("episode_titles", "Write improved episode titles", _episode_titles),
]


class FakeLLMUnhandled(ValueError):
    """Raised for a prompt no handler recognises (not retriable)."""


def _dump_unhandled(prompt: str) -> Path:
    root = Path(os.getenv("FAKE_LLM_UNHANDLED_DIR", "fake_llm_unhandled"))
    root.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:16]
    step = os.getenv("DATA_SYS_STEP_ID", "x")
    path = root / f"step{step}_{digest}.txt"
    path.write_text(prompt, encoding="utf-8")
    return path


def classify(prompt: str) -> str | None:
    for name, marker, _handler in HANDLERS:
        if marker in prompt:
            return name
    return None


def fake_response_text(prompt: str) -> tuple[str, str]:
    """Return (handler_name, response_text) for a prompt."""
    for name, marker, handler in HANDLERS:
        if marker in prompt:
            payload = handler(prompt, _rng_for(prompt, name))
            text = payload if isinstance(payload, str) else json.dumps(payload, ensure_ascii=False)
            return name, text
    path = _dump_unhandled(prompt)
    raise FakeLLMUnhandled(f"FAKE_LLM_UNHANDLED prompt dumped to {path}")


def estimate_tokens(text: str) -> int:
    return int(math.ceil(len(text) / 4.0))


def fake_latency() -> None:
    raw = os.getenv("FAKE_LLM_LATENCY_SEC", "").strip()
    if not raw:
        return
    try:
        delay = float(raw)
    except ValueError:
        return
    if delay > 0:
        time.sleep(delay)
