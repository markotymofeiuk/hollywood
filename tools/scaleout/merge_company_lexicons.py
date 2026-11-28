#!/usr/bin/env python3
"""Merge company lexicons into one with room for SF1 and SF10 company names.

    merge_company_lexicons.py <out.json> <base.json> <extra.json>...

The shipped lexicon (base) allows 17,460 distinct names, and the generator
gives up at about 93% of that (row 16,178), short of SF1's 19,300 companies.
Each extra lexicon is a sample of the same step-18 prompt (company_lexicon in
generate_bootstrap_artifacts_api.py) run for the target scale. Word lists and
templates are unioned in order (base first, duplicates dropped); tier_styles
and country_style_bias are kept from the base. The result is checked with the
generator's own validator and its name capacity is recorded in meta.
"""
from __future__ import annotations

import itertools
import json
import string
import sys
from pathlib import Path

GEN = Path(__file__).resolve().parents[2] / "generator"
sys.path.insert(0, str(GEN))

from generate_bootstrap_artifacts_api import _validate_company_lexicon  # noqa: E402
from generate_companies_procedural import _canonicalize_company_template  # noqa: E402

LIST_KEYS = {
    "prefix": "prefixes",
    "suffix": "suffixes",
    "abstract": "abstract_nouns",
    "geo": "geographic_words",
    "motion": "motion_words",
    "mythic": "mythic_words",
    "material": "material_words",
}


def capacity(lex: dict) -> int:
    """Distinct names, deduplicated the way generate_companies_procedural does."""
    pools = {slot: list(dict.fromkeys(lex.get(key, []))) for slot, key in LIST_KEYS.items()}
    pools["local_word"] = pools["geo"]
    names: set[str] = set()
    for raw in dict.fromkeys(lex["templates"]):
        template = _canonicalize_company_template(raw)
        slots = sorted({field for _, field, _, _ in string.Formatter().parse(template) if field})
        for combo in itertools.product(*(pools[s] for s in slots)):
            name = " ".join(template.format(**dict(zip(slots, combo))).split()).strip()
            if name:
                names.add(name.casefold())
    return len(names)


def main() -> None:
    out_path, base_path, *extra_paths = sys.argv[1:]
    base = json.loads(Path(base_path).read_text())
    extras = [json.loads(Path(p).read_text()) for p in extra_paths]
    merged = json.loads(json.dumps(base))
    for key in [*LIST_KEYS.values(), "templates"]:
        values = list(base.get(key, []))
        seen = {str(v).casefold() for v in values}
        for lex in extras:
            for v in lex.get(key, []):
                if str(v).strip() and str(v).casefold() not in seen:
                    seen.add(str(v).casefold())
                    values.append(v)
        merged[key] = values
    _validate_company_lexicon(merged)
    merged["meta"] = {
        **dict(base.get("meta", {})),
        "merged": {
            "base": Path(base_path).name,
            "extras": [Path(p).name for p in extra_paths],
            "method": "ordered union of word lists and templates; tier_styles and country_style_bias from base",
            "name_capacity": capacity(merged),
            "name_capacity_base": capacity(base),
        },
    }
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(json.dumps(merged, indent=2, ensure_ascii=False) + "\n")
    sizes = {k: len(merged[k]) for k in [*LIST_KEYS.values(), "templates"]}
    print(f"{out_path}: {merged['meta']['merged']['name_capacity']:,} names (base {merged['meta']['merged']['name_capacity_base']:,}) {sizes}")


if __name__ == "__main__":
    main()
