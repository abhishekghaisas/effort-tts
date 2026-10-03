"""Fixed 30-line evaluation script: loader, validator, and EARS-overlap check.

    python -m efforttts.baselines.test_script

Checks that (a) the script is well-formed (30 lines, 10 per intent, unique ids,
varied length) and (b) no line exactly matches or shares a 4-word run with any
EARS transcript, so the lines are genuinely unseen if the TTS is fine-tuned on
EARS. The overlap check needs transcripts.json (download --meta-only).
"""
from __future__ import annotations

import json
import re
import sys
from collections import Counter
from pathlib import Path

from efforttts import config

SCRIPT_PATH = config.REPO_ROOT / "configs" / "test_script.yaml"
INTENTS = ("calm", "urgent", "shout")
SHINGLE_N = 4


def load_script(path: Path | None = None) -> list[dict]:
    import yaml

    data = yaml.safe_load(Path(path or SCRIPT_PATH).read_text())
    return data["lines"]


def normalize(text: str) -> list[str]:
    return re.sub(r"[^a-z0-9' ]+", " ", text.lower()).split()


def shingles(words: list[str], n: int = SHINGLE_N) -> set[tuple[str, ...]]:
    return {tuple(words[i : i + n]) for i in range(len(words) - n + 1)}


def validate(lines: list[dict]) -> list[str]:
    problems = []
    ids = [l["id"] for l in lines]
    if len(lines) != 30:
        problems.append(f"expected 30 lines, found {len(lines)}")
    if len(set(ids)) != len(ids):
        problems.append("duplicate ids")
    counts = Counter(l["intent"] for l in lines)
    for k in INTENTS:
        if counts.get(k, 0) != 10:
            problems.append(f"intent '{k}' has {counts.get(k, 0)} lines, expected 10")
    if len({" ".join(normalize(l["text"])) for l in lines}) != len(lines):
        problems.append("duplicate texts")
    return problems


def check_overlap(lines: list[dict], transcripts: dict[str, str]) -> list[tuple[str, str, str]]:
    """Return (line_id, transcript_key, reason) for exact matches and shared 4-word runs."""
    t_words = {k: normalize(v) for k, v in transcripts.items()}
    t_sh = {k: shingles(w) for k, w in t_words.items()}
    hits = []
    for l in lines:
        w = normalize(l["text"])
        sh = shingles(w)
        for k, tw in t_words.items():
            if w == tw:
                hits.append((l["id"], k, "identical text"))
            elif sh & t_sh[k]:
                hits.append((l["id"], k, f"shared run: {' '.join(next(iter(sh & t_sh[k])))}"))
    return hits


def main() -> int:
    lines = load_script()
    problems = validate(lines)
    n_words = sorted(len(normalize(l["text"])) for l in lines)
    print(f"{len(lines)} lines; words per line: min {n_words[0]}, median {n_words[len(n_words)//2]}, max {n_words[-1]}")
    print("per intent:", dict(Counter(l["intent"] for l in lines)))
    for p in problems:
        print("PROBLEM:", p)

    tpath = config.raw_dir() / "transcripts.json"
    if tpath.exists():
        hits = check_overlap(lines, json.loads(tpath.read_text()))
        print(f"EARS overlap check ({len(json.loads(tpath.read_text()))} transcripts): {len(hits)} hits")
        for h in hits:
            print("  OVERLAP:", h)
        problems += [str(h) for h in hits]
    else:
        print("transcripts.json not found; overlap check skipped (run download --meta-only)")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())