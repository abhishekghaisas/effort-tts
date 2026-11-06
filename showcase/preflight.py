#!/usr/bin/env python3
"""Check every input of the showcase BEFORE building: do the roots exist, are they scored (Step A and B), do they hold the settings we want, are the wavs there?
Run in the project env, repo root:   python showcase/preflight.py --include-finetune
Prints one line per source root and a verdict. Nothing is written."""

import argparse
from pathlib import Path

import build_showcase as B
import numpy as np
import pandas as pd

NEED = [
    "speaker",
    "line_id",
    "text",
    "exaggeration",
    "path",
    "score_nolevel",
    "wer",
    "sim_to_centroid",
    "words_per_s",
    "audio_s",
    "hypothesis",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="data_local/baselines")
    ap.add_argument("--speakers", nargs="*", default=["p023", "p045", "p056"])
    ap.add_argument("--include-finetune", action="store_true")
    a = ap.parse_args()
    base, problems = Path(a.base), 0
    for group, rel, settings in B.QWEN + B.QWEN_ANGER + B.CB + (B.FT if a.include_finetune else []):
        root = base / rel
        tag = f"{group:17s} {rel:32s}"
        if not (root / "gap_clips.csv").exists():
            optional = group == "qwen_anger"
            print(
                f"{'optional' if optional else 'MISSING '} {tag} no gap_clips.csv: run Step A (Chatterbox env) then Step B (project env, --default-ex {0.5 if group.startswith('chatterbox') else 0})"
            )
            problems += 0 if optional else 1
            continue
        g = pd.read_csv(root / "gap_clips.csv")
        miss_cols = [c for c in NEED if c not in g.columns]
        have = sorted({round(float(x), 2) for x in g.exaggeration})
        miss_set = [s for s in settings if not any(np.isclose(s, h) for h in have)]
        scored = float(g.wer.notna().mean()) if "wer" in g else 0.0
        wavs = float(np.mean([(root / p).exists() for p in g["path"]])) if "path" in g else 0.0
        spk_missing = [s for s in a.speakers if s not in set(g.speaker)]
        emo = (root / "emotion_dim.csv").exists()
        notes = []
        if miss_cols:
            notes.append(f"missing columns {miss_cols}")
        if miss_set:
            notes.append(f"settings not found {miss_set}")
        if spk_missing:
            notes.append(f"speakers missing {spk_missing}")
        if scored < 0.99:
            notes.append(
                f"ASR/similarity on only {scored:.0%} of clips: run Step A, then Step B again"
            )
        if wavs < 0.99:
            notes.append(f"{1 - wavs:.0%} of wav files not found")
        if group.startswith("qwen") and not emo and rel.split("/")[-1] != "dev_base":
            notes.append(
                "no emotion_dim.csv (arousal and valence will be blank): run emotion_dim.py on this root"
            )
        hard = bool(miss_cols or miss_set or spk_missing or scored < 0.99 or wavs < 0.99)
        problems += int(hard)
        print(
            f"{'PROBLEM ' if hard else 'ok      '} {tag} {len(g):4d} clips; settings {have if len(have) <= 8 else str(len(have)) + ' values'}"
            + (" | " + "; ".join(notes) if notes else "")
        )
    full = base / "chatterbox/cfg0.50/gap_clips.csv"
    print(
        f"{'ok      ' if full.exists() else 'optional'} chatterbox_full    {'full baseline found (metrics only)' if full.exists() else 'full Chatterbox baseline not found: the pass-rate comparison will lack it'}"
    )
    print(
        "\nREADY to build"
        if not problems
        else f"\n{problems} source(s) need attention before building"
    )


if __name__ == "__main__":
    main()
