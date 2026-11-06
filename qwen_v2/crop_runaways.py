#!/usr/bin/env python3
"""Crop runaway clips to the length of the sentence and write a new root you can score and listen to like any other.

Why: when steering stops the model from ending its utterance, the clip keeps going (babble, laughter, noise) after the line
is finished. If the LINE itself is fine, the "maniacal" part may be only the runaway tail, and a duration-bounded generation
(or a crop) would keep the shout and drop the tail. This script simulates that offline, without regenerating anything.

For every clip longer than  factor x (unsteered duration of the same speaker and line) + pad  it keeps only that much audio
(100 ms fade-out); shorter clips are copied unchanged. The unsteered durations come from the --ref root (default dev_base).
New root: <base>/<tag>_head with the same layout and a results.csv (audio_s updated, orig_audio_s added), so Step A, Step B,
sweep_summary.py, make_ladder.py and worst_clips.py all work on it. Run in the project env, repo root:
    python qwen_v2/crop_runaways.py dev_L21 dev_L14 dev2_L14_18_21_24
Outputs are EARS-derived voices: local only.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf

ap = argparse.ArgumentParser()
ap.add_argument("tags", nargs="+")
ap.add_argument("--base", default="data_local/baselines/qwen_v2")
ap.add_argument("--ref", default="dev_base")
ap.add_argument("--factor", type=float, default=1.4)
ap.add_argument("--pad", type=float, default=0.6)
ap.add_argument("--min-s", type=float, default=1.5)
a = ap.parse_args()
base = Path(a.base)

ref = pd.read_csv(base / a.ref / "results.csv")
ref = ref[ref.status == "ok"]
expected = ref.groupby(["speaker", "line_id"])["audio_s"].median().to_dict()

for tag in a.tags:
    src, dst = base / tag, base / f"{tag}_head"
    df = pd.read_csv(src / "results.csv")
    df = df[df.status == "ok"].copy()
    new_len, n_cropped = [], 0
    for _, r in df.iterrows():
        x, sr = sf.read(str(src / r["path"]), dtype="float32")
        limit = max(a.min_s, a.factor * expected.get((r["speaker"], r["line_id"]), 4.0) + a.pad)
        out = dst / r["path"]
        out.parent.mkdir(parents=True, exist_ok=True)
        if len(x) / sr > limit:
            x = x[: int(limit * sr)].copy()
            fade = int(0.1 * sr)
            x[-fade:] *= np.linspace(1.0, 0.0, fade, dtype="float32")
            n_cropped += 1
        sf.write(str(out), x, sr, subtype="FLOAT")
        new_len.append(round(len(x) / sr, 3))
    df["orig_audio_s"] = df["audio_s"]
    df["audio_s"] = new_len
    df.to_csv(dst / "results.csv", index=False)
    print(f"{tag}: {len(df)} clips, {n_cropped} cropped -> {dst}")
