#!/usr/bin/env python3
"""Final level stage: set the loudness of steered clips independently of the voice quality (EARS-derived audio: local only).

Steering raises speech level by +8 dB at k = 0.35 and +25 dB at k = 1 (near full scale). This stage first brings every clip to the level of
the unsteered reference clip of the same speaker and line (same 95th-percentile 25 ms frame RMS used by level_by_k.py), then adds a chosen gain
(real loud reading is about +9 dB over regular), and keeps the peak below 0.98 by scaling down if needed (the achieved gain is reported).
    python qwen_v2/level_stage.py dev2_L21_trim --gain-db 9            # writes dev2_L21_trim_lvl9
    python qwen_v2/level_stage.py dev2_L21_trim --gain-db 0 --gain-db 6 --gain-db 12
Then listen with make_ladder.py --rungs dev_base:0 dev2_L21_trim_lvl0:1.25 dev2_L21_trim_lvl6:1.25 dev2_L21_trim_lvl12:1.25 ...
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.io import wavfile

ap = argparse.ArgumentParser()
ap.add_argument("tag")
ap.add_argument("--base", default="data_local/baselines/qwen_v2")
ap.add_argument("--ref", default="dev_base")
ap.add_argument("--gain-db", type=float, action="append", required=True)
ap.add_argument("--peak", type=float, default=0.98)
a = ap.parse_args()
base = Path(a.base)


def level_db(x, sr):
    n = int(0.025 * sr)
    fr = x[: len(x) // n * n].reshape(-1, n)
    return float(20 * np.log10(np.percentile(np.sqrt((fr**2).mean(axis=1)) + 1e-9, 95)))


def read(p):
    sr, x = wavfile.read(str(p))
    x = x.astype("float32")
    return sr, (x.mean(axis=1) if x.ndim > 1 else x)


ref = pd.read_csv(base / a.ref / "results.csv")
ref = ref[ref.status == "ok"]
ref_level = {}
for _, r in ref.iterrows():
    sr, x = read(base / a.ref / r["path"])
    ref_level[(r["speaker"], r["line_id"])] = level_db(x, sr)

df = pd.read_csv(base / a.tag / "results.csv")
df = df[df.status == "ok"].copy()
for g in a.gain_db:
    dst = base / f"{a.tag}_lvl{g:g}"
    achieved, limited = [], 0
    for _, r in df.iterrows():
        sr, x = read(base / a.tag / r["path"])
        target = ref_level.get((r["speaker"], r["line_id"]))
        scale = 10 ** (
            ((target if target is not None else level_db(x, sr)) + g - level_db(x, sr)) / 20
        )
        y = x * scale
        peak = float(np.abs(y).max())
        if peak > a.peak:
            y *= a.peak / peak
            limited += 1
        achieved.append(level_db(y, sr) - (target if target is not None else level_db(x, sr)))
        (dst / r["path"]).parent.mkdir(parents=True, exist_ok=True)
        wavfile.write(str(dst / r["path"]), sr, y.astype("float32"))
    out = df.copy()
    out["achieved_gain_db"] = np.round(achieved, 2)
    out.to_csv(dst / "results.csv", index=False)
    print(
        f"{a.tag} +{g:g} dB -> {dst}: {len(df)} clips, peak-limited {limited}, achieved gain median {np.median(achieved):.1f} dB"
    )
