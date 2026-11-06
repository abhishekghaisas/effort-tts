#!/usr/bin/env python3
"""Does steering make the speech LOUDER, or only change its spectrum? (the effort estimator is level-free by design)

Per strength k, the paired change in speech level versus the unsteered clip (same speaker and line), in dB. Level = 95th percentile
of 25 ms frame RMS (dBFS), which ignores pauses. Also the sample peak: over_range counts clips whose peak exceeds 1.0 (they clip
when played), near_full counts peaks above 0.98. Real EARS speech for reference: loud reading is about +9 dB over regular reading.
Run in the project env, repo root:
    python qwen_v2/level_by_k.py dev2_L21
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
a = ap.parse_args()
base = Path(a.base)


def level_db(path):
    x, sr = sf.read(str(path), dtype="float32")
    if x.ndim > 1:
        x = x.mean(axis=1)
    n = int(0.025 * sr)
    frames = x[: len(x) // n * n].reshape(-1, n)
    rms = np.sqrt((frames**2).mean(axis=1)) + 1e-9
    return float(20 * np.log10(np.percentile(rms, 95))), float(np.abs(x).max())


def table(tag):
    d = pd.read_csv(base / tag / "results.csv")
    d = d[d.status == "ok"].copy()
    vals = [level_db(base / tag / p) for p in d["path"]]
    d["level"], d["peak"] = [v[0] for v in vals], [v[1] for v in vals]
    return d


ref = table(a.ref).set_index(["speaker", "line_id", "seed"])["level"]
print(f"unsteered level: median {ref.median():.1f} dBFS\n")
for tag in a.tags:
    d = table(tag)
    d["delta_db"] = (
        d["level"].to_numpy() - ref.reindex(list(zip(d.speaker, d.line_id, d.seed))).to_numpy()
    )
    out = d.groupby("exaggeration").agg(
        n=("level", "size"),
        level_dbfs=("level", "median"),
        delta_db_median=("delta_db", "median"),
        delta_db_p10=("delta_db", lambda s: s.quantile(0.1)),
        delta_db_p90=("delta_db", lambda s: s.quantile(0.9)),
        peak_median=("peak", "median"),
        peak_max=("peak", "max"),
        over_range=("peak", lambda s: int((s > 1.0).sum())),
        near_full=("peak", lambda s: int((s > 0.98).sum())),
    )
    print(f"== {tag} ==\n{out.round(2).to_string()}\n")
