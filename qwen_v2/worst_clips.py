#!/usr/bin/env python3
"""Per strength k: quality indicators and the worst clips, to check whether our automatic measures agree with what you hear.

Run in the project env (needs numpy, scipy, pandas, soundfile) from the repo root, after Step B (and ideally Step A) on the roots:
    python qwen_v2/worst_clips.py dev2_L21 dev_L21
Indicators per clip, computed straight from the audio (no model):
  clip_frac   share of samples with |x| > 0.98 (clipping)
  hf_ratio    share of spectral energy above 4 kHz (harsh, hissy or noisy texture)
  flatness    median spectral flatness over frames (1 = noise-like, near 0 = tonal)
From the scoring tables when present: sim_centroid (identity), wer, audio_s, effort.
Prints a table per root and k, then the 3 worst clips per k (lowest similarity; highest hf_ratio if there is no similarity yet)
with their paths so you can listen. Outputs are local only.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
from scipy.signal import stft

ap = argparse.ArgumentParser()
ap.add_argument("roots", nargs="+", help="folder names under --base, e.g. dev2_L21")
ap.add_argument("--base", default="data_local/baselines/qwen_v2")
ap.add_argument("--top", type=int, default=3)
a = ap.parse_args()
pd.set_option("display.width", 200)


def indicators(path):
    x, sr = sf.read(str(path), dtype="float32")
    if x.ndim > 1:
        x = x.mean(axis=1)
    clip = float((np.abs(x) > 0.98).mean())
    f, _, Z = stft(x, fs=sr, nperseg=1024, noverlap=512)
    P = np.abs(Z) ** 2 + 1e-12
    hf = float(P[f >= 4000].sum() / P.sum())
    flat = np.exp(np.log(P).mean(axis=0)) / P.mean(axis=0)
    active = P.sum(axis=0) > np.percentile(P.sum(axis=0), 30)  # skip near-silent frames
    return clip, hf, float(np.median(flat[active])) if active.any() else float("nan")


for name in a.roots:
    root = Path(a.base) / name
    d = pd.read_csv(root / "gap_clips.csv")
    ind = [indicators(root / p) for p in d["path"]]
    d["clip_frac"], d["hf_ratio"], d["flatness"] = zip(*ind)
    if "sim_to_centroid" in d and "sim_centroid" not in d:
        d["sim_centroid"] = d["sim_to_centroid"]  # the clip-level column is called sim_to_centroid
    rows = []
    if "sim_to_centroid" in d and "sim_centroid" not in d:
        d["sim_centroid"] = d["sim_to_centroid"]  # the clip-level column is called sim_to_centroid
    for k, h in d.groupby("exaggeration"):
        row = {
            "k": k,
            "n": len(h),
            "effort_med": round(h.score_nolevel.median(), 1),
            "audio_s_max": round(h.audio_s.max(), 1),
            "clip>0.1%": f"{int((h.clip_frac > 0.001).sum())}/{len(h)}",
            "hf_ratio_med": round(h.hf_ratio.median(), 3),
            "flatness_med": round(h.flatness.median(), 3),
        }
        if "sim_centroid" in h and h.sim_centroid.notna().any():
            row.update(
                {
                    "sim_med": round(h.sim_centroid.median(), 3),
                    "sim_p10": round(h.sim_centroid.quantile(0.1), 3),
                    "sim<0.80": f"{int((h.sim_centroid < 0.80).sum())}/{len(h)}",
                }
            )
        if "wer" in h and h.wer.notna().any():
            row["wer>0.2"] = f"{int((h.wer > 0.2).sum())}/{len(h)}"
        rows.append(row)
    print(f"\n===== {name} =====")
    print(pd.DataFrame(rows).to_string(index=False))
    key = "sim_centroid" if "sim_centroid" in d and d.sim_centroid.notna().any() else "hf_ratio"
    print(
        f"\nworst {a.top} per k by {'lowest similarity' if key == 'sim_centroid' else 'highest hf_ratio'}:"
    )
    for k, h in d.groupby("exaggeration"):
        w = h.sort_values(key, ascending=(key == "sim_centroid")).head(a.top)
        for _, r in w.iterrows():
            print(
                f"  k={k:g}  {key}={r[key]:.3f}  effort={r.score_nolevel:.1f}  {root / r['path']}"
            )
