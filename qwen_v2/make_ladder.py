#!/usr/bin/env python3
"""Concatenate one line at increasing steering strength into a single WAV so you can hear the whole ladder in one go.

    python qwen_v2/make_ladder.py --spk p023 --line D03
Default rungs: unsteered, then k = 0.10 ... 1.5 (dev2_L21) and k = 2 (dev_L21, the distorted end). Missing files are skipped.
0.8 s of silence separates the rungs; the printed order matches the audio. Output stays local (EARS-derived voices): do not share it.
"""

import argparse
from pathlib import Path

import numpy as np
import soundfile as sf

ap = argparse.ArgumentParser()
ap.add_argument("--root", default="data_local/baselines/qwen_v2")
ap.add_argument("--spk", required=True)
ap.add_argument("--line", required=True)
ap.add_argument(
    "--rungs",
    nargs="*",
    default=[
        "dev_base:0",
        "dev2_L21:0.10",
        "dev2_L21:0.20",
        "dev2_L21:0.35",
        "dev2_L21:0.50",
        "dev2_L21:0.75",
        "dev2_L21:1",
        "dev2_L21:1.25",
        "dev2_L21:1.5",
        "dev_L21:2",
    ],
    help="TAG:k pairs",
)
ap.add_argument("--out", default=None)
ap.add_argument(
    "--equal-level",
    action="store_true",
    help="scale every rung to the level of the first (unsteered) rung, so you hear voice quality without the level change",
)
a = ap.parse_args()


def level_db(x, sr):
    n = int(0.025 * sr)
    frames = x[: len(x) // n * n].reshape(-1, n)
    return 20 * np.log10(np.percentile(np.sqrt((frames**2).mean(axis=1)) + 1e-9, 95))


chunks, order, sr0, ref_db = [], [], None, None
for r in a.rungs:
    tag, k = r.split(":")
    p = Path(a.root) / tag / a.spk / f"{a.line}_k{float(k):.2f}_s0.wav"
    if not p.exists():
        print("missing:", p)
        continue
    x, sr = sf.read(str(p), dtype="float32")
    sr0 = sr0 or sr
    note = ""
    if a.equal_level:
        ref_db = level_db(x, sr) if ref_db is None else ref_db
        gain = 10 ** ((ref_db - level_db(x, sr)) / 20)
        gain = min(gain, 0.99 / max(float(np.abs(x).max()) * 1.0, 1e-9)) if gain > 1 else gain
        x = (x * gain).astype("float32")
        note = f", scaled {20 * np.log10(gain):+.1f} dB"
    chunks += [x, np.zeros(int(0.8 * sr), dtype="float32")]
    order.append(f"k={float(k):g} ({len(x) / sr:.1f}s{note})")
if not chunks:
    raise SystemExit("no clips found")
out = a.out or f"ladder_{a.spk}_{a.line}{'_equal' if a.equal_level else ''}.wav"
sf.write(out, np.concatenate(chunks), sr0)
print("wrote", out, "| order:", " -> ".join(order))
