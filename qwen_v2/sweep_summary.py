#!/usr/bin/env python3
"""Summarize a dev sweep after Step B has been run on every dev_* root (project env, from the repo root):
    for d in data_local/baselines/qwen_v2/dev_*; do python -m efforttts.eval.score_baseline --root $d --default-ex 0; done
    python qwen_v2/sweep_summary.py
Per layer configuration and strength k:
  terminated      clips that ended on their own (audio <= --max-audio, default 10 s, fine for the dev lines); the others hit the
                  token limit (runaway). Check the unsteered maximum printed first before using this on longer lines.
  effort_*_term   effort statistics on terminated clips only (a 40 s runaway is babble, its effort score is meaningless)
  delta_*_term    paired change versus the unsteered clip (same speaker, line, seed), terminated clips only
  share_up_term   share of terminated clips whose effort went up
Roots ending in _b (bounded generation) report capped clips (reached the length cap, so they were faded out) separately, with their own
similarity and WER: a high WER on capped clips would mean words were cut off at the end.
"""

import argparse
from pathlib import Path

import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--root", default="data_local/baselines/qwen_v2")
ap.add_argument(
    "--bound-factor",
    type=float,
    default=1.5,
    help="for roots whose name ends in _b (generated with qwen_gen.py --bound-factor): the factor used, to recognise capped clips",
)
ap.add_argument(
    "--max-audio",
    type=float,
    default=10.0,
    help="seconds; longer clips are counted as runaway (a 150-token limit ends at 12 s, a 500-token limit at 40 s)",
)
a = ap.parse_args()
root = Path(a.root)

base = pd.read_csv(root / "dev_base" / "gap_clips.csv")
b = base.set_index(["speaker", "line_id", "seed"])["score_nolevel"]
bt = base[base.audio_s <= a.max_audio]


def med(t, col, nd=3):
    col = {"sim_centroid": "sim_to_centroid"}.get(col, col) if "sim_to_centroid" in t else col
    return round(float(t[col].median()), nd) if col in t and t[col].notna().any() else None


print(
    f"unsteered (k=0): max audio {base.audio_s.max():.1f} s, n={len(base)}, terminated {len(bt)}/{len(base)}, median effort {bt.score_nolevel.median():.2f}, "
    f"p90 {bt.score_nolevel.quantile(0.9):.2f}, words/s {bt.words_per_s.median():.2f}, "
    f"similarity {med(bt, 'sim_centroid')}, WER {med(bt, 'wer')}  (similarity and WER need Step A, then Step B again)\n"
)


def capped_flags(g, factor):
    """A clip is capped when it reached the token limit qwen_gen.py computed from the word count."""
    words = g["text"].astype(str).str.split().str.len()
    limit = ((12.5 * factor * words / 2.4).astype(int) + 8).clip(lower=30)
    return (g["audio_s"] >= (limit - 2) / 12.5 - 0.01).to_numpy()


rows = []
for d in sorted(list(root.glob("dev_L*")) + list(root.glob("dev2_L*"))):
    f = d / "gap_clips.csv"
    if not f.exists():
        print("missing Step B output for", d.name)
        continue
    g = pd.read_csv(f)
    g["base"] = [b.get((s, l, sd)) for s, l, sd in zip(g.speaker, g.line_id, g.seed)]
    g["delta"] = g.score_nolevel - g["base"]
    bounded = d.name.endswith("_b")
    g["capped"] = capped_flags(g, a.bound_factor) if bounded else False
    for k, h in g.groupby("exaggeration"):
        t = h[~h.capped] if bounded else h[h.audio_s <= a.max_audio]
        c = h[h.capped]
        rows.append(
            {
                "config": d.name.replace("dev_", "").replace("dev2_", "2:"),
                "k": k,
                "terminated": f"{len(t)}/{len(h)}",
                "effort_med_term": round(t.score_nolevel.median(), 2) if len(t) else None,
                "effort_p90_term": round(t.score_nolevel.quantile(0.9), 2) if len(t) else None,
                "delta_med_term": round(t.delta.median(), 2) if len(t) else None,
                "share_up_term": round((t.delta > 0).mean(), 2) if len(t) else None,
                "words_per_s_term": round(t.words_per_s.median(), 2) if len(t) else None,
                "sim_term": med(t, "sim_centroid"),
                "wer_term": med(t, "wer"),
                "capped": f"{len(c)}/{len(h)}" if bounded else "-",
                "sim_capped": med(c, "sim_centroid") if bounded and len(c) else None,
                "wer_capped": med(c, "wer") if bounded and len(c) else None,
            }
        )
print(pd.DataFrame(rows).to_string(index=False))
