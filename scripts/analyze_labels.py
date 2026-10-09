#!/usr/bin/env python3
"""Join listening labels with automatic scores and report aggregates only.

Run from the repo root in the project env:
    python scripts/analyze_labels.py

Reads gap_clips.csv and listening_labels_ex1.60.csv (written by label_clips.py).
Prints counts, metric medians per label, and how well each automatic metric detects each
label (AUC, 0.5 = no signal). No per-clip rows and no EARS data are printed.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path("data_local/baselines/chatterbox/cfg0.50")
LABELS = ["calm_when_loud", "loud_when_calm", "duplicated", "unintelligible"]
REAL_LOUD = 7.15  # real EARS loud, same estimator
METRICS = {
    "effort": "score_nolevel",
    "effort_spectral": "score_spectral",
    "wer": "wer",
    "dup_ratio": "dup_ratio",
    "sim_centroid": "sim_to_centroid",
    "words_per_s": "words_per_s",
    "audio_s": "audio_s",
    "cpp_db": "cpp_db",
}


def auc(score, y):
    """P(score of a positive > score of a negative); None if too few of either."""
    y = np.asarray(y, bool)
    s = pd.Series(np.asarray(score, float))
    ok = s.notna().to_numpy()
    s, y = s[ok], y[ok]
    n1, n0 = int(y.sum()), int((~y).sum())
    if n1 < 3 or n0 < 3:
        return None
    r = s.rank().to_numpy()
    return (r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clips", default=str(BASE / "gap_clips.csv"))
    ap.add_argument("--labels", default=str(BASE / "listening_labels_ex1.60.csv"))
    a = ap.parse_args()
    pd.set_option("display.width", 200)

    c = pd.read_csv(a.clips)
    lab = pd.read_csv(a.labels)[["job_id", "labels"]]
    d = lab.merge(c, on="job_id", how="left", validate="one_to_one")
    d["dup_ratio"] = (d["hypothesis"].fillna("").str.split().str.len()
                      / d["text"].str.split().str.len())
    for L in LABELS + ["ok"]:
        d[L] = d["labels"].str.split("|").apply(lambda xs, L=L: L in xs)
    d["any_problem"] = ~d["ok"]
    n = len(d)
    print(f"{n} labeled clips at exaggeration {d['exaggeration'].iloc[0]}\n")

    print("== Label counts (clips can have several) ==")
    cnt = pd.DataFrame({"n": d[["ok"] + LABELS].sum(), })
    cnt["pct"] = (100 * cnt["n"] / n).round(1)
    print(cnt.to_string(), f"\nany problem: {int(d['any_problem'].sum())} ({100 * d['any_problem'].mean():.1f}%)\n")

    print("== By intent (counts) ==")
    print(d.groupby("intent")[["ok"] + LABELS].sum().to_string(), "\n")
    print("== By speaker (counts) ==")
    print(d.groupby("speaker")[["ok"] + LABELS].sum().to_string(), "\n")
    pl = d.groupby("line_id")["any_problem"].sum().sort_values(ascending=False)
    print("== Lines with the most problem clips (of 5 speakers) ==")
    print(pl[pl > 0].head(10).to_string(), "\n")

    print("== Co-occurrence of problem labels ==")
    print(pd.DataFrame({x: {y: int((d[x] & d[y]).sum()) for y in LABELS} for x in LABELS}).to_string(), "\n")

    cols = {k: v for k, v in METRICS.items() if v in d.columns}
    print("== Metric medians: ok vs each label ==")
    med = {"ok": d.loc[d["ok"], list(cols.values())].median()}
    for L in LABELS:
        if d[L].sum():
            med[L] = d.loc[d[L], list(cols.values())].median()
    print(pd.DataFrame(med).rename(index={v: k for k, v in cols.items()}).round(3).to_string(), "\n")

    print("== Detector AUC (0.5 = no signal; >0.5 means higher metric goes with the label) ==")
    subsets = {
        "calm_when_loud": d["intent"].isin(["shout", "urgent"]),
        "loud_when_calm": d["intent"] == "calm",
        "duplicated": pd.Series(True, index=d.index),
        "unintelligible": pd.Series(True, index=d.index),
    }
    rows = {}
    for L, m in subsets.items():
        sub = d[m]
        rows[f"{L} (n+={int(sub[L].sum())}/{len(sub)})"] = {
            k: auc(sub[v], sub[L]) for k, v in cols.items()}
    print(pd.DataFrame(rows).round(2).to_string(), "\n")

    print("== Existing rules vs your labels ==")
    dupflag = d["dup_ratio"] >= 1.5
    print("dup_ratio >= 1.5 vs 'duplicated':")
    print(pd.crosstab(dupflag.rename("flag"), d["duplicated"].rename("label")).to_string())
    werflag = d["wer"] > 0.5
    print("WER > 0.5 vs 'unintelligible':")
    print(pd.crosstab(werflag.rename("flag"), d["unintelligible"].rename("label")).to_string(), "\n")

    print("== Usable clips at this setting ==")
    ok = d[d["ok"]]
    hi = ok[ok["score_nolevel"] >= REAL_LOUD]
    print(f"labeled ok: {len(ok)}/{n} ({100 * len(ok) / n:.0f}%); "
          f"ok and effort >= real loud ({REAL_LOUD}): {len(hi)} ({100 * len(hi) / n:.0f}%)")
    print("ok clips: median effort "
          f"{ok['score_nolevel'].median():.2f}, median similarity {ok['sim_to_centroid'].median():.3f}")
    if len(hi):
        print("ok & high effort: median effort "
              f"{hi['score_nolevel'].median():.2f}, median similarity {hi['sim_to_centroid'].median():.3f}")
    print("ok share by speaker:", d.groupby("speaker")["ok"].mean().round(2).to_dict())


if __name__ == "__main__":
    main()