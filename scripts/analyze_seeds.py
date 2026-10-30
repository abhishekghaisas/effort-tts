#!/usr/bin/env python3
"""Seed-to-seed stability and automatic best-of-N headroom at one exaggeration setting.

Run from the repo root after Step A and Step B have scored the new seeds:
    python scripts/analyze_seeds.py

Prints aggregates only. Questions it answers:
  1. How much of the effort score is seed noise versus a real line or speaker effect?
  2. Do speech problems (the combined automatic filter) repeat for the same line and speaker
     across seeds, or move around?
  3. With several draws per line and speaker, how often does at least one draw pass the
     automatic filter AND reach real-loud effort, and how similar is the best such draw?

The speech-problem filter is the tentative rule tuned on the first 150 labeled clips
(WER > 0.1 or duplication ratio >= 1.15 or words/s < 2.3). It is a proxy for the human
"ok" label, not a replacement for it.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path("data_local/baselines/chatterbox/cfg0.50")
REAL_LOUD = 7.15
DEFAULT_SIM = 0.936
REAL_ANGER_SIM = 0.948
FOCUS = ["C09", "C02", "C04", "C10", "S03", "S05"]


def eta2(df, col, key):
    """Share of variance explained by a grouping, and the share expected by chance."""
    m = df.groupby(key)[col].transform("mean")
    ss_t = ((df[col] - df[col].mean()) ** 2).sum()
    return ((m - df[col].mean()) ** 2).sum() / ss_t, (df[key].nunique() - 1) / (len(df) - 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clips", default=str(BASE / "gap_clips.csv"))
    ap.add_argument("--exaggeration", type=float, default=1.6)
    a = ap.parse_args()
    pd.set_option("display.width", 200)

    d = pd.read_csv(a.clips)
    d = d[d["exaggeration"].round(2) == round(a.exaggeration, 2)].copy()
    if "status" in d.columns:
        d = d[d["status"] == "ok"]
    d["dup_ratio"] = (
        d["hypothesis"].fillna("").str.split().str.len() / d["text"].str.split().str.len()
    )
    d["flag"] = (d["wer"] > 0.1) | (d["dup_ratio"] >= 1.15) | (d["words_per_s"] < 2.3)
    d["dup15"] = d["dup_ratio"] >= 1.5
    seeds = sorted(d["seed"].unique())
    cells = d.groupby(["speaker", "line_id"]).size()
    print(
        f"{len(d)} clips at exaggeration {a.exaggeration}; seeds {[int(x) for x in seeds]}; "
        f"{len(cells)} speaker x line cells with {cells.min()} to {cells.max()} draws each\n"
    )
    if len(seeds) < 2:
        raise SystemExit("Need at least 2 seeds. Run Step A and Step B on the new clips first.")

    print("== 1. Per-seed sanity check (these should look alike) ==")
    t = d.groupby("seed").agg(
        n=("flag", "size"),
        effort=("score_nolevel", "median"),
        wer_median=("wer", "median"),
        flag_rate=("flag", "mean"),
        dup15_rate=("dup15", "mean"),
        sim=("sim_to_centroid", "median"),
    )
    print(t.round(3).to_string(), "\n")

    print("== 2. Effort: seed noise versus line and speaker effects ==")
    g = d.groupby(["speaker", "line_id"])["score_nolevel"]
    within = g.var(ddof=1).mean()
    total = d["score_nolevel"].var(ddof=1)
    print(
        f"SD of effort across seeds for the same speaker and line: {np.sqrt(within):.2f} "
        f"(total SD {np.sqrt(total):.2f}; seed noise = {100 * within / total:.0f}% of total variance)"
    )
    for key in ["line_id", "speaker", "intent"]:
        e, chance = eta2(d, "score_nolevel", key)
        print(f"variance explained by {key}: {e:.2f} (about {chance:.2f} expected by chance)")
    ln = d.groupby("line_id")["score_nolevel"].agg(["mean", "std", "count"])
    ln["se"] = ln["std"] / np.sqrt(ln["count"])
    ln["intent"] = d.groupby("line_id")["intent"].first()
    ln = ln.sort_values("mean")
    print("\nLowest and highest mean effort by line (all speakers and seeds):")
    print(
        pd.concat([ln.head(5), ln.tail(5)])[["intent", "mean", "se", "count"]].round(2).to_string()
    )
    if "S03" in ln.index:
        print(f"S03 rank: {list(ln.index).index('S03') + 1} of {len(ln)} (1 = lowest effort)")
    print(
        "\nMean effort by intent:",
        d.groupby("intent")["score_nolevel"].mean().round(2).to_dict(),
        "\n",
    )

    print("== 3. Do speech problems repeat for the same line and speaker? ==")
    piv = d.pivot_table(
        index=["speaker", "line_id"], columns="seed", values="flag", aggfunc="first"
    ).dropna()
    piv = piv.astype(bool)
    base = float(piv.to_numpy().mean())
    rates = []
    for i in seeds:
        for j in seeds:
            if i != j and piv[i].sum():
                rates.append(piv.loc[piv[i], j].mean())
    rep = float(np.mean(rates)) if rates else float("nan")
    print(
        f"overall flag rate {base:.2f}; P(flagged in another seed | flagged in this seed) = {rep:.2f}; "
        f"persistence ratio {rep / base:.1f}x (1.0x = pure chance, higher = the same cells fail repeatedly)"
    )
    byline = d.groupby("line_id")["flag"].mean().sort_values(ascending=False)
    print("lines with the highest flagged share:", byline.head(8).round(2).to_dict())
    foc = d[d["line_id"].isin(FOCUS)].pivot_table(
        index="line_id", columns="seed", values="flag", aggfunc="sum"
    )
    foc.columns = [f"seed {c}" for c in foc.columns]
    print("\nFlagged clips among 5 speakers, focus lines:")
    print(foc.to_string(), "\n")
    print(
        "duplication ratio >= 1.5 (clips per line, top 5):",
        d[d["dup15"]].groupby("line_id").size().sort_values(ascending=False).head(5).to_dict(),
        "\n",
    )

    print(
        f"== 4. Automatic best-of-N: passes filter AND effort >= {REAL_LOUD} (proxy for human ok) =="
    )
    d["auto_ok"] = (~d["flag"]) & (d["score_nolevel"] >= REAL_LOUD)
    rows = []
    for n in range(1, len(seeds) + 1):
        sub = d[d["seed"].isin(seeds[:n])]
        best = (
            sub[sub["auto_ok"]]
            .sort_values("sim_to_centroid", ascending=False)
            .groupby(["speaker", "line_id"])
            .head(1)
        )
        sim = best["sim_to_centroid"]
        rows.append(
            {
                "draws": n,
                "cells_covered": f"{len(best)}/{len(cells)}",
                "coverage": round(len(best) / len(cells), 2),
                "median_sim": round(sim.median(), 3) if len(sim) else None,
                "share>=0.936": round((sim >= DEFAULT_SIM).mean(), 2) if len(sim) else None,
                f"share>={REAL_ANGER_SIM}": round((sim >= REAL_ANGER_SIM).mean(), 2)
                if len(sim)
                else None,
            }
        )
    print(pd.DataFrame(rows).to_string(index=False))
    cov = d.groupby(["speaker", "line_id"])["auto_ok"].any().groupby("speaker").mean().round(2)
    print("coverage with all draws, by speaker:", cov.to_dict())
    covi = (
        d.groupby(["speaker", "line_id", "intent"])["auto_ok"]
        .any()
        .groupby("intent")
        .mean()
        .round(2)
    )
    print("coverage with all draws, by intent:", covi.to_dict())
    print("(real loud similarity 0.967; exaggeration-0.5 median 0.936)")


if __name__ == "__main__":
    main()
