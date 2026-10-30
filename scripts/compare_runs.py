#!/usr/bin/env python3
"""Compare several scored runs side by side on the same speakers and lines. Aggregates only.

Run from the repo root (project env) after Step A and Step B for every root:
    python scripts/compare_runs.py \
        --root stock=data_local/baselines/chatterbox/cfg0.50 \
        --root v0=data_local/baselines/effort_ft/cfg0.50 \
        --root v0_extrap=data_local/baselines/effort_ft_extrap/cfg0.50 \
        --root v1=data_local/baselines/effort_v1/cfg0.50 \
        --speakers p023 p045 p056 --line-ids S02 S03 S05 S08 U01 U05 U08 C02 C06 C09

Each root needs gap_clips.csv (from score_baseline). Only seed 0 is used unless --seed is changed.
"flag" is the tentative speech-problem rule (WER > 0.1 or duplication ratio >= 1.15 or words/s < 2.3).
"""

import argparse
from pathlib import Path

import pandas as pd

BINS = [-100, -5, 0, 3, 6, 100]


def load(path, name, a):
    d = pd.read_csv(Path(path) / "gap_clips.csv")
    d["system"] = name
    n_words = d["text"].str.split().str.len()
    d["dup_ratio"] = d["hypothesis"].fillna("").str.split().str.len() / n_words
    d["flag"] = (d["wer"] > 0.1) | (d["dup_ratio"] >= 1.15) | (d["words_per_s"] < 2.3)
    if a.seed >= 0:
        d = d[d["seed"] == a.seed]
    if a.speakers:
        d = d[d["speaker"].isin(a.speakers)]
    if a.line_ids:
        d = d[d["line_id"].isin(a.line_ids)]
    return d.copy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", action="append", required=True, help="NAME=PATH (repeat)")
    ap.add_argument("--speakers", nargs="*", default=None)
    ap.add_argument("--line-ids", nargs="*", default=None)
    ap.add_argument("--seed", type=int, default=0, help="-1 = all seeds")
    ap.add_argument("--real-loud", type=float, default=7.15)
    a = ap.parse_args()
    pd.set_option("display.width", 220)

    runs = {}
    for r in a.root:
        name, path = r.split("=", 1)
        runs[name] = load(path, name, a)
    print({n: len(d) for n, d in runs.items()}, "clips after filtering\n")

    print("== 1. By setting (medians) ==")
    for n, d in runs.items():
        g = d.groupby("exaggeration").agg(
            n=("job_id", "size"),
            effort=("score_nolevel", "median"),
            effort_p90=("score_nolevel", lambda s: s.quantile(0.9)),
            sim=("sim_to_centroid", "median"),
            wer_median=("wer", "median"),
            flag_rate=("flag", "mean"),
            words_per_s=("words_per_s", "median"),
        )
        print(f"-- {n}\n{g.round(3).to_string()}\n")

    print("== 2. Summary per run ==")
    rows = {}
    for n, d in runs.items():
        med = d.groupby("exaggeration")["score_nolevel"].median()
        hi = d[d["score_nolevel"] >= 3]
        up = d[d["score_nolevel"] >= 0]
        rows[n] = {
            "settings": f"{med.index.min():g}-{med.index.max():g}",
            "effort_range": round(float(med.iloc[-1] - med.iloc[0]), 2),
            "max_setting_median": round(float(med.iloc[-1]), 2),
            "spearman": round(
                float(d["exaggeration"].corr(d["score_nolevel"], method="spearman")), 2
            ),
            "share_effort>=3": round(len(hi) / len(d), 3),
            "n_effort>=3": len(hi),
            "sim_median_effort>=0": round(float(up["sim_to_centroid"].median()), 3)
            if len(up)
            else None,
            "sim_median_all": round(float(d["sim_to_centroid"].median()), 3),
            "flag_rate_all": round(float(d["flag"].mean()), 3),
        }
    print(pd.DataFrame(rows).to_string())
    print(f"(real regular -5.43, real loud {a.real_loud})\n")

    print("== 3. Identity at matched achieved effort (median similarity; n) ==")
    tab = {}
    for n, d in runs.items():
        b = pd.cut(d["score_nolevel"], BINS)
        g = d.groupby(b, observed=False)["sim_to_centroid"].agg(["median", "size"])
        tab[n] = g["median"].round(3).astype(str) + " (n=" + g["size"].astype(str) + ")"
    print(pd.DataFrame(tab).to_string())

    print("\n== 4. By intent: median effort at each run's highest setting ==")
    out = {}
    for n, d in runs.items():
        top = d[d["exaggeration"] == d["exaggeration"].max()]
        out[n] = top.groupby("intent")["score_nolevel"].median().round(2)
    print(pd.DataFrame(out).to_string())


if __name__ == "__main__":
    main()
