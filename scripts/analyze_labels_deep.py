#!/usr/bin/env python3
"""Second-stage analysis of the listening labels (aggregates only).

Run from the repo root in the project env:
    python scripts/analyze_labels_deep.py

Answers three questions left open by analyze_labels.py:
  1. Are problem rates driven by line length / intent (calm lines are longer)?
  2. Which simple rule best catches duplicated / unintelligible clips (threshold sweeps)?
  3. Among clips judged ok with real-loud effort, how high is speaker similarity?

CAUTION: thresholds are tuned on the same 150 labeled clips, so precision and recall are
optimistic. Validate any chosen rule on freshly generated clips (new seeds) before using it.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path("data_local/baselines/chatterbox/cfg0.50")
REAL_LOUD = 7.15
DEFAULT_SIM = 0.936  # median similarity at exaggeration 0.5
REAL_ANGER_SIM = 0.948


def auc(score, y):
    y = np.asarray(y, bool)
    s = pd.Series(np.asarray(score, float))
    ok = s.notna().to_numpy()
    s, y = s[ok], y[ok]
    n1, n0 = int(y.sum()), int((~y).sum())
    if n1 < 3 or n0 < 3:
        return None
    r = s.rank().to_numpy()
    return (r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def sweep(title, flags, truth):
    print(f"-- {title}  (positives: {int(truth.sum())}/{len(truth)})")
    rows = []
    for name, f in flags:
        tp = int((f & truth).sum())
        n = int(f.sum())
        rows.append({"rule": name, "flagged": n, "correct": tp,
                     "precision": round(tp / n, 2) if n else None,
                     "recall": round(tp / truth.sum(), 2) if truth.sum() else None})
    print(pd.DataFrame(rows).to_string(index=False), "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clips", default=str(BASE / "gap_clips.csv"))
    ap.add_argument("--labels", default=str(BASE / "listening_labels_ex1.60.csv"))
    ap.add_argument("--plausibility", default="configs/line_plausibility.csv",
                    help="optional: your 1-3 rating of how plausible a loud delivery is per calm line")
    a = ap.parse_args()
    pd.set_option("display.width", 200)

    c = pd.read_csv(a.clips)
    lab = pd.read_csv(a.labels)[["job_id", "labels"]]
    d = lab.merge(c, on="job_id", how="left", validate="one_to_one")
    d["n_words"] = d["text"].str.split().str.len()
    d["dup_ratio"] = d["hypothesis"].fillna("").str.split().str.len() / d["n_words"]
    for L in ["ok", "calm_when_loud", "loud_when_calm", "duplicated", "unintelligible"]:
        d[L] = d["labels"].str.split("|").apply(lambda xs, L=L: L in xs)
    d["speech_problem"] = d["duplicated"] | d["unintelligible"]
    d["group"] = np.where(d["intent"] == "calm", "calm lines", "shout+urgent lines")
    d["len_bin"] = pd.cut(d["n_words"], [0, 4, 9, 100], labels=["1-4 words", "5-9 words", "10+ words"])
    print(f"{len(d)} labeled clips\n")

    print("== 1. Problem rates by line length and intent group (% of clips) ==")
    g = d.groupby(["group", "len_bin"], observed=True)
    t = g.agg(n=("ok", "size"), ok=("ok", "mean"), unintelligible=("unintelligible", "mean"),
              duplicated=("duplicated", "mean"), loud_when_calm=("loud_when_calm", "mean"),
              words_per_s=("words_per_s", "median"))
    for k in ["ok", "unintelligible", "duplicated", "loud_when_calm"]:
        t[k] = (100 * t[k]).round(0)
    print(t.round(2).to_string(), "\n")

    print("== 2. Detector AUC within intent group (0.5 = none; <0.5 means LOW metric goes with label) ==")
    metrics = {"effort": "score_nolevel", "wer": "wer", "dup_ratio": "dup_ratio",
               "words_per_s": "words_per_s", "audio_s": "audio_s", "sim": "sim_to_centroid"}
    rows = {}
    for grp, sub in d.groupby("group"):
        for L in ["unintelligible", "duplicated"]:
            rows[f"{grp} / {L} (n+={int(sub[L].sum())}/{len(sub)})"] = {
                k: auc(sub[v], sub[L]) for k, v in metrics.items() if v in sub.columns}
    print(pd.DataFrame(rows).round(2).to_string(), "\n")

    print("== 3. Threshold sweeps (all clips; optimistic, tuned on this data) ==")
    sweep("duplicated: words heard / words in line >= t",
          [(f"dup_ratio >= {t}", d["dup_ratio"] >= t) for t in (1.5, 1.3, 1.2, 1.15, 1.1)], d["duplicated"])
    sweep("unintelligible: Whisper WER > t",
          [(f"wer > {t}", d["wer"] > t) for t in (0.5, 0.3, 0.2, 0.1, 0.0)], d["unintelligible"])
    sweep("unintelligible: words per second < t",
          [(f"words_per_s < {t}", d["words_per_s"] < t) for t in (2.8, 2.5, 2.3, 2.0)], d["unintelligible"])
    combo = (d["wer"] > 0.1) | (d["dup_ratio"] >= 1.15) | (d["words_per_s"] < 2.3)
    sweep("duplicated OR unintelligible: combined rule",
          [("wer > 0.1 OR dup_ratio >= 1.15 OR words_per_s < 2.3", combo),
           ("wer > 0.1 OR dup_ratio >= 1.15", (d["wer"] > 0.1) | (d["dup_ratio"] >= 1.15)),
           ("words_per_s < 2.3", d["words_per_s"] < 2.3)], d["speech_problem"])

    print("== 4. Similarity headroom among clips labeled ok with effort >= real loud ==")
    hi = d[d["ok"] & (d["score_nolevel"] >= REAL_LOUD)]
    sim = hi["sim_to_centroid"]
    print(f"n = {len(hi)} (of {len(d)}); similarity min / 25% / median / 75% / 90% / max: "
          + " / ".join(f"{sim.quantile(q):.3f}" for q in (0, .25, .5, .75, .9, 1)))
    print(f"share with similarity >= {DEFAULT_SIM} (exaggeration 0.5 median): {(sim >= DEFAULT_SIM).mean():.2f}; "
          f">= {REAL_ANGER_SIM} (real anger): {(sim >= REAL_ANGER_SIM).mean():.2f}")
    print("count by speaker:", hi.groupby("speaker").size().to_dict())
    print("by intent group:", hi.groupby("group").size().to_dict())

    pl_path = Path(a.plausibility)
    if pl_path.exists():
        pl = pd.read_csv(pl_path)[["line_id", "loud_plausible"]].dropna()
        e = d.merge(pl, on="line_id", how="inner")
        print("\n== 5. Calm lines by how plausible a loud delivery is (1 absurd, 2 odd, 3 plausible) ==")
        if e.empty:
            print("no rated lines matched")
        else:
            t = e.groupby("loud_plausible").agg(
                lines=("line_id", "nunique"), clips=("ok", "size"), ok=("ok", "mean"),
                loud_when_calm=("loud_when_calm", "mean"), unintelligible=("unintelligible", "mean"),
                duplicated=("duplicated", "mean"))
            for k in ["ok", "loud_when_calm", "unintelligible", "duplicated"]:
                t[k] = (100 * t[k]).round(0)
            print(t.to_string())
            print("If loud_when_calm stays high even where loud is plausible, the problem is delivery "
                  "quality, not just a mismatch with the text.")
    else:
        print(f"\n(no plausibility file at {pl_path}; fill in line_plausibility.csv to enable section 5)")


if __name__ == "__main__":
    main()