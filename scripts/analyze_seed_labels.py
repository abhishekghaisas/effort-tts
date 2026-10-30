#!/usr/bin/env python3
"""Check the automatic speech-problem filter against human labels on fresh clips.

Run from the repo root after labeling the new seeds:
    python scripts/analyze_seed_labels.py

Inputs (in data_local/baselines/chatterbox/cfg0.50/):
    gap_clips.csv                       scores for every clip
    listening_labels_ex1.60.csv         your seed-0 labels (150 clips)
    listening_labels_seeds_ex1.60.csv   your labels for the new seeds (100 clips)
    listening_labels_retest_ex1.60.csv  optional: relabeled seed-0 clips (test-retest)

Prints aggregates only. The filter is: WER > 0.1 OR duplication ratio >= 1.15 OR words/s < 2.3.
"Human speech problem" = you labeled the clip duplicated or unintelligible.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path("data_local/baselines/chatterbox/cfg0.50")
RULES = {
    "wer > 0.1": lambda d: d["wer"] > 0.1,
    "dup_ratio >= 1.15": lambda d: d["dup_ratio"] >= 1.15,
    "words_per_s < 2.3": lambda d: d["words_per_s"] < 2.3,
}


def load(path, batch):
    d = pd.read_csv(path)[["job_id", "labels"]]
    d["batch"] = batch
    return d


def clean(row):
    """Drop label combinations that cannot apply to the line's intent (probable key slips)."""
    labs = set(row["labels"].split("|"))
    removed = 0
    if row["intent"] != "calm" and "loud_when_calm" in labs:
        labs.discard("loud_when_calm")
        removed += 1
    if row["intent"] == "calm" and "calm_when_loud" in labs:
        labs.discard("calm_when_loud")
        removed += 1
    if len(labs) > 1:
        labs.discard("ok")
    return pd.Series({"labset": labs or {"ok"}, "removed": removed})


def kappa(a, b):
    a, b = np.asarray(a, bool), np.asarray(b, bool)
    po = (a == b).mean()
    pe = a.mean() * b.mean() + (1 - a.mean()) * (1 - b.mean())
    return (po - pe) / (1 - pe) if pe < 1 else float("nan")


def prf(flag, truth):
    tp = int((flag & truth).sum())
    fp = int((flag & ~truth).sum())
    fn = int((~flag & truth).sum())
    return (
        tp,
        fp,
        fn,
        (tp / (tp + fp) if tp + fp else float("nan")),
        (tp / (tp + fn) if tp + fn else float("nan")),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clips", default=str(BASE / "gap_clips.csv"))
    ap.add_argument("--seed0", default=str(BASE / "listening_labels_ex1.60.csv"))
    ap.add_argument("--new", default=str(BASE / "listening_labels_seeds_ex1.60.csv"))
    ap.add_argument("--retest", default=str(BASE / "listening_labels_retest_ex1.60.csv"))
    a = ap.parse_args()
    pd.set_option("display.width", 200)

    c = pd.read_csv(a.clips)
    c["dup_ratio"] = (
        c["hypothesis"].fillna("").str.split().str.len() / c["text"].str.split().str.len()
    )
    c["flag"] = (
        RULES["wer > 0.1"](c) | RULES["dup_ratio >= 1.15"](c) | RULES["words_per_s < 2.3"](c)
    )
    lab = pd.concat([load(a.seed0, "seed0"), load(a.new, "new")])
    d = lab.merge(c, on="job_id", how="left", validate="one_to_one")
    d[["labset", "removed"]] = d.apply(clean, axis=1)
    d["sp"] = d["labset"].apply(lambda s: bool(s & {"duplicated", "unintelligible"}))
    d["sounds_calm"] = d["labset"].apply(lambda s: "calm_when_loud" in s)
    print(
        f"{len(d)} labeled clips ({(d.batch == 'seed0').sum()} seed 0, {(d.batch == 'new').sum()} new seeds); "
        f"{int(d['removed'].sum())} impossible label flags removed as probable key slips "
        f"({int((d['removed'] > 0).sum())} clips)\n"
    )
    new = d[d.batch == "new"]
    s0 = d[d.batch == "seed0"]

    print("== 1. Per line: human speech-problem rate vs filter flag rate ==")
    focus = sorted(new["line_id"].unique())
    rows = []
    for ln in focus:
        n_, s_ = new[new.line_id == ln], s0[s0.line_id == ln]
        rows.append(
            {
                "line": ln,
                "intent": n_["intent"].iloc[0],
                "words": len(n_["text"].iloc[0].split()),
                "human_seed0": round(s_["sp"].mean(), 2) if len(s_) else None,
                "n0": len(s_),
                "human_new": round(n_["sp"].mean(), 2),
                "n_new": len(n_),
                "filter_new": round(n_["flag"].mean(), 2),
                "ok_new": round((n_["labset"].apply(lambda s: s == {"ok"})).mean(), 2),
                "sounds_calm_new": int(n_["sounds_calm"].sum()),
            }
        )
    print(
        pd.DataFrame(rows)
        .sort_values(["intent", "human_new"], ascending=[True, False])
        .to_string(index=False),
        "\n",
    )

    print(
        "== 2. Filter vs human speech problem (precision = flagged clips that were real problems) =="
    )
    out = []
    for name, sub in [
        ("new seeds (fresh clips)", new),
        ("seed 0 (where it was tuned)", s0[s0.line_id.isin(focus)]),
        ("seed 0, all 150 clips", s0),
    ]:
        for rule, f in [("combined filter", sub["flag"])] + [(k, v(sub)) for k, v in RULES.items()]:
            tp, fp, fn, p, r = prf(f, sub["sp"])
            out.append(
                {
                    "clips": name,
                    "rule": rule,
                    "flagged": int(f.sum()),
                    "correct": tp,
                    "missed": fn,
                    "precision": round(p, 2),
                    "recall": round(r, 2),
                }
            )
    print(pd.DataFrame(out).to_string(index=False))
    print(
        f"base rate of human speech problems: new {new['sp'].mean():.2f}, seed 0 (these lines) "
        f"{s0[s0.line_id.isin(focus)]['sp'].mean():.2f}\n"
    )

    print("== 3. False positives on new seeds: which rule fired (clips) ==")
    fp = new[new["flag"] & ~new["sp"]]
    if len(fp):
        t = pd.DataFrame({k: v(fp).groupby(fp["line_id"]).sum() for k, v in RULES.items()})
        t["false_positives"] = fp.groupby("line_id").size()
        print(t.fillna(0).astype(int).to_string(), "\n")
    else:
        print("none\n")

    print("== 4. Does the effort estimator track 'sounds calm' on shout and urgent lines? ==")
    loud = d[d["intent"].isin(["shout", "urgent"])]
    for nm, grp in [("all labeled", loud), ("new seeds only", loud[loud.batch == "new"])]:
        pos, neg = grp[grp["sounds_calm"]], grp[~grp["sounds_calm"]]
        if len(pos) >= 3:
            ranks = grp["score_nolevel"].rank()
            auc = 1 - (ranks[grp["sounds_calm"]].sum() - len(pos) * (len(pos) + 1) / 2) / (
                len(pos) * len(neg)
            )
            print(
                f"{nm}: sounds-calm clips {len(pos)} of {len(grp)}; median effort {pos['score_nolevel'].median():.1f} "
                f"vs {neg['score_nolevel'].median():.1f}; AUC for low effort detecting 'sounds calm' = {auc:.2f}"
            )
        else:
            print(f"{nm}: only {len(pos)} sounds-calm clips, too few")
    print()

    if Path(a.retest).exists():
        print("== 5. Test-retest: your relabeling of seed-0 clips against your original labels ==")
        rt = load(a.retest, "retest").merge(c[["job_id", "intent"]], on="job_id")
        rt[["labset", "removed"]] = rt.apply(clean, axis=1)
        rt["sp"] = rt["labset"].apply(lambda s: bool(s & {"duplicated", "unintelligible"}))
        m = rt.merge(s0[["job_id", "labset", "sp"]], on="job_id", suffixes=("_re", "_orig"))
        same = (m["labset_re"] == m["labset_orig"]).mean()
        print(
            f"{len(m)} clips relabeled; identical label set {same:.2f}; speech-problem agreement "
            f"{(m['sp_re'] == m['sp_orig']).mean():.2f}; kappa {kappa(m['sp_re'], m['sp_orig']):.2f} "
            f"(speech-problem rate original {m['sp_orig'].mean():.2f}, retest {m['sp_re'].mean():.2f})"
        )
    else:
        print("(no retest file; label_clips.py --max-seed 0 would create one, see README)")


if __name__ == "__main__":
    main()
