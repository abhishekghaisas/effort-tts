#!/usr/bin/env python3
"""Compare two scored systems (e.g. stock Chatterbox vs the effort fine-tune). Aggregates only.

Run from the repo root (project env) after Step A and Step B have been run for BOTH roots:
    python scripts/compare_systems.py --a data_local/baselines/chatterbox/cfg0.50 \
                                      --b data_local/baselines/effort_ft/cfg0.50 \
                                      --name-a stock --name-b effort_ft

Needs gap_clips.csv in each root. The speech-problem flag is the tentative rule from the label analysis
(WER > 0.1 or duplication ratio >= 1.15 or words/s < 2.3); it over-flags some lines and is only a proxy.
"""

import argparse
from pathlib import Path

import pandas as pd

REAL_LOUD_DEFAULT = 7.15
EFFORT_BINS = [-100, -5, 0, 3, 6, 9, 100]


def load(root, name):
    d = pd.read_csv(Path(root) / "gap_clips.csv")
    d["system"] = name
    n_words = d["text"].str.split().str.len()
    d["dup_ratio"] = d["hypothesis"].fillna("").str.split().str.len() / n_words
    d["flag"] = (d["wer"] > 0.1) | (d["dup_ratio"] >= 1.15) | (d["words_per_s"] < 2.3)
    return d


def by_setting(d):
    g = d.groupby("exaggeration").agg(
        n=("job_id", "size"),
        effort=("score_nolevel", "median"),
        effort_vs_default=("delta_nolevel", "median"),
        sim=("sim_to_centroid", "median"),
        wer_median=("wer", "median"),
        flag_rate=("flag", "mean"),
        words_per_s=("words_per_s", "median"),
    )
    return g.round(3)


def seed_noise(d, cells=None):
    """Mean SD of effort across seeds within each speaker x line cell, per setting.
    cells: optional {setting: set of (speaker, line)} to restrict to."""
    rows = {}
    for ex, g in d.groupby("exaggeration"):
        if g["seed"].nunique() >= 2:
            if cells is not None:
                keep = cells.get(ex)
                if not keep:
                    continue
                g = g[[(s, l) in keep for s, l in zip(g["speaker"], g["line_id"])]]
            sd = g.groupby(["speaker", "line_id"])["score_nolevel"].std(ddof=1).mean()
            rows[ex] = (
                (round(float(sd), 2), g.groupby(["speaker", "line_id"]).ngroups)
                if cells is not None
                else round(float(sd), 2)
            )
    return rows


def multi_seed_cells(d):
    """{setting: set of (speaker, line)} with at least 2 seeds."""
    out = {}
    for ex, g in d.groupby("exaggeration"):
        c = g.groupby(["speaker", "line_id"])["seed"].nunique()
        out[ex] = set(c[c >= 2].index)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True)
    ap.add_argument("--b", required=True)
    ap.add_argument("--name-a", default="A")
    ap.add_argument("--name-b", default="B")
    ap.add_argument("--real-loud", type=float, default=None)
    ap.add_argument(
        "--main-seed",
        type=int,
        default=0,
        help="sections 1-4 and 6 use only this seed (balanced across speakers); section 5 uses all seeds; -1 = all",
    )
    a = ap.parse_args()
    pd.set_option("display.width", 200)

    all_sys = {a.name_a: load(a.a, a.name_a), a.name_b: load(a.b, a.name_b)}
    systems = (
        all_sys
        if a.main_seed < 0
        else {n: d[d["seed"] == a.main_seed].copy() for n, d in all_sys.items()}
    )
    print(
        {
            n: f"{len(d)} of {len(all_sys[n])} clips used for sections 1-4 and 6"
            for n, d in systems.items()
        }
    )
    loud = a.real_loud
    if loud is None:
        try:
            from efforttts.eval.score_baseline import ears_context

            loud = ears_context(set(systems[a.name_a].speaker)).get("real loud", REAL_LOUD_DEFAULT)
        except Exception:
            loud = REAL_LOUD_DEFAULT
    print(f"real-loud effort reference: {loud:.2f}\n")

    print("== 1. By setting (exaggeration or scalar); medians ==")
    for name, d in systems.items():
        print(f"-- {name}\n{by_setting(d).to_string()}\n")

    print("== 2. Does effort follow the setting? (Spearman, setting vs effort) ==")
    for name, d in systems.items():
        per_spk = {
            s: round(float(g["exaggeration"].corr(g["score_nolevel"], method="spearman")), 2)
            for s, g in d.groupby("speaker")
        }
        print(
            f"{name}: overall {d['exaggeration'].corr(d['score_nolevel'], method='spearman'):.2f}; by speaker {per_spk}"
        )

    print(
        "\n== 3. Identity at matched achieved effort (median similarity to speaker centroid; n) =="
    )
    tab = {}
    for name, d in systems.items():
        b = pd.cut(d["score_nolevel"], EFFORT_BINS)
        g = d.groupby(b, observed=False)["sim_to_centroid"].agg(["median", "size"])
        tab[name] = g["median"].round(3).astype(str) + " (n=" + g["size"].astype(str) + ")"
    print(pd.DataFrame(tab).to_string())

    print(f"\n== 4. Near real-loud effort (within 1.5 of {loud:.2f}) ==")
    rows = {}
    for name, d in systems.items():
        n = d[(d["score_nolevel"] - loud).abs() <= 1.5]
        rows[name] = {
            "clips": len(n),
            "similarity": round(float(n["sim_to_centroid"].median()), 3),
            "wer_median": round(float(n["wer"].median()), 3),
            "speech_problem_rate": round(float(n["flag"].mean()), 2),
            "words_per_s": round(float(n["words_per_s"].median()), 2),
        }
    print(pd.DataFrame(rows).to_string())

    print("\n== 5. Seed noise: mean SD of effort across seeds for the same speaker and line ==")
    for name, d in all_sys.items():
        print(f"{name}: {seed_noise(d) or 'only one seed per setting (generate more seeds)'}")
    ca, cb = multi_seed_cells(all_sys[a.name_a]), multi_seed_cells(all_sys[a.name_b])
    matched = {ex: ca[ex] & cb[ex] for ex in set(ca) & set(cb) if ca[ex] & cb[ex]}
    print(
        "matched cells (same speaker and line, 2+ seeds in both systems), SD and number of cells:"
    )
    if matched:
        for name, d in all_sys.items():
            print(f"  {name}: {seed_noise(d, matched)}")
    else:
        print("  none yet")

    print(
        "\n== 6. Pilot criteria (proposed): similarity >= 0.90 near real loud, seed SD <= 2.0, "
        "problem rate <= 30%, monotone response =="
    )
    nb = systems[a.name_b]
    near = nb[(nb["score_nolevel"] - loud).abs() <= 1.5]
    sn = seed_noise(all_sys[a.name_b])
    mn = {n: seed_noise(d, matched).get(1.6) for n, d in all_sys.items()} if matched else {}
    print(
        f"{a.name_b}: similarity near loud {near['sim_to_centroid'].median():.3f} (baseline's is in section 4); "
        f"problem rate {near['flag'].mean():.2f}; seed SD at the highest setting "
        f"{sn.get(max(sn)) if sn else 'n/a'}; matched-cell seed SD at 1.6 (SD, cells) {mn or 'n/a'}; Spearman {nb['exaggeration'].corr(nb['score_nolevel'], method='spearman'):.2f}"
    )


if __name__ == "__main__":
    main()
