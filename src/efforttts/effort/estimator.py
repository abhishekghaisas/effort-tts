"""Effort estimator v0.

Trains a small linear model to separate LOUD from REGULAR reading, then reads
its decision value as a continuous effort score. The point is NOT the
loud-vs-regular accuracy (level alone nearly solves it); it is the ablation:

  level     : level only                       (the confound)
  spectral  : spectral shape + voice quality   (no level, no pitch)
  nolevel   : spectral + pitch + voicing       (everything except level)
  all       : nolevel + level

and whether the level-free model still ranks styles and emotions sensibly on
data it never trained on (whisper, highpitch, emotions, yelling, screaming).

Two normalization modes:
  relative : features minus the speaker's own median on FREEFORM speech
             (a calm reference clip, matching the real use case)
  absolute : raw features (speaker- and microphone-dependent)

Training uses only split == "train" speakers; headline metrics use split ==
"val" speakers. The "test" split is dropped on load and never touched.

    python -m efforttts.effort.estimator [--mode relative] [--C 1.0]

Writes processed_dir()/estimator/{effort_scores.csv, estimator.joblib}.
"""
from __future__ import annotations

import argparse

import numpy as np

from efforttts import config
from efforttts.effort.features import FEATURE_GROUPS

F0_ST = "f0_median_st"  # 12*log2(F0 in Hz): semitones, so relative mode gives pitch ratios
SPECTRAL = [f for f in FEATURE_GROUPS["spectral"]]
PITCH = [F0_ST, "f0_range_st", "f0_std_st"]
LEVEL = ["level_active_db"]
FEATURE_SETS = {
    "level": LEVEL,
    "spectral": SPECTRAL,
    "nolevel": SPECTRAL + PITCH + ["voiced_frac"],
    "all": LEVEL + SPECTRAL + PITCH + ["voiced_frac"],
}
ALL_NAMES = sorted({f for fs in FEATURE_SETS.values() for f in fs})
REF_CATEGORY = "freeform"  # calm natural monologue, used as each speaker's baseline
PRIMARY_SET = "nolevel"


def load_features(path=None):
    import pandas as pd

    df = pd.read_csv(path or config.processed_dir() / "features.csv")
    df[F0_ST] = 12 * np.log2(df["f0_median_hz"])
    n_test = int((df.split == "test").sum())
    if n_test:
        print(f"[note] dropping {n_test} files from held-out TEST speakers (never used here)")
    return df[df.split != "test"].reset_index(drop=True)


def make_relative(df, feats=None):
    """Subtract each speaker's median on REF_CATEGORY files from every feature."""
    feats = feats or ALL_NAMES
    base = df[df.category == REF_CATEGORY].groupby("speaker")[feats].median()
    missing = set(df.speaker) - set(base.index)
    if missing:
        raise ValueError(f"No {REF_CATEGORY} reference files for speakers: {sorted(missing)}")
    out = df.copy()
    out[feats] = out[feats].to_numpy() - base.reindex(out.speaker)[feats].to_numpy()
    return out


def _pipeline(C: float):
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    return make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        LogisticRegression(C=C, class_weight="balanced", max_iter=2000),
    )


def _xy(df):
    d = df[(df.category == "reading") & df.label.isin(["regular", "loud"])]
    return d, (d.label == "loud").astype(int).to_numpy()


def pair_accuracy(d, score) -> tuple[float, int]:
    """Fraction of same-speaker, same-text pairs where loud scores above regular."""
    t = d.assign(score=np.asarray(score)).pivot_table(
        index=["speaker", "text_id"], columns="label", values="score")
    t = t.dropna(subset=["loud", "regular"])
    return float((t["loud"] > t["regular"]).mean()), len(t)


def run_ablation(df, C: float):
    import pandas as pd
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import GroupKFold, cross_val_predict

    rows, models, frames, scores = [], {}, {}, {}
    for mode in ("relative", "absolute"):
        d = make_relative(df) if mode == "relative" else df
        frames[mode] = d
        tr, ytr = _xy(d[d.split == "train"])
        va, yva = _xy(d[d.split == "val"])
        cv = GroupKFold(n_splits=min(4, tr.speaker.nunique()))
        for name, feats in FEATURE_SETS.items():
            oof = cross_val_predict(_pipeline(C), tr[feats], ytr, groups=tr.speaker, cv=cv,
                                    method="decision_function")
            model = _pipeline(C).fit(tr[feats], ytr)
            sv = model.decision_function(va[feats])
            acc, npairs = pair_accuracy(va, sv)
            rows.append({"mode": mode, "set": name, "n_feat": len(feats),
                         "cv_auc": roc_auc_score(ytr, oof),
                         "val_auc": roc_auc_score(yva, sv),
                         "val_pair_acc": acc, "n_pairs": npairs})
            models[(mode, name)] = model
            # honest scores for every reading regular/loud row: out-of-fold for train, held-out for val
            scores[(mode, name)] = pd.concat([pd.Series(oof, index=tr.index), pd.Series(sv, index=va.index)])
    return pd.DataFrame(rows), models, frames, scores


def level_matched_check(df, scores, mode: str = "relative", n_bins: int = 3):
    """Does the level-free model still rank loud above regular when the LEVEL gap
    between the two readings (same speaker, same words) is small?

    Pairs are split into bins by raw level difference. Scores are out-of-fold
    (train speakers) or held-out (val speakers), so nothing is scored in-sample.
    """
    import pandas as pd

    d, _ = _xy(df[df.split.isin(["train", "val"])])
    d = d.assign(s_nolevel=scores[(mode, "nolevel")].reindex(d.index),
                 s_level=scores[(mode, "level")].reindex(d.index))
    t = d.pivot_table(index=["speaker", "text_id"], columns="label",
                      values=["level_active_db", "s_nolevel", "s_level"]).dropna()
    gap = t[("level_active_db", "loud")] - t[("level_active_db", "regular")]
    bins = pd.qcut(gap, n_bins, duplicates="drop")
    out = pd.DataFrame({
        "bin": bins,
        "gap_db": gap,
        "nolevel_acc": t[("s_nolevel", "loud")] > t[("s_nolevel", "regular")],
        "level_acc": t[("s_level", "loud")] > t[("s_level", "regular")],
    }).groupby("bin", observed=True).agg(
        pairs=("gap_db", "size"), median_gap_db=("gap_db", "median"),
        nolevel_acc=("nolevel_acc", "mean"), level_acc=("level_acc", "mean"))
    out.index = [f"level gap tercile {i + 1}" for i in range(len(out))]
    return out


def _group_key(r) -> str:
    return f"{r['category']}/{r['label']}"


def print_coefficients(models, mode: str) -> None:
    import pandas as pd

    for name in ("nolevel", "all"):
        m = models[(mode, name)]
        coef = pd.Series(m[-1].coef_[0], index=FEATURE_SETS[name]).sort_values(key=np.abs, ascending=False)
        print(f"\n== Standardized coefficients: {name} ({mode}) ==")
        print(coef.round(2).to_string())


def print_ood(d, models, mode: str) -> None:
    import pandas as pd

    keys = d.apply(_group_key, axis=1)
    want = keys.str.startswith("reading/") | keys.str.startswith("emotion_freeform/") | keys.isin(
        ["nonspeech/cheering", "nonspeech/yelling", "nonspeech/screaming", "freeform/nan"])
    sub = d[want].copy()
    sub["group"] = keys[want]
    for name in ("nolevel", "all", "level"):
        sub[name] = models[(mode, name)].decision_function(sub[FEATURE_SETS[name]])
    t = sub.groupby("group").agg(n=("stem", "size"), nolevel=("nolevel", "median"),
                                 all=("all", "median"), level=("level", "median"))
    print(f"\n== Median effort score by group ({mode}); trained ONLY on reading regular vs loud ==")
    print("   (whisper, highpitch, emotions, nonspeech were never in training)")
    print(t.sort_values("nolevel", ascending=False).round(2).to_string())


def main() -> None:
    import joblib
    import pandas as pd

    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["relative", "absolute"], default="relative",
                    help="normalization for the saved scores and OOD tables")
    ap.add_argument("--C", type=float, default=1.0)
    args = ap.parse_args()

    df = load_features()
    print(f"Speakers: {df.groupby('split').speaker.nunique().to_dict()}")

    res, models, frames, scores = run_ablation(df, args.C)
    pd.set_option("display.width", 160)
    print("\n== Ablation: loud vs regular reading (AUC; pair acc = same speaker & text) ==")
    print("   cv_auc: grouped-by-speaker CV on train speakers; val_*: held-out val speakers")
    print(res.round(3).to_string(index=False))

    d = frames[args.mode]
    lm = level_matched_check(df, scores, args.mode)
    print(f"\n== Matched pairs split by how big the LEVEL gap is ({args.mode}) ==")
    print("   If nolevel_acc stays high where the level gap is small, spectral cues carry")
    print("   effort information beyond loudness. Scores are out-of-fold / held-out.")
    print(lm.round(3).to_string())
    print_coefficients(models, args.mode)
    print_ood(d, models, args.mode)

    # scores for every file, primary mode
    out = d[["speaker", "stem", "path", "category", "label", "text_id", "split"]].copy()
    for name in FEATURE_SETS:
        out[f"score_{name}"] = models[(args.mode, name)].decision_function(d[FEATURE_SETS[name]])
    outdir = config.processed_dir() / "estimator"
    outdir.mkdir(parents=True, exist_ok=True)
    out.to_csv(outdir / "effort_scores.csv", index=False)
    joblib.dump({"mode": args.mode, "ref_category": REF_CATEGORY, "feature_sets": FEATURE_SETS,
                 "models": {n: models[(args.mode, n)] for n in FEATURE_SETS}},
                outdir / "estimator.joblib")

    # clips to LISTEN to: highest-scoring speech outside the training styles
    cand = out[out.category.isin(["emotion_freeform", "emotion", "freeform"])]
    top = cand.sort_values(f"score_{PRIMARY_SET}", ascending=False).head(10)
    print(f"\n== Top 10 non-reading speech clips by score_{PRIMARY_SET} (listen to these) ==")
    print(top[["speaker", "stem", f"score_{PRIMARY_SET}"]].round(2).to_string(index=False))
    print(f"\nSaved: {outdir}")


if __name__ == "__main__":
    main()