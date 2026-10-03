import sys

import numpy as np
import pandas as pd

from efforttts.effort import estimator as E
from efforttts.effort.features import ALL_FEATURES

BASE = dict(level_active_db=-37, alpha_ratio_db=-12, tilt_db_per_khz=-5, hammarberg_db=28,
            cpp_db=1.6, hnr_v_db=7, alpha_ratio_v_db=-12, tilt_v_db_per_khz=-5,
            f0_median_hz=190, f0_range_st=8, f0_std_st=2.5, voiced_frac=0.7, active_frac=0.9)
LOUD = dict(level_active_db=9, alpha_ratio_db=3.6, tilt_db_per_khz=0.8, hammarberg_db=-6.6,
            cpp_db=0.8, hnr_v_db=0.7, alpha_ratio_v_db=3.0, tilt_v_db_per_khz=0.8)
NOISE = dict(level_active_db=1.5, alpha_ratio_db=1.5, tilt_db_per_khz=0.5, hammarberg_db=2,
             cpp_db=0.3, hnr_v_db=1, alpha_ratio_v_db=1.5, tilt_v_db_per_khz=0.5,
             f0_median_hz=8, f0_range_st=1, f0_std_st=0.3, voiced_frac=0.05, active_frac=0.02)
SHARED = [f"rainbow_{i:02d}" for i in range(1, 9)] + ["sentences_01", "sentences_02"]
REG_ONLY = [f"sentences_{i:02d}" for i in range(15, 25)]
LOUD_ONLY = ["sentences_05", "sentences_06"]


def synth_features(seed=0):
    """17 speakers with LARGE per-speaker offsets (so absolute level is a poor cue)."""
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(1, 18):
        spk, split = f"p{i:03d}", "train" if i <= 12 else "val"
        off = {k: (rng.normal(0, 8) if k == "level_active_db" else rng.normal(0, 0.15 * abs(v)))
               for k, v in BASE.items()}

        def add(cat, label, text_id, shift=None, f0_mult=1.0):
            r = {"speaker": spk, "stem": f"{cat}_{label}_{text_id}_{len(rows)}", "path": "x",
                 "category": cat, "label": label, "text_id": text_id, "text": "", "split": split,
                 "duration_s": 5.0, "is_speech": True}
            for k in ALL_FEATURES:
                v = BASE[k] + off[k] + rng.normal(0, NOISE[k]) + (shift or {}).get(k, 0)
                r[k] = v * f0_mult if k == "f0_median_hz" else v
            rows.append(r)

        for j in range(6):
            add("freeform", np.nan, f"ff{j}")
        for t in SHARED + REG_ONLY:
            add("reading", "regular", t)
        for t in SHARED + LOUD_ONLY:
            add("reading", "loud", t, LOUD, 1.15)
        for j in range(3):
            add("reading", "whisper", f"w{j}", dict(level_active_db=-14, alpha_ratio_db=7, cpp_db=-0.7), 4.0)
            add("reading", "highpitch", f"h{j}", dict(level_active_db=7, alpha_ratio_db=2.5), 1.7)
        add("emotion_freeform", "anger", "a", {k: 0.8 * v for k, v in LOUD.items()}, 1.2)
        add("emotion_freeform", "neutral", "n")
        add("nonspeech", "screaming", "s", {k: 2.5 * v for k, v in LOUD.items()}, 2.5)
    df = pd.DataFrame(rows)
    df[E.F0_ST] = 12 * np.log2(df["f0_median_hz"])
    return df


def test_make_relative_centers_reference():
    df = synth_features()
    rel = E.make_relative(df)
    ref = rel[rel.category == "freeform"].groupby("speaker")[E.ALL_NAMES].median()
    assert np.allclose(ref.to_numpy(), 0.0, atol=1e-9)


def test_pair_accuracy():
    d = pd.DataFrame({"speaker": ["a", "a", "a", "a"], "text_id": ["t1", "t1", "t2", "t2"],
                      "label": ["loud", "regular", "loud", "regular"]})
    acc, n = E.pair_accuracy(d, [2.0, 1.0, 0.5, 1.0])
    assert (acc, n) == (0.5, 2)


def test_ablation_separates_level_from_spectral():
    df = synth_features()
    res, _, _, _ = E.run_ablation(df, C=1.0)
    g = res.set_index(["mode", "set"])
    # level-free model works under speaker-relative normalization
    assert g.loc[("relative", "nolevel"), "val_auc"] > 0.95
    # Use grouped CV over 12 speakers: 5 val speakers are too few to be a stable estimate.
    # Raw level is a poor cue across speakers with different gains; relative level is far better.
    assert g.loc[("absolute", "level"), "cv_auc"] < g.loc[("relative", "level"), "cv_auc"] - 0.05
    # The level-free absolute model still beats absolute level-only.
    assert g.loc[("absolute", "nolevel"), "cv_auc"] > g.loc[("absolute", "level"), "cv_auc"]


def test_end_to_end_main(tmp_path, monkeypatch):
    monkeypatch.setenv("EFFORT_DATA_DIR", str(tmp_path))
    from efforttts import config

    synth_features().drop(columns=[E.F0_ST]).to_csv(config.processed_dir() / "features.csv", index=False)
    monkeypatch.setattr(sys, "argv", ["estimator"])
    E.main()
    out = config.processed_dir() / "estimator"
    assert (out / "effort_scores.csv").exists() and (out / "estimator.joblib").exists()
    sc = pd.read_csv(out / "effort_scores.csv")
    med = sc.groupby(sc.category + "/" + sc.label.astype(str)).score_nolevel.median()
    assert med["nonspeech/screaming"] > med["reading/loud"] > med["reading/regular"]


def test_level_matched_check_counts_all_pairs():
    df = synth_features()
    _, _, _, scores = E.run_ablation(df, C=1.0)
    out = E.level_matched_check(df, scores)
    assert out.pairs.sum() == 17 * 10  # 17 speakers x 10 shared texts
    assert out.median_gap_db.is_monotonic_increasing
    assert ((out.nolevel_acc >= 0) & (out.nolevel_acc <= 1)).all()