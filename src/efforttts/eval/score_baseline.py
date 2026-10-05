"""STEP B of scoring (PROJECT env: needs scikit-learn and matplotlib).

    python -m efforttts.eval.score_baseline [--cfg-weight 0.5]

For every generated clip: effort features (same extractor as EARS) -> made relative to that
speaker's calm EARS baseline (median of their freeform features) -> the saved level-free
estimator. LEVEL IS EXCLUDED: TTS output gain is arbitrary, which is exactly why the
estimator was built without it. Also reports each clip's change versus the same line at the
default exaggeration (paired, so text effects cancel).

If Step A has run, merges WER and speaker similarity. Writes gap_clips.csv and gap_report.png
next to results.csv and prints summary tables, with real-EARS reference levels for context.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from efforttts import config
from efforttts.effort import estimator as E
from efforttts.effort.features import extract_features
from efforttts.eval.wer import normalize_words

DEFAULT_SETS = ("nolevel", "spectral")


def read_wav(path: Path) -> tuple[np.ndarray, int]:
    from scipy.io import wavfile

    sr, x = wavfile.read(str(path))
    if x.dtype == np.int16:
        x = x.astype(np.float64) / 32768.0
    elif x.dtype == np.int32:
        x = x.astype(np.float64) / 2**31
    else:
        x = x.astype(np.float64)
    if x.ndim > 1:
        x = x.mean(axis=1)
    return x, sr


def clip_features(x: np.ndarray, sr: int) -> dict:
    f = extract_features(x, sr)
    with np.errstate(invalid="ignore", divide="ignore"):
        f[E.F0_ST] = 12 * np.log2(f["f0_median_hz"])
    return f


def speaker_baselines(ears_features):
    """Median of each speaker's real freeform features (their calm reference)."""
    return ears_features[ears_features.category == E.REF_CATEGORY].groupby("speaker")[E.ALL_NAMES].median()


def score_clips(df, baselines, bundle, sets=DEFAULT_SETS):
    """Add score_<set> columns: relative-to-baseline features through the saved models."""
    missing = set(df.speaker) - set(baselines.index)
    if missing:
        raise SystemExit(f"No EARS freeform baseline (features.csv) for speakers {sorted(missing)}")
    rel = df.copy()
    rel[E.ALL_NAMES] = rel[E.ALL_NAMES].to_numpy() - baselines.reindex(rel.speaker)[E.ALL_NAMES].to_numpy()
    out = df.copy()
    for s in sets:
        out[f"score_{s}"] = bundle["models"][s].decision_function(rel[bundle["feature_sets"][s]])
    return out


def paired_delta(df, col: str, default_ex: float):
    """col minus the same speaker/line/seed at the default exaggeration."""
    base = (df[np.isclose(df.exaggeration, default_ex)]
            .set_index(["speaker", "line_id", "seed"])[col])
    idx = df.set_index(["speaker", "line_id", "seed"]).index
    return df[col].to_numpy() - base.reindex(idx).to_numpy()


def build_clip_table(root: Path, bundle, ears_features, default_ex: float, sets=DEFAULT_SETS):
    import pandas as pd

    gen = pd.read_csv(root / "results.csv")
    gen = gen[gen.status == "ok"].reset_index(drop=True)
    rows = []
    for r in gen.itertuples():
        x, sr = read_wav(root / r.path)
        rows.append(clip_features(x, sr))
    feats = pd.DataFrame(rows)
    df = pd.concat([gen, feats], axis=1)
    df["words_per_s"] = [len(normalize_words(t)) / a if a else np.nan for t, a in zip(df.text, df.audio_s)]
    df = score_clips(df, speaker_baselines(ears_features), bundle, sets)
    for s in sets:
        df[f"delta_{s}"] = paired_delta(df, f"score_{s}", default_ex)
    asr = root / "asr_speaker.csv"
    if asr.exists():
        df = df.merge(pd.read_csv(asr), on="job_id", how="left")
    else:
        print("[note] asr_speaker.csv not found: run Step A in the Chatterbox env for WER/similarity")
        for c in ("wer", "sim_to_centroid", "sim_to_reference"):
            df[c] = np.nan
    return df


def ears_context(speakers):
    """Median level-free scores of REAL EARS speech for the same speakers (same scale)."""
    import pandas as pd

    p = config.processed_dir() / "estimator" / "effort_scores.csv"
    if not p.exists():
        return {}
    es = pd.read_csv(p)
    es = es[es.speaker.isin(speakers)]
    keys = {"real regular": ("reading", "regular"), "real loud": ("reading", "loud"),
            "real anger": ("emotion_freeform", "anger"), "real yelling": ("nonspeech", "yelling")}
    out = {}
    for name, (c, l) in keys.items():
        v = es[(es.category == c) & (es.label == l)]["score_nolevel"]
        if len(v):
            out[name] = float(v.median())
    return out


def real_similarity_context(root: Path):
    import pandas as pd

    p = root / "real_calibration.csv"
    if not p.exists():
        return {}
    c = pd.read_csv(p)
    return {f"real {l}": float(g.sim_to_centroid.median()) for l, g in c.groupby("label")}


def print_tables(df, ctx, sim_ctx):
    import pandas as pd

    pd.set_option("display.width", 200)
    g = df.groupby("exaggeration").agg(
        n=("job_id", "size"), effort=("score_nolevel", "median"),
        effort_vs_default=("delta_nolevel", "median"), wer_median=("wer", "median"),
        wer_mean=("wer", "mean"), sim_centroid=("sim_to_centroid", "median"),
        sim_reference=("sim_to_reference", "median"), words_per_s=("words_per_s", "median"))
    print("\n== By exaggeration (medians; effort = level-free score, EARS-referenced) ==")
    print(g.round(3).to_string())
    print("\n== Median effort by line intent x exaggeration ==")
    print(df.pivot_table(index="intent", columns="exaggeration", values="score_nolevel",
                         aggfunc="median").round(2).to_string())
    print("\n== Median effort by speaker x exaggeration ==")
    print(df.pivot_table(index="speaker", columns="exaggeration", values="score_nolevel",
                         aggfunc="median").round(2).to_string())
    if "score_spectral" in df:
        print("\n== Same, spectral-only estimator (no pitch features) ==")
        print(df.groupby("exaggeration").score_spectral.median().round(2).to_string())
    if ctx:
        print("\n== Real EARS speech, same speakers, same estimator ==")
        print({k: round(v, 2) for k, v in ctx.items()})
    if sim_ctx:
        print("\n== Real speech similarity to the speaker's own centroid (identity drop from real effort) ==")
        print({k: round(v, 3) for k, v in sim_ctx.items()})


def save_plot(df, ctx, sim_ctx, out: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    exs = sorted(df.exaggeration.unique())
    fig, ax = plt.subplots(2, 2, figsize=(12, 9))

    a = ax[0, 0]
    for spk, g in df.groupby("speaker"):
        m = g.groupby("exaggeration").score_nolevel.median()
        a.plot(m.index, m.values, lw=0.8, alpha=0.5, label=spk)
    a.plot(exs, df.groupby("exaggeration").score_nolevel.median().values, "k-o", lw=2.5, label="median")
    for name, v in ctx.items():
        a.axhline(v, ls=":", c="gray"); a.text(exs[0], v, f" {name}", va="bottom", fontsize=8, color="gray")
    a.set_xlabel("exaggeration"); a.set_ylabel("effort score (level-free)"); a.set_title("Effort vs exaggeration")
    a.legend(fontsize=7, ncol=2)

    a = ax[0, 1]
    if df.wer.notna().any():
        g = df.groupby("exaggeration").wer
        a.plot(exs, g.mean().values, "o-", label="mean"); a.plot(exs, g.median().values, "s--", label="median")
        a.legend()
    a.set_xlabel("exaggeration"); a.set_ylabel("word error rate"); a.set_title("Intelligibility (Whisper)")

    a = ax[1, 0]
    if df.sim_to_centroid.notna().any():
        a.plot(exs, df.groupby("exaggeration").sim_to_centroid.median().values, "k-o", lw=2, label="generated")
        for name, v in sim_ctx.items():
            a.axhline(v, ls=":", c="gray"); a.text(exs[0], v, f" {name}", va="bottom", fontsize=8, color="gray")
    a.set_xlabel("exaggeration"); a.set_ylabel("cosine similarity to speaker centroid"); a.set_title("Identity")

    a = ax[1, 1]
    if df.sim_to_centroid.notna().any():
        sc = a.scatter(df.score_nolevel, df.sim_to_centroid, c=df.exaggeration, cmap="viridis", s=12, alpha=0.7)
        fig.colorbar(sc, ax=a, label="exaggeration")
    a.set_xlabel("effort score"); a.set_ylabel("similarity to centroid"); a.set_title("Effort vs identity (per clip)")
    plt.tight_layout()
    plt.savefig(out, dpi=110)
    plt.close(fig)


def main(argv: list[str] | None = None) -> None:
    import joblib

    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg-weight", type=float, default=0.5)
    ap.add_argument("--default-ex", type=float, default=0.5)
    args = ap.parse_args(argv)

    root = config.data_root() / "baselines" / "chatterbox" / f"cfg{args.cfg_weight:.2f}"
    bundle = joblib.load(config.processed_dir() / "estimator" / "estimator.joblib")
    if bundle["mode"] != "relative":
        raise SystemExit("Saved estimator is not in relative mode; rerun: python -m efforttts.effort.estimator")
    ears = E.load_features()
    df = build_clip_table(root, bundle, ears, args.default_ex)
    df.to_csv(root / "gap_clips.csv", index=False)
    ctx = ears_context(set(df.speaker))
    sim_ctx = real_similarity_context(root)
    print(f"{len(df)} clips scored -> {root / 'gap_clips.csv'}")
    print_tables(df, ctx, sim_ctx)
    save_plot(df, ctx, sim_ctx, root / "gap_report.png")
    print(f"\nplot -> {root / 'gap_report.png'}")


if __name__ == "__main__":
    main()