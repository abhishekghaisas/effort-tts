"""Acoustic features for vocal effort.

Design rule: ONLY `level_active_db` depends on recording gain. Every other
feature is computed from spectral shape, periodicity, or pitch, and is
invariant to scaling the waveform. That lets the estimator be trained with and
without level, to test whether it learns effort or just loudness
(see FEATURE_GROUPS).

Features are computed on ACTIVE frames (within ACTIVE_RANGE_DB of the file's
95th-percentile frame level), so pauses do not dilute them. Voiced-only
features need >= MIN_VOICED frames, otherwise NaN (e.g. whispered speech).

Approximations (this is a prototype, not Praat):
  * Voicing is strict (r >= 0.70, F0 70-700 Hz, runs >= 4 frames, and frames more than
    ~1 octave from a robust per-clip F0 anchor are dropped) because breath and fricative
    noise otherwise gets "voiced" F0 values pinned near the upper limit.
  * F0 / voicing come from a normalized autocorrelation (Praat-style octave
    cost, quarter-sample lag interpolation, pre-emphasized input). Pre-emphasis
    stops low-frequency rumble being read as periodicity (seen on whispered
    speech); interpolation stops octave errors at high F0. Still approximate.
  * cpp_db is an UNSMOOTHED cepstral peak prominence in dB-cepstrum units.
    Use it relatively (compare conditions), not against published CPPS values.
  * F0 values are speaker-dependent; speaker-relative normalization is a
    separate later step.

CLI:
    python -m efforttts.effort.features [--speakers 1 2] [--workers 3] [--force]
Cached results are versioned (FEATURE_VERSION); changing the code + bumping it recomputes.
Reads processed_dir()/manifest.csv, writes processed_dir()/features.csv.
"""
from __future__ import annotations

import argparse
import time
from concurrent.futures import ProcessPoolExecutor
from math import gcd
from pathlib import Path

import numpy as np

from efforttts import config

SR = 16_000
WIN = 640  # 40 ms
HOP = 160  # 10 ms
NFFT = 2048
F0_MIN, F0_MAX = 70.0, 700.0
LAG_MIN = int(SR / F0_MAX)  # 16
LAG_MAX = int(SR / F0_MIN)  # 266
ACTIVE_RANGE_DB = 30.0
VOICED_R = 0.70  # strict: breath/fricative noise reaches r~0.45-0.6 near the F0 limit
F0_GATE = 2.2  # keep voiced frames within this factor of the clip's robust F0 anchor
STRONG_R = 0.85  # frames this periodic define the anchor
MIN_RUN = 4  # voiced runs shorter than this many frames (40 ms) are discarded
MIN_VOICED = 10
OCTAVE_COST = 0.01
F0_PREFILTER = "preemph"  # "preemph" | "none"; F0/voicing path only (as in Praat)
PREEMPH = 0.97
UP = 4  # autocorrelation interpolation factor (quarter-sample lags)
BLOCK = 1500  # frames per block, bounds memory
# Bump whenever feature computation changes. Cached per-speaker results are keyed
# by this, so stale features can never be silently reused.
FEATURE_VERSION = 4

FEATURE_GROUPS = {
    "level": ["level_active_db"],
    "spectral": [
        "alpha_ratio_db",
        "hammarberg_db",
        "tilt_db_per_khz",
        "cpp_db",
        "alpha_ratio_v_db",
        "tilt_v_db_per_khz",
        "hnr_v_db",
    ],
    "f0": ["f0_median_hz", "f0_range_st", "f0_std_st"],
    "other": ["voiced_frac", "active_frac"],
}
ALL_FEATURES = [f for g in FEATURE_GROUPS.values() for f in g]

_WINDOW = np.hanning(WIN)
_FREQS = np.fft.rfftfreq(NFFT, 1.0 / SR)
_LAGS = np.arange(LAG_MIN, LAG_MAX + 1)
_LAGS_UP = np.arange(LAG_MIN * UP, LAG_MAX * UP + 1)
_OCT_PENALTY = OCTAVE_COST * np.log2(F0_MIN * (_LAGS_UP / UP) / SR)
_AW = np.fft.irfft(np.abs(np.fft.rfft(_WINDOW, NFFT)) ** 2, NFFT * UP)[: (LAG_MAX + 2) * UP]
_AW = _AW / _AW[0]
_LOW = (_FREQS >= 50) & (_FREQS < 1000)
_HIGH = (_FREQS >= 1000) & (_FREQS < 5000)
_H_LO = _FREQS < 2000
_H_HI = (_FREQS >= 2000) & (_FREQS < 5000)
_TILT = (_FREQS >= 100) & (_FREQS <= 5000)
_XK = _FREQS[_TILT] / 1000.0
_XKC = _XK - _XK.mean()
_LAGC = _LAGS - _LAGS.mean()
_EPS = 1e-12


def _resample(x: np.ndarray, sr: int) -> np.ndarray:
    from scipy.signal import resample_poly  # lazy

    g = gcd(SR, sr)
    return resample_poly(x, SR // g, sr // g)


def _remove_short_runs(mask: np.ndarray, min_len: int) -> np.ndarray:
    """Drop runs of True shorter than min_len (isolated 'voiced' frames are almost always noise)."""
    m = np.asarray(mask, dtype=bool)
    out = m.copy()
    padded = np.concatenate([[False], m, [False]])
    edges = np.flatnonzero(padded[1:] != padded[:-1])
    for start, end in zip(edges[::2], edges[1::2]):
        if end - start < min_len:
            out[start:end] = False
    return out


def _f0_path_signal(x: np.ndarray) -> np.ndarray:
    """Signal used for F0/voicing only. LF rumble is correlated at short lags and
    would otherwise be mistaken for periodicity near F0_MAX."""
    if F0_PREFILTER == "preemph":
        return np.append(x[0], x[1:] - PREEMPH * x[:-1])
    return x


def _block_features(fr: np.ndarray, frp: np.ndarray) -> dict[str, np.ndarray]:
    """fr: (B, WIN) frames; frp: same frames of the pre-emphasized signal (F0/voicing only)."""
    fr = fr - fr.mean(axis=1, keepdims=True)
    frame_db = 10 * np.log10(np.mean(fr * fr, axis=1) + _EPS)

    P = np.abs(np.fft.rfft(fr * _WINDOW, n=NFFT, axis=1)) ** 2  # (B, 1025)
    dB = 10 * np.log10(P + _EPS)
    ar = np.arange(len(fr))

    # --- spectral shape (gain-invariant) ---
    alpha = 10 * np.log10(P[:, _HIGH].sum(1) + _EPS) - 10 * np.log10(P[:, _LOW].sum(1) + _EPS)
    hamm = 10 * np.log10(P[:, _H_LO].max(1) + _EPS) - 10 * np.log10(P[:, _H_HI].max(1) + _EPS)
    d = dB[:, _TILT]
    tilt = ((d - d.mean(1, keepdims=True)) @ _XKC) / np.sum(_XKC**2)

    # --- periodicity / F0 from normalized autocorrelation (Boersma-style) ---
    # Uses PRE-EMPHASIZED frames: low-frequency rumble is highly correlated at
    # short lags and otherwise gets mistaken for periodicity near F0_MAX.
    frp = frp - frp.mean(axis=1, keepdims=True)
    Pp = np.abs(np.fft.rfft(frp * _WINDOW, n=NFFT, axis=1)) ** 2
    ac = np.fft.irfft(Pp, n=NFFT * UP, axis=1)[:, : (LAG_MAX + 2) * UP]  # band-limited interp
    r = ac / (ac[:, :1] + _EPS) / _AW[None, :]
    seg = r[:, LAG_MIN * UP : LAG_MAX * UP + 1]
    idx = (seg - _OCT_PENALTY[None, :]).argmax(axis=1)
    rmax = seg[ar, idx]
    li = idx + LAG_MIN * UP
    a, b, c = r[ar, li - 1], r[ar, li], r[ar, li + 1]
    den = a - 2 * b + c
    delta = np.where(den < -1e-9, 0.5 * (a - c) / np.where(den < -1e-9, den, -1.0), 0.0)
    f0 = SR / ((li + np.clip(delta, -1, 1)) / UP)
    rc = np.clip(rmax, 1e-3, 0.999)
    hnr = 10 * np.log10(rc / (1 - rc))

    # --- cepstral peak prominence (unsmoothed) ---
    cc = np.fft.irfft(dB, n=NFFT, axis=1)[:, LAG_MIN : LAG_MAX + 1]
    pk_i = cc.argmax(axis=1)
    pk = cc[ar, pk_i]
    slope = ((cc - cc.mean(1, keepdims=True)) @ _LAGC) / np.sum(_LAGC**2)
    line_at_pk = cc.mean(1) + slope * (_LAGS[pk_i] - _LAGS.mean())
    cpp = pk - line_at_pk

    return {
        "frame_db": frame_db,
        "alpha": alpha,
        "hamm": hamm,
        "tilt": tilt,
        "rmax": rmax,
        "f0": f0,
        "hnr": hnr,
        "cpp": cpp,
    }


def extract_features(x: np.ndarray, sr: int) -> dict[str, float]:
    """Utterance-level features from a mono float waveform (full scale = 1.0)."""
    x = np.asarray(x, dtype=np.float64)
    if sr != SR:
        x = _resample(x, sr)
    if len(x) < WIN:
        x = np.pad(x, (0, WIN - len(x)))
    xp = _f0_path_signal(x)
    view = np.lib.stride_tricks.sliding_window_view(x, WIN)[::HOP]
    viewp = np.lib.stride_tricks.sliding_window_view(xp, WIN)[::HOP]
    parts = [
        _block_features(
            np.array(view[i : i + BLOCK], dtype=np.float64),
            np.array(viewp[i : i + BLOCK], dtype=np.float64),
        )
        for i in range(0, len(view), BLOCK)
    ]
    f = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}

    out = {k: float("nan") for k in ALL_FEATURES}
    db = f["frame_db"]
    thr = np.percentile(db, 95) - ACTIVE_RANGE_DB
    active = (db >= thr) & (db > -90)
    n_act = int(active.sum())
    out["active_frac"] = n_act / len(db)
    if n_act == 0:
        return out

    out["level_active_db"] = float(10 * np.log10(np.mean(10 ** (db[active] / 10))))
    for key, name in (
        ("alpha", "alpha_ratio_db"),
        ("hamm", "hammarberg_db"),
        ("tilt", "tilt_db_per_khz"),
        ("cpp", "cpp_db"),
    ):
        out[name] = float(f[key][active].mean())

    voiced = _remove_short_runs(active & (f["rmax"] >= VOICED_R), MIN_RUN)
    if voiced.sum() >= MIN_VOICED:
        # Gate outliers (breath/fricative bursts, octave slips) around a robust anchor
        # taken from the most strongly periodic frames, then re-apply the run filter.
        strong = voiced & (f["rmax"] >= STRONG_R)
        base = f["f0"][strong] if strong.sum() >= MIN_VOICED else f["f0"][voiced]
        anchor = float(np.median(base))
        voiced = voiced & (f["f0"] >= anchor / F0_GATE) & (f["f0"] <= anchor * F0_GATE)
        voiced = _remove_short_runs(voiced, MIN_RUN)
    n_v = int(voiced.sum())
    out["voiced_frac"] = n_v / n_act
    if n_v >= MIN_VOICED:
        out["alpha_ratio_v_db"] = float(f["alpha"][voiced].mean())
        out["tilt_v_db_per_khz"] = float(f["tilt"][voiced].mean())
        out["hnr_v_db"] = float(f["hnr"][voiced].mean())
        st = 12 * np.log2(f["f0"][voiced])
        out["f0_median_hz"] = float(np.median(f["f0"][voiced]))
        out["f0_range_st"] = float(np.percentile(st, 90) - np.percentile(st, 10))
        out["f0_std_st"] = float(st.std())
    return out


# ----------------------------------------------------------------------------
# Batch processing over the manifest
# ----------------------------------------------------------------------------
def _worker(args: tuple[str, str]) -> tuple[str, float]:
    import pandas as pd
    import soundfile as sf

    speaker, part_path = args
    t0 = time.time()
    man = pd.read_csv(config.processed_dir() / "manifest.csv", keep_default_na=False)
    rows = []
    for r in man[man.speaker == speaker].itertuples():
        x, sr = sf.read(str(config.raw_dir() / r.path), dtype="float32")
        if x.ndim > 1:
            x = x.mean(axis=1)
        rows.append({"speaker": r.speaker, "stem": r.stem, **extract_features(x, sr)})
    pd.DataFrame(rows).to_csv(part_path, index=False)
    return speaker, time.time() - t0


def _group_key(r) -> str:
    if r["category"] in ("reading", "nonspeech"):
        return f"{r['category']}/{r['label']}"
    return r["category"]


def _paired_deltas(df, a: str, b: str, feats: list[str]):
    """Per (speaker, text_id) difference of style a minus style b, same words."""
    rd = df[df.category == "reading"]
    A = rd[rd.label == a].set_index(["speaker", "text_id"])[feats]
    B = rd[rd.label == b].set_index(["speaker", "text_id"])[feats]
    common = A.index.intersection(B.index)
    return (A.loc[common] - B.loc[common]), len(common)


def _delta_table(df) -> None:
    """Each file's feature minus its OWN speaker's median on regular reading,
    then medians per group. Removes between-speaker gain/voice differences."""
    import pandas as pd

    feats = ["level_active_db", "alpha_ratio_db", "hammarberg_db", "cpp_db"]
    base = (df[(df.category == "reading") & (df.label == "regular")]
            .groupby("speaker")[feats].median())
    d = df.copy()
    d["g"] = d["category"] + "/" + d["label"].astype(str)
    keep = d.g.str.startswith("emotion_freeform/") | d.g.isin(
        ["reading/loud", "reading/highpitch", "nonspeech/yelling", "nonspeech/screaming"])
    d = d[keep].join(base, on="speaker", rsuffix="_base")
    for f in feats:
        d[f] = d[f] - d[f + "_base"]
    t = d.groupby("g")[feats].median().sort_values("alpha_ratio_db", ascending=False)
    t.columns = [c.replace("_db", "") + " (vs own regular)" for c in t.columns]
    print("\n== Delta vs speaker's own regular reading, sorted by alpha ratio ==")
    print(t.round(2).to_string())


def _summarize(df) -> None:
    import pandas as pd

    pd.set_option("display.width", 200)
    df = df.copy()
    df["group"] = df.apply(_group_key, axis=1)
    show = ["level_active_db", "alpha_ratio_db", "tilt_db_per_khz", "hammarberg_db",
            "cpp_db", "hnr_v_db", "f0_median_hz", "voiced_frac"]
    g = df.groupby("group")[show].median().sort_values("level_active_db")
    print("\n== Median features by group ==")
    print(g.round(2).to_string())

    feats = ["level_active_db", "alpha_ratio_db", "tilt_db_per_khz", "hammarberg_db",
             "cpp_db", "hnr_v_db", "f0_median_hz"]
    for a in ("loud", "highpitch", "whisper"):
        d, n = _paired_deltas(df, a, "regular", feats)
        print(f"\n== Paired same-text, same-speaker delta: {a} - regular  (n={n}) ==")
        res = pd.DataFrame({"mean_delta": d.mean(), "frac_positive": (d > 0).mean()})
        print(res.round(2).to_string())
    _delta_table(df)


def main() -> None:
    import pandas as pd

    ap = argparse.ArgumentParser()
    ap.add_argument("--speakers", type=int, nargs="*", default=[])
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    mpath = config.processed_dir() / "manifest.csv"
    if not mpath.exists():
        raise SystemExit("Missing manifest.csv. Run: python -m efforttts.data.manifest")
    man = pd.read_csv(mpath, keep_default_na=False)
    speakers = sorted(man.speaker.unique())
    if args.speakers:
        want = {f"p{i:03d}" for i in args.speakers}
        speakers = [s for s in speakers if s in want]

    parts = config.processed_dir() / "features_parts"
    parts.mkdir(parents=True, exist_ok=True)
    part_of = lambda s: parts / f"{s}.v{FEATURE_VERSION}.csv"  # noqa: E731
    todo = [(s, str(part_of(s))) for s in speakers if args.force or not part_of(s).exists()]
    print(f"{len(speakers)} speakers, {len(todo)} to process, {len(speakers) - len(todo)} cached")
    if todo:
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            for name, secs in ex.map(_worker, todo):
                print(f"[done] {name}  {secs:.0f}s")

    feats = pd.concat([pd.read_csv(part_of(s)) for s in speakers])
    keep = ["speaker", "stem", "path", "category", "label", "text_id", "text", "split",
            "duration_s", "is_speech"]
    df = man[man.speaker.isin(speakers)][keep].merge(feats, on=["speaker", "stem"], how="left")
    out = config.processed_dir() / "features.csv"
    df.to_csv(out, index=False)
    print(f"\nFeatures: {len(df)} files -> {out}")
    print(f"Files with NaN level (no active frames): {int(df.level_active_db.isna().sum())}")
    _summarize(df)


if __name__ == "__main__":
    main()