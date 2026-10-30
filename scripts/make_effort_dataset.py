#!/usr/bin/env python3
"""Build the effort-conditioned fine-tuning dataset from EARS (run on the Mac, project env).

    python scripts/make_effort_dataset.py --dry-run     # statistics only, writes nothing
    python scripts/make_effort_dataset.py               # writes ./EffortDataset (git-ignored)

Reads  data_local/ears_processed/manifest.csv              (transcripts, split, duration)
       data_local/ears_processed/estimator/effort_scores.csv (estimator scores)
       data_local/ears_raw/<speaker>/<stem>.wav             (48 kHz audio)
Writes EffortDataset/wavs/<id>.wav + <id>.txt   16 kHz mono PCM16, levels untouched
       EffortDataset/effort_manifest.jsonl      {"id", "effort", "refs": [...], ...}
       EffortDataset/summary.json

Targets: training-split speech clips with transcripts, 1 to 30 s, text <= 240 characters. Excluded styles
(default): whisper, highpitch, lowpitch, fast, slow, because the estimator scores pitch and rate manipulations
as effort; what remains is regular and loud reading plus the scripted emotion clips.
Effort label: 0.5 at the median score of regular reading, 1.6 at the median score of loud reading
(linear in between and beyond), clipped to [0.2, 2.0]; this matches the scale the base model's
exaggeration knob already responds to (baseline: 0.5 is regular, 1.6 is about real loud).
References: 8 s windows of calm, speech-dense freeform monologue of the same speaker (fallback:
concatenated regular reading clips, which are then removed from the targets).

EARS is CC-NC: keep EffortDataset out of git and do not share it.
"""

import argparse
import json
import math
import random
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.io import wavfile
from scipy.signal import resample_poly

warnings.filterwarnings("ignore", category=wavfile.WavFileWarning)
SR_OUT = 16000
FRAME_S = 0.05


def truthy(s):
    return s.astype(str).str.strip().str.lower().isin(["true", "1", "yes"])


def read_mono(path):
    sr, x = wavfile.read(str(path))
    if x.dtype.kind in "iu":
        x = x.astype(np.float32) / float(np.iinfo(x.dtype).max + 1)
    else:
        x = x.astype(np.float32)
    if x.ndim > 1:
        x = x.mean(axis=1)
    return sr, x


def to16k(x, sr):
    if sr == SR_OUT:
        return x
    g = math.gcd(sr, SR_OUT)
    return resample_poly(x, SR_OUT // g, sr // g).astype(np.float32)


def write_wav(path, x):
    wavfile.write(str(path), SR_OUT, (np.clip(x, -1.0, 1.0) * 32767).astype(np.int16))


def frame_db(x):
    n = len(x) // int(FRAME_S * SR_OUT)
    f = x[: n * int(FRAME_S * SR_OUT)].reshape(n, -1)
    return 20 * np.log10(np.sqrt((f**2).mean(axis=1)) + 1e-9)


def candidate_windows(x, win_s, hop_s=2.0):
    """(active fraction, mean active level in dB, start frame) for each window."""
    db = frame_db(x)
    if len(db) == 0:
        return []
    active = db > db.max() - 35.0
    win, hop = int(win_s / FRAME_S), int(hop_s / FRAME_S)
    out = []
    for s in range(0, len(db) - win + 1, hop):
        a = active[s : s + win]
        out.append((float(a.mean()), float(db[s : s + win][a].mean()) if a.any() else -99.0, s))
    return out


def build_refs_freeform(spk, stems, raw, n_refs, win_s):
    cands, audio = [], {}
    for stem in stems[:6]:
        sr, x = read_mono(raw / spk / f"{stem}.wav")
        audio[stem] = to16k(x, sr)
        cands += [(f, lv, stem, s) for f, lv, s in candidate_windows(audio[stem], win_s)]
    good = [c for c in cands if c[0] >= 0.7]
    if not good:
        return []
    med = float(np.median([c[1] for c in good]))
    calm = sorted(
        (c for c in good if c[1] <= med + 3.0), key=lambda c: -c[0]
    )  # calm-ish, speech-dense
    chosen, used = [], {}
    win = int(win_s / FRAME_S)
    for f, lv, stem, s in calm:
        if all(abs(s - u) >= win for u in used.get(stem, [])):
            used.setdefault(stem, []).append(s)
            a = int(s * FRAME_S * SR_OUT)
            chosen.append(audio[stem][a : a + int(win_s * SR_OUT)])
        if len(chosen) >= n_refs:
            break
    return chosen


def build_refs_concat(spk, rows, raw, n_refs, win_s, rng):
    """Fallback: concatenate regular reading clips (0.3 s gaps) into >= win_s references."""
    stems = list(rows["stem"])
    rng.shuffle(stems)
    refs, used, cur, dur = [], [], [], 0.0
    gap = np.zeros(int(0.3 * SR_OUT), dtype=np.float32)
    for stem in stems:
        sr, x = read_mono(raw / spk / f"{stem}.wav")
        x = to16k(x, sr)
        cur += [x, gap]
        used.append(stem)
        dur += len(x) / SR_OUT + 0.3
        if dur >= win_s:
            refs.append(
                np.concatenate(cur)[: int(win_s * SR_OUT)]
                if dur > 1.5 * win_s
                else np.concatenate(cur)
            )
            cur, dur = [], 0.0
        if len(refs) >= n_refs:
            break
    return refs, used


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data_local")
    ap.add_argument("--out", default="EffortDataset")
    ap.add_argument(
        "--speakers", nargs="*", default=None, help="default: all speakers in the train split"
    )
    ap.add_argument("--score-col", default="score_nolevel")
    ap.add_argument("--anchor-regular", type=float, default=None)
    ap.add_argument("--anchor-loud", type=float, default=None)
    ap.add_argument("--n-refs", type=int, default=8)
    ap.add_argument("--ref-seconds", type=float, default=8.0)
    ap.add_argument("--max-text-chars", type=int, default=240)
    ap.add_argument("--min-dur", type=float, default=1.0)
    ap.add_argument("--max-dur", type=float, default=30.0)
    ap.add_argument(
        "--exclude-labels",
        nargs="*",
        default=["whisper", "highpitch", "lowpitch", "fast", "slow"],
        help="styles to leave out (the estimator scores pitch and rate manipulations as effort)",
    )
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    rng = random.Random(a.seed)

    data = Path(a.data_dir)
    proc, raw = data / "ears_processed", data / "ears_raw"
    man = pd.read_csv(proc / "manifest.csv")
    sc = pd.read_csv(proc / "estimator" / "effort_scores.csv")[["speaker", "stem", a.score_col]]
    df = man.merge(sc, on=["speaker", "stem"], how="left")
    df["text"] = df["text"].fillna("").astype(str).str.strip()

    spks = a.speakers or sorted(df.loc[df["split"] == "train", "speaker"].unique())
    df = df[df["speaker"].isin(spks)].copy()
    print(f"speakers ({len(spks)}): {spks}")
    if not spks:
        raise SystemExit("no training speakers found (check the 'split' column)")

    # ---- effort anchors from regular and loud reading of these speakers ----
    reg = df[df["stem"].str.match(r"^(rainbow|sentences)_\d+_regular$") & df[a.score_col].notna()]
    loud = df[df["stem"].str.match(r"^(rainbow|sentences)_\d+_loud$") & df[a.score_col].notna()]
    reg_a = a.anchor_regular if a.anchor_regular is not None else float(reg[a.score_col].median())
    loud_a = a.anchor_loud if a.anchor_loud is not None else float(loud[a.score_col].median())
    print(
        f"anchors ({a.score_col}): regular {reg_a:.2f} (n={len(reg)}), loud {loud_a:.2f} (n={len(loud)}); "
        f"baseline analysis had -5.43 and 7.15"
    )
    if not loud_a > reg_a:
        raise SystemExit("loud anchor must exceed regular anchor")

    def label_of(s):
        return float(np.clip(0.5 + 1.1 * (s - reg_a) / (loud_a - reg_a), 0.2, 2.0))

    # ---- targets ----
    t = df[truthy(df["has_transcript"]) & truthy(df["is_speech"])].copy()
    excl = [x.lower() for x in a.exclude_labels]
    lab = t["label"].astype(str).str.lower()
    t = t[~lab.isin(excl) & ~t["stem"].str.lower().str.contains("whisper")]
    print(f"excluded styles: {excl}")
    t = t[(t["duration_s"] >= a.min_dur) & (t["duration_s"] <= a.max_dur)]
    t = t[(t["text"].str.len() > 0) & (t["text"].str.len() <= a.max_text_chars)]
    t = t[t[a.score_col].notna()]
    print(f"\ncandidate targets: {len(t)} clips, {t['duration_s'].sum() / 3600:.2f} h")

    # ---- references ----
    ref_audio, pools, exclude = {}, {}, set()
    for spk in spks:
        ff = df[(df["speaker"] == spk) & df["stem"].str.startswith("freeform")]["stem"].tolist()
        refs = []
        if ff and not a.dry_run:
            refs = build_refs_freeform(spk, sorted(ff), raw, a.n_refs, a.ref_seconds)
        elif ff:
            refs = [None] * a.n_refs
        if len(refs) < 3:
            regs = t[
                (t["speaker"] == spk) & t["stem"].str.match(r"^(rainbow|sentences)_\d+_regular$")
            ]
            if a.dry_run:
                refs = [None] * 3
            else:
                refs, used = build_refs_concat(spk, regs, raw, a.n_refs, a.ref_seconds, rng)
                exclude |= {(spk, u) for u in used}
            print(
                f"  {spk}: freeform references unavailable; used concatenated regular reading ({len(refs)})"
            )
        pools[spk] = [f"ref_{spk}_{k:02d}" for k in range(len(refs))]
        for k, r in enumerate(refs):
            ref_audio[f"ref_{spk}_{k:02d}"] = r
    if exclude:
        t = t[[(s, st) not in exclude for s, st in zip(t["speaker"], t["stem"])]]

    t["effort"] = t[a.score_col].map(label_of)
    t["id"] = t["speaker"] + "_" + t["stem"]

    # ---- summary ----
    print(
        f"\nfinal targets: {len(t)} clips, {t['duration_s'].sum() / 3600:.2f} h; "
        f"references: {sum(len(v) for v in pools.values())}"
    )
    g = t.groupby("label").agg(
        n=("id", "size"),
        mean_dur=("duration_s", "mean"),
        score=(a.score_col, "median"),
        effort=("effort", "median"),
    )
    print("\nby label (median effort label):")
    print(g.round(2).sort_values("effort").to_string())
    print("\nper speaker:", t.groupby("speaker").size().to_dict())
    print(
        "effort label quantiles (5/25/50/75/95%):",
        [round(float(q), 2) for q in t["effort"].quantile([0.05, 0.25, 0.5, 0.75, 0.95])],
    )
    if a.dry_run:
        print("\n(dry run: nothing written)")
        return

    # ---- write ----
    out = Path(a.out)
    (out / "wavs").mkdir(parents=True, exist_ok=True)
    for rid, x in ref_audio.items():
        write_wav(out / "wavs" / f"{rid}.wav", x)
        (out / "wavs" / f"{rid}.txt").write_text("reference clip", encoding="utf-8")
    n = 0
    with open(out / "effort_manifest.jsonl", "w", encoding="utf-8") as mf:
        for row in t.itertuples():
            sr, x = read_mono(raw / row.speaker / f"{row.stem}.wav")
            write_wav(out / "wavs" / f"{row.id}.wav", to16k(x, sr))
            (out / "wavs" / f"{row.id}.txt").write_text(row.text, encoding="utf-8")
            mf.write(
                json.dumps(
                    {
                        "id": row.id,
                        "speaker": row.speaker,
                        "stem": row.stem,
                        "label": str(row.label),
                        "effort": round(float(row.effort), 4),
                        "raw_score": round(float(getattr(row, a.score_col)), 3),
                        "refs": pools[row.speaker],
                    }
                )
                + "\n"
            )
            n += 1
            if n % 200 == 0:
                print(f"  wrote {n}/{len(t)}")
    json.dump(
        {
            "targets": len(t),
            "hours": float(t["duration_s"].sum() / 3600),
            "speakers": spks,
            "anchor_regular": reg_a,
            "anchor_loud": loud_a,
            "score_col": a.score_col,
            "refs_per_speaker": {k: len(v) for k, v in pools.items()},
        },
        open(out / "summary.json", "w"),
        indent=2,
    )
    print(
        f"\ndone: {out}/ ({n} targets). Zip wavs/ and effort_manifest.jsonl for Colab. Keep it out of git."
    )


if __name__ == "__main__":
    main()
