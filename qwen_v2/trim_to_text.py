#!/usr/bin/env python3
"""Offline end-of-speech guard: cut each clip right after the last word of the TARGET text, drop the runaway tail.

A hard length cap (qwen_gen.py --bound-factor 1.5) cut real speech, because shouted speech is slower than the cap assumes. This instead takes
clips generated with a LOOSE cap (the 150-token clips in dev2_L21 are 12 s long and already contain the whole sentence plus the tail), transcribes
them with Whisper word timestamps, aligns the words to the target text and crops where the voice has stopped after the last target word (searching up to 0.6 s past Whisper's end time, because the shouted last word was being clipped early; 60 ms fade-out).
Clips with no usable alignment are left unchanged and counted.

Run in the CHATTERBOX env (needs transformers, scipy, pandas), repo root:
    python qwen_v2/trim_to_text.py dev2_L21            # new root dev2_L21_trim, only clips with k >= 1.0 (--min-k)
Then Step A, Step B, sweep_summary.py and make_ladder.py on the new root as usual. Outputs are EARS-derived voices: local only.
"""

import argparse
import difflib
import re
from pathlib import Path

import numpy as np
import pandas as pd


def norm(text):
    return re.findall(r"[a-z0-9']+", str(text).lower())


def end_of_text(chunks, target):
    """chunks: [(word, start, end)] from the ASR; target: list of words. Returns the end time (s) of the last target word, or None."""
    hyp = [norm(w)[0] if norm(w) else "" for w, _, _ in chunks]
    if not hyp or not target:
        return None
    last = len(target) - 1
    blocks = [
        b
        for b in difflib.SequenceMatcher(None, hyp, target, autojunk=False).get_matching_blocks()
        if b.size
    ]
    best = None  # (target index, hyp index) of the furthest matched target word
    for i, j, n in blocks:
        for t in range(j, j + n):
            if best is None or t > best[0]:
                best = (t, i + (t - j))
    if best is None:
        return None
    t_idx, h_idx = best
    end = chunks[h_idx][2]
    if t_idx < last:  # the final words were not recognised: estimate their length
        end += 0.35 * (last - t_idx)
    return end


def level_db(x, sr):
    n = int(0.025 * sr)
    fr = x[: len(x) // n * n].reshape(-1, n)
    return float(20 * np.log10(np.percentile(np.sqrt((fr**2).mean(axis=1)) + 1e-9, 95)))


def voice_offset(x, sr, start_t, silence_db, max_extend, min_quiet=0.08):
    """First time at or after start_t where the voice has really stopped: the start of the first run of >= min_quiet seconds whose
    25 ms RMS is silence_db below the clip's speech level. Searches at most max_extend seconds; returns start_t + max_extend if none."""
    thr = 10 ** ((level_db(x, sr) - silence_db) / 20)
    hop, win = int(0.010 * sr), int(0.025 * sr)
    quiet = 0
    for i in range(int(start_t * sr), min(len(x), int((start_t + max_extend) * sr)) - win, hop):
        if np.sqrt(np.mean(x[i : i + win] ** 2)) < thr:
            quiet += hop
            if quiet >= int(min_quiet * sr):
                return (i + win - quiet) / sr
        else:
            quiet = 0
    return start_t + max_extend


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tags", nargs="+")
    ap.add_argument("--base", default="data_local/baselines/qwen_v2")
    ap.add_argument("--suffix", default="_trim")
    ap.add_argument("--min-k", type=float, default=1.0)
    ap.add_argument(
        "--pad", type=float, default=0.10, help="minimum seconds kept after the last word"
    )
    ap.add_argument(
        "--max-extend",
        type=float,
        default=0.6,
        help="seconds to search past the last word for the voice to actually stop",
    )
    ap.add_argument(
        "--silence-db",
        type=float,
        default=30.0,
        help="quiet = this many dB below the clip's speech level",
    )
    ap.add_argument("--fade", type=float, default=0.06)
    a = ap.parse_args()

    from scipy.io import wavfile
    from scipy.signal import resample_poly
    from transformers import pipeline

    asr = pipeline("automatic-speech-recognition", model="openai/whisper-small.en", device=-1)
    base = Path(a.base)
    for tag in a.tags:
        src, dst = base / tag, base / f"{tag}{a.suffix}"
        df = pd.read_csv(src / "results.csv")
        df = df[(df.status == "ok") & (df.exaggeration >= a.min_k)].copy()
        status, new_len = [], []
        for n, (_, r) in enumerate(df.iterrows(), 1):
            sr, x = wavfile.read(str(src / r["path"]))
            x = x.astype("float32")
            if x.ndim > 1:
                x = x.mean(axis=1)
            x16 = resample_poly(x, 16000, sr).astype("float32")
            out = asr({"raw": x16, "sampling_rate": 16000}, return_timestamps="word")
            chunks = [
                (
                    c["text"],
                    c["timestamp"][0] or 0.0,
                    c["timestamp"][1]
                    if c["timestamp"][1] is not None
                    else (c["timestamp"][0] or 0.0) + 0.3,
                )
                for c in out.get("chunks", [])
            ]
            end = end_of_text(chunks, norm(r["text"]))
            dur = len(x) / sr
            if end is None:
                st, y = "no_match", x
            else:
                cut = max(end + a.pad, voice_offset(x, sr, end, a.silence_db, a.max_extend) + 0.05)
                if cut >= dur - 0.05:
                    st, y = "unchanged", x
                else:
                    y = x[: int(cut * sr)].copy()
                    nf = min(len(y), int(a.fade * sr))
                    y[-nf:] *= np.linspace(1.0, 0.0, nf, dtype="float32")
                    st = "trimmed"
            (dst / r["path"]).parent.mkdir(parents=True, exist_ok=True)
            wavfile.write(str(dst / r["path"]), sr, y.astype("float32"))
            status.append(st)
            new_len.append(round(len(y) / sr, 3))
            if n % 24 == 0:
                print(f"  {tag}: {n}/{len(df)}")
        df["orig_audio_s"] = df["audio_s"]
        df["audio_s"] = new_len
        df["trim_status"] = status
        df.to_csv(dst / "results.csv", index=False)
        print(f"{tag} -> {dst}: {pd.Series(status).value_counts().to_dict()}")
        print(
            df.groupby("exaggeration")["trim_status"]
            .value_counts()
            .unstack(fill_value=0)
            .to_string()
        )


if __name__ == "__main__":
    main()
