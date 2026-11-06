#!/usr/bin/env python3
"""Does the emotion drift as k rises? A crude automatic proxy for "the voice turns manic".

Runs the small speech-emotion classifier superb/wav2vec2-base-superb-er (Apache-2.0; 4 classes from IEMOCAP: neu, hap, ang, sad; about 63% accuracy
on acted dialogue) on every clip and reports, per strength k, the mean probability of each class and its paired change versus the unsteered clip of the
same speaker and line. It is NOT validated on synthetic shouting: read it as a direction of drift (e.g. "hap" rising on a fear line), never as ground truth,
and always next to human labels. Run in the CHATTERBOX env (needs transformers, scipy, pandas), repo root:
    python qwen_v2/emotion_drift.py dev2_L21_trim2 --group both --line-ids D01 D03     # --group all|line|speaker|both; all lines by default
Writes <root>/emotion_drift.csv. Outputs are EARS-derived voices: local only.
"""

import argparse
from pathlib import Path

import pandas as pd


def summarize(df, base, labels, group="all"):
    """df, base: one row per clip with columns speaker, line_id, exaggeration and one probability column per label."""
    key = ["speaker", "line_id"]
    b = base.groupby(key)[labels].mean()
    d = df.copy()
    for lab in labels:
        d[f"d_{lab}"] = (
            d[lab].to_numpy() - b[lab].reindex(list(zip(d.speaker, d.line_id))).to_numpy()
        )
    group = {
        "all": ["exaggeration"],
        "line": ["line_id", "exaggeration"],
        "speaker": ["speaker", "exaggeration"],
        "both": ["speaker", "line_id", "exaggeration"],
    }[group]
    out = d.groupby(group).agg(
        n=("speaker", "size"),
        **{lab: (lab, "mean") for lab in labels},
        **{f"d_{lab}": (f"d_{lab}", "median") for lab in labels},
    )
    out["top"] = (
        d.assign(top=d[labels].idxmax(axis=1))
        .groupby(group)["top"]
        .agg(lambda s: s.value_counts().to_dict())
    )
    return out


def classify(root, clf, read_16k, line_ids=None):
    res = pd.read_csv(root / "results.csv")
    res = res[res.status == "ok"]
    if line_ids:
        res = res[res.line_id.isin(line_ids)]
    rows = []
    for _, r in res.iterrows():
        scores = clf({"raw": read_16k(root / r["path"]), "sampling_rate": 16000}, top_k=None)
        rows.append(
            {
                "speaker": r["speaker"],
                "line_id": r["line_id"],
                "exaggeration": r["exaggeration"],
                **{s["label"]: s["score"] for s in scores},
            }
        )
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tags", nargs="+")
    ap.add_argument("--base", default="data_local/baselines/qwen_v2")
    ap.add_argument("--ref", default="dev_base")
    ap.add_argument("--line-ids", nargs="*", default=None)
    ap.add_argument("--by-line", action="store_true", help="same as --group line")
    ap.add_argument("--group", choices=["all", "line", "speaker", "both"], default="all")
    a = ap.parse_args()

    from scipy.io import wavfile
    from scipy.signal import resample_poly
    from transformers import pipeline

    clf = pipeline("audio-classification", model="superb/wav2vec2-base-superb-er", device=-1)

    def read_16k(p):
        sr, x = wavfile.read(str(p))
        x = x.astype("float32")
        x = x.mean(axis=1) if x.ndim > 1 else x
        return resample_poly(x, 16000, sr).astype("float32")

    base_root = Path(a.base)
    base = classify(base_root / a.ref, clf, read_16k, a.line_ids)
    labels = [c for c in base.columns if c not in ("speaker", "line_id", "exaggeration")]
    pd.set_option("display.width", 220)
    print(
        f"unsteered ({a.ref}): mean probabilities {base[labels].mean().round(3).to_dict()}, top class counts {base[labels].idxmax(axis=1).value_counts().to_dict()}\n"
    )
    for tag in a.tags:
        df = classify(base_root / tag, clf, read_16k, a.line_ids)
        df.to_csv(base_root / tag / "emotion_drift.csv", index=False)
        print(
            f"== {tag} ==\n{summarize(df, base, labels, 'line' if a.by_line else a.group).round(3).to_string()}\n"
        )


if __name__ == "__main__":
    main()
