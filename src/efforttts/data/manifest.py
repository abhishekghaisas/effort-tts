"""Build one table describing every EARS audio file.

    python -m efforttts.data.manifest                  # all downloaded speakers
    python -m efforttts.data.manifest --speakers 1 2 3
    python -m efforttts.data.manifest --workers 2 --force

Per-speaker results are cached in processed_dir()/manifest_parts/ so reruns
and Colab restarts only process new speakers. Final table:
processed_dir()/manifest.csv

Columns: speaker, stem, path, category, label, text_id, text, has_transcript,
is_speech, needs_asr, duration_s, sr, peak_dbfs, rms_dbfs, clip_frac, split.
Audio is NOT modified or normalized here; levels are measured as recorded.
"""
from __future__ import annotations

import argparse
import json
import math
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from efforttts import config
from efforttts.data.inventory import AUDIO_EXTS, classify

SPEECH_CATEGORIES = {"reading", "emotion", "emotion_freeform", "freeform"}
CLIP_THRESHOLD = 0.999


def audio_stats(x: np.ndarray) -> dict:
    """Level statistics of a float waveform (full scale = 1.0)."""
    x = np.asarray(x, dtype=np.float64)
    if x.size == 0:
        return {"peak_dbfs": float("-inf"), "rms_dbfs": float("-inf"), "clip_frac": 0.0}
    a = np.abs(x)
    peak = float(a.max())
    rms = float(np.sqrt(np.mean(x * x)))
    db = lambda v: 20.0 * math.log10(max(v, 1e-12))  # noqa: E731
    return {
        "peak_dbfs": db(peak),
        "rms_dbfs": db(rms),
        "clip_frac": float(np.mean(a >= CLIP_THRESHOLD)),
    }


def build_speaker_rows(spk_dir: Path, transcripts: dict[str, str]) -> list[dict]:
    import soundfile as sf  # lazy import

    rows = []
    for p in sorted(spk_dir.rglob("*")):
        if p.suffix.lower() not in AUDIO_EXTS:
            continue
        cat, label, text_id = classify(p.stem)
        x, sr = sf.read(str(p), dtype="float32", always_2d=False)
        if x.ndim > 1:  # EARS is mono; guard anyway
            x = x.mean(axis=1)
        text = transcripts.get(p.stem, "")
        has_tx = p.stem in transcripts
        is_speech = cat in SPEECH_CATEGORIES
        rows.append(
            {
                "speaker": spk_dir.name,
                "stem": p.stem,
                "path": str(p.relative_to(config.raw_dir())),
                "category": cat,
                "label": label,
                "text_id": text_id,
                "text": text,
                "has_transcript": has_tx,
                "is_speech": is_speech,
                "needs_asr": is_speech and not has_tx,
                "duration_s": len(x) / sr,
                "sr": sr,
                **audio_stats(x),
            }
        )
    return rows


def _worker(args: tuple[str, str]) -> str:
    import pandas as pd

    spk_dir, part_path = args
    transcripts = json.loads((config.raw_dir() / "transcripts.json").read_text())
    rows = build_speaker_rows(Path(spk_dir), transcripts)
    pd.DataFrame(rows).to_csv(part_path, index=False)
    return Path(spk_dir).name


def _group_key(r) -> str:
    if r["category"] in ("reading", "nonspeech"):
        return f"{r['category']}/{r['label']}"
    return r["category"]


def main() -> None:
    import pandas as pd

    ap = argparse.ArgumentParser()
    ap.add_argument("--speakers", type=int, nargs="*", default=[])
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--force", action="store_true", help="recompute cached speakers")
    args = ap.parse_args()

    raw = config.raw_dir()
    if not (raw / "transcripts.json").exists():
        raise SystemExit("Missing transcripts.json. Run the download step with --meta-only first.")
    splits_path = config.processed_dir() / "splits.json"
    if not splits_path.exists():
        raise SystemExit("Missing splits.json. Run: python -m efforttts.data.splits")
    splits = json.loads(splits_path.read_text())

    if args.speakers:
        dirs = [raw / f"p{i:03d}" for i in args.speakers]
    else:
        dirs = sorted(d for d in raw.glob("p[0-9][0-9][0-9]") if d.is_dir())
    dirs = [d for d in dirs if d.is_dir()]
    if not dirs:
        raise SystemExit(f"No speaker folders under {raw}")

    parts_dir = config.processed_dir() / "manifest_parts"
    parts_dir.mkdir(parents=True, exist_ok=True)
    todo = []
    for d in dirs:
        part = parts_dir / f"{d.name}.csv"
        if args.force or not part.exists():
            todo.append((str(d), str(part)))
    print(f"{len(dirs)} speakers, {len(todo)} to process, {len(dirs) - len(todo)} cached")

    if todo:
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            for name in ex.map(_worker, todo):
                print(f"[done] {name}")

    df = pd.concat([pd.read_csv(parts_dir / f"{d.name}.csv", keep_default_na=False) for d in dirs])
    df["split"] = df["speaker"].map(splits)
    out = config.processed_dir() / "manifest.csv"
    df.to_csv(out, index=False)

    pd.set_option("display.width", 160)
    print(f"\nManifest: {len(df)} files, {df.speaker.nunique()} speakers -> {out}")
    print("\n== Speakers / files per split ==")
    print(df.groupby("split", dropna=False).agg(speakers=("speaker", "nunique"), files=("stem", "size")).to_string())

    df["group"] = df.apply(_group_key, axis=1)
    g = df.groupby("group").agg(
        files=("stem", "size"),
        med_peak_dbfs=("peak_dbfs", "median"),
        med_rms_dbfs=("rms_dbfs", "median"),
        clip_pct=("clip_frac", lambda s: 100 * s.mean()),
        any_clipped=("clip_frac", lambda s: int((s > 0).sum())),
    )
    print("\n== Level as recorded, by group (sorted by median RMS) ==")
    print(g.sort_values("med_rms_dbfs").round(2).to_string())

    over = int((df.peak_dbfs > 0).sum())
    print(f"\nFiles with peak above full scale (>0 dBFS): {over}")
    worst = df.sort_values("clip_frac", ascending=False).head(5)
    print("Most clipped files:")
    print(worst[["speaker", "stem", "peak_dbfs", "clip_frac"]].round(4).to_string(index=False))

    print(f"\nSpeech files needing ASR (no transcript): {int(df.needs_asr.sum())} "
          f"({df[df.needs_asr].duration_s.sum() / 3600:.2f} h)")


if __name__ == "__main__":
    main()