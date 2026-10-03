"""Pick clean reference clips for zero-shot TTS from EARS freeform speech.

    python -m efforttts.baselines.references                # val speakers
    python -m efforttts.baselines.references --speakers 23 45

For each speaker, scans every freeform_speech_* file with a sliding 10 s window
and keeps the window with the highest speech density (fewest pauses/breaths).
Writes float32 WAVs to data_root()/baselines/references/ plus references.json
recording the source file and start time. LISTEN to each reference before
trusting it: density cannot tell calm speech from laughter or coughing.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from efforttts import config

WIN_S = 10.0
HOP_S = 1.0
FRAME_S = 0.02
ACTIVE_RANGE_DB = 30.0


def best_window(x: np.ndarray, sr: int, win_s: float = WIN_S, hop_s: float = HOP_S) -> tuple[float, float]:
    """Return (start_seconds, speech_density) of the densest win_s window.

    Density = fraction of 20 ms frames within ACTIVE_RANGE_DB of the clip's 95th-percentile level.
    """
    fr = int(FRAME_S * sr)
    n = len(x) // fr
    if n * FRAME_S < win_s:
        raise ValueError(f"clip is {len(x) / sr:.1f}s, shorter than the {win_s:.0f}s window")
    db = 10 * np.log10(np.mean(x[: n * fr].reshape(n, fr) ** 2, axis=1) + 1e-12)
    active = db >= np.percentile(db, 95) - ACTIVE_RANGE_DB
    w, h = int(round(win_s / FRAME_S)), int(round(hop_s / FRAME_S))
    starts = np.arange(0, n - w + 1, h)
    dens = np.array([active[s : s + w].mean() for s in starts])
    i = int(np.argmax(dens))  # first maximum wins ties
    return float(starts[i] * FRAME_S), float(dens[i])


def freeform_files(speaker: str, manifest_path: Path | None = None) -> list[Path]:
    manifest_path = manifest_path or config.processed_dir() / "manifest.csv"
    out = []
    with open(manifest_path, newline="") as f:
        for r in csv.DictReader(f):
            if r["speaker"] == speaker and r["category"] == "freeform":
                out.append(config.raw_dir() / r["path"])
    return sorted(out)


def build_references(speakers: list[str], out_dir: Path | None = None) -> dict:
    import soundfile as sf  # lazy

    out_dir = out_dir or config.data_root() / "baselines" / "references"
    out_dir.mkdir(parents=True, exist_ok=True)
    info = {}
    for spk in speakers:
        files = freeform_files(spk)
        if not files:
            raise SystemExit(f"No freeform files found for {spk}; download it and rebuild the manifest.")
        best = None
        for p in files:
            x, sr = sf.read(str(p), dtype="float32", always_2d=False)
            if x.ndim > 1:
                x = x.mean(axis=1)
            try:
                start, dens = best_window(x, sr)
            except ValueError:
                continue
            if best is None or dens > best["density"]:
                best = {"source": str(p.relative_to(config.raw_dir())), "start_s": start,
                        "density": dens, "sr": sr, "audio": x}
        if best is None:
            raise SystemExit(f"No freeform clip for {spk} is long enough.")
        a = int(best["start_s"] * best["sr"])
        seg = best.pop("audio")[a : a + int(WIN_S * best["sr"])]
        sf.write(str(out_dir / f"{spk}.wav"), seg, best["sr"], subtype="FLOAT")
        info[spk] = best
        print(f"{spk}: {best['source']} @ {best['start_s']:.0f}s  speech density {best['density']:.2f}")
    (out_dir / "references.json").write_text(json.dumps(info, indent=1))
    return info


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--speakers", type=int, nargs="*", default=[])
    args = ap.parse_args()
    if args.speakers:
        speakers = [f"p{i:03d}" for i in args.speakers]
    else:
        splits = json.loads((config.processed_dir() / "splits.json").read_text())
        speakers = sorted(s for s, v in splits.items() if v == "val")
    build_references(speakers)


if __name__ == "__main__":
    main()