"""Scan downloaded EARS speakers and compare against transcripts.json.

    python -m efforttts.data.inventory            # all speakers present
    python -m efforttts.data.inventory --speakers 1 2 3

Writes processed_dir()/inventory.csv and prints a summary. Answers:
  * do file stems match transcript keys?
  * how are freeform / other files named?
  * is every speaker complete?
  * audio format facts (sample rate, bit depth, channels).
Only reads audio headers, so it is fast.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from efforttts import config

AUDIO_EXTS = {".wav", ".flac", ".opus", ".ogg"}
READING_RE = re.compile(r"^((?:rainbow|sentences)_\d+)_([a-z]+)$")
EMO_RE = re.compile(r"^emo_([a-z]+)_sentences$")
EMO_FREE_RE = re.compile(r"^emo_([a-z]+)_freeform$")
FREE_RE = re.compile(r"^freeform_speech_\d+$")
INTERJ_RE = re.compile(r"^interjection_([a-z_]+)$")
NONSPEECH_RE = re.compile(r"^(?:nonverbal|vegetative|melodic)_([a-z_]+)$")


def classify(stem: str) -> tuple[str, str | None, str | None]:
    """Return (category, label, text_id) for a file stem.

    reading          : label = style (loud, regular, ...), text_id = e.g. 'rainbow_01'
    emotion          : scripted emotional sentences, label = emotion
    emotion_freeform : improvised emotional speech (no transcript), label = emotion
    freeform         : freeform monologue (no transcript)
    interjection     : interjection_* files, label = name
    nonspeech        : nonverbal_* / vegetative_* / melodic_* (screaming, coughing, ...)
    other            : anything unrecognized
    """
    m = READING_RE.match(stem)
    if m:
        return "reading", m.group(2), m.group(1)
    m = EMO_RE.match(stem)
    if m:
        return "emotion", m.group(1), None
    m = EMO_FREE_RE.match(stem)
    if m:
        return "emotion_freeform", m.group(1), None
    if FREE_RE.match(stem):
        return "freeform", None, None
    m = INTERJ_RE.match(stem)
    if m:
        return "interjection", m.group(1), None
    m = NONSPEECH_RE.match(stem)
    if m:
        return "nonspeech", m.group(1), None
    return "other", None, None


def scan_speaker(spk_dir: Path, transcript_keys: set[str]) -> list[dict]:
    import soundfile as sf  # lazy: keeps classify() importable without audio libs

    rows = []
    for p in sorted(spk_dir.rglob("*")):
        if p.suffix.lower() not in AUDIO_EXTS:
            continue
        cat, label, text_id = classify(p.stem)
        row = {
            "speaker": spk_dir.name,
            "stem": p.stem,
            "path": str(p.relative_to(config.raw_dir())),
            "category": cat,
            "label": label,
            "text_id": text_id,
            "in_transcripts": p.stem in transcript_keys,
        }
        try:
            info = sf.info(str(p))
            row.update(
                duration_s=info.duration,
                sr=info.samplerate,
                channels=info.channels,
                subtype=info.subtype,
                error="",
            )
        except Exception as e:  # noqa: BLE001 - report, don't crash the scan
            row.update(duration_s=None, sr=None, channels=None, subtype=None, error=str(e)[:120])
        rows.append(row)
    return rows


def main() -> None:
    import pandas as pd

    ap = argparse.ArgumentParser()
    ap.add_argument("--speakers", type=int, nargs="*", default=[])
    args = ap.parse_args()

    raw = config.raw_dir()
    tpath = raw / "transcripts.json"
    if not tpath.exists():
        raise SystemExit("transcripts.json missing. Run: python -m efforttts.data.download --meta-only")
    keys = set(json.loads(tpath.read_text()))

    if args.speakers:
        dirs = [raw / f"p{i:03d}" for i in args.speakers]
    else:
        dirs = sorted(d for d in raw.glob("p[0-9][0-9][0-9]") if d.is_dir())
    dirs = [d for d in dirs if d.is_dir()]
    if not dirs:
        raise SystemExit(f"No speaker folders found under {raw}")

    rows = []
    for d in dirs:
        rows += scan_speaker(d, keys)
    df = pd.DataFrame(rows)
    out = config.processed_dir() / "inventory.csv"
    df.to_csv(out, index=False)

    pd.set_option("display.width", 160)
    print(f"\nSpeakers scanned: {len(dirs)}   audio files: {len(df)}   -> {out}")
    print(f"Transcript keys expected per speaker: {len(keys)}")

    print("\n== Audio format ==")
    print(df.groupby(["sr", "channels", "subtype"], dropna=False).size().to_string())
    if df["error"].astype(bool).any():
        print(f"\n!! {int(df['error'].astype(bool).sum())} files failed to read, e.g.:")
        print(df.loc[df["error"].astype(bool), ["path", "error"]].head(5).to_string(index=False))

    print("\n== By category (files, total hours) ==")
    g = df.groupby("category")["duration_s"].agg(["count", lambda s: s.sum() / 3600])
    g.columns = ["files", "hours"]
    print(g.round(2).to_string())

    print("\n== Reading files per style (mean per speaker) ==")
    rd = df[df.category == "reading"]
    if len(rd):
        per = rd.groupby(["speaker", "label"]).size().unstack(fill_value=0)
        print(per.mean().round(1).to_string())
        print("\n== Reading minutes per style (mean per speaker) ==")
        mins = rd.groupby("label")["duration_s"].sum() / 60 / len(dirs)
        print(mins.round(2).to_string())

    print("\n== Completeness vs transcripts.json ==")
    have = df[df.in_transcripts].groupby("speaker")["stem"].nunique()
    for d in dirs:
        n = int(have.get(d.name, 0))
        missing = sorted(keys - set(df[(df.speaker == d.name)]["stem"]))
        flag = "" if not missing else f"  missing {len(missing)}: {missing[:4]}{'...' if len(missing) > 4 else ''}"
        print(f"{d.name}: {n}/{len(keys)}{flag}")

    unk = df[df.category == "other"]
    print(f"\n== Unrecognized stems ({unk.stem.nunique()} unique) ==")
    if len(unk):
        print(unk.stem.value_counts().to_string())
    transcribed = df.category.isin(["reading", "emotion"])
    stray = df[transcribed & (~df.in_transcripts)]
    if len(stray):
        print(f"\n!! {len(stray)} classified files with stems NOT in transcripts.json, e.g.:")
        print(stray.stem.drop_duplicates().head(10).to_string(index=False))


if __name__ == "__main__":
    main()