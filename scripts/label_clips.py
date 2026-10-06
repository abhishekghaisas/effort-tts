#!/usr/bin/env python3
"""Blind listening-label helper for generated clips (macOS: uses `afplay`).

Plays each clip at one exaggeration setting in a fixed random order and records what
you hear. Scores (effort, WER, similarity, transcript) are deliberately NOT shown, so
your labels stay independent of the metrics. Progress is saved after every clip and the
session resumes where you stopped.

Run from the repo root (project env is fine):
    python label_clips.py                      # exaggeration 1.6, all clips
    python label_clips.py --limit 10           # quick trial
    python label_clips.py --audio-dir PATH     # if the clips are not found automatically

Keys (single keypress, no Enter needed except where noted):
    o   ok: delivery fits the line, speech is clear           (confirms immediately)
    c   sounds CALM but the line is meant to be loud/urgent   (toggle)
    l   sounds LOUD but the line is meant to be calm          (toggle)
    d   duplicated speech (a phrase is repeated)              (toggle)
    u   unintelligible / garbled                              (toggle)
    Enter   confirm the toggled labels (e.g. press d then u, then Enter)
    r   replay            b   go back one clip and relabel it        q   quit (saved)
A clip can carry several labels. "ok" cannot be combined with others.
"""
import argparse
import csv
import os
import random
import subprocess
import sys
import termios
import time
import tty
from pathlib import Path

import pandas as pd

BASE = Path("data_local/baselines/chatterbox/cfg0.50")
LABELS = {
    "c": "calm_when_loud",
    "l": "loud_when_calm",
    "d": "duplicated",
    "u": "unintelligible",
}
FIELDS = ["job_id", "speaker", "line_id", "intent", "exaggeration", "labels", "seconds"]


def resolve_audio(rel, audio_dir):
    """Find the wav for a relative path from gap_clips.csv."""
    roots = [audio_dir] if audio_dir else [BASE, BASE / "audio", BASE / "wavs", BASE / "wav", BASE / "clips"]
    for r in roots:
        p = Path(r) / rel
        if p.exists():
            return p
    root = Path(audio_dir) if audio_dir else BASE
    hits = list(root.rglob(Path(rel).name)) if root.exists() else []
    hits = [h for h in hits if str(h).endswith(str(rel))]
    return hits[0] if hits else None


def load_done(out_path):
    done = {}
    if out_path.exists():
        with open(out_path, newline="") as f:
            for row in csv.DictReader(f):
                done[row["job_id"]] = row
    return done


def save(out_path, order, done):
    tmp = out_path.with_suffix(".tmp")
    with open(tmp, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        for jid in order:
            if jid in done:
                w.writerow({k: done[jid].get(k, "") for k in FIELDS})
    os.replace(tmp, out_path)


def key():
    return os.read(sys.stdin.fileno(), 1).decode(errors="ignore")


def play(path):
    return subprocess.Popen(["afplay", str(path)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def stop(proc):
    if proc is not None and proc.poll() is None:
        proc.terminate()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", default=str(BASE / "gap_clips.csv"))
    ap.add_argument("--exaggeration", type=float, default=1.6)
    ap.add_argument("--audio-dir", default=None)
    ap.add_argument("--seed", type=int, default=0, help="fixed shuffle so resuming keeps the same order")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    d = pd.read_csv(a.csv)
    d = d[d["exaggeration"].round(2) == round(a.exaggeration, 2)].copy()
    if "status" in d.columns:
        d = d[d["status"] == "ok"]
    if d.empty:
        sys.exit(f"No clips at exaggeration {a.exaggeration} in {a.csv}")
    rows = d.to_dict("records")
    random.Random(a.seed).shuffle(rows)
    if a.limit:
        rows = rows[: a.limit]
    order = [r["job_id"] for r in rows]

    out = Path(a.out) if a.out else BASE / f"listening_labels_ex{a.exaggeration:.2f}.csv"
    done = load_done(out)
    print(f"{len(rows)} clips, {sum(j in done for j in order)} already labeled -> {out}")

    missing = [r for r in rows[:3] if resolve_audio(r["path"], a.audio_dir) is None]
    if missing:
        sys.exit(f"Cannot find audio such as {missing[0]['path']}; pass --audio-dir PATH")

    i = 0
    while i < len(rows) and rows[i]["job_id"] in done:
        i += 1

    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    proc = None
    try:
        tty.setcbreak(fd)
        while i < len(rows):
            r = rows[i]
            path = resolve_audio(r["path"], a.audio_dir)
            if path is None:
                print(f"[skip] audio not found: {r['path']}")
                i += 1
                continue
            sel = set()
            t0 = time.time()
            stop(proc)
            proc = play(path)
            print(f"\n[{i + 1}/{len(rows)}] intended: {str(r['intent']).upper()}")
            print(f"    \"{r['text']}\"")
            print("    keys: o ok | c calm-when-loud | l loud-when-calm | d duplicated | "
                  "u unintelligible | Enter confirm | r replay | b back | q quit")
            go_back = False
            while True:
                k = key()
                if k in ("q", "\x03"):
                    stop(proc)
                    save(out, order, done)
                    print("\nSaved. Quitting.")
                    return summary(done, order)
                if k == "r":
                    stop(proc)
                    proc = play(path)
                elif k == "b":
                    go_back = True
                    break
                elif k == "o":
                    sel = set()
                    labels = "ok"
                    break
                elif k in LABELS:
                    sel ^= {LABELS[k]}
                    print("    selected:", ", ".join(sorted(sel)) or "(none)")
                elif k in ("\n", "\r"):
                    if sel:
                        labels = "|".join(sorted(sel))
                        break
                    print("    nothing selected; press o for ok or toggle a label first")
            if go_back:
                stop(proc)
                if i > 0:
                    i -= 1
                    done.pop(rows[i]["job_id"], None)
                    save(out, order, done)
                    print("    (back one clip)")
                continue
            done[r["job_id"]] = {
                "job_id": r["job_id"], "speaker": r["speaker"], "line_id": r["line_id"],
                "intent": r["intent"], "exaggeration": r["exaggeration"],
                "labels": labels, "seconds": round(time.time() - t0, 1),
            }
            save(out, order, done)
            print(f"    -> {labels}")
            i += 1
        print("\nAll clips labeled.")
    finally:
        stop(proc)
        termios.tcsetattr(fd, termios.TCSADRAIN, old)
    summary(done, order)


def summary(done, order):
    n = sum(j in done for j in order)
    counts = {}
    for j in order:
        if j in done:
            for lab in done[j]["labels"].split("|"):
                counts[lab] = counts.get(lab, 0) + 1
    print(f"\n{n} labeled. Counts:", counts)


if __name__ == "__main__":
    main()