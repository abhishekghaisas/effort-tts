#!/usr/bin/env python3
"""Report which Effort-tts steps have been executed (read-only; prints aggregates only).

Run from the repo root (either environment is fine, pandas is needed):
    python scripts/check_status.py
"""

import os
import subprocess
from pathlib import Path

import pandas as pd

DATA = Path(os.environ.get("EFFORT_DATA_DIR", "data_local"))
B = DATA / "baselines" / "chatterbox" / "cfg0.50"
SCRIPTS = [
    "inspect_exag16.py",
    "label_clips.py",
    "analyze_labels.py",
    "analyze_labels_deep.py",
    "analyze_seeds.py",
]
todo = []


def line(ok, msg, fix=None):
    print(("[ok]   " if ok else "[TODO] ") + msg)
    if not ok and fix:
        todo.append(fix)
    return ok


def read(p):
    return pd.read_csv(p) if Path(p).exists() else None


def mtime(p):
    return Path(p).stat().st_mtime if Path(p).exists() else 0


print(f"Data folder: {B}\n")

print("== Repo and files ==")
txt = Path("README.md").read_text() if Path("README.md").exists() else ""
last = next((l for l in txt.splitlines() if "Last updated" in l), "")
line(
    "2026-10-09" in last,
    f"README is the latest version ({last.strip() or 'no date found'})",
    "copy the latest README.md into the repo root",
)
for s in SCRIPTS:
    line(Path("scripts", s).exists(), f"scripts/{s} present", f"put {s} in scripts/")
lc = Path("scripts/label_clips.py")
line(
    lc.exists() and "--min-seed" in lc.read_text(),
    "label_clips.py is the patched version (has --min-seed)",
    "replace scripts/label_clips.py with the patched version",
)
pl = read("configs/line_plausibility.csv")
line(
    pl is not None and pl["loud_plausible"].notna().sum() == 10,
    "configs/line_plausibility.csv rated (10 calm lines)",
    "rate configs/line_plausibility.csv",
)

print("\n== Generation (expect 1050 rows, all ok; 1.6 has seeds 0, 1, 2 with 150 each) ==")
r = read(B / "results.csv")
if r is None:
    line(False, "results.csv missing", "run chatterbox_gen")
else:
    n_ok = int((r["status"] == "ok").sum())
    line(
        len(r) == 1050 and n_ok == len(r),
        f"results.csv: {len(r)} rows, {n_ok} ok",
        "finish generation: python -m efforttts.baselines.chatterbox_gen --exaggerations 1.6 --seeds 3",
    )
    x = r[r["exaggeration"].astype(float).round(2) == 1.6]
    print("       exaggeration 1.6 clips per seed:", x.groupby("seed").size().to_dict())

print("\n== Scoring (Step A then Step B) ==")
g = read(B / "gap_clips.csv")
aps = read(B / "asr_speaker.csv")
n_res = 0 if r is None else len(r)
step_a_done = aps is not None and n_res > 0 and len(aps) >= n_res
fix_scoring = (
    "run Step B (project env): python -m efforttts.eval.score_baseline"
    if step_a_done
    else "run Step A (chatterbox env: python -m efforttts.eval.asr_speaker), then Step B "
    "(project env: python -m efforttts.eval.score_baseline)"
)
if aps is not None:
    print(
        f"       asr_speaker.csv rows: {len(aps)} (Step A output; {'covers all clips' if step_a_done else 'incomplete'})"
    )
if g is None:
    line(False, "gap_clips.csv missing", fix_scoring)
else:
    scored = int(g["wer"].notna().sum()) if "wer" in g else 0
    ok_rows = line(
        len(g) == 1050 and scored == len(g),
        f"gap_clips.csv: {len(g)} rows, {scored} with WER/similarity",
        fix_scoring,
    )
    x = g[g["exaggeration"].round(2) == 1.6]
    print("       1.6 clips per seed in gap_clips.csv:", x.groupby("seed").size().to_dict())
    stale = mtime(B / "gap_clips.csv") < mtime(B / "results.csv")
    line(
        not stale,
        "gap_clips.csv is newer than results.csv",
        "rerun Step B (scores are older than the clips)" if ok_rows else None,
    )

print("\n== Listening labels ==")
for f, want in [
    ("listening_labels_ex1.60.csv", 150),
    ("listening_labels_ex1.60_backup.csv", 150),
    ("listening_labels_seeds_ex1.60.csv", 100),
]:
    d = read(B / f)
    n = 0 if d is None else len(d)
    note = {
        "listening_labels_ex1.60.csv": "original labels",
        "listening_labels_ex1.60_backup.csv": "backup before the new seeds",
        "listening_labels_seeds_ex1.60.csv": "new seeds of 10 focus lines",
    }[f]
    line(
        n >= want,
        f"{f}: {'missing' if d is None else n} of {want} ({note})",
        None
        if f.endswith("backup.csv")
        else "label_clips.py --lines C09 C02 C04 C10 S03 S05 C05 S01 U01 C07 --min-seed 1 --out "
        f"{B}/listening_labels_seeds_ex1.60.csv"
        if "seeds" in f
        else None,
    )

print("\n== Git ==")
for cmd in (["git", "status", "-sb"], ["git", "remote", "-v"], ["git", "log", "--oneline", "-3"]):
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=10).stdout.strip()
        print("  $ " + " ".join(cmd) + "\n    " + out.replace("\n", "\n    "))
    except Exception as e:  # noqa: BLE001
        print("  could not run", " ".join(cmd), e)

print("\n== Pending actions, in order ==")
if todo:
    for i, t in enumerate(todo, 1):
        print(f"{i}. {t}")
    print("then: python scripts/analyze_seeds.py and paste the output")
else:
    print("All checks pass: run python scripts/analyze_seeds.py and paste the output.")
