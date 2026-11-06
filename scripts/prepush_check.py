#!/usr/bin/env python3
"""Run before `git add` / `git push`. Lists files that should not leave this machine and exits with code 1 if it finds any.

    python scripts/prepush_check.py                    # default, mode single: ONE repo that also hosts the app; showcase audio is allowed
    python scripts/prepush_check.py --mode main        # the research repo: no audio, no model files, no EARS-derived data or transcripts
    python scripts/prepush_check.py --mode showcase    # the exported showcase repo: only the curated audio folders are allowed

It checks every tracked file and every untracked file that is not ignored (git ls-files), flags audio, model or array files, data folders, EARS transcripts and derived tables, files over 20 MB, and
strings that look like tokens or a hard-coded app password. Notebooks are listed for a manual look (outputs can hold paths or text).
"""

import argparse
import re
import subprocess
import sys
from pathlib import Path

AUDIO = {".wav", ".flac", ".mp3", ".ogg", ".m4a"}
BINARY = {
    ".pt",
    ".pth",
    ".npz",
    ".npy",
    ".safetensors",
    ".ckpt",
    ".joblib",
    ".pkl",
    ".bin",
    ".zip",
    ".gz",
    ".tar",
    ".tgz",
    ".h5",
}
DATA_DIRS = {
    "data_local",
    "EffortDataset",
    "chatterbox-finetuning",
    "ears_raw",
    "ears_processed",
    "preprocess",
}
NAMES = {
    "ref_transcripts.json",
    "transcripts.json",
    "tokens.npz",
    "features.csv",
    "effort_scores.csv",
    "inventory.csv",
    "asr_speaker.csv",
    "real_calibration.csv",
    "gap_clips.csv",
    "results.csv",
}
SECRET = re.compile(
    r"(ghp_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}|hf_[A-Za-z0-9]{20,}|sk-[A-Za-z0-9]{20,}|APP_PASSWORD\s*=\s*[\"'][^\"'.]{4,}[\"'])"
)
SHOWCASE_OK_DIRS = ("audio", "audio_failed")
SHOWCASE_OK_NAMES = {"manifest.csv", "metrics.csv"}


def git(*args):
    return subprocess.run(
        ["git", *args], capture_output=True, text=True, check=True
    ).stdout.splitlines()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--mode",
        choices=["single", "main", "showcase"],
        default="single",
        help="single: one repo that also hosts the app (showcase audio allowed); main: research repo without showcase outputs; showcase: exported minimal repo",
    )
    ap.add_argument(
        "--allow",
        action="append",
        default=[],
        help="path to let through even if it matches a rule (repeatable), e.g. qwen_v2/directions_06B.pt",
    )
    ap.add_argument("--max-mb", type=float, default=20.0)
    a = ap.parse_args()
    files = sorted(set(git("ls-files")) | set(git("ls-files", "--others", "--exclude-standard")))
    bad, review = [], []
    for f in files:
        p = Path(f)
        if not p.exists() or f in a.allow:
            continue
        parts = set(p.parts)
        why = []
        in_ok_dir = (a.mode == "showcase" and p.parts and p.parts[0] in SHOWCASE_OK_DIRS) or (
            a.mode == "single"
            and len(p.parts) > 1
            and p.parts[0] == "showcase"
            and p.parts[1] in SHOWCASE_OK_DIRS
        )
        if p.suffix.lower() in AUDIO and not in_ok_dir:
            why.append("audio file")
        if p.suffix.lower() in BINARY:
            why.append("model, array or archive file")
        if parts & DATA_DIRS:
            why.append("data folder")
        if p.name in NAMES and not (a.mode == "showcase" and p.name in SHOWCASE_OK_NAMES):
            why.append("EARS-derived table or transcript")
        if a.mode == "main" and (
            p.name in SHOWCASE_OK_NAMES
            and "showcase" in parts
            or (p.parts[:2] and p.parts[0] == "showcase" and p.parts[1] in SHOWCASE_OK_DIRS)
        ):
            why.append("showcase output (belongs in the showcase repo)")
        if p.stat().st_size > a.max_mb * 1e6 and not in_ok_dir:
            why.append(f"{p.stat().st_size / 1e6:.0f} MB")
        if (
            p.suffix
            in {".py", ".md", ".txt", ".toml", ".json", ".yaml", ".yml", ".sh", ".ipynb", ".csv"}
            and p.stat().st_size < 2e6
        ):
            m = SECRET.search(p.read_text(errors="ignore"))
            if m:
                why.append(f"looks like a secret: {m.group(0)[:12]}...")
        if p.suffix == ".ipynb":
            review.append(f)
        if why:
            bad.append((f, "; ".join(why)))
    groups = {}
    for f, w in bad:  # one line per reason and folder instead of one line per file
        parts = Path(f).parts
        key = (
            w,
            "/".join(parts[:2]) + "/"
            if len(parts) > 2
            else (parts[0] + "/" if len(parts) > 1 else f),
        )
        groups.setdefault(key, []).append(f)
    for (w, where), items in sorted(groups.items(), key=lambda kv: -len(kv[1])):
        extra = f"  e.g. {items[0]}" if len(items) > 1 else ""
        print(f"BLOCK  {len(items):4d} x  {where:45s} <- {w}{extra}")
    for f in review:
        print(
            f"review {f}   <- notebook: check its outputs for paths, text or tokens before pushing"
        )
    print(
        f"\n{len(files)} files checked in mode '{a.mode}': {len(bad)} to block, {len(review)} notebooks to look at."
    )
    if bad:
        print(
            "Fix: add the paths to .gitignore (and `git rm --cached <path>` if already tracked), then run this check again."
        )
    else:
        print("Nothing blocked.")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
