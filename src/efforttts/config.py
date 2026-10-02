"""Path configuration shared by Mac and Colab.

Set EFFORT_DATA_DIR to relocate all heavy data (raw audio, processed
features, checkpoints). Default is ./data_local (git-ignored).
On Colab, point it at Google Drive, e.g.
    /content/drive/MyDrive/effort-tts-data
"""
from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def data_root() -> Path:
    root = Path(os.environ.get("EFFORT_DATA_DIR", REPO_ROOT / "data_local"))
    root.mkdir(parents=True, exist_ok=True)
    return root


def _sub(name: str) -> Path:
    p = data_root() / name
    p.mkdir(parents=True, exist_ok=True)
    return p


def raw_dir() -> Path:
    return _sub("ears_raw")


def processed_dir() -> Path:
    return _sub("ears_processed")


def checkpoints_dir() -> Path:
    return _sub("checkpoints")


def outputs_dir() -> Path:
    return _sub("outputs")
