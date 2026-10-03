"""Download EARS speakers and metadata into config.raw_dir().

Idempotent: speakers already extracted are skipped. Zips are deleted after
extraction to save disk.

    python -m efforttts.data.download --meta-only
    python -m efforttts.data.download --speakers 1 2 3
    python -m efforttts.data.download --all

EARS is CC-NC 4.0 (non-commercial). Cite the paper if you use it.
"""
from __future__ import annotations

import argparse
import shutil
import tempfile
import urllib.request
import zipfile
from pathlib import Path

from efforttts import config

N_SPEAKERS = 107
RELEASE_URL = (
    "https://github.com/facebookresearch/ears_dataset/releases/download/dataset/p{sid:03d}.zip"
)
META_URLS = {
    "transcripts.json": (
        "https://github.com/facebookresearch/ears_dataset/raw/refs/heads/main/transcripts.json"
    ),
    "speaker_statistics.json": (
        "https://raw.githubusercontent.com/facebookresearch/ears_dataset/main/speaker_statistics.json"
    ),
}


def _fetch(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    with urllib.request.urlopen(url) as resp, open(tmp, "wb") as f:
        shutil.copyfileobj(resp, f, length=1 << 20)
    tmp.replace(dest)


def download_metadata() -> None:
    for name, url in META_URLS.items():
        dest = config.raw_dir() / name
        if dest.exists():
            print(f"[skip] {name}")
            continue
        print(f"[get ] {name}")
        _fetch(url, dest)


def download_speaker(sid: int) -> Path:
    """Download and extract one speaker to raw_dir()/pXXX. Returns the folder."""
    name = f"p{sid:03d}"
    final = config.raw_dir() / name
    if final.exists() and any(final.iterdir()):
        print(f"[skip] {name} already extracted")
        return final

    with tempfile.TemporaryDirectory(dir=config.raw_dir()) as td:
        zpath = Path(td) / f"{name}.zip"
        print(f"[get ] {name} ...")
        _fetch(RELEASE_URL.format(sid=sid), zpath)
        extract_dir = Path(td) / "x"
        with zipfile.ZipFile(zpath) as zf:
            zf.extractall(extract_dir)
        zpath.unlink()
        # The zip may contain a top-level pXXX/ folder or loose files.
        src = extract_dir / name if (extract_dir / name).is_dir() else extract_dir
        if final.exists():
            shutil.rmtree(final)
        shutil.move(str(src), str(final))
    print(f"[done] {name}")
    return final


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--speakers", type=int, nargs="*", default=[], help="speaker numbers, e.g. 1 2 3")
    ap.add_argument("--all", action="store_true", help=f"all {N_SPEAKERS} speakers")
    ap.add_argument("--meta-only", action="store_true", help="only transcripts + speaker stats")
    args = ap.parse_args()

    download_metadata()
    if args.meta_only:
        return
    ids = list(range(1, N_SPEAKERS + 1)) if args.all else args.speakers
    bad = [i for i in ids if not 1 <= i <= N_SPEAKERS]
    if bad:
        raise SystemExit(f"Speaker numbers must be 1..{N_SPEAKERS}: {bad}")
    for sid in ids:
        download_speaker(sid)


if __name__ == "__main__":
    main()