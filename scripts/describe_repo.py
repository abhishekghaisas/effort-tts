#!/usr/bin/env python3
"""Describe the effort-tts data layout and module interfaces (read-only; no transcript text, no audio).

Run from the repo root (project env):
    python scripts/describe_repo.py

Prints folder structure, CSV/parquet column names, the key format of transcripts.json (text values are
replaced by their length), a few audio paths, and function names / CLI flags in src/efforttts/{data,effort}.
Paste the whole output back.
"""

import ast
import json
import os
import re
from collections import Counter
from pathlib import Path

DATA = Path(os.environ.get("EFFORT_DATA_DIR", "data_local"))
SRC = Path("src/efforttts")


def redact(v):
    if isinstance(v, str):
        return f"<str len={len(v)}>"
    if v is None or isinstance(v, (int, float, bool)):
        return v
    if isinstance(v, list):
        return [redact(x) for x in v[:2]] + (["..."] if len(v) > 2 else [])
    if isinstance(v, dict):
        return {k: redact(x) for k, x in list(v.items())[:6]}
    return type(v).__name__


print(f"== 1. Folder structure under {DATA} (files by extension, 2 levels) ==")
if not DATA.exists():
    print("data folder not found; set EFFORT_DATA_DIR or run from the repo root")
else:
    for sub in sorted(p for p in DATA.iterdir() if p.is_dir()):
        ext = Counter(f.suffix.lower() or "(none)" for f in sub.rglob("*") if f.is_file())
        print(f"{sub}/  {dict(ext.most_common(6))}")
        for sub2 in sorted(p for p in sub.iterdir() if p.is_dir())[:6]:
            ext2 = Counter(f.suffix.lower() or "(none)" for f in sub2.rglob("*") if f.is_file())
            print(f"    {sub2.name}/  {dict(ext2.most_common(4))}")
    top = [f.name for f in DATA.iterdir() if f.is_file()]
    print("files at top level:", top[:20])

print("\n== 2. Tables and JSON files (outside baselines/) ==")
for f in sorted(DATA.rglob("*")):
    if not f.is_file() or "baselines" in f.parts or f.stat().st_size > 2e9:
        continue
    try:
        if f.suffix == ".csv":
            import pandas as pd

            d = pd.read_csv(f, nrows=3)
            print(f"{f}  ({f.stat().st_size // 1024} KB)  columns: {list(d.columns)[:25]}")
        elif f.suffix == ".parquet":
            import pandas as pd

            d = pd.read_parquet(f)
            print(f"{f}  rows={len(d)}  columns: {list(d.columns)[:25]}")
        elif f.suffix == ".json":
            obj = json.load(open(f, encoding="utf-8"))
            if isinstance(obj, dict):
                keys = list(obj.keys())
                print(
                    f"{f}  dict with {len(obj)} keys; first keys: {keys[:3]}; "
                    f"first value: {redact(obj[keys[0]]) if keys else None}"
                )
            else:
                print(
                    f"{f}  {type(obj).__name__} len={len(obj)}; first: {redact(obj[0]) if obj else None}"
                )
    except Exception as e:
        print(f"{f}  (could not read: {type(e).__name__})")

print("\n== 3. A few audio paths ==")
wavs = [p for p in DATA.rglob("*.wav") if "baselines" not in p.parts]
print(f"{len(wavs)} wav files outside baselines/")
for p in wavs[:3]:
    print("  ", p)

print("\n== 4. Module interfaces ==")
for d in ("data", "effort", "baselines", "eval"):
    for f in sorted((SRC / d).glob("*.py")):
        if f.name.startswith("test_") or f.name == "__init__.py":
            continue
        try:
            src = f.read_text()
            tree = ast.parse(src)
        except Exception:
            continue
        funcs = [
            f"{n.name}({', '.join(a.arg for a in n.args.args)})"
            for n in tree.body
            if isinstance(n, ast.FunctionDef) and not n.name.startswith("_")
        ]
        flags = re.findall(r"add_argument\(\s*[\"']([^\"']+)", src)
        print(f"{f}\n    functions: {funcs[:12]}\n    CLI flags: {flags}")
