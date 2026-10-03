"""Speaker-disjoint train/val/test split, stratified by gender.

Needs only speaker_statistics.json (metadata for all 107 speakers), not audio,
so the split is fixed once, up front, even if you have downloaded few speakers.

    python -m efforttts.data.splits            # uses configs/data.yaml
    python -m efforttts.data.splits --show     # print raw structure of the stats file

Writes processed_dir()/splits.json: {"p001": "train", ...}
"""
from __future__ import annotations

import argparse
import json
import random
import re
from collections import defaultdict
from pathlib import Path

from efforttts import config


def _norm_id(x) -> str:
    m = re.fullmatch(r"p?0*(\d+)", str(x).strip().lower())
    if not m:
        raise ValueError(f"Unrecognized speaker id: {x!r}")
    return f"p{int(m.group(1)):03d}"


def load_speaker_stats(path: Path | None = None) -> dict[str, dict]:
    """Return {speaker_id: record}. Tolerates dict-of-records or list-of-records."""
    path = path or config.raw_dir() / "speaker_statistics.json"
    data = json.loads(Path(path).read_text())
    out: dict[str, dict] = {}
    if isinstance(data, dict) and all(isinstance(v, dict) for v in data.values()):
        for k, v in data.items():
            out[_norm_id(k)] = v
    elif isinstance(data, list):
        for rec in data:
            idkey = next(
                (k for k in rec if k.lower() in ("id", "speaker", "speaker_id", "name")), None
            )
            if idkey is None:
                raise ValueError(f"No id field in record: {str(rec)[:200]}")
            out[_norm_id(rec[idkey])] = rec
    else:
        raise ValueError(
            "Unexpected speaker_statistics.json layout. Run with --show and send me the output."
        )
    return out


def gender_of(rec: dict) -> str:
    key = next((k for k in rec if re.search(r"gender|sex", k, re.I)), None)
    if key is None:
        raise ValueError(f"No gender/sex field. Fields present: {sorted(rec)}")
    return str(rec[key]).strip().lower()


def _allocate(sizes: dict[str, int], n: int) -> dict[str, int]:
    """Split n across groups proportionally (largest remainder), capped by group size."""
    total = sum(sizes.values())
    if n > total:
        raise ValueError(f"Requested {n} speakers but only {total} available")
    exact = {g: n * s / total for g, s in sizes.items()}
    alloc = {g: int(v) for g, v in exact.items()}
    order = sorted(sizes, key=lambda g: (-(exact[g] - alloc[g]), g))
    i = 0
    while sum(alloc.values()) < n:
        g = order[i % len(order)]
        if alloc[g] < sizes[g]:
            alloc[g] += 1
        i += 1
    return alloc


def make_splits(genders: dict[str, str], n_test: int, n_val: int, seed: int) -> dict[str, str]:
    groups: dict[str, list[str]] = defaultdict(list)
    for spk in sorted(genders):
        groups[genders[spk]].append(spk)
    rng = random.Random(seed)
    for g in sorted(groups):
        rng.shuffle(groups[g])

    sizes = {g: len(v) for g, v in groups.items()}
    test_q = _allocate(sizes, n_test)
    val_q = _allocate({g: sizes[g] - test_q[g] for g in sizes}, n_val)

    split: dict[str, str] = {}
    for g, spks in groups.items():
        t, v = test_q[g], val_q[g]
        for s in spks[:t]:
            split[s] = "test"
        for s in spks[t : t + v]:
            split[s] = "val"
        for s in spks[t + v :]:
            split[s] = "train"
    return split


def main() -> None:
    import yaml

    ap = argparse.ArgumentParser()
    ap.add_argument("--show", action="store_true", help="print start of speaker_statistics.json")
    ap.add_argument("--test", type=int, default=None)
    ap.add_argument("--val", type=int, default=None)
    ap.add_argument("--seed", type=int, default=None)
    args = ap.parse_args()

    spath = config.raw_dir() / "speaker_statistics.json"
    if not spath.exists():
        raise SystemExit("Missing speaker_statistics.json. Run: python -m efforttts.data.download --meta-only")
    if args.show:
        print(spath.read_text()[:800])
        return

    cfg = yaml.safe_load((config.REPO_ROOT / "configs" / "data.yaml").read_text())
    n_test = args.test if args.test is not None else cfg["holdout_speakers"]
    n_val = args.val if args.val is not None else cfg.get("val_speakers", 5)
    seed = args.seed if args.seed is not None else cfg["seed"]

    stats = load_speaker_stats(spath)
    genders = {s: gender_of(r) for s, r in stats.items()}
    split = make_splits(genders, n_test, n_val, seed)

    out = config.processed_dir() / "splits.json"
    out.write_text(json.dumps(dict(sorted(split.items())), indent=1))

    print(f"{len(split)} speakers -> {out}  (seed={seed})")
    counts: dict[tuple[str, str], int] = defaultdict(int)
    for s, sp in split.items():
        counts[(genders[s], sp)] += 1
    for g in sorted({k[0] for k in counts}):
        row = {sp: counts.get((g, sp), 0) for sp in ("train", "val", "test")}
        print(f"  {g:>12}: {row}")


if __name__ == "__main__":
    main()