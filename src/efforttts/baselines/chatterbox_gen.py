"""Generate the Chatterbox baseline: fixed 30-line script x exaggeration x voices x seeds.

Run inside the CHATTERBOX environment (Python 3.11) after
`pip install -e . --no-deps` from the repo root:

    python -m efforttts.baselines.references                 # once: pick reference clips
    python -m efforttts.baselines.chatterbox_gen --dry-run   # job count + time estimate
    python -m efforttts.baselines.chatterbox_gen             # run (resumable: rerun to continue)
    python -m efforttts.baselines.chatterbox_gen --seeds 3   # add seeds 1 and 2 later

Output: data_root()/baselines/chatterbox/cfg<w>/<speaker>/<line>_ex<e>_s<seed>.wav
(32-bit float, model sample rate) and results.csv (one row per attempt). Existing
finished clips are skipped. Chatterbox's watermark stays intact.
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

from efforttts import config
from efforttts.baselines.test_script import load_script, normalize

EXAGGERATIONS = [0.3, 0.5, 0.8, 1.2, 1.6]
RESULT_FIELDS = ["job_id", "speaker", "line_id", "intent", "text", "exaggeration", "cfg_weight",
                 "seed", "path", "audio_s", "gen_s", "status"]
MAX_CONSECUTIVE_FAILURES = 5


def job_id(speaker: str, line_id: str, ex: float, seed: int) -> str:
    return f"{speaker}/{line_id}_ex{ex:.2f}_s{seed}"


def build_jobs(speakers: list[str], lines: list[dict], exaggerations: list[float],
               seeds: list[int]) -> list[dict]:
    """Deterministic job list, ordered seed -> speaker -> line -> exaggeration."""
    jobs = []
    for seed in seeds:
        for spk in speakers:
            for ln in lines:
                for ex in exaggerations:
                    jobs.append({"job_id": job_id(spk, ln["id"], ex, seed), "speaker": spk,
                                 "line_id": ln["id"], "intent": ln["intent"], "text": ln["text"],
                                 "exaggeration": ex, "seed": seed})
    return jobs


def done_ids(results_path: Path, root: Path) -> set[str]:
    """Jobs already finished: status ok AND the wav file still exists."""
    if not results_path.exists():
        return set()
    done = set()
    with open(results_path, newline="") as f:
        for r in csv.DictReader(f):
            if r["status"] == "ok" and (root / r["path"]).exists():
                done.add(r["job_id"])
    return done


def estimate_hours(jobs: list[dict], words_per_s: float = 3.3, rtf: float = 2.4) -> float:
    """Rough compute time from the Mac smoke test (RTF ~2.4); an estimate, not a promise."""
    audio_s = sum(len(normalize(j["text"])) / words_per_s for j in jobs)
    return audio_s * rtf / 3600


def pick_device() -> str:
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def run(jobs: list[dict], refs_dir: Path, root: Path, cfg_weight: float) -> None:
    import soundfile as sf
    import torch

    device = pick_device()
    print(f"device: {device}; {len(jobs)} jobs to run")
    if device != "cuda":
        _orig = torch.load

        def _load(*a, **k):
            k.setdefault("map_location", torch.device(device))
            return _orig(*a, **k)

        torch.load = _load

    for spk in sorted({j["speaker"] for j in jobs}):
        if not (refs_dir / f"{spk}.wav").exists():
            raise SystemExit(f"Missing reference {refs_dir / (spk + '.wav')}. "
                             "Run: python -m efforttts.baselines.references")

    from chatterbox.tts import ChatterboxTTS

    model = ChatterboxTTS.from_pretrained(device=device)
    results_path = root / "results.csv"
    root.mkdir(parents=True, exist_ok=True)
    new_file = not results_path.exists()
    fails = 0
    with open(results_path, "a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=RESULT_FIELDS)
        if new_file:
            w.writeheader()
        for i, j in enumerate(jobs, 1):
            rel = Path(j["speaker"]) / f"{j['line_id']}_ex{j['exaggeration']:.2f}_s{j['seed']}.wav"
            row = {**j, "cfg_weight": cfg_weight, "path": str(rel), "audio_s": "", "gen_s": "", "status": ""}
            try:
                torch.manual_seed(j["seed"])
                t0 = time.time()
                wav = model.generate(j["text"], audio_prompt_path=str(refs_dir / f"{j['speaker']}.wav"),
                                     exaggeration=j["exaggeration"], cfg_weight=cfg_weight)
                dt = time.time() - t0
                x = wav.squeeze().cpu().numpy()
                (root / rel).parent.mkdir(parents=True, exist_ok=True)
                sf.write(str(root / rel), x, model.sr, subtype="FLOAT")
                row.update(audio_s=f"{len(x) / model.sr:.2f}", gen_s=f"{dt:.1f}", status="ok")
                fails = 0
            except Exception as e:  # log and continue; abort only on repeated failures
                row["status"] = f"error: {type(e).__name__}: {str(e)[:120]}"
                fails += 1
            w.writerow(row)
            fh.flush()
            print(f"[{i}/{len(jobs)}] {j['job_id']}  {row['status']}  {row['gen_s']}s")
            if fails >= MAX_CONSECUTIVE_FAILURES:
                raise SystemExit(f"{fails} failures in a row; stopping. See {results_path}")
            if device == "mps" and i % 50 == 0:
                torch.mps.empty_cache()


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--speakers", type=int, nargs="*", default=[], help="default: val speakers")
    ap.add_argument("--exaggerations", type=float, nargs="+", default=EXAGGERATIONS)
    ap.add_argument("--seeds", type=int, default=1, help="number of seeds (0..N-1)")
    ap.add_argument("--cfg-weight", type=float, default=0.5)
    ap.add_argument("--limit", type=int, default=0, help="only run the first N pending jobs")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)

    if args.speakers:
        speakers = [f"p{i:03d}" for i in args.speakers]
    else:
        splits = json.loads((config.processed_dir() / "splits.json").read_text())
        speakers = sorted(s for s, v in splits.items() if v == "val")

    lines = load_script()
    jobs = build_jobs(speakers, lines, args.exaggerations, list(range(args.seeds)))
    root = config.data_root() / "baselines" / "chatterbox" / f"cfg{args.cfg_weight:.2f}"
    done = done_ids(root / "results.csv", root)
    pending = [j for j in jobs if j["job_id"] not in done]
    if args.limit:
        pending = pending[: args.limit]
    print(f"speakers {speakers}; {len(jobs)} total jobs, {len(done & {j['job_id'] for j in jobs})} done, "
          f"{len(pending)} pending; output {root}")
    print(f"rough compute time for pending jobs: {estimate_hours(pending):.1f} h (from Mac smoke test)")
    if args.dry_run or not pending:
        return
    run(pending, config.data_root() / "baselines" / "references", root, args.cfg_weight)


if __name__ == "__main__":
    main()