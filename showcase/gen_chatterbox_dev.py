#!/usr/bin/env python3
"""Stock Chatterbox clips for the 8 dev lines, 3 dev speakers, same references as Qwen, in the harness layout (Mac, CHATTERBOX env, repo root):
    python showcase/gen_chatterbox_dev.py
About 120 clips (3 speakers x 8 lines x 5 exaggeration settings), roughly 10 to 15 minutes on MPS. Then Step A and Step B with --default-ex 0.5:
    python -m efforttts.eval.asr_speaker --root data_local/baselines/chatterbox_dev/cfg0.50
    python -m efforttts.eval.score_baseline --root data_local/baselines/chatterbox_dev/cfg0.50 --default-ex 0.5
Chatterbox's watermark stays on. Output is EARS-derived voices: local only. Resumable.
"""

import argparse
import csv
import json
import re
import time
from pathlib import Path

import torch
import torchaudio

FIELDS = [
    "job_id",
    "speaker",
    "line_id",
    "intent",
    "text",
    "exaggeration",
    "cfg_weight",
    "seed",
    "path",
    "audio_s",
    "gen_s",
    "status",
]
ap = argparse.ArgumentParser()
ap.add_argument("--lines", default="qwen_v2/dev_lines.json")
ap.add_argument("--refs", default="data_local/baselines/references")
ap.add_argument("--speakers", nargs="*", default=["p023", "p045", "p056"])
ap.add_argument("--exaggerations", type=float, nargs="+", default=[0.3, 0.5, 0.8, 1.2, 1.6])
ap.add_argument("--cfg-weight", type=float, default=0.5)
ap.add_argument("--seeds", type=int, nargs="+", default=[0])
ap.add_argument("--out", default="data_local/baselines/chatterbox_dev/cfg0.50")
ap.add_argument("--dry-run", action="store_true")
a = ap.parse_args()

lines = json.load(open(a.lines))
refs = {
    m.group(0): f
    for f in sorted(Path(a.refs).glob("*.wav"))
    if (m := re.search(r"p\d{3}", f.name)) and m.group(0) in a.speakers
}
root = Path(a.out)
root.mkdir(parents=True, exist_ok=True)
results = root / "results.csv"
done = set()
if results.exists():
    with open(results, newline="") as f:
        done = {r["job_id"] for r in csv.DictReader(f) if r.get("status") == "ok"}
jobs = [
    (s, ln, ex, sd)
    for s in sorted(refs)
    for ln in lines
    for ex in a.exaggerations
    for sd in a.seeds
    if f"{s}/{ln['line_id']}_ex{ex:.2f}_s{sd}" not in done
]
print(f"speakers {sorted(refs)}; {len(jobs)} pending jobs ({len(done)} done); output {root}")
if a.dry_run or not jobs:
    raise SystemExit(0)

device = "mps" if torch.backends.mps.is_available() else "cpu"
_orig_load = torch.load  # the usual workaround for loading Chatterbox checkpoints on a Mac
torch.load = lambda *args, **kw: _orig_load(*args, **{**kw, "map_location": torch.device(device)})
from chatterbox.tts import ChatterboxTTS

model = ChatterboxTTS.from_pretrained(device=device)
new = not results.exists()
fh = open(results, "a", newline="")
w = csv.DictWriter(fh, fieldnames=FIELDS)
if new:
    w.writeheader()
for i, (spk, ln, ex, sd) in enumerate(jobs, 1):
    rel = f"{spk}/{ln['line_id']}_ex{ex:.2f}_s{sd}.wav"
    (root / spk).mkdir(exist_ok=True)
    row = {
        "job_id": rel[:-4],
        "speaker": spk,
        "line_id": ln["line_id"],
        "intent": ln["intent"],
        "text": ln["text"],
        "exaggeration": ex,
        "cfg_weight": a.cfg_weight,
        "seed": sd,
        "path": rel,
        "audio_s": "",
        "gen_s": "",
        "status": "ok",
    }
    t0 = time.time()
    try:
        torch.manual_seed(sd)
        wav = model.generate(
            ln["text"], audio_prompt_path=str(refs[spk]), exaggeration=ex, cfg_weight=a.cfg_weight
        )
        torchaudio.save(str(root / rel), wav.cpu(), model.sr)
        row["audio_s"] = round(wav.shape[-1] / model.sr, 3)
    except Exception as e:
        row["status"] = "error: " + str(e)[:100]
    row["gen_s"] = round(time.time() - t0, 2)
    w.writerow(row)
    fh.flush()
    print(f"[{i}/{len(jobs)}] {row['job_id']} {row['status']} {row['gen_s']}s")
fh.close()
