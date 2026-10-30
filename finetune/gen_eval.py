#!/usr/bin/env python3
"""Generate the fixed 30-line evaluation set with the fine-tuned (or stock) Chatterbox on Colab.

Output layout matches the baseline generator: <out>/<tag>/<speaker>/<line>_ex<scalar>_s<seed>.wav and
<out>/<tag>/results.csv with columns job_id, speaker, line_id, intent, text, exaggeration, cfg_weight, seed,
path, audio_s, gen_s, status. Resumable: finished clips are skipped.

Example (fine-tune, stage 1, one seed):
  python gen_eval.py --adapter DRIVE/output/new_lang_adapter --tag effort_ft --scalars 0.5 0.8 1.2 1.6 2.0 \
      --seeds 0 --refs DRIVE/refs_val --lines eval_lines.json --out DRIVE/eval
Stock control: --adapter none --tag stock_t4
Speaker ids are read from the reference file names (first pNNN in the name).
"""

import argparse
import csv
import json
import os
import re
import sys
import time
from pathlib import Path

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


def find_refs(refs_dir, only=None):
    refs = {}
    for f in sorted(Path(refs_dir).glob("*.wav")):
        m = re.search(r"p\d{3}", f.name)
        if m and (not only or m.group(0) in only):
            refs[m.group(0)] = f
    return refs


def job_id(spk, line, ex, seed):
    return f"{spk}/{line}_ex{ex:.2f}_s{seed}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--adapter", required=True, help="LoRA adapter folder, or 'none' for stock Chatterbox"
    )
    ap.add_argument("--tag", required=True)
    ap.add_argument("--scalars", type=float, nargs="+", required=True)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--cfg-weight", type=float, default=0.5)
    ap.add_argument("--lines", default="eval_lines.json")
    ap.add_argument(
        "--refs", required=True, help="folder with one reference wav per validation speaker"
    )
    ap.add_argument("--out", required=True)
    ap.add_argument("--speakers", nargs="*", default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    lines = json.load(open(a.lines))
    refs = find_refs(a.refs, a.speakers)
    if not refs:
        sys.exit(
            f"no reference wavs with a pNNN id in the name found in {a.refs}: {[p.name for p in Path(a.refs).glob('*')]}"
        )
    root = Path(a.out) / a.tag
    root.mkdir(parents=True, exist_ok=True)
    results = root / "results.csv"
    done = set()
    if results.exists():
        with open(results, newline="") as f:
            done = {r["job_id"] for r in csv.DictReader(f) if r.get("status") == "ok"}

    jobs = []
    for spk in sorted(refs):
        for ln in lines:
            for ex in a.scalars:
                for seed in a.seeds:
                    jid = job_id(spk, ln["line_id"], ex, seed)
                    if jid not in done:
                        jobs.append((spk, ln, ex, seed, jid))
    total = len(refs) * len(lines) * len(a.scalars) * len(a.seeds)
    print(
        f"speakers {sorted(refs)}; {total} jobs, {len(done)} done, {len(jobs)} pending; output {root}"
    )
    if a.limit:
        jobs = jobs[: a.limit]
    if a.dry_run:
        print("dry run; first pending:", [j[4] for j in jobs[:3]])
        return

    import torch
    import torchaudio

    sys.path.insert(0, "/content/tk")
    os.chdir("/content/tk")
    from src.chatterbox_.tts import ChatterboxTTS

    tts = ChatterboxTTS.from_local("/content/tk/pretrained_models", device="cuda")
    if a.adapter != "none":
        from peft import PeftModel

        tts.t3 = PeftModel.from_pretrained(tts.t3, a.adapter).merge_and_unload().to("cuda").eval()
    print("model ready; adapter:", a.adapter)

    new_file = not results.exists()
    out_f = open(results, "a", newline="")
    w = csv.DictWriter(out_f, fieldnames=FIELDS)
    if new_file:
        w.writeheader()
    cur = None
    for i, (spk, ln, ex, seed, jid) in enumerate(jobs, 1):
        rel = f"{spk}/{ln['line_id']}_ex{ex:.2f}_s{seed}.wav"
        (root / spk).mkdir(exist_ok=True)
        row = {
            "job_id": jid,
            "speaker": spk,
            "line_id": ln["line_id"],
            "intent": ln["intent"],
            "text": ln["text"],
            "exaggeration": ex,
            "cfg_weight": a.cfg_weight,
            "seed": seed,
            "path": rel,
            "audio_s": "",
            "gen_s": "",
            "status": "ok",
        }
        t0 = time.time()
        try:
            if cur != spk:
                tts.prepare_conditionals(str(refs[spk]), exaggeration=ex)
                cur = spk
            torch.manual_seed(seed)
            wav = tts.generate(ln["text"], exaggeration=ex, cfg_weight=a.cfg_weight)
            if torch.isnan(wav).any():
                raise RuntimeError("NaN in output")
            torchaudio.save(str(root / rel), wav.cpu(), tts.sr)
            row["audio_s"] = round(wav.shape[-1] / tts.sr, 3)
        except Exception as e:
            row["status"] = "error: " + str(e)[:100]
        row["gen_s"] = round(time.time() - t0, 2)
        w.writerow(row)
        out_f.flush()
        print(f"[{i}/{len(jobs)}] {jid} {row['status']} {row['gen_s']}s")
    out_f.close()


if __name__ == "__main__":
    main()
