#!/usr/bin/env python3
"""Generate evaluation clips with Qwen3-TTS (optionally steered) in the v1.0 harness layout.

Output: <out>/<tag>/<speaker>/<line>_k<k>_s<seed>.wav and <out>/<tag>/results.csv with the SAME columns as the Chatterbox
runs (the steering strength k is written to the `exaggeration` column, cfg_weight is 0), so the existing scoring works:
    python -m efforttts.eval.asr_speaker --root <out>/<tag>                       (Chatterbox env)
    python -m efforttts.eval.score_baseline --root <out>/<tag> --default-ex 0     (project env)
Resumable. With no --direction (or k = 0) this is plain zero-shot cloning.

Steering (untested): forward hooks on talker.model.layers[i] add k * direction to the hidden state at DECODE steps only
(sequence length 1), so the reference prompt itself is never steered. Direction file: torch.save({"layers": {i: tensor}}).

Example (Mac CPU, zero-shot baseline subset):
    python qwen_v2/qwen_gen.py --size 0.6B --refs data_local/baselines/references \
        --ref-transcripts data_local/baselines/references/ref_transcripts.json --lines finetune/eval_lines.json \
        --tag qwen06_base --ks 0 --seeds 0 --speakers p023 p045 p056 \
        --line-ids S02 S03 S05 S08 U01 U05 U08 C02 C06 C09 --out data_local/baselines/qwen_v2
"""

import argparse
import csv
import json
import re
import time
from pathlib import Path

try:
    import numpy as np
    import soundfile as sf
    import torch
except ImportError:  # lets --dry-run plan jobs without the ML stack installed
    sf = torch = np = None

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


def find_refs(d, only=None):
    refs = {}
    for f in sorted(Path(d).glob("*.wav")):
        m = re.search(r"p\d{3}", f.name)
        if m and (not only or m.group(0) in only):
            refs[m.group(0)] = f
    return refs


class Steerer:
    """Adds k * direction[layer] to the talker's hidden state at decode steps."""

    def __init__(self, tts, direction_path, layers):
        self.k = 0.0
        self.handles = []
        if not direction_path:
            return
        blob = torch.load(direction_path, map_location="cpu")
        mods = tts.model.talker.model.layers
        for l in layers:
            v = blob["layers"][l].float()
            self.handles.append(mods[l].register_forward_hook(self._make(v)))
        print(f"steering hooks on talker layers {layers} (of {len(mods)})")

    def _make(self, v):
        def hook(module, inputs, output):
            if self.k == 0.0:
                return output
            is_tuple = isinstance(output, tuple)
            hs = output[0] if is_tuple else output
            if hs.shape[1] != 1:  # prefill of the prompt: leave untouched
                return output
            hs = hs + self.k * v.to(device=hs.device, dtype=hs.dtype)
            return (hs,) + tuple(output[1:]) if is_tuple else hs

        return hook


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", default="0.6B", choices=["0.6B", "1.7B"])
    ap.add_argument("--device", default="cpu", help="cpu | mps | cuda:0")
    ap.add_argument("--refs", required=True, help="folder with pNNN.wav reference clips")
    ap.add_argument("--ref-transcripts", required=True)
    ap.add_argument("--lines", required=True)
    ap.add_argument("--line-ids", nargs="*", default=None)
    ap.add_argument("--speakers", nargs="*", default=None)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--out", default="data_local/baselines/qwen_v2")
    ap.add_argument("--ks", type=float, nargs="+", default=[0.0])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--direction", default=None)
    ap.add_argument("--layers", type=int, nargs="*", default=[14, 18, 21])
    ap.add_argument("--max-new-tokens", type=int, default=500)
    ap.add_argument(
        "--bound-factor",
        type=float,
        default=None,
        help="cap max_new_tokens per line at 12.5 tokens/s x factor x (words / 2.4 words per s) + 8 (never above --max-new-tokens); "
        "a clip that reaches the cap gets a fade-out. 1.5 is a reasonable start for steered runs",
    )
    ap.add_argument("--fade-ms", type=float, default=150.0)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    lines = json.load(open(a.lines))
    if a.line_ids:
        lines = [l for l in lines if l["line_id"] in set(a.line_ids)]
    refs = find_refs(a.refs, a.speakers)
    transcripts = json.load(open(a.ref_transcripts))
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
            for k in a.ks:
                for seed in a.seeds:
                    jid = f"{spk}/{ln['line_id']}_k{k:.2f}_s{seed}"
                    if jid not in done:
                        jobs.append((spk, ln, k, seed, jid))
    print(f"speakers {sorted(refs)}; {len(jobs)} pending jobs ({len(done)} done); output {root}")
    if a.limit:
        jobs = jobs[: a.limit]
    if a.dry_run or not jobs:
        print("dry run / nothing pending")
        return

    from qwen_tts import Qwen3TTSModel

    dtype = torch.float32 if a.device in ("cpu", "mps") else torch.float16
    tts = Qwen3TTSModel.from_pretrained(
        f"Qwen/Qwen3-TTS-12Hz-{a.size}-Base", device_map=a.device, dtype=dtype
    )
    steerer = Steerer(tts, a.direction, a.layers)
    prompts = {}
    for spk, path in refs.items():
        wav, sr = sf.read(str(path), dtype="float32")
        if wav.ndim > 1:
            wav = wav.mean(axis=1)
        text = transcripts.get(spk)
        prompts[spk] = tts.create_voice_clone_prompt(
            ref_audio=(wav, sr), ref_text=text, x_vector_only_mode=text is None
        )
        print(f"{spk}: prompt built ({'ICL with transcript' if text else 'x-vector only'})")

    new_file = not results.exists()
    fh = open(results, "a", newline="")
    w = csv.DictWriter(fh, fieldnames=FIELDS)
    if new_file:
        w.writeheader()
    for i, (spk, ln, k, seed, jid) in enumerate(jobs, 1):
        rel = f"{spk}/{ln['line_id']}_k{k:.2f}_s{seed}.wav"
        (root / spk).mkdir(exist_ok=True)
        row = {
            "job_id": jid,
            "speaker": spk,
            "line_id": ln["line_id"],
            "intent": ln["intent"],
            "text": ln["text"],
            "exaggeration": k,
            "cfg_weight": 0,
            "seed": seed,
            "path": rel,
            "audio_s": "",
            "gen_s": "",
            "status": "ok",
        }
        t0 = time.time()
        try:
            steerer.k = k
            torch.manual_seed(seed)
            limit = a.max_new_tokens
            if a.bound_factor:
                limit = min(
                    a.max_new_tokens,
                    max(30, int(12.5 * a.bound_factor * len(ln["text"].split()) / 2.4) + 8),
                )
            wavs, sr = tts.generate_voice_clone(
                text=ln["text"],
                language="English",
                voice_clone_prompt=prompts[spk],
                max_new_tokens=limit,
            )
            x = wavs[0]
            capped = bool(a.bound_factor) and len(x) / sr >= (limit - 2) / 12.5
            if capped:  # the model did not stop by itself: fade out instead of cutting abruptly
                n_fade = min(len(x), int(a.fade_ms / 1000 * sr))
                x = x.copy()
                x[-n_fade:] *= np.linspace(1.0, 0.0, n_fade, dtype=x.dtype)
            sf.write(str(root / rel), x, sr, subtype="FLOAT")
            row["audio_s"] = round(len(x) / sr, 3)
            if capped:
                print(f"  [capped at {limit} tokens]")
        except Exception as e:
            row["status"] = "error: " + str(e)[:100]
        row["gen_s"] = round(time.time() - t0, 2)
        w.writerow(row)
        fh.flush()
        print(f"[{i}/{len(jobs)}] {jid} {row['status']} {row['gen_s']}s")
    fh.close()


if __name__ == "__main__":
    main()
