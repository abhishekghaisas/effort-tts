#!/usr/bin/env python3
"""Qwen3-TTS reconnaissance: load the Base model, clone one reference, generate one line, print the model structure.

Run in a Python 3.12 environment with `pip install -U qwen-tts soundfile` (and `brew install sox` on a Mac):
    python qwen_recon.py --size 0.6B --ref data_local/baselines/references/p023.wav \
        --ref-transcripts data_local/baselines/references/ref_transcripts.json --speaker p023
Without a transcript it falls back to x-vector-only cloning (lower quality). CPU by default.
Paste the printed structure and signature back: they decide where the steering hooks go.
"""

import argparse
import inspect
import json
import time

import soundfile as sf
import torch

ap = argparse.ArgumentParser()
ap.add_argument("--size", default="0.6B", choices=["0.6B", "1.7B"])
ap.add_argument("--ref", required=True)
ap.add_argument("--ref-text", default=None)
ap.add_argument("--ref-transcripts", default=None)
ap.add_argument("--speaker", default=None)
ap.add_argument("--device", default="cpu", help="cpu | mps | cuda:0")
ap.add_argument("--text", default="Everybody out of the building, move, move!")
ap.add_argument("--out", default="qwen_recon.wav")
ap.add_argument("--max-new-tokens", type=int, default=400)
a = ap.parse_args()

ref_text = a.ref_text
if ref_text is None and a.ref_transcripts and a.speaker:
    ref_text = json.load(open(a.ref_transcripts)).get(a.speaker)
print("reference transcript:", "yes" if ref_text else "NO (x-vector-only mode)")

from qwen_tts import Qwen3TTSModel

dtype = torch.float32 if a.device in ("cpu", "mps") else torch.float16
t0 = time.time()
tts = Qwen3TTSModel.from_pretrained(
    f"Qwen/Qwen3-TTS-12Hz-{a.size}-Base", device_map=a.device, dtype=dtype
)
print(f"loaded in {time.time() - t0:.0f}s on {a.device} ({dtype})")

wav, sr = sf.read(a.ref, dtype="float32")
if wav.ndim > 1:
    wav = wav.mean(axis=1)
prompt = tts.create_voice_clone_prompt(
    ref_audio=(wav, sr), ref_text=ref_text, x_vector_only_mode=ref_text is None
)
print(
    "prompt items:",
    len(prompt),
    "| fields:",
    [k for k in dir(prompt[0]) if not k.startswith("_")][:12],
)

torch.manual_seed(0)
t0 = time.time()
wavs, out_sr = tts.generate_voice_clone(
    text=a.text, language="English", voice_clone_prompt=prompt, max_new_tokens=a.max_new_tokens
)
dt = time.time() - t0
sf.write(a.out, wavs[0], out_sr)
dur = len(wavs[0]) / out_sr
print(f"generated {dur:.2f}s of audio in {dt:.0f}s (RTF {dt / dur:.1f}) -> {a.out}")

print("\n== generate_voice_clone signature ==")
print(inspect.signature(tts.generate_voice_clone))

print("\n== model structure (depth 3; ModuleList lengths and parameter counts) ==")
root = getattr(tts, "model", tts)
print(type(root).__name__)


def summarize(m, depth=0, max_depth=3):
    for name, child in m.named_children():
        n = sum(p.numel() for p in child.parameters())
        extra = f" ModuleList[{len(child)}]" if isinstance(child, torch.nn.ModuleList) else ""
        print(f"{'  ' * depth}{name}: {type(child).__name__}{extra}  {n / 1e6:.1f}M")
        if depth + 1 < max_depth:
            summarize(child, depth + 1, max_depth)


summarize(root)
cfg = getattr(root, "config", None)
for part in ("talker_config", "code_predictor_config", "text_config"):
    sub = getattr(cfg, part, None)
    if sub is not None:
        print(
            part,
            {
                k: getattr(sub, k, None)
                for k in (
                    "hidden_size",
                    "num_hidden_layers",
                    "num_attention_heads",
                    "num_code_groups",
                )
            },
        )
