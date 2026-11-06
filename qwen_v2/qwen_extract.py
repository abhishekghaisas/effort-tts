#!/usr/bin/env python3
"""Extract an effort direction from Qwen3-TTS using EARS matched loud and regular readings (TRAINING speakers only).

Method A (as in arXiv 2610.07647, adapted): for each training speaker, clone from a REGULAR reference clip and from a LOUD
reference clip (same texts, same speaker), generate the same two target sentences, and record the talker's hidden state at
every decode step in every layer. Direction(layer) = mean over speakers of [mean(loud) - mean(regular)].
The targets are EARS sentences that are NOT among the 30 evaluation lines (checked below). Writes directions.pt and prints,
per layer, the norm and how consistent the per-speaker directions are (mean cosine to the average direction).

    python qwen_v2/qwen_extract.py --size 0.6B --eval-lines finetune/eval_lines.json --out qwen_v2/directions_06B.pt
Untested; about 25 minutes on a Mac CPU for the defaults (12 speakers x 3 reference texts x 2 styles x 2 targets).
"""

import argparse
import json
import re
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import soundfile as sf
    import torch
except ImportError:  # lets --dry-run plan jobs without the ML stack installed
    sf = torch = None

ap = argparse.ArgumentParser()
ap.add_argument("--size", default="0.6B", choices=["0.6B", "1.7B"])
ap.add_argument("--device", default="cpu")
ap.add_argument("--data-dir", default="data_local")
ap.add_argument("--eval-lines", required=True)
ap.add_argument(
    "--speakers", nargs="*", default=None, help="default: the train-split speakers in the manifest"
)
ap.add_argument("--n-ref-texts", type=int, default=3)
ap.add_argument("--n-targets", type=int, default=2)
ap.add_argument("--max-target-chars", type=int, default=200)
ap.add_argument(
    "--loud-emotion",
    default=None,
    help="use scripted EARS emotion clips (stem emo_<name>_*, with a transcript), e.g. anger, as the 'loud' side instead of loud reading",
)
ap.add_argument("--n-loud-refs", type=int, default=2)
ap.add_argument(
    "--match-level",
    type=float,
    default=None,
    help="scale every reference clip to this level (dBFS, 95th percentile of 25 ms frame RMS, e.g. -26) so the direction reflects voice quality, not recording level",
)
ap.add_argument("--max-new-tokens", type=int, default=300)
ap.add_argument("--out", default="qwen_v2/directions.pt")
ap.add_argument("--dry-run", action="store_true")
a = ap.parse_args()

data = Path(a.data_dir)
man = pd.read_csv(data / "ears_processed" / "manifest.csv")
man["text"] = man["text"].fillna("").astype(str)
spks = a.speakers or sorted(man.loc[man["split"] == "train", "speaker"].unique())
eval_texts = {l["text"].strip().lower() for l in json.load(open(a.eval_lines))}

plan = []  # (speaker, [(stem_regular, stem_loud, text_regular, text_loud)...], [target texts])
for s in spks:
    d = man[man["speaker"] == s].set_index("stem")
    pairs = []
    for stem in sorted(d.index):
        m = re.match(r"^((?:rainbow|sentences)_\d+)_regular$", stem)
        if (
            m
            and f"{m.group(1)}_loud" in d.index
            and d.loc[stem, "text"]
            and d.loc[f"{m.group(1)}_loud", "text"]
        ):
            pairs.append(
                (
                    stem,
                    f"{m.group(1)}_loud",
                    d.loc[stem, "text"],
                    d.loc[f"{m.group(1)}_loud", "text"],
                )
            )
    pairs = pairs[: a.n_ref_texts]
    regular_refs = [(p[0], p[2]) for p in pairs]
    loud_refs = [(p[1], p[3]) for p in pairs]
    if a.loud_emotion:
        em = [
            st
            for st in sorted(d.index)
            if re.match(rf"^emo_{re.escape(a.loud_emotion)}_", st) and d.loc[st, "text"].strip()
        ]
        loud_refs = [(st, d.loc[st, "text"]) for st in em[: a.n_loud_refs]]
    # targets: the shortest available EARS sentences (any style) that are not evaluation lines and not reference texts
    ref_stems = {x[0] for x in pairs} | {x[1] for x in pairs}
    cands, why = [], {"sentence stems": 0, "no text": 0, "too long": 0, "eval line": 0}
    for stem in d.index:
        if not re.match(r"^sentences_\d+_", stem) or stem in ref_stems:
            continue
        why["sentence stems"] += 1
        t = d.loc[stem, "text"].strip()
        if not t:
            why["no text"] += 1
        elif len(t) > a.max_target_chars:
            why["too long"] += 1
        elif t.lower() in eval_texts:
            why["eval line"] += 1
        else:
            cands.append((len(t), stem, t))
    targets = []
    for _, stem, t in sorted(cands):
        if t not in targets:
            targets.append(t)
        if len(targets) >= a.n_targets:
            break
    if s == spks[0]:
        print(f"target candidates for {s}: {len(cands)} usable; counts: {why}")
    plan.append((s, regular_refs, loud_refs, targets))
n_gen = sum((len(r) + len(lo)) * len(t) for _, r, lo, t in plan)
print(f"{len(spks)} speakers; {n_gen} generations planned")
if n_gen == 0:
    raise SystemExit(
        "Nothing to generate: no usable reference pairs or target sentences (see the candidate counts above)."
    )
if a.dry_run:
    for s, r, lo, t in plan[:3]:
        print(
            s, "regular:", [x[0] for x in r], "loud side:", [x[0] for x in lo], "targets:", len(t)
        )
    raise SystemExit

from qwen_tts import Qwen3TTSModel

dtype = torch.float32 if a.device in ("cpu", "mps") else torch.float16
tts = Qwen3TTSModel.from_pretrained(
    f"Qwen/Qwen3-TTS-12Hz-{a.size}-Base", device_map=a.device, dtype=dtype
)
layers = tts.model.talker.model.layers
n_layers = len(layers)
acc = {}  # (style, speaker) -> {layer: [sum_vector, count]}
state = {"key": None}


def make_hook(l):
    def hook(module, inputs, output):
        hs = output[0] if isinstance(output, tuple) else output
        if hs.shape[1] == 1 and state["key"] is not None:
            slot = acc[state["key"]].setdefault(l, [torch.zeros(hs.shape[-1]), 0])
            slot[0] += hs[:, -1, :].float().mean(dim=0).cpu()
            slot[1] += 1
        return output

    return hook


handles = [layers[l].register_forward_hook(make_hook(l)) for l in range(n_layers)]
t0 = time.time()
done = 0
for s, regular_refs, loud_refs, targets in plan:
    for style in ("regular", "loud"):
        acc[(style, s)] = {}
    for style, refs in (("regular", regular_refs), ("loud", loud_refs)):
        for stem, text in refs:
            wav, sr = sf.read(str(data / "ears_raw" / s / f"{stem}.wav"), dtype="float32")
            if wav.ndim > 1:
                wav = wav.mean(axis=1)
            if a.match_level is not None:
                n_ = int(0.025 * sr)
                fr = wav[: len(wav) // n_ * n_].reshape(-1, n_)
                cur = 20 * np.log10(np.percentile(np.sqrt((fr**2).mean(axis=1)) + 1e-9, 95))
                g = 10 ** ((a.match_level - cur) / 20)
                wav = (wav * min(g, 0.99 / max(float(np.abs(wav).max()), 1e-9))).astype("float32")
            prompt = tts.create_voice_clone_prompt(
                ref_audio=(wav, sr), ref_text=text, x_vector_only_mode=False
            )
            state["key"] = (style, s)
            for tgt in targets:
                torch.manual_seed(0)
                tts.generate_voice_clone(
                    text=tgt,
                    language="English",
                    voice_clone_prompt=prompt,
                    max_new_tokens=a.max_new_tokens,
                )
                done += 1
            state["key"] = None
    print(f"{s} done ({done}/{n_gen}, {time.time() - t0:.0f}s)")
for h in handles:
    h.remove()

per_speaker = defaultdict(dict)  # layer -> speaker -> direction
for s in spks:
    for l in range(n_layers):
        lo, re_ = acc[("loud", s)].get(l), acc[("regular", s)].get(l)
        if lo and re_ and lo[1] and re_[1]:
            per_speaker[l][s] = lo[0] / lo[1] - re_[0] / re_[1]
out = {}
print("\nlayer  norm(mean dir)  consistency (mean cosine of per-speaker dirs to the mean)")
for l in range(n_layers):
    if not per_speaker[l]:
        continue
    stack = torch.stack(list(per_speaker[l].values()))
    mean = stack.mean(dim=0)
    cos = torch.nn.functional.cosine_similarity(stack, mean[None], dim=1).mean().item()
    out[l] = mean
    print(f"{l:5d}  {mean.norm().item():10.3f}  {cos:8.3f}")
torch.save(
    {
        "layers": out,
        "meta": {
            "size": a.size,
            "speakers": spks,
            "n_ref_texts": a.n_ref_texts,
            "targets": a.n_targets,
        },
    },
    a.out,
)
print("saved", a.out)
