#!/usr/bin/env python3
"""Arousal, dominance, valence of every clip, with real EARS emotional speech as reference points (research use only).

Model: audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim (Wav2Vec2-Large-Robust pruned to 12 layers, fine-tuned on MSP-Podcast v1.7; outputs arousal, dominance and
valence roughly in 0..1; license CC BY-NC-SA 4.0, research purposes only). It is trained on natural podcast speech, not on shouting TTS, so read it as a relative instrument:
how the three numbers move with k, and where steered clips land relative to REAL regular, loud, anger, fear, ecstasy ... speech from the same speakers (--anchors).
"Manic" should show up as steered clips landing near ecstasy (high arousal, positive valence) instead of near fear or distress for a fear line.

Run in the CHATTERBOX env (transformers, torch, scipy, pandas), repo root:
    python qwen_v2/emotion_dim.py dev2_L21_trim2 dev2_L21 --anchors --group both --line-ids D01 D02 D03
Clips longer than --max-audio-s (runaways) are skipped. Outputs are EARS-derived voices: local only.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

DIMS = ["arousal", "dominance", "valence"]
ANCHORS = {
    "regular": "rainbow_01_regular",
    "loud": "rainbow_01_loud",
    "neutral": "emo_neutral_sentences",
    "anger": "emo_anger_sentences",
    "fear": "emo_fear_sentences",
    "distress": "emo_distress_sentences",
    "pain": "emo_pain_sentences",
    "extasy": "emo_extasy_sentences",
    "amusement": "emo_amusement_sentences",
    "sadness": "emo_sadness_sentences",
}


def old_name(key):
    """Newer torch names the weight-norm parameters ...parametrizations.weight.original0/1; the checkpoint still has ...weight_g / weight_v (same tensors)."""
    return key.replace("parametrizations.weight.original0", "weight_g").replace(
        "parametrizations.weight.original1", "weight_v"
    )


def summarize(df, base, group="all"):
    """df, base: one row per clip with speaker, line_id, exaggeration and the three dimension columns."""
    b = base.groupby(["speaker", "line_id"])[DIMS].mean()
    d = df.copy()
    for m in DIMS:
        d[f"d_{m[:3]}"] = d[m].to_numpy() - b[m].reindex(list(zip(d.speaker, d.line_id))).to_numpy()
    keys = {
        "all": ["exaggeration"],
        "line": ["line_id", "exaggeration"],
        "speaker": ["speaker", "exaggeration"],
        "both": ["speaker", "line_id", "exaggeration"],
    }[group]
    return d.groupby(keys).agg(
        n=("speaker", "size"),
        **{m[:3]: (m, "mean") for m in DIMS},
        **{f"d_{m[:3]}": (f"d_{m[:3]}", "median") for m in DIMS},
    )


def load_model():
    import torch
    from torch import nn
    from transformers import Wav2Vec2Processor
    from transformers.models.wav2vec2.modeling_wav2vec2 import (
        Wav2Vec2Model,
        Wav2Vec2PreTrainedModel,
    )

    class RegressionHead(nn.Module):
        def __init__(self, config):
            super().__init__()
            self.dense = nn.Linear(config.hidden_size, config.hidden_size)
            self.dropout = nn.Dropout(config.final_dropout)
            self.out_proj = nn.Linear(config.hidden_size, config.num_labels)

        def forward(self, features, **kwargs):
            x = self.dropout(features)
            x = torch.tanh(self.dense(x))
            return self.out_proj(self.dropout(x))

    class EmotionModel(Wav2Vec2PreTrainedModel):
        def __init__(self, config):
            super().__init__(config)
            self.config = config
            self.wav2vec2 = Wav2Vec2Model(config)
            self.classifier = RegressionHead(config)
            self.post_init()  # newer transformers need post_init(); older ones accept it too

        def forward(self, input_values):
            hidden = torch.mean(self.wav2vec2(input_values)[0], dim=1)
            return hidden, self.classifier(hidden)

    name = "audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim"
    processor = Wav2Vec2Processor.from_pretrained(name)
    model = EmotionModel.from_pretrained(name)
    # independent check: every weight of our class must exist in the checkpoint (allowing the weight-norm renaming) AND have the same values
    import torch as _t
    from huggingface_hub import hf_hub_download

    try:
        from safetensors.torch import load_file

        ckpt = load_file(hf_hub_download(name, "model.safetensors"))
    except Exception:
        ckpt = _t.load(hf_hub_download(name, "pytorch_model.bin"), map_location="cpu")
    missing, different = [], []
    for k, v in model.state_dict().items():
        ck = k if k in ckpt else old_name(k) if old_name(k) in ckpt else None
        if ck is None:
            missing.append(k)
        elif ckpt[ck].shape != v.shape or not _t.allclose(ckpt[ck].float(), v.float(), atol=1e-5):
            different.append(k)
    if missing or different:
        raise SystemExit(
            f"Model weights do not match the checkpoint (not found: {missing[:4]}; different values: {different[:4]}): "
            "copy the EmotionModel class from the model card. Outputs would be random."
        )
    print(
        f"emotion model loaded; all {len(model.state_dict())} weights found in the checkpoint with identical values"
    )
    model.eval()

    def avd(x16):
        y = processor(x16, sampling_rate=16000)["input_values"][0].reshape(1, -1)
        with torch.no_grad():
            return model(torch.from_numpy(y))[1].numpy()[0]

    return avd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tags", nargs="+")
    ap.add_argument("--base", default="data_local/baselines/qwen_v2")
    ap.add_argument("--ref", default="dev_base")
    ap.add_argument("--line-ids", nargs="*", default=None)
    ap.add_argument("--group", choices=["all", "line", "speaker", "both"], default="all")
    ap.add_argument("--max-audio-s", type=float, default=10.0)
    ap.add_argument(
        "--anchors",
        action="store_true",
        help="also score real EARS reference clips of the dev speakers",
    )
    ap.add_argument("--data-dir", default="data_local")
    ap.add_argument("--anchor-speakers", nargs="*", default=["p023", "p045", "p056"])
    a = ap.parse_args()

    from scipy.io import wavfile
    from scipy.signal import resample_poly

    avd = load_model()

    def read16(p, max_s=None):
        sr, x = wavfile.read(str(p))
        x = x.astype("float32")
        x = x.mean(axis=1) if x.ndim > 1 else x
        if x.dtype != np.float32 or np.abs(x).max() > 4:  # integer WAV
            x = x / 32768.0
        if max_s:
            x = x[: int(max_s * sr)]
        return resample_poly(x, 16000, sr).astype("float32")

    def score(root):
        res = pd.read_csv(root / "results.csv")
        res = res[(res.status == "ok") & (res.audio_s <= a.max_audio_s)]
        if a.line_ids:
            res = res[res.line_id.isin(a.line_ids)]
        rows = [
            {
                "speaker": r["speaker"],
                "line_id": r["line_id"],
                "exaggeration": r["exaggeration"],
                **dict(zip(DIMS, avd(read16(root / r["path"])))),
            }
            for _, r in res.iterrows()
        ]
        return pd.DataFrame(rows)

    pd.set_option("display.width", 220)
    base_root = Path(a.base)
    base = score(base_root / a.ref)
    print(
        f"unsteered ({a.ref}): mean arousal {base.arousal.mean():.2f}, dominance {base.dominance.mean():.2f}, valence {base.valence.mean():.2f} (n={len(base)})\n"
    )
    if a.anchors:
        rows = []
        for spk in a.anchor_speakers:
            for label, stem in ANCHORS.items():
                p = Path(a.data_dir) / "ears_raw" / spk / f"{stem}.wav"
                if p.exists():
                    rows.append(
                        {"anchor": label, "speaker": spk, **dict(zip(DIMS, avd(read16(p, 12.0))))}
                    )
        an = pd.DataFrame(rows)
        print("REAL EARS reference clips (mean over the dev speakers; first 12 s of each clip):")
        print(
            an.groupby("anchor")[DIMS]
            .mean()
            .reindex([k for k in ANCHORS if k in set(an.anchor)])
            .round(2)
            .to_string(),
            "\n",
        )
    for tag in a.tags:
        df = score(base_root / tag)
        df.to_csv(base_root / tag / "emotion_dim.csv", index=False)
        print(f"== {tag} ==\n{summarize(df, base, a.group).round(2).to_string()}\n")


if __name__ == "__main__":
    main()
