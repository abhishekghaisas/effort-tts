"""STEP A of scoring (run in the CHATTERBOX env: needs torch + transformers).

For every generated clip: ASR transcript + word error rate (Whisper) and speaker
similarity (WavLM x-vector embeddings, cosine). Also embeds REAL speech from the same
speakers (their other freeform speech, plus regular/loud reading and anger/ecstasy/neutral
freeform) so we can see how much similarity drops from real vocal effort alone.

    python -m efforttts.eval.asr_speaker [--cfg-weight 0.5]

Similarity is measured against (a) the speaker's CENTROID of real freeform windows that
exclude the reference clip, and (b) the exact reference clip Chatterbox was conditioned on.
Resumable: finished clips in asr_speaker.csv are skipped. Outputs next to results.csv.
Models: openai/whisper-small.en (Apache-2.0 per its model card) and
microsoft/wavlm-base-plus-sv (license: see Microsoft's link on its card [UNVERIFIED]).
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from efforttts import config
from efforttts.eval.wer import wer

ASR_MODEL = "openai/whisper-small.en"
SPK_MODEL = "microsoft/wavlm-base-plus-sv"
SR16 = 16_000
FRAME_S = 0.02
ACTIVE_RANGE_DB = 30.0
CALIBRATION = [("reading", "regular"), ("reading", "loud"), ("emotion_freeform", "anger"),
               ("emotion_freeform", "extasy"), ("emotion_freeform", "neutral")]


def read_audio_16k(path: Path) -> np.ndarray:
    """Read a WAV (PCM or float, any rate), return mono float32 at 16 kHz."""
    from math import gcd

    from scipy.io import wavfile
    from scipy.signal import resample_poly

    sr, x = wavfile.read(str(path))
    if x.dtype == np.int16:
        x = x.astype(np.float32) / 32768.0
    elif x.dtype == np.int32:
        x = x.astype(np.float32) / 2**31
    else:
        x = x.astype(np.float32)
    if x.ndim > 1:
        x = x.mean(axis=1)
    if sr != SR16:
        g = gcd(SR16, sr)
        x = resample_poly(x, SR16 // g, sr // g).astype(np.float32)
    return x


def window_density(x: np.ndarray, sr: int, start_s: float, win_s: float) -> float:
    """Fraction of 20 ms frames within ACTIVE_RANGE_DB of the clip's loudest region."""
    fr = int(FRAME_S * sr)
    n = len(x) // fr
    db = 10 * np.log10(np.mean(x[: n * fr].reshape(n, fr) ** 2, axis=1) + 1e-12)
    active = db >= np.percentile(db, 95) - ACTIVE_RANGE_DB
    a, b = int(start_s / FRAME_S), int((start_s + win_s) / FRAME_S)
    return float(active[a:b].mean())


def pick_centroid_windows(audios: dict[str, np.ndarray], ref_source: str, ref_start: float,
                          win_s: float = 10.0, step_s: float = 30.0, max_windows: int = 6,
                          min_density: float = 0.85, sr: int = SR16) -> list[tuple[str, float]]:
    """Windows from the speaker's real freeform speech that do NOT overlap the reference clip,
    spread across files (round-robin) and skipping windows with long pauses."""
    per_src: dict[str, list[float]] = {}
    for src in sorted(audios):
        x, starts, s = audios[src], [], 0.0
        while s + win_s <= len(x) / sr:
            overlaps = src == ref_source and s < ref_start + win_s and s + win_s > ref_start
            if not overlaps and window_density(x, sr, s, win_s) >= min_density:
                starts.append(s)
            s += step_s
        per_src[src] = starts
    out: list[tuple[str, float]] = []
    i = 0
    while len(out) < max_windows and any(per_src.values()):
        for src in sorted(per_src):
            if per_src[src] and len(out) < max_windows:
                out.append((src, per_src[src].pop(0)))
        i += 1
    return out


def make_embedder(device: str):
    import torch
    from transformers import Wav2Vec2FeatureExtractor, WavLMForXVector

    fe = Wav2Vec2FeatureExtractor.from_pretrained(SPK_MODEL)
    model = WavLMForXVector.from_pretrained(SPK_MODEL).to(device).eval()

    def embed(x16: np.ndarray) -> np.ndarray:
        inp = fe(x16, sampling_rate=SR16, return_tensors="pt", padding=True)
        with torch.no_grad():
            e = model(**{k: v.to(device) for k, v in inp.items()}).embeddings
        return torch.nn.functional.normalize(e, dim=-1)[0].cpu().numpy()

    return embed


def make_transcriber(model_name: str, device: str):
    from transformers import pipeline

    pipe = pipeline("automatic-speech-recognition", model=model_name, device=device)

    def transcribe(x16: np.ndarray) -> str:
        return pipe({"raw": x16, "sampling_rate": SR16})["text"].strip()

    return transcribe


def _unit(v: np.ndarray) -> np.ndarray:
    return v / (np.linalg.norm(v) + 1e-12)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg-weight", type=float, default=0.5)
    ap.add_argument("--asr-model", default=ASR_MODEL)
    ap.add_argument("--asr-device", default="cpu")
    ap.add_argument("--speaker-device", default="cpu")
    args = ap.parse_args()

    from efforttts.baselines.references import freeform_files

    root = config.data_root() / "baselines" / "chatterbox" / f"cfg{args.cfg_weight:.2f}"
    refs_dir = config.data_root() / "baselines" / "references"
    with open(root / "results.csv", newline="") as f:
        gen = [r for r in csv.DictReader(f) if r["status"] == "ok"]
    ref_info = json.loads((refs_dir / "references.json").read_text())
    speakers = sorted({r["speaker"] for r in gen})
    print(f"{len(gen)} generated clips, speakers {speakers}")

    embed = make_embedder(args.speaker_device)
    transcribe = make_transcriber(args.asr_model, args.asr_device)

    # ---- per-speaker reference embedding, centroid of other real speech, calibration ----
    centroids, ref_emb, cal_rows = {}, {}, []
    manifest = list(csv.DictReader(open(config.processed_dir() / "manifest.csv", newline="")))
    for spk in speakers:
        ref_emb[spk] = embed(read_audio_16k(refs_dir / f"{spk}.wav"))
        audios = {str(p.relative_to(config.raw_dir())): read_audio_16k(p) for p in freeform_files(spk)}
        wins = pick_centroid_windows(audios, ref_info[spk]["source"], ref_info[spk]["start_s"])
        if not wins:
            raise SystemExit(f"No usable non-reference freeform windows for {spk}")
        embs = [embed(audios[s][int(a * SR16): int((a + 10.0) * SR16)]) for s, a in wins]
        centroids[spk] = _unit(np.mean(embs, axis=0))
        print(f"{spk}: centroid from {len(wins)} real windows; "
              f"ref-vs-centroid similarity {float(ref_emb[spk] @ centroids[spk]):.3f}")
        for r in manifest:
            if r["speaker"] == spk and (r["category"], r["label"]) in CALIBRATION:
                e = embed(read_audio_16k(config.raw_dir() / r["path"]))
                cal_rows.append({"speaker": spk, "stem": r["stem"], "category": r["category"],
                                 "label": r["label"], "sim_to_centroid": float(e @ centroids[spk])})
    with open(root / "real_calibration.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["speaker", "stem", "category", "label", "sim_to_centroid"])
        w.writeheader()
        w.writerows(cal_rows)
    print(f"real calibration: {len(cal_rows)} clips -> {root / 'real_calibration.csv'}")

    # ---- generated clips (resumable) ----
    out_path = root / "asr_speaker.csv"
    fields = ["job_id", "hypothesis", "wer", "sim_to_centroid", "sim_to_reference"]
    done = set()
    if out_path.exists():
        done = {r["job_id"] for r in csv.DictReader(open(out_path, newline=""))}
    todo = [r for r in gen if r["job_id"] not in done]
    print(f"{len(done)} already scored, {len(todo)} to do")
    with open(out_path, "a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        if not done:
            w.writeheader()
        for i, r in enumerate(todo, 1):
            x = read_audio_16k(root / r["path"])
            hyp = transcribe(x)
            e = embed(x)
            w.writerow({"job_id": r["job_id"], "hypothesis": hyp, "wer": f"{wer(r['text'], hyp):.4f}",
                        "sim_to_centroid": f"{float(e @ centroids[r['speaker']]):.4f}",
                        "sim_to_reference": f"{float(e @ ref_emb[r['speaker']]):.4f}"})
            fh.flush()
            if i % 25 == 0:
                print(f"[{i}/{len(todo)}]")
    print(f"done -> {out_path}")


if __name__ == "__main__":
    main()