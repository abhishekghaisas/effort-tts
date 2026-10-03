"""Chatterbox smoke test: does it run here, how fast, how much memory?

Run inside the CHATTERBOX environment (Python 3.11), not the project venv:

    python scripts/chatterbox_smoke.py --ref path/to/freeform_speech_01.wav

It crops a short reference clip, generates one line at several `exaggeration`
values, saves the wavs, and prints timing (RTF = compute seconds per audio
second; below 1.0 is faster than real time) and, on Apple GPUs, memory use.
Chatterbox outputs carry an imperceptible watermark; do not strip it.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path


def pick_device() -> str:
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def crop_reference(src: Path, dst: Path, start_s: float, seconds: float) -> None:
    """Cut a mono reference clip. Chatterbox clones from roughly 5-10 s of speech."""
    import soundfile as sf

    x, sr = sf.read(str(src), dtype="float32", always_2d=False)
    if x.ndim > 1:
        x = x.mean(axis=1)
    a = int(start_s * sr)
    b = a + int(seconds * sr)
    if b > len(x):
        raise SystemExit(f"Reference is {len(x) / sr:.1f}s; need at least {start_s + seconds:.1f}s")
    dst.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(dst), x[a:b], sr)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", required=True, type=Path, help="long calm clip of one speaker")
    ap.add_argument("--ref-start", type=float, default=20.0, help="seconds into the clip")
    ap.add_argument("--ref-seconds", type=float, default=10.0)
    ap.add_argument("--text", default="Get back from the edge right now!")
    ap.add_argument("--exaggerations", type=float, nargs="+", default=[0.3, 0.5, 0.8, 1.2])
    ap.add_argument("--cfg-weight", type=float, default=0.5)
    ap.add_argument("--out", type=Path, default=Path("chatterbox_smoke_out"))
    args = ap.parse_args()

    import torch
    import torchaudio

    device = pick_device()
    print(f"device: {device}")
    if device != "cuda":
        # Checkpoints may have been saved on CUDA; force loading onto this device.
        _orig_load = torch.load

        def _load(*a, **k):
            k.setdefault("map_location", torch.device(device))
            return _orig_load(*a, **k)

        torch.load = _load

    ref = args.out / "reference.wav"
    crop_reference(args.ref, ref, args.ref_start, args.ref_seconds)
    print(f"reference crop -> {ref}  (LISTEN to it: it must be clean speech, not a pause)")

    from chatterbox.tts import ChatterboxTTS

    t0 = time.time()
    model = ChatterboxTTS.from_pretrained(device=device)
    print(f"model loaded in {time.time() - t0:.1f}s")

    for ex in args.exaggerations:
        try:
            t = time.time()
            wav = model.generate(
                args.text, audio_prompt_path=str(ref), exaggeration=ex, cfg_weight=args.cfg_weight
            )
            dt = time.time() - t
            dur = wav.shape[-1] / model.sr
            out = args.out / f"exaggeration_{ex:.2f}.wav"
            torchaudio.save(str(out), wav.cpu(), model.sr)
            print(f"exaggeration {ex:.2f}: {dur:.1f}s audio in {dt:.1f}s (RTF {dt / dur:.2f}) -> {out}")
        except Exception as e:  # report and continue, e.g. a value outside the allowed range
            print(f"exaggeration {ex:.2f}: FAILED: {type(e).__name__}: {str(e)[:200]}")

    if device == "mps":
        try:
            print(f"MPS driver memory: {torch.mps.driver_allocated_memory() / 1e9:.1f} GB")
        except Exception:
            pass


if __name__ == "__main__":
    main()