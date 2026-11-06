#!/usr/bin/env python3
"""Transcribe the 5 validation reference clips with Whisper (local only).

Qwen3-TTS's best cloning mode (ICL) needs the transcript of the reference clip, and EARS freeform speech has none.
Run in the CHATTERBOX env (it has transformers) from the repo root:
    python scripts/transcribe_refs.py
Writes data_local/baselines/references/ref_transcripts.json ({speaker: text}). The transcripts are EARS-derived text:
keep them local (data_local is git-ignored); do not commit or share them.
"""

import json

from efforttts import config
from efforttts.eval.asr_speaker import make_transcriber, read_audio_16k

refs = config.data_root() / "baselines" / "references"
transcribe = make_transcriber("openai/whisper-small.en", "cpu")
out = {}
for p in sorted(refs.glob("p*.wav")):
    out[p.stem] = transcribe(read_audio_16k(p))
    print(p.stem, "->", len(out[p.stem].split()), "words")
(refs / "ref_transcripts.json").write_text(json.dumps(out, indent=1))
print("wrote", refs / "ref_transcripts.json")
