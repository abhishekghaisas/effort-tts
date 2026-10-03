# Effort-tts

Speaker-preserving, continuous **vocal-effort control** for zero-shot TTS, plus an
**effort-realism evaluation harness**. The goal is high-effort speech (loud, strained,
projected) in a cloned speaker's own voice, controlled by a reproducible scalar instead of
a free-text tag, and a way to measure whether synthetic effort is realistic and whether
speaker identity survives.

Solo research project. Research and portfolio use only (see *Data and licensing*).

## Status

Week 1 of a four-week plan: evidence and go/no-go.

**Built**
- EARS download, inventory, QA, speaker-disjoint train/val/test split (seeded, gender-stratified)
- Per-file manifest with level and clipping statistics
- Gain-invariant acoustic features (spectral tilt, alpha ratio, Hammarberg, cepstral peak
  prominence, pitch, voicing) with a versioned cache
- Effort estimator v0 (logistic model, loud vs regular reading) with a feature-set ablation
  and out-of-distribution checks
- Fixed 30-line evaluation script with an EARS-overlap check
- Chatterbox baseline generation (resumable; reference-clip picker)

**Not yet built:** scoring of generated audio, speaker-similarity and WER metrics, the
gap report, fine-tuning, listening test, demo.

## Known limitations

- EARS "loud" reading reaches roughly anger-level spectral effort, not yelling or screaming;
  the extreme end exists only as non-speech clips.
- The pitch tracker is a simple autocorrelation method with strict voicing rules. Spot-check
  F0 on new audio before trusting it.
- Only 5 validation speakers: validation differences of a few AUC points are noise.
- The estimator is trained on loud vs regular reading only; level and spectral cues are
  physically correlated, so "effort" here means a gain-invariant acoustic proxy.

## Setup

Two environments, because Chatterbox pins its own dependency versions.

**Project environment** (Python 3.10+; data, features, estimator, tests):
```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest
```

**Chatterbox environment** (Python 3.11; baseline generation only):
```bash
python3.11 -m venv ~/venvs/chatterbox && source ~/venvs/chatterbox/bin/activate
pip install chatterbox-tts "setuptools<81"
pip install -e . --no-deps
```
Model weights (about 3 GB) download from Hugging Face on first use. Heavy data lives in
`data_local/` (git-ignored) or wherever `EFFORT_DATA_DIR` points, for example a Google
Drive folder on Colab.

## Pipeline

Project environment:
```bash
python -m efforttts.data.download --meta-only
python -m efforttts.data.download --speakers 1 2 3        # per-speaker zips, resumable
python -m efforttts.data.inventory                         # file inventory vs transcripts
python -m efforttts.data.splits                            # speaker-disjoint split
python -m efforttts.data.manifest                          # per-file table + level stats
python -m efforttts.effort.features --workers 3            # acoustic features (cached by version)
python -m efforttts.effort.estimator                       # ablation + scores
python -m efforttts.baselines.test_script                  # validate script, check EARS overlap
```

Chatterbox environment:
```bash
python -m efforttts.baselines.references                   # reference clip per speaker
python -m efforttts.baselines.chatterbox_gen --dry-run     # job count + time estimate
python -m efforttts.baselines.chatterbox_gen               # resumable
```

## Layout

- `src/efforttts/data`: download, inventory, splits, manifest
- `src/efforttts/effort`: acoustic features, estimator
- `src/efforttts/baselines`: test script, reference picker, Chatterbox generation
- `src/efforttts/eval`, `train`, `demo`: planned
- `configs/`: data policy, fixed 30-line test script (evaluation only; never train on it)
- `scripts/`: smoke tests; `notebooks/`: Colab bootstrap; `tests/`: pytest suite

## Data and licensing

- **EARS** (Meta Research / University of Hamburg) is licensed **CC-NC 4.0**
  (non-commercial); check the exact license text in the EARS repository. The dataset, transcripts, and derived tables are **not** in this
  repository. Any weights trained on it should be treated as non-commercial.
- Chatterbox is MIT-licensed and watermarks every output; keep the watermark intact and
  disclose synthetic audio in demos.
- Consenting adult voices only. No minors' voices in training or demo data.
- Verify the license of every model and dataset before use.
