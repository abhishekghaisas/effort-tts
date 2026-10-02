# effort-tts

Speaker-preserving, continuous vocal-effort control for zero-shot TTS, plus an
effort-realism evaluation harness. Research project; data is CC-NC (EARS).

## Setup (Mac)
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
pytest
```
Install torch separately when you need it (https://pytorch.org).

## Setup (Colab)
Open `notebooks/colab_bootstrap.ipynb`. It mounts Drive, clones the repo,
installs it, and sets `EFFORT_DATA_DIR` to a Drive folder.

## Layout
- `src/efforttts/data`      loaders, preprocessing
- `src/efforttts/effort`    effort estimator
- `src/efforttts/baselines` baseline generation
- `src/efforttts/eval`      metrics and benchmark
- `src/efforttts/train`     LoRA training
- `src/efforttts/demo`      slider demo
- `scripts/`                thin CLI entry points (notebooks call these)
- `configs/`, `tests/`, `docs/`, `notebooks/`

## Guardrails
Consenting adult data only. No minors' voices in training or demos.
Keep TTS output watermarks intact and disclose synthetic audio.
Verify the license of every model and dataset before use.
