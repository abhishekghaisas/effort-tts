# Showcase: vocal effort control for cloned voices (Chatterbox v1.0, Qwen3-TTS steering v2.0)

A Streamlit site that plays the curated outputs of both systems side by side, with the metrics that justify the curation and the negative results. Development lines and three development
speakers, one sample per setting, one listener so far: a demonstration of what was tried, not a held-out evaluation.

## Run order (what to do now)
1. Chatterbox clips on the same lines and references (Chatterbox env, about 10 to 15 minutes) and their Step A and B (below).
2. `python showcase/preflight.py --include-finetune` (project env): one line per source root; fix every MISSING or PROBLEM line before building.
3. `python showcase/build_showcase.py --include-finetune`: read the printed pass rates first.
4. `pip install -r showcase/requirements.txt`, then `streamlit run showcase/app.py`.
5. Hosting decision (private by default), then deploy or keep local.

## Build it (about 40 minutes, mostly generation)
1. Same lines and references for both models. Qwen already has them (dev_base, dev2_L21, dev2_L21_trim2 with Step A, Step B and emotion_dim). Chatterbox needs them (Chatterbox env, repo root):
   ```bash
   python showcase/gen_chatterbox_dev.py                      # 120 clips, about 10 to 15 minutes
   python -m efforttts.eval.asr_speaker --root data_local/baselines/chatterbox_dev/cfg0.50
   python -m efforttts.eval.score_baseline --root data_local/baselines/chatterbox_dev/cfg0.50 --default-ex 0.5     # project env
   ```
2. Build the curated folder (project env): `python showcase/build_showcase.py --include-finetune`. It prints how many clips passed the ONE filter used for every model (ends on its own within 10 s, WER <= 0.2,
   similarity >= 0.80, >= 1.6 words/s, no duplicated words) and writes `manifest.csv`, `metrics.csv`, `curation_log.txt` and level-matched 16-bit WAVs under `audio/`.
3. Run it locally: `pip install -r showcase/requirements.txt` then `streamlit run showcase/app.py`.

## Tuning the filter
`python showcase/build_showcase.py --include-finetune --max-wer 0.25` (also `--min-sim`, `--min-wps`, `--failed-examples N`). The first real build removed 100 of 534 clips, mostly for WER above 0.2 (Whisper errs on shouted speech and a 6 to 9 word line fails after two
mistakes). To see how many removed clips were borderline, run in the project env:
```python
import pandas as pd; m = pd.read_csv('showcase/manifest.csv')
print(m[(m.reason == 'WER above 0.2') & (m.wer <= 0.34)].groupby('group').size())   # removed clips with 2 to 3 word errors on a 9-word line
```
The app plays a few removed clips ("What did not work") so the curation is audible.

## Publish, simple version (decided 2026-11-04): ONE repository, EARS data and transcripts git-ignored, everything else pushed
1. `.gitignore`: `data_local/`, `EffortDataset/`, `chatterbox-finetuning/`, `.streamlit/secrets.toml`, `qwen_recon.wav`, `ladder_*.wav`, `tokens.npz`. Pin the app's versions: `pip freeze | grep -iE '^(streamlit|pandas|numpy|scipy)==' > showcase/requirements.txt`.
2. `python scripts/prepush_check.py` (mode single: showcase audio allowed; still blocks data folders, transcripts, stray audio, model files and secrets; `--allow <path>` lets a file through, e.g. `qwen_v2/directions_06B.pt`), then `git add -A`, commit, push.
3. share.streamlit.io: Create app, this repository, branch main, main file `showcase/app.py`, matching Python version, Secrets: `APP_PASSWORD = "..."` (optional). The repository's visibility decides who can download the audio; the password only guards the app.
The two-repository route below (`export_repo.py`) is kept for later in case a separate public repository is ever wanted.

## Publish, optional two-repository version (the research repo stays private, the hosted app lives in its own small repo)
1. Research repo: add the ignore entries (`data_local/`, `EffortDataset/`, `chatterbox-finetuning/`, `qwen_recon.wav`, `ladder_*.wav`, `tokens.npz`, `qwen_v2/directions_*.pt`, `showcase/audio/`, `showcase/audio_failed/`, `showcase/manifest.csv`, `showcase/metrics.csv`, `showcase/filter.json`,
   `showcase/curation_log.txt`, `.streamlit/secrets.toml`), run `python scripts/prepush_check.py --mode main`, fix every BLOCK line, then commit and push to the PRIVATE repo.
2. `python showcase/export_repo.py --out ../effort-tts-showcase --pin` (in the environment where the app works) writes a minimal repo: the app, the curated audio, a pinned `requirements.txt`, NOTICE.md and a `.gitignore`.
3. In that folder: `git init -b main`, `python ../effort-tts/scripts/prepush_check.py --mode showcase`, commit, `gh repo create effort-tts-showcase --private --source=. --push`.
4. share.streamlit.io: create the app from that repo, branch main, main file `app.py`, matching Python version, and `APP_PASSWORD = "..."` under Secrets.
5. Visibility: the password protects the APP, not the files. If the GitHub repo is public, anyone can download the audio from GitHub. Repository visibility is the real control.

## Before anything goes online (decision needed)
The voices are synthetic clones of EARS participants (CC BY-NC 4.0, non-commercial). The project's own rules say: keep the repository private until a publishing decision, do not redistribute EARS-derived data, do not upload
EARS voices to third-party services. A hosted Streamlit app is a third-party service, so choose explicitly:
- **Private (recommended default).** Private GitHub repo plus a private app, viewers by invitation; or run it only locally for demos; or the built-in password gate (set `APP_PASSWORD` in the app's secrets).
- **Public.** Only after you accept the consequences of publishing clones of real people's voices, even licensed ones; keep the attribution, the non-commercial notice and the "synthetic" labels in the app; consider using
  only your own consented recordings for the public version. `showcase/audio/` must not go into a public repository unless this is decided.

Streamlit Community Cloud notes (from its documentation and third-party summaries, which disagree in places; check the app's Share settings and the current plan page before relying on them): an app inherits its
repository's visibility (private repo gives a private app, public repo a public app); a private app is limited to invited viewers and the docs say one private app at a time; free apps hibernate when idle and have a memory cap
(reported as roughly 0.7 to 2.7 GB in 2024). The audio is small (about 150 KB per clip, a few hundred clips), so the memory cap is not the problem.

## Content rules the app follows
Level-matched audio by default; a loudness ramp is optional and only raises level (real loud speech is about +9 dB over regular). Flags on strong settings (Qwen k >= 1.25: can sound excited or manic; Chatterbox exaggeration >= 1.2: identity and
stability cost). Chatterbox clips keep Resemble AI's watermark. Disclose that every clip is synthetic.