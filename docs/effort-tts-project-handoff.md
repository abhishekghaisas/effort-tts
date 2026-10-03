# Project Handoff: Speaker-Preserving Vocal-Effort Control for TTS

*Prepared 2026-10-02 from a research chat. Purpose: let a fresh chat pick up the work without re-deriving anything. Items marked **[UNVERIFIED]** came from secondary sources or memory and must be checked before relying on them.*

---

## 1. One-paragraph summary

The goal is a one-month, solo software-engineering project: make a zero-shot TTS system produce **high-effort speech (shouting, strained or projected delivery) in a cloned speaker's own voice**, with a **continuous, reproducible effort control** (a scalar, not a text tag), plus an **evaluation harness** that measures whether synthetic effort is realistic and whether speaker identity survives. Existing commercial and open systems already offer coarse shout/scream tags, so the project targets the gaps those tags leave (Section 4), not the absence of the feature.

## 2. Who and what

- Builder: a software engineer (not a speech-research specialist). Hardware: MacBook Air M5 (fanless) for development; Google Colab (AI Pro) for GPU training.
- Origin of the idea: a hypothetical ad-agency scenario (dubbing with a young child's voice from messy samples). That led to a search for under-served technical problems in voice acting.
- Timeline: **one month** (a one-week version was considered and declined; see Section 8 for a fallback if the fine-tune fails).

## 3. Verification findings (what already exists)

| Area | Finding | Source |
|---|---|---|
| ElevenLabs v3 | Has delivery tags such as `[whispers]`, `[shouts]`. Tags are natural-language directions, not guaranteed commands; effect varies by voice; a calm voice asked to shout tends to sound muted or unnatural; exact reproducibility not guaranteed. | elevenlabs.io docs (audio tags help article); third-party guides |
| Fish Audio OpenAudio S1 | Tone markers `(shouting)`, `(screaming)`, `(whispering)`. Code Apache-2.0; **weights CC-BY-NC-SA-4.0 (non-commercial)**. S2 Pro reportedly uses free-form tags incl. `[screaming]`, `[shouting]`, `[loud]` and a research-only license **[UNVERIFIED license details]**. | github.com/fishaudio/fish-speech; huggingface.co/fishaudio/openaudio-s1-mini |
| EARS dataset | 100 h, 107 speakers, 48 kHz, anechoic. Reading styles: regular, loud, whisper, fast, slow, high pitch, low pitch. Also 22 emotional styles and ~18 min freeform per speaker. Transcripts provided for reading portions. Spans whispering to yelling/screaming. **License: CC-NC 4.0** (non-commercial). | github.com/facebookresearch/ears_dataset ; arXiv 2406.06185 |
| Vocal-effort control research | Flow-matching TTS with continuous, disentangled vocal-effort and articulation control; trained on ~11 h of Expresso (default, enunciated, fast, projected; 4 speakers) + LJ Speech; pseudo-labels (effort 0.3 neutral, 0.9 projected). Framed as intelligibility in noise (Lombard), not acting. | arXiv 2606.23176 (Akti & Waibel) |
| Zero-shot Lombard | Style-embedding manipulation (PCA) gives Lombard control for any speaker without explicit Lombard training data. | arXiv 2601.12966 |
| GTR-Voice | Controls glottalization, tenseness, resonance; 3.6 h from one professional voice actor, 125 GTR combinations; dataset and models open-sourced. | arXiv 2406.10514 |
| Data scarcity | Literature notes that collecting hours of one speaker's high-effort speech (e.g., shouting) is difficult. | arXiv 1810.12051 |
| Closest evaluation work | SceneTTS-Bench scores drama dubbing on timbre consistency, under-acting ratio, rate discontinuity (scene-level). EmergentTTS-Eval uses audio-LLM judges. Neither is specific to vocal-effort realism. | arXiv 2609.26255; arXiv 2505.23009 |

### Correction to earlier claims in the research chat
Earlier I implied shouting synthesis was "poorly served" and that suitable data barely exists. Both were overstated: shout tags ship in commercial and open models, and EARS provides a public multi-speaker loud/yelling set. The real gap is narrower (Section 4).

## 4. The gap this project targets

*Based only on my searches; absence of evidence is not proof. Re-check before committing publicly.*

1. **Speaker-faithful effort transfer.** From a calm or messy reference clip to a plausible shout *in that speaker's timbre*. Current tag-based systems are voice-dependent and degrade when the voice and tag conflict.
2. **Continuous, reproducible intensity control** instead of non-deterministic text tags. Existing scalar control is framed around intelligibility in noise, not performance.
3. **An effort-realism benchmark.** Nothing found that checks whether synthetic shouting has plausible vocal-effort acoustics (spectral tilt, F0, voice quality) rather than just being louder or higher-pitched.

## 5. Project definition

**Deliverables**
1. **Effort estimator** (acoustic features and a small classifier/regressor trained on EARS style labels) that outputs a continuous effort score and must work after loudness normalization.
2. **Baseline gap report**: measured effort vs speaker similarity vs WER for Chatterbox, ElevenLabs v3 (if API access), and optionally Fish (research only), on a fixed test script.
3. **Fine-tuned model**: LoRA on a permissively licensed zero-shot TTS, conditioned on an effort scalar, preserving cloned identity.
4. **Benchmark + short write-up**, small demo (slider), code and model card.

**Candidate base models (choose in Week 1 by fine-tunability)**
- **Chatterbox** (Resemble AI): MIT, ~0.5B, zero-shot cloning from ~5 s, has an "exaggeration" parameter, adds a perceptual watermark to outputs. Default candidate. Fine-tune tooling maturity **[UNVERIFIED]**.
- **Orpheus 3B**: sources conflict on weights license (Apache-2.0 vs Llama 3.2 terms). **[UNVERIFIED, check before use]**.
- **Qwen3-TTS** (reported Apache-2.0, ~600M) **[UNVERIFIED]**; **StyleTTS2** (MIT) as a well-trodden fine-tuning path.
- Avoid for deployment: F5-TTS weights (CC-BY-NC), Fish weights (NC), XTTS v2 (non-commercial).

**Design notes**
- Effort lives in spectral cues (tilt, voice quality, F0), not just level. **Do not let loudness normalization erase effort cues**; apply final loudness in post.
- Dense effort labels: train the estimator on EARS style labels, then score all EARS audio (including emotional/freeform) to get continuous pseudo-labels.
- Split data **by speaker** (hold out ~10, balanced by gender). Include a **calm-reference-only** test (reference clip is calm; target is shouted), because that is the real use case.
- Possible stretch: word-level effort contours via forced alignment; speech-to-speech effort converter if EARS turns out near-parallel.

## 6. Evaluation plan

Proposed metrics (**set numeric thresholds after Week 1 baselines**, not before):
- **Controllability:** rank correlation between requested effort and estimator-measured effort.
- **Identity preservation:** speaker-verification embedding similarity vs neutral output, across effort levels.
- **Intelligibility:** ASR WER vs neutral.
- **Naturalness:** automatic MOS predictor plus an informal listening test (15-20 raters in Week 4).
- **Estimator sanity:** confirm it tracks spectral cues, not loudness.

Fixed 30-line test script (never trained on): calm, urgent, and full-shout lines; varied length; hard phonemes.

## 7. Four-week plan

**Week 1: Evidence and go/no-go (mostly on the Mac; cheap GPU or API for baselines)**
1. Download EARS; build loader; check whether styles share sentences per speaker; decide normalization policy.
2. Build the effort estimator; validate on held-out speakers.
3. Generate baselines for the 30 lines across 6+ voices.
4. Compute effort / speaker-similarity / WER and produce the gap plot (this is your evidence).
5. Tiny LoRA run on the candidate base model; **go/no-go**. If it fails, switch candidate before Week 2.

**Week 2: Train v0 (Colab L4)**
- Dense labels, speaker-disjoint splits, LoRA with effort conditioning. Overfit 2 speakers first as a sanity check, then full split.

**Week 3: Harden (Colab)**
- Calm-reference tests on held-out speakers. Fix clipping artifacts, intelligibility loss at high effort, and speaker drift (consider a speaker-consistency loss). Stretch: word-level contours.

**Week 4: Evaluate and ship**
- Full benchmark, listening test, slider demo, release code + eval harness + model card (weights are research-only because EARS is NC).

**Fallback if fine-tuning fails:** training-free **best-of-N** selection. Generate many candidates with tag-based systems, score with the estimator and a speaker-similarity model, return the one closest to the requested effort at acceptable identity. Reuses the same harness.

## 8. Compute and environment

- **Mac (M5 Air, fanless):** 16 GB unified memory standard, configurable to 24/32 GB, 153 GB/s bandwidth. **User's actual RAM config unknown.** Good for data work, estimator, metrics, API-based baselines, debugging. Not recommended for the sustained fine-tune (thermal throttling; CUDA-assuming training repos; tight memory). Local Chatterbox inference via Metal/CPU should be tested on day 1 **[UNVERIFIED]**.
- **Colab (Google AI Pro):** exact compute-unit allocation and background-execution status **[UNVERIFIED, check in Colab]**. Reported for Colab Pro: 100 CU/month; GPUs T4/L4/A100 subject to availability; sessions up to 24 h while units last; no guaranteed GPU; running out of units drops you to free-tier limits. Burn-rate figures from sources disagree (A100 ~5.4 vs ~15 CU/hr), so use the in-app meter.
- **Rough budget:** ~30-40 GPU-hours total. On an L4 (~1.7 CU/hr, 22.5 GB) that is roughly 50-70 CU. Use L4 as the workhorse; reserve A100 for a final run only if needed. **Avoid T4 for training** (15 GB, no BF16, no FlashAttention 2).
- **Engineering conventions:** git repo; notebooks only clone/install/call scripts; checkpoint to Drive frequently with resume logic from day one; batch size adapts to VRAM; preprocess EARS once (resample, subset) and store on Drive. Raw 48 kHz audio is on the order of tens of GB (my rough arithmetic), so do not redownload each session.
- **Suggested repo layout:** `data/` (loaders, preprocessing), `effort/` (estimator), `baselines/` (generation scripts), `eval/` (metrics), `train/` (LoRA training), `demo/`.

## 9. Ethics and licensing guardrails

- Use **consenting adult data only**. Keep minors out of training and demo data entirely. The origin scenario involved a child's voice; cloning minors' voices raised consent issues (e.g., an open letter from ~1,000 agents/actors objecting to indefinite AI-training rights over child performers' voices, and guidance advising against cloned minors' voices in commercial advertising even with guardian consent). Making cloned voices more extreme worsens that problem.
- EARS is **CC-NC 4.0**: fine for research/portfolio; any commercial product needs your own consented recordings. Trained weights inherit this limitation.
- Keep Chatterbox's output watermark intact; disclose synthetic audio in demos.
- Verify the current license of every model/tool before use.

## 10. Open items and unverified claims (check first)

1. Is EARS near-parallel across styles (same sentences)? Shapes training design.
2. Does Chatterbox (or the chosen base) actually fine-tune smoothly? Current license and tooling.
3. Orpheus / Qwen3-TTS license terms; Expresso license.
4. Colab AI Pro: compute-unit balance, background execution, which GPUs you actually get.
5. Mac RAM configuration.
6. Re-run a prior-art search just before starting (the field moves fast); I found no specific prior work on gaps 1-3, which is not proof none exists.
7. Estimator risk: confirm it measures vocal effort, not loudness.

## 11. Parked idea (not part of this project)

**Mic-only vocal-load monitoring for recording sessions:** estimate voice-actor fatigue and cumulative "vocal dose" from the studio microphone signal (F0, speech level, smoothed cepstral peak prominence) instead of wearables. A 16-actor pilot found perceived fatigue as early as 2 hours into a 4-hour session, with cepstral peak prominence and spectral tilt tracking it. Hard parts: separating character voice from fatigue, signal-chain effects, no labeled dataset, privacy. Possible later link: the same estimator infrastructure.

## 12. Immediate next steps

1. Check Colab compute-unit balance and Mac RAM.
2. Create the repo with the layout above.
3. Write the 30-line fixed test script.
4. Build the EARS loader and preprocessing (Mac-friendly, Drive-friendly), then the effort estimator.

## 13. Suggested opening message for the new chat

> I'm continuing a project from an earlier research chat. Please read the attached handoff document. We're on Day 1 of Week 1. First task: write the EARS loader and preprocessing script that runs on my MacBook Air M5 and on Colab, writes to Google Drive, resamples audio, preserves effort cues (no peak/level normalization that erases them), splits by speaker, and checks whether the reading styles share sentences per speaker. Then we'll build the effort estimator.
