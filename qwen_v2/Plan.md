# v2.0 plan: Qwen3-TTS activation steering for vocal effort

Living document. Working agreement: after every step, update this file and the README (results, decisions, script status, next
steps) and hand both over. Last updated: 2026-11-03.

## Question
Can a training-free steering direction, derived from EARS matched loud and regular readings, give Qwen3-TTS a continuous,
reproducible vocal-effort control that keeps the cloned speaker's identity better than stock Chatterbox's exaggeration knob,
without artifacts?

## Ground rules for v2 (decided 2026-10-31)
- Directions are extracted from the 12 TRAINING speakers only (p001 to p012).
- DEV work (choosing layers, strengths, modes) uses 3 validation speakers (p023, p045, p056) and 8 NEW dev lines
  (`dev_lines.json`, 0 overlap with the 30 test lines). Nothing is tuned on the 30 test lines.
- The FINAL evaluation uses the 10 never-touched test speakers and the 30 test lines, once, with settings frozen beforehand.
  (Numbers on the validation speakers with settings chosen on them are optimistic and must not be the headline.)
- The effort estimator is NOT validated on steered synthetic audio: report an artifact measure (blind human label,
  similarity, WER, runaway rate) beside every effort number. Effort scores of runaway or distorted clips are meaningless.
- All EARS-derived artifacts stay local and out of git: reference clips and transcripts, `directions_*.pt`, generated clips,
  ladders. Consenting adults only; no minors. Keep the repository private until a publishing decision.

## Phases and status
| # | Phase | Status |
|---|---|---|
| 0 | Mac setup, reference transcripts, recon of the model structure | done 2026-10-30 |
| 1 | Zero-shot Qwen baseline in the v1.0 harness (no steering) | generated for 3 speakers x 10 test lines (30 clips); NOT scored yet; full 5-speaker x 30-line run pending |
| 2 | Direction extraction, method A (generation-based pairs), 12 training speakers | done 2026-10-30; method B (teacher-forced natural pairs) not tried |
| 3 | Layer and strength sweeps on dev | sweeps 1 to 4 (layer 21, k 0.05 to 2.0) and the spread test generated and scored (effort, identity, WER); equal-level ladder heard; level check done, peak check and bounded-generation test pending |
| 3c | Runaway-crop test (is only the tail broken at k = 2, 4?) | done: not supported (heads degrade too) |
| 3b | Identity and WER (Step A) on steered roots | done for dev_base, dev2_L21, dev2_L14_18_21_24 and dev_L21_head |
| 4 | Final evaluation on test speakers with the 30 lines | not started (needs the test speakers downloaded) |
| 5 | Blind human check (about 50 clips, 2+ raters if possible) and write-up | not started |

## FINAL DELIVERABLE (decided 2026-11-03): a Streamlit showcase of both systems
The finish line is a dashboard that plays the curated outputs of Chatterbox (v1.0) and Qwen3-TTS steering (v2.0) in all presentably good variations, with the metrics and the negative results, to show what was tried. Files in `showcase/`:
`gen_chatterbox_dev.py` (stock Chatterbox on the same 8 dev lines, 3 dev speakers and references as Qwen), `build_showcase.py` (one common presentability filter, level-matched 16-bit audio, manifest, metrics with pass rates), `app.py` (tabs: Overview,
Compare, Ladder, Metrics, What did not work, About and ethics; optional loudness ramp; optional password), `showcase_lib.py`, `requirements.txt`, `README.md`. Tested: the filter and loudness helpers, the builder on synthetic scored roots, and the app against a stand-in
for Streamlit (it ran end to end); not tested in real Streamlit or on the real clips.
Curation is visible, not hidden: only clips passing the filter are playable, and pass rates per model and setting are shown. Qwen clips used: k = 0 (dev_base), 0.35 to 0.75 (dev2_L21), 1.0 to 1.5 (dev2_L21_trim2, trimmed); Chatterbox: exaggeration 0.3 to 1.6; optional LoRA v0 and v1 clips as the negative result.
DECISION STILL NEEDED: public or private hosting. The ground rules above (no redistribution of EARS-derived audio, keep the repository private until a publishing decision) mean the default is private or local; see `showcase/README.md`.
Now optional or dropped because of this plan: the final evaluation on the 10 untouched test speakers, the second rater, the anger and valence-orthogonalized directions (the showcase reports the manic drift as a limitation).

## Pass bar (set before running; proposed)
Provisional dev check (2026-10-31, estimator-based, one seed, dev speakers; NOT the final test): effort above +3 at clean k (yes, k >= 0.35), similarity above
effort +3 at least 0.90 (yes: 0.944 at effort 8, 0.906 at effort 20), WER near 0 (yes up to k = 1), runaways near 0 (yes up to k = 1). Not yet checked: Spearman
over the clean range, seed SD, the speech-problem flag rate, perceptual shouting, the blind artifact rate.

Spearman of effort vs k >= 0.7; effort at the top clean k at least +3 and ideally above +5; similarity above effort +3 at least
0.90 (v1.0 stock Chatterbox: 0.873 to 0.888); speech-problem flag rate at most 0.40; seed SD at most 2.5 at matched effort;
runaway (non-terminating) rate near 0; blind artifact rate (maniacal, distorted, wrong voice) low and reported.

## Environment and commands (in order)
```bash
# one-time: Qwen environment (Python 3.12)
brew install python@3.12 sox
python3.12 -m venv ~/venvs/qwen3tts && source ~/venvs/qwen3tts/bin/activate && pip install -U qwen-tts soundfile
# reference transcripts for the 5 validation references (Chatterbox env, repo root; local only)
python scripts/transcribe_refs.py
# recon (Qwen env)
python qwen_v2/qwen_recon.py --size 0.6B --ref data_local/baselines/references/p023.wav \
  --ref-transcripts data_local/baselines/references/ref_transcripts.json --speaker p023
# zero-shot baseline subset (Qwen env)
python qwen_v2/qwen_gen.py --size 0.6B --refs data_local/baselines/references --ref-transcripts data_local/baselines/references/ref_transcripts.json \
  --lines finetune/eval_lines.json --tag qwen06_base --ks 0 --seeds 0 --speakers p023 p045 p056 \
  --line-ids S02 S03 S05 S08 U01 U05 U08 C02 C06 C09 --out data_local/baselines/qwen_v2
# direction extraction from training speakers, about 1 hour of compute (Qwen env; keep the Mac awake: caffeinate -i)
python qwen_v2/qwen_extract.py --size 0.6B --eval-lines finetune/eval_lines.json --out qwen_v2/directions_06B.pt
# sweep 1 (single layers, k = 1 2 4) and sweep 2 / 3 (small k, 150-token limit); Qwen env
caffeinate -i bash qwen_v2/run_dev_sweep.sh
KS="0.05 0.1 0.2 0.35 0.5" caffeinate -i bash qwen_v2/run_dev_sweep2.sh "21"      # sweep 2 (default KS)
KS="0.6 0.75 1 1.25 1.5" caffeinate -i bash qwen_v2/run_dev_sweep2.sh "21"        # sweep 3, same root dev2_L21
KS="0.1 0.2 0.35 0.5" caffeinate -i bash qwen_v2/run_dev_sweep2.sh "14 18 21 24"  # spread test, root dev2_L14_18_21_24
# crop runaway clips to the sentence length (project env) -> roots <tag>_head, then score and listen to them like any other root
python qwen_v2/crop_runaways.py dev_L21 dev_L14 dev2_L14_18_21_24
python qwen_v2/make_ladder.py --spk p023 --line D03 --rungs dev_base:0 dev_L21_head:1 dev_L21_head:2 dev_L21_head:4
# scoring: Step A (Chatterbox env) then Step B (project env); Qwen roots need --default-ex 0
python -m efforttts.eval.asr_speaker --root data_local/baselines/qwen_v2/dev2_L21
python -m efforttts.eval.score_baseline --root data_local/baselines/qwen_v2/dev2_L21 --default-ex 0
python qwen_v2/sweep_summary.py            # per layer and k: terminated share, effort change, similarity, WER
python qwen_v2/worst_clips.py dev2_L21     # clipping, harshness, similarity; worst clips per k to listen to
python qwen_v2/make_ladder.py --spk p023 --line D03   # one WAV: the same line at increasing k
```
In zsh, do not name a shell variable `LINES`; use `LINE_IDS` and `$=LINE_IDS` to split it.

## Results so far
**Recon (2026-10-30; Mac CPU, Qwen3-TTS-12Hz-0.6B-Base, qwen-tts 0.1.1, torch 2.15, transformers 4.57.3).** Load 60 s; 2.96 s of audio
in 8 s (RTF 2.6, fp32, no flash-attn); 1.7B about 3x slower. Structure: `model.talker` (906M) = `model` (`layers` ModuleList[28],
hidden 1024, 16 heads), `text_projection`, `codec_head` (first codebook), `code_predictor` (142M, `lm_head` ModuleList[15] for the other
15 codebooks); `speaker_encoder` (9M); 16 code groups at 12 Hz. ICL prompt fields: `ref_code`, `ref_spk_embedding`, `ref_text`,
`icl_mode`, `x_vector_only_mode`. Hooks go on `model.talker.model.layers[i]`, applied at decode steps (sequence length 1) only.

**Zero-shot baseline subset (30 clips, seed 0, 10 test lines).** 30 of 30 generated, 3.8 to 9.6 s each on CPU, plus one 63 s outlier
(p056/U08, probable non-termination; check `audio_s` in `results.csv`). Not scored. Dev baseline (24 clips, 8 dev lines, k = 0): median
effort -3.49 (calm -4.34, shout -2.16, urgent -3.63), 2.61 words/s, all 24 terminated. Real EARS references for these 3 speakers on
the same estimator: regular -2.63, loud 10.49, anger 18.07, yelling 21.18.

**Direction extraction (method A; 12 training speakers x 3 reference texts x 2 styles x 2 targets = 144 generations).** The first run
planned 0 generations because of a target-selection bug (fixed: shortest usable sentences, diagnostics, hard stop). The real run
took 10,940 s of wall time (two stalls of about 100 minutes each, most likely the Mac sleeping; compute is about 4 to 5 minutes per
speaker). Direction norm grows with depth (0.67 at layer 0, 1.9 at 14, 3.65 at 21, 5.5 at 27; hidden-state norms also grow, not
recorded). Per-speaker directions agree: mean cosine to the mean direction 0.80 to 0.91 at every layer (inflated slightly because each
speaker is in the mean; random vectors would give about 0.29). Saved `qwen_v2/directions_06B.pt` (EARS-derived, local only).

**Sweep 1 (single layers, k = 1, 2, 4; 3 speakers x 8 dev lines = 24 clips per cell; 500-token limit).** Terminated clips (<= 20 s) and
paired effort change vs unsteered, terminated clips only:

| layer | k = 1 terminated | k = 1 effort change | k = 2 terminated | k = 4 terminated |
|---|---|---|---|---|
| 14 | 21/24 | +25.6 | 10/24 | 1/24 |
| 18 | 22/24 | +25.8 | 7/24 | 0/24 |
| 21 | 24/24 | +24.5 | 11/24 | 0/24 |
| 24 | 23/24 | +26.1 | 10/24 | 1/24 |
| 27 | 23/24 | +23.1 | 17/24 | 2/24 |

At k = 1 the median effort of terminated clips is 17.8 to 21.1 for every layer (calm lines 18.7 to 24 too), above real loud (10.5) and near
real yelling (21). The band of layers 20 to 23 (k = 1, 2, 4) ran to the token limit on every clip (64 to 100 s each) and was not scored.
Times above 100 s (up to 4,991 s) are the Mac sleeping, not generation. Effort numbers at k = 2 and 4 are survivors and not meaningful.

**Listening (one rater).** k = 1 on p023 (layer 21): distress appropriate to the line, slightly louder, identity completely retained,
but not shouting. Maniacal and distorted outputs come mostly from k = 2 and k = 4. Consequences: the clean range is about k <= 1; the
estimator's "effort 20" overstates shoutiness on steered audio and is not validated on synthetic speech; a uniform plateau across
calm and shout lines and across speakers looks like a texture, not graded effort, so a smooth dose-response at small k is needed.

**Listening, equal-level ladder (one rater; speaker and line not recorded; 2026-10-31; clarified by the rater the same day).** The unsteered clip is
absolutely calm, and from there the voice becomes subtly louder and more forceful as k rises, still fairly calm at k = 0.5; at k = 0.75 it is a little
louder; from k = 1.0 the loudness is very noticeable and keeps increasing steadily (1.25, 1.5); k = 2.0 introduces distortion. Second note, same day: actual clear
shouting at k = 1.0, 1.25 and 1.5; some distortion at the very END of the clip at the next rung up (reported as 1.75: `dev2_L21` holds 288 clips = 12 strengths x 24, i.e. a
sweep with k = 1.75 and 2.0 had also been run by the user; those 48 clips are generated with the 150-token cap and not yet scored) and slightly more distortion at 2.0. Reading: (1) the perceptual response is continuous and monotone from k = 0 to about 1.5, which is what a control needs; (2) the estimator and the ear
agree on direction but not on scale: the estimator called k = 0.5 "real loud" (8.3) while the ear called it a subtle increase on a calm voice, and the estimator
plateaus at k >= 1 (about 20) while the ear keeps hearing the loudness rise through 1.5; (3) the clearly audible range sits where the model starts failing
(k = 1.25: 18 of 24 terminate, similarity 0.882, WER 0.13; k = 1.5: 13 of 24, 0.864, 0.11); (4) at equal level the remaining "loudness" could be real voice
quality, or partly saturation (the raw clips are near full scale; peaks not yet measured), which the peak check will separate; (5) distortion reported at the very END of a clip points to the end-of-speech failure and not to the voice itself, so
bounding the generation length and fading out might extend the clean shouting range beyond 1.5 (to be tested).
The earlier raw-playback note (k = 1 on p023: distressed, slightly louder, not shouting) came from a different line and playback and is not reconciled yet.

**Anger direction extracted (`directions_06B_anger.pt`, 2026-11-03): 12 training speakers, regular reading references vs the scripted EARS anger clip (one per speaker), 4 targets, 192 generations, 5,058 s (no stall).**
Direction norm by layer is the same size as the loud-reading direction (layer 14: 1.75 vs 1.90; layer 21: 3.42 vs 3.65; layer 27: 5.53 vs 5.52), but the agreement between speakers is lower: mean cosine 0.66 to 0.74 at every layer
(0.677 at layer 21) against 0.80 to 0.91 for loud reading (0.825 at layer 21). Likely reason: each speaker's anger side rests on a single reference clip with anger-themed text (4 generations against 12 on the regular side), so the
per-speaker estimates are noisier. Not yet swept: whether it gives lower valence at matched arousal.

**Anger direction vs loud-reading direction, like for like (2026-11-03; layer 21; dev lines D01 to D03 x 3 speakers, n = 9 cells per k unless noted; `emotion_dim.py --group all`; Step A and B done; trimmed roots `dev2_L21_anger_trim` for k >= 0.75).**
Paired change in arousal / valence versus the unsteered clip:

| k | loud-reading direction | anger direction |
|---|---|---|
| 0.25 | +0.07 / +0.02 (k = 0.20) | +0.09 / +0.01 |
| 0.50 | +0.20 / +0.05 | +0.25 / +0.04 |
| 0.75 | +0.31 / +0.06 (n = 8) | +0.36 / +0.04 |
| 1.00 | +0.40 / +0.13 | +0.47 / +0.03 (n = 8; trimmed, n = 9: +0.46 / +0.07) |
| 1.25 | +0.42 / +0.27 | +0.42 / +0.13 (n = 8; trimmed: +0.43 / +0.15) |

(The loud-direction column reproduces the hand-computed medians in the earlier table, which validates them.) Other measures for the anger direction, k = 0.25, 0.5, 0.75, 1.0, 1.25 (all clips; trimmed in brackets for k >= 0.75): similarity 0.956, 0.943, 0.919, 0.882 (0.894), 0.857 (0.864);
WER mean 0.058, 0.061, 0.042, 0.095, 1.577 (0.057, 0.095, 1.567), WER median 0 except 0.05 at k = 1.25; effort estimator 0.1, 6.0, 13.6, 17.0 (18.2), 16.0 (15.8); words/s 2.65, 2.64, 2.74, 2.21 (2.37), 2.21 (2.38). For the loud direction at k = 1.0: similarity 0.906, WER mean 0.023, effort 20.4.
Reading: (1) at matched arousal (a gain of +0.40 to +0.47) the anger direction changes valence by +0.03 to +0.07, against +0.13 for the loud-reading direction, and at k = 1.25 by +0.13 to +0.15 against +0.27: about half the drift towards positive valence; (2) it still does not move valence DOWN as real anger,
fear and distress do (-0.22 to -0.30), it just drifts less; (3) identity at matched arousal is similar (0.894 at +0.46 against 0.906 at +0.40); (4) its useful range ends earlier: at k = 1.25 some clips are garbled even after trimming (WER mean 1.57 against 0.12 for the loud direction), so the anger
direction is usable to about k = 1.0; (5) the estimator rates it lower at the same k (17 to 18 against 20) while the arousal model rates it higher (+0.47 against +0.40): the two instruments measure different things; (6) calm lines get the highest estimator effort at k = 1.0 (20.6), so the control stays
text-independent. Caveats: n = 9 cells, one clip each; one anger reference clip per speaker; not yet heard by the rater (does D03 still sound manic with it?).

**Listening, anger direction (`dev2_L21_anger_trim` ladder; p045 D03 is the line used earlier; one rater; 2026-11-03): the best result so far. The manic state is gone and there is no distortion; the loudness decreases slightly as k rises, and the higher k sounds like
more "rational urgency" (controlled, commanding urgency, not panic or mania).** This agrees with the measurements: valence drifts about half as much as with the loud-reading direction, and arousal and the effort estimator peak near k = 1.0 and fall slightly at 1.25 (arousal +0.47 to +0.42; effort 18.2 to 15.8), so
a slight loudness decrease at higher k is expected, not an artefact of the equal-level ladder. The metrics still show some garbled clips at k = 1.25 (WER mean 1.57), so the showcase filter, not the ladder, decides which clips are playable. The earlier open question about the last word "out" was not answered explicitly; the
trim is the same one used here and the rater reported no distortion.

**Chatterbox on the same dev lines, speakers and references as Qwen (`chatterbox_dev`, 2026-11-03; 120 clips, stock, cfg 0.5, seed 0; Step A and B done; preflight reports every showcase source ready).**
Median effort / similarity / WER mean / words per second by exaggeration: 0.3: -3.85 / 0.935 / 0.015 / 2.61; 0.5: -2.83 / 0.932 / 0.012 / 2.85; 0.8: -1.11 / 0.922 / 0.037 / 3.13; 1.2: 4.03 / 0.921 / 0.060 / 3.26; 1.6: 7.76 / 0.891 / 0.080 / 2.72.
Effort at 1.6 by speaker: p023 9.06, p045 9.11, p056 4.64 (the knob reaches real-loud level only on two speakers). On this subset the identity cost of stock Chatterbox is smaller than on the full 5-speaker x 30-line baseline
(0.932 to 0.891, a drop of 0.04, against 0.936 to 0.838), so the earlier "0.84 at the top" must be quoted as the FULL baseline only. Like for like on the dev set: at estimator effort of about 8, stock Chatterbox (exaggeration 1.6) has similarity 0.891, Qwen with the
loud-reading direction (k = 0.5) 0.944 (unsteered 0.949); Qwen reaches effort 17 to 20 with similarity 0.88 to 0.91, which Chatterbox's range does not reach.

**Showcase build, first real run (2026-11-04): 434 of 534 candidate clips passed the common filter; audio folder 65 MB (about 150 KB per clip).** Pass rate (clips shown / candidates) by setting: Chatterbox dev set 0.3: 23/24, 0.5: 23/24, 0.8: 20/24, 1.2: 19/24, 1.6: 16/24 (0.67);
Chatterbox full baseline (5 speakers x 30 lines) 0.88, 0.87, 0.88, 0.75, and 0.42 at 1.6 (188 of 450, three seeds); Qwen loud-reading direction k = 0: 22/24, 0.35: 23/24, 0.5: 22/24, 0.6: 21/24, 0.75: 17/24, 1.0: 23/24, 1.25: 18/24, 1.5: 15/24 (0.63);
Qwen anger direction k = 0.25: 22/24, 0.5: 19/24, 0.75: 20/24, 1.0: 19/24, 1.25: 13/24 (0.54, WER mean 1.57 over all candidates); LoRA v0 (3 settings) 0.78, 0.62, 0.81; LoRA v1 0.78, 0.78, 0.67. Among shown clips the median valence is 0.41 to 0.43 for the anger direction up to
k = 1.0 (0.61 at 1.25) against 0.51 at k = 1.0 and 0.67 at 1.25 for the loud-reading direction. At comparable effort the Qwen settings keep far more clips than Chatterbox (effort about 8: Qwen k = 0.5 92% vs Chatterbox 1.6 67%; Qwen k = 1.0 96%).
Even unsteered Qwen loses 2 of 24 clips to the filter (reasons in `showcase/curation_log.txt`), so the filter is strict by design. Warnings about non-data WAV chunks were silenced in `showcase_lib.py`.

**Why clips were removed by the filter (`curation_log.txt`, first real build, 100 clips removed of 534; each clip counted once under the first rule it fails: audio length, missing scores, WER, similarity, pace, duplicated words):**
Chatterbox dev (19 removed): WER above 0.2: 8, slower than 1.6 words/s: 6, similarity below 0.8: 4, duplicated words: 1. Chatterbox LoRA v0 (14): WER 12, slow 2; LoRA v1 (7): WER 7. Qwen loud-reading direction (31): WER 11, did not stop within 10 s: 4, duplicated words: 5, similarity below 0.8: 6, slow: 5.
Qwen anger direction (29): WER 14, duplicated words 2, similarity 6, slow 7. WER above 0.2 is the leading reason everywhere, including unsteered Qwen; with 6 to 9 word lines two recognizer errors already exceed 0.2, and Whisper is less reliable on shouted speech, so the filter is conservative (it
drops some good clips). The builder now takes `--max-wer`, `--min-sim` and `--min-wps`, exports two removed clips per model and reason (`audio_failed/`), and the app plays them under "What did not work" so the curation is audible.

**Sweeps 2 and 3 (layer 21 only, k = 0.05 to 1.5, 150-token limit, 10 strengths x 24 clips = 240 clips; Step A and B done).** Effort is the
median over all clips of the root (at k >= 1.25 it includes runaway clips); paired change and terminated counts come from `sweep_summary.py`
(terminated clips only); similarity is to the speaker's centroid of real speech.

| k | effort median | paired change | similarity | WER mean | words/s | terminated |
|---|---|---|---|---|---|---|
| 0 (dev_base) | -3.49 | | 0.949 | 0.063 | 2.61 | 24/24 |
| 0.05 | -1.70 | +1.5 | 0.951 | 0.039 | 2.88 | 24/24 |
| 0.10 | -1.65 | +1.8 | 0.951 | 0.052 | 2.77 | 24/24 |
| 0.20 | -0.89 | +3.1 | 0.952 | 0.044 | 2.86 | 24/24 |
| 0.35 | 3.15 | +7.9 | 0.950 | 0.048 | 2.91 | 24/24 |
| 0.50 | 8.26 | +12.5 | 0.944 | 0.020 | 2.63 | 24/24 |
| 0.60 | 14.68 | +19.0 | 0.930 | 0.048 | 2.65 | 24/24 |
| 0.75 | 15.40 | +19.8 | 0.923 | 0.064 | 2.68 | 22/24 |
| 1.00 | 20.43 | +24.5 | 0.906 | 0.023 | 2.85 | 24/24 |
| 1.25 | 20.49 | +23.5 | 0.882 | 0.130 | 2.11 | 18/24 |
| 1.50 | 18.63 | +24.9 | 0.864 | 0.109 | 1.20 | 13/24 |
| 1.75 | 17.40 | +22.9 | 0.840 | 2.892 (median 0.06) | 1.22 | 13/24 |
| 2.00 | 15.29 | +22.9 | 0.770 | 5.995 (median 0.35) | 0.92 | 11/24 |

**Above k = 1, looking only at clips that end on their own** (the all-clip similarity above is dragged down by the runaway clips, whose 12 s heads contain babble):

| k | ended on their own | similarity | WER median |
|---|---|---|---|
| 1.00 | 24/24 | 0.906 | 0.0 |
| 1.25 | 18/24 | 0.905 | 0.0 |
| 1.50 | 13/24 | 0.883 | 0.0 |
| 1.75 | 13/24 | 0.837 | 0.0 |
| 2.00 | 11/24 | 0.819 | 0.3 |

So at k = 1.25 and 1.5, where the rater heard clear shouting, the clips that stop normally keep their identity (0.905, 0.883) and intelligibility (WER 0): the
limit there is the share that never stops (25% and 46% runaways), not the voice. From k = 1.75 the voice itself degrades even in clips that stop (0.837, 0.819, WER 0.3
at k = 2), which matches the crop result at k = 2 and the rater's end-of-clip distortion.

Real EARS speech for the same 3 speakers on the same estimator: regular -2.63, loud 10.49, anger 18.07, yelling 21.18; similarity to the centroid:
regular 0.976, neutral 0.962, loud 0.954, anger 0.934, ecstasy 0.920. Findings (dev speakers, dev lines, one seed, estimator-based):
- A smooth, monotone dose-response: flat to k about 0.2, then rising through real loud (k about 0.5) and anger (k about 0.6 to 0.75) to a plateau
  near 20 (k >= 1), i.e. the estimator's yelling level; the plateau means the estimator cannot rank anything beyond k = 1.
- Identity is untouched to k = 0.35 (0.950), costs 0.005 at k = 0.5, 0.019 at 0.6 and 0.043 at k = 1 (real speech loses 0.022 at loud and 0.042 at
  anger). Stock Chatterbox at effort about 7 had similarity 0.838, 0.10 below its own default. WER stays near 0 up to k = 1.
- Effort rises on calm lines as well as shout lines (k = 0.5: calm 7.0, shout 7.5, urgent 9.0), so the control is text-independent. All three speakers
  respond (k = 0.5: p023 8.3, p045 11.6, p056 4.9, the last starting from -8.0).
- Beyond k = 1 termination, identity, pacing and WER degrade together (k = 1.25: 18 of 24 terminate, similarity 0.882; k = 1.5: 13 of 24, 0.864).
- CAUTION: at k = 1 the estimator says "yelling" while the one listener heard distress, slightly louder, not shouting. The estimator is level-free
  and unvalidated on steered synthetic speech, so these effort numbers are provisional until the ladder listening, the level check and blind labels.

**Level check (`level_by_k.py dev2_L21`, 2026-10-31): steering changes the speech LEVEL enormously.** Unsteered level is -31.5 dBFS (95th percentile of
25 ms frame RMS). Paired median change versus the unsteered clip, by k: 0.05: +0.3 dB; 0.10: +1.5; 0.20: +3.5; 0.35: +8.3; 0.50: +13.8; 0.60: +20.8;
0.75: +20.4; 1.00: +25.3; 1.25: +26.4; 1.50: +26.5 (p10 to p90 at k = 1: +21 to +29 dB). Real EARS loud reading is about +9 dB over regular reading.
Consequences: (1) k about 0.35 matches real loud in LEVEL, yet its spectral effort is only 3.1 against real loud 10.5, so level moves before spectral effort,
unlike human speech; (2) at k >= 0.6 the level is within about 11 dB of full scale (-6 dBFS at k = 1), so peaks probably clip when played, which may explain
part of the "distortion"; (3) the estimator is level-free, but the model's own effort and level changes are coupled, so a high effort score here may be
partly "much louder and saturated"; (4) one listener called k = 1 "slightly louder", which does not fit +25 dB (playback normalization or a different file?).
The unsteered output level is also low (-31.5 dBFS), so a final loudness stage is needed anyway.

**Bounded-generation sweep (`dev2_L21_b`: layer 21, k = 1.0, 1.25, 1.5, 1.75, 2.0; `--bound-factor 1.5`; 120 clips; generated 2026-10-31, NOT yet scored).** Each line was
capped at 12.5 x 1.5 x (words / 2.4) + 8 tokens (54 to 93 tokens, i.e. 4.3 to 7.4 s for the dev lines) and a clip that reached the cap was faded out over 150 ms.
Counts of capped clips from the generation log: k = 1.0: 0/24; 1.25: 8/24; 1.5: 14/24; 1.75: 15/24; 2.0: 13/24 (by speaker: p023 12, p045 12, p056 26 of 120).
Capped is not the same as runaway: the cap implies a minimum pace of about 1.6 words/s, and shouted speech can be slower than that (all-clip pace at k = 1.5 was 1.2
words/s), so some capped clips may have lost their last word or words. LISTENING RESULT (2026-10-31, one rater): at k = 1.25 and 1.5 the capped clips LOSE THEIR LAST WORD, so the 1.5 cap cuts real speech and is too tight (it assumes at least about
1.6 words/s). Step A (WER on capped clips) can quantify it; `sweep_summary.py` now
reports capped clips separately for roots ending in `_b` (sim_capped, wer_capped).

**Trim-to-text guard (`dev2_L21_trim`: the 120 clips of `dev2_L21` with k >= 1.0, cropped 0.25 s after the last target word via Whisper word timestamps; Step A and B done).**
Trim status over all k: 58 trimmed, 58 unchanged, 4 no_match (2 at k = 1.75 and 2 at k = 2.0). Effect on all 24 clips per k (untrimmed -> trimmed):

| k | trimmed / unchanged / no_match | similarity | WER mean (median) | words/s | clips <= 10 s after trimming |
|---|---|---|---|---|---|
| 1.00 | 5 / 19 / 0 | 0.906 -> 0.906 | 0.023 -> 0.023 (0) | 2.85 -> 2.85 | 24/24 |
| 1.25 | 13 / 11 / 0 | 0.882 -> 0.894 | 0.130 -> 0.099 (0) | 2.11 -> 2.49 | 24/24 |
| 1.50 | 16 / 8 / 0 | 0.864 -> 0.871 | 0.109 -> 0.114 (0) | 1.20 -> 2.50 | 22/24 |
| 1.75 | 12 / 10 / 2 | 0.840 -> 0.844 | 2.892 -> 0.191 (0) | 1.22 -> 2.51 | 21/24 |
| 2.00 | 12 / 10 / 2 | 0.770 -> 0.807 | 5.995 -> 2.808 (0.3) | 0.92 -> 2.45 | 22/24 |

Effort after trimming (estimator, saturated): 21.0, 21.1, 19.7, 20.5, 18.4. For comparison, the capped clips of the bounded root (`dev2_L21_b`, cap 1.5x) had similarity 0.840,
0.828, 0.851, 0.810 and WER median 0.06, 0.0, 0.10, 0.63 at k = 1.25 to 2.0, against 0.908, 0.884, 0.837, 0.819 for its clips that stopped by themselves; the rater heard the last
word lost in capped clips. Reading: trimming turns the runaway clips into usable ones at k = 1.25 (24 of 24, similarity 0.894, WER median 0) and mostly at 1.5 (22 of 24, 0.871); the
formerly runaway clips keep slightly lower identity than the ones that stopped naturally (0.894 vs 0.905 at 1.25; 0.871 vs 0.883 at 1.5); from k = 1.75 the voice itself is degraded
(WER mean 0.19 at 1.75, 2.8 at 2.0) and trimming cannot repair it. Remaining weak spots: 2 to 3 clips per k >= 1.5 stay longer than 10 s or fail to align.
The equal-level ladder of the trimmed clips (p045 D03) was built (durations 2.9, 3.4, 3.2, 3.9, 4.2, 2.7 s for k = 0, 1, 1.25, 1.5, 1.75, 2.0; every steered rung scaled by about
-23 dB); the rater's notes on it (p045 D03, "Fire in the kitchen, everybody out!"): at k = 1.0 the last word ends well; from k = 1.25 the last word "out" is clipped progressively more but never
fully lost, i.e. the trim cut slightly early (Whisper's end time of a shouted word is early), so `trim_to_text.py` now cuts where the voice actually stops (search up to 0.6 s past Whisper's end,
60 ms fade; minimum pad 0.1 s); the dev2_L21_trim numbers above come from the old fixed-pad version and should be regenerated into a new root (`--suffix _trim2`).

**Trim v2 (`dev2_L21_trim2`: cut where the voice stops, up to 0.6 s past Whisper's last-word end; Step A and B done).** Status: 55 trimmed, 61 unchanged, 4 no_match (2 each at k = 1.75 and 2.0). Scores
(24 clips per k): similarity 0.905, 0.894, 0.873, 0.839, 0.819 and WER mean 0.023, 0.123, 0.074, 0.201, 4.095 at k = 1.0, 1.25, 1.5, 1.75, 2.0 (WER median 0 except 0.3 at k = 2.0), words/s 2.85, 2.42, 2.27, 2.34, 2.41; effort
20.4, 21.1, 20.0, 19.9, 18.0. These are within noise of the fixed-pad version (similarity 0.906, 0.894, 0.871, 0.844, 0.807), so the metrics cannot tell whether the last word is now whole: that needs the ear.
Ladder p045 D03 built: k = 1.25 now 3.6 s (was 3.2 s), k = 1.5 3.8 s (was 3.9 s). Listening notes on "out" pending.

**Emotion-drift proxy, first result (`emotion_drift.py`, 4-class IEMOCAP classifier, `dev2_L21_trim2`, lines D01 to D03, one clip per speaker, line and k).** Unsteered clips are already half "happy" for this classifier
(mean hap 0.48, neu 0.50, ang 0.005, sad 0.02; top class neu 5, hap 4 of 9 clips). Steered clips are classified "hap" with probability 0.86 to 1.00 in almost every cell from k = 1.0 to 2.0, on all three speakers, and
"ang" stays near 0 (exceptions: p023 D01 k = 1.25 ang 0.25; p045 D02 k = 1.0 ang 0.30; p056 D03 at k = 1.75 and 2.0 flip to "neu" 0.67 and 0.99, probably garbled or weak clips). Reading: the steering pushes towards the
classifier's high-arousal positive class (happy or excited), not anger, which is consistent with the "maniac" impression, but the classifier saturates and does NOT separate D03 from D01 and D02 on p045 (hap 0.91 to
1.00 vs 0.95 to 0.995 vs 0.67 to 0.99), and it calls some unsteered calm clips happy, so it cannot confirm the line dependence and cannot be used as a manic detector. It is replaced by a dimensional model:
`emotion_dim.py` (audeering wav2vec2-large-robust-12-ft-emotion-msp-dim: arousal, dominance, valence about 0..1; CC BY-NC-SA 4.0, research only; trained on natural podcast speech), run with `--anchors` to score REAL EARS regular,
loud, neutral, anger, fear, distress, pain, ecstasy, amusement and sadness clips of the same speakers as reference points.

**Dimensional emotion results (`emotion_dim.py`, audeering arousal / dominance / valence model; 2026-11-03; loaded and verified: 234 weights identical to the checkpoint; trained on natural podcast speech, research use).**
Real EARS clips of the three dev speakers (first 12 s; arousal / dominance / valence): regular 0.31 / 0.42 / 0.46; loud 0.51 / 0.55 / 0.46; neutral 0.18 / 0.33 / 0.27; anger 0.70 / 0.71 / 0.24; fear 0.48 / 0.51 / 0.19;
distress 0.62 / 0.66 / 0.16; pain 0.45 / 0.55 / 0.08; ecstasy 0.71 / 0.73 / 0.63; amusement 0.51 / 0.58 / 0.71; sadness 0.20 / 0.32 / 0.17. So the model separates real emotions sensibly: anger and distress are high arousal
with LOW valence, ecstasy is high arousal with HIGH valence. Unsteered synthetic clips (dev lines D01 to D03, n = 9): 0.40 / 0.48 / 0.37.
Steered clips, paired change vs the unsteered clip of the same speaker and line (median over the 9 speaker x line cells, one clip each, from the printed per-cell table; k = 0.75 has 8 cells):

| k | change in arousal | change in valence |
|---|---|---|
| 0.05 | +0.04 | +0.01 |
| 0.10 | +0.02 | 0.00 |
| 0.20 | +0.07 | +0.02 |
| 0.35 | +0.10 | +0.01 |
| 0.50 | +0.20 | +0.05 |
| 0.60 | +0.27 | +0.08 |
| 0.75 | +0.32 | +0.06 |
| 1.00 | +0.40 | +0.13 |
| 1.25 | +0.42 | +0.27 |

Real speech relative to real regular (arousal / valence): loud +0.20 / 0.00; anger +0.39 / -0.22; fear +0.17 / -0.27; distress +0.31 / -0.30; pain +0.14 / -0.38; ecstasy +0.40 / +0.17; amusement +0.20 / +0.25.
Reading: (1) arousal rises smoothly and does not plateau at k = 1 (it keeps rising to 1.25 and above, values up to 1.0 and beyond at k >= 1.5, outside the model's training range), which matches the rater's ears better
than the saturating effort estimator; k = 0.5 matches REAL LOUD speech on both axes (+0.20 / +0.05 against +0.20 / 0.00), and k = 1.0 matches the arousal of real anger and ecstasy (+0.40); (2) VALENCE drifts upward from k about 1.0
(+0.13) and clearly at 1.25 (+0.27, beyond real ecstasy +0.17), while real anger, fear and distress go DOWN by 0.22 to 0.30: at anger-level arousal the steered voice is closer to ecstasy than to anger; the mean valence of the trimmed
clips at k = 1.0, 1.25, 1.5, 1.75, 2.0 is 0.53, 0.62, 0.66, 0.62, 0.67 against 0.37 unsteered (real ecstasy 0.63, real fear 0.19); (3) this quantifies the "manic" impression: high arousal with high valence, wrong for a
fear, alarm or anger line; on p045 D03 the valence is 0.76 to 0.84 at k = 1.25 to 1.75 against 0.41 to 0.53 on D01, as the rater heard, but p056 shows high valence on D02 and D03 as well, so the drift is general and the
D03 impression is partly a matter of how strong it is on that line; (4) acting window: k about 0.75 to 1.0 has clearly raised arousal with valence change still at or below +0.13; from k = 1.25 the drift is large. Caveats: one clip per
cell, a model trained on natural speech and used on synthetic shouting, arousal and dominance values above 1 at high k, and no human valence labels yet.

**EMOTION FINDING (same rater, same ladder): the emotion goes wrong at higher k.** For a fear or urgency line ("Fire in the kitchen, everybody out!") the steered voice sounds like a
maniac at the higher variations ("certified maniac"), while the earlier raw note at k = 1.0 on p023 D01 found distress appropriate to the line. So the effort direction carries an
arousal that is not neutral: past about k = 1.0 to 1.25 it may turn an urgent or frightened line into an excited or manic one. The estimator, similarity and WER cannot see this. Consequences:
(1) the usable window for ACTING is narrower than the window for loudness: roughly k <= 1.0 (maybe 1.25) for this line; (2) "emotion fits the line" and "maniacal" must be human labels in the
evaluation, and an automatic emotion-drift measure (a speech-emotion classifier, not yet chosen or verified) would help; (3) mitigation ideas: an anger-based direction, a direction orthogonalized
against excited or joyful emotion directions extracted from the EARS emotion clips, and per-line k chosen by the director. Follow-up listening (same rater; the other shout lines were ALSO heard on p045, so speaker is not a confound): a little of the manic sound is present on them too, but not nearly as evident as on
D03. Within one speaker the line is therefore what matters: "Fire in the kitchen, everybody out!" is pushed to a manic delivery far more than the other shout lines. The speaker-sensitivity idea
(p045 responds about 1.7x more at the same k: effort at k = 0.6 is 19.2 vs 10.3 for p023 and 11.5 for p056) is not supported by this observation and stays only as a separate calibration question.
Working hypothesis: steering amplifies the arousal the text already implies (at k = 1.0 the distress fit the line), and for an alarm line such as fire that amplified panic reads as manic at
higher k. Still untested across speakers: D03 on p023 and p056.

**Spread test (layers 14, 18, 21 and 24 together; k = 0.1, 0.2, 0.35, 0.5 on EACH layer; 150-token limit; 96 clips).** Runaways are
identifiable from generation time (about 29 to 33 s, against 6 to 17 s for clips that ended): 0, 2, 7 and 13 of 24 ran long at k = 0.1, 0.2,
0.35, 0.5 per layer (total push 0.4, 0.8, 1.4, 2.0), i.e. 24, 22, 17 and 11 of 24 terminated. For comparison single-layer 21 terminated
120 of 120 up to k = 0.5 and 24 of 24 at k = 1, and 11 of 24 at k = 2 (sweep 1). So termination tracks the TOTAL push and spreading
over four layers did not widen the clean range. Scored: effort 4.97, 16.35, 20.57, 17.72 at k = 0.1, 0.2, 0.35, 0.5 per layer; similarity 0.949, 0.921,
0.862, 0.828; WER mean 0.04, 0.10, 0.29, 2.09; words/s 2.70, 2.75, 2.14, 0.88. At matched effort the single layer is as good or better (layer 21,
k = 0.6: effort 14.7, similarity 0.930, WER 0.05, against spread k = 0.2: effort 16.4, similarity 0.921, WER 0.10), so spreading brings no benefit.

**Runaway-crop test (`dev_L21_head`: each clip cut to 1.4 x the unsteered duration + 0.6 s; Step A and B done).** The tail-only hypothesis is NOT
supported. Heads at k = 1: effort 20.4, similarity 0.906, WER mean 0.023 (as before). At k = 2: effort 17.0, similarity 0.815, WER median 0.30,
2.21 words/s. At k = 4: effort 9.5, similarity 0.662, WER median 1.0, 1.67 words/s. So the line itself is already degraded at k = 2 and destroyed at
k = 4; the maniacal sound is not only the runaway tail. (The crop window can still include the start of the tail; a tighter crop or a forced-alignment
crop would sharpen this, but it does not change the working range.)

## Decisions log
- 2026-10-30: Chatterbox work closed as v1.0 (tag `v1.0-chatterbox`); partial and full Chatterbox fine-tuning shelved in
  `finetune_patch/v1_1_shelved/` (never run). v2.0 = steering Qwen3-TTS. Checked beforehand that no Qwen or steering work existed.
- 2026-10-30: develop on the 0.6B model on the Mac CPU (RTF about 2.6); move to 1.7B on a T4 only if 0.6B is the bottleneck.
- 2026-10-30: extraction method A first (generation-based pairs, as in arXiv 2610.07647); hooks are additive and decode-only.
- 2026-10-31: dev = validation speakers + new dev lines; final = test speakers + the 30 lines (see ground rules).
- 2026-10-31: sweeps from number 2 on use `--max-new-tokens 150` so a runaway costs about 30 s, not 100 s.
- 2026-10-31: bug fixed in `sweep_summary.py`: its runaway cutoff was 20 s, but a 150-token runaway lasts only 12 s, so it would have
  counted those as terminated. The default is now 10 s (valid for the dev lines; check the printed unsteered maximum for longer lines).
- 2026-10-31: Qwen roots are scored with `--default-ex 0`; the strength k is stored in the `exaggeration` column.

- 2026-10-31: working range on layer 21 is k <= 1.0; candidate operating points on the estimator's scale: k about 0.5 (real-loud level), 0.6 to 0.75
  (anger-like), 1.0 (yelling-like, identity cost 0.043). Layer 21 stays the provisional single layer; spreading over layers is dropped.
- 2026-10-31: bug fixed in `sweep_summary.py` and `worst_clips.py`: the clip-level similarity column is `sim_to_centroid`, not `sim_centroid`.
- 2026-10-31: trim-to-text adopted as the end-of-speech guard. Working range for clean shouting by ear and metrics: k = 1.0 (no trim needed: 24/24, similarity 0.905), 1.25 (24/24
  after trimming, 0.894), 1.5 (22/24, 0.871); k >= 1.75 is not recommended (voice degrades). Provisional default for the shouting setting: k = 1.25 with trimming.
- 2026-10-31: a hard length cap (`--bound-factor 1.5`) is rejected because it truncates the last word of shouted clips; the end-of-speech guard is to be done by trimming loosely capped
  clips to the end of the last target word with ASR timestamps (`trim_to_text.py`; in production this could also stop generation when the text is complete).
- 2026-10-31: the limit at k = 1.25 to 1.5 is termination (25% and 46% runaways), not identity or intelligibility of the clips that do stop; from k = 1.75 the voice degrades too.
  Working range for clean shouting stays k 1.0 to 1.5 and the bounded-generation test decides whether all clips can use 1.25 to 1.5.
- 2026-10-31: perceptual operating points on layer 21 by one rater at equal level: unsteered absolutely calm, subtle steady increase up to k = 0.5, a little louder at 0.75,
  clearly loud from 1.0 rising to 1.5, distortion at 2.0. Second note: clear shouting at k = 1.0, 1.25 and 1.5, distortion creeping in at the very end of the clip above that.
  The estimator is no longer trusted as the effort measure (it overstates k = 0.5 and is blind above k = 1); blind forcefulness ratings are to be collected.
- 2026-10-31: level is to be treated as a SEPARATE knob from voice quality: reference clips are level-matched during extraction (`qwen_extract.py
  --match-level -26`), the output level is normalized in post and then set by a controlled gain scaled with the target effort, and listening uses an
  equal-level ladder (`make_ladder.py --equal-level`) next to the raw one. Peaks and over-range clips are now reported by `level_by_k.py`.
- 2026-10-31: `qwen_extract.py` gained `--loud-emotion anger` (scripted EARS anger clips as the loud side) for a second, stronger direction.

## Open questions and next steps (priority order)
0. SHOWCASE (see above): (a) `python showcase/gen_chatterbox_dev.py` in the Chatterbox env, then Step A and Step B on `chatterbox_dev/cfg0.50` with `--default-ex 0.5`; (b) `python showcase/build_showcase.py --include-finetune` (project env);
   (c) `streamlit run showcase/app.py`; (d) decide public or private and deploy accordingly. Still wanted for the write-up text inside the app: the last-word verdict on the `_trim2` ladder.
1. Perceptual calibration (equal-level ladder heard, see results): the ear hears a continuous rise from k = 0 to about 1.5 and no shouting yet; the estimator
   over-calls the middle (k = 0.5 "real loud") and saturates at k >= 1. Still to do:
   a blind, two-rater version with a forcefulness rating (1 to 5) at equal level, and an artifact label, to replace the estimator as the effort measure for steered audio.
1b. Level and clipping: re-run `python qwen_v2/level_by_k.py dev2_L21 dev_L21_head` (now also prints peaks and over-range clips). The equal-level ladder has been heard (above). Still to do: say how the RAW files were played (a loudness-normalizing
   player would explain "slightly louder" at k = 1 against +25 dB measured), and whether the raw ladder sounds clipped.
1b2. Bounded-generation test at k 1.25 and 1.5 (offline): `python qwen_v2/crop_runaways.py dev2_L21`, then Step A and Step B on `dev2_L21_head`, then
   `sweep_summary.py`. If the cropped heads keep similarity >= 0.90 and WER near 0, the stopping failure (not the voice) is what limits the perceptual range, and a
   duration-bounded generation or an end-of-speech guard would unlock k up to 1.5.
1b3. (generated; listening says a 1.5 cap cuts the last word, so a hard cap is not enough) Bounded generation (new `qwen_gen.py --bound-factor`, caps max_new_tokens per line from the word count and fades out a clip that reaches the cap; the repo copies of
   `qwen_gen.py`, `run_dev_sweep2.sh` and `crop_runaways.py` must be the new versions, else the env vars are silently ignored):
   `EXTRA="--bound-factor 1.5" TAG_SUFFIX="_b" KS="1 1.25 1.5 1.75 2" caffeinate -i bash qwen_v2/run_dev_sweep2.sh "21"` (root `dev2_L21_b`, about 15 minutes; the first printed
   line must say `dev2_L21_b` and `120 pending jobs`), then Step A and B on `dev2_L21_b`, `sweep_summary.py`, and the equal-level ladder with `--rungs dev_base:0 dev2_L21_b:1 ...`.
   Also a tighter offline crop: `crop_runaways.py dev2_L21 dev_L21 --factor 1.15 --pad 0.3 --suffix _tight`.
1b4. (done 2026-10-31) k = 1.75 and 2.0 in `dev2_L21` scored; result above: the limit at 1.25 to 1.5 is the share that never stops, not identity. This makes 1b3 (bounded
   generation) the top priority: if capped and faded clips at k = 1.25 and 1.5 keep similarity >= 0.88 and WER near 0, the shouting range is usable up to 1.5 for all clips.
1b5. (done; re-run as `dev2_L21_trim2` with the voice-offset end; scores in results; listening on the last word of p045 D03 at k = 1.25 and 1.5 pending) Trim-to-text guard (`trim_to_text.py`, Chatterbox env): take the loosely capped clips already in `dev2_L21` (150 tokens = 12 s, the whole sentence plus the tail), transcribe with
   Whisper word timestamps, align to the target text and crop 0.25 s after the last target word. `python qwen_v2/trim_to_text.py dev2_L21` (about 10 minutes; only k >= 1.0), then Step A and B
   on `dev2_L21_trim`, `sweep_summary.py`, and an equal-level ladder with `--rungs dev_base:0 dev2_L21_trim:1 dev2_L21_trim:1.25 dev2_L21_trim:1.5 dev2_L21_trim:1.75 dev2_L21_trim:2`.
   Success: all 24 clips per k end after their last word, similarity at k = 1.25 and 1.5 near 0.88 to 0.90, WER near 0, the last word present and the end clean.
1e0. LISTEN: compare the two directions on the problem line: `python qwen_v2/make_ladder.py --spk p045 --line D03 --equal-level --rungs dev_base:0 dev2_L21:0.75 dev2_L21:1 dev2_L21_anger:0.75 dev2_L21_anger_trim:1 dev2_L21_anger_trim:1.25` (is the manic sound weaker with the anger direction at similar loudness? where does it garble?).
1e. EMOTION (new): (a) answered in part: on p045 the manic sound is strong on D03 and weaker on D01 and D02, so it is line-dependent; next, D03 on p023 and p056 (equal-level ladders) to see whether it holds across speakers, and
   `emotion_drift.py --group both` to quantify; (b) laughter check on the transcripts, e.g. share of hypotheses at each k containing ha, laugh,
   brackets or asterisks; (c) add the labels "emotion fits the line" and "maniacal" to the blind check; (d) try the anger direction and an emotion-orthogonalized direction and compare the manic rate at
   equal effort; (e0) DONE (results above): valence drifts up from k about 1.0, arousal keeps rising; the acting window is about k 0.75 to 1.0.
(e1) (DONE, see the anger-versus-loud table: about half the valence drift at matched arousal, usable to k about 1.0; listening on D03 NEXT) to widen the window: a direction whose valence stays low. Step 1 (about 45 to 60 minutes of CPU): `caffeinate -i python qwen_v2/qwen_extract.py --size 0.6B --eval-lines finetune/eval_lines.json --loud-emotion anger --n-targets 4 --out qwen_v2/directions_06B_anger.pt`
   (add `--speakers p001 ... p008` to halve the time), then sweep it: `DIRECTION=qwen_v2/directions_06B_anger.pt TAG_SUFFIX="_anger" KS="0.25 0.5 0.75 1 1.25" EXTRA="--max-new-tokens 150" caffeinate -i bash qwen_v2/run_dev_sweep2.sh "21"`,
   score with `emotion_dim.py` (valence at matched arousal) and Step A and B. Step 2 only if needed: also extract `--loud-emotion extasy`, then `orthogonalize_direction.py` (valence direction = ecstasy minus anger at matched arousal, projected out of the loud-reading direction).
   (e) automatic emotion-drift proxy: `emotion_drift.py` (superb/wav2vec2-base-superb-er, Apache-2.0, 4 IEMOCAP classes, about 63% accuracy; not validated on synthetic shouting):
   `python qwen_v2/emotion_drift.py dev2_L21_trim2 --by-line --line-ids D03 D01 D02` in the Chatterbox env prints, per k (and line), mean class probabilities and the paired change versus the
   unsteered clip (does hap or ang rise on a fear line?). Direction of drift only; compare with the rater's labels.
1c. Level-matched direction (about 1 hour on the CPU): `python qwen_v2/qwen_extract.py --size 0.6B --eval-lines finetune/eval_lines.json --match-level -26
   --out qwen_v2/directions_06B_lm.pt`; then the anger variant with `--loud-emotion anger --n-targets 4`. Sweep each at small k and compare effort at EQUAL level.
1d. Level stage (written: `level_stage.py`, tested on synthetic audio): `python qwen_v2/level_stage.py dev2_L21_trim --gain-db 0 --gain-db 6 --gain-db 9 --gain-db 12` writes `<tag>_lvl<G>` roots at
   the unsteered level plus G dB with a peak limit of 0.98; listen at k = 1.25 for the G that sounds like a natural shout (real loud is about +9 dB). Also run the peak check
   (`level_by_k.py dev2_L21`, still pending).
2. `worst_clips.py` on `dev2_L21` (and the `_head` roots) to check that the automatic measures agree with the ears; then a blind label
   (ok / maniacal / distorted / wrong voice / not loud enough) on about 50 clips, ideally two raters, to validate the estimator on steered audio.
3. If the clean range does not reach shouting: a direction from the EARS anger clips (change the extractor's loud reference to
   `emo_anger_sentences`); steer the code predictor's layers (higher codebooks may carry strain); the 1.7B model on a T4 (daily
   Colab allowance); a loudness stage in post (adds loudness, not effort).
4. Mitigations for the stopping problem if higher k is needed: remove the end-of-speech component from the vector (codec_head row
   2150), clamp instead of add, a norm cap, or a ramp that fades out the steering.
5. Zero-shot baseline for all 5 validation speakers x 30 lines (Step A and B) for the comparison against v1.0.
6. Download the 10 test speakers, build references and transcripts, freeze settings, run the final evaluation (5 seeds if affordable).
7. Blind human check, 2+ raters, with labels ok / maniacal / distorted / wrong voice / not loud enough; report the artifact rate.
8. Re-run the prior-art search before any release (arXiv 2610.07647 overlaps in base model and quantity).

## Risks
- The effort direction is not emotion-neutral: at k >= 1.25 an urgent line can sound manic (listening, 2026-10-31); invisible to the estimator, similarity and WER.
- Overlap with arXiv 2610.07647 (same base model and quantity, framed as Lombard intelligibility). Our angle: acting-level loud
  delivery in a cloned voice, identity and realism evaluation with human labels, the director.
- The effort estimator can be fooled by artifacts (see ground rules); one rater so far (kappa 0.53 test-retest in v1.0).
- Steering is additive at every decode step and compounds through the key-value cache; large k leaves the model's familiar path.
- Qwen3-TTS ICL cloning needs a reference transcript (Whisper, local only); the reference prompt carries its own style and may fight steering.
- bf16 is unavailable on a T4 (fp16 or fp32); long runs on the Mac need `caffeinate -i` and the lid open.

## Script inventory
| script | purpose | status |
|---|---|---|
| `scripts/transcribe_refs.py` | Whisper transcripts of the 5 references (Chatterbox env) | ran |
| `qwen_recon.py` | load, clone, structure print | ran |
| `qwen_gen.py` | generation in the v1.0 layout; `--ks`, `--direction`, `--layers` | ran (baseline and steered) |
| `qwen_extract.py` | direction extraction, method A | ran after a fix |
| `dev_lines.json` | 8 dev lines (shout, urgent, calm) | in use |
| `run_dev_sweep.sh`, `run_dev_sweep2.sh` | sweep 1; sweeps 2 and 3 and the spread test (`KS` env var) | ran |
| `sweep_summary.py` | terminated share, paired effort change, similarity, WER per layer and k; capped clips reported separately for `_b` roots | ran; the capped part tested on synthetic data only |
| `qwen_gen.py --bound-factor F --fade-ms N` | length-bounded generation (cap from the word count) with a fade-out at the cap | ran (dev2_L21_b, 120 clips, caps printed as `[capped at N tokens]`) |
| `run_dev_sweep2.sh` env vars `EXTRA`, `TAG_SUFFIX` | extra generator flags; write to a separate root | syntax-checked only |
| `make_ladder.py` | one WAV of a line at increasing k; `--equal-level` scales each rung to the unsteered level | tested on synthetic audio only |
| `worst_clips.py` | clipping, harshness, similarity; worst clips per k | tested on synthetic audio only |
| `trim_to_text.py` | crop clips after the last target word (Whisper word timestamps, alignment to the text, 100 ms fade) into `<tag>_trim` | ran on dev2_L21 (120 clips, results above) |
| `showcase/` (5 files + README) | curated Streamlit showcase of Chatterbox and Qwen outputs (see above) | helpers, builder and app smoke-tested on synthetic data; the generator and the real build not run |
| `emotion_dim.py` | arousal, dominance, valence per clip and k (dimensional model), paired change vs unsteered, real EARS emotional anchors | ran (dev2_L21_trim2, dev2_L21; results above) |
| `orthogonalize_direction.py` | project a valence direction (ecstasy minus anger) out of the effort direction, per layer | projection math tested on random vectors; not run on real directions |
| `emotion_drift.py` | per-k emotion class probabilities and paired change vs unsteered (4-class IEMOCAP classifier) | ran on dev2_L21_trim2 (D01 to D03): saturates at hap, not discriminative |
| `level_stage.py` | set the final level independently of steering (unsteered level + chosen gain, peak limit) into `<tag>_lvl<G>` | tested on synthetic audio only |
| `crop_runaways.py` | crop runaway clips to the sentence length into `<tag><suffix>` roots (`--suffix`, `--factor`, `--pad`) | ran (dev_L21 with the default window; results above) |
| `level_by_k.py` | paired change in speech level (dB) per k; peaks and over-range clips | ran (level part); peak columns tested on synthetic audio only |
| `qwen_extract.py --loud-emotion anger` / `--match-level DBFS` | anger-based direction; level-matched reference prompts | job planning tested on synthetic data; not run |
| `compare_runs.py` (repo root, from v1.0) | side-by-side comparison of scored runs | ran on Chatterbox runs |

## Change log
- 2026-10-30: v1.0 closed and tagged; v2.0 plan written; Qwen env installed; recon run; `qwen_gen.py` and `qwen_extract.py` written;
  extractor bug fixed (0 targets); baseline subset generated; directions extracted from 12 training speakers.
- 2026-10-31: sweep 1 generated and scored (k = 1 overshoots, k >= 2 runs long); listening notes (clean range k <= 1, distortion at
  k = 2 and 4); `sweep_summary.py`, `worst_clips.py`, `make_ladder.py`, `run_dev_sweep2.sh` written; sweep 2 (k <= 0.5) generated, all
  clips ended normally; this plan restructured and the README brought up to date.
- 2026-10-31 (later): spread test generated (termination tracks total push; spreading did not help); `sweep_summary.py` threshold bug fixed;
  `crop_runaways.py` written to test whether the maniacal output is the runaway tail; plan and README updated.
- 2026-10-31 (later still): sweeps 2 and 3 (layer 21, k 0.05 to 1.5), the spread test and the runaway-crop test scored with Step A and B (tables above);
  found a smooth dose-response with identity retained and WER near 0 up to k = 1, no benefit from spreading over layers, tail-only hypothesis rejected;
  fixed the similarity column in two scripts; added `level_by_k.py` and the anger option for the extractor; pass bar checked on dev (below).
- 2026-10-31 (level check): steering raises speech level by +8 dB at k = 0.35, +14 at 0.5, +25 at k = 1 (near full scale); decided to treat level as a separate
  knob; added peak statistics, an equal-level ladder and level-matched extraction; the "slightly louder" listening note now needs a check of how it was played.
- 2026-10-31 (listening): equal-level ladder heard by one rater (unsteered calm, subtle steady rise to k 0.5, louder from 0.75, clearly loud from 1.0 to 1.5, distortion at 2.0; first note
  was misread as "no change below 0.5" and corrected the same day); the estimator's over-call at 0.5 and plateau above 1.0 recorded; bounded-generation test at k 1.25 and 1.5 added to the next steps.
- 2026-10-31 (listening, second note): clear shouting at k = 1.0 to 1.5 at equal level, distortion at the end of the clip above that; k = 1.75 was never generated (to be reconciled);
  added bounded generation with a fade-out, a tighter crop option and sweep env vars to test whether the ending, not the voice, limits the clean range.
- 2026-10-31 (reconciled): `dev2_L21` already contains k = 1.75 and 2.0 (288 clips); the 1.75 rung heard by the rater exists. A re-run of the bounded sweep with the OLD scripts did
  nothing (0 pending, wrong root), so the repo copies of `qwen_gen.py`, `run_dev_sweep2.sh` and `crop_runaways.py` still need replacing with the new versions.
- 2026-10-31 (k = 1.75 and 2.0 scored): terminated-only similarity 0.906, 0.905, 0.883, 0.837, 0.819 at k = 1.0, 1.25, 1.5, 1.75, 2.0; terminated share 24, 18, 13, 13, 11 of 24;
  bounded generation is now the priority.
- 2026-10-31 (bounded sweep generated): 120 clips into `dev2_L21_b`; capped 0, 8, 14, 15, 13 of 24 at k = 1.0 to 2.0; `sweep_summary.py` extended to report capped clips; Step A and B, endings
  listening and a check for cut-off last words are next.
- 2026-10-31 (listening on the bounded clips): capped clips lose their last word at k = 1.25 to 1.5; hard cap rejected; `trim_to_text.py` written as the end-of-speech guard (alignment logic tested, not yet run).
- 2026-10-31 (trim results): trimming recovers k = 1.25 (24/24, similarity 0.894, WER median 0) and mostly k = 1.5 (22/24, 0.871); k >= 1.75 stays degraded; `level_stage.py` written;
  listening notes on the trimmed ladder, the peak check and the level choice are next.
- 2026-10-31 (trimmed-ladder listening): last word progressively clipped from k = 1.25 (trim cut early) and the emotion of a fear line turns maniacal at higher k; `trim_to_text.py` changed to follow the voice offset;
  emotion appropriateness added as a risk, a label and a next-step block.
- 2026-10-31 (trim v2 scored): voice-offset trimming gives the same metrics as the fixed-pad version (similarity 0.894 at k = 1.25, 0.873 at 1.5); the last-word check is by ear; `emotion_drift.py` written as a crude automatic proxy for the manic drift.
- 2026-10-31 (emotion follow-up): the manic sound also appears, more weakly, on the other shout lines (first note wrongly assumed other speakers; see the corrected entry below).
- 2026-10-31 (emotion follow-up, corrected): the weaker-manic ladders were the same speaker (p045), so the line, not the speaker, explains the difference; speaker-sensitivity hypothesis downgraded; D03 on other speakers is the next check.
- 2026-10-31 (emotion_drift run): the 4-class classifier labels steered clips happy (not angry) but saturates and does not separate D03 from the other lines; replaced by the dimensional `emotion_dim.py` with real emotional anchors.
- 2026-10-31 (emotion_dim first run): model downloaded (661 MB) but loading failed on the newer transformers (custom class must call post_init(), not init_weights()); fixed, and the head check now compares the class's weights with the checkpoint's keys. Not yet run on clips.
- 2026-11-03 (date correction): this plan was stamped 2026-10-31 for everything after the first sweeps, but the work since then (bounded-generation sweep, trim v1 and v2, level and emotion tools, the listening notes) ran over 2026-10-31 to 2026-11-03 and the exact day of each step was not recorded; earlier entries keep their original stamps. The original one-month budget (2026-10-02 to 2026-11-02) has now elapsed, so the remaining work should be treated as a time-boxed wrap-up: finish the emotion check, the baseline for all validation speakers, the final test-speaker evaluation, a second rater and the write-up.
- 2026-11-03 (emotion_dim results): the dimensional model shows arousal rising smoothly with k (k = 0.5 matches real loud; k = 1.0 matches real anger and ecstasy arousal) and valence drifting UP from k about 1.0 (+0.13) and clearly at 1.25 (+0.27), where real anger, fear and
  distress go down: the manic impression is high arousal with high valence. Acting window k about 0.75 to 1.0. Next: an anger-based direction (low valence), then, if needed, a valence-orthogonalized one.
- 2026-11-03 (emotion_dim second run): the head check stopped on two names (`...pos_conv_embed.conv.parametrizations.weight.original0/1`), which are the newer torch names of the checkpoint's `weight_g` / `weight_v`; all classifier weights were found. The check now maps those names and also compares every tensor's values with the checkpoint.
- 2026-11-03 (final deliverable): the project will end in a Streamlit showcase of both systems' best variations with visible curation, metrics and negative results; `showcase/` written and smoke-tested on synthetic data; hosting must be decided (private by default); test-speaker evaluation, second rater and extra directions become optional.
- 2026-11-03: the anger direction with trimming is the preferred Qwen variant for the showcase (default in the app): no manic sound, no distortion, higher k reads as controlled urgency; operating range k about 0.5 to 1.25 (arousal peaks near 1.0); loud-reading direction kept as the second variant.
- 2026-11-03 (anger vs loud, scored): at matched arousal the anger direction halves the positive valence drift (+0.03 to +0.07 vs +0.13 at arousal +0.4; +0.13 to +0.15 vs +0.27 at k = 1.25), identity is similar, but garbling starts at k = 1.25 (WER mean 1.57); showcase builder and app now carry both Qwen directions.
- 2026-11-03 (anger direction extracted): norms like the loud direction, speaker agreement lower (cosine 0.66 to 0.74 vs 0.80 to 0.91; one anger clip per speaker); sweep at layer 21 and a like-for-like valence-at-matched-arousal comparison are next.
- 2026-11-03 (listening, anger direction): best result so far (no manic state, no distortion, higher k = more rational urgency); made the default Qwen variant in the showcase app.
- 2026-11-03 (showcase build started): added `showcase/preflight.py` (checks every source root before building; tested on synthetic data) and a 'Journey' tab (the whole experiment history in one table, numbers copied from this plan); run order written at the top of `showcase/README.md`; the real build, the Chatterbox dev generation and a run in real Streamlit are still to do.
- 2026-11-03 (Chatterbox dev clips scored, preflight READY): on the 3-speaker dev subset stock Chatterbox keeps similarity 0.932 to 0.891 over exaggeration 0.5 to 1.6 (full baseline: 0.936 to 0.838); effort 7.8 at the top, reached on p023 and p045 only; the app's Journey row now quotes both numbers. Build is next.
- 2026-11-04 (showcase built on real clips): preflight READY; build kept 434 of 534 clips (65 MB); pass rates written to the plan; next: run the app in real Streamlit, inspect the Compare, Ladder and Metrics tabs, then the hosting decision.
- 2026-11-04 (removal reasons read): WER is the main reason for removal; added adjustable thresholds and exported examples of removed clips to the showcase (builder and app); tested on synthetic data, rebuild on real clips next.
- 2026-11-04 (borderline removals): of the clips removed for WER above 0.2, those with WER up to 0.34 (two or three word errors on a 9-word line) were: Chatterbox dev 6 of 8, Qwen loud-reading 7 of 11, Qwen anger 6 of 14, LoRA v0 12 of 12, LoRA v1 7 of 7. So relaxing the WER limit would recover at most about 6 to 7 clips per Qwen or Chatterbox group (and only if they pass the other rules); the rest are clearly worse. Proposed: rebuild with `--max-wer 0.25` (two word errors allowed), listen to a few of the newly included clips, keep 0.2 if any sounds bad.
- 2026-11-04 (rebuild with --max-wer 0.25, adopted): 456 of 534 clips shown (was 434), audio 67 MB. Gains: LoRA v0 +10 and v1 +7 (their longer test lines sit just above 0.2), Chatterbox dev +1 (0.8: 21), Qwen loud-reading +3 (0.75: 18, 1.25: 19, 1.5: 16), Qwen anger +1 (0.75: 21), Chatterbox full baseline +14 (1.6: 202 of 450 = 45%). New pass rates: Chatterbox dev 96%, 96%, 88%, 79%, 67% at 0.3 to 1.6; Qwen loud-reading k = 0, 0.35, 0.5, 0.6, 0.75, 1.0, 1.25, 1.5: 92%, 96%, 92%, 88%, 75%, 96%, 79%, 67%; Qwen anger k = 0, 0.25, 0.5, 0.75, 1.0, 1.25: 92%, 92%, 79%, 88%, 79%, 54%. Only 5 of the 22 added clips belong to the two models being compared. The builder default is now 0.25 and it writes `filter.json`, which the app reads, so the on-screen filter text always matches the build.
- 2026-11-04 (publishing plan): decided direction is GitHub plus Streamlit Community Cloud. Two repositories: the research repo (private; `scripts/prepush_check.py --mode main` blocks audio, model files, data folders, transcripts and secrets) and a minimal showcase repo made by `showcase/export_repo.py` (app, curated audio, pinned requirements, NOTICE.md; checked with `--mode showcase`). Visibility of the showcase repo and app is still to be decided; default private with an `APP_PASSWORD` gate, noting that the gate protects the app, not the audio files in a public repository. Tools tested on synthetic data and a throwaway git repository; not yet run on the real repository. Removed the deprecated `use_container_width` argument from the app (newer Streamlit may reject it) and added the EARS reference to the About tab.
- 2026-11-04 (publishing simplified by the user): one repository; EARS data and transcripts stay git-ignored, everything else (including the showcase audio) is pushed and the app is deployed from `showcase/app.py`. `scripts/prepush_check.py` gained mode `single` (default) and `--allow`; `export_repo.py` stays as an optional route. Visibility not stated: the repository stays private unless the user says otherwise (the audio is as visible as the repository).