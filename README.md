# Effort-tts

Speaker-preserving, continuous **vocal-effort control** for zero-shot TTS, plus an
**effort-realism evaluation harness**. The goal is high-effort speech (loud, strained,
projected) in a cloned speaker's own voice, controlled by a reproducible scalar instead of
a free-text tag, and a way to measure whether synthetic effort is realistic and whether
speaker identity survives.

Solo research project. Research and portfolio use only (see *Data and licensing*).

## Versions

**v1.0: Chatterbox (2026-10-02 to 2026-10-30). Closed.** Everything built and measured with Chatterbox as the TTS model:
the EARS pipeline and level-free effort estimator, the 30-line evaluation set, the Chatterbox baseline (effort control strong
and monotone, Spearman 0.75, but about 9x a real speaker's identity drop at loud effort, pacing entangled with the knob,
54% speech-problem rate at the top, text-independent), the human-label study (reliability kappa 0.53), the generation,
scoring and comparison harness, and a negative result: effort-conditioned LoRA fine-tunes (v0, v1) and an extrapolation
control did not reproduce EARS loud speech (best Spearman 0.33, median effort at the top scalar about -3 against +7 for
stock), plus a token probe (mild loud reading is partly visible in S3 tokens, AUC 0.75). Chatterbox full or partial
fine-tuning is not a guaranteed route and is shelved (`finetune_patch/v1_1_shelved/`, never run). Tag: `v1.0-chatterbox`.

**v2.0: Qwen3-TTS activation steering (started 2026-10-30; in progress, last updated 2026-11-03).** Training-free steering of
Qwen3-TTS (reported Apache-2.0, 0.6B and 1.7B Base, zero-shot cloning) along an effort direction derived from the EARS matched
loud and regular readings; the steering strength k is the continuous, reproducible control. Not tried before (checked against both
handoffs and the earlier project chats). Closest prior art: Akti and Waibel, arXiv 2610.07647 (vocal effort steering on Qwen3-TTS, framed
as Lombard speech in noise); also arXiv 2609.33810 (speaking-rate steering, additive vs clamp). The evaluation reuses the v1.0
harness. Full plan, ground rules (dev vs final), commands, results, decisions and change log: `qwen_v2/PLAN.md`.
Status in one paragraph (dev results, one seed, estimator-based): a shared effort direction exists across 12 training speakers; on layer 21 the
steering strength k gives a smooth, monotone, text-independent effort control from k about 0.2 to 1.0 (estimator effort -3.5 unsteered, 8 at k = 0.5
about real loud, 15 at k = 0.6 to 0.75 about anger, 20 at k = 1 about yelling) with speaker similarity 0.944 at k = 0.5 and 0.906 at k = 1 (unsteered
0.949; stock Chatterbox lost 0.10 at effort 7), WER near 0 and all clips ending normally; beyond k = 1 termination, identity and intelligibility degrade
(k = 2 and 4 are maniacal or distorted, and not only in the runaway tail). Spreading over several layers did not help. Scored through k = 2.0: among clips that end on their own, similarity is 0.906 at k = 1, 0.905 at 1.25, 0.883 at 1.5,
0.837 at 1.75 and 0.819 at 2.0 (WER 0 up to 1.75), but the share that end on their own falls from 24 of 24 to 18, 13, 13 and 11 of 24, so at k = 1.25 to 1.5 the
termination failure, not identity, is the limit. A hard length cap (1.5x the expected duration) fixed the stopping but cut the last word of shouted clips, so the
end-of-speech guard is now an offline trim to the end of the last target word using ASR word timestamps (`qwen_v2/trim_to_text.py`). After trimming, all 24 dev clips at k = 1.25 are usable (similarity 0.894, WER median 0) and 22 of 24 at k = 1.5 (0.871); k = 1.75 and above stays degraded.
`qwen_v2/level_stage.py` sets the final loudness independently of the steering.
(The voice-offset re-trim scores the same as the fixed-pad one, so the last-word check is by ear; a dimensional arousal / dominance / valence model scored against real EARS emotional clips (`qwen_v2/emotion_dim.py`) shows arousal rising smoothly with k, k = 0.5 matching real loud speech and k = 1.0 matching the arousal of real anger and ecstasy, but valence drifting UP from k about 1.0 while real anger, fear and distress go down, i.e. the manic impression is high arousal with high valence; the acting window is about k = 0.75 to 1.0 an anger-based direction, tested next, roughly halves that valence drift at matched arousal (+0.03 to +0.07 against +0.13) with similar identity but garbles clips from k = 1.25, and one listener finds it the best result so far (no manic state, no distortion, higher k sounds like controlled, rational urgency), so it is the default variant in the showcase, with the loud-reading direction as the second; a first 4-class classifier (`emotion_drift.py`) only labelled steered clips "happy" and saturated)
In listening, the trim cut the last word slightly early from k = 1.25 (now fixed to cut where the voice stops), and for a fear line ("Fire in the kitchen, everybody out!") the voice turned
maniacal at higher k, (weaker on the other shout lines of the same speaker, so it depends on the line: steering seems to amplify the arousal the text implies, and an alarm line turns manic at higher k), so the effort direction is not emotion-neutral and the acting-appropriate window is about k <= 1.0 to 1.25; "emotion fits the line" and "maniacal" become evaluation labels. Listening (one rater, equal-level ladder): the unsteered clip is absolutely calm, the voice rises subtly and steadily through k = 0.5, is louder at 0.75, and is clearly
shouting at k = 1.0, 1.25 and 1.5, with distortion creeping in at the very end of the clip above that (and more at 2.0); an earlier raw-playback note called k = 1
"slightly louder, not shouting" (different line and playback, not reconciled). The estimator overstates k = 0.5 and plateaus at k >= 1, so blind forcefulness
ratings, an anger-based direction, bounded generation (to fix the ending) and the final evaluation on the 10 untouched test speakers with the 30 test lines come next. A level check showed that steering also
raises speech level by about +8 dB at k = 0.35, +14 at 0.5 and +25 at k = 1 (near full scale, likely clipping), so level and voice quality are to be
separated: level-matched direction extraction, an equal-level listening ladder, and a controlled loudness stage in post.

**Final deliverable (decided 2026-11-03): a Streamlit showcase** of the best variations of both systems (`showcase/`), with one visible curation filter, metrics and the negative results. Hosting is to be decided: the voices are clones of EARS participants (CC BY-NC 4.0), so the default is private or local (see `showcase/README.md`). The final test-speaker evaluation, a second rater and further directions are now optional.

## Target system and scope

Intended end state: given a line of text and a cloned voice, the system decides how much vocal
effort the line calls for (a level from regular speech up to loud or angry delivery) and
renders the line in that speaker's own voice at that effort.

Two parts:
1. **Effort-controlled TTS:** a model fine-tuned on an effort scalar that preserves the cloned
   identity. This is the main research contribution; the baseline gap it must close is
   measured under *Baseline results*.
2. **Director:** maps a line (optionally with context) to a target effort level. Planned as a
   prompted language model with a rubric, calibrated to the estimator scale and checked
   against human per-line ratings, so it needs little or no training. Even before the
   fine-tune, "director + Chatterbox exaggeration knob" is the baseline system that the
   fine-tune must beat.

Scope boundaries: effort only (no emotion classification and no separate rendering of anger
versus joy); upper limit loud or angry delivery (what EARS speech supports); text in, audio
out (converting an existing recording to higher effort is a stretch goal, input type to be
confirmed); one effort level per line first, word-level contours later.

## Why this matters (impact if it succeeds)

These are the intended outcomes, not results. Nothing below is demonstrated yet.

**The problem.** Current TTS systems offer high-effort delivery mostly as free-text tags
(for example "shouts"). The effect varies by voice, is not reproducible, and tends to break
down when the voice and the requested delivery conflict. The Week 1 baseline already shows a
measurable version of this: in Chatterbox, raising the exaggeration knob to reach real-loud
effort costs about nine times the speaker-similarity drop that a real speaker shows when
speaking loudly (see *Baseline results*), and the knob ignores what the text calls for.

**What success would give people**
- **Dubbing, ADR and game audio:** loud or strained lines in a cast member's own voice, at a
  specified intensity, repeatable across takes, instead of re-recording or hand-picking the
  best of many generations.
- **Voice performers' vocal health:** high-effort lines are the ones that strain a voice over
  a long session. Synthetic stand-ins for pickups or placeholders could reduce repeated
  takes, with the performer's consent and control. (Whether this actually helps is untested
  and depends on the performer's workflow.)
- **Speech in noise and assistive speech:** related research frames continuous vocal-effort
  control as a way to improve intelligibility in noise. A speaker-preserving version could
  let a synthetic voice raise its effort without sounding like someone else.
- **Research:** an effort-realism benchmark that measures whether synthetic loud speech has
  plausible spectral and voice-quality cues (not just higher level or pitch) and whether
  identity survives. No benchmark found in my searches does this; re-check before claiming
  novelty publicly.

**Tiers of success.** Even if fine-tuning fails, the project still delivers (1) the effort
estimator, (2) a measured gap report for existing systems, and (3) the evaluation harness. A
training-free best-of-N selector is the fallback. A working effort-conditioned model that
keeps identity is the strongest outcome.

**Risks and what this project does about them.** More convincing high-effort cloned speech
can also make impersonation, fake distress audio, and non-consensual voice use more
believable. The project therefore uses consenting adult data only, excludes minors entirely,
keeps Chatterbox's output watermark intact, discloses synthetic audio in demos, and releases
weights only for research (EARS is non-commercial). Any commercial use would need its own
consented recordings and a fresh licensing and ethics review.

**Limits.** EARS speech reaches roughly loud/angry delivery, not true shouting, so the
supported claim is "high effort up to loud/angry". The effort score is a gain-invariant
acoustic proxy, not a perceptual measure, and has not been validated on synthetic audio.

## Status

**v2.0 (Qwen3-TTS steering) is in progress; see *Versions* and `qwen_v2/PLAN.md`.** v1.0 (Chatterbox) was closed on 2026-10-30;
its status follows. Last updated 2026-11-03. The original one-month budget (2026-10-02 to 2026-11-02) has elapsed; remaining work is a time-boxed wrap-up (see `qwen_v2/PLAN.md`).

**Built (v1.0)**
- EARS download, inventory, QA, speaker-disjoint train/val/test split (seeded, gender-stratified)
- Per-file manifest with level and clipping statistics
- Gain-invariant acoustic features (spectral tilt, alpha ratio, Hammarberg, cepstral peak
  prominence, pitch, voicing) with a versioned cache
- Effort estimator v0 (logistic model, loud vs regular reading) with a feature-set ablation
  and out-of-distribution checks
- Fixed 30-line evaluation script with an EARS-overlap check (0 hits)
- Listening tools: `scripts/inspect_exag16.py` (clip picker) and `scripts/label_clips.py` (blind labeling helper), `scripts/analyze_labels.py` and `scripts/analyze_labels_deep.py` (label vs score analysis)
- Chatterbox baseline generation (resumable; reference-clip picker). Full run complete and
  verified: 750/750 jobs ok (5 validation speakers x 30 lines x 5 exaggeration values,
  cfg_weight 0.5, 1 seed)
- Baseline scoring: Whisper word error rate, WavLM speaker similarity (with a real-speech
  calibration), and level-free effort scoring with a 4-panel report. Run on the real
  baseline (see *Baseline results*)
- Test suite: 38 tests passing

**Not done in v1.0 (now closed or superseded):** multi-seed runs, second-stage label analysis (stratification, detector rules,
similarity headroom), labels for other exaggeration settings, the written gap report, the multi-rater listening test, demo.
The fine-tune go/no-go was completed and was negative (LoRA v0 and v1; see *Versions*).

**Built (v2.0)**
- Qwen3-TTS environment (Python 3.12) and recon of the model structure; reference transcripts for the validation references
- `qwen_v2/qwen_gen.py` (zero-shot and steered generation in the v1.0 layout), `qwen_extract.py` (direction extraction),
  `run_dev_sweep.sh` and `run_dev_sweep2.sh`, `sweep_summary.py`, `make_ladder.py`, `worst_clips.py`, `crop_runaways.py`, `trim_to_text.py`, `level_stage.py`, `dev_lines.json`
- Directions extracted from the 12 training speakers; sweeps 1 to 3 (layer 21, k 0.05 to 1.5), a multi-layer spread test and a runaway-crop test generated and
  scored (dose-response table in `qwen_v2/PLAN.md`); `level_by_k.py` and an anger-direction option for the extractor written

**Working agreement:** README and `qwen_v2/PLAN.md` are updated after every step.

## Baseline results (Chatterbox, Week 1)

Median effort is the level-free estimator score, referenced to the same speakers' real EARS
speech. Similarity is WavLM x-vector cosine to the speaker's centroid of real freeform windows.

| exaggeration | effort (median) | WER mean / median | similarity to centroid |
|---|---|---|---|
| 0.3 | -4.57 | 0.027 / 0.000 | 0.936 |
| 0.5 (default) | -3.80 | 0.017 / 0.000 | 0.936 |
| 0.8 | -1.10 | 0.045 / 0.000 | 0.926 |
| 1.2 | 3.59 | 0.065 / 0.000 | 0.893 |
| 1.6 (3 seeds, n=450) | 7.28 | 0.312* / 0.069 | 0.842 |

\*The 1.6 WER mean is dominated by one clip (WER 74); prefer the median or the fraction of
clips with WER > 0.5 (see findings).

Real EARS reference effort (same speakers, same estimator): regular -5.43, loud 7.15,
anger 12.85, yelling 21.18.
Real similarity to own centroid: regular 0.978, loud 0.967, anger 0.948, ecstasy 0.923
(EARS spells it "extasy").

**Findings** (Chatterbox, 5 validation speakers, 1 seed, 1 reference clip per speaker)

*Effort and identity*
- Exaggeration raises effort monotonically for all 5 speakers, spanning roughly real regular
  to real loud (1.6 is about real loud). It never reaches anger or yelling. A spectral-only
  estimator (no pitch features) shows the same trend, so the rise is not just pitch.
- Identity drops much more than for real speakers. Real regular to loud costs about 0.011
  similarity; Chatterbox 0.5 to 1.6 (effort about equal to real loud) costs about 0.098.
  Synthetic audio also sits roughly 0.04 below real speech at any effort, so compare drops.
  Even among clips judged ok by ear, median similarity at 1.6 is 0.837 (all 1.6 clips: 0.838;
  exaggeration 0.5: 0.936), so the identity cost is not caused by a few bad clips and
  filtering for quality does not recover it.
- The knob's reach is speaker-dependent: at 1.6, median effort ranges from 3.4 (p056) to 10.8
  (p023) over 3 seeds (p057 8.5, p045 7.5, p059 6.0); the ordering is unchanged from seed 0.
- Replication with 3 seeds at 1.6 (n=450; other settings still 1 seed): median effort 7.28
  (seed 0 alone: 7.05), similarity 0.842 (0.838), WER median 0.069 (0.055). The seed-0
  findings were not a fluke. Median effort by intent at 1.6 is calm 8.0, shout 7.0, urgent 7.0,
  so calm lines get at least as much effort as shout lines. The labeled analyses below use
  seed 0 only. The `effort_vs_default` column in the score table is still seed-0 paired.
- Line intent (calm, urgent, shout) barely changes effort at any setting; exaggeration acts
  as a global setting not tied to what the text calls for. At 1.6 the per-line medians range
  from 1.2 (S03) to 11.1 (S05) but do not track intent (calm, shout and urgent line medians
  are all about 7), and within-line spread (5 to 15 points) is as large as the spread between
  lines. S03 ("Everybody out of the building, move, move!") was the lowest line and sounded
  calm by ear in 4 speakers; it may be sampling variability and needs multiple seeds.
- The estimator scores 1.6 as real-loud, yet listening finds about half of the clips
  problematic (below), so effort score and perceived realism are not the same thing.

*Intelligibility and duplication (automatic)*
- WER at 1.6: the seed-0 mean (0.632; 0.312 over all 3 seeds) is dominated by a single clip
  (WER 74, a Whisper repetition loop on a 3-word line); without it the means are about 0.14
  and 0.15. On seed 0, 5.3% of clips (8/150) have
  WER > 0.5 and 1.3% have WER > 1.0.
- Duplication flag (words heard / words in the line >= 1.5): flagged clips per setting, out of
  150, are 2, 1, 4, 2 and 8 for exaggeration 0.3, 0.5, 0.8, 1.2 and 1.6. At 1.6 all 8 sit on
  three lines: C05 "Take your time." (4 of 5 speakers), C02 (2) and S01 "Watch out!" (2).
  Small counts: the rise at 1.6 is suggestive, not established.

*Blind listening labels at 1.6 (150 clips, one listener, labels can overlap)*
- ok 77 (51%); unintelligible 43 (29%); loud_when_calm 33 (22%); duplicated 13 (9%);
  calm_when_loud 10 (7%; 9 on shout or urgent lines, 1 on a calm line). My pre-labeling
  recollection (about 10% unintelligible, the other two problems about equal) was off.
  Overlaps: 9 of 13 duplicated clips are also unintelligible; 12 of 33 loud_when_calm clips are
  also unintelligible.
- By intent: only 6 of 50 calm-line clips are ok (33 loud_when_calm, 22 unintelligible, 6
  duplicated), versus 35 of 50 for shout lines and 36 of 50 for urgent lines. Seven calm lines
  (C01, C02, C03, C06, C07, C09, C10) had a problem in all 5 speakers; among shout and urgent
  lines only S08 had 3. Whether loud_when_calm is a failure depends on the line: some calm
  sentences can plausibly be said loudly (calling across a room), others cannot. A per-line
  plausibility rating (1 absurd, 2 odd, 3 plausible; `configs/line_plausibility.csv`) did not
  explain it: loud_when_calm is 53% on the 3 lines rated absurd, 73% on the 3 rated odd and
  70% on the 4 rated plausible (15, 15 and 20 clips), so the label as recorded mostly tracks
  the displayed intent, not plausibility (groups too small to resolve differences).
  Unintelligibility on calm lines is a failure either way, and it is higher on the 3 lines
  rated absurd to say loudly (67%, 10 of 15 clips) than on the other 7 (34%, 12 of 35);
  but this is carried by two lines: C09 (unintelligible for 5 of 5 speakers) and
  C02 (4 of 5); the third absurd line, C08, is 1 of 5. Length (C10, 23 words: 1 of 5) and
  apparent phonetic difficulty (C04 and C07: 1 and 2 of 5) do not explain it, and no shout or
  urgent line exceeds 3 of 5. C09 failing for every speaker, including those that rarely fail,
  makes it a line effect, not a speaker effect. Whether a text-effort mismatch is the cause is
  not established (5 clips per line, one seed). Calm lines are mostly long (8 of
  10 have 10+ words), but length does not explain the gap: among lines of 10+ words, shout and
  urgent clips are 76% ok and 18% unintelligible, versus 5% ok and 50% unintelligible for calm
  clips. Phonetic difficulty may still confound (several calm lines look demanding, e.g. C04,
  C06, C07).
- By speaker (ok share): p023 0.60, p045 0.60, p057 0.60, p059 0.47, p056 0.30. p056 has 18 of
  the 43 unintelligible clips and 5 of the 10 calm_when_loud clips (its knob also reaches
  least far); p059 has 12 unintelligible. loud_when_calm is spread evenly (8, 6, 4, 8, 7).
- Usable yield: 51% of clips are ok, and 25% (37/150) are ok with effort at or above real
  loud (35 on shout or urgent lines, 2 on calm lines). Per speaker: p023 13 of 30, p057 10,
  p045 9, p056 3, p059 2, so best-of-N at 1.6 is weak for p056 and p059. Among those 37 clips,
  similarity is min 0.652, median 0.835, 90th percentile 0.927, max 0.952; only 2 reach the
  exaggeration-0.5 median (0.936) and 1 reaches real anger (0.948). Selecting among single
  draws cannot recover real-speaker identity retention (real loud 0.967).

*Automatic metrics against the labels* (AUC; 0.5 = no signal; first pass, pooled across
intents, small positive counts)
- calm_when_loud (shout and urgent lines, 9 positives of 100): the estimator detects it.
  Effort AUC 0.83 and spectral 0.80 (low effort goes with the label; median effort 4.4 vs
  7.1 for ok clips). Confirms the earlier listening to the lowest-scoring shout lines.
- loud_when_calm (calm lines, 33 of 50): weak (effort 0.69, similarity 0.64, WER 0.61).
- duplicated (13): duplication ratio 0.83, WER 0.76, audio duration 0.75, words per second 0.79
  (low rate goes with the label). The >= 1.5 rule flags 8 clips, 3 correct (precision 38%,
  recall 23%). Loosening raises recall (0.69 at >= 1.2, 0.77 at >= 1.15) but precision stays 24%
  to 38%, because garbled clips also gain words: the ratio is a general speech-problem flag,
  not a clean duplication detector. Within intent groups it holds up (AUC 0.90 on calm lines,
  0.77 on shout and urgent lines, with only 6 to 7 positives each).
- unintelligible (43): words per second 0.80 (slower), duplication ratio 0.78, audio duration
  0.77, WER 0.76. WER > 0.5 flags 8 clips, 5 correct (precision 63%, recall 12%): Whisper
  reads most of what the listener found unintelligible. Unintelligible clips are slower
  (median 2.0 vs 2.8 words/s) and longer (5.4 vs 3.1 s). Within intent groups the signals hold
  (WER 0.70 to 0.77; words per second 0.74 to 0.83 with the direction flipped), so they are
  not just a calm-vs-shout artifact. WER precision is only 51% to 65% at any cutoff from 0.1
  to 0.5 (base rate 29%), and words per second is biased against short lines (C05 runs at 0.8
  words/s).
- Combined filter (WER > 0.1 or duplication ratio >= 1.15 or words per second < 2.3): flags 81
  of 150 clips and catches 43 of the 47 duplicated-or-unintelligible clips. The 69 clips it
  keeps are 94% free of those problems (65/69) versus 69% unfiltered, at the cost of
  discarding 38 clips that were fine. In-sample and optimistic until validated on fresh seeds (on seeds 1 and 2 it flags 54% of
  clips again, but it over-flags some lines; see the seed analysis).

*Seed stability at 1.6 (3 seeds, n=450, `analyze_seeds.py`)*
- Per-seed medians agree (effort 7.05, 7.33, 7.71; similarity 0.838 to 0.850; combined-filter
  flag rate 0.54 in every seed).
- Same speaker, same line, different seed: effort SD 3.25 against a total SD of 4.33, so seed
  noise is 56% of the effort variance. For scale, regular-to-loud spans about 12.6 points
  (-5.4 to 7.15) and the step from exaggeration 0.8 to 1.2 moves the median by 4.7. The
  scalar is not reproducible at 1.6 (seed noise at lower settings is not yet measured).
- Real line and speaker effects exist: line explains 24% of effort variance (6% expected by
  chance), speaker 27% (1% by chance), intent 1%. S03 is the lowest line by a wide margin
  (mean 0.7 over 15 clips, SE 1.1; next lowest 4.7), so its low effort is persistent, not
  noise. Calm lines average the highest effort (7.8 vs 7.0 shout, 7.1 urgent).
- Filter persistence: P(flagged in another seed | flagged) = 0.70 vs a 0.54 base rate (1.3x).
  C09 and C02 are flagged for 4, 4, 4 and 4, 4, 3 of 5 speakers per seed, C10 for 1, 2, 0, which
  supports real line-specific failures. But the filter flags 93% to 100% of clips on U01, C07,
  S01, C05 and S04, where seed-0 human labels found unintelligible speech in only 1 to 2 of 5,
  so it over-flags some lines (hard-to-transcribe lines for Whisper; the speaking-rate rule
  penalizes short lines).
- Duplication ratio >= 1.5 over all seeds: C05 10 of 15, S01 9 of 15, C02 2, S08 1. These are
  the two shortest lines, but seed-0 labels confirmed duplication for only 1 of 4 flagged C05
  clips and 1 of 2 flagged S01 clips, so the ratio is unreliable on 2 to 3 word lines.
- Automatic best-of-N (passes the filter and effort >= 7.15; a proxy for human ok): cells
  covered 23%, 34%, 43% with 1, 2, 3 draws; median similarity of the best passing draw 0.842,
  0.867, 0.868; only 8% exceed 0.936 and 3% exceed 0.948. Coverage with all draws by speaker:
  p023 83%, p057 53%, p045 47%, p059 20%, p056 10%. Coverage is pessimistic because the
  filter over-flags; the similarity gain (about +0.03 from extra draws) is the reliable part.

*Labels for the new seeds (seeds 1 and 2; 100 clips from 10 lines, 10 clips per line)*
- Totals: loud_when_calm 42, unintelligible 37, ok 32, duplicated 11, calm_when_loud 7. Three
  clips carry impossible label combinations (probable key slips): U01 twice and C09 once.
- Clips with a speech problem (duplicated or unintelligible), out of 10 per line: C05 10, C02 9,
  C09 7, C04 7, S01 5, C10 2, C07 1, U01 1, S03 0, S05 0. The two shortest lines are the
  weakest ("Take your time.", 3 words: 10 of 10; "Watch out!", 2 words: 5 of 10).
- By intent: the six calm lines are ok in only 4 of 60 clips (7%); the three shout lines in 21
  of 30; U01 (urgent) in 7 of 10.
- Effort estimator against ears (small n): S05, the highest-effort line, is ok in 10 of 10 with
  no calm-sounding clips; S03, the lowest-effort line, sounds calm in 3 of 10 (ok in 7 of 10);
  U01 sounds calm in 2 of 10.
- Per-line labels swing between batches (for example C05 had 1 of 5 unintelligible on seed 0
  but 9 of 10 on seeds 1 and 2; C04 1 of 5 vs 7 of 10), so either seed variance is large or
  labeling drifted. A test-retest relabeling of seed-0 clips is needed to tell which.
- Automatic filter on the fresh clips (speech-problem base rate 0.42 on these chosen lines):
  it flags 71 of 100 clips, 36 correct, 6 missed (precision 0.51, recall 0.86), close to its
  in-sample 0.53 / 0.91, so it generalizes but is weak (lift about 1.2 on these lines). By rule:
  duplication ratio >= 1.15 is the most precise (0.76 precision, 0.67 recall; seed 0: 0.71 /
  0.47), WER > 0.1 gives 0.54 / 0.76, words/s < 2.3 gives 0.55 / 0.57.
- False positives on the fresh clips come from specific lines and rules: U01 9 (all from
  WER > 0.1, speech sounded fine; cause not yet checked, possibly ASR or text normalization),
  C07 9 (speaking rate 8, WER 6, ratio 3), S01 5, S03 4, C09 3.
- Effort estimator vs "sounds calm" on shout and urgent lines: AUC 0.86 on the new seeds
  (6 of 40 sounds-calm; median effort -3.3 vs 6.6) and 0.83 on all labeled clips (15 of 140;
  median -0.1 vs 7.1). The estimator tracks that failure on fresh clips.
- Test-retest by line (my tally from the relabeling log, 5 seed-0 clips per line; clips with a
  speech problem, original to retest): C09 5 to 5, C07 2 to 2, S01 2 to 2, C04 1 to 2, C05 1 to 4.
  Four lines are stable; C05 changed because duplication on "Take your time." is flagged more
  readily now (a learning effect), so seed-0 labels probably under-count duplication on short
  lines. Clip-level (25 clips): speech-problem agreement 0.76, kappa 0.53 (moderate), identical
  label set only 0.52, and the problem rate rose from 0.44 to 0.60 (5 clips moved from ok to
  problem, 1 the other way). That is a shift toward stricter labeling, not just random noise,
  though 6 discordant clips cannot show it is significant.

*Implication for the plan*
- Selection can clean up speech problems but cannot close the identity gap at 1.6 (ok clips
  0.835 vs 0.936 at the default setting; best of 3 draws only reaches a median of 0.868), which supports attempting the fine-tune, subject to
  the still-unverified tooling and licensing checks.

## Fine-tune feasibility (research, 2026-10-28)

Sources were web pages and model cards; nothing here was run. Items marked [UNVERIFIED] need a
check before relying on them.

**Base model: standard Chatterbox (0.5B, Llama-based T3) is the best fit.**
- MIT license. Community LoRA and full fine-tuning toolkits exist
  (`gokhaneraslan/chatterbox-finetuning`: LoRA r=64 on Turbo config, offline preprocessing,
  training at 16 kHz, LoRA advised for datasets of 10 hours or less, about 60% less VRAM).
  Many community adapters were trained with it (Uzbek on 111 h, Indic, Swahili, Bangla).
  The toolkit is lightly maintained (last commit Feb 2026 as of a May 2026 snapshot, about 99
  stars, 3 contributors) [re-check].
- The exaggeration knob is a scalar input (`emotion_adv` in `T3Cond`) that already exists in
  the model. Verified in the cloned toolkit (2026-10-28): `src/model.py:120` sets it to 0.5 for
  every training example; `cond_enc.py` builds `emotion_adv_fc = nn.Linear(1, n_channels,
  bias=False)` and concatenates its output into the conditioning; `tts_turbo.py` sets
  `hp.emotion_adv = False` (Turbo has no slot, so train with `is_turbo = False`). So the
  effort-conditioned fine-tune can reuse the slot: feed each clip's effort label through
  `emotion_adv` instead of 0.5. No architecture change is needed.
- Chatterbox-Turbo (350M) ignores exaggeration, cfg_weight and min_p, so it is not a base for
  this. Chatterbox-Multilingual (0.5B) has the same slot as the standard model.
- Design risk, confirmed in the toolkit's `preprocess_file_based.py` and `dataset.py`:
  `speaker_emb` is computed from the whole target clip, `prompt_tokens` are the first 3 s of
  the target clip, the forward pass masks the loss on those positions, and 20% of the time
  both are zeroed (conditioning dropout). So effort would leak through the prompt and the
  scalar would be ignored. The pilot patch takes both from a different same-speaker clip.
- Tokenizer check: `setup.py` downloads the English T3 weights (`t3_cfg.safetensors`) but saves
  the multilingual grapheme tokenizer as `tokenizer.json`, and the config sets
  `new_vocab_size = 2454` (resize with mean-initialised new rows). Whether English token ids
  line up with the original 704-token vocabulary is [UNVERIFIED]. The pilot uses the original
  English `tokenizer.json` with `new_vocab_size = 704`, so there is no resize and the merged T3
  loads in stock `chatterbox-tts` for evaluation. Training pairs must
  use a calm reference clip of the same speaker with a loud or angry target clip, which is
  also the real use case.
- Scalar scale: the base model reacts to 0.5 to 1.6 (baseline: 0.5 is regular, 1.6 is about real
  loud). Pilot mapping proposal: label = 0.5 + 1.1 * (score + 5.43) / 12.58, capped at 2.0
  (real regular 0.5, real loud 1.6, real anger about 2.0).
- Check which base the toolkit's standard mode loads (the README says a 2454-token multilingual
  vocabulary; our baseline used the English 704-token model). If different, regenerate a small
  baseline on that base for a fair comparison.
- Also decide whether to make `emotion_adv_fc` (the scalar's projection) trainable; the toolkit's
  default LoRA targets are attention and MLP layers plus the speaker encoder.

**Alternatives**
- Qwen3-TTS (Apache-2.0; 0.6B and 1.7B Base for cloning from about 3 s; positioned for
  fine-tuning; community LoRA/full pipeline needs about a 24 GB GPU for the 1.7B model). No
  scalar effort slot, so one would have to be added. Second choice.
- Zonos v0.1 (Apache-2.0; 1.6B transformer and hybrid; cloning from 5 to 30 s; built-in
  conditioning on speaking rate, pitch variation and emotions including anger). Closest to
  ready-made continuous control, but training code is [UNVERIFIED], the model is larger, and
  it depends on espeak.
- Orpheus 3B: code is Apache-2.0 but the weights descend from Llama 3.2 so the weight license is
  [UNVERIFIED]; built for named voices rather than a scalar slot; larger. Lower priority.
- F5-TTS and Fish weights are non-commercial (from the handoff); not checked: StyleTTS2,
  VoxCPM2, CosyVoice, IndexTTS2.

**Prior-art update (the field moved since the handoff).** Continuous vocal-effort control for
TTS now has several papers, all framed around intelligibility in noise (Lombard effect):
arXiv 2601.12966 (style-embedding directions), 2606.23176 (flow-matching model with continuous
effort and articulation control, word-level emphasis, Interspeech 2026), 2610.07647 (training-free
activation steering; speaker similarity 89 to 95% by their metric; no verified repo) and
2610.20181 (Flow-GRPO post-training). Activation steering of autoregressive TTS has also been
shown for speaking rate (2609.33810) and emotion. Do not claim novelty for "scalar effort
control" per se; the angle left is acting-level high effort (loud or angry) in a cloned voice,
the effort-realism plus identity plus intelligibility evaluation with human labels, and the
director. Re-run the search before any public release.

**Compute.** Google AI plans now bundle Colab Pro benefits (priority for T4, L4 and A100); the
compute-unit system is unchanged. Measured rates (March 2026): L4 about 1.7 units/h, A100-40GB
about 5.4, A100-80GB about 7.5; other sources report higher A100 rates, so use the in-app meter.
With 100 units that is roughly 58 L4-hours or 18 A100-40GB-hours. A LoRA run on the 0.5B T3
should fit on an L4 (22.5 GB) with gradient checkpointing [estimate].

**Pilot files (`finetune_patch/`, 2026-10-28; syntax-checked, not run, no torch in my sandbox).**
`config.py` (English vocab 704, LoRA r=32, `emotion_adv_fc` trained, 6 s prompts, new
`effort_manifest`, `ref_conditioning`, `cond_dropout`), `dataset.py` (new `EffortDataset` and
`data_collator_effort`), `train.py` (uses them when `effort_manifest` is set), and
`apply_effort_patch.py` (edits `src/model.py`: `effort` argument, `emotion_adv` from it,
no prompt masking). Also `scripts/make_effort_dataset.py` (EARS clips to 16 kHz wavs and txt, effort labels from
`score_nolevel` anchored at regular = 0.5 and loud = 1.6, same-speaker freeform reference pool,
manifest; excludes whisper, highpitch, lowpitch, fast and slow because the estimator scores pitch and
rate manipulations as effort; dry run on the 12 training speakers: 588 clips, 2.12 h, 49 per speaker)
and `finetune/effort_pilot_colab.ipynb` (Drive layout, toolkit checkout, patch, English tokenizer,
smoke test, full run). Neither the notebook nor any training code has been run yet. Toolkit commit used: `fac31c46ec96b37283a363a1a96c2a0e56640e03`.
Colab note: this account offers only a T4 GPU and a v5e-1 TPU. The pilot targets the T4 (15 GB, no bf16):
`train.py` picks fp16 on GPUs below compute capability 8, config uses batch 4 x accumulation 4, and training
resumes from the newest checkpoint. The TPU is not used (CUDA-based toolkit, variable-length batches).
Smoke test on a Colab T4 (2026-10-29): fp16; LoRA r=32, 22.65M trainable of 555M (4.1%) including
`emotion_adv_fc`; preprocessing 684/684 clips in 76 s; EffortDataset 588 targets, labels 0.20 / 0.68 (median) /
2.00; 1 epoch = 36 steps in 260 s (7.2 s/step), loss 6.33 to 5.18 over the epoch, no NaN. Full run (2026-10-29, Colab T4, fp16): 12 epochs = 432 steps in 39.6 min (about 4 s/step once warmed up;
the smoke test's 7.2 s/step included startup). Loss 6.30 to about 3.40 (mean 3.73), flat around 3.4 to 3.5
after epoch 6. Grad-norm spikes (59 near step 100, 54 near epoch 7, 40 near epoch 11.7); one visible loss
bump (3.66 to 4.21 around epoch 2.5 to 3) that recovered by epoch 3.5; gradient clipping bounds the updates.
No validation loss exists, so the loss says nothing about whether the scalar is used. Adapter saved to Drive
(`effort-pilot/output/new_lang_adapter`, about 90 MB). Next: listen, then evaluate on the 5 validation
speakers (effort vs scalar, similarity, speech-problem rate) against the baseline.
First A/B on one line ("Everybody out of the building, move, move!", reference ref_p001_00, seed 0, T4):
`emotion_adv_fc` weights changed by only 0.65% (relative norm) during training, so the scalar's own projection
barely moved (Adam at 1e-4 for 432 steps); any effect comes through the LoRA layers. Clip duration vs scalar
0.5 / 1.0 / 1.6 / 2.0: stock 2.72 / 2.12 / 1.96 / 1.96 s (strong speed-up with exaggeration), fine-tuned
2.28 / 2.24 / 2.24 / 1.96 s (less coupling); cfg_weight 0.3 gives slower pacing (2.40 and 2.48 s at 1.6 and 2.0).
Generation speed on the T4 is about 28 tokens/s (roughly 1 s per second of audio). One listener's first
impression: intelligible, slight change with the scalar, the 1.0 clip closest to the right urgency. Not yet
measured on the validation speakers.
Evaluation generation (to run on Colab): `finetune/gen_eval.py` + `finetune/eval_lines.json` (the 30 lines,
reconstructed from the per-line table; check against `gap_clips.csv`) generate the validation speakers' clips
at chosen scalars and seeds from reference wavs in Drive `effort-pilot/refs_val/`, resumable, writing
`eval/<tag>/results.csv` in the baseline's column layout. Stage 1: scalars 0.5 0.8 1.2 1.6 2.0, seed 0 (750
clips, matching the baseline grid). Stage 2: seeds 1 and 2 at 1.6 and 2.0 for seed noise. v0 evaluation result (5 validation speakers, 30 lines, seed 0, 150 clips per setting; stage 2 gave seeds 1 and 2 for
p023 and 20 lines of p045 at 1.6 and 2.0). Median effort by setting 0.5 / 0.8 / 1.2 / 1.6 / 2.0: stock -3.80 / -1.10 /
3.59 / 7.05 / n.a.; fine-tune -5.45 / -5.18 / -4.28 / -3.77 / -3.00 (real regular -5.43, real loud 7.15). Spearman of effort vs
setting: stock 0.77 (0.77 to 0.86 per speaker), fine-tune 0.23 (0.24 to 0.43). Median similarity: stock 0.936 / 0.926 /
0.893 / 0.838; fine-tune 0.933 / 0.930 / 0.935 / 0.936 / 0.927. Speech-problem flag rate at 1.6: stock 0.54, fine-tune 0.30.
Words/s about 3.0 flat for the fine-tune (stock 2.55 at 1.6). Seed SD of effort on 50 matched cells at 1.6: stock 2.45,
fine-tune 1.75. Only 2 fine-tune clips fall within 1.5 of real-loud effort (9 clips above effort 3, none above 9).
Verdict: v0 keeps identity, intelligibility and pacing stable but loses almost all of the effort control (range about 2.5
effort points over 0.5 to 2.0 versus about 10.8 for stock); the calibration at the low end is right (0.5 gives real-regular
effort). The identity advantage at matched effort is confounded by speaker and by the tiny n at high effort. Candidate
causes (not yet tested): the scalar projection barely trained (0.65% change), too few steps or too low a learning rate
for an effect that is small relative to the token loss, and only 20% high-effort training clips.
Distribution check on the fine-tune (all seeds pooled): effort 10% / 50% / 90% quantiles by setting 0.5: -10.04 / -5.45 /
-1.90; 0.8: -9.82 / -5.18 / -1.29; 1.2: -8.56 / -4.28 / -1.00; 1.6: -7.36 / -2.90 / 1.04; 2.0: -6.68 / -2.67 / 1.11, so the whole
distribution shifts right by about 3 points with the scalar (the scalar acts, weakly) and the median gains only 0.23 between
1.6 and 2.0 when pooled (seed 0 alone: +0.77), so saturation versus a slope of about 1.7 points per unit is unresolved. 17 of
948 clips reach effort >= 3 (7 at 1.6, 10 at 2.0, none at or below 1.2; p045 9, p023 7, p059 1; shout 13, urgent 4, calm 0),
median similarity of those 17 is about 0.93, WER mostly 0. p045 S01 ("Watch out!") reaches >= 3 for all three seeds at 1.6.
The effect is line-dependent (exclamatory lines only), which suggests it is mostly the base model's text-driven prosody
and not the EARS loud mapping that was trained in.
v1 and extrapolation results (2026-10-30; 5 validation speakers x 10 lines, seed 0, 50 clips per setting; extrapolation on
p023, p045, p056 only). Median effort by setting: stock 0.3 to 1.6 -5.64 / -3.80 / -2.09 / 3.40 / 6.93; v0 0.5 to 2.0 -5.42 /
-5.44 / -4.31 / -4.44 / -2.97; v1 0.5 / 1.0 / 1.5 / 2.0 / 3.0 -6.33 / -4.61 / -4.57 / -3.91 / -3.09. Summary (stock, v0, v1):
effort range 12.57 / 2.45 / 3.23; Spearman 0.75 / 0.21 / 0.33; share of clips with effort >= 3: 29.2% / 2.4% / 0.4% (73, 6, 1
clips); median similarity 0.912 / 0.932 / 0.923; speech-problem flag rate 0.42 / 0.40 / 0.35. v0 extrapolation (scalars 2, 3, 4,
6; 3 speakers): median effort -2.78 / -2.82 / -3.85 / -2.87, Spearman 0.03, similarity about 0.92 throughout: scalars above 2.0
do nothing. By intent at the top setting v1 differs (calm -4.78, shout -2.10, urgent -1.16) while stock does not (calm 7.39,
shout 7.68, urgent 5.05). v1 training: 564 steps in 45.6 min, loss 6.6 to about 1.5 (lower than v0's 3.4, likely partly
memorization from 3x repeated clips), grad norm rising from about 1.5 to 5-9, no NaN. Verdict: v1 misses the pre-set bar
(Spearman >= 0.6, median effort >= +3 at the top scalar, similarity >= 0.90) by a wide margin; both LoRA fine-tunes keep
identity and intelligibility but do not reproduce the EARS loud mapping. Decision: stop fine-tuning with this recipe.
Token probe (`token_probe.py`; 12 training speakers, 204 regular + 108 loud reading clips, logistic regression on S3 token
frequencies, speaker-held-out CV): length-only AUC 0.476; token-frequency AUC 0.745 (raw) and 0.754 (speaker-centred);
shuffled-label controls 0.46 and 0.35. So mild loud reading is partly visible in the S3 tokens (and not through duration),
which does not support the idea that effort lives only in the frozen decoder; a T3-only fine-tune is not ruled out in
principle. The v0/v1 failure is more likely a subtle signal against the token loss with about 2 h of data and 400 to 560 steps.
The probe is crude (frequencies only), covers mild loud reading rather than shouting, and says nothing about the decoder.
SHELVED Chatterbox v1.1 pack (2026-10-30, files in `finetune_patch/v1_1_shelved/`, notebook `effort_v1_1_shelved_colab.ipynb`, untested, not run): a diagnostic (v1 adapter on
TRAINING speakers p001 to p006, scalars 0.5 and 3.0, to tell 'does not generalize to new speakers' from 'never learned the
mapping'), then a partial fine-tune (cond encoder, speech head, last 12 of 30 transformer layers, lr 5e-5, scalar projection at
20x, 8 epochs, high-effort clips repeated 2x), with a switch to full fine-tuning (about 8.5 GB of weights, gradients and
Adam states before activations; tight on a T4, optional 8-bit optimizer). `gen_eval.py` gained `--t3-weights` to load full
weights. Colab T4 time is a daily allowance, not a monthly one.
v1 plan (files in `finetune_patch/v1/`, notebook `effort_v1_colab.ipynb`, untested): scalar projection trained at 20x the LoRA
rate (LoRA 2e-4), loud clips labeled 3.0 and regular 0.5 (more leverage for the scalar), the 8 lowest-arousal emotion styles
dropped, clips with raw score >= 2.5 repeated 3x (about 55% of items), 12 epochs; all label changes applied at load time
from the existing manifest. Same session also runs the v0 extrapolation test (scalars 2, 3, 4, 6). Pass bar for v1:
Spearman >= 0.6 and median effort >= +3 at the high scalars with similarity >= 0.90; otherwise stop fine-tuning.
Scoring: patched copies of
`eval/asr_speaker.py` and `eval/score_baseline.py` (`eval_patch/`, only change: a `--root DIR` option) score the
fine-tune from `data_local/baselines/effort_ft/cfg0.50/`; `scripts/compare_systems.py` compares two scored roots
(by setting, Spearman of effort vs setting, similarity at matched achieved effort, near-real-loud summary,
seed noise, pilot criteria).

Applied and verified on the Mac (2026-10-28): `src/model.py` diff shows exactly the three intended
changes; `train.py` and `config.py` carry the new flag and settings. Nothing has been trained yet.

**Plan.**
- A (primary): LoRA fine-tune of standard Chatterbox with per-clip effort labels through
  `emotion_adv`, calm-reference and loud-target pairs, EARS subset of about 5 to 10 hours.
- B (training-free, new): steer the T3 activations with an effort vector estimated from EARS
  loud-minus-regular matched pairs, calibrated against the estimator. Cheap to try on the Mac
  and a better fallback than best-of-N.
- C (fallback): best-of-N with the estimator and the speech-problem filter (only about +0.03
  similarity from 3 draws).
- Pilot success criteria (my proposal, from the baseline at effort near real loud): speaker
  similarity of at least 0.90 (baseline 0.84), seed SD of effort at most 2.0 (baseline 3.25),
  speech-problem filter rate at most 30% (baseline 54%), monotone response to the scalar.

## Known limitations

- EARS "loud" reading reaches roughly anger-level spectral effort, not yelling or screaming;
  the extreme end exists only as non-speech clips.
- The pitch tracker is a simple autocorrelation method with strict voicing rules. Spot-check
  F0 on new audio before trusting it.
- Only 5 validation speakers: validation differences of a few AUC points are noise.
- Baseline results are one seed and one reference clip per speaker, so there are no error
  bars, and speaker differences may partly be reference-clip effects.
- The estimator is trained on real speech and has not been validated on synthetic audio;
  artifacts could inflate its score. Effort scores need listening checks.
- Whisper WER counts duplicated speech (repeated phrases) as errors, so it conflates
  duplication with unintelligibility, and WER > 0.5 catches only about 12% of the clips
  labeled unintelligible by ear.
- Listening labels come from one listener in a single pass with no second rater, and the
  unintelligible threshold is subjective. Detector rules tuned on the 150 labeled clips are
  optimistic until validated on fresh clips. Test-retest on 25 clips gave kappa 0.53 and a drift
  toward stricter labeling, so seed-0 labels probably under-count problems (especially
  duplication on short lines); exact label combinations are unreliable, and the binary
  problem / no-problem and "sounds calm" labels are the ones to trust. Conclusions that do not
  depend on labels (effort, identity, seed noise) and large effects (calm lines at 1.6, the
  estimator's AUC) are robust to this; precise per-line rates are not.
- WER measures Whisper and the TTS together; speaker-similarity thresholds are dataset
  dependent, so generated clips are compared with real-speech calibration, not a fixed cutoff.
- The estimator is trained on loud vs regular reading only; level and spectral cues are
  physically correlated, so "effort" here means a gain-invariant acoustic proxy. Scores are
  rank-only (zero is the loud/regular boundary, not "calm").
- Scoring emits a harmless `WavFileWarning` (non-data chunk skipped) when reading some WAVs.

**v2.0 limitations (2026-10-31)**
- The effort estimator is not validated on steered synthetic audio and rated k = 1 clips as near-yelling (about 20) while a listener
  heard "slightly louder, distressed"; effort scores of runaway or distorted clips are meaningless. One rater so far.
- Development uses the 0.6B model on a Mac CPU, 3 validation speakers, 8 dev lines and one seed; choices made on the validation
  speakers are optimistic, so the headline numbers must come from the untouched test speakers with the 30 test lines.
- Steering is additive at every decode step; strengths of k = 2 and above produced runaway generations (no end of speech) and
  maniacal or distorted audio. Only the talker is steered; the code predictor (15 of 16 codebooks) is not.
- Directions come from generation-based pairs (a regular or a loud reference clip, same target text), so they may encode reference
  differences other than effort (rate, clip length, recording level). The speaker-agreement figure (cosine 0.80 to 0.91) is slightly
  inflated because each speaker is in the mean.
- Mac wall times are unreliable (the machine slept during long runs); `gen_s` values above 100 s are sleep artifacts.
- The effort direction is not emotion-neutral: at k >= 1.25 an urgent or frightened line can sound manic, which the estimator, similarity and WER cannot see.
- The effort estimator saturates near 20 (real yelling), so it cannot rank steered clips beyond k = 1; the dose-response numbers are estimator-based and one seed.

## Setup

Two environments, because Chatterbox pins its own dependency versions.

**Project environment** (Python 3.10+; data, features, estimator, tests):
```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest
```

**Chatterbox environment** (Python 3.11; baseline generation, Whisper ASR, speaker embeddings):
```bash
python3.11 -m venv ~/venvs/chatterbox && source ~/venvs/chatterbox/bin/activate
pip install chatterbox-tts "setuptools<81"
pip install -e . --no-deps
```
Model weights (Chatterbox about 3 GB, plus Whisper and WavLM for scoring) download from
Hugging Face on first use. The project environment also needs `matplotlib` for the report. Heavy data lives in
`data_local/` (git-ignored) or wherever `EFFORT_DATA_DIR` points, for example a Google
Drive folder on Colab.

**Qwen environment** (Python 3.12; v2.0 steering experiments; CPU is enough for the 0.6B model):
```bash
brew install python@3.12 sox
python3.12 -m venv ~/venvs/qwen3tts && source ~/venvs/qwen3tts/bin/activate
pip install -U qwen-tts soundfile
```
`qwen-tts` pins transformers 4.57.3 and pulls torch. Model weights (about 2 GB for 0.6B-Base) download on first use. Reference
transcripts come from `scripts/transcribe_refs.py` (Chatterbox environment). Keep the Mac awake for long runs: `caffeinate -i`.

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
python -m efforttts.eval.asr_speaker                       # Step A: WER + speaker similarity
```

Project environment, after Step A:
```bash
python -m efforttts.eval.score_baseline                    # Step B: effort scores, tables, plot
```
Outputs land in `data_local/baselines/chatterbox/cfg0.50/` (`results.csv`, `asr_speaker.csv`,
`real_calibration.csv`, `gap_clips.csv`, `gap_report.png`).

Listening analysis (macOS, project env, run from the repo root):
```bash
python scripts/inspect_exag16.py     # picks clips to relisten to from existing scores
python scripts/label_clips.py        # blind labeling of the 1.6 clips (resumable)
python scripts/analyze_labels.py     # join labels with scores; detector AUCs, usable-clip yield
python scripts/analyze_labels_deep.py # stratified rates, threshold sweeps, similarity headroom
python scripts/analyze_seeds.py      # seed-to-seed stability and automatic best-of-N (needs several seeds)
python scripts/analyze_seed_labels.py # filter vs your labels on fresh seeds, test-retest agreement
python scripts/compare_systems.py --a data_local/baselines/chatterbox/cfg0.50 --b data_local/baselines/effort_ft/cfg0.50 --name-a stock --name-b effort_ft
python scripts/check_status.py       # which steps have been run, and what is pending
```
`label_clips.py` plays each clip with `afplay` in a fixed random order, shows only the
intended intent and the line text (no scores), and saves one row per clip to
`listening_labels_ex1.60.csv` (labels: ok, calm_when_loud, loud_when_calm, duplicated,
unintelligible; several can apply except ok). Keys: o, c, l, d, u, Enter, r, b, q.

Quick check that a generation run finished cleanly (expect `750 Counter({'ok': 750})`):
```bash
python3 - <<'EOF'
import csv, collections
r = list(csv.DictReader(open("data_local/baselines/chatterbox/cfg0.50/results.csv")))
print(len(r), collections.Counter(x["status"] for x in r))
EOF
```

### v2.0 commands
The ordered v2.0 command list (recon, baseline, extraction, sweeps, scoring, listening tools) lives in `qwen_v2/PLAN.md`
under *Environment and commands*; in zsh use `LINE_IDS` and `$=LINE_IDS`, never a variable called `LINES`.

## Next steps

**v2.0 (current; details and priorities in `qwen_v2/PLAN.md`)**
1. Perceptual calibration: listen to the ladders (k 0.35 to 1.0) for the first rung that sounds loud and the first that sounds like shouting; run `level_by_k.py` (done: level rises +8 dB at k = 0.35 and +25 dB at k = 1; now also checks peaks), listen to the raw and the equal-level ladders, extract a level-matched direction (`--match-level -26`) and an anger-based direction (`qwen_extract.py --loud-emotion anger`) and sweep it at small k.
2. Step A (identity, WER) on the steered roots and `worst_clips.py` to check the automatic measures against the ears.
3. If the clean range does not reach shouting: an anger-based direction, steering the code predictor, the 1.7B model on a T4, or a loudness stage in post.
4. Zero-shot baseline on all 5 validation speakers x 30 lines; then the final evaluation on the 10 test speakers with the frozen settings.
5. Blind human check (2+ raters) with an artifact label, then the write-up. Re-run the prior-art search first.

**v1.0 (closed 2026-10-30; kept for the record)**

1. Adopt a short written labeling rubric (what counts as duplicated, unintelligible, sounds
   calm) before any further labeling, and plan the multi-rater listening test (Week 4), since
   one listener's labels have kappa 0.53 against themselves. Further single-listener labeling
   has diminishing returns for the go/no-go.
2. Investigate the filter's false positives: print the Whisper transcripts for U01 and C07 and
   fix the cause (text normalization in `eval/wer.py`; measure speaking rate over active
   speech instead of total time, or skip it for short lines and lines with pauses). Keep the
   duplication ratio >= 1.15 (precision about 0.7 to 0.76) as the high-precision gate and the
   effort estimator as the detector for calm-sounding shout lines.
3. Extend to 5 seeds if useful (resumable), and run the other exaggeration settings with
   `--seeds 3` to see whether the effort seed noise (SD 3.25 at 1.6) is smaller at lower
   settings, and for fresh clips to validate the combined speech-problem
   filter and any other rule tuned on the 150 labeled clips.
4. Label the 1.2 clips (and optionally 0.8) with `label_clips.py --exaggeration 1.2` to find
   the best operating point for a best-of-N fallback and see how failure rates scale.
5. Optional: relabel the calm-line clips with a question that does not depend on the displayed
   intent ("is this a natural loud reading of the line?"), because loud_when_calm as recorded
   mostly tracks the displayed intent. Keep appropriateness separate from effort realism and
   intelligibility in the benchmark.
6. Change the WER reporting: clip per-clip WER at 1.0, report the fraction with WER > 0.5 per
   setting, and add the speech-problem filter (tentative cutoffs above) once validated.
7. Fine-tune pilot (see *Fine-tune feasibility*): clone the toolkit, confirm in code how
   `emotion_adv` and the prompt are handled, patch per-clip effort labels and calm references,
   train on 2 speakers, and test against the success criteria. In parallel try the
   activation-steering route on the Mac. Re-run the prior-art search before any public release.
8. Decide scope: "high effort up to loud/angry" (supported by EARS) versus true shouting.
9. Director (line to effort level), Week 3 to 4: write the rubric, collect your intended-loudness
   ratings for a larger line set (the 30-line script is too small to evaluate it), measure
   rank agreement, and score "director + exaggeration knob" end to end as the system baseline.

## Layout

- `src/efforttts/data`: download, inventory, splits, manifest
- `src/efforttts/effort`: acoustic features, estimator
- `src/efforttts/baselines`: test script, reference picker, Chatterbox generation
- `src/efforttts/eval`: WER, ASR + speaker similarity (Step A), effort scoring and report (Step B)
- `src/efforttts/train`, `demo`: planned
- `showcase/`: the Streamlit showcase (generator for Chatterbox dev clips, builder with one presentability filter, app, README with the hosting decision)
- `qwen_v2/`: v2.0 Qwen3-TTS steering (plan, generation, extraction, sweeps, listening and artifact tools); `qwen_v2/directions_*.pt` is EARS-derived: keep it out of git
- `finetune_patch/`: v1.0 Chatterbox fine-tune pack (v0, v1) and the shelved `v1_1_shelved/` (never run); `compare_systems.py`, `compare_runs.py`, `token_probe.py`: v1.0 analysis tools
- `configs/`: data policy, fixed 30-line test script (evaluation only; never train on it)
- `scripts/`: smoke tests; `notebooks/`: Colab bootstrap; `tests/`: pytest suite

## Data and licensing

- **EARS** (Meta Research / University of Hamburg) is licensed **CC-NC 4.0**
  (non-commercial); check the exact license text in the EARS repository. The dataset, transcripts, and derived tables are **not** in this
  repository. Any weights trained on it should be treated as non-commercial. The aggregate
  reference numbers above are summary statistics only; keep this repository private until a
  publishing decision is made.
- Chatterbox is MIT-licensed and watermarks every output; keep the watermark intact and
  disclose synthetic audio in demos.
- Qwen3-TTS (v2.0) is reported to be Apache-2.0 (confirm on the model card before any release). I have not verified whether its outputs are
  watermarked; disclose synthetic audio in demos either way.
- EARS-derived v2.0 artifacts stay local and out of git: reference clips and their Whisper transcripts, `directions_*.pt`, generated clips and
  ladders. Suggested `.gitignore` entries: `data_local/`, `qwen_recon.wav`, `ladder_*.wav`, `qwen_v2/directions_*.pt`, `EffortDataset/`, `tokens.npz`.
- Consenting adult voices only. No minors' voices in training or demo data.
- Verify the license of every model and dataset before use.