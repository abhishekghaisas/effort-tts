# Effort-tts

Speaker-preserving, continuous **vocal-effort control** for zero-shot TTS, plus an
**effort-realism evaluation harness**. The goal is high-effort speech (loud, strained,
projected) in a cloned speaker's own voice, controlled by a reproducible scalar instead of
a free-text tag, and a way to measure whether synthetic effort is realistic and whether
speaker identity survives.

Solo research project. Research and portfolio use only (see *Data and licensing*).

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

Week 1 of a four-week plan: evidence and go/no-go. Last updated 2026-10-03.

**Built**
- EARS download, inventory, QA, speaker-disjoint train/val/test split (seeded, gender-stratified)
- Per-file manifest with level and clipping statistics
- Gain-invariant acoustic features (spectral tilt, alpha ratio, Hammarberg, cepstral peak
  prominence, pitch, voicing) with a versioned cache
- Effort estimator v0 (logistic model, loud vs regular reading) with a feature-set ablation
  and out-of-distribution checks
- Fixed 30-line evaluation script with an EARS-overlap check (0 hits)
- Listening tools: `scripts/inspect_exag16.py` (clip picker) and `scripts/label_clips.py` (blind labeling helper), `scripts/analyze_labels.py` (label vs score analysis)
- Chatterbox baseline generation (resumable; reference-clip picker). Full run complete and
  verified: 750/750 jobs ok (5 validation speakers x 30 lines x 5 exaggeration values,
  cfg_weight 0.5, 1 seed)
- Baseline scoring: Whisper word error rate, WavLM speaker similarity (with a real-speech
  calibration), and level-free effort scoring with a 4-panel report. Run on the real
  baseline (see *Baseline results*)
- Test suite: 38 tests passing

**Not yet done:** multi-seed runs, WER-tail analysis, best-of-N headroom, fine-tune go/no-go,
the written gap report, listening test, demo.

## Baseline results (Chatterbox, Week 1)

Median effort is the level-free estimator score, referenced to the same speakers' real EARS
speech. Similarity is WavLM x-vector cosine to the speaker's centroid of real freeform windows.

| exaggeration | effort (median) | WER mean / median | similarity to centroid |
|---|---|---|---|
| 0.3 | -4.57 | 0.027 / 0.000 | 0.936 |
| 0.5 (default) | -3.80 | 0.017 / 0.000 | 0.936 |
| 0.8 | -1.10 | 0.045 / 0.000 | 0.926 |
| 1.2 | 3.59 | 0.065 / 0.000 | 0.893 |
| 1.6 | 7.05 | 0.632* / 0.055 | 0.838 |

\*The 1.6 WER mean is dominated by one clip (WER 74); prefer the median or the fraction of
clips with WER > 0.5 (see findings).

Real EARS reference effort (same speakers, same estimator): regular -5.43, loud 7.15,
anger 12.85, yelling 21.18.
Real similarity to own centroid: regular 0.978, loud 0.967, anger 0.948, ecstasy 0.923
(EARS spells it "extasy").

**Findings**
- Exaggeration raises effort monotonically for all 5 speakers, spanning roughly real regular
  to real loud. It never reaches anger or yelling. A spectral-only estimator (no pitch
  features) shows the same trend, so the rise is not just pitch.
- Identity drops much more than for real speakers. Real regular to loud costs about 0.011
  similarity; Chatterbox 0.5 to 1.6 (effort about equal to real loud) costs about 0.098.
  Synthetic audio also sits roughly 0.04 below real speech at any effort, so compare drops,
  not absolute values.
- Exaggeration 1.6 WER: the mean (0.632) is dominated by a single clip (WER 74, a Whisper
  repetition loop on a 3-word line); without it the mean is about 0.14. By Whisper, 5.3% of
  clips (8/150) have WER > 0.5 and 1.3% have WER > 1.0. The worst clips are mostly short
  lines (2 to 4 words) that are repeated or garbled (relistening confirmed duplicated speech in some), and the same line ("Take your time")
  fails for 4 of 5 speakers. By ear about 10% are unintelligible, roughly twice Whisper's
  rate, so ASR WER under-reports perceived problems. Speech is also slower (2.55 words/s vs
  about 2.9 elsewhere).
- Duplication check (words heard / words in the line >= 1.5, a rough cutoff): flagged clips
  per setting, out of 150, are 2, 1, 4, 2 and 8 for exaggeration 0.3, 0.5, 0.8, 1.2 and 1.6.
  At 1.6 all 8 sit on three lines: C05 "Take your time." (4 of 5 speakers), C02 (2) and S01
  "Watch out!" (2). Of the 8 clips with WER > 0.5, 6 are duplicates and 2 (S04, C06) are
  garbled, so Whisper-detected garbling is about 1.3% of clips, versus about 10% unintelligible
  by ear (which may include duplicates; labels needed). Small counts: the rise at 1.6 is
  suggestive, not established.
- By listening, 1.6 sounds like an unconvincing imitation of effort even though the
  estimator scores it as real-loud, so effort score and perceived realism are not the same
  thing.
- Estimator scores at 1.6 show no clear low-effort tail: no clip is at or below the
  default-setting median (-3.8), the three lowest shout lines score -3.6, -2.5 and -1.2, and
  the 10th percentile is about 2 to 3. The highest-scoring calm lines (11.7 to 15.8) are mostly p023, the speaker the
  knob moves furthest. Relistening to the 8 lowest-effort shout lines (scores -3.6 to 3.0): some do sound
  calm, so the estimator flags at least part of problem (1). Four of the 8 are the same line
  (S03, "Everybody out of the building, move, move!", four different speakers), confirmed by
  ear as sounding calm. S03 has the lowest line median at 1.6 (1.2; the next lowest is 3.0).
  But line medians do not track intent: calm lines (median of line medians about 7.3) score
  as high as shout lines (about 6.9) and urgent lines (about 6.8), and several calm lines
  (C06 10.6, C07 9.2, C04 9.1) outscore most shout lines. With 5 clips per line, within-line
  spread (typically 5 to 15 points) is as large as the spread between lines, so the S03 result
  may be sampling variability rather than something about its text. Needs multiple seeds.
  The 8 highest-effort calm lines and the 8 highest-WER clips were also relistened to and
  sounded relatively fine, so in this sample neither the estimator's top end nor WER located
  problems (2) and (3); only problem (1) was partly located (low-scoring shout lines). Some high-WER clips
  contained duplicated speech, which Whisper WER counts as errors even when the speech is
  clear, so duplication is a separate failure mode from unintelligibility. Not
  yet checked: whether high-scoring shout lines sound loud.
- Listening to all 150 clips at 1.60 (2026-10-03): some are fine, but many have one of three
  problems: (1) a line meant to be loud sounds calm, (2) a line meant to be calm is
  unnecessarily loud, (3) the clip is unintelligible. Rough counts by ear (no filenames noted): about 10% unintelligible; the other two problems
  roughly equal in frequency. Per-clip labels not yet recorded.
  (1) is a controllability failure; (2) is the knob ignoring text intent; (3) matches the WER tail.
- Line intent (calm, urgent, shout) barely changes effort at any setting; the knob does.
  This is consistent with problems (1) and (2) above: exaggeration acts as a global setting
  that is not tied to what the text calls for.
- The knob's reach is speaker-dependent: at 1.6, median effort ranges from 3.5 (p056) to
  9.9 (p023).
- Within a setting, similarity at a given effort varies widely (about 0.6 to 0.95 at effort
  5 to 8), which is what a best-of-N fallback would exploit. Not yet quantified.

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
  duplication with unintelligibility; a separate duplication check is needed.
- WER measures Whisper and the TTS together; speaker-similarity thresholds are dataset
  dependent, so generated clips are compared with real-speech calibration, not a fixed cutoff.
- The estimator is trained on loud vs regular reading only; level and spectral cues are
  physically correlated, so "effort" here means a gain-invariant acoustic proxy. Scores are
  rank-only (zero is the loud/regular boundary, not "calm").
- Scoring emits a harmless `WavFileWarning` (non-data chunk skipped) when reading some WAVs.

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

## Next steps

1. Record per-clip perceptual labels for the 1.6 clips (ok / calm when loud intended / loud
   when calm intended / duplicated / unintelligible), because neither the estimator's top end nor WER
   located problems (2) and (3). Then compare labels with estimator score, WER, similarity,
   line, speaker. Also run the variance split (line vs speaker vs intent) and a 5-seed
   experiment on S03 and a high-scoring line such as S05 to separate line effects from
   sampling variability.
2. Change WER reporting: clip per-clip WER at 1.0 and report the fraction with WER > 0.5 for
   every exaggeration setting, since means are dominated by single outliers. Add a
   duplication flag (heard / intended words >= 1.5) to the scoring output and report it per setting.
3. Re-run with `--seeds 3` (at least 0.8, 1.2, 1.6) to separate sampling noise from speaker
   effects.
4. Quantify best-of-N headroom from the existing clips: among clips with effort at or above
   real loud, the fraction with similarity above real anger.
5. Fine-tune go/no-go (Chatterbox tooling, base-model licenses, Colab throughput are all
   still unverified). Re-run the prior-art search before any public release.
6. Decide scope: "high effort up to loud/angry" (supported by EARS) versus true shouting.

## Layout

- `src/efforttts/data`: download, inventory, splits, manifest
- `src/efforttts/effort`: acoustic features, estimator
- `src/efforttts/baselines`: test script, reference picker, Chatterbox generation
- `src/efforttts/eval`: WER, ASR + speaker similarity (Step A), effort scoring and report (Step B)
- `src/efforttts/train`, `demo`: planned
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
- Consenting adult voices only. No minors' voices in training or demo data.
- Verify the license of every model and dataset before use.