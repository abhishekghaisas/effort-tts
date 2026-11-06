#!/usr/bin/env bash
# Second dev sweep: SMALL strengths (the first sweep showed k = 1 already overshoots) and a tight token limit (150 tokens = 12 s).
# Strengths: KS="0.05 0.1 0.2 0.35 0.5" by default; override with e.g.  KS="0.1 0.3" bash qwen_v2/run_dev_sweep2.sh "21"
# Usage (repo root, qwen3tts env), one quoted layer set per configuration:
#   caffeinate -i bash qwen_v2/run_dev_sweep2.sh "18" "21" "24"
set -e
KS="${KS:-0.05 0.1 0.2 0.35 0.5}"
EXTRA="${EXTRA:-}"            # e.g. EXTRA="--bound-factor 1.5"
DIRECTION="${DIRECTION:-qwen_v2/directions_06B.pt}"   # e.g. DIRECTION=qwen_v2/directions_06B_anger.pt
TAG_SUFFIX="${TAG_SUFFIX:-}"  # e.g. TAG_SUFFIX="_b" writes to dev2_L21_b instead of dev2_L21
cd "$(dirname "$0")/.."
COMMON="--size 0.6B --refs data_local/baselines/references --ref-transcripts data_local/baselines/references/ref_transcripts.json --lines qwen_v2/dev_lines.json --speakers p023 p045 p056 --seeds 0 --out data_local/baselines/qwen_v2 --max-new-tokens 150"
for cfg in "$@"; do
  tag="dev2_L$(echo $cfg | tr ' ' '_')$TAG_SUFFIX"
  python qwen_v2/qwen_gen.py $COMMON --tag "$tag" --ks $KS $EXTRA --direction $DIRECTION --layers $cfg
done
echo "sweep 2 generation finished"