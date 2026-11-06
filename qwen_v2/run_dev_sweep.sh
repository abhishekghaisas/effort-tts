#!/usr/bin/env bash
# Layer and strength sweep on DEV lines (not the 30 test lines) with 3 validation speakers. About 1 hour on a Mac CPU.
# Run from the repo root, in the qwen3tts env, and keep the Mac awake:   caffeinate -i bash qwen_v2/run_dev_sweep.sh
set -e
cd "$(dirname "$0")/.."
COMMON="--size 0.6B --refs data_local/baselines/references --ref-transcripts data_local/baselines/references/ref_transcripts.json --lines qwen_v2/dev_lines.json --speakers p023 p045 p056 --seeds 0 --out data_local/baselines/qwen_v2"
python qwen_v2/qwen_gen.py $COMMON --tag dev_base --ks 0
for cfg in "14" "18" "21" "24" "27" "20 21 22 23"; do
  tag="dev_L$(echo $cfg | tr ' ' '_')"
  python qwen_v2/qwen_gen.py $COMMON --tag "$tag" --ks 1 2 4 --direction qwen_v2/directions_06B.pt --layers $cfg
done
echo "sweep generation finished"