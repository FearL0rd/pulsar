#!/usr/bin/env bash
# Sequential unified KV bench: Laguna then DeepSeek-V4-Flash.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
BENCH="$ROOT/docs/examples/kv-codec-bench.sh"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=2,1
export PULSAR_GPU=0
export SKIP_WARM=1
export PASSES=2
export SPEED=1
export GREEDY=1
export TF=1

run_one() {
  local model="$1"
  echo
  echo "========== $(date -u +%H:%M:%S) $model =========="
  MODEL="$model" "$BENCH"
}

run_one /home/cesar/models/Laguna-S-2.1-Q2K-RouterF32-AttnQ8-SExpQ8-OutQ6.gguf
run_one /home/cesar/models/DeepSeek-V4-Flash-0731-Abliterated-DS4-Headroom128.gguf
echo BOTH_DONE
