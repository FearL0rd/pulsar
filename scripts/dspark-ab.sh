#!/usr/bin/env bash
# DSpark A/B runbook (task: dspark heads on the DFlash trunk).
# Run on substrate from ~/pulsar. Downloads the Avesed Qwen3.6-27B DSpark
# draft, converts it, and A/Bs against nextn MTP and plain DFlash on the
# ThinkingCap target. Assumes the thinkingcap container is STOPPED
# (needs both cards).
set -euo pipefail
cd "$(dirname "$0")/.."

TARGET=/mnt/models/ThinkingCap-Qwen3.6-27B-Q4_K_M.gguf
DSPARK_DIR=/mnt/models/dspark-27b-hf
DSPARK_GGUF=/mnt/models/Qwen3.6-27B-DSpark-draft.gguf
MATH_PROMPT="Compute the sum of all integers n between 1 and 100 such that n^2 + n + 41 is divisible by 3. Show your work."
PROSE_PROMPT="Write a reflective essay about the experience of walking through a city at night, focusing on sound and memory."
N=200

echo "== ensure cards are free =="
docker stop thinkingcap >/dev/null 2>&1 || true
sleep 2

echo "== build =="
cargo build --release 2>&1 | grep -E "^error" && exit 1 || true
test -x target/release/pulsar-cli

echo "== fetch dspark draft =="
if [ ! -f "$DSPARK_DIR/model.safetensors" ]; then
  mkdir -p "$DSPARK_DIR"
  for f in config.json model.safetensors; do
    curl -L --fail -o "$DSPARK_DIR/$f" \
      "https://huggingface.co/Avesed/Qwen3.6-27B-DSpark/resolve/main/$f"
  done
fi

echo "== convert =="
if [ ! -f "$DSPARK_GGUF" ]; then
  python3 -m pip install --quiet --user gguf safetensors numpy 2>/dev/null || true
  python3 scripts/convert-dspark-draft.py "$DSPARK_DIR" "$DSPARK_GGUF"
fi

run() { # label, extra env..., -- , cli args...
  local label="$1"; shift
  local envs=()
  while [ "$1" != "--" ]; do envs+=("$1"); shift; done
  shift
  echo "--- $label ---"
  env "${envs[@]}" ./target/release/pulsar-cli "$@" 2>&1 \
    | grep -E "tok/s|accept|dspark|drafted" || true
}

for mode in math prose; do
  if [ "$mode" = math ]; then P="$MATH_PROMPT"; else P="$PROSE_PROMPT"; fi
  echo
  echo "===== $mode ====="
  run "nextn MTP (baseline)" PULSAR_MTP=1 -- \
    -m "$TARGET" --ctx 4096 -p "$P" -n $N --temp 0
  run "dspark draft, heads ON (conf 0.5)" PULSAR_DFLASH="$DSPARK_GGUF" -- \
    -m "$TARGET" --ctx 4096 -p "$P" -n $N --temp 0
  run "dspark draft, conf OFF (markov only)" PULSAR_DFLASH="$DSPARK_GGUF" PULSAR_DSPARK_CONF=off -- \
    -m "$TARGET" --ctx 4096 -p "$P" -n $N --temp 0
  run "dspark draft, heads OFF (plain dflash path)" PULSAR_DFLASH="$DSPARK_GGUF" PULSAR_NO_DSPARK=1 -- \
    -m "$TARGET" --ctx 4096 -p "$P" -n $N --temp 0
done

echo
echo "== output-identity gate: dflash greedy must match plain greedy =="
A=$(./target/release/pulsar-cli -m "$TARGET" --ctx 2048 -p "$MATH_PROMPT" -n 64 --temp 0 2>&1 | grep "ids" | tail -1)
B=$(PULSAR_DFLASH="$DSPARK_GGUF" ./target/release/pulsar-cli -m "$TARGET" --ctx 2048 -p "$MATH_PROMPT" -n 64 --temp 0 2>&1 | grep "ids" | tail -1)
if [ "$A" = "$B" ]; then echo "IDENTITY PASS"; else echo "IDENTITY MISMATCH"; echo "plain:  $A"; echo "dspark: $B"; fi
