#!/usr/bin/env bash
set -u
LOG=/home/cesar/pulsar/docs/examples/kv-bench-both.log
for i in $(seq 1 180); do
  if grep -q '^BOTH_DONE$' "$LOG" 2>/dev/null; then
    echo FINISHED
    tail -80 "$LOG"
    exit 0
  fi
  if ! pgrep -f 'kv-codec-bench' >/dev/null && ! pgrep -x pulsar-cli >/dev/null; then
    echo DIED
    tail -50 "$LOG"
    exit 1
  fi
  last=$(tail -1 "$LOG" 2>/dev/null || true)
  echo "tick $i $(date +%H:%M:%S) | $last"
  sleep 30
done
echo TIMEOUT
tail -30 "$LOG"
exit 2
