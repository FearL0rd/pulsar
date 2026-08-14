#!/usr/bin/env bash
# KV codec quality panel: teacher-force the same text once per PULSAR_KV
# codec and report full-softmax KL divergence against the exact f32 cache.
# The teacher-force loop runs the decode path token by token, so every
# position reads back the quantized cache - this measures exactly what a
# codec costs at generation time.
#
# Pick a prompt long enough that the cache is actually deep at the later
# positions; a 20-token prompt measures almost nothing.
#
# usage: kld-ab.sh MODEL.gguf PROMPT.txt [CODEC...]
set -euo pipefail

MODEL=${1:?usage: kld-ab.sh MODEL.gguf PROMPT.txt [CODEC...]}
PROMPT=${2:?usage: kld-ab.sh MODEL.gguf PROMPT.txt [CODEC...]}
shift 2
CODECS=("$@")
[ ${#CODECS[@]} -eq 0 ] && CODECS=(fp8 int8 q8_0 turbo8 q4_0 turbo4)
CLI=${CLI:-./target/release/pulsar-cli}
OUT=${OUT:-/tmp/pulsar-kld}

mkdir -p "$OUT"
run() { # $1 = codec
    PULSAR_KV=$1 "$CLI" -m "$MODEL" -f "$PROMPT" \
        --teacher-force --dump-logits "$OUT/$1.bin" 2>"$OUT/$1.log" \
        || { echo "kld-ab: $1 run failed, see $OUT/$1.log" >&2; return 1; }
    # pulsar prints "<codec> KV cache on (A GiB -> B GiB over N layers)"
    grep -o 'KV cache on ([^)]*)' "$OUT/$1.log" | head -1 || true
}

echo "kld-ab: reference pass, PULSAR_KV=f32" >&2
run f32 >/dev/null

printf '%-10s %10s %10s %10s %10s %9s\n' codec median mean p95 max top-1
for kv in "${CODECS[@]}"; do
    size=$(run "$kv") || continue
    python3 "$(dirname "$0")/kld.py" "$OUT/f32.bin" "$OUT/$kv.bin" "$kv"
    [ -n "$size" ] && echo "           ${size}"
done
