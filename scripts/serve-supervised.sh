#!/usr/bin/env bash
# Run pulsar-serve so a bad context resize cannot leave nothing serving.
#
# /ctx and /kv re-exec the process. A request the VRAM cannot satisfy dies
# at cudaMalloc during the reload, and because exec REPLACED the old
# process there is no server left - the browser gets ERR_CONNECTION_REFUSED
# and the box needs an ssh to recover. pulsar-serve writes the running ctx
# to $PULSAR_CTX_STATE once the KV is allocated and it is listening, so a
# load that dies never records itself. This loop restarts at the last value
# that actually worked.
#
#   serve-supervised.sh MODEL.gguf [--host H] [--port P] [--ctx N] [extra...]
#
# Env: PULSAR_CTX_STATE (state file, default alongside the model),
# plus anything the engine reads (PULSAR_TIERS, PULSAR_CPU, PULSAR_KV...).
set -uo pipefail

BIN="${PULSAR_SERVE_BIN:-$(dirname "$0")/../target/release/pulsar-serve}"
[ -x "$BIN" ] || { echo "serve-supervised: no binary at $BIN" >&2; exit 1; }

MODEL=""
CTX=""
ARGS=()
while [ $# -gt 0 ]; do
    case "$1" in
        -m|--model) MODEL="$2"; shift 2 ;;
        --ctx) CTX="$2"; shift 2 ;;
        *) ARGS+=("$1"); shift ;;
    esac
done
[ -n "$MODEL" ] || { echo "serve-supervised: -m MODEL.gguf required" >&2; exit 1; }

STATE="${PULSAR_CTX_STATE:-${MODEL}.ctx-state}"
export PULSAR_CTX_STATE="$STATE"

# resume at the last ctx that reached "listening", if one was recorded
if [ -z "$CTX" ] && [ -r "$STATE" ]; then
    CTX="$(cat "$STATE" 2>/dev/null)"
fi
CTX="${CTX:-8192}"

FAILS=0
while :; do
    echo "serve-supervised: starting at ctx $CTX"
    "$BIN" -m "$MODEL" --ctx "$CTX" "${ARGS[@]}"
    RC=$?

    # 0 = a clean exit we did not ask for (the engine exits 0 on a fatal
    # load error too), anything else = crash. Either way, decide whether to
    # come back and at what ctx.
    GOOD="$(cat "$STATE" 2>/dev/null || true)"
    if [ -n "$GOOD" ] && [ "$GOOD" != "$CTX" ]; then
        # the run recorded a DIFFERENT working ctx than we launched with,
        # i.e. a /ctx resize succeeded and a later one failed: go back to it
        echo "serve-supervised: exit $RC; falling back to last good ctx $GOOD"
        CTX="$GOOD"
        FAILS=0
        continue
    fi

    # the ctx we launched with is itself suspect: halve it and retry, so a
    # box that cannot hold the recorded value still comes back up
    FAILS=$((FAILS + 1))
    if [ "$FAILS" -ge 5 ] || [ "$CTX" -le 1024 ]; then
        echo "serve-supervised: giving up after $FAILS attempts (ctx $CTX)" >&2
        exit 1
    fi
    CTX=$((CTX / 2))
    echo "serve-supervised: exit $RC; retrying at ctx $CTX"
    sleep 2
done
