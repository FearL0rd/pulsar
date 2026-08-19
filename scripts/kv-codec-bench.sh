#!/usr/bin/env bash
# Moved to docs/examples/kv-codec-bench.sh — keep this path working.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
exec "$ROOT/docs/examples/kv-codec-bench.sh" "$@"
