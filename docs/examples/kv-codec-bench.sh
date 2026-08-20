#!/usr/bin/env bash
# Unified model-agnostic PULSAR_KV bench (docs/examples).
#
# Combines:
#   bench_kv.sh     — GPU auto-pick, tok/s, greedy-id first-divergence
#   bench_kv_tf.sh  — teacher-forced top-1 / |Δlogit| / top-5 Jaccard
#   kv-codec-bench  — all codecs, activation/fallback, multi-pass, HTML chart
#
# Usage:
#   MODEL=/path/to/model.gguf ./docs/examples/kv-codec-bench.sh
#   MODEL=... FMTS="f32 q4_0 turbo4" PASSES=2 TF=1 ./docs/examples/kv-codec-bench.sh
#   MODEL=... PROMPT_FILE=corpus.txt TF=1 ./docs/examples/kv-codec-bench.sh
#   CUDA_VISIBLE_DEVICES=2,1 PULSAR_GPU=0 PULSAR_ATTN_GPU=1 MODEL=... ./docs/examples/kv-codec-bench.sh
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
cd "$ROOT"
# shellcheck source=pulsar_gpu_lib.sh
source "$HERE/pulsar_gpu_lib.sh"

CLI="${PULSAR_CLI:-$ROOT/target/release/pulsar-cli}"
[ -x "$CLI" ] || { echo "build first: cargo build --release -p engine" >&2; exit 1; }

MODEL="${MODEL:?set MODEL= to a gguf}"
[ -f "$MODEL" ] || { echo "missing MODEL=$MODEL" >&2; exit 1; }

FMTS="${FMTS:-f32 fp8 fp16 int8 q8_0 q4_0 turbo8 turbo4 turbo3 turbo2 turbo3_tcq turbo2_tcq turbo1_tcq}"
N="${N:-64}"
CTX="${CTX:-512}"
PASSES="${PASSES:-2}"
SPEED="${SPEED:-1}"
GREEDY="${GREEDY:-1}"
TF="${TF:-1}"
MIN_VRAM_MB="${PULSAR_MIN_VRAM_MB:-8192}"
STEM="$(basename "$MODEL" .gguf)"
OUT="${OUT:-$ROOT/docs/examples/kv-bench-$STEM}"

PROMPT="${PROMPT:-The three most important inventions of the twentieth century were}"
TF_PROMPT="${TF_PROMPT:-List the first eight Fibonacci numbers, then explain each in one short sentence. The Fibonacci sequence starts with zero and one, and every following term is the sum of the two preceding terms. It appears across mathematics, nature, computer science, and art. Rabbit population modeling, spiral phyllotaxis in sunflowers, pinecone scales, the golden ratio, recursive algorithms, dynamic programming memos, memoized search, AVX-dominated numeric kernels, and quiescent memory access patterns all connect back to it. Write clearly and keep each explanation to a single short sentence.}"
if [ -n "${PROMPT_FILE:-}" ]; then
  TF_PROMPT="$(cat "$PROMPT_FILE")"
  PROMPT="$TF_PROMPT"
fi

ATTN_FAMILY="$(resolve_attn_family "$MODEL")"
echo "model family for ATTN: $ATTN_FAMILY${GGUF_ARCH:+ (general.architecture=$GGUF_ARCH)}"

# shellcheck source=kv_bench_topo.sh
source "$HERE/kv_bench_topo.sh"


mkdir -p "$OUT"
STAMP="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
CSV="$OUT/results.csv"
echo "pass,codec,tok_s,wall_s,tokens,rc,activated,fallback,kv_line,ids_head" > "$CSV"

# Grep ONE file only — two files prefixes "path:" and poisons numbers.
decode_line() {
  grep -E 'pulsar: [0-9]+ tokens in' "$1" 2>/dev/null | grep -v prefill | tail -1 || true
}

ids_to_nums() {
  # pulsar: ids [1, 2, 3] → one id per line
  grep -E 'pulsar: ids' "$1" 2>/dev/null | tail -1 \
    | sed 's/.*ids //; s/[][,]//g' | tr ' ' '\n' | grep -E '^[0-9]+$' || true
}

classify_activation() {
  local fmt="$1" log="$2"
  if [ "$fmt" = f32 ]; then echo "1 0"; return; fi
  if grep -qE "${fmt} KV cache on" "$log" 2>/dev/null; then echo "1 0"; return; fi
  echo "0 1"
}

run_decode() {
  local pass="$1" fmt="$2"
  local tag="${pass}_${fmt}"
  local log="$OUT/${tag}.log"
  local t0 t1 rc line tps tokens sec kvline ids activated fallback
  t0=$(date +%s.%N)
  set +e
  if [ "$fmt" = f32 ]; then
    env PULSAR_PROFILE=1 "$CLI" -m "$MODEL" --ctx "$CTX" -p "$PROMPT" -n "$N" --temp 0 \
      >"$OUT/${tag}.out" 2>"$log"
  else
    env PULSAR_KV="$fmt" PULSAR_PROFILE=1 "$CLI" -m "$MODEL" --ctx "$CTX" -p "$PROMPT" -n "$N" --temp 0 \
      >"$OUT/${tag}.out" 2>"$log"
  fi
  rc=$?
  set -e
  t1=$(date +%s.%N)
  sec=$(awk -v a="$t0" -v b="$t1" 'BEGIN{printf "%.2f", b-a}')
  line=$(decode_line "$log")
  tps=$(printf '%s\n' "$line" | grep -oE '[0-9]+\.[0-9]+ tok/s' | awk '{print $1}' || true)
  tokens=$(printf '%s\n' "$line" | grep -oE '[0-9]+ tokens in' | awk '{print $1}' || true)
  kvline=$(grep -E 'KV cache on \(' "$log" 2>/dev/null | tail -1 | sed 's/^.*pulsar: //' || true)
  ids=$(grep -E 'pulsar: ids' "$log" 2>/dev/null | tail -1 | sed 's/pulsar: ids //' | cut -c1-80 || true)
  read -r activated fallback <<<"$(classify_activation "$fmt" "$log")"
  if [ "$fmt" != f32 ] && [ "$activated" != 1 ]; then
    echo "  WARN: $fmt did not activate (arch fallback to f32)" >&2
  fi
  ids_to_nums "$log" > "$OUT/${tag}.nums"
  [ -z "$tps" ] && tps="0"
  [ -z "$tokens" ] && tokens="0"
  printf '%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n' \
    "$pass" "$fmt" "$tps" "$sec" "$tokens" "$rc" "$activated" "$fallback" \
    "$(echo "$kvline" | tr ',' ';' | tr '\n' ' ')" \
    "$(echo "$ids" | tr ',' ';' | tr '\n' ' ')"
}

id_diff() {
  awk 'NR==FNR{a[NR]=$1; m=NR; next} {b[FNR]=$1}
    END{ n=(m<FNR?m:FNR); c=0; first=0;
         for(i=1;i<=n;i++) if(a[i]!=b[i]){ c++; if(!first) first=i }
         if(n==0){ print "0/0 differ, first at #0"; exit }
         printf "%d/%d differ, first at #%d", c, n, first }' "$1" "$2"
}

# ---- speed + greedy-id collection ----
if [ "$SPEED" = "1" ]; then
  if [ "${SKIP_WARM:-0}" != "1" ]; then
    echo "=== pass 0 warmup (f32, discarded) ==="
    run_decode warm f32 | tee -a "$CSV"
    echo
  fi
  for pass in $(seq 1 "$PASSES"); do
    echo "=== pass $pass (canonical is last) ==="
    for fmt in $FMTS; do
      echo "-- $fmt --"
      run_decode "$pass" "$fmt" | tee -a "$CSV"
    done
    echo
  done
fi

LAST="$PASSES"
[ "$SPEED" = "1" ] || LAST=0

# ---- greedy first-divergence (bench_kv.sh) ----
if [ "$GREEDY" = "1" ] && [ "$SPEED" = "1" ]; then
  echo "quality (greedy ids — FIRST divergence is the real signal;"
  echo "          total mismatches is dominated by autoregressive chaos):"
  if [ "$PASSES" -ge 2 ] && [ -s "$OUT/1_f32.nums" ] && [ -s "$OUT/${LAST}_f32.nums" ]; then
    noise=$(id_diff "$OUT/1_f32.nums" "$OUT/${LAST}_f32.nums")
    printf '  %-12s %s   ← noise floor (f32 pass1 vs last)\n' "f32xf32" "$noise"
    echo "f32xf32,$noise" > "$OUT/greedy-quality.csv"
  else
    echo "  (need PASSES>=2 to compute f32xf32 noise floor)"
    echo "f32xf32," > "$OUT/greedy-quality.csv"
  fi
  for fmt in $FMTS; do
    if [ "$fmt" = f32 ]; then
      n=$(wc -l < "$OUT/${LAST}_f32.nums" 2>/dev/null || echo 0)
      printf '  %-12s baseline (%s tokens)\n' "$fmt" "$n"
      echo "f32,baseline $n tokens" >> "$OUT/greedy-quality.csv"
    elif [ -s "$OUT/${LAST}_f32.nums" ] && [ -s "$OUT/${LAST}_${fmt}.nums" ]; then
      result=$(id_diff "$OUT/${LAST}_f32.nums" "$OUT/${LAST}_${fmt}.nums")
      printf '  %-12s %s\n' "$fmt" "$result"
      echo "$fmt,$result" >> "$OUT/greedy-quality.csv"
    else
      printf '  %-12s (no id dump)\n' "$fmt"
    fi
  done
  echo
fi

# ---- teacher-force logit quality (bench_kv_tf.sh) ----
if [ "$TF" = "1" ]; then
  echo "=== teacher-force (chaos-free logits) ==="
  tf_run() {
    local fmt="$1" tag="${2:-$1}"
    local log="$OUT/tf_${tag}.log"
    set +e
    if [ "$fmt" = f32 ]; then
      env PULSAR_PROFILE=1 "$CLI" -m "$MODEL" --ctx "$CTX" -p "$TF_PROMPT" --teacher-force \
        > "$OUT/tf_${tag}.json" 2> "$log"
    else
      env PULSAR_KV="$fmt" PULSAR_PROFILE=1 "$CLI" -m "$MODEL" --ctx "$CTX" -p "$TF_PROMPT" --teacher-force \
        > "$OUT/tf_${tag}.json" 2> "$log"
    fi
    set -e
    if [ "$fmt" != f32 ]; then
      grep -qE "${fmt} KV cache on" "$log" \
        || echo "  WARN: $fmt did not activate during teacher-force" >&2
    fi
    printf '  %-12s %s positions\n' "$tag" "$(grep -c . "$OUT/tf_${tag}.json" 2>/dev/null || echo 0)"
  }
  echo "teacher-forcing f32 baseline..."
  tf_run f32
  echo "teacher-forcing f32 again (logit noise floor)..."
  tf_run f32 f32_noise
  echo "teacher-forcing each KV format..."
  for fmt in $FMTS; do
    [ "$fmt" = f32 ] && continue
    tf_run "$fmt"
  done
  echo
  echo "quality (per-position logits vs f32 baseline; chaos-free):"
  TFQ="$OUT/tf-quality.csv"
  echo "codec,n,top1_pct,mean_dlogit,max_dlogit,jaccard" > "$TFQ"
  compare_tf() {
    python3 - "$1" "$2" "$3" "$TFQ" <<'PY'
import json, sys
def load(p):
    d = {}
    try:
        fh = open(p, encoding="utf-8")
    except OSError:
        return d
    for line in fh:
        line = line.strip()
        if not line:
            continue
        try:
            o = json.loads(line)
        except json.JSONDecodeError:
            continue
        top = o.get("top") or []
        if not top:
            continue
        d[o["pos"]] = (top[0][0], float(top[0][1]), {t[0] for t in top[:5]})
    return d
a, b = load(sys.argv[1]), load(sys.argv[2])
label = sys.argv[3]
outp = sys.argv[4]
common = sorted(set(a) & set(b))
if not common:
    print(f"  {label:<12} (no overlapping positions — check tf_*.log)")
    open(outp, "a", encoding="utf-8").write(f"{label},0,,,,\n")
    raise SystemExit(0)
agree = sum(1 for p in common if a[p][0] == b[p][0])
d = [abs(a[p][1] - b[p][1]) for p in common]
j = [len(a[p][2] & b[p][2]) / 5.0 for p in common]
n = len(common)
top1 = 100 * agree / n
mean_d = sum(d) / n
max_d = max(d)
jac = sum(j) / n
print(f"  {label:<12} {n} pos | top-1 {top1:5.1f}% | mean |Δlogit| {mean_d:.4f} (max {max_d:.3f}) | top-5 Jac {jac:.2f}")
open(outp, "a", encoding="utf-8").write(f"{label},{n},{top1:.2f},{mean_d:.6f},{max_d:.6f},{jac:.4f}\n")
PY
  }
  echo "  f32xf32 (noise floor):"
  compare_tf "$OUT/tf_f32.json" "$OUT/tf_f32_noise.json" f32xf32
  for fmt in $FMTS; do
    [ "$fmt" = f32 ] && continue
    echo "  $fmt vs f32:"
    compare_tf "$OUT/tf_f32.json" "$OUT/tf_${fmt}.json" "$fmt"
  done
  echo
fi

echo "=== canonical (pass $LAST) ==="
python3 - "$CSV" "$OUT" "$STAMP" "$MODEL" "$N" "$CTX" "$STEM" "$LAST" <<'PY'
import csv, json, sys, html
from pathlib import Path
csv_path, out_dir, stamp, model, n, ctx, stem, last = sys.argv[1:9]
out = Path(out_dir)
rows = list(csv.DictReader(open(csv_path, encoding="utf-8")))
canon = [r for r in rows if r["pass"] == last]
print(f"{'codec':<14} {'tok/s':>8} {'wall_s':>8} {'act':>4} {'fb':>3}  kv")
for r in canon:
    print(f"{r['codec']:<14} {r['tok_s']:>8} {r['wall_s']:>8} {r['activated']:>4} {r['fallback']:>3}  {(r['kv_line'] or '')[:70]}")

greedy = {}
gq = out / "greedy-quality.csv"
if gq.exists():
    for line in gq.read_text(encoding="utf-8").splitlines():
        if "," in line:
            k, v = line.split(",", 1)
            greedy[k] = v
tfq = {}
tfp = out / "tf-quality.csv"
if tfp.exists():
    for rec in csv.DictReader(tfp.open(encoding="utf-8")):
        tfq[rec["codec"]] = rec

max_tps = max((float(r["tok_s"] or 0) for r in canon), default=1) or 1
bars = []
for r in canon:
    t = float(r["tok_s"] or 0)
    w = 100.0 * t / max_tps
    if r.get("fallback") == "1" and r["codec"] != "f32":
        color = "#475569"
        tag = " (fallback f32)"
    elif r["codec"] == "f32":
        color, tag = "#64748b", ""
    elif "turbo" in r["codec"]:
        color, tag = "#10b981", ""
    else:
        color, tag = "#6366f1", ""
    bars.append(
        f'<div class="row"><div class="label">{html.escape(r["codec"])}{tag}</div>'
        f'<div class="barwrap"><div class="bar" style="width:{w:.1f}%;background:{color}"></div></div>'
        f'<div class="val">{t:.2f}</div></div>'
    )

def tf_cell(fmt):
    rec = tfq.get(fmt) or {}
    if not rec.get("n") or rec.get("n") == "0":
        return "—"
    return f"{rec.get('top1_pct','')}% / Δ{rec.get('mean_dlogit','')} / J{rec.get('jaccard','')}"

title = html.escape(stem)
html_doc = f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<title>PULSAR_KV bench — {title}</title>
<style>
body {{ font: 15px/1.4 ui-sans-serif, system-ui, sans-serif; margin: 32px; background:#0b1020; color:#e8eefc; }}
h1 {{ font-size: 22px; margin: 0 0 6px; }}
.sub {{ color:#93a0c2; margin-bottom: 20px; }}
.row {{ display:grid; grid-template-columns: 220px 1fr 70px; gap:10px; align-items:center; margin: 6px 0; }}
.label {{ font-family: ui-monospace, monospace; font-size: 13px; }}
.barwrap {{ background:#1a2340; border-radius: 6px; overflow:hidden; height: 22px; }}
.bar {{ height:100%; border-radius:6px; }}
.val {{ text-align:right; font-variant-numeric: tabular-nums; }}
table {{ border-collapse: collapse; margin-top: 24px; width:100%; font-size: 13px; }}
th, td {{ border-bottom: 1px solid #243056; padding: 6px 8px; text-align:left; }}
th {{ color:#93a0c2; font-weight:600; }}
</style></head><body>
<h1>PULSAR_KV bench — {title}</h1>
<div class="sub">{html.escape(stamp)} · n={html.escape(n)} · ctx={html.escape(ctx)} · pass {html.escape(last)} · {html.escape(model)}</div>
{''.join(bars)}
<table>
<tr><th>codec</th><th>tok/s</th><th>activated</th><th>greedy vs f32</th><th>TF top-1 / |Δlogit| / Jac</th><th>KV</th></tr>
{''.join(
    f"<tr><td>{html.escape(r['codec'])}</td><td>{html.escape(r['tok_s'])}</td>"
    f"<td>{'yes' if r['activated']=='1' else 'fallback'}</td>"
    f"<td>{html.escape(greedy.get(r['codec'],''))}</td>"
    f"<td>{html.escape(tf_cell(r['codec']))}</td>"
    f"<td>{html.escape(r['kv_line'])}</td></tr>"
    for r in canon
)}
</table>
</body></html>
"""
(out / "chart.html").write_text(html_doc, encoding="utf-8")
(out / "canonical.json").write_text(json.dumps({"speed": canon, "greedy": greedy, "tf": tfq}, indent=2), encoding="utf-8")
print(f"wrote {out / 'chart.html'}")
print(f"wrote {out / 'canonical.json'}")
PY

echo DONE
