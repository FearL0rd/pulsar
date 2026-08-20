# Shared cache + GPU topology for kv-codec-bench.sh (same policy as bench_kv.sh).
# Sourced only. Expects MODEL, ATTN_FAMILY, MIN_VRAM_MB, TF, SPEED, GREEDY, PASSES, FMTS, N, CTX.
# ---- host expert cache (auto from MemAvailable) — same as bench_kv.sh ----
if [ -n "${PULSAR_CACHE_GB:-}" ]; then
  CACHE_GB="$PULSAR_CACHE_GB"
else
  _AVAIL_KB=$(awk '/^MemAvailable:/ {print $2}' /proc/meminfo 2>/dev/null || echo 0)
  _AVAIL_GB=$(( ${_AVAIL_KB:-0} / 1024 / 1024 ))
  _HEADROOM="${PULSAR_CACHE_HEADROOM_GB:-16}"
  CACHE_GB=$(( _AVAIL_GB - _HEADROOM ))
  [ "$CACHE_GB" -lt 8 ] && CACHE_GB=8
  AUTO_CACHE_NOTE=" (auto: ${_AVAIL_GB}G avail - ${_HEADROOM}G headroom)"
fi
ATTN_VRAM_USER="${PULSAR_ATTN_VRAM_GB-}"
# CPU lane off by default: the lane's AVX2 vs GPU split is not reproducible
# and swamps the f32-vs-f32 teacher-force floor (bench_kv_tf.sh).
CPU="${PULSAR_CPU:-off}"
CPU_STEAL="${PULSAR_CPU_STEAL:-0}"

calc_attn_vram_gb() {
  local free_mb="${1:-0}"
  [[ "$free_mb" =~ ^[0-9]+$ ]] || free_mb=0
  local free_gb=$(( (free_mb + 512) / 1024 ))
  [ "$free_gb" -lt 1 ] && free_gb=1
  local tier_reserve="${PULSAR_ATTN_TIER_RESERVE_GB:-8}"
  local by_half=$(( free_gb / 2 ))
  local by_tier=$(( free_gb - tier_reserve ))
  [ "$by_tier" -lt 0 ] && by_tier=0
  local budget=$by_half
  if [ "$by_tier" -gt 0 ] && [ "$by_tier" -lt "$budget" ]; then budget=$by_tier; fi
  local floor=6
  local ceil=$(( free_gb - 4 ))
  [ "$ceil" -lt "$floor" ] && ceil=$floor
  [ "$budget" -lt "$floor" ] && budget=$floor
  [ "$budget" -gt "$ceil" ] && budget=$ceil
  echo "$budget"
}

# ---- GPU auto-selection (denylist + SM + PCIe) — same as bench_kv.sh ----
command -v nvidia-smi >/dev/null || { echo "ERROR: nvidia-smi not found" >&2; exit 1; }
if [ "$TF" = "1" ]; then
  command -v python3 >/dev/null || { echo "ERROR: python3 required for TF=1" >&2; exit 1; }
fi

mapfile -t GPU_ROWS < <(
  nvidia-smi --query-gpu=index,name,memory.total,memory.free,pcie.link.gen.max,pcie.link.width.max,pcie.link.gen.current,pcie.link.width.current,compute_cap \
    --format=csv,noheader,nounits 2>/dev/null | sed 's/, /,/g'
)
[ "${#GPU_ROWS[@]}" -gt 0 ] || { echo "ERROR: no GPUs reported by nvidia-smi" >&2; exit 1; }

CAND_IDX=(); CAND_NAME=(); CAND_TOTAL=(); CAND_FREE=(); CAND_PCIE=(); CAND_CC=()
is_denylisted() {
  local u="${1^^}"
  case "$u" in
    *1030*|*1050*|*1060*|*1650\ MAX-Q*|*MX150*|*MX250*|*MX330*|*UHD*|*P600*|*P620*) return 0 ;;
  esac
  return 1
}

echo "scanning GPUs (min ${MIN_VRAM_MB} MiB total VRAM)..."
for row in "${GPU_ROWS[@]}"; do
  IFS=',' read -r idx name total free gen width cgen cwidth cc_raw <<<"$row"
  idx="${idx// /}"; name="${name# }"; total="${total// /}"; free="${free// /}"
  gen="${gen// /}"; width="${width// /}"; cgen="${cgen// /}"; cwidth="${cwidth// /}"
  [[ "$total" =~ ^[0-9]+$ ]] || total=0
  [[ "$free" =~ ^[0-9]+$ ]] || free=0
  [[ "$gen" =~ ^[0-9]+$ ]] || gen="$cgen"
  [[ "$width" =~ ^[0-9]+$ ]] || width="$cwidth"
  [[ "$gen" =~ ^[0-9]+$ ]] || gen=0
  [[ "$width" =~ ^[0-9]+$ ]] || width=0
  pcie=$(( gen * width ))
  cc_raw="${cc_raw// /}"
  if [[ "$cc_raw" == *.* ]]; then cc_major="${cc_raw%%.*}"; cc_minor="${cc_raw#*.}"; else cc_major="$cc_raw"; cc_minor=0; fi
  [[ "$cc_major" =~ ^[0-9]+$ ]] || cc_major=0
  [[ "$cc_minor" =~ ^[0-9]+$ ]] || cc_minor=0
  cc=$(( cc_major * 10 + cc_minor ))
  if is_denylisted "$name"; then echo "  hide  GPU $idx  $name  (${total} MiB) — denylist"; continue; fi
  if [ "$total" -lt "$MIN_VRAM_MB" ]; then echo "  hide  GPU $idx  $name  (${total} MiB < ${MIN_VRAM_MB} MiB min)"; continue; fi
  echo "  cand  GPU $idx  $name  free=${free} MiB  PCIe gen${gen} x${width}  sm_${cc_raw:-?} (score=${pcie})"
  CAND_IDX+=("$idx"); CAND_NAME+=("$name"); CAND_TOTAL+=("$total")
  CAND_FREE+=("$free"); CAND_PCIE+=("$pcie"); CAND_CC+=("$cc")
done
n_cand=${#CAND_IDX[@]}
[ "$n_cand" -ge 1 ] || { echo "ERROR: no capable GPUs (need >= ${MIN_VRAM_MB} MiB after denylist)" >&2; exit 1; }

STREAM_I=0
for ((i = 1; i < n_cand; i++)); do
  better=0
  if [ "${CAND_CC[$i]}" -gt "${CAND_CC[$STREAM_I]}" ]; then better=1
  elif [ "${CAND_CC[$i]}" -eq "${CAND_CC[$STREAM_I]}" ]; then
    if [ "${CAND_PCIE[$i]}" -gt "${CAND_PCIE[$STREAM_I]}" ]; then better=1
    elif [ "${CAND_PCIE[$i]}" -eq "${CAND_PCIE[$STREAM_I]}" ]; then
      if [ "${CAND_FREE[$i]}" -gt "${CAND_FREE[$STREAM_I]}" ]; then better=1
      elif [ "${CAND_FREE[$i]}" -eq "${CAND_FREE[$STREAM_I]}" ] && [ "${CAND_TOTAL[$i]}" -gt "${CAND_TOTAL[$STREAM_I]}" ]; then better=1; fi
    fi
  fi
  [ "$better" -eq 1 ] && STREAM_I=$i
done
STREAM_PHYS="${CAND_IDX[$STREAM_I]}"; STREAM_NAME="${CAND_NAME[$STREAM_I]}"; STREAM_FREE="${CAND_FREE[$STREAM_I]}"

ATTN_I=""; ATTN_PHYS=""; ATTN_NAME=""
if [ "$n_cand" -ge 2 ]; then
  for ((i = 0; i < n_cand; i++)); do
    [ "$i" -eq "$STREAM_I" ] && continue
    if [ -z "$ATTN_I" ] || [ "${CAND_FREE[$i]}" -gt "${CAND_FREE[$ATTN_I]}" ]; then ATTN_I=$i
    elif [ "${CAND_FREE[$i]}" -eq "${CAND_FREE[$ATTN_I]}" ] && [ "${CAND_TOTAL[$i]}" -gt "${CAND_TOTAL[$ATTN_I]}" ]; then ATTN_I=$i; fi
  done
  ATTN_PHYS="${CAND_IDX[$ATTN_I]}"; ATTN_NAME="${CAND_NAME[$ATTN_I]}"; ATTN_FREE="${CAND_FREE[$ATTN_I]}"
fi

if [ -n "${PULSAR_GPU:-}" ]; then
  MANUAL=1
  echo
  echo "PULSAR_GPU set — auto-pick skipped (honoring your roles)"
  export CUDA_DEVICE_ORDER=PCI_BUS_ID
  if [ -z "${CUDA_VISIBLE_DEVICES:-}" ]; then
    CUDA_VISIBLE_DEVICES="$(IFS=','; echo "${CAND_IDX[*]}")"
    export CUDA_VISIBLE_DEVICES
    echo "CUDA_VISIBLE_DEVICES unset — defaulting to capable cards: $CUDA_VISIBLE_DEVICES"
  fi
  ATTN_DECISION_NOTE="manual roles (PULSAR_GPU set); family=$ATTN_FAMILY${GGUF_ARCH:+ arch=$GGUF_ARCH}"
else
  export CUDA_DEVICE_ORDER=PCI_BUS_ID
  export PULSAR_GPU=0
  unset PULSAR_ATTN_GPU
  _force_attn="$(should_force_attn "$ATTN_FAMILY" "$n_cand")"
  if [ "$n_cand" -ge 2 ]; then
    if [ "$_force_attn" -eq 1 ] && [ -n "${ATTN_PHYS:-}" ]; then
      CVD="${STREAM_PHYS},${ATTN_PHYS}"
      for ((i = 0; i < n_cand; i++)); do
        [ "$i" -eq "$STREAM_I" ] && continue
        [ "${CAND_IDX[$i]}" = "$ATTN_PHYS" ] && continue
        CVD="${CVD},${CAND_IDX[$i]}"
      done
      export CUDA_VISIBLE_DEVICES="$CVD"
      export PULSAR_ATTN_GPU=1
    else
      CVD="$STREAM_PHYS"
      for ((i = 0; i < n_cand; i++)); do
        [ "$i" -eq "$STREAM_I" ] && continue
        CVD="${CVD},${CAND_IDX[$i]}"
      done
      export CUDA_VISIBLE_DEVICES="$CVD"
    fi
  else
    export CUDA_VISIBLE_DEVICES="${STREAM_PHYS}"
  fi
  ATTN_DECISION_NOTE="$(attn_policy_note "$ATTN_FAMILY" "$_force_attn" "${GGUF_ARCH:-}")"
fi

export PULSAR_CACHE_GB="$CACHE_GB"
_attn_offload_on=0
if [ -n "${PULSAR_ATTN_GPU:-}" ] && [[ "${PULSAR_ATTN_GPU}" != "off" && "${PULSAR_ATTN_GPU}" != "-1" ]]; then
  _attn_offload_on=1
fi
_mla_auto_budget=0
if [ "$_attn_offload_on" -eq 0 ] && { [ "$ATTN_FAMILY" = "mla" ] || [ "$ATTN_FAMILY" = "k3" ]; } \
  && [ -n "${ATTN_PHYS:-}" ] && [ -n "${ATTN_FREE:-}" ]; then
  _mla_auto_budget=1
fi
if [ -n "$ATTN_VRAM_USER" ]; then
  if [[ "$ATTN_VRAM_USER" == "off" || "$ATTN_VRAM_USER" == "0" ]]; then
    unset PULSAR_ATTN_VRAM_GB
    ATTN_VRAM_NOTE=" (user: off)"
  else
    export PULSAR_ATTN_VRAM_GB="$ATTN_VRAM_USER"
    ATTN_VRAM_NOTE=" (user override)"
  fi
elif [ "$_attn_offload_on" -eq 1 ] && [ -n "${ATTN_PHYS:-}" ] && [ -n "${ATTN_FREE:-}" ]; then
  ATTN_VRAM_GB="$(calc_attn_vram_gb "$ATTN_FREE")"
  export PULSAR_ATTN_VRAM_GB="$ATTN_VRAM_GB"
  ATTN_VRAM_NOTE=" (auto: budget ${ATTN_VRAM_GB}G on attn)"
elif [ "$_mla_auto_budget" -eq 1 ]; then
  ATTN_VRAM_GB="$(calc_attn_vram_gb "$ATTN_FREE")"
  export PULSAR_ATTN_VRAM_GB="$ATTN_VRAM_GB"
  ATTN_VRAM_NOTE=" (auto ${ATTN_FAMILY}: budget ${ATTN_VRAM_GB}G)"
else
  unset PULSAR_ATTN_VRAM_GB
  ATTN_VRAM_NOTE=" (engine default)"
fi
unset PULSAR_TIERS 2>/dev/null || true
if [[ "$CPU" == "off" || "$CPU" == "0" ]]; then unset PULSAR_CPU; else export PULSAR_CPU="$CPU"; fi
export PULSAR_CPU_STEAL="$CPU_STEAL"

echo
if [ -n "${MANUAL:-}" ]; then
  echo "manual topology:"
  echo "  CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
  echo "  PULSAR_GPU=$PULSAR_GPU (stream)   PULSAR_ATTN_GPU=${PULSAR_ATTN_GPU:-unset} (attn)"
else
  echo "selected topology:"
  echo "  STREAM primary  physical GPU $STREAM_PHYS  $STREAM_NAME  (free ${STREAM_FREE} MiB, cc ${CAND_CC[$STREAM_I]}, PCIe ${CAND_PCIE[$STREAM_I]})"
  if [ -n "${ATTN_PHYS:-}" ]; then
    echo "  extra GPU       physical GPU $ATTN_PHYS  $ATTN_NAME  (free ${ATTN_FREE} MiB)"
  else
    echo "  extra GPU       (none)"
  fi
fi
echo "CUDA_DEVICE_ORDER=$CUDA_DEVICE_ORDER"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "PULSAR_GPU=$PULSAR_GPU  PULSAR_ATTN_GPU=${PULSAR_ATTN_GPU:-unset}"
echo "ATTN policy: ${ATTN_DECISION_NOTE:-n/a}"
echo "PULSAR_CACHE_GB=$PULSAR_CACHE_GB${AUTO_CACHE_NOTE:-}"
echo "PULSAR_ATTN_VRAM_GB=${PULSAR_ATTN_VRAM_GB:-unset}${ATTN_VRAM_NOTE}"
echo "PULSAR_CPU=${PULSAR_CPU:-off}  SPEED=$SPEED GREEDY=$GREEDY TF=$TF PASSES=$PASSES"
echo "PULSAR_KV will cycle: $FMTS"
echo "model: $MODEL"
echo "N=$N ctx=$CTX"
echo
