# Prefill chunk pipeline (PULSAR_PIPE) — design

Status: design locked, not yet implemented. Evidence below is from the
2026-08-23 profiling session (nsys sqlite at ~/prefill-tp.sqlite on
substrate, 7,436-token prompt, ctx 16384, MTP off, 3-card FFN TP).

## Why

TP prefill is structure-bound, not kernel-bound:

- Per-GPU busy over the profiled span: dev0 49%, dev1 45%, dev2 10%.
- dev1's idle is 2,524 gaps > 500us summing to 8.95s (avg 3.5ms); gaps
  under 500us total 0.06s. The idle is per-layer hop stalls, not
  scheduling noise.
- Proof it is not compute: PULSAR_FP4 (2.79x on the dominant GEMM)
  moved the wall 11.15 -> 11.14s. Cutting hop bytes/latency
  (PULSAR_TP_BF16 + PULSAR_P2P, now default on) gave 11.15 -> 10.12s.
- Dropping the x4 card entirely (PULSAR_TP3=off) gives 9.33s prefill
  but costs decode (34.3 -> 31.8 tok/s no-MTP): fewer hops beats more
  FLOPs, which is the same conclusion from the other direction.

Nothing inside one chunk can fill a 3.5ms stall: layer i+1 depends on
layer i. The filler has to be another chunk.

## Design

Two 512-token chunks in flight, chunk c+1 running exactly one layer
behind chunk c:

- Correctness: both cross-chunk dependencies are layer-i-to-layer-i.
  Attention at layer i of chunk c+1 needs KV rows layer i wrote for
  chunk c; GDN at layer i needs the recurrent state chunk c's layer i
  left. A per-(chunk, layer) CUDA event enforces both at once.
- Bit-exactness: per-chunk kernel order and math are unchanged; the
  stagger only interleaves chunks. check.sh's gates must stay green
  unchanged.
- Concurrency mechanism: two HOST THREADS, alternating chunk parity.
  The build already uses --default-stream=per-thread, so each thread
  gets its own default stream per device and no kernel wrapper needs a
  stream parameter. Cross-thread ordering is CUDA events only (the
  TpLink / banks ev_p/ev_b/ev_in pattern already models this).
- Graphs OFF in pipe mode initially: a span replays as one graph launch,
  so per-layer events cannot be recorded inside it. Plain launches cost
  ~0.6ms/span (measured note in eval_qwen35_span); acceptable for v1.
  v2 can capture per-parity graphs with per-layer event nodes.
- Scratch: the chunk-scratch subset of State + Qwen35Rt + TpBank must
  exist once per parity (cur, normed, xq, after_attn, FFN mid/gate/up,
  tb.* card-B mirrors). KV cache, GDN/conv states, weights, graphs
  cache stay shared. Implementation order:
    1. Mechanical "lane" extraction: move chunk scratch into a
       ChunkLane struct, engine still uses one lane. No behavior
       change; check.sh gate.
    2. Second lane + two-thread driver in forward_qwen35_inner behind
       PULSAR_PIPE=1, per-layer events, graphs disabled in pipe mode.
    3. Measure vs the 9.33s 2-card reference; iterate (graphs per
       parity, 3-lane if the tail card still bubbles).
- Scope guards for v1: dense qwen35 + TP only; no banks, no dflash, no
  MTP prefill interleave (mtp_prefill_fill stays outside the pipe,
  runs after both lanes drain); pipe only when n_chunks >= 3.
- Memory: one extra chunk-scratch set (~hundreds of MB at chunk 512).
  If the 262k-ctx KV auto-sizer cannot afford it, run pipe with
  PULSAR_PREFILL_CHUNK=256 (two 256 lanes ~= one 512 chunk).

## Stage-2 shape (refined after stage 1 landed, d78425d)

- Lanes live as lane-indexed scratch: Qwen35Rt.sc -> Vec<RtScratch>,
  TpBank.sc -> Vec<TbScratch>, FfnBank.sc -> Vec<FcScratch> (TbScratch/
  FcScratch already carry their own TpLinks and pos cells, so a lane is
  hop-self-contained on cards B/C by construction).
- Driver: std::thread::scope; split_at_mut hands each worker thread
  exclusive &mut to ITS lane's scratch before spawn (no aliasing), plus
  a shared view for the truly shared mutables.
- Shared mutables needing the narrow unsafe (raw ptr + device-event
  ordering contract): KV caches (st.kcache/vcache/tp_kcache), GdnState
  s/conv buffers, and nothing else. st.tok and weights are read-only;
  config scalars (ctx, kvq, prof, dev) copy per thread.
- State chunk-scratch that must move into the lane (from st.* usage
  census): cur, normed, attn_out, after_attn, q, k, v, heads, xq,
  gate_act, up_act, ffn_mid, midq, ffn_out, plus the rope/MoE set
  qrot, krot, shared_out, moe_out, router_logits, router_selected,
  router_weights. Head-stage buffers (last_row, head_xq, logits,
  amax_out, st.normed's head use) stay on State - the head runs outside
  the pipe after both lanes drain.
- eval_qwen35_layer signature becomes (&self, lane: LaneView, shared,
  il, l, pos, t); the non-pipe path builds a LaneView over lane 0 so
  decode and single-chunk prefill are untouched.
- Per-(chunk,layer) events: ring of 2 XEvents per layer; lane p waits
  the other lane's event[il] before layer il, records its own after.
  XEvent records on the recording THREAD's default stream and gates the
  waiter's - exactly the cross-thread edge needed (verified in kernels
  lib: XEvent/DeviceBuf/TpLink are Send).

## Targets

- Baseline 3-card with default hop levers: 10.11s (735 tok/s).
- 2-card reference (hops halved the crude way): 9.33s.
- Pipeline target: ~7s (~1050 tok/s) — max per-GPU busy plus the
  serial head/tail, hops hidden behind the other lane's compute.

## Related measured facts (same session)

- Staged f32 hops moved 106 GB through host RAM on this one prefill
  (61.8 GB H2D + 44.5 GB D2H, 5.67s of transfer time) before
  bf16+p2p defaults landed (f2e2944).
- Scalar GQA attention kernel is 20% of kernel time (no tensor cores,
  ctx^2 growth) and chunked-GDN is the other structural rewrite; both
  deferred until pulsar serves long-context.
