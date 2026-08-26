# qwen4exp (Qwen3.8-Flash-Next 180B-A6B) port notes

References:
- unsloth llama.cpp fork, branch `qwen4exp/qwen3.8-flash-next`
  (mainline PR #27742, open as of 2026-08-26), built + serving on this
  box at /mnt/models/llama.cpp-qwen4exp — the correctness oracle for
  the whole port. `src/models/qwen4exp.cpp` is the reference graph.
- Model on substrate: /mnt/models/qwen38-flash-next-gguf/UD-Q4_K_XL/
  (104 GiB, 4 shards). Everything below verified against the real file.
- Measured on the fork (2026-08-26, 8 expert layers/GPU, rest CPU-mmap):
  pp512 90-173 t/s (page-cache dependent), tg64 ~21 t/s, flat to 16k.
  The pulsar port has to beat this to be worth serving.

## Why this model

125B backbone + 51B n-gram table + 4B MTP, 6B active. The n-gram table
(26.8 GiB of the file) is pure `get_rows` — near-zero compute, ideal
CPU-mmap resident. Only 12 of 48 layers carry KV (24 KiB/token fp16 →
6 GiB at the native 262144; 3 GiB with int8/turbo codecs), the GDN
layers are O(1) state. After the TP KV work (15d6e76, 7ced49b, e6e57f7)
context is NOT the constraint on this box — expert placement is, and
that is pulsars warm-census/tier machinerys home turf. Also: the
qwen4exp graph in the fork hard-asserts on quantized KV
(qwen4exp.cpp:544); pulsar KV codecs have no such restriction once the
attention arm honors PULSAR_KV (same fix shape as 58bc312).

## Shape (verified from the GGUF header + tensor census)

- arch `qwen4exp`, 48 layers, n_embd 2560, vocab 248320, ctx 262144
- full attention every 4th layer (il%4==3, 12 layers): 24 Q heads x
  head_dim 256, 2 KV heads, PARTIAL rotary 64/256, freq_base 10e6,
  M-RoPE sections [11,11,10,0]; attn_q (2560,12288) = q+gate
  interleaved, sigmoid output gate (same as qwen35); q/k norms (256,)
- GDN layers (other 36): conv_kernel 4, state 128, 16 k-heads, 48
  v-heads (time_step_rank 48), inner 6144; conv_dim = 2*2048+6144 =
  10240; merged attn_qkv (2560,10240) + attn_gate (2560,6144).
  ONE functional delta vs qwen35 GDN: SIGMOID output gate, not silu.
- MoE EVERY layer: 512 experts top-10, ff_exp 640, softmax router
  (ffn_gate_inp F32), shared expert ff 640 with sigmoid gate
  (ffn_gate_inp_shexp F32 (2560,))
- Hyper-connections (NEW): hc_count 4, low_rank 320. Per layer, for
  attn and ffn each: hc_*_norm (10240,), hc_*_down (10240,320),
  hc_*_up (320,10240), hc_*_inject (10240,4); plus output_hc_{norm,
  down,up} before the head. Residual stream is 4x2560; read = norm ->
  down -> silu -> up (low-rank gate), write = inject-weighted. NOT the
  dsv4 formulation (Sinkhorn, full-rank) — self-contained, do not share.
- PLE n-gram embeddings (NEW): per_layer_token_embd (160, 320001536)
  IQ4_NL, 26.8 GiB — 16 hash heads x ~20M rows each (head_offsets /
  head_vocab_sizes in metadata), 3-gram, hash multipliers in
  ple.layer_multipliers. Host-side: hash the token history, gather
  rows (160-dim per layer input, embedding_length_per_layer_input
  160), feed every layer. blk.1 additionally has a ple mixer
  (ple_conv1d/ple_key/ple_value + norms), ple.layers=[1].
- QSA indexer (NEW) on the 12 full-attn layers: indexer.q_proj
  (2560,512) 4 heads x 128, indexer.k_proj (2560,128) 1 head, norms;
  top_k 2048, compress_ratio 4. Dense attention is bit-identical below
  top_k + ratio - 1 = 2051 cached tokens (free oracle), ~3% position
  divergence at 8192 per the PR.
- MTP: ABSENT from this GGUF — unsloth conversion dropped the 4B MTP
  block. Spec decode needs our own convert pass from HF safetensors
  (Qwen/Qwen3.8-Flash-Next) or PULSAR_NGRAM as stopgap.
- Vision: out of scope (pulsar is text-only today).

## Plan: new `Family::Qwen4Exp`, kernels shared with Qwen35

K3 is the precedent: own family + graph, GDN chunk/decode kernels
reused (the mma v2 stack applies as-is; sigmoid gate is a kernel flag
or epilogue swap). MoE machinery (build + warm census + tiers)
unchanged. HC and PLE are new, self-contained lanes.

### Phase 1 — load + census
arch detect, hparams (ssm.*, hc.*, ple.*, indexer.*), tensor map,
tokenizer (stock qwen), PLE table stays CPU always (get_rows lane).
Gate: loads, census prints, layer/device plan sane.

### Phase 2 — dense text graph, greedy parity
Full forward: HC residual (4 streams), PLE gather + layer-1 mixer, GDN
(sigmoid gate), gated full attention (dense, no indexer), MoE top-10 +
shared expert. Decode-only graph; prefill loops tokens like qwen35.
Gate: greedy token match vs the fork at prompts < 2051 tokens (dense ==
QSA there, so parity is exact by construction). check.sh selftest added.

### Phase 3 — placement + perf
Expert tiers from warm census (hotlist seed via hotlist-gen), TP where
it pays, prefill batching through the chunk pipeline. Gate: bench.sh
beats the fork numbers above (they are the bar).

### Phase 4 — QSA indexer
Small GEMM + topk into the existing paged windows; unlocks >2051-token
exactness and long-ctx speed. Reuse the 2051-token dense oracle as the
selftest (bit-identical below budget, jaccard vs reference above).

### Phase 5 — MTP (needs tensors we do not have yet)
Own converter for the MTP block, then the qwen35 spec loop applies
(spec_safe_prefix_cache already reasons about GDN rewind).

## Open questions
- interleaved mrope (rope type 40, IMROPE) vs pulsar mrope: verify
  the text-only position path matches (sections [11,11,10,0], text
  positions collapse to standard rope over 64 dims — confirm in ref).
- PLE hash exact spec (multipliers/offsets math) — read from the fork
  converter + set_input path, then mirror.
- shared-expert gate: sigmoid(x·w) scaling, confirm epilogue order vs
  ref.
