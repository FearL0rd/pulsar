# KV cache codecs

`PULSAR_KV` selects how the KV cache is stored. The default is `f32`
(bit-exact), which flips automatically to a quantized codec when an f32
cache at the requested context would starve the expert cache. On a
streaming model a too-big KV never OOMs, it just quietly eats the VRAM the
experts wanted, so the auto rule trades exactness for throughput rather
than letting the run degrade invisibly.

| codec | layout | vs f32 |
|---|---|---|
| `f32` | exact | 1x |
| `fp8` | e4m3 + per-row f32 scale | ~3.9x |
| `int8` | int8 + per-row f32 scale | ~4.0x |
| `fp16` | IEEE half | ~2.0x |
| `q8_0` | 32-wide blocks, f16 scale + 32 int8 | ~3.8x |
| `q4_0` | 32-wide blocks, f16 scale + 16 nibbles | ~7.1x |
| `turbo8` / `turbo4` | q8_0 / q4_0 with a fixed orthogonal rotation | same as base |

The turbo codecs fold a rotation into K before append and into Q before
attention. Since the rotation is orthogonal the scores are unchanged
(`(Q@Piᵀ)·(K@Piᵀ)ᵀ = Q@Kᵀ`), but per-32-block outliers get spread across
the block so no single lane dominates the block scale. V is untouched.

MLA models (`Family::Mla`, `K3`) keep their own compact latent cache and
accept only `fp8` / `fp16` / `f32`. Dense qwen35 (`n_expert == 1`) runs
the dense-split path and stays on f32.

## Measuring

`scripts/kld-ab.sh MODEL.gguf PROMPT.txt [CODEC...]` teacher-forces the
same text once per codec and reports full-softmax KL divergence against
the exact f32 cache, plus top-1 agreement. Teacher-force runs the decode
path token by token, so every position reads back the quantized cache and
the number reflects what the codec costs at generation time.

Use a prompt long enough that the cache is actually deep at the later
positions. A 20-token prompt measures almost nothing. Both panels below
use the same 366-token passage.

A second f32 run scores exactly 0.000000 / 100%, so these numbers are
codec loss and not decode nondeterminism.

## Panels

Laguna-S-2.1 Q2K (Qwen35 family, GQA per-head cache), 366 positions:

| codec | median | mean | p95 | max | top-1 |
|---|---|---|---|---|---|
| int8 | 0.006844 | 0.014234 | 0.054154 | 0.183114 | 93.99% |
| q8_0 | 0.006473 | 0.013348 | 0.043782 | 0.468668 | 93.99% |
| turbo8 | 0.006511 | 0.013092 | 0.043377 | 0.197810 | 92.90% |
| fp8 | 0.008073 | 0.015043 | 0.055868 | 0.127170 | 92.90% |
| turbo4 | 0.022370 | 0.039871 | 0.121238 | 0.572356 | 90.71% |
| q4_0 | 0.023702 | 0.044723 | 0.147275 | 0.800544 | 89.89% |

DeepSeek-V4-Flash-0731 UD-Q2_K_XL (Dsv4 family, fused 512-wide latent
row), same passage:

| codec | median | mean | p95 | max | top-1 |
|---|---|---|---|---|---|
| q8_0 | 0.008632 | 0.017436 | 0.069854 | 0.144986 | 92.37% |
| int8 | 0.009744 | 0.018209 | 0.059204 | 0.249939 | 93.22% |
| fp8 | 0.011914 | 0.024100 | 0.076610 | 0.601992 | 92.66% |
| turbo4 | 0.017760 | 0.032939 | 0.128123 | 0.323266 | 92.37% |
| q4_0 | 0.031917 | 0.052479 | 0.162358 | 0.955787 | 89.55% |

## What the panels say

**fp8 is the worst of the 8-bit codecs on both models.** It loses to int8
at an identical stride (`head_dim+4`) and identical bytes, because e4m3
spends four of its eight bits on exponent range that the per-row scale
already provides. That is why the automatic default is `int8`. The MLA
latent cache keeps fp8 only because it is the sole quantized latent
format.

**The rotation earns its keep where outliers dominate, which is at 4 bits
and more on dsv4 than on Laguna.** On Laguna, turbo8 and q8_0 are a wash
and turbo4 barely beats q4_0. On dsv4 the fused latent row has far
stronger outliers and turbo4 nearly halves the median against q4_0
(0.0178 vs 0.0319), cuts the worst case by 3x (0.323 vs 0.956), and lands
at the same 92.37% top-1 as q8_0 while using half the bytes. For long
context on dsv4, `PULSAR_KV=turbo4` is the codec to reach for.

**Read `max`, not just `median`.** On dsv4, q8_0 and fp8 have similar
medians but fp8's worst position is 4x worse. A rare badly-quantized
position is a rare wrong token, which is what a user actually notices.
