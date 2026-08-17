# Blackwell block-scaled FP4 mma, measured

Everything here was derived on a 5060 Ti (GB206, sm_120a) by running the
instruction and comparing against a CPU reference, not read off a datasheet.
The probes live in the session scratchpad; the results are reproduced below so
nobody has to re-derive them.

## Availability

FP4 tensor ops are architecture-specific. Plain `sm_120` rejects them:

    Instruction 'mma with block scale' not supported on .target 'sm_120'
    Feature '.kind::mxf4nvf4' not supported on .target 'sm_120'

`sm_120a` accepts them. `build.rs` requests `120a` (commit c44365f); before
that the 5060 Tis silently JIT'd the `compute_89` PTX floor. Note that
`-arch=sm_120a` still emits a `compute_120` PTX image for JIT and fails to
assemble; use `-gencode arch=compute_120a,code=sm_120a` for a cubin-only build.

The 4060 Ti is Ada (sm_89) and has no FP4 units, so it stays on the dp4a/int8
path. The per-device dense split already routes that correctly.

## Which instruction

Three block-scaled forms assemble. Only one matches the NVFP4 weight layout
pulsar already stores.

| kind | k | operands | scale | granularity |
|---|---|---|---|---|
| `mxf4nvf4` | 64 | e2m1 x e2m1 | `ue4m3`, `scale_vec::4X` | one per 16 |
| `mxf4` | 64 | e2m1 x e2m1 | `ue8m0`, `scale_vec::2X` | one per 32 |
| `mxf8f6f4` | 32 | e2m1 x e4m3 | `ue8m0`, `scale_vec::1X` | one per 32 |

`mxf8f6f4` is the only W4A8 form (fp4 weights, fp8 activations), but its
`ue8m0` power-of-two scale at one-per-32 does not fit our per-16 `ue4m3`
weight scales, so it cannot carry them. `mxf4nvf4` fits the stored format
exactly and forces fp4 on both operands (W4A4).

The syntax that assembles, with the trailing scale-type qualifier that is easy
to miss (omitting it gives a bare "Arguments mismatch"):

    mma.sync.aligned.m16n8k64.row.col.kind::mxf4nvf4.block_scale.scale_vec::4X.f32.e2m1.e2m1.f32.ue4m3
      {d0,d1,d2,d3}, {a0,a1,a2,a3}, {b0,b1}, {c0,c1,c2,c3},
      sa, {byte-id-a, thread-id-a}, sb, {byte-id-b, thread-id-b};

A is 4 registers, B is 2, C/D are 4 f32. `byte-id` must be 0 and `thread-id`
must be 0 or 1 for `scale_vec::4X`; ptxas range-checks both.

## Measured instruction rate

Eight independent accumulator chains per thread, 36 SMs, so this is issue
throughput rather than dependency latency. All three forms issue at the same
25.6 G inst/s, which means k-per-instruction converts directly into math.

| path | k/inst | TOPS |
|---|---|---|
| int8 `m16n8k16` (what the prefill GEMM uses today) | 16 | 105 |
| w4a8 `m16n8k32 mxf8f6f4` | 32 | 210 |
| fp4 `m16n8k64 mxf4nvf4` | 64 | 419 |

4x the tensor throughput of the current path, and the weights feed in with no
dequantization at all.

## Fragment layout

Thread `t` in the warp, `g = t >> 2`, `l = t & 3`. Register `i`, nibble `j`
(bits `4j..4j+3`).

    A:  row = g + 8 * (i & 1)      k = l * 8 + 32 * (i >> 1) + j
    B:  col = g                    k = l * 8 + 32 * i + j
    D:  d0,d1 at row g,     col l*2 + {0,1}
        d2,d3 at row g + 8, col l*2 + {0,1}

Verified against a random-data CPU reference: a wrong layout does not produce a
constant output ratio across all 32 lanes.

## Scale operand

`sa` holds four scale bytes, one per k-group of 16, in byte order 0..3. The
hardware reads each row's scales from one specific lane:

    A: row r  <- lane (r & 7) * 4 + 2 * thread-id + (r >> 3)
    B: col n  <- lane n * 4 + 2 * thread-id

Both were mapped by perturbing exactly one scale byte to 2.0 against a
background of 1.0 and observing which outputs moved and by how much.

## Scale encoding

`ue4m3` is OCP e4m3 with the sign bit ignored: `0x38` and `0xb8` both decode to
1.0, `0x7f` and `0xff` are NaN. Normals are `2^(e-7) * (1 + m/8)`, subnormals
at `e == 0` are `2^-6 * m/8`.

This matters for how pulsar stores NVFP4. Our decoder pairs a doubled-e2m1
codebook (`mxfp4_lookup4`, which decodes to int8) with a compensating 0.5 in
`ue4m3_half`. The two cancel, so the bytes in the file are already true e2m1
nibbles and true e4m3 scales. They feed the hardware unmodified.

## What still has to be built

1. An f32 to NVFP4 activation quantizer (per-16 groups, `ue4m3` scale),
   mirroring `quantize_q8_k`, plus its State scratch buffer.
2. A GEMM staging weight nibbles into smem in mma order. The file packs a
   16-value sub-block as low nibbles for values 0..7 and high nibbles for
   8..15, while the mma wants eight consecutive values per register, so the
   stage needs a nibble shuffle. Amortize it over the token tile.
3. Accuracy gate. This moves prefill activations from q8_K to fp4, so the
   token-exact match against llama.cpp will not survive by construction.
   Land it behind an env flag and measure quality separately before making it
   the default. Decode is unaffected: at MTP depth 2 the token count stays
   under the GEMM dispatch threshold.
