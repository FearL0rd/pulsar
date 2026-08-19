# kv-codec-bench.sh

Unified, **model-agnostic** `PULSAR_KV` bench. One script now covers what
used to be split across [`bench_kv.sh`](./bench_kv.md) (tok/s + greedy-id
first-divergence) and [`bench_kv_tf.sh`](./bench_kv_tf.md) (teacher-forced
logits), plus activation/fallback tracking and an HTML chart.

```sh
cd /path/to/pulsar
MODEL=/home/cesar/models/Laguna-S-2.1-Q2K-RouterF32-AttnQ8-SExpQ8-OutQ6.gguf \
  ./docs/examples/kv-codec-bench.sh
```

GPU auto-selection lives in [`kv_bench_topo.sh`](./kv_bench_topo.sh) and
matches `runpulsar.sh` / `bench_kv.sh` (denylist hides 1060-class cards,
stream primary = highest SM, `PCI_BUS_ID` order). Pin by setting both
`CUDA_VISIBLE_DEVICES` and `PULSAR_GPU`.

## What it does

1. **Topology** — scan `nvidia-smi`, hide denylisted / undersized cards,
   export `CUDA_VISIBLE_DEVICES` + `PULSAR_CACHE_GB` + optional attn budget.
2. **Speed** — warmup f32, then `PASSES` (default 2) decode runs per codec.
   Last pass is canonical (`scripts/bench.sh` rule).
3. **Greedy quality** — first-divergence of greedy ids vs f32. `f32xf32`
   (pass 1 vs last) is the noise floor. Read FIRST divergence, not totals.
4. **Teacher-force** — same token sequence through every codec; top-1 %,
   mean |Δlogit|, top-5 Jaccard vs f32. `f32xf32` should be ~100% / 0.0.
5. **Chart** — `docs/examples/kv-bench-<stem>/chart.html` plus CSV/JSON.

Codecs the arch cannot honor (turbo3 / turbo2 / `*_tcq` off dsv4) stay in
the table as fallback and do not abort the sweep.

## Environment

| var | default | what |
|---|---|---|
| `MODEL` | required | any pulsar gguf |
| `FMTS` | all 13 codecs | space-separated `PULSAR_KV` names |
| `N` | `64` | generated tokens (speed + greedy) |
| `CTX` | `512` | context |
| `PASSES` | `2` | decode repeats; last is canonical |
| `SPEED` | `1` | `0` = skip tok/s loop |
| `GREEDY` | `1` | `0` = skip first-divergence |
| `TF` | `1` | `0` = skip teacher-force (needs `python3`) |
| `PROMPT` | short bench.sh line | decode / greedy prompt |
| `TF_PROMPT` | ~120-token Fibonacci paragraph | teacher-force sequence |
| `PROMPT_FILE` | unset | overrides both prompts from a file |
| `SKIP_WARM` | `0` | `1` = skip discarded f32 warmup |
| `OUT` | `docs/examples/kv-bench-<stem>` | artifact directory |
| `PULSAR_CLI` | `target/release/pulsar-cli` | binary |

GPU / memory knobs match `bench_kv.sh`: `PULSAR_MIN_VRAM_MB`, `PULSAR_GPU`,
`PULSAR_ATTN_GPU`, `CUDA_VISIBLE_DEVICES`, `PULSAR_CACHE_GB`,
`PULSAR_ATTN_VRAM_GB`, `PULSAR_CPU` (default **off** so the TF floor stays
bit-clean), `PULSAR_CPU_STEAL`.

```sh
# speed only (no teacher-force)
TF=0 MODEL=model.gguf ./docs/examples/kv-codec-bench.sh

# quality only, long corpus
SPEED=0 GREEDY=0 PROMPT_FILE=passage.txt MODEL=model.gguf ./docs/examples/kv-codec-bench.sh

# pin 3090+V100, hide 1060
CUDA_VISIBLE_DEVICES=2,1 PULSAR_GPU=0 \
  MODEL=model.gguf ./docs/examples/kv-codec-bench.sh
```

A thin wrapper remains at `scripts/kv-codec-bench.sh` and execs this file.
