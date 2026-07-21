#!/usr/bin/env python3
"""Convert a DeepSpec-style DFlash/DSpark draft (HF safetensors) into a
pulsar dflash-draft gguf.

Maps the trunk 1:1 onto the tensor names DraftModel::load expects and, when
the checkpoint carries them (DSpark), also emits the markov head and the
confidence head. embed_tokens / lm_head are skipped: the draft shares the
target model's embedding table and output head (mtp_use_dedicated_embeddings
is false on every published checkpoint).

usage: convert-dspark-draft.py <hf_dir_or_repo_snapshot> <out.gguf>
needs: pip install gguf safetensors numpy
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open

import gguf

RENAME = {
    "fc.weight": "dflash_fc.weight",
    "hidden_norm.weight": "dflash_hidden_norm.weight",
    "norm.weight": "output_norm.weight",
    "markov_head.markov_w1.weight": "markov_w1.weight",
    "markov_head.markov_w2.weight": "markov_w2.weight",
    "confidence_head.proj.weight": "confidence_proj.weight",
}
LAYER_RENAME = {
    "input_layernorm.weight": "attn_norm.weight",
    "self_attn.q_proj.weight": "attn_q.weight",
    "self_attn.k_proj.weight": "attn_k.weight",
    "self_attn.v_proj.weight": "attn_v.weight",
    "self_attn.q_norm.weight": "attn_q_norm.weight",
    "self_attn.k_norm.weight": "attn_k_norm.weight",
    "self_attn.o_proj.weight": "attn_output.weight",
    "post_attention_layernorm.weight": "post_attention_norm.weight",
    "mlp.gate_proj.weight": "ffn_gate.weight",
    "mlp.up_proj.weight": "ffn_up.weight",
    "mlp.down_proj.weight": "ffn_down.weight",
}
SKIP = {"embed_tokens.weight", "lm_head.weight"}


def main():
    src, out = Path(sys.argv[1]), Path(sys.argv[2])
    cfg = json.loads((src / "config.json").read_text())
    dfc = cfg.get("dflash_config", {})
    target_layer_ids = cfg.get("target_layer_ids") or dfc["target_layer_ids"]
    mask_id = cfg.get("mask_token_id") or dfc.get("mask_token_id")
    markov_rank = int(cfg.get("markov_rank") or dfc.get("markov_rank") or 0)
    has_conf = bool(cfg.get("enable_confidence_head"))

    w = gguf.GGUFWriter(str(out), "dflash-draft")
    w.add_uint32("dflash-draft.block_count", cfg["num_hidden_layers"])
    w.add_uint32("dflash-draft.embedding_length", cfg["hidden_size"])
    w.add_uint32("dflash-draft.feed_forward_length", cfg["intermediate_size"])
    w.add_uint32("dflash-draft.attention.head_count", cfg["num_attention_heads"])
    w.add_uint32("dflash-draft.attention.head_count_kv", cfg["num_key_value_heads"])
    w.add_uint32("dflash-draft.attention.key_length", cfg["head_dim"])
    rp = cfg.get("rope_parameters") or {}
    rope_theta = float(cfg.get("rope_theta") or rp["rope_theta"])
    w.add_float32("dflash-draft.rope.freq_base", rope_theta)
    if rp.get("rope_type") == "yarn":
        w.add_float32("dflash-draft.rope.scaling.factor", float(rp["factor"]))
        w.add_uint32(
            "dflash-draft.rope.scaling.original_context_length",
            int(rp["original_max_position_embeddings"]),
        )
    elif int(cfg.get("max_position_embeddings", 0)) == 262144:
        # z-lab drafts train with the TARGET's yarn (factor 64 over a
        # 4096 native window) even though the HF config says "default";
        # the known-good 35B conversion carried exactly these values
        w.add_float32("dflash-draft.rope.scaling.factor", 64.0)
        w.add_uint32("dflash-draft.rope.scaling.original_context_length", 4096)
    block_size = cfg.get("block_size") or dfc.get("block_size")
    if block_size is None:
        block_size = 16  # DeepSpec Qwen3.6-35B default; fal config omits it
        print("WARNING: config has no block_size, defaulting to 16", file=sys.stderr)
    w.add_uint32("dflash-draft.dflash.block_size", block_size)
    w.add_uint32("dflash-draft.dflash.mask_token_id", mask_id)
    w.add_array("dflash-draft.dflash.target_layer_ids", target_layer_ids)
    # dspark keys mark DeepSpec-trained drafts (next-token row convention);
    # do not tag plain z-lab style DFlash conversions
    if markov_rank or has_conf:
        w.add_uint32("dflash-draft.dspark.markov_rank", markov_rank)
        w.add_bool("dflash-draft.dspark.confidence_head", has_conf)
        w.add_bool(
            "dflash-draft.dspark.confidence_with_markov",
            bool(cfg.get("confidence_head_with_markov")),
        )

    files = sorted(src.glob("*.safetensors"))
    # confidence bias is a scalar: carry it in metadata, not as a tensor
    conf_bias = 0.0
    for f in files:
        with safe_open(str(f), framework="pt") as st:
            if "confidence_head.proj.bias" in st.keys():
                conf_bias = float(st.get_tensor("confidence_head.proj.bias").float()[0])
    if has_conf:
        w.add_float32("dflash-draft.dspark.confidence_bias", conf_bias)
    assert files, f"no safetensors in {src}"
    n_written = 0
    for f in files:
        with safe_open(str(f), framework="pt") as st:
            for name in st.keys():
                if name in SKIP or name == "confidence_head.proj.bias":
                    continue
                if name in RENAME:
                    out_name = RENAME[name]
                elif name.startswith("layers."):
                    _, il, rest = name.split(".", 2)
                    out_name = f"blk.{il}.{LAYER_RENAME[rest]}"
                else:
                    print(f"WARNING: unmapped tensor {name}, skipping", file=sys.stderr)
                    continue
                data = st.get_tensor(name).float().numpy()
                # matrices must be Q8_0: the engine's upload path passes
                # F16 through RAW and the q8_0 matmuls would read noise.
                # confidence_proj stays F16 (host-side reader).
                if out_name == "confidence_proj.weight":
                    w.add_tensor(out_name, data.astype(np.float16))
                elif data.ndim >= 2:
                    q = gguf.quants.quantize(data, gguf.GGMLQuantizationType.Q8_0)
                    w.add_tensor(
                        out_name,
                        q,
                        raw_shape=q.shape,
                        raw_dtype=gguf.GGMLQuantizationType.Q8_0,
                    )
                else:
                    w.add_tensor(out_name, data)
                n_written += 1

    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    print(f"wrote {out} ({n_written} tensors, markov_rank={markov_rank}, confidence={has_conf})")


if __name__ == "__main__":
    main()
