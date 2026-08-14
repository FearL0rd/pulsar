#!/usr/bin/env python3
"""Rewrite the unsloth DSpark draft (arch `dflash`) as a 3-layer `deepseek4`
gguf so pulsar's existing dsv4 loader can build it.

The draft is already a complete stack of dsv4 layers under pulsar's own
tensor names (MLA lora pair, latent kv, output lora pair, sinks, sinkhorn
hyper-connection gates, router + shared expert + 256 MXFP4 routed experts).
The only reasons `Model::load` cannot take it as-is:

  1. the architecture string and every `dflash.*` metadata key, and
  2. token_embd / output, which a draft shares with its target instead of
     carrying.

So this copies the tensors through untouched, renames the metadata prefix,
and pulls the two shared tensors out of the target. Draft-only keys
(block_size, target_layers, mask_token_id) survive the rename as
`deepseek4.*`; pulsar ignores keys it does not read, and the speculative
glue needs them later.

usage: convert-dspark-dsv4.py DRAFT.gguf TARGET_SHARD.gguf OUT.gguf

TARGET_SHARD is whichever shard of the target holds token_embd.weight and
output.weight (shard 2 of the unsloth UD-Q2_K_XL split).
"""
import sys

import gguf

SRC_ARCH = "dflash"
DST_ARCH = "deepseek4"
SHARED = ("token_embd.weight", "output.weight")


def field_value(field):
    v = field.contents()
    if isinstance(v, bytes):
        v = v.decode("utf-8")
    return v


def main():
    if len(sys.argv) != 4:
        sys.exit(__doc__)
    draft_path, target_path, out_path = sys.argv[1:4]

    draft = gguf.GGUFReader(draft_path)
    writer = gguf.GGUFWriter(out_path, DST_ARCH)

    # ---- metadata: dflash.* -> deepseek4.*, everything else verbatim
    copied = renamed = 0
    for key, field in draft.fields.items():
        if key == "general.architecture":
            continue  # the writer already emitted DST_ARCH
        if key in ("GGUF.version", "GGUF.tensor_count", "GGUF.kv_count"):
            continue
        new_key = key
        if key.startswith(SRC_ARCH + "."):
            new_key = DST_ARCH + "." + key[len(SRC_ARCH) + 1 :]
            renamed += 1
        val = field_value(field)
        types = field.types
        if types and types[0] == gguf.GGUFValueType.ARRAY:
            writer.add_array(new_key, val)
        else:
            writer.add_key_value(new_key, val, types[0])
        copied += 1
    print(f"metadata: {copied} keys ({renamed} renamed to {DST_ARCH}.*)")

    # ---- tensors: draft first, then the two the draft borrows
    # No raw_shape: reader .data already carries the BYTE shape in numpy
    # order, which is exactly what the writer converts back to logical.
    # Passing .shape (logical, ggml order) makes it read 256 experts as a
    # row length and reject MXFP4.
    for t in draft.tensors:
        writer.add_tensor(t.name, t.data, raw_dtype=t.tensor_type)
    print(f"tensors: {len(draft.tensors)} from the draft")

    target = gguf.GGUFReader(target_path)
    found = {}
    for t in target.tensors:
        if t.name in SHARED:
            writer.add_tensor(t.name, t.data, raw_dtype=t.tensor_type)
            found[t.name] = (t.tensor_type, t.data.nbytes)
    missing = [n for n in SHARED if n not in found]
    if missing:
        sys.exit(f"{target_path}: missing {missing} (wrong shard?)")
    for n, (ty, nb) in found.items():
        print(f"  + {n} {ty} {nb / 1e9:.2f} GB from the target")

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()

    # ---- self-check: every draft tensor must survive byte-identical in
    # type, shape and size. A silent shape transpose here would only show
    # up as garbage logits much later.
    out = gguf.GGUFReader(out_path)
    got = {t.name: t for t in out.tensors}
    for t in draft.tensors:
        o = got.get(t.name)
        if o is None:
            sys.exit(f"self-check: {t.name} missing from output")
        if [int(d) for d in o.shape] != [int(d) for d in t.shape]:
            sys.exit(
                f"self-check: {t.name} shape {[int(d) for d in o.shape]} "
                f"!= {[int(d) for d in t.shape]}"
            )
        if o.tensor_type != t.tensor_type or o.data.nbytes != t.data.nbytes:
            sys.exit(f"self-check: {t.name} type/size changed")
    arch = field_value(out.fields["general.architecture"])
    n_layer = field_value(out.fields[f"{DST_ARCH}.block_count"])
    print(f"self-check OK: arch={arch}, {n_layer} layers, {len(got)} tensors")


if __name__ == "__main__":
    main()
