#!/usr/bin/env python3
"""KL divergence between two --teacher-force --dump-logits runs.

usage: kld.py ref.bin test.bin [LABEL]

With LABEL, prints one table row instead of the multi-line panel, so
kld-ab.sh can sweep a list of codecs into a single table.

Each file: u32 LE n_vocab, then one n_vocab f32 LE row per position
(pulsar-cli --teacher-force --dump-logits PATH). KLD is computed per
position over the full softmax (ref || test), f64, chunked so memory
stays flat. Reports the buun-style panel: median/mean/p95/max KLD plus
top-1 agreement.
"""
import sys

import numpy as np


def open_rows(path):
    f = open(path, "rb")
    n_vocab = int(np.fromfile(f, dtype=np.uint32, count=1)[0])
    return f, n_vocab


def main():
    ref_path, test_path = sys.argv[1], sys.argv[2]
    fa, va = open_rows(ref_path)
    fb, vb = open_rows(test_path)
    if va != vb:
        sys.exit(f"vocab mismatch: {va} vs {vb}")

    CHUNK = 64
    klds, agree, total = [], 0, 0
    while True:
        a = np.fromfile(fa, dtype=np.float32, count=CHUNK * va)
        b = np.fromfile(fb, dtype=np.float32, count=CHUNK * va)
        if a.size != b.size:
            sys.exit(f"row count mismatch ({ref_path}: +{a.size//va}, {test_path}: +{b.size//va})")
        if a.size == 0:
            break
        a = a.reshape(-1, va).astype(np.float64)
        b = b.reshape(-1, va).astype(np.float64)

        def log_softmax(x):
            x = x - x.max(axis=1, keepdims=True)
            return x - np.log(np.exp(x).sum(axis=1, keepdims=True))

        la, lb = log_softmax(a), log_softmax(b)
        klds.append((np.exp(la) * (la - lb)).sum(axis=1))
        agree += int((a.argmax(axis=1) == b.argmax(axis=1)).sum())
        total += a.shape[0]

    kld = np.concatenate(klds)
    label = sys.argv[3] if len(sys.argv) > 3 else None
    if label:
        print(
            f"{label:<10} {np.median(kld):>10.6f} {kld.mean():>10.6f}"
            f" {np.percentile(kld, 95):>10.6f} {kld.max():>10.6f}"
            f" {100.0 * agree / total:>8.2f}%"
        )
        return
    print(f"positions {total}, vocab {va}")
    print(
        f"KLD  median {np.median(kld):.6f}  mean {kld.mean():.6f}"
        f"  p95 {np.percentile(kld, 95):.6f}  max {kld.max():.6f}"
    )
    print(f"top-1 agreement {100.0 * agree / total:.2f}%")


if __name__ == "__main__":
    main()
