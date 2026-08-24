/* Qwen3.5/3.6 (qwen35moe) hybrid GDN kernels.
 *
 * Included by pulsar_kernels.cu after gqa_kernels.inc (reuses
 * gqa_test_randf). The Gated DeltaNet layers keep O(1) recurrent
 * state instead of KV: a depthwise conv window over the mixed qkv
 * projection and a per-head [128 x 128] delta-rule state. Decode-only
 * shapes (n_tok == 1); prefill loops tokens.
 *
 * Reference: llama.cpp delta-net-base.cpp build_delta_net_autoregressive
 * + qwen35moe.cpp build_layer_attn_linear.
 */

/* Depthwise causal conv (kernel size K, per channel) + SiLU over one
 * token. state [K-1][n_chan] rolls forward: window = [state | x].
 * kern layout matches gguf ssm_conv1d [K, n_chan]: taps contiguous
 * per channel, tap K-1 = current token. */
__global__ static void qwen35_conv_step_kernel(
        float *out,          /* [n_chan] */
        const float *x,      /* [n_chan] this token's qkv row */
        const float *kern,   /* [n_chan][K] */
        float *state,        /* [K-1][n_chan] rolling window */
        uint32_t n_chan,
        uint32_t k) {
    uint32_t c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= n_chan) return;
    const float *kc = kern + (uint64_t)c * k;
    float acc = kc[k - 1] * x[c];
    for (uint32_t t = 0; t + 1 < k; t++) {
        acc += kc[t] * state[(uint64_t)t * n_chan + c];
    }
    /* roll: state rows shift left, x enters at the back */
    for (uint32_t t = 0; t + 2 < k; t++) {
        state[(uint64_t)t * n_chan + c] = state[(uint64_t)(t + 1) * n_chan + c];
    }
    state[(uint64_t)(k - 2) * n_chan + c] = x[c];
    out[c] = acc / (1.0f + expf(-acc)); /* silu */
}

extern "C" int pulsar_qwen35_conv_step(
        void *out_dev, const void *x_dev, const void *kern_dev,
        void *state_dev, uint32_t n_chan, uint32_t k) {
    if (!out_dev || !x_dev || !kern_dev || !state_dev || n_chan == 0 || k < 2) {
        return 0;
    }
    qwen35_conv_step_kernel<<<(n_chan + 255u) / 256u, 256>>>(
            (float *)out_dev, (const float *)x_dev, (const float *)kern_dev,
            (float *)state_dev, n_chan, k);
    return cuda_ok(cudaGetLastError(), "qwen35 conv step launch");
}

/* L2-normalize rows in place: x / sqrt(sum(x^2) + eps). NOT rms norm
 * (no 1/n, no weight) - ggml_l2_norm semantics. One block per row. */
__global__ static void qwen35_l2_norm_kernel(
        float *x, uint32_t rows, uint32_t dim, float eps) {
    uint32_t row = blockIdx.x;
    if (row >= rows) return;
    float *xr = x + (uint64_t)row * dim;
    float sum = 0.0f;
    for (uint32_t i = threadIdx.x; i < dim; i += blockDim.x) {
        sum += xr[i] * xr[i];
    }
    __shared__ float partial[256];
    partial[threadIdx.x] = sum;
    __syncthreads();
    for (uint32_t st = blockDim.x >> 1; st > 0; st >>= 1) {
        if (threadIdx.x < st) partial[threadIdx.x] += partial[threadIdx.x + st];
        __syncthreads();
    }
    const float scale = rsqrtf(partial[0] + eps);
    for (uint32_t i = threadIdx.x; i < dim; i += blockDim.x) {
        xr[i] *= scale;
    }
}

extern "C" int pulsar_qwen35_l2_norm(
        void *x_dev, uint32_t rows, uint32_t dim, float eps) {
    if (!x_dev || rows == 0 || dim == 0) return 0;
    qwen35_l2_norm_kernel<<<rows, 256>>>((float *)x_dev, rows, dim, eps);
    return cuda_ok(cudaGetLastError(), "qwen35 l2 norm launch");
}

/* Gated DeltaNet autoregressive step, one token, all heads in one
 * launch. Per v-head h with state S [dim x dim] (S[i][j]: i = k-dim,
 * j = v-dim), k/q from k-head h / (H_v / H_k):
 *   q_h *= 1/sqrt(dim)
 *   S   *= exp(g[h])
 *   sk_j = sum_i S[i][j] k_i
 *   d_j  = (v_j - sk_j) * beta[h]
 *   S[i][j] += k_i * d_j
 *   o_j  = sum_i S[i][j] q_i
 * One block per head, dim threads (thread j owns column j: S[i][j]
 * reads are coalesced across the block). dim <= 256. */
__global__ static void qwen35_gdn_step_kernel(
        float *out,          /* [H_v][dim] */
        float *state,        /* [H_v][dim][dim] in-place */
        const float *q,      /* [H_k][dim] (already conv+l2, pre-scale) */
        const float *k,      /* [H_k][dim] */
        const float *v,      /* [H_v][dim] */
        const float *g,      /* [H_v] log-decay (negative) */
        const float *beta,   /* [H_v] in (0,1) */
        uint32_t h_v,
        uint32_t h_k,
        uint32_t dim) {
    uint32_t h = blockIdx.x;
    uint32_t j = threadIdx.x;
    if (h >= h_v || j >= dim) return;
    /* ggml_repeat TILES along the head dim: v-head h reads k-head
     * h %% h_k (not the GQA-style block mapping h / (h_v/h_k)) */
    const uint32_t kh = h % h_k;
    const float *qh = q + (uint64_t)kh * dim;
    const float *kk = k + (uint64_t)kh * dim;
    float *S = state + (uint64_t)h * dim * dim;
    const float decay = expf(g[h]);
    const float qscale = rsqrtf((float)dim);

    __shared__ float sq[256];
    __shared__ float sk_sh[256];
    for (uint32_t i = j; i < dim; i += blockDim.x) {
        sq[i] = qh[i] * qscale;
        sk_sh[i] = kk[i];
    }
    __syncthreads();

    /* decay + sk in one pass down column j */
    float sk = 0.0f;
    for (uint32_t i = 0; i < dim; i++) {
        float s = S[(uint64_t)i * dim + j] * decay;
        S[(uint64_t)i * dim + j] = s;
        sk += s * sk_sh[i];
    }
    const float d = (v[(uint64_t)h * dim + j] - sk) * beta[h];
    /* rank-1 update + output in a second pass */
    float o = 0.0f;
    for (uint32_t i = 0; i < dim; i++) {
        float s = S[(uint64_t)i * dim + j] + sk_sh[i] * d;
        S[(uint64_t)i * dim + j] = s;
        o += s * sq[i];
    }
    out[(uint64_t)h * dim + j] = o;
}

extern "C" int pulsar_qwen35_gdn_step(
        void *out_dev, void *state_dev,
        const void *q_dev, const void *k_dev, const void *v_dev,
        const void *g_dev, const void *beta_dev,
        uint32_t h_v, uint32_t h_k, uint32_t dim) {
    if (!out_dev || !state_dev || !q_dev || !k_dev || !v_dev || !g_dev ||
        !beta_dev || h_v == 0 || h_k == 0 || h_v % h_k != 0 || dim == 0 ||
        dim > 256u) {
        return 0;
    }
    qwen35_gdn_step_kernel<<<h_v, dim>>>(
            (float *)out_dev, (float *)state_dev,
            (const float *)q_dev, (const float *)k_dev, (const float *)v_dev,
            (const float *)g_dev, (const float *)beta_dev, h_v, h_k, dim);
    return cuda_ok(cudaGetLastError(), "qwen35 gdn step launch");
}

/* Deinterleave the attention layers' fused q projection: input rows
 * are per-head [q dim | gate dim]; outputs are contiguous q and gate. */
__global__ static void qwen35_split_gate_kernel(
        float *q, float *gate, const float *fused,
        uint32_t n_head, uint32_t dim) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t total = n_head * dim;
    if (idx >= total) return;
    uint32_t h = idx / dim;
    uint32_t d = idx % dim;
    q[idx] = fused[(uint64_t)h * 2 * dim + d];
    gate[idx] = fused[(uint64_t)h * 2 * dim + dim + d];
}

extern "C" int pulsar_qwen35_split_gate(
        void *q_dev, void *gate_dev, const void *fused_dev,
        uint32_t n_head, uint32_t dim) {
    if (!q_dev || !gate_dev || !fused_dev || n_head == 0 || dim == 0) return 0;
    uint32_t total = n_head * dim;
    qwen35_split_gate_kernel<<<(total + 255u) / 256u, 256>>>(
            (float *)q_dev, (float *)gate_dev, (const float *)fused_dev,
            n_head, dim);
    return cuda_ok(cudaGetLastError(), "qwen35 split gate launch");
}

/* out *= sigmoid(gate) elementwise (attention output gating; also the
 * shared expert's scalar gate with gate broadcast handled host-side). */
__global__ static void qwen35_sigmoid_gate_kernel(
        float *x, const float *gate, uint32_t n) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    x[i] *= 1.0f / (1.0f + expf(-gate[i]));
}

extern "C" int pulsar_qwen35_sigmoid_gate(
        void *x_dev, const void *gate_dev, uint32_t n) {
    if (!x_dev || !gate_dev || n == 0) return 0;
    qwen35_sigmoid_gate_kernel<<<(n + 255u) / 256u, 256>>>(
            (float *)x_dev, (const float *)gate_dev, n);
    return cuda_ok(cudaGetLastError(), "qwen35 sigmoid gate launch");
}

/* Laguna per-head output gate: x is [n_tok][n_head][head_dim], gate is
 * [n_tok][n_head] (gating="per-head"), so one softplus'd scalar scales a
 * whole head row. softplus is computed in the numerically stable form
 * (log1p(exp(-|g|)) + max(g,0)) so large logits do not overflow expf. */
__global__ static void laguna_head_gate_kernel(
        float *x, const float *gate, uint32_t n_tok, uint32_t n_head, uint32_t head_dim) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t n = n_tok * n_head * head_dim;
    if (i >= n) return;
    uint32_t head = i / head_dim;           /* flat (tok, head) index */
    float g = gate[head];
    float sp = log1pf(expf(-fabsf(g))) + fmaxf(g, 0.0f);
    x[i] *= sp;
}

extern "C" int pulsar_laguna_head_gate(
        void *x_dev, const void *gate_dev,
        uint32_t n_tok, uint32_t n_head, uint32_t head_dim) {
    if (!x_dev || !gate_dev || n_tok == 0 || n_head == 0 || head_dim == 0) return 0;
    uint32_t n = n_tok * n_head * head_dim;
    laguna_head_gate_kernel<<<(n + 255u) / 256u, 256>>>(
            (float *)x_dev, (const float *)gate_dev, n_tok, n_head, head_dim);
    return cuda_ok(cudaGetLastError(), "laguna head gate launch");
}


/* Batched variants for chain verify / chunked prefill: n_tok tokens
 * processed SEQUENTIALLY inside one launch (the recurrences are
 * per-token, but looping on-device removes 2*n_tok launch round-trips
 * per layer and keeps the working set in registers). */

/* conv+silu over n_tok tokens: thread owns a channel, window rides in
 * registers (k <= 8). x/out are [n_tok][n_chan]; state as in the
 * single-step kernel and left at the post-batch value. */
__global__ static void qwen35_conv_batch_kernel(
        float *out, const float *x, const float *kern, float *state,
        uint32_t n_chan, uint32_t k, uint32_t n_tok) {
    uint32_t c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= n_chan) return;
    const float *kc = kern + (uint64_t)c * k;
    float win[8];
    for (uint32_t t = 0; t + 1 < k; t++) {
        win[t] = state[(uint64_t)t * n_chan + c];
    }
    for (uint32_t tok = 0; tok < n_tok; tok++) {
        const float cur = x[(uint64_t)tok * n_chan + c];
        float acc = kc[k - 1] * cur;
        for (uint32_t t = 0; t + 1 < k; t++) acc += kc[t] * win[t];
        out[(uint64_t)tok * n_chan + c] = acc / (1.0f + expf(-acc));
        for (uint32_t t = 0; t + 2 < k; t++) win[t] = win[t + 1];
        win[k - 2] = cur;
    }
    for (uint32_t t = 0; t + 1 < k; t++) {
        state[(uint64_t)t * n_chan + c] = win[t];
    }
}

extern "C" int pulsar_qwen35_conv_batch(
        void *out_dev, const void *x_dev, const void *kern_dev,
        void *state_dev, uint32_t n_chan, uint32_t k, uint32_t n_tok) {
    if (!out_dev || !x_dev || !kern_dev || !state_dev || n_chan == 0 ||
        k < 2 || k > 8 || n_tok == 0) {
        return 0;
    }
    qwen35_conv_batch_kernel<<<(n_chan + 255u) / 256u, 256>>>(
            (float *)out_dev, (const float *)x_dev, (const float *)kern_dev,
            (float *)state_dev, n_chan, k, n_tok);
    return cuda_ok(cudaGetLastError(), "qwen35 conv batch launch");
}

/* GDN delta rule over n_tok tokens: thread j holds its state COLUMN
 * S[:,j] in registers (dim <= 128 for the register budget) across the
 * whole batch; k/q broadcast through shared memory per step. Layouts:
 * q/k [n_tok][H_k][dim], v [n_tok][H_v][dim], g/beta [n_tok][H_v],
 * out [n_tok][H_v][dim]. State read once, written once. */
__global__ static void qwen35_gdn_batch_kernel(
        float *out, float *state,
        const float *q, const float *k, const float *v,
        const float *g, const float *beta,
        uint32_t h_v, uint32_t h_k, uint32_t dim, uint32_t n_tok) {
    uint32_t h = blockIdx.x;
    uint32_t j = threadIdx.x;
    if (h >= h_v || j >= dim) return;
    const uint32_t kh = h % h_k; /* ggml_repeat tiles */
    float *Sg = state + (uint64_t)h * dim * dim;
    const float qscale = rsqrtf((float)dim);

    float S[128];
    for (uint32_t i = 0; i < dim; i++) S[i] = Sg[(uint64_t)i * dim + j];

    __shared__ float sq[128];
    __shared__ float sk[128];
    for (uint32_t tok = 0; tok < n_tok; tok++) {
        const float *qh = q + ((uint64_t)tok * h_k + kh) * dim;
        const float *kk = k + ((uint64_t)tok * h_k + kh) * dim;
        __syncthreads();
        for (uint32_t i = j; i < dim; i += blockDim.x) {
            sq[i] = qh[i] * qscale;
            sk[i] = kk[i];
        }
        __syncthreads();
        const float decay = expf(g[(uint64_t)tok * h_v + h]);
        float dot_k = 0.0f;
        for (uint32_t i = 0; i < dim; i++) {
            S[i] *= decay;
            dot_k += S[i] * sk[i];
        }
        const float d =
            (v[((uint64_t)tok * h_v + h) * dim + j] - dot_k) *
            beta[(uint64_t)tok * h_v + h];
        float o = 0.0f;
        for (uint32_t i = 0; i < dim; i++) {
            S[i] += sk[i] * d;
            o += S[i] * sq[i];
        }
        out[((uint64_t)tok * h_v + h) * dim + j] = o;
    }
    for (uint32_t i = 0; i < dim; i++) Sg[(uint64_t)i * dim + j] = S[i];
}

/* ---- chunked GDN prefill (PULSAR_GDN_CHUNK) ----------------------------
 * The sequential delta rule S_i = a_i (I - b_i k_i k_i^T) S_{i-1} +
 * b_i k_i v_i^T runs one token at a time with two block barriers per
 * token - h_v blocks x dim threads leaves the GPU >90% idle. The
 * chunked form (Yang et al., gated delta rule) hides the recurrence:
 * substituting the WRITTEN residual vt_i = b_i(v_i - a_i k_i^T S_{i-1})
 * turns the recurrence into plain gated linear attention over vt, and
 * vt itself solves a C x C unit-lower-triangular system:
 *
 *   (I + A) VT = R,  A[i][j] = b_i (k_i . k_j) exp(c_i - c_j)  (j < i)
 *   R[i]        = b_i (v_i - exp(c_i) (k_i^T S0))
 *   o_i         = exp(c_i) q^_i^T S0
 *                 + sum_{j<=i} (q^_i . k_j) exp(c_i - c_j) vt_j
 *   S_C         = exp(c_C) S0 + sum_i exp(c_C - c_i) k_i vt_i^T
 *
 * with c_i = cumsum(g) inside the chunk (log space) and q^ = q*rsqrt(d).
 * Every decay ratio that appears is exp of a sum of negatives, so
 * nothing overflows. Everything but the C-step substitution is a
 * barrier-free per-thread GEMM slice; the FLOP count is ~2.5x the
 * sequential kernel (the C^2 terms) but the loops are long and
 * ILP-friendly instead of 2-barriers-per-token latency chains.
 * Thread j keeps state COLUMN S[:,j] in registers, exactly like the
 * sequential kernel, so the two share the global state layout and a
 * tail < C tokens just runs the sequential kernel after. Selftest
 * compares the two kernels directly. */
/* 32 not 64: the C^2 terms (Gram, solve, output mix) are the chunked
 * form's FLOP tax; at C=64 they made it a net loss on CUDA cores.
 * Tensor-core GEMMs (v2) are what make big chunks pay. */
#define GDN_CHUNK 32u

static size_t gdn_chunk_smem_bytes(uint32_t dim) {
    return (size_t)(GDN_CHUNK * (dim + 1u)      /* k tile */
                    + GDN_CHUNK * (GDN_CHUNK + 1u)  /* A */
                    + GDN_CHUNK * (dim + 1u)    /* vt */
                    + 2u * GDN_CHUNK)           /* c (cum log g), beta */
                   * 4u;
}

__global__ static void qwen35_gdn_chunk_kernel(
        float *out, float *state,
        const float *q, const float *k, const float *v,
        const float *g, const float *beta,
        uint32_t h_v, uint32_t h_k, uint32_t dim, uint32_t n_tok) {
    const uint32_t C = GDN_CHUNK;
    const uint32_t h = blockIdx.x;
    const uint32_t j = threadIdx.x;
    if (h >= h_v || j >= dim) return;
    const uint32_t kh = h % h_k;
    float *Sg = state + (uint64_t)h * dim * dim;
    const float qscale = rsqrtf((float)dim);
    const uint32_t ldk = dim + 1u, lda = C + 1u;

    extern __shared__ float gdn_sm[];
    float *ksm = gdn_sm;                 /* [C][ldk] */
    float *A = ksm + C * ldk;            /* [C][lda] strictly lower */
    float *vt = A + C * lda;             /* [C][ldk] col j = thread j */
    float *cg = vt + C * ldk;            /* [C] cumulative log decay */
    float *bsm = cg + C;                 /* [C] beta */

    float S[128];
    for (uint32_t i = 0; i < dim; i++) S[i] = Sg[(uint64_t)i * dim + j];

    for (uint32_t base = 0; base + C <= n_tok; base += C) {
        /* stage K tile, beta, and the in-chunk cumulative log decay */
        for (uint32_t i = j; i < C * dim; i += blockDim.x) {
            const uint32_t r = i / dim, d = i % dim;
            ksm[r * ldk + d] = k[((uint64_t)(base + r) * h_k + kh) * dim + d];
        }
        if (j == 0) {
            float acc = 0.0f;
            for (uint32_t r = 0; r < C; r++) {
                acc += g[(uint64_t)(base + r) * h_v + h];
                cg[r] = acc;
                bsm[r] = beta[(uint64_t)(base + r) * h_v + h];
            }
        }
        __syncthreads();

        /* A[i][j'] = b_i (k_i . k_j') exp(c_i - c_j'), j' < i: 128
         * threads split the C*(C-1)/2 dots */
        for (uint32_t t = j; t < C * C; t += blockDim.x) {
            const uint32_t r = t / C, cix = t % C;
            if (cix < r) {
                float dot = 0.0f;
                for (uint32_t d = 0; d < dim; d++)
                    dot += ksm[r * ldk + d] * ksm[cix * ldk + d];
                A[r * lda + cix] = bsm[r] * dot * expf(cg[r] - cg[cix]);
            }
        }
        /* R straight into vt: thread j owns dim column j */
        for (uint32_t r = 0; r < C; r++) {
            float ks0 = 0.0f;
            for (uint32_t d = 0; d < dim; d++)
                ks0 += ksm[r * ldk + d] * S[d];
            vt[r * ldk + j] = bsm[r]
                    * (v[((uint64_t)(base + r) * h_v + h) * dim + j]
                       - expf(cg[r]) * ks0);
        }
        __syncthreads();

        /* forward substitution: vt_i -= sum_{j'<i} A[i][j'] vt_j' */
        for (uint32_t r = 1; r < C; r++) {
            float acc = 0.0f;
            for (uint32_t p = 0; p < r; p++)
                acc += A[r * lda + p] * vt[p * ldk + j];
            vt[r * ldk + j] -= acc;
            __syncthreads();
        }

        /* A is free after the solve: reuse it for the causal decayed
         * output Gram QG[r][p] = (q^_r . k_p) exp(c_r - c_p), p <= r,
         * computed ONCE across the block (the first cut recomputed every
         * q.k dot in all 128 threads - 3x slower than sequential) */
        __syncthreads();
        for (uint32_t t = j; t < C * C; t += blockDim.x) {
            const uint32_t r = t / C, p = t % C;
            if (p <= r) {
                const float *qr = q + ((uint64_t)(base + r) * h_k + kh) * dim;
                float qk = 0.0f;
                for (uint32_t d = 0; d < dim; d++)
                    qk += qr[d] * ksm[p * ldk + d];
                A[r * lda + p] = qk * qscale * expf(cg[r] - cg[p]);
            }
        }
        __syncthreads();
        for (uint32_t r = 0; r < C; r++) {
            const float *qr = q + ((uint64_t)(base + r) * h_k + kh) * dim;
            float qs0 = 0.0f;
            for (uint32_t d = 0; d < dim; d++)
                qs0 += qr[d] * qscale * S[d];
            float o = expf(cg[r]) * qs0;
            for (uint32_t p = 0; p <= r; p++)
                o += A[r * lda + p] * vt[p * ldk + j];
            out[((uint64_t)(base + r) * h_v + h) * dim + j] = o;
        }
        const float bC = expf(cg[C - 1u]);
        for (uint32_t d = 0; d < dim; d++) {
            float acc = bC * S[d];
            for (uint32_t r = 0; r < C; r++)
                acc += expf(cg[C - 1u] - cg[r]) * ksm[r * ldk + d] * vt[r * ldk + j];
            S[d] = acc;
        }
        __syncthreads();
    }
    for (uint32_t i = 0; i < dim; i++) Sg[(uint64_t)i * dim + j] = S[i];
}

static bool gdn_chunk_on() {
    const char *e = getenv("PULSAR_GDN_CHUNK");
    return e && atoi(e) == 1;
}

extern "C" int pulsar_qwen35_gdn_batch(
        void *out_dev, void *state_dev,
        const void *q_dev, const void *k_dev, const void *v_dev,
        const void *g_dev, const void *beta_dev,
        uint32_t h_v, uint32_t h_k, uint32_t dim, uint32_t n_tok) {
    if (!out_dev || !state_dev || !q_dev || !k_dev || !v_dev || !g_dev ||
        !beta_dev || h_v == 0 || h_k == 0 || h_v % h_k != 0 || dim == 0 ||
        dim > 128u || n_tok == 0) {
        return 0;
    }
    uint32_t done = 0;
    if (gdn_chunk_on() && n_tok >= GDN_CHUNK) {
        /* per-device smem attribute (>48KB when dim=128): same rule as
         * the gqa mma kernel - cudaFuncSetAttribute is per device */
        static signed char ck_ok[16];
        int dev = 0;
        (void)cudaGetDevice(&dev);
        const size_t sm = gdn_chunk_smem_bytes(dim);
        if (dev >= 0 && dev < 16) {
            if (ck_ok[dev] == 0) {
                int ok = 1;
                if (sm > 48u * 1024u)
                    ok = cuda_ok(cudaFuncSetAttribute(
                                 (const void *)qwen35_gdn_chunk_kernel,
                                 cudaFuncAttributeMaxDynamicSharedMemorySize,
                                 (int)gdn_chunk_smem_bytes(128u)),
                                 "gdn chunk smem");
                ck_ok[dev] = ok ? 1 : -1;
            }
            if (ck_ok[dev] == 1) {
                const uint32_t full = (n_tok / GDN_CHUNK) * GDN_CHUNK;
                qwen35_gdn_chunk_kernel<<<h_v, dim, sm>>>(
                        (float *)out_dev, (float *)state_dev,
                        (const float *)q_dev, (const float *)k_dev, (const float *)v_dev,
                        (const float *)g_dev, (const float *)beta_dev, h_v, h_k, dim, full);
                if (!cuda_ok(cudaGetLastError(), "qwen35 gdn chunk launch")) return 0;
                done = full;
            }
        }
    }
    if (done < n_tok) {
        /* tail (or the whole batch when chunking is off/small) runs the
         * sequential kernel from the carried state */
        const uint64_t row = dim;
        qwen35_gdn_batch_kernel<<<h_v, dim>>>(
                (float *)out_dev + (uint64_t)done * h_v * row,
                (float *)state_dev,
                (const float *)q_dev + (uint64_t)done * h_k * row,
                (const float *)k_dev + (uint64_t)done * h_k * row,
                (const float *)v_dev + (uint64_t)done * h_v * row,
                (const float *)g_dev + (uint64_t)done * h_v,
                (const float *)beta_dev + (uint64_t)done * h_v,
                h_v, h_k, dim, n_tok - done);
        return cuda_ok(cudaGetLastError(), "qwen35 gdn batch launch");
    }
    return 1;
}

/* Per-row scale: x[t][:] *= s[t] (batched shared-expert scalar gates). */
__global__ static void qwen35_row_scale_kernel(
        float *x, const float *s, uint32_t n_rows, uint32_t dim) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_rows * dim) return;
    x[idx] *= s[idx / dim];
}

extern "C" int pulsar_qwen35_row_scale(
        void *x_dev, const void *s_dev, uint32_t n_rows, uint32_t dim) {
    if (!x_dev || !s_dev || n_rows == 0 || dim == 0) return 0;
    uint32_t total = n_rows * dim;
    qwen35_row_scale_kernel<<<(total + 255u) / 256u, 256>>>(
            (float *)x_dev, (const float *)s_dev, n_rows, dim);
    return cuda_ok(cudaGetLastError(), "qwen35 row scale launch");
}

/* Non-causal draft attention (DFlash block diffusion): n_q query rows
 * attend over ALL n_kv rows. q [n_q][n_head][dim]; k/v contiguous rows
 * [n_kv][n_kv_head][dim]; GQA block mapping (flash_attn_ext semantics:
 * head h reads kv head h / (n_head/n_kv_head) - NOT the GDN tile map).
 * One block per (q row, head), online softmax over row chunks. */
__global__ static void qwen35_draft_attn_kernel(
        float *out, const float *q, const float *k, const float *v,
        uint32_t n_q, uint32_t n_kv, uint32_t n_head, uint32_t n_kv_head,
        uint32_t dim, float scale) {
    uint32_t t = blockIdx.x;
    uint32_t h = blockIdx.y;
    if (t >= n_q || h >= n_head) return;
    const uint32_t kvh = h / (n_head / n_kv_head);
    const float *qh = q + ((uint64_t)t * n_head + h) * dim;
    const uint32_t tid = threadIdx.x;
    __shared__ float sq[128];
    __shared__ float s_w[128];
    __shared__ float red[128];
    for (uint32_t d = tid; d < dim; d += 128u) sq[d] = qh[d];
    __syncthreads();
    float acc = 0.0f; /* dim <= 128: thread d owns output dim d */
    float den = 0.0f;
    float m = -INFINITY;
    const uint64_t kv_stride = (uint64_t)n_kv_head * dim;
    for (uint32_t base = 0; base < n_kv; base += 128u) {
        uint32_t chunk = n_kv - base < 128u ? n_kv - base : 128u;
        float s = -INFINITY;
        if (tid < chunk) {
            const float *krow = k + (uint64_t)(base + tid) * kv_stride + (uint64_t)kvh * dim;
            float dot = 0.0f;
            for (uint32_t d = 0; d < dim; d++) dot += sq[d] * krow[d];
            s = dot * scale;
        }
        red[tid] = s;
        __syncthreads();
        for (uint32_t st = 64u; st != 0; st >>= 1u) {
            if (tid < st) red[tid] = fmaxf(red[tid], red[tid + st]);
            __syncthreads();
        }
        float m_new = fmaxf(m, red[0]);
        float corr = expf(m - m_new);
        acc *= corr;
        den *= corr;
        if (tid < chunk) s_w[tid] = expf(s - m_new);
        __syncthreads();
        for (uint32_t r = 0; r < chunk; r++) {
            float wgt = s_w[r];
            den += wgt;
            if (tid < dim) {
                acc += wgt * v[(uint64_t)(base + r) * kv_stride + (uint64_t)kvh * dim + tid];
            }
        }
        m = m_new;
        __syncthreads();
    }
    if (tid < dim) {
        out[((uint64_t)t * n_head + h) * dim + tid] = acc / den;
    }
}

extern "C" int pulsar_qwen35_draft_attn(
        void *out_dev, const void *q_dev, const void *k_dev, const void *v_dev,
        uint32_t n_q, uint32_t n_kv, uint32_t n_head, uint32_t n_kv_head,
        uint32_t dim, float scale) {
    if (!out_dev || !q_dev || !k_dev || !v_dev || n_q == 0 || n_kv == 0 ||
        n_head == 0 || n_kv_head == 0 || n_head % n_kv_head != 0 ||
        dim == 0 || dim > 128u) {
        return 0;
    }
    dim3 grid(n_q, n_head, 1);
    qwen35_draft_attn_kernel<<<grid, 128>>>(
            (float *)out_dev, (const float *)q_dev, (const float *)k_dev,
            (const float *)v_dev, n_q, n_kv, n_head, n_kv_head, dim, scale);
    return cuda_ok(cudaGetLastError(), "qwen35 draft attn launch");
}

/* ---- DFlash2 (incoai) draft additions ---------------------------------- */

/* Grouped block-conv wrapping the draft's attention and MLP: for row p
 * of ONE draft block (position == row),
 *   out[p][c] = sum_t (base[side][t][c] + delta[p][side][t][g(c)]) * h[p-t][c]
 * for t < taps and p >= t. `delta` is the kernel-projection output row
 * laid out [p][2][taps][n_groups]; `side` 0 = prepare (before attn/mlp),
 * 1 = finish (after), selecting both the base plane and the delta half. */
__global__ static void qwen35_dflash2_conv_kernel(
        float *out, const float *h, const float *delta, const float *base,
        uint32_t n_tok, uint32_t n_embd, uint32_t taps, uint32_t group,
        uint32_t side) {
    const uint64_t idx = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    const uint64_t total = (uint64_t)n_tok * n_embd;
    if (idx >= total) return;
    const uint32_t p = idx / n_embd, c = idx % n_embd;
    const uint32_t n_groups = n_embd / group;
    const float *dp = delta + ((uint64_t)p * 2u + side) * taps * n_groups;
    const float *bp = base + (uint64_t)side * taps * n_embd;
    float acc = 0.0f;
    for (uint32_t t = 0; t < taps && t <= p; t++) {
        acc += (bp[t * n_embd + c] + dp[t * n_groups + c / group])
               * h[(uint64_t)(p - t) * n_embd + c];
    }
    out[idx] = acc;
}

extern "C" int pulsar_qwen35_dflash2_conv(
        void *out_dev, const void *h_dev, const void *delta_dev,
        const void *base_dev, uint32_t n_tok, uint32_t n_embd,
        uint32_t taps, uint32_t group, uint32_t side) {
    if (!out_dev || !h_dev || !delta_dev || !base_dev || n_tok == 0 ||
        n_embd == 0 || taps == 0 || taps > 8u || group == 0 ||
        n_embd % group != 0 || side > 1u || out_dev == h_dev) {
        return 0;
    }
    const uint64_t total = (uint64_t)n_tok * n_embd;
    qwen35_dflash2_conv_kernel<<<(uint32_t)((total + 255u) / 256u), 256>>>(
            (float *)out_dev, (const float *)h_dev, (const float *)delta_dev,
            (const float *)base_dev, n_tok, n_embd, taps, group, side);
    return cuda_ok(cudaGetLastError(), "qwen35 dflash2 conv launch");
}

/* One top-k step over the draft's logits rows: per row, argmax over the
 * vocab, record (id, value) at slot `it`, then mask the id to -inf so
 * the next call yields the next candidate. k calls = row-wise top-k
 * with no host round-trips. */
__global__ static void qwen35_dflash2_topk_step_kernel(
        float *logits, uint32_t vocab, uint32_t it, uint32_t k,
        int32_t *ids, float *vals) {
    const uint32_t row = blockIdx.x;
    float *lr = logits + (uint64_t)row * vocab;
    float best = -INFINITY;
    uint32_t best_i = 0;
    for (uint32_t i = threadIdx.x; i < vocab; i += blockDim.x) {
        const float v = lr[i];
        if (v > best) { best = v; best_i = i; }
    }
    __shared__ float sv[256];
    __shared__ uint32_t si[256];
    sv[threadIdx.x] = best;
    si[threadIdx.x] = best_i;
    __syncthreads();
    for (uint32_t stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            /* strict > with lower-index tie-break, matching argmax_rows */
            if (sv[threadIdx.x + stride] > sv[threadIdx.x] ||
                (sv[threadIdx.x + stride] == sv[threadIdx.x] &&
                 si[threadIdx.x + stride] < si[threadIdx.x])) {
                sv[threadIdx.x] = sv[threadIdx.x + stride];
                si[threadIdx.x] = si[threadIdx.x + stride];
            }
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        ids[(uint64_t)row * k + it] = (int32_t)si[0];
        vals[(uint64_t)row * k + it] = sv[0];
        lr[si[0]] = -INFINITY;
    }
}

extern "C" int pulsar_qwen35_dflash2_topk_step(
        void *logits_dev, uint32_t n_rows, uint32_t vocab, uint32_t it,
        uint32_t k, void *ids_dev, void *vals_dev) {
    if (!logits_dev || !ids_dev || !vals_dev || n_rows == 0 || vocab == 0 ||
        k == 0 || it >= k) {
        return 0;
    }
    qwen35_dflash2_topk_step_kernel<<<n_rows, 256>>>(
            (float *)logits_dev, vocab, it, k, (int32_t *)ids_dev,
            (float *)vals_dev);
    return cuda_ok(cudaGetLastError(), "qwen35 dflash2 topk launch");
}

/* Gather f16 codebook rows (by token id) into an f32 buffer:
 * out[i][r] = (float)book[ids[i]][r]. */
__global__ static void qwen35_dflash2_gather_kernel(
        float *out, const __half *book, const int32_t *ids,
        uint32_t n, uint32_t rank) {
    const uint64_t idx = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    const uint64_t total = (uint64_t)n * rank;
    if (idx >= total) return;
    const uint32_t i = idx / rank, r = idx % rank;
    out[idx] = __half2float(book[(uint64_t)ids[i] * rank + r]);
}

extern "C" int pulsar_qwen35_dflash2_gather(
        void *out_dev, const void *book_dev, const void *ids_dev,
        uint32_t n, uint32_t rank) {
    if (!out_dev || !book_dev || !ids_dev || n == 0 || rank == 0) return 0;
    const uint64_t total = (uint64_t)n * rank;
    qwen35_dflash2_gather_kernel<<<(uint32_t)((total + 255u) / 256u), 256>>>(
            (float *)out_dev, (const __half *)book_dev,
            (const int32_t *)ids_dev, n, rank);
    return cuda_ok(cudaGetLastError(), "qwen35 dflash2 gather launch");
}

/* Split the conv output rows [t][2*key+value] into contiguous q/k/v
 * batch buffers in ONE launch (replaces 3*t blocking d2d copies). */
__global__ static void qwen35_split_qkv_kernel(
        float *q, float *k, float *v, const float *x,
        uint32_t n_tok, uint32_t key_dim, uint32_t value_dim) {
    const uint32_t conv_dim = 2u * key_dim + value_dim;
    uint64_t idx = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    const uint64_t total = (uint64_t)n_tok * conv_dim;
    if (idx >= total) return;
    const uint32_t tok = idx / conv_dim;
    const uint32_t c = idx % conv_dim;
    const float val = x[idx];
    if (c < key_dim) {
        q[(uint64_t)tok * key_dim + c] = val;
    } else if (c < 2u * key_dim) {
        k[(uint64_t)tok * key_dim + (c - key_dim)] = val;
    } else {
        v[(uint64_t)tok * value_dim + (c - 2u * key_dim)] = val;
    }
}

extern "C" int pulsar_qwen35_split_qkv(
        void *q_dev, void *k_dev, void *v_dev, const void *x_dev,
        uint32_t n_tok, uint32_t key_dim, uint32_t value_dim) {
    if (!q_dev || !k_dev || !v_dev || !x_dev || n_tok == 0 ||
        key_dim == 0 || value_dim == 0) {
        return 0;
    }
    const uint64_t total = (uint64_t)n_tok * (2u * key_dim + value_dim);
    qwen35_split_qkv_kernel<<<(uint32_t)((total + 255u) / 256u), 256>>>(
            (float *)q_dev, (float *)k_dev, (float *)v_dev,
            (const float *)x_dev, n_tok, key_dim, value_dim);
    return cuda_ok(cudaGetLastError(), "qwen35 split qkv launch");
}

/* Ring scatter/gather for the DFlash feature ring (slot = pos % cap):
 * scatter writes t source rows into ring[(pos+i)%cap][off..off+row);
 * gather reads n rows from ring[(start+i)%cap] into a contiguous dst.
 * One launch each (replaces per-row blocking d2d copies). */
__global__ static void qwen35_ring_scatter_kernel(
        float *ring, const float *src, uint32_t pos, uint32_t cap,
        uint32_t n_rows, uint32_t row_elems, uint32_t ring_stride,
        uint32_t ring_off) {
    uint64_t idx = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    const uint64_t total = (uint64_t)n_rows * row_elems;
    if (idx >= total) return;
    const uint32_t i = idx / row_elems;
    const uint32_t d = idx % row_elems;
    const uint32_t slot = (pos + i) % cap;
    ring[(uint64_t)slot * ring_stride + ring_off + d] = src[idx];
}

extern "C" int pulsar_qwen35_ring_scatter(
        void *ring_dev, const void *src_dev, uint32_t pos, uint32_t cap,
        uint32_t n_rows, uint32_t row_elems, uint32_t ring_stride,
        uint32_t ring_off) {
    if (!ring_dev || !src_dev || cap == 0 || n_rows == 0 || row_elems == 0) {
        return 0;
    }
    const uint64_t total = (uint64_t)n_rows * row_elems;
    qwen35_ring_scatter_kernel<<<(uint32_t)((total + 255u) / 256u), 256>>>(
            (float *)ring_dev, (const float *)src_dev, pos, cap, n_rows,
            row_elems, ring_stride, ring_off);
    return cuda_ok(cudaGetLastError(), "qwen35 ring scatter launch");
}

__global__ static void qwen35_ring_gather_kernel(
        float *dst, const float *ring, uint32_t start, uint32_t cap,
        uint32_t n_rows, uint32_t row_elems) {
    uint64_t idx = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    const uint64_t total = (uint64_t)n_rows * row_elems;
    if (idx >= total) return;
    const uint32_t i = idx / row_elems;
    const uint32_t d = idx % row_elems;
    const uint32_t slot = (start + i) % cap;
    dst[idx] = ring[(uint64_t)slot * row_elems + d];
}

extern "C" int pulsar_qwen35_ring_gather(
        void *dst_dev, const void *ring_dev, uint32_t start, uint32_t cap,
        uint32_t n_rows, uint32_t row_elems) {
    if (!dst_dev || !ring_dev || cap == 0 || n_rows == 0 || row_elems == 0) {
        return 0;
    }
    const uint64_t total = (uint64_t)n_rows * row_elems;
    qwen35_ring_gather_kernel<<<(uint32_t)((total + 255u) / 256u), 256>>>(
            (float *)dst_dev, (const float *)ring_dev, start, cap, n_rows,
            row_elems);
    return cuda_ok(cudaGetLastError(), "qwen35 ring gather launch");
}

/* GDN mixing coefficients on-device (replaces two host readbacks per
 * GDN layer per forward): g = a * softplus(alpha + dt_bias),
 * beta = sigmoid(beta_raw). In place over [n_tok][n_head]. */
__global__ static void qwen35_gdn_coeffs_kernel(
        float *g_alpha, float *beta, const float *a, const float *dt,
        uint32_t n_tok, uint32_t n_head) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_tok * n_head) return;
    const uint32_t h = idx % n_head;
    const float x = g_alpha[idx] + dt[h];
    const float sp = x > 20.0f ? x : (x < -20.0f ? expf(x) : log1pf(expf(x)));
    g_alpha[idx] = a[h] * sp;
    beta[idx] = 1.0f / (1.0f + expf(-beta[idx]));
}

extern "C" int pulsar_qwen35_gdn_coeffs(
        void *g_alpha_dev, void *beta_dev, const void *a_dev,
        const void *dt_dev, uint32_t n_tok, uint32_t n_head) {
    if (!g_alpha_dev || !beta_dev || !a_dev || !dt_dev || n_tok == 0 ||
        n_head == 0) {
        return 0;
    }
    const uint32_t total = n_tok * n_head;
    qwen35_gdn_coeffs_kernel<<<(total + 255u) / 256u, 256>>>(
            (float *)g_alpha_dev, (float *)beta_dev, (const float *)a_dev,
            (const float *)dt_dev, n_tok, n_head);
    return cuda_ok(cudaGetLastError(), "qwen35 gdn coeffs launch");
}

/* x[row][:] *= sigmoid(s[row]) (shared-expert scalar gates without the
 * per-forward host readback). */
__global__ static void qwen35_row_sigmoid_scale_kernel(
        float *x, const float *sv, uint32_t n_rows, uint32_t dim) {
    uint64_t idx = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= (uint64_t)n_rows * dim) return;
    const float g = 1.0f / (1.0f + expf(-sv[idx / dim]));
    x[idx] *= g;
}

extern "C" int pulsar_qwen35_row_sigmoid_scale(
        void *x_dev, const void *s_dev, uint32_t n_rows, uint32_t dim) {
    if (!x_dev || !s_dev || n_rows == 0 || dim == 0) return 0;
    const uint64_t total = (uint64_t)n_rows * dim;
    qwen35_row_sigmoid_scale_kernel<<<(uint32_t)((total + 255u) / 256u), 256>>>(
            (float *)x_dev, (const float *)s_dev, n_rows, dim);
    return cuda_ok(cudaGetLastError(), "qwen35 row sigmoid scale launch");
}

/* NEOX-paired YaRN rope (DFlash draft: trained with rope_scaling yarn
 * factor 64 / orig 4096 on theta 10M - plain rope smears its positions
 * 64x). Pair (i, i+half); cos/sin from the mla yarn helpers (ggml
 * semantics: pass attn_factor 1.0 and the kernel-internal
 * 1 + 0.1 ln(1/freq_scale) supplies the HF mscale). */
__global__ static void qwen35_rope_yarn_kernel(
        float *x, uint32_t n_tok, uint32_t n_head, uint32_t head_dim,
        uint32_t pos0, float freq_base, float freq_scale,
        float ext_factor, float attn_factor, float beta_fast,
        float beta_slow, uint32_t n_ctx_orig) {
    const uint32_t half = head_dim / 2u;
    uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t pairs = n_tok * n_head * half;
    if (gid >= pairs) return;
    const uint32_t p = gid % half;
    const uint32_t tmp = gid / half;
    const uint32_t h = tmp % n_head;
    const uint32_t tok = tmp / n_head;
    float c0 = 0.0f, c1 = 0.0f;
    if (ext_factor != 0.0f) {
        mla_rope_corr_dims(head_dim, n_ctx_orig, freq_base,
                           beta_fast, beta_slow, &c0, &c1);
    }
    const float theta_extrap = (float)(pos0 + tok) *
        powf(freq_base, -2.0f * (float)p / (float)head_dim);
    float cs, sn;
    mla_rope_yarn(theta_extrap, freq_scale, c0, c1, (int)(2u * p),
                  ext_factor, attn_factor, &cs, &sn);
    float *row = x + ((uint64_t)tok * n_head + h) * head_dim;
    const float x0 = row[p];
    const float x1 = row[p + half];
    row[p] = x0 * cs - x1 * sn;
    row[p + half] = x0 * sn + x1 * cs;
}

/* NEOX yarn rope over a PARTIAL rotation width: pairs (p, p+rot/2) live
 * inside the first rot_dim lanes of each head, lanes >= rot_dim pass
 * through untouched (llama.cpp partial-rope convention). Laguna rotates
 * 64 of 128 on its full-attention layers and all 128 on sliding ones. */
__global__ static void rope_yarn_partial_kernel(
        float *x, uint32_t n_tok, uint32_t n_head, uint32_t head_dim,
        uint32_t rot_dim, uint32_t pos0, float freq_base, float freq_scale,
        float ext_factor, float attn_factor, float beta_fast,
        float beta_slow, uint32_t n_ctx_orig) {
    const uint32_t half = rot_dim / 2u;
    uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t pairs = n_tok * n_head * half;
    if (gid >= pairs) return;
    const uint32_t p = gid % half;
    const uint32_t tmp = gid / half;
    const uint32_t h = tmp % n_head;
    const uint32_t tok = tmp / n_head;
    float c0 = 0.0f, c1 = 0.0f;
    if (ext_factor != 0.0f) {
        mla_rope_corr_dims(rot_dim, n_ctx_orig, freq_base,
                           beta_fast, beta_slow, &c0, &c1);
    }
    const float theta_extrap = (float)(pos0 + tok) *
        powf(freq_base, -2.0f * (float)p / (float)rot_dim);
    float cs, sn;
    mla_rope_yarn(theta_extrap, freq_scale, c0, c1, (int)(2u * p),
                  ext_factor, attn_factor, &cs, &sn);
    /* row stride is the FULL head_dim; only the first rot_dim rotate */
    float *row = x + ((uint64_t)tok * n_head + h) * head_dim;
    const float x0 = row[p];
    const float x1 = row[p + half];
    row[p] = x0 * cs - x1 * sn;
    row[p + half] = x0 * sn + x1 * cs;
}

extern "C" int pulsar_rope_yarn_partial(
        void *x_dev, uint32_t n_tok, uint32_t n_head, uint32_t head_dim,
        uint32_t rot_dim, uint32_t pos0, float freq_base, float freq_scale,
        float ext_factor, float attn_factor, float beta_fast,
        float beta_slow, uint32_t n_ctx_orig) {
    if (!x_dev || n_tok == 0 || n_head == 0 || rot_dim < 2u || rot_dim > head_dim) return 0;
    uint32_t pairs = n_tok * n_head * (rot_dim / 2u);
    rope_yarn_partial_kernel<<<(pairs + 255u) / 256u, 256>>>(
            (float *)x_dev, n_tok, n_head, head_dim, rot_dim, pos0,
            freq_base, freq_scale, ext_factor, attn_factor,
            beta_fast, beta_slow, n_ctx_orig);
    return cuda_ok(cudaGetLastError(), "rope yarn partial launch");
}

extern "C" int pulsar_qwen35_rope_yarn(
        void *x_dev, uint32_t n_tok, uint32_t n_head, uint32_t head_dim,
        uint32_t pos0, float freq_base, float freq_scale, float ext_factor,
        float attn_factor, float beta_fast, float beta_slow,
        uint32_t n_ctx_orig) {
    if (!x_dev || n_tok == 0 || n_head == 0 || head_dim == 0 ||
        (head_dim & 1u) != 0) {
        return 0;
    }
    const uint32_t pairs = n_tok * n_head * (head_dim / 2u);
    qwen35_rope_yarn_kernel<<<(pairs + 255u) / 256u, 256>>>(
            (float *)x_dev, n_tok, n_head, head_dim, pos0, freq_base,
            freq_scale, ext_factor, attn_factor, beta_fast, beta_slow,
            n_ctx_orig);
    return cuda_ok(cudaGetLastError(), "qwen35 rope yarn launch");
}

/* ---- selftest ----------------------------------------------------------- */

extern "C" int pulsar_qwen35_selftest(void) {
    int ok = 1;
    const uint32_t DIM = 128, HV = 8, HK = 4, NCH = 512, K = 4;

    /* conv step vs host reference, 3 sequential tokens */
    {
        float *kern = (float *)malloc((size_t)NCH * K * 4);
        float *state = (float *)calloc((size_t)(K - 1) * NCH, 4);
        float *state_ref = (float *)calloc((size_t)(K - 1) * NCH, 4);
        for (uint32_t i = 0; i < NCH * K; i++) kern[i] = gqa_test_randf();
        void *dkern, *dstate, *dx, *dout;
        cudaMalloc(&dkern, (size_t)NCH * K * 4);
        cudaMalloc(&dstate, (size_t)(K - 1) * NCH * 4);
        cudaMalloc(&dx, NCH * 4);
        cudaMalloc(&dout, NCH * 4);
        cudaMemcpy(dkern, kern, (size_t)NCH * K * 4, cudaMemcpyHostToDevice);
        cudaMemcpy(dstate, state, (size_t)(K - 1) * NCH * 4, cudaMemcpyHostToDevice);
        float *x = (float *)malloc(NCH * 4);
        float *got = (float *)malloc(NCH * 4);
        for (int step = 0; step < 3 && ok; step++) {
            for (uint32_t c = 0; c < NCH; c++) x[c] = gqa_test_randf();
            cudaMemcpy(dx, x, NCH * 4, cudaMemcpyHostToDevice);
            ok &= pulsar_qwen35_conv_step(dout, dx, dkern, dstate, NCH, K);
            cudaMemcpy(got, dout, NCH * 4, cudaMemcpyDeviceToHost);
            for (uint32_t c = 0; c < NCH && ok; c++) {
                float acc = kern[c * K + K - 1] * x[c];
                for (uint32_t t = 0; t + 1 < K; t++) {
                    acc += kern[c * K + t] * state_ref[t * NCH + c];
                }
                float want = acc / (1.0f + expf(-acc));
                if (fabsf(want - got[c]) > 1.0e-5f) {
                    fprintf(stderr, "qwen35 selftest: conv mismatch step %d ch %u\n", step, c);
                    ok = 0;
                }
            }
            for (uint32_t t = 0; t + 2 < K; t++) {
                memcpy(state_ref + t * NCH, state_ref + (t + 1) * NCH, NCH * 4);
            }
            memcpy(state_ref + (K - 2) * NCH, x, NCH * 4);
        }
        cudaFree(dkern); cudaFree(dstate); cudaFree(dx); cudaFree(dout);
        free(kern); free(state); free(state_ref); free(x); free(got);
    }

    /* gdn step vs host reference, 3 sequential tokens */
    {
        const uint64_t sbytes = (uint64_t)HV * DIM * DIM * 4;
        float *S_ref = (float *)calloc((size_t)HV * DIM * DIM, 4);
        float *q = (float *)malloc((size_t)HK * DIM * 4);
        float *k = (float *)malloc((size_t)HK * DIM * 4);
        float *v = (float *)malloc((size_t)HV * DIM * 4);
        float g[HV], beta[HV];
        void *dS, *dq, *dk, *dv, *dg, *dbeta, *dout;
        cudaMalloc(&dS, sbytes);
        cudaMemset(dS, 0, sbytes);
        cudaMalloc(&dq, (size_t)HK * DIM * 4);
        cudaMalloc(&dk, (size_t)HK * DIM * 4);
        cudaMalloc(&dv, (size_t)HV * DIM * 4);
        cudaMalloc(&dg, HV * 4);
        cudaMalloc(&dbeta, HV * 4);
        cudaMalloc(&dout, (size_t)HV * DIM * 4);
        float *got = (float *)malloc((size_t)HV * DIM * 4);
        for (int step = 0; step < 3 && ok; step++) {
            for (uint32_t i = 0; i < HK * DIM; i++) { q[i] = gqa_test_randf(); k[i] = gqa_test_randf(); }
            for (uint32_t i = 0; i < HV * DIM; i++) v[i] = gqa_test_randf();
            for (uint32_t h = 0; h < HV; h++) {
                g[h] = -fabsf(gqa_test_randf());          /* log-decay < 0 */
                beta[h] = 0.5f + 0.4f * gqa_test_randf(); /* ~(0,1) */
            }
            cudaMemcpy(dq, q, (size_t)HK * DIM * 4, cudaMemcpyHostToDevice);
            cudaMemcpy(dk, k, (size_t)HK * DIM * 4, cudaMemcpyHostToDevice);
            cudaMemcpy(dv, v, (size_t)HV * DIM * 4, cudaMemcpyHostToDevice);
            cudaMemcpy(dg, g, HV * 4, cudaMemcpyHostToDevice);
            cudaMemcpy(dbeta, beta, HV * 4, cudaMemcpyHostToDevice);
            ok &= pulsar_qwen35_gdn_step(dout, dS, dq, dk, dv, dg, dbeta, HV, HK, DIM);
            cudaMemcpy(got, dout, (size_t)HV * DIM * 4, cudaMemcpyDeviceToHost);
            for (uint32_t h = 0; h < HV && ok; h++) {
                const float *qh = q + (h % HK) * DIM;
                const float *kh = k + (h % HK) * DIM;
                float *S = S_ref + (uint64_t)h * DIM * DIM;
                const float decay = expf(g[h]);
                const float qs = 1.0f / sqrtf((float)DIM);
                for (uint32_t i = 0; i < DIM * DIM; i++) S[i] *= decay;
                for (uint32_t j = 0; j < DIM; j++) {
                    float sk = 0.0f;
                    for (uint32_t i = 0; i < DIM; i++) sk += S[i * DIM + j] * kh[i];
                    float d = (v[h * DIM + j] - sk) * beta[h];
                    float o = 0.0f;
                    for (uint32_t i = 0; i < DIM; i++) {
                        S[i * DIM + j] += kh[i] * d;
                        o += S[i * DIM + j] * qh[i] * qs;
                    }
                    float diff = fabsf(o - got[h * DIM + j]);
                    if (diff > 5.0e-4f * fmaxf(1.0f, fabsf(o))) {
                        fprintf(stderr,
                                "qwen35 selftest: gdn mismatch step %d h %u j %u: %g vs %g\n",
                                step, h, j, o, got[h * DIM + j]);
                        ok = 0;
                        break;
                    }
                }
            }
        }
        cudaFree(dS); cudaFree(dq); cudaFree(dk); cudaFree(dv);
        cudaFree(dg); cudaFree(dbeta); cudaFree(dout);
        free(S_ref); free(q); free(k); free(v); free(got);
    }

    /* l2 norm + split gate + sigmoid gate quick checks */
    {
        const uint32_t R = 4, D = 128;
        float *x = (float *)malloc((size_t)R * D * 4);
        for (uint32_t i = 0; i < R * D; i++) x[i] = gqa_test_randf() * 3.0f;
        void *dx;
        cudaMalloc(&dx, (size_t)R * D * 4);
        cudaMemcpy(dx, x, (size_t)R * D * 4, cudaMemcpyHostToDevice);
        ok &= pulsar_qwen35_l2_norm(dx, R, D, 1.0e-6f);
        float *got = (float *)malloc((size_t)R * D * 4);
        cudaMemcpy(got, dx, (size_t)R * D * 4, cudaMemcpyDeviceToHost);
        for (uint32_t r = 0; r < R && ok; r++) {
            float ss = 0.0f;
            for (uint32_t i = 0; i < D; i++) ss += x[r * D + i] * x[r * D + i];
            float sc = 1.0f / sqrtf(ss + 1.0e-6f);
            for (uint32_t i = 0; i < D; i++) {
                if (fabsf(x[r * D + i] * sc - got[r * D + i]) > 1.0e-5f) {
                    fprintf(stderr, "qwen35 selftest: l2 norm mismatch\n");
                    ok = 0;
                    break;
                }
            }
        }
        cudaFree(dx); free(x); free(got);

        const uint32_t H = 4, HD = 64;
        float *fused = (float *)malloc((size_t)H * 2 * HD * 4);
        for (uint32_t i = 0; i < H * 2 * HD; i++) fused[i] = (float)i;
        void *dfused, *dq2, *dgate;
        cudaMalloc(&dfused, (size_t)H * 2 * HD * 4);
        cudaMalloc(&dq2, (size_t)H * HD * 4);
        cudaMalloc(&dgate, (size_t)H * HD * 4);
        cudaMemcpy(dfused, fused, (size_t)H * 2 * HD * 4, cudaMemcpyHostToDevice);
        ok &= pulsar_qwen35_split_gate(dq2, dgate, dfused, H, HD);
        float *gq = (float *)malloc((size_t)H * HD * 4);
        float *gg = (float *)malloc((size_t)H * HD * 4);
        cudaMemcpy(gq, dq2, (size_t)H * HD * 4, cudaMemcpyDeviceToHost);
        cudaMemcpy(gg, dgate, (size_t)H * HD * 4, cudaMemcpyDeviceToHost);
        for (uint32_t h = 0; h < H && ok; h++) {
            for (uint32_t d = 0; d < HD; d++) {
                if (gq[h * HD + d] != fused[h * 2 * HD + d] ||
                    gg[h * HD + d] != fused[h * 2 * HD + HD + d]) {
                    fprintf(stderr, "qwen35 selftest: split gate mismatch\n");
                    ok = 0;
                    break;
                }
            }
        }
        ok &= pulsar_qwen35_sigmoid_gate(dq2, dgate, H * HD);
        float *gs = (float *)malloc((size_t)H * HD * 4);
        cudaMemcpy(gs, dq2, (size_t)H * HD * 4, cudaMemcpyDeviceToHost);
        for (uint32_t i = 0; i < H * HD && ok; i++) {
            float want = gq[i] / (1.0f + expf(-gg[i]));
            if (fabsf(want - gs[i]) > 1.0e-4f * fmaxf(1.0f, fabsf(want))) {
                fprintf(stderr, "qwen35 selftest: sigmoid gate mismatch\n");
                ok = 0;
            }
        }
        cudaFree(dfused); cudaFree(dq2); cudaFree(dgate);
        free(fused); free(gq); free(gg); free(gs);
    }

    /* batch kernels vs the single-step kernels (sequential equivalence) */
    {
        const uint32_t T = 5;
        /* conv */
        float *kern = (float *)malloc((size_t)NCH * K * 4);
        float *xs = (float *)malloc((size_t)T * NCH * 4);
        for (uint32_t i = 0; i < NCH * K; i++) kern[i] = gqa_test_randf();
        for (uint32_t i = 0; i < T * NCH; i++) xs[i] = gqa_test_randf();
        void *dkern, *dstate1, *dstate2, *dx, *dout1, *dout2;
        cudaMalloc(&dkern, (size_t)NCH * K * 4);
        cudaMalloc(&dstate1, (size_t)(K - 1) * NCH * 4);
        cudaMalloc(&dstate2, (size_t)(K - 1) * NCH * 4);
        cudaMalloc(&dx, (size_t)T * NCH * 4);
        cudaMalloc(&dout1, (size_t)T * NCH * 4);
        cudaMalloc(&dout2, (size_t)T * NCH * 4);
        cudaMemcpy(dkern, kern, (size_t)NCH * K * 4, cudaMemcpyHostToDevice);
        cudaMemset(dstate1, 0, (size_t)(K - 1) * NCH * 4);
        cudaMemset(dstate2, 0, (size_t)(K - 1) * NCH * 4);
        cudaMemcpy(dx, xs, (size_t)T * NCH * 4, cudaMemcpyHostToDevice);
        for (uint32_t t = 0; t < T; t++) {
            ok &= pulsar_qwen35_conv_step(
                    (float *)dout1 + (size_t)t * NCH,
                    (const float *)dx + (size_t)t * NCH, dkern, dstate1, NCH, K);
        }
        ok &= pulsar_qwen35_conv_batch(dout2, dx, dkern, dstate2, NCH, K, T);
        float *o1 = (float *)malloc((size_t)T * NCH * 4);
        float *o2 = (float *)malloc((size_t)T * NCH * 4);
        float *st1 = (float *)malloc((size_t)(K - 1) * NCH * 4);
        float *st2 = (float *)malloc((size_t)(K - 1) * NCH * 4);
        cudaMemcpy(o1, dout1, (size_t)T * NCH * 4, cudaMemcpyDeviceToHost);
        cudaMemcpy(o2, dout2, (size_t)T * NCH * 4, cudaMemcpyDeviceToHost);
        cudaMemcpy(st1, dstate1, (size_t)(K - 1) * NCH * 4, cudaMemcpyDeviceToHost);
        cudaMemcpy(st2, dstate2, (size_t)(K - 1) * NCH * 4, cudaMemcpyDeviceToHost);
        for (uint32_t i = 0; i < T * NCH && ok; i++) {
            if (o1[i] != o2[i]) { fprintf(stderr, "qwen35 selftest: conv batch out mismatch\n"); ok = 0; }
        }
        for (uint32_t i = 0; i < (K - 1) * NCH && ok; i++) {
            if (st1[i] != st2[i]) { fprintf(stderr, "qwen35 selftest: conv batch state mismatch\n"); ok = 0; }
        }
        cudaFree(dkern); cudaFree(dstate1); cudaFree(dstate2);
        cudaFree(dx); cudaFree(dout1); cudaFree(dout2);
        free(kern); free(xs); free(o1); free(o2); free(st1); free(st2);

        /* gdn batch vs single steps */
        const uint64_t sbytes = (uint64_t)HV * DIM * DIM * 4;
        float *q = (float *)malloc((size_t)T * HK * DIM * 4);
        float *kv = (float *)malloc((size_t)T * HK * DIM * 4);
        float *vv = (float *)malloc((size_t)T * HV * DIM * 4);
        float *gg = (float *)malloc((size_t)T * HV * 4);
        float *bb = (float *)malloc((size_t)T * HV * 4);
        for (uint32_t i = 0; i < T * HK * DIM; i++) { q[i] = gqa_test_randf(); kv[i] = gqa_test_randf(); }
        for (uint32_t i = 0; i < T * HV * DIM; i++) vv[i] = gqa_test_randf();
        for (uint32_t i = 0; i < T * HV; i++) { gg[i] = -fabsf(gqa_test_randf()); bb[i] = 0.5f + 0.4f * gqa_test_randf(); }
        void *dS1, *dS2, *dq, *dk, *dv, *dg, *db, *do1, *do2;
        cudaMalloc(&dS1, sbytes); cudaMemset(dS1, 0, sbytes);
        cudaMalloc(&dS2, sbytes); cudaMemset(dS2, 0, sbytes);
        cudaMalloc(&dq, (size_t)T * HK * DIM * 4);
        cudaMalloc(&dk, (size_t)T * HK * DIM * 4);
        cudaMalloc(&dv, (size_t)T * HV * DIM * 4);
        cudaMalloc(&dg, (size_t)T * HV * 4);
        cudaMalloc(&db, (size_t)T * HV * 4);
        cudaMalloc(&do1, (size_t)T * HV * DIM * 4);
        cudaMalloc(&do2, (size_t)T * HV * DIM * 4);
        cudaMemcpy(dq, q, (size_t)T * HK * DIM * 4, cudaMemcpyHostToDevice);
        cudaMemcpy(dk, kv, (size_t)T * HK * DIM * 4, cudaMemcpyHostToDevice);
        cudaMemcpy(dv, vv, (size_t)T * HV * DIM * 4, cudaMemcpyHostToDevice);
        cudaMemcpy(dg, gg, (size_t)T * HV * 4, cudaMemcpyHostToDevice);
        cudaMemcpy(db, bb, (size_t)T * HV * 4, cudaMemcpyHostToDevice);
        for (uint32_t t = 0; t < T; t++) {
            ok &= pulsar_qwen35_gdn_step(
                    (float *)do1 + (size_t)t * HV * DIM, dS1,
                    (const float *)dq + (size_t)t * HK * DIM,
                    (const float *)dk + (size_t)t * HK * DIM,
                    (const float *)dv + (size_t)t * HV * DIM,
                    (const float *)dg + (size_t)t * HV,
                    (const float *)db + (size_t)t * HV, HV, HK, DIM);
        }
        ok &= pulsar_qwen35_gdn_batch(do2, dS2, dq, dk, dv, dg, db, HV, HK, DIM, T);
        float *go1 = (float *)malloc((size_t)T * HV * DIM * 4);
        float *go2 = (float *)malloc((size_t)T * HV * DIM * 4);
        float *gs1 = (float *)malloc(sbytes);
        float *gs2 = (float *)malloc(sbytes);
        cudaMemcpy(go1, do1, (size_t)T * HV * DIM * 4, cudaMemcpyDeviceToHost);
        cudaMemcpy(go2, do2, (size_t)T * HV * DIM * 4, cudaMemcpyDeviceToHost);
        cudaMemcpy(gs1, dS1, sbytes, cudaMemcpyDeviceToHost);
        cudaMemcpy(gs2, dS2, sbytes, cudaMemcpyDeviceToHost);
        for (uint32_t i = 0; i < T * HV * DIM && ok; i++) {
            float diff = fabsf(go1[i] - go2[i]);
            if (diff > 1.0e-5f * fmaxf(1.0f, fabsf(go1[i]))) {
                fprintf(stderr, "qwen35 selftest: gdn batch out mismatch at %u\n", i);
                ok = 0;
            }
        }
        for (uint64_t i = 0; i < (uint64_t)HV * DIM * DIM && ok; i++) {
            float diff = fabsf(gs1[i] - gs2[i]);
            if (diff > 1.0e-5f * fmaxf(1.0f, fabsf(gs1[i]))) {
                fprintf(stderr, "qwen35 selftest: gdn batch state mismatch\n");
                ok = 0;
            }
        }
        cudaFree(dS1); cudaFree(dS2); cudaFree(dq); cudaFree(dk); cudaFree(dv);
        cudaFree(dg); cudaFree(db); cudaFree(do1); cudaFree(do2);
        free(q); free(kv); free(vv); free(gg); free(bb);
        free(go1); free(go2); free(gs1); free(gs2);
    }

    /* draft attention vs host reference (non-causal GQA) */
    {
        const uint32_t NQ = 3, NKV = 70, NH = 8, NKVH = 2, D = 128;
        const float scale = 1.0f / sqrtf((float)D);
        float *q = (float *)malloc((size_t)NQ * NH * D * 4);
        float *k = (float *)malloc((size_t)NKV * NKVH * D * 4);
        float *v = (float *)malloc((size_t)NKV * NKVH * D * 4);
        for (uint32_t i = 0; i < NQ * NH * D; i++) q[i] = gqa_test_randf();
        for (uint32_t i = 0; i < NKV * NKVH * D; i++) { k[i] = gqa_test_randf(); v[i] = gqa_test_randf(); }
        void *dq, *dk, *dv, *dout;
        cudaMalloc(&dq, (size_t)NQ * NH * D * 4);
        cudaMalloc(&dk, (size_t)NKV * NKVH * D * 4);
        cudaMalloc(&dv, (size_t)NKV * NKVH * D * 4);
        cudaMalloc(&dout, (size_t)NQ * NH * D * 4);
        cudaMemcpy(dq, q, (size_t)NQ * NH * D * 4, cudaMemcpyHostToDevice);
        cudaMemcpy(dk, k, (size_t)NKV * NKVH * D * 4, cudaMemcpyHostToDevice);
        cudaMemcpy(dv, v, (size_t)NKV * NKVH * D * 4, cudaMemcpyHostToDevice);
        ok &= pulsar_qwen35_draft_attn(dout, dq, dk, dv, NQ, NKV, NH, NKVH, D, scale);
        float *got = (float *)malloc((size_t)NQ * NH * D * 4);
        cudaMemcpy(got, dout, (size_t)NQ * NH * D * 4, cudaMemcpyDeviceToHost);
        for (uint32_t t = 0; t < NQ && ok; t++) {
            for (uint32_t h = 0; h < NH && ok; h++) {
                const uint32_t kvh = h / (NH / NKVH);
                const float *qh = q + ((size_t)t * NH + h) * D;
                float *sc = (float *)malloc(NKV * 4);
                float mx = -1.0e30f;
                for (uint32_t r = 0; r < NKV; r++) {
                    const float *kr = k + ((size_t)r * NKVH + kvh) * D;
                    float dot = 0.0f;
                    for (uint32_t d2 = 0; d2 < D; d2++) dot += qh[d2] * kr[d2];
                    sc[r] = dot * scale;
                    if (sc[r] > mx) mx = sc[r];
                }
                float den = 0.0f;
                for (uint32_t r = 0; r < NKV; r++) { sc[r] = expf(sc[r] - mx); den += sc[r]; }
                for (uint32_t d2 = 0; d2 < D && ok; d2++) {
                    float o = 0.0f;
                    for (uint32_t r = 0; r < NKV; r++) {
                        o += sc[r] * v[((size_t)r * NKVH + kvh) * D + d2];
                    }
                    o /= den;
                    float diff = fabsf(o - got[((size_t)t * NH + h) * D + d2]);
                    if (diff > 2.0e-4f * fmaxf(1.0f, fabsf(o))) {
                        fprintf(stderr, "qwen35 selftest: draft attn mismatch\n");
                        ok = 0;
                    }
                }
                free(sc);
            }
        }
        cudaFree(dq); cudaFree(dk); cudaFree(dv); cudaFree(dout);
        free(q); free(k); free(v); free(got);
    }

    /* chunked GDN vs the sequential kernel on identical inputs: 200
     * tokens = 3 full chunks + an 8-token sequential tail through the
     * dispatch, prod shape (HV heads, HK kv heads, dim 128). The
     * sequential kernel IS the reference; the chunked form reorders
     * float ops heavily (C^2 terms, exp ratios), so the tolerance is
     * looser than exact phases. */
    if (ok) {
        const uint32_t NT = 200;
        const uint64_t qk_e = (uint64_t)NT * HK * DIM;
        const uint64_t v_e = (uint64_t)NT * HV * DIM;
        const uint64_t gb_e = (uint64_t)NT * HV;
        const uint64_t st_e = (uint64_t)HV * DIM * DIM;
        float *q = (float *)malloc(qk_e * 4), *k = (float *)malloc(qk_e * 4);
        float *v = (float *)malloc(v_e * 4);
        float *g = (float *)malloc(gb_e * 4), *b = (float *)malloc(gb_e * 4);
        for (uint64_t i = 0; i < qk_e; i++) { q[i] = gqa_test_randf() - 0.5f; k[i] = gqa_test_randf() - 0.5f; }
        /* the model L2-normalizes q/k before this kernel; without it
         * beta*|k|^2 >> 1 and the delta-rule map EXPANDS - 200 raw
         * random tokens overflow BOTH kernels (how this test first
         * failed). Normalize per row like production. */
        for (uint64_t r = 0; r < (uint64_t)NT * HK; r++) {
            float nq = 0.f, nk = 0.f;
            for (uint32_t d = 0; d < DIM; d++) {
                nq += q[r * DIM + d] * q[r * DIM + d];
                nk += k[r * DIM + d] * k[r * DIM + d];
            }
            nq = 1.0f / sqrtf(nq + 1e-6f); nk = 1.0f / sqrtf(nk + 1e-6f);
            for (uint32_t d = 0; d < DIM; d++) { q[r * DIM + d] *= nq; k[r * DIM + d] *= nk; }
        }
        for (uint64_t i = 0; i < v_e; i++) v[i] = gqa_test_randf() - 0.5f;
        for (uint64_t i = 0; i < gb_e; i++) {
            g[i] = -0.05f - 0.5f * gqa_test_randf(); /* log decay < 0 */
            b[i] = 0.9f * gqa_test_randf() + 0.05f;
        }
        void *dq, *dk, *dv, *dg, *db, *dst, *dout;
        cudaMalloc(&dq, qk_e * 4); cudaMalloc(&dk, qk_e * 4);
        cudaMalloc(&dv, v_e * 4); cudaMalloc(&dg, gb_e * 4);
        cudaMalloc(&db, gb_e * 4); cudaMalloc(&dst, st_e * 4);
        cudaMalloc(&dout, v_e * 4);
        cudaMemcpy(dq, q, qk_e * 4, cudaMemcpyHostToDevice);
        cudaMemcpy(dk, k, qk_e * 4, cudaMemcpyHostToDevice);
        cudaMemcpy(dv, v, v_e * 4, cudaMemcpyHostToDevice);
        cudaMemcpy(dg, g, gb_e * 4, cudaMemcpyHostToDevice);
        cudaMemcpy(db, b, gb_e * 4, cudaMemcpyHostToDevice);
        float *o_ref = (float *)malloc(v_e * 4), *o_ck = (float *)malloc(v_e * 4);
        float *s_ref = (float *)malloc(st_e * 4), *s_ck = (float *)malloc(st_e * 4);
        cudaMemset(dst, 0, st_e * 4);
        unsetenv("PULSAR_GDN_CHUNK");
        ok &= pulsar_qwen35_gdn_batch(dout, dst, dq, dk, dv, dg, db, HV, HK, DIM, NT);
        cudaMemcpy(o_ref, dout, v_e * 4, cudaMemcpyDeviceToHost);
        cudaMemcpy(s_ref, dst, st_e * 4, cudaMemcpyDeviceToHost);
        cudaMemset(dst, 0, st_e * 4);
        setenv("PULSAR_GDN_CHUNK", "1", 1);
        ok &= pulsar_qwen35_gdn_batch(dout, dst, dq, dk, dv, dg, db, HV, HK, DIM, NT);
        unsetenv("PULSAR_GDN_CHUNK");
        cudaMemcpy(o_ck, dout, v_e * 4, cudaMemcpyDeviceToHost);
        cudaMemcpy(s_ck, dst, st_e * 4, cudaMemcpyDeviceToHost);
        if (cudaDeviceSynchronize() != cudaSuccess) ok = 0;
        float od = 0.0f, sd = 0.0f;
        uint64_t bad_ref = 0, bad_ck = 0;
        int64_t first_bad = -1;
        for (uint64_t i = 0; i < v_e; i++) {
            if (!isfinite(o_ref[i])) bad_ref++;
            if (!isfinite(o_ck[i])) { bad_ck++; if (first_bad < 0) first_bad = (int64_t)i; }
            od = fmaxf(od, fabsf(o_ref[i] - o_ck[i]));
        }
        for (uint64_t i = 0; i < st_e; i++) sd = fmaxf(sd, fabsf(s_ref[i] - s_ck[i]));
        if (bad_ref || bad_ck)
            fprintf(stderr, "qwen35 selftest: gdn chunked nonfinite ref=%llu ck=%llu first_ck=%lld (tok %lld)\n",
                    (unsigned long long)bad_ref, (unsigned long long)bad_ck,
                    (long long)first_bad, (long long)(first_bad / ((int64_t)8 * 128)));
        int pass = ok && od < 2e-3f && sd < 2e-3f;
        fprintf(stderr, "qwen35 selftest: gdn chunked (n_tok=%u): %s (out diff %.2e, state diff %.2e)\n",
                NT, pass ? "PASS" : "FAIL", od, sd);
        if (!pass) ok = 0;
        cudaFree(dq); cudaFree(dk); cudaFree(dv); cudaFree(dg);
        cudaFree(db); cudaFree(dst); cudaFree(dout);
        free(q); free(k); free(v); free(g); free(b);
        free(o_ref); free(o_ck); free(s_ref); free(s_ck);
    }

    if (cudaDeviceSynchronize() != cudaSuccess) ok = 0;
    if (ok) fprintf(stderr, "qwen35 selftest: PASS\n");
    return ok;
}
