/* Kimi-K3 kernels: KDA (Kimi Delta Attention) and Attention Residuals.
 *
 * Included by pulsar_kernels.cu after qwen35_kernels.inc, which supplies the
 * pieces K3 shares with Qwen3.6's Gated DeltaNet: the depthwise conv step,
 * l2-norm, and the elementwise sigmoid gate (K3's MLA output gate is exactly
 * pulsar_qwen35_sigmoid_gate).
 *
 * KDA differs from GDN in one load-bearing way: the forget gate is per
 * CHANNEL of the key dimension, not one scalar per head. Diag(alpha_t) sits
 * between the state and the delta-rule update:
 *
 *   S_t = (I - beta_t k_t k_t^T) Diag(alpha_t) S_{t-1} + beta_t k_t v_t^T
 *
 * so the decay multiplies row i of the state rather than the whole matrix.
 *
 * Reference: MoonshotAI/Kimi-K3 tech report S2.1.1 and llama.cpp PR #26185
 * (ggml-cuda/gated_delta_net.cu, KDA = true branch; src/models/kimi-k3.cpp).
 */

/* KDA mixing coefficients on-device, the lower-bounded-decay form.
 *
 * Kimi-Linear used an unbounded log-decay, g = -exp(A_log) * softplus(z).
 * K3 swaps the activation entirely (it is NOT a clamp on the old one):
 *
 *   g = g_min * sigmoid(exp(A_log) * (z + dt_bias))
 *
 * with g_min = -5 fixed, which pins the cumulative decay over a 16-token
 * tile inside BF16 range - the whole point of the change upstream, where it
 * lets every chunk tile hit tensor cores. Here it just means alpha never
 * underflows. ssm_a holds -exp(A_log) folded at conversion time, so
 * exp(A_log) == -a[h] and the argument is -(a[h] * z).
 *
 * g is per channel: [n_tok][n_head][head_dim]. beta stays per head and is a
 * plain sigmoid, as in GDN. Both are updated in place. */
__global__ static void k3_kda_coeffs_kernel(
        float *g,            /* [n_tok][n_head][head_dim] raw f_b(f_a(x)) */
        float *beta,         /* [n_tok][n_head] raw logits */
        const float *a,      /* [n_head] = -exp(A_log) */
        const float *dt,     /* [n_head][head_dim] bias */
        uint32_t n_tok, uint32_t n_head, uint32_t head_dim,
        float g_min) {
    const uint64_t total = (uint64_t)n_tok * n_head * head_dim;
    uint64_t idx = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total) {
        const uint32_t c = (uint32_t)(idx % ((uint64_t)n_head * head_dim));
        const uint32_t h = c / head_dim;
        const float z = g[idx] + dt[c];
        g[idx] = g_min / (1.0f + expf(a[h] * z)); /* g_min*sigmoid(-a[h]*z) */
    }
    if (idx < (uint64_t)n_tok * n_head) {
        beta[idx] = 1.0f / (1.0f + expf(-beta[idx]));
    }
}

extern "C" int pulsar_k3_kda_coeffs(
        void *g_dev, void *beta_dev, const void *a_dev, const void *dt_dev,
        uint32_t n_tok, uint32_t n_head, uint32_t head_dim, float g_min) {
    if (!g_dev || !beta_dev || !a_dev || !dt_dev || n_tok == 0 ||
        n_head == 0 || head_dim == 0) {
        return 0;
    }
    const uint64_t total = (uint64_t)n_tok * n_head * head_dim;
    k3_kda_coeffs_kernel<<<(uint32_t)((total + 255u) / 256u), 256>>>(
            (float *)g_dev, (float *)beta_dev, (const float *)a_dev,
            (const float *)dt_dev, n_tok, n_head, head_dim, g_min);
    return cuda_ok(cudaGetLastError(), "k3 kda coeffs launch");
}

/* KDA autoregressive step, one token, all heads in one launch. Same block
 * shape as qwen35_gdn_step_kernel (one block per head, thread j owns state
 * column j so S[i][j] reads stay coalesced), with the scalar decay replaced
 * by a per-row vector:
 *   S[i][j] *= alpha[i]              (alpha = exp(g), g < 0)
 *   kv_j     = sum_i S[i][j] k_i
 *   d_j      = (v_j - kv_j) * beta
 *   S[i][j] += k_i * d_j
 *   o_j      = sum_i S[i][j] q_i
 * alpha is computed once per row into shared memory rather than per
 * (row, column): the naive form costs dim expf calls in every thread. */
__global__ static void k3_kda_step_kernel(
        float *out,          /* [n_head][dim] */
        float *state,        /* [n_head][dim][dim] in place */
        const float *q,      /* [n_head][dim] conv+silu+l2, pre-scale */
        const float *k,      /* [n_head][dim] */
        const float *v,      /* [n_head][dim] */
        const float *g,      /* [n_head][dim] channel-wise log-decay */
        const float *beta,   /* [n_head] */
        uint32_t n_head,
        uint32_t dim) {
    uint32_t h = blockIdx.x;
    uint32_t j = threadIdx.x;
    if (h >= n_head || j >= dim) return;
    const float *qh = q + (uint64_t)h * dim;
    const float *kh = k + (uint64_t)h * dim;
    const float *gh = g + (uint64_t)h * dim;
    float *S = state + (uint64_t)h * dim * dim;
    const float qscale = rsqrtf((float)dim);

    __shared__ float sq[256];
    __shared__ float sk[256];
    __shared__ float sa[256];
    for (uint32_t i = j; i < dim; i += blockDim.x) {
        sq[i] = qh[i] * qscale;
        sk[i] = kh[i];
        sa[i] = expf(gh[i]);
    }
    __syncthreads();

    /* decay + kv in one pass down column j */
    float kv = 0.0f;
    for (uint32_t i = 0; i < dim; i++) {
        const float s = S[(uint64_t)i * dim + j] * sa[i];
        S[(uint64_t)i * dim + j] = s;
        kv += s * sk[i];
    }
    const float d = (v[(uint64_t)h * dim + j] - kv) * beta[h];
    /* rank-1 update + output in a second pass */
    float o = 0.0f;
    for (uint32_t i = 0; i < dim; i++) {
        const float s = S[(uint64_t)i * dim + j] + sk[i] * d;
        S[(uint64_t)i * dim + j] = s;
        o += s * sq[i];
    }
    out[(uint64_t)h * dim + j] = o;
}

extern "C" int pulsar_k3_kda_step(
        void *out_dev, void *state_dev,
        const void *q_dev, const void *k_dev, const void *v_dev,
        const void *g_dev, const void *beta_dev,
        uint32_t n_head, uint32_t dim) {
    if (!out_dev || !state_dev || !q_dev || !k_dev || !v_dev || !g_dev ||
        !beta_dev || n_head == 0 || dim == 0 || dim > 256u) {
        return 0;
    }
    k3_kda_step_kernel<<<n_head, dim>>>(
            (float *)out_dev, (float *)state_dev,
            (const float *)q_dev, (const float *)k_dev, (const float *)v_dev,
            (const float *)g_dev, (const float *)beta_dev, n_head, dim);
    return cuda_ok(cudaGetLastError(), "k3 kda step launch");
}

/* Upper bound on banked checkpoints: ceil(n_layer / attn_res_block) + 1.
 * K3 is 93 layers with block 12, so 9. The launcher rejects anything
 * larger rather than silently overrunning the score array. */
#define K3_ATTN_RES_MAX_CKPT 63

/* Attention Residuals: the layer reads a learned convex mix of the banked
 * block checkpoints and the live residual stream, instead of the stream
 * alone. Attention runs over DEPTH, not sequence - at decode there are at
 * most ceil(n_layer/block_size)+1 checkpoints for the current token, so the
 * whole thing is one small softmax per layer and the state never grows with
 * context.
 *
 * score_i = sum_c rmsnorm(src_i)[c] * w[c], and the same for the live
 * stream; softmax over the n_ckpt+1 scores; then the weighted sum uses the
 * RAW vectors, not the normalized ones (mirrors _apply_attn_res upstream -
 * normalizing only the scores is deliberate, it stops a large-magnitude
 * layer from dominating the weights).
 *
 * One block per token, blockDim.x threads striding n_embd. n_ckpt is small
 * so the per-checkpoint loop is serial. */
__global__ static void k3_attn_res_kernel(
        float *out,          /* [n_tok][n_embd] (may alias cur) */
        const float *cur,    /* [n_tok][n_embd] live residual stream */
        const float *ckpt,   /* [n_ckpt][n_tok][n_embd] banked */
        const float *w,      /* [n_embd] score weights */
        uint32_t n_tok, uint32_t n_embd, uint32_t n_ckpt, float eps) {
    const uint32_t tok = blockIdx.x;
    if (tok >= n_tok) return;
    /* Static, not `extern __shared__`: the score array is tiny and
     * bounded (ceil(n_layer/block_size)+1, 9 for K3), and mixing a
     * dynamic region with a static one here had the compiler reading
     * past it - a __shared__ OOB the CPU-reference selftest cannot see,
     * because it only shows up at the real n_embd and checkpoint count. */
    __shared__ float sh[K3_ATTN_RES_MAX_CKPT + 1];
    __shared__ float red[256];

    /* score every source, including the live stream at index n_ckpt */
    for (uint32_t s = 0; s <= n_ckpt; s++) {
        const float *src = s == n_ckpt
                ? cur + (uint64_t)tok * n_embd
                : ckpt + ((uint64_t)s * n_tok + tok) * n_embd;
        float ss = 0.0f, sw = 0.0f;
        for (uint32_t i = threadIdx.x; i < n_embd; i += blockDim.x) {
            ss += src[i] * src[i];
        }
        red[threadIdx.x] = ss;
        __syncthreads();
        for (uint32_t st = blockDim.x >> 1; st > 0; st >>= 1) {
            if (threadIdx.x < st) red[threadIdx.x] += red[threadIdx.x + st];
            __syncthreads();
        }
        const float scale = rsqrtf(red[0] / (float)n_embd + eps);
        __syncthreads();
        for (uint32_t i = threadIdx.x; i < n_embd; i += blockDim.x) {
            sw += src[i] * scale * w[i];
        }
        red[threadIdx.x] = sw;
        __syncthreads();
        for (uint32_t st = blockDim.x >> 1; st > 0; st >>= 1) {
            if (threadIdx.x < st) red[threadIdx.x] += red[threadIdx.x + st];
            __syncthreads();
        }
        if (threadIdx.x == 0) sh[s] = red[0];
        __syncthreads();
    }

    /* softmax over the n_ckpt+1 scores, then mix the RAW vectors */
    if (threadIdx.x == 0) {
        float mx = sh[0];
        for (uint32_t s = 1; s <= n_ckpt; s++) mx = fmaxf(mx, sh[s]);
        float sum = 0.0f;
        for (uint32_t s = 0; s <= n_ckpt; s++) {
            sh[s] = expf(sh[s] - mx);
            sum += sh[s];
        }
        const float inv = 1.0f / sum;
        for (uint32_t s = 0; s <= n_ckpt; s++) sh[s] *= inv;
    }
    __syncthreads();

    float *orow = out + (uint64_t)tok * n_embd;
    const float *crow = cur + (uint64_t)tok * n_embd;
    for (uint32_t i = threadIdx.x; i < n_embd; i += blockDim.x) {
        float acc = crow[i] * sh[n_ckpt];
        for (uint32_t s = 0; s < n_ckpt; s++) {
            acc += ckpt[((uint64_t)s * n_tok + tok) * n_embd + i] * sh[s];
        }
        orow[i] = acc;
    }
}

extern "C" int pulsar_k3_attn_res(
        void *out_dev, const void *cur_dev, const void *ckpt_dev,
        const void *w_dev, uint32_t n_tok, uint32_t n_embd,
        uint32_t n_ckpt, float eps) {
    if (!out_dev || !cur_dev || !w_dev || n_tok == 0 || n_embd == 0) return 0;
    if (n_ckpt > 0 && !ckpt_dev) return 0;
    if (n_ckpt > K3_ATTN_RES_MAX_CKPT) return 0;
    k3_attn_res_kernel<<<n_tok, 256>>>(
            (float *)out_dev, (const float *)cur_dev, (const float *)ckpt_dev,
            (const float *)w_dev, n_tok, n_embd, n_ckpt, eps);
    return cuda_ok(cudaGetLastError(), "k3 attn res launch");
}

/* ---- selftest ----------------------------------------------------------- */

/* Test hook: run pulsar_glu standalone so an act_op can be checked without
 * going through a whole MoE launch. Not used on any forward path. */
__global__ static void pulsar_glu_eval_kernel(
        float *out, const float *g, const float *u, uint32_t n, uint32_t op) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    out[i] = pulsar_glu(g[i], u[i], op);
}

extern "C" int pulsar_glu_eval(
        void *out_dev, const void *g_dev, const void *u_dev,
        uint32_t n, uint32_t op) {
    if (!out_dev || !g_dev || !u_dev || n == 0) return 0;
    pulsar_glu_eval_kernel<<<(n + 255u) / 256u, 256>>>(
            (float *)out_dev, (const float *)g_dev, (const float *)u_dev, n, op);
    return cuda_ok(cudaGetLastError(), "pulsar glu eval launch");
}

extern "C" int pulsar_k3_selftest(void) {
    int ok = 1;
    const uint32_t DIM = 128, H = 6;

    /* kda coeffs vs host reference */
    {
        const uint32_t NT = 3, TOT = NT * H * DIM;
        float *g = (float *)malloc(TOT * 4), *gr = (float *)malloc(TOT * 4);
        float *b = (float *)malloc(NT * H * 4), *br = (float *)malloc(NT * H * 4);
        float *a = (float *)malloc(H * 4), *dt = (float *)malloc(H * DIM * 4);
        for (uint32_t i = 0; i < TOT; i++) { g[i] = gqa_test_randf() * 3.0f; gr[i] = g[i]; }
        for (uint32_t i = 0; i < NT * H; i++) { b[i] = gqa_test_randf() * 3.0f; br[i] = b[i]; }
        /* a = -exp(A_log) is strictly negative */
        for (uint32_t i = 0; i < H; i++) a[i] = -expf(gqa_test_randf());
        for (uint32_t i = 0; i < H * DIM; i++) dt[i] = gqa_test_randf();
        void *dg, *db, *da, *ddt;
        cudaMalloc(&dg, TOT * 4); cudaMalloc(&db, NT * H * 4);
        cudaMalloc(&da, H * 4); cudaMalloc(&ddt, H * DIM * 4);
        cudaMemcpy(dg, g, TOT * 4, cudaMemcpyHostToDevice);
        cudaMemcpy(db, b, NT * H * 4, cudaMemcpyHostToDevice);
        cudaMemcpy(da, a, H * 4, cudaMemcpyHostToDevice);
        cudaMemcpy(ddt, dt, H * DIM * 4, cudaMemcpyHostToDevice);
        ok &= pulsar_k3_kda_coeffs(dg, db, da, ddt, NT, H, DIM, -5.0f);
        cudaMemcpy(g, dg, TOT * 4, cudaMemcpyDeviceToHost);
        cudaMemcpy(b, db, NT * H * 4, cudaMemcpyDeviceToHost);
        for (uint32_t i = 0; i < TOT && ok; i++) {
            const uint32_t c = i % (H * DIM);
            const float z = gr[i] + dt[c];
            /* written the other way round from the kernel: explicit
             * exp(A_log) = -a[h] and an explicit sigmoid */
            const float want = -5.0f * (1.0f / (1.0f + expf(-(-a[c / DIM]) * z)));
            if (fabsf(want - g[i]) > 1.0e-5f) {
                fprintf(stderr, "k3 selftest: kda coeff g mismatch at %u: %g vs %g\n",
                        i, (double)want, (double)g[i]);
                ok = 0;
            }
            /* the whole point of the lower bound: alpha stays representable */
            if (ok && !(g[i] >= -5.0f && g[i] <= 0.0f)) {
                fprintf(stderr, "k3 selftest: g out of [-5,0] at %u: %g\n", i, (double)g[i]);
                ok = 0;
            }
        }
        for (uint32_t i = 0; i < NT * H && ok; i++) {
            const float want = 1.0f / (1.0f + expf(-br[i]));
            if (fabsf(want - b[i]) > 1.0e-6f) {
                fprintf(stderr, "k3 selftest: kda coeff beta mismatch at %u\n", i);
                ok = 0;
            }
        }
        cudaFree(dg); cudaFree(db); cudaFree(da); cudaFree(ddt);
        free(g); free(gr); free(b); free(br); free(a); free(dt);
    }

    /* kda step vs host reference, 3 sequential tokens (state carries) */
    if (ok) {
        const uint64_t sb = (uint64_t)H * DIM * DIM * 4;
        float *S = (float *)calloc((size_t)H * DIM * DIM, 4);
        float *q = (float *)malloc((size_t)H * DIM * 4);
        float *k = (float *)malloc((size_t)H * DIM * 4);
        float *v = (float *)malloc((size_t)H * DIM * 4);
        float *g = (float *)malloc((size_t)H * DIM * 4);
        float *beta = (float *)malloc(H * 4);
        float *got = (float *)malloc((size_t)H * DIM * 4);
        void *dS, *dq, *dk, *dv, *dg, *dbeta, *dout;
        cudaMalloc(&dS, sb); cudaMemset(dS, 0, sb);
        cudaMalloc(&dq, (size_t)H * DIM * 4);
        cudaMalloc(&dk, (size_t)H * DIM * 4);
        cudaMalloc(&dv, (size_t)H * DIM * 4);
        cudaMalloc(&dg, (size_t)H * DIM * 4);
        cudaMalloc(&dbeta, H * 4);
        cudaMalloc(&dout, (size_t)H * DIM * 4);
        for (int step = 0; step < 3 && ok; step++) {
            for (uint32_t i = 0; i < H * DIM; i++) {
                q[i] = gqa_test_randf();
                k[i] = gqa_test_randf();
                v[i] = gqa_test_randf();
                g[i] = -fabsf(gqa_test_randf()) * 2.0f; /* channel-wise, < 0 */
            }
            for (uint32_t h = 0; h < H; h++) beta[h] = 0.5f + 0.4f * gqa_test_randf();
            cudaMemcpy(dq, q, (size_t)H * DIM * 4, cudaMemcpyHostToDevice);
            cudaMemcpy(dk, k, (size_t)H * DIM * 4, cudaMemcpyHostToDevice);
            cudaMemcpy(dv, v, (size_t)H * DIM * 4, cudaMemcpyHostToDevice);
            cudaMemcpy(dg, g, (size_t)H * DIM * 4, cudaMemcpyHostToDevice);
            cudaMemcpy(dbeta, beta, H * 4, cudaMemcpyHostToDevice);
            ok &= pulsar_k3_kda_step(dout, dS, dq, dk, dv, dg, dbeta, H, DIM);
            cudaMemcpy(got, dout, (size_t)H * DIM * 4, cudaMemcpyDeviceToHost);

            const float qs = 1.0f / sqrtf((float)DIM);
            for (uint32_t h = 0; h < H && ok; h++) {
                float *Sh = S + (size_t)h * DIM * DIM;
                for (uint32_t i = 0; i < DIM; i++) {
                    const float al = expf(g[h * DIM + i]);
                    for (uint32_t j = 0; j < DIM; j++) Sh[i * DIM + j] *= al;
                }
                for (uint32_t j = 0; j < DIM; j++) {
                    float kv = 0.0f;
                    for (uint32_t i = 0; i < DIM; i++) kv += Sh[i * DIM + j] * k[h * DIM + i];
                    const float d = (v[h * DIM + j] - kv) * beta[h];
                    float o = 0.0f;
                    for (uint32_t i = 0; i < DIM; i++) {
                        Sh[i * DIM + j] += k[h * DIM + i] * d;
                        o += Sh[i * DIM + j] * q[h * DIM + i] * qs;
                    }
                    if (fabsf(o - got[h * DIM + j]) > 2.0e-4f * fmaxf(1.0f, fabsf(o))) {
                        fprintf(stderr,
                                "k3 selftest: kda step mismatch step %d h %u j %u: %g vs %g\n",
                                step, h, j, (double)o, (double)got[h * DIM + j]);
                        ok = 0;
                        break;
                    }
                }
            }
        }
        cudaFree(dS); cudaFree(dq); cudaFree(dk); cudaFree(dv);
        cudaFree(dg); cudaFree(dbeta); cudaFree(dout);
        free(S); free(q); free(k); free(v); free(g); free(beta); free(got);
    }

    /* attn res vs host reference, including the n_ckpt == 0 identity case */
    if (ok) {
        const uint32_t NE = 320, NT = 3, NC = 4;
        float *cur = (float *)malloc((size_t)NT * NE * 4);
        float *ck = (float *)malloc((size_t)NC * NT * NE * 4);
        float *w = (float *)malloc(NE * 4);
        float *got = (float *)malloc((size_t)NT * NE * 4);
        for (uint32_t i = 0; i < NT * NE; i++) cur[i] = gqa_test_randf();
        for (uint32_t i = 0; i < NC * NT * NE; i++) ck[i] = gqa_test_randf();
        for (uint32_t i = 0; i < NE; i++) w[i] = gqa_test_randf();
        void *dcur, *dck, *dw, *dout;
        cudaMalloc(&dcur, (size_t)NT * NE * 4);
        cudaMalloc(&dck, (size_t)NC * NT * NE * 4);
        cudaMalloc(&dw, NE * 4);
        cudaMalloc(&dout, (size_t)NT * NE * 4);
        cudaMemcpy(dcur, cur, (size_t)NT * NE * 4, cudaMemcpyHostToDevice);
        cudaMemcpy(dck, ck, (size_t)NC * NT * NE * 4, cudaMemcpyHostToDevice);
        cudaMemcpy(dw, w, NE * 4, cudaMemcpyHostToDevice);

        ok &= pulsar_k3_attn_res(dout, dcur, dck, dw, NT, NE, 0, 1.0e-5f);
        cudaMemcpy(got, dout, (size_t)NT * NE * 4, cudaMemcpyDeviceToHost);
        for (uint32_t i = 0; i < NT * NE && ok; i++) {
            if (fabsf(got[i] - cur[i]) > 1.0e-6f) {
                fprintf(stderr, "k3 selftest: attn res n_ckpt=0 not identity at %u\n", i);
                ok = 0;
            }
        }

        if (ok) ok &= pulsar_k3_attn_res(dout, dcur, dck, dw, NT, NE, NC, 1.0e-5f);
        cudaMemcpy(got, dout, (size_t)NT * NE * 4, cudaMemcpyDeviceToHost);
        float *sc = (float *)malloc((NC + 1) * 4);
        for (uint32_t t = 0; t < NT && ok; t++) {
            for (uint32_t s = 0; s <= NC; s++) {
                const float *src = s == NC ? cur + (size_t)t * NE
                                           : ck + ((size_t)s * NT + t) * NE;
                float ss = 0.0f;
                for (uint32_t i = 0; i < NE; i++) ss += src[i] * src[i];
                const float scale = 1.0f / sqrtf(ss / (float)NE + 1.0e-5f);
                float acc = 0.0f;
                for (uint32_t i = 0; i < NE; i++) acc += src[i] * scale * w[i];
                sc[s] = acc;
            }
            float mx = sc[0];
            for (uint32_t s = 1; s <= NC; s++) mx = fmaxf(mx, sc[s]);
            float sum = 0.0f;
            for (uint32_t s = 0; s <= NC; s++) { sc[s] = expf(sc[s] - mx); sum += sc[s]; }
            for (uint32_t s = 0; s <= NC; s++) sc[s] /= sum;
            for (uint32_t i = 0; i < NE && ok; i++) {
                float want = cur[(size_t)t * NE + i] * sc[NC];
                for (uint32_t s = 0; s < NC; s++) {
                    want += ck[((size_t)s * NT + t) * NE + i] * sc[s];
                }
                const float g_ = got[(size_t)t * NE + i];
                if (fabsf(want - g_) > 2.0e-5f * fmaxf(1.0f, fabsf(want))) {
                    fprintf(stderr, "k3 selftest: attn res mismatch tok %u ch %u: %g vs %g\n",
                            t, i, (double)want, (double)g_);
                    ok = 0;
                }
            }
        }
        cudaFree(dcur); cudaFree(dck); cudaFree(dw); cudaFree(dout);
        free(cur); free(ck); free(w); free(got); free(sc);
    }

    /* situ-glu act op vs the formula it is meant to be */
    if (ok) {
        const uint32_t N = 64;
        float *g = (float *)malloc(N * 4), *u = (float *)malloc(N * 4);
        float *got = (float *)malloc(N * 4);
        for (uint32_t i = 0; i < N; i++) {
            /* span the soft-cap regime, not just the near-linear origin */
            g[i] = gqa_test_randf() * 60.0f;
            u[i] = gqa_test_randf() * 60.0f;
        }
        void *dg, *du, *dout;
        cudaMalloc(&dg, N * 4); cudaMalloc(&du, N * 4); cudaMalloc(&dout, N * 4);
        cudaMemcpy(dg, g, N * 4, cudaMemcpyHostToDevice);
        cudaMemcpy(du, u, N * 4, cudaMemcpyHostToDevice);
        ok &= pulsar_glu_eval(dout, dg, du, N, 4u);
        cudaMemcpy(got, dout, N * 4, cudaMemcpyDeviceToHost);
        for (uint32_t i = 0; i < N && ok; i++) {
            const float a = 4.0f * tanhf(g[i] / 4.0f) * (1.0f / (1.0f + expf(-g[i])));
            const float want = a * (25.0f * tanhf(u[i] / 25.0f));
            if (fabsf(want - got[i]) > 1.0e-4f * fmaxf(1.0f, fabsf(want))) {
                fprintf(stderr, "k3 selftest: situ mismatch at %u: %g vs %g\n",
                        i, (double)want, (double)got[i]);
                ok = 0;
            }
            /* the bound the soft cap exists to provide */
            if (ok && fabsf(got[i]) > 4.0f * 25.0f) {
                fprintf(stderr, "k3 selftest: situ exceeded b1*b2 at %u: %g\n",
                        i, (double)got[i]);
                ok = 0;
            }
        }
        cudaFree(dg); cudaFree(du); cudaFree(dout);
        free(g); free(u); free(got);
    }

    fprintf(stderr, "k3-selftest: %s\n", ok ? "PASS" : "FAIL");
    return ok;
}
