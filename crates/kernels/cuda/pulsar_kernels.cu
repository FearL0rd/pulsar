/* pulsar CUDA kernel library.
 *
 * gqa_kernels.inc is derived verbatim from the NeutronStar fork of ds4
 * (github.com/antirez/ds4), MIT License:
 *   Copyright (c) 2026 The ds4.c authors
 *   Copyright (c) 2023-2026 The ggml authors
 * The shim below provides the minimal glue the .inc expects (a tensor is
 * a device pointer plus a byte count) so the kernels build standalone.
 */
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct ds4_gpu_tensor {
    void *ptr;
    uint64_t bytes;
} ds4_gpu_tensor;

static int cuda_ok(cudaError_t err, const char *what) {
    if (err == cudaSuccess) return 1;
    fprintf(stderr, "pulsar-kernels: %s: %s\n", what, cudaGetErrorString(err));
    return 0;
}

static ds4_gpu_tensor *ds4_gpu_tensor_alloc(uint64_t bytes) {
    ds4_gpu_tensor *t = (ds4_gpu_tensor *)calloc(1, sizeof(*t));
    if (!t) return NULL;
    t->bytes = bytes;
    if (!cuda_ok(cudaMalloc(&t->ptr, bytes), "cudaMalloc")) {
        free(t);
        return NULL;
    }
    return t;
}

static void ds4_gpu_tensor_free(ds4_gpu_tensor *t) {
    if (!t) return;
    if (t->ptr) (void)cudaFree(t->ptr);
    free(t);
}

static int ds4_gpu_tensor_write(ds4_gpu_tensor *t, uint64_t off,
                                const void *src, uint64_t bytes) {
    if (!t || off + bytes > t->bytes) return 0;
    return cuda_ok(cudaMemcpy((char *)t->ptr + off, src, bytes,
                              cudaMemcpyHostToDevice), "h2d");
}

static int ds4_gpu_tensor_read(const ds4_gpu_tensor *t, uint64_t off,
                               void *dst, uint64_t bytes) {
    if (!t || off + bytes > t->bytes) return 0;
    return cuda_ok(cudaMemcpy(dst, (const char *)t->ptr + off, bytes,
                              cudaMemcpyDeviceToHost), "d2h");
}

#include "gqa_kernels.inc"

static float f16_to_f32_host(uint16_t h) {
    /* scalar IEEE 754 half -> float, host side (no device intrinsics) */
    uint32_t sign = (uint32_t)(h & 0x8000u) << 16;
    uint32_t exp = (h >> 10) & 0x1F;
    uint32_t man = h & 0x3FF;
    uint32_t bits;
    if (exp == 0) {
        if (man == 0) {
            bits = sign;
        } else {
            exp = 127 - 15 + 1;
            while ((man & 0x400) == 0) { man <<= 1; exp--; }
            man &= 0x3FF;
            bits = sign | (exp << 23) | (man << 13);
        }
    } else if (exp == 31) {
        bits = sign | 0x7F800000u | (man << 13);
    } else {
        bits = sign | ((exp - 15 + 127) << 23) | (man << 13);
    }
    float f;
    memcpy(&f, &bits, sizeof(f));
    return f;
}

/* ---- pulsar-native Q8_0 matmul ----------------------------------------
 * GGML q8_0 block: 32 int8 quants + one f16 scale (34 bytes). Weights are
 * row-major q8_0; activations are f32. One thread block per (row, token),
 * 256 threads reduce across in_dim. Correctness-first: tuning happens at
 * parity time, against measurements. */

typedef struct __align__(2) {
    uint16_t scale_f16;
    int8_t q[32];
} q8_0_block;

__device__ static float f16_to_f32(uint16_t h) {
    return __half2float(__ushort_as_half(h));
}

__global__ static void q8_0_matmul_kernel(
        float *out,                /* [n_tok][out_dim] */
        const q8_0_block *w,       /* [out_dim][in_dim/32] */
        const float *x,            /* [n_tok][in_dim] */
        uint32_t in_dim,
        uint32_t out_dim,
        uint32_t n_tok) {
    const uint32_t row = blockIdx.x;
    const uint32_t tok = blockIdx.y;
    if (row >= out_dim || tok >= n_tok) return;
    const uint32_t blocks = in_dim / 32u;
    const q8_0_block *wr = w + (uint64_t)row * blocks;
    const float *xt = x + (uint64_t)tok * in_dim;
    float acc = 0.0f;
    for (uint32_t b = threadIdx.x; b < blocks; b += blockDim.x) {
        const q8_0_block *blk = &wr[b];
        float s = f16_to_f32(blk->scale_f16);
        float dot = 0.0f;
        const float *xb = xt + (uint64_t)b * 32u;
        for (int i = 0; i < 32; i++) dot += (float)blk->q[i] * xb[i];
        acc += s * dot;
    }
    __shared__ float red[256];
    red[threadIdx.x] = acc;
    __syncthreads();
    for (uint32_t s = blockDim.x / 2u; s != 0; s >>= 1u) {
        if (threadIdx.x < s) red[threadIdx.x] += red[threadIdx.x + s];
        __syncthreads();
    }
    if (threadIdx.x == 0) out[(uint64_t)tok * out_dim + row] = red[0];
}

extern "C" int pulsar_q8_0_matmul(
        void *out_dev,
        const void *w_dev,
        const void *x_dev,
        uint32_t in_dim,
        uint32_t out_dim,
        uint32_t n_tok) {
    if (in_dim == 0 || in_dim % 32u != 0 || out_dim == 0 || n_tok == 0) return 0;
    dim3 grid(out_dim, n_tok, 1);
    q8_0_matmul_kernel<<<grid, 256>>>(
            (float *)out_dev, (const q8_0_block *)w_dev,
            (const float *)x_dev, in_dim, out_dim, n_tok);
    return cuda_ok(cudaGetLastError(), "q8_0 matmul launch");
}

/* CPU-reference selftest: quantize random weights to q8_0 on the host,
 * run both pipelines, compare. */
static uint16_t f32_to_f16_bits(float f) {
    /* scalar IEEE 754 float -> half (round-to-nearest-even), host side */
    uint32_t bits;
    memcpy(&bits, &f, sizeof(bits));
    uint32_t sign = (bits >> 16) & 0x8000u;
    int32_t exp = (int32_t)((bits >> 23) & 0xFF) - 127 + 15;
    uint32_t man = bits & 0x7FFFFFu;
    if (exp <= 0) {
        if (exp < -10) return (uint16_t)sign;
        man |= 0x800000u;
        uint32_t shift = (uint32_t)(14 - exp);
        uint32_t half_man = man >> shift;
        uint32_t rem = man & ((1u << shift) - 1u);
        uint32_t halfway = 1u << (shift - 1u);
        if (rem > halfway || (rem == halfway && (half_man & 1u))) half_man++;
        return (uint16_t)(sign | half_man);
    }
    if (exp >= 31) return (uint16_t)(sign | 0x7C00u);
    uint32_t half_man = man >> 13;
    uint32_t rem = man & 0x1FFFu;
    if (rem > 0x1000u || (rem == 0x1000u && (half_man & 1u))) {
        half_man++;
        if (half_man == 0x400u) { half_man = 0; exp++; if (exp >= 31) return (uint16_t)(sign | 0x7C00u); }
    }
    return (uint16_t)(sign | ((uint32_t)exp << 10) | half_man);
}

extern "C" int pulsar_q8_0_matmul_selftest(void) {
    const uint32_t in_dim = 4096, out_dim = 512, n_tok = 3;
    const uint32_t blocks = in_dim / 32u;
    q8_0_block *w = (q8_0_block *)malloc((uint64_t)out_dim * blocks * sizeof(*w));
    float *wf = (float *)malloc((uint64_t)out_dim * in_dim * sizeof(float));
    float *x = (float *)malloc((uint64_t)n_tok * in_dim * sizeof(float));
    float *ref = (float *)calloc((uint64_t)n_tok * out_dim, sizeof(float));
    float *gpu = (float *)malloc((uint64_t)n_tok * out_dim * sizeof(float));

    for (uint64_t i = 0; i < (uint64_t)n_tok * in_dim; i++) x[i] = gqa_test_randf();
    /* quantize: per 32-block, scale = amax/127, q = round(v/scale) */
    for (uint32_t r = 0; r < out_dim; r++) {
        for (uint32_t b = 0; b < blocks; b++) {
            float amax = 0.0f, vals[32];
            for (int i = 0; i < 32; i++) {
                vals[i] = gqa_test_randf();
                float a = fabsf(vals[i]);
                if (a > amax) amax = a;
            }
            float scale = amax / 127.0f;
            q8_0_block *blk = &w[(uint64_t)r * blocks + b];
            blk->scale_f16 = f32_to_f16_bits(scale);
            float s = f16_to_f32_host(blk->scale_f16);
            for (int i = 0; i < 32; i++) {
                int qi = scale > 0.0f ? (int)lrintf(vals[i] / scale) : 0;
                if (qi > 127) qi = 127;
                if (qi < -127) qi = -127;
                blk->q[i] = (int8_t)qi;
                wf[(uint64_t)r * in_dim + b * 32u + i] = s * (float)qi;
            }
        }
    }
    /* reference matmul on the dequantized weights */
    for (uint32_t t = 0; t < n_tok; t++)
        for (uint32_t r = 0; r < out_dim; r++) {
            double acc = 0.0;
            for (uint32_t i = 0; i < in_dim; i++)
                acc += (double)wf[(uint64_t)r * in_dim + i] * x[(uint64_t)t * in_dim + i];
            ref[(uint64_t)t * out_dim + r] = (float)acc;
        }

    void *w_dev = NULL, *x_dev = NULL, *out_dev = NULL;
    const uint64_t w_bytes = (uint64_t)out_dim * blocks * sizeof(*w);
    const uint64_t x_bytes = (uint64_t)n_tok * in_dim * sizeof(float);
    const uint64_t o_bytes = (uint64_t)n_tok * out_dim * sizeof(float);
    int ok = cuda_ok(cudaMalloc(&w_dev, w_bytes), "w alloc") &&
             cuda_ok(cudaMalloc(&x_dev, x_bytes), "x alloc") &&
             cuda_ok(cudaMalloc(&out_dev, o_bytes), "out alloc") &&
             cuda_ok(cudaMemcpy(w_dev, w, w_bytes, cudaMemcpyHostToDevice), "w h2d") &&
             cuda_ok(cudaMemcpy(x_dev, x, x_bytes, cudaMemcpyHostToDevice), "x h2d") &&
             pulsar_q8_0_matmul(out_dev, w_dev, x_dev, in_dim, out_dim, n_tok) &&
             cuda_ok(cudaDeviceSynchronize(), "sync") &&
             cuda_ok(cudaMemcpy(gpu, out_dev, o_bytes, cudaMemcpyDeviceToHost), "d2h");
    float maxd = 0.0f, maxref = 0.0f;
    if (ok) {
        for (uint64_t i = 0; i < (uint64_t)n_tok * out_dim; i++) {
            float d = fabsf(gpu[i] - ref[i]);
            if (d > maxd) maxd = d;
            float a = fabsf(ref[i]);
            if (a > maxref) maxref = a;
        }
        ok = maxd <= 1e-3f * (maxref > 1.0f ? maxref : 1.0f);
    }
    fprintf(stderr, "q8_0-matmul-selftest: %s (max abs diff %.2e, max |ref| %.2e)\n",
            ok ? "PASS" : "FAIL", (double)maxd, (double)maxref);
    if (w_dev) cudaFree(w_dev);
    if (x_dev) cudaFree(x_dev);
    if (out_dev) cudaFree(out_dev);
    free(w); free(wf); free(x); free(ref); free(gpu);
    return ok;
}


/* ---- sigmoid router + top-k select ------------------------------------
 * Warp-per-token select, derived from ds4's glm_router_select_kernel (the
 * Hy3 router mirrors GLM: probs = sigmoid(logits), selection score =
 * prob + bias, route weights = selected probs normalized * scale).
 * pulsar contract: bias is an explicit device pointer, not a model-map
 * offset. n_expert <= 256, k_used <= n_expert. */

__device__ __forceinline__ static float router_sigmoid(float x) {
    if (x >= 0.0f) {
        const float e = expf(-x);
        return 1.0f / (1.0f + e);
    }
    const float e = expf(x);
    return e / (1.0f + e);
}

__device__ __forceinline__ static bool router_better(
        float av, uint32_t ai, float bv, uint32_t bi) {
    return av > bv || (av == bv && ai < bi);
}

__global__ static void router_select_kernel(
        int32_t *selected,         /* [n_tok][k_used] */
        float *weights,            /* [n_tok][k_used] */
        const float *logits,       /* [n_tok][n_expert] */
        const float *bias,         /* [n_expert] */
        uint32_t n_expert,
        uint32_t k_used,
        float weight_scale,
        uint32_t n_tok) {
    const uint32_t lane = threadIdx.x;
    const uint32_t token = blockIdx.x * blockDim.y + threadIdx.y;
    if (token >= n_tok || lane >= 32u) return;

    const float *log = logits + (uint64_t)token * n_expert;
    int32_t *sel = selected + (uint64_t)token * k_used;
    float *w = weights + (uint64_t)token * k_used;

    float local_prob[8];
    float local_score[8];
    #pragma unroll
    for (uint32_t j = 0; j < 8u; j++) {
        const uint32_t e = lane + j * 32u;
        if (e < n_expert) {
            const float p = router_sigmoid(log[e]);
            local_prob[j] = p;
            local_score[j] = p + bias[e];
        } else {
            local_prob[j] = 0.0f;
            local_score[j] = -INFINITY;
        }
    }
    __syncwarp();

    float sum = 0.0f;
    for (uint32_t k = 0; k < k_used; k++) {
        float best_score = -INFINITY;
        float best_prob = 0.0f;
        uint32_t best_idx = UINT32_MAX;
        #pragma unroll
        for (uint32_t j = 0; j < 8u; j++) {
            const uint32_t e = lane + j * 32u;
            if (router_better(local_score[j], e, best_score, best_idx)) {
                best_score = local_score[j];
                best_prob = local_prob[j];
                best_idx = e;
            }
        }
        #pragma unroll
        for (uint32_t mask = 16u; mask > 0u; mask >>= 1u) {
            const float other_score = __shfl_xor_sync(0xffffffffu, best_score, mask);
            const float other_prob = __shfl_xor_sync(0xffffffffu, best_prob, mask);
            const uint32_t other_idx = __shfl_xor_sync(0xffffffffu, best_idx, mask);
            if (router_better(other_score, other_idx, best_score, best_idx)) {
                best_score = other_score;
                best_prob = other_prob;
                best_idx = other_idx;
            }
        }
        #pragma unroll
        for (uint32_t j = 0; j < 8u; j++) {
            if (lane + j * 32u == best_idx) local_score[j] = -INFINITY;
        }
        if (lane == 0) {
            sel[k] = (int32_t)best_idx;
            w[k] = best_prob;
        }
        sum += best_prob;
    }

    if (lane == 0) {
        sum = fmaxf(sum, 6.103515625e-5f);
        for (uint32_t k = 0; k < k_used; k++) w[k] = w[k] / sum * weight_scale;
    }
}

extern "C" int pulsar_router_select(
        void *selected_dev,        /* int32 [n_tok][k_used] */
        void *weights_dev,         /* f32   [n_tok][k_used] */
        const void *logits_dev,    /* f32   [n_tok][n_expert] */
        const void *bias_dev,      /* f32   [n_expert] */
        uint32_t n_expert,
        uint32_t k_used,
        float weight_scale,
        uint32_t n_tok) {
    if (n_expert == 0 || n_expert > 256u || k_used == 0 || k_used > n_expert ||
        n_tok == 0) {
        return 0;
    }
    dim3 block(32, 4, 1);
    router_select_kernel<<<(n_tok + 3u) / 4u, block>>>(
            (int32_t *)selected_dev, (float *)weights_dev,
            (const float *)logits_dev, (const float *)bias_dev,
            n_expert, k_used, weight_scale, n_tok);
    return cuda_ok(cudaGetLastError(), "router select launch");
}

/* CPU-reference selftest across Hy3-like and GLM-like shapes. */
static int router_selftest_one(uint32_t n_expert, uint32_t k_used,
                               float scale, uint32_t n_tok) {
    float *logits = (float *)malloc((uint64_t)n_tok * n_expert * sizeof(float));
    float *bias = (float *)malloc((uint64_t)n_expert * sizeof(float));
    int32_t *sel_ref = (int32_t *)malloc((uint64_t)n_tok * k_used * sizeof(int32_t));
    float *w_ref = (float *)malloc((uint64_t)n_tok * k_used * sizeof(float));
    int32_t *sel_gpu = (int32_t *)malloc((uint64_t)n_tok * k_used * sizeof(int32_t));
    float *w_gpu = (float *)malloc((uint64_t)n_tok * k_used * sizeof(float));

    for (uint64_t i = 0; i < (uint64_t)n_tok * n_expert; i++)
        logits[i] = gqa_test_randf() * 4.0f;
    for (uint32_t e = 0; e < n_expert; e++) bias[e] = gqa_test_randf();

    for (uint32_t t = 0; t < n_tok; t++) {
        const float *log = logits + (uint64_t)t * n_expert;
        float prob[256], score[256];
        for (uint32_t e = 0; e < n_expert; e++) {
            prob[e] = 1.0f / (1.0f + expf(-log[e]));
            score[e] = prob[e] + bias[e];
        }
        float sum = 0.0f;
        for (uint32_t k = 0; k < k_used; k++) {
            uint32_t best = UINT32_MAX;
            for (uint32_t e = 0; e < n_expert; e++) {
                if (best == UINT32_MAX || score[e] > score[best]) best = e;
            }
            sel_ref[(uint64_t)t * k_used + k] = (int32_t)best;
            w_ref[(uint64_t)t * k_used + k] = prob[best];
            sum += prob[best];
            score[best] = -INFINITY;
        }
        sum = fmaxf(sum, 6.103515625e-5f);
        for (uint32_t k = 0; k < k_used; k++)
            w_ref[(uint64_t)t * k_used + k] =
                w_ref[(uint64_t)t * k_used + k] / sum * scale;
    }

    void *log_dev = NULL, *bias_dev = NULL, *sel_dev = NULL, *w_dev = NULL;
    const uint64_t log_bytes = (uint64_t)n_tok * n_expert * sizeof(float);
    const uint64_t bias_bytes = (uint64_t)n_expert * sizeof(float);
    const uint64_t sel_bytes = (uint64_t)n_tok * k_used * sizeof(int32_t);
    const uint64_t w_bytes = (uint64_t)n_tok * k_used * sizeof(float);
    int ok = cuda_ok(cudaMalloc(&log_dev, log_bytes), "logits alloc") &&
             cuda_ok(cudaMalloc(&bias_dev, bias_bytes), "bias alloc") &&
             cuda_ok(cudaMalloc(&sel_dev, sel_bytes), "sel alloc") &&
             cuda_ok(cudaMalloc(&w_dev, w_bytes), "w alloc") &&
             cuda_ok(cudaMemcpy(log_dev, logits, log_bytes, cudaMemcpyHostToDevice), "logits h2d") &&
             cuda_ok(cudaMemcpy(bias_dev, bias, bias_bytes, cudaMemcpyHostToDevice), "bias h2d") &&
             pulsar_router_select(sel_dev, w_dev, log_dev, bias_dev,
                                  n_expert, k_used, scale, n_tok) &&
             cuda_ok(cudaDeviceSynchronize(), "sync") &&
             cuda_ok(cudaMemcpy(sel_gpu, sel_dev, sel_bytes, cudaMemcpyDeviceToHost), "sel d2h") &&
             cuda_ok(cudaMemcpy(w_gpu, w_dev, w_bytes, cudaMemcpyDeviceToHost), "w d2h");
    float maxd = 0.0f;
    uint32_t idx_mismatch = 0;
    if (ok) {
        for (uint64_t i = 0; i < (uint64_t)n_tok * k_used; i++) {
            if (sel_gpu[i] != sel_ref[i]) idx_mismatch++;
            float d = fabsf(w_gpu[i] - w_ref[i]);
            if (d > maxd) maxd = d;
        }
        ok = idx_mismatch == 0 && maxd <= 1e-5f;
    }
    fprintf(stderr,
            "router-selftest n_expert=%u k=%u: %s (idx mismatches %u, max w diff %.2e)\n",
            n_expert, k_used, ok ? "PASS" : "FAIL", idx_mismatch, (double)maxd);
    if (log_dev) cudaFree(log_dev);
    if (bias_dev) cudaFree(bias_dev);
    if (sel_dev) cudaFree(sel_dev);
    if (w_dev) cudaFree(w_dev);
    free(logits); free(bias); free(sel_ref); free(w_ref); free(sel_gpu); free(w_gpu);
    return ok;
}

extern "C" int pulsar_router_selftest(void) {
    /* Hy3-like (64 experts, top-8), GLM-like (256, top-8), odd token count */
    return router_selftest_one(64, 8, 2.5f, 7) &&
           router_selftest_one(256, 8, 1.0f, 5) &&
           router_selftest_one(96, 6, 1.5f, 1);
}

extern "C" int pulsar_gqa_selftest(void) { return ds4_gpu_gqa_selftest(); }
