/* qwen4exp (Qwen3.8-Flash-Next) kernels: hyper-connection residual and
 * the PLE n-gram lane. The wide residual is [rows][n_hc*n_embd], stream
 * c at column offset c*n_embd (reference reshape (n_embd, hc, T)). */

/* grouped RMS norm: reduce over ONE stream, scale by the full-width
 * gamma (converter pre-folds 1+w). One block per (row, stream). */
__global__ static void q4e_hc_norm_kernel(
        float *y, const float *x, const float *gamma,
        uint32_t n_embd, uint32_t n_hc, float eps) {
    const uint32_t r = blockIdx.x, c = blockIdx.y;
    const float *xs = x + ((uint64_t)r * n_hc + c) * n_embd;
    float *ys = y + ((uint64_t)r * n_hc + c) * n_embd;
    __shared__ float red[256];
    float acc = 0.0f;
    for (uint32_t i = threadIdx.x; i < n_embd; i += blockDim.x) {
        const float v = xs[i];
        acc += v * v;
    }
    red[threadIdx.x] = acc;
    __syncthreads();
    for (uint32_t s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) red[threadIdx.x] += red[threadIdx.x + s];
        __syncthreads();
    }
    const float inv = rsqrtf(red[0] / (float)n_embd + eps);
    for (uint32_t i = threadIdx.x; i < n_embd; i += blockDim.x) {
        ys[i] = xs[i] * inv * gamma[(uint64_t)c * n_embd + i];
    }
}

extern "C" int pulsar_q4e_hc_norm(
        void *y, const void *x, const void *gamma,
        uint32_t n_embd, uint32_t n_hc, uint32_t rows, float eps) {
    if (!y || !x || !gamma || !n_embd || !n_hc || !rows) return 0;
    dim3 grid(rows, n_hc);
    q4e_hc_norm_kernel<<<grid, 256>>>(
            (float *)y, (const float *)x, (const float *)gamma, n_embd, n_hc, eps);
    return cuda_ok(cudaGetLastError(), "q4e hc norm launch");
}

/* mixed[r][i] = (1/hc) * sum_c xn[r][c][i] * sigmoid(gate_lin[r][c][i]) */
__global__ static void q4e_hc_collapse_kernel(
        float *mixed, const float *xn, const float *gl,
        uint32_t n_embd, uint32_t n_hc, uint32_t rows) {
    const uint64_t idx = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= (uint64_t)rows * n_embd) return;
    const uint32_t r = idx / n_embd, i = idx % n_embd;
    const uint64_t base = (uint64_t)r * n_hc * n_embd + i;
    float acc = 0.0f;
    for (uint32_t c = 0; c < n_hc; c++) {
        const float v = xn[base + (uint64_t)c * n_embd];
        const float g = gl[base + (uint64_t)c * n_embd];
        acc += v / (1.0f + expf(-g));
    }
    mixed[idx] = acc / (float)n_hc;
}

extern "C" int pulsar_q4e_hc_collapse(
        void *mixed, const void *xn, const void *gl,
        uint32_t n_embd, uint32_t n_hc, uint32_t rows) {
    if (!mixed || !xn || !gl || !n_embd || !n_hc || !rows) return 0;
    const uint64_t n = (uint64_t)rows * n_embd;
    q4e_hc_collapse_kernel<<<(unsigned)((n + 255) / 256), 256>>>(
            (float *)mixed, (const float *)xn, (const float *)gl, n_embd, n_hc, rows);
    return cuda_ok(cudaGetLastError(), "q4e hc collapse launch");
}

/* mode 0 (hc combine): res[r][c][i] += out[r][i] * 2*sigmoid(coef[r][c]/n_hc)
 * mode 1 (ple value):  res[r][c][i] += out[r][i] * coef[r][c]  (coef ready) */
__global__ static void q4e_stream_scatter_kernel(
        float *res, const float *out, const float *coef,
        uint32_t n_embd, uint32_t n_hc, uint32_t rows, uint32_t mode) {
    const uint64_t idx = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= (uint64_t)rows * n_hc * n_embd) return;
    const uint32_t i = idx % n_embd;
    const uint32_t c = (idx / n_embd) % n_hc;
    const uint32_t r = idx / ((uint64_t)n_hc * n_embd);
    float w = coef[(uint64_t)r * n_hc + c];
    if (mode == 0u) w = 2.0f / (1.0f + expf(-w / (float)n_hc));
    res[idx] += out[(uint64_t)r * n_embd + i] * w;
}

extern "C" int pulsar_q4e_stream_scatter(
        void *res, const void *out, const void *coef,
        uint32_t n_embd, uint32_t n_hc, uint32_t rows, uint32_t mode) {
    if (!res || !out || !coef || !n_embd || !n_hc || !rows) return 0;
    const uint64_t n = (uint64_t)rows * n_hc * n_embd;
    q4e_stream_scatter_kernel<<<(unsigned)((n + 255) / 256), 256>>>(
            (float *)res, (const float *)out, (const float *)coef,
            n_embd, n_hc, rows, mode);
    return cuda_ok(cudaGetLastError(), "q4e stream scatter launch");
}

/* res[r][c][i] = emb[r][i] : the wide residual starts as hc copies */
__global__ static void q4e_replicate_kernel(
        float *res, const float *emb, uint32_t n_embd, uint32_t n_hc, uint32_t rows) {
    const uint64_t idx = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= (uint64_t)rows * n_hc * n_embd) return;
    const uint32_t i = idx % n_embd;
    const uint32_t r = idx / ((uint64_t)n_hc * n_embd);
    res[idx] = emb[(uint64_t)r * n_embd + i];
}

extern "C" int pulsar_q4e_replicate(
        void *res, const void *emb, uint32_t n_embd, uint32_t n_hc, uint32_t rows) {
    if (!res || !emb || !n_embd || !n_hc || !rows) return 0;
    const uint64_t n = (uint64_t)rows * n_hc * n_embd;
    q4e_replicate_kernel<<<(unsigned)((n + 255) / 256), 256>>>(
            (float *)res, (const float *)emb, n_embd, n_hc, rows);
    return cuda_ok(cudaGetLastError(), "q4e replicate launch");
}

/* x = silu(x * scale), in place */
__global__ static void q4e_silu_scale_kernel(float *x, float scale, uint64_t n) {
    const uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float v = x[i] * scale;
    x[i] = v / (1.0f + expf(-v));
}

extern "C" int pulsar_q4e_silu_scale(void *x, float scale, uint64_t n) {
    if (!x || !n) return 0;
    q4e_silu_scale_kernel<<<(unsigned)((n + 255) / 256), 256>>>((float *)x, scale, n);
    return cuda_ok(cudaGetLastError(), "q4e silu scale launch");
}

/* PLE stream gate: coef[r][c] = sigmoid(sgn(s)*sqrt(clamp(|s|,1e-6,inf))),
 * s = dot(key[r][c][:], query[r][c][:]) / sqrt(n_embd). Block per (r,c). */
__global__ static void q4e_ple_gate_kernel(
        float *coef, const float *keyn, const float *queryn,
        uint32_t n_embd, uint32_t n_hc) {
    const uint32_t r = blockIdx.x, c = blockIdx.y;
    const uint64_t base = ((uint64_t)r * gridDim.y + c) * n_embd;
    __shared__ float red[256];
    float acc = 0.0f;
    for (uint32_t i = threadIdx.x; i < n_embd; i += blockDim.x) {
        acc += keyn[base + i] * queryn[base + i];
    }
    red[threadIdx.x] = acc;
    __syncthreads();
    for (uint32_t s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) red[threadIdx.x] += red[threadIdx.x + s];
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        float s = red[0] * rsqrtf((float)n_embd);
        const float mag = sqrtf(fmaxf(fabsf(s), 1e-6f));
        s = (s < 0.0f ? -mag : mag);
        coef[(uint64_t)r * n_hc + c] = 1.0f / (1.0f + expf(-s));
    }
}

extern "C" int pulsar_q4e_ple_gate(
        void *coef, const void *keyn, const void *queryn,
        uint32_t n_embd, uint32_t n_hc, uint32_t rows) {
    if (!coef || !keyn || !queryn || !n_embd || !n_hc || !rows) return 0;
    dim3 grid(rows, n_hc);
    q4e_ple_gate_kernel<<<grid, 256>>>(
            (float *)coef, (const float *)keyn, (const float *)queryn, n_embd, n_hc);
    return cuda_ok(cudaGetLastError(), "q4e ple gate launch");
}

/* Dilated depthwise causal conv + silu, qwen35_conv_batch shape with a
 * dilation: out[t][c] = silu(sum_j kern[c][j] * x[t - (k-1-j)*dil][c]).
 * state ring holds the previous (k-1)*dil columns, oldest first. */
__global__ static void q4e_conv_dil_kernel(
        float *out, const float *x, const float *kern, float *state,
        uint32_t n_chan, uint32_t k, uint32_t dil, uint32_t n_tok) {
    const uint32_t c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= n_chan) return;
    const float *kc = kern + (uint64_t)c * k;
    const uint32_t hist = (k - 1) * dil; /* <= 9 for kern 4, dil 3 */
    float win[9];
    for (uint32_t t = 0; t < hist; t++) win[t] = state[(uint64_t)t * n_chan + c];
    for (uint32_t tok = 0; tok < n_tok; tok++) {
        const float cur = x[(uint64_t)tok * n_chan + c];
        float acc = kc[k - 1] * cur;
        for (uint32_t j = 0; j + 1 < k; j++) acc += kc[j] * win[j * dil];
        out[(uint64_t)tok * n_chan + c] = acc / (1.0f + expf(-acc));
        for (uint32_t t = 0; t + 1 < hist; t++) win[t] = win[t + 1];
        win[hist - 1] = cur;
    }
    for (uint32_t t = 0; t < hist; t++) state[(uint64_t)t * n_chan + c] = win[t];
}

extern "C" int pulsar_q4e_conv_dil(
        void *out, const void *x, const void *kern, void *state,
        uint32_t n_chan, uint32_t k, uint32_t dil, uint32_t n_tok) {
    if (!out || !x || !kern || !state || !n_chan || k < 2 || k > 4 ||
        !dil || (k - 1) * dil > 9 || !n_tok) {
        return 0;
    }
    q4e_conv_dil_kernel<<<(n_chan + 255u) / 256u, 256>>>(
            (float *)out, (const float *)x, (const float *)kern,
            (float *)state, n_chan, k, dil, n_tok);
    return cuda_ok(cudaGetLastError(), "q4e conv dil launch");
}
