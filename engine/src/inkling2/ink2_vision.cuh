// src/inkling2/ink2_vision.cuh - Inkling-2's image input in strata-ink2: the ViT + projector, and the image rows of a
// prompt.  Included by ink2_engine.cu inside its anonymous namespace, after the prompt path (Prefill, the expert pools).
//
// Model (checkpoint model.visual.*, transformers#51050 Inkling2VisionModel; checked in fp32 against it, see
// terminal/tools/inkling2/vision_oracle.py):
//   * input: the picture's 2 x 2 native patches of 20 x 20 px per image token (ink2_image.hpp), a 1200-vector each;
//     patch_embedding 1200 -> 1024 (no bias) + a learned 32 x 32 position table resampled bilinearly (align_corners
//     False, edge-clamped) to the native patch grid; a class token (class + its position embedding) in front;
//     LayerNorm (ln_pre).  temporal_patch_size is unused for a still picture.
//   * 23 pre-LayerNorm blocks, hidden 1024, 16 heads of 64, biased q/k/v/out, full attention within the picture,
//     2D axial RoPE (theta 1e4, 16 frequencies): channel pair j < 16 turns by (column + 1) * f_j, pair j >= 16 by
//     (row + 1) * f_{j-16}, interleaved pairs; the class token sits at (0, 0).  Layer scale on both branches
//     (x + ls_1 * attn, x + ls_2 * mlp), MLP 1024 -> 4096 -> 1024 with the exact (erf) GELU.  No final norm.
//   * projector: the 4 patch states of a token concatenated (4096) -> proj_in 4096 -> GELU -> proj_out 2304, no bias.
//   * the language model reads these rows in place of the image placeholders' rows AFTER its embedding norm (the
//     reference masked_scatter's them into inputs_embeds = embed_norm(embed(ids))); positions stay plain token indices.
//
// Numerics: fp32 throughout (cuBLAS SGEMM, pedantic math, on fp32 copies of the BF16 weights made per use).
// CBA: attention as SGEMM over query chunks with the scores in VRAM, fine to a few thousand ViT tokens (a 1000-token
// picture: ~0.2 s) but 7 s at the cap (16k ViT tokens, 4096 image tokens); a flash-style TF32 / fp16 kernel if big
// pictures matter.  One picture at a time, no batching across a prompt's pictures.
//
// Memory: nothing at load and nothing kept on the GPU.
// CBA: the weights cross PCIe at every encode (~650 MB, about 90 ms on this box: most of a small picture's cost); a
// resident copy when VRAM is free (no offload) is the upgrade.  The weights are read at the first picture into pinned host
// memory (607 MB BF16 + 6 MB fp32) and streamed to the GPU matrix by matrix during each encode.  An encode's working
// VRAM (weight staging 100 MB + 20 KB per ViT token + 32 MB) comes from the tail of the NVFP4 expert pool when the
// experts are offloaded (those slots are evicted, then handed back empty: the pool refills them on demand), else from
// free VRAM beyond the margin, and goes back as the encode ends.
//
// Protocol (Hy4-Flash's 1.6b): GEN ... image=<path> ... <ids>, one image= key per <|unused_200054|> of the ids in
// prompt order; the engine decodes the files, expands each placeholder to its rows x cols image tokens and encodes
// them before the prompt is read.  INFO images=1 at the start says it can.
#pragma once

constexpr int VH = 1024, VHEADS = 16, VHD = 64, VFF = 4096, VLAYERS = 23, VGRID = 32, VMERGE_IN = 4 * VH, VPROJ = 4096;
constexpr float VEPS = 1e-5f;
constexpr int IMAGE_TOKEN = 200054;   // <|unused_200054|> (processor_config.json image_token)
constexpr size_t VSC = (size_t) 8 << 20;   // scores / MLP scratch, floats (32 MB): attention in query chunks of VSC / S
// small fp32 tensors of one layer, offsets
constexpr int SM_BQ = 0, SM_BK = 1024, SM_BV = 2048, SM_BO = 3072, SM_B1 = 4096, SM_B2 = 8192, SM_LN1W = 9216,
              SM_LN1B = 10240, SM_LN2W = 11264, SM_LN2B = 12288, SM_LS1 = 13312, SM_LS2 = 14336, SM_N = 15360;

struct Vis {
    int max_tokens = 4096;     // --image-max-tokens: per picture (bigger ones are scaled down first)
    std::string dump;          // --dump-image-emb: the image rows of the last encode, float32 [n][2304]
    // weights (pinned host, loaded at the first picture)
    bool loaded = false;
    uint16_t* wb = nullptr;    // the BF16 matrices back to back
    float* ws = nullptr;       // the small fp32 tensors
    size_t nb = 0, ns = 0;
    size_t m_patch = 0, m_in = 0, m_out = 0, m_l[VLAYERS][6] = {};   // matrices (q k v o fc1 fc2) in wb
    size_t s_pos = 0, s_cls = 0, s_lnw = 0, s_lnb = 0, s_l[VLAYERS] = {};
    cublasHandle_t cb = nullptr;
    // the request
    struct Span { int pos, n; uint64_t key; size_t row; };
    std::vector<ink2img::Picture> pics;
    std::vector<Span> spans, hist;   // this prompt's pictures; those of the tokens the engine holds (serve's hist)
    float* emb = nullptr;            // pinned [rows][H]: the image rows of this prompt
    size_t emb_cap = 0;
    bool pending = false;            // the prompt being read has image rows to put in
} g_vis;

double vis_ms() {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

bool vis_available() { return !g_is_gguf && E.ck && E.ck->st.tensors.count("model.visual.proj_out.weight"); }

void vis_load() {
    if (g_vis.loaded) return;
    const double t0 = vis_ms();
    const std::string P = "model.visual.", B = P + "backbone.";
    std::vector<std::pair<std::string, size_t*>> mats = {{B + "embeddings.patch_embedding.weight", &g_vis.m_patch},
                                                         {P + "proj_in.weight", &g_vis.m_in}, {P + "proj_out.weight", &g_vis.m_out}};
    const char* lm[6] = {"self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.out_proj", "mlp.c_fc", "mlp.c_proj"};
    for (int l = 0; l < VLAYERS; ++l)
        for (int j = 0; j < 6; ++j) mats.push_back({B + "layers." + std::to_string(l) + "." + lm[j] + ".weight", &g_vis.m_l[l][j]});
    size_t nb = 0;
    for (auto& m : mats) {
        const StTensor& t = E.ck->t(m.first);
        if (t.dtype != "BF16") { std::fprintf(stderr, "%s: %s, expected BF16\n", m.first.c_str(), t.dtype.c_str()); std::exit(1); }
        *m.second = nb;
        nb += (t.end - t.begin) / 2;
    }
    g_vis.ns = (size_t) VGRID * VGRID * VH + 3 * VH + (size_t) VLAYERS * SM_N;
    CK(cudaHostAlloc((void**) &g_vis.wb, nb * 2, cudaHostAllocDefault));
    CK(cudaHostAlloc((void**) &g_vis.ws, g_vis.ns * 4, cudaHostAllocDefault));
    g_vis.nb = nb;
    for (auto& m : mats) {
        const auto b = E.ck->bytes(m.first);
        std::memcpy(g_vis.wb + *m.second, b.data(), b.size());
    }
    size_t o = 0;
    auto put = [&](const std::string& name, size_t want) {
        const vec v = E.ck->f32(name);
        if (v.size() != want) { std::fprintf(stderr, "%s: %zu values, expected %zu\n", name.c_str(), v.size(), want); std::exit(1); }
        std::memcpy(g_vis.ws + o, v.data(), want * 4);
        o += want;
    };
    g_vis.s_pos = o; put(B + "embeddings.position_embedding", (size_t) VGRID * VGRID * VH);
    g_vis.s_cls = o; put(B + "embeddings.class_embedding", VH);
    {   // the class token's input: class_embedding + cls_position_embedding
        const vec c = E.ck->f32(B + "embeddings.cls_position_embedding");
        for (int i = 0; i < VH; ++i) g_vis.ws[g_vis.s_cls + i] += c[i];
    }
    g_vis.s_lnw = o; put(B + "ln_pre.weight", VH);
    g_vis.s_lnb = o; put(B + "ln_pre.bias", VH);
    const char* sm[12] = {"self_attn.q_proj.bias", "self_attn.k_proj.bias", "self_attn.v_proj.bias", "self_attn.out_proj.bias",
                          "mlp.c_fc.bias", "mlp.c_proj.bias", "ln_1.weight", "ln_1.bias", "ln_2.weight", "ln_2.bias",
                          "ls_1.gamma", "ls_2.gamma"};
    for (int l = 0; l < VLAYERS; ++l) {
        g_vis.s_l[l] = o;
        for (int j = 0; j < 12; ++j) put(B + "layers." + std::to_string(l) + "." + sm[j], j == 4 ? VFF : VH);
    }
    CK(cublasCreate(&g_vis.cb) == CUBLAS_STATUS_SUCCESS ? cudaSuccess : cudaErrorUnknown);
    cublasSetStream(g_vis.cb, cudaStreamPerThread);
    cublasSetMathMode(g_vis.cb, CUBLAS_PEDANTIC_MATH);
    g_vis.loaded = true;
    std::fprintf(stderr, "[vision] weights read: %.0f MB BF16 + %.1f MB fp32 in pinned host memory, %.2f s\n", nb * 2 / 1e6,
                 g_vis.ns * 4 / 1e6, (vis_ms() - t0) / 1000);
}

// ---------------------------------------------------------------------------------------------------------------- kernels
__global__ void kv_bf16_f32(const uint16_t* __restrict__ s, float* __restrict__ d, size_t n) {
    for (size_t i = blockIdx.x * (size_t) blockDim.x + threadIdx.x; i < n; i += (size_t) gridDim.x * blockDim.x)
        d[i] = __uint_as_float((uint32_t) s[i] << 16);
}
__device__ __forceinline__ float kv_block_sum(float v, float* sh) {   // blockDim 256
    v = warp_sum(v);
    if ((threadIdx.x & 31) == 0) sh[threadIdx.x >> 5] = v;
    __syncthreads();
    float t = threadIdx.x < 8 ? sh[threadIdx.x] : 0.0f;
    if (threadIdx.x < 32) t = warp_sum(t);
    if (threadIdx.x == 0) sh[8] = t;
    __syncthreads();
    const float r = sh[8];
    __syncthreads();
    return r;
}
// LayerNorm over rows of 1024, block 256
__global__ void kv_ln(const float* __restrict__ x, const float* __restrict__ w, const float* __restrict__ b, float* __restrict__ y) {
    __shared__ float sh[9];
    const float* r = x + (size_t) blockIdx.x * VH;
    float v[4], s = 0.0f;
    for (int j = 0; j < 4; ++j) { v[j] = r[threadIdx.x + 256 * j]; s += v[j]; }
    const float mean = kv_block_sum(s, sh) / VH;
    float q = 0.0f;
    for (int j = 0; j < 4; ++j) { v[j] -= mean; q += v[j] * v[j]; }
    const float inv = rsqrtf(kv_block_sum(q, sh) / VH + VEPS);
    float* o = y + (size_t) blockIdx.x * VH;
    for (int j = 0; j < 4; ++j) { const int i = threadIdx.x + 256 * j; o[i] = v[j] * inv * w[i] + b[i]; }
}
// softmax of rows of n, block 256
__global__ void kv_softmax(float* __restrict__ s, int n) {
    __shared__ float sh[9];
    float* r = s + (size_t) blockIdx.x * n;
    float m = -INFINITY;
    for (int i = threadIdx.x; i < n; i += 256) m = fmaxf(m, r[i]);
    m = warp_max(m);
    if ((threadIdx.x & 31) == 0) sh[threadIdx.x >> 5] = m;
    __syncthreads();
    if (threadIdx.x == 0) { float t = sh[0]; for (int k = 1; k < 8; ++k) t = fmaxf(t, sh[k]); sh[8] = t; }
    __syncthreads();
    m = sh[8];
    __syncthreads();
    float z = 0.0f;
    for (int i = threadIdx.x; i < n; i += 256) { const float e = expf(r[i] - m); r[i] = e; z += e; }
    const float inv = 1.0f / kv_block_sum(z, sh);
    for (int i = threadIdx.x; i < n; i += 256) r[i] *= inv;
}
// (row, col) of native patch i (block order: 2 x 2 per token, tokens in raster order over `cols` tokens a row)
__device__ __forceinline__ void kv_patch_rc(int i, int cols, int& row, int& col) {
    const int t = i >> 2;
    row = (t / cols) * 2 + ((i >> 1) & 1);
    col = (t % cols) * 2 + (i & 1);
}
// one axis of the position table's bilinear resampling (align_corners False, edge clamp), as transformers'
// _interpolation_axis_taps_weights
__device__ __forceinline__ void kv_taps(int idx, int size, int& t0, int& t1, float& w0, float& w1) {
    const float src = ((float) idx + 0.5f) * (float) VGRID / (float) size - 0.5f, f = floorf(src);
    t0 = min(max((int) f, 0), VGRID - 1);
    t1 = min(max((int) f + 1, 0), VGRID - 1);
    w0 = fmaxf(1.0f - fabsf(src - f), 0.0f);
    w1 = fmaxf(1.0f - fabsf(src - f - 1.0f), 0.0f);
}
// x[0] = class token; x[1 + i] += position embedding of patch i.  grid N + 1, block 256
__global__ void kv_embed(float* __restrict__ x, const float* __restrict__ pos, const float* __restrict__ cls, int rows, int cols) {
    if (blockIdx.x == 0) {
        for (int d = threadIdx.x; d < VH; d += 256) x[d] = cls[d];
        return;
    }
    int r, c, h0, h1, w0i, w1i;
    float hw0, hw1, ww0, ww1;
    kv_patch_rc(blockIdx.x - 1, cols, r, c);
    kv_taps(r, 2 * rows, h0, h1, hw0, hw1);
    kv_taps(c, 2 * cols, w0i, w1i, ww0, ww1);
    const float* p00 = pos + (size_t) (h0 * VGRID + w0i) * VH;
    const float* p01 = pos + (size_t) (h0 * VGRID + w1i) * VH;
    const float* p10 = pos + (size_t) (h1 * VGRID + w0i) * VH;
    const float* p11 = pos + (size_t) (h1 * VGRID + w1i) * VH;
    float* o = x + (size_t) blockIdx.x * VH;
    for (int d = threadIdx.x; d < VH; d += 256)
        o[d] += p00[d] * (hw0 * ww0) + p01[d] * (hw0 * ww1) + p10[d] * (hw1 * ww0) + p11[d] * (hw1 * ww1);
}
// q, k += bias, then the 2D RoPE (interleaved pairs; pair j < 16: column, j >= 16: row; class token at (0, 0)).
// grid S, block 512 (16 heads x 32 pairs)
__global__ void kv_qk_rope(float* __restrict__ q, float* __restrict__ k, float* __restrict__ v, const float* __restrict__ bq,
                           const float* __restrict__ bk, const float* __restrict__ bv, int cols) {
    const int s = blockIdx.x, hd = threadIdx.x >> 5, j = threadIdx.x & 31;
    int r = -1, c = -1;
    if (s > 0) kv_patch_rc(s - 1, cols, r, c);
    const float p = (float) (j < 16 ? c + 1 : r + 1);
    const int jj = j & 15;
    const float inv = 1.0f / powf(10000.0f, (float) (2 * jj) / 32.0f);
    const float a = p * inv, cs = cosf(a), sn = sinf(a);
    const size_t i = (size_t) s * VH + hd * VHD + 2 * j;
    float x0 = q[i] + bq[hd * VHD + 2 * j], x1 = q[i + 1] + bq[hd * VHD + 2 * j + 1];
    q[i] = x0 * cs - x1 * sn;
    q[i + 1] = x1 * cs + x0 * sn;
    x0 = k[i] + bk[hd * VHD + 2 * j];
    x1 = k[i + 1] + bk[hd * VHD + 2 * j + 1];
    k[i] = x0 * cs - x1 * sn;
    k[i + 1] = x1 * cs + x0 * sn;
    v[i] += bv[hd * VHD + 2 * j];
    v[i + 1] += bv[hd * VHD + 2 * j + 1];
}
// x += ls * (y + b), rows of 1024
__global__ void kv_resid(float* __restrict__ x, const float* __restrict__ y, const float* __restrict__ b,
                         const float* __restrict__ ls, size_t n) {
    for (size_t i = blockIdx.x * (size_t) blockDim.x + threadIdx.x; i < n; i += (size_t) gridDim.x * blockDim.x) {
        const int d = (int) (i % VH);
        x[i] += ls[d] * (y[i] + b[d]);
    }
}
// y = gelu(y + b) (exact erf form), rows of `w` (b may be null)
__global__ void kv_gelu(float* __restrict__ y, const float* __restrict__ b, int w, size_t n) {
    for (size_t i = blockIdx.x * (size_t) blockDim.x + threadIdx.x; i < n; i += (size_t) gridDim.x * blockDim.x) {
        const float v = y[i] + (b ? b[i % w] : 0.0f);
        y[i] = 0.5f * v * (1.0f + erff(v * 0.70710678118654752f));
    }
}
unsigned kv_grid(size_t n) { return (unsigned) std::min<size_t>((n + 255) / 256, 8192); }

// -------------------------------------------------------------------------------------------------- the encode's VRAM
struct VisArena { uint8_t* p = nullptr; size_t bytes = 0; int pool_saved = -1; };
void vis_graphs_drop() {   // the decode graph and the verify graphs bake the slot counts in (as pool_resize)
    if (E.exec) { CK(cudaGraphExecDestroy(E.exec)); E.exec = nullptr; }
    for (auto& x : g_vexec) if (x) { CK(cudaGraphExecDestroy(x)); x = nullptr; }
}
std::string vis_arena_get(VisArena& a, size_t bytes) {
    CK(cudaDeviceSynchronize());
    a.bytes = bytes;
    if (E.offload) {   // the NVFP4 pool's last slots, evicted (as pool_resize shrinking, without giving the memory up)
        const size_t eb = ex_bytes(0);
        const int k = (int) ((bytes + eb - 1) / eb), ns = E.nslot_nv, n = ns - k;
        if (n < 2 * FETCH_MAX)
            return "the picture needs " + std::to_string(bytes >> 20) + " MiB of VRAM for its encoder, more than the expert pool can lend";
        if (E.cpu) promote_cancel();
        k_pool_evict<<<(k + 255) / 256, 256>>>(E.owner_nv, E.slot_of, E.tabA, E.tabB, E.stamp_nv, n, ns);
        CK(cudaDeviceSynchronize());
        a.pool_saved = ns;
        E.nslot_nv = n;
        vis_graphs_drop();
        a.p = E.pool_nv + (size_t) n * eb;
        std::fprintf(stderr, "[vision] %zu MiB lent by the expert pool (%d NVFP4 slots)\n", bytes >> 20, k);
        return "";
    }
    const size_t fr = vram_free(), keep = vram_keep();
    if (fr < bytes + keep || cudaMalloc((void**) &a.p, bytes) != cudaSuccess) {
        (void) cudaGetLastError();
        a.p = nullptr;
        return "the picture needs " + std::to_string(bytes >> 20) + " MiB of VRAM for its encoder, " +
               std::to_string(fr > keep ? (fr - keep) >> 20 : 0) + " MiB are free";
    }
    return "";
}
void vis_arena_put(VisArena& a) {
    CK(cudaDeviceSynchronize());
    if (a.pool_saved >= 0) {   // the slots come back empty (owner -1, stamp 0): the pool fills them on demand
        E.nslot_nv = a.pool_saved;
        vis_graphs_drop();
    } else if (a.p) {
        CK(cudaFree(a.p));
    }
    a = VisArena();
}

// -------------------------------------------------------------------------------------------------------- the ViT
// one picture -> its rows x cols image rows [n][H] (host)
std::string vis_run(const ink2img::Picture& pic, float* out) {
    vis_load();
    const int NT = pic.rows * pic.cols, N = 4 * NT, S = N + 1;
    const size_t SH = (size_t) S * VH, WMAX = (size_t) VPROJ * VMERGE_IN, WS_BYTES = (size_t) 4 << 20;
    auto al = [](size_t b) { return (b + 255) / 256 * 256; };
    const size_t b_small = al(g_vis.ns * 4), b_stage = al(WMAX * 2), b_wf = al(WMAX * 4), b_act = al((5 * SH + VSC) * 4);
    VisArena A;
    const size_t free0 = vram_free();
    std::string err = vis_arena_get(A, b_small + b_stage + b_wf + WS_BYTES + b_act);
    if (!err.empty()) return err;
    const double t0 = vis_ms();
    uint8_t* p = A.p;
    float* sm = (float*) p;           p += b_small;
    uint16_t* stg = (uint16_t*) p;    p += b_stage;
    float* wf = (float*) p;           p += b_wf;
    void* cws = p;                    p += WS_BYTES;
    float* x = (float*) p;
    float *y = x + SH, *q = y + SH, *k = q + SH, *v = k + SH, *sc = v + SH;
    cublasSetWorkspace(g_vis.cb, cws, WS_BYTES);
    const cudaStream_t st = cudaStreamPerThread;
    CK(cudaMemcpyAsync(sm, g_vis.ws, g_vis.ns * 4, cudaMemcpyHostToDevice, st));
    size_t h2d = g_vis.ns * 4;
    auto wmat = [&](size_t off, size_t n, float* dst) {   // a BF16 matrix -> fp32 at dst
        CK(cudaMemcpyAsync(stg, g_vis.wb + off, n * 2, cudaMemcpyHostToDevice, st));
        kv_bf16_f32<<<kv_grid(n), 256, 0, st>>>(stg, dst, n);
        h2d += n * 2;
    };
    const float one = 1.0f, zero = 0.0f;
    bool ok = true;
    // Y[T][n] = X[T][K] . W[n][K]^T (row-major; ldw: W's row stride)
    auto gemm = [&](int T, int n, int K, const float* X, int ldx, const float* W, int ldw, float* Y, int ldy, float alpha) {
        if (cublasSgemm(g_vis.cb, CUBLAS_OP_T, CUBLAS_OP_N, n, T, K, &alpha, W, ldw, X, ldx, &zero, Y, ldy) != CUBLAS_STATUS_SUCCESS)
            ok = false;
    };
    // the patches -> x[1..N], the class token at x[0], the position table, ln_pre
    CK(cudaMemcpyAsync(k, pic.patches.data(), (size_t) N * ink2img::PIN * 4, cudaMemcpyHostToDevice, st));
    h2d += (size_t) N * ink2img::PIN * 4;
    wmat(g_vis.m_patch, (size_t) VH * ink2img::PIN, wf);
    gemm(N, VH, ink2img::PIN, k, ink2img::PIN, wf, ink2img::PIN, x + VH, VH, one);
    kv_embed<<<S, 256, 0, st>>>(x, sm + g_vis.s_pos, sm + g_vis.s_cls, pic.rows, pic.cols);
    kv_ln<<<S, 256, 0, st>>>(x, sm + g_vis.s_lnw, sm + g_vis.s_lnb, y);
    std::swap(x, y);
    const int qc_max = (int) std::max<size_t>(1, std::min<size_t>(S, VSC / S)), mc = (int) (VSC / VFF);
    const size_t M1 = (size_t) VH * VH, M4 = (size_t) VFF * VH;
    for (int l = 0; l < VLAYERS; ++l) {
        const float* s = sm + g_vis.s_l[l];
        float *wq = wf, *wk = wf + M1, *wv = wf + 2 * M1, *wo = wf + 3 * M1, *w1 = wf + 4 * M1, *w2 = w1 + M4;
        wmat(g_vis.m_l[l][0], M1, wq); wmat(g_vis.m_l[l][1], M1, wk); wmat(g_vis.m_l[l][2], M1, wv);
        wmat(g_vis.m_l[l][3], M1, wo); wmat(g_vis.m_l[l][4], M4, w1); wmat(g_vis.m_l[l][5], M4, w2);
        kv_ln<<<S, 256, 0, st>>>(x, s + SM_LN1W, s + SM_LN1B, y);
        gemm(S, VH, VH, y, VH, wq, VH, q, VH, one);
        gemm(S, VH, VH, y, VH, wk, VH, k, VH, one);
        gemm(S, VH, VH, y, VH, wv, VH, v, VH, one);
        kv_qk_rope<<<S, 512, 0, st>>>(q, k, v, s + SM_BQ, s + SM_BK, s + SM_BV, pic.cols);
        for (int h = 0; h < VHEADS; ++h)   // attention -> y (the heads' outputs side by side)
            for (int q0 = 0; q0 < S; q0 += qc_max) {
                const int qc = std::min(qc_max, S - q0);
                gemm(qc, S, VHD, q + (size_t) q0 * VH + h * VHD, VH, k + h * VHD, VH, sc, S, 0.125f);   // 1 / sqrt(64)
                kv_softmax<<<qc, 256, 0, st>>>(sc, S);
                if (cublasSgemm(g_vis.cb, CUBLAS_OP_N, CUBLAS_OP_N, VHD, qc, S, &one, v + h * VHD, VH, sc, S, &zero,
                                y + (size_t) q0 * VH + h * VHD, VH) != CUBLAS_STATUS_SUCCESS) ok = false;
            }
        gemm(S, VH, VH, y, VH, wo, VH, q, VH, one);
        kv_resid<<<kv_grid(SH), 256, 0, st>>>(x, q, s + SM_BO, s + SM_LS1, SH);
        kv_ln<<<S, 256, 0, st>>>(x, s + SM_LN2W, s + SM_LN2B, y);
        for (int r0 = 0; r0 < S; r0 += mc) {
            const int m = std::min(mc, S - r0);
            gemm(m, VFF, VH, y + (size_t) r0 * VH, VH, w1, VH, sc, VFF, one);
            kv_gelu<<<kv_grid((size_t) m * VFF), 256, 0, st>>>(sc, s + SM_B1, VFF, (size_t) m * VFF);
            gemm(m, VH, VFF, sc, VFF, w2, VFF, q + (size_t) r0 * VH, VH, one);
        }
        kv_resid<<<kv_grid(SH), 256, 0, st>>>(x, q, s + SM_B2, s + SM_LS2, SH);
    }
    // projector: the 4 patch rows of a token side by side are x[1 + 4t .. 1 + 4t + 3] (block order), already contiguous
    wmat(g_vis.m_in, (size_t) VPROJ * VMERGE_IN, wf);
    gemm(NT, VPROJ, VMERGE_IN, x + VH, VMERGE_IN, wf, VMERGE_IN, q, VPROJ, one);
    kv_gelu<<<kv_grid((size_t) NT * VPROJ), 256, 0, st>>>(q, nullptr, VPROJ, (size_t) NT * VPROJ);
    wmat(g_vis.m_out, (size_t) H * VPROJ, wf);
    gemm(NT, H, VPROJ, q, VPROJ, wf, VPROJ, k, H, one);
    CK(cudaMemcpyAsync(out, k, (size_t) NT * H * 4, cudaMemcpyDeviceToHost, st));
    CK(cudaStreamSynchronize(st));
    const double t1 = vis_ms();
    const size_t arena = A.bytes;
    vis_arena_put(A);
    if (!ok) return "cuBLAS failed in the picture encoder";
    const size_t free1 = vram_free();
    std::fprintf(stderr, "[vision] %dx%d px (decoded %dx%d) -> %d x %d image tokens, ViT %d tokens: %.0f ms, %.0f MiB VRAM, "
                         "%.0f MB weights over PCIe; VRAM free %.0f MiB before, %.0f after\n", pic.w, pic.h, pic.src_w, pic.src_h,
                 pic.rows, pic.cols, S, t1 - t0, arena / 1048576.0, h2d / 1e6, free0 / 1048576.0, free1 / 1048576.0);
    return "";
}

// ------------------------------------------------------------------------------------------------------- the prompt
// This prompt's pictures: each IMAGE_TOKEN of `ids` (one per picture, prompt order) becomes its rows x cols image
// tokens.  `paths` empty and no placeholder: a text prompt (nothing to do).  "" or the error.
std::string vis_prepare(std::vector<int>& ids, const std::vector<std::string>& paths, bool pixels = false) {
    g_vis.spans.clear();
    g_vis.pics.clear();
    g_vis.pending = false;
    const size_t nph = (size_t) std::count(ids.begin(), ids.end(), IMAGE_TOKEN);
    if (paths.empty() && nph == 0) return "";
    if (!vis_available()) return "this model has no image encoder (model.visual.*; a GGUF file is text only here)";
    if (nph != paths.size())
        return std::to_string(paths.size()) + " picture(s) for " + std::to_string(nph) + " image placeholder(s) in the prompt";
    g_vis.pics.resize(paths.size());
    for (size_t i = 0; i < paths.size(); ++i) {
        const std::string e = pixels ? ink2img::load_pixels(paths[i], g_vis.pics[i])
                                     : ink2img::load_picture(paths[i], g_vis.max_tokens, g_vis.pics[i]);
        if (!e.empty()) return e;
    }
    std::vector<int> out;
    out.reserve(ids.size());
    size_t k = 0, row = 0;
    for (int t : ids) {
        if (t != IMAGE_TOKEN) { out.push_back(t); continue; }
        const ink2img::Picture& p = g_vis.pics[k++];
        const int n = p.rows * p.cols;
        g_vis.spans.push_back({(int) out.size(), n, p.key, row});
        row += (size_t) n;
        out.insert(out.end(), (size_t) n, IMAGE_TOKEN);
    }
    ids.swap(out);
    return "";
}

// The reused prefix (the first `start` tokens, already in the engine's state) holds the same pictures as this prompt's
// at the same places, else nothing is reused: two different pictures of the same size have the same token ids.
// CBA: one history (g_vis.hist, set by vis_done); a cache of several sessions needs each session's own spans.
size_t vis_reuse(size_t start) {
    if (!start) return 0;
    auto in = [start](const std::vector<Vis::Span>& v) {
        std::vector<uint64_t> r;
        for (const auto& s : v) if ((size_t) s.pos < start) { r.push_back((uint64_t) s.pos); r.push_back((uint64_t) s.n); r.push_back(s.key); }
        return r;
    };
    return in(g_vis.spans) == in(g_vis.hist) ? start : 0;
}

// Encodes the pictures whose rows are read from `start` on (the others are in the reused prefix).  "" or the error.
std::string vis_encode(size_t start) {
    if (g_vis.spans.empty()) return "";
    size_t rows = 0;
    for (const auto& s : g_vis.spans) rows += (size_t) s.n;
    if (rows > g_vis.emb_cap) {
        if (g_vis.emb) CK(cudaFreeHost(g_vis.emb));
        CK(cudaHostAlloc((void**) &g_vis.emb, rows * H * 4, cudaHostAllocDefault));
        g_vis.emb_cap = rows;
    }
    for (size_t i = 0; i < g_vis.spans.size(); ++i) {
        const Vis::Span& s = g_vis.spans[i];
        if ((size_t) (s.pos + s.n) <= start) continue;
        const std::string e = vis_run(g_vis.pics[i], g_vis.emb + s.row * H);
        if (!e.empty()) return e;
    }
    g_vis.pics.clear();   // (the patches: up to tens of MB a picture)
    if (!g_vis.dump.empty()) {
        if (FILE* f = std::fopen(g_vis.dump.c_str(), "wb")) {
            std::fwrite(g_vis.emb, 4, rows * H, f);
            std::fclose(f);
        }
    }
    g_vis.pending = true;
    return "";
}

// forward_prefill_device: the chunk [pos0, pos0 + T)'s image rows over its normed embeddings (PF.h)
void vis_inject(int T, int pos0) {
    if (!g_vis.pending || pos0 < 0) return;
    for (const auto& s : g_vis.spans) {
        const int a = std::max(pos0, s.pos), b = std::min(pos0 + T, s.pos + s.n);
        if (a >= b) continue;
        CK(cudaMemcpyAsync(PF.h + (size_t) (a - pos0) * H, g_vis.emb + (s.row + (size_t) (a - s.pos)) * H,
                           (size_t) (b - a) * H * 4, cudaMemcpyHostToDevice, cudaStreamPerThread));
    }
}

// a request is done: its pictures are those of the engine's state from now on
void vis_done() {
    g_vis.hist = g_vis.spans;
    g_vis.pending = false;
}

// --vis-pixels FILE: the ViT alone on the processor's pixel tensor (no language model loaded), rows to --dump-image-emb
int vis_cli_pixels(const std::string& path) {
    std::vector<int> ids = {IMAGE_TOKEN};
    std::string e = vis_prepare(ids, {path}, true);
    if (e.empty()) e = vis_encode(0);
    if (!e.empty()) { std::fprintf(stderr, "%s\n", e.c_str()); return 1; }
    std::printf("vision: %zu image rows%s%s\n", ids.size(), g_vis.dump.empty() ? "" : " written to ", g_vis.dump.c_str());
    return 0;
}
