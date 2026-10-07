// src/glm/model.cpp - GLM-5.3 weights, KV cache and the forward pass.  See the header for the math.
#include "strata/glm/model.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdio>
#include <cstring>
#include <mutex>
#include <thread>

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

namespace strata::glm {
namespace {

double now_ms() {
    using namespace std::chrono;
    return duration<double, std::milli>(steady_clock::now().time_since_epoch()).count();
}

uint64_t available_ram() {
#if defined(_WIN32)
    MEMORYSTATUSEX m{};
    m.dwLength = sizeof m;
    if (GlobalMemoryStatusEx(&m)) return m.ullAvailPhys;
    return 0;
#else
    FILE* f = std::fopen("/proc/meminfo", "r");
    if (!f) return 0;
    char line[256];
    unsigned long long kb = 0;
    while (std::fgets(line, sizeof line, f))
        if (std::sscanf(line, "MemAvailable: %llu kB", &kb) == 1) break;
    std::fclose(f);
    return kb * 1024ull;
#endif
}

int physical_cores() {
#if defined(_WIN32)
    DWORD len = 0;
    GetLogicalProcessorInformationEx(RelationProcessorCore, nullptr, &len);
    std::vector<uint8_t> buf(len);
    int n = 0;
    if (len && GetLogicalProcessorInformationEx(RelationProcessorCore, (PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX) buf.data(), &len)) {
        for (DWORD off = 0; off < len;) {
            auto* p = (PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX) (buf.data() + off);
            if (p->Relationship == RelationProcessorCore) ++n;
            off += p->Size;
        }
    }
    if (n > 0) return n;
#endif
    const unsigned h = std::thread::hardware_concurrency();
    return h > 1 ? (int) h / 2 : 1;
}

inline float sigmoidf(float x) { return 1.0f / (1.0f + std::exp(-x)); }

}  // namespace

GlmModel::~GlmModel() {
    experts_.reset();
    io_.reset();
    for (void* b : blocks_) aligned_free_bytes(b);
}

// ---------------------------------------------------------------------------------------------------- loading

// The dense reads are queued on the I/O pool as they are named, and `load` waits for all of them at the end: the
// pointer into a block is known before its bytes arrive.
struct GlmModel::Pending {
    std::mutex m;
    std::condition_variable cv;
    int left = 0;
    bool failed = false;
};

const uint8_t* GlmModel::load_raw(const std::string& name, uint64_t expect_bytes, const char* dtype, std::string& err) {
    const TensorSpan* t = ct_.find(name);
    if (!t) { err = "missing tensor " + name; return nullptr; }
    if (t->dtype != dtype || (expect_bytes && t->bytes != expect_bytes)) {
        err = name + ": expected " + dtype + " of " + std::to_string(expect_bytes) + " B, the container has " + t->dtype +
              " of " + std::to_string(t->bytes) + " B";
        return nullptr;
    }
    const AlignedSpan a = aligned_span(t->offset, t->bytes);
    void* mem = aligned_alloc_bytes((size_t) a.len);
    if (!mem) { err = "out of memory loading " + name; return nullptr; }
    blocks_.push_back(mem);
    dense_bytes_ += t->bytes;
    auto p = pending_;
    {
        std::lock_guard<std::mutex> lk(p->m);
        ++p->left;
    }
    io_->submit(ReadJob{t->shard, a.off, a.len, mem, [p](bool ok) {
                            std::lock_guard<std::mutex> lk(p->m);
                            if (!ok) p->failed = true;
                            if (--p->left == 0) p->cv.notify_all();
                        }});
    return (const uint8_t*) mem + a.skip;
}

bool GlmModel::load_q4(const std::string& name, int O, int I, Q4& out, std::string& err) {
    if (I % kGroup) { err = name + ": input width is not a multiple of 64"; return false; }
    const uint8_t* codes = load_raw(name, (uint64_t) O * I / 2, "U8", err);
    if (!codes) return false;
    const uint8_t* scales = load_raw(name + ".qs", (uint64_t) O * (I / kGroup) * 4, "F32", err);
    if (!scales) return false;
    out = Q4{O, I, codes, (const float*) scales};
    return true;
}

const float* GlmModel::load_f32(const std::string& name, uint64_t n, std::string& err) {
    return (const float*) load_raw(name, n * 4, "F32", err);
}

bool GlmModel::load(const std::string& dir, const ModelOptions& opt, std::string& err) {
    const double t0 = now_ms();
    if (!cpu_has_avx2()) { err = "this engine needs a CPU with AVX2 and FMA"; return false; }
    if (!load_config(dir, c_, err)) return false;
    if (!ct_.open(dir, err)) return false;
    const GlmConfig& c = c_;
    const int D = c.hidden, H = c.n_heads;
    if (c.kv_lora > 512 || c.qk_rope > 256) { err = "kv_lora_rank > 512 or qk_rope_head_dim > 256 is not supported"; return false; }

    pool_ = std::make_unique<Pool>(opt.threads > 0 ? opt.threads : physical_cores());
    io_ = std::make_unique<IoPool>(ct_.shard_paths(), std::max(1, opt.io_threads));

    pending_ = std::make_shared<Pending>();
    Pending& pending = *pending_;
    // embed_tokens and lm_head: int8, one scale per row
    {
        const uint8_t* ec = load_raw("model.embed_tokens.weight", (uint64_t) c.vocab * D, "U8", err);
        const uint8_t* es = ec ? load_raw("model.embed_tokens.weight.qs", (uint64_t) c.vocab * 4, "F32", err) : nullptr;
        const uint8_t* hc = es ? load_raw("lm_head.weight", (uint64_t) c.vocab * D, "U8", err) : nullptr;
        const uint8_t* hs = hc ? load_raw("lm_head.weight.qs", (uint64_t) c.vocab * 4, "F32", err) : nullptr;
        if (!hs) return false;
        embed_ = Q8R{c.vocab, D, ec, (const float*) es};
        lm_head_ = Q8R{c.vocab, D, hc, (const float*) hs};
        final_norm_ = load_f32("model.norm.weight", D, err);
        if (!final_norm_) return false;
    }
    layers_.assign(c.n_layers, LayerW{});
    std::vector<std::vector<ExpertSpan>> spans(c.n_layers);
    const int sI = c.moe_inter * std::max(1, c.n_shared);
    bool ok = true;
    for (int l = 0; l < c.n_layers && ok; ++l) {
        LayerW& L = layers_[l];
        const std::string p = "model.layers." + std::to_string(l) + ".";
        ok = (L.in_ln = load_f32(p + "input_layernorm.weight", D, err)) &&
             (L.post_ln = load_f32(p + "post_attention_layernorm.weight", D, err)) &&
             (L.q_a_ln = load_f32(p + "self_attn.q_a_layernorm.weight", c.q_lora, err)) &&
             (L.kv_a_ln = load_f32(p + "self_attn.kv_a_layernorm.weight", c.kv_lora, err)) &&
             load_q4(p + "self_attn.q_a_proj.weight", c.q_lora, D, L.q_a, err) &&
             load_q4(p + "self_attn.q_b_proj.weight", H * c.qk_head(), c.q_lora, L.q_b, err) &&
             load_q4(p + "self_attn.kv_a_proj_with_mqa.weight", c.kv_lora + c.qk_rope, D, L.kv_a, err) &&
             load_q4(p + "self_attn.kv_b_proj.weight", H * (c.qk_nope + c.v_head), c.kv_lora, L.kv_b, err) &&
             load_q4(p + "self_attn.o_proj.weight", D, H * c.v_head, L.o, err);
        if (!ok) break;
        L.sparse = c.sparse(l);
        if (!L.sparse) {
            ok = load_q4(p + "mlp.gate_proj.weight", c.dense_inter, D, L.gate, err) &&
                 load_q4(p + "mlp.up_proj.weight", c.dense_inter, D, L.up, err) &&
                 load_q4(p + "mlp.down_proj.weight", D, c.dense_inter, L.down, err);
            continue;
        }
        ok = (L.router = load_f32(p + "mlp.gate.weight", (uint64_t) c.n_experts * D, err)) &&
             (L.router_bias = load_f32(p + "mlp.gate.e_score_correction_bias", c.n_experts, err));
        if (ok && c.n_shared > 0)
            ok = load_q4(p + "mlp.shared_experts.gate_proj.weight", sI, D, L.sh_gate, err) &&
                 load_q4(p + "mlp.shared_experts.up_proj.weight", sI, D, L.sh_up, err) &&
                 load_q4(p + "mlp.shared_experts.down_proj.weight", D, sI, L.sh_down, err);
        if (!ok) break;
        spans[l].resize(c.n_experts);
        for (int e = 0; e < c.n_experts && ok; ++e) ok = expert_span(ct_, c, l, e, spans[l][e], err);
    }
    {
        std::unique_lock<std::mutex> lk(pending.m);
        pending.cv.wait(lk, [&] { return pending.left == 0; });
    }
    if (!ok) return false;
    if (pending.failed) { err = "a read of the dense weights failed"; return false; }

    rope_.init(c.qk_rope, c.rope_theta);
    max_ctx_ = std::max(16, opt.max_context);
    kv_.reset(c.n_layers, max_ctx_, c.kv_lora, c.qk_rope, opt.kv);
    n_past_ = 0;
    prefetch_ = opt.prefetch;
    gpu_prefill_ = opt.gpu_prefill;
    if (opt.gpu) {
#ifdef STRATA_GLM_CUDA
        gpu_ = std::make_unique<CudaBackend>();
        if (!gpu_->init(opt.device, D, c.moe_inter, opt.vram_budget, opt.vram_reserve, opt.promote_after, err)) return false;
        if (opt.verbose) std::fprintf(stderr, "strata-glm: CUDA device %d, %d expert slots (%.2f GB), prefill %s\n",
            opt.device, gpu_->slots(), gpu_->bytes() / 1e9, gpu_prefill_ ? "GPU" : "CPU");
#else
        err = "this strata-glm was built without CUDA; rebuild with -DSTRATA_ENABLE_CUDA=ON or omit --gpu";
        return false;
#endif
    }

    // what is left of the RAM budget goes to the expert cache
    const uint64_t avail = available_ram();
    uint64_t budget = opt.expert_ram;
    if (budget == 0) {
        const uint64_t total = opt.ram_budget ? opt.ram_budget : avail * 85 / 100;
        const uint64_t reserve = 2ull << 30;   // activations, the prompt's scratch, the OS
        budget = total > reserve ? total - reserve : 0;
        // the dense weights and the KV are already allocated (and counted in `avail`'s drop) when ram_budget is
        // not given; with an explicit budget they come out of it
        if (opt.ram_budget) budget = budget > dense_bytes_ + kv_bytes() ? budget - dense_bytes_ - kv_bytes() : 0;
    }
    experts_ = std::make_unique<ExpertCache>(*io_, c, std::move(spans), budget);
    watchdog_ = std::make_unique<Watchdog>(opt.watchdog_seconds);
    if (opt.verbose)
        std::fprintf(stderr,
                     "strata-glm: %d layers, hidden %d, %d experts (top-%d), vocab %d | dense %.2f GB resident | KV %.2f GB "
                     "(%d tokens) | expert cache %d slots = %.1f GB | %d threads, %d I/O | loaded in %.1f s\n",
                     c.n_layers, D, c.n_experts, c.topk, c.vocab, dense_bytes_ / 1e9, kv_bytes() / 1e9, max_ctx_,
                     experts_->slots(), experts_->slots() * (double) experts_->slot_bytes() / 1e9, pool_->size(),
                     io_->threads(), (now_ms() - t0) / 1e3);
    return true;
}

// ---------------------------------------------------------------------------------------------------- forward

void GlmModel::attention(int layer, const float* h, int S, int pos0, float* out) {
    const GlmConfig& c = c_;
    const LayerW& L = layers_[layer];
    const int H = c.n_heads, qh = c.qk_head(), kvl = c.kv_lora, nr = c.qk_rope, vh = c.v_head;
    const int row = kvl + nr;
    Pool& pool = *pool_;
    std::vector<float> QR((size_t) S * c.q_lora), Q((size_t) S * H * qh), comp((size_t) S * row);
    gemm(L.q_a, h, S, QR.data());
    for (int s = 0; s < S; ++s) rmsnorm(&QR[(size_t) s * c.q_lora], &QR[(size_t) s * c.q_lora], L.q_a_ln, c.q_lora, c.eps);
    gemm(L.q_b, QR.data(), S, Q.data());
    gemm(L.kv_a, h, S, comp.data());
    // the rows are written through the cache's format; an f32 cache is then read in place, a compact one is
    // decoded into `keys` (one copy of the layer's rows per call: 18 MB per layer at 8K tokens)
    static thread_local std::vector<float> keys;
    keys.resize((size_t)(pos0 + S) * row);
    float* kv = keys.data();
    for (int s = 0; s < S; ++s) {
        const int pos = pos0 + s;
        float* dst = kv + (size_t) pos * row;
        rmsnorm(dst, &comp[(size_t) s * row], L.kv_a_ln, kvl, c.eps);   // the normed latent
        std::memcpy(dst + kvl, &comp[(size_t) s * row + kvl], (size_t) nr * sizeof(float));
        rope_.apply(dst + kvl, pos);                                       // the roped key, shared by every head
        kv_.write(layer, pos, dst);
        for (int hd = 0; hd < H; ++hd) rope_.apply(&Q[((size_t) s * H + hd) * qh + c.qk_nope], pos);
    }
    if (const float* in_place = kv_.f32_layer(layer)) kv = const_cast<float*>(in_place);
    else kv_.read_layer(layer, pos0 + S, kv);
    const float scale = 1.0f / std::sqrt((float) qh);
    std::vector<float> ctx((size_t) S * H * vh);
#ifdef STRATA_GLM_CUDA
    if (gpu_ && gpu_prefill_ && S > 1) {
        gpu_->attention(L.kv_b, Q.data(), kv, S, pos0, H, c.qk_nope, nr, vh, ctx.data());
        gemm(L.o, ctx.data(), S, out);
        return;
    }
#endif
    if (S >= 4 && kvl % 16 == 0 && nr % 8 == 0) {
        // a prompt: one head at a time, its kv_b slices unpacked to floats once for all S queries (227 s of a
        // 1,536-token CPU prompt went to the per-query path below, which unpacks them for every query)
        pool.parallel_for(H, 1, [&](int64_t b, int64_t e) {
            static thread_local std::vector<float> wk, wv;
            wk.resize((size_t) c.qk_nope * kvl);
            wv.resize((size_t) vh * kvl);
            for (int64_t hd = b; hd < e; ++hd) {
                const int rbase = (int) hd * (c.qk_nope + vh);
                q4_dequant_rows(L.kv_b, rbase, c.qk_nope, wk.data());
                q4_dequant_rows(L.kv_b, rbase + c.qk_nope, vh, wv.data());
                mla_head_prompt(wk.data(), wv.data(), &Q[(size_t) hd * qh], H * qh, kv, kvl, nr, c.qk_nope, vh, S, pos0,
                                scale, &ctx[(size_t) hd * vh], H * vh);
            }
        });
        gemm(L.o, ctx.data(), S, out);
        return;
    }
    pool.parallel_for((int64_t) S * H, 1, [&](int64_t b, int64_t e) {
        static thread_local std::vector<float> sc;
        static thread_local Act act;
        float qabs[512], clat[512];
        for (int64_t i = b; i < e; ++i) {
            const int s = (int) (i / H), hd = (int) (i % H);
            const int pos = pos0 + s, nt = pos + 1;
            const float* q = &Q[((size_t) s * H + hd) * qh];
            const int rbase = hd * (c.qk_nope + vh);
            std::fill(qabs, qabs + kvl, 0.0f);
            q4_rows_t(L.kv_b, rbase, c.qk_nope, q, qabs);   // W_k^T q_nope
            sc.resize(nt);
            for (int t = 0; t < nt; ++t) {
                const float* kt = kv + (size_t) t * row;
                sc[t] = (dot_f32(qabs, kt, kvl) + dot_f32(q + c.qk_nope, kt + kvl, nr)) * scale;
            }
            softmax_inplace(sc.data(), nt);
            std::fill(clat, clat + kvl, 0.0f);
            for (int t = 0; t < nt; ++t) axpy_f32(clat, sc[t], kv + (size_t) t * row, kvl);
            act.prepare(clat, 1, kvl);
            q4_rows(L.kv_b, act, rbase + c.qk_nope, rbase + c.qk_nope + vh, &ctx[((size_t) s * H + hd) * vh], vh,
                    rbase + c.qk_nope);
        }
    });
    gemm(L.o, ctx.data(), S, out);
}

void GlmModel::gemm(const Q4& w, const float* x, int S, float* out) {
#ifdef STRATA_GLM_CUDA
    if (gpu_ && gpu_prefill_ && S > 1) { gpu_->gemm(w, x, S, out); return; }
#endif
    q4_gemm(*pool_, w, x, S, out);
}

void GlmModel::dense_mlp(const LayerW& L, const float* h, int S, float* out) {
    const int I = L.gate.O;
    std::vector<float> g((size_t) S * I), u((size_t) S * I);
    gemm(L.gate, h, S, g.data());
    gemm(L.up, h, S, u.data());
    for (size_t i = 0; i < g.size(); ++i) g[i] = silu(g[i]) * u[i];
    gemm(L.down, g.data(), S, out);
}

void GlmModel::prefetch(int layer, const float* x) {
    if (layer >= c_.n_layers || !layers_[layer].sparse) return;
    const auto& L = layers_[layer];
    std::vector<float> norm(c_.hidden), scores(c_.n_experts);
    rmsnorm(norm.data(), x, L.post_ln, c_.hidden, c_.eps);
    for (int e = 0; e < c_.n_experts; ++e)
        scores[e] = sigmoidf(dot_f32(L.router + (size_t)e * c_.hidden, norm.data(), c_.hidden)) + L.router_bias[e];
    std::vector<int> ids;
    for (int k = 0; k < c_.topk; ++k) {
        int e = (int)(std::max_element(scores.begin(), scores.end()) - scores.begin());
        scores[e] = -INFINITY;
#ifdef STRATA_GLM_CUDA
        if (gpu_ && gpu_->contains(layer, e)) continue;
#endif
        ids.push_back(e);
    }
    experts_->prefetch(layer, ids.data(), (int)ids.size());
}

bool GlmModel::moe(int layer, const float* h, int S, float* out, std::string& err, const std::atomic<bool>* cancel) {
    const GlmConfig& c = c_;
    const LayerW& L = layers_[layer];
    const int D = c.hidden, E = c.n_experts, K = c.topk;
    Pool& pool = *pool_;
    // the router: sigmoid scores; the correction bias only decides WHICH experts, the weights are the scores
    std::vector<float> logit((size_t) S * E);
    pool.parallel_for(E, 8, [&](int64_t b, int64_t e) {
        for (int64_t x = b; x < e; ++x)
            for (int s = 0; s < S; ++s) logit[(size_t) s * E + x] = dot_f32(L.router + (size_t) x * D, h + (size_t) s * D, D);
    });
    std::vector<int> idx((size_t) S * K);
    std::vector<float> w((size_t) S * K);
    std::vector<float> choice(E);
    for (int s = 0; s < S; ++s) {
        float* lg = &logit[(size_t) s * E];
        for (int x = 0; x < E; ++x) { lg[x] = sigmoidf(lg[x]); choice[x] = lg[x] + L.router_bias[x]; }
        for (int k = 0; k < K; ++k) {
            int best = -1;
            float bv = -1e30f;
            for (int x = 0; x < E; ++x)
                if (choice[x] > bv) { bv = choice[x]; best = x; }
            if (best < 0) best = k;   // non-finite scores: degrade deterministically rather than index -1
            idx[(size_t) s * K + k] = best;
            w[(size_t) s * K + k] = lg[best];
            choice[best] = -1e30f;
        }
        float* ws = &w[(size_t) s * K];
        if (c.norm_topk) {
            float sm = 1e-20f;
            for (int k = 0; k < K; ++k) sm += ws[k];
            for (int k = 0; k < K; ++k) ws[k] /= sm;
        }
        for (int k = 0; k < K; ++k) ws[k] *= c.routed_scale;
    }
    // the experts this chunk needs, each with the rows routed to it
    std::vector<int> uniq;
    std::vector<std::vector<std::pair<int, float>>> rows;
    {
        std::vector<int> pos(E, -1);
        for (int s = 0; s < S; ++s)
            for (int k = 0; k < K; ++k) {
                const int x = idx[(size_t) s * K + k];
                if (pos[x] < 0) { pos[x] = (int) uniq.size(); uniq.push_back(x); rows.emplace_back(); }
                rows[pos[x]].emplace_back(s, w[(size_t) s * K + k]);
            }
    }
    std::vector<int> slot(uniq.size(), -1), need, positions;
    for (size_t j = 0; j < uniq.size(); ++j) {
#ifdef STRATA_GLM_CUDA
        if (gpu_ && gpu_->contains(layer, uniq[j])) continue;
#endif
        need.push_back(uniq[j]); positions.push_back((int)j);
    }
    std::vector<int> pinned(need.size(), -1);
    // Release every pin on cancellation, I/O failure or a CUDA/allocator exception.
    struct Pins {
        ExpertCache& cache; std::vector<int>& slots;
        ~Pins() { for (int s : slots) if (s >= 0) cache.release(s); }
    } pins{*experts_, pinned};
    experts_->request(layer, need.data(), (int)need.size(), pinned.data());
    for (size_t j = 0; j < positions.size(); ++j) slot[positions[j]] = pinned[j];

    // the shared expert is resident: compute it while the misses arrive
    std::fill(out, out + (size_t) S * D, 0.0f);
    if (c.n_shared > 0) {
        const int sI = L.sh_gate.O;
        std::vector<float> g((size_t) S * sI), u((size_t) S * sI);
        gemm(L.sh_gate, h, S, g.data());
        gemm(L.sh_up, h, S, u.data());
        for (size_t i = 0; i < g.size(); ++i) g[i] = silu(g[i]) * u[i];
        gemm(L.sh_down, g.data(), S, out);
    }
    // routed experts in the order their bytes are ready
    std::vector<char> done(uniq.size(), 0);
    std::vector<float> xe, ge, ue, ye;
    Act a;
    bool ok = true;
    for (size_t n = 0; n < uniq.size(); ++n) {
        watchdog_->touch();
        if (cancel && cancel->load()) { err = "cancelled"; return false; }
        int j = -1;
        for (size_t q = 0; q < uniq.size(); ++q)
            if (!done[q] && (slot[q] < 0 || experts_->ready(slot[q]))) { j = (int) q; break; }
        if (j < 0) {
            for (size_t q = 0; q < uniq.size(); ++q)
                if (!done[q]) { j = (int) q; break; }
            if (!experts_->wait(slot[j])) {
                err = "reading expert " + std::to_string(uniq[j]) + " of layer " + std::to_string(layer) + " failed";
                ok = false;
                break;
            }
        }
        done[j] = 1;
        const ExpertView v = slot[j] >= 0 ? experts_->view(slot[j]) : ExpertView{};
        const auto& rs = rows[j];
        const int nr = (int) rs.size(), I = c.moe_inter;
        xe.resize((size_t) nr * D);
        for (int r = 0; r < nr; ++r) std::memcpy(&xe[(size_t) r * D], h + (size_t) rs[r].first * D, (size_t) D * sizeof(float));
        ge.resize((size_t) nr * I);
        ue.resize((size_t) nr * I);
        ye.resize((size_t) nr * D);
#ifdef STRATA_GLM_CUDA
        if (gpu_ && (slot[j] < 0 || (gpu_prefill_ && S > 1)))
            gpu_->expert(layer, uniq[j], slot[j] >= 0 ? &v : nullptr, xe.data(), nr, ye.data());
        else
#endif
        {
            a.prepare(xe.data(), nr, D);
            q4_gemm(pool, v.gate, a, ge.data());
            q4_gemm(pool, v.up, a, ue.data());
            for (size_t i = 0; i < ge.size(); ++i) ge[i] = silu(ge[i]) * ue[i];
            q4_gemm(pool, v.down, ge.data(), nr, ye.data());
        }
        for (int r = 0; r < nr; ++r) axpy_f32(out + (size_t) rs[r].first * D, rs[r].second, &ye[(size_t) r * D], D);
    }
#ifdef STRATA_GLM_CUDA
    // Promote after the layer is finished: no resident expert selected above may be evicted mid-layer.
    if (ok && gpu_ && S == 1)
        for (size_t j = 0; j < uniq.size(); ++j)
            if (slot[j] >= 0) gpu_->promote(layer, uniq[j], experts_->view(slot[j]));
#endif
    return ok;
}

bool GlmModel::forward(const int* ids, int S, float* logits, float* all_logits, std::string& err,
                       const std::atomic<bool>* cancel, int attention_chunk, const std::function<void(int)>* on_layer) {
    Watchdog::Guard watch(*watchdog_);
    const GlmConfig& c = c_;
    const int D = c.hidden;
    if (S <= 0) { err = "no tokens"; return false; }
    if (n_past_ + S > max_ctx_) {
        err = "the context is full (" + std::to_string(n_past_ + S) + " > " + std::to_string(max_ctx_) + " tokens)";
        return false;
    }
    std::vector<float> x((size_t) S * D), hbuf((size_t) S * D), tmp((size_t) S * D);
    for (int s = 0; s < S; ++s) {
        if (ids[s] < 0 || ids[s] >= c.vocab) { err = "token id " + std::to_string(ids[s]) + " is outside the vocabulary"; return false; }
        q8r_row(embed_, ids[s], &x[(size_t) s * D]);
    }
    const int pos0 = n_past_;
    for (int l = 0; l < c.n_layers; ++l) {
        watchdog_->touch();
        if (cancel && cancel->load()) { err = "cancelled"; return false; }
        const LayerW& L = layers_[l];
        for (int s = 0; s < S; ++s) rmsnorm(&hbuf[(size_t) s * D], &x[(size_t) s * D], L.in_ln, D, c.eps);
        double t = now_ms();
        for (int b = 0; b < S; b += attention_chunk) {
            watchdog_->touch();
            if (cancel && cancel->load()) { err = "cancelled"; return false; }
            attention(l, hbuf.data() + (size_t)b * D, std::min(S - b, attention_chunk), pos0 + b,
                      tmp.data() + (size_t)b * D);
        }
        st_.attn_ms += now_ms() - t;
        for (size_t i = 0; i < x.size(); ++i) x[i] += tmp[i];
        if (S == 1 && prefetch_) prefetch(l + 1, x.data());
        if (cancel && cancel->load()) { err = "cancelled"; return false; }
        for (int s = 0; s < S; ++s) rmsnorm(&hbuf[(size_t) s * D], &x[(size_t) s * D], L.post_ln, D, c.eps);
        t = now_ms();
        if (L.sparse) {
            const uint64_t w0 = experts_->stats().wait_us;
            if (!moe(l, hbuf.data(), S, tmp.data(), err, cancel)) return false;
            st_.expert_wait_ms += (experts_->stats().wait_us - w0) / 1e3;
            st_.moe_ms += now_ms() - t;
        } else {
            for (int b = 0; b < S; b += attention_chunk) {
                if (cancel && cancel->load()) { err = "cancelled"; return false; }
                dense_mlp(L, hbuf.data() + (size_t)b * D, std::min(S - b, attention_chunk), tmp.data() + (size_t)b * D);
            }
            st_.dense_ms += now_ms() - t;
        }
        for (size_t i = 0; i < x.size(); ++i) x[i] += tmp[i];
        if (on_layer) (*on_layer)(l + 1);
    }
    n_past_ += S;
    const double t = now_ms();
    std::vector<float> last(D);
    if (all_logits) {
        for (int s = 0; s < S; ++s) {
            watchdog_->touch();
            rmsnorm(last.data(), &x[(size_t) s * D], final_norm_, D, c.eps);
            q8r_gemv(*pool_, lm_head_, last.data(), all_logits + (size_t) s * c.vocab);
        }
        if (logits) std::memcpy(logits, all_logits + (size_t) (S - 1) * c.vocab, (size_t) c.vocab * sizeof(float));
    } else if (logits) {
        rmsnorm(last.data(), &x[(size_t) (S - 1) * D], final_norm_, D, c.eps);
        q8r_gemv(*pool_, lm_head_, last.data(), logits);
    }
    st_.head_ms += now_ms() - t;
    return true;
}

int GlmModel::gpu_slots() const {
#ifdef STRATA_GLM_CUDA
    return gpu_ ? gpu_->slots() : 0;
#else
    return 0;
#endif
}
uint64_t GlmModel::gpu_hits() const {
#ifdef STRATA_GLM_CUDA
    return gpu_ ? gpu_->stats().hits : 0;
#else
    return 0;
#endif
}

}  // namespace strata::glm
