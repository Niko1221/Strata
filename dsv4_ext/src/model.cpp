#include "dsv4/model.hpp"

#include "dsv4/dequant.hpp"
#include "dsv4/gpu.hpp"
#include "dsv4/mem_plan.hpp"
#include "dsv4/ops.hpp"

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <map>
#include <numeric>
#include <thread>
#include <unordered_map>
#ifdef _OPENMP
#include <omp.h>
#endif
#include "dsv4/ops.hpp"
namespace dsv4 {
namespace {

using Clock = std::chrono::steady_clock;
double secs(Clock::time_point a, Clock::time_point b) { return std::chrono::duration<double>(b - a).count(); }

// ---------------------------------------------------------------- activation "QAT" simulation (model.py act_quant / fp4_act_quant)
float pow2_ceil(float v) { int e; float m = std::frexp(v, &e); return std::ldexp(1.f, m == 0.5f ? e - 1 : e); }
float fp8_round(float t) {
    if (t == 0.f) return 0.f;
    const float a = std::fabs(t); int e; std::frexp(a, &e);
    const int E = e - 1;
    const float step = E < -6 ? std::ldexp(1.f, -9) : std::ldexp(1.f, E - 3);
    float r = std::nearbyint(a / step) * step;
    if (r > 448.f) r = 448.f;
    return std::copysign(r, t);
}
void fp8_sim(float* x, int n, int blk) {
    for (int b = 0; b < n; b += blk) {
        float amax = 1e-4f;
        for (int i = b; i < std::min(n, b + blk); ++i) amax = std::max(amax, std::fabs(x[i]));
        const float s = pow2_ceil(amax / 448.f);
        for (int i = b; i < std::min(n, b + blk); ++i) x[i] = fp8_round(std::min(448.f, std::max(-448.f, x[i] / s))) * s;
    }
}
float fp4_round(float t) {
    const float a = std::fabs(t);
    float r;
    if (a <= 0.25f) r = 0.f; else if (a < 0.75f) r = 0.5f; else if (a <= 1.25f) r = 1.f; else if (a < 1.75f) r = 1.5f;
    else if (a <= 2.5f) r = 2.f; else if (a < 3.5f) r = 3.f; else if (a <= 5.f) r = 4.f; else r = 6.f;
    return std::copysign(r, t);
}
void fp4_sim(float* x, int n, int blk) {
    for (int b = 0; b < n; b += blk) {
        float amax = 6.f * std::ldexp(1.f, -126);
        for (int i = b; i < std::min(n, b + blk); ++i) amax = std::max(amax, std::fabs(x[i]));
        const float s = pow2_ceil(amax / 6.f);
        for (int i = b; i < std::min(n, b + blk); ++i) x[i] = fp4_round(std::min(6.f, std::max(-6.f, x[i] / s))) * s;
    }
}
void hadamard(float* x, int n) {  // natural-order Walsh-Hadamard, scaled by n^-0.5 (rotate_activation)
    for (int h = 1; h < n; h <<= 1)
        for (int i = 0; i < n; i += 2 * h)
            for (int j = i; j < i + h; ++j) { const float a = x[j], b = x[j + h]; x[j] = a + b; x[j + h] = a - b; }
    const float s = 1.f / std::sqrt((float) n);
    for (int i = 0; i < n; ++i) x[i] *= s;
}

// ---------------------------------------------------------------- tensors
struct Tn {
    const GgufTensor* g = nullptr;
    const uint8_t* h = nullptr;   // host (mmap)
    uint8_t* d = nullptr;         // device copy (only for matrices that are matvec'd on the GPU)
    bool no_gpu = false;          // the device has no kernel for this type: never uploaded, host only
    uint32_t type = 0;
    int64_t in = 0, out = 1, ne = 1;
    size_t rb = 0;                // bytes per row
    explicit operator bool() const { return g != nullptr; }
    const float* f() const { return (const float*) h; }
};

struct Comp {  // KV compressor state (model.py Compressor), decode-only
    int ratio = 0, d = 0, coff = 1;
    bool overlap = false, rotate = false;
    Tn wkv, wg, ape, norm;
    std::vector<float> kvs, scs;   // [coff*ratio][coff*d]
    float* cache = nullptr;        // [ctx/ratio + 1][d]
    std::vector<float> own;        // storage when cache is not shared with the attention kv buffer
};

struct Layer {
    int ratio = 0;
    Tn attn_norm, q_a, q_a_norm, q_b, kv, kv_norm, o_a, o_b, sinks;
    Tn ffn_norm, gate_inp, exp_bias, tid2eid;
    Tn sh_gate, sh_up, sh_down;
    Tn hc_attn_fn, hc_attn_base, hc_attn_scale, hc_ffn_fn, hc_ffn_base, hc_ffn_scale;
    Comp ac, ic;
    Tn i_q_b, i_proj;
    bool has_idx = false;
    std::vector<float> kvbuf;      // [win + ctx/ratio + 1][hd]: ring window, then compressed entries
    // routed experts
    Tn eg, eu, ed;
    size_t bg = 0, bu = 0, bd = 0, bpe = 0;     // bytes per expert: gate, up, down, total
    int slots = 0;
    bool gpu_experts = false;   // the device has kernels for eg/eu/ed: only then are VRAM slots worth anything
    uint8_t* dpool = nullptr;
    std::vector<int> slot_of, occ;              // expert -> slot, slot -> expert
    std::vector<uint64_t> freq;                 // routing frequency (profile counts + runtime)
    std::vector<int> ram_idx;                   // expert -> index in the RAM arena (-1 = not in RAM)
    std::vector<uint8_t> ram;                   // RAM arena: only experts the profile marks as used
    // LRU arena for MISS experts (neither in `ram` nor in a VRAM slot). Without it every miss re-reads
    // bpe bytes from the mmap'ed shard: for a model larger than RAM that is a disk read per expert.
    std::vector<uint8_t> cpool;                 // ncache slots of bpe bytes each
    std::vector<int> cslot_of, cocc;            // expert -> slot, slot -> expert
    std::vector<uint64_t> cstamp;               // slot -> last use (LRU clock)
    int ncache = 0;
};

struct ExpSrc { const uint8_t *g, *u, *d; const uint8_t* packed; };  // packed != null: contiguous [g|u|d]

uint64_t fnv(uint64_t h, uint64_t v) { for (int i = 0; i < 8; ++i) { h ^= (v >> (8 * i)) & 0xff; h *= 1099511628211ULL; } return h; }

void cpu_mv(uint32_t type, const uint8_t* W, int64_t rows, int64_t in, const float* x, float* y) {
    const size_t rb = row_bytes(type, in);
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (int64_t r = 0; r < rows; ++r) y[r] = row_dot(type, W + (size_t) r * rb, x, in);
}

}  // namespace

std::string decode_token_text(const std::string& tok) {
    static std::map<uint32_t, uint8_t> inv;
    if (inv.empty()) {
        std::vector<int> bs, cs;
        for (int b = 33; b <= 126; ++b) bs.push_back(b);
        for (int b = 161; b <= 172; ++b) bs.push_back(b);
        for (int b = 174; b <= 255; ++b) bs.push_back(b);
        cs = bs; int n = 0;
        for (int b = 0; b < 256; ++b) if (std::find(bs.begin(), bs.end(), b) == bs.end()) { bs.push_back(b); cs.push_back(256 + n++); }
        for (size_t i = 0; i < bs.size(); ++i) inv[(uint32_t) cs[i]] = (uint8_t) bs[i];
    }
    std::string out;
    for (size_t i = 0; i < tok.size();) {
        const unsigned char c = (unsigned char) tok[i];
        uint32_t cp; int len;
        if (c < 0x80) { cp = c; len = 1; } else if ((c >> 5) == 6) { cp = c & 0x1f; len = 2; } else if ((c >> 4) == 14) { cp = c & 0x0f; len = 3; } else { cp = c & 0x07; len = 4; }
        for (int k = 1; k < len && i + k < tok.size(); ++k) cp = (cp << 6) | ((unsigned char) tok[i + k] & 0x3f);
        auto it = inv.find(cp);
        if (it != inv.end()) out.push_back((char) it->second); else out.append(tok, i, (size_t) len);
        i += (size_t) len;
    }
    return out;
}

// ================================================================================================
struct Model::Impl {
    RunOpts o;
    Dsv4Config c;
    GgufHeader h;
    ExpertInventory inv;
    std::unordered_map<std::string, const GgufTensor*> by_name;
    std::vector<const uint8_t*> maps; std::vector<size_t> map_len;
    std::vector<Layer> L;
    int nrun = 0, hd = 0, rd = 0, win = 0, dim = 0, hc = 4;
    Tn embd, outw, outnorm, out_hc_fn, out_hc_base, out_hc_scale;
    std::vector<float> cos_y, sin_y, cos_p, sin_p;   // YaRN (compressed layers) / plain (window-only layers) rope tables
    uint64_t model_hash = 0;
    std::vector<std::string> toks;
    // device scratch
    float *d_x = nullptr, *d_y = nullptr, *d_g = nullptr, *d_u = nullptr, *d_a = nullptr, *d_yh = nullptr;
    std::vector<void*> allocs;
    uint8_t* d_stage = nullptr;      // device pool MISS experts are staged through (see gpu::staging)
    bool have_profile = false;
    uint64_t cache_clock = 0;                   // LRU clock of the MISS arenas
    std::vector<std::pair<int, int>> missed_now;      // (layer, expert) missed during the current token
    int cur_pos = 0;

    ~Impl() {
        for (void* p : allocs) gpu::release(p);
        for (size_t i = 0; i < maps.size(); ++i) if (maps[i]) munmap((void*) maps[i], map_len[i]);
    }
    // vram_used counts every byte handed out by the device layer, so the residency plan can subtract the
    // weights and scratch that are ALREADY on the card instead of guessing at them. On the CUDA backend
    // cudaMemGetInfo reports what the driver has left, not what this model still needs: without this the
    // planner hands out more expert slots than fit and the dpool allocation fails.
    void* galloc(size_t n) { void* p = gpu::alloc(n); if (p) { allocs.push_back(p); vram_used += n; } return p; }
    uint64_t vram_used = 0, vram_scratch = 0;

    Tn tn(const std::string& name, bool required, std::string& err) {
        Tn t;
        auto it = by_name.find(name);
        if (it == by_name.end()) { if (required && err.empty()) err = "missing tensor: " + name; return t; }
        t.g = it->second; t.h = maps[(size_t) t.g->shard] + t.g->abs_offset; t.type = t.g->type;
        t.in = (int64_t) t.g->dims[0]; t.out = t.g->dims.size() > 1 ? (int64_t) t.g->dims[1] : 1; t.ne = t.g->dims.size() > 2 ? (int64_t) t.g->dims[2] : 1;
        t.rb = row_bytes(t.type, t.in);
        return t;
    }
    bool need_supported(const Tn& t, std::string& err) {
        if (!t.g) return true;
        if (t.type == T_I32) return true;  // the tid2eid hash table is read raw
        if (!dequant_supported(t.type)) {
            err = "tensor " + t.g->name + " has ggml type " + ggml_type_str(t.type) + " which is not supported" +
                  (t.g->known_type ? "" : " (unknown ggml id: this build knows ids 0..30 and 39)");
            return false;
        }
        if (!row_bytes(t.type, t.in)) {
            err = "tensor " + t.g->name + ": " + std::to_string(t.in) + " elements per row is not a whole number of " +
                  ggml_type_str(t.type) + " blocks";
            return false;
        }
        return true;
    }
    void upload(Tn& t) {
        if (!o.gpu || !t.g || t.d) return;
        // Uploading weights that no device kernel can decode would only burn VRAM: leave them on the host.
        if (!gpu::type_supported(t.type)) { t.no_gpu = true; return; }
        t.d = (uint8_t*) galloc(t.rb * (size_t) t.out);
        if (!t.d) { std::fprintf(stderr, "warning: VRAM allocation failed for %s: staying on the CPU\n", t.g->name.c_str()); return; }
        gpu::h2d(t.d, t.h, t.rb * (size_t) t.out);
    }
    void mv(const Tn& w, const float* x, float* y, int64_t row0 = 0, int64_t nrows = -1) {
        if (nrows < 0) nrows = w.out;
        if (o.gpu && w.d) {
            gpu::h2d(d_x, x, (size_t) w.in * 4);
            // matvec returns false when this backend has no kernel for the type: fall back to the host
            // rather than reading back a d_y that was never written.
            if (gpu::matvec(w.type, w.d + (size_t) row0 * w.rb, nrows, w.in, d_x, d_y)) {
                gpu::d2h(y, d_y, (size_t) nrows * 4);
                return;
            }
            if (!warned_host_fallback) {
                warned_host_fallback = true;
                std::fprintf(stderr, "warning: no device kernel for ggml type %s (%s): those matvecs run on the host\n",
                             ggml_type_str(w.type).c_str(), w.g ? w.g->name.c_str() : "?");
            }
        }
        cpu_mv(w.type, w.h + (size_t) row0 * w.rb, nrows, w.in, x, y);
    }
    bool warned_host_fallback = false;

    // ------------------------------------------------------------ expert sources / residency
    static ExpSrc packed(const Layer& l, const uint8_t* p) {
        ExpSrc s; s.packed = p; s.g = p; s.u = p + l.bg; s.d = p + l.bg + l.bu; return s;
    }
    ExpSrc src(const Layer& l, int e) const {
        if (l.ram_idx[(size_t) e] >= 0) return packed(l, l.ram.data() + (size_t) l.ram_idx[(size_t) e] * l.bpe);
        ExpSrc s{};
        s.g = l.eg.h + (size_t) e * l.bg; s.u = l.eu.h + (size_t) e * l.bu; s.d = l.ed.h + (size_t) e * l.bd;
        return s;
    }
    /// Weights of a MISS expert: the profile's RAM copy, else the LRU arena (copied out of the mmap once),
    /// else the mmap itself when the arena is disabled.
    ExpSrc src_miss(Layer& y, int e) {
        if (y.ncache <= 0 || y.ram_idx[(size_t) e] >= 0) return src(y, e);
        const int have = y.cslot_of[(size_t) e];
        if (have >= 0) { y.cstamp[(size_t) have] = ++cache_clock; ++st->cache_hits; return packed(y, y.cpool.data() + (size_t) have * y.bpe); }
        int slot = -1;
        for (int s = 0; s < y.ncache; ++s) if (y.cocc[(size_t) s] < 0) { slot = s; break; }
        if (slot < 0) {  // full: evict the least recently used
            uint64_t best = UINT64_MAX;
            for (int s = 0; s < y.ncache; ++s) if (y.cstamp[(size_t) s] < best) { best = y.cstamp[(size_t) s]; slot = s; }
            if (y.cocc[(size_t) slot] >= 0) y.cslot_of[(size_t) y.cocc[(size_t) slot]] = -1;
            ++st->cache_evictions;
        }
        uint8_t* dst = y.cpool.data() + (size_t) slot * y.bpe;
        std::memcpy(dst, y.eg.h + (size_t) e * y.bg, y.bg);
        std::memcpy(dst + y.bg, y.eu.h + (size_t) e * y.bu, y.bu);
        std::memcpy(dst + y.bg + y.bu, y.ed.h + (size_t) e * y.bd, y.bd);
        y.cocc[(size_t) slot] = e; y.cslot_of[(size_t) e] = slot; y.cstamp[(size_t) slot] = ++cache_clock;
        ++st->cache_loads; st->cache_bytes += y.bpe;
        return packed(y, dst);
    }
    void upload_slot(Layer& l, int slot, int e) {
        ExpSrc s = src(l, e);
        uint8_t* dst = l.dpool + (size_t) slot * l.bpe;
        if (s.packed) gpu::h2d(dst, s.packed, l.bpe);
        else { gpu::h2d(dst, s.g, l.bg); gpu::h2d(dst + l.bg, s.u, l.bu); gpu::h2d(dst + l.bg + l.bu, s.d, l.bd); }
        st->h2d_bytes += l.bpe;
        if (o.verify_slots) {
            std::vector<uint8_t> back(l.bpe);
            gpu::d2h(back.data(), dst, l.bpe);
            if (std::memcmp(back.data(), s.g, l.bg) || std::memcmp(back.data() + l.bg, s.u, l.bu) || std::memcmp(back.data() + l.bg + l.bu, s.d, l.bd)) {
                std::fprintf(stderr, "FATAL: verify_slot failed (slot %d, expert %d)\n", slot, e); std::abort();
            }
        }
        if (l.occ[(size_t) slot] >= 0) l.slot_of[(size_t) l.occ[(size_t) slot]] = -1;
        l.occ[(size_t) slot] = e; l.slot_of[(size_t) e] = slot;
    }

    // ------------------------------------------------------------ profile
    bool profile_load(const std::string& path, std::string& err) {
        FILE* f = std::fopen(path.c_str(), "rb");
        if (!f) { err = "cannot open profile " + path; return false; }
        char magic[8]; uint32_t ver, nl, ne; uint64_t mh;
        bool ok = std::fread(magic, 1, 8, f) == 8 && std::fread(&ver, 4, 1, f) == 1 && std::fread(&nl, 4, 1, f) == 1 && std::fread(&ne, 4, 1, f) == 1 && std::fread(&mh, 8, 1, f) == 1;
        if (!ok || std::memcmp(magic, "DSV4PROF", 8) != 0 || ver != 1) { std::fclose(f); err = "not a DSV4 expert profile (version 1)"; return false; }
        if ((int) nl != c.n_layer || (int) ne != c.n_expert) { std::fclose(f); err = "profile layer/expert counts do not match the model"; return false; }
        if (mh != model_hash) { std::fclose(f); err = "profile was built for a different model (hash mismatch)"; return false; }
        std::vector<uint64_t> cnt((size_t) nl * ne);
        ok = std::fread(cnt.data(), 8, cnt.size(), f) == cnt.size();
        std::fclose(f);
        if (!ok) { err = "truncated profile"; return false; }
        for (int l = 0; l < c.n_layer; ++l) for (int e = 0; e < c.n_expert; ++e) L[(size_t) l].freq[(size_t) e] = cnt[(size_t) l * ne + e];
        return true;
    }

    Stats* st = nullptr;
    // ------------------------------------------------------------ load
    bool load(const std::vector<std::string>& shards, const RunOpts& opts, std::string& err) {
        const auto T0 = Clock::now();
        auto ph = [&](const char* what) { std::fprintf(stderr, "[load %6.1fs] %s\n", secs(T0, Clock::now()), what); };
        o = opts;
        if (!gguf_read_header(shards, h, err)) return false;
        if (h.tensors.empty()) { err = "no tensors found: pass the first shard of a multi-file GGUF (…-00001-of-0000N.gguf) so all shards are loaded"; return false; }
        if (!config_from_gguf(h, c, err)) return false;
        if (!inventory_from_gguf(h, c, inv, err)) return false;
        if (c.gating_func != 1 && c.gating_func != 2 && c.gating_func != 4) { err = "unsupported expert_gating_func " + std::to_string(c.gating_func); return false; }
        if (c.idx_key_len & (c.idx_key_len - 1)) { err = "indexer key length must be a power of two (Hadamard)"; return false; }
#ifdef _OPENMP
        // Every matvec here is memory-bound and embarrassingly parallel over the output rows: leaving this on
        // one core costs roughly 20x on a 28-thread machine. 0 means "use the whole machine".
        if (o.threads <= 0) {
            int hw = (int) std::thread::hardware_concurrency();
            o.threads = hw > 0 ? hw : 8;
        }
        omp_set_num_threads(o.threads);
#else
        if (o.verbose) std::fprintf(stderr, "warning: this build has NO OpenMP: every matvec runs on a single thread (expect ~20x slower)\n");
#endif
        for (const GgufTensor& t : h.tensors) by_name[t.name] = &t;
        for (const std::string& f : h.files) {
            int fd = ::open(f.c_str(), O_RDONLY);
            struct stat sb;
            if (fd < 0 || fstat(fd, &sb) != 0) { err = "cannot open " + f; return false; }
            void* m = mmap(nullptr, (size_t) sb.st_size, PROT_READ, MAP_SHARED, fd, 0);
            ::close(fd);
            if (m == MAP_FAILED) { err = "mmap failed for " + f; return false; }
            // MADV_RANDOM switches the kernel's readahead OFF. We never touch a single byte at a time: an
            // expert is three contiguous runs (gate|up|down, ~5.5 MiB) and a matvec walks whole rows, so
            // demand paging at page granularity is already efficient.
            //
            // WILLNEED was the wrong advice here: it asks the kernel to prefetch the WHOLE shard, which on
            // an 80 GiB model in front of a machine with less free RAM than that evicts the pages the very
            // next forward pass needs. The page cache then thrashes on every token and the miss path reads
            // the disk again. RANDOM keeps the cache for what is actually touched.
            madvise(m, (size_t) sb.st_size, MADV_RANDOM);
            maps.push_back((const uint8_t*) m); map_len.push_back((size_t) sb.st_size);
        }
        // Every tensor must lie inside its shard: a truncated or mismatched file would otherwise
        // turn into a segfault deep inside the dequantiser.
        for (const GgufTensor& t : h.tensors) {
            if (t.shard < 0 || (size_t) t.shard >= maps.size()) { err = "tensor " + t.name + " refers to a missing shard"; return false; }
            const uint64_t need = t.nbytes ? t.abs_offset + t.nbytes : t.abs_offset;
            if (need > map_len[(size_t) t.shard]) {
                err = "tensor " + t.name + " (" + ggml_type_str(t.type) + ", " + std::to_string(t.nbytes) +
                      " bytes at offset " + std::to_string(t.abs_offset) + ") is outside " + h.files[(size_t) t.shard] +
                      " (" + std::to_string(map_len[(size_t) t.shard]) + " bytes): the shard is truncated or the GGUF is inconsistent";
                return false;
            }
        }
        ph("mmap + header checks done");
        model_hash = 1469598103934665603ULL;
        for (uint64_t v : {(uint64_t) c.n_layer, (uint64_t) c.n_expert, (uint64_t) c.ff_exp, (uint64_t) c.n_embd, (uint64_t) h.tensors.size()}) model_hash = fnv(model_hash, v);
        for (uint64_t b : inv.bytes_per_expert) model_hash = fnv(model_hash, b);
        if (const GgufValue* t = h.find("tokenizer.ggml.tokens")) toks = t->strs;

        dim = c.n_embd; hd = c.key_len; rd = c.rope_dim; win = c.sliding_window; hc = c.hc_count;
        nrun = (o.max_layers > 0 && o.max_layers < c.n_layer) ? o.max_layers : c.n_layer;
        if (o.ctx < 8) o.ctx = 8;

        if (o.gpu) {
            if (!gpu::init(err)) return false;
            const uint64_t before_scratch = vram_used;
            d_x = (float*) galloc(16384 * 4); d_y = (float*) galloc((size_t) std::max<int64_t>(c.vocab, 16384) * 4 + 4096);
            d_g = (float*) galloc((size_t) gpu::kMaxHit * c.ff_exp * 4); d_u = (float*) galloc((size_t) gpu::kMaxHit * c.ff_exp * 4);
            d_a = (float*) galloc((size_t) gpu::kMaxHit * c.ff_exp * 4); d_yh = (float*) galloc((size_t) dim * 4);
            if (!d_x || !d_y || !d_g || !d_u || !d_a || !d_yh) { err = "cannot allocate GPU scratch"; return false; }
            vram_scratch = vram_used - before_scratch;
        }
        embd = tn("token_embd.weight", true, err); outw = tn("output.weight", true, err); outnorm = tn("output_norm.weight", true, err);
        out_hc_fn = tn("output_hc_fn.weight", true, err); out_hc_base = tn("output_hc_base.weight", true, err); out_hc_scale = tn("output_hc_scale.weight", true, err);
        if (!err.empty()) return false;
        if (!need_supported(embd, err) || !need_supported(outw, err)) return false;

        yarn_table(rd, o.ctx, (int) c.yarn_orig_ctx, c.rope_base_compress, c.yarn_factor, c.yarn_beta_fast, c.yarn_beta_slow, cos_y, sin_y);
        yarn_table(rd, o.ctx, 0, c.rope_base, c.yarn_factor, c.yarn_beta_fast, c.yarn_beta_slow, cos_p, sin_p);

        L.assign((size_t) c.n_layer, Layer());
        auto init_comp = [&](Comp& cp, int ratio, int d, bool rotate, const Tn& wkv, const Tn& wg, const Tn& ape, const Tn& norm, float* cache) {
            cp.ratio = ratio; cp.d = d; cp.overlap = ratio == 4; cp.coff = 1 + (cp.overlap ? 1 : 0); cp.rotate = rotate;
            cp.wkv = wkv; cp.wg = wg; cp.ape = ape; cp.norm = norm; cp.cache = cache;
            return wkv.out == (int64_t) cp.coff * d && wg.out == (int64_t) cp.coff * d;
        };
        for (int l = 0; l < nrun; ++l) {
            Layer& y = L[(size_t) l];
            auto N = [&](const char* s, bool req = true) { return tn("blk." + std::to_string(l) + "." + s, req, err); };
            y.ratio = c.compress_ratios[(size_t) l];
            y.attn_norm = N("attn_norm.weight"); y.q_a = N("attn_q_a.weight"); y.q_a_norm = N("attn_q_a_norm.weight"); y.q_b = N("attn_q_b.weight");
            y.kv = N("attn_kv.weight"); y.kv_norm = N("attn_kv_a_norm.weight"); y.o_a = N("attn_output_a.weight"); y.o_b = N("attn_output_b.weight"); y.sinks = N("attn_sinks.weight");
            y.ffn_norm = N("ffn_norm.weight"); y.gate_inp = N("ffn_gate_inp.weight");
            if (l < c.n_hash_layers) y.tid2eid = N("ffn_gate_tid2eid.weight"); else y.exp_bias = N("exp_probs_b.bias", false);
            y.sh_gate = N("ffn_gate_shexp.weight"); y.sh_up = N("ffn_up_shexp.weight"); y.sh_down = N("ffn_down_shexp.weight");
            y.hc_attn_fn = N("hc_attn_fn.weight"); y.hc_attn_base = N("hc_attn_base.weight"); y.hc_attn_scale = N("hc_attn_scale.weight");
            y.hc_ffn_fn = N("hc_ffn_fn.weight"); y.hc_ffn_base = N("hc_ffn_base.weight"); y.hc_ffn_scale = N("hc_ffn_scale.weight");
            y.eg = N("ffn_gate_exps.weight"); y.eu = N("ffn_up_exps.weight"); y.ed = N("ffn_down_exps.weight");
            if (!err.empty()) return false;
            y.kvbuf.assign((size_t) (win + (y.ratio ? o.ctx / y.ratio + 1 : 0)) * hd, 0.f);
            if (y.ratio) {
                if (!init_comp(y.ac, y.ratio, hd, false, N("attn_compressor_kv.weight"), N("attn_compressor_gate.weight"), N("attn_compressor_ape.weight"), N("attn_compressor_norm.weight"), y.kvbuf.data() + (size_t) win * hd) && err.empty())
                    err = "compressor tensor shapes do not match ratio " + std::to_string(y.ratio) + " at layer " + std::to_string(l);
                y.has_idx = by_name.count("blk." + std::to_string(l) + ".indexer.attn_q_b.weight") > 0;
                if (y.has_idx) {
                    y.i_q_b = N("indexer.attn_q_b.weight"); y.i_proj = N("indexer.proj.weight");
                    y.ic.own.assign((size_t) (o.ctx / y.ratio + 1) * c.idx_key_len, 0.f);
                    if (!init_comp(y.ic, y.ratio, c.idx_key_len, true, N("indexer_compressor_kv.weight"), N("indexer_compressor_gate.weight"), N("indexer_compressor_ape.weight"), N("indexer_compressor_norm.weight"), y.ic.own.data()) && err.empty())
                        err = "indexer compressor shapes mismatch at layer " + std::to_string(l);
                }
            }
            if (!err.empty()) return false;
            for (Tn* t : {&y.q_a, &y.q_b, &y.kv, &y.o_a, &y.o_b, &y.gate_inp, &y.sh_gate, &y.sh_up, &y.sh_down, &y.ac.wkv, &y.ac.wg, &y.i_q_b, &y.ic.wkv, &y.ic.wg, &y.eg, &y.eu, &y.ed})
                if (!need_supported(*t, err)) return false;
            y.bg = y.eg.rb * (size_t) y.eg.out; y.bu = y.eu.rb * (size_t) y.eu.out; y.bd = y.ed.rb * (size_t) y.ed.out; y.bpe = y.bg + y.bu + y.bd;
            if (y.eg.ne != c.n_expert) { err = "expert tensor expert-count mismatch at layer " + std::to_string(l); return false; }
            y.slot_of.assign((size_t) c.n_expert, -1); y.ram_idx.assign((size_t) c.n_expert, -1); y.freq.assign((size_t) c.n_expert, 0);
            // A VRAM slot only pays off if experts_hit() can run this layer's three matrices on the device.
            y.gpu_experts = o.gpu && gpu::experts_supported(y.eg.type, y.eu.type, y.ed.type);
            for (Tn* t : {&y.q_a, &y.q_b, &y.kv, &y.o_a, &y.o_b, &y.gate_inp, &y.sh_gate, &y.sh_up, &y.sh_down, &y.ac.wkv, &y.ac.wg, &y.i_q_b, &y.ic.wkv, &y.ic.wg}) upload(*t);
        }
        upload(outw);
        ph("non-expert weights uploaded to GPU");
        // ---- staging pool for MISS experts.
        // A MISS used to be decoded on the host: an IQ1_M expert is ~5.5 MiB of packed weights that
        // row_dot() expands element by element, which at 6 experts x 43 layers is several billion
        // decodes per token - more than the rest of the forward pass combined. With a device pool the
        // same expert is copied up and decoded by the kernels that already exist, so the expert cost
        // drops to a PCIe transfer that overlaps the other launches. Sized for one layer's worth of
        // kMaxHit experts: layers are evaluated one at a time, so the pool is reused, not duplicated.
        if (o.gpu) {
            size_t maxb = 0;
            for (int l = 0; l < nrun; ++l) maxb = std::max(maxb, L[(size_t) l].bpe);
            if (maxb > 0) {
                const size_t need = (size_t) gpu::kMaxHit * maxb;
                d_stage = (uint8_t*) galloc(need);
                if (d_stage && !gpu::staging(d_stage, need)) d_stage = nullptr;
                if (!d_stage && o.verbose)
                    std::fprintf(stderr, "note: no staging pool for MISS experts (%zu bytes): they stay on the host\n", need);
            }
        }
        reset();

        // ---- expert residency plan: VRAM slots per layer from --expert-vram-pct / --expert-cache
        std::string perr;
        if (!o.profile_in.empty()) {
            if (profile_load(o.profile_in, perr)) have_profile = true;
            else std::fprintf(stderr, "warning: incompatible expert profile (%s): starting without profile\n", perr.c_str());
        }
        if (o.gpu) {
            size_t fb = 0, tb = 0; gpu::mem_info(&fb, &tb);
            MemPlanInput in;
            in.vram_total = o.vram_total_mib >= 0 ? o.vram_total_mib << 20 : (int64_t) tb;
            in.vram_free = o.vram_free_mib >= 0 ? o.vram_free_mib << 20 : (int64_t) fb;
            in.reserve = o.reserve_mib << 20; in.n_expert = c.n_expert; in.pct = o.vram_pct; in.abs_slots = o.abs_slots;
            // No fixed costs here, and that is not an oversight: BOTH backends report free VRAM already net
            // of what this model has put on the card (the emulated one subtracts its own pool, cudaMemGetInfo
            // subtracts every cudaMalloc including ours and other processes'). The KV cache, the activations
            // and the prefill buffers are host memory (std::vector<float>), so they cost no VRAM at all.
            // Counting them again is what made the plan die with "fixed costs exceed the usable VRAM" on a
            // 24 GiB card that had plenty of room for experts. dsv4_plan still estimates them, because there
            // it is a what-if tool with no device in front of it.
            in.weights = 0; in.kv = 0; in.act = 0; in.prefill = 0; in.mtp = 0; in.scratch = 0;
            in.bytes_per_expert.assign((size_t) c.n_layer, 0);
            // A layer the device cannot evaluate gets 0 bytes here, so the planner gives it no slots: VRAM
            // would be spent on experts that always end up computed on the host anyway.
            for (int l = 0; l < nrun; ++l) in.bytes_per_expert[(size_t) l] = L[(size_t) l].gpu_experts ? L[(size_t) l].bpe : 0;
            in.ram_expert_bytes = inv.total_expert_bytes;
            MemPlan pl = plan_expert_memory(in);
            if (!pl.ok) { err = pl.err; return false; }
            if (o.verbose) std::fprintf(stderr, "%s", plan_report(in, pl).c_str());
            for (int l = 0; l < nrun; ++l) {
                Layer& y = L[(size_t) l];
                y.slots = pl.slots_layer[(size_t) l];
                if (y.slots <= 0) { y.slots = 0; continue; }
                y.dpool = (uint8_t*) galloc((size_t) y.slots * y.bpe);
                if (!y.dpool) { err = "VRAM allocation for the expert cache failed at layer " + std::to_string(l) + ": lower --expert-vram-pct or raise --expert-vram-reserve-mib"; return false; }
                y.occ.assign((size_t) y.slots, -1);
            }
        }
        // ---- RAM: only experts the profile marks as used are copied into RAM (the rest stay on disk via mmap)
        // uint64_t ram_bytes = 0; int ram_n = 0;
        // for (int l = 0; l < nrun; ++l) {
        //     Layer& y = L[(size_t) l];
        //     int n = 0;
        //     for (int e = 0; e < c.n_expert; ++e) if (o.ram_all || (have_profile && y.freq[(size_t) e] > 0)) y.ram_idx[(size_t) e] = n++;
        //     y.ram.resize((size_t) n * y.bpe);
        //     for (int e = 0; e < c.n_expert; ++e) if (y.ram_idx[(size_t) e] >= 0) {
        //         uint8_t* dst = y.ram.data() + (size_t) y.ram_idx[(size_t) e] * y.bpe;
        //         std::memcpy(dst, y.eg.h + (size_t) e * y.bg, y.bg); std::memcpy(dst + y.bg, y.eu.h + (size_t) e * y.bu, y.bu); std::memcpy(dst + y.bg + y.bu, y.ed.h + (size_t) e * y.bd, y.bd);
        //     }
        //     ram_bytes += y.ram.size(); ram_n += n;
        // }
        //        // ---- RAM: only experts the profile marks as used are copied into RAM (the rest stay on disk via mmap)
        // uint64_t ram_bytes = 0; int ram_n = 0;
        // // The mapping is MADV_RANDOM (no readahead), so a plain memcpy reads 4 KiB per page fault.
        // // WILLNEED asks the kernel for big sequential reads instead; it works despite MADV_RANDOM.
        // auto willneed = [](const uint8_t* p, size_t len) {
        //     const uintptr_t a = (uintptr_t) p & ~(uintptr_t) 4095;
        //     madvise((void*) a, len + ((uintptr_t) p - a), MADV_WILLNEED);
        // };
        // for (int l = 0; l < nrun; ++l) {
        //     Layer& y = L[(size_t) l];
        //     int n = 0;
        //     for (int e = 0; e < c.n_expert; ++e) if (o.ram_all || (have_profile && y.freq[(size_t) e] > 0)) y.ram_idx[(size_t) e] = n++;
        //     y.ram.resize((size_t) n * y.bpe);
        //     if (n == c.n_expert) {   // whole layer wanted (--expert-ram all): prefetch it in one go
        //         willneed(y.eg.h, (size_t) c.n_expert * y.bg);
        //         willneed(y.eu.h, (size_t) c.n_expert * y.bu);
        //         willneed(y.ed.h, (size_t) c.n_expert * y.bd);
        //     }
        //     #pragma omp parallel for schedule(static)
        //     for (int e = 0; e < c.n_expert; ++e) if (y.ram_idx[(size_t) e] >= 0) {
        //         uint8_t* dst = y.ram.data() + (size_t) y.ram_idx[(size_t) e] * y.bpe;
        //         std::memcpy(dst, y.eg.h + (size_t) e * y.bg, y.bg);
        //         std::memcpy(dst + y.bg, y.eu.h + (size_t) e * y.bu, y.bu);
        //         std::memcpy(dst + y.bg + y.bu, y.ed.h + (size_t) e * y.bd, y.bd);
        //     }
        //     ram_bytes += y.ram.size(); ram_n += n;
        // }

        // ---- RAM: only experts the profile marks as used are copied into RAM (the rest stay on disk via mmap)
        // Copy with pread() from every OpenMP thread instead of memcpy out of the mmap. The mapping is MADV_RANDOM,
        // so a memcpy page-faults 4 KiB at a time and a cold start crawled at 70-300 MB/s. An expert is three
        // contiguous runs, so pread() issues MiB-sized reads and keeps the NVMe queue full.
        uint64_t ram_bytes = 0; int ram_n = 0;
        std::vector<int> sfd;
        for (const std::string& f : h.files) sfd.push_back(::open(f.c_str(), O_RDONLY));
        auto rd = [&](const Tn& t, size_t off, uint8_t* dst, size_t len) {
            const int fd = sfd[(size_t) t.g->shard];
            size_t done = 0;
            while (done < len) {
                const ssize_t n = fd >= 0 ? ::pread(fd, dst + done, len - done, (off_t) (t.g->abs_offset + off + done)) : -1;
                if (n <= 0) { std::memcpy(dst + done, t.h + off + done, len - done); break; }   // fallback: the old mmap path
                done += (size_t) n;
            }
        };
        for (int l = 0; l < nrun; ++l) {
            Layer& y = L[(size_t) l];
            int n = 0;
            for (int e = 0; e < c.n_expert; ++e) if (o.ram_all || (have_profile && y.freq[(size_t) e] > 0)) y.ram_idx[(size_t) e] = n++;
            y.ram.resize((size_t) n * y.bpe);
            #pragma omp parallel for schedule(dynamic, 1)
            for (int e = 0; e < c.n_expert; ++e) if (y.ram_idx[(size_t) e] >= 0) {
                uint8_t* dst = y.ram.data() + (size_t) y.ram_idx[(size_t) e] * y.bpe;
                rd(y.eg, (size_t) e * y.bg, dst, y.bg);
                rd(y.eu, (size_t) e * y.bu, dst + y.bg, y.bu);
                rd(y.ed, (size_t) e * y.bd, dst + y.bg + y.bu, y.bd);
            }
            ram_bytes += y.ram.size(); ram_n += n;
        }
        for (int fd : sfd) if (fd >= 0) ::close(fd);
        ph("experts copied to RAM");
        // // ---- LRU arena for MISS experts: a host-RAM budget split over the layers, one slot per expert.
        // Without it a miss is a fresh mmap read of bpe bytes per expert per token, which on a model bigger
        // than RAM is a synchronous NTFS read on the critical path.
        uint64_t cache_bytes_plan = 0; int cache_slots_plan = 0;
        if (o.ram_cache_mib > 0) {
            uint64_t sumb = 0;
            for (int l = 0; l < nrun; ++l) sumb += L[(size_t) l].bpe;
            if (sumb) {
                const uint64_t budget = (uint64_t) o.ram_cache_mib << 20;
                const int q = (int) std::min<uint64_t>(budget / sumb, (uint64_t) c.n_expert);
                for (int l = 0; l < nrun; ++l) L[(size_t) l].ncache = q;
                uint64_t left = budget - (uint64_t) q * sumb;
                bool grew = true;
                while (grew) {  // spend the remainder one slot at a time, last layers first
                    grew = false;
                    for (int l = nrun - 1; l >= 0; --l) {
                        Layer& y = L[(size_t) l];
                        if (y.bpe == 0 || y.ncache >= c.n_expert || left < y.bpe) continue;
                        ++y.ncache; left -= y.bpe; grew = true;
                    }
                }
                for (int l = 0; l < nrun; ++l) {
                    Layer& y = L[(size_t) l];
                    if (y.ncache <= 0) continue;
                    y.cpool.resize((size_t) y.ncache * y.bpe);
                    y.cslot_of.assign((size_t) c.n_expert, -1);
                    y.cocc.assign((size_t) y.ncache, -1);
                    y.cstamp.assign((size_t) y.ncache, 0);
                    cache_bytes_plan += y.cpool.size(); cache_slots_plan += y.ncache;
                }
            }
        }
        // ---- VRAM preload: hottest experts per layer
        int preloaded = 0;
        if (o.gpu && have_profile)
            for (int l = 0; l < nrun; ++l) {
                Layer& y = L[(size_t) l];
                std::vector<int> ord(c.n_expert); std::iota(ord.begin(), ord.end(), 0);
                std::stable_sort(ord.begin(), ord.end(), [&](int a, int b) { return y.freq[(size_t) a] > y.freq[(size_t) b]; });
                for (int k = 0; k < y.slots && k < c.n_expert && y.freq[(size_t) ord[(size_t) k]] > 0; ++k) { upload_slot(y, k, ord[(size_t) k]); ++preloaded; }
            }
        if (o.verbose) {
            int64_t sl = 0; for (int l = 0; l < nrun; ++l) sl += L[(size_t) l].slots;
            std::fprintf(stderr, "expert profile: %s\nexperts in RAM: %d (%.2f GiB) | VRAM slots: %lld, preloaded: %d | rest: mmap on disk\n",
                         have_profile ? ("loaded " + o.profile_in).c_str() : "none", ram_n, ram_bytes / 1073741824.0, (long long) sl, preloaded);
            std::fprintf(stderr, "miss LRU arena: %d slots (%.2f GiB) | threads: %d\n",
                         cache_slots_plan, cache_bytes_plan / 1073741824.0, o.threads);
            if (o.gpu) {
                size_t fb2 = 0, tb2 = 0; gpu::mem_info(&fb2, &tb2);
                std::fprintf(stderr, "device: %s | VRAM %.2f/%.2f GiB used, %.2f GiB free (weights %.2f GiB, scratch %.2f GiB)\n",
                             gpu::is_emulated() ? "CPU-EMULATED" : "CUDA",
                             vram_used / 1073741824.0, tb2 / 1073741824.0, fb2 / 1073741824.0,
                             (vram_used - vram_scratch) / 1073741824.0, vram_scratch / 1073741824.0);
            }
        }
        return true;
    }

    void reset() {
        for (Layer& y : L) {
            for (Comp* cp : {&y.ac, &y.ic}) if (cp->ratio) {
                cp->kvs.assign((size_t) cp->coff * cp->ratio * cp->coff * cp->d, 0.f);
                cp->scs.assign((size_t) cp->coff * cp->ratio * cp->coff * cp->d, -INFINITY);
            }
        }
    }

    // ------------------------------------------------------------ attention
    bool comp_step(Comp& cp, const float* x, int pos, const float* cosT, const float* sinT) {
        const int ratio = cp.ratio, d = cp.d, W = cp.coff * d;
        std::vector<float> kv((size_t) W), sc((size_t) W);
        mv(cp.wkv, x, kv.data()); mv(cp.wg, x, sc.data());
        const float* ape = cp.ape.f() + (size_t) (pos % ratio) * W;
        for (int i = 0; i < W; ++i) sc[(size_t) i] += ape[i];
        const int row = (cp.overlap ? ratio : 0) + pos % ratio;
        std::memcpy(&cp.kvs[(size_t) row * W], kv.data(), (size_t) W * 4);
        std::memcpy(&cp.scs[(size_t) row * W], sc.data(), (size_t) W * 4);
        if ((pos + 1) % ratio) return false;
        std::vector<float> out((size_t) d);
        const int T = cp.overlap ? 2 * ratio : ratio;
        for (int ch = 0; ch < d; ++ch) {
            double mx = -INFINITY;
            auto v_at = [&](int t) { return cp.kvs[(size_t) t * W + ((cp.overlap && t >= ratio) ? d : 0) + ch]; };
            auto s_at = [&](int t) { return cp.scs[(size_t) t * W + ((cp.overlap && t >= ratio) ? d : 0) + ch]; };
            for (int t = 0; t < T; ++t) mx = std::max(mx, (double) s_at(t));
            double z = 0, a = 0;
            for (int t = 0; t < T; ++t) { const double e = std::exp((double) s_at(t) - mx); z += e; a += e * v_at(t); }
            out[(size_t) ch] = (float) (a / z);
        }
        if (cp.overlap) {
            std::memmove(&cp.kvs[0], &cp.kvs[(size_t) ratio * W], (size_t) ratio * W * 4);
            std::memmove(&cp.scs[0], &cp.scs[(size_t) ratio * W], (size_t) ratio * W * 4);
        }
        rmsnorm(out.data(), cp.norm.f(), c.rms_eps, d, out.data());
        const int fr = pos + 1 - ratio;
        rotary(out.data(), d, rd, cosT + (size_t) fr * (rd / 2), sinT + (size_t) fr * (rd / 2), false);
        if (cp.rotate) { hadamard(out.data(), d); if (o.qat_sim) fp4_sim(out.data(), d, 32); }
        else if (o.qat_sim) fp8_sim(out.data(), d - rd, 64);
        std::memcpy(cp.cache + (size_t) (pos / ratio) * d, out.data(), (size_t) d * 4);
        return true;
    }

    std::vector<int> indexer(Layer& y, const float* x, const float* qr, int pos, const float* cs, const float* sn, const float* cosT, const float* sinT) {
        const int ih = c.idx_heads, id = c.idx_key_len;
        std::vector<float> q((size_t) ih * id);
        mv(y.i_q_b, qr, q.data());
        for (int hh = 0; hh < ih; ++hh) {
            float* qh = &q[(size_t) hh * id];
            rotary(qh, id, rd, cs, sn, false);
            hadamard(qh, id);
            if (o.qat_sim) fp4_sim(qh, id, 32);
        }
        comp_step(y.ic, x, pos, cosT, sinT);
        std::vector<float> wgt((size_t) ih);
        cpu_mv(y.i_proj.type, y.i_proj.h, ih, dim, x, wgt.data());
        const float ws = std::pow((float) id, -0.5f) * std::pow((float) ih, -0.5f);
        for (float& v : wgt) v *= ws;
        const int ncomp = (pos + 1) / y.ratio, k = std::min(c.idx_topk, ncomp);
        std::vector<int> sel;
        if (k <= 0) return sel;
        std::vector<float> score((size_t) ncomp, 0.f);
        for (int t = 0; t < ncomp; ++t) {
            const float* kt = y.ic.cache + (size_t) t * id; float acc = 0.f;
            for (int hh = 0; hh < ih; ++hh) {
                float dot = 0.f; const float* qh = &q[(size_t) hh * id];
                for (int i = 0; i < id; ++i) dot += qh[i] * kt[i];
                acc += (dot > 0.f ? dot : 0.f) * wgt[(size_t) hh];
            }
            score[(size_t) t] = acc;
        }
        std::vector<int> ord((size_t) ncomp); std::iota(ord.begin(), ord.end(), 0);
        std::partial_sort(ord.begin(), ord.begin() + k, ord.end(), [&](int a, int b) { return score[(size_t) a] != score[(size_t) b] ? score[(size_t) a] > score[(size_t) b] : a < b; });
        for (int i = 0; i < k; ++i) sel.push_back(ord[(size_t) i] + win);
        return sel;
    }

    void attention(int l, const float* x, int pos, float* out) {
        Layer& y = L[(size_t) l];
        const int nh = c.n_head;
        const bool yarn = y.ratio > 0;
        const float* cosT = yarn ? cos_y.data() : cos_p.data(); const float* sinT = yarn ? sin_y.data() : sin_p.data();
        const float* cs = cosT + (size_t) pos * (rd / 2); const float* sn = sinT + (size_t) pos * (rd / 2);
        std::vector<float> qa((size_t) c.q_lora), qr((size_t) c.q_lora), q((size_t) nh * hd);
        mv(y.q_a, x, qa.data());
        rmsnorm(qa.data(), y.q_a_norm.f(), c.rms_eps, c.q_lora, qr.data());
        mv(y.q_b, qr.data(), q.data());
        for (int hh = 0; hh < nh; ++hh) { float* qh = &q[(size_t) hh * hd]; rmsnorm(qh, nullptr, c.rms_eps, hd, qh); rotary(qh, hd, rd, cs, sn, false); }
        std::vector<float> kv((size_t) hd), kvn((size_t) hd);
        mv(y.kv, x, kv.data());
        rmsnorm(kv.data(), y.kv_norm.f(), c.rms_eps, hd, kvn.data());
        rotary(kvn.data(), hd, rd, cs, sn, false);
        if (o.qat_sim) fp8_sim(kvn.data(), hd - rd, 64);
        std::memcpy(&y.kvbuf[(size_t) (pos % win) * hd], kvn.data(), (size_t) hd * 4);
        std::vector<int> idx, tmp; int r, cc;
        window_topk(win, 1, pos, idx, r, cc);
        if (y.ratio) {
            comp_step(y.ac, x, pos, cosT, sinT);
            if (y.has_idx) tmp = indexer(y, x, qr.data(), pos, cs, sn, cosT, sinT);
            else compress_topk(y.ratio, 1, pos, win, tmp, r, cc);
            idx.insert(idx.end(), tmp.begin(), tmp.end());
        }
        std::vector<float> o_(( size_t) nh * hd);
        sparse_attn_token(q.data(), nh, hd, y.kvbuf.data(), idx.data(), (int) idx.size(), y.sinks.f(), 1.f / std::sqrt((float) hd), o_.data());
        for (int hh = 0; hh < nh; ++hh) rotary(&o_[(size_t) hh * hd], hd, rd, cs, sn, true);
        const int G = c.o_groups; const int gin = nh * hd / G;
        std::vector<float> t((size_t) G * c.o_lora);
        for (int g = 0; g < G; ++g) mv(y.o_a, &o_[(size_t) g * gin], &t[(size_t) g * c.o_lora], (int64_t) g * c.o_lora, c.o_lora);
        mv(y.o_b, t.data(), out);
    }

    // ------------------------------------------------------------ MoE (router -> HIT on GPU / MISS on CPU -> shared expert)
    void moe(int l, const float* x, int token, float* y_out) {
        Layer& y = L[(size_t) l];
        const int K = c.n_expert_used, ff = c.ff_exp, E = c.n_expert;
        std::vector<float> lg((size_t) E);
        mv(y.gate_inp, x, lg.data());
        int32_t idx[16]; float w[16];
        const bool hash = l < c.n_hash_layers;
        const Score fn = c.gating_func == 1 ? Score::Softmax : c.gating_func == 2 ? Score::Sigmoid : Score::SqrtSoftplus;
        gate_route(lg.data(), E, K, fn, (!hash && y.exp_bias) ? y.exp_bias.f() : nullptr, hash ? (const int32_t*) y.tid2eid.h + (size_t) token * K : nullptr, c.exp_w_scale, idx, w);
        for (int k = 0; k < K; ++k) { last_routing_p[(size_t) l * K + k] = idx[k]; y.freq[(size_t) idx[k]]++; }

        // Every routed expert goes to the device when it can: the resident ones by VRAM pointer, the
        // others by host pointer through the staging pool. Only when the device refuses (no kernel for
        // these types, or no staging pool) does an expert fall back to the host loop below.
        gpu::ExpPtrs p{};   // value-initialised: n = 0, pointers unused beyond n
        int miss[16], nmiss = 0, nres = 0, nstaged = 0, nstaged_dev = 0;
        const bool device_experts = o.gpu && y.gpu_experts && (d_stage || gpu::is_emulated());
        p.bpe = y.bpe;
        for (int k = 0; k < K; ++k) {
            const int e = idx[k];
            int s = (o.gpu && y.slots > 0) ? y.slot_of[(size_t) e] : -1;
            if (s < 0 && o.gpu && y.slots > 0) {   // admit into a free slot (static profile mode fills the rest on first use)
                for (int q = 0; q < y.slots; ++q) if (y.occ[(size_t) q] < 0) { upload_slot(y, q, e); ++st->admits; s = q; break; }
            }
            if (s >= 0) {
                if (p.n >= gpu::kMaxHit) { miss[nmiss++] = k; continue; }
                uint8_t* base = y.dpool + (size_t) s * y.bpe;
                p.gate[p.n] = base; p.up[p.n] = base + y.bg; p.down[p.n] = base + y.bg + y.bu; p.w[p.n] = w[k]; ++p.n;
                ++nres;
            } else {
                missed_now.push_back({l, e});
                if (!device_experts || p.n >= gpu::kMaxHit) { miss[nmiss++] = k; ++nstaged; continue; }
                ExpSrc s2 = src_miss(y, e);
                p.hg[p.n] = s2.g; p.hu[p.n] = s2.u; p.hd[p.n] = s2.d; p.w[p.n] = w[k]; ++p.n;
                ++nstaged_dev;
            }
        }
        // hits/misses count VRAM residency, not which unit did the arithmetic: an expert staged from
        // host memory is still a miss for the residency plan, and the hit rate the report shows is what
        // tells the operator whether the slot budget is right.
        st->hits += (uint64_t) nres;
        st->misses += (uint64_t) (nstaged + nmiss + nstaged_dev);
        const float lim = c.swiglu_clamp_exp[(size_t) l];
        auto t0 = Clock::now();
        bool hit_ok = false;
        if (p.n > 0) {  // launch the experts asynchronously...
            gpu::h2d(d_x, x, (size_t) dim * 4);
            hit_ok = gpu::experts_hit(p, y.eg.type, y.eu.type, y.ed.type, ff, dim, lim, d_x, d_g, d_u, d_a, d_yh);
            if (!hit_ok) {   // no kernel, or a device error: those experts are computed on the host instead
                nmiss = 0;
                for (int k = 0; k < K; ++k) miss[nmiss++] = k;
                st->hits -= (uint64_t) nres;   // they were already counted as misses: residency, not device
                p.n = 0;
                if (!warned_host_fallback) {
                    warned_host_fallback = true;
                    std::fprintf(stderr, "warning: no device kernel for the experts of layer %d (%s/%s/%s): they run on the host\n",
                                 l, ggml_type_str(y.eg.type).c_str(), ggml_type_str(y.eu.type).c_str(), ggml_type_str(y.ed.type).c_str());
                }
            }
        }
        std::vector<float> acc((size_t) dim, 0.f), g((size_t) ff), u((size_t) ff), a((size_t) ff), yy((size_t) dim);
        for (int mi = 0; mi < nmiss; ++mi) {  // ...while the CPU computes whatever the device could not take
            const int k = miss[mi]; ExpSrc s = src_miss(y, idx[k]);
            cpu_mv(y.eg.type, s.g, ff, dim, x, g.data()); cpu_mv(y.eu.type, s.u, ff, dim, x, u.data());
            swiglu_clamped(g.data(), u.data(), ff, lim, a.data());
            for (float& v : a) v *= w[k];
            cpu_mv(y.ed.type, s.d, dim, ff, a.data(), yy.data());
            for (int i = 0; i < dim; ++i) acc[(size_t) i] += yy[(size_t) i];
        }
        auto t1 = Clock::now();
        if (hit_ok && p.n > 0) {
            std::vector<float> yh((size_t) dim);
            gpu::d2h(yh.data(), d_yh, (size_t) dim * 4);   // blocks until the GPU is done
            for (int i = 0; i < dim; ++i) acc[(size_t) i] += yh[(size_t) i];
        }
        auto t2 = Clock::now();
        st->cpu_miss_s += secs(t0, t1); st->gpu_hit_s += secs(t1, t2);
        // shared expert
        std::vector<float> sg((size_t) ff), su((size_t) ff), sa((size_t) ff), sy((size_t) dim);
        mv(y.sh_gate, x, sg.data()); mv(y.sh_up, x, su.data());
        swiglu_clamped(sg.data(), su.data(), ff, c.swiglu_clamp_shexp[(size_t) l], sa.data());
        mv(y.sh_down, sa.data(), sy.data());
        for (int i = 0; i < dim; ++i) y_out[i] = acc[(size_t) i] + sy[(size_t) i];
    }
    int32_t* last_routing_p = nullptr;

    // ------------------------------------------------------------ block / forward
    void block(int l, std::vector<float>& hres, int token, int pos) {
        Layer& y = L[(size_t) l];
        std::vector<float> yv((size_t) dim), post((size_t) hc), comb((size_t) hc * hc), nrm((size_t) dim), a((size_t) dim), hn((size_t) hc * dim);
        hc_pre(hres.data(), hc, dim, y.hc_attn_fn.f(), y.hc_attn_scale.f(), y.hc_attn_base.f(), c.rms_eps, c.hc_sinkhorn_iters, c.hc_eps, yv.data(), post.data(), comb.data());
        rmsnorm(yv.data(), y.attn_norm.f(), c.rms_eps, dim, nrm.data());
        attention(l, nrm.data(), pos, a.data());
        hc_post(a.data(), hres.data(), post.data(), comb.data(), hc, dim, hn.data());
        hres = hn;
        hc_pre(hres.data(), hc, dim, y.hc_ffn_fn.f(), y.hc_ffn_scale.f(), y.hc_ffn_base.f(), c.rms_eps, c.hc_sinkhorn_iters, c.hc_eps, yv.data(), post.data(), comb.data());
        rmsnorm(yv.data(), y.ffn_norm.f(), c.rms_eps, dim, nrm.data());
        moe(l, nrm.data(), token, a.data());
        hc_post(a.data(), hres.data(), post.data(), comb.data(), hc, dim, hn.data());
        hres = hn;
    }

    void forward(int token, int pos, std::vector<float>* logits) {
        auto t0 = Clock::now();
        std::vector<float> e((size_t) dim), hres((size_t) hc * dim), x((size_t) dim), xn((size_t) dim);
        dequant_row(embd.type, embd.h + (size_t) token * embd.rb, dim, e.data());
        for (int j = 0; j < hc; ++j) std::memcpy(&hres[(size_t) j * dim], e.data(), (size_t) dim * 4);
        for (int l = 0; l < nrun; ++l) block(l, hres, token, pos);
        hc_head(hres.data(), hc, dim, out_hc_fn.f(), out_hc_scale.f(), out_hc_base.f(), c.rms_eps, c.hc_eps, x.data());
        rmsnorm(x.data(), outnorm.f(), c.rms_eps, dim, xn.data());
        if (logits) { logits->assign((size_t) c.vocab, 0.f); mv(outw, xn.data(), logits->data()); }
        st->total_s += secs(t0, Clock::now());
    }

    void end_token() {
        if (o.adapt_swaps > 0 && o.gpu) {
            std::stable_sort(missed_now.begin(), missed_now.end(), [&](const std::pair<int, int>& a, const std::pair<int, int>& b) {
                return L[(size_t) a.first].freq[(size_t) a.second] > L[(size_t) b.first].freq[(size_t) b.second]; });
            int budget = o.adapt_swaps;
            for (auto& m : missed_now) {
                if (budget <= 0) break;
                Layer& y = L[(size_t) m.first];
                if (y.slots == 0 || y.slot_of[(size_t) m.second] >= 0) continue;
                int vs = 0; uint64_t vf = UINT64_MAX;
                for (int s = 0; s < y.slots; ++s) { const uint64_t f = y.occ[(size_t) s] < 0 ? 0 : y.freq[(size_t) y.occ[(size_t) s]]; if (f < vf) { vf = f; vs = s; } }
                if (y.freq[(size_t) m.second] > vf) { upload_slot(y, vs, m.second); ++st->swaps; --budget; }
            }
        }
        missed_now.clear();
        ++st->tokens;
    }

    bool save_profile(std::string& err) const {
        if (o.profile_out.empty()) return true;
        const std::string tmp = o.profile_out + ".tmp";
        FILE* f = std::fopen(tmp.c_str(), "wb");
        if (!f) { err = "cannot write " + tmp; return false; }
        const uint32_t ver = 1, nl = (uint32_t) c.n_layer, ne = (uint32_t) c.n_expert;
        std::fwrite("DSV4PROF", 1, 8, f); std::fwrite(&ver, 4, 1, f); std::fwrite(&nl, 4, 1, f); std::fwrite(&ne, 4, 1, f); std::fwrite(&model_hash, 8, 1, f);
        for (int l = 0; l < c.n_layer; ++l) std::fwrite(L[(size_t) l].freq.data(), 8, (size_t) c.n_expert, f);
        std::fflush(f); fsync(fileno(f)); std::fclose(f);
        if (std::rename(tmp.c_str(), o.profile_out.c_str()) != 0) { err = "rename failed for " + o.profile_out; return false; }
        return true;
    }

    void selfcheck() const {
        std::map<uint32_t, const GgufTensor*> first;
        for (const GgufTensor& t : h.tensors) if (t.dims.size() >= 2 && dequant_supported(t.type) && !first.count(t.type) && t.name.find("tid2eid") == std::string::npos) first[t.type] = &t;
        std::printf("dequantisation self-check (first 3 rows of the first tensor of each type; weights should be finite, ~zero mean, small):\n");
        for (auto& kv : first) {
            const GgufTensor& t = *kv.second;
            const uint8_t* base = maps[(size_t) t.shard] + t.abs_offset;
            const int64_t in = (int64_t) t.dims[0]; const size_t rb = row_bytes(t.type, in);
            std::vector<float> v((size_t) in * 3);
            for (int r = 0; r < 3; ++r) dequant_row(t.type, base + (size_t) r * rb, in, &v[(size_t) r * in]);
            double s = 0, s2 = 0, mx = 0; bool fin = true;
            for (float f : v) { fin = fin && std::isfinite(f); s += f; s2 += (double) f * f; mx = std::max(mx, (double) std::fabs(f)); }
            const double m = s / v.size();
            std::printf("  %-8s %-40s finite=%d mean=%+.5f std=%.5f absmax=%.4f\n", ggml_type_str(kv.first).c_str(), t.name.c_str(), (int) fin, m, std::sqrt(s2 / v.size() - m * m), mx);
        }
    }
};

// ================================================================================================
Model::Model() : p_(new Impl()) { p_->st = &stats; }
Model::~Model() {}
bool Model::load(const std::vector<std::string>& s, const RunOpts& o, std::string& err) {
    if (!p_->load(s, o, err)) return false;
    last_routing.assign((size_t) p_->c.n_layer * p_->c.n_expert_used, -1);
    p_->last_routing_p = last_routing.data();
    return true;
}
void Model::reset() { p_->reset(); }
void Model::forward(int token, int pos, std::vector<float>* logits) { p_->forward(token, pos, logits); }
void Model::end_token() { p_->end_token(); }
bool Model::save_profile(std::string& err) const { return p_->save_profile(err); }
void Model::selfcheck() const { p_->selfcheck(); }
uint64_t Model::model_hash() const { return p_->model_hash; }
const Dsv4Config& Model::config() const { return p_->c; }
const std::vector<std::string>& Model::tokens() const { return p_->toks; }

}  // namespace dsv4
