// src/kernels/qsa_verify_batch_parity.cpp - the opt-in decode QSA selection and attention paths, bitwise (synthetic,
// no model):
//
//   STRATA_BATCH_QSA_ATTN (verify.cpp): a batch window's scores once per slot, one top-k and one qsa_decode_attn_multi
//     for the window, against the per-row calls verify.cpp makes without it.  Window shapes main runs: 2 slots x 1 row
//     (--batch), 2 x 2 and 4 x 2 (--batch-mtp: a token and a proposal per slot), 6 x 1 (more pool sets than one
//     launch takes) and 1 x 4 (one slot).
//   STRATA_QSA_TOPK_SCAN, STRATA_QSA_SCORES_TRANSPOSED, STRATA_ATTN_MERGE_ACTIVE (read once per process): every run
//     checks the selections against qsa_block_topk_ref and the attention against qsa_decode_attn_step (which keeps
//     the original merge) in process.  --write-ref FILE (a run without the switches) stores a hash of every output,
//     --ref FILE (a run with them) requires the same hashes.  ctest runs both.
//
// Shapes as served: 24 q heads, 2 KV heads, head_dim 256, INT8 K/V (an fp16 scale per 64 values), page size 4 through
// a random page permutation, the 4 x 128 indexer with random pooled keys (random selections), a 200,000-cell capacity
// (max_blocks 50,002, which takes the wide top-k kernel, as --max-context 200000 does).  Every row has its own position
// (slot j's rows end 8 j cells before slot 0's).  Also the MTP drafter's call (one row over a 32K window: 512 chunks),
// block scores for 1-8 queries, and the top-k on tie-heavy scores.
//
//   qsa_verify_batch_parity [--selftest] [--ctx 1024,16384,65536] [--write-ref FILE | --ref FILE] [--bench [--reps N]]
//
// --bench adds graph-replay times: us per attention layer (12 layers, each with its own pools) for both window paths
// and their scores / top-k / attention split, the MTP call, and the top-k per launch.
#define DPCT_PROFILING_ENABLED
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include "strata/kernels/kv_q8.hpp"
#include "strata/kernels/qsa.hpp"
#include "strata/kernels/qsa_decode_attn.hpp"
#include "strata/kernels/qsa_select.hpp"
#include "strata/qsa_decode_attn_multi.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <numeric>
#include <random>
#include <string>
#include <vector>

namespace k = strata::kernels;
namespace sx = sycl::ext::oneapi::experimental;

namespace {
int g_fail = 0;
constexpr int kMaxSlots = 6;               // the most slots of a shape below
constexpr int kRows = 8;                   // kVerifyMaxT
constexpr int64_t kCapacity = 200000;      // cells: max_blocks 50,002

uint32_t hsh(uint32_t x) {
    x ^= x >> 16; x *= 0x7feb352du; x ^= x >> 15; x *= 0x846ca68bu; x ^= x >> 16;
    return x;
}
void fill_i8(sycl::queue& q, int8_t* p, size_t n, uint32_t seed) {
    const uint32_t s0 = hsh(seed * 0x9e3779b9u + 1);
    q.parallel_for(sycl::range<1>(n), [=](sycl::id<1> i) {
        const uint32_t h = hsh((uint32_t) i[0] ^ s0) ^ hsh((uint32_t) (i[0] >> 32) + s0);
        p[i] = (int8_t) ((int) (h % 255u) - 127);
    });
}
void fill_scale(sycl::queue& q, uint16_t* p, size_t n, uint32_t seed) {
    const uint32_t s0 = hsh(seed * 0x9e3779b9u + 7);
    q.parallel_for(sycl::range<1>(n), [=](sycl::id<1> i) {
        const uint32_t h = hsh((uint32_t) i[0] ^ s0);
        p[i] = sycl::bit_cast<uint16_t>(sycl::half(0.002f + (float) (h % 1000u) * 2e-5f));
    });
}
void fill_f32(sycl::queue& q, float* p, size_t n, uint32_t seed, float amp) {
    const uint32_t s0 = hsh(seed * 0x9e3779b9u + 13);
    q.parallel_for(sycl::range<1>(n), [=](sycl::id<1> i) {
        const uint32_t h = hsh((uint32_t) i[0] ^ s0);
        p[i] = ((float) (h & 0xffffffu) / 16777216.0f * 2.0f - 1.0f) * amp;
    });
}
template <typename T> T* dalloc(sycl::queue& q, size_t n) {
    T* p = sycl::malloc_device<T>(n + 64, q);
    if (!p) { std::fprintf(stderr, "malloc_device failed (%zu)\n", n); std::exit(2); }
    q.memset(p, 0, (n + 64) * sizeof(T)).wait();
    return p;
}
template <typename T> std::vector<T> fetch(sycl::queue& q, const T* d, size_t n) {
    std::vector<T> h(n);
    q.memcpy(h.data(), d, n * sizeof(T)).wait();
    return h;
}
uint64_t fnv(const void* p, size_t n, uint64_t h = 1469598103934665603ull) {
    const uint8_t* b = (const uint8_t*) p;
    for (size_t i = 0; i < n; ++i) { h ^= b[i]; h *= 1099511628211ull; }
    return h;
}
double now_us() {
    return std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

struct SlotLayer {   // one slot's QSA state of one layer
    int8_t *k_q = nullptr, *v_q = nullptr;
    uint16_t *k_s = nullptr, *v_s = nullptr;
    int32_t* pt = nullptr;
    float *pooled = nullptr, *dead = nullptr;
    k::QsaAttnPools pools() const {
        k::QsaAttnPools p;
        p.k_q = k_q; p.v_q = v_q; p.k_scale = k_s; p.v_scale = v_s; p.page_table = pt;
        return p;
    }
};
struct Shape {
    const char* name;
    int slots, per;   // rows = slots * per, a slot's rows contiguous
};
struct Ctx {
    sycl::queue& q;
    k::QsaShapes s;
    int64_t ctx = 0;
    int layers = 0;
    int64_t cap = 0, max_blocks = 0, scratch_floats = 0;
    std::vector<SlotLayer> sl;   // [layers][kMaxSlots]
    float *qcur = nullptr, *qidx = nullptr, *scores = nullptr, *scratch = nullptr, *attn = nullptr;
    int32_t *sel = nullptr, *steps = nullptr;
    std::vector<int32_t> h_steps;
    SlotLayer& at(int l, int j) { return sl[(size_t) l * kMaxSlots + j]; }
};

void make_ctx(Ctx& c) {
    sycl::queue& q = c.q;
    const k::QsaShapes& s = c.s;
    const int64_t HD = s.head_dim, NKV = s.n_head_kv, NG = HD / k::KV_Q8_GROUP;
    c.cap = k::qsa_selection_width(k::kTopkMaxCells, s);
    c.max_blocks = kCapacity / s.idx_block + 2;
    c.scratch_floats = (int64_t) k::qsa_decode_attn_scratch_floats(c.cap, s);
    c.sl.resize((size_t) c.layers * kMaxSlots);
    std::mt19937 rng(1234u + (uint32_t) c.ctx);
    const int64_t cells = ((c.ctx + 64 + 3) / 4) * 4, pages = cells / s.page_size;
    for (int l = 0; l < c.layers; ++l)
        for (int j = 0; j < kMaxSlots; ++j) {
            SlotLayer& x = c.at(l, j);
            const size_t rows = (size_t) pages * NKV * s.page_size;
            x.k_q = dalloc<int8_t>(q, rows * HD);
            x.v_q = dalloc<int8_t>(q, rows * HD);
            x.k_s = dalloc<uint16_t>(q, rows * NG);
            x.v_s = dalloc<uint16_t>(q, rows * NG);
            const uint32_t seed = (uint32_t) (l * kMaxSlots + j + 1) * 101u + (uint32_t) c.ctx;
            fill_i8(q, x.k_q, rows * HD, seed);
            fill_i8(q, x.v_q, rows * HD, seed + 1);
            fill_scale(q, x.k_s, rows * NG, seed + 2);
            fill_scale(q, x.v_s, rows * NG, seed + 3);
            std::vector<int32_t> perm((size_t) pages);
            std::iota(perm.begin(), perm.end(), 0);
            std::shuffle(perm.begin(), perm.end(), rng);
            x.pt = dalloc<int32_t>(q, (size_t) pages);
            q.memcpy(x.pt, perm.data(), perm.size() * 4).wait();
            const size_t prow = (size_t) (cells / s.idx_block + 2);
            x.pooled = dalloc<float>(q, prow * s.idx_dim);
            x.dead = dalloc<float>(q, (size_t) s.idx_dim);
            fill_f32(q, x.pooled, prow * s.idx_dim, seed + 4, 1.0f);
            fill_f32(q, x.dead, (size_t) s.idx_dim, seed + 5, 1.0f);
        }
    c.qcur = dalloc<float>(q, (size_t) kRows * s.n_head * HD);
    c.qidx = dalloc<float>(q, (size_t) kRows * s.idx_n_head * s.idx_dim);
    c.scores = dalloc<float>(q, (size_t) kRows * c.max_blocks);
    c.scratch = dalloc<float>(q, (size_t) kRows * c.scratch_floats);
    c.attn = dalloc<float>(q, (size_t) c.layers * kRows * s.n_head * HD);
    c.sel = dalloc<int32_t>(q, (size_t) c.layers * kRows * c.cap);
    c.steps = dalloc<int32_t>(q, (size_t) kRows * k::kStepCount);
    fill_f32(q, c.qcur, (size_t) kRows * s.n_head * HD, 77, 1.0f);
    fill_f32(q, c.qidx, (size_t) kRows * s.idx_n_head * s.idx_dim, 78, 1.0f);
    q.wait();
}
void free_ctx(Ctx& c) {
    sycl::queue& q = c.q;
    for (auto& x : c.sl)
        for (void* p : {(void*) x.k_q, (void*) x.v_q, (void*) x.k_s, (void*) x.v_s, (void*) x.pt, (void*) x.pooled,
                        (void*) x.dead})
            sycl::free(p, q);
    for (void* p : {(void*) c.qcur, (void*) c.qidx, (void*) c.scores, (void*) c.scratch, (void*) c.attn, (void*) c.sel,
                    (void*) c.steps})
        sycl::free(p, q);
    c.sl.clear();
}

// the rows' steps: slot j's rows at ctx - 8 j + r
void set_steps(Ctx& c, const Shape& sh) {
    c.h_steps.assign((size_t) kRows * k::kStepCount, 0);
    for (int j = 0; j < sh.slots; ++j)
        for (int r = 0; r < sh.per; ++r)
            k::qsa_step_fill(c.h_steps.data() + (size_t) (j * sh.per + r) * k::kStepCount, c.ctx - 8 * j + r, c.s);
    c.q.memcpy(c.steps, c.h_steps.data(), c.h_steps.size() * 4).wait();
}
// the selection a row returns: its first min(n_kv, width) ids
int64_t n_sel(const Ctx& c, int t) {
    const int32_t* st = c.h_steps.data() + (size_t) t * k::kStepCount;
    return std::min<int64_t>(st[k::kStepNKv], st[k::kStepWidth]);
}

// kRow: verify.cpp's batch window without STRATA_BATCH_QSA_ATTN (every call one row); kOne: with it (scores per slot,
// one top-k, one qsa_decode_attn_multi); kRef: per row, qsa_block_topk_ref and qsa_decode_attn_step.
enum Path { kRow = 0, kOne = 1, kRef = 2 };
const char* const path_name[3] = {"row", "one", "ref"};

// one layer's selection and attention for the window; mask: 1 scores, 2 top-k, 4 attention
void attn_layer(Ctx& c, const Shape& sh, int l, Path p, int mask = 7) {
    const k::QsaShapes& s = c.s;
    void* st = &c.q;
    const int64_t NH = s.n_head, HD = s.head_dim, IQ = s.idx_n_head, ID = s.idx_dim;
    const int n = sh.slots * sh.per;
    float* attn_l = c.attn + (size_t) l * kRows * NH * HD;
    int32_t* sel_l = c.sel + (size_t) l * kRows * c.cap;
    if (p != kOne) {
        for (int t = 0; t < n; ++t) {
            const SlotLayer& x = c.at(l, t / sh.per);
            const int32_t* st_t = c.steps + t * k::kStepCount;
            float* sc_t = c.scores + (size_t) t * c.max_blocks;
            int32_t* sel_t = sel_l + (size_t) t * c.cap;
            if (mask & 1)
                k::qsa_block_scores(x.pooled, x.dead, c.qidx + t * IQ * ID, st_t, 1, c.max_blocks, s, sc_t, st);
            if (mask & 2) {
                if (p == kRef) k::qsa_block_topk_ref(sc_t, st_t, 1, c.max_blocks, c.cap, s, sel_t, st);
                else k::qsa_block_topk(sc_t, st_t, 1, c.max_blocks, c.cap, s, sel_t, st);
            }
            if (mask & 4) {
                float* scr = c.scratch + (size_t) t * c.scratch_floats;
                if (p == kRef) k::qsa_decode_attn_step(c.qcur + t * NH * HD, x.pools(), sel_t, st_t, c.cap, s, scr,
                                                       attn_l + t * NH * HD, st);
                else k::qsa_decode_attn_batch(c.qcur + t * NH * HD, x.pools(), sel_t, st_t, c.cap, s, scr,
                                              attn_l + t * NH * HD, 1, st);
            }
        }
        return;
    }
    k::QsaAttnPools pp[kMaxSlots];
    int rows[kMaxSlots];
    for (int j = 0; j < sh.slots; ++j) {
        const int f = j * sh.per;
        const SlotLayer& x = c.at(l, j);
        if (mask & 1)
            k::qsa_block_scores(x.pooled, x.dead, c.qidx + f * IQ * ID, c.steps + f * k::kStepCount, sh.per,
                                c.max_blocks, s, c.scores + (size_t) f * c.max_blocks, st);
        pp[j] = x.pools();
        rows[j] = sh.per;
    }
    if (mask & 2) k::qsa_block_topk(c.scores, c.steps, n, c.max_blocks, c.cap, s, sel_l, st);
    if (mask & 4) k::qsa_decode_attn_multi(c.qcur, pp, rows, sh.slots, sel_l, c.steps, c.cap, s, c.scratch, attn_l, st);
}

template <class F> double time_graph(sycl::queue& q, int reps, F&& body) {
    sx::command_graph g{q.get_context(), q.get_device()};
    g.begin_recording(q);
    body();
    g.end_recording();
    auto ex = g.finalize();
    for (int i = 0; i < 3; ++i) q.ext_oneapi_graph(ex);
    q.wait();
    std::vector<double> v;
    for (int b = 0; b < 7; ++b) {
        const double t0 = now_us();
        for (int r = 0; r < reps; ++r) q.ext_oneapi_graph(ex);
        q.wait();
        v.push_back((now_us() - t0) / reps);
    }
    std::sort(v.begin(), v.end());
    return v[v.size() / 2];
}

std::map<std::string, uint64_t> g_ref, g_out;
bool g_compare = false;
void record(const std::string& key, uint64_t h) {
    g_out[key] = h;
    if (!g_compare) return;
    auto it = g_ref.find(key);
    if (it == g_ref.end()) { std::printf("  ref %-28s missing\n", key.c_str()); ++g_fail; }
    else if (it->second != h) {
        std::printf("  ref %-28s DIFFERS from the run without the switches\n", key.c_str());
        ++g_fail;
    }
}
void same_f32(const char* what, const float* a, const float* b, size_t n) {
    size_t nd = 0;
    double mabs = 0;
    for (size_t i = 0; i < n; ++i)
        if (std::memcmp(&a[i], &b[i], 4) != 0) {
            ++nd;
            mabs = std::max(mabs, std::fabs((double) a[i] - (double) b[i]));
        }
    if (nd) { std::printf("  %-34s %zu of %zu floats differ (max abs %.3g)\n", what, nd, n, mabs); ++g_fail; }
}
// rows [0, n) of two [kRows, cap] selections: each row's first n_sel ids
void same_sel(const char* what, const Ctx& c, int n, const int32_t* a, const int32_t* b) {
    for (int t = 0; t < n; ++t)
        if (std::memcmp(a + (size_t) t * c.cap, b + (size_t) t * c.cap, (size_t) n_sel(c, t) * 4) != 0) {
            std::printf("  %-34s row %d: the selection differs\n", what, t);
            ++g_fail;
        }
}
uint64_t sel_hash(const Ctx& c, int n, const int32_t* v, uint64_t h) {
    for (int t = 0; t < n; ++t) h = fnv(v + (size_t) t * c.cap, (size_t) n_sel(c, t) * 4, h);
    return h;
}
}  // namespace

int main(int argc, char** argv) {
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    std::vector<int64_t> ctxs = {1024, 16384, 65536};
    std::string write_ref, ref;
    bool bench = false;
    int reps = 20;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--selftest") {}
        else if (a == "--bench") bench = true;
        else if (a == "--ctx" && i + 1 < argc) {
            ctxs.clear();
            const std::string v = argv[++i];
            for (size_t p = 0; p < v.size();) {
                size_t e = v.find(',', p);
                if (e == std::string::npos) e = v.size();
                ctxs.push_back(std::atoll(v.substr(p, e - p).c_str()));
                p = e + 1;
            }
        } else if (a == "--write-ref" && i + 1 < argc) write_ref = argv[++i];
        else if (a == "--ref" && i + 1 < argc) ref = argv[++i];
        else if (a == "--reps" && i + 1 < argc) reps = std::atoi(argv[++i]);
        else {
            std::fprintf(stderr, "usage: qsa_verify_batch_parity [--selftest] [--ctx 1024,16384,65536] "
                                 "[--write-ref FILE | --ref FILE] [--bench [--reps N]]\n");
            return 2;
        }
    }
    if (!ref.empty()) {
        FILE* f = std::fopen(ref.c_str(), "r");
        if (!f) { std::fprintf(stderr, "qsa_verify_batch_parity: cannot read %s\n", ref.c_str()); return 2; }
        char key[128];
        unsigned long long h = 0;
        while (std::fscanf(f, "%127s %llx", key, &h) == 2) g_ref[key] = h;
        std::fclose(f);
        g_compare = true;
    }
    sycl::queue q{sycl::gpu_selector_v, {sycl::property::queue::in_order()}};
    std::printf("qsa_verify_batch_parity: %s, switches: TOPK_SCAN=%s SCORES_TRANSPOSED=%s MERGE_ACTIVE=%s, ref: %s\n",
                q.get_device().get_info<sycl::info::device::name>().c_str(),
                std::getenv("STRATA_QSA_TOPK_SCAN") ? std::getenv("STRATA_QSA_TOPK_SCAN") : "-",
                std::getenv("STRATA_QSA_SCORES_TRANSPOSED") ? std::getenv("STRATA_QSA_SCORES_TRANSPOSED") : "-",
                std::getenv("STRATA_ATTN_MERGE_ACTIVE") ? std::getenv("STRATA_ATTN_MERGE_ACTIVE") : "-",
                g_compare ? "compare" : (write_ref.empty() ? "none" : "write"));
    const Shape shapes[] = {{"2x1", 2, 1}, {"2x2", 2, 2}, {"4x2", 4, 2}, {"6x1", 6, 1}, {"1x4", 1, 4}};
    for (int64_t ctx : ctxs) {
        Ctx c{q, k::qsa_real_shapes()};
        c.ctx = ctx;
        c.layers = bench ? 12 : 2;
        make_ctx(c);
        const int64_t NH = c.s.n_head, HD = c.s.head_dim;
        std::printf("\n=== context %lld (cap %lld, max_blocks %lld, %d layers) ===\n", (long long) ctx,
                    (long long) c.cap, (long long) c.max_blocks, c.layers);
        const size_t attn_n = (size_t) c.layers * kRows * NH * HD, sel_n = (size_t) c.layers * kRows * c.cap;
        for (const Shape& sh : shapes) {
            const int n = sh.slots * sh.per;
            set_steps(c, sh);
            std::vector<float> a[3];
            std::vector<int32_t> sv[3];
            for (int p : {kRef, kRow, kOne}) {
                q.memset(c.attn, 0, attn_n * 4).wait();
                q.memset(c.sel, 0, sel_n * 4).wait();
                for (int l = 0; l < c.layers; ++l) attn_layer(c, sh, l, (Path) p);
                q.wait();
                a[p] = fetch(q, c.attn, attn_n);
                sv[p] = fetch(q, c.sel, sel_n);
            }
            const int f0 = g_fail;
            for (int p : {kRow, kOne})
                for (int l = 0; l < c.layers; ++l) {
                    char w[96];
                    std::snprintf(w, sizeof w, "%s %s vs ref, layer %d:", sh.name, path_name[p], l);
                    const size_t ao = (size_t) l * kRows * NH * HD, so = (size_t) l * kRows * c.cap;
                    same_f32(w, a[kRef].data() + ao, a[p].data() + ao, (size_t) n * NH * HD);
                    same_sel(w, c, n, sv[kRef].data() + so, sv[p].data() + so);
                }
            const bool ok = g_fail == f0;
            const int hl = std::min(2, c.layers);   // the hashes cover layers 0 and 1, the same with --bench
            uint64_t h = fnv(a[kOne].data(), (size_t) hl * kRows * NH * HD * 4);
            for (int l = 0; l < hl; ++l)
                h = sel_hash(c, n, sv[kOne].data() + (size_t) l * kRows * c.cap, h);
            char key[64];
            std::snprintf(key, sizeof key, "attn_%lld_%s", (long long) ctx, sh.name);
            record(key, h);
            std::printf("  %-4s (%d rows): row, one and ref %s\n", sh.name, n, ok ? "bitwise equal" : "DIFFER");
            if (bench) {
                double t[2], comp[2][3];
                for (int p : {kRow, kOne}) {
                    t[p] = time_graph(q, reps, [&] {
                        for (int l = 0; l < c.layers; ++l) attn_layer(c, sh, l, (Path) p);
                    }) / c.layers;
                    for (int m = 0; m < 3; ++m)
                        comp[p][m] = time_graph(q, reps, [&] {
                            for (int l = 0; l < c.layers; ++l) attn_layer(c, sh, l, (Path) p, 1 << m);
                        }) / c.layers;
                }
                std::printf("  RESULT ctx=%lld shape=%s rows=%d us_per_layer row=%.2f one=%.2f "
                            "split(scores/topk/attn) row=%.1f/%.1f/%.1f one=%.1f/%.1f/%.1f\n",
                            (long long) ctx, sh.name, n, t[kRow], t[kOne], comp[kRow][0], comp[kRow][1], comp[kRow][2],
                            comp[kOne][0], comp[kOne][1], comp[kOne][2]);
            }
        }
        // block scores for 1..8 queries of one slot in one call, against one call per query
        for (int nq = 1; nq <= 8; ++nq) {
            const Shape sh{"S", 1, nq};
            set_steps(c, sh);
            const SlotLayer& x = c.at(0, 0);
            const size_t sn = (size_t) nq * c.max_blocks;
            q.memset(c.scores, 0, sn * 4).wait();
            k::qsa_block_scores(x.pooled, x.dead, c.qidx, c.steps, nq, c.max_blocks, c.s, c.scores, &q);
            q.wait();
            const auto one = fetch(q, c.scores, sn);
            q.memset(c.scores, 0, sn * 4).wait();
            for (int t = 0; t < nq; ++t)
                k::qsa_block_scores(x.pooled, x.dead, c.qidx + (size_t) t * c.s.idx_n_head * c.s.idx_dim,
                                    c.steps + t * k::kStepCount, 1, c.max_blocks, c.s,
                                    c.scores + (size_t) t * c.max_blocks, &q);
            q.wait();
            char w[64];
            std::snprintf(w, sizeof w, "scores, %d queries vs one by one:", nq);
            same_f32(w, fetch(q, c.scores, sn).data(), one.data(), sn);
            char key[64];
            std::snprintf(key, sizeof key, "scores%d_%lld", nq, (long long) ctx);
            record(key, fnv(one.data(), one.size() * 4));
        }
        // the top-k on tie-heavy scores (8 distinct values), 6 queries, against qsa_block_topk_ref
        {
            const Shape sh{"T", 2, 3};
            set_steps(c, sh);
            const uint32_t s0 = 0x51ed27u + (uint32_t) ctx;
            float* sc = c.scores;
            q.parallel_for(sycl::range<1>((size_t) 6 * c.max_blocks), [=](sycl::id<1> i) {
                 sc[i] = (float) (hsh((uint32_t) i[0] ^ s0) % 8u) * 0.25f;
             }).wait();
            q.memset(c.sel, 0, (size_t) kRows * c.cap * 4).wait();
            k::qsa_block_topk(c.scores, c.steps, 6, c.max_blocks, c.cap, c.s, c.sel, &q);
            q.wait();
            const auto ids = fetch(q, c.sel, (size_t) kRows * c.cap);
            q.memset(c.sel, 0, (size_t) kRows * c.cap * 4).wait();
            k::qsa_block_topk_ref(c.scores, c.steps, 6, c.max_blocks, c.cap, c.s, c.sel, &q);
            q.wait();
            same_sel("top-k on ties vs qsa_block_topk_ref:", c, 6, fetch(q, c.sel, (size_t) kRows * c.cap).data(),
                     ids.data());
            char key[64];
            std::snprintf(key, sizeof key, "topk_ties_%lld", (long long) ctx);
            record(key, sel_hash(c, 6, ids.data(), 1469598103934665603ull));
            if (bench) {
                auto topk = [&](int nq) {
                    return time_graph(q, reps, [&] {
                        for (int l = 0; l < c.layers; ++l)
                            k::qsa_block_topk(c.scores, c.steps, nq, c.max_blocks, c.cap, c.s, c.sel, &q);
                    }) / c.layers;
                };
                const double t1 = topk(1), t6 = topk(6);
                std::printf("  RESULT ctx=%lld topk_us_per_launch q1=%.2f q6=%.2f\n", (long long) ctx, t1, t6);
            }
        }
        // the MTP drafter's call: one row over a 32K window (512 chunks), ids = the window's cells
        {
            const int64_t capm = 32768;
            const int nm = std::min(4, c.layers);
            const int64_t sfl = (int64_t) k::qsa_decode_attn_scratch_floats(capm, c.s);
            float* scr = dalloc<float>(q, (size_t) sfl);
            int32_t* idm = dalloc<int32_t>(q, (size_t) capm * nm);
            int32_t* stm = dalloc<int32_t>(q, (size_t) k::kStepCount * nm);
            float* outm = dalloc<float>(q, (size_t) nm * NH * HD);
            const int64_t nkv = ctx + 1, W = std::min<int64_t>(nkv, capm);
            std::vector<int32_t> hid((size_t) capm * nm, 0), hst((size_t) k::kStepCount * nm);
            for (int u = 0; u < nm; ++u) {
                for (int64_t i = 0; i < W; ++i) hid[(size_t) (u * capm + i)] = (int32_t) (nkv - W + i);
                int32_t* sp = hst.data() + u * k::kStepCount;
                sp[k::kStepPos] = (int32_t) (nkv - 1);
                sp[k::kStepNKv] = (int32_t) nkv;
                sp[k::kStepNBid] = (int32_t) (nkv / c.s.idx_block);
                sp[k::kStepWidth] = (int32_t) W;
            }
            q.memcpy(idm, hid.data(), hid.size() * 4).wait();
            q.memcpy(stm, hst.data(), hst.size() * 4).wait();
            auto body = [&](bool step) {
                for (int u = 0; u < nm; ++u) {   // one call per layer, each on that layer's pools
                    const int32_t* id_u = idm + (size_t) u * capm;
                    const int32_t* st_u = stm + u * k::kStepCount;
                    float* out_u = outm + (size_t) u * NH * HD;
                    if (step)
                        k::qsa_decode_attn_step(c.qcur, c.at(u, 0).pools(), id_u, st_u, capm, c.s, scr, out_u, &q);
                    else
                        k::qsa_decode_attn_batch(c.qcur, c.at(u, 0).pools(), id_u, st_u, capm, c.s, scr, out_u, 1, &q);
                }
            };
            body(true);
            q.wait();
            const auto ref_m = fetch(q, outm, (size_t) nm * NH * HD);
            body(false);
            q.wait();
            const auto om = fetch(q, outm, (size_t) nm * NH * HD);
            same_f32("MTP call vs qsa_decode_attn_step:", ref_m.data(), om.data(), om.size());
            char key[64];
            std::snprintf(key, sizeof key, "mtp_%lld", (long long) ctx);
            record(key, fnv(om.data(), (size_t) std::min(2, nm) * NH * HD * 4));
            if (bench) {
                const double tm = time_graph(q, reps, [&] { body(false); }) / nm;
                std::printf("  RESULT ctx=%lld mtp_call_us=%.2f (window %lld)\n", (long long) ctx, tm, (long long) W);
            }
            sycl::free(scr, q); sycl::free(idm, q); sycl::free(stm, q); sycl::free(outm, q);
        }
        free_ctx(c);
    }
    if (!write_ref.empty()) {
        FILE* f = std::fopen(write_ref.c_str(), "w");
        if (!f) { std::fprintf(stderr, "qsa_verify_batch_parity: cannot write %s\n", write_ref.c_str()); return 2; }
        for (auto& kv : g_out) std::fprintf(f, "%s %016llx\n", kv.first.c_str(), (unsigned long long) kv.second);
        std::fclose(f);
        std::printf("\nwrote %zu hashes to %s\n", g_out.size(), write_ref.c_str());
    }
    std::printf("\nqsa_verify_batch_parity: %s (%d failures)\n", g_fail ? "FAIL" : "PASS", g_fail);
    return g_fail ? 1 : 0;
}
