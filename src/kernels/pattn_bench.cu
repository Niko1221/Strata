// src/kernels/pattn_bench.cu - STRATA_PATTN_SM60's single-GPU bench: the prompt path's QSA attention replayed on real
// selections (a STRATA_QSA_DUMP file: int32 {qsa layer, p0, T, cap} + T x cap cell ids per record), fixed-seed int8
// K/V (+ fp16 scales, the `--kv int8` format) and queries, identity page table (page = 4 cells).  12 QSA layers x
// 1,536 queries (the prompt chunk at these contexts) per case.
//
//   production : qsa_decode_attn_batch, 32 queries a call, exactly as src/prefill/prefill.cpp calls it
//   candidates : qsa_pattn_batch variants
// Every candidate's output [T, 24, 256] is compared with production's byte for byte (memcmp), all 12 layers.
// Production selection (exact scores + top-k, 256 queries a call) is timed for the shares.
//
//   pattn_bench <dump> [reps=5] [what=attn,sel] [cases=26k,88k,115k] [variants=all]
#include "strata/kernels/kv_q8.hpp"
#include "strata/kernels/qsa.hpp"
#include "strata/kernels/qsa_decode_attn.hpp"
#include "strata/kernels/qsa_pattn.hpp"
#include "strata/kernels/qsa_select.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <string>
#include <vector>

#if !defined(_WIN32)   // 64-bit file offsets (a dump runs to GBs)
#define _fseeki64 fseeko
#define _ftelli64 ftello
#endif

using namespace strata::kernels;

#define CK(x)                                                                                                    \
    do {                                                                                                         \
        const cudaError_t e_ = (x);                                                                              \
        if (e_ != cudaSuccess) {                                                                                 \
            std::fprintf(stderr, "%s:%d %s: %s\n", __FILE__, __LINE__, #x, cudaGetErrorString(e_));            \
            std::exit(2);                                                                                        \
        }                                                                                                        \
    } while (0)

namespace {

constexpr int HD = 256, NKV = 2, NH = 24, ZV = NH * HD, CAP = 2051;

struct Rng {
    uint64_t s;
    explicit Rng(uint64_t seed) : s(seed * 0x9E3779B97F4A7C15ull + 1) {}
    uint64_t next() { s ^= s >> 12; s ^= s << 25; s ^= s >> 27; return s * 0x2545F4914F6CDD1Dull; }
    float uni() { return (float) ((next() >> 40) * (1.0 / 16777216.0)); }
};
struct Rec { int layer, p0, T, cap; long long off; };
struct Case { std::string name; int p0 = 0, T = 0; std::vector<std::vector<int32_t>> ids; };

std::vector<int32_t> read_ids(const char* fn, const Rec& r) {
    std::vector<int32_t> v((size_t) r.T * r.cap);
    std::FILE* f = std::fopen(fn, "rb");
    if (!f) { std::perror(fn); std::exit(1); }
    _fseeki64(f, r.off + 16, SEEK_SET);
    if (std::fread(v.data(), 4, v.size(), f) != v.size()) { std::fprintf(stderr, "short read\n"); std::exit(1); }
    std::fclose(f);
    return v;
}
const Rec* find_rec(const std::vector<Rec>& recs, int layer, int p0) {
    for (const Rec& r : recs) if (r.layer == layer && r.p0 == p0) return &r;
    return nullptr;
}
double median(std::vector<double> v) {
    std::sort(v.begin(), v.end());
    return v.empty() ? 0.0 : v[v.size() / 2];
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: pattn_bench <qsa_ids dump> [reps=5] [what=attn,sel] [cases=26k,88k,115k] [variants=all]\n");
        return 1;
    }
    const char* dump = argv[1];
    const int reps = argc > 2 ? std::atoi(argv[2]) : 5;
    const std::string what = argc > 3 ? argv[3] : "attn,sel";
    const std::string want_cases = argc > 4 ? argv[4] : "26k,88k,115k";
    const std::string want_var = argc > 5 ? argv[5] : "all";
    const bool do_attn = what.find("attn") != std::string::npos, do_sel = what.find("sel") != std::string::npos;

    int dev = 0;
    CK(cudaGetDevice(&dev));
    cudaDeviceProp pr{};
    CK(cudaGetDeviceProperties(&pr, dev));
    std::printf("device %d: %s (pci %02x:%02x), sm_%d%d, %d SMs, clock %d MHz, mem clock %d MHz\n", dev, pr.name,
                pr.pciBusID, pr.pciDeviceID, pr.major, pr.minor, pr.multiProcessorCount, pr.clockRate / 1000,
                pr.memoryClockRate / 1000);
    {
        char buf[2048];
        qsa_pattn_info(buf, sizeof buf);
        std::printf("pattn kernels:\n%s", buf);
    }
    std::fflush(stdout);

    std::vector<Rec> recs;
    {
        std::FILE* f = std::fopen(dump, "rb");
        if (!f) { std::perror(dump); return 1; }
        _fseeki64(f, 0, SEEK_END);
        const long long size = _ftelli64(f);
        long long off = 0;
        while (off + 16 <= size) {
            _fseeki64(f, off, SEEK_SET);
            int32_t h[4];
            if (std::fread(h, 4, 4, f) != 4) break;
            if (h[3] != CAP || h[2] <= 0 || h[2] > 8192 || h[0] < 0 || h[0] > 11) {
                std::fprintf(stderr, "bad record at %lld\n", off);
                return 1;
            }
            recs.push_back({h[0], h[1], h[2], h[3], off});
            off += 16 + (long long) h[2] * h[3] * 4;
        }
        std::fclose(f);
        std::printf("dump: %zu records\n", recs.size());
    }

    // the cases (they expect a dump of reads at about 26K, 109K and 112K tokens): 26K = two 768-query records of the 26K read; 88K = the 109K
    // record's blocks scaled by 87,804 / 109,308 (no record near 88K); 115K = the 112,380 record (its one missing layer
    // takes the 109,308 record shifted)
    std::vector<Case> cases;
    const auto want = [&](const char* n) { return want_cases.find(n) != std::string::npos; };
    if (want("26k")) {
        Case c; c.name = "26K"; c.p0 = 24576; c.T = 1536;
        for (int L = 0; L < 12; ++L) {
            const Rec *a = find_rec(recs, L, 24576), *b = find_rec(recs, L, 25344);
            if (!a || !b || a->T != 768 || b->T != 768) { std::fprintf(stderr, "26K records missing\n"); return 1; }
            std::vector<int32_t> v = read_ids(dump, *a), w = read_ids(dump, *b);
            v.insert(v.end(), w.begin(), w.end());
            c.ids.push_back(std::move(v));
        }
        cases.push_back(std::move(c));
    }
    if (want("88k")) {
        Case c; c.name = "88K(syn)"; c.p0 = 109308 - 21504; c.T = 1536;
        const double s = (double) c.p0 / 109308.0;
        for (int L = 0; L < 12; ++L) {
            const Rec* a = find_rec(recs, L, 109308);
            if (!a || a->T != 1536) { std::fprintf(stderr, "109K records missing\n"); return 1; }
            std::vector<int32_t> v = read_ids(dump, *a);
            for (int32_t& x : v) x = (int32_t) (4 * (int64_t) std::floor((x / 4) * s) + (x % 4));
            for (int t = 0; t < c.T; ++t) std::sort(v.begin() + (size_t) t * CAP, v.begin() + (size_t) (t + 1) * CAP);
            c.ids.push_back(std::move(v));
        }
        cases.push_back(std::move(c));
    }
    // synthetic short contexts (no dump record): "s0" positions 0..1535 (width = pos + 1 < cap: every cell, the last
    // chunk partial, chunks past the width absent), "s2k" positions 1536..3071 (a contiguous window of `cap` cells)
    for (const int p0 : {0, 1536}) {
        if (!want(p0 == 0 ? "s0" : "s2k")) continue;
        Case c; c.name = p0 == 0 ? "short 0-1.5K (syn)" : "short 1.5K-3K (syn)"; c.p0 = p0; c.T = 1536;
        std::vector<int32_t> v((size_t) c.T * CAP, 0);
        for (int t = 0; t < c.T; ++t) {
            const int n_kv = p0 + t + 1, width = std::min(n_kv, CAP);
            for (int j = 0; j < width; ++j) v[(size_t) t * CAP + j] = n_kv - width + j;
            for (int j = width; j < CAP; ++j) v[(size_t) t * CAP + j] = n_kv - 1;   // unused past the width
        }
        for (int L = 0; L < 12; ++L) c.ids.push_back(v);
        cases.push_back(std::move(c));
    }
    if (want("115k")) {
        const int p0 = 112380;
        Case c; c.name = "115K"; c.p0 = p0; c.T = 1536;
        for (int L = 0; L < 12; ++L) {
            const Rec* a = find_rec(recs, L, p0);
            if (a && a->T == 1536) { c.ids.push_back(read_ids(dump, *a)); continue; }
            const Rec* b = find_rec(recs, L, 109308);
            if (!b || b->T != 1536) { std::fprintf(stderr, "records missing\n"); return 1; }
            std::vector<int32_t> v = read_ids(dump, *b);
            for (int32_t& x : v) x += p0 - 109308;
            std::printf("case %s: layer %d has no record; it uses the 109,308 record shifted by %d\n", c.name.c_str(), L,
                        p0 - 109308);
            c.ids.push_back(std::move(v));
        }
        cases.push_back(std::move(c));
    }
    for (const Case& c : cases)
        for (int L = 0; L < 12; ++L)
            for (int t = 0; t < c.T; ++t) {
                const int32_t* r = c.ids[L].data() + (size_t) t * CAP;
                for (int j = 0; j < CAP; ++j)
                    if (r[j] < 0 || r[j] > c.p0 + t || (j > 0 && r[j] < r[j - 1])) {
                        std::fprintf(stderr, "case %s layer %d query %d: bad id %d at %d\n", c.name.c_str(), L, t, r[j], j);
                        return 1;
                    }
            }

    // K/V: int8 codes + fp16 scales per 64 values; identity page table; one masked page (page -1) to exercise the
    // masked-cell path of every kernel (cells of page `masked_page` are dropped as an unresident KV block would be)
    QsaShapes s = qsa_real_shapes();
    int max_pos = 0;
    for (const Case& c : cases) max_pos = std::max(max_pos, c.p0 + c.T);
    const int64_t n_pages = max_pos / s.page_size + 2;
    const int64_t rows = n_pages * NKV * s.page_size;
    QsaAttnPools pools{};
    {
        Rng rng(7);
        std::vector<int8_t> codes((size_t) rows * HD);
        std::vector<uint16_t> sc((size_t) rows * (HD / KV_Q8_GROUP));
        std::vector<int32_t> pt((size_t) n_pages);
        for (int64_t i = 0; i < n_pages; ++i) pt[i] = (int32_t) i;
        const char* mp = std::getenv("PATTN_MASK_PAGES");   // e.g. "37,4096": pages left unresident (page -1)
        if (mp) {
            std::string m = mp;
            size_t i = 0;
            while (i < m.size()) {
                size_t j = m.find(',', i);
                if (j == std::string::npos) j = m.size();
                const long long pg = std::atoll(m.substr(i, j - i).c_str());
                if (pg >= 0 && pg < n_pages) { pt[(size_t) pg] = -1; std::printf("page %lld masked\n", pg); }
                i = j + 1;
            }
        }
        int8_t *k_q, *v_q;
        uint16_t *k_s, *v_s;
        int32_t* page_table;
        CK(cudaMalloc(&k_q, codes.size()));
        CK(cudaMalloc(&v_q, codes.size()));
        CK(cudaMalloc(&k_s, sc.size() * 2));
        CK(cudaMalloc(&v_s, sc.size() * 2));
        CK(cudaMalloc(&page_table, pt.size() * 4));
        for (int kv = 0; kv < 2; ++kv) {
            for (auto& x : codes) x = (int8_t) ((int) (rng.next() % 255) - 127);
            for (auto& x : sc) x = __half_as_ushort(__float2half(0.002f + 0.028f * rng.uni()));
            CK(cudaMemcpy(kv ? v_q : k_q, codes.data(), codes.size(), cudaMemcpyHostToDevice));
            CK(cudaMemcpy(kv ? v_s : k_s, sc.data(), sc.size() * 2, cudaMemcpyHostToDevice));
        }
        CK(cudaMemcpy(page_table, pt.data(), pt.size() * 4, cudaMemcpyHostToDevice));
        pools.k_q = k_q; pools.v_q = v_q; pools.k_scale = k_s; pools.v_scale = v_s; pools.page_table = page_table;
    }
    const int TMAX = 1536, ATTN_BATCH = 32;   // prefill.cpp: m.attn_batch
    const long long stride = (long long) qsa_decode_attn_scratch_floats(CAP, s);
    const uint64_t scratch_floats = (uint64_t) ATTN_BATCH * (uint64_t) stride;   // prefill.cpp's m.attn_scratch
    float *q = nullptr, *attn_ref = nullptr, *attn_new = nullptr, *scratch = nullptr;
    int32_t *ids_dev = nullptr, *steps_dev = nullptr;
    CK(cudaMalloc(&q, (size_t) TMAX * ZV * 4));
    CK(cudaMalloc(&attn_ref, (size_t) TMAX * ZV * 4));
    CK(cudaMalloc(&attn_new, (size_t) TMAX * ZV * 4));
    CK(cudaMalloc(&scratch, (size_t) scratch_floats * 4));
    CK(cudaMalloc(&ids_dev, (size_t) 12 * TMAX * CAP * 4));
    CK(cudaMalloc(&steps_dev, (size_t) TMAX * kStepCount * 4));
    {
        Rng rng(11);
        std::vector<float> h((size_t) TMAX * ZV);
        for (auto& x : h) x = 2.0f * rng.uni() - 1.0f;
        CK(cudaMemcpy(q, h.data(), h.size() * 4, cudaMemcpyHostToDevice));
    }
    cudaStream_t st;
    CK(cudaStreamCreateWithFlags(&st, cudaStreamNonBlocking));
    cudaEvent_t e0, e1;
    CK(cudaEventCreate(&e0));
    CK(cudaEventCreate(&e1));
    const auto timed = [&](auto&& fn) {
        CK(cudaEventRecord(e0, st));
        fn();
        CK(cudaEventRecord(e1, st));
        CK(cudaEventSynchronize(e1));
        float ms = 0;
        CK(cudaEventElapsedTime(&ms, e0, e1));
        return (double) ms;
    };
    const double fma_attn = 2.0 * 12 * 256 * 2 * 2051;   // per query: scores + values, both KV heads

    struct Var { std::string key, name; std::function<void(int, float*)> run; };
    for (const Case& c : cases) {
        std::printf("\n== case %s: p0 %d, %d queries per call, 12 QSA layers\n", c.name.c_str(), c.p0, c.T);
        {
            std::vector<int32_t> h((size_t) c.T * kStepCount);
            for (int t = 0; t < c.T; ++t) qsa_step_fill(h.data() + (size_t) t * kStepCount, c.p0 + t, s);
            CK(cudaMemcpy(steps_dev, h.data(), h.size() * 4, cudaMemcpyHostToDevice));
            for (int L = 0; L < 12; ++L)
                CK(cudaMemcpy(ids_dev + (size_t) L * TMAX * CAP, c.ids[L].data(), (size_t) c.T * CAP * 4,
                              cudaMemcpyHostToDevice));
        }
        const auto ids_of = [&](int L) { return ids_dev + (size_t) L * TMAX * CAP; };
        double t_prod = 0, t_best = 0;
        if (do_attn) {
            const auto per32 = [&](int L, float* out, const std::function<void(int64_t, int64_t, float*)>& call) {
                for (int t0 = 0; t0 < c.T; t0 += ATTN_BATCH) call(t0, std::min(ATTN_BATCH, c.T - t0), out);
                (void) L;
            };
            const auto prod_layer = [&](int L, float* out) {
                per32(L, out, [&](int64_t t0, int64_t nb, float* o) {
                    qsa_decode_attn_batch(q + (size_t) t0 * ZV, pools, ids_of(L) + (size_t) t0 * CAP,
                                          steps_dev + (size_t) t0 * kStepCount, CAP, s, scratch, o + (size_t) t0 * ZV, nb,
                                          st);
                });
            };
            std::vector<Var> vars;
            for (const int v : {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13}) {
                static const char* names[] = {"", "pattn 1: chunk qreg+Kstage + merge v2", "pattn 2: fused",
                                              "pattn 3: fused, V staged", "pattn 4: fused, V 2 dims/thread",
                                              "pattn 5: fused, K magic", "pattn 6: fused, K magic, V 2 dims",
                                              "pattn 7: fused, reduce24", "pattn 8: fused, magic, V 2 dims, reduce24",
                                              "pattn 9: fused, V 2 dims, reduce24",
                                              "pattn 10: split, V 1 dim", "pattn 11: split, V 2 dims",
                                              "pattn 12: split, reduce24, V 2 dims",
                                              "pattn 13: split, magic, reduce24, V 2 dims"};
                const char* nm = names[v];
                vars.push_back({"p" + std::to_string(v), nm, [&, v](int L, float* out) {
                                    if (!qsa_pattn_batch(q, pools, ids_of(L), steps_dev, CAP, s, scratch, scratch_floats,
                                                         out, c.T, st, v)) {
                                        std::fprintf(stderr, "qsa_pattn_batch refused variant %d\n", v);
                                        std::exit(1);
                                    }
                                }});
            }
            // split combinations (qsa_pattn.cu: 1000 + 100 S + 10 V + mask); PATTN_SPLIT=1003,3013,... adds them,
            // mask 1 / 2 = one kernel alone (timing only: the parity line then says DIFFERENT)
            if (const char* sp = std::getenv("PATTN_SPLIT")) {
                std::string m = sp;
                size_t i = 0;
                while (i < m.size()) {
                    size_t j = m.find(',', i);
                    if (j == std::string::npos) j = m.size();
                    const int code = std::atoi(m.substr(i, j - i).c_str());
                    i = j + 1;
                    if (code < 1000) continue;
                    // their own scratch (production's call reuses `scratch`); a value-kernel-only code first gets
                    // valid blocks from a score-kernel-only run (the last batch's, which every batch then reads)
                    static float* scratch2 = nullptr;
                    // PATTN_SCRATCH_MUL=k (bench only): k times prefill's scratch for the split variants (batch size)
                    static const uint64_t split_floats = scratch_floats * (uint64_t) (std::getenv("PATTN_SCRATCH_MUL")
                                                                                         ? std::atoi(std::getenv("PATTN_SCRATCH_MUL")) : 1);
                    if (!scratch2) CK(cudaMalloc(&scratch2, (size_t) split_floats * 4));
                    vars.push_back({"s" + std::to_string(code), "split " + std::to_string(code), [&, code](int L, float* out) {
                                        static int filled_for = -1;
                                        if (code % 10 == 2 && filled_for != code) {
                                            qsa_pattn_batch(q, pools, ids_of(0), steps_dev, CAP, s, scratch2,
                                                            split_floats, out, c.T, st, 1301);
                                            filled_for = code;
                                        }
                                        if (!qsa_pattn_batch(q, pools, ids_of(L), steps_dev, CAP, s, scratch2,
                                                             split_floats, out, c.T, st, code)) {
                                            std::fprintf(stderr, "qsa_pattn_batch refused %d\n", code);
                                            std::exit(1);
                                        }
                                    }});
                }
            }
            std::vector<float> ref((size_t) c.T * ZV), got((size_t) c.T * ZV);
            const auto parity = [&](const Var& v) {
                long long bad_layers = 0, bad_words = 0;
                for (int L = 0; L < 12; ++L) {
                    CK(cudaMemsetAsync(attn_ref, 0xFF, (size_t) c.T * ZV * 4, st));
                    CK(cudaMemsetAsync(attn_new, 0xEE, (size_t) c.T * ZV * 4, st));
                    prod_layer(L, attn_ref);
                    v.run(L, attn_new);
                    CK(cudaStreamSynchronize(st));
                    CK(cudaMemcpy(ref.data(), attn_ref, ref.size() * 4, cudaMemcpyDeviceToHost));
                    CK(cudaMemcpy(got.data(), attn_new, got.size() * 4, cudaMemcpyDeviceToHost));
                    if (std::memcmp(ref.data(), got.data(), ref.size() * 4) != 0) {
                        ++bad_layers;
                        for (size_t i = 0; i < ref.size(); ++i) bad_words += std::memcmp(&ref[i], &got[i], 4) != 0;
                    }
                }
                std::printf("  parity %-40s %s (%lld of 12 layers differ, %lld words)\n", v.name.c_str(),
                            bad_layers == 0 ? "IDENTICAL (memcmp, 12 layers)" : "DIFFERENT", bad_layers, bad_words);
                std::fflush(stdout);
                return bad_layers == 0;
            };
            const auto bench = [&](const std::string& name, const std::function<void(int, float*)>& run_layer,
                                   double ref_ms) {
                for (int L = 0; L < 12; ++L) run_layer(L, attn_new);
                CK(cudaStreamSynchronize(st));
                std::vector<double> tot;
                for (int r = 0; r < reps; ++r)
                    tot.push_back(timed([&] { for (int L = 0; L < 12; ++L) run_layer(L, attn_new); }));
                const double ms = median(tot);
                std::printf("  attn %-40s %8.2f ms / 12 layers  %6.3f ms/layer  %.2f TFMA/s", name.c_str(), ms, ms / 12,
                            fma_attn * c.T * 12 / (ms * 1e-3) / 1e12);
                if (ref_ms > 0) std::printf("  x%.3f\n", ref_ms / ms); else std::printf("\n");
                std::fflush(stdout);
                return ms;
            };
            static const bool env_v2 = std::getenv("STRATA_ATTN_MERGE_V2") && std::atoi(std::getenv("STRATA_ATTN_MERGE_V2"));
            t_prod = bench(env_v2 ? "production (decode batch 32, merge v2)" : "production (decode batch 32, merge v1)",
                           prod_layer, 0);
            t_best = t_prod;
            for (const Var& v : vars) {
                if (want_var != "all" && want_var.find(v.key) == std::string::npos) continue;
                parity(v);
                const double ms = bench(v.name, v.run, t_prod);
                t_best = std::min(t_best, ms);
            }
        }
        if (do_sel) {
            const int SEL_BATCH = 256;                     // prefill.cpp: m.sel_batch
            const int64_t max_blocks = 1048576 / 4 + 2;   // --max-context 1048576
            static float *scores = nullptr, *q_idx = nullptr, *pooled = nullptr, *dead = nullptr;
            static int32_t* sel = nullptr;
            if (!scores) {
                CK(cudaMalloc(&scores, (size_t) SEL_BATCH * max_blocks * 4));
                CK(cudaMalloc(&q_idx, (size_t) TMAX * 512 * 4));
                CK(cudaMalloc(&sel, (size_t) TMAX * CAP * 4));
                const int64_t nbk = max_pos / 4 + 2;
                CK(cudaMalloc(&pooled, (size_t) nbk * 128 * 4));
                CK(cudaMalloc(&dead, 128 * 4));
                Rng rng(13);
                std::vector<float> h((size_t) nbk * 128);
                for (auto& x : h) x = 2.0f * rng.uni() - 1.0f;
                CK(cudaMemcpy(pooled, h.data(), h.size() * 4, cudaMemcpyHostToDevice));
                CK(cudaMemcpy(dead, h.data(), 128 * 4, cudaMemcpyHostToDevice));
                h.resize((size_t) TMAX * 512);
                for (auto& x : h) x = 2.0f * rng.uni() - 1.0f;
                CK(cudaMemcpy(q_idx, h.data(), h.size() * 4, cudaMemcpyHostToDevice));
            }
            std::vector<int32_t> steps_h((size_t) c.T * kStepCount);
            for (int t = 0; t < c.T; ++t) qsa_step_fill(steps_h.data() + (size_t) t * kStepCount, c.p0 + t, s);
            const auto active_of = [&](int t0, int nb) {
                return (int64_t) steps_h[(size_t) (t0 + nb - 1) * kStepCount + kStepNBid] + 1;
            };
            const auto sel_layer = [&] {
                for (int t0 = 0; t0 < c.T; t0 += SEL_BATCH) {
                    const int nb = std::min(SEL_BATCH, c.T - t0);
                    const int32_t* st0 = steps_dev + (size_t) t0 * kStepCount;
                    const float* qi = q_idx + (size_t) t0 * 512;
                    if (!qsa_block_scores_tc(pooled, dead, qi, st0, nb, max_blocks, s, scores, st, active_of(t0, nb)))
                        qsa_block_scores(pooled, dead, qi, st0, nb, max_blocks, s, scores, st, active_of(t0, nb));
                    qsa_block_topk(scores, st0, nb, max_blocks, CAP, s, sel + (size_t) t0 * CAP, st, active_of(t0, nb));
                }
            };
            sel_layer();
            CK(cudaStreamSynchronize(st));
            {   // FNV-1a of every query's selected cells (compare across runs with different top-k switches)
                std::vector<int32_t> h((size_t) c.T * CAP);
                CK(cudaMemcpy(h.data(), sel, h.size() * 4, cudaMemcpyDeviceToHost));
                uint64_t f = 1469598103934665603ull;
                for (int32_t x : h) { f ^= (uint32_t) x; f *= 1099511628211ull; }
                std::printf("  sel  ids hash %016llx\n", (unsigned long long) f);
            }
            std::vector<double> tot;
            for (int r = 0; r < reps; ++r) tot.push_back(timed(sel_layer));
            const double ms = median(tot);
            std::printf("  sel  %-40s %8.2f ms / 12 layers  %6.3f ms/layer\n", "production (scores + top-k)", 12 * ms, ms);
            {
                std::vector<double> ts, tk;
                for (int r = 0; r < reps; ++r) {
                    ts.push_back(timed([&] {
                        for (int t0 = 0; t0 < c.T; t0 += SEL_BATCH) {
                            const int nb = std::min(SEL_BATCH, c.T - t0);
                            const int32_t* st0 = steps_dev + (size_t) t0 * kStepCount;
                            const float* qi = q_idx + (size_t) t0 * 512;
                            if (!qsa_block_scores_tc(pooled, dead, qi, st0, nb, max_blocks, s, scores, st, active_of(t0, nb)))
                                qsa_block_scores(pooled, dead, qi, st0, nb, max_blocks, s, scores, st, active_of(t0, nb));
                        }
                    }));
                    tk.push_back(timed([&] {
                        for (int t0 = 0; t0 < c.T; t0 += SEL_BATCH) {
                            const int nb = std::min(SEL_BATCH, c.T - t0);
                            qsa_block_topk(scores, steps_dev + (size_t) t0 * kStepCount, nb, max_blocks, CAP, s,
                                           sel + (size_t) t0 * CAP, st, active_of(t0, nb));
                        }
                    }));
                }
                std::printf("  sel  scores alone %.3f ms/layer, top-k alone %.3f ms/layer (on the last batch's scores)\n",
                            median(ts), median(tk));
            }
            if (do_attn) {
                const double a0 = t_prod / 12, a1 = t_best / 12;
                std::printf("  attn + sel per layer: production %.2f ms, best candidate %.2f ms: %+.1f%%\n", a0 + ms,
                            a1 + ms, 100.0 * ((a1 + ms) / (a0 + ms) - 1.0));
            }
        }
        std::fflush(stdout);
    }
    std::printf("\ndone\n");
    return 0;
}
