// tools/hip/tune_decode.cpp - measure the decode kernels' block shapes on THIS GPU and write a tuning table.
//
// What it tunes (include/strata/kernels/decode_tuning.hpp): rows per thread block of the grouped GPU experts
// (gate/up and down, `native_expert_grouped`) and of `iq_mmvq`.  Every candidate computes every row with the same
// warp, lanes and order, so it is bitwise equal to the default - and this tool checks that on the machine before a
// candidate may win (a candidate that differs by one bit is reported and never written).
//
// The workload is the verify pass's: windows of 1, 3 and 5 tokens (the MTP and lookup windows; weighted 0.2 / 0.3 /
// 0.5), `--hit-frac` of each token's 10 experts on the GPU, experts drawn from a pool larger than the 7900 XTX's
// 96 MB Infinity Cache (decode streams ~1 GB of experts per token, so a cache-warm benchmark would lie), every
// candidate measured in every round, interleaved, and a winner kept only when its weighted median beats the
// default by more than --min-gain AND a second interleaved confirmation agrees.
//
// Run it with the Strata server STOPPED (it needs ~1 GB of free VRAM, and a busy GPU makes timings meaningless):
//
//   build-hip/tune_decode --expert 18:20 --out tools/hip/gfx1100-decode-tuning.txt
//
// tools/hip/autotune.py finds the model's types and shapes, runs this, then confirms the table end to end.
#include "strata/kernels/decode_tuning.hpp"
#include "strata/kernels/iq_kernels.hpp"

#include <hip/hip_runtime.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <memory>
#include <random>
#include <sstream>
#include <string>
#include <vector>

namespace K = strata::kernels;

namespace {
#define HIP_CHECK(call)                                                                                    \
    do {                                                                                                   \
        const hipError_t e_ = (call);                                                                      \
        if (e_ != hipSuccess) {                                                                            \
            std::fprintf(stderr, "tune_decode: %s:%d %s: %s\n", __FILE__, __LINE__, #call, hipGetErrorString(e_)); \
            std::exit(2);                                                                                  \
        }                                                                                                  \
    } while (0)

struct Dev {
    void* p = nullptr;
    size_t n = 0;
    explicit Dev(size_t bytes) : n(bytes) { if (bytes) HIP_CHECK(hipMalloc(&p, bytes)); }
    ~Dev() { if (p) (void) hipFree(p); }
    Dev(const Dev&) = delete;
    Dev& operator=(const Dev&) = delete;
    template<typename T> T* as() const { return static_cast<T*>(p); }
};

struct Options {
    std::vector<std::pair<int, int>> experts;            // (gate/up type, down type)
    std::vector<std::array<long long, 3>> mmvq;          // (type, n_in, n_out)
    int64_t n_embd = 2560, n_ff = 640;
    int k_used = 10;
    double hit_frac = 0.6;
    int pool_mib = 768;
    int rounds = 9, reps = 40;
    double min_gain = 0.03;
    std::string out;
    bool quiet = false;
};

const int kWindows[] = {1, 3, 5};
const double kWeights[] = {0.2, 0.3, 0.5};

int block_elems(int type) {
    switch (type) {
        case 20: return 32;   // IQ4_NL
        case 42: return 64;   // Q2_0
        default: return 256;
    }
}

// random block bytes, then a small finite fp16 scale where the format starts with one (all but IQ1_M), so timings
// run on ordinary numbers rather than NaN/Inf paths
void fill_blocks(std::vector<uint8_t>& host, int type, std::mt19937& rng) {
    std::uniform_int_distribution<int> byte(0, 255);
    for (uint8_t& b : host) b = (uint8_t) byte(rng);
    if (type == 29) return;
    const size_t bb = K::iq_row_bytes(type, block_elems(type));
    if (bb == 0) return;
    const uint16_t half_small = 0x2C00;   // 0.0625
    for (size_t off = 0; off + bb <= host.size(); off += bb) std::memcpy(&host[off], &half_small, 2);
}

double median(std::vector<double> v) {
    if (v.empty()) return 0.0;
    std::sort(v.begin(), v.end());
    const size_t m = v.size() / 2;
    return v.size() % 2 ? v[m] : 0.5 * (v[m - 1] + v[m]);
}

double free_mib() {
    size_t fr = 0, tot = 0;
    HIP_CHECK(hipMemGetInfo(&fr, &tot));
    return (double) fr / (1024.0 * 1024.0);
}

// ---------------------------------------------------------------- grouped experts
struct Plan {           // one verify window's GPU groups (device arrays)
    Dev ptr, start, count, dst, tok;
    int groups = 0, entries = 0;
    Plan(int cap) : ptr(sizeof(unsigned long long) * cap), start(sizeof(int32_t) * (cap + 1)),
                    count(sizeof(int32_t)), dst(sizeof(int32_t) * cap), tok(sizeof(int32_t) * cap) {}
};

struct ExpertBench {
    K::NativeExpertLayout L;
    int64_t pool = 0;
    Dev blobs;
    std::vector<std::unique_ptr<Plan>> plans[3];    // per window, several distinct expert sets
    int cap[3] = {0, 0, 0};
    std::unique_ptr<Dev> xq[3], scratch[3], out[3];

    ExpertBench(const K::NativeExpertLayout& lay, int pool_mib)
        : L(lay), pool(std::max<int64_t>(64, ((int64_t) pool_mib << 20) / (int64_t) lay.bytes)),
          blobs((size_t) pool * lay.bytes) {}
};

void make_expert_bench(ExpertBench& b, const Options& o, std::mt19937& rng) {
    // the blobs: gate | up | down, each part filled as its own format
    {
        std::vector<uint8_t> gu((size_t) b.L.down_off), dn(b.L.bytes - b.L.down_off);
        for (int64_t i = 0; i < b.pool; ++i) {
            fill_blocks(gu, b.L.gu_type, rng);
            fill_blocks(dn, b.L.d_type, rng);
            uint8_t* base = b.blobs.as<uint8_t>() + (size_t) i * b.L.bytes;
            HIP_CHECK(hipMemcpy(base, gu.data(), gu.size(), hipMemcpyHostToDevice));
            HIP_CHECK(hipMemcpy(base + b.L.down_off, dn.data(), dn.size(), hipMemcpyHostToDevice));
        }
    }
    const int kPlans = 16;
    for (int w = 0; w < 3; ++w) {
        const int T = kWindows[w];
        const int cap = T * o.k_used;
        b.cap[w] = cap;
        // activations: T tokens of n_embd, quantized by the engine's own q8_1 kernel
        std::normal_distribution<float> nd(0.0f, 1.0f);
        std::vector<float> xh((size_t) T * (size_t) b.L.n_embd);
        for (float& v : xh) v = nd(rng);
        Dev xf(xh.size() * sizeof(float));
        HIP_CHECK(hipMemcpy(xf.p, xh.data(), xf.n, hipMemcpyHostToDevice));
        b.xq[w] = std::make_unique<Dev>((size_t) T * (size_t) (b.L.n_embd / 32) * 36);
        K::quantize_q8_1_rows(xf.as<float>(), T, b.L.n_embd, b.xq[w]->p, nullptr);
        HIP_CHECK(hipDeviceSynchronize());
        b.scratch[w] = std::make_unique<Dev>(K::native_expert_scratch_bytes(cap, b.L.n_ff));
        b.out[w] = std::make_unique<Dev>((size_t) cap * (size_t) b.L.n_embd * sizeof(float));
        // entries: hit_frac of the window's T*K picks; distinct experts, a few shared by two tokens
        const int entries = std::max(1, (int) std::lround(o.hit_frac * cap));
        const int groups = std::max(1, std::min(entries, (int) std::lround(entries * (T == 1 ? 1.0 : 0.8))));
        for (int pl = 0; pl < kPlans; ++pl) {
            auto p = std::make_unique<Plan>(cap);
            p->groups = groups;
            p->entries = entries;
            std::vector<int64_t> pick(b.pool);
            for (int64_t i = 0; i < b.pool; ++i) pick[(size_t) i] = i;
            std::shuffle(pick.begin(), pick.end(), rng);
            std::vector<unsigned long long> ptr(cap, 0);
            std::vector<int32_t> start(cap + 1, entries), dst(cap, 0), tok(cap, 0);
            for (int g = 0; g < groups; ++g)
                ptr[g] = (unsigned long long) (uintptr_t) (b.blobs.as<uint8_t>() + (size_t) pick[(size_t) g] * b.L.bytes);
            // spread the entries over the groups (the first entries - groups get a second entry)
            int e = 0;
            for (int g = 0; g < groups; ++g) {
                start[g] = e;
                const int n = 1 + (g < entries - groups ? 1 : 0);
                for (int j = 0; j < n; ++j, ++e) {
                    dst[e] = e;
                    tok[e] = (g + j) % T;
                }
            }
            start[groups] = e;
            const int32_t cnt = groups;
            HIP_CHECK(hipMemcpy(p->ptr.p, ptr.data(), ptr.size() * sizeof(ptr[0]), hipMemcpyHostToDevice));
            HIP_CHECK(hipMemcpy(p->start.p, start.data(), start.size() * sizeof(int32_t), hipMemcpyHostToDevice));
            HIP_CHECK(hipMemcpy(p->count.p, &cnt, sizeof(cnt), hipMemcpyHostToDevice));
            HIP_CHECK(hipMemcpy(p->dst.p, dst.data(), dst.size() * sizeof(int32_t), hipMemcpyHostToDevice));
            HIP_CHECK(hipMemcpy(p->tok.p, tok.data(), tok.size() * sizeof(int32_t), hipMemcpyHostToDevice));
            b.plans[w].push_back(std::move(p));
        }
    }
}

void run_grouped(ExpertBench& b, int w, int plan, int gu, int dn, hipStream_t s) {
    const Plan& p = *b.plans[w][(size_t) plan];
    K::native_expert_grouped_rows(b.L, p.ptr.as<unsigned long long>(), p.start.as<int32_t>(), p.count.as<int32_t>(),
                                  p.dst.as<int32_t>(), p.tok.as<int32_t>(), b.cap[w], b.cap[w], b.xq[w]->p,
                                  b.scratch[w]->p, b.out[w]->as<float>(), gu, dn, s);
}

// The engine's own default: the real entry point, no table active in this process.  The default candidate and the
// bitwise reference are THIS call (its heuristics may pick the multi sub-warp kernels), not an explicit-rows pair -
// so the tuner's numbers describe the launch the engine actually makes without a table.
void run_default(ExpertBench& b, int w, int plan, hipStream_t s) {
    const Plan& p = *b.plans[w][(size_t) plan];
    K::native_expert_grouped(b.L, p.ptr.as<unsigned long long>(), p.start.as<int32_t>(), p.count.as<int32_t>(),
                             p.dst.as<int32_t>(), p.tok.as<int32_t>(), b.cap[w], b.cap[w], b.xq[w]->p,
                             b.scratch[w]->p, b.out[w]->as<float>(), s, 0);
}

std::vector<uint8_t> grouped_output(ExpertBench& b, int w, int gu, int dn, hipStream_t s) {
    HIP_CHECK(hipMemsetAsync(b.out[w]->p, 0, b.out[w]->n, s));
    run_grouped(b, w, 0, gu, dn, s);
    HIP_CHECK(hipStreamSynchronize(s));
    std::vector<uint8_t> h(b.out[w]->n);
    HIP_CHECK(hipMemcpy(h.data(), b.out[w]->p, h.size(), hipMemcpyDeviceToHost));
    return h;
}

std::vector<uint8_t> default_output(ExpertBench& b, int w, hipStream_t s) {
    HIP_CHECK(hipMemsetAsync(b.out[w]->p, 0, b.out[w]->n, s));
    run_default(b, w, 0, s);
    HIP_CHECK(hipStreamSynchronize(s));
    std::vector<uint8_t> h(b.out[w]->n);
    HIP_CHECK(hipMemcpy(h.data(), b.out[w]->p, h.size(), hipMemcpyDeviceToHost));
    return h;
}

// microseconds per call: `reps` calls cycling through the plans, timed by events
double time_grouped(ExpertBench& b, int w, int gu, int dn, int reps, hipStream_t s, hipEvent_t e0, hipEvent_t e1) {
    const int np = (int) b.plans[w].size();
    for (int i = 0; i < 2; ++i) run_grouped(b, w, i % np, gu, dn, s);
    HIP_CHECK(hipEventRecord(e0, s));
    for (int i = 0; i < reps; ++i) run_grouped(b, w, i % np, gu, dn, s);
    HIP_CHECK(hipEventRecord(e1, s));
    HIP_CHECK(hipEventSynchronize(e1));
    float ms = 0.0f;
    HIP_CHECK(hipEventElapsedTime(&ms, e0, e1));
    return 1000.0 * ms / reps;
}

double time_default(ExpertBench& b, int w, int reps, hipStream_t s, hipEvent_t e0, hipEvent_t e1) {
    const int np = (int) b.plans[w].size();
    for (int i = 0; i < 2; ++i) run_default(b, w, i % np, s);
    HIP_CHECK(hipEventRecord(e0, s));
    for (int i = 0; i < reps; ++i) run_default(b, w, i % np, s);
    HIP_CHECK(hipEventRecord(e1, s));
    HIP_CHECK(hipEventSynchronize(e1));
    float ms = 0.0f;
    HIP_CHECK(hipEventElapsedTime(&ms, e0, e1));
    return 1000.0 * ms / reps;
}

// ---------------------------------------------------------------- iq_mmvq
struct MmvqBench {
    int type;
    int n_in, n_out;
    int64_t copies;
    size_t wbytes;
    Dev w;
    std::unique_ptr<Dev> xq[3], y[3];
    MmvqBench(int t, int ni, int no, int pool_mib)
        : type(t), n_in(ni), n_out(no), copies(0), wbytes(K::iq_row_bytes(t, ni) * (size_t) no),
          w(std::max<size_t>(1, wbytes) * (size_t) std::max<int64_t>(2, ((int64_t) pool_mib << 20) / (int64_t) std::max<size_t>(1, wbytes))) {
        copies = (int64_t) (w.n / wbytes);
    }
};

void make_mmvq_bench(MmvqBench& b, std::mt19937& rng) {
    std::vector<uint8_t> h(b.wbytes);
    for (int64_t c = 0; c < b.copies; ++c) {
        fill_blocks(h, b.type, rng);
        HIP_CHECK(hipMemcpy(b.w.as<uint8_t>() + (size_t) c * b.wbytes, h.data(), h.size(), hipMemcpyHostToDevice));
    }
    std::normal_distribution<float> nd(0.0f, 1.0f);
    for (int wi = 0; wi < 3; ++wi) {
        const int T = kWindows[wi];
        std::vector<float> xh((size_t) T * (size_t) b.n_in);
        for (float& v : xh) v = nd(rng);
        Dev xf(xh.size() * sizeof(float));
        HIP_CHECK(hipMemcpy(xf.p, xh.data(), xf.n, hipMemcpyHostToDevice));
        b.xq[wi] = std::make_unique<Dev>((size_t) T * (size_t) (b.n_in / 32) * 36);
        K::quantize_q8_1_rows(xf.as<float>(), T, b.n_in, b.xq[wi]->p, nullptr);
        b.y[wi] = std::make_unique<Dev>((size_t) T * (size_t) b.n_out * sizeof(float));
    }
    HIP_CHECK(hipDeviceSynchronize());
}

void run_mmvq(MmvqBench& b, int wi, int copy, int rows, hipStream_t s) {
    K::iq_mmvq_rows(b.type, b.w.as<uint8_t>() + (size_t) copy * b.wbytes, b.xq[wi]->p, b.y[wi]->as<float>(), b.n_in,
                    b.n_out, kWindows[wi], rows, s);
}

std::vector<uint8_t> mmvq_output(MmvqBench& b, int wi, int rows, hipStream_t s) {
    HIP_CHECK(hipMemsetAsync(b.y[wi]->p, 0, b.y[wi]->n, s));
    run_mmvq(b, wi, 0, rows, s);
    HIP_CHECK(hipStreamSynchronize(s));
    std::vector<uint8_t> h(b.y[wi]->n);
    HIP_CHECK(hipMemcpy(h.data(), b.y[wi]->p, h.size(), hipMemcpyDeviceToHost));
    return h;
}

double time_mmvq(MmvqBench& b, int wi, int rows, int reps, hipStream_t s, hipEvent_t e0, hipEvent_t e1) {
    for (int i = 0; i < 2; ++i) run_mmvq(b, wi, (int) (i % b.copies), rows, s);
    HIP_CHECK(hipEventRecord(e0, s));
    for (int i = 0; i < reps; ++i) run_mmvq(b, wi, (int) (i % b.copies), rows, s);
    HIP_CHECK(hipEventRecord(e1, s));
    HIP_CHECK(hipEventSynchronize(e1));
    float ms = 0.0f;
    HIP_CHECK(hipEventElapsedTime(&ms, e0, e1));
    return 1000.0 * ms / reps;
}

// ---------------------------------------------------------------- the search
// Weighted (by window) median microseconds of each candidate, measured interleaved over `rounds` rounds.
template<typename Cand, typename TimeFn>
std::map<Cand, double> measure(const std::vector<Cand>& cands, int rounds, TimeFn&& t) {
    std::map<Cand, std::vector<double>> per[3];
    for (int r = 0; r < rounds; ++r)
        for (int w = 0; w < 3; ++w) {
            // rotate the order each round so no candidate always runs first (clock ramp, cache state)
            for (size_t i = 0; i < cands.size(); ++i) {
                const Cand& c = cands[(i + (size_t) r) % cands.size()];
                per[w][c].push_back(t(c, w));
            }
        }
    std::map<Cand, double> score;
    for (const Cand& c : cands) {
        double s = 0.0;
        for (int w = 0; w < 3; ++w) s += kWeights[w] * median(per[w][c]);
        score[c] = s;
    }
    return score;
}

struct Choice {
    bool changed = false;
    double default_us = 0.0, best_us = 0.0, confirm_default_us = 0.0, confirm_best_us = 0.0;
};

template<typename Cand, typename TimeFn>
Cand choose(const std::vector<Cand>& cands, const Cand& dflt, const Options& o, TimeFn&& t, Choice& ch) {
    const std::map<Cand, double> s = measure(cands, o.rounds, t);
    Cand best = dflt;
    for (const auto& kv : s)
        if (kv.second < s.at(best)) best = kv.first;
    ch.default_us = s.at(dflt);
    ch.best_us = s.at(best);
    if (best == dflt || ch.best_us > ch.default_us * (1.0 - o.min_gain)) return dflt;
    // confirm: the winner against the default alone, interleaved, as many rounds again
    const std::map<Cand, double> c = measure(std::vector<Cand>{dflt, best}, o.rounds, t);
    ch.confirm_default_us = c.at(dflt);
    ch.confirm_best_us = c.at(best);
    if (ch.confirm_best_us > ch.confirm_default_us * (1.0 - o.min_gain)) return dflt;
    ch.changed = true;
    return best;
}

bool parse_pair(const char* s, int& a, int& b) { return std::sscanf(s, "%d:%d", &a, &b) == 2; }

void usage() {
    std::fprintf(stderr,
                 "usage: tune_decode [--expert GU:DOWN]... [--mmvq TYPE:N_IN:N_OUT]... [--n-embd N] [--n-ff N]\n"
                 "                   [--k K] [--hit-frac F] [--pool-mib M] [--rounds R] [--reps N] [--min-gain F]\n"
                 "                   [--out FILE] [--quiet]\n"
                 "  --expert   ggml types of the experts' gate/up and down (default 18:20 = IQ3_XXS:IQ4_NL); repeatable\n"
                 "  --mmvq     an i-quant dense matrix iq_mmvq runs in decode; repeatable (none by default)\n"
                 "  --hit-frac share of a token's K experts served from VRAM (default 0.6)\n"
                 "  --pool-mib distinct weights the timings cycle through (default 768; keep it well above 96 MB)\n"
                 "  --out      write the table (only shapes whose winner differs from the default are listed)\n");
}
}  // namespace

int main(int argc, char** argv) {
    Options o;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&]() -> const char* {
            if (i + 1 >= argc) { usage(); std::exit(2); }
            return argv[++i];
        };
        if (a == "--expert") {
            int gu = 0, dn = 0;
            if (!parse_pair(next(), gu, dn)) { usage(); return 2; }
            o.experts.push_back({gu, dn});
        } else if (a == "--mmvq") {
            long long t = 0, ni = 0, no = 0;
            if (std::sscanf(next(), "%lld:%lld:%lld", &t, &ni, &no) != 3) { usage(); return 2; }
            o.mmvq.push_back({t, ni, no});
        } else if (a == "--n-embd") o.n_embd = std::atoll(next());
        else if (a == "--n-ff") o.n_ff = std::atoll(next());
        else if (a == "--k") o.k_used = std::atoi(next());
        else if (a == "--hit-frac") o.hit_frac = std::atof(next());
        else if (a == "--pool-mib") o.pool_mib = std::atoi(next());
        else if (a == "--rounds") o.rounds = std::max(3, std::atoi(next()));
        else if (a == "--reps") o.reps = std::max(5, std::atoi(next()));
        else if (a == "--min-gain") o.min_gain = std::atof(next());
        else if (a == "--out") o.out = next();
        else if (a == "--quiet") o.quiet = true;
        else { usage(); return a == "--help" || a == "-h" ? 0 : 2; }
    }
    if (o.experts.empty()) o.experts.push_back({18, 20});
    if (o.hit_frac <= 0.0 || o.hit_frac > 1.0 || o.k_used < 1 || o.n_embd % 256 || o.n_ff % 32) {
        std::fprintf(stderr, "tune_decode: bad shape or --hit-frac\n");
        return 2;
    }
    const K::DecodeIdentity id = K::decode_identity();
    std::fprintf(stderr, "tune_decode: %s, HIP runtime %lld, toolchain %s\n", id.arch.c_str(), id.runtime,
                 id.toolchain.c_str());
    if (id.arch != "gfx1100")
        std::fprintf(stderr, "tune_decode: note - Strata's HIP backend is validated on gfx1100 only; measuring anyway\n");
    const double need = o.pool_mib * (1.0 + (double) o.mmvq.size()) + 256.0;
    if (free_mib() < need) {
        std::fprintf(stderr, "tune_decode: %.0f MiB of VRAM free, %.0f needed - stop the Strata server (it holds the expert "
                             "cache) or lower --pool-mib\n", free_mib(), need);
        return 3;
    }
    hipStream_t s;
    HIP_CHECK(hipStreamCreateWithFlags(&s, hipStreamNonBlocking));
    hipEvent_t e0, e1;
    HIP_CHECK(hipEventCreate(&e0));
    HIP_CHECK(hipEventCreate(&e1));
    std::mt19937 rng(20261004u);
    std::vector<K::DecodeTuningRow> rows;
    std::ostringstream report;
    int refused = 0;

    for (const auto& [gu_t, dn_t] : o.experts) {
        if (!K::iq_supported(gu_t) || (dn_t != 20 && dn_t != 23 && dn_t != 42)) {
            std::fprintf(stderr, "tune_decode: expert types %d:%d are not native expert formats; skipped\n", gu_t, dn_t);
            continue;
        }
        ExpertBench b(K::native_expert_layout(gu_t, dn_t, o.n_embd, o.n_ff), o.pool_mib);
        make_expert_bench(b, o, rng);
        // every candidate pair, checked bitwise against the engine's true default before it may be timed
        const std::vector<int> gc = K::decode_rows_candidates(K::DecodeKernel::GateUp);
        const std::vector<int> dc = K::decode_rows_candidates(K::DecodeKernel::Down);
        const int d0 = K::kDefaultExpertRows;
        std::vector<std::pair<int, int>> cands;
        std::vector<std::vector<uint8_t>> ref(3);
        for (int w = 0; w < 3; ++w) ref[w] = default_output(b, w, s);
        for (int g : gc)
            for (int d : dc) {
                if (g == d0 && d == d0) { cands.push_back({g, d}); continue; }   // the default pair IS the default
                // path (timed as such below); comparing it bitwise to itself decides nothing
                bool same = true;
                for (int w = 0; w < 3 && same; ++w) same = grouped_output(b, w, g, d, s) == ref[w];
                if (same) cands.push_back({g, d});
                else {
                    ++refused;
                    std::fprintf(stderr, "tune_decode: experts %d:%d gu=%d down=%d is NOT bitwise equal to the default - "
                                         "excluded (please report this build)\n", gu_t, dn_t, g, d);
                }
            }
        Choice ch;
        const std::pair<int, int> best = choose(cands, {d0, d0}, o,
                                                [&](const std::pair<int, int>& c, int w) {
                                                    if (c.first == d0 && c.second == d0)
                                                        return time_default(b, w, o.reps, s, e0, e1);
                                                    return time_grouped(b, w, c.first, c.second, o.reps, s, e0, e1);
                                                }, ch);
        char line[256];
        std::snprintf(line, sizeof(line),
                      "experts %d:%d (n_embd %lld, n_ff %lld): default chain %.1f us, best %d/%d %.1f us%s\n", gu_t,
                      dn_t, (long long) o.n_embd, (long long) o.n_ff, ch.default_us, best.first, best.second,
                      ch.changed ? ch.confirm_best_us : ch.best_us,
                      ch.changed ? " (confirmed)" : " (kept the default)");
        report << line;
        if (!o.quiet) std::fputs(line, stderr);
        if (best.first != d0) rows.push_back({K::DecodeKernel::GateUp, gu_t, o.n_embd, o.n_ff, best.first});
        if (best.second != d0) rows.push_back({K::DecodeKernel::Down, dn_t, o.n_ff, o.n_embd, best.second});
        std::printf("RESULT experts %d %d %lld %lld default_us=%.2f chosen=%d/%d chosen_us=%.2f changed=%d\n", gu_t, dn_t,
                    (long long) o.n_embd, (long long) o.n_ff, ch.default_us, best.first, best.second,
                    ch.changed ? ch.confirm_best_us : ch.default_us, ch.changed ? 1 : 0);
    }

    for (const auto& m : o.mmvq) {
        const int t = (int) m[0], ni = (int) m[1], no = (int) m[2];
        if (!K::iq_supported(t) || ni % 256 || no <= 0) {
            std::fprintf(stderr, "tune_decode: mmvq %d:%d:%d is not an i-quant shape; skipped\n", t, ni, no);
            continue;
        }
        MmvqBench b(t, ni, no, o.pool_mib);
        make_mmvq_bench(b, rng);
        const int d0 = K::kDefaultMmvqRows;
        std::vector<int> cands;
        std::vector<std::vector<uint8_t>> ref(3);
        for (int w = 0; w < 3; ++w) ref[w] = mmvq_output(b, w, d0, s);
        for (int r : K::decode_rows_candidates(K::DecodeKernel::Mmvq)) {
            bool same = true;
            for (int w = 0; w < 3 && same; ++w) same = mmvq_output(b, w, r, s) == ref[w];
            if (same) cands.push_back(r);
            else {
                ++refused;
                std::fprintf(stderr, "tune_decode: mmvq %d:%d:%d rows=%d is NOT bitwise equal to the default - excluded\n",
                             t, ni, no, r);
            }
        }
        Choice ch;
        const int best = choose(cands, d0, o, [&](int r, int w) { return time_mmvq(b, w, r, o.reps, s, e0, e1); }, ch);
        char line[256];
        std::snprintf(line, sizeof(line), "mmvq %d %dx%d: default %d %.1f us, best %d %.1f us%s\n", t, ni, no, d0,
                      ch.default_us, best, ch.changed ? ch.confirm_best_us : ch.best_us,
                      ch.changed ? " (confirmed)" : " (kept the default)");
        report << line;
        if (!o.quiet) std::fputs(line, stderr);
        if (best != d0) rows.push_back({K::DecodeKernel::Mmvq, t, ni, no, best});
        std::printf("RESULT mmvq %d %d %d default_us=%.2f chosen=%d chosen_us=%.2f changed=%d\n", t, ni, no, ch.default_us,
                    best, ch.changed ? ch.confirm_best_us : ch.default_us, ch.changed ? 1 : 0);
    }

    std::ostringstream table;
    table << "# made by tools/hip/tune_decode on this GPU; only shapes whose measured winner beat the default by more\n"
          << "# than " << o.min_gain * 100.0 << "% (twice, interleaved) are listed - every listed variant was checked "
          << "bitwise equal to the default\n";
    {
        std::istringstream rep(report.str());
        std::string l;
        while (std::getline(rep, l)) table << "# " << l << "\n";
    }
    table << K::DecodeTuningTable::header(id) << "\n";
    for (const K::DecodeTuningRow& r : rows) table << K::DecodeTuningTable::line(r) << "\n";
    if (!o.out.empty()) {
        std::ofstream f(o.out);
        if (!(f << table.str())) {
            std::fprintf(stderr, "tune_decode: cannot write %s\n", o.out.c_str());
            return 1;
        }
        std::fprintf(stderr, "tune_decode: wrote %s (%zu shapes changed)\n", o.out.c_str(), rows.size());
    } else {
        std::fputs(table.str().c_str(), stdout);
    }
    std::printf("SUMMARY changed=%zu refused=%d\n", rows.size(), refused);
    (void) hipEventDestroy(e0);
    (void) hipEventDestroy(e1);
    (void) hipStreamDestroy(s);
    return refused ? 4 : 0;
}
