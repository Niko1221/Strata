// src/kernels/fuse_gu_bench.cpp - STRATA_FUSE_GU (p58/fuse-gu): the grouped decode experts' default gate/up path
// (native_gu_kernel<14> + swiglu_q8_1_entries_kernel) against the fused Q6_K gate/up + SwiGLU + q8_1 kernel
// (fuse_gu.cuh), on a real UD-Q6_K_XL layer's 512 expert blobs, bit for bit and in microseconds.
//
//     build/fuse_gu_bench <shard.gguf> [--layers 10,30] [--t 1,2,3,4,5,6] [--dist uni,s50,cal44]
//                         [--choices def,f1,f2,f3,f4,f5] [--plans 128] [--reps 120] [--rounds 5] [--dev 0]
//                         [--graph 1] [--tsv out.tsv] [--tag text]
//
// Choices: def = STRATA_FUSE_GU off (the reference, always first); fN = STRATA_FUSE_GU=N (fuse_gu.hpp's variants).
// Routing distributions (`--plans` windows per (T, dist), each T rows x top-10 distinct experts of 512):
//   uni    every row's 10 experts uniform (U(6) ~ 57 groups)
//   s50    row 0 uniform, every later row takes 5 of its 10 from the window's experts so far (stream-like reuse)
//   calN   Zipf popularity over a random expert order, the exponent bisected so the mean number of distinct experts at
//          T=6 is N (cal44: the decode8 traces' U(6) = 44, ~1.4 rows per expert)
// Activations: T rows of N(0,1) with two outlier channels (x20), quantized by quantize_q8_1_rows as the window does.
//
// Per (layer, T, dist, choice): every plan's call (gate/up + SwiGLU/q8_1 + down) is run once and its `out` rows and
// q8_1 h blocks are compared byte for byte with def's (plans_identical, max_abs_out, hq_bytes_diff); then timings,
// medians over `--rounds` of `--reps` back-to-back calls cycling over the plans (cold weights: the plans' experts are
// spread over 2.3 GB): eager_all / eager_gu (stream launches; gu = gate/up + SwiGLU + q8_1, native_expert_set_mode
// phase 1) and, with --graph 1, graph_all / graph_gu (the same calls captured in one CUDA graph, as the decode
// windows run them).  GB/s = expert bytes the call reads / time (gate/up bytes only for the gu columns).
// One TSV row per cell (--tsv, appended; header written when the file is new).  Exit 1 if any cell differs.
#include "strata/artifact/gguf_reader.hpp"
#include "strata/kernels/cpu/native_expert.hpp"
#include "strata/kernels/fuse_gu.hpp"
#include "strata/kernels/iq_kernels.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <random>
#include <sstream>
#include <string>
#include <vector>

namespace cpu = strata::kernels::cpu;
namespace K = strata::kernels;

namespace {

constexpr int kNE = 512, kTop = 10, kH = 2560, kFF = 640, kMaxT = 8;

#define FB_CK(x)                                                                                             \
    do {                                                                                                     \
        cudaError_t e_ = (x);                                                                                \
        if (e_ != cudaSuccess) {                                                                             \
            std::fprintf(stderr, "CUDA %s at %s:%d: %s\n", #x, __FILE__, __LINE__, cudaGetErrorString(e_)); \
            std::exit(2);                                                                                    \
        }                                                                                                    \
    } while (0)

std::vector<std::string> split_list(const std::string& s) {
    std::vector<std::string> out;
    std::stringstream ss(s);
    std::string t;
    while (std::getline(ss, t, ','))
        if (!t.empty()) out.push_back(t);
    return out;
}

struct Plan {
    int G = 0, E = 0;
    std::vector<int> expert;                // per group
    std::vector<int32_t> start, tok, dst;   // start: G + 1; tok / dst: E
};

// rows[t] = the 10 experts of row t (top-k slot order)
Plan make_plan(const std::vector<std::vector<int>>& rows) {
    Plan p;
    std::vector<int> gid(kNE, -1);
    for (const auto& r : rows)
        for (int e : r)
            if (gid[(size_t) e] < 0) { gid[(size_t) e] = p.G++; p.expert.push_back(e); }
    std::vector<std::vector<std::pair<int, int>>> ent((size_t) p.G);   // (token, slot)
    for (int t = 0; t < (int) rows.size(); ++t)
        for (int k = 0; k < kTop; ++k) ent[(size_t) gid[(size_t) rows[(size_t) t][(size_t) k]]].push_back({t, k});
    p.start.push_back(0);
    for (int g = 0; g < p.G; ++g) {
        for (auto [t, k] : ent[(size_t) g]) { p.tok.push_back(t); p.dst.push_back(t * kTop + k); }
        p.start.push_back((int32_t) p.tok.size());
    }
    p.E = (int) p.tok.size();
    return p;
}

// k distinct picks from weights w (without replacement), skipping `taken`
void pick_weighted(const std::vector<double>& w, int k, std::vector<char>& taken, std::vector<int>& out, std::mt19937_64& rng) {
    std::uniform_real_distribution<double> u(0.0, 1.0);
    double tot = 0.0;
    for (int i = 0; i < kNE; ++i)
        if (!taken[(size_t) i]) tot += w[(size_t) i];
    for (int n = 0; n < k; ++n) {
        double r = u(rng) * tot;
        int pick = -1;
        for (int i = 0; i < kNE; ++i) {
            if (taken[(size_t) i]) continue;
            pick = i;
            r -= w[(size_t) i];
            if (r <= 0.0) break;
        }
        taken[(size_t) pick] = 1;
        tot -= w[(size_t) pick];
        out.push_back(pick);
    }
}

std::vector<std::vector<int>> rows_zipf(int T, const std::vector<double>& w, std::mt19937_64& rng) {
    std::vector<std::vector<int>> rows((size_t) T);
    for (int t = 0; t < T; ++t) {
        std::vector<char> taken(kNE, 0);
        pick_weighted(w, kTop, taken, rows[(size_t) t], rng);
    }
    return rows;
}

std::vector<double> zipf_weights(double s, const std::vector<int>& order) {
    std::vector<double> w(kNE);
    for (int i = 0; i < kNE; ++i) w[(size_t) order[(size_t) i]] = 1.0 / std::pow((double) (i + 1), s);
    return w;
}

double mean_unique(int T, const std::vector<double>& w, int n, std::mt19937_64& rng) {
    double sum = 0.0;
    for (int i = 0; i < n; ++i) sum += make_plan(rows_zipf(T, w, rng)).G;
    return sum / n;
}

std::vector<std::vector<int>> rows_for(const std::string& dist, int T, const std::vector<double>& zw, std::mt19937_64& rng) {
    if (dist.rfind("cal", 0) == 0) return rows_zipf(T, zw, rng);
    std::vector<double> flat(kNE, 1.0);
    std::vector<std::vector<int>> rows((size_t) T);
    std::vector<int> seen;
    for (int t = 0; t < T; ++t) {
        std::vector<char> taken(kNE, 0);
        auto& r = rows[(size_t) t];
        if (dist == "s50" && t > 0) {   // 5 from the window's experts so far, the rest uniform
            std::vector<int> pool = seen;
            std::shuffle(pool.begin(), pool.end(), rng);
            for (int e : pool) {
                if ((int) r.size() == kTop / 2) break;
                if (!taken[(size_t) e]) { taken[(size_t) e] = 1; r.push_back(e); }
            }
        }
        pick_weighted(flat, kTop - (int) r.size(), taken, r, rng);
        std::shuffle(r.begin(), r.end(), rng);
        for (int e : r)
            if (std::find(seen.begin(), seen.end(), e) == seen.end()) seen.push_back(e);
    }
    return rows;
}

struct Choice { std::string name; int fuse = 0; };

}  // namespace

int main(int argc, char** argv) {
    setvbuf(stdout, nullptr, _IONBF, 0);
    if (argc < 2) {
        std::fprintf(stderr, "usage: fuse_gu_bench <shard.gguf> [--layers 10,30] [--t 1,...,6] [--dist uni,s50,cal44] "
                             "[--choices def,f1,...] [--plans 128] [--reps 120] [--rounds 5] [--dev 0] [--graph 1] [--tsv f] [--tag s]\n");
        return 2;
    }
    const std::string shard = argv[1];
    std::vector<int> layers = {10}, Ts = {1, 2, 3, 4, 5, 6};
    std::vector<std::string> dists = {"uni", "s50", "cal44"}, choice_s = {"def", "f1", "f2", "f3", "f4", "f5"};
    int plans_n = 128, reps = 120, rounds = 5, dev = 0, graph = 1;
    std::string tsv, tag = "-";
    for (int i = 2; i + 1 < argc; i += 2) {
        const std::string a = argv[i], v = argv[i + 1];
        if (a == "--layers") { layers.clear(); for (auto& s : split_list(v)) layers.push_back(std::atoi(s.c_str())); }
        else if (a == "--t") { Ts.clear(); for (auto& s : split_list(v)) Ts.push_back(std::atoi(s.c_str())); }
        else if (a == "--dist") dists = split_list(v);
        else if (a == "--choices") choice_s = split_list(v);
        else if (a == "--plans") plans_n = std::max(4, std::atoi(v.c_str()));
        else if (a == "--reps") reps = std::max(10, std::atoi(v.c_str()));
        else if (a == "--rounds") rounds = std::max(1, std::atoi(v.c_str()));
        else if (a == "--dev") dev = std::atoi(v.c_str());
        else if (a == "--graph") graph = std::atoi(v.c_str());
        else if (a == "--tsv") tsv = v;
        else if (a == "--tag") tag = v;
        else { std::fprintf(stderr, "unknown option %s\n", a.c_str()); return 2; }
    }
    for (int T : Ts)
        if (T < 1 || T > kMaxT) { std::fprintf(stderr, "T %d out of 1..%d\n", T, kMaxT); return 2; }
    std::vector<Choice> choices = {{"def", 0}};
    for (const auto& c : choice_s) {
        if (c == "def") continue;
        if (c.size() == 2 && c[0] == 'f' && c[1] >= '1' && c[1] <= '5') choices.push_back({c, c[1] - '0'});
        else { std::fprintf(stderr, "choice %s is not def or f1..f5\n", c.c_str()); return 2; }
    }
    if (layers.empty() || Ts.empty() || dists.empty()) {
        std::fprintf(stderr, "fuse_gu_bench: no fused cells requested; layers, T and distributions must be nonempty\n");
        return 2;
    }
    if (choices.size() == 1) {
        std::fprintf(stderr, "fuse_gu_bench: no fused choices requested; validation needs f1..f5\n");
        return 2;
    }
    // These knobs bypass fuse_gu_launch in native_expert_grouped. A fallback-to-fallback comparison is not parity.
    for (const char* name : {"STRATA_GROUPED_V1", "STRATA_OLD_IQ_MMVQ"}) {
        const char* v = std::getenv(name);
        if (v != nullptr && v[0] != '\0' && v[0] != '0') {
            std::fprintf(stderr, "fuse_gu_bench: unset %s; it bypasses the fused kernel\n", name);
            return 2;
        }
    }
    const char* exp_mode = std::getenv("STRATA_EXP_MODE");
    if (exp_mode != nullptr && std::atoi(exp_mode) > 0) {
        std::fprintf(stderr, "fuse_gu_bench: STRATA_EXP_MODE must be unset or 0; another layout bypasses fusion\n");
        return 2;
    }
    // Architecture defaults can also select a nonzero layout. Pin the CUDA layout for this benchmark only.
    K::native_expert_set_mode(0, 0);
    FB_CK(cudaSetDevice(dev));
    cudaDeviceProp prop{};
    FB_CK(cudaGetDeviceProperties(&prop, dev));
    int mclk = 0, bus = 0;
    cudaDeviceGetAttribute(&mclk, cudaDevAttrMemoryClockRate, dev);
    cudaDeviceGetAttribute(&bus, cudaDevAttrGlobalMemoryBusWidth, dev);
    const double peak = 2.0 * (double) mclk * 1e3 * (double) bus / 8.0;
    char busid[64] = {};
    cudaDeviceGetPCIBusId(busid, sizeof busid, dev);
    std::printf("# device %d: %s (bus %s, sm_%d%d, %d SMs, L2 %d KB), peak DRAM %.1f GB/s\n", dev, prop.name, busid, prop.major,
                prop.minor, prop.multiProcessorCount, prop.l2CacheSize / 1024, peak / 1e9);
    for (const char* name : {"STRATA_FUSE_GU", "STRATA_OLD_IQ_MMVQ", "STRATA_GROUPED_V1", "STRATA_EXP_MODE", "STRATA_Q6K_SPLIT", "STRATA_EXPDEC", "STRATA_Q8_DOWN"}) {
        const char* v = std::getenv(name);
        if (v && v[0]) std::printf("# note: %s=%s in the environment\n", name, v);
    }
    std::unique_ptr<strata::GgufModel> model;
    try {
        model = std::make_unique<strata::GgufModel>(strata::GgufModel::open(shard));
    } catch (const std::exception& ex) {
        std::fprintf(stderr, "fuse_gu_bench: %s\n", ex.what());
        return 2;
    }
    FILE* tf = nullptr;
    if (!tsv.empty()) {
        FILE* probe = std::fopen(tsv.c_str(), "r");
        const bool fresh = probe == nullptr;
        if (probe) std::fclose(probe);
        tf = std::fopen(tsv.c_str(), "a");
        if (!tf) { std::fprintf(stderr, "cannot open %s\n", tsv.c_str()); return 2; }
        if (fresh)
            std::fprintf(tf, "tag\tgpu\tlayer\tT\tdist\tchoice\tmean_groups\tmean_entries\tplans\tplans_identical\tmax_abs_out\thq_bytes_diff\t"
                             "eager_all_us\teager_gu_us\tgraph_all_us\tgraph_gu_us\tall_GBps\tgu_GBps\tgraph_all_pct_peak\tgraph_gu_pct_peak\n");
    }
    cudaStream_t s;
    FB_CK(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking));
    cudaEvent_t ea, eb;
    FB_CK(cudaEventCreate(&ea));
    FB_CK(cudaEventCreate(&eb));
    // activations
    std::mt19937_64 rng(20261111);
    std::normal_distribution<float> nd(0.f, 1.f);
    std::vector<float> xh((size_t) kMaxT * kH);
    for (auto& v : xh) v = nd(rng);
    for (int t = 0; t < kMaxT; ++t) { xh[(size_t) t * kH + 7] *= 20.f; xh[(size_t) t * kH + 1000] *= 20.f; }
    const int capMax = kMaxT * kTop;
    float* dx = nullptr;
    void *dxq = nullptr, *dscr = nullptr;
    float* dout = nullptr;
    FB_CK(cudaMalloc((void**) &dx, xh.size() * 4));
    FB_CK(cudaMalloc(&dxq, (size_t) kMaxT * kH / 32 * 36));
    const size_t scr_bytes = K::native_expert_scratch_bytes(capMax, kFF);
    FB_CK(cudaMalloc(&dscr, scr_bytes));
    FB_CK(cudaMalloc((void**) &dout, (size_t) capMax * kH * 4));
    FB_CK(cudaMemcpy(dx, xh.data(), xh.size() * 4, cudaMemcpyHostToDevice));
    K::quantize_q8_1_rows(dx, kMaxT, kH, dxq, s);
    FB_CK(cudaStreamSynchronize(s));
    long long bad_cells = 0, validated_cells = 0;
    for (int l : layers) {
        const char* roles[3] = {"gate", "up", "down"};
        const strata::TensorInfo* t[3] = {};
        size_t sh[3] = {};
        for (int r = 0; r < 3; ++r) t[r] = model->find("blk." + std::to_string(l) + ".ffn_" + roles[r] + "_exps.weight", &sh[r]);
        if (!t[0] || !t[1] || !t[2]) { std::printf("# layer %d: no expert tensors, skipped\n", l); continue; }
        cpu::NativeFmt f;
        std::string err;
        if (!cpu::native_fmt((int) t[0]->type, (int) t[2]->type, kH, kFF, f, err)) { std::printf("# layer %d: %s, skipped\n", l, err.c_str()); continue; }
        const K::NativeExpertLayout L = K::native_expert_layout(f.gu_type, f.d_type, kH, kFF);
        if (L.bytes != f.bytes || L.up_off != f.up_off || L.down_off != f.down_off) {
            std::printf("# layer %d: the GPU layout differs from the CPU one, skipped\n", l);
            continue;
        }
        if (!K::native_expert_fuse_gu_takes(L)) {
            std::printf("# layer %d: gate/up type %d / down %d: the fused kernel does not take it, skipped\n", l, f.gu_type, f.d_type);
            continue;
        }
        const size_t stride = (L.bytes + 255) / 256 * 256, dn_role = f.bytes - f.down_off;
        uint8_t* blobs = nullptr;
        FB_CK(cudaMalloc((void**) &blobs, (size_t) kNE * stride));
        for (int e = 0; e < kNE; ++e) {
            uint8_t* b = blobs + (size_t) e * stride;
            FB_CK(cudaMemcpy(b, model->shard(sh[0]).tensor_data(*t[0]) + (size_t) e * f.up_off, f.up_off, cudaMemcpyHostToDevice));
            FB_CK(cudaMemcpy(b + f.up_off, model->shard(sh[1]).tensor_data(*t[1]) + (size_t) e * f.up_off, f.up_off, cudaMemcpyHostToDevice));
            FB_CK(cudaMemcpy(b + f.down_off, model->shard(sh[2]).tensor_data(*t[2]) + (size_t) e * dn_role, dn_role, cudaMemcpyHostToDevice));
        }
        std::printf("# layer %d: gate/up %s, down %s, blob %.2f MB (gate+up %.2f MB), 512 experts on the device\n", l,
                    t[0]->type_name(), t[2]->type_name(), (double) L.bytes / 1e6, 2.0 * (double) L.up_off / 1e6);
        std::vector<int> order(kNE);
        for (int i = 0; i < kNE; ++i) order[(size_t) i] = i;
        std::shuffle(order.begin(), order.end(), rng);
        for (const std::string& dist : dists) {
            std::vector<double> zw;
            if (dist.rfind("cal", 0) == 0) {   // bisect the Zipf exponent for mean U(6) = target
                const double target = std::atof(dist.c_str() + 3);
                double lo = 0.0, hi = 3.0;
                std::mt19937_64 crng(7);
                for (int it = 0; it < 18; ++it) {
                    const double mid = 0.5 * (lo + hi);
                    if (mean_unique(6, zipf_weights(mid, order), 200, crng) > target) lo = mid; else hi = mid;
                }
                zw = zipf_weights(0.5 * (lo + hi), order);
                std::printf("# dist %s: Zipf exponent %.3f\n", dist.c_str(), 0.5 * (lo + hi));
            } else if (dist != "uni" && dist != "s50") {
                std::printf("# dist %s unknown, skipped\n", dist.c_str());
                continue;
            }
            for (int T : Ts) {
                const int cap = T * kTop;
                std::vector<Plan> plans;
                double mg = 0.0, me = 0.0;
                for (int i = 0; i < plans_n; ++i) {
                    plans.push_back(make_plan(rows_for(dist, T, zw, rng)));
                    mg += plans.back().G;
                    me += plans.back().E;
                }
                mg /= plans_n;
                me /= plans_n;
                // device plan arrays at fixed caps
                std::vector<unsigned long long> hp((size_t) plans_n * cap, 0);
                std::vector<int32_t> hs((size_t) plans_n * (cap + 1), 0), hn((size_t) plans_n), ht((size_t) plans_n * cap, 0), hd((size_t) plans_n * cap, 0);
                double bytes_all = 0.0, bytes_gu = 0.0;
                for (int i = 0; i < plans_n; ++i) {
                    const Plan& p = plans[(size_t) i];
                    for (int g = 0; g < p.G; ++g) hp[(size_t) i * cap + g] = (unsigned long long) (blobs + (size_t) p.expert[(size_t) g] * stride);
                    for (int g = 0; g <= p.G; ++g) hs[(size_t) i * (cap + 1) + g] = p.start[(size_t) g];
                    hn[(size_t) i] = p.G;
                    for (int e = 0; e < p.E; ++e) { ht[(size_t) i * cap + e] = p.tok[(size_t) e]; hd[(size_t) i * cap + e] = p.dst[(size_t) e]; }
                    bytes_all += (double) p.G * (double) L.bytes;
                    bytes_gu += (double) p.G * 2.0 * (double) L.up_off;
                }
                bytes_all /= plans_n;
                bytes_gu /= plans_n;
                unsigned long long* dp = nullptr;
                int32_t *ds = nullptr, *dn = nullptr, *dt = nullptr, *dd = nullptr;
                FB_CK(cudaMalloc((void**) &dp, hp.size() * 8));
                FB_CK(cudaMalloc((void**) &ds, hs.size() * 4));
                FB_CK(cudaMalloc((void**) &dn, hn.size() * 4));
                FB_CK(cudaMalloc((void**) &dt, ht.size() * 4));
                FB_CK(cudaMalloc((void**) &dd, hd.size() * 4));
                FB_CK(cudaMemcpy(dp, hp.data(), hp.size() * 8, cudaMemcpyHostToDevice));
                FB_CK(cudaMemcpy(ds, hs.data(), hs.size() * 4, cudaMemcpyHostToDevice));
                FB_CK(cudaMemcpy(dn, hn.data(), hn.size() * 4, cudaMemcpyHostToDevice));
                FB_CK(cudaMemcpy(dt, ht.data(), ht.size() * 4, cudaMemcpyHostToDevice));
                FB_CK(cudaMemcpy(dd, hd.data(), hd.size() * 4, cudaMemcpyHostToDevice));
                auto call = [&](int i) {
                    const int p = i % plans_n;
                    K::native_expert_grouped(L, dp + (size_t) p * cap, ds + (size_t) p * (cap + 1), dn + p, dd + (size_t) p * cap,
                                             dt + (size_t) p * cap, cap, cap, dxq, dscr, dout, s, 0);
                };
                // the q8_1 h blocks: native_expert_grouped's scratch layout (gate | up | h | hq, 256-aligned FP32 parts)
                const size_t fpart = (((size_t) cap * kFF * 4) + 255) & ~(size_t) 255;
                const uint8_t* dhq = (const uint8_t*) dscr + 3 * fpart;
                const size_t hq_bytes = (size_t) cap * kFF / 32 * 36, out_n = (size_t) cap * kH;
                std::vector<std::vector<float>> ref_out((size_t) plans_n);
                std::vector<std::vector<uint8_t>> ref_hq((size_t) plans_n);
                std::printf("# layer %d T=%d %s: %d plans, mean %.1f groups (%.2f rows per expert), %.1f entries, %.1f MB per call\n", l, T,
                            dist.c_str(), plans_n, mg, me / mg, me, bytes_all / 1e6);
                for (size_t ci = 0; ci < choices.size(); ++ci) {
                    const Choice& c = choices[ci];
                    K::native_expert_set_fuse_gu(c.fuse);
                    K::native_expert_set_mode(0, 0);
                    int same_plans = 0;
                    double max_abs = 0.0;
                    long long hq_diff = 0, nonfinite = 0;
                    std::vector<float> o(out_n);
                    std::vector<uint8_t> q(hq_bytes);
                    for (int i = 0; i < plans_n; ++i) {
                        FB_CK(cudaMemsetAsync(dout, 0, out_n * 4, s));
                        FB_CK(cudaMemsetAsync(dscr, 0, scr_bytes, s));
                        call(i);
                        FB_CK(cudaStreamSynchronize(s));
                        FB_CK(cudaGetLastError());
                        FB_CK(cudaMemcpy(o.data(), dout, out_n * 4, cudaMemcpyDeviceToHost));
                        FB_CK(cudaMemcpy(q.data(), dhq, hq_bytes, cudaMemcpyDeviceToHost));
                        if (ci == 0) {
                            for (float v : o) nonfinite += std::isfinite(v) ? 0 : 1;
                            ref_out[(size_t) i] = o;
                            ref_hq[(size_t) i] = q;
                            ++same_plans;
                            continue;
                        }
                        const bool so = std::memcmp(o.data(), ref_out[(size_t) i].data(), out_n * 4) == 0;
                        const bool sq = std::memcmp(q.data(), ref_hq[(size_t) i].data(), hq_bytes) == 0;
                        same_plans += (so && sq) ? 1 : 0;
                        if (!so)
                            for (size_t k = 0; k < out_n; ++k) max_abs = std::max(max_abs, (double) std::fabs(o[k] - ref_out[(size_t) i][k]));
                        if (!sq)
                            for (size_t k = 0; k < hq_bytes; ++k) hq_diff += q[k] != ref_hq[(size_t) i][k];
                    }
                    if (nonfinite) std::printf("# note: def output has %lld non-finite values\n", nonfinite);
                    auto eager = [&](int phase) {
                        K::native_expert_set_mode(0, phase);
                        for (int i = 0; i < 20; ++i) call(i);
                        std::vector<double> rs;
                        for (int r = 0; r < rounds; ++r) {
                            FB_CK(cudaEventRecord(ea, s));
                            for (int i = 0; i < reps; ++i) call(r * reps + i);
                            FB_CK(cudaEventRecord(eb, s));
                            FB_CK(cudaEventSynchronize(eb));
                            float ms = 0.f;
                            FB_CK(cudaEventElapsedTime(&ms, ea, eb));
                            rs.push_back(1000.0 * ms / reps);
                        }
                        K::native_expert_set_mode(0, 0);
                        std::sort(rs.begin(), rs.end());
                        return rs[rs.size() / 2];
                    };
                    auto graphed = [&](int phase) {
                        K::native_expert_set_mode(0, phase);
                        call(0);   // the first call's host-side setup (env reads, notes) outside the capture
                        FB_CK(cudaStreamSynchronize(s));
                        cudaGraph_t gr = nullptr;
                        cudaGraphExec_t ge = nullptr;
                        FB_CK(cudaStreamBeginCapture(s, cudaStreamCaptureModeThreadLocal));
                        for (int i = 0; i < reps; ++i) call(i);
                        FB_CK(cudaStreamEndCapture(s, &gr));
                        FB_CK(cudaGraphInstantiate(&ge, gr, 0));
                        FB_CK(cudaGraphLaunch(ge, s));
                        std::vector<double> rs;
                        for (int r = 0; r < rounds; ++r) {
                            FB_CK(cudaEventRecord(ea, s));
                            FB_CK(cudaGraphLaunch(ge, s));
                            FB_CK(cudaEventRecord(eb, s));
                            FB_CK(cudaEventSynchronize(eb));
                            float ms = 0.f;
                            FB_CK(cudaEventElapsedTime(&ms, ea, eb));
                            rs.push_back(1000.0 * ms / reps);
                        }
                        cudaGraphExecDestroy(ge);
                        cudaGraphDestroy(gr);
                        K::native_expert_set_mode(0, 0);
                        std::sort(rs.begin(), rs.end());
                        return rs[rs.size() / 2];
                    };
                    const double ea_us = eager(0), eg_us = eager(1);
                    const double ga_us = graph ? graphed(0) : 0.0, gg_us = graph ? graphed(1) : 0.0;
                    const double ref_all = graph ? ga_us : ea_us, ref_gu = graph ? gg_us : eg_us;
                    const double gbs_all = bytes_all / (ref_all * 1e-6) / 1e9, gbs_gu = bytes_gu / (ref_gu * 1e-6) / 1e9;
                    const bool ok = same_plans == plans_n;
                    if (ci > 0) ++validated_cells;
                    if (!ok) ++bad_cells;
                    std::printf("%s L%d T=%d %-6s %-4s identical %d/%d max_abs_out %.3g hq_diff %lld | eager all %.1f gu %.1f | graph all %.1f gu %.1f us | %.0f / %.0f GB/s (%.1f%% / %.1f%% peak) %s\n",
                                prop.name, l, T, dist.c_str(), c.name.c_str(), same_plans, plans_n, max_abs, hq_diff, ea_us, eg_us, ga_us, gg_us,
                                gbs_all, gbs_gu, 100.0 * gbs_all * 1e9 / peak, 100.0 * gbs_gu * 1e9 / peak, ok ? "IDENTICAL" : "DIFFERENT");
                    if (tf) {
                        std::fprintf(tf, "%s\t%s\t%d\t%d\t%s\t%s\t%.2f\t%.2f\t%d\t%d\t%.6g\t%lld\t%.2f\t%.2f\t%.2f\t%.2f\t%.1f\t%.1f\t%.2f\t%.2f\n",
                                     tag.c_str(), busid, l, T, dist.c_str(), c.name.c_str(), mg, me, plans_n, same_plans, max_abs, hq_diff,
                                     ea_us, eg_us, ga_us, gg_us, gbs_all, gbs_gu, 100.0 * gbs_all * 1e9 / peak, 100.0 * gbs_gu * 1e9 / peak);
                        std::fflush(tf);
                    }
                }
                K::native_expert_set_fuse_gu(0);
                cudaFree(dp);
                cudaFree(ds);
                cudaFree(dn);
                cudaFree(dt);
                cudaFree(dd);
            }
        }
        cudaFree(blobs);
    }
    if (tf) std::fclose(tf);
    if (validated_cells == 0) {
        std::fprintf(stderr, "fuse_gu_bench: no fused cells validated; check the model, layers, T and distributions\n");
        return 2;
    }
    std::printf("\nfuse_gu_bench: %s (%lld cells differ, %lld fused cells validated)\n", bad_cells ? "FAILED" : "ok", bad_cells, validated_cells);
    return bad_cells ? 1 : 0;
}
