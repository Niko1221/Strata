// src/prefill/moe_ws_down_bench.cu - moe_ws_down_bench: the weight-stationary down (ws::down) on one GP100 die with
// and without STRATA_WS_FAST (ws::set_fast), byte for byte and timed.  Not built by default (EXCLUDE_FROM_ALL; the
// STRATA_EXPERIMENTAL_SM60 build; needs STRATA_MMQ_WS=1 and a cc 6.0 device).
//
//   moe_ws_down_bench <shard.gguf> [--layers L,..] [--t T,..] [--models ln0.5,ln1.0,flat32] [--reps N]
//
// Per layer whose down experts are IQ4_NL or Q2_0: all 512 experts' down matrices read from the GGUF into one VRAM
// blob, per T and routing model the counts of the T * 10 routed rows over the experts ("lnS": lognormal popularity,
// sigma S, each <= T; "flatN": experts of exactly N rows), ws::chunks' table, the GU rows random and H's q8_1 rows
// made by mmq::swiglu_quant as the prompt path makes them.  ws::down runs once with set_fast(0) (the reference) and
// once with set_fast(1); the two Dm buffers (prefilled with different bytes) must be equal.  Then both are timed,
// interleaved, a warm-up round first; medians.  Exit 1 on any difference.
#include "strata/artifact/gguf_reader.hpp"
#include "strata/prefill/moe_mmq.hpp"
#include "strata/prefill/moe_ws.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#define CK(x)                                                                                                       \
    do {                                                                                                            \
        cudaError_t e_ = (x);                                                                                       \
        if (e_ != cudaSuccess) {                                                                                    \
            std::fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(e_), __FILE__, __LINE__);         \
            std::exit(3);                                                                                           \
        }                                                                                                           \
    } while (0)

namespace ws = strata::prefill::ws;
namespace mmq = strata::prefill::mmq;

namespace {

constexpr int64_t kEmbd = 2560, kFf = 640, kK = 10, kNE = 512;
constexpr int kIQ4_NL = 20, kQ2_0 = 42;

__global__ void fill_gu(float* x, long long n, unsigned long long seed) {
    const long long i = (long long) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    unsigned long long h = seed ^ (unsigned long long) i * 0x9E3779B97F4A7C15ull;
    h ^= h >> 33; h *= 0xFF51AFD7ED558CCDull; h ^= h >> 33; h *= 0xC4CEB9FE1A85EC53ull; h ^= h >> 33;
    const float u = (float) (h & 0xFFFFFF) / 16777216.0f - 0.5f;
    x[i] = ((h >> 40) % 97 == 0) ? 24.0f * u : 4.0f * u;
}

std::vector<std::string> split_list(const char* s) {
    std::vector<std::string> v;
    std::string t(s);
    size_t p = 0;
    while (p < t.size()) {
        size_t q = t.find(',', p);
        if (q == std::string::npos) q = t.size();
        v.push_back(t.substr(p, q - p));
        p = q + 1;
    }
    return v;
}

// the counts of T * K rows over the 512 experts
std::vector<int32_t> make_counts(const std::string& model, int T, int layer) {
    const int rows = T * (int) kK;
    std::vector<int32_t> cnt(kNE, 0);
    if (model.rfind("flat", 0) == 0) {
        const int n = std::atoi(model.c_str() + 4);
        int left = rows;
        for (int e = 0; e < kNE && left > 0; ++e) {
            cnt[e] = std::min(n, left);
            left -= cnt[e];
        }
        return cnt;
    }
    const double sigma = std::atof(model.c_str() + 2);
    std::mt19937_64 rng(1000003ull * (uint64_t) T + (uint64_t) (sigma * 1000) + 7 + (uint64_t) layer);
    std::normal_distribution<double> nd(0.0, 1.0);
    std::vector<double> wgt(kNE);
    for (auto& x : wgt) x = std::exp(sigma * nd(rng));
    const double S = std::accumulate(wgt.begin(), wgt.end(), 0.0);
    std::vector<std::pair<double, int>> fr;
    int sum = 0;
    for (int e = 0; e < kNE; ++e) {
        const double cc = rows * wgt[e] / S;
        cnt[e] = std::min((int) std::floor(cc), T);
        sum += cnt[e];
        fr.push_back({ cc - std::floor(cc), e });
    }
    std::sort(fr.begin(), fr.end(), [](auto& a, auto& b) { return a.first > b.first; });
    for (size_t i = 0; sum < rows && i < 4 * (size_t) kNE; ++i) {
        const int e = fr[i % kNE].second;
        if (cnt[e] < T) { ++cnt[e]; ++sum; }
    }
    return cnt;
}

double median(std::vector<double> v) {
    std::sort(v.begin(), v.end());
    return v.empty() ? 0.0 : v[v.size() / 2];
}

}  // namespace

int main(int argc, char** argv) {
    setvbuf(stdout, nullptr, _IONBF, 0);
    std::vector<std::string> shards, models = { "ln0.5", "ln1.0", "flat32" };
    std::vector<int> layers = { 0, 2 }, ts = { 768, 1024, 1536, 2048, 4096 };
    int reps = 15;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&]() -> const char* {
            if (i + 1 >= argc) {
                std::fprintf(stderr, "%s needs a value\n", a.c_str());
                std::exit(2);
            }
            return argv[++i];
        };
        if (a == "--layers") { layers.clear(); for (auto& s : split_list(next())) layers.push_back(std::atoi(s.c_str())); }
        else if (a == "--t") { ts.clear(); for (auto& s : split_list(next())) ts.push_back(std::atoi(s.c_str())); }
        else if (a == "--models") models = split_list(next());
        else if (a == "--reps") reps = std::atoi(next());
        else if (a.size() > 5 && a.substr(a.size() - 5) == ".gguf") shards.push_back(a);
        else {
            std::fprintf(stderr, "usage: moe_ws_down_bench <shard.gguf> [--layers L,..] [--t T,..] "
                                 "[--models ln0.5,ln1.0,flat32] [--reps N]\n");
            return 2;
        }
    }
    if (shards.empty()) {
        std::fprintf(stderr, "moe_ws_down_bench: give a model shard (.gguf)\n");
        return 2;
    }
    cudaDeviceProp p{};
    CK(cudaGetDeviceProperties(&p, 0));
    std::printf("device 0: %s, cc %d.%d, %d SMs, clock %d kHz; ws::mode() %d\n", p.name, p.major, p.minor,
                p.multiProcessorCount, p.clockRate, ws::mode());
    if (ws::mode() != 1) {
        std::fprintf(stderr, "needs the WS kernels (STRATA_EXPERIMENTAL_SM60 build, STRATA_MMQ_WS=1, a cc 6.0 device)\n");
        return 2;
    }
    std::unique_ptr<strata::GgufModel> model;
    try {
        model = std::make_unique<strata::GgufModel>(strata::GgufModel::open(shards[0]));
    } catch (const std::exception& ex) {
        std::fprintf(stderr, "%s\n", ex.what());
        return 2;
    }
    cudaEvent_t e0, e1;
    CK(cudaEventCreate(&e0));
    CK(cudaEventCreate(&e1));
    auto time_once = [&](auto&& f) {
        CK(cudaEventRecord(e0));
        f();
        CK(cudaEventRecord(e1));
        CK(cudaEventSynchronize(e1));
        CK(cudaGetLastError());
        float ms = 0;
        CK(cudaEventElapsedTime(&ms, e0, e1));
        return (double) ms;
    };
    long long fails = 0, checks = 0;
    for (int l : layers) {
        size_t sh = 0;
        const strata::TensorInfo* t = model->find("blk." + std::to_string(l) + ".ffn_down_exps.weight", &sh);
        if (t == nullptr) {
            std::printf("layer %d: no ffn_down_exps\n", l);
            ++fails;
            continue;
        }
        const int dt = (int) t->type;
        if (dt != kIQ4_NL && dt != kQ2_0) {
            std::printf("layer %d: down type %d is not IQ4_NL / Q2_0, skipped\n", l, dt);
            continue;
        }
        const size_t per = mmq::matrix_bytes(dt, kEmbd, kFf);
        uint8_t* W = nullptr;
        CK(cudaMalloc(&W, (size_t) kNE * per));
        CK(cudaMemcpy(W, model->shard(sh).tensor_data(*t), (size_t) kNE * per, cudaMemcpyHostToDevice));
        std::printf("\n##### layer %d: down %s (%d), %zu B per expert\n", l, t->type_name(), dt, per);
        int64_t* dxo = nullptr;
        CK(cudaMalloc(&dxo, kNE * 8));
        for (int T : ts)
            for (const std::string& mdl : models) {
                const std::vector<int32_t> cnt = make_counts(mdl, T, l);
                std::vector<int32_t> order;
                int R = 0;
                for (int e = 0; e < kNE; ++e)
                    if (cnt[e] > 0) { order.push_back(e); R += cnt[e]; }
                std::stable_sort(order.begin(), order.end(),
                                 [&](int a, int b) { return cnt[a] != cnt[b] ? cnt[a] < cnt[b] : a < b; });
                const int n = (int) order.size();
                std::vector<int32_t> bounds(n + 1, 0);
                for (int j = 0; j < n; ++j) bounds[j + 1] = bounds[j] + cnt[order[j]];
                std::vector<ws::Chunk> ch(ws::max_chunks(R, n));
                const size_t nch = ws::chunks(order.data(), (size_t) n, cnt.data(), bounds.data(), ch.data());
                const double u = (double) R / n;
                std::vector<int64_t> hxo(n);
                for (int j = 0; j < n; ++j) hxo[j] = (int64_t) order[j] * (int64_t) per;
                CK(cudaMemcpy(dxo, hxo.data(), (size_t) n * 8, cudaMemcpyHostToDevice));
                ws::Chunk* dch = nullptr;
                CK(cudaMalloc(&dch, nch * sizeof(ws::Chunk)));
                CK(cudaMemcpy(dch, ch.data(), nch * sizeof(ws::Chunk), cudaMemcpyHostToDevice));
                const size_t gub = (size_t) R * 2 * kFf * 4, dmb = (size_t) R * kEmbd * 4, hqb = mmq::q8_bytes(R, kFf);
                float *GU = nullptr, *DMa = nullptr, *DMb = nullptr;
                void* HQ = nullptr;
                CK(cudaMalloc(&GU, gub));
                CK(cudaMalloc(&HQ, hqb));
                CK(cudaMalloc(&DMa, dmb));
                CK(cudaMalloc(&DMb, dmb));
                fill_gu<<<(unsigned) (((long long) R * 2 * kFf + 255) / 256), 256>>>(GU, (long long) R * 2 * kFf,
                                                                                    0xD0A11ull + (uint64_t) R + (uint64_t) l);
                CK(cudaGetLastError());
                CK(cudaMemset(HQ, 0, hqb));
                mmq::swiglu_quant(GU, HQ, R, false, nullptr);
                CK(cudaDeviceSynchronize());
                CK(cudaGetLastError());
                ws::Layer L;
                L.gu_type = -1; L.d_type = dt; L.w = W; L.xoff = dxo; L.down_off = 0;
                L.chunks = dch; L.nchunks = (int64_t) nch; L.rows = R; L.n = n; L.hq = HQ;
                ws::Layer La = L, Lb = L;
                La.dm = DMa;
                Lb.dm = DMb;
                CK(cudaMemset(DMa, 0xFF, dmb));
                CK(cudaMemset(DMb, 0x00, dmb));
                ws::set_fast(0);
                ws::down(La, nullptr);
                ws::set_fast(1);
                ws::down(Lb, nullptr);
                ws::set_fast(0);
                CK(cudaDeviceSynchronize());
                CK(cudaGetLastError());
                std::vector<uint8_t> a(dmb), b(dmb);
                CK(cudaMemcpy(a.data(), DMa, dmb, cudaMemcpyDeviceToHost));
                CK(cudaMemcpy(b.data(), DMb, dmb, cudaMemcpyDeviceToHost));
                long long rows_diff = 0;
                for (int r = 0; r < R; ++r)
                    rows_diff += std::memcmp(a.data() + (size_t) r * kEmbd * 4, b.data() + (size_t) r * kEmbd * 4, kEmbd * 4) != 0;
                const bool tier32 = dt == kIQ4_NL && u >= 20.0;
                ++checks;
                if (rows_diff) ++fails;
                std::vector<double> ta, tb;
                for (int rep = -1; rep < reps; ++rep) {
                    const double x = time_once([&] { ws::set_fast(0); ws::down(La, nullptr); });
                    const double y = time_once([&] { ws::set_fast(1); ws::down(Lb, nullptr); ws::set_fast(0); });
                    if (rep >= 0) { ta.push_back(x); tb.push_back(y); }
                }
                const double ma = median(ta), mb = median(tb);
                std::printf("L%d %-6s T %5d  rows %6d over %3d experts (u %6.2f, %zu chunks; %s): default %8.3f ms, FAST %8.3f ms, "
                            "x%.3f; Dm %s\n",
                            l, mdl.c_str(), T, R, n, u, nch, tier32 ? "32-token tier, fast kernel" : "default kernel either way",
                            ma, mb, mb / ma, rows_diff ? ("DIFFER in " + std::to_string(rows_diff) + " rows").c_str() : "memcmp equal");
                CK(cudaFree(dch));
                CK(cudaFree(GU));
                CK(cudaFree(HQ));
                CK(cudaFree(DMa));
                CK(cudaFree(DMb));
            }
        CK(cudaFree(dxo));
        CK(cudaFree(W));
    }
    std::printf("\n%s: %lld checks, %lld failures\n", fails ? "FAIL" : "ALL CHECKS EQUAL", checks, fails);
    return fails ? 1 : 0;
}
