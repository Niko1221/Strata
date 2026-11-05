// src/kernels/cpu/cpu_dispatch_dump.cpp - round 0216: what the CPU pool's expert calls (native_gu_rows + native_down_rows, both
// arithmetics) put out, as a hash per (pair, token count, row range).  Run the same binary with a change switched off and on (or a
// baseline build and a candidate build) and compare the lines: an E change (bit-exact against the code it replaces) gives the same text.
//
//     cpu_dispatch_dump [--pairs iq4_xs/q5_0,...] [--nt-max 8] [--experts 3]
//
// For every pair: --experts random experts (ggml-quantized random weights, valid scales), nt = 1..nt-max tokens, whole rows and
// row ranges that start and end off any vector boundary (tails), the Default and the WindowInvariant entry points.
#include "strata/kernels/cpu/expert_layout.hpp"
#include "strata/kernels/cpu/native_expert.hpp"

#include "ggml.h"
#include "ggml-cpu.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

namespace cpu = strata::kernels::cpu;

namespace {

constexpr int64_t kH = 2560, kFF = 640;
constexpr int kMaxT = 8;

int type_of(const std::string& s) {
    for (int t = 0; t < GGML_TYPE_COUNT; ++t) {
        const char* n = ggml_type_name((ggml_type) t);
        if (n != nullptr && s == n) return t;
    }
    return -1;
}

std::vector<std::string> split(const std::string& s) {
    std::vector<std::string> v;
    size_t from = 0;
    for (;;) {
        const size_t c = s.find(',', from);
        v.push_back(s.substr(from, c == std::string::npos ? std::string::npos : c - from));
        if (c == std::string::npos) break;
        from = c + 1;
    }
    return v;
}

uint64_t fnv(uint64_t h, const void* p, size_t n) {
    const uint8_t* b = (const uint8_t*) p;
    for (size_t i = 0; i < n; ++i) h = (h ^ b[i]) * 1099511628211ull;
    return h;
}

}  // namespace

int main(int argc, char** argv) {
    setvbuf(stdout, nullptr, _IONBF, 0);
    std::vector<std::string> pairs = split("iq3_xxs/q5_0,iq3_xxs/iq4_nl,iq4_xs/q5_0,iq4_xs/iq4_nl,iq3_s/iq4_nl,iq2_s/iq4_nl,iq4_xs/q8_0");
    int nt_max = 8, experts = 3;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--pairs" && i + 1 < argc) pairs = split(argv[++i]);
        else if (a == "--nt-max" && i + 1 < argc) nt_max = std::atoi(argv[++i]);
        else if (a == "--experts" && i + 1 < argc) experts = std::atoi(argv[++i]);
        else { std::fprintf(stderr, "usage: cpu_dispatch_dump [--pairs a/b,...] [--nt-max 8] [--experts 3]\n"); return 2; }
    }
    ggml_cpu_init();
    for (const std::string& pair : pairs) {
        const size_t slash = pair.find('/');
        const int gt = slash == std::string::npos ? -1 : type_of(pair.substr(0, slash));
        const int dt = slash == std::string::npos ? -1 : type_of(pair.substr(slash + 1));
        cpu::NativeFmt f;
        std::string err;
        if (gt < 0 || dt < 0 || !cpu::native_fmt(gt, dt, kH, kFF, f, err)) {
            std::printf("%s: %s\n", pair.c_str(), err.empty() ? "unknown types" : err.c_str());
            continue;
        }
        ggml_quantize_init((ggml_type) gt);
        ggml_quantize_init((ggml_type) dt);
        std::mt19937 rng(4321 + gt * 64 + dt);
        std::normal_distribution<float> nd(0.f, 0.02f), xd(0.f, 1.f);
        std::vector<float> row((size_t) std::max(kH, kFF));
        const std::vector<float> imatrix((size_t) std::max(kH, kFF), 1.0f);
        auto quant = [&](int type, uint8_t* dst, int64_t n) {
            ggml_quantize_chunk((ggml_type) type, row.data(), dst, 0, 1, n, ggml_quantize_requires_imatrix((ggml_type) type) ? imatrix.data() : nullptr);
        };
        std::vector<uint8_t> blob(f.bytes + 64);
        std::vector<std::vector<uint8_t>> xa(kMaxT, std::vector<uint8_t>(cpu::kNativeActBytes)), ha(kMaxT, std::vector<uint8_t>(cpu::kNativeHBytes));
        std::vector<std::vector<float>> ffv(kMaxT, std::vector<float>(kFF)), outv(kMaxT, std::vector<float>(kH));
        const void* act[kMaxT];
        const void* hq[kMaxT];
        float* ff[kMaxT];
        float* out[kMaxT];
        for (int t = 0; t < kMaxT; ++t) { act[t] = xa[(size_t) t].data(); hq[t] = ha[(size_t) t].data(); ff[t] = ffv[(size_t) t].data(); out[t] = outv[(size_t) t].data(); }
        uint64_t total = 1469598103934665603ull;
        for (int e = 0; e < experts; ++e) {
            for (int64_t r = 0; r < 2 * kFF; ++r) {
                for (int64_t i = 0; i < kH; ++i) row[(size_t) i] = nd(rng);
                quant(f.gu_type, blob.data() + (size_t) r * f.gu_row, kH);
            }
            for (int64_t r = 0; r < kH; ++r) {
                for (int64_t i = 0; i < kFF; ++i) row[(size_t) i] = nd(rng);
                quant(f.d_type, blob.data() + f.down_off + (size_t) r * f.d_row, kFF);
            }
            for (int t = 0; t < kMaxT; ++t) {
                std::vector<float> x(kH), h(kFF);
                for (auto& v : x) v = xd(rng);
                for (auto& v : h) v = 0.3f * xd(rng);
                cpu::native_quant_act(f, x.data(), xa[(size_t) t].data());
                cpu::native_quant_h(f, h.data(), ha[(size_t) t].data());
            }
            for (int nt = 1; nt <= nt_max && nt <= kMaxT; ++nt) {
                // whole range, then a range starting and ending off any 8/16/64 boundary
                const int ranges[2][4] = {{0, (int) kFF, 0, (int) kH}, {3, (int) kFF - 5, 7, (int) kH - 9}};
                for (int inv = 0; inv < 2; ++inv) {
                    for (int rg = 0; rg < 2; ++rg) {
                        for (auto& v : ffv) std::fill(v.begin(), v.end(), 0.f);
                        for (auto& v : outv) std::fill(v.begin(), v.end(), 0.f);
                        if (inv) {
                            cpu::native_gu_rows_invariant(f, blob.data(), act, nt, ff, ranges[rg][0], ranges[rg][1]);
                            cpu::native_down_rows_invariant(f, blob.data(), hq, nt, out, ranges[rg][2], ranges[rg][3]);
                        } else {
                            cpu::native_gu_rows(f, blob.data(), act, nt, ff, ranges[rg][0], ranges[rg][1]);
                            cpu::native_down_rows(f, blob.data(), hq, nt, out, ranges[rg][2], ranges[rg][3]);
                        }
                        uint64_t hg = 1469598103934665603ull, hd = 1469598103934665603ull;
                        for (int t = 0; t < nt; ++t) {
                            hg = fnv(hg, ffv[(size_t) t].data(), sizeof(float) * (size_t) kFF);
                            hd = fnv(hd, outv[(size_t) t].data(), sizeof(float) * (size_t) kH);
                        }
                        total = fnv(fnv(total, &hg, 8), &hd, 8);
                        std::printf("%s e%d nt%d %s range%d gu %016llx down %016llx\n", pair.c_str(), e, nt, inv ? "inv" : "def", rg,
                                    (unsigned long long) hg, (unsigned long long) hd);
                    }
                }
            }
        }
        std::printf("%s TOTAL %016llx\n", pair.c_str(), (unsigned long long) total);
    }
    return 0;
}
