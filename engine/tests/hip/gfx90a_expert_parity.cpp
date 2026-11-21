// M1 portable IQ3_S expert gate. This is type 21, NOT IQ3_XXS/type 18.
// GPU dots use the actual stored Q8_1 bytes and GGML's scalar dequantizer.
// The CPU pool is checked separately against GGML vec_dot on its own Q8_K /
// Q8_0 ABI: that ABI must not be relabeled as the GPU's half-scale Q8_1.
// Not a real pack inventory or complete M2 gate: no MMQ/I8/nibble/MMAP/cache
// offload, combine, alternate layouts, graph replay, or performance coverage.
#include "strata/kernels/iq_kernels.hpp"
#include "strata/kernels/native_mmvq.hpp"
#include "strata/kernels/cpu/native_expert.hpp"
#include "ggml.h"
#include "ggml-cpu.h"
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

#if !defined(STRATA_HIP_GFX90A) || !defined(STRATA_HIP_WAVE64) || defined(STRATA_HIP_GFX906)
#error "Build with the gfx90a production target's wave64 definitions"
#endif

namespace K = strata::kernels;
namespace C = strata::kernels::cpu;
namespace {
void check(hipError_t e, const char* what) {
    if (e != hipSuccess) throw std::runtime_error(std::string(what) + ": " + hipGetErrorString(e));
}
void require(bool ok, const char* what) { if (!ok) throw std::runtime_error(what); }
struct Buffer {
    void* p = nullptr;
    explicit Buffer(size_t bytes) { check(hipMalloc(&p, bytes), "hipMalloc"); }
    ~Buffer() { if (p) hipFree(p); }
    Buffer(const Buffer&) = delete;
    Buffer& operator=(const Buffer&) = delete;
    void put(const void* src, size_t n) { check(hipMemcpy(p, src, n, hipMemcpyHostToDevice), "upload"); }
    void get(void* dst, size_t n) const { check(hipMemcpy(dst, p, n, hipMemcpyDeviceToHost), "download"); }
    template<class T> T* as() const { return static_cast<T*>(p); }
};
struct Stream {
    hipStream_t s = nullptr;
    Stream() { check(hipStreamCreateWithFlags(&s, hipStreamNonBlocking), "create stream"); }
    ~Stream() { if (s) hipStreamDestroy(s); }
};
int device_check() {
    int count = 0;
    const hipError_t e = hipGetDeviceCount(&count);
    if (e == hipErrorNoDevice || (e == hipSuccess && count == 0)) {
        std::puts("SKIP gfx90a_expert_parity: no HIP device; executed=0");
        return 77;
    }
    check(e, "hipGetDeviceCount");
    int dev = 0; check(hipGetDevice(&dev), "hipGetDevice");
    hipDeviceProp_t prop{}; check(hipGetDeviceProperties(&prop, dev), "hipGetDeviceProperties");
    const std::string raw = prop.gcnArchName;
    require(raw.substr(0, raw.find(':')) == "gfx90a" && prop.warpSize == 64,
            "requires exact gfx90a (feature suffix allowed) and physical wave64");
    std::printf("gfx90a_expert_parity device=%d arch=%s wave=%d\n", dev, raw.c_str(), prop.warpSize);
    return 0;
}

std::vector<uint8_t> matrix(int type, int rows, int cols, unsigned seed) {
    // All tested formats have one half scale at offset 0. Legal random index,
    // scale-code and sign payloads exercise the real IQ3_S codebook.
    const int block = type == 21 ? 256 : type == 20 ? 32 : 64;
    const size_t bytes = ggml_row_size((ggml_type) type, cols);
    const size_t bb = bytes / (cols / block);
    std::vector<uint8_t> result((size_t) rows * bytes);
    uint32_t state = seed;
    for (auto& v : result) { state = state * 1664525u + 1013904223u; v = (uint8_t) (state >> 24); }
    for (size_t b = 0; b < result.size() / bb; ++b) {
        const ggml_fp16_t d = ggml_fp32_to_fp16(std::ldexp((b & 1) ? -1.0f : 1.0f, -13 - (int) (b % 3)));
        std::memcpy(result.data() + b * bb, &d, 2);
    }
    return result;
}

std::vector<float> decode(int type, const uint8_t* data, int rows, int cols) {
    const auto* tr = ggml_get_type_traits((ggml_type) type);
    require(tr && tr->to_float, "missing independent GGML dequantizer");
    const size_t rb = ggml_row_size((ggml_type) type, cols);
    std::vector<float> f((size_t) rows * cols);
    for (int r = 0; r < rows; ++r) tr->to_float(data + (size_t) r * rb, f.data() + (size_t) r * cols, cols);
    return f;
}

std::vector<uint8_t> input(int tokens, int width, unsigned seed, std::vector<float>& x) {
    x.resize((size_t) tokens * width);
    std::vector<uint8_t> q((size_t) tokens * (width / 32) * 36, 0);
    for (int t = 0; t < tokens; ++t) for (int b = 0; b < width / 32; ++b) {
        uint8_t* dst = q.data() + ((size_t) t * (width / 32) + b) * 36;
        const float d = t == 1 ? 0.0f : std::ldexp(1.0f, -8 + b % 2);
        int sum = 0;
        for (int l = 0; l < 32; ++l) {
            const int v = t == 1 ? 0 : l == 0 ? 127 : (int) ((l * 31 + b * 7 + t * 19 + seed) % 255) - 127;
            const int8_t code = (int8_t) v;
            std::memcpy(dst + 4 + l, &code, 1);
            x[(size_t) t * width + b * 32 + l] = d * v; sum += v;
        }
        const ggml_fp16_t dh = ggml_fp32_to_fp16(d), sh = ggml_fp32_to_fp16(d * sum);
        std::memcpy(dst, &dh, 2); std::memcpy(dst + 2, &sh, 2);
    }
    return q;
}
std::vector<float> decode_q8(const std::vector<uint8_t>& q) {
    require(q.size() % 36 == 0, "bad Q8_1 byte count");
    std::vector<float> x(q.size() / 36 * 32);
    for (size_t b = 0; b < q.size() / 36; ++b) {
        ggml_fp16_t h; std::memcpy(&h, q.data() + b * 36, 2);
        const float d = ggml_fp16_to_fp32(h);
        for (int i = 0; i < 32; ++i) {
            int8_t code; std::memcpy(&code, q.data() + b * 36 + 4 + i, 1);
            x[b * 32 + i] = d * code;
        }
    }
    return x;
}

struct Metrics {
    double err2 = 0, ref2 = 0, max_abs = 0, max_rel = 0;
    void add(float got, double ref, double budget) {
        const double e = std::fabs((double) got - ref);
        require(std::isfinite(got) && std::isfinite(ref) && e <= budget, "stage exceeds independent oracle budget");
        err2 += e * e; ref2 += ref * ref; max_abs = std::max(max_abs, e);
        max_rel = std::max(max_rel, e / std::max(std::fabs(ref), 1e-12));
    }
    void report(const char* stage) const {
        require(ref2 > 0, "vacuous all-zero stage oracle");
        std::printf(" stage=%s abs=%.3e rel=%.3e l2=%.3e\n", stage, max_abs, max_rel, std::sqrt(err2 / ref2));
    }
};
double budget(int n, double magnitude) {
    const double nu = n * std::ldexp(1.0, -24);
    return 8 * nu / (1 - nu) * magnitude + 2e-6;
}
double dot(const float* a, const float* b, int n, double& magnitude) {
    double sum = 0; magnitude = 0;
    for (int i = 0; i < n; ++i) { const double v = (double) a[i] * b[i]; sum += v; magnitude += std::fabs(v); }
    return sum;
}

// CPU-native paths read Q8_K for IQ3_S GU, Q8_0 for both tested downs.
// Compare them to GGML on identical activation bytes, never to a Q8_1 dot.
void cpu_reference(const C::NativeFmt& f, const std::vector<uint8_t>& blob, const float* x) {
    const auto* gu = ggml_get_type_traits_cpu((ggml_type) f.gu_type);
    const auto* down = ggml_get_type_traits_cpu((ggml_type) f.d_type);
    require(gu && gu->vec_dot && down && down->vec_dot, "missing GGML CPU dot oracle");
    require((int) gu->vec_dot_type == f.gu_act && (int) down->vec_dot_type == f.d_act,
            "CPU activation ABI mismatch");
    std::vector<uint8_t> act(f.act_bytes), hq(f.h_bytes);
    std::vector<float> h((size_t) f.n_ff), out((size_t) f.n_embd);
    C::native_quant_act(f, x, act.data());
    const void* ap[] = {act.data()}; float* hp[] = {h.data()};
    C::native_gu_rows_invariant(f, blob.data(), ap, 1, hp, 0, (int) f.n_ff);
    Metrics hm, dm;
    for (int r = 0; r < f.n_ff; ++r) {
        float g = 0, u = 0;
        gu->vec_dot((int) f.n_embd, &g, 0, blob.data() + (size_t) r * f.gu_row, 0, act.data(), 0, 1);
        gu->vec_dot((int) f.n_embd, &u, 0, blob.data() + f.up_off + (size_t) r * f.gu_row, 0, act.data(), 0, 1);
        const float ref = (g / (1.0f + std::exp(-g))) * u;
        hm.add(h[r], ref, budget((int) f.n_embd, std::fabs(ref)));
    }
    C::native_quant_h(f, h.data(), hq.data());
    const void* hqp[] = {hq.data()}; float* op[] = {out.data()};
    C::native_down_rows_invariant(f, blob.data(), hqp, 1, op, 0, (int) f.n_embd);
    for (int r = 0; r < f.n_embd; ++r) {
        float ref = 0;
        down->vec_dot((int) f.n_ff, &ref, 0, blob.data() + f.down_off + (size_t) r * f.d_row, 0, hq.data(), 0, 1);
        dm.add(out[r], ref, budget((int) f.n_ff, std::fabs(ref)));
    }
    hm.report("CPU-GU-SwiGLU-vs-GGML-same-ABI"); dm.report("CPU-down-vs-GGML-same-ABI");
}

void quantized_hidden(const std::vector<float>& h, const std::vector<uint8_t>& q) {
    require(h.size() / 32 * 36 == q.size(), "hidden Q8_1 ABI mismatch");
    for (size_t b = 0; b < h.size() / 32; ++b) {
        float amax = 0; double sum = 0, magnitude = 0;
        for (int l = 0; l < 32; ++l) {
            const float v = h[b * 32 + l]; require(std::isfinite(v), "nonfinite hidden activation");
            amax = std::max(amax, std::fabs(v)); sum += v; magnitude += std::fabs(v);
        }
        const float d = amax / 127.0f;
        ggml_fp16_t dh, sh; std::memcpy(&dh, q.data() + b * 36, 2); std::memcpy(&sh, q.data() + b * 36 + 2, 2);
        require(dh == ggml_fp32_to_fp16(d), "hidden Q8_1 stored scale mismatch");
        require(std::isfinite(ggml_fp16_to_fp32(sh)), "nonfinite Q8_1 sum");
        // Half storage rounds once; the reduction itself is FP32.
        require(std::fabs(ggml_fp16_to_fp32(sh) - sum) <= budget(32, magnitude) + std::fabs(sum) / 1024 + 1e-7,
                "hidden Q8_1 stored sum mismatch");
        for (int l = 0; l < 32; ++l) {
            int8_t code; std::memcpy(&code, q.data() + b * 36 + 4 + l, 1);
            const float ratio = amax ? h[b * 32 + l] / d : 0;
            const int expected = (int) std::round(ratio);
            // HIP division rounding can straddle an exact half-integer; only
            // that boundary permits its adjacent integer, not arbitrary drift.
            const bool boundary = std::fabs(std::fabs(ratio - std::trunc(ratio)) - 0.5f) <= 4e-5f;
            require(code == expected || (boundary && std::abs((int) code - expected) == 1), "hidden Q8_1 code mismatch");
        }
    }
}

void run(int down_type, int width, int hidden, const std::vector<int>& counts, unsigned seed,
         hipStream_t s, int& executed) {
    const int groups = (int) counts.size();
    int entries = 0; for (int n : counts) entries += n;
    require(entries > 0 && groups > 0, "invalid nonempty test fixture");
    const auto L = K::native_expert_layout(21, down_type, width, hidden);
    require(K::native_expert_supported(21, down_type, width, hidden), "required expert geometry unavailable");
    C::NativeFmt cpu; std::string error;
    require(C::native_fmt(21, down_type, width, hidden, cpu, error), error.c_str());
    require(cpu.gu_row == L.gu_row && cpu.d_row == L.d_row && cpu.bytes == L.bytes && cpu.up_off == L.up_off && cpu.down_off == L.down_off,
            "CPU/GPU weight blob ABI mismatch");
    std::vector<uint8_t> blobs((size_t) groups * L.bytes);
    for (int g = 0; g < groups; ++g) {
        const auto gate = matrix(21, hidden, width, seed + g * 11);
        const auto up = matrix(21, hidden, width, seed + g * 11 + 1);
        const auto down = matrix(down_type, width, hidden, seed + g * 11 + 2);
        std::memcpy(blobs.data() + (size_t) g * L.bytes, gate.data(), gate.size());
        std::memcpy(blobs.data() + (size_t) g * L.bytes + L.up_off, up.data(), up.size());
        std::memcpy(blobs.data() + (size_t) g * L.bytes + L.down_off, down.data(), down.size());
    }
    std::vector<float> x;
    const auto xq = input(4, width, seed, x);
    const auto xstored = decode_q8(xq);
    // Every group uses shared token IDs in a different order and scatters into
    // a full reverse permutation; the empty group consumes no entry.
    std::vector<int32_t> starts(groups + 1, 0), tokens(entries), dst(entries);
    for (int g = 0; g < groups; ++g) starts[g + 1] = starts[g] + counts[g];
    for (int e = 0; e < entries; ++e) { tokens[e] = (e * 3) % 4; dst[e] = entries - 1 - e; }
    Buffer dw(blobs.size()), dx(x.size() * 4), dxq(xq.size()), dp((size_t) groups * sizeof(unsigned long long));
    Buffer db(starts.size() * 4), dn(4), dt(tokens.size() * 4), dd(dst.size() * 4);
    std::vector<unsigned long long> pointers(groups);
    for (int g = 0; g < groups; ++g) pointers[g] = (unsigned long long) (uintptr_t) (dw.as<uint8_t>() + (size_t) g * L.bytes);
    dw.put(blobs.data(), blobs.size()); dx.put(x.data(), x.size() * 4); dp.put(pointers.data(), pointers.size() * sizeof(pointers[0]));
    db.put(starts.data(), starts.size() * 4); dn.put(&groups, 4); dt.put(tokens.data(), tokens.size() * 4); dd.put(dst.data(), dst.size() * 4);
    K::quantize_q8_1_rows(dx.as<float>(), 4, width, dxq.p, (void*) s);
    check(hipStreamSynchronize(s), "input quantization");
    std::vector<uint8_t> gotq(xq.size()); dxq.get(gotq.data(), gotq.size());
    require(gotq == xq, "expert input quantizer differs from exact CPU Q8_1 fixture");
    const size_t scratch_size = K::native_expert_scratch_bytes(entries, hidden);
    Buffer scratch(scratch_size);
    check(hipMemset(scratch.p, 0xff, scratch_size), "scratch sentinel");
    constexpr size_t guard = 16; constexpr float fence = -12345.25f;
    std::vector<float> out((size_t) entries * width + 2 * guard, fence);
    std::fill(out.begin() + guard, out.end() - guard, std::nanf(""));
    Buffer dy(out.size() * 4); dy.put(out.data(), out.size() * 4);
    // V1 materializes every stage in production scratch. Mode 0 is the M1
    // portable layout; opt-in fused/I8/nibble paths are deliberately separate.
    K::native_grouped_set_v1(true);
    K::native_expert_grouped(L, dp.as<unsigned long long>(), db.as<int32_t>(), dn.as<int32_t>(), dd.as<int32_t>(), dt.as<int32_t>(),
                             groups, entries, dxq.p, scratch.p, dy.as<float>() + guard, (void*) s);
    check(hipGetLastError(), "expert launch"); check(hipStreamSynchronize(s), "expert synchronize");
    dy.get(out.data(), out.size() * 4);
    for (size_t i = 0; i < guard; ++i) require(out[i] == fence && out[out.size() - 1 - i] == fence, "expert output guard overwritten");
    const size_t nf = (size_t) entries * hidden, fa = (nf * 4 + 255) & ~(size_t) 255;
    std::vector<uint8_t> stages(scratch_size); scratch.get(stages.data(), stages.size());
    std::vector<float> gate(nf), up(nf), h(nf);
    std::memcpy(gate.data(), stages.data(), nf * 4); std::memcpy(up.data(), stages.data() + fa, nf * 4);
    std::memcpy(h.data(), stages.data() + 2 * fa, nf * 4);
    std::vector<uint8_t> hq(nf / 32 * 36); std::memcpy(hq.data(), stages.data() + 3 * fa, hq.size());
    quantized_hidden(h, hq); const auto hstored = decode_q8(hq);
    Metrics gm, um, hm, dm;
    for (int g = 0; g < groups; ++g) {
        const uint8_t* blob = blobs.data() + (size_t) g * L.bytes;
        const auto G = decode(21, blob, hidden, width), U = decode(21, blob + L.up_off, hidden, width);
        const auto D = decode(down_type, blob + L.down_off, width, hidden);
        for (int e = starts[g]; e < starts[g + 1]; ++e) {
            const float* act = xstored.data() + (size_t) tokens[e] * width;
            for (int r = 0; r < hidden; ++r) {
                double mag = 0; const size_t i = (size_t) e * hidden + r;
                const double gr = dot(G.data() + (size_t) r * width, act, width, mag);
                gm.add(gate[i], gr, budget(width, mag));
                const double ur = dot(U.data() + (size_t) r * width, act, width, mag);
                um.add(up[i], ur, budget(width, mag));
                const double hr = ((double) gate[i] / (1 + std::exp(-(double) gate[i]))) * up[i];
                hm.add(h[i], hr, 3e-6 * std::fabs(hr) + 1e-8);
            }
            for (int r = 0; r < width; ++r) {
                double mag = 0; const double ref = dot(D.data() + (size_t) r * hidden, hstored.data() + (size_t) e * hidden, hidden, mag);
                dm.add(out[guard + (size_t) dst[e] * width + r], ref, budget(hidden, mag));
                if (tokens[e] == 1) require(out[guard + (size_t) dst[e] * width + r] == 0, "zero token expert output is nonzero");
            }
        }
    }
    gm.report("GPU-gate"); um.report("GPU-up"); hm.report("GPU-SwiGLU"); dm.report("GPU-down-scatter");
    cpu_reference(cpu, std::vector<uint8_t>(blobs.begin(), blobs.begin() + L.bytes), x.data());
    // Default grouped launches must preserve all results of the materialized
    // pipeline, not merely agree with another GPU implementation by themselves.
    const auto v1 = out;
    K::native_grouped_set_v1(false);
    std::fill(out.begin() + guard, out.end() - guard, std::nanf("")); dy.put(out.data(), out.size() * 4);
    K::native_expert_grouped(L, dp.as<unsigned long long>(), db.as<int32_t>(), dn.as<int32_t>(), dd.as<int32_t>(), dt.as<int32_t>(),
                             groups, entries, dxq.p, scratch.p, dy.as<float>() + guard, (void*) s);
    check(hipGetLastError(), "default expert launch"); check(hipStreamSynchronize(s), "default expert synchronize");
    dy.get(out.data(), out.size() * 4);
    require(std::memcmp(out.data(), v1.data(), out.size() * 4) == 0, "default grouped pipeline differs from checked V1 pipeline");
    std::vector<uint8_t> after(blobs.size()); dw.get(after.data(), after.size()); require(after == blobs, "expert modified weights");
    dxq.get(gotq.data(), gotq.size()); require(gotq == xq, "expert modified Q8_1 input");
    ++executed;
    std::printf("case=%d IQ3_S/%s geometry=%d/%d groups=%d entries=%d seed=%u PASS\n", executed,
                down_type == 20 ? "IQ4_NL" : "Q2_0", width, hidden, groups, entries, seed);
}
} // namespace

int main() {
    int executed = 0;
    try {
        const int status = device_check(); if (status) return status;
        // Isolate the portable M1 path from process environment tuning knobs.
        for (const char* name : {"STRATA_EXPERT_V2", "STRATA_EXPERT_V2K", "STRATA_GU_SWIGLU"})
            require(setenv(name, "0", 1) == 0, "cannot disable optional expert layout");
        K::native_expert_set_mode(0, 0); K::native_expert_set_expdec(0); K::iq_set_old_kernels(false);
        require(C::native_experts_available(), "native CPU expert reference not linked");
        ggml_cpu_init();
        Stream stream;
        // Capability queries reject a non-block-aligned shape without launch.
        require(!K::native_expert_supported(21, 20, 255, 64), "invalid geometry accepted");
        for (int down : {20, 42}) {
            run(down, 256, 64, {1}, 17, stream.s, executed);
            run(down, 256, 64, {3, 0, 2}, 31, stream.s, executed);
            run(down, 256, 64, {0, 1, 0, 2, 1, 0, 3, 0, 1, 2}, 47, stream.s, executed);
            run(down, 2560, 640, {2, 1}, 71, stream.s, executed);
        }
        require(executed == 8, "missing required expert cases");
        std::printf("gfx90a_expert_parity PASS executed=%d expected=8 GU=IQ3_S down=IQ4_NL,Q2_0 MMQ=not-used\n", executed);
        std::puts("NOT_COVERED: real-pack inventory, other formats/activation ABIs, combine, cache/mmap/offload, graphs, alternate layouts, performance");
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "gfx90a_expert_parity FAIL executed=%d: %s\n", executed, e.what());
        return 1;
    }
}
