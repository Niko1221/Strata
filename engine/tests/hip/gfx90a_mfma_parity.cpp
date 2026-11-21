// Primitive probes only: this file does not enable a production MFMA dispatch.
// Signatures: ROCm/llvm-project tag rocm-7.2.4,
// clang/include/clang/Basic/BuiltinsAMDGPU.def (BF16_1k takes short4 raw bits).
// Layout checked for all A/B/C/D elements with AMD's official calculator:
// https://github.com/ROCm/amd_matrix_instruction_calculator
//   -a gfx90a -i v_mfma_f32_16x16x16f16 -A/-B/-C/-D --matrix-layout
// The BF16_1k and I32_16X16X16I8 instructions use the same coordinates.
#include "strata/platform/hip_compat/strata_hip.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

#if !defined(STRATA_HIP_GFX90A) || !defined(STRATA_HIP_WAVE64) || defined(STRATA_HIP_GFX906)
#error "Build this primitive probe with the gfx90a production target's definitions"
#endif
#if defined(__HIP_DEVICE_COMPILE__)
#if !defined(__gfx90a__)
#error "The MFMA primitive probe must target gfx90a"
#endif
#if !__has_builtin(__builtin_amdgcn_mfma_f32_16x16x16f16) || \
    !__has_builtin(__builtin_amdgcn_mfma_f32_16x16x16bf16_1k) || \
    !__has_builtin(__builtin_amdgcn_mfma_i32_16x16x16i8)
#error "This compiler lacks a required gfx90a MFMA builtin; no fallback is allowed"
#endif
#endif

#define HIP_CHECK(call) do { \
    const hipError_t error = (call); \
    if (error != hipSuccess) { \
        std::fprintf(stderr, "%s: %s\n", #call, hipGetErrorString(error)); \
        return 2; \
    } \
} while (0)

namespace {
constexpr int kM = 16, kN = 16, kMaxK = 64, kTiles = 8, kLdy = 21, kGuard = 16;
constexpr int kOutputCount = 2 * kGuard + kTiles * kM * kLdy;
constexpr uint32_t kFloatGuard = 0x7fc12345u;
constexpr int32_t kIntGuard = 0x5a6b7c8d;
const char* const kFixture[] = {"identity A", "identity B", "one-hot", "coordinates", "cancellation",
                              "rounded random", "range", "zero product"};

template <bool BF16>
__global__ void float_probe(const uint16_t* a, const uint16_t* b, const float* c, float* out, int K) {
#if defined(__HIP_DEVICE_COMPILE__)
    using h4 = _Float16 __attribute__((ext_vector_type(4)));
    using s4 = short __attribute__((ext_vector_type(4)));
    using f4 = float __attribute__((ext_vector_type(4)));
    const int tile = blockIdx.x, lane = threadIdx.x;
    const int outer = lane % 16, group = lane / 16;
    f4 acc;
    for (int i = 0; i < 4; ++i) acc[i] = c[tile * kM * kN + (4 * group + i) * kN + outer];
    for (int k0 = 0; k0 < K; k0 += 16) {
        // A[row=outer][k], B[k][col=outer], four consecutive K values per lane.
        uint16_t ar[4], br[4];
        for (int i = 0; i < 4; ++i) {
            const int k = k0 + 4 * group + i;
            ar[i] = a[tile * kM * kMaxK + outer * kMaxK + k];
            br[i] = b[tile * kMaxK * kN + k * kN + outer];
        }
        if constexpr (BF16) {
            s4 av, bv;
            __builtin_memcpy(&av, ar, sizeof(av));
            __builtin_memcpy(&bv, br, sizeof(bv));
            acc = __builtin_amdgcn_mfma_f32_16x16x16bf16_1k(av, bv, acc, 0, 0, 0);
        } else {
            h4 av, bv;
            __builtin_memcpy(&av, ar, sizeof(av));
            __builtin_memcpy(&bv, br, sizeof(bv));
            acc = __builtin_amdgcn_mfma_f32_16x16x16f16(av, bv, acc, 0, 0, 0);
        }
    }
    for (int i = 0; i < 4; ++i)
        out[kGuard + tile * kM * kLdy + (4 * group + i) * kLdy + outer] = acc[i];
#else
    (void) a; (void) b; (void) c; (void) out; (void) K;
#endif
}

__global__ void int_probe(const int8_t* a, const int8_t* b, const int32_t* c, int32_t* out, int K) {
#if defined(__HIP_DEVICE_COMPILE__)
    using i4 = int __attribute__((ext_vector_type(4)));
    const int tile = blockIdx.x, lane = threadIdx.x;
    const int outer = lane % 16, group = lane / 16;
    i4 acc;
    for (int i = 0; i < 4; ++i) acc[i] = c[tile * kM * kN + (4 * group + i) * kN + outer];
    for (int k0 = 0; k0 < K; k0 += 16) {
        uint32_t av = 0, bv = 0;
        for (int i = 0; i < 4; ++i) {
            const int k = k0 + 4 * group + i;
            av |= uint32_t(uint8_t(a[tile * kM * kMaxK + outer * kMaxK + k])) << (8 * i);
            bv |= uint32_t(uint8_t(b[tile * kMaxK * kN + k * kN + outer])) << (8 * i);
        }
        acc = __builtin_amdgcn_mfma_i32_16x16x16i8(__builtin_bit_cast(int, av), __builtin_bit_cast(int, bv), acc, 0, 0, 0);
    }
    for (int i = 0; i < 4; ++i)
        out[kGuard + tile * kM * kLdy + (4 * group + i) * kLdy + outer] = acc[i];
#else
    (void) a; (void) b; (void) c; (void) out; (void) K;
#endif
}

uint32_t float_bits(float value) {
    uint32_t bits;
    std::memcpy(&bits, &value, sizeof(bits));
    return bits;
}
float bits_float(uint32_t bits) {
    float value;
    std::memcpy(&value, &bits, sizeof(value));
    return value;
}
uint16_t encode(float value, bool bf16) {
    if (bf16) {
        const uint32_t bits = float_bits(value);
        return uint16_t((bits + 0x7fffu + ((bits >> 16) & 1u)) >> 16); // finite RNE
    }
    const _Float16 half = (_Float16) value;
    uint16_t bits;
    static_assert(sizeof(half) == sizeof(bits));
    std::memcpy(&bits, &half, sizeof(bits));
    return bits;
}
double decode(uint16_t bits, bool bf16) {
    if (bf16) return bits_float(uint32_t(bits) << 16);
    // Decode the stored FP16 bits independently of the device load/MFMA path.
    const int exponent = (bits >> 10) & 31, mantissa = bits & 1023;
    const double magnitude = exponent == 0 ? std::ldexp(double(mantissa), -24)
                                          : std::ldexp(double(1024 + mantissa), exponent - 25);
    return (bits & 0x8000) ? -magnitude : magnitude;
}

uint32_t next(uint32_t& state) {
    state ^= state << 13; state ^= state >> 17; state ^= state << 5;
    return state;
}

float input_value(int tile, bool A, int outer, int k, bool bf16, uint32_t& state) {
    switch (tile) {
        case 0: return A ? float(outer == k) : float(3 * k + outer - 20) / 8;
        case 1: return A ? float(outer - 2 * k + 9) / 8 : float(k == outer);
        case 2: return A ? (outer == 11 && k == 13 ? 2.0f : 0.0f)
                         : (k == 13 && outer == 7 ? -3.0f : 0.0f);
        case 3: return float((A ? 17 * outer + 3 * k : 5 * k + 7 * outer) % 19 - 9) / 8;
        case 4: return A ? float((k & 1) ? -(outer % 7 + 1) : outer % 7 + 1) / 8
                         : float((k / 2 + outer) % 7 - 3) / 8;
        case 5: return float(int(next(state) % 20001) - 10000) / 7000;
        case 6: { // BF16 values exceed FP16 range: an accidental FP16 path cannot pass.
            const float scale = bf16 ? 1048576.0f : 128.0f;
            return A ? float((outer + k) % 7 - 3) * scale : float((k + 2 * outer) % 7 - 3) / scale;
        }
        default: return A ? 0.0f : float(k + outer + 1) / 8;
    }
}

bool float_results(const std::vector<uint16_t>& a, const std::vector<uint16_t>& b, const std::vector<float>& c,
                   const std::vector<float>& out, int K, bool bf16) {
    const char* dtype = bf16 ? "BF16_1k" : "FP16";
    double worst = 0, worst_relative = 0, squared_error = 0, squared_reference = 0;
    for (int tile = 0; tile < kTiles; ++tile) {
        bool nonzero = false;
        for (int row = 0; row < kM; ++row) for (int col = 0; col < kN; ++col) {
            double ref = c[tile * kM * kN + row * kN + col], magnitude = std::abs(ref);
            for (int k = 0; k < K; ++k) {
                const double product = decode(a[tile * kM * kMaxK + row * kMaxK + k], bf16) *
                                       decode(b[tile * kMaxK * kN + k * kN + col], bf16);
                ref += product;
                magnitude += std::abs(product);
            }
            const float got = out[kGuard + tile * kM * kLdy + row * kLdy + col];
            // Binary-fraction fixtures are exact in FP32. Only rounded-random
            // inputs use gamma_(K+1)*sum(abs(products), abs(C)), fixed in advance.
            const double n_u = (K + 1) * std::ldexp(1.0, -24);
            const double bound = tile == 5 ? (n_u / (1 - n_u)) * magnitude : 0;
            const double error = std::abs(double(got) - ref);
            if (!std::isfinite(got) || !std::isfinite(ref) || error > bound) {
                std::fprintf(stderr, "FAIL %s K=%d %s (%d,%d): got=%.9g ref=%.17g error=%.9g bound=%.9g\n",
                             dtype, K, kFixture[tile], row, col, got, ref, error, bound);
                return false;
            }
            nonzero |= ref != 0;
            if (error > worst) worst = error;
            if (ref != 0 && error / std::abs(ref) > worst_relative) worst_relative = error / std::abs(ref);
            squared_error += error * error;
            squared_reference += ref * ref;
        }
        if (!nonzero) { std::fprintf(stderr, "FAIL %s fixture has no nonzero reference\n", dtype); return false; }
    }
    for (int i = 0; i < kOutputCount; ++i) {
        const int relative = i - kGuard;
        const bool matrix = relative >= 0 && relative < kTiles * kM * kLdy && relative % kLdy < kN;
        if (!matrix && float_bits(out[i]) != kFloatGuard) {
            std::fprintf(stderr, "FAIL %s K=%d output guard/padding at %d\n", dtype, K, i);
            return false;
        }
    }
    std::printf("PASS primitive %s K=%d tiles=%d elements=%d max_abs=%.9g max_rel=%.9g rel_l2=%.9g\n",
                dtype, K, kTiles, kTiles * kM * kN, worst, worst_relative, std::sqrt(squared_error / squared_reference));
    return true;
}
} // namespace

int main() {
    int count = 0;
    const hipError_t discovery = hipGetDeviceCount(&count);
    if (discovery == hipErrorNoDevice || (discovery == hipSuccess && count == 0)) {
        std::puts("SKIP gfx90a_mfma_parity: no HIP device (MI210 execution remains required)");
        return 77;
    }
    HIP_CHECK(discovery);
    int device = 0;
    HIP_CHECK(hipGetDevice(&device));
    hipDeviceProp_t prop{};
    HIP_CHECK(hipGetDeviceProperties(&prop, device));
    if (std::strncmp(prop.gcnArchName, "gfx90a", 6) != 0 ||
        (prop.gcnArchName[6] != '\0' && prop.gcnArchName[6] != ':') || prop.warpSize != 64) {
        std::fprintf(stderr, "FAIL expected gfx90a wave64, got %s wave%d\n", prop.gcnArchName, prop.warpSize);
        return 1;
    }
    std::printf("gfx90a_mfma_parity primitive-only: device=%d arch=%s wave=%d\n", device, prop.gcnArchName, prop.warpSize);
    int launches = 0;
    uint16_t *da = nullptr, *db = nullptr;
    float *dc = nullptr, *dy = nullptr;
    HIP_CHECK(hipMalloc((void**) &da, kTiles * kM * kMaxK * sizeof(uint16_t)));
    HIP_CHECK(hipMalloc((void**) &db, kTiles * kMaxK * kN * sizeof(uint16_t)));
    HIP_CHECK(hipMalloc((void**) &dc, kTiles * kM * kN * sizeof(float)));
    HIP_CHECK(hipMalloc((void**) &dy, kOutputCount * sizeof(float)));
    for (bool bf16 : {false, true}) for (int K : {16, 64}) {
        std::vector<uint16_t> a(kTiles * kM * kMaxK), b(kTiles * kMaxK * kN);
        std::vector<float> c(kTiles * kM * kN), out(kOutputCount, bits_float(kFloatGuard));
        uint32_t state = 0x12345678u;
        for (int tile = 0; tile < kTiles; ++tile) {
            for (int row = 0; row < kM; ++row) for (int k = 0; k < K; ++k)
                a[tile * kM * kMaxK + row * kMaxK + k] = encode(input_value(tile, true, row, k, bf16, state), bf16);
            for (int k = 0; k < K; ++k) for (int col = 0; col < kN; ++col)
                b[tile * kMaxK * kN + k * kN + col] = encode(input_value(tile, false, col, k, bf16, state), bf16);
            for (int row = 0; row < kM; ++row) for (int col = 0; col < kN; ++col)
                c[tile * kM * kN + row * kN + col] = (tile + 1) * 0.25f + (row * kN + col) / 32.0f;
        }
        HIP_CHECK(hipMemcpy(da, a.data(), a.size() * sizeof(uint16_t), hipMemcpyHostToDevice));
        HIP_CHECK(hipMemcpy(db, b.data(), b.size() * sizeof(uint16_t), hipMemcpyHostToDevice));
        HIP_CHECK(hipMemcpy(dc, c.data(), c.size() * sizeof(float), hipMemcpyHostToDevice));
        HIP_CHECK(hipMemcpy(dy, out.data(), out.size() * sizeof(float), hipMemcpyHostToDevice));
        if (bf16) hipLaunchKernelGGL(float_probe<true>, dim3(kTiles), dim3(64), 0, 0, da, db, dc, dy, K);
        else hipLaunchKernelGGL(float_probe<false>, dim3(kTiles), dim3(64), 0, 0, da, db, dc, dy, K);
        HIP_CHECK(hipGetLastError());
        HIP_CHECK(hipDeviceSynchronize());
        ++launches;
        HIP_CHECK(hipMemcpy(out.data(), dy, out.size() * sizeof(float), hipMemcpyDeviceToHost));
        if (!float_results(a, b, c, out, K, bf16)) return 1;
    }
    HIP_CHECK(hipFree(dy)); HIP_CHECK(hipFree(dc)); HIP_CHECK(hipFree(db)); HIP_CHECK(hipFree(da));

    int8_t *dia = nullptr, *dib = nullptr;
    int32_t *dic = nullptr, *diy = nullptr;
    HIP_CHECK(hipMalloc((void**) &dia, kTiles * kM * kMaxK));
    HIP_CHECK(hipMalloc((void**) &dib, kTiles * kMaxK * kN));
    HIP_CHECK(hipMalloc((void**) &dic, kTiles * kM * kN * sizeof(int32_t)));
    HIP_CHECK(hipMalloc((void**) &diy, kOutputCount * sizeof(int32_t)));
    for (int K : {16, 64}) {
        std::vector<int8_t> a(kTiles * kM * kMaxK), b(kTiles * kMaxK * kN);
        std::vector<int32_t> c(kTiles * kM * kN), out(kOutputCount, kIntGuard);
        uint32_t state = 0x89abcdefu;
        for (int tile = 0; tile < kTiles; ++tile) {
            for (int row = 0; row < kM; ++row) for (int k = 0; k < K; ++k) {
                int value = int(input_value(tile, true, row, k, false, state) * 8);
                if (tile == 5) value = int(next(state) & 255) - 128;
                if (tile == 6) value = (k & 1) ? -128 : 127;
                a[tile * kM * kMaxK + row * kMaxK + k] = int8_t(value);
            }
            for (int k = 0; k < K; ++k) for (int col = 0; col < kN; ++col) {
                int value = int(input_value(tile, false, col, k, false, state) * 8);
                if (tile == 5) value = int(next(state) & 255) - 128;
                if (tile == 6) value = (col & 1) ? -128 : 127;
                b[tile * kMaxK * kN + k * kN + col] = int8_t(value);
            }
            for (int i = 0; i < kM * kN; ++i) c[tile * kM * kN + i] = 1 + 1000 * tile + i;
        }
        HIP_CHECK(hipMemcpy(dia, a.data(), a.size(), hipMemcpyHostToDevice));
        HIP_CHECK(hipMemcpy(dib, b.data(), b.size(), hipMemcpyHostToDevice));
        HIP_CHECK(hipMemcpy(dic, c.data(), c.size() * sizeof(int32_t), hipMemcpyHostToDevice));
        HIP_CHECK(hipMemcpy(diy, out.data(), out.size() * sizeof(int32_t), hipMemcpyHostToDevice));
        hipLaunchKernelGGL(int_probe, dim3(kTiles), dim3(64), 0, 0, dia, dib, dic, diy, K);
        HIP_CHECK(hipGetLastError());
        HIP_CHECK(hipDeviceSynchronize());
        ++launches;
        HIP_CHECK(hipMemcpy(out.data(), diy, out.size() * sizeof(int32_t), hipMemcpyDeviceToHost));
        for (int tile = 0; tile < kTiles; ++tile) for (int row = 0; row < kM; ++row) for (int col = 0; col < kN; ++col) {
            int64_t ref = c[tile * kM * kN + row * kN + col];
            for (int k = 0; k < K; ++k)
                ref += int64_t(a[tile * kM * kMaxK + row * kMaxK + k]) * b[tile * kMaxK * kN + k * kN + col];
            const int32_t got = out[kGuard + tile * kM * kLdy + row * kLdy + col];
            if (got != ref) { // These fixtures cannot overflow INT32; every element must be exact.
                std::fprintf(stderr, "FAIL INT8 K=%d %s (%d,%d): got=%d ref=%lld\n", K, kFixture[tile], row, col, got, (long long) ref);
                return 1;
            }
        }
        for (int i = 0; i < kOutputCount; ++i) {
            const int relative = i - kGuard;
            const bool matrix = relative >= 0 && relative < kTiles * kM * kLdy && relative % kLdy < kN;
            if (!matrix && out[i] != kIntGuard) { std::fprintf(stderr, "FAIL INT8 output guard/padding at %d\n", i); return 1; }
        }
        std::printf("PASS primitive INT8 K=%d tiles=%d elements=%d exact=1\n", K, kTiles, kTiles * kM * kN);
    }
    HIP_CHECK(hipFree(diy)); HIP_CHECK(hipFree(dic)); HIP_CHECK(hipFree(dib)); HIP_CHECK(hipFree(dia));
    if (launches != 6) { std::fprintf(stderr, "FAIL expected 6 primitive launches, got %d\n", launches); return 1; }
    std::printf("PASS gfx90a_mfma_parity: primitive_launches=%d tiles=%d elements=%d skipped=0 production_dispatch=not_tested\n",
                launches, launches * kTiles, launches * kTiles * kM * kN);
    return 0;
}
