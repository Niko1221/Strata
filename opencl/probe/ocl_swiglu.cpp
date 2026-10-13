// opencl/probe/ocl_swiglu.cpp - SwiGLU + q8_1 activation quantisation, on the MI50, via OpenCL.
//
// This is the pass between the gate/up and down matrices of one expert: gu produces 2*n_ff values,
// SwiGLU folds them into n_ff, and those are quantised to q8_1 so the down kernel can do integer dots.
// Without it the MI50 tier could not chain an expert on its own.
//
// Transcribed from src/kernels/cuda/iq_kernels.cu (swiglu_q8_1_entries_kernel, q8_1_store) and
// include/strata/kernels/q8_1_finite.hpp.  Four details carry the value:
//
//   * the SwiGLU product is rounded on its own - g/(1+exp(-g)) times up, one rounding, and it must not
//     contract into the first add of the block sum.  #pragma OPENCL FP_CONTRACT OFF is what guarantees
//     that here (the CUDA side spells it __fmul_rn, and the HIP side needs contract(off) for the same
//     reason - this is the third back end, same requirement).
//   * the block's amax and sum come from an xor butterfly over the 32 lanes, offsets 16,8,4,2,1
//     (q8_1_store lines 1334-1339).  The max is order-insensitive but the *sum* is not: it is stored as
//     an fp16 the down kernel multiplies by, so a different reduction tree gives different low bits.
//     Four spellings are swept here so the difference is measured rather than assumed.
//   * d = amax/127 is clamped to the largest finite half (q8_1_finite), and so is the stored sum.
//     Without it one huge activation among 32 makes d or sum an fp16 inf, the dot products read
//     inf*0, and a NaN in the residual ends as a token repeated forever.
//   * q = roundf(x / d) clamped to +-127, and 0 for an all-zero block.  roundf, not rint: ties go away
//     from zero, so a value landing exactly on a half step must round up in magnitude.  A block whose
//     amax is 254 has d = 2.0 and a value of 1.0 lands exactly on the tie - that block is planted below
//     on purpose, because rint would quietly give a different code.
//
// n_ff is a multiple of 32, so a 32-lane work-group is exactly one q8_1 block and the row bounds are
// uniform across it - the same argument the CUDA kernel makes.
//
// Verified two ways: `q8_1_from_h` takes the activations directly so the reduction, the quantiser and the
// store can be checked bit-exactly against a host model with no exp() in the way, and `swiglu_q8_1`
// exercises the fused form whose only inexactness is the driver's exp().
//
// Usage: ocl_swiglu [repeat] [copies] [-1|0|1]

#define CL_TARGET_OPENCL_VERSION 200
#include <CL/cl.h>

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace {

const char* err_name(cl_int e) {
    switch (e) {
        case CL_SUCCESS: return "CL_SUCCESS";
        case CL_BUILD_PROGRAM_FAILURE: return "CL_BUILD_PROGRAM_FAILURE";
        case CL_INVALID_KERNEL: return "CL_INVALID_KERNEL";
        case CL_INVALID_KERNEL_NAME: return "CL_INVALID_KERNEL_NAME";
        case CL_OUT_OF_RESOURCES: return "CL_OUT_OF_RESOURCES";
        case CL_MEM_OBJECT_ALLOCATION_FAILURE: return "CL_MEM_OBJECT_ALLOCATION_FAILURE";
        case CL_INVALID_WORK_GROUP_SIZE: return "CL_INVALID_WORK_GROUP_SIZE";
        case CL_INVALID_PROGRAM: return "CL_INVALID_PROGRAM";
        default: return "CL_ERR";
    }
}

// ---- host model of the same pass, so the GPU output can be diffed byte for byte ----------------------
//
// q8_1_store's butterfly is a tree over the *lane* index, so the host has to replay the tree in the same
// order rather than call std::accumulate: floating point addition is not associative and the sum is
// stored, so the order is part of the answer.

uint16_t f32_to_f16(float v) {
    uint32_t x;
    std::memcpy(&x, &v, 4);
    const uint32_t sign = (x >> 16) & 0x8000u;
    int32_t e = (int32_t) ((x >> 23) & 0xFFu) - 127 + 15;
    uint32_t m = x & 0x7FFFFFu;
    if (e >= 31) return (uint16_t) (sign | 0x7C00u);          // inf / overflow
    if (e <= 0) {
        if (e < -10) return (uint16_t) sign;
        m |= 0x800000u;
        const uint32_t sh = (uint32_t) (14 - e);
        uint32_t r = m >> sh;
        const uint32_t rem = m & ((1u << sh) - 1u);           // round to nearest even
        const uint32_t half = 1u << (sh - 1);
        if (rem > half || (rem == half && (r & 1u))) r += 1;
        return (uint16_t) (sign | r);
    }
    uint32_t r = m >> 13;
    const uint32_t rem = m & 0x1FFFu;
    if (rem > 0x1000u || (rem == 0x1000u && (r & 1u))) {
        r += 1;
        if (r == 0x400u) { r = 0; e += 1; if (e >= 31) return (uint16_t) (sign | 0x7C00u); }
    }
    return (uint16_t) (sign | ((uint32_t) e << 10) | r);
}

constexpr float kHalfMax = 65504.0f;
float q8_1_finite(float v) { return std::fabs(v) > kHalfMax ? std::copysign(kHalfMax, v) : v; }

/// Same as q8_1_finite.hpp's q8_1_quant: roundf (ties away from zero), clamped to +-127, 0 if all-zero.
int8_t q8_1_quant(float xi, float d, float amax) {
    if (amax == 0.0f) return 0;
    const float q = std::round(xi / d);
    const float c = q > 127.0f ? 127.0f : (q < -127.0f ? -127.0f : q);
    return (int8_t) c;
}

/// One q8_1 block from 32 activations, replaying q8_1_store's xor butterfly in the same order.
void host_q8_1(const float* h, uint16_t* ds, int8_t* qs) {
    float amax_v[32], sum_v[32];
    for (int i = 0; i < 32; ++i) { amax_v[i] = std::fabs(h[i]); sum_v[i] = h[i]; }
    for (int o = 16; o > 0; o >>= 1)
        for (int i = 0; i < 32; ++i)
            if (!(i & o)) {
                amax_v[i] = std::fmax(amax_v[i], amax_v[i ^ o]);
                sum_v[i] = sum_v[i] + sum_v[i ^ o];
            }
    const float amax = amax_v[0], sum = sum_v[0];
    const float d = q8_1_finite(amax / 127.0f);
    for (int i = 0; i < 32; ++i) qs[i] = q8_1_quant(h[i], d, amax);
    ds[0] = f32_to_f16(q8_1_finite(d));
    ds[1] = f32_to_f16(q8_1_finite(sum));
}

// ---- the kernel source ------------------------------------------------------------------------------
//
// The reduction is a knob because the answer depends on it: only the xor butterfly reproduces
// q8_1_store's addition order.  Local-memory forms need the two __local scratch pointers, the built-in
// reduce forms do not, so the parameter list is assembled from a macro rather than duplicated.

enum Reduce { R_XOR = 0, R_TREE = 1, R_SUBGROUP = 2, R_WORKGROUP = 3 };

struct Mode {
    const char* name;
    Reduce r;
    bool local;      // needs __local scratch arguments
};

const Mode kModes[] = {
    {"xor butterfly (CUDA order)", R_XOR, true},
    {"lane+off tree (local mem)",  R_TREE, true},
    {"sub_group_reduce_max/add",   R_SUBGROUP, false},
    {"work_group_reduce_max/add",  R_WORKGROUP, false},
};
constexpr int kNumModes = (int) (sizeof(kModes) / sizeof(kModes[0]));

/// `expo` picks the transcendental: 0 = exp (the accurate one), 1 = native_exp (the fast one, the
/// closest match for CUDA's __expf).
std::string source(Reduce r, int expo, bool fused) {
    std::string s;
    s += "#pragma OPENCL FP_CONTRACT OFF\n";   // the SwiGLU product must round on its own
    if (r == R_SUBGROUP) s += "#pragma OPENCL EXTENSION cl_khr_subgroups : enable\n";
    s += "#define IQ_R " + std::to_string((int) r) + "\n";
    s += "#define IQ_EXPFN " + std::string(expo ? "native_exp" : "exp") + "\n";
    if (kModes[r].local) {
        s += "#define IQ_LOCAL_DECL , __local float* lmem, __local float* lsc\n";
        s += "#define IQ_LOCAL_ARGS , lmem, lsc\n";
    } else {
        s += "#define IQ_LOCAL_DECL\n";
        s += "#define IQ_LOCAL_ARGS\n";
    }
    s += R"CLC(
inline float f16_max_clamp(float v) {
    return fabs(v) > 65504.0f ? copysign(65504.0f, v) : v;
}

// q = round(x/d) clamped to +-127, and 0 when the block is all zeros.  round() in OpenCL C is
// "ties away from zero", which is roundf on the CUDA side; rint() would be ties-to-even and differs.
inline int q8_1_quant(float xi, float d, float amax) {
    if (amax == 0.0f) return 0;
    const float q = round(xi / d);
    return (int) (q > 127.0f ? 127.0f : (q < -127.0f ? -127.0f : q));
}

// One 32-value q8_1 block: lane = position in the block.  out_ds holds the clamped f32 scale and sum in
// slots 2b, 2b+1 (the host converts to fp16 with the same round-to-nearest-even the hardware uses).
inline void q8_1_store(float h, uint lane, __global float* out_ds, __global char* out_qs,
                       uint block IQ_LOCAL_DECL)
{
    float amax, sum;
#if IQ_R == 0
    // q8_1_store's butterfly: x[i] op x[i^16], then i^8, i^4, i^2, i^1.  Every lane ends up with the
    // same value, but the addition *order* is the one the reference sum was built with.
    lmem[lane] = fabs(h);
    lsc[lane] = h;
    for (uint o = 16u; o > 0u; o >>= 1u) {
        barrier(CLK_LOCAL_MEM_FENCE);
        const float va = lmem[lane ^ o];
        const float vs = lsc[lane ^ o];
        barrier(CLK_LOCAL_MEM_FENCE);
        lmem[lane] = fmax(lmem[lane], va);
        lsc[lane] = lsc[lane] + vs;
    }
    amax = lmem[lane];
    sum = lsc[lane];
#elif IQ_R == 1
    // lane op lane+off: same max, a different tree for the sum
    lmem[lane] = fabs(h);
    lsc[lane] = h;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (uint o = 16u; o > 0u; o >>= 1u) {
        if (lane < o) {
            lmem[lane] = fmax(lmem[lane], lmem[lane + o]);
            lsc[lane] = lsc[lane] + lsc[lane + o];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    amax = lmem[0];
    sum = lsc[0];
#elif IQ_R == 2
    amax = sub_group_reduce_max(fabs(h));
    sum = sub_group_reduce_add(h);
#else
    amax = work_group_reduce_max(fabs(h));
    sum = work_group_reduce_add(h);
#endif

    const float d = f16_max_clamp(amax / 127.0f);
    out_qs[block * 32u + lane] = (char) q8_1_quant(h, d, amax);
    if (lane == 0u) {
        out_ds[block * 2u + 0u] = d;
        out_ds[block * 2u + 1u] = f16_max_clamp(sum);
    }
}
)CLC";
    const char* body_fused = R"CLC(
// n_ff is a multiple of 32, so one 32-lane work-group is exactly one q8_1 block and every lane of it
// reads the same row, so the bounds are work-group uniform.  Gate and up are interleaved per row:
// [gate row 0 | up row 0 | gate row 1 | up row 1 | ...].
//
// get_global_id(0) is the element index, not the work-group index - the host dispatches exactly one
// work-item per value, so `i` is the global id and the block is i/32.  Multiplying the global id by the
// block width again is a 32x overshoot on every index, including the stores.
__kernel __attribute__((reqd_work_group_size(32, 1, 1)))
void swiglu_q8_1(__global const float* gu, __global float* out_ds, __global char* out_qs,
                 const uint n_ff, const uint n_values IQ_LOCAL_DECL)
{
    const uint i = get_global_id(0);
    if (i >= n_values) return;          // dead while the host dispatches exactly n_values (a multiple of 32)
    const uint lane = i & 31u;
    const uint block = i >> 5;
    const uint row = i / n_ff;
    const uint col = i - row * n_ff;
    const float g = gu[row * 2u * n_ff + col];
    const float u = gu[row * 2u * n_ff + n_ff + col];
    const float h = (g / (1.0f + IQ_EXPFN(-g))) * u;
    q8_1_store(h, lane, out_ds, out_qs, block IQ_LOCAL_ARGS);
}
)CLC";
    const char* body_plain = R"CLC(
// The same quantiser with h handed straight in, so the reduction, the quantiser and the store can be
// checked without exp() in the way.  n_values is a multiple of 32, so no lane returns early - the
// barriers below need the whole work-group to reach them.
__kernel __attribute__((reqd_work_group_size(32, 1, 1)))
void q8_1_from_h(__global const float* h_in, __global float* out_ds, __global char* out_qs,
                 const uint n_values IQ_LOCAL_DECL)
{
    const uint i = get_global_id(0);    // the element index, not the work-group index
    if (i >= n_values) return;
    const uint lane = i & 31u;
    const uint block = i >> 5;
    q8_1_store(h_in[i], lane, out_ds, out_qs, block IQ_LOCAL_ARGS);
}
)CLC";
    s += fused ? body_fused : body_plain;
    return s;
}

/// The SwiGLU activation the host model pairs with the kernel's.
float host_swiglu(float g, float u) { return (g / (1.0f + std::exp(-g))) * u; }

/// What q8_1_store should have produced for a given activation array.
struct Expect {
    std::vector<uint16_t> ds;
    std::vector<int8_t> qs;
};

Expect host_expect(const std::vector<float>& h, unsigned n_blocks) {
    Expect e;
    e.ds.resize((size_t) n_blocks * 2);
    e.qs.resize((size_t) n_blocks * 32);
    uint16_t ds[2];
    int8_t qs[32];
    for (unsigned b = 0; b < n_blocks; ++b) {
        host_q8_1(&h[(size_t) b * 32], ds, qs);
        e.ds[b * 2] = ds[0];
        e.ds[b * 2 + 1] = ds[1];
        std::memcpy(&e.qs[(size_t) b * 32], qs, 32);
    }
    return e;
}

}  // namespace

int main(int argc, char** argv) {
    int want_plat = -1;
    int repeat = 50;
    int mul = 1;
    int pos = 0;
    for (int i = 1; i < argc; ++i) {
        if (argv[i][0] == '-') want_plat = std::atoi(argv[i]);
        else if (pos++ == 0) repeat = std::atoi(argv[i]);
        else mul = std::atoi(argv[i]);
    }
    if (repeat < 1) repeat = 1;

    const unsigned n_ff = 640;                     // expert_feed_forward_length for this model
    const unsigned rows = 64u * (unsigned) mul;    // expert-token pairs, e.g. 64 experts x 1 token
    const unsigned n_values = rows * n_ff;
    const unsigned n_blocks = n_values / 32u;

    // deterministic pseudo-random gate/up values in the range a real SwiGLU sees
    std::vector<float> gu((size_t) rows * 2 * n_ff);
    uint32_t st = 0x12345678u;
    auto rnd = [&st]() { st = st * 1664525u + 1013904223u; return (float) ((st >> 8) & 0xFFFFFF) / 8388608.0f - 1.0f; };
    for (auto& v : gu) v = rnd() * 6.0f;

    // host activations, plus the blocks that exist to break things
    std::vector<float> hv(n_values);
    for (unsigned r = 0; r < rows; ++r)
        for (unsigned c = 0; c < n_ff; ++c)
            hv[(size_t) r * n_ff + c] = host_swiglu(gu[(size_t) r * 2 * n_ff + c],
                                                    gu[(size_t) r * 2 * n_ff + n_ff + c]);
    // The fused kernel rebuilds these from gu, so it never sees the planted blocks below; keep the
    // unpatched array as its reference or the comparison blames exp() for the host's own edits.
    const std::vector<float> hv_plain = hv;
    if (n_values >= 128) {
        for (unsigned i = 0; i < 32; ++i) hv[i] = 0.0f;                             // all-zero block -> d = 0
        for (unsigned i = 32; i < 64; ++i) hv[i] = (i & 1) ? 1e6f : -1e6f;          // d and sum overflow half
        hv[64] = 254.0f; for (unsigned i = 65; i < 96; ++i) hv[i] = 1.0f;           // amax 254 -> d = 2.0, q = round(0.5)
        hv[96] = -254.0f; for (unsigned i = 97; i < 128; ++i) hv[i] = -1.0f;        // the same with the sign flipped
    }
    std::printf("shape: %u expert-token rows x n_ff %u = %u q8_1 blocks\n", rows, n_ff, n_blocks);
    std::printf("planted blocks: 0 all-zero, 1 +-1e6 overflow, 2 amax=254 (q=round(0.5) tie), 3 as 2 negated\n");

    cl_uint n_plat = 0;
    clGetPlatformIDs(0, nullptr, &n_plat);
    std::vector<cl_platform_id> plats(n_plat);
    clGetPlatformIDs(n_plat, plats.data(), nullptr);
    cl_platform_id plat = nullptr;
    for (cl_uint i = 0; i < n_plat; ++i) {
        size_t n = 0;
        clGetPlatformInfo(plats[i], CL_PLATFORM_VENDOR, 0, nullptr, &n);
        std::vector<char> v(n + 1, 0);
        clGetPlatformInfo(plats[i], CL_PLATFORM_VENDOR, n, v.data(), nullptr);
        if (want_plat < 0 ? std::strstr(v.data(), "Advanced Micro") != nullptr : (int) i == want_plat)
            plat = plats[i];
    }
    if (!plat) { std::printf("no matching platform\n"); return 1; }

    cl_uint nd = 0;
    clGetDeviceIDs(plat, CL_DEVICE_TYPE_ALL, 0, nullptr, &nd);
    std::vector<cl_device_id> devs(nd);
    clGetDeviceIDs(plat, CL_DEVICE_TYPE_ALL, nd, devs.data(), nullptr);
    cl_device_id dev = devs[0];
    {
        size_t n = 0;
        clGetDeviceInfo(dev, CL_DEVICE_NAME, 0, nullptr, &n);
        std::vector<char> v(n + 1, 0);
        clGetDeviceInfo(dev, CL_DEVICE_NAME, n, v.data(), nullptr);
        std::printf("device: %s\n", v.data());
    }

    cl_int rc = CL_SUCCESS;
    cl_context ctx = clCreateContext(nullptr, 1, &dev, nullptr, nullptr, &rc);
    cl_command_queue q = clCreateCommandQueue(ctx, dev, CL_QUEUE_PROFILING_ENABLE, &rc);

    cl_mem bgu = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                gu.size() * 4, gu.data(), &rc);
    cl_mem bh = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                               hv.size() * 4, hv.data(), &rc);
    cl_mem bds = clCreateBuffer(ctx, CL_MEM_WRITE_ONLY, (size_t) n_blocks * 2 * 4, nullptr, &rc);
    cl_mem bqs = clCreateBuffer(ctx, CL_MEM_WRITE_ONLY, (size_t) n_blocks * 32, nullptr, &rc);
    if (rc != CL_SUCCESS) { std::printf("buffers: %s\n", err_name(rc)); return 1; }

    std::vector<float> gpu_ds((size_t) n_blocks * 2);
    std::vector<int8_t> gpu_qs((size_t) n_blocks * 32);

    // the host model of the quantiser, over each of the two activation arrays
    const Expect exp_q = host_expect(hv, n_blocks);         // section 1: the planted blocks included
    const Expect exp_sw = host_expect(hv_plain, n_blocks);  // section 2: what the fused kernel should see

    const size_t g[1] = {(size_t) n_blocks * 32u}, l[1] = {32};

    auto run = [&](int mode_i, int expo, bool fused, const Expect& E, float& ms_out, int& qs_bad,
                   int& qs_near, int& qs_far, int& ds_bad) {
        const Mode& m = kModes[mode_i];
        const std::string src = source(m.r, expo, fused);
        const char* srcp = src.c_str();
        cl_int crc = CL_SUCCESS;
        cl_program p = clCreateProgramWithSource(ctx, 1, &srcp, nullptr, &crc);
        crc = clBuildProgram(p, 1, &dev, "-cl-std=CL2.0", nullptr, nullptr);
        if (crc != CL_SUCCESS) {
            size_t n = 0;
            clGetProgramBuildInfo(p, dev, CL_PROGRAM_BUILD_LOG, 0, nullptr, &n);
            std::vector<char> log(n + 1, 0);
            clGetProgramBuildInfo(p, dev, CL_PROGRAM_BUILD_LOG, n, log.data(), nullptr);
            std::printf("      build failed (%s): %.400s\n", err_name(crc), log.data());
            clReleaseProgram(p);
            return false;
        }
        cl_kernel k = clCreateKernel(p, fused ? "swiglu_q8_1" : "q8_1_from_h", &crc);
        if (crc != CL_SUCCESS) {
            std::printf("      clCreateKernel: %s\n", err_name(crc));
            clReleaseProgram(p);
            return false;
        }
        cl_uint ai = 0;
        if (fused) {
            clSetKernelArg(k, ai++, sizeof(bgu), &bgu);
            clSetKernelArg(k, ai++, sizeof(bds), &bds);
            clSetKernelArg(k, ai++, sizeof(bqs), &bqs);
            clSetKernelArg(k, ai++, sizeof(n_ff), &n_ff);
            clSetKernelArg(k, ai++, sizeof(n_values), &n_values);
        } else {
            clSetKernelArg(k, ai++, sizeof(bh), &bh);
            clSetKernelArg(k, ai++, sizeof(bds), &bds);
            clSetKernelArg(k, ai++, sizeof(bqs), &bqs);
            clSetKernelArg(k, ai++, sizeof(n_values), &n_values);
        }
        if (m.local) {
            clSetKernelArg(k, ai++, 32 * sizeof(float), nullptr);
            clSetKernelArg(k, ai++, 32 * sizeof(float), nullptr);
        }
        const cl_int e1 = clEnqueueNDRangeKernel(q, k, 1, nullptr, g, l, 0, nullptr, nullptr);
        const cl_int e2 = clFinish(q);
        if (e1 != CL_SUCCESS || e2 != CL_SUCCESS) {
            std::printf("      enqueue failed (%s / %s)\n", err_name(e1), err_name(e2));
            clReleaseKernel(k);
            clReleaseProgram(p);
            return false;
        }
        clEnqueueReadBuffer(q, bds, CL_TRUE, 0, (size_t) n_blocks * 2 * 4, gpu_ds.data(), 0, nullptr, nullptr);
        clEnqueueReadBuffer(q, bqs, CL_TRUE, 0, (size_t) n_blocks * 32, gpu_qs.data(), 0, nullptr, nullptr);

        qs_bad = qs_near = qs_far = ds_bad = 0;
        for (size_t i = 0; i < gpu_qs.size(); ++i) {
            const int d = (int) gpu_qs[i] - (int) E.qs[i];
            if (d == 0) continue;
            ++qs_bad;
            if (d == 1 || d == -1) ++qs_near; else ++qs_far;
        }
        for (size_t i = 0; i < (size_t) n_blocks * 2; ++i)
            if (f32_to_f16(gpu_ds[i]) != E.ds[i]) ++ds_bad;

        for (int i = 0; i < 3; ++i) clEnqueueNDRangeKernel(q, k, 1, nullptr, g, l, 0, nullptr, nullptr);
        clFinish(q);
        const auto t0 = std::chrono::steady_clock::now();
        for (int i = 0; i < repeat; ++i) clEnqueueNDRangeKernel(q, k, 1, nullptr, g, l, 0, nullptr, nullptr);
        clFinish(q);
        const auto t1 = std::chrono::steady_clock::now();
        ms_out = (float) (std::chrono::duration<double, std::milli>(t1 - t0).count() / repeat);

        clReleaseKernel(k);
        clReleaseProgram(p);
        return true;
    };

    // ---- 1. the quantiser alone: this must be bit exact --------------------------------------------
    std::printf("\n[1] q8_1_from_h - the reduction, the quantiser and the store, checked bit for bit\n");
    std::printf("%-30s %14s %10s %10s %10s\n", "reduction", "qs mismatched", "ds mismatched", "ms", "Mval/s");
    for (int mi = 0; mi < kNumModes; ++mi) {
        float ms = 0.0f;
        int bad = 0, near = 0, far = 0, dsb = 0;
        if (!run(mi, 0, false, exp_q, ms, bad, near, far, dsb)) {
            std::printf("%-30s %14s\n", kModes[mi].name, "unusable");
            continue;
        }
        char qsb[48];
        std::snprintf(qsb, sizeof(qsb), "%d/%zu", bad, gpu_qs.size());
        char dsb_s[48];
        std::snprintf(dsb_s, sizeof(dsb_s), "%d/%zu", dsb, (size_t) n_blocks * 2);
        std::printf("%-30s %14s %10s %10.4f %10.1f%s\n", kModes[mi].name, qsb, dsb_s, ms,
                    (double) n_values / (ms * 1e-3) / 1e6, bad == 0 && dsb == 0 ? "   EXACT" : "");

        // where the first scale differs, since a reduction-order effect has to land on one block
        for (size_t i = 0; i < (size_t) n_blocks * 2; ++i)
            if (f32_to_f16(gpu_ds[i]) != exp_q.ds[i]) {
                std::printf("      first ds mismatch: block %zu %s  host %04x  gpu %04x\n", i / 2,
                            (i & 1) ? "sum" : "d", exp_q.ds[i], f32_to_f16(gpu_ds[i]));
                break;
            }

        // first mismatching block, so the fault is not guessed at
        if (mi == 0 && (bad || dsb)) {
            for (unsigned b = 0; b < n_blocks && b < 6; ++b) {
                uint16_t ds[2]; int8_t qs[32];
                host_q8_1(&hv[(size_t) b * 32], ds, qs);
                float amp[32], smp[32];
                for (int i = 0; i < 32; ++i) { amp[i] = std::fabs(hv[b * 32 + i]); smp[i] = hv[b * 32 + i]; }
                for (int o = 16; o > 0; o >>= 1)
                    for (int i = 0; i < 32; ++i)
                        if (!(i & o)) { amp[i] = std::fmax(amp[i], amp[i ^ o]); smp[i] = smp[i] + smp[i ^ o]; }
                const float hd = q8_1_finite(amp[0] / 127.0f), hs = q8_1_finite(smp[0]);
                std::printf("    block %u h[0..3]=%.6g %.6g %.6g %.6g\n", b, hv[b * 32], hv[b * 32 + 1],
                            hv[b * 32 + 2], hv[b * 32 + 3]);
                std::printf("      host  d=%.9g (%04x) sum=%.9g (%04x)  amax=%.9g\n", (double) hd, ds[0],
                            (double) hs, ds[1], (double) amp[0]);
                std::printf("      gpu   d=%.9g (%04x) sum=%.9g (%04x)\n", (double) gpu_ds[b * 2],
                            f32_to_f16(gpu_ds[b * 2]), (double) gpu_ds[b * 2 + 1], f32_to_f16(gpu_ds[b * 2 + 1]));
                std::printf("      host qs:");
                for (int i = 0; i < 8; ++i) std::printf(" %d", (int) qs[i]);
                std::printf("\n      gpu  qs:");
                for (int i = 0; i < 8; ++i) std::printf(" %d", (int) gpu_qs[(size_t) b * 32 + i]);
                std::printf("\n");
            }
        }
    }

    // ---- 2. the fused SwiGLU: the only inexactness left is the driver's exp() ----------------------
    std::printf("\n[2] swiglu_q8_1 - fused; the host model uses libm exp, so codes may differ by 1\n");
    std::printf("%-30s %6s %22s %14s %10s %10s\n", "reduction", "exp", "qs off by 1 / more", "ds mismatched", "ms", "Mval/s");
    for (int mi = 0; mi < kNumModes; ++mi) {
        for (int expo = 0; expo < 2; ++expo) {
            float ms = 0.0f;
            int bad = 0, near = 0, far = 0, dsb = 0;
            if (!run(mi, expo, true, exp_sw, ms, bad, near, far, dsb)) {
                std::printf("%-30s %6s %22s\n", kModes[mi].name, expo ? "native" : "exp", "unusable");
                continue;
            }
            char nb[48];
            std::snprintf(nb, sizeof(nb), "%d / %d", near, far);
            char dsb_s[48];
            std::snprintf(dsb_s, sizeof(dsb_s), "%d/%zu", dsb, (size_t) n_blocks * 2);
            std::printf("%-30s %6s %22s %14s %10.4f %10.1f%s\n", kModes[mi].name, expo ? "native" : "exp",
                        nb, dsb_s, ms, (double) n_values / (ms * 1e-3) / 1e6,
                        (bad == 0 && dsb == 0) ? "   EXACT" : "");
        }
    }

    // ---- 3. what one expert's activation pass costs -------------------------------------------------
    // One expert-token row is n_ff = 640 values = 20 q8_1 blocks.  A real decode step offloads ~192
    // (expert, token) pairs of activations per token, but the same 640-wide row shape, so the per-row
    // cost below is the number that goes into the budget.
    std::printf("\n[3] cost per expert-token row (n_ff = 640, 20 q8_1 blocks)\n");
    {
        float ms = 0.0f;
        int bad = 0, near = 0, far = 0, dsb = 0;
        if (run(0, 1, true, exp_sw, ms, bad, near, far, dsb)) {
            const double per_row_us = ms * 1e3 / rows;
            std::printf("    xor butterfly, native_exp: %.4f ms for %u rows = %.3f us/row\n", ms, rows, per_row_us);
            std::printf("    192 offloaded rows -> %.2f ms/token, %.2f ms/token for 96 (half the misses)\n",
                        per_row_us * 192.0 / 1e3, per_row_us * 96.0 / 1e3);
        }
    }
    return 0;
}
