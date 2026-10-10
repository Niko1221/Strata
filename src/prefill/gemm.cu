// src/prefill/gemm.cu - see include/strata/prefill/gemm.hpp.
#include "strata/prefill/gemm.hpp"
#include "strata/kernels/dequant_bf16.hpp"
#if defined(STRATA_PREFILL_MMQ) && defined(__HIPCC__)
#include "strata/prefill/moe_mmq.hpp"
#endif

#ifdef STRATA_USE_HIP
#include "wmma_gemm.h"
#endif

#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <atomic>

#if defined(__HIPCC__) && defined(STRATA_HIPBLASLT_AVAILABLE)
// The HIP compatibility shim maps CUDA shuffle spellings to Strata helpers.
// hipBLASLt's public headers declare native HIP shuffle functions, so keep
// those declarations from being macro-expanded in this translation unit.
#undef __shfl_xor_sync
#undef __shfl_down_sync
#undef __shfl_up_sync
#undef __shfl_sync
#undef __ballot_sync
#endif

#include <algorithm>
#include <climits>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>

#if defined(__HIPCC__) && defined(STRATA_HIPBLASLT_AVAILABLE)
#include "hipblaslt_tuning.hpp"
#include <hip/hip_runtime_api.h>
#include <hipblaslt/hipblaslt.h>
#include <hipblaslt/hipblaslt-ext.hpp>
#include <map>
#include <set>
#include <tuple>
#endif

#if defined(__HIPCC__)
#include <hip/hip_fp16.h>
#endif

namespace strata::prefill {
namespace {
#if defined(__HIPCC__)
// Y's rows, written by an FP16-out GEMM as FP16 at the start of each FP32 row (ldc = 2 ldy halves), widened in place.
// Float c overwrites halves 2c and 2c+1, so a row is walked from its end in blocks: a block's halves are read into
// registers, the block syncs, then writes its floats - which only cover halves of blocks already read.
// STRATA_DBG_NAN: the FP16 outputs that are not finite (an FP32-accumulated sum past 65504 becomes inf in FP16).
__device__ unsigned long long g_f16_nonfinite = 0;
__global__ void widen_rows_f16(float* __restrict__ Y, int64_t n, int64_t ldy, int count_nonfinite) {
    float* y = Y + (int64_t) blockIdx.x * ldy;
    const __half* h = reinterpret_cast<const __half*>(y);
    const int64_t nb = (n + blockDim.x - 1) / blockDim.x;
    for (int64_t b = nb - 1; b >= 0; --b) {
        const int64_t c = b * blockDim.x + threadIdx.x;
        const float v = c < n ? __half2float(h[c]) : 0.0f;
        if (count_nonfinite && c < n && !isfinite(v)) atomicAdd(&g_f16_nonfinite, 1ull);
        __syncthreads();
        if (c < n) y[c] = v;
        __syncthreads();
    }
}
// BF16 weight rows -> FP16, saturated (bench/results/2026-10-04-rdna2-fp16-prompt: none leaves FP16's range)
__global__ void bf16_to_f16_rows(const uint16_t* __restrict__ s, __half* __restrict__ d, int64_t n) {
    for (int64_t i = blockIdx.x * (int64_t) blockDim.x + threadIdx.x; i < n; i += (int64_t) gridDim.x * blockDim.x) {
        const float f = __uint_as_float((uint32_t) s[i] << 16);
        d[i] = __float2half(isnan(f) ? f : fminf(fmaxf(f, -65504.0f), 65504.0f));   // as hf_sat: a NaN stays NaN
    }
}
#endif

void ck(cublasStatus_t s, const char* what) {
    if (s != CUBLAS_STATUS_SUCCESS) {
        std::fprintf(stderr, "prefill gemm: %s: cuBLAS status %d\n", what, (int) s);
        std::exit(1);
    }
}

// #247/#325: on Windows (seen on gfx1201), hipBLAS can return success with the correct BF16/FP16 product for some
// shapes (hc up once T >= 96, the router) and still leave hipErrorInvalidValue set, which the next kernel's error
// check turns into an exit. The multiply has finished, so that one stale error is cleared after a GEMM that succeeded;
// any other error still stops the engine. Windows only: on Linux a stale hipErrorInvalidValue is a real error from
// an earlier call and keeps being reported. A no-op everywhere else (CUDA compiles none of it).
#if defined(__HIPCC__) && defined(_WIN32)
void absorb_hipblas_sticky(const char* what) {
    const hipError_t sticky = hipGetLastError();
    if (sticky == hipSuccess || sticky == hipErrorInvalidValue) return;
    std::fprintf(stderr, "prefill gemm: %s left %s\n", what, hipGetErrorString(sticky));
    std::exit(1);
}
#define STRATA_ABSORB_HIPBLAS_STICKY(what) absorb_hipblas_sticky(what)
#else
#define STRATA_ABSORB_HIPBLAS_STICKY(what) ((void) 0)
#endif

// A setup call whose failure the engine survives (the handle keeps its defaults), as before #240 - but said.
void note(cublasStatus_t s, const char* what) {
    if (s != CUBLAS_STATUS_SUCCESS) std::fprintf(stderr, "prefill gemm: %s: cuBLAS status %d (continuing)\n", what, (int) s);
}

#if defined(__HIPCC__) && defined(STRATA_HIPBLASLT_AVAILABLE)
struct HipLtCallKey {
    strata::prefill::hipblaslt::InputType type;
    int t;
    int n;
    int k;
    int ldy;
    uint32_t beta_bits;

    bool operator<(const HipLtCallKey& other) const {
        return std::tie(type, n, k, ldy, t, beta_bits) <
               std::tie(other.type, other.n, other.k, other.ldy, other.t, other.beta_bits);
    }
};

struct HipLtCachedAlgo {
    bool supported = false;
    hipblasLtMatmulAlgo_t algo{};
    size_t workspace_bytes = 0;
};

struct HipLtState {
    hipblasLtHandle_t handle = nullptr;
    void* workspace = nullptr;
    size_t workspace_bytes = 0;
    strata::prefill::hipblaslt::TuningTable table;
    std::map<HipLtCallKey, HipLtCachedAlgo> cache;
    uint64_t lt_launches = 0;
    uint64_t fallbacks = 0;
    std::set<std::tuple<strata::prefill::hipblaslt::InputType, int, int, int, int>> fallback_shapes;

    ~HipLtState() {
        if (std::getenv("STRATA_HIPBLASLT_VERBOSE")) {
            std::fprintf(stderr, "prefill gemm: hipBLASLt summary launches=%llu fallbacks=%llu unique_fallback_shapes=%zu\n",
                         (unsigned long long) lt_launches, (unsigned long long) fallbacks, fallback_shapes.size());
            for (const auto& shape : fallback_shapes) {
                const auto type = std::get<0>(shape);
                std::fprintf(stderr, "prefill gemm: fallback shape dtype=%s T=%d N=%d K=%d ldy=%d\n",
                             type == strata::prefill::hipblaslt::InputType::bf16 ? "bf16" : "f16",
                             std::get<1>(shape), std::get<2>(shape), std::get<3>(shape), std::get<4>(shape));
            }
        }
        if (handle) hipblasLtDestroy(handle);
    }
};

struct HipLtDescriptors {
    hipblasLtMatmulDesc_t op = nullptr;
    hipblasLtMatrixLayout_t a = nullptr;
    hipblasLtMatrixLayout_t b = nullptr;
    hipblasLtMatrixLayout_t c = nullptr;

    ~HipLtDescriptors() {
        if (op) hipblasLtMatmulDescDestroy(op);
        if (a) hipblasLtMatrixLayoutDestroy(a);
        if (b) hipblasLtMatrixLayoutDestroy(b);
        if (c) hipblasLtMatrixLayoutDestroy(c);
    }

    bool init(hipDataType type, int t, int n, int k, int ldy, int ldb = 0) {
        const hipblasOperation_t trans_a = HIPBLAS_OP_T;
        const hipblasOperation_t trans_b = HIPBLAS_OP_N;
        if (hipblasLtMatmulDescCreate(&op, HIPBLAS_COMPUTE_32F, HIP_R_32F) != HIPBLAS_STATUS_SUCCESS ||
            hipblasLtMatmulDescSetAttribute(op, HIPBLASLT_MATMUL_DESC_TRANSA, &trans_a, sizeof(trans_a)) !=
                HIPBLAS_STATUS_SUCCESS ||
            hipblasLtMatmulDescSetAttribute(op, HIPBLASLT_MATMUL_DESC_TRANSB, &trans_b, sizeof(trans_b)) !=
                HIPBLAS_STATUS_SUCCESS ||
            hipblasLtMatrixLayoutCreate(&a, type, k, n, k) != HIPBLAS_STATUS_SUCCESS ||
            hipblasLtMatrixLayoutCreate(&b, type, k, t, ldb > k ? ldb : k) != HIPBLAS_STATUS_SUCCESS ||
            hipblasLtMatrixLayoutCreate(&c, HIP_R_32F, n, t, ldy) != HIPBLAS_STATUS_SUCCESS) {
            return false;
        }
        return true;
    }
};

std::unique_ptr<HipLtState> create_hipblaslt_state(void* workspace, size_t workspace_bytes) {
    const char* path = std::getenv("STRATA_HIPBLASLT_TUNING");
    if (!path || !*path) return nullptr;

    auto state = std::make_unique<HipLtState>();
    state->workspace = workspace;
    state->workspace_bytes = workspace_bytes;
    if (hipblasLtCreate(&state->handle) != HIPBLAS_STATUS_SUCCESS) {
        std::fprintf(stderr, "prefill gemm: hipBLASLt handle creation failed; using hipBLASEx\n");
        return nullptr;
    }

    int version = 0;
    if (hipblasLtGetVersion(state->handle, &version) != HIPBLAS_STATUS_SUCCESS) {
        std::fprintf(stderr, "prefill gemm: hipBLASLt version query failed; using hipBLASEx\n");
        return nullptr;
    }
    int device = 0;
    hipDeviceProp_t properties{};
    if (hipGetDevice(&device) != hipSuccess || hipGetDeviceProperties(&properties, device) != hipSuccess) {
        std::fprintf(stderr, "prefill gemm: HIP device query failed; using hipBLASEx\n");
        return nullptr;
    }
    std::string arch(properties.gcnArchName);
    const auto suffix = arch.find(':');
    if (suffix != std::string::npos) arch.resize(suffix);

    std::string error;
    if (!state->table.load(path, arch, version, error)) {
        std::fprintf(stderr, "prefill gemm: %s; using hipBLASEx\n", error.c_str());
        return nullptr;
    }
    std::fprintf(stderr, "prefill gemm: hipBLASLt tuning enabled (%zu rows, %s, version %d)\n",
                 state->table.rows().size(), arch.c_str(), version);
    return state;
}

HipLtCachedAlgo resolve_hipblaslt_algo(HipLtState& state, strata::prefill::hipblaslt::InputType type, int t,
                                       int n, int k, int ldy, float beta) {
    uint32_t beta_bits = 0;
    static_assert(sizeof(beta_bits) == sizeof(beta));
    std::memcpy(&beta_bits, &beta, sizeof(beta));
    const HipLtCallKey key{type, t, n, k, ldy, beta_bits};
    const auto cached = state.cache.find(key);
    if (cached != state.cache.end()) return cached->second;

    HipLtCachedAlgo resolved;
    const bool verbose = std::getenv("STRATA_HIPBLASLT_VERBOSE") != nullptr;
    const auto* row = state.table.closest(type, n, k, ldy, t);
    if (!row) {
        if (verbose) {
            std::fprintf(stderr, "prefill gemm: Lt fallback; no calibration for dtype=%s T=%d N=%d K=%d ldy=%d\n",
                         type == strata::prefill::hipblaslt::InputType::bf16 ? "bf16" : "f16", t, n, k, ldy);
        }
        return state.cache.emplace(key, resolved).first->second;
    }

    HipLtDescriptors desc;
    const hipDataType input_type = type == strata::prefill::hipblaslt::InputType::bf16 ? HIP_R_16BF : HIP_R_16F;
    if (!desc.init(input_type, t, n, k, ldy)) {
        return state.cache.emplace(key, resolved).first->second;
    }

    std::vector<int> solution_ids{row->solution_id};
    std::vector<hipblasLtMatmulHeuristicResult_t> candidates;
    if (hipblaslt_ext::getAlgosFromIndex(state.handle, solution_ids, candidates) != HIPBLAS_STATUS_SUCCESS ||
        candidates.empty() || candidates.front().state != HIPBLAS_STATUS_SUCCESS ||
        hipblaslt_ext::getIndexFromAlgo(candidates.front().algo) != row->solution_id) {
        if (verbose) {
            std::fprintf(stderr, "prefill gemm: Lt fallback; solution %d unavailable for T=%d N=%d K=%d ldy=%d\n",
                         row->solution_id, t, n, k, ldy);
        }
        return state.cache.emplace(key, resolved).first->second;
    }

    const float alpha = 1.0f;
    size_t required_workspace = 0;
    auto algo = candidates.front().algo;
    if (hipblaslt_ext::matmulIsAlgoSupported(state.handle, desc.op, &alpha, desc.a, desc.b, &beta, desc.c, desc.c,
                                             algo, required_workspace) != HIPBLAS_STATUS_SUCCESS) {
        if (verbose) {
            std::fprintf(stderr, "prefill gemm: Lt fallback; solution %d rejects actual T=%d N=%d K=%d ldy=%d beta=%.9g\n",
                         row->solution_id, t, n, k, ldy, beta);
        }
        return state.cache.emplace(key, resolved).first->second;
    }

    resolved.supported = true;
    resolved.algo = algo;
    resolved.workspace_bytes = required_workspace;
    if (verbose) {
        std::fprintf(stderr,
                     "prefill gemm: Lt solution=%d dtype=%s T=%d N=%d K=%d ldy=%d beta=%.9g workspace=%zu\n",
                     row->solution_id, type == strata::prefill::hipblaslt::InputType::bf16 ? "bf16" : "f16", t, n,
                     k, ldy, beta, required_workspace);
    }
    return state.cache.emplace(key, resolved).first->second;
}

// the hipBLASLt solution index the tuning table gives this call (-1: none / unsupported)
int hipblaslt_solution_for(void* opaque_state, strata::prefill::hipblaslt::InputType type, int64_t t, int64_t n, int64_t k,
                           int64_t ldy, float beta) {
    auto* state = static_cast<HipLtState*>(opaque_state);
    if (!state || t <= 0 || n <= 0 || k <= 0 || t > INT_MAX || n > INT_MAX || k > INT_MAX || ldy > INT_MAX || ldy < n) return -1;
    const auto resolved = resolve_hipblaslt_algo(*state, type, (int) t, (int) n, (int) k, (int) ldy, beta);
    if (!resolved.supported || resolved.workspace_bytes > state->workspace_bytes) return -1;
    auto algo = resolved.algo;
    return hipblaslt_ext::getIndexFromAlgo(algo);
}

bool try_hipblaslt(void* opaque_state, strata::prefill::hipblaslt::InputType type, const uint16_t* x,
                   const uint16_t* w, float* y, int64_t t, int64_t n, int64_t k, int64_t ldy, float beta,
                   void* stream, int64_t ldx = 0) {
    auto* state = static_cast<HipLtState*>(opaque_state);
    if (!state || t <= 0 || n <= 0 || k <= 0 || t > INT_MAX || n > INT_MAX || k > INT_MAX || ldy > INT_MAX ||
        ldy < n) {
        return false;
    }
    const auto resolved = resolve_hipblaslt_algo(*state, type, (int) t, (int) n, (int) k, (int) ldy, beta);
    if (!resolved.supported) {
        ++state->fallbacks;
        state->fallback_shapes.emplace(type, (int) t, (int) n, (int) k, (int) ldy);
        return false;
    }
    if (resolved.workspace_bytes > state->workspace_bytes) {
        ++state->fallbacks;
        state->fallback_shapes.emplace(type, (int) t, (int) n, (int) k, (int) ldy);
        if (std::getenv("STRATA_HIPBLASLT_VERBOSE")) {
            std::fprintf(stderr, "prefill gemm: Lt fallback; solution needs %zu workspace bytes, have %zu\n",
                         resolved.workspace_bytes, state->workspace_bytes);
        }
        return false;
    }

    HipLtDescriptors desc;
    const hipDataType input_type = type == strata::prefill::hipblaslt::InputType::bf16 ? HIP_R_16BF : HIP_R_16F;
    if (!desc.init(input_type, (int) t, (int) n, (int) k, (int) ldy, (int) ldx)) return false;
    const float alpha = 1.0f;
    const hipblasStatus_t status = hipblasLtMatmul(state->handle, desc.op, &alpha, w, desc.a, x, desc.b, &beta, y,
                                                   desc.c, y, desc.c, &resolved.algo, state->workspace,
                                                   state->workspace_bytes, (hipStream_t) stream);
    if (status == HIPBLAS_STATUS_SUCCESS) {
        ++state->lt_launches;
        return true;
    }

    std::fprintf(stderr, "prefill gemm: hipBLASLt launch failed with status %d\n", (int) status);
    if (beta != 0.0f) {
        std::fprintf(stderr, "prefill gemm: refusing a fallback after hipBLASLt failed with nonzero beta\n");
        std::exit(1);
    }
    auto* mutable_state = static_cast<HipLtState*>(opaque_state);
    uint32_t beta_bits = 0;
    std::memcpy(&beta_bits, &beta, sizeof(beta_bits));
    auto cached = mutable_state->cache.find(HipLtCallKey{type, (int) t, (int) n, (int) k, (int) ldy, beta_bits});
    if (cached != mutable_state->cache.end()) cached->second.supported = false;
    ++mutable_state->fallbacks;
    mutable_state->fallback_shapes.emplace(type, (int) t, (int) n, (int) k, (int) ldy);
    return false;
}
#endif

}  // namespace

bool prompt_f16() {
#if defined(__HIPCC__)
    constexpr int kMaxDev = 64;
    static std::atomic<int8_t> cached[kMaxDev] = {};   // 0 unknown, 1 off, 2 on
    int dev = 0;
    if (hipGetDevice(&dev) != hipSuccess || dev < 0 || dev >= kMaxDev) { (void) hipGetLastError(); return false; }
    if (const int8_t c = cached[dev].load(std::memory_order_relaxed)) return c == 2;
    // Opt-in (#835): only STRATA_HIP_PROMPT_F16=1 turns it on.  It changes the prompt path's numbers (an FP16 rounding of
    // each 16-bit GEMM's output), so no card gets it unasked.  On gfx103x it is ~2x faster, so we say so once.
    const char* e = std::getenv("STRATA_HIP_PROMPT_F16");
    const bool on = e != nullptr && e[0] == '1';
    hipDeviceProp_t p{};
    if (hipGetDeviceProperties(&p, dev) != hipSuccess) { (void) hipGetLastError(); p.gcnArchName[0] = 0; }
    if (on) std::fprintf(stderr, "strata prefill: HIP device %d%s%s - STRATA_HIP_PROMPT_F16=1: the prompt's 16-bit GEMMs run FP16 in and out "
                                 "(rocBLAS is tuned only for that on gfx103x; not bit-identical to the default)%s",
                         dev, p.gcnArchName[0] ? " " : "", p.gcnArchName, "\n");
    else if (e == nullptr && std::strncmp(p.gcnArchName, "gfx103", 6) == 0)
        std::fprintf(stderr, "strata prefill: HIP device %d %s - tip: STRATA_HIP_PROMPT_F16=1 reads long prompts about 2x faster here "
                             "(FP16 prompt GEMMs, rocBLAS is tuned only for FP16 on gfx103x; the numbers differ slightly, see docs/AMD_HIP.md)%s",
                     dev, p.gcnArchName, "\n");
    cached[dev].store(on ? 2 : 1, std::memory_order_relaxed);
    return on;
#else
    return false;
#endif
}

Gemm::~Gemm() {
#if defined(STRATA_PREFILL_MMQ) && defined(__HIPCC__)
    delete static_cast<strata::prefill::mmq::Context*>(mmq_ctx_);
    if (mmq_buf_) cudaFree(mmq_buf_);
#endif
#if defined(__HIPCC__) && defined(STRATA_HIPBLASLT_AVAILABLE)
    delete static_cast<HipLtState*>(hipblaslt_state_);
#endif
    if (handle_) cublasDestroy((cublasHandle_t) handle_);
    if (tc_w_) cudaFree(tc_w_);
    if (tc_x_) cudaFree(tc_x_);
    if (!external_) {
        if (scratch_) cudaFree(scratch_);
        if (workspace_) cudaFree(workspace_);
    }
}

bool Gemm::init_external(void* stream, uint16_t* scratch, int64_t scratch_elems, void* workspace, size_t ws_bytes,
                         std::string& err) {
    cublasHandle_t h = nullptr;
    if (const cublasStatus_t s = cublasCreate(&h); s != CUBLAS_STATUS_SUCCESS) {
        err = "prefill gemm: cublasCreate: cuBLAS status " + std::to_string((int) s);
        return false;
    }
    handle_ = h;
    stream_ = stream;
    external_ = true;
    note(cublasSetStream(h, (cudaStream_t) stream), "cublasSetStream");
    workspace_ = workspace;
    note(cublasSetWorkspace(h, workspace_, ws_bytes), "cublasSetWorkspace");
    note(cublasSetMathMode(h, CUBLAS_DEFAULT_MATH), "cublasSetMathMode");
    scratch_ = scratch;
    scratch_elems_ = scratch_elems;
#if defined(__HIPCC__) && defined(STRATA_HIPBLASLT_AVAILABLE)
    hipblaslt_state_ = create_hipblaslt_state(workspace_, ws_bytes).release();
#endif
    return true;
}

void Gemm::rebind(uint16_t* scratch, int64_t scratch_elems, void* workspace, size_t ws_bytes) {
    scratch_ = scratch;
    scratch_elems_ = scratch_elems;
    workspace_ = workspace;
    cublasSetWorkspace((cublasHandle_t) handle_, workspace_, ws_bytes);
#if defined(__HIPCC__) && defined(STRATA_HIPBLASLT_AVAILABLE)
    if (hipblaslt_state_) {
        auto* state = static_cast<HipLtState*>(hipblaslt_state_);
        state->workspace = workspace_;
        state->workspace_bytes = ws_bytes;
    }
#endif
}

bool Gemm::init(void* stream, int64_t scratch_elems, std::string& err) {
    // #240: every failure names the call and the real status, so "no VRAM" can be told from a broken install
    cublasHandle_t h = nullptr;
    if (const cublasStatus_t s = cublasCreate(&h); s != CUBLAS_STATUS_SUCCESS) {
        err = "prefill gemm: cublasCreate: cuBLAS status " + std::to_string((int) s);
        return false;
    }
    handle_ = h;
    stream_ = stream;
    note(cublasSetStream(h, (cudaStream_t) stream), "cublasSetStream");
    // A fixed workspace so the handle never allocates on the way (and graphs could capture it later).
    const size_t ws = 32u << 20;
    if (const cudaError_t e = cudaMalloc(&workspace_, ws); e != cudaSuccess) {
        err = std::string("prefill gemm: workspace of 32 MiB: ") + cudaGetErrorString(e);
        return false;
    }
    note(cublasSetWorkspace(h, workspace_, ws), "cublasSetWorkspace");
    note(cublasSetMathMode(h, CUBLAS_DEFAULT_MATH), "cublasSetMathMode");
#if defined(__HIPCC__) && defined(STRATA_HIPBLASLT_AVAILABLE)
    hipblaslt_state_ = create_hipblaslt_state(workspace_, ws).release();
#endif
    if (scratch_elems > 0) {
        if (const cudaError_t e = cudaMalloc((void**) &scratch_, (size_t) scratch_elems * 2); e != cudaSuccess) {
            err = "prefill gemm: dequant scratch of " + std::to_string(scratch_elems * 2 >> 20) + " MiB: " +
                  cudaGetErrorString(e);
            return false;
        }
    }
    scratch_elems_ = scratch_elems;
    return true;
}

#if !defined(__HIPCC__)
// ---- below sm_80: BF16 GEMMs without BF16 tensor cores (PR #655, #540, #395) ----------------------------------------
// Turing and Volta have no BF16 tensor cores: cublasGemmEx on CUDA_R_16BF inputs falls back to a SIMT fp32 kernel
// (magma_sgemmEx), a fifth of a 4K prompt on an RTX 2080 Ti (#655) and ~10% of a V100's prompt (#540).  BF16 -> FP16
// is exact for every value inside FP16's normal range (the 7-bit mantissa fits in 10 bits); weights and normalized
// activations sit there.  With the FP16 path, the weight is converted once per call and the activations in row
// slices, into the instance's own buffers, and the product runs as the FP16 GEMM (fp32 accumulate) on the tensor
// cores.  Finite values beyond FP16's range are clamped to +-65504 (#540) instead of becoming Inf.  A beta = 1
// product (STRATA_PREFILL_BF16X2's remainder, ~2^-9 of the original, deep in FP16's subnormal band) keeps cuBLAS.
//   Default: on for compute capability 7.0 - 7.4 (Volta: only the experimental STRATA_EXPERIMENTAL_SM60 build runs
//   there), OFF for 7.5 (RTX 20 in the ready-made engine: the sums round differently, so it is opt-in until it has
//   been gated).  STRATA_BF16_TC=1 / =0 turns it on / off on any 7.x card (=2, a test mode: on any card).
// Pascal (6.x) has no tensor cores and cuBLAS has no BF16 GEMM for it (#395, measured NOT_SUPPORTED on a P40): both
// operands are widened to fp32 by an exact shift and the product is cublasSgemm with the same fp32 accumulator.
namespace {
__global__ void bf16_to_f16_kernel(const uint16_t* __restrict__ in, __half* __restrict__ out, int64_t n) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float f = __uint_as_float((uint32_t) in[i] << 16);
        if (f > 65504.0f && !isinf(f)) f = 65504.0f;
        else if (f < -65504.0f && !isinf(f)) f = -65504.0f;
        out[i] = __float2half_rn(f);
    }
}
void bf16_to_f16(const uint16_t* in, uint16_t* out, int64_t n, cudaStream_t st) {
    if (n <= 0) return;
    bf16_to_f16_kernel<<<(unsigned) ((n + 255) / 256), 256, 0, st>>>(in, reinterpret_cast<__half*>(out), n);
}
#if defined(STRATA_EXPERIMENTAL_SM60)   // Pascal runs only the experimental build
__global__ void bf16_to_f32_kernel(const uint16_t* __restrict__ in, float* __restrict__ out, int64_t n) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = __uint_as_float((uint32_t) in[i] << 16);
}
void bf16_to_f32(const uint16_t* in, float* out, int64_t n, cudaStream_t st) {
    if (n <= 0) return;
    bf16_to_f32_kernel<<<(unsigned) ((n + 255) / 256), 256, 0, st>>>(in, out, n);
}
#endif
// The current device's compute capability as 10 * major + minor, per device (a layer split can mix cards); 0: not
// known, read as "not an old card" so a failed query keeps the cuBLAS BF16 call.
int current_cc() {
    static std::atomic<int> cc[64] = {};
    int dev = 0;
    if (cudaGetDevice(&dev) != cudaSuccess || dev < 0 || dev >= 64) { cudaGetLastError(); return 0; }
    int v = cc[dev].load(std::memory_order_relaxed);
    if (v == 0) {
        int major = 0, minor = 0;
        if (cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, dev) != cudaSuccess ||
            cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, dev) != cudaSuccess) {
            cudaGetLastError();
            return 0;
        }
        v = 10 * major + minor;
        cc[dev].store(v, std::memory_order_relaxed);
    }
    return v;
}
// 0: cuBLAS's BF16 GEMM, 1: through FP16 (tensor cores), 2: through FP32 (Pascal)
int bf16_path() {
    static const int forced = [] {
        const char* v = std::getenv("STRATA_BF16_TC");
        return v != nullptr && v[0] != '\0' ? std::atoi(v) : -1;
    }();
    const int cc = current_cc();
    if (forced == 2 && cc > 0) return 1;   // a test mode: the FP16 path on any card (gemm_bf16_parity on sm_80+)
    if (cc <= 0 || cc >= 80) return 0;
    if (cc < 70) return 2;
    if (forced >= 0) return forced != 0 ? 1 : 0;
    return cc < 75 ? 1 : 0;
}
bool grow(uint16_t*& p, int64_t& have, int64_t want) {   // `have`, `want`: 2-byte elements
    if (have >= want) return true;
    if (p) cudaFree(p);
    p = nullptr;
    have = 0;
    if (cudaMalloc((void**) &p, (size_t) want * 2) != cudaSuccess) { cudaGetLastError(); p = nullptr; return false; }
    have = want;
    return true;
}
constexpr int64_t kXSliceElems = 16ll << 20;   // 32 MiB of FP16 activations per slice (64 MiB as fp32)
}  // namespace
#endif
bool Gemm::bf16_hcd_exact(const uint16_t* X, int64_t ldx, const uint16_t* W, float* Y, int64_t T, int64_t N, int64_t K) {
#if defined(__HIPCC__) && defined(STRATA_HIPBLASLT_AVAILABLE)
    static std::atomic<bool> told{false};
    if (!hipblaslt_state_ || N != 320 || K != 10240 || T < 1) {
        if (N == 320 && K == 10240 && !told.exchange(true))
            std::fprintf(stderr, "strata: STRATA_HCD_EXACT: no hipBLASLt tuning table is loaded, so the exact kernel has nothing to match: "
                                 "the hyper-connection down projection stays on hipBLAS\n");
        return false;
    }
    // The kernel copies hipBLASLt solution 1176 / 1177's summation order, which is only true of the library build the
    // table was calibrated with: take it only when the table makes hipBLASLt pick one of those two for this T.
    const int id = hipblaslt_solution_for(hipblaslt_state_, strata::prefill::hipblaslt::InputType::bf16, T, N, K, N, 0.0f);
    if (id != 1176 && id != 1177) {
        if (T >= 4096 && !told.exchange(true))   // (a short chunk may legitimately get another solution: not worth a line)
            std::fprintf(stderr, "strata: STRATA_HCD_EXACT: the tuning table gives the hyper-connection down projection (N 320, K 10240) "
                                 "solution %d at T %lld, not 1176 / 1177: hipBLASLt runs it (a changed table or library; "
                                 "tools/hip/gfx1151-hipblaslt-100401.txt is the one measured)\n",
                         id, (long long) T);
        return false;
    }
    return strata_pf_hcdown_exact_bf16(X, ldx, W, Y, T, N, K, stream_);
#else
    (void) X; (void) ldx; (void) W; (void) Y; (void) T; (void) N; (void) K;
    return false;
#endif
}

void Gemm::bf16(const uint16_t* X, const uint16_t* W, float* Y, int64_t T, int64_t N, int64_t K, int64_t ldy,
                float beta, int64_t ldx) {
    if (T <= 0 || N <= 0) return;
    if (ldy <= 0) ldy = N;
    if (ldx <= K) ldx = 0;
#ifdef STRATA_USE_HIP
    // #313 (opt-in STRATA_WMMA_GEMM=1; STRATA_WMMA_BF16=0 excludes bf16): RDNA3 / RDNA3.5 WMMA dense GEMM, before
    // hipBLASLt; it returns false for shapes it does not take (and off gfx11) and this falls through
    static const bool wmma_on = [] { const char* v = std::getenv("STRATA_WMMA_GEMM"); return v && v[0] != 0 && v[0] != '0'; }();
    static const bool bf16_on = [] { const char* v = std::getenv("STRATA_WMMA_BF16"); return !v || (v[0] != 0 && v[0] != '0'); }();
    if (wmma_on && bf16_on && ldx == 0 && T >= 16 && (beta == 0.0f || beta == 1.0f) &&
        strata_wmma_gemm_bf16(X, W, Y, T, N, K, ldy, beta, stream_)) {
        return;
    }
#endif
#if defined(__HIPCC__)
    if (f16_io_) {   // X is the FP16 image (set_act_f16); W goes through the dequantization scratch as FP16
        // the only accumulating BF16 product is the BF16X2 low part, which bf16x2_mode() turns off with this path
        if (beta != 0.0f) { std::fprintf(stderr, "prefill gemm: an accumulating BF16 GEMM on the FP16 prompt path\n"); std::exit(1); }
        const int64_t rows = std::min<int64_t>(N, scratch_elems_ / K);
        if (rows <= 0) { std::fprintf(stderr, "prefill gemm: scratch too small for K=%lld\n", (long long) K); std::exit(1); }
        for (int64_t r0 = 0; r0 < N; r0 += rows) {
            const int64_t n = std::min(rows, N - r0);
            bf16_to_f16_rows<<<1024, 256, 0, (cudaStream_t) stream_>>>(W + r0 * K, (__half*) scratch_, n * K);
            f16_inplace(X, scratch_, Y + r0, T, n, K, ldy);
        }
        return;
    }
#endif
    const float alpha = 1.0f;
#if defined(__HIPCC__) && defined(STRATA_HIPBLASLT_AVAILABLE)
    if (try_hipblaslt(hipblaslt_state_, strata::prefill::hipblaslt::InputType::bf16, X, W, Y, T, N, K, ldy,
                      beta, stream_, ldx)) {
        STRATA_ABSORB_HIPBLAS_STICKY("hipBLASLt bf16");
        return;
    }
#endif
#if !defined(__HIPCC__)
#if defined(STRATA_EXPERIMENTAL_SM60)
    constexpr bool kPascalPath = true;
#else
    constexpr bool kPascalPath = false;
#endif
    if (const int path = K > 0 ? bf16_path() : 0; (path == 1 && N > 1 && beta == 0.0f) || (path == 2 && kPascalPath)) {
        // FP16 (path 1) or FP32 (path 2, Pascal) copies of the weight and of X, converted in tiles.  Issue #1757: when
        // the buffers did not fit in the VRAM that was left (a stage of a layer split keeps ~100 MiB), this fell through
        // to the raw BF16 cuBLAS call, which a Pascal card answers with CUBLAS_STATUS_INTERNAL_ERROR (14).  Now the
        // weight is converted in tiles of N rows (the whole weight when it fits, as before) and X in slices; a card with
        // no room even for a tile says so.  A single output row stays cuBLAS's GEMV for path 1 (faster than the
        // conversions).  Every tile is a disjoint block of Y, so each gets the caller's beta.
        const int mult = path == 2 ? 2 : 1;   // 2-byte elements per converted value
        int64_t w_rows = N;
        int64_t x_rows = std::max<int64_t>(1, std::min<int64_t>(T, kXSliceElems / K));
        bool fit = false;
        for (;;) {
            if (grow(tc_w_, tc_w_elems_, mult * w_rows * K) && grow(tc_x_, tc_x_elems_, mult * x_rows * K)) { fit = true; break; }
            if (w_rows <= 64 && x_rows <= 64) break;
            w_rows = std::max<int64_t>(std::min<int64_t>(w_rows, 64), w_rows / 2);
            x_rows = std::max<int64_t>(std::min<int64_t>(x_rows, 64), x_rows / 2);
        }
        if (!fit && path == 2) {   // Pascal has no BF16 cuBLAS call to fall back to; Volta keeps its raw one below
            std::fprintf(stderr, "prefill gemm: no GPU memory left for the %s conversion buffers of a %lld x %lld BF16 weight "
                                 "(Pascal cards run the BF16 products through them); free VRAM on this card: raise "
                                 "--vram-reserve-mib, lower --max-context or --kv-resident, or give it fewer layers\n",
                         "FP32", (long long) N, (long long) K);
            std::exit(1);
        }
        if (fit) {
            for (int64_t n0 = 0; n0 < N; n0 += w_rows) {
                const int64_t nn = std::min<int64_t>(w_rows, N - n0);
                if (path == 2) {
#if defined(STRATA_EXPERIMENTAL_SM60)
                    float* const wf = reinterpret_cast<float*>(tc_w_);
                    float* const xf = reinterpret_cast<float*>(tc_x_);
                    bf16_to_f32(W + n0 * K, wf, nn * K, (cudaStream_t) stream_);
                    for (int64_t t0 = 0; t0 < T; t0 += x_rows) {
                        const int64_t n = std::min<int64_t>(x_rows, T - t0);
                        bf16_to_f32(X + t0 * K, xf, n * K, (cudaStream_t) stream_);
                        ck(cublasSgemm((cublasHandle_t) handle_, CUBLAS_OP_T, CUBLAS_OP_N, (int) nn, (int) n, (int) K, &alpha,
                                       wf, (int) K, xf, (int) K, &beta, Y + t0 * ldy + n0, (int) ldy),
                           "cublasSgemm (bf16 on Pascal)");
                    }
#endif
                } else {
                    bf16_to_f16(W + n0 * K, tc_w_, nn * K, (cudaStream_t) stream_);
                    for (int64_t t0 = 0; t0 < T; t0 += x_rows) {
                        const int64_t n = std::min<int64_t>(x_rows, T - t0);
                        bf16_to_f16(X + t0 * K, tc_x_, n * K, (cudaStream_t) stream_);
                        f16(tc_x_, tc_w_, Y + t0 * ldy + n0, n, nn, K, ldy, beta);
                    }
                }
            }
            return;
        }
    }
#endif
    // Column-major view: Y^T[N, T] = W[N, K] (stored K x N col-major, transposed) . X^T[K, T].
    ck(cublasGemmEx((cublasHandle_t) handle_, CUBLAS_OP_T, CUBLAS_OP_N, (int) N, (int) T, (int) K, &alpha, W,
                    CUDA_R_16BF, (int) K, X, CUDA_R_16BF, (int) (ldx ? ldx : K), &beta, Y, CUDA_R_32F, (int) ldy,
                    CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT),
       "cublasGemmEx");
    STRATA_ABSORB_HIPBLAS_STICKY("cublasGemmEx");
}

void Gemm::f16(const uint16_t* X, const uint16_t* W, float* Y, int64_t T, int64_t N, int64_t K, int64_t ldy,
               float beta) {
    if (T <= 0 || N <= 0) return;
    if (ldy <= 0) ldy = N;
#ifdef STRATA_USE_HIP
    // #313 (opt-in STRATA_WMMA_GEMM=1): RDNA3 / RDNA3.5 WMMA dense GEMM, before hipBLASLt (falls through when false)
    static const bool wmma_on = [] { const char* v = std::getenv("STRATA_WMMA_GEMM"); return v && v[0] != 0 && v[0] != '0'; }();
    static const bool pf_on = [] { const char* v = std::getenv("STRATA_PF_GEMM"); return v && v[0] == '1'; }();
    static const int64_t pf_min_t = [] { const char* v = std::getenv("STRATA_PF_SWITCH_MIN_T"); return v ? (int64_t) std::atoll(v) : (int64_t) 0; }();
    if (pf_on && T >= pf_min_t && strata_pf_gemm_f16(X, W, Y, T, N, K, ldy, beta, stream_)) return;   // S23: opt-in
    if (wmma_on && T >= 16 && (beta == 0.0f || beta == 1.0f) &&
        strata_wmma_gemm_f16(X, W, Y, T, N, K, ldy, beta, stream_)) {
        return;
    }
#endif
#if defined(__HIPCC__)
    if (f16_io_ && beta == 0.0f) { f16_inplace(X, W, Y, T, N, K, ldy); return; }
#endif
    const float alpha = 1.0f;
#if defined(__HIPCC__) && defined(STRATA_HIPBLASLT_AVAILABLE)
    if (try_hipblaslt(hipblaslt_state_, strata::prefill::hipblaslt::InputType::f16, X, W, Y, T, N, K, ldy,
                      beta, stream_)) {
        STRATA_ABSORB_HIPBLAS_STICKY("hipBLASLt f16");
        return;
    }
#endif
#if !defined(__HIPCC__)
    const auto f16_gemm = [&](cublasGemmAlgo_t algo) {
        return cublasGemmEx((cublasHandle_t) handle_, CUBLAS_OP_T, CUBLAS_OP_N, (int) N, (int) T, (int) K, &alpha,
                            W, CUDA_R_16F, (int) K, X, CUDA_R_16F, (int) K, &beta, Y, CUDA_R_32F, (int) ldy,
                            CUBLAS_COMPUTE_32F, algo);
    };
    // Issue #1650: with a second CUDA device initialized in the process, cuBLAS 12 answers the default algorithm of a
    // few shapes - narrow ones, and the set moves with K - with CUBLAS_STATUS_INTERNAL_ERROR; a fixed algorithm keeps
    // a kernel that state does not break.  Only the shapes that fail are moved (the failing call itself is harmless);
    // the wide shapes, for which the fixed algorithms are markedly slower, stay on the default.
    const int64_t shape = ((int64_t) K << 32) | (uint32_t) N;
    cublasGemmAlgo_t algo = CUBLAS_GEMM_DEFAULT;
    for (int i = 0; i < f16_algo_n_; ++i)
        if (f16_algo_shape_[i] == shape) { algo = CUBLAS_GEMM_ALGO2; break; }
    cublasStatus_t st = f16_gemm(algo);
    if (st == CUBLAS_STATUS_INTERNAL_ERROR && algo == CUBLAS_GEMM_DEFAULT) {
        if (f16_algo_n_ < kF16AlgoSlots) f16_algo_shape_[f16_algo_n_++] = shape;
        st = f16_gemm(CUBLAS_GEMM_ALGO2);
    }
    ck(st, "cublasGemmEx f16");
#else
    ck(cublasGemmEx((cublasHandle_t) handle_, CUBLAS_OP_T, CUBLAS_OP_N, (int) N, (int) T, (int) K, &alpha, W,
                    CUDA_R_16F, (int) K, X, CUDA_R_16F, (int) K, &beta, Y, CUDA_R_32F, (int) ldy,
                    CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT),
       "cublasGemmEx f16");
#endif
    STRATA_ABSORB_HIPBLAS_STICKY("cublasGemmEx f16");
}

#if defined(STRATA_PREFILL_MMQ) && defined(__HIPCC__)
bool Gemm::native_mmq(const uint16_t* X, int type, const void* W, float* Y, int64_t T, int64_t N, int64_t K,
                      int64_t ldy) {
    namespace mmq = strata::prefill::mmq;
    constexpr int64_t kRows = 1024, kMaxK = 8192;
    static const bool enabled = [] {
        const char* e = std::getenv("STRATA_DENSE_MMQ");
        return e && e[0] == '1';
    }();
    if (!enabled || mmq_failed_ || !mmq::built() || !mmq::fits(type, N)) return false;
    // K must be a multiple of 256: llama.cpp's MMQ loads the weights in 256-value K chunks, and the
    // chunk past a partial row reads past the row (the next row's bytes - or, for the last weight
    // row, bytes past the tensor, which no caller guarantees to be zeros). The MoE path is safe
    // because its gather buffers carry a zeroed tail; a dense tensor is passed as-is, so a K that
    // is not a full chunk multiple (e.g. the 640-value shared-expert down: 20 blocks of 32) stays on
    // the dequantize + cuBLAS path.
    if (K > kMaxK || K % 256 != 0) return false;
    int64_t R = scratch_elems_ / (2 * K);   // FP32 activations for a chunk fit in the FP16 dequant scratch
    if (R > kRows) R = kRows;
    if (R > T) R = T;
    if (R < 1) return false;
    if (ldy <= 0) ldy = N;

    const auto up = [](size_t v) { return (v + 255) & ~(size_t) 255; };
    const size_t ident_off = up(mmq::q8_bytes(kRows, kMaxK));
    const size_t bounds_off = ident_off + up((size_t) kRows * 4);
    if (!mmq_buf_) {
        if (cudaMalloc(&mmq_buf_, bounds_off + 256) != cudaSuccess) {
            cudaGetLastError();
            mmq_failed_ = true;
            return false;
        }
        mmq_ctx_ = new mmq::Context();
        mmq::iota((int32_t*) ((uint8_t*) mmq_buf_ + ident_off), kRows, stream_);
        std::fprintf(stderr, "prefill gemm: dense MMQ on (STRATA_DENSE_MMQ=1)\n");
    }
    void* xq = mmq_buf_;
    const int32_t* ident = (const int32_t*) ((uint8_t*) mmq_buf_ + ident_off);
    int32_t* bounds = (int32_t*) ((uint8_t*) mmq_buf_ + bounds_off);
    float* xf = (float*) scratch_;
    auto* ctx = static_cast<mmq::Context*>(mmq_ctx_);

    for (int64_t r0 = 0; r0 < T; r0 += R) {
        const int64_t rows = (T - r0 < R) ? T - r0 : R;
        mmq::f16_to_f32(X + r0 * K, xf, rows * K, stream_);
        mmq::quantize(xf, nullptr, xq, type, K, K, rows, stream_);
        mmq::set_bounds(bounds, (int32_t) rows, stream_);
        mmq::Product p;
        p.w = W;
        p.type = type;
        p.w_rows = N;
        p.w_cols = K;
        p.expert_bytes = mmq::matrix_bytes(type, N, K);
        p.n = 1;
        p.xq = xq;
        p.bounds = bounds;
        p.ids = ident;
        p.total_rows = rows;
        p.max_rows = rows;
        p.dst = Y + r0 * ldy;
        p.ld_dst = ldy;
        ctx->run(p, stream_);
    }
    return true;
}
#endif
void Gemm::f16_inplace(const uint16_t* X, const uint16_t* W, float* Y, int64_t T, int64_t N, int64_t K, int64_t ldy) {
#if defined(__HIPCC__)
    const float one = 1.0f, zero = 0.0f;
    ck(cublasGemmEx((cublasHandle_t) handle_, CUBLAS_OP_T, CUBLAS_OP_N, (int) N, (int) T, (int) K, &one, W, CUDA_R_16F,
                    (int) K, X, CUDA_R_16F, (int) K, &zero, Y, CUDA_R_16F, (int) (2 * ldy), CUBLAS_COMPUTE_32F,
                    CUBLAS_GEMM_DEFAULT),
       "cublasGemmEx f16 out");
    static const bool dbg_nan = std::getenv("STRATA_DBG_NAN") != nullptr;
    widen_rows_f16<<<(unsigned) T, 256, 0, (cudaStream_t) stream_>>>(Y, N, ldy, dbg_nan ? 1 : 0);
    if (dbg_nan) {   // debug only: a sync per GEMM
        unsigned long long bad = 0, zero_count = 0;
        (void) hipStreamSynchronize((hipStream_t) stream_);
        if (hipMemcpyFromSymbol(&bad, HIP_SYMBOL(g_f16_nonfinite), sizeof bad) == hipSuccess && bad != 0) {
            std::fprintf(stderr, "strata prefill: STRATA_DBG_NAN: %llu non-finite FP16 GEMM outputs (T=%lld N=%lld K=%lld; FP16 ends at 65504)%s",
                         bad, (long long) T, (long long) N, (long long) K, "\n");
            (void) hipMemcpyToSymbol(HIP_SYMBOL(g_f16_nonfinite), &zero_count, sizeof zero_count);
        }
    }
#else
    (void) X; (void) W; (void) Y; (void) T; (void) N; (void) K; (void) ldy;
#endif
}

void Gemm::native(const uint16_t* X, int ggml_type, const void* W_blocks, float* Y, int64_t T, int64_t N, int64_t K,
                  int64_t ldy, float beta, int64_t ldx) {
#ifdef STRATA_USE_HIP
    // S23 (opt-in STRATA_PF_PAD=1, with STRATA_PF_GEMM=1 on chunks of STRATA_PF_SWITCH_MIN_T+ tokens): the weight
    // dequantized into the scratch with row stride K + 64 halves, so no 4 KB-multiple stride camps on the memory
    // channels (gemm_probe8: K 6144 19 -> 29 TFLOPS from W alone, 35 with X padded too).  Bitwise the same products.
    static const bool pad_on = [] { const char* v = std::getenv("STRATA_PF_PAD"); return v && v[0] == '1'; }();
    static const bool pf_on = [] { const char* v = std::getenv("STRATA_PF_GEMM"); return v && v[0] == '1'; }();
    static const int64_t pf_min_t = [] { const char* v = std::getenv("STRATA_PF_SWITCH_MIN_T"); return v ? (int64_t) std::atoll(v) : (int64_t) 0; }();
    if (pad_on && pf_on && T >= pf_min_t && T >= 64 && N >= 512 && K % 32 == 0 && N * (K + 64) <= scratch_elems_) {
        const int64_t ldw = K + 64;
        if (strata::kernels::dequant_f16_ld(ggml_type, W_blocks, 0, N, K, ldw, scratch_, stream_) &&
            strata_pf_gemm_f16_ld(X, ldx > 0 ? ldx : K, scratch_, ldw, Y, T, N, K, ldy, beta, stream_)) {
            return;
        }
    }
#endif
    if (ldx > 0 && ldx != K) {
#ifdef STRATA_USE_HIP
        // X arrived padded but the weight cannot be (an i-quant without a strided dequant, or the scratch is too
        // small for N x (K + 64)): the weight unpadded, X read at its stride - the same products, the same bits
        const int64_t rows = scratch_elems_ / K;
        bool ok = rows > 0;
        if (ldy <= 0) ldy = N;
        for (int64_t r0 = 0; ok && r0 < N; r0 += rows) {
            const int64_t n = (N - r0 < rows) ? N - r0 : rows;
            strata::kernels::dequant_f16(ggml_type, W_blocks, r0, n, K, scratch_, stream_);
            ok = strata_pf_gemm_f16_ld(X, ldx, scratch_, K, Y + r0, T, n, K, ldy, beta, stream_);
        }
        if (ok) return;
#endif
        std::fprintf(stderr, "prefill gemm: a padded X (ldx %lld, K %lld) needs STRATA_PF_PAD's path\n",
                     (long long) ldx, (long long) K);
        std::exit(1);
    }
#if defined(STRATA_PREFILL_MMQ) && defined(__HIPCC__)
    if (beta == 0.0f && (ldx == 0 || ldx == K) && native_mmq(X, ggml_type, W_blocks, Y, T, N, K, ldy)) return;
#endif
    if (N * K > scratch_elems_) {
        // Too large for the scratch at once: in row slices.
        const int64_t rows = scratch_elems_ / K;
        if (rows <= 0) { std::fprintf(stderr, "prefill gemm: scratch too small for K=%lld\n", (long long) K); std::exit(1); }
        if (ldy <= 0) ldy = N;
        for (int64_t r0 = 0; r0 < N; r0 += rows) {
            const int64_t n = (N - r0 < rows) ? N - r0 : rows;
            strata::kernels::dequant_f16(ggml_type, W_blocks, r0, n, K, scratch_, stream_);
            f16(X, scratch_, Y + r0, T, n, K, ldy, beta);
        }
        return;
    }
    strata::kernels::dequant_f16(ggml_type, W_blocks, 0, N, K, scratch_, stream_);
    f16(X, scratch_, Y, T, N, K, ldy, beta);
}

}  // namespace strata::prefill
