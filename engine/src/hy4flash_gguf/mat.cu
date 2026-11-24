// src/hy4flash_gguf/mat.cu - see include/strata/hy4flash_gguf/mat.hpp.
#include "strata/hy4flash_gguf/mat.hpp"
#include "strata/kernels/bf16_gemv.hpp"

#include "strata/kernels/native_mmvq.hpp"
#include "strata/prefill/gemm.hpp"
#include "mmq/moe_mmq.hpp"

#if defined(__HIPCC__)
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#else
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#endif

#include <cublas_v2.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>
#include <stdexcept>

namespace MMQ = strata::hy4_flash::mmq;

namespace strata::hy4_flash {
namespace { thread_local bool decode_equivalent = false; }
bool mat_decode_equivalent() { return decode_equivalent; }
void mat_set_decode_equivalent(bool enabled) { decode_equivalent = enabled; }
namespace {

constexpr int kBlock = 256;

/// F32 -> FP16, the whole activation in one launch. `__float2half` is round-to-nearest-even, which is what
/// `Gemm::native` expects of an FP16 activation (it is what the main engine's own writers produce).
__global__ __launch_bounds__(kBlock) void to_f16_k(const float* __restrict__ x, __half* __restrict__ y,
                                                   long long n) {
    for (long long i = blockIdx.x * (long long) blockDim.x + threadIdx.x; i < n;
         i += (long long) gridDim.x * blockDim.x)
        y[i] = __float2half(x[i]);
}

/// Store MMQ bounds for one group covering all activation rows.
__global__ void set_bounds_k(int* b, int rows) {
    b[0] = 0;
    b[1] = rows;
}

unsigned grid_for(long long n) {
    const long long g = (n + kBlock - 1) / kBlock;
    return (unsigned) (g < 1 ? 1 : (g > 65535 ? 65535 : g));
}

}  // namespace

/// The definition the header keeps opaque.
struct MatCtx {
    // Keep device workspaces when switching compute streams.
    std::unique_ptr<strata::prefill::Gemm> gemm;
    bool strict_router = false;
    void* cublas = nullptr;            ///< our own handle for the router's SGEMM (Gemm keeps its own private)
    void* stream = nullptr;
    std::uint16_t* f16 = nullptr;      ///< the activation arena, bump-allocated
    std::size_t f16_elems = 0;
    std::size_t f16_used = 0;
    std::uint16_t* wscratch = nullptr; ///< Gemm's dequantisation scratch (owned here, not by Gemm::init)
    std::int64_t wscratch_elems = 0;
    void* workspace = nullptr;
    std::size_t ws_bytes = 0;
    std::unique_ptr<MMQ::Context> mctx;   ///< llama.cpp's MMQ keeps a stream-k fixup pool; one per context
    void* q8 = nullptr;                   ///< the quantised activation arena, bump-allocated like `f16`
    std::size_t q8_cap = 0, q8_used = 0;
    std::int32_t* ident = nullptr;        ///< iota(chunk_max): MMQ's dst row map, the identity here
    std::int32_t* dbounds = nullptr;      ///< {0, T}: one "expert" covering every row
    int ident_rows = 0;
    int bounds_rows = -1;              ///< what `dbounds` currently says, so it is written once a chunk
    /// Cache one quantized image per staged activation for reuse across projections.
    struct Q8Image {
        std::uint64_t stamp = 0;
        std::int64_t n = 0;
        int type = -1;
        void* p = nullptr;
    };
    std::uint64_t stamp = 0;           ///< counted up at every staging; 0 is "not staged", never cached
    static constexpr int kQ8Cache = 16;
    Q8Image q8_cache[kQ8Cache];
    int q8_cached = 0;
    long q8_reused = 0;
    long dense_mmq_taken = 0, dense_mmq_declined = 0;
    // The shape the Gemm would need, kept so it can be built on the first fallback instead of up front.
    int chunk_max = 0;
    std::int64_t widest_row = 0, want_weight_elems = 0;
    bool gemm_deferred = false;        ///< the arena, the scratch and the handle are not allocated yet
    bool ok = false;
    long overflow = 0;                 ///< preps that did not fit the arena (reported once)
    bool overflow_said = false;

    ~MatCtx() {
        if (f16 != nullptr) cudaFree(f16);
        if (wscratch != nullptr) cudaFree(wscratch);
        if (workspace != nullptr) cudaFree(workspace);
        if (cublas != nullptr) cublasDestroy((cublasHandle_t) cublas);
        if (q8 != nullptr) cudaFree(q8);
        if (ident != nullptr) cudaFree(ident);
        if (dbounds != nullptr) cudaFree(dbounds);
    }
};

namespace {

/// The FP16 arena, the weight dequantisation scratch, the cuBLAS workspace and the Gemm over them. Split
/// out of `mat_ctx` because it is now called from two places: up front when MMQ cannot cover the pass, and
/// on the first matrix that declines MMQ when it can.
bool build_gemm(MatCtx* mc, void* stream, std::string& err);

/// STRATA_HY4_FLASH_DENSE_MMQ=0 turns it off; on by default where it is available. Read in three places now -
/// the allocation decision, the staging decision and the product itself - so it is declared here.
bool dense_mmq_on();

/// One per thread. A forward pass is serial within an engine and the context survives chunks on purpose:
/// `Gemm::init_external` builds a cuBLAS handle, and the dequantisation scratch is tens of MB.
std::unique_ptr<MatCtx>& ctx_slot() {
    static thread_local std::unique_ptr<MatCtx> c;
    return c;
}

}  // namespace

namespace {

bool build_gemm(MatCtx* mc, void* stream, std::string& err) {
    constexpr int kImages = 8;
    mc->f16_elems = (std::size_t) mc->widest_row * (std::size_t) mc->chunk_max * kImages;
    if (cudaMalloc(&mc->f16, mc->f16_elems * sizeof(std::uint16_t)) != cudaSuccess) {
        cudaGetLastError();
        err = "mat_ctx: " + std::to_string(mc->f16_elems * 2 / (1u << 20)) +
              " MiB for the FP16 activation arena";
        mc->f16_elems = 0;
        return false;
    }
    mc->wscratch_elems = mc->want_weight_elems;
    if (cudaMalloc(&mc->wscratch, (std::size_t) mc->wscratch_elems * sizeof(std::uint16_t)) !=
        cudaSuccess) {
        cudaGetLastError();
        err = "mat_ctx: " + std::to_string((std::size_t) mc->wscratch_elems * 2 / (1u << 20)) +
              " MiB for the weight dequantisation scratch";
        mc->wscratch_elems = 0;
        return false;
    }
    // cuBLAS's own workspace. `init_external` takes a caller-owned one so nothing in this path allocates
    // on the token path later.
    mc->ws_bytes = 32u << 20;
    if (cudaMalloc(&mc->workspace, mc->ws_bytes) != cudaSuccess) {
        cudaGetLastError();
        err = "mat_ctx: 32 MiB for the cuBLAS workspace";
        mc->ws_bytes = 0;
        return false;
    }
    std::string gerr;
    mc->gemm = std::make_unique<strata::prefill::Gemm>();
    if (!mc->gemm->init_external(stream, mc->wscratch, mc->wscratch_elems, mc->workspace, mc->ws_bytes,
                                 gerr)) {
        err = "mat_ctx: prefill::Gemm::init_external: " + gerr;
        mc->gemm.reset();
        return false;
    }
    mc->gemm_deferred = false;
    return true;
}

}  // namespace

MatCtx* mat_ctx(int chunk_max, std::int64_t widest_row, std::int64_t weight_elems, void* stream,
                std::string& err) {
    std::unique_ptr<MatCtx>& slot = ctx_slot();
    const std::size_t want_f16 = (std::size_t) widest_row * (std::size_t) chunk_max;
    if (slot != nullptr && slot->ok && slot->gemm_deferred && slot->chunk_max >= chunk_max &&
        slot->widest_row >= widest_row && slot->want_weight_elems >= weight_elems) {
        // Nothing to rebuild: the deferred context owns no stream-bound object except the router's handle,
        // which `mat_router_f32` re-binds per call.
        slot->stream = stream;
        return slot.get();
    }
    if (slot != nullptr && slot->ok && !slot->gemm_deferred && slot->f16_elems >= want_f16 &&
        slot->wscratch_elems >= weight_elems) {
        if (slot->stream == stream) return slot.get();
        // Destroy the old GEMM handle before rebinding the buffers to another stream.
        MatCtx* mc = slot.get();
        mc->gemm.reset();
        auto g = std::make_unique<strata::prefill::Gemm>();
        std::string gerr;
        if (!g->init_external(stream, mc->wscratch, mc->wscratch_elems, mc->workspace, mc->ws_bytes, gerr)) {
            err = "mat_ctx: prefill::Gemm::init_external on a new stream: " + gerr;
            slot.reset();   // the context is unusable now; the caller falls back to the matvecs
            return nullptr;
        }
        mc->gemm = std::move(g);
        mc->stream = stream;
        mc->f16_used = 0;
        return mc;
    }
    if (chunk_max <= gemm_t() || widest_row <= 0 || weight_elems <= 0 || stream == nullptr) {
        err = "mat_ctx: a stream, a chunk above the GEMM threshold and positive extents";
        return nullptr;
    }
    slot.reset(new MatCtx());
    MatCtx* mc = slot.get();
    mc->stream = stream;
    mc->chunk_max = chunk_max;
    mc->widest_row = widest_row;
    mc->want_weight_elems = weight_elems;

    // Determine MMQ coverage before allocating fallback workspace.
    if (MMQ::built() && dense_mmq_on()) {
        constexpr int kQ8Images = 6;
        mc->q8_cap = kQ8Images * MMQ::q8_bytes((std::int64_t) chunk_max, widest_row);
        const bool have = cudaMalloc(&mc->q8, mc->q8_cap) == cudaSuccess &&
                          cudaMalloc(&mc->ident, (std::size_t) chunk_max * sizeof(std::int32_t)) ==
                              cudaSuccess &&
                          cudaMalloc(&mc->dbounds, 2 * sizeof(std::int32_t)) == cudaSuccess;
        if (!have) {
            cudaGetLastError();
            if (mc->q8 != nullptr) { cudaFree(mc->q8); mc->q8 = nullptr; }
            if (mc->ident != nullptr) { cudaFree(mc->ident); mc->ident = nullptr; }
            mc->q8_cap = 0;
        } else {
            MMQ::iota(mc->ident, (std::int64_t) chunk_max, stream);
            mc->ident_rows = chunk_max;
            mc->mctx = std::make_unique<MMQ::Context>();
        }
    }

    // Defer GEMM initialization while MMQ covers all projections.
    if (mc->q8 != nullptr) {
        mc->gemm_deferred = true;
        mc->ok = true;
        std::fprintf(stderr,
                     "hy4_flash: the prompt's projections on int8 MMQ above %d columns (%.0f MiB of "
                     "quantised images; the FP16 arena and the dequantisation scratch are not allocated "
                     "unless a matrix declines MMQ)\n",
                     gemm_t(), (double) mc->q8_cap / (1024.0 * 1024.0));
        return mc;
    }
    if (!build_gemm(mc, stream, err)) {
        slot.reset();
        return nullptr;
    }
    mc->ok = true;
    std::fprintf(stderr, "hy4_flash: the prompt's projections on tensor-core GEMMs above %d columns "
                         "(%.0f MiB of activation images, %.0f MiB of dequantisation scratch)\n",
                 gemm_t(), (double) (mc->f16_elems * 2) / (1024.0 * 1024.0),
                 (double) ((std::size_t) weight_elems * 2) / (1024.0 * 1024.0));
    return mc;
}

void mat_ctx_release() { ctx_slot().reset(); }

std::size_t mat_ctx_owned_bytes() {
    const auto* m=ctx_slot().get();
    if(!m)return 0;
    return m->q8_cap+(m->ident?(std::size_t)m->ident_rows*4:0)+(m->dbounds?8:0)+
        m->f16_elems*2+(std::size_t)m->wscratch_elems*2+m->ws_bytes;
}

std::size_t mat_ctx_plan_bytes(int tokens, std::int64_t widest_row, std::int64_t weight_elems) {
    if(tokens<=gemm_t())return 0;
    const std::size_t q8=MMQ::built()&&dense_mmq_on()
        ?6*MMQ::q8_bytes(tokens,widest_row)+(std::size_t)tokens*4+8:0;
    return q8+(std::size_t)tokens*widest_row*8*2+(std::size_t)weight_elems*2+(32u<<20);
}

void mat_pass_begin(MatCtx* mc) {
    if (mc != nullptr) mc->f16_used = 0;
    if (mc != nullptr) mc->q8_used = 0;
    if (mc != nullptr) mc->q8_cached = 0;
}

int gemm_t() {
    static const int t = [] {
        const char* v = std::getenv("STRATA_HY4_FLASH_GEMM_T");
        if (v == nullptr) return kGemmT;
        const int want = std::atoi(v);
        return want < 0 ? kGemmT : (want > kGemmT ? kGemmT : want);
    }();
    return t;
}

/// STRATA_HY4_FLASH_GEMM_SITES, a hex or decimal mask over MatSite. All sites by default.
unsigned gemm_sites() {
    static const unsigned m = [] {
        const char* v = std::getenv("STRATA_HY4_FLASH_GEMM_SITES");
        if (v == nullptr) return (unsigned) kSiteAll;
        return (unsigned) std::strtoul(v, nullptr, 0);
    }();
    return m;
}

bool mat_gemm_path(const MatCtx* mc, int T, unsigned site) {
    return mc != nullptr && mc->ok && T > gemm_t() && (site & gemm_sites()) != 0;
}

namespace {

/// The FP16 activation image uses token-major rows in the pass arena.
const std::uint16_t* prep(MatCtx* mc, const float* x, std::int64_t n, int T, void* stream) {
    if (!mat_gemm_path(mc, T) || x == nullptr || n <= 0) return nullptr;
    if (stream != mc->stream) return nullptr;   // see mat_apply
    // Allocate the fallback context only when a matrix declines MMQ.
    if (mc->gemm_deferred) {
        std::string gerr;
        if (!build_gemm(mc, stream, gerr)) {
            if (!mc->overflow_said) {
                mc->overflow_said = true;
                std::fprintf(stderr, "hy4_flash: a matrix declined MMQ and the dequantise + cuBLAS "
                                     "fallback could not be built (%s); it takes the matvec, which "
                                     "refuses more than 8 columns\n", gerr.c_str());
            }
            return nullptr;
        }
        std::fprintf(stderr, "hy4_flash: a matrix declined MMQ, so the FP16 arena (%.0f MiB) and the "
                             "dequantisation scratch (%.0f MiB) were allocated after all\n",
                     (double) (mc->f16_elems * 2) / (1024.0 * 1024.0),
                     (double) ((std::size_t) mc->wscratch_elems * 2) / (1024.0 * 1024.0));
    }
    const std::size_t want = (std::size_t) n * (std::size_t) T;
    if (mc->f16_used + want > mc->f16_elems) {
        // Fall back to matvec if this matrix cannot use the GEMM workspace.
        if (!mc->overflow_said) {
            mc->overflow_said = true;
            std::fprintf(stderr, "hy4_flash: the FP16 activation arena is full (%zu of %zu elements); those "
                                 "projections take the matvec path\n", mc->f16_used, mc->f16_elems);
        }
        ++mc->overflow;
        return nullptr;
    }
    std::uint16_t* at = mc->f16 + mc->f16_used;
    mc->f16_used += want;
    to_f16_k<<<grid_for((long long) want), kBlock, 0, static_cast<cudaStream_t>(stream)>>>(
        x, reinterpret_cast<__half*>(at), (long long) want);
    return at;
}

/// STRATA_HY4_FLASH_DENSE_MMQ=0 turns it off; on by default where it is available. Read in two places, so it
/// lives here rather than beside `dense_mmq`.
bool dense_mmq_on() {
    static const bool v = [] {
        const char* e = std::getenv("STRATA_HY4_FLASH_DENSE_MMQ");
        return e == nullptr || *e != '0';
    }();
    return v;
}

/// True when the FP16 activation image has to be built UP FRONT, because there is no MMQ for `mat_apply`
/// to try first: without MMQ a deferred image would be built on the next line anyway, and the eager form
/// keeps the code path a caller on another stream takes exactly as it was.
bool eager_f16(const MatCtx* mc) {
    return mc == nullptr || mc->q8 == nullptr || mc->mctx == nullptr || !dense_mmq_on();
}

}  // namespace

MatIn mat_stage(MatCtx* mc, const float* x, void* q8, std::int64_t n, int T, void* stream, unsigned site) {
    kernels::native_quantize_q8_1_wide(x, q8, (int) n, T, stream);
    return mat_staged(mc, x, q8, n, T, stream, site);
}

MatIn mat_staged(MatCtx* mc, const float* x, const void* q8, std::int64_t n, int T, void* stream,
                 unsigned site) {
    MatIn in;
    in.mc = mc;
    in.q8 = q8;
    in.f32 = x; // BF16 projections consume the original activation, including decode.
    // Only on the GEMM path and only on this context's stream, for exactly the reasons the FP16 image is:
    // a consumer on another stream would read what was written on this one with nothing ordering the two.
    if (mat_gemm_path(mc, T, site) && stream == mc->stream) {
        in.f32 = x;
        in.stamp = ++mc->stamp;
    }
    // Build an FP16 image only for backends that read it.
    in.f16 = (mat_gemm_path(mc, T, site) && eager_f16(mc)) ? prep(mc, x, n, T, stream) : nullptr;
    in.n = n;
    return in;
}

namespace {

void dense_mmq_say(const char* why) {
    static bool said = false;
    if (said) return;
    said = true;
    std::fprintf(stderr, "hy4_flash: some prompt projections stay on dequantise + cuBLAS - %s\n", why);
}

/// Run a dense projection through quantized MMQ.
bool dense_mmq(MatCtx* mc, int ggml_type, const void* w, const float* x, float* y, std::int64_t n_in,
               std::int64_t n_out, int T, std::uint64_t stamp, void* stream) {
    if (mc == nullptr || mc->q8 == nullptr || mc->mctx == nullptr || !dense_mmq_on()) return false;
    if (x == nullptr || stamp == 0 || T > mc->ident_rows) return false;
    if (n_in % 256 != 0) {
        dense_mmq_say("a width that is not a multiple of 256 (MMQ would read past the row)");
        ++mc->dense_mmq_declined;
        return false;
    }
    if (!MMQ::supported(ggml_type) || !MMQ::fits(ggml_type, n_out)) {
        dense_mmq_say("a weight type or row count llama.cpp's MMQ has no tile for "
                      "(K-quants need STRATA_MMQ_KQ=1 and the DSO beside the executable)");
        ++mc->dense_mmq_declined;
        return false;
    }
    // ---- the image this matrix reads, made once per (activation, width, type) in a block.
    void* xq = nullptr;
    for (int i = 0; i < mc->q8_cached; ++i)
        if (mc->q8_cache[i].stamp == stamp && mc->q8_cache[i].n == n_in &&
            mc->q8_cache[i].type == ggml_type) {
            xq = mc->q8_cache[i].p;
            ++mc->q8_reused;
            break;
        }
    const bool fresh = xq == nullptr;
    if (fresh) {
        const std::size_t want = MMQ::q8_bytes(T, n_in);
        if (mc->q8_used + want > mc->q8_cap) {
            dense_mmq_say("the quantised activation arena is full for this block "
                          "(STRATA_HY4_FLASH_DENSE_MMQ=0 for the dequantise + cuBLAS path)");
            ++mc->dense_mmq_declined;
            return false;
        }
        xq = static_cast<char*>(mc->q8) + mc->q8_used;
        mc->q8_used += want;
        if (mc->q8_cached < MatCtx::kQ8Cache)
            mc->q8_cache[mc->q8_cached++] = MatCtx::Q8Image{stamp, n_in, ggml_type, xq};
    }
    // Once per column count, not once per site: the last chunk of a prompt is shorter than the rest, and
    // every site within a chunk has the same T.
    if (mc->bounds_rows != T) {
        set_bounds_k<<<1, 1, 0, static_cast<cudaStream_t>(stream)>>>(mc->dbounds, T);
        mc->bounds_rows = T;
    }

    // The activation is [n_in, T] with a token's values contiguous, i.e. [T, n_in] row-major with n_in
    // floats a row - the same reading `Gemm::native` is given, so no transpose here either.
    if (fresh) MMQ::quantize(x, nullptr, xq, ggml_type, n_in, n_in, T, stream);

    MMQ::Product p;
    p.w = w;
    p.type = ggml_type;
    p.w_rows = n_out;
    p.w_cols = n_in;
    p.expert_bytes = MMQ::matrix_bytes(ggml_type, n_out, n_in);
    p.n = 1;                 ///< one "expert": the whole matrix, every row of the activation
    p.xq = xq;
    p.bounds = mc->dbounds;  ///< {0, T}
    p.ids = mc->ident;       ///< dst row = row, i.e. [T, n_out] in order
    p.total_rows = T;
    p.max_rows = T;
    p.dst = y;
    p.ld_dst = n_out;
    mc->mctx->run(p, stream);
    ++mc->dense_mmq_taken;
    return true;
}

}  // namespace

void mat_apply(const MatIn& x, int ggml_type, const void* w, float* y, std::int64_t n_in,
               std::int64_t n_out, int T, void* stream, unsigned site) {
    if (decode_equivalent && T>1 && T<=8 && ggml_type!=30 && ggml_type!=0) {
        static const bool batch=[] {const char* value=std::getenv("STRATA_HY4_FLASH_VERIFY_BATCH_DENSE");return value&&std::atoi(value)!=0;}();
        if(batch) {
            const bool old=kernels::native_mmvq_multi_exact();
            kernels::native_mmvq_set_multi_exact(true);
            kernels::native_mmvq(ggml_type,w,x.q8,y,(int)n_in,(int)n_out,T,stream);
            kernels::native_mmvq_set_multi_exact(old);
            return;
        }
        const std::size_t stride=kernels::native_q8_1_bytes((int)n_in,1);
        for(int i=0;i<T;++i)
            kernels::native_mmvq(ggml_type,w,static_cast<const char*>(x.q8)+i*stride,
                                y+(std::size_t)i*n_out,(int)n_in,(int)n_out,1,stream);
        return;
    }
    if (ggml_type == 0) {
        if (!x.f32) throw std::invalid_argument("F32 projection needs its F32 activation");
        if (!mat_router_f32(x.mc, static_cast<const float*>(w), x.f32, y, n_in, n_out, T, stream))
            mat_f32(static_cast<const float*>(w), x.f32, y, (int)n_in, (int)n_out, T, stream);
        return;
    }
    if (ggml_type == 30) {
        if (!x.f32) throw std::invalid_argument("BF16 projection needs its F32 activation");
        kernels::bf16_gemv_fp32_mmvf_cols(x.f32, static_cast<const uint16_t*>(w), y, n_in, n_out, T, stream);
        return;
    }
    // Keep short decode projections on MMVQ.
    const bool gemm_ok = mat_gemm_path(x.mc, T, site) && stream == x.mc->stream;
    if (!gemm_ok || (x.f16 == nullptr && x.f32 == nullptr)) {
        kernels::native_mmvq(ggml_type, w, x.q8, y, (int) n_in, (int) n_out, T, stream);
        return;
    }
    // ---- MMQ FIRST, when it covers this matrix: the weight stays quantised and nothing is dequantised.
    // Reached only past the guard above, so every condition that protects decode protects this too.
    if (dense_mmq(x.mc, ggml_type, w, x.f32, y, n_in, n_out, T, x.stamp, stream)) return;

    // The token-major FP16 rows match the GEMM input layout without a transpose.
    const std::uint16_t* f16 = x.f16 != nullptr ? x.f16 : prep(x.mc, x.f32, n_in, T, stream);
    if (f16 == nullptr) {
        kernels::native_mmvq(ggml_type, w, x.q8, y, (int) n_in, (int) n_out, T, stream);
        return;
    }
    x.mc->gemm->native(f16, ggml_type, w, y, T, n_out, n_in);
}

bool mat_ctx_require_prefill(MatCtx* mc, void* stream, std::string& err) {
    if(!mc || !mc->ok || mc->stream!=stream){err="planned prefill: matrix context unavailable";return false;}
    if(MMQ::built() && dense_mmq_on() && !mc->q8){err="planned prefill: Q8 workspace allocation failed";return false;}
    if(!mc->cublas){
        cublasHandle_t h=nullptr;
        auto rc=cublasCreate(&h);
        if(rc==CUBLAS_STATUS_SUCCESS)rc=cublasSetMathMode(h,CUBLAS_PEDANTIC_MATH);
        if(rc!=CUBLAS_STATUS_SUCCESS){
            if(h)cublasDestroy(h);
            err="planned prefill: router backend initialization failed, status="+std::to_string((int)rc);
            return false;
        }
        mc->cublas=h;
    }
    mc->strict_router=true;
    return true;
}

bool mat_router_f32(MatCtx* mc, const float* w, const float* x, float* y, std::int64_t n_in,
                    std::int64_t n_out, int T, void* stream) {
    // The same three conditions as the GEMM path, for the same reasons: no context, too few columns (one
    // token reads the weight once either way, and decode must keep the kernel it was measured on), or
    // another stream than the one this context's handle is bound to.
    if (mc == nullptr || !mc->ok || T <= gemm_t() || stream != mc->stream) return false;
    if (w == nullptr || x == nullptr || y == nullptr || n_in <= 0 || n_out <= 0) return false;

    if (mc->cublas == nullptr) {
        cublasHandle_t h = nullptr;
        if (cublasCreate(&h) != CUBLAS_STATUS_SUCCESS) return false;
        // PEDANTIC MATH, i.e. no TF32 substitution behind our back: the default on Hopper would let cuBLAS
        // run an FP32 GEMM on TF32 tensor cores, which is the precision the comment in mat.hpp declines.
        if (cublasSetMathMode(h, CUBLAS_PEDANTIC_MATH) != CUBLAS_STATUS_SUCCESS) {
            cublasDestroy(h);
            return false;
        }
        mc->cublas = h;
        std::fprintf(stderr, "hy4_flash: the router's product on cuBLAS SGEMM above %d columns\n", gemm_t());
    }
    cublasHandle_t h = (cublasHandle_t) mc->cublas;
    if (cublasSetStream(h, static_cast<cudaStream_t>(stream)) != CUBLAS_STATUS_SUCCESS) {
        if(mc->strict_router)throw std::runtime_error("planned prefill: router stream setup failed");
        return false;
    }

    // Express the activation and weight layouts in cuBLAS column-major coordinates.
    const float alpha = 1.0f, beta = 0.0f;
    const cublasStatus_t rc =
        cublasSgemm(h, CUBLAS_OP_T, CUBLAS_OP_N, (int) n_out, T, (int) n_in, &alpha, w, (int) n_in, x,
                    (int) n_in, &beta, y, (int) n_out);
    if(rc!=CUBLAS_STATUS_SUCCESS && mc->strict_router)
        throw std::runtime_error("planned prefill: router SGEMM failed, status="+std::to_string((int)rc));
    return rc == CUBLAS_STATUS_SUCCESS;
}

}  // namespace strata::hy4_flash
