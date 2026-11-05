#include "strata/cann/ops.hpp"

#include <aclnn/acl_meta.h>
#include <aclnnop/aclnn_add.h>
#include <aclnnop/aclnn_matmul.h>
#include <aclnnop/aclnn_rms_norm.h>
#include <aclnnop/aclnn_silu.h>
#include <aclnnop/aclnn_mul.h>
#include <aclnnop/aclnn_softmax.h>
#include <aclnnop/aclnn_sub.h>
#include <aclnnop/aclnn_exp.h>
#include <aclnnop/aclnn_div.h>
#include <aclnnop/aclnn_cast.h>

#include <acl/acl.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <memory>
#include <random>

namespace strata::cann {
namespace {

[[noreturn]] void fail(const char * what, int code) {
    const char * recent = aclGetRecentErrMsg();
    std::string msg = std::string("cann: ") + what + " failed (status " + std::to_string(code) + ")";
    if (recent && *recent) {
        msg += ": ";
        msg += recent;
    }
    throw Error(msg, code);
}

void check_acl(aclError err, const char * what) {
    if (err != ACL_SUCCESS) {
        fail(what, (int) err);
    }
}

void check_status(aclnnStatus s, const char * what) {
    if (s != 0) {
        fail(what, (int) s);
    }
}

aclDataType to_acl(DType t) {
    return t == DType::F16 ? ACL_FLOAT16 : ACL_FLOAT;
}

size_t elem_size(DType t) {
    return t == DType::F16 ? 2 : 4;
}

// Non-owning 2D row-major ND tensor over `t.data` (aclnn wants stride {cols, 1}).
aclTensor * make_tensor_1d(const Tensor & t, int64_t n) {
    int64_t shape[1] = { n };
    int64_t stride[1] = { 1 };
    return aclCreateTensor(shape, 1, to_acl(t.dtype), stride, 0, ACL_FORMAT_ND, shape, 1, t.data);
}

aclTensor * make_tensor(const Tensor & t, int64_t rows, int64_t cols, int64_t offset_elems = 0) {
    int64_t shape[2] = { rows, cols };
    int64_t stride[2] = { cols, 1 };
    return aclCreateTensor(shape, 2, to_acl(t.dtype), stride, offset_elems, ACL_FORMAT_ND, shape, 2, t.data);
}

aclTensor * make_tensor3(const Tensor & t, int64_t rows, int64_t cols, int64_t batch, int64_t offset_elems = 0) {
    int64_t shape[3] = { batch, rows, cols };
    int64_t stride[3] = { rows * cols, cols, 1 };
    return aclCreateTensor(shape, 3, to_acl(t.dtype), stride, offset_elems, ACL_FORMAT_ND, shape, 3, t.data);
}

struct Exec {
    aclOpExecutor * exec = nullptr;
    void * workspace = nullptr;
    uint64_t size = 0;

    ~Exec() {
        if (workspace) {
            aclrtFree(workspace);
        }
    }
};

// One workspace buffer for the whole process. Allocating it per op cost milliseconds and made a
// 40-layer token look like 33 s; the ops themselves are microseconds.
void * g_ws = nullptr;
size_t g_ws_size = 0;

// Profiling: a single op measures 0.043 ms in isolation while a token takes seconds, so the cost
// has to be attributed rather than guessed at.
long long g_op_count = 0;
double g_op_seconds = 0.0;

void ensure_workspace(size_t need) {
    if (need <= g_ws_size) {
        return;
    }
    // Grow generously: a reallocation has to wait for the queue, and the ops themselves only need
    // a few hundred kilobytes. Freeing a buffer that a queued kernel still points at corrupts it.
    const size_t want = std::max<size_t>(need, 64u << 20);
    if (g_ws) {
        aclrtSynchronizeDevice();
        aclrtFree(g_ws);
        g_ws = nullptr;
        g_ws_size = 0;
    }
    check_acl(aclrtMalloc(&g_ws, want, ACL_MEM_MALLOC_HUGE_FIRST), "aclrtMalloc(workspace)");
    g_ws_size = want;
}

// Each op follows the same two-phase aclnn contract; this template keeps that boilerplate in one place.
template <typename GetWs, typename Launch>
void execute(Stream & stream, GetWs get_ws, Launch launch, const char * what) {
    uint64_t ws_size = 0;
    aclOpExecutor * exec = nullptr;
    check_status(get_ws(&ws_size, &exec), what);
    ensure_workspace((size_t) ws_size);
    void * ws = ws_size > 0 ? g_ws : nullptr;
    const auto op_t0 = std::chrono::steady_clock::now();
    check_status(launch(ws, ws_size, exec, (aclrtStream) stream.handle()), what);
    // Keep this: host uploads go through aclrtMemcpy, which is not ordered against queued kernels,
    // so without a sync per op a later upload can overwrite a buffer an earlier kernel still reads.
    // STRATA_CANN_NO_SYNC exists only to measure what the sync costs; it is not safe for real runs.
    static const bool no_sync = getenv("STRATA_CANN_NO_SYNC") != nullptr;
    if (!no_sync) {
        stream.synchronize();
    }
    g_op_count += 1;
    g_op_seconds += std::chrono::duration<double>(std::chrono::steady_clock::now() - op_t0).count();
}

} // namespace

// Profiling counters, printed when the workspace is released. This exists because a single op
// measures 0.043 ms in isolation while a token takes seconds, so the cost has to be attributed.
void print_op_stats() {
    if (g_op_count > 0 && getenv("STRATA_CANN_STATS") != nullptr) {
        printf("ops: %lld calls, %.3f s total, %.3f ms each\n", g_op_count, g_op_seconds,
               1000.0 * g_op_seconds / (double) g_op_count);
    }
}

void release_workspace() {
    print_op_stats();
    if (g_ws) {
        aclrtFree(g_ws);
        g_ws = nullptr;
        g_ws_size = 0;
    }
}

void f32_to_f16(const float * src, uint16_t * dst, size_t n) {
    for (size_t i = 0; i < n; ++i) {
        uint32_t x;
        memcpy(&x, &src[i], 4);
        uint32_t sign = (x >> 16) & 0x8000u;
        int32_t exp = (int32_t) ((x >> 23) & 0xff) - 127;
        uint32_t man = x & 0x7fffffu;
        uint16_t out;
        if (exp > 15) {
            out = (uint16_t) (sign | 0x7c00u);
        } else if (exp < -14) {
            out = (uint16_t) sign;
        } else {
            out = (uint16_t) (sign | ((uint32_t) (exp + 15) << 10) | (man >> 13));
        }
        dst[i] = out;
    }
}

void f16_to_f32(const uint16_t * src, float * dst, size_t n) {
    for (size_t i = 0; i < n; ++i) {
        uint16_t h = src[i];
        uint32_t sign = (h >> 15) & 1u;
        uint32_t exp = (h >> 10) & 0x1fu;
        uint32_t man = h & 0x3ffu;
        float v;
        if (exp == 0) {
            v = std::ldexp((float) man, -24);
        } else if (exp == 31) {
            v = man ? NAN : INFINITY;
        } else {
            v = std::ldexp((float) (man | 0x400u), (int) exp - 25);
        }
        dst[i] = sign ? -v : v;
    }
}

void matmul(Stream & stream, const Tensor & a, const Tensor & b, Tensor & out) {
    const int64_t m = a.ne1, k = a.ne0, n = b.ne0;
    if (b.ne1 != k || out.ne1 != m || out.ne0 != n) {
        throw Error("cann: matmul shape mismatch (a is [m,k], b is [k,n], out is [m,n])", -1);
    }
    aclTensor * ta = make_tensor(a, m, k);
    aclTensor * tb = make_tensor(b, k, n);
    aclTensor * tc = make_tensor(out, m, n);
    execute(
        stream,
        [&](uint64_t * ws, aclOpExecutor ** ex) {
            return aclnnMatmulGetWorkspaceSize(ta, tb, tc, 0, ws, ex);
        },
        [&](void * ws, uint64_t ws_size, aclOpExecutor * ex, aclrtStream s) {
            return aclnnMatmul(ws, ws_size, ex, s);
        },
        "aclnnMatmul");
    aclDestroyTensor(ta);
    aclDestroyTensor(tb);
    aclDestroyTensor(tc);
}

// ---- batched forms, one call for a whole head batch ----

void bmm(Stream & stream, const Tensor & a, const Tensor & c, Tensor & out) {
    const int64_t m = a.ne1, k = a.ne0, n = c.ne0, b = out.ne2;
    if (c.ne1 != k || out.ne1 != m || out.ne0 != n || a.ne2 != b || c.ne2 != b) {
        throw Error("cann: bmm shape mismatch (a is [B,m,k], b is [B,k,n], out is [B,m,n])", -1);
    }
    aclTensor * ta = make_tensor3(a, m, k, b);
    aclTensor * tb = make_tensor3(c, k, n, b);
    aclTensor * tc = make_tensor3(out, m, n, b);
    execute(
        stream,
        [&](uint64_t * ws, aclOpExecutor ** ex) { return aclnnMatmulGetWorkspaceSize(ta, tb, tc, 0, ws, ex); },
        [&](void * ws, uint64_t ws_size, aclOpExecutor * ex, aclrtStream s) {
            return aclnnMatmul(ws, ws_size, ex, s);
        },
        "aclnnMatmul(batched)");
    aclDestroyTensor(ta);
    aclDestroyTensor(tb);
    aclDestroyTensor(tc);
}

void mul_bcast(Stream & stream, const Tensor & a, const Tensor & b, Tensor & out) {
    aclTensor * ta = make_tensor3(a, a.ne1, a.ne0, a.ne2);
    aclTensor * tb = make_tensor3(b, b.ne1, b.ne0, b.ne2);
    aclTensor * tc = make_tensor3(out, out.ne1, out.ne0, out.ne2);
    execute(
        stream,
        [&](uint64_t * ws, aclOpExecutor ** ex) { return aclnnMulGetWorkspaceSize(ta, tb, tc, ws, ex); },
        [&](void * ws, uint64_t ws_size, aclOpExecutor * ex, aclrtStream s) {
            return aclnnMul(ws, ws_size, ex, s);
        },
        "aclnnMul(batched)");
    aclDestroyTensor(ta);
    aclDestroyTensor(tb);
    aclDestroyTensor(tc);
}

void add_bcast(Stream & stream, const Tensor & a, const Tensor & b, Tensor & out) {
    aclTensor * ta = make_tensor3(a, a.ne1, a.ne0, a.ne2);
    aclTensor * tb = make_tensor3(b, b.ne1, b.ne0, b.ne2);
    aclTensor * tc = make_tensor3(out, out.ne1, out.ne0, out.ne2);
    float alpha_value = 1.0f;
    aclScalar * alpha = aclCreateScalar(&alpha_value, ACL_FLOAT);
    execute(
        stream,
        [&](uint64_t * ws, aclOpExecutor ** ex) {
            return aclnnAddGetWorkspaceSize(ta, tb, alpha, tc, ws, ex);
        },
        [&](void * ws, uint64_t ws_size, aclOpExecutor * ex, aclrtStream s) {
            return aclnnAdd(ws, ws_size, ex, s);
        },
        "aclnnAdd(batched)");
    aclDestroyScalar(alpha);
    aclDestroyTensor(ta);
    aclDestroyTensor(tb);
    aclDestroyTensor(tc);
}

void sub_bcast(Stream & stream, const Tensor & a, const Tensor & b, Tensor & out) {
    aclTensor * ta = make_tensor3(a, a.ne1, a.ne0, a.ne2);
    aclTensor * tb = make_tensor3(b, b.ne1, b.ne0, b.ne2);
    aclTensor * tc = make_tensor3(out, out.ne1, out.ne0, out.ne2);
    float alpha_value = -1.0f;
    aclScalar * alpha = aclCreateScalar(&alpha_value, ACL_FLOAT);
    execute(
        stream,
        [&](uint64_t * ws, aclOpExecutor ** ex) {
            return aclnnAddGetWorkspaceSize(ta, tb, alpha, tc, ws, ex);
        },
        [&](void * ws, uint64_t ws_size, aclOpExecutor * ex, aclrtStream s) {
            return aclnnAdd(ws, ws_size, ex, s);
        },
        "aclnnAdd(sub, batched)");
    aclDestroyScalar(alpha);
    aclDestroyTensor(ta);
    aclDestroyTensor(tb);
    aclDestroyTensor(tc);
}
// out = cast(x) to `dst_type`, elementwise; the only op here whose types may differ
void cast(Stream & stream, const Tensor & x, DType dst_type, Tensor & out) {
    aclTensor * tx = x.ne2 > 1 ? make_tensor3(x, x.ne1, x.ne0, x.ne2) : make_tensor(x, x.ne1, x.ne0);
    aclTensor * ty = out.ne2 > 1 ? make_tensor3(out, out.ne1, out.ne0, out.ne2) : make_tensor(out, out.ne1, out.ne0);
    const aclDataType dt = to_acl(dst_type);
    execute(
        stream,
        [&](uint64_t * ws, aclOpExecutor ** ex) { return aclnnCastGetWorkspaceSize(tx, dt, ty, ws, ex); },
        [&](void * ws, uint64_t ws_size, aclOpExecutor * ex, aclrtStream s) {
            return aclnnCast(ws, ws_size, ex, s);
        },
        "aclnnCast");
    aclDestroyTensor(tx);
    aclDestroyTensor(ty);
}

void rms_norm(Stream & stream, const Tensor & x, const Tensor & gamma, double eps, Tensor & out) {
    aclTensor * tx = make_tensor(x, x.ne1, x.ne0);
    aclTensor * tg = make_tensor_1d(gamma, gamma.ne0);
    aclTensor * ty = make_tensor(out, out.ne1, out.ne0);

    // aclnnRmsNorm also returns the reciprocal std per row; the engine does not need it here.
    Buffer rstd((size_t) x.ne1 * sizeof(float));
    int64_t rstd_shape[1] = { x.ne1 };
    int64_t rstd_stride[1] = { 1 };
    aclTensor * trstd =
        aclCreateTensor(rstd_shape, 1, ACL_FLOAT, rstd_stride, 0, ACL_FORMAT_ND, rstd_shape, 1, rstd.data());

    execute(
        stream,
        [&](uint64_t * ws, aclOpExecutor ** ex) {
            return aclnnRmsNormGetWorkspaceSize(tx, tg, eps, ty, trstd, ws, ex);
        },
        [&](void * ws, uint64_t ws_size, aclOpExecutor * ex, aclrtStream s) {
            return aclnnRmsNorm(ws, ws_size, ex, s);
        },
        "aclnnRmsNorm");
    aclDestroyTensor(trstd);
    aclDestroyTensor(tx);
    aclDestroyTensor(tg);
    aclDestroyTensor(ty);
}

void silu(Stream & stream, const Tensor & x, Tensor & out) {
    aclTensor * tx = make_tensor(x, x.ne1, x.ne0);
    aclTensor * ty = make_tensor(out, out.ne1, out.ne0);
    execute(
        stream,
        [&](uint64_t * ws, aclOpExecutor ** ex) { return aclnnSiluGetWorkspaceSize(tx, ty, ws, ex); },
        [&](void * ws, uint64_t ws_size, aclOpExecutor * ex, aclrtStream s) {
            return aclnnSilu(ws, ws_size, ex, s);
        },
        "aclnnSilu");
    aclDestroyTensor(tx);
    aclDestroyTensor(ty);
}

void add(Stream & stream, const Tensor & a, const Tensor & b, Tensor & out) {
    aclTensor * ta = make_tensor(a, a.ne1, a.ne0);
    aclTensor * tb = make_tensor(b, b.ne1, b.ne0);
    aclTensor * tc = make_tensor(out, out.ne1, out.ne0);
    float alpha_value = 1.0f;
    aclScalar * alpha = aclCreateScalar(&alpha_value, ACL_FLOAT);
    execute(
        stream,
        [&](uint64_t * ws, aclOpExecutor ** ex) { return aclnnAddGetWorkspaceSize(ta, tb, alpha, tc, ws, ex); },
        [&](void * ws, uint64_t ws_size, aclOpExecutor * ex, aclrtStream s) {
            return aclnnAdd(ws, ws_size, ex, s);
        },
        "aclnnAdd");
    aclDestroyScalar(alpha);
    aclDestroyTensor(ta);
    aclDestroyTensor(tb);
    aclDestroyTensor(tc);
}

void softmax(Stream & stream, const Tensor & x, Tensor & out) {
    aclTensor * tx = make_tensor(x, x.ne1, x.ne0);
    aclTensor * ty = make_tensor(out, out.ne1, out.ne0);
    execute(
        stream,
        [&](uint64_t * ws, aclOpExecutor ** ex) { return aclnnSoftmaxGetWorkspaceSize(tx, 1, ty, ws, ex); },
        [&](void * ws, uint64_t ws_size, aclOpExecutor * ex, aclrtStream s) {
            return aclnnSoftmax(ws, ws_size, ex, s);
        },
        "aclnnSoftmax");
    aclDestroyTensor(tx);
    aclDestroyTensor(ty);
}


void exp(Stream & stream, const Tensor & x, Tensor & out) {
    aclTensor * tx = make_tensor(x, x.ne1, x.ne0);
    aclTensor * ty = make_tensor(out, out.ne1, out.ne0);
    execute(
        stream,
        [&](uint64_t * ws, aclOpExecutor ** ex) { return aclnnExpGetWorkspaceSize(tx, ty, ws, ex); },
        [&](void * ws, uint64_t ws_size, aclOpExecutor * ex, aclrtStream s) { return aclnnExp(ws, ws_size, ex, s); },
        "aclnnExp");
    aclDestroyTensor(tx);
    aclDestroyTensor(ty);
}

void muls(Stream & stream, const Tensor & x, float scalar, Tensor & out) {
    aclTensor * tx = make_tensor(x, x.ne1, x.ne0);
    aclTensor * ty = make_tensor(out, out.ne1, out.ne0);
    aclScalar * alpha = aclCreateScalar(&scalar, ACL_FLOAT);
    execute(
        stream,
        [&](uint64_t * ws, aclOpExecutor ** ex) { return aclnnMulsGetWorkspaceSize(tx, alpha, ty, ws, ex); },
        [&](void * ws, uint64_t ws_size, aclOpExecutor * ex, aclrtStream s) { return aclnnMuls(ws, ws_size, ex, s); },
        "aclnnMuls");
    aclDestroyScalar(alpha);
    aclDestroyTensor(tx);
    aclDestroyTensor(ty);
}

void sub(Stream & stream, const Tensor & a, const Tensor & b, Tensor & out) {
    aclTensor * ta = make_tensor(a, a.ne1, a.ne0);
    aclTensor * tb = make_tensor(b, b.ne1, b.ne0);
    aclTensor * tc = make_tensor(out, out.ne1, out.ne0);
    float alpha_value = 1.0f;
    aclScalar * alpha = aclCreateScalar(&alpha_value, ACL_FLOAT);
    execute(
        stream,
        [&](uint64_t * ws, aclOpExecutor ** ex) { return aclnnSubGetWorkspaceSize(ta, tb, alpha, tc, ws, ex); },
        [&](void * ws, uint64_t ws_size, aclOpExecutor * ex, aclrtStream s) { return aclnnSub(ws, ws_size, ex, s); },
        "aclnnSub");
    aclDestroyScalar(alpha);
    aclDestroyTensor(ta);
    aclDestroyTensor(tb);
    aclDestroyTensor(tc);
}

void mul(Stream & stream, const Tensor & a, const Tensor & b, Tensor & out) {
    aclTensor * ta = make_tensor(a, a.ne1, a.ne0);
    aclTensor * tb = make_tensor(b, b.ne1, b.ne0);
    aclTensor * tc = make_tensor(out, out.ne1, out.ne0);
    execute(
        stream,
        [&](uint64_t * ws, aclOpExecutor ** ex) { return aclnnMulGetWorkspaceSize(ta, tb, tc, ws, ex); },
        [&](void * ws, uint64_t ws_size, aclOpExecutor * ex, aclrtStream s) { return aclnnMul(ws, ws_size, ex, s); },
        "aclnnMul");
    aclDestroyTensor(ta);
    aclDestroyTensor(tb);
    aclDestroyTensor(tc);
}

void div(Stream & stream, const Tensor & a, const Tensor & b, Tensor & out) {
    aclTensor * ta = make_tensor(a, a.ne1, a.ne0);
    aclTensor * tb = make_tensor(b, b.ne1, b.ne0);
    aclTensor * tc = make_tensor(out, out.ne1, out.ne0);
    execute(
        stream,
        [&](uint64_t * ws, aclOpExecutor ** ex) { return aclnnDivGetWorkspaceSize(ta, tb, tc, ws, ex); },
        [&](void * ws, uint64_t ws_size, aclOpExecutor * ex, aclrtStream s) { return aclnnDiv(ws, ws_size, ex, s); },
        "aclnnDiv");
    aclDestroyTensor(ta);
    aclDestroyTensor(tb);
    aclDestroyTensor(tc);
}

void sigmoid(Stream & stream, const Tensor & x, Tensor & out) {
    const size_t bytes = (size_t) x.elements() * 2;
    Buffer neg(bytes), ex(bytes), den(bytes), one(bytes);
    Tensor t_neg{ neg.data(), DType::F16, x.ne0, x.ne1 };
    Tensor t_exp{ ex.data(), DType::F16, x.ne0, x.ne1 };
    Tensor t_den{ den.data(), DType::F16, x.ne0, x.ne1 };
    Tensor t_one{ one.data(), DType::F16, x.ne0, x.ne1 };
    std::vector<float> ones((size_t) x.elements(), 1.0f);
    std::vector<uint16_t> h(x.elements());
    f32_to_f16(ones.data(), h.data(), h.size());
    one.copy_from_host(h.data(), bytes);

    muls(stream, x, -1.0f, t_neg);
    exp(stream, t_neg, t_exp);
    add(stream, t_one, t_exp, t_den);
    div(stream, t_one, t_den, out);
}

void rope(Stream & stream, const Tensor & x, const Tensor & cos, const Tensor & sin, int64_t n_rot, Tensor & out) {
    const int64_t T = x.ne1;
    const int64_t head_dim = x.ne0;
    const int64_t half = n_rot / 2;
    if (head_dim % 2 != 0 || n_rot % 2 != 0 || n_rot > head_dim) {
        throw Error("cann: rope needs an even n_rot <= head_dim", -1);
    }
    const size_t half_bytes = (size_t) half * 2;

    Buffer t1(half_bytes), t2(half_bytes), t3(half_bytes);
    for (int64_t t = 0; t < T; ++t) {
        Tensor xa{ (char *) x.data + (size_t) t * head_dim * 2, DType::F16, half, 1 };
        Tensor xb{ (char *) x.data + (size_t) t * head_dim * 2 + half_bytes, DType::F16, half, 1 };
        Tensor c{ (char *) cos.data + (size_t) t * half * 2, DType::F16, half, 1 };
        Tensor s{ (char *) sin.data + (size_t) t * half * 2, DType::F16, half, 1 };
        Tensor oa{ (char *) out.data + (size_t) t * head_dim * 2, DType::F16, half, 1 };
        Tensor ob{ (char *) out.data + (size_t) t * head_dim * 2 + half_bytes, DType::F16, half, 1 };
        Tensor ta{ t1.data(), DType::F16, half, 1 };
        Tensor tb{ t2.data(), DType::F16, half, 1 };
        Tensor tc{ t3.data(), DType::F16, half, 1 };

        // out_a = x_a * cos - x_b * sin
        mul(stream, xa, c, ta);
        mul(stream, xb, s, tb);
        sub(stream, ta, tb, oa);
        // out_b = x_a * sin + x_b * cos
        mul(stream, xa, s, ta);
        mul(stream, xb, c, tb);
        add(stream, ta, tb, ob);

        if (head_dim > n_rot) {
            Tensor tail_in{ (char *) x.data + (size_t) t * head_dim * 2 + (size_t) n_rot * 2, DType::F16,
                            head_dim - n_rot, 1 };
            Tensor tail_out{ (char *) out.data + (size_t) t * head_dim * 2 + (size_t) n_rot * 2, DType::F16,
                             head_dim - n_rot, 1 };
            muls(stream, tail_in, 1.0f, tail_out);
        }
        (void) tc;
    }
}

// ---- CPU oracle ----

std::vector<float> matmul_ref(const std::vector<float> & a, const std::vector<float> & b, int64_t m, int64_t k,
                              int64_t n) {
    std::vector<float> c((size_t) m * n, 0.0f);
    for (int64_t i = 0; i < m; ++i) {
        for (int64_t j = 0; j < n; ++j) {
            float acc = 0.0f;
            for (int64_t t = 0; t < k; ++t) {
                acc += a[(size_t) i * k + t] * b[(size_t) t * n + j];
            }
            c[(size_t) i * n + j] = acc;
        }
    }
    return c;
}

std::vector<float> rms_norm_ref(const std::vector<float> & x, const std::vector<float> & gamma, int64_t rows,
                                int64_t cols, double eps) {
    std::vector<float> out(x.size());
    for (int64_t r = 0; r < rows; ++r) {
        double ss = 0.0;
        for (int64_t c = 0; c < cols; ++c) {
            double v = x[(size_t) r * cols + c];
            ss += v * v;
        }
        float scale = (float) (1.0 / std::sqrt(ss / (double) cols + eps));
        for (int64_t c = 0; c < cols; ++c) {
            out[(size_t) r * cols + c] = x[(size_t) r * cols + c] * scale * gamma[(size_t) c];
        }
    }
    return out;
}

std::vector<float> silu_ref(const std::vector<float> & x) {
    std::vector<float> out(x.size());
    for (size_t i = 0; i < x.size(); ++i) {
        out[i] = x[i] / (1.0f + std::exp(-x[i]));
    }
    return out;
}

std::vector<float> add_ref(const std::vector<float> & a, const std::vector<float> & b) {
    std::vector<float> out(a.size());
    for (size_t i = 0; i < a.size(); ++i) {
        out[i] = a[i] + b[i];
    }
    return out;
}

std::vector<float> softmax_ref(const std::vector<float> & x, int64_t rows, int64_t cols) {
    std::vector<float> out(x.size());
    for (int64_t r = 0; r < rows; ++r) {
        float mx = -INFINITY;
        for (int64_t c = 0; c < cols; ++c) {
            mx = std::max(mx, x[(size_t) r * cols + c]);
        }
        double sum = 0.0;
        for (int64_t c = 0; c < cols; ++c) {
            double e = std::exp((double) x[(size_t) r * cols + c] - mx);
            out[(size_t) r * cols + c] = (float) e;
            sum += e;
        }
        for (int64_t c = 0; c < cols; ++c) {
            out[(size_t) r * cols + c] = (float) (out[(size_t) r * cols + c] / sum);
        }
    }
    return out;
}

} // namespace strata::cann
