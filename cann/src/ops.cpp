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

#include <acl/acl.h>

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

// Each op follows the same two-phase aclnn contract; this template keeps that boilerplate in one place.
template <typename GetWs, typename Launch>
void execute(Stream & stream, GetWs get_ws, Launch launch, const char * what) {
    uint64_t ws_size = 0;
    aclOpExecutor * exec = nullptr;
    check_status(get_ws(&ws_size, &exec), what);
    void * ws = nullptr;
    if (ws_size > 0) {
        check_acl(aclrtMalloc(&ws, ws_size, ACL_MEM_MALLOC_HUGE_FIRST), "aclrtMalloc(workspace)");
    }
    aclnnStatus s = launch(ws, ws_size, exec, (aclrtStream) stream.handle());
    if (ws) {
        aclrtFree(ws);
    }
    check_status(s, what);
    stream.synchronize();
}

} // namespace

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
