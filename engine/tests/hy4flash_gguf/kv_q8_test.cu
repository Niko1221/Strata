// Compare the Q8_0 latent cache against a host reference.
#include "strata/hy4flash_gguf/attention.hpp"
#include "strata/hy4flash_gguf/kv_cache.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <vector>

namespace h = strata::hy4_flash;

namespace {

int g_fail = 0;
#define CHECK(c, ...)                                                                    \
    do {                                                                                 \
        if (!(c)) {                                                                      \
            std::printf("FAIL %s:%d: ", __FILE__, __LINE__);                             \
            std::printf(__VA_ARGS__);                                                    \
            std::printf("\n");                                                           \
            ++g_fail;                                                                    \
        }                                                                                \
    } while (0)
#define CU(x)                                                                             \
    do {                                                                                  \
        cudaError_t e_ = (x);                                                             \
        if (e_ != cudaSuccess) {                                                          \
            std::printf("CUDA %s at %s:%d\n", cudaGetErrorString(e_), __FILE__, __LINE__); \
            std::exit(2);                                                                 \
        }                                                                                 \
    } while (0)

constexpr int kLora = 512, kRot = 64, kHead = 64, kV = 128;

float h2f(__half x) { return __half2float(x); }

uint16_t bf16_bits(float f) {
    uint32_t u;
    std::memcpy(&u, &f, 4);
    u += 0x7fffu + ((u >> 16) & 1u);
    return (uint16_t) (u >> 16);
}
float bf16_val(uint16_t b) {
    uint32_t u = (uint32_t) b << 16;
    float f;
    std::memcpy(&f, &u, 4);
    return f;
}

/// One token's latent to Q8_0 the way the device does: the F16 scale first, the codes against it.
void q8_row(const float* x, int8_t* q, __half* sc) {
    for (int g = 0; g < kLora / 32; ++g) {
        float amax = 0.f;
        for (int k = 0; k < 32; ++k) amax = std::max(amax, std::fabs(x[g * 32 + k]));
        const __half dh = __float2half_rn(amax / 127.0f);
        const float d = h2f(dh);
        sc[g] = dh;
        for (int k = 0; k < 32; ++k) {
            const int v = d > 0.f ? (int) std::nearbyint(x[g * 32 + k] / d) : 0;
            q[g * 32 + k] = (int8_t) std::max(-127, std::min(127, v));
        }
    }
}

struct Data {
    int n_kv = 0, n_tok = 0;
    std::vector<float> lat, pe;        // [n_kv][kLora], [n_kv][kRot]
    std::vector<float> qa, qp, sinks;  // [kHead][n_tok][kLora], [n_tok][kHead][kRot], [kHead]
    std::vector<uint16_t> wv;          // [kHead][kV][kLora] BF16
};

Data make(int n_kv, int n_tok, uint32_t seed) {
    std::mt19937 rng(seed);
    std::normal_distribution<float> n01(0.f, 1.f);
    Data d;
    d.n_kv = n_kv;
    d.n_tok = n_tok;
    d.lat.resize((size_t) n_kv * kLora);
    d.pe.resize((size_t) n_kv * kRot);
    // An RMS-normed latent with a few outlier channels, which is the case a 32-wide absmax scale handles worst.
    for (int j = 0; j < n_kv; ++j)
        for (int c = 0; c < kLora; ++c)
            d.lat[(size_t) j * kLora + c] = n01(rng) * (c % 97 == 7 ? 8.f : 1.f);
    for (auto& v : d.pe) v = n01(rng);
    d.qa.resize((size_t) kHead * n_tok * kLora);
    d.qp.resize((size_t) n_tok * kHead * kRot);
    for (auto& v : d.qa) v = 1.2f * n01(rng);
    for (auto& v : d.qp) v = 1.2f * n01(rng);
    d.sinks.resize(kHead);
    for (auto& v : d.sinks) v = n01(rng);
    d.wv.resize((size_t) kHead * kV * kLora);
    for (auto& v : d.wv) v = bf16_bits(0.05f * n01(rng));
    return d;
}

/// The attention output in double over the given rows: out [n_tok][kHead * kV].
std::vector<double> host_attention(const Data& d, const std::vector<float>& lat, const std::vector<float>& pe,
                                   int window, float scale) {
    std::vector<double> out((size_t) d.n_tok * kHead * kV, 0.0);
    std::vector<double> s((size_t) d.n_kv), latent(kLora);
    for (int hh = 0; hh < kHead; ++hh)
        for (int i = 0; i < d.n_tok; ++i) {
            const int pos_i = d.n_kv - d.n_tok + i;
            const float* qa = &d.qa[((size_t) hh * d.n_tok + i) * kLora];
            const float* qp = &d.qp[((size_t) i * kHead + hh) * kRot];
            double m = d.sinks[hh];
            for (int j = 0; j < d.n_kv; ++j) {
                const bool masked = j > pos_i || (window > 0 && pos_i - j >= window);
                if (masked) { s[j] = -INFINITY; continue; }
                double a = 0;
                for (int c = 0; c < kLora; ++c) a += (double) qa[c] * lat[(size_t) j * kLora + c];
                for (int c = 0; c < kRot; ++c) a += (double) qp[c] * pe[(size_t) j * kRot + c];
                s[j] = a * scale;
                m = std::max(m, s[j]);
            }
            double den = std::exp(d.sinks[hh] - m);
            std::fill(latent.begin(), latent.end(), 0.0);
            for (int j = 0; j < d.n_kv; ++j) {
                if (s[j] == -INFINITY) continue;
                const double e = std::exp(s[j] - m);
                den += e;
                for (int c = 0; c < kLora; ++c) latent[c] += e * lat[(size_t) j * kLora + c];
            }
            for (int e = 0; e < kV; ++e) {
                double a = 0;
                const uint16_t* w = &d.wv[((size_t) hh * kV + e) * kLora];
                for (int c = 0; c < kLora; ++c) a += bf16_val(w[c]) * (latent[c] / den);
                out[(size_t) i * kHead * kV + hh * kV + e] = a;
            }
        }
    return out;
}

struct Err {
    double max_abs = 0, rel_rms = 0, mag = 0;
};
Err compare(const std::vector<float>& got, const std::vector<double>& ref) {
    Err e;
    double se = 0, sr = 0;
    for (size_t x = 0; x < ref.size(); ++x) {
        const double dd = (double) got[x] - ref[x];
        e.max_abs = std::max(e.max_abs, std::fabs(dd));
        e.mag = std::max(e.mag, std::fabs(ref[x]));
        se += dd * dd;
        sr += ref[x] * ref[x];
    }
    e.rel_rms = std::sqrt(se / std::max(sr, 1e-300));
    return e;
}

/// Fill a cache of `format` from `d` through the append path (`dev` picks the device-offset form, in two
/// pieces so the second lands at a non-zero offset) and run mla_attention over it.
std::vector<float> run(const Data& d, int format, bool dev, int window, float scale, h::KvCache& c) {
    std::string err;
    if (!h::kv_cache_init(c, 1, kLora, kRot, d.n_kv, format, err)) {
        std::printf("init: %s\n", err.c_str());
        std::exit(2);
    }
    cudaStream_t st;
    CU(cudaStreamCreate(&st));
    float *dl, *dp;
    int32_t* dat;
    CU(cudaMalloc(&dl, d.lat.size() * 4));
    CU(cudaMalloc(&dp, d.pe.size() * 4));
    CU(cudaMalloc(&dat, 4));
    CU(cudaMemcpy(dl, d.lat.data(), d.lat.size() * 4, cudaMemcpyHostToDevice));
    CU(cudaMemcpy(dp, d.pe.data(), d.pe.size() * 4, cudaMemcpyHostToDevice));
    const int first = d.n_kv / 3;
    const int parts[2][2] = {{0, first}, {first, d.n_kv - first}};
    for (const auto& pr : parts) {
        bool ok;
        if (dev) {
            CU(cudaMemcpyAsync(dat, &pr[0], 4, cudaMemcpyHostToDevice, st));
            ok = h::kv_cache_append_dev(c, 0, dat, dl + (size_t) pr[0] * kLora, dp + (size_t) pr[0] * kRot, pr[1],
                                        err, st);
        } else {
            ok = h::kv_cache_append(c, 0, pr[0], dl + (size_t) pr[0] * kLora, dp + (size_t) pr[0] * kRot, pr[1],
                                    err, st);
        }
        if (!ok) {
            std::printf("append: %s\n", err.c_str());
            std::exit(2);
        }
        CU(cudaStreamSynchronize(st));
    }
    c.len = d.n_kv;

    std::vector<int32_t> qpos(d.n_tok), kvpos(d.n_kv);
    for (int i = 0; i < d.n_tok; ++i) qpos[i] = d.n_kv - d.n_tok + i;
    for (int j = 0; j < d.n_kv; ++j) kvpos[j] = j;
    float *qa, *qp, *sk, *out;
    int32_t *dq, *dk;
    uint16_t* wv;
    CU(cudaMalloc(&qa, d.qa.size() * 4));
    CU(cudaMalloc(&qp, d.qp.size() * 4));
    CU(cudaMalloc(&sk, d.sinks.size() * 4));
    CU(cudaMalloc(&wv, d.wv.size() * 2));
    CU(cudaMalloc(&dq, qpos.size() * 4));
    CU(cudaMalloc(&dk, kvpos.size() * 4));
    const size_t n_out = (size_t) d.n_tok * kHead * kV;
    CU(cudaMalloc(&out, n_out * 4));
    CU(cudaMemcpy(qa, d.qa.data(), d.qa.size() * 4, cudaMemcpyHostToDevice));
    CU(cudaMemcpy(qp, d.qp.data(), d.qp.size() * 4, cudaMemcpyHostToDevice));
    CU(cudaMemcpy(sk, d.sinks.data(), d.sinks.size() * 4, cudaMemcpyHostToDevice));
    CU(cudaMemcpy(wv, d.wv.data(), d.wv.size() * 2, cudaMemcpyHostToDevice));
    CU(cudaMemcpy(dq, qpos.data(), qpos.size() * 4, cudaMemcpyHostToDevice));
    CU(cudaMemcpy(dk, kvpos.data(), kvpos.size() * 4, cudaMemcpyHostToDevice));
    const size_t sb = h::mla_attention_scratch_bytes(kLora, kV, kHead, d.n_tok, d.n_kv);
    void* scratch;
    CU(cudaMalloc(&scratch, sb));
    h::mla_attention(qa, qp, c.kv_cmpr[0], c.k_pe[0], h::kv_format(c), sk, wv, 30, kLora, kRot, kV, kHead,
                     d.n_tok, d.n_kv, dq, dk, window, scale, out, scratch, st);
    CU(cudaStreamSynchronize(st));
    std::vector<float> got(n_out);
    CU(cudaMemcpy(got.data(), out, n_out * 4, cudaMemcpyDeviceToHost));
    for (void* p : {(void*) dl, (void*) dp, (void*) dat, (void*) qa, (void*) qp, (void*) sk, (void*) wv,
                    (void*) dq, (void*) dk, (void*) out, scratch})
        cudaFree(p);
    cudaStreamDestroy(st);
    return got;
}

void check_rows(const Data& d, const h::KvCache& c, const char* how) {
    const size_t rb = h::kv_cache_lat_row_bytes(kLora, h::kKvQ8);
    std::vector<unsigned char> rows((size_t) d.n_kv * rb);
    std::vector<__half> pe((size_t) d.n_kv * kRot);
    CU(cudaMemcpy(rows.data(), c.kv_cmpr[0], rows.size(), cudaMemcpyDeviceToHost));
    CU(cudaMemcpy(pe.data(), c.k_pe[0], pe.size() * 2, cudaMemcpyDeviceToHost));
    long bad = 0;
    std::vector<unsigned char> want(rb);
    for (int j = 0; j < d.n_kv; ++j) {
        q8_row(&d.lat[(size_t) j * kLora], (int8_t*) want.data(), (__half*) (want.data() + kLora));
        if (std::memcmp(want.data(), &rows[(size_t) j * rb], rb) != 0) ++bad;
        for (int k = 0; k < kRot; ++k) {
            const __half w = __float2half_rn(d.pe[(size_t) j * kRot + k]);
            if (std::memcmp(&w, &pe[(size_t) j * kRot + k], 2) != 0) ++bad;
        }
    }
    std::printf("  append (%s): %d rows, %ld differ from the host Q8_0\n", how, d.n_kv, bad);
    CHECK(bad == 0, "append %s: %ld rows differ", how, bad);
}

}  // namespace

int main() {
    int dev = 0, major = 0;
    cudaDeviceProp prop{};
    CU(cudaGetDevice(&dev));
    CU(cudaGetDeviceProperties(&prop, dev));
    major = prop.major;
    const char* mma = std::getenv("STRATA_HY4_FLASH_ATTN_MMA");
    std::printf("kv_q8_test on %s (sm_%d%d), %s kernels\n", prop.name, prop.major, prop.minor,
                (mma != nullptr && std::atoi(mma) == 0) || major < 8 ? "scalar" : "tensor-core");
    const float scale = 0.07f;
    struct Case { int n_kv, n_tok, window; };
    const Case cases[] = {{1000, 1, 0}, {1000, 1, 512}, {4096, 1, 0}, {1031, 64, 0}, {2048, 64, 512}};
    for (const Case& k : cases) {
        const Data d = make(k.n_kv, k.n_tok, 1234u + (uint32_t) k.n_kv + (uint32_t) k.n_tok);
        std::printf("n_kv %d, %d queries, window %d\n", k.n_kv, k.n_tok, k.window);
        // The dequantised rows the Q8_0 cache holds, for the read check.
        std::vector<float> deq(d.lat.size());
        {
            std::vector<int8_t> q(kLora);
            std::vector<__half> sc(kLora / 32);
            for (int j = 0; j < d.n_kv; ++j) {
                q8_row(&d.lat[(size_t) j * kLora], q.data(), sc.data());
                for (int c = 0; c < kLora; ++c) deq[(size_t) j * kLora + c] = (float) q[c] * h2f(sc[c >> 5]);
            }
        }
        std::vector<float> pe16(d.pe.size());
        for (size_t x = 0; x < d.pe.size(); ++x) pe16[x] = h2f(__float2half_rn(d.pe[x]));
        std::vector<float> lat16(d.lat.size());
        for (size_t x = 0; x < d.lat.size(); ++x) lat16[x] = h2f(__float2half_rn(d.lat[x]));
        const auto ref = host_attention(d, d.lat, d.pe, k.window, scale);
        const auto ref_q8 = host_attention(d, deq, pe16, k.window, scale);
        const auto ref_16 = host_attention(d, lat16, pe16, k.window, scale);

        Err e32, e16, eq8, eq8_dev, rd16, rdq8;
        {
            h::KvCache c;
            e32 = compare(run(d, h::kKvF32, false, k.window, scale, c), ref);
        }
        {
            h::KvCache c;
            const auto got = run(d, h::kKvF16, true, k.window, scale, c);
            e16 = compare(got, ref);
            rd16 = compare(got, ref_16);
        }
        {
            h::KvCache c;
            const auto got = run(d, h::kKvQ8, false, k.window, scale, c);
            check_rows(d, c, "host offset");
            eq8 = compare(got, ref);
            rdq8 = compare(got, ref_q8);
        }
        {
            h::KvCache c;
            const auto got = run(d, h::kKvQ8, true, k.window, scale, c);
            check_rows(d, c, "device offset");
            eq8_dev = compare(got, ref);
        }
        std::printf("  read   (kernel vs host on the same rows): f16 rel_rms %.2e   q8_0 rel_rms %.2e\n",
                    rd16.rel_rms, rdq8.rel_rms);
        std::printf("  format (vs host on the original rows):   f32 %.2e   f16 %.2e   q8_0 %.2e (dev %.2e)"
                    "   max_abs q8_0 %.2e of %.2e\n",
                    e32.rel_rms, e16.rel_rms, eq8.rel_rms, eq8_dev.rel_rms, eq8.max_abs, eq8.mag);
        // The kernels read a Q8_0 row as well as they read an F16 one: within 4x of the F16 read error.
        CHECK(rdq8.rel_rms <= std::max(4 * rd16.rel_rms, 1e-5), "q8 read %.3e vs f16 read %.3e", rdq8.rel_rms,
              rd16.rel_rms);
        CHECK(eq8.rel_rms == eq8_dev.rel_rms, "host- and device-offset appends gave different outputs");
        // 8-bit codes on a 32-wide absmax scale. This data puts an 8x outlier channel in every 97, the case such a
        // scale handles worst: measured 1.1-1.5% of the output on a 4090, against F16 0.03-0.05%.
        CHECK(eq8.rel_rms < 3e-2, "q8 quantisation error %.3e", eq8.rel_rms);
    }
    std::printf(g_fail ? "FAILED (%d)\n" : "OK\n", g_fail);
    return g_fail ? 1 : 0;
}
