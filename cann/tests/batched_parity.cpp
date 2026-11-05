// cann/tests/batched_parity.cpp - the batched primitives that a batched GDN would sit on.
//
// bmm, mul_bcast and add_bcast exist so a whole head batch can go through one aclnn call instead
// of one call per head. They are checked here against plain CPU loops before anything depends on
// them.
#include <acl/acl.h>

#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

#include "strata/cann/device.hpp"
#include "strata/cann/ops.hpp"

using namespace strata::cann;

namespace {

std::vector<float> round_f16(const std::vector<float> & v) {
    std::vector<uint16_t> h(v.size());
    f32_to_f16(v.data(), h.data(), v.size());
    std::vector<float> out(v.size());
    f16_to_f32(h.data(), out.data(), out.size());
    return out;
}

void upload(const std::vector<float> & v, Buffer & b) {
    std::vector<uint16_t> h(v.size());
    f32_to_f16(v.data(), h.data(), h.size());
    b.copy_from_host(h.data(), h.size() * 2);
}

std::vector<float> to_host(Stream & stream, const Buffer & b, int64_t n) {
    std::vector<uint16_t> h((size_t) n);
    b.copy_to_host(h.data(), h.size() * 2);
    stream.synchronize();
    std::vector<float> out((size_t) n);
    f16_to_f32(h.data(), out.data(), out.size());
    return out;
}

double rel_l2(const std::vector<float> & a, const std::vector<float> & b) {
    double num = 0.0, den = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        num += (double) (a[i] - b[i]) * (a[i] - b[i]);
        den += (double) b[i] * b[i];
    }
    return den > 0.0 ? std::sqrt(num / den) : 0.0;
}

} // namespace

int main() {
    setvbuf(stdout, nullptr, _IONBF, 0);

    const int64_t B = 8, M = 16, K = 32, N = 16;
    std::mt19937 rng(71);
    std::normal_distribution<float> dist(0.0f, 0.4f);

    std::vector<float> A((size_t) (B * M * K)), C((size_t) (B * K * N));
    for (auto & v : A) v = round_f16({ dist(rng) })[0];
    for (auto & v : C) v = round_f16({ dist(rng) })[0];

    // CPU reference for out[b,m,n] = sum_k A[b,m,k] * C[b,k,n]
    std::vector<float> ref((size_t) (B * M * N), 0.0f);
    for (int64_t b = 0; b < B; ++b) {
        for (int64_t m = 0; m < M; ++m) {
            for (int64_t n = 0; n < N; ++n) {
                double acc = 0.0;
                for (int64_t k = 0; k < K; ++k) {
                    acc += (double) A[(size_t) ((b * M + m) * K + k)] * C[(size_t) ((b * K + k) * N + n)];
                }
                ref[(size_t) ((b * M + m) * N + n)] = (float) acc;
            }
        }
    }

    Device device(0);
    Stream stream(device);
    Buffer bA(A.size() * 2), bC(C.size() * 2), bO(ref.size() * 2);
    upload(A, bA);
    upload(C, bC);

    Tensor tA{ bA.data(), DType::F16, K, M, B };
    Tensor tC{ bC.data(), DType::F16, N, K, B };
    Tensor tO{ bO.data(), DType::F16, N, M, B };
    bmm(stream, tA, tC, tO);
    const std::vector<float> got = to_host(stream, bO, (int64_t) ref.size());

    printf("batched ops: B=%lld M=%lld K=%lld N=%lld\n\n", (long long) B, (long long) M, (long long) K,
           (long long) N);
    printf("%-22s %12s  %s\n", "check", "rel L2", "result");
    double r = rel_l2(got, ref);
    printf("%-22s %12.6f  %s\n", "bmm", r, r <= 0.01 ? "PASS" : "FAIL");
    bool ok = r <= 0.01;

    // broadcast multiply: a scalar per batch entry, then a row per batch entry
    std::vector<float> s((size_t) B), row((size_t) (B * N));
    for (int64_t b = 0; b < B; ++b) {
        s[(size_t) b] = round_f16({ dist(rng) })[0];
        for (int64_t n = 0; n < N; ++n) {
            row[(size_t) (b * N + n)] = round_f16({ dist(rng) })[0];
        }
    }
    Buffer bs((size_t) B * 2), brow((size_t) (B * N) * 2), br1((size_t) ref.size() * 2), br2((size_t) ref.size() * 2);
    upload(s, bs);
    upload(row, brow);
    Tensor ts{ bs.data(), DType::F16, 1, 1, B };
    Tensor trow{ brow.data(), DType::F16, N, 1, B };
    Tensor tr1{ br1.data(), DType::F16, N, M, B };
    Tensor tr2{ br2.data(), DType::F16, N, M, B };

    mul_bcast(stream, tO, ts, tr1);
    const std::vector<float> got_s = to_host(stream, br1, (int64_t) ref.size());
    std::vector<float> ref_s(ref.size());
    for (int64_t b = 0; b < B; ++b) {
        for (int64_t m = 0; m < M; ++m) {
            for (int64_t n = 0; n < N; ++n) {
                ref_s[(size_t) ((b * M + m) * N + n)] =
                    got[(size_t) ((b * M + m) * N + n)] * s[(size_t) b];
            }
        }
    }
    r = rel_l2(got_s, ref_s);
    printf("%-22s %12.6f  %s\n", "mul_bcast [B,1,1]", r, r <= 0.01 ? "PASS" : "FAIL");
    ok = ok && r <= 0.01;

    mul_bcast(stream, tO, trow, tr2);
    const std::vector<float> got_row = to_host(stream, br2, (int64_t) ref.size());
    std::vector<float> ref_row(ref.size());
    for (int64_t b = 0; b < B; ++b) {
        for (int64_t m = 0; m < M; ++m) {
            for (int64_t n = 0; n < N; ++n) {
                ref_row[(size_t) ((b * M + m) * N + n)] =
                    got[(size_t) ((b * M + m) * N + n)] * row[(size_t) (b * N + n)];
            }
        }
    }
    r = rel_l2(got_row, ref_row);
    printf("%-22s %12.6f  %s\n", "mul_bcast [B,1,N]", r, r <= 0.01 ? "PASS" : "FAIL");
    ok = ok && r <= 0.01;

    add_bcast(stream, tO, trow, tr1);
    const std::vector<float> got_add = to_host(stream, br1, (int64_t) ref.size());
    std::vector<float> ref_add(ref.size());
    for (int64_t b = 0; b < B; ++b) {
        for (int64_t m = 0; m < M; ++m) {
            for (int64_t n = 0; n < N; ++n) {
                ref_add[(size_t) ((b * M + m) * N + n)] =
                    got[(size_t) ((b * M + m) * N + n)] + row[(size_t) (b * N + n)];
            }
        }
    }
    r = rel_l2(got_add, ref_add);
    printf("%-22s %12.6f  %s\n", "add_bcast [B,1,N]", r, r <= 0.01 ? "PASS" : "FAIL");
    ok = ok && r <= 0.01;

    shutdown();
    if (!ok) {
        printf("\nthe batched primitives differ from the CPU reference\n");
        return 1;
    }
    printf("\nbatched primitives match the CPU reference\n");
    return 0;
}
