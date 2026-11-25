// src/qwen35/memt_cpu_test.cpp - the FP32 scalar reference for MemT-1.0's
// Native Memory Block, with a deterministic fixture (no model, no GPU).
//
// It prints labeled vectors; tools/memt_cpu_oracle.py computes the same fixture
// in Python and compares them.  The C++ binary alone checks internal invariants
// (a query does not move the state, a commit does) so `ctest` is useful on its
// own.
#include "memt_math.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

namespace M = strata::qwen35::memt;

namespace {

constexpr int H = 2;    // heads
constexpr int D = 4;    // head_dim (K == V)
constexpr int N = 6;    // hidden
constexpr int R = 4;    // gate rank (== D in the release)
constexpr int T = 3;    // tokens
constexpr int HD = H * D;
constexpr int QKV = 3 * HD;
constexpr int ROWS = QKV + 2 * HD + 2 * R;

// Values in [-1, 1) with exact rational arithmetic, so C++ and Python build the
// same fixture without libm differences.
float value(int i) {
    const int raw = (i * 37 + 11) % 101;
    return (float) (raw - 50) / 50.0f;
}

void fill(std::vector<float>& out, int count, int seed) {
    out.resize((size_t) count);
    for (int i = 0; i < count; ++i) out[(size_t) i] = value(seed + i);
}

void print_vector(const char* label, const std::vector<float>& v) {
    std::printf("%s %zu\n", label, v.size());
    for (size_t i = 0; i < v.size(); ++i) {
        std::printf("%.9g%c", v[i], i + 1 == v.size() ? '\n' : ' ');
    }
}

void check(bool ok, const char* what) {
    if (!ok) {
        std::fprintf(stderr, "memt_cpu_test: FAIL %s\n", what);
        std::exit(1);
    }
}

}  // namespace

int main() {

    std::vector<float> in_proj, hidden, f_down, g_down, decay_scale, decay_bias, o_norm;
    fill(in_proj, ROWS * N, 1);
    fill(hidden, T * N, 101);
    fill(f_down, HD * R, 211);
    fill(g_down, HD * R, 307);
    fill(decay_scale, H, 401);
    fill(decay_bias, HD, 409);
    fill(o_norm, D, 503);

    // state0: the published snapshot every token in this request reads
    std::vector<float> state0, state;
    fill(state0, H * D * D, 601);
    state = state0;

    // v comes from the released in_proj; each row group is [q|k|v|erase|write|decay|og]
    std::vector<float> proj((size_t) T * ROWS, 0.0f);
    for (int t = 0; t < T; ++t) {
        for (int row = 0; row < ROWS; ++row) {
            double acc = 0.0;
            for (int j = 0; j < N; ++j) {
                acc += (double) in_proj[(size_t) row * N + j] * hidden[(size_t) t * N + j];
            }
            proj[(size_t) t * ROWS + row] = (float) acc;
        }
    }

    std::vector<float> read_out((size_t) T * HD);
    std::vector<float> erase_all((size_t) T * HD), write_all((size_t) T * HD);
    std::vector<float> decay_all((size_t) T * HD), og_all((size_t) T * HD);

    for (int t = 0; t < T; ++t) {
        const float* p = proj.data() + (size_t) t * ROWS;
        const float* q_raw = p;
        const float* k_raw = p + HD;
        const float* v = p + 2 * HD;
        const float* erase_raw = p + 3 * HD;
        const float* write_raw = p + 4 * HD;
        const float* decay_latent = p + 5 * HD;
        const float* og_latent = p + 5 * HD + R;

        std::vector<float> q(q_raw, q_raw + HD);
        std::vector<float> k(k_raw, k_raw + HD);
        for (int h = 0; h < H; ++h) {
            M::l2norm_in_place(q.data() + h * D, D);   // per head, over D
            M::l2norm_in_place(k.data() + h * D, D);
        }

        std::vector<float> erase(HD), write(HD), log_decay(HD), out_gate(HD);
        for (int h = 0; h < H; ++h) {
            for (int d = 0; d < D; ++d) {
                const int hd = h * D + d;
                float decay_value = decay_bias[(size_t) hd];
                for (int r = 0; r < R; ++r) {
                    decay_value += f_down[(size_t) hd * R + r] * decay_latent[r];
                }
                const M::Gates gates =
                    M::token_gates(erase_raw[hd], write_raw[hd], decay_value,
                                   decay_scale[(size_t) h]);
                erase[(size_t) hd] = gates.erase;
                write[(size_t) hd] = gates.write;
                log_decay[(size_t) hd] = gates.log_decay;
                float og = 0.0f;
                for (int r = 0; r < R; ++r) {
                    og += g_down[(size_t) hd * R + r] * og_latent[r];
                }
                out_gate[(size_t) hd] = og;
            }
        }

        // k carries the release's erase gate (exp(0.5*logsigmoid)), once
        std::vector<float> k_eff(k);
        for (int i = 0; i < HD; ++i) k_eff[(size_t) i] *= erase[(size_t) i];

        // the read always sees the request's frozen snapshot
        std::vector<float> raw_read(HD);
        M::memory_read(q.data(), state0.data(), raw_read.data(), H, D);
        M::gated_rmsnorm(raw_read.data(), out_gate.data(), o_norm.data(),
                         read_out.data() + (size_t) t * HD, H, D, 1e-6f, 2.0f);

        M::memory_update(state.data(), k_eff.data(), v, log_decay.data(),
                         write.data(), H, D);

        for (int i = 0; i < HD; ++i) {
            erase_all[(size_t) t * HD + i] = erase[(size_t) i];
            write_all[(size_t) t * HD + i] = write[(size_t) i];
            decay_all[(size_t) t * HD + i] = log_decay[(size_t) i];
            og_all[(size_t) t * HD + i] = out_gate[(size_t) i];
        }
    }

    // A query must not move the state; a commit must.
    std::vector<float> copy(state0);
    std::vector<float> probe_q((size_t) HD, 0.0f);
    std::vector<float> probe_out((size_t) HD);
    M::memory_read(probe_q.data(), copy.data(), probe_out.data(), H, D);
    check(copy == state0, "a read changed the snapshot");
    check(state != state0, "the update did not change the state");

    std::printf("GEOM %d %d %d %d %d\n", H, D, N, R, T);
    print_vector("READ", read_out);
    print_vector("STATE", state);
    print_vector("ERASE", erase_all);
    print_vector("WRITE", write_all);
    print_vector("LOGDECAY", decay_all);
    print_vector("OUTGATE", og_all);
    return 0;
}
