// src/qwen35/memt_pool_test.cpp - MemT-1.0's session memory pool and the
// query/commit event semantics (header-only, no GPU).
#include "memt_pool.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

namespace M = strata::qwen35::memt;

namespace {

int g_checks = 0;
int g_fail = 0;

void check(bool ok, const char* what) {
    ++g_checks;
    if (!ok) {
        std::fprintf(stderr, "FAIL: %s\n", what);
        ++g_fail;
    }
}

std::vector<float> values(int count, int seed) {
    std::vector<float> out((size_t) count);
    for (int i = 0; i < count; ++i) out[(size_t) i] = (float) ((i * 37 + seed) % 29 - 14) / 14.0f;
    return out;
}

std::vector<float> gates(int count, int seed, bool decay = false) {
    std::vector<float> out = values(count, seed);
    for (float& x : out) x = decay ? -0.25f * (x + 1.0f) : 0.5f + 0.25f * (x + 1.0f);
    return out;
}

bool close(const std::vector<float>& a, const std::vector<float>& b, float tol) {
    if (a.size() != b.size()) return false;
    for (size_t i = 0; i < a.size(); ++i)
        if (std::fabs(a[i] - b[i]) > tol) return false;
    return true;
}

void hand_case() {
    const int H = 1, D = 2, T = 2;
    std::vector<float> state = {1, 0, 0, 1};
    const std::vector<float> q = {1, 0, 0, 1};
    const std::vector<float> k = {0, 1, 0, 1};
    const std::vector<float> v = {1, 1, 1, 1};
    const std::vector<float> erase = {1, 1, 1, 1};
    const std::vector<float> write = {1, 1, 1, 1};
    const std::vector<float> decay = {0, 0, 0, 0};
    const std::vector<float> gate = {1, 1, 1, 1};
    const std::vector<float> norm = {1, 1};
    std::vector<float> out((size_t) T * H * D);

    M::Pool pool(1, 1, H, D);
    pool.import_session(0, state.data());
    M::commit(pool.at(0, 0), q.data(), k.data(), v.data(), erase.data(), write.data(),
              decay.data(), gate.data(), norm.data(), out.data(), T, H, D, 1, 1e-6f, 2.0f);
    // both tokens read the call's snapshot (the identity): token 0's query
    // [1, 0] sees row 0, token 1's query [0, 1] sees row 1
    check(out[0] > 0.1f && std::fabs(out[1]) < 1e-6f, "hand: token 0 reads the identity row 0");
    check(std::fabs(out[2]) < 1e-6f && out[3] > 0.1f, "hand: token 1 reads the identity row 1");

    M::Pool single(1, 1, H, D);
    single.import_session(0, state.data());
    std::vector<float> out2((size_t) T * H * D);
    M::commit(single.at(0, 0), q.data(), k.data(), v.data(), erase.data(), write.data(),
              decay.data(), gate.data(), norm.data(), out2.data(), T, H, D, 2, 1e-6f, 2.0f);
    // one write event reaches the same state and the same reads as two
    check(close(out, out2, 1e-6f), "hand: the event size changes neither the read nor the state");
}

}  // namespace

// A small deterministic generator, so a property case is reproducible from its
// index alone (no <random>: the test stays dependency-free).
struct Rng {
    unsigned long long s;
    explicit Rng(unsigned long long seed) : s(seed * 6364136223846793005ULL + 1442695040888963407ULL) {}
    float next() {   // [-1, 1)
        s = s * 6364136223846793005ULL + 1442695040888963407ULL;
        return (float) ((s >> 40) & 0xFFFFFF) / (float) 0x800000 - 1.0f;
    }
};

// A batch window's rows read their own sessions; a verify window's rows read one snapshot.
void rows_case() {
    const int H = 2, D = 3, per = H * D, rows = 6;
    std::vector<float> s0 = values(H * D * D, 3), s1 = values(H * D * D, 5);
    std::vector<float> q = values(rows * per, 7), gate = values(rows * per, 11), norm = values(D, 13);
    M::Pool pool(2, 1, H, D);
    pool.import_session(0, s0.data());
    pool.import_session(1, s1.data());
    const int slot[rows] = {0, 1, 1, 0, 1, 0};      // the window's row -> session table (hrow's rows, as the engine fills it)
    std::vector<float> out((size_t) rows * per);
    for (int r = 0; r < rows; ++r)
        M::query(pool.at(slot[r], 0), q.data() + (size_t) r * per, gate.data() + (size_t) r * per, norm.data(),
                 out.data() + (size_t) r * per, 1, H, D, 1e-6f, 2.0f);
    for (int r = 0; r < rows; ++r) {                 // every row equals its own session's single-token query
        std::vector<float> one = pool.export_session(slot[r]), single((size_t) per);
        M::query(one.data(), q.data() + (size_t) r * per, gate.data() + (size_t) r * per, norm.data(), single.data(), 1,
                 H, D, 1e-6f, 2.0f);
        bool same = true;
        for (int i = 0; i < per; ++i) same = same && std::fabs(out[(size_t) r * per + i] - single[(size_t) i]) <= 1e-9f;
        check(same, "row window: each row reads its own session's state");
    }
    check(pool.export_session(0) == s0 && pool.export_session(1) == s1,
          "row window: a query leaves every session's state as it was");

    // the verify window: T rows of ONE session, one snapshot for all of them
    std::vector<float> verify((size_t) rows * per), one_by_one((size_t) rows * per);
    M::query(pool.at(0, 0), q.data(), gate.data(), norm.data(), verify.data(), rows, H, D, 1e-6f, 2.0f);
    for (int r = 0; r < rows; ++r)
        M::query(pool.at(0, 0), q.data() + (size_t) r * per, gate.data() + (size_t) r * per, norm.data(),
                 one_by_one.data() + (size_t) r * per, 1, H, D, 1e-6f, 2.0f);
    check(close(verify, one_by_one, 1e-9f), "verify window: every row reads the same snapshot (one row at a time)");
}

// Properties over generated cases (bounded, seeded): the session semantics must
// not depend on how the tokens are split into write events, a commit's read must
// equal a query of the state the call started with, and a query must be pure.
void property_cases() {
    for (int c = 0; c < 12; ++c) {
        Rng rng((unsigned long long) c + 1);
        const int H = 1 + c % 3, D = 2 + c % 5;
        const int T = 1 + (c * 7) % 12;
        const int per = H * D;
        auto field = [&](int count, float lo, float hi) {
            std::vector<float> out((size_t) count);
            for (float& x : out) x = lo + (hi - lo) * (0.5f + 0.5f * rng.next());
            return out;
        };
        std::vector<float> state = field(H * D * D, -0.5f, 0.5f);
        std::vector<float> q = field(T * per, -1.0f, 1.0f);
        std::vector<float> k = field(T * per, -1.0f, 1.0f);
        std::vector<float> v = field(T * per, -1.0f, 1.0f);
        std::vector<float> erase = field(T * per, 0.05f, 1.0f);    // exp(0.5*logsigmoid) in (0, 1)
        std::vector<float> write = field(T * per, 0.05f, 0.95f);   // sigmoid in (0, 1)
        std::vector<float> log_decay = field(T * per, -0.5f, -0.01f);
        std::vector<float> out_gate = field(T * per, -1.0f, 1.0f);
        std::vector<float> o_norm = field(D, 0.5f, 1.5f);

        // the read of a commit equals a query of the call-start state
        std::vector<float> start = state, qout((size_t) T * per);
        M::query(start.data(), q.data(), out_gate.data(), o_norm.data(), qout.data(), T, H, D,
                 1e-6f, 2.0f);
        check(start == state, "generated case: query is pure");

        // Within ONE call, neither the state nor the read depends on the
        // write-event size (the call freezes one snapshot for all its tokens).
        std::vector<float> by_one = state, by_all = state, by_three = state;
        std::vector<float> out_one((size_t) T * per), out_all((size_t) T * per), out_three((size_t) T * per);
        M::commit(by_one.data(), q.data(), k.data(), v.data(), erase.data(), write.data(),
                  log_decay.data(), out_gate.data(), o_norm.data(), out_one.data(), T, H, D, 1,
                  1e-6f, 2.0f);
        M::commit(by_all.data(), q.data(), k.data(), v.data(), erase.data(), write.data(),
                  log_decay.data(), out_gate.data(), o_norm.data(), out_all.data(), T, H, D, T,
                  1e-6f, 2.0f);
        M::commit(by_three.data(), q.data(), k.data(), v.data(), erase.data(), write.data(),
                  log_decay.data(), out_gate.data(), o_norm.data(), out_three.data(), T, H, D, 3,
                  1e-6f, 2.0f);
        check(by_one == by_all, "generated case: one event and T events give the same state");
        check(by_all == by_three, "generated case: 3-token events give the same state");
        check(close(out_one, out_all, 1e-6f), "generated case: the read does not depend on the events");
        check(close(out_all, qout, 1e-6f), "generated case: a commit's read is a query of the start state");

        // Across CALLS the final state is still the same (the update does not
        // read its own output), but each call freezes its own snapshot, so the
        // reads must differ -- that is the documented semantics, not a bug.
        std::vector<float> by_calls = state, out_calls((size_t) T * per);
        for (int t = 0; t < T; ++t) {
            M::commit(by_calls.data(), q.data() + (size_t) t * per, k.data() + (size_t) t * per,
                      v.data() + (size_t) t * per, erase.data() + (size_t) t * per,
                      write.data() + (size_t) t * per, log_decay.data() + (size_t) t * per,
                      out_gate.data() + (size_t) t * per, o_norm.data(), out_calls.data() + (size_t) t * per,
                      1, H, D, 1, 1e-6f, 2.0f);
        }
        check(by_calls == by_all, "generated case: call boundaries do not change the final state");
        if (T > 1) {
            float worst = 0.0f;
            for (int i = 0; i < T * per; ++i)
                worst = std::max(worst, std::fabs(out_calls[(size_t) i] - out_all[(size_t) i]));
            check(worst > 1e-7f, "generated case: separate calls read their own snapshots");
        }
        bool finite = true;
        for (float x : by_all) finite = finite && std::isfinite(x);
        for (float x : out_all) finite = finite && std::isfinite(x);
        check(finite, "generated case: finite state and read");
    }
}

int main() {
    hand_case();
    rows_case();
    property_cases();

    const int H = 2, D = 4, T = 9, chunk = 4;
    const int per = H * D;
    M::Pool pool(2, 1, H, D);
    std::vector<float> state0 = values(H * D * D, 3);
    pool.import_session(0, state0.data());

    std::vector<float> q = values(T * per, 11);
    std::vector<float> k = values(T * per, 23);
    std::vector<float> v = values(T * per, 37);
    std::vector<float> erase = gates(T * per, 41);
    std::vector<float> write = gates(T * per, 53);
    std::vector<float> decay = gates(T * per, 67, true);
    std::vector<float> gate = values(T * per, 79);
    std::vector<float> norm = values(D, 83);

    // a query never moves the published state
    std::vector<float> qout((size_t) T * per);
    std::vector<float> before = pool.export_session(0);
    M::query(pool.at(0, 0), q.data(), gate.data(), norm.data(), qout.data(), T, H, D,
             1e-6f, 2.0f);
    check(pool.export_session(0) == before, "query leaves the state unchanged");

    // a commit moves it; the token update does not depend on the event size
    std::vector<float> out4((size_t) T * per), out9((size_t) T * per);
    std::vector<float> state4 = pool.export_session(0), state9 = pool.export_session(0);
    M::commit(state4.data(), q.data(), k.data(), v.data(), erase.data(), write.data(),
              decay.data(), gate.data(), norm.data(), out4.data(), T, H, D, chunk,
              1e-6f, 2.0f);
    M::commit(state9.data(), q.data(), k.data(), v.data(), erase.data(), write.data(),
              decay.data(), gate.data(), norm.data(), out9.data(), T, H, D, T,
              1e-6f, 2.0f);
    check(close(state4, state9, 1e-6f), "state is the same for any event size");
    for (float x : out4) check(std::isfinite(x), "commit read is finite");
    // every token reads the state the call started with, so the write-event size
    // changes neither the read nor the state (memt_sglang's per-sample snapshot)
    check(close(out4, out9, 1e-6f), "the read does not depend on the write-event size");
    check(close(qout, out4, 1e-6f), "a commit's read is a query of the call-start state");

    // a chunk exactly at the end and a one-token tail are both handled
    std::vector<float> out_tail((size_t) T * per);
    std::vector<float> state_tail = pool.export_session(0);
    M::commit(state_tail.data(), q.data(), k.data(), v.data(), erase.data(), write.data(),
              decay.data(), gate.data(), norm.data(), out_tail.data(), T, H, D, 5,
              1e-6f, 2.0f);
    check(close(state_tail, state4, 1e-6f), "5 + 4 events reach the same state");
    check(close(out_tail, out4, 1e-6f), "5 + 4 events reach the same reads");

    std::vector<float> imported = values(H * D * D, 97);
    pool.import_session(1, imported.data());
    check(pool.export_session(1) == imported && pool.export_session(0) == before,
          "import_session is per session");

    if (g_fail == 0) std::printf("memt_pool_test: %d checks passed\n", g_checks);
    return g_fail == 0 ? 0 : 1;
}
