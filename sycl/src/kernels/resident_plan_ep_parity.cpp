// src/kernels/resident_plan_ep_parity.cpp (SYCL port only) - the two plans of an expert-parallel pair
// (resident_plan_ep, ep_kernels.hpp) against a host replay of their contract.
//
// Each expert is resident on card 0, on card 1, on both, or on neither.  Owner: card 0 when it holds the expert, else
// card 1.  For each role the plan must be resident_plan's layout over exactly the entries that card owns: distinct
// owned experts in routing order, each with its entries ascending, ptr[g] = the slot on that card, counts[1] = its
// entries.  Together the two plans cover every entry once.  An expert on neither card leaves an empty plan and sets
// *plan_err.  And with every expert on card 0, role 1 is resident_plan's all-resident plan word for word.
// n = 1..80 entries, ids from pools of 3..512 experts.  GPU, synthetic, no model.
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include "strata/kernels/verify_kernels.hpp"
#include "strata/kernels/ep_kernels.hpp"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

namespace k = strata::kernels;

int main() {
    const int K = 10, NE = 512;
    const long long capx = (long long) k::kVerifyMaxT * K;
    const long long ptr_off = ((4 + (capx + 1) + 2 * capx) + 1) & ~1ll;
    const size_t plan_words = (size_t) (ptr_off + 4 * capx + capx + 16);
    const long long blob = 4096;
    uint8_t* const base = reinterpret_cast<uint8_t*>(uintptr_t(0x40000000));   // never dereferenced
    sycl::queue& q = dpct::get_in_order_queue();
    auto* d_ids = sycl::malloc_device<int32_t>(128, q);
    auto* d_r0 = sycl::malloc_device<int32_t>(NE, q);
    auto* d_r1 = sycl::malloc_device<int32_t>(NE, q);
    auto* d_plan = sycl::malloc_device<int32_t>(plan_words, q);
    auto* d_plan2 = sycl::malloc_device<int32_t>(plan_words, q);
    auto* d_err = sycl::malloc_device<uint32_t>(1, q);
    std::mt19937 rng(1402);
    int fails = 0, cases = 0;

    auto read_plan = [&](int32_t* d) {
        std::vector<int32_t> h(plan_words);
        q.memcpy(h.data(), d, plan_words * 4).wait();
        return h;
    };
    // the host replay: the plan of `role` over `ids`
    auto expect = [&](const std::vector<int32_t>& ids, const std::vector<int32_t>& r0, const std::vector<int32_t>& r1,
                      int role, std::vector<int>& groups_e, std::vector<std::vector<int>>& groups_i) {
        groups_e.clear();
        groups_i.clear();
        for (int i = 0; i < (int) ids.size(); ++i) {
            const int e = ids[(size_t) i];
            const int owner = r0[(size_t) e] >= 0 ? 1 : 2;
            if (owner != role) continue;
            int g = 0;
            while (g < (int) groups_e.size() && groups_e[(size_t) g] != e) ++g;
            if (g == (int) groups_e.size()) { groups_e.push_back(e); groups_i.emplace_back(); }
            groups_i[(size_t) g].push_back(i);
        }
    };

    for (int pool : {3, 10, 40, 160, 512}) {
        for (int n = 1; n <= 80; n += (n < 12 ? 1 : 7)) {
            for (int rep = 0; rep < 4; ++rep) {
                ++cases;
                std::vector<int32_t> ids((size_t) n), r0((size_t) NE, -1), r1((size_t) NE, -1);
                std::vector<int> pick((size_t) pool);
                for (int& p : pick) p = (int) (rng() % NE);
                for (auto& v : ids) v = pick[rng() % (size_t) pool];
                // ownership: mostly one card, some on both (card 0 wins), on neither only in the error case below
                for (int e = 0; e < NE; ++e) {
                    const unsigned r = rng() % 100;
                    if (r < 45) r0[(size_t) e] = (int) (rng() % 9000);
                    else if (r < 90) r1[(size_t) e] = (int) (rng() % 9000);
                    else { r0[(size_t) e] = (int) (rng() % 9000); r1[(size_t) e] = (int) (rng() % 9000); }
                }
                const bool make_missing = rep == 3;
                if (make_missing) { r0[(size_t) ids[0]] = -1; r1[(size_t) ids[0]] = -1; }
                q.memcpy(d_ids, ids.data(), (size_t) n * 4).wait();
                q.memcpy(d_r0, r0.data(), NE * 4).wait();
                q.memcpy(d_r1, r1.data(), NE * 4).wait();
                int covered = 0;
                for (int role = 1; role <= 2; ++role) {
                    q.memset(d_plan, 0xff, plan_words * 4).wait();
                    q.memset(d_err, 0, 4).wait();
                    k::resident_plan_ep(d_ids, n, K, d_r0, d_r1, role, NE, base, nullptr, blob, d_plan, capx, &q, d_err);
                    q.wait();
                    const std::vector<int32_t> h = read_plan(d_plan);
                    uint32_t err = 0;
                    q.memcpy(&err, d_err, 4).wait();
                    if (make_missing) {
                        if (err != 1 || h[0] != 0 || h[1] != 0) {
                            if (fails++ < 10) std::printf("FAIL pool %d n %d role %d: a missing expert gave err %u groups %d\n", pool, n, role, err, h[0]);
                        }
                        continue;
                    }
                    std::vector<int> ge;
                    std::vector<std::vector<int>> gi;
                    expect(ids, r0, r1, role, ge, gi);
                    const int32_t* start = h.data() + 4;
                    const int32_t* dst = start + capx + 1;
                    const int32_t* tok = dst + capx;
                    const auto* ptr = (const unsigned long long*) (h.data() + ptr_off);
                    const int32_t* start2 = h.data() + ptr_off + 4 * capx;
                    int ents = 0;
                    for (const auto& v : gi) ents += (int) v.size();
                    bool ok = err == 0 && h[0] == (int) ge.size() && h[1] == ents && h[2] == 0 && start[ge.size()] == ents &&
                              start2[0] == ents;
                    int at = 0;
                    for (size_t g = 0; ok && g < ge.size(); ++g) {
                        const auto& own = role == 1 ? r0 : r1;
                        ok = ptr[g] == (unsigned long long) (base + (size_t) own[(size_t) ge[g]] * (size_t) blob) &&
                             start[g] == at;
                        for (int i : gi[g]) {
                            ok = ok && dst[at] == i && tok[at] == i / K;
                            ++at;
                        }
                    }
                    covered += ents;
                    if (!ok && fails++ < 10)
                        std::printf("FAIL pool %d n %d role %d: groups %d (want %zu) entries %d (want %d) err %u\n", pool, n,
                                    role, h[0], ge.size(), h[1], ents, err);
                }
                if (!make_missing && covered != n && fails++ < 10)
                    std::printf("FAIL pool %d n %d: the two plans cover %d of %d entries\n", pool, n, covered, n);

                // every expert on card 0: role 1 is resident_plan's all-resident plan word for word
                if (!make_missing) {
                    for (int e = 0; e < NE; ++e) if (r0[(size_t) e] < 0) r0[(size_t) e] = (int) (rng() % 9000);
                    q.memcpy(d_r0, r0.data(), NE * 4).wait();
                    q.memset(d_plan, 0xff, plan_words * 4).wait();
                    q.memset(d_plan2, 0xff, plan_words * 4).wait();
                    k::resident_plan_ep(d_ids, n, K, d_r0, d_r1, 1, NE, base, nullptr, blob, d_plan, capx, &q, d_err);
                    k::resident_plan(d_ids, n, K, d_r0, NE, base, nullptr, blob, d_plan2, capx, nullptr, 0, &q, d_err);
                    q.wait();
                    if (read_plan(d_plan) != read_plan(d_plan2) && fails++ < 10)
                        std::printf("FAIL pool %d n %d: role 1 with every expert on card 0 differs from resident_plan\n", pool, n);
                }
            }
        }
    }
    std::printf("resident_plan_ep_parity: %d cases, %d failures\n", cases, fails);
    return fails == 0 ? 0 : 1;
}
