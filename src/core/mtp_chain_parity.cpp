// src/core/mtp_chain_parity.cpp - the draft chain at up to kChainMax (11) outputs (--pipeline-windows 3), checked
// against the chain it replaces.
//
// MtpDrafterBase is src/core/mtp.cpp + include/strata/core/mtp.hpp as they were at STRATA_MTP_PARITY_BASE (the commit
// before the 11-output chain), copied into the build directory by CMake with the class renamed, so the old
// and the new code run side by side in this one process on the same model files.  Four drafters go through the same
// rounds in lockstep (random T, a, tokens, forced tokens and source rows, after a prompt long enough to wrap the
// drafter's K/V ring), one at a time on one GPU:
//
//   base  the old code, n_out = n7 in 1..7                   \  1. every output (token and probability bits) the same,
//   a     the new code at its default chain max (max_t - 1)  /     every round: n_out <= 7 is bit-identical to today
//   b     the new code, set_chain_max(11), 11 outputs queued at once, forced further when a forced all of its own:
//         2. b's first n7 outputs are a's (an 11-output chain shares the prefix)
//   c     the new code as b, but launched with n_queue (1..11) and grown with chain_extend one step ahead of the
//         outputs it has read, closed early in some rounds: 3. c's outputs are b's prefix (all 11 when it is extended
//         to the end: step-wise equals the full queue)
//
// The cells a longer chain writes past a shorter one's are never read before they are written again (every read of a
// round is of a cell that round or an earlier one wrote with the same rows), so the four stay comparable round after
// round.  Also checked: 4. source_done() fires after the source rows are in the bound buffer (b, its bound buffer
// filled with a sentinel before each launch); 5. a's default chain max still refuses n_out = 8; 6. the max_cells
// clamp (b at the end of the context: chain_n() = the room left, no error); 7. the serial draft() of base and a
// after the rounds (the staging rows it uses did not move).
//
// Usage: mtp_chain_parity --pack DIR --native SHARD1.gguf --mtp RT_DIR [--rounds 300] [--seed 1] [--max-cells 32768]
//        [--window 8192] [--prompt 9000] [--hnorm stream|all] [--kv int8|f16] [--ref base|base2|new2]
// --ref base2 / new2: checks 1 and 7 compare two old / two new instances instead of old against new (the harness's own
// determinism).
// One GPU: pick it with CUDA_VISIBLE_DEVICES.  Exit 0: all checks pass; 1: a mismatch; 2: setup failed.
#include "strata/core/mtp.hpp"
#define MtpDrafter MtpDrafterBase
#include "mtp_base.hpp"
#undef MtpDrafter

#include "strata/artifact/gguf_split.hpp"
#include "strata/core/layer.hpp"
#include "strata/core/native_head.hpp"
#include "strata/core/session.hpp"
#include "strata/core/weights.hpp"

#include <cuda_runtime.h>

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <random>
#include <set>
#include <sstream>
#include <string>
#include <vector>

namespace {

using strata::core::MtpDrafter;
using strata::core::MtpDrafterBase;

struct Opts {
    std::string pack, native, mtp;
    int rounds = 300;
    uint32_t seed = 1;
    int64_t max_cells = 32768, window = 8192, prompt = 9000;
    bool hnorm_stream = true, kv_int8 = true;
    std::string ref = "base";   // check 1's reference: base (the old code), base2 / new2 (a second old / new instance)
};

int fail_setup(const std::string& what) {
    std::fprintf(stderr, "mtp_chain_parity: %s\n", what.c_str());
    return 2;
}

bool same_bits(float x, float y) { return std::memcmp(&x, &y, sizeof(float)) == 0; }

// every tensor name in <pack>/index.txt: the drafter needs only the table's metadata (output.weight's vocabulary,
// token_embd.weight's presence), the native embedding and head serve the rest
bool pack_names(const std::string& pack, std::set<std::string>& out, std::string& err) {
    std::ifstream f(pack + "/index.txt");
    if (!f) { err = "cannot open " + pack + "/index.txt"; return false; }
    std::string line;
    while (std::getline(f, line)) {
        if (line.empty() || line[0] == '#') continue;
        std::istringstream is(line);
        std::string name;
        if (is >> name) out.insert(name);
    }
    return !out.empty();
}

// poll a chain to its end (reading its early outputs on the way, as the pipelined loop does)
template <class D> bool finish(D& d, std::string& err) {
    for (;;) {
        if (d.chain_outputs_ready(err) < 0) return false;
        const int r = d.chain_poll(err);
        if (r < 0) return false;
        if (r == 1) return true;
    }
}

}  // namespace

int main(int argc, char** argv) {
    Opts o;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&]() -> std::string { return i + 1 < argc ? argv[++i] : std::string(); };
        if (a == "--pack") o.pack = next();
        else if (a == "--native") o.native = next();
        else if (a == "--mtp") o.mtp = next();
        else if (a == "--rounds") o.rounds = std::atoi(next().c_str());
        else if (a == "--seed") o.seed = (uint32_t) std::strtoul(next().c_str(), nullptr, 10);
        else if (a == "--max-cells") o.max_cells = std::atoll(next().c_str());
        else if (a == "--window") o.window = std::atoll(next().c_str());
        else if (a == "--prompt") o.prompt = std::atoll(next().c_str());
        else if (a == "--hnorm") o.hnorm_stream = next() == "stream";
        else if (a == "--kv") o.kv_int8 = next() == "int8";
        else if (a == "--ref") o.ref = next();
        else return fail_setup("unknown argument " + a);
    }
    if (o.pack.empty() || o.native.empty() || o.mtp.empty())
        return fail_setup("usage: mtp_chain_parity --pack DIR --native SHARD1.gguf --mtp RT_DIR [--rounds N] [--seed S] "
                          "[--max-cells N] [--window N] [--prompt N] [--hnorm stream|all] [--kv int8|f16] [--ref base|base2|new2]");
    if (o.ref != "base" && o.ref != "base2" && o.ref != "new2") return fail_setup("--ref is base, base2 or new2");
    constexpr int kMaxT = 8;   // the drafter's rows under --pipeline-windows (kVerifyMaxT)
    constexpr int kC = MtpDrafter::kChainMax;
    if (o.prompt + (int64_t) o.rounds * kMaxT + 64 >= o.max_cells) return fail_setup("--max-cells is too small");

    cudaSetDevice(0);
    {
        cudaDeviceProp prop{};
        cudaGetDeviceProperties(&prop, 0);
        std::fprintf(stderr, "mtp_chain_parity: %s (PCI %02x:%02x), %d rounds, seed %u, max_cells %lld, window %lld, "
                             "prompt %lld, hnorm %s, kv %s\n",
                     prop.name, prop.pciBusID, prop.pciDeviceID, o.rounds, o.seed, (long long) o.max_cells,
                     (long long) o.window, (long long) o.prompt, o.hnorm_stream ? "stream" : "all",
                     o.kv_int8 ? "int8" : "f16");
    }
    std::string err;
    static const strata::core::ModelGeometry g{};
    const int64_t HCN = g.hc * g.n_embd;

    // ---- the model side the drafter reads: a one-layer session (its K/V's rope table), the weight table's metadata,
    // the native embedding and head
    strata::core::qsa_set_kv_int8(o.kv_int8);
    strata::core::QsaState prim;
    void* prim_arena = nullptr;
    {
        const uint64_t sb = strata::core::qsa_state_bytes(g, o.max_cells);
        if (cudaMalloc(&prim_arena, sb) != cudaSuccess) return fail_setup("the session state does not fit");
        if (strata::core::qsa_state_init(g, o.max_cells, prim_arena, prim) == 0) return fail_setup("session state init");
    }
    strata::core::SessionState ss;
    ss.max_cells = o.max_cells;
    ss.qsa_states = &prim;
    ss.qsa_ord0 = 0;
    ss.qsa_alloc = 1;
    ss.k = 10;

    std::set<std::string> skip;
    if (!pack_names(o.pack, skip, err)) return fail_setup(err);
    uint64_t pool = 0;
    if (!strata::core::WeightTable::pool_bytes(o.pack, pool, err, &skip)) return fail_setup(err);
    pool = std::max<uint64_t>(pool, 4096);
    void* wt_arena = nullptr;
    if (cudaMalloc(&wt_arena, pool) != cudaSuccess) return fail_setup("weight arena");
    strata::core::WeightTable wt;
    if (!wt.load(o.pack, wt_arena, pool, err, &skip)) return fail_setup(err);
    const strata::core::WeightRef* wo = wt.find("output.weight");
    if (wo == nullptr) return fail_setup("output.weight is missing from the pack index");
    const int64_t n_vocab = wo->ne1;
    const std::vector<std::string> shards = strata::gguf_split_paths(o.native);
    strata::core::NativeEmbed embed;
    if (!embed.load(shards, g.n_embd, n_vocab, err)) return fail_setup(err);
    strata::core::set_native_embed(&embed);
    strata::core::NativeHead head;
    if (!head.load(shards, g.n_embd, n_vocab, err)) return fail_setup(err);

    // ---- the four drafters: each its own bound rows, one source the chains copy from
    float* src_R = nullptr;
    float* bound[6] = {};
    float* check_R = nullptr;
    if (cudaMalloc((void**) &src_R, (size_t) kMaxT * HCN * sizeof(float)) != cudaSuccess ||
        cudaMalloc((void**) &check_R, (size_t) kMaxT * HCN * sizeof(float)) != cudaSuccess)
        return fail_setup("row buffers");
    for (float*& b : bound) {
        if (cudaMalloc((void**) &b, (size_t) kMaxT * HCN * sizeof(float)) != cudaSuccess) return fail_setup("row buffers");
        cudaMemset(b, 0, (size_t) kMaxT * HCN * sizeof(float));   // draft() (check 7) reads rows a chain may not have
    }

    MtpDrafterBase base, base2;   // base2: --ref base2 only (two old instances: the harness's own determinism)
    MtpDrafter da, da2, db, dc;   // da2: --ref new2 only (two new instances, each with its own weights)
    auto setup = [&](auto& d, const auto* shared, float* bound_R, auto&& before_chain) -> bool {
        d.set_hnorm_per_stream(o.hnorm_stream);
        d.set_force_capture(true);
        d.set_prompt_len(o.prompt);
        if (!d.load(o.mtp, g, ss, kMaxT, err, o.window, shared)) return false;
        if (!d.bind(wt, &head, bound_R, err, shared)) return false;
        d.set_source_R(src_R);
        if (!d.prepare_prefill(err)) return false;
        before_chain(d);
        return d.prepare_chain(err);
    };
    auto none = [](auto&) {};
    const bool use_base = o.ref != "new2", use_base2 = o.ref == "base2", use_da2 = o.ref == "new2";
    if (use_base && !setup(base, (const MtpDrafterBase*) nullptr, bound[0], none)) return fail_setup("base: " + err);
    if (use_base2 && !setup(base2, (const MtpDrafterBase*) nullptr, bound[4], none)) return fail_setup("base2: " + err);
    if (use_da2 && !setup(da2, (const MtpDrafter*) nullptr, bound[5], none)) return fail_setup("a2: " + err);
    if (!setup(da, (const MtpDrafter*) nullptr, bound[1], none)) return fail_setup("a: " + err);
    if (!setup(db, &da, bound[2], [](MtpDrafter& d) { d.set_chain_max(MtpDrafter::kChainMax); d.set_source_done(true); }))
        return fail_setup("b: " + err);
    if (!setup(dc, &da, bound[3], [](MtpDrafter& d) { d.set_chain_max(MtpDrafter::kChainMax); })) return fail_setup("c: " + err);
    std::fprintf(stderr, "mtp_chain_parity: drafters ready (chain max: a %d, b %d, c %d)\n", da.chain_max(),
                 db.chain_max(), dc.chain_max());

    std::mt19937 rng(o.seed);
    // c's extend / close choices: their own stream (how often they are drawn depends on the GPU's timing), so a seed
    // gives the same rounds run after run
    std::mt19937 rng_c(o.seed ^ 0x9e3779b9u);
    auto rnd = [&](int n) { return (int) (rng() % (uint32_t) n); };
    std::uniform_real_distribution<float> uni(-1.0f, 1.0f);
    auto tok = [&]() { return (int32_t) (rng() % (uint32_t) n_vocab); };

    // ---- the prompt: identical K/V in all four (cells below prompt - window - 64 are skipped by each alike)
    {
        const int64_t chunk = 512;
        std::vector<float> rows((size_t) (chunk * HCN));
        std::vector<int32_t> next((size_t) chunk);
        float* d_rows = nullptr;
        if (cudaMalloc((void**) &d_rows, rows.size() * sizeof(float)) != cudaSuccess) return fail_setup("prompt rows");
        const auto t0 = std::chrono::steady_clock::now();
        for (int64_t c0 = 0; c0 < o.prompt; c0 += chunk) {
            const int64_t n = std::min(chunk, o.prompt - c0);
            for (int64_t i = 0; i < n * HCN; ++i) rows[(size_t) i] = uni(rng);
            for (int64_t i = 0; i < n; ++i) next[(size_t) i] = tok();
            // (a pageable upload can return before its DMA lands, and the drafters' streams do not wait for the
            // legacy stream: without the sync the drafter that ran first read rows still being written)
            cudaMemcpy(d_rows, rows.data(), (size_t) (n * HCN) * sizeof(float), cudaMemcpyHostToDevice);
            cudaDeviceSynchronize();
            auto pf = [&](auto& d, bool on) { return !on || d.prefill(d_rows, next.data(), n, c0, err, true); };
            if (!pf(base, use_base) || !pf(base2, use_base2) || !pf(da2, use_da2) || !pf(da, true) || !pf(db, true) ||
                !pf(dc, true))
                return fail_setup("prefill: " + err);
        }
        cudaFree(d_rows);
        std::fprintf(stderr, "mtp_chain_parity: prompt of %lld cells in %.1f s\n", (long long) o.prompt,
                     std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
    }

    // ---- the rounds
    int bad = 0;
    auto mismatch = [&](int round, const char* what, int j, int32_t t0, float p0, int32_t t1, float p1) {
        if (++bad <= 20)
            std::fprintf(stderr, "MISMATCH round %d %s output %d: %d (%.9g) vs %d (%.9g)\n", round, what, j, (int) t0,
                         (double) p0, (int) t1, (double) p1);
    };
    int64_t p = o.prompt;
    int64_t n_cmp1 = 0, n_cmp2 = 0, n_cmp3 = 0, c_full_ext = 0, c_closed = 0, c_queued_all = 0, b_forced_more = 0;
    int64_t src_checks = 0;
    double ms_b = 0;
    std::vector<float> src_host((size_t) kMaxT * HCN), chk_host((size_t) kMaxT * HCN);
    cudaStream_t chk = nullptr;
    cudaStreamCreateWithFlags(&chk, cudaStreamNonBlocking);
    for (int r = 0; r < o.rounds; ++r) {
        const int T = 1 + rnd(kMaxT), a = rnd(T);
        const int n7 = 1 + rnd(kMaxT - 1), nf7 = rnd(n7 + 1), early7 = rnd(n7 + 1);
        int32_t tokens[kMaxT], force[16];
        for (int t = 0; t < T; ++t) tokens[t] = tok();
        for (int32_t& f : force) f = tok();
        // b forces further only where a's outputs cannot see it: a forced every step it ran (steps 1..n7-1)
        int nfb = nf7;
        if (nf7 >= n7 - 1) {
            nfb = nf7 + rnd(kC - nf7);   // up to 10 forced: steps 1..10
            if (nfb > nf7) ++b_forced_more;
        }
        for (int64_t i = 0; i < (int64_t) T * HCN; ++i) src_host[(size_t) i] = uni(rng);
        cudaMemcpy(src_R, src_host.data(), (size_t) T * HCN * sizeof(float), cudaMemcpyHostToDevice);
        cudaDeviceSynchronize();   // (see the prompt's upload)

        // 1. the reference (base: the old code) and a, n7 outputs
        auto check1 = [&](auto& x, auto& y, const char* what) -> bool {
            if (!x.chain_launch(T, tokens, p, a, force, nf7, n7, early7, err) || !finish(x, err)) return false;
            if (!y.chain_launch(T, tokens, p, a, force, nf7, n7, early7, err) || !finish(y, err)) return false;
            for (int j = 0; j < n7; ++j) {
                ++n_cmp1;
                if (x.chain_tok()[j] != y.chain_tok()[j] || !same_bits(x.chain_prob()[j], y.chain_prob()[j]))
                    mismatch(r, what, j, x.chain_tok()[j], x.chain_prob()[j], y.chain_tok()[j], y.chain_prob()[j]);
            }
            return true;
        };
        if (use_base && !use_base2 && !check1(base, da, "base/a")) return fail_setup("check 1 chain: " + err);
        if (use_base2 && (!check1(base, base2, "base/base2") || !da.chain_launch(T, tokens, p, a, force, nf7, n7, early7, err) ||
                          !finish(da, err)))
            return fail_setup("check 1 chain: " + err);
        if (use_da2 && !check1(da2, da, "a2/a")) return fail_setup("check 1 chain: " + err);
        if (da.chain_n() != n7 || da.source_done() != nullptr) return fail_setup("a: chain_n / source_done");

        // 2. b, 11 outputs queued at once; 4. its source_done after the source copy
        cudaMemsetAsync(bound[2], 0xff, (size_t) kMaxT * HCN * sizeof(float), chk);
        cudaStreamSynchronize(chk);
        const auto tb = std::chrono::steady_clock::now();
        if (!db.chain_launch(T, tokens, p, a, force, nfb, kC, kC, err)) return fail_setup("b chain: " + err);
        if (db.source_done() == nullptr) return fail_setup("b: no source_done event");
        cudaStreamWaitEvent(chk, db.source_done(), 0);
        cudaMemcpyAsync(check_R, bound[2], (size_t) T * HCN * sizeof(float), cudaMemcpyDeviceToDevice, chk);
        if (!finish(db, err)) return fail_setup("b chain: " + err);
        ms_b += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - tb).count();
        cudaStreamSynchronize(chk);
        cudaMemcpy(chk_host.data(), check_R, (size_t) T * HCN * sizeof(float), cudaMemcpyDeviceToHost);
        ++src_checks;
        if (std::memcmp(chk_host.data(), src_host.data(), (size_t) T * HCN * sizeof(float)) != 0) {
            ++bad;
            std::fprintf(stderr, "MISMATCH round %d: the bound rows at source_done are not the source rows\n", r);
        }
        if (db.chain_n() != kC) return fail_setup("b: chain_n is not 11");
        for (int j = 0; j < n7; ++j) {
            ++n_cmp2;
            if (db.chain_tok()[j] != da.chain_tok()[j] || !same_bits(db.chain_prob()[j], da.chain_prob()[j]))
                mismatch(r, "b/a prefix", j, db.chain_tok()[j], db.chain_prob()[j], da.chain_tok()[j], da.chain_prob()[j]);
        }

        // 3. c, step by step
        const int nq = 1 + rnd(kC);
        const bool may_close = rnd(2) == 0;
        if (!dc.chain_launch(T, tokens, p, a, force, nfb, kC, kC, err, nq)) return fail_setup("c chain: " + err);
        if (!dc.chain_open()) ++c_queued_all;
        // chain_cap: what it may queue (n_out); chain_n: what is queued so far (the scheduler reads the bound from cap)
        if (dc.chain_cap() != kC || dc.chain_n() != std::min(nq, kC)) return fail_setup("c: chain_cap / chain_n at launch");
        bool closed = false;
        for (;;) {
            const int ready = dc.chain_outputs_ready(err);
            if (ready < 0) return fail_setup("c chain: " + err);
            if (dc.chain_open() && ready >= dc.chain_n() - 1) {   // the step queued ahead is all that is left
                if (may_close && rng_c() % 4 == 0) {
                    dc.chain_close();
                    closed = true;
                } else if (dc.chain_extend(ready + (int) (rng_c() % 2), err) < 0) {
                    return fail_setup("c extend: " + err);
                }
            }
            const int pr = dc.chain_poll(err);
            if (pr < 0) return fail_setup("c chain: " + err);
            if (pr == 1) break;
        }
        if (closed) ++c_closed;
        else if (nq < kC) ++c_full_ext;
        if (!closed && dc.chain_n() != kC) return fail_setup("c: an unclosed chain ended short of 11");
        for (int j = 0; j < dc.chain_n(); ++j) {
            ++n_cmp3;
            if (dc.chain_tok()[j] != db.chain_tok()[j] || !same_bits(dc.chain_prob()[j], db.chain_prob()[j]))
                mismatch(r, "c/b", j, dc.chain_tok()[j], dc.chain_prob()[j], db.chain_tok()[j], db.chain_prob()[j]);
        }
        for (int j = dc.chain_n(); j <= kC; ++j)
            if (dc.chain_tok()[j] != 0 || dc.chain_prob()[j] != 0.0f) {
                ++bad;
                std::fprintf(stderr, "MISMATCH round %d: c output %d past chain_n is not cleared\n", r, j);
            }
        p += a + 1;
    }

    // 7. the serial draft(): base and a after the same history
    int64_t n_cmp7 = 0;
    for (int r = 0; r < 8; ++r) {
        const int T = 2 + rnd(kMaxT - 1), a = rnd(T);
        int32_t tokens[kMaxT], d0[kMaxT], d1[kMaxT];
        float p0[kMaxT], p1[kMaxT];
        int n0 = 0, n1 = 0;
        for (int t = 0; t < T; ++t) tokens[t] = tok();
        const bool ok7 = use_base2 ? base.draft(T, tokens, p, a, d0, err, p0, 0.0f, &n0) &&
                                         base2.draft(T, tokens, p, a, d1, err, p1, 0.0f, &n1)
                         : use_da2 ? da2.draft(T, tokens, p, a, d0, err, p0, 0.0f, &n0) &&
                                         da.draft(T, tokens, p, a, d1, err, p1, 0.0f, &n1)
                                   : base.draft(T, tokens, p, a, d0, err, p0, 0.0f, &n0) &&
                                         da.draft(T, tokens, p, a, d1, err, p1, 0.0f, &n1);
        if (!ok7) return fail_setup("draft: " + err);
        if (n0 != n1) { ++bad; std::fprintf(stderr, "MISMATCH draft %d: %d vs %d drafts\n", r, n0, n1); }
        for (int j = 0; j < std::min(n0, n1); ++j) {
            ++n_cmp7;
            if (d0[j] != d1[j] || !same_bits(p0[j], p1[j])) mismatch(r, "draft", j, d0[j], p0[j], d1[j], p1[j]);
        }
        p += a + 1;
    }

    // 5. a's chain max is today's: 8 outputs are refused
    {
        int32_t tokens[1] = {tok()};
        std::string e;
        if (da.chain_launch(1, tokens, p, 0, nullptr, 0, kMaxT, kMaxT, e)) {
            ++bad;
            std::fprintf(stderr, "MISMATCH: a accepted an 8-output chain\n");
            finish(da, e);
        } else if (da.chain_live()) {
            return fail_setup("a: a refused chain left the drafter live");
        }
    }
    // 6. the end of the context: b at room 5 (11 asked); no error, chain_n = the room (a chain within the default
    // chain max is staged as before, unclamped, so a is not asked here)
    {
        int32_t tokens[1] = {tok()};
        int32_t force[16];
        for (int32_t& f : force) f = tok();
        const int64_t pb = o.max_cells - 6;
        if (!db.chain_launch(1, tokens, pb, 0, force, 10, kC, kC, err) || !finish(db, err)) return fail_setup("b edge: " + err);
        if (db.chain_n() != 5 || db.chain_cap() != 5) {
            ++bad;
            std::fprintf(stderr, "MISMATCH: the max_cells clamp gave b %d / cap %d (want 5)\n", db.chain_n(),
                         db.chain_cap());
        }
        for (int j = db.chain_n(); j <= kC; ++j)
            if (db.chain_tok()[j] != 0) { ++bad; std::fprintf(stderr, "MISMATCH: b edge output %d not cleared\n", j); }
    }
    const cudaError_t last = cudaDeviceSynchronize();
    if (last != cudaSuccess) return fail_setup(std::string("device: ") + cudaGetErrorString(last));

    std::printf("mtp_chain_parity: %d rounds (p %lld -> %lld)\n", o.rounds, (long long) o.prompt, (long long) p);
    const char* pair = o.ref == "base" ? "new = old code" : o.ref == "base2" ? "old = old" : "new = new";
    std::printf("  1. n_out <= 7, %-14s:        %lld outputs compared\n", pair, (long long) n_cmp1);
    std::printf("  2. 11-output prefix = n7 outputs:      %lld outputs compared (%lld rounds b forced further)\n",
                (long long) n_cmp2, (long long) b_forced_more);
    std::printf("  3. step-wise = full queue:            %lld outputs compared (%lld extended to 11, %lld closed early, "
                "%lld queued whole)\n", (long long) n_cmp3, (long long) c_full_ext, (long long) c_closed,
                (long long) c_queued_all);
    std::printf("  4. source_done after the copy:        %lld checks\n", (long long) src_checks);
    std::printf("  5/6. chain max 7 refuses 8, max_cells clamp: checked\n");
    std::printf("  7. serial draft(), %-14s:    %lld outputs compared\n", pair, (long long) n_cmp7);
    std::printf("  b: %.2f ms per 11-output chain (host wall, launch to done)\n", ms_b / std::max(1, o.rounds));
    std::printf("%s (%d mismatches)\n", bad == 0 ? "PASS" : "FAIL", bad);
    return bad == 0 ? 0 : 1;
}
