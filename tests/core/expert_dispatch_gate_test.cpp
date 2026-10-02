// tests/core/expert_dispatch_gate_test.cpp - the verify-window CPU dispatch's per-token activation gate.
//
// expert_pool_dispatch_multi quantizes an activation row ONLY for a token that has a CPU job (any routed
// expert with kind < 0).  The E-6 conditional doorbell publishes NO x row for an all-resident group, so an
// un-gated row would be read stale.  The consumer-visible invariants asserted here: an all-resident token
// whose x row is a SIGNALING NaN must not raise FE_INVALID in the calling thread, the valid miss token's
// expert output must be FINITE and NONZERO, and a mixed batch must produce bitwise the same miss output and
// counters as the same miss in isolation.  Arms: the packed Q2_0 layout (the else branch, skipped with a
// reason on CPUs without AVX-512-VNNI - the packed expert kernels' scalar fallback is not in the pool path)
// and the native branches via ONE manifest per process (the process-wide layout is loaded, not reset): the
// manifest file in argv[1] (e.g. tests/data/native_experts/q2_0.txt or iq2_xs.txt) runs ONE representative
// layer per distinct (gu_type, d_type) pair, so every genuinely different format of the model is covered
// exactly once.
// Source-only: the path under test makes no CUDA calls (plan == null, no remote tiers, no lookahead).
// Run: build/expert_dispatch_gate_test <manifest.txt>   (two CTest invocations: q2_0.txt, iq2_xs.txt)
#include "strata/core/expert_source.hpp"
#include "strata/kernels/cpu/expert.hpp"
#include "strata/kernels/cpu/expert_layout.hpp"
#include "strata/kernels/cpu/pool.hpp"

#include <cfenv>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace c = strata::kernels::cpu;
namespace core = strata::core;
namespace fs = std::filesystem;

namespace {

void require(bool ok, const std::string& message) {
    if (!ok) throw std::runtime_error(message);
}

struct TempDirectory {
    fs::path path;
    TempDirectory() {
        const auto stamp = std::chrono::steady_clock::now().time_since_epoch().count();
        path = fs::temp_directory_path() / ("strata-dispatch-gate-test-" + std::to_string(stamp));
        fs::create_directories(path);
    }
    ~TempDirectory() {
        std::error_code ignored;
        fs::remove_all(path, ignored);
    }
};

// ONE blob, for the sole miss expert.  Every 16-bit word is 0x3c00: an fp16 scale of 1.0 wherever the
// format reads a scale, and codes that decode to a finite value (q2_0 codes 0 -> -1 * d, etc.), so any
// route through the blob produces finite, nonzero outputs.  512 x 1.38 MB is not needed: the hits never
// touch the source.
struct FiniteSource : core::ExpertSource {
    std::vector<uint8_t> mem;
    explicit FiniteSource(int64_t blob_bytes) : mem((size_t) blob_bytes) {
        for (size_t i = 0; i < mem.size(); i += 2) {
            const uint16_t h = 0x3c00;   // fp16 1.0
            std::memcpy(&mem[i], &h, 2);
        }
    }
    const uint8_t* blob(int64_t, int64_t) override { return mem.data(); }
    void begin_layer(int64_t, const int32_t*, int64_t) override {}
};

struct Geometry {
    int64_t n_layers = 1, n_expert = 512, k = 10;
};

void fill_dispatch(core::ExpertDispatch& d, core::ExpertSource* src, c::ExpertPool* pool,
                   const int32_t* host_res, int64_t layer, const Geometry& g) {
    d.pool = pool;
    d.src = src;
    d.n_expert = g.n_expert;
    d.host_res = host_res;
    d.plan = nullptr;           // the host-plan fallback: kinds come from host_res (no GPU plan)
    d.remote_count = 0;
    d.lookahead = nullptr;
    d.split_rows = true;
    d.layers = layer;
    d.failed = false;
    d.fail = nullptr;
    d.act_multi.assign((size_t) c::MAXT, c::ActQ{});
    d.nact_multi.clear();
    d.usage.clear();            // empty: the usage loop is off
    d.multi_misses = 0;
    d.multi_entries = 0;
    d.cache_hits = 0;
    d.cache_refused = 0;
    d.experts = 0;
    d.missing = 0;
}

// token A: experts 0..9 all resident (kind >= 0), its x row a SIGNALING NaN - an un-gated quantizer would
// operate on it and raise FE_INVALID; token B: experts 10..18 resident, expert 19 a CPU miss (valid x).
void run_arm(const char* label, c::ExpertPool& pool, FiniteSource& src, const Geometry& g, int64_t layer) {
    const int64_t H = c::H;
    std::vector<int32_t> host_res((size_t) (g.n_layers * g.n_expert), -1);
    const int64_t row0 = layer * g.n_expert;
    for (int32_t e = 0; e <= 18; ++e) host_res[(size_t) (row0 + e)] = e;   // layer `layer`, residents 0..18
    // expert 19 stays -1: B's CPU miss (all twenty routed entries above are distinct).

    std::mt19937 rng(11);
    std::vector<float> x_a((size_t) H, std::numeric_limits<float>::signaling_NaN());   // stale: must not be read
    std::vector<float> x_b((size_t) H);
    // A known small finite activation (+/-1e-2, deterministic): a large activation drives the gate sums
    // to magnitude ~hundreds and the silu saturates/underflows (the miss output comes out exactly zero),
    // while a too-small row (1e-4) pushes the intermediate's q8_1 fp16 scale into the subnormal region.
    // At 1e-2 the gates stay ~0.3-0.5 (the silu is linear), the intermediate is ~0.05-0.3 and its fp16
    // scale ~2e-3 (normal), so the miss is guaranteed finite AND nonzero.
    for (float& v : x_b) v = (rng() & 1) ? 1e-2f : -1e-2f;

    const int32_t ids_a[10] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
    const int32_t ids_b[10] = {10, 11, 12, 13, 14, 15, 16, 17, 18, 19};
    int32_t ids_mixed[20];
    std::memcpy(ids_mixed, ids_a, sizeof ids_a);
    std::memcpy(ids_mixed + 10, ids_b, sizeof ids_b);

    core::ExpertDispatch dm;
    fill_dispatch(dm, &src, &pool, host_res.data(), layer, g);
    std::vector<float> x_mixed(2 * (size_t) H);
    std::memcpy(x_mixed.data(), x_a.data(), (size_t) H * 4);
    std::memcpy(x_mixed.data() + H, x_b.data(), (size_t) H * 4);
    std::vector<float> out1(2 * 10 * (size_t) H, 0.0f);
    const int64_t hits0 = dm.cache_hits, miss0 = dm.multi_misses, ent0 = dm.multi_entries;
    std::feclearexcept(FE_ALL_EXCEPT);          // the all-resident signaling-NaN row must not be operated on
    core::expert_pool_dispatch_multi(dm, x_mixed.data(), ids_mixed, 2, 10, out1.data());
    require(!dm.failed, std::string(label) + ": dispatch failed");
    require(fetestexcept(FE_INVALID) == 0,
            std::string(label) + ": the all-resident signaling-NaN row raised FE_INVALID (the gate is broken)");

    // the control run: ONLY B (the same miss expert 19)
    core::ExpertDispatch db;
    fill_dispatch(db, &src, &pool, host_res.data(), layer, g);
    std::vector<float> out2(10 * (size_t) H, 0.0f);
    const int64_t hits1 = db.cache_hits, miss1 = db.multi_misses, ent1 = db.multi_entries;
    core::expert_pool_dispatch_multi(db, x_b.data(), ids_b, 1, 10, out2.data());
    require(!db.failed, std::string(label) + ": control dispatch failed");

    // every hit output row is exactly +0.0 (rows 0..18; row 19 is the miss)
    for (size_t i = 0; i < (size_t) (19 * H); ++i)
        require(out1[i] == 0.0f, std::string(label) + ": a hit row is not zero");
    // the miss rows: FINITE, at least one known nonzero value, and bitwise equal between batch and isolation
    const float* om = out1.data() + (size_t) (19 * H);
    const float* oc = out2.data() + (size_t) (9 * H);
    int nonzero = 0;
    for (int64_t i = 0; i < H; ++i) {
        require(std::isfinite(om[i]) && std::isfinite(oc[i]), std::string(label) + ": a miss output is not finite");
        nonzero += om[i] != 0.0f;
    }
    require(nonzero > 0, std::string(label) + ": the miss expert produced only zeros (the blob is not real)");
    require(std::memcmp(om, oc, sizeof(float) * (size_t) H) == 0,
            std::string(label) + ": the miss row differs between the batch and isolation");
    // counters: 10 + 9 hits, one distinct miss, one entry, 20 routed experts
    require(dm.cache_hits - hits0 == 19 && dm.multi_misses - miss0 == 1 && dm.multi_entries - ent0 == 1 &&
            dm.experts == 20,
            std::string(label) + ": counters (hits/misses/entries/experts) differ");
    require(db.cache_hits - hits1 == 9 && db.multi_misses - miss1 == 1 && db.multi_entries - ent1 == 1,
            std::string(label) + ": control counters differ");

    std::printf("%-18s ok (no FE_INVALID from the stale row; miss output finite, nonzero and bitwise-equal "
                "to isolation; 19 hits / 1 miss)\n", label);
}

}  // namespace

int main(int argc, char** argv) {
    try {
        c::ExpertPool pool(2, false, false);   // two workers, no pinning: fine for a small batch
        Geometry g;
        bool any_run = false;

        // ARM 1: the packed Q2_0 layout (the process-wide default: native === false) - the else branch.
        // The packed expert kernels (act_quant_q8_1) are AVX-512-VNNI and the scalar fallback is only for
        // oracle/direct-kernel tests, not the pool path: on a CPU without the features THIS arm is skipped
        // with a reason; the native arms (the production path) below still run where their dispatch allows.
        if (c::cpu_features().usable()) {
            require(!strata::kernels::cpu::expert_layout().native, "packed arm: the global layout is already native");
            FiniteSource psrc((int64_t) c::BLOB);
            run_arm("packed Q2_0", pool, psrc, g, 0);
            any_run = true;
        } else {
            std::printf("packed Q2_0 arm      skipped (the packed expert kernels need AVX-512-VNNI; the scalar "
                        "fallback is not in the pool path)\n");
        }

        // ARM 2: the native branches.  The manifest file (argv[1], e.g. tests/data/native_experts/q2_0.txt
        // or iq2_xs.txt) is staged as native_experts.txt and loaded ONCE - one fixture process per manifest
        // (two CTest invocations) because the process-wide layout is loaded, not reset.  Every layer of the
        // loaded layout with gu_type 42 exercises the act_quant_any branch, every other layer the
        // native_quant_act branch; a mixed manifest covers both in one process, a single-type manifest one.
        if (argc > 1) {
            TempDirectory tmp;
            const fs::path manifest = argv[1];
            require(fs::exists(manifest), "native arm: " + manifest.string() + " is missing");
            const fs::path staged = tmp.path / "native_experts.txt";
            {
                std::ifstream in(manifest);
                std::ofstream out(staged, std::ios::binary);
                int64_t layers = 0;
                std::string line;
                while (std::getline(in, line)) {
                    out << line << '\n';
                    if (!line.empty() && line[0] != '#') ++layers;
                }
                g.n_layers = layers;   // `out` closes here, before the loader opens the staged file
            }
            require(g.n_layers > 0, "native arm: no layer lines in " + manifest.string());
            std::string err;
            // Evaluate the load's bool BEFORE building the message: in the argument list the `err` string
            // would be read before the call had a chance to write it.
            const bool loaded =
                strata::kernels::cpu::expert_layout_load(tmp.path.string(), g.n_layers, g.n_expert, err);
            require(loaded, "native arm: expert_layout_load: " + err);
            require(strata::kernels::cpu::expert_layout().native, "native arm: the load did not set native");
            const std::string base = manifest.filename().string();
            const auto& lay = strata::kernels::cpu::expert_layout();
            // One representative layer per DISTINCT (gu_type, d_type) pair: the model's 48 layers repeat a
            // handful of formats, and running the same path 48 times adds nothing the first run did not.
            std::vector<std::pair<int64_t, int64_t>> distinct;
            for (int64_t l = 0; l < lay.n_layers; ++l) {
                const std::pair<int64_t, int64_t> pair(lay.fmt[(size_t) l].gu_type,
                                                      lay.fmt[(size_t) l].d_type);
                bool seen = false;
                for (const auto& p : distinct)
                    if (p == pair) { seen = true; break; }
                if (seen) continue;
                distinct.push_back(pair);
                FiniteSource ns((int64_t) lay.blob_bytes(l));
                const std::string lbl = "native " + base + " gu" + std::to_string(pair.first) +
                                        "/d" + std::to_string(pair.second);
                run_arm(lbl.c_str(), pool, ns, g, l);
                any_run = true;
            }
            if (distinct.empty())
                std::printf("native %-20s skipped (no layers in this manifest)\n", base.c_str());
        } else {
            std::printf("native arms         skipped (pass a native_experts manifest file, e.g. "
                        "tests/data/native_experts/q2_0.txt or iq2_xs.txt, to test the native branches)\n");
        }
        if (!any_run) {
            std::printf("expert_dispatch_gate_test: SKIP (no arm could run on this machine)\n");
            return 77;
        }
        std::printf("expert_dispatch_gate_test: OK\n");
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "expert_dispatch_gate_test: %s\n", e.what());
        return 1;
    }
}
