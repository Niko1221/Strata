// tests/core/glm53_geometry_test.cpp - the planner's geometry for the glm-dsa family (P1.S9, port round 1).
//
// plan.hpp was one model's constants.  The point of this test is that the glm-dsa numbers are the ones the
// checkpoint actually has, and that the qwen4exp numbers did not move.  Every expected value here is either
// measured on D:\models\GLM-5.3-colibri-int4-g64 or derived from the geometry by hand, and the two are shown
// next to each other so a disagreement is visible.  No model, no GPU.
#include "strata/plan/plan.hpp"

#include <cstdio>
#include <string>

using namespace strata::plan;

namespace {
int g_fail = 0;
void check(bool ok, const std::string& what) {
    std::printf("  %-84s %s\n", what.c_str(), ok ? "ok" : "FAIL");
    if (!ok) ++g_fail;
}
}  // namespace

int main() {
    std::printf("glm53_geometry_test\n");

    const Geometry glm = derived_glm_dsa();
    check(glm.ml_cache_elems() == 576, "the MLA cache is the 512 latent + the 64 rope slice, not an expanded key");

    // 576 elements + one fp16 scale per 64 = 594 B per token per layer; 78 layers.
    check(kv_bytes_per_token(glm) == 46332, "glm-dsa INT8 KV is 594 B/token/layer, 46,332 B/token over 78 layers");

    // The checkpoint's `indexer_types` has 21 'full' layers.  GLM-5.2/5.3 has no kpool, so the key is cached
    // per token, not per block - that is the whole difference from qsa.hpp's pooled store.
    const Geometry glm_idx = derived_glm_dsa(true);
    check(kv_bytes_per_token(glm_idx) == 46332 + 21 * 128, "with the indexer converted, 21 keys add 2,688 B/token");

    check(state_bytes(glm) == 0, "glm-dsa has no recurrent state: nothing is held for a sequence");

    // Measured on the container: 6,291,456 code bytes + 786,432 scale bytes per tensor, three tensors.
    check(glm_expert_blob() == 21233664, "the int4-g64 expert blob is 21,233,664 B, 15.4x the IQ2_XS blob");

    // The qwen4exp side must not move.  1,056 B per QSA layer + 32 B for the one cached key, x 12 layers.
    const Geometry qwen;
    check(qwen.arch == "qwen4exp", "the default geometry is still qwen4exp");
    check(kv_bytes_per_token(qwen) == 13056, "qwen4exp KV is unchanged at 13,056 B/token");
    check(state_bytes(qwen) == 117669888, "qwen4exp GDN state is unchanged at 117,669,888 B (117.7 MB)");

    // The planner's pool is a qwen4exp number.  GLM's dense weights are bigger than the whole pool, so the
    // refusal has to say that, not "reduce --max-context".
    Costs c;
    c.expert_blob = glm_expert_blob();
    c.dense_bytes = 11595965440;   // measured: analysis.dense_bytes
    std::string err;
    try {
        make_plan(32768, glm, c);
    } catch (const DoesNotClose& e) {
        err = e.what();
    }
    check(err.find("wrong pool") != std::string::npos,
          "the qwen4exp pool refuses GLM by naming the dense weights, not the context");

    // With a pool that can actually hold GLM's dense weights, the slots divide by the GLM blob:
    // 16,000,000,000 - 1,518,206,976 KV - 11,595,965,440 dense = 2,885,827,584 -> 135 slots.
    const Plan p = make_plan(32768, glm, c, 16000000000ull);
    check(p.kv_bytes == 1518206976ull, "32k context costs 1,518,206,976 B of KV on GLM");
    check(p.cache_slots == 135, "135 expert cache slots at 21,233,664 B each, not 2,086 at 1,382,400 B");

    return g_fail ? 1 : 0;
}
