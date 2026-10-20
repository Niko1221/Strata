// metal/tests/session_limits_test.cpp - the session file's read limits are the sizes the Metal window saves.
// Regression: conversation_session_read_limits sized the running-state arrays from the generic session carve
// (12 QSA layers x 1536 B of indexer tail = 18432) while the window saves 12 x 2048 B = 24576, so the engine
// refused its own file ("element count 24576 exceeds this engine's limit 18432").  No GPU, no model: the window's
// sizes are stubbed with the numbers Window::checkpoint_sizes returns.  Exit 0 = every check passed.
#include "strata/core/conversation_file.hpp"
#include "strata/core/conversation_snapshot.hpp"

#include <cstdio>
#include <string>
#include <vector>

namespace strata::metal_engine {
bool window_checkpoint_sizes(size_t& gdn, size_t& ple, size_t& tails, size_t& dead, size_t& block_pos) {
    gdn = 36 * (4 * 128 * 48 * 128 + 4 * 10240 * 3);
    ple = (size_t) 4 * 9 * 10240;
    tails = 12 * (4 * 4 * 128);
    dead = 12 * (4 * 128);
    block_pos = 0;
    return true;
}
bool window_checkpoint_save(strata::core::ConversationCheckpoint&, std::string&) { return false; }
bool window_checkpoint_restore(const strata::core::ConversationCheckpoint&, std::string&) { return false; }
}  // namespace strata::metal_engine

using namespace strata::core;

static int failures = 0;
static void check(bool ok, const char* what) {
    std::printf("%s %s\n", ok ? "ok  " : "FAIL", what);
    failures += !ok;
}

int main() {
    ModelGeometry g;
    SessionState ss;
    ss.max_cells = 4096;
    ss.layer_lo = 0; ss.layer_hi = g.n_layers;
    ss.gdn_alloc = g.n_gdn_layers();
    ss.qsa_alloc = g.n_qsa_layers();
    std::vector<QsaState> qsa((size_t) g.n_qsa_layers());
    for (auto& q : qsa) q.max_cells = ss.max_cells;
    ss.qsa_states = qsa.data();
    QsaState draft;
    draft.max_cells = ss.max_cells;

    SessionReadLimits limits;
    std::string err;
    check(conversation_session_read_limits(limits, ss, g, draft, 262144, 4, err), "read limits built");
    size_t gdn, ple, tails, dead, block_pos;
    strata::metal_engine::window_checkpoint_sizes(gdn, ple, tails, dead, block_pos);
    check(limits.max_state_bytes[0] == gdn, "limit gdn = window gdn");
    check(limits.max_state_bytes[2] == tails, "limit tails = window tails (24576)");
    check(limits.max_state_bytes[3] == dead, "limit dead = window dead");
    check(limits.max_state_bytes[4] == block_pos, "limit block_pos = window block_pos");
    std::printf("tails limit %llu\n", (unsigned long long) limits.max_state_bytes[2]);
    return failures ? 1 : 0;
}
