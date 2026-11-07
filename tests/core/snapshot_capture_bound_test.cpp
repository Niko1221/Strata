// Actual capture/estimator code with host copies through the PRIVATE CUDA shim.
// No native GPU or model inference is exercised. GPU-only restore kernels must not run.
#include "strata/core/conversation_snapshot.hpp"
#include "strata/core/checkpoint_allocation.hpp"
#include "strata/kernels/kv_stream.hpp"
#include <stdexcept>
#include <cstdio>
#include <cstdlib>
#include "../../src/core/conversation_copy.hpp"
using ConvCheckpoint = strata::core::ConversationCheckpoint;
// Generated from the unchanged production CheckpointPool class, not a mock.
#include "checkpoint_pool_under_test.hpp"
namespace strata::core {
strata::kernels::QsaAttnPools qsa_attn_pools(const QsaState&) { std::abort(); }
}
namespace strata::kernels {
void kv_stream_reset(const KvStreamMap&, void*) { std::abort(); }
void kv_ring_restore(const QsaAttnPools&, const KvHostPools&, int, int64_t, int64_t, int64_t, const QsaShapes&, void*) { std::abort(); }
}
using namespace strata::core;
#define CHECK(x) do { if (!(x)) throw std::runtime_error(#x); } while (0)
int main() {
    try {
        ModelGeometry g; g.n_layers = 3; g.qsa_interval = 4;
        g.ssm_state_size = 2; g.ssm_v_heads = 2; g.ssm_conv_channels = 8;
        ConversationStateSizes z; std::string error;
        CHECK(conversation_state_sizes(g, z, error));
        std::vector<uint8_t> running(z.gdn, 37);
        SessionState ss; ss.max_cells = 96; ss.layer_hi = 3; ss.gdn_alloc = 3;
        ss.gdn_state = reinterpret_cast<float*>(running.data());
        // Pre-patch HEAD 54828c7: checkpoint_save resizes directly, without
        // reserve. Compare to that std::vector operation, not another patched
        // build. A nonempty vector makes resize's geometric growth observable.
        ConversationCheckpoint partial;
        const size_t old = z.gdn * 3 / 4;
        partial.gdn.resize(old);
        std::vector<uint8_t> baseline(old);
        baseline.resize(z.gdn);
        CHECK(conversation_checkpoint_save(partial, ss, g, error));
        CHECK(partial.gdn.capacity() == (ContextLockPolicy::enabled() ? z.gdn : baseline.capacity()));
        CHECK(std::equal(running.begin(), running.end(), partial.gdn.begin()));
        // Baseline copy_checkpoint_into cleared the stage directory, then pushed
        // parts without reserve. Retained directory capacity must grow identically
        // at zero/unset, while selected context uses its exact charged reservation.
        ConversationCheckpoint nested_source, nested_destination;
        nested_source.stage_parts.resize(4); nested_destination.stage_parts.resize(3);
        std::vector<ConversationCheckpoint> baseline_parts(3); baseline_parts.clear();
        for (int i = 0; i < 4; ++i) baseline_parts.emplace_back();
        const auto nested_bound = checkpoint_copy_bound(nested_destination, nested_source);
        conversation_detail::copy_checkpoint_into(nested_destination, nested_source);
        CHECK(nested_destination.stage_parts.capacity() ==
              (ContextLockPolicy::enabled() ? 4 : baseline_parts.capacity()));
        CHECK(nested_destination.bytes() <= nested_bound.resident);
        // The baseline pool clears marks but retains payload backing and returns
        // unused storage. Exercise the actual production class and real capture.
        CheckpointPool checkpoint_pool(true, 3);
        for (int i = 0; i < 3; ++i) {
            ConversationCheckpoint c; c.gdn.resize(z.gdn); c.gdn.reserve(65536);
            c.ids = {1, 2}; c.used = 8; c.pinned = true;
            checkpoint_pool.give(std::move(c));
        }
        CHECK(checkpoint_pool.held() == 3);
        auto returned = checkpoint_pool.take();
        CHECK(returned.ids.empty() && returned.used == 0 && !returned.pinned);
        const auto* retained = returned.gdn.data();
        checkpoint_pool.give(std::move(returned));
        returned = checkpoint_pool.take(); CHECK(returned.gdn.data() == retained);
        checkpoint_pool.give(std::move(returned));
        const std::vector<int32_t> ids{1, 2, 3}; const std::vector<ConversationImageKey> imgs;
        std::vector<ConversationCheckpoint> copies(1), moved(1), pool(2);
        copies[0].ids = {1, 2}; moved[0].ids = {1};
        CHECK(conversation_checkpoint_save(copies[0], ss, g, error));
        CHECK(conversation_checkpoint_save(moved[0], ss, g, error));
        moved[0].gdn.reserve(65536);
        for (auto& c : pool) { c.gdn.reserve(65536); c.ple.reserve(16384); c.stage_parts.reserve(3); }
        const ConversationView view{ids, imgs, copies, true};
        ConversationCaptureBound bound;
        CHECK(conversation_snapshot_allocation_bound({}, view, ss, g, nullptr, pool, moved, bound, error));
        size_t old_estimate = 0;
        CHECK(conversation_snapshot_bytes(view, ss, g, nullptr, old_estimate, error));
        CHECK(bound.resident > old_estimate);
        SavedConversation image;
        CHECK(conversation_snapshot_save(image, view, ss, g, nullptr, error, {}, nullptr, &pool, &moved));
        CHECK(pool.empty() && image.checkpoints.size() == 2);
        CHECK(image.bytes() == bound.resident);
        CHECK(image.live.gdn.size() == running.size());
        CHECK(std::equal(running.begin(), running.end(), image.live.gdn.begin()));
        CHECK(conversation_snapshot_validate(image, ss, g, nullptr, error));
        ConversationCache exact(bound.resident, 1), short_budget(bound.resident - 1, 1);
        CHECK(exact.make_room(bound.resident, 0)); CHECK(!short_budget.make_room(bound.resident, 0));
        CHECK(exact.put(std::move(image)));
        // Live replacement must charge the whole new allocation beside old backing.
        std::vector<ConversationCheckpoint> small(1); small[0].gdn.resize(1);
        ConversationCaptureBound growth;
        CHECK(conversation_snapshot_allocation_bound({}, view, ss, g, nullptr, small, {}, growth, error));
        CHECK(growth.additional >= context_allocation_bytes(z.gdn));
        SavedConversation grown;
        CHECK(conversation_snapshot_save(grown, view, ss, g, nullptr, error, {}, nullptr, &small));
        CHECK(grown.bytes() <= growth.resident);
        // In particular the geometric disabled-path live resize must also be
        // bounded, not merely the selected-context exact reserve.
        std::vector<ConversationCheckpoint> geometric(1);
        geometric[0].gdn.resize(old);
        ConversationCaptureBound geometric_bound;
        CHECK(conversation_snapshot_allocation_bound({}, view, ss, g, nullptr, geometric, {}, geometric_bound, error));
        SavedConversation geometric_image;
        CHECK(conversation_snapshot_save(geometric_image, view, ss, g, nullptr, error, {}, nullptr, &geometric));
        CHECK(geometric_image.live.gdn.capacity() == (ContextLockPolicy::enabled() ? z.gdn : baseline.capacity()));
        CHECK(geometric_image.bytes() <= geometric_bound.resident);
        SavedConversation unpublished;
        const auto original = ss.gdn_state; ss.gdn_state = nullptr;
        CHECK(!conversation_snapshot_save(unpublished, view, ss, g, nullptr, error));
        CHECK(unpublished.bytes() == 0); ss.gdn_state = original;
        // Moved checkpoint directory growth is also baseline-dependent: reserve
        // copies only, then push moves (versus exact combined reserve when selected).
        std::vector<ConversationCheckpoint> more_moves(2), spare;
        more_moves[0].ids = {1}; more_moves[0].gdn.resize(z.gdn, 19);
        more_moves[1].ids = {1, 2}; more_moves[1].gdn.resize(z.gdn, 23);
        for (int i = 0; i < 3; ++i) spare.push_back(checkpoint_pool.take());
        std::vector<ConversationCheckpoint> baseline_directory;
        baseline_directory.reserve(copies.size());
        baseline_directory.resize(copies.size());
        baseline_directory.emplace_back(); baseline_directory.emplace_back();
        ConversationCaptureBound moved_bound;
        CHECK(conversation_snapshot_allocation_bound({}, view, ss, g, nullptr, spare, more_moves, moved_bound, error));
        SavedConversation moved_image;
        CHECK(conversation_snapshot_save(moved_image, view, ss, g, nullptr, error, {}, nullptr, &spare, &more_moves));
        CHECK(moved_image.checkpoints.capacity() == (ContextLockPolicy::enabled() ? 3 : baseline_directory.capacity()));
        CHECK(moved_image.bytes() <= moved_bound.resident);
        CHECK(spare.size() == 1);
        const auto* unused = spare[0].gdn.data();
        if (!ContextLockPolicy::enabled()) {
            for (auto& c : spare) checkpoint_pool.give(std::move(c));
            CHECK(checkpoint_pool.held() == 1);
            CHECK(checkpoint_pool.take().gdn.data() == unused);
        } else {
            CHECK(moved_image.bytes() <= moved_bound.resident);
            const auto* moved_backing = moved_image.checkpoints[copies.size()].gdn.data();
            {
                CapturedCheckpointMoveRollback restore{moved_image, more_moves, copies.size(), true};
                ConversationCache reject(moved_image.bytes() - 1, 1);
                CHECK(!reject.put(std::move(moved_image))); // refused image remains recoverable
            }
            CHECK(more_moves[0].gdn.data() == moved_backing && more_moves[0].gdn[0] == 19);
            CHECK(more_moves[1].ids == std::vector<int32_t>({1, 2}) && more_moves[1].gdn[0] == 23);
            // A copy failure AFTER moving must restore ownership, not only avoid
            // partial-image publication. Fail the host transfer AFTER metadata
            // validation and checkpoint movement, through the private test shim.
            std::vector<ConversationCheckpoint> rollback(1);
            rollback[0].ids = {1}; rollback[0].gdn.resize(z.gdn, 42);
            const auto* owned = rollback[0].gdn.data();
            strata_test_copy_error = cudaErrorMemoryAllocation;
            CHECK(!conversation_snapshot_save(unpublished, view, ss, g, nullptr, error, {}, nullptr, nullptr, &rollback));
            CHECK(rollback[0].ids == std::vector<int32_t>{1});
            CHECK(rollback[0].gdn.data() == owned && rollback[0].gdn[0] == 42);
            strata_test_copy_error = cudaSuccess;
        }
        std::puts("snapshot admission/capture backing test passed (host transfers mocked)");
        return 0;
    } catch (const std::exception& e) { std::fprintf(stderr, "%s\n", e.what()); return 1; }
}
