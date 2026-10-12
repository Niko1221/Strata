// Real CUDA storage and real image ownership. Pattern data is not model inference.
#include "strata/core/branch_handles.hpp"
#include "strata/core/branch_session.hpp"
#include "strata/core/conversation_snapshot.hpp"
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

using namespace strata::core;
static int checks = 0;
static void check(bool ok, const char* message) {
    ++checks;
    if (!ok) { std::fprintf(stderr, "FAIL: %s\n", message); std::exit(1); }
}
static void gpu(cudaError_t result) { check(result == cudaSuccess, cudaGetErrorString(result)); }

static void exercise(int format) {
    ModelGeometry g;
    g.n_layers = 4;
    g.ssm_state_size = 2; g.ssm_v_heads = 2; g.ssm_conv_channels = 8;
    qsa_set_kv_int8(format == 1);
    qsa_set_kv_q4(format == 2);
    qsa_set_kv_hybrid(format == 3);
    qsa_set_kv_elastic(false, 0);
    qsa_set_kv_resident(0);
    SessionState state;
    QsaState q;
    state.max_cells = 64; state.layer_hi = 4;
    state.gdn_alloc = 3; state.qsa_alloc = 1; state.qsa_states = &q;
    void* arena = nullptr;
    gpu(cudaMalloc(&arena, qsa_state_bytes(g, state.max_cells)));
    check(qsa_state_init(g, state.max_cells, arena, q) != 0, "initialize actual QSA pools");
    qsa_state_zero(q, g, nullptr);
    ConversationStateSizes sizes;
    std::string error;
    check(conversation_state_sizes(g, sizes, error), "measure running-state layout");
    gpu(cudaMalloc(&state.gdn_state, sizes.gdn));
    gpu(cudaMalloc(&state.ple_hist, sizes.ple));
    gpu(cudaMemset(state.gdn_state, 17, sizes.gdn));
    gpu(cudaMemset(state.ple_hist, 19, sizes.ple));
    const auto pages_before = BranchPages::counters();
    const auto snapshots_before = BranchSession::snapshot_counters();
    const auto digest_before = branch_digest_counters();
    BranchHandleCounters handles_work;
    {
        BranchSession session(state, g, "lifecycle-fixture");
        BranchHandles<BranchSessionImage> handles(handles_work);
        check(session.bind(error), error.c_str());
        check(session.begin(0, 9, error), error.c_str());
        for (const auto& pool : session.pools()) gpu(cudaMemset(pool.address, 23, pool.bytes));
        check(session.finish(std::vector<int32_t>(9, 42), {1.0f, -2.0f}, error), error.c_str());
        const auto root = handles.insert(session.freeze(error));
        const auto address = session.pools().front().address;
        const auto original_pages = session.ownership();
        const auto page_baseline = BranchPages::counters();
        const auto snapshot_baseline = BranchSession::snapshot_counters();
        const auto handle_baseline = handles_work;
        check(snapshot_baseline.live_images == snapshots_before.live_images + 1, "one actual root image");
        check(snapshot_baseline.payload_capacity_bytes > snapshots_before.payload_capacity_bytes, "root host payload measured");
        uint64_t last = root;
        for (int cycle = 0; cycle < 64; ++cycle) {
            for (int branch = 0; branch < 2; ++branch) {
                check(session.restore(*handles.find(root), error), error.c_str());
                check(session.begin(9, 1, error), error.c_str());
                for (const auto& pool : session.pools()) {
                    const auto cell = pool.pooled ? 9 / strata::kernels::qsa_real_shapes().idx_block : 9;
                    gpu(cudaMemset(static_cast<unsigned char*>(pool.address) + cell * pool.bytes_per_cell,
                                   31 + branch, pool.bytes_per_cell));
                }
                gpu(cudaMemset(state.gdn_state, 41 + branch, sizes.gdn));
                check(session.finish({71 + branch}, {float(branch), float(cycle)}, error), error.c_str());
                const auto child = handles.insert(session.freeze(error));
                check(child > last, "released handles are never reused");
                last = child;
                check(BranchSession::snapshot_counters().live_images == snapshot_baseline.live_images + 1,
                      "one independently owned child image");
                check(session.restore(*handles.find(root), error), error.c_str());
                check(handles.drop(child), "release actual child");
                check(handles.drop(child), "release retry is idempotent");
                check(handles.find(child) == nullptr, "stale handle cannot restore a later image");
                const auto pages = BranchPages::counters();
                const auto snapshots = BranchSession::snapshot_counters();
                check(pages.live_bytes == page_baseline.live_bytes, "physical pages return to root baseline");
                check(pages.allocations - page_baseline.allocations == pages.releases - page_baseline.releases,
                      "every child physical allocation was released");
                check(pages.mappings - page_baseline.mappings == pages.unmappings - page_baseline.unmappings,
                      "temporary page mappings balance");
                check(pages.address_reservations - page_baseline.address_reservations ==
                      pages.address_releases - page_baseline.address_releases, "virtual addresses do not accumulate");
                check(snapshots.live_images == snapshot_baseline.live_images, "image owners return to baseline");
                check(snapshots.payload_capacity_bytes == snapshot_baseline.payload_capacity_bytes,
                      "owned checkpoint payload capacity returns to baseline");
                check(snapshots.allocations - snapshot_baseline.allocations == snapshots.releases - snapshot_baseline.releases,
                      "image constructions and destructions balance");
                check(handles_work.live_bytes == handle_baseline.live_bytes, "handle allocations return to baseline");
                check(session.pools().front().address == address, "rewind preserves graph addresses");
                const auto current = session.ownership();
                check(current.size() == original_pages.size(), "root page extent retained");
                for (size_t i = 0; i < current.size(); ++i)
                    check(current[i].allocation_id == original_pages[i].allocation_id, "root owns its original physical page");
                check(session.ids() == std::vector<int32_t>(9, 42) && session.logits() == std::vector<float>({1.0f, -2.0f}),
                      "root consumed frontier survives every cycle");
            }
        }
        handles.clear();
        check(handles_work.live_entries == 0 && handles_work.live_bytes == 0, "all handle nodes freed before teardown");
        check(BranchSession::snapshot_counters().live_images == snapshots_before.live_images, "all image owners released");
        check(BranchSession::snapshot_counters().payload_capacity_bytes == snapshots_before.payload_capacity_bytes,
              "all checkpoint payload buffers released");
        check(session.reset(error), error.c_str());
        check(session.ids().empty() && session.logits().empty(), "reset removes consumed state");
    }
    const auto pages_after = BranchPages::counters();
    const auto snapshots_after = BranchSession::snapshot_counters();
    const auto digest_after = branch_digest_counters();
    check(pages_after.live_bytes == pages_before.live_bytes, "session teardown frees every derived page");
    check(pages_after.allocations - pages_before.allocations == pages_after.releases - pages_before.releases,
          "session page allocation calls balance");
    check(pages_after.mappings - pages_before.mappings == pages_after.unmappings - pages_before.unmappings,
          "session mappings balance after destruction");
    check(pages_after.address_reservations - pages_before.address_reservations ==
          pages_after.address_releases - pages_before.address_releases, "session addresses balance after destruction");
    check(digest_after.workspace_live_bytes == digest_before.workspace_live_bytes, "CUDA digest scratch released");
    check(digest_after.workspace_allocations - digest_before.workspace_allocations ==
          digest_after.workspace_releases - digest_before.workspace_releases, "scratch allocation calls balance");
    check(snapshots_after.allocations - snapshots_before.allocations == snapshots_after.releases - snapshots_before.releases,
          "all captured images destroyed before CUDA context exit");
    check(handles_work.allocations == handles_work.releases, "all handle allocation calls balance");
    std::printf("{\"format\":%d,\"cycles\":64,\"children\":128,\"images_allocated\":%llu,\"images_released\":%llu,"
                "\"live_images\":%llu,\"payload_capacity_bytes\":%llu,\"page_allocations\":%llu,\"page_releases\":%llu,"
                "\"live_page_bytes\":%llu,\"handle_peak_bytes\":%llu,\"handle_live_bytes\":%llu}\n", format,
                (unsigned long long)(snapshots_after.allocations - snapshots_before.allocations),
                (unsigned long long)(snapshots_after.releases - snapshots_before.releases),
                (unsigned long long)snapshots_after.live_images, (unsigned long long)snapshots_after.payload_capacity_bytes,
                (unsigned long long)(pages_after.allocations - pages_before.allocations),
                (unsigned long long)(pages_after.releases - pages_before.releases), (unsigned long long)pages_after.live_bytes,
                (unsigned long long)handles_work.peak_bytes, (unsigned long long)handles_work.live_bytes);
    gpu(cudaFreeHost(q.host_step)); gpu(cudaFreeHost(q.host_pos));
    gpu(cudaFree(state.gdn_state)); gpu(cudaFree(state.ple_hist)); gpu(cudaFree(arena));
}

int main() {
    gpu(cudaSetDevice(0));
    for (int format = 0; format < 4; ++format) exercise(format);
    std::printf("{\"status\":\"PASS\",\"checks\":%d,\"formats\":4,\"model_forward\":false,\"gpu_used\":true}\n", checks);
}
