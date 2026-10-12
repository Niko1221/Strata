#include "strata/core/branch_driver.hpp"
#include "strata/core/expert_source.hpp"
#include "strata/core/branch_session.hpp"
#include "strata/core/branch_handles.hpp"
#include "strata/core/conversation_cache.hpp"
#include <exception>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#if defined(__GLIBC__)
#include <malloc.h>
#endif

namespace strata::core {
namespace {
void error_reply(const std::string& message) {
    std::cout << "{\"status\":\"error\",\"message\":\"";
    for (unsigned char c : message) {
        if (c == '"' || c == '\\') std::cout << '\\';
        std::cout << (c < 32 ? ' ' : (char)c);
    }
    std::cout << "\"}" << std::endl;
}
}
int branch_driver(const WeightTable& weights, const ModelGeometry& geometry, SessionState& state,
                  const VerifyHits& hits, const NativeHead* head, PoolMultiFn pool, void* user,
                  ExpertDispatch& dispatch, int pcie_mode) {
    std::string error;
    // Images never leave this process; model identity is fixed by its loaded
    // weights. The API validates the external model/build contract before launch.
    BranchSession session(state, geometry, "loaded-model-single-process-v1");
    if (!session.bind(error)) { error_reply(error); return 1; }
    Verifier forward;
    // Strata requires capacity for at least two rows even when run() consumes
    // just one. The actual forced forward below always passes T=1.
    if (!forward.init(weights, geometry, state, hits, head, 2, error)) { error_reply(error); return 1; }
    // The verifier consumes the GPU plan produced by this exact dispatch.
    // Without this binding, resident experts are omitted from both CPU and GPU work.
    forward.set_pcie_mode(pcie_mode);
    dispatch.plan = forward.plan_sink();
    struct PlanReset {
        ExpertDispatch& dispatch;
        ~PlanReset() { dispatch.plan = nullptr; }
    } plan_reset{dispatch};
    bool& pool_failed = dispatch.failed;
    forward.set_head_sampling(false); // forced inputs; the API owns RNG/grammar
    BranchHandleCounters handle_work;
    BranchHandles<BranchSessionImage> images(handle_work);
    ConversationCache ram_cache(size_t(1024) * 1024 * 1024, 8);
    uint64_t ram_hits = 0, ram_misses = 0;
    std::vector<float> logits((size_t)forward.vocab());
    std::cout << "{\"status\":\"ready\",\"vocab\":" << forward.vocab()
              << ",\"capacity\":" << state.max_cells << "}" << std::endl;
    std::string line;
    while (std::getline(std::cin, line)) {
        std::istringstream in(line);
        std::string command;
        in >> command;
        error.clear();
        if (command == "quit") {
            std::cout << "{\"status\":\"closed\"}" << std::endl;
            return 0;
        }
        if (command == "reset") {
            if (!session.reset(error)) { error_reply(error); continue; }
            std::cout << "{\"status\":\"ok\",\"consumed\":0}" << std::endl;
        } else if (command == "feed") {
            std::vector<int32_t> tokens;
            int64_t token;
            bool valid = true;
            while (in >> token) {
                if (token < 0 || token >= forward.vocab()) { valid = false; break; }
                tokens.push_back((int32_t)token);
            }
            if (!valid || !in.eof() || tokens.empty() || tokens.size() > (size_t)state.max_cells - session.ids().size()) {
                error_reply("invalid token list or context capacity"); continue;
            }
            bool ok = true;
            for (int32_t t : tokens) {
                const auto position = (int64_t)session.ids().size();
                int32_t unused_sample = 0;
                pool_failed = false;
                if (!session.begin(position, 1, error) ||
                    !forward.run(1, &t, position, pool, user, &unused_sample, error) || pool_failed ||
                    !forward.commit(1, error) || !forward.copy_logits(0, logits.data()) ||
                    !session.finish({t}, logits, error)) { ok = false; break; }
            }
            if (!ok) { error_reply(error.empty() ? "native forward or recurrent commit failed" : error); continue; }
            std::cout << "{\"status\":\"ok\",\"consumed\":" << session.ids().size() << "}" << std::endl;
        } else if (command == "capture") {
            auto image = session.freeze(error);
            if (!image) { error_reply(error); continue; }
            try {
                const auto handle = images.insert(std::move(image));
                std::cout << "{\"status\":\"ok\",\"handle\":" << handle
                          << ",\"consumed\":" << session.ids().size() << "}" << std::endl;
            } catch (const std::exception& e) { error_reply(e.what()); }
        } else if (command == "restore" || command == "drop") {
            uint64_t index;
            std::string extra;
            if (!(in >> index) || (in >> extra)) {
                error_reply("missing derived checkpoint handle"); continue;
            }
            if (command == "drop") {
                if (!images.drop(index)) { error_reply("missing derived checkpoint handle"); continue; }
            } else {
                const auto* image = images.find(index);
                if (!image) { error_reply("missing derived checkpoint handle"); continue; }
                if (!session.restore(*image, error)) { error_reply(error); continue; }
            }
            std::cout << "{\"status\":\"ok\",\"consumed\":" << session.ids().size() << "}" << std::endl;
        } else if (command == "logits") {
            std::string path, extra;
            if (!(in >> path) || (in >> extra) || session.logits().empty()) { error_reply("missing path or frontier"); continue; }
            std::ofstream out(path, std::ios::binary | std::ios::trunc);
            out.write((const char*)session.logits().data(), (std::streamsize)session.logits().size()*sizeof(float));
            out.close();
            if (!out) { error_reply("frontier write failed"); continue; }
            std::cout << "{\"status\":\"ok\",\"bytes\":" << session.logits().size()*sizeof(float) << "}" << std::endl;
        } else if (command == "park") {
            std::string extra;
            size_t estimate = 0;
            if ((in >> extra) || !session.host_capture_bytes(estimate, error)) {
                error_reply(error.empty() ? "park takes no arguments" : error); continue;
            }
            if (!ram_cache.make_room(estimate)) { error_reply("RAM checkpoint admission refused"); continue; }
            SavedConversation image;
            if (!session.capture_host(image, error)) { error_reply(error); continue; }
            if (!ram_cache.put(std::move(image))) { error_reply("RAM checkpoint admission refused"); continue; }
            const auto metrics = ram_cache.storage_metrics();
            std::cout << "{\"status\":\"ok\",\"consumed\":" << session.ids().size()
                      << ",\"bytes\":" << ram_cache.bytes() << ",\"payload_bytes\":" << metrics.payload_bytes
                      << ",\"entries\":" << ram_cache.size() << "}" << std::endl;
        } else if (command == "activate") {
            std::string path, extra;
            if (!(in >> std::quoted(path)) || (in >> extra)) { error_reply("activate requires exact history path"); continue; }
            std::ifstream input(path);
            std::vector<int32_t> ids;
            int64_t token;
            bool valid = bool(input);
            while (input >> token) {
                if (token < 0 || token >= forward.vocab() || ids.size() >= (size_t)state.max_cells) {
                    valid = false; break;
                }
                ids.push_back((int32_t)token);
            }
            if (!valid || !input.eof() || ids.empty()) { error_reply("invalid exact RAM history"); continue; }
            const auto* image = ram_cache.exact(ids, {}, false);
            if (!image) {
                ++ram_misses;
                std::cout << "{\"status\":\"ok\",\"hit\":false}" << std::endl;
                continue;
            }
            if (image->next_logits.size() != (size_t)forward.vocab() ||
                !session.restore_host(*image, ids, image->next_logits, error)) {
                error_reply(error.empty() ? "invalid cached frontier extent" : error); continue;
            }
            ++ram_hits;
            std::cout << "{\"status\":\"ok\",\"hit\":true,\"consumed\":" << session.ids().size() << "}" << std::endl;
        } else if (command == "ram_clear") {
            const auto count = ram_cache.clear();
            std::cout << "{\"status\":\"ok\",\"removed\":" << count << "}" << std::endl;
        } else if (command == "ram_stats") {
            const auto metrics = ram_cache.storage_metrics();
            std::cout << "{\"status\":\"ok\",\"bytes\":" << ram_cache.bytes()
                      << ",\"entries\":" << ram_cache.size() << ",\"hits\":" << ram_hits << ",\"misses\":" << ram_misses
                      << ",\"payload_bytes\":" << metrics.payload_bytes << ",\"payload_allocations\":" << metrics.allocations
                      << ",\"payload_owners\":" << metrics.owners << ",\"copied_bytes\":" << metrics.copied_bytes << "}" << std::endl;
        } else if (command == "write") {
            std::string path, extra;
            if (!(in >> std::quoted(path)) || (in >> extra)) { error_reply("missing checkpoint path"); continue; }
            size_t bytes = 0;
            if (!session.write(path, bytes, error)) { error_reply(error); continue; }
            std::cout << "{\"status\":\"ok\",\"bytes\":" << bytes << "}" << std::endl;
        } else if (command == "load") {
            std::string path, ids_path, logits_path, extra;
            if (!(in >> std::quoted(path) >> std::quoted(ids_path) >> std::quoted(logits_path)) || (in >> extra)) {
                error_reply("cold load requires state, exact history and frontier paths"); continue;
            }
            std::ifstream inputs(ids_path);
            std::vector<int32_t> ids;
            int64_t token;
            bool valid = bool(inputs);
            while (inputs >> token) {
                if (token < 0 || token >= forward.vocab() || ids.size() >= (size_t)state.max_cells) {
                    valid = false; break;
                }
                ids.push_back((int32_t)token);
            }
            std::ifstream frontier(logits_path, std::ios::binary | std::ios::ate);
            const auto bytes = (std::streamsize)forward.vocab()*sizeof(float);
            if (!valid || !inputs.eof() || ids.empty() || !frontier || frontier.tellg() != bytes) {
                error_reply("invalid cold history or frontier extent"); continue;
            }
            std::vector<float> next_logits((size_t)forward.vocab());
            frontier.seekg(0);
            frontier.read((char*)next_logits.data(), bytes);
            if (!frontier) { error_reply("incomplete cold frontier"); continue; }
            if (!session.load(path, ids, next_logits, error)) { error_reply(error); continue; }
            std::cout << "{\"status\":\"ok\",\"consumed\":" << session.ids().size() << "}" << std::endl;
        } else if (command == "audit") {
            uint64_t index;
            std::string path, extra;
            if (!(in >> index >> std::quoted(path)) || (in >> extra)) {
                error_reply("audit requires a retained handle and output path"); continue;
            }
            const auto* image = images.find(index);
            if (!image) { error_reply("audit requires a retained handle and output path"); continue; }
            BranchSessionAudit result;
            if (!session.audit(*image, path, result, error)) { error_reply(error); continue; }
            std::cout << "{\"status\":\"ok\",\"bytes\":" << result.bytes << ",\"consumed\":" << result.consumed
                      << ",\"qsa_page_cells\":" << result.qsa_page_cells
                      << ",\"index_block_cells\":" << result.index_block_cells
                      << ",\"allocation_page_bytes\":" << vmm_granularity() << ",\"regions\":[";
            bool first = true;
            for (const auto& r : result.regions) {
                if (!first) std::cout << ',';
                first = false;
                std::cout << "{\"kind\":\"" << r.kind << "\",\"offset\":" << r.offset << ",\"bytes\":" << r.bytes
                          << ",\"pool\":" << r.pool << ",\"qsa_ordinal\":" << r.qsa_ordinal
                          << ",\"pool_offset\":" << r.pool_offset << ",\"bytes_per_cell\":" << r.bytes_per_cell
                          << ",\"filled_prefix_bytes\":" << r.filled_prefix_bytes
                          << ",\"allocation_id\":" << r.page.allocation_id << ",\"driver_handle\":" << r.page.driver_handle
                          << ",\"owners\":" << r.page.owners << '}';
            }
            std::cout << "]}" << std::endl;
        } else if (command == "pages") {
            std::cout << "{\"status\":\"ok\",\"pages\":[";
            bool first = true;
            for (const auto& page : session.ownership()) {
                if (!first) std::cout << ',';
                first = false;
                std::cout << '[' << page.allocation_id << ',' << page.driver_handle << ',' << page.owners << ',' << page.bytes
                          << ",\"" << page.kind << "\"," << page.pool << ']';
            }
            std::cout << "]}" << std::endl;
        } else if (command == "stats") {
            const auto p = BranchPages::counters();
            const auto s = session.counters();
            const auto e = vmm_failure_counters();
            const auto h = branch_digest_counters();
            const auto snapshots = BranchSession::snapshot_counters();
            std::cout << "{\"status\":\"ok\",\"consumed\":" << session.ids().size()
                      << ",\"forward_tokens\":" << forward.completed_forward_tokens
                      << ",\"windows\":" << forward.windows
              << ",\"resident_expert_entries\":" << dispatch.cache_hits
              << ",\"cpu_expert_entries\":" << dispatch.cache_refused
              << ",\"pcie_expert_groups\":" << dispatch.pcie_experts
                      << ",\"allocations\":" << p.allocations << ",\"releases\":" << p.releases
                      << ",\"live_bytes\":" << p.live_bytes << ",\"cow_pages\":" << p.cow_pages
                      << ",\"copied_bytes\":" << p.copied_bytes << ",\"mappings\":" << p.mappings
                      << ",\"unmappings\":" << p.unmappings
                      << ",\"address_reservations\":" << p.address_reservations
                      << ",\"address_releases\":" << p.address_releases
                      << ",\"vmm_driver_failures\":" << e.driver_failures
                      << ",\"vmm_out_of_memory_failures\":" << e.out_of_memory_failures
                      << ",\"vmm_last_driver_error\":" << e.last_error
                      << ",\"vmm_last_failed_operation\":\"" << e.last_operation << "\""
                      << ",\"audit_read_bytes\":" << p.audit_read_bytes
                      << ",\"integrity_checked_pages\":" << p.integrity_checked_pages
                      << ",\"integrity_mismatches\":" << p.integrity_mismatches
                      << ",\"integrity_device_hashed_bytes\":" << h.device_hashed_bytes
                      << ",\"integrity_digest_copy_bytes\":" << h.digest_copy_bytes
                      << ",\"integrity_kernel_calls\":" << h.kernel_calls
                      << ",\"integrity_workspace_allocations\":" << h.workspace_allocations
                      << ",\"integrity_workspace_releases\":" << h.workspace_releases
                      << ",\"integrity_workspace_live_bytes\":" << h.workspace_live_bytes
                      << ",\"binding_copy_bytes\":" << s.binding_copy_bytes
                      << ",\"saved_running_bytes\":" << s.saved_running_bytes
                      << ",\"restored_running_bytes\":" << s.restored_running_bytes
                      << ",\"written_file_bytes\":" << s.written_file_bytes
                      << ",\"loaded_file_bytes\":" << s.loaded_file_bytes
                      << ",\"captures\":" << s.captures << ",\"restores\":" << s.restores
                      << ",\"handle_allocations\":" << handle_work.allocations
                      << ",\"handle_releases\":" << handle_work.releases
                      << ",\"handle_live_entries\":" << handle_work.live_entries
                      << ",\"handle_live_bytes\":" << handle_work.live_bytes
                      << ",\"handle_peak_bytes\":" << handle_work.peak_bytes
                      << ",\"handle_issued\":" << handle_work.issued
                      << ",\"handle_lookup_steps\":" << handle_work.lookup_steps
                      << ",\"snapshot_allocations\":" << snapshots.allocations
                      << ",\"snapshot_releases\":" << snapshots.releases
                      << ",\"snapshot_live_images\":" << snapshots.live_images
                      << ",\"snapshot_payload_capacity_bytes\":" << snapshots.payload_capacity_bytes
                      << ",\"snapshot_peak_payload_capacity_bytes\":" << snapshots.peak_payload_capacity_bytes
                      << ",\"host_heap_allocated_bytes\":";
#if defined(__GLIBC__)
#if __GLIBC_PREREQ(2, 33)
            const auto heap = mallinfo2();
            std::cout << heap.uordblks + heap.hblkhd;
#else
            std::cout << "null";
#endif
#else
            std::cout << "null";
#endif
            std::cout << "}" << std::endl;
        } else error_reply("unknown command");
    }
    return 0;
}
} // namespace strata::core
