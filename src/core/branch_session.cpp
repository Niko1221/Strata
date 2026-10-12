#include "strata/core/branch_session.hpp"
#include "strata/core/conversation_snapshot.hpp"
#include <algorithm>
#include <cmath>
#include <cuda_runtime.h>
#include <exception>
#include <fstream>
#include <mutex>

namespace strata::core {
namespace {
std::mutex snapshot_mutex;
BranchSnapshotCounters snapshot_work;
struct SnapshotLifetime {
    uint64_t payload_bytes = 0;
    SnapshotLifetime() {
        std::lock_guard<std::mutex> lock(snapshot_mutex);
        ++snapshot_work.allocations;
        ++snapshot_work.live_images;
    }
    SnapshotLifetime(const SnapshotLifetime&) = delete;
    SnapshotLifetime& operator=(const SnapshotLifetime&) = delete;
    void publish(uint64_t bytes) {
        std::lock_guard<std::mutex> lock(snapshot_mutex);
        payload_bytes = bytes;
        snapshot_work.payload_capacity_bytes += bytes;
        snapshot_work.peak_payload_capacity_bytes = std::max(snapshot_work.peak_payload_capacity_bytes,
                                                            snapshot_work.payload_capacity_bytes);
    }
    ~SnapshotLifetime() {
        // This member is declared first and destroyed last, after the image's
        // vectors and page owners have actually released their resources.
        std::lock_guard<std::mutex> lock(snapshot_mutex);
        ++snapshot_work.releases;
        --snapshot_work.live_images;
        snapshot_work.payload_capacity_bytes -= payload_bytes;
    }
};
const char* pool_kinds[] = {"k", "v", "k_scale", "v_scale", "indexer_pool"};
bool fail(std::string& error, const char* message) {
    error = std::string("branch session: ") + message;
    return false;
}
bool sync(std::string& error) {
    const auto result = cudaDeviceSynchronize();
    return result == cudaSuccess || fail(error, cudaGetErrorString(result));
}
void* pool_pointer(const QsaState& q, size_t part) {
    if (part == 4) return q.idx_pooled;
    if (part == 2) return q.k_scale;
    if (part == 3) return q.v_scale;
    if (q.kv_hybrid) return part == 0 ? (void*)q.k_q : q.v_q4;
    if (q.kv_q4) return part == 0 ? q.k_q4 : q.v_q4;
    if (q.kv_int8) return part == 0 ? q.k_q : q.v_q;
    return part == 0 ? q.k_pool : q.v_pool;
}
void set_pointer(QsaState& q, size_t part, void* p) {
    if (part == 4) q.idx_pooled = (float*)p;
    else if (part == 2) q.k_scale = (uint16_t*)p;
    else if (part == 3) q.v_scale = (uint16_t*)p;
    else if (q.kv_hybrid) { if (part == 0) q.k_q = (int8_t*)p; else q.v_q4 = (uint8_t*)p; }
    else if (q.kv_q4) { if (part == 0) q.k_q4 = (uint8_t*)p; else q.v_q4 = (uint8_t*)p; }
    else if (q.kv_int8) { if (part == 0) q.k_q = (int8_t*)p; else q.v_q = (int8_t*)p; }
    else { if (part == 0) q.k_pool = (uint16_t*)p; else q.v_pool = (uint16_t*)p; }
}
uint64_t running_bytes(const ConversationCheckpoint& c) {
    return c.gdn.size() + c.ple.size() + c.tails.size() + c.dead.size() + c.block_pos.size();
}
SessionFileIdentity file_identity(const std::string& contract, const std::vector<int64_t>& layout) {
    // Native format detects accidental corruption. Cryptographic file, model
    // and execution fingerprints are checked by the outer API manifest.
    return {session_hash64(contract.data(), contract.size(), 0),
            session_hash64(layout.data(), layout.size()*sizeof(int64_t), 0)};
}
}

struct BranchSessionSnapshot {
    SnapshotLifetime lifetime;
    std::string contract;
    std::vector<int64_t> layout;
    std::vector<BranchPageSnapshot> pages;
    ConversationCheckpoint running;
    std::vector<float> logits;
};
BranchSnapshotCounters BranchSession::snapshot_counters() {
    std::lock_guard<std::mutex> lock(snapshot_mutex);
    return snapshot_work;
}
struct BranchSession::Binding {
    QsaState* state = nullptr;
    size_t part = 0;
    void* original = nullptr;
    uint64_t bytes = 0, per_cell = 0;
    BranchPages memory;
};

BranchSession::BranchSession(SessionState& session, const ModelGeometry& geometry, std::string contract)
    : session_(session), geometry_(geometry), contract_(std::move(contract)) {}
BranchSession::~BranchSession() {
    if (bindings_.empty()) return;
    if (cudaDeviceSynchronize() != cudaSuccess) std::terminate();
    for (const auto& b : bindings_) set_pointer(*b->state, b->part, b->original);
}
bool BranchSession::bind(std::string& error) {
    if (state_ != State::unbound || contract_.empty()) return fail(error, "invalid bind state or empty contract");
    const auto& g = geometry_;
    ConversationStateSizes sizes;
    if (!conversation_session_sizes(g, session_, sizes, error) || session_.max_cells <= 0 ||
        !session_.qsa_states || session_.qsa_alloc <= 0 || !sync(error)) return false;
    layout_ = {g.n_embd, g.n_layers, g.qsa_interval, g.ssm_state_size, g.ssm_k_heads, g.ssm_v_heads,
        g.ssm_d_conv, g.ssm_conv_channels, g.ssm_value_dim, g.n_head, g.n_head_kv, g.head_dim,
        g.idx_q_heads, g.idx_key_dim, g.hc, g.hc_lr, g.n_expert, g.n_ff,
        session_.max_cells, session_.layer_lo, session_.layer_hi, session_.qsa_ord0, session_.qsa_alloc};
    std::vector<std::unique_ptr<Binding>> staged;
    try {
        for (int64_t i = 0; i < session_.qsa_alloc; ++i) {
            auto& q = session_.qsa_states[session_.qsa_ord0 + i];
            if (q.kv_mode != 0 || q.kv_elastic >= 0 || q.n_pages != q.n_slots)
                return fail(error, "hot pages require fully resident, identity-mapped QSA pools");
            std::vector<int32_t> table((size_t)q.n_pages);
            if (cudaMemcpy(table.data(), q.page_table, table.size()*4, cudaMemcpyDeviceToHost) != cudaSuccess)
                return fail(error, "cannot validate attention page table");
            for (size_t p = 0; p < table.size(); ++p)
                if (table[p] != (int32_t)p) return fail(error, "attention page table is not the identity");
            std::array<uint64_t, 5> parts{};
            if (!conversation_kv_part_sizes(q, g, q.max_cells, true, parts, error)) return false;
            parts[4] = (uint64_t)q.idx_pooled_rows * g.idx_key_dim * sizeof(float);
            layout_.insert(layout_.end(), {q.kv_int8, q.kv_q4, q.kv_hybrid, q.kv_rot,
                q.max_cells, q.n_pages, q.n_slots, q.idx_pooled_rows});
            const uint64_t cells = (uint64_t)q.n_pages * strata::kernels::qsa_real_shapes().page_size;
            for (size_t p = 0; p < parts.size(); ++p) {
                if (!parts[p]) continue;
                auto b = std::make_unique<Binding>();
                b->state = &q; b->part = p; b->original = pool_pointer(q, p); b->bytes = parts[p];
                b->per_cell = p == 4 ? (uint64_t)g.idx_key_dim * sizeof(float) : parts[p] / cells;
                if (!b->original || !b->memory.reserve(parts[p]) || !b->memory.grow(parts[p]))
                    return fail(error, "cannot allocate private session mapping");
                if (cudaMemcpy(b->memory.data(), b->original, parts[p], cudaMemcpyDeviceToDevice) != cudaSuccess ||
                    !sync(error)) return fail(error, "cannot copy initial session pools");
                counters_.binding_copy_bytes += parts[p];
                staged.push_back(std::move(b));
            }
        }
    } catch (const std::exception& e) { return fail(error, e.what()); }
    bindings_ = std::move(staged);
    for (const auto& b : bindings_) set_pointer(*b->state, b->part, b->memory.data());
    state_ = State::ready;
    return true;
}
bool BranchSession::begin(int64_t position, int64_t count, std::string& error) {
    if (state_ != State::ready || position != (int64_t)ids_.size() || count <= 0 ||
        position < 0 || position > session_.max_cells || count > session_.max_cells - position)
        return fail(error, "invalid consumed position, capacity, or transition");
    if (!sync(error)) { state_ = State::failed; return false; }
    const auto shape = strata::kernels::qsa_real_shapes();
    for (const auto& b : bindings_) {
        uint64_t first, end;
        if (b->part == 4) {
            first = position / shape.idx_block;
            end = (position + count) / shape.idx_block + 1; // includes the moving spare row
        } else {
            first = position / shape.page_size * shape.page_size;
            end = (position + count + shape.page_size - 1) / shape.page_size * shape.page_size;
        }
        if (!b->memory.prepare_write(first*b->per_cell, (end-first)*b->per_cell)) {
            state_ = State::failed;
            return fail(error, "cannot prepare private tail; restore a committed image before reuse");
        }
    }
    pending_ = count;
    state_ = State::writing;
    return true;
}
bool BranchSession::reset(std::string& error) {
    if (state_ == State::unbound) return fail(error, "cannot reset an unbound session");
    if (!sync(error)) { state_ = State::failed; return false; }
    state_ = State::failed;
    for (const auto& b : bindings_)
        if (!b->memory.clear() || !b->memory.grow(b->bytes))
            return fail(error, "cannot allocate fresh private pages");
    ConversationStateSizes sizes;
    if (!conversation_session_sizes(geometry_, session_, sizes, error)) return false;
    auto zero = [&](void* p, size_t bytes) {
        return !bytes || (p && cudaMemset(p, 0, bytes) == cudaSuccess);
    };
    if (!zero(session_.gdn_state, sizes.gdn) || !zero(session_.ple_hist, session_.ple_hist ? sizes.ple : 0))
        return fail(error, "cannot clear recurrent state");
    for (int64_t i = 0; i < session_.qsa_alloc; ++i) {
        const auto& q = session_.qsa_states[session_.qsa_ord0 + i];
        if (!zero(q.idx_tail, sizes.tail) || !zero(q.idx_dead, sizes.dead) || !zero(q.idx_block_pos, sizes.block_pos))
            return fail(error, "cannot clear indexer state");
    }
    if (!sync(error)) return false;
    session_.ple_prev[0] = session_.ple_prev[1] = session_.ple_token = -1;
    ids_.clear(); logits_.clear(); pending_ = 0; state_ = State::ready;
    return true;
}
bool BranchSession::finish(const std::vector<int32_t>& tokens, const std::vector<float>& logits, std::string& error) {
    if (state_ != State::writing || (int64_t)tokens.size() != pending_ || logits.empty() ||
        std::any_of(tokens.begin(), tokens.end(), [](int32_t t) { return t < 0; }) ||
        std::any_of(logits.begin(), logits.end(), [](float x) { return !std::isfinite(x); }))
        return fail(error, "invalid consumed token or frontier result");
    if (!sync(error)) { state_ = State::failed; return false; }
    try {
        auto ids = ids_;
        ids.insert(ids.end(), tokens.begin(), tokens.end());
        auto next = logits;
        ids_.swap(ids); logits_.swap(next);
    } catch (const std::exception& e) { state_ = State::failed; return fail(error, e.what()); }
    state_ = State::ready;
    pending_ = 0;
    return true;
}
BranchSessionImage BranchSession::freeze(std::string& error) {
    if (state_ != State::ready || ids_.empty() || !sync(error)) {
        fail(error, "only a consumed, synchronized frontier can be captured");
        return {};
    }
    try {
        auto image = std::make_shared<BranchSessionSnapshot>();
        image->contract = contract_; image->layout = layout_;
        image->running.ids = ids_; image->logits = logits_;
        if (!conversation_checkpoint_save(image->running, session_, geometry_, error)) return {};
        counters_.saved_running_bytes += running_bytes(image->running);
        for (const auto& b : bindings_) {
            auto pages = b->memory.freeze();
            if (!pages) { fail(error, "cannot protect captured pages"); return {}; }
            image->pages.push_back(std::move(pages));
        }
        image->lifetime.publish(image->running.bytes() + image->logits.capacity() * sizeof(float));
        ++counters_.captures;
        return image;
    } catch (const std::exception& e) { fail(error, e.what()); return {}; }
}
bool BranchSession::restore(const BranchSessionImage& image, std::string& error) {
    if (state_ == State::unbound || !image || image->contract != contract_ || image->layout != layout_ ||
        image->pages.size() != bindings_.size()) return fail(error, "checkpoint contract or layout differs");
    if (!conversation_checkpoint_validate(image->running, session_, geometry_, error) || !sync(error)) return false;
    state_ = State::failed; // Any partial restore is private and cannot be frozen.
    for (size_t i = 0; i < bindings_.size(); ++i)
        if (!bindings_[i]->memory.restore(image->pages[i])) return fail(error, "cannot activate checkpoint pages");
    // The pooled spare row is already in the frozen page image. Restoring it a
    // second time would write a shared, read-only page for no semantic change.
    if (!conversation_checkpoint_restore_running(image->running, session_, geometry_, error)) return false;
    counters_.restored_running_bytes += running_bytes(image->running);
    try { ids_ = image->running.ids; logits_ = image->logits; }
    catch (const std::exception& e) { return fail(error, e.what()); }
    pending_ = 0; state_ = State::ready; ++counters_.restores;
    return true;
}
bool BranchSession::write(const std::string& path, size_t& bytes, std::string& error) {
    bytes = 0;
    if (state_ != State::ready || ids_.empty() || !sync(error))
        return fail(error, "cold write requires a consumed, synchronized frontier");
    try {
        const std::vector<ConversationImageKey> images;
        const std::vector<ConversationCheckpoint> checkpoints;
        SavedConversation meta;
        std::vector<SessionKvSource> sources;
        const ConversationView view{ids_, images, checkpoints, false};
        if (!conversation_snapshot_sources(meta, sources, view, session_, geometry_, nullptr, error) ||
            !session_file_write(path, meta, sources, file_identity(contract_, layout_), bytes, error)) return false;
        counters_.written_file_bytes += bytes;
        return true;
    } catch (const std::exception& e) { return fail(error, e.what()); }
}
bool BranchSession::host_capture_bytes(size_t& bytes, std::string& error) const {
    if (state_ != State::ready || ids_.empty() || logits_.empty())
        return fail(error, "host capture requires a consumed frontier");
    const std::vector<ConversationImageKey> images;
    const std::vector<ConversationCheckpoint> checkpoints;
    if (!conversation_snapshot_bytes({ids_, images, checkpoints, false}, session_, geometry_, nullptr, bytes, error))
        return false;
    if (logits_.size() > (SIZE_MAX - bytes) / sizeof(float)) return fail(error, "host frontier size overflow");
    bytes += logits_.size() * sizeof(float);
    return true;
}
bool BranchSession::capture_host(SavedConversation& image, std::string& error) {
    size_t bytes = 0;
    if (!host_capture_bytes(bytes, error)) return false;
    try {
        const std::vector<ConversationImageKey> images;
        const std::vector<ConversationCheckpoint> checkpoints;
        SavedConversation captured;
        if (!conversation_snapshot_save(captured, {ids_, images, checkpoints, false},
                                        session_, geometry_, nullptr, error)) return false;
        captured.next_logits = logits_;
        image = std::move(captured);
        return true;
    } catch (const std::exception& e) { return fail(error, e.what()); }
}
bool BranchSession::restore_host(const SavedConversation& image, const std::vector<int32_t>& expected_ids,
                                const std::vector<float>& frontier, std::string& error) {
    if (state_ == State::unbound || expected_ids.empty() || expected_ids.size() > (size_t)session_.max_cells ||
        image.live.ids != expected_ids || !image.live.imgs.empty() || !image.checkpoints.empty() || image.cvec ||
        frontier.empty() || std::any_of(frontier.begin(), frontier.end(), [](float x) { return !std::isfinite(x); }))
        return fail(error, "invalid host checkpoint history or frontier");
    if (!conversation_snapshot_validate(image, session_, geometry_, nullptr, error)) return false;
    try {
        auto next_ids = expected_ids;
        auto next_logits = frontier;
        if (!reset(error)) return false;
        state_ = State::failed;
        if (conversation_snapshot_restore(image, session_, geometry_, nullptr, error) != ConversationRestore::restored ||
            !sync(error)) return false;
        ids_.swap(next_ids); logits_.swap(next_logits);
        pending_ = 0; state_ = State::ready;
        return true;
    } catch (const std::exception& e) { return fail(error, e.what()); }
}
bool BranchSession::load(const std::string& path, const std::vector<int32_t>& expected_ids,
                         const std::vector<float>& frontier, std::string& error) {
    if (state_ == State::unbound || expected_ids.empty() || expected_ids.size() > (size_t)session_.max_cells ||
        std::any_of(expected_ids.begin(), expected_ids.end(), [](int32_t t) { return t < 0; }) ||
        frontier.empty() || std::any_of(frontier.begin(), frontier.end(), [](float x) { return !std::isfinite(x); }))
        return fail(error, "invalid cold history or frontier");
    try {
        SessionReadLimits limits;
        SavedConversation image;
        size_t bytes = 0;
        if (!conversation_session_read_limits(limits, session_, geometry_, nullptr, session_.max_cells, 0, error) ||
            !session_file_read(path, file_identity(contract_, layout_), image, bytes, error, limits)) return false;
        if (image.live.ids != expected_ids || !image.live.imgs.empty() || !image.checkpoints.empty() || image.cvec)
            return fail(error, "cold checkpoint history differs");
        if (!conversation_snapshot_validate(image, session_, geometry_, nullptr, error)) return false;
        auto next_logits = frontier; // Allocate before touching private device state.
        if (!reset(error)) return false;
        state_ = State::failed;
        if (conversation_snapshot_restore(image, session_, geometry_, nullptr, error) != ConversationRestore::restored ||
            !sync(error)) return false;
        ids_.swap(image.live.ids); logits_.swap(next_logits);
        pending_ = 0; state_ = State::ready;
        counters_.loaded_file_bytes += bytes;
        return true;
    } catch (const std::exception& e) { return fail(error, e.what()); }
}
std::vector<BranchSessionPool> BranchSession::pools() const {
    std::vector<BranchSessionPool> out;
    for (const auto& b : bindings_) out.push_back({b->memory.data(), b->bytes, b->per_cell, b->part == 4});
    return out;
}
std::vector<BranchPageInfo> BranchSession::ownership() const {
    std::vector<BranchPageInfo> out;
    for (size_t i = 0; i < bindings_.size(); ++i) {
        const auto& b = bindings_[i];
        for (auto page : b->memory.ownership()) {
            page.kind = pool_kinds[b->part];
            page.pool = (int64_t)i;
            out.push_back(page);
        }
    }
    return out;
}
bool BranchSession::audit(const BranchSessionImage& image, const std::string& path,
                          BranchSessionAudit& report, std::string& error) const {
    if (!image || image->contract != contract_ || image->layout != layout_ ||
        image->pages.size() != bindings_.size()) return fail(error, "invalid audit image");
    if (!sync(error)) return false;
    try {
        std::ofstream output(path, std::ios::binary | std::ios::trunc);
        if (!output) return fail(error, "cannot open audit output");
        BranchSessionAudit result;
        result.consumed = image->running.ids.size();
        const auto shape = strata::kernels::qsa_real_shapes();
        result.qsa_page_cells = shape.page_size; result.index_block_cells = shape.idx_block;
        auto append = [&](BranchAuditRegion region, const void* data, size_t bytes) {
            if (!bytes) return true;
            region.offset = result.bytes; region.bytes = bytes;
            output.write((const char*)data, (std::streamsize)bytes);
            if (!output) return false;
            result.bytes += bytes;
            result.regions.push_back(std::move(region));
            return true;
        };
        for (size_t i = 0; i < bindings_.size(); ++i) {
            const auto& binding = bindings_[i];
            const auto pages = BranchPages::ownership(image->pages[i]);
            uint64_t pool_offset = 0;
            const auto complete = binding->part == 4 ? result.consumed / shape.idx_block :
                result.consumed / shape.page_size * shape.page_size;
            const auto filled = complete * binding->per_cell;
            for (size_t p = 0; p < pages.size(); ++p) {
                std::vector<uint8_t> bytes;
                if (!BranchPages::read_page(image->pages[i], p, bytes)) return fail(error, "cannot read saved physical page");
                BranchAuditRegion region;
                region.kind = pool_kinds[binding->part]; region.pool = (int64_t)i;
                region.qsa_ordinal = binding->state - session_.qsa_states;
                region.pool_offset = pool_offset; region.bytes_per_cell = binding->per_cell;
                region.filled_prefix_bytes = pool_offset < filled ? std::min(pages[p].bytes, filled - pool_offset) : 0;
                region.page = pages[p];
                if (!append(region, bytes.data(), bytes.size())) return fail(error, "cannot write page audit bytes");
                pool_offset += pages[p].bytes;
            }
        }
        auto field = [&](const char* kind, const auto& values) {
            BranchAuditRegion region; region.kind = kind;
            return append(region, values.data(), values.size() * sizeof(values[0]));
        };
        const auto& running = image->running;
        if (!field("gdn", running.gdn) || !field("ple", running.ple) || !field("tails", running.tails) ||
            !field("dead", running.dead) || !field("block_pos", running.block_pos) ||
            !field("ids", running.ids) || !field("logits", image->logits)) return fail(error, "cannot write running audit bytes");
        output.close();
        if (!output) return fail(error, "incomplete audit output");
        report = std::move(result);
        return true;
    } catch (const std::exception& e) { return fail(error, e.what()); }
}
} // namespace strata::core
