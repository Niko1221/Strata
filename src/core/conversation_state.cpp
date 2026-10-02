#include "strata/core/conversation_snapshot.hpp"
#include "conversation_checked.hpp"
#include "strata/core/on_device.hpp"

#include <algorithm>
#include <array>
#include <limits>

namespace strata::core {
namespace {
using conversation_detail::add;
using conversation_detail::product;

std::array<int64_t, 18> geometry_key(const ModelGeometry& g) {
    return {g.n_embd, g.n_layers, g.qsa_interval, g.ssm_state_size, g.ssm_k_heads,
            g.ssm_v_heads, g.ssm_d_conv, g.ssm_conv_channels, g.ssm_value_dim,
            g.n_head, g.n_head_kv, g.head_dim, g.idx_q_heads, g.idx_key_dim,
            g.hc, g.hc_lr, g.n_expert, g.n_ff};
}

bool fail(std::string& error, const char* message) {
    error = std::string("conversation snapshot: ") + message;
    return false;
}

bool sync(std::string& error) {
    const auto status = cudaDeviceSynchronize();
    if (status == cudaSuccess) return true;
    error = std::string("conversation snapshot synchronize: ") + cudaGetErrorString(status);
    return false;
}

bool copy(void* dst, const void* src, size_t bytes, std::string& error) {
    if (!bytes) return true;
    if (!dst || !src) return fail(error, "missing running-state buffer");
    const auto status = cudaMemcpy(dst, src, bytes, cudaMemcpyDefault);
    if (status == cudaSuccess) return true;
    error = std::string("conversation snapshot running-state copy: ") + cudaGetErrorString(status);
    return false;
}

// The session's carve (#216): `qsa_states` keeps global ordinals and holds the owned ones
// [qsa_ord0, qsa_ord0 + qsa_alloc); `gdn_state` holds `gdn_alloc` rows. A whole-model session owns every layer.
size_t owned_qsa(const SessionState& ss) { return (size_t) std::max<int64_t>(ss.qsa_alloc, 0); }
const QsaState& owned(const SessionState& ss, size_t j) { return ss.qsa_states[(size_t) ss.qsa_ord0 + j]; }
// The flat K/V order is the stage order, so an entry belongs to the stage whose owned layers it counts.
size_t total_owned(const ConversationStages& stages) {
    size_t n = 0;
    for (const auto& st : stages) n += owned_qsa(*st.ss);
    return n;
}

bool image_keys(const std::vector<ConversationImageKey>& images, size_t tokens) {
    int64_t previous = -1;
    for (const auto& image : images) {
        if (image.start <= previous || image.start < 0 || (uint64_t) image.start >= tokens) return false;
        previous = image.start;
    }
    return true;
}

bool checkpoint_targets(const SessionState& ss, const ModelGeometry& g, size_t tokens,
                        ConversationStateSizes& z, std::string& error) {
    if (!conversation_session_sizes(g, ss, z, error)) return false;
    if (ss.max_cells < 0 || tokens > (uint64_t) ss.max_cells || (z.gdn && !ss.gdn_state) ||
        (owned_qsa(ss) && !ss.qsa_states)) return fail(error, "invalid session running-state targets");
    for (size_t j = 0; j < owned_qsa(ss); ++j) {
        const auto& st = owned(ss, j);
        const auto block = strata::kernels::qsa_real_shapes().idx_block;
        size_t pooled_bytes = 0;
        if (!st.idx_tail || !st.idx_dead || !st.idx_block_pos || !st.idx_pooled ||
            st.max_cells < 0 || tokens > (uint64_t) st.max_cells ||
            (tokens && tokens / (uint64_t) block >= (uint64_t) std::max<int64_t>(0, st.idx_pooled_rows)) ||
            !product(pooled_bytes, {tokens / (uint64_t) block + 1, (uint64_t) g.idx_key_dim, sizeof(float)}))
            return fail(error, "invalid indexer running-state target");
    }
    return true;
}

// A checkpoint is stage 0's part plus one part per later stage. Each part must match the stage that owns
// it: equal payload sizes cannot tell two same-sized carves with different ordinals apart, so the stored
// carve and the mirrored ids decide, and each part validates against its own stage's sizes.
bool parts_validate(const ConversationCheckpoint& c, const ConversationStages& stages,
                    const ModelGeometry& g, std::string& error) {
    if (c.stage_parts.size() + 1 != stages.size()) return fail(error, "checkpoint stage parts do not match the split");
    for (size_t i = 1; i < stages.size(); ++i) {
        const auto& part = c.stage_parts[i - 1];
        if (part.layer_lo != stages[i].ss->layer_lo || part.layer_hi != stages[i].ss->layer_hi || part.ids != c.ids)
            return fail(error, "stage part is not from its stage's carve");
        if (!conversation_checkpoint_validate(part, *stages[i].ss, g, error)) return false;
    }
    return true;
}

bool view_validate(const ConversationView& view, const ConversationStages& stages,
                   const ModelGeometry& g, std::string& error) {
    if (stages.empty()) return fail(error, "no stage to validate against");
    ConversationStateSizes z;
    if (!checkpoint_targets(*stages[0].ss, g, view.ids.size(), z, error)) return false;
    if (view.ids.empty() || !image_keys(view.images, view.ids.size()) ||
        std::any_of(view.ids.begin(), view.ids.end(), [](int32_t id) { return id < 0; }))
        return fail(error, "invalid live token/image metadata");
    for (const auto& c : view.checkpoints) {
        if (c.ids.empty() || c.ids.size() > view.ids.size() ||
            !std::equal(c.ids.begin(), c.ids.end(), view.ids.begin()))
            return fail(error, "checkpoint is not a live token prefix");
        if (!conversation_checkpoint_validate(c, *stages[0].ss, g, error) ||
            !parts_validate(c, stages, g, error)) return false;
        size_t image = 0;
        for (const auto& key : view.images) {
            if ((uint64_t) key.start >= c.ids.size()) break;
            if (image >= c.imgs.size() || !(c.imgs[image++] == key))
                return fail(error, "checkpoint image identity differs");
        }
        if (image != c.imgs.size()) return fail(error, "checkpoint image prefix differs");
    }
    return true;
}

bool metadata_bytes(const ConversationCheckpoint& c, size_t& total) {
    size_t ids = 0, images = 0, directory = 0;
    if (!product(ids, {c.ids.size(), sizeof(int32_t)}) ||
        !product(images, {c.imgs.size(), sizeof(ConversationImageKey)}) ||
        !product(directory, {c.stage_parts.capacity(), sizeof(ConversationCheckpoint)})) return false;
    for (size_t n : {ids, images, directory, c.gdn.size(), c.ple.size(), c.tails.size(), c.dead.size(), c.block_pos.size()})
        if (!add(total, n)) return false;
    for (const auto& part : c.stage_parts)
        if (!metadata_bytes(part, total)) return false;
    return true;
}
} // namespace

bool conversation_state_sizes(const ModelGeometry& g, ConversationStateSizes& z, std::string& error) {
    z = {};
    const auto key = geometry_key(g);
    for (size_t i = 0; i < key.size(); ++i)
        if (key[i] < 0 || (i != 1 && key[i] == 0)) return fail(error, "invalid model geometry");
    size_t recurrence = 0, convolution = 0;
    if (!product(recurrence, {(uint64_t) g.ssm_state_size, (uint64_t) g.ssm_v_heads, (uint64_t) g.ssm_state_size}) ||
        !product(convolution, {(uint64_t) g.ssm_conv_channels, (uint64_t) (g.ssm_d_conv - 1)}) ||
        !add(recurrence, convolution) ||
        !product(z.gdn, {(uint64_t) g.n_gdn_layers(), recurrence, sizeof(float)}) ||
        !product(z.ple, {strata::kernels::NG_HIST, strata::kernels::NG_HC_DIM, sizeof(float)}) ||
        !product(z.tail, {(uint64_t) (strata::kernels::qsa_real_shapes().idx_block - 1),
                          (uint64_t) g.idx_key_dim, sizeof(float)}) ||
        !product(z.dead, {(uint64_t) g.idx_key_dim, sizeof(float)}))
        return fail(error, "running-state byte count overflow");
    z.block_pos = sizeof(int32_t);
    size_t total = 0;
    if (!product(total, {(uint64_t) g.n_qsa_layers(), z.tail}) ||
        !product(total, {(uint64_t) g.n_qsa_layers(), z.dead}) ||
        !product(total, {(uint64_t) g.n_qsa_layers(), z.block_pos}))
        return fail(error, "indexer byte count overflow");
    return true;
}

bool conversation_session_sizes(const ModelGeometry& g, const SessionState& ss, ConversationStateSizes& z,
                                std::string& error) {
    if (!conversation_state_sizes(g, z, error)) return false;
    const int64_t n_gdn = g.n_gdn_layers(), n_qsa = g.n_qsa_layers();
    if (ss.gdn_alloc < 0 || ss.gdn_alloc > n_gdn || ss.qsa_ord0 < 0 || ss.qsa_alloc < 0 ||
        ss.qsa_ord0 > n_qsa || ss.qsa_alloc > n_qsa - ss.qsa_ord0)
        return fail(error, "invalid session layer carve");
    // z.gdn is n_gdn whole rows; the session holds gdn_alloc of them (all of them for the default full range)
    z.gdn = n_gdn > 0 ? z.gdn / (size_t) n_gdn * (size_t) ss.gdn_alloc : 0;
    return true;
}

bool conversation_checkpoint_validate(const ConversationCheckpoint& c, const SessionState& ss,
                                      const ModelGeometry& g, std::string& error) {
    ConversationStateSizes z;
    if (!checkpoint_targets(ss, g, c.ids.size(), z, error)) return false;
    if (c.layer_lo != ss.layer_lo || c.layer_hi != ss.layer_hi)
        return fail(error, "checkpoint from another session layer carve");
    const size_t layers = owned_qsa(ss);
    if (c.gdn.size() != z.gdn || c.ple.size() != (ss.ple_hist ? z.ple : 0) ||
        c.tails.size() != layers * z.tail || c.dead.size() != layers * z.dead ||
        c.block_pos.size() != layers * z.block_pos || !image_keys(c.imgs, c.ids.size()))
        return fail(error, "invalid checkpoint running-state payload");
    return true;
}

bool conversation_checkpoint_save(ConversationCheckpoint& c, const SessionState& ss,
                                  const ModelGeometry& g, std::string& error) {
    ConversationStateSizes z;
    if (!checkpoint_targets(ss, g, c.ids.size(), z, error)) return false;
    c.layer_lo = ss.layer_lo; c.layer_hi = ss.layer_hi;   // the carve this payload can restore into
    const size_t layers = owned_qsa(ss);
    c.gdn.resize(z.gdn); c.ple.resize(ss.ple_hist ? z.ple : 0);
    c.tails.resize(layers * z.tail); c.dead.resize(layers * z.dead); c.block_pos.resize(layers * z.block_pos);
    if (!copy(c.gdn.data(), ss.gdn_state, c.gdn.size(), error) ||
        !copy(c.ple.data(), ss.ple_hist, c.ple.size(), error)) return false;
    for (size_t j = 0; j < layers; ++j) {
        const auto& st = owned(ss, j);
        if (!copy(c.tails.data() + j * z.tail, st.idx_tail, z.tail, error) ||
            !copy(c.dead.data() + j * z.dead, st.idx_dead, z.dead, error) ||
            !copy(c.block_pos.data() + j * z.block_pos, st.idx_block_pos, z.block_pos, error)) return false;
    }
    return true;
}

bool conversation_checkpoint_restore(const ConversationCheckpoint& c, SessionState& ss,
                                     const ModelGeometry& g, std::string& error) {
    if (!conversation_checkpoint_validate(c, ss, g, error)) return false;
    ConversationStateSizes z;
    if (!conversation_session_sizes(g, ss, z, error)) return false;
    if (!copy(ss.gdn_state, c.gdn.data(), c.gdn.size(), error) ||
        !copy(ss.ple_hist, c.ple.data(), c.ple.size(), error)) return false;
    for (size_t j = 0; j < owned_qsa(ss); ++j) {
        const auto& st = owned(ss, j);
        if (!copy(st.idx_tail, c.tails.data() + j * z.tail, z.tail, error) ||
            !copy(st.idx_dead, c.dead.data() + j * z.dead, z.dead, error) ||
            !copy(st.idx_block_pos, c.block_pos.data() + j * z.block_pos, z.block_pos, error)) return false;
        if (!c.ids.empty()) {
            const size_t row = c.ids.size() / strata::kernels::qsa_real_shapes().idx_block;
            if (!copy(st.idx_pooled + row * g.idx_key_dim, c.dead.data() + j * z.dead, z.dead, error)) return false;
        }
    }
    const size_t tokens = c.ids.size();
    ss.ple_prev[0] = tokens >= 2 ? c.ids[tokens - 2] : -1;
    ss.ple_prev[1] = tokens >= 1 ? c.ids[tokens - 1] : -1;
    return sync(error);
}

bool conversation_snapshot_bytes(const ConversationView& view, const ConversationStages& stages,
                                 const ModelGeometry& g, const QsaState& draft, size_t& bytes, std::string& error) {
    bytes = 0;
    if (!view_validate(view, stages, g, error)) return false;
    size_t ids = 0, images = 0, checkpoints = 0, layers = 0, directory = 0;
    const size_t later = stages.size() - 1;   // one live part per later stage
    if (!product(ids, {view.ids.size(), sizeof(int32_t)}) ||
        !product(images, {view.images.size(), sizeof(ConversationImageKey)}) ||
        !product(checkpoints, {view.checkpoints.size(), sizeof(ConversationCheckpoint)}) ||
        !product(layers, {total_owned(stages) + 1, sizeof(ConversationKv)}) ||
        !product(directory, {later, sizeof(ConversationCheckpoint)}))
        return fail(error, "snapshot metadata byte count overflow");
    for (size_t n : {ids, images, checkpoints, layers, directory})
        if (!add(bytes, n)) return fail(error, "snapshot byte count overflow");
    for (size_t i = 0; i < stages.size(); ++i) {   // the live running state, one payload per stage
        ConversationStateSizes z;
        if (!conversation_session_sizes(g, *stages[i].ss, z, error)) return false;
        const size_t owned_layers = owned_qsa(*stages[i].ss);
        size_t tails = 0, dead = 0, positions = 0;
        if (!product(tails, {owned_layers, z.tail}) || !product(dead, {owned_layers, z.dead}) ||
            !product(positions, {owned_layers, z.block_pos})) return fail(error, "indexer byte count overflow");
        for (size_t n : {tails, dead, positions, z.gdn, stages[i].ss->ple_hist ? z.ple : 0})
            if (!add(bytes, n)) return fail(error, "snapshot byte count overflow");
        if (i > 0) for (size_t n : {ids, images})   // every live part mirrors the live ids and images
            if (!add(bytes, n)) return fail(error, "snapshot byte count overflow");
    }
    for (const auto& c : view.checkpoints)
        if (!metadata_bytes(c, bytes)) return fail(error, "checkpoint byte count overflow");
    const int64_t upto = (int64_t) view.ids.size(); // view_validate bounds this by signed max_cells
    for (const auto& st : stages) {
        for (size_t j = 0; j < owned_qsa(*st.ss); ++j) {
            const size_t n = conversation_kv_bytes(owned(*st.ss, j), g, upto, true);
            if (!n || !add(bytes, n)) return fail(error, "invalid or overflowing K/V byte estimate");
        }
    }
    const size_t dn = conversation_kv_bytes(draft, g, upto, false);
    if (!dn || !add(bytes, dn)) return fail(error, "invalid or overflowing K/V byte estimate");
    return true;
}

bool conversation_snapshot_capture_bytes(const ConversationKvReuse& reuse, const ConversationView& view,
                                         const ConversationStages& stages, const ModelGeometry& g,
                                         const QsaState& draft, size_t& bytes, std::string& error) {
    if (!conversation_snapshot_bytes(view, stages, g, draft, bytes, error)) return false;
    if (reuse.kv.empty()) return true;
    const size_t layers = total_owned(stages) + 1;
    if (reuse.kv.size() != layers || reuse.unchanged_tokens < 0 ||
        reuse.unchanged_tokens > reuse.captured_tokens || reuse.unchanged_tokens > int64_t(view.ids.size()))
        return fail(error, "invalid retained K/V prefix");
    auto account = [&](const ConversationKv& image, const QsaState& st, bool index) -> bool {
        if (!conversation_kv_validate(image, st, g, reuse.captured_tokens, index, error)) return false;
        const size_t fresh = conversation_kv_bytes(st, g, int64_t(view.ids.size()), index);
        size_t retained = 0;
        if (!conversation_kv_capture_bytes(image, st, g, int64_t(view.ids.size()), index, retained, error)) return false;
        bytes -= fresh;
        if (!add(bytes, retained)) return fail(error, "retained K/V allocation overflow");
        return true;
    };
    size_t entry = 0;
    for (const auto& st : stages)   // stage order, drafter last
        for (size_t j = 0; j < owned_qsa(*st.ss); ++j)
            if (!account(reuse.kv[entry++], owned(*st.ss, j), true)) return false;
    if (!account(reuse.kv[entry], draft, false)) return false;
    size_t directory = 0;
    if (!product(directory, {reuse.kv.capacity() - layers, sizeof(ConversationKv)}) || !add(bytes, directory))
        return fail(error, "retained K/V directory overflow");
    return true;
}

bool conversation_snapshot_save(SavedConversation& image, const ConversationView& view,
                                const ConversationStages& stages, const ModelGeometry& g,
                                const QsaState& draft, std::string& error,
                                ConversationKvReuse reuse, size_t* reused_bytes) {
    size_t estimate = 0;
    if (!conversation_snapshot_capture_bytes(reuse, view, stages, g, draft, estimate, error)) return false;
    // Build into a new object so a failure cannot publish a partial snapshot.
    SavedConversation captured;
    captured.geometry = geometry_key(g);
    const SessionState& first = *stages[0].ss;
    captured.layer_lo = first.layer_lo; captured.layer_hi = first.layer_hi;
    captured.live.ids = view.ids; captured.live.imgs = view.images;
    captured.cvec = view.cvec; captured.checkpoints = view.checkpoints;
    const int64_t unchanged = reuse.kv.empty() ? 0 : reuse.unchanged_tokens;
    captured.kv = std::move(reuse.kv);
    captured.kv.resize(total_owned(stages) + 1);
    const int64_t upto = (int64_t) view.ids.size();
    // cudaDeviceSynchronize drains only the device the calling thread is bound to, so each stage is read on
    // its own device and drained there first: a copy must not race kernels still writing its buffers.
    size_t entry = 0;
    for (size_t i = 0; i < stages.size(); ++i) {
        const SessionState& ss = *stages[i].ss;
        const OnDevice on(stages[i].dev);
        if (!sync(error)) return false;
        ConversationCheckpoint& part = i == 0 ? captured.live : captured.live.stage_parts.emplace_back();
        if (i != 0) { part.ids = captured.live.ids; part.imgs = captured.live.imgs; }
        if (!conversation_checkpoint_save(part, ss, g, error)) return false;
        for (size_t j = 0; j < owned_qsa(ss); ++j)
            if (!conversation_kv_save(captured.kv[entry++], owned(ss, j), g, upto, true, error,
                                      unchanged, reused_bytes)) return false;
    }
    // The draft's final cell may not have been computed when the output cap was reached. Refresh that page
    // even when the main prefix continued unchanged; the drafter lives on the last stage, already drained.
    {
        const OnDevice on(stages.back().dev);
        if (!conversation_kv_save(captured.kv.back(), draft, g, upto, false, error,
                                  std::max<int64_t>(0, unchanged - 1), reused_bytes)) return false;
    }
    image = std::move(captured);
    return true;
}

bool conversation_snapshot_validate_state(const SavedConversation& image, const ConversationStages& stages,
                                          const ModelGeometry& g, std::string& error) {
    if (image.geometry != geometry_key(g)) return fail(error, "incompatible runtime geometry");
    // an image holds one carve's running state and K/V per stage: same layer ranges or nothing
    if (image.layer_lo != stages[0].ss->layer_lo || image.layer_hi != stages[0].ss->layer_hi)
        return fail(error, "snapshot from another session layer range");
    const ConversationView view{image.live.ids, image.live.imgs, image.checkpoints, image.cvec};
    return view_validate(view, stages, g, error) &&
           conversation_checkpoint_validate(image.live, *stages[0].ss, g, error) &&
           parts_validate(image.live, stages, g, error);
}

bool conversation_snapshot_validate(const SavedConversation& image, const ConversationStages& stages,
                                    const ModelGeometry& g, const QsaState& draft, std::string& error) {
    if (!conversation_snapshot_validate_state(image, stages, g, error)) return false;
    if (image.kv.size() != total_owned(stages) + 1) return fail(error, "invalid K/V layer count");
    const int64_t upto = (int64_t) image.live.ids.size();
    size_t entry = 0;
    for (size_t i = 0; i < stages.size(); ++i)
        for (size_t j = 0; j < owned_qsa(*stages[i].ss); ++j)
            if (!conversation_kv_validate(image.kv[entry++], owned(*stages[i].ss, j), g, upto, true, error)) return false;
    return conversation_kv_validate(image.kv.back(), draft, g, upto, false, error);
}

std::vector<ConversationKvEntry> conversation_kv_entries(const ConversationStages& stages, const QsaState& draft) {
    std::vector<ConversationKvEntry> entries;
    for (const auto& st : stages)
        for (size_t j = 0; j < owned_qsa(*st.ss); ++j) entries.push_back({&owned(*st.ss, j), st.dev, true});
    entries.push_back({&draft, stages.back().dev, false});   // the drafter lives on the last stage
    return entries;
}

bool conversation_live_save(SavedConversation& image, const ConversationView& view, const ConversationStages& stages,
                            const ModelGeometry& g, std::string& error) {
    if (!view_validate(view, stages, g, error)) return false;
    SavedConversation captured;
    captured.geometry = geometry_key(g);
    captured.layer_lo = stages[0].ss->layer_lo; captured.layer_hi = stages[0].ss->layer_hi;
    captured.live.ids = view.ids; captured.live.imgs = view.images; captured.cvec = view.cvec;
    for (size_t i = 0; i < stages.size(); ++i) {   // each stage drained on its own device first, as in the save
        const OnDevice on(stages[i].dev);
        if (!sync(error)) return false;
        ConversationCheckpoint& part = i == 0 ? captured.live : captured.live.stage_parts.emplace_back();
        if (i != 0) { part.ids = captured.live.ids; part.imgs = captured.live.imgs; }
        if (!conversation_checkpoint_save(part, *stages[i].ss, g, error)) return false;
    }
    image = std::move(captured);
    return true;
}

bool conversation_live_restore(const ConversationCheckpoint& live, const ConversationStages& stages,
                               const ModelGeometry& g, std::string& error) {
    for (size_t i = 0; i < stages.size(); ++i) {
        const OnDevice on(stages[i].dev);
        if (!sync(error)) return false;
        if (!conversation_checkpoint_restore(i == 0 ? live : live.stage_parts[i - 1], *stages[i].ss, g, error))
            return false;
    }
    return true;
}

ConversationRestore conversation_snapshot_restore(const SavedConversation& image, const ConversationStages& stages,
                                                   const ModelGeometry& g, const QsaState& draft, std::string& error) {
    if (!conversation_snapshot_validate(image, stages, g, draft, error)) return ConversationRestore::invalid;
    const int64_t upto = (int64_t) image.live.ids.size();
    // Each stage is drained on its own device before anything there is written: cudaDeviceSynchronize covers
    // only the device the calling thread is bound to, so kernels still running on another stage could land
    // after these copies. After a drain nothing is enqueued - these copies are synchronous.
    size_t entry = 0;
    for (size_t i = 0; i < stages.size(); ++i) {
        SessionState& ss = *stages[i].ss;
        const OnDevice on(stages[i].dev);
        if (!sync(error)) return ConversationRestore::transfer_failed;
        for (size_t j = 0; j < owned_qsa(ss); ++j)
            if (!conversation_kv_restore(image.kv[entry++], owned(ss, j), g, upto, true, error))
                return ConversationRestore::transfer_failed;
        const ConversationCheckpoint& part = i == 0 ? image.live : image.live.stage_parts[i - 1];
        if (!conversation_checkpoint_restore(part, ss, g, error)) return ConversationRestore::transfer_failed;
    }
    {
        const OnDevice on(stages.back().dev);   // the drafter lives on the last stage, already drained
        if (!conversation_kv_restore(image.kv.back(), draft, g, upto, false, error))
            return ConversationRestore::transfer_failed;
    }
    return ConversationRestore::restored;
}
} // namespace strata::core
