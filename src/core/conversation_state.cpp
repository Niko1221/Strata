#include "strata/core/conversation_snapshot.hpp"
#include "conversation_checked.hpp"
#include "conversation_copy.hpp"

#include <algorithm>
#include <array>
#include <exception>
#include <limits>
#include <new>

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
    const auto status = conversation_detail::copy_nonblocking(dst, src, bytes);
    if (status == cudaSuccess) return true;
    error = std::string("conversation snapshot running-state copy: ") + cudaGetErrorString(status);
    return false;
}

// The codec a checkpoint says it holds. An unknown code is a refusal, not fp32 by default: reading coded bytes as
// fp32 would double every value count and put nonsense in the running state.
bool codec_of(const ConversationCheckpoint& c, StateCodec& out, std::string& error) {
    if (state_codec_by_code(c.state_codec, out)) return true;
    return fail(error, "checkpoint names a running-state codec this build does not know");
}

// A stored copy of the running state may be coded, so it does not cross the device boundary as one fp32 block. It
// crosses in slices: one slice of fp32 values in a buffer of kStateCodecChunkFloats floats, coded or widened there,
// so a coded capture or restore costs that much host memory and no more. Fp32 is the single copy it has always
// been - no scratch, no second pass, the same bytes in the same order.
bool copy_state_out(void* dst, const void* src, size_t fp32_bytes, StateCodec codec, std::string& error) {
    if (codec == StateCodec::Fp32) return copy(dst, src, fp32_bytes, error);
    const size_t floats = fp32_bytes / 4;
    std::vector<float> scratch;
    for (size_t at = 0; at < floats; ) {
        const size_t n = std::min(kStateCodecChunkFloats, floats - at);
        scratch.resize(n);
        if (!copy(scratch.data(), (const float*) src + at, n * 4, error)) return false;
        if (!state_encode(scratch.data(), n, codec, (uint8_t*) dst + state_codec_size(at * 4, codec), error)) return false;
        at += n;
    }
    return true;
}

bool copy_state_in(void* dst, const void* src, size_t fp32_bytes, StateCodec codec, std::string& error) {
    if (codec == StateCodec::Fp32) return copy(dst, src, fp32_bytes, error);
    const size_t floats = fp32_bytes / 4;
    std::vector<float> scratch;
    for (size_t at = 0; at < floats; ) {
        const size_t n = std::min(kStateCodecChunkFloats, floats - at);
        scratch.resize(n);
        if (!state_decode((const uint8_t*) src + state_codec_size(at * 4, codec), n, codec, scratch.data(), error)) return false;
        if (!copy((float*) dst + at, scratch.data(), n * 4, error)) return false;
        at += n;
    }
    return true;
}

// The session's carve (#216): `qsa_states` keeps global ordinals and holds the owned ones
// [qsa_ord0, qsa_ord0 + qsa_alloc); `gdn_state` holds `gdn_alloc` rows. A whole-model session owns every layer.
size_t owned_qsa(const SessionState& ss) { return (size_t) std::max<int64_t>(ss.qsa_alloc, 0); }
const QsaState& owned(const SessionState& ss, size_t j) { return ss.qsa_states[(size_t) ss.qsa_ord0 + j]; }

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

bool view_validate(const ConversationView& view, const SessionState& ss,
                   const ModelGeometry& g, std::string& error) {
    ConversationStateSizes z;
    if (!checkpoint_targets(ss, g, view.ids.size(), z, error)) return false;
    if (view.ids.empty() || !image_keys(view.images, view.ids.size()) ||
        std::any_of(view.ids.begin(), view.ids.end(), [](int32_t id) { return id < 0; }))
        return fail(error, "invalid live token/image metadata");
    for (const auto& c : view.checkpoints) {
        if (!c.stage_parts.empty() || c.ids.empty() || c.ids.size() > view.ids.size() ||
            !std::equal(c.ids.begin(), c.ids.end(), view.ids.begin()))
            return fail(error, "checkpoint is not a live token prefix");
        if (!conversation_checkpoint_validate(c, ss, g, error)) return false;
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

bool metadata_bytes(const ConversationCheckpoint& c, size_t& total, StateCodec codec) {
    size_t ids = 0, images = 0;
    if (!product(ids, {c.ids.size(), sizeof(int32_t)}) ||
        !product(images, {c.imgs.size(), sizeof(ConversationImageKey)})) return false;
    // the bytes a stored copy of this checkpoint would hold: its values, held the way `codec` holds them
    StateCodec from = StateCodec::Fp32;
    if (!state_codec_by_code(c.state_codec, from)) return false;
    const size_t gdn = state_codec_size(state_codec_fp32_bytes(c.gdn.size(), from), codec);
    const size_t ple = state_codec_size(state_codec_fp32_bytes(c.ple.size(), from), codec);
    const size_t tails = state_codec_size(state_codec_fp32_bytes(c.tails.size(), from), codec);
    const size_t dead = state_codec_size(state_codec_fp32_bytes(c.dead.size(), from), codec);
    for (size_t n : {ids, images, gdn, ple, tails, dead, c.block_pos.size()})
        if (!add(total, n)) return false;
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
    StateCodec codec = StateCodec::Fp32;
    if (!codec_of(c, codec, error)) return false;
    const size_t layers = owned_qsa(ss);
    // every array is checked at the size this checkpoint says it holds: a coded image is the size its codec makes,
    // and a file that lies about either fails here rather than restoring as nonsense
    if (c.gdn.size() != state_codec_size(z.gdn, codec) ||
        c.ple.size() != state_codec_size(ss.ple_hist ? z.ple : 0, codec) ||
        c.tails.size() != state_codec_size(layers * z.tail, codec) ||
        c.dead.size() != state_codec_size(layers * z.dead, codec) ||
        c.block_pos.size() != layers * z.block_pos || !image_keys(c.imgs, c.ids.size()))
        return fail(error, "invalid checkpoint running-state payload");
    return true;
}

#if defined(STRATA_GPU_METAL)
}  // namespace strata::core
namespace strata::metal_engine {
// the running state lives in the Metal decode window (metal/src/engine/window.cpp), not in the session's buffers
bool window_checkpoint_save(strata::core::ConversationCheckpoint& c, std::string& error);
bool window_checkpoint_restore(const strata::core::ConversationCheckpoint& c, std::string& error);
}  // namespace strata::metal_engine
namespace strata::core {
#endif

bool conversation_checkpoint_save(ConversationCheckpoint& c, const SessionState& ss,
                                  const ModelGeometry& g, std::string& error) {
#if defined(STRATA_GPU_METAL)
    (void) ss; (void) g;
    return strata::metal_engine::window_checkpoint_save(c, error);
#endif
    ConversationStateSizes z;
    if (!checkpoint_targets(ss, g, c.ids.size(), z, error)) return false;
    // the caller asks for a coding by setting c.state_codec; a checkpoint left at 0 gets the bytes as computed
    StateCodec codec = StateCodec::Fp32;
    if (!codec_of(c, codec, error)) return false;
    const size_t layers = owned_qsa(ss);
    // Built beside the checkpoint and moved in only once every array has been copied. A coded capture allocates the
    // stored arrays plus one bounded staging slice (kStateCodecChunkFloats floats, 4 MiB) per array, so the capture
    // that cannot allocate is a refusal with `c` exactly as it was: no half-written snapshot reaches the caller, and
    // nothing here writes the session - the running state is only ever read. Fp32 resizes to the bytes it always
    // sized and writes the same bytes in the same order.
    std::vector<uint8_t> gdn, ple, tails, dead, block_pos;
    try {
        gdn.resize(state_codec_size(z.gdn, codec));
        ple.resize(state_codec_size(ss.ple_hist ? z.ple : 0, codec));
        tails.resize(state_codec_size(layers * z.tail, codec));
        dead.resize(state_codec_size(layers * z.dead, codec));
        block_pos.resize(layers * z.block_pos);
        if (!copy_state_out(gdn.data(), ss.gdn_state, z.gdn, codec, error) ||
            !copy_state_out(ple.data(), ss.ple_hist, ss.ple_hist ? z.ple : 0, codec, error)) return false;
        for (size_t j = 0; j < layers; ++j) {
            const auto& st = owned(ss, j);
            if (!copy_state_out(tails.data() + state_codec_size(j * z.tail, codec), st.idx_tail, z.tail, codec,
                                error) ||
                !copy_state_out(dead.data() + state_codec_size(j * z.dead, codec), st.idx_dead, z.dead, codec,
                                error) ||
                !copy(block_pos.data() + j * z.block_pos, st.idx_block_pos, z.block_pos, error)) return false;
        }
    } catch (const std::exception&) {
        // the refusal is what a memory-tight capture can still afford, and this is the path where it cannot: a
        // throw past here ends the engine instead of refusing one checkpoint (strata/core/conversation_report.hpp)
        try { error = "not enough memory to capture the running state"; } catch (...) {}
        return false;
    }
    c.gdn = std::move(gdn); c.ple = std::move(ple);
    c.tails = std::move(tails); c.dead = std::move(dead); c.block_pos = std::move(block_pos);
    return true;
}

bool conversation_checkpoint_restore(const ConversationCheckpoint& c, SessionState& ss,
                                     const ModelGeometry& g, std::string& error) {
#if defined(STRATA_GPU_METAL)
    (void) g;
    if (!strata::metal_engine::window_checkpoint_restore(c, error)) return false;
    ss.ple_prev[0] = c.ids.size() >= 2 ? c.ids[c.ids.size() - 2] : -1;
    ss.ple_prev[1] = c.ids.size() >= 1 ? c.ids[c.ids.size() - 1] : -1;
    return true;
#endif
    if (!conversation_checkpoint_validate(c, ss, g, error)) return false;
    ConversationStateSizes z;
    if (!conversation_session_sizes(g, ss, z, error)) return false;
    StateCodec codec = StateCodec::Fp32;
    if (!codec_of(c, codec, error)) return false;
    if (!sync(error)) return false;   // the copies below no longer ride the legacy stream's implicit ordering
    if (!copy_state_in(ss.gdn_state, c.gdn.data(), z.gdn, codec, error) ||
        !copy_state_in(ss.ple_hist, c.ple.data(), ss.ple_hist ? z.ple : 0, codec, error)) return false;
    // The indexer's spare row is put back from the same values as its dead key, and both then live in the session's
    // fp32 buffers: under a codec the two arrays are widened once here and the loop below reads fp32 either way.
    std::vector<uint8_t> tails, dead;
    const uint8_t* tail_bytes = c.tails.data();
    const uint8_t* dead_bytes = c.dead.data();
    if (codec != StateCodec::Fp32 &&
        (!conversation_detail::convert_state(c.tails, codec, tails, StateCodec::Fp32, error) ||
         !conversation_detail::convert_state(c.dead, codec, dead, StateCodec::Fp32, error))) return false;
    if (codec != StateCodec::Fp32) { tail_bytes = tails.data(); dead_bytes = dead.data(); }
    for (size_t j = 0; j < owned_qsa(ss); ++j) {
        const auto& st = owned(ss, j);
        if (!copy(st.idx_tail, tail_bytes + j * z.tail, z.tail, error) ||
            !copy(st.idx_dead, dead_bytes + j * z.dead, z.dead, error) ||
            !copy(st.idx_block_pos, c.block_pos.data() + j * z.block_pos, z.block_pos, error)) return false;
        if (!c.ids.empty()) {
            const size_t row = c.ids.size() / strata::kernels::qsa_real_shapes().idx_block;
            if (!copy(st.idx_pooled + row * g.idx_key_dim, dead_bytes + j * z.dead, z.dead, error)) return false;
        }
    }
    const size_t tokens = c.ids.size();
    ss.ple_prev[0] = tokens >= 2 ? c.ids[tokens - 2] : -1;
    ss.ple_prev[1] = tokens >= 1 ? c.ids[tokens - 1] : -1;
    return sync(error);
}

bool conversation_checkpoints_store(const std::vector<ConversationCheckpoint>& chain, StateCodec codec,
                                    std::vector<ConversationCheckpoint>& out, std::string& error) {
    // Every array below is a fresh allocation beside the one it comes from: a park or a SAVE under memory pressure
    // can throw, and a throw that reaches the serve loop ends the engine. Reported as a refused capture instead.
    try {
        std::vector<ConversationCheckpoint> built;
        built.reserve(chain.size());
        for (const auto& c : chain)
            if (!conversation_detail::copy_checkpoint_storing(c, codec, built.emplace_back(), error)) return false;
        out = std::move(built);
    } catch (const std::exception&) {
        out = {};
        // the refusal wants a heap block too, and this is the path where there is none: never let the report throw
        try { error = "not enough memory to store the running state"; } catch (...) {}
        return false;
    }
    return true;
}

bool conversation_checkpoints_widen(std::vector<ConversationCheckpoint>& chain, std::string& error) {
    // Resuming a parked conversation no longer calls this: the chain is handed to the engine in the codecs its
    // snapshots were stored in, and conversation_checkpoint_restore widens the one a rewind selects. This stays for
    // callers that want a whole chain in fp32 up front, and is tested for it.
    // Widening allocates the fp32 array beside the coded one, so the same allocation failure as above can happen
    // here, and a caller that cannot admit it has a `false` to act on rather than an exception. convert_state
    // builds beside and swaps in, so a chain left half-widened still says per checkpoint how each array is held.
    try {
        for (auto& c : chain) {
            StateCodec codec = StateCodec::Fp32;
            if (!codec_of(c, codec, error)) return false;
            if (codec == StateCodec::Fp32) continue;   // the live chain is these bytes already
            if (!conversation_detail::convert_state(c.gdn, codec, c.gdn, StateCodec::Fp32, error) ||
                !conversation_detail::convert_state(c.ple, codec, c.ple, StateCodec::Fp32, error) ||
                !conversation_detail::convert_state(c.tails, codec, c.tails, StateCodec::Fp32, error) ||
                !conversation_detail::convert_state(c.dead, codec, c.dead, StateCodec::Fp32, error)) return false;
            c.state_codec = 0;
            if (!conversation_checkpoints_widen(c.stage_parts, error)) return false;
        }
    } catch (const std::exception&) {
        try { error = "not enough memory to widen the running state"; } catch (...) {}
        return false;
    }
    return true;
}

bool conversation_snapshot_bytes(const ConversationView& view, const SessionState& ss,
                                 const ModelGeometry& g, const QsaState* draft, size_t& bytes, std::string& error) {
    bytes = 0;
    if (!view_validate(view, ss, g, error)) return false;
    ConversationStateSizes z;
    if (!conversation_session_sizes(g, ss, z, error)) return false;
    size_t ids = 0, images = 0, checkpoints = 0, layers = 0, tails = 0, dead = 0, positions = 0;
    const auto qsa = (uint64_t) owned_qsa(ss);
    if (!product(ids, {view.ids.size(), sizeof(int32_t)}) ||
        !product(images, {view.images.size(), sizeof(ConversationImageKey)}) ||
        !product(checkpoints, {view.checkpoints.size(), sizeof(ConversationCheckpoint)}) ||
        !product(layers, {qsa + (draft ? 1 : 0), sizeof(ConversationKv)}) ||
        !product(tails, {qsa, z.tail}) || !product(dead, {qsa, z.dead}) ||
        !product(positions, {qsa, z.block_pos})) return fail(error, "snapshot metadata byte count overflow");
    for (size_t n : {ids, images, checkpoints, layers, positions})
        if (!add(bytes, n)) return fail(error, "snapshot byte count overflow");
    // The live running state is written the way `view.codec` writes it, exactly like the checkpoints below, so it is
    // estimated that way too: charging it at fp32 while capturing bf16 asks the cache for twice what the park needs
    // and refuses a park that fits. `positions` is int32 block positions and stays its own size whatever the codec.
    const size_t live_gdn = state_codec_size(z.gdn, view.codec);
    const size_t live_ple = ss.ple_hist ? state_codec_size(z.ple, view.codec) : 0;
    const size_t live_tails = state_codec_size(tails, view.codec);
    const size_t live_dead = state_codec_size(dead, view.codec);
    for (size_t n : {live_gdn, live_ple, live_tails, live_dead})
        if (!add(bytes, n)) return fail(error, "snapshot byte count overflow");
    for (const auto& c : view.checkpoints)
        if (!metadata_bytes(c, bytes, view.codec)) return fail(error, "checkpoint byte count overflow");
    // a coded capture crosses the device boundary one slice at a time, and that slice is held while it does: no
    // more of the running state than one slice, and no more than one slice at all
    if (view.codec != StateCodec::Fp32 &&
        !add(bytes, std::min(z.gdn, kStateCodecChunkFloats * sizeof(float))))
        return fail(error, "snapshot byte count overflow");
    const int64_t upto = (int64_t) view.ids.size(); // view_validate bounds this by signed max_cells
    for (uint64_t i = 0; i < qsa + (draft ? 1 : 0); ++i) {
        const auto& st = i == qsa ? *draft : owned(ss, (size_t) i);
        const size_t n = conversation_kv_bytes(st, g, upto, i != qsa);
        if (!n || !add(bytes, n)) return fail(error, "invalid or overflowing K/V byte estimate");
    }
    return true;
}

bool conversation_snapshot_capture_bytes(const ConversationKvReuse& reuse, const ConversationView& view,
                                         const SessionState& ss, const ModelGeometry& g,
                                         const QsaState* draft, size_t& bytes, std::string& error) {
    if (!conversation_snapshot_bytes(view, ss, g, draft, bytes, error)) return false;
    if (reuse.kv.empty()) return true;
    const size_t layers = owned_qsa(ss) + (draft ? 1 : 0);
    if (reuse.kv.size() != layers || reuse.unchanged_tokens < 0 ||
        reuse.unchanged_tokens > reuse.captured_tokens || reuse.unchanged_tokens > int64_t(view.ids.size()))
        return fail(error, "invalid retained K/V prefix");
    for (size_t i = 0; i < layers; ++i) {
        const bool index = i < owned_qsa(ss);
        const auto& st = index ? owned(ss, i) : *draft;
        if (!conversation_kv_validate(reuse.kv[i], st, g, reuse.captured_tokens, index, error)) return false;
        const size_t fresh = conversation_kv_bytes(st, g, int64_t(view.ids.size()), index);
        size_t retained = 0;
        if (!conversation_kv_capture_bytes(reuse.kv[i], st, g, int64_t(view.ids.size()), index, retained, error)) return false;
        bytes -= fresh;
        if (!add(bytes, retained)) return fail(error, "retained K/V allocation overflow");
    }
    size_t directory = 0;
    if (!product(directory, {reuse.kv.capacity() - layers, sizeof(ConversationKv)}) || !add(bytes, directory))
        return fail(error, "retained K/V directory overflow");
    return true;
}

bool conversation_snapshot_save(SavedConversation& image, const ConversationView& view,
                                const SessionState& ss, const ModelGeometry& g,
                                const QsaState* draft, std::string& error,
                                ConversationKvReuse reuse, size_t* reused_bytes) {
    size_t estimate = 0;
    if (!conversation_snapshot_capture_bytes(reuse, view, ss, g, draft, estimate, error) || !sync(error)) return false;
    // Build into a new object so a failure cannot publish a partial snapshot.
    SavedConversation captured;
    captured.geometry = geometry_key(g);
    captured.layer_lo = ss.layer_lo; captured.layer_hi = ss.layer_hi;
    captured.live.ids = view.ids; captured.live.imgs = view.images;
    captured.cvec = view.cvec;
    if (!conversation_checkpoints_store(view.checkpoints, view.codec, captured.checkpoints, error)) return false;
    const int64_t unchanged = reuse.kv.empty() ? 0 : reuse.unchanged_tokens;
    captured.kv = std::move(reuse.kv);
    const size_t layers = owned_qsa(ss);
    captured.kv.resize(layers + (draft ? 1 : 0));
    captured.live.state_codec = (uint8_t) view.codec;
    if (!conversation_checkpoint_save(captured.live, ss, g, error)) return false;
    const int64_t upto = (int64_t) view.ids.size();
    for (size_t j = 0; j < layers; ++j)
        if (!conversation_kv_save(captured.kv[j], owned(ss, j), g, upto, true, error,
                                  unchanged, reused_bytes)) return false;
    // The draft's final cell may not have been computed when the output cap was
    // reached. Refresh that page even when the main prefix continued unchanged.
    if (draft && !conversation_kv_save(captured.kv.back(), *draft, g, upto, false, error,
                                       std::max<int64_t>(0, unchanged - 1), reused_bytes)) return false;
    image = std::move(captured);
    return true;
}

bool conversation_snapshot_sources(SavedConversation& meta, std::vector<SessionKvSource>& sources,
                                   const ConversationView& view, const SessionState& ss, const ModelGeometry& g,
                                   const QsaState& draft, std::string& error) {
    if (!view_validate(view, ss, g, error) || !sync(error)) return false;
    SavedConversation captured;
    captured.geometry = geometry_key(g);
    captured.layer_lo = ss.layer_lo; captured.layer_hi = ss.layer_hi;
    captured.live.ids = view.ids; captured.live.imgs = view.images;
    captured.cvec = view.cvec;
    if (!conversation_checkpoints_store(view.checkpoints, view.codec, captured.checkpoints, error)) return false;
    captured.live.state_codec = (uint8_t) view.codec;
    if (!conversation_checkpoint_save(captured.live, ss, g, error)) return false;
    const int64_t upto = (int64_t) view.ids.size();
    const size_t layers = owned_qsa(ss);
    std::vector<SessionKvSource> out(layers + 1);
    for (size_t j = 0; j < layers; ++j)
        if (!conversation_kv_source(out[j], owned(ss, j), g, upto, true, error)) return false;
    if (!conversation_kv_source(out.back(), draft, g, upto, false, error)) return false;
    meta = std::move(captured);
    sources = std::move(out);
    return true;
}

bool conversation_session_read_limits(SessionReadLimits& limits, const SessionState& ss, const ModelGeometry& g,
                                      const QsaState& draft, uint64_t max_tokens, uint64_t max_checkpoints,
                                      std::string& error) {
    ConversationStateSizes z;
    if (!conversation_session_sizes(g, ss, z, error)) return false;
    if (ss.max_cells < 0) return fail(error, "invalid session cells");
    const uint64_t tokens = std::min<uint64_t>(max_tokens, (uint64_t) ss.max_cells);
    const size_t layers = owned_qsa(ss);
    if ((owned_qsa(ss) && !ss.qsa_states)) return fail(error, "invalid session running-state targets");
    size_t tails = 0, dead = 0, block_pos = 0;
    if (!product(tails, {layers, z.tail}) || !product(dead, {layers, z.dead}) ||
        !product(block_pos, {layers, z.block_pos})) return fail(error, "indexer byte count overflow");
    SessionReadLimits l = limits;   // keeps the caller's admit / progress / max_file_bytes
    l.max_tokens = tokens;
    l.max_checkpoints = max_checkpoints;
    l.max_kv_layers = layers + 1;
    l.geometry = geometry_key(g);
    l.layer_range = std::make_pair(ss.layer_lo, ss.layer_hi);
    l.max_state_bytes = {z.gdn, ss.ple_hist ? z.ple : 0, tails, dead, block_pos};
    l.max_kv_bytes.assign(layers + 1, {});
    for (size_t j = 0; j < layers; ++j) {
        const auto& st = owned(ss, j);
        const int64_t upto = (int64_t) std::min<uint64_t>(tokens, (uint64_t) std::max<int64_t>(st.max_cells, 0));
        if (!conversation_kv_part_sizes(st, g, upto, true, l.max_kv_bytes[j], error)) return false;
    }
    const int64_t dupto = (int64_t) std::min<uint64_t>(tokens, (uint64_t) std::max<int64_t>(draft.max_cells, 0));
    if (!conversation_kv_part_sizes(draft, g, dupto, false, l.max_kv_bytes.back(), error)) return false;
    limits = std::move(l);
    return true;
}

bool conversation_snapshot_validate(const SavedConversation& image, const SessionState& ss,
                                    const ModelGeometry& g, const QsaState* draft, std::string& error) {
    if (!image.live.stage_parts.empty())
        return fail(error, "a running checkpoint's stage parts belong in the stage images, not in live.stage_parts");
    if (image.geometry != geometry_key(g)) return fail(error, "incompatible runtime geometry");
    // an image holds exactly one carve's running state and K/V: same layer range or nothing
    if (image.layer_lo != ss.layer_lo || image.layer_hi != ss.layer_hi)
        return fail(error, "snapshot from another session layer range");
    const ConversationView view{image.live.ids, image.live.imgs, image.checkpoints, image.cvec};
    if (!view_validate(view, ss, g, error) || !conversation_checkpoint_validate(image.live, ss, g, error)) return false;
    const size_t layers = owned_qsa(ss);
    if (image.kv.size() != layers + (draft ? 1 : 0)) return fail(error, "invalid K/V layer count");
    const int64_t upto = (int64_t) image.live.ids.size();
    for (size_t j = 0; j < layers; ++j)
        if (!conversation_kv_validate(image.kv[j], owned(ss, j), g, upto, true, error)) return false;
    return !draft || conversation_kv_validate(image.kv.back(), *draft, g, upto, false, error);
}

ConversationRestore conversation_snapshot_restore(const SavedConversation& image, SessionState& ss,
                                                   const ModelGeometry& g, const QsaState* draft, std::string& error) {
    if (!conversation_snapshot_validate(image, ss, g, draft, error)) return ConversationRestore::invalid;
    if (!sync(error)) return ConversationRestore::transfer_failed;
    const int64_t upto = (int64_t) image.live.ids.size();
    for (size_t j = 0; j < owned_qsa(ss); ++j)
        if (!conversation_kv_restore(image.kv[j], owned(ss, j), g, upto, true, error))
            return ConversationRestore::transfer_failed;
    if ((draft && !conversation_kv_restore(image.kv.back(), *draft, g, upto, false, error)) ||
        !conversation_checkpoint_restore(image.live, ss, g, error)) return ConversationRestore::transfer_failed;
    return ConversationRestore::restored;
}

// the draft layer's K/V included (the stage that owns the draft head, or no layer split)
bool conversation_snapshot_bytes(const ConversationView& view, const SessionState& ss, const ModelGeometry& g,
                                 const QsaState& draft, size_t& bytes, std::string& error) {
    return conversation_snapshot_bytes(view, ss, g, &draft, bytes, error);
}
bool conversation_snapshot_capture_bytes(const ConversationKvReuse& reuse, const ConversationView& view,
                                         const SessionState& ss, const ModelGeometry& g, const QsaState& draft,
                                         size_t& bytes, std::string& error) {
    return conversation_snapshot_capture_bytes(reuse, view, ss, g, &draft, bytes, error);
}
bool conversation_snapshot_save(SavedConversation& image, const ConversationView& view, const SessionState& ss,
                                const ModelGeometry& g, const QsaState& draft, std::string& error,
                                ConversationKvReuse reuse, size_t* reused_bytes) {
    return conversation_snapshot_save(image, view, ss, g, &draft, error, std::move(reuse), reused_bytes);
}
bool conversation_snapshot_validate(const SavedConversation& image, const SessionState& ss, const ModelGeometry& g,
                                    const QsaState& draft, std::string& error) {
    // the whole-session form: a layer split's image (stage images) is restored stage by stage, never through here
    if (!image.stage_images.empty()) return fail(error, "a layer split's image restored as a single session");
    return conversation_snapshot_validate(image, ss, g, &draft, error);
}
ConversationRestore conversation_snapshot_restore(const SavedConversation& image, SessionState& ss,
                                                   const ModelGeometry& g, const QsaState& draft, std::string& error) {
    return conversation_snapshot_restore(image, ss, g, &draft, error);
}
} // namespace strata::core
