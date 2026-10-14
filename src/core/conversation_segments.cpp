// The per-segment tier: the independent, content-addressed files a conversation is the sum of. This file holds the
// vocabulary (kinds, addresses, names), the manifest format, the writer (dedup) and the reader (restore), and the
// store's own GC. It is host code only: a segment is written and read by conversation_file.cpp, so a segment file is
// byte-for-byte a session_file_write of its slice.
#include "strata/core/conversation_segments.hpp"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iterator>
#include <limits>
#include <new>
#include <system_error>
#include <utility>

namespace strata::core {
namespace {

// Little-endian, fixed-width manifest writer with a trailing hash of everything before it (same SessionHasher as the
// session file and the sidecar: corruption detection, not authentication).
struct ManifestWriter {
    std::vector<uint8_t> bytes;
    SessionHasher hash{0};
    void raw(const void* p, size_t n) {
        const uint8_t* b = static_cast<const uint8_t*>(p);
        bytes.insert(bytes.end(), b, b + n);
        hash.update(p, n);
    }
    void u8(uint8_t v) { raw(&v, 1); }
    void u16(uint16_t v) { raw(&v, 2); }
    void u32(uint32_t v) { raw(&v, 4); }
    void u64(uint64_t v) { raw(&v, 8); }
    void i64(int64_t v) { raw(&v, 8); }
    void finish() {
        const uint64_t d = hash.digest();
        const uint8_t* b = reinterpret_cast<const uint8_t*>(&d);
        bytes.insert(bytes.end(), b, b + 8);
    }
};

// The reader of the same stream. Bounds-check against (n - 8) so the trailing digest is never consumed as a field,
// and finish() requires the stream to end exactly on it and the hash to agree.
struct ManifestReader {
    const uint8_t* p = nullptr;
    size_t n = 0, at = 0;
    SessionHasher hash{0};
    bool raw(void* dst, size_t k) {
        if (k > n || at + k + 8 > n) return false;
        std::memcpy(dst, p + at, k);
        hash.update(p + at, k);
        at += k;
        return true;
    }
    bool u8(uint8_t& v) { return raw(&v, 1); }
    bool u16(uint16_t& v) { return raw(&v, 2); }
    bool u32(uint32_t& v) { return raw(&v, 4); }
    bool u64(uint64_t& v) { return raw(&v, 8); }
    bool i64(int64_t& v) { return raw(&v, 8); }
    bool finish() {
        if (at + 8 != n) return false;
        uint64_t want = 0;
        std::memcpy(&want, p + at, 8);
        return want == hash.digest();
    }
};

void manifest_serialize(const SegmentManifest& manifest, std::vector<uint8_t>& out) {
    ManifestWriter w;
    w.raw(kSegmentManifestMagic, 8);
    w.u32(kSegmentManifestVersion);
    w.u64(manifest.model);
    w.u64(manifest.config);
    w.u8(manifest.cvec ? 1 : 0);
    w.u64(manifest.conversation_key);
    w.u64(manifest.head_frontier);
    w.i64(manifest.turn_token);
    w.u32((uint32_t) manifest.segments.size());
    for (const SegmentRef& ref : manifest.segments) {
        w.u64(ref.seg_id);
        w.u8((uint8_t) ref.kind);
        w.u16(ref.domain);
        w.u64(ref.token_begin);
        w.u64(ref.token_end);
        w.i64(ref.role_token);
        w.u8(ref.has_state ? 1 : 0);
        w.u32(ref.stages);
        w.u64(ref.bytes);
        for (uint64_t part : ref.stage_bytes) w.u64(part);
    }
    w.u64(manifest.live_tokens);
    w.finish();
    out = std::move(w.bytes);
}

bool manifest_parse(const uint8_t* data, size_t n, SegmentManifest& out, std::string& error) {
    ManifestReader r{data, n};
    char magic[8];
    if (!r.raw(magic, 8) || std::memcmp(magic, kSegmentManifestMagic, 8) != 0) {
        error = "unknown segment manifest magic"; return false;
    }
    uint32_t version = 0;
    if (!r.u32(version) || version != kSegmentManifestVersion) {
        error = "unknown segment manifest version"; return false;
    }
    uint8_t cvec = 0;
    if (!r.u64(out.model) || !r.u64(out.config) || !r.u8(cvec) || cvec > 1) {
        error = "bad segment manifest header"; return false;
    }
    out.cvec = cvec != 0;
    if (!r.u64(out.conversation_key) || !r.u64(out.head_frontier) || !r.i64(out.turn_token)) {
        error = "truncated segment manifest header"; return false;
    }
    uint32_t segments = 0;
    if (!r.u32(segments) || segments > kMaxSegments) { error = "bad segment count"; return false; }
    out.segments.clear();
    out.segments.reserve(segments);
    for (uint32_t i = 0; i < segments; ++i) {
        SegmentRef ref;
        uint8_t kind = 0, has_state = 0;
        if (!r.u64(ref.seg_id) || !r.u8(kind) || !r.u16(ref.domain) || !r.u64(ref.token_begin) ||
            !r.u64(ref.token_end) || !r.i64(ref.role_token) || !r.u8(has_state) || !r.u32(ref.stages) ||
            !r.u64(ref.bytes)) {
            error = "truncated segment reference"; return false;
        }
        ref.kind = (SegmentKind) kind;
        ref.has_state = has_state != 0;
        if (ref.stages > kMaxSegmentStages) { error = "too many stage files in a segment"; return false; }
        ref.stage_bytes.resize((size_t) ref.stages + 1);
        for (uint64_t& part : ref.stage_bytes)
            if (!r.u64(part)) { error = "truncated segment stage bytes"; return false; }
        out.segments.push_back(std::move(ref));
    }
    if (!r.u64(out.live_tokens)) { error = "truncated segment manifest tail"; return false; }
    if (!r.finish()) { error = "segment manifest checksum or length mismatch"; return false; }
    return true;
}

// The sum of images: the ids and the attention K/V concatenate; the recurrent state is the LAST part's snapshot (it
// is a running summary, not a sum - §5.2 of the design). Used both by the restore and by the materialization policy.
ConversationBuffer concat_buffer(const std::vector<SavedConversation>& parts, size_t layer,
                                 ConversationBuffer ConversationKv::*member) {
    size_t total = 0;
    for (const SavedConversation& part : parts)
        if (layer < part.kv.size()) total += (part.kv[layer].*member).size();
    ConversationBuffer out;
    out.resize(total);
    size_t at = 0;
    for (const SavedConversation& part : parts) {
        if (layer >= part.kv.size()) continue;
        const ConversationBuffer& source = part.kv[layer].*member;
        const size_t count = source.size();
        if (count) {
            out.visit(at, count, [&](uint8_t* dst, size_t n, size_t off) {
                return source.visit(off - at, n, [&](const uint8_t* src, size_t c, size_t) {
                    std::memcpy(dst, src, c);
                    return true;
                });
            });
        }
        at += count;
    }
    return out;
}

SavedConversation concat_images(const std::vector<SavedConversation>& parts) {
    SavedConversation out;
    if (parts.empty()) return out;
    out.geometry = parts.front().geometry;
    out.layer_lo = parts.front().layer_lo;
    out.layer_hi = parts.front().layer_hi;
    out.cvec = parts.front().cvec;
    for (const SavedConversation& part : parts)
        out.checkpoints.insert(out.checkpoints.end(), part.checkpoints.begin(), part.checkpoints.end());
    // With --layer-split the later stages travel with each segment: the sum keeps them in order (one image per
    // stage per segment).
    for (const SavedConversation& part : parts)
        out.stage_images.insert(out.stage_images.end(), part.stage_images.begin(), part.stage_images.end());
    std::vector<int32_t> ids;
    for (const SavedConversation& part : parts) ids.insert(ids.end(), part.live.ids.begin(), part.live.ids.end());
    out.live = parts.back().live;   // the recurrent snapshot is the last segment's, valid only under its chain
    out.live.ids = std::move(ids);
    const size_t layers = parts.front().kv.size();
    out.kv.resize(layers);
    for (size_t i = 0; i < layers; ++i) {
        const ConversationKv& first = parts.front().kv[i];
        ConversationKv layer;
        layer.format = first.format; layer.cells = first.cells; layer.heads = first.heads;
        layer.head_dim = first.head_dim; layer.page_size = first.page_size;
        layer.pooled_rows = first.pooled_rows; layer.idx_dim = first.idx_dim;
        layer.k = concat_buffer(parts, i, &ConversationKv::k);
        layer.v = concat_buffer(parts, i, &ConversationKv::v);
        layer.k_scale = concat_buffer(parts, i, &ConversationKv::k_scale);
        layer.v_scale = concat_buffer(parts, i, &ConversationKv::v_scale);
        layer.pooled = concat_buffer(parts, i, &ConversationKv::pooled);
        out.kv[i] = std::move(layer);
    }
    return out;
}

// ---- Slicing a captured image into the per-segment images store_image() writes. The token range [b, e) of the whole
// image becomes a byte range of each K/V part. The boundary for token t is `size * t / cells` (cells is the image's
// page-aligned capacity, the row count the buffer holds), computed the SAME way for the end of one segment and the
// start of the next, so the slices tile the buffer EXACTLY: the sum of the parts is the whole, for every plan. The
// LAST segment takes the tail of each buffer (the partial page and the indexer's moving spare row), so no byte is
// lost. The K/V geometry metadata travels unchanged; the running state and the checkpoints travel only with the
// segment that has_state (the last one) - the sum keeps the last part's state, so the earlier ones need none.
ConversationBuffer slice_buffer(const ConversationBuffer& src, size_t begin, size_t end) {
    ConversationBuffer out;
    if (end <= begin) return out;
    out.resize(end - begin);
    out.visit(0, end - begin, [&](uint8_t* p, size_t n, size_t at) {
        return src.read(p, begin + at, n);
    });
    return out;
}

void slice_kv(const ConversationKv& src, ConversationKv& dst, size_t b, size_t e, bool last) {
    dst.format = src.format; dst.cells = src.cells; dst.heads = src.heads; dst.head_dim = src.head_dim;
    dst.page_size = src.page_size; dst.pooled_rows = src.pooled_rows; dst.idx_dim = src.idx_dim;
    const uint64_t denom = src.cells > 0 ? (uint64_t) src.cells : 1;
    auto cut = [&](const ConversationBuffer& buf, size_t& lo, size_t& hi) {
        const uint64_t size = (uint64_t) buf.size();
        lo = (size_t) (size * b / denom);
        hi = last ? (size_t) size : (size_t) (size * e / denom);
        if (hi < lo) hi = lo;
        if (hi > (size_t) size) hi = (size_t) size;
    };
    size_t lo = 0, hi = 0;
    cut(src.k, lo, hi);        dst.k = slice_buffer(src.k, lo, hi);
    cut(src.v, lo, hi);        dst.v = slice_buffer(src.v, lo, hi);
    cut(src.k_scale, lo, hi);  dst.k_scale = slice_buffer(src.k_scale, lo, hi);
    cut(src.v_scale, lo, hi);  dst.v_scale = slice_buffer(src.v_scale, lo, hi);
    cut(src.pooled, lo, hi);   dst.pooled = slice_buffer(src.pooled, lo, hi);
}

SavedConversation slice_image(const SavedConversation& image, size_t b, size_t e, bool last) {
    SavedConversation out;
    out.geometry = image.geometry;
    out.layer_lo = image.layer_lo;
    out.layer_hi = image.layer_hi;
    out.cvec = image.cvec;
    out.live.ids.assign(image.live.ids.begin() + (std::ptrdiff_t) b, image.live.ids.begin() + (std::ptrdiff_t) e);
    // The recurrent state is a running summary, NOT a sum: a segment carries the snapshot at ITS OWN frontier. The
    // last segment carries the captured live state; a segment that ends exactly on one of the image's checkpoints
    // carries that checkpoint - this is how the effort frontier (the engine checkpoints in FRONT of the effort turn)
    // gets its state, so a restore up to it is usable. concat keeps the last part's state, so the frontier's own
    // segment is the one that must carry it. A frontier with neither a live match nor a checkpoint has no snapshot
    // (a declared limitation).
    const ConversationCheckpoint* state = nullptr;
    if (last) {
        state = &image.live;
        out.checkpoints = image.checkpoints;
    } else {
        const std::vector<int32_t> head(image.live.ids.begin(), image.live.ids.begin() + (std::ptrdiff_t) e);
        for (const ConversationCheckpoint& c : image.checkpoints)
            if (c.stage_parts.empty() && c.ids == head) { state = &c; break; }
    }
    if (state != nullptr) {
        out.live.imgs = state->imgs;
        out.live.gdn = state->gdn;
        out.live.ple = state->ple;
        out.live.tails = state->tails;
        out.live.dead = state->dead;
        out.live.block_pos = state->block_pos;
        out.live.used = state->used;
    }
    out.kv.resize(image.kv.size());
    for (size_t layer = 0; layer < image.kv.size(); ++layer) slice_kv(image.kv[layer], out.kv[layer], b, e, last);
    out.stage_images.resize(image.stage_images.size());
    for (size_t s = 0; s < image.stage_images.size(); ++s)
        out.stage_images[s] = slice_image(image.stage_images[s], b, e, last);
    return out;
}

} // namespace

const char* segment_kind_name(SegmentKind kind) {
    switch (kind) {
        case SegmentKind::Sys: return "sys";
        case SegmentKind::Read: return "read";
        case SegmentKind::Write: return "write";
        case SegmentKind::ToolCall: return "tool_call";
        case SegmentKind::ToolResult: return "tool_result";
        case SegmentKind::Think: return "think";
        case SegmentKind::Answer: return "answer";
        case SegmentKind::User: return "user";
        case SegmentKind::Other: return "other";
    }
    return "other";
}

// The chain's head: H(model || config). The first segment (the systemprompt) folds this in, so a chat's systemprompt
// address depends only on its tokens and the identity - two chats of one client share it and dedup.
uint64_t segment_head_frontier(uint64_t model, uint64_t config) {
    SessionIdentityBuilder builder(0x5353454751ull);   // "SSEGQ" seed, fixed
    builder.u64("model", model);
    builder.u64("config", config);
    return builder.digest();
}

// H(prev_frontier || kind || domain || tokens || role_token || model || config). SessionIdentityBuilder length-
// delimits every field, so no two field lists give the same stream; the tokens are their int32 bytes.
uint64_t segment_hash(SegmentKind kind, uint16_t domain, const std::vector<int32_t>& tokens,
                      int64_t role_token, uint64_t prev_frontier, uint64_t model, uint64_t config) {
    SessionIdentityBuilder builder(0x5345474d31ull);   // "SEGM1" seed, fixed
    builder.u64("prev", prev_frontier);
    builder.u64("kind", (uint64_t) kind);
    builder.u64("domain", domain);
    builder.i64("role", role_token);
    builder.i64("count", (int64_t) tokens.size());
    if (!tokens.empty()) builder.bytes("tokens", tokens.data(), tokens.size() * sizeof(int32_t));
    builder.u64("model", model);
    builder.u64("config", config);
    return builder.digest();
}

std::string segment_file_name(uint64_t seg_id, size_t stage) {
    char hex[17];
    std::snprintf(hex, sizeof hex, "%016llx", (unsigned long long) seg_id);
    std::string name(hex, 16);
    if (stage == 0) return name + kSegmentExt;
    return name + ".stage" + std::to_string(stage) + kSegmentExt;
}

std::string segment_manifest_name(uint64_t serial) {
    return std::string(kSegmentFilePrefix) + std::to_string(serial) + kSegmentManifestExt;
}

std::vector<SegmentPlan> segment_plan_split(const std::vector<int32_t>& ids, int64_t turn_token,
                                            int64_t tail_role_token) {
    std::vector<SegmentPlan> plan;
    const size_t n = ids.size();
    if (n == 0) return plan;
    // The end of the system prompt: the FIRST turn boundary. With no --turn-token the systemprompt is not separable
    // and the whole prompt becomes one legacy segment (the harness declared no phase).
    size_t sys_end = n;
    if (turn_token >= 0)
        for (size_t i = 0; i < n; ++i)
            if (ids[i] == (int32_t) turn_token) { sys_end = i; break; }
    // The effort turn: the LAST --tail-role-token. It is the frontier of the thinking-effort case, so a change of
    // effort only rewrites [tail_pos, n).
    size_t tail_pos = sys_end;
    if (tail_role_token >= 0)
        for (size_t i = n; i-- > 0;)
            if (ids[i] == (int32_t) tail_role_token) { tail_pos = i; break; }
    if (tail_pos < sys_end || tail_pos > n) tail_pos = sys_end;
    if (sys_end == 0 && tail_pos >= n) {   // no usable boundary at all: one segment covers the whole prompt
        plan.push_back({SegmentKind::Other, 0, (uint64_t) n, -1, true});
        return plan;
    }
    if (sys_end > 0) plan.push_back({SegmentKind::Sys, 0, (uint64_t) sys_end, -1, false});
    if (tail_pos > sys_end) plan.push_back({SegmentKind::Other, 0, (uint64_t) (tail_pos - sys_end), -1, false});
    if (tail_pos < n) plan.push_back({SegmentKind::Answer, 0, (uint64_t) (n - tail_pos), tail_role_token, false});
    if (plan.empty()) plan.push_back({SegmentKind::Other, 0, (uint64_t) n, -1, false});
    plan.back().has_state = true;
    return plan;
}

std::string SegmentStore::segment_path(const std::string& directory, uint64_t seg_id, size_t stage) {
    std::filesystem::path base(directory);
    return (base / segment_file_name(seg_id, stage)).string();
}

bool SegmentStore::valid_manifest_chain(const SegmentManifest& manifest) {
    uint64_t cursor = 0;
    for (const SegmentRef& ref : manifest.segments) {
        if (ref.token_begin != cursor) return false;           // a hole or an overlap: the chain is broken
        if (ref.token_end < ref.token_begin) return false;
        cursor = ref.token_end;
    }
    return cursor == manifest.live_tokens;                     // the ranges must cover the whole prompt
}

const SegmentStore::ManifestEntry* SegmentStore::find_manifest(const std::string& path) const {
    for (const ManifestEntry& entry : manifests_)
        if (entry.path.string() == path) return &entry;
    return nullptr;
}

bool SegmentStore::read_manifest(const std::filesystem::path& path, SegmentManifest& out, bool& other_identity,
                                 std::string& error) const {
    other_identity = false;
    std::ifstream in(path, std::ios::binary);
    if (!in) { error = "cannot open segment manifest"; return false; }
    std::vector<uint8_t> data((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    if (in.bad()) { error = "cannot read segment manifest"; return false; }
    if (!manifest_parse(data.data(), data.size(), out, error)) return false;
    if (out.model != identity_.model || out.config != identity_.config) { other_identity = true; return false; }
    if (!valid_manifest_chain(out)) { error = "segment manifest is not a valid chain"; return false; }
    return true;
}

bool SegmentStore::write_manifest(const std::filesystem::path& path, const SegmentManifest& manifest,
                                  std::string& error) const {
    std::vector<uint8_t> data;
    manifest_serialize(manifest, data);
    const std::filesystem::path temporary(path.string() + ".tmp");
    {
        std::ofstream out(temporary, std::ios::binary | std::ios::trunc);
        if (!out) { error = "cannot create segment manifest"; return false; }
        out.write(reinterpret_cast<const char*>(data.data()), (std::streamsize) data.size());
        out.flush();
        if (!out) { error = "cannot write segment manifest"; return false; }
    }
    std::error_code ec;
    std::filesystem::rename(temporary, path, ec);
    if (ec) { std::filesystem::remove(temporary, ec); error = "cannot commit segment manifest"; return false; }
    return true;
}

std::filesystem::path SegmentStore::next_manifest_path() {
    std::error_code ec;
    for (;;) {
        std::filesystem::path path = directory_ / segment_manifest_name(++serial_);
        if (!std::filesystem::exists(path, ec) && !ec) return path;
        ec.clear();
    }
}

void SegmentStore::index_segment_file(uint64_t seg_id, uint32_t stages, uint64_t bytes, SegmentKind kind,
                                      std::filesystem::file_time_type stamp) {
    auto it = segment_files_.find(seg_id);
    if (it != segment_files_.end()) return;   // already counted: a segment's bytes are counted once
    SegmentFile file;
    file.seg_id = seg_id;
    file.stages = stages;
    file.bytes = bytes;
    file.kind = kind;
    file.stamp = stamp;
    segment_files_.emplace(seg_id, file);
    segment_bytes_ += bytes;
}

void SegmentStore::bump_refcounts(const SegmentManifest& manifest) {
    for (const SegmentRef& ref : manifest.segments) {
        ++segment_refs_[ref.seg_id];
        auto it = segment_files_.find(ref.seg_id);
        if (it != segment_files_.end()) it->second.kind = ref.kind;   // the first listing sets the GC priority
    }
}

void SegmentStore::drop_refcounts(const SegmentManifest& manifest) {
    for (const SegmentRef& ref : manifest.segments) {
        auto it = segment_refs_.find(ref.seg_id);
        if (it == segment_refs_.end()) continue;
        if (--it->second == 0) {
            segment_refs_.erase(it);
            remove_segment(ref.seg_id);
        }
    }
}

void SegmentStore::remove_segment(uint64_t seg_id) {
    auto it = segment_files_.find(seg_id);
    if (it == segment_files_.end()) { segment_refs_.erase(seg_id); return; }
    const uint32_t stages = it->second.stages;
    const uint64_t bytes = it->second.bytes;
    std::error_code ec;
    std::filesystem::remove(segment_path(directory_.string(), seg_id, 0), ec);
    for (size_t k = 1; k <= stages; ++k) {
        ec.clear();
        std::filesystem::remove(segment_path(directory_.string(), seg_id, k), ec);
    }
    segment_bytes_ -= segment_bytes_ < bytes ? segment_bytes_ : bytes;
    segment_files_.erase(it);
    segment_refs_.erase(seg_id);
}

bool SegmentStore::write_segment_files(const SegmentInput& input, uint64_t seg_id, std::string& error,
                                       std::vector<uint64_t>* created) {
    const std::string base = directory_.string();
    const std::string first = segment_path(base, seg_id, 0);
    std::error_code ec;
    const bool existed = std::filesystem::exists(first, ec) && !ec;
    std::vector<std::pair<std::string, uint64_t>> wrote;
    auto ensure = [&](size_t stage, const SavedConversation& image) -> bool {
        const std::string path = segment_path(base, seg_id, stage);
        std::error_code e;
        if (std::filesystem::exists(path, e) && !e) return true;   // dedup: an identical file is never rewritten
        size_t bytes = 0;
        SessionWriteOptions options;
        options.durable = true;
        SessionStatus status;
        if (!session_file_write(path, image, identity_, bytes, error, options, &status)) return false;
        wrote.emplace_back(path, (uint64_t) bytes);
        return true;
    };
    bool ok = ensure(0, input.image);
    for (size_t k = 0; ok && k < input.stage_images.size(); ++k) ok = ensure(k + 1, input.stage_images[k]);
    if (!ok) {
        for (const auto& file : wrote) std::filesystem::remove(file.first, ec);   // never leave a half segment
        return false;
    }
    if (!wrote.empty()) {
        uint64_t added = 0;
        for (const auto& file : wrote) added += file.second;
        if (!existed) {
            index_segment_file(seg_id, (uint32_t) input.stage_images.size(), added, input.kind,
                               std::filesystem::file_time_type::clock::now());
            if (created) created->push_back(seg_id);
        } else {
            auto it = segment_files_.find(seg_id);
            if (it != segment_files_.end()) { it->second.bytes += added; segment_bytes_ += added; }
        }
    }
    return true;
}

std::vector<SegmentInput> SegmentStore::materialize(const std::vector<SegmentInput>& inputs) const {
    if (min_tokens_ == 0 || inputs.empty()) return inputs;
    std::vector<SegmentInput> out;
    for (const SegmentInput& in : inputs) {
        const bool fuse = !out.empty() && in.kind != SegmentKind::Sys && out.back().kind != SegmentKind::Sys &&
                          in.tokens.size() < min_tokens_;
        if (!fuse) { out.push_back(in); continue; }
        SegmentInput& previous = out.back();
        std::vector<SavedConversation> parts{previous.image, in.image};
        previous.tokens.insert(previous.tokens.end(), in.tokens.begin(), in.tokens.end());
        previous.image = concat_images(parts);
        previous.has_state = in.has_state;
        previous.kind = SegmentKind::Other;   // the fused segment is the escape hatch
        if (!in.stage_images.empty()) previous.stage_images = in.stage_images;
    }
    return out;
}

bool SegmentStore::write_segment(const SegmentInput& input, uint64_t prev_frontier, uint64_t& seg_id_out,
                                 std::string& error) {
    std::lock_guard<std::mutex> lock(mu_);
    if (!enabled_) { error = "segment store is disabled"; return false; }
    if (input.tokens.empty()) { error = "a segment needs at least one token"; return false; }
    const uint64_t seg_id = segment_hash(input.kind, input.domain, input.tokens, input.role_token, prev_frontier,
                                         identity_.model, identity_.config);
    if (!write_segment_files(input, seg_id, error, nullptr)) return false;
    seg_id_out = seg_id;
    return true;
}

bool SegmentStore::store(uint64_t conversation_key, int64_t turn_token, bool cvec,
                         const std::vector<SegmentInput>& inputs, std::string& error, std::string* manifest_path) {
    std::lock_guard<std::mutex> lock(mu_);
    if (!enabled_) { error = "segment store is disabled"; return false; }
    if (inputs.empty()) { error = "store needs at least one segment"; return false; }
    const std::vector<SegmentInput> segments = materialize(inputs);
    SegmentManifest manifest;
    manifest.model = identity_.model;
    manifest.config = identity_.config;
    manifest.cvec = cvec;
    manifest.conversation_key = conversation_key;
    manifest.head_frontier = segment_head_frontier(identity_.model, identity_.config);
    manifest.turn_token = turn_token;
    uint64_t frontier = manifest.head_frontier, cursor = 0;
    std::vector<uint64_t> created;
    for (const SegmentInput& input : segments) {
        if (input.tokens.empty()) {
            for (uint64_t seg_id : created) remove_segment(seg_id);
            error = "a segment needs at least one token";
            return false;
        }
        const uint64_t seg_id = segment_hash(input.kind, input.domain, input.tokens, input.role_token, frontier,
                                             identity_.model, identity_.config);
        if (!write_segment_files(input, seg_id, error, &created)) {
            for (uint64_t id : created) remove_segment(id);   // roll back the segments this call created
            return false;
        }
        SegmentRef ref;
        ref.seg_id = seg_id;
        ref.kind = input.kind;
        ref.domain = input.domain;
        ref.token_begin = cursor;
        ref.token_end = cursor + input.tokens.size();
        ref.role_token = input.role_token;
        ref.has_state = input.has_state;
        ref.stages = (uint32_t) input.stage_images.size();
        uint64_t total = 0;
        for (size_t stage = 0; stage <= input.stage_images.size(); ++stage) {
            std::error_code ec;
            const uint64_t bytes = std::filesystem::file_size(segment_path(directory_.string(), seg_id, stage), ec);
            if (!ec) total += bytes;
            ref.stage_bytes.push_back(ec ? 0 : bytes);
        }
        ref.bytes = total;
        cursor = ref.token_end;
        frontier = seg_id;
        manifest.segments.push_back(std::move(ref));
    }
    manifest.live_tokens = cursor;
    const std::filesystem::path path = next_manifest_path();
    if (!write_manifest(path, manifest, error)) {
        for (uint64_t id : created) remove_segment(id);
        return false;
    }
    ManifestEntry entry;
    entry.path = path;
    entry.manifest = manifest;
    std::error_code ec;
    entry.stamp = std::filesystem::last_write_time(path, ec);
    if (ec) entry.stamp = std::filesystem::file_time_type::clock::now();
    manifests_.push_back(std::move(entry));
    bump_refcounts(manifest);   // the segment bytes were already indexed by write_segment_files
    if (manifest_path != nullptr) *manifest_path = path.string();
    return true;
}

// The incremental writer: ONE captured image + a plan. Each plan run is hashed with the SAME chain rule best() uses,
// so a run whose <seg_id>.seg already exists is a HIT and is neither sliced nor rewritten - a growing chat pays only
// for its new tail, and a shared systemprompt is written once. The manifest of this conversation_key is replaced in
// place (bump the new refs, then drop the ones the new chain no longer lists), so a re-park of an unchanged chain is
// a no-op file overwrite and the conversation keeps exactly one manifest.
bool SegmentStore::store_image(uint64_t conversation_key, int64_t turn_token, bool cvec, const SavedConversation& image,
                               const std::vector<SegmentPlan>& plan, std::string& error, std::string* manifest_path) {
    std::lock_guard<std::mutex> lock(mu_);
    if (!enabled_) { error = "segment store is disabled"; return false; }
    const std::vector<int32_t>& ids = image.live.ids;
    if (ids.empty() || plan.empty()) { error = "store_image needs a non-empty image and plan"; return false; }
    uint64_t total = 0;
    for (const SegmentPlan& p : plan) total += p.tokens;
    if (total != (uint64_t) ids.size()) { error = "segment plan does not cover the live ids"; return false; }
    // --conversation-cache-segment-min-tokens: a short non-system run is fused with the one before it (the same
    // policy the materialize() pass applies to SegmentInput), so a short chat does not pay one file per message. A
    // systemprompt is never a fusion partner. The fused run keeps the first run's opening role token.
    std::vector<SegmentPlan> effective;
    for (const SegmentPlan& p : plan) {
        if (p.tokens == 0) continue;
        const bool fuse = min_tokens_ > 0 && !effective.empty() && p.kind != SegmentKind::Sys &&
                          effective.back().kind != SegmentKind::Sys && p.tokens < min_tokens_;
        if (fuse) {
            effective.back().tokens += p.tokens;
            effective.back().has_state = p.has_state;
            effective.back().kind = SegmentKind::Other;
        } else {
            effective.push_back(p);
        }
    }
    if (effective.empty()) { error = "segment plan is empty"; return false; }

    SegmentManifest manifest;
    manifest.model = identity_.model;
    manifest.config = identity_.config;
    manifest.cvec = cvec;
    manifest.conversation_key = conversation_key;
    manifest.head_frontier = segment_head_frontier(identity_.model, identity_.config);
    manifest.turn_token = turn_token;
    uint64_t frontier = manifest.head_frontier, cursor = 0;
    std::vector<uint64_t> created;
    for (const SegmentPlan& p : effective) {
        if (p.tokens == 0) continue;
        const size_t begin = (size_t) cursor;
        const size_t end = begin + (size_t) p.tokens;
        const std::vector<int32_t> slice(ids.begin() + (std::ptrdiff_t) begin, ids.begin() + (std::ptrdiff_t) end);
        const uint64_t seg_id = segment_hash(p.kind, p.domain, slice, p.role_token, frontier,
                                             identity_.model, identity_.config);
        std::error_code ec;
        const bool exists = std::filesystem::exists(segment_path(directory_.string(), seg_id, 0), ec) && !ec;
        if (!exists) {                                  // idempotent: an existing segment is a HIT, never rewritten
            SegmentInput input;
            input.kind = p.kind;
            input.domain = p.domain;
            input.tokens = slice;
            input.role_token = p.role_token;
            input.has_state = p.has_state;
            input.image = slice_image(image, begin, end, end == ids.size());   // ONLY this slice is materialized + freed
            if (!write_segment_files(input, seg_id, error, &created)) {
                for (uint64_t id : created) remove_segment(id);
                return false;
            }
        }
        SegmentRef ref;
        ref.seg_id = seg_id;
        ref.kind = p.kind;
        ref.domain = p.domain;
        ref.token_begin = cursor;
        ref.token_end = cursor + p.tokens;
        ref.role_token = p.role_token;
        ref.has_state = p.has_state;
        auto fit = segment_files_.find(seg_id);
        ref.stages = fit != segment_files_.end() ? fit->second.stages : 0;
        uint64_t bytes = 0;
        for (size_t stage = 0; stage <= ref.stages; ++stage) {
            std::error_code sec;
            const uint64_t part = std::filesystem::file_size(segment_path(directory_.string(), seg_id, stage), sec);
            ref.stage_bytes.push_back(sec ? 0 : part);
            if (!sec) bytes += part;
        }
        ref.bytes = bytes;
        cursor = ref.token_end;
        frontier = seg_id;
        manifest.segments.push_back(std::move(ref));
    }
    manifest.live_tokens = cursor;

    ManifestEntry* existing = nullptr;
    for (ManifestEntry& entry : manifests_)
        if (entry.manifest.conversation_key == conversation_key && entry.manifest.cvec == cvec) {
            existing = &entry;
            break;
        }
    if (existing != nullptr) {
        SegmentManifest old = existing->manifest;
        if (!write_manifest(existing->path, manifest, error)) {
            for (uint64_t id : created) remove_segment(id);
            return false;
        }
        bump_refcounts(manifest);   // the new refs FIRST, so a shared prefix never momentarily hits refcount 0
        drop_refcounts(old);        // then the refs the new chain no longer lists (the superseded tail) go
        existing->manifest = std::move(manifest);
        std::error_code ec;
        existing->stamp = std::filesystem::last_write_time(existing->path, ec);
        if (ec) existing->stamp = std::filesystem::file_time_type::clock::now();
        if (manifest_path != nullptr) *manifest_path = existing->path.string();
        return true;
    }
    const std::filesystem::path path = next_manifest_path();
    if (!write_manifest(path, manifest, error)) {
        for (uint64_t id : created) remove_segment(id);
        return false;
    }
    ManifestEntry entry;
    entry.path = path;
    entry.manifest = manifest;
    std::error_code ec;
    entry.stamp = std::filesystem::last_write_time(path, ec);
    if (ec) entry.stamp = std::filesystem::file_time_type::clock::now();
    manifests_.push_back(std::move(entry));
    bump_refcounts(manifest);
    if (manifest_path != nullptr) *manifest_path = path.string();
    return true;
}

bool SegmentStore::open(const SegmentStoreConfig& config, SessionFileIdentity identity, std::string& error) {
    std::lock_guard<std::mutex> lock(mu_);
    enabled_ = false;
    directory_ = config.directory;
    identity_ = identity;
    budget_ = config.budget_bytes;
    min_tokens_ = config.min_tokens;
    turn_token_ = config.turn_token;
    serial_ = 0;
    segment_bytes_ = 0;
    segment_evictions_ = stale_files_kept_ = foreign_files_kept_ = orphan_files_kept_ = 0;
    manifests_.clear();
    segment_refs_.clear();
    segment_files_.clear();
    if (directory_.empty() || !identity.model || !identity.config) return true;   // inert
    try {
        std::error_code ec;
        std::filesystem::create_directories(directory_, ec);
        if (ec || !std::filesystem::is_directory(directory_, ec) || ec) {
            error = "cannot create or inspect segment directory " + directory_.string();
            return false;
        }
        std::vector<std::filesystem::path> manifest_files, segment_files;
        for (std::filesystem::directory_iterator it(directory_, ec), end; !ec && it != end; it.increment(ec)) {
            const auto status = it->symlink_status(ec);
            if (ec) break;
            if (std::filesystem::is_directory(status)) continue;
            if (!std::filesystem::is_regular_file(status)) continue;
            const std::string name = it->path().filename().string();
            const std::string ext = it->path().extension().string();
            if (ext == kSegmentManifestExt) manifest_files.push_back(it->path());
            else if (ext == kSegmentExt) segment_files.push_back(it->path());
            else if (ext == ".sess" || ext == ".meta") continue;   // the .sess tier's files: not ours, left alone
            else ++stale_files_kept_;                              // a leftover temp or an unknown kind: kept
        }
        if (ec) { error = "cannot scan segment directory " + directory_.string(); return false; }
        // Index the segment files first: the manifests then only attach refcounts and the kinds.
        std::sort(segment_files.begin(), segment_files.end());
        for (const std::filesystem::path& path : segment_files) {
            const std::string name = path.filename().string();
            const size_t dot = name.find('.');
            const std::string hex = name.substr(0, dot == std::string::npos ? name.size() : dot);
            uint64_t seg_id = 0;
            bool valid = hex.size() == 16 && hex.find_first_not_of("0123456789abcdefABCDEF") == std::string::npos;
            if (valid) {
                try { seg_id = std::stoull(hex, nullptr, 16); } catch (...) { valid = false; }
            }
            if (!valid) { ++stale_files_kept_; continue; }
            uint32_t stages = 0;
            const size_t stage_at = name.find(".stage");
            if (stage_at != std::string::npos) {
                const std::string tail = name.substr(stage_at + 6, name.find('.', stage_at + 6) - (stage_at + 6));
                if (!tail.empty() && tail.find_first_not_of("0123456789") == std::string::npos) {
                    try { stages = (uint32_t) std::stoul(tail); } catch (...) { stages = 0; }
                }
            }
            std::error_code sec;
            const uint64_t bytes = std::filesystem::file_size(path, sec);
            std::filesystem::file_time_type stamp = std::filesystem::last_write_time(path, sec);
            if (sec) stamp = std::filesystem::file_time_type::clock::now();
            // A stage file contributes to its segment's stage count when the segment's own files are indexed below;
            // index each file under its seg_id, keeping the largest stage seen.
            auto found = segment_files_.find(seg_id);
            if (found == segment_files_.end()) {
                index_segment_file(seg_id, stages, sec ? 0 : bytes, SegmentKind::Other, stamp);
            } else {
                if (stages > found->second.stages) found->second.stages = stages;
                if (!sec) { found->second.bytes += bytes; segment_bytes_ += bytes; }
            }
        }
        std::sort(manifest_files.begin(), manifest_files.end());
        for (const std::filesystem::path& path : manifest_files) {
            if (path.filename().string().rfind(kSegmentFilePrefix, 0) != 0) { ++stale_files_kept_; continue; }
            SegmentManifest manifest;
            bool other_identity = false;
            std::string parse_error;
            if (!read_manifest(path, manifest, other_identity, parse_error)) {
                if (other_identity) ++foreign_files_kept_;
                else ++stale_files_kept_;
                continue;   // ignored, never removed
            }
            // A manifest whose chain references a segment file that is not there is REPORTED stale (a hole: best()
            // and load() will degrade to the longest valid run) but is still indexed - the missing file is never
            // re-created here and nothing is deleted.
            bool complete = true;
            for (const SegmentRef& ref : manifest.segments) {
                for (size_t stage = 0; stage <= ref.stages; ++stage) {
                    std::error_code sec;
                    if (!std::filesystem::exists(segment_path(directory_.string(), ref.seg_id, stage), sec) || sec) {
                        complete = false;
                        break;
                    }
                }
                if (!complete) break;
            }
            if (!complete) ++stale_files_kept_;
            ManifestEntry entry;
            entry.path = path;
            entry.manifest = std::move(manifest);
            std::error_code sec;
            entry.stamp = std::filesystem::last_write_time(path, sec);
            if (sec) entry.stamp = std::filesystem::file_time_type::clock::now();
            const std::string name = path.filename().string();
            const std::string number = name.substr(std::strlen(kSegmentFilePrefix),
                    name.size() - std::strlen(kSegmentFilePrefix) - std::strlen(kSegmentManifestExt));
            try {
                const uint64_t parsed = std::stoull(number);
                if (parsed > serial_) serial_ = parsed;
            } catch (...) {}
            manifests_.push_back(std::move(entry));
        }
        for (const ManifestEntry& entry : manifests_) bump_refcounts(entry.manifest);
        for (const auto& kv : segment_files_)
            if (segment_refs_.find(kv.first) == segment_refs_.end()) ++orphan_files_kept_;
        enabled_ = true;
        enforce_budget_locked();
        return true;
    } catch (const std::bad_alloc&) {
        manifests_.clear(); segment_refs_.clear(); segment_files_.clear(); segment_bytes_ = 0; enabled_ = false;
        error = "not enough RAM to index segment directory"; return false;
    } catch (const std::exception& e) {
        manifests_.clear(); segment_refs_.clear(); segment_files_.clear(); segment_bytes_ = 0; enabled_ = false;
        error = std::string("cannot index segment directory: ") + e.what(); return false;
    }
}

size_t SegmentStore::refcount(uint64_t seg_id) const {
    std::lock_guard<std::mutex> lock(mu_);
    auto it = segment_refs_.find(seg_id);
    return it == segment_refs_.end() ? 0 : it->second;
}

// Erasing a conversation removes its manifest and DECREMENTS the refcount of every segment it lists: a segment the
// conversation shared with another (the systemprompt) is kept; only a segment whose refcount reaches 0 loses its
// files. If the manifest itself cannot be removed, nothing is dropped (the entry is left, as a failed eviction is).
bool SegmentStore::erase(const std::string& manifest_path, std::string& error) {
    std::lock_guard<std::mutex> lock(mu_);
    auto found = std::find_if(manifests_.begin(), manifests_.end(),
                              [&](const ManifestEntry& entry) { return entry.path.string() == manifest_path; });
    if (found == manifests_.end()) return true;
    std::error_code ec;
    const bool removed = std::filesystem::remove(found->path, ec);
    ec.clear();
    if (!removed && std::filesystem::exists(found->path)) {
        error = "cannot remove segment manifest";
        return false;
    }
    drop_refcounts(found->manifest);
    manifests_.erase(found);
    return true;
}

// The longest contiguous run of segments from the head that is an exact prefix of the prompt. No K/V is read: the
// prompt's own tokens are hashed at each ref's range and compared to the stored address, so a divergence (the effort
// case) or a missing file (a GC'd or damaged segment) stops the run at that point. Later segments are never tried
// once the chain breaks: attention is causal, so a segment after a hole can never be a valid reuse.
SegmentMatch SegmentStore::best_impl(const std::vector<int32_t>& prompt, bool cvec, double similarity,
                                     int64_t n_min) const {
    std::lock_guard<std::mutex> lock(mu_);
    SegmentMatch best;
    if (!enabled_ || prompt.empty() || similarity < 0.0 || similarity >= 1.0) return best;
    const std::string base = directory_.string();
    for (const ManifestEntry& entry : manifests_) {
        const SegmentManifest& manifest = entry.manifest;
        if (manifest.cvec != cvec) continue;
        uint64_t frontier = manifest.head_frontier, cursor = 0;
        size_t valid_index = 0;
        int64_t valid_tokens = 0;
        uint64_t valid_seg = 0;
        uint64_t valid_bytes = 0;
        bool have = false, live = false;
        for (size_t k = 0; k < manifest.segments.size(); ++k) {
            const SegmentRef& ref = manifest.segments[k];
            if (ref.token_begin != cursor) break;                          // a hole or overlap: the chain is broken
            if ((uint64_t) prompt.size() < ref.token_end) break;           // the prompt does not reach this segment
            const std::vector<int32_t> slice(prompt.begin() + (std::ptrdiff_t) ref.token_begin,
                                             prompt.begin() + (std::ptrdiff_t) ref.token_end);
            if (segment_hash(ref.kind, ref.domain, slice, ref.role_token, frontier, identity_.model, identity_.config)
                    != ref.seg_id) break;                                  // the prompt diverges here
            bool present = true;
            for (size_t stage = 0; stage <= ref.stages; ++stage) {
                std::error_code ec;
                if (!std::filesystem::exists(segment_path(base, ref.seg_id, stage), ec) || ec) {
                    present = false;
                    break;
                }
            }
            if (!present) break;                                           // a missing segment file breaks the run
            frontier = ref.seg_id;
            cursor = ref.token_end;
            valid_index = k;
            valid_tokens = (int64_t) ref.token_end;
            valid_seg = ref.seg_id;
            valid_bytes += ref.bytes;                     // the files of segments 0..k, for the restore preflight
            have = true;
            live = (k + 1 == manifest.segments.size());
        }
        if (!have || valid_tokens < n_min) continue;
        if ((double) valid_tokens / (double) prompt.size() <= similarity) continue;
        if (valid_tokens > best.tokens) best = {entry.path.string(), valid_index, valid_seg, valid_tokens, live,
                                                valid_bytes};
    }
    return best;
}

// Reads the sum: the segments 0..match.index, and none past it. Each segment is an ordinary session file, so the
// identity, the header, the bounds and the payload hash are checked by session_file_read exactly as for a .sess.
bool SegmentStore::load(const SegmentMatch& match, SavedConversation& image, const std::vector<SessionReadLimits>& limits,
                        std::string& error) const {
    std::lock_guard<std::mutex> lock(mu_);
    if (!enabled_) { error = "segment store is disabled"; return false; }
    const ManifestEntry* entry = find_manifest(match.manifest_path);
    if (entry == nullptr) { error = "unknown segment manifest"; return false; }
    if (entry->manifest.segments.empty() || match.index >= entry->manifest.segments.size()) {
        error = "segment frontier is out of range";
        return false;
    }
    const SessionReadLimits open_limits;
    auto for_stage = [&](size_t i) -> const SessionReadLimits& {
        if (limits.empty()) return open_limits;
        return limits[i < limits.size() ? i : limits.size() - 1];
    };
    std::vector<SavedConversation> parts;
    parts.reserve(match.index + 1);
    for (size_t k = 0; k <= match.index; ++k) {
        const SegmentRef& ref = entry->manifest.segments[k];
        SavedConversation part;
        size_t bytes = 0;
        if (!session_file_read(segment_path(directory_.string(), ref.seg_id, 0), identity_, part, bytes, error,
                               for_stage(0))) return false;
        std::vector<SavedConversation> stage_images;
        stage_images.reserve(ref.stages);
        for (size_t stage = 1; stage <= ref.stages; ++stage) {
            SavedConversation stage_image;
            size_t stage_bytes = 0;
            if (!session_file_read(segment_path(directory_.string(), ref.seg_id, stage), identity_, stage_image,
                                   stage_bytes, error, for_stage(stage))) return false;
            stage_images.push_back(std::move(stage_image));
        }
        part.stage_images = std::move(stage_images);
        parts.push_back(std::move(part));
    }
    image = concat_images(parts);
    return true;
}

namespace {
// The GC order by kind (§6.3 of the design): the volatile tail (think/answer/tool) goes before read/write, and a
// systemprompt last. A shared segment never reaches this - a refcount > 1 is filtered out before the order matters.
int segment_eviction_priority(SegmentKind kind) {
    switch (kind) {
        case SegmentKind::Think: case SegmentKind::Answer:
        case SegmentKind::ToolCall: case SegmentKind::ToolResult: return 0;
        case SegmentKind::Read: case SegmentKind::Write: return 1;
        case SegmentKind::User: case SegmentKind::Other: return 2;
        case SegmentKind::Sys: return 3;
    }
    return 2;
}
} // namespace

// The segment tier's OWN GC. It is separate from the .sess tier's: it spends segment_bytes_ against budget_ and
// counts segment_evictions_, and it never touches the spill tier's bytes_ or disk_evictions_. Only a segment with a
// refcount <= 1 can leave (a shared one is never removed); among those the most expendable kind goes first and, for
// one kind, the oldest. An orphan (refcount 0) goes before any owned segment. When only shared segments remain the
// loop stops: the budget cannot be met without touching data another conversation still needs.
void SegmentStore::enforce_budget_locked() {
    if (budget_ == 0) return;
    while (segment_bytes_ > budget_ && !segment_files_.empty()) {
        auto victim = segment_files_.end();
        for (auto it = segment_files_.begin(); it != segment_files_.end(); ++it) {
            auto ref = segment_refs_.find(it->first);
            const size_t count = ref == segment_refs_.end() ? 0 : ref->second;
            if (count > 1) continue;
            if (victim == segment_files_.end()) { victim = it; continue; }
            auto victim_ref = segment_refs_.find(victim->first);
            const size_t victim_count = victim_ref == segment_refs_.end() ? 0 : victim_ref->second;
            const int priority = segment_eviction_priority(it->second.kind);
            const int victim_priority = segment_eviction_priority(victim->second.kind);
            const bool better = priority < victim_priority ||
                                (priority == victim_priority && count < victim_count) ||
                                (priority == victim_priority && count == victim_count &&
                                 it->second.stamp < victim->second.stamp);
            if (better) victim = it;
        }
        if (victim == segment_files_.end()) break;
        remove_segment(victim->first);
        ++segment_evictions_;
    }
}

void SegmentStore::enforce_budget() {
    std::lock_guard<std::mutex> lock(mu_);
    enforce_budget_locked();
}

} // namespace strata::core
