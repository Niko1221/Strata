// src/core/conversation_stage_split_test.cpp - the conversation snapshot core on a layer split: per-stage K/V
// sources, the stage table, read limits built from several session carves, and a full two-carve image written to
// and read back from a format-v2 session file.  Two fake carves share one device: no `GpuStage`, no split engine
// config - the core functions only ever see `SessionState`s, exactly as they will through the serve loop.
#include "strata/core/conversation_snapshot.hpp"
#include "strata/core/conversation_file.hpp"
#include "strata/kernels/kv_q4.hpp"
#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <string>
#include <vector>

using namespace strata::core;
using namespace strata::kernels;

namespace {
int checks = 0;
void check(bool ok, const char* label) {
    ++checks;
    if (!ok) { std::fprintf(stderr, "FAIL: %s\n", label); std::exit(1); }
}
void cuda_check(cudaError_t e) {
    if (e != cudaSuccess) { std::fprintf(stderr, "CUDA: %s\n", cudaGetErrorString(e)); std::exit(1); }
}

struct Fixture {
    ModelGeometry g;
    QsaState state;
    std::vector<void*> device, host;
    std::array<void*,5> sources{};
    std::array<size_t,5> sizes{};

    template<class T> void alloc(T*& p, size_t n, bool pinned = false) {
        if (!n) return;
        void* raw = nullptr;
        if (pinned) {
            cuda_check(cudaHostAlloc(&raw, n, cudaHostAllocMapped));
            host.push_back(raw);
            void* mapped = nullptr;
            cuda_check(cudaHostGetDevicePointer(&mapped, raw, 0));
            p = static_cast<T*>(mapped);
        } else {
            cuda_check(cudaMalloc(&raw, n)); device.push_back(raw); p = static_cast<T*>(raw);
        }
    }
    Fixture(int fmt, int mode) {
        auto& st = state;
        st.kv_mode = mode; st.kv_int8 = fmt==kKvInt8; st.kv_q4 = fmt==kKvQ4; st.kv_hybrid = fmt==3;
        st.n_pages = 24; st.n_slots = mode ? 4 : st.n_pages;
        st.max_cells = st.n_pages * 4; st.idx_pooled_rows = mode==2 ? 2 : st.max_cells/4+2;
        const size_t per = fmt==kKvQ4 ? kv_q4_bytes_per_head((int)g.head_dim) : g.head_dim*((st.kv_int8 || st.kv_hybrid) ? 1:2);
        const size_t rows = st.max_cells*g.n_head_kv, slot_rows = st.n_slots*4*g.n_head_kv;
        sizes = {rows*per, rows*per, fmt==kKvInt8 ? rows*(g.head_dim/64)*2:0,
                 fmt==kKvInt8 ? rows*(g.head_dim/64)*2:0, (size_t)st.idx_pooled_rows*g.idx_key_dim*4};
        if (fmt==3) {
            sizes = {rows*per, rows*kv_q4_bytes_per_head((int)g.head_dim), rows*(g.head_dim/64)*2, 0,
                     (size_t)st.idx_pooled_rows*g.idx_key_dim*4};
            alloc(st.k_q,sizes[0]); alloc(st.v_q4,sizes[1]); alloc(st.k_scale,sizes[2]);
            sources[0]=st.k_q; sources[1]=st.v_q4; sources[2]=st.k_scale;
        } else if (fmt==kKvQ4) {
            alloc(st.k_q4,slot_rows*per); alloc(st.v_q4,slot_rows*per);
            if (mode) {alloc(st.host.k_q4,sizes[0],true); alloc(st.host.v_q4,sizes[1],true);}
            sources[0]=mode?st.host.k_q4:st.k_q4; sources[1]=mode?st.host.v_q4:st.v_q4;
        } else if (fmt==kKvInt8) {
            alloc(st.k_q,slot_rows*per); alloc(st.v_q,slot_rows*per);
            alloc(st.k_scale,slot_rows*(g.head_dim/64)*2); alloc(st.v_scale,slot_rows*(g.head_dim/64)*2);
            if (mode) {
                alloc(st.host.k_q,sizes[0],true); alloc(st.host.v_q,sizes[1],true);
                alloc(st.host.k_scale,sizes[2],true); alloc(st.host.v_scale,sizes[3],true);
            }
            sources[0]=mode?st.host.k_q:st.k_q; sources[1]=mode?st.host.v_q:st.v_q;
            sources[2]=mode?st.host.k_scale:st.k_scale; sources[3]=mode?st.host.v_scale:st.v_scale;
        } else {
            alloc(st.k_pool,slot_rows*per); alloc(st.v_pool,slot_rows*per);
            if (mode) {alloc(st.host.k_pool,sizes[0],true); alloc(st.host.v_pool,sizes[1],true);}
            sources[0]=mode?st.host.k_pool:st.k_pool; sources[1]=mode?st.host.v_pool:st.v_pool;
        }
        alloc(st.idx_pooled,sizes[4]); sources[4]=st.idx_pooled;
        alloc(st.page_table,st.n_pages*4);
        if (mode==1) {
            auto& m=st.map;
            m.page_table=st.page_table; m.n_blocks=st.n_pages; m.n_slots=st.n_slots;
            alloc(m.slot_block,st.n_slots*4); alloc(m.slot_stamp,st.n_slots*4); alloc(m.slot_ref,st.n_slots*4);
            alloc(m.miss_block,st.n_slots*4); alloc(m.miss_slot,st.n_slots*4); alloc(m.ctl,kKvCtlInts*4);
        } else if (mode==2) {
            kv_ring_table(st.page_table,st.n_pages,st.n_slots,nullptr);
        }
    }
    void fill(uint8_t salt) {
        for (size_t i=0;i<sources.size();++i) {
            std::vector<uint8_t> data(sizes[i]);
            for (size_t j=0;j<data.size();++j) data[j]=(uint8_t)(salt+i*31+j*7+j/257);
            if (!data.empty()) cuda_check(cudaMemcpy(sources[i],data.data(),data.size(),cudaMemcpyDefault));
        }
    }
    void fill_after(uint8_t salt, int64_t first_dirty) {
        for (size_t i=0;i<sources.size();++i) {
            const size_t offset = i == 4 ? size_t(first_dirty/4)*g.idx_key_dim*4
                                         : (sizes[i]/size_t(state.max_cells))*size_t((first_dirty/4)*4);
            if (sizes[i] > offset)
                cuda_check(cudaMemset(static_cast<uint8_t*>(sources[i])+offset, salt, sizes[i]-offset));
        }
    }
    ~Fixture() { for (void* p:device) cudaFree(p); for (void* p:host) cudaFreeHost(p); }
};

// The geometry: 8 layers -> QSA layers 3 and 7 (interval 4), six GDN rows.  Stage 0 carves layers 0..3 (QSA
// ordinal 0, GDN rows 0..2), stage 1 carves 4..7 (QSA ordinal 1, GDN rows 3..5).  GDN dimensions are shrunk as in
// conversation_snapshot_test so the carves stay small; the QSA dimensions stay at the compiled values.
constexpr int64_t kTokens = 65;   // ids 1..65, under every carve's 96 cells

struct Split {
    Fixture stage0{kKvInt8, 0}, stage1{kKvInt8, 0}, draft{kKvInt8, 0};
    ModelGeometry& g = stage0.g;
    std::array<QsaState, 2> qsa{};   // global QSA ordinals: stage 0 owns [0], stage 1 owns [1]
    SessionState ss0{}, ss1{};
    ConversationStateSizes z0{}, z1{};
    std::vector<int32_t> ids;
    std::vector<ConversationImageKey> images;
    std::vector<ConversationCheckpoint> checkpoints;
    std::string err;

    Split() {
        g.n_layers = 8;
        g.ssm_state_size = 2; g.ssm_v_heads = 2; g.ssm_conv_channels = 8;
        qsa[0] = stage0.state; qsa[1] = stage1.state;
        ss0.max_cells = 96; ss0.qsa_states = qsa.data(); ss0.qsa_ord0 = 0; ss0.qsa_alloc = 1;
        ss0.gdn_ord0 = 0; ss0.gdn_alloc = 3; ss0.layer_lo = 0; ss0.layer_hi = 4;
        ss1.max_cells = 96; ss1.qsa_states = qsa.data(); ss1.qsa_ord0 = 1; ss1.qsa_alloc = 1;
        ss1.gdn_ord0 = 3; ss1.gdn_alloc = 3; ss1.layer_lo = 4; ss1.layer_hi = 8;
        check(conversation_session_sizes(g, ss0, z0, err), "stage 0 carve sizes");
        check(conversation_session_sizes(g, ss1, z1, err), "stage 1 carve sizes");
        check(z0.gdn && z1.gdn && z0.gdn == z1.gdn, "the two carves hold equal GDN rows");
        check(z0.tail && z0.dead && z0.block_pos, "per-layer indexer sizes");
        stage0.alloc(ss0.gdn_state, z0.gdn); stage0.alloc(ss0.ple_hist, z0.ple);
        stage0.alloc(qsa[0].idx_tail, z0.tail); stage0.alloc(qsa[0].idx_dead, z0.dead);
        stage0.alloc(qsa[0].idx_block_pos, z0.block_pos);
        stage1.alloc(ss1.gdn_state, z1.gdn); stage1.alloc(ss1.ple_hist, z1.ple);
        stage1.alloc(qsa[1].idx_tail, z1.tail); stage1.alloc(qsa[1].idx_dead, z1.dead);
        stage1.alloc(qsa[1].idx_block_pos, z1.block_pos);
        ids.resize(kTokens);
        for (size_t i = 0; i < ids.size(); ++i) ids[i] = (int32_t) i + 1;
    }
    void fill(uint8_t salt) {   // distinct pools and live state per carve, the draft on its own salt
        stage0.fill(salt); stage1.fill((uint8_t) (salt + 1)); draft.fill((uint8_t) (salt + 2));
        for (const auto& [p, n, s] : std::vector<std::tuple<void*, size_t, uint8_t>>{
                 {ss0.gdn_state, z0.gdn, salt}, {ss0.ple_hist, z0.ple, salt}, {qsa[0].idx_tail, z0.tail, salt},
                 {qsa[0].idx_dead, z0.dead, salt}, {qsa[0].idx_block_pos, z0.block_pos, salt},
                 {ss1.gdn_state, z1.gdn, (uint8_t) (salt + 3)}, {ss1.ple_hist, z1.ple, (uint8_t) (salt + 3)},
                 {qsa[1].idx_tail, z1.tail, (uint8_t) (salt + 3)}, {qsa[1].idx_dead, z1.dead, (uint8_t) (salt + 3)},
                 {qsa[1].idx_block_pos, z1.block_pos, (uint8_t) (salt + 3)}})
            cuda_check(cudaMemset(p, s, n));
        // A real indexer maintains its spare pooled row (the one at ids/4) as a copy of idx_dead; the K/V
        // capture records that row and conversation_checkpoint_restore re-establishes it from the checkpoint.
        for (auto& st : qsa)
            cuda_check(cudaMemcpy(st.idx_pooled + (size_t) (kTokens / 4) * (size_t) g.idx_key_dim, st.idx_dead,
                                  (size_t) g.idx_key_dim * sizeof(float), cudaMemcpyDeviceToDevice));
    }
};

bool same_bytes(const std::vector<uint8_t>& a, uint8_t byte) {
    return !a.empty() && std::all_of(a.begin(), a.end(), [byte](uint8_t b) { return b == byte; });
}
bool equal_kv(const ConversationKv& a, const ConversationKv& b) {
    return a.k == b.k && a.v == b.v && a.k_scale == b.k_scale && a.v_scale == b.v_scale && a.pooled == b.pooled;
}
// A source captured from `state` directly, against which a split source must match: identical metadata and
// identical leading bytes of every non-empty part.
bool direct_source(SessionKvSource& out, const QsaState& state, const ModelGeometry& g, int64_t upto,
                   bool index, std::string& err, const SessionKvSource& through) {
    if (!conversation_kv_source(out, state, g, upto, index, err)) return false;
    if (out.format != through.format || out.cells != through.cells || out.heads != through.heads ||
        out.head_dim != through.head_dim || out.page_size != through.page_size ||
        out.pooled_rows != through.pooled_rows || out.idx_dim != through.idx_dim || out.sizes != through.sizes)
        return false;
    std::vector<uint8_t> a(256), b(256);
    for (size_t part = 0; part < 5; ++part) {
        const size_t n = std::min<size_t>(256, through.sizes[part]);
        if (!n) continue;
        std::fill(a.begin(), a.end(), 0); std::fill(b.begin(), b.end(), 0);
        if (!through.read(part, 0, a.data(), n) || !out.read(part, 0, b.data(), n)) return false;
        if (!std::equal(a.begin(), a.begin() + (ptrdiff_t) n, b.begin())) return false;
    }
    return true;
}

void split_battery(const std::vector<bool>& draft_of) {
    Split sp;
    sp.fill(7);
    const std::vector<const SessionState*> sessions{&sp.ss0, &sp.ss1};

    // (a) sources in stage-group order: stage 0's QSA source, then stage 1's, the draft closing its owner's group.
    {
        const ConversationView view{sp.ids, sp.images, sp.checkpoints, true};
        SavedConversation meta;
        std::vector<SessionKvSource> sources;
        check(conversation_snapshot_sources_split(meta, sources, view, sessions, draft_of, sp.draft.state, sp.g, sp.err),
              "split sources captured");
        check(sources.size() == 3, "one source per owned QSA layer plus the draft");
        size_t at = 0;
        for (size_t k = 0; k < sessions.size(); ++k) {
            SessionKvSource want;
            check(direct_source(want, sp.qsa[k], sp.g, kTokens, true, sp.err, sources[at]),
                  "stage source matches its carve's own pool");
            ++at;
            if (draft_of[k]) {
                SessionKvSource d;
                check(direct_source(d, sp.draft.state, sp.g, kTokens, false, sp.err, sources[at]),
                      "draft source sits at its stage's group end");
                ++at;
            }
        }
        // (b) the stage table comes from the carves; the live checkpoint holds stage 0's state and one part per
        // later stage, each captured from that stage's own targets (the per-stage pattern at generate.cpp:9948).
        const SessionStageGeom stage1{4, 8, 1, 3, draft_of[1]};
        check(meta.stage_table == std::vector<SessionStageGeom>{stage1}, "stage table built from the carves");
        check(meta.layer_lo == 0 && meta.layer_hi == 4, "the image's range is the main carve's");
        check(meta.live.ids == sp.ids && meta.live.imgs == sp.images, "live token metadata");
        check(meta.live.gdn.size() == sp.z0.gdn && meta.live.ple.size() == sp.z0.ple &&
              meta.live.tails.size() == sp.z0.tail && meta.live.dead.size() == sp.z0.dead &&
              meta.live.block_pos.size() == sp.z0.block_pos, "live state is stage 0's carve");
        check(meta.live.stage_parts.size() == 1, "one live stage part");
        const auto& part = meta.live.stage_parts[0];
        check(part.ids == sp.ids && part.gdn.size() == sp.z1.gdn && part.ple.size() == sp.z1.ple &&
              part.tails.size() == sp.z1.tail && part.dead.size() == sp.z1.dead &&
              part.block_pos.size() == sp.z1.block_pos, "the part is sized by stage 1's carve");
        ConversationCheckpoint fresh;   // stage 1 captured directly: the same live bytes, byte for byte
        fresh.ids = sp.ids;
        check(conversation_checkpoint_save(fresh, sp.ss1, sp.g, sp.err), "direct stage 1 capture");
        check(part.gdn == fresh.gdn && part.ple == fresh.ple && part.tails == fresh.tails &&
              part.dead == fresh.dead && part.block_pos == fresh.block_pos, "the part holds stage 1's own bytes");
        check(part.gdn != meta.live.gdn, "the two carves really are different state");

        // (d) read limits for the whole split.
        SessionReadLimits limits;
        check(conversation_session_read_limits_split(limits, sessions, sp.g, draft_of, sp.draft.state, 1000, 4, sp.err),
              "split read limits built");
        check(limits.max_kv_layers == 3, "max_kv_layers is the owned layers plus the draft");
        check(limits.max_kv_bytes.size() == 3, "one K/V byte bound per layer, in group order");
        at = 0;
        for (size_t k = 0; k < sessions.size(); ++k) {
            std::array<uint64_t, 5> want{};
            check(conversation_kv_part_sizes(sp.qsa[k], sp.g, 96, true, want, sp.err), "direct part sizes");
            check(limits.max_kv_bytes[at++] == want, "the stage's bound comes from its own pool");
            if (draft_of[k]) {
                check(conversation_kv_part_sizes(sp.draft.state, sp.g, 96, false, want, sp.err), "direct draft sizes");
                check(limits.max_kv_bytes[at++] == want, "the draft's bound is at its group end");
            }
        }
        check(limits.max_state_bytes == (std::array<uint64_t, 5>{sp.z0.gdn, sp.z0.ple, sp.z0.tail, sp.z0.dead,
                                                                 sp.z0.block_pos}),
              "main state bounds from stage 0's carve");
        check(limits.max_stage_state_bytes.size() == 1 &&
              limits.max_stage_state_bytes[0] == (std::array<uint64_t, 5>{sp.z1.gdn, sp.z1.ple, sp.z1.tail,
                                                                          sp.z1.dead, sp.z1.block_pos}),
              "per-stage part bounds from stage 1's carve");
        check(limits.stages.has_value() && *limits.stages == std::vector<SessionStageGeom>{stage1},
              "the limits carry the live stage geometry");
        check(limits.max_stages == 1, "exactly one later stage admitted");
        check(limits.max_tokens == 96, "tokens capped by the main session's cells");
        check(limits.max_checkpoints == 4, "checkpoint cap kept");
        check(limits.geometry == meta.geometry, "geometry bound matches the captured key");
        check(limits.layer_range == std::make_pair<int64_t, int64_t>(0, 4), "layer range is the main carve's");

        // (e) the full two-carve image round-trips through the format-v2 file.
        const auto dir = std::filesystem::temp_directory_path() / "strata-stage-split-test";
        std::error_code ec; std::filesystem::remove_all(dir, ec); std::filesystem::create_directories(dir);
        const SessionFileIdentity id{0x1111222233334444ull, 0x5555666677778888ull};
        const std::string path = (dir / "split.bin").string();
        size_t written = 0;
        check(session_file_write(path, meta, sources, id, written, sp.err), "the split file was written");
        SessionReadLimits read = limits;
        read.max_file_bytes = session_read_max_file_bytes(read);
        check(read.max_file_bytes != UINT64_MAX && written <= read.max_file_bytes, "the closed limits bound the file");
        SavedConversation back; size_t got = 0;
        check(session_file_read(path, id, back, got, sp.err, read), "the split file read back under its own limits");
        check(back.stage_table == meta.stage_table, "the stage table round-trips");
        check(back.layer_lo == 0 && back.layer_hi == 4 && back.geometry == meta.geometry, "ranges round-trip");
        check(back.live.ids == sp.ids && back.live.stage_parts.size() == 1 &&
              back.live.stage_parts[0].ids == sp.ids, "live parts round-trip");
        check(back.live.gdn == meta.live.gdn && back.live.stage_parts[0].gdn == part.gdn &&
              back.live.stage_parts[0].tails == part.tails, "live and part bytes round-trip");
        check(back.kv.size() == 3, "one K/V layer per source");
        std::vector<ConversationKv> want_kv(3);
        check(conversation_kv_save(want_kv[0], sp.qsa[0], sp.g, kTokens, true, sp.err) &&
              conversation_kv_save(want_kv[1], sp.qsa[1], sp.g, kTokens, true, sp.err) &&
              conversation_kv_save(want_kv[2], sp.draft.state, sp.g, kTokens, false, sp.err), "direct K/V captures");
        at = 0;
        for (size_t k = 0; k < sessions.size(); ++k) {
            check(equal_kv(back.kv.at(at), want_kv[0 + k]), "each K/V layer round-trips as its own carve's bytes");
            ++at;
            if (draft_of[k]) { check(equal_kv(back.kv.at(at), want_kv[2]), "the draft layer round-trips at its group end"); ++at; }
        }

        // A single-device runtime refuses this file before reading any state: its engaged (empty) stage table and
        // max_stages differ from the file's, which is exactly what protects momo's single-GPU sessions.
        SessionReadLimits single;
        check(conversation_session_read_limits(single, sp.ss0, sp.g, sp.draft.state, 1000, 4, sp.err),
              "single-carve read limits built");
        check(single.stages.has_value() && single.stages->empty(), "single-device limits engage an empty stage table");
        check(single.max_stages == 0, "a single-device runtime admits no later stages");
        SavedConversation none; size_t n = 0; std::string rerr;
        check(!session_file_read(path, id, none, n, rerr, single), "the split file is refused by a single-carve runtime");
        check(none.stage_table.empty() && none.kv.empty(), "refusal left the caller's image untouched");

        // ---- per-stage validate and restore.
        std::vector<SessionState*> carves{&sp.ss0, &sp.ss1};
        check(conversation_snapshot_validate_split(back, carves, draft_of, &sp.draft.state, sp.g, sp.err),
              "validate_split accepts the image of its own runtime");

        {   // A stage table that differs from the runtime's carves is another split: refused.
            SavedConversation wrong = back; wrong.stage_table[0].qsa_alloc = 2; std::string werr;
            check(!conversation_snapshot_validate_split(wrong, carves, draft_of, &sp.draft.state, sp.g, werr),
                  "a foreign stage table is refused");
        }
        {   // A missing draft group in a stage that owns it: refused.
            SavedConversation bad = back;
            bad.kv.erase(bad.kv.begin() + (draft_of[0] ? 1 : 2));
            std::string berr;
            check(!conversation_snapshot_validate_split(bad, carves, draft_of, &sp.draft.state, sp.g, berr),
                  "a missing draft group is refused");
        }
        {   // live.stage_parts must match the stage table exactly.
            SavedConversation bad = back; std::string berr;
            bad.live.stage_parts.clear();
            check(!conversation_snapshot_validate_split(bad, carves, draft_of, &sp.draft.state, sp.g, berr),
                  "a split image without live stage parts is refused");
            bad = back; bad.live.stage_parts.push_back(bad.live.stage_parts[0]);
            check(!conversation_snapshot_validate_split(bad, carves, draft_of, &sp.draft.state, sp.g, berr),
                  "too many live stage parts are refused");
        }
        {   // The same file against a runtime whose stage 1 carves fewer cells than the conversation: extents past
            // that stage's own caps are typed invalid, before anything transfers.
            Split narrow; narrow.fill(11); narrow.qsa[1].max_cells = 40;   // the file holds 65 cells
            std::vector<SessionState*> tight{&narrow.ss0, &narrow.ss1};
            std::string nerr;
            check(conversation_snapshot_restore_split(back, tight, draft_of, &narrow.draft.state, narrow.g, nerr)
                      == ConversationRestore::invalid,
                  "extents past stage 1's own carve are typed invalid");
        }

        {   // Tampering: one byte flipped inside stage 1's K/V section (post-read, pre-restore) is an invalid
            // image, and the refusal happens before any target array is written.
            Split dst; dst.fill(3);
            std::vector<SessionState*> fresh{&dst.ss0, &dst.ss1};
            const size_t stage1_kv = draft_of[0] ? 2 : 1;   // stage 1's layer at its group position
            ConversationCheckpoint pre0, pre1; pre0.ids = back.live.ids; pre1.ids = back.live.ids;
            check(conversation_checkpoint_save(pre0, dst.ss0, dst.g, sp.err) &&
                  conversation_checkpoint_save(pre1, dst.ss1, dst.g, sp.err), "pristine fresh carve state captured");
            std::vector<ConversationKv> pre_kv(3);
            check(conversation_kv_save(pre_kv[0], dst.qsa[0], dst.g, kTokens, true, sp.err) &&
                  conversation_kv_save(pre_kv[1], dst.qsa[1], dst.g, kTokens, true, sp.err) &&
                  conversation_kv_save(pre_kv[2], dst.draft.state, dst.g, kTokens, false, sp.err),
                  "pristine fresh pools captured");
            SavedConversation tampered = back;
            tampered.kv[stage1_kv].cells ^= (int64_t) 1 << 40;   // a byte flip inside the extents field
            std::string terr;
            check(conversation_snapshot_restore_split(tampered, fresh, draft_of, &dst.draft.state, dst.g, terr)
                      == ConversationRestore::invalid,
                  "the tampered K/V section is typed invalid");
            ConversationCheckpoint post0, post1; post0.ids = back.live.ids; post1.ids = back.live.ids;
            check(conversation_checkpoint_save(post0, dst.ss0, dst.g, sp.err) &&
                  conversation_checkpoint_save(post1, dst.ss1, dst.g, sp.err), "recaptured after the refusal");
            check(post0.gdn == pre0.gdn && post0.tails == pre0.tails && post1.gdn == pre1.gdn &&
                  post1.block_pos == pre1.block_pos, "the refusal wrote no state into either carve");
            std::vector<ConversationKv> post_kv(3);
            check(conversation_kv_save(post_kv[0], dst.qsa[0], dst.g, kTokens, true, sp.err) &&
                  conversation_kv_save(post_kv[1], dst.qsa[1], dst.g, kTokens, true, sp.err) &&
                  conversation_kv_save(post_kv[2], dst.draft.state, dst.g, kTokens, false, sp.err), "recaptured pools");
            check(equal_kv(post_kv[0], pre_kv[0]) && equal_kv(post_kv[1], pre_kv[1]) &&
                  equal_kv(post_kv[2], pre_kv[2]), "the refusal wrote nothing into any pool");

            // The round trip: capture -> write v2 -> read -> validate -> restore into fresh carves -> recaptured
            // state and pools match the file byte for byte, per stage.
            check(conversation_snapshot_restore_split(back, fresh, draft_of, &dst.draft.state, dst.g, sp.err)
                      == ConversationRestore::restored,
                  "restore_split restored the image into the fresh runtime");
            ConversationCheckpoint got0, got1; got0.ids = back.live.ids; got1.ids = back.live.ids;
            check(conversation_checkpoint_save(got0, dst.ss0, dst.g, sp.err) &&
                  conversation_checkpoint_save(got1, dst.ss1, dst.g, sp.err), "recaptured from the fresh carves");
            check(got0.gdn == back.live.gdn && got0.ple == back.live.ple && got0.tails == back.live.tails &&
                  got0.dead == back.live.dead && got0.block_pos == back.live.block_pos,
                  "stage 0's running state round-trips");
            const auto& want_part = back.live.stage_parts[0];
            check(got1.gdn == want_part.gdn && got1.ple == want_part.ple && got1.tails == want_part.tails &&
                  got1.dead == want_part.dead && got1.block_pos == want_part.block_pos,
                  "stage 1's part state round-trips");
            size_t at = 0;
            for (size_t k = 0; k < fresh.size(); ++k) {
                ConversationKv after;
                check(conversation_kv_save(after, dst.qsa[k], dst.g, kTokens, true, sp.err) &&
                      equal_kv(after, back.kv.at(at)), "each stage's K/V round-trips into its own carve");
                ++at;
                if (draft_of[k]) {
                    ConversationKv draft_after;
                    check(conversation_kv_save(draft_after, dst.draft.state, dst.g, kTokens, false, sp.err) &&
                          equal_kv(draft_after, back.kv.at(at)), "the draft layer round-trips at its group end");
                    ++at;
                }
            }
        }
    }

    // (c) chain checkpoints already carrying host stage parts pass through without recapture.
    {
        ConversationCheckpoint chain;
        chain.ids.assign(sp.ids.begin(), sp.ids.begin() + 32);
        chain.gdn.assign(sp.z0.gdn, 0x5A); chain.ple.assign(sp.z0.ple, 0x5A);
        chain.tails.assign(sp.z0.tail, 0x5A); chain.dead.assign(sp.z0.dead, 0x5A);
        chain.block_pos.assign(sp.z0.block_pos, 0x5A);
        ConversationCheckpoint cpart;
        cpart.ids = chain.ids;
        cpart.gdn.assign(sp.z1.gdn, 0xB4); cpart.ple.assign(sp.z1.ple, 0xB4);
        cpart.tails.assign(sp.z1.tail, 0xB4); cpart.dead.assign(sp.z1.dead, 0xB4);
        cpart.block_pos.assign(sp.z1.block_pos, 0xB4);
        chain.stage_parts.push_back(std::move(cpart));
        sp.checkpoints.push_back(chain);

        const ConversationView view{sp.ids, sp.images, sp.checkpoints, true};
        SavedConversation meta;
        std::vector<SessionKvSource> sources;
        check(conversation_snapshot_sources_split(meta, sources, view, sessions, draft_of, sp.draft.state, sp.g, sp.err),
              "split capture accepted a checkpointed chain");
        check(meta.checkpoints.size() == 1 && meta.checkpoints[0].stage_parts.size() == 1,
              "the chain checkpoint kept its part");
        check(same_bytes(meta.checkpoints[0].gdn, 0x5A) && same_bytes(meta.checkpoints[0].stage_parts[0].gdn, 0xB4) &&
              same_bytes(meta.checkpoints[0].stage_parts[0].tails, 0xB4),
              "chain arrays pass through as given, main and part alike");
        check(meta.checkpoints[0].ids == chain.ids && meta.checkpoints[0].stage_parts[0].ids == chain.ids,
              "the part keeps its parent's ids");

        // A part count that contradicts the stage layout is refused before anything is written.
        sp.checkpoints[0].stage_parts.clear();
        SavedConversation bad;
        std::vector<SessionKvSource> bad_sources;
        std::string berr;
        check(!conversation_snapshot_sources_split(bad, bad_sources, view, sessions, draft_of, sp.draft.state, sp.g, berr),
              "a checkpoint without its stage parts is refused");
        check(bad_sources.empty() && bad.checkpoints.empty(), "refusal captured nothing");
        sp.checkpoints[0].stage_parts.push_back(chain.stage_parts[0]);

        // One carve through the split builders: an engaged empty table, no per-stage bounds.
        SessionReadLimits one;
        check(conversation_session_read_limits_split(one, {&sp.ss0}, sp.g, {true}, sp.draft.state, 1000, 4, sp.err),
              "one-carve limits through the split builder");
        check(one.stages.has_value() && one.stages->empty(), "one carve engages an empty stage table");
        check(one.max_stages == 0 && one.max_stage_state_bytes.empty(), "one carve admits no later stages");
        check(one.max_kv_layers == 2, "one carve: its QSA layer plus the draft");
    }
}
} // namespace

int main() {
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || !devices) return 77;
    for (const std::vector<bool>& draft_of : {std::vector<bool>{true, false}, std::vector<bool>{false, true}})
        split_battery(draft_of);
    std::printf("conversation_stage_split_test: %d checks passed\n", checks);
    return 0;
}
