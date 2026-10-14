// The per-segment disk tier that replaces "one growing .sess per conversation" with independent, content-addressed
// segment files: each phase of a turn (systemprompt, a file read, a file write/edit, a tool call and its result, the
// thinking block, the answer, the user turn) is its OWN file, named after the content address of the segment. The
// conversation is then the ORDERED LIST of references (the manifest, "<stem>.segs"): the state is the sum of the
// segments, not a file that grows.
//
// Why the name is the content address (segment_hash): a segment's K/V was computed against everything that precedes
// it (attention is causal and the GDN/QSA/PLE state is all-or-nothing), so two segments with the same tokens but a
// DIFFERENT preceding chain are NOT the same segment and must not share a file. The address folds the previous
// segment's address in, so an identical systemprompt (same head) deduplicates to one file shared by every chat that
// opens with it, while a repeated block after a different history does not.
//
// The layer is opt-in and off unless a caller brings it up (D-7): with no segment directory nothing is created and
// no byte is written, and the existing ".sess" + ".meta" v2 tier is the compatibility path (a running store ignores
// it and the older tier reads it unchanged). A segment file is an ordinary session file (conversation_file.hpp), so
// a segment is byte-for-byte a session_file_write of its own slice.
//
// Host code only (no CUDA).
#pragma once

#include "strata/core/conversation_cache.hpp"
#include "strata/core/conversation_file.hpp"
#include "strata/core/conversation_spill.hpp"   // conversation_key / conversation_header_length (inline helpers)

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <map>
#include <mutex>
#include <string>
#include <vector>

namespace strata::core {

// The closed taxonomy of §3 of the design: one segment per kind. Adding a kind means bumping the manifest version;
// an open sub-label goes in `domain` (e.g. which tool), which is why `Other` (15) exists.
enum class SegmentKind : uint8_t {
    Sys = 0,         // systemprompt(s) and the fixed header turns - the star case of the dedup
    Read = 1,        // a file a tool injected into the context
    Write = 2,       // a tool turn that mutates state (write/edit) and its echo
    ToolCall = 3,    // the assistant's tool-call request
    ToolResult = 4,  // what the tool returned
    Think = 5,       // the reasoning block, when the template emits one
    Answer = 6,      // the visible assistant turn
    User = 7,        // the user turn that opens the exchange
    Other = 15,      // the escape hatch for a segment the harness declares without a known kind
};

// Human name, for diagnostics and the test output.
const char* segment_kind_name(SegmentKind kind);

// The manifest's own limits. The segment table has its own cap (the checkpoints keep theirs, kMaxSidecarEntries).
constexpr size_t kMaxSegments = 4096;
constexpr uint32_t kMaxSegmentStages = 64;
constexpr uint32_t kSegmentManifestVersion = 1;
constexpr char kSegmentManifestMagic[8] = {'S', 'S', 'E', 'G', '1', '\0', '\0', '\0'};
constexpr char kSegmentFilePrefix[] = "strata-conv-";   // manifests share the tier's prefix; segments do NOT
constexpr char kSegmentManifestExt[] = ".segs";
constexpr char kSegmentExt[] = ".seg";

// The head of the chain, before any segment: H(model || config). A segment whose predecessor is the head is the
// first one of a chat (the systemprompt), and its address depends only on the systemprompt tokens and the identity.
uint64_t segment_head_frontier(uint64_t model, uint64_t config);

// The content address of a segment: H(prev_frontier || kind || domain || tokens || role_token || model || config),
// built with SessionIdentityBuilder so every field is length-delimited and two lists never collide. Stable across
// runs; changes if the PRECEDING chain changes (even with identical tokens); unaffected by a LATER segment.
uint64_t segment_hash(SegmentKind kind, uint16_t domain, const std::vector<int32_t>& tokens,
                      int64_t role_token, uint64_t prev_frontier, uint64_t model, uint64_t config);

// The file name of a segment and of its k-th stage: "<hex16>.seg" (stage 0) and "<hex16>.stage<k>.seg" (k >= 1).
std::string segment_file_name(uint64_t seg_id, size_t stage);
// The manifest file name of a conversation: "strata-conv-<serial>.segs".
std::string segment_manifest_name(uint64_t serial);

// One reference of the chain: the segment's content address, its typed range in the prompt, and where its files are.
struct SegmentRef {
    uint64_t seg_id = 0;                    // = the frontier after this segment; the file's name
    SegmentKind kind = SegmentKind::Other;
    uint16_t domain = 0;                    // open sub-label (which tool); 0 = none
    uint64_t token_begin = 0;               // start of the range (the previous segment's end)
    uint64_t token_end = 0;                 // end of the range (= the next segment's begin; the last == live_tokens)
    int64_t role_token = -1;                // the role token that opens the segment, or -1
    bool has_state = false;                 // 1 = the segment carries the recurrent-state snapshot at its frontier
    uint32_t stages = 0;                    // extra stage files beyond the first (0 = one file, no --layer-split)
    uint64_t bytes = 0;                     // sum of the bytes of every file of this segment (stage 0 + stages)
    std::vector<uint64_t> stage_bytes;      // stages + 1 entries, stage 0 first (for the GC and the restore preflight)
};

// The manifest is the state: the ordered list of references plus the identity needed to re-derive the chain.
struct SegmentManifest {
    uint64_t model = 0, config = 0;         // SessionFileIdentity, matched against the store's identity_
    bool cvec = true;                       // the steering mode (0/1), as the sidecar records it
    uint64_t conversation_key = 0;          // conversation_key(live.ids, turn_token) - the writer's stable key
    uint64_t head_frontier = 0;             // H(model || config): the chain's head
    int64_t turn_token = -1;                // the --turn-token in force (to re-derive boundaries); -1 = none
    std::vector<SegmentRef> segments;       // the chain, in order
    uint64_t live_tokens = 0;               // the sum of the segments' tokens = the whole prompt length
};

// What the caller hands the store for one segment: its kind, its token slice (the prompt range it covers), the role
// token that opens it, and its own session image (the K/V slice, computed under the chain that precedes it).
struct SegmentInput {
    SegmentKind kind = SegmentKind::Other;
    uint16_t domain = 0;
    std::vector<int32_t> tokens;
    int64_t role_token = -1;
    bool has_state = false;
    SavedConversation image;                    // stage 0: its live.ids must equal `tokens`
    std::vector<SavedConversation> stage_images; // the later stages, with --layer-split
};

// One entry of a write plan: a contiguous token run the caller wants as its OWN segment. Unlike SegmentInput it
// carries NO image - store_image() slices ONE captured conversation image for this range, so a segment file holds
// only its K/V slice and the chain is the sum. `tokens` is a LENGTH (the ranges are contiguous from 0), so the plan
// is position-independent and identical for two prompts that share a prefix (the thinking-effort case).
struct SegmentPlan {
    SegmentKind kind = SegmentKind::Other;
    uint16_t domain = 0;
    uint64_t tokens = 0;                    // tokens this segment covers; the plan's lengths sum to live_tokens
    int64_t role_token = -1;                // the role token that opens it (--turn-token / --tail-role-token), or -1
    bool has_state = false;                 // 1 = this segment carries the recurrent-state snapshot (the last one)
};

// The default plan for a prompt: [sys][pre][effort+answer] split at the FIRST --turn-token (the end of the system
// prompt) and at the LAST --tail-role-token (the effort turn - the frontier of the thinking-effort case). With no
// turn token it is a single legacy `other` segment covering the whole prompt. The KINDS are a heuristic (the harness
// does not declare them); the token RANGES are exact, so the same plan is produced for two prompts that share a
// prefix. A length of 0 never appears (an empty run is dropped).
std::vector<SegmentPlan> segment_plan_split(const std::vector<int32_t>& ids, int64_t turn_token,
                                            int64_t tail_role_token);

// What best() returns: the chosen frontier (a segment index), the tokens of the reusable prefix, and whether it is
// the whole conversation. `tokens` is the prefix length, so read_tokens(prompt) = prompt.size() - tokens.
struct SegmentMatch {
    std::string manifest_path;
    size_t index = 0;                       // the ref index of the frontier
    uint64_t seg_id = 0;                    // the frontier's content address
    int64_t tokens = 0;                     // token_end of the frontier = the reusable prefix length
    bool live = false;                      // the frontier is the last ref (the whole conversation)
    uint64_t bytes = 0;                     // the sum of the files of segments 0..index (the restore preflight)
    explicit operator bool() const { return tokens > 0; }
};

// The segment tier's own settings, separate from the .sess tier's (P-1): its own budget and its own GC.
struct SegmentStoreConfig {
    std::filesystem::path directory;        // where the manifests and the segment files live (empty = inert)
    uint64_t budget_bytes = 0;              // --conversation-cache-segment-budget-mib: own byte budget (0 = none)
    size_t min_tokens = 0;                  // --conversation-cache-segment-min-tokens: materialization threshold
    int64_t turn_token = -1;                // the --turn-token in force
};

// The independent segment store. It shares no counter with ConversationSpillCache: its budget and its evictions are
// its own (segment_bytes_, segment_evictions_), so the existing tier's bytes_ and disk_evictions are untouched.
class SegmentStore {
public:
    // Opens (creating if needed) the segment directory and indexes the manifests and the segment files already in
    // it. The scan NEVER deletes: a corrupt manifest, a foreign identity and a segment file nothing references are
    // counted (stale/foreign/orphan) and left in place - only the GC (enforce_budget) removes. The refcount of every
    // segment is RECOMPUTED here from the union of the manifests' tables (never persisted).
    bool open(const SegmentStoreConfig& config, SessionFileIdentity identity, std::string& error);

    bool enabled() const { std::lock_guard<std::mutex> lk(mu_); return enabled_; }
    size_t size() const { std::lock_guard<std::mutex> lk(mu_); return manifests_.size(); }      // manifests
    size_t segment_count() const { std::lock_guard<std::mutex> lk(mu_); return segment_files_.size(); }
    uint64_t bytes() const { std::lock_guard<std::mutex> lk(mu_); return segment_bytes_; }       // its OWN budget
    size_t segment_evictions() const { std::lock_guard<std::mutex> lk(mu_); return segment_evictions_; }
    size_t stale_files_kept() const { std::lock_guard<std::mutex> lk(mu_); return stale_files_kept_; }
    size_t foreign_files_kept() const { std::lock_guard<std::mutex> lk(mu_); return foreign_files_kept_; }
    size_t orphan_files_kept() const { std::lock_guard<std::mutex> lk(mu_); return orphan_files_kept_; }
    // How many manifests reference this segment (0 = an orphan: no owner, the GC may reclaim it).
    size_t refcount(uint64_t seg_id) const;

    // Writes the conversation's chain: every NEW segment as its own file (an existing <seg_id>.seg is NOT rewritten -
    // that is the dedup) and the manifest LAST (temp + rename). On any failure the segments written this call are
    // removed and the previous manifest (if any) is left untouched. `manifest_path` receives the committed manifest.
    bool store(uint64_t conversation_key, int64_t turn_token, bool cvec, const std::vector<SegmentInput>& inputs,
               std::string& error, std::string* manifest_path = nullptr);

    // Writes a conversation from ONE captured image and a plan: each plan run becomes its own content-addressed file
    // holding ONLY its K/V slice (the attention K/V of the prefix is the concatenation of the slices, in order), and
    // the manifest is written LAST (temp + rename). A segment whose <seg_id>.seg is already on disk is NEITHER
    // sliced NOR rewritten (idempotent), so a growing chat pays only for its new tail and a shared systemprompt is
    // written once. The manifest of this conversation_key is REPLACED in place (one manifest per conversation, and a
    // re-park of an unchanged chain is a no-op file overwrite). Peak extra RAM is ONE segment slice: the image is
    // sliced, written and freed segment by segment. Returns false (and removes only the segments this call created)
    // if any write fails; the previous manifest is untouched until its successor is safe.
    bool store_image(uint64_t conversation_key, int64_t turn_token, bool cvec, const SavedConversation& image,
                     const std::vector<SegmentPlan>& plan, std::string& error, std::string* manifest_path = nullptr);

    // Writes one segment file atomically and returns its content address. Idempotent: if "<seg_id>.seg" already
    // exists it is left as it is (same address => same bytes) and only indexed. `prev_frontier` is the address of the
    // segment before it (segment_head_frontier() for the first).
    bool write_segment(const SegmentInput& input, uint64_t prev_frontier, uint64_t& seg_id_out, std::string& error);

    // The longest contiguous run of segments from the head that is an exact prefix of the prompt, matched from the
    // manifests alone (the K/V is never read). A missing segment file, a range that does not line up or an address
    // that does not recompute breaks the chain at that point: the run stops there (the caller re-prefills the tail).
    template<class Token>
    SegmentMatch best(const std::vector<Token>& prompt, bool cvec, double similarity, int64_t n_min) const {
        std::vector<int32_t> ids;
        ids.reserve(prompt.size());
        for (const Token& token : prompt) ids.push_back((int32_t) token);
        return best_impl(ids, cvec, similarity, n_min);
    }

    // Reads ONLY the segments 0..match.index (the sum) back into `image`: the attention K/V is the concatenation of
    // the segments' slices and the recurrent state is the LAST segment's snapshot (it is not a sum). The segments
    // after the frontier are never opened.
    bool load(const SegmentMatch& match, SavedConversation& image, const std::vector<SessionReadLimits>& limits,
              std::string& error) const;

    // Removes a conversation's manifest and DECREMENTS the refcount of every segment it lists; a segment's files go
    // only when its refcount reaches 0 (a segment another conversation shares is kept).
    bool erase(const std::string& manifest_path, std::string& error);

    // The segment tier's OWN GC: while segment_bytes_ > budget_bytes_ it removes the best candidate, orphans
    // (refcount 0) first, then refcount==1 segments, the volatile kinds (think/answer/tool) before read/write and a
    // shared systemprompt last; a refcount>1 segment is NEVER removed. Counter: segment_evictions_.
    void enforce_budget();

    // The full path of a segment file: "<dir>/<hex16>.seg" (stage 0) or "<dir>/<hex16>.stage<k>.seg".
    static std::string segment_path(const std::string& directory, uint64_t seg_id, size_t stage);

private:
    struct ManifestEntry {
        std::filesystem::path path;
        SegmentManifest manifest;
        std::filesystem::file_time_type stamp{};
    };
    struct SegmentFile {
        uint64_t seg_id = 0;
        uint32_t stages = 0;                      // extra stage files (0 = one file)
        uint64_t bytes = 0;
        SegmentKind kind = SegmentKind::Other;    // the kind of the ref that first indexed it (for the GC priority)
        std::filesystem::file_time_type stamp{};
    };

    const ManifestEntry* find_manifest(const std::string& path) const;
    bool read_manifest(const std::filesystem::path& path, SegmentManifest& out, bool& other_identity,
                       std::string& error) const;
    bool write_manifest(const std::filesystem::path& path, const SegmentManifest& manifest, std::string& error) const;
    std::filesystem::path next_manifest_path();
    // Writes the files of one segment (dedup: a file already on disk is not rewritten). On a failure it removes only
    // what this call created and reports it, so a half segment never survives. `created` (optional) receives the
    // seg_id when a brand-new segment was written this call, for the store's rollback.
    bool write_segment_files(const SegmentInput& input, uint64_t seg_id, std::string& error,
                             std::vector<uint64_t>* created);
    // The materialization policy (§6.4): with min_tokens > 0 a non-system segment shorter than the threshold is
    // fused with the one before it (an Other segment), so a short chat does not pay one file per message.
    std::vector<SegmentInput> materialize(const std::vector<SegmentInput>& inputs) const;
    // Index a segment file's bytes/stages/kind/stamp (idempotent: a segment already indexed is not double counted).
    void index_segment_file(uint64_t seg_id, uint32_t stages, uint64_t bytes, SegmentKind kind,
                            std::filesystem::file_time_type stamp);
    void bump_refcounts(const SegmentManifest& manifest);
    void drop_refcounts(const SegmentManifest& manifest);
    void remove_segment(uint64_t seg_id);
    void enforce_budget_locked();   // the GC body, for callers that already hold mu_ (open)
    SegmentMatch best_impl(const std::vector<int32_t>& prompt, bool cvec, double similarity, int64_t n_min) const;
    static bool valid_manifest_chain(const SegmentManifest& manifest);

    bool enabled_ = false;
    SessionFileIdentity identity_{};
    std::filesystem::path directory_;
    uint64_t budget_ = 0;
    size_t min_tokens_ = 0;
    int64_t turn_token_ = -1;
    uint64_t serial_ = 0;
    uint64_t segment_bytes_ = 0;
    size_t segment_evictions_ = 0, stale_files_kept_ = 0, foreign_files_kept_ = 0, orphan_files_kept_ = 0;
    std::vector<ManifestEntry> manifests_;             // oldest first
    std::map<uint64_t, size_t> segment_refs_;          // seg_id -> how many manifests list it
    std::map<uint64_t, SegmentFile> segment_files_;    // seg_id -> its files on disk
    mutable std::mutex mu_;                            // the park writer drains the same directory
};

} // namespace strata::core
