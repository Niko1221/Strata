// On-disk persistence of a conversation's state (SavedConversation): one file holding the running state, the
// checkpoints the caller passes (the engine passes only the deepest one), every QSA layer's authoritative K/V and
// the draft layer's K/V.  Pure host code: no CUDA.  Formats v1/v2 (state-first) and v3 (K/V-first), and the
// fail-closed read order, are in docs/DETAILS.md (Session files).
//
// Format v1 is little-endian, with fixed-width integers and IEEE-754 floats; a big-endian build is refused at
// compile time.  The hashes detect accidental corruption; they do not authenticate a file: restore trusted files only.
#pragma once

#include "strata/core/conversation_cache.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace strata::core {

// What the file is bound to: the model files it was computed with, and the engine settings that change what the
// saved bytes mean.  A read with any other identity is refused before the payload is parsed.
struct SessionFileIdentity {
    uint64_t model = 0, config = 0;
};

// Non-cryptographic 64-bit hash with xxHash64-style rounds (NOT the standard XXH64 stream): corruption and identity
// checks, not adversarial input.  Streaming and one-shot agree.
class SessionHasher {
public:
    explicit SessionHasher(uint64_t seed = 0);
    void update(const void* data, size_t n);
    uint64_t digest() const;
private:
    void block(const uint8_t* p);
    uint64_t lane_[4];
    uint8_t pending_[32];
    size_t npending_ = 0;
    uint64_t total_ = 0;
};
uint64_t session_hash64(const void* data, size_t n, uint64_t seed);

// Typed, length-delimited fields for an identity hash: each field is (tag, type, length, bytes), so two different
// field lists never give the same byte stream, and a double is hashed by its exact bits (no decimal rounding).
class SessionIdentityBuilder {
public:
    explicit SessionIdentityBuilder(uint64_t seed) : hash_(seed) {}
    void str(const std::string& tag, const std::string& value);
    void i64(const std::string& tag, int64_t value);
    void u64(const std::string& tag, uint64_t value);
    void f64(const std::string& tag, double value);
    void bytes(const std::string& tag, const void* data, size_t n);
    uint64_t digest() const { return hash_.digest(); }
private:
    void field(const std::string& tag, uint8_t type, const void* data, uint64_t n);
    SessionHasher hash_;
};

// The RESOLVED rope configuration (rope_scaling.hpp, after the CLI flags and the model file's keys met).  The cached
// K is post-RoPE, so a different rotation makes the saved K mean something else.
struct SessionRope {
    int64_t type = 0;
    double freq_base = 0, factor = 0, freq_scale = 0, orig_ctx = 0, ext_factor = 0, attn_factor = 0, beta_fast = 0,
           beta_slow = 0;
};

// Every engine setting that changes what the saved state means.  Sampling, seeds, draft tuning, the expert tier and
// the prompt-reading chunk are not here: they change what is computed next, not what the saved cells hold.
struct SessionConfig {
    std::string engine_version;     // a file is bound to one engine version
    std::string backend;            // "cuda" or "hip": no cross-backend restore
    std::string kv;                 // --kv
    int64_t max_context = 0, kv_resident = 0, mtp_window = 0;
    bool kv_rot = false;            // STRATA_KV_ROT
    SessionRope rope;
    // the control vectors as loaded: a digest of the tables the engine uploaded (every file's content x its exact
    // scale, summed), the mode, the layer range and the direction; 0 when none is loaded
    uint64_t cvec = 0;
    // the arithmetic switches that change the computed state (name, value), in a fixed order
    std::vector<std::pair<std::string, int64_t>> switches;
};
uint64_t session_config_fingerprint(const SessionConfig& config);

// One model input the engine loaded and the role it plays ("native shard 1", "expert blk.3.ffn_up", ...).  The role,
// not the path, enters the fingerprint, so a moved model folder still matches.  Every (role, file) pair counts: one
// file in two roles is two entries (its bytes are read once).  `optional`: a missing file is recorded as absent
// instead of failing (a pack without experts.bin).
struct SessionModelFile {
    std::string role, path;
    bool optional = false;
};
// Sampled fingerprint of the model inputs: role, size, first and last MiB of each entry, in the given order (a file
// listed under several roles is read once).  Paths are UTF-8 (wide APIs on Windows; invalid UTF-8 is refused).  An
// edit in the middle of a file that keeps its size is NOT detected: model files must not change while sessions saved
// with them are kept.  `beat` (optional) is called after each distinct file is read.
bool session_model_fingerprint(const std::vector<SessionModelFile>& files, uint64_t& fingerprint, std::string& error,
                               const std::function<void()>& beat = {});

// What the engine loaded, as its loaders resolved it (generate.cpp fills it).  `experts`: (role, file) of the expert
// source - the pack's experts.bin, or every expert tensor read from a GGUF in place, per layer and role, as
// native_experts.txt resolved it (ExpertSource::model_inputs).
struct SessionInputs {
    std::vector<std::string> native_shards, native_dense, native_head;
    std::string embedding, ple, pack, mtp;
    std::vector<std::pair<std::string, std::string>> experts;
};
// The fingerprint's list: every role with its file, none merged (a path in two roles is two entries).
std::vector<SessionModelFile> session_model_inputs(const SessionInputs& inputs);

// Why a session file operation failed, for the caller's answer (the server maps it to an HTTP status):
//   invalid   refused: not a session file, corrupt, another model or configuration, over this engine's limits, a
//             link, nothing to save (400; nothing changed)
//   storage   no room on the disk: the free-space preflight, or ENOSPC / EDQUOT / EFBIG (507)
//   memory    the RAM preflight refused it, or an allocation failed (503)
//   io        any other failure: permissions, read/write/flush/rename errors, a missing folder (500)
enum class SessionError { none, invalid, storage, memory, io };
const char* session_error_name(SessionError e);
struct SessionStatus {
    SessionError error = SessionError::none;
    // write: the new file has replaced `path` (the rename happened).  Together with a failure: the file's bytes are
    // complete and flushed, but the folder flush after the rename failed, so the new name may not survive a power loss
    bool published = false;
    // write (POSIX): this filesystem cannot flush a folder (fsync EINVAL); the save succeeded without that flush
    bool dir_flush_unsupported = false;
    // read/apply: the opened file's header is verified; 1, 2 or 3, its .sess format version (0 = no verified header).
    // Together with a read failure of SessionError::memory this says the refusal is a descriptor-format (v2/v3) file's
    // RAM admission, not a bad file: the caller keeps the file and its cache index entry
    uint32_t file_version = 0;
};

// Progress of a file transfer: (bytes done, bytes in the file).  A write reports once when its temporary file is
// created (0) and after every block it successfully writes to the OS (16 MiB, or less for the last one: a small file reports too);
// a read reports after every block it reads (16 MiB, or what is left; the first one, which holds the header, too).
// Each report follows a completed transfer of at most 16 MiB: no step between two reports moves more.
using SessionProgress = std::function<void(uint64_t done, uint64_t total)>;
// A step that does not stream - one call that blocks until it is done, over far more than 16 MiB or on the OS's
// schedule (flushing the file, the rename and folder flush) - is announced before it starts, with the bytes it
// concerns.  The engine turns this into an explicit, bounded allowance (session_phase_limit_s) for its watchdog and
// for the server; when the allowance runs out the step is stuck.
using SessionPhase = std::function<void(const char* phase, uint64_t bytes)>;
// The longest a non-streaming step over `bytes` may take before both sides call it stuck: 60 s plus one second per
// 4 MiB (a slow disk or network folder at 4 MiB/s), at most an hour.  1.2 GB: 345 s.
int64_t session_phase_limit_s(uint64_t bytes);

// Where a disk-only save's time went, in nanoseconds per stage, so the log can separate the request's blocked
// time from background work.  Only the file/delta writers' own stages are filled here; the caller adds whatever
// it measured itself (state capture, admission, waiting).
struct SessionWriteTimings {
    // The stages nest, they do not sum to the request's blocked time: kv_source_ns is measured inside
    // kv_ns, rht_ns inside state_ns, and the padded tail block's write inside both os_write_ns and
    // finish_ns.  A delta append has no publish stage and leaves publish_ns at 0.
    uint64_t state_ns = 0;      // geometry, live and checkpoint descriptors into the write buffer; includes rht_ns
    uint64_t kv_ns = 0;         // the K/V section (headers, payload framing), including the source reads below
    uint64_t kv_source_ns = 0;  //   the SessionKvSource::read calls inside kv_ns (the state stays frozen for them)
    uint64_t rht_ns = 0;        //   the opt-in RHT state encoding inside state_ns
    uint64_t os_write_ns = 0;   // the block writes handed to the OS; the padded tail block also counts in finish_ns
    uint64_t finish_ns = 0;     // the tail block's write plus truncate/flush/close of the temporary file
    uint64_t publish_ns = 0;    // rename and folder flush (base files only)
};

// Optional knobs of a write.
struct SessionWriteOptions {
    // nullptr (the default): no per-stage accounting; otherwise the writer adds its stages to *timings.
    SessionWriteTimings* timings = nullptr;
    // the free-space PREFLIGHT: refused when the disk has less than the new file plus this many bytes free before the
    // write starts.  Not a reservation: other writers can take the space afterwards.
    uint64_t min_free_bytes = 0;
    bool durable = true;            // flush the file before the rename and (POSIX) the folder after it
    SessionProgress progress;
    // "flush" (the file's flush, `bytes` = the file) and "publish" (rename and folder flush), before each starts
    SessionPhase phase;
    // tests: return an errno value to make a step fail ("write", "file_flush", "rename", "dir_flush"), 0 to go on
    std::function<int(const char* step)> fault;
    // --conversation-cache-disk-state-ref, prefix saves only: a non-live checkpoint whose five running-state arrays
    // are all byte-identical to the live state's is written as on-disk references to it (the descriptor format,
    // format v2, or v3 when layout_v3 is on) instead of a second copy.  No hit: the file stays byte-identical v1.
    // Has no effect on the v1 byte layout.
    bool state_ref = false;
    // --conversation-cache-disk-state-codec-rht: opt-in lossy GDN encoding for ordinary disk-only base and delta
    // writes.  Prefix saves and other session-write paths do not set this.
    bool state_codec_rht = false;
    // the K/V-first file layout (format v3 .sess files, version-3 delta records): geometry/layer range/cvec, then
    // the K/V headers and payload, then the live and checkpoint state descriptors, so a writer that reads its K/V
    // from the running session releases those sources before it serializes any state.  Opt-in for the disk-only
    // background-state path; prefix, slot and all default writes leave it off and keep the v1/v2 layouts.  A v3
    // file's state section always uses the v2 descriptor coding (RAW/REF/RHT); without this flag nothing changes.
    bool layout_v3 = false;
    // --conversation-cache-disk-background-state: write the v3 base/delta with the producer on the calling thread
    // and every OS write on a second thread (two pre-allocated 16 MiB buffers; the producer blocks when both are
    // queued).  on_sources_released fires as soon as the K/V sources are read, while the writer thread is still
    // draining the buffers, truncating the padded tail, flushing and closing.  REQUIRES layout_v3: the K/V-first
    // order is what makes the early release safe, so any other combination is refused as 'invalid'.  The write call
    // itself still returns only when the file is complete and (a base) published; a failure is reported exactly as
    // the synchronous path reports it.  The initial progress report and the publish phase run on the calling thread;
    // later progress reports and the flush phase run on the writer thread. The caller reads `timings` only after
    // the call returns (the internal join orders the threads).
    bool background_state = false;
    // the sources-released notification: called EXACTLY ONCE per write attempt that reached the payload, when no
    // further SessionKvSource::read call will be made - after the last K/V source read of a successful write (for
    // a v3 file this is before the state section is serialized), or at the failure exit after a source read failed,
    // so a caller waiting to unfreeze its session is woken on every path.  Never called for a refusal before any
    // disk access (sizing, free-space preflight).  A write without sources (a captured image) calls it once after
    // the payload.  For v1/v2 files the state section precedes the K/V, so the call lands after the states.
    std::function<void()> on_sources_released;
};

// Writes a temporary file beside `path` (a unique hidden name, created exclusively: never an existing file or link;
// mode 0600 on POSIX), flushes it, renames it over `path` and, on POSIX, flushes the folder (Windows has no folder
// flush: the rename is MoveFileExW with MOVEFILE_WRITE_THROUGH).  A failure BEFORE the rename keeps an existing file
// at `path` and removes only this call's own temporary file.  A failure of the folder flush AFTER the rename returns
// false with `status->published`: the new file is at `path` (the old one is gone), its directory entry may not
// survive a power loss.  A filesystem that cannot flush a folder (EINVAL) is not a failure; `status` says so.
bool session_file_write(const std::string& path, const SavedConversation& image, const SessionFileIdentity& id,
                        size_t& bytes, std::string& error, const SessionWriteOptions& options = {},
                        SessionStatus* status = nullptr);

// One K/V layer read straight from where it lives (device or pinned host pool) into the file's staging buffer,
// without a full host copy first.  `read(part, offset, dst, n)` copies bytes [offset, offset+n) of part
// 0..4 = k, v, k_scale, v_scale, pooled; after a failed read `error()` (optional) says why.  The file bytes are
// identical to writing a captured ConversationKv.
struct SessionKvSource {
    int format = 0;
    int64_t cells = 0, heads = 0, head_dim = 0, page_size = 0, pooled_rows = 0, idx_dim = 0;
    std::array<size_t, 5> sizes{};
    // --conversation-cache-disk-delta: the bytes of each part BEFORE this record's data.  sizes[part] stays the full
    // part size; session_delta_write stores [first_bytes[part], sizes[part]) only.  Zero (the default, and what the
    // plain session_file_write ignores) means the whole part.
    std::array<size_t, 5> first_bytes{};
    std::function<bool(size_t part, size_t offset, void* dst, size_t n)> read;
    std::function<std::string()> error;
};
// Count the exact size of a complete v1 `.sess` file without reading sources, copying GPU data or performing file I/O.
// Failure reports invalid conversation data or a size that cannot be represented. This is not a general v2 estimate:
// descriptor overhead and REF savings are excluded.
bool session_file_full_bytes(const SavedConversation& image, const std::vector<SessionKvSource>& sources,
                             uint64_t& bytes, std::string& error);

// `meta` carries everything but the K/V (meta.kv must be empty); `kv` supplies the layers in order.
bool session_file_write(const std::string& path, const SavedConversation& meta, const std::vector<SessionKvSource>& kv,
                        const SessionFileIdentity& id, size_t& bytes, std::string& error,
                        const SessionWriteOptions& options = {}, SessionStatus* status = nullptr);

// The checkpoints worth a disk write: the deepest one (the next turn's resume point when the live tail was
// rewritten).  Earlier checkpoints only serve edits further back and cost ~118 MB each at this geometry.
std::vector<ConversationCheckpoint> session_checkpoints_to_save(const std::vector<ConversationCheckpoint>& chain);
// The same choice without a copy (nullptr for an empty chain).
const ConversationCheckpoint* session_deepest_checkpoint(const std::vector<ConversationCheckpoint>& chain);

// What a SAVE copies into RAM besides what the engine already holds, for its RAM preflight.
struct SessionSaveLive {
    uint64_t state_bytes = 0;   // the live running state (gdn + ple + every owned QSA layer's tail, dead, block_pos)
    uint64_t tokens = 0, images = 0;
    uint64_t kv_layers = 0;     // K/V sources (owned QSA layers + the draft)
    uint64_t codec_scratch = 0; // bounded state-codec scratch, 0 when the opt-in codec is off
    // The opt-in disk-only background task moves the deepest checkpoint into its own meta and uses two write buffers.
    // Slot SAVE and the other synchronous paths leave this off: they retain the second checkpoint copy and one buffer.
    bool background_state = false;
};
// The peak, with saturating arithmetic: the default path's two copies of the deepest checkpoint (the list handed to
// the capture and the one inside the captured image), or the background path's single moved copy; the copied live
// state with its token and image lists, the K/V source directory, bounded codec scratch when enabled, one or two
// 16 MiB write buffers, and 1 MiB for the small vectors.
uint64_t session_save_peak_bytes(const ConversationCheckpoint* deepest, const SessionSaveLive& live);
// The SAVE's RAM PREFLIGHT, BEFORE anything is copied: `admit` is asked for session_save_peak_bytes; only when it
// agrees is the deepest checkpoint copied into `out`.  On a refusal nothing was allocated, `out` is empty, `chain`
// and the live state are untouched and no file was opened.  Not a reservation.
bool session_save_checkpoints(const std::vector<ConversationCheckpoint>& chain, const SessionSaveLive& live,
                              const std::function<bool(uint64_t need_bytes, std::string& why)>& admit,
                              std::vector<ConversationCheckpoint>& out, std::string& why);

// What a read checks before it allocates.  Every count is checked against the bytes left in the payload and against
// these limits BEFORE its array is allocated; the defaults accept whatever the payload can hold.  The engine sets all
// of them from its runtime (conversation_session_read_limits), and then the file size is bounded too.
struct SessionReadLimits {
    uint64_t max_file_bytes = UINT64_MAX;
    uint64_t max_tokens = UINT64_MAX;       // token and image counts of the live state and of each checkpoint
    uint64_t max_checkpoints = UINT64_MAX;
    uint64_t max_kv_layers = UINT64_MAX;
    // the runtime's geometry and layer range: a file with others is refused before its state arrays are read
    std::optional<std::array<int64_t, 18>> geometry;
    std::optional<std::pair<int64_t, int64_t>> layer_range;
    // the most bytes of each running-state array (gdn, ple, tails, dead, block_pos), live and per checkpoint
    std::array<uint64_t, 5> max_state_bytes{UINT64_MAX, UINT64_MAX, UINT64_MAX, UINT64_MAX, UINT64_MAX};
    // per K/V layer (the QSA layers, then the draft): the most bytes of each part (k, v, k_scale, v_scale, pooled).
    // A layer past the end of the list is bounded by the payload only.
    std::vector<std::array<uint64_t, 5>> max_kv_bytes;
    // Opt-in ACCEPTED CONFIG (0 = off): a base file or delta record saved under another engine CONFIGURATION is
    // read when its stored config fingerprint equals this value; the model fingerprint must still match.  Only the
    // disk cache's recompute gate sets it (conversation_spill.hpp), and only for an entry that gate approved with a
    // passing recompute record; every other reader leaves it 0, so today's refusals are unchanged.
    uint64_t accept_config = 0;
    // The RAM ADMISSION callback: session-file readers screen their baseline, then account decoded state descriptors and
    // check allocations; delta readers ask before each count-based allocation. False refuses the read. Not a reservation.
    std::function<bool(uint64_t need_bytes, std::string& why)> admit;
    SessionProgress progress;
};
// The largest file these limits admit (UINT64_MAX when one of them is open); session_file_read applies it.
uint64_t session_read_max_file_bytes(const SessionReadLimits& limits);
// Estimated RAM for a `file_bytes` file: the file-backed image, read buffer and overhead, plus optional
// caller-supplied `decoded_state_extra` (RAW=0, REF=logical, RHT=logical-stored).
uint64_t session_read_peak_bytes(uint64_t file_bytes, const SessionReadLimits& limits, uint32_t file_version = 1,
                                 uint64_t decoded_state_extra = 0);
// Full validation (a regular file with one link, size, header, identity, bounded parse, payload hash) before `image`
// is replaced.  The file is opened without following a symbolic link (or a reparse point) at its name, and the checks
// are made on the opened handle.  On failure `image` is unchanged and `status->error` says what kind of failure.
bool session_file_read(const std::string& path, const SessionFileIdentity& id, SavedConversation& image,
                       size_t& bytes, std::string& error, const SessionReadLimits& limits = {},
                       SessionStatus* status = nullptr);

// ---- Streaming reads: a session whose K/V does not fit in RAM. Two passes over one file: format v1 unchanged,
// and the descriptor formats (v2 states-first, v3 K/V-first) served by the same two passes.
//
// `session_file_read_streamed` reads and validates the whole file once, without materializing the K/V: `meta` gets
// the running state and the K/V headers (its K/V buffers stay empty), the whole payload hash is checked, and when
// `kv_part_bytes` is given it receives each K/V layer's five part sizes exactly as this pass parsed and hashed them.
// The caller validates `meta` against its runtime, then calls `session_file_apply_streamed`, which reads the file
// again and hands every K/V block - at most 16 MiB, in order - to `sink`.  The apply pass is bound to the read pass
// by `kv_part_bytes`: it refuses a file whose layer count or any part size differs BEFORE applying that part, so
// only the bytes the read pass admitted can reach the caller's state.  It also recomputes the payload hash and
// compares it to the file's trailer.  The first applied block is the gate: everything before it can still be
// refused cleanly, a divergence found after it (a same-size edit, or a torn re-read) ends the caller - the caller
// must not go on.  An atomic replacement that is itself self-consistent and keeps the same sizes is NOT detected:
// the file must have no writer but this engine. For any descriptor-format file (v2 or v3) the caller must pass
// finite bounds with `bounds->admit` non-null; the legacy no-bounds form remains for compatibility but does not
// provide descriptor admission protection.  A v3 file's state section follows its K/V and is consumed HASH-ONLY in
// the apply walk - no second decode: its integrity is the read pass's full decode plus the same trailing payload
// hash the K/V gets, so a v3 state corruption surfaces no later than the trailer check, after any K/V reached the
// sink, and follows the identical fail-stop path.  Peak RAM: two 16 MiB buffers and `meta`.
//
// `layer` counts as the save wrote them (main layers, then the draft); `part` 0..4 is k, v, k_scale, v_scale,
// pooled; `offset` is within the part.
using SessionKvBlock = std::function<bool(size_t layer, size_t part, uint64_t offset, const void* data, size_t n)>;
bool session_file_read_streamed(const std::string& path, const SessionFileIdentity& id, SavedConversation& meta,
                                size_t& bytes, std::string& error, const SessionReadLimits& limits = {},
                                std::vector<std::array<uint64_t, 5>>* kv_part_bytes = nullptr,
                                SessionStatus* status = nullptr);
bool session_file_apply_streamed(const std::string& path, const SessionFileIdentity& id, const SessionKvBlock& sink,
                                 const std::vector<std::array<uint64_t, 5>>& kv_part_bytes, std::string& error,
                                 SessionStatus* status = nullptr, const SessionReadLimits* bounds = nullptr);
// Finite read limits exactly bounding what re-parsing a file whose validated metadata is `meta` may allocate (the
// counts and logical state-array sizes of `meta` itself, as the delta apply pass derives its bounds from the expected
// record). An apply pass of a descriptor-format (v2/v3) file uses these instead of the unlimited defaults, so its
// parse can never allocate beyond what the read pass admitted; `out.admit` is left to the caller.
void session_read_bounds_from_meta(const SavedConversation& meta, SessionReadLimits& out);
// The ADDITIONAL RAM peak of the streaming apply pass while the read pass's metadata is still resident: the apply
// pass's freshly parsed metadata (bounded by `bounds`), one 16 MiB K/V block and the small I/O state.  The resident
// read-pass metadata is not counted again; streamed K/V payload bytes are excluded.
uint64_t session_stream_apply_peak_bytes(const SessionReadLimits& bounds);
// Estimated RAM for a streaming read of a `file_bytes` file: the two block buffers, small vectors and file-backed
// metadata, plus optional descriptor expansion not already covered by bounded logical-state metadata; K/V stays on disk.
uint64_t session_read_stream_peak_bytes(uint64_t file_bytes, const SessionReadLimits& limits, uint32_t file_version = 1,
                                        uint64_t decoded_state_extra = 0);
// One decoded metadata state bounded by the accepted token, checkpoint, state-array, and K/V-layer counts. Includes
// the actual SavedConversation/ConversationCheckpoint/ConversationKv objects and their vector storage; streamed K/V
// payload bytes are excluded.
uint64_t session_read_meta_peak_bytes(const SessionReadLimits& limits);
// Additional peak for the delta read/apply walk after the base/latest metadata is already resident: one retained
// expected record (including per-layer offset/size arrays), the apply parser's metadata staging, and bounded I/O
// scratch. Checkpoint-tail bookkeeping is admitted separately by the caller. Checked/saturating and independent of
// record count and streamed K/V payload length.
uint64_t session_delta_walk_peak_bytes(const SessionReadLimits& limits);

// ---- Session digest: what a base `.sess` file holds, independent of the on-disk format.  One streaming pass over
// the file: the K/V is hashed block by block (never materialized) and the running state is decoded exactly as a
// restore decodes it, so a v1 file, a v2 REF file and a K/V-first v3 file of the same conversation give the same
// three digests.  The lossy RHT state codec (opt-in) decodes to different floats: its file still digests equal in
// `tokens` and `kv`, and `state_exact` is false there so callers compare those two only.
struct SessionFileDigest {
    uint32_t file_version = 0; // the verified header's format version (1, 2 or 3)
    uint64_t tokens = 0;       // the live checkpoint's and every checkpoint's token ids and image keys, in order
    uint64_t kv = 0;           // every K/V layer's header and its five parts, in layer/part order
    uint64_t state = 0;        // the decoded running state: gdn, ple, tails, dead, block_pos, used; live then checkpoints
    bool state_exact = true;   // false when an array used RHT: `state` is a lossy decode, do not compare it
    bool state_ref = false;    // the file stored REF state descriptors (decoded content is exact)
    bool state_rht = false;    // the file stored RHT-coded GDN
};
// Reads and validates one BASE session file and fills `out`; a delta record stream (a `.dlog`) is refused with a
// message naming it - pass the base file the chain belongs to.  `id` is the model/config identity the file must
// carry; no runtime geometry is needed.  On failure `out` is unchanged and `error` says why.
bool session_file_digest(const std::string& path, const SessionFileIdentity& id, SessionFileDigest& out,
                         std::string& error, SessionStatus* status = nullptr);

// The folder the free-space preflight asks about for `path`, by the rules of Windows or of POSIX (exposed so both can
// be tested anywhere).  Windows: backslashes and a trailing backslash, as GetDiskFreeSpaceExW requires for a UNC
// folder (\\server\share\dir\) and gives a drive root (C:\); empty for a bare file name (the current folder).
std::string session_free_space_dir(const std::string& path, bool windows_rules);

// ---- Delta records (--conversation-cache-disk-delta): append-only saves of one conversation.
//
// A chain is a base session file followed by zero or more delta records in one `.dlog` file beside it.  Every record
// carries the state at one save (running state, the deepest checkpoint, the token/image ids) and only the K/V that
// changed since the previous record: a conversation that grows (or a client that resends the same history) rewrites
// a few tens of MiB instead of the whole 2+ GiB file.  The records are self-delimiting, hashed and bound to the same
// model/config identity as session files; the caller keeps the committed length of the `.dlog` (a crash mid-append
// leaves a torn record past it that no reader reaches, and the next append overwrites it).
//
// A record's layout, little-endian: magic 'STRSDLT\1', version, model, config, first_token, then - in record
// versions 1 and 2 - the same meta prefix as a session file (geometry, layer range, cvec, the live checkpoint, the
// checkpoints), the K/V layer count and per layer its header plus, for each part, `u64 offset`, `u64 size` and the
// bytes [offset, offset+size) - offset is the byte within the full part where the record's data starts - then the
// payload hash and the 'STRSEND\1' end marker.  Record version 3 is the K/V-first layout: geometry, layer range and
// cvec, then the K/V count and the same per-layer directory, then the live and checkpoint state descriptors, then
// the payload hash and the end marker.  Base file format and record versions are independent: a chain may mix a
// v1/v2 base with v2/v3 records and the other way around.
struct SessionDeltaRecord {
    int64_t first_token = 0;                          // the token the record's K/V starts at (informational)
    SavedConversation meta;                           // live + checkpoints + K/V headers; the K/V buffers stay empty
    std::vector<std::array<uint64_t, 5>> kv_offset;   // bytes skipped per layer/part (where the record's data goes)
    std::vector<std::array<uint64_t, 5>> kv_bytes;    // bytes the record holds per layer/part
    // the absolute file offset where the record's payload ends (its trailer starts); the read pass records it and
    // the v3 apply walk consumes the state section hash-only up to it (no second decode)
    uint64_t payload_end = 0;
};

// Appends one record for `meta` + `kv` at `at` (the caller's committed length; the file is created if absent and cut
// to `at` first, dropping an interrupted append's bytes).  kv[i].sizes are the full part sizes, kv[i].first_bytes the
// offsets this record's data starts at.  `bytes` receives the record's length.  `durable` flushes the file before
// returning, `phase`/`progress`/`fault` behave as in SessionWriteOptions.
bool session_delta_write(const std::string& path, uint64_t at, const SavedConversation& meta,
                         const std::vector<SessionKvSource>& kv, int64_t first_token, const SessionFileIdentity& id,
                         uint64_t& bytes, std::string& error, const SessionWriteOptions& options = {},
                         SessionStatus* status = nullptr);

// Reads the record starting at `at` (< `limit`), validating its magic, version, identity, bounds and payload hash;
// `next` is where the following record starts.  The K/V is not materialized: `out` gets the meta and each part's
// (offset, size).  `status->error` is invalid for a torn/corrupt record, io/storage/memory otherwise.
bool session_delta_read(const std::string& path, uint64_t at, uint64_t limit, const SessionFileIdentity& id,
                        const SessionReadLimits& limits, SessionDeltaRecord& out, uint64_t& next, std::string& error,
                        SessionStatus* status = nullptr);

// The apply pass of one read record: re-reads it and hands its K/V blocks to `sink` at (offset + at within the
// part), refusing a record whose structure differs from `expected` BEFORE a block is applied, and re-checking the
// payload hash at the end.  A v3 record's state section is consumed hash-only (no second decode); its integrity is
// the read pass's full decode plus the trailing hash, so a state corruption fails at that check, after the record's
// K/V reached the sink.  As with session_file_apply_streamed, a failure after the first sink call leaves the
// caller's state partly written: the caller must fail-stop.  The optional accept_config is the read pass's
// SessionReadLimits::accept_config for a gate-approved legacy configuration (0 = off); the model must still match.
bool session_delta_apply(const std::string& path, uint64_t at, const SessionFileIdentity& id,
                         const SessionDeltaRecord& expected, const SessionKvBlock& sink, std::string& error,
                         SessionStatus* status = nullptr, uint64_t accept_config = 0);

// The bytes one record takes, for a sizing/preflight pass (saturating: `overflow` says the size does not fit).
// `layout_v3` must be the same value the matching session_delta_write receives, so the count and the written record
// can never disagree about the layout.
uint64_t session_delta_record_bytes(const SavedConversation& meta, const std::vector<SessionKvSource>& kv,
                                    bool* overflow = nullptr, bool state_codec_rht = false, bool layout_v3 = false);

// The per-part byte offset a delta record for `kept_tokens` unchanged tokens starts at (host-testable pure
// arithmetic, the same keep rule conversation_kv_save applies): main-layer parts round down to whole cells through
// the last full page, the pooled part to whole indexer rows; the draft layer (index=false) rewrites every pooled row
// and is handed one token less by the caller (its moving cell).  False when an offset lands past its part.
bool session_delta_first_bytes(SessionKvSource& source, int64_t kept_tokens, bool index, int64_t idx_block,
                               int64_t idx_key_dim, std::string& error);

} // namespace strata::core
