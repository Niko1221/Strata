// L2 2.0 — (v3: L2-1
// Write path: capture+enqueue in FIFO, background writer does SSD append.
// Background thread: append to SSD main file + write mini.
// Block budget
// Same-file serial
// Read path: prefer mini state (L2-1) if watermark matches, fallback to main.
// Entry points:
#ifndef STRATA_CONVERSATION_FILE_L2_INC
#define STRATA_CONVERSATION_FILE_L2_INC
#pragma once

#include <atomic>
#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <deque>
#include <ctime>
#include <filesystem>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

namespace strata::core {
namespace l2two {

// ---- env ) ----
inline int64_t env_i64(const char* k, int64_t def) {
    const char* v = std::getenv(k);
    return (v && *v) ? atoll(v) : def;
}
inline double env_f64(const char* k, double def) {
    const char* v = std::getenv(k);
    return (v && *v) ? atof(v) : def;
}
inline const char* state_dir() { const char* d = std::getenv("STRATA_L2_STATE_DIR"); return (d && *d) ? d : nullptr; }
inline uint64_t sink_gap() { return (uint64_t) env_i64("STRATA_L2_STATE_SINK_TOKENS", 262144); }
inline double sink_min_s() { return env_f64("STRATA_L2_SINK_MIN_S", 60.0); }
inline int64_t writer_budget_mib() { return env_i64("STRATA_L2_WRITER_BUDGET_MIB", 2048); }
inline int64_t writer_join_ms() { return env_i64("STRATA_L2_WRITER_JOIN_MS", 5000); }

// ---- split (state sink_min_s ----
template <typename CkCounts, typename Live>
inline bool hook_split(const Live& live, const CkCounts& old_live) {
    const char* d = state_dir();
    if (!d) return false;
    const uint64_t gap_tokens = sink_gap();
    if (gap_tokens == 0) return false;   // SINK_TOKENS=0: always write the aligned live state
    const uint64_t live_n = (uint64_t) live.ids.size(), old_n = (uint64_t) old_live.ids;
    const uint64_t gap = live_n > old_n ? live_n - old_n : 0;
    const bool split = gap < gap_tokens;
    if (std::getenv("STRATA_L2_DEBUG"))
        std::fprintf(stderr, "strata L2 hook_split: live=%llu old=%llu gap=%llu sink_gap=%llu -> split=%d\n",
                     (unsigned long long) live_n, (unsigned long long) old_n,
                     (unsigned long long) gap, (unsigned long long) sink_gap(), (int) split);
    if (!split) {
        static double last_sink = 0.0;   // Sink
        const double now = [] { return std::chrono::duration<double>(
            std::chrono::steady_clock::now().time_since_epoch()).count(); }();
        if (sink_min_s() <= 0 || now - last_sink >= sink_min_s()) { last_sink = now; return false; }
        return true;
    }
    return true;
}

// ---- 2: live (split ) ----
inline uint64_t hook_field(bool split, uint64_t fresh, uint64_t old) {
    return split ? old : fresh;
}

// [translated from Chinese: ---- 3: counts (split co]
template <typename PutFn, typename Checkpoint, typename CkCounts>
inline void hook_put_counts(PutFn&& put_ck_counts, bool split, const Checkpoint& live,
                            const CkCounts& old_live) {
    if (!split) { put_ck_counts(live); return; }
    Checkpoint lagged = live;
    lagged.ids.resize((size_t) old_live.ids);
    lagged.imgs.resize((size_t) old_live.imgs);
    lagged.gdn.resize((size_t) old_live.gdn);
    lagged.ple.resize((size_t) old_live.ple);
    lagged.tails.resize((size_t) old_live.tails);
    lagged.dead.resize((size_t) old_live.dead);
    lagged.block_pos.resize((size_t) old_live.block_pos);
    put_ck_counts(lagged);
}

// ---- 4: saveinc state (L2-1 ) ----
inline void hook_post_saveinc(const std::string& path, const SavedConversation& meta,
                              const SessionFileIdentity& id, const SessionWriteOptions& opt, bool split) {
    const char* d = state_dir();
    if (!split && d) {
        // L2-2 (state ) —
        const size_t slash0 = path.find_last_of('/');
        std::fprintf(stderr, "strata L2 sink: %s aligned, live=%zu tok (state landed to L2-2)\n",
                     (slash0 == std::string::npos ? path.c_str() : path.substr(slash0 + 1).c_str()),
                     meta.live.ids.size());
        return;
    }
    if (!split || !d) return;
    const size_t slash = path.find_last_of('/');
    const std::string spath = std::string(d) + "/" +
        (slash == std::string::npos ? path : path.substr(slash + 1)) + ".state";
    try {
        SavedConversation mini;
        mini.geometry = meta.geometry; mini.layer_lo = meta.layer_lo; mini.layer_hi = meta.layer_hi;
        mini.live = meta.live;
        std::vector<SessionKvSource> no_kv;
        size_t sbytes = 0; std::string serr;
        SessionWriteOptions sopt;
        sopt.min_free_bytes = opt.min_free_bytes;
        if (!session_file_write(spath, mini, no_kv, id, sbytes, serr, sopt, nullptr))
            std::fprintf(stderr, "strata L2 mini save FAILED: %s (%zu bytes)\n", serr.c_str(), sbytes);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "strata L2 mini save EXC: %s\n", e.what());
    } catch (...) { }
}

// ----  5:  (state  L2-1; ; ) ----
inline void hook_post_read(const std::string& path, const SessionFileIdentity& id,
                           SavedConversation& image) {
    const char* d = state_dir();
    if (!d) return;
    const size_t slash = path.find_last_of('/');
    const std::string mini_path = std::string(d) + "/" +
        (slash == std::string::npos ? path : path.substr(slash + 1)) + ".state";
    std::error_code ec;
    if (!std::filesystem::exists(mini_path, ec)) return;
    uint64_t cover = UINT64_MAX;
    for (const auto& k : image.kv)
        if (k.cells > 0) cover = std::min<uint64_t>(cover, (uint64_t) k.cells);
    if (cover == UINT64_MAX || image.kv.empty()) return;
    SavedConversation mini;
    size_t mb = 0;
    std::string merr;
    SessionReadLimits no_limits;
    if (!session_file_read(mini_path, id, mini, mb, merr, no_limits, nullptr)) return;
    const std::string leaf = slash == std::string::npos ? path : path.substr(slash + 1);
    if (mini.live.ids.empty()) return;
    // A mini is only an overlay for the SAME lineage.  Older, divergent, or
    // checkpoint-incompatible minis are ignored; the main file's embedded live
    // remains a valid degraded restore instead of failing the whole snapshot.
    if (mini.live.ids.size() < image.live.ids.size()) {
        std::fprintf(stderr, "strata L2 assemble: %s mini@%zu < main@%zu (stale mini ignored)\n",
                     leaf.c_str(), mini.live.ids.size(), image.live.ids.size());
        return;
    }
    auto prefix = [](const std::vector<int32_t>& a, const std::vector<int32_t>& b) {
        return a.size() <= b.size() && std::equal(a.begin(), a.end(), b.begin());
    };
    if (!prefix(image.live.ids, mini.live.ids)) {
        std::fprintf(stderr, "strata L2 assemble: %s mini diverges from main live (ignored)\n", leaf.c_str());
        return;
    }
    for (const auto& ck : image.checkpoints) {
        if (!prefix(ck.ids, mini.live.ids)) {
            std::fprintf(stderr, "strata L2 assemble: %s checkpoint is not a live token prefix (mini ignored)\n", leaf.c_str());
            return;
        }
    }
    if ((uint64_t) mini.live.ids.size() <= cover) {
        std::fprintf(stderr, "strata L2 assemble: %s mini@%zu kv@%llu (assembled, saved prefill %lld tok vs lagged)\n",
                     leaf.c_str(), mini.live.ids.size(), (unsigned long long) cover,
                     (long long) mini.live.ids.size());
        image.live = std::move(mini.live);
    } else {
        std::fprintf(stderr, "strata L2 assemble: %s mini@%zu > kv@%llu (lagging fallback, prefill %lld tok)\n",
                     leaf.c_str(), mini.live.ids.size(), (unsigned long long) cover,
                     (long long) (mini.live.ids.size() - cover));
    }
}

// ---- v3 WriterPool: saveinc (FIFO capture+ ) ----
// [translated from Chinese: (meta + KV ), read
struct WriterJob {
    std::string path;
    uint64_t gen = 0;
    SessionFileIdentity id;
    SavedConversation meta;
    std::vector<SessionKvSource> kv;
    std::shared_ptr<std::vector<std::vector<uint8_t>>> kv_bytes;
    std::vector<SessionSaveIncSlot> keep;
    SessionWriteOptions opt;
    bool split = true;
    size_t bytes = 0;                          // Budget accounting
};

// generation-tracking: (pending→committed/failed
struct CommitRecord {
    std::string path;
    uint64_t gen = 0;
    int status = 0;
    int64_t committed_tokens = 0;   // Committed token depth
    std::string err;           // Failure category
    double ts = 0;             // Completion time
};

class WriterPool {
public:
    static WriterPool& instance() { static WriterPool p; return p; }
    ~WriterPool() { stop(); }

    // FIFO true ( ); false =
    // [translated from Chinese: pending → join writer_jo]
    bool submit(WriterJob&& job) {
        const uint64_t budget = (uint64_t)(writer_budget_mib() << 20);
        if (job.bytes > budget) return false;
        std::unique_lock<std::mutex> lk(mu_);
        const auto deadline = std::chrono::steady_clock::now() +
                              std::chrono::milliseconds(writer_join_ms());
        for (;;) {
            const bool has_pending = has_path_locked(job.path);
            if (!has_pending && queued_bytes_ + job.bytes <= budget) break;
            if (cv_done_.wait_until(lk, deadline) == std::cv_status::timeout) {
                has_path_locked(job.path);
                if (has_path_locked(job.path) || queued_bytes_ + job.bytes > budget)
                    return false;
                break;
            }
        }
        queued_bytes_ += job.bytes;
        record_locked(job.path, job.gen, 0, 0, "");
        queue_.push_back(std::move(job));
        cv_work_.notify_one();
        return true;
    }

    void start() {
        std::lock_guard<std::mutex> lk(mu_);
        if (thr_) return;
        stop_ = false;
        thr_ = new std::thread([this] { loop(); });   // ( detached on stop )
    }
    void stop() {
        { std::lock_guard<std::mutex> lk(mu_); stop_ = true; }
        cv_work_.notify_all();
        if (thr_) { if (thr_->joinable()) thr_->join(); delete thr_; thr_ = nullptr; }
    }
    bool enabled() const { return writer_budget_mib() > 0; }
    size_t pending_count() { std::lock_guard<std::mutex> g(mu_); return queue_.size(); }   // Gateway forward-awareness

    // generation-tracking: path+gen ; 0=pending 1=committed 2=failed -1=
    int commit_status(const std::string& path, uint64_t gen, int64_t* tokens, std::string* err) {
        std::lock_guard<std::mutex> g(mu_);
        for (auto it = records_.rbegin(); it != records_.rend(); ++it) {
            if (it->path == path && it->gen == gen) {
                if (tokens) *tokens = it->committed_tokens;
                if (err) *err = it->err;
                return it->status;
            }
        }
        return -1;
    }
    // [translated from Chinese: (serve /slots
    std::vector<CommitRecord> commit_records() {
        std::lock_guard<std::mutex> g(mu_);
        return std::vector<CommitRecord>(records_.begin(), records_.end());
    }

private:
    void record_locked(const std::string& path, uint64_t gen, int status,
                       int64_t tokens, const std::string& err) {
        for (auto& r : records_) {
            if (r.path == path && r.gen == gen) {
                r.status = status; r.committed_tokens = tokens; r.err = err;
                r.ts = (double)std::time(nullptr);
                return;
            }
        }
        CommitRecord r; r.path = path; r.gen = gen; r.status = status;
        r.committed_tokens = tokens; r.err = err; r.ts = (double)std::time(nullptr);
        records_.push_back(std::move(r));
        if (records_.size() > 256) records_.pop_front();
    }
    bool has_path_locked(const std::string& p) const {
        for (const auto& j : queue_) if (j.path == p) return true;
        return false;
    }
    void loop() {
        for (;;) {
            WriterJob job;
            {
                std::unique_lock<std::mutex> lk(mu_);
                cv_work_.wait(lk, [&] { return stop_ || !queue_.empty(); });
                if (stop_ && queue_.empty()) return;
                job = std::move(queue_.front());
                queue_.pop_front();
                queued_bytes_ -= job.bytes;
                // Pending includes executing.
            }
            size_t out = 0;
            std::string err;
            SessionStatus st;
            const auto writer_started = std::chrono::steady_clock::now();
            uint64_t base_bytes = 0, growth_bytes = 0;
            std::error_code size_ec;
            base_bytes = std::filesystem::file_size(job.path, size_ec);
            if (size_ec) base_bytes = 0;
            const uint64_t copy_ms = 0;
            // In-place generation: gateway pending admission serializes
            // writers for this target; restore waits for the same generation.
            // A rare crash-damaged file is handled by the restore/self-heal path.
            const std::string& write_path = job.path;
            const auto saveinc_started = std::chrono::steady_clock::now();
            const bool ok = session_file_saveinc(write_path, job.meta, job.kv, job.id, job.keep,
                                                  out, err, job.opt, &st);
            const uint64_t saveinc_ms = (uint64_t)std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::steady_clock::now() - saveinc_started).count();
            const uint64_t final_bytes = std::filesystem::file_size(write_path, size_ec);
            if (!size_ec && final_bytes > base_bytes) growth_bytes = final_bytes - base_bytes;
            const bool landed = ok && err.empty();
            const uint64_t writer_ms = (uint64_t)std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::steady_clock::now() - writer_started).count();
            {
                std::lock_guard<std::mutex> g(mu_);
                if (landed) {
                    record_locked(job.path, job.gen, 1, (int64_t)job.meta.live.ids.size(), "");
                    std::fprintf(stderr, "strata L2 writer: %s gen=%llu landed %zu bytes (async)\n",
                                 job.path.c_str(), (unsigned long long)job.gen, out);
                } else {
                    record_locked(job.path, job.gen, 2, 0, err);
                    std::fprintf(stderr, "strata L2 writer: %s gen=%llu FAILED: %s\n",
                                 job.path.c_str(), (unsigned long long)job.gen, err.c_str());
                }
            }
            // session_file_saveinc already published exactly one official mini.
            {
                std::lock_guard<std::mutex> g(mu_);
                std::printf("L2W q %zu\n", queue_.size());
                // generation-tracking: (serve per-path+gen
                std::printf("L2C %s %llu %d %lld copy_bytes=0 copy_ms=%llu saveinc_ms=%llu "
                            "writer_ms=%llu growth_bytes=%llu copy_used=0 inplace=1\n", job.path.c_str(),
                            (unsigned long long)job.gen, landed ? 1 : 2,
                            (long long)(landed ? job.meta.live.ids.size() : 0),
                            (unsigned long long)copy_ms, (unsigned long long)saveinc_ms,
                            (unsigned long long)writer_ms, (unsigned long long)growth_bytes);
                std::fflush(stdout);
            }
            cv_done_.notify_all();
        }
    }

    std::mutex mu_;
    std::condition_variable cv_work_, cv_done_;
    std::deque<WriterJob> queue_;
    std::deque<CommitRecord> records_;
    uint64_t queued_bytes_ = 0;
    std::thread* thr_ = nullptr;
    bool stop_ = false;
};

// ---- generate.cpp WriterPool ) ----
inline bool conversation_file_l2_writer_enabled() {
    const bool e = WriterPool::instance().enabled();

                 (int) e, (long long) writer_budget_mib(), state_dir() ? state_dir() : "(null)");
    return e;
}
inline void conversation_file_l2_writer_start() { WriterPool::instance().start(); }
inline WriterPool& conversation_file_l2_writer() { return WriterPool::instance(); }

// Deep copy
// [translated from Chinese: false = (read ) →
template <typename Peek>
inline bool conversation_file_l2_make_job(
    const std::string& path, const SessionFileIdentity& id,
    const std::vector<SessionSaveIncSlot>& keep, const SessionWriteOptions& opt,
    SavedConversation& meta, const std::vector<SessionKvSource>& sources, const Peek& peek,
    SavedConversation& out_meta, std::vector<SessionKvSource>& out_kv,
    std::shared_ptr<std::vector<std::vector<uint8_t>>>& out_bytes, bool& split_hint) {
    (void) path; (void) id; (void) keep; (void) opt;
    // split hook_split (gap vs sink_gap), peek.live_ids old
    const uint64_t old_n = (uint64_t) peek.live_ids;
    const uint64_t live_n = (uint64_t) meta.live.ids.size();
    split_hint = live_n <= old_n || (live_n - old_n) < sink_gap();
    out_bytes = std::make_shared<std::vector<std::vector<uint8_t>>>();
    auto& layers = *out_bytes;
    layers.resize(sources.size());
    size_t total = 0;
    // Layout: per-layer 5-segment incremental concatenation.
    std::vector<std::array<uint64_t, 10>> offs(sources.size());
    for (size_t l = 0; l < sources.size(); ++l) {
        const auto& s = sources[l];
        std::vector<uint8_t> buf;
        // Read start point
        // May be below want.
        for (size_t p = 0; p < 5; ++p) {
            const uint64_t osz = (l < peek.part_bytes.size() && p < peek.part_bytes[l].size())
                                     ? peek.part_bytes[l][p] : 0;
            const uint64_t nsz = s.sizes[p];
            const uint64_t kBlock = 16ull << 20;
            const uint64_t want = osz - (osz % kBlock);
            if (nsz <= want) { offs[l][2*p] = UINT64_MAX; offs[l][2*p+1] = want; continue; }
            const uint64_t d = nsz - want;
            const size_t before = buf.size();
            buf.resize(before + (size_t) d);
            if (!s.read(p, (size_t) want, buf.data() + before, (size_t) d)) return false;
            offs[l][2*p] = before;        // copy_at
            offs[l][2*p+1] = want;
        }
        layers[l] = std::move(buf);
        total += layers[l].size();
    }
    out_meta = std::move(meta);
    out_kv.resize(sources.size());
    for (size_t l = 0; l < sources.size(); ++l) {
        out_kv[l] = sources[l];
        auto bytes = out_bytes;
        auto o = offs;
        out_kv[l].read = [bytes, o, l](size_t part, size_t off, void* dst, size_t n) -> bool {
            const uint64_t copy_at = o[l][2*part], part_osz = o[l][2*part+1];
            if (copy_at == UINT64_MAX) return false;   // No incremental segment
            if ((uint64_t) off < part_osz) return false;
            const uint64_t at = copy_at + ((uint64_t) off - part_osz);
            const auto& lay = (*bytes)[l];
            if (at + n > lay.size()) return false;
            std::memcpy(dst, lay.data() + (size_t) at, n);
            return true;
        };
    }
    return true;
}

// pool.submit generate.cpp
inline bool WriterPool_submit_alias(const std::string& path, const SessionFileIdentity& id,
                                    SavedConversation&& meta, std::vector<SessionKvSource>&& kv,
                                    const std::shared_ptr<std::vector<std::vector<uint8_t>>>& bytes,
                                    const std::vector<SessionSaveIncSlot>& keep,
                                    const SessionWriteOptions& opt, bool split,
                                    uint64_t gen) {
    WriterJob job;
    job.path = path; job.id = id; job.gen = gen;
    job.meta = std::move(meta); job.kv = std::move(kv);
    job.kv_bytes = bytes;
    job.keep = keep; job.opt = opt; job.split = split;
    // Background writer does not belong to any foreground request.
    // Commit status goes through L2W/L2C only.
    job.opt.progress = {};
    job.opt.phase = {};
    job.bytes = 64 << 20;
    for (const auto& l : *bytes) job.bytes += l.size();
    return WriterPool::instance().submit(std::move(job));
}

// generation-tracking: gen ( ; deque <map> inc include
inline uint64_t next_gen(const std::string& path) {
    static std::mutex g_mu;
    static std::deque<std::pair<std::string, uint64_t>> g_seq;
    std::lock_guard<std::mutex> g(g_mu);
    for (auto& e : g_seq) {
        if (e.first == path) return ++e.second;
    }
    g_seq.emplace_back(path, 1);
    return 1;
}

} // namespace l2two
} // namespace strata::core
#endif
