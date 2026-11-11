// [giphoo] L2 2.0 — 两层会话缓存定制模块 (v3: L2-1 写缓冲层完全体)
//   写路径: saveinc FIFO 内只做 capture+入队 (~100ms) → 请求立即解锁;
//           后台 WriterPool 线程 append SSD 主档 + 写 mini(tmpfs) — 流量做平.
//           块预算: 队列字节上限 (超限 FIFO 同步降级 backpressure);
//           同档串行: pending 期间该档新 saveinc join 等 writer 追平 (保 peek 一致).
//   读路径: restore 组装 — state 来源优先 L2-1 迷你档 (水位校验), 失配回退档内旧 state
//   入口: v2.inc hook_split/hook_field/hook_put_counts/hook_post_saveinc; cpp hook_post_read
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

// ---- 配置 (全部 env 参数化, 默认值仅为起点 — 按实测调整) ----
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
inline int64_t writer_budget_mib() { return env_i64("STRATA_L2_WRITER_BUDGET_MIB", 2048); }   // 队列字节预算
inline int64_t writer_join_ms() { return env_i64("STRATA_L2_WRITER_JOIN_MS", 5000); }         // 同档 pending 的 join 上限

// ---- 入口 1: 本轮是否 split (state 滞后)? 全局节流见 sink_min_s ----
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
        static double last_sink = 0.0;   // 沉降节流 (sink_min_s 参数化, 见头部)
        const double now = [] { return std::chrono::duration<double>(
            std::chrono::steady_clock::now().time_since_epoch()).count(); }();
        if (sink_min_s() <= 0 || now - last_sink >= sink_min_s()) { last_sink = now; return false; }
        return true;
    }
    return true;
}

// ---- 入口 2: live 字段写入长度 (split 轮压到旧水位; 沉降轮原值) ----
inline uint64_t hook_field(bool split, uint64_t fresh, uint64_t old) {
    return split ? old : fresh;
}

// ---- 入口 3: counts 写入 (split 轮压旧值 — 段与 counts 必须一致, 否则读侧越界) ----
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

// ---- 入口 4: saveinc 成功后的 state 迷你档 (L2-1 覆盖式快照; 失败不致命) ----
inline void hook_post_saveinc(const std::string& path, const SavedConversation& meta,
                              const SessionFileIdentity& id, const SessionWriteOptions& opt, bool split) {
    const char* d = state_dir();
    if (!split && d) {
        // [giphoo] L2-2 落地登记: 沉降轮 (state 对齐写主档) — 事件/规模/水位审计
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
        sopt.use_v3 = false;   // mini stays one file; gateway hot/archive paths are file-oriented
        sopt.min_free_bytes = opt.min_free_bytes;
        if (!session_file_write(spath, mini, no_kv, id, sbytes, serr, sopt, nullptr))
            std::fprintf(stderr, "strata L2 mini save FAILED: %s (%zu bytes)\n", serr.c_str(), sbytes);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "strata L2 mini save EXC: %s\n", e.what());
    } catch (...) { }
}

// ---- ?? 5: ??????? (state ???? L2-1; ????; ??????) ----
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

inline uint64_t archive_disk_bytes(const std::string& path) {
    std::error_code ec;
    namespace fs = std::filesystem;
    if (!fs::is_directory(path, ec)) {
        const uint64_t n = fs::file_size(path, ec);
        return ec ? 0 : n;
    }
    uint64_t total = 0;
    for (fs::recursive_directory_iterator it(path, fs::directory_options::skip_permission_denied, ec), end;
         !ec && it != end; it.increment(ec)) {
        if (it->is_regular_file(ec) && !it->is_symlink(ec)) total += it->file_size(ec);
    }
    return total;
}

// ---- v3 WriterPool: saveinc 异步写缓冲 (FIFO 只 capture+入队; 落盘后台做平) ----
// 队列项持有深拷贝 (meta + KV 增量段字节), read 回调从拷贝读 — 引擎内存解锁, 请求立即继续.
struct WriterJob {
    std::string path;
    uint64_t gen = 0;                            // [giphoo] issue#4-A: 代际 (每档单调递增)
    SessionFileIdentity id;
    SavedConversation meta;                    // capture 已是 host 拷贝 (move 入队)
    std::vector<SessionKvSource> kv;           // read 回调指向 kv_bytes
    std::shared_ptr<std::vector<std::vector<uint8_t>>> kv_bytes;   // [layer] 增量段深拷贝
    std::vector<SessionSaveIncSlot> keep;
    SessionWriteOptions opt;
    bool split = true;
    size_t bytes = 0;                          // 预算记账
};

// [giphoo] issue#4-A: 代际提交记录 (pending→committed/failed 全程可查询, 不在出队时释放语义)
struct CommitRecord {
    std::string path;
    uint64_t gen = 0;
    int status = 0;            // 0=pending(含执行中) 1=committed 2=failed
    int64_t committed_tokens = 0;   // 已提交 token 深度
    std::string err;           // 失败类别
    double ts = 0;             // 完成时刻 (epoch)
};

class WriterPool {
public:
    static WriterPool& instance() { static WriterPool p; return p; }
    ~WriterPool() { stop(); }   // static 析构兜底 (进程退出不挂: join writer 线程)

    // FIFO 内调用: 入队 true (调用方立即返回成功); false = 同步降级.
    // 同档 pending 或预算满 → join 等追平 (上限 writer_join_ms).
    bool submit(WriterJob&& job) {
        const uint64_t budget = (uint64_t)(writer_budget_mib() << 20);
        if (job.bytes > budget) return false;
        std::unique_lock<std::mutex> lk(mu_);
        const auto deadline = std::chrono::steady_clock::now() +
                              std::chrono::milliseconds(writer_join_ms());
        for (;;) {
            const bool has_pending = has_path_locked(job.path);
            if (!has_pending && queued_bytes_ + job.bytes <= budget) break;   // 可入队
            if (cv_done_.wait_until(lk, deadline) == std::cv_status::timeout) {
                has_path_locked(job.path);   // 最后再验一次 (无副作用)
                if (has_path_locked(job.path) || queued_bytes_ + job.bytes > budget)
                    return false;            // 等不到: 同步降级
                break;
            }
        }
        queued_bytes_ += job.bytes;
        record_locked(job.path, job.gen, 0, 0, "");   // [giphoo] issue#4-A: 入队即登记 pending
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
    bool enabled() const { return writer_budget_mib() > 0; }   // 0 = 关 (全同步)
    size_t pending_count() { std::lock_guard<std::mutex> g(mu_); return queue_.size(); }   // 网关前滚感知

    // [giphoo] issue#4-A: 按 path+gen 查询提交状态; 0=pending 1=committed 2=failed -1=未知代
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
    // 全量快照 (serve /slots 透传)
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
        if (records_.size() > 256) records_.pop_front();   // 上限 256 代 (滚动)
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
                // pending 语义含执行中: 记录保持 pending 直到完成
            }
            size_t out = 0;
            std::string err;
            SessionStatus st;
            const auto writer_started = std::chrono::steady_clock::now();
            uint64_t base_bytes = 0, growth_bytes = 0;
            base_bytes = archive_disk_bytes(job.path);
            const uint64_t copy_ms = 0;
            // [giphoo] In-place generation: gateway pending admission serializes
            // writers for this target; restore waits for the same generation.
            // A rare crash-damaged file is handled by the restore/self-heal path.
            const std::string& write_path = job.path;
            const auto saveinc_started = std::chrono::steady_clock::now();
            const bool ok = session_file_saveinc(write_path, job.meta, job.kv, job.id, job.keep,
                                                  out, err, job.opt, &st);
            const uint64_t saveinc_ms = (uint64_t)std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::steady_clock::now() - saveinc_started).count();
            const uint64_t final_bytes = archive_disk_bytes(write_path);
            if (final_bytes > base_bytes) growth_bytes = final_bytes - base_bytes;
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
            {   // [giphoo] v3: 协议行 — serve 维护深度 (gateway 前滚轮询依据)
                std::lock_guard<std::mutex> g(mu_);
                std::printf("L2W q %zu\n", queue_.size());
                // [giphoo] issue#4-A: 代际提交协议行 (serve 维护 per-path+gen 状态表)
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
    std::deque<CommitRecord> records_;   // [giphoo] issue#4-A: 代际提交记录
    uint64_t queued_bytes_ = 0;
    std::thread* thr_ = nullptr;
    bool stop_ = false;
};

// ---- generate.cpp 的桥接接口 (上游只调这四个函数, 不见 WriterPool 细节) ----
inline bool conversation_file_l2_writer_enabled() {
    const bool e = WriterPool::instance().enabled();
    std::fprintf(stderr, "strata L2 dbg: writer_enabled=%d budget=%lld dir=%s\n",
                 (int) e, (long long) writer_budget_mib(), state_dir() ? state_dir() : "(null)");
    return e;
}
inline void conversation_file_l2_writer_start() { WriterPool::instance().start(); }
inline WriterPool& conversation_file_l2_writer() { return WriterPool::instance(); }

// 深拷贝构造 job: meta move + KV 增量段 [osz,nsz) 字节拷出 (osz 来自 peek 的档内段大小).
// 返回 false = 拷贝失败 (read 回调拒) → 调用方走同步.
template <typename Peek>
inline bool conversation_file_l2_make_job(
    const std::string& path, const SessionFileIdentity& id,
    const std::vector<SessionSaveIncSlot>& keep, const SessionWriteOptions& opt,
    SavedConversation& meta, const std::vector<SessionKvSource>& sources, const Peek& peek,
    SavedConversation& out_meta, std::vector<SessionKvSource>& out_kv,
    std::shared_ptr<std::vector<std::vector<uint8_t>>>& out_bytes, bool& split_hint) {
    (void) id; (void) keep; (void) opt;   // (job 侧自带; 此处不重复记)
    const bool segmented = std::filesystem::is_directory(path);
    // split 判定: 与 hook_split 同语义 (gap vs sink_gap), peek.live_ids 充当 old 水位
    const uint64_t old_n = (uint64_t) peek.live_ids;
    const uint64_t live_n = (uint64_t) meta.live.ids.size();
    split_hint = live_n <= old_n || (live_n - old_n) < sink_gap();
    out_bytes = std::make_shared<std::vector<std::vector<uint8_t>>>();
    auto& layers = *out_bytes;
    layers.resize(sources.size());
    size_t total = 0;
    // 布局: 每层 5 段增量拼接; 记 (拷贝起点, 段原起点) — read 回调收到全段偏移 off, 换算 = copy_at + (off - part_osz)
    std::vector<std::array<uint64_t, 10>> offs(sources.size());   // 5 对 (copy_at, part_osz)
    for (size_t l = 0; l < sources.size(); ++l) {
        const auto& s = sources[l];
        std::vector<uint8_t> buf;
        // 读起点: 档内段大小向下对齐 block (16 MiB) — writer 的 keep 从 block 边界累计,
        // 可能低于 want; 拷贝多含 [blockdown, want) 冗余旧段 (安全, 每段至多多 16 MiB)
        for (size_t p = 0; p < 5; ++p) {
            const uint64_t osz = (l < peek.part_bytes.size() && p < peek.part_bytes[l].size())
                                     ? peek.part_bytes[l][p] : 0;
            const uint64_t nsz = s.sizes[p];
            const uint64_t kBlock = 16ull << 20;
            // V3 deltas start at the exact old byte boundary. A tail consolidation
            // starts at the immutable base watermark; pooled part 4 is always a
            // complete latest-generation replacement, not an append tail.
            uint64_t v3_from = osz;
            if (segmented && peek.consolidate_tail && l < peek.base_part_bytes.size() &&
                p < peek.base_part_bytes[l].size()) v3_from = peek.base_part_bytes[l][p];
            const uint64_t want = segmented ? (p == 4 ? 0 : v3_from) : osz - (osz % kBlock);
            if (nsz <= want) { offs[l][2*p] = UINT64_MAX; offs[l][2*p+1] = want; continue; }
            const uint64_t d = nsz - want;
            const size_t before = buf.size();
            buf.resize(before + (size_t) d);
            if (!s.read(p, (size_t) want, buf.data() + before, (size_t) d)) return false;
            offs[l][2*p] = before;        // copy_at
            offs[l][2*p+1] = want;        // 段原起点 (writer 全段坐标系)
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
            if (copy_at == UINT64_MAX) return false;   // 无增量段
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

// pool.submit 的便捷重载 (对齐 generate.cpp 的调用形态)
inline bool WriterPool_submit_alias(const std::string& path, const SessionFileIdentity& id,
                                    SavedConversation&& meta, std::vector<SessionKvSource>&& kv,
                                    const std::shared_ptr<std::vector<std::vector<uint8_t>>>& bytes,
                                    const std::vector<SessionSaveIncSlot>& keep,
                                    const SessionWriteOptions& opt, bool split,
                                    uint64_t gen) {   // [giphoo] issue#4-A: 代际
    WriterJob job;
    job.path = path; job.id = id; job.gen = gen;
    job.meta = std::move(meta); job.kv = std::move(kv);
    job.kv_bytes = bytes;
    job.keep = keep; job.opt = opt; job.split = split;
    // 后台 writer 不属于任何前台 session/probe 应答：禁止把 SESSION/SWAIT 进度行混入引擎 stdout 协议。
    // 提交状态只走 L2W/L2C；网关通过 /slots 轮询，不依赖后台进度行。
    job.opt.progress = {};
    job.opt.phase = {};
    job.bytes = 64 << 20;   // meta 近似
    for (const auto& l : *bytes) job.bytes += l.size();
    return WriterPool::instance().submit(std::move(job));
}

// [giphoo] issue#4-A: 每档单调递增 gen (进程内静态; deque 线性查 — 档数少, 避免 <map> 在 inc 的 include 顺序问题)
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
