// src/ngram/ple_reader.cpp - see include/strata/ngram/ple_reader.hpp.
#include "strata/ngram/ple_reader.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstring>
#include <deque>
#include <exception>
#include <memory>
#include <mutex>
#include <thread>
#include <unordered_map>

namespace strata::ngram {

using platform::Completion;
using platform::DirectFile;
using platform::now_us;

double ReaderStats::percentile(double q) const {
    if (read_us.empty()) return 0.0;
    std::vector<float> v(read_us);
    const size_t k = std::min(v.size() - 1, (size_t) (q * (double) (v.size() - 1) + 0.5));
    std::nth_element(v.begin(), v.begin() + (ptrdiff_t) k, v.end());
    return v[k];
}

namespace {

constexpr size_t LATENCY_RING = 65536;
constexpr uint32_t WAYS = 8;
constexpr uint32_t EMPTY = 0xFFFFFFFFu;

/// Set-associative row cache: 8 ways per set, round-robin replacement inside a set. Bounded by construction.
struct RowCache {
    uint64_t sets = 0;
    uint32_t rb = ROW_BYTES;        // bytes per row
    std::vector<uint32_t> keys;     // sets * WAYS
    std::vector<uint8_t> data;      // sets * WAYS * rb
    std::vector<uint8_t> next;      // per-set replacement pointer
    uint64_t used = 0;

    void init(uint64_t rows, uint32_t row_bytes = ROW_BYTES) {
        sets = rows / WAYS;
        rb = row_bytes;
        keys.assign(sets * WAYS, EMPTY);
        data.assign(sets * WAYS * rb, 0);
        next.assign(sets, 0);
        used = 0;
    }
    static uint64_t mix(uint32_t r) {
        uint64_t x = r * 0x9E3779B97F4A7C15ull;
        return x ^ (x >> 29);
    }
    const uint8_t* find(uint32_t row) const {
        if (sets == 0) return nullptr;
        const uint64_t s = mix(row) % sets;
        for (uint32_t w = 0; w < WAYS; ++w)
            if (keys[s * WAYS + w] == row) return &data[(s * WAYS + w) * rb];
        return nullptr;
    }
    void insert(uint32_t row, const uint8_t* bytes) {
        if (sets == 0 || find(row) != nullptr) return;
        const uint64_t s = mix(row) % sets;
        uint32_t w = next[s];
        next[s] = (uint8_t) ((w + 1) % WAYS);
        if (keys[s * WAYS + w] == EMPTY) ++used;
        keys[s * WAYS + w] = row;
        std::memcpy(&data[(s * WAYS + w) * rb], bytes, rb);
    }
};

struct Use {
    uint32_t row;
    uint32_t in_page;      // byte offset of the row inside the read buffer
    uint8_t* dst;
};

struct Job {
    uint64_t offset = 0;   // aligned file offset
    uint32_t length = 0;   // PAGE or 2 * PAGE (a row that straddles a page boundary)
    uint32_t ticket = 0;
    std::vector<Use> uses;
    double issued_us = 0;
    bool keepalive = false;// no rows and no ticket: it only keeps the SSD awake (`set_keepalive`)
};

struct TicketState {
    uint32_t pending = 0;  // jobs not yet completed
};

/// One `read_batch` request, shared with the batch readers. Job j reads the page at `off[j]` and serves the rows
/// uses[begin[j] .. begin[j+1]) (row indices, ascending; a row across a page boundary is a use of both pages); jobs
/// are numbered in the order of their first row.
struct BatchJobs {
    const uint32_t* rows = nullptr;
    uint8_t* out = nullptr;
    uint32_t rb = 0;
    uint64_t table_offset = 0;
    const uint64_t* off = nullptr;
    const uint32_t* begin = nullptr;
    const uint32_t* uses = nullptr;
    size_t n_jobs = 0;
    size_t per_group = 1;                         // jobs per completion group
    std::atomic<size_t> next{0};                  // the next job to read
    std::unique_ptr<std::atomic<uint32_t>[]> left;// jobs not yet done, per group
    std::atomic<bool> failed{false};
};

/// A batch reader: its own handle (completions to its own port) and read buffers.
struct BatchReader {
    DirectFile file;
    uint8_t* slab = nullptr;                      // depth slots of a page
    std::vector<uint32_t> slot_job;
    std::vector<double> issued;
    std::vector<float> lat;                       // read latencies of the current request
    uint64_t bytes = 0;
    uint64_t reads = 0;
    std::string error;
    std::thread th;
};

}  // namespace

// THREADING. With `io_thread` (the default) one worker thread owns every DirectFile call: it submits queued jobs
// and reaps completions, so `issue` on the caller's thread only builds jobs and wakes it (~11 us per ReadFile
// no longer lands on the host loop). `mu` guards everything below except `file`, which only the worker touches
// (plus `wake`, which is thread-safe). Without `io_thread` the caller does all of it, as before.
struct PleReader::Impl {
    DirectFile file;
    uint64_t table_offset = 0;
    uint64_t n_rows = 0;
    uint32_t row_bytes = ROW_BYTES;
    uint32_t max_inflight = 0;
    uint8_t* slab = nullptr;              // max_inflight slots of 2 pages
    std::vector<uint32_t> free_slots;
    std::vector<Job> inflight;            // indexed by slot
    std::vector<Completion> delayed;      // completed but held back by fault injection
    std::deque<Job> queue;                // not yet submitted
    std::unordered_map<uint32_t, TicketState> tickets;
    uint32_t next_ticket = 1;
    RowCache cache;
    ReaderStats stats;
    size_t ring_pos = 0;
    double delay_us = 0;
    std::string error;
    // the keep-alive (`set_keepalive`), steady-clock microseconds; `keep_us` 0 = off
    double keep_us = 0;
    double keep_window_us = 0;
    double last_issue_us = 0;             // the last `issue`
    double last_read_us = 0;              // the last read that went out, rows or keep-alive
    uint64_t rng = 0x9E3779B97F4A7C15ull;

    bool threaded = false;
    bool stop = false;
    std::mutex mu;
    std::condition_variable cv_work;      // worker: there is something to submit
    std::condition_variable cv_done;      // collectors: a ticket may have completed
    std::thread worker;

    // the batch readers (`set_batch_readers`): `bmu` guards the fields below it; `bcall` makes one request at a time
    std::string path;
    std::vector<std::unique_ptr<BatchReader>> readers;
    uint32_t batch_depth = 0;
    std::mutex bcall;
    std::mutex bmu;
    std::condition_variable bcv_work;     // readers: a request was posted (`bgen`) or `bstop`
    std::condition_variable bcv_done;     // the caller: a group completed, a read failed, or every reader is idle
    uint64_t bgen = 0;
    unsigned bactive = 0;
    bool bstop = false;
    BatchJobs* bcur = nullptr;
    // the request's scratch, kept between requests
    std::vector<uint32_t> b_missed, b_ent, b_job_of, b_begin, b_cur, b_uses;
    std::vector<uint64_t> b_off, b_map;

    void reader_loop(BatchReader& r, uint64_t seen) {   // `seen`: the request count when the reader was made
        std::unique_lock<std::mutex> lk(bmu);
        for (;;) {
            bcv_work.wait(lk, [&] { return bstop || bgen != seen; });
            if (bstop) return;
            seen = bgen;
            BatchJobs* b = bcur;
            lk.unlock();
            run_reads(r, *b);
            lk.lock();
            if (--bactive == 0) bcv_done.notify_all();
        }
    }

    /// One reader's share of a request: jobs from the shared counter, `batch_depth` in flight, until none is left.
    void run_reads(BatchReader& r, BatchJobs& b) {
        uint32_t live = 0;
        auto fail = [&](const std::string& e) {
            if (r.error.empty()) r.error = e;
            b.failed.store(true);
            { std::lock_guard<std::mutex> lk(bmu); }
            bcv_done.notify_all();
        };
        auto go = [&](uint32_t s) {
            if (b.failed.load()) return;
            const size_t j = b.next.fetch_add(1);
            if (j >= b.n_jobs) return;
            r.slot_job[s] = (uint32_t) j;
            r.issued[s] = now_us();
            std::string e;
            if (!r.file.submit(b.off[j], r.slab + (size_t) s * PAGE, PAGE, s, e)) { fail(e); return; }
            ++r.reads;
            ++live;
        };
        for (uint32_t s = 0; s < batch_depth; ++s) go(s);
        std::vector<uint32_t> finished;
        Completion got[64];
        while (live > 0) {
            const int n = r.file.wait(got, 64, -1);
            finished.clear();
            for (int i = 0; i < n; ++i) {
                if (got[i].tag == DirectFile::WAKE_TAG || got[i].tag >= batch_depth) continue;
                const uint32_t s = (uint32_t) got[i].tag, j = r.slot_job[s];
                --live;
                r.lat.push_back((float) (now_us() - r.issued[s]));
                r.bytes += got[i].bytes;
                bool ok = got[i].ok;
                if (!ok) fail("PleReader: a table read failed");
                const uint8_t* buf = r.slab + (size_t) s * PAGE;
                const uint64_t lo = b.off[j];
                for (uint32_t u = b.begin[j]; ok && u < b.begin[j + 1]; ++u) {
                    // the part of the row this page holds (all of it, or one side of a page boundary)
                    const uint32_t i_row = b.uses[u];
                    const uint64_t at = b.table_offset + (uint64_t) b.rows[i_row] * b.rb;
                    const uint64_t a = std::max(at, lo), e = std::min(at + b.rb, lo + PAGE);
                    if (e > lo + got[i].bytes) { fail("PleReader: short read inside the table"); ok = false; break; }
                    std::memcpy(b.out + (size_t) i_row * b.rb + (a - at), buf + (a - lo), (size_t) (e - a));
                }
                if (ok) finished.push_back(j);
                go(s);
            }
            if (finished.empty()) continue;
            bool group_done = false;
            for (uint32_t j : finished) group_done |= b.left[j / b.per_group].fetch_sub(1) == 1;
            if (group_done) {
                { std::lock_guard<std::mutex> lk(bmu); }
                bcv_done.notify_all();
            }
        }
    }

    void stop_readers() {
        {
            std::lock_guard<std::mutex> lk(bmu);
            bstop = true;
        }
        bcv_work.notify_all();
        for (auto& r : readers) {
            if (r->th.joinable()) r->th.join();
            r->file.close();
            DirectFile::free_aligned(r->slab);
        }
        readers.clear();
        bstop = false;
    }

    uint8_t* slot_buf(uint32_t s) { return slab + (size_t) s * 2 * PAGE; }
    bool busy() const { return free_slots.size() < max_inflight || !delayed.empty(); }

    void cancel_queued() {
        for (const Job& job : queue) {
            auto it = tickets.find(job.ticket);
            if (it != tickets.end() && it->second.pending > 0) --it->second.pending;
        }
        queue.clear();
    }

    void record_latency(double us) {
        stats.read_us_sum += us;
        if (stats.read_us.size() < LATENCY_RING) stats.read_us.push_back((float) us);
        else stats.read_us[ring_pos++ % LATENCY_RING] = (float) us;
    }

    bool pump() {
        while (!queue.empty() && !free_slots.empty()) {
            const uint32_t s = free_slots.back();
            free_slots.pop_back();
            inflight[s] = std::move(queue.front());
            queue.pop_front();
            Job& j = inflight[s];
            j.issued_us = now_us();
            std::string kerr;             // a keep-alive read that cannot go out must not fail the reader
            if (!file.submit(j.offset, slot_buf(s), j.length, s, j.keepalive ? kerr : error)) {
                if (j.keepalive) {                 // no more of them: the SSD may sleep as before
                    keep_us = 0;
                    j.keepalive = false;
                    free_slots.push_back(s);
                    continue;
                }
                j.uses.clear();
                auto it = tickets.find(j.ticket);
                if (it != tickets.end() && it->second.pending > 0) --it->second.pending;
                free_slots.push_back(s);
                cancel_queued();
                return false;
            }
            last_read_us = j.issued_us;
            if (!j.keepalive) {                    // a keep-alive read counts when it completes (`finish`)
                stats.submit_us += now_us() - j.issued_us;
                ++stats.reads;
            }
        }
        return true;
    }

    /// When the next keep-alive read is due, or < 0 when none is: it is off, the reader failed, or no rows
    /// were asked for within the window (then the SSD may sleep; the next `issue` re-arms it).
    double keepalive_due(double now) const {
        if (keep_us <= 0 || !error.empty() || last_issue_us <= 0 || now - last_issue_us > keep_window_us) return -1;
        return last_read_us + keep_us;
    }

    /// One page of the table, a different one each time, so the SSD really reads (not its controller's buffer).
    void queue_keepalive() {
        rng ^= rng << 13;
        rng ^= rng >> 7;
        rng ^= rng << 17;
        const uint64_t first = table_offset / PAGE, end = (table_offset + n_rows * (uint64_t) row_bytes) / PAGE;
        Job j;
        j.offset = (first + (end > first ? rng % (end - first) : 0)) * PAGE;
        j.length = PAGE;
        j.keepalive = true;
        queue.push_back(std::move(j));
    }

    bool finish(const Completion& c) {
        const uint32_t s = (uint32_t) c.tag;
        if (s >= inflight.size()) {
            error = "PleReader: invalid table-read completion";
            cancel_queued();
            return false;
        }
        Job& j = inflight[s];
        if (j.keepalive) {                         // no rows: it counts once it is done, with how long it took;
            if (c.ok) {                            // a failed one turns the keep-alive off, not the reader
                ++stats.keepalive_reads;
                stats.keepalive_us_max = std::max(stats.keepalive_us_max, now_us() - j.issued_us);
            } else {
                keep_us = 0;
            }
            j.keepalive = false;
            free_slots.push_back(s);
            return error.empty() ? pump() : true;
        }
        if (!c.ok) {
            error = "PleReader: a table read failed";
            cancel_queued();
            j.uses.clear();
            auto it = tickets.find(j.ticket);
            if (it != tickets.end() && it->second.pending > 0) --it->second.pending;
            free_slots.push_back(s);
            return false;
        }
        record_latency(now_us() - j.issued_us);
        stats.bytes += c.bytes;
        const uint8_t* buf = slot_buf(s);
        for (const Use& u : j.uses) {
            if (u.in_page + row_bytes > c.bytes) {
                error = "PleReader: short read inside the table";
                cancel_queued();
                j.uses.clear();
                auto it = tickets.find(j.ticket);
                if (it != tickets.end() && it->second.pending > 0) --it->second.pending;
                free_slots.push_back(s);
                return false;
            }
        }
        for (const Use& u : j.uses) {
            std::memcpy(u.dst, buf + u.in_page, row_bytes);
            cache.insert(u.row, buf + u.in_page);
        }
        auto it = tickets.find(j.ticket);
        if (it != tickets.end() && it->second.pending > 0) --it->second.pending;
        j.uses.clear();
        free_slots.push_back(s);
        return error.empty() ? pump() : true;
    }

    /// Completions (or wake packets) just returned by `file.wait`, applying fault injection.
    bool process(const Completion* got, int n) {
        bool ok = true;
        for (int i = 0; i < n; ++i) {
            if (got[i].tag == DirectFile::WAKE_TAG) continue;
            if (got[i].tag >= inflight.size()) {
                error = "PleReader: invalid table-read completion";
                cancel_queued();
                ok = false;
                continue;
            }
            if (delay_us > 0 && now_us() - inflight[(uint32_t) got[i].tag].issued_us < delay_us) {
                delayed.push_back(got[i]);
                ++stats.late_injected;
                continue;
            }
            if (!finish(got[i])) ok = false;   // drain the rest of this batch so no completed slot is stranded
        }
        return ok;
    }

    bool release_delayed() {
        const double now = now_us();
        for (size_t i = 0; i < delayed.size();) {
            if (now - inflight[(uint32_t) delayed[i].tag].issued_us >= delay_us) {
                const Completion c = delayed[i];
                delayed.erase(delayed.begin() + (ptrdiff_t) i);
                if (!finish(c)) return false;
            } else {
                ++i;
            }
        }
        return true;
    }

    /// Caller-thread mode: process whatever has completed; blocks up to `timeout_ms` for the first completion.
    bool drain(int timeout_ms) {
        Completion got[64];
        if (!delayed.empty()) {
            if (!release_delayed()) return false;
            timeout_ms = 0;                        // keep polling the held completions
        }
        const int n = file.wait(got, 64, timeout_ms);
        return process(got, n);
    }

    void worker_loop() {
        std::unique_lock<std::mutex> lk(mu);
        for (;;) {
            // Idle: wait for work. While rows are being asked for, the wait ends in time for a keep-alive read.
            while (!(stop || !queue.empty() || busy())) {
                const double now = now_us();
                const double due = keepalive_due(now);
                if (due < 0) cv_work.wait(lk);
                else if (now >= due) queue_keepalive();
                else cv_work.wait_for(lk, std::chrono::microseconds(std::max<int64_t>(1000, (int64_t) (due - now))));
            }
            if (stop && !busy()) break;
            if (error.empty() && !pump() && error.empty()) error = "PleReader: submit failed";
            if (!delayed.empty() && !release_delayed() && error.empty()) error = "PleReader: read failed";
            if (!error.empty()) {
                cv_done.notify_all();
                if (!busy()) { cv_work.wait(lk, [&] { return stop; }); break; }
            }
            if (!busy()) { cv_done.notify_all(); continue; }
            // Block in the port WITHOUT the lock, so `issue` can queue work; `issue` wakes us with a packet.
            const int timeout = delayed.empty() ? -1 : 0;
            lk.unlock();
            Completion got[64];
            const int n = file.wait(got, 64, timeout);
            lk.lock();
            if (!process(got, n) && error.empty()) error = "PleReader: read failed";
            cv_done.notify_all();
        }
    }
};

PleReader::PleReader() : impl_(new Impl) {}
PleReader::~PleReader() {
    close();
    delete impl_;
}

bool PleReader::open(const std::string& path, uint64_t table_offset, uint64_t n_rows, uint32_t max_inflight,
                     uint64_t cache_rows, std::string& err, bool io_thread, uint32_t row_bytes) {
    close();
    if (max_inflight == 0 || max_inflight > 1024) { err = "PleReader: max_inflight must be 1..1024"; return false; }
    if (row_bytes == 0 || row_bytes > PAGE) { err = "PleReader: row_bytes must be 1..4096"; return false; }
    if (!impl_->file.open(path, err)) return false;
    impl_->path = path;
    impl_->row_bytes = row_bytes;
    if (table_offset + n_rows * (uint64_t) row_bytes > impl_->file.size()) {
        err = "PleReader: the table extends past the end of " + path;
        close();
        return false;
    }
    impl_->table_offset = table_offset;
    impl_->n_rows = n_rows;
    impl_->max_inflight = max_inflight;
    impl_->slab = (uint8_t*) DirectFile::alloc_aligned((size_t) max_inflight * 2 * PAGE);
    if (impl_->slab == nullptr) { err = "PleReader: cannot allocate read buffers"; close(); return false; }
    impl_->inflight.assign(max_inflight, Job{});
    impl_->free_slots.clear();
    for (uint32_t s = max_inflight; s-- > 0;) impl_->free_slots.push_back(s);
    impl_->cache.init(cache_rows, row_bytes);
    impl_->error.clear();
    impl_->keep_us = 0;
    impl_->last_issue_us = impl_->last_read_us = 0;
    impl_->rng = 0x9E3779B97F4A7C15ull ^ (uint64_t) now_us();
    if (impl_->rng == 0) impl_->rng = 1;
    reset_stats();
    impl_->stop = false;
    impl_->threaded = io_thread;
    if (io_thread) {
        try {
            impl_->worker = std::thread([this] { impl_->worker_loop(); });
        } catch (const std::exception& e) {
            err = std::string("PleReader: cannot create I/O worker: ") + e.what();
            close();
            return false;
        } catch (...) {
            err = "PleReader: cannot create I/O worker";
            close();
            return false;
        }
    }
    return true;
}

void PleReader::close() {
    Impl& m = *impl_;
    {
        std::lock_guard<std::mutex> call(m.bcall);   // after a batch request still running
        m.stop_readers();
    }
    if (m.worker.joinable()) {
        {
            std::lock_guard<std::mutex> lk(m.mu);
            m.stop = true;
            m.queue.clear();                       // unsubmitted work is dropped; in-flight reads still drain
        }
        m.cv_work.notify_all();
        m.file.wake();
        m.worker.join();
    }
    // Outstanding reads must finish before their buffers are released (caller-thread mode, or a worker that
    // stopped on an error).
    if (m.file.is_open()) {
        while (m.free_slots.size() < m.max_inflight) {
            Completion c[64];
            const int n = m.file.wait(c, 64, -1);
            if (n == 0) break;
            for (int i = 0; i < n; ++i)
                if (c[i].tag != DirectFile::WAKE_TAG) m.free_slots.push_back((uint32_t) c[i].tag);
        }
    }
    m.file.close();
    DirectFile::free_aligned(m.slab);
    m.slab = nullptr;
    m.queue.clear();
    m.tickets.clear();
    m.delayed.clear();
    m.inflight.clear();
    m.free_slots.clear();
    m.cache.init(0);
    m.threaded = false;
    m.stop = false;
    m.keep_us = 0;
    m.last_issue_us = m.last_read_us = 0;
}

bool PleReader::is_open() const { return impl_->file.is_open(); }
uint32_t PleReader::row_bytes() const { return impl_->row_bytes; }

PleReader::Ticket PleReader::issue(const uint32_t* rows, size_t n, uint8_t* out_raw) {
    Impl& m = *impl_;
    std::unique_lock<std::mutex> lk(m.mu, std::defer_lock);
    if (m.threaded) lk.lock();
    const uint32_t id = m.next_ticket++;
    if (m.next_ticket == 0) m.next_ticket = 1;
    const double now = now_us();
    const bool rearm = m.keep_us > 0 && (m.last_issue_us <= 0 || now - m.last_issue_us > m.keep_window_us);
    m.last_issue_us = now;
    TicketState& ts = m.tickets[id];
    std::unordered_map<uint64_t, size_t> by_page;     // aligned offset -> index in `jobs`
    std::vector<Job> jobs;
    for (size_t i = 0; i < n; ++i) {
        const uint32_t rb = m.row_bytes;
        uint8_t* dst = out_raw + i * rb;
        ++m.stats.requests;
        if (rows[i] >= m.n_rows) {
            std::memset(dst, 0, rb);
            continue;
        }
        if (const uint8_t* hit = m.cache.find(rows[i])) {
            std::memcpy(dst, hit, rb);
            ++m.stats.cache_hits;
            continue;
        }
        const uint64_t at = m.table_offset + (uint64_t) rows[i] * rb;
        const uint64_t first = at / PAGE * PAGE;
        const uint32_t length = (uint32_t) ((at + rb - 1) / PAGE * PAGE - first + PAGE);
        auto f = by_page.find(first);
        if (f != by_page.end()) {
            Job& j = jobs[f->second];
            j.length = std::max(j.length, length);
            j.uses.push_back(Use{rows[i], (uint32_t) (at - first), dst});
            ++m.stats.dedup_rows;
            continue;
        }
        by_page.emplace(first, jobs.size());
        Job j;
        j.offset = first;
        j.length = length;
        j.ticket = id;
        j.uses.push_back(Use{rows[i], (uint32_t) (at - first), dst});
        jobs.push_back(std::move(j));
    }
    // Sorted by offset: prefill chunks then read the SSD in near-sequential order.
    std::sort(jobs.begin(), jobs.end(), [](const Job& a, const Job& b) { return a.offset < b.offset; });
    ts.pending = (uint32_t) jobs.size();
    const bool has_jobs = ts.pending > 0;
    for (Job& j : jobs) m.queue.push_back(std::move(j));
    if (m.threaded) {
        lk.unlock();
        if (has_jobs) {
            m.cv_work.notify_one();
            m.file.wake();                         // in case the worker is blocked in the port
        } else if (rearm) {
            m.cv_work.notify_one();                // the keep-alive had lapsed: start it again
        }
    } else if (!m.pump() && m.error.empty()) {
        m.error = "PleReader: submit failed";
    }
    return Ticket{id};
}

bool PleReader::collect(Ticket t, std::string& err) {
    Impl& m = *impl_;
    const double start = now_us();
    if (m.threaded) {
        std::unique_lock<std::mutex> lk(m.mu);
        auto it = m.tickets.find(t.id);
        if (it == m.tickets.end()) { err = "PleReader: unknown ticket"; return false; }
        m.cv_done.wait(lk, [&] {
            const auto current = m.tickets.find(t.id);
            return current == m.tickets.end() || current->second.pending == 0;
        });
        it = m.tickets.find(t.id);
        if (it == m.tickets.end()) { err = "PleReader: unknown ticket"; return false; }
        if (!m.error.empty()) { err = m.error; return false; }
        m.stats.wait_us += now_us() - start;
        m.tickets.erase(it);
        return true;
    }
    auto it = m.tickets.find(t.id);
    if (it == m.tickets.end()) { err = "PleReader: unknown ticket"; return false; }
    while (it->second.pending > 0) {
        const bool drained = m.drain(-1);
        if (!drained && m.error.empty()) m.error = "PleReader: read failed";
        it = m.tickets.find(t.id);
        if (it == m.tickets.end()) { err = "PleReader: unknown ticket"; return false; }
    }
    if (!m.error.empty()) { err = m.error; return false; }
    m.stats.wait_us += now_us() - start;
    m.tickets.erase(it);
    return true;
}

bool PleReader::set_batch_readers(unsigned k, std::string& err) {
    Impl& m = *impl_;
    std::lock_guard<std::mutex> call(m.bcall);
    m.stop_readers();
    if (k == 0) return true;
    if (!m.file.is_open()) { err = "PleReader: not open"; return false; }
    m.batch_depth = std::max<uint32_t>(8, m.max_inflight / k);
    for (unsigned i = 0; i < k; ++i) {
        auto r = std::make_unique<BatchReader>();
        if (!r->file.open(m.path, err, true)) { m.readers.push_back(std::move(r)); m.stop_readers(); return false; }
        r->slab = (uint8_t*) DirectFile::alloc_aligned((size_t) m.batch_depth * PAGE);
        r->slot_job.assign(m.batch_depth, 0);
        r->issued.assign(m.batch_depth, 0.0);
        const bool have = r->slab != nullptr;
        m.readers.push_back(std::move(r));
        if (!have) { err = "PleReader: cannot allocate batch read buffers"; m.stop_readers(); return false; }
    }
    uint64_t g0;
    {
        std::lock_guard<std::mutex> lk(m.bmu);
        g0 = m.bgen;
    }
    try {
        for (auto& r : m.readers) {
            BatchReader* p = r.get();
            p->th = std::thread([&m, p, g0] { m.reader_loop(*p, g0); });
        }
    } catch (const std::exception& e) {
        err = std::string("PleReader: cannot create a batch reader: ") + e.what();
        m.stop_readers();
        return false;
    }
    return true;
}

unsigned PleReader::batch_readers() const { return (unsigned) impl_->readers.size(); }

bool PleReader::read_batch(const uint32_t* rows, size_t n, uint8_t* out_raw, std::string& err,
                           const std::function<void(size_t)>& ready) {
    Impl& m = *impl_;
    std::unique_lock<std::mutex> call(m.bcall);
    // (pages and rows are 32-bit in the request's tables)
    if (m.readers.empty() || 2 * (uint64_t) n >= 0xFFFFFFFFull ||
        (m.table_offset + m.n_rows * (uint64_t) m.row_bytes) / PAGE >= 0xFFFFFFFFull) {
        call.unlock();
        const Ticket t = issue(rows, n, out_raw);
        if (!collect(t, err)) return false;
        if (ready) ready(n);
        return true;
    }
    const uint32_t rb = m.row_bytes;
    // the rows the cache holds (and out-of-range rows: zeros), as `issue` serves them
    m.b_missed.clear();
    bool rearm = false;
    {
        std::unique_lock<std::mutex> lk(m.mu, std::defer_lock);
        if (m.threaded) lk.lock();
        const double now = now_us();
        rearm = m.keep_us > 0 && (m.last_issue_us <= 0 || now - m.last_issue_us > m.keep_window_us);
        m.last_issue_us = now;
        m.stats.requests += n;
        for (size_t i = 0; i < n; ++i) {
            uint8_t* dst = out_raw + i * rb;
            if (rows[i] >= m.n_rows) { std::memset(dst, 0, rb); continue; }
            if (const uint8_t* hit = m.cache.find(rows[i])) {
                std::memcpy(dst, hit, rb);
                ++m.stats.cache_hits;
                continue;
            }
            m.b_missed.push_back((uint32_t) i);
        }
    }
    if (rearm) m.cv_work.notify_one();
    // one 4 KiB read per page, numbered in the order of the first row that needs it. A row across a page boundary
    // is a use of both pages, each copying its part: 8 KiB reads cost this drive far more than two pages (3% of
    // them took the 32K prompt's reads from ~970K to ~735K a second)
    const size_t nm = m.b_missed.size();
    size_t ne = nm;   // (row, page) uses: a row on two pages counts twice
    for (size_t k = 0; k < nm; ++k) ne += (m.table_offset + (uint64_t) rows[m.b_missed[k]] * rb) % PAGE + rb > PAGE;
    int bits = 1;
    while (((size_t) 1 << bits) < 2 * ne) ++bits;   // the page map stays at most half full
    const size_t cap = (size_t) 1 << bits;
    m.b_map.assign(cap, ~0ull);                     // page << 32 | job
    m.b_off.clear();
    m.b_begin.clear();
    m.b_ent.resize(ne);
    m.b_job_of.resize(ne);
    uint64_t dedup = 0;
    for (size_t k = 0, e = 0; k < nm; ++k) {
        const uint64_t at = m.table_offset + (uint64_t) rows[m.b_missed[k]] * rb;
        for (uint64_t page = at / PAGE; page <= (at + rb - 1) / PAGE; ++page, ++e) {
            size_t h = (size_t) ((page * 0x9E3779B97F4A7C15ull) >> (64 - bits));
            while (m.b_map[h] != ~0ull && (m.b_map[h] >> 32) != page) h = (h + 1) & (cap - 1);
            uint32_t j;
            if (m.b_map[h] != ~0ull) {
                j = (uint32_t) m.b_map[h];
                ++dedup;
            } else {
                j = (uint32_t) m.b_off.size();
                m.b_map[h] = page << 32 | j;
                m.b_off.push_back(page * PAGE);
                m.b_begin.push_back(0);
            }
            m.b_ent[e] = m.b_missed[k];
            m.b_job_of[e] = j;
            ++m.b_begin[j];
        }
    }
    const size_t nj = m.b_off.size();
    m.b_begin.push_back(0);
    m.b_cur.resize(nj);
    for (size_t j = 0, sum = 0; j <= nj; ++j) {   // counts -> starts, and each job's cursor
        const uint32_t c = m.b_begin[j];
        m.b_begin[j] = (uint32_t) sum;
        if (j < nj) m.b_cur[j] = (uint32_t) sum;
        sum += c;
    }
    m.b_uses.resize(ne);
    for (size_t e = 0; e < ne; ++e) m.b_uses[m.b_cur[m.b_job_of[e]]++] = m.b_ent[e];
    if (nj == 0) {
        if (ready) ready(n);
        return true;
    }
    BatchJobs b;
    b.rows = rows;
    b.out = out_raw;
    b.rb = rb;
    b.table_offset = m.table_offset;
    b.off = m.b_off.data();
    b.begin = m.b_begin.data();
    b.uses = m.b_uses.data();
    b.n_jobs = nj;
    b.per_group = std::max<size_t>(64, (nj + 47) / 48);   // ~48 steps of `ready`
    const size_t ng = (nj + b.per_group - 1) / b.per_group;
    b.left.reset(new std::atomic<uint32_t>[ng]);
    for (size_t g = 0; g < ng; ++g) b.left[g].store((uint32_t) std::min(b.per_group, nj - g * b.per_group));
    for (auto& r : m.readers) {
        r->lat.clear();
        r->bytes = r->reads = 0;
        r->error.clear();
    }
    const double start = now_us();
    {
        std::lock_guard<std::mutex> lk(m.bmu);
        m.bcur = &b;
        m.bactive = (unsigned) m.readers.size();
        ++m.bgen;
    }
    m.bcv_work.notify_all();
    size_t cached = 0;   // b_missed[0, cached) are in the row cache
    for (size_t g = 0; g < ng; ++g) {
        {
            std::unique_lock<std::mutex> lk(m.bmu);
            m.bcv_done.wait(lk, [&] { return b.left[g].load() == 0 || b.failed.load(); });
        }
        if (b.failed.load()) break;
        const size_t jn = (g + 1) * b.per_group;
        const size_t rows_in = jn < nj ? m.b_uses[m.b_begin[jn]] : n;   // every row before job jn's first is in
        if (m.cache.sets > 0) {   // the rows read so far go to the row cache too, as `finish` puts them
            std::unique_lock<std::mutex> lk(m.mu, std::defer_lock);
            if (m.threaded) lk.lock();
            for (; cached < nm && m.b_missed[cached] < rows_in; ++cached)
                m.cache.insert(rows[m.b_missed[cached]], out_raw + (size_t) m.b_missed[cached] * rb);
        }
        if (ready) ready(rows_in);
    }
    {
        std::unique_lock<std::mutex> lk(m.bmu);   // no reader may touch `b` after this
        m.bcv_done.wait(lk, [&] { return m.bactive == 0; });
        m.bcur = nullptr;
    }
    std::unique_lock<std::mutex> lk(m.mu, std::defer_lock);
    if (m.threaded) lk.lock();
    for (auto& r : m.readers) {
        for (float us : r->lat) m.record_latency(us);
        m.stats.bytes += r->bytes;
        m.stats.reads += r->reads;
        if (err.empty() && !r->error.empty()) err = r->error;
    }
    m.stats.dedup_rows += dedup;
    m.stats.wait_us += now_us() - start;
    m.last_read_us = now_us();
    if (b.failed.load()) {
        if (err.empty()) err = "PleReader: a table read failed";
        return false;
    }
    return true;
}

void PleReader::set_keepalive(double period_ms, double window_s) {
    Impl& m = *impl_;
    {
        std::lock_guard<std::mutex> lk(m.mu);
        m.keep_us = m.threaded && period_ms > 0 ? period_ms * 1000.0 : 0.0;
        m.keep_window_us = window_s > 0 ? window_s * 1e6 : 0.0;
    }
    m.cv_work.notify_all();
}

void PleReader::set_injected_delay_us(double delay_us) {
    std::lock_guard<std::mutex> lk(impl_->mu);
    impl_->delay_us = delay_us < 0 ? 0 : delay_us;
}
const ReaderStats& PleReader::stats() const { return impl_->stats; }
ReaderStats PleReader::snapshot() const {
    Impl& m = *impl_;
    std::unique_lock<std::mutex> lk(m.mu, std::defer_lock);
    if (m.threaded) lk.lock();
    return m.stats;
}
void PleReader::reset_stats() {
    Impl& m = *impl_;
    std::unique_lock<std::mutex> lk(m.mu, std::defer_lock);
    if (m.threaded) lk.lock();                     // the worker may be counting a keep-alive read
    m.stats = ReaderStats{};
    m.ring_pos = 0;
}
uint64_t PleReader::cache_capacity() const { return impl_->cache.sets * WAYS; }
uint64_t PleReader::cache_size() const { return impl_->cache.used; }

}  // namespace strata::ngram
