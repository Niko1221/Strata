// Copyright (c) 2026 Peasant Smith and the Project Maya contributors (MIT; https://github.com/mw00/project-maya, v1.1.17).
// Ported into Strata (third_party/project-maya/NOTICE.md, LICENSE); changes are in the git history of this file.
// src/ds4/ds4_main.cpp - maya-ds4: DeepSeek-V4.1-Flash on Maya.
//
//   maya-ds4 --pack <model>/maya-ds4 --ids 0,128803,... --n 16          greedy tokens after the ids
//   maya-ds4 --pack ... --ids ... --dump-logits out.f32                every position's logits (the oracle's check)
//   maya-ds4 --pack ... --serve                                       Maya's engine protocol on stdin / stdout
//
// --serve speaks what serve/server.py's StrataEngine expects: INFO lines, then `READY <context> stop`; per request
// `GEN <max_new> [key=value ...] <id,id,...>` answered by `T <id>` lines and `DONE <generated> <prompt> <prompt ms>
// <decode ms> <stop|length|cancel> 0 0 <reused> 0 0`; `STOP` ends a request early, `QUIT` the engine.  A prompt that
// continues the state (the conversation so far) reuses it; one that only shares the last prompt's start goes back to
// that prompt's end; anything else starts over.  `GENI <max_new> [keys] <file> <ids>` carries the prompt's pictures
// (the file: see read_pics); the state is shared only where the same picture sits at the same place.  With
// STRATA_DS4_SLOT_KEEP=1 the conversations are kept when another one takes the model (a slot each: the state, the
// compressed KV included), and across runs in STRATA_DS4_SLOT_DIR - so a request that does not continue the state but
// does continue a kept conversation reads only its new tokens.
// `shared=N` (or `pin=N`) on a GEN marks the first N prompt tokens as a start other requests share (a system prompt and
// its tools): the read stops there for a restore point, so a request behind the same start - another agent - goes back
// to it instead of reading it again (also without slots).  `marks=a,b,..` and `tail=P` ask for more restore points (the
// start of the last user message, a point near the prompt's tail; ds4_marks.hpp, STRATA_DS4_RESTORE_POINTS of them at
// most a conversation, 4; 0: none kept), where a prompt that leaves the conversation after one goes back to it instead
// of further.  STRATA_DS4_REUSE=0 turns every reuse off (each request reads its whole prompt).
#if defined(__APPLE__)
#include <execinfo.h>
#include <csignal>
#include <unistd.h>
#include <libproc.h>   // proc_pid_rusage: the bytes this process had the storage read (io_read_bytes)
#endif
#include "ds4_launch.h"   // ds4_dbg
#include "ds4_evt.hpp"
#include "ds4_marks.hpp"
#include "ds4_model.hpp"
#include "ds4_experts_rule.hpp"
#include "ds4_os.hpp"
#include "strata/platform/io_priority.hpp"
#include "ds4_packinfo.hpp"
#include "ds4_slotmeta.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <atomic>
#include <chrono>
#include <climits>
#include <cmath>
#include <condition_variable>
#include <csignal>
#include <filesystem>
#include <memory>
#include <cstdio>
#include <fstream>
#include <functional>
#include <sstream>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <iostream>
#include <mutex>
#include <string>
#include <thread>
#include <vector>
#if defined(_WIN32)
#include <io.h>
#else
#include <unistd.h>
#include <cerrno>
#endif

namespace {

double now_ms() {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

// the bytes this process had the storage read for it (Linux's /proc/self/io, macOS's rusage; -1 where there is none): the
// Monitor's disk reads a token, the experts read in place from the model's files included
long long io_read_bytes() {
#ifdef __linux__
    std::FILE* f = std::fopen("/proc/self/io", "r");
    if (f == nullptr) return -1;
    char line[128];
    long long v = -1;
    while (std::fgets(line, sizeof line, f) != nullptr)
        if (std::sscanf(line, "read_bytes: %lld", &v) == 1) break;
    std::fclose(f);
    return v;
#elif defined(__APPLE__)
    // (the same number `iotop`/Activity Monitor's "Bytes read" shows: every read the storage did for this process, the
    // page cache's misses of the experts read in place included; a page-cache hit counts no storage read, which is what
    // the 60 tps go/no-go needs)
    struct rusage_info_v4 ri;
    std::memset(&ri, 0, sizeof ri);
    if (proc_pid_rusage(getpid(), RUSAGE_INFO_V4, (rusage_info_t*) &ri) != 0) return -1;
    return (long long) ri.ri_diskio_bytesread;
#else
    return -1;
#endif
}

std::vector<int> parse_ids(const std::string& s) {
    std::vector<int> v;
    const char* p = s.c_str();
    while (*p) {
        char* e = nullptr;
        const long x = std::strtol(p, &e, 10);
        if (e == p) break;
        v.push_back((int) x);
        p = e;
        while (*p == ',' || *p == ' ') ++p;
    }
    return v;
}

void usage() {
    std::fprintf(stderr,
                 "maya-ds4 --pack <model>/maya-ds4 [--gpus 0,1,...] [--max-context N] [--reserve-mib N]\n"
                 "           [--vram-experts F] [--cpu-threads N] [--expert-profile FILE|none] [--profile-readonly] [--ssd-cache GB] [--ssd-policy lru|lfu|lfulru] [--no-expert-cache] [--verbose]\n"
                 "           [--tp N] [--threads on|off] [--prefill-chunk N] [--dspark on|off|none|auto] [--dspark-block N] [--dspark-conf F] [--experts N|auto]\n"
                 "           [--vision on|off] [--vision-resident (--serve: keep the picture tower on the card; default: in RAM between pictures)] [--mmap-experts on|off|auto] [--parallel N (--serve: N conversations at once)]\n"
                 "           (--ids I,J,... [--images FILE] [--n N] [--dump-logits FILE] | --serve\n"
                 "           | --encode-image FILE.rgb --size WxH --dump-image OUT\n"
                 "           | --pack-info)\n");
}

// stdin lines on a thread, so a request can see STOP while it decodes.  read(2), not std::cin - glibc's exit()
// flushes stdio and waits on stdin's lock, which getline holds while it waits for input (the GLM serve's pattern)
struct Lines {
    std::mutex mu;
    std::condition_variable cv;
    std::deque<std::string> q;
    bool eof = false;
    std::atomic<bool> stop{false};
    // --parallel: `STOP <rid>` ends that request alone (stop_rids), a bare STOP, QUIT or the input's end all of them
    std::vector<long long> stop_rids;
    std::atomic<bool> stop_all{false};
    bool stopped(long long rid) {
        if (stop_all.load()) return true;
        std::lock_guard<std::mutex> lk(mu);
        return std::find(stop_rids.begin(), stop_rids.end(), rid) != stop_rids.end();
    }
    void forget(long long rid) {   // (the request is over: a late STOP of it changes nothing)
        std::lock_guard<std::mutex> lk(mu);
        stop_rids.erase(std::remove(stop_rids.begin(), stop_rids.end(), rid), stop_rids.end());
    }
    // the next line if one is waiting (false: none now) - the several-conversation loop reads between its steps
    bool poll(std::string& l, bool& end) {
        std::lock_guard<std::mutex> lk(mu);
        end = q.empty() && eof;
        if (q.empty()) return false;
        l = q.front();
        q.pop_front();
        return true;
    }
    void start() {
        std::thread([this] {
            std::string l, buf;
            char chunk[4096];
            auto getline_fd = [&](std::string& out) -> bool {
                for (;;) {
                    const size_t nlpos = buf.find('\n');
                    if (nlpos != std::string::npos) {
                        out.assign(buf, 0, nlpos);
                        buf.erase(0, nlpos + 1);
                        return true;
                    }
#if defined(_WIN32)
                    const int n = _read(0, chunk, (unsigned) sizeof chunk);
#else
                    const ssize_t n = ::read(0, chunk, sizeof chunk);
                    if (n < 0 && errno == EINTR) continue;
#endif
                    if (n <= 0) {
                        if (buf.empty()) return false;
                        out.swap(buf);
                        buf.clear();
                        return true;
                    }
                    buf.append(chunk, (size_t) n);
                }
            };
            while (getline_fd(l)) {
                if (!l.empty() && l.back() == '\r') l.pop_back();
                std::lock_guard<std::mutex> lk(mu);
                if (l == "STOP" || l.rfind("STOP ", 0) == 0) {
                    stop.store(true);
                    if (l == "STOP") stop_all.store(true);
                    else stop_rids.push_back(std::atoll(l.c_str() + 5));
                    cv.notify_all();
                    continue;
                }
                if (l == "QUIT") stop.store(true), stop_all.store(true);
                q.push_back(l);
                cv.notify_all();
            }
            std::lock_guard<std::mutex> lk(mu);
            eof = true;
            stop.store(true);
            stop_all.store(true);
            cv.notify_all();
        }).detach();
    }
    bool next(std::string& l) {
        std::unique_lock<std::mutex> lk(mu);
        cv.wait(lk, [&] { return !q.empty() || eof; });
        if (q.empty()) return false;
        l = q.front();
        q.pop_front();
        return true;
    }
};

// The drafts between two stats: each draft position (1st, 2nd, ..) - how often it was verified with every draft
// before it kept, and how often it was kept then
std::string per_position(const strata::ds4::Stats& s0, const strata::ds4::Stats& s1) {
    std::string pp = "per position (kept / reached):";
    for (int i = 0; i < 8; ++i) {
        const long long r = s1.draft_reached[i] - s0.draft_reached[i], k = s1.draft_kept[i] - s0.draft_kept[i];
        if (r <= 0) break;
        char b[64];
        std::snprintf(b, sizeof b, " %d: %lld/%lld %.0f%%", i + 1, k, r, 100.0 * (double) k / (double) r);
        pp += b;
    }
    return pp;
}

static std::FILE* g_round_log = nullptr;
static std::string g_round_tag;
static int g_spec_depth = 64;   // --dspark-depth: at most this many drafts a round
static std::vector<float> g_round_conf;
static int g_round_nd = 0;

// The drafts' cut on the product of their confidences (DeepInfer's cumulative rule): a draft is verified while the chance
// that it and every draft before it are kept stays at least this - a long window whose later drafts are likely lost is
// not verified whole.  STRATA_DS4_DSPARK_CUM=F, a request's dspark_cum key; 0: off, only the floor on each draft's own
// confidence (dspark_conf).  0.5 is the default on gfx1200 / gfx1201 (RDNA4: R9700, RX 9000) only: measured on 4x R9700, DeepSeek 2.25 bpw,
// six 128-token decodes an arm in one load - 71.7 -> 76.9 tok/s (medians), the same ids.  Project Maya ran 0.5 on a Radeon 8065S (Strix Halo,
// gfx1151: +6.5% decode, the same tokens), but that is a single datapoint on a box the 0.2.7 defaults have not been tested on, so
// Strix keeps 0 until it is; the CUDA measurements in 0.2.7 had it flat, so CUDA keeps 0.  STRATA_DS4_DSPARK_CUM overrides everywhere.
float dspark_cum_default(const strata::ds4::Model& m) {
    static const char* const e = std::getenv("STRATA_DS4_DSPARK_CUM");
    return strata::ds4::dspark_cum_value(e != nullptr, e != nullptr ? (float) std::atof(e) : 0.0f, m.gpus_are_gfx12(),
                                         strata::ds4::defaults_026());
}

// One round of decoding after `tok` (the token at pos(), not fed yet): with the drafter, its drafts while their
// confidence holds (at most `room` - 1 of them), verified together; else a plain step.  `out` gets the tokens that follow
// tok (the accepted drafts, then the trunk's next), the last of them the next round's tok.
struct RoundTimes {   // DS4_ROUND_TIMES: where a decode round's time goes (ms, summed), printed at exit
    double draft = 0, verify = 0, step = 0;
    unsigned long long draft_d = 0, verify_d = 0, step_d = 0;
    long rounds = 0, spec_rounds = 0, steps = 0;
    ~RoundTimes() {
        if (ds4_dbg("DS4_ROUND_TIMES") && rounds > 0)
            std::fprintf(stderr, "rounds: %ld (%ld speculative, %ld plain steps): draft %.1f ms (%.2f a round), verify %.1f ms (%.2f), plain step %.1f ms (%.2f)\n",
                         rounds, spec_rounds, steps, draft, spec_rounds ? draft / (double) spec_rounds : 0.0, verify, spec_rounds ? verify / (double) spec_rounds : 0.0,
                         step, steps ? step / (double) steps : 0.0);
        if (ds4_dbg("DS4_ROUND_TIMES") && rounds > 0)
            std::fprintf(stderr, "dispatches a round: draft %.0f, verify %.0f, plain step %.0f\n", spec_rounds ? (double) draft_d / (double) spec_rounds : 0.0, spec_rounds ? (double) verify_d / (double) spec_rounds : 0.0, steps ? (double) step_d / (double) steps : 0.0);
    }
} g_rt;
#if defined(STRATA_METAL)
extern "C" unsigned long long ds4_dispatch_seq();
#endif
#if defined(STRATA_METAL)
extern "C" void ds4_timeline_gate(int);
#endif
static void tl_gate(bool open) {
#if defined(STRATA_METAL)
    static const bool only_draft = std::getenv("DS4_TL_DRAFT") != nullptr;
    if (only_draft) ds4_timeline_gate(open ? 1 : 0);
#endif
}
static unsigned long long rt_seq() {
#if defined(STRATA_METAL)
    return ds4_dispatch_seq();
#else
    return 0;
#endif
}
static double rt_now() { return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now().time_since_epoch()).count(); }
bool round(strata::ds4::Model& m, int tok, int room, bool spec, float conf_min, float temperature, float top_p, int top_k,
           float min_p, unsigned long long seed, unsigned long long& counter, std::vector<int>& out, int& offered,
           int& accepted, std::string& err, float cum_min = -1.0f) {
    out.clear();
    offered = accepted = 0;
    if (spec && room > 1 && m.draft_block() > 0) {
        std::vector<int> drafts;
        std::vector<float> conf;
        ++g_rt.rounds;
        static const bool tl_init = (tl_gate(false), true);
        (void) tl_init;
        tl_gate(true);
        const double rt0 = rt_now();
        const unsigned long long rs0 = rt_seq();
        if (!m.draft(tok, drafts, conf, err, temperature, top_p, top_k, min_p, seed, counter)) return false;
        g_rt.draft += rt_now() - rt0;
        tl_gate(false);
        g_rt.draft_d += rt_seq() - rs0;
        g_round_conf = conf;
        g_round_nd = (int) drafts.size();
        int keep = 0;
        if (cum_min < 0) cum_min = dspark_cum_default(m);
        float cum = 1.0f;
        while (keep < (int) drafts.size() && keep < room - 1 && keep < g_spec_depth && conf[(size_t) keep] >= conf_min &&
               (cum *= conf[(size_t) keep]) >= cum_min)
            ++keep;
        if (keep > 0) {
            drafts.resize((size_t) keep);
            int acc = 0, next = -1;
            const double rt1 = rt_now();
            const unsigned long long rs1 = rt_seq();
            if (!m.verify(tok, drafts, temperature, top_p, top_k, min_p, seed, counter, acc, next, err)) return false;
            g_rt.verify += rt_now() - rt1;
            g_rt.verify_d += rt_seq() - rs1;
            ++g_rt.spec_rounds;
            if (g_round_log != nullptr) {   // DS4_ROUND_LOG: tag, drafts offered, accepted, the confidences
                std::fprintf(g_round_log, "%s\t%d\t%d", g_round_tag.c_str(), keep, acc);
                for (int i = 0; i < keep; ++i) std::fprintf(g_round_log, "\t%.4f", g_round_conf[(size_t) i]);
                std::fprintf(g_round_log, "\n");
            }
            counter += (unsigned long long) acc + 1;
            for (int i = 0; i < acc; ++i) out.push_back(drafts[(size_t) i]);
            out.push_back(next);
            offered = keep;
            accepted = acc;
            return true;
        }
    }
    const double rt2 = rt_now();
    const unsigned long long rs2 = rt_seq();
    if (!m.step(tok, true, err)) return false;
    out.push_back(m.sample(temperature, top_p, top_k, min_p, seed, counter++));
    g_rt.step += rt_now() - rt2;
    g_rt.step_d += rt_seq() - rs2;
    ++g_rt.steps;
    return true;
}

struct Req {
    long long max_new = 0;
    float temperature = 0.0f, top_p = 1.0f, min_p = 0.0f;
    int top_k = 64;
    unsigned long long seed = 0;
    int spec = -1;          // spec=0|1: this request drafts or not (-1: as the engine was started)
    float conf = -1.0f;     // dspark_conf=F: the drafts' confidence floor for this request (-1: the engine's)
    long long pin = 0;      // pin=N: the first N prompt tokens are a start other requests share (a system prompt and its
                            // tools): the state at N is kept as a checkpoint, so another agent behind the same start
                            // goes back to it instead of reading it again
    // marks=<pos>,... and shared=<pos>: restore points the read keeps (the server's: the start of the last user message;
    // the end of a start other conversations share, given up last) - a later prompt that leaves this one past one of them
    // goes back there (ds4_marks.hpp); pin=N is shared=N.  tail=<pos>: the point near the prompt's tail, kept like a mark
    std::vector<int64_t> marks;
    long long shared = -1, tail = -1;
    long long rid = -1;     // rid=<id> (--parallel: every request's, its lines tagged with it)
    float cum = -1.0f;      // dspark_cum=F: the cut on the product of the confidences (-1: the engine's, dspark_cum_default)
    // the penalties (penalty_last_n / _repeat / _freq / _present, llama.cpp's): over the last pen_n tokens of the state
    // (-1: the whole context, which the engine caps at its window; 0: none)
    int pen_n = 0;
    float pen_repeat = 1.0f, pen_freq = 0.0f, pen_present = 0.0f;
    std::string file;   // GENI: the pictures
    std::vector<int> ids;
};

// A prompt's picture: where its span starts, its pixels (the server's resize: RGB rows, both sides multiples of the
// patch size), a hash of them; once encoded, its span's rows
struct Pic {
    int64_t start = 0;
    int width = 0, height = 0;
    uint64_t hash = 0;
    size_t len = 0;   // the span (0: not known yet)
    std::vector<uint8_t> rgb;
};

// GENI's file: int32 'DSI1', int32 count, then per picture int64 start, int32 width, int32 height and its
// height x width x 3 RGB bytes, in prompt order
namespace {
// After a fatal GPU error (an illegal memory access, a launch failure, a hardware exception) the device context is gone: every later
// CUDA / HIP call of this process fails, so an engine that keeps serving only turns each request into an error until someone restarts
// it (#4350, an unattended agent loop is dead).  The engine says so on stdout and exits; the server starts it again on the next
// request, like any engine death, and fails only the request in flight with a 503 (retryable).
bool gpu_fatal_text(const std::string& e) {
    static const char* const k[] = {"illegal memory access", "unspecified launch failure", "hardware exception", "illegal instruction",
                                    "misaligned address", "device-side assert", "uncorrectable ECC", "launch timed out",
                                    "launch failure", "device lost", "ERROR_DEVICE_LOST"};
    for (const char* w : k)
        if (e.find(w) != std::string::npos) return true;
    return false;
}
[[noreturn]] void die_gpu_fatal(const std::string& e) {
    std::fprintf(stderr, "ds4: fatal GPU error (%s): the GPU context cannot be used again, so the engine ends and the server starts it again\n",
                 e.c_str());
    std::fflush(stderr);
    std::printf("ERR fatal GPU error: %s\n", e.c_str());
    std::fflush(stdout);
    std::_Exit(3);   // (not a clean unwind: freeing device memory on a dead context can hang, as the Qwen engine's #224 exit knows)
}
// DS4_INJECT_FATAL=N (a test hook, off when unset): the Nth GEN request fails with the text of a real fatal error, taking the same path
bool inject_fatal_now() {
    static const long n = [] { const char* v = std::getenv("DS4_INJECT_FATAL"); return v != nullptr ? std::atol(v) : 0L; }();
    static long seen = 0;
    if (!(n > 0 && ++seen == n)) return false;
    // DS4_INJECT_FATAL_ONCE=<file>: fires once across restarts (the file is made when it fires; while it exists nothing is injected)
    if (const char* f = std::getenv("DS4_INJECT_FATAL_ONCE"); f != nullptr && *f) {
        if (std::FILE* t = std::fopen(f, "rb")) { std::fclose(t); return false; }
        if (std::FILE* t = std::fopen(f, "wb")) std::fclose(t);
    }
    return true;
}
}  // namespace

bool read_pics(const std::string& path, std::vector<Pic>& v, std::string& err) {
    std::FILE* f = std::fopen(path.c_str(), "rb");
    if (f == nullptr) {
        err = "cannot open the pictures' file " + path;
        return false;
    }
    char magic[4] = {};
    int32_t n = 0;
    bool ok = std::fread(magic, 1, 4, f) == 4 && std::memcmp(magic, "DSI1", 4) == 0 && std::fread(&n, 4, 1, f) == 1 &&
              n >= 0 && n <= 4096;
    for (int i = 0; ok && i < n; ++i) {
        Pic p;
        int32_t wh[2] = {};
        ok = std::fread(&p.start, 8, 1, f) == 1 && std::fread(wh, 4, 2, f) == 2 && wh[0] > 0 && wh[1] > 0 &&
             (int64_t) wh[0] * wh[1] <= (64 << 20);
        if (!ok) break;
        p.width = wh[0], p.height = wh[1];
        p.rgb.resize((size_t) p.width * p.height * 3);
        ok = std::fread(p.rgb.data(), 1, p.rgb.size(), f) == p.rgb.size();
        uint64_t h = 1469598103934665603ull;   // FNV-1a over the size and the pixels
        for (int32_t x : wh)
            for (int b = 0; b < 4; ++b) h = (h ^ (uint8_t) (x >> (8 * b))) * 1099511628211ull;
        for (uint8_t b : p.rgb) h = (h ^ b) * 1099511628211ull;
        p.hash = h;
        v.push_back(std::move(p));
    }
    std::fclose(f);
    if (!ok) err = path + ": not a pictures' file (DSI1)";
    return ok;
}

// the encoded spans, the latest few (a conversation sends its pictures with every turn)
struct SpanCache {
    std::deque<std::pair<uint64_t, std::shared_ptr<const std::vector<uint16_t>>>> q;
    std::shared_ptr<const std::vector<uint16_t>> get(strata::ds4::Model& m, const Pic& p, std::string& err) {
        for (size_t i = 0; i < q.size(); ++i)
            if (q[i].first == p.hash) {
                auto r = q[i].second;
                q.erase(q.begin() + (long) i);
                q.emplace_front(p.hash, r);
                return r;
            }
        auto rows = std::make_shared<std::vector<uint16_t>>();
        if (!m.encode_image(p.rgb.data(), p.width, p.height, *rows, err)) return nullptr;
        q.emplace_front(p.hash, rows);
        if (q.size() > 8) q.pop_back();
        return rows;
    }
};

bool parse_gen(const std::string& line, Req& r, bool geni) {
    const char* p = line.c_str() + (geni ? 5 : 4);
    char* e = nullptr;
    r.max_new = std::strtoll(p, &e, 10);
    if (e == p) return false;
    p = e;
    for (;;) {
        while (*p == ' ') ++p;
        const char* s = p;
        while (*p && *p != ' ') ++p;
        if (p == s) break;
        const std::string tok(s, (size_t) (p - s));
        const size_t eq = tok.find('=');
        if (eq == std::string::npos && geni && r.file.empty()) {
            r.file = tok;
            continue;
        }
        if (eq == std::string::npos) {
            r.ids = parse_ids(tok);
            break;
        }
        const std::string k = tok.substr(0, eq), v = tok.substr(eq + 1);
        if (k == "temperature") r.temperature = std::strtof(v.c_str(), nullptr);
        else if (k == "top_p") r.top_p = std::strtof(v.c_str(), nullptr);
        else if (k == "top_k") r.top_k = std::atoi(v.c_str());
        else if (k == "min_p") r.min_p = std::strtof(v.c_str(), nullptr);
        else if (k == "seed") r.seed = std::strtoull(v.c_str(), nullptr, 10);
        else if (k == "spec") r.spec = std::atoi(v.c_str()) != 0;
        else if (k == "dspark_conf") r.conf = std::strtof(v.c_str(), nullptr);
        else if (k == "pin") r.pin = std::atoll(v.c_str());
        else if (k == "shared") r.shared = std::atoll(v.c_str());
        else if (k == "tail") r.tail = std::atoll(v.c_str());
        else if (k == "marks") {
            std::stringstream ss(v);
            std::string x;
            while (std::getline(ss, x, ',')) {
                const long long n = std::atoll(x.c_str());
                if (n > 0) r.marks.push_back(n);
            }
        }
        else if (k == "dspark_cum" || k == "spec_cum") r.cum = std::strtof(v.c_str(), nullptr);
        else if (k == "spec_min_p") r.conf = std::strtof(v.c_str(), nullptr);   // (the server's strata_tune names)
        else if (k == "penalty_last_n") r.pen_n = std::atoi(v.c_str());
        else if (k == "penalty_repeat") r.pen_repeat = std::strtof(v.c_str(), nullptr);
        else if (k == "penalty_freq") r.pen_freq = std::strtof(v.c_str(), nullptr);
        else if (k == "penalty_present") r.pen_present = std::strtof(v.c_str(), nullptr);
        else if (k == "rid") r.rid = std::atoll(v.c_str());
    }
    return !r.ids.empty() && (!geni || !r.file.empty());
}

int serve(strata::ds4::Model& m, const std::string& eos_list, bool spec, float conf_min, const std::string& pack) {
    std::vector<int> eos = parse_ids(eos_list);
    for (const std::string& l : m.info_lines()) std::printf("%s\n", l.c_str());
    if (m.seq_count() > 1)   // (--parallel: n conversations at once, GEN rid=<id>)
        std::printf("READY %lld stop multi=%d\n", (long long) m.max_context(), m.seq_count());
    else
        std::printf("READY %lld stop\n", (long long) m.max_context());
    std::fflush(stdout);
    std::fprintf(stderr, "maya ds4 serve: GEN <max_new> [keys] <ids>; GENI <max_new> [keys] <pictures> <ids>; STOP, QUIT\n");
    Lines in;
    in.start();
    std::string line;
    unsigned long long counter = 0;
    std::vector<Pic> held;   // the pictures the state holds (start, hash, span)
    // (DS4_SERVE_DUMP=<prefix>, tests: request i's logits - the prompt's last row, then every decode step's rows - into
    //  <prefix>.<i>.f32, to compare a request's logits byte for byte across processes and histories)
    const char* dump_prefix = std::getenv("DS4_SERVE_DUMP");
    int n_req = 0;
    std::vector<float> dump_lg;
    std::FILE* dump_f = nullptr;
    auto dump_rows = [&]() {
        if (dump_f != nullptr && m.logits(dump_lg)) std::fwrite(dump_lg.data(), 4, dump_lg.size(), dump_f);
    };
    // STRATA_DS4_RESTORE_TAIL_AUTO (1): the tail point is left out where its cost is measured higher than reading its
    // tokens again; 0: kept on every read that reaches it
    static const bool tail_auto = [] {
        const char* v = std::getenv("STRATA_DS4_RESTORE_TAIL_AUTO");
        return v == nullptr || std::atoi(v) != 0;
    }();
    strata::ds4::TailCost tail_cost;
    SpanCache spans;
    const int img_tok = m.image_token();
    // CONVERSATION SLOTS: a request that does not continue the conversation in the model first sets that one aside
    // (Model::slot_save: its window rings, the compressors' partial groups and every source's compressed KV and index
    // keys), then takes back the kept conversation this prompt does continue (slot_load) - chats, and an agent's side
    // requests, that interleave no longer re-read each other's prompts (70k and 185k tokens alternating: 75-200 s a
    // switch at ~930 tok/s).  STRATA_DS4_SLOT_KEEP=1 turns them on and keeps them across runs, in STRATA_DS4_SLOT_DIR
    // (default <pack>/slots): a slot is a .bin (the state) and a .meta (this model's identity, the conversation's
    // tokens and its size), both written under a temporary name first; a slot whose .meta is missing, foreign or does
    // not match its file, or unused for STRATA_DS4_SLOT_DAYS (7) days, is removed at start.  STRATA_DS4_SLOT_CAP (4) is
    // the conversations kept in RAM; past it the least recently used is written to the folder and read back from there
    // (RAM first, the folder past the cap), and the engine's end writes the ones in RAM to the folder, so the next run
    // reads on from them.  STRATA_DS4_SLOT_GB (16) caps the folder's .bin files: at start, and after every write, the
    // folder's least recently used conversation is removed while they are over it (never the one just written, so a
    // conversation bigger than the cap stays).  STRATA_DS4_SLOT_MIN (1024) is the shortest conversation set aside (a
    // shorter one is read again in about the time its slot would take).  Text only: a conversation that holds pictures
    // is not set aside, and a prompt with pictures takes none back.
    const auto env_num = [](const char* k, double d) { const char* v = std::getenv(k); return v ? std::atof(v) : d; };
    const bool slot_keep =
        std::getenv("STRATA_DS4_SLOT_KEEP") != nullptr && std::atoi(std::getenv("STRATA_DS4_SLOT_KEEP")) != 0;
    const bool reuse = std::getenv("STRATA_DS4_REUSE") == nullptr || std::atoi(std::getenv("STRATA_DS4_REUSE")) != 0;
    int slot_max = slot_keep ? std::max(0, (int) env_num("STRATA_DS4_SLOT_CAP", 4)) : 0;
    const size_t slot_min = (size_t) std::max(1.0, env_num("STRATA_DS4_SLOT_MIN", 1024));
    const double slot_days = env_num("STRATA_DS4_SLOT_DAYS", 7);
    const uint64_t slot_budget = (uint64_t) (env_num("STRATA_DS4_SLOT_GB", 16) * 1073741824.0);
    namespace fs = std::filesystem;
    std::filesystem::path slot_dir;
    struct Slot {
        std::vector<int32_t> tokens;
        std::string path;                            // empty: the state is in data (RAM)
        uint64_t bytes = 0, used = 0;
        std::shared_ptr<std::vector<uint8_t>> data;   // the RAM slot's bytes (null: a file slot)
        std::vector<int64_t> cks;                     // the positions short of its end it can be taken back at
    };
    std::vector<Slot> slots;
    uint64_t slot_clock = 0, slot_seq = 0;
    const std::string slot_identity = m.slot_identity();
    const auto note = [&](const std::string& msg) {
        std::fprintf(stderr, "ds4 slots: %s\n", msg.c_str());
        std::printf("NOTE sessions: %s\n", msg.c_str());
        std::fflush(stdout);
    };
    const auto gb = [](uint64_t b) {
        char s[32];
        std::snprintf(s, sizeof s, "%.2f GB", (double) b / 1e9);
        return std::string(s);
    };
    const auto meta_of = [](const std::string& bin) { return bin.substr(0, bin.size() - 4) + ".meta"; };
    const auto meta_write = [&](const std::string& path, const std::vector<int32_t>& toks, uint64_t bytes,
                                const std::vector<int64_t>& cks) -> bool {
        return strata::ds4::slotmeta::write(path, slot_identity, toks, bytes, cks);
    };
    // a .meta of this model -> its tokens and the slot's size (false: foreign, damaged or not this model's)
    const auto meta_read = [&](const std::string& path, std::vector<int32_t>& toks, uint64_t& bytes,
                               std::vector<int64_t>& cks) -> bool {
        return strata::ds4::slotmeta::read(path, slot_identity, m.max_context(), toks, bytes, cks);
    };
    const auto slot_age_days = [](const std::string& p) {
        std::error_code ec;
        const auto t = fs::last_write_time(p, ec);
        if (ec) return 1e9;
        return std::chrono::duration<double>(fs::file_time_type::clock::now() - t).count() / 86400.0;
    };
    const auto slot_drop = [&](size_t i) {
        if (slots[i].data)   // a RAM slot: its buffer simply dies
            slots[i].data.reset();
        else {
            std::error_code ec;
            std::filesystem::remove(slots[i].path, ec);
            std::filesystem::remove(meta_of(slots[i].path), ec);
        }
        slots.erase(slots.begin() + (long) i);
    };
    const auto slot_count = [&](int& ram, int& disk) {
        ram = disk = 0;
        for (const Slot& s : slots) (s.data ? ram : disk) += 1;
    };
    // the folder's conversations (their .bin; a .meta beside each is a few KB: identity and tokens)
    const auto slot_folder_bytes = [&]() {
        uint64_t total = 0;
        std::error_code ec;
        for (const auto& de : std::filesystem::directory_iterator(slot_dir, ec)) {
            const std::string fn = de.path().filename().string();
            if (fn.rfind("slot-", 0) == 0 && fn.size() > 9 && fn.compare(fn.size() - 4, 4, ".bin") == 0) {
                const uint64_t bytes = (uint64_t) std::filesystem::file_size(de.path(), ec);
                if (!ec) total += bytes;
            }
        }
        return total;
    };
    // the folder over STRATA_DS4_SLOT_GB: its least recently used conversation goes first, and never `keep` (the one just
    // written), so a conversation bigger than the cap stays
    const auto slot_trim = [&](const std::string& keep) {
        const uint64_t before = slot_folder_bytes();
        size_t dropped = 0;
        while (slot_folder_bytes() > slot_budget) {
            size_t lru = SIZE_MAX;
            for (size_t i = 0; i < slots.size(); ++i)
                if (!slots[i].data && slots[i].path != keep && (lru == SIZE_MAX || slots[i].used < slots[lru].used))
                    lru = i;
            if (lru == SIZE_MAX) break;
            slot_drop(lru);
            ++dropped;
        }
        if (dropped > 0) {
            char msg[256];
            std::snprintf(msg, sizeof msg, "the folder held %s (over STRATA_DS4_SLOT_GB, %s): %zu least recently used "
                          "conversation(s) removed, %s left", gb(before).c_str(), gb(slot_budget).c_str(), dropped,
                          gb(slot_folder_bytes()).c_str());
            note(msg);
        }
    };
    // a RAM slot into the folder (its bytes, then its .meta); false: not written (its bytes go, then)
    const auto slot_write_file = [&](Slot& s) -> bool {
        std::error_code ec;
        const auto space = std::filesystem::space(slot_dir, ec);
        if (!ec && space.available < ((uint64_t) 8 << 30)) {   // (8 GB free, the GLM engine's guard)
            std::fprintf(stderr, "ds4 slots: under 8 GB free in %s - the conversation is not written\n",
                         slot_dir.string().c_str());
            return false;
        }
        const auto stamp = std::chrono::duration_cast<std::chrono::milliseconds>(
                               std::chrono::system_clock::now().time_since_epoch()).count();
        const std::string path =
            (slot_dir / ("slot-" + std::to_string(stamp) + "-" + std::to_string(++slot_seq) + ".bin")).string();
        {
            std::ofstream f(path + ".tmp", std::ios::binary | std::ios::trunc);
            f.write((const char*) s.data->data(), (std::streamsize) s.data->size());
            f.flush();
            if (!f) {
                std::filesystem::remove(path + ".tmp", ec);
                std::fprintf(stderr, "ds4 slots: writing %s.tmp failed (disk full?)\n", path.c_str());
                return false;
            }
        }
        fs::rename(path + ".tmp", path, ec);
        if (ec || !meta_write(meta_of(path), s.tokens, s.data->size(), s.cks)) {
            std::filesystem::remove(path + ".tmp", ec);
            std::filesystem::remove(path, ec);
            std::fprintf(stderr, "ds4 slots: writing %s failed\n", path.c_str());
            return false;
        }
        s.path = path;
        s.bytes = s.data->size();
        s.data.reset();
        slot_trim(path);   // (the folder over its cap: the least recently used conversation goes, not this one)
        return true;
    };
    // the conversation in the model into a slot (the state is lost otherwise: another one takes the model)
    const auto slot_put = [&](const char* why) {
        if (slot_max <= 0) return;
        const std::vector<int32_t>& t = m.tokens();
        if (t.size() < slot_min) return;   // (a fresh engine's 0-token state is no conversation: nothing to set aside)
        const auto t0 = std::chrono::steady_clock::now();
        auto buf = std::make_shared<std::vector<uint8_t>>();
        std::string e;
        const uint64_t bytes = m.slot_save(*buf, e);   // (a state unchanged since it was taken refuses here: nothing
        if (bytes == 0) {                             // changes, its slot stays)
            if (e != "it is already set aside")
                std::fprintf(stderr, "ds4 slots: setting the conversation aside failed (%s) - it is read from token 0 "
                                     "when it comes back\n", e.c_str());
            return;
        }
        for (size_t i = slots.size(); i-- > 0;)   // an earlier state of this conversation is superseded
            if (slots[i].tokens.size() <= t.size() && std::equal(slots[i].tokens.begin(), slots[i].tokens.end(), t.begin()))
                slot_drop(i);
        slots.push_back({t, "", bytes, ++slot_clock, nullptr, m.check_positions()});
        slots.back().data = buf;
        while (true) {   // past the cap: the least recently used conversation in RAM to the folder, or gone
            int ram = 0, disk = 0;
            slot_count(ram, disk);
            if (ram <= slot_max) break;
            size_t lru = SIZE_MAX;
            for (size_t i = 0; i < slots.size(); ++i)
                if (slots[i].data && (lru == SIZE_MAX || slots[i].used < slots[lru].used)) lru = i;
            if (lru == SIZE_MAX) break;
            if (!slot_write_file(slots[lru])) slot_drop(lru);
        }
        int ram = 0, disk = 0;
        slot_count(ram, disk);
        char msg[256];
        std::snprintf(msg, sizeof msg, "saved a %zu-token conversation%s (%s in RAM in %.1f s); %d in RAM, %d in %s",
                      t.size(), why, gb(bytes).c_str(),
                      std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count(), ram, disk,
                      slot_dir.string().c_str());
        note(msg);
    };
    // the kept conversation this prompt shares the most of - past `floor` tokens, what the model itself can go back
    // to - back in the model: the tokens it covers (0: none).  A conversation the prompt continues whole comes back
    // whole (`whole`); one it leaves part way - a retried prompt, a history the client renders again without the
    // earlier reasoning - comes back at its latest checkpoint the prompt still agrees with (the end of a prompt, the
    // position before it, the anchor where its run of continued prompts began).
    const auto slot_take = [&](const std::vector<int>& prompt, size_t floor, bool& whole) -> size_t {
        whole = false;
        if (slot_max <= 0 || prompt.size() < 2) return 0;
        std::vector<strata::ds4::slotmeta::Cand> cands;
        for (const Slot& sl : slots) cands.push_back({&sl.tokens, &sl.cks});
        size_t len = 0;
        const size_t best = strata::ds4::slotmeta::pick(cands, prompt, floor, &len);
        if (best == SIZE_MAX) return 0;
        const size_t whole_len = slots[best].tokens.size();
        const auto t0 = std::chrono::steady_clock::now();
        const bool ram = slots[best].data != nullptr;
        std::string e;
        bool back = false;
        if (ram) back = m.slot_load(slots[best].data->data(), slots[best].data->size(), (int64_t) whole_len, e);
        else {
            std::ifstream f(slots[best].path, std::ios::binary);
            std::vector<uint8_t> file;
            if (f) {
                f.seekg(0, std::ios::end);
                const std::streamoff n = f.tellg();
                f.seekg(0, std::ios::beg);
                if (n > 0) {
                    file.resize((size_t) n);
                    if (f.read((char*) file.data(), n))
                        back = m.slot_load(file.data(), file.size(), (int64_t) whole_len, e);
                    else e = "reading " + slots[best].path + " failed";
                } else e = slots[best].path + " is empty";
            } else e = "cannot read " + slots[best].path;
        }
        if (!back) {
            note("taking a conversation back failed (" + (e.empty() ? std::string("it did not load") : e) +
                 ") - reading its prompt again");
            slot_drop(best);
            return 0;
        }
        if (len < whole_len && !(m.restore((int64_t) len) && (size_t) m.pos() == len)) {
            note("going back to token " + std::to_string(len) + " of a kept conversation failed - reading its prompt again");
            m.reset();
            return 0;
        }
        whole = len == whole_len;
        slots[best].used = ++slot_clock;
        if (!ram) {   // its age counts from its last use (a file slot's only)
            std::error_code ec;
            fs::last_write_time(meta_of(slots[best].path), fs::file_time_type::clock::now(), ec);
        }
        char msg[256];
        std::string part;
        if (!whole) part = " at token " + std::to_string(len) + " of " + std::to_string(whole_len);
        std::snprintf(msg, sizeof msg, "took back a %zu-token conversation%s from %s (%.1f s); %zu new tokens to read",
                      len, part.c_str(), ram ? "RAM" : "the folder",
                      std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count(),
                      prompt.size() - len);
        note(msg);
        return len;
    };
    // SHARED STARTS (#607): the state at the end of a system prompt and its tools (the shared start of a prompt read from
    // token 0) is also kept apart from its conversation, in RAM, so a NEW conversation behind the same start - after an
    // unrelated one, with or without STRATA_DS4_SLOT_KEEP - takes it back and reads only the rest.  It is the state the read
    // itself keeps at that point (a whole read up to it).  STRATA_DS4_PREFIX_KEEP (2) starts at most, the least recently
    // used giving way, STRATA_DS4_PREFIX_GB (2) their bytes at most, STRATA_DS4_PREFIX_MIN (1024) the shortest start kept;
    // STRATA_DS4_PREFIX_KEEP=0: none.  Starts with pictures are not kept.
    struct Prefix {
        std::vector<int32_t> tokens;
        std::vector<uint8_t> data;
        uint64_t used = 0;
    };
    std::vector<Prefix> prefixes;
    uint64_t prefix_clock = 0;
    const int prefix_max = m.marks_cap() > 0 ? std::max(0, (int) env_num("STRATA_DS4_PREFIX_KEEP", 2)) : 0;
    const uint64_t prefix_budget = (uint64_t) (std::max(0.0, env_num("STRATA_DS4_PREFIX_GB", 2)) * 1073741824.0);
    const size_t prefix_min = (size_t) std::max(1.0, env_num("STRATA_DS4_PREFIX_MIN", 1024));
    // the longest kept start the prompt begins with, past `floor` tokens and short of its last token (none: -1)
    const auto prefix_find = [&](const std::vector<int>& prompt, const std::vector<Pic>& want, size_t floor) -> int {
        int best = -1;
        for (size_t i = 0; i < prefixes.size(); ++i) {
            const auto& t = prefixes[i].tokens;
            if (t.size() <= floor || t.size() + 1 > prompt.size() ||
                (best >= 0 && t.size() <= prefixes[(size_t) best].tokens.size()))
                continue;
            if (!std::equal(t.begin(), t.end(), prompt.begin())) continue;
            bool pic = false;   // (a picture inside the start: its KV needs that picture - none is kept)
            for (const Pic& q : want) pic = pic || (size_t) q.start < t.size();
            if (!pic) best = (int) i;
        }
        return best;
    };
    // the state as the read keeps it at a shared start: kept apart (a copy of it), unless it already is
    // (at > 0: the state as it stood at that restore point of the read, which went on past it - no stop at the start)
    const auto prefix_keep = [&](size_t at = 0) {
        const std::vector<int32_t> t = at > 0 ? std::vector<int32_t>(m.tokens().begin(), m.tokens().begin() + (long) at)
                                              : m.tokens();
        if (prefix_max <= 0 || t.size() < prefix_min) return;
        for (Prefix& p : prefixes)
            if (p.tokens == t) {
                p.used = ++prefix_clock;
                return;
            }
        const auto t0 = std::chrono::steady_clock::now();
        Prefix p;
        std::string e;
        const uint64_t bytes = m.slot_save(p.data, e, at > 0 ? (int64_t) at : -1);
        if (bytes == 0) {
            std::fprintf(stderr, "ds4: keeping the shared start failed (%s)\n", e.c_str());
            return;
        }
        if (bytes > prefix_budget) return;
        p.tokens = t;
        p.used = ++prefix_clock;
        prefixes.push_back(std::move(p));
        uint64_t total = 0;
        for (const Prefix& x : prefixes) total += x.data.size();
        while (prefixes.size() > 1 && ((int) prefixes.size() > prefix_max || total > prefix_budget)) {
            size_t lru = 0;
            for (size_t i = 1; i < prefixes.size(); ++i)
                if (prefixes[i].used < prefixes[lru].used) lru = i;
            total -= prefixes[lru].data.size();
            prefixes.erase(prefixes.begin() + (long) lru);
        }
        std::fprintf(stderr, "ds4: kept a %zu-token shared start (%s in RAM in %.2f s); %zu kept, %s\n", t.size(),
                     gb(bytes).c_str(), std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count(),
                     prefixes.size(), gb(total).c_str());
    };
    // kept start i back in the model (false: it did not load - the state is reset, and the start is dropped)
    const auto prefix_take = [&](int i) -> bool {
        Prefix& p = prefixes[(size_t) i];
        const auto t0 = std::chrono::steady_clock::now();
        std::string e;
        if (!m.slot_load(p.data.data(), p.data.size(), (int64_t) p.tokens.size(), e) ||
            (size_t) m.pos() != p.tokens.size()) {
            std::fprintf(stderr, "ds4: taking a shared start back failed (%s) - reading it again\n", e.c_str());
            prefixes.erase(prefixes.begin() + i);
            m.reset();
            return false;
        }
        p.used = ++prefix_clock;
        std::fprintf(stderr, "ds4: took back a %zu-token shared start (%.2f s)\n", p.tokens.size(),
                     std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
        return true;
    };
    if (slot_max > 0) {
        slot_dir = std::getenv("STRATA_DS4_SLOT_DIR") ? std::filesystem::path(std::getenv("STRATA_DS4_SLOT_DIR"))
                                                      : std::filesystem::path(pack) / "slots";
        std::error_code ec;
        std::filesystem::create_directories(slot_dir, ec);
        if (!std::filesystem::is_directory(slot_dir, ec)) {
            std::fprintf(stderr, "ds4 slots: %s is not writable - conversations are not set aside\n",
                         slot_dir.string().c_str());
            slot_max = 0;
        } else {
            struct Found {
                std::string bin;
                std::vector<int32_t> toks;
                uint64_t bytes;
                std::vector<int64_t> cks;
                fs::file_time_type t;
            };
            std::vector<Found> found;
            size_t refused = 0;
            for (const auto& de : std::filesystem::directory_iterator(slot_dir, ec)) {   // a previous run's slots
                const std::string fn = de.path().filename().string(), p = de.path().string();
                if (fn.rfind("slot-", 0) != 0) continue;
                const bool is_bin = fn.size() > 9 && fn.compare(fn.size() - 4, 4, ".bin") == 0;
                const bool is_meta = fn.size() > 10 && fn.compare(fn.size() - 5, 5, ".meta") == 0;
                if (!is_bin && !is_meta) {   // a .tmp an interrupted write left
                    std::filesystem::remove(de.path(), ec);
                    continue;
                }
                if (!is_meta) continue;      // a .bin is judged by its .meta
                const std::string bin = p.substr(0, p.size() - 5) + ".bin";
                Found fd{bin, {}, 0, {}, fs::last_write_time(de.path(), ec)};
                const uint64_t have = fs::exists(bin, ec) ? (uint64_t) fs::file_size(bin, ec) : 0;
                if (meta_read(p, fd.toks, fd.bytes, fd.cks) && have == fd.bytes && fd.bytes > 0 && slot_age_days(p) <= slot_days)
                    found.push_back(std::move(fd));
                else {
                    ++refused;
                    std::filesystem::remove(p, ec);
                    std::filesystem::remove(bin, ec);
                }
            }
            for (const auto& de : std::filesystem::directory_iterator(slot_dir, ec)) {   // a .bin without its .meta
                const std::string p = de.path().string(), fn = de.path().filename().string();
                if (fn.rfind("slot-", 0) == 0 && fn.size() > 9 && fn.compare(fn.size() - 4, 4, ".bin") == 0 &&
                    !fs::exists(meta_of(p), ec))
                    std::filesystem::remove(de.path(), ec);
            }
            std::sort(found.begin(), found.end(), [](const Found& a, const Found& b) { return a.t < b.t; });
            uint64_t total = 0;
            for (auto& fd : found) {
                total += fd.bytes;
                slots.push_back({std::move(fd.toks), fd.bin, fd.bytes, ++slot_clock, nullptr, std::move(fd.cks)});
            }
            std::string msg = "folder " + slot_dir.string() + ": " + std::to_string(slots.size()) +
                              " conversation(s) kept from earlier runs (" + gb(total) + ")";
            if (refused) msg += ", " + std::to_string(refused) + " not this model's or incomplete - removed";
            note(msg);
            slot_trim("");   // (the folder over STRATA_DS4_SLOT_GB: the least recently used conversation goes first)
        }
    }
    if (slot_max > 0) {
#if !defined(_WIN32)
        // (the server ends the engine with QUIT and waits for it; a SIGTERM to the whole process group - systemd
        // stopping a service, a supervisor killing the server's group - must not end it mid-write, and a server that
        // is already gone must not end it with SIGPIPE when it writes its last lines)
        std::signal(SIGTERM, SIG_IGN);
        std::signal(SIGINT, SIG_IGN);
        std::signal(SIGPIPE, SIG_IGN);
#endif
    }
    // WP-8's option (1): the post-prompt cache settle runs in the gaps between requests and gives way the moment one arrives
    // (DS4_SETTLE_IDLE=0 turns it off).  An agent's turn ends, its tool runs, and the swaps the next answer wants go in during
    // that idle - which is what the 800 ms prompt-side budget cannot do for a run that has no next request.
    std::atomic<bool> settle_cancel{false};
    std::thread settle_th;
    const bool settle_idle_on = [] {
        const char* v = std::getenv("DS4_SETTLE_IDLE");
        return v == nullptr || std::atoi(v) != 0;
    }();
    const auto settle_stop = [&] {
        if (!settle_th.joinable()) return;
        settle_cancel.store(true, std::memory_order_release);
        settle_th.join();
    };
    const auto settle_start = [&] {
        if (!settle_idle_on || settle_th.joinable()) return;
        settle_cancel.store(false, std::memory_order_release);
        settle_th = std::thread([&m, &settle_cancel] { m.settle_idle(settle_cancel); });
    };
    // STAT for the Monitor over a decode so far (from t1, the counters s_dec and the storage's io_dec then): the GLM
    // engine's keys (speed, the VRAM hit, promotions a token) and DeepSeek's own - experts a token on the CPU lane, MB
    // a token the storage read (Linux), the n-gram memory's (Engram) read time a token, the drafts' acceptance
    const auto stat_text = [&](double t1, const strata::ds4::Stats& s_dec, long long io_dec, long long produced,
                               long long d_off, long long d_acc) -> std::string {
        const strata::ds4::Stats& s = m.stats();
        const double ms = now_ms() - t1;
        const double fed = (double) std::max<int64_t>(1, s.tokens - s_dec.tokens);
        const double ex = (double) (s.experts - s_dec.experts), cx = (double) (s.cpu_experts - s_dec.cpu_experts),
                     sx = (double) (s.stream_experts - s_dec.stream_experts);
        std::string out = "STAT";
        char b[160];
        std::snprintf(b, sizeof b, " tok_s=%.2f ms_tok=%.2f vram_hit=%.4f cpu=%.3f ram=%.3f promo=%.3f engram_ms=%.3f",
                      ms > 0 ? 1000.0 * (double) produced / ms : 0.0, produced > 0 ? ms / (double) produced : 0.0,
                      ex > 0 ? 1.0 - (cx + sx) / ex : 1.0, cx / fed, (double) (s.ram_experts - s_dec.ram_experts) / fed,
                      (double) (s.swaps - s_dec.swaps) / fed, (s.engram_ms - s_dec.engram_ms) / fed);
        out += b;
        if (sx > 0) {   // (the PCIe lane: experts a token streamed to the GPU)
            std::snprintf(b, sizeof b, " pcie=%.3f", sx / fed);
            out += b;
        }
        const long long io = io_read_bytes();
        if (io >= 0 && io_dec >= 0) {
            std::snprintf(b, sizeof b, " disk_mb=%.3f", (double) (io - io_dec) / 1e6 / fed);
            out += b;
        }
        if (d_off > 0) {
            std::snprintf(b, sizeof b, " drafts=%.3f", (double) d_acc / (double) d_off);
            out += b;
        }
        if (s.skipped > s_dec.skipped) {   // (STRATA_DS4_DISK_SKIP_W: experts left out a token)
            std::snprintf(b, sizeof b, " skipped=%.3f", (double) (s.skipped - s_dec.skipped) / fed);
            out += b;
        }
        if (s.skip_on >= 0) {   // (... _AUTO: skipping now)
            std::snprintf(b, sizeof b, " skip_on=%d", s.skip_on);
            out += b;
        }
        return out;
    };
    // a request's prompt into the bound conversation (one conversation's requests and --parallel's alike): its
    // pictures read, the state it continues reused (in place, a checkpoint, or a kept conversation), the rest read.
    // Its lines carry `tag` (" <rid>" with --parallel), stop_now() is its STOP; false: it ended here (its ERR or
    // DONE written)
    struct Prompt {
        double t0 = 0;
        size_t from = 0;         // the tokens the state already held
        size_t done = 0;         // the prompt's tokens in the state so far
        size_t next_pp = 0;      // (the next PP line's)
        bool continued = false;  // (the conversation goes on - its anchor stays)
        std::vector<Pic> pics;
        std::vector<strata::ds4::MarkAt> cuts;   // the restore points the read stops at (ds4_marks.hpp)
        size_t next_cut = 0;                     // (the next of them)
        int64_t tail_at = -1;                    // (the tail point the read keeps, TailCost)
        double ms_main = 0.0, ms_tail = 0.0;
    };
    // a request's prompt in three steps (Project Maya #587): prompt_begin (its pictures read, the state it continues
    // reused - in place, a checkpoint, or a kept conversation), prompt_piece (the next tokens of the rest read: one
    // conversation reads it in one piece, --parallel with STRATA_DS4_READ_PIECES a piece at a time with the other
    // conversations' tokens between) and prompt_end (the state's checkpoints)
    const auto prompt_begin = [&](Req& r, bool geni, const std::string& tag, Prompt& P) -> bool {
        const double t0 = now_ms();
        P.t0 = t0;
        std::vector<Pic>& pics = P.pics;
        pics.clear();
        {
            std::string e;
            if (geni && img_tok < 0) e = "this engine has no vision tower, so it cannot read pictures";
            else if (geni && read_pics(r.file, pics, e))
                for (const Pic& q : pics)
                    if (q.start < 0 || (size_t) q.start >= r.ids.size() || r.ids[(size_t) q.start] != img_tok)
                        e = "a picture's span does not start at an image token of the prompt";
            if (!e.empty()) {
                std::printf("ERR%s %s\n", tag.c_str(), e.c_str());
                std::printf("DONE%s 0 %zu 0.0 0.0 cancel 0 0 0 0 0\n", tag.c_str(), r.ids.size());
                std::fflush(stdout);
                return false;
            }
        }
        if (!reuse) {   // (STRATA_DS4_REUSE=0, for A/Bs and tests: every request is a fresh read of its whole prompt)
            m.reset();
            held.clear();
        }
        const auto& st = m.tokens();
        const size_t n_state = st.size();
        size_t c = 0;
        while (c < n_state && c < r.ids.size() && st[c] == r.ids[c]) ++c;
        // the same tokens with another picture (or none) do not share the state from that picture's span on
        auto same = [](const std::vector<Pic>& v, const Pic& q) {
            for (const Pic& x : v)
                if (x.start == q.start && x.hash == q.hash) return true;
            return false;
        };
        for (const Pic& q : pics)
            if ((size_t) q.start < c && !same(held, q)) c = (size_t) q.start;
        for (const Pic& q : held)
            if ((size_t) q.start < c && !same(pics, q)) c = (size_t) q.start;
        size_t from = 0;
        bool took = false, continued = false;   // (continued: the conversation goes on - its anchor stays)
        // the conversation continues, reused in place - a conversation has to be in the model first: a fresh engine
        // (the start of a run, so every restart) holds 0 tokens, and every prompt "continues" that to token 0; taken
        // as a continuation, the take below would never run for the folder's conversations, the case they are for
        if ((n_state > 0 || slot_max <= 0) && c == n_state && n_state < r.ids.size()) {   // (slots off: as before)
            from = n_state;
            continued = true;
        }
        else {
            // CONVERSATION SLOTS: this prompt does not continue the conversation in the model - that one is set aside
            // first (another one takes the model, and its state is lost otherwise), then the kept conversation this
            // prompt does continue is taken back - if it goes further back than the checkpoints the state itself holds
            if (held.empty()) slot_put("");
            const int64_t limit = (int64_t) std::min(c, r.ids.size() - 1);
            const int64_t rs = m.restorable(limit);
            bool whole = false;
            // (a kept shared start, when it covers more than the state itself can go back to; a kept conversation only
            //  when it covers more than either)
            const int pi = prefix_find(r.ids, pics, rs > 0 ? (size_t) rs : 0);
            const size_t plen = pi >= 0 ? prefixes[(size_t) pi].tokens.size() : 0;
            const size_t back = pics.empty() ? slot_take(r.ids, std::max(rs > 0 ? (size_t) rs : 0, plen), whole) : 0;
            if (back > 0) {
                from = back;
                took = true;
                continued = whole;
            } else if (pi >= 0 && prefix_take(pi)) {
                from = plen;
                took = true;   // (the state that comes back holds no pictures)
            } else if (rs >= 0 && m.restore(limit))
                from = (size_t) m.pos();
            else m.reset();
        }
        if (took || from == 0) held.clear();   // (a slot is text only: the state that comes back holds no pictures)
        if ((int64_t) r.ids.size() + std::max(0LL, r.max_new) > m.max_context()) {
            std::printf("ERR%s the prompt (%zu tokens) and max_new %lld are more than the context %lld\n", tag.c_str(),
                        r.ids.size(), r.max_new, (long long) m.max_context());
            std::fflush(stdout);
            return false;
        }
        std::string err;
        bool ok = true;
        // the pictures whose spans the prompt reads (the rest the state holds): encoded (or the cache's), checked
        if (!pics.empty()) {
            std::vector<strata::ds4::Image> imgs;
            const double te = now_ms();
            int encoded = 0;
            for (Pic& q : pics) {
                for (const Pic& x : held)
                    if (x.start == q.start && x.hash == q.hash) q.len = x.len;
                if (q.len > 0 && (size_t) q.start + q.len <= from) continue;
                std::shared_ptr<const std::vector<uint16_t>> rows = spans.get(m, q, err);
                if (!rows) {
                    ok = false;
                    break;
                }
                ++encoded;
                q.len = rows->size() / (size_t) m.dim();
                bool fits = (size_t) q.start + q.len <= r.ids.size() - 1;
                for (size_t i = 0; fits && i < q.len; ++i) fits = r.ids[(size_t) q.start + i] == img_tok;
                if (!fits) {
                    err = "a picture's span (" + std::to_string(q.len) + " tokens at " + std::to_string(q.start) +
                          ") is not a run of image tokens in the prompt";
                    ok = false;
                    break;
                }
                imgs.push_back({q.start, rows});
            }
            if (ok) {
                m.set_images(std::move(imgs));
                std::fprintf(stderr, "maya ds4 serve: %zu pictures, %d read (%.0f ms)\n", pics.size(), encoded,
                             now_ms() - te);
            }
            for (Pic& q : pics) q.rgb = {};   // (only the start, hash and span stay)
            m.vision_idle();   // (the tower's graphics memory back before the prompt and its experts need it)
        }
        if (!ok) {
            std::fprintf(stderr, "ds4: %s\n", err.c_str());
            std::printf("ERR%s %s\n", tag.c_str(), err.c_str());
            std::printf("DONE%s 0 %zu 0.0 0.0 cancel 0 0 0 0 0\n", tag.c_str(), r.ids.size());
            std::fflush(stdout);
            m.set_images({});
            m.reset();
            held.clear();
            return false;
        }
        // the restore points the request asks for (marks= / tail= / shared= / pin=N; ds4_marks.hpp picks the ones this
        // read reaches, and leaves the prompt's last token to the usual end): the read stops at each and keeps the state
        const size_t total = r.ids.size();
        std::vector<strata::ds4::MarkAt>& cuts = P.cuts;
        cuts.clear();
        size_t& next_cut = P.next_cut;
        next_cut = 0;
        // the tail point (tail=): only where the read reaches it, and - once two reads measured it - where its run costs
        // no more than reading its tokens again (TailCost, ds4_marks.hpp; Maya #668: a 4-GPU split or experts outside
        // VRAM make the stop cost 4-9x); one in 32 is still kept to measure again
        int64_t& tail_at = P.tail_at;
        tail_at = -1;
        P.ms_main = P.ms_tail = 0.0;
        if (m.marks_cap() > 0) {
            std::vector<int64_t> asked = r.marks;
            if (r.tail > (long long) from && r.tail < (long long) total - 1) {
                const bool was = tail_cost.costly();
                if (!tail_auto || tail_cost.keep()) {
                    asked.push_back(r.tail);
                    tail_at = r.tail;
                }
                static bool said_off = false;
                if (tail_auto && was && !said_off) {
                    std::fprintf(stderr, "ds4 serve: the tail restore point is left out on this PC: its read stop cost "
                                         "%.1fx a long read's time a token (more than reading its tokens again: the last "
                                         "tail run %zu tokens in %.0f ms, the long read %.2f ms a token); one in 32 still "
                                         "keeps it, to measure again (STRATA_DS4_RESTORE_TAIL_AUTO=0 keeps it always)\n",
                                 tail_cost.ratio, tail_cost.last_tail_tokens, tail_cost.last_tail_ms,
                                 tail_cost.last_main_ms_token);
                    said_off = true;
                }
            }
            const long long shared = r.shared >= 0 ? r.shared : (r.pin >= 256 ? r.pin : -1);
            if (!asked.empty() || shared >= 0) {
                std::vector<std::pair<int64_t, int64_t>> spans;
                for (const Pic& q : pics) spans.push_back({q.start, (int64_t) q.len});
                cuts = strata::ds4::mark_plan(asked, shared, (int64_t) from, (int64_t) total, m.marks_cap(), 16, spans);
            }
        }
        // a STOP while the prompt is read ends it there (the engine's run of chunks at a layer or chunk, windows between
        // feeds, --parallel's between pieces) and goes back to the state before this prompt - kept here first, in
        // checkpoint 1, which the prompt's own end overwrites as before - so the next request still shares the
        // conversation up to it
        if (from > 0 && from < total) m.checkpoint(1);
        if (!cuts.empty() && cuts[0].pos == (int64_t) from) m.mark(cuts[next_cut++].shared);   // (the state as it is)
        P.from = from;
        P.done = from;
        P.next_pp = from + 64;
        P.continued = continued;
        return true;
    };
    // the prompt's next `piece` tokens (or the rest): PP lines every 64 tokens, restore points kept on the way, the last
    // token with a checkpoint before it.  `yield_fn`: the model ends a run of chunks early at a chunk every stage reaches
    // once it says true (Model::set_yield) and the piece returns with the prompt not all read.  false: the request ended
    // here (its DONE written, the conversation before it kept when it can be)
    const auto prompt_piece = [&](Req& r, const std::string& tag, const std::function<bool()>& stop_now, Prompt& P,
                                  size_t piece, const std::function<bool()>& yield_fn) -> bool {
        const size_t total = r.ids.size();
        const size_t from = P.from;
        const double t0 = P.t0;
        const size_t start = P.done;
        std::vector<Pic>& pics = P.pics;
        std::vector<strata::ds4::MarkAt>& cuts = P.cuts;
        size_t& next_cut = P.next_cut;
        size_t& done = P.done;
        size_t& next_pp = P.next_pp;
        int64_t& tail_at = P.tail_at;
        double& ms_main = P.ms_main;
        double& ms_tail = P.ms_tail;
        std::string err;
        bool ok = true;
        m.set_progress([&](int64_t d) {   // (a run of chunks: each one through every layer)
            const double ms = now_ms() - t0;
            std::printf("PP%s %zu %zu %.1f %.1f %zu\n", tag.c_str(), done + (size_t) d, r.ids.size(), ms,
                        ms > 0 ? 1000.0 * (double) (done + (size_t) d - from) / ms : 0.0, from);
            std::fflush(stdout);
        });
        // (DS4_LAST_WINDOW=1: the last token in a window of its own after the checkpoint, as before - for A/B runs)
        static const bool last_window = std::getenv("DS4_LAST_WINDOW") != nullptr;
        m.set_cancel(stop_now);
        bool stopped = false;

        while (ok && done < total) {
            if (stop_now()) {
                stopped = true;
                break;
            }
            if (done - start >= piece) break;
            size_t n = std::min<size_t>({(size_t) m.feed_unit(), total - done, piece - (done - start)});
            if (next_cut < cuts.size() && (size_t) cuts[next_cut].pos < done + n) n = (size_t) cuts[next_cut].pos - done;
            // (the rest of the prompt in one read that keeps the restore points left on the way, where that costs at most
            //  a chunk's rows more than stopping at each - a run of chunks has a fixed cost of its own, the CPU lane's
            //  experts streamed for it: a point near the prompt's tail read alone cost as much as the rest; not past a
            //  shared start a read from token 0 reaches first: the read stops there and prefix_keep keeps its state)
            std::vector<std::pair<int64_t, bool>> inner;
            const size_t rest = total - done;
            // (a shared start a read from token 0 reaches first is kept from the one read too - its mark, then the state at
            //  it saved apart - unless STRATA_DS4_PREFIX_STOP=1 stops the read there as before)
            static const bool start_stop = std::getenv("STRATA_DS4_PREFIX_STOP") != nullptr;
            const bool start_kept = start_stop && prefix_max > 0 && from == 0 && next_cut == 0 && next_cut < cuts.size() &&
                                    cuts[0].shared && (size_t) cuts[0].pos >= prefix_min;
            const bool keep_start = !start_stop && prefix_max > 0 && from == 0 && next_cut == 0 && next_cut < cuts.size() &&
                                    cuts[0].shared && (size_t) cuts[0].pos >= prefix_min;
            if (next_cut < cuts.size() && !last_window && !start_kept && piece >= rest && rest <= (size_t) m.feed_unit()) {
                std::vector<int64_t> rel;
                for (size_t j = next_cut; j < cuts.size(); ++j) rel.push_back(cuts[j].pos - (int64_t) done);
                if (m.marks_inside((int) rest, rel))
                    for (size_t j = next_cut; j < cuts.size(); ++j) inner.push_back({cuts[j].pos, cuts[j].shared});
            }
            if (!inner.empty()) n = rest;
            // (the prompt's end: the last token in the same run, the checkpoint before it kept on the way)
            const bool fin = done + n == total;
            const double f0 = tail_at >= 0 ? now_ms() : 0.0;   // (TailCost: the run's wall time)
            size_t took = n;   // (fewer: a yield - the model ended the run early, the rest a later piece's)
            m.set_yield(inner.empty() ? yield_fn : nullptr);
            if (!inner.empty()) {
                ok = m.feed_marks(r.ids.data() + done, (int) n, 1, 1, inner, err);
                if (ok) next_cut = cuts.size();   // (the points the read went past: kept by it)
                if (ok && keep_start && inner.front().second && (int64_t) inner.front().first == cuts[0].pos) {
                    bool pic = false;
                    for (const Pic& q : pics) pic = pic || (int64_t) q.start < cuts[0].pos;
                    if (!pic) prefix_keep((size_t) cuts[0].pos);
                }
            } else if (fin && last_window) {
                ok = (n < 2 || m.feed(r.ids.data() + done, (int) n - 1, 0, err));
                if (ok && n >= 2 && (size_t) m.fed() < n - 1) took = (size_t) m.fed();
                else {
                    if (ok) m.checkpoint(1);
                    ok = ok && m.feed(r.ids.data() + done + n - 1, 1, 1, err);
                }
            } else {
                ok = fin ? m.feed_mark(r.ids.data() + done, (int) n, 1, 1, err)
                         : m.feed(r.ids.data() + done, (int) n, 0, err);
                if (ok) took = (size_t) m.fed();
            }
            m.set_yield(nullptr);
            if (ok && tail_at >= 0 && took == n) {
                const bool after = done >= (size_t) tail_at;
                (after ? ms_tail : ms_main) += now_ms() - f0;
                if (after && done + took == total && tail_auto) {   // (the read's end: its cost measured)
                    const bool was = tail_cost.costly(), known = tail_cost.measured();
                    if (tail_cost.add(ms_main, (size_t) tail_at - from, ms_tail, total - (size_t) tail_at) &&
                        ((was && !tail_cost.costly()) || (!known && tail_cost.measured() && !tail_cost.costly())))
                        std::fprintf(stderr, "ds4 serve: the tail restore point is kept%s: its read stop costs %.1fx a "
                                             "long read's time a token (2x or less: no more than reading its tokens "
                                             "again; the last tail run %zu tokens in %.0f ms, the long read %.2f ms a "
                                             "token)\n", was ? " again" : "", tail_cost.ratio, tail_cost.last_tail_tokens,
                                     tail_cost.last_tail_ms, tail_cost.last_main_ms_token);
                }
            }
            done += took;
            if (ok && next_cut < cuts.size() && done == (size_t) cuts[next_cut].pos) {
                const bool sh = cuts[next_cut++].shared;
                m.mark(sh);
                // (a shared start read from token 0 with no stop before it: kept apart for new conversations, #607)
                bool pic = false;
                for (const Pic& q : pics) pic = pic || (size_t) q.start < done;
                if (sh && from == 0 && next_cut == 1 && !pic) prefix_keep();
            }
            if (ok && (done >= next_pp || done == total)) {
                const double ms = now_ms() - t0;
                std::printf("PP%s %zu %zu %.1f %.1f %zu\n", tag.c_str(), done, r.ids.size(), ms,
                            ms > 0 ? 1000.0 * (double) (done - from) / ms : 0.0, from);
                std::fflush(stdout);
                next_pp = done + 64;
            }
            if (took < n) break;
        }
        m.set_cancel(nullptr);
        m.set_progress(nullptr);   // (its lambda reads this call's locals)
        if (ok && !stopped && done < total) return true;   // (a piece: more to read)
        m.set_images({});
        if (inject_fatal_now()) die_gpu_fatal("injected (DS4_INJECT_FATAL): an illegal memory access was encountered");
        if (!ok && gpu_fatal_text(err)) die_gpu_fatal(err);
        if (stopped || (!ok && m.cancelled())) {
            const bool back = from > 0 && m.restore((int64_t) from) && (size_t) m.pos() == from;
            if (!back) {
                m.reset();
                held.clear();
            }
            std::fprintf(stderr, "ds4 serve: stopped while reading the prompt of %zu tokens, after %.0f ms%s\n",
                         total, now_ms() - t0, back ? "; the conversation before it is kept" : "");
            std::printf("DONE%s 0 %zu %.1f 0.0 cancel 0 0 %zu 0 0\n", tag.c_str(), r.ids.size(), now_ms() - t0,
                        back ? from : (size_t) 0);
            std::fflush(stdout);
            return false;
        }
        if (!ok) {
            std::fprintf(stderr, "ds4: %s\n", err.c_str());
            std::printf("ERR%s %s\n", tag.c_str(), err.c_str());
            std::printf("DONE%s 0 %zu 0.0 0.0 cancel 0 0 0 0 0\n", tag.c_str(), r.ids.size());
            std::fflush(stdout);
            m.reset();
            held.clear();
            return false;
        }
        return true;
    };
    const auto prompt_end = [&](Prompt& P) {
        held = std::move(P.pics);
        m.checkpoint(0);
        if (!P.continued && slot_max > 0) m.pin();   // (a new run of prompts: where it may be rendered again from)
    };
    const auto prompt_in = [&](Req& r, bool geni, const std::string& tag, const std::function<bool()>& stop_now,
                               Prompt& P) -> bool {
        if (!prompt_begin(r, geni, tag, P) || !prompt_piece(r, tag, stop_now, P, r.ids.size(), nullptr)) return false;
        prompt_end(P);
        return true;
    };
    const bool multi = m.seq_count() > 1;
    while (!multi) {   // one conversation: a request at a time
        settle_start();   // (nothing in flight: this is the idle)
        const bool have_line = in.next(line);
        settle_stop();    // (a line arrived, or the input closed)
        if (!have_line) break;
        if (line == "QUIT") break;
        const bool geni = line.rfind("GENI ", 0) == 0;
        if (!geni && line.rfind("GEN ", 0) != 0) {
            std::printf("ERR unsupported command '%s' (GEN, GENI, STOP and QUIT)\n", line.substr(0, line.find(' ')).c_str());
            std::fflush(stdout);
            continue;
        }
        in.stop.store(false);
        Req r;
        if (!parse_gen(line, r, geni)) {
            std::printf("ERR expected: GEN <max_new> [keys] <id,id,...> or GENI <max_new> [keys] <file> <id,id,...>\n");
            std::fflush(stdout);
            continue;
        }
        m.begin_request();   // (DISK_SKIP_AUTO: between requests only)
        Prompt P;
        if (!prompt_in(r, geni, "", [&in] { return in.stop.load(); }, P)) continue;
        const double t0 = P.t0;
        const size_t from = P.from;
        std::string err;
        const double t1 = now_ms();
        long long produced = 0;
        const char* finish = "length";
        const long long limit = r.max_new > 0 ? r.max_new : m.max_context() - (long long) r.ids.size();
        long long d_off = 0, d_acc = 0;
        bool cool = false;
        if (dump_prefix != nullptr) {
            if (dump_f != nullptr) std::fclose(dump_f);
            dump_f = std::fopen((std::string(dump_prefix) + "." + std::to_string(++n_req) + ".f32").c_str(), "wb");
            dump_rows();
        }
        // this request's penalties (a repeat penalty of 0 or below, or a value that is not a number: neutral) for every
        // pick of its answer, the drafts' verify included
        m.set_penalties(r.pen_n < 0 ? INT_MAX : r.pen_n,
                        std::isfinite(r.pen_repeat) && r.pen_repeat > 0.0f ? r.pen_repeat : 1.0f,
                        std::isfinite(r.pen_freq) ? r.pen_freq : 0.0f, std::isfinite(r.pen_present) ? r.pen_present : 0.0f);
        int tok = m.sample(r.temperature, r.top_p, r.top_k, r.min_p, r.seed, counter++);
        std::vector<int> got;
        // STAT for the Monitor every 8 tokens and at the end, over this answer so far: the GLM engine's keys (speed, the
        // VRAM hit, promotions a token) and DeepSeek's own - experts a token on the CPU lane, MB a token the storage
        // read (Linux), the n-gram memory's (Engram) read time a token, the drafts' acceptance
        const strata::ds4::Stats s_dec = m.stats();
        const long long io_dec = io_read_bytes();
        auto stat_line = [&]() {
            std::printf("%s\n", stat_text(t1, s_dec, io_dec, produced, d_off, d_acc).c_str());
        };
        bool nan_stop = false;
        // test only: STRATA_TEST_NAN_STOP_AT=<n> takes the NaN stop (the same ERR, reset and state drop) once, on the n-th request
        // this engine serves, at its first poll.  Unset (or 0) it does nothing at all.
        static const long test_nan_at = [] { const char* v = std::getenv("STRATA_TEST_NAN_STOP_AT"); return v != nullptr ? std::atol(v) : 0L; }();
        static long served = 0;
        const bool test_nan = test_nan_at > 0 && ++served == test_nan_at;
        for (;;) {
            if (in.stop.load()) {
                finish = "cancel";
                break;
            }
            // (the router had no pick to make: every score it met was NaN, so the model's numbers are gone and what follows is
            //  nonsense: said once in the log by the model, here the request ends with an error instead of streaming it.  Polled at
            //  the first token and every 8 - one small copy from the device, not a per-token sync)
            if ((produced == 0 || produced % 8 == 0) && ((test_nan && produced == 0) || m.router_nonfinite())) {
                std::printf("ERR the model's numbers went non-finite (NaN reached the router), so the answer would be nonsense: the request is stopped and the conversation dropped\n");
                std::fflush(stdout);
                nan_stop = true;
                finish = "cancel";
                break;
            }
            if (tok < 0) {   // (a sample's CUDA error, printed by the sampler: no T -1 to the client; Project Maya #619)
                std::fprintf(stderr, "ds4 serve: the sample failed\n");
                finish = "cancel";
                break;
            }
            std::printf("T %d\n", tok);
            ++produced;
            if (produced % 8 == 0) stat_line();
            std::fflush(stdout);
            bool end = false;
            for (int e : eos) end = end || tok == e;
            if (end) {
                finish = "stop";
                break;
            }
            if (produced >= limit || m.pos() >= m.max_context()) break;
            if (got.empty()) {   // the next round: the drafts (if any) and the token after them
                const int room = (int) std::min<long long>(limit - produced, m.max_context() - m.pos());
                int off = 0, acc = 0;
                // a round whose drafts were all below the confidence floor: the next one does without the drafter
                // (a request's spec / dspark_conf keys over the engine's: the speed calibration measures both ways)
                const bool try_spec = (r.spec < 0 ? spec : r.spec != 0) && !cool;
                if (!round(m, tok, room, try_spec, r.conf >= 0 ? r.conf : conf_min, r.temperature, r.top_p, r.top_k,
                           r.min_p, r.seed, counter, got,
                           off, acc, err, r.cum)) {
                    if (gpu_fatal_text(err)) die_gpu_fatal(err);
                    std::fprintf(stderr, "ds4: %s\n", err.c_str());
                    finish = "cancel";
                    break;
                }
                d_off += off;
                d_acc += acc;
                cool = try_spec && off == 0;
                std::reverse(got.begin(), got.end());
                dump_rows();
            }
            tok = got.back();
            got.pop_back();
        }
        if (dump_f != nullptr) std::fflush(dump_f);
        // (a stop within accepted drafts leaves the state past it: the next prompt shares less with it, and goes back to
        //  the checkpoint at this prompt's end)
        stat_line();
        const double t2 = now_ms();
        // (a request that ended on an ERR line - the NaN stop above - is over with that line: a DONE after it would be read by the
        //  next request as its own, and every answer after it would arrive one request late)
        if (!nan_stop) {
            std::printf("DONE %lld %zu %.1f %.1f %s %lld %lld %zu 0 0\n", produced, r.ids.size(), t1 - t0, t2 - t1, finish,
                        d_acc, d_off, from);
            std::fflush(stdout);
        }
        const auto& s = m.stats();
        std::fprintf(stderr,
                     "maya ds4 serve: prompt %zu tokens = %zu reused + %zu read in %.0f ms (%.1f tok/s), %lld generated "
                     "in %.0f ms (%.2f tok/s); CPU lane %.1f%% of the experts\n",
                     r.ids.size(), from, r.ids.size() - from, t1 - t0,
                     t1 > t0 ? 1000.0 * (double) (r.ids.size() - from) / (t1 - t0) : 0.0, produced, t2 - t1,
                     t2 > t1 ? 1000.0 * (double) produced / (t2 - t1) : 0.0,
                     s.experts ? 100.0 * (double) s.cpu_experts / (double) s.experts : 0.0);
        if (d_off > 0) std::fprintf(stderr, "maya ds4 serve: drafts %lld of %lld kept; %s\n", d_acc, d_off,
                                    per_position(s_dec, s).c_str());
        if (nan_stop) {   // (its state holds the NaNs: the next request starts from nothing)
            m.reset();
            held.clear();
        }
        m.save_profile();
    }
    // ---- SEVERAL CONVERSATIONS AT ONCE (--parallel <n>: READY ... multi=<n>).  GEN lines carry rid=<id>, and every
    // line a request causes carries it after its keyword (T <rid> <id>, PP <rid> ..., DONE <rid> ..., ERR <rid> ...);
    // STOP <rid> ends that request, a bare STOP every one.  Each request runs in one of the n conversations the model
    // holds (Model::seq_bind): the idle one whose state its prompt reuses the most, else an empty one, else the least
    // recently used (whose conversation is set aside first with STRATA_DS4_SLOT_KEEP, as one conversation's is).  A new
    // request's prompt - its pictures too - is read while the others wait; then each running request writes a token a
    // step.  A request alone decodes in its rounds as with one conversation (DSpark's drafts); two or more go through
    // Model::step_multi - a token of each through the model together, no drafts.
    if (multi) {
        struct Act {
            Req r;
            Prompt P;
            int seq = 0;
            double t1 = 0;
            long long produced = 0, limit = 0, d_off = 0, d_acc = 0;
            const char* finish = "length";
            bool cool = false, done = false;
            bool reading = false;   // its prompt is being read, a piece a turn
            bool ended = false;     // it ended while its prompt was read (its DONE written)
            unsigned long long counter = 0;   // (its own: its picks the same alone or beside others)
            int tok = -1;                     // written next, not fed yet
            std::vector<int> got;             // a round's tokens still to write, the last first
        };
        const int nseq = m.seq_count();
        std::vector<long long> seq_rid((size_t) nseq, -1);   // the request in each conversation (-1: idle)
        std::vector<uint64_t> seq_used((size_t) nseq, 0);    // (the least recently used goes first)
        std::vector<std::vector<Pic>> held_of((size_t) nseq);   // the pictures each one holds (`held`: the bound one's)
        uint64_t use_clock = 0;
        std::vector<std::unique_ptr<Act>> acts;
        std::deque<std::string> waiting;   // GEN lines no conversation is free for yet
        std::deque<Act*> read_line;        // the prompts being read, in turn: a piece each, then to the back
        // STRATA_DS4_READ_PIECES=1 (Project Maya #587, default off): the prompts being read take turns, a piece each (a chunk
        // per pipeline stage: the split's stages all busy), and every running request writes a token between two
        // pieces - a long prompt no longer holds every answer, and a prompt whole in one piece goes ahead of the longer
        // reads.  A prompt read alone goes in one run until a line comes (Model::set_yield).  Off: each prompt is read
        // whole as it arrives, the others waiting.  STRATA_DS4_READ_PIECE=<chunks> a piece is that many chunks a
        // stage (1); STRATA_DS4_READ_ROUNDS=<n> the answers write n tokens each between two pieces (1)
        const auto env_count = [](const char* name, int dflt) {
            const char* v = std::getenv(name);
            const int x = v != nullptr ? std::atoi(v) : 0;
            return x > 0 ? x : dflt;
        };
        const bool read_pieces = env_count("STRATA_DS4_READ_PIECES", 0) > 0 && m.prompt_chunk() > 0;
        const size_t read_piece = m.prompt_chunk() > 0 ? (size_t) m.prompt_chunk() * (size_t) m.pipeline_stages() *
                                                             (size_t) env_count("STRATA_DS4_READ_PIECE", 1)
                                                       : (size_t) m.feed_unit();
        const int read_rounds = env_count("STRATA_DS4_READ_ROUNDS", 1);
        // the reader's lines posted and taken: one waiting, a prompt read alone yields to it
        std::atomic<long long> lines_posted{0};
        long long lines_taken = 0;
        // the Monitor's counters: over everything written since the engine was last idle
        double busy_t1 = 0;
        long long busy_n = 0, busy_io = -1;
        strata::ds4::Stats busy_s;
        std::string err;
        std::fprintf(stderr, "maya ds4 serve: %d conversations at once (--parallel): GEN needs rid=<id>; STOP <rid>\n",
                     nseq);
        const auto bind = [&](int s) {
            const int b = m.seq_bound();
            if (s == b) return;
            held_of[(size_t) b] = std::move(held);
            m.seq_bind(s);
            held = std::move(held_of[(size_t) s]);
            held_of[(size_t) s].clear();
        };
        const auto penalties = [&](const Req& r) {
            m.set_penalties(r.pen_n < 0 ? INT_MAX : r.pen_n,
                            std::isfinite(r.pen_repeat) && r.pen_repeat > 0.0f ? r.pen_repeat : 1.0f,
                            std::isfinite(r.pen_freq) ? r.pen_freq : 0.0f,
                            std::isfinite(r.pen_present) ? r.pen_present : 0.0f);
        };
        const auto release = [&](const Act& a) {   // (its conversation free for the next request)
            seq_rid[(size_t) a.seq] = -1;
            seq_used[(size_t) a.seq] = ++use_clock;
            in.forget(a.r.rid);
        };
        const auto finish_act = [&](Act& a) {
            const double t2 = now_ms();
            std::printf("DONE %lld %lld %zu %.1f %.1f %s %lld %lld %zu 0 0\n", a.r.rid, a.produced, a.r.ids.size(),
                        a.t1 - a.P.t0, t2 - a.t1, a.finish, a.d_acc, a.d_off, a.P.from);
            std::fflush(stdout);
            std::fprintf(stderr,
                         "maya ds4 serve: request %lld: prompt %zu tokens = %zu reused + %zu read in %.0f ms (%.1f tok/s), "
                         "%lld generated in %.0f ms (%.2f tok/s, %zu conversations running; drafts %lld of %lld kept)\n",
                         a.r.rid, a.r.ids.size(), a.P.from, a.r.ids.size() - a.P.from, a.t1 - a.P.t0,
                         a.t1 > a.P.t0 ? 1000.0 * (double) (a.r.ids.size() - a.P.from) / (a.t1 - a.P.t0) : 0.0,
                         a.produced, t2 - a.t1, t2 > a.t1 ? 1000.0 * (double) a.produced / (t2 - a.t1) : 0.0,
                         acts.size() + 1, a.d_acc, a.d_off);
            release(a);
        };
        // one round of the running requests: each writes its next token, the model makes the next ones (a plain step, its rounds with
        // drafts alone, or one multi step for several)
        const auto round_turn = [&]() {
        // each running request writes its next token; those whose next one the model must make go below
        std::vector<Act*> need;
        const long long n0 = busy_n;
        size_t n_answer = 0;
        // (with several running, a round's tokens all at once: every one then needs the model in the same step, so
        //  their rounds go through it together - the drafts of each in one window; Project Maya #611)
        size_t several_n = 0;
        for (auto& ap : acts) several_n += ap->reading ? 0 : 1;
        const bool several = several_n > 1;
        for (auto& ap : acts) {
            Act& a = *ap;
            if (a.reading) continue;
            ++n_answer;
            for (;;) {
                if (in.stopped(a.r.rid)) {
                    a.finish = "cancel";
                    a.done = true;
                    break;
                }
                std::printf("T %lld %d\n", a.r.rid, a.tok);
                ++a.produced;
                ++busy_n;
                bool eos_tok = false;
                for (int e : eos) eos_tok = eos_tok || a.tok == e;
                if (eos_tok) {
                    a.finish = "stop";
                    a.done = true;
                    break;
                }
                if (a.produced >= a.limit || m.seq_pos(a.seq) >= m.max_context()) {
                    a.done = true;
                    break;
                }
                if (!a.got.empty()) {
                    a.tok = a.got.back();
                    a.got.pop_back();
                    if (several) continue;
                    break;
                }
                need.push_back(&a);
                break;
            }
        }
        if (busy_n / 8 != n0 / 8) {   // the Monitor's line, every 8 tokens of all of them
            std::string st = stat_text(busy_t1, busy_s, busy_io, busy_n, 0, 0);
            st += " active=" + std::to_string(n_answer);
            std::printf("%s\n", st.c_str());
        }
        std::fflush(stdout);
        if (need.size() == 1) {   // alone (or the last of several still running): its rounds, as with one conversation
            Act& a = *need[0];
            bind(a.seq);
            const int room = (int) std::min<long long>(a.limit - a.produced, m.max_context() - m.pos());
            penalties(a.r);
            int off = 0, acc = 0;
            const bool try_spec = (a.r.spec < 0 ? spec : a.r.spec != 0) && !a.cool;
            if (!round(m, a.tok, room, try_spec, a.r.conf >= 0 ? a.r.conf : conf_min, a.r.temperature, a.r.top_p,
                       a.r.top_k, a.r.min_p, a.r.seed, a.counter, a.got, off, acc, err, a.r.cum)) {
                std::fprintf(stderr, "ds4: %s\n", err.c_str());
                a.finish = "cancel";
                a.done = true;
            } else {
                a.d_off += off;
                a.d_acc += acc;
                a.cool = try_spec && off == 0;
                std::reverse(a.got.begin(), a.got.end());
                a.tok = a.got.back();
                a.got.pop_back();
            }
        } else if (need.size() > 1) {   // several: a round of each through the model together
            // each one's drafts (DSpark, in one drafter pass where they fit one, else a conversation at a time: its
            // own drafter rings) while their confidence holds, its share of a window at most; then every one's token
            // and drafts through the model at once (step_multi_spec) and each its verdict (verify_multi).  No drafts
            // this round: a token of each
            const int nn = (int) need.size(), share = m.verify_max() / nn;
            const auto cancel_all = [&] {
                std::fprintf(stderr, "ds4: %s\n", err.c_str());
                for (Act* a : need) {
                    a->finish = "cancel";
                    a->done = true;
                }
            };
            std::vector<std::vector<int>> drafts((size_t) nn);
            std::vector<std::vector<float>> confs((size_t) nn);
            std::vector<char> tried((size_t) nn, 0);
            std::vector<int> rooms((size_t) nn, 1), dsv, dys, dix;   // (the ones that draft: conversation, y, index)
            bool any = false, failed = false;
            for (int i = 0; i < nn; ++i) {
                Act& a = *need[(size_t) i];
                int room = (int) std::min<long long>(a.limit - a.produced, m.max_context() - m.seq_pos(a.seq));
                rooms[(size_t) i] = room = std::min(room, share);
                if (!((a.r.spec < 0 ? spec : a.r.spec != 0) && !a.cool && room > 1 && m.draft_block() > 0)) continue;
                tried[(size_t) i] = 1;
                dsv.push_back(a.seq);
                dys.push_back(a.tok);
                dix.push_back(i);
            }
            // their drafts: in one drafter pass where they fit one (Model::draft_multi), else one after another (the
            // drafter's argmax either way: verify_multi takes each draft as a point)
            const int nd = (int) dsv.size();
            if (nd > 1 && m.can_draft_multi(dsv.data(), nd)) {
                std::vector<std::vector<int>> dd;
                std::vector<std::vector<float>> cc;
                failed = !m.draft_multi(dsv.data(), dys.data(), nd, dd, cc, err);
                for (int j = 0; j < nd && !failed; ++j) {
                    drafts[(size_t) dix[(size_t) j]] = std::move(dd[(size_t) j]);
                    confs[(size_t) dix[(size_t) j]] = std::move(cc[(size_t) j]);
                }
            } else
                for (int j = 0; j < nd && !failed; ++j) {
                    bind(dsv[(size_t) j]);
                    failed = !m.draft(dys[(size_t) j], drafts[(size_t) dix[(size_t) j]], confs[(size_t) dix[(size_t) j]],
                                      err);
                }
            for (int i = 0; i < nn && !failed; ++i) {
                if (!tried[(size_t) i]) continue;
                Act& a = *need[(size_t) i];
                const int room = rooms[(size_t) i];
                std::vector<int>& d = drafts[(size_t) i];
                const std::vector<float>& conf = confs[(size_t) i];
                const float cmin = a.r.conf >= 0 ? a.r.conf : conf_min;
                const float cum_min = a.r.cum >= 0 ? a.r.cum : dspark_cum_default(m);
                int keep = 0;
                float cum = 1.0f;
                while (keep < (int) d.size() && keep < room - 1 && keep < g_spec_depth && conf[(size_t) keep] >= cmin &&
                       (cum *= conf[(size_t) keep]) >= cum_min)
                    ++keep;
                d.resize((size_t) keep);
                any = any || keep > 0;
            }
            if (failed) cancel_all();
            else if (!any) {
                std::vector<int> sv, tv;
                for (Act* a : need) {
                    sv.push_back(a->seq);
                    tv.push_back(a->tok);
                }
                if (!m.step_multi(sv.data(), tv.data(), nn, err)) cancel_all();
                else
                    for (int i = 0; i < nn; ++i) {
                        Act& a = *need[(size_t) i];
                        penalties(a.r);
                        a.tok = m.sample_multi(i, a.r.temperature, a.r.top_p, a.r.top_k, a.r.min_p, a.r.seed,
                                               a.counter++);
                        a.cool = tried[(size_t) i] != 0;   // (its drafter tried and kept none: rests a round)
                    }
            } else {
                std::vector<std::vector<int>> rows((size_t) nn);
                std::vector<const int*> tp((size_t) nn);
                std::vector<int> sv((size_t) nn), lens((size_t) nn);
                for (int i = 0; i < nn; ++i) {
                    rows[(size_t) i].push_back(need[(size_t) i]->tok);
                    rows[(size_t) i].insert(rows[(size_t) i].end(), drafts[(size_t) i].begin(), drafts[(size_t) i].end());
                    tp[(size_t) i] = rows[(size_t) i].data();
                    lens[(size_t) i] = (int) rows[(size_t) i].size();
                    sv[(size_t) i] = need[(size_t) i]->seq;
                }
                if (!m.step_multi_spec(sv.data(), tp.data(), lens.data(), nn, err)) cancel_all();
                else
                    for (int i = 0; i < nn; ++i) {
                        Act& a = *need[(size_t) i];
                        const std::vector<int>& d = drafts[(size_t) i];
                        penalties(a.r);
                        int acc = 0, next = -1;
                        if (!m.verify_multi(i, d, a.r.temperature, a.r.top_p, a.r.top_k, a.r.min_p, a.r.seed, a.counter,
                                            acc, next, err)) {
                            cancel_all();
                            break;
                        }
                        a.counter += (unsigned long long) acc + 1;
                        a.d_off += (long long) d.size();
                        a.d_acc += acc;
                        a.cool = tried[(size_t) i] && d.empty();
                        a.got.assign(d.begin(), d.begin() + acc);
                        a.got.push_back(next);
                        std::reverse(a.got.begin(), a.got.end());
                        a.tok = a.got.back();
                        a.got.pop_back();
                    }
            }
        }
        };
        // its prompt read: its first token picked, and it writes from the next turn on
        const auto start_answer = [&](Act& a) {
            prompt_end(a.P);
            a.reading = false;
            a.t1 = now_ms();
            a.limit = a.r.max_new > 0 ? a.r.max_new : m.max_context() - (long long) a.r.ids.size();
            penalties(a.r);
            a.tok = m.sample(a.r.temperature, a.r.top_p, a.r.top_k, a.r.min_p, a.r.seed, a.counter++);
        };
        // the next `piece` tokens of its prompt into its conversation (a STOP ends it here, its DONE written); `yield`:
        // the piece ends early at a chunk the model reaches once a line is waiting
        const auto read_turn = [&](Act& a, size_t piece, bool yield) {
            bind(a.seq);
            const long long rid = a.r.rid;
            std::function<bool()> y;
            if (yield) y = [&] { return lines_posted.load(std::memory_order_acquire) != lines_taken; };
            const bool read = prompt_piece(a.r, " " + std::to_string(rid), [&in, rid] { return in.stopped(rid); }, a.P,
                                           piece, y);
            if (!read) {
                a.ended = a.done = true;
                return;
            }
            if (a.P.done == a.r.ids.size()) start_answer(a);
        };
        // a prompt whole in one piece joins the line ahead of the longer reads, behind the other one-piece prompts
        const auto one_piece = [&](const Act* a) { return a->r.ids.size() - a->P.done <= read_piece; };
        const auto join_read = [&](Act* a) {
            if (!one_piece(a)) {
                read_line.push_back(a);
                return;
            }
            auto at = read_line.begin();
            while (at != read_line.end() && one_piece(*at)) ++at;
            read_line.insert(at, a);
        };
        // a GEN line into a free conversation: its prompt begun, and read in turn (arrival, or with pictures: whole now)
        const auto admit = [&](const std::string& ln) {
            const bool geni = ln.rfind("GENI ", 0) == 0;
            auto a = std::make_unique<Act>();
            Req& r = a->r;
            if (!parse_gen(ln, r, geni)) {
                std::printf("ERR expected: GEN <max_new> [keys] <id,id,...> or GENI <max_new> [keys] <file> <id,id,...>\n");
                std::fflush(stdout);
                return;
            }
            if (r.rid < 0) {
                std::printf("ERR this engine runs several conversations: GEN needs rid=<id>\n");
                std::fflush(stdout);
                return;
            }
            // the conversation: the idle one the prompt reuses the most of (its whole state continued, else a
            // checkpoint it can go back to), else an empty one, else the least recently used
            int best = -1;
            int64_t best_u = -1;
            for (int s = 0; s < nseq; ++s) {
                if (seq_rid[(size_t) s] >= 0) continue;
                bind(s);
                const auto& st = m.tokens();
                size_t c = 0;
                while (c < st.size() && c < r.ids.size() && st[c] == r.ids[c]) ++c;
                int64_t u = 0;
                if (!st.empty() && c == st.size() && st.size() < r.ids.size()) u = (int64_t) st.size();
                else if (c > 0) u = std::max<int64_t>(0, m.restorable((int64_t) std::min(c, r.ids.size() - 1)));
                const bool better = best < 0 || u > best_u ||
                                    (u == best_u && (st.empty() != m.seq_tokens(best).empty()
                                                         ? st.empty()
                                                         : seq_used[(size_t) s] < seq_used[(size_t) best]));
                if (better) best = s, best_u = u;
            }
            if (best < 0) {   // (the server gives each conversation one request: a request over that waits)
                waiting.push_front(ln);
                return;
            }
            bind(best);
            seq_used[(size_t) best] = ++use_clock;
            if (acts.empty()) m.begin_request();   // (DISK_SKIP_AUTO: between requests only - none runs now)
            const long long rid = r.rid;
            if (!prompt_begin(r, geni, " " + std::to_string(rid), a->P)) {
                in.forget(rid);
                return;
            }
            a->seq = best;
            a->reading = true;
            if (!read_pieces || !a->P.pics.empty()) {   // (pictures: the prompt whole now)
                read_turn(*a, r.ids.size(), false);
                if (a->ended) {
                    in.forget(rid);
                    return;
                }
            }
            seq_rid[(size_t) best] = rid;
            if (acts.empty()) {
                busy_t1 = now_ms();
                busy_n = 0;
                busy_s = m.stats();
                busy_io = io_read_bytes();
            }
            if (a->reading) join_read(a.get());
            acts.push_back(std::move(a));
        };
        // the loop's events (ds4_evt.hpp, Project Maya #587), run one at a time on this thread: the reader's lines, then the
        // turns - a token of every answer running (a round), or the next prompt's piece, in turn.  A line outranks a
        // turn, so every request that came is admitted before the next one; nothing polls: with no request running
        // the engine waits for a line
        enum : int { kTurn = 1, kLine = 2 };
        strata::ds4::evt::Engine eng;
        bool quit = false;
        bool turn_queued = false;   // (one turn in the queue at a time: each queues the next as it ends)
        int rounds = 0;             // (the rounds since the last piece)
        const auto admit_waiting = [&] {
            while (!quit && !waiting.empty() && std::find(seq_rid.begin(), seq_rid.end(), -1LL) != seq_rid.end()) {
                const std::string g = waiting.front();
                waiting.pop_front();
                admit(g);
            }
        };
        // the next prompt in line reads a piece (its first token picked once it is all read), then goes to the back.
        // Alone - no other prompt read, no answer running - the rest of it in one run, the pipeline full throughout,
        // until a line comes (a request: the run ends at the next chunk all stages reach, and pieces again)
        const auto read_turn_next = [&](bool answering) {
            Act* a = read_line.front();
            read_line.pop_front();
            const bool alone = read_line.empty() && !answering;
            if (!a->done) read_turn(*a, alone ? a->r.ids.size() : read_piece, alone);
            if (a->reading && !a->done) read_line.push_back(a);
        };
        std::function<void()> settle;
        // the next turn: a piece once the rounds since the last one are due while a prompt is being read, else a round
        // while an answer runs
        const auto queue_turn = [&] {
            if (turn_queued || acts.empty()) return;
            turn_queued = true;
            eng.add([&] {
                turn_queued = false;
                const bool answering = std::any_of(acts.begin(), acts.end(),
                                                   [](const std::unique_ptr<Act>& a) { return !a->reading; });
                if (!read_line.empty() && (!answering || rounds >= read_rounds)) {
                    read_turn_next(answering);
                    rounds = 0;
                } else if (answering) {
                    round_turn();
                    ++rounds;
                }
                settle();
            }, kTurn);
        };
        // after every event: a STOP of a prompt waiting for its turn (ended now, the conversation before it kept), the
        // requests ended (their conversations to the requests waiting), then the next turn - or, none running, idle
        settle = [&] {
            for (auto& ap : acts)
                if (ap->reading && !ap->done && in.stopped(ap->r.rid)) read_turn(*ap, 0, false);
            bool freed = false;
            for (size_t i = 0; i < acts.size();) {
                if (!acts[i]->done) {
                    ++i;
                    continue;
                }
                std::unique_ptr<Act> a = std::move(acts[i]);
                acts.erase(acts.begin() + (long) i);
                read_line.erase(std::remove(read_line.begin(), read_line.end(), a.get()), read_line.end());
                if (a->ended) release(*a);   // (its DONE written as its prompt was read)
                else finish_act(*a);
                freed = true;
            }
            if (freed) admit_waiting();
            if (!acts.empty()) {
                queue_turn();
                return;
            }
            if (quit) {
                eng.stop();
                return;
            }
            if (waiting.empty()) {
                m.save_profile();
                settle_start();   // (nothing in flight: this is the idle)
            }
        };
        // a line: a request (admitted when a conversation is free), QUIT (every request stopped, the engine's end once
        // they are), anything else answered at once (STOP <rid> is the reader's)
        const auto on_line = [&](const std::string& ln) {
            ++lines_taken;
            if (ln == "QUIT") {
                quit = true;
                in.stop_all.store(true);
            } else if (ln.rfind("GEN ", 0) == 0 || ln.rfind("GENI ", 0) == 0) {
                settle_stop();   // (a request: the idle ends)
                waiting.push_back(ln);
                admit_waiting();
            } else {
                std::printf("ERR unsupported command '%s' (GEN, GENI, STOP and QUIT)\n",
                            ln.substr(0, ln.find(' ')).c_str());
                std::fflush(stdout);
            }
            settle();
        };
        settle_start();   // (nothing in flight: this is the idle)
        std::thread reader([&] {   // (the reader's lines into the loop; the input's end is a QUIT)
            std::string l;
            bool quit_seen = false;
            while (!quit_seen && in.next(l)) {
                quit_seen = l == "QUIT";
                lines_posted.fetch_add(1, std::memory_order_release);
                eng.add([&, l] { on_line(l); }, kLine);
            }
            if (!quit_seen) {
                lines_posted.fetch_add(1, std::memory_order_release);
                eng.add([&] { on_line("QUIT"); }, kLine);
            }
        });
        eng.run();
        reader.join();
        settle_stop();
    }
    // the engine ends (QUIT or the end of its input): the conversation in the model is set aside, and every one in
    // RAM is written to the folder, so the next run reads on from them
    if (slot_max > 0) {
        if (held.empty()) slot_put(" as the engine ended");
        size_t written = 0;
        while (true) {   // (a write can take the folder's least recently used conversation away: the list is asked again)
            size_t i = SIZE_MAX;
            for (size_t k = 0; k < slots.size(); ++k)
                if (slots[k].data) {
                    i = k;
                    break;
                }
            if (i == SIZE_MAX) break;
            if (slot_write_file(slots[i])) ++written;
            else slot_drop(i);   // (its state could not be written: there is nothing more to do at the engine's end)
        }
        if (written > 0) {
            int ram = 0, disk = 0;
            slot_count(ram, disk);
            char msg[256];
            std::snprintf(msg, sizeof msg, "wrote %zu conversation(s) to %s; %d there in all", written,
                          slot_dir.string().c_str(), disk);
            note(msg);
        }
    }
    settle_stop();   // (the idle thread must be off the cache before the profile is saved and the engine ends)
    m.save_profile();
    return 0;
}


// ---- Strata addition: teacher-forced scoring (the quality gate) -------------------------------------------------
// --score IDS.json --score-out RUN.tsv: every segment of a tools/quant_quality.py ids file (name + ids) is read through
// the model from a fresh state; per position the log-probability of the text's next token and the 64 most likely
// tokens go out in the STRATA_LOGPOS row format that `quant_quality.py compare` reads (natural logs, full-vocabulary
// log-softmax in double).  RUN.tsv.seg.json lists each segment's first line and size.
struct ScoreSeg {
    std::string name;
    std::vector<int> ids;
};

bool read_score_ids(const std::string& path, std::vector<ScoreSeg>& segs, std::string& err, int vocab = 0) {
    std::FILE* f = std::fopen(path.c_str(), "rb");
    if (f == nullptr) {
        err = path + ": cannot open";
        return false;
    }
    std::string s;
    char buf[1 << 16];
    size_t k;
    while ((k = std::fread(buf, 1, sizeof buf, f)) > 0) s.append(buf, k);
    std::fclose(f);
    size_t p = 0;
    for (;;) {
        const size_t n0 = s.find("\"name\"", p);
        if (n0 == std::string::npos) break;
        const size_t q0 = s.find('"', s.find(':', n0) + 1);
        const size_t q1 = s.find('"', q0 + 1);
        const size_t i0 = s.find("\"ids\"", q1);
        if (q0 == std::string::npos || q1 == std::string::npos || i0 == std::string::npos) break;
        const size_t b0 = s.find('[', i0), b1 = s.find(']', b0);
        if (b0 == std::string::npos || b1 == std::string::npos) break;
        ScoreSeg g;
        g.name = s.substr(q0 + 1, q1 - q0 - 1);
        g.ids = parse_ids(s.substr(b0 + 1, b1 - b0 - 1));
        segs.push_back(std::move(g));
        p = b1;
    }
    if (segs.empty()) {
        err = path + ": no segments (expected quant_quality.py's ids file)";
        return false;
    }
    // an ids file made with another model's tokenizer (the committed data/quant_quality/v1/set.ids.json is Qwen's: ids up to 248,069)
    // reads as nonsense here, or fails deep in a run: refuse it up front, naming the segment (the model's vocabulary: m.vocab())
    for (const ScoreSeg& g : segs)
        for (size_t i = 0; vocab > 0 && i < g.ids.size(); ++i)
            if (g.ids[i] < 0 || g.ids[i] >= vocab) {
                err = path + ": segment '" + g.name + "' has token id " + std::to_string(g.ids[i]) + " at position " + std::to_string(i) +
                      ", outside this model's vocabulary (0.." + std::to_string(vocab - 1) +
                      "): the ids file was made with another tokenizer (DeepSeek ids: tools/quant_quality.py ids with the pack's tokenizer folder)";
                return false;
            }
    return true;
}

int score(strata::ds4::Model& m, const std::string& ids_path, const std::string& out_path, int window, int eos) {
    std::vector<ScoreSeg> segs;
    std::string err;
    if (!read_score_ids(ids_path, segs, err, m.vocab())) {
        std::fprintf(stderr, "strata-dsv41: %s\n", err.c_str());
        return 1;
    }
    // DS4_SCORE_PREFIX=N (a test of the long-prompt layer skip): the segments are joined into one stream, its first N
    // tokens are read as one prompt (logits of the last row only: the path a long prompt takes, where the skip applies)
    // and only the positions after them are scored, in windows
    const long long prefix_n = std::getenv("DS4_SCORE_PREFIX") ? std::atoll(std::getenv("DS4_SCORE_PREFIX")) : 0;
    if (prefix_n > 0 && segs.size() > 1) {
        ScoreSeg all;
        all.name = "concat";
        for (const ScoreSeg& sg : segs) all.ids.insert(all.ids.end(), sg.ids.begin(), sg.ids.end());
        segs.assign(1, all);
    }
    std::FILE* out = std::fopen(out_path.c_str(), "wb");
    std::FILE* sj = std::fopen((out_path + ".seg.json").c_str(), "wb");
    if (out == nullptr || sj == nullptr) {
        std::fprintf(stderr, "strata-dsv41: cannot write %s\n", out_path.c_str());
        return 1;
    }
    std::fprintf(sj, "{\"segments\": [");
    const int V = m.vocab();
    long long line = 0;
    std::vector<float> lg;
    std::vector<int> order;
    const double t0 = now_ms();
    for (size_t si = 0; si < segs.size(); ++si) {
        const std::vector<int>& ids = segs[si].ids;
        m.reset();
        const long long first = line;
        double sum_nll = 0.0;
        long long top1 = 0, cnt = 0;
        size_t a0 = 0;
        if (prefix_n > 0 && (size_t) prefix_n + 1 < ids.size()) {
            a0 = (size_t) prefix_n;
            if (!m.feed(ids.data(), (int) a0, 1, err)) {
                std::fprintf(stderr, "strata-dsv41: %s\n", err.c_str());
                return 1;
            }
        }
        for (size_t a = a0; a + 1 < ids.size();) {
            const int T = (int) std::min<size_t>((size_t) window, ids.size() - 1 - a);   // (positions a .. a+T-1 are scored)
            if (!m.feed(ids.data() + a, T, T, err) || !m.logits(lg) || lg.size() < (size_t) T * (size_t) V) {
                std::fprintf(stderr, "strata-dsv41: %s\n", err.empty() ? "no logits" : err.c_str());
                return 1;
            }
            for (int t = 0; t < T; ++t) {
                const float* r = lg.data() + (size_t) t * (size_t) V;
                float mx = r[0];
                for (int v = 1; v < V; ++v) mx = std::max(mx, r[v]);
                double z = 0.0;
                for (int v = 0; v < V; ++v) z += std::exp((double) (r[v] - mx));
                const double lz = std::log(z) + (double) mx;
                order.resize((size_t) V);
                std::iota(order.begin(), order.end(), 0);
                std::partial_sort(order.begin(), order.begin() + 64, order.end(), [&](int x, int y) {
                    return r[x] > r[y] || (r[x] == r[y] && x < y);
                });
                const int tgt = ids[a + (size_t) t + 1];
                const double lt = (double) r[tgt] - lz;
                const double la = (double) r[order[0]] - lz;
                const double le = eos >= 0 && eos < V ? (double) r[eos] - lz : -1e30;
                // logp(target) with the end token left out of the distribution, as the engine's row has it
                const double lt_ne = tgt == eos ? lt : std::log(std::max(1e-300, std::exp(lt) / std::max(1e-300, 1.0 - std::exp(le))));
                std::fprintf(out, "%zu\t%d\t%.6f\t%d\t%.6f\t%d\t%.6f\t%.6f", a + (size_t) t, tgt, lt, order[0], la,
                             order[0] == tgt ? 1 : 0, le, lt_ne);
                for (int j = 0; j < 64; ++j) std::fprintf(out, "\t%d:%.6f", order[(size_t) j], (double) r[order[(size_t) j]] - lz);
                std::fprintf(out, "\n");
                ++line;
                ++cnt;
                top1 += order[0] == tgt;
                sum_nll -= lt;
            }
            a += (size_t) T;
        }
        std::fprintf(sj, "%s{\"name\": \"%s\", \"first_line\": %lld, \"end_line\": %lld, \"n\": %lld, \"ppl\": %.6f, \"top1\": %.6f}", si ? ", " : "",
                     segs[si].name.c_str(), first, first + cnt, cnt, std::exp(sum_nll / (double) std::max(1LL, cnt)),
                     (double) top1 / (double) std::max(1LL, cnt));
        std::fprintf(stderr, "score %s: %lld positions, ppl %.4f, top-1 %.4f (%.0f s)\n", segs[si].name.c_str(), cnt,
                     std::exp(sum_nll / (double) std::max(1LL, cnt)), (double) top1 / (double) std::max(1LL, cnt),
                     (now_ms() - t0) / 1000.0);
        std::fflush(out);
    }
    std::fprintf(sj, "]}\n");
    std::fclose(sj);
    std::fclose(out);
    return 0;
}
// ---- Strata H100 addition: profile run -------------------------------------------------------------------------
// --profile-run IDS.json: every segment is fed through the model from a fresh state (no logits); the engine's expert
// profile (--expert-profile FILE, counts a layer and expert) and DS4_TRACE collect the routing.  The profile is saved
// every 10 segments.  --window sets the chunk (a chunked prompt path above 16).
int profile_run(strata::ds4::Model& m, const std::string& ids_path, int window) {
    std::vector<ScoreSeg> segs;
    std::string err;
    if (!read_score_ids(ids_path, segs, err, m.vocab())) {
        std::fprintf(stderr, "strata-dsv41: %s\n", err.c_str());
        return 1;
    }
    const double t0 = now_ms();
    long long tot = 0;
    for (size_t si = 0; si < segs.size(); ++si) {
        const std::vector<int>& ids = segs[si].ids;
        m.reset();
        for (size_t a = 0; a < ids.size();) {
            const int T = (int) std::min<size_t>((size_t) window, ids.size() - a);
            if (!m.feed(ids.data() + a, T, 0, err)) {
                std::fprintf(stderr, "strata-dsv41: %s\n", err.c_str());
                return 1;
            }
            a += (size_t) T;
        }
        tot += (long long) ids.size();
        std::fprintf(stderr, "profile-run %zu/%zu %s: %zu tokens, %lld in all, %.0f s, %.1f tok/s\n", si + 1, segs.size(),
                     segs[si].name.c_str(), ids.size(), tot, (now_ms() - t0) / 1000.0, 1000.0 * (double) tot / (now_ms() - t0));
        if ((si + 1) % 10 == 0) m.save_profile();
    }
    m.save_profile();
    return 0;
}
// ---- Strata H100 addition: DSpark acceptance log ---------------------------------------------------------------
// --spec-eval IDS.json --n N: each segment is a prompt: fed from reset, then N greedy tokens decoded with the drafter, every
// draft verified (--dspark-conf 0); DS4_ROUND_LOG=file gets a line a round (segment name, drafts, accepted, confidences).
int spec_eval(strata::ds4::Model& m, const std::string& ids_path, int n, int window, float conf_min, bool spec) {
    std::vector<ScoreSeg> segs;
    std::string err;
    if (!read_score_ids(ids_path, segs, err, m.vocab())) {
        std::fprintf(stderr, "strata-dsv41: %s\n", err.c_str());
        return 1;
    }
    if (const char* f = std::getenv("DS4_ROUND_LOG")) g_round_log = std::fopen(f, "wb");
    const double t0 = now_ms();
    for (size_t si = 0; si < segs.size(); ++si) {
        const std::vector<int>& ids = segs[si].ids;
        g_round_tag = segs[si].name;
        m.reset();
        for (size_t a = 0; a < ids.size();) {
            const int T = (int) std::min<size_t>((size_t) window, ids.size() - a);
            const bool last = a + (size_t) T == ids.size();
            if (!m.feed(ids.data() + a, T, last ? 1 : 0, err)) {
                std::fprintf(stderr, "strata-dsv41: %s\n", err.c_str());
                return 1;
            }
            a += (size_t) T;
        }
        unsigned long long counter = 0;
        int tok = m.sample(0.0f, 1.0f, 1, 0.0f, 1, counter++);
        std::vector<int> got;
        int emitted = 1;
        long long rounds = 0, acc_all = 0;
        bool stop = false;
        while (emitted < n && !stop) {
            int off = 0, acc = 0;
            if (!round(m, tok, n - emitted, spec, conf_min, 0.0f, 1.0f, 1, 0.0f, 1, counter, got, off, acc, err)) {
                std::fprintf(stderr, "strata-dsv41: %s\n", err.c_str());
                return 1;
            }
            ++rounds;
            acc_all += acc;
            for (int t : got) {
                ++emitted;
                if (t == 1) stop = true;
            }
            tok = got.back();
        }
        std::fprintf(stderr, "spec-eval %zu/%zu %s: prompt %zu, %d tokens, %lld rounds, %.2f tokens a round, accepted %lld (%.0f s)\n", si + 1,
                     segs.size(), segs[si].name.c_str(), ids.size(), emitted, rounds, (double) (emitted - 1) / (double) std::max(1LL, rounds),
                     acc_all, (now_ms() - t0) / 1000.0);
        if (g_round_log != nullptr) std::fflush(g_round_log);
    }
    return 0;
}
// --multi-check (tests, with --parallel N): N conversations - the first len - 5 i tokens of the --ids prompt each -
// decode n greedy tokens alone (a plain step at a time, one conversation after another), then all of them together
// (Model::step_multi).  Each row's logits must be its own step's alone bit for bit, its tokens the same.  (The times
// are the steps' alone and together on tiers the first pass warmed; the server's real questions are the measure.)
int check_multi(strata::ds4::Model& m, const std::vector<int>& ids, int n) {
    const int N = m.seq_count(), V = m.vocab();
    std::string err;
    std::vector<std::vector<int>> prompts((size_t) N);
    for (int i = 0; i < N; ++i) {
        const size_t len = ids.size() > (size_t) (5 * i + 2) ? ids.size() - (size_t) (5 * i) : ids.size();
        prompts[(size_t) i].assign(ids.begin(), ids.begin() + (long) len);
    }
    const auto prompt = [&](int i) -> bool {
        m.seq_bind(i);
        m.reset();
        const std::vector<int>& p = prompts[(size_t) i];
        for (size_t done = 0; done < p.size();) {
            const size_t k = std::min<size_t>((size_t) m.feed_unit(), p.size() - done);
            if (!m.feed(p.data() + done, (int) k, done + k == p.size() ? 1 : 0, err)) return false;
            done += k;
        }
        return true;
    };
    std::vector<std::vector<int>> alone((size_t) N), together((size_t) N);
    std::vector<std::vector<std::vector<float>>> lg_alone((size_t) N);
    std::vector<float> lg;
    double ms_alone = 0, ms_together = 0;
    for (int i = 0; i < N; ++i) {
        if (!prompt(i)) {
            std::fprintf(stderr, "multi check: %s\n", err.c_str());
            return 1;
        }
        int tok = m.sample(0.0f, 1.0f, 0, 0.0f, 0, 0);
        alone[(size_t) i].push_back(tok);
        const double t0 = now_ms();
        for (int k = 1; k < n; ++k) {
            if (!m.step(tok, true, err) || !m.logits(lg)) {
                std::fprintf(stderr, "multi check: %s\n", err.c_str());
                return 1;
            }
            lg_alone[(size_t) i].push_back(lg);
            tok = m.sample(0.0f, 1.0f, 0, 0.0f, 0, 0);
            alone[(size_t) i].push_back(tok);
        }
        ms_alone += now_ms() - t0;
    }
    for (int i = 0; i < N; ++i) {
        if (!prompt(i)) {
            std::fprintf(stderr, "multi check: %s\n", err.c_str());
            return 1;
        }
        together[(size_t) i].push_back(m.sample(0.0f, 1.0f, 0, 0.0f, 0, 0));
    }
    long long rows = 0, same_rows = 0;
    double max_d = 0;
    std::vector<int> sv((size_t) N), tv((size_t) N);
    for (int k = 1; k < n; ++k) {
        for (int i = 0; i < N; ++i) {
            sv[(size_t) i] = i;
            tv[(size_t) i] = together[(size_t) i].back();
        }
        const double t0 = now_ms();
        if (!m.step_multi(sv.data(), tv.data(), N, err) || !m.logits(lg) || lg.size() < (size_t) N * V) {
            std::fprintf(stderr, "multi check: %s\n", err.empty() ? "no logits" : err.c_str());
            return 1;
        }
        for (int i = 0; i < N; ++i) together[(size_t) i].push_back(m.sample_multi(i, 0.0f, 1.0f, 0, 0.0f, 0, 0));
        ms_together += now_ms() - t0;
        for (int i = 0; i < N; ++i) {   // (while the tokens agree: the same history, so the same logits are due)
            if (together[(size_t) i][(size_t) k - 1] != alone[(size_t) i][(size_t) k - 1]) continue;
            const float* a = lg_alone[(size_t) i][(size_t) k - 1].data();
            const float* b = lg.data() + (size_t) i * V;
            ++rows;
            if (std::memcmp(a, b, (size_t) V * 4) == 0) ++same_rows;
            else
                for (int v = 0; v < V; ++v) max_d = std::max(max_d, (double) std::fabs(a[v] - b[v]));
        }
    }
    bool tokens_same = true;
    for (int i = 0; i < N; ++i) {
        size_t d = 0;
        while (d < alone[(size_t) i].size() && alone[(size_t) i][d] == together[(size_t) i][d]) ++d;
        const bool same = d == alone[(size_t) i].size();
        tokens_same = tokens_same && same;
        std::printf("multi check: conversation %d (%zu prompt tokens): %d tokens %s", i, prompts[(size_t) i].size(), n,
                    same ? "the same alone and together\n" : "");
        if (!same) std::printf("differ from token %zu on\n", d);
    }
    const double steps = (double) (n - 1);
    std::printf("multi check: %d conversations x %d tokens: logits bit for bit in %lld of %lld rows (max |d| %.3g); "
                "tokens the same: %s; alone %.2f tok/s, together %.2f tok/s in all (x%.2f)\n",
                N, n, same_rows, rows, max_d, tokens_same ? "yes" : "NO",
                ms_alone > 0 ? 1000.0 * steps * N / ms_alone : 0.0, ms_together > 0 ? 1000.0 * steps * N / ms_together : 0.0,
                ms_together > 0 ? ms_alone / ms_together : 0.0);
    // ... then with the drafter, greedy, so the tokens alone are due again - every draft verified, whatever its
    // confidence, so rejected windows are undone often: each conversation alone (draft / verify, one conversation's
    // rounds), then together (step_multi_spec / verify_multi, as the server's rounds, each conversation's drafts its
    // share of a window) with the drafts made one conversation after another, then in one drafter pass (draft_multi)
    bool spec_same = true;
    double ms_spec_alone = 0;
    for (int pass = 0; pass < 3 && m.draft_block() > 0; ++pass) {
        std::vector<std::vector<int>> spec_t((size_t) N);
        long long offered = 0, kept = 0, rounds = 0, one_pass = 0;
        double ms_spec = 0;
        for (int i = 0; i < N; ++i) {
            if (!prompt(i)) {
                std::fprintf(stderr, "multi check: %s\n", err.c_str());
                return 1;
            }
            spec_t[(size_t) i].push_back(m.sample(0.0f, 1.0f, 0, 0.0f, 0, 0));
        }
        if (pass == 0)
            for (int i = 0; i < N; ++i) {
                std::vector<int>& got = spec_t[(size_t) i];
                m.seq_bind(i);
                const double t0 = now_ms();
                while ((int) got.size() < n) {
                    const int room = std::min(m.verify_max(), n - (int) got.size());
                    std::vector<int> drafts;
                    std::vector<float> conf;
                    int acc = 0, next = -1;
                    if (room > 1 && !m.draft(got.back(), drafts, conf, err)) {
                        std::fprintf(stderr, "multi check: %s\n", err.c_str());
                        return 1;
                    }
                    if ((int) drafts.size() > room - 1) drafts.resize((size_t) std::max(0, room - 1));
                    if (drafts.empty()) {
                        if (!m.step(got.back(), true, err)) {
                            std::fprintf(stderr, "multi check: %s\n", err.c_str());
                            return 1;
                        }
                        next = m.sample(0.0f, 1.0f, 0, 0.0f, 0, 0);
                    } else if (!m.verify(got.back(), drafts, 0.0f, 1.0f, 0, 0.0f, 0, 0, acc, next, err)) {
                        std::fprintf(stderr, "multi check: %s\n", err.c_str());
                        return 1;
                    }
                    got.insert(got.end(), drafts.begin(), drafts.begin() + acc);
                    got.push_back(next);
                    offered += (long long) drafts.size();
                    kept += acc;
                    ++rounds;
                }
                ms_spec += now_ms() - t0;
            }
        else {
            const int share = m.verify_max() / N;
            const double t0 = now_ms();
            for (;;) {
                std::vector<int> sv;
                for (int i = 0; i < N; ++i)
                    if ((int) spec_t[(size_t) i].size() < n) sv.push_back(i);
                if (sv.empty()) break;
                const int k = (int) sv.size();
                std::vector<std::vector<int>> rws((size_t) k), drafts((size_t) k);
                std::vector<const int*> tp((size_t) k);
                std::vector<int> lens((size_t) k), dsv, dys, dix;
                for (int j = 0; j < k; ++j) {
                    const std::vector<int>& got = spec_t[(size_t) sv[(size_t) j]];
                    if (std::min(share, n - (int) got.size()) > 1) {
                        dsv.push_back(sv[(size_t) j]);
                        dys.push_back(got.back());
                        dix.push_back(j);
                    }
                }
                const int nd = (int) dsv.size();
                if (pass == 2 && nd > 1 && m.can_draft_multi(dsv.data(), nd)) {
                    std::vector<std::vector<int>> dd;
                    std::vector<std::vector<float>> cc;
                    if (!m.draft_multi(dsv.data(), dys.data(), nd, dd, cc, err)) {
                        std::fprintf(stderr, "multi check: %s\n", err.c_str());
                        return 1;
                    }
                    for (int j = 0; j < nd; ++j) drafts[(size_t) dix[(size_t) j]] = dd[(size_t) j];
                    ++one_pass;
                } else
                    for (int j = 0; j < nd; ++j) {
                        std::vector<float> conf;
                        m.seq_bind(dsv[(size_t) j]);
                        if (!m.draft(dys[(size_t) j], drafts[(size_t) dix[(size_t) j]], conf, err)) {
                            std::fprintf(stderr, "multi check: %s\n", err.c_str());
                            return 1;
                        }
                    }
                for (int j = 0; j < k; ++j) {
                    const std::vector<int>& got = spec_t[(size_t) sv[(size_t) j]];
                    const int room = std::min(share, n - (int) got.size());
                    if ((int) drafts[(size_t) j].size() > room - 1) drafts[(size_t) j].resize((size_t) std::max(0, room - 1));
                    rws[(size_t) j].push_back(got.back());
                    rws[(size_t) j].insert(rws[(size_t) j].end(), drafts[(size_t) j].begin(), drafts[(size_t) j].end());
                    tp[(size_t) j] = rws[(size_t) j].data();
                    lens[(size_t) j] = (int) rws[(size_t) j].size();
                }
                if (!m.step_multi_spec(sv.data(), tp.data(), lens.data(), k, err)) {
                    std::fprintf(stderr, "multi check: %s\n", err.c_str());
                    return 1;
                }
                for (int j = 0; j < k; ++j) {
                    int acc = 0, next = -1;
                    if (!m.verify_multi(j, drafts[(size_t) j], 0.0f, 1.0f, 0, 0.0f, 0, 0, acc, next, err)) {
                        std::fprintf(stderr, "multi check: %s\n", err.c_str());
                        return 1;
                    }
                    std::vector<int>& got = spec_t[(size_t) sv[(size_t) j]];
                    got.insert(got.end(), drafts[(size_t) j].begin(), drafts[(size_t) j].begin() + acc);
                    got.push_back(next);
                    offered += (long long) drafts[(size_t) j].size();
                    kept += acc;
                }
                ++rounds;
            }
            ms_spec = now_ms() - t0;
        }
        if (pass == 0) ms_spec_alone = ms_spec;
        bool same_all = true;
        for (int i = 0; i < N; ++i) {
            size_t d = 0;
            while (d < alone[(size_t) i].size() && alone[(size_t) i][d] == spec_t[(size_t) i][d]) ++d;
            const bool same = d == alone[(size_t) i].size();
            same_all = same_all && same;
            if (!same) std::printf("multi check: conversation %d with the drafter (pass %d): differs from token %zu on\n", i, pass, d);
        }
        spec_same = spec_same && same_all;
        static const char* const what[3] = {"alone with the drafter", "together, drafts one after another",
                                            "together, drafts in one pass"};
        std::printf("multi check: %d conversations x %d tokens %s: tokens the same: %s; %lld rounds%s, drafts %lld of "
                    "%lld kept; %.2f tok/s in all (x%.2f over together without, x%.2f over alone with the drafter)\n",
                    N, n, what[pass], same_all ? "yes" : "NO", rounds,
                    pass == 2 ? (" (" + std::to_string(one_pass) + " drafted in one pass)").c_str() : "", kept, offered,
                    ms_spec > 0 ? 1000.0 * steps * N / ms_spec : 0.0, ms_spec > 0 ? ms_together / ms_spec : 0.0,
                    ms_spec > 0 ? ms_spec_alone / ms_spec : 0.0);
    }
    return tokens_same && spec_same && same_rows == rows ? 0 : 1;
}

}  // namespace

#ifndef DS4_BUILD_ID
#define DS4_BUILD_ID "unknown"
#endif

#if defined(STRATA_METAL)
#include "sampler_stat.inc"
#endif
int main(int argc, char** argv) {
    strata::ds4::os::fine_timer();
    strata::platform::restore_normal_io_priority();   // #4733
#if defined(STRATA_METAL)
    if (argc > 1 && std::string(argv[1]) == "--sampler-test") return sampler_stat::run(argc - 1, argv + 1);   // (the sampler's statistics harness)
#endif
#if defined(__APPLE__)
    if (std::getenv("DS4_BT")) {   // a crash prints where it was (no debugger on the Mac box)
        signal(SIGSEGV, [](int) { void* fr[40]; const int n = backtrace(fr, 40); backtrace_symbols_fd(fr, n, 2); _exit(139); });
        signal(SIGBUS, [](int) { void* fr[40]; const int n = backtrace(fr, 40); backtrace_symbols_fd(fr, n, 2); _exit(138); });
        signal(SIGABRT, [](int) { void* fr[40]; const int n = backtrace(fr, 40); backtrace_symbols_fd(fr, n, 2); _exit(134); });
    }
#endif
    strata::ds4::Options o;
    std::string ids_s, dump, eos = "1";
    int n = 0, window = 0;   // (0: the model's feed unit - a whole prompt on the chunked path; 16 for the scoring runs)
    bool spec = true;
    float temp = 0.0f;   // the CLI's sampling (0: greedy)
    unsigned long long cli_seed = 1;
    bool serve_mode = false, multi_check = false, vision_resident = false, vision_lazy_asked = false;   // (--vision-lazy: the default now, still accepted)
    bool pack_info_mode = false;
    std::string images, enc_rgb, enc_out;
    std::string score_ids, score_out, prof_ids, spec_ids;
    int enc_w = 0, enc_h = 0;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto val = [&](const char* what) -> std::string {
            if (i + 1 >= argc) {
                std::fprintf(stderr, "%s needs a value\n", what);
                std::exit(2);
            }
            return argv[++i];
        };
        if (a == "--pack") o.pack = val("--pack");
        else if (a == "--gpus") {
            for (int g : parse_ids(val("--gpus"))) o.gpus.push_back(g);
        } else if (a == "--max-context") o.max_context = std::atoll(val("--max-context").c_str());
        else if (a == "--reserve-mib") o.reserve_mib = std::atoll(val("--reserve-mib").c_str());
        else if (a == "--vram-experts") { o.vram_experts = std::atof(val("--vram-experts").c_str()); o.vram_experts_auto = false; }
        else if (a == "--cpu-threads") o.cpu_threads = std::atoi(val("--cpu-threads").c_str());
        else if (a == "--expert-profile") o.profile = val("--expert-profile");
        else if (a == "--profile-readonly") o.profile_readonly = true;   // (read the placement, never write the counts back)
        else if (a == "--ssd-cache") o.ssd_cache_gb = std::atof(val("--ssd-cache").c_str());
        else if (a == "--ssd-policy") {
            const std::string v = val("--ssd-policy");
            o.ssd_policy = v == "lfu" ? 1 : v == "lfulru" ? 2 : 0;
        }
        else if (a == "--eos") eos = val("--eos");
        else if (a == "--no-expert-cache") o.expert_cache = false;
        else if (a == "--dspark") {
            const std::string v = val("--dspark");
            spec = v != "off";
            o.dspark = v != "none";   // none: not even loaded
            o.dspark_auto = v == "auto";
        } else if (a == "--dspark-conf") o.dspark_conf = (float) std::atof(val("--dspark-conf").c_str());
        else if (a == "--experts") {
            const std::string v = val("--experts");
            o.experts = v == "auto" ? -1 : std::max(1, std::atoi(v.c_str()));
        }
        else if (a == "--dspark-block") o.dspark_block = std::max(0, std::atoi(val("--dspark-block").c_str()));
        else if (a == "--dspark-depth") g_spec_depth = std::max(1, std::atoi(val("--dspark-depth").c_str()));
        else if (a == "--tp") o.tp = std::max(1, std::atoi(val("--tp").c_str()));
        else if (a == "--prefill-chunk") o.prefill_chunk = std::max(0, std::atoi(val("--prefill-chunk").c_str()));
        else if (a == "--threads") o.threads = val("--threads") != "off";
        else if (a == "--vision") o.vision = val("--vision") != "off";
        else if (a == "--vision-resident") vision_resident = true;
        else if (a == "--vision-lazy") (void) (vision_lazy_asked = true);
        else if (a == "--mmap-experts") {
            const std::string v = val("--mmap-experts");
            o.mmap_experts = v == "on" ? 1 : v == "off" ? 0 : -1;
        } else if (a == "--images") images = val("--images");
        else if (a == "--encode-image") enc_rgb = val("--encode-image");
        else if (a == "--dump-image") enc_out = val("--dump-image");
        else if (a == "--size") {
            const std::string v = val("--size");
            enc_w = std::atoi(v.c_str());
            enc_h = v.find('x') != std::string::npos ? std::atoi(v.c_str() + v.find('x') + 1) : 0;
        }
        else if (a == "--temperature") temp = (float) std::atof(val("--temperature").c_str());
        else if (a == "--seed") cli_seed = std::strtoull(val("--seed").c_str(), nullptr, 10);
        else if (a == "--verbose" || a == "-v") {
            o.verbose = true;
            std::fprintf(stderr, "ds4: build %s (%s)\n", DS4_BUILD_ID, STRATA_VERSION);
        } else if (a == "--version") {
            std::printf("strata-dsv41 %s build %s\n", STRATA_VERSION, DS4_BUILD_ID);
            return 0;
        }
        else if (a == "--ids") ids_s = val("--ids");
        else if (a == "--n") n = std::atoi(val("--n").c_str());
        else if (a == "--window") window = std::max(1, std::atoi(val("--window").c_str()));   // (> 16: the chunked prompt path)
        else if (a == "--dump-logits") dump = val("--dump-logits");
        else if (a == "--score") score_ids = val("--score");
        else if (a == "--score-out") score_out = val("--score-out");
        else if (a == "--profile-run") prof_ids = val("--profile-run");
        else if (a == "--spec-eval") spec_ids = val("--spec-eval");
        else if (a == "--serve") serve_mode = true;
        else if (a == "--parallel") {   // (--serve: conversations at once)
            o.seqs = std::atoi(val("--parallel").c_str());
            if (o.seqs < 1 || o.seqs > 8) {
                std::fprintf(stderr, "--parallel takes 1 to 8\n");
                return 2;
            }
        }
        else if (a == "--multi-check") multi_check = true;   // (tests: --parallel's rows against each alone)
        else if (a == "--pack-info") pack_info_mode = true;
        else if (a == "--layer-split") (void) val("--layer-split");   // the server's (this engine balances its own split)
        else if (a == "--help" || a == "-h") {
            usage();
            return 0;
        } else {
            std::fprintf(stderr, "unknown option %s\n", a.c_str());
            usage();
            return 2;
        }
    }
    if (pack_info_mode) {   // the pack's description only: no device, no weights, no lock
        if (o.pack.empty()) {
            usage();
            return 2;
        }
        return strata::ds4::pack_info(o.pack, stdout);
    }
    if (multi_check && (o.seqs < 2 || ids_s.empty())) {
        std::fprintf(stderr, "--multi-check needs --parallel 2..8 and --ids\n");
        return 2;
    }
    if (o.pack.empty() || (!serve_mode && ids_s.empty() && enc_rgb.empty() && score_ids.empty() && prof_ids.empty() && spec_ids.empty())) {
        usage();
        return 2;
    }
    if (o.profile.empty()) o.profile = o.pack + "/expert-profile.bin";   // "none": no profile
    if (o.profile == "none") o.profile.clear();
    o.vision_only = !enc_rgb.empty();
    o.vision_lazy = serve_mode && !vision_resident && !o.vision_only;   // (--vision-lazy is the same as the default; kept for configs that name it)
    strata::ds4::Model m;
    std::string err;
    if (!m.load(o, err)) {
        std::fprintf(stderr, "maya-ds4: %s\n", err.c_str());
        return 1;
    }
    if (!o.vision_only) std::fprintf(stderr, "%s", m.describe().c_str());
    if (const char* idle = std::getenv("DS4_TEST_IDLE_MS")) std::this_thread::sleep_for(std::chrono::milliseconds(std::atoi(idle)));   // (tests: a person typing)
    if (serve_mode) return serve(m, eos, spec, o.dspark_conf, o.pack);
    const int window_given = window;
    if (window == 0) window = 16;   // (the scoring / profiling runs; the plain run below takes the model's feed unit)
    if (!prof_ids.empty()) return profile_run(m, prof_ids, window);
    if (!spec_ids.empty()) return spec_eval(m, spec_ids, n, window, o.dspark_conf, spec);
    if (!score_ids.empty()) return score(m, score_ids, score_out.empty() ? score_ids + ".tsv" : score_out, window, std::atoi(eos.c_str()));

    if (!enc_rgb.empty()) {   // the vision tower alone: a picture's span rows (BF16 [rows][dim]) into --dump-image
        std::vector<uint8_t> px((size_t) std::max(0, enc_w) * std::max(0, enc_h) * 3);
        std::FILE* f = std::fopen(enc_rgb.c_str(), "rb");
        const bool got = f != nullptr && !px.empty() && std::fread(px.data(), 1, px.size(), f) == px.size();
        if (f != nullptr) std::fclose(f);
        if (!got) {
            std::fprintf(stderr, "maya-ds4: %s: expected %d x %d RGB bytes (--size WxH)\n", enc_rgb.c_str(), enc_w, enc_h);
            return 1;
        }
        std::vector<uint16_t> span;
        for (int k = 0; k < 2; ++k) {   // (the second: warm)
            const double te = now_ms();
            if (!m.encode_image(px.data(), enc_w, enc_h, span, err)) {
                std::fprintf(stderr, "maya-ds4: %s\n", err.c_str());
                return 1;
            }
            std::fprintf(stderr, "vision: %d x %d pixels -> a span of %zu rows in %.0f ms\n", enc_w, enc_h,
                         span.size() / (size_t) m.dim(), now_ms() - te);
        }
        std::FILE* o2 = enc_out.empty() ? nullptr : std::fopen(enc_out.c_str(), "wb");
        if (o2 != nullptr) {
            std::fwrite(span.data(), 2, span.size(), o2);
            std::fclose(o2);
        }
        return 0;
    }
    if (!ids_s.empty() && ids_s[0] == '@') {   // --ids @file: the ids in a file (a command line is 32 KB on Windows)
        std::ifstream idf(ids_s.substr(1));
        std::stringstream ss;
        ss << idf.rdbuf();
        ids_s = ss.str();
    }
    const std::vector<int> ids = parse_ids(ids_s);
    if (multi_check) return check_multi(m, ids, std::max(2, n));
    if (!images.empty()) {   // --images: a GENI file's pictures, for the prompt's spans
        std::vector<Pic> pics;
        if (!read_pics(images, pics, err)) {
            std::fprintf(stderr, "maya-ds4: %s\n", err.c_str());
            return 1;
        }
        std::vector<strata::ds4::Image> imgs;
        for (const Pic& q : pics) {
            auto rows = std::make_shared<std::vector<uint16_t>>();
            const double te = now_ms();
            if (!m.encode_image(q.rgb.data(), q.width, q.height, *rows, err)) {
                std::fprintf(stderr, "maya-ds4: %s\n", err.c_str());
                return 1;
            }
            std::fprintf(stderr, "picture at %lld: %d x %d pixels, %zu tokens, %.0f ms\n", (long long) q.start, q.width,
                         q.height, rows->size() / (size_t) m.dim(), now_ms() - te);
            imgs.push_back({q.start, rows});
        }
        m.set_images(std::move(imgs));
    }
    std::FILE* df = dump.empty() ? nullptr : std::fopen(dump.c_str(), "wb");
    std::vector<float> lg;
    if (window_given == 0) window = df != nullptr ? 16 : std::max(16, std::min(m.feed_unit(), 1 << 30));
    m.begin_request();
    const double t0 = now_ms();
    for (size_t a = 0; a < ids.size();) {
        const int T = (int) std::min<size_t>((size_t) window, ids.size() - a);
        const bool last = a + (size_t) T == ids.size();
        const double tf0 = now_ms();
        if (!m.feed(ids.data() + a, T, df != nullptr ? T : (last ? 1 : 0), err)) {
            std::fprintf(stderr, "maya-ds4: %s\n", err.c_str());
            return 1;
        }
        if (std::getenv("DS4_FEEDLOG") != nullptr)   // each feed's time: the incremental turn (--window W: a long prefix, then a short one)
            std::fprintf(stderr, "feed at %zu: %d tokens in %.0f ms (%.1f tok/s)\n", a, T, now_ms() - tf0, 1000.0 * T / (now_ms() - tf0));
        if (df != nullptr && m.logits(lg)) std::fwrite(lg.data(), 4, lg.size(), df);
        a += (size_t) T;
    }
    m.set_images({});
    const double t1 = now_ms();
    std::fprintf(stderr, "prompt: %zu tokens in %.0f ms (%.2f tok/s, windows of %d)\n", ids.size(), t1 - t0,
                 1000.0 * (double) ids.size() / (t1 - t0), window);
#if defined(STRATA_METAL)
    if (ds4_dbg("DS4_SEQ_LOG")) std::fprintf(stderr, "dispatch seq after the prompt: %llu\n", ds4_dispatch_seq());
#endif
    const strata::ds4::Stats s0 = m.stats();
    if (s0.eg_rows > 0)
        std::fprintf(stderr, "prompt's Engram rows: %lld, %.1f%% without a read (repeats and the row cache), %lld read\n",
                     (long long) s0.eg_rows, 100.0 * (double) s0.eg_hits / (double) s0.eg_rows, (long long) s0.eg_reads);
    const long long io0 = io_read_bytes();
    std::vector<int> out;
    std::vector<double> tok_ms;   // (DS4_TOKLOG: each token's time since the one before - the latency ledger)
    double t_prev = t1;
    unsigned long long counter = 0;
    long long d_off = 0, d_acc = 0, rounds = 0;
    bool cool = false;
    if (n > 0) {
        int tok = m.sample(temp, 1.0f, temp > 0 ? 64 : 1, 0.0f, cli_seed, counter++);
        std::vector<int> got;
        for (;;) {
            if (tok < 0) {   // (a sample's CUDA error, printed by the sampler)
                std::fprintf(stderr, "strata-dsv41: the sample failed\n");
                return 1;
            }
            out.push_back(tok);
            {
                const double tn = now_ms();
                tok_ms.push_back(tn - t_prev);
                t_prev = tn;
            }
            std::printf("%d,", tok);
            std::fflush(stdout);
            if (ds4_dbg("DS4_PROGRESS") && out.size() % 200 == 0) {   // (sustained-run time series)
                static double tp = now_ms();
                static size_t np = 0;
                const double tn = now_ms();
                std::fprintf(stderr, "progress %zu tokens: %.2f tok/s over the last %zu\n", out.size(), 1000.0 * (double) (out.size() - np) / (tn - tp), out.size() - np);
                tp = tn;
                np = out.size();
            }
            if ((tok == 1 && std::getenv("DS4_NO_EOS") == nullptr) || (int) out.size() >= n) break;   // (DS4_NO_EOS: decode n tokens whatever)
            if (got.empty()) {
                int off = 0, acc = 0;
                const bool try_spec = spec && !cool;
                if (!round(m, tok, n - (int) out.size(), try_spec, o.dspark_conf, temp, 1.0f, temp > 0 ? 64 : 1, 0.0f,
                           cli_seed, counter, got, off, acc, err)) {
                    std::fprintf(stderr, "maya-ds4: %s\n", err.c_str());
                    return 1;
                }
                d_off += off;
                d_acc += acc;
                ++rounds;
                cool = try_spec && off == 0;
                std::reverse(got.begin(), got.end());
            }
            tok = got.back();
            got.pop_back();
        }
        std::printf("\n");
        if (df != nullptr && m.logits(lg)) std::fwrite(lg.data(), 4, lg.size(), df);
    }
    if (spec && rounds > 0) {
        const auto& s1 = m.stats();
        const double nv = (double) std::max<int64_t>(1, s1.drafts_rounds - s0.drafts_rounds);
        std::fprintf(stderr, "dspark: %lld rounds, %lld drafts offered, %lld accepted (%.2f tokens a round); drafting "
                             "%.2f ms a round, verifying %.1f ms in all (%lld windows)\n",
                     rounds, d_off, d_acc, (double) out.size() / (double) rounds, (s1.draft_ms - s0.draft_ms) / nv,
                     s1.verify_ms - s0.verify_ms, (long long) (s1.drafts_offered > 0 ? 0 : 0));
        std::fprintf(stderr, "dspark %s\n", per_position(s0, s1).c_str());
    }
    if (std::getenv("DS4_TOKLOG") != nullptr && tok_ms.size() > 8) {   // the decode's tokens after the first (which carries the prompt's tail)
        std::vector<double> v(tok_ms.begin() + 1, tok_ms.end());
        std::sort(v.begin(), v.end());
        std::fprintf(stderr, "token latency ms: p50 %.0f p95 %.0f p99 %.0f max %.0f mean %.0f (%zu tokens)\n", v[v.size() / 2],
                     v[(size_t) ((double) v.size() * 0.95)], v[(size_t) ((double) v.size() * 0.99)], v.back(),
                     std::accumulate(v.begin(), v.end(), 0.0) / (double) v.size(), v.size());
    }
    const double t2 = now_ms();
    const auto& s = m.stats();
    const double nt = std::max<double>(1.0, (double) (s.tokens - s0.tokens));
    std::fprintf(stderr, "decode: %zu tokens in %.0f ms (%.2f tok/s); CPU lane: %lld of %lld experts, streamed to the GPU: "
                         "%lld; the cache moved %lld (%lld in all)\n", out.size(), t2 - t1,
                 out.size() > 1 ? 1000.0 * (double) (out.size() - 1) / (t2 - t1) : 0.0,
                 (long long) (s.cpu_experts - s0.cpu_experts), (long long) (s.experts - s0.experts),
                 (long long) (s.stream_experts - s0.stream_experts), (long long) (s.swaps - s0.swaps),
                 (long long) s.swaps);
    std::fprintf(stderr, "decode per token: layers %.2f ms (of it: routing waits %.2f, CPU lane %.2f, the rest host launches), "
                         "head wait %.2f, Engram reads %.2f\n",
                 (s.layer_ms - s0.layer_ms) / nt, (s.gpu_wait_ms - s0.gpu_wait_ms) / nt, (s.cpu_ms - s0.cpu_ms) / nt,
                 (s.head_ms - s0.head_ms) / nt, (s.engram_ms - s0.engram_ms) / nt);
    if (s.eg_rows > s0.eg_rows) {   // the Engram rows: how many the row cache served, the reads' time behind the layers
        const double rows = (double) (s.eg_rows - s0.eg_rows);
        std::fprintf(stderr, "Engram per token: %.0f rows, %.1f%% without a read (the row cache holds %lld of %lld), "
                             "%.1f read, the reads %.2f ms from asking to the last one (left to wait: %.2f)\n",
                     rows / nt, 100.0 * (double) (s.eg_hits - s0.eg_hits) / rows, (long long) s.eg_cached,
                     (long long) s.eg_cache_cap, (double) (s.eg_reads - s0.eg_reads) / nt, (s.eg_io_ms - s0.eg_io_ms) / nt,
                     (s.engram_ms - s0.engram_ms) / nt);
    }
    const long long io1 = io_read_bytes();
    if (io0 >= 0 && io1 >= 0)
        std::fprintf(stderr, "decode reads from the disk: %.1f MB a token\n", (double) (io1 - io0) / 1e6 / nt);
    if (s.skipped > s0.skipped)   // STRATA_DS4_DISK_SKIP_W
        std::fprintf(stderr, "decode left out %.1f experts a token not in RAM (%.1f%% of the routed)\n",
                     (double) (s.skipped - s0.skipped) / nt,
                     100.0 * (double) (s.skipped - s0.skipped) / std::max<double>(1.0, (double) (s.experts - s0.experts)));
    if (s.store_slots_max > 0) {   // --ssd-cache
        const double h = (double) (s.store_hits - s0.store_hits), m = (double) (s.store_misses - s0.store_misses);
        std::fprintf(stderr,
                     "store: %.1f%% of the CPU lane's experts were in RAM (%.1f hits, %.1f misses a token); read %.1f MB a token "
                     "(demand and guesses), the CPU lane waited %.1f ms a token for them; %lld prefetched, %lld used, %lld given up "
                     "unused; slots %lld of %lld, %lld evictions, %lld failed reads; the GPU computed %.1f experts a token from the store "
                     "(%lld slots visible)\n",
                     100.0 * h / std::max(1.0, h + m), h / nt, m / nt, (double) (s.store_bytes - s0.store_bytes) / 1e6 / nt,
                     (s.store_wait_ms - s0.store_wait_ms) / nt, (long long) (s.store_prefetched - s0.store_prefetched),
                     (long long) (s.store_pf_used - s0.store_pf_used), (long long) (s.store_pf_wasted - s0.store_pf_wasted),
                     (long long) s.store_slots, (long long) s.store_slots_max, (long long) (s.store_evictions - s0.store_evictions),
                     (long long) (s.store_fails - s0.store_fails), (double) (s.store_zc - s0.store_zc) / nt,
                     (long long) s.store_zc_visible);
    }
    if (s.skip_on >= 0)
        std::fprintf(stderr, "DISK_SKIP auto: %s at the end, %lld switches\n", s.skip_on ? "on" : "off",
                     (long long) s.skip_switches);
    if (s.ahead_picks > s0.ahead_picks) {   // lookahead
        const double p = (double) (s.ahead_picks - s0.ahead_picks), c = (double) (s.ahead_cold - s0.ahead_cold);
        std::fprintf(stderr, "lookahead: the guesses a layer before held %.1f%% of the picks (their top k %.1f%%, two layers "
                             "before %.1f%%); file-read picks asked for early %.1f%% (%lld of %lld); %.1f experts asked "
                             "for early a token\n",
                     100.0 * (double) (s.ahead_hit - s0.ahead_hit) / p, 100.0 * (double) (s.ahead_hit_k - s0.ahead_hit_k) / p,
                     100.0 * (double) (s.ahead_hit2 - s0.ahead_hit2) / p,
                     c > 0 ? 100.0 * (double) (s.ahead_cold_hit - s0.ahead_cold_hit) / c : 0.0,
                     (long long) (s.ahead_cold_hit - s0.ahead_cold_hit), (long long) (s.ahead_cold - s0.ahead_cold),
                     (double) (s.ahead_pf - s0.ahead_pf) / nt);
        std::string rk = "lookahead: picked by the guess's rank (1 ..):";
        for (int i = 0; i < 16; ++i) {
            const long long n = (long long) (s.ahead_rank_n[i] - s0.ahead_rank_n[i]);
            if (n <= 0) break;
            char b[24];
            std::snprintf(b, sizeof b, " %.0f%%", 100.0 * (double) (s.ahead_rank_hit[i] - s0.ahead_rank_hit[i]) / (double) n);
            rk += b;
        }
        std::fprintf(stderr, "%s\n", rk.c_str());
    }
    if (s.prof_tokens > s0.prof_tokens) {   // DS4_PROF
        const double pt = (double) (s.prof_tokens - s0.prof_tokens);
        double sum = 0;
        for (int i = 0; i < 6; ++i) sum += (s.prof_ms[i] - s0.prof_ms[i]) / pt;
        std::fprintf(stderr, "GPU per token (DS4_PROF): %.2f ms - between layers %.2f, attention %.2f, routing + shared "
                             "expert %.2f, experts gate/up %.2f, waiting for the CPU lane %.2f, down + post %.2f\n", sum,
                     (s.prof_ms[0] - s0.prof_ms[0]) / pt, (s.prof_ms[1] - s0.prof_ms[1]) / pt,
                     (s.prof_ms[2] - s0.prof_ms[2]) / pt, (s.prof_ms[3] - s0.prof_ms[3]) / pt,
                     (s.prof_ms[4] - s0.prof_ms[4]) / pt, (s.prof_ms[5] - s0.prof_ms[5]) / pt);
    }
    if (df != nullptr) std::fclose(df);
    m.save_profile();
    return 0;
}
