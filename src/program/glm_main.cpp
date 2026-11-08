// src/program/generate.cpp - P2.S6: `strata generate`.
//
// THE DRIVER, and the first program in this project that answers a question.  Everything below it is a
// component; this is the thing that composes them into a token:
//
//     embed_row(token)  ->  48 captured layer graphs (the CPU expert pool behind the doorbell)  ->
//     lm_head(R)        ->  sample        ->  embed_row(next)  ->  ...
//
// WHAT IT IS NOT.  There is no tokenizer here.  `pack/full/tokenizer/` and `tools/strata_tokenizer.py` exist,
// and a C++ BPE is Phase 1's deliverable rather than this program's, so the prompt arrives as IDS via
// `--tokens`.  That is not a placeholder: it is exactly what Gate C1 needs, because C1 compares logits against
// llama.cpp on the SAME ids, and a tokenizer on only one side of that comparison is a second variable.
//
// AND IT IS PHASE 2, so hit rate is `h = 0` and the number it prints is slow on purpose
// (`phase-2-correct-engine.md:5-9`).  What it is FOR is the honest tok/s figure and the logit dump.

#include "strata/core/expert_cache.hpp"
#include "strata/platform/env_alias.hpp"
#include "strata/core/expert_source.hpp"
#include "strata/core/remote_experts.hpp"
#include "strata/core/on_device.hpp"
#include "strata/core/layer.hpp"
#include "strata/core/layout.hpp"
#include "strata/core/session.hpp"
#include "strata/core/weights.hpp"
#include "strata/kernels/cpu/expert.hpp"
#include "strata/kernels/cpu/pool.hpp"
#include "strata/kernels/cpu/expert_layout.hpp"
#include "strata/kernels/ngram.hpp"
#include "strata/kernels/s2_expert_grouped.hpp"
#include "strata/kernels/sampler.hpp"
#include "strata/kernels/verify_kernels.hpp"
#include "strata/kernels/shared_expert.hpp"
#include "strata/kernels/native_moe.hpp"
#include "strata/kernels/native_gdn.hpp"
#include "strata/kernels/native_router.hpp"
#include "strata/kernels/native_qsa.hpp"
#include "strata/kernels/native_qsa_indexer.hpp"
#include "strata/kernels/native_rope.hpp"
#include "strata/kernels/mrope.hpp"
#include "strata/kernels/kv_q4.hpp"
#include "strata/kernels/qsa.hpp"
#include "strata/core/native_head.hpp"
#include "strata/core/verify.hpp"
#include "strata/core/mtp.hpp"
#include "strata/prefill/prefill.hpp"
#include "strata/prefill/failure.hpp"
#include "strata/core/native_dense.hpp"
#include "strata/core/glm_model.hpp"
#include "strata/core/mtp_rows.hpp"
#include "strata/core/split_groups.hpp"
#include "strata/core/prefix_cut.hpp"
#include "strata/core/prefill_pieces.hpp"
#include "strata/core/prefill_gaps.hpp"
#include "strata/program/logits_selection.hpp"
#include "strata/program/conv_cache.hpp"
#include "strata/spec/draft_policy.hpp"
#include "strata/spec/suffix_drafter.hpp"
#include "strata/kernels/cvec.hpp"
#include "strata/core/progress.hpp"
#ifndef NOMINMAX
#define NOMINMAX   // gguf_reader.hpp includes windows.h
#endif
#include "strata/artifact/gguf_reader.hpp"
#if defined(_WIN32)
#include <windows.h>
#include <psapi.h>
#include <io.h>
#else
#include <unistd.h>
#include <cerrno>
#endif

#include <cuda_runtime.h>

#include <array>
#include <chrono>
#include <algorithm>
#include <iostream>
#include <thread>
#include <atomic>
#include <condition_variable>
#include <deque>
#include <functional>
#include <map>
#include <mutex>
#include <charconv>
#include <climits>
#include <cmath>
#include <limits>
#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <csignal>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <set>
#include <vector>
namespace {

using Clock = std::chrono::steady_clock;


// ---- issue #31: what the watchdog prints before it stops a stalled engine
struct MemSample {
    unsigned long long faults = 0, rss_mib = 0, avail_mib = 0, commit_mib = 0;   // faults: hard (Linux) / all (Windows)
};
MemSample mem_sample() {
    MemSample m;
#if defined(_WIN32)
    PROCESS_MEMORY_COUNTERS pmc{};
    if (K32GetProcessMemoryInfo(GetCurrentProcess(), &pmc, sizeof pmc)) {
        m.faults = pmc.PageFaultCount;
        m.rss_mib = pmc.WorkingSetSize >> 20;
        m.commit_mib = pmc.PagefileUsage >> 20;
    }
    MEMORYSTATUSEX ms{};
    ms.dwLength = sizeof ms;
    if (GlobalMemoryStatusEx(&ms)) m.avail_mib = ms.ullAvailPhys >> 20;
#else
    if (std::FILE* f = std::fopen("/proc/self/stat", "r")) {
        char buf[4096];
        const size_t n = std::fread(buf, 1, sizeof buf - 1, f);
        buf[n] = 0;
        std::fclose(f);
        const char* s = std::strrchr(buf, ')');   // fields after the command name: 3 state ... 12 majflt
        for (int field = 2; s && field < 12; ++field) s = std::strchr(s + 1, ' ');
        if (s) m.faults = std::strtoull(s + 1, nullptr, 10);
    }
    auto kb = [](const char* path, const char* key) -> unsigned long long {
        unsigned long long v = 0;
        if (std::FILE* f = std::fopen(path, "r")) {
            char line[256];
            const size_t kl = std::strlen(key);
            while (std::fgets(line, sizeof line, f))
                if (std::strncmp(line, key, kl) == 0) { v = std::strtoull(line + kl, nullptr, 10); break; }
            std::fclose(f);
        }
        return v;
    };
    m.rss_mib = kb("/proc/self/status", "VmRSS:") >> 10;
    m.commit_mib = kb("/proc/self/status", "VmSwap:") >> 10;
    m.avail_mib = kb("/proc/meminfo", "MemAvailable:") >> 10;
#endif
    return m;
}

void stall_report(std::FILE* f, uint64_t layers_during) {
    strata::core::Progress& p = strata::core::progress();
    std::fprintf(f, "maya serve: stall report (engine %s): stage \"%s %lld\" for %lld s; %llu layers served since the "
                    "last finished step (0 = stopped, more = slow)\n", STRATA_VERSION, p.where.load(),
                 (long long) p.detail.load(), (long long) ((strata::core::progress_now_ms() - p.since_ms.load()) / 1000),
                 (unsigned long long) layers_during);
    for (int pass = 0; pass < 2; ++pass) {
        if (pass == 1) {
            std::this_thread::sleep_for(std::chrono::seconds(2));
            std::fprintf(f, "  2 s later:\n");
        }
        if (auto fn = strata::core::diag_pool_fn().load()) fn(f);
        if (auto fn = strata::core::diag_verify_fn().load()) fn(f);
        const MemSample m = mem_sample();
        std::fprintf(f, "  memory: %llu MiB resident, %llu MiB %s, %llu MiB RAM available; %llu %s\n", m.rss_mib,
                     m.commit_mib,
#if defined(_WIN32)
                     "committed", m.avail_mib, m.faults, "page faults so far"
#else
                     "in swap", m.avail_mib, m.faults, "major page faults so far"
#endif
        );
        std::fflush(f);
    }
#if defined(_WIN32)
    // every thread's stack, to read against this build's PDB: a few MB beside the engine's working directory
    if (HMODULE dbg = LoadLibraryA("dbghelp.dll")) {
        using Fn = BOOL(WINAPI*)(HANDLE, DWORD, HANDLE, int, void*, void*, void*);
        if (auto write = (Fn) GetProcAddress(dbg, "MiniDumpWriteDump")) {
            char path[64];
            std::snprintf(path, sizeof path, "maya-stall-%lu.dmp", (unsigned long) GetCurrentProcessId());
            HANDLE h = CreateFileA(path, GENERIC_WRITE, 0, nullptr, CREATE_ALWAYS, FILE_ATTRIBUTE_NORMAL, nullptr);
            if (h != INVALID_HANDLE_VALUE) {
                const int kThreadInfo = 0x1000;   // MiniDumpWithThreadInfo; MiniDumpNormal = 0
                const BOOL ok = write(GetCurrentProcess(), GetCurrentProcessId(), h, kThreadInfo, nullptr, nullptr, nullptr);
                CloseHandle(h);
                char full[MAX_PATH];
                if (!GetFullPathNameA(path, MAX_PATH, full, nullptr)) std::snprintf(full, sizeof full, "%s", path);
                std::fprintf(f, "  %s the thread stacks to %s (attach it to the issue)\n", ok ? "wrote" : "could not write",
                             full);
            }
        }
    }
#endif
}

// issue #29: a request whose heartbeat (tokens, prompt chunks or layers, verify windows) stops for STRATA_WATCHDOG_S
// seconds (default_s when unset; 0 = off) is stuck on a flag nobody will raise - end the engine with where it was, so
// the server starts it again instead of the GPU spinning forever.  Issue #31: before it does, it reports what every
// part was doing (stall_report), so one occurrence says where the wait is.  Issue #40: the GLM serve has it too.
void start_serve_watchdog(int default_s) {
    const char* ws = std::getenv("STRATA_WATCHDOG_S");
    const int limit = ws ? std::atoi(ws) : default_s;
    if (limit <= 0) return;
    std::thread([limit] {
        strata::core::Progress& p = strata::core::progress();
        uint64_t last = p.beats.load(), ticks_at = p.ticks.load();
        auto since = std::chrono::steady_clock::now();
        for (;;) {
            std::this_thread::sleep_for(std::chrono::seconds(1));
            const auto now = std::chrono::steady_clock::now();
            const uint64_t b = p.beats.load();
            if (!p.busy.load() || b != last) { last = b; ticks_at = p.ticks.load(); since = now; continue; }
            if (now - since < std::chrono::seconds(limit)) continue;
            std::fprintf(stderr, "maya serve: no progress for %d s during a request (%s %lld) - stopping "
                                 "the engine so the server starts it again (issue #29)\n",
                         limit, p.where.load(), (long long) p.detail.load());
            stall_report(stderr, p.ticks.load() - ticks_at);
            std::fflush(stderr);
            // not abort(): Windows Error Reporting can hold the process open, and a core dump of tens of GB of
            // pinned expert tiers takes minutes - the stall report above already says where it stopped
            std::_Exit(3);
        }
    }).detach();
}

/// STRATA_TRACE=1: the VRAM left at a step of the startup (finds what fills the card after the cache is sized)
void mem_mark(const char* where) {
    static const bool on = std::getenv("STRATA_TRACE") != nullptr;
    if (!on) return;
    size_t free_b = 0, total_b = 0;
    cudaMemGetInfo(&free_b, &total_b);
    std::fprintf(stderr, "maya trace: %lld MiB free after %s\n", (long long) (free_b >> 20), where);
}

int argmax(const std::vector<float>& v) {
    int best = 0;
    for (size_t i = 1; i < v.size(); ++i)
        if (v[i] > v[best]) best = (int) i;
    return best;
}
namespace {

struct Options {
    std::string glm_pack;             // --glm-pack: a GLM-5.3-Flash pack dir (tools/iq_pack.py output)
    std::vector<int64_t> tokens;      // the prompt, PRE-TOKENIZED
    int64_t max_new = 16;
    int64_t max_context = 4096;
    bool greedy = true;
    uint64_t seed = 0;
    int top_k = 20;
    float top_p = 0.95f;
    float temperature = 1.0f;
    std::string dump_logits;
    std::string dump_routing;         // --dump-routing: tools/make_profile.py's input
    std::string expert_profile;
    std::string expert_profile_save;
    float expert_profile_save_min = 0.0f;
    std::string prefill_arg;          // --prefill (the GLM engine reads it as STRATA_GLM_PREFILL)
    std::string dflash;               // --dflash: a DFlash2 draft GGUF (STRATA_GLM_DFLASH)
    std::string spec_follow, spec_follow_save;   // STRATA_GLM_SPEC_FOLLOW / _SAVE
    int spec = 0;
    float spec_min_p = 0.0f;
    bool spec_min_p_set = false;
    bool stop_eos = false;
    bool eos_given = false;
    std::vector<int64_t> eos_ids;
    std::string layer_split;
    bool prefill_auto = false;
    int64_t prefill_chunk = 0;
    std::string mtp;
    int64_t mtp_window = 32768;
    bool serve = false;
    std::string kv = "fp16";
    int64_t kv_resident = 0;
    std::string kv_streaming;
    int expert_cache = 0;
    int vram_reserve_mib = 0;
    bool vram_reserve_given = false;
    int vram_reserve_later_mib = -1;
};

static void usage() {
    std::fprintf(stderr,
                 "strata-glm --glm-pack DIR [options]\n\n"
                 "  --glm-pack DIR       a GLM-5.3-Flash pack (tools/glm pack; tools/iq_pack.py output):\n"
                 "                       the verified Glm5Model runner (src/core/glm_*), wire contract\n"
                 "                       `T <id>` per token, `ERR <message>`, DONE summary; --serve reads\n"
                 "                       `GEN <max_new> <id,id,...>` lines (QUIT ends)\n"
                 "  --parallel N         n conversations at once, 1..8 (STRATA_GLM_SEQS)\n"
                 "  --max-context N      the KV/state capacity (默认 4096)\n"
                 "  --max-new N          tokens to generate (default 16)\n"
                 "  --kv fp16|int8       the DSA latent cache: int8 is the quantized one (STRATA_GLM_KV_INT8)\n"
                 "  --kv-resident N      KV streaming: keep N cells in VRAM, stream the rest\n"
                 "  --kv-streaming on|off\n"
                 "  --expert-cache auto|N   the VRAM expert pool: auto takes the free VRAM less the reserve\n"
                 "  --vram-reserve-mib N / --vram-reserve-later-mib N\n"
                 "  --expert-profile FILE / --expert-profile-save FILE / --expert-profile-save-every N\n"
                 "  --prefill auto|N|0   the batched prompt chunk (STRATA_GLM_PREFILL)\n"
                 "  --layer-split S      layer split across GPUs (auto | K1,K2,..)\n"
                 "  --dflash GGUF        the DFlash2 drafter  (STRATA_GLM_DFLASH)\n"
                 "  --spec N/--spec-min-p/--spec-follow/--spec-follow-save\n"
                 "  --tokens \"1,2,3\" / --tokens-file PATH (prompt ids)\n"
                 "  --greedy/--seed N/--top-k N/--top-p F/--temperature F\n"
                 "  --serve/--stop-eos/--eos-ids/--dump-logits/--dump-routing\n"
                 "  --mtp on|off / --mtp-window N\n");
}
bool parse_i64_list(const char* s, std::vector<int64_t>& out, std::string& err) {
    out.clear();
    std::string text(s);
    for (char& c : text) if (c == ',') c = ' ';
    std::istringstream input(text);
    std::string token;
    while (input >> token) {
        int32_t id = 0;
        const auto parsed = std::from_chars(token.data(), token.data() + token.size(), id);
        if (parsed.ec != std::errc{} || parsed.ptr != token.data() + token.size() || id < 0) {
            err = "invalid token id: expected an integer in [0, 2147483647]";
            out.clear();
            return false;
        }
        out.push_back(id);
    }
    if (out.empty()) { err = "token list was empty"; return false; }
    return true;
}

// ================================ the glm5-next pack path (M3.1) ================================
//
// A GLM-5.3-Flash pack (tools/iq_pack.py output) runs the VERIFIED runner - Glm5Model, the same
// path glm_pack_test drives (docs/GLM5-FLASH.md §9) - behind this driver's wire contract:
// `T <id>` per generated token on stdout, `ERR <message>` when a request cannot run, the DONE
// summary at the end, and --serve reading `GEN <max_new> <id,id,...>` lines (QUIT ends).
//
// NOT wired here yet, and refused rather than approximated: prompt-prefix reuse and per-request
// state continuity across GEN lines (each GEN starts a fresh sequence), drafts,
// the batched prompt path (the prompt runs one token at a time), images, control vectors and
// layer splits.  Sampling is the engine's own sampler (strata::kernels::sample_tokens, greedy by
// default), so the top-k/top-p/temperature chain behaves exactly as in the qwen4exp path - the
// penalties too (#273): the request's history (its prompt, then what it wrote) rides in
// SamplerParams::host_history, and every pick counts its last penalty_last_n tokens
// (Glm5Model::pick_tokens; a verify window's row t: those, then the drafts before it).
// The stop set is GLM's own, from the GGUF metadata (the ids llama.cpp marks EOG): 154820
// <|endoftext|> (eos), 154827 <|user|> (eot, the assistant yields the floor) and 154829 <|observation|> (a tool
// call waits for its result); --eos-ids
// overrides it.  Serve stops on them by default; non-serve stops only when asked (--stop-eos or
// --eos-ids), like the qwen path.
// a variable of this process's environment (the GLM model reads its switches from there)
static void set_env(const char* k, const char* v) {
#if defined(_WIN32)
    _putenv_s(k, v);
#else
    setenv(k, v, 1);
#endif
}

// A failure in the request's own work - an allocation, a launch - is thrown (strata/prefill/failure.hpp) and
// caught here: it ends this request with an error, not the process.  The engine lives and the next request runs.
template <typename F>
static bool guarded(F&& f, std::string& err) {
    try {
        return f();
    } catch (const std::exception& e) {
        err = e.what();
        return false;
    }
}

// Whether the CUDA context itself is broken (a sticky error): nothing else can run, so the engine has to end -
// the server starts a new one.  Every other failure is the request's.
static bool cuda_context_broken(const std::string& why) {
    static const char* kSticky[] = {"illegal memory access", "device-side assert", "unspecified launch failure",
                                    "device uninitialized", "cuda context is broken"};
    std::string low;
    low.reserve(why.size());
    for (char c : why) low.push_back((char) std::tolower((unsigned char) c));
    for (const char* p : kSticky)
        if (low.find(p) != std::string::npos) return true;
    return false;
}

static int glm_pack_generate(const Options& o) {
#if !defined(_WIN32)
    // serving: the server ends the engine with QUIT (or by closing its input), and the engine then sets the conversation
    // aside (STRATA_GLM_SLOT_KEEP).  A SIGTERM to the whole process group (systemd stopping a service, a supervisor
    // killing the server's group) must not end it mid-write; a server that is already gone must not end it with
    // SIGPIPE when it writes its last lines.
    if (o.serve) {
        std::signal(SIGTERM, SIG_IGN);
        std::signal(SIGINT, SIG_IGN);
        std::signal(SIGPIPE, SIG_IGN);
    }
#endif
    strata::core::Glm5Model model;
    std::string err;
    // the prompt chunk, set in the config's args like Strata's: --prefill auto | N (a chunk of N tokens) | 0
    // (the engine reads it as STRATA_GLM_PREFILL; without either, auto)
    if (!o.prefill_arg.empty()) set_env("STRATA_GLM_PREFILL", o.prefill_arg.c_str());
    // the DFlash2 drafter (--dflash, else STRATA_GLM_DFLASH): loaded with the model, on the device of its head
    if (!o.dflash.empty()) set_env("STRATA_GLM_DFLASH", o.dflash.c_str());
    // --spec-follow / --spec-follow-save (else STRATA_GLM_SPEC_FOLLOW / _SAVE): read below, once the vocabulary is known
    if (!o.spec_follow.empty()) set_env("STRATA_GLM_SPEC_FOLLOW", o.spec_follow.c_str());
    if (!o.spec_follow_save.empty()) set_env("STRATA_GLM_SPEC_FOLLOW_SAVE", o.spec_follow_save.c_str());
    // the expert profile (Strata's profile.bin: setup passes data/expert-profile-glm.bin) and the routing trace
    // (--dump-routing: tools/make_profile.py's input), before the load reads the one and the first prompt the other
    if (!o.expert_profile.empty()) model.set_expert_profile(o.expert_profile);
    if (!o.dump_routing.empty() && !strata::core::Glm5Model::set_routing_dump(o.dump_routing, err)) {
        std::fprintf(stderr, "maya generate: --dump-routing: %s\n", err.c_str());
        return 1;
    }
    // the layer split across the visible GPUs: STRATA_GLM_SPLIT, else --layer-split (auto | K1,K2,..)
    if (!model.load_pack_env(o.glm_pack, o.max_context, err, o.layer_split)) {
        std::fprintf(stderr, "maya generate: %s\n", err.c_str());
        return 1;
    }
    const int64_t n_vocab = model.geometry().n_vocab;
    // the driver samples on the device (sample_token), so the fast path never downloads the logits
    model.set_host_logits(false);
    if (model.fast()) {
        // INFO facts for the server's Monitor tab (strata app): the expert tiers this engine runs with
        const auto st = model.fast_stats();
        // (+ the CPU lane setup's calibration reads: its threads and its PCIe share, as the engine started)
        std::printf("INFO context=%lld kv=%s kv_resident=%lld expert_slots=%lld expert_cache_mib=%lld engine_kind=glm-fast "
                    "experts=%d vram_slots=%lld vram_gb=%.1f ram_slots=%lld ram_gb=%.1f mtp=%d dflash=%d vision_lend=%zu "
                    "cpu_threads=%d pcie_share=%.2f logit_bias=1\n",
                    (long long) o.max_context, model.kv_int8() ? "int8" : "fp16", (long long) model.kv_resident(),
                    (long long) st.pool_slots,
                    (long long) (st.pool_gb * 1024.0), model.geometry().n_expert * (model.geometry().n_layers - model.geometry().dense_lead),
                    (long long) st.pool_slots, st.pool_gb, (long long) st.ram_slots, st.ram_gb, model.has_mtp() ? 1 : 0,
                    model.dflash_ready() ? 1 : 0, model.vision_lend_bytes(), model.cpu_lane_threads(), model.pcie_share());
        // the Monitor's map of each GPU: what each part holds on its card, in bytes (Glm5Model::VramReport) - the
        // server finds the card NVML numbers by its PCI address and adds what NVML sees besides
        for (const auto& r : model.vram_report())
            std::printf("VRAM dev=%d pci=%s total=%llu free=%llu weights=%llu kv=%llu kv_ram=%llu act=%llu drafter=%llu "
                        "window=%llu pool=%llu tail=%llu borrow=%llu vision=%llu slots=%lld ctx=%lld layers=%d-%d\n",
                        r.dev, r.pci.empty() ? "-" : r.pci.c_str(), (unsigned long long) r.total,
                        (unsigned long long) r.free, (unsigned long long) r.weights, (unsigned long long) r.kv,
                        (unsigned long long) r.kv_ram, (unsigned long long) r.act, (unsigned long long) r.drafter,
                        (unsigned long long) r.window, (unsigned long long) r.pool, (unsigned long long) r.tail,
                        (unsigned long long) r.borrow, (unsigned long long) r.vision, (long long) r.slots,
                        (long long) r.ctx, r.l0, r.l1);
        // (Windows on an APU: the commit - the page file - kept experts off the GPU; the server's log, Monitor and
        // /v1/status say so, as for DeepSeek's engine)
        if (const auto cc = model.commit_capped(); cc.first > 0) {
            std::printf("NOTE experts: %.0f GB stay off the GPU - Windows' page file is too small to hold them there; "
                        "a page file of %zu GB keeps them on the GPU (System > About > Advanced system settings > "
                        "Performance > Advanced > Virtual memory), then restart Maya\n", (double) cc.first / 1e9, cc.second);
            std::printf("INFO gpu_commit_capped_gb=%.0f pagefile_hint_gb=%zu\n", std::ceil((double) cc.first / 1e9),
                        cc.second);
        }
    }

    strata::kernels::SamplerParams sp;
    sp.greedy = o.greedy;
    sp.top_k = o.top_k;
    sp.top_p = o.top_p;
    sp.temperature = o.temperature;
    sp.seed = o.seed;

    // The model's stop contract, in one place: the three template ends plus the tool-call closer, so a
    // finished call ends the turn as a natural stop on every path, with no --eos-ids anywhere.  The flag
    // still overrides (evals pass 2147483647 to disable eos entirely).
    const std::vector<int64_t> glm_eos = {154820, 154827, 154829, 154844};
    const std::vector<int64_t>& eos = o.eos_given ? o.eos_ids : glm_eos;
    // STRATA_GLM_KL_DUMP=<file>,<n> (a test of the K/V cache formats): --tokens-file's tokens teacher-forced - all but
    // the last n through the prompt path, those n one at a time through the decode - and the logits after each of the
    // n into file (n x n_vocab F32: row i predicts token size - n + i + 1)
    if (const char* kd = std::getenv("STRATA_GLM_KL_DUMP")) {
        const std::string spec(kd);
        const size_t c = spec.rfind(',');
        const std::string path = spec.substr(0, c);
        const int64_t n = c == std::string::npos ? 64 : std::atoll(spec.c_str() + c + 1);
        const auto& t = o.tokens;
        if (n <= 0 || (int64_t) t.size() <= n) {
            std::fprintf(stderr, "maya generate: STRATA_GLM_KL_DUMP needs more than %lld tokens\n", (long long) n);
            return 2;
        }
        model.set_host_logits(true);
        std::string err;
        std::vector<float> lg;
        const std::vector<int32_t> head(t.begin(), t.end() - n);
        const bool pok = model.fast() && model.prefill(head, err, (int32_t) t[t.size() - (size_t) n]);
        if (!pok && (!err.empty() || !model.forward(head, lg, err))) {
            std::fprintf(stderr, "maya generate: %s\n", err.c_str());
            return 1;
        }
        std::FILE* f = std::fopen(path.c_str(), "wb");
        if (f == nullptr) {
            std::fprintf(stderr, "maya generate: cannot write %s\n", path.c_str());
            return 1;
        }
        for (size_t i = t.size() - (size_t) n; i < t.size(); ++i) {
            if (!model.forward({(int32_t) t[i]}, lg, err) || (int64_t) lg.size() < n_vocab) {
                std::fprintf(stderr, "maya generate: %s\n", err.empty() ? "no logits" : err.c_str());
                std::fclose(f);
                return 1;
            }
            std::fwrite(lg.data(), sizeof(float), (size_t) n_vocab, f);
        }
        std::fclose(f);
        std::fprintf(stderr, "maya generate: %lld positions' logits in %s\n", (long long) n, path.c_str());
        return 0;
    }

    const bool stop_on_eos = o.eos_given || o.stop_eos || o.serve;

    // ---- stdin on its own thread (the qwen serve's pattern): a STOP line must reach a request that is
    // still running.  read(2), not std::cin - glibc's exit() flushes stdio and waits on stdin's lock,
    // which getline holds while it waits for input.
    std::atomic<bool> stop_req{false}, quit_req{false};
    // WRAP (the server is stopping): a request still thinking closes its reasoning at the next token and answers
    std::atomic<bool> wrap_req{false};
    // PAUSE (the server's fair slices): STOP, and the conversation's state is kept with what the request wrote (the
    // snapshot moves to the last token fed), so the request that continues it reads only its last token
    std::atomic<bool> pause_req{false}, pause_keep{false};
    // several conversations at once (STRATA_GLM_SEQS): STOP <rid> / PAUSE <rid> [keep] for one request; the request
    // whose prompt is being read stops at the next chunk
    std::mutex ctl_mu;
    std::map<int, int> ctl;   // rid -> 1 STOP, 2 PAUSE, 3 PAUSE keep
    std::atomic<int> admitting_rid{-1};
    // STRATA_GLM_PREFILL_INTERLEAVE (strata/core/prefill_pieces.hpp): the other requests decode between the pieces of
    // a prompt being read, so a STOP / PAUSE for the request being admitted stops its read alone (admit_stop) - not
    // stop_req, which every decoding request reads as its own.  Off: stop_req as before (nothing decodes meanwhile)
    static const strata::core::PrefillInterleave interleave =
        strata::core::prefill_interleave_parse(std::getenv("STRATA_GLM_PREFILL_INTERLEAVE"));
    // STRATA_GLM_PREFILL_GAPS (strata/core/prefill_gaps.hpp): the other requests decode at the chunk boundaries of a
    // prompt being read (one call, its chunks pipelined as without) - like interleave, a STOP for the request being
    // read stops its read alone.  Over interleave when both are set.
    static const strata::core::PrefillGaps read_gaps =
        strata::core::prefill_gaps_parse(std::getenv("STRATA_GLM_PREFILL_GAPS"));
    std::atomic<bool> admit_stop{false};
    const auto read_stopped = [&] { return stop_req.load() || admit_stop.load(); };
    std::mutex in_mu;
    std::condition_variable in_cv;
    std::deque<std::string> in_lines;
    bool in_eof = false;
    if (o.serve) {
        std::thread([&] {
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
                if (l == "STOP") { stop_req.store(true); continue; }
                if (l.rfind("STOP ", 0) == 0 || (l.rfind("PAUSE ", 0) == 0 && l != "PAUSE keep")) {
                    std::istringstream ls(l);
                    std::string c0, kw;
                    int rid = -1;
                    ls >> c0 >> rid;
                    // (no request id there: nothing to stop - a negative one would match admitting_rid's idle -1 and
                    // set the global stop, which every running request reads as its own)
                    if (ls.fail() || rid < 0) continue;
                    ls >> kw;
                    {
                        std::lock_guard<std::mutex> lk(ctl_mu);
                        ctl[rid] = c0 == "STOP" ? 1 : kw == "keep" ? 3 : 2;
                    }
                    if (rid == admitting_rid.load()) {
                        admit_stop.store(true);
                        if (interleave.tokens == 0 && read_gaps.percent == 0) stop_req.store(true);
                    }
                    continue;
                }
                if (l == "PAUSE" || l == "PAUSE keep") {   // keep: the prompt's state is set aside first
                    pause_keep.store(l == "PAUSE keep");
                    pause_req.store(true);
                    stop_req.store(true);
                    continue;
                }
                if (l == "WRAP") { wrap_req.store(true); continue; }
                if (l == "QUIT") {                       // cancel an in-flight request AND queue the line:
                    quit_req.store(true);
                    stop_req.store(true);                // (the decode checks stop_req; reset by the next GEN)
                }
                std::lock_guard<std::mutex> lk(in_mu);   // a GEN written before QUIT must still run (pipes)
                in_lines.push_back(l);
                in_cv.notify_one();
            }
            std::lock_guard<std::mutex> lk(in_mu);
            in_eof = true;
            in_cv.notify_one();
        }).detach();
    }
    const auto next_line = [&](std::string& out) -> bool {
        model.lane_drift_check();
        model.lane_retime();
        std::unique_lock<std::mutex> lk(in_mu);
        in_cv.wait(lk, [&] { return !in_lines.empty() || in_eof || quit_req.load(); });
        if (in_lines.empty()) return false;   // EOF or QUIT with nothing queued behind it
        out = std::move(in_lines.front());
        in_lines.pop_front();
        return true;
    };

    std::vector<int32_t> snap_tokens;   // the prompt prefix the model's saved state belongs to (conversation reuse)
    // images (GENI): this request's <|image|> positions and rows, and a hash of the rows the saved state read - the
    // same tokens with another picture must not reuse it
    std::vector<int64_t> cur_img_pos;
    std::vector<float> cur_img_rows;
    // the thinking budget (GEN keys think_budget=<n> think_end=<id>, the server's run config): a reasoning block still
    // open after n tokens is closed - the next token is </think> instead of the sampled one - and the answer follows
    // in the same decode
    int64_t req_think_budget = 0;
    int32_t req_think_end = -1;
    // the request's logit_bias (GEN key logit_bias=<id>:<bias>,...): added to its logits before every sample
    std::vector<std::pair<int32_t, float>> req_bias;
    uint64_t snap_img_hash = 1469598103934665603ull;
    const auto img_hash = [&](size_t len) {
        uint64_t h = 1469598103934665603ull;   // FNV-1a over the positions and rows inside [0, len)
        const size_t E = (size_t) model.geometry().n_embd;
        for (size_t k = 0; k < cur_img_pos.size() && (size_t) cur_img_pos[k] < len; ++k) {
            const unsigned char* b = (const unsigned char*) (cur_img_rows.data() + k * E);
            for (size_t j = 0; j < E * sizeof(float); ++j) h = (h ^ b[j]) * 1099511628211ull;
            h = (h ^ (uint64_t) cur_img_pos[k]) * 1099511628211ull;
        }
        return h;
    };
    // CONVERSATION SLOTS (the fast path): a request that does not continue the conversation in the model first sets
    // that conversation aside in a file (Glm5Model::slot_save: its snapshot and DSA caches), then takes back the slot
    // it does continue (slot_load) - chats, and an agent's side requests, that interleave no longer re-read each
    // other's prompts.  STRATA_GLM_SLOTS=<n> files (default 4, 0 off), for conversations of STRATA_GLM_SLOT_MIN tokens
    // or more (1024), in STRATA_GLM_SLOT_DIR (default <pack>/slots, emptied at start), at most STRATA_GLM_SLOT_GB on
    // disk (16) and never below 8 GB free.  Text only: a conversation with pictures is not set aside.  Or the slots
    // live in RAM and nothing reaches the disk: STRATA_GLM_SLOT_RAM_GB (0, off) caps their bytes there - the states
    // are host buffers that die with the run, so SLOT_DIR, SLOT_GB (1) and the disk-space guard do not apply.
    struct Slot {
        std::vector<int32_t> tokens;
        std::string path;                       // empty when the slot is a RAM buffer (slot_ram)
        uint64_t bytes = 0, used = 0;
        std::shared_ptr<std::vector<uint8_t>> data;   // the RAM slot's bytes (null: a file slot)
        // a paused request's prompt (PAUSE keep): the next turn of that chat continues it, not the written answer
        // (its history drops the reasoning), so its own continuations do not supersede it until it is taken back
        bool anchor = false;
        // the end of a system prompt (slot_prefix): other chats start with it, so its conversation's later states
        // never supersede it - it goes only as the least recently used (#179)
        bool prefix = false;
    };
    // (static: the multi path's Ctx, a local struct, uses it as a default member value - MSVC accepts that of a static
    // only, and a local constant left Ctx without a default constructor: "std::construct_at: no matching function")
    static constexpr uint64_t kNoImages = 1469598103934665603ull;   // img_hash() of a prefix without pictures
    std::vector<Slot> slots;
    uint64_t slot_clock = 0, slot_seq = 0;
    bool snap_in_slot = false;   // the model's snapshot is one of the slots, unchanged since
    const auto env_num = [](const char* k, double d) { const char* v = getenv(k); return v ? std::atof(v) : d; };
    int slot_max = model.fast() && getenv("STRATA_GLM_NO_REUSE") == nullptr ? (int) env_num("STRATA_GLM_SLOTS", 4) : 0;
    const size_t slot_min = (size_t) std::max(1.0, env_num("STRATA_GLM_SLOT_MIN", 1024));
    const uint64_t slot_budget = (uint64_t) (env_num("STRATA_GLM_SLOT_GB", 16) * 1073741824.0);
    // slots in RAM: the conversation's state is parked in a host buffer instead of a file.  The cap moves to RAM too
    // (slot_cap below); the states die with the run, so SLOT_KEEP, SLOT_DAYS, SLOT_DIR and the 8 GB-free disk guard
    // are theirs no longer.  Nothing on the disk is touched: in RAM mode the slots folder is neither used nor wiped.
    const bool slot_ram = env_num("STRATA_GLM_SLOT_RAM_GB", 0) > 0;
    const uint64_t slot_cap =
        slot_ram ? (uint64_t) (env_num("STRATA_GLM_SLOT_RAM_GB", 16) * 1073741824.0) : slot_budget;
    // STRATA_GLM_SLOT_KEEP=1: the slots outlive the engine - the folder is not emptied at start (its files are indexed
    // and kept when they belong to this model), the conversation in the model is set aside when the engine ends
    // (QUIT or end of input), and a file not used for STRATA_GLM_SLOT_DAYS days (7) is removed.  Every slot then has a
    // .meta beside it: this model's identity, the conversation's tokens and the slot's size; a slot whose .meta is
    // missing, foreign or does not match its file is deleted.  Both files are written under a temporary name first.
    const bool slot_keep = getenv("STRATA_GLM_SLOT_KEEP") != nullptr && std::atoi(getenv("STRATA_GLM_SLOT_KEEP")) != 0;
    const double slot_days = env_num("STRATA_GLM_SLOT_DAYS", 7);
    namespace fs = std::filesystem;
    std::filesystem::path slot_dir;
    // what a kept slot must match: the session format, the engine version, the pack and the model's shards (names and
    // sizes) and whether the draft block is loaded.  The split itself is checked by slot_load (each part's layers).
    const std::string slot_identity = [&] {
        std::ostringstream id;
        id << "maya-session-1 engine=" << STRATA_VERSION << " mtp=" << (model.has_mtp() ? 1 : 0);
        std::error_code ec;
        for (const char* f : {"index.txt", "dense.bin", "native_experts.txt"})
            id << " " << f << ":" << (unsigned long long) fs::file_size(fs::path(o.glm_pack) / f, ec);
        std::vector<std::string> shards;
        for (const auto& de : fs::directory_iterator(fs::path(o.glm_pack).parent_path(), ec))
            if (de.path().extension() == ".gguf")
                shards.push_back(de.path().filename().string() + ":" + std::to_string(de.file_size(ec)));
        std::sort(shards.begin(), shards.end());
        for (const auto& s : shards) id << " " << s;
        return id.str();
    }();
    constexpr uint64_t kMetaMagic = 0x3153534559414d47ull;   // "GMAYESS1"
    const auto meta_of = [](const std::string& bin) { return bin.substr(0, bin.size() - 4) + ".meta"; };
    const auto meta_write = [&](const std::string& path, const std::vector<int32_t>& toks, uint64_t bytes) -> bool {
        const std::string tmp = path + ".tmp";
        {
            std::ofstream f(tmp, std::ios::binary | std::ios::trunc);
            const uint64_t il = slot_identity.size(), nt = toks.size();
            f.write((const char*) &kMetaMagic, 8);
            f.write((const char*) &il, 8);
            f.write(slot_identity.data(), (std::streamsize) il);
            f.write((const char*) &bytes, 8);
            f.write((const char*) &nt, 8);
            f.write((const char*) toks.data(), (std::streamsize) (nt * sizeof(int32_t)));
            f.write((const char*) &kMetaMagic, 8);
            f.flush();
            if (!f) return false;
        }
        std::error_code ec;
        fs::rename(tmp, path, ec);
        return !ec;
    };
    // a .meta of this model -> its tokens and the slot's size (false: foreign, damaged or not this model's)
    const auto meta_read = [&](const std::string& path, std::vector<int32_t>& toks, uint64_t& bytes) -> bool {
        std::ifstream f(path, std::ios::binary);
        uint64_t m0 = 0, il = 0, nt = 0, m1 = 0;
        if (!f.read((char*) &m0, 8) || m0 != kMetaMagic || !f.read((char*) &il, 8) || il != slot_identity.size())
            return false;
        std::string id(il, '\0');
        if (!f.read(id.data(), (std::streamsize) il) || id != slot_identity) return false;
        if (!f.read((char*) &bytes, 8) || !f.read((char*) &nt, 8) || nt == 0 || nt > (uint64_t) o.max_context) return false;
        toks.resize(nt);
        return (bool) f.read((char*) toks.data(), (std::streamsize) (nt * sizeof(int32_t))) && f.read((char*) &m1, 8) &&
               m1 == kMetaMagic;
    };
    const auto slot_age_days = [](const std::string& p) {
        std::error_code ec;
        const auto t = fs::last_write_time(p, ec);
        if (ec) return 1e9;
        return std::chrono::duration<double>(fs::file_time_type::clock::now() - t).count() / 86400.0;
    };
    // NOTE lines on stdout: the server shows them in its own log (what happens to the conversations)
    const auto note = [&](const std::string& m) {
        std::fprintf(stderr, "glm slots: %s\n", m.c_str());
        if (o.serve) std::printf("NOTE sessions: %s\n", m.c_str());
    };
    const auto gb = [](uint64_t b) {
        char s[32];
        std::snprintf(s, sizeof s, "%.2f GB", (double) b / 1e9);
        return std::string(s);
    };
    if (slot_max > 0 && slot_ram)
        note("slots parked in RAM (" + gb(slot_cap) + " max) - nothing is written to disk, the states die with the run");
    if (slot_max > 0 && !slot_ram) {
        slot_dir = getenv("STRATA_GLM_SLOT_DIR") ? std::filesystem::path(getenv("STRATA_GLM_SLOT_DIR"))
                                                 : std::filesystem::path(o.glm_pack) / "slots";
        std::error_code ec;
        std::filesystem::create_directories(slot_dir, ec);
        struct Found { std::string bin; std::vector<int32_t> toks; uint64_t bytes; fs::file_time_type t; };
        std::vector<Found> found;
        size_t refused = 0;
        for (const auto& de : std::filesystem::directory_iterator(slot_dir, ec)) {   // a previous run's slots
            const std::string fn = de.path().filename().string(), p = de.path().string();
            if (fn.rfind("slot-", 0) != 0) continue;
            const bool is_bin = fn.size() > 9 && fn.compare(fn.size() - 4, 4, ".bin") == 0;
            const bool is_meta = fn.size() > 10 && fn.compare(fn.size() - 5, 5, ".meta") == 0;
            if (!slot_keep || (!is_bin && !is_meta)) {   // not kept, or a .tmp an interrupted write left
                std::filesystem::remove(de.path(), ec);
                continue;
            }
            if (!is_meta) continue;                      // a .bin is judged by its .meta
            const std::string bin = p.substr(0, p.size() - 5) + ".bin";
            Found fd{bin, {}, 0, fs::last_write_time(de.path(), ec)};
            const uint64_t have = fs::exists(bin, ec) ? (uint64_t) fs::file_size(bin, ec) : 0;
            if (meta_read(p, fd.toks, fd.bytes) && have == fd.bytes && fd.bytes > 0 && slot_age_days(p) <= slot_days) {
                found.push_back(std::move(fd));
            } else {
                ++refused;
                std::filesystem::remove(p, ec);
                std::filesystem::remove(bin, ec);
            }
        }
        if (slot_keep)                                   // a .bin without its .meta was never completed
            for (const auto& de : std::filesystem::directory_iterator(slot_dir, ec)) {
                const std::string p = de.path().string(), fn = de.path().filename().string();
                if (fn.rfind("slot-", 0) == 0 && fn.size() > 9 && fn.compare(fn.size() - 4, 4, ".bin") == 0 &&
                    !fs::exists(meta_of(p), ec))
                    std::filesystem::remove(de.path(), ec);
            }
        std::sort(found.begin(), found.end(), [](const Found& a, const Found& b) { return a.t < b.t; });
        uint64_t total = 0;
        for (auto& fd : found) {
            total += fd.bytes;
            slots.push_back({std::move(fd.toks), fd.bin, fd.bytes, ++slot_clock});
            // a system prompt's state (slot_prefix) goes only as the least recently used, after a restart too: the
            // .meta has no flag for it, its tokens tell (no user turn)
            slots.back().prefix = strata::core::is_system_prefix(slots.back().tokens, 154827);
        }
        if (slot_keep) {
            std::ostringstream m;
            m << "folder " << slot_dir.string() << ": " << slots.size() << " conversation(s) kept from earlier runs ("
              << gb(total) << ")";
            if (refused) m << ", " << refused << " not this model's or incomplete - removed";
            note(m.str());
        }
        if (!std::filesystem::is_directory(slot_dir, ec)) {
            std::fprintf(stderr, "glm slots: %s is not writable - conversations are not set aside\n",
                         slot_dir.string().c_str());
            slot_max = 0;
        }
    }
    const auto slot_drop = [&](size_t i) {
        if (slots[i].data)                       // a RAM slot: its buffer simply dies
            slots[i].data.reset();
        else {
            std::error_code ec;
            std::filesystem::remove(slots[i].path, ec);
            std::filesystem::remove(meta_of(slots[i].path), ec);
        }
        slots.erase(slots.begin() + (long) i);
    };
    const auto slot_lru = [&](size_t keep) {   // the least recently used slot other than `keep`
        size_t lru = SIZE_MAX;
        for (size_t i = 0; i < slots.size(); ++i)
            if (i != keep && (lru == SIZE_MAX || slots[i].used < slots[lru].used)) lru = i;
        return lru;
    };
    // the conversation in the model (its snapshot, snap_tokens) into a slot
    const auto slot_put = [&](const char* why) {
        if (slot_max <= 0 || snap_in_slot || snap_tokens.size() < slot_min || snap_img_hash != kNoImages) return;
        for (size_t i = slots.size(); i-- > 0;)   // an earlier state of this conversation is superseded
            if (!slots[i].anchor && !slots[i].prefix && slots[i].tokens.size() <= snap_tokens.size() &&
                std::equal(slots[i].tokens.begin(), slots[i].tokens.end(), snap_tokens.begin()))
                slot_drop(i);
        while (!slots.empty() && slots.size() >= (size_t) slot_max) slot_drop(slot_lru(SIZE_MAX));
        if (slot_ram) {   // the conversation is parked in a host buffer; nothing touches the disk
            const auto t0 = std::chrono::steady_clock::now();
            std::string e;
            auto buf = std::make_shared<std::vector<uint8_t>>();
            buf->reserve(1 << 20);
            const uint64_t bytes = model.slot_save(*buf, e);
            if (bytes == 0) {
                note("setting the conversation aside in RAM failed (" + (e.empty() ? std::string("it did not copy") : e) +
                     ") - it is not parked and will be read from token 0 when it comes back");
                return;
            }
            slots.push_back({snap_tokens, "", bytes, ++slot_clock});
            slots.back().data = buf;
            snap_in_slot = true;
            uint64_t total = 0;
            for (const auto& s : slots) total += s.bytes;
            size_t evicted = 0;
            while (slots.size() > 1 && total > slot_cap) {
                const size_t i = slot_lru(slots.size() - 1);
                total -= slots[i].bytes;
                slot_drop(i);
                ++evicted;
            }
            char m[288];
            std::snprintf(m, sizeof m, "saved a %zu-token conversation%s (%s in RAM in %.1f s); %zu kept, %s%s",
                          snap_tokens.size(), why, gb(bytes).c_str(),
                          std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count(), slots.size(),
                          gb(total).c_str(), evicted ? ", least recently used dropped to stay under the RAM cap" : "");
            note(m);
            return;
        }
        std::error_code ec;
        const auto space = std::filesystem::space(slot_dir, ec);
        if (!ec && space.available < (uint64_t) 8 << 30) {
            std::fprintf(stderr, "glm slots: under 8 GB free in %s - the conversation is not set aside\n",
                         slot_dir.string().c_str());
            return;
        }
        const auto t0 = std::chrono::steady_clock::now();
        const auto stamp = std::chrono::duration_cast<std::chrono::milliseconds>(
                               std::chrono::system_clock::now().time_since_epoch()).count();
        const std::string path = (slot_dir / ("slot-" + std::to_string(stamp) + "-" + std::to_string(++slot_seq) +
                                              ".bin")).string();
        std::string e;
        const uint64_t bytes = model.slot_save(path + ".tmp", e);
        if (bytes == 0 || (std::filesystem::rename(path + ".tmp", path, ec), ec) ||
            !meta_write(meta_of(path), snap_tokens, bytes)) {
            std::filesystem::remove(path + ".tmp", ec);
            std::filesystem::remove(path, ec);
            std::fprintf(stderr, "glm slots: setting the conversation aside failed (%s)\n",
                         e.empty() ? "writing its files" : e.c_str());
            return;
        }
        slots.push_back({snap_tokens, path, bytes, ++slot_clock});
        snap_in_slot = true;
        uint64_t total = 0;
        for (const auto& s : slots) total += s.bytes;
        size_t evicted = 0;
        while (slots.size() > 1 && total > slot_cap) {
            const size_t i = slot_lru(slots.size() - 1);
            total -= slots[i].bytes;
            slot_drop(i);
            ++evicted;
        }
        char m[256];
        std::snprintf(m, sizeof m, "saved a %zu-token conversation%s (%s in %.1f s); %zu kept, %s%s",
                      snap_tokens.size(), why, gb(bytes).c_str(),
                      std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count(), slots.size(),
                      gb(total).c_str(), evicted ? ", least recently used removed to stay under the size cap" : "");
        note(m);
    };
    // the longest slot this prompt continues, back in the model: the tokens it covers (0: none)
    const auto slot_take = [&](const std::vector<int32_t>& prompt) -> size_t {
        if (slot_max <= 0 || !cur_img_pos.empty()) return 0;
        size_t best = SIZE_MAX, len = 0;
        for (size_t i = 0; i < slots.size(); ++i) {
            const auto& t = slots[i].tokens;
            if (t.size() > len && t.size() < prompt.size() && std::equal(t.begin(), t.end(), prompt.begin())) {
                best = i;
                len = t.size();
            }
        }
        if (best == SIZE_MAX) return 0;
        const auto t0 = std::chrono::steady_clock::now();
        std::string e;
        const bool back = slots[best].data
                              ? model.slot_load(*slots[best].data, (int64_t) len, e)
                              : model.slot_load(slots[best].path, (int64_t) len, e);
        if (!back) {
            note("taking a conversation back failed (" + e + ") - reading its prompt again");
            slot_drop(best);
            snap_tokens.clear();
            return 0;
        }
        snap_tokens = slots[best].tokens;
        snap_img_hash = kNoImages;
        snap_in_slot = true;
        slots[best].used = ++slot_clock;
        slots[best].anchor = false;
        if (slot_keep && !slots[best].data) {   // its age counts from its last use (a file slot's only)
            std::error_code ec;
            fs::last_write_time(meta_of(slots[best].path), fs::file_time_type::clock::now(), ec);
        }
        char m[160];
        std::snprintf(m, sizeof m, "took back a %zu-token conversation from %s (%.1f s); %zu new tokens to read", len,
                      slots[best].data ? "RAM" : "disk",
                      std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count(),
                      prompt.size() - len);
        note(m);
        return len;
    };
    // a request's DFlash window and draft cut (its GEN line's spec= / spec_min_p=; 0 / -1: as the engine started)
    int req_spec = 0;
    double req_spec_min_p = -1.0;
    int req_spec_n_min = -1;       // ... its drafts kept whatever their confidence (spec_n_min=; -1: the engine's)
    double req_spec_cum = -1.0;    // ... and the cut on the product of the confidences (spec_cum=; -1: the engine's)
    // THE PROMPT of a request (run_request, and the several-conversation scheduler's admission): the conversation
    // reuse (the snapshot, else a slot), the prompt path for all but the last token, the token path for the rest - the
    // state saved before the last token and the first sample's logits on the tail.  0, or 1 after an ERR line.
    // `tag` prefixes the PP / ERR lines ("<rid> " with several conversations at once).
    std::string tag;
    // CONVERSATION CHECKPOINTS (Glm5Model::ckpt_save): besides the snapshot (the last prompt but its last token), up to
    // STRATA_GLM_CKPTS (4; 0 = none) states of the conversation in the model at earlier positions, in host RAM (~150 MB
    // each with GLM-5.3-Flash): at the end of the system prompt (the first user turn's marker), at the start of the
    // prompt's last turn (its last user or tool-result marker) and every STRATA_GLM_CKPT_EVERY (16384) fresh prompt
    // tokens - only where at least 512 fresh tokens lie before.  A prompt that leaves the conversation before the
    // snapshot's position - an edited or replaced last message, another chat with the same system prompt and tools, a
    // client that rewrote an older turn - restores the longest checkpoint it starts with and reads only the rest,
    // where it read everything from token 0.  One conversation at a time (not the several-conversation scheduler) and
    // text prompts only; STRATA_GLM_NO_REUSE turns it off with the rest of the reuse.
    struct Ckpt {
        std::vector<int32_t> tokens;   // the conversation's first tokens the checkpoint holds the state after (empty: free)
        uint64_t used = 0;
    };
    std::vector<Ckpt> ckpts;
    uint64_t ckpt_clock = 0;
    const int ckpt_max = model.fast() && model.seq_count() <= 1 && getenv("STRATA_GLM_NO_REUSE") == nullptr
                             ? std::max(0, std::min(16, (int) env_num("STRATA_GLM_CKPTS", 4)))
                             : 0;
    const int64_t ckpt_every = std::max<int64_t>(0, (int64_t) env_num("STRATA_GLM_CKPT_EVERY", 16384));
    // a prompt writes the state from position `from` on: the checkpoints past it no longer hold this conversation
    const auto ckpt_drop_past = [&](size_t from) {
        for (auto& c : ckpts)
            if (c.tokens.size() > from) c.tokens.clear();
    };
    // the positions in [from + 512, end) a prompt read from `from` keeps checkpoints at (see above), ascending
    const auto ckpt_marks = [&](const std::vector<int32_t>& prompt, size_t from, size_t end) {
        constexpr int32_t kUserTok = 154827, kObsTok = 154829;   // <|user|>, <|observation|>
        std::vector<size_t> at;
        if (ckpt_max <= 0 || !cur_img_pos.empty() || end <= from + 512) return at;
        size_t first = 0, last = 0;
        for (size_t i = 1; i < end; ++i)
            if (prompt[i] == kUserTok || prompt[i] == kObsTok) {
                if (first == 0 && prompt[i] == kUserTok) first = i;
                last = i;
            }
        std::vector<size_t> turns;
        for (const size_t p : {first, last})
            if (p >= from + 512 && p < end) turns.push_back(p);
        at = turns;
        // the periodic ones, but none within 512 tokens of a turn boundary (that one holds the same state, more useful)
        if (ckpt_every > 0)
            for (size_t p = from + (size_t) ckpt_every; p < end; p += (size_t) ckpt_every) {
                bool by_turn = false;   // (not `near`: Windows' headers define it away)
                for (const size_t t : turns) by_turn = by_turn || (p + 512 > t && t + 512 > p);
                if (!by_turn) at.push_back(p);
            }
        std::sort(at.begin(), at.end());
        at.erase(std::unique(at.begin(), at.end()), at.end());
        // at most ckpt_max a prompt, the latest kept (the turn's start, then the ones before it)
        if ((int) at.size() > ckpt_max) at.erase(at.begin(), at.end() - ckpt_max);
        // every prompt-path call between them must reach its minimum (STRATA_GLM_PREFILL_MIN, 32: fewer go through the
        // token path, one token at a time): only the piece after the last one may be shorter
        std::vector<size_t> kept;
        for (const size_t p : at)
            if (p >= (kept.empty() ? from : kept.back()) + 64) kept.push_back(p);
        return kept;
    };
    // ... and with the disk slots on, the state at the end of the system prompt (p) also into a slot of its own: the
    // checkpoints live with the model's state, so an unrelated conversation in between drops them, and another chat
    // with this system prompt read it all again (#179); the slot outlives that.  Once per system prompt.
    const auto slot_prefix = [&](const std::vector<int32_t>& prompt, size_t p) {
        if (slot_max <= 0 || p < slot_min || !cur_img_pos.empty()) return;
        for (auto& sl : slots)
            if (sl.prefix && sl.tokens.size() == p && std::equal(sl.tokens.begin(), sl.tokens.end(), prompt.begin())) {
                sl.used = ++slot_clock;
                return;
            }
        if (!model.snapshot_save()) return;
        snap_tokens.assign(prompt.begin(), prompt.begin() + (long) p);
        snap_img_hash = img_hash(p);
        snap_in_slot = false;
        slot_put(" (its system prompt, for other chats that start with it)");
        if (snap_in_slot && !slots.empty() && slots.back().tokens.size() == p) slots.back().prefix = true;
    };
    // the state now (after prompt[0, p)) into a free or the least recently used checkpoint
    const auto ckpt_keep = [&](const std::vector<int32_t>& prompt, size_t p, bool from_snapshot = false) {
        if ((int) ckpts.size() < ckpt_max) ckpts.emplace_back();
        size_t c = 0;
        for (size_t i = 0; i < ckpts.size(); ++i) {
            if (ckpts[i].tokens.empty()) { c = i; break; }
            if (ckpts[i].used < ckpts[c].used) c = i;
        }
        std::string e;
        ckpts[c].tokens.clear();
        if (!(from_snapshot ? model.ckpt_save_snapshot((int) c, e) : model.ckpt_save((int) c, e))) {
            std::fprintf(stderr, "glm checkpoints: not kept at %zu (%s)\n", p, e.c_str());
            return;
        }
        ckpts[c].tokens.assign(prompt.begin(), prompt.begin() + (long) p);
        ckpts[c].used = ++ckpt_clock;
    };
    // A route that found a non-finite score (or an expert id out of range) means a NaN or an infinity in the state the
    // request ran on: its conversation must not be served again.  The kept snapshot and every slot this prompt
    // extends (one may have been the poisoned state's source) are dropped and the model is reset, so the server's
    // retry reads the prompt from token 0.  The ERR line names it "non-finite" (Glm5Model::route_ok's words).
    const auto drop_if_poisoned = [&](const std::vector<int32_t>& prompt, const std::string& why) {
        if (why.find("non-finite") == std::string::npos) return;
        snap_tokens.clear();
        snap_in_slot = false;
        ckpt_drop_past(0);
        for (size_t i = slots.size(); i-- > 0;) {
            const auto& t = slots[i].tokens;
            if (t.size() <= prompt.size() && std::equal(t.begin(), t.end(), prompt.begin())) slot_drop(i);
        }
        model.reset();
        std::fprintf(stderr, "glm nan guard: %s - this conversation's state is dropped; it is read again from token 0\n",
                     why.c_str());
    };
    // set by the several-conversation scheduler while it admits a request (STRATA_GLM_PREFILL_INTERLEAVE): the piece the
    // prompt path reads next (0: the rest of the segment, as before) and what runs between two pieces - 0 go on,
    // 1 stopped (the request's STOP), 2 failed (prompt_between_err says why; every request ended, the state reset)
    std::function<size_t()> prompt_piece;
    std::function<int()> prompt_between;
    std::string prompt_between_err;
    // STRATA_GLM_PREFILL_GAPS: true inside a gap (the decode steps at a chunk boundary of a prompt's read, or between
    // two of its calls) - no tier boundary runs there, nothing binds another context (a PAUSE's snapshot waits for the
    // read's end: deferred_pauses), and an engine error leaves the contexts' reset to gap_reset, after the read has
    // ended (the pool's tail is lent in it); gap_failed: a gap inside a prompt-path call failed (prompt_between_err
    // says why); gaps_over: a gap of this read overran <max gap ms>, so the rest of the read has none
    bool in_read_gap = false, gap_failed = false, read_gap_first = false, gaps_over = false;
    // a request PAUSEd inside a gap: its context stays busy (nothing else binds or reads into it) until its snapshot is
    // taken as it would have been at once - after the read's call, once its first token is taken (flush_pauses)
    struct DeferredPause {
        int ctx = 0;
        std::vector<int32_t> prompt, emitted;
        bool keep = false;
    };
    std::vector<DeferredPause> deferred_pauses;
    std::function<void()> gap_reset;
    std::chrono::steady_clock::time_point gap_mark;   // the read's start (after any restore), then each gap's end
    const auto prep_prompt = [&](const std::vector<int32_t>& prompt, size_t& reuse, const char*& finish,
                                 std::chrono::steady_clock::time_point& t0) -> int {
        // CONVERSATION REUSE (the fast path): the previous prompt's state, saved just before its last token, is
        // restored when this prompt extends it - a chat's next turn reads only what is new
        strata::core::progress_at("the request: restoring a conversation");
        const bool snap_ok = model.fast() && !snap_tokens.empty() && snap_tokens.size() < prompt.size() &&
                             std::equal(snap_tokens.begin(), snap_tokens.end(), prompt.begin()) &&
                             img_hash(snap_tokens.size()) == snap_img_hash && getenv("STRATA_GLM_NO_REUSE") == nullptr;
        // the longest checkpoint this prompt starts with (it holds less than the snapshot when both match)
        int ck = -1;
        if (ckpt_max > 0 && cur_img_pos.empty())
            for (int c = 0; c < (int) ckpts.size(); ++c) {
                const auto& t = ckpts[(size_t) c].tokens;
                if (!t.empty() && t.size() < prompt.size() && (ck < 0 || t.size() > ckpts[(size_t) ck].tokens.size()) &&
                    std::equal(t.begin(), t.end(), prompt.begin()))
                    ck = c;
            }
        const size_t ck_len = ck >= 0 ? ckpts[(size_t) ck].tokens.size() : 0;
        bool from_ckpt = false;
        if (snap_ok && snap_tokens.size() >= ck_len && model.snapshot_restore()) reuse = snap_tokens.size();
        if (reuse == 0 && ck >= 0) {
            slot_put("");              // the conversation in the model goes on differently past the checkpoint: kept
            size_t slot_len = 0;       // a slot that holds more of this prompt is taken instead (below)
            if (slot_max > 0)
                for (const auto& sl : slots)
                    if (sl.tokens.size() > slot_len && sl.tokens.size() < prompt.size() &&
                        std::equal(sl.tokens.begin(), sl.tokens.end(), prompt.begin()))
                        slot_len = sl.tokens.size();
            std::string ce;
            if (slot_len > ck_len) {
            } else if (model.ckpt_restore(ck, ce)) {
                reuse = ck_len;
                from_ckpt = true;
                ckpts[(size_t) ck].used = ++ckpt_clock;
                snap_tokens.clear();   // the snapshot's position lies past the checkpoint: overwritten from here on
                std::fprintf(stderr, "glm checkpoints: %zu of the prompt's %zu tokens from a checkpoint\n", reuse,
                             prompt.size());
            } else {
                std::fprintf(stderr, "glm checkpoints: %s\n", ce.c_str());
                ckpts[(size_t) ck].tokens.clear();
            }
        }
        // THE TURN'S START: a prompt that goes on from the snapshot and opens its last turn (<|user|> / <|observation|>)
        // within 512 tokens of it gets no checkpoint there from ckpt_marks - so an edit of that turn's message read the
        // whole conversation again (13.5K tokens: 20.8 s).  The snapshot itself (the previous prompt but its last
        // token, a few tokens before the turn) is kept as a checkpoint instead: copied to host RAM on a side thread
        // while this prompt runs, so it costs the prompt nothing.  STRATA_GLM_CKPT_TURN=0: off.
        if (reuse > 0 && !from_ckpt && reuse == snap_tokens.size() && reuse >= 512 && ckpt_max > 0 &&
            cur_img_pos.empty()) {
            static const bool turn_on = [] { const char* v = getenv("STRATA_GLM_CKPT_TURN"); return !v || atoi(v) != 0; }();
            constexpr int32_t kUserTok = 154827, kObsTok = 154829;   // <|user|>, <|observation|>
            size_t last = 0;
            for (size_t i = prompt.size() - 1; i-- > reuse;)
                if (prompt[i] == kUserTok || prompt[i] == kObsTok) { last = i; break; }
            bool have = false;
            for (const auto& c : ckpts)
                have = have || (c.tokens.size() == reuse && std::equal(c.tokens.begin(), c.tokens.end(), prompt.begin()));
            if (turn_on && last >= reuse && last < reuse + 512 && !have) ckpt_keep(prompt, reuse, true);
        }
        if (reuse == 0 && slot_max > 0) {
            slot_put("");              // the conversation in the model, before this prompt overwrites it
            reuse = slot_take(prompt);
            if (reuse > 0) ckpt_drop_past(0);   // another conversation's state now
        } else if (reuse > 0 && !from_ckpt && slot_max > 0 && getenv("STRATA_GLM_SLOT_ROUNDTRIP") != nullptr) {
            // a test: the restored state goes through a slot file and back - the same tokens must follow
            snap_in_slot = false;
            slot_put("");
            const size_t back = slot_take(prompt);
            if (back != reuse) {
                std::fprintf(stderr, "glm slots: ROUNDTRIP took back %zu of %zu tokens\n", back, reuse);
                reuse = back;
            }
        }
        if (reuse == 0) model.reset();
        ckpt_drop_past(reuse);   // this prompt writes the state from `reuse` on (all of it after a reset)
        std::vector<float> lg;
        t0 = std::chrono::steady_clock::now();
        gap_mark = t0;   // (STRATA_GLM_PREFILL_GAPS: the first gap's stage starts with the read, not the restore)
        // the prompt one CHUNK at a time (PP reports per chunk: the server shows it and counts it as a heartbeat);
        // the state is saved before the LAST prompt token, so an identical or extending prompt can reuse it
        bool snapped = false;
        // the batched prompt path (glm_prefill.cu) reads all but the last prompt token in chunks, layer by layer;
        // the last one goes through the token path, which leaves the logits the first sample reads
        const int pchunk = model.fast() ? model.prefill_chunk() : 0;
        size_t i_first = reuse;
        if (pchunk > 0 && prompt.size() - reuse > 1) {
            // one call for everything but the last token: the two halves of a split pipeline the chunks; PP lines
            // (the server's progress and heartbeat) come from the progress callback, possibly on the tail's thread.
            // A checkpoint position (ckpt_marks) ends a call: the state there is kept, and the next call goes on.
            std::mutex pp_mu;
            const auto pp_line = [&](size_t done) {
                if (!o.serve) return;
                std::lock_guard<std::mutex> lk(pp_mu);
                const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
                std::printf("PP %s%zu %zu %.1f %.1f %zu\n", tag.c_str(), reuse + done, prompt.size(), ms,
                            (double) done / (ms / 1000.0 + 1e-9), reuse);
                std::fflush(stdout);
            };
            std::vector<size_t> marks = ckpt_marks(prompt, reuse, prompt.size() - 1);
            // the first user turn's marker: the end of the system prompt (ckpt_marks' `first`)
            const size_t sys_end = strata::core::system_prompt_end(prompt.data(), prompt.size(), 154827);   // <|user|>
            // without checkpoints (STRATA_GLM_CKPTS=0, the several-conversation scheduler) the read still ends a call
            // there with the slots on, for slot_prefix: another conversation with this system prompt (a new conversation
            // of the same client, another request of the same kind) restores it instead of reading it again.
            // STRATA_GLM_PREFIX_CUT=0: off
            static const bool prefix_cut_on = env_num("STRATA_GLM_PREFIX_CUT", 1) != 0;
            if (ckpt_max <= 0 && slot_max > 0 && prefix_cut_on && cur_img_pos.empty())
                strata::core::add_mark(marks, strata::core::prefix_cut(prompt.data(), prompt.size(), reuse, slot_min,
                                                                        154827), reuse);
            size_t at = reuse;   // the next token the prompt path reads
            bool pok = true, between_failed = false;
            // one call to the next checkpoint mark (or the prompt's last token) - with STRATA_GLM_PREFILL_INTERLEAVE and
            // requests decoding, a piece of it, and their decode steps between two pieces (prompt_between)
            size_t mi = 0, seg_start = at;
            // STRATA_GLM_PREFILL_GAPS: a gap before the read too (a round for the running requests) - back-to-back reads
            // would otherwise stall them for the last stage of one and the first of the next
            if (read_gap_first && prompt_between && at < prompt.size() - 1) {
                const int r = prompt_between();
                if (r == 1) {
                    pok = false;
                    err = "cancelled";
                } else if (r == 2) {
                    pok = false;
                    between_failed = true;
                }
            }
            int seg_chunk = 0;   // (exact pieces: the chunk the segment's uncut read takes, from its first piece on)
            while (pok && at < prompt.size() - 1) {
                const size_t seg_end = mi < marks.size() ? marks[mi] : prompt.size() - 1;
                size_t piece = prompt_piece ? prompt_piece() : 0;
                if (piece > 0 && interleave.exact) {
                    // whole chunks of the uncut read, each call cut in them: the same arithmetic, the same answer
                    if (at == seg_start || seg_chunk == 0) seg_chunk = model.prefill_chunk_for((int64_t) (seg_end - seg_start));
                    piece = strata::core::prefill_aligned_piece(piece, (size_t) seg_chunk);
                }
                const size_t end = strata::core::prefill_piece_end(at, seg_end, piece);
                // (an interleaved piece cut freely (,cut): both GPUs of a split at work - two chunks, or ~4096-token ones
                // for a larger piece: a call fills and drains the split's pipeline, a GPU idle for a chunk each way)
                model.prefill_min_chunks = piece > 0 && !interleave.exact ? std::max<int>(2, (int) (piece / 4096)) : 1;
                model.prefill_force_chunk = seg_chunk > 0 && interleave.exact ? seg_chunk : 0;
                const std::vector<int32_t> part(prompt.begin() + (long) at, prompt.begin() + (long) end);
                const size_t base = at - reuse;
                model.prefill_progress = [&](int64_t done, int64_t total) {
                    (void) total;
                    pp_line(base + (size_t) done);
                    return !read_stopped();
                };
                pok = guarded([&] { return model.prefill(part, err, prompt[end]); }, err);   // (the NextN block's caches read the next token)
                model.prefill_progress = nullptr;
                model.prefill_min_chunks = 1;
                model.prefill_force_chunk = 0;
                if (!pok && gap_failed) between_failed = true;   // (a decode gap inside the call failed)
                if (pok) {
                    at = end;
                    if (end == seg_end && mi < marks.size()) {
                        if (ckpt_max > 0) ckpt_keep(prompt, end);   // (no checkpoints: the mark is a prefix cut's only)
                        if (end == sys_end) slot_prefix(prompt, end);
                        ++mi;
                        seg_start = end;
                        seg_chunk = 0;
                    }
                    // more to read: the decoding requests' turn (none after the last piece - its last token's logits
                    // must still be on the tail when the first sample reads them)
                    if (at < prompt.size() - 1 && prompt_between) {
                        pp_line(at - reuse);
                        const int r = prompt_between();
                        if (r == 1) {
                            pok = false;
                            err = "cancelled";
                        } else if (r == 2) {
                            pok = false;
                            between_failed = true;
                        }
                    }
                }
            }
            if (between_failed) {   // every context was reset (multi_fail): nothing of this prompt is left to keep
                drop_if_poisoned(prompt, prompt_between_err);   // (a non-finite error: its slots too, as below)
                std::printf("ERR %s%s\n", tag.c_str(), prompt_between_err.c_str());
                if (gap_reset) gap_reset();   // (a gap's failure: the reset multi_fail left for after the read)
                return 1;
            }
            // a call the prompt path did not take (fewer tokens than STRATA_GLM_PREFILL_MIN: false, no error) leaves the
            // rest to the token path below - from where the prompt path stopped, not from `reuse`
            i_first = at;
            if (pok) {
                i_first = prompt.size() - 1;
                pp_line(prompt.size() - 1 - reuse);
            } else if (err == "cancelled") {
                snap_tokens.clear();
                model.reset();
                ckpt_drop_past(0);
                finish = "cancel";
                i_first = prompt.size();
            } else if (!err.empty()) {
                snap_tokens.clear();
                drop_if_poisoned(prompt, err);
                std::printf("ERR %s%s\n", tag.c_str(), err.c_str());
                return 1;
            }
        }
        for (size_t i = i_first; i < prompt.size();) {
            strata::core::progress_at("the prompt: the token path, at token", (int64_t) i);
            strata::core::progress_beat();
            if (model.fast() && !snapped && i == prompt.size() - 1) {
                if (model.snapshot_save()) {
                    snap_tokens.assign(prompt.begin(), prompt.end() - 1);
                    snap_img_hash = img_hash(snap_tokens.size());
                    snap_in_slot = false;
                } else {
                    snap_tokens.clear();
                }
                snapped = true;
            }
            size_t chunk = std::min<size_t>(8, prompt.size() - i);
            if (model.fast() && !snapped && i + chunk > prompt.size() - 1) chunk = prompt.size() - 1 - i;
            std::vector<int32_t> part(prompt.begin() + (long) i, prompt.begin() + (long) (i + chunk));
            const bool ok = model.forward(part, lg, err);
            if (!ok) {
                snap_tokens.clear();
                drop_if_poisoned(prompt, err);
                std::printf("ERR %s%s\n", tag.c_str(), err.c_str());
                return 1;
            }
            i += chunk;
            if (o.serve) {
                const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
                std::printf("PP %s%zu %zu %.1f %.1f %zu\n", tag.c_str(), i, prompt.size(), ms,
                            (double) (i - reuse) / (ms / 1000.0 + 1e-9), reuse);
            }
            if (read_stopped()) { finish = "cancel"; break; }
        }
        return 0;
    };
    // PAUSE: the snapshot moves from the prompt to the last token fed (the decode feeds every written token but
    // the last), so prompt + what was written continues it with one token to read instead of all it wrote.
    // Text only, and only when the state stands exactly there (the decode paths leave pos = fed tokens).
    const auto pause_snapshot = [&](const std::vector<int32_t>& prompt, const std::vector<int32_t>& emitted,
                                    bool keep) {
        if (model.fast() && cur_img_pos.empty() && !emitted.empty() && getenv("STRATA_GLM_NO_REUSE") == nullptr) {
            const int64_t p = model.position();
            const int64_t full = (int64_t) (prompt.size() + emitted.size());
            if (keep && slot_max > 0 && p > (int64_t) prompt.size() && p < full) {
                // the request's first slice: its prompt's state goes to a slot (unless one holds it already) and is
                // anchored there, for this chat's next turn
                bool held = false;
                for (auto& sl : slots)
                    if (sl.tokens == snap_tokens) sl.anchor = held = true;
                if (!held && !snap_tokens.empty()) {
                    slot_put(" (a paused request's prompt, kept for its next turn)");
                    for (auto& sl : slots)
                        if (sl.tokens == snap_tokens) sl.anchor = true;
                }
            }
            if (p > (int64_t) prompt.size() && p < full && model.snapshot_save()) {
                snap_tokens.assign(prompt.begin(), prompt.end());
                snap_tokens.insert(snap_tokens.end(), emitted.begin(), emitted.begin() + (p - (int64_t) prompt.size()));
                snap_img_hash = img_hash(snap_tokens.size());
                snap_in_slot = false;
                std::fprintf(stderr, "glm slots: paused at %lld tokens (%lld written): the state is kept there\n",
                             (long long) p, (long long) emitted.size());
            } else if (p > (int64_t) prompt.size()) {
                std::fprintf(stderr, "glm slots: paused, but the state stands at %lld of %lld tokens: not kept\n",
                             (long long) p, (long long) full);
            }
        }
    };
    // STRATA_GLM_SPEC_FOLLOW=<file> (--spec-follow): every request decodes this continuation instead of its own picks
    // and ends with it (Glm5Model::set_follow) - a speed A/B on one text.  STRATA_GLM_SPEC_FOLLOW_SAVE=<file>
    // (--spec-follow-save): each request's decoded ids are written there, the next run's continuation.  A text made by
    // another model, or by the same model with other settings, favours the drafts that made it: compare a change on
    // the same model and its own text.
    std::vector<int32_t> follow_ids;
    const char* follow_save = getenv("STRATA_GLM_SPEC_FOLLOW_SAVE");
    if (follow_save != nullptr && follow_save[0] == '\0') follow_save = nullptr;
    if (const char* fp = getenv("STRATA_GLM_SPEC_FOLLOW"); fp != nullptr && fp[0] != '\0') {
        std::ifstream input(fp);
        std::string text((std::istreambuf_iterator<char>(input)), std::istreambuf_iterator<char>());
        std::vector<int64_t> v;
        std::string e;
        if (!input && !input.eof()) e = "cannot read it";
        else if (!parse_i64_list(text.c_str(), v, e)) e = "not a list of token ids: " + e;
        for (const int64_t t : v)
            if (e.empty() && t >= n_vocab) e = "token " + std::to_string(t) + " is past the vocabulary";
        if (!e.empty()) {
            std::fprintf(stderr, "maya generate: MAYA_GLM_SPEC_FOLLOW %s: %s\n", fp, e.c_str());
            return 2;
        }
        follow_ids.assign(v.begin(), v.end());
        std::fprintf(stderr, "glm follow: every request decodes the %zu tokens of %s (MAYA_GLM_SPEC_FOLLOW)\n",
                     follow_ids.size(), fp);
    }
    const auto run_request = [&](const std::vector<int64_t>& ids, int64_t max_new,
                                 const strata::kernels::SamplerParams& rq_in) -> int {
        strata::kernels::SamplerParams sp = rq_in;   // per request (the server sends sampling keys per GEN)
        stop_req.store(false);
        pause_req.store(false);
        pause_keep.store(false);
        model.force_next(-1);
        model.set_follow(follow_ids);   // STRATA_GLM_SPEC_FOLLOW: this request's tokens (empty: its own picks)
        if (!follow_ids.empty()) max_new = std::min<int64_t>(max_new, (int64_t) follow_ids.size());
        bool in_think = req_think_budget > 0 && req_think_end >= 0;
        std::vector<int32_t> prompt;
        prompt.reserve(ids.size());
        for (const int64_t t : ids) {
            if (t < 0 || t >= n_vocab) {
                std::printf("ERR token %lld out of range [0, %lld)\n", (long long) t, (long long) n_vocab);
                return 1;
            }
            prompt.push_back((int32_t) t);
        }
        if (prompt.empty() || max_new <= 0) {
            std::printf("ERR empty prompt or max_new\n");
            return 1;
        }
        // the penalties' history (#273): what the state is fed - the prompt, then every token written (on_token)
        std::vector<int32_t> pen_hist;
        if (sp.penalty_last_n > 0) {
            pen_hist = prompt;
            sp.host_history = &pen_hist;
        }
        size_t reuse = 0;
        const char* finish = "length";
        auto t0 = std::chrono::steady_clock::now();
        if (prep_prompt(prompt, reuse, finish, t0) != 0) return 1;
        std::vector<float> lg;
        if (stop_req.load()) finish = "cancel";
        const auto t1 = std::chrono::steady_clock::now();
        const auto st1 = model.fast_stats();
        int64_t produced = 0;
        // one emitted token: the T line, the live tier counters every 8 tokens, the stop conditions (false: stop)
        // LOOP GUARD: a heavily quantized model can fall into repeating a short pattern ("0 0 0 ...") until the
        // thinking budget or max_tokens ends it - minutes of nothing.  When the last W tokens are one pattern of at most
        // 32 tokens repeated exactly, thinking is closed at once (as the budget does); in the answer, what the model is in the
        // middle of is closed (a tool call's argument, else the turn), and a second loop ends it.
        // W: STRATA_GLM_LOOP_THINK (256) while thinking, STRATA_GLM_LOOP_ANSWER (1024) in the answer - long enough
        // for legitimately repetitive output (a zero-filled array); 0 turns that check off.
        static const int loop_w_think = (int) env_num("STRATA_GLM_LOOP_THINK", 256);
        static const int loop_w_answer = (int) env_num("STRATA_GLM_LOOP_ANSWER", 1024);
        std::vector<int> gen_hist;
        std::vector<int32_t> emitted;   // every token this request wrote (a PAUSE keeps the state they lead to)
        bool in_arg = false;     // inside a tool call's <arg_value> (the answer's position, for the loop guard)
        int guard_fired = 0;     // soft landings in this answer: a second loop ends it
        const auto looping = [&](int w) -> int {   // the pattern's period, 0 when the tail is not one
            const int n = (int) gen_hist.size();
            if (w <= 0 || n < w) return 0;
            for (int per = 1; per <= 32 && per < w / 4; ++per) {
                bool rep = true;
                for (int i = n - w + per; i < n && rep; ++i) rep = gen_hist[(size_t) i] == gen_hist[(size_t) (i - per)];
                if (rep) return per;
            }
            return 0;
        };
        const auto on_token = [&](int tok) -> bool {
            std::printf("T %d\n", tok);
            ++produced;
            emitted.push_back((int32_t) tok);
            if (sp.host_history != nullptr) pen_hist.push_back((int32_t) tok);   // (fed next: the picks after count it)
            strata::core::progress_at("decode: after token", produced);   // the watchdog's heartbeat (issue #40)
            strata::core::progress_beat();
            gen_hist.push_back(tok);
            if (in_think && tok != req_think_end) {
                if (const int per = looping(loop_w_think)) {
                    model.force_next(req_think_end);   // as the budget does: the next token closes the reasoning
                    in_think = false;
                    char m[200];
                    std::snprintf(m, sizeof m, "thinking was repeating a %d-token pattern (%d tokens in): closed, "
                                  "answering", per, (int) produced);
                    std::fprintf(stderr, "glm loop guard: %s\n", m);
                    if (o.serve) std::printf("NOTE %s\n", m);
                    gen_hist.clear();
                }
            } else if (!in_think) {
                // where the answer stands: inside a tool call's argument value, the place a loop must be closed for
                // an agent to go on (GLM's single-token tags)
                constexpr int kArgValue = 154849, kArgValueEnd = 154850, kUser = 154827;
                if (tok == kArgValue) in_arg = true;
                if (tok == kArgValueEnd) in_arg = false;
                if (const int per = looping(loop_w_answer)) {
                    char m[240];
                    if (guard_fired == 0) {
                        // a soft landing: close what the model is in the middle of instead of cutting the response -
                        // a tool call's argument ends (the call completes with what was written, the agent runs it
                        // and sees the result), plain text ends the turn (finish "stop")
                        model.force_next(in_arg ? kArgValueEnd : kUser);
                        std::snprintf(m, sizeof m, "the answer was repeating a %d-token pattern (%d tokens in): %s",
                                      per, (int) produced, in_arg ? "the tool call's argument was closed there"
                                                                  : "the turn was ended there");
                    } else {
                        std::snprintf(m, sizeof m, "the answer kept repeating a %d-token pattern (%d tokens in): "
                                      "ended", per, (int) produced);
                    }
                    std::fprintf(stderr, "glm loop guard: %s\n", m);
                    if (o.serve) std::printf("NOTE %s\n", m);
                    gen_hist.clear();
                    if (guard_fired++ > 0) {   // it did not stop: the hard way
                        finish = "length";
                        return false;
                    }
                }
            }
            if (in_think && (tok == req_think_end || produced >= req_think_budget || wrap_req.load())) {
                if (tok != req_think_end) model.force_next(req_think_end);   // the budget: the next token closes it
                in_think = false;
                gen_hist.clear();
            }
            // live expert-tier counters for the server's Monitor (serve/server.py parses STAT lines), every 8 tokens
            if (o.serve && model.fast() && produced % 8 == 0) {
                const auto sn = model.fast_stats();
                const double el = std::chrono::duration<double>(std::chrono::steady_clock::now() - t1).count();
                const double n = (double) produced;
                const uint64_t h = sn.hits - st1.hits, m = sn.misses - st1.misses;
                std::printf("STAT tok_s=%.2f ms_tok=%.2f vram_hit=%.4f ram_fetch=%.3f disk=%.3f promo=%.3f "
                            "vram_used=%lld vram_slots=%lld ram_used=%lld ram_slots=%lld ram_gb=%.1f vram_gb=%.1f\n",
                            n / std::max(1e-9, el), 1000.0 * el / n, (double) h / (double) std::max<uint64_t>(1, h + m),
                            (double) (sn.ram_hits - st1.ram_hits) / n, (double) (sn.disk_reads - st1.disk_reads) / n,
                            (double) (sn.promotions - st1.promotions) / n, (long long) sn.pool_used,
                            (long long) sn.pool_slots, (long long) sn.ram_used, (long long) sn.ram_slots, sn.ram_gb,
                            sn.pool_gb);
            }
            bool eos_hit = false;
            for (const int64_t e : eos)
                if (e == tok) eos_hit = true;
            if (eos_hit && stop_on_eos) {
                finish = "stop";
                return false;
            }
            if (stop_req.load()) {
                finish = "cancel";
                return false;
            }
            return produced < max_new;
        };
        const uint64_t spec0 = model.spec_steps_, hits0 = model.spec_hits_;
        if (std::strcmp(finish, "cancel") != 0 && model.fast() && model.dflash_ready() && req_spec != 1) {
            // the DFlash decode: a block of drafts a round, one verify window checks them (same tokens as the loop
            // below for the same samples); --spec T caps a round's window at T tokens
            int64_t n_spec = 0;
            const int spec = req_spec >= 2 ? req_spec : o.spec;   // this request's window, else --spec
            const int n_max = spec >= 2 ? spec - 1 : -1;            // -1: the engine's (decode_dflash)
            // the drafts stop at a selector confidence below the cut: the request's, else --spec-min-p, else the
            // engine's (-1: decode_dflash picks it for the window it verifies with)
            const float p_min = req_spec_min_p >= 0.0 && req_spec_min_p <= 1.0 ? (float) req_spec_min_p
                                : o.spec_min_p_set                         ? (float) o.spec_min_p
                                                                           : -1.0f;
            if (!model.decode_dflash(sp, max_new, n_max, p_min, on_token, n_spec, err, req_spec_n_min,
                                     req_spec_cum >= 0.0 && req_spec_cum <= 1.0 ? (float) req_spec_cum : -1.0f)) {
                drop_if_poisoned(prompt, err);
                std::printf("ERR %s\n", err.c_str());
                return 1;
            }
        } else if (std::strcmp(finish, "cancel") != 0 && model.fast() && model.mtp_ready() && req_spec != 1) {
            // the one-GPU MTP decode (glm_mtp.cu; by default on the Strix / Gorgon Halo APUs): the NextN block drafts,
            // one verify window checks the drafts (same tokens as the loop below for the same samples)
            int64_t n_spec = 0;
            const uint64_t rounds0 = model.mtp_rounds_;
            uint64_t len0[9];
            std::copy(model.mtp_round_n_, model.mtp_round_n_ + 9, len0);
            if (!model.decode_mtp(sp, max_new, on_token, n_spec, err)) {
                drop_if_poisoned(prompt, err);
                std::printf("ERR %s\n", err.c_str());
                return 1;
            }
            if (model.mtp_rounds_ > rounds0) {
                std::string lens;
                for (int i = 0; i < 9; ++i)
                    if (model.mtp_round_n_[i] > len0[i])
                        lens += " " + std::to_string(i) + ":" + std::to_string(model.mtp_round_n_[i] - len0[i]);
                std::fprintf(stderr, "glm mtp: %llu rounds, %.2f tokens a round | rounds by draft length%s\n",
                             (unsigned long long) (model.mtp_rounds_ - rounds0),
                             (double) n_spec / (double) (model.mtp_rounds_ - rounds0), lens.c_str());
            }
        } else if (std::strcmp(finish, "cancel") != 0 && model.fast() && model.spec_ready()) {
            // the pipelined speculative decode: the two halves of the split work on consecutive tokens, the head
            // running the NextN block's draft (same tokens as the loop below)
            int64_t n_spec = 0;
            if (!model.decode_spec(sp, max_new, on_token, n_spec, err)) {
                drop_if_poisoned(prompt, err);
                std::printf("ERR %s\n", err.c_str());
                return 1;
            }
        } else {
            for (;;) {
                if (stop_req.load()) {
                    finish = "cancel";
                    break;
                }
                int tok = model.sample_token(sp, err);
                if (tok < 0) {
                    drop_if_poisoned(prompt, err);
                    std::printf("ERR %s\n", err.c_str());
                    return 1;
                }
                tok = model.forced(tok);
                sp.counter += 1;
                if (!on_token(tok)) break;
                if (!model.forward({(int32_t) tok}, lg, err)) {
                    drop_if_poisoned(prompt, err);
                    std::printf("ERR %s\n", err.c_str());
                    return 1;
                }
            }
        }
        const auto t2 = std::chrono::steady_clock::now();
        const double prompt_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        const double decode_ms = std::chrono::duration<double, std::milli>(t2 - t1).count();
        // the decode phase's expert-tier counters (the fast path; zeros on the reference path)
        const auto st2 = model.fast_stats();
        const uint64_t d_hits = st2.hits - st1.hits, d_miss = st2.misses - st1.misses;
        std::printf("DONE %lld %lld %.1f %.1f %s %llu %llu %zu %llu %llu\n", (long long) produced,
                    (long long) prompt.size(), prompt_ms, decode_ms, finish,
                    (unsigned long long) (model.spec_hits_ - hits0), (unsigned long long) (model.spec_steps_ - spec0),
                    reuse, (unsigned long long) d_hits, (unsigned long long) (d_hits + d_miss));
        std::fflush(stdout);
        if (pause_req.load() && std::strcmp(finish, "cancel") == 0) pause_snapshot(prompt, emitted, pause_keep.load());
        // this machine's expert usage, for the next start's warm-up (after DONE: the client is not kept waiting)
        if (model.fast()) model.save_usage();
        if (model.fast() && produced > 0) {
            const double n = (double) produced;
            std::fprintf(stderr,
                         "glm stat: decode %.2f ms/tok (%.2f tok/s) | vram hit %.2f%% | ram fetches %.2f/tok | "
                         "disk %.2f reads/tok in %.2f waits/tok, %.2f ms/tok | promotions %.2f/tok | cpu lane %.2f/tok "
                         "in %.2f ms/tok | prompt %.1f ms/tok | vram %lld/%lld ram %lld/%lld\n",
                         decode_ms / n, 1000.0 * n / decode_ms,
                         100.0 * (double) d_hits / (double) std::max<uint64_t>(1, d_hits + d_miss),
                         (double) (st2.ram_hits - st1.ram_hits) / n, (double) (st2.disk_reads - st1.disk_reads) / n,
                         (double) (st2.miss_layers - st1.miss_layers) / n, (st2.disk_ms - st1.disk_ms) / n,
                         (double) (st2.promotions - st1.promotions) / n,
                         (double) (st2.cpu_experts - st1.cpu_experts) / n, (st2.cpu_ms - st1.cpu_ms) / n,
                         prompt_ms / (double) std::max<size_t>(1, prompt.size()), (long long) st2.pool_used,
                         (long long) st2.pool_slots, (long long) st2.ram_used, (long long) st2.ram_slots);
            if (st2.disk_skips > st1.disk_skips)
                std::fprintf(stderr, "glm skip: %.2f disk-only experts/tok left out (MAYA_GLM_DISK_SKIP_W / _FROM)\n",
                             (double) (st2.disk_skips - st1.disk_skips) / n);
            if (model.spec_steps_ > spec0)
                std::fprintf(stderr, "glm spec: %llu of %llu drafts accepted (%.1f%%)\n",
                             (unsigned long long) (model.spec_hits_ - hits0),
                             (unsigned long long) (model.spec_steps_ - spec0),
                             100.0 * (double) (model.spec_hits_ - hits0) / (double) (model.spec_steps_ - spec0));
            if (!follow_ids.empty())
                std::fprintf(stderr, "glm follow: %zu tokens of the continuation decoded, %zu of them differ from the "
                                     "model's own pick\n", model.follow_used(), model.follow_differ());
            if (model.kv_resident() > 0) {
                // KV streaming, cumulative over the process: the blocks the decode's selections named, and those
                // read from RAM (every part summed)
                const auto kv = model.kv_stream_stats();
                std::fprintf(stderr, "glm kv: KV streaming: %.2f%% of %llu block reads hit VRAM, %.1f MiB read from RAM%s\n",
                             kv.lookups ? 100.0 * (double) (kv.lookups - kv.misses) / (double) kv.lookups : 100.0,
                             (unsigned long long) kv.lookups,
                             (double) kv.misses * kv.block_bytes / 1048576.0,
                             kv.overflow ? " - OVERFLOW (too few resident positions)" : "");
            }
        }
        model.set_follow({});
        // STRATA_GLM_SPEC_FOLLOW_SAVE: this request's decoded ids, the continuation another run can follow
        if (follow_save != nullptr && !emitted.empty()) {
            const std::string tmp = std::string(follow_save) + ".tmp";
            {
                std::ofstream f(tmp, std::ios::trunc);
                for (size_t i = 0; i < emitted.size(); ++i) f << (i ? "," : "") << emitted[i];
                f << "\n";
            }
            std::error_code ec;
            std::filesystem::rename(tmp, follow_save, ec);
            if (ec)
                std::fprintf(stderr, "glm follow: could not write %s: %s\n", follow_save, ec.message().c_str());
            else
                std::fprintf(stderr, "glm follow: %zu decoded ids written to %s (MAYA_GLM_SPEC_FOLLOW_SAVE)\n",
                             emitted.size(), follow_save);
        }
        return 0;
    };

    // ---- SEVERAL CONVERSATIONS AT ONCE (serve, STRATA_GLM_SEQS=<n> > 1).  GEN lines carry rid=<id>, and every line
    // a request causes carries it after its keyword (T <rid> <id>, PP <rid> ..., DONE <rid> ..., ERR <rid> ...);
    // STOP <rid> and PAUSE <rid> [keep] end one request, STOP alone all of them.  Each request runs in one of the n
    // sequence states, a CONTEXT: an idle context whose snapshot the prompt extends is taken as it stands (nothing
    // read, nothing loaded), else the least recently used idle one, whose conversation goes to a slot first.  A
    // request alone decodes through the speculative path (the fastest for one); as soon as a second one comes, both go
    // through the pipelined decode (Glm5Model::multi_issue), each card of the split on a different conversation.
    const bool multi = o.serve && model.fast() && model.seq_count() > 1;
    if (multi && !follow_ids.empty()) {
        std::fprintf(stderr, "glm follow: not used with MAYA_GLM_SEQS=%d - one conversation at a time only\n",
                     model.seq_count());
        follow_ids.clear();
    }
    // STRATA_GLM_MULTI_MTP=<drafts>: the running requests' batched steps carry verify windows - each one's last token
    // and the NextN block's drafts after it (Glm5Model::mtpw_*, strata/core/mtp_rows.hpp)
    const bool mtp_rows = multi && model.mtpw_ok();
    if (multi && strata::core::Glm5Model::multi_mtp_setting() > 0 && !mtp_rows)
        std::fprintf(stderr, "glm multi mtp: MAYA_GLM_MULTI_MTP is set but not used here - it needs MAYA_GLM_SPLIT_ROWS "
                             "on a split (or one GPU), the NextN block loaded (no MAYA_GLM_NO_MTP) and no --dflash\n");
    if (mtp_rows)
        std::fprintf(stderr, "glm multi mtp: the running requests verify up to %d drafts each in their batched steps "
                             "(%d rows a group)\n", strata::core::Glm5Model::multi_mtp_setting(),
                     strata::core::Glm5Model::rows_max());
    if (multi && model.dflash_ready())
        std::fprintf(stderr, "glm dflash: not used with MAYA_GLM_SEQS=%d - the several-conversation scheduler decodes "
                             "without it (and without the MTP block, which --dflash leaves unloaded)\n",
                     model.seq_count());
    struct Ctx {   // a context's driver-side state while another one is bound (the bound one's is in the globals)
        std::vector<int32_t> snap_tokens;
        uint64_t snap_img_hash = kNoImages;
        bool snap_in_slot = false;
        uint64_t used = 0;
        bool busy = false;
    };
    std::vector<Ctx> ctxs((size_t) model.seq_count());
    int cur_ctx = 0;
    uint64_t ctx_clock = 0;
    const auto bind_ctx = [&](int c) {
        if (c != cur_ctx) {
            Ctx& was = ctxs[(size_t) cur_ctx];
            was.snap_tokens = std::move(snap_tokens);
            was.snap_img_hash = snap_img_hash;
            was.snap_in_slot = snap_in_slot;
            Ctx& now = ctxs[(size_t) c];
            snap_tokens = std::move(now.snap_tokens);
            now.snap_tokens.clear();
            snap_img_hash = now.snap_img_hash;
            snap_in_slot = now.snap_in_slot;
            cur_ctx = c;
        }
        model.seq_bind_all(c);   // (the pipelined decode binds the model's sequences itself)
    };
    struct Gen {
        int rid = -1, ctx = 0;
        strata::kernels::SamplerParams sp;
        int64_t max_new = 0, produced = 0;
        std::vector<int32_t> prompt, emitted;
        std::vector<int32_t> pen;   // the penalties' history (sp.host_history): the prompt, then every token written
        bool in_think = false;
        int64_t think_budget = 0;
        int32_t think_end = -1;
        std::vector<int> hist;   // the loop guard's window
        bool in_arg = false;
        int guard = 0;
        int32_t force = -1;      // the next token instead of the sampled one (the thinking budget, the loop guard)
        int32_t pending = -1;    // written, not fed yet
        bool ready = false;      // just admitted: the first sample's logits are on the tail
        bool inflight = false, done = false, pause = false, keep = false;
        const char* finish = "length";
        size_t reuse = 0;
        std::chrono::steady_clock::time_point t0, t1;
        uint64_t spec_steps0 = 0, spec_hits0 = 0;
        strata::core::Glm5Model::FastStats st1;
        // STRATA_GLM_MULTI_MTP: the NextN block's drafts for the positions after `pending` (made at position
        // draft_at after draft_tok: stale once either moved), and this request's verify statistics
        std::vector<int32_t> drafts;
        int64_t draft_at = -1;
        int32_t draft_tok = -1;
        uint64_t mtp_steps = 0, mtp_drafts = 0, mtp_acc = 0;
    };
    std::vector<std::unique_ptr<Gen>> gens;
    std::deque<Gen*> flight;     // issued, oldest first
    std::deque<int> flight_groups;   // split rows: the sizes of the issued groups, oldest first
    // a request no idle context can hold yet (its prompt needs the large first context, which is busy): it waits
    // here with what its GEN line said, and is admitted when one frees
    struct Pending {
        int rid;
        std::vector<int64_t> ids;
        int64_t max_new;
        strata::kernels::SamplerParams rq;
        int64_t think_budget;
        int32_t think_end;
        std::vector<int64_t> img_pos;
        std::vector<float> img_rows;
        std::vector<std::pair<int32_t, float>> bias;
    };
    std::deque<Pending> pending;
    // the room a context must leave for the answer (at most what the request asked for): a long prompt goes to a
    // context with this much left after it, and its max tokens are cut to that context
    static const int64_t multi_room = (int64_t) env_num("STRATA_GLM_MULTI_ROOM", 16384);
    static const int multi_boundary_every = (int) env_num("STRATA_GLM_MULTI_BOUNDARY", 32);
    static const bool multi_spec = env_num("STRATA_GLM_MULTI_SPEC", 1) != 0;
    static const int split_rows = strata::core::Glm5Model::split_rows_setting();
    // STRATA_GLM_EVEN_GROUPS=1: the split's groups cut evenly from all the running requests, two at least, instead of
    // k at a time (strata/core/split_groups.hpp): k 4 no longer makes one group - and idle GPUs - at 4 requests or fewer
    static const bool even_groups = env_num("STRATA_GLM_EVEN_GROUPS", 0) != 0;
    // Reuse split-row scheduler buffers; allocating vectors for each decode group defeats part of the batching gain.
    std::vector<Gen*> split_group;
    std::vector<int> split_seqs;
    std::vector<int32_t> split_toks;
    std::vector<strata::kernels::SamplerParams> split_sps;
    if (split_rows > 0) {
        const size_t cap = (size_t) split_rows;
        split_group.reserve(cap);
        split_seqs.reserve(cap);
        split_toks.reserve(cap);
        split_sps.reserve(cap);
    }
    int since_boundary = 0;
    uint64_t multi_n = 0;        // tokens since the engine last had nothing to do (the STAT lines' rate)
    std::chrono::steady_clock::time_point multi_t1;
    strata::core::Glm5Model::FastStats multi_st1;
    const auto input_pending = [&]() -> bool {
        std::lock_guard<std::mutex> lk(in_mu);
        return !in_lines.empty() || quit_req.load();
    };
    const auto try_line = [&](std::string& out) -> bool {
        std::lock_guard<std::mutex> lk(in_mu);
        if (in_lines.empty()) return false;
        out = std::move(in_lines.front());
        in_lines.pop_front();
        return true;
    };
    // one written token of a request: its T line, the loop guard and the thinking budget (as run_request's
    // on_token), the stop conditions.  false: the request ends here (g.finish says why)
    const auto gen_token = [&](Gen& g, int tok) -> bool {
        std::printf("T %d %d\n", g.rid, tok);
        ++g.produced;
        ++multi_n;
        g.emitted.push_back((int32_t) tok);
        if (g.sp.host_history != nullptr) g.pen.push_back((int32_t) tok);
        strata::core::progress_at("decode: after token", g.produced);
        strata::core::progress_beat();
        g.hist.push_back(tok);
        const auto looping = [&](int w) -> int {
            const int n = (int) g.hist.size();
            if (w <= 0 || n < w) return 0;
            for (int per = 1; per <= 32 && per < w / 4; ++per) {
                bool rep = true;
                for (int i = n - w + per; i < n && rep; ++i) rep = g.hist[(size_t) i] == g.hist[(size_t) (i - per)];
                if (rep) return per;
            }
            return 0;
        };
        static const int loop_w_think = (int) env_num("STRATA_GLM_LOOP_THINK", 256);
        static const int loop_w_answer = (int) env_num("STRATA_GLM_LOOP_ANSWER", 1024);
        if (g.in_think && tok != g.think_end) {
            if (const int per = looping(loop_w_think)) {
                g.force = g.think_end;
                g.in_think = false;
                char m[200];
                std::snprintf(m, sizeof m, "thinking was repeating a %d-token pattern (%d tokens in): closed, "
                              "answering", per, (int) g.produced);
                std::fprintf(stderr, "glm loop guard: %s\n", m);
                std::printf("NOTE %s\n", m);
                g.hist.clear();
            }
        } else if (!g.in_think) {
            constexpr int kArgValue = 154849, kArgValueEnd = 154850, kUser = 154827;
            if (tok == kArgValue) g.in_arg = true;
            if (tok == kArgValueEnd) g.in_arg = false;
            if (const int per = looping(loop_w_answer)) {
                char m[240];
                if (g.guard == 0) {
                    g.force = g.in_arg ? kArgValueEnd : kUser;
                    std::snprintf(m, sizeof m, "the answer was repeating a %d-token pattern (%d tokens in): %s", per,
                                  (int) g.produced, g.in_arg ? "the tool call's argument was closed there"
                                                             : "the turn was ended there");
                } else {
                    std::snprintf(m, sizeof m, "the answer kept repeating a %d-token pattern (%d tokens in): ended",
                                  per, (int) g.produced);
                }
                std::fprintf(stderr, "glm loop guard: %s\n", m);
                std::printf("NOTE %s\n", m);
                g.hist.clear();
                if (g.guard++ > 0) {
                    g.finish = "length";
                    return false;
                }
            }
        }
        if (g.in_think && (tok == g.think_end || g.produced >= g.think_budget || wrap_req.load())) {
            if (tok != g.think_end) g.force = g.think_end;
            g.in_think = false;
            g.hist.clear();
        }
        if (multi_n % 8 == 0) {   // the Monitor's live counters, over everything decoded since the engine was idle
            const auto sn = model.fast_stats();
            const double el = std::chrono::duration<double>(std::chrono::steady_clock::now() - multi_t1).count();
            const double n = (double) std::max<uint64_t>(1, multi_n);
            const uint64_t h = sn.hits - multi_st1.hits, m = sn.misses - multi_st1.misses;
            std::printf("STAT tok_s=%.2f ms_tok=%.2f vram_hit=%.4f ram_fetch=%.3f disk=%.3f promo=%.3f "
                        "vram_used=%lld vram_slots=%lld ram_used=%lld ram_slots=%lld ram_gb=%.1f vram_gb=%.1f "
                        "active=%zu\n",
                        n / std::max(1e-9, el), 1000.0 * el / n, (double) h / (double) std::max<uint64_t>(1, h + m),
                        (double) (sn.ram_hits - multi_st1.ram_hits) / n,
                        (double) (sn.disk_reads - multi_st1.disk_reads) / n,
                        (double) (sn.promotions - multi_st1.promotions) / n, (long long) sn.pool_used,
                        (long long) sn.pool_slots, (long long) sn.ram_used, (long long) sn.ram_slots, sn.ram_gb,
                        sn.pool_gb, gens.size());
        }
        for (const int64_t e : eos)
            if (e == tok && stop_on_eos) {
                g.finish = "stop";
                return false;
            }
        {
            std::lock_guard<std::mutex> lk(ctl_mu);
            const auto it = ctl.find(g.rid);
            if (it != ctl.end()) {
                g.pause = it->second >= 2;
                g.keep = it->second == 3;
                ctl.erase(it);
                g.finish = "cancel";
                return false;
            }
        }
        if (stop_req.load()) {
            g.finish = "cancel";
            return false;
        }
        return g.produced < g.max_new;
    };
    // the PAUSE snapshots a read's gaps kept back (STRATA_GLM_PREFILL_GAPS): taken once no prompt is being read and no
    // request waits for its first token (the tail's logits are its own until then), their contexts idle again
    const auto flush_pauses = [&] {
        for (const auto& d : deferred_pauses) {
            bind_ctx(d.ctx);
            pause_snapshot(d.prompt, d.emitted, d.keep);
            ctxs[(size_t) d.ctx].busy = false;
        }
        deferred_pauses.clear();
    };
    // a request's end: its DONE line (the engine's counters over its lifetime are shared with the others running
    // meanwhile), the paused state kept, its context idle again
    const auto gen_finish = [&](Gen& g) {
        const auto t2 = std::chrono::steady_clock::now();
        const double prompt_ms = std::chrono::duration<double, std::milli>(g.t1 - g.t0).count();
        const double decode_ms = std::chrono::duration<double, std::milli>(t2 - g.t1).count();
        const auto st2 = model.fast_stats();
        const uint64_t d_hits = st2.hits - g.st1.hits, d_miss = st2.misses - g.st1.misses;
        std::printf("DONE %d %lld %lld %.1f %.1f %s %llu %llu %zu %llu %llu\n", g.rid, (long long) g.produced,
                    (long long) g.prompt.size(), prompt_ms, decode_ms, g.finish,
                    (unsigned long long) (model.spec_hits_ - g.spec_hits0 + g.mtp_acc),
                    (unsigned long long) (model.spec_steps_ - g.spec_steps0 + g.mtp_drafts), g.reuse,
                    (unsigned long long) d_hits, (unsigned long long) (d_hits + d_miss));
        if (mtp_rows && g.mtp_steps > 0)
            std::fprintf(stderr, "glm multi mtp: request %d: %llu of %llu drafts accepted (%.1f%%) in %llu verify steps, "
                                 "%.2f tokens a step\n", g.rid, (unsigned long long) g.mtp_acc,
                         (unsigned long long) g.mtp_drafts,
                         100.0 * (double) g.mtp_acc / (double) std::max<uint64_t>(1, g.mtp_drafts),
                         (unsigned long long) g.mtp_steps,
                         (double) (g.mtp_steps + g.mtp_acc) / (double) g.mtp_steps);
        bool deferred = false;
        if (g.pause && std::strcmp(g.finish, "cancel") == 0) {
            if (in_read_gap) {   // (another context's prompt is being read and stays bound: the snapshot after it)
                deferred_pauses.push_back({g.ctx, g.prompt, g.emitted, g.keep});
                deferred = true;
            } else {
                bind_ctx(g.ctx);
                pause_snapshot(g.prompt, g.emitted, g.keep);
            }
        }
        if (g.produced > 0)
            std::fprintf(stderr, "glm multi: request %d done (%s): %lld tokens in %.1f s (%.1f tok/s), %zu running\n",
                         g.rid, g.finish, (long long) g.produced, decode_ms / 1000.0,
                         1000.0 * (double) g.produced / std::max(1.0, decode_ms), gens.size() - 1);
        {
            std::lock_guard<std::mutex> lk(ctl_mu);
            ctl.erase(g.rid);
        }
        ctxs[(size_t) g.ctx].busy = deferred;   // (a deferred snapshot: busy until flush_pauses)
        ctxs[(size_t) g.ctx].used = ++ctx_clock;
        g.done = true;
    };
    const auto gen_sweep = [&] {
        gens.erase(std::remove_if(gens.begin(), gens.end(), [](const std::unique_ptr<Gen>& p) { return p->done; }),
                   gens.end());
        if (gens.empty()) {
            // (a prompt being read between decode steps keeps a STOP for all: its read ends too)
            if (admitting_rid.load() >= 0 && stop_req.load()) admit_stop.store(true);
            stop_req.store(false);
            // this machine's expert usage, for the next start's warm-up (not inside a prompt's read)
            if (model.fast() && !in_read_gap) model.save_usage();
            if (!deferred_pauses.empty() && !in_read_gap && admitting_rid.load() < 0) flush_pauses();
        }
    };
    // the token a request takes after one came back: forced, else as sampled
    const auto gen_took = [&](Gen& g, int tok) {
        if (g.force >= 0) {
            tok = g.force;
            g.force = -1;
        }
        if (!gen_token(g, tok)) gen_finish(g);
        else g.pending = tok;
    };
    // STRATA_GLM_MULTI_MTP: the groups of verify windows in flight, oldest first - each past its front (every part but
    // the last), `back` once its last part and head are queued too
    struct WinGroup {
        std::vector<Gen*> g;
        std::vector<int> seqs, counts;
        std::vector<int32_t> toks;
        std::vector<strata::kernels::SamplerParams> sps;
        bool back = false;
    };
    std::deque<WinGroup> wflight;
    const auto win_back = [&](WinGroup& w) -> bool {
        if (w.back) return true;
        w.back = true;
        return model.mtpw_back(w.seqs.data(), w.counts.data(), (int) w.seqs.size(), w.toks.data(), w.sps.data(), err);
    };
    // the oldest group back: its picks; each window's rows written up to the first draft that differs from what was
    // written before it (T lines, the stop conditions, a forced token - gen_took's rules); the commit; then the next
    // windows' drafts - the next group's last part is queued between the drafts and their wait, so the last GPU goes
    // on with it; the requests that ended, ended
    const auto win_take = [&]() -> bool {
        WinGroup w = std::move(wflight.front());
        wflight.pop_front();
        const int n = (int) w.seqs.size();
        int rows = 0;
        for (const int c : w.counts) rows += c;
        std::vector<int32_t> picks((size_t) rows);
        if (!win_back(w) || !model.mtpw_picks(w.seqs.data(), w.counts.data(), n, picks.data(), err)) return false;
        std::vector<int> keep((size_t) n, 1), nd((size_t) n, 0);
        std::vector<int32_t> written, last((size_t) n, -1);
        std::vector<char> ended((size_t) n, 0);
        const int D = strata::core::Glm5Model::multi_mtp_setting();
        for (int i = 0, row = 0; i < n; row += w.counts[(size_t) i], ++i) {
            Gen& g = *w.g[(size_t) i];
            g.inflight = false;
            std::vector<int32_t> wr;
            keep[(size_t) i] = strata::core::mtp_accept(
                w.toks.data() + row, picks.data() + row, w.counts[(size_t) i],
                [&](int, int32_t pick, int32_t& out) {
                    int32_t tok = pick;
                    if (g.force >= 0) {   // (gen_took's: the forced token instead of the pick)
                        tok = g.force;
                        g.force = -1;
                    }
                    out = tok;
                    wr.push_back(tok);
                    g.sp.counter += 1;
                    if (!gen_token(g, tok)) {
                        ended[(size_t) i] = 1;
                        return strata::core::MtpTake::stop;
                    }
                    return g.force >= 0 ? strata::core::MtpTake::forced : strata::core::MtpTake::more;
                });
            const int k = keep[(size_t) i];
            g.mtp_steps += 1;
            g.mtp_drafts += (uint64_t) (w.counts[(size_t) i] - 1);
            g.mtp_acc += (uint64_t) (k - 1);
            for (int t = 0; t + 1 < k; ++t) written.push_back(wr[(size_t) t]);   // (the tokens at p0 + 1 .. p0 + k - 1)
            last[(size_t) i] = wr[(size_t) k - 1];
            g.drafts.clear();
            g.draft_at = -1;
            if (!ended[(size_t) i]) {
                g.pending = last[(size_t) i];
                // the next window's drafts: no more than its tokens left after the pending one and its context's
                // room (draft j's NextN cache entry is at the window's start + j - 1; the position is still this
                // window's start + its rows until the commit below)
                nd[(size_t) i] = (int) std::max<int64_t>(
                    0, std::min<int64_t>({(int64_t) D, g.max_new - g.produced - 1,
                                          model.seq_max_ctx(g.ctx) - 2 -
                                              (model.seq_pos(g.ctx) - w.counts[(size_t) i] + k)}));
            }
        }
        if (!model.mtpw_commit(w.seqs.data(), w.counts.data(), keep.data(), n, err)) return false;
        if (!model.mtpw_draft_queue(w.seqs.data(), keep.data(), n, written.data(), last.data(), nd.data(), err))
            return false;
        if (!wflight.empty() && !win_back(wflight.front())) return false;
        {
            std::vector<int32_t> out((size_t) n * (size_t) std::max(1, D), -1);
            if (!model.mtpw_draft_take(out.data(), err)) return false;
            for (int i = 0; i < n; ++i) {
                Gen& g = *w.g[(size_t) i];
                if (ended[(size_t) i]) continue;
                for (int j = 0; j < D && out[(size_t) i * D + (size_t) j] >= 0; ++j)
                    g.drafts.push_back(out[(size_t) i * D + (size_t) j]);
                g.draft_at = model.seq_pos(g.ctx);
                g.draft_tok = g.pending;
            }
        }
        for (int i = 0; i < n; ++i)
            if (ended[(size_t) i]) gen_finish(*w.g[(size_t) i]);
        since_boundary += n;
        return true;
    };
    // every issued token back (before a prompt, a boundary or the speculative path)
    const auto multi_drain = [&]() -> bool {
        while (!wflight.empty())
            if (!win_take()) return false;
        while (!flight.empty()) {
            Gen& g = *flight.front();
            flight.pop_front();
            const int tok = model.multi_take(g.ctx, err);
            g.inflight = false;
            if (tok < 0) return false;
            gen_took(g, tok);
        }
        flight_groups.clear();
        return true;
    };
    // a just-admitted request's first token, while the tail's logits are still its own
    const auto resolve_ready = [&]() -> bool {
        for (auto& gp : gens) {
            Gen& g = *gp;
            if (!g.ready || g.done) continue;
            bind_ctx(g.ctx);
            const int tok = model.sample_token(g.sp, err);
            g.ready = false;
            if (tok < 0) return false;
            g.sp.counter += 1;
            gen_took(g, model.forced(tok));
        }
        return true;
    };
    // an engine error in the middle: every running request gets it, the sequences are reset
    uint64_t multi_fails = 0;      // (a prompt read between decode steps sees one happen: STRATA_GLM_PREFILL_INTERLEAVE)
    std::string multi_fail_why;
    const auto multi_fail = [&](const std::string& why) {
        ++multi_fails;
        multi_fail_why = why;
        for (auto& gp : gens) {
            if (gp->done) continue;
            if (why.find("non-finite") != std::string::npos)   // the slots it may have come from (drop_if_poisoned)
                for (size_t i = slots.size(); i-- > 0;) {
                    const auto& t = slots[i].tokens;
                    if (t.size() <= gp->prompt.size() && std::equal(t.begin(), t.end(), gp->prompt.begin()))
                        slot_drop(i);
                }
            std::printf("ERR %d %s\n", gp->rid, why.c_str());
            ctxs[(size_t) gp->ctx].busy = false;
            gp->done = true;
        }
        flight.clear();
        flight_groups.clear();
        wflight.clear();
        if (in_read_gap) {   // (a prompt is being read: its call ends first, then gap_reset resets them all)
            gap_failed = true;
            prompt_between_err = "the engine failed while other requests decoded during this prompt's read: " + why;
            gen_sweep();
            return;
        }
        for (const auto& d : deferred_pauses) ctxs[(size_t) d.ctx].busy = false;   // (their states go with the rest)
        deferred_pauses.clear();
        for (int c = 0; c < (int) ctxs.size(); ++c) {
            bind_ctx(c);
            snap_tokens.clear();
            model.reset();
        }
        gen_sweep();
    };
    // the contexts' reset a failed gap left (STRATA_GLM_PREFILL_GAPS; multi_fail), once the read has ended
    gap_reset = [&] {
        if (!gap_failed) return;
        gap_failed = false;
        for (const auto& d : deferred_pauses) ctxs[(size_t) d.ctx].busy = false;   // (their states go with the rest)
        deferred_pauses.clear();
        for (int c = 0; c < (int) ctxs.size(); ++c) {
            bind_ctx(c);
            snap_tokens.clear();
            model.reset();
        }
    };
    // the decode steps between two pieces of a prompt being read in context read_ctx (STRATA_GLM_PREFILL_INTERLEAVE;
    // set below, after multi_step): prompt_between's 0 / 1 / 2
    std::function<int(int)> interleave_round;
    // one decode gap of a prompt being read (STRATA_GLM_PREFILL_GAPS; set below, after multi_step): 0 / 1 / 2 as
    // prompt_between's; gap_mark: the read's start, then each gap's end
    std::function<int()> read_gap;
    uint64_t gap_count = 0, gap_tok = 0;
    double gap_ms = 0.0;
    // the last gap's decode round (ms; 0: none yet) - a boundary's gap is due once its budget is half of one
    // (prefill_gap_due; STRATA_GLM_PREFILL_GAP_CREDIT=0: every boundary, as before)
    double gap_round_ms = 0.0;
    static const bool gap_credit = env_num("STRATA_GLM_PREFILL_GAP_CREDIT", 1) != 0;
    const auto gen_admit = [&](int rid, const std::vector<int64_t>& ids, int64_t max_new,
                               const strata::kernels::SamplerParams& rq) {
        if (!multi_drain() || !resolve_ready()) {
            const std::string why = err;
            multi_fail(why);
            std::printf("ERR %d %s\n", rid, why.c_str());   // (this request too: its client would wait forever)
            return;
        }
        if (!deferred_pauses.empty()) flush_pauses();   // (a PAUSE in the last read's gaps: its state, kept first)
        auto g = std::make_unique<Gen>();
        g->rid = rid;
        g->sp = rq;
        g->max_new = max_new;
        g->in_think = req_think_budget > 0 && req_think_end >= 0;
        g->think_budget = req_think_budget;
        g->think_end = req_think_end;
        for (const int64_t t : ids) {
            if (t < 0 || t >= n_vocab) {
                std::printf("ERR %d token %lld out of range [0, %lld)\n", rid, (long long) t, (long long) n_vocab);
                return;
            }
            g->prompt.push_back((int32_t) t);
        }
        if (g->prompt.empty() || max_new <= 0) {
            std::printf("ERR %d empty prompt or max_new\n", rid);
            return;
        }
        if (g->sp.penalty_last_n > 0) {   // its penalties' history (#273; the Gen lives behind a unique_ptr: it stays put)
            g->pen = g->prompt;
            g->sp.host_history = &g->pen;
        }
        // the context: an idle one that holds the prompt and the room after it, and whose snapshot this prompt
        // extends (the longest), else the least recently used of those
        const int64_t want = (int64_t) g->prompt.size() + std::min<int64_t>(max_new, multi_room) + 8;
        const auto fits = [&](int c) { return model.seq_max_ctx(c) >= want; };
        int best = -1;
        size_t blen = 0;
        for (int c = 0; c < (int) ctxs.size(); ++c) {
            if (ctxs[(size_t) c].busy || !fits(c)) continue;
            const auto& st = c == cur_ctx ? snap_tokens : ctxs[(size_t) c].snap_tokens;
            const uint64_t hh = c == cur_ctx ? snap_img_hash : ctxs[(size_t) c].snap_img_hash;
            if (!st.empty() && st.size() < g->prompt.size() && st.size() > blen &&
                std::equal(st.begin(), st.end(), g->prompt.begin()) && img_hash(st.size()) == hh) {
                best = c;
                blen = st.size();
            }
        }
        if (best < 0)
            for (int c = 0; c < (int) ctxs.size(); ++c)
                if (!ctxs[(size_t) c].busy && fits(c) &&
                    (best < 0 || ctxs[(size_t) c].used < ctxs[(size_t) best].used))
                    best = c;
        if (best < 0) {
            bool ever = false;
            for (int c = 0; c < (int) ctxs.size(); ++c) ever = ever || fits(c);
            if (!ever) {
                std::printf("ERR %d prompt (%zu tokens) + room for the answer exceeds every context\n", rid,
                            g->prompt.size());
                return;
            }
            pending.push_back({rid, ids, max_new, rq, req_think_budget, req_think_end, cur_img_pos, cur_img_rows,
                               req_bias});
            std::fprintf(stderr, "glm multi: request %d waits for a context that holds %lld tokens\n", rid,
                         (long long) want);
            return;
        }
        g->max_new = std::min<int64_t>(max_new, model.seq_max_ctx(best) - (int64_t) g->prompt.size() - 8);
        {   // its logit_bias: the table of the context it runs in (no earlier request's sample reads it any more)
            std::string be;
            g->sp.logit_bias = model.logit_bias_table(best, req_bias, be);
            if (!be.empty()) {
                std::printf("ERR %d %s\n", rid, be.c_str());
                return;
            }
        }
        if (gens.empty()) stop_req.store(false);
        gap_failed = false;
        bind_ctx(best);
        model.set_image_rows(cur_img_pos, cur_img_rows);
        model.force_next(-1);
        ctxs[(size_t) best].busy = true;
        g->ctx = best;
        tag = std::to_string(rid) + " ";
        // STRATA_GLM_PREFILL_GAPS: the running requests decode at the read's chunk boundaries and between its calls
        // (the split's batched rows: they never bind a context; text prompts only, as interleave)
        const bool use_gaps = read_gaps.percent > 0 && cur_img_pos.empty() && split_rows > 0 && model.split_rows_ok();
        if (read_gaps.percent > 0 && !use_gaps && !gens.empty()) {
            static bool told = false;
            if (!cur_img_pos.empty()) {
                std::fprintf(stderr, "glm multi: request %d's prompt has images: read whole, no decode gaps "
                                     "(MAYA_GLM_PREFILL_GAPS is for text prompts)\n", rid);
            } else if (!told) {
                told = true;
                std::fprintf(stderr, "glm multi: MAYA_GLM_PREFILL_GAPS needs a split with MAYA_GLM_SPLIT_ROWS: off "
                                     "here%s\n", interleave.tokens > 0 ? " (MAYA_GLM_PREFILL_INTERLEAVE instead)" : "");
            }
        }
        gaps_over = false;
        if (use_gaps) {
            model.prefill_gap_wanted = [&] {
                if (gaps_over ||
                    !std::any_of(gens.begin(), gens.end(), [](const std::unique_ptr<Gen>& p) { return !p->done; }))
                    return false;
                if (!gap_credit) return true;
                const double stage_ms =
                    std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - gap_mark).count();
                return strata::core::prefill_gap_due(read_gaps, stage_ms, gap_round_ms);
            };
            model.prefill_gap = [&] { return read_gap(); };
            prompt_between = [&] { return read_gap(); };
            read_gap_first = true;
            gap_count = gap_tok = 0;
            gap_ms = 0.0;
        }
        // STRATA_GLM_PREFILL_INTERLEAVE: while other requests decode, the prompt in pieces with their steps between
        // (text prompts only: a prompt's image rows are the driver's globals while it is read)
        if (interleave.tokens > 0 && read_gaps.percent == 0 && !cur_img_pos.empty() && !gens.empty())
            std::fprintf(stderr, "glm multi: request %d's prompt has images: read whole, not in pieces "
                                 "(MAYA_GLM_PREFILL_INTERLEAVE is for text prompts)\n", rid);
        if (!use_gaps && interleave.tokens > 0 && cur_img_pos.empty()) {
            // (STRATA_GLM_PREFILL_INTERLEAVE_ALWAYS=1, a test: the pieces with nothing decoding too - to compare with
            // the prompt read whole, the decode alone)
            static const bool always = env_num("STRATA_GLM_PREFILL_INTERLEAVE_ALWAYS", 0) != 0;
            prompt_piece = [&]() -> size_t {
                for (const auto& gp : gens)
                    if (!gp->done) return interleave.tokens;
                return always ? interleave.tokens : 0;   // nothing decodes (any more): the rest in one call, as before
            };
            prompt_between = [&, best]() { return interleave_round(best); };
        }
        admit_stop.store(false);
        admitting_rid.store(rid);
        const int rc = prep_prompt(g->prompt, g->reuse, g->finish, g->t0);
        admitting_rid.store(-1);
        prompt_piece = nullptr;
        prompt_between = nullptr;
        model.prefill_gap = nullptr;
        model.prefill_gap_wanted = nullptr;
        read_gap_first = false;
        if (use_gaps && gap_count > 0)
            std::fprintf(stderr, "glm multi: request %d's prompt (%zu tokens) read with %llu decode gaps: %.1f s of "
                                 "decode, %llu tokens for the others (%.1f tok/s there)\n",
                         rid, g->prompt.size(), (unsigned long long) gap_count, gap_ms / 1000.0,
                         (unsigned long long) gap_tok, 1000.0 * (double) gap_tok / std::max(1.0, gap_ms));
        tag.clear();
        if (rc != 0) {
            ctxs[(size_t) best].busy = false;
            admit_stop.store(false);
            std::lock_guard<std::mutex> lk(ctl_mu);   // (a STOP for it that came during its prompt: nothing to stop now)
            ctl.erase(rid);
            return;
        }
        // a STOP <rid> during its prompt (with interleave: admit_stop alone)
        const bool stopped_all = stop_req.exchange(false), stopped_this = admit_stop.exchange(false);
        if (stopped_all || stopped_this) g->finish = "cancel";
        g->t1 = std::chrono::steady_clock::now();
        g->st1 = model.fast_stats();
        g->spec_steps0 = model.spec_steps_;
        g->spec_hits0 = model.spec_hits_;
        if (gens.empty()) {
            multi_n = 0;
            multi_t1 = g->t1;
            multi_st1 = g->st1;
        }
        if (std::strcmp(g->finish, "cancel") == 0) {
            gens.push_back(std::move(g));
            gen_finish(*gens.back());
            gen_sweep();
            return;
        }
        g->ready = true;
        gens.push_back(std::move(g));
        if (gens.size() > 1 && !resolve_ready()) multi_fail(err);   // the others decode: its first token now
        if (!deferred_pauses.empty() && gens.size() > 1) flush_pauses();   // (its first token taken: the tail is free)
        gen_sweep();
    };
    // the waiting requests, in order, that an idle context can now hold (a STOP for one ends it unread)
    const auto admit_pending = [&] {
        for (size_t i = 0; i < pending.size();) {
            const Pending& w = pending[i];
            bool stopped = false;
            {
                std::lock_guard<std::mutex> lk(ctl_mu);
                const auto it = ctl.find(w.rid);
                if (it != ctl.end()) {
                    stopped = true;
                    ctl.erase(it);
                }
            }
            if (stopped) {
                std::printf("DONE %d 0 %zu 0.0 0.0 cancel 0 0 0 0 0\n", w.rid, w.ids.size());
                pending.erase(pending.begin() + (long) i);
                continue;
            }
            const int64_t want = (int64_t) w.ids.size() + std::min<int64_t>(w.max_new, multi_room) + 8;
            bool free_fit = false;
            for (int c = 0; c < (int) ctxs.size(); ++c)
                free_fit = free_fit || (!ctxs[(size_t) c].busy && model.seq_max_ctx(c) >= want);
            if (!free_fit) {
                ++i;
                continue;
            }
            Pending w2 = std::move(pending[i]);
            pending.erase(pending.begin() + (long) i);
            req_think_budget = w2.think_budget;
            req_think_end = w2.think_end;
            req_bias = std::move(w2.bias);
            cur_img_pos = std::move(w2.img_pos);
            cur_img_rows = std::move(w2.img_rows);
            gen_admit(w2.rid, w2.ids, w2.max_new, w2.rq);
        }
    };
    // one step of the scheduler: a stint of the speculative path for a request alone, else one token back from the
    // pipelined decode (and every request with a token to feed issued first).  Errors end the running requests.
    bool step_no_spec = false;   // (between a prompt's pieces: the speculative stint runs until input comes - not there)
    const auto multi_step = [&] {
        std::vector<float> lg;
        // (a PAUSE in the last read's gaps: its snapshot, once no request waits for its first token)
        if (!deferred_pauses.empty() && !in_read_gap && admitting_rid.load() < 0 &&
            std::none_of(gens.begin(), gens.end(), [](const std::unique_ptr<Gen>& p) { return p->ready && !p->done; }))
            flush_pauses();
        Gen* only = gens.size() == 1 ? gens[0].get() : nullptr;
        if (only != nullptr && !only->inflight && multi_spec && !step_no_spec && model.spec_ready()) {
            Gen& g = *only;
            bind_ctx(g.ctx);
            if (g.pending >= 0) {   // its last token through the token path: the next one's logits on the tail
                if (!model.forward({g.pending}, lg, err)) {
                    multi_fail(err);
                    return;
                }
                g.pending = -1;
            }
            g.ready = false;
            model.force_next(g.force);
            g.force = -1;
            int32_t last = -1, forced_unused = -1;
            bool interrupted = false;
            int64_t n_spec = 0;
            const bool ok = model.decode_spec(g.sp, g.max_new - g.produced, [&](int tok) {
                forced_unused = -1;   // a force handed over is consumed by the token that comes next
                last = tok;
                if (!gen_token(g, tok)) return false;
                if (g.force >= 0) {
                    model.force_next(g.force);
                    forced_unused = g.force;
                    g.force = -1;
                }
                if (input_pending()) {
                    interrupted = true;
                    return false;
                }
                return true;
            }, n_spec, err);
            if (forced_unused >= 0) {   // handed over, never taken: it stays the request's
                g.force = forced_unused;
                model.force_next(-1);
            }
            if (!ok) {
                multi_fail(err);
                return;
            }
            // not interrupted by input: decode_spec ended the request (a stop, its budget, the context)
            if (interrupted) g.pending = last;
            else gen_finish(g);
            gen_sweep();
            return;
        }
        // a request admitted alone that the speculative path does not take: its first token, from its prompt's logits
        if (!resolve_ready()) {
            multi_fail(err);
            return;
        }
        // STRATA_GLM_MULTI_MTP: the ready requests' windows (their last token and the drafts made for the positions
        // after it) in groups as the split's batched rows take them, each group through every part but the last;
        // then the oldest group back (win_take) - its last part queued now if it was not
        // (STRATA_GLM_MULTI_MTP_MAX, 4: the windows only while that many requests or fewer run - with more, every
        // GPU is already busy and a window's rows cost about what the tokens they save cost; 0: always. Switching
        // drains what the other way has in flight)
        static const int mtp_max = (int) env_num("STRATA_GLM_MULTI_MTP_MAX", 4);
        int n_run = 0;
        for (auto& gp : gens) n_run += !gp->done;
        const bool use_win = mtp_rows && (mtp_max <= 0 || n_run <= mtp_max);
        if ((use_win && !flight.empty()) || (!use_win && !wflight.empty())) {
            if (!multi_drain()) {
                multi_fail(err);
                return;
            }
            gen_sweep();
            return;
        }
        if (use_win) {
            int ready = 0, active = 0;
            for (auto& gp : gens) {
                if (gp->done) continue;
                ready += !gp->inflight && gp->pending >= 0;
                active += gp->inflight || gp->pending >= 0;
            }
            const int k = std::max(1, split_rows > 0 ? split_rows : model.seq_count());
            const std::vector<int> sizes = strata::core::split_groups(ready, active, k, even_groups);
            std::vector<Gen*> rdy;
            for (auto& gp : gens)
                if (!gp->done && !gp->inflight && gp->pending >= 0) rdy.push_back(gp.get());
            size_t at = 0;
            for (const int gs : sizes) {
                WinGroup w;
                for (int j = 0; j < gs && at < rdy.size(); ++j) w.g.push_back(rdy[at++]);
                for (Gen* gp : w.g) {
                    const int64_t pos = model.seq_pos(gp->ctx);
                    const int d = gp->draft_at == pos && gp->draft_tok == gp->pending ? (int) gp->drafts.size() : 0;
                    w.seqs.push_back(gp->ctx);
                    w.counts.push_back(strata::core::mtp_rows_count(d, gp->max_new - gp->produced,
                                                                    model.seq_max_ctx(gp->ctx) - 1 - pos));
                    w.sps.push_back(gp->sp);
                }
                strata::core::mtp_rows_fit(w.counts, strata::core::Glm5Model::rows_max());
                for (size_t j = 0; j < w.g.size(); ++j) {
                    w.toks.push_back(w.g[j]->pending);
                    for (int t = 1; t < w.counts[j]; ++t) w.toks.push_back(w.g[j]->drafts[(size_t) t - 1]);
                }
                if (w.g.empty()) continue;
                if (!model.mtpw_front(w.seqs.data(), w.counts.data(), (int) w.seqs.size(), w.toks.data(), err)) {
                    multi_fail(err);
                    return;
                }
                for (Gen* gp : w.g) gp->inflight = true;
                wflight.push_back(std::move(w));
            }
            if (wflight.empty()) {
                gen_sweep();
                return;
            }
            if (!win_take()) {
                multi_fail(err);
                return;
            }
            if (multi_boundary_every > 0 &&
                since_boundary >= multi_boundary_every * (int) std::max<size_t>(1, gens.size())) {
                since_boundary = 0;
                if (!multi_drain() || !model.multi_boundary(err)) {
                    multi_fail(err);
                    return;
                }
            }
            gen_sweep();
            return;
        }
        // ONE GPU: every request with a token to feed goes through one step together, each weight read once for all
        if (model.rows_ok()) {
            std::vector<Gen*> rg;
            std::vector<int> seqs, out;
            std::vector<int32_t> toks;
            std::vector<strata::kernels::SamplerParams> sps;
            for (auto& gp : gens)
                if (!gp->done && gp->pending >= 0 && (int) rg.size() < strata::core::Glm5Model::rows_max()) {
                    rg.push_back(gp.get());
                    seqs.push_back(gp->ctx);
                    toks.push_back(gp->pending);
                    sps.push_back(gp->sp);
                }
            if (!rg.empty()) {
                out.resize(rg.size());
                if (!model.multi_rows(seqs.data(), toks.data(), sps.data(), (int) rg.size(), out.data(), err)) {
                    multi_fail(err);
                    return;
                }
                for (size_t i = 0; i < rg.size(); ++i) {   // (the step keeps the tiers' boundary itself)
                    Gen& g = *rg[i];
                    g.sp.counter += 1;
                    g.pending = -1;
                    gen_took(g, out[i]);
                }
            }
            gen_sweep();
            return;
        }
        if (split_rows > 0 && model.split_rows_ok()) {
            split_group.clear();
            const auto issue = [&]() -> bool {          // one batched step of the gathered group
                if (split_group.empty()) return true;
                split_seqs.clear();
                split_toks.clear();
                split_sps.clear();
                for (Gen* gp : split_group) {
                    split_seqs.push_back(gp->ctx);
                    split_toks.push_back(gp->pending);
                    split_sps.push_back(gp->sp);
                }
                if (!model.multi_issue_rows(split_seqs.data(), split_toks.data(), split_sps.data(),
                                           (int) split_group.size(), err))
                    return false;
                for (Gen* gp : split_group) {
                    gp->sp.counter += 1;
                    gp->pending = -1;
                    gp->inflight = true;
                    flight.push_back(gp);
                }
                flight_groups.push_back((int) split_group.size());
                split_group.clear();
                return true;
            };
            int ready = 0, active = 0;
            for (auto& gp : gens) {
                if (gp->done) continue;
                ready += !gp->inflight && gp->pending >= 0;
                active += gp->inflight || gp->pending >= 0;
            }
            const std::vector<int> sizes = strata::core::split_groups(ready, active, split_rows, even_groups);
            size_t gi = 0;
            for (auto& gp : gens) {
                if (gp->done || gp->inflight || gp->pending < 0) continue;
                split_group.push_back(gp.get());
                if (gi < sizes.size() && (int) split_group.size() == sizes[gi]) {
                    ++gi;
                    if (!issue()) { multi_fail(err); return; }
                }
            }
            if (!issue()) { multi_fail(err); return; }
            if (flight.empty()) { gen_sweep(); return; }
            const int k = flight_groups.front();   // the oldest group, all of it
            flight_groups.pop_front();
            for (int j = 0; j < k; ++j) {
                Gen& g = *flight.front();
                flight.pop_front();
                const int tok = model.multi_take(g.ctx, err);
                g.inflight = false;
                if (tok < 0) { multi_fail(err); return; }
                gen_took(g, tok);
            }
            since_boundary += k;
            if (multi_boundary_every > 0 && !in_read_gap &&
                since_boundary >= multi_boundary_every * (int) std::max<size_t>(1, gens.size())) {
                since_boundary = 0;
                if (!multi_drain() || !model.multi_boundary(err)) { multi_fail(err); return; }
            }
            gen_sweep();
            return;
        }
        for (auto& gp : gens) {
            Gen& g = *gp;
            if (g.done || g.inflight || g.pending < 0) continue;
            if (!model.multi_issue(g.ctx, g.pending, g.sp, err)) {
                multi_fail(err);
                return;
            }
            g.sp.counter += 1;
            g.pending = -1;
            g.inflight = true;
            flight.push_back(&g);
        }
        if (flight.empty()) {
            gen_sweep();
            return;
        }
        Gen& g = *flight.front();
        flight.pop_front();
        const int tok = model.multi_take(g.ctx, err);
        g.inflight = false;
        if (tok < 0) {
            multi_fail(err);
            return;
        }
        gen_took(g, tok);
        if (multi_boundary_every > 0 && ++since_boundary >= multi_boundary_every * (int) std::max<size_t>(1, gens.size())) {
            since_boundary = 0;
            if (!multi_drain() || !model.multi_boundary(err)) {
                multi_fail(err);
                return;
            }
        }
        gen_sweep();
    };

    // Between two pieces of a prompt being read (STRATA_GLM_PREFILL_INTERLEAVE): `rounds` rounds of the scheduler's own
    // steps - each until every request running at its start has a token more (or ended) - then every issued token back
    // (the prompt path needs the GPUs to itself) and the reading context bound again.  The steps are multi_step's, so
    // a request that ends meanwhile ends as it would (DONE, its pause snapshot), and a STOP for one ends it; an engine
    // error fails them all (multi_fail resets every context, the one being read too).  The new prompt's own STOP is
    // admit_stop.  0: go on, 1: the read was stopped, 2: failed (prompt_between_err).
    interleave_round = [&](int read_ctx) -> int {
        if (read_stopped()) return 1;
        if (gens.empty()) return 0;   // (nothing decodes: no sweep either - an empty one clears stop_req)
        const uint64_t fails0 = multi_fails;
        const auto failed = [&] {
            prompt_between_err = "the engine failed while other requests decoded between this prompt's pieces: " +
                                 multi_fail_why;
            return 2;
        };
        step_no_spec = true;
        for (int r = 0; r < interleave.rounds && !gens.empty(); ++r) {
            std::map<int, int64_t> want;   // rid -> the produced count that ends this round for it
            for (const auto& gp : gens)
                if (!gp->done) want[gp->rid] = gp->produced + 1;
            // (bounded: a step takes one token back at least while any request runs)
            for (int it = 0; it < 4 * (int) want.size() + 8 && !want.empty(); ++it) {
                multi_step();
                if (multi_fails != fails0) {
                    step_no_spec = false;
                    return failed();
                }
                for (auto w = want.begin(); w != want.end();) {
                    const Gen* g = nullptr;
                    for (const auto& gp : gens)
                        if (gp->rid == w->first) g = gp.get();
                    w = g == nullptr || g->done || g->produced >= w->second ? want.erase(w) : std::next(w);
                }
            }
        }
        step_no_spec = false;
        if (!multi_drain()) {
            multi_fail(err);
            return failed();
        }
        if (std::any_of(gens.begin(), gens.end(), [](const std::unique_ptr<Gen>& p) { return p->done; })) gen_sweep();
        bind_ctx(read_ctx);
        return read_stopped() ? 1 : 0;
    };

    // One decode gap of a prompt's read (STRATA_GLM_PREFILL_GAPS): at a chunk boundary inside a prompt-path call (every
    // part idle, the pool's tail lent; Glm5Model::prefill_gap) or between two of its calls.  The running requests' own
    // steps (multi_step's: the split's batched rows, which never bind a context) in rounds - each until every request
    // running at its start has a token more or ended - for percent / (100 - percent) of the time the read ran since
    // the last gap (one round at least), then every issued token back: the read goes on with the GPUs to itself.  No
    // tier boundary runs in it (in_read_gap), a request that ends meanwhile ends as it would (a PAUSE's snapshot after
    // the read: deferred_pauses), a STOP for one ends it; an engine error fails them all, the contexts' reset waits
    // for the read's end (gap_reset).  A gap that overruns <max gap ms> (its one round can, where experts stream) ends
    // the gaps of this read (gaps_over).  0: go on, 1: the read was stopped, 2: failed (prompt_between_err).
    read_gap = [&]() -> int {
        if (read_stopped()) return 1;
        const auto t_in = std::chrono::steady_clock::now();
        const double stage_ms = std::chrono::duration<double, std::milli>(t_in - gap_mark).count();
        if (gens.empty() || gaps_over) {
            gap_mark = t_in;
            return 0;
        }
        const double budget = strata::core::prefill_gap_budget_ms(read_gaps, stage_ms);
        const uint64_t fails0 = multi_fails, n0 = multi_n;
        in_read_gap = true;
        step_no_spec = true;
        int rounds = 0;
        for (bool first = true; !gens.empty() && multi_fails == fails0; first = false) {
            if (!first && std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t_in).count() >=
                              budget)
                break;
            ++rounds;
            std::map<int, int64_t> want;   // rid -> the produced count that ends this round for it
            for (const auto& gp : gens)
                if (!gp->done) want[gp->rid] = gp->produced + 1;
            if (want.empty()) break;   // (every request ended: nothing to decode)
            for (int it = 0; it < 4 * (int) want.size() + 8 && !want.empty() && multi_fails == fails0; ++it) {
                multi_step();
                for (auto w = want.begin(); w != want.end();) {
                    const Gen* g = nullptr;
                    for (const auto& gp : gens)
                        if (gp->rid == w->first) g = gp.get();
                    w = g == nullptr || g->done || g->produced >= w->second ? want.erase(w) : std::next(w);
                }
            }
        }
        if (multi_fails == fails0 && !multi_drain()) multi_fail(err);
        step_no_spec = false;
        in_read_gap = false;
        gap_mark = std::chrono::steady_clock::now();
        ++gap_count;
        gap_tok += multi_n - n0;
        const double this_ms = std::chrono::duration<double, std::milli>(gap_mark - t_in).count();
        gap_ms += this_ms;
        if (rounds > 0) gap_round_ms = this_ms / rounds;   // (the drain's share too: a gap costs it once)
        if (this_ms > read_gaps.max_ms && !gaps_over) {
            // (a round is one token for every running request, whatever the budget: where their experts stream from
            // RAM or disk inside the loan it can take far longer than the stage - the rest of this read has no gaps)
            gaps_over = true;
            std::fprintf(stderr, "glm multi: a decode gap took %.0f ms (more than %.0f): no more gaps in this read\n",
                         this_ms, read_gaps.max_ms);
        }
        if (multi_fails != fails0) return 2;
        if (std::any_of(gens.begin(), gens.end(), [](const std::unique_ptr<Gen>& p) { return p->done; })) gen_sweep();
        return read_stopped() ? 1 : 0;
    };

    // STRATA_GLM_MULTI_BENCH=<n>,<prompt>,<gen>[,<boundary rounds>] (needs STRATA_GLM_SEQS >= n): n synthetic
    // prompts of <prompt> tokens, then <gen> greedy tokens each - sequence 0 alone through the token path first (the
    // reference), then all n through the selected together, split-rows or pipelined path; prints both rates and whether
    // same tokens both ways.
    if (const char* mb = getenv("STRATA_GLM_MULTI_BENCH")) {
        int n = 2, L = 512, G = 256, K = 32, same_prompt = 0;
        std::sscanf(mb, "%d,%d,%d,%d,%d", &n, &L, &G, &K, &same_prompt);
        if (n > model.seq_count() || n < 1 || L < 2 || G < 2) {
            std::fprintf(stderr, "multi bench: %d sequences asked, %d states (MAYA_GLM_SEQS)\n", n, model.seq_count());
            return 2;
        }
        model.set_host_logits(false);
        strata::kernels::SamplerParams g0 = sp;
        g0.greedy = true;
        std::vector<std::vector<int32_t>> prompts((size_t) n);
        uint64_t rs = 0x9e3779b97f4a7c15ull;
        for (int s = 0; s < n; ++s) {
            if (same_prompt) rs = 0x9e3779b97f4a7c15ull;
            for (int i = 0; i < L; ++i) {
                if (!o.tokens.empty()) {   // --tokens: a real prompt (repeated / cut to L), the same for every sequence
                    prompts[(size_t) s].push_back((int32_t) o.tokens[(size_t) i % o.tokens.size()]);
                    continue;
                }
                rs = rs * 6364136223846793005ull + 1442695040888963407ull;
                prompts[(size_t) s].push_back((int32_t) (1000 + (rs >> 33) % 120000));
            }
        }
        std::vector<float> lg;
        const auto now = [] { return std::chrono::steady_clock::now(); };
        const auto secs = [](auto a, auto b) { return std::chrono::duration<double>(b - a).count(); };
        for (int s = 0; s < n; ++s) {   // every prompt read once, its state saved before its last token
            model.seq_bind_all(s);
            model.reset();
            const std::vector<int32_t> head(prompts[(size_t) s].begin(), prompts[(size_t) s].end() - 1);
            if (!model.prefill(head, err, prompts[(size_t) s].back()) && !err.empty()) {
                std::fprintf(stderr, "multi bench: prefill: %s\n", err.c_str());
                return 1;
            }
            if (!model.snapshot_save()) {
                std::fprintf(stderr, "multi bench: snapshot\n");
                return 1;
            }
        }
        // the reference: sequence 0 alone, the classic token path - once untimed first (the tiers warm up on it, as
        // they would before any timed run; and is the token path itself repeatable here?), then timed
        std::vector<int> solo, solo2;
        model.seq_bind_all(0);
        double solo_s = 0;
        strata::core::Glm5Model::FastStats st_a, st_b;   // the expert tiers over each timed phase
        const auto tiers = [](const char* what, const strata::core::Glm5Model::FastStats& a,
                              const strata::core::Glm5Model::FastStats& b, int tokens) {
            const double h = (double) (b.hits - a.hits), m = (double) (b.misses - a.misses), t = std::max(1, tokens);
            std::printf("  %s: VRAM hit %.1f%%, a token: %.1f RAM fetches, %.2f disk reads, %.1f promotions, %.1f CPU "
                        "experts\n", what, 100.0 * h / std::max(1.0, h + m), (double) (b.ram_hits - a.ram_hits) / t,
                        (double) (b.disk_reads - a.disk_reads) / t, (double) (b.promotions - a.promotions) / t,
                        (double) (b.cpu_experts - a.cpu_experts) / t);
        };
        for (std::vector<int>* run : {&solo2, &solo}) {
            model.snapshot_restore();
            st_a = model.fast_stats();
            if (!model.forward({prompts[0].back()}, lg, err)) { std::fprintf(stderr, "multi bench: %s\n", err.c_str()); return 1; }
            const auto t0 = now();
            run->push_back(model.last_tok_);
            while ((int) run->size() < G) {
                if (!model.forward({(int32_t) run->back()}, lg, err)) { std::fprintf(stderr, "multi bench: %s\n", err.c_str()); return 1; }
                run->push_back(model.last_tok_);
            }
            solo_s = secs(t0, now());
            st_b = model.fast_stats();
        }
        const strata::core::Glm5Model::FastStats solo_a = st_a, solo_b = st_b;
        // all n through the pipelined decode
        std::vector<std::vector<int>> out((size_t) n);
        for (int s = 0; s < n; ++s) {
            model.seq_bind_all(s);
            model.snapshot_restore();
            if (!model.forward({prompts[(size_t) s].back()}, lg, err)) { std::fprintf(stderr, "multi bench: %s\n", err.c_str()); return 1; }
            out[(size_t) s].push_back(model.last_tok_);
        }
        std::deque<int> q;
        st_a = model.fast_stats();
        const auto t0 = now();
        int since = 0, boundaries = 0;
        // STRATA_GLM_MULTI_MTP: every sequence's verify windows through the batched rows (the server's order of calls)
        const bool wrows = model.mtpw_ok() && (model.rows_ok() || (split_rows > 0 && model.split_rows_ok()));
        const bool rows = model.rows_ok() && !wrows;   // one GPU: all n together, one step per token of each
        const bool srows = !rows && !wrows && split_rows > 0 && model.split_rows_ok();
        const char* path = wrows ? "mtp-windows" : rows ? "together" : srows ? "split-rows" : "pipelined";
        uint64_t w_steps = 0, w_drafts = 0, w_acc = 0;
        if (wrows) {
            const int D = strata::core::Glm5Model::multi_mtp_setting();
            std::vector<std::vector<int32_t>> drafts((size_t) n);
            struct BG {
                std::vector<int> seqs, counts;
                std::vector<int32_t> toks;
                bool back = false;
            };
            const auto back = [&](BG& b) -> bool {
                if (b.back) return true;
                b.back = true;
                const std::vector<strata::kernels::SamplerParams> sps(b.seqs.size(), g0);
                return model.mtpw_back(b.seqs.data(), b.counts.data(), (int) b.seqs.size(), b.toks.data(), sps.data(),
                                       err);
            };
            for (;;) {
                std::vector<int> act;
                for (int s = 0; s < n; ++s)
                    if ((int) out[(size_t) s].size() < G) act.push_back(s);
                if (act.empty()) break;
                std::vector<BG> gs;
                size_t first = 0;
                for (const int k : strata::core::split_groups((int) act.size(), (int) act.size(),
                                                              std::max(1, split_rows > 0 ? split_rows : n), even_groups)) {
                    BG b;
                    for (int j = 0; j < k; ++j) {
                        const int s = act[first + (size_t) j];
                        b.seqs.push_back(s);
                        b.counts.push_back(strata::core::mtp_rows_count(
                            (int) drafts[(size_t) s].size(), G - (int64_t) out[(size_t) s].size(),
                            model.seq_max_ctx(s) - 1 - model.seq_pos(s)));
                    }
                    first += (size_t) k;
                    strata::core::mtp_rows_fit(b.counts, strata::core::Glm5Model::rows_max());
                    for (size_t j = 0; j < b.seqs.size(); ++j) {
                        const int s = b.seqs[j];
                        b.toks.push_back((int32_t) out[(size_t) s].back());
                        for (int t = 1; t < b.counts[j]; ++t) b.toks.push_back(drafts[(size_t) s][(size_t) t - 1]);
                    }
                    if (!model.mtpw_front(b.seqs.data(), b.counts.data(), (int) b.seqs.size(), b.toks.data(), err)) {
                        std::fprintf(stderr, "multi bench: %s\n", err.c_str());
                        return 1;
                    }
                    gs.push_back(std::move(b));
                }
                for (size_t gi = 0; gi < gs.size(); ++gi) {
                    BG& b = gs[gi];
                    const int m = (int) b.seqs.size();
                    int rows_n = 0;
                    for (const int c : b.counts) rows_n += c;
                    std::vector<int32_t> picks((size_t) rows_n), written, last((size_t) m);
                    std::vector<int> keep((size_t) m), nd((size_t) m, D);
                    if (!back(b) || !model.mtpw_picks(b.seqs.data(), b.counts.data(), m, picks.data(), err)) {
                        std::fprintf(stderr, "multi bench: %s\n", err.c_str());
                        return 1;
                    }
                    for (int i = 0, row = 0; i < m; row += b.counts[(size_t) i], ++i) {
                        const int s = b.seqs[(size_t) i];
                        std::vector<int32_t> wr;
                        keep[(size_t) i] = strata::core::mtp_accept(
                            b.toks.data() + row, picks.data() + row, b.counts[(size_t) i],
                            [&](int, int32_t pick, int32_t&) {
                                wr.push_back(pick);
                                out[(size_t) s].push_back(pick);
                                return (int) out[(size_t) s].size() < G ? strata::core::MtpTake::more
                                                                        : strata::core::MtpTake::stop;
                            });
                        const int k = keep[(size_t) i];
                        ++w_steps;
                        w_drafts += (uint64_t) (b.counts[(size_t) i] - 1);
                        w_acc += (uint64_t) (k - 1);
                        for (int t = 0; t + 1 < k; ++t) written.push_back(wr[(size_t) t]);
                        last[(size_t) i] = wr[(size_t) k - 1];
                        // (no more drafts than the tokens left after the pending one, nor past the context)
                        nd[(size_t) i] = (int) std::max<int64_t>(
                            0, std::min<int64_t>({(int64_t) D, (int64_t) G - (int64_t) out[(size_t) s].size() - 1,
                                                  model.seq_max_ctx(s) - 2 -
                                                      (model.seq_pos(s) - b.counts[(size_t) i] + k)}));
                    }
                    std::vector<int32_t> dr((size_t) m * (size_t) std::max(1, D), -1);
                    if (!model.mtpw_commit(b.seqs.data(), b.counts.data(), keep.data(), m, err) ||
                        !model.mtpw_draft_queue(b.seqs.data(), keep.data(), m, written.data(), last.data(), nd.data(),
                                                err) ||
                        (gi + 1 < gs.size() && !back(gs[gi + 1])) || !model.mtpw_draft_take(dr.data(), err)) {
                        std::fprintf(stderr, "multi bench: %s\n", err.c_str());
                        return 1;
                    }
                    for (int i = 0; i < m; ++i) {
                        auto& d = drafts[(size_t) b.seqs[(size_t) i]];
                        d.clear();
                        for (int j = 0; j < D && dr[(size_t) i * D + (size_t) j] >= 0; ++j)
                            d.push_back(dr[(size_t) i * D + (size_t) j]);
                    }
                    since += m;
                }
                if (K > 0 && since >= K * n) {
                    if (!model.multi_boundary(err)) {
                        std::fprintf(stderr, "multi bench: %s\n", err.c_str());
                        return 1;
                    }
                    ++boundaries;
                    since = 0;
                }
            }
            std::printf("  mtp windows: %llu of %llu drafts accepted (%.1f%%), %.2f tokens a window\n",
                        (unsigned long long) w_acc, (unsigned long long) w_drafts,
                        100.0 * (double) w_acc / (double) std::max<uint64_t>(1, w_drafts),
                        (double) (w_steps + w_acc) / (double) std::max<uint64_t>(1, w_steps));
        }
        if (rows) {
            std::vector<int> seqs((size_t) n), next((size_t) n);
            std::vector<int32_t> toks((size_t) n);
            const std::vector<strata::kernels::SamplerParams> sps((size_t) n, g0);
            for (int s = 0; s < n; ++s) seqs[(size_t) s] = s;
            while ((int) out[0].size() < G) {
                for (int s = 0; s < n; ++s) toks[(size_t) s] = out[(size_t) s].back();
                if (!model.multi_rows(seqs.data(), toks.data(), sps.data(), n, next.data(), err)) { std::fprintf(stderr, "multi bench: %s\n", err.c_str()); return 1; }
                for (int s = 0; s < n; ++s) out[(size_t) s].push_back(next[(size_t) s]);
                ++boundaries;   // (one before every step, in the step itself)
            }
        }
        for (int s = 0; s < n && !rows && !srows && !wrows; ++s) {
            if (!model.multi_issue(s, out[(size_t) s].back(), g0, err)) { std::fprintf(stderr, "multi bench: %s\n", err.c_str()); return 1; }
            q.push_back(s);
        }
        if (srows) {
            std::deque<int> qg;
            const auto issue_rows = [&](const std::vector<int>& seqs) -> bool {
                // (STRATA_GLM_EVEN_GROUPS: the groups the server makes, from the sequences still decoding)
                int active = 0;
                for (int s = 0; s < n; ++s) active += (int) out[(size_t) s].size() < G;
                size_t first = 0;
                for (const int k : strata::core::split_groups((int) seqs.size(), active, split_rows, even_groups)) {
                    std::vector<int> group_seqs;
                    std::vector<int32_t> toks;
                    std::vector<strata::kernels::SamplerParams> sps((size_t) k, g0);
                    for (int j = 0; j < k; ++j) {
                        const int s = seqs[first + (size_t) j];
                        group_seqs.push_back(s);
                        toks.push_back((int32_t) out[(size_t) s].back());
                    }
                    if (!model.multi_issue_rows(group_seqs.data(), toks.data(), sps.data(), k, err)) {
                        std::fprintf(stderr, "multi bench: %s\n", err.c_str());
                        return false;
                    }
                    for (const int s : group_seqs) q.push_back(s);
                    qg.push_back(k);
                    first += (size_t) k;
                }
                return true;
            };
            std::vector<int> initial((size_t) n);
            for (int s = 0; s < n; ++s) initial[(size_t) s] = s;
            if (!issue_rows(initial)) return 1;
            while (!qg.empty()) {
                std::vector<int> again;
                const auto take_group = [&](int k) -> bool {
                    for (int j = 0; j < k; ++j) {
                        const int s = q.front();
                        q.pop_front();
                        const int tok = model.multi_take(s, err);
                        if (tok < 0) {
                            std::fprintf(stderr, "multi bench: %s\n", err.c_str());
                            return false;
                        }
                        out[(size_t) s].push_back(tok);
                        if ((int) out[(size_t) s].size() < G) again.push_back(s);
                    }
                    return true;
                };
                const int k = qg.front();
                qg.pop_front();
                if (!take_group(k)) return 1;
                since += k;
                if (K > 0 && since >= K * n) {
                    while (!qg.empty()) {
                        const int k2 = qg.front();
                        qg.pop_front();
                        if (!take_group(k2)) return 1;
                    }
                    if (!model.multi_boundary(err)) {
                        std::fprintf(stderr, "multi bench: %s\n", err.c_str());
                        return 1;
                    }
                    ++boundaries;
                    since = 0;
                }
                if (!issue_rows(again)) return 1;
            }
        }
        while (!q.empty()) {
            const int s = q.front();
            q.pop_front();
            const int tok = model.multi_take(s, err);
            if (tok < 0) { std::fprintf(stderr, "multi bench: %s\n", err.c_str()); return 1; }
            out[(size_t) s].push_back(tok);
            std::vector<int> again;
            if ((int) out[(size_t) s].size() < G) again.push_back(s);
            if (K > 0 && ++since >= K * n) {   // drain: every token in flight taken, the tiers' boundary, then on
                since = 0;
                ++boundaries;
                while (!q.empty()) {
                    const int s2 = q.front();
                    q.pop_front();
                    const int t2 = model.multi_take(s2, err);
                    if (t2 < 0) { std::fprintf(stderr, "multi bench: %s\n", err.c_str()); return 1; }
                    out[(size_t) s2].push_back(t2);
                    if ((int) out[(size_t) s2].size() < G) again.push_back(s2);
                }
                if (!model.multi_boundary(err)) { std::fprintf(stderr, "multi bench: %s\n", err.c_str()); return 1; }
            }
            for (const int s2 : again) {
                if (!model.multi_issue(s2, out[(size_t) s2].back(), g0, err)) { std::fprintf(stderr, "multi bench: %s\n", err.c_str()); return 1; }
                q.push_back(s2);
            }
        }
        const double multi_s = secs(t0, now());
        st_b = model.fast_stats();
        tiers("solo", solo_a, solo_b, G);
        tiers(path, st_a, st_b, n * G);
        const auto agree = [](const std::vector<int>& a, const std::vector<int>& b) {
            size_t k = 0;
            while (k < a.size() && k < b.size() && a[k] == b[k]) ++k;
            return k;
        };
        const size_t same = agree(solo, out[0]);
        std::printf("  solo vs its untimed warm-up run: %zu of %zu agree\n", agree(solo, solo2), solo.size());
        for (int s = 1; s < n; ++s)
            std::printf("  seq %d vs seq 0: %zu agree, vs solo: %zu\n", s, agree(out[(size_t) s], out[0]),
                        agree(out[(size_t) s], solo));
        std::printf("MULTI BENCH %d seqs, prompt %d, %d tokens each: solo %.1f tok/s | %s %.1f tok/s total, "
                    "%.1f per sequence (%d boundaries) | sequence 0 same tokens: %zu of %zu\n", n, L, G,
                    (G - 1) / solo_s, path, (double) n * (G - 1) / multi_s,
                    (G - 1) / multi_s, boundaries, same, solo.size());
        for (int s = 0; s < n; ++s) {
            std::printf("  seq %d:", s);
            for (int i = 0; i < 12 && i < (int) out[(size_t) s].size(); ++i) std::printf(" %d", out[(size_t) s][(size_t) i]);
            std::printf("\n");
        }
        std::printf("  solo :");
        for (int i = 0; i < 12 && i < (int) solo.size(); ++i) std::printf(" %d", solo[(size_t) i]);
        std::printf("\n");
        // STRATA_GLM_MULTI_DUMP=<file>: every sequence's tokens, a line each (then the solo run's) - two builds' or
        // settings' runs compared whole, not by their first 12
        if (const char* dump = getenv("STRATA_GLM_MULTI_DUMP")) {
            if (std::FILE* f = std::fopen(dump, "w")) {
                for (int s = 0; s <= n; ++s) {
                    const std::vector<int>& v = s < n ? out[(size_t) s] : solo;
                    if (s < n) std::fprintf(f, "seq %d:", s);
                    else std::fputs("solo:", f);
                    for (const int t : v) std::fprintf(f, " %d", t);
                    std::fputc('\n', f);
                }
                std::fclose(f);
            }
        }
        std::fflush(stdout);
        return 0;
    }

    if (!o.serve) {
        if (o.tokens.empty()) {
            std::fprintf(stderr, "maya generate: --tokens is required (this build has no tokenizer; see the "
                                 "header of src/program/generate.cpp)\n");
            return 2;
        }
        // STRATA_GLM_SCORE=<c>: score --tokens instead of generating.  Every token goes through the one-token decode
        // path (the tiers and the CPU lane as in a decode); the tokens after the first c are scored given the ones
        // before: mean NLL, perplexity, top-1 hits.  STRATA_GLM_SCORE_DUMP=<file> writes each scored position's
        // log-softmax (n_vocab floats); STRATA_GLM_SCORE_REF=<file> reads such a dump and reports KL(ref || this) and
        // how often both pick the same top token - the quality yardstick for engine changes and for quants.
        // STRATA_GLM_SCORE_TOPK=<file> is the compact reference (the FP8 model's, tools/maya_quant): an int32 k, then
        // per scored position k int32 token ids (best first) and their k float log-probs.  KL over those k tokens plus
        // one bucket for the rest of the vocabulary.
        if (const char* sc = getenv("STRATA_GLM_SCORE")) {
            const size_t c = (size_t) std::max(0L, std::atol(sc));
            const char* dump = getenv("STRATA_GLM_SCORE_DUMP");
            const char* ref = getenv("STRATA_GLM_SCORE_REF");
            const char* tkf = getenv("STRATA_GLM_SCORE_TOPK");
            std::FILE* df = dump && dump[0] ? std::fopen(dump, "wb") : nullptr;
            std::FILE* rf = ref && ref[0] ? std::fopen(ref, "rb") : nullptr;
            std::FILE* tf = tkf && tkf[0] ? std::fopen(tkf, "rb") : nullptr;
            int32_t tk = 0;
            if ((dump && dump[0] && !df) || (ref && ref[0] && !rf) || (tkf && tkf[0] && !tf) ||
                (tf && (std::fread(&tk, sizeof tk, 1, tf) != 1 || tk <= 0 || tk > n_vocab))) {
                std::fprintf(stderr, "maya generate: cannot open the score dump/reference\n");
                return 1;
            }
            std::vector<int32_t> tid((size_t) std::max(tk, 1));
            std::vector<float> tlp((size_t) std::max(tk, 1));
            model.set_host_logits(true);
            model.reset();
            std::vector<float> lg, ls((size_t) n_vocab), rl((size_t) n_vocab);
            double nll = 0.0, kl = 0.0, kl_max = 0.0;
            size_t n = 0, top1 = 0, same = 0, nref = 0;
            const auto ts = std::chrono::steady_clock::now();
            // STRATA_GLM_SCORE_PREFILL=1: the unscored context (all but its last token) through the prompt path, so a
            // long context is scored in minutes (the KV-cache formats at 128K); the scored tokens stay token by token
            size_t i0 = 0;
            if (getenv("STRATA_GLM_SCORE_PREFILL") != nullptr && model.fast() && model.prefill_chunk() > 0 && c > 2 &&
                c < o.tokens.size()) {
                const std::vector<int32_t> part(o.tokens.begin(), o.tokens.begin() + (long) (c - 1));
                if (!model.prefill(part, err, (int32_t) o.tokens[c - 1])) {
                    std::printf("ERR %s\n", err.empty() ? "prefill" : err.c_str());
                    return 1;
                }
                i0 = c - 1;
            }
            for (size_t i = i0; i + 1 < o.tokens.size(); ++i) {
                if (!model.forward({(int32_t) o.tokens[i]}, lg, err)) {
                    std::printf("ERR %s\n", err.c_str());
                    return 1;
                }
                if (i + 1 <= c) continue;
                float mx = lg[0];
                size_t am = 0;
                for (size_t v = 1; v < (size_t) n_vocab; ++v)
                    if (lg[v] > mx) {
                        mx = lg[v];
                        am = v;
                    }
                double se = 0.0;
                for (size_t v = 0; v < (size_t) n_vocab; ++v) se += std::exp((double) lg[v] - mx);
                const double lse = mx + std::log(se);
                for (size_t v = 0; v < (size_t) n_vocab; ++v) ls[v] = (float) (lg[v] - lse);
                const size_t nx = (size_t) o.tokens[i + 1];
                nll -= ls[nx];
                top1 += am == nx;
                ++n;
                if (df) std::fwrite(ls.data(), sizeof(float), ls.size(), df);
                if (rf && std::fread(rl.data(), sizeof(float), rl.size(), rf) == rl.size()) {
                    double k = 0.0;
                    size_t ra = 0;
                    for (size_t v = 0; v < (size_t) n_vocab; ++v) {
                        const double p = std::exp((double) rl[v]);
                        if (p > 0.0) k += p * ((double) rl[v] - (double) ls[v]);
                        if (rl[v] > rl[ra]) ra = v;
                    }
                    kl += k;
                    kl_max = std::max(kl_max, k);
                    same += ra == am;
                    ++nref;
                }
                if (tf && std::fread(tid.data(), sizeof(int32_t), (size_t) tk, tf) == (size_t) tk &&
                    std::fread(tlp.data(), sizeof(float), (size_t) tk, tf) == (size_t) tk) {
                    double k = 0.0, pr = 0.0, pt = 0.0;
                    for (int32_t j = 0; j < tk; ++j) {
                        const double p = std::exp((double) tlp[j]);
                        k += p * ((double) tlp[j] - (double) ls[(size_t) tid[j]]);
                        pr += p;
                        pt += std::exp((double) ls[(size_t) tid[j]]);
                    }
                    const double rr = std::max(1.0 - pr, 1e-12), rt = std::max(1.0 - pt, 1e-12);
                    if (pr < 1.0) k += rr * (std::log(rr) - std::log(rt));
                    kl += k;
                    kl_max = std::max(kl_max, k);
                    same += (size_t) tid[0] == am;
                    ++nref;
                }
            }
            if (df) std::fclose(df);
            if (rf) std::fclose(rf);
            if (tf) std::fclose(tf);
            const double sec = std::chrono::duration<double>(std::chrono::steady_clock::now() - ts).count();
            std::printf("SCORE %zu tokens: nll %.5f ppl %.4f top1 %.2f%%", n, nll / (double) std::max<size_t>(1, n),
                        std::exp(nll / (double) std::max<size_t>(1, n)), 100.0 * (double) top1 / (double) std::max<size_t>(1, n));
            if (nref > 0)
                std::printf(" | vs ref: KL %.6f (max %.4f), same top token %.2f%%", kl / (double) nref, kl_max,
                            100.0 * (double) same / (double) nref);
            const auto stc = model.fast_stats();
            std::printf(" | %.1f ms/token | cpu lane %llu experts, vram hit %.2f%%\n",
                        1000.0 * sec / (double) std::max<size_t>(1, o.tokens.size() - 1),
                        (unsigned long long) stc.cpu_experts,
                        100.0 * (double) stc.hits / (double) std::max<uint64_t>(1, stc.hits + stc.misses));
            return 0;
        }
        const int rc = run_request(o.tokens, o.max_new, sp);
        return rc;
    }

    // STRATA_GLM_WARM_PROMPT=<n> (default 0 = off; 32 or more): one synthetic prompt through the whole part chain
    // before the server is told READY, so the session's first request does not pay what only a session's first
    // prompt pays (issue #8: it read 95-133 ms/token against 23-44 warm - the first chunk carries the one-time work:
    // every kernel's first launch, the prompt path's own allocations, the tiers' first reads).  It costs about one
    // chunk of the smallest part's size (~15 s on a four-way split, beside the tiers' own 90-151 s warm-up), its ids
    // are spread over the vocabulary so the routes are not one expert's, and the state is zeroed right after it: no
    // conversation can see it (a request with no slot in use resets anyway, before its own prompt).
    if (const char* wv = getenv("STRATA_GLM_WARM_PROMPT")) {
        const int wn = std::atoi(wv);
        if (wn > 0 && model.prefill_chunk() > 0) {
            const int wtn = std::max(32, wn);
            std::vector<int32_t> wids((size_t) wtn);
            for (int i = 0; i < wtn; ++i) wids[(size_t) i] = (int32_t) (100 + (i * 131) % 20000);
            std::fprintf(stderr, "glm fast: warming the prompt path with a %d-token prompt before READY "
                                 "(STRATA_GLM_WARM_PROMPT)\n", wtn);
            const auto wt0 = std::chrono::steady_clock::now();
            std::string werr;
            const bool wok = model.prefill(wids, werr);
            const double wms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - wt0).count();
            model.reset();
            if (wok) {
                std::fprintf(stderr, "glm fast: the prompt path is warm (%d tokens in %.0f ms, %.1f ms/token)\n", wtn,
                             wms, wms / (double) wtn);
            } else {
                std::fprintf(stderr, "glm fast: the prompt warm-up did not run%s%s - the first prompt pays it\n",
                             werr.empty() ? "" : ": ", werr.c_str());
            }
        }
    }
    // issue #40: a prompt that stopped moving (the GPU idle, a host thread spinning on a flag) waited forever - the
    // server's own stall check counts a spinning thread as work.  A prompt layer or a token is the heartbeat; a big
    // chunk's layer on a slow PC takes seconds, so three minutes of none is a hang.
    start_serve_watchdog(180);
    // the VRAM left on every GPU once everything is loaded.  Below ~256 MiB (a split's later GPUs: 512 - their verify
    // windows' graphs are captured on first use) a later allocation fails, and under Windows' WDDM the driver pages GPU
    // memory to system RAM instead: decode falls to a few tok/s and a request can stall for good.  Said, never changed:
    // the line names the setting that leaves room (--vram-reserve-mib is deducted before the expert pool is sized).
    const auto pools = model.fast() ? model.pool_vram() : std::vector<strata::core::Glm5Model::PoolVram>();
    const auto free_mib_on = [](int dev) -> int64_t {
        int cur = 0;
        cudaGetDevice(&cur);
        cudaSetDevice(dev);
        size_t fb = 0, tb = 0;
        const bool ok = cudaMemGetInfo(&fb, &tb) == cudaSuccess;
        if (!ok) cudaGetLastError();
        cudaSetDevice(cur);
        return ok ? (int64_t) (fb >> 20) : -1;
    };
    for (size_t i = 0; i < pools.size(); ++i) {
        const auto& pv = pools[i];
        const int64_t fm = free_mib_on(pv.dev);
        if (fm < 0) continue;
        const int64_t ok_mib = i == 0 ? 256 : 512;
        if (fm >= ok_mib) {
            std::fprintf(stderr, "maya serve: CUDA%d: %lld MiB of VRAM free with everything loaded\n", pv.dev, (long long) fm);
        } else if (pv.adapted) {
            std::fprintf(stderr, "maya serve: WARNING: CUDA%d: %lld MiB of VRAM free with everything loaded - this card "
                                 "only just fits the model (the VRAM reserve was lowered to %d MiB so the expert pool "
                                 "fits): requests may stall. Close other programs that use the GPU, or lower "
                                 "--max-context\n", pv.dev, (long long) fm, pv.reserve_mib);
        } else {
            std::fprintf(stderr, "maya serve: CUDA%d: %lld MiB of VRAM free with everything loaded - LOW: requests may "
                                 "stall; add %s %lld to the config's args (or lower --max-context)\n", pv.dev,
                         (long long) fm, i == 0 ? "--vram-reserve-mib" : "--vram-reserve-later-mib",
                         (long long) (pv.reserve_mib + (i == 0 ? 512 : 768) - fm));
            // a number of the user's own (--expert-cache N) is not checked against the VRAM once written (auto is)
            if (pv.fixed > 0)
                std::fprintf(stderr, "maya serve: the expert pool was set with --expert-cache %lld: a smaller one - or "
                                     "--expert-cache auto, which checks the free VRAM once the pool is written - leaves "
                                     "room\n", (long long) pv.fixed);
        }
    }
    // the serve wire contract (serve/server.py): READY <max_context> [stop] once, then GEN <max_new>
    // [<key>=<value> ...] <id,id,...> lines - the sampling keys are per request and the ids are the LAST
    // token.  "stop" is NOT advertised: a request runs to its end (no mid-request cancel in M3.1).
    // "pause": PAUSE keeps the written state; "multi=<n>": n conversations at once (GEN rid=<id>, see gen_admit)
    if (multi) std::printf("READY %lld stop pause multi=%d\n", (long long) o.max_context, model.seq_count());
    else std::printf("READY %lld stop pause\n", (long long) o.max_context);
    std::fprintf(stderr, "maya generate: glm5-next serve (M3.1): GEN <max_new> [keys] <ids>, GENI <max_new> [keys] "
                         "<embeddings> <ids>; QUIT to end\n");
    // GENI: the embeddings file is one or more maya-vision records (int32 'SVE1', n, nx, ny, n_embd, then n x n_embd
    // floats) in prompt order; their rows stand in for the prompt's <|image|> tokens, in order (GLM-5.3's image
    // tokens take no positions of their own: the rows only replace the embeddings)
    constexpr int64_t kGlmImageToken = 154854;   // <|image|>
    std::vector<int32_t> lp_ctx;   // LOGP: the context the saved state belongs to (the state before its last token)
    // #477 --expert-profile-save: what the expert tiers learned, every --expert-profile-save-every minutes before the
    // next request (a prompt's lent slots are back by then) and after the last one
    Clock::time_point profile_saved_at = Clock::now();
    const auto save_profile = [&](const char* why) {
        if (o.expert_profile_save.empty() || !model.fast()) return;
        std::string e;
        if (model.save_expert_profile(o.expert_profile_save, e))
            std::fprintf(stderr, "maya serve: expert profile saved to %s (%s)\n", o.expert_profile_save.c_str(), why);
        else
            std::fprintf(stderr, "maya serve: the expert profile was not saved: %s\n", e.c_str());
        profile_saved_at = Clock::now();
    };
    std::string line;
    for (;;) {
        if (multi && !pending.empty()) admit_pending();
        if (multi && !gens.empty()) {   // requests running: a line when one is there, else one scheduler step
            if (!try_line(line)) {
                struct StepScope {
                    StepScope() { strata::core::progress().busy.store(true); }
                    ~StepScope() { strata::core::progress().busy.store(false); }
                } step_scope;
                multi_step();
                continue;
            }
        } else if (!next_line(line)) {
            break;
        }
        std::istringstream ss(line);
        std::string cmd;
        ss >> cmd;
        if (cmd == "QUIT") break;
        if (multi && !gens.empty() && (cmd == "VLEND" || cmd == "VRECLAIM" || cmd == "LOGP")) {
            std::printf("ERR %s needs the engine idle (%zu requests running)\n", cmd.c_str(), gens.size());
            continue;
        }
        if (cmd == "STOP") continue;   // a stray one between requests: ignore
        if (o.expert_profile_save_min > 0 &&
            Clock::now() - profile_saved_at >= std::chrono::duration<double>(o.expert_profile_save_min * 60.0))
            save_profile("periodic");
        // the watchdog watches a command from here until this iteration ends, whichever way it ends
        struct BusyScope {
            BusyScope() { strata::core::progress().busy.store(true); strata::core::progress_at("the request"); }
            ~BusyScope() { strata::core::progress().busy.store(false); strata::core::progress_at("idle"); }
        } busy_scope;
        // LOGP <n_ctx> <id,id,...>: the summed log-probability of the tokens from n_ctx on given those before
        // ("LP <sum> <n> <1 if every one was the most likely>") - the multiple-choice scoring of the zero-shot suite
        // (tools/maya_quant/zs_tasks.py).  A context identical to the previous request's is restored, not read again
        // (the choices of one question share it).
        // VLEND / VRECLAIM (between requests): the on-demand vision encoder is about to start on the first GPU - the
        // expert pool's tail is emptied and freed for it ("VLENT <bytes>") - and has ended ("VRECLAIMED"; an ERR while
        // its memory is not free yet: try again)
        if (cmd == "VLEND") {
            size_t lent = 0;
            if (model.vision_lend(lent, err)) std::printf("VLENT %zu\n", lent);
            else std::printf("ERR %s\n", err.c_str());
            std::fflush(stdout);
            continue;
        }
        if (cmd == "VRECLAIM") {
            if (model.vision_reclaim(err)) std::printf("VRECLAIMED\n");
            else std::printf("ERR %s\n", err.c_str());
            std::fflush(stdout);
            continue;
        }
        if (cmd == "LOGP") {
            int64_t n_ctx = 0;
            std::string ids_s;
            ss >> n_ctx >> ids_s;
            std::vector<int64_t> t64;
            std::string e;
            if (n_ctx < 2 || ids_s.empty() || !parse_i64_list(ids_s.c_str(), t64, e) || (int64_t) t64.size() <= n_ctx) {
                std::printf("ERR malformed LOGP line\n");
                std::fflush(stdout);
                continue;
            }
            std::vector<int32_t> t(t64.begin(), t64.end());
            snap_tokens.clear();                  // the saved state is LOGP's now, not a conversation's
            ckpt_drop_past(0);                    // ... and so are the caches the checkpoints stood on
            model.set_host_logits(true);
            const std::vector<int32_t> ctx(t.begin(), t.begin() + n_ctx);
            bool ok = true;
            if (!(model.fast() && ctx == lp_ctx && model.snapshot_restore())) {
                model.reset();
                std::vector<int32_t> head(t.begin(), t.begin() + n_ctx - 1);
                ok = model.prefill(head, err, t[(size_t) n_ctx - 1]);
                if (!ok && err.empty()) {   // no prompt path: the token path, 8 at a time
                    ok = true;
                    std::vector<float> lg0;
                    for (size_t i = 0; ok && i < head.size(); i += 8) {
                        const std::vector<int32_t> part(head.begin() + (long) i,
                                                        head.begin() + (long) std::min(head.size(), i + 8));
                        ok = model.forward(part, lg0, err);
                    }
                }
                lp_ctx.clear();
                if (ok && model.fast() && model.snapshot_save()) lp_ctx = ctx;
            }
            std::vector<float> lgv;
            double lp = 0.0;
            bool greedy = true;
            for (int64_t j = n_ctx; ok && j < (int64_t) t.size(); ++j) {
                ok = model.forward({t[(size_t) j - 1]}, lgv, err);
                if (!ok) break;
                float mx = lgv[0];
                size_t am = 0;
                for (size_t v = 1; v < lgv.size(); ++v)
                    if (lgv[v] > mx) { mx = lgv[v]; am = v; }
                double se = 0.0;
                for (const float x : lgv) se += std::exp((double) x - mx);
                lp += (double) lgv[(size_t) t[(size_t) j]] - mx - std::log(se);
                greedy = greedy && am == (size_t) t[(size_t) j];
            }
            model.set_host_logits(false);
            if (!ok) {
                lp_ctx.clear();
                std::printf("ERR %s\n", err.c_str());
            } else {
                std::printf("LP %.6f %lld %d\n", lp, (long long) (t.size() - (size_t) n_ctx), greedy ? 1 : 0);
            }
            std::fflush(stdout);
            continue;
        }
        if (cmd == "GEN" || cmd == "GENI") {
            int64_t max_new = 0;
            ss >> max_new;
            strata::kernels::SamplerParams rq = sp;
            bool any_key = false, greedy_req = false;
            req_think_budget = 0;
            req_think_end = -1;
            double req_pcie = -1.0;   // setup's calibration: this request's CPU lane (as the engine started: -1 / 0)
            int req_threads = 0;
            int req_chunks = 0;   // ... and its disk reads' pieces (0: as the engine started)
            double req_msw = -1.0;   // ... and its light-miss bar / first layer (STRATA_GLM_MISS_SKIP_W / _FROM; < 0: the engine's)
            int req_msf = -1;
            int req_rid = -1;
            req_spec = 0;
            req_spec_min_p = -1.0;
            req_spec_n_min = -1;
            req_spec_cum = -1.0;
            req_bias.clear();
            std::string bias_err;
            int pen_n = 0;
            float pen_repeat = 1.0f, pen_freq = 0.0f, pen_present = 0.0f;
            std::string tok2, ids, emb_path;
            while (ss >> tok2) {
                const size_t eq = tok2.find('=');
                if (eq == std::string::npos) {
                    if (cmd == "GENI" && emb_path.empty()) emb_path = tok2;   // GENI: the file, then the ids
                    else ids = tok2;   // the ids list is the last token on the line
                    continue;
                }
                const std::string k = tok2.substr(0, eq), v = tok2.substr(eq + 1);
                if (k == "temperature") {   // temperature=0 (or below): greedy, whatever else the line says
                    rq.temperature = (float) std::atof(v.c_str());
                    if (rq.temperature <= 0.0f) greedy_req = true;
                    else any_key = true;
                }
                else if (k == "top_p") { rq.top_p = (float) std::atof(v.c_str()); any_key = true; }
                else if (k == "top_k") { rq.top_k = std::atoi(v.c_str()); any_key = true; }
                else if (k == "min_p") { rq.min_p = (float) std::atof(v.c_str()); any_key = true; }
                else if (k == "seed") { rq.seed = (uint64_t) std::atoll(v.c_str()); }
                else if (k == "think_budget") { req_think_budget = std::atoll(v.c_str()); }
                else if (k == "think_end") { req_think_end = std::atoi(v.c_str()); }
                else if (k == "pcie_frac") { req_pcie = std::atof(v.c_str()); }
                else if (k == "cpu_threads") { req_threads = std::atoi(v.c_str()); }
                else if (k == "read_chunks") { req_chunks = std::atoi(v.c_str()); }
                else if (k == "miss_skip_w") { req_msw = std::atof(v.c_str()); }
                else if (k == "miss_skip_from") { req_msf = std::atoi(v.c_str()); }
                else if (k == "rid") { req_rid = std::atoi(v.c_str()); }
                else if (k == "spec") { req_spec = std::atoi(v.c_str()); }
                else if (k == "spec_min_p") { req_spec_min_p = std::atof(v.c_str()); }
                else if (k == "spec_n_min") { req_spec_n_min = std::atoi(v.c_str()); }
                else if (k == "spec_cum") { req_spec_cum = std::atof(v.c_str()); }
                else if (k == "logit_bias") {   // <id>:<bias>,... - -100 or below bans the token
                    for (size_t at = 0; at < v.size() && bias_err.empty();) {
                        size_t end = v.find(',', at);
                        if (end == std::string::npos) end = v.size();
                        const std::string e1 = v.substr(at, end - at);
                        const size_t colon = e1.find(':');
                        char* p1 = nullptr;
                        char* p2 = nullptr;
                        const long id = colon == std::string::npos ? -1 : std::strtol(e1.c_str(), &p1, 10);
                        const float b = colon == std::string::npos ? 0.0f : std::strtof(e1.c_str() + colon + 1, &p2);
                        if (id < 0 || id >= n_vocab || p1 != e1.c_str() + colon || p2 == nullptr || *p2 != '\0' ||
                            !std::isfinite(b))
                            bias_err = "malformed logit_bias entry '" + e1 + "'";
                        else
                            req_bias.emplace_back((int32_t) id, b <= -100.0f ? -INFINITY : b);
                        at = end + 1;
                    }
                }
                // the penalties (#273), llama.cpp's: over the last penalty_last_n tokens the request's state was fed
                // (its prompt, then what it wrote).  They do not choose between greedy and sampled
                else if (k == "penalty_last_n") { pen_n = std::atoi(v.c_str()); }
                else if (k == "penalty_repeat") { pen_repeat = std::strtof(v.c_str(), nullptr); }
                else if (k == "penalty_freq") { pen_freq = std::strtof(v.c_str(), nullptr); }
                else if (k == "penalty_present") { pen_present = std::strtof(v.c_str(), nullptr); }
            }
            if (any_key) rq.greedy = false;   // any sampler key switches the request to the sampled path
            if (greedy_req) rq.greedy = true;
            // a repeat penalty of 0 or below (some clients' "off"; the logits would be divided by it) or a value that is
            // not a number counts as neutral
            if (!std::isfinite(pen_repeat) || pen_repeat <= 0.0f) pen_repeat = 1.0f;
            if (!std::isfinite(pen_freq)) pen_freq = 0.0f;
            if (!std::isfinite(pen_present)) pen_present = 0.0f;
            if (pen_n < 0) pen_n = INT_MAX;   // -1, llama.cpp's "the whole context": every token the window keeps (4096)
            // as llama.cpp: no window, or every penalty neutral, is no penalty - the request's picks are exactly as
            // without the keys (no history kept, nothing uploaded)
            const bool pen_on = pen_n > 0 && (pen_repeat != 1.0f || pen_freq != 0.0f || pen_present != 0.0f);
            if (pen_on) {
                rq.penalty_last_n = pen_n;
                rq.penalty_repeat = pen_repeat;
                rq.penalty_freq = pen_freq;
                rq.penalty_present = pen_present;
            }
            if (!bias_err.empty()) {
                if (multi && req_rid >= 0) std::printf("ERR %d %s\n", req_rid, bias_err.c_str());
                else std::printf("ERR %s\n", bias_err.c_str());
                std::fflush(stdout);
                continue;
            }
            if (!req_bias.empty() && (rq.greedy || rq.temperature <= 0.0f)) {
                // greedy with a bias: the sampled path's top_k = 1 is the same argmax (ties to the lower id) over the
                // biased logits - the greedy shortcuts (the argmax at the end of a forward, the windows' argmax) read
                // the unbiased ones.  (Greedy with penalties stays greedy: every shortcut leaves a request with
                // penalties to the sampler, whose greedy kernel counts them - the same pick, ~75 us less a token)
                rq.greedy = false;
                rq.temperature = 1.0f;
                rq.top_k = 1;
                rq.top_p = 1.0f;
                rq.min_p = 0.0f;
            }
            // the CPU lane for this request: what it names, else what the engine started with
            model.set_pcie_share(req_pcie >= 0.0 && req_pcie <= 1.0 ? req_pcie : -1.0);
            model.set_cpu_lane_threads(req_threads > 0 ? req_threads : 0);
            model.set_read_chunks(req_chunks > 0 ? req_chunks : 0);
            model.set_miss_skip(req_msw, req_msf);
            std::vector<int64_t> toks;
            std::string e;
            if (max_new <= 0 || ids.empty() || !parse_i64_list(ids.c_str(), toks, e)) {
                if (multi && req_rid >= 0) std::printf("ERR %d malformed GEN line\n", req_rid);
                else std::printf("ERR malformed GEN line\n");
                continue;
            }
            lp_ctx.clear();   // a generation saves its own state over LOGP's
            cur_img_pos.clear();
            cur_img_rows.clear();
            if (cmd == "GENI") {
                const int64_t E = model.geometry().n_embd;
                std::FILE* f = std::fopen(emb_path.c_str(), "rb");
                std::string ge = f ? "" : "cannot open " + emb_path;
                while (f && ge.empty()) {
                    int32_t hdr[5];
                    // the end of the records only between two of them: a header cut short is a broken file, and a
                    // record cannot hold more rows than the prompt has tokens (not a multi-GB allocation off one)
                    const size_t got = std::fread(hdr, 1, sizeof hdr, f);
                    if (got == 0 && std::feof(f)) break;
                    if (got != sizeof hdr || hdr[0] != 0x31455653 || hdr[1] <= 0 || hdr[4] != E ||
                        (size_t) hdr[1] > toks.size() - cur_img_rows.size() / (size_t) E) {
                        ge = "not a maya-vision record for this model";
                        break;
                    }
                    const size_t at = cur_img_rows.size();
                    cur_img_rows.resize(at + (size_t) hdr[1] * (size_t) E);
                    if (std::fread(cur_img_rows.data() + at, sizeof(float) * (size_t) E, (size_t) hdr[1], f) !=
                        (size_t) hdr[1])
                        ge = "short embeddings file";
                }
                if (f) std::fclose(f);
                for (size_t i = 0; ge.empty() && i < toks.size(); ++i)
                    if (toks[i] == kGlmImageToken) cur_img_pos.push_back((int64_t) i);
                if (ge.empty() && (int64_t) cur_img_pos.size() * E != (int64_t) cur_img_rows.size())
                    ge = std::to_string(cur_img_pos.size()) + " image tokens, " +
                         std::to_string(cur_img_rows.size() / (size_t) E) + " image rows";
                if (!ge.empty()) {
                    cur_img_pos.clear();
                    cur_img_rows.clear();
                    if (multi && req_rid >= 0) std::printf("ERR %d %s\n", req_rid, ge.c_str());
                    else std::printf("ERR %s\n", ge.c_str());
                    std::fflush(stdout);
                    continue;
                }
            }
            if (multi) {
                if (req_rid < 0) { std::printf("ERR this engine runs several conversations: GEN needs rid=<id>\n"); std::fflush(stdout); }
                else gen_admit(req_rid, toks, max_new, rq);
                continue;
            }
            {
                std::string be;
                rq.logit_bias = model.logit_bias_table(0, req_bias, be);
                if (!be.empty()) {
                    std::printf("ERR %s\n", be.c_str());
                    std::fflush(stdout);
                    continue;
                }
            }
            model.set_image_rows(cur_img_pos, cur_img_rows);
            run_request(toks, max_new, rq);
            model.set_image_rows({}, {});
            continue;
        }
        if (cmd.empty()) continue;
        std::printf("ERR unsupported command '%s' (GEN, GENI, LOGP, VLEND, VRECLAIM, STOP and QUIT)\n", cmd.c_str());
    }
    // the engine ends (QUIT, or the server gone): the conversation in the model is kept for the next start
    if (multi) {   // the requests still running end here (their DONE lines), then every context's conversation
        for (const Pending& w : pending) std::printf("DONE %d 0 %zu 0.0 0.0 cancel 0 0 0 0 0\n", w.rid, w.ids.size());
        pending.clear();
        multi_drain();
        for (auto& gp : gens)
            if (!gp->done) {
                gp->finish = "cancel";
                gen_finish(*gp);
            }
        gen_sweep();
        if (slot_keep && !slot_ram)
            for (int c = 0; c < (int) ctxs.size(); ++c) {
                bind_ctx(c);
                slot_put(" on shutdown");
            }
        return 0;
    }
    if (slot_keep && !slot_ram) slot_put(" on shutdown");
    save_profile("quit");   // #477
    strata::core::Glm5Model::set_routing_dump("", err);   // --dump-routing: the trace closed whole
    return 0;
}

}  // namespace


}  // namespace

int main(int argc, char** argv) {
    strata::platform::env_aliases();   // (MAYA_* settings under their older names too)
    std::setvbuf(stdout, nullptr, _IONBF, 0);   // unbuffered: the interesting output is the output before a crash
    if (std::getenv("CUDA_MODULE_LOADING") == nullptr) set_env("CUDA_MODULE_LOADING", "EAGER");
    Options o;
    bool have_tokens = false;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&](const char* what) -> const char* {
            if (i + 1 >= argc) { std::fprintf(stderr, "%s needs a value\n", what); std::exit(2); }
            return argv[++i];
        };
        if (a == "--help" || a == "-h") { usage(); return 0; }
        else if (a == "--glm-pack") o.glm_pack = next("--glm-pack");
        else if (a == "--glm-ram-resident") set_env("STRATA_GLM_RAM_RESIDENT", "1");
        else if (a == "--glm-ram-slack") set_env("STRATA_GLM_RAM_SLACK", next("--glm-ram-slack"));
        else if (a == "--parallel") {   // the sequence states (STRATA_GLM_SEQS): n conversations at once
            const int n = std::atoi(next("--parallel"));
            if (n < 1 || n > 8) { std::fprintf(stderr, "--parallel takes 1..8\n"); return 2; }
            set_env("STRATA_GLM_SEQS", std::to_string(n).c_str());
        }
        else if (a == "--tokens") { std::string e;
            if (!parse_i64_list(next("--tokens"), o.tokens, e)) { std::fprintf(stderr, "%s\n", e.c_str()); return 2; }
            have_tokens = true; }
        else if (a == "--tokens-file") {
            if (have_tokens) { std::fprintf(stderr, "supply one token input only\n"); return 2; }
            std::ifstream input(next("--tokens-file"));
            if (!input) { std::fprintf(stderr, "cannot open token file\n"); return 2; }
            std::string text((std::istreambuf_iterator<char>(input)), std::istreambuf_iterator<char>());
            if (input.bad()) { std::fprintf(stderr, "cannot read token file\n"); return 2; }
            std::string error;
            if (text.find('\0') != std::string::npos || !parse_i64_list(text.c_str(), o.tokens, error)) {
                std::fprintf(stderr, "malformed token file: %s\n", error.c_str()); return 2; }
            have_tokens = true; }
        else if (a == "--max-new") o.max_new = std::atoll(next("--max-new"));
        else if (a == "--max-context") o.max_context = std::atoll(next("--max-context"));
        else if (a == "--greedy") o.greedy = true;
        else if (a == "--seed") { o.seed = (uint64_t) std::atoll(next("--seed")); o.greedy = false; }
        else if (a == "--top-k") o.top_k = std::atoi(next("--top-k"));
        else if (a == "--top-p") o.top_p = (float) std::atof(next("--top-p"));
        else if (a == "--temperature") o.temperature = (float) std::atof(next("--temperature"));
        else if (a == "--dump-logits") o.dump_logits = next("--dump-logits");
        else if (a == "--dump-routing") o.dump_routing = next("--dump-routing");
        else if (a == "--kv") o.kv = next("--kv");
        else if (a == "--kv-resident") o.kv_resident = std::atoll(next("--kv-resident"));
        else if (a == "--kv-streaming") o.kv_streaming = next("--kv-streaming");
        else if (a == "--expert-cache") {
            const std::string v = next("--expert-cache");
            o.expert_cache = (v == "auto") ? -1 : std::atoi(v.c_str());
        }
        else if (a == "--vram-reserve-mib") { o.vram_reserve_mib = std::atoi(next("--vram-reserve-mib")); o.vram_reserve_given = true; }
        else if (a == "--vram-reserve-later-mib") o.vram_reserve_later_mib = std::atoi(next("--vram-reserve-later-mib"));
        else if (a == "--prefill") {
            const std::string v = next("--prefill");
            o.prefill_arg = v;
            o.prefill_auto = v == "auto";
            o.prefill_chunk = o.prefill_auto ? 8192 : std::atoll(v.c_str());
        }
        else if (a == "--expert-profile") o.expert_profile = next("--expert-profile");
        else if (a == "--expert-profile-save") o.expert_profile_save = next("--expert-profile-save");
        else if (a == "--expert-profile-save-every") o.expert_profile_save_min = std::atof(next("--expert-profile-save-every"));
        else if (a == "--spec") o.spec = std::atoi(next("--spec"));
        else if (a == "--spec-follow") o.spec_follow = next("--spec-follow");
        else if (a == "--spec-follow-save") o.spec_follow_save = next("--spec-follow-save");
        else if (a == "--spec-min-p") { o.spec_min_p = std::atof(next("--spec-min-p")); o.spec_min_p_set = true; }
        else if (a == "--stop-eos") { o.stop_eos = true; o.eos_given = true; }
        else if (a == "--eos-ids") { std::string e;
            if (!parse_i64_list(next("--eos-ids"), o.eos_ids, e)) { std::fprintf(stderr, "--eos-ids: %s\n", e.c_str()); return 2; }
            o.stop_eos = true; o.eos_given = true; }
        else if (a == "--layer-split") o.layer_split = next("--layer-split");
        else if (a == "--vision") {         /* accepted for the server's config symmetry: the glm path takes
                                             * pictures through GENI unconditionally (the flag gates the qwen
                                             * path's M-RoPE table, which the glm engine does not use) */ }
        else if (a == "--dflash") o.dflash = next("--dflash");
        else if (a == "--serve") o.serve = true;
        else if (a == "--mtp") o.mtp = next("--mtp");
        else if (a == "--mtp-window") o.mtp_window = std::atoll(next("--mtp-window"));
        else {
            std::fprintf(stderr, "unknown argument: %s\n", a.c_str());
            usage();
            return 2;
        }
    }
    // --kv-streaming on|off: on streams with --kv-resident's window (32768 when it gives none), off keeps every cell in
    if (o.kv_streaming == "on") { if (o.kv_resident <= 0) o.kv_resident = 32768; }
    else if (o.kv_streaming == "off") { o.kv_resident = 0; }
    else if (!o.kv_streaming.empty()) { std::fprintf(stderr, "strata-glm: --kv-streaming takes on or off (not %s)\n", o.kv_streaming.c_str()); return 2; }
    if (o.glm_pack.empty()) { usage(); return 2; }
    // --kv int8 on the GLM path: the INT8 latent cache (Glm5Model::lat_q8_, read at load)
    if (o.kv == "int8" && std::getenv("STRATA_GLM_KV_INT8") == nullptr) set_env("STRATA_GLM_KV_INT8", "1");
    // --kv-resident N: KV streaming of the DSA latent cache (Glm5Model::kv_resident_cells, read at load)
    if (o.kv_resident < 0) { std::fprintf(stderr, "strata-glm: --kv-resident must be 0 or more\n"); return 2; }
    if (o.kv_resident > 0 && std::getenv("STRATA_GLM_KV_RESIDENT") == nullptr)
        set_env("STRATA_GLM_KV_RESIDENT", std::to_string(o.kv_resident).c_str());
    if (o.kv_streaming == "off") set_env("STRATA_GLM_KV_RESIDENT", "0");
    // --parallel on the GLM path: STRATA_GLM_SEQS, for the SLOTS... (goes into the engine's multi path)
    if (o.expert_cache != 0)
        set_env("STRATA_GLM_EXPERT_CACHE", o.expert_cache < 0 ? "auto" : std::to_string(o.expert_cache).c_str());
    if (o.vram_reserve_given) set_env("STRATA_GLM_RESERVE_MB", std::to_string(o.vram_reserve_mib).c_str());
    if (o.vram_reserve_later_mib >= 0)
        set_env("STRATA_GLM_RESERVE_LATER_MB", std::to_string(o.vram_reserve_later_mib).c_str());
    return glm_pack_generate(o);
}
