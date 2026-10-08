// strata-metal: Strata's engine line protocol (serve/server.py <-> engine) over llama.cpp's Metal backend.
//
// macOS / Apple Silicon only.  The server starts it as `strata-metal --serve <args>` and talks to it over stdin/stdout
// exactly as it talks to the CUDA engine (src/program/generate.cpp): INFO/READY at start, then GEN, GENI, STOP, SAVE,
// RESTORE, VRAM and QUIT.  A protocol adapter, not a port of the CUDA engine: that engine's kernels and expert
// cache are CUDA code, and llama.cpp already runs this model on Metal (docs/MACOS.md).
//
// The model is the original GGUF (no Strata pack): llama.cpp's qwen4exp architecture, every tensor in unified memory.
//
// Conversation reuse.  The gated-delta-net layers keep a recurrent state that cannot be cut back to any position, so a
// request whose ids leave the held ones goes back to a checkpoint: the recurrent part of the state (the attention KV
// is cut with seq_rm), taken while reading a prompt just before each <|im_start|> - the message boundaries a chat
// template's re-render leaves intact.  Never a state that does not match: the held ids are compared by their content
// (a token id, or for an image's <|image_pad|> cells the picture's hash and the row), no checkpoint -> the whole prompt.
//
// Images (GENI): strata-vision's records, placed at the prompt's <|image_pad|> runs with the CUDA engine's M-RoPE
// positions (t = p, h = p + y, w = p + x; the text after an image goes on at p + max(nx, ny)).
//
// Speculative decoding (--mtp <MTP-only GGUF>): the checkpoint's own MTP head drafts --spec tokens, the trunk checks
// them in one pass (llama.cpp's draft-mtp, #29761); the output is the trunk's, drafts only decide how many tokens a
// pass yields.  metal/mtp_gguf.py makes the file from the checkpoint.
//
// Batch slots (--batch N, the server's "parallel"): BGEN admissions into sequences 1..N, decoded together a token a
// window between commands (BT / BDONE); BSTOP ends one.  Not here: the expert cache (VRAM), BYIELD (never gives way).
#include "common.h"
#include "llama.h"
#include "sampling.h"
#include "speculative.h"

#include <unistd.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#ifndef STRATA_VERSION
#define STRATA_VERSION "0"
#endif
#ifndef STRATA_LLAMA_COMMIT
#define STRATA_LLAMA_COMMIT "unknown"
#endif

namespace {

using Clock = std::chrono::steady_clock;
double ms_since(Clock::time_point t) { return std::chrono::duration<double, std::milli>(Clock::now() - t).count(); }
const bool g_phases = std::getenv("STRATA_PHASES") != nullptr;   // per-phase timings of the MTP loop, to stderr

// ---------------------------------------------------------------------------------------------- stdout / stdin
// The protocol owns the real stdout.  llama.cpp's common library prints plain LOG() lines to stdout, so main() moves fd 1
// onto stderr and the protocol writes to its own copy of the original stdout: no library line can reach the server.
FILE* g_proto = stdout;

void out(const char* fmt, ...) __attribute__((format(printf, 1, 2)));
void out(const char* fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    std::vfprintf(g_proto, fmt, ap);
    va_end(ap);
    std::fflush(g_proto);
}

// STOP has to reach a running request, so stdin is read on its own thread.  A GEN line clears the stop flag before
// it is queued: a STOP the server sent for an earlier request (after that one's DONE) cannot cancel the next.
std::atomic<bool> g_stop{false};
std::unique_ptr<std::atomic<bool>[]> g_bstop;   // BSTOP <slot>: that slot ends at its next window
int g_slots = 0;
std::mutex g_mu;
std::condition_variable g_cv;
std::deque<std::string> g_lines;

void read_stdin() {
    std::string line;
    int c;
    for (;;) {
        line.clear();
        while ((c = std::fgetc(stdin)) != EOF && c != '\n') line.push_back((char) c);
        if (!line.empty() && line.back() == '\r') line.pop_back();
        if (c == EOF && line.empty()) line = "QUIT";      // the server is gone
        if (line == "STOP") { g_stop = true; continue; }
        if (line.rfind("BSTOP ", 0) == 0) {
            const int b = std::atoi(line.c_str() + 6);
            if (b >= 0 && b < g_slots) g_bstop[(size_t) b] = true;
            continue;
        }
        // BYIELD <slot>: a long prompt read may give way to a shorter waiting one (#656).  This engine reads a prompt
        // to its end; the server treats that as "it did not give way".
        if (line.rfind("BYIELD", 0) == 0) continue;
        if (line.rfind("GEN", 0) == 0 || line.rfind("BGEN", 0) == 0) g_stop = false;
        { std::lock_guard<std::mutex> lk(g_mu); g_lines.push_back(line); }
        g_cv.notify_one();
        if (c == EOF) return;
    }
}

// The next command; with `wait` false only one that is there already (batch slots decode in the meantime).
bool next_line(std::string& s, bool wait = true) {
    std::unique_lock<std::mutex> lk(g_mu);
    if (wait) g_cv.wait(lk, [] { return !g_lines.empty(); });
    if (g_lines.empty()) return false;
    s = std::move(g_lines.front());
    g_lines.pop_front();
    return true;
}

// ---------------------------------------------------------------------------------------------- options
llama_token g_image_pad = 248056;   // qwen4exp.ple.image_token_id; --image-pad-id for another model (the tests)
struct Options {
    std::string gguf, mtp;
    std::string eagle3;                                // --eagle3 <gguf>: an EAGLE3 draft model instead of --mtp
    int64_t max_context = 32768;
    std::string kv = "int8";
    std::vector<llama_token> eos = {248044, 248046};   // <|endoftext|>, <|im_end|>: the CUDA engine's --eos-ids
    int n_batch = 2048;
    int checkpoints = 8;
    int threads = 0;
    int spec = 3;                                      // drafted tokens per pass with --mtp
    int batch = 0;                                     // --batch N: batch slots (the server's "parallel")
    int poll = -1;                                     // --poll N: ggml's CPU busy-wait level (llama.cpp default 50)
};

bool parse_ids(const char* s, std::vector<llama_token>& v) {
    v.clear();
    while (*s) {
        char* e = nullptr;
        long x = std::strtol(s, &e, 10);
        if (e == s || x < 0) return false;
        v.push_back((llama_token) x);
        if (*e && *e != ',') return false;
        s = *e == ',' ? e + 1 : e;
    }
    return !v.empty();
}

bool parse_args(int argc, char** argv, Options& o) {
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto val = [&](const char* name) -> const char* {
            if (i + 1 >= argc) { std::fprintf(stderr, "strata-metal: %s needs a value\n", name); std::exit(2); }
            return argv[++i];
        };
        if (a == "--serve") continue;
        else if (a == "--gguf") o.gguf = val("--gguf");
        else if (a == "--mtp") o.mtp = val("--mtp");
        else if (a == "--eagle3") o.eagle3 = val("--eagle3");
        else if (a == "--spec") o.spec = std::atoi(val("--spec"));
        else if (a == "--batch" || a == "--slots") o.batch = std::max(0, std::min(16, std::atoi(val("--batch"))));
        else if (a == "--max-context") o.max_context = std::atoll(val("--max-context"));
        else if (a == "--kv") o.kv = val("--kv");
        else if (a == "--prefill-batch") o.n_batch = std::atoi(val("--prefill-batch"));
        else if (a == "--checkpoints") o.checkpoints = std::max(0, std::atoi(val("--checkpoints")));
        else if (a == "--threads") o.threads = std::atoi(val("--threads"));
        else if (a == "--poll") o.poll = std::max(0, std::min(100, std::atoi(val("--poll"))));
        else if (a == "--image-pad-id") g_image_pad = (llama_token) std::atoi(val("--image-pad-id"));
        else if (a == "--eos-ids") {
            if (!parse_ids(val("--eos-ids"), o.eos)) { std::fprintf(stderr, "strata-metal: bad --eos-ids\n"); return false; }
        } else if (a.rfind("--", 0) == 0) {
            // a flag of the CUDA engine (a shared config): said once, not fatal.  Its value, if any, is skipped too.
            std::fprintf(stderr, "strata-metal: %s is not used by the Metal engine (ignored)\n", a.c_str());
            if (i + 1 < argc && std::strncmp(argv[i + 1], "--", 2) != 0) ++i;
        }
    }
    if (o.gguf.empty()) { std::fprintf(stderr, "strata-metal: --gguf <the model's first GGUF shard> is required\n"); return false; }
    if (o.max_context < 512) { std::fprintf(stderr, "strata-metal: --max-context must be 512 or more\n"); return false; }
    if (!o.mtp.empty() && !o.eagle3.empty()) { std::fprintf(stderr, "strata-metal: --mtp or --eagle3, not both\n"); return false; }
    if (o.spec < 1 || o.spec > 16) { std::fprintf(stderr, "strata-metal: --spec takes 1 to 16 drafted tokens\n"); return false; }
    if (o.kv != "int8" && o.kv != "q4_0" && o.kv != "k8v4" && o.kv != "f16") {
        std::fprintf(stderr, "strata-metal: --kv takes int8, q4_0, k8v4 or f16\n");
        return false;
    }
    return true;
}

// ---------------------------------------------------------------------------------------------- images
constexpr uint64_t kImageBit = 1ull << 63;

uint64_t fnv1a(const void* p, size_t n, uint64_t h = 1469598103934665603ull) {
    const auto* b = static_cast<const unsigned char*>(p);
    for (size_t i = 0; i < n; ++i) h = (h ^ b[i]) * 1099511628211ull;
    return h;
}

// What one request's prompt is: the ids, each cell's content key (an id, or an image row), its M-RoPE position
// (t, h, w; text has t = h = w) and its embedding row (images only), and the position the output goes on at.
struct Prompt {
    std::vector<llama_token> ids;
    std::vector<uint64_t> keys;
    std::vector<std::array<llama_pos, 3>> pos;
    std::vector<const float*> rows;
    std::vector<float> data;
    llama_pos next = 0;
};

// The CUDA engine's GENI placement (src/program/generate.cpp): "" or why the request is refused.  `mrope`: 2-D image
// positions (qwen4exp); a model without M-RoPE gives the rows ordinary consecutive positions, as llama.cpp's mtmd does.
std::string place_images(Prompt& P, const std::string& path, int n_embd, bool mrope) {
    struct Img { int64_t n, nx, ny; size_t off; uint64_t hash; };
    std::vector<Img> imgs;
    if (!path.empty()) {
        std::FILE* f = std::fopen(path.c_str(), "rb");
        if (!f) return "cannot open " + path;
        std::string err;
        for (;;) {
            int32_t hdr[5];
            const size_t got = std::fread(hdr, sizeof(int32_t), 5, f);
            if (got == 0) break;
            if (got != 5 || hdr[0] != 0x31455653 || hdr[1] < 1 || hdr[2] < 1 || hdr[3] < 1 ||
                (int64_t) hdr[2] * hdr[3] != hdr[1] || hdr[4] != n_embd) {
                err = "bad embeddings file (expected strata-vision records of width " + std::to_string(n_embd) + ")";
                break;
            }
            const size_t off = P.data.size(), cnt = (size_t) hdr[1] * (size_t) hdr[4];
            P.data.resize(off + cnt);
            if (std::fread(P.data.data() + off, sizeof(float), cnt, f) != cnt) { err = "short embeddings file"; break; }
            const int64_t grid[3] = {hdr[1], hdr[2], hdr[3]};
            imgs.push_back({hdr[1], hdr[2], hdr[3], off,
                            fnv1a(P.data.data() + off, cnt * sizeof(float), fnv1a(grid, sizeof grid))});
        }
        std::fclose(f);
        if (!err.empty()) return err;
    }
    const size_t n = P.ids.size();
    P.keys.resize(n);
    P.pos.resize(n);
    P.rows.assign(n, nullptr);
    llama_pos p = 0;
    size_t i = 0, k = 0;
    while (i < n) {
        if (P.ids[i] != g_image_pad || imgs.empty()) {
            P.keys[i] = (uint64_t) (uint32_t) P.ids[i];
            P.pos[i] = {p, p, p};
            ++p;
            ++i;
            continue;
        }
        if (k >= imgs.size()) return "the prompt has more images than the embeddings file";
        const Img& im = imgs[k++];
        for (int64_t j = 0; j < im.n; ++j)
            if (i + (size_t) j >= n || P.ids[i + (size_t) j] != g_image_pad)
                return "image " + std::to_string(k) + " has " + std::to_string((long long) im.n) +
                       " rows but fewer <|image_pad|> tokens";
        for (int64_t j = 0; j < im.n; ++j) {
            const llama_pos y = (llama_pos) (j / im.nx), x = (llama_pos) (j % im.nx);
            P.keys[i + (size_t) j] = kImageBit | ((im.hash + (uint64_t) j * 0x9E3779B97F4A7C15ull) >> 1);
            const llama_pos q = mrope ? p : p + (llama_pos) j;
            P.pos[i + (size_t) j] = mrope ? std::array<llama_pos, 3>{p, p + y, p + x} : std::array<llama_pos, 3>{q, q, q};
            P.rows[i + (size_t) j] = P.data.data() + im.off + (size_t) j * (size_t) n_embd;
        }
        i += (size_t) im.n;
        p += mrope ? (llama_pos) std::max(im.nx, im.ny) : (llama_pos) im.n;
    }
    if (k != imgs.size()) return "the embeddings file has more images than the prompt";
    if (!imgs.empty() && P.ids.back() == g_image_pad) return "the prompt cannot end in an image";
    P.next = p;
    return "";
}

// ---------------------------------------------------------------------------------------------- the engine
struct Checkpoint {
    size_t n;                                 // cells the state covers: live[0, n)
    llama_pos pos;                            // the position of cell n (the first one cut)
    std::vector<uint8_t> tgt, dft, spec;      // recurrent part of the trunk, the draft context's, the drafter's
};

// What one sequence of the context holds: seq 0 is the solo path's (GEN, with the MTP drafts), seq b + 1 batch slot b's.
struct Seq {
    std::vector<llama_token> live;            // the cells it holds
    std::vector<uint64_t> keys;               // ... and their content keys
    std::deque<Checkpoint> ckpts;             // ascending n
};

// A batch slot decoding between requests (after its BGEN admission): one token a window, its own sampler.
struct Slot {
    bool active = false;
    common_sampler* smpl = nullptr;
    llama_token pending = 0;                  // produced and sent, not yet fed
    llama_pos pos = 0;                        // its position
    int64_t produced = 0, max_new = 0;
    Clock::time_point t0;
};

struct Engine {
    Options o;
    common_params params;
    common_init_result_ptr init;
    common_speculative_init_result_ptr spec_init;
    common_speculative* spec = nullptr;
    llama_model* model = nullptr;
    llama_context* ctx = nullptr;
    llama_context* ctx_dft = nullptr;
    const llama_vocab* vocab = nullptr;
    llama_token im_start = -1;
    int n_embd = 0;
    std::vector<Seq> seqs;
    std::vector<Slot> slots;

    llama_memory_t mem() const { return llama_get_memory(ctx); }

    bool load(std::string& err) {
        common_params& p = params;
        p.model.path = o.gguf;
        p.n_parallel = 1 + o.batch;               // seq 0 and one per batch slot, each with the whole context
        p.kv_unified = false;                     // own cells per sequence: a copied sequence (copy_seq) then rolled back
                                                  // to a checkpoint read wrong in the unified cache (measured)
        p.n_ctx = (int32_t) (o.max_context * p.n_parallel);
        p.n_batch = o.n_batch;
        p.n_ubatch = std::min(o.n_batch, 2048);   // 2048: prompts +9% on an M5 Max (llama-bench pp4096, 268 -> 293 tok/s)
        p.n_gpu_layers = 999;                     // everything in unified memory
        p.fit_params = false;
        p.warmup = false;
        p.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_AUTO;
        p.cache_type_k = o.kv == "f16" ? GGML_TYPE_F16 : o.kv == "q4_0" ? GGML_TYPE_Q4_0 : GGML_TYPE_Q8_0;
        p.cache_type_v = o.kv == "f16" ? GGML_TYPE_F16 : o.kv == "int8" ? GGML_TYPE_Q8_0 : GGML_TYPE_Q4_0;
        if (o.threads > 0) p.cpuparams.n_threads = p.cpuparams_batch.n_threads = o.threads;
        if (o.poll >= 0) p.cpuparams.poll = p.cpuparams_batch.poll = (uint32_t) o.poll;
        postprocess_cpu_params(p.cpuparams, nullptr);       // what llama.cpp's argument parser does (thread counts)
        postprocess_cpu_params(p.cpuparams_batch, &p.cpuparams);
        postprocess_cpu_params(p.speculative.draft.cpuparams, &p.cpuparams);
        postprocess_cpu_params(p.speculative.draft.cpuparams_batch, &p.cpuparams_batch);
        const std::string& draft_path = o.mtp.empty() ? o.eagle3 : o.mtp;
        if (!draft_path.empty()) {
            // the draft: the model's MTP head, or an EAGLE3 model (it reads 3 of the target's layers; llama.cpp's
            // common_speculative drives both through the same begin / process / draft / accept calls)
            p.speculative.types = {o.mtp.empty() ? COMMON_SPECULATIVE_TYPE_DRAFT_EAGLE3 : COMMON_SPECULATIVE_TYPE_DRAFT_MTP};
            p.speculative.draft.mparams.path = draft_path;
            p.speculative.draft.n_max = o.spec;
            p.speculative.draft.n_gpu_layers = 999;
            p.speculative.draft.cache_type_k = p.cache_type_k;
            p.speculative.draft.cache_type_v = p.cache_type_v;
            const auto lim = common_speculative_get_output_limits(p.n_batch, p.n_parallel, common_speculative_n_max(&p.speculative));
            p.n_outputs_max = lim.total;
            p.n_outputs_max_per_seq = lim.per_seq;
        }
        init = common_init_from_params(p);
        model = init ? init->model() : nullptr;
        ctx = init ? init->context() : nullptr;
        if (!model) { err = "could not load " + o.gguf; return false; }
        if (!ctx) { err = "could not create the context (too long a --max-context for this Mac's memory?)"; return false; }
        vocab = llama_model_get_vocab(model);
        n_embd = llama_model_n_embd(model);
        if (!draft_path.empty()) {
            common_params pd = common_base_params_to_speculative(p);
            spec_init = common_speculative_init_from_params(pd, model, ctx);
            ctx_dft = spec_init ? spec_init->context() : nullptr;
            if (!ctx_dft) { err = "could not load the draft model " + draft_path; return false; }
            p.speculative.draft.ctx_tgt = ctx;
            p.speculative.draft.ctx_dft = ctx_dft;
            spec = common_speculative_init(p.speculative, 1);   // the solo path's sequence (0) drafts
            if (!spec) { err = "could not start speculative decoding with " + draft_path; return false; }
        }
        seqs.resize((size_t) p.n_parallel);
        slots.resize((size_t) o.batch);
        llama_token t[4];
        // a message's first token, where the checkpoints go: Qwen's <|im_start|>, else GPT-OSS's (harmony) <|start|>
        for (const char* s : {"<|im_start|>", "<|start|>"})
            if (im_start < 0 && llama_tokenize(vocab, s, (int32_t) std::strlen(s), t, 4, false, true) == 1) im_start = t[0];
        return true;
    }

    std::vector<uint8_t> state(llama_context* c, llama_seq_id sq) {
        std::vector<uint8_t> d(llama_state_seq_get_size_ext(c, sq, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY));
        if (!d.empty() && llama_state_seq_get_data_ext(c, d.data(), d.size(), sq, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY) == 0)
            d.clear();
        return d;
    }

    void checkpoint(llama_seq_id sq, size_t n, llama_pos pos) {
        if (o.checkpoints == 0) return;
        Checkpoint c{n, pos, state(ctx, sq), {}, {}};
        if (c.tgt.empty()) return;
        if (sq == 0 && ctx_dft) c.dft = state(ctx_dft, 0);
        if (sq == 0 && spec) common_speculative_get_state(spec, 0, c.spec);
        auto& ck = seqs[(size_t) sq].ckpts;
        while (!ck.empty() && ck.back().n >= n) ck.pop_back();
        ck.push_back(std::move(c));
        while ((int) ck.size() > o.checkpoints) ck.pop_front();
    }

    void clear(llama_seq_id sq) {
        llama_memory_seq_rm(mem(), sq, -1, -1);
        if (sq == 0 && ctx_dft) llama_memory_seq_rm(llama_get_memory(ctx_dft), 0, -1, -1);
        seqs[(size_t) sq] = Seq{};
    }

    // Back to at most p held cells: the latest checkpoint at or below p, else nothing held.
    void rollback(llama_seq_id sq, size_t p) {
        Seq& S = seqs[(size_t) sq];
        while (!S.ckpts.empty() && S.ckpts.back().n > p) S.ckpts.pop_back();
        if (!S.ckpts.empty()) {
            const Checkpoint& c = S.ckpts.back();
            bool ok = llama_state_seq_set_data_ext(ctx, c.tgt.data(), c.tgt.size(), sq, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY) &&
                      llama_memory_seq_rm(mem(), sq, c.pos, -1);
            if (ok && sq == 0 && ctx_dft) {
                ok = (c.dft.empty() || llama_state_seq_set_data_ext(ctx_dft, c.dft.data(), c.dft.size(), 0,
                                                                    LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY)) &&
                     llama_memory_seq_rm(llama_get_memory(ctx_dft), 0, c.pos, -1);
                if (ok && !c.spec.empty()) common_speculative_set_state(spec, 0, c.spec);
            }
            // EAGLE3's draft context lags one cell (its last pair waits for the next token), and llama.cpp keeps that
            // boundary in a checkpoint only for a recurrent target: for another (GPT-OSS) the restored draft would
            // jump a position.  One cell less of both, read again, gives EAGLE3 the boundary from fresh features.
            size_t n = c.n;
            if (ok && sq == 0 && !o.eagle3.empty() && c.spec.empty() && n > 0) {
                ok = llama_memory_seq_rm(mem(), sq, c.pos - 1, -1) &&
                     llama_memory_seq_rm(llama_get_memory(ctx_dft), 0, c.pos - 1, -1);
                --n;
            }
            if (ok) {
                S.live.resize(n);
                S.keys.resize(n);
                return;
            }
            std::fprintf(stderr, "strata-metal: checkpoint at %zu could not be restored; reading the prompt again\n", c.n);
        }
        clear(sq);
    }

    // Another sequence's whole state into `dst` (the solo request that a second request turned into a batch slot, and
    // back): the prompt it already read is not read again.  false: dst is left empty.
    bool copy_seq(llama_seq_id src, llama_seq_id dst) {
        clear(dst);
        std::vector<uint8_t> buf(llama_state_seq_get_size(ctx, src));
        if (buf.empty() || llama_state_seq_get_data(ctx, buf.data(), buf.size(), src) == 0 ||
            llama_state_seq_set_data(ctx, buf.data(), buf.size(), dst) == 0) {
            clear(dst);
            return false;
        }
        seqs[(size_t) dst] = seqs[(size_t) src];
        if (dst == 0) {                           // the drafter knows none of it: it starts from the new tokens
            for (auto& c : seqs[0].ckpts) { c.dft.clear(); c.spec.clear(); }
            if (ctx_dft) llama_memory_seq_rm(llama_get_memory(ctx_dft), 0, -1, -1);
        }
        return true;
    }

    // Cells [a, b) of P into sequence sq (and the drafter), the last with logits when `logits`.  false: err says why.
    // A batch holds one kind of entry (token ids or embedding rows), so text and a picture go in as separate batches.
    // `drafter`: the MTP drafter follows too (seq 0 only).  Not for a prompt with pictures: llama.cpp's MTP batch pairs
    // each cell with the next one's token, and a text cell before a picture's rows makes a batch it refuses ("entries
    // with both a token id and an embedding cannot be mixed").  The drafter then drafts without that prompt: weaker
    // drafts, the same answer.
    bool feed(llama_seq_id sq, const Prompt& P, size_t a, size_t b, bool logits, std::string& err, bool drafter = true) {
        Seq& S = seqs[(size_t) sq];
        for (size_t s = a; s < b;) {
            const bool image = P.rows[s] != nullptr;
            size_t e = s + 1;
            while (e < b && (P.rows[e] != nullptr) == image) ++e;
            common_batch batch(ctx);
            for (size_t i = s; i < e; ++i) {
                const bool lg = logits && i + 1 == b;
                if (image) {
                    const llama_pos p4[4] = {P.pos[i][0], P.pos[i][1], P.pos[i][2], 0};
                    batch.add_embd({P.rows[i], 1, (size_t) n_embd}, p4, sq, lg);
                } else {
                    batch.add(P.ids[i], P.pos[i][0], sq, lg);
                }
            }
            const int rc = llama_process(ctx, LLAMA_PROCESS_TYPE_DECODE, batch.get());
            if (rc != 0) { err = "llama_process failed (" + std::to_string(rc) + ")"; return false; }
            if (spec && sq == 0 && drafter && !common_speculative_process(spec, batch)) {
                err = "the draft model could not follow the batch";
                return false;
            }
            S.live.insert(S.live.end(), P.ids.begin() + (long) s, P.ids.begin() + (long) e);
            S.keys.insert(S.keys.end(), P.keys.begin() + (long) s, P.keys.begin() + (long) e);
            s = e;
        }
        return true;
    }

    common_sampler* sampler(const std::string& line_keys) {
        common_params_sampling s = params.sampling;
        s.temp = 0.0f;                        // absent keys: greedy, as the CUDA engine
        s.top_k = 20;
        s.top_p = 1.0f;
        s.min_p = 0.0f;
        s.penalty_last_n = 64;
        size_t i = 0;
        while (i < line_keys.size()) {
            size_t j = line_keys.find(' ', i);
            if (j == std::string::npos) j = line_keys.size();
            std::string kv = line_keys.substr(i, j - i);
            i = j + 1;
            size_t eq = kv.find('=');
            if (eq == std::string::npos) continue;
            std::string k = kv.substr(0, eq);
            const char* v = kv.c_str() + eq + 1;
            if (k == "temperature") s.temp = std::strtof(v, nullptr);
            else if (k == "top_p") s.top_p = std::strtof(v, nullptr);
            else if (k == "top_k") s.top_k = std::atoi(v);
            else if (k == "min_p") s.min_p = std::strtof(v, nullptr);
            else if (k == "penalty_repeat") s.penalty_repeat = std::strtof(v, nullptr);
            else if (k == "penalty_freq") s.penalty_freq = std::strtof(v, nullptr);
            else if (k == "penalty_present") s.penalty_present = std::strtof(v, nullptr);
            else if (k == "penalty_last_n") s.penalty_last_n = std::atoi(v);
            else if (k == "seed") s.seed = (uint32_t) std::strtoull(v, nullptr, 10);
        }
        if (s.top_k <= 0) s.top_k = 64;
        return common_sampler_init(model, s);
    }

    bool decoding() const {
        return std::any_of(slots.begin(), slots.end(), [](const Slot& x) { return x.active; });
    }

    //   GEN <max_new> [key=value ...] <id,id,...>   |   GENI <max_new> [key=value ...] <embeddings file> <id,id,...>
    // slot >= 0: BGEN <slot> ... / BGENI <slot> ... (the line passed here without "B" and the slot): the prompt and its
    // first token as a GEN 1, then BADM <slot> <1: it goes on in the slot's windows | 0: done>.
    void gen(const std::string& line, bool geni, int slot = -1, int64_t slot_max_new = 0) {
        const llama_seq_id sq = slot + 1;
        Seq& S = seqs[(size_t) sq];
        const size_t sp = line.rfind(' ');
        const size_t head = geni ? 5 : 4;
        Prompt P;
        char* endp = nullptr;
        const long long max_new = std::strtoll(line.c_str() + head, &endp, 10);
        auto refuse = [&](const std::string& why) {
            out("ERR %s\n", why.c_str());
        };
        if (sp == std::string::npos || sp < head || max_new < 0 || !parse_ids(line.c_str() + sp + 1, P.ids))
            return refuse("expected: GEN <max_new> <id,id,...> or GENI <max_new> <file> <id,id,...>");
        std::string keys_txt(static_cast<const char*>(endp), line.c_str() + sp), file;
        if (geni) {                                   // the embeddings file: the first word without an '='
            std::string rest;
            size_t i = 0;
            while (i < keys_txt.size()) {
                size_t j = keys_txt.find(' ', i);
                if (j == std::string::npos) j = keys_txt.size();
                const std::string w = keys_txt.substr(i, j - i);
                i = j + 1;
                if (w.empty()) continue;
                if (file.empty() && w.find('=') == std::string::npos) file = w;
                else rest += " " + w;
            }
            keys_txt = rest;
            if (file.empty()) return refuse("expected: GENI <max_new> <file> <id,id,...>");
        }
        const size_t n = P.ids.size();
        if ((int64_t) n >= o.max_context)
            return refuse("the prompt (" + std::to_string(n) + " tokens) does not fit the context (" +
                          std::to_string(o.max_context) + ")");
        const llama_token n_vocab = llama_vocab_n_tokens(vocab);
        for (llama_token t : P.ids)
            if (t >= n_vocab) return refuse("token id " + std::to_string(t) + " is outside the vocabulary");
        const auto rope = llama_model_rope_type(model);
        if (const std::string e = place_images(P, file, n_embd, rope == LLAMA_ROPE_TYPE_MROPE || rope == LLAMA_ROPE_TYPE_IMROPE);
            !e.empty())
            return refuse(e);

        // reuse: the held cells that start the prompt - this sequence's, or an idle one's that holds more of it (copied
        // over); the last prompt token is always read again (the first pass)
        auto prefix = [&](const Seq& x) {
            size_t q = 0;
            while (q < x.keys.size() && q < n && x.keys[q] == P.keys[q]) ++q;
            return q;
        };
        size_t p = prefix(S);
        for (size_t other = 0; other < seqs.size(); ++other) {
            const bool busy = other > 0 && slots[other - 1].active;
            if ((llama_seq_id) other == sq || busy) continue;
            const size_t q = prefix(seqs[other]);
            if (q > p + 16 && copy_seq((llama_seq_id) other, sq)) p = q;
        }
        p = std::min(p, n - 1);
        if (p < S.live.size()) rollback(sq, p);
        const size_t resume = S.live.size();
        out("RESUME %zu\n", resume);

        std::string err;
        const auto t0 = Clock::now();
        const char* finish = "length";
        bool cancelled = false;
        const bool drafts = spec && sq == 0;
        // the prompt but its last token, in chunks of n_batch, each also ending at a message boundary (a checkpoint)
        size_t at = resume;
        while (at + 1 < n) {
            size_t end = std::min(n - 1, at + (size_t) o.n_batch);
            size_t b = at + 1;
            while (b < end && P.ids[b] != im_start) ++b;
            const bool boundary = b < end;
            if (boundary) end = b;
            if (!feed(sq, P, at, end, false, err, !geni)) { clear(sq); return refuse(err); }
            at = end;
            if (boundary) checkpoint(sq, at, P.pos[at][0]);
            const double ms = ms_since(t0);
            out("PP %zu %zu %.0f %.1f\n", at, n, ms, (at - resume) / std::max(1e-3, ms / 1000.0));
            if (g_stop) { cancelled = true; finish = "cancel"; break; }
        }

        // decode: each pass reads the pending token (the prompt's last, then the last one produced) and the drafts
        const int64_t limit = slot >= 0 ? 1 : max_new;      // an admission produces the first token only
        int64_t produced = 0, drafted = 0, accepted = 0;
        double prompt_ms = ms_since(t0);
        const auto t1 = Clock::now();
        double ph[4] = {0, 0, 0, 0};                         // STRATA_PHASES: draft, target, catch-up, sample
        int ph_n = 0;
        common_sampler* smpl = nullptr;
        llama_token pending = P.ids[n - 1];
        llama_pos pending_pos = P.pos[n - 1][0];
        uint64_t pending_key = P.keys[n - 1];
        if (!cancelled) {
            smpl = sampler(keys_txt);
            if (drafts) common_speculative_begin(spec, 0, std::vector<llama_token>(P.ids.begin(), P.ids.end() - 1));
            llama_pos next = P.next;                         // where the output goes on
            std::vector<llama_token> draft;
            bool first = true, done = false;
            while (!done) {
                draft.clear();
                const int64_t room = std::min<int64_t>(limit - produced - 1, o.max_context - (int64_t) S.live.size() - 2);
                if (drafts && room > 0) {
                    common_speculative_get_draft_params(spec, 0) = {
                        /* .drafting = */ true, /* .n_max = */ (int32_t) std::min<int64_t>(room, o.spec),
                        /* .pos0 = */ pending_pos, /* .id_last = */ pending, /* .prompt = */ &S.live, /* .result = */ &draft};
                    const auto tp = Clock::now();
                    common_speculative_draft(spec);
                    if (g_phases) ph[0] += ms_since(tp);
                    // drafting wrote the drafts into the draft context: out again, the verify pass feeds it the
                    // trunk's own (speculative-simple does the same)
                    llama_memory_seq_rm(llama_get_memory(ctx_dft), 0, pending_pos, -1);
                }
                common_batch batch(ctx);
                batch.add(pending, pending_pos, sq, true);
                for (size_t i = 0; i < draft.size(); ++i) batch.add(draft[i], next + (llama_pos) i, sq, true);
                const auto tv = Clock::now();
                const bool fail_tgt = llama_process(ctx, LLAMA_PROCESS_TYPE_DECODE, batch.get()) != 0;
                if (g_phases && !fail_tgt) { llama_synchronize(ctx); ph[1] += ms_since(tv); }
                const auto tc = Clock::now();
                if (fail_tgt || (drafts && !common_speculative_process(spec, batch))) {
                    common_sampler_free(smpl);
                    clear(sq);
                    return refuse("the decode pass failed");
                }
                if (g_phases && ctx_dft) { llama_synchronize(ctx_dft); ph[2] += ms_since(tc); }
                if (first) { prompt_ms = ms_since(t0); first = false; }
                const auto ts = Clock::now();
                const std::vector<llama_token> ids = common_sampler_sample_and_accept_n(smpl, ctx, draft);
                if (g_phases) { ph[3] += ms_since(ts); ++ph_n; }
                drafted += (int64_t) draft.size();
                accepted += (int64_t) ids.size() - 1;
                if (drafts) common_speculative_accept(spec, 0, (uint16_t) (ids.size() - 1));
                // the pending token and the accepted drafts are in the memory now; ids.back() is the next pending one
                S.live.push_back(pending);
                S.keys.push_back(pending_key);
                size_t take = 0;
                for (llama_token t : ids) {
                    out("T %d\n", t);
                    ++produced;
                    ++take;
                    if (std::find(o.eos.begin(), o.eos.end(), t) != o.eos.end() || llama_vocab_is_eog(vocab, t)) {
                        finish = "stop";
                        done = true;
                        break;
                    }
                    if (produced >= limit) { finish = "length"; done = true; break; }
                }
                for (size_t i = 0; i + 1 < take; ++i) { S.live.push_back(ids[i]); S.keys.push_back((uint64_t) (uint32_t) ids[i]); }
                const llama_pos keep = next + (llama_pos) take - 1;     // positions from here on were not emitted
                // the rejected drafts out again (the recurrent state keeps --spec snapshots for this, n_rs_seq)
                if (!llama_memory_seq_rm(mem(), sq, keep, -1) ||
                    (drafts && !llama_memory_seq_rm(llama_get_memory(ctx_dft), 0, keep, -1))) {
                    common_sampler_free(smpl);
                    clear(sq);
                    return refuse("the rejected drafts could not be taken back out of the state");
                }
                pending = ids[take - 1];
                pending_key = (uint64_t) (uint32_t) pending;
                pending_pos = keep;
                next = keep + 1;
                if (!done && g_stop) { finish = "cancel"; done = true; }
                if (!done && (int64_t) S.live.size() + 2 >= o.max_context) { finish = "length"; done = true; }
            }
            if (slot < 0) {                                  // the solo path's sampler ends with its request
                common_sampler_free(smpl);
                smpl = nullptr;
            }
        }
        // the last token produced is not fed: the sequence holds the prompt and every token but the last, as the CUDA
        // engine's do (the server's slot_held) - the next prompt carries it, and is read from there with no rollback
        const size_t read_n = cancelled ? at - resume : n - resume;
        // DONE <generated> <prompt> <prompt ms> <decode ms> <finish> <drafts accepted> <drafts offered> <reused>
        //      <hits> <lookups> <RAM blobs> <file blobs> <file MB> <prompt tokens read> <offloaded>
        if (g_phases && ph_n > 0)      // STRATA_PHASES=1: ms per verify pass, to stderr (the target and catch-up synced)
            std::fprintf(stderr, "PHASES passes %d draft %.2f target %.2f catchup %.2f sample %.2f ms/pass\n", ph_n,
                         ph[0] / ph_n, ph[1] / ph_n, ph[2] / ph_n, ph[3] / ph_n);
        out("DONE %lld %zu %.1f %.1f %s %lld %lld %zu 0 0 0 0 0.0 %zu 0\n", (long long) produced, n, prompt_ms,
            ms_since(t1), finish, (long long) accepted, (long long) drafted, resume, read_n);
        if (slot >= 0) {
            // the slot goes on when its first token is not the end and it may write more; it holds the prompt, and from
            // here every token it produces but the last (the server's slot_held)
            const bool cont = !cancelled && !g_stop && produced == 1 && slot_max_new > 1 && std::strcmp(finish, "length") == 0;
            if (cont) {
                Slot& X = slots[(size_t) slot];
                X = Slot{true, smpl, pending, pending_pos, 1, slot_max_new, Clock::now()};
                g_bstop[(size_t) slot] = false;
            } else if (smpl) {
                common_sampler_free(smpl);
            }
            out("BADM %d %d\n", slot, cont ? 1 : 0);
        }
    }

    void slot_done(int b, const char* finish) {
        Slot& X = slots[(size_t) b];
        out("BDONE %d %lld %s %.1f\n", b, (long long) X.produced, finish, ms_since(X.t0));
        common_sampler_free(X.smpl);
        X = Slot{};
    }

    // One batch window: every decoding slot's pending token in one pass, a token each (BT <slot> <id>), BDONE at its end.
    void window() {
        std::vector<int> who;
        for (int b = 0; b < (int) slots.size(); ++b) {
            if (!slots[(size_t) b].active) continue;
            if (g_bstop[(size_t) b]) { slot_done(b, "cancel"); continue; }
            who.push_back(b);
        }
        if (who.empty()) return;
        llama_batch batch = llama_batch_init((int32_t) who.size(), 0, 1);
        for (int b : who) {
            const int32_t i = batch.n_tokens++;
            batch.token[i] = slots[(size_t) b].pending;
            batch.pos[i] = slots[(size_t) b].pos;
            batch.n_seq_id[i] = 1;
            batch.seq_id[i][0] = b + 1;
            batch.logits[i] = true;
        }
        const int rc = llama_decode(ctx, batch);
        llama_batch_free(batch);
        if (rc != 0) {
            std::fprintf(stderr, "strata-metal: a batch window failed; its slots end\n");
            for (int b : who) { clear(b + 1); slot_done(b, "cancel"); }
            return;
        }
        for (size_t k = 0; k < who.size(); ++k) {
            const int b = who[k];
            Slot& X = slots[(size_t) b];
            Seq& S = seqs[(size_t) b + 1];
            // row k is this slot's (not sample_and_accept_n(smpl, ctx, std::vector<int>{k}, {}): llama_tokens is a
            // std::vector<int32_t> too, so that picks the overload that takes {k} as a one-token draft and samples row 0)
            const llama_token t = common_sampler_sample(X.smpl, ctx, (int) k);
            common_sampler_accept(X.smpl, t, true);
            S.live.push_back(X.pending);
            S.keys.push_back((uint64_t) (uint32_t) X.pending);
            X.pending = t;
            X.pos += 1;
            ++X.produced;
            out("BT %d %d\n", b, t);
            if (std::find(o.eos.begin(), o.eos.end(), t) != o.eos.end() || llama_vocab_is_eog(vocab, t)) slot_done(b, "stop");
            else if (X.produced >= X.max_new || (int64_t) S.live.size() + 2 >= o.max_context) slot_done(b, "length");
        }
    }

    void save(const std::string& path) {
        const auto t0 = Clock::now();
        const Seq& S = seqs[0];
        const size_t bytes = llama_state_seq_save_file(ctx, path.c_str(), 0, S.live.data(), S.live.size());
        if (bytes == 0) { out("SERR io 0 could not write %s\n", path.c_str()); return; }
        out("SAVED %zu %zu %.1f\n", S.live.size(), bytes, ms_since(t0));
    }

    void restore(const std::string& path) {
        const auto t0 = Clock::now();
        std::vector<llama_token> toks((size_t) o.max_context);
        size_t n = 0;
        clear(0);
        const size_t bytes = llama_state_seq_load_file(ctx, path.c_str(), 0, toks.data(), toks.size(), &n);
        if (bytes == 0) {
            clear(0);
            out("SERR invalid 0 could not read a session for this model and context from %s\n", path.c_str());
            return;
        }
        Seq& S = seqs[0];
        S.live.assign(toks.begin(), toks.begin() + (long) n);
        // a picture's cells cannot be told apart from another picture's after a restore: they match nothing, so a
        // prompt is reused up to its first image at most.  The drafter starts empty: drafting resumes from new tokens.
        S.keys.resize(n);
        for (size_t i = 0; i < n; ++i) S.keys[i] = S.live[i] == g_image_pad ? kImageBit : (uint64_t) (uint32_t) S.live[i];
        out("RESTORED %zu %zu %.1f\n", n, bytes, ms_since(t0));
    }
};

}  // namespace

int main(int argc, char** argv) {
    // the protocol keeps the original stdout; anything a library prints to fd 1 goes to the log (stderr)
    const int proto_fd = dup(STDOUT_FILENO);
    if (proto_fd < 0 || dup2(STDERR_FILENO, STDOUT_FILENO) < 0 || !(g_proto = fdopen(proto_fd, "w"))) {
        std::fprintf(stderr, "strata-metal: cannot set up stdout\n");
        return 1;
    }
    Engine e;
    if (!parse_args(argc, argv, e.o)) return 2;
    llama_backend_init();
    std::string err;
    if (!e.load(err)) {
        out("ERR %s\n", err.c_str());
        return 1;
    }
    out("INFO engine=" STRATA_VERSION "-metal backend=metal llama=" STRATA_LLAMA_COMMIT " context=%lld kv=%s "
        "batch_slots=%d checkpoints=%d im_start=%d n_embd=%d mtp=%d spec=%d\n", (long long) e.o.max_context,
        e.o.kv.c_str(), e.o.batch, e.o.checkpoints, e.im_start, e.n_embd, e.spec ? 1 : 0, e.spec ? e.o.spec : 0);
    out("READY %lld stop\n", (long long) e.o.max_context);
    g_slots = e.o.batch;
    g_bstop.reset(new std::atomic<bool>[(size_t) std::max(1, g_slots)]());
    std::thread(read_stdin).detach();
    for (;;) {
        std::string line;
        if (!next_line(line, !e.decoding())) {         // slots decoding and no command: one batch window
            e.window();
            continue;
        }
        if (line == "QUIT") break;
        if (line.rfind("BGEN ", 0) == 0 || line.rfind("BGENI ", 0) == 0) {
            // BGEN <slot> <max_new> ...: read as a GEN 1 in that slot's sequence, then BADM
            const bool img = line.rfind("BGENI ", 0) == 0;
            char* e1 = nullptr;
            const long b = std::strtol(line.c_str() + (img ? 6 : 5), &e1, 10);
            char* e2 = nullptr;
            const long long mn = std::strtoll(e1, &e2, 10);
            if (b < 0 || b >= e.o.batch || e.slots[(size_t) b].active || mn < 1 || e2 == e1) {
                out("ERR BGEN: no such free slot (--batch %d) or a bad max_new\n", e.o.batch);
                continue;
            }
            e.gen(std::string(img ? "GENI 1" : "GEN 1") + e2, img, (int) b, mn);
        } else if (line.rfind("GENI ", 0) == 0) e.gen(line, true);
        else if (line.rfind("GEN ", 0) == 0) e.gen(line, false);
        else if (line.rfind("SAVE ", 0) == 0) e.save(line.substr(5));
        else if (line.rfind("RESTORE ", 0) == 0) e.restore(line.substr(8));
        else if (line.rfind("VRAM", 0) == 0) out("ERR the Metal engine has no expert cache to resize\n");
        else if (!line.empty()) out("ERR unknown command: %.40s\n", line.c_str());
    }
    if (e.spec) common_speculative_free(e.spec);
    e.spec_init.reset();
    e.init.reset();
    llama_backend_free();
    return 0;
}
