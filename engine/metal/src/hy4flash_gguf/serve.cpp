// engine/metal/src/hy4flash_gguf/serve.cpp - the hy4-flash engine path of `strata` on Metal: the resident request loop.
//
// `strata`'s family switch hands a hy_v4_flash model to strata::hy4_flash::run (include/strata/hy4flash_gguf/engine.hpp).
// On CUDA that lands in src/hy4flash_gguf/engine.cpp; here it lands on the native Metal engine beside this file.
//
// THE PROTOCOL IS NOT NEW. Its definition of record is include/strata/hy4flash_gguf/serve.hpp and its reader is
// `StrataEngine` in terminal/serve/server.py, so the grammar below is theirs:
//     out   READY <max_context>
//     in    GEN <max_new> [key=value ...] <id,id,...>
//     out   T <id>                           one a token, as it is produced
//     out   DONE <generated> <prompt_tokens> <prompt_ms> <decode_ms> <stop|length|nonfinite> <acc> <off> <reuse>
//     in    QUIT
//     out   ERR <message>
// The ids are the LAST field of a GEN line, so the keys are everything between it and max_new. An unknown key is
// ignored rather than refused: a server that sends one must not lose the request.
//
// READY does NOT advertise `stop`: honouring STOP inside a running request needs a reader that is not on the
// request's thread, and the server treats the capability as absent when the word is missing.
//
// SETUP ORDER IS LOAD-BEARING and is the same sequence main.cpp uses: open, load_dense, open_experts, the forward
// pass (its scratch), the MTP block, THEN plan_arena. The arena's size is decided from measured residency, so
// anything allocated after it overcommits by its own size. A prefill chunk also needs a slot floor in every expert
// size class, not only arena bytes.
//
// A request that produces a non-finite value ends with `nonfinite` and the process stays up: Forward distinguishes
// that from an engine failure, and the loop must not turn one into the other.
#include "strata/hy4flash_gguf/engine.hpp"

#include "dispatch.h"
#include "experts.h"
#include "forward.h"
#include "vis_bundle.h"
#include "vis_pos.h"
#include "vis_tower.h"
#include "model.h"
#include "mtp.h"
#include "mtl_rt.h"
#include "strata/hy4flash_gguf/sampler.hpp"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>

namespace strata::hy4_flash {
namespace {

constexpr int32_t kContextMax = 33792;
/// The trunk's `mrope_section`, [time, height, width] over the 32 rotary pairs.  The GGUF does NOT
/// carry `rope.dimension_sections` - only `rope.dimension_count = 64` - so this comes from the
/// authority's config.json, where it appears under both rope_parameters entries and must sum to
/// head_dim / 2 = 32.  The loader asserts that sum below rather than trusting the literal.
constexpr int kMropeSection[3] = {12, 10, 10};

constexpr int kPrefillChunkDefault = 4096;   ///< the Mac default; --prefill-chunk overrides it

const char* flag_value(int argc, char** argv, const char* flag, const char* alias = nullptr) {
    for (int i = 1; i + 1 < argc; ++i)
        if (std::strcmp(argv[i], flag) == 0 || (alias != nullptr && std::strcmp(argv[i], alias) == 0))
            return argv[i + 1];
    return nullptr;
}

bool has_flag(int argc, char** argv, const char* flag) {
    for (int i = 1; i < argc; ++i)
        if (std::strcmp(argv[i], flag) == 0) return true;
    return false;
}

struct Request {
    int max_new = 0;
    SamplerConfig sampler;
    bool ignore_eos = false;
    std::vector<int32_t> ids;
    std::string bundle;     ///< a GENI line's picture file; empty for GEN
};

/// `GEN <max_new> [key=value ...] <id,id,...>`, or `GENI <max_new> [key=value ...] <file> <id,id,...>`;
/// `base` supplies what the line does not override.  The picture file is the field BEFORE the ids, which
/// is how terminal/serve/server.py writes it (`GENI {left}{keys}{pf} {embeddings}` then the ids).
bool parse_gen(const std::string& line, const SamplerConfig& base, Request& out, std::string& err) {
    std::vector<std::string> f;
    for (size_t i = 0; i < line.size();) {
        while (i < line.size() && line[i] == ' ') ++i;
        const size_t s = i;
        while (i < line.size() && line[i] != ' ') ++i;
        if (i > s) f.emplace_back(line.substr(s, i - s));
    }
    if (f.size() < 3) { err = "GEN needs a token count and ids"; return false; }
    out = Request{};
    out.sampler = base;
    out.max_new = std::atoi(f[1].c_str());
    if (out.max_new < 1) { err = "GEN max_new must be at least 1"; return false; }
    const size_t keys_end = f[0] == "GENI" ? (f.size() >= 2 ? f.size() - 2 : 2) : f.size() - 1;
    for (size_t i = 2; i < keys_end; ++i) {
        const size_t eq = f[i].find('=');
        if (eq == std::string::npos) continue;
        const std::string k = f[i].substr(0, eq), v = f[i].substr(eq + 1);
        if (k == "temperature") out.sampler.temperature = (float) std::atof(v.c_str());
        else if (k == "top_p") out.sampler.top_p = (float) std::atof(v.c_str());
        else if (k == "top_k") out.sampler.top_k = std::atoi(v.c_str());
        else if (k == "seed") out.sampler.seed = std::strtoull(v.c_str(), nullptr, 10);
        else if (k == "ignore_eos") out.ignore_eos = v == "1";
    }
    if (f[0] == "GENI") {
        if (f.size() < 4) { err = "GENI needs a picture file and ids"; return false; }
        out.bundle = f[f.size() - 2];
        if (out.bundle.find('=') != std::string::npos) { err = "GENI has no picture file"; return false; }
    }
    const std::string& last = f.back();
    for (size_t i = 0; i < last.size();) {
        const size_t c = last.find(',', i);
        const std::string t = last.substr(i, c == std::string::npos ? std::string::npos : c - i);
        if (!t.empty()) out.ids.push_back((int32_t) std::strtol(t.c_str(), nullptr, 10));
        if (c == std::string::npos) break;
        i = c + 1;
    }
    if (out.ids.empty()) { err = "GEN had no ids"; return false; }
    if (out.sampler.top_p <= 0.0f) out.sampler.top_p = 1.0f;   // the server omits it rather than sending 0
    return true;
}

void say_err(const std::string& what) {
    std::printf("ERR %s\n", what.c_str());
    std::fflush(stdout);
}

}  // namespace

int run(int argc, char** argv) {
    // Before anything, and before a device exists: an environment that would change the arithmetic is refused
    // rather than honoured silently. The offline driver does the same; a second entry point that skipped it
    // would be a way in for exactly the switches this engine is frozen against.
    if (const std::string bad = strata::hy4_flash::rejected_env(); !bad.empty()) {
        std::fprintf(stderr, "strata hy4_flash: %s\n", bad.c_str());
        return 2;
    }
    const char* path = flag_value(argc, argv, "--native", "--model");
    if (path == nullptr) {
        std::fprintf(stderr, "strata hy4_flash: --native PATH names the model's first shard\n");
        return 2;
    }
    if (!has_flag(argc, argv, "--serve")) {
        // A refusal, not a half-run: the offline paths (token dumps, parity, snapshots) are strata_hy4_flash_metal's and
        // naming them is more use than running a different thing here.
        std::fprintf(stderr, "strata hy4_flash: this path serves requests; give --serve. For offline runs use "
                             "strata_hy4_flash_metal (--tokens, --greedy, --block-test, --save-prefill)\n");
        return 2;
    }
    const char* pack = flag_value(argc, argv, "--pack");
    if (pack == nullptr) {
        std::fprintf(stderr, "strata hy4_flash: --pack DIR names the prepared pack\n");
        return 2;
    }
    int context = 4096;
    if (const char* v = flag_value(argc, argv, "--context")) context = std::atoi(v);
    if (context < 2 || context > kContextMax) {
        std::fprintf(stderr, "strata hy4_flash: --context %d is outside 2..%d\n", context, kContextMax);
        return 2;
    }
    const bool want_mtp = has_flag(argc, argv, "--mtp");
    int draft_k = 2;
    if (const char* v = flag_value(argc, argv, "--draft-tokens")) draft_k = std::atoi(v);
    if (draft_k < 0 || draft_k >= strata::hy4_flash::Forward::kMaxWindow) {
        std::fprintf(stderr, "strata hy4_flash: --draft-tokens %d is outside 0..%d\n", draft_k,
                     strata::hy4_flash::Forward::kMaxWindow - 1);
        return 2;
    }
    int chunk = kPrefillChunkDefault;
    if (const char* v = flag_value(argc, argv, "--prefill-chunk")) chunk = std::atoi(v);
    if (chunk < 1 || chunk > kContextMax) {
        std::fprintf(stderr, "strata hy4_flash: --prefill-chunk %d is outside 1..%d\n", chunk, kContextMax);
        return 2;
    }
    SamplerConfig base;
    {
        std::string err;
        if (!sampler_config_from_args(argc, argv, base, err)) {
            std::fprintf(stderr, "strata hy4_flash: %s\n", err.c_str());
            return 2;
        }
    }

    mtl::Context cx;
    cx.lock_safe_math();
    strata::hy4_flash::Model m;
    if (const std::string err = strata::hy4_flash::open(m, path, context, want_mtp); !err.empty()) {
        std::fprintf(stderr, "strata hy4_flash: %s: %s\n", path, err.c_str());
        return 2;
    }
    // A chunk that ends inside an indexer kpool group leaves that group to be pooled by the next chunk.
    if (chunk % m.geo.idx_kpool != 0) {
        std::fprintf(stderr, "strata hy4_flash: --prefill-chunk %d is not a multiple of the indexer's kpool (%d)\n",
                     chunk, m.geo.idx_kpool);
        return 2;
    }
    std::fprintf(stderr, "strata hy4_flash: %s\n%s\n", path, strata::hy4_flash::describe(m).c_str());
    if (const std::string err = strata::hy4_flash::load_dense(m, cx); !err.empty()) {
        std::fprintf(stderr, "strata hy4_flash: %s\n", err.c_str());
        return 2;
    }
    if (const std::string err = strata::hy4_flash::open_experts(m); !err.empty()) {
        std::fprintf(stderr, "strata hy4_flash: %s\n", err.c_str());
        return 2;
    }
    strata::hy4_flash::Forward fw;
    strata::hy4_flash::Mtp mtp;
    if (const std::string err = fw.open(m, cx, pack, strata::hy4_flash::Prefill::from_env(chunk)); !err.empty()) {
        std::fprintf(stderr, "strata hy4_flash: the forward pass could not start: %s\n", err.c_str());
        return 2;
    }
    if (want_mtp)
        if (const std::string err = mtp.open(m, cx, fw.library()); !err.empty()) {
            std::fprintf(stderr, "strata hy4_flash: the MTP block could not start: %s\n", err.c_str());
            return 2;
        }
    const strata::hy4_flash::ArenaPlan plan = strata::hy4_flash::plan_arena(m, cx);
    std::fprintf(stderr, "  %s\n", strata::hy4_flash::arena_line(plan).c_str());
    const std::string profile = std::string(pack) + "/expert-profile.bin";
    const int min_slots = 2 * m.stream.n_experts() + 64;
    if (const std::string err = strata::hy4_flash::make_arena(m, cx, plan, profile, min_slots); !err.empty()) {
        std::fprintf(stderr, "strata hy4_flash: %s\n", err.c_str());
        return 2;
    }
    for (int c = 0; c < m.stream.n_classes(); ++c)
        if (m.stream.class_slots(c) < 2 * m.stream.n_experts()) {
            std::fprintf(stderr, "strata hy4_flash: prefill needs %d slots in every expert size class and class %d "
                                 "has %d; raise STRATA_METAL_ARENA_GB\n",
                         2 * m.stream.n_experts(), c, m.stream.class_slots(c));
            return 2;
        }
    if (const std::string err = fw.attach_prefill(); !err.empty()) {
        std::fprintf(stderr, "strata hy4_flash: %s\n", err.c_str());
        return 2;
    }

    // --vision-shard: the mmproj beside the trunk.  terminal/strata_control/hy4_flash.py passes it whenever
    // `--vision on`, and refuses to offer the flag at all unless this binary carries the marker the
    // capability probe looks for (hy4_flash.py:35) - which the encode line below provides.
    strata::hy4_flash::VisionShard vshard;
    strata::hy4_flash::VisionTower tower;
    if (const char* vs = flag_value(argc, argv, "--vision-shard")) {
        if (const std::string err = strata::hy4_flash::open_vision(vshard, cx, vs, path, m.geo.hidden_dim);
            !err.empty()) {
            std::fprintf(stderr, "strata hy4_flash: --vision-shard %s: %s\n", vs, err.c_str());
            return 2;
        }
        if (const std::string err = tower.open(cx, vshard, fw.library()); !err.empty()) {
            std::fprintf(stderr, "strata hy4_flash: the vision tower could not start: %s\n", err.c_str());
            return 2;
        }
        std::fprintf(stderr, "  %s\n", strata::hy4_flash::describe_vision(vshard).c_str());
    }

    if (const char* vs = flag_value(argc, argv, "--vision-shard"); vs != nullptr) {
        const int sum = kMropeSection[0] + kMropeSection[1] + kMropeSection[2];
        if (sum * 2 != m.geo.rope_dim) {
            std::fprintf(stderr, "strata hy4_flash: mrope_section %d+%d+%d sums to %d, but this trunk's rotary "
                                 "width is %d and the authority requires the sum to be half of it\n",
                         kMropeSection[0], kMropeSection[1], kMropeSection[2], sum, m.geo.rope_dim);
            return 2;
        }
    }

    std::vector<int32_t> stop_ids = m.geo.stop_tokens;
    if (has_flag(argc, argv, "--ignore-eos")) stop_ids.clear();
    const int64_t vocab = fw.vocab();

    std::printf("READY %d\n", context);
    std::fflush(stdout);

    std::string line;
    while (true) {
        if (!std::getline(std::cin, line)) return 0;
        while (!line.empty() && (line.back() == '\r' || line.back() == '\n')) line.pop_back();
        if (line.empty()) continue;
        if (line == "QUIT") return 0;
        if (line == "STOP") continue;   // nothing is running; READY did not promise mid-request STOP
        if (line.rfind("GEN ", 0) != 0 && line.rfind("GENI ", 0) != 0) {
            say_err("this engine serves GEN, GENI and QUIT only (got: " + line.substr(0, 64) + ")");
            continue;
        }
        Request rq;
        std::string err;
        if (!parse_gen(line, base, rq, err)) { say_err(err); continue; }
        bool out_of_range = false;
        for (const int32_t id : rq.ids) out_of_range = out_of_range || id < 0 || id >= vocab;
        if (out_of_range) { say_err("token outside vocabulary"); continue; }
        if ((int) rq.ids.size() + rq.max_new > context) {
            say_err("the prompt and its " + std::to_string(rq.max_new) + " new tokens need " +
                    std::to_string(rq.ids.size() + (size_t) rq.max_new) + " of " + std::to_string(context) +
                    " context positions");
            continue;
        }

        // Every request starts from position 0: the caches and positions of the previous one are forgotten here,
        // so one request cannot read another's KV.
        fw.reset();
        if (want_mtp) mtp.reset();
        std::vector<int32_t> seq = rq.ids;

        // ---- pictures ------------------------------------------------------------------------------
        // Every buffer here is scoped to the request: the spans point into `rows`, and both die before the
        // next line is read.  fw.set_image_prompt(nullptr, ...) at the end is what keeps a picture from
        // outliving its request - the same mistake ignore_eos made once already.
        std::vector<strata::hy4_flash::BundlePicture> pics;
        std::vector<float> rows;
        std::vector<strata::hy4_flash::Forward::ImageSpanRows> spans;
        std::vector<int32_t> axes;
        int32_t mrope_delta = 0;
        if (!rq.bundle.empty()) {
            if (!vshard.loaded()) { say_err("this engine was started without --vision-shard"); continue; }
            if (const std::string e = strata::hy4_flash::read_bundle(pics, rq.bundle, seq.data(),
                                                               (int64_t) seq.size(), vshard.geo);
                !e.empty()) {
                say_err(e);
                continue;
            }
            const int32_t od = vshard.geo.out_hidden;
            std::string why;
            int64_t total = 0;
            for (const auto& p : pics) total += p.count;
            rows.resize((size_t) total * (size_t) od);
            int64_t at = 0;
            double t_enc = mtl::now_s();
            for (const auto& p : pics) {
                std::vector<float> one;
                if (const std::string e = tower.encode(cx, vshard, p.rgb.data(), p.h, p.w, one); !e.empty()) {
                    why = e;
                    break;
                }
                if ((int64_t) one.size() != p.count * (int64_t) od) {
                    why = "the tower returned " + std::to_string(one.size() / (size_t) od) +
                          " rows where the prompt holds " + std::to_string(p.count);
                    break;
                }
                std::memcpy(rows.data() + (size_t) at * (size_t) od, one.data(), one.size() * sizeof(float));
                spans.push_back({p.start, p.count, nullptr});
                at += p.count;
            }
            if (!why.empty()) { say_err(why); continue; }
            at = 0;
            for (size_t i = 0; i < spans.size(); ++i) {   // after `rows` has stopped growing
                spans[i].rows = rows.data() + (size_t) at * (size_t) od;
                at += spans[i].count;
            }
            // The trunk's 3-axis positions for the prompt, from the picture geometry the HEADER carried.
            std::vector<strata::hy4_flash::ImageSpan> isp;
            std::vector<std::pair<int32_t, int32_t>> grids;
            for (const auto& p : pics)
                grids.emplace_back(p.h / vshard.geo.patch, p.w / vshard.geo.patch);
            if (const std::string e = strata::hy4_flash::build_image_spans(isp, seq.data(), (int64_t) seq.size(),
                                                                     vshard.geo.image_token, grids,
                                                                     vshard.geo.merge);
                !e.empty()) {
                say_err(e);
                continue;
            }
            if (const std::string e = strata::hy4_flash::build_mrope_positions(axes, (int64_t) seq.size(), isp,
                                                                         vshard.geo.merge);
                !e.empty()) {
                say_err(e);
                continue;
            }
            mrope_delta = (int32_t) ((int64_t) axes[(size_t) seq.size() - 1] + 1 - (int64_t) seq.size());
            fw.set_image_prompt(&spans, axes.data(), (int64_t) seq.size(), mrope_delta,
                                3 * kMropeSection[1], 3 * kMropeSection[2]);
            // The marker terminal/strata_control/hy4_flash.py:35 probes for, and the same line vision.cu prints.
            std::fprintf(stderr, "HY4_FLASH_VISION encoded pictures=%d rows=%lld encode_s=%.3f\n",
                         (int) pics.size(), (long long) total, mtl::now_s() - t_enc);
        }
        struct ClearImages {
            strata::hy4_flash::Forward& fw;
            ~ClearImages() { fw.set_image_prompt(nullptr, nullptr, 0, 0, 0, 0); }
        } clear_images{fw};

        const double t_start = mtl::now_s();
        strata::hy4_flash::Forward::PrefillSink sink;
        if (want_mtp) sink.hidden = [&](int pos, int n, const float* h) { mtp.set_hidden(pos + 1, n, h); };
        if (const std::string e = fw.prompt(seq.data(), (int) seq.size(), sink); !e.empty()) {
            if (strata::hy4_flash::Forward::is_nonfinite(e)) {
                std::printf("DONE 0 %d %.1f %.1f nonfinite 0 0 0\n", (int) seq.size(),
                            (mtl::now_s() - t_start) * 1e3, 0.0);
                std::fflush(stdout);
                continue;
            }
            say_err("prefill: " + e);
            continue;
        }
        const double prompt_ms = (mtl::now_s() - t_start) * 1e3;

        int produced = 0;
        const char* finish = "length";
        const bool greedy = rq.sampler.greedy();
        Sampler sampler(rq.sampler);
        // Per request: a GEN line's ignore_eos must not outlive it, or one request disarms the stop ids for
        // every later one on the same process.
        const std::vector<int32_t> req_stop = rq.ignore_eos ? std::vector<int32_t>{} : stop_ids;
        const auto is_stop = [&req_stop](int32_t id) {
            for (const int32_t s : req_stop) if (s == id) return true;
            return false;
        };
        const double t_decode = mtl::now_s();
        std::string decode_err;
        if (greedy && want_mtp && draft_k > 0) {
            // The drafted path emits through the callback, so the stop check runs on the tokens it commits.
            bool stopped = false;
            decode_err = strata::hy4_flash::speculative_greedy(
                fw, mtp, seq, rq.max_new, draft_k, context, strata::hy4_flash::MtpProbe::none, [&](int32_t tok) {
                    if (stopped) return;
                    if (is_stop(tok)) { stopped = true; finish = "stop"; return; }
                    ++produced;
                    std::printf("T %d\n", (int) tok);
                    std::fflush(stdout);
                });
        } else {
            for (int i = 0; i < rq.max_new; ++i) {
                int32_t next;
                if (greedy) {
                    next = fw.last_token();
                } else {
                    const int64_t picked = sampler.pick(fw.last_logits(), vocab, seq);
                    next = (int32_t) picked;
                }
                // The id comes back from the GPU and indexes token_embd's rows in the next window, so it is
                // range-checked here rather than there. Out of range means the logits were not finite.
                if (next < 0 || next >= vocab) { finish = "nonfinite"; break; }
                if (is_stop(next)) { finish = "stop"; break; }
                seq.push_back(next);
                ++produced;
                std::printf("T %d\n", (int) next);
                std::fflush(stdout);
                if ((int32_t) seq.size() >= context) break;
                if (const std::string e = fw.window(seq.data(), (int) seq.size() - 1, 1); !e.empty()) {
                    decode_err = e;
                    break;
                }
            }
        }
        if (!decode_err.empty()) {
            if (!strata::hy4_flash::Forward::is_nonfinite(decode_err)) {
                say_err("decode: " + decode_err);
                continue;
            }
            finish = "nonfinite";
        }
        const double decode_ms = (mtl::now_s() - t_decode) * 1e3;
        const unsigned long long acc = want_mtp ? (unsigned long long) mtp.stats().accepted : 0;
        const unsigned long long off = want_mtp ? (unsigned long long) mtp.stats().offered : 0;
        std::printf("DONE %d %d %.1f %.1f %s %llu %llu 0\n", produced, (int) rq.ids.size(), prompt_ms, decode_ms,
                    finish, acc, off);
        std::fflush(stdout);
    }
}

}  // namespace strata::hy4_flash
