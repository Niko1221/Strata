// hy4-flash model loading, inference and verification benchmarks.
#include "strata/hy4flash_gguf/engine.hpp"

#include "strata/artifact/gguf_reader.hpp"
#include "strata/artifact/model_family.hpp"
#include "strata/core/native_dense.hpp"
#include "strata/core/weights.hpp"
#include "strata/hy4flash_gguf/attention.hpp"
#include "strata/hy4flash_gguf/dfly.hpp"
#include "strata/hy4flash_gguf/dfly_loop.hpp"
#include "strata/hy4flash_gguf/forward.hpp"
#include "strata/hy4flash_gguf/mat.hpp"
#include "strata/hy4flash_gguf/sampler.hpp"
#include "strata/hy4flash_gguf/serve.hpp"
#include "strata/hy4flash_gguf/mtp.hpp"
#include "strata/hy4flash_gguf/geometry.hpp"
#include "strata/hy4flash_gguf/indexer.hpp"
#include "strata/hy4flash_gguf/kv_cache.hpp"
#include "strata/hy4flash_gguf/cpu/expert_layout.hpp"
#include "strata/kernels/native_mmvq.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <charconv>
#include <cerrno>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <exception>
#include <fstream>
#include <map>
#include <set>
#include <string>
#include <vector>

// The profiler range `--verify-bench` brackets for `nsys --capture-range=cudaProfilerApi`. Declared here
// because this toolkit ships libcudart's two symbols without cuda_profiler_api.h.
extern "C" cudaError_t cudaProfilerStart(void);
extern "C" cudaError_t cudaProfilerStop(void);

namespace strata::hy4_flash {
namespace {

const char* model_path(int argc, char** argv) {
    for (int i = 1; i + 1 < argc; ++i)
        if (std::strcmp(argv[i], "--native") == 0 || std::strcmp(argv[i], "--model") == 0) return argv[i + 1];
    return nullptr;
}

bool has_flag(int argc, char** argv, const char* flag) {
    for (int i = 1; i < argc; ++i)
        if (std::strcmp(argv[i], flag) == 0) return true;
    return false;
}

const char* flag_value(int argc, char** argv, const char* flag) {
    for (int i = 1; i + 1 < argc; ++i)
        if (std::strcmp(argv[i], flag) == 0) return argv[i + 1];
    return nullptr;
}

/// Validate packed tensor shapes before uploading model weights.
struct Loaded {
    core::WeightTable table;
    core::NativeDense dense;
    void* arena = nullptr;
    uint64_t arena_bytes = 0;
    int64_t native_served = 0;
    int64_t shape_only = 0;

    ~Loaded() {
        // The arena outlives every WeightRef that points into it, so it is freed last and only here.
        if (arena) cudaFree(arena);
    }
};

bool load_pack(const std::string& pack, const std::string& shard, const Geometry& g, Loaded& out,
               std::string& err) try {
    const std::vector<std::string> shards = gguf_split_paths(shard);

    // Exclude shape-only pack entries from the dense arena.
    std::set<std::string> skip;
    if (!core::WeightTable::index_shape_only(pack, skip, err)) return false;
    out.shape_only = (int64_t) skip.size();
    // Of those, the ones NativeDense uploads and attaches itself.
    std::set<std::string> served;
    if (!core::NativeDense::served_names(shards, /*include_ple_key=*/false, served, err)) return false;
    out.native_served = (int64_t) served.size();

    if (!core::WeightTable::pool_bytes(pack, out.arena_bytes, err, skip.empty() ? nullptr : &skip)) return false;
    if (const cudaError_t ce = cudaMalloc(&out.arena, out.arena_bytes); ce != cudaSuccess) {
        cudaGetLastError();
        size_t free_b = 0, total_b = 0;
        cudaMemGetInfo(&free_b, &total_b);
        err = "cudaMalloc(" + std::to_string(out.arena_bytes) + ") for the weight arena failed (" +
              cudaGetErrorString(ce) + "): " + std::to_string(free_b >> 20) + " MiB of " +
              std::to_string(total_b >> 20) + " MiB free";
        return false;
    }
    if (!out.table.load(pack, out.arena, out.arena_bytes, err, skip.empty() ? nullptr : &skip)) return false;
    if (!out.dense.load(shards, out.table, err, /*include_ple_key=*/false, 0, g.trunk_blocks)) return false;

    // One line per expert ordinal, not per block: this model's leading block has a dense FFN and routes nothing.
    if (!cpu::expert_layout_load(pack, g.n_expert_ordinals(), g.n_experts, err,
                                          /*block_base=*/g.dense_blocks, g.expert_dim, g.expert_ff,
                                          g.swiglu_clamp, g.trunk_blocks - g.dense_blocks))
        return false;
    return true;
} catch (const std::exception& e) {
    err = std::string("loading the pack: ") + e.what();
    return false;
}

/// Validate packed tensor metadata against the GGUF descriptors.
bool verify_against_gguf(const std::string& shard, const Loaded& l, int64_t& checked, std::string& err) try {
    checked = 0;
    const core::WeightTable& t = l.table;
    for (const std::string& path : gguf_split_paths(shard)) {
        const GgufFile gguf(path);
        for (const TensorInfo& ti : gguf.tensors()) {
            const core::WeightRef* w = t.find(ti.name);
            if (w == nullptr) continue;   // experts and the Engram table are not in the arena, by design
            const int64_t ne0 = ti.shape.empty() ? 0 : (int64_t) ti.shape[0];
            const int64_t ne1 = ti.shape.size() > 1 ? (int64_t) ti.shape[1] : 0;
            const int64_t ne2 = ti.shape.size() > 2 ? (int64_t) ti.shape[2] : 0;
            if (w->ne0 != ne0 || w->ne1 != ne1 || w->ne2 != ne2) {
                err = ti.name + ": the pack says [" + std::to_string(w->ne0) + ", " + std::to_string(w->ne1) +
                      ", " + std::to_string(w->ne2) + "], the model [" + std::to_string(ne0) + ", " +
                      std::to_string(ne1) + ", " + std::to_string(ne2) + "]";
                return false;
            }
            ++checked;
        }
    }
    return true;
} catch (const std::exception& e) {
    err = std::string("verifying the pack against the model: ") + e.what();
    return false;
}

/// Report the detected architecture, geometry and model support status.
void print_status(const char* path, const std::string& guard, const std::string& shape) {
    std::printf("%s (architecture '%s')\n", model_family_title(ModelFamily::Hy4Flash),
                model_family_arch(ModelFamily::Hy4Flash));
    if (path) std::printf("  model file     %s\n", path);
    std::printf("  metadata       %s\n", guard.empty() ? "readable, every key this engine needs is present"
                                                       : guard.c_str());
    if (!shape.empty()) std::printf("%s\n", shape.c_str());
    std::printf("  expert tiers   --expert-vram <GiB> sizes the GPU expert pool.\n"
                "                 Default: preload all trunk experts into the host file cache.\n"
                "                 --resident-experts keeps the GPU complement in RAM and swaps on demand.\n");
}

bool read_budget(int argc, char** argv, const char* flag, const char* env, std::size_t& bytes) {
    const char* value = std::getenv(env);
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], flag) == 0) {
            if (++i == argc) return false;
            value = argv[i];
        }
    }
    if (!value) return true;
    char* end = nullptr;
    errno = 0;
    const double gib = std::strtod(value, &end);
    const double size = gib * (1ULL << 30);
    if (errno == ERANGE || end == value || *end || !std::isfinite(size) || gib < 0 ||
        (gib > 0 && size < 1) || size >= (double) SIZE_MAX) return false;
    bytes = static_cast<std::size_t>(size);
    return true;
}

}  // namespace

namespace {

/// Check the drafter on a real prefix and validate the target cache after verification.
int dfly_check(int argc, char** argv, const char* draft_path, const char* path, const Geometry& g,
               const char* pack, const std::vector<int32_t>& tokens, const ForwardOptions& fo, DenseResidency& dense,
               KvCache& cache, int reserve) {
    const int64_t vocab = g.vocab_size;
    int depth = 600, k = 3, steps = 32;
    if (const char* v = flag_value(argc, argv, "--dfly-depth")) depth = std::atoi(v);
    if (const char* v = flag_value(argc, argv, "--dfly-k")) k = std::atoi(v);
    if (const char* v = flag_value(argc, argv, "--dfly-steps")) steps = std::max(0, std::atoi(v));
    const bool synthetic = has_flag(argc, argv, "--dfly-synthetic");
    DflyDrafter::BindOptions bo;
    if (const char* v = flag_value(argc, argv, "--dfly-negative-control")) {
        if (std::string(v) != "reverse-layers") {
            std::fprintf(stderr, "hy4_flash: --dfly-negative-control knows only reverse-layers\n");
            return 2;
        }
        bo.reverse_layers = true;
    }
    if (k < 1 || k > DflyDrafter::kMaxK) { std::fprintf(stderr, "hy4_flash: --dfly-k must be 1..%d\n", DflyDrafter::kMaxK); return 2; }
    const int w = k + 1;
    if (depth < 2 || depth > (int) tokens.size() || depth + steps + w + 1 > reserve) {
        std::fprintf(stderr, "hy4_flash: --dfly-check needs 2 <= depth <= %zu ids and depth + steps + %d <= %d reserved\n",
                     tokens.size(), w + 1, reserve);
        return 2;
    }
    if (kv_format(cache) == kKvF32) { std::fprintf(stderr, "hy4_flash: --dfly-check needs F16 or Q8_0 KV\n"); return 2; }
    std::string err;
    // Every sparse block's selection is kept; the context picks the drafter's five BY BLOCK ID.
    const int n_blk = (int) g.layers.size() - g.n_mtp_blocks;
    DflyCapture cap;
    cap.mode = DflyCapture::Mode::prefill_tail;
    for (int b = 0; b < n_blk; ++b)
        if (g.layers[(std::size_t) b].attention != Attention::Sliding) cap.sel_blocks.push_back(b);
    if (!cap.init(g, 1, err)) { std::fprintf(stderr, "hy4_flash: %s\n", err.c_str()); return 2; }

    // ---- prefill, the last chunk with logits and the capture
    std::vector<int32_t> history(tokens.begin(), tokens.begin() + depth);
    const int chunk = prefill_chunk_max();
    std::vector<float> lg;
    cache.len = 0;
    for (int at = 0; at < depth; at += chunk) {
        const int n = std::min(chunk, depth - at);
        const bool last = at + n == depth;
        const std::vector<int32_t> part(history.begin() + at, history.begin() + at + n);
        ForwardOptions co = fo;
        co.history = &history;
        co.want_logits = last;
        co.logits_last_only = last && n > kGemmT;
        co.dfly = last ? &cap : nullptr;
        if (!forward_prefill(path, g, pack, part, lg, err, co) || (last && !cap.valid)) {
            std::fprintf(stderr, "hy4_flash: --dfly-check prefill at %d: %s\n", at, err.c_str());
            return 2;
        }
    }
    const bool resident = dense_experts_complete(dense);
    std::printf("DF_CONF depth=%d k=%d steps=%d synthetic=%d negative_control=%s experts_resident=%d reserve=%d\n",
                depth, k, steps, synthetic ? 1 : 0, bo.reverse_layers ? "reverse-layers" : "none", resident ? 1 : 0,
                reserve);
    if (!resident && !has_flag(argc, argv, "--verify-allow-partial")) {
        std::fprintf(stderr, "hy4_flash: --dfly-check needs the expert tier COMPLETE (or --verify-allow-partial)\n");
        return 2;
    }
    Sampler greedy(SamplerConfig{});
    const std::size_t last_row = lg.size() / (std::size_t) vocab - 1;
    int32_t anchor = (int32_t) greedy.pick(lg.data() + last_row * (std::size_t) vocab, vocab, history);
    if (anchor < 0) { std::fprintf(stderr, "hy4_flash: non-finite prefill logits\n"); return 2; }

    // ---- the drafter, after the target is warm
    DflyDrafter drafter;
    if (!drafter.load(draft_path, g, err) || !drafter.bind(dense, cache, err, bo)) {
        std::fprintf(stderr, "hy4_flash: --dfly-check: %s\n", err.c_str());
        return 2;
    }
    const std::array<int, 5> L = drafter.target_layers();
    std::printf("DF_DRAFTER weights=%.1f MiB scratch=%.1f MiB layers=%d,%d,%d,%d,%d mask=%d\n",
                drafter.weight_bytes() / 1048576.0, drafter.scratch_bytes() / 1048576.0, L[0], L[1], L[2], L[3], L[4],
                drafter.mask_id());
    std::fflush(stdout);
    const auto is_drafter_block = [&](int b) { return std::find(L.begin(), L.end(), b) != L.end(); };

    // Raw KV bytes of rows [lo, hi) of one block.
    const auto kv_rows = [&](int b, int lo, int hi, std::vector<unsigned char>& v) {
        for (int part = 0; part < 2; ++part) {
            const auto row_bytes = part == 0 ? kv_cache_lat_row_bytes(cache.kv_lora, kv_format(cache))
                                             : kv_cache_pe_row_bytes(cache.n_rot, kv_format(cache));
            const std::size_t bytes = (std::size_t) (hi - lo) * row_bytes, at = v.size();
            v.resize(at + bytes);
            const unsigned char* base = (const unsigned char*) (part == 0 ? cache.kv_cmpr[(std::size_t) b]
                                                                         : cache.k_pe[(std::size_t) b]);
            if (bytes && cudaMemcpy(v.data() + at, base + (std::size_t) lo * row_bytes, bytes, cudaMemcpyDeviceToHost) !=
                             cudaSuccess)
                return false;
        }
        return true;
    };
    struct Snap { std::vector<unsigned char> prefix, others, mine, idx; int len = 0; };
    const auto snap = [&](int a, Snap& s) {
        s = Snap{};
        s.len = cache.len;
        for (int b = 0; b < n_blk; ++b) {
            if (!kv_rows(b, 0, a, s.prefix)) return false;
            if (!kv_rows(b, a, a + w, is_drafter_block(b) ? s.mine : s.others)) return false;
        }
        std::string e;
        return dfly_indexer_snapshot(dense, s.idx, e);
    };

    long fails = 0, checks = 0;
    const auto gate = [&](bool ok, const char* what, int step) {
        ++checks;
        if (!ok) { ++fails; std::printf("DF_FAIL %s step=%d\n", what, step); }
    };
    // One checked proposal at position a from `ctx`; `again` repeats it and compares.
    const auto checked = [&](int step, int32_t anc, int a, const DflyContext& ctx, std::vector<int32_t>& out,
                             bool again) -> bool {
        out.assign((std::size_t) k, -1);   // a fallback round counts as no match
        Snap pre, post;
        if (!snap(a, pre)) { std::fprintf(stderr, "hy4_flash: snapshot failed\n"); return false; }
        std::vector<int32_t> d((std::size_t) k, -1);
        std::vector<float> c((std::size_t) k, -1.f);
        const auto t0 = std::chrono::steady_clock::now();
        const DflyResult r = drafter.propose(anc, a, ctx, k, d.data(), c.data(), err);
        const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        if (r != DflyResult::ok) {
            std::printf("DF_FAIL propose=%s step=%d: %s\n", dfly_result_name(r), step, err.c_str());
            ++fails; ++checks;
            return r == DflyResult::safe_fallback;   // the others leave nothing to compare
        }
        drafter.drop_pending();
        bool in_vocab = true, conf_ok = true;
        for (int j = 0; j < k; ++j) {
            in_vocab = in_vocab && d[(std::size_t) j] >= 0 && d[(std::size_t) j] < vocab;
            conf_ok = conf_ok && std::isfinite(c[(std::size_t) j]) && c[(std::size_t) j] >= 0.f && c[(std::size_t) j] <= 1.f;
        }
        if (!snap(a, post)) { std::fprintf(stderr, "hy4_flash: snapshot failed\n"); return false; }
        gate(in_vocab, "ids_in_vocab", step);
        gate(conf_ok, "confidence_finite_0_1", step);
        gate(post.prefix == pre.prefix, "kv_prefix_unchanged", step);
        gate(post.others == pre.others, "kv_window_other_blocks_unchanged", step);
        gate(post.mine != pre.mine, "kv_window_drafter_blocks_written", step);
        gate(post.idx == pre.idx, "indexer_unchanged", step);
        gate(post.len == pre.len && post.len == a, "cache_len_unchanged", step);
        std::printf("DF_STEP step=%d a=%d engaged=%d result=ok ms=%.2f draft=", step, a, ctx.engaged ? 1 : 0, ms);
        for (int j = 0; j < k; ++j) std::printf("%s%d", j ? "," : "", d[(std::size_t) j]);
        std::printf(" conf=");
        for (int j = 0; j < k; ++j) std::printf("%s%.4f", j ? "," : "", c[(std::size_t) j]);
        std::printf("\n");
        if (again) {
            std::vector<int32_t> d2((std::size_t) k, -1);
            std::vector<float> c2((std::size_t) k, -1.f);
            const DflyResult r2 = drafter.propose(anc, a, ctx, k, d2.data(), c2.data(), err);
            drafter.drop_pending();
            Snap post2;
            if (!snap(a, post2)) return false;
            gate(r2 == DflyResult::ok && d2 == d && c2 == c, "deterministic_ids", step);
            gate(post2.mine == post.mine && post2.prefix == pre.prefix, "deterministic_kv", step);
        }
        out = d;
        std::fflush(stdout);
        return true;
    };

    int a = depth;
    if (synthetic) {
        // A capture used as a container: one random hidden row, then a synthetic selection.
        DflyCapture syn;
        syn.mode = DflyCapture::Mode::prefill_tail;
        syn.sel_blocks.assign(L.begin(), L.end());
        std::sort(syn.sel_blocks.begin(), syn.sel_blocks.end());
        if (!syn.init(g, 1, err)) { std::fprintf(stderr, "hy4_flash: %s\n", err.c_str()); return 2; }
        uint64_t st = 0x9E3779B97F4A7C15ull;
        const auto rnd = [&]() { st ^= st << 13; st ^= st >> 7; st ^= st << 17; return st; };
        std::vector<float> h((std::size_t) g.hidden_dim);
        for (float& x : h) {   // roughly unit Gaussian, Irwin-Hall of 4
            double s = 0;
            for (int i = 0; i < 4; ++i) s += (double) (rnd() >> 11) / 9007199254740992.0;
            x = (float) ((s - 2.0) * std::sqrt(3.0));
        }
        cudaMemcpy(syn.hidden, h.data(), h.size() * 4, cudaMemcpyHostToDevice);
        const int p = a - 1, grp = p / g.idx_kpool;
        // (ii): up to 511 random distinct earlier groups in ascending order, then the tail, -1 padded.
        std::vector<int32_t> sel((std::size_t) syn.sel_cap, -1);
        {
            std::vector<int> groups;
            for (int x = 0; x < grp; ++x)
                if (rnd() % 3 != 0) groups.push_back(x);
            if ((int) groups.size() > g.idx_top_k / g.idx_kpool - 1) groups.resize((std::size_t) (g.idx_top_k / g.idx_kpool - 1));
            std::size_t at = 0;
            for (int x : groups)
                for (int q = 0; q < g.idx_kpool; ++q) sel[at++] = x * g.idx_kpool + q;
            at = (std::size_t) (g.idx_top_k - g.idx_kpool);
            for (int q = 0; q <= p % g.idx_kpool; ++q) sel[at + (std::size_t) q] = grp * g.idx_kpool + q;
        }
        for (int engaged = 0; engaged < 2; ++engaged) {
            syn.valid = true;
            syn.pos_begin = p;
            syn.n_rows = 1;
            syn.engaged = engaged != 0;
            syn.n_sel = engaged ? g.idx_top_k : 0;
            for (void* sp : syn.sel) cudaMemcpy(sp, sel.data(), (std::size_t) g.idx_top_k * 4, cudaMemcpyHostToDevice);
            syn.generation += 1;
            DflyContext ctx;
            if (!dfly_context_from_capture(syn, 0, L, ctx, err)) { std::fprintf(stderr, "hy4_flash: %s\n", err.c_str()); return 2; }
            std::vector<int32_t> d;
            if (!checked(engaged, anchor, a, ctx, d, true)) return 2;
        }
    } else {
        std::vector<std::vector<int32_t>> drafts;
        std::vector<int32_t> gen{anchor};   // gen[s] is the anchor of step s
        for (int s = 0; s <= steps; ++s) {
            DflyContext ctx;
            if (!dfly_context_from_capture(cap, 0, L, ctx, err)) { std::fprintf(stderr, "hy4_flash: %s\n", err.c_str()); return 2; }
            std::vector<int32_t> d;
            if (!checked(s, anchor, a, ctx, d, s == 0)) return 2;
            drafts.push_back(d);
            if (s == steps) break;
            // one greedy target step on the anchor, capturing its row for the next round
            history.push_back(anchor);
            ForwardOptions co = fo;
            co.history = &history;
            co.want_logits = true;
            co.logits_last_only = false;
            co.dfly = &cap;
            const std::vector<int32_t> one{anchor};
            if (!forward_prefill(path, g, pack, one, lg, err, co) || !cap.valid) {
                std::fprintf(stderr, "hy4_flash: --dfly-check AR step at %d: %s\n", a, err.c_str());
                return 2;
            }
            anchor = (int32_t) greedy.pick(lg.data(), vocab, history);
            if (anchor < 0) { std::fprintf(stderr, "hy4_flash: non-finite logits at %d\n", a); return 2; }
            gen.push_back(anchor);
            ++a;
        }
        // Teacher-forced acceptance against the target's own greedy continuation (reported, not gated).
        long rounds = 0, acc = 0;
        std::vector<long> reach((std::size_t) k, 0), hit((std::size_t) k, 0);
        for (std::size_t s = 0; s < drafts.size(); ++s) {
            if (s + (std::size_t) k >= gen.size()) break;
            int r = 0;
            for (int j = 0; j < k; ++j) {
                ++reach[(std::size_t) j];
                if (drafts[s][(std::size_t) j] != gen[s + 1 + (std::size_t) j]) break;
                ++hit[(std::size_t) j];
                ++r;
            }
            acc += r;
            ++rounds;
        }
        std::printf("DF_ACCEPT rounds=%ld mean_accepted=%.3f tau=%.3f per_position=", rounds,
                    rounds ? (double) acc / rounds : 0.0, rounds ? 1.0 + (double) acc / rounds : 0.0);
        for (int j = 0; j < k; ++j)
            std::printf("%s%ld/%ld", j ? "," : "", hit[(std::size_t) j], reach[(std::size_t) j]);
        std::printf("\n");
    }
    std::printf("DF_RESULT %s checks=%ld fails=%ld\n", fails == 0 ? "PASS" : "FAIL", checks, fails);
    return fails == 0 ? 0 : 1;
}

}  // namespace

namespace {

/// Use the same DFly request loop for CLI generation and serving.
int run_dfly_cli(const char* path, const Geometry& g, const char* pack, const std::vector<int32_t>& tokens,
                 const ForwardOptions& fo, DenseResidency& dense, KvCache& cache, const SamplerConfig& sc,
                 const std::vector<int32_t>& stop_ids, int max_new, const std::string& draft_model, int draft_max,
                 DflyWindow window) {
    std::printf("  draft          dfly, k <= %d, window %s, %s\n", draft_max, dfly_window_name(window),
                draft_model.c_str());
    std::fflush(stdout);
    std::string err;
    DflyRuntime rt;
    if (!dfly_runtime_init(rt, g, err)) { std::fprintf(stderr, "\nhy4_flash: %s\n", err.c_str()); return 2; }
    const char* dir = std::getenv("HY4_FLASH_LOGITS_DIR");
    const int64_t vocab = g.vocab_size;
    int step = 0;
    bool write_failed = false;
    DflyRequest req;
    req.prompt = &tokens;
    req.max_new = max_new;
    req.k_max = draft_max;
    req.window = window;
    req.stop_tokens = &stop_ids;
    req.sampler = sc;
    req.draft_path = draft_model;
    req.emit = [&](int32_t, const float* row) {
        if (dir != nullptr) {
            char file[2048];
            std::snprintf(file, sizeof file, "%s/step-%03d.f32", dir, step);
            std::FILE* f = std::fopen(file, "wb");
            const bool ok = f != nullptr && std::fwrite(row, sizeof(float), (std::size_t) vocab, f) == (std::size_t) vocab;
            if (f != nullptr) std::fclose(f);
            if (!ok) { std::fprintf(stderr, "\nhy4_flash: cannot write %s\n", file); write_failed = true; return false; }
        }
        ++step;
        return true;
    };
    std::vector<int32_t> produced;
    DflyStats st;
    const DflyFinish fin = dfly_generate(path, g, pack, fo, dense, cache, rt, req, produced, st, err);
    std::printf("  dfly           finish %s; %ld rounds, offered %ld, accepted %ld, tau %s; %ld one-row steps, "
                "%ld fallbacks\n",
                dfly_finish_name(fin), st.rounds, st.offered, st.accepted,
                st.rounds > 0 ? std::to_string((double) st.round_outputs / (double) st.rounds).c_str() : "undefined",
                st.ar_steps, st.fallbacks);
    if (window != DflyWindow::fixed)
        std::printf("                 drafted %ld, verify cut in %ld rounds, none verified in %ld\n", st.drafted,
                    st.gated, st.kv0_chosen);
    std::printf("                 per position");
    for (std::size_t i = 0; i < st.reach.size(); ++i) std::printf(" %ld/%ld", st.hit[i], st.reach[i]);
    std::printf("; (k,r)");
    for (const auto& kv : st.kr) std::printf(" (%d,%d):%ld", kv.first.first, kv.first.second, kv.second);
    std::printf("; rounds by kv");
    for (std::size_t i = 0; i < st.k_rounds.size(); ++i) std::printf("%s%ld", i ? "/" : " ", st.k_rounds[i]);
    std::printf("\n");
    const double dec_tok = produced.size() > 1 ? (double) (produced.size() - 1) : 0.0;
    std::printf("                 prefill %.0f ms; decode %.0f ms for %.0f tokens (%.2f t/s; propose %.0f ms, "
                "verify %.0f ms)\n",
                st.prefill_ms, st.decode_ms, dec_tok, st.decode_ms > 0 ? dec_tok * 1e3 / st.decode_ms : 0.0,
                st.propose_ms, st.verify_ms);
    if (fin == DflyFinish::stop || fin == DflyFinish::length) {
        std::printf("  generated     ");
        for (int32_t t : produced) std::printf(" %d", t);
        std::printf("\n");
        return 0;
    }
    if (write_failed) return 2;
    std::fprintf(stderr, "\nhy4_flash: %s\n", err.empty() ? dfly_finish_name(fin) : err.c_str());
    return fin == DflyFinish::restart_worker ? 3 : 2;
}

}  // namespace

int run(int argc, char** argv) {
    const char* path = model_path(argc, argv);
    if (path == nullptr) {
        std::fprintf(stderr, "hy4_flash: no --model / --native; this path is only reached with one\n");
        return 2;
    }

    ForwardOptions fo{};
    fo.resident_experts = has_flag(argc, argv, "--resident-experts");
    if (has_flag(argc, argv, "--expert-host") || std::getenv("STRATA_HY4_FLASH_EXPERT_HOST_GIB")) {
        std::fprintf(stderr, "hy4_flash: remove --expert-host / STRATA_HY4_FLASH_EXPERT_HOST_GIB; "
                             "use --resident-experts for GPU/RAM placement or omit it for full host preload\n");
        return 2;
    }
    if (!read_budget(argc, argv, "--expert-vram", "STRATA_HY4_FLASH_EXPERT_VRAM_GIB", fo.expert_vram_bytes)) {
        std::fprintf(stderr, "hy4_flash: --expert-vram requires finite nonnegative GiB; "
                             "positive budgets must be at least one byte\n");
        return 2;
    }

    std::string guard, shape;
    Geometry g;
    try {
        const GgufModel model = GgufModel::open(path);
        const GgufFile& meta = model.meta();
        guard = check_architecture(meta);
        if (guard.empty()) {
            // 0: take the file's own context_length until the engine has a budget to size a limit from.
            const std::string err = read_geometry(meta, g, 0, model.find("per_layer_token_embd.weight"));
            if (!err.empty()) {
                print_status(path, guard, {});
                std::fprintf(stderr, "\nhy4_flash: %s: %s\n", path, err.c_str());
                return 2;
            }
            shape = describe(g);
        }
    } catch (const std::exception& e) {
        std::fprintf(stderr, "hy4_flash: %s: %s\n", path, e.what());
        return 2;
    }

    print_status(path, guard, shape);
    if (!guard.empty()) {
        std::fprintf(stderr, "\nhy4_flash: %s: %s\n", path, guard.c_str());
        return 2;
    }

    // Check the pack against the GGUF before loading weights.
    const char* pack = flag_value(argc, argv, "--pack");
    if (pack == nullptr) {
        std::fprintf(stderr, "\nhy4_flash: --pack DIR is needed to load the weights (build one with "
                             "tools/iq_pack.py)\n");
        return 2;
    }
    std::string err;
    {
        Loaded loaded;
        if (!load_pack(pack, path, g, loaded, err)) {
            std::fprintf(stderr, "\nhy4_flash: %s\n", err.c_str());
            return 2;
        }
        int64_t checked = 0;
        if (!verify_against_gguf(path, loaded, checked, err)) {
            std::fprintf(stderr, "\nhy4_flash: %s\n", err.c_str());
            return 2;
        }
        std::printf("  weights        arena %.2f GiB; %lld from the GGUF (%lld of them native projections); "
                    "%lld shapes verified\n",
                    (double) loaded.arena_bytes / (double) (1u << 30), (long long) loaded.shape_only,
                    (long long) loaded.native_served, (long long) checked);
        std::printf("  experts        layout read: %lld ordinals, block_base %lld\n",
                    (long long) cpu::expert_layout().n_layers,
                    (long long) cpu::expert_layout().block_base);

    } // The validated loader weights are not consumed by forward_prefill.

    // Run a forward pass when input tokens are supplied.
    const bool serving = has_flag(argc, argv, "--serve");
    if (has_flag(argc, argv, "--mtp") && !serving) {
        std::fprintf(stderr,"hy4-flash MTP currently uses --serve; launch it with strata.py launch --mtp on\n");
        return 2;
    }
    const char* tok_arg = flag_value(argc, argv, "--tokens");
    // Accept token IDs from a file for prompts longer than the command-line limit.
    std::string tok_file_text;
    if (const char* tf = flag_value(argc, argv, "--tokens-file")) {
        std::FILE* f = std::fopen(tf, "rb");
        if (f == nullptr) {
            std::fprintf(stderr, "hy4_flash: --tokens-file %s: cannot open\n", tf);
            return 2;
        }
        char buf[65536];
        std::size_t n = 0;
        while ((n = std::fread(buf, 1, sizeof buf, f)) > 0) tok_file_text.append(buf, n);
        std::fclose(f);
        if (tok_arg != nullptr)
            std::fprintf(stderr, "hy4_flash: --tokens-file overrides --tokens\n");
        tok_arg = tok_file_text.c_str();
    }
    if (tok_arg == nullptr && !serving) {
        std::fprintf(stderr, "\nhy4_flash: the weights load. Add --tokens a,b,c (or --tokens-file FILE, which\n"
                             "         a long context needs: argv cannot hold 32k ids) for a prefill.\n");
        return 2;
    }
    std::vector<int32_t> tokens;
    for (const char* p = tok_arg == nullptr ? "" : tok_arg; *p != '\0';) {
        char* end = nullptr;
        const long v = std::strtol(p, &end, 10);
        if (end == p) break;
        tokens.push_back((int32_t) v);
        p = *end == ',' ? end + 1 : end;
    }
    if (tokens.empty() && !serving) {
        std::fprintf(stderr, "\nhy4_flash: --tokens needs a comma-separated list of ids\n");
        return 2;
    }

    int max_new = 1;
    if (const char* v = flag_value(argc, argv, "--max-new")) max_new = std::atoi(v);
    if (max_new < 1) max_new = 1;
    const int64_t vocab = g.vocab_size;
    const int n_blocks = (int) g.layers.size() - g.n_mtp_blocks;
    int n_moe = 0;
    for (int i = 0; i < n_blocks; ++i)
        if (g.layers[(std::size_t) i].expert_ordinal >= 0) ++n_moe;

    // Keep the latent KV cache across calls to continue the same sequence.
    int reserve = (int) tokens.size() + max_new;
    if (serving) {
        reserve = 4096;
        if (const char* v = flag_value(argc, argv, "--context")) reserve = std::max(64, std::atoi(v));
    }
    if (const char* e = std::getenv("HY4_FLASH_DIRECT_CONTEXT")) reserve = std::max(reserve, std::atoi(e));
    KvCache cache;
    if (!kv_cache_init(cache, n_blocks, g.kv_lora_rank, g.rope_dim, reserve, err)) {
        std::fprintf(stderr, "\nhy4_flash: %s\n", err.c_str());
        return 2;
    }
    std::printf("  kv cache       %d positions x %d blocks x %d latent = %.1f MiB%s\n", reserve, n_blocks,
                g.kv_lora_rank + g.rope_dim,
                (double) kv_cache_bytes(n_blocks, g.kv_lora_rank, g.rope_dim, reserve, kv_format(cache)) / (1024.0 * 1024.0),
                kv_format(cache) == kKvQ8 ? " (Q8_0 latent, F16 rope)" : kv_format(cache) == kKvF16 ? " (F16)" : " (F32)");

    // Use the reference VDR=2 dot grouping only for this model's Q6_K tensors.
    kernels::native_mmvq_set_q6_c50(true);
    // Select the reference multi-column layout for compatible projections.
    kernels::native_mmvq_set_multi_exact(false);
    // Set the row grouping for small-K Q4_K expert projections.
    kernels::native_mmvq_set_rows(2);

    // Optionally log selected expert IDs per step and block.
    struct RouteDump { std::FILE* f; int step; };
    static RouteDump rd{nullptr, 0};
    if (const char* p = std::getenv("HY4_FLASH_ROUTING_DUMP")) {
        rd.f = std::fopen(p, "w");
        if (rd.f == nullptr) { std::fprintf(stderr, "\nhy4_flash: cannot write %s\n", p); return 2; }
        fo.on_routing = [](int block, const int32_t* ids, void*) {
            int host[64];
            const int k = 16;
            if (cudaMemcpy(host, ids, (std::size_t) k * 4, cudaMemcpyDeviceToHost) != cudaSuccess) return;
            std::fprintf(rd.f, "%d %d", rd.step, block);
            for (int i = 0; i < k; ++i) std::fprintf(rd.f, " %d", host[i]);
            std::fputc('\n', rd.f);
        };
    }
    if (const char* v = std::getenv("STRATA_POISON_SCRATCH")) fo.poison_scratch = std::atoi(v) != 0;
    fo.cache = &cache;
    // Held here, not inside the pass, so it survives from one decode step to the next - which is the whole
    // of what it buys. The first step fills it; every step after reads it.
    DenseResidency dense;
    fo.dense = &dense;
    // ---- THE SAMPLER. Greedy by default, which is exactly what this engine did before it existed.
    SamplerConfig sc{};
    {
        std::string serr;
        if (!sampler_config_from_args(argc, argv, sc, serr)) {
            std::fprintf(stderr, "hy4_flash: %s\n", serr.c_str());
            return 2;
        }
    }
    Sampler sampler(sc);
    std::printf("  sampler        %s\n", sampler_describe(sc).c_str());

    // Resolve stop tokens from model metadata and command-line overrides.
    std::vector<int32_t> stop_ids = g.stop_tokens;
    if (has_flag(argc, argv, "--ignore-eos")) stop_ids.clear();
    if (const char* v = flag_value(argc, argv, "--stop-ids")) {
        stop_ids.clear();
        for (const char* p = v; *p != '\0';) {
            char* end = nullptr;
            const long id = std::strtol(p, &end, 10);
            if (end == p) break;
            if (id >= 0) stop_ids.push_back((int32_t) id);
            p = (*end == ',') ? end + 1 : end;
        }
    }
    const auto is_stop = [&stop_ids](int64_t id) {
        for (int32_t s : stop_ids)
            if ((int64_t) s == id) return true;
        return false;
    };
    if (stop_ids.empty()) {
        std::printf("  stop           none - generation runs to --max-new whatever the model says\n");
    } else {
        std::printf("  stop           ");
        for (std::size_t i = 0; i < stop_ids.size(); ++i)
            std::printf("%s%d", i == 0 ? "" : ", ", stop_ids[i]);
        std::printf(" (from the file's tokenizer metadata)\n");
    }
    // Measure teacher-forced NLL using the previous row's logits for each target token.
    {
        const auto start = std::chrono::steady_clock::now();
        if (serving) {
            std::printf("INFO experts=%lld\n", (long long)n_moe * g.n_experts);
            fo.on_expert_tier = [](int slots, std::size_t bytes) {
                std::printf("INFO expert_slots=%d expert_cache_mib=%zu\n", slots, bytes >> 20);
                std::fflush(stdout);
            };
            // Full-host mode allocates its GPU tier on the first forward.
            if (!fo.resident_experts) fo.on_expert_tier(0, 0);
        }
        if (!prepare_experts(path, g, fo, err)) {
            std::fprintf(stderr, "hy4_flash: expert residency initialization: %s\n", err.c_str());
            return 2;
        }
        const std::chrono::duration<double> elapsed = std::chrono::steady_clock::now() - start;
        std::fprintf(stderr, "HY4_FLASH_EXPERT_HOST init_ms=%.1f\n", elapsed.count() * 1000);
    }
    if (const char* output = flag_value(argc, argv, "--dump-logits")) {
        const char* value = flag_value(argc, argv, "--prefix-tokens");
        size_t prefix = 0;
        if (value) {
            const char* end = value + std::strlen(value);
            const auto parsed = std::from_chars(value, end, prefix);
            if (parsed.ec != std::errc{} || parsed.ptr != end) prefix = 0;
        }
        if (!prefix || prefix >= tokens.size()) {
            std::fprintf(stderr, "--dump-logits requires --prefix-tokens in [1, token count - 1]\n");
            return 2;
        }
        std::ofstream rows(output, std::ios::binary);
        if (!rows) {
            std::fprintf(stderr, "cannot open %s for writing logits\n", output);
            return 2;
        }
        fo.history = &tokens;
        fo.want_logits = fo.logits_last_only = true;
        for (size_t at = 0; at < tokens.size() - 1;) {
            const size_t take = at == 0 ? prefix : 1;
            const std::vector<int32_t> part(tokens.begin() + at, tokens.begin() + at + take);
            std::vector<float> logits;
            if (!forward_prefill(path, g, pack, part, logits, err, fo) || logits.size() != size_t(vocab)) {
                std::fprintf(stderr, "--dump-logits failed at token %zu: %s\n", at, err.c_str());
                return 2;
            }
            if (!rows.write(reinterpret_cast<const char*>(logits.data()), logits.size() * sizeof(float))) {
                std::fprintf(stderr, "cannot write logits to %s\n", output);
                return 2;
            }
            at += take;
        }
        rows.close();
        return rows ? 0 : 2;
    }
    if (has_flag(argc, argv, "--ppl")) {
        const int c = prefill_chunk_max();
        std::printf("  ppl            teacher-forced NLL over %zu tokens, chunks of %d\n", tokens.size(), c);
        std::fflush(stdout);
        double nll = 0.0;
        long scored = 0;
        double chunk_nll = 0.0;
        long chunk_scored = 0;
        std::size_t at = 0;
        while (at < tokens.size()) {
            const std::size_t take = std::min((std::size_t) c, tokens.size() - at);
            const std::vector<int32_t> part(tokens.begin() + (long) at, tokens.begin() + (long) (at + take));
            std::vector<float> lg;
            std::string perr;
            ForwardOptions co = fo;
            co.history = &tokens;
            co.want_logits = true;
            co.logits_last_only = false;
            if (!forward_prefill(path, g, pack, part, lg, perr, co)) {
                std::fprintf(stderr, "hy4_flash: --ppl at token %zu: %s\n", at, perr.c_str());
                return 2;
            }
            if (lg.size() < take * (std::size_t) vocab) {
                std::fprintf(stderr, "hy4_flash: --ppl needs every row's logits; got %zu of %zu floats\n",
                             lg.size(), take * (std::size_t) vocab);
                return 2;
            }
            for (std::size_t i = 0; i < take; ++i) {
                const std::size_t next = at + i + 1;
                if (next >= tokens.size()) break;   // the last token has nothing to predict
                const float* row = lg.data() + i * (std::size_t) vocab;
                double mx = row[0];
                for (int64_t v = 1; v < vocab; ++v) mx = row[v] > mx ? row[v] : mx;
                double sum = 0.0;
                for (int64_t v = 0; v < vocab; ++v) sum += std::exp((double) row[v] - mx);
                nll += -((double) row[tokens[next]] - mx - std::log(sum));
                ++scored;
            }
            at += take;
            // Report per-chunk NLL separately from the running mean.
            std::printf("                 %zu/%zu, chunk NLL %.6f, running NLL %.6f\n", at, tokens.size(),
                        scored > chunk_scored ? (nll - chunk_nll) / (double) (scored - chunk_scored) : 0.0,
                        scored > 0 ? nll / (double) scored : 0.0);
            chunk_nll = nll;
            chunk_scored = scored;
            std::fflush(stdout);
        }
        if (scored == 0) {
            std::fprintf(stderr, "hy4_flash: --ppl scored nothing; --tokens needs at least two ids\n");
            return 2;
        }
        std::printf("NLL %.6f PPL %.6f over %ld positions\n", nll / (double) scored,
                    std::exp(nll / (double) scored), scored);
        return 0;
    }

    // ---- `--dfly-check DRAFT.gguf`: the drafter on a real prefix (see `dfly_check` above). Opt-in; returns.
    if (const char* dp = flag_value(argc, argv, "--dfly-check"))
        return dfly_check(argc, argv, dp, path, g, pack, tokens, fo, dense, cache, reserve);

    // Compare a W-token verify pass against teacher-forced single-token decode. Rewind
    // to the same indexer group boundary for each repetition and rotate widths to limit
    // ordering effects. Verify timing includes the copy of all logits rows to the host.
    if (const char* vb = flag_value(argc, argv, "--verify-bench")) {
        std::vector<int> widths;
        for (const char* p = vb; *p != '\0';) {
            char* end = nullptr;
            const long w = std::strtol(p, &end, 10);
            if (end == p) break;
            if (w >= 1 && w <= kGemmT) widths.push_back((int) w);
            p = *end == ',' ? end + 1 : end;
        }
        if (widths.empty()) {
            std::fprintf(stderr, "hy4_flash: --verify-bench needs widths in 1..%d, e.g. 1,2,3,4,5,8\n", kGemmT);
            return 2;
        }
        int reps = 24;
        if (const char* v = flag_value(argc, argv, "--verify-reps")) reps = std::max(1, std::atoi(v));
        const int ar_steps = 48, ar_warm = 8;
        const int max_w = *std::max_element(widths.begin(), widths.end());
        const int kp = g.idx_kpool > 0 ? g.idx_kpool : 1;
        const int need_after = std::max(max_w, ar_warm + ar_steps);
        int depth = ((int) tokens.size() - need_after) / kp * kp;
        if (const char* v = flag_value(argc, argv, "--verify-depth")) depth = std::atoi(v) / kp * kp;
        if (depth < kp || depth + need_after > (int) tokens.size() || depth + need_after > reserve) {
            std::fprintf(stderr,
                         "hy4_flash: --verify-bench needs --tokens of at least depth + %d ids and a cache of as many "
                         "positions (depth %d, %zu ids, %d reserved)\n",
                         need_after, depth, tokens.size(), reserve);
            return 2;
        }
        const auto run = [&](int at, int n, bool want, std::vector<float>& out) -> bool {
            const std::vector<int32_t> part(tokens.begin() + at, tokens.begin() + at + n);
            ForwardOptions co = fo;
            co.history = &tokens;
            co.want_logits = want;
            co.logits_last_only = false;
            if (!forward_prefill(path, g, pack, part, out, err, co)) {
                std::fprintf(stderr, "\nhy4_flash: --verify-bench at %d (+%d): %s\n", at, n, err.c_str());
                return false;
            }
            return true;
        };
        const auto ms_since = [](std::chrono::steady_clock::time_point t0) {
            return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        };
        const auto summary = [](std::vector<double> v) {
            std::sort(v.begin(), v.end());
            const std::size_t n = v.size();
            struct S { double p10, med, p90; };
            return S{v[n / 10], v[n / 2], v[std::min(n - 1, n * 9 / 10)]};
        };
        const int chunk = prefill_chunk_max();
        std::vector<float> lg;
        cache.len = 0;
        // Check actual residency after the first prefill chunk allocates the tier. Partial
        // residency changes the expert execution paths and requires explicit opt-in for
        // this benchmark. Report residency alongside the measurements.
        const bool allow_partial = has_flag(argc, argv, "--verify-allow-partial");
        bool resident = false;
        for (int at = 0; at < depth; at += chunk) {
            if (!run(at, std::min(chunk, depth - at), false, lg)) return 2;
            if (at != 0) continue;
            resident = dense_experts_complete(dense);
            if (!resident && !allow_partial) {
                std::size_t vfree = 0, vtotal = 0;
                cudaMemGetInfo(&vfree, &vtotal);
                std::fprintf(stderr,
                             "\nhy4_flash: --verify-bench needs the expert tier COMPLETE and it is not "
                             "(%zu MiB free of %zu). The tier is filled from the VRAM free at the first "
                             "pass, so raise --expert-vram, or wait for the card: with a partial tier the "
                             "prompt and decode take other expert paths than a W-token verify can, and "
                             "both the cost ratio and the row comparison stop answering what they say "
                             "they do. Pass --verify-allow-partial to measure that configuration on "
                             "purpose.\n",
                             vfree >> 20, vtotal >> 20);
                return 2;
            }
        }

        // Reject a verification width if its target logits fail the exactness check.
        if (has_flag(argc, argv, "--verify-require-exact")) {
            if (!resident) {
                std::fprintf(stderr, "hy4_flash: --verify-require-exact needs the expert tier COMPLETE\n");
                return 2;
            }
            if (kv_format(cache) == kKvF32) { std::fprintf(stderr, "hy4_flash: --verify-require-exact needs F16 or Q8_0 KV\n"); return 2; }
            std::vector<int> offs{0, 1, 2, 3};
            if (const char* v = flag_value(argc, argv, "--verify-offsets")) {
                offs.clear();
                for (const char* q = v; *q != '\0';) {
                    char* end = nullptr;
                    const long o = std::strtol(q, &end, 10);
                    if (end == q) break;
                    if (o >= 0 && o < 64) offs.push_back((int) o);
                    q = *end == ',' ? end + 1 : end;
                }
            }
            const int n_blk = (int) g.layers.size() - g.n_mtp_blocks;
            DflyCapture cap;
            for (int b = 0; b < n_blk; ++b)
                if (g.layers[(std::size_t) b].attention != Attention::Sliding) cap.sel_blocks.push_back(b);
            std::string e2;
            if (!cap.init(g, DflyDrafter::kMaxK + 1, e2)) { std::fprintf(stderr, "hy4_flash: %s\n", e2.c_str()); return 2; }
            struct RowCap { std::vector<unsigned char> logits, hidden, sel; };
            const auto finite = [](const std::vector<unsigned char>& v) {
                const float* f = (const float*) v.data();
                for (std::size_t i = 0; i < v.size() / 4; ++i) if (!std::isfinite(f[i])) return false;
                return true;
            };
            const auto grab = [&](int at, int n, bool exact, std::vector<RowCap>& rows) -> bool {
                const std::vector<int32_t> part(tokens.begin() + at, tokens.begin() + at + n);
                ForwardOptions co = fo;
                co.history = &tokens;
                co.want_logits = true;
                co.logits_last_only = false;
                co.dfly = &cap;
                cap.mode = DflyCapture::Mode::verify_window;
                co.exact_columns = exact && !has_flag(argc, argv, "--verify-negative-control");
                std::vector<float> lg;
                if (!forward_prefill(path, g, pack, part, lg, err, co) || !cap.valid) {
                    std::fprintf(stderr, "\nhy4_flash: --verify-require-exact at %d (+%d): %s\n", at, n, err.c_str());
                    return false;
                }
                for (int r = 0; r < n; ++r) {
                    RowCap rc;
                    rc.logits.assign((const unsigned char*) (lg.data() + (std::size_t) r * vocab),
                                     (const unsigned char*) (lg.data() + (std::size_t) (r + 1) * vocab));
                    rc.hidden.resize((std::size_t) g.hidden_dim * 4);
                    cudaMemcpy(rc.hidden.data(), (const float*) cap.hidden + (std::size_t) r * g.hidden_dim,
                               rc.hidden.size(), cudaMemcpyDeviceToHost);
                    if (cap.engaged)
                        for (std::size_t i = 0; i < cap.sel.size(); ++i) {
                            std::vector<unsigned char> t((std::size_t) cap.n_sel * 4);
                            cudaMemcpy(t.data(), (const int32_t*) cap.sel[i] + (std::size_t) r * cap.n_sel, t.size(),
                                       cudaMemcpyDeviceToHost);
                            rc.sel.insert(rc.sel.end(), t.begin(), t.end());
                        }
                    rows.push_back(std::move(rc));
                }
                return true;
            };
            const auto kv_raw = [&](int lo, int hi) {
                std::vector<unsigned char> v;
                for (std::size_t b = 0; b < cache.kv_cmpr.size(); ++b)
                    for (int part = 0; part < 2; ++part) {
                        const auto row_bytes = part == 0 ? kv_cache_lat_row_bytes(cache.kv_lora, kv_format(cache))
                                             : kv_cache_pe_row_bytes(cache.n_rot, kv_format(cache));
                        const std::size_t bytes = (std::size_t) (hi - lo) * row_bytes;
                        const std::size_t at = v.size();
                        v.resize(at + bytes);
                        const unsigned char* base = (const unsigned char*) (part == 0 ? cache.kv_cmpr[b] : cache.k_pe[b]);
                        cudaMemcpy(v.data() + at, base + (std::size_t) lo * row_bytes, bytes, cudaMemcpyDeviceToHost);
                    }
                return v;
            };
            const auto snap = [&](std::vector<unsigned char>& out) {
                std::string e3;
                if (!dfly_indexer_snapshot(dense, out, e3)) { std::fprintf(stderr, "hy4_flash: %s\n", e3.c_str()); return false; }
                return true;
            };
            long fails = 0, checks = 0;
            const auto check = [&](bool ok, const char* what, int w, int o, int m) {
                ++checks;
                if (!ok) { ++fails; std::printf("VX_FAIL %s W=%d offset=%d m=%d\n", what, w, o, m); }
            };
            std::printf("VX_CONF depth=%d graphs=eligible(exact-candidate) multi_exact=%d experts_resident=1 sel_blocks=%zu\n", depth,
                        kernels::native_mmvq_multi_exact() ? 1 : 0, cap.sel_blocks.size());
            std::vector<RowCap> sink;
            for (int W2 : widths) {
                if (W2 < 1 || W2 > DflyDrafter::kMaxK + 1) continue;
                for (int o : offs) {
                    const int at0 = depth + o;
                    if (at0 + W2 > (int) tokens.size() || at0 + W2 > reserve) continue;
                    // A window that straddles the engaged boundary must be REFUSED by the pass, not run.
                    if (indexer_plan(g.idx_top_k, g.idx_kpool, g.idx_select_tail, at0, 1).engaged !=
                        indexer_plan(g.idx_top_k, g.idx_kpool, g.idx_select_tail, at0, W2).engaged) {
                        std::vector<RowCap> none;
                        cache.len = at0;
                        const bool ran = grab(at0, W2, true, none);
                        check(!ran, "boundary-not-refused", W2, o, -1);
                        std::printf("VX W=%d offset=%d refused-at-boundary\n", W2, o);
                        cache.len = depth;
                        continue;
                    }
                    const auto start = [&]() -> bool {
                        // The indexer rewinds lazily, inside the next pass, so a snapshot taken right after
                        // `cache.len = depth` would still hold the previous run's counters. Re-feed the last
                        // aligned group as decode steps so the state is settled and the same for every run.
                        cache.len = depth - kp;
                        for (int j = -kp; j < o; ++j) { sink.clear(); if (!grab(depth + j, 1, false, sink)) return false; }
                        return true;
                    };
                    // reference: W decode steps, the indexer state after each
                    std::vector<RowCap> ra;
                    std::vector<std::vector<unsigned char>> rs((std::size_t) W2 + 1);
                    if (!start() || !snap(rs[0])) return 2;
                    for (int j = 0; j < W2; ++j)
                        if (!grab(at0 + j, 1, false, ra) || !snap(rs[(std::size_t) j + 1])) return 2;
                    const auto ref_kv = kv_raw(depth, at0 + W2);
                    // candidate: one exact pass
                    std::vector<RowCap> rc;
                    std::vector<unsigned char> cs;
                    if (!start() || !grab(at0, W2, true, rc) || !snap(cs)) return 2;
                    for (int j = 0; j < W2; ++j) {
                        check(finite(rc[(std::size_t) j].logits) && finite(ra[(std::size_t) j].logits) &&
                                  rc[(std::size_t) j].logits == ra[(std::size_t) j].logits, "logits", W2, o, j);
                        check(finite(rc[(std::size_t) j].hidden) && rc[(std::size_t) j].hidden == ra[(std::size_t) j].hidden,
                              "hidden", W2, o, j);
                        check(rc[(std::size_t) j].sel == ra[(std::size_t) j].sel, "selection", W2, o, j);
                    }
                    check(kv_raw(depth, at0 + W2) == ref_kv, "kv", W2, o, -1);
                    check(cs == rs[(std::size_t) W2], "indexer", W2, o, -1);
                    // window rewind at every accepted length, then one more decode step from there
                    for (int m = 0; m <= W2; ++m) {
                        rc.clear();
                        std::string e4;
                        if (!start() || !grab(at0, W2, true, rc)) return 2;
                        if (!dfly_rewind(dense, cache, cap, at0 + m, e4)) {
                            check(false, "rewind-call", W2, o, m);
                            std::printf("VX_ERR %s\n", e4.c_str());
                            continue;
                        }
                        std::vector<unsigned char> after;
                        if (!snap(after)) return 2;
                        check(after == rs[(std::size_t) m] && cache.len == at0 + m, "rewind-state", W2, o, m);
                        if (m < W2) {
                            sink.clear();
                            if (!grab(at0 + m, 1, false, sink)) return 2;
                            check(sink[0].logits == ra[(std::size_t) m].logits && sink[0].hidden == ra[(std::size_t) m].hidden,
                                  "rewind-continue", W2, o, m);
                        }
                    }
                    std::printf("VX W=%d offset=%d done\n", W2, o);
                    std::fflush(stdout);
                }
            }
            // ---- T7: a pass that dies half way, then the forced reset, then a brand-new request.
            {
                const auto settle = [&](std::vector<RowCap>& out) -> bool {
                    cache.len = depth - kp;
                    for (int j = -kp; j < 0; ++j) { sink.clear(); if (!grab(depth + j, 1, false, sink)) return false; }
                    return grab(depth, 1, false, out);
                };
                std::vector<RowCap> ref;
                if (!settle(ref)) return 2;
                // the failing pass: the hook stops it at block 20, after blocks 0..19 wrote KV and indexer rows
                {
                    ForwardOptions co = fo;
                    co.history = &tokens;
                    co.want_logits = true;
                    co.logits_last_only = false;
                    co.dfly = &cap;
                    co.exact_columns = true;
                    cap.mode = DflyCapture::Mode::verify_window;
                    co.on_block = +[](int b, const float*, void*) -> bool { return b < 20; };
                    cache.len = depth;
                    const std::vector<int32_t> part(tokens.begin() + depth, tokens.begin() + depth + 2);
                    std::vector<float> lg;
                    const bool ran = forward_prefill(path, g, pack, part, lg, err, co);
                    check(!ran && !cap.valid, "fault-injected-pass-fails", 2, 0, -1);
                }
                std::string e5;
                const DflyResetResult rr = dfly_abort_reset(dense, cache, e5);
                check(rr == DflyResetResult::clean && cache.len == 0, "reset-clean", 0, 0, -1);
                std::vector<unsigned char> z;
                if (!snap(z)) return 2;
                bool zero = z.size() == 5 * sizeof(int32_t);
                for (unsigned char c : z) zero = zero && c == 0;
                check(zero, "reset-state-zero", 0, 0, -1);
                // a new request: prefill the same prefix in chunks, then the same decode step
                cache.len = 0;
                std::vector<float> lg0;
                bool ok = true;
                for (int at = 0; ok && at < depth; at += chunk) ok = run(at, std::min(chunk, depth - at), false, lg0);
                std::vector<RowCap> again;
                ok = ok && settle(again);
                check(ok && again.size() == 1 && again[0].logits == ref[0].logits && again[0].hidden == ref[0].hidden,
                      "after-reset-same-as-before", 0, 0, -1);
                std::printf("VX T7 done\n");
            }
            // T8: return to earlier widths and both sides of the engagement boundary. At depth 2048
            // this exercises all eight graph keys in both directions, after warmup and capture.
            if (depth >= 2 * kp) {
                const int sequence[] = {4, 1, 3, 2, 2, 3, 1, 4};
                for (int cycle = 0; cycle < 3; ++cycle) for (int i = 0; i < 8; ++i) {
                    const int w = sequence[i], at = depth - (i % 2 == 0 ? kp : 0);
                    if (at + w > (int)tokens.size() || at + w > reserve) continue;
                    if (indexer_plan(g.idx_top_k, kp, g.idx_select_tail, at, 1).engaged !=
                        indexer_plan(g.idx_top_k, kp, g.idx_select_tail, at, w).engaged) continue;
                    const auto restart = [&]() {
                        cache.len = depth - 2 * kp;
                        for (int j = depth - 2 * kp; j < at; ++j) {
                            sink.clear();
                            if (!grab(j, 1, false, sink)) return false;
                        }
                        return true;
                    };
                    std::vector<RowCap> ref, got;
                    std::vector<unsigned char> ref_idx, got_idx;
                    if (!restart()) return 2;
                    for (int j = 0; j < w; ++j) if (!grab(at + j, 1, false, ref)) return 2;
                    if (!snap(ref_idx)) return 2;
                    const auto ref_kv = kv_raw(at, at + w);
                    if (!restart() || !grab(at, w, true, got) || !snap(got_idx)) return 2;
                    bool same = got.size() == ref.size();
                    for (std::size_t j = 0; same && j < ref.size(); ++j)
                        same = finite(got[j].logits) && finite(got[j].hidden) &&
                               got[j].logits == ref[j].logits && got[j].hidden == ref[j].hidden &&
                               got[j].sel == ref[j].sel;
                    check(same && got_idx == ref_idx && kv_raw(at, at + w) == ref_kv,
                          "graph-key-switch", w, at - depth, cycle);
                }
                std::printf("VX T8 graph-key-switch done\n");
            }
            std::printf("VX_RESULT %s checks=%ld fails=%ld\n", fails == 0 ? "PASS" : "FAIL", checks, fails);
            return fails == 0 ? 0 : 1;
        }

        // Use the same tokens and weights to compare eager and graph execution.
        if (const char* pv = flag_value(argc, argv, "--verify-probe")) {
            const int W = std::atoi(pv);
            // Independent of the `--verify-bench` width list: the probe makes its own reference rows, and
            // coupling the two only means "the tool rejected me for a width it could have handled".
            if (W < 1 || W > kGemmT || depth + W > (int) tokens.size() || depth + W > reserve) {
                std::fprintf(stderr,
                             "hy4_flash: --verify-probe takes 1..%d and needs depth + W ids and cache "
                             "positions (W %d, depth %d, %zu ids, %d reserved)\n",
                             kGemmT, W, depth, tokens.size(), reserve);
                return 2;
            }
            // Read the per-block head count for sliding and sparse attention.
            std::vector<std::size_t> attn_w((std::size_t) g.layers.size(), 0);
            for (std::size_t b = 0; b < g.layers.size(); ++b)
                attn_w[b] = (std::size_t) g.v_head_dim * (std::size_t) g.layers[b].n_heads;
            struct Probe {
                int n_tok = 1, top_k = 0;
                std::size_t width = 0;                              // floats of state per token
                const std::vector<std::size_t>* attn_w = nullptr;   // block -> floats of attention output per token
                std::map<int, std::vector<float>> state;            // block -> [width * n_tok]
                std::map<int, std::vector<float>> attn;             // block -> [attn_width * n_tok]
                std::map<int, std::vector<int32_t>> ids;            // block -> [top_k * n_tok]
                std::string err;
            };
            const auto on_block = +[](int block, const float* state, void* user) -> bool {
                auto* p = static_cast<Probe*>(user);
                std::vector<float>& dst = p->state[block];
                dst.resize(p->width * (std::size_t) p->n_tok);
                if (cudaMemcpy(dst.data(), state, dst.size() * sizeof(float), cudaMemcpyDeviceToHost) !=
                    cudaSuccess) {
                    cudaGetLastError();
                    p->err = "block " + std::to_string(block) + ": state copy failed";
                    return false;
                }
                return true;
            };
            const auto on_attn = +[](int block, const float* kqv, void* user) -> void {
                auto* p = static_cast<Probe*>(user);
                if (p->attn_w == nullptr || (std::size_t) block >= p->attn_w->size()) return;
                std::vector<float>& dst = p->attn[block];
                dst.resize((*p->attn_w)[(std::size_t) block] * (std::size_t) p->n_tok);
                if (cudaMemcpy(dst.data(), kqv, dst.size() * sizeof(float), cudaMemcpyDeviceToHost) !=
                    cudaSuccess) {
                    cudaGetLastError();
                    dst.clear();
                }
            };
            const auto on_routing = +[](int block, const int32_t* ids, void* user) -> void {
                auto* p = static_cast<Probe*>(user);
                std::vector<int32_t>& dst = p->ids[block];
                dst.resize((std::size_t) p->top_k * (std::size_t) p->n_tok);
                if (cudaMemcpy(dst.data(), ids, dst.size() * sizeof(int32_t), cudaMemcpyDeviceToHost) !=
                    cudaSuccess) {
                    cudaGetLastError();
                    dst.clear();
                }
            };
            const auto probe_run = [&](int at, int n, Probe* pr, std::vector<float>& out) -> bool {
                const std::vector<int32_t> part(tokens.begin() + at, tokens.begin() + at + n);
                ForwardOptions co = fo;
                co.history = &tokens;
                co.want_logits = true;
                co.logits_last_only = false;
                if (pr != nullptr) {
                    pr->n_tok = n;
                    pr->width = (std::size_t) g.hc_width();
                    pr->attn_w = &attn_w;
                    pr->top_k = g.top_k;
                    pr->state.clear();
                    pr->attn.clear();
                    pr->ids.clear();
                    co.on_block = on_block;
                    co.on_attn = on_attn;
                    co.on_routing = on_routing;
                    co.user = pr;
                }
                if (!forward_prefill(path, g, pack, part, out, err, co)) {
                    std::fprintf(stderr, "\nhy4_flash: --verify-probe at %d (+%d): %s%s\n", at, n, err.c_str(),
                                 pr != nullptr && !pr->err.empty() ? (" / " + pr->err).c_str() : "");
                    return false;
                }
                return true;
            };
            // A row of logits, or a block's state slice for one token, against the same thing from another
            // pass: the largest absolute gap and how many floats are not identical.
            const auto cmp = [](const float* a, const float* b, std::size_t n) {
                struct D { double max_abs; long ndiff; };
                D d{0.0, 0};
                for (std::size_t i = 0; i < n; ++i) {
                    const double g2 = std::fabs((double) a[i] - (double) b[i]);
                    if (g2 > d.max_abs) d.max_abs = g2;
                    if (a[i] != b[i]) ++d.ndiff;
                }
                return d;
            };
            const auto env_on = [](const char* name, bool dflt) {
                const char* v = std::getenv(name);
                return v == nullptr ? dflt : std::atoi(v) != 0;
            };
            std::printf("VP_CONF depth=%d W=%d per_query=%d multi_exact=%d graphs=%d experts_resident=%d "
                        "width=%d top_k=%d\n",
                        depth, W, env_on("STRATA_HY4_FLASH_VERIFY_PER_QUERY", false) ? 1 : 0,
                        kernels::native_mmvq_multi_exact() ? 1 : 0, env_on("STRATA_HY4_FLASH_GRAPHS", true) ? 1 : 0,
                        resident ? 1 : 0, g.hc_width(), g.top_k);

            // A: eager single-token reference, every block kept.
            std::vector<Probe> pa((std::size_t) W);
            std::vector<std::vector<float>> la((std::size_t) W);
            for (int j = 0; j < W; ++j)
                if (!probe_run(depth + j, 1, &pa[(std::size_t) j], la[(std::size_t) j])) return 2;
            // B: the same steps with no hooks, so the first captures a graph and the rest replay it.
            cache.len = depth;
            std::vector<std::vector<float>> lb((std::size_t) W);
            for (int j = 0; j < W; ++j)
                if (!probe_run(depth + j, 1, nullptr, lb[(std::size_t) j])) return 2;
            for (int j = 0; j < W; ++j) {
                const auto d = cmp(la[(std::size_t) j].data(), lb[(std::size_t) j].data(), (std::size_t) vocab);
                std::printf("VPGRAPH row=%d eager_vs_graph_max_abs=%.3g ndiff=%ld\n", j, d.max_abs, d.ndiff);
            }
            // The KV rows the pass itself wrote, which split a block in two: `mla_project` (and the FP16
            // append) on one side, the attention and the FFN on the other. The residual state after a
            // block does not say which, and the two want different fixes.
            const auto half_to_float = [](std::uint16_t h) {
                const std::uint32_t sign = (std::uint32_t) (h & 0x8000u) << 16;
                const std::uint32_t exp = (h >> 10) & 0x1fu, man = h & 0x3ffu;
                std::uint32_t bits;
                if (exp == 0) {
                    if (man == 0) bits = sign;                           // +-0
                    else {                                               // subnormal: renormalise
                        std::uint32_t e = 127 - 15 + 1, m = man;
                        while ((m & 0x400u) == 0) { m <<= 1; --e; }
                        bits = sign | (e << 23) | ((m & 0x3ffu) << 13);
                    }
                } else if (exp == 0x1fu) {
                    bits = sign | 0x7f800000u | (man << 13);             // inf / NaN
                } else {
                    bits = sign | ((exp + 127 - 15) << 23) | (man << 13);
                }
                float f;
                std::memcpy(&f, &bits, sizeof f);
                return f;
            };
            const auto kv_snapshot = [&](std::vector<std::vector<float>>& out) {
                out.assign(cache.kv_cmpr.size(), {});
                const std::size_t row = (std::size_t) cache.kv_lora + (std::size_t) cache.n_rot;
                std::vector<std::uint16_t> h((std::size_t) cache.kv_lora), hp((std::size_t) cache.n_rot);
                for (std::size_t b = 0; b < cache.kv_cmpr.size(); ++b) {
                    out[b].assign(row * (std::size_t) W, 0.f);
                    for (int j = 0; j < W; ++j) {
                        const std::size_t at = (std::size_t) (depth + j);
                        cudaMemcpy(h.data(),
                                   (const std::uint16_t*) cache.kv_cmpr[b] + at * (std::size_t) cache.kv_lora,
                                   h.size() * sizeof(std::uint16_t), cudaMemcpyDeviceToHost);
                        cudaMemcpy(hp.data(),
                                   (const std::uint16_t*) cache.k_pe[b] + at * (std::size_t) cache.n_rot,
                                   hp.size() * sizeof(std::uint16_t), cudaMemcpyDeviceToHost);
                        for (std::size_t i = 0; i < h.size(); ++i)
                            out[b][(std::size_t) j * row + i] = half_to_float(h[i]);
                        for (std::size_t i = 0; i < hp.size(); ++i)
                            out[b][(std::size_t) j * row + h.size() + i] = half_to_float(hp[i]);
                    }
                }
            };
            std::vector<std::vector<float>> kv_a, kv_d;
            if (kv_format(cache) == kKvF16) kv_snapshot(kv_a);

            // C: one W-token pass over the same tokens.
            cache.len = depth;
            Probe pc;
            std::vector<float> lc;
            if (!probe_run(depth, W, &pc, lc)) return 2;
            cache.len = depth;

            // Feed reference block states to isolate errors introduced within each block.
            Probe pd;
            std::vector<float> ld;
            std::map<int, float*> teach;                   // block -> device [width, W], A's rows side by side
            {
                std::vector<float> host;
                for (const auto& kv : pa[0].state) {
                    const int b = kv.first;
                    host.assign(pc.width * (std::size_t) W, 0.f);
                    bool have = true;
                    for (int j = 0; j < W && have; ++j) {
                        const auto it = pa[(std::size_t) j].state.find(b);
                        have = it != pa[(std::size_t) j].state.end() && it->second.size() == pc.width;
                        if (have)
                            std::copy(it->second.begin(), it->second.end(),
                                      host.begin() + (std::size_t) j * pc.width);
                    }
                    if (!have) continue;
                    float* dev = nullptr;
                    if (cudaMalloc(&dev, host.size() * sizeof(float)) != cudaSuccess ||
                        cudaMemcpy(dev, host.data(), host.size() * sizeof(float), cudaMemcpyHostToDevice) !=
                            cudaSuccess) {
                        cudaGetLastError();
                        std::fprintf(stderr, "hy4_flash: --verify-probe: no room for the teacher state\n");
                        return 2;
                    }
                    teach[b] = dev;
                }
            }
            {
                static std::map<int, float*>* g_teach = nullptr;
                g_teach = &teach;
                const std::vector<int32_t> part(tokens.begin() + depth, tokens.begin() + depth + W);
                ForwardOptions co = fo;
                co.history = &tokens;
                co.want_logits = true;
                co.logits_last_only = false;
                pd.n_tok = W;
                pd.width = (std::size_t) g.hc_width();
                pd.attn_w = &attn_w;
                pd.top_k = g.top_k;
                co.on_block = on_block;
                co.on_attn = on_attn;
                co.user = &pd;
                co.teacher_state = +[](int block, void*) -> const float* {
                    const auto it = g_teach->find(block);
                    return it == g_teach->end() ? nullptr : it->second;
                };
                if (!forward_prefill(path, g, pack, part, ld, err, co)) {
                    std::fprintf(stderr, "\nhy4_flash: --verify-probe (teacher) at %d (+%d): %s\n", depth, W,
                                 err.c_str());
                    return 2;
                }
                cache.len = depth;
            }
            for (const auto& kv : teach) cudaFree(kv.second);
            if (kv_format(cache) == kKvF16) {
                kv_snapshot(kv_d);
                const std::size_t row = (std::size_t) cache.kv_lora + (std::size_t) cache.n_rot;
                for (int j = 0; j < W; ++j) {
                    int first = -1;
                    double worst = 0.0;
                    long bad = 0;
                    for (std::size_t b = 0; b < kv_a.size() && b < kv_d.size(); ++b) {
                        if (kv_a[b].empty() || kv_d[b].empty()) continue;
                        const auto d = cmp(kv_d[b].data() + (std::size_t) j * row,
                                           kv_a[b].data() + (std::size_t) j * row, row);
                        if (d.ndiff > 0) {
                            if (first < 0) first = (int) b;
                            ++bad;
                            if (d.max_abs > worst) worst = d.max_abs;
                        }
                    }
                    std::printf("VPKV row=%d first_block=%d blocks_with_diff=%ld worst_max_abs=%.3g\n", j,
                                first, bad, worst);
                }
            }

            std::vector<int> blocks;
            for (const auto& kv : pc.state) blocks.push_back(kv.first);
            std::sort(blocks.begin(), blocks.end());
            // Per block, what THIS block generated for each row (phase D), most-guilty first.
            for (int j = 0; j < W; ++j) {
                struct G { int block; double max_abs; long ndiff; };
                std::vector<G> gen;
                for (int b : blocks) {
                    const auto itd = pd.state.find(b);
                    const auto itr = pa[(std::size_t) j].state.find(b);
                    if (itd == pd.state.end() || itr == pa[(std::size_t) j].state.end()) continue;
                    const auto d = cmp(itd->second.data() + (std::size_t) j * pd.width,
                                       itr->second.data(), pd.width);
                    if (d.ndiff > 0) gen.push_back(G{b, d.max_abs, d.ndiff});
                }
                std::sort(gen.begin(), gen.end(), [](const G& x, const G& y) { return x.max_abs > y.max_abs; });
                std::printf("VPGEN row=%d generating_blocks=%zu of %zu", j, gen.size(), blocks.size());
                for (std::size_t i = 0; i < gen.size() && i < 8; ++i)
                    std::printf(" | blk %d %.3g (%ld)", gen[i].block, gen[i].max_abs, gen[i].ndiff);
                std::printf("\n");
                // ...and for each of those blocks, whether the ATTENTION half already differs. The state
                // after the block cannot say; `mla_attention`'s own output can.
                for (std::size_t i = 0; i < gen.size() && i < 8; ++i) {
                    const int b = gen[i].block;
                    const auto ia = pd.attn.find(b);
                    const auto ir = pa[(std::size_t) j].attn.find(b);
                    if (ia == pd.attn.end() || ir == pa[(std::size_t) j].attn.end() || ia->second.empty() ||
                        ir->second.empty())
                        continue;
                    const std::size_t aw = attn_w[(std::size_t) b];
                    if (ia->second.size() < aw * (std::size_t) W || ir->second.size() < aw) continue;
                    const auto d = cmp(ia->second.data() + (std::size_t) j * aw, ir->second.data(), aw);
                    std::printf("VPATTN row=%d block=%d heads=%d attn_max_abs=%.3g attn_ndiff=%ld of %zu "
                                "-> %s half\n",
                                j, b, (int) g.layers[(std::size_t) b].n_heads, d.max_abs, d.ndiff, aw,
                                d.ndiff > 0 ? "ATTENTION" : "FFN");
                }
            }
            for (int j = 0; j < W; ++j) {
                const Probe& ref_p = pa[(std::size_t) j];
                int first = -1, first_ids = -1, ndirty = 0;
                double last_abs = 0.0;
                for (int b : blocks) {
                    const auto itc = pc.state.find(b);
                    const auto itr = ref_p.state.find(b);
                    if (itc == pc.state.end() || itr == ref_p.state.end()) continue;
                    const float* row = itc->second.data() + (std::size_t) j * pc.width;
                    const auto d = cmp(row, itr->second.data(), pc.width);
                    if (d.ndiff > 0) {
                        if (first < 0) first = b;
                        ++ndirty;
                        last_abs = d.max_abs;
                    }
                    // Printed for the first few parting blocks only: enough to see where it starts and how
                    // it grows, without 43 lines a row.
                    if (first >= 0 && b <= first + 2)
                        std::printf("VPBLK row=%d block=%d max_abs=%.3g ndiff=%ld of %zu\n", j, b, d.max_abs,
                                    d.ndiff, pc.width);
                    const auto ic = pc.ids.find(b);
                    const auto ir = ref_p.ids.find(b);
                    if (ic != pc.ids.end() && ir != ref_p.ids.end() && !ic->second.empty() &&
                        !ir->second.empty()) {
                        const std::int32_t* a2 = ic->second.data() + (std::size_t) j * pc.top_k;
                        long bad = 0;
                        for (int t = 0; t < pc.top_k; ++t)
                            if (a2[t] != ir->second[(std::size_t) t]) ++bad;
                        if (bad > 0 && first_ids < 0) {
                            first_ids = b;
                            std::printf("VPIDS row=%d block=%d expert_ids_differ=%ld of %d\n", j, b, bad,
                                        pc.top_k);
                        }
                    }
                }
                const auto dl = cmp(lc.data() + (std::size_t) j * (std::size_t) vocab, la[(std::size_t) j].data(),
                                    (std::size_t) vocab);
                std::printf("VPROW row=%d first_block=%d first_routing_block=%d dirty_blocks=%d of %zu "
                            "last_block_max_abs=%.3g logits_max_abs=%.3g logits_ndiff=%ld\n",
                            j, first, first_ids, ndirty, blocks.size(), last_abs, dl.max_abs, dl.ndiff);
            }
            std::fflush(stdout);
            return 0;
        }

        // the reference rows: plain decode, one token a call
        std::vector<std::vector<float>> ref((std::size_t) max_w);
        for (int j = 0; j < max_w; ++j)
            if (!run(depth + j, 1, true, ref[(std::size_t) j])) return 2;
        cache.len = depth;
        std::vector<double> ar;
        for (int j = 0; j < ar_warm + ar_steps; ++j) {
            const auto t0 = std::chrono::steady_clock::now();
            if (!run(depth + j, 1, true, lg)) return 2;
            if (j >= ar_warm) ar.push_back(ms_since(t0));
        }
        cache.len = depth;
        std::size_t vfree = 0, vtotal = 0;
        cudaMemGetInfo(&vfree, &vtotal);
        const auto a = summary(ar);
        std::printf("VB_CONF depth=%d reps=%d ar_steps=%d multi_exact=%d experts_resident=%d free_mib=%zu "
                    "total_mib=%zu\n",
                    depth, reps, ar_steps, kernels::native_mmvq_multi_exact() ? 1 : 0, resident ? 1 : 0,
                    vfree >> 20, vtotal >> 20);
        std::printf("VB,W,mode,depth,median_ms,p10_ms,p90_ms,n,ratio_to_ar\n");
        std::printf("VB,1,ar,%d,%.3f,%.3f,%.3f,%zu,1.000\n", depth, a.med, a.p10, a.p90, ar.size());
        std::vector<std::vector<double>> times(widths.size());
        // Only this loop, for `nsys profile --capture-range=cudaProfilerApi`; a no-op without a profiler.
        cudaProfilerStart();
        for (int rep = 0; rep < reps; ++rep) {
            for (std::size_t k = 0; k < widths.size(); ++k) {
                const std::size_t wi = (k + (std::size_t) rep) % widths.size();
                const int W = widths[wi];
                const auto t0 = std::chrono::steady_clock::now();
                if (!run(depth, W, true, lg)) return 2;
                times[wi].push_back(ms_since(t0));
                cache.len = depth;
                if (rep != 0) continue;
                // Compare the entire vocabulary row against the corresponding decode step.
                for (int j = 0; j < W; ++j) {
                    const float* row = lg.data() + (std::size_t) j * (std::size_t) vocab;
                    const float* rr = ref[(std::size_t) j].data();
                    double max_abs = 0.0;
                    long ndiff = 0;
                    int64_t am = 0, ar0 = 0, ar1 = -1;
                    for (int64_t v = 0; v < vocab; ++v) {
                        const double d = std::fabs((double) row[v] - (double) rr[v]);
                        if (d > max_abs) max_abs = d;
                        if (row[v] != rr[v]) ++ndiff;
                        if (row[v] > row[am]) am = v;
                        if (rr[v] > rr[ar0]) { ar1 = ar0; ar0 = v; }
                        else if (v != ar0 && (ar1 < 0 || rr[v] > rr[ar1])) ar1 = v;
                    }
                    std::printf("VBPAR W=%d row=%d max_abs=%.3g ndiff=%ld argmax_eq=%d ref_top1_top2=%.4f\n", W, j,
                                max_abs, ndiff, am == ar0 ? 1 : 0,
                                ar1 >= 0 ? (double) (rr[ar0] - rr[ar1]) : 0.0);
                }
            }
        }
        cudaProfilerStop();
        for (std::size_t wi = 0; wi < widths.size(); ++wi) {
            const auto s = summary(times[wi]);
            std::printf("VB,%d,verify,%d,%.3f,%.3f,%.3f,%zu,%.3f\n", widths[wi], depth, s.med, s.p10, s.p90,
                        times[wi].size(), a.med > 0.0 ? s.med / a.med : 0.0);
        }
        std::fflush(stdout);
        return 0;
    }

    // The CLI and server use the same DFly generation loop.
    const char* draft_kind = flag_value(argc, argv, "--draft");
    std::string draft_model;
    int draft_max = 3;
    DflyWindow draft_window = DflyWindow::fixed;
    if (draft_kind != nullptr) {
        if (std::string(draft_kind) != "dfly") {
            std::fprintf(stderr, "\nhy4_flash: --draft %s: only dfly is supported\n", draft_kind);
            return 2;
        }
        if (has_flag(argc, argv, "--mtp")) {
            std::fprintf(stderr, "\nhy4_flash: --draft dfly and --mtp are two drafters; choose one\n");
            return 2;
        }
        const char* dm = flag_value(argc, argv, "--draft-model");
        if (dm == nullptr) {
            std::fprintf(stderr, "\nhy4_flash: --draft dfly needs --draft-model PATH (the Q8_0 drafter)\n");
            return 2;
        }
        if (kv_format(cache) == kKvF32) {
            std::fprintf(stderr, "hy4_flash: --draft dfly requires F16 or Q8_0 KV\n");
            return 2;
        }
        draft_model = dm;
        if (const char* v = flag_value(argc, argv, "--draft-max")) {
            char* end = nullptr;
            const long k = std::strtol(v, &end, 10);
            if (end == v || *end != '\0' || k < 0 || k > DflyDrafter::kMaxK) {
                std::fprintf(stderr, "\nhy4_flash: --draft-max %s: must be an integer in 0..%d\n", v,
                             DflyDrafter::kMaxK);
                return 2;
            }
            draft_max = (int) k;
        }
        if (const char* v = flag_value(argc, argv, "--draft-window")) {
            if (!dfly_window_parse(v, draft_window)) {
                std::fprintf(stderr, "\nhy4_flash: --draft-window %s: fixed or confidence\n", v);
                return 2;
            }
        } else if (has_flag(argc, argv, "--draft-window")) {
            std::fprintf(stderr, "\nhy4_flash: --draft-window needs fixed or confidence\n");
            return 2;
        }
    } else if (flag_value(argc, argv, "--draft-model") != nullptr || flag_value(argc, argv, "--draft-max") != nullptr ||
               has_flag(argc, argv, "--draft-window")) {
        std::fprintf(stderr, "\nhy4_flash: --draft-model / --draft-max / --draft-window need --draft dfly\n");
        return 2;
    }
    if (serving) {
        // Keep the loaded weights and cache alive across requests.
        std::printf("  serve          %d context positions, chunks of %d; the protocol is the engine's own\n"
                    "                 (GEN / T / DONE on stdin and stdout)\n", reserve,
                    prefill_chunk_max());
        std::fflush(stdout);
        const std::string gguf_path = path, pack_path = pack;
        ServeState ss{};
        auto cache_option = [&](const char* key, int& out) {
            if (!has_flag(argc, argv, key)) return true;
            const char* value = flag_value(argc, argv, key);
            if (!value) return false;
            const char* end = value + std::strlen(value);
            auto result = std::from_chars(value, end, out);
            return result.ec == std::errc{} && result.ptr == end && out >= 0;
        };
        if (!cache_option("--conversation-cache-mib", ss.conversation_cache_mib) ||
            !cache_option("--conversation-cache-slots", ss.conversation_cache_slots)) {
            std::fprintf(stderr, "hy4-flash conversation cache budgets require nonnegative integers\n");
            return 2;
        }
        if (const char* shard = flag_value(argc, argv, "--vision-shard")) {
#ifdef STRATA_HY4_FLASH_VISION
            ss.vision_shard = shard;
            if (!probe_vision(shard, err)) {
                std::fprintf(stderr, "hy4-flash vision: %s\n", err.c_str());
                return 2;
            }
#else
            std::fprintf(stderr, "This build has no hy4-flash vision encoder\n");
            return 2;
#endif
        }
        ss.gguf = &gguf_path;
        ss.pack = &pack_path;
        ss.g = &g;
        ss.cache = &cache;
        ss.fo = &fo;
        ss.max_context = reserve;
        ss.vocab = vocab;
        ss.sampler = sc;
        ss.stop_tokens = stop_ids;
        std::unique_ptr<Mtp> mtp;
        if (has_flag(argc, argv, "--mtp")) {
            int draft = 3;
            if (const char* value = flag_value(argc, argv, "--draft-tokens")) draft = std::atoi(value);
            if (draft < 1 || draft > 7) {
                std::fprintf(stderr, "hy4-flash --draft-tokens must be 1..7\n"); return 2;
            }
            try { mtp = std::make_unique<Mtp>(gguf_path, g, reserve); }
            catch (const std::exception& e) { std::fprintf(stderr,"hy4-flash MTP: %s\n",e.what()); return 2; }
            ss.mtp = mtp.get(); ss.draft_tokens = draft;
        }
        // The drafter is read and checked here, before READY, so a wrong file fails the start and not the
        // first request; it is bound to the target after the first pass with logits (dfly_generate).
        DflyRuntime dfly_rt;
        if (draft_kind != nullptr && draft_max > 0) {
            std::string derr;
            if (!dfly_runtime_init(dfly_rt, g, derr) || !dfly_rt.drafter.load(draft_model, g, derr)) {
                std::fprintf(stderr, "\nhy4_flash: --draft dfly: %s\n", derr.c_str());
                return 2;
            }
            std::printf("  draft          dfly, k <= %d, window %s, %s; every request, sample-and-match\n",
                        draft_max, dfly_window_name(draft_window), draft_model.c_str());
            std::fflush(stdout);
            ss.dfly = &dfly_rt;
            ss.dfly_model = &draft_model;
            ss.dfly_k_max = draft_max;
            ss.dfly_window = draft_window;
        } else if (draft_kind != nullptr) {
            std::printf("  draft          dfly OFF: --draft-max 0; plain decoding\n");
        }
        return serve_loop(ss);
    }
    if (!sc.greedy())
        std::printf("                 seed %llu - the same seed, prompt and parameters repeat this run\n",
                    (unsigned long long) sampler.seed());
    if (draft_kind != nullptr) {
        if (draft_max > 0) {
            // The seed the plain path resolved (a random one when none was given), so the Dfly loop's sampler
            // draws the same numbers the plain loop below would.
            SamplerConfig dsc = sc;
            dsc.seed = sampler.seed();
            return run_dfly_cli(path, g, pack, tokens, fo, dense, cache, dsc, stop_ids, max_new, draft_model,
                                draft_max, draft_window);
        }
        std::printf("  draft          dfly OFF for this run: --draft-max 0; plain decoding\n");
    }

    std::vector<int32_t> history = tokens;   // the Engram hash needs the whole sequence, not one step of it
    std::vector<int32_t> step = tokens;
    std::vector<int32_t> produced;
    std::vector<double> step_ms;
    PhaseTime pt{};
    const bool want_phases = std::getenv("STRATA_HY4_FLASH_PHASES") != nullptr;
    PhaseTime warm{};   ///< the same, reset after the first step: step 0 fills the residency and is not typical
    int reported = -1;   ///< the residency size last printed, so growth is reported and nothing else is
    // Split the prompt at the maximum supported forward width.
    const int chunk = prefill_chunk_max();
    // Accumulate prompt phase timings separately from decode.
    PhaseTime pf{};
    long pf_tokens = 0;
    // Time the first chunk separately because it also uploads persistent dense weights.
    PhaseTime pf0{};
    long pf0_tokens = 0;
    std::printf("  prefill        %zu tokens through %d trunk blocks in chunks of %d; the first chunk also\n"
                "                 fills the dense residency, so it is the slow one\n", tokens.size(),
                n_blocks, chunk);
    std::fflush(stdout);
    // Reset the cache between prefill timing repetitions.
    int pf_repeat = 1;
    if (const char* v = flag_value(argc, argv, "--prefill-repeat")) pf_repeat = std::max(1, std::atoi(v));
    for (int pass = 0; pass < pf_repeat; ++pass) {
    if (pass > 0) {
        cache.len = 0;
        pf = PhaseTime{};
        pf_tokens = 0;
    }
    if ((int) tokens.size() > chunk) {
        const auto t_pf = std::chrono::steady_clock::now();
        std::size_t done = 0;
        while (tokens.size() - done > (std::size_t) chunk) {
            const std::vector<int32_t> part(tokens.begin() + (long) done,
                                            tokens.begin() + (long) done + chunk);
            std::vector<float> ignored;
            ForwardOptions co = fo;
            co.history = &history;
            co.want_logits = false;
            if (want_phases) co.timing = done == 0 ? &pf0 : &pf;
            if (!forward_prefill(path, g, pack, part, ignored, err, co)) {
                std::fprintf(stderr, "\nhy4_flash: prompt chunk at %zu: %s\n", done, err.c_str());
                return 2;
            }
            done += (std::size_t) chunk;
            if (done == (std::size_t) chunk) pf0_tokens += chunk;
            else pf_tokens += chunk;
        }
        // The tail goes through the loop below, which wants the logits of its last row.
        step.assign(tokens.begin() + (long) done, tokens.end());
        const double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t_pf).count();
        std::printf("  prompt    %s %zu of %zu tokens cached in %.2f s (%.1f token/s); %zu left for the\n"
                    "                 first generating call\n",
                    pf_repeat > 1 ? (pass == 0 ? "COLD" : "warm") : "    ", done, tokens.size(), secs,
                    secs > 0.0 ? (double) done / secs : 0.0, step.size());
        std::fflush(stdout);
    }
    }   // --prefill-repeat

    for (int n = 0; n < max_new; ++n) {
        std::vector<float> logits;
        rd.step = n;
        fo.history = &history;
        // Only use the multi-row head above the GEMM threshold.
        fo.logits_last_only = (int) step.size() > kGemmT;
        if (want_phases) fo.timing = &pt;
        const auto t0 = std::chrono::steady_clock::now();
        if (!forward_prefill(path, g, pack, step, logits, err, fo)) {
            std::fprintf(stderr, "\nhy4_flash: %s\n", err.c_str());
            return 2;
        }
        // The last prompt row predicts the next token.
        const std::size_t last =
            fo.logits_last_only ? 0 : (std::size_t) ((int64_t) step.size() - 1) * vocab;
        const int64_t argmax = sampler.pick(logits.data() + last, vocab, history);
        if (argmax < 0) {
            std::fprintf(stderr, "\nhy4_flash: the logits are not all finite at step %d\n", n);
            return 2;
        }
        // Dump the entire logits row for each step so comparisons include non-argmax values.
        if (const char* dir = std::getenv("HY4_FLASH_LOGITS_DIR")) {
            char file[2048];
            std::snprintf(file, sizeof file, "%s/step-%03d.f32", dir, n);
            std::FILE* f = std::fopen(file, "wb");
            if (f == nullptr) {
                std::fprintf(stderr, "\nhy4_flash: cannot write %s\n", file);
                return 2;
            }
            const std::size_t wrote = std::fwrite(logits.data() + last, sizeof(float), (std::size_t) vocab, f);
            std::fclose(f);
            if (wrote != (std::size_t) vocab) {
                std::fprintf(stderr, "\nhy4_flash: short write to %s\n", file);
                return 2;
            }
        }
        const double secs =
            std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        std::printf("  token %-2d       %lld (logit %.4f, %d cached) %.2f ms (%.2f t/s)\n", n,
                    (long long) argmax, (double) logits[last + (std::size_t) argmax], cache.len, secs * 1e3,
                    secs > 0.0 ? 1.0 / secs : 0.0);
        // Refresh the memory report when the residency grows.
        if (dense.tier_filled != reported) {
            reported = dense.tier_filled;
            if (dense.host_bytes != 0)
                std::printf("  host tier      %.1f GiB of the expert range warmed into the page cache; a slice\n"
                            "                 then costs a memcpy, not a cold read\n",
                            (double) dense.host_bytes / (double) (1u << 30));
            std::printf("  expert tier    %.2f GiB on the card: %d of %d slots filled, of %d experts total\n",
                        (double) dense.tier_bytes / (double) (1u << 30), dense.tier_filled,
                        dense.tier_slots, n_moe * g.n_experts);
        }
        step_ms.push_back(secs * 1e3);
        if (n == 0) {
            // Exclude initial weight uploads from decode phase timing.
            const double w0 = pt.seen_wait, r0 = pt.seen_read, p0 = pt.seen_post;
            const long f0 = pt.seen_fetch;
            warm = PhaseTime{};
            pt = PhaseTime{};
            pt.seen_wait = w0; pt.seen_read = r0; pt.seen_post = p0; pt.seen_fetch = f0;
        } else {
            warm = pt;
        }
        std::fflush(stdout);
        std::fflush(stdout);
        produced.push_back((int32_t) argmax);
        history.push_back((int32_t) argmax);
        step.assign(1, (int32_t) argmax);   // from here on, one token at a time against the cache
        // AFTER it is recorded and its logits written: a stop token is a token the model produced, and the
        // comparison files are keyed by step, so dropping it would renumber every later capture.
        if (is_stop(argmax)) {
            std::printf("  stop           %lld ends the turn at step %d of %d\n", (long long) argmax, n,
                        max_new);
            break;
        }
    }

    if (dense.cpu_experts != 0)
        std::printf("  CPU experts    %llu misses computed on CPU; %.2f MiB result upload\n",
                    dense.cpu_experts, (double)dense.cpu_result_bytes/(1024.0*1024.0));
    if (dense.tier_hits + dense.tier_misses > 0) {
        const double n = (double) (dense.tier_hits + dense.tier_misses);
        std::printf("  tier hit rate  %.3f (%ld hits, %ld misses); H2D %.2f GiB, D2D %.2f GiB over the run\n",
                    (double) dense.tier_hits / n, dense.tier_hits, dense.tier_misses,
                    (double) dense.h2d_bytes / (double) (1u << 30),
                    (double) dense.d2d_bytes / (double) (1u << 30));
    }

    {   // S1 of the Engram I/O plan: what the row read costs, from the run being quoted
        double secs = 0.0;
        long steps = 0, requests = 0, recur = 0, distinct = 0;
        strata::hy4_flash::engram_read_cost(secs, steps, requests, recur, distinct);
        if (steps > 0) {
            std::printf("  engram rows    %.3f ms a token reading %ld rows a step (%.1f us a row)\n",
                        secs * 1e3 / (double) steps, requests / steps,
                        secs * 1e6 / (double) requests);
            if (distinct > 0)
                std::printf("                 %ld requests, %ld distinct, recurrence %.3f%s\n", requests,
                            distinct, (double) recur / (double) requests,
                            recur == 0 ? " (no row was ever asked for twice)" : "");
            else
                std::printf("                 STRATA_HY4_FLASH_ENGRAM_STATS=1 for the row recurrence rate\n");
        }
    }

    {
        double secs = 0.0, backlog = 0.0;
        long n = 0;
        strata::hy4_flash::latent_moe_route_sync_cost(secs, backlog, n);
        if (n > 0 && !step_ms.empty()) {
            const double t = (double) step_ms.size();
            std::printf("  route readback %ld drains a run: %.2f ms a token waiting on queued GPU work,\n"
                        "                 %.2f ms a token on the readback itself\n",
                        n, backlog * 1e3 / t, secs * 1e3 / t);
        }
    }

    // Report timing quantiles across decode steps.
    if (step_ms.size() > 1) {
        std::vector<double> w(step_ms.begin() + 1, step_ms.end());
        std::sort(w.begin(), w.end());
        const auto q = [&w](double f) {
            return w[(std::size_t) ((double) (w.size() - 1) * f + 0.5)];
        };
        std::printf("  warm decode    %zu steps: min %.2f  median %.2f  p90 %.2f ms  ->  %.2f t/s at the "
                    "median\n", w.size(), w.front(), q(0.5), q(0.9), 1000.0 / q(0.5));
        std::printf("  TTFT           %.0f ms (prefill of %zu tokens; expert startup reported separately)\n",
                    step_ms.front(), tokens.size());
    }

    if (want_phases) {
        // Warm only: step 0 reads every dense weight once and says nothing about a steady-state step.
        const PhaseTime& w = warm;
        const int n = w.passes > 0 ? w.passes : 1;
        const double parts = w.engram_read + w.engram_gpu + w.weights + w.attention + w.moe + w.dense_ffn +
                             w.head;
        struct Row { const char* name; double secs; } rows[] = {
            {"engram rows (file)", w.engram_read}, {"engram layer", w.engram_gpu},
            {"dense weights", w.weights},          {"attention", w.attention},
            {"moe (incl. fetch)", w.moe},          {"dense ffn", w.dense_ffn},
            {"output head", w.head},
        };
        // The cold chunk, named as such and never averaged with the warm ones.
        if (pf0_tokens > 0) {
            const double c = pf0.engram_read + pf0.engram_gpu + pf0.weights + pf0.attention + pf0.moe +
                             pf0.dense_ffn + pf0.head;
            std::fprintf(stderr, "\nper-phase WALL CLOCK of the FIRST prompt chunk, %ld tokens - this is the\n"
                                 "chunk that fills the dense residency, so `dense weights` here is a\n"
                                 "once-per-process upload reported per token:\n", pf0_tokens);
            struct CRow { const char* name; double secs; } crows[] = {
                {"engram rows (file)", pf0.engram_read}, {"engram layer", pf0.engram_gpu},
                {"dense weights", pf0.weights},          {"attention", pf0.attention},
                {"moe (incl. fetch)", pf0.moe},          {"dense ffn", pf0.dense_ffn},
                {"output head", pf0.head},
            };
            for (const CRow& r : crows)
                std::fprintf(stderr, "  %-22s %8.2f ms/token  %5.1f%%\n", r.name,
                             1e3 * r.secs / (double) pf0_tokens, c > 0 ? 100.0 * r.secs / c : 0.0);
            std::fprintf(stderr, "  %-22s %8.2f ms/token  (%.1f token/s)\n", "sum of phases",
                         1e3 * c / (double) pf0_tokens, c > 0 ? (double) pf0_tokens / c : 0.0);
            if (pf_tokens == 0)
                std::fprintf(stderr, "  (no warm chunk: this prompt had only one. Use a prompt longer than\n"
                                     "   two chunks to see the steady state.)\n");
        }
        if (pf_tokens > 0) {
            // THE PROMPT, per token of it. Not comparable to the warm rows below: a chunk processes `chunk`
            // tokens in one pass, and its first chunk also fills the dense residency.
            const double pparts = pf.engram_read + pf.engram_gpu + pf.weights + pf.attention + pf.moe +
                                  pf.dense_ffn + pf.head;
            struct PRow { const char* name; double secs; } prows[] = {
                {"engram rows (file)", pf.engram_read}, {"engram layer", pf.engram_gpu},
                {"dense weights", pf.weights},          {"attention", pf.attention},
                {"moe (incl. fetch)", pf.moe},          {"dense ffn", pf.dense_ffn},
                {"output head", pf.head},
            };
            std::fprintf(stderr, "\nper-phase WALL CLOCK of the WARM prompt chunks, %ld tokens in chunks\n"
                                 "of %d (the first chunk is reported above and excluded here):\n",
                         pf_tokens, chunk);
            for (const PRow& r : prows)
                std::fprintf(stderr, "  %-22s %8.2f ms/token  %5.1f%%\n", r.name,
                             1e3 * r.secs / (double) pf_tokens,
                             pparts > 0 ? 100.0 * r.secs / pparts : 0.0);
            std::fprintf(stderr, "  %-22s %8.2f ms/token  (%.1f token/s)\n", "sum of phases",
                         1e3 * pparts / (double) pf_tokens,
                         pparts > 0 ? (double) pf_tokens / pparts : 0.0);
        }
        std::fprintf(stderr, "\nper-phase WALL CLOCK, %d warm step(s), one token each:\n", n);
        for (const Row& r : rows)
            std::fprintf(stderr, "  %-22s %8.1f ms/token  %5.1f%%\n", r.name, 1e3 * r.secs / n,
                         parts > 0 ? 100.0 * r.secs / parts : 0.0);
        std::fprintf(stderr, "  %-22s %8.1f ms/token\n", "sum of phases", 1e3 * parts / n);
        if (w.fetches > 0) {
            const double per = 1e3 / (double) w.fetches;
            std::fprintf(stderr, "  of which the slice fetch, %ld of them a step:\n",
                         w.fetches / (long) n);
            std::fprintf(stderr, "    %-20s %8.1f ms/token  %6.3f ms each  (waiting on the GPU)\n",
                         "fetch: wait", 1e3 * w.fetch_wait / n, w.fetch_wait * per);
            std::fprintf(stderr, "    %-20s %8.1f ms/token  %6.3f ms each  (the file)\n",
                         "fetch: read", 1e3 * w.fetch_read / n, w.fetch_read * per);
            std::fprintf(stderr, "    %-20s %8.1f ms/token  %6.3f ms each  (enqueue)\n",
                         "fetch: enqueue", 1e3 * w.fetch_post / n, w.fetch_post * per);
        }
        std::fprintf(stderr, "  %-22s %8.1f ms/token  (the rest is the embedding, the hc_post chain and\n"
                             "                         the host-side setup each call repeats)\n",
                     "whole call", 1e3 * w.total / n);
    }

    std::printf("  generated     ");
    for (int32_t t : produced) std::printf(" %d", t);
    std::printf("\n");
    std::fprintf(stderr, "\nhy4_flash: %zu token(s) generated against a latent KV cache.\n", produced.size());
    return 0;
}

}  // namespace strata::hy4_flash
