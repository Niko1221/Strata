// src/dsv4/dsv4_engine.cu - the DeepSeek V4-Flash (deepseek4) engine path.
//
// Slice map (notes/engines/202610-dsv4-engine-integration-plan.md section 6.2).  This file is one translation
// unit organized in sections, the way src/qwen35/q36_engine.cu is; the section banners below are the anchors
// the later slices fill, in design order.  Geometry comes from include/strata/dsv4/geometry.hpp
// (kDsv4CompiledGeometry); the shared core/kCompiledGeometry is Qwen's and is never referenced here, so the
// Qwen/qwen35 paths stay byte-identical.
//
//   [S0] THIS SLICE: family wiring, geometry summary, pack + GGUF load in the section-5.2 order, expert-pool
//        registration, KV/workspace arena budget, residency report, clean exit.  No forward, no kernels.
//   [S1] tail: token embed, rmsnorm, hyper-connection small ops, sinkhorn(20), LM head, sampler; replay/dump
//   [S2] MoE chain: sqrtsoftplus routing, tid2eid hash layers, NVFP4 expert pool (CPU), shared expert, combine
//   [S3] dense MLA attention (hash layers 0/1)
//   [S4] DSA compressor (ratio 4 / 128 layers)
//   [S5] indexer + sparse attention (ratio-4 layers, EXACT top-512)
//   [S6] decode loop + KV write path
//   [S7] expert pool tiers (VRAM cache) + performance
//
// The section-5.2 load order is a hard rule (the GGUF is 164 GB and the page cache is the expert tier's RAM):
//   1. dense.bin  -> the WeightTable arena (0.79 GiB of F32/BF16 dense tensors), no GGUF page touched;
//   2. the 370 native (quantized) tensors, uploaded from the GGUF mmap;
//   3. MADV_DONTNEED on exactly the spans step 2 read (their pages leave the cache);
//   4. only then the expert tier: native_experts.txt v3 + the experts.bin mmap (FileExpertSource).
// The GGUF's expert merger tensors (~145 GiB) are NEVER read: the layout check reads the header only, and the
// upload list is the pack index's kind-0 rows, which name no expert merger.

#include <algorithm>
#include <chrono>
#include <cinttypes>
#include <cmath>
#include <filesystem>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <memory>
#include <set>
#include <sstream>
#include <string>
#include <vector>

#include <cuda_runtime.h>

#if !defined(_WIN32)
#include <sys/mman.h>
#endif

#include "dsv4_attn_time.h"

#include "strata/artifact/gguf_reader.hpp"
#include "strata/artifact/model_family.hpp"
#include "strata/core/expert_source.hpp"
#include "strata/core/expert_cache.hpp"
#include "strata/core/weights.hpp"
#include "strata/dsv4/engine.hpp"
#include "strata/dsv4/geometry.hpp"
#include "strata/dsv4/layout.hpp"
#include "strata/kernels/cpu/expert_layout.hpp"
#include "strata/kernels/cpu/hadamard.hpp"
#include "strata/kernels/elementwise.hpp"
#include "strata/kernels/iq_kernels.hpp"
#include "strata/kernels/native_mmvq.hpp"

#include "dsv4_moe.h"
#include "dsv4_moe_time.h"
#include "strata/dsv4/attn.hpp"
#include "strata/dsv4/comp.hpp"

#define CK(x)                                                                                                     \
    do {                                                                                                          \
        cudaError_t e_ = (x);                                                                                     \
        if (e_ != cudaSuccess) {                                                                                  \
            std::fprintf(stderr, "strata-dsv4: CUDA error %s at %s:%d\n", cudaGetErrorString(e_), __FILE__,       \
                         __LINE__);                                                                               \
            std::exit(1);                                                                                         \
        }                                                                                                         \
    } while (0)

using Clock = std::chrono::steady_clock;
static double ms_since(Clock::time_point t0) {
    return std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
}

namespace {

// ============================================================================================== [S0] helpers

#if !defined(_WIN32)
constexpr int kAdviseRandom = MADV_RANDOM;
constexpr int kAdviseDrop = MADV_DONTNEED;
/// madvise(2) on the page-aligned span holding [p, p+bytes).  Best effort: a failure here never fails a load.
void advise_span(const void* p, uint64_t bytes, int advice) {
    if (!p || bytes == 0) return;
    const uintptr_t page = 4096;
    const uintptr_t a = (uintptr_t) p & ~(page - 1);
    const uintptr_t b = ((uintptr_t) p + bytes + page - 1) & ~(page - 1);
    if (b > a) (void) madvise((void*) a, (size_t) (b - a), advice);
}
#else
constexpr int kAdviseRandom = 0;
constexpr int kAdviseDrop = 0;
void advise_span(const void*, uint64_t, int) {}
#endif

std::string dir_of(const std::string& path) {
    const size_t slash = path.find_last_of("/\\");
    return slash == std::string::npos ? std::string(".") : path.substr(0, slash);
}

const char* gguf_type_name(uint32_t type) {
    switch (type) {
    case strata::dsv4::kGgufF32: return "F32";
    case strata::dsv4::kGgufQ8_0: return "Q8_0";
    case strata::dsv4::kGgufI32: return "I32";
    case strata::dsv4::kGgufBF16: return "BF16";
    case strata::dsv4::kGgufNVFP4: return "NVFP4";
    default: return "?";
    }
}

// ======================================================================================= [S0] command line

struct Options {
    std::string model;          // --model / --native: the GGUF (single shard for this model)
    std::string pack;           // --pack: the pack dir; default <model's dir>/pack
    int64_t ctx = 32768;        // --max-context / --ctx
    bool serve = false;         // --serve: accepted, S1 wires the protocol
    bool have_tokens = false;   // --tokens: accepted, S1+ wires generation
    // [S1] the tail forward + the replay/dump comparison infrastructure (the S2-S5 acceptance gate).
    bool forward = false;       // --dsv4-forward: run the S1 GPU forward after the load
    std::string dump_dir;       // --dsv4-dump-dir DIR: stage dumps (raw f32 .bin) land in DIR/<pass>/
    std::string replay_dir;     // --dsv4-replay DIR: inject DIR/<pass>_L%02d_in_{main,hc}.bin at each layer
    std::string pass_name = "prefill";  // --dsv4-pass NAME: the <pass> component of the dump/replay file names
    std::string prompt_ids;     // --dsv4-prompt-ids "0,128803,...": default is the golden 13-token prompt
    // [S5] decode: after the prefill pass, append the argmax and run N single-token passes
    // (decode1..N; golden generated ids are 734, 260, 4654)
    int decode = 0;             // --dsv4-decode N
    // [G1] multi-row decode: ONE R-row pass (R = verify_rows) over positions start_pos..start_pos+R-1,
    // fed the draft batch in --dsv4-verify-ids, run after the --dsv4-decode passes and leaving the state
    // where it put it (so it must be the last pass of the run).  --dsv4-mrow-bench R instead repeats R-row
    // passes --dsv4-decode (K) times for the c(R) timing, feeding each row the running argmax.
    int verify_rows = 0;        // --dsv4-verify-rows R
    std::string verify_ids;     // --dsv4-verify-ids "a,b,c,..." (R tokens, the draft batch)
    int mrow_bench = 0;         // --dsv4-mrow-bench R
    std::string bench_ids;      // --dsv4-mrow-bench-ids "a,b,...": distinct per-row tokens (real routing)
    // [chunk] long-prompt prefill: split the prompt into W-row passes inside ONE process, each pass
    // continuing the KV / compressor / ring state the previous one left (start_pos = chunk offset:
    // dsv4ref run_pass's decode contract, which R-01/R-06 made bit-exact for s > 1).  0 keeps the
    // single-pass behaviour; W is capped at the sliding window (a pass wider than the ring has no
    // defined ring shift, dsv4_attn.cu's start_pos == 0 check).
    int64_t prefill_chunk = 0;  // --dsv4-prefill-chunk W
};

bool parse_args(int argc, char** argv, Options& o) {
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&]() -> std::string {
            if (i + 1 >= argc) { std::fprintf(stderr, "strata-dsv4: %s needs a value\n", a.c_str()); std::exit(2); }
            return argv[++i];
        };
        // strata's names where they mean the same thing (the family switch passes strata's command line on)
        if (a == "--model" || a == "--native") o.model = next();
        else if (a == "--pack") o.pack = next();
        else if (a == "--max-context" || a == "--ctx") o.ctx = std::atoll(next().c_str());
        else if (a == "--gpu") (void) next();   // applied by strata before the switch (CUDA_VISIBLE_DEVICES)
        else if (a == "--serve") o.serve = true;
        else if (a == "--tokens") { (void) next(); o.have_tokens = true; }
        else if (a == "--dsv4-forward") o.forward = true;
        else if (a == "--dsv4-dump-dir") o.dump_dir = next();
        else if (a == "--dsv4-replay") o.replay_dir = next();
        else if (a == "--dsv4-pass") o.pass_name = next();
        else if (a == "--dsv4-prompt-ids") o.prompt_ids = next();
        else if (a == "--dsv4-decode") o.decode = std::atoi(next().c_str());
        else if (a == "--dsv4-verify-rows") o.verify_rows = std::atoi(next().c_str());
        else if (a == "--dsv4-verify-ids") o.verify_ids = next();
        else if (a == "--dsv4-mrow-bench") o.mrow_bench = std::atoi(next().c_str());
        else if (a == "--dsv4-mrow-bench-ids") o.bench_ids = next();
        else if (a == "--dsv4-prefill-chunk") o.prefill_chunk = std::atoll(next().c_str());
        else if (a.rfind("--", 0) == 0) {
            // the family switch forwards strata's whole command line; an option this engine does not have is
            // said and skipped (with its value), not fatal (q36_engine.cu's serve-mode rule, always on in S0)
            const bool val = i + 1 < argc && std::string(argv[i + 1]).rfind("--", 0) != 0;
            std::fprintf(stderr, "strata-dsv4: ignoring %s%s%s (not an option of this engine)\n", a.c_str(),
                         val ? " " : "", val ? argv[i + 1] : "");
            if (val) ++i;
        } else {
            std::fprintf(stderr,
                         "usage: strata-dsv4 --model GGUF [--pack DIR] [--max-context 32768]\n"
                         "       [--dsv4-forward] [--dsv4-dump-dir DIR] [--dsv4-replay DIR]\n"
                         "       [--dsv4-pass prefill] [--dsv4-prompt-ids \"0,128803,...\"] [--dsv4-decode N]\n"
                         "       [--dsv4-prefill-chunk W]  (W-row prefill passes over one process, default 0\n"
                         "       (S1: loads the pack + GGUF, prints the geometry and expert-pool residency\n"
                         "       report; with --dsv4-forward (implied by --dsv4-dump-dir/--dsv4-replay) runs\n"
                         "       the GPU tail forward embed -> 43 identity layers -> hc_head -> output_norm\n"
                         "       -> head -> logits, dumping/replaying per-stage raw f32 tensors)\n");
            return false;
        }
    }
    if (o.model.empty()) { std::fprintf(stderr, "strata-dsv4: --model is required\n"); return false; }
    if (o.pack.empty()) o.pack = dir_of(o.model) + "/pack";
    if (o.ctx < 256) { std::fprintf(stderr, "strata-dsv4: bad --max-context\n"); return false; }
    if (!o.dump_dir.empty() || !o.replay_dir.empty()) o.forward = true;
    return true;
}

// ============================================================================================ [S0] the model

/// One native (GGUF-quantized) tensor's device copy: raw GGUF blocks, interpreted by the S1+ kernels.
struct NativeTensor {
    std::string name;
    uint8_t* dev = nullptr;
    uint32_t type = 0;
    int64_t ne0 = 0, ne1 = 0;
    uint64_t bytes = 0;
};

struct Model {
    strata::dsv4::Dsv4Geometry g;
    std::shared_ptr<strata::GgufModel> file;   // kept mapped: S7's GGUF-resident expert path reads through it

    strata::core::WeightTable dense;           // the dense.bin tensors (F32 / BF16), in one device arena
    void* dense_arena = nullptr;
    uint64_t dense_arena_bytes = 0;

    std::vector<NativeTensor> native;          // the pack index's kind-0 rows, one device copy each
    uint64_t native_bytes = 0;

    strata::core::FileExpertSource experts;    // experts.bin, mapped lazily (145.13 GiB: no arena could hold it)

    // [S7a-3] the VRAM expert tier: [0] this card's free VRAM, [1] the peer card's (its slots read over
    // P2P).  Either may stay invalid (env opt-out / no peer).
    strata::core::ExpertCache cache[2];
    bool cache_peer = false;                   // device 0 may read device 1's memory
    std::vector<float> heat;                   // [S7b] profile-seeded routing heat (empty: off)

    void* kv_arena = nullptr;                  // [S6] KV cache + workspace, one flat device arena
    uint64_t kv_arena_bytes = 0;
};

/// The pack index's kind-0 (native GGUF) rows: the WeightTable skip set AND the dsv4 native upload list, in
/// index order.  The same 19-field line src/core/weights.cpp parses; here only the name and kind columns matter.
bool index_native_rows(const std::string& pack, std::vector<std::string>& names, int64_t& rows, std::string& err) {
    std::ifstream in(pack + "/index.txt");
    if (!in) { err = pack + "/index.txt: cannot open"; return false; }
    names.clear();
    rows = 0;
    std::string line;
    while (std::getline(in, line)) {
        if (line.empty() || line[0] == '#') continue;
        ++rows;
        std::istringstream ss(line);
        std::string name;
        long long file_id = -1, kind = -1;
        if (!(ss >> name >> file_id >> kind)) { err = pack + "/index.txt: a malformed line: " + line; return false; }
        if (kind == 0) names.push_back(name);
    }
    return true;
}

/// The geometry the file declares must be the geometry the kernels are compiled for, field by field (the guard
/// pins these too; this is the load path's own check, so a mismatch names the field rather than a symptom).
bool geometry_matches_compiled(const strata::dsv4::Dsv4Geometry& g, std::string& err) {
    const strata::dsv4::Dsv4Geometry& c = strata::dsv4::kDsv4CompiledGeometry;
#define DSV4_PIN(field)                                                                                           \
    if (g.field != c.field) {                                                                                     \
        err = "geometry." #field " is " + std::to_string(g.field) + ", this engine is compiled for " +            \
              std::to_string(c.field);                                                                            \
        return false;                                                                                             \
    }
    DSV4_PIN(n_embd) DSV4_PIN(n_embd_out) DSV4_PIN(n_layers) DSV4_PIN(n_vocab)
    DSV4_PIN(n_head) DSV4_PIN(head_dim) DSV4_PIN(q_lora_rank) DSV4_PIN(kv_lora_rank) DSV4_PIN(n_rot)
    DSV4_PIN(sliding_window) DSV4_PIN(o_groups) DSV4_PIN(o_lora_rank) DSV4_PIN(yarn_orig_ctx)
    DSV4_PIN(idx_heads) DSV4_PIN(idx_key_dim) DSV4_PIN(idx_top_k)
    DSV4_PIN(hc) DSV4_PIN(hc_sinkhorn_iters)
    DSV4_PIN(n_expert) DSV4_PIN(n_expert_used) DSV4_PIN(n_expert_shared) DSV4_PIN(n_ff) DSV4_PIN(n_hash_layers)
#undef DSV4_PIN
    return true;
}

/// The KV cache + workspace arena, section 4.1 of the design: a flat fp32 append-only cache per layer class.
///   hash layers 0/1 (pure hash):   ctx            * kv_lora * 4 B
///   even layers 2..42 (r = 4):     (win + ctx/4)  * kv_lora * 4 B, plus indexer keys (ctx/4) * idx_key_dim * 4 B
///   odd layers 3..41 (r = 128):    (win + ctx/128) * kv_lora * 4 B
/// Compressed slots carry their rope phase (j * ratio) fixed at write time; the indexer keys stay fp32
/// (the EXACT top-512 contract).  [S6] owns the write path; S0 sizes and budgets.
uint64_t kv_arena_bytes(const strata::dsv4::Dsv4Geometry& g, int64_t ctx, uint64_t& kv_bytes,
                        uint64_t& idx_bytes) {
    kv_bytes = 0;
    idx_bytes = 0;
    for (int64_t l = 0; l < g.n_layers; ++l) {
        const int64_t ratio = strata::dsv4::dsv4_compress_ratio(l);
        const int64_t slots = ratio == 0 ? ctx : g.sliding_window + ctx / ratio;
        kv_bytes += (uint64_t) slots * (uint64_t) g.kv_lora_rank * 4;
        if (strata::dsv4::dsv4_has_indexer(l)) idx_bytes += (uint64_t) (ctx / ratio) * (uint64_t) g.idx_key_dim * 4;
    }
    // workspace: logits + the S1+ scratch, a fixed reservation until the kernels say what they need
    const uint64_t workspace = (uint64_t) g.n_vocab * 4 + (64ull << 20);
    return kv_bytes + idx_bytes + workspace;
}

void print_geometry(const strata::dsv4::Dsv4Geometry& g) {
    std::fprintf(stderr,
                 "strata-dsv4: DeepSeek V4-Flash: %lld layers, n_embd %lld (hc %lld x %lld = %lld streams), "
                 "vocab %lld\n",
                 (long long) g.n_layers, (long long) g.n_embd, (long long) g.hc, (long long) g.n_embd,
                 (long long) g.n_embd_out, (long long) g.n_vocab);
    std::fprintf(stderr,
                 "strata-dsv4: MLA: %lld heads x %lld, q_lora %lld, kv_lora %lld, rot %lld, win %lld, "
                 "o %lld groups x %lld; rope theta %.0f (compress %.0f), yarn x%.0f of %lld\n",
                 (long long) g.n_head, (long long) g.head_dim, (long long) g.q_lora_rank,
                 (long long) g.kv_lora_rank, (long long) g.n_rot, (long long) g.sliding_window,
                 (long long) g.o_groups, (long long) g.o_lora_rank, g.rope_theta, g.compress_rope_theta,
                 g.yarn_factor, (long long) g.yarn_orig_ctx);
    std::fprintf(stderr,
                 "strata-dsv4: DSA: indexer %lld heads x %lld top-%lld on the ratio-4 layers; compress ratio "
                 "4 (even 2..42) / 128 (odd 3..41), layers 0-1 pure hash, layer 2 hash+DSA\n",
                 (long long) g.idx_heads, (long long) g.idx_key_dim, (long long) g.idx_top_k);
    std::fprintf(stderr,
                 "strata-dsv4: MoE: %lld experts top-%lld + %lld shared, ff %lld, route_scale %.2f, "
                 "swiglu clamp %.1f; HC %lld streams sinkhorn-%lld\n",
                 (long long) g.n_expert, (long long) g.n_expert_used, (long long) g.n_expert_shared,
                 (long long) g.n_ff, g.route_scale, g.swiglu_clamp, (long long) g.hc,
                 (long long) g.hc_sinkhorn_iters);
}

// ============================================================================================== [S0] load

bool load(const Options& o, Model& m, std::string& err) {
    const auto t0 = Clock::now();
    m.file = std::make_shared<strata::GgufModel>(strata::GgufModel::open(o.model));   // throws on a bad file
    const strata::GgufModel& gm = *m.file;
    if (strata::model_family(gm.meta()) != strata::ModelFamily::DeepseekV4) {
        err = "strata-dsv4 runs DeepSeek V4-Flash (deepseek4) files only";
        return false;
    }
    if (std::string guard = strata::check_architecture(gm.meta()); !guard.empty()) { err = guard; return false; }
    if (!strata::dsv4::dsv4_geometry_from_gguf(gm.meta(), m.g, err)) return false;
    if (!geometry_matches_compiled(m.g, err)) return false;
    const strata::dsv4::Dsv4Geometry& g = m.g;
    size_t checked = 0;
    if (!strata::dsv4::dsv4_check_gguf_layout(gm.meta(), g, err, &checked)) return false;
    std::fprintf(stderr, "strata-dsv4: GGUF layout checked: %zu tensors against the dsv4 tables\n", checked);
    print_geometry(g);

    // ---- section 5.2, step 0: the pack index decides which rows are native (kind 0); every other row is a
    // dense.bin tensor for the WeightTable.  The native rows' dst fields are placeholders (the pack never holds
    // their bytes), so they MUST be skipped by the canonical loader and served from the GGUF below.
    std::vector<std::string> native_names;
    int64_t index_rows = 0;
    if (!index_native_rows(o.pack, native_names, index_rows, err)) return false;
    const std::set<std::string> skip(native_names.begin(), native_names.end());
    std::fprintf(stderr, "strata-dsv4: pack index: %lld tensors (%zu native GGUF rows, %lld dense.bin rows)\n",
                 (long long) index_rows, native_names.size(), (long long) (index_rows - (int64_t) native_names.size()));

    // ---- section 5.2, step 1: dense.bin -> the WeightTable arena.  No GGUF page is read here.
    const auto td = Clock::now();
    if (!strata::core::WeightTable::pool_bytes(o.pack, m.dense_arena_bytes, err, &skip)) return false;
    CK(cudaMalloc(&m.dense_arena, m.dense_arena_bytes));
    if (!m.dense.load(o.pack, m.dense_arena, m.dense_arena_bytes, err, &skip)) return false;
    const strata::core::LoadReport& lr = m.dense.report();
    std::fprintf(stderr,
                 "strata-dsv4: dense.bin: %zu tensors, %.2f GiB on device (read %.0f ms, upload %.0f ms; "
                 "%zu native rows skipped: served from the GGUF)\n",
                 lr.tensors, (double) m.dense_arena_bytes / 1073741824.0, lr.read_ms, lr.upload_ms, skip.size());

    // ---- section 5.2, step 2: the native (quantized) tensors, uploaded from the GGUF mmap.  The map is
    // advised MADV_RANDOM span by span as it is read; the expert merger tensors are never in this list.
    const auto tn = Clock::now();
    std::vector<std::pair<const uint8_t*, uint64_t>> uploaded;
    uploaded.reserve(native_names.size());
    for (const std::string& name : native_names) {
        size_t shard = 0;
        const strata::TensorInfo* t = gm.find(name, &shard);
        if (!t) { err = "pack index names " + name + " as native, but the GGUF has no such tensor"; return false; }
        if (!gm.in_bounds(*t, shard)) { err = "GGUF tensor " + name + " lies outside its shard"; return false; }
        const uint8_t* src = gm.shard(shard).tensor_data(*t);
        const uint64_t nb = strata::tensor_payload_bytes(*t);
        advise_span(src, nb, kAdviseRandom);
        NativeTensor w;
        w.name = name;
        w.type = t->type;
        w.ne0 = (int64_t) t->shape[0];
        w.ne1 = t->shape.size() > 1 ? (int64_t) t->shape[1] : 1;
        w.bytes = nb;
        CK(cudaMalloc(&w.dev, nb));
        CK(cudaMemcpy(w.dev, src, nb, cudaMemcpyHostToDevice));   // synchronous: the span is droppable at once
        uploaded.emplace_back(src, nb);
        m.native.push_back(w);
        m.native_bytes += nb;
    }
    CK(cudaDeviceSynchronize());
    // ---- section 5.2, step 3: exactly the uploaded spans leave the page cache.
    for (const auto& [p, nb] : uploaded) advise_span(p, nb, kAdviseDrop);
    std::fprintf(stderr, "strata-dsv4: native: %zu tensors, %.2f GiB on device in %.0f ms (GGUF spans dropped)\n",
                 m.native.size(), (double) m.native_bytes / 1073741824.0, ms_since(tn));

    // ---- section 5.2, step 4: the expert tier, only now.  native_experts.txt v3 sizes the NVFP4 blobs with
    // the RUNTIME geometry (43 x 256 experts of 4096->2048; the compiled-for Qwen H/FF would mis-size them),
    // then FileExpertSource mmaps experts.bin lazily: 145.13 GiB never fits RAM, and is not meant to be read now.
#if !defined(STRATA_DSV4_CPU_POOL)
    err = "this build has no native-expert support (STRATA_NATIVE_EXPERTS=OFF): the V4 pack's experts are NVFP4";
    return false;
#else
    const auto te = Clock::now();
    if (!strata::kernels::cpu::expert_layout_load(o.pack, g.n_layers, g.n_expert, g.n_embd, g.n_ff, err))
        return false;
    // A v5 pack may carry Hadamard-folded experts (hadamard.txt, native_expert.hpp `prism.hadamard`).  The CPU
    // native path rotates the activation, but this engine reads the blobs into its own CUDA GEMV, which does not:
    // a folded pack would load and then compute plausible-looking garbage.  Refuse it instead.
    if (strata::kernels::cpu::hadamard_spec().any()) {
        err = "strata-dsv4: this pack's routed experts are Hadamard-folded (APR/Gyro, native_experts.txt v5); "
              "the dsv4 engine path does not rotate activations and would compute wrong answers";
        return false;
    }
    if (!m.experts.open(o.pack, g.n_layers, g.n_expert, err)) return false;
    const strata::kernels::cpu::ExpertLayout& lay = strata::kernels::cpu::expert_layout();
    std::fprintf(stderr,
                 "strata-dsv4: experts: layout v%d, %lld layers x %lld experts, blob %.2f MiB (%llu B), "
                 "experts.bin %.2f GiB mapped in %.0f ms\n",
                 lay.version, (long long) g.n_layers, (long long) lay.n_expert,
                 (double) lay.max_blob / 1048576.0, (unsigned long long) lay.max_blob,
                 (double) lay.total / 1073741824.0, ms_since(te));
#endif

    // ---- the KV cache + workspace arena (section 4.1), budgeted and allocated.
    uint64_t kv_bytes = 0, idx_bytes = 0;
    m.kv_arena_bytes = kv_arena_bytes(g, o.ctx, kv_bytes, idx_bytes);
    CK(cudaMalloc(&m.kv_arena, m.kv_arena_bytes));
    CK(cudaMemset(m.kv_arena, 0, m.kv_arena_bytes));
    std::fprintf(stderr,
                 "strata-dsv4: KV arena @ %lld ctx: %.2f MiB (latent %.1f + indexer keys %.1f + workspace)\n",
                 (long long) o.ctx, (double) m.kv_arena_bytes / 1048576.0, (double) kv_bytes / 1048576.0,
                 (double) idx_bytes / 1048576.0);
    // ---- [S7a-2] the file tier's read-ahead: whole-blob preads on I/O threads fill the page cache for
    // the layers the MoE's warm window predicts, instead of every expert blob being faulted in a page at
    // a time on the demand H2D copy.  On by default for dsv4 (STRATA_IO_PREFETCH=0 opts out,
    // STRATA_IO_PREFETCH_THREADS overrides the worker count); pure I/O scheduling, no numeric change.
    if (const char* v = std::getenv("STRATA_IO_PREFETCH"); v == nullptr || std::atoi(v) != 0) {
        const int th = [](const char* t) { return t != nullptr ? std::atoi(t) : 0; }(
            std::getenv("STRATA_IO_PREFETCH_THREADS"));
        m.experts.set_io_prefetch(true, th);
        if (m.experts.io_prefetch())
            std::fprintf(stderr, "strata-dsv4: expert io prefetch ON: whole-blob preads on %d threads\n",
                         th > 0 ? th : 8);
        else
            std::fprintf(stderr, "strata-dsv4: expert io prefetch unavailable (Linux mapped tier only)\n");
    }
    // ---- [S7a-3] the VRAM expert tier: every free GiB of BOTH cards as expert slot arenas.  A routed
    // expert's home is cache[expert & 1] (with the peer cache off, everything homes to cache 0); the
    // MoE path admits first-touch and evicts the coldest resident of the same CACHE (A1: across layers
    // by default, within one layer under STRATA_DSV4_EVICT_SCOPE=layer).  The fill doubles as the copy
    // the kernel reads, so a miss costs exactly the staging H2D the uncached path paid and a hit costs
    // nothing.  Device-1 experts are
    // COMPUTED on device 1 (a device-0 kernel reading device-1 slots over P2P measures ~0.9 GiB/s
    // inside the GEMV - only the 16 KiB/token input and the outputs cross PCIe).  STRATA_DSV4_CACHE=0
    // opts out, STRATA_DSV4_CACHE1=0 leaves the peer card out, STRATA_DSV4_CACHE_SLOTS0/1 pin the
    // slot counts.
    // ---- [A1] eviction scope.  `STRATA_DSV4_EVICT_SCOPE=layer` restores R4.2g's per-layer slot ranges
    // and the same-layer victim search exactly (the pre-A1 behaviour); the default `global` opens each
    // pool with no per-layer ranges, so a miss may take the slot of the coldest resident of ANY layer
    // in that cache (dsv4_moe.cu's dsv4_admit_heat).  Why: one 4k MoE call routes ~124 experts while a
    // layer's quota is slots/n_layers ~= 65 on device 0, so the quota rejects admissions the pool as a
    // whole has room for.  Measured on the run's own route trace
    // (runs/202610-dsv4-mvp/pfevict/sim-4k-sweep.txt): quota+prior100 72.3% vs global+prior0 82.7%
    // true hit rate at the same 6325 slots.
    const bool evict_global = [] {
        const char* s = std::getenv("STRATA_DSV4_EVICT_SCOPE");
        return s == nullptr || std::strcmp(s, "layer") != 0;
    }();
    if (const char* v = std::getenv("STRATA_DSV4_CACHE"); v == nullptr || std::atoi(v) != 0) {
        const uint64_t blob = (uint64_t) strata::kernels::cpu::expert_layout().max_blob;
        auto env_slots = [](const char* name, int64_t dflt) {
            const char* s = std::getenv(name);
            return s != nullptr ? std::atoll(s) : dflt;
        };
        size_t free0 = 0, tot0 = 0;
        CK(cudaMemGetInfo(&free0, &tot0));
        // 1500 MiB headroom: the transient MoE/attention workspaces of forward_tail plus the driver's
        const int64_t slots0 = env_slots("STRATA_DSV4_CACHE_SLOTS0",
                                         (int64_t) ((free0 - 1500ull * 1048576) / blob));
        if (slots0 > 0) {
            m.cache[0].set_per_layer_admission(!evict_global);
            if (!m.cache[0].open(slots0, g.n_layers, g.n_expert, (int64_t) blob, err)) return false;
            std::fprintf(stderr, "strata-dsv4: expert cache: device 0: %lld slots (%.1f%% of the pool, %.1f GiB)\n",
                         (long long) m.cache[0].slots(),
                         100.0 * m.cache[0].slots() / (g.n_layers * g.n_expert), m.cache[0].gib());
        }
        if (const char* v1 = std::getenv("STRATA_DSV4_CACHE1"); v1 == nullptr || std::atoi(v1) != 0) {
            int can = 0;
            if (cudaDeviceCanAccessPeer(&can, 0, 1) == cudaSuccess && can) {
                CK(cudaSetDevice(1));
                size_t free1 = 0, tot1 = 0;
                CK(cudaMemGetInfo(&free1, &tot1));
                const int64_t slots1 = env_slots("STRATA_DSV4_CACHE_SLOTS1",
                                                 (int64_t) ((free1 - 700ull * 1048576) / blob));
                if (slots1 > 0) {
                    m.cache[1].set_per_layer_admission(!evict_global);
                    if (!m.cache[1].open(slots1, g.n_layers, g.n_expert, (int64_t) blob, err))
                        return false;
                }
                CK(cudaSetDevice(0));
                if (m.cache[1].valid()) {
                    if (cudaDeviceEnablePeerAccess(1, 0) == cudaSuccess) {
                        m.cache_peer = true;
                        std::fprintf(stderr,
                                     "strata-dsv4: expert cache: device 1: %lld slots (%.1f%%, %.1f GiB), peer access ON\n",
                                     (long long) m.cache[1].slots(),
                                     100.0 * m.cache[1].slots() / (g.n_layers * g.n_expert),
                                     m.cache[1].gib());
                    } else {
                        m.cache[1].close();
                        std::fprintf(stderr, "strata-dsv4: expert cache: peer access refused; device 1 left out\n");
                    }
                }
            } else {
                std::fprintf(stderr, "strata-dsv4: expert cache: no peer path to device 1; single-card tier\n");
            }
        }
    }
    // ---- [S7b] heat-driven admission: prefill both caches with the routing profile's hottest
    // experts (the golden-workload working set, 4266 pairs, fits the 6326 slots) and seed the
    // online heat table from the ranking.  Default <pack>/expert-profile.bin;
    // STRATA_DSV4_EXPERT_PROFILE overrides, STRATA_DSV4_PROFILE=0 opts out.  The prefill only
    // warms the cache - compute is unchanged (a resident expert's bytes are identical either way).
    if (m.cache[0].valid()) {
        const char* pv = std::getenv("STRATA_DSV4_PROFILE");
        const bool prof_on = pv == nullptr || std::atoi(pv) != 0;
        const char* penv = std::getenv("STRATA_DSV4_EXPERT_PROFILE");
        const std::string prof = penv != nullptr ? penv : o.pack + "/expert-profile.bin";
        if (prof_on && std::filesystem::exists(prof)) {
            std::vector<std::pair<int32_t, int32_t>> ranked;
            int64_t pslots = 0;
            if (!strata::core::read_expert_profile(prof, g.n_layers, g.n_expert, ranked, pslots, err))
                return false;
            const auto pf0 = Clock::now();
            int64_t filled[2] = {0, 0};
            for (int d = 0; d < 2; ++d) {
                if (!m.cache[d].valid()) continue;
                CK(cudaSetDevice(d));
                for (const auto& pr : ranked) {
                    const int32_t l = pr.first, e = pr.second;
                    if ((e & 1) != d) continue;
                    const int32_t slot = m.cache[d].admit(l, e);
                    if (slot < 0) continue;   // this layer's per-layer quota is full
                    const uint8_t* blob = m.experts.blob_stable(l, e);
                    if (!blob) {
                        err = "dsv4: the profile names an expert with no blob";
                        return false;
                    }
                    if (!m.cache[d].fill_slot_queued(slot, blob, err,
                                                     (int64_t) lay.blob_bytes(l)))
                        return false;
                    ++filled[d];
                }
                if (!m.cache[d].sync_queued(err)) return false;
                CK(cudaSetDevice(0));
            }
            // seed the online heat: rank 0 is hottest, the unranked stay cold.
            // [A1] STRATA_DSV4_EVICT_PRIOR (0..100, default 0) scales that seed.  It is a PREDICTION of
            // how good each profiled expert will be, and a large one drowns the online counts for the
            // 4266 profiled pairs, so the eviction gate almost never fires on them.  Measured on the
            // 4k route trace: prior 100 = 72.3% true hit, prior 0 = 82.7%, same slots, same policy.
            const char* pr = std::getenv("STRATA_DSV4_EVICT_PRIOR");
            const double prior = pr != nullptr ? std::atof(pr) : 0.0;
            m.heat.assign((size_t) (g.n_layers * g.n_expert), 0.0f);
            const double R = (double) ranked.size();
            for (size_t r = 0; r < ranked.size(); ++r)
                m.heat[(size_t) (ranked[r].first * g.n_expert + ranked[r].second)] =
                    (float) (prior * (R - (double) r) / R);
            std::fprintf(stderr,
                         "strata-dsv4: expert profile %s: prefilled %lld + %lld slots in %.1f s, heat "
                         "seeded (prior %.0f), eviction scope %s\n",
                         prof.c_str(), (long long) filled[0], (long long) filled[1],
                         ms_since(pf0) / 1000.0, prior, evict_global ? "global" : "layer");
        }
    }
    std::fprintf(stderr, "strata-dsv4: load complete in %.1f s\n", ms_since(t0) / 1000.0);
    return true;
}

// ============================================================================== [S0] the residency report

void print_residency(const Model& m, int64_t ctx) {
    size_t free0 = 0, total0 = 0;
    CK(cudaMemGetInfo(&free0, &total0));   // the load device (0): after dense + native + KV
    const strata::kernels::cpu::ExpertLayout& lay = strata::kernels::cpu::expert_layout();
    const uint64_t reserve = 700ull << 20;   // generate.cpp's auto-cache reserve rule (S7 wires the cache itself)
    const uint64_t slots0 = lay.max_blob && free0 > reserve ? (free0 - reserve) / lay.max_blob : 0;
    const uint64_t experts_total = (uint64_t) m.g.n_layers * (uint64_t) lay.n_expert;
    std::fprintf(stderr,
                 "strata-dsv4: residency: device 0 holds dense %.2f GiB + native %.2f GiB + KV %.2f GiB; "
                 "free %.2f of %.2f GiB -> up to %llu expert slots (%.1f%% of %llu)\n",
                 (double) m.dense_arena_bytes / 1073741824.0, (double) m.native_bytes / 1073741824.0,
                 (double) m.kv_arena_bytes / 1073741824.0, (double) free0 / 1073741824.0,
                 (double) total0 / 1073741824.0, (unsigned long long) slots0,
                 experts_total ? 100.0 * (double) slots0 / (double) experts_total : 0.0,
                 (unsigned long long) experts_total);
    int devices = 0;
    CK(cudaGetDeviceCount(&devices));
    for (int d = 0; d < devices; ++d) {
        cudaDeviceProp prop{};
        CK(cudaGetDeviceProperties(&prop, d));
        if (d == 0) continue;
        CK(cudaSetDevice(d));
        size_t fr = 0, tot = 0;
        CK(cudaMemGetInfo(&fr, &tot));
        const uint64_t slots = lay.max_blob && fr > reserve ? (fr - reserve) / lay.max_blob : 0;
        std::fprintf(stderr, "strata-dsv4: residency: device %d (%s) untouched, free %.2f of %.2f GiB -> up to "
                             "%llu expert slots (S7 splits the cache)\n",
                     d, prop.name, (double) fr / 1073741824.0, (double) tot / 1073741824.0,
                     (unsigned long long) slots);
    }
    CK(cudaSetDevice(0));
    (void) ctx;
}

}  // namespace

// ============================================================================================ [S1] the tail
// The S1 GPU forward: embed (iq_embed_rows, reused) -> 43 identity layers (S2-S5 fill them) -> hc_head (the
// one new kernel: the sigmoid-gated 4-stream weighted sum, dsv4ref.cpp:472 is the verified host original) ->
// output_norm (rms_norm_weighted, reused) -> head (native_quantize_q8_1 + native_mmvq, the q36 production
// path) -> logits.  All accumulation FP32 (double where dsv4ref uses double: the hc_head reductions).
// hc_pre/hc_post/sinkhorn20 belong to the LAYERS and arrive with S2; the tail itself has no sinkhorn.
//
// The replay/dump infrastructure mirrors dsv4ref's contract exactly so tools/dsv4/compare.py and
// compare_layers.py consume the engine's dumps unchanged:
//   dump:   --dsv4-dump-dir DIR  ->  DIR/<pass>/{embed,L00_post_attn_{main,hc},...,hc_head_out,final_norm,
//           logits}.bin, raw f32 row-major, same names/shapes as dsv4ref's dump_bin (dsv4ref.cpp:338).
//   replay: --dsv4-replay DIR    ->  before layer L, if DIR/<pass>_L%02d_in_hc.bin exists it is uploaded over
//           the 4-stream state h (dsv4ref.cpp:1262-1278 semantics; the _in_main.bin side is read back only
//           for a consistency note - the engine's main stream is the mean of h, not a separate tensor).
// Because S1's layers are the identity, a replay injection at L00 must reappear BIT-EXACT in L00's post_attn/
// post_moe dumps - the replay path's self-proof.  Layer-level numeric acceptance against the golden post_*
// tensors is S2-S5's business (the golden tensors went through the real layers).

namespace {

constexpr int64_t kGoldenPrompt[] = {0,     128803, 21750, 260, 15255, 2019, 396,
                                     16803, 1234,   3737,  16,  128804, 128822};

std::vector<int64_t> parse_prompt_ids(const std::string& s) {
    if (s.empty()) return std::vector<int64_t>(std::begin(kGoldenPrompt), std::end(kGoldenPrompt));
    std::vector<int64_t> ids;
    std::stringstream ss(s);
    std::string tok;
    while (std::getline(ss, tok, ',')) {
        if (tok.empty()) continue;
        ids.push_back(std::atoll(tok.c_str()));
    }
    return ids;
}

/// Raw f32 dump, dsv4ref.cpp:338's exact contract (row-major, no header) so the compare tools read it as-is.
void dump_bin(const std::string& path, const float* p, int64_t n) {
    FILE* f = std::fopen(path.c_str(), "wb");
    if (!f) {
        std::fprintf(stderr, "strata-dsv4: cannot write %s\n", path.c_str());
        std::exit(2);
    }
    if (std::fwrite(p, sizeof(float), (size_t) n, f) != (size_t) n) {
        std::fprintf(stderr, "strata-dsv4: short write %s\n", path.c_str());
        std::exit(2);
    }
    std::fclose(f);
}

void dump_dev(const std::string& path, const float* dev, int64_t n, std::vector<float>& host) {
    host.resize((size_t) n);
    CK(cudaMemcpy(host.data(), dev, (size_t) n * sizeof(float), cudaMemcpyDeviceToHost));
    dump_bin(path, host.data(), n);
}

/// h.mean(dim=1) over the 4 HC streams, dsv4ref.cpp:1316-1322's double accumulate (the golden *_main
/// definition, MANIFEST "HC 4-stream representation").  Computed on the host copy so a replay-injected h
/// reproduces dsv4ref's *_main dumps bit-for-bit.
void hc_main_of(const std::vector<float>& h, int64_t s, int64_t n, std::vector<float>& main_out) {
    main_out.resize((size_t) (s * n));
    for (int64_t j = 0; j < s; ++j)
        for (int64_t i = 0; i < n; ++i) {
            double acc = 0;
            for (int c = 0; c < 4; ++c) acc += h[(size_t) ((j * 4 + c) * n + i)];
            main_out[(size_t) (j * n + i)] = (float) (acc / 4.0);
        }
}

/// hc_head (dsv4ref.cpp:472, the global output_hc_* parameters): per token,
///   rsqrt = 1/sqrt(mean(x^2) + rms_eps) over the flattened 4*n (MANIFEST trap 18),
///   p_c   = sigmoid(dot(fn_c, x) * rsqrt * scale[0] + base[c]) + hc_eps,
///   y[i]  = sum_c p_c * x[c*n + i].
/// One block per token; the reductions accumulate in double because dsv4ref's do (sq and the dots through
/// dot_fp32 are the compare reference for the 1e-6 band).
__global__ void dsv4_hc_head_kernel(const float* __restrict__ x, const float* __restrict__ fn,
                                    const float* __restrict__ base, const float* __restrict__ scale,
                                    int64_t n, double rms_eps, float hc_eps, float* __restrict__ y) {
    constexpr int kThreads = 256;
    const int64_t n4 = 4 * n;
    const float* xt = x + (int64_t) blockIdx.x * n4;
    float* yt = y + (int64_t) blockIdx.x * n;
    __shared__ double red[kThreads];
    __shared__ float p[4];
    const int tid = threadIdx.x;
    // phase 1: rsqrt over the flattened 4n
    float sq = 0.f;
    for (int64_t i = tid; i < n4; i += kThreads) sq += xt[i] * xt[i];
    red[tid] = (double) sq;
    __syncthreads();
    for (int off = kThreads / 2; off > 0; off >>= 1) {
        if (tid < off) red[tid] += red[tid + off];
        __syncthreads();
    }
    const float rsqrt = (float) (1.0 / sqrt(red[0] / (double) n4 + rms_eps));
    // phase 2: the four gate dots
    for (int c = 0; c < 4; ++c) {
        const float* f = fn + (int64_t) c * n4;
        float acc = 0.f;
        for (int64_t i = tid; i < n4; i += kThreads) acc += f[i] * xt[i];
        red[tid] = (double) acc;
        __syncthreads();
        for (int off = kThreads / 2; off > 0; off >>= 1) {
            if (tid < off) red[tid] += red[tid + off];
            __syncthreads();
        }
        if (tid == 0) {
            const float v = (float) red[0] * rsqrt * scale[0] + base[c];
            p[c] = 1.0f / (1.0f + expf(-v)) + hc_eps;
        }
        __syncthreads();
    }
    // phase 3: the gated 4-stream sum (c-ordered float adds, dsv4ref's order)
    for (int64_t i = tid; i < n; i += kThreads) {
        float acc = 0.f;
        for (int c = 0; c < 4; ++c) acc += p[c] * xt[(int64_t) c * n + i];
        yt[i] = acc;
    }
}

struct TailWeights {
    const NativeTensor* embd = nullptr;    // token_embd.weight, Q8_0 [vocab][n_embd]
    const NativeTensor* head = nullptr;    // output.weight, Q8_0 [vocab][n_embd]
    const float* hc_fn = nullptr;          // output_hc_fn.weight, f32 [4][4*n_embd]
    const float* hc_base = nullptr;        // output_hc_base.weight, f32 [4]
    const float* hc_scale = nullptr;       // output_hc_scale.weight, f32 [1]
    const float* out_norm = nullptr;       // output_norm.weight, f32 [n_embd]
};

bool tail_weights(const Model& m, TailWeights& w, std::string& err) {
    auto native = [&](const char* name) -> const NativeTensor* {
        for (const NativeTensor& t : m.native)
            if (t.name == name) return &t;
        return nullptr;
    };
    auto dense = [&](const char* name) -> const float* {
        const strata::core::WeightRef* r = m.dense.find(name);
        return r ? (const float*) r->data : nullptr;
    };
    w.embd = native("token_embd.weight");
    w.head = native("output.weight");
    w.hc_fn = dense("output_hc_fn.weight");
    w.hc_base = dense("output_hc_base.weight");
    w.hc_scale = dense("output_hc_scale.weight");
    w.out_norm = dense("output_norm.weight");
    if (!w.embd || !w.head || !w.hc_fn || !w.hc_base || !w.hc_scale || !w.out_norm) {
        err = "tail weights missing: embd/head (native) or output_hc_*/output_norm (dense) not found";
        return false;
    }
    if (w.embd->type != strata::dsv4::kGgufQ8_0 || w.head->type != strata::dsv4::kGgufQ8_0) {
        err = "token_embd/output are not Q8_0 in this pack";
        return false;
    }
    return true;
}

// ==== [S2] MoE wiring: the per-layer weight gather for dsv4_moe_half (src/dsv4/dsv4_moe.cu).  The layout
// table's mutual exclusion holds here: hash layers 0-2 carry ffn_gate_tid2eid (and no exp_probs_b), routed
// layers the reverse - dsv4_moe_half refuses a mismatch.
bool moe_layer_weights(const Model& m, int64_t layer, strata::dsv4::Dsv4MoeLayer& w, std::string& err) {
    const std::string p = "blk." + std::to_string(layer) + ".";
    auto native = [&](const char* suffix) -> const NativeTensor* {
        const std::string name = p + suffix;
        for (const NativeTensor& t : m.native)
            if (t.name == name) return &t;
        return nullptr;
    };
    auto dense = [&](const char* suffix) -> const float* {
        const strata::core::WeightRef* r = m.dense.find(p + suffix);
        return r ? (const float*) r->data : nullptr;
    };
    w.ffn_norm = dense("ffn_norm.weight");
    w.gate_w = (const uint16_t*) dense("ffn_gate_inp.weight");   // raw bf16 in the arena (kind 4)
    w.gate_b = dense("exp_probs_b.bias");   // absent (null) on the hash layers 0-2 (layer 2 is hash-side)
    w.hc_fn = dense("hc_ffn_fn.weight");
    w.hc_base = dense("hc_ffn_base.weight");
    w.hc_scale = dense("hc_ffn_scale.weight");
    if (const NativeTensor* t = native("ffn_gate_tid2eid.weight")) w.tid2eid = (const int32_t*) t->dev;
    const NativeTensor* sg = native("ffn_gate_shexp.weight");
    const NativeTensor* su = native("ffn_up_shexp.weight");
    const NativeTensor* sd = native("ffn_down_shexp.weight");
    if (sg && su && sd) {
        w.sh_gate = sg->dev;
        w.sh_up = su->dev;
        w.sh_down = sd->dev;
        w.sh_type = sg->type;
    }
    if (!w.ffn_norm || !w.gate_w || !w.hc_fn || !w.hc_base || !w.hc_scale || !sg || !su || !sd) {
        err = "S2: layer " + std::to_string(layer) +
              " MoE weights missing (ffn_norm / ffn_gate_inp / hc_ffn_* / *_shexp)";
        return false;
    }
    if (sg->type != strata::dsv4::kGgufQ8_0 || su->type != strata::dsv4::kGgufQ8_0 ||
        sd->type != strata::dsv4::kGgufQ8_0) {
        err = "S2: the shared expert is not Q8_0 in this pack";
        return false;
    }
    return true;
}

// ==== [S3/S4] attention wiring: rope tables, the per-layer weight gather, the KV arena carve.  The rope
// builder is dsv4ref.cpp:359's rope_build verbatim (double-accumulated tables, YaRN correction when
// orig > 0); dsv4_attn_test.cpp carries the same code and the S3 gates validated it.

struct HostRope {
    int64_t half = 32;
    std::vector<float> cosv, sinv;
};

HostRope rope_build_host(int64_t dim, int64_t maxpos, int64_t orig, double base, double factor,
                         double beta_fast, double beta_slow) {
    HostRope r;
    r.half = dim / 2;
    std::vector<double> f((size_t) r.half);
    for (int64_t i = 0; i < r.half; ++i)
        f[(size_t) i] = std::pow(base, -static_cast<double>(2 * i) / dim);
    if (orig > 0) {
        auto fcd = [&](double nr) {
            return static_cast<double>(dim) *
                   std::log(static_cast<double>(orig) / (nr * 2.0 * M_PI)) /
                   (2.0 * std::log(base));
        };
        const int64_t low = std::max<int64_t>(0, (int64_t) std::floor(fcd(beta_fast)));
        const int64_t high = std::min<int64_t>(dim - 1, (int64_t) std::ceil(fcd(beta_slow)));
        for (int64_t i = 0; i < r.half; ++i) {
            double ramp = (static_cast<double>(i) - low) / static_cast<double>(high - low);
            ramp = std::min(1.0, std::max(0.0, ramp));
            const double smooth = 1.0 - ramp;
            f[(size_t) i] = f[(size_t) i] / factor * (1.0 - smooth) + f[(size_t) i] * smooth;
        }
    }
    r.cosv.resize((size_t) maxpos * (size_t) r.half);
    r.sinv.resize((size_t) maxpos * (size_t) r.half);
    for (int64_t p = 0; p < maxpos; ++p)
        for (int64_t i = 0; i < r.half; ++i) {
            const double ang = static_cast<double>(p) * f[(size_t) i];
            r.cosv[(size_t) (p * r.half + i)] = static_cast<float>(std::cos(ang));
            r.sinv[(size_t) (p * r.half + i)] = static_cast<float>(std::sin(ang));
        }
    return r;
}

struct DevRope {
    float* cos = nullptr;
    float* sin = nullptr;
    strata::dsv4::Dsv4RopeTables table() const {
        strata::dsv4::Dsv4RopeTables t;
        t.cos = cos;
        t.sin = sin;
        return t;
    }
    void free() {
        cudaFree(cos);
        cudaFree(sin);
        cos = sin = nullptr;
    }
};

bool rope_upload(const HostRope& h, DevRope& d, std::string& err) {
    const size_t bytes = h.cosv.size() * 4;
    if (cudaMalloc(&d.cos, bytes) != cudaSuccess || cudaMalloc(&d.sin, bytes) != cudaSuccess) {
        err = "rope table allocation failed";
        return false;
    }
    CK(cudaMemcpy(d.cos, h.cosv.data(), bytes, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d.sin, h.sinv.data(), bytes, cudaMemcpyHostToDevice));
    return true;
}

/// One layer's attention-half weights (hc attn side + block + compressor).  Q8_0 mats are the pack's
/// native rows; norms/sinks/hc are the dense arena's F32 tensors; the compressor projections are the
/// dense arena's raw BF16 payloads (dsv4ref reads the same GGUF rows as BF16).
bool attn_half_weights(const Model& m, int64_t layer, strata::dsv4::Dsv4AttnHalfWeights& w,
                       std::string& err) {
    const std::string p = "blk." + std::to_string(layer) + ".";
    auto native = [&](const char* suffix) -> const NativeTensor* {
        const std::string name = p + suffix;
        for (const NativeTensor& t : m.native)
            if (t.name == name) return &t;
        return nullptr;
    };
    auto dense = [&](const char* suffix) -> const float* {
        const strata::core::WeightRef* r = m.dense.find(p + suffix);
        return r ? (const float*) r->data : nullptr;
    };
    w.hc_fn = dense("hc_attn_fn.weight");
    w.hc_base = dense("hc_attn_base.weight");
    w.hc_scale = dense("hc_attn_scale.weight");
    w.attn_norm = dense("attn_norm.weight");
    const NativeTensor* qa = native("attn_q_a.weight");
    const NativeTensor* qb = native("attn_q_b.weight");
    const NativeTensor* kv = native("attn_kv.weight");
    const NativeTensor* oa = native("attn_output_a.weight");
    const NativeTensor* ob = native("attn_output_b.weight");
    w.attn.q_norm = dense("attn_q_a_norm.weight");
    w.attn.kv_norm = dense("attn_kv_a_norm.weight");
    w.attn.sinks = dense("attn_sinks.weight");
    if (qa && qb && kv && oa && ob) {
        w.attn.wq_a = qa->dev;
        w.attn.wq_b = qb->dev;
        w.attn.wkv = kv->dev;
        w.attn.wo_a = oa->dev;
        w.attn.wo_b = ob->dev;
    }
    w.comp.wkv = (const uint16_t*) dense("attn_compressor_kv.weight");
    w.comp.wgate = (const uint16_t*) dense("attn_compressor_gate.weight");
    w.comp.ape = dense("attn_compressor_ape.weight");
    w.comp.norm = dense("attn_compressor_norm.weight");
    // [S5] the r=4 layers' indexer (tensor names per layout.hpp dsv4_indexer_tensors, errata #16):
    // proj/kv/gate are the dense arena's RAW BF16 payloads (errata #9: 2 B/elem, convert in-kernel),
    // attn_q_b is the pack's native Q8_0.
    w.idx.proj = (const uint16_t*) dense("indexer.proj.weight");
    const NativeTensor* iqb = native("indexer.attn_q_b.weight");
    w.idx.comp.wkv = (const uint16_t*) dense("indexer_compressor_kv.weight");
    w.idx.comp.wgate = (const uint16_t*) dense("indexer_compressor_gate.weight");
    w.idx.comp.ape = dense("indexer_compressor_ape.weight");
    w.idx.comp.norm = dense("indexer_compressor_norm.weight");
    const int64_t ratio = strata::dsv4::dsv4_compress_ratio(layer);
    if (!w.hc_fn || !w.hc_base || !w.hc_scale || !w.attn_norm) {
        err = "S5: layer " + std::to_string(layer) + " hc_attn/attn_norm weights missing";
        return false;
    }
    // S5 runs the full attention on every layer class (hash / r=128 / r=4 indexer alike)
    if (!qa || !qb || !kv || !oa || !ob || !w.attn.q_norm || !w.attn.kv_norm || !w.attn.sinks) {
        err = "S5: layer " + std::to_string(layer) + " attention weights missing";
        return false;
    }
    if (qa->type != strata::dsv4::kGgufQ8_0 || qb->type != strata::dsv4::kGgufQ8_0 ||
        kv->type != strata::dsv4::kGgufQ8_0 || oa->type != strata::dsv4::kGgufQ8_0 ||
        ob->type != strata::dsv4::kGgufQ8_0) {
        err = "S5: layer " + std::to_string(layer) + " attention mats are not Q8_0 in this pack";
        return false;
    }
    if (ratio > 0 && (!w.comp.wkv || !w.comp.wgate || !w.comp.ape || !w.comp.norm)) {
        err = "S5: layer " + std::to_string(layer) + " compressor weights missing";
        return false;
    }
    if (ratio == 4) {
        if (!w.idx.proj || !iqb || !w.idx.comp.wkv || !w.idx.comp.wgate || !w.idx.comp.ape ||
            !w.idx.comp.norm) {
            err = "S5: layer " + std::to_string(layer) + " indexer weights missing";
            return false;
        }
        if (iqb->type != strata::dsv4::kGgufQ8_0) {
            err = "S5: layer " + std::to_string(layer) + " indexer.attn_q_b is not Q8_0 in this pack";
            return false;
        }
        w.idx.wq_b = iqb->dev;
    }
    return true;
}

/// The S1..S5 forward: the prefill pass plus --dsv4-decode single-token decode passes.  Returns false
/// with `err` set on a hard failure; a NaN anywhere is fatal (dsv4ref's gate_nan contract, so a poisoned
/// stage names itself instead of the stage it poisoned).
bool forward_tail(const Options& o, Model& m, std::string& err) {
    const strata::dsv4::Dsv4Geometry& g = m.g;
    const int64_t N = g.n_embd, V = g.n_vocab;
    TailWeights w;
    if (!tail_weights(m, w, err)) return false;
    const std::vector<int64_t> ids = parse_prompt_ids(o.prompt_ids);
    const int64_t s = (int64_t) ids.size();
    if (s == 0) { err = "empty --dsv4-prompt-ids"; return false; }
    // [G1] sequence capacity: prompt + appended tokens (+ a multi-row verify batch, or the R rows each
    // of the --dsv4-mrow-bench iterations appends)
    int64_t s_all = s + std::max(0, o.decode) + std::max(0, o.verify_rows) +
                    (int64_t) std::max(0, o.mrow_bench) * std::max(0, o.decode);
    // [R-06] every pass is at most this many rows wide: the prefill pass is s rows, the verify pass is
    // --dsv4-verify-rows, each benchmark pass is --dsv4-mrow-bench.  The per-pass device buffers and
    // workspaces are sized for the WIDEST pass, not for the prompt - a short prompt with a large R used
    // to size them for s rows and run_pass then wrote s_cur = R rows into them (device OOB).
    // [chunk] a chunked prefill never runs a pass wider than its chunk (nor the ring), so the widest
    // pass is the chunk, not the prompt: sizing for the prompt would allocate s x V logits and an
    // s-row workspace, which a 4k+ prompt cannot fit next to the expert cache.
    int64_t chunk_w = o.prefill_chunk;
    if (chunk_w > 0 && chunk_w > g.sliding_window) {
        std::fprintf(stderr,
                     "strata-dsv4: --dsv4-prefill-chunk %lld exceeds the sliding window %lld (a pass "
                     "wider than the ring has no defined ring shift); using %lld\n",
                     (long long) chunk_w, (long long) g.sliding_window, (long long) g.sliding_window);
        chunk_w = g.sliding_window;
    }
    const int64_t s_ws = std::max<int64_t>(chunk_w > 0 ? chunk_w : s,
                                           std::max<int64_t>(std::max<int64_t>(0, o.verify_rows),
                                                             std::max<int64_t>(0, o.mrow_bench)));
    s_all = std::max(s_all, s_ws);
    std::fprintf(stderr, "strata-dsv4: S5 forward: prefill pass %s, %lld tokens (13-token golden "
                         "prompt%s), embed -> %lld layers (hash / r=128 / r=4 indexer attention and MoE "
                         "all real) -> hc_head -> output_norm -> head%s\n",
                 o.pass_name.c_str(), (long long) s,
                 o.prompt_ids.empty() ? "" : " overridden by --dsv4-prompt-ids", (long long) g.n_layers,
                 o.decode > 0 ? ", then --dsv4-decode steps" : "");

    // device buffers; the native_mmvq family requires an explicit non-null stream.  Everything but the
    // token list is sized for the widest pass (s_ws rows, [R-06]); decode passes run one token through it.
    cudaStream_t stream = nullptr;
    CK(cudaStreamCreate(&stream));
    int32_t* d_tokens = nullptr;
    float *d_emb = nullptr, *d_h = nullptr, *d_y = nullptr, *d_logits = nullptr;
    void* d_q8 = nullptr;
    CK(cudaMalloc(&d_tokens, (size_t) s_all * sizeof(int32_t)));
    CK(cudaMalloc(&d_emb, (size_t) (s_ws * N) * 4));
    CK(cudaMalloc(&d_h, (size_t) (s_ws * 4 * N) * 4));
    CK(cudaMalloc(&d_y, (size_t) (s_ws * N) * 4));
    CK(cudaMalloc(&d_q8, (size_t) (s_ws * (N / 32) * 36)));   // q8_1: 36 B per 32 values
    CK(cudaMalloc(&d_logits, (size_t) (s_ws * V) * 4));
    std::vector<int32_t> ids32((size_t) s);
    for (int64_t j = 0; j < s; ++j) ids32[(size_t) j] = (int32_t) ids[(size_t) j];
    CK(cudaMemcpy(d_tokens, ids32.data(), (size_t) s * sizeof(int32_t), cudaMemcpyHostToDevice));
    std::vector<float> host, main_out;
    const auto t0 = Clock::now();
    // [S2] the MoE workspace (the layer loop below runs the real MoE half through it)
    strata::dsv4::Dsv4MoeWs* moe_ws = strata::dsv4::dsv4_moe_ws_new(g, s_ws, err);
    if (!moe_ws) return false;
    strata::dsv4::dsv4_moe_ws_set_layers(moe_ws, (int) g.n_layers);   // [S7a-2] warm wrap
    strata::dsv4::dsv4_moe_ws_set_cache(moe_ws,
                                        m.cache[0].valid() ? &m.cache[0] : nullptr,
                                        m.cache[1].valid() ? &m.cache[1] : nullptr);   // [S7a-3]
    if (!m.heat.empty())
        strata::dsv4::dsv4_moe_ws_set_heat(moe_ws, m.heat.data(), (int64_t) m.heat.size());  // [S7b]

    // [S3/S4] attention side: rope tables (hash layers: plain theta 1e4; compress layers: YaRN
    // theta 160000, factor 16, orig 65536, beta 32/1), the shared workspaces, and the per-layer
    // compressor states.  All prefill-mode sizing (start_pos == 0, s tokens).
    DevRope rope_plain, rope_yarn;
    if (!rope_upload(rope_build_host(g.n_rot, o.ctx, 0, g.rope_theta, 0, 0, 0), rope_plain, err))
        return false;
    if (!rope_upload(rope_build_host(g.n_rot, o.ctx, g.yarn_orig_ctx, g.compress_rope_theta,
                                     g.yarn_factor, g.yarn_beta_fast, g.yarn_beta_slow),
                     rope_yarn, err))
        return false;
    strata::dsv4::Dsv4AttnWorkspace attn_ws;
    if (!strata::dsv4::dsv4_attn_workspace_init(attn_ws, (int) s_ws, (int) g.sliding_window)) {
        err = "S4: attention workspace allocation failed";
        return false;
    }
    // compressor workspace: W covers coff*d (attention r=4: 2*512; indexer: 2*128), topk rows cover the
    // window + the widest compressed section.  Decode's indexer topk rows are win + min(512, nc) wide, so
    // max_comp must reach idx_top_k (512), not just the prefill blocks (s/4 + 1).
    const int64_t max_comp = std::max<int64_t>(std::max<int64_t>(1, s_ws / 4 + 1), 512);
    // [R-07] the r=128 layers see win + (start_pos+s)/128 compressed slots once the context is full, and
    // the mask upload (dsv4_comp.cu:1060) copies s*(win+kc) int32 into cws.topk, which is sized
    // max_s*(win+max_comp).  A context wider than max_comp*128 therefore writes past that buffer on the
    // first pass that reaches the end of the context; refuse the configuration here instead of silently
    // clipping the context.
    const int64_t kc_ctx = (o.ctx + 127) / 128;
    if (kc_ctx > max_comp) {
        err = "--ctx " + std::to_string((long long) o.ctx) + " needs " +
              std::to_string((long long) kc_ctx) +
              " compressed slots per r=128 layer (ctx/128) but the workspace holds " +
              std::to_string((long long) max_comp) + "; maximum supported --ctx is " +
              std::to_string((long long) (max_comp * 128));
        return false;
    }
    strata::dsv4::Dsv4CompWorkspace comp_ws;
    if (!strata::dsv4::dsv4_comp_workspace_init(comp_ws, (int) s_ws, 2048, max_comp,
                                                (int) g.sliding_window, N, 8)) {
        err = "S5: compressor workspace allocation failed";
        return false;
    }
    // [S5] indexer scratch: q [s][64*128] + wgt [s][64], downloaded once per layer-pass for the host-side
    // EXACT top-512 selection
    strata::dsv4::Dsv4IndexerWorkspace idx_ws;
    if (!strata::dsv4::dsv4_indexer_workspace_init(idx_ws, s_ws)) {
        err = "S5: indexer workspace allocation failed";
        return false;
    }
    std::vector<strata::dsv4::Dsv4CompState> comp_states((size_t) g.n_layers);
    std::vector<strata::dsv4::Dsv4CompState> idx_states((size_t) g.n_layers);
    for (int64_t l = 0; l < g.n_layers; ++l) {
        const int64_t ratio = strata::dsv4::dsv4_compress_ratio(l);
        if (ratio > 0 &&
            !strata::dsv4::dsv4_comp_state_init(comp_states[(size_t) l], ratio, g.kv_lora_rank,
                                                ratio == 4 ? 2 : 1, stream)) {
            err = "S5: compressor state allocation failed";
            return false;
        }
        // the indexer's own compressor: ratio 4, d = idx_key_dim (128), coff 2 (dsv4ref load_layer :586)
        if (strata::dsv4::dsv4_has_indexer(l) &&
            !strata::dsv4::dsv4_comp_state_init(idx_states[(size_t) l], 4, g.idx_key_dim, 2, stream)) {
            err = "S5: indexer compressor state allocation failed";
            return false;
        }
    }
    // KV arena carve: the layout of kv_arena_bytes() - per-layer kv slices in layer order, then the
    // indexer key section in layer order, then the workspace reservation.
    std::vector<float*> kv_ptr((size_t) g.n_layers, nullptr);
    std::vector<float*> idx_ptr((size_t) g.n_layers, nullptr);
    {
        float* base = (float*) m.kv_arena;
        uint64_t off = 0;
        for (int64_t l = 0; l < g.n_layers; ++l) {
            const int64_t ratio = strata::dsv4::dsv4_compress_ratio(l);
            const int64_t slots = ratio == 0 ? o.ctx : g.sliding_window + o.ctx / ratio;
            kv_ptr[(size_t) l] = base + off;
            off += (uint64_t) slots * (uint64_t) g.kv_lora_rank;
        }
        for (int64_t l = 0; l < g.n_layers; ++l) {
            if (!strata::dsv4::dsv4_has_indexer(l)) continue;
            idx_ptr[(size_t) l] = base + off;
            off += (uint64_t) (o.ctx / 4) * (uint64_t) g.idx_key_dim;
        }
    }

    // ---- one pass over the stack: embed -> layers -> tail -> last-row argmax (-1 with err set on a hard
    // failure).  pass names the dump/replay file prefix (prefill / decode1..N); start_pos + s_cur follow
    // dsv4ref run_pass's decode contract (the ring window offset win = 128 semantics, dsv4ref :850).
    // [S7b] engine-level phase timing (STRATA_DSV4_TIME): decode-time split attention / MoE / tail
    const bool eton = std::getenv("STRATA_DSV4_TIME") != nullptr;
    double etm_embed = 0, etm_attn = 0, etm_moe = 0, etm_tail = 0, etm_attn_gpu = 0;
    double etm_attn_cls[3] = {0, 0, 0};   // hash / r=128 / r=4+indexer
    int64_t etm_attn_clsn[3] = {0, 0, 0};
    int64_t etm_passes = 0;
    cudaEvent_t ea0 = nullptr, ea1 = nullptr;
    if (eton) {
        if (cudaEventCreate(&ea0) != cudaSuccess) ea0 = nullptr;
        if (cudaEventCreate(&ea1) != cudaSuccess) ea1 = nullptr;
    }
    auto run_pass = [&](const std::string& pass, int64_t start_pos, int64_t s_cur,
                        const int32_t* batch_ids, bool chunk, bool dump) -> int64_t {
        // [R-06] the workspaces and the per-pass device buffers are sized for s_ws rows: reject a pass
        // that is wider (clean error, not a device OOB write / abort)
        if (s_cur <= 0 || s_cur > s_ws) {
            char msg[256];
            std::snprintf(msg, sizeof msg,
                          "pass %s: %lld rows exceeds the workspace width %lld",
                          pass.c_str(), (long long) s_cur, (long long) s_ws);
            err = msg;
            return -1;
        }
        // [R-02] start_pos + s_cur must fit inside --ctx: the rope tables and the KV arena are sized for
        // ctx positions, and every layer-pass would read past both (clean reject, not an OOB write)
        if (start_pos < 0 || start_pos + s_cur > o.ctx) {
            char msg[256];
            std::snprintf(msg, sizeof msg,
                          "pass %s: positions %lld..%lld exceed --ctx %lld",
                          pass.c_str(), (long long) start_pos,
                          (long long) (start_pos + s_cur - 1), (long long) o.ctx);
            err = msg;
            return -1;
        }
        // [R-03] token ids become embedding-table / tid2eid row indices, so they must be real vocabulary
        // indices before anything is uploaded (clean reject, not an OOB read)
        for (int64_t j = 0; j < s_cur; ++j) {
            const int32_t t = batch_ids[(size_t) j];
            if (t < 0 || (int64_t) t >= V) {
                char msg[256];
                std::snprintf(msg, sizeof msg,
                              "pass %s: token id %lld at position %lld is outside [0, %lld)",
                              pass.c_str(), (long long) t, (long long) (start_pos + j),
                              (long long) V);
                err = msg;
                return -1;
            }
        }
        const auto pt0 = Clock::now();  // [S7d diag] pre-loop segment
        const std::string dir =
            (o.dump_dir.empty() || !dump) ? std::string() : o.dump_dir + "/" + pass;
        if (!dir.empty()) std::filesystem::create_directories(dir);
        // ---- embed (iq_embed_rows: the Q8_0 table's rows gathered + dequantized on the GPU)
        const size_t row_bytes = (size_t) (w.embd->bytes / (uint64_t) w.embd->ne1);
        strata::kernels::iq_embed_rows((int) w.embd->type, w.embd->dev, row_bytes, d_tokens + start_pos,
                                       (int) s_cur, N, d_emb, stream);
        if (!dir.empty()) dump_dev(dir + "/embed.bin", d_emb, s_cur * N, host);
        // embed -> 4 identical streams (MANIFEST: the [s, 4, n] HC state starts as 4 copies)
        for (int64_t j = 0; j < s_cur; ++j)
        for (int c = 0; c < 4; ++c)
            CK(cudaMemcpyAsync(d_h + ((int64_t) j * 4 + c) * N, d_emb + j * N, (size_t) N * 4,
                               cudaMemcpyDeviceToDevice, stream));
    CK(cudaDeviceSynchronize());
        if (std::getenv("STRATA_DSV4_LAYERTIME") != nullptr)
            std::fprintf(stderr, "layertime %s pre %.3f ms\n", pass.c_str(), ms_since(pt0));

        // ---- 43 layers: the attention half is real on every layer class (S3 hash, S4 r=128, S5 r=4
        // indexer), and the MoE half is real (S2).  Replay + dump
        // wire here so every later slice inherits the acceptance gate S1 proved: the layer's input state
        // comes from the replay pack, post_attn is dumped from it, then the MoE half runs and post_moe is
        // dumped.
    for (int64_t layer = 0; layer < g.n_layers; ++layer) {
        if (!o.replay_dir.empty()) {
            char fn[640];
            std::snprintf(fn, sizeof fn, "%s/%s_L%02lld_in_hc.bin", o.replay_dir.c_str(), pass.c_str(),
                          (long long) layer);
            if (FILE* f = std::fopen(fn, "rb")) {
                host.resize((size_t) (s_cur * 4 * N));
                if (std::fread(host.data(), sizeof(float), host.size(), f) != host.size()) {
                    std::fprintf(stderr, "strata-dsv4: replay: short read %s\n", fn);
                    std::exit(2);
                }
                std::fclose(f);
                CK(cudaMemcpy(d_h, host.data(), host.size() * 4, cudaMemcpyHostToDevice));
                // the _in_main side is informational here (the engine's main stream is mean(h)); note the
                // agreement once so a mismatched replay pack is seen at L00, not in S2's numbers
                std::snprintf(fn, sizeof fn, "%s/%s_L%02lld_in_main.bin", o.replay_dir.c_str(),
                              pass.c_str(), (long long) layer);
                if (layer == 0)
                    std::fprintf(stderr, "strata-dsv4: replay: injecting %s/%s_L%%02d_in_hc.bin at every "
                                         "layer that has one\n", o.replay_dir.c_str(), pass.c_str());
            }
        }
        // ---- attention half (src/dsv4/dsv4_comp.cu drives the per-class path):
        //   hash layers 0/1:  hc_pre -> rmsnorm -> dense MLA over the window cache (plain rope)
        //   r=128 (odd 3..41): + compressor (YaRN rope, phase j*128) -> MLA over window + compressed
        //   r=4 (even 2..42, L02 the hash+DSA hybrid included): compressor + indexer top-512 sparse
        //                     attention (S5)
        // [S7d diag] host-slack probe: sleeps below measure how much host time the GPU
        // pipeline can absorb (STRATA_DSV4_HOSTSLEEP_US, default 0 = off).
        {
            static const long hs_us = [] {
                const char* e = std::getenv("STRATA_DSV4_HOSTSLEEP_US");
                return e != nullptr ? std::atol(e) : 0L;
            }();
            if (hs_us > 0)
                std::this_thread::sleep_for(std::chrono::microseconds(hs_us));
        }
        const auto lt0 = Clock::now();  // [S7d diag] per-layer wall split
        // ---- [S7a-2b] predictive read-ahead at the LAYER HEAD: the previous pass's routing for the
        // two layers just ahead goes to the file tier's I/O workers here, before this layer's own
        // attention work is enqueued, so those whole-blob preads overlap the ~43 ms/layer attention GPU
        // wait instead of only the MoE staging behind the route sync.  Pure I/O scheduling: it only
        // enqueues preads on the source's threads (the demand path is untouched), and a wrong
        // prediction just wastes readahead - the numbers are identical either way.
        strata::dsv4::dsv4_moe_prefetch_ahead(moe_ws, m.experts, layer);
        int64_t cls_ratio = -1;
        bool cls_idx = false;
        {
            strata::dsv4::Dsv4AttnHalfWeights aw;
            if (!attn_half_weights(m, layer, aw, err)) return -1;
            const int64_t ratio = strata::dsv4::dsv4_compress_ratio(layer);
            const bool has_idx = strata::dsv4::dsv4_has_indexer(layer);
            cls_ratio = ratio;
            cls_idx = has_idx;
            const auto ta0 = Clock::now();
            if (ea0 != nullptr && ea1 != nullptr) CK(cudaEventRecord(ea0, stream));
            if (!strata::dsv4::dsv4_attn_half(g, layer, aw, rope_plain.table(), rope_yarn.table(), d_h,
                                              kv_ptr[(size_t) layer],
                                              ratio > 0 ? &comp_states[(size_t) layer] : nullptr,
                                              has_idx ? &idx_states[(size_t) layer] : nullptr,
                                              idx_ptr[(size_t) layer], attn_ws, comp_ws, idx_ws, s_cur,
                                              start_pos, stream, err, chunk))
                return -1;
            if (ea0 != nullptr && ea1 != nullptr) CK(cudaEventRecord(ea1, stream));
            if (eton) etm_attn += ms_since(ta0);
        }
        const auto lt1 = Clock::now();  // [S7d diag] after the attention half
        // dump post_attn from the state the attention half left (dsv4ref's dump contract)
        if (!dir.empty()) {
            host.resize((size_t) (s_cur * 4 * N));
            CK(cudaMemcpy(host.data(), d_h, host.size() * 4, cudaMemcpyDeviceToHost));
            hc_main_of(host, s_cur, N, main_out);
            char path[640];
            std::snprintf(path, sizeof path, "%s/L%02lld_post_attn_hc.bin", dir.c_str(),
                          (long long) layer);
            dump_bin(path, host.data(), s_cur * 4 * N);
            std::snprintf(path, sizeof path, "%s/L%02lld_post_attn_main.bin", dir.c_str(),
                          (long long) layer);
            dump_bin(path, main_out.data(), s_cur * N);
        }
        // ---- [S2] MoE half: hc_pre(ffn) -> rmsnorm -> sqrtsoftplus routing -> the NVFP4 routed experts +
        // the Q8_0 shared expert -> the k = 6 combine -> hc_post (src/dsv4/dsv4_moe.cu).
        // S2 acceptance hook (task: "inject the golden layer entry / attn exit"): while the attention half
        // is identity (S3's slice), DIR/<pass>_L%02d_in_moe_hc.bin - when present - overwrites the state
        // with the golden attention output so post_moe isolates the MoE block's own numerics.
        if (!o.replay_dir.empty()) {
            char fn[640];
            std::snprintf(fn, sizeof fn, "%s/%s_L%02lld_in_moe_hc.bin", o.replay_dir.c_str(),
                          pass.c_str(), (long long) layer);
            if (FILE* f = std::fopen(fn, "rb")) {
                host.resize((size_t) (s_cur * 4 * N));
                if (std::fread(host.data(), sizeof(float), host.size(), f) != host.size()) {
                    std::fprintf(stderr, "strata-dsv4: replay: short read %s\n", fn);
                    std::exit(2);
                }
                std::fclose(f);
                CK(cudaMemcpy(d_h, host.data(), host.size() * 4, cudaMemcpyHostToDevice));
            }
        }
        {
            strata::dsv4::Dsv4MoeLayer mw;
            if (!moe_layer_weights(m, layer, mw, err)) return -1;
            // decode: tokens[0] of this pass is the appended token (dsv4ref run_pass's current id)
            const auto em0 = Clock::now();
            if (!strata::dsv4::dsv4_moe_half(g, layer, mw, m.experts, d_h, d_tokens + start_pos, s_cur,
                                             moe_ws, stream, err))
                return -1;
            if (eton) {
                etm_moe += ms_since(em0);
                if (ea0 != nullptr && ea1 != nullptr) {
                    // dsv4_moe_half drains the stream, so both attention events have executed
                    float el = 0;
                    if (cudaEventElapsedTime(&el, ea0, ea1) == cudaSuccess) {
                        etm_attn_gpu += el;
                        // [S7b] per-class attention split: hash / r=128 / r=4+indexer
                        const int cls = cls_ratio <= 0 ? 0 : (cls_idx ? 2 : 1);
                        etm_attn_cls[cls] += el;
                        ++etm_attn_clsn[cls];
                    }
                }
            }
        }
        // dump post_moe from the state the MoE half left
        if (std::getenv("STRATA_DSV4_LAYERTIME") != nullptr &&
            (std::getenv("STRATA_DSV4_LAYERTIME_ALL") != nullptr || layer < 4 ||
             layer == g.n_layers - 1))
            std::fprintf(stderr, "layertime %s L%02lld attn %.3f moe %.3f\n", pass.c_str(),
                         (long long) layer,
                         std::chrono::duration<double, std::milli>(lt1 - lt0).count(),
                         std::chrono::duration<double, std::milli>(Clock::now() - lt1).count());
        if (!dir.empty()) {
            host.resize((size_t) (s_cur * 4 * N));
            CK(cudaMemcpy(host.data(), d_h, host.size() * 4, cudaMemcpyDeviceToHost));
            hc_main_of(host, s_cur, N, main_out);
            char path[640];
            std::snprintf(path, sizeof path, "%s/L%02lld_post_moe_hc.bin", dir.c_str(),
                          (long long) layer);
            dump_bin(path, host.data(), s_cur * 4 * N);
            std::snprintf(path, sizeof path, "%s/L%02lld_post_moe_main.bin", dir.c_str(),
                          (long long) layer);
            dump_bin(path, main_out.data(), s_cur * N);
        }
    }

        // ---- the tail: hc_head -> output_norm -> head -> logits
        const auto et0 = Clock::now();
        dsv4_hc_head_kernel<<<(unsigned) s_cur, 256, 0, stream>>>(d_h, w.hc_fn, w.hc_base, w.hc_scale, N,
                                                                  g.rms_eps, (float) g.hc_eps, d_y);
        CK(cudaGetLastError());
        if (!dir.empty()) dump_dev(dir + "/hc_head_out.bin", d_y, s_cur * N, host);
        strata::kernels::rms_norm_weighted(d_y, w.out_norm, s_cur, N, (float) g.rms_eps, stream);
        if (!dir.empty()) dump_dev(dir + "/final_norm.bin", d_y, s_cur * N, host);
        // native_mmvq takes 1..8 columns per launch: the 13-token prompt goes as 8 + 5
        for (int64_t c0 = 0; c0 < s_cur; c0 += 8) {
            const int nc = (int) std::min<int64_t>(8, s_cur - c0);
            strata::kernels::native_quantize_q8_1(d_y + c0 * N, d_q8, (int) N, nc, stream);
            strata::kernels::native_mmvq((int) w.head->type, w.head->dev, d_q8, d_logits + c0 * V,
                                         (int) N, (int) V, nc, stream);
        }
        CK(cudaGetLastError());
        CK(cudaDeviceSynchronize());
        if (!dir.empty()) dump_dev(dir + "/logits.bin", d_logits, s_cur * V, host);

        // ---- gates: NaN scan over every stage we still hold, logits shape is [s, V] by construction
        host.resize((size_t) (s_cur * V));
        CK(cudaMemcpy(host.data(), d_logits, host.size() * 4, cudaMemcpyDeviceToHost));
        int64_t nans = 0;
        for (float v : host) nans += std::isnan(v) ? 1 : 0;
        if (nans) {
            std::fprintf(stderr, "strata-dsv4: FATAL: %lld NaN in logits [%lld, %lld]\n",
                         (long long) nans, (long long) s_cur, (long long) V);
            err = "NaN in logits";
            return -1;
        }
        const float* last = host.data() + (size_t) (s_cur - 1) * V;
        int64_t best = 0;
        for (int64_t i = 1; i < V; ++i)
            if (last[i] > last[best]) best = i;
        std::fprintf(stderr,
                     "strata-dsv4: S5 pass %s done in %.0f ms: logits [%lld, %lld] no NaN, last-row "
                     "argmax %lld\n",
                     pass.c_str(), ms_since(t0), (long long) s_cur, (long long) V, (long long) best);
        if (start_pos > 0 && s_cur > 1) {
            // [G1] multi-row pass: every row's argmax (row r must predict what the single-row path
            // produced at start_pos + r, which is what the verify/dump comparison checks row by row).
            // [chunk] a chunked prefill pass prints its rows too (chunk c's row r is prompt row
            // start_pos + r, so the per-row argmax list of a chunk run can be laid against the single
            // pass's 128-row list).
            for (int64_t r = 0; r < s_cur; ++r) {
                const float* row = host.data() + (size_t) r * V;
                int64_t b = 0;
                for (int64_t i = 1; i < V; ++i)
                    if (row[i] > row[b]) b = i;
                std::fprintf(stderr, "strata-dsv4: %s row %lld argmax %lld\n", pass.c_str(),
                             (long long) r, (long long) b);
            }
        }
        if (chunk && start_pos == 0 && s_cur > 1) {
            // [chunk] the first chunk is the only prefill-mode pass; mirror the row list above so every
            // chunk reports where its rows land (the gate lays chunk row r against single-pass row
            // start_pos + r).
            for (int64_t r = 0; r < s_cur; ++r) {
                const float* row = host.data() + (size_t) r * V;
                int64_t b = 0;
                for (int64_t i = 1; i < V; ++i)
                    if (row[i] > row[b]) b = i;
                std::fprintf(stderr, "strata-dsv4: %s row %lld argmax %lld (pos %lld)\n", pass.c_str(),
                             (long long) r, (long long) b, (long long) (start_pos + r));
            }
        }
        if (eton) {
            etm_tail += ms_since(et0);
            ++etm_passes;
        }
        return best;
    };

    // ---- the passes: prefill (under the --dsv4-pass name), then --dsv4-decode single-token steps feeding
    // the argmax back (dsv4ref main() :1461: start_pos is the sequence length BEFORE the append, so the
    // appended token is the pass's only row; golden generated ids are 734, 260, 4654)
    //
    // [chunk] --dsv4-prefill-chunk W splits the prompt into W-row passes inside this process: pass i
    // covers positions [i*W, min((i+1)*W, s)) with start_pos = i*W, so it continues the KV ring, the
    // compressor carry states and the expert cache the previous pass left (no process restart, no
    // re-load of the 1.6 GB pinned expert pool).  The LAST pass carries the --dsv4-pass name and its
    // dumps, so a chunked run's dump tree is the single pass's tail rows (the gate compares them).
    // The pass that starts at start_pos == 0 is a prefill-mode pass (bulk pooling); the continuations
    // go through the decode contract (per-row ring writes, block-completion pooling).
    const int64_t n_chunks = chunk_w > 0 ? (s + chunk_w - 1) / chunk_w : 1;
    int64_t cur = 0;
    if (n_chunks <= 1) {
        cur = run_pass(o.pass_name, 0, s, ids32.data(), false, true);
        if (cur < 0) return false;
    } else {
        const auto pf0 = Clock::now();
        int64_t ch0 = 0, cm0 = 0, ev0 = 0;
        strata::dsv4::dsv4_moe_ws_cache_stats(moe_ws, ch0, cm0);
        ev0 = strata::dsv4::dsv4_moe_ws_evictions(moe_ws);
        std::fprintf(stderr,
                     "strata-dsv4: chunked prefill: %lld tokens in %lld passes of %lld rows (sliding "
                     "window %lld)\n",
                     (long long) s, (long long) n_chunks, (long long) chunk_w,
                     (long long) g.sliding_window);
        for (int64_t ci = 0; ci < n_chunks; ++ci) {
            const int64_t off = ci * chunk_w;
            const int64_t sw = std::min<int64_t>(chunk_w, s - off);
            const bool last = ci + 1 == n_chunks;
            char pname[96];
            // non-final passes get their own name and no dump (a chunk's dumps are not a gate artifact;
            // the last pass is the one compared against the single-pass run)
            std::snprintf(pname, sizeof pname, "%s_c%lld", o.pass_name.c_str(), (long long) (ci + 1));
            int64_t ch1 = 0, cm1 = 0, ev1 = 0;
            const auto ck0 = Clock::now();
            cur = run_pass(last ? o.pass_name : std::string(pname), off, sw, ids32.data() + off, true,
                           last);
            if (cur < 0) return false;
            const double cms = ms_since(ck0);
            strata::dsv4::dsv4_moe_ws_cache_stats(moe_ws, ch1, cm1);
            ev1 = strata::dsv4::dsv4_moe_ws_evictions(moe_ws);
            const int64_t dh = ch1 - ch0, dm = cm1 - cm0, de = ev1 - ev0;
            std::fprintf(stderr,
                         "strata-dsv4: chunk %lld/%lld pos %lld..%lld (%lld rows) %.1f ms | experts "
                         "hits +%lld misses +%lld evict +%lld (chunk hit %.1f%%, cumulative "
                         "hits %lld misses %lld) | last-row argmax %lld\n",
                         (long long) (ci + 1), (long long) n_chunks, (long long) off,
                         (long long) (off + sw - 1), (long long) sw, cms, (long long) dh,
                         (long long) dm, (long long) de,
                         (dh + dm) > 0 ? 100.0 * (double) dh / (double) (dh + dm) : 0.0,
                         (long long) ch1, (long long) cm1, (long long) cur);
            ch0 = ch1;
            cm0 = cm1;
            ev0 = ev1;
        }
        const double pms = ms_since(pf0);
        std::fprintf(stderr,
                     "strata-dsv4: chunked prefill done: %lld tokens, %lld passes, %.1f ms total "
                     "(%.2f t/s), TTFT %.1f ms\n",
                     (long long) s, (long long) n_chunks, pms, pms > 0 ? 1000.0 * (double) s / pms : 0.0,
                     pms);
    }
    if (o.prompt_ids.empty())
        std::fprintf(stderr, "strata-dsv4: prefill argmax %lld vs golden 9544: %s\n", (long long) cur,
                     cur == 9544 ? "MATCH" : "MISMATCH");
    if (eton) {  // [diag] the prefill split, before the accumulators are reset for the decode passes
        std::fprintf(stderr,
                     "strata-dsv4: prefill phase ms: embed %.3f attn %.3f (gpu %.3f) moe %.3f "
                     "tail %.3f (sum %.3f)\n",
                     etm_embed, etm_attn, etm_attn_gpu, etm_moe, etm_tail,
                     etm_embed + etm_attn + etm_moe + etm_tail);
        std::fprintf(stderr,
                     "strata-dsv4: prefill attn gpu ms/call by class: hash %.3f (n=%lld) r128 %.3f "
                     "(n=%lld) idx %.3f (n=%lld)\n",
                     etm_attn_clsn[0] ? etm_attn_cls[0] / etm_attn_clsn[0] : 0,
                     (long long) etm_attn_clsn[0],
                     etm_attn_clsn[1] ? etm_attn_cls[1] / etm_attn_clsn[1] : 0,
                     (long long) etm_attn_clsn[1],
                     etm_attn_clsn[2] ? etm_attn_cls[2] / etm_attn_clsn[2] : 0,
                     (long long) etm_attn_clsn[2]);
        strata::dsv4::dsv4_moe_ws_report(moe_ws, "prefill");
    }
    static const int64_t kGoldenGen[3] = {734, 260, 4654};
    std::vector<int32_t> all = ids32;
    if (eton) {  // [S7b] report decode-only splits: drop the prefill pass from the accumulators
        etm_embed = etm_attn = etm_moe = etm_tail = etm_attn_gpu = 0;
        for (int c = 0; c < 3; ++c) {
            etm_attn_cls[c] = 0;
            etm_attn_clsn[c] = 0;
        }
        etm_passes = 0;
        strata::dsv4::dsv4_moe_ws_reset_timing(moe_ws);
    }
    // [G1] --dsv4-mrow-bench owns --dsv4-decode as its iteration count, so the single-row loop is off
    const int n_single = o.mrow_bench > 0 ? 0 : o.decode;
    for (int step = 1; step <= n_single; ++step) {
        const int64_t start_pos = (int64_t) all.size();
        all.push_back((int32_t) cur);
        const int32_t cur32 = (int32_t) cur;
        CK(cudaMemcpy(d_tokens + start_pos, &cur32, sizeof(int32_t), cudaMemcpyHostToDevice));
        char pname[32];
        std::snprintf(pname, sizeof pname, "decode%d", step);
        cur = run_pass(pname, start_pos, 1, &cur32, false, true);
        if (cur < 0) return false;
        if (o.prompt_ids.empty() && step <= 3)
            std::fprintf(stderr, "strata-dsv4: %s argmax %lld vs golden %lld: %s\n", pname,
                         (long long) cur, (long long) kGoldenGen[step - 1],
                         cur == kGoldenGen[step - 1] ? "MATCH" : "MISMATCH");
    }

    // [G1] multi-row verify / benchmark passes.  The verify pass runs ONCE with the explicit draft batch
    // and must be the last pass (it leaves the KV state at start_pos + R); the benchmark repeats R-row
    // passes so the per-pass time is measured from the same steady state as the single-row decode.
    if (o.verify_rows > 0) {
        std::vector<int32_t> fids;
        for (const char* c = o.verify_ids.c_str(); *c != 0;) {
            while (*c == ',' || *c == ' ') ++c;
            if (*c == 0) break;
            char* end = nullptr;
            const long long v = std::strtoll(c, &end, 10);
            if (end == c) break;
            fids.push_back((int32_t) v);
            c = end;
        }
        if ((int64_t) fids.size() < o.verify_rows) {
            err = "--dsv4-verify-ids needs at least --dsv4-verify-rows tokens";
            return false;
        }
        const int64_t start_pos = (int64_t) all.size();
        CK(cudaMemcpy(d_tokens + start_pos, fids.data(), (size_t) o.verify_rows * 4,
                      cudaMemcpyHostToDevice));
        for (int r = 0; r < o.verify_rows; ++r) all.push_back(fids[(size_t) r]);
        char pname[32];
        std::snprintf(pname, sizeof pname, "mrow%d", o.verify_rows);
        cur = run_pass(pname, start_pos, o.verify_rows, fids.data(), false, true);
        if (cur < 0) return false;
    }
    if (o.mrow_bench > 0) {
        const int64_t R = o.mrow_bench;
        // distinct tokens per row matter: the real verify batch routes R DIFFERENT top-6 sets, and the
        // expert union (not R x 6) is what c(R) actually pays for.  Without ids, all rows share one token.
        std::vector<int32_t> bids;
        if (!o.bench_ids.empty())
            for (const char* c = o.bench_ids.c_str(); *c != 0;) {
                while (*c == ',' || *c == ' ') ++c;
                if (*c == 0) break;
                char* end = nullptr;
                const long long v = std::strtoll(c, &end, 10);
                if (end == c) break;
                bids.push_back((int32_t) v);
                c = end;
            }
        for (int step = 1; step <= o.decode; ++step) {
            const int64_t start_pos = (int64_t) all.size();
            std::vector<int32_t> batch_ids((size_t) R);
            for (int64_t r = 0; r < R; ++r) {
                // in pass order: the ids are a real greedy chain, so pass `step` feeds exactly the R
                // consecutive tokens the verify batch would carry (no cycling until the chain runs out)
                const int64_t bi = ((int64_t) (step - 1) * R + r) % (int64_t) bids.size();
                const int32_t f = bids.empty() ? (int32_t) cur : bids[(size_t) bi];
                CK(cudaMemcpy(d_tokens + start_pos + r, &f, sizeof(int32_t),
                              cudaMemcpyHostToDevice));
                batch_ids[(size_t) r] = f;
                all.push_back(f);
            }
            char pname[32];
            std::snprintf(pname, sizeof pname, "mrowb%d", step);
            cur = run_pass(pname, start_pos, R, batch_ids.data(), false, true);
            if (cur < 0) return false;
        }
    }

    {
        int64_t ch = 0, cm = 0;
        strata::dsv4::dsv4_moe_ws_cache_stats(moe_ws, ch, cm);
        if (m.cache[0].valid())
            std::fprintf(stderr,
                         "strata-dsv4: expert cache: hits %lld / misses %lld (%.1f%%); resident %lld/%lld slots on device 0, %lld/%lld on device 1; evictions %lld\n",
                         (long long) ch, (long long) cm,
                         ch + cm > 0 ? 100.0 * ch / (ch + cm) : 0.0,
                         (long long) m.cache[0].resident(), (long long) m.cache[0].slots(),
                         (long long) (m.cache[1].valid() ? m.cache[1].resident() : 0),
                         (long long) (m.cache[1].valid() ? m.cache[1].slots() : 0),
                         (long long) strata::dsv4::dsv4_moe_ws_evictions(moe_ws));
        const strata::core::FileExpertSource::IoCounters ic = m.experts.io_counters();
        std::fprintf(stderr,
                     "strata-dsv4: expert io: demand reads %lld; demand residency cached %.1f / uncached %.1f GiB; "
                     "pread %.1f GiB in %llu calls (%.0f ms thread time); pf jobs %llu used %llu unused %llu "
                     "dropped %llu; decode waited %.0f ms on file reads\n",
                     (long long) m.experts.reads(), (double) ic.cached_bytes / 1073741824.0,
                     (double) ic.uncached_bytes / 1073741824.0, (double) ic.pread_bytes / 1073741824.0,
                     (unsigned long long) ic.pread_n, (double) ic.pread_us / 1000.0,
                     (unsigned long long) ic.pf_jobs, (unsigned long long) ic.pf_used,
                     (unsigned long long) ic.pf_unused, (unsigned long long) ic.pf_dropped,
                     (double) ic.crit_us / 1000.0);
    }
    if (eton && etm_passes > 0) {
        std::fprintf(stderr,
                     "strata-dsv4: engine phase ms/pass over %lld passes: attn %.3f (gpu %.3f) "
                     "moe %.3f tail %.3f\n",
                     (long long) etm_passes, etm_attn / etm_passes, etm_attn_gpu / etm_passes,
                     etm_moe / etm_passes, etm_tail / etm_passes);
        std::fprintf(stderr,
                     "strata-dsv4: attn gpu ms/call by class: hash %.3f (n=%lld) r128 %.3f (n=%lld) "
                     "idx %.3f (n=%lld)\n",
                     etm_attn_clsn[0] ? etm_attn_cls[0] / etm_attn_clsn[0] : 0,
                     (long long) etm_attn_clsn[0],
                     etm_attn_clsn[1] ? etm_attn_cls[1] / etm_attn_clsn[1] : 0,
                     (long long) etm_attn_clsn[1],
                     etm_attn_clsn[2] ? etm_attn_cls[2] / etm_attn_clsn[2] : 0,
                     (long long) etm_attn_clsn[2]);
    }
    if (ea0 != nullptr) cudaEventDestroy(ea0);
    if (ea1 != nullptr) cudaEventDestroy(ea1);
    for (auto& st : comp_states) strata::dsv4::dsv4_comp_state_free(st);
    for (auto& st : idx_states) strata::dsv4::dsv4_comp_state_free(st);
    strata::dsv4::dsv4_indexer_workspace_free(idx_ws);
    strata::dsv4::dsv4_comp_workspace_free(comp_ws);
    strata::dsv4::dsv4_attn_workspace_free(attn_ws);
    rope_plain.free();
    rope_yarn.free();
    strata::dsv4::dsv4_moe_ws_free(moe_ws);
    cudaFree(d_tokens);
    cudaFree(d_emb);
    cudaFree(d_h);
    cudaFree(d_y);
    cudaFree(d_q8);
    cudaFree(d_logits);
    cudaStreamDestroy(stream);
    return true;
}

}  // namespace

// ==================================================================================== [S2] MoE chain (DONE)
// src/dsv4/dsv4_moe.{h,cu} + the wiring above forward_tail: hc_pre (ffn side) with sinkhorn(20), rmsnorm,
// sqrtsoftplus routing (route_scale 1.5; the tid2eid hash tables of layers 0-2 SELECT while the weights are
// still gathered from the gate scores; routed layers pick top-6 on score + bias, the bias selecting only),
// the NVFP4 grouped GEMV with the V4 swiglu clamp variant and the per-expert weight_scale_2 tail, the Q8_0
// shared expert unweighted, the k = 6 combine, hc_post.  STRATA_DSV4_MOE_CPU=1 runs the expert stage on the
// host with dsv4ref's exact arithmetic (the slice's CPU-pool path and the GPU kernels' cross-check).

// ================================================================ [S3] dense MLA attention (DONE, a7ddf0d5)
// Layers 0/1: full MLA attention over the uncompressed KV (q_lora 1024, kv_lora 512, 64 x 512 heads), in
// src/dsv4/dsv4_attn.cu behind dsv4_attn_forward, wired into the layer loop above.

// ==================================================================================== [S4] compressor (DONE)
// The r = 4 / r = 128 pooling stack: kv/gate projections, per-block APE, compressed-slot rope phase
// j * ratio, the overlap carry state (coff = 2 on r = 4).  src/dsv4/dsv4_comp.cu; the r = 128 layers run
// compressor + MLA over window + compressed slots through dsv4_comp_attn_forward, and S5 built the r = 4
// layers' indexer + sparse attention on the same compressor primitives.

// ================================================================ [S5] indexer + sparse attn (DONE)
// The ratio-4 layers' indexer (64 x 128, EXACT top-512 over fp32 keys - the score + selection run on the
// host, mirroring dsv4ref indexer_forward :895's accumulation order and stable-sort contract) and the
// sparse attention over the window plus the selected compressed slots, in src/dsv4/dsv4_comp.cu behind
// dsv4_idx_attn_forward.  L02 runs the same full DSA attention path (its hash side lives in the MoE
// half's tid2eid routing).  The decode pass is wired: --dsv4-decode N chains N single-token passes
// (window ring offset win = 128, compressor slot start_pos / ratio, indexer over all compressed slots).

// ============================================================ [S6] decode + KV (DONE at golden scale)
// The token loop and the KV append into this file's arena are wired through run_pass's start_pos
// contract (acceptance: prefill argmax 9544, decode1..3 argmax 734/260/4654).  Remaining: CUDA-graph
// decode and the wider-context soak.

// ============================================================================== [S7] expert pool tiers (later)
// The VRAM expert cache (auto slots, the numbers print_residency forecasts), the CPU pool workers, and the
// performance pass.

// ================================================================================================ [S0] run

int strata::dsv4::run(int argc, char** argv) {
    Options o;
    if (!parse_args(argc, argv, o)) return 2;
#if !defined(_WIN32)
    setenv("CUDA_MODULE_LOADING", "EAGER", 0);   // every kernel loaded with the context, not in the first prompt
#endif
    CK(cudaSetDevice(0));
    Model m;
    std::string err;
    try {
        if (!load(o, m, err)) { std::fprintf(stderr, "strata-dsv4: %s\n", err.c_str()); return 1; }
    } catch (const std::exception& e) {
        std::fprintf(stderr, "strata-dsv4: %s\n", e.what());
        return 1;
    }
    print_residency(m, o.ctx);
    if (o.forward) {
        try {
            if (!forward_tail(o, m, err)) { std::fprintf(stderr, "strata-dsv4: %s\n", err.c_str()); return 1; }
        } catch (const std::exception& e) {
            std::fprintf(stderr, "strata-dsv4: %s\n", e.what());
            return 1;
        }
    } else if (o.serve || o.have_tokens)
        std::fprintf(stderr, "strata-dsv4: S0 loads and reports only; generation arrives with S1-S7\n");
    attn_time_flush();
    moe_time_flush();
    std::fprintf(stderr, "strata-dsv4: %s OK, exiting clean\n", o.forward ? "S5 forward" : "S0 load");
    return 0;
}
