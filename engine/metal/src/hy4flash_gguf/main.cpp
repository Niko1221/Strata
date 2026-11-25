// engine/metal/src/hy4flash_gguf/main.cpp - the hy4-flash Metal engine's entry point.
//
// The flags are the CUDA engine's (src/hy4flash_gguf/engine.cpp), so the same launch command works on either:
// --native / --model, --pack, --context, --serve, --mtp, --draft-tokens.  --expert-vram and --resident-experts
// are accepted and ignored: on Apple Silicon there is one pool of memory and the arena rules
// (metal/src/engine/arena_budget.h) size it.
//
// WHAT THIS BUILD DOES AND WHAT IT DOES NOT.  It reads the geometry, plans the dense residency, reads the dense
// tensors into memory, sizes and fills the expert arena, replays a routing trace through it (--expert-test), and
// runs the forward pass - DECODE as a window of at most 8 positions, one command buffer with a cut at every MoE
// router, and PREFILL as chunks of --prefill-chunk positions, one command buffer per block so each expert of a
// layer is read from the SSD once for the whole chunk - with the acceptance entry points on top: --block-test,
// --head-test, --logits-out (the teacher-forced logits), --greedy and --prefill-consistency (prefill's logits
// against the decode path's, bitwise).
//
// --mtp --draft-tokens K turns --greedy into self-speculative decoding with the MTP block (mtp.h): the output is
// the plain greedy output token for token, because the verify window is bitwise the decode path.  --serve is
// refused rather than faked; saying so beats a binary that starts, answers nothing and leaves the caller to guess
// which half is missing.
#include "dispatch.h"
#include "dump.h"
#include "experts.h"
#include "forward.h"
#include "model.h"
#include "mtl_rt.h"
#include "mtp.h"
#include "prefill_tol.h"
#include "ref_quant.h"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include <ctime>
#include <dirent.h>      // the kernels directory, to compare its mtime against the binary's
#include <sys/stat.h>   // the snapshot binds to the model file's size and mtime (prefill_snap.h)

namespace {

constexpr int32_t kContextDefault = 8192;
// The top tier is a 32768-token prompt and then 256 generated tokens, so 32768 positions is one too few.
// 33792 = 33 x 1024 covers the prompt, the 256 and some slack.  The memory that costs is 42 x C x 576 x 2 B of
// KV and 12 x C x 384 x 2 B of indexer cache, so 33792 asks for 0.060 GB more than 32768 would.
constexpr int32_t kContextMax = 33792;   // the file declares 1,048,576
constexpr int kMaxWindowT = strata::hy4_flash::Forward::kMaxWindow;   // a decode window
constexpr int kPrefillChunkDefault = 1024;   // the prefill scratch is sized from it

/// "A:B,C:D" -> the position ranges to write (--logits-range).  Empty on a parse error, which `err` names.
std::vector<std::pair<int, int>> parse_ranges(const char* s, int n, std::string& err) {
    std::vector<std::pair<int, int>> out;
    const char* p = s;
    while (*p != 0) {
        char* e1 = nullptr;
        const long a = std::strtol(p, &e1, 10);
        if (e1 == p || *e1 != ':') { err = std::string("--logits-range: expected A:B at '") + p + "'"; return {}; }
        char* e2 = nullptr;
        const long b = std::strtol(e1 + 1, &e2, 10);
        if (e2 == e1 + 1) { err = std::string("--logits-range: expected A:B at '") + p + "'"; return {}; }
        if (a < 0 || b <= a || b > n) {
            err = "--logits-range " + std::to_string(a) + ":" + std::to_string(b) + " is outside 0.." +
                  std::to_string(n);
            return {};
        }
        out.push_back({(int) a, (int) b});
        p = e2;
        if (*p == ',') ++p;
        else if (*p != 0) { err = std::string("--logits-range: expected ',' at '") + p + "'"; return {}; }
    }
    if (out.empty()) err = "--logits-range is empty";
    return out;
}

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

/// A routing trace: "HY4R", version 1, then n_ord / n_experts / topk / n_pos and the routed ids as
/// int32[n_pos][n_ord][topk].  Written by tmp/tests/hy4_flash/pyref/make_expert_profile.py from the reference's own
/// `ffn_moe_topk-<block>` dumps, so the replay below drives the arena with the routing the model really makes.
struct Trace {
    uint32_t n_ord = 0, n_experts = 0, topk = 0, n_pos = 0;
    std::vector<int32_t> ids;
    const int32_t* at(uint32_t pos, uint32_t ord) const {
        return ids.data() + ((size_t) pos * n_ord + ord) * topk;
    }
};

std::string read_trace(const char* path, Trace& t) {
    FILE* f = std::fopen(path, "rb");
    if (f == nullptr) return std::string("cannot open ") + path;
    char magic[4];
    uint32_t h[5];
    if (std::fread(magic, 1, 4, f) != 4 || std::memcmp(magic, "HY4R", 4) != 0 || std::fread(h, 4, 5, f) != 5) {
        std::fclose(f);
        return std::string(path) + " is not a routing trace (HY4R)";
    }
    if (h[0] != 1) { std::fclose(f); return std::string(path) + ": trace version is not 1"; }
    t.n_ord = h[1];
    t.n_experts = h[2];
    t.topk = h[3];
    t.n_pos = h[4];
    const size_t n = (size_t) t.n_pos * t.n_ord * t.topk;
    t.ids.resize(n);
    const bool ok = std::fread(t.ids.data(), 4, n, f) == n;
    std::fclose(f);
    if (!ok) return std::string(path) + " is shorter than its header says";
    return {};
}

/// A non-finite value that tried to enter an f16 cache ends the REQUEST with `finish = nonfinite`, it does not
/// kill the process and it is not a warning at the end of the run.  Every window's return value goes through
/// here, so the three drivers below cannot forget it.  Returns the exit status for a one-shot driver (0: the
/// request ended, cleanly, with `nonfinite`), or -1 when the error is a real failure.
int nonfinite_or_fail(const std::string& err, const char* what, int where) {
    if (strata::hy4_flash::Forward::is_nonfinite(err)) {
        std::printf("\n  finish         nonfinite (%s %d): %s\n", what, where, err.c_str());
        std::fflush(stdout);
        return 0;
    }
    return -1;
}

/// Replays `windows` decode windows of T positions each through the arena.  `pred_wrong` of each token's
/// predictions are replaced by a deterministic wrong expert, so the read-ahead's yield can be read against the
/// quality of the prediction: 0 is its upper bound and larger values bracket it.
/// With `verify`, every expert the window asked for is compared byte for byte against a fresh pread of the GGUF.
int replay(strata::hy4_flash::Model& m, const Trace& tr, int T, int windows, int pred_wrong, bool verify) {
    const int no = m.stream.n_layers();
    const uint32_t topk = tr.topk;
    std::vector<int32_t> ids((size_t) T * topk);
    std::vector<float> pred((size_t) T * 512);
    std::vector<uint8_t> want, got;
    uint64_t checked = 0, wrong = 0;
    for (int w = 0; w < windows; ++w) {
        const uint32_t p0 = (uint32_t) ((uint64_t) w * T % tr.n_pos);
        for (int o = 0; o < no; ++o) {
            for (int t = 0; t < T; ++t)
                std::memcpy(&ids[(size_t) t * topk], tr.at((p0 + (uint32_t) t) % tr.n_pos, (uint32_t) o),
                            topk * 4);
            const float* p = nullptr;
            if (o + 1 < no) {
                std::fill(pred.begin(), pred.end(), -1.0f);
                for (int t = 0; t < T; ++t) {
                    const int32_t* nxt = tr.at((p0 + (uint32_t) t) % tr.n_pos, (uint32_t) (o + 1));
                    for (uint32_t k = 0; k < topk; ++k) {
                        int32_t e = nxt[k];
                        if ((int) k < pred_wrong) e = (int32_t) ((e * 7919 + 13 * (o + 1)) % (int) tr.n_experts);
                        pred[(size_t) t * 512 + e] = 1.0f;
                    }
                }
                p = pred.data();
            }
            strata::hy4_flash::at_cut(m, o, T, (int) topk, ids.data(), p);
            if (!verify) continue;
            // THE POINT OF THE WHOLE LAYER: a slot holds the expert's GGUF bytes unchanged, so a hit and a miss
            // feed the kernels the same bytes.
            //
            // THE OFFSETS HERE MUST BE COMPUTED INDEPENDENTLY of the fill's.  Asking
            // `ExpertStream::slice_offset` would be the very arithmetic the fill used, so a systematic error in
            // it would compare wrong bytes against the same wrong bytes and pass.  These come from
            // `Model::experts`, which model.cpp built from the GGUF's own tensor directory before the stream was
            // opened - a different code path reading the same file (`open_experts` checks the two agree, so this
            // check is what gives that comparison its value).
            // ONE `wrong` PER EXPERT, and `checked` counts every expert looked at: an absent expert still
            // counts as checked, and one expert's three slices are one failure, not three - otherwise the two
            // numbers are not comparable unless both are 0.
            const strata::hy4_flash::ExpertSlices& sl = m.experts[(size_t) o];
            for (int i = 0; i < T * (int) topk; ++i) {
                const int e = ids[(size_t) i];
                ++checked;
                const uint32_t code = m.stream.smap_of(o, e);
                if (code == 0xFFFFFFFFu) { ++wrong; continue; }   // the window asked for it and it is not there
                const uint8_t* h = m.stream.chunk((int) (code >> 28)).as<uint8_t>() +
                                   (uint64_t) (code & 0x0FFFFFFFu) * 16384ull;
                uint64_t off = 0;
                bool ok = true;
                for (int k = 0; k < 3 && ok; ++k) {
                    const uint64_t n = sl.stride[k];
                    const uint64_t at = sl.off[k] + (uint64_t) e * sl.stride[k];
                    want.resize((size_t) n);
                    ok = m.stream.read(want.data(), n, at) &&
                         std::memcmp(want.data(), h + off, (size_t) n) == 0;
                    off += n;
                }
                if (!ok) ++wrong;
            }
        }
    }
    if (verify)
        std::printf("  expert bytes   %llu experts asked for, re-read from the GGUF with INDEPENDENT offsets "
                    "(Model::experts) and compared with the arena: %llu differ\n",
                    (unsigned long long) checked, (unsigned long long) wrong);
    return wrong == 0 ? 0 : 4;
}

// ---- the acceptance entry points' file reading ---------------------------------------------------------------------
std::vector<int32_t> read_tokens(const char* path, std::string& err) {
    std::vector<int32_t> out;
    FILE* f = std::fopen(path, "r");
    if (f == nullptr) {
        err = std::string("cannot open ") + path;
        return out;
    }
    long v = 0;
    while (std::fscanf(f, "%ld", &v) == 1) out.push_back((int32_t) v);
    std::fclose(f);
    if (out.empty()) err = std::string(path) + " holds no token ids";
    return out;
}

std::vector<uint8_t> read_all(const char* path, std::string& err) {
    std::vector<uint8_t> out;
    FILE* f = std::fopen(path, "rb");
    if (f == nullptr) {
        err = std::string("cannot open ") + path;
        return out;
    }
    std::fseek(f, 0, SEEK_END);
    const long n = std::ftell(f);
    std::fseek(f, 0, SEEK_SET);
    out.resize((size_t) (n > 0 ? n : 0));
    if (!out.empty() && std::fread(out.data(), 1, out.size(), f) != out.size())
        err = std::string("short read of ") + path;
    std::fclose(f);
    return out;
}

void usage(const char* exe) {
    std::printf(
        "strata_hy4_flash_metal - hy4-flash (hy_v4_flash) on Metal, Stage 1\n"
        "\n"
        "  %s --native MODEL-00001-of-00002.gguf [--pack DIR] [--context N] [--dry-run]\n"
        "\n"
        "  --native, --model PATH   the model's shard 1; the other shards are found beside it\n"
        "  --pack DIR               the pack from tools/iq_pack.py.  The engine reads engram_token_map.bin\n"
        "                           from it, so it is a run-time dependency of the forward pass\n"
        "  --context N              positions to reserve (default %d, this build's limit %d)\n"
        "  --dry-run                read the headers and print the plan; read no weights\n"
        "  --no-arena               read the dense weights but do not size or fill the expert arena\n"
        "  --expert-profile PATH    the arena's starting ranking (ordinal numbering); default <pack>/expert-profile.bin\n"
        "  --expert-test PATH       replay a routing trace (HY4R) through the arena: the expert supply of a\n"
        "                           decode window, without the forward pass (plan P1.3)\n"
        "  --windows N              how many windows to replay (default 64)\n"
        "  --window-t T             positions per window, 1..%d (default 1: one token at a time)\n"
        "  --pred-wrong K           spoil K of each token's %d read-ahead predictions, to bracket the yield\n"
        "  --verify                 re-read every expert from the GGUF and compare it with the arena\n"
        "\n"
        "  the forward pass (plan P1.4).  --tokens FILE is a whitespace-separated list of ids:\n"
        "  --tokens FILE            the sequence to run, through the decode path a window at a time\n"
        "  --logits-out FILE        write [T][vocab] f32 for every position: gate 2's teacher-forced logits\n"
        "  --logits-range A:B[,C:D] write only those position ranges (S1.3's four segments, so an 8192-position\n"
        "                           run writes 1 GB instead of 4)\n"
        "  --greedy N               continue greedily for N tokens and print 'greedy: <ids>'\n"
        "  --pred-accuracy          report how often the cross-layer read-ahead's top-k is what the next\n"
        "                           ordinal really routed to (the number plan P1.4 asks for)\n"
        "\n"
        "  prefill (plan P1.5).  Chunks of C positions, each expert of a layer read from the SSD once per chunk:\n"
        "  --prefill-chunk C        run the prompt through prefill in chunks of C positions (default %d with\n"
        "                           --prefill, 0 = the decode path, which is what P1.4 measured)\n"
        "  --prefill                --prefill-chunk %d\n"
        "  --prefill-consistency    run the prompt BOTH ways and compare the per-position logits: bitwise (plan\n"
        "                           P1.5) and by tolerance (prefill_tol.h: KL / top-1 against the decode path).\n"
        "                           Needs vocab x compared positions x 4 B of host memory\n"
        "  --consistency-stride S   compare every S-th prompt position and the last 64 (default 1)\n"
        "  --consistency-cont N     then N continuation positions from the prompt's state (default 8)\n"
        "  --consistency-out F      per-position KL / argmax rows to F\n"
        "  --consistency-tol F      the tolerance verdict against F's thresholds decides the exit status\n"
        "  --consistency-force-route  prefill uses the decode path's routing (a pre-pass), so only arithmetic differs.\n"
        "                           Never bitwise by construction, so it REQUIRES --consistency-tol (the forced\n"
        "                           thresholds are their own file).  An unknown --consistency-* flag is an error\n"
        "  prefill snapshots (prefill_snap.h): run the prompt once, measure decode many times.  An 8K run\n"
        "  spends 450 of its 510 s on the prompt, and an A/B pays that twice for a prompt that did not change.\n"
        "  --save-prefill P         after the prompt, write the state a decode continuation reads to P (the MLA\n"
        "                           latent cache, the indexer cache, the pooled-key counts, the last token;\n"
        "                           about 0.5 GB at 8K).  Needs --tokens\n"
        "  --load-prefill P         skip the prompt and continue from P.  REFUSED unless the model file, the\n"
        "                           prompt, --context, --prefill-chunk, the geometry and the MTP history all\n"
        "                           match what P was taken with, and unless P's payload matches its own sha256.\n"
        "                           Not with --logits-out / --dump / --prefill-consistency / --pred-accuracy:\n"
        "                           those describe the prompt, which did not run\n"
        "  --decode-logits-out P    every GENERATED position's logits (--logits-out is the prompt's).  This is\n"
        "                           what makes `prefill then decode` and `load then decode` comparable bitwise\n"
        "  --block-test L --in HC --dump DIR [--force-topk F] [--force-sel F]\n"
        "                           run ONLY block L from the HC state in HC ([T][hc][n_embd] f32) and dump the\n"
        "                           reference's own node names (ADR §9).  --force-topk takes the reference's\n"
        "                           ffn_moe_topk-L.bin, --force-sel its indexer_sel-L.bin\n"
        "  --head-test --in HC [--dump DIR]\n"
        "                           run only the output head from the final HC state\n"
        "  --mtp                    greedy self-speculative decoding with the MTP block (plan P2.4): every round\n"
        "                           drafts k tokens and verifies them in one window; the output is the plain\n"
        "                           greedy output, token for token (HY4_FLASH_MTP_PROBE=draft_random|accept_all are the\n"
        "                           negative controls; HY4_FLASH_MTP_EXPERT_GB sizes the MTP's expert pool)\n"
        "  --draft-tokens K         drafts a round with --mtp (default 2, 0..7)\n"
        "  --serve                  accepted; not implemented in this build (plan P1.6)\n"
        "  --expert-vram GIB, --resident-experts\n"
        "                           accepted and ignored: one memory pool here, sized by arena_budget.h\n"
        "  --help                   this text\n"
        "\n"
        "  STRATA_METAL_ARENA_GB    the arena outright; the default here is the largest value observed to\n"
        "                           pass on this machine, %.0f GB - not a proven ceiling (experts.h)\n"
        "  HY4_FLASH_PREFETCH=0           no cross-layer read-ahead (ADR §5's `pred` becomes nullptr)\n"
        "  HY4_FLASH_PREFILL_ATTN / _MOE / _HEAD\n"
        "                           positions per attention / MoE / head dispatch inside a prefill chunk\n"
        "                           (defaults 64 / 256 / 8; they trade GPU bandwidth against scratch, and no\n"
        "                           value depends on them - see forward.h)\n",
        exe, kContextDefault, kContextMax, kMaxWindowT, 16, kPrefillChunkDefault, kPrefillChunkDefault,
        strata::hy4_flash::kArenaObservedGB);
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2 || has_flag(argc, argv, "--help") || has_flag(argc, argv, "-h")) {
        usage(argv[0]);
        return argc < 2 ? 2 : 0;
    }

    // Before anything is read: an environment that would change the arithmetic is refused, not downgraded.
    if (const std::string bad = strata::hy4_flash::rejected_env(); !bad.empty()) {
        std::fprintf(stderr, "strata_hy4_flash_metal: %s\n", bad.c_str());
        return 2;
    }

    const char* path = flag_value(argc, argv, "--native", "--model");
    if (path == nullptr) {
        std::fprintf(stderr, "strata_hy4_flash_metal: --native MODEL-00001-of-00002.gguf is needed\n");
        return 2;
    }
    int32_t context = kContextDefault;
    if (const char* v = flag_value(argc, argv, "--context")) context = std::atoi(v);
    if (context < 64 || context > kContextMax) {
        std::fprintf(stderr, "strata_hy4_flash_metal: --context %d is outside 64..%d, this build's limit\n", context,
                     kContextMax);
        return 2;
    }

    // An execution request this build cannot honour is refused BEFORE loading 8 GB of weights, and with a
    // non-zero status: a caller that asked to serve must not see a successful exit.
    for (const char* f : {"--serve"})
        if (has_flag(argc, argv, f)) {
            std::fprintf(stderr, "strata_hy4_flash_metal: %s is not implemented in this build (plan P1.6)\n", f);
            return 3;
        }
    // --mtp: the MTP block's dense tensors are kept resident and the greedy loop drafts k a round
    const bool want_mtp = has_flag(argc, argv, "--mtp");
    // The default is 2: a draft step's acceptance RISES with the generated length while a verify position's
    // marginal cost is flat, so the second draft pays for itself and a third adds little.  The emitted text does
    // not depend on k - the draft only proposes, the trunk's verify decides.
    int draft_k = 2;
    if (const char* v = flag_value(argc, argv, "--draft-tokens")) draft_k = std::atoi(v);
    if (draft_k < 0 || draft_k > kMaxWindowT - 1) {
        std::fprintf(stderr, "strata_hy4_flash_metal: --draft-tokens %d is outside 0..%d\n", draft_k, kMaxWindowT - 1);
        return 2;
    }
    const strata::hy4_flash::MtpProbe mtp_probe = strata::hy4_flash::mtp_probe_from_env();
    if (want_mtp && mtp_probe != strata::hy4_flash::MtpProbe::none)
        std::fprintf(stdout, "strata_hy4_flash_metal: HY4_FLASH_MTP_PROBE=%s - a NEGATIVE CONTROL of the speculative decoding: %s\n",
                     std::getenv("HY4_FLASH_MTP_PROBE"),
                     mtp_probe == strata::hy4_flash::MtpProbe::accept_all
                         ? "the verify is skipped and the output is WRONG ON PURPOSE"
                         : "the drafts are random; the output must still be the plain greedy one");

    strata::hy4_flash::Model m;
    // Every --consistency-* control this build has.  A spelling that is not in this list is a STARTUP ERROR, not a
    // silently dropped argument: a build that does not have the flag would otherwise run the comparison with FREE
    // routing and have it recorded as a forced-routing acceptance.  argv is the only place that is visible, so
    // argv is where it is caught.
    static constexpr const char* kConsistencyFlags[] = {
        "--consistency-stride", "--consistency-cont", "--consistency-out", "--consistency-tol",
        "--consistency-force-route",
    };
    for (int i = 1; i < argc; ++i) {
        if (std::strncmp(argv[i], "--consistency", 13) != 0) continue;
        bool known = false;
        for (const char* k : kConsistencyFlags) known = known || std::strcmp(argv[i], k) == 0;
        if (!known) {
            std::fprintf(stderr, "strata_hy4_flash_metal: %s is not a flag this build has, so it would be silently ignored; "
                                 "this build's --consistency-* flags are:", argv[i]);
            for (const char* k : kConsistencyFlags) std::fprintf(stderr, " %s", k);
            std::fprintf(stderr, "\n");
            return 2;
        }
    }
    // HY4_FLASH_PREFILL_PROBE (a deliberate wrongness) and HY4_FLASH_PREFILL_NOISE (a deliberate perturbation) exist only for
    // --prefill-consistency's negative and benign controls.  Anywhere else they would put WRONG caches under a
    // decode measurement or a saved snapshot with nothing but a stdout line to show for it - so they are refused.
    if (has_flag(argc, argv, "--prefill-consistency")) {
        const char* why = has_flag(argc, argv, "--head-test")                 ? "--head-test"
                          : flag_value(argc, argv, "--block-test") != nullptr ? "--block-test"
                                                                              : nullptr;
        if (why != nullptr) {   // both return before the judge could run, so it would be a silent no-op
            std::fprintf(stderr, "strata_hy4_flash_metal: %s returns before --prefill-consistency could judge anything - run "
                                 "them separately\n", why);
            return 2;
        }
        // A sampled run's bitwise verdict covers only the sampled positions, so it must not be the exit status:
        // --consistency-stride 1000 would otherwise exit 0 for a prefill that differs everywhere else.  The
        // tolerance verdict names the count it judged, so it is allowed.
        const char* cs = flag_value(argc, argv, "--consistency-stride");
        if (cs != nullptr && std::atoi(cs) > 1 && flag_value(argc, argv, "--consistency-tol") == nullptr) {
            std::fprintf(stderr, "strata_hy4_flash_metal: --consistency-stride %s compares only some positions, so the bitwise "
                                 "verdict cannot be the exit status: give --consistency-tol FILE too\n", cs);
            return 2;
        }
    }
    if (has_flag(argc, argv, "--consistency-force-route")) {
        if (!has_flag(argc, argv, "--prefill-consistency")) {
            std::fprintf(stderr, "strata_hy4_flash_metal: --consistency-force-route is a control of --prefill-consistency only\n");
            return 2;
        }
        // A forced run is NEVER bitwise - holding the routing fixed leaves the two paths' arithmetic different on
        // purpose - so `rc = bitwise ? 0 : 4` below hands it 4 whatever the numbers are, which is the same 4 a
        // benign free-routing run returns.  Without --consistency-tol such a run therefore judges nothing.  The
        // forced thresholds are their own file, a different one from the free-routing thresholds, so the judge
        // cannot be defaulted in - the caller has to name it.
        if (flag_value(argc, argv, "--consistency-tol") == nullptr) {
            std::fprintf(stderr, "strata_hy4_flash_metal: --consistency-force-route is never bitwise, so without "
                                 "--consistency-tol FILE its exit status is 4 either way and it judges nothing: "
                                 "give --consistency-tol FILE too (the forced thresholds are their own file)\n");
            return 2;
        }
    }
    for (const char* e : {"HY4_FLASH_PREFILL_PROBE", "HY4_FLASH_PREFILL_NOISE"}) {
        const char* v = std::getenv(e);
        if (v != nullptr && *v != 0 && std::atof(v) != 0 && !has_flag(argc, argv, "--prefill-consistency")) {
            std::fprintf(stderr, "strata_hy4_flash_metal: %s=%s is a control of --prefill-consistency only (forward.h): refusing "
                                 "to run without it\n", e, v);
            return 2;
        }
    }
    if (const std::string err = strata::hy4_flash::open(m, path, context, want_mtp); !err.empty()) {
        std::fprintf(stderr, "strata_hy4_flash_metal: %s: %s\n", path, err.c_str());
        return 2;
    }
    std::printf("strata_hy4_flash_metal: %s\n%s\n", path, strata::hy4_flash::describe(m).c_str());

    const char* tok_path = flag_value(argc, argv, "--tokens");
    // Read and validate --tokens HERE, right after open() gives us the geometry and BEFORE load_dense's 8 GB and
    // make_arena: after those, a hand-written token file with one bad id takes minutes to be rejected.
    std::vector<int32_t> tokens;
    if (tok_path != nullptr) {
        std::string err;
        tokens = read_tokens(tok_path, err);
        if (!err.empty()) {
            std::fprintf(stderr, "strata_hy4_flash_metal: %s\n", err.c_str());
            return 2;
        }
        if ((int32_t) tokens.size() > context) {
            std::fprintf(stderr, "strata_hy4_flash_metal: %zu tokens but --context %d\n", tokens.size(), context);
            return 2;
        }
    if (const char* pv = std::getenv("HY4_FLASH_PREFILL_PROBE"); pv != nullptr && *pv != 0) {
        // Probes 1 and 2 read sub-batch 0's offset instead of their own, which is the IDENTITY when the chunk has
        // only one sub-batch - so on a short prompt the negative control would pass on a wrong-on-purpose build.
        // The sub-batch sizes are attn 64 / moe 256 by default (forward.h).
        const int pr = std::atoi(pv);
        const int sub = pr == 1 ? (std::getenv("HY4_FLASH_PREFILL_ATTN") ? std::atoi(std::getenv("HY4_FLASH_PREFILL_ATTN")) : 64)
                       : pr == 2 ? (std::getenv("HY4_FLASH_PREFILL_MOE") ? std::atoi(std::getenv("HY4_FLASH_PREFILL_MOE")) : 256)
                                 : 0;
        if (sub > 0 && (int) tokens.size() <= sub) {
            std::fprintf(stderr, "strata_hy4_flash_metal: HY4_FLASH_PREFILL_PROBE=%d needs a prompt longer than its sub-batch (%d): "
                                 "below that it substitutes sub-batch 0 for itself and changes nothing\n", pr, sub);
            return 2;
        }
    }
        // A token id outside the vocabulary must be an ERR here: the id goes straight into token_embd's row
        // lookup on the GPU, where nothing bounds-checks either, so a hand-written token file is otherwise
        // enough to read outside the weights.  It needs no nan and no unusual state to do it.
        for (size_t i = 0; i < tokens.size(); ++i)
            if (tokens[i] < 0 || tokens[i] >= m.geo.vocab_size) {
                std::fprintf(stderr, "strata_hy4_flash_metal: ERR %s position %zu holds token id %d, outside 0..%d (ADR §8)\n",
                             tok_path, i, tokens[i], m.geo.vocab_size - 1);
                return 2;
            }
    }

    // The pack is where engram_token_map.bin lives: its absence is reported now rather than in the middle of the
    // first forward pass.
    std::string pack_dir;
    if (const char* pack = flag_value(argc, argv, "--pack")) {
        pack_dir = pack;
        const std::string map = std::string(pack) + "/engram_token_map.bin";
        if (FILE* f = std::fopen(map.c_str(), "rb")) {
            std::fseek(f, 0, SEEK_END);
            const long n = std::ftell(f);
            std::fclose(f);
            std::printf("  pack           %s: engram_token_map.bin %ld entries%s\n", pack, n / 4,
                        n / 4 == m.geo.vocab_size ? "" : " (NOT vocab_size: the map is another model's)");
        } else {
            std::fprintf(stderr, "strata_hy4_flash_metal: %s has no engram_token_map.bin (build the pack with "
                                 "tools/iq_pack.py)\n", pack);
            return 2;
        }
    }

    if (has_flag(argc, argv, "--dry-run")) {
        std::printf("  dry run        no weights read\n");
        return 0;
    }

    mtl::Context cx;
    cx.lock_safe_math();   // every library this engine compiles is safe-math, and a fast-math switch is fatal
    std::printf("  device         %s, working set %.2f GB\n", cx.device_name().c_str(),
                (double) cx.recommended_working_set() / 1e9);
    // WHICH MSL THIS BINARY WILL COMPILE, and whether it is the one it was built with.  The sources are read at
    // RUN TIME, so a binary can compile a tree newer than itself and run old host code against new kernels with
    // nothing in the log to say so.  The build pins a copy beside the binary, but all targets in one build
    // directory share that copy, so building anything there refreshes it for binaries already built.  Comparing
    // mtimes catches exactly that, and costs one stat per kernel file.
    {
        const std::string kd = strata_hy4_flash_metal::kernel_dir();
        std::printf("  kernels        hy4_flash %s\n                 shared %s\n", kd.c_str(),
                    cx.kernel_dir.empty() ? "(none resolved)" : cx.kernel_dir.c_str());
        struct stat se {};
        time_t newest = 0;
        if (DIR* d = ::opendir(kd.c_str()); d != nullptr) {
            while (const dirent* e = ::readdir(d)) {
                if (e->d_name[0] == '.') continue;
                struct stat sf {};
                if (::stat((kd + "/" + e->d_name).c_str(), &sf) == 0 && sf.st_mtime > newest) newest = sf.st_mtime;
            }
            ::closedir(d);
        }
        if (newest != 0 && ::stat(argv[0], &se) == 0) {
            char kb[32] = {}, bb[32] = {};
            std::strftime(kb, sizeof kb, "%F %T", std::localtime(&newest));
            std::strftime(bb, sizeof bb, "%F %T", std::localtime(&se.st_mtime));
            if (newest > se.st_mtime)
                std::printf("  kernels        !! the kernels were pinned at %s, AFTER this binary was built "
                            "(%s): this run mixes them and its numbers are not this binary's\n", kb, bb);
            else
                std::printf("  kernels        pinned %s, binary %s (the binary is the newer of the two)\n", kb, bb);
        }
    }
    if (const std::string err = strata::hy4_flash::load_dense(m, cx); !err.empty()) {
        std::fprintf(stderr, "strata_hy4_flash_metal: %s\n", err.c_str());
        return 2;
    }
    std::printf("  dense resident %llu B = %.4f GiB = %.3f GB in %.1f s (%.2f GB/s); buffer %.4f GiB\n",
                (unsigned long long) m.dense_bytes, (double) m.dense_bytes / (double) (1ull << 30),
                (double) m.dense_bytes / 1e9, m.load_s, (double) m.dense_bytes / 1e9 / (m.load_s > 0 ? m.load_s : 1),
                (double) m.dense.bytes / (double) (1ull << 30));

    if (has_flag(argc, argv, "--no-arena")) {
        std::printf("  expert arena   not built (--no-arena)\n");
        return 0;
    }

    // ---- the expert arena.  Its size is decided AFTER the dense weights are in AND after the forward pass has
    // its scratch, so the resident bytes in the line below are measured rather than assumed (experts.h).  That
    // ordering is required: a prefill chunk of 1024 positions is about 1.5 GB of scratch, and an arena planned
    // before it was allocated would overcommit by that much.
    if (const std::string err = strata::hy4_flash::open_experts(m); !err.empty()) {
        std::fprintf(stderr, "strata_hy4_flash_metal: %s\n", err.c_str());
        return 2;
    }

    // ---- what the forward pass will be asked for (the flags are parsed before the arena, see above)
    const char* block_arg = flag_value(argc, argv, "--block-test");
    const bool want_head = has_flag(argc, argv, "--head-test");
    const bool want_forward = tok_path != nullptr || want_head;
    int T = block_arg != nullptr ? kMaxWindowT : 1;
    if (const char* v = flag_value(argc, argv, "--window-t")) T = std::atoi(v);
    if (T < 1 || T > kMaxWindowT) {
        std::fprintf(stderr, "strata_hy4_flash_metal: --window-t %d is outside 1..%d\n", T, kMaxWindowT);
        return 2;
    }
    int chunk = has_flag(argc, argv, "--prefill") ? kPrefillChunkDefault : 0;
    if (const char* v = flag_value(argc, argv, "--prefill-chunk")) chunk = std::atoi(v);
    const bool want_consistency = has_flag(argc, argv, "--prefill-consistency");
    if (want_consistency && chunk <= 0) chunk = kPrefillChunkDefault;
    if (chunk < 0 || chunk > kContextMax) {
        std::fprintf(stderr, "strata_hy4_flash_metal: --prefill-chunk %d is outside 0..%d\n", chunk, kContextMax);
        return 2;
    }
    if (chunk > 0 && chunk % m.geo.idx_kpool != 0) {
        // a kpool group is 4 positions and a pooled key is written when its group fills; a chunk that ends inside
        // a group is handled, but then the group is pooled only in the NEXT chunk, which no test covers
        std::fprintf(stderr, "strata_hy4_flash_metal: --prefill-chunk %d is not a multiple of the indexer's kpool (%d)\n",
                     chunk, m.geo.idx_kpool);
        return 2;
    }

    strata::hy4_flash::Forward fw;
    strata::hy4_flash::Mtp mtp;
    if (want_forward || block_arg != nullptr) {
        const strata::hy4_flash::Prefill pfc = strata::hy4_flash::Prefill::from_env(chunk);
        if (const std::string err = fw.open(m, cx, pack_dir, pfc); !err.empty()) {
            std::fprintf(stderr, "strata_hy4_flash_metal: the forward pass could not start: %s\n", err.c_str());
            return 2;
        }
        std::printf("  forward        kernels compiled; window T = %d, %s read-ahead; prefill %s\n", T,
                    strata::hy4_flash::prefetch_enabled() ? "cross-layer" : "no",
                    chunk > 0 ? ("chunks of " + std::to_string(chunk)).c_str() : "off (the decode path)");
        if (want_mtp) {
            if (const std::string err = mtp.open(m, cx, fw.library()); !err.empty()) {
                std::fprintf(stderr, "strata_hy4_flash_metal: the MTP block could not start: %s\n", err.c_str());
                return 2;
            }
            std::printf("  mtp            k = %d drafts a round, expert pool %.2f GB\n", draft_k,
                        (double) mtp.expert_pool_bytes() / 1e9);
        }
        std::printf("  buffers        %.3f GB of Metal allocations before the arena is planned\n",
                    (double) cx.allocated() / 1e9);
        std::fflush(stdout);
    }

    const strata::hy4_flash::ArenaPlan plan = strata::hy4_flash::plan_arena(m, cx);
    std::printf("  %s\n", strata::hy4_flash::arena_line(plan).c_str());
    std::fflush(stdout);
    std::string profile;
    if (const char* p = flag_value(argc, argv, "--expert-profile")) profile = p;
    else if (const char* p = std::getenv("STRATA_METAL_EXPERT_PROFILE")) profile = p;
    else if (const char* pack = flag_value(argc, argv, "--pack")) profile = std::string(pack) + "/expert-profile.bin";
    else profile = "(none)";
    // A PREFILL CHUNK NEEDS A FLOOR ON THE SLOTS OF EVERY SIZE CLASS, not just on the arena's bytes.  Its layer
    // asks for all 512 experts of that layer at once and then streams the next layer's 512 in behind the compute,
    // and both generations are pinned until the following handshake - so one class must hold 2 x 512 slots plus a
    // margin or `ExpertStream::victim` finds nothing and ends the process ("no free slot").  min_slots raises the
    // arena past the byte budget if it has to (stream.cpp lays the floor out before the ranking), so the line
    // below prints what the arena really became.
    const int min_slots = chunk > 0 ? 2 * m.stream.n_experts() + 64 : 64;
    if (const std::string err = strata::hy4_flash::make_arena(m, cx, plan, profile, min_slots); !err.empty()) {
        std::fprintf(stderr, "strata_hy4_flash_metal: %s\n", err.c_str());
        return 2;
    }
    if (chunk > 0)
        for (int c = 0; c < m.stream.n_classes(); ++c)
            if (m.stream.class_slots(c) < 2 * m.stream.n_experts()) {
                std::fprintf(stderr, "strata_hy4_flash_metal: prefill needs %d slots in every expert size class and class %d "
                                     "has %d; raise STRATA_METAL_ARENA_GB (a chunk asks for a whole layer at "
                                     "once)\n", 2 * m.stream.n_experts(), c, m.stream.class_slots(c));
                return 2;
            }
    if (chunk > 0)
        if (const std::string err = fw.attach_prefill(); !err.empty()) {
            std::fprintf(stderr, "strata_hy4_flash_metal: %s\n", err.c_str());
            return 2;
        }


    std::printf("  expert supply  %d ordinals x %d experts, %.3f MB each on average; arena holds %.3f GB\n",
                m.stream.n_layers(), m.stream.n_experts(),
                plan.expert_total / m.stream.n_layers() / m.stream.n_experts() / 1e6,
                (double) m.stream.arena_used_bytes() / 1e9);

    // ---- the supply itself: a routing trace replayed through the arena.  This is not the forward pass; it is
    // the half of it that checks the experts a decode window asks for are there, in the arena, with the file's
    // bytes in them.
    int rc = 0;
    if (const char* tp = flag_value(argc, argv, "--expert-test")) {
        Trace tr;
        if (const std::string err = read_trace(tp, tr); !err.empty()) {
            std::fprintf(stderr, "strata_hy4_flash_metal: %s\n", err.c_str());
            return 2;
        }
        if ((int) tr.n_ord < m.stream.n_layers()) {
            std::fprintf(stderr, "strata_hy4_flash_metal: %s has %u ordinals, the model has %d\n", tp, tr.n_ord,
                         m.stream.n_layers());
            return 2;
        }
        int wt = 1, windows = 64, pred_wrong = 0;
        if (const char* v = flag_value(argc, argv, "--window-t")) wt = std::atoi(v);
        if (const char* v = flag_value(argc, argv, "--windows")) windows = std::atoi(v);
        if (const char* v = flag_value(argc, argv, "--pred-wrong")) pred_wrong = std::atoi(v);
        if (wt < 1 || wt > kMaxWindowT) {
            std::fprintf(stderr, "strata_hy4_flash_metal: --window-t %d is outside 1..%d\n", wt, kMaxWindowT);
            return 2;
        }
        std::printf("  expert test    %s: %u positions, topk %u; %d windows of T = %d, %d predictions spoiled\n",
                    tp, tr.n_pos, tr.topk, windows, wt, pred_wrong);
        std::fflush(stdout);
        m.stream.reset_stats();
        rc = replay(m, tr, wt, windows, pred_wrong, has_flag(argc, argv, "--verify"));
        m.stream.report(stdout);
        if (const std::string bad = m.stream.check_invariants(); !bad.empty()) {
            std::fprintf(stderr, "strata_hy4_flash_metal: the arena's invariant broke: %s\n", bad.c_str());
            rc = rc ? rc : 5;
        }
        return rc;
    }

    // ---- the forward pass ----------------------------------------------------------------------------------------
    if (!want_forward) {
        std::printf("  nothing asked  give --tokens (with --logits-out / --greedy / --block-test), or --head-test\n");
        return 0;
    }


    // ---- --head-test: the output head alone, from an injected final HC state
    if (want_head) {
        const char* in = flag_value(argc, argv, "--in");
        if (in == nullptr) {
            std::fprintf(stderr, "strata_hy4_flash_metal: --head-test needs --in <l_out41.f32>\n");
            return 2;
        }
        std::string err;
        const std::vector<uint8_t> raw = read_all(in, err);
        if (!err.empty()) {
            std::fprintf(stderr, "strata_hy4_flash_metal: %s\n", err.c_str());
            return 2;
        }
        const size_t per = 4 * (size_t) m.geo.hc_width();
        const int n = (int) (raw.size() / per);
        if (n < 1) {
            std::fprintf(stderr, "strata_hy4_flash_metal: %s holds no whole HC state (%zu B, one position is %zu B)\n", in,
                         raw.size(), per);
            return 2;
        }
        strata::hy4_flash::Dump dump;
        if (const char* dd = flag_value(argc, argv, "--dump")) {
            std::vector<int32_t> tk = tokens;
            if ((int) tk.size() != n) tk.assign((size_t) n, 0);
            if (const std::string e = dump.open(dd, tk); !e.empty()) {
                std::fprintf(stderr, "strata_hy4_flash_metal: %s\n", e.c_str());
                return 2;
            }
        }
        // `T` (--window-t), not kMaxWindowT: hard-wiring 8 here would make a T-invariance check on the head
        // compare two identical runs - a vacuous pass.
        std::printf("  head test      %s: %d positions, windows of %d\n", in, n, T);
        for (int p = 0; p < n; p += T) {
            const int nt = std::min(T, n - p);
            if (const std::string e = fw.head_window((const float*) (raw.data() + (size_t) p * per), nt, &dump);
                !e.empty()) {
                std::fprintf(stderr, "strata_hy4_flash_metal: head at %d: %s\n", p, e.c_str());
                return 4;
            }
        }
        if (const std::string e = dump.close(); !e.empty()) {
            std::fprintf(stderr, "strata_hy4_flash_metal: %s\n", e.c_str());
            return 2;
        }
        std::printf("  head test      done\n");
        return 0;
    }

    // ---- --block-test: one block, from the reference's own input, dumping the reference's node names
    if (block_arg != nullptr) {
        const int L = std::atoi(block_arg);
        const char* in = flag_value(argc, argv, "--in");
        const char* dd = flag_value(argc, argv, "--dump");
        if (in == nullptr || dd == nullptr) {
            std::fprintf(stderr, "strata_hy4_flash_metal: --block-test needs --in <hc.f32> and --dump <dir>\n");
            return 2;
        }
        std::string err;
        const std::vector<uint8_t> hcraw = read_all(in, err);
        if (!err.empty()) {
            std::fprintf(stderr, "strata_hy4_flash_metal: %s\n", err.c_str());
            return 2;
        }
        const size_t per = 4 * (size_t) m.geo.hc_width();
        const int n = (int) (hcraw.size() / per);
        if (n != (int) tokens.size()) {
            std::fprintf(stderr, "strata_hy4_flash_metal: %s holds %d positions but --tokens has %zu\n", in, n,
                         tokens.size());
            return 2;
        }
        std::vector<int32_t> ftopk, fsel, fnsel;
        // Both injection files are untrusted inputs that end up as INDICES (expert ids, cell numbers).  They are
        // validated here, where the message can name the file, and again in forward.cpp's at_cut for the ids,
        // which is the last point before anything indexes with them.
        if (const char* f = flag_value(argc, argv, "--force-topk")) {
            const std::vector<uint8_t> raw = read_all(f, err);
            if (!err.empty()) {
                std::fprintf(stderr, "strata_hy4_flash_metal: %s\n", err.c_str());
                return 2;
            }
            ftopk.assign((const int32_t*) raw.data(), (const int32_t*) (raw.data() + raw.size()));
            if ((int) ftopk.size() != n * m.geo.top_k) {
                std::fprintf(stderr, "strata_hy4_flash_metal: %s holds %zu ids, want %d x %d\n", f, ftopk.size(), n,
                             m.geo.top_k);
                return 2;
            }
            for (size_t i = 0; i < ftopk.size(); ++i)
                if (ftopk[i] < 0 || ftopk[i] >= m.geo.n_experts) {
                    std::fprintf(stderr, "strata_hy4_flash_metal: ERR %s entry %zu is expert %d, outside 0..%d\n", f, i,
                                 ftopk[i], m.geo.n_experts - 1);
                    return 2;
                }
        }
        if (const char* f = flag_value(argc, argv, "--force-sel")) {
            // The reference's indexer_sel is a [T][n_kv] f16 MASK, 0 = attended, and it can name up to 2051
            // cells - past the 2048 the engine's own selection ever produces, which is why the injection
            // buffer is kSelCap wide.
            const std::vector<uint8_t> raw = read_all(f, err);
            if (!err.empty()) {
                std::fprintf(stderr, "strata_hy4_flash_metal: %s\n", err.c_str());
                return 2;
            }
            const size_t n_kv = raw.size() / 2 / (size_t) n;
            const uint16_t* mask = (const uint16_t*) raw.data();
            fsel.assign((size_t) n * strata::hy4_flash::Forward::kSelCap, 0);
            fnsel.assign((size_t) n, 0);
            for (int t = 0; t < n; ++t) {
                int k = 0;
                for (size_t c = 0; c < n_kv; ++c)
                    if (ref::f16_to_f32(mask[(size_t) t * n_kv + c]) == 0.0f) {
                        if (k >= strata::hy4_flash::Forward::kSelCap) {
                            std::fprintf(stderr, "strata_hy4_flash_metal: position %d selects more than %d cells\n", t,
                                         strata::hy4_flash::Forward::kSelCap);
                            return 2;
                        }
                        fsel[(size_t) t * strata::hy4_flash::Forward::kSelCap + k++] = (int32_t) c;
                    }
                fnsel[(size_t) t] = k;
            }
            // The cells index the KV / indexer cache on the GPU, which has `context` positions and no bounds
            // checking.  The mask's own width decides them, so a mask from a longer run would read past the end.
            for (int t = 0; t < n; ++t)
                for (int k = 0; k < fnsel[(size_t) t]; ++k) {
                    const int32_t c = fsel[(size_t) t * strata::hy4_flash::Forward::kSelCap + k];
                    if (c < 0 || c >= context) {
                        std::fprintf(stderr, "strata_hy4_flash_metal: ERR %s position %d selects cell %d, outside "
                                             "0..%d (--context)\n", f, t, c, context - 1);
                        return 2;
                    }
                }
        }
        strata::hy4_flash::Dump dump;
        if (const std::string e = dump.open(dd, tokens); !e.empty()) {
            std::fprintf(stderr, "strata_hy4_flash_metal: %s\n", e.c_str());
            return 2;
        }
        std::printf("  block test     block %d, %d positions, windows of %d%s%s\n", L, n, T,
                    ftopk.empty() ? "" : ", routing injected", fsel.empty() ? "" : ", selection injected");
        std::fflush(stdout);
        fw.reset();
        const double t0 = mtl::now_s();
        for (int p = 0; p < n; p += T) {
            const int nt = std::min(T, n - p);
            strata::hy4_flash::Inject inj;
            inj.hc = (const float*) (hcraw.data() + (size_t) p * per);
            if (!ftopk.empty()) inj.topk = ftopk.data() + (size_t) p * m.geo.top_k;
            if (!fsel.empty()) {
                inj.sel = fsel.data() + (size_t) p * strata::hy4_flash::Forward::kSelCap;
                inj.n_sel = fnsel.data() + (size_t) p;
            }
            if (const std::string e = fw.block_window(L, tokens.data(), p, nt, inj, &dump); !e.empty()) {
                if (const int rc = nonfinite_or_fail(e, "position", p); rc >= 0) {
                    dump.close();
                    return rc;
                }
                std::fprintf(stderr, "strata_hy4_flash_metal: block %d at position %d: %s\n", L, p, e.c_str());
                return 4;
            }
            if (p % 128 == 0) {
                std::printf("    %d/%d (%.1f s)\n", p, n, mtl::now_s() - t0);
                std::fflush(stdout);
            }
        }
        // The running total only; the DETECTION is per window above, so a non-zero here with a zero exit status
        // would be a bug in that path rather than a warning.
        if (const uint32_t bad = fw.cache_nonfinite(); bad != 0)
            std::printf("  nonfinite      %u in all (every window was checked as it finished)\n", bad);
        if (const std::string e = dump.close(); !e.empty()) {
            std::fprintf(stderr, "strata_hy4_flash_metal: %s\n", e.c_str());
            return 2;
        }
        std::printf("  block test     done in %.1f s; the dump is in %s\n", mtl::now_s() - t0, dd);
        m.stream.report(stdout);
        return 0;
    }

    // ---- the whole trunk: teacher-forced logits, then an optional greedy continuation
    const char* lout = flag_value(argc, argv, "--logits-out");
    int gen = 0;
    if (const char* v = flag_value(argc, argv, "--greedy")) gen = std::atoi(v);
    const bool pred_acc = has_flag(argc, argv, "--pred-accuracy");
    const int nprompt = (int) tokens.size();
    std::vector<std::pair<int, int>> ranges;   // --logits-range; empty = every position
    if (const char* v = flag_value(argc, argv, "--logits-range")) {
        std::string e;
        ranges = parse_ranges(v, nprompt, e);
        if (!e.empty()) {
            std::fprintf(stderr, "strata_hy4_flash_metal: %s\n", e.c_str());
            return 2;
        }
    }
    // BEFORE opening anything: --load-prefill skips the prompt, and these four describe the prompt.  Checked
    // here rather than next to the load because `fopen(lout, "wb")` below TRUNCATES its target, so a rejected
    // command would destroy a reference logits file on its way to the error.
    if (has_flag(argc, argv, "--load-prefill")) {
        const char* why = lout != nullptr                                       ? "--logits-out"
                          : flag_value(argc, argv, "--dump") != nullptr          ? "--dump"
                          : has_flag(argc, argv, "--prefill-consistency")        ? "--prefill-consistency"
                          : has_flag(argc, argv, "--pred-accuracy")              ? "--pred-accuracy"
                                                                                 : nullptr;
        if (why != nullptr) {
            std::fprintf(stderr, "strata_hy4_flash_metal: --load-prefill skips the prompt, so %s would have nothing to "
                                 "report - run them without it\n", why);
            return 2;
        }
    }
    FILE* lf = nullptr;
    if (lout != nullptr) {
        lf = std::fopen(lout, "wb");
        if (lf == nullptr) {
            std::fprintf(stderr, "strata_hy4_flash_metal: cannot write %s\n", lout);
            return 2;
        }
    }
    const int32_t vocab = m.geo.vocab_size;
    // One place decides whether a position's logits are written, so the decode path and the prefill path cannot
    // disagree about what a --logits-range run contains.
    auto wanted = [&](int pos) {
        if (ranges.empty()) return true;
        for (const auto& r : ranges)
            if (pos >= r.first && pos < r.second) return true;
        return false;
    };
    auto write_logits = [&](int pos, int n, const float* lg) {
        if (lf == nullptr) return;
        for (int i = 0; i < n; ++i)
            if (wanted(pos + i)) std::fwrite(lg + (size_t) i * (size_t) vocab, 4, (size_t) vocab, lf);
    };
    // --dump on the whole-trunk path writes every MoE block's routing and logits
    strata::hy4_flash::Dump tdump;
    if (const char* dd = flag_value(argc, argv, "--dump"))
        if (const std::string e = tdump.open(dd, tokens); !e.empty()) {
            std::fprintf(stderr, "strata_hy4_flash_metal: %s\n", e.c_str());
            return 2;
        }
    std::vector<int32_t> seq = tokens;
    uint64_t pred_hit = 0, pred_tot = 0;
    const int topk = m.geo.top_k;
    fw.reset();

    // ---- the prefill snapshot (prefill_snap.h): run the prompt once, measure decode many times
    const char* snap_out = flag_value(argc, argv, "--save-prefill");
    const char* snap_in = flag_value(argc, argv, "--load-prefill");
    strata::hy4_flash::SnapBinding sb;
    std::vector<strata::hy4_flash::SnapSlice> snap_extra;
    if (snap_out != nullptr || snap_in != nullptr) {
        if (tok_path == nullptr) {
            std::fprintf(stderr, "strata_hy4_flash_metal: --save-prefill / --load-prefill need --tokens (the prompt is "
                                 "half of what binds a snapshot to a state)\n");
            return 2;
        }
        // A file's identity: size, mtime to the nanosecond, and a digest of its first and last MB.  Not the
        // whole file - 8 GB would cost more than the prefill this saves - so a same-size edit confined to the
        // middle, with the timestamp preserved, is a residual risk this does not catch.
        auto ident = [&](const char* fp, uint64_t& bytes, int64_t& mt, int64_t& mt_ns, std::string* edges,
                         std::string& err) {
            struct stat st {};
            if (::stat(fp, &st) != 0) { err = std::string("cannot stat ") + fp; return false; }
            bytes = (uint64_t) st.st_size;
            mt = (int64_t) st.st_mtime;
            mt_ns = (int64_t) st.st_mtimespec.tv_nsec;
            if (edges == nullptr) return true;
            std::FILE* f = std::fopen(fp, "rb");
            if (f == nullptr) { err = std::string("cannot read ") + fp; return false; }
            const size_t edge = bytes < (1u << 20) ? (size_t) bytes : (size_t) (1u << 20);
            std::vector<unsigned char> buf(2 * edge);
            bool ok = std::fread(buf.data(), 1, edge, f) == edge;
            ok = ok && std::fseek(f, (long) (bytes - edge), SEEK_SET) == 0;
            ok = ok && std::fread(buf.data() + edge, 1, edge, f) == edge;
            std::fclose(f);
            if (!ok) { err = std::string("cannot read the ends of ") + fp; return false; }
            *edges = strata::hy4_flash::sha256_hex(buf.data(), buf.size());
            return true;
        };
        std::string ierr;
        if (!ident(path, sb.model_bytes, sb.model_mtime, sb.model_mtime_ns, &sb.model_edges_sha, ierr)) {
            std::fprintf(stderr, "strata_hy4_flash_metal: %s\n", ierr.c_str());
            return 2;
        }
        const std::string mp(path);
        sb.model_name = mp.substr(mp.find_last_of('/') == std::string::npos ? 0 : mp.find_last_of('/') + 1);
        // The Engram shard and the pack's token map are NOT in the model file, and they decide which rows the
        // prompt read - so a snapshot taken with a different shard or map describes different state while
        // every other field matches.
        if (!m.engram_path.empty() &&
            !ident(m.engram_path.c_str(), sb.engram_bytes, sb.engram_mtime, sb.engram_mtime_ns, nullptr, ierr)) {
            std::fprintf(stderr, "strata_hy4_flash_metal: %s\n", ierr.c_str());
            return 2;
        }
        if (const char* pk = flag_value(argc, argv, "--pack")) {
            const std::string tm = std::string(pk) + "/engram_token_map.bin";
            std::FILE* f = std::fopen(tm.c_str(), "rb");
            if (f != nullptr) {
                std::vector<unsigned char> all;
                unsigned char chunk_buf[1 << 16];
                size_t got = 0;
                while ((got = std::fread(chunk_buf, 1, sizeof chunk_buf, f)) > 0)
                    all.insert(all.end(), chunk_buf, chunk_buf + got);
                std::fclose(f);
                sb.token_map_sha = strata::hy4_flash::sha256_hex(all.data(), all.size());
            }
        }
        // Deliberate wrongness: each of these changes the prompt's arithmetic, and a normal run prints no
        // warning because its own switches are unset - so a mutated snapshot would pass for a clean one.
        // WHAT BELONGS HERE: a switch that can change THE BYTES THIS SNAPSHOT HOLDS, which means the prompt's
        // arithmetic or its execution path.  HY4_FLASH_MUT and HY4_FLASH_PREFILL_PROBE are deliberate wrongness;
        // HY4_FLASH_FORCE_SEL replaces the indexer's selection; the three sub-batch sizes change how a chunk is
        // split (proven not to change the result, but they change the path the saved state came through).
        //
        // WHAT DOES NOT: anything that only changes WHEN bytes are read or in what order work is issued.
        // HY4_FLASH_PREFETCH and HY4_FLASH_OVERLAP change WHEN bytes are read, never which (the engine says so at
        // start-up), so they cannot alter the prompt's state - and binding them would refuse legitimate runs of
        // the very sweep a snapshot is built to make cheap.  HY4_FLASH_MTP_PROBE affects drafting, which happens
        // during decode, after everything a snapshot holds.
        // The two prefill MATRIX-UNIT kernels change the prompt's VALUES, not only its path - the MMA attention
        // sums in a different fp32 order than the decode kernels (forward.h), the dense MMA likewise (dispatch.h)
        // - and they are recorded BELOW this loop, unconditionally, because both default to ON: a switch listed
        // here is recorded only when it is set, so with the default the mode string would stay EMPTY and a
        // snapshot written before either kernel existed would load into a build that has it with every field
        // matching, while the state it holds came from the old arithmetic.  HY4_FLASH_PREFILL_IDX is listed like the
        // other sub-batch sizes, and HY4_FLASH_PREFILL_NOISE perturbs every block's output on purpose
        // (--prefill-consistency's benign control).
        for (const char* e : {"HY4_FLASH_MUT", "HY4_FLASH_PREFILL_PROBE", "HY4_FLASH_PREFILL_NOISE", "HY4_FLASH_PREFILL_ATTN",
                              "HY4_FLASH_PREFILL_MOE", "HY4_FLASH_PREFILL_HEAD", "HY4_FLASH_PREFILL_IDX", "HY4_FLASH_FORCE_SEL"}) {
            const char* v = std::getenv(e);
            if (v != nullptr && *v != 0) sb.mode += std::string(sb.mode.empty() ? "" : ",") + e + "=" + v;
        }
        // (the requested setting, not the shape-by-shape "did it qualify": the shapes follow from the model,
        // which the identity fields above already bind)
        const int pf_mma = strata::hy4_flash::Prefill::from_env(chunk).mma;
        sb.mode += std::string(sb.mode.empty() ? "" : ",") + "mma=" + std::to_string(pf_mma) + ",dense_mm=" +
                   (!strata::hy4_flash::prefill_mm_enabled() ? "0" : strata::hy4_flash::prefill_mm_half() ? "half" : "1");
        // The routed experts' prefill GEMM, for the same reason and in the same place: it defaults to ON, so a
        // switch recorded only when SET would leave the mode string unable to tell this build's arithmetic from
        // the GEMV engine that wrote an older snapshot.  `v1` is the arithmetic's revision, so a later change to
        // the GEMM can invalidate these snapshots in turn.
        sb.mode += ",moe_mm=v1:" + std::string(strata_hy4_flash_metal::prefill_moe_mm_name());
        sb.tokens_sha = strata::hy4_flash::sha256_hex(tokens.data(), 4 * tokens.size());
        sb.n_tokens = nprompt;
        sb.chunk = chunk;
        // The MTP's `h` history is the caller's state, not the engine's: it leaves through PrefillSink::hidden
        // and lives in the Mtp object.  Row q is h_{q-1} and hidden(0) is zero, so rows 0..nprompt are what a
        // continuation from position nprompt reads, and they are contiguous from hidden(0).
        if (want_mtp) {
            sb.mtp_rows = nprompt + 1;
            snap_extra.push_back({"mtp_hidden", mtp.hidden(0),
                                  4 * (uint64_t) sb.mtp_rows * (uint64_t) m.geo.hidden_dim});
        }
    }
    bool snap_loaded = false;
    if (snap_in != nullptr) {
        // (the four prompt-describing flags were refused above, before --logits-out could truncate its target)
        const double t0 = mtl::now_s();
        if (const std::string e = fw.snapshot_load(snap_in, sb, snap_extra); !e.empty()) {
            std::fprintf(stderr, "strata_hy4_flash_metal: %s\n", e.c_str());
            return 2;
        }
        snap_loaded = true;
        const int32_t used = fw.snapshot_used();
        std::printf("  prefill snap   loaded %s in %.2f s: %d positions, next token %d (the prompt did NOT "
                    "run; the expert arena holds the profile's experts, not the prompt's)\n",
                    snap_in, mtl::now_s() - t0, (int) used, (int) fw.last_token());
        if (used != nprompt) {
            // Cannot happen - `used` is compared against nothing, but n_tokens is part of the binding and the
            // prompt is run to completion before a save.  If it ever does, the generation loop below would
            // start at the wrong position, so it stops here instead.
            std::fprintf(stderr, "strata_hy4_flash_metal: the snapshot holds %d positions but the prompt is %d tokens\n",
                         (int) used, nprompt);
            return 2;
        }
    }

    // ---- the prompt through PREFILL
    if (chunk > 0 && !snap_loaded) {
        // --prefill-consistency keeps the compared positions' logits so the decode path can be compared against
        // them below: every --consistency-stride S-th position and the last 64 (the next-token logits among
        // them).  S = 1 is every position.
        int cstride = 1;
        if (const char* v = flag_value(argc, argv, "--consistency-stride")) cstride = std::max(1, std::atoi(v));
        auto compared = [&](int pos) { return pos % cstride == 0 || pos >= nprompt - 64; };
        std::vector<int> kslot;   // position -> its row in `keep`, or -1
        std::vector<float> keep;
        if (want_consistency) {
            kslot.assign((size_t) nprompt, -1);
            int nk = 0;
            for (int p = 0; p < nprompt; ++p)
                if (compared(p)) kslot[(size_t) p] = nk++;
            const double gb = (double) nk * (double) vocab * 4 / 1e9;
            if (gb > 4.0) {
                std::fprintf(stderr, "strata_hy4_flash_metal: --prefill-consistency on %d positions needs %.1f GB of host "
                                     "memory for the logits; raise --consistency-stride\n", nk, gb);
                return 2;
            }
            keep.assign((size_t) nk * (size_t) vocab, 0.0f);
        }
        strata::hy4_flash::Forward::PrefillSink sink;
        sink.all_positions = lf != nullptr || want_consistency;
        // the MTP block's h input: hidden(q) = the trunk's result_norm at q - 1 (mtp.h)
        if (want_mtp) sink.hidden = [&](int pos, int n, const float* h) { mtp.set_hidden(pos + 1, n, h); };
        sink.logits = [&](int pos, int n, const float* lg) {
            write_logits(pos, n, lg);
            if (!keep.empty())
                for (int i = 0; i < n; ++i)
                    if (const int k = kslot[(size_t) (pos + i)]; k >= 0)
                        std::memcpy(&keep[(size_t) k * (size_t) vocab], lg + (size_t) i * (size_t) vocab,
                                    4 * (size_t) vocab);
        };
        // The routing of the two runs.  Always recorded under --prefill-consistency (41 x n x 16 ids): the flip
        // rate - the (position, ordinal) top-k SETS that differ - is what turns a tiny perturbation into a large
        // KL, because the routing is so often a near tie.  --consistency-force-route runs the decode path over
        // the prompt FIRST and makes prefill use its routing (with prefill's own probabilities behind the
        // weights, as --force-topk does), so the judge sees the arithmetic alone.
        std::vector<int32_t> pf_route, dec_route;
        const bool force_route = want_consistency && has_flag(argc, argv, "--consistency-force-route");
        const int n_ord = m.stream.n_layers();
        if (force_route) {
            const double tr = mtl::now_s();
            dec_route.assign((size_t) n_ord * (size_t) nprompt * (size_t) topk, -1);
            for (int p = 0; p < nprompt;) {
                const int nt = std::min(T, nprompt - p);
                if (const std::string e = fw.window(seq.data(), p, nt); !e.empty()) {
                    std::fprintf(stderr, "strata_hy4_flash_metal: routing pre-pass at %d: %s\n", p, e.c_str());
                    return 4;
                }
                const std::vector<int32_t>& r = fw.routed_ids();
                for (int o = 0; o < n_ord; ++o)
                    std::memcpy(&dec_route[((size_t) o * nprompt + p) * topk], &r[(size_t) o * nt * topk],
                                4 * (size_t) nt * topk);
                p += nt;
            }
            fw.reset();
            std::printf("  consistency    routing pre-pass: the decode path's top-%d over %d positions in %.1f s; prefill "
                        "will use it (--consistency-force-route)\n", topk, nprompt, mtl::now_s() - tr);
        }
        if (want_consistency) fw.set_prompt_routing(&pf_route, force_route ? dec_route.data() : nullptr, nprompt);
        const double t0 = mtl::now_s();
        const std::string pe = fw.prompt(seq.data(), nprompt, sink, &tdump);
        fw.set_prompt_routing(nullptr, nullptr, 0);
        if (const std::string& e = pe; !e.empty()) {
            std::fprintf(stderr, "strata_hy4_flash_metal: %s\n", e.c_str());
            if (lf != nullptr) std::fclose(lf);
            return 4;
        }
        const double pf_s = mtl::now_s() - t0;
        const strata::hy4_flash::Timing& tm = fw.timing();
        // prefill tok/s: the prompt's tokens divided by the wall time to the first token's logits.
        std::printf("  prefill        %d positions in %.2f s = %.2f tok/s (chunks of %d, %llu of them)\n",
                    nprompt, pf_s, nprompt / pf_s, chunk, (unsigned long long) tm.pf_chunks);
        std::printf("  prefill split  per chunk: gpu %.0f ms, host at the cuts %.0f ms (of it SSD wait %.0f ms), "
                    "encode %.0f ms, head %.0f ms, wall %.0f ms\n",
                    tm.pf_gpu_ms / (double) std::max<uint64_t>(1, tm.pf_chunks),
                    tm.pf_host_ms / (double) std::max<uint64_t>(1, tm.pf_chunks),
                    tm.pf_wait_ms / (double) std::max<uint64_t>(1, tm.pf_chunks),
                    tm.pf_encode_ms / (double) std::max<uint64_t>(1, tm.pf_chunks),
                    tm.pf_head_ms / (double) std::max<uint64_t>(1, tm.pf_chunks),
                    tm.pf_wall_ms / (double) std::max<uint64_t>(1, tm.pf_chunks));
        std::printf("  prefill engram %.0f ms in all = %.1f ms per chunk, %.3f ms per prompt position (kept OUT "
                    "of the decode counter: it used to be divided by the DECODE position count)\n",
                    tm.pf_engram_ms, tm.pf_engram_ms / (double) std::max<uint64_t>(1, tm.pf_chunks),
                    tm.pf_engram_ms / (double) std::max<uint64_t>(1, tm.pf_positions));
        std::printf("  prefill io     %.2f GB delivered by the SSD while the prompt ran = %.2f GB per chunk\n",
                    (double) tm.pf_bytes_read / 1e9,
                    (double) tm.pf_bytes_read / 1e9 / (double) std::max<uint64_t>(1, tm.pf_chunks));
        if (!tm.pf_block_ms.empty()) {
            // where the GPU time goes, by block kind: block 0 is the dense FFN, the sliding blocks have 64
            // attention heads and no indexer, the sparse ones 32 heads and the indexer
            double sl = 0, sp = 0, de = 0;
            int nsl = 0, nsp = 0, nde = 0;
            for (size_t i = 0; i < tm.pf_block_ms.size(); ++i) {
                const strata::hy4_flash::LayerSpec& ls = m.geo.block((int) i);
                const bool dense = ls.expert_ordinal < 0;
                if (dense) { de += tm.pf_block_ms[i]; ++nde; }
                else if (ls.attention == strata::hy4_flash::Attention::Sliding) { sl += tm.pf_block_ms[i]; ++nsl; }
                else { sp += tm.pf_block_ms[i]; ++nsp; }
            }
            const double c = (double) std::max<uint64_t>(1, tm.pf_chunks);
            std::printf("  prefill blocks per chunk: %d dense-FFN %.0f ms (%.0f each), %d sliding %.0f ms "
                        "(%.0f each), %d sparse %.0f ms (%.0f each)\n",
                        nde, de / c, nde ? de / c / nde : 0.0, nsl, sl / c, nsl ? sl / c / nsl : 0.0, nsp, sp / c,
                        nsp ? sp / c / nsp : 0.0);
        }
        if (lf != nullptr) {
            std::fclose(lf);
            lf = nullptr;
            std::printf("  logits         written to %s\n", lout);
        }
        if (const uint32_t bad = fw.cache_nonfinite(); bad != 0)
            std::printf("  WARNING        %u values written to an f16 cache were not finite (ADR §4.2)\n", bad);

        // ---- the consistency judge: the same prompt again, one position at a time through the decode path.
        // Two verdicts: BITWISE (still printed) and the TOLERANCE judge (prefill_tol.h) that an MMA prefill is
        // held to - KL / top-1 against the decode path, thresholds from --consistency-tol FILE.
        // CONTINUATION: --consistency-cont N (default 8) positions decoded after the prompt, teacher-forced with
        // the same tokens on both sides (the prefill side's own greedy picks), so the state the prompt LEFT -
        // the caches a generation reads - is judged too, not only the prompt's logits.
        if (want_consistency) {
            int ncont = 8;
            if (const char* v = flag_value(argc, argv, "--consistency-cont")) ncont = std::max(0, std::atoi(v));
            ncont = std::min(ncont, context - nprompt);
            std::vector<int32_t> cseq = seq;
            cseq.resize((size_t) nprompt);
            std::vector<float> ckeep((size_t) ncont * (size_t) vocab);
            {
                int32_t next = strata::hy4_flash::tol_argmax(&keep[(size_t) kslot[(size_t) nprompt - 1] * (size_t) vocab],
                                                       vocab);
                for (int i = 0; i < ncont; ++i) {
                    cseq.push_back(next);
                    if (const std::string e = fw.window(cseq.data(), nprompt + i, 1); !e.empty()) {
                        std::fprintf(stderr, "strata_hy4_flash_metal: continuation at %d: %s\n", nprompt + i, e.c_str());
                        return 4;
                    }
                    std::memcpy(&ckeep[(size_t) i * (size_t) vocab], fw.logits(), 4 * (size_t) vocab);
                    next = strata::hy4_flash::tol_argmax(fw.logits(), vocab);
                }
            }
            fw.reset();
            std::printf("  consistency    the same %d positions again through the decode path (window T = %d), "
                        "compared at every %d-th and the last 64, then %d continuation positions\n",
                        nprompt, T, cstride, ncont);
            std::fflush(stdout);
            uint64_t diff_floats = 0, diff_pos = 0;
            int first_bad = -1;
            double max_abs = 0, max_rel = 0;
            uint64_t argmax_diff = 0;
            uint64_t flips = 0, flip_pos = 0, route_n = 0;   // (position, ordinal) top-k sets that differ
            std::vector<strata::hy4_flash::TolPos> tol;
            auto bitwise_count = [&](int pos, const float* a, const float* b) {
                uint64_t d = 0;
                for (int32_t v = 0; v < vocab; ++v)
                    if (a[v] != b[v]) {
                        ++d;
                        const double ad = std::fabs((double) a[v] - (double) b[v]);
                        max_abs = std::max(max_abs, ad);
                        const double den = std::max(std::fabs((double) a[v]), 1e-30);
                        max_rel = std::max(max_rel, ad / den);
                    }
                if (d != 0) {
                    diff_floats += d;
                    ++diff_pos;
                    if (first_bad < 0) first_bad = pos;
                }
            };
            const double td = mtl::now_s();
            const int total = nprompt + ncont;
            for (int p = 0; p < total;) {
                const int nt = p < nprompt ? std::min(T, nprompt - p) : 1;
                if (const std::string e = fw.window(cseq.data(), p, nt); !e.empty()) {
                    std::fprintf(stderr, "strata_hy4_flash_metal: decode at %d: %s\n", p, e.c_str());
                    return 4;
                }
                if (p < nprompt && !pf_route.empty()) {
                    const std::vector<int32_t>& r = fw.routed_ids();
                    for (int i = 0; i < nt; ++i) {
                        bool any = false;
                        for (int o = 0; o < n_ord; ++o) {
                            std::vector<int32_t> a(&r[((size_t) o * nt + i) * topk], &r[((size_t) o * nt + i) * topk] + topk);
                            const int32_t* bp = &pf_route[((size_t) o * nprompt + p + i) * topk];
                            std::vector<int32_t> b(bp, bp + topk);
                            std::sort(a.begin(), a.end());
                            std::sort(b.begin(), b.end());
                            ++route_n;
                            if (a != b) { ++flips; any = true; }
                        }
                        if (any) ++flip_pos;
                    }
                }
                for (int i = 0; i < nt; ++i) {
                    const int q = p + i;
                    const float* b = fw.logits() + (size_t) i * (size_t) vocab;   // the decode path: the reference
                    const float* a = nullptr;                                       // the prefill path
                    if (q < nprompt) {
                        if (kslot[(size_t) q] < 0) continue;
                        a = &keep[(size_t) kslot[(size_t) q] * (size_t) vocab];
                    } else {
                        a = &ckeep[(size_t) (q - nprompt) * (size_t) vocab];
                    }
                    bitwise_count(q, a, b);
                    tol.push_back(strata::hy4_flash::tol_compare(q, q >= nprompt, b, a, vocab));
                    if (tol.back().am_ref != tol.back().am_test) ++argmax_diff;
                }
                p += nt;
                if (p % 512 == 0 || p == total) {
                    std::printf("    %d/%d (%.1f s)\n", p, total, mtl::now_s() - td);
                    std::fflush(stdout);
                }
            }
            const bool bitwise = diff_floats == 0;
            std::printf("  CONSISTENCY    %s: %llu of %zu compared positions differ, %llu floats, max abs %.3e, "
                        "max rel %.3e, argmax differs at %llu positions%s\n",
                        bitwise ? "PASS (bitwise)" : "DIFFERS", (unsigned long long) diff_pos, tol.size(),
                        (unsigned long long) diff_floats, max_abs, max_rel, (unsigned long long) argmax_diff,
                        first_bad < 0 ? "" : ("; first at position " + std::to_string(first_bad)).c_str());
            std::printf("  routing        top-%d sets that differ between the two runs: %llu of %llu (position, ordinal) "
                        "pairs = %.4f; %llu of %d positions have at least one%s\n", topk, (unsigned long long) flips,
                        (unsigned long long) route_n, route_n ? (double) flips / (double) route_n : 0.0,
                        (unsigned long long) flip_pos, nprompt, force_route ? " (forced: must be 0)" : "");
            const strata::hy4_flash::TolSummary ts = strata::hy4_flash::tol_summarize(tol, nprompt - 1);
            std::vector<strata::hy4_flash::TolPos> tc;
            for (const auto& x : tol)
                if (x.cont) tc.push_back(x);
            const strata::hy4_flash::TolSummary tsc = strata::hy4_flash::tol_summarize(tc, -1);
            std::printf("  tolerance      KL(decode || prefill) over %zu positions: mean %.3e, p99 %.3e, max %.3e (reported, "
                        "not a limit), KL > 0.1 at %.4f of them, > 1 at %.4f; "
                        "top-1 agreement %.4f (largest deficit %.3g logits); next token KL %.3e, argmax %s (decode top-1/2 gap %.3g, "
                        "deficit %.3g); continuation (%zu): KL mean %.3e, max %.3e, top-1 %.4f\n",
                        ts.n, ts.kl_mean, ts.kl_p99, ts.kl_max, ts.tail_01, ts.tail_1, ts.top1, ts.max_deficit, ts.next_kl,
                        ts.next_top1 ? "same" : "DIFFERS", ts.next_gap12, ts.next_deficit,
                        tsc.n, tsc.kl_mean, tsc.kl_max, tsc.top1);
            if (const char* co = flag_value(argc, argv, "--consistency-out")) {
                if (FILE* f = std::fopen(co, "w")) {
                    std::fprintf(f, "# pos cont kl argmax_decode argmax_prefill max_abs gap12 deficit\n");
                    for (const auto& x : tol)
                        std::fprintf(f, "%d %d %.9e %d %d %.6e %.6e %.6e\n", x.pos, x.cont, x.kl, x.am_ref, x.am_test,
                                     x.max_abs, x.gap12, x.deficit);
                    std::fclose(f);
                } else {
                    std::fprintf(stderr, "strata_hy4_flash_metal: cannot write %s\n", co);
                    return 2;
                }
            }
            int rc = bitwise ? 0 : 4;
            if (const char* tf = flag_value(argc, argv, "--consistency-tol")) {
                strata::hy4_flash::TolThresholds th;
                std::string err, why;
                if (!th.load(tf, err)) {
                    std::fprintf(stderr, "strata_hy4_flash_metal: %s\n", err.c_str());
                    return 2;
                }
                const bool ok = strata::hy4_flash::tol_pass(ts, th, why);
                std::printf("  TOLERANCE      %s against %s (kl_mean %.3g, kl_p99 %.3g, tail01_max %.4g, tail1_max %.4g, "
                            "top1_min %.4f, next_margin %.3g)%s%s\n",
                            ok ? "PASS" : "FAIL", tf, th.kl_mean, th.kl_p99, th.tail01_max, th.tail1_max, th.top1_min,
                            th.next_margin,
                            ok ? "" : ": ",
                            why.c_str());
                rc = ok ? 0 : 4;
            }
            m.stream.report(stdout);
            if (const std::string e = tdump.close(); !e.empty()) {
                std::fprintf(stderr, "strata_hy4_flash_metal: %s\n", e.c_str());
                return 2;
            }
            return rc;
        }
        if (gen <= 0) {
            m.stream.report(stdout);
            if (const std::string e = tdump.close(); !e.empty()) {
                std::fprintf(stderr, "strata_hy4_flash_metal: %s\n", e.c_str());
                return 2;
            }
            return 0;
        }
    }

    const double t_prompt0 = mtl::now_s();
    int done = 0;
    if (snap_loaded) done = nprompt;
    while (chunk <= 0 && !snap_loaded && done < (int) tokens.size()) {
        const int nt = std::min(T, (int) tokens.size() - done);
        if (const std::string e = fw.window(seq.data(), done, nt, &tdump); !e.empty()) {
            const int rc = nonfinite_or_fail(e, "position", done);
            if (lf != nullptr) std::fclose(lf);
            if (rc >= 0) { tdump.close(); return rc; }
            std::fprintf(stderr, "strata_hy4_flash_metal: position %d: %s\n", done, e.c_str());
            return 4;
        }
        write_logits(done, nt, fw.logits());
        if (want_mtp) mtp.set_hidden(done + 1, nt, fw.result_norm());
        if (pred_acc) {

            // the read-ahead's accuracy: how many of the predicted top-k the next ordinal really routed to
            const std::vector<int32_t>& real = fw.routed_ids();
            const std::vector<int32_t>& pr = fw.predicted_ids();
            for (int o = 1; o < m.stream.n_layers(); ++o)
                for (int t = 0; t < nt; ++t) {
                    const size_t at = ((size_t) o * nt + (size_t) t) * (size_t) topk;
                    if (pr[at] < 0) continue;
                    for (int k = 0; k < topk; ++k) {
                        bool found = false;
                        for (int j = 0; j < topk; ++j)
                            found = found || real[at + (size_t) j] == pr[at + (size_t) k];
                        pred_hit += found ? 1 : 0;
                    }
                    pred_tot += (uint64_t) topk;
                }
        }
        done += nt;
        if (done % 64 == 0 || done == (int) tokens.size()) {
            const double el = mtl::now_s() - t_prompt0;
            std::printf("    prompt %d/%zu (%.1f s, %.2f positions/s)\n", done, tokens.size(), el, done / el);
            std::fflush(stdout);
        }
    }
    const double prompt_s = mtl::now_s() - t_prompt0;
    if (lf != nullptr) {
        std::fclose(lf);
        lf = nullptr;
        std::printf("  logits         written to %s\n", lout);
    }
    if (chunk <= 0)
        std::printf("  prompt         %zu positions in %.2f s (%.3f positions/s, window T = %d; this is the DECODE\n"
                    "                 path, not prefill - give --prefill-chunk for that)\n", tokens.size(),
                    prompt_s, tokens.size() / prompt_s, T);
    if (pred_tot > 0)
        std::printf("  read-ahead     the predicted top-%d was in the next ordinal's choice %llu of %llu times "
                    "= %.4f (%.2f of %d)\n", topk, (unsigned long long) pred_hit, (unsigned long long) pred_tot,
                    (double) pred_hit / (double) pred_tot, (double) pred_hit / (double) pred_tot * topk, topk);

    // THE SAVE.  After the prompt (either path) and, with --mtp, after `catch_up` - the hidden rows are what
    // the snapshot carries, and they are only complete once the prompt's rows have been set.
    if (snap_out != nullptr) {
        if (snap_loaded) {
            std::fprintf(stderr, "strata_hy4_flash_metal: --save-prefill after --load-prefill would rewrite the snapshot "
                                 "from itself\n");
            return 2;
        }
        const double t0 = mtl::now_s();
        if (const std::string e = fw.snapshot_save(snap_out, sb, snap_extra); !e.empty()) {
            std::fprintf(stderr, "strata_hy4_flash_metal: %s\n", e.c_str());
            return 2;
        }
        struct stat ss {};
        const double gb = ::stat(snap_out, &ss) == 0 ? (double) ss.st_size / 1e9 : 0.0;
        std::printf("  prefill snap   saved %s in %.2f s (%.2f GB, %d positions, next token %d)\n", snap_out,
                    mtl::now_s() - t0, gb, (int) fw.snapshot_used(), (int) fw.last_token());
    }

    // --decode-logits-out: every GENERATED position's logits.  --logits-out covers the PROMPT only (it is
    // written from the prompt loops), so it cannot compare a run that skipped the prompt against one that ran
    // it - which is exactly the judge a prefill snapshot needs (prefill_snap.h).
    FILE* dlf = nullptr;
    int dl_rows = 0;
    if (const char* dl = flag_value(argc, argv, "--decode-logits-out")) {
        // Not on the MTP path: there a token can come from an accepted draft, and `last_logits` is then the
        // verify window's, not the row the token was chosen from.  Writing it anyway would make two runs
        // compare equal for the wrong reason.
        if (want_mtp) {
            std::fprintf(stderr, "strata_hy4_flash_metal: --decode-logits-out is for the plain greedy path; with --mtp a "
                                 "token may come from an accepted draft and last_logits() is not its source\n");
            return 2;
        }
        dlf = std::fopen(dl, "wb");
        if (dlf == nullptr) {
            std::fprintf(stderr, "strata_hy4_flash_metal: cannot write %s\n", dl);
            return 2;
        }
    }

    if (gen > 0 && want_mtp) {
        // the MTP's rows of the prompt, cache-only (mtp.h): row q needs x_q and h_{q-1}, so rows 0 .. n - 1 are
        // known now and row n waits for the first generated token
        const double tc = mtl::now_s();
        if (const std::string e = mtp.catch_up(seq.data(), nprompt); !e.empty()) {
            std::fprintf(stderr, "strata_hy4_flash_metal: MTP catch-up: %s\n", e.c_str());
            return 4;
        }
        std::printf("  mtp catch-up   %d prompt rows in %.2f s (cache-only)\n", nprompt, mtl::now_s() - tc);
    }
    if (gen > 0) {
        // THE GPU FILLER (mtl_rt.h set_seg_fill), OPT-IN: armed for decode only when STRATA_METAL_SEG_FILL is set
        // (cut | always).  Decode's host turns read as idleness to the GPU's clock governor, and filling them keeps
        // the clock up - but once the resident experts run during those turns the GPU already holds its top clock
        // almost all of the time, so the filler is not worth a permanent CPU core by default.  It still pays when
        // the host's turns are long SSD waits with no GPU work behind them (a small arena at k = 0).  Never during
        // the prompt.
        if (std::getenv("STRATA_METAL_SEG_FILL") != nullptr) {
            if (const std::string e = cx.set_seg_fill(1); !e.empty())
                std::printf("  gpu filler     NOT available (%s); decode runs without it\n", e.c_str());
            else if (cx.seg_fill_mode() != 0)
                std::printf("  gpu filler     on for decode (%s), one buffer %.0f us of GPU\n",
                            cx.seg_fill_mode() == 1 ? "cut" : "always", cx.seg_fill_buffer_us());
            else
                std::printf("  gpu filler     off (STRATA_METAL_SEG_FILL)\n");
        }
        std::printf("  greedy:");
        std::fflush(stdout);
        // Reset AFTER the prompt so the segment timing below is decode's alone: the prefill chunks run the same
        // `run_graph_seg` with a different T, and one accumulator cannot be divided by two position counts.
        // The position count has to be remembered with it - dividing the reset accumulator by `tm.positions`,
        // which counts the prompt too, understates every per-position figure below.
        const uint64_t seg_pos0 = fw.timing().positions;
        // THE SAME HAZARD ONE FIELD OVER: `host_cut_ms` and `gpu_ms` are also added to by prefill - every chunk's
        // 41 cuts land there - while the print divides them by the DECODE positions.  So the decode line subtracts
        // what the accumulators held before the first decode window, which is what makes "host at the cuts" the
        // number a cut-latency change moves.
        const strata::hy4_flash::Timing tm0 = fw.timing();
        // HY4_FLASH_SEG_PROFILE=<cuts per window> turns on the per-block in-situ profile.  It is a MEASUREMENT
        // switch: it changes which accumulator a microsecond lands in and nothing else - no dispatch, no
        // order, no value - so a run with it set stays bit-identical, and it is not in `rejected_env`'s list
        // for that reason.
        if (const char* sp = std::getenv("HY4_FLASH_SEG_PROFILE")) cx.set_cut_period(std::atoi(sp));
        cx.reset_seg_timing();
        const double t0 = mtl::now_s();
        double first = 0;
        int made = 0;
        if (want_mtp) {
            const std::string e = strata::hy4_flash::speculative_greedy(
                fw, mtp, seq, gen, draft_k, context, mtp_probe, [&](int32_t tok) {
                    ++made;
                    if (made == 1) first = mtl::now_s() - t0;
                    std::printf(" %d", (int) tok);
                    std::fflush(stdout);
                });
            if (!e.empty()) {
                std::printf("\n");
                if (const int rc = nonfinite_or_fail(e, "generation", made); rc >= 0) { tdump.close(); return rc; }
                std::fprintf(stderr, "strata_hy4_flash_metal: speculative generation %d: %s\n", made, e.c_str());
                return 4;
            }
        }
        for (int i = 0; !want_mtp && i < gen; ++i) {
            // the last position the head ran for continues the sequence, whether that was a prefill chunk or a
            // decode window (Forward::last_token)
            const int32_t next = fw.last_token();
            // That value comes back FROM THE GPU and must not go into the sequence unchecked, because the next
            // window hands it to token_embd's row lookup.  It can only be out of range when the logits were
            // not finite (the scan's comparisons are then all false and the index keeps its initial value), so
            // the request ends the way the cache guard ends it rather than reading outside the weights.
            if (next < 0 || next >= fw.vocab()) {
                std::printf("\n  finish         nonfinite (generation %d): the head's argmax returned %d, outside "
                            "0..%d - the logits were not finite (ADR 0001 §4.2)\n", i, (int) next, fw.vocab() - 1);
                tdump.close();
                return 0;
            }
            seq.push_back(next);
            ++made;
            // the logits the token was chosen from, before the next window overwrites them
            if (dlf != nullptr) {
                // Checked: an unchecked fwrite that returned 0 on a full disk would leave a SHORT artifact
                // that a later successful fclose reported as fine, and the equality judge would then compare
                // two truncated files.
                if (std::fwrite(fw.last_logits(), 4, (size_t) fw.vocab(), dlf) != (size_t) fw.vocab()) {
                    std::fprintf(stderr, "\nstrata_hy4_flash_metal: writing --decode-logits-out failed at position %d\n",
                                 dl_rows);
                    std::fclose(dlf);
                    return 2;
                }
                ++dl_rows;
            }
            std::printf(" %d", (int) next);
            std::fflush(stdout);
            if ((int32_t) seq.size() >= context) break;
            const double ta = mtl::now_s();
            if (const std::string e = fw.window(seq.data(), (int) seq.size() - 1, 1); !e.empty()) {
                std::printf("\n");
                if (const int rc = nonfinite_or_fail(e, "generation", i); rc >= 0) { tdump.close(); return rc; }
                std::fprintf(stderr, "strata_hy4_flash_metal: generation %d: %s\n", i, e.c_str());
                return 4;
            }
            if (i == 0) first = mtl::now_s() - ta;
        }
        const double s = mtl::now_s() - t0;
        cx.set_seg_fill(0);   // disarmed before the reports: decode is over, the filler's core is free again
        std::printf("\n  decode         %d tokens in %.2f s = %.3f tok/s (the first took %.0f ms); context %zu\n",
                    made, s, made / s, first * 1e3, seq.size());
        if (want_mtp) {
            const strata::hy4_flash::MtpStats& ms = mtp.stats();
            const double r = (double) std::max<uint64_t>(1, ms.rounds);
            std::printf("  mtp            k = %d: %llu rounds, %llu drafts offered, %llu accepted = acceptance %.4f, "
                        "%.3f tokens a round%s\n", draft_k, (unsigned long long) ms.rounds,
                        (unsigned long long) ms.offered, (unsigned long long) ms.accepted,
                        ms.offered ? (double) ms.accepted / (double) ms.offered : 0.0, (double) made / r,
                        mtp_probe == strata::hy4_flash::MtpProbe::none ? "" : " (NEGATIVE CONTROL, HY4_FLASH_MTP_PROBE)");
            std::printf("  mtp accepted   per round:");
            for (size_t i = 0; i < ms.accept_hist.size(); ++i)
                std::printf(" %zu:%llu", i, (unsigned long long) ms.accept_hist[i]);
            std::printf("\n  mtp time       per round: draft %.2f ms (of it expert reads %.2f ms), verify %.2f ms, "
                        "commit %.3f ms; catch-up rows %llu, full rows %llu\n", ms.draft_ms / r,
                        ms.expert_read_ms / r, ms.verify_ms / r, ms.commit_ms / r,
                        (unsigned long long) ms.cache_rows, (unsigned long long) ms.full_rows);
            std::printf("  mtp experts    %llu hits, %llu misses = hit rate %.4f, %.2f GB read (%.1f MB a draft "
                        "step)\n", (unsigned long long) ms.expert_hits, (unsigned long long) ms.expert_misses,
                        ms.expert_hits + ms.expert_misses
                            ? (double) ms.expert_hits / (double) (ms.expert_hits + ms.expert_misses) : 0.0,
                        ms.expert_bytes / 1e9, ms.expert_bytes / 1e6 / (double) std::max<uint64_t>(1, ms.full_rows));
            std::printf("  mtp experts    by half of the tokens: first %llu hits / %llu misses, second %llu hits / %llu "
                        "misses\n", (unsigned long long) ms.half_hits[0], (unsigned long long) ms.half_misses[0],
                        (unsigned long long) ms.half_hits[1], (unsigned long long) ms.half_misses[1]);
        }
        strata::hy4_flash::Timing tm = fw.timing();
        tm.gpu_ms -= tm0.gpu_ms;
        tm.wall_ms -= tm0.wall_ms;
        tm.host_cut_ms -= tm0.host_cut_ms;
        tm.engram_wait_ms -= tm0.engram_wait_ms;
        tm.eng_tail_ms -= tm0.eng_tail_ms;
        tm.eng_hash_ms -= tm0.eng_hash_ms;
        tm.eng_join_ms -= tm0.eng_join_ms;
        tm.positions -= tm0.positions;
        const double n = (double) (tm.positions > 0 ? tm.positions : 1);
        std::printf("  per position   gpu %.2f ms, wall %.2f ms, host at the cuts %.2f ms, Engram wait %.2f ms, "
                    "expert wait %.2f ms (decode only)\n", tm.gpu_ms / n, tm.wall_ms / n, tm.host_cut_ms / n,
                    tm.engram_wait_ms / n, tm.expert_wait_ms / n);
        // Engram's three thirds.  `tail` is this window being charged for the PREVIOUS window's unfinished
        // readers; `hash` is pure CPU over the window's tokens and therefore MUST NOT grow with context - if it
        // does, one of our assumptions is wrong and that is the finding.
        if (tm.engram_wait_ms > 0)
            std::printf("  of Engram      previous window's tail %.2f ms, hash+threads %.2f ms, this window's "
                        "read wait %.2f ms per position (the rest is prepare_window)\n",
                        tm.eng_tail_ms / n, tm.eng_hash_ms / n, tm.eng_join_ms / n);
        // `gpu_ms` above is GPUEndTime-GPUStartTime of the command buffers, and that span CONTAINS the host's
        // turns at the cuts - so it cannot say whether a longer context costs more GPU COMPUTE or more WAITING.
        // `spin` is the half that is compute (we busy-wait while the GPU works towards the cut); `cut` is the
        // half that is us.  The three must be read against each other, not alone: if the gap between two
        // contexts sits in `spin` it is the kernels, if it sits in `cut` it is the SSD and the host.
        const mtl::Context::SegTiming& st = cx.seg_timing();
        const double sn = (double) (tm.positions > seg_pos0 ? tm.positions - seg_pos0 : 1);
        std::printf("  segments       spin %.2f ms, at the cuts %.2f ms, encode %.2f ms, commit %.2f ms, "
                    "final wait %.2f ms (gpu span %.2f ms) per position\n", st.spin_ms / sn, st.cut_ms / sn,
                    st.encode_ms / sn, st.commit_ms / sn, st.wait_ms / sn, st.gpu_ms / sn);
        // The counts are the check on the divisor, not decoration: a decode window is 41 cuts, so anything but
        // 41.0 here means the accumulator and the position count do not cover the same windows.
        std::printf("  segment counts %llu command buffers, %llu encoders, %llu cuts over %.0f positions = "
                    "%.1f / %.1f / %.1f each\n", (unsigned long long) st.n_cb, (unsigned long long) st.n_enc,
                    (unsigned long long) st.n_cut, sn, st.n_cb / sn, st.n_enc / sn, st.n_cut / sn);
        if (const mtl::Context::FillStats fs = cx.fill_stats(); fs.n > 0)
            std::printf("  gpu filler     %llu buffers, %.2f ms of filler GPU time per position\n",
                        (unsigned long long) fs.n, fs.gpu_ms / sn);
        // The dispatch count is the check on the hand census off forward.cpp: a kernel sum can be low because
        // the kernels are slower in the pipeline than in isolation, or because whole kernels were never
        // counted, and only this number tells the two apart.
        std::printf("  dispatches     %.1f per position in the segmented path, %.1f outside it (run_graph)\n",
                    st.n_dis / sn, st.n_dis_plain / sn);
        if (const int per = cx.cut_period(); per > 0) {
            // The assumption first: the cuts are indexed modulo `per`, so unless a window really has `per` of
            // them the numbers below are several blocks added together.  Say so instead of printing them.
            const double got = st.n_cut / sn;
            if (std::fabs(got - (double) per) > 1e-9) {
                std::printf("  seg profile    NOT USABLE: %.3f cuts per position, not the %d assumed - the "
                            "per-block index is not aligned\n", got, per);
            } else {
                std::printf("  seg profile    per block, in situ: ordinal spin_ms cut_ms (cut k follows the "
                            "k-th router, so k is the ordinal)\n");
                double tot = 0;
                for (int k = 0; k < per; ++k) {
                    const double w = (double) (st.n_at[k] > 0 ? st.n_at[k] : 1);
                    // Which block cut k belongs to is NOT assumed: the ordinal is looked up in the geometry,
                    // the same table forward.cpp encodes from.  Without the kind the numbers cannot be read -
                    // a sliding block has 64 heads and no indexer, a sparse one 32 heads and the indexer, so
                    // "every block is slow" and "the sparse blocks are slow" need different work.
                    int bid = -1;
                    const char* kind = "?";
                    const char* eng = "";
                    for (int b = 0; b < m.geo.trunk_blocks; ++b) {
                        const strata::hy4_flash::LayerSpec& ls = m.geo.block(b);
                        if (ls.expert_ordinal != k) continue;
                        bid = b;
                        kind = ls.attention == strata::hy4_flash::Attention::Sliding ? "sliding" : "sparse ";
                        if (ls.engram_table >= 0) eng = " engram";
                        break;
                    }
                    std::printf("  seg %2d         block %2d %s spin %6.3f  cut %6.3f  (%llu windows)%s\n", k,
                                bid, kind, st.spin_at[k] / w, st.cut_at[k] / w,
                                (unsigned long long) st.n_at[k], eng);
                    tot += st.spin_at[k] / w;
                }
                // The parts have to add up to the total that was measured independently, or one of the two
                // is wrong: `spin_ms / positions` counts the same microseconds a different way.
                std::printf("  seg profile    the %d blocks' spin adds to %.2f ms against %.2f ms measured "
                            "whole (%.2f%%)\n", per, tot, st.spin_ms / sn,
                            st.spin_ms > 0 ? 100.0 * (tot - st.spin_ms / sn) / (st.spin_ms / sn) : 0.0);
            }
        }
    }
    if (dlf != nullptr) {
        // fclose unconditionally: `fflush(..) == 0 && fclose(..) == 0` skips the close when the flush fails,
        // which leaks the stream on exactly the path that needs it closed.
        const bool flushed = std::fflush(dlf) == 0 && std::ferror(dlf) == 0;
        const bool closed = std::fclose(dlf) == 0;
        const bool ok = flushed && closed;
        dlf = nullptr;
        if (!ok) {
            std::fprintf(stderr, "strata_hy4_flash_metal: writing --decode-logits-out failed\n");
            return 2;
        }
        std::printf("  decode logits  %d positions x %d floats written\n", dl_rows, (int) fw.vocab());
    }
    if (const std::string e = tdump.close(); !e.empty()) {
        std::fprintf(stderr, "strata_hy4_flash_metal: %s\n", e.c_str());
        return 2;
    }
    if (const uint32_t bad = fw.cache_nonfinite(); bad != 0)
        std::printf("  nonfinite      %u in all (every window was checked as it finished)\n", bad);
    m.stream.report(stdout);
    return rc;
}
