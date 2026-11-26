// The RTX 50 ("sm_120" family) switch set and the per-card table, in one place and without any CUDA dependency (so a unit test can run it).
//
// What it decides: whether the sm_120 fast mode runs on a given card, and with which numbers (tile sizes per batch range, chunk size, how the experts
// and the FP8 trunk are placed, memory budgets, drafting numbers). The CUDA side (cuda/ink_sm120_cfg.cuh) only reads the answer.
//
// Switches (environment, read once per device):
//   STRATA_INK_SM120=1          fast mode on (changes answers slightly: FP8 trunk, MX8 -> SF4 transcode, FP16 sums). 0 = the old path, always.
//   STRATA_INK_SM120_EXACT=1    "exact" form: only the parts that do not change a single bit of the answer (to be proven by the gates); with
//                               STRATA_INK_SM120 unset it is the same as =1 with the lossless set.
//   unset                       the default: kSm120FastDefault below (true since 11-26): a card with a MEASURED row (RTX 5090, RTX 5060 Ti 16 GB) runs its row's fast set, every other card the old path.
//   STRATA_INK_SM120_<PART>=0|1 one part on or off (PF_ATTN FP8_TRUNK MOE_OPT DENSE_OPT DSPARK_OPT C64_SPEC COMPACT TC_SMALLM PLD PACK PREFILL
//                               CORE_GRAPH L2PF TC_ATTN SPEC_ADAPT); an explicit 1 turns the part on on any sm_120 card, an explicit 0 turns it off.
//   STRATA_INK_SM120_ROW=<id>   use the table row <id> although the card has no measured row of its own (own risk, a note is printed).
// Old names, still accepted as aliases: STRATA_INK_RTX5090 (= STRATA_INK_SM120), STRATA_INK_RTX5090_EXACT, STRATA_INK_RTX5060TI,
// STRATA_INK_RTX5060TI_EXACT, STRATA_INK_5060TI_<PART> / STRATA_INK_5060TI_<NUMBER>. The new name wins when both are set.
//
// A switch on a card that is not sm_120 (cc 12.0) is refused cleanly: one line on stderr, the default path runs. A cc 12.0 card without a measured
// row gets the default path (and the same one line) unless STRATA_INK_SM120_ROW names a row.
//
// The matching is by device name, number of SMs and VRAM, never by the chip family alone: an RTX 5070 or an RTX PRO 6000 Blackwell is also cc 12.0.
#pragma once

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

namespace strata::core {

// ------------------------------------------------------------------------------------------------------------------------------------------
// THE default. Flip this one constant when the owner decides that fast mode is the default on the measured RTX 50 cards (after FASTQ passed).
// Only a card with a measured row ever gets it; every other card keeps the old path. STRATA_INK_SM120=0 always gives the old path.
// ------------------------------------------------------------------------------------------------------------------------------------------
constexpr bool kSm120FastDefault = true;   // owner 11-26 after FASTQ: fast mode is the default on the two measured rows; STRATA_INK_SM120=0 is the old path

enum Sm120Part : uint32_t {
    P_PF_ATTN    = 1u << 0,   // pipelined FP16-accumulate tensor-core FlashAttention prefill (5090 kernels)
    P_FP8_TRUNK  = 1u << 1,   // FP8 trunk/controller GEMMs (cache copy on big cards, in place on small ones: see fp8_mode)
    P_MOE_OPT    = 1u << 2,   // 5090 MoE path: dynamic tiles, L1-direct M=1, parallel binning, MX8 -> SF4 transcode of the 4 single-pass layers
    P_DENSE_OPT  = 1u << 3,   // 5090 dense path: fused depth/bank mix, BF16 residual states in prefill, fused sink + sconv
    P_DSPARK_OPT = 1u << 4,   // deep verify tree (anc32), Root-PLD-First, leaf graft, 2-block chain, exact Q8ROW argmax screen, 32K DSpark KV
    P_C64_SPEC   = 1u << 5,   // multi-user (C2..C64) drafting behind the one hook; the implementation is c64_mode
    P_COMPACT    = 1u << 6,   // all experts in VRAM on a <= 20 GiB card: GPU MX8 -> SF4 transcode, single slab, FP8 KV, mapped embed
    P_TC_SMALLM  = 1u << 7,   // sm_120 NVFP4 tensor cores from M >= 4, INT8 tensor-core Q8ROW linear for M > 16, direct-MMA kernels
    P_PLD        = 1u << 8,   // prompt-lookup drafting graft on the C1 tree (5060 Ti builder)
    P_PACK       = 1u << 9,   // static full-layer VRAM packing when not everything fits
    P_PREFILL    = 1u << 10,  // layer-major single-sweep prefill / macro prefill
    P_CORE_GRAPH = 1u << 11,  // C1 prefix+core CUDA graph and calibrated verify tree sizing
    P_L2PF       = 1u << 12,  // L2 prefetch hints in the NVFP4 MoE kernels
    // The two lossy halves of COMPACT, switchable apart (FASTQ on the 5060 Ti failed with COMPACT on, passed with it off). They act only while
    // P_COMPACT is on; naming COMPACT=1 without naming them turns both on (the hand-over behaviour). COMPACT also always turns the attention weights
    // and the LM head into Q8_ROW / FP8 and the embedding into mapped host memory: that part cannot be split off (it is what makes room).
    P_COMPACT_XCODE = 1u << 13,  // compact: MX8 experts of layers 20-23 are transcoded to SF4 on load (without it SQ-20 does not fit and compact falls back)
    P_COMPACT_KV    = 1u << 14,  // compact: FP8 (E4M3) KV cache on a one-slot run
    P_FP8_INPLACE   = 1u << 15,  // the 5060 Ti hand-over's in-place FP8 trunk (known bad: FASTQ failed); in no fast set, explicit request only
    P_SPEC_ADAPT    = 1u << 17,  // the draft window follows the measured step cost per row and the measured acceptance (fast only: it changes with timing; the exact form keeps the static model)
    P_TC_ATTN       = 1u << 16,  // tensor-core split-K decode attention of the global visits (FP16 tensor cores, FP32 accumulate; changes rounding: fast only, never in an exact set)
    // COMPACT on a card that cannot hold all the experts (8 and 12 GB): the compact trunk (Q8_ROW attention and head, the embedding in mapped host memory: about 2.3 GiB less than
    // the BF16 trunk) is still built and the VRAM it frees goes to the adaptive expert tier; the experts that do not fit stay in RAM (CPU lane). Without this part COMPACT refuses
    // on such a card and the old path runs. Acts only while P_COMPACT is on.
    P_COMPACT_HYBRID = 1u << 18,
};
constexpr uint32_t kSm120AllParts = (1u << 19) - 1u;
constexpr uint32_t kSm120CompactSubParts = P_COMPACT_XCODE | P_COMPACT_KV;

// ONE FP8 trunk for every card: Cache = the 5090 implementation (a +641 MiB FP8 copy next to the BF16 / Q8_ROW trunk; it simply does not start when the card
// has no room for the copy, and the trunk keeps its BF16 / Q8_ROW GEMMs). InPlace = the 5060 Ti hand-over's in-place conversion: FASTQ 2026-11-26 failed with it
// (tool calls 6/26 against 23/26 off; with FP8_TRUNK=0 24/26); it is reachable only by the explicit part FP8_INPLACE and is in no fast set.
enum class Sm120Fp8 : uint8_t { None, Cache, InPlace };
enum class Sm120C64  : uint8_t { None, Bspec, K8v4 };     // Bspec: mode-3 batched spec with staging (5090); K8v4: K8V4 multi-slot DSpark chain verify (5060 Ti)
// MoE kernel variant for one batch range. For tile_rows <= 64: Base (shipped k_moe_*_tc), Bk128 (5090: BK=128 SF4 / BK=64 MX8 pipelined kernels),
// Direct (5060 Ti: direct-MMA SF4, per-row global scale). For tile_rows == 128: Tr128 (5090 512-thread kernels), Tr128Small (5060 Ti 256-thread),
// Tr128Bk128 (5060 Ti BK=128 swizzled SF4). A variant that does not exist for a tile size or a format falls back to the nearest one in the wrapper.
enum class Sm120Tc   : uint8_t { Base, Bk128, Direct, Tr128, Tr128Small, Tr128Bk128 };

struct Sm120MoeRange {
    int m_max;            // this range holds M <= m_max (the last range holds everything above)
    int tile_rows;        // 16 / 32 / 64 / 128
    Sm120Tc sf4;          // kernel variant for the SF4 experts (layers 0..19 of SQ-20)
    Sm120Tc mx8;          // kernel variant for the MX8 experts (layers 20..23)
    int pf_bytes;         // L2 prefetch window per weight row (bytes, 0 = none)
};

struct Sm120Row {
    const char* id;          // "5090"
    const char* name_has;    // substring of the CUDA device name
    int sm_min, sm_max;      // number of SMs (inclusive window)
    int vram_gib_min, vram_gib_max;   // total VRAM in GiB (rounded), inclusive window
    bool measured;           // true only when the numbers below were measured on this card
    const char* note;

    uint32_t fast_parts;     // parts on in fast mode
    uint32_t exact_parts;    // parts that stay on in the exact form
    Sm120Fp8 fp8_mode;
    Sm120C64 c64_mode;

    // tiles and kernel variants per batch range (the first range whose m_max >= M applies)
    Sm120MoeRange moe[5];
    int n_moe;

    // chunk size and memory budgets
    int c1_chunk_big;                // C1 prefill chunk when enough VRAM is free (0 = the engine default); 4096 on a 5090
    int c1_chunk_big_headroom_mib;   // free VRAM needed after the experts and the K/V still to grow, for c1_chunk_big
    int c1_chunk;                    // C1 chunk of the compact / prefill_lm path (256 on a 16 GB card), 0 = default
    int compact_max_vram_gib;        // "compact" (all experts in VRAM) is tried only on cards with at most this much VRAM
    int fp8_cache_headroom_mib;      // free VRAM the FP8 cache needs after its own pool (5090: 512)
    int ws_extra_fp8_mib;            // extra workspace reserve for the K/V sizing when the FP8 cache is on (5090: 768)
    int ws_extra_dspark_mib;         // extra workspace reserve when the deep-tree DSpark path is on (5090: 1536)
    int experts_in_vram_policy;      // 0 = as the loader decides, 1 = all 3,072 SQ-20 experts in VRAM when the budget allows (16 GB cards)

    // drafting numbers
    int   spec_beam, c1_max_nodes, c1_pld_max_m, c64_draft_k;
    float spec_cost, spec_plain;
    float pcie_share;

    // The row's fast set is the DEFAULT (no switch set) only when FASTQ passed on this card kind. A measured row can have default_on = false: the speed is measured
    // and fast mode is one switch away, but the old path stays the default because quality or speed did not pass (rows lane 11-26).
    bool default_on = false;

    // The compute capability minor of the card kind the row is for: 0 for the RTX 50 / RTX PRO Blackwell cards (cc 12.0), 1 for the DGX Spark GB10 (cc 12.1). A row never
    // matches a card of the other minor, so the 5090 row can not be taken by a GB10 and the GB10 row can not be taken by an RTX card.
    int cc_minor = 0;
};

// Default knobs shared by rows (the 5090 and the 5060 Ti rows are the measured ones; the others copy the family they belong to and stay unmeasured).
// 5090 row: tile 16 up to M=192, 32 up to 512, 64 up to 1023, 128 from 1024; BK=128 / BK=64 pipelined kernels, TR=128 512-thread kernels (hand-over 11-26).
// 5060 Ti row: tile 16 up to M=128, 32 up to 160, 64 up to 1536, 128 above; direct-MMA SF4 kernels, base MX8, BK=128 swizzled TR=128 SF4 (hand-over 11-26).
// Which variant wins per range on each card is decided by the A/B runs of the INT-SM120 lane (RESULTS.md); until a row is re-measured it keeps
// what its own hand-over used.
#define STRATA_SM120_BIG_MOE \
    { {192, 16, Sm120Tc::Bk128, Sm120Tc::Bk128, 256}, {512, 32, Sm120Tc::Bk128, Sm120Tc::Bk128, 0}, \
      {1023, 64, Sm120Tc::Bk128, Sm120Tc::Bk128, 0}, {1 << 30, 128, Sm120Tc::Tr128, Sm120Tc::Tr128, 0} }, 4
#define STRATA_SM120_SMALL_MOE \
    { {128, 16, Sm120Tc::Direct, Sm120Tc::Base, 128}, {160, 32, Sm120Tc::Direct, Sm120Tc::Base, 128}, \
      {1536, 64, Sm120Tc::Direct, Sm120Tc::Base, 128}, {1 << 30, 128, Sm120Tc::Tr128Bk128, Sm120Tc::Tr128, 128} }, 4

constexpr uint32_t kSm120BigFast = P_PF_ATTN | P_FP8_TRUNK | P_MOE_OPT | P_DENSE_OPT | P_DSPARK_OPT | P_C64_SPEC;   // P_TC_ATTN joins this set only after the long-context A/B and the KL check (PERF-LONGCTX)
// The exact form must not change the DRAFTING POLICY against off (tree shape, window, beam, node count, spec_cost, slot cap): with DFlash 2 as the only drafter DSPARK_OPT (deep verify tree, 2-block chain, row spec numbers) does, and the Spark and the R9700 saw exact differ from off for that reason, so the 5090 exact set is EMPTY: only the bit-identical kernels that have their own switches (PDL, PFQ, the sm_120a build) differ from off.
constexpr uint32_t kSm120BigExact = 0;
// The 5060 Ti fast set = COMPACT (MX8 -> SF4 transcode + FP8 KV + single slab, attention weights Q8_ROW) + the rest, WITHOUT any FP8 trunk. FASTQ 11-26: with the in-place FP8
// trunk on, fast mode lost 17 of 26 tool calls; with FP8_TRUNK=0 (attention in Q8_ROW) it is at or above off on every suite (report-5060ti-fp8t0.md). The in-place FP8 trunk is
// explicit-only (STRATA_INK_SM120_FP8_INPLACE=1); FP8_TRUNK (the 5090 cache engine) cannot start next to compact (no room), so it is not in this set either.
constexpr uint32_t kSm120SmallFast = P_COMPACT | P_TC_SMALLM | P_PLD | P_PACK | P_PREFILL | P_CORE_GRAPH | P_C64_SPEC | P_L2PF;
// Exact set of the small cards, trimmed by measurement (5060 Ti, new SQ-20 hybrid load, 2026-11-26): PACK (moves experts between the CPU and the GPU lane: rounding differs)
// and PREFILL (layer-major prefill: tokens differ from gen:9000 on) are out. Then PLD (tokens differ at gen:9000, 2 of 2 runs) and CORE_GRAPH (differ at gen:30 from token 67 when
// combined with the others) are out too. C64_SPEC + L2PF: tokens identical to off at 30, 9000 and 12000 (2 rounds each), speed equal to off. The speed-ups of the others are fast-only.
// C64_SPEC is out of the exact set too: its multi-user memory plan (FP8 KV, host-resident SWA) assumes COMPACT; with exact (no compact) a 64-slot run on a 16 GB card died ("not enough free GPU memory for the working buffers", global KV all in host RAM). L2PF alone is bit-identical.
constexpr uint32_t kSm120SmallExact = P_L2PF;

// The table. Order matters: the first row that matches wins, so the more specific names come first (" Ti" before the plain name).
// PLACEHOLDER rows (measured = false): 5070, 5050 (their numbers come from a rental later), and every other cc 12.0 card. They are listed so
// that the match reports the card by name, and so that a row can be filled in without touching the code.
inline const Sm120Row* ink_sm120_rows(int* n) {
    static const Sm120Row rows[] = {
        // id, name, sm window, vram window GiB, measured, note,
        //   fast parts, exact parts, fp8, c64, moe ranges, n_moe,
        //   c1_chunk_big, headroom, c1_chunk, compact_max, fp8_cache_headroom, ws_fp8, ws_dspark, experts policy,
        //   beam, max_nodes, pld_max_m, c64_k, cost, plain, pcie_share
        {"5090", "RTX 5090", 160, 192, 30, 34, true, "measured on a 32 GB RTX 5090 (hand-over 11-26)",
         kSm120BigFast, kSm120BigExact, Sm120Fp8::Cache, Sm120C64::Bspec, STRATA_SM120_BIG_MOE,
         4096, 4800, 0, 0, 512, 768, 1536, 0,
         4, 12, 24, 3, 8.0f, 1.02f, 0.28f, true},
        {"5060ti-16", "RTX 5060 Ti", 34, 38, 15, 17, true, "measured on a 16 GB RTX 5060 Ti (hand-over 11-26)",
         kSm120SmallFast, kSm120SmallExact, Sm120Fp8::Cache, Sm120C64::K8v4, STRATA_SM120_SMALL_MOE,
         0, 0, 256, 20, 512, 768, 0, 1,   // the 5090's FP8 cache: 512 MiB headroom, 768 MiB reserved for it in the K/V sizing (PLACEHOLDER: measure on a 5060 Ti)
         1, 12, 24, 2, 2.0f, 1.02f, 0.28f, true},
        // ---- placeholders: not measured, the old path runs unless STRATA_INK_SM120_ROW names a row ----
        {"5080", "RTX 5080", 80, 88, 15, 17, false, "placeholder: 16 GB, 84 SMs; rental later",
         kSm120SmallFast, kSm120SmallExact, Sm120Fp8::Cache, Sm120C64::K8v4, STRATA_SM120_SMALL_MOE,
         0, 0, 256, 20, 0, 0, 0, 1,
         1, 12, 24, 2, 2.0f, 1.02f, 0.28f},
        {"5070ti", "RTX 5070 Ti", 66, 74, 15, 17, false, "placeholder: 16 GB, 70 SMs; rental later",
         kSm120SmallFast, kSm120SmallExact, Sm120Fp8::Cache, Sm120C64::K8v4, STRATA_SM120_SMALL_MOE,
         0, 0, 256, 20, 0, 0, 0, 1,
         1, 12, 24, 2, 2.0f, 1.02f, 0.28f},
        {"5070", "RTX 5070", 44, 52, 11, 13, false, "PLACEHOLDER: 12 GB, 48 SMs; the owner's card, numbers from a rental later (compact does not fit: pack path)",
         kSm120SmallFast, kSm120SmallExact, Sm120Fp8::Cache, Sm120C64::K8v4, STRATA_SM120_SMALL_MOE,
         0, 0, 256, 20, 0, 0, 0, 0,
         1, 12, 24, 2, 2.0f, 1.02f, 0.28f},
        {"5060ti-8", "RTX 5060 Ti", 34, 38, 7, 9, false, "placeholder: 8 GB variant of the 5060 Ti",
         kSm120SmallFast, kSm120SmallExact, Sm120Fp8::Cache, Sm120C64::K8v4, STRATA_SM120_SMALL_MOE,
         0, 0, 256, 20, 0, 0, 0, 0,
         1, 12, 24, 2, 2.0f, 1.02f, 0.28f},
        {"5060", "RTX 5060", 28, 32, 7, 9, false, "placeholder: 8 GB, 30 SMs; rental later",
         kSm120SmallFast, kSm120SmallExact, Sm120Fp8::Cache, Sm120C64::K8v4, STRATA_SM120_SMALL_MOE,
         0, 0, 256, 20, 0, 0, 0, 0,
         1, 12, 24, 2, 2.0f, 1.02f, 0.28f},
        {"5050", "RTX 5050", 18, 22, 7, 9, false, "PLACEHOLDER: 8 GB, 20 SMs; numbers from a rental later",
         kSm120SmallFast, kSm120SmallExact, Sm120Fp8::Cache, Sm120C64::K8v4, STRATA_SM120_SMALL_MOE,
         0, 0, 256, 20, 0, 0, 0, 0,
         1, 12, 24, 2, 2.0f, 1.02f, 0.28f},
        {"pro6000", "RTX PRO 6000", 180, 200, 90, 100, false, "placeholder: RTX PRO 6000 Blackwell (96 GB, 188 SMs)",
         kSm120BigFast, kSm120BigExact, Sm120Fp8::Cache, Sm120C64::Bspec, STRATA_SM120_BIG_MOE,
         4096, 4800, 0, 0, 512, 768, 1536, 0,
         4, 12, 24, 3, 12.0f, 1.02f, 0.28f},
        {"pro5000", "RTX PRO 5000", 100, 120, 40, 80, false, "placeholder: RTX PRO 5000 Blackwell (48 or 72 GB, 110 SMs)",
         kSm120BigFast, kSm120BigExact, Sm120Fp8::Cache, Sm120C64::Bspec, STRATA_SM120_BIG_MOE,
         4096, 4800, 0, 0, 512, 768, 1536, 0,
         4, 12, 24, 3, 12.0f, 1.02f, 0.28f},
        {"pro4500", "RTX PRO 4500", 76, 90, 30, 34, false, "placeholder: RTX PRO 4500 Blackwell (32 GB, 82 SMs)",
         kSm120BigFast, kSm120BigExact, Sm120Fp8::Cache, Sm120C64::Bspec, STRATA_SM120_BIG_MOE,
         4096, 4800, 0, 0, 512, 768, 1536, 0,
         4, 12, 24, 3, 12.0f, 1.02f, 0.28f},
        {"pro4000", "RTX PRO 4000", 66, 74, 22, 26, false, "placeholder: RTX PRO 4000 Blackwell (24 GB, 70 SMs)",
         kSm120SmallFast, kSm120SmallExact, Sm120Fp8::Cache, Sm120C64::K8v4, STRATA_SM120_SMALL_MOE,
         0, 0, 256, 20, 0, 0, 0, 0,
         1, 12, 24, 2, 2.0f, 1.02f, 0.28f},
        {"pro2000", "RTX PRO 2000", 24, 32, 15, 17, false, "placeholder: RTX PRO 2000 Blackwell (16 GB)",
         kSm120SmallFast, kSm120SmallExact, Sm120Fp8::Cache, Sm120C64::K8v4, STRATA_SM120_SMALL_MOE,
         0, 0, 256, 20, 0, 0, 0, 1,
         1, 12, 24, 2, 2.0f, 1.02f, 0.28f},
        // ---- DGX Spark (NVIDIA GB10, cc 12.1, 48 SMs, 121 GB of memory shared with the CPU, about 273 GB/s) ----
        // Matched by name + SM count + memory AND cc 12.1, so it never takes the 5090 row and the 5090 never takes it. The whole model sits in the unified memory (no hybrid split,
        // no compact, no PACK), so the set is the 5090's "big" set: FP8 trunk with the cache copy (641 MiB, room for it), the NVFP4 tensor-core MoE, PF_ATTN, drafting.
        // NOT measured and NOT default yet (measured = false: the old path runs unless STRATA_INK_SM120_ROW=gb10 or a part switch asks); the rows of the lane's tests
        // (RESULTS.md of the DGX-SPARK lane) decide the tiles, the draft numbers and default_on. The numbers below are the 5090's, kept until measured:
        // spec_cost 8 / beam 4 / 12 nodes are the 5090's after PDL, c64_draft_k 3; c1_chunk_big 4096 needs 4800 MiB of headroom, which the unified memory has.
        {"gb10", "GB10", 44, 52, 100, 128, false, "placeholder: DGX Spark GB10 (cc 12.1, 48 SMs, 121 GB unified memory); numbers are the 5090's until measured",
         kSm120BigFast, kSm120BigExact, Sm120Fp8::Cache, Sm120C64::Bspec, STRATA_SM120_BIG_MOE,
         4096, 4800, 0, 0, 512, 768, 1536, 0,
         4, 12, 24, 3, 8.0f, 1.02f, 0.0f, false, 1},
    };
    if (n) *n = (int)(sizeof(rows) / sizeof(rows[0]));
    return rows;
}

struct Sm120Dev {
    std::string name;
    int cc_major = 0, cc_minor = 0;
    int sm_count = 0;
    uint64_t vram_bytes = 0;
    int ordinal = 0;
};

// True only for cc 12.0 (GB202 / GB203 / GB205 / GB206 / GB207: the RTX 50 cards and the RTX PRO Blackwell cards). cc 12.1 (GB10) is another SASS family.
inline bool ink_sm120_is_cc120(const Sm120Dev& d) { return d.cc_major == 12 && d.cc_minor == 0; }
// cc 12.0 or 12.1: the cards a table row can exist for. Which one a row is for is the row's cc_minor (the GB10 is its own row, never the RTX rows).
inline bool ink_sm120_is_cc12x(const Sm120Dev& d) { return d.cc_major == 12 && (d.cc_minor == 0 || d.cc_minor == 1); }

inline int ink_sm120_vram_gib(uint64_t bytes) { return (int)((bytes + (512ull << 20)) >> 30); }

// The first row whose name, SM window and VRAM window all match. nullptr when no row matches (an unknown cc 12.0 card).
inline const Sm120Row* ink_sm120_match(const Sm120Dev& d) {
    if (!ink_sm120_is_cc12x(d)) return nullptr;
    int n = 0;
    const Sm120Row* rows = ink_sm120_rows(&n);
    const int gib = ink_sm120_vram_gib(d.vram_bytes);
    for (int i = 0; i < n; ++i) {
        const Sm120Row& r = rows[i];
        if (d.cc_minor != r.cc_minor) continue;
        if (d.name.find(r.name_has) == std::string::npos) continue;
        if (d.sm_count < r.sm_min || d.sm_count > r.sm_max) continue;
        if (gib < r.vram_gib_min || gib > r.vram_gib_max) continue;
        return &r;
    }
    return nullptr;
}

inline const Sm120Row* ink_sm120_row_by_id(const char* id) {
    if (!id || !*id) return nullptr;
    int n = 0;
    const Sm120Row* rows = ink_sm120_rows(&n);
    for (int i = 0; i < n; ++i) if (std::strcmp(rows[i].id, id) == 0) return &rows[i];
    return nullptr;
}

// ------------------------------------------------------------------------------------------------------------------------------------------
// The decision. `get` returns the value of an environment variable or nullptr (a test passes its own).
// ------------------------------------------------------------------------------------------------------------------------------------------
struct Sm120Decision {
    enum Mode { Off = 0, Exact = 1, Fast = 2 };
    Mode mode = Off;                  // what runs
    const Sm120Row* row = nullptr;    // the row whose numbers apply (nullptr: no sm_120 feature runs)
    uint32_t parts = 0;               // the parts that are on
    uint32_t forced_on = 0;           // parts an explicit STRATA_INK_SM120_<PART>=1 turned on
    bool asked = false;               // something asked for sm_120 features (a switch was set, or the default is on)
    bool refused = false;             // asked on a card that cannot do it
    bool forced_row = false;          // STRATA_INK_SM120_ROW
    bool default_on_applied = false;  // mode Fast came from kSm120FastDefault, not from a switch
    std::string note;                 // one line for stderr, empty when there is nothing to say
};

template <class Getenv>
inline bool ink_sm120_flag(Getenv&& get, const char* name, const char* alias, int* tri) {
    const char* v = get(name);
    if ((!v || !*v) && alias) v = get(alias);
    if (!v || !*v) { *tri = -1; return false; }
    *tri = (v[0] == '0') ? 0 : 1;
    return true;
}

struct Sm120PartName { Sm120Part bit; const char* name; const char* alias; };
inline const Sm120PartName* ink_sm120_part_names(int* n) {
    static const Sm120PartName t[] = {
        {P_PF_ATTN, "PF_ATTN", nullptr},
        {P_FP8_TRUNK, "FP8_TRUNK", "STRATA_INK_5060TI_FP8_TRUNK"},
        {P_MOE_OPT, "MOE_OPT", nullptr},
        {P_DENSE_OPT, "DENSE_OPT", nullptr},
        {P_DSPARK_OPT, "DSPARK_OPT", nullptr},
        {P_C64_SPEC, "C64_SPEC", "STRATA_INK_5060TI_C64_SPEC"},
        {P_COMPACT, "COMPACT", "STRATA_INK_5060TI_COMPACT"},
        {P_COMPACT_XCODE, "COMPACT_XCODE", nullptr},
        {P_COMPACT_KV, "COMPACT_KV", nullptr},
        {P_FP8_INPLACE, "FP8_INPLACE", nullptr},
        {P_TC_ATTN, "TC_ATTN", nullptr},
        {P_SPEC_ADAPT, "SPEC_ADAPT", nullptr},
        {P_COMPACT_HYBRID, "COMPACT_HYBRID", nullptr},
        {P_TC_SMALLM, "TC_SMALLM", "STRATA_INK_5060TI_TC_SMALLM"},
        {P_PLD, "PLD", "STRATA_INK_5060TI_PLD"},
        {P_PACK, "PACK", "STRATA_INK_5060TI_PACK"},
        {P_PREFILL, "PREFILL", "STRATA_INK_5060TI_PREFILL"},
        {P_CORE_GRAPH, "CORE_GRAPH", "STRATA_INK_5060TI_CORE_GRAPH"},
        {P_L2PF, "L2PF", "STRATA_INK_5060TI_SM120_PF"},
    };
    if (n) *n = (int)(sizeof(t) / sizeof(t[0]));
    return t;
}

// COMPACT's two sub-parts: with COMPACT on and the sub-part not named in any switch, it is on; with COMPACT off, both are off.
inline uint32_t ink_sm120_compact_subparts(uint32_t parts, uint32_t named) {
    if (parts & P_COMPACT) parts |= (kSm120CompactSubParts & ~named);
    else parts &= ~kSm120CompactSubParts;
    // COMPACT_HYBRID (8 / 12 GB cards): the KV stays FP16 unless COMPACT_KV is named. Teacher-forced KL on the TAILSKIP screen (RTX 5060 8 GB, rows lane 11-26): the FP8 KV is the part that
    // costs on long structured contexts (agent_8k_0: PPL 6.14 with the FP8 KV against 5.38 with FP16 KV, old path 5.18); the Q8_ROW trunk alone sits at the engine's chunk-noise floor.
    if ((parts & P_COMPACT_HYBRID) && !(named & P_COMPACT_KV)) parts &= ~P_COMPACT_KV;
    return parts;
}

template <class Getenv>
inline Sm120Decision ink_sm120_decide(const Sm120Dev& dev, Getenv&& env_get) {
    Sm120Decision out;
    int tri = -1;
    // master and exact, new name first, the two old names as aliases
    int master = -1, exact = -1;
    {
        // the names are read with env_get("STRATA_...") calls so that tools/settings_register.py finds them (the register lists every setting the engine reads)
        auto flag_of = [](const char* a, const char* b, const char* c) -> int {
            const char* v = (a && *a) ? a : ((b && *b) ? b : ((c && *c) ? c : nullptr));
            return v ? ((v[0] == '0') ? 0 : 1) : -1;
        };
        master = flag_of(env_get("STRATA_INK_SM120"), env_get("STRATA_INK_RTX5090"), env_get("STRATA_INK_RTX5060TI"));
        exact = flag_of(env_get("STRATA_INK_SM120_EXACT"), env_get("STRATA_INK_RTX5090_EXACT"), env_get("STRATA_INK_RTX5060TI_EXACT"));
    }
    // part switches
    uint32_t part_on = 0, part_off = 0;
    {
        int n = 0;
        const Sm120PartName* names = ink_sm120_part_names(&n);
        for (int i = 0; i < n; ++i) {
            std::string nm = std::string("STRATA_INK_SM120_") + names[i].name;
            if (ink_sm120_flag(env_get, nm.c_str(), names[i].alias, &tri)) {
                if (tri == 1) part_on |= names[i].bit; else part_off |= names[i].bit;
            }
        }
    }
    const bool asked_switch = (master == 1) || (exact == 1) || (part_on != 0);
    if (master == 0) { out.mode = Sm120Decision::Off; return out; }   // "=0" is always the old path
    // THE DEFAULT (owner, 11-26, after FASTQ): with no switch set, a cc 12.0 card that has a MEASURED table row runs that row's fast set, silently. Every other
    // card (placeholder rows: 5080, 5070 Ti, 5070, 5060 Ti 8 GB, 5060, 5050, RTX PRO; unknown cards; every non-RTX 50 card) keeps the old path and prints nothing.
    // A part switched OFF alone (STRATA_INK_SM120_<PART>=0) takes that part out of the default set.
    const Sm120Row* dflt_row = nullptr;
    if (!asked_switch && master < 0 && kSm120FastDefault && ink_sm120_is_cc12x(dev)) {
        dflt_row = ink_sm120_match(dev);
        if (dflt_row && !(dflt_row->measured && dflt_row->default_on)) dflt_row = nullptr;
    }
    const bool want_default = (dflt_row != nullptr);
    out.asked = asked_switch || want_default;
    if (!out.asked) return out;

    char buf[400];
    if (!ink_sm120_is_cc12x(dev)) {
        out.refused = true;
        std::snprintf(buf, sizeof(buf),
                      "[InkCUDA] RTX 50 switch ignored: GPU %d (%s, sm_%d%d) is not an sm_120 card; the default path runs",
                      dev.ordinal, dev.name.c_str(), dev.cc_major, dev.cc_minor);
        out.note = buf;
        return out;
    }
    const Sm120Row* row = ink_sm120_match(dev);
    const char* forced_id = env_get("STRATA_INK_SM120_ROW");
    if (forced_id && *forced_id) {
        const Sm120Row* fr = ink_sm120_row_by_id(forced_id);
        if (!fr) {
            std::snprintf(buf, sizeof(buf), "[InkCUDA] STRATA_INK_SM120_ROW=%s names no table row; the default path runs", forced_id);
            out.note = buf;
            out.refused = true;
            return out;
        }
        row = fr;
        out.forced_row = true;
    }
    if (!row) {
        // an unknown cc 12.0 card: the old path, except for parts an explicit =1 asked for (they run with the 5090 row's generic numbers)
        out.refused = (part_on == 0);
        std::snprintf(buf, sizeof(buf),
                      "[InkCUDA] RTX 50 switch: no measured table row for GPU %d (%s, %d SMs, %d GiB); %s. Row ids: 5090, 5060ti-16 "
                      "(STRATA_INK_SM120_ROW=<id> uses one of them at your own risk)",
                      dev.ordinal, dev.name.c_str(), dev.sm_count, ink_sm120_vram_gib(dev.vram_bytes),
                      part_on ? "only the parts you named run" : "the default path runs");
        out.note = buf;
        if (part_on == 0) return out;
        row = ink_sm120_row_by_id("5090");
        out.row = row;
        out.mode = Sm120Decision::Fast;
        out.parts = ink_sm120_compact_subparts(part_on & ~part_off, part_on | part_off);
        out.forced_on = part_on;
        return out;
    }
    if (!row->measured && !out.forced_row && part_on == 0) {
        out.refused = true;
        std::snprintf(buf, sizeof(buf),
                      "[InkCUDA] RTX 50 switch: GPU %d (%s, %d SMs, %d GiB) matches the table row '%s' but that row is a placeholder (not measured); "
                      "the default path runs. STRATA_INK_SM120_ROW=<5090|5060ti-16> uses a measured row at your own risk",
                      dev.ordinal, dev.name.c_str(), dev.sm_count, ink_sm120_vram_gib(dev.vram_bytes), row->id);
        out.note = buf;
        out.row = row;
        return out;
    }
    if (out.forced_row) {
        std::snprintf(buf, sizeof(buf), "[InkCUDA] RTX 50 switch: GPU %d (%s) runs table row '%s' (%s) because STRATA_INK_SM120_ROW is set",
                      dev.ordinal, dev.name.c_str(), row->id, row->measured ? "measured on another card of this kind" : "placeholder, not measured");
        out.note = buf;
    }
    out.row = row;
    const bool want_exact = (exact == 1);
    out.mode = want_exact ? Sm120Decision::Exact : Sm120Decision::Fast;
    out.default_on_applied = want_default;
    if (want_default) {
        std::snprintf(buf, sizeof(buf),
                      "[InkCUDA] RTX 50 fast mode is the default on GPU %d (%s, table row '%s'): STRATA_INK_SM120=0 turns it off, STRATA_INK_SM120_EXACT=1 gives the bit-identical form",
                      dev.ordinal, dev.name.c_str(), row->id);
        out.note = buf;
    }
    uint32_t base = want_exact ? row->exact_parts : row->fast_parts;
    if (master < 0 && exact < 0 && part_on != 0 && !want_default) base = 0;   // a part switch alone turns that part on, not the whole set
    out.parts = ink_sm120_compact_subparts((base | part_on) & ~part_off, part_on | part_off);
    out.forced_on = part_on;
    return out;
}

inline int ink_sm120_moe_range(const Sm120Row& r, int M) {
    for (int i = 0; i < r.n_moe; ++i) if (M <= r.moe[i].m_max) return i;
    return r.n_moe - 1;
}

}  // namespace strata::core
