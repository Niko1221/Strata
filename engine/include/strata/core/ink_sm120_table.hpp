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
    // The architecture family the row belongs to: 120 (RTX 50), 89 (RTX 40 / Ada), 86 (RTX 30 / Ampere GA10x, RTX A-series). A row matches only a device of its family.
    // The Ada / Ampere rows (NV-ADA-AMPERE lane) use only the parts whose kernels run on that family (see ink_sm120_arch_parts()).
    int cc = 120;
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

// 16 GB cards other than the 5060 Ti (RTX 5080, 5070 Ti; rows lane 11-26): the same tiles as the small rows, but Bk128 for the SF4 and MX8 experts below M = 1536: +7-9% mean steps/s on both cards.
#define STRATA_SM120_SMALL_MOE_BK \
    { {128, 16, Sm120Tc::Bk128, Sm120Tc::Bk128, 128}, {160, 32, Sm120Tc::Bk128, Sm120Tc::Bk128, 128}, \
      {1536, 64, Sm120Tc::Bk128, Sm120Tc::Bk128, 128}, {1 << 30, 128, Sm120Tc::Tr128Bk128, Sm120Tc::Tr128, 128} }, 4

constexpr uint32_t kSm120BigFast = P_PF_ATTN | P_FP8_TRUNK | P_MOE_OPT | P_DENSE_OPT | P_DSPARK_OPT | P_C64_SPEC;   // P_TC_ATTN joins this set only after the long-context A/B and the KL check (PERF-LONGCTX)
// The 5090 row's default set: the big set plus TC_ATTN (11-26: FASTQ tools 24/26 and 24/26, code 11/11 both, math 146/150 and 148/150, needles 20/20 both with and without it, KL at the engine's noise floor,
// 128K decode 84 -> 238 tok/s; STRATA_INK_SM120_TC_ATTN=0 takes it out). The other big rows (placeholders, DGX Spark) keep the big set until they are measured.
constexpr uint32_t kSm120Fast5090 = kSm120BigFast | P_TC_ATTN;
// The exact form must not change the DRAFTING POLICY against off (tree shape, window, beam, node count, spec_cost, slot cap): with DFlash 2 as the only drafter DSPARK_OPT (deep verify tree, 2-block chain, row spec numbers) does, and the Spark and the R9700 saw exact differ from off for that reason, so the 5090 exact set is EMPTY: only the bit-identical kernels that have their own switches (PDL, PFQ, the sm_120a build) differ from off.
constexpr uint32_t kSm120BigExact = 0;
// The RTX PRO cards (rows lane 11-26): the 5090 set plus TC_ATTN. With the 5090 row alone the PRO 5000 72 GB, the PRO 6000 and the PRO 4500 / 4000 decoded SLOWER than off at 32K
// (PRO 5000 72 GB, DSpark: 150 off, 111 fast; DFlash 2: 152 off, 119 fast); fast + TC_ATTN gives 186 (DSpark) and 146 (DFlash 2), the 4K numbers do not move, and its teacher-forced KL
// against off is the fast set's (0.006-0.011 for both, fast against fast + TC_ATTN 0.0007-0.0026, decode path, 1024 tokens x 4 sequences).
constexpr uint32_t kSm120ProFast = kSm120BigFast | P_TC_ATTN;
// The RTX 5090 Laptop GPU (GB203, 82 SMs, 24 GB; OLARES-0223, Olares One, 11-27): the big set WITHOUT MOE_OPT and without TC_ATTN. Measured on the card against the old path (SQ-20 v2, DFlash 2, C1, 128 new tokens,
// 2 rounds interleaved): the PRO 4000 / 5090 sets lose on json, count and tool decode (the deep verify tree offers 394-440 nodes for 262-274 on the old path) and MOE_OPT costs short-prompt decode
// (the M = 1 kernels) while giving the 4K prefill (10.4k against 6.8k tok/s): without TC_ATTN and MOE_OPT the decode mean of the six prompts is 241 tok/s against 201 (old path), no prompt more than 3% lower,
// 4K document decode 187 against 118, 4K prefill 6.8k against 4.2k tok/s. rel0223/olares/RESULTS.md.
constexpr uint32_t kSm120Laptop5090Fast = kSm120BigFast & ~P_MOE_OPT;
// The 8 and 12 GB cards (experts do not all fit): COMPACT with the compact trunk and the hybrid expert placement (the adaptive tier gets the 2.3 GiB the trunk frees), FP16 KV (the rule in
// ink_sm120_compact_subparts), the tensor-core small-M kernels. PACK, PREFILL, PLD and CORE_GRAPH are out: measured on the RTX 5060 8 GB they made it slower (wider verify trees cost CPU
// reads: 10-30 tok/s against 28-37). The batched cold-expert split between the CPU lane and PCIe comes from the card's own launch probe.
constexpr uint32_t kSm120HybridFast = P_COMPACT | P_COMPACT_XCODE | P_COMPACT_KV | P_COMPACT_HYBRID | P_TC_SMALLM | P_L2PF;
// The DGX Spark's exact set is empty: DSPARK_OPT replaces the drafting numbers (cost, plain margin) and the verify tree, which moved the text against off with a drafter on (GB10, json and tool
// prompts); without it exact = off on every prompt, with and without a drafter. (The 5090 and PRO exact set above is empty for the same reason.)
constexpr uint32_t kSm120SparkExact = 0;
// The 5060 Ti fast set = COMPACT (MX8 -> SF4 transcode + FP8 KV + single slab, attention weights Q8_ROW) + the rest, WITHOUT any FP8 trunk. FASTQ 11-26: with the in-place FP8
// trunk on, fast mode lost 17 of 26 tool calls; with FP8_TRUNK=0 (attention in Q8_ROW) it is at or above off on every suite (report-5060ti-fp8t0.md). The in-place FP8 trunk is
// explicit-only (STRATA_INK_SM120_FP8_INPLACE=1); FP8_TRUNK (the 5090 cache engine) cannot start next to compact (no room), so it is not in this set either.
constexpr uint32_t kSm120SmallFast = P_COMPACT | P_TC_SMALLM | P_PLD | P_PACK | P_PREFILL | P_CORE_GRAPH | P_C64_SPEC | P_L2PF;
// Exact set of the small cards, trimmed by measurement (5060 Ti, new SQ-20 hybrid load, 2026-11-26): PACK (moves experts between the CPU and the GPU lane: rounding differs)
// and PREFILL (layer-major prefill: tokens differ from gen:9000 on) are out. Then PLD (tokens differ at gen:9000, 2 of 2 runs) and CORE_GRAPH (differ at gen:30 from token 67 when
// combined with the others) are out too. C64_SPEC + L2PF: tokens identical to off at 30, 9000 and 12000 (2 rounds each), speed equal to off. The speed-ups of the others are fast-only.
// C64_SPEC is out of the exact set too: its multi-user memory plan (FP8 KV, host-resident SWA) assumes COMPACT; with exact (no compact) a 64-slot run on a 16 GB card died ("not enough free GPU memory for the working buffers", global KV all in host RAM). L2PF alone is bit-identical.
constexpr uint32_t kSm120SmallExact = P_L2PF;

// NV-ADA-AMPERE candidate sets (to be trimmed by the per-card A/B runs; see rel0223/nv-ada-ampere/RESULTS.md). The sm_120 FP4 kernels do not exist here, so no TC_SMALLM / PREFILL / L2PF.
constexpr uint32_t kAdaBigFast    = P_PF_ATTN | P_FP8_TRUNK | P_DENSE_OPT | P_DSPARK_OPT | P_C64_SPEC | P_TC_ATTN;
constexpr uint32_t kAmpereBigFast = P_PF_ATTN | P_DENSE_OPT | P_DSPARK_OPT | P_C64_SPEC | P_TC_ATTN;
// Measured on an RTX 3060 12 GB, an RTX 4060 Ti 8 GB and 16 GB (NV-ADA-AMPERE, SQ-20 v2, 11-27): no PACK (no gain with every expert resident, -30% at 9K on a hybrid load), no PF_ATTN (+14-18% prefill at 9-12K but -16% at 16K and -39% at 32K on the 16 GB compact card).
constexpr uint32_t kAdaSmallFast    = P_COMPACT | P_COMPACT_HYBRID | P_PLD | P_CORE_GRAPH | P_C64_SPEC | P_TC_ATTN;
constexpr uint32_t kAmpereSmallFast = P_COMPACT | P_COMPACT_HYBRID | P_PLD | P_CORE_GRAPH | P_C64_SPEC | P_TC_ATTN;
// 16 GB rows (4080, 4070 Ti SUPER, 4060 Ti 16 GB, A4000) carry NO COMPACT_HYBRID: it lets COMPACT run at 2..15 slots and the engine then died at the start with 3 MiB free (NV-ADA, 11-27).
// Only the 12 GB and 8 GB rows (the sets above) hold it.
constexpr uint32_t kAdaSmallFast16    = kAdaSmallFast & ~P_COMPACT_HYBRID;
constexpr uint32_t kAmpereSmallFast16 = kAmpereSmallFast & ~P_COMPACT_HYBRID;
constexpr uint32_t kAdaExact = 0;
#define STRATA_SM120_ADA_MOE { {1 << 30, 16, Sm120Tc::Base, Sm120Tc::Base, 0} }, 1

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
         kSm120Fast5090, kSm120BigExact, Sm120Fp8::Cache, Sm120C64::Bspec, STRATA_SM120_BIG_MOE,
         4096, 4800, 0, 0, 512, 768, 1536, 0,
         4, 12, 24, 3, 8.0f, 1.02f, 0.28f, true},
        {"5060ti-16", "RTX 5060 Ti", 34, 38, 15, 17, true, "measured on a 16 GB RTX 5060 Ti (hand-over 11-26)",
         kSm120SmallFast, kSm120SmallExact, Sm120Fp8::Cache, Sm120C64::K8v4, STRATA_SM120_SMALL_MOE,
         0, 0, 256, 20, 512, 768, 0, 1,   // the 5090's FP8 cache: 512 MiB headroom, 768 MiB reserved for it in the K/V sizing (PLACEHOLDER: measure on a 5060 Ti)
         1, 12, 24, 2, 2.0f, 1.02f, 0.28f, true},
        // ---- placeholders: not measured, the old path runs unless STRATA_INK_SM120_ROW names a row ----
        {"5080", "RTX 5080", 80, 88, 15, 17, true, "measured (rows lane 11-26, Vast 5080 16 GB): fast = compact; +60% C1 decode, 4K prefill 2.7x; 32K prefill and decode LOSE (K/V in host memory): default off, STRATA_INK_SM120=1 turns it on",
         kSm120SmallFast, kSm120SmallExact, Sm120Fp8::Cache, Sm120C64::K8v4, STRATA_SM120_SMALL_MOE_BK,
         0, 0, 256, 20, 0, 0, 0, 1,
         1, 12, 24, 2, 2.0f, 1.02f, 0.28f, false},
        {"5070ti", "RTX 5070 Ti", 66, 74, 15, 17, true, "measured (rows lane 11-26, Vast 5070 Ti 16 GB): fast = compact; C1 decode +50%, 4K prefill 2.4x; 32K prefill loses: default off, STRATA_INK_SM120=1 turns it on",
         kSm120SmallFast, kSm120SmallExact, Sm120Fp8::Cache, Sm120C64::K8v4, STRATA_SM120_SMALL_MOE_BK,
         0, 0, 256, 20, 0, 0, 0, 1,
         1, 12, 24, 2, 2.0f, 1.02f, 0.28f, false},
        {"5070", "RTX 5070", 44, 52, 11, 13, true, "measured (rows lane 11-26, Vast 5070 12 GB, Ryzen 9 7900X, 31 GB): COMPACT_HYBRID with FP16 KV; C1 decode +30-60%, 4K prefill +37%; FASTQ within noise, KL at the engine floor",
         kSm120HybridFast, kSm120SmallExact, Sm120Fp8::Cache, Sm120C64::K8v4, STRATA_SM120_SMALL_MOE,
         0, 0, 256, 20, 0, 0, 0, 0,
         1, 12, 24, 2, 2.0f, 1.02f, 0.28f, true},
        {"5060ti-8", "RTX 5060 Ti", 34, 38, 7, 9, false, "placeholder: 8 GB variant of the 5060 Ti",
         kSm120SmallFast, kSm120SmallExact, Sm120Fp8::Cache, Sm120C64::K8v4, STRATA_SM120_SMALL_MOE,
         0, 0, 256, 20, 0, 0, 0, 0,
         1, 12, 24, 2, 2.0f, 1.02f, 0.28f},
        {"5060", "RTX 5060", 28, 32, 7, 9, true, "measured (rows lane 11-26, Vast 5060 8 GB, Ryzen 5 7600X, 31 GB): COMPACT_HYBRID with FP16 KV; C1 decode 2.2x on prose / tool, 1.5x on code",
         kSm120HybridFast, kSm120SmallExact, Sm120Fp8::Cache, Sm120C64::K8v4, STRATA_SM120_SMALL_MOE,
         0, 0, 256, 20, 0, 0, 0, 0,
         1, 12, 24, 2, 2.0f, 1.02f, 0.28f, false},
        {"5050", "RTX 5050", 18, 22, 7, 9, false, "PLACEHOLDER: 8 GB, 20 SMs; numbers from a rental later",
         kSm120SmallFast, kSm120SmallExact, Sm120Fp8::Cache, Sm120C64::K8v4, STRATA_SM120_SMALL_MOE,
         0, 0, 256, 20, 0, 0, 0, 0,
         1, 12, 24, 2, 2.0f, 1.02f, 0.28f},
        {"pro6000", "RTX PRO 6000", 180, 200, 90, 100, true, "measured (rows lane 11-26, Vast PRO 6000 WS, 500 W; the Max-Q has a lower power limit): 4K / 32K prefill 2.1x / 2.2x, C4 +60%, C8 +82%",
         kSm120ProFast, kSm120BigExact, Sm120Fp8::Cache, Sm120C64::Bspec, STRATA_SM120_BIG_MOE,
         4096, 4800, 0, 0, 512, 768, 1536, 0,
         4, 12, 24, 3, 8.0f, 1.02f, 0.28f, true},
        {"pro5000", "RTX PRO 5000", 100, 120, 40, 80, true, "measured (rows lane 11-26, 72 GB on l620 at 300 W; 48 GB on Vast): 4K / 32K prefill 2.0x / 2.5x, C4 +67%, C8 +96%",
         kSm120ProFast, kSm120BigExact, Sm120Fp8::Cache, Sm120C64::Bspec, STRATA_SM120_BIG_MOE,
         4096, 4800, 0, 0, 512, 768, 1536, 0,
         4, 12, 24, 3, 8.0f, 1.02f, 0.28f, true},
        {"pro4500", "RTX PRO 4500", 76, 90, 30, 34, true, "measured (rows lane 11-26, Vast PRO 4500 32 GB, 200 W): 4K / 32K prefill 2.1x / 2.6x, C4 +73%, C8 +102%",
         kSm120ProFast, kSm120BigExact, Sm120Fp8::Cache, Sm120C64::Bspec, STRATA_SM120_BIG_MOE,
         4096, 4800, 0, 0, 512, 768, 1536, 0,
         4, 12, 24, 3, 8.0f, 1.02f, 0.28f, true},
        {"pro4000", "RTX PRO 4000", 66, 74, 22, 26, true, "measured (rows lane 11-26, Vast PRO 4000 24 GB, 145 W): 4K / 32K prefill 1.7x / 2.2x, C4 +36%, C8 +39%; peak 23.4 GB of 24.4 (the 5090 row, not compact)",
         kSm120ProFast, kSm120BigExact, Sm120Fp8::Cache, Sm120C64::Bspec, STRATA_SM120_BIG_MOE,
         4096, 4800, 0, 0, 512, 768, 1536, 0,
         4, 12, 24, 3, 8.0f, 1.02f, 0.28f, true},
        // The laptop 5090 (name "NVIDIA GeForce RTX 5090 Laptop GPU", 82 SMs, 23,975 MiB). The VRAM window is wide on purpose: an Olares app sees the HAMi limit (nvidia.com/gpumem, 22-24 GiB) as the total.
        // measured, default_on = false until the gate (FASTQ, KL, identity, --parallel 1..64) has passed on the card (OLARES-0223); STRATA_INK_SM120=1 turns it on.
        {"5090m", "RTX 5090 Laptop", 78, 86, 20, 25, true, "measured on an RTX 5090 Laptop GPU 24 GB (Olares One, 11-27): C1 decode mean 241 against 201 tok/s on the old path, 4K prefill 6.8k against 4.2k",
         kSm120Laptop5090Fast, kSm120BigExact, Sm120Fp8::Cache, Sm120C64::Bspec, STRATA_SM120_BIG_MOE,
         4096, 4800, 0, 0, 512, 768, 1536, 0,
         4, 12, 24, 3, 8.0f, 1.02f, 0.28f, false},
        {"pro2000", "RTX PRO 2000", 24, 32, 15, 17, false, "placeholder: RTX PRO 2000 Blackwell (16 GB)",
         kSm120SmallFast, kSm120SmallExact, Sm120Fp8::Cache, Sm120C64::K8v4, STRATA_SM120_SMALL_MOE,
         0, 0, 256, 20, 0, 0, 0, 1,
         1, 12, 24, 2, 2.0f, 1.02f, 0.28f},
        // ---- DGX Spark (NVIDIA GB10, cc 12.1, 48 SMs, 121 GB of memory shared with the CPU, about 273 GB/s) ----
        // Matched by name + SM count + memory AND cc 12.1, so it never takes the 5090 row and the 5090 never takes it. The whole model sits in the unified memory (no hybrid split,
        // no compact, no PACK), so the set is the 5090's "big" set: FP8 trunk with the cache copy (641 MiB, room for it), the NVFP4 tensor-core MoE, PF_ATTN, drafting.
        // MEASURED on a GB10 (Vast, 11-27, SQ-20 v2, DFlash 2; rel0223/spark/RESULTS.md). The set is the 5090's "big" fast set: dropping MOE_OPT or the FP8 trunk lost prefill and decode, TC_ATTN lost
        // at C1 (code 65 vs 115 tok/s), PDL and SPEC_ADAPT changed nothing. What the 5090 numbers got wrong is the drafting: spec_cost 8 / plain 1.02 made fast decode at the plain rate (53 tok/s) on
        // prose and chat, below off; cost 5 / plain 0 gives C1 decode prose 79 / chat 94 / json 99 / code 115 / count 183 / tool 112 against off 55 / 69 / 81 / 78 / 97 / 81 (mean 113.6 vs 76.8).
        // The exact set is EMPTY: DSPARK_OPT in it changed the window numbers and the verify tree, so exact tokens differed from off with a drafter (json, tool); with it out, exact = off on every prompt.
        // default_on: FASTQ subset on the GB10 (fast vs off: needles 8/8 and 8/8, tools 24/26 and 24/26, agent 7/10 and 6/10, code 11/11 and 11/11) is not worse than off. KNOWN: with the IQ3_S GGUF the fast set is mixed (C32 167 vs 190 tok/s at the default 64 drafting slots, json / code C1 slower); see RESULTS.md.
        {"gb10", "GB10", 44, 52, 100, 128, true, "measured on a DGX Spark GB10 (cc 12.1, 48 SMs, 121 GB unified memory, 11-27)",
         kSm120BigFast, kSm120SparkExact, Sm120Fp8::Cache, Sm120C64::Bspec, STRATA_SM120_BIG_MOE,
         4096, 4800, 0, 0, 512, 768, 1536, 0,
         4, 12, 24, 3, 5.0f, 0.0f, 0.0f, true, 1},
        // ---- NV-ADA-AMPERE lane: RTX 40 (cc 8.9) and RTX 30 / A-series (cc 8.6). Candidate sets until a card is measured (measured = false: old path unless STRATA_INK_SM120_ROW=<id>) ----
#define STRATA_ADA_ROW(ID, NAME, SMLO, SMHI, GLO, GHI, NOTE, FAST, EXACT, FP8, C64, FAM)         {ID, NAME, SMLO, SMHI, GLO, GHI, false, NOTE, FAST, EXACT, FP8, C64, STRATA_SM120_ADA_MOE,          0, 0, ((FAST) & P_COMPACT) ? 256 : 0, ((FAST) & P_COMPACT) ? 20 : 0, 512, 768, 1536, 0, 1, 12, 24, 2, 2.0f, 1.02f, 0.28f, false, (FAM) % 10, FAM}
#define STRATA_ADA_ROW_MEAS(ID, NAME, SMLO, SMHI, GLO, GHI, NOTE, FAST, EXACT, FP8, C64, FAM)         {ID, NAME, SMLO, SMHI, GLO, GHI, true, NOTE, FAST, EXACT, FP8, C64, STRATA_SM120_ADA_MOE,          0, 0, ((FAST) & P_COMPACT) ? 256 : 0, ((FAST) & P_COMPACT) ? 20 : 0, 512, 768, 1536, 0, 1, 12, 24, 2, 2.0f, 1.02f, 0.28f, false, (FAM) % 10, FAM}
        STRATA_ADA_ROW("4090", "RTX 4090", 120, 136, 22, 26, "RTX 4090 24 GB", kAdaBigFast, kAdaExact, Sm120Fp8::Cache, Sm120C64::Bspec, 89),
        STRATA_ADA_ROW("4080", "RTX 4080", 74, 84, 15, 17, "RTX 4080 16 GB", kAdaSmallFast16, kAdaExact, Sm120Fp8::Cache, Sm120C64::K8v4, 89),
        STRATA_ADA_ROW("4070tis", "RTX 4070 Ti SUPER", 64, 68, 15, 17, "RTX 4070 Ti SUPER 16 GB", kAdaSmallFast16, kAdaExact, Sm120Fp8::Cache, Sm120C64::K8v4, 89),
        STRATA_ADA_ROW("4070ti", "RTX 4070 Ti", 58, 62, 11, 13, "RTX 4070 Ti 12 GB", kAdaSmallFast, kAdaExact, Sm120Fp8::Cache, Sm120C64::K8v4, 89),
        STRATA_ADA_ROW("4070", "RTX 4070", 44, 58, 11, 13, "RTX 4070 / 4070 SUPER 12 GB", kAdaSmallFast, kAdaExact, Sm120Fp8::Cache, Sm120C64::K8v4, 89),
        STRATA_ADA_ROW_MEAS("4060ti-16", "RTX 4060 Ti", 32, 36, 15, 17, "RTX 4060 Ti 16 GB", kAdaSmallFast16, kAdaExact, Sm120Fp8::Cache, Sm120C64::K8v4, 89),
        STRATA_ADA_ROW("4060ti-8", "RTX 4060 Ti", 32, 36, 7, 9, "RTX 4060 Ti 8 GB", kAdaSmallFast, kAdaExact, Sm120Fp8::Cache, Sm120C64::K8v4, 89),
        STRATA_ADA_ROW("4060", "RTX 4060", 22, 26, 7, 9, "RTX 4060 8 GB", kAdaSmallFast, kAdaExact, Sm120Fp8::Cache, Sm120C64::K8v4, 89),
        STRATA_ADA_ROW("3090", "RTX 3090", 80, 86, 22, 26, "RTX 3090 / 3090 Ti 24 GB", kAmpereBigFast, kAdaExact, Sm120Fp8::None, Sm120C64::Bspec, 86),
        STRATA_ADA_ROW("3080ti", "RTX 3080 Ti", 78, 82, 11, 13, "RTX 3080 Ti 12 GB", kAmpereSmallFast, kAdaExact, Sm120Fp8::None, Sm120C64::K8v4, 86),
        STRATA_ADA_ROW("3080", "RTX 3080", 66, 72, 9, 13, "RTX 3080 10 / 12 GB", kAmpereSmallFast, kAdaExact, Sm120Fp8::None, Sm120C64::K8v4, 86),
        STRATA_ADA_ROW("3070", "RTX 3070", 44, 50, 7, 9, "RTX 3070 / 3070 Ti 8 GB", kAmpereSmallFast, kAdaExact, Sm120Fp8::None, Sm120C64::K8v4, 86),
        STRATA_ADA_ROW("3060ti", "RTX 3060 Ti", 36, 40, 7, 9, "RTX 3060 Ti 8 GB", kAmpereSmallFast, kAdaExact, Sm120Fp8::None, Sm120C64::K8v4, 86),
        STRATA_ADA_ROW_MEAS("3060", "RTX 3060", 26, 30, 11, 13, "RTX 3060 12 GB", kAmpereSmallFast, kAdaExact, Sm120Fp8::None, Sm120C64::K8v4, 86),
        STRATA_ADA_ROW("a4000", "RTX A4000", 44, 50, 15, 17, "RTX A4000 16 GB", kAmpereSmallFast16, kAdaExact, Sm120Fp8::None, Sm120C64::K8v4, 86),
        STRATA_ADA_ROW("a5000", "RTX A5000", 60, 66, 23, 25, "RTX A5000 24 GB", kAmpereBigFast, kAdaExact, Sm120Fp8::None, Sm120C64::Bspec, 86),
        STRATA_ADA_ROW("a6000", "RTX A6000", 80, 86, 47, 49, "RTX A6000 48 GB", kAmpereBigFast, kAdaExact, Sm120Fp8::None, Sm120C64::Bspec, 86),
#undef STRATA_ADA_ROW
#undef STRATA_ADA_ROW_MEAS
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

// The family a device belongs to for this table: 120, 89 (Ada), 86 (Ampere GA10x) or 0 (anything else: the table does nothing).
inline int ink_sm120_family(const Sm120Dev& d) {
    if (d.cc_major == 12 && (d.cc_minor == 0 || d.cc_minor == 1)) return 120;   // (a row's cc_minor says which: 0 the RTX 50 cards, 1 the GB10)
    if (d.cc_major == 8 && d.cc_minor == 9) return 89;
    if (d.cc_major == 8 && d.cc_minor == 6) return 86;
    return 0;
}

// The parts whose kernels exist on a family. The sm_120 FP4 tensor-core kernels (NVFP4 mma: TC_SMALLM, the K256 / layer-major prefill, the L2 prefetch of those kernels) need
// sm_120; the cublasLt FP8 trunk needs FP8 tensor cores (sm_89 and up); PDL (sm_90) is not a part. Everything else is plain mma.sync / dp4a code that runs from sm_80.
constexpr uint32_t kSm120AllPartsMask = 0xFFFFFFFFu;
constexpr uint32_t kSm120NoFp4Parts = P_TC_SMALLM | P_PREFILL | P_L2PF | P_FP8_INPLACE;
inline uint32_t ink_sm120_arch_parts(int family) {
    switch (family) {
        case 120: return kSm120AllPartsMask;
        case 89:  return ~kSm120NoFp4Parts;
        case 86:  return ~(kSm120NoFp4Parts | P_FP8_TRUNK);
        default:  return 0;
    }
}

inline int ink_sm120_vram_gib(uint64_t bytes) { return (int)((bytes + (512ull << 20)) >> 30); }

// The first row whose name, SM window and VRAM window all match. nullptr when no row matches (an unknown cc 12.0 card).
inline const Sm120Row* ink_sm120_match(const Sm120Dev& d) {
    const int fam = ink_sm120_family(d);
    if (fam == 0) return nullptr;
    int n = 0;
    const Sm120Row* rows = ink_sm120_rows(&n);
    const int gib = ink_sm120_vram_gib(d.vram_bytes);
    for (int i = 0; i < n; ++i) {
        const Sm120Row& r = rows[i];
        if (d.cc_minor != r.cc_minor) continue;
        if (r.cc != fam) continue;
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
    if (!asked_switch && master < 0 && kSm120FastDefault && ink_sm120_family(dev) != 0) {
        dflt_row = ink_sm120_match(dev);
        if (dflt_row && !(dflt_row->measured && dflt_row->default_on)) dflt_row = nullptr;
    }
    const bool want_default = (dflt_row != nullptr);
    out.asked = asked_switch || want_default;
    if (!out.asked) return out;

    char buf[400];
    const int fam = ink_sm120_family(dev);
    const char* fl = fam == 89 ? "RTX 40" : fam == 86 ? "RTX 30" : "RTX 50";   // the label in the lines below
    if (fam == 0) {
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
                      "[InkCUDA] %s switch: no measured table row for GPU %d (%s, %d SMs, %d GiB); %s. Row ids: 5090, 5060ti-16 "
                      "(STRATA_INK_SM120_ROW=<id> uses one of them at your own risk)",
                      fl, dev.ordinal, dev.name.c_str(), dev.sm_count, ink_sm120_vram_gib(dev.vram_bytes),
                      part_on ? "only the parts you named run" : "the default path runs");
        out.note = buf;
        if (part_on == 0) return out;
        row = ink_sm120_row_by_id("5090");
        out.row = row;
        out.mode = Sm120Decision::Fast;
        out.parts = ink_sm120_compact_subparts(part_on & ~part_off, part_on | part_off) & ink_sm120_arch_parts(ink_sm120_family(dev));
        out.forced_on = part_on & ink_sm120_arch_parts(ink_sm120_family(dev));
        return out;
    }
    if (!row->measured && !out.forced_row && part_on == 0) {
        out.refused = true;
        std::snprintf(buf, sizeof(buf),
                      "[InkCUDA] %s switch: GPU %d (%s, %d SMs, %d GiB) matches the table row '%s' but that row is a placeholder (not measured); "
                      "the default path runs. STRATA_INK_SM120_ROW=<row id> uses a row at your own risk",
                      fl, dev.ordinal, dev.name.c_str(), dev.sm_count, ink_sm120_vram_gib(dev.vram_bytes), row->id);
        out.note = buf;
        out.row = row;
        return out;
    }
    if (out.forced_row) {
        std::snprintf(buf, sizeof(buf), "[InkCUDA] %s switch: GPU %d (%s) runs table row '%s' (%s) because STRATA_INK_SM120_ROW is set",
                      fl, dev.ordinal, dev.name.c_str(), row->id, row->measured ? "measured on another card of this kind" : "placeholder, not measured");
        out.note = buf;
    }
    out.row = row;
    const bool want_exact = (exact == 1);
    out.mode = want_exact ? Sm120Decision::Exact : Sm120Decision::Fast;
    out.default_on_applied = want_default;
    if (want_default) {
        std::snprintf(buf, sizeof(buf),
                      "[InkCUDA] %s fast mode is the default on GPU %d (%s, table row '%s'): STRATA_INK_SM120=0 turns it off, STRATA_INK_SM120_EXACT=1 gives the bit-identical form",
                      fl, dev.ordinal, dev.name.c_str(), row->id);
        out.note = buf;
    }
    uint32_t base = want_exact ? row->exact_parts : row->fast_parts;
    if (master < 0 && exact < 0 && part_on != 0 && !want_default) base = 0;   // a part switch alone turns that part on, not the whole set
    out.parts = ink_sm120_compact_subparts((base | part_on) & ~part_off, part_on | part_off);
    // the parts whose kernels do not exist on this family (the FP4 tensor-core kernels need sm_120, the cublasLt FP8 trunk needs sm_89) never run on it
    const uint32_t avail = ink_sm120_arch_parts(ink_sm120_family(dev));
    if ((part_on & ~avail) != 0 && out.note.empty()) {
        std::snprintf(buf, sizeof(buf), "[InkCUDA] GPU %d (%s, sm_%d%d): the parts you named that need other silicon (mask 0x%x) are ignored",
                      dev.ordinal, dev.name.c_str(), dev.cc_major, dev.cc_minor, (unsigned)(part_on & ~avail));
        out.note = buf;
    }
    out.parts &= avail;
    // DENSE_OPT, DSPARK_OPT and MOE_OPT allocate the 5090's big-card scratch; on a card of 16 GB or less next to COMPACT they die (79 MiB free: cuBLAS error 7 / "invalid argument" at ink_cuda_backend.cu:1191,
    // RTX 4060 Ti 16 GB, 11-27). A row's set never turns them on there; only an explicit STRATA_INK_SM120_<PART>=1 does (own risk).
    if (ink_sm120_family(dev) != 120 && ink_sm120_vram_gib(dev.vram_bytes) <= 16) out.parts &= ~((P_DENSE_OPT | P_DSPARK_OPT | P_MOE_OPT) & ~part_on);
    out.forced_on = part_on & avail;
    return out;
}

inline int ink_sm120_moe_range(const Sm120Row& r, int M) {
    for (int i = 0; i < r.n_moe; ++i) if (M <= r.moe[i].m_max) return i;
    return r.n_moe - 1;
}

}  // namespace strata::core
