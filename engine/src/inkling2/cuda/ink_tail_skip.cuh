// engine/src/inkling2/cuda/ink_tail_skip.cuh - opt-in TAIL SKIP for the Inkling-2-Mini Top-8 router (lane TAILSKIP, 2026-11-26).
//
// After the exact Top-8 selection ((sigmoid + bias) order), the experts of rank >= first_rank whose sigmoid share
// (p_k / sum of the 8 selected p) is <= thr are dropped (and, with cap > 0, only while the dropped share stays <= cap).
// The remaining weights are renormalised to the same route_scale (renorm = 1) or keep their old value (renorm = 0).
// A dropped slot gets expert id -1 and weight 0; k_ink_moe_bin skips id -1, so the expert is neither read nor computed, and the
// router zeroes the dropped slot's y_pair row so the 8-slot reduction needs no change.
//
// OFF by default: no STRATA_INK_TAIL_SKIP in the environment means the router runs exactly as before (same float operations).
// Spec (environment STRATA_INK_TAIL_SKIP or InkBackend::set_tail_skip): "off", or comma separated
//   thr=0.05       share threshold (0.05 = 5 %), default 0.05
//   rank=7         first dropped rank, 1-based, 2..8 (7 = ranks 7 and 8, 8 = rank 8 only), default 7
//   visits=18-41   visit numbers the skip applies to: "all", "a-b", or "a-b+c-d"; default 18-41 (Core Pass 1 and 2); 0..5 Prelude,
//                  6..17 Core Pass 0, 18..29 Pass 1, 30..41 Pass 2, 42..47 Coda
//   cap=0.08       cumulative dropped share cap per token and visit, 0 = no cap (default)
//   renorm=1       renormalise the kept weights (default 1)
//   stats=1        count skipped slots per visit (read with InkBackend::tail_stats)
// Not applied on the host-streamed (hybrid) path and on the dual-GPU expert-parallel path.
#pragma once

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace strata::ink {

constexpr int kInkTailVisits = 48;
constexpr int kInkTailStatW = 14;   // per visit: [0] token-visits seen, [2..8] dropped slots by rank 2..8, [9] dropped share * 1e6,
                                     // [10] sum over MoE calls of the experts the exact router would have read, [11] of the experts read with the skip,
                                     // [12] routed slots that are cold (not in VRAM), [13] of those, slots skipped or swapped for a resident expert

struct InkTailCfg {
    bool on = false;
    int first_rank = 7;                 // 1-based
    float thr = 0.05f;
    float cap = 0.0f;
    int renorm = 1;
    uint64_t visit_mask = 0;            // bit v = visit v
    bool stats = false;
    // lane TAILSKIP part 2: calibrated bias, cold-aware skip, resident fallback
    int mode_fb = 0;                    // 1: a skipped slot is swapped for fb[layer][expert] when that is resident (else skipped)
    int cold_only = 0;                  // 1: only a cold slot (not resident) is eligible
    float comp_scale = 1.0f;            // scale of the bias
    int calib = 0;                      // 1: nothing is skipped; the dropped experts' weighted outputs are summed per layer (tail_calib)
    std::string comp_path, hot_path, fb_path;   // [24,2304] f32 bias, [24,128] u8 virtual hot mask (1 = resident), [24,128] u16 fallback table
};

// Kernel argument (by value). on = 0: the kernel behaves exactly as before.
struct InkTailArgs {
    int on = 0;
    int first_rank0 = 6;                // 0-based first droppable index in the rank-ordered list
    int renorm = 1;
    int visit = 0;
    float thr = 0.0f;
    float cap = 0.0f;
    float* y_pair = nullptr;            // [M * 8, 2304] zeroed for a dropped slot
    unsigned long long* cnt = nullptr;  // [48 * kInkTailStatW] or nullptr
    int ip = 0;                         // physical layer of the visit
    int mode_fb = 0, cold_only = 0, calib = 0;
    float comp_scale = 1.0f;
    const void* ex = nullptr;           // InkExpertDev* of the layer (real tiers: tier 2 = cold)
    const uint8_t* hot = nullptr;       // [24 * 128] virtual hot mask (overrides the tiers)
    const uint16_t* fb = nullptr;       // [24 * 128]
    const float* comp = nullptr;        // [24 * 2304]
    int* mask_out = nullptr;            // [tokens] drop candidates (calibration)
};

inline InkTailCfg& ink_tail_cfg_ref() { static InkTailCfg c; return c; }
inline int& ink_tail_cur_visit() { static int v = -1; return v; }
inline unsigned long long*& ink_tail_cnt_dev() { static unsigned long long* p = nullptr; return p; }

inline bool ink_tail_parse_visits(const std::string& s, uint64_t& mask, std::string& err) {
    mask = 0;
    if (s == "all") { mask = (1ull << kInkTailVisits) - 1; return true; }
    size_t i = 0;
    while (i < s.size()) {
        size_t j = s.find('+', i);
        if (j == std::string::npos) j = s.size();
        const std::string part = s.substr(i, j - i);
        int a = -1, b = -1;
        if (std::sscanf(part.c_str(), "%d-%d", &a, &b) != 2) {
            if (std::sscanf(part.c_str(), "%d", &a) != 1) { err = "bad visits: " + s; return false; }
            b = a;
        }
        if (a < 0 || b < a || b >= kInkTailVisits) { err = "visits out of 0..47: " + s; return false; }
        for (int v = a; v <= b; ++v) mask |= (1ull << v);
        i = j + 1;
    }
    if (!mask) { err = "empty visits"; return false; }
    return true;
}

// The two opt-in presets (owner decision 2026-11-26): `--turbo` / STRATA_INK_TAIL_SKIP=turbo and `--turbo-max` / STRATA_INK_TAIL_SKIP=turbo-max.
// turbo: Pass 2 only (visits 30..41), ranks 7-8 at <= 8 % share, kept weights not renormalised: about 3 % of the expert reads, KL about 1e-3 (docs/INKLING_TURBO.md).
// turbo-max: ranks 6-8 at <= 8 % share on every visit, renormalised: about 16 % of the reads, KL about 1.3e-2.
inline const char* ink_tail_preset(const std::string& name) {
    if (name == "turbo") return "thr=0.08,rank=7,visits=30-41,cap=0,renorm=0";
    if (name == "turbo-max" || name == "turbo_max") return "thr=0.08,rank=6,visits=all,cap=0,renorm=1";
    return nullptr;
}

inline bool ink_tail_parse(const std::string& spec_in, InkTailCfg& out, std::string& err) {
    InkTailCfg c;
    const char* preset = ink_tail_preset(spec_in);
    const std::string spec = preset ? std::string(preset) : spec_in;
    if (spec.empty() || spec == "off" || spec == "0") { out = c; return true; }
    c.on = true;
    (void)ink_tail_parse_visits("18-41", c.visit_mask, err);
    size_t i = 0;
    while (i < spec.size()) {
        size_t j = spec.find(',', i);
        if (j == std::string::npos) j = spec.size();
        const std::string kv = spec.substr(i, j - i);
        const size_t eq = kv.find('=');
        const std::string k = kv.substr(0, eq);
        const std::string v = eq == std::string::npos ? "" : kv.substr(eq + 1);
        if (k == "thr") c.thr = (float)std::atof(v.c_str());
        else if (k == "rank") c.first_rank = std::atoi(v.c_str());
        else if (k == "cap") c.cap = (float)std::atof(v.c_str());
        else if (k == "renorm") c.renorm = std::atoi(v.c_str());
        else if (k == "stats") c.stats = std::atoi(v.c_str()) != 0;
        else if (k == "mode") c.mode_fb = (v == "fb") ? 1 : 0;
        else if (k == "cold") c.cold_only = std::atoi(v.c_str()) != 0;
        else if (k == "compscale") c.comp_scale = (float)std::atof(v.c_str());
        else if (k == "calib") c.calib = std::atoi(v.c_str()) != 0;
        else if (k == "comp") c.comp_path = v;
        else if (k == "hot") c.hot_path = v;
        else if (k == "fb") c.fb_path = v;
        else if (k == "visits") { if (!ink_tail_parse_visits(v, c.visit_mask, err)) return false; }
        else if (k == "on") { /* accepted */ }
        else { err = "unknown tail-skip key: " + k; return false; }
        i = j + 1;
    }
    if (c.first_rank < 2 || c.first_rank > 8) { err = "rank must be 2..8"; return false; }
    if (!(c.thr >= 0.0f && c.thr <= 0.5f)) { err = "thr must be 0..0.5"; return false; }
    if (!(c.cap >= 0.0f && c.cap <= 1.0f)) { err = "cap must be 0..1"; return false; }
    out = c;
    return true;
}

// The environment is read once, on the first use.
inline InkTailCfg& ink_tail_cfg() {
    static bool once = false;
    InkTailCfg& c = ink_tail_cfg_ref();
    if (!once) {
        once = true;
        if (const char* e = std::getenv("STRATA_INK_TAIL_SKIP"); e && e[0]) {
            std::string err;
            InkTailCfg t;
            if (ink_tail_parse(e, t, err)) c = t;
            else std::fprintf(stderr, "[InkCUDA] STRATA_INK_TAIL_SKIP ignored: %s\n", err.c_str());
            if (c.on)
                std::fprintf(stderr, "[InkCUDA] STRATA_INK_TAIL_SKIP is ON (answers differ slightly): thr=%.4f first rank=%d cap=%.3f renorm=%d visits mask=0x%llx\n",
                             c.thr, c.first_rank, c.cap, c.renorm, (unsigned long long)c.visit_mask);
        }
    }
    return c;
}

}  // namespace strata::ink
