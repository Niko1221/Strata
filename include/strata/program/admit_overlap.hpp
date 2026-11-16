// include/strata/program/admit_overlap.hpp - STRATA_ADMIT_OVERLAP: which batch admissions read their prompt beside the
// decoding slot groups (header-only: rules and counters, CPU-tested by admit_overlap_test).
//
// The existing concurrent reader and doorbell threads can keep groups decoding during an admission. This policy
// requires the serial chunk plan, no CPU prompt share, no idle pad rows and RAM copies of all missing experts.
// Normally no expert-cache slots may be lent. LONG is separately opt-in: it permits a fully RAM-backed loan, whose
// experts neighbours compute on the CPU (rounding can differ from the serial schedule). Other reads stay serial.
#pragma once

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>
#include <algorithm>

namespace strata::program::admit_overlap {

/// STRATA_ADMIT_OVERLAP: on for a value whose integer is not 0 (unset, empty, "0": off).
inline bool env_on(const char* v) { return v != nullptr && v[0] != '\0' && std::atoi(v) != 0; }

/// STRATA_ADMIT_OVERLAP_MIN=N: only a read of at least N tokens goes beside the slots (default 64; the thread, the
/// doorbell thread and the drain after it are not worth it for a few tokens).
inline constexpr int64_t kDefaultMinTokens = 64;
inline int64_t env_min_tokens(const char* v) {
    if (v == nullptr || v[0] == '\0') return kDefaultMinTokens;
    const long long n = std::atoll(v);
    return n < 0 ? 0 : (int64_t) n;
}

/// A stream's priority lasts for its entire lifetime, including solo windows. Keep the existing concurrent mode's
/// rule; the admission mode only raises priorities when its separate comparison switch explicitly asks for it.
inline bool windows_top_priority(const char* piped_concurrent, const char* conc_prio, const char* admit_overlap,
                                 const char* admit_prio) {
    if (conc_prio != nullptr && std::atoi(conc_prio) == 0) return false;   // (upstream's test: set and 0, or empty)
    if (piped_concurrent != nullptr && std::atoi(piped_concurrent) != 0) return true;
    return env_on(admit_overlap) && env_on(admit_prio);
}

/// STRATA_ADMIT_OVERLAP_LONG=1 (opt-in, with STRATA_ADMIT_OVERLAP=1; off: the rule above, to the byte): a read of at
/// least STRATA_ADMIT_OVERLAP_LONG_MIN tokens (default 8192) may go beside the slots with a loan out - the long reads,
/// which always lend (the layer-major walk's ring sits in the borrowed slots).  The read is unchanged: the serial path's
/// run, its chunk plan and its loan (lend() runs before the decision), so its bits are the solo read's.  The windows
/// beside it are today's windows with the loan out (as between the pieces of a read, or with a retained prompt loan): the
/// lent experts are missing in VRAM and run on the CPU from their RAM copy - so every lent expert (and every other one
/// outside VRAM) must have one: covered() with the stages' lend floors (`loan_floor`), checked for the file source
/// whether or not it stages.  The slots' tokens are those of windows with the loan out, not of the serial schedule (in
/// which no window runs during the read): see OVERLAP-LONG.md.
inline constexpr int64_t kDefaultLongMinTokens = 8192;
inline int64_t env_long_min_tokens(const char* v) {
    if (v == nullptr || v[0] == '\0') return kDefaultLongMinTokens;
    const long long n = std::atoll(v);
    return n < 0 ? 0 : (int64_t) n;
}

/// Why a read goes the serial way (counted per reason).
enum class Why : int {
    none = 0,       ///< it goes beside the slots
    off,            ///< STRATA_ADMIT_OVERLAP unset
    not_piped,      ///< not the pipelined groups (one GPU, --batch-groups 1, no --batch)
    solo,           ///< not a batch admission (a GEN, not a BGEN)
    legacy,         ///< STRATA_PIPED_CONCURRENT=1 is set: that mode reads (with its give-way), not this one
    setup,          ///< a setting that keeps the concurrent read off (concurrent_read_off, the source aside)
    pad_rows,       ///< idle slots carry pad rows (multi-slot groups without STRATA_PIPED_MTP): a window
                    ///< beside the read would write the admitted slot, whose conversation is parked after the read
    idle,           ///< no slot decoding: nothing to overlap
    short_read,     ///< fewer than STRATA_ADMIT_OVERLAP_MIN tokens
    pictures,       ///< a prompt with pictures
    co_read,        ///< BCO co-reads waiting (their pass is not this read)
    elastic_kv,     ///< the elastic K/V grows during a read (an allocation on the read's thread)
    cpu_share,      ///< STRATA_BATCH_CPU_SHARE / STRATA_PREFILL_CPU_SHARE_BATCH: the serial read takes a CPU share
    swaps_pending,  ///< an adaptive swap not landed yet (it would copy into a slot a read reads in place)
    loan,           ///< a cache slot is lent (this read's loan or a kept one): the windows would compute it on the CPU
    source,         ///< an expert outside VRAM without a RAM copy (the GGUF source would stage it from the file)
    file_reads,     ///< an earlier overlapped read met a blob from the file: no more overlap this session
    loan_uncovered, ///< STRATA_ADMIT_OVERLAP_LONG: a lent expert (or one outside VRAM) has no RAM copy
    kCount
};

inline const char* why_text(Why w) {
    switch (w) {
        case Why::none: return "beside the slots";
        case Why::off: return "off";
        case Why::not_piped: return "not the pipelined groups";
        case Why::solo: return "not a batch admission";
        case Why::legacy: return "STRATA_PIPED_CONCURRENT reads instead";
        case Why::setup: return "a setting keeps the concurrent read off";
        case Why::pad_rows: return "idle slots carry pad rows";
        case Why::idle: return "no slot decoding";
        case Why::short_read: return "short read";
        case Why::pictures: return "pictures";
        case Why::co_read: return "co-read";
        case Why::elastic_kv: return "elastic K/V";
        case Why::cpu_share: return "CPU share";
        case Why::swaps_pending: return "adaptive swaps pending";
        case Why::loan: return "lent cache slots";
        case Why::source: return "an expert outside VRAM has no RAM copy";
        case Why::file_reads: return "file reads seen";
        case Why::loan_uncovered: return "a lent expert has no RAM copy (STRATA_ADMIT_OVERLAP_LONG)";
        case Why::kCount: break;
    }
    return "?";
}

/// One read's facts, in the order the rule looks at them.
struct Read {
    bool on = false;              ///< STRATA_ADMIT_OVERLAP
    bool piped = false;           ///< the pipelined slot groups (--batch-groups > 1 on a layer split)
    bool admission = false;       ///< a BGEN's read (a batch slot is being admitted)
    bool legacy = false;          ///< STRATA_PIPED_CONCURRENT=1 (its own path takes the read)
    bool setup_off = false;       ///< concurrent_read_off() without the source and the batched draft K/V
    bool pad_rows = false;        ///< a group window writes its idle slots (no packed rows)
    bool runtime_off = false;     ///< file reads were seen during an overlapped read
    bool slots_decoding = false;  ///< some batch slot is active
    int64_t tokens = 0;           ///< the read's length
    int64_t min_tokens = kDefaultMinTokens;
    bool pictures = false;
    bool co_read = false;
    bool elastic_kv = false;
    bool cpu_share = false;
    bool swaps_pending = false;
    bool loan_out = false;        ///< any stage has slots lent, after this read's lend()
    bool source_ok = true;        ///< covered() below (true where the source never stages)
    bool long_on = false;         ///< STRATA_ADMIT_OVERLAP_LONG
    int64_t long_min = kDefaultLongMinTokens;
    bool loan_ok = true;          ///< long_on with a loan out: covered() with the lend floors holds
};

/// STRATA_ADMIT_OVERLAP_LONG: the loan does not keep this read serial (a long read; the coverage is checked apart).
inline bool long_loan_allowed(const Read& r) { return r.long_on && r.tokens >= r.long_min && r.tokens > 0; }

/// The first reason the read goes the serial way, or Why::none.
inline Why decide(const Read& r) {
    if (!r.on) return Why::off;
    if (!r.piped) return Why::not_piped;
    if (!r.admission) return Why::solo;
    if (r.legacy) return Why::legacy;
    if (r.setup_off) return Why::setup;
    if (r.pad_rows) return Why::pad_rows;
    if (r.runtime_off) return Why::file_reads;
    if (!r.slots_decoding) return Why::idle;
    if (r.tokens < r.min_tokens || r.tokens <= 0) return Why::short_read;
    if (r.pictures) return Why::pictures;
    if (r.co_read) return Why::co_read;
    if (r.elastic_kv) return Why::elastic_kv;
    if (r.cpu_share) return Why::cpu_share;
    if (r.swaps_pending) return Why::swaps_pending;
    if (r.loan_out) {
        if (!long_loan_allowed(r)) return Why::loan;
        if (!r.loan_ok) return Why::loan_uncovered;
    }
    if (!r.source_ok) return Why::source;
    return Why::none;
}

/// Every (layer, expert) a read or a window can ask for outside VRAM - not resident (host_res < 0), or in a slot at or
/// above its layer's lend floor (floor[l] >= 0: the lowest slot a loan may take on that layer's stage) - must have its
/// RAM copy (has_ram(l, e)).  need: how many were checked; covers: all of them have it; at_l / at_e: the first one
/// without it.  (91073726's check of STRATA_PIPED_CONCURRENT on the resident GGUF source, run before every read.)
struct Coverage {
    int64_t need = 0;
    bool covers = true;
    int64_t at_l = -1, at_e = -1;
};
template <class HasRam>
inline Coverage covered(const int32_t* host_res, int64_t n_layers, int64_t n_expert, const int32_t* floor, HasRam&& has_ram) {
    Coverage c;
    for (int64_t l = 0; l < n_layers; ++l) {
        const int32_t f = floor != nullptr ? floor[l] : -1;
        for (int64_t e = 0; e < n_expert; ++e) {
            const int32_t r = host_res[l * n_expert + e];
            if (r >= 0 && (f < 0 || r < f)) continue;   // in VRAM, never lent
            ++c.need;
            if (!has_ram(l, e)) {
                c.covers = false;
                c.at_l = l;
                c.at_e = e;
                return c;
            }
        }
    }
    return c;
}

/// The lend floor per layer for covered(): for each stage part [lb, le) with a loan out, its first lent slot (the slots
/// at or above it are the loan's); -1 (never lent) elsewhere.  parts: (lb, le, first lent slot or < 0 for no loan).
struct LoanPart {
    int64_t lb = 0, le = 0;
    int32_t first = -1;
};
inline std::vector<int32_t> loan_floor(int64_t n_layers, const std::vector<LoanPart>& parts) {
    std::vector<int32_t> f((size_t) std::max<int64_t>(0, n_layers), -1);
    for (const LoanPart& p : parts) {
        if (p.first < 0) continue;
        for (int64_t l = std::max<int64_t>(0, p.lb); l < std::min(p.le, n_layers); ++l) f[(size_t) l] = p.first;
    }
    return f;
}

/// What the mode did, for the log: reads beside the slots (their tokens, time, the slots' rows meanwhile) and the
/// serial ones per reason.
struct Stats {
    int64_t beside = 0, beside_tokens = 0, beside_rows = 0;
    double beside_ms = 0.0;
    int64_t serial[(int) Why::kCount] = {};
    int64_t serial_total() const {
        int64_t n = 0;
        for (int i = (int) Why::idle; i < (int) Why::kCount; ++i) n += serial[i];
        return n;
    }
    void note_serial(Why w) {
        if (w != Why::none) ++serial[(int) w];
    }
    void note_beside(int64_t tokens, double ms, int64_t rows) {
        ++beside;
        beside_tokens += tokens;
        beside_ms += ms;
        beside_rows += rows;
    }
    /// "N reads beside the slots (T tokens in X ms, R rows decoded meanwhile = S rows/s); M serial (reason k, ...)"
    std::string text() const {
        char b[256];
        std::snprintf(b, sizeof b, "%lld reads beside the slots (%lld tokens in %.0f ms, the slots decoded %lld rows "
                                   "meanwhile = %.1f rows/s); %lld read the serial way",
                      (long long) beside, (long long) beside_tokens, beside_ms, (long long) beside_rows,
                      beside_ms > 0.0 ? 1000.0 * (double) beside_rows / beside_ms : 0.0, (long long) serial_total());
        std::string s = b;
        bool first = true;
        for (int i = (int) Why::idle; i < (int) Why::kCount; ++i) {
            if (serial[i] == 0) continue;
            std::snprintf(b, sizeof b, "%s%s %lld", first ? " (" : ", ", why_text((Why) i), (long long) serial[i]);
            s += b;
            first = false;
        }
        if (!first) s += ")";
        return s;
    }
};

/// The summary is said after the 1st, 2nd, 4th, 8th ... and then every 64th read that the mode looked at.
inline bool say_now(int64_t looked) { return looked > 0 && (looked <= 64 ? (looked & (looked - 1)) == 0 : looked % 64 == 0); }

}  // namespace strata::program::admit_overlap
