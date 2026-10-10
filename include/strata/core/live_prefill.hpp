#pragma once

#include <algorithm>
#include <charconv>
#include <cstdint>
#include <exception>
#include <limits>
#include <string>
#include <string_view>

namespace strata::core {

// Only call after Prefill::run_cooperative has returned: on_chunk alone is not a lifetime boundary.
// Return the old loan before exposing resize permission, and never lend again after cancellation or
// a failed service. The caller preserves the exact next prompt position throughout this operation.
template <class Drain, class ReturnLoan, class Service, class Cancelled, class Lend>
bool live_prefill_pause(bool& active, Drain drain, ReturnLoan return_loan, Service service,
                       Cancelled cancelled, Lend lend, std::string& err) {
    if (!drain(err) || !return_loan(err)) return false;
    active = false;
    if (cancelled()) { err = "cancelled"; return false; }
    if (!service(err)) return false;
    if (cancelled()) { err = "cancelled"; return false; }
    if (!lend(err)) return false;
    active = true;
    return true;
}

// Sized-slot loans must fit wholly within the active prefix. The offsets themselves stay stable under VMM.
inline int64_t live_prefill_first(const uint64_t* offsets, int64_t slots, uint64_t bytes, int64_t keep = 128) {
    if (!offsets || keep < 0 || slots <= keep || bytes == 0 || offsets[slots] - offsets[keep] < bytes) return -1;
    return std::upper_bound(offsets + keep, offsets + slots, offsets[slots] - bytes) - offsets - 1;
}

inline int64_t live_prefill_floor(const uint64_t* offsets, int64_t capacity, uint64_t bytes, int64_t keep = 128) {
    if (!offsets || keep < 0 || capacity <= keep || bytes == 0 || offsets[capacity] - offsets[keep] < bytes) return -1;
    return std::lower_bound(offsets + keep + 1, offsets + capacity + 1, offsets[keep] + bytes) - offsets;
}

inline bool live_prefill_parse_min_retained(std::string_view text, int64_t& out) {
    if (text.empty() || text.front() < '0' || text.front() > '9') return false;
    int64_t value = 0;
    const auto parsed = std::from_chars(text.data(), text.data() + text.size(), value);
    if (parsed.ec != std::errc{} || parsed.ptr != text.data() + text.size() || value > 128) return false;
    out = value;
    return true;
}

// Match the session's environment override, not just its command-line flag. This policy is
// intentionally limited to fixed KV even if the current cache happens to disable kvg_start().
inline bool live_prefill_kv_grow_requested(bool cli, const char* env) {
    return env && env[0] != '\0' ? env[0] != '0' : cli;
}

inline uint64_t live_prefill_mapped_bytes(uint64_t bytes, uint64_t quantum) {
    if (!quantum || bytes > std::numeric_limits<uint64_t>::max() - (quantum - 1))
        return std::numeric_limits<uint64_t>::max();
    return (bytes + quantum - 1) / quantum * quantum;
}

// Preserve the largest prefix that can still fund the COMPLETE minimum chunk, using
// actual sized offsets. Candidate search is arithmetic only: relayout mutates raw views.
inline int64_t live_prefill_retained(const uint64_t* offsets, int64_t slots, uint64_t bytes,
                                     int64_t minimum = 128) {
    if (minimum < 0 || minimum > 128) return -1;
    const int64_t first = live_prefill_first(offsets, slots, bytes, minimum);
    return first < 0 ? -1 : std::min<int64_t>(128, first);
}

// A lower logical floor is useful only when it releases a physical VMM block. Fresh GPU
// shortage may select it; RAM-only controls do not. Once lowered, a hold must not regrow
// just because free memory rebounds. Ordinary growth remains the caller's separate policy.
inline int64_t live_prefill_control_floor(const uint64_t* offsets, int64_t current,
                                         int64_t normal, int64_t pressure,
                                         uint64_t quantum, bool gpu_shortage) {
    if (!offsets || current < 0 || normal < 0 || pressure < 0 || pressure > normal) return -1;
    const bool useful = live_prefill_mapped_bytes(offsets[pressure], quantum) <
                        live_prefill_mapped_bytes(offsets[normal], quantum);
    return gpu_shortage && useful ? pressure : std::min(current, normal);
}

enum class LivePrefillRebind { applied, restored, invalid };

// Restore a previously valid carve while all old mappings are still owned. The caller
// MUST terminate on invalid; neither a successful MEMORY ACK nor GEN is safe afterwards.
template<class Apply, class Restore>
LivePrefillRebind live_prefill_rebind(Apply apply, Restore restore, std::string& err) {
    try {
        if (apply(err)) return LivePrefillRebind::applied;
    } catch (const std::exception& e) { err = e.what(); }
      catch (...) { err = "prompt relayout threw"; }
    std::string rollback;
    try {
        if (restore(rollback)) return LivePrefillRebind::restored;
    } catch (const std::exception& e) { rollback = e.what(); }
      catch (...) { rollback = "prompt rollback threw"; }
    err += "; prior prompt layout could not be restored: " + rollback;
    return LivePrefillRebind::invalid;
}

// A donor can never belong to this loan, including a borrower retained earlier in the same pass. Prefer a
// duplicate still backed by the GPU; otherwise use the coldest measured RAM expert of this same-sized layer.
template <class Resident>
inline int32_t live_prefill_donor(const int32_t* slots, const float* heat, int64_t experts, int32_t first,
                                  Resident resident) {
    int32_t donor = -1;
    for (int32_t e = 0; e < experts; ++e) {
        if (slots[e] >= first || !resident(e) || (slots[e] < 0 && !heat)) continue;
        const bool gpu = slots[e] >= 0;
        const bool old_gpu = donor >= 0 && slots[donor] >= 0;
        if (donor < 0 || (gpu && !old_gpu) || (gpu == old_gpu && heat && heat[e] < heat[donor])) donor = e;
    }
    return donor;
}

} // namespace strata::core
