// include/strata/program/adaptive_slots.hpp - the arithmetic of `--parallel-recycle` that a host test can walk.
//
// `--parallel-recycle` turns `--batch N` into a CAP: the engine keeps no batch slot session while it
// is idle and grows the layout inside the BGEN that needs it.  Two pieces of that decide whether the engine is safe
// and were worth their own header:
//
//   * `--vram-elastic`'s conditions, which `--parallel-recycle` shares: the option IMPLIES it (the cache has to be
//     segmented to hand VRAM to the slots), but only where `--vram-elastic` itself can work.  Both the block that
//     applies it and the adaptive validation read this one rule, so the adaptive feature can never stay on after the
//     elastic cache has been refused;
//   * the grow target.  A BGEN grows for its own slot, for the BGEN lines already queued behind it (one cache resize
//     per burst) and for the BCO co-reads waiting for it; a co-read may only name a slot that is actually reserved.
#pragma once

namespace strata::program::adaptive_slots {

/// `--vram-elastic`'s final veto (#533), shared by the block that applies it and by `--parallel-recycle`, which
/// depends on it.  nullptr: it can be on.  Every input is settled before the batch slot sessions are sized (-serve,
/// the final layer-split decision, the helper caches and --vram-segment-mib).
inline const char* vram_elastic_veto(bool serve, bool multi_gpu, const bool remote_helpers[3], bool segment_ok,
                                     bool hip) {
    if (!serve) return "it works between requests of --serve";
    if (multi_gpu) return "a layer split has a cache per GPU";
    for (int i = 0; i < 3; ++i)
        if (remote_helpers != nullptr && remote_helpers[i]) return "the helper caches on other GPUs";
    if (!segment_ok) return "--vram-segment-mib is below 64";
    if (hip) return "it is NVIDIA-only for now";
    return nullptr;
}

/// UX: `--parallel-recycle` with no explicit `--vram-elastic` turns the elastic cache on, with the
/// default 512 MiB segments (or whatever `--vram-segment-mib` says).  Only where `--vram-elastic` itself can work:
/// `veto` (vram_elastic_veto) non-null keeps the old refusal.
inline bool implies_vram_elastic(bool recycle_on, bool explicit_elastic, const char* veto) {
    return recycle_on && !explicit_elastic && veto == nullptr;
}

/// The slots the BGEN for slot `b` must reserve: its own, every BGEN line already queued behind it, and the BCO
/// co-reads waiting for it.  Never above `cap` (the --batch the operator asked for).
inline int grow_target(int b, int queued_bgen_need, int co_need, int cap) {
    int want = b + 1;
    if (queued_bgen_need > want) want = queued_bgen_need;
    if (co_need > want) want = co_need;
    return want > cap ? cap : want;
}

/// A co-read may only name a slot whose session exists.  BCO lines arrive - and may outlive an idle shrink - before
/// the BGEN that grows for them, so the invariant is `slot < reserved`, never `slot < cap`.
inline bool co_slot_ok(int slot, int reserved) { return slot >= 0 && slot < reserved; }

/// Slots the engine currently has a session for: with adaptive slots only the reserved ones, otherwise the
/// --batch cap.  A BYIELD may only park a partial prompt into such a slot (slot_reserved can be below the cap).
inline bool slot_has_session(int slot, int reserved, bool adaptive, int cap) {
    if (slot < 0) return false;
    return slot < (adaptive ? reserved : cap);
}

}  // namespace strata::program::adaptive_slots
