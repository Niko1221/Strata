#pragma once

#include <cstdint>
#include <cstdlib>
#include <optional>

#include "strata/core/conversation_memory.hpp"

namespace strata::core {

/// The 0.2.7 residency policy for the PLE (n-gram) table: keep the whole table in RAM, or leave it on direct reads?
///
/// The table is not the only thing that wants RAM.  The engine copies every routed expert into a pinned host arena at
/// start (about 51 GiB for IQ3_S), after the table has been opened, and an unbuffered read into that arena needs the
/// pages locked.  A locked table takes its share of the lockable memory first, so on a 64 GB PC (Windows, RTX 5070,
/// 2026-10-26) the policy that only looked at "table + 8 GiB available" admitted the table and the expert load then
/// failed with ERROR_WORKING_SET_QUOTA (1453).  The table is therefore admitted only when RAM holds the table AND the
/// arena the engine will still fill AND the headroom, all measured against what is available BEFORE either is
/// allocated.  Unknown memory telemetry, an unknown table size or an unknown arena size admit nothing (the direct
/// reads of 0.2.6 stay).
///
/// `experts_ram_bytes`: the native expert layout's total (every layer's blob x experts), 0 when it is not known.
inline bool ple_resident_admit(std::optional<uint64_t> available, uint64_t table_bytes, uint64_t experts_ram_bytes,
                               uint64_t headroom_bytes) {
    if (!available || table_bytes == 0 || experts_ram_bytes == 0) return false;
    if (table_bytes > UINT64_MAX - experts_ram_bytes) return false;
    return conversation_memory_admit(available, table_bytes + experts_ram_bytes, headroom_bytes);
}

/// May the residency policy run at all?  STRATA_PLE_AUTO_RAM when it is set (0 off, anything else on: an explicit value wins), otherwise on
/// except under STRATA_DEFAULTS=026 (`v026`: v0.2.6 had no such policy, the direct reads stayed).
inline bool ple_auto_ram_allowed(const char* env_value, bool v026) {
    if (env_value != nullptr) return std::atoi(env_value) != 0;
    return !v026;
}

} // namespace strata::core
