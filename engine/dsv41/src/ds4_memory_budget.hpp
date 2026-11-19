#pragma once
// src/dsv41/ds4_memory_budget.hpp - Strictly bounded memory budgeting for 32GB host RAM and 16GB VRAM
#include <cstdint>
#include <string>

namespace strata::dsv41 {

struct MemoryBudgetConfig {
    uint64_t max_host_budget_bytes   = 24ULL * 1024 * 1024 * 1024; // 24 GiB host budget cap
    uint64_t min_os_headroom_bytes   = 4ULL * 1024 * 1024 * 1024;  // 4 GiB minimum free RAM
    uint64_t max_vram_budget_bytes   = 14ULL * 1024 * 1024 * 1024; // 14 GiB out of 16 GiB VRAM
    uint64_t pinned_staging_bytes    = 256ULL * 1024 * 1024;       // 256 MiB pinned staging
    uint64_t engram_cache_bytes      = 128ULL * 1024 * 1024;       // 128 MiB initial Engram cache
};

struct MemoryBudgetAudit {
    bool within_limits              = false;
    uint64_t planned_host_bytes      = 0;
    uint64_t planned_vram_bytes      = 0;
    uint64_t estimated_headroom_bytes= 0;
    std::string rejection_reason    = "";
};

inline MemoryBudgetAudit audit_memory_plan(
    uint64_t dense_bytes,
    uint64_t expert_resident_bytes,
    uint64_t scratch_bytes,
    uint64_t available_physical_ram,
    const MemoryBudgetConfig& cfg = MemoryBudgetConfig{}
) {
    MemoryBudgetAudit audit;
    audit.planned_host_bytes = dense_bytes + expert_resident_bytes + scratch_bytes +
                               cfg.pinned_staging_bytes + cfg.engram_cache_bytes;

    if (audit.planned_host_bytes > cfg.max_host_budget_bytes) {
        audit.within_limits = false;
        audit.rejection_reason = "Host memory plan exceeds 24 GiB budget cap";
        return audit;
    }

    if (available_physical_ram < audit.planned_host_bytes + cfg.min_os_headroom_bytes) {
        audit.within_limits = false;
        audit.rejection_reason = "Insufficient available physical RAM to maintain 4 GiB OS headroom";
        return audit;
    }

    audit.within_limits = true;
    audit.estimated_headroom_bytes = available_physical_ram - audit.planned_host_bytes;
    return audit;
}

} // namespace strata::dsv41
