#pragma once
#include <cstddef>
#include <cstdint>

namespace strata::hy4_flash {
// Fixed addresses and capacities: graph topology never depends on routed IDs.
constexpr int kWindowTokens = 8;
constexpr int kWindowTopK = 16;
constexpr int kWindowEntries = kWindowTokens * kWindowTopK;
constexpr int kWindowStage = 32;
constexpr int kWindowBatches = (kWindowEntries + kWindowStage - 1) / kWindowStage;
struct alignas(64) WindowMailbox {
    volatile uint32_t request=0, plan_ready=0, done=0, error=0;
    volatile uint32_t ready[kWindowBatches]={}, consumed[kWindowBatches]={};
    volatile uint32_t cpu_input_ready=0, cpu_ready=0;
    /// Device clock cycles THIS LAYER's kernels spent stalled: [0] on the staged PCIe experts, [1] on the
    /// CPU's rows. A branch whose work overlapped the GPU's own costs nothing here, which is the whole
    /// point - a host-side duration cannot tell an overlapped copy from one the GPU waited out.
    volatile unsigned long long stall[2]={};
    uint32_t cpu_mask[kWindowTokens]={};
    int32_t ids[kWindowEntries]={};
    // Phase zero is resident; subsequent phases own at most 32 distinct misses.
    int32_t index[kWindowBatches+1][kWindowEntries]={};
};
struct WindowExperts {
    WindowMailbox* mailbox=nullptr;
    int32_t* index=nullptr;
    void* gate=nullptr; void* up=nullptr; void* down=nullptr;
    std::size_t gate_stride=0, up_stride=0, down_stride=0;
    float* cpu_input=nullptr;
    const float* cpu_parts=nullptr;
    void* scratch=nullptr;
    int tokens=0;
};

// CPU context for draft experts placed in host RAM.
struct alignas(64) DraftMailbox {
    volatile uint32_t request=0, input_ready=0, done=0, error=0;
    int32_t ids[kWindowTopK]={};
};
struct DraftExperts {
    DraftMailbox* mailbox=nullptr;   ///< device address of the mapped host mailbox
    float* input=nullptr;            ///< [expert_dim] mapped host, written by the publish kernel
    const float* parts=nullptr;      ///< [top_k, expert_dim] mapped host, written by the CPU
};


}
