#pragma once

#include "strata/hy4flash_gguf/cpu/native_expert.hpp"
#include "strata/hy4flash_gguf/window.hpp"
#include <cstdint>
#include <memory>
#include <string>

namespace strata::hy4_flash {

/// Persistent CPU expert pool. GPU-compatible mode uses the GPU activation quantization.
class CpuExperts {
public:
    explicit CpuExperts(int workers = 8);
    ~CpuExperts();
    CpuExperts(const CpuExperts&) = delete;
    CpuExperts& operator=(const CpuExperts&) = delete;

    static bool make_format(int gu_type, int down_type, int64_t dim, int64_t ff, float clamp,
                            cpu::NativeFmt& out, std::string& err, bool gpu_compatible = false);
    /// Plane pointers are indexed by job. ranks[j] selects the destination for job j.
    bool compute(const cpu::NativeFmt& layer,
                 const uint8_t* const gates[16], const uint8_t* const ups[16],
                 const uint8_t* const downs[16], const int* ranks, int count,
                 const float* input, float* parts, std::string& err);
    // One weight triple per distinct CPU expert. Each routed token owns a separate output.
    struct WindowJob {
        const uint8_t* plane[3]={};
        int entries[kWindowTokens]={};
        int count=0;
    };
    bool compute_window(const cpu::NativeFmt& layer, const WindowJob* jobs,
                        int count, int tokens, const float* input, float* parts, std::string& err);
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace strata::hy4_flash
