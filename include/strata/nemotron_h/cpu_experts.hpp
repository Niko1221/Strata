// include/strata/nemotron_h/cpu_experts.hpp - Nemotron-H's routed experts on the CPU (the hybrid mode's misses).
//
// A Nemotron-H expert is down(relu(up . x)^2): two GGUF matrices, no gate.  Each row is one ggml-cpu `vec_dot`
// against the activation quantized with the weight type's `vec_dot_type` - what llama.cpp's CPU backend computes
// for the same tensors (its -cmoe layout).  The weights are read where they are: the mapped GGUF's expert slices.
#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>

namespace strata::nemotron_h {

/// One layer's expert geometry on the CPU.
struct CpuFmt {
    int up_type = -1, d_type = -1;   ///< ggml types of up and of down
    int up_act = -1, d_act = -1;     ///< their vec_dot_type
    int64_t n_embd = 0, n_ff = 0;    ///< the expert's input (the MoE latent) and hidden widths
    size_t up_row = 0, d_row = 0;    ///< bytes per weight row
    size_t act_bytes = 0, h_bytes = 0;
};
/// false (with a reason) when ggml-cpu has no dot product for a type or the shapes are not whole blocks.
bool cpu_fmt(int up_type, int d_type, int64_t n_embd, int64_t n_ff, CpuFmt& f, std::string& err);

/// One routed entry: token `tok`'s row of x through the expert at (up, down) into `out` (n_embd floats).
struct CpuJob {
    const uint8_t* up = nullptr;
    const uint8_t* down = nullptr;
    int tok = 0;
    float* out = nullptr;
};

/// A small pool: the calling thread and `threads - 1` workers split each phase's rows.
class CpuExperts {
public:
    explicit CpuExperts(int threads);
    ~CpuExperts();
    CpuExperts(const CpuExperts&) = delete;
    CpuExperts& operator=(const CpuExperts&) = delete;
    /// x: [T][n_embd] floats (only the jobs' tokens are read); n jobs, up to `cap` per call (grows as needed).
    void run(const CpuFmt& f, const float* x, int T, const CpuJob* jobs, int n);
    int threads() const;

private:
    struct Impl;
    std::unique_ptr<Impl> p_;
};

}  // namespace strata::nemotron_h
