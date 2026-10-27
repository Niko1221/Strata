// include/strata/kernels/cpu/nvfp4_modelopt.hpp - ModelOpt NVFP4 CPU expert kernels (AVX-512 VNNI + AVX2) and thread pool.
#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <thread>
#include <vector>

namespace strata::kernels::cpu {

// Per-expert byte layout inside a layer's 1,415,577,600-byte NVFP4 expert arena
// (keeps each projection's weights and scales adjacent in the same 2 MB hugepage for CPU/GPU TLB locality):
inline constexpr size_t EXP_W_BYTES = 819200;    // [640, 1280] or [2560, 320] U8
inline constexpr size_t EXP_S_BYTES = 102400;    // [640, 160] or [2560, 40] F8_E4M3
inline constexpr size_t EXP_OFF_GATE_W = 0;
inline constexpr size_t EXP_OFF_GATE_S = 819200;
inline constexpr size_t EXP_OFF_UP_W = 921600;
inline constexpr size_t EXP_OFF_UP_S = 1740800;
inline constexpr size_t EXP_OFF_DOWN_W = 1843200;
inline constexpr size_t EXP_OFF_DOWN_S = 2662400;
inline constexpr size_t EXP_STRIDE_BYTES = 2764800;
inline constexpr size_t LAYER_EXPERT_BYTES = 512ULL * EXP_STRIDE_BYTES;   // 1,415,577,600 B

inline constexpr int kMaxJobToks = 64;

struct CpuExpertJob {
    const uint8_t* exp_base = nullptr;
    float gate_s2 = 0.0f;
    float up_s2 = 0.0f;
    float down_s2 = 0.0f;
    float weight = 0.0f;
    int nt = 1;
    const int8_t* q_x[kMaxJobToks]{};
    const float* d_x[kMaxJobToks]{};
    const int32_t* neg_bias_x[kMaxJobToks]{};
    const float* x_f32 = nullptr;
    uint16_t* y_out[kMaxJobToks]{};
    float* y_f32 = nullptr;
    alignas(64) int8_t q_h[kMaxJobToks][640];
    alignas(64) float d_h[kMaxJobToks][40];
    alignas(64) int32_t neg_bias_h[kMaxJobToks][160];
    alignas(64) float h_f32[640];
    alignas(64) float y_buf_f32[2560];
};

// AVX-512 VNNI kernels (defined in src/kernels/cpu/nvfp4_modelopt_avx512.cpp)
void quantize_act_bf16_2560_avx512(const uint16_t* x_bf16, int8_t* q_x, float* d_x, int32_t* neg_bias_x);
void gu_tile32_modelopt_avx512(CpuExpertJob& jb, int r0);
void down_tile80_modelopt_avx512(CpuExpertJob& jb, int r0);
void down_tile80_decode_modelopt_avx512(const CpuExpertJob& jb, int r0);
void gu_tile32_fp8_avx512(CpuExpertJob& jb, int r0);
void down_tile80_decode_fp8_avx512(const CpuExpertJob& jb, int r0);

// AVX2 + FMA fallback kernels (defined in src/kernels/cpu/nvfp4_modelopt_avx2.cpp)
void quantize_act_bf16_2560_avx2(const uint16_t* x_bf16, int8_t* q_x, float* d_x, int32_t* neg_bias_x);
void gu_tile32_modelopt_avx2(CpuExpertJob& jb, int r0);
void down_tile80_modelopt_avx2(CpuExpertJob& jb, int r0);
void down_tile80_decode_modelopt_avx2(const CpuExpertJob& jb, int r0);
void gu_tile32_fp8_avx2(CpuExpertJob& jb, int r0);
void down_tile80_decode_fp8_avx2(const CpuExpertJob& jb, int r0);

// Runtime-dispatched entry points (defined in src/kernels/cpu/nvfp4_modelopt_pool.cpp)
void quantize_act_bf16_2560(const uint16_t* x_bf16, int8_t* q_x, float* d_x, int32_t* neg_bias_x);

class Nvfp4CpuPool {
public:
    Nvfp4CpuPool() = default;
    ~Nvfp4CpuPool() { stop(); }

    Nvfp4CpuPool(const Nvfp4CpuPool&) = delete;
    Nvfp4CpuPool& operator=(const Nvfp4CpuPool&) = delete;

    void init();
    void stop();
    void ensure_host_pinned();
    int core_for_thread(int t) const {
        return (t >= 0 && t < static_cast<int>(cores_.size())) ? cores_[t] : -1;
    }
    void run_batch(CpuExpertJob* jobs, int n_jobs);
    void run_decode_batch(CpuExpertJob* jobs, int n_jobs, float* y_accum_f32);
    void run_mtp_decode_batch(CpuExpertJob* jobs, int n_jobs, float* y_accum_f32);
    void run_ple_reads(int fd, const uint64_t* offsets, uint8_t* const* dsts, int n_reads);
    void run_quantize_batch(const uint16_t* h_mixed, const uint8_t* tok_needed,
                            int8_t* act_q, float* act_d, int32_t* act_nb, int n_toks);

    int num_threads() const { return n_threads_; }

private:
    void launch_phase(int ph, int n_jobs, int tiles_per_job);
    void do_work();
    void worker_loop(int cpu);

    bool inited_ = false;
    int n_threads_ = 1;
    std::vector<int> cores_;
    std::vector<std::thread> workers_;
    alignas(64) std::atomic<uint32_t> gen_{0};
    alignas(64) std::atomic<int> next_task_{0};
    alignas(64) std::atomic<int> done_cnt_{0};
    std::atomic<bool> stop_{false};
    CpuExpertJob* jobs_ = nullptr;
    int ple_fd_ = -1;
    const uint64_t* ple_offsets_ = nullptr;
    uint8_t* const* ple_dsts_ = nullptr;
    const uint16_t* q_h_mixed_ = nullptr;
    const uint8_t* q_tok_needed_ = nullptr;
    int8_t* q_act_q_ = nullptr;
    float* q_act_d_ = nullptr;
    int32_t* q_act_nb_ = nullptr;
    int q_n_toks_ = 0;
    int n_jobs_ = 0;
    int n_tasks_ = 0;
    int phase_ = 0;
};

}  // namespace strata::kernels::cpu
