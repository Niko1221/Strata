// src/kernels/cpu/nvfp4_modelopt_pool.cpp - Runtime ISA dispatcher and persistent physical-core thread pool for NVFP4 CPU experts.
#include "strata/kernels/cpu/nvfp4_modelopt.hpp"
#include "strata/artifact/safetensors_reader.hpp"
#include "strata/kernels/cpu/pool.hpp"

#include <immintrin.h>

#include <algorithm>
#include <cstdlib>

namespace strata::kernels::cpu {
namespace {

bool g_cpu_isa_inited = false;
bool g_cpu_has_avx512vnni = false;

void init_cpu_isa() {
    if (g_cpu_isa_inited) return;
#if defined(__x86_64__) && (defined(__GNUC__) || defined(__clang__))
    g_cpu_has_avx512vnni = __builtin_cpu_supports("avx512f") &&
                           __builtin_cpu_supports("avx512bw") &&
                           __builtin_cpu_supports("avx512vl") &&
                           __builtin_cpu_supports("avx512dq") &&
                           __builtin_cpu_supports("avx512vnni") &&
                           __builtin_cpu_supports("avx512bf16");
    if (const char* force_avx2 = std::getenv("STRATA_FORCE_AVX2")) {
        if (std::atoi(force_avx2) != 0) g_cpu_has_avx512vnni = false;
    }
#endif
    g_cpu_isa_inited = true;
}

void dispatch_gu_tile32(CpuExpertJob& jb, int r0) {
    if (g_cpu_has_avx512vnni) {
        gu_tile32_modelopt_avx512(jb, r0);
    } else {
        gu_tile32_modelopt_avx2(jb, r0);
    }
}

void dispatch_down_tile80(CpuExpertJob& jb, int r0) {
    if (g_cpu_has_avx512vnni) {
        down_tile80_modelopt_avx512(jb, r0);
    } else {
        down_tile80_modelopt_avx2(jb, r0);
    }
}

void dispatch_down_tile80_decode(const CpuExpertJob& jb, int r0) {
    if (g_cpu_has_avx512vnni) {
        down_tile80_decode_modelopt_avx512(jb, r0);
    } else {
        down_tile80_decode_modelopt_avx2(jb, r0);
    }
}

void dispatch_gu_tile32_fp8(CpuExpertJob& jb, int r0) {
    if (g_cpu_has_avx512vnni) {
        gu_tile32_fp8_avx512(jb, r0);
    } else {
        gu_tile32_fp8_avx2(jb, r0);
    }
}

void dispatch_down_tile80_decode_fp8(const CpuExpertJob& jb, int r0) {
    if (g_cpu_has_avx512vnni) {
        down_tile80_decode_fp8_avx512(jb, r0);
    } else {
        down_tile80_decode_fp8_avx2(jb, r0);
    }
}

}  // namespace

void quantize_act_bf16_2560(const uint16_t* x_bf16, int8_t* q_x, float* d_x, int32_t* neg_bias_x) {
    if (g_cpu_has_avx512vnni) {
        quantize_act_bf16_2560_avx512(x_bf16, q_x, d_x, neg_bias_x);
    } else {
        quantize_act_bf16_2560_avx2(x_bf16, q_x, d_x, neg_bias_x);
    }
}

void Nvfp4CpuPool::init() {
    if (inited_) return;
    init_cpu_isa();
    cores_ = strata::kernels::cpu::physical_cores(false);
    int n = static_cast<int>(cores_.size());
    if (n <= 0) {
        n = static_cast<int>(std::max(1u, std::thread::hardware_concurrency() / 2));
    }
    if (const char* env_t = std::getenv("STRATA_NVFP4_THREADS")) {
        int req = std::atoi(env_t);
        if (req >= 1) n = req;
    }
    n_threads_ = n;
    workers_.reserve(n_threads_ - 1);
    for (int t = 1; t < n_threads_; ++t) {
        int cpu = (t < static_cast<int>(cores_.size())) ? cores_[t] : -1;
        workers_.emplace_back([this, cpu]() { worker_loop(cpu); });
    }
    inited_ = true;
}

void Nvfp4CpuPool::ensure_host_pinned() {
    thread_local bool tls_pinned = false;
    if (!tls_pinned && !cores_.empty() && cores_[0] >= 0) {
        strata::kernels::cpu::pin_current_thread(cores_[0]);
        tls_pinned = true;
    }
}

void Nvfp4CpuPool::stop() {
    if (!inited_) return;
    stop_.store(true, std::memory_order_release);
    gen_.fetch_add(1, std::memory_order_acq_rel);
    for (auto& w : workers_) {
        if (w.joinable()) w.join();
    }
    workers_.clear();
    inited_ = false;
}

void Nvfp4CpuPool::run_batch(CpuExpertJob* jobs, int n_jobs) {
    if (n_jobs <= 0) return;
    jobs_ = jobs;
    n_jobs_ = n_jobs;

    // Phase 1: 20 tiles of 32 gate+up rows per expert
    launch_phase(1, n_jobs, 20);

    // Phase 2: 32 tiles of 80 down rows per expert
    launch_phase(2, n_jobs, 32);
}

void Nvfp4CpuPool::run_decode_batch(CpuExpertJob* jobs, int n_jobs, float* y_accum_f32) {
    if (n_jobs <= 0) return;
    jobs_ = jobs;
    n_jobs_ = n_jobs;

    // Phase 1: 20 tiles of 32 gate+up rows per expert
    launch_phase(1, n_jobs, 20);

    // Phase 4: 32 tiles of 80 down rows per expert (writing pre-weighted float32 into jb.y_f32)
    launch_phase(4, n_jobs, 32);

    if (n_jobs >= 1) {
        if (y_accum_f32 != jobs[0].y_f32) {
            std::copy_n(jobs[0].y_f32, 2560, y_accum_f32);
        }
        for (int j = 1; j < n_jobs; ++j) {
            const float* src = jobs[j].y_f32;
#pragma GCC ivdep
            for (int c = 0; c < 2560; ++c) {
                y_accum_f32[c] += src[c];
            }
        }
    }
}

void Nvfp4CpuPool::run_mtp_decode_batch(CpuExpertJob* jobs, int n_jobs, float* y_accum_f32) {
    if (n_jobs <= 0) return;
    jobs_ = jobs;
    n_jobs_ = n_jobs;

    // Phase 6: 20 tiles of 32 FP8 gate+up rows per expert
    launch_phase(6, n_jobs, 20);

    // Phase 7: 32 tiles of 80 FP8 down rows per expert (writing pre-weighted float32 into jb.y_f32)
    launch_phase(7, n_jobs, 32);

    if (n_jobs >= 1) {
        if (y_accum_f32 != jobs[0].y_f32) {
            std::copy_n(jobs[0].y_f32, 2560, y_accum_f32);
        }
        for (int j = 1; j < n_jobs; ++j) {
            const float* src = jobs[j].y_f32;
#pragma GCC ivdep
            for (int c = 0; c < 2560; ++c) {
                y_accum_f32[c] += src[c];
            }
        }
    }
}

void Nvfp4CpuPool::run_ple_reads(int fd, const uint64_t* offsets, uint8_t* const* dsts, int n_reads) {
    if (n_reads <= 0) return;
    ple_fd_ = fd;
    ple_offsets_ = offsets;
    ple_dsts_ = dsts;
    launch_phase(5, n_reads, 1);
}

void Nvfp4CpuPool::run_quantize_batch(const uint16_t* h_mixed, const uint8_t* tok_needed,
                                      int8_t* act_q, float* act_d, int32_t* act_nb, int n_toks) {
    if (n_toks <= 0) return;
    if (n_toks <= 32 || n_threads_ <= 1) {
        for (int t = 0; t < n_toks; ++t) {
            if (tok_needed[t]) {
                quantize_act_bf16_2560(h_mixed + static_cast<size_t>(t) * 2560,
                                       act_q + static_cast<size_t>(t) * 2560,
                                       act_d + static_cast<size_t>(t) * 160,
                                       act_nb + static_cast<size_t>(t) * 640);
            }
        }
        return;
    }
    q_h_mixed_ = h_mixed;
    q_tok_needed_ = tok_needed;
    q_act_q_ = act_q;
    q_act_d_ = act_d;
    q_act_nb_ = act_nb;
    q_n_toks_ = n_toks;
    const int n_chunks = (n_toks + 3) / 4;
    launch_phase(8, n_chunks, 1);
}

void Nvfp4CpuPool::launch_phase(int ph, int n_jobs, int tiles_per_job) {
    const int total = n_jobs * tiles_per_job;
    n_tasks_ = total;
    phase_ = ph;
    next_task_.store(0, std::memory_order_relaxed);
    done_cnt_.store(0, std::memory_order_relaxed);
    gen_.fetch_add(1, std::memory_order_acq_rel);
    do_work();
    while (done_cnt_.load(std::memory_order_acquire) < n_threads_ - 1) {
        _mm_pause();
    }
}

void Nvfp4CpuPool::do_work() {
    const int total = n_tasks_;
    const int ph = phase_;
    if (ph == 5) {
        const int fd = ple_fd_;
        const uint64_t* offs = ple_offsets_;
        uint8_t* const* dsts = ple_dsts_;
        while (true) {
            int idx = next_task_.fetch_add(1, std::memory_order_relaxed);
            if (idx >= total) break;
            strata::artifact::pread_exact(fd, dsts[idx], 160, offs[idx]);
        }
        return;
    }
    if (ph == 8) {
        const uint16_t* h_mixed = q_h_mixed_;
        const uint8_t* tok_needed = q_tok_needed_;
        int8_t* act_q = q_act_q_;
        float* act_d = q_act_d_;
        int32_t* act_nb = q_act_nb_;
        const int n_toks = q_n_toks_;
        while (true) {
            int idx = next_task_.fetch_add(1, std::memory_order_relaxed);
            if (idx >= total) break;
            const int t0 = idx * 4;
            const int t1 = std::min(n_toks, t0 + 4);
            for (int t = t0; t < t1; ++t) {
                if (tok_needed[t]) {
                    quantize_act_bf16_2560(h_mixed + static_cast<size_t>(t) * 2560,
                                           act_q + static_cast<size_t>(t) * 2560,
                                           act_d + static_cast<size_t>(t) * 160,
                                           act_nb + static_cast<size_t>(t) * 640);
                }
            }
        }
        return;
    }
    CpuExpertJob* jobs = jobs_;
    while (true) {
        int idx = next_task_.fetch_add(1, std::memory_order_relaxed);
        if (idx >= total) break;
        if (ph == 1) {
            int j = idx / 20;
            int r0 = (idx % 20) * 32;
            dispatch_gu_tile32(jobs[j], r0);
        } else if (ph == 2) {
            int j = idx / 32;
            int r0 = (idx % 32) * 80;
            dispatch_down_tile80(jobs[j], r0);
        } else if (ph == 6) {
            int j = idx / 20;
            int r0 = (idx % 20) * 32;
            dispatch_gu_tile32_fp8(jobs[j], r0);
        } else if (ph == 7) {
            int j = idx / 32;
            int r0 = (idx % 32) * 80;
            dispatch_down_tile80_decode_fp8(jobs[j], r0);
        } else {
            int j = idx / 32;
            int r0 = (idx % 32) * 80;
            dispatch_down_tile80_decode(jobs[j], r0);
        }
    }
}

void Nvfp4CpuPool::worker_loop(int cpu) {
    if (cpu >= 0) {
        strata::kernels::cpu::pin_current_thread(cpu);
    }
    uint32_t my_gen = 0;
    while (true) {
        uint32_t g = gen_.load(std::memory_order_acquire);
        if (g == my_gen) {
            _mm_pause();
            continue;
        }
        if (stop_.load(std::memory_order_acquire)) break;
        my_gen = g;
        do_work();
        done_cnt_.fetch_add(1, std::memory_order_release);
    }
}

}  // namespace strata::kernels::cpu
