#include "strata/hy4flash_gguf/cpu_experts.hpp"
#include "strata/hy4flash_gguf/cpu/pool.hpp"

#ifdef STRATA_Q81_ALIGNED
#include "cpu/q81_aligned.hpp"
#include "strata/kernels/cpu/expert_layout.hpp"
#include <cstdlib>
#endif

#include <array>
#include <cmath>
#include <mutex>
#include <stdexcept>

namespace strata::hy4_flash {


struct CpuExperts::Impl {
    cpu::ExpertPool pool;
    std::mutex mutex;
    alignas(64) std::array<uint8_t, cpu::kNativeActBytes> activation{};
    std::array<cpu::NativeFmt, 16> formats{};
    std::array<cpu::ExpertJobMulti, 16> jobs{};
    alignas(64) uint8_t window_activations[kWindowTokens][cpu::kNativeActBytes]{};
    std::array<cpu::NativeFmt, kWindowEntries> window_formats{};
    std::array<cpu::ExpertJobMulti, kWindowEntries> window_jobs{};
    explicit Impl(int workers) : pool(workers > 0 ? workers : 8, true, true) {
        // The adaptive split uses the first CPU job to estimate execution cost.
        pool.prepare_split_multi_native();
    }
};

CpuExperts::CpuExperts(int workers) {
    if (!cpu::native_experts_available())
        throw std::runtime_error("hy4-flash CPU experts require native CPU kernels");
    impl_ = std::make_unique<Impl>(workers);
}
CpuExperts::~CpuExperts() = default;

static bool make_base_format(int gu_type, int down_type, int64_t dim, int64_t ff, float clamp,
                             cpu::NativeFmt& out, std::string& err, bool gpu_compatible) {
    // Native Q4_K/Q6_K trunk rows, plus the Q8_0 rows the MTP draft block is stored in. Do not
    // silently select a different weight format. Q8_0 has no Q8_1-activation CPU dot (native_q81.cpp
    // takes Q4_K/Q6_K), so it is the native-activation path only and says so rather than falling back.
    const auto supported = [](int t) { return t == 12 || t == 14 || t == 8; };
    if (!supported(gu_type) || !supported(down_type)) {
        err = "hy4-flash CPU experts support Q4_K/Q6_K/Q8_0 planes only"; return false;
    }
    if (gpu_compatible && (gu_type == 8 || down_type == 8)) {
        err = "hy4-flash CPU experts: Q8_0 planes have no Q8_1-compatible dot product"; return false;
    }
    if (dim <= 0 || ff <= 0 || ff > cpu::kNativeFfMax ||
        !std::isfinite(clamp) || clamp < 0.f) {
        err = "hy4-flash CPU experts: invalid geometry or clamp"; return false;
    }
    cpu::NativeFmt f{};
    if (!cpu::native_fmt(gu_type, down_type, dim, ff, f, err)) return false;
    f.swiglu_clamp = clamp;
    if (gpu_compatible) {
        f.gpu_q8_1 = true;
        f.gu_act = f.d_act = 9; // GGML_TYPE_Q8_1, handled by the explicit CPU compatible branch.
        f.act_bytes = (size_t) dim / 32 * 36;
        f.h_bytes = (size_t) ff / 32 * 36;
    }
    out = f;
    return true;
}

bool CpuExperts::make_format(int gu_type, int down_type, int64_t dim, int64_t ff, float clamp,
                             cpu::NativeFmt& out, std::string& err, bool gpu_compatible) {
    if (!make_base_format(gu_type, down_type, dim, ff, clamp, out, err, gpu_compatible)) return false;
#ifdef STRATA_Q81_ALIGNED
    const char* option = std::getenv("STRATA_HY4_FLASH_CPU_Q81_ALIGNED");
    const char* reference = std::getenv("STRATA_HY4_FLASH_REFERENCE_OPS_LIBRARY");
    const bool requested = option ? std::atoi(option) != 0 : reference && *reference;
    if (requested && gpu_compatible && dim == 1536 && ff == 1536 && gu_type == 12 &&
        (down_type == 12 || down_type == 14) && clamp == 10.f &&
        strata::kernels::cpu::cpu_avx512_ok() && strata::kernels::cpu::cpu_avx2_ok() &&
        __builtin_cpu_supports("avx512dq")) {
        try {
            out.q81_aligned_table = cpu::q81_aligned::device_tables();
        } catch (const std::exception& ex) {
            err = ex.what();
            return false;
        }
    }
#endif
    return true;
}

static bool validate_layer(const cpu::NativeFmt& layer, cpu::NativeFmt& f, std::string& err) {
    if (!make_base_format(layer.gu_type, layer.d_type, layer.n_embd, layer.n_ff,
                     layer.swiglu_clamp, f, err, layer.gpu_q8_1)) return false;
    f.q81_aligned_table = layer.q81_aligned_table;
    // The adapter accepts the unrotated native hy4-flash layout, not a hidden Hadamard/scaled variant.
    if (layer.had_gu || layer.had_d || layer.tail_off != 0) {
        err = "hy4-flash CPU experts: rotated/scaled layouts are unsupported"; return false;
    }
    return true;
}

bool CpuExperts::compute(const cpu::NativeFmt& layer,
                         const uint8_t* const gates[16], const uint8_t* const ups[16],
                         const uint8_t* const downs[16], const int* ranks, int count,
                         const float* input, float* parts, std::string& err) {
    if (count < 0 || count > 16) { err = "hy4-flash CPU experts: count must be in [0,16]"; return false; }
    if (count == 0) return true;
    if (!gates || !ups || !downs || !ranks || !input || !parts) {
        err = "hy4-flash CPU experts: null batch argument"; return false;
    }
    cpu::NativeFmt f{};
    if (!validate_layer(layer, f, err)) return false;
    unsigned seen = 0;
    for (int j = 0; j < count; ++j) {
        if (ranks[j] < 0 || ranks[j] >= 16 || (seen & (1u << ranks[j])) ||
            !gates[j] || !ups[j] || !downs[j]) {
            err = "hy4-flash CPU experts: invalid/duplicate rank or null weight plane"; return false;
        }
        seen |= 1u << ranks[j];
    }
    for (int64_t i = 0; i < f.n_embd; ++i)
        if (!std::isfinite(input[i])) { err = "hy4-flash CPU experts: non-finite input"; return false; }
    try {
        std::lock_guard<std::mutex> lock(impl_->mutex);
        cpu::native_quant_act(f, input, impl_->activation.data());
        for (int j = 0; j < count; ++j) {
            auto& fmt = impl_->formats[j];
            fmt = f;
            fmt.gate_data = gates[j]; fmt.up_data = ups[j]; fmt.down_data = downs[j];
            auto& job = impl_->jobs[j];
            job = cpu::ExpertJobMulti{};
            job.blob = gates[j]; job.fmt = &fmt; job.nt = 1;
            job.nact[0] = impl_->activation.data();
            job.out[0] = parts + (size_t) ranks[j] * (size_t) f.n_embd;
        }
        impl_->pool.run_split_multi_native(f, impl_->jobs.data(), count,
                                           cpu::NativeArithmetic::WindowInvariant);
        for (int j = 0; j < count; ++j)
            for (int64_t i = 0; i < f.n_embd; ++i)
                if (!std::isfinite(parts[(size_t) ranks[j] * (size_t) f.n_embd + i])) {
                    err = "hy4-flash CPU experts: non-finite result"; return false;
                }
        return true;
    } catch (const std::exception& e) {
        err = std::string("hy4-flash CPU experts: ") + e.what(); return false;
    }
}

bool CpuExperts::compute_window(const cpu::NativeFmt& layer, const WindowJob* jobs, int count,
                                int tokens, const float* input, float* parts, std::string& err) {
    static_assert(kWindowTokens <= cpu::MAXT);
    if (!jobs || !input || !parts || tokens < 1 || tokens > kWindowTokens ||
        count < 1 || count > tokens * kWindowTopK) {
        err = "invalid CPU expert window";
        return false;
    }
    cpu::NativeFmt f{};
    if (!validate_layer(layer, f, err)) return false;
    std::lock_guard<std::mutex> lock(impl_->mutex);
    bool quantized[kWindowTokens] = {};
    bool assigned[kWindowEntries] = {};
    for (int j = 0; j < count; ++j) {
        const auto& source = jobs[j];
        if (source.count < 1 || source.count > tokens ||
            !source.plane[0] || !source.plane[1] || !source.plane[2]) {
            err = "invalid CPU window expert";
            return false;
        }
        auto& fmt = impl_->window_formats[j];
        fmt = f;
        fmt.gate_data = source.plane[0];
        fmt.up_data = source.plane[1];
        fmt.down_data = source.plane[2];
        auto& job = impl_->window_jobs[j];
        job.fmt = &fmt;
        job.blob = fmt.gate_data;
        job.nt = source.count;
        for (int i = 0; i < source.count; ++i) {
            const int entry = source.entries[i];
            if (entry < 0 || entry >= tokens * kWindowTopK || assigned[entry]) {
                err = "invalid CPU window output rank";
                return false;
            }
            assigned[entry] = true;
            const int t = entry / kWindowTopK;
            if (!quantized[t]) {
                const float* x = input + size_t(t) * f.n_embd;
                for (int d = 0; d < f.n_embd; ++d) {
                    if (!std::isfinite(x[d])) {
                        err = "non-finite CPU window input";
                        return false;
                    }
                }
                cpu::native_quant_act(f, x, impl_->window_activations[t]);
                quantized[t] = true;
            }
            job.nact[i] = impl_->window_activations[t];
            job.out[i] = parts + size_t(entry) * f.n_embd;
        }
    }
    impl_->pool.run_split_multi_native(f, impl_->window_jobs.data(), count,
                                       cpu::NativeArithmetic::WindowInvariant);
    for (int j = 0; j < count; ++j) {
        const auto& job = impl_->window_jobs[j];
        for (int i = 0; i < job.nt; ++i) {
            for (int d = 0; d < f.n_embd; ++d) {
                if (!std::isfinite(job.out[i][d])) {
                    err = "non-finite CPU window output";
                    return false;
                }
            }
        }
    }
    return true;
}
} // namespace strata::hy4_flash
