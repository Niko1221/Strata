// Compare slot-addressed expert batches against contiguous weights for identical routed rows.
#include "strata/hy4flash_gguf/attention.hpp"
#include "strata/hy4flash_gguf/prefill_moe.hpp"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#if defined(__HIP_PLATFORM_AMD__) || defined(STRATA_USE_HIP)
#include <hip/hip_runtime.h>
#else
#include <cuda_runtime.h>
#endif

namespace {

int g_fail = 0;

void check(bool ok, const std::string& what) {
    std::printf("  %-66s %s\n", what.c_str(), ok ? "ok" : "FAIL");
    if (!ok) ++g_fail;
}

/// Q8_0: 32 values a block, one FP16 scale then 32 int8s. Used for all three roles so the test needs no
/// K-quant packer; `moe_mmq_usable` accepts it and the addressing under test is type-independent.
constexpr int kQ8Block = 32;
constexpr std::size_t kQ8BlockBytes = 2 + kQ8Block;

std::size_t q8_row_bytes(int cols) { return (std::size_t) (cols / kQ8Block) * kQ8BlockBytes; }

/// One expert's [rows, cols] matrix as Q8_0, deterministic in `seed` so the two layouts hold identical bits.
void make_q8(std::vector<uint8_t>& out, int rows, int cols, uint32_t seed) {
    out.assign((std::size_t) rows * q8_row_bytes(cols), 0);
    std::mt19937 rng(seed);
    for (int r = 0; r < rows; ++r) {
        uint8_t* row = out.data() + (std::size_t) r * q8_row_bytes(cols);
        for (int b = 0; b < cols / kQ8Block; ++b) {
            uint8_t* blk = row + (std::size_t) b * kQ8BlockBytes;
            // a small positive scale, as an FP16 bit pattern: 0x1C00 is 1/256
            blk[0] = 0x00;
            blk[1] = 0x1C;
            for (int i = 0; i < kQ8Block; ++i) blk[2 + i] = (uint8_t) (int8_t) ((int) (rng() % 61) - 30);
        }
    }
}

bool dev_ok() {
    int n = 0;
    return cudaGetDeviceCount(&n) == cudaSuccess && n > 0;
}

}  // namespace

int main() {
    using namespace strata::hy4_flash;
    if (!dev_ok()) {
        std::printf("prefill_slot_parity: no CUDA device, skipping\n");
        return 0;
    }

    // Small but real shapes: the expert widths are what MMQ tiles over, and 128 is a multiple of its
    // tile requirements while staying a few MiB.
    const int expert_dim = 128, expert_ff = 128;
    const int n_expert = 16, k = 4, n_tok = 64;
    const int gate_type = 8, up_type = 8, down_type = 8;   // GGML_TYPE_Q8_0

    if (!moe_mmq_usable(gate_type, up_type, down_type, expert_dim, expert_ff, n_tok)) {
        std::printf("prefill_slot_parity: this build declines the batched MoE here, skipping\n");
        return 0;
    }

    // ---- the weights, once. Both layouts hold the same bytes for the same expert.
    std::vector<std::vector<uint8_t>> gate(n_expert), up(n_expert), down(n_expert);
    for (int e = 0; e < n_expert; ++e) {
        make_q8(gate[(std::size_t) e], expert_ff, expert_dim, 1000u + (uint32_t) e);
        make_q8(up[(std::size_t) e], expert_ff, expert_dim, 2000u + (uint32_t) e);
        make_q8(down[(std::size_t) e], expert_dim, expert_ff, 3000u + (uint32_t) e);
    }
    const std::size_t gu_stride = gate[0].size();
    const std::size_t d_stride = down[0].size();

    // Compare contiguous expert rows against the same rows in scattered slots.
    const int n_slot = n_expert + 5;
    std::vector<int32_t> perm((std::size_t) n_expert);
    {
        std::vector<int32_t> free_slots((std::size_t) n_slot);
        for (int s = 0; s < n_slot; ++s) free_slots[(std::size_t) s] = s;
        std::mt19937 rng(20261109u);
        std::shuffle(free_slots.begin(), free_slots.end(), rng);
        for (int e = 0; e < n_expert; ++e) perm[(std::size_t) e] = free_slots[(std::size_t) e];
    }

    auto fill_plane = [&](std::vector<uint8_t>& host, const std::vector<std::vector<uint8_t>>& src,
                          std::size_t stride, int slots, const int32_t* at) {
        host.assign((std::size_t) slots * stride, 0);
        for (int e = 0; e < n_expert; ++e) {
            const std::size_t s = at != nullptr ? (std::size_t) at[(std::size_t) e] : (std::size_t) e;
            std::memcpy(host.data() + s * stride, src[(std::size_t) e].data(), stride);
        }
    };
    std::vector<uint8_t> a_g, a_u, a_d, b_g, b_u, b_d;
    fill_plane(a_g, gate, gu_stride, n_expert, nullptr);
    fill_plane(a_u, up, gu_stride, n_expert, nullptr);
    fill_plane(a_d, down, d_stride, n_expert, nullptr);
    fill_plane(b_g, gate, gu_stride, n_slot, perm.data());
    fill_plane(b_u, up, gu_stride, n_slot, perm.data());
    fill_plane(b_d, down, d_stride, n_slot, perm.data());

    // Cover duplicate picks within a token, an unused expert and skewed routing.
    std::vector<int32_t> ids((std::size_t) n_tok * k);
    std::vector<float> rw((std::size_t) n_tok * k);
    {
        std::mt19937 rng(7u);
        for (int t = 0; t < n_tok; ++t)
            for (int j = 0; j < k; ++j) {
                int e;
                if (j == 0) e = 3;                              // every token picks 3: a hot expert
                else if (t % 8 == 0 && j == 1) e = 3;           // and sometimes twice
                else e = (int) (rng() % (unsigned) (n_expert - 1));  // expert 15 is never picked
                ids[(std::size_t) t * k + j] = e;
                rw[(std::size_t) t * k + j] = 0.1f + 0.01f * (float) ((t + j) % 17);
            }
    }

    std::vector<float> routed_in((std::size_t) expert_dim * n_tok);
    {
        std::mt19937 rng(11u);
        std::uniform_real_distribution<float> d(-1.f, 1.f);
        for (float& v : routed_in) v = d(rng);
    }

    // ---- device
    auto up_dev = [](const void* src, std::size_t n) {
        void* p = nullptr;
        if (cudaMalloc(&p, n) != cudaSuccess) return (void*) nullptr;
        if (cudaMemcpy(p, src, n, cudaMemcpyHostToDevice) != cudaSuccess) { cudaFree(p); return (void*) nullptr; }
        return p;
    };
    void* d_ag = up_dev(a_g.data(), a_g.size());
    void* d_au = up_dev(a_u.data(), a_u.size());
    void* d_ad = up_dev(a_d.data(), a_d.size());
    void* d_bg = up_dev(b_g.data(), b_g.size());
    void* d_bu = up_dev(b_u.data(), b_u.size());
    void* d_bd = up_dev(b_d.data(), b_d.size());
    void* d_in = up_dev(routed_in.data(), routed_in.size() * 4);
    void* d_ids = up_dev(ids.data(), ids.size() * 4);
    void* d_rw = up_dev(rw.data(), rw.size() * 4);
    // `slot_of` for layout B; expert 15 is unrouted but still mapped, which must not matter.
    void* d_slot = up_dev(perm.data(), perm.size() * 4);
    float* d_out_a = nullptr;
    float* d_out_b = nullptr;
    void* d_scratch = nullptr;
    const std::size_t scratch_a = latent_moe_mmq_scratch_bytes(expert_dim, expert_ff, n_expert, k, n_tok);
    const std::size_t scratch_b = latent_moe_mmq_scratch_bytes(expert_dim, expert_ff, n_slot, k, n_tok);
    const std::size_t scratch = scratch_a > scratch_b ? scratch_a : scratch_b;
    bool alloc_ok = d_ag && d_au && d_ad && d_bg && d_bu && d_bd && d_in && d_ids && d_rw && d_slot &&
                    scratch != 0 &&
                    cudaMalloc((void**) &d_out_a, (std::size_t) expert_dim * n_tok * 4) == cudaSuccess &&
                    cudaMalloc((void**) &d_out_b, (std::size_t) expert_dim * n_tok * 4) == cudaSuccess &&
                    cudaMalloc(&d_scratch, scratch) == cudaSuccess;
    if (!alloc_ok) {
        std::printf("prefill_slot_parity: cannot allocate, skipping\n");
        return 0;
    }
    cudaStream_t st{};
    cudaStreamCreate(&st);

    ExpertGroup ga{};
    ga.gate = d_ag; ga.gate_stride = gu_stride;
    ga.up = d_au;   ga.up_stride = gu_stride;
    ga.down = d_ad; ga.down_stride = d_stride;
    ExpertGroup gb{};
    gb.gate = d_bg; gb.gate_stride = gu_stride;
    gb.up = d_bu;   gb.up_stride = gu_stride;
    gb.down = d_bd; gb.down_stride = d_stride;

    const float clamp = 10.0f;
    const bool ok_a = latent_moe_experts_mmq(
        (const float*) d_in, ga, 0, gate_type, up_type, down_type, expert_dim, expert_ff, n_expert, k,
        n_tok, (const int32_t*) d_ids, (const float*) d_rw, clamp, d_out_a, d_scratch, scratch, st);
    const bool ok_b = latent_moe_experts_mmq(
        (const float*) d_in, gb, 0, gate_type, up_type, down_type, expert_dim, expert_ff, n_expert, k,
        n_tok, (const int32_t*) d_ids, (const float*) d_rw, clamp, d_out_b, d_scratch, scratch, st,
        (const int32_t*) d_slot, n_slot);
    cudaStreamSynchronize(st);

    std::printf("prefill_slot_parity: %d experts in %d slots, %d tokens x top-%d\n", n_expert, n_slot,
                n_tok, k);
    check(ok_a, "the equidistant layout runs");
    check(ok_b, "the scattered layout with slot_of runs");

    if (ok_a && ok_b) {
        std::vector<float> ha((std::size_t) expert_dim * n_tok), hb(ha.size());
        cudaMemcpy(ha.data(), d_out_a, ha.size() * 4, cudaMemcpyDeviceToHost);
        cudaMemcpy(hb.data(), d_out_b, hb.size() * 4, cudaMemcpyDeviceToHost);
        std::size_t diff = 0;
        double worst = 0.0;
        bool finite = true;
        for (std::size_t i = 0; i < ha.size(); ++i) {
            if (!std::isfinite(ha[i]) || !std::isfinite(hb[i])) finite = false;
            if (ha[i] != hb[i]) {
                ++diff;
                const double d = std::fabs((double) ha[i] - (double) hb[i]);
                if (d > worst) worst = d;
            }
        }
        check(finite, "both outputs are finite");
        // Hold weights, routing and tiles fixed so only slot addressing changes.
        check(diff == 0, "scattered == equidistant, bit for bit (" + std::to_string(diff) + " of " +
                             std::to_string(ha.size()) + " differ, worst " + std::to_string(worst) + ")");
        // And the result is not trivially zero, which would make the comparison vacuous.
        double mag = 0.0;
        for (float v : ha) mag += std::fabs((double) v);
        check(mag > 0.0, "the output is not all zero");
    }

    for (void* p : {d_ag, d_au, d_ad, d_bg, d_bu, d_bd, d_in, d_ids, d_rw, d_slot, d_scratch})
        cudaFree(p);
    cudaFree(d_out_a);
    cudaFree(d_out_b);
    cudaStreamDestroy(st);
    std::printf("%s\n", g_fail ? "FAILED" : "all ok");
    return g_fail ? 1 : 0;
}
