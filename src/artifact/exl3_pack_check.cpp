// src/artifact/exl3_pack_check.cpp - build the engine WeightTable from an EXL3 model and validate it.
#include "strata/artifact/exl3_pack.hpp"

#include "strata/core/layout.hpp"
#include "strata/kernels/exl3.hpp"

#include <cuda_runtime.h>

#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

namespace {

// The roles `--dump` writes, one file per role, for tools/exl3/check_engine_form.py to compare against
// the converter's transforms.  Staged roles are their bytes; an EXL3 role is trellis|suh|svh.
const char* kDumpRoles[] = {
    "blk.0.ssm_a", "blk.0.ssm_dt.bias", "blk.0.ssm_alpha.weight", "blk.0.ssm_beta.weight",
    "blk.0.ssm_conv1d.weight", "blk.0.hc_attn_norm.weight", "blk.0.hc_ffn_norm.weight",
    "blk.3.attn_q_norm.weight", "blk.3.attn_k_norm.weight", "blk.3.indexer.q_norm.weight",
    "blk.3.indexer.k_norm.weight", "blk.1.ple_norm_key.weight", "blk.1.ple_norm_query.weight",
    "blk.1.ple_norm_conv.weight", "output_hc_norm.weight", "blk.3.indexer.q_proj.weight",
    "blk.3.indexer.k_proj.weight", "blk.0.ffn_gate_inp.weight", "blk.0.ffn_gate_shexp.weight",
    "blk.0.attn_qkv.weight", "blk.0.attn_gate.weight", "blk.0.ssm_out.weight",
};

int dump_roles(const strata::core::WeightTable& wt, const char* dir) {
    std::error_code ec;
    std::filesystem::create_directories(dir, ec);
    for (const char* name : kDumpRoles) {
        const strata::core::WeightRef* w = wt.find(name);
        if (!w) { std::fprintf(stderr, "dump: no role %s\n", name); return 1; }
        std::vector<uint8_t> buf;
        if (w->exl3) {
            const auto* m = (const strata::kernels::Exl3Mat*) w->exl3;
            const size_t trellis = (size_t) m->ki * m->nj * (m->bits * 16) * 2;
            buf.resize(trellis + (size_t) (m->ki * 16 + m->nj * 16) * 2);
            if (cudaMemcpy(buf.data(), m->trellis, trellis, cudaMemcpyDeviceToHost) != cudaSuccess ||
                cudaMemcpy(buf.data() + trellis, m->suh, (size_t) m->ki * 32, cudaMemcpyDeviceToHost) != cudaSuccess ||
                cudaMemcpy(buf.data() + trellis + (size_t) m->ki * 32, m->svh, (size_t) m->nj * 32,
                           cudaMemcpyDeviceToHost) != cudaSuccess) {
                std::fprintf(stderr, "dump: %s: copy failed\n", name);
                return 1;
            }
        } else {
            buf.resize(w->bytes);
            if (cudaMemcpy(buf.data(), w->data, w->bytes, cudaMemcpyDeviceToHost) != cudaSuccess) {
                std::fprintf(stderr, "dump: %s: copy failed\n", name);
                return 1;
            }
        }
        std::ofstream out(std::string(dir) + "/" + name + ".bin", std::ios::binary);
        out.write((const char*) buf.data(), (std::streamsize) buf.size());
        std::printf("%-32s %zu B\n", name, buf.size());
    }
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: %s <exl3_dir> [--run | --dump <dir>]\n", argv[0]);
        return 2;
    }
    const bool dump = argc >= 4 && std::strcmp(argv[2], "--dump") == 0;
    try {
        strata::core::Exl3Pack pack(argv[1]);
        strata::core::WeightTable wt;
        std::string err;
        if (!pack.build(wt, err)) { std::fprintf(stderr, "build failed: %s\n", err.c_str()); return 1; }
        std::printf("roles: %zu   EXL3 linears: %.1f MiB   staged arena: %.1f MiB\n", wt.all().size(),
                    (double) pack.exl3_bytes() / (1 << 20), (double) pack.arena_bytes() / (1 << 20));
        std::printf("check_all: OK\n");
        if (dump) return dump_roles(wt, argv[3]);
        if (argc < 3) return 0;
        // Run every dense EXL3 projection on the real weights, exactly as the engine's gemv_quantized does.
        strata::kernels::exl3_gemv_reserve(8192, 262144);
        int bad = 0, ran = 0;
        for (const auto& kv : wt.all()) {
            const strata::core::WeightRef& r = kv.second;
            if (r.exl3 == nullptr) continue;
            const auto* m = (const strata::kernels::Exl3Mat*) r.exl3;
            const int k = m->ki * 16, n = m->nj * 16;
            float* dx = nullptr; float* dy = nullptr;
            (void) cudaMalloc(&dx, (size_t) k * 4); (void) cudaMalloc(&dy, (size_t) n * 4);
            (void) cudaMemset(dx, 0, (size_t) k * 4);
            strata::kernels::exl3_gemv_f32(dx, m->suh, m->svh, m->trellis, m->ki, m->nj, m->bits, m->cb, dy, nullptr);
            const cudaError_t e = cudaDeviceSynchronize();
            if (e != cudaSuccess) { std::printf("  FAULT %-32s ki=%d nj=%d bits=%d: %s\n", kv.first.c_str(), m->ki, m->nj, m->bits, cudaGetErrorString(e)); ++bad; }
            (void) cudaFree(dx); (void) cudaFree(dy);
            ++ran;
        }
        std::printf("ran %d EXL3 projections, %d faults\n", ran, bad);
        return bad ? 1 : 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
}
