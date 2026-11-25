// engine/src/inkling2/cuda/ink_vit_parity.cu - check the CUDA picture reader (ink_cuda_vision.cuh) against the CPU
// reference (CpuVisionEncoder, ink_vision.h) on real pictures: per-token cosine, max |diff|, time and VRAM.
//
//   ink_vit_parity VISION.strata PICTURE [PICTURE ...] [--device N] [--no-cpu] [--dump PREFIX]
//
// --dump PREFIX writes each picture's GPU embeddings as raw FP32 to PREFIX.<k>.f32 (to compare two builds or modes with cmp).
//
// Exit code 0 when every picture's worst per-token cosine is >= 0.9999 (FP32) and the GPU encode succeeded.
// Not built by default: cmake --build <dir> --target ink_vit_parity
//
// MIT License - Copyright (c) 2026 Strata contributors
#include "ink_cuda_vision.cuh"

#include <chrono>
#include <cstdio>
#include <string>
#include <vector>

using namespace ink2;

static double ms_since(std::chrono::steady_clock::time_point t0) {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
}

static size_t free_vram(int dev) {
    size_t f = 0, t = 0;
    cudaSetDevice(dev);
    cudaMemGetInfo(&f, &t);
    return f;
}

int main(int argc, char** argv) {
    std::setvbuf(stdout, nullptr, _IOLBF, 0);
    std::string vision;
    std::vector<std::string> pics;
    int dev = 0;
    bool run_cpu = true;
    std::string dump;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--device" && i + 1 < argc) dev = std::atoi(argv[++i]);
        else if (a == "--no-cpu") run_cpu = false;
        else if (a == "--dump" && i + 1 < argc) dump = argv[++i];
        else if (vision.empty()) vision = a;
        else pics.push_back(a);
    }
    if (vision.empty() || pics.empty()) {
        std::fprintf(stderr, "usage: ink_vit_parity VISION.strata PICTURE [PICTURE ...] [--device N] [--no-cpu] [--dump PREFIX]\n");
        return 2;
    }
    std::string err;
    VisionShardInfo vi;
    if (!vision_probe(vision, vi, err)) { std::fprintf(stderr, "vision shard: %s\n", err.c_str()); return 2; }

    const size_t f0 = free_vram(dev);
    cuvit::CudaVit gpu;
    auto t0 = std::chrono::steady_clock::now();
    if (!gpu.load(vision, dev, err)) { std::fprintf(stderr, "GPU load failed: %s\n", err.c_str()); return 1; }
    const double load_ms = ms_since(t0);
    const size_t f1 = free_vram(dev);
    std::printf("gpu load: %.0f ms, weights hold %.0f MiB VRAM\n", load_ms, (double)(f0 - f1) / 1048576.0);

    CpuVisionEncoder cpu;
    if (run_cpu && !cpu.load(vision, err)) { std::fprintf(stderr, "CPU load failed: %s\n", err.c_str()); return 1; }

    bool all_ok = true;
    int pic_no = 0;
    for (const auto& path : pics) {
        const int k_pic = pic_no++;
        ImageRec rec;
        if (!prepare_picture(path, vi.c, rec, err)) { std::fprintf(stderr, "%s: %s\n", path.c_str(), err.c_str()); return 1; }
        std::printf("%s: %dx%d -> grid %dx%d, %d tokens, %d patches\n", path.c_str(), rec.orig_w, rec.orig_h,
                    rec.grid_h, rec.grid_w, rec.num_lm_tokens(), rec.num_lm_tokens() * 4);
        std::vector<float> g, g2, c;
        t0 = std::chrono::steady_clock::now();
        if (!gpu.encode(rec, g, err)) { std::printf("  GPU encode FAILED: %s\n", err.c_str()); all_ok = false; continue; }
        const double g_ms = ms_since(t0);
        t0 = std::chrono::steady_clock::now();
        if (!gpu.encode(rec, g2, err)) { std::printf("  GPU encode (repeat) FAILED: %s\n", err.c_str()); all_ok = false; continue; }
        const double g2_ms = ms_since(t0);
        bool repeat_same = g.size() == g2.size();
        for (size_t i = 0; repeat_same && i < g.size(); ++i) repeat_same = (g[i] == g2[i]);
        std::printf("  GPU: %.1f ms first, %.1f ms repeat, repeat bit-identical: %s\n", g_ms, g2_ms, repeat_same ? "yes" : "no");
        if (!dump.empty()) {
            const std::string fn = dump + "." + std::to_string(k_pic) + ".f32";
            if (FILE* fo = std::fopen(fn.c_str(), "wb")) { std::fwrite(g.data(), 4, g.size(), fo); std::fclose(fo); }
        }
        if (!run_cpu) continue;
        t0 = std::chrono::steady_clock::now();
        if (!cpu.encode(rec, c, err)) { std::printf("  CPU encode FAILED: %s\n", err.c_str()); all_ok = false; continue; }
        const double c_ms = ms_since(t0);
        if (c.size() != g.size()) { std::printf("  size mismatch %zu vs %zu\n", g.size(), c.size()); all_ok = false; continue; }
        const int n_tok = rec.num_lm_tokens();
        const int dim = static_cast<int>(g.size() / std::max(1, n_tok));
        double worst = 1.0, mean = 0.0, max_abs = 0.0, worst_rel_l2 = 0.0;
        for (int t = 0; t < n_tok; ++t) {
            double dot = 0, ng = 0, nc = 0, dd = 0;
            for (int d = 0; d < dim; ++d) {
                const double a = g[(size_t)t * dim + d], b = c[(size_t)t * dim + d];
                dot += a * b; ng += a * a; nc += b * b; dd += (a - b) * (a - b);
                max_abs = std::max(max_abs, std::fabs(a - b));
            }
            const double cs = dot / (std::sqrt(ng * nc) + 1e-30);
            worst = std::min(worst, cs);
            mean += cs;
            worst_rel_l2 = std::max(worst_rel_l2, std::sqrt(dd / (nc + 1e-30)));
        }
        mean /= std::max(1, n_tok);
        std::printf("  CPU: %.1f ms | per-token cosine mean %.7f worst %.7f | worst rel L2 %.2e | max |diff| %.2e\n",
                    c_ms, mean, worst, worst_rel_l2, max_abs);
        if (worst < 0.9999) all_ok = false;
    }
    std::printf("%s\n", all_ok ? "PARITY OK" : "PARITY FAILED");
    return all_ok ? 0 : 1;
}
