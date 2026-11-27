// engine/src/inkling2/cuda/ink_vit_check.cu - the GPU picture reader against the CPU reference, on the same pictures
//
// MIT License
// Copyright (c) 2026 Niko1221 and the Strata contributors
//
// Usage: strata-inkling-vit-check --vision Inkling-2-Mini-Vision.strata --image A.png [--image B.jpg ...] [--reps 3] [--cpu-cache DIR] [--skip-cpu]
//
// For every picture: the CPU reference (ink2::CpuVisionEncoder, FP32) and the GPU reader (ink_vision_cuda.cuh) each make the [tokens, 2304] embeddings;
// the tool prints the time of each, the largest absolute and relative error, and the cosine per row (smallest and mean), and the memory the GPU reader
// holds (weights after load, working buffers of the picture, the lowest free memory seen while it ran).  The CPU result is kept in --cpu-cache DIR so a
// second run (another STRATA_INK_VIT_PREC) does not read the picture on the CPU again.
#define INKLING_DEFINE_STBI
#include "ink_vision_cuda.cuh"

#include <atomic>
#include <thread>

using namespace strata::ink;

static double ms_since(std::chrono::steady_clock::time_point t) {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t).count();
}

static std::string base_name(const std::string& p) {
    size_t s = p.find_last_of("/\\");
    return s == std::string::npos ? p : p.substr(s + 1);
}

int main(int argc, char** argv) {
    std::string vision, cache;
    std::vector<std::string> images;
    int reps = 3, gpu = 0;
    bool skip_cpu = false;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&]() -> std::string {
            if (i + 1 >= argc) { std::fprintf(stderr, "missing value after %s\n", a.c_str()); std::exit(2); }
            return argv[++i];
        };
        if (a == "--vision") vision = next();
        else if (a == "--image") images.push_back(next());
        else if (a == "--reps") reps = std::max(1, std::atoi(next().c_str()));
        else if (a == "--gpu") gpu = std::atoi(next().c_str());
        else if (a == "--cpu-cache") cache = next();
        else if (a == "--skip-cpu") skip_cpu = true;
        else { std::fprintf(stderr, "unknown option %s\n", a.c_str()); return 2; }
    }
    if (vision.empty() || images.empty()) {
        std::fprintf(stderr, "usage: %s --vision FILE --image FILE [--image FILE ...] [--reps N] [--gpu N] [--cpu-cache DIR] [--skip-cpu]\n", argv[0]);
        return 2;
    }
    std::string err;
    ink2::VisionShardInfo info;
    if (!ink2::vision_probe(vision, info, err)) { std::fprintf(stderr, "vision: %s\n", err.c_str()); return 1; }
    cudaSetDevice(gpu);
    cudaDeviceProp prop{};
    cudaGetDeviceProperties(&prop, gpu);
    std::printf("device: %s sm_%d%d, %.1f GiB; reader precision: %s\n", prop.name, prop.major, prop.minor, (double)prop.totalGlobalMem / 1073741824.0,
                InkVisionCuda::default_precision(prop.major * 10 + prop.minor) == VitPrec::TC2 ? "tc2 (BF16 tensor cores, two-part activation)" :
                (InkVisionCuda::default_precision(prop.major * 10 + prop.minor) == VitPrec::BF16 ? "bf16 (BF16 tensor cores)" : "fp32"));
    cudaFree(nullptr);
    size_t free0 = 0, tot = 0;
    cudaMemGetInfo(&free0, &tot);

    InkVisionCuda gv;
    ink2::CpuVisionEncoder cv;
    const auto tl = std::chrono::steady_clock::now();
    if (!gv.load(vision, gpu, err)) { std::fprintf(stderr, "GPU reader load failed: %s\n", err.c_str()); return 1; }
    const double load_ms = ms_since(tl);
    size_t free1 = 0;
    cudaMemGetInfo(&free1, &tot);
    std::printf("GPU reader loaded in %.0f ms: weights %.1f MiB, card memory used by the load %.1f MiB (weights + the cuBLAS handle)\n", load_ms,
                (double)gv.weight_bytes() / 1048576.0, (double)(free0 - free1) / 1048576.0);

    int bad = 0;
    for (const std::string& f : images) {
        ink2::ImageRec rec;
        if (!ink2::prepare_picture(f, info.c, rec, err)) { std::fprintf(stderr, "%s: %s\n", f.c_str(), err.c_str()); return 1; }
        const int n_tok = rec.num_lm_tokens(), D = info.c.out_hidden;
        const int S = rec.grid_h * rec.grid_w * info.c.merge * info.c.merge + 1;
        std::vector<float> ref;
        double cpu_ms = -1.0;
        const std::string cache_file = cache.empty() ? "" : cache + "/" + base_name(f) + ".cpu.f32";
        bool have_ref = false;
        if (!cache_file.empty()) {
            if (FILE* fp = std::fopen(cache_file.c_str(), "rb")) {
                ref.resize((size_t)n_tok * D);
                have_ref = std::fread(ref.data(), 4, ref.size(), fp) == ref.size();
                std::fclose(fp);
                if (have_ref) std::printf("  (CPU reference read from %s)\n", cache_file.c_str());
            }
        }
        if (!have_ref && !skip_cpu) {
            const auto t0 = std::chrono::steady_clock::now();
            if (!cv.loaded && !cv.load(vision, err)) { std::fprintf(stderr, "CPU load failed: %s\n", err.c_str()); return 1; }
            const auto t1 = std::chrono::steady_clock::now();
            if (!cv.encode(rec, ref, err)) { std::fprintf(stderr, "CPU encode failed: %s\n", err.c_str()); return 1; }
            cpu_ms = ms_since(t1);
            (void)t0;
            have_ref = true;
            if (!cache_file.empty()) {
                if (FILE* fp = std::fopen(cache_file.c_str(), "wb")) { std::fwrite(ref.data(), 4, ref.size(), fp); std::fclose(fp); }
            }
        }
        // GPU: the first run, then `reps` timed runs; the first one is also watched for the lowest free memory
        std::vector<float> emb;
        std::atomic<bool> stop{false};
        std::atomic<size_t> min_free{~(size_t)0};
        std::thread sampler([&] {
            cudaSetDevice(gpu);
            while (!stop.load()) {
                size_t fr = 0, t = 0;
                if (cudaMemGetInfo(&fr, &t) == cudaSuccess) {
                    size_t cur = min_free.load();
                    while (fr < cur && !min_free.compare_exchange_weak(cur, fr)) {}
                }
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            }
        });
        const auto tg = std::chrono::steady_clock::now();
        const bool ok1 = gv.encode(rec, emb, err);
        const double first_ms = ms_since(tg);
        stop.store(true);
        sampler.join();
        if (!ok1) { std::fprintf(stderr, "GPU encode failed for %s: %s\n", f.c_str(), err.c_str()); return 1; }
        std::vector<double> times;
        for (int r = 0; r < reps; ++r) {
            std::vector<float> e2;
            const auto t0 = std::chrono::steady_clock::now();
            if (!gv.encode(rec, e2, err)) { std::fprintf(stderr, "GPU encode failed: %s\n", err.c_str()); return 1; }
            times.push_back(ms_since(t0));
        }
        std::sort(times.begin(), times.end());
        const double med = times[times.size() / 2];
        std::printf("%s: %dx%d px, %d image tokens (S=%d)\n", base_name(f).c_str(), rec.orig_w, rec.orig_h, n_tok, S);
        std::printf("  time: CPU %s, GPU first %.0f ms, GPU median of %d %.0f ms (min %.0f, max %.0f)\n",
                    cpu_ms >= 0 ? (std::to_string((long long)cpu_ms) + " ms").c_str() : "(cached)", first_ms, reps, med, times.front(), times.back());
        std::printf("  GPU memory: weights %.0f MiB + working buffers %.0f MiB = %.0f MiB; lowest free memory seen during the picture %.0f MiB (before the load %.0f MiB)\n",
                    (double)gv.weight_bytes() / 1048576.0, (double)gv.last_act_bytes() / 1048576.0,
                    (double)(gv.weight_bytes() + gv.last_act_bytes()) / 1048576.0, (double)min_free.load() / 1048576.0, (double)free0 / 1048576.0);
        if (!have_ref) { std::printf("  (no CPU reference: error not computed)\n"); continue; }
        if (ref.size() != emb.size()) { std::printf("  SIZE MISMATCH cpu %zu gpu %zu\n", ref.size(), emb.size()); ++bad; continue; }
        double max_abs = 0, sum_abs = 0, max_ref = 0, max_rel_row = 0, min_cos = 2.0, sum_cos = 0;
        int worst_row = 0, worst_cos_row = 0;
        for (int r = 0; r < n_tok; ++r) {
            double dot = 0, na = 0, nb = 0, ne = 0;
            for (int c = 0; c < D; ++c) {
                const double a = ref[(size_t)r * D + c], b = emb[(size_t)r * D + c];
                const double e = std::fabs(a - b);
                if (e > max_abs) { max_abs = e; worst_row = r; }
                sum_abs += e;
                max_ref = std::max(max_ref, std::fabs(a));
                dot += a * b; na += a * a; nb += b * b; ne += (a - b) * (a - b);
            }
            const double cs = dot / (std::sqrt(na) * std::sqrt(nb) + 1e-30);
            const double rel = std::sqrt(ne) / (std::sqrt(na) + 1e-30);
            if (cs < min_cos) { min_cos = cs; worst_cos_row = r; }
            sum_cos += cs;
            max_rel_row = std::max(max_rel_row, rel);
        }
        std::printf("  error vs CPU: max abs %.3e (at row %d; largest CPU value %.3f), mean abs %.3e, max relative error per row (norm of error / norm of row) %.3e\n",
                    max_abs, worst_row, max_ref, sum_abs / ((double)n_tok * D), max_rel_row);
        std::printf("  cosine per row: min %.8f (row %d), mean %.8f\n", min_cos, worst_cos_row, sum_cos / n_tok);
    }
    return bad ? 1 : 0;
}
