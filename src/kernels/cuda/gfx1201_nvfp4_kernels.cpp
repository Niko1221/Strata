// src/kernels/cuda/gfx1201_nvfp4_kernels.cpp - HIP module loader and kernel launcher for embedded RDNA4 HSACO kernels.
#include "strata/kernels/gfx1201_nvfp4_kernels.hpp"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <sys/stat.h>
#include <utility>
#include <vector>

namespace strata::kernels {
namespace {

#define STRATA_GFX1201_HIP_CHECK(call)                                                                      \
    do {                                                                                                    \
        hipError_t err__ = (call);                                                                          \
        if (err__ != hipSuccess) {                                                                          \
            std::fprintf(stderr, "HIP error at %s:%d: %s (%d)\n", __FILE__, __LINE__,                       \
                         hipGetErrorString(err__), (int) err__);                                            \
            std::exit(1);                                                                                   \
        }                                                                                                   \
    } while (0)

}  // namespace

Gfx1201KernelCache::Gfx1201KernelCache(std::string cache_dir)
    : cache_dir_(std::move(cache_dir)) {}

Gfx1201KernelCache::~Gfx1201KernelCache() {
    for (auto& kv : kernels_) {
        if (kv.second.mod) {
            (void) hipModuleUnload(kv.second.mod);
        }
    }
}

const LoadedKernel& Gfx1201KernelCache::get(const std::string& key) {
    auto it = kernels_.find(key);
    if (it != kernels_.end()) {
        return it->second;
    }
    if (!cache_dir_.empty()) {
        std::string hsaco_path = cache_dir_ + "/" + key + ".hsaco";
        std::string json_path = cache_dir_ + "/" + key + ".json";
        if (file_exists(hsaco_path) && file_exists(json_path)) {
            LoadedKernel lk = load_from_disk(key, hsaco_path, json_path);
            auto [ins_it, _] = kernels_.emplace(key, lk);
            return ins_it->second;
        }
    }
    const EmbeddedKernelDesc* emb = find_embedded_kernel(key);
    if (!emb) {
        std::fprintf(stderr, "[gfx1201-nvfp4] Fatal: kernel '%s' not found in embedded gfx1201 kernel table!\n",
                     key.c_str());
        std::exit(1);
    }
    LoadedKernel lk;
    lk.kernel_name = emb->kernel_name;
    lk.bx = emb->bx;
    lk.by = emb->by;
    lk.bz = emb->bz;
    lk.gx = emb->gx;
    lk.gy = emb->gy;
    lk.gz = emb->gz;
    STRATA_GFX1201_HIP_CHECK(hipModuleLoadData(&lk.mod, emb->hsaco_data));
    STRATA_GFX1201_HIP_CHECK(hipModuleGetFunction(&lk.func, lk.mod, lk.kernel_name.c_str()));
    auto [ins_it, _] = kernels_.emplace(key, lk);
    return ins_it->second;
}

void Gfx1201KernelCache::launch(const std::string& key, int gx, int gy, int gz, void** args, hipStream_t stream) {
    const LoadedKernel& lk = get(key);
    if (profile_enabled_) {
        (void) hipDeviceSynchronize();
        auto t0 = std::chrono::steady_clock::now();
        STRATA_GFX1201_HIP_CHECK(
            hipModuleLaunchKernel(lk.func, gx, gy, gz, lk.bx, lk.by, lk.bz, 0, stream, args, nullptr));
        (void) hipDeviceSynchronize();
        double us = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count();
        prof_us_[tag_ + key] += us;
        prof_cnt_[tag_ + key] += 1;
    } else {
        STRATA_GFX1201_HIP_CHECK(
            hipModuleLaunchKernel(lk.func, gx, gy, gz, lk.bx, lk.by, lk.bz, 0, stream, args, nullptr));
    }
}

void Gfx1201KernelCache::launch_default(const std::string& key, void** args, hipStream_t stream) {
    const LoadedKernel& lk = get(key);
    launch(key, lk.gx, lk.gy, lk.gz, args, stream);
}

void Gfx1201KernelCache::set_profile(bool en, std::string tag) {
    profile_enabled_ = en;
    tag_ = std::move(tag);
}

void Gfx1201KernelCache::dump_profile() const {
    if (prof_us_.empty()) return;
    std::vector<std::pair<std::string, double>> v(prof_us_.begin(), prof_us_.end());
    std::sort(v.begin(), v.end(), [](const auto& a, const auto& b) { return a.second > b.second; });
    double tot = 0.0;
    for (const auto& kv : v) tot += kv.second;
    std::printf("\n=== gfx1201-nvfp4 Per-Kernel Profile (Total: %.2f ms) ===\n", tot / 1000.0);
    for (const auto& kv : v) {
        int c = prof_cnt_.at(kv.first);
        std::printf("  %-36s : %8.2f ms (%4d calls, %6.1f us/call, %5.1f%%)\n",
                    kv.first.c_str(), kv.second / 1000.0, c, kv.second / c, 100.0 * kv.second / tot);
    }
}

bool Gfx1201KernelCache::file_exists(const std::string& p) {
    struct stat st {};
    return ::stat(p.c_str(), &st) == 0 && st.st_size > 0;
}

LoadedKernel Gfx1201KernelCache::load_from_disk(const std::string& key, const std::string& hsaco_path,
                                                const std::string& json_path) {
    std::ifstream jf(json_path);
    if (!jf) {
        std::fprintf(stderr, "Failed to open metadata JSON: %s\n", json_path.c_str());
        std::exit(1);
    }
    std::string jtxt((std::istreambuf_iterator<char>(jf)), std::istreambuf_iterator<char>());
    LoadedKernel lk;

    auto kn_pos = jtxt.find("\"kernel_name\"");
    if (kn_pos == std::string::npos) {
        std::fprintf(stderr, "Invalid kernel JSON for %s\n", key.c_str());
        std::exit(1);
    }
    auto q1 = jtxt.find('"', jtxt.find(':', kn_pos) + 1);
    auto q2 = jtxt.find('"', q1 + 1);
    lk.kernel_name = jtxt.substr(q1 + 1, q2 - q1 - 1);

    auto parse_vec3 = [&](const char* field, int& x, int& y, int& z) {
        auto p = jtxt.find(field);
        if (p == std::string::npos) return;
        auto b1 = jtxt.find('[', p);
        auto b2 = jtxt.find(']', b1);
        std::string sub = jtxt.substr(b1 + 1, b2 - b1 - 1);
        std::sscanf(sub.c_str(), "%d, %d, %d", &x, &y, &z);
    };
    parse_vec3("\"block\"", lk.bx, lk.by, lk.bz);
    parse_vec3("\"default_grid\"", lk.gx, lk.gy, lk.gz);

    std::ifstream hf(hsaco_path, std::ios::binary);
    std::vector<char> hsaco((std::istreambuf_iterator<char>(hf)), std::istreambuf_iterator<char>());
    if (hsaco.empty()) {
        std::fprintf(stderr, "Empty HSACO file: %s\n", hsaco_path.c_str());
        std::exit(1);
    }
    STRATA_GFX1201_HIP_CHECK(hipModuleLoadData(&lk.mod, hsaco.data()));
    STRATA_GFX1201_HIP_CHECK(hipModuleGetFunction(&lk.func, lk.mod, lk.kernel_name.c_str()));
    return lk;
}

}  // namespace strata::kernels
