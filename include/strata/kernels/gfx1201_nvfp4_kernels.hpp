// include/strata/kernels/gfx1201_nvfp4_kernels.hpp - Embedded RDNA4 (gfx1201) HSACO table and HIP module cache.
#pragma once

#include <hip/hip_runtime_api.h>

#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>
#include <unordered_map>

namespace strata::kernels {

struct EmbeddedKernelDesc {
    const char* key;
    const char* kernel_name;
    int bx, by, bz;
    int gx, gy, gz;
    const uint8_t* hsaco_data;
    size_t hsaco_size;
};

const EmbeddedKernelDesc* find_embedded_kernel(std::string_view key);
size_t embedded_kernel_count();
const EmbeddedKernelDesc* embedded_kernels_begin();

struct LoadedKernel {
    hipModule_t mod = nullptr;
    hipFunction_t func = nullptr;
    std::string kernel_name;
    int bx = 256;
    int by = 1;
    int bz = 1;
    int gx = 1;
    int gy = 1;
    int gz = 1;
};

class Gfx1201KernelCache {
public:
    explicit Gfx1201KernelCache(std::string cache_dir);
    ~Gfx1201KernelCache();

    Gfx1201KernelCache(const Gfx1201KernelCache&) = delete;
    Gfx1201KernelCache& operator=(const Gfx1201KernelCache&) = delete;

    const LoadedKernel& get(const std::string& key);
    void launch(const std::string& key, int gx, int gy, int gz, void** args, hipStream_t stream = nullptr);
    void launch_default(const std::string& key, void** args, hipStream_t stream = nullptr);
    void set_profile(bool en, std::string tag = "");
    void dump_profile() const;

private:
    static bool file_exists(const std::string& p);
    static LoadedKernel load_from_disk(const std::string& key, const std::string& hsaco_path,
                                       const std::string& json_path);

    std::string cache_dir_;
    std::unordered_map<std::string, LoadedKernel> kernels_;
    bool profile_enabled_ = false;
    std::string tag_;
    std::unordered_map<std::string, double> prof_us_;
    std::unordered_map<std::string, int> prof_cnt_;
};

}  // namespace strata::kernels
