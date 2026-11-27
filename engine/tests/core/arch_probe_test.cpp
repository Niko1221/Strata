// The message an engine gives when its binary has no GPU code for the card (strata/core/arch_probe.hpp, #6759): it names the engine,
// the card and its compute capability, the CUDA error, and the way out (the CUDA 12 package for Pascal / Volta, else the right package or a rebuild).
#include "strata/core/arch_probe.hpp"

#include <cstdio>
#include <string>

using strata::core::no_gpu_code_message;

namespace {
int failed = 0;
#define CHECK(x) do { if (!(x)) { std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #x); ++failed; } } while (0)
bool has(const std::string& s, const char* part) { return s.find(part) != std::string::npos; }
}  // namespace

int main() {
    const std::string v100 = no_gpu_code_message("strata-inkling", "Tesla V100-PCIE-32GB", 7, 0, "no kernel image is available for execution on the device");
    CHECK(has(v100, "strata-inkling"));
    CHECK(has(v100, "Tesla V100-PCIE-32GB"));
    CHECK(has(v100, "compute capability 7.0"));
    CHECK(has(v100, "no kernel image is available"));
    CHECK(has(v100, "CUDA 12 engine package"));
    CHECK(has(v100, "-DCMAKE_CUDA_ARCHITECTURES=70"));
    CHECK(has(v100, "STRATA_EXPERIMENTAL_SM60=ON"));

    const std::string p61 = no_gpu_code_message("strata-hy4", "GeForce GTX 1080 Ti", 6, 1, "no kernel image");
    CHECK(has(p61, "strata-hy4") && has(p61, "6.1") && has(p61, "CMAKE_CUDA_ARCHITECTURES=61"));

    // a newer card the binary was not built for: no word about CUDA 12 / Volta
    const std::string ada = no_gpu_code_message("strata-hy4", "NVIDIA GeForce RTX 3060", 8, 6, "no kernel image is available for execution on the device");
    CHECK(has(ada, "compute capability 8.6"));
    CHECK(!has(ada, "Volta"));
    CHECK(has(ada, "-DCMAKE_CUDA_ARCHITECTURES=86"));
    CHECK(has(ada, "strata update"));

    if (failed) {
        std::fprintf(stderr, "arch_probe_test: %d failed\n", failed);
        return 1;
    }
    std::printf("arch_probe_test: ok\n");
    return 0;
}
