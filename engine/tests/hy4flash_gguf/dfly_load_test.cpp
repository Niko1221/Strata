// Load a real DFly and target GGUF pair. Skip when the files or CUDA are unavailable.
#include "strata/artifact/gguf_reader.hpp"
#include "strata/hy4flash_gguf/dfly.hpp"
#include "strata/hy4flash_gguf/geometry.hpp"

#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <functional>
#include <string>

using namespace strata::hy4_flash;

int main() {
    const char* dp = std::getenv("STRATA_DFLY_GGUF");
    const char* tp = std::getenv("STRATA_HY4_FLASH_GGUF");
    if (dp == nullptr || tp == nullptr) {
        std::printf("SKIP: set STRATA_DFLY_GGUF and STRATA_HY4_FLASH_GGUF\n");
        return 77;
    }
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) {
        std::printf("SKIP: no CUDA device\n");
        return 77;
    }
    Geometry g;
    try {
        const strata::GgufFile meta(tp);
        const std::string e = read_geometry(meta, g, 0);
        if (!e.empty()) { std::printf("FAIL: target geometry: %s\n", e.c_str()); return 1; }
    } catch (const std::exception& e) {
        std::printf("FAIL: target: %s\n", e.what());
        return 1;
    }
    int fails = 0;
    {
        DflyDrafter d;
        std::string err;
        const bool ok = d.load(dp, g, err);
        const auto& L = d.target_layers();
        const bool layers = L[0] == 25 && L[1] == 29 && L[2] == 33 && L[3] == 37 && L[4] == 41;
        if (!ok || !layers || d.mask_id() != 120009 || d.vocab() != g.vocab_size || d.weight_bytes() == 0) {
            std::printf("FAIL: the real pair: ok=%d layers=%d mask=%d err=%s\n", ok, layers, d.mask_id(), err.c_str());
            ++fails;
        } else {
            std::printf("ok   the real pair loads: %.1f MiB, layers 25,29,33,37,41, mask %d\n",
                        d.weight_bytes() / 1048576.0, d.mask_id());
        }
    }
    struct Neg { const char* what; const char* needle; std::function<void(Geometry&)> edit; };
    const Neg negs[] = {
        {"a sliding target layer", "target layer 33", [](Geometry& x) { x.layers[33].attention = Attention::Sliding; }},
        {"another indexer top_k", "top_k", [](Geometry& x) { x.idx_top_k = 1024; }},
        {"another rope base", "rope.freq_base", [](Geometry& x) { x.rope_theta_sparse = 1e4f; }},
        {"another vocab", "vocab", [](Geometry& x) { x.vocab_size = 120000; }},
        {"another head count", "head count", [](Geometry& x) { x.layers[25].n_heads = 64; }},
    };
    for (const Neg& n : negs) {
        Geometry bad = g;
        n.edit(bad);
        DflyDrafter d;
        std::string err;
        const bool ok = d.load(dp, bad, err);
        if (ok || err.find(n.needle) == std::string::npos) {
            std::printf("FAIL: %s: load=%d err=%s\n", n.what, ok, err.c_str());
            ++fails;
        } else {
            std::printf("ok   %s is refused: %s\n", n.what, err.c_str());
        }
    }
    {
        DflyDrafter d;
        std::string err;
        if (d.load(tp, g, err)) { std::printf("FAIL: the target file loaded as a drafter\n"); ++fails; }
        else std::printf("ok   the target file is not a drafter: %s\n", err.c_str());
    }
    std::printf("%s\n", fails == 0 ? "PASS" : "FAIL");
    return fails == 0 ? 0 : 1;
}
