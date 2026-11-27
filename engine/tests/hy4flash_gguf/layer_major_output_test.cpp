// Model-backed regression for the outputs consumed by MTP and the GPU sampler.
#include "strata/hy4flash_gguf/forward.hpp"
#include "strata/hy4flash_gguf/geometry.hpp"
#include "strata/hy4flash_gguf/kv_cache.hpp"
#include "strata/hy4flash_gguf/model_file.hpp"
#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <utility>
#include <vector>

namespace {
void check(cudaError_t rc) {
    if (rc != cudaSuccess) throw std::runtime_error(cudaGetErrorString(rc));
}

struct DeviceRows {
    float* data = nullptr;
    explicit DeviceRows(size_t count) { check(cudaMalloc(&data, count * sizeof(float))); }
    ~DeviceRows() { cudaFree(data); }
};

std::vector<float> read_rows(const float* device, size_t count) {
    std::vector<float> result(count);
    check(cudaMemcpy(result.data(), device, count * sizeof(float), cudaMemcpyDeviceToHost));
    return result;
}
}

int main(int argc, char** argv) try {
    if (argc != 3) throw std::runtime_error("usage: hy4_flash_layer_major_output_test MODEL PACK");
#ifdef _WIN32
    _putenv_s("STRATA_HY4_FLASH_PREFILL_LAYER_MAJOR", "1");
    _putenv_s("STRATA_HY4_FLASH_PREFILL_CHUNK", "32");
#else
    setenv("STRATA_HY4_FLASH_PREFILL_LAYER_MAJOR", "1", 1);
    setenv("STRATA_HY4_FLASH_PREFILL_CHUNK", "32", 1);
#endif
    auto model = strata::GgufModel::open(argv[1]);
    strata::hy4_flash::Geometry geom;
    auto err = strata::hy4_flash::read_geometry(model.meta(), geom, 128, model.find("per_layer_token_embd.weight"));
    if (!err.empty()) throw std::runtime_error(err);
    strata::hy4_flash::KvCache cache;
    if (!strata::hy4_flash::kv_cache_init(cache, geom.trunk_blocks, geom.kv_lora_rank, geom.rope_dim, 128, err))
        throw std::runtime_error(err);
    strata::hy4_flash::DenseResidency dense;
    strata::hy4_flash::ForwardOptions opt;
    opt.cache = &cache;
    opt.dense = &dense;
    opt.expert_vram_bytes = 16ull << 30;
    opt.logits_last_only = true;
    if (!strata::hy4_flash::prepare_experts(argv[1], geom, opt, err)) throw std::runtime_error(err);

    // Aligned, single-row, multi-row short tails, and the first tail taking the large head.
    std::vector<float> prefix_hidden;
    for (int length : {64, 65, 66, 71, 72, 73}) {
        std::vector<int32_t> ids(length);
        for (int i = 0; i < length; ++i) ids[i] = 100 + i;
        opt.history = &ids;
        const size_t width = geom.hidden_dim;
        DeviceRows hidden((length + 1) * width);
        opt.hidden_out = hidden.data;
        std::vector<float> reference_logits, reference_hidden, last_from_all;
        // Use the same head row count for each host/device pair. A one-row head
        // can take a different numerical path than the head computing all rows.
        for (int pass = 0; pass < 4; ++pass) {
            const bool all = pass < 2;
            const bool gpu = pass % 2 == 1;
            cache.len = 0;
            check(cudaMemset(hidden.data, 0xff, (length + 1) * width * sizeof(float)));
            const float* logits = nullptr;
            opt.hidden_all_rows = all;
            opt.device_logits_out = gpu ? &logits : nullptr;
            std::vector<float> host;
            if (!strata::hy4_flash::forward_prefill(argv[1], geom, argv[2], ids, host, err, opt))
                throw std::runtime_error(err);
            if (cache.len != length) throw std::runtime_error("KV length did not advance to the end");
            const size_t valid = (all ? length : 1) * width;
            auto rows = read_rows(hidden.data, valid + width);
            if (!std::all_of(rows.begin(), rows.begin() + valid, [](float x) { return std::isfinite(x); }) ||
                !std::all_of(rows.begin() + valid, rows.end(), [](float x) { return std::isnan(x); }))
                throw std::runtime_error("hidden rows missing or written beyond requested extent");
            rows.resize(valid);
            if (!gpu) {
                if (host.size() != size_t(geom.vocab_size)) throw std::runtime_error("missing last logits");
                if (all) {
                    if (length == 64) prefix_hidden = rows;
                    else if (!std::equal(prefix_hidden.begin(), prefix_hidden.end(), rows.begin()))
                        throw std::runtime_error("appending a short tail changed a completed chunk's hidden rows");
                    last_from_all.assign(rows.end() - width, rows.end());
                } else if (length > 64 && length <= 72 && rows != last_from_all) {
                    // Both variants use the same multi-column head for a short tail.
                    throw std::runtime_error("last-only hidden output selected the wrong tail row");
                }
                reference_logits = std::move(host);
                reference_hidden = std::move(rows);
            } else if (!host.empty() || !logits || read_rows(logits, geom.vocab_size) != reference_logits ||
                       rows != reference_hidden) {
                throw std::runtime_error("last hidden row or device logits differ from host reference");
            }
        }
        std::printf("length %d: hidden extents, final row and device logits agree\n", length);
    }
    return 0;
} catch (const std::exception& e) {
    std::fprintf(stderr, "%s\n", e.what());
    return 1;
}
