// Model-dependent prompt encoder parity. Supply a local DFlash2 GGUF; no downloads.
// Two independent drafters receive the same synthetic, padded feature planes.
// Compare every physical FP16 K/V cell after partial groups and ring wraps.
#include "strata/core/dflash.hpp"
#include "strata/core/layer.hpp"
#include "strata/core/layout.hpp"
#include "strata/core/session.hpp"
#include "strata/kernels/bf16_bits.hpp"
#include "strata/spec/dflash_policy.hpp"
#include <cuda_runtime.h>
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <utility>
#include <vector>

using namespace strata::core;
namespace {
void checked(cudaError_t error) {
    if (error != cudaSuccess) throw std::runtime_error(cudaGetErrorString(error));
}
void flag(const char* name, const char* value) {
#ifdef _WIN32
    _putenv_s(name, value ? value : "");
#else
    if (value) setenv(name, value, 1); else unsetenv(name);
#endif
}
struct Allocation {
    void* data = nullptr;
    explicit Allocation(size_t bytes) { checked(cudaMalloc(&data, bytes)); }
    ~Allocation() { if (data) cudaFree(data); }
    Allocation(const Allocation&) = delete;
};
struct Primary {
    QsaState state;
    void* arena = nullptr;
    explicit Primary(const ModelGeometry& g, int64_t capacity) {
        QsaStateInitOptions options;
        options.force_owned_kv = options.force_f16_kv = options.disable_elastic = true;
        options.disable_streaming = options.no_indexer = true;
        checked(cudaMalloc(&arena, qsa_state_bytes(g, capacity, true, 0, options)));
        if (!qsa_state_init(g, capacity, arena, state, nullptr, 0, options))
            throw std::runtime_error("cannot initialize primary QSA state");
        qsa_state_zero(state, g, nullptr);
        checked(cudaDeviceSynchronize());
    }
    ~Primary() {
        if (state.host_step) cudaFreeHost(state.host_step);
        if (state.host_pos) cudaFreeHost(state.host_pos);
        if (arena) cudaFree(arena);
    }
};
uint64_t compare(const DFlashDrafter& before, const DFlashDrafter& after) {
    const auto& geometry = before.artifact().geom();
    const auto shapes = qsa_shapes(before.pool_geometry());
    uint64_t compared = 0;
    for (size_t layer = 0; layer < before.kv_states().size(); ++layer) {
        const auto& a = before.kv_states()[layer];
        const auto& b = after.kv_states()[layer];
        if (a.n_slots != b.n_slots) throw std::runtime_error("physical KV geometry changed");
        const size_t count = a.n_slots * shapes.page_size * geometry.n_head_kv * geometry.head_dim;
        std::vector<uint16_t> expected(count), actual(count);
        for (int side = 0; side < 2; ++side) {
            checked(cudaMemcpy(expected.data(), side ? a.v_pool : a.k_pool, count * 2, cudaMemcpyDeviceToHost));
            checked(cudaMemcpy(actual.data(), side ? b.v_pool : b.k_pool, count * 2, cudaMemcpyDeviceToHost));
            if (expected != actual) {
                const auto at = std::mismatch(expected.begin(), expected.end(), actual.begin()).first - expected.begin();
                size_t differences = 0;
                for (size_t j = 0; j < count; ++j) if (expected[j] != actual[j]) {
                    if (differences < 8) std::fprintf(stderr, "KV mismatch layer=%zu side=%d index=%zu before=%04x after=%04x\n",
                                                     layer, side, j, expected[j], actual[j]);
                    ++differences;
                }
                std::fprintf(stderr, "different elements %zu of %zu\n", differences, count);
                throw std::runtime_error("KV bits differ, layer " + std::to_string(layer) +
                    " side " + std::to_string(side) + " element " + std::to_string(at));
            }
            compared += count * 2;
        }
    }
    return compared;
}
} // namespace

int main(int argc, char** argv) {
    if (argc < 2 || cudaSetDevice(0) != cudaSuccess) {
        std::puts("SKIP: supply a local DFlash2 GGUF and a CUDA GPU");
        return 77;
    }
    try {
        flag("STRATA_DF_PARITY", nullptr); flag("STRATA_DF_DBG", nullptr); flag("STRATA_DF_EVENTS", nullptr);
        flag("STRATA_DFLASH2_CONTEXT_GRAPH", "0"); flag("STRATA_DFLASH2_DIRECT_QUANT", "0");
        flag("STRATA_DFLASH2_BF16_TILE", "0");
        bool graphs = true, distant = false, timing = false, context_only = false, grid = false, memory_only = false;
        for (int i = 2; i < argc; ++i) {
            if (std::string(argv[i]) == "--no-graphs") graphs = false;
            else if (std::string(argv[i]) == "--long") distant = true;
            else if (std::string(argv[i]) == "--bench") timing = true;
            else if (std::string(argv[i]) == "--context-only") context_only = true;
            else if (std::string(argv[i]) == "--grid") grid = true;
            else if (std::string(argv[i]) == "--memory-only") memory_only = true;
            else throw std::runtime_error("expected --no-graphs, --long, --bench, --context-only or --grid / --memory-only");
        }
        if (grid && context_only) throw std::runtime_error("choose --grid or --context-only");
        flag("STRATA_DFLASH2_PROMPT_GRID", "0");
        flag("STRATA_DFLASH_GRAPH", graphs ? "1" : "0");
        const int64_t capacity = distant ? 327680 : 16384;
        ModelGeometry geometry;
        Primary primary(geometry, capacity);
        SessionState session;
        session.qsa_states = &primary.state; session.qsa_alloc = 1; session.max_cells = capacity;
        DFlashDrafter before, after;
        std::string error;
        if (!before.load(argv[1], error) || !after.load(argv[1], error)) throw std::runtime_error(error);
        if (!before.artifact().geom().is_dflash2()) throw std::runtime_error("expected a DFlash2 artifact");
        flag("STRATA_DFLASH2_PROMPT_BATCH", grid ? "1" : "0");
        if (!before.upload(geometry, session, 0, capacity, -1, error)) throw std::runtime_error(error);
        flag("STRATA_DFLASH2_PROMPT_BATCH", context_only ? "0" : "1");
        flag("STRATA_DFLASH2_CONTEXT_GRAPH", context_only ? "1" : "0");
        flag("STRATA_DFLASH2_PROMPT_GRID", grid ? "1" : "0");
        if (!after.upload(geometry, session, 0, capacity, -1, error)) throw std::runtime_error(error);
        if (!context_only && !grid && after.vram_bytes() <= before.vram_bytes())
            throw std::runtime_error("prompt batching was not enabled in the candidate");
        std::printf("ACCOUNTED_VRAM before=%llu after=%llu delta=%lld\n",
                    (unsigned long long)before.vram_bytes(), (unsigned long long)after.vram_bytes(),
                    (long long)after.vram_bytes() - (long long)before.vram_bytes());
        if (memory_only) return 0;
        const auto& dg = before.artifact().geom();
        int64_t position = 0;
        std::vector<std::pair<int64_t,int>> cases;
        for (int rows : {1,7,8,9,63,64,65,127,128,129,255,256,257,511,512,513,1024,2049,512,512}) {
            cases.push_back({position, rows});
            position += rows;
        }
        const int64_t contiguous_end = position;
        if (distant) { cases.push_back({262141,512}); cases.push_back({263679,513}); }
        uint64_t bytes = 0;
        for (const auto& [position, rows] : cases) {
            const int stride = rows + 3, taps = (int)dg.target_layers.size(), hidden = (int)dg.hidden;
            std::vector<uint16_t> source((size_t)taps * stride * hidden, 0x7fc0);
            for (int tap = 0; tap < taps; ++tap) for (int row = 0; row < rows; ++row)
                for (int c = 0; c < hidden; ++c) {
                    const float value = ((int)((position + row) * 131 + tap * 17 + c * 7) % 997 - 498) * 0.003f;
                    source[((size_t)tap * stride + row) * hidden + c] = strata::kernels::bf16_from_f32(value);
                }
            Allocation input(source.size() * 2);
            checked(cudaMemcpy(input.data, source.data(), source.size() * 2, cudaMemcpyHostToDevice));
            // Drafters use nonblocking streams. Pageable H2D copy completion
            // must be explicit before either stream reads this borrowed input.
            checked(cudaDeviceSynchronize());
            if (!before.add_context((const uint16_t*)input.data, taps, stride, position, rows, error) ||
                !after.add_context((const uint16_t*)input.data, taps, stride, position, rows, error))
                throw std::runtime_error(error);
            bytes += compare(before, after);
            std::printf("prompt fusion pos=%lld rows=%d: all five KV layers bit-identical\n", (long long)position, rows);
        }
        const int64_t physical_cells = before.kv_states()[0].n_slots * qsa_shapes(before.pool_geometry()).page_size;
        int expected_graphs = 0;
        if (graphs && context_only) expected_graphs = 3; // only 1/7/8-row tails appear in this fixture
        else if (graphs) {
            const int cap = strata::spec::dflash_prompt_batch_rows(std::min(capacity, physical_cells));
            for (int rows : {64,128,256,512}) expected_graphs += rows <= cap;
        }
        if (before.context_graph_count() != (grid ? expected_graphs : 0) || after.context_graph_count() != expected_graphs)
            throw std::runtime_error("context graph capture silently fell back or used an unexpected cache key");
        std::printf("context graphs: original=%d candidate=%d\n", before.context_graph_count(), after.context_graph_count());
        if (timing) {
            constexpr int rows = 512;
            const int taps = (int)dg.target_layers.size(), hidden = (int)dg.hidden;
            std::vector<uint16_t> features((size_t)taps * rows * hidden, strata::kernels::bf16_from_f32(0.125f));
            Allocation input(features.size() * 2);
            checked(cudaMemcpy(input.data, features.data(), features.size() * 2, cudaMemcpyHostToDevice));
            checked(cudaDeviceSynchronize());
            auto measure = [&](DFlashDrafter& drafter, int64_t pos) {
                const auto start = std::chrono::steady_clock::now();
                if (!drafter.add_context((const uint16_t*)input.data, taps, rows, pos, rows, error))
                    throw std::runtime_error(error);
                return std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now() - start).count();
            };
            for (int rep = -10; rep < 24; ++rep) {
                const int64_t pos = ((rep + 10) * rows) % (capacity - rows);
                double old_ms, new_ms;
                if (rep % 2 == 0) { old_ms = measure(before, pos); new_ms = measure(after, pos); }
                else { new_ms = measure(after, pos); old_ms = measure(before, pos); }
                if (rep >= 0) std::printf("BENCH rep=%d rows=512 original_ms=%.6f batched_ms=%.6f\n", rep, old_ms, new_ms);
            }
            compare(before, after);
        }
        const int64_t physical = before.kv_states()[0].n_slots * qsa_shapes(before.pool_geometry()).page_size;
        std::printf("PASS: %llu bytes compared, graphs=%s, %lld contiguous physical KV wraps, distant=%s\n",
                     (unsigned long long)bytes, graphs ? "on" : "off", (long long)(contiguous_end / physical), distant ? "yes" : "no");
    } catch (const std::exception& error) {
        std::fprintf(stderr, "FAIL: %s\n", error.what());
        return 1;
    }
}
