// Bitwise Q4_K pair regression.
#include "q4_pair_reference.cuh"
#include "../../src/kernels/cuda/q4_pair_reuse.hpp"
#include "strata/artifact/gguf_reader.hpp"
#include "strata/kernels/native_mmvq.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace k = strata::kernels;
namespace {
void ck(cudaError_t e) { if (e != cudaSuccess) throw std::runtime_error(cudaGetErrorString(e)); }
struct Device {
    void* p = nullptr;
    explicit Device(std::size_t bytes) { ck(cudaMalloc(&p, bytes)); }
    ~Device() { cudaFree(p); }
    Device(const Device&) = delete;
    template<class T> T* as() const { return static_cast<T*>(p); }
};
struct Stream {
    cudaStream_t s;
    Stream() { ck(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking)); }
    ~Stream() { cudaStreamDestroy(s); }
};
struct Graph {
    cudaGraph_t g = nullptr;
    cudaGraphExec_t ex = nullptr;
    ~Graph() { if (ex) cudaGraphExecDestroy(ex); if (g) cudaGraphDestroy(g); }
    void finish(cudaStream_t s) {
        ck(cudaStreamEndCapture(s, &g));
        ck(cudaGraphInstantiate(&ex, g, nullptr, nullptr, 0));
    }
};
constexpr uint32_t sentinel = 0x7faabcde;
constexpr int guard = 32;
std::size_t compared = 0;

__global__ void produce(const uint8_t* src, uint8_t* dst, std::size_t bytes,
                       const int32_t* id_src, int32_t* ids, int groups) {
    k::pdl_trigger();
    k::pdl_wait();
    // Delay the last block after permitting dependents. A consumer without a
    // wait can otherwise pass when this small producer finishes before launch.
    if (blockIdx.x + 1 == gridDim.x) __nanosleep(50000);
    const std::size_t i = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < bytes) dst[i] = src[i];
    if (blockIdx.x + 1 == gridDim.x && threadIdx.x < groups)
        ids[threadIdx.x] = id_src[threadIdx.x];
}

void pair(int variant, int rows, const void* gate, std::size_t gs, const void* up,
          std::size_t us, const void* x, std::size_t xs, float* yg, float* yu,
          int ni, int no, int groups, cudaStream_t s, const int32_t* ids) {
    if (variant == 1) {
        k::q4_pair_reuse_launch(gate, gs, up, us, x, xs, yg, yu, ni, no, groups, rows, s, ids,
                                groups == 64 ? 16 : 0);
    } else if (variant == 3) {
        if (!k::native_mmvq_grouped_pair_wide(12, gate, gs, up, us, x, xs, yg, yu, ni, no, groups, 16, s, ids, 1))
            throw std::runtime_error("wide pair refused");
    } else if (variant == 2) {
        if (!k::native_mmvq_grouped_pair(12, gate, gs, up, us, x, xs, yg, yu, ni, no, groups, s, ids, 1))
            throw std::runtime_error("product pair refused");
    } else {
#define REF(R) q4_reference::launch<R>(gate, gs, up, us, x, xs, yg, yu, ni, no, groups, s, ids)
        switch (rows) {
        case 1: REF(1); break;
        case 2: REF(2); break;
        case 4: REF(4); break;
        case 8: REF(8); break;
        default: throw std::runtime_error("invalid rows");
        }
#undef REF
    }
    ck(cudaGetLastError());
}

std::vector<uint8_t> weights(int ni, int no, int experts, std::size_t stride, unsigned seed,
                             const void* real = nullptr) {
    const auto bytes = k::native_mmvq_weight_bytes(12, ni, no);
    std::vector<uint8_t> w(stride * experts);
    std::mt19937 rng(seed);
    for (auto& b : w) b = uint8_t(rng());
    for (int e = 0; e < experts; ++e) {
        auto* dst = w.data() + e * stride;
        if (real) { std::memcpy(dst, static_cast<const uint8_t*>(real) + e * bytes, bytes); continue; }
        for (std::size_t b = 0; b < bytes; b += 144) {
            // Independent scales/mins, random packed high bits, including zero.
            uint16_t d = uint16_t(0x1800 + rng() % 0x1400), m = uint16_t(0x1800 + rng() % 0x1400);
            if (b % (144 * 19) == 0) d = 0;
            if (b % (144 * 23) == 0) m = 0;
            std::memcpy(dst + b, &d, 2); std::memcpy(dst + b + 2, &m, 2);
        }
    }
    return w;
}

std::size_t programmatic_edges(cudaGraph_t g) {
#if CUDART_VERSION >= 12030
    std::size_t n = 0;
#if CUDART_VERSION >= 13000
    ck(cudaGraphGetEdges(g, nullptr, nullptr, nullptr, &n));
#else
    ck(cudaGraphGetEdges_v2(g, nullptr, nullptr, nullptr, &n));
#endif
    std::vector<cudaGraphNode_t> a(n), b(n);
    std::vector<cudaGraphEdgeData> ed(n);
#if CUDART_VERSION >= 13000
    ck(cudaGraphGetEdges(g, a.data(), b.data(), ed.data(), &n));
#else
    ck(cudaGraphGetEdges_v2(g, a.data(), b.data(), ed.data(), &n));
#endif
    return std::count_if(ed.begin(), ed.end(), [](const auto& e) { return e.type == cudaGraphDependencyTypeProgrammatic; });
#else
    (void) g; return 0;
#endif
}

void* pair_function(cudaGraph_t g) {
    std::size_t n = 0;
    ck(cudaGraphGetNodes(g, nullptr, &n));
    std::vector<cudaGraphNode_t> nodes(n);
    ck(cudaGraphGetNodes(g, nodes.data(), &n));
    void* found = nullptr;
    for (const auto node : nodes) {
        cudaGraphNodeType type;
        ck(cudaGraphNodeGetType(node, &type));
        if (type != cudaGraphNodeTypeKernel) continue;
        cudaKernelNodeParams params{};
        ck(cudaGraphKernelNodeGetParams(node, &params));
        if (params.func == reinterpret_cast<void*>(produce)) continue;
        if (found) throw std::runtime_error("unexpected extra kernel in pair graph");
        found = params.func;
    }
    if (!found) throw std::runtime_error("pair graph has no consumer");
    return found;
}

void check_case(int ni, int no, int rows, bool shared, bool implicit_ids,
                unsigned seed, bool pdl, const void* gate_real = nullptr, const void* up_real = nullptr) {
    constexpr int experts = 16, groups = 16;
    const auto wb = k::native_mmvq_weight_bytes(12, ni, no), gs = wb + 144, us = wb + 288;
    const auto qb = k::native_q8_1_bytes(ni), xs = shared ? 0 : qb + 36;
    const auto qbytes = shared ? qb : groups * xs;
    const std::size_t count = std::size_t(groups) * no, outbytes = (count + 2 * guard) * 4;
    auto g = weights(ni, no, experts, gs, seed, gate_real);
    auto u = weights(ni, no, experts, us, seed + 17, up_real);
    Device dg(g.size()), du(u.size()), input(std::size_t(ni) * groups * 4), qpacked(qb * groups);
    Device qsource(qbytes), q(qbytes), idsource(groups * 4), ids(groups * 4);
    Device rg(outbytes), ru(outbytes), yg(outbytes), yu(outbytes);
    Stream stream; const auto s = stream.s;
    ck(cudaMemcpyAsync(dg.p, g.data(), g.size(), cudaMemcpyHostToDevice, s));
    ck(cudaMemcpyAsync(du.p, u.data(), u.size(), cudaMemcpyHostToDevice, s));
    ck(cudaStreamSynchronize(s));
    std::vector<uint32_t> blank(count + 2 * guard, sentinel);
    std::vector<int32_t> hid(groups);
    std::mt19937 rng(seed + 113);
    k::native_mmvq_set_rows(rows);
    auto reset = [&] {
        for (Device* p : {&yg, &yu}) ck(cudaMemcpyAsync(p->p, blank.data(), outbytes, cudaMemcpyHostToDevice, s));
        ck(cudaMemsetAsync(q.p, 0xff, qbytes, s));
        ck(cudaMemsetAsync(ids.p, 0xff, groups * 4, s));
    };
    auto launch = [&](int variant, Device& a, Device& b) {
        k::launch_pdl(produce, dim3(unsigned((qbytes + 255) / 256)), 256, 0, s,
                      qsource.as<uint8_t>(), q.as<uint8_t>(), qbytes, idsource.as<int32_t>(), ids.as<int32_t>(), groups);
        pair(variant, rows, dg.p, gs, du.p, us, q.p, xs, a.as<float>() + guard, b.as<float>() + guard,
             ni, no, groups, s, implicit_ids ? nullptr : ids.as<int32_t>());
    };
    Graph candidate, product;
    // Record both before changing data. Replay must read new IDs and activations.
    for (int variant : {1, 2}) {
        k::pdl_scope() = pdl;
        ck(cudaStreamBeginCapture(s, cudaStreamCaptureModeThreadLocal));
        launch(variant, yg, yu);
        auto& graph = variant == 1 ? candidate : product;
        graph.finish(s);
        k::pdl_scope() = false;
        if (variant == 1 && pdl && programmatic_edges(graph.g) == 0)
            throw std::runtime_error("PDL requested but candidate graph has no programmatic edge");
    }
    const bool reused = k::q4_pair_reuse_enabled(ni, no, groups, rows);
    if ((pair_function(candidate.g) == pair_function(product.g)) != reused)
        throw std::runtime_error("product graph selected the wrong pair kernel");
    if (reused && pdl && programmatic_edges(product.g) == 0)
        throw std::runtime_error("product graph lost its programmatic edge");
    for (int replay = 0; replay < 3; ++replay) {
        std::vector<float> hx(std::size_t(ni) * groups);
        for (std::size_t i = 0; i < hx.size(); ++i) {
            const float r = float(int(rng() % 20001) - 10000) / 4096.f;
            hx[i] = replay == 0 ? r : replay == 1 ? (i % 2 ? -1.f : 1.f) * (0.5f + std::abs(r)) : 0.f;
        }
        for (int i = 0; i < groups; ++i) hid[i] = replay == 0 ? int(rng() % experts) : (experts - 1 - i);
        if (replay != 0) { hid[3] = hid[2]; hid[5] = -1; hid[15] = -1; }
        ck(cudaMemcpyAsync(input.p, hx.data(), hx.size() * 4, cudaMemcpyHostToDevice, s));
        k::native_quantize_q8_1_wide(input.as<float>(), qpacked.p, ni, groups, s);
        ck(cudaMemsetAsync(qsource.p, 0x7e, qbytes, s));
        if (shared) ck(cudaMemcpyAsync(qsource.p, qpacked.p, qb, cudaMemcpyDeviceToDevice, s));
        else ck(cudaMemcpy2DAsync(qsource.p, xs, qpacked.p, qb, qb, groups, cudaMemcpyDeviceToDevice, s));
        ck(cudaMemcpyAsync(idsource.p, hid.data(), groups * 4, cudaMemcpyHostToDevice, s));
        for (Device* p : {&rg, &ru}) ck(cudaMemcpyAsync(p->p, blank.data(), outbytes, cudaMemcpyHostToDevice, s));
        k::pdl_scope() = false;
        launch(0, rg, ru);
        ck(cudaStreamSynchronize(s));
        std::vector<uint32_t> expected[2] = {blank, blank};
        ck(cudaMemcpy(expected[0].data(), rg.p, outbytes, cudaMemcpyDeviceToHost));
        ck(cudaMemcpy(expected[1].data(), ru.p, outbytes, cudaMemcpyDeviceToHost));
        for (int variant : {1, 2}) for (int graph_mode : {0, 1}) {
            reset();
            if (graph_mode) ck(cudaGraphLaunch(variant == 1 ? candidate.ex : product.ex, s));
            else { k::pdl_scope() = pdl; launch(variant, yg, yu); k::pdl_scope() = false; }
            ck(cudaStreamSynchronize(s));
            for (int bank = 0; bank < 2; ++bank) {
                std::vector<uint32_t> got(blank.size());
                ck(cudaMemcpy(got.data(), bank ? yu.p : yg.p, outbytes, cudaMemcpyDeviceToHost));
                for (std::size_t i = 0; i < got.size(); ++i) {
                    const bool valid = i >= guard && i < count + guard &&
                        (implicit_ids || hid[(i - guard) / no] >= 0);
                    float f; std::memcpy(&f, &got[i], 4);
                    if (got[i] != expected[bank][i] || (!valid && got[i] != sentinel) || (valid && !std::isfinite(f)))
                        throw std::runtime_error("pair mismatch ni=" + std::to_string(ni) + " no=" + std::to_string(no) +
                            " rows=" + std::to_string(rows) + " replay=" + std::to_string(replay) +
                            " variant=" + std::to_string(variant) + " index=" + std::to_string(i));
                    if (valid) ++compared;
                }
            }
        }
    }
    std::printf("PASS ni=%d no=%d rows=%d shared=%d implicit_ids=%d pdl=%d real=%d\n",
                ni, no, rows, int(shared), int(implicit_ids), int(pdl), int(gate_real != nullptr));
}

// The wide single-launch form (4 tokens x 16 experts in one 64-group launch): the per-token reference
// loop frozen as the oracle, the wide product compared against it byte for byte, plus the same dispatch,
// PDL-edge, guard-region, graph-replay and changed-data coverage as check_case.
void check_case_wide(int ni, int no, int rows, unsigned seed, bool pdl,
                     const void* gate_real = nullptr, const void* up_real = nullptr) {
    constexpr int experts = 512, tok = 4, groups = 64, groups16 = 16;
    const auto wb = k::native_mmvq_weight_bytes(12, ni, no), gs = wb + 144, us = wb + 288;
    const auto qb = k::native_q8_1_bytes(ni);
    const std::size_t count = std::size_t(groups) * no, outbytes = (count + 2 * guard) * 4;
    const std::size_t qbytes = qb * tok;
    auto g = weights(ni, no, experts, gs, seed, gate_real);
    auto u = weights(ni, no, experts, us, seed + 17, up_real);
    Device dg(g.size()), du(u.size()), input(std::size_t(ni) * tok * 4), qpacked(qbytes);
    Device qsource(qbytes), q(qbytes), idsource(groups * 4), ids(groups * 4);
    Device rg(outbytes), ru(outbytes), yg(outbytes), yu(outbytes);
    Stream stream; const auto s = stream.s;
    ck(cudaMemcpyAsync(dg.p, g.data(), g.size(), cudaMemcpyHostToDevice, s));
    ck(cudaMemcpyAsync(du.p, u.data(), u.size(), cudaMemcpyHostToDevice, s));
    ck(cudaStreamSynchronize(s));
    std::vector<uint32_t> blank(count + 2 * guard, sentinel);
    std::vector<int32_t> hid(groups);
    std::mt19937 rng(seed + 113);
    k::native_mmvq_set_rows(rows);
    auto reset = [&] {
        for (Device* p : {&yg, &yu}) ck(cudaMemcpyAsync(p->p, blank.data(), outbytes, cudaMemcpyHostToDevice, s));
        ck(cudaMemsetAsync(q.p, 0xff, qbytes, s));
        ck(cudaMemsetAsync(ids.p, 0xff, groups * 4, s));
    };
    auto produce_q = [&] {
        k::launch_pdl(produce, dim3(unsigned((qbytes + 255) / 256)), 256, 0, s,
                      qsource.as<uint8_t>(), q.as<uint8_t>(), qbytes, idsource.as<int32_t>(), ids.as<int32_t>(), groups);
    };
    // layout 0: the per-token loop - four 16-group launches, each token's own activation and ids.
    // layout 1: the wide form - one 64-group launch with the activation stride.
    auto launch = [&](int variant, int layout, Device& a, Device& b) {
        produce_q();
        if (layout == 0)
            for (int t = 0; t < tok; ++t)
                pair(variant, rows, dg.p, gs, du.p, us, q.as<uint8_t>() + std::size_t(t) * qb, 0,
                     a.as<float>() + guard + (std::size_t(t) * groups16) * no,
                     b.as<float>() + guard + (std::size_t(t) * groups16) * no,
                     ni, no, groups16, s, ids.as<int32_t>() + std::size_t(t) * groups16);
        else
            pair(variant, rows, dg.p, gs, du.p, us, q.p, qb, a.as<float>() + guard, b.as<float>() + guard,
                 ni, no, groups, s, ids.as<int32_t>());
    };
    const bool reused = k::q4_pair_reuse_enabled(ni, no, groups, rows);
    if (!reused) {
        // The wide entry declines wherever the reuse kernel does not apply; the caller then runs the
        // per-token loop. Prove the decline and stop: the wide comparisons below need the reuse kernel.
        Stream decline_stream;
        if (k::native_mmvq_grouped_pair_wide(12, dg.p, gs, du.p, us, q.p, qb, nullptr, nullptr,
                                             ni, no, groups, groups16, decline_stream.s,
                                             ids.as<int32_t>(), 1))
            throw std::runtime_error("wide entry accepted a shape the reuse kernel declines");
        std::printf("PASS wide ni=%d no=%d rows=%d pdl=%d real=%d declined=%d\n",
                    ni, no, rows, int(pdl), int(gate_real != nullptr), 1);
        return;
    }
    Graph candidate, product;
    for (int variant : {1, 3}) {
        k::pdl_scope() = pdl;
        ck(cudaStreamBeginCapture(s, cudaStreamCaptureModeThreadLocal));
        launch(variant, 1, yg, yu);
        auto& graph = variant == 1 ? candidate : product;
        graph.finish(s);
        k::pdl_scope() = false;
        if (variant == 1 && pdl && programmatic_edges(graph.g) == 0)
            throw std::runtime_error("PDL requested but wide candidate graph has no programmatic edge");
    }
    if ((pair_function(candidate.g) == pair_function(product.g)) != reused)
        throw std::runtime_error("wide product graph selected the wrong pair kernel");
    if (pdl && programmatic_edges(product.g) == 0)
        throw std::runtime_error("wide product graph lost its programmatic edge");
    for (int replay = 0; replay < 3; ++replay) {
        std::vector<float> hx(std::size_t(ni) * tok);
        for (std::size_t i = 0; i < hx.size(); ++i) {
            const float r = float(int(rng() % 20001) - 10000) / 4096.f;
            hx[i] = replay == 0 ? r : replay == 1 ? (i % 2 ? -1.f : 1.f) * (0.5f + std::abs(r)) : 0.f;
        }
        for (int i = 0; i < groups; ++i) hid[i] = replay == 0 ? int(rng() % experts) : (experts - 1 - i);
        if (replay != 0) { hid[3] = hid[2]; hid[5] = -1; hid[17] = -1; hid[63] = -1; }
        ck(cudaMemcpyAsync(input.p, hx.data(), hx.size() * 4, cudaMemcpyHostToDevice, s));
        k::native_quantize_q8_1_wide(input.as<float>(), qpacked.p, ni, tok, s);
        ck(cudaMemsetAsync(qsource.p, 0x7e, qbytes, s));
        ck(cudaMemcpyAsync(qsource.p, qpacked.p, qbytes, cudaMemcpyDeviceToDevice, s));
        ck(cudaMemcpyAsync(idsource.p, hid.data(), groups * 4, cudaMemcpyHostToDevice, s));
        // The oracle: the per-token loop on the frozen reference kernel, in its own buffers.
        for (Device* p : {&rg, &ru}) ck(cudaMemcpyAsync(p->p, blank.data(), outbytes, cudaMemcpyHostToDevice, s));
        k::pdl_scope() = false;
        launch(0, 0, rg, ru);
        ck(cudaStreamSynchronize(s));
        std::vector<uint32_t> expected[2] = {blank, blank};
        ck(cudaMemcpy(expected[0].data(), rg.p, outbytes, cudaMemcpyDeviceToHost));
        ck(cudaMemcpy(expected[1].data(), ru.p, outbytes, cudaMemcpyDeviceToHost));
        for (int variant : {1, 3}) for (int graph_mode : {0, 1}) {
            reset();
            if (graph_mode) ck(cudaGraphLaunch(variant == 1 ? candidate.ex : product.ex, s));
            else { k::pdl_scope() = pdl; launch(variant, 1, yg, yu); k::pdl_scope() = false; }
            ck(cudaStreamSynchronize(s));
            for (int bank = 0; bank < 2; ++bank) {
                std::vector<uint32_t> got(blank.size());
                ck(cudaMemcpy(got.data(), bank ? yu.p : yg.p, outbytes, cudaMemcpyDeviceToHost));
                for (std::size_t i = 0; i < got.size(); ++i) {
                    const bool valid = i >= guard && i < count + guard && hid[(i - guard) / no] >= 0;
                    float f; std::memcpy(&f, &got[i], 4);
                    if (got[i] != expected[bank][i] || (!valid && got[i] != sentinel) || (valid && !std::isfinite(f))) {
                        float gf, ef; std::memcpy(&gf, &got[i], 4); std::memcpy(&ef, &expected[bank][i], 4);
                        throw std::runtime_error("wide pair mismatch ni=" + std::to_string(ni) + " no=" + std::to_string(no) +
                            " rows=" + std::to_string(rows) + " replay=" + std::to_string(replay) +
                            " variant=" + std::to_string(variant) + " graph=" + std::to_string(graph_mode) +
                            " index=" + std::to_string(i) + " got=" + std::to_string(gf) +
                            " expected=" + std::to_string(ef) + " hid=" + std::to_string(hid[(i - guard) / no]));
                    }
                    if (valid) ++compared;
                }
            }
        }
    }
    std::printf("PASS wide ni=%d no=%d rows=%d pdl=%d real=%d\n",
                ni, no, rows, int(pdl), int(gate_real != nullptr));
}

} // namespace

int main() {
    try {
        int n = 0; if (cudaGetDeviceCount(&n) != cudaSuccess || n == 0) return 77;
        cudaDeviceProp prop{}; ck(cudaGetDeviceProperties(&prop, 0));
        const bool pdl = k::pdl_supported();
        std::printf("GPU %s sm_%d%d PDL=%d\n", prop.name, prop.major, prop.minor, int(pdl));
        for (int rows : {1, 2, 4, 8}) {
            for (int ni : {256, 512, 1536, 1792, 2048, 2304, 4096, 16384}) {
                // Product chooses one row above the small-K threshold.
                const int rr = ni >= 2048 ? 1 : rows;
                if (rows != 1 && ni >= 2048) continue;
                check_case(ni, 259, rr, true, false, unsigned(ni + rows), pdl);
                check_case(ni, 17, rr, false, false, unsigned(ni + rows + 101), pdl);
            }
            for (int no : {1, 3, 8, 1536}) check_case(1536, no, rows, false, true, unsigned(no + rows), pdl);
        }
        // The wide 64-group form, on the measured H20 shape only: the wide entry declines where the
        // reuse kernel does not apply, and the caller falls back to the per-token loop.
        for (int rows : {2, 4}) check_case_wide(1536, 1536, rows, unsigned(9000 + rows), pdl);
        {
            Stream decline_stream;
            if (k::native_mmvq_grouped_pair_wide(12, nullptr, 0, nullptr, 0, nullptr, 0, nullptr, nullptr,
                                                 1792, 1536, 64, 16, decline_stream.s, nullptr, 1))
                throw std::runtime_error("wide entry accepted a non-1536 width");
        }
        if (const char* path = std::getenv("STRATA_HY4_FLASH_GGUF")) {
            strata::GgufFile gg(path); bool found = false;
            for (const auto& t : gg.tensors()) {
                const auto pos = t.name.find(".ffn_gate_exps.weight");
                if (pos == std::string::npos || t.type != 12 || t.shape.size() != 3 || t.shape[2] < 16) continue;
                auto name = t.name; name.replace(pos + 5, 4, "up");
                const auto* up = gg.find(name);
                if (!up || up->type != t.type || up->shape != t.shape) throw std::runtime_error("real gate/up mismatch");
                for (int rows : {1, 2, 4, 8}) check_case(int(t.shape[0]), int(t.shape[1]), rows, true, false,
                    711, pdl, gg.tensor_data(t), gg.tensor_data(*up));
                for (int rows : {2, 4}) check_case_wide(int(t.shape[0]), int(t.shape[1]), rows,
                    711, pdl, gg.tensor_data(t), gg.tensor_data(*up));
                found = true; break;
            }
            if (!found) throw std::runtime_error("no real Q4_K pair");
        } else std::puts("SKIP real weights: STRATA_HY4_FLASH_GGUF is unset");
        std::printf("PASS %zu finite outputs bitwise equal; guards and skipped groups unchanged\n", compared);
        return 0;
    } catch (const std::exception& e) { std::fprintf(stderr, "FAIL: %s\n", e.what()); return 1; }
}
