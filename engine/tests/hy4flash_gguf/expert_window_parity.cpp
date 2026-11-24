// Exact window matvecs versus the one-token grouped calls on the same Q8_1 bytes.
// Synthetic cases always run; STRATA_HY4_FLASH_GGUF adds real gate/up and both down types.
#include "strata/artifact/gguf_reader.hpp"
#include "strata/kernels/native_mmvq.hpp"
#include "strata/kernels/pdl.hpp"
#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <stdexcept>
#include <vector>

namespace {
void ck(cudaError_t e) { if (e != cudaSuccess) throw std::runtime_error(cudaGetErrorString(e)); }
struct Device {
    void* p = nullptr;
    explicit Device(std::size_t n) { ck(cudaMalloc(&p, n)); }
    ~Device() { cudaFree(p); }
    Device(const Device&) = delete;
    template<class T> T* as() { return static_cast<T*>(p); }
};
std::size_t values = 0;
void compare(const std::vector<float>& a, const std::vector<float>& b) {
    for (std::size_t i = 0; i < a.size(); ++i)
        if (!std::isfinite(a[i]) || !std::isfinite(b[i]) || std::memcmp(&a[i], &b[i], 4))
            throw std::runtime_error("nonfinite or unequal output at " + std::to_string(i));
    values += a.size();
}
void run(const char* name, int type, int ni, int no, const void* wa, const void* wb, cudaStream_t stream) {
    using namespace strata::kernels;
    const auto ws = native_mmvq_weight_bytes(type, ni, no);
    constexpr int experts = 16, k = 4;
    Device w(ws * experts), w2(ws * experts);
    ck(cudaMemcpy(w.p, wa, ws * experts, cudaMemcpyHostToDevice));
    if (wb) ck(cudaMemcpy(w2.p, wb, ws * experts, cudaMemcpyHostToDevice));
    std::mt19937 rng(95317);
    std::uniform_real_distribution<float> dist(-2.f, 2.f);
    for (int width = 1; width <= 4; ++width) for (int shared : {0, 2, 4}) {
        const int groups = width * k;
        std::vector<int32_t> ids;
        for (int t = 0; t < width; ++t)
            for (int p = 0; p < k; ++p) ids.push_back(p < shared ? p : shared + t * (k - shared) + p - shared);
        std::sort(ids.begin(), ids.end());
        std::vector<float> inputs(std::size_t(ni) * groups);
        for (float& v : inputs) v = dist(rng);
        const auto xs = native_q8_1_bytes(ni, 1), bytes = std::size_t(no) * groups * 4;
        Device in(inputs.size() * 4), q(xs * groups), di(groups * 4), out(bytes), out2(bytes), ref(bytes), ref2(bytes);
        ck(cudaMemcpy(in.p, inputs.data(), inputs.size() * 4, cudaMemcpyHostToDevice));
        ck(cudaMemcpy(di.p, ids.data(), groups * 4, cudaMemcpyHostToDevice));
        native_quantize_q8_1_wide(in.as<float>(), q.p, ni, groups, stream);
        auto oracle = [&] {
            for (int g = 0; g < groups; ++g) {
                native_mmvq_grouped_context(type, w.p, ws, q.as<char>() + g * xs, xs,
                    ref.as<float>() + g * no, ni, no, 1, stream, di.as<int32_t>() + g, 1);
                if (wb) native_mmvq_grouped_context(type, w2.p, ws, q.as<char>() + g * xs, xs,
                    ref2.as<float>() + g * no, ni, no, 1, stream, di.as<int32_t>() + g, 1);
            }
        };
        auto window = [&] {
            if (!native_mmvq_window(type, w.p, ws, q.p, xs, out.as<float>(), ni, no, groups, stream,
                                    di.as<int32_t>(), wb ? w2.p : nullptr, ws, wb ? out2.as<float>() : nullptr))
                throw std::runtime_error("window refused supported case");
        };
        oracle(); window(); ck(cudaStreamSynchronize(stream));
        std::vector<float> a(bytes / 4), b(bytes / 4);
        auto check = [&] {
            ck(cudaMemcpy(a.data(), out.p, bytes, cudaMemcpyDeviceToHost));
            ck(cudaMemcpy(b.data(), ref.p, bytes, cudaMemcpyDeviceToHost)); compare(a, b);
            if (wb) {
                ck(cudaMemcpy(a.data(), out2.p, bytes, cudaMemcpyDeviceToHost));
                ck(cudaMemcpy(b.data(), ref2.p, bytes, cudaMemcpyDeviceToHost)); compare(a, b);
            }
        };
        check();
        // Replay with different expert ids proves that the grouping is read on the device each time.
        cudaGraph_t graph = nullptr; cudaGraphExec_t exec = nullptr;
        ck(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal)); window();
        ck(cudaStreamEndCapture(stream, &graph)); ck(cudaGraphInstantiate(&exec, graph, nullptr, nullptr, 0));
        ck(cudaGraphDestroy(graph));
        for (auto& id : ids) id = experts - 1 - id;
        std::sort(ids.begin(), ids.end());
        ck(cudaMemcpy(di.p, ids.data(), groups * 4, cudaMemcpyHostToDevice));
        oracle(); ck(cudaGraphLaunch(exec, stream)); ck(cudaStreamSynchronize(stream)); check();
        ck(cudaGraphExecDestroy(exec));
        std::printf("PASS %s width=%d shared=%d%% eager+graph\n", name, width, shared * 25);
    }
}
std::vector<uint8_t> synthetic(int type, int ni, int no, unsigned seed) {
    const int block = type == 12 ? 144 : 210;
    std::vector<uint8_t> w(strata::kernels::native_mmvq_weight_bytes(type, ni, no) * 16);
    std::mt19937 rng(seed);
    for (auto& b : w) b = uint8_t(rng());
    for (std::size_t p = 0; p < w.size(); p += block) {
        const uint16_t d = uint16_t(0x2400 | (rng() & 1023));
        std::memcpy(w.data() + p + (type == 12 ? 0 : 208), &d, 2);
        if (type == 12) std::memcpy(w.data() + p + 2, &d, 2);
    }
    return w;
}
}
int main() {
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) return 77;
    try {
        strata::kernels::native_mmvq_set_q6_c50(true);
        strata::kernels::pdl_scope() = strata::kernels::pdl_supported();
        cudaStream_t stream; ck(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
        for (int type : {12, 14}) for (int ni : {1536, 3072}) {
            const int no = 259; // partial final row block as well as multiple K iterations
            auto w = synthetic(type, ni, no, 71), w2 = synthetic(type, ni, no, 97);
            run(type == 12 ? "synthetic Q4 pair" : "synthetic Q6", type, ni, no, w.data(),
                type == 12 ? w2.data() : nullptr, stream);
        }
        if (const char* path = std::getenv("STRATA_HY4_FLASH_GGUF")) {
            strata::GgufFile gg(path);
            int seen = 0;
            for (const auto& t : gg.tensors()) {
                const bool gu = t.name.find(".ffn_gate_exps.weight") != std::string::npos && t.type == 12 && !(seen & 1);
                const bool dn4 = t.name.find(".ffn_down_exps.weight") != std::string::npos && t.type == 12 && !(seen & 2);
                const bool dn6 = t.name.find(".ffn_down_exps.weight") != std::string::npos && t.type == 14 && !(seen & 4);
                if (!gu && !dn4 && !dn6) continue;
                if (t.shape.size() < 3 || t.shape[2] < 16) throw std::runtime_error("too few experts");
                const void* up = nullptr;
                if (gu) {
                    std::string name = t.name; name.replace(name.find("gate_exps"), 9, "up_exps");
                    const auto* u = gg.find(name);
                    if (!u || u->type != t.type || u->shape != t.shape) throw std::runtime_error("up shape");
                    up = gg.tensor_data(*u);
                }
                run(t.name.c_str(), int(t.type), int(t.shape[0]), int(t.shape[1]), gg.tensor_data(t), up, stream);
                seen |= gu ? 1 : dn4 ? 2 : 4;
            }
            if (seen != 7) throw std::runtime_error("missing real expert types");
        } else std::puts("SKIP real weights: set STRATA_HY4_FLASH_GGUF");
        ck(cudaStreamDestroy(stream));
        std::printf("PASS %zu finite values identical\n", values);
        return 0;
    } catch (const std::exception& e) { std::fprintf(stderr, "FAIL: %s\n", e.what()); return 1; }
}
