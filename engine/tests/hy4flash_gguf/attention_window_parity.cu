// T2b: merged attention against independent queries, including device-length graph replay.
#include "strata/hy4flash_gguf/attention.hpp"
#include "strata/hy4flash_gguf/kv_cache.hpp"
#include "strata/artifact/gguf_reader.hpp"
#include "strata/kernels/native_mmvq.hpp"
#include "strata/kernels/pdl.hpp"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
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
    void put(const void* src, std::size_t n) { ck(cudaMemcpy(p, src, n, cudaMemcpyHostToDevice)); }
};
std::size_t values = 0;
void equal(float a, float b) {
    if (!std::isfinite(a) || !std::isfinite(b) || std::memcmp(&a, &b, 4))
        throw std::runtime_error("nonfinite or unequal attention value at check " + std::to_string(values));
    ++values;
}
std::vector<uint8_t> weights(int type, int ni, int no, int heads, unsigned seed) {
    std::vector<uint8_t> w(strata::kernels::native_mmvq_weight_bytes(type, ni, no) * heads);
    std::mt19937 rng(seed);
    for (auto& b : w) b = uint8_t(rng());
    const int block_bytes = type == 8 ? 34 : 144;
    for (std::size_t p = 0; p < w.size(); p += block_bytes) {
        const uint16_t d = uint16_t(0x1800 | (rng() & 1023));
        std::memcpy(w.data() + p, &d, 2);
        if (type != 8) std::memcpy(w.data() + p + 2, &d, 2);
    }
    return w;
}
void run(const char* name, int nope, int kr, int rot, int vh, int heads, int kt, int vt,
         const void* wk, const void* wv, cudaStream_t stream, int format = strata::hy4_flash::kKvF16) {
    using namespace strata::hy4_flash;
    if (!mla_per_query_supported(kr, rot, heads, true, kt, vt))
        throw std::runtime_error("unsupported test geometry");
    constexpr int stride = 4352, nsel = 64;
    const int qh = nope + rot;
    Device dk(strata::kernels::native_mmvq_weight_bytes(kt, nope, kr) * heads);
    Device dv(strata::kernels::native_mmvq_weight_bytes(vt, kr, vh) * heads);
    dk.put(wk, strata::kernels::native_mmvq_weight_bytes(kt, nope, kr) * heads);
    dv.put(wv, strata::kernels::native_mmvq_weight_bytes(vt, kr, vh) * heads);
    std::mt19937 rng(93021);
    std::uniform_real_distribution<float> dist(-.5f, .5f);
    std::vector<__half> keys(std::size_t(stride) * kr), pe(std::size_t(stride) * rot);
    for (auto& v : keys) v = __float2half(dist(rng));
    for (auto& v : pe) v = __float2half(dist(rng));
    std::vector<float> sinks(heads); for (auto& v : sinks) v = dist(rng);
    std::vector<int32_t> kp(stride); for (int j = 0; j < stride; ++j) kp[j] = j;
    Device keys_d(stride * kv_cache_lat_row_bytes(kr, format)), pe_d(pe.size() * 2), kp_d(kp.size() * 4), sink_d(heads * 4);
    if (format == kKvQ8) {
        const auto row_bytes = kv_cache_lat_row_bytes(kr, format);
        std::vector<uint8_t> packed(stride * row_bytes);
        for (int j = 0; j < stride; ++j) {
            auto* row = packed.data() + j * row_bytes;
            for (int c = 0; c < kr; ++c) row[c] = uint8_t(rng());
            for (int g = 0; g < kr / 32; ++g) {
                const __half scale = __float2half((1.f + std::fabs(dist(rng))) / 256.f);
                std::memcpy(row + kr + g * sizeof(scale), &scale, sizeof(scale));
            }
        }
        keys_d.put(packed.data(), packed.size());
    } else {
        keys_d.put(keys.data(), keys.size() * 2);
    }
    pe_d.put(pe.data(), pe.size() * 2);
    kp_d.put(kp.data(), kp.size() * 4); sink_d.put(sinks.data(), heads * 4);
    for (int width = 1; width <= 6; ++width) {
        const std::size_t na = std::size_t(kr) * heads * width, no = std::size_t(vh) * heads * width;
        std::vector<float> q(std::size_t(qh) * heads * width), qp(std::size_t(rot) * heads * width);
        for (auto& v : q) v = dist(rng); for (auto& v : qp) v = dist(rng);
        Device q_d(q.size()*4), qp_d(qp.size()*4), absorbed(na*4), ref_a(na*4), out(no*4), ref(no*4);
        q_d.put(q.data(), q.size()*4); qp_d.put(qp.data(), qp.size()*4);
        Device a_sc(mla_absorb_scratch_bytes(nope, width, heads));
        Device at_sc(mla_attention_scratch_bytes(kr, vh, heads, width, stride, nsel, true));
        Device pos_d(width*4), counts_d(width*4), heads_d(heads*width*4), sel_d(nsel*width*4);
        std::vector<int32_t> pos(width), counts(width), head_of(heads*width), sel(nsel*width);
        for (int j = 0; j < heads*width; ++j) head_of[j] = j/width;
        heads_d.put(head_of.data(), head_of.size()*4);
        MlaPerQuery pq{counts_d.as<int32_t>(), heads_d.as<int32_t>()};
        for (int depth : {124, 1500, 2044, 2048, 4096}) for (int window : {0, 512})
        for (bool sparse : {false, true}) {
            auto upload = [&](int base) {
                for (int t = 0; t < width; ++t) {
                    pos[t] = base+t; counts[t] = pos[t]+1;
                    for (int j = 0; j < nsel; ++j)
                        sel[t*nsel+j] = j%9 == 0 ? -1 : j*counts[t]/nsel;
                }
                pos_d.put(pos.data(), width*4); counts_d.put(counts.data(), width*4);
                sel_d.put(sel.data(), sel.size()*4);
            };
            const float scale = 1.f/std::sqrt(float(qh));
            auto merged = [&] {
                mla_absorb(q_d.as<float>(), dk.p, kt, qh, nope, kr, heads, width,
                           absorbed.as<float>(), a_sc.p, stream, &pq);
                mla_attention(absorbed.as<float>(), qp_d.as<float>(), keys_d.p, pe_d.p, format,
                    sink_d.as<float>(), dv.p, vt, kr, rot, vh, heads, width, counts.back(),
                    pos_d.as<int32_t>(), kp_d.as<int32_t>(), window, scale, out.as<float>(), at_sc.p,
                    stream, stride, nullptr, sparse ? sel_d.as<int32_t>() : nullptr,
                    sparse ? nsel : 0, nsel, &pq);
            };
            auto oracle = [&] {
                for (int t = 0; t < width; ++t) {
                    mla_absorb(q_d.as<float>()+std::size_t(t)*qh*heads, dk.p, kt, qh, nope, kr, heads, 1,
                        ref_a.as<float>()+std::size_t(t)*kr*heads, a_sc.p, stream);
                    mla_attention(ref_a.as<float>()+std::size_t(t)*kr*heads,
                        qp_d.as<float>()+std::size_t(t)*rot*heads, keys_d.p, pe_d.p, format,
                        sink_d.as<float>(), dv.p, vt, kr, rot, vh, heads, 1, counts[t],
                        pos_d.as<int32_t>()+t, kp_d.as<int32_t>(), window, scale,
                        ref.as<float>()+std::size_t(t)*vh*heads, at_sc.p, stream, stride, nullptr,
                        sparse ? sel_d.as<int32_t>()+t*nsel : nullptr, sparse ? nsel : 0, nsel);
                }
            };
            auto check = [&] {
                ck(cudaStreamSynchronize(stream));
                std::vector<float> a(na), b(na), c(no), d(no);
                ck(cudaMemcpy(a.data(), absorbed.p, na*4, cudaMemcpyDeviceToHost));
                ck(cudaMemcpy(b.data(), ref_a.p, na*4, cudaMemcpyDeviceToHost));
                ck(cudaMemcpy(c.data(), out.p, no*4, cudaMemcpyDeviceToHost));
                ck(cudaMemcpy(d.data(), ref.p, no*4, cudaMemcpyDeviceToHost));
                for (int h = 0; h < heads; ++h) for (int t = 0; t < width; ++t)
                    for (int k = 0; k < kr; ++k) equal(a[(h*width+t)*kr+k], b[(t*heads+h)*kr+k]);
                for (std::size_t j = 0; j < no; ++j) equal(c[j], d[j]);
            };
            upload(depth); merged(); oracle(); check();
            cudaGraph_t graph = nullptr; cudaGraphExec_t exec = nullptr;
            ck(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal)); merged();
            ck(cudaStreamEndCapture(stream, &graph)); ck(cudaGraphInstantiate(&exec, graph, nullptr, nullptr, 0));
            ck(cudaGraphDestroy(graph));
            for (int delta : {1, 5}) {
                upload(depth+delta); ck(cudaGraphLaunch(exec, stream)); oracle(); check();
            }
            ck(cudaGraphExecDestroy(exec));
            std::printf("PASS %s w=%d depth=%d window=%d sparse=%d eager+2replays\n",
                        name, width, depth, window, int(sparse));
        }
    }
}
}
int main() {
    int devices = 0; if (cudaGetDeviceCount(&devices) != cudaSuccess || !devices) return 77;
    try {
        strata::kernels::pdl_scope() = strata::kernels::pdl_supported();
        cudaStream_t stream; ck(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
        auto wk = weights(12, 512, 512, 64, 12), wv = weights(12, 512, 128, 64, 91);
        auto qk = weights(8, 512, 512, 64, 17), qv = weights(8, 512, 128, 64, 97);
        run("Q4_K", 512, 512, 64, 128, 64, 12, 12, wk.data(), wv.data(), stream);
        run("Q8_0", 512, 512, 64, 128, 64, 8, 8, qk.data(), qv.data(), stream);
        run("Q8_0 drafter with Q8_0 KV", 512, 512, 64, 128, 64, 8, 8, qk.data(), qv.data(), stream,
            strata::hy4_flash::kKvQ8);
        run("Q8_0/Q4_K", 512, 512, 64, 128, 64, 8, 12, qk.data(), wv.data(), stream);
        run("Q4_K/Q8_0", 512, 512, 64, 128, 64, 12, 8, wk.data(), qv.data(), stream);
        if (const char* path = std::getenv("STRATA_HY4_FLASH_GGUF")) {
            strata::GgufFile gg(path);
            const auto* k = gg.find("blk.0.attn_k_b.weight");
            const auto* v = gg.find("blk.0.attn_v_b.weight");
            if (!k || !v || k->shape.size() < 3 || v->shape.size() < 3)
                throw std::runtime_error("missing real attention weights");
            run("real weights", int(k->shape[0]), int(k->shape[1]), 64, int(v->shape[1]), int(k->shape[2]),
                int(k->type), int(v->type), gg.tensor_data(*k), gg.tensor_data(*v), stream);
        } else std::puts("SKIP real weights: set STRATA_HY4_FLASH_GGUF");
        ck(cudaStreamDestroy(stream)); std::printf("PASS %zu finite values identical\n", values); return 0;
    } catch (const std::exception& e) { std::fprintf(stderr, "FAIL: %s\n", e.what()); return 1; }
}
