#include "../../src/qwen35/q36_kv_cache.cuh"
#include <cstdlib>
#include <vector>
using namespace strata::qwen35;
static void ck(cudaError_t e) { if (e != cudaSuccess) std::abort(); }
int main() {
    // Byte-lifecycle only; production attention/MTP arithmetic remains a separate qualification.
    for (auto type : {Q36KvType::Fp32, Q36KvType::Fp16}) {
        const auto layout = make_kv_layout(type, 513, 2, 256);
        void *a = nullptr, *b = nullptr;
        ck(cudaMalloc(&a, layout.allocation_bytes)); ck(cudaMalloc(&b, layout.allocation_bytes));
        ck(cudaMemset(a, 0x5a, layout.allocation_bytes)); ck(cudaMemset(b, 0xa5, layout.allocation_bytes));
        for (size_t prefix : {size_t(0), size_t(1), size_t(257), size_t(513)}) {
            ck(cudaMemset(b, 0xa5, layout.allocation_bytes));
            ck(q36_copy_kv_prefix({b, layout}, {a, layout}, prefix, nullptr));
            std::vector<unsigned char> bytes(layout.allocation_bytes);
            ck(cudaMemcpy(bytes.data(), b, bytes.size(), cudaMemcpyDeviceToHost));
            for (size_t head = 0; head < layout.kv_heads; ++head)
                for (size_t i = 0; i < layout.head_stride; ++i)
                    if (bytes[head * layout.head_stride + i] != (i < layout.prefix_bytes(prefix) ? 0x5a : 0xa5)) std::abort();
        }
        ck(q36_clear_kv({b, layout}, nullptr));
        std::vector<unsigned char> bytes(layout.allocation_bytes);
        ck(cudaMemcpy(bytes.data(), b, bytes.size(), cudaMemcpyDeviceToHost));
        for (auto value : bytes) if (value) std::abort();
        ck(cudaFree(a)); ck(cudaFree(b));
    }
}
