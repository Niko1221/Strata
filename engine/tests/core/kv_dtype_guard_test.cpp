// #6681: the truth table behind the hip/metal/sycl init guard. fp16 and fp8 every backend; the integer quantized
// formats (k8v8/k8v4) only where the pool really stores them (CPU, CUDA) — the fp16-only pools reject them at init
// instead of silently allocating fp16-sized pages.
#include "ink_batch.h"

#include <cstdio>
#include <string>

using namespace ink2;

namespace {
int failed = 0;
#define CHECK(x) do { if (!(x)) { std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #x); ++failed; } } while (0)
}  // namespace

int main() {
    CHECK(!kv_dtype_is_int_quant(KvDtype::FP16));
    CHECK(!kv_dtype_is_int_quant(KvDtype::FP8_E4M3));
    CHECK(kv_dtype_is_int_quant(KvDtype::K8V8));
    CHECK(kv_dtype_is_int_quant(KvDtype::K8V4));
    // the names the reject message and the bench line print (the strings the user sees after #6681's guard fires)
    CHECK(std::string(kv_dtype_name(KvDtype::K8V8)) == "k8v8");
    CHECK(std::string(kv_dtype_name(KvDtype::K8V4)) == "k8v4");
    CHECK(std::string(kv_dtype_name(KvDtype::FP8_E4M3)) == "fp8");
    if (failed) { std::fprintf(stderr, "%d check(s) failed\n", failed); return 1; }
    std::printf("ok\n");
    return 0;
}
