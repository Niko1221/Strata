// engine/tests/core/kv_dtype_guard_test.cpp - Unit test for #6681 KV dtype guard truth table (no GPU required)
// MIT License - Copyright (c) 2026 Strata contributors
#include "ink_batch.h"

#include <cstdio>
#include <string>

using namespace ink2;

namespace {
int failed = 0;
#define CHECK(expr)                                                                         \
    do {                                                                                    \
        if (!(expr)) {                                                                      \
            std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #expr);            \
            ++failed;                                                                       \
        }                                                                                   \
    } while (0)
}  // namespace

int main() {
    CHECK(!kv_dtype_is_int_quant(KvDtype::FP16));
    CHECK(!kv_dtype_is_int_quant(KvDtype::FP8_E4M3));
    CHECK(kv_dtype_is_int_quant(KvDtype::K8V8));
    CHECK(kv_dtype_is_int_quant(KvDtype::K8V4));

    CHECK(std::string(kv_dtype_name(KvDtype::FP16)) == "fp16");
    CHECK(std::string(kv_dtype_name(KvDtype::FP8_E4M3)) == "fp8");
    CHECK(std::string(kv_dtype_name(KvDtype::K8V8)) == "k8v8");
    CHECK(std::string(kv_dtype_name(KvDtype::K8V4)) == "k8v4");

    if (failed != 0) {
        std::fprintf(stderr, "kv_dtype_guard_test: %d check(s) failed\n", failed);
        return 1;
    }
    std::printf("kv_dtype_guard_test: OK\n");
    return 0;
}
