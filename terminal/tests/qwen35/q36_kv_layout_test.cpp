#include "strata/qwen35/kv_layout.hpp"
#include <cstdlib>
#include <iostream>
#include <limits>
using namespace strata::qwen35;
static void check(bool value) { if (!value) std::abort(); }
template<class F> static void rejects(F f) {
    bool rejected = false;
    try { f(); } catch (const std::exception&) { rejected = true; }
    check(rejected);
}
int main() {
    for (size_t cells : {size_t(1), size_t(257), size_t(98304), size_t(131072)}) {
        auto f32 = make_kv_layout(Q36KvType::Fp32, cells, 2, 256);
        auto f16 = make_kv_layout(Q36KvType::Fp16, cells, 2, 256);
        check(f32.allocation_bytes == cells * 2048 && f32.padding_bytes == 0);
        check(f16.allocation_bytes == cells * 1024 && f16.padding_bytes == 0);
        check(f32.prefix_bytes(cells) == cells * 1024);
        require_kv_identity(f32, f32, 0);
        rejects([&] { require_kv_identity(f32, f16, 1); });
        rejects([&] { f32.prefix_bytes(cells + 1); });
        auto other = f32; ++other.version;
        rejects([&] { require_kv_identity(f32, other, 1); });
        other = f32; ++other.scale_schema;
        rejects([&] { require_kv_identity(f32, other, 1); });
        const auto q8 = make_kv_layout(Q36KvType::Q8_0, cells, 2, 256);
        check(!q8.executable() && q8.allocation_bytes == cells * 544);
        check(q8.payload_bytes + q8.block_scale_bytes == q8.allocation_bytes);
        const auto nv = make_kv_layout(Q36KvType::Nvfp4, cells, 2, 256);
        check(!nv.executable() && nv.block_scale_offset % 256 == 0 && nv.outer_scale_offset % 256 == 0);
        check(nv.allocation_bytes == nv.payload_bytes + nv.block_scale_bytes + nv.outer_scale_bytes + nv.padding_bytes);
        if (cells == 131072) { check(f32.allocation_bytes * 22 == 5905580032ULL); check(nv.allocation_bytes * 22 == 853540864ULL); }
        rejects([&] { require_kv_identity(q8, q8, 1); });
    }
    for (auto type : {Q36KvType::Fp32, Q36KvType::Fp16, Q36KvType::Q8_0, Q36KvType::Nvfp4}) {
        rejects([&] { make_kv_layout(type, 0, 2, 256); });
        rejects([&] { make_kv_layout(type, 1, 0, 256); });
        rejects([&] { make_kv_layout(type, std::numeric_limits<size_t>::max(), 2, 256); });
    }
    rejects([] { make_kv_layout(Q36KvType::Q8_0, 1, 2, 255); });
    rejects([] { make_kv_layout(Q36KvType::Nvfp4, 1, 2, 255); });
    rejects([] { kv_align(std::numeric_limits<size_t>::max(), 256); });
    rejects([] { kv_align(7, 3); });
    std::cout << "Q36 KV layout and identity fixtures passed (proposal codecs are not executable)\n";
}
