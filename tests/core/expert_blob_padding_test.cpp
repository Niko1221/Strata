// tests/core/expert_blob_padding_test.cpp - Q36 plan Phase 4 (docs/Q36_MEMORY_COMPACTION.md).  The expert arena used
// to be memset to zero whole and every expert then written over it; Phase 4 keeps only the bytes expert_blob() does
// not write at zero.  This checks that the padding-only form leaves byte-identical memory to the whole memset, for a
// layout with and without stride padding and with an NVFP4 scale tail.  No GPU, no model.
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

namespace {
int g_fail = 0;

// The two forms place_experts uses (Phase 4 replaces the first with the second).  `written` is what expert_blob()
// fills - L.L.down_off + L.src_dsz, i.e. [gate | up | down] - `stride` the slot step, `tail` kMmqTail.  The written
// bytes are a distinct pattern per expert so a wrong bound shows up as a mismatch, not two zeroes agreeing.
void form_full_memset(std::vector<uint8_t>& b, size_t stride, size_t written, int64_t ne, size_t tail) {
    std::memset(b.data(), 0, stride * (size_t) ne + tail);
    for (int64_t e = 0; e < ne; ++e)
        for (size_t i = 0; i < written; ++i) b[(size_t) e * stride + i] = (uint8_t) (e + 1);
}
void form_padding_only(std::vector<uint8_t>& b, size_t stride, size_t written, int64_t ne, size_t tail) {
    std::memset(b.data(), 0xAA, b.size());   // garbage: the written bytes and the zeros must cover every byte
    for (int64_t e = 0; e < ne; ++e) {
        for (size_t i = 0; i < written; ++i) b[(size_t) e * stride + i] = (uint8_t) (e + 1);
        std::memset(b.data() + (size_t) e * stride + written, 0, stride - written);
    }
    std::memset(b.data() + (size_t) ne * stride, 0, tail);
}

void identical(size_t stride, size_t written, int64_t ne, size_t tail, const char* what) {
    std::vector<uint8_t> a(stride * (size_t) ne + tail), b(a.size());
    form_full_memset(a, stride, written, ne, tail);
    form_padding_only(b, stride, written, ne, tail);
    const bool same = a == b;
    std::printf("  %-58s %s  (stride=%zu written=%zu ne=%lld tail=%zu)\n", what, same ? "ok" : "FAIL", stride,
                written, (long long) ne, tail);
    if (!same) ++g_fail;
}
}  // namespace

int main() {
    std::printf("expert_blob_padding_test\n");
    // a layout with stride padding, the Q4_K/Q6_K-ish shape (written < stride)
    identical(512, 480, 4, 4096, "padding present: zero [written, stride) and the tail");
    // no padding: stride == written, so the per-expert clear is a no-op and only the tail matters
    identical(480, 480, 4, 4096, "no padding (stride == written)");
    // an NVFP4 layout: 16 scale bytes between `written` (down_off + src_dsz) and the aligned stride
    identical(528, 512, 3, 4096, "a 16-byte scale tail inside [written, stride)");
    // one expert and no tail
    identical(256, 256, 1, 0, "one expert, no tail");
    std::printf(g_fail ? "expert_blob_padding_test: FAILED\n" : "expert_blob_padding_test: all passed\n");
    return g_fail ? 1 : 0;
}
