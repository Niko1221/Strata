// metal/tests/window_cells_test.cpp - the park's K/V repack shuffle, model-free: the image holds exactly the
// bytes of the window (round-trip identity), nothing is written past it (the [cell]->[page] index mix-up that
// wrote 4x past the scratch is the bug this catches), and every unit lands at its page/head/in-page address.
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

// The real shuffle from src/core/conversation_snapshot.cpp (built into strata_metal_host).
namespace strata::core {
void window_cells(void* dst, const void* src, int64_t cells, int64_t page, int64_t heads, size_t unit,
                  bool to_image);
}

namespace {

int fails = 0;

void check(bool ok, const char* what) {
    if (ok) return;
    std::printf("FAIL: %s\n", what);
    ++fails;
}
void one(int64_t cells, int64_t page, int64_t heads, size_t unit, std::mt19937& rng) {
    const int64_t pages = (cells + page - 1) / page;
    std::vector<uint8_t> win((size_t) cells * heads * unit);
    for (auto& b : win) b = static_cast<uint8_t>(rng());
    const size_t img_size = (size_t) pages * heads * page * unit;
    const size_t pad = 64;
    std::vector<uint8_t> buf(pad + img_size + pad, 0xA5);
    uint8_t* img = buf.data() + pad;

    strata::core::window_cells(img, win.data(), cells, page, heads, unit, true);
    check(std::memcmp(buf.data(), std::vector<uint8_t>(pad, 0xA5).data(), pad) == 0 &&
          std::memcmp(buf.data() + pad + img_size, std::vector<uint8_t>(pad, 0xA5).data(), pad) == 0,
          "park wrote outside the image");
    for (int64_t pg = 0, p = 0; p < cells; ++pg, p += page)
        for (int64_t h = 0; h < heads; ++h)
            for (int64_t r = 0; r < page && p + r < cells; ++r) {
                const size_t wi = (size_t) (((p + r) * heads + h) * unit);
                const size_t ii = (size_t) (((pg * heads + h) * page + r) * unit);
                check(std::memcmp(img + ii, win.data() + wi, unit) == 0, "unit at wrong image address");
                if (fails) return;
            }
    std::vector<uint8_t> back(win.size(), 0);
    strata::core::window_cells(back.data(), img, cells, page, heads, unit, false);
    check(back == win, "restore is not the inverse of the park");
}

}  // namespace

int main() {
    std::mt19937 rng(20261019);
    one(5, 4, 2, 3, rng);      // trailing partial page, tiny unit
    one(0, 4, 2, 3, rng);      // nothing to park
    one(64, 16, 8, 512, rng);  // int8 cells: 512-byte unit, whole pages
    one(100, 16, 2, 512, rng); // partial last page at the real cell size
    one(33, 1, 3, 17, rng);    // page = 1: the shuffle is the identity
    if (fails) { std::printf("window_cells_test: %d FAILURES\n", fails); return 1; }
    std::printf("window_cells_test: PASS\n");
    return 0;
}
