// engine/tests/core/ink_tail_skip_test.cpp - the opt-in router tail skip's spec parser and its two presets (--turbo, --turbo-max); host only, no GPU.
// The router itself runs on the card (cuda/ink_moe_cuda.cuh); its numbers are in phaseA-tests/rel0223/tailskip/RESULTS.md.
#include "cuda/ink_tail_skip.cuh"

#include <cstdio>
#include <string>

using namespace strata::ink;

static int g_fail = 0;
#define CHECK(c) do { if (!(c)) { std::fprintf(stderr, "FAIL line %d: %s\n", __LINE__, #c); ++g_fail; } } while (0)

static bool close_f(float a, float b) { return a - b < 1e-6f && b - a < 1e-6f; }

int main() {
    std::string err;
    InkTailCfg c;
    // off, empty and 0 are off
    for (const char* s : {"", "off", "0"}) { c.on = true; CHECK(ink_tail_parse(s, c, err)); CHECK(!c.on); }

    // --turbo: Pass 2 (visits 30..41), ranks 7-8 at <= 8 %, no renormalisation
    CHECK(ink_tail_parse("turbo", c, err));
    CHECK(c.on && c.first_rank == 7 && close_f(c.thr, 0.08f) && c.renorm == 0 && close_f(c.cap, 0.0f));
    uint64_t pass2 = 0;
    for (int v = 30; v <= 41; ++v) pass2 |= 1ull << v;
    CHECK(c.visit_mask == pass2);

    // --turbo-max: ranks 6-8 at <= 8 % on all 48 visits, renormalised
    CHECK(ink_tail_parse("turbo-max", c, err));
    CHECK(c.on && c.first_rank == 6 && close_f(c.thr, 0.08f) && c.renorm == 1);
    CHECK(c.visit_mask == (1ull << kInkTailVisits) - 1);
    CHECK(ink_tail_parse("turbo_max", c, err) && c.first_rank == 6);

    // the presets are the documented specs
    CHECK(std::string(ink_tail_preset("turbo")) == "thr=0.08,rank=7,visits=30-41,cap=0,renorm=0");
    CHECK(std::string(ink_tail_preset("turbo-max")) == "thr=0.08,rank=6,visits=all,cap=0,renorm=1");
    CHECK(ink_tail_preset("something") == nullptr);

    // explicit specs: defaults, every key, visit lists
    CHECK(ink_tail_parse("thr=0.05", c, err) && c.on && c.first_rank == 7 && c.renorm == 1);   // default visits 18-41
    uint64_t p12 = 0;
    for (int v = 18; v <= 41; ++v) p12 |= 1ull << v;
    CHECK(c.visit_mask == p12);
    CHECK(ink_tail_parse("thr=0.1,rank=5,visits=0-5+42-47,cap=0.08,renorm=0,stats=1,cold=1,mode=fb,compscale=0.5,calib=1,comp=a.bin,hot=h.bin,fb=f.bin", c, err));
    CHECK(c.first_rank == 5 && close_f(c.cap, 0.08f) && c.renorm == 0 && c.stats && c.cold_only && c.mode_fb && c.calib && close_f(c.comp_scale, 0.5f));
    CHECK(c.comp_path == "a.bin" && c.hot_path == "h.bin" && c.fb_path == "f.bin");
    uint64_t pc = 0;
    for (int v = 0; v <= 5; ++v) pc |= 1ull << v;
    for (int v = 42; v <= 47; ++v) pc |= 1ull << v;
    CHECK(c.visit_mask == pc);

    // refusals: nothing is switched on by a wrong value
    InkTailCfg bad;
    CHECK(!ink_tail_parse("rank=1", bad, err));          // rank 1 is never dropped
    CHECK(!ink_tail_parse("rank=9", bad, err));
    CHECK(!ink_tail_parse("thr=0.9", bad, err));
    CHECK(!ink_tail_parse("visits=40-50", bad, err));
    CHECK(!ink_tail_parse("visits=5-3", bad, err));
    CHECK(!ink_tail_parse("bogus=1", bad, err));

    if (g_fail) { std::fprintf(stderr, "%d check(s) failed\n", g_fail); return 1; }
    std::printf("ink_tail_skip_test: ok\n");
    return 0;
}
