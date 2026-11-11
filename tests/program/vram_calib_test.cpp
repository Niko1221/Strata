// tests/program/vram_calib_test.cpp - STRATA_VRAM_CALIB (#5571): the measured reserve's host arithmetic (no GPU).
#include "strata/program/vram_calib.hpp"

#include <cstdio>
#include <string>
#include <vector>

using namespace strata::program::vram_calib;

namespace {
int g_fail = 0;
void check(bool ok, const std::string& what) {
    if (ok) return;
    std::fprintf(stderr, "FAIL: %s\n", what.c_str());
    ++g_fail;
}
constexpr long long MiB = 1ll << 20;
}  // namespace

int main() {
    // the key: stable, and it changes with the binary, any argument or any sizing variable, but not with the calibration's own
    const std::vector<std::string> args = {"--serve", "--prefill", "16384"};
    const std::vector<std::string> env = {"STRATA_A=1", "STRATA_B=2"};
    const std::string k = file_key("123:456", args, env);
    check(k.size() == 16, "the key is 16 hex digits");
    check(k == file_key("123:456", args, env), "the key is stable");
    check(k != file_key("123:457", args, env), "a rebuilt binary measures again");
    check(k != file_key("123:456", {"--serve", "--prefill", "8192"}, env), "another argument measures again");
    check(k != file_key("123:456", args, {"STRATA_A=1", "STRATA_B=3"}), "another variable measures again");
    check(file_key("x", {"ab", "c"}, {}) != file_key("x", {"a", "bc"}, {}), "arguments are separated");

    char e0[] = "STRATA_B=2", e1[] = "PATH=/bin", e2[] = "STRATA_A=1", e3[] = "STRATA_VRAM_CALIB=1", e4[] = "STRATA_VRAM_CALIB_MARGIN_MIB=256";
    char* envp[] = {e0, e1, e2, e3, e4, nullptr};
    const std::vector<std::string> se = sizing_env(envp);
    check(se == env, "sizing_env keeps the STRATA_* variables, sorted, without PATH or the calibration's own switches");
    char m0[] = "STRATA_B=2", m1[] = "STRATA_TRACE=1", m2[] = "STRATA_A=1", m3[] = "STRATA_DECODE_TIMING=1", m4[] = "STRATA_VERIFY_NODES=2",
         m5[] = "STRATA_DUMMY_NODES=8", m6[] = "STRATA_PREFILL_TIMING=1", m7[] = "STRATA_DISPATCH_REPORT=/tmp/d.json";
    char* envm[] = {m0, m1, m2, m3, m4, m5, m6, m7, nullptr};
    check(sizing_env(envm) == env, "the measurement-only switches do not change the key");
    check(measure_only("STRATA_TRACE=1") && !measure_only("STRATA_TRACE_X=1") && !measure_only("STRATA_TRACEX=1"),
          "measure_only matches the whole name");
    check(!measure_only("STRATA_VERIFY_PROFILE=1"), "a switch that allocates (the verifier's profile buffers) is not measurement-only");
    check(binary_key("123:456") == binary_key("123:456") && binary_key("123:456") != binary_key("123:457"),
          "the binary's file follows the binary alone");
    check(binary_key("123:456").rfind("binary-", 0) == 0 && binary_key("123:456").size() == 7 + 16, "binary-<16 hex>");

    // the file round-trips
    const std::vector<Card> cards = {{0, 361 * MiB, 104 * MiB}, {1, 556 * MiB, 131 * MiB}};
    const std::vector<Card> back = parse(format(cards));
    check(back.size() == 2 && back[0].dev == 0 && back[0].need == 361 * MiB && back[0].grow == 104 * MiB &&
          back[1].dev == 1 && back[1].need == 556 * MiB && back[1].grow == 131 * MiB, "format/parse round-trip");
    check(parse("").empty() && parse("garbage").empty(), "an empty or broken file is no calibration");

    // the reserve: need + grow rounded up to a MiB, plus the margin; the numbers measured on two RX 6900 XT (#5571)
    check(reserve_mib(cards, 0, kDefaultMarginMib) == 361 + 104 + 192, "card 0: 361 + 104 + 192");
    check(reserve_mib(cards, 1, kDefaultMarginMib) == 556 + 131 + 192, "card 1: 556 + 131 + 192");
    check(reserve_mib(cards, 2, kDefaultMarginMib) == -1, "a card without a measurement keeps the old sizing");
    check(reserve_mib({{0, MiB + 1, 0}}, 0, 0) == 2, "rounded up, not down");
    check(reserve_mib({{0, -1, 0}}, 0, 192) == -1, "an unmeasured need is no calibration");
    check(reserve_mib({{0, 10 * MiB, -5}}, 0, 0) == 10, "a negative growth counts as none");

    // need: free when the cache was sized - free now - the cache, never negative, unknown without both figures
    check(need_bytes(14000 * MiB, 500 * MiB, 13000 * MiB) == 500 * MiB, "need = mark - now - cache");
    check(need_bytes(1000 * MiB, 900 * MiB, 200 * MiB) == 0, "need never negative");
    check(need_bytes(-1, 500, 0) == -1 && need_bytes(500, -1, 0) == -1, "need unknown without both figures");

    // growth: the lowest free figure after any request, and it never shrinks
    long long grow = 0, low = 500 * MiB;
    check(update_grow(grow, 500 * MiB, low, 450 * MiB) && grow == 50 * MiB, "a request that lowers free grows it");
    check(!update_grow(grow, 500 * MiB, low, 480 * MiB) && grow == 50 * MiB, "a higher free figure later changes nothing");
    check(!update_grow(grow, 500 * MiB, low, -1) && grow == 50 * MiB, "a failed reading changes nothing");
    long long kept = 80 * MiB, low2 = 500 * MiB;
    check(!update_grow(kept, 500 * MiB, low2, 450 * MiB) && kept == 80 * MiB, "a growth kept from the file is not lowered");

    // the margin: a number or a tier, anything else the default
    check(margin_mib(nullptr) == 192 && margin_mib("") == 192, "unset or empty: the default 192");
    check(margin_mib("tight") == 96 && margin_mib("normal") == 192 && margin_mib("safe") == 384, "the three tiers");
    check(margin_mib("256") == 256 && margin_mib("0") == 0, "a number of MiB");
    check(margin_mib("-5") == 192 && margin_mib("12x") == 192 && margin_mib("huge") == 192, "a bad value falls back to the default");

    if (g_fail == 0) std::printf("vram_calib_test: all checks passed\n");
    return g_fail == 0 ? 0 : 1;
}
