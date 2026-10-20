// NVIDIA architecture defaults: dispatch, opt-out and explicit user overrides; no GPU required.
#include "strata/core/nv_defaults.hpp"
#include <cstdio>
#include <cstdlib>
#include <string>
static int fails = 0;
#define CHECK(c) do { if (!(c)) { std::printf("FAIL line %d: %s\n", __LINE__, #c); ++fails; } } while (0)
static void put(const char* k, const char* v) {
#if defined(_WIN32)
    _putenv_s(k, v ? v : "");
#else
    if (v) setenv(k, v, 1); else unsetenv(k);
#endif
}
int main() {
    using namespace strata::core;
    put("STRATA_EMULATE_CC", nullptr);
    for (const char* k : {"STRATA_SM70_PERF", "STRATA_SM60_PERF", "STRATA_SM86_PERF"}) put(k, nullptr);
    const auto table = nv_default_env(7, 0);
    CHECK(table.size() == 6);
    for (const auto& kv : table) put(kv.first.c_str(), nullptr);
    CHECK(nv_default_env(7, 5).empty());
    CHECK(nv_default_env(7, 2).empty());
    CHECK(nv_default_env(8, 0).empty());
    CHECK(nv_default_env(8, 6).size() == 9);
    CHECK(nv_default_env(6, 0).size() == 7);
    put("STRATA_SM70_PERF", "0");
    CHECK(nv_default_env(7, 0).empty());
    CHECK(apply_nv_defaults(7, 0).empty());
    for (const auto& kv : table) CHECK(std::getenv(kv.first.c_str()) == nullptr);
    CHECK(!nv_default_env(8, 6).empty()); // opt-out affects only Volta
    put("STRATA_SM70_PERF", nullptr);
    put("STRATA_SIDE_MIXER", "0");
    put("STRATA_GR_UP75", "custom");
    const auto set = apply_nv_defaults(7, 0);
    CHECK(set.size() == 4);
    CHECK(std::string(std::getenv("STRATA_SIDE_MIXER")) == "0");
    CHECK(std::string(std::getenv("STRATA_GR_UP75")) == "custom");
    for (const auto& kv : table) {
        if (kv.first == "STRATA_SIDE_MIXER" || kv.first == "STRATA_GR_UP75") continue;
        const char* v = std::getenv(kv.first.c_str());
        CHECK(v && std::string(v) == kv.second);
    }
    CHECK(apply_nv_defaults(7, 0).empty()); // idempotent
    for (const auto& kv : table) put(kv.first.c_str(), nullptr);
    return fails ? 1 : 0;
}
