// CPU-only test of the config derivation.  g++ -std=c++17 -I../src/qwen36 -I$STRATA/include config_test.cpp && ./a.out shape.gguf
#include <cstdio>
#include "qwen36_config.hpp"
static int fails = 0;
#define CHECK(c) do { if (!(c)) { std::printf("FAIL line %d: %s\n", __LINE__, #c); ++fails; } } while (0)
int main(int argc, char** argv) {
    auto m = strata::GgufModel::open(argv[1]);
    q36::Config c = q36::load_config(m);
    std::printf("%s\n", c.describe().c_str());
    CHECK(c.arch == "qwen35moe"); CHECK(c.n_layer == 4); CHECK(c.n_gdn == 3 && c.n_attn == 1);
    CHECK(c.is_gdn[0] && c.is_gdn[1] && c.is_gdn[2] && !c.is_gdn[3]);
    CHECK(c.n_embd == 256 && c.n_vocab == 512);
    CHECK(c.ssm_hk == 2 && c.ssm_hv == 4 && c.ssm_S == 128 && c.ssm_conv == 4 && c.ssm_qkv == 1024 && c.ssm_inner == 512);
    CHECK(c.n_head == 4 && c.n_head_kv == 2 && c.head_dim == 128 && c.rot_dim == 32);
    CHECK(c.n_expert == 8 && c.n_used == 2 && c.n_ff_exp == 64 && c.n_ff_shexp == 64);
    CHECK(c.rope_theta == 1e7f);
    std::printf(fails ? "config_test: %d FAILED\n" : "config_test: all passed (%d failures)\n", fails);
    return fails != 0;
}
