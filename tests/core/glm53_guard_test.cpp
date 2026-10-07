// tests/core/glm53_guard_test.cpp - the architecture guard for the glm-dsa family (P1.S2, port round 1).
//
// The guard is the only thing standing between a GLM file and a graph written for a different model, so the
// test is not "does it accept a good file" but "does it refuse the ones that would load and be wrong".
// Synthetic metadata only: no model, no GPU, no gguf-py.
#include "gguf_fixture.hpp"

#include "strata/artifact/gguf_reader.hpp"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <filesystem>
#include <string>
#include <vector>

namespace fs = std::filesystem;
using strata::check_architecture;

namespace {
int g_fail = 0;
void check(bool ok, const std::string& what) {
    std::printf("  %-84s %s\n", what.c_str(), ok ? "ok" : "FAIL");
    if (!ok) ++g_fail;
}

struct TempDir {
    fs::path path;
    TempDir() {
        path = fs::temp_directory_path() /
               ("strata-glm53-guard-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
        fs::create_directories(path);
    }
    ~TempDir() {
        std::error_code ignored;
        fs::remove_all(path, ignored);
    }
};

// Measured on D:\models\GLM-5.3-colibri-int4-g64: 78 main layers, hidden 6144, 256 experts, 8 active,
// 64 heads, MLA q_lora_rank 2048 / kv_lora_rank 512, indexer top_k 2048 / key_length 128 / head_count 32.
std::vector<fixture::Kv> glm_keys() {
    return {fixture::str("general.architecture", "glm-dsa"),
            fixture::u32("glm-dsa.block_count", 78),
            fixture::u32("glm-dsa.embedding_length", 6144),
            fixture::u32("glm-dsa.expert_count", 256),
            fixture::u32("glm-dsa.expert_used_count", 8),
            fixture::u32("glm-dsa.attention.head_count", 64),
            fixture::u32("glm-dsa.attention.head_count_kv", 64),
            fixture::u32("glm-dsa.attention.q_lora_rank", 2048),
            fixture::u32("glm-dsa.attention.kv_lora_rank", 512),
            fixture::u32("glm-dsa.attention.indexer.head_count", 32),
            fixture::u32("glm-dsa.attention.indexer.key_length", 128),
            fixture::u32("glm-dsa.attention.indexer.top_k", 2048)};
}

std::string err_of(const fs::path& p) {
    strata::GgufFile g(p.string());
    return check_architecture(g);
}
}  // namespace

int main() {
    std::printf("glm53_guard_test\n");
    TempDir d;
    const std::vector<fixture::Tensor> one = {{"token_embd.weight", {6144, 8}, 1, 1}};

    const fs::path good = d.path / "good.gguf";
    fixture::write(good, glm_keys(), one);
    check(err_of(good).empty(), "the measured glm-dsa geometry passes");

    auto named = [&](const std::string& key, uint64_t value) {
        std::vector<fixture::Kv> kv = glm_keys();
        for (auto& k : kv)
            if (k.key == key) k.u = value;
        const fs::path p = d.path / (key + ".gguf");
        fixture::write(p, kv, one);
        return err_of(p);
    };

    // A file that carries the appended NextN block says 79, not 78. That is legal, not a mismatch.
    check(named("glm-dsa.block_count", 79).empty(), "block_count 79 (78 + the NextN block) is legal");

    check(named("glm-dsa.embedding_length", 4096).find("embedding_length") != std::string::npos,
          "hidden 4096 (GLM-5.3-Flash, a different graph) is refused by name");

    check(named("glm-dsa.expert_used_count", 10).find("expert_used_count") != std::string::npos,
          "expert_used_count 10 (the qwen4exp top-k) is refused");

    std::vector<fixture::Kv> missing = glm_keys();
    missing.erase(std::find_if(missing.begin(), missing.end(),
                               [](const fixture::Kv& k) { return k.key == "glm-dsa.attention.kv_lora_rank"; }));
    const fs::path miss = d.path / "missing.gguf";
    fixture::write(miss, missing, one);
    check(err_of(miss).find("missing glm-dsa.attention.kv_lora_rank") != std::string::npos,
          "a missing MLA rank is refused, not defaulted");

    // glm4moe is a real llama.cpp family with a real GGUF key set. It is not the graph this engine has, so
    // the file must be refused even though every value in it is a plausible number.
    std::vector<fixture::Kv> wrong = glm_keys();
    wrong[0] = fixture::str("general.architecture", "glm4moe");
    const fs::path bad = d.path / "wrong.gguf";
    fixture::write(bad, wrong, one);
    check(err_of(bad).find("this engine supports 'qwen4exp' and 'glm-dsa'") != std::string::npos,
          "a real but unsupported family (glm4moe) is refused with the list");

    // Asking for the qwen guard on a glm-dsa file is the mistake the dispatcher exists to prevent.
    {
        strata::GgufFile g(good.string());
        check(strata::check_architecture(g, strata::Qwen4ExpGuard{}).find("requires 'qwen4exp'") != std::string::npos,
              "the qwen guard still refuses a glm-dsa file by name");
    }

    // The qwen4exp path is unchanged: the guard still wants its own keys and names them.
    const fs::path qwen = d.path / "qwen.gguf";
    fixture::write(qwen, {fixture::str("general.architecture", "qwen4exp")}, one);
    check(err_of(qwen).find("missing qwen4exp.block_count") != std::string::npos,
          "the qwen4exp guard still reads its own keys");

    return g_fail ? 1 : 0;
}
