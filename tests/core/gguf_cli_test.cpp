// Exercise the actual CLI entry point, including its return code, without a shell or a GPU.
#define main strata_gguf_main
#include "../../src/artifact/gguf_reader.cpp"
#undef main

#include "gguf_fixture.hpp"

#include <chrono>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <limits>

namespace {
struct TempDir {
    std::filesystem::path path = std::filesystem::temp_directory_path() /
        ("strata-gguf-cli-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    TempDir() { std::filesystem::create_directories(path); }
    ~TempDir() {
        std::error_code ignored;
        std::filesystem::remove_all(path, ignored);
    }
};

// Patch one field of the sole tensor directory entry while keeping the file's real payload unchanged.
void patch_tensor(const std::filesystem::path& path, bool offset, uint64_t value) {
    std::ifstream input(path, std::ios::binary);
    std::vector<uint8_t> bytes((std::istreambuf_iterator<char>(input)), {});
    input.close();
    strata::Cursor c(bytes.data(), bytes.size());
    (void) c.read<uint32_t>(); // magic
    (void) c.read<uint32_t>(); // version
    (void) c.read<uint64_t>(); // tensor count
    const uint64_t n_kv = c.read<uint64_t>();
    for (uint64_t i = 0; i < n_kv; ++i) {
        (void) c.str();
        (void) strata::read_value(c, (strata::MetaType) c.read<uint32_t>());
    }
    (void) c.str();
    const uint32_t nd = c.read<uint32_t>();
    size_t pos = c.pos(); // first dimension
    if (offset) {
        for (uint32_t i = 0; i < nd; ++i) (void) c.read<uint64_t>();
        (void) c.read<uint32_t>(); // type
        pos = c.pos();
    }
    std::memcpy(bytes.data() + pos, &value, sizeof value);
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    out.write(reinterpret_cast<const char*>(bytes.data()), (std::streamsize) bytes.size());
    if (!out) throw std::runtime_error("cannot patch fixture");
}

int run(const std::filesystem::path& path, bool check = true) {
    std::string program = "strata-gguf", file = path.string(), flag = "--check";
    char* argv[] = {program.data(), file.data(), flag.data(), nullptr};
    return strata_gguf_main(check ? 3 : 2, argv);
}
} // namespace

int main() {
    try {
        TempDir dir;
        const auto path = dir.path / "probe.gguf";
        const std::vector<fixture::Kv> keys = {
            fixture::str("general.architecture", "qwen4exp"),
            fixture::u32("qwen4exp.block_count", 48),
            fixture::u32("qwen4exp.embedding_length", 2560),
            fixture::u32("qwen4exp.expert_count", 512),
            fixture::u32("qwen4exp.expert_used_count", 10),
            fixture::u32("qwen4exp.attention.head_count", 24),
            fixture::u32("qwen4exp.attention.head_count_kv", 2),
        };
        int failures = 0;
        auto expect = [&](int got, int wanted, const char* what) {
            std::printf("  %s: %s (exit %d, expected %d)\n", what, got == wanted ? "ok" : "FAIL", got, wanted);
            if (got != wanted) ++failures;
        };
        auto write = [&] { fixture::write(path, keys, {{"probe", {1}, 0}}); };
        write();
        expect(run(path), 0, "valid file");
        expect(run(path, false), 0, "inspection without --check");
        fixture::write(path, {fixture::str("general.architecture", "llama")}, {{"probe", {1}, 0}});
        expect(run(path), 1, "unsupported architecture");
        expect(run(path, false), 0, "inspection still accepts another architecture");
        fixture::write(path, {}, {{"probe", {1}, 0}});
        expect(run(path), 1, "missing architecture");
        write();
        patch_tensor(path, true, (std::numeric_limits<uint64_t>::max)() - 3);
        expect(run(path), 1, "offset that wraps the last gap to eight bytes");
        write();
        patch_tensor(path, true, 8);
        expect(run(path), 1, "ordinary offset past EOF");
        write();
        patch_tensor(path, false, uint64_t{1} << 62);
        expect(run(path), 1, "tensor byte count that wraps to zero");
        fixture::write(path, keys, {{"probe", {1}, 0}}, 1);
        expect(run(path), 1, "truncated payload");
        return failures ? 1 : 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "%s\n", e.what());
        return 1;
    }
}
