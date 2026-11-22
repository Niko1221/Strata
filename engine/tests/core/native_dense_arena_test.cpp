// tests/core/native_dense_arena_test.cpp - STRATA_DENSE_ARENA=1: NativeDense::load puts the native projection matrices in
// one device allocation (strata/core/dense_arena.hpp).  Checks, on a synthetic GGUF of Q8_0 matrices from 272 B to
// 1.06 MiB (sizes that are not multiples of 256 B among them):
//   1. every matrix's device bytes are its GGUF payload, byte for byte;
//   2. every matrix starts on a 256 B boundary, at the offset dense_arena::place gives it after the one before (one
//      allocation, in the file's order);
//   3. tensor_count, weight_bytes and allocated_bytes are the arena's (the payload; the arena rounded to 2 MiB);
//   4. release() frees the arena and leaves nothing counted.
// The switch is read once per process, so this is its own test: ctest sets STRATA_DENSE_ARENA=1, and run by hand the test
// sets it itself.  Needs a CUDA device (exits 77 without one).
#include "gguf_fixture.hpp"

#include "strata/core/dense_arena.hpp"
#include "strata/core/native_dense.hpp"
#include "strata/core/weights.hpp"

#include <cuda_runtime.h>

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <set>
#include <string>
#include <vector>

namespace fs = std::filesystem;
using strata::core::NativeDense;
using strata::core::WeightTable;
namespace arena = strata::core::dense_arena;

namespace {
int g_fail = 0;
void check(bool ok, const std::string& what) {
    std::printf("  %-88s %s\n", what.c_str(), ok ? "ok" : "FAIL");
    if (!ok) ++g_fail;
}

struct TempDir {
    fs::path path;
    TempDir() {
        path = fs::temp_directory_path() /
               ("strata-dense-arena-test-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
        fs::create_directories(path);
    }
    ~TempDir() {
        std::error_code ignored;
        fs::remove_all(path, ignored);
    }
};

struct Matrix { const char* name; uint64_t ne0, ne1; uint8_t seed; };
// in the GGUF's order, which is the order `load` places them in
const Matrix kMatrices[] = {
    {"blk.0.attn_qkv.weight", 256, 4, 1},        // 1,088 B: the next starts 192 B later
    {"blk.0.attn_gate.weight", 256, 7, 2},       // 1,904 B
    {"blk.1.ssm_out.weight", 512, 2048, 3},      // 1,114,112 B: over a MiB
    {"blk.1.attn_output.weight", 256, 1, 4},     // 272 B
    {"blk.2.attn_v.weight", 2560, 100, 5},       // 272,000 B
};
uint64_t q8_bytes(const Matrix& m) { return m.ne0 / 32 * 34 * m.ne1; }

std::vector<fixture::Kv> arch_keys() {
    return {fixture::str("general.architecture", "qwen4exp"), fixture::u32("qwen4exp.block_count", 48),
            fixture::u32("qwen4exp.embedding_length", 2560), fixture::u32("qwen4exp.expert_count", 512),
            fixture::u32("qwen4exp.expert_used_count", 10), fixture::u32("qwen4exp.attention.head_count", 24),
            fixture::u32("qwen4exp.attention.head_count_kv", 2)};
}

// shape-only quantized rows (what iq_pack writes for a tensor the GGUF serves): no bytes in the pool
void write_pack(const fs::path& dir) {
    fs::create_directories(dir);
    std::string idx = "# align 256 pool 256 tensors " + std::to_string(std::size(kMatrices)) + "\n";
    for (const Matrix& m : kMatrices)
        idx += std::string(m.name) + " 0 0 0 0 0 0 " + std::to_string(m.ne0) + " " + std::to_string(m.ne1) +
               " 8 0 32 0 0 0 0 0 0 0\n";
    std::ofstream(dir / "index.txt", std::ios::binary) << idx;
    std::vector<char> bytes(16, 0);
    std::ofstream(dir / "dense.bin", std::ios::binary).write(bytes.data(), (std::streamsize) bytes.size());
}
}  // namespace

int main() {
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) {
        std::printf("native_dense_arena_test: no CUDA device, skipped\n");
        return 77;
    }
    if (const char* v = std::getenv("STRATA_DENSE_ARENA"); v == nullptr || v[0] != '1') {
#if defined(_WIN32)
        _putenv_s("STRATA_DENSE_ARENA", "1");
#else
        setenv("STRATA_DENSE_ARENA", "1", 1);
#endif
    }
    TempDir tmp;
    const fs::path pack = tmp.path / "pack";
    write_pack(pack);
    const std::string shard = (tmp.path / "dense.gguf").string();
    std::vector<fixture::Tensor> tensors;
    for (const Matrix& m : kMatrices) tensors.push_back({m.name, {m.ne0, m.ne1}, 8, m.seed});
    fixture::write(shard, arch_keys(), tensors);

    // generate.cpp's order: served_names, pool_bytes, the table's load, NativeDense::load
    const std::vector<std::string> shards{shard};
    std::set<std::string> skip;
    std::string err;
    uint64_t pool = 0;
    void* pool_dev = nullptr;
    WeightTable wt;
    NativeDense dense;
    const bool ok = NativeDense::served_names(shards, true, skip, err) &&
                    WeightTable::pool_bytes(pack.string(), pool, err, &skip) &&
                    cudaMalloc(&pool_dev, pool ? pool : 256) == cudaSuccess &&
                    wt.load(pack.string(), pool_dev, pool ? pool : 256, err, &skip) && dense.load(shards, wt, err, true);
    std::printf("STRATA_DENSE_ARENA=1, %zu Q8_0 matrices\n", std::size(kMatrices));
    check(ok, "the pack and the native matrices load " + err);
    if (ok) {
        const uint8_t* base = nullptr;
        uint64_t used = 0, payload = 0;
        bool bytes_same = true, aligned = true, placed = true, attached = true;
        for (const Matrix& m : kMatrices) {
            const strata::core::WeightRef* r = wt.find(m.name);
            const uint64_t n = q8_bytes(m), at = arena::place(used, n);
            payload += n;
            if (r == nullptr || r->native_data == nullptr || r->native_type != 8 || r->native_bytes != n) {
                attached = false;
                continue;
            }
            const uint8_t* p = (const uint8_t*) r->native_data;
            if (base == nullptr) base = p;
            aligned = aligned && (uintptr_t) p % arena::kAlign == 0;
            placed = placed && p == base + at;
            std::vector<uint8_t> back((size_t) n);
            bytes_same = bytes_same && cudaMemcpy(back.data(), p, back.size(), cudaMemcpyDeviceToHost) == cudaSuccess;
            for (uint64_t i = 0; i < n && bytes_same; ++i) bytes_same = back[(size_t) i] == fixture::pattern(m.seed, i);
        }
        check(attached, "every matrix is served natively, Q8_0, with its payload's byte count");
        check(bytes_same, "1. each matrix's device bytes are its GGUF payload");
        check(aligned, "2. each matrix starts on a 256 B boundary");
        check(placed, "2. ... at dense_arena::place's offset from the first: one allocation, in the file's order");
        check(dense.tensor_count() == std::size(kMatrices), "3. tensor_count counts the arena's matrices");
        check(dense.weight_bytes() == payload, "3. weight_bytes is the payload (" + std::to_string(payload) + " B)");
        check(dense.allocated_bytes() == arena::granule_bytes(used),
              "3. allocated_bytes is the arena rounded to 2 MiB (" + std::to_string(arena::granule_bytes(used)) + " B)");
        dense.release();
        check(dense.tensor_count() == 0 && dense.weight_bytes() == 0, "4. release leaves no matrix and no weight bytes counted");
        check(cudaGetLastError() == cudaSuccess && cudaDeviceSynchronize() == cudaSuccess, "4. release frees the arena without a CUDA error");
    }
    if (pool_dev) cudaFree(pool_dev);
    std::printf(g_fail ? "native_dense_arena_test: %d FAILED\n" : "native_dense_arena_test: all passed\n", g_fail);
    return g_fail ? 1 : 0;
}
