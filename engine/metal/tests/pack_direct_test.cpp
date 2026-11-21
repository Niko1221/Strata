#include "strata/core/weights.hpp"
#include "strata/gpu/runtime.hpp"

#include <array>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <set>
#include <unistd.h>
#include <vector>

namespace fs = std::filesystem;
namespace gpu = strata::gpu;
using strata::core::WeightTable;

namespace {
constexpr size_t kRaw = (8 << 20) + 64;   // Exercise the tail after a full loader chunk.
constexpr size_t kAligned = (kRaw + 255) / 256 * 256;

bool make_pack(const fs::path& dir) {
    std::ofstream index(dir / "index.txt");
    index << "# align 256 pool " << kAligned + 3 * 256 << " tensors 4\n"
          << "raw 0 0 0 " << kRaw << " 0 " << kRaw << " 1 " << kRaw << " 0 0 0 0 0 0 0 0 0 0\n"
          << "bf16 0 1 " << kRaw << " 8 " << kAligned << " 4 2 1 0 0 0 0 0 0 0 0 0 0\n"
          << "f16 0 3 " << kRaw + 8 << " 8 " << kAligned + 256 << " 4 2 1 0 0 0 0 0 0 0 0 0 0\n"
          << "scales 0 0 " << kRaw + 16 << " 10 " << kAligned + 512 << " 14 2 1 4 0 2 0 1 4 8 2 1 0\n";
    std::ofstream dense(dir / "dense.bin", std::ios::binary);
    std::vector<uint8_t> raw(kRaw);
    for (size_t i = 0; i < raw.size(); ++i) raw[i] = (uint8_t) (i ^ (i >> 23));
    dense.write(reinterpret_cast<const char*>(raw.data()), raw.size());
    const std::array<float, 4> values{1.0f, -2.0f, 1.5f, -0.5f};
    const std::array<uint16_t, 5> planes{0xabcd, 0x1234, 0x3c00, 0xc000, 0xa1a2};
    dense.write(reinterpret_cast<const char*>(values.data()), sizeof values);
    dense.write(reinterpret_cast<const char*>(planes.data()), sizeof planes);
    index.close();
    dense.close();
    return index && dense;
}

bool compare(const fs::path& dir, const std::set<std::string>* skip) {
    std::string error;
    uint64_t bytes = 0;
    if (!WeightTable::pool_bytes(dir.string(), bytes, error, skip)) {
        std::cerr << error << '\n';
        return false;
    }
    std::vector<uint8_t> baseline(bytes);
    for (bool direct : {false, true}) {
        setenv("STRATA_METAL_PACK_DIRECT", direct ? "1" : "0", 1);
        void* arena = nullptr;
        if (gpu::malloc_device(&arena, bytes) != gpu::kSuccess) return false;
        WeightTable table;
        const bool loaded = table.load(dir.string(), arena, bytes, error, skip);
        const auto* host = static_cast<const uint8_t*>(gpu::host_view_span(arena, bytes));
        bool valid = loaded && host && !gpu::host_view_span(arena, bytes + 1);
        if (valid && direct) valid = table.report().upload_ms == 0 && std::memcmp(baseline.data(), host, bytes) == 0;
        if (valid && !direct) std::memcpy(baseline.data(), host, bytes);
        if (valid) {
            const std::array<uint16_t, 2> bf16{0x3f80, 0xc000}, f16{0x3e00, 0xb800};
            const std::array<float, 2> widened{1.0f, -2.0f};
            const auto matches = [&](const char* name, size_t offset, const void* expected, size_t size) {
                const auto* ref = table.find(name);
                if (!ref || !ref->data) return false;
                const auto* data = static_cast<const uint8_t*>(gpu::host_view_span(ref->data, ref->bytes));
                return data && std::memcmp(data + offset, expected, size) == 0;
            };
            valid = matches("bf16", 0, bf16.data(), sizeof bf16) &&
                    matches("f16", 0, f16.data(), sizeof f16) &&
                    matches("scales", 4, widened.data(), sizeof widened);
            if (skip) valid = valid && table.find("raw") && !table.find("raw")->data;
        }
        gpu::free_device(arena);
        if (!valid) {
            std::cerr << "pack load failed (direct=" << direct << ", compact=" << (skip != nullptr)
                      << "): " << error << '\n';
            return false;
        }
    }
    return true;
}
}  // namespace

int main() {
    std::string name = (fs::temp_directory_path() / "strata-pack-direct-XXXXXX").string();
    if (!mkdtemp(name.data())) return 1;
    const fs::path dir(name);
    const std::set<std::string> skip{"raw"};
    const bool ok = make_pack(dir) && compare(dir, nullptr) && compare(dir, &skip);
    fs::remove_all(dir);
    return ok ? 0 : 1;
}
