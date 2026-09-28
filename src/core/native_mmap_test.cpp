#include "strata/core/expert_source.hpp"
#include "strata/kernels/cpu/expert_layout.hpp"
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <vector>

namespace fs = std::filesystem;
using namespace strata::kernels::cpu;
void require(bool ok, const char* label) {
    if (!ok) { std::fprintf(stderr, "FAIL: %s\n", label); std::exit(1); }
}
int main() {
    auto dir = fs::temp_directory_path() / ("strata-native-mmap-" +
        std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    require(fs::create_directory(dir), "exclusive fixture directory");
    NativeFmt a, b;
    std::string err;
    require(native_fmt(21, 20, 2560, 640, a, err), "IQ3_S/IQ4_NL fixture format");
    require(native_fmt(18, 42, 2560, 640, b, err), "IQ3_XXS/Q2_0 fixture format");
    require(a.bytes != b.bytes, "different layer strides");
    {
        std::ofstream layout(dir / "native_experts.txt");
        layout << "0 21 20 0 " << a.bytes << '\n';
        layout << "1 18 42 " << 2*a.bytes << ' ' << b.bytes << '\n';
        std::ofstream data(dir / "experts.bin", std::ios::binary);
        for (int i = 0; i < 4; ++i) {
            std::vector<char> bytes((size_t)(i < 2 ? a.bytes : b.bytes), (char)(31+i));
            data.write(bytes.data(), bytes.size());
        }
    }
    require(expert_layout_load(dir.string(), 2, 2, err), "native layout load");
    strata::core::FileExpertSource src;
    require(src.open(dir.string(), 2, 2, err), "native mapping opens");
    for (int l = 0; l < 2; ++l) for (int e = 0; e < 2; ++e) {
        const auto* ptr = src.blob(l,e);
        require(ptr != nullptr, "valid blob");
        const size_t n = (size_t)(l == 0 ? a.bytes : b.bytes);
        for (size_t i = 0; i < n; ++i) require(ptr[i] == (uint8_t)(31+l*2+e), "exact mapped bytes/stride");
    }
    require(!src.blob(0,2) && !src.blob(2,0) && !src.blob(-1,0) && !src.blob(0,-1), "axis bounds");
    src.close(); require(!src.mapped() && !src.blob(0,0), "close invalidates mapping");
    require(!src.open(dir.string(), 1, 2, err), "geometry mismatch rejected");
    fs::resize_file(dir / "experts.bin", 2*a.bytes + 2*b.bytes - 1);
    require(!src.open(dir.string(), 2, 2, err), "truncated native file rejected");
    fs::remove(dir / "native_experts.txt");
    require(expert_layout_load(dir.string(), 2, 2, err), "canonical layout load");
    {
        std::ofstream data(dir / "experts.bin", std::ios::binary | std::ios::trunc);
        for (int i=0;i<4;++i) {
            std::vector<char> bytes(BLOB, (char)(50+i));data.write(bytes.data(), bytes.size());
        }
    }
    require(src.open(dir.string(),2,2,err), "canonical mapping retained");
    require(src.blob(1,1)[0]==53 && src.blob(1,1)[BLOB-1]==53, "canonical offsets retained");
    src.close();fs::remove(dir / "experts.bin");fs::remove(dir);
    std::puts("PASS: native variable strides, full blob bytes, bounds, truncation, geometry, close and canonical compatibility");
}
