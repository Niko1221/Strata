#include "strata/program/vision_records.hpp"
#include <filesystem>
#include <fstream>
#include <limits>
#include <chrono>
#include <iostream>

int main() {
    const auto path = std::filesystem::temp_directory_path() /
        ("strata-vision-records-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    struct Cleanup { std::filesystem::path p; ~Cleanup() { std::filesystem::remove(p); } } cleanup{path};
    auto write = [&](int32_t n, int32_t nx, int32_t ny, int32_t width, float value, int payload, bool partial = false) {
        std::ofstream f(path, std::ios::binary | std::ios::trunc);
        int32_t h[] = {0x31455653, n, nx, ny, width};
        f.write((const char*) h, partial ? 3 : sizeof(h));
        for (int i = 0; i < payload; ++i) f.write((const char*) &value, sizeof(value));
    };
    std::vector<float> rows;
    std::vector<strata::program::VisionRecord> records;
    std::string err;
    auto read = [&](int max_rows = 4) { return strata::program::read_vision_records(path.string(), 2, max_rows, rows, records, err); };
    int failures = 0;
    auto check = [&](bool ok, const char* name) { if (!ok) { std::cerr << name << '\n'; ++failures; } };
    write(4,2,2,2,0.25f,8);check(read() && rows.size()==8 && records.size()==1,"valid grid");
    write(4,2,2,2,0.25f,7);check(!read(),"short payload");
    write(4,2,2,2,std::numeric_limits<float>::infinity(),8);check(!read(),"infinity");
    write(4,2,2,2,std::numeric_limits<float>::quiet_NaN(),8);check(!read(),"NaN");
    write(4,2,3,2,0.25f,8);check(!read(),"grid mismatch");
    write(4,2,2,3,0.25f,12);check(!read(),"width mismatch");
    write(2147483647,2147483647,1,2,0.25f,0);check(!read() && rows.empty(),"oversized record before allocation");
    write(4,2,2,2,0.25f,0,true);check(!read(),"partial header");
    write(4,2,2,2,0.25f,8);check(!read(3),"prompt row budget");
    { std::ofstream f(path, std::ios::binary | std::ios::app); int32_t h[]={0x31455653,1,1,1,2};f.write((const char*)h,sizeof(h)); }
    check(!read(),"combined row budget");
    return failures ? 1 : 0;
}
