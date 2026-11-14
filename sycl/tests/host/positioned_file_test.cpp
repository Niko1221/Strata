// Exercise the same positional read helper used by GgufExpertSource, without SYCL.
#include "strata/positioned_file.hpp"
#include <atomic>
#include <fstream>
#include <iostream>
#include <thread>
#include <vector>
#ifdef _WIN32
#include <winioctl.h>
#endif

int main() {
    namespace p = strata::platform;
    const std::string path = "positioned-file-test.bin";
    const uint64_t far_offset = (uint64_t{4} << 30) + 19;
    // A sparse fixture proves offsets are not truncated to 32 bits.
#ifdef _WIN32
    HANDLE out = CreateFileW(L"positioned-file-test.bin", GENERIC_WRITE, 0, nullptr,
                             CREATE_NEW, FILE_ATTRIBUTE_NORMAL, nullptr);
    if (out == INVALID_HANDLE_VALUE) return 1;
    DWORD got;
    if (!DeviceIoControl(out, FSCTL_SET_SPARSE, nullptr, 0, nullptr, 0, &got, nullptr)) {
        CloseHandle(out);
        std::filesystem::remove(path);
        std::cerr << "Sparse fixture requires a filesystem supporting sparse files.\n";
        return 1;
    }
    LARGE_INTEGER at{};
    DWORD written = 0;
    bool created = WriteFile(out, "near", 4, &written, nullptr) && written == 4;
    at.QuadPart = far_offset;
    created = created && SetFilePointerEx(out, at, nullptr, FILE_BEGIN)
              && WriteFile(out, "far!", 4, &written, nullptr) && written == 4;
    CloseHandle(out);
#else
    int out = ::open(path.c_str(), O_WRONLY | O_CREAT | O_EXCL, 0600);
    if (out < 0) return 1;
    bool created = ::pwrite(out, "near", 4, 0) == 4 && ::pwrite(out, "far!", 4, far_offset) == 4;
    ::close(out);
#endif
    if (!created) { std::filesystem::remove(path); return 1; }
    auto h = p::open_positioned(path);
    if (h == p::invalid_positioned_handle()) { std::filesystem::remove(path); return 1; }
    std::atomic<bool> ok{true};
    std::vector<std::thread> workers;
    for (int i = 0; i < 8; ++i) workers.emplace_back([&, i] {
        for (int j = 0; j < 200; ++j) {
            const bool high = (i + j) % 2;
            char bytes[4];
            if (p::read_positioned(h, bytes, 4, high ? far_offset : 0) != 4 ||
                std::string(bytes, 4) != (high ? "far!" : "near")) ok = false;
        }
    });
    for (auto& t : workers) t.join();
    char tail[8];
    ok = ok && p::read_positioned(h, tail, 8, far_offset + 2) == 2;
    ok = ok && p::read_positioned(h, tail, 8, far_offset + 4) == 0;
    p::close_positioned(h);
    ok = ok && p::read_positioned(p::invalid_positioned_handle(), tail, 8, 0) == -1;
    std::filesystem::remove(path);
    ok = ok && p::open_positioned(path) == p::invalid_positioned_handle();
    std::cout << (ok ? "PASS" : "FAIL") << ": concurrent reads, >4 GiB offsets, short reads, EOF, invalid handle, missing file\n";
    return ok ? 0 : 1;
}
