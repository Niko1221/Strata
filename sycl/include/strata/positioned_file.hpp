// Buffered positional reads for GGUF shards. Each Windows request owns its
// OVERLAPPED and event; concurrent reads never change a shared file cursor.
// Used by both the engine (core/gguf_expert_source.cpp, --stream-experts) and
// the host-only regression test (tests/host/positioned_file_test.cpp), which
// can then exercise the same reader without a GPU or a model.
//
// Contract: the number of bytes read (0 at or past EOF), or -1 on a real
// failure. On POSIX a read interrupted by a signal is retried, as the
// GGUF callers expect (they reject <= 0, so EOF-as-0 changes nothing).
#pragma once
#include <algorithm>
#include <cerrno>
#include <cstdint>
#include <filesystem>
#include <limits>
#include <string>
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <fcntl.h>
#include <unistd.h>
#endif

namespace strata::platform {
#ifdef _WIN32
using PositionedHandle = HANDLE;
inline PositionedHandle invalid_positioned_handle() { return INVALID_HANDLE_VALUE; }
inline PositionedHandle open_positioned(const std::string& path) {
    return CreateFileW(std::filesystem::u8path(path).c_str(), GENERIC_READ, FILE_SHARE_READ,
                       nullptr, OPEN_EXISTING, FILE_FLAG_OVERLAPPED, nullptr);
}
inline void close_positioned(PositionedHandle h) { CloseHandle(h); }
inline int64_t read_positioned(PositionedHandle h, void* dst, size_t bytes, uint64_t offset) {
    OVERLAPPED ov{};
    ov.Offset = static_cast<DWORD>(offset);
    ov.OffsetHigh = static_cast<DWORD>(offset >> 32);
    ov.hEvent = CreateEventW(nullptr, TRUE, FALSE, nullptr);
    if (!ov.hEvent) return -1;
    DWORD got = 0;
    BOOL ok = ReadFile(h, dst, static_cast<DWORD>(std::min<size_t>(bytes, MAXDWORD)), &got, &ov);
    DWORD error = ok ? ERROR_SUCCESS : GetLastError();
    if (!ok && error == ERROR_IO_PENDING) {
        ok = GetOverlappedResult(h, &ov, &got, TRUE);
        error = ok ? ERROR_SUCCESS : GetLastError();
    }
    CloseHandle(ov.hEvent);
    return ok ? static_cast<int64_t>(got) : error == ERROR_HANDLE_EOF ? 0 : -1;
}
#else
using PositionedHandle = int;
inline PositionedHandle invalid_positioned_handle() { return -1; }
inline PositionedHandle open_positioned(const std::string& path) {
    return ::open(path.c_str(), O_RDONLY | O_CLOEXEC);
}
inline void close_positioned(PositionedHandle h) { ::close(h); }
inline int64_t read_positioned(PositionedHandle h, void* dst, size_t bytes, uint64_t offset) {
    if (offset > static_cast<uint64_t>(std::numeric_limits<off_t>::max())) return -1;
    ssize_t got;
    do {
        got = ::pread(h, dst, std::min<size_t>(bytes, std::numeric_limits<ssize_t>::max()),
                      static_cast<off_t>(offset));
    } while (got < 0 && errno == EINTR);
    return got;
}
#endif
} // namespace strata::platform
