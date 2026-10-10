#include "strata/core/idle_cache.hpp"
#include <cuda_runtime.h>
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <limits>
#include <random>
#include <system_error>
#if defined(_WIN32)
#include <io.h>
#include <process.h>
#else
#include <unistd.h>
#endif

namespace strata::core {
namespace {
bool runtime(cudaError_t r, std::string& err) {
    if (r == cudaSuccess) return true;
    err = std::string("idle cache CUDA: ") + cudaGetErrorString(r);
    return false;
}
FILE* open_file(const std::filesystem::path& p, bool write) {
#if defined(_WIN32)
    return _wfopen(p.c_str(), write ? L"wb" : L"rb");
#else
    return std::fopen(p.c_str(), write ? "wb" : "rb");
#endif
}
struct File {
    FILE* f;
    ~File() { if (f) std::fclose(f); }
};
constexpr size_t chunk_bytes = 8u << 20;
constexpr uint64_t magic = 0x31454c4449564b53ull; // SKVIDLE1
uint64_t hash_bytes(const uint8_t* p, size_t n) {
    uint64_t h = 1469598103934665603ull;
    for (size_t i = 0; i < n; ++i) { h ^= p[i]; h *= 1099511628211ull; }
    return h;
}
int64_t unix_seconds() {
    return std::chrono::duration_cast<std::chrono::seconds>(
        std::chrono::system_clock::now().time_since_epoch()).count();
}
bool put(FILE* f, uint64_t x) { return std::fwrite(&x, sizeof x, 1, f) == 1; }
bool get(FILE* f, uint64_t expected) {
    uint64_t x = 0;
    return std::fread(&x, sizeof x, 1, f) == 1 && x == expected;
}
} // namespace

IdleDeviceArena::~IdleDeviceArena() { release(); }

bool IdleDeviceArena::allocate(size_t bytes, bool pageable, std::string& err) {
    if (address_ || bytes == 0) { err = "idle arena: invalid allocation"; return false; }
    bytes_ = bytes;
    pageable_ = pageable;
    if (!runtime(cudaGetDevice(&device_), err)) return false;
    if (!pageable) {
        void* p = nullptr;
        if (!runtime(cudaMalloc(&p, bytes), err)) return false;
        address_ = reinterpret_cast<uintptr_t>(p);
        allocated_ = bytes;
        mapped_ = true;
        return true;
    }
    if (!vmm_available() || !range_.reserve(bytes)) { err = "idle cache requires CUDA virtual memory support"; return false; }
    address_ = reinterpret_cast<uintptr_t>(range_.base());
    allocated_ = range_.chunks() * vmm_granularity();
    if (!map(err)) { release(); return false; }
    return true;
}

bool IdleDeviceArena::map(std::string& err) {
    if (mapped_) return true;
    if (!pageable_ || !address_) { err = "idle arena: no reserved address"; return false; }
    if (!range_.map_range(0, range_.chunks(), [] { return VmmChunk{0}; })) {
        err = "idle arena: cannot map the session memory"; return false;
    }
    mapped_ = true;
    return true;
}

bool IdleDeviceArena::unmap(std::string& err) {
    if (!pageable_) { err = "idle arena was not allocated with VMM"; return false; }
    for (int64_t i = 0; i < range_.chunks(); ++i) {
        if (!range_.mapped(i)) continue;
        const VmmChunk h = range_.unmap(i);
        if (!h) { err = "idle arena: could not unmap a session page"; return false; }
        vmm_chunk_free(h);
    }
    mapped_ = false;
    return true;
}

bool IdleDeviceArena::remember_constant(const void* p, size_t n, std::string& err) {
    const auto a = reinterpret_cast<uintptr_t>(p);
    if (!mapped_ || a < address_ || n > bytes_ || a - address_ > bytes_ - n) {
        err = "idle arena constant outside allocation"; return false;
    }
    Constant c{static_cast<size_t>(a - address_), std::vector<uint8_t>(n)};
    if (!runtime(cudaMemcpy(c.bytes.data(), p, n, cudaMemcpyDeviceToHost), err)) return false;
    constants_.push_back(std::move(c));
    return true;
}

bool IdleDeviceArena::fresh(std::string& err) {
    if (!map(err) || !runtime(cudaMemset(data(), 0, bytes_), err)) return false;
    for (const auto& c : constants_)
        if (!runtime(cudaMemcpy(static_cast<uint8_t*>(data()) + c.offset, c.bytes.data(), c.bytes.size(), cudaMemcpyHostToDevice), err)) return false;
    return runtime(cudaDeviceSynchronize(), err);
}

void IdleDeviceArena::release() {
    if (pageable_) range_.release();
    else if (address_) cudaFree(data());
    address_ = 0; bytes_ = allocated_ = 0; mapped_ = false;
    constants_.clear();
}

IdleSnapshot::~IdleSnapshot() { std::string ignored; erase(ignored); }

bool IdleSnapshot::init(const std::string& directory, int expiry_seconds, std::string& err) {
    try {
        const auto root = std::filesystem::absolute(std::filesystem::u8path(directory));
        std::filesystem::create_directories(root);
        // Only this feature's expired files; no recursive cleanup or model data.
        for (const auto& e : std::filesystem::directory_iterator(root)) {
            const auto name = e.path().filename().string();
            if (e.is_symlink() || !e.is_regular_file() || name.rfind("strata-kv-idle-", 0) != 0 ||
                (e.path().extension() != ".bin" && e.path().extension() != ".tmp")) continue;
            File f{open_file(e.path(), false)};
            uint64_t h[3]{};
            const bool valid_header = f.f && std::fread(h, sizeof h, 1, f.f) == 1 && h[0] == magic;
            const bool expired = valid_header ? h[2] <= static_cast<uint64_t>(unix_seconds()) :
                std::filesystem::file_time_type::clock::now() - e.last_write_time() >= std::chrono::seconds(expiry_seconds);
            if (f.f) { std::fclose(f.f); f.f = nullptr; }
            if (expired) { std::error_code ec; std::filesystem::remove(e.path(), ec); }
        }
        std::random_device r;
        nonce_ = (static_cast<uint64_t>(r()) << 32) ^ r();
        expiry_seconds_ = expiry_seconds;
        path_ = root / ("strata-kv-idle-" + std::to_string(nonce_) + ".bin");
        return true;
    } catch (const std::exception& e) { err = e.what(); return false; }
}

uint64_t IdleSnapshot::host_bytes() const {
    uint64_t n = 0;
    for (auto x : host_sizes_) n += x;
    return n;
}

bool IdleSnapshot::save(const std::vector<IdleDeviceArena*>& arenas,
                        const std::vector<std::vector<uint8_t>*>& hosts, std::string& err, int expires_in_seconds) {
    auto temp = path_; temp.replace_extension(".tmp");
    File f{open_file(temp, true)};
    if (!f.f) { err = "cannot create idle cache snapshot"; return false; }
    bool ok = true;
    host_sizes_.clear();
    for (auto* h : hosts) host_sizes_.push_back(h->size());
    std::vector<uint8_t> buffer(chunk_bytes);
    uint64_t written = 5 * sizeof(uint64_t);
    ok = put(f.f, magic) && put(f.f, nonce_) && put(f.f, static_cast<uint64_t>(unix_seconds() +
         (expires_in_seconds < 0 ? expiry_seconds_ : expires_in_seconds))) &&
         put(f.f, arenas.size()) && put(f.f, hosts.size());
    auto region = [&](const void* p, size_t bytes, bool device) {
        if (!ok) return;
        ok = put(f.f, bytes); written += sizeof(uint64_t);
        for (size_t off = 0; ok && off < bytes;) {
            const size_t n = std::min(chunk_bytes, bytes - off);
            const uint8_t* src = static_cast<const uint8_t*>(p) + off;
            if (device) {
                ok = runtime(cudaMemcpy(buffer.data(), src, n, cudaMemcpyDeviceToHost), err);
                src = buffer.data();
            }
            if (ok) ok = put(f.f, hash_bytes(src, n)) && std::fwrite(src, 1, n, f.f) == n;
            off += n; written += n + sizeof(uint64_t);
        }
    };
    for (auto* a : arenas) region(a->data(), a->size(), true);
    for (auto* h : hosts) region(h->data(), h->size(), false);
    if (ok) ok = std::fflush(f.f) == 0;
#if defined(_WIN32)
    if (ok) ok = _commit(_fileno(f.f)) == 0;
#else
    if (ok) ok = fsync(fileno(f.f)) == 0;
#endif
    if (std::fclose(f.f) != 0) ok = false;
    f.f = nullptr;
    std::error_code ec;
    if (ok) { std::filesystem::rename(temp, path_, ec); ok = !ec; }
    if (!ok) {
        std::filesystem::remove(temp, ec);
        if (err.empty()) err = "idle cache snapshot write/commit failed";
        return false;
    }
    disk_bytes_ = written;
    return true;
}

bool IdleSnapshot::restore(const std::vector<IdleDeviceArena*>& arenas,
                           const std::vector<std::vector<uint8_t>*>& hosts, std::string& err) {
    File f{open_file(path_, false)};
    uint64_t expiry = 0;
    bool ok = f.f && get(f.f, magic) && get(f.f, nonce_) && std::fread(&expiry, sizeof expiry, 1, f.f) == 1 &&
              expiry > static_cast<uint64_t>(unix_seconds()) &&
              get(f.f, arenas.size()) && get(f.f, hosts.size()) && hosts.size() == host_sizes_.size();
    std::vector<uint8_t> buffer(chunk_bytes);
    auto region = [&](void* p, size_t bytes, bool device) {
        if (!ok) return;
        ok = get(f.f, bytes);
        for (size_t off = 0; ok && off < bytes;) {
            const size_t n = std::min(chunk_bytes, bytes - off);
            uint64_t hash = 0;
            ok = std::fread(&hash, sizeof hash, 1, f.f) == 1 && std::fread(buffer.data(), 1, n, f.f) == n &&
                 hash_bytes(buffer.data(), n) == hash;
            if (ok) {
                auto* dst = static_cast<uint8_t*>(p) + off;
                if (device) ok = runtime(cudaMemcpy(dst, buffer.data(), n, cudaMemcpyHostToDevice), err);
                else std::memcpy(dst, buffer.data(), n);
            }
            off += n;
        }
    };
    for (auto* a : arenas) region(a->data(), a->size(), true);
    for (size_t i = 0; ok && i < hosts.size(); ++i) {
        hosts[i]->resize(host_sizes_[i]);
        region(hosts[i]->data(), hosts[i]->size(), false);
    }
    if (ok) ok = std::fgetc(f.f) == EOF && !std::ferror(f.f);
    if (!ok && err.empty()) err = "idle snapshot missing, truncated, or checksum mismatch";
    return ok;
}

bool IdleSnapshot::erase(std::string& err) {
    if (path_.empty()) return true;
    std::error_code ec;
    std::filesystem::remove(path_, ec);
    if (ec) { err = "cannot delete idle cache snapshot: " + ec.message(); return false; }
    auto temp = path_; temp.replace_extension(".tmp");
    std::filesystem::remove(temp, ec);
    if (ec) { err = "cannot delete incomplete idle cache snapshot: " + ec.message(); return false; }
    disk_bytes_ = 0;
    host_sizes_.clear();
    return true;
}
} // namespace strata::core
