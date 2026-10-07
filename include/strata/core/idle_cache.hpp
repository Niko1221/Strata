#pragma once

#include "strata/core/vmm.hpp"
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

namespace strata::core {

// Keeps graph-bound addresses reserved while their physical VRAM is absent.
// Disabled allocations use cudaMalloc, preserving the normal allocation path.
class IdleDeviceArena {
public:
    ~IdleDeviceArena();
    IdleDeviceArena() = default;
    IdleDeviceArena(const IdleDeviceArena&) = delete;
    IdleDeviceArena& operator=(const IdleDeviceArena&) = delete;
    bool allocate(size_t bytes, bool pageable, std::string& err);
    bool map(std::string& err);
    bool unmap(std::string& err);
    bool remember_constant(const void* ptr, size_t bytes, std::string& err);
    bool fresh(std::string& err);
    void release();
    void* data() const { return reinterpret_cast<void*>(address_); }
    size_t size() const { return bytes_; }
    size_t physical_bytes() const { return allocated_; }
    bool resident() const { return mapped_; }
private:
    struct Constant { size_t offset; std::vector<uint8_t> bytes; };
    uintptr_t address_ = 0;
    VmmRange range_;
    size_t bytes_ = 0, allocated_ = 0;
    int device_ = 0;
    bool pageable_ = false, mapped_ = false;
    std::vector<Constant> constants_;
};

// One process-local snapshot. Metadata/pointers never come from the file: the
// caller supplies the expected regions. Invalid files are a cache miss.
class IdleSnapshot {
public:
    ~IdleSnapshot();
    bool init(const std::string& directory, int expiry_seconds, std::string& err);
    bool save(const std::vector<IdleDeviceArena*>& arenas,
              const std::vector<std::vector<uint8_t>*>& hosts, std::string& err, int expires_in_seconds = -1);
    bool restore(const std::vector<IdleDeviceArena*>& arenas,
                 const std::vector<std::vector<uint8_t>*>& hosts, std::string& err);
    bool erase(std::string& err);
    const std::filesystem::path& path() const { return path_; }
    uint64_t disk_bytes() const { return disk_bytes_; }
    uint64_t host_bytes() const;
private:
    std::filesystem::path path_;
    uint64_t nonce_ = 0, disk_bytes_ = 0;
    int expiry_seconds_ = 0;
    std::vector<size_t> host_sizes_;
};

} // namespace strata::core
