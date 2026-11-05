// cann/include/strata/cann/gguf.hpp - read the model file the port has to load.
//
// Only what the port needs to find a tensor and hand its bytes to a decoder: the header, the
// metadata values that describe the architecture, and the tensor index. Weights stay on disk;
// `tensor_bytes` reads a range on demand, so a 22 GB file costs nothing to open.
#pragma once

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

namespace strata::cann {

struct GgufTensor {
    std::string name;
    uint32_t type = 0;          // ggml type id (12 = Q4_K, 14 = Q6_K, ...)
    std::vector<uint64_t> dims; // ne[0] first, as the file stores them
    uint64_t offset = 0;        // from the start of the data section
    uint64_t elements() const;
};

class Gguf {
public:
    explicit Gguf(const std::string & path);

    uint32_t version() const { return version_; }
    uint64_t data_offset() const { return data_offset_; }
    std::string arch() const;
    const std::vector<GgufTensor> & tensors() const { return tensors_; }

    // Metadata accessors; 0 / "" when absent.
    int64_t metadata_int(const std::string & key) const;
    std::string metadata_str(const std::string & key) const;

    // Everything the file declares, for tools that report what a model actually contains.
    const std::unordered_map<std::string, int64_t> & int_metadata() const { return ints_; }
    const std::unordered_map<std::string, std::string> & str_metadata() const { return strings_; }

    const GgufTensor * find(const std::string & name) const;

    // Read `bytes` starting at `offset` inside the tensor's data.
    std::vector<uint8_t> tensor_bytes(const GgufTensor & t, uint64_t offset, uint64_t bytes) const;

private:
    std::string path_;
    uint32_t version_ = 0;
    uint64_t data_offset_ = 0;
    std::vector<GgufTensor> tensors_;
    std::unordered_map<std::string, std::string> strings_;
    std::unordered_map<std::string, int64_t> ints_;
};

} // namespace strata::cann
