// dsv4/gguf_header.hpp - header-only GGUF reader (metadata + tensor infos, never the weights).
// Self-contained: no Strata dependency, C++17, little-endian hosts only.
#pragma once
#include <cstdint>
#include <map>
#include <string>
#include <vector>

namespace dsv4 {

enum GgufType : uint32_t {
    GGUF_U8 = 0, GGUF_I8 = 1, GGUF_U16 = 2, GGUF_I16 = 3, GGUF_U32 = 4, GGUF_I32 = 5, GGUF_F32 = 6,
    GGUF_BOOL = 7, GGUF_STR = 8, GGUF_ARR = 9, GGUF_U64 = 10, GGUF_I64 = 11, GGUF_F64 = 12
};

constexpr uint64_t kMaxStoredNums = 65536;  // numeric arrays longer than this keep only their length
// String arrays are kept whole: tokenizer.ggml.tokens (129280 entries here) and tokenizer.ggml.merges are
// what the runner decodes generated ids with, so a truncated copy makes every answer come out empty.
// The cap below only guards against a corrupt header, not against a real vocabulary.
constexpr uint64_t kMaxStoredStrs = 1u << 21;

struct GgufValue {
    uint32_t type = 0;
    uint32_t elem_type = 0;
    uint64_t arr_len = 0;
    int64_t i = 0;
    uint64_t u = 0;
    double f = 0;
    std::string s;
    std::vector<double> nums;
    std::vector<std::string> strs;
    bool nums_complete() const { return nums.size() == arr_len; }
};

struct GgufTensor {
    std::string name;
    std::vector<uint64_t> dims;  // ggml order: dims[0] is the contiguous (row) dimension
    uint32_t type = 0;
    uint64_t offset = 0;         // relative to the shard's tensor-data section, as stored in the file
    uint64_t abs_offset = 0;     // offset from the START OF FILE: mmap(shard) + abs_offset is the data
    uint64_t nbytes = 0;         // 0 when the ggml type is unknown (known_type == false)
    bool known_type = false;
    int shard = 0;
};

struct GgufHeader {
    uint32_t version = 0;
    std::vector<std::string> files;
    std::vector<uint64_t> data_start;   // per shard: where the tensor data begins (after alignment)
    std::map<std::string, GgufValue> kv;  // first shard wins on duplicates
    std::vector<GgufTensor> tensors;
    const GgufValue* find(const std::string& key) const {
        auto it = kv.find(key);
        return it == kv.end() ? nullptr : &it->second;
    }
};

/// ggml type id -> name / block elements / block bytes.  False for ids not in the table.
bool ggml_type_info(uint32_t type, const char** name, int* block_elems, int* block_bytes);
std::string ggml_type_str(uint32_t type);  // "type#N" for ids not in the table

/// Reads the headers of one or more shards (merging metadata and tensor lists).
bool gguf_read_header(const std::vector<std::string>& paths, GgufHeader& out, std::string& err);

}  // namespace dsv4
