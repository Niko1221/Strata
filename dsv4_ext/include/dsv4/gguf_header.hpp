/**
 * @file gguf_header.hpp
 * @brief Header-only GGUF reader: metadata + tensor infos (never the weights).
 *
 * Self-contained C++17 module for parsing GGUF v3+ file format headers. Reads only metadata
 * key-value pairs and tensor information (names, shapes, types, offsets) — never the actual
 * weight data. Supports multi-shard models by merging metadata and concatenating tensor lists.
 *
 * @par Design decisions
 *   - Little-endian hosts only (all modern x86/ARM).
 *   - No Strata dependency: pure library, linkable anywhere.
 *   - String arrays (tokenizer.ggml.tokens, merges) are kept whole — truncation makes every
 *     decoded answer come out empty. Numeric arrays longer than `kMaxStoredNums` keep only length.
 *   - Tensor data offsets use `abs_offset` (from file start), so `mmap(shard) + abs_offset` gives
 *     direct pointer to tensor data without arithmetic at runtime.
 *
 * @par Usage
 *   @code
 *   GgufHeader h;
 *   std::string err;
 *   gguf_read_header({"model-00001-of-00003.gguf"}, h, err);
 *   const auto* vocab = h.find("tokenizer.ggml.tokens");
 *   for (const auto& t : h.tensors) {
 *       printf("%s [%zu] type=%u offset=%lu\n", t.name.c_str(), t.dims.size(), t.type, t.abs_offset);
 *   }
 *   @endcode
 */
#pragma once

#include <cstdint>
#include <map>
#include <string>
#include <vector>

namespace dsv4 {

/**
 * @enum GgufType
 * @brief GGUF value type codes (matches spec v3+).
 *
 * These map directly to the GGUF binary format's type field for each key-value pair.
 */
enum GgufType : uint32_t {
    GGUF_U8 = 0,   GGUF_I8 = 1,    GGUF_U16 = 2,   GGUF_I16 = 3,   GGUF_U32 = 4,   GGUF_I32 = 5,   GGUF_F32 = 6,
    GGUF_BOOL = 7, GGUF_STR = 8,   GGUF_ARR = 9,   GGUF_U64 = 10,  GGUF_I64 = 11,  GGUF_F64 = 12
};

/**
 * @def kMaxStoredNums
 * @brief Maximum numeric array elements to store fully (65536).
 *
 * Numeric arrays longer than this keep only their length — the values are not stored.
 * This guards against excessive memory usage from corrupt headers with huge arrays.
 */
constexpr uint64_t kMaxStoredNums = 65536;

/**
 * @def kMaxStoredStrs
 * @brief Maximum string array elements to store (2^21 ≈ 2M).
 *
 * String arrays are kept whole: tokenizer.ggml.tokens (129280 entries in DeepSeek-V4-Flash) and
 * tokenizer.ggml.merges are what the runner decodes generated ids with, so a truncated copy makes
 * every answer come out empty. The cap only guards against corrupt headers, not real vocabularies.
 */
constexpr uint64_t kMaxStoredStrs = 1u << 21;

/**
 * @struct GgufValue
 * @brief A single GGUF key-value pair with its typed value data.
 *
 * Supports all GGUF types: scalars (i/u/f), strings, booleans, and arrays (numeric, string).
 * Arrays may be partial (`nums_complete()` returns false for truncated numeric arrays).
 */
struct GgufValue {
    uint32_t type = 0;              ///< GgufType enum: the value's type code.
    uint32_t elem_type = 0;         ///< For arrays: element type of nested arrays (0 if scalar).
    uint64_t arr_len = 0;           ///< Array length (number of elements/strings).
    int64_t i = 0;                  ///< Scalar integer value.
    uint64_t u = 0;                 ///< Scalar unsigned integer value.
    double f = 0;                   ///< Scalar float/double value.
    std::string s;                  ///< String value (for GGUF_STR type).
    std::vector<double> nums;       ///< Numeric array values (partial if arr_len > kMaxStoredNums).
    std::vector<std::string> strs;  ///< String array values (always complete, up to kMaxStoredStrs).

    /**
     * @brief Check if the numeric array is fully stored.
     * @return true if nums.size() == arr_len (complete); false if truncated.
     */
    bool nums_complete() const { return nums.size() == arr_len; }
};

/**
 * @struct GgufTensor
 * @brief Information about a single tensor in the GGUF file.
 *
 * Contains name, shape (ggml order: dims[0] = contiguous row dimension), quantization type,
 * and byte offsets for locating tensor data within the file. The `abs_offset` field allows
 * direct mmap-based access without runtime arithmetic.
 */
struct GgufTensor {
    std::string name;               ///< Tensor name (e.g., "model.blk.0.attn_weight").
    std::vector<uint64_t> dims;     ///< Shape in ggml order: dims[0] is the contiguous (row) dimension.
    uint32_t type = 0;              ///< GgmlType enum: quantization format of tensor data.
    uint64_t offset = 0;            ///< Offset from start of shard's tensor-data section.
    uint64_t abs_offset = 0;        ///< Absolute offset from START OF FILE: mmap(shard) + abs_offset → data pointer.
    uint64_t nbytes = 0;            ///< Tensor byte size (0 when type is unknown / known_type == false).
    bool known_type = false;        ///< true if the ggml quantization type is recognized.
    int shard = 0;                  ///< Shard index this tensor belongs to (0-based).
};

/**
 * @struct GgufHeader
 * @brief Complete parsed GGUF header: version, files, metadata, and tensor list.
 *
 * For multi-shard models, `files` contains all shard paths, `data_start` has per-shard
 * tensor-data offsets, and tensors from all shards are concatenated in file order.
 * Metadata keys are merged with first-shard-wins semantics for duplicates.
 */
struct GgufHeader {
    uint32_t version = 0;                           ///< GGUF format version (3+).
    std::vector<std::string> files;                  ///< Shard file paths in order.
    std::vector<uint64_t> data_start;                ///< Per-shard: byte offset where tensor data begins.
    std::map<std::string, GgufValue> kv;             ///< Key-value metadata (first shard wins on duplicates).
    std::vector<GgufTensor> tensors;                 ///< All tensors from all shards, in file order.

    /**
     * @brief Find a metadata key by name.
     * @param key  Metadata key to look up (e.g., "general.architecture").
     * @return Pointer to the GgufValue, or nullptr if not found.
     */
    const GgufValue* find(const std::string& key) const {
        auto it = kv.find(key);
        return it == kv.end() ? nullptr : &it->second;
    }
};

/**
 * @brief Query ggml type metadata: name, block elements, block bytes.
 * @param type       GgmlType enum value to query.
 * @param name       Output: C-string type name (e.g., "IQ1_M"), or nullptr.
 * @param block_elems  Output: elements per quantization block, or 0 if unknown.
 * @param block_bytes  Output: bytes per block, or 0 if unknown.
 * @return true if the type is recognized and info was written; false otherwise.
 */
bool ggml_type_info(uint32_t type, const char** name, int* block_elems, int* block_bytes);

/**
 * @brief Convert a ggml type id to human-readable string.
 * @param type  GgmlType enum value.
 * @return Type name (e.g., "IQ1_M") for known types, or "type#N" for unknown ids.
 */
std::string ggml_type_str(uint32_t type);

/**
 * @brief Parse GGUF headers from one or more shard files.
 *
 * Reads metadata and tensor information from all shards in order. Metadata keys are merged
 * with first-shard-wins semantics. Tensor lists are concatenated. Computes `abs_offset` for
 * each tensor by adding the shard's `data_start` offset to the per-tensor file offset.
 *
 * @param paths  Vector of shard file paths in order (e.g., ["shard-1.gguf", "shard-2.gguf"]).
 * @param out    Output: populated GgufHeader with merged metadata and all tensors.
 * @param err    Output error message on failure.
 * @return true if all headers parsed successfully; false otherwise (err contains details).
 */
bool gguf_read_header(const std::vector<std::string>& paths, GgufHeader& out, std::string& err);

}  // namespace dsv4
