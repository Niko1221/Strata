// Native, read-only weight sources. No CUDA, torch, GGUF or generated model files.
#pragma once
#include <array>
#include <cstdint>
#include <filesystem>
#include <map>
#include <memory>
#include <span>
#include <string>
#include <vector>

namespace strata::weights {
enum class DType { U8, I8, BOOL, I16, U16, I32, U32, I64, U64, F16, BF16, F32, F64, F8_E4M3, F8_E5M2 };
enum class Family { Dense, Expert, Ngram, Mtp, Vision };
const char* dtype_name(DType);
const char* family_name(Family);
uint64_t dtype_bytes(DType);
Family weight_family(const std::string&);

// Shapes use HF row-major order. Packed logical shape is assigned by the model
// adapter, never guessed by reversing dimensions or by the container reader.
struct TensorDesc {
    std::string name;
    DType dtype;
    Family family;
    std::vector<uint64_t> physical_shape;
    size_t file_id = 0;
    uint64_t offset = 0, bytes = 0; // absolute file byte range, not data-relative
};
struct ShardDesc {
    std::filesystem::path path;
    uint64_t file_bytes = 0, header_bytes = 0, data_offset = 0;
    size_t tensor_count = 0;
};
struct ReadRequest {
    const TensorDesc* tensor;
    uint64_t relative_offset;
    std::span<uint8_t> destination;
};
struct IoStats {
    uint64_t header_bytes = 0, metadata_bytes = 0, data_calls = 0, staging_peak_bytes = 0;
    std::array<uint64_t, 5> data_bytes{};
};
class WeightSource {
public:
    virtual ~WeightSource() = default;
    virtual const TensorDesc& tensor(const std::string&) const = 0;
    // Destination spans belong to the caller. Source descriptions live as long
    // as the source; reads are synchronous and serial (not thread-safe).
    virtual void read_many(std::span<const ReadRequest>) = 0;
    virtual const IoStats& io_stats() const = 0;
    std::vector<uint8_t> read(const TensorDesc&);
};
class SafetensorsSource final : public WeightSource {
public:
    explicit SafetensorsSource(const std::filesystem::path& model_dir);
    ~SafetensorsSource();
    SafetensorsSource(const SafetensorsSource&) = delete;
    SafetensorsSource& operator=(const SafetensorsSource&) = delete;
    const TensorDesc& tensor(const std::string&) const override;
    const std::map<std::string, TensorDesc>& tensors() const;
    const std::vector<ShardDesc>& shards() const;
    void read_many(std::span<const ReadRequest>) override;
    const IoStats& io_stats() const override;
    // Once sealed, expert fetches fail instead of quietly falling back to disk.
    void seal_expert_reads();
    // After all resident families have loaded, release startup file handles.
    // Tensor descriptions and counters remain valid; any later payload read fails.
    // PLE owns independent DirectFile handles and is unaffected.
    void seal_resident_reads();
    std::string metadata_text(const std::string& filename);
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
} // namespace strata::weights
