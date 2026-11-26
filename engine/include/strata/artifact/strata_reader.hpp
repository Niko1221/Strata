// include/strata/artifact/strata_reader.hpp - Native .strata v1.0 container reader (docs/FORMAT-SPEC.md §2)
#pragma once

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

namespace strata::artifact {

constexpr size_t STRATA_PAGE_SIZE = 4096;
constexpr size_t STRATA_PAGE_SIZE_16K = 16384;
constexpr size_t STRATA_TABLE_OFFSET_4K = 0x10;   // section table of the 4 KiB layout (every file up to 0.2.22)
constexpr size_t STRATA_TABLE_OFFSET_16K = 0x20;  // section table of the 16 KiB layout (strata_layout.hpp tells them apart)
constexpr uint16_t STRATA_NUM_SECTIONS = 6;

enum class StrataLayout : uint8_t {
    Page4K  = 0,  // table at 0x10, sections 4 KiB aligned, Section 3 exactly as long as its entries
    Page16K = 1,  // table at 0x20, sections 16 KiB aligned, Section 3 padded to 16 KiB
};

enum class StrataProfile : uint16_t {
    Tiered      = 0,
    Unified     = 1,
    EngramShard = 2,
    VisionShard = 3,
};

#pragma pack(push, 1)
struct StrataSectionEntry {
    uint64_t file_offset;
    uint64_t byte_length;
    uint8_t  sha256[32];
    uint32_t section_id;
    uint32_t flags;
    uint64_t reserved;
};
static_assert(sizeof(StrataSectionEntry) == 64, "StrataSectionEntry must be 64 B");

struct StrataExpertEntry {
    uint16_t expert_id;
    uint8_t  gu_code_id;
    uint8_t  dn_code_id;
    uint16_t cluster_id;
    uint16_t flags;
    uint64_t body_offset;
    uint32_t body_bytes;
    uint32_t dn_offset;
    uint32_t aux_offsets[2];
};
static_assert(sizeof(StrataExpertEntry) == 32, "StrataExpertEntry must be 32 B");
#pragma pack(pop)

struct StrataHeader {
    uint32_t flags = 0;
    StrataProfile profile = StrataProfile::Tiered;
    uint16_t num_sections = 0;
    StrataLayout layout = StrataLayout::Page4K;
    uint32_t page_size = STRATA_PAGE_SIZE;                    // the alignment of the sections
    uint32_t section_table_offset = STRATA_TABLE_OFFSET_4K;   // where the 6 section entries start in the header frame
    uint16_t tile_layout_id = 0;                              // 16 KiB layout only: the u16 at 0x10, recorded, not interpreted
    StrataSectionEntry sections[STRATA_NUM_SECTIONS]{};
    uint8_t engram_shard_sha256[32]{};
};

class StrataFileReader {
public:
    StrataFileReader() = default;
    ~StrataFileReader() { close(); }
    StrataFileReader(const StrataFileReader&) = delete;
    StrataFileReader& operator=(const StrataFileReader&) = delete;

    bool open(const std::string& path, std::string& err);
    void close();
    bool is_open() const noexcept { return fp_ != nullptr; }

    const StrataHeader& header() const noexcept { return hdr_; }
    uint8_t format_minor() const noexcept { return static_cast<uint8_t>(hdr_.flags & 0xFFu); }
    const std::string& manifest_json() const noexcept { return manifest_json_; }
    const std::vector<StrataExpertEntry>& expert_directory() const noexcept { return experts_; }
    const std::vector<uint16_t>& vram_load_order() const noexcept { return vram_load_order_; }

    // A top-level string / unsigned number of the manifest ("arch", "num_physical_layers"); false when absent or of another kind.
    bool manifest_top_string(const char* key, std::string& out) const;
    bool manifest_top_uint(const char* key, uint64_t& out) const;
    // The Inkling Vision shard: header profile VisionShard, or a manifest whose top-level arch is inkling2_vision (a file whose
    // header carries another profile value still says what it is in its manifest).
    bool is_vision_shard() const;
    uint64_t file_size() const noexcept { return file_size_; }

    bool read_section(uint32_t section_id, std::vector<uint8_t>& out, std::string& err) const;
    bool read_at(uint64_t abs_offset, void* dst, size_t bytes, std::string& err) const;

    // Convenience: read one expert body from Section 4 by [moe_layer * n_experts + rank]
    bool read_expert_body(size_t dir_index, std::vector<uint8_t>& out, std::string& err) const;

private:
    std::FILE* fp_ = nullptr;
    uint64_t file_size_ = 0;
    StrataHeader hdr_{};
    std::string manifest_json_;
    std::vector<StrataExpertEntry> experts_;
    std::vector<uint16_t> vram_load_order_;
};

}  // namespace strata::artifact
