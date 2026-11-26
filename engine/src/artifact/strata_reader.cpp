// src/artifact/strata_reader.cpp - Native .strata v1.0 container reader (docs/FORMAT-SPEC.md §2)
// Reads both header layouts (4 KiB and 16 KiB aligned): strata_layout.hpp.
#include "strata/artifact/strata_reader.hpp"
#include "strata/artifact/strata_layout.hpp"

#include <zstd.h>

#include <cstring>

namespace strata::artifact {
namespace {

bool seek_abs(std::FILE* fp, uint64_t off) noexcept {
#if defined(_WIN32)
    return _fseeki64(fp, static_cast<__int64>(off), SEEK_SET) == 0;
#else
    return fseeko(fp, static_cast<off_t>(off), SEEK_SET) == 0;
#endif
}

}  // namespace

void StrataFileReader::close() {
    if (fp_) {
        std::fclose(fp_);
        fp_ = nullptr;
    }
    file_size_ = 0;
    manifest_json_.clear();
    experts_.clear();
    vram_load_order_.clear();
}

bool StrataFileReader::manifest_top_string(const char* key, std::string& out) const {
    return strata_manifest_top_value(manifest_json_, key, true, out);
}

bool StrataFileReader::manifest_top_uint(const char* key, uint64_t& out) const {
    std::string t;
    if (!strata_manifest_top_value(manifest_json_, key, false, t)) return false;
    out = std::strtoull(t.c_str(), nullptr, 10);
    return true;
}

bool StrataFileReader::is_vision_shard() const {
    if (hdr_.profile == StrataProfile::VisionShard) return true;
    std::string arch;
    return manifest_top_string("arch", arch) && arch == "inkling2_vision";
}

bool StrataFileReader::open(const std::string& path, std::string& err) {
    close();
    fp_ = std::fopen(path.c_str(), "rb");
    if (!fp_) {
        err = "Failed to open .strata file: " + path;
        return false;
    }
    uint8_t frame[STRATA_PAGE_SIZE];
    if (std::fread(frame, 1, STRATA_PAGE_SIZE, fp_) != STRATA_PAGE_SIZE) {
        err = "Short read on .strata 4 KiB header frame";
        close();
        return false;
    }
    file_size_ = strata_file_size(fp_);
    if (!strata_parse_header_frame(frame, STRATA_PAGE_SIZE, file_size_, hdr_, err)) {
        close();
        return false;
    }

    // Decompress ZSTD Section 0 manifest
    std::vector<uint8_t> s0;
    if (!read_section(0, s0, err)) {
        close();
        return false;
    }
    if (!s0.empty()) {
        unsigned long long const rsize = ZSTD_getFrameContentSize(s0.data(), s0.size());
        if (rsize == ZSTD_CONTENTSIZE_ERROR) {
            err = "Section 0 manifest is not valid ZSTD";
            close();
            return false;
        }
        if (rsize != ZSTD_CONTENTSIZE_UNKNOWN) {
            manifest_json_.resize(static_cast<size_t>(rsize));
            size_t const dsize = ZSTD_decompress(manifest_json_.data(), manifest_json_.size(), s0.data(), s0.size());
            if (ZSTD_isError(dsize)) {
                err = std::string("ZSTD decompress failed: ") + ZSTD_getErrorName(dsize);
                close();
                return false;
            }
            manifest_json_.resize(dsize);
        } else {
            // Streaming decompress fallback
            ZSTD_DCtx* dctx = ZSTD_createDCtx();
            ZSTD_inBuffer in_b{s0.data(), s0.size(), 0};
            std::vector<char> chunk(1 << 20);
            while (in_b.pos < in_b.size) {
                ZSTD_outBuffer out_b{chunk.data(), chunk.size(), 0};
                size_t ret = ZSTD_decompressStream(dctx, &out_b, &in_b);
                if (ZSTD_isError(ret)) {
                    err = std::string("ZSTD stream decompress failed: ") + ZSTD_getErrorName(ret);
                    ZSTD_freeDCtx(dctx);
                    close();
                    return false;
                }
                manifest_json_.append(chunk.data(), out_b.pos);
            }
            ZSTD_freeDCtx(dctx);
        }
    }

    // Load Section 3 expert directory (and optional v1.1 vram_load_order priority table when section[3].flags & 1)
    if (hdr_.sections[3].byte_length > 0) {
        std::vector<uint8_t> s3;
        if (!read_section(3, s3, err)) {
            close();
            return false;
        }
        const bool has_vram_order = ((hdr_.sections[3].flags & 1u) != 0);
        // 4 KiB layout: the length is exact.  16 KiB layout: it is padded, the manifest says how many entries there are.
        const uint64_t manifest_n = (hdr_.layout == StrataLayout::Page16K) ? strata_manifest_expert_count(manifest_json_) : 0;
        size_t n = 0;
        if (!strata_section3_entry_count(hdr_, s3.size(), has_vram_order, manifest_n, n, err)) {
            close();
            return false;
        }
        if (has_vram_order) {
            experts_.resize(n);
            vram_load_order_.resize(n);
            const size_t dir_bytes = n * sizeof(StrataExpertEntry);
            std::memcpy(experts_.data(), s3.data(), dir_bytes);
            std::memcpy(vram_load_order_.data(), s3.data() + dir_bytes, n * sizeof(uint16_t));
        } else {
            // padded 16 KiB Section 3 of a manifest without a count (Hy4): the zero entries at its tail are padding, a real
            // entry always has a body
            if (hdr_.layout == StrataLayout::Page16K && manifest_n == 0) {
                static const uint8_t kZero[sizeof(StrataExpertEntry)] = {};
                while (n > 0 && std::memcmp(s3.data() + (n - 1) * sizeof(StrataExpertEntry), kZero, sizeof(kZero)) == 0) --n;
            }
            experts_.resize(n);
            if (n) std::memcpy(experts_.data(), s3.data(), n * sizeof(StrataExpertEntry));
        }
    }

    return true;
}

bool StrataFileReader::read_at(uint64_t abs_offset, void* dst, size_t bytes, std::string& err) const {
    if (!fp_) {
        err = "File not open";
        return false;
    }
    if (!seek_abs(fp_, abs_offset)) {
        err = "Seek failed at offset " + std::to_string(abs_offset);
        return false;
    }
    if (std::fread(dst, 1, bytes, fp_) != bytes) {
        err = "Short read at offset " + std::to_string(abs_offset);
        return false;
    }
    return true;
}

bool StrataFileReader::read_section(uint32_t section_id, std::vector<uint8_t>& out, std::string& err) const {
    if (section_id >= STRATA_NUM_SECTIONS) {
        err = "Invalid section_id";
        return false;
    }
    const auto& sec = hdr_.sections[section_id];
    // a header whose lengths run past the file would ask for an allocation of that size: say so instead
    if (file_size_ != 0 && (sec.file_offset > file_size_ || sec.byte_length > file_size_ - sec.file_offset) && sec.byte_length != 0) {
        err = "Section " + std::to_string(section_id) + " extends past the end of the file (truncated file?)";
        return false;
    }
    out.resize(static_cast<size_t>(sec.byte_length));
    if (sec.byte_length == 0) return true;
    return read_at(sec.file_offset, out.data(), out.size(), err);
}

bool StrataFileReader::read_expert_body(size_t dir_index, std::vector<uint8_t>& out, std::string& err) const {
    if (dir_index >= experts_.size()) {
        err = "Expert directory index out of bounds";
        return false;
    }
    const auto& e = experts_[dir_index];
    out.resize(e.body_bytes);
    return read_at(hdr_.sections[4].file_offset + e.body_offset, out.data(), e.body_bytes, err);
}

}  // namespace strata::artifact
