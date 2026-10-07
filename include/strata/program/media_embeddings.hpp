#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <iosfwd>
#include <limits>
#include <stdexcept>
#include <vector>

namespace strata::program {

class MediaError : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

enum class MediaKind : uint32_t { Image = 1, Video = 2 };
using MediaPosition = std::array<int32_t, 3>;  // t, h, w

struct MediaLimits {
    uint64_t max_tokens = 1u << 20;
    uint64_t max_spans = 128;
    uint64_t max_rows = 16384;
    uint64_t max_width = 16384;
    uint64_t max_bytes = 256u << 20;
    int32_t max_position = std::numeric_limits<int32_t>::max();
    uint32_t vocab_size = uint32_t(std::numeric_limits<int32_t>::max()) + 1u;
    uint32_t expected_width = 0;
    std::vector<int32_t> allowed_pad_ids;
};

struct VisualSpan {
    uint64_t start = 0;
    int32_t pad_id = 0;
    MediaKind kind = MediaKind::Image;
    uint64_t advance = 0;
    std::vector<MediaPosition> positions;
    std::vector<float> embeddings;
    uint32_t nx = 0, ny = 0;
};

struct MediaBundle {
    uint32_t width = 0;
    std::vector<int32_t> tokens;
    std::vector<VisualSpan> spans;
};

struct LegacyImage {
    uint32_t width = 0, nx = 0, ny = 0;
    std::vector<float> embeddings;
};

struct MediaRow {
    static constexpr uint32_t text = std::numeric_limits<uint32_t>::max();
    uint32_t span = text;
    uint64_t row = 0;
};

struct MediaPositionPlan {
    std::vector<MediaPosition> positions;
    std::vector<MediaRow> rows;
};

void validate_media(const MediaBundle& bundle, const MediaLimits& limits = {});
MediaBundle read_media(std::istream& in, const MediaLimits& limits = {});
void write_media(std::ostream& out, const MediaBundle& bundle, const MediaLimits& limits = {});
MediaPositionPlan media_positions(const MediaBundle& bundle, uint64_t capacity, const MediaLimits& limits = {});
std::vector<LegacyImage> read_legacy_images(std::istream& in, const MediaLimits& limits = {});
void write_legacy_images(std::ostream& out, const std::vector<LegacyImage>& images, const MediaLimits& limits = {});
MediaBundle adapt_legacy_images(const std::vector<LegacyImage>& images, const std::vector<int32_t>& tokens,
                              int32_t pad_id, const MediaLimits& limits = {});
uint64_t media_span_fingerprint(const MediaBundle& bundle, size_t index, const MediaLimits& limits = {});

}  // namespace strata::program
