#include "strata/program/media_embeddings.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <istream>
#include <ostream>
#include <string>

namespace strata::program {
namespace {
constexpr uint64_t kHeader = 64, kSpan = 64;
constexpr uint32_t kV1 = 0x31455653, kV2 = 0x32455653;
constexpr uint64_t kI32 = uint64_t(std::numeric_limits<int32_t>::max());
static_assert(sizeof(float) == 4 && std::numeric_limits<float>::is_iec559, "SVE requires IEEE float32");

void need(bool ok, const char* message) {
    if (!ok) throw MediaError(message);
}
uint64_t add(uint64_t a, uint64_t b) {
    need(b <= std::numeric_limits<uint64_t>::max() - a, "media size overflow");
    return a + b;
}
uint64_t mul(uint64_t a, uint64_t b) {
    need(a == 0 || b <= std::numeric_limits<uint64_t>::max() / a, "media size overflow");
    return a * b;
}
size_t size_of(uint64_t n) {
    need(n <= std::numeric_limits<size_t>::max(), "media size exceeds address space");
    return size_t(n);
}
uint64_t wire_size(uint64_t nt, uint64_t ns, uint64_t nr, uint64_t width) {
    return add(add(kHeader, mul(nt, 4)), add(mul(ns, kSpan), mul(nr, add(12, mul(width, 4)))));
}
void check_limits(const MediaLimits& l) {
    need(l.max_tokens && l.max_spans && l.max_rows && l.max_width && l.max_bytes, "invalid media limits");
    for (uint64_t n : {l.max_tokens, l.max_spans, l.max_rows, l.max_width, l.max_bytes})
        need(n <= uint64_t(std::numeric_limits<int64_t>::max()), "invalid media limits");
    need(l.max_position >= 0 && l.vocab_size > 0 && uint64_t(l.vocab_size) <= kI32 + 1, "invalid position/token limits");
    need(l.expected_width <= l.max_width, "invalid expected width");
    for (int32_t pad : l.allowed_pad_ids) need(pad >= 0 && uint32_t(pad) < l.vocab_size, "invalid allowed pad ID");
}
void width_ok(uint32_t width, const MediaLimits& l) {
    need(width > 0 && width <= l.max_width && width <= kI32, "invalid embedding width");
    need(l.expected_width == 0 || width == l.expected_width, "embedding width mismatch");
}
void token_ok(int32_t token, const MediaLimits& l) {
    need(token >= 0 && uint32_t(token) < l.vocab_size, "invalid token ID");
}
void span_meta(const VisualSpan& s, uint64_t count, uint64_t nt, uint64_t end, const MediaLimits& l) {
    need(count > 0 && s.start >= end && s.start <= nt && count <= nt - s.start, "invalid span range");
    token_ok(s.pad_id, l);
    need(l.allowed_pad_ids.empty() || std::find(l.allowed_pad_ids.begin(), l.allowed_pad_ids.end(), s.pad_id) !=
         l.allowed_pad_ids.end(), "unsupported pad ID");
    need(s.advance > 0 && s.advance <= uint64_t(l.max_position) + 1, "invalid position advance");
    need(s.nx <= kI32 && s.ny <= kI32, "invalid image grid");
    if (s.kind == MediaKind::Image) {
        need(s.nx > 0 && s.ny > 0 && mul(s.nx, s.ny) == count, "invalid image grid");
        need(s.advance == std::max(s.nx, s.ny), "invalid image position advance");
    } else {
        need(s.kind == MediaKind::Video, "invalid media kind");
        need(s.nx == 0 && s.ny == 0, "video grid must be zero");
    }
}
void floats_ok(const std::vector<float>& v, uint64_t n) {
    need(v.size() == n, "embedding length mismatch");
    for (float f : v) need(std::isfinite(f), "non-finite embedding");
}
uint64_t read_u(std::istream& in, int bytes) {
    uint64_t value = 0;
    for (int i = 0; i < bytes; ++i) {
        const int c = in.get();
        need(c != std::char_traits<char>::eof(), "truncated media file");
        value |= uint64_t(uint8_t(c)) << (i * 8);
    }
    return value;
}
int32_t signed_u(uint64_t u) {
    return u <= kI32 ? int32_t(u) : int32_t(int64_t(u) - int64_t(uint64_t(1) << 32));
}
int32_t read_i(std::istream& in) { return signed_u(read_u(in, 4)); }
void write_u(std::ostream& out, uint64_t u, int bytes) {
    for (int i = 0; i < bytes; ++i) out.put(char(uint8_t(u >> (i * 8))));
    if (!out) throw std::ios_base::failure("short media write");
}
uint32_t float_bits(float f) {
    uint32_t u;
    std::memcpy(&u, &f, 4);
    return u;
}
std::vector<float> read_floats(std::istream& in, uint64_t count) {
    std::vector<float> result;
    std::array<uint8_t, 16384> buf{};
    for (uint64_t done = 0; done < count;) {
        const size_t n = size_of(std::min<uint64_t>(count - done, buf.size() / 4));
        in.read(reinterpret_cast<char*>(buf.data()), std::streamsize(n * 4));
        need(in.gcount() == std::streamsize(n * 4) && !in.bad(), "truncated media file");
        const size_t start = result.size();
        result.resize(size_of(add(start, n)));
        for (size_t j = 0; j < n; ++j) {
            uint32_t u = 0;
            for (int k = 0; k < 4; ++k) u |= uint32_t(buf[j * 4 + size_t(k)]) << (k * 8);
            float f;
            std::memcpy(&f, &u, 4);
            need(std::isfinite(f), "non-finite embedding");
            result[start + j] = f;
        }
        done += n;
    }
    return result;
}
void write_floats(std::ostream& out, const std::vector<float>& values) {
    std::array<uint8_t, 16384> buf{};
    for (size_t done = 0; done < values.size();) {
        const size_t n = std::min(values.size() - done, buf.size() / 4);
        for (size_t j = 0; j < n; ++j) {
            const uint32_t u = float_bits(values[done + j]);
            for (int k = 0; k < 4; ++k) buf[j * 4 + size_t(k)] = uint8_t(u >> (k * 8));
        }
        out.write(reinterpret_cast<const char*>(buf.data()), std::streamsize(n * 4));
        if (!out) throw std::ios_base::failure("short media write");
        done += n;
    }
}
void eof(std::istream& in) {
    need(in.peek() == std::char_traits<char>::eof() && in.eof() && !in.bad(), "trailing media bytes or read error");
}
void legacy_ok(const LegacyImage& im, const MediaLimits& l) {
    width_ok(im.width, l);
    need(im.nx > 0 && im.ny > 0 && im.nx <= kI32 && im.ny <= kI32, "invalid image grid");
    const uint64_t n = mul(im.nx, im.ny);
    need(n <= l.max_rows && n <= kI32, "image row budget exceeded");
    need(add(20, mul(mul(n, im.width), 4)) <= l.max_bytes, "image byte budget exceeded");
    floats_ok(im.embeddings, mul(n, im.width));
}
void hash_u(uint64_t& h, uint64_t u, int bytes) {
    for (int i = 0; i < bytes; ++i) h = (h ^ uint8_t(u >> (8 * i))) * 1099511628211ull;
}
void validate_structure(const MediaBundle& b, const MediaLimits& l) {
    check_limits(l);
    width_ok(b.width, l);
    need(b.tokens.size() <= l.max_tokens && b.spans.size() <= l.max_spans, "media count budget exceeded");
    for (int32_t token : b.tokens) token_ok(token, l);
    uint64_t rows = 0, end = 0, base = 0;
    for (const VisualSpan& s : b.spans) {
        const uint64_t count = s.positions.size();
        rows = add(rows, count);
        need(rows <= l.max_rows, "row budget exceeded");
        need(wire_size(b.tokens.size(), b.spans.size(), rows, b.width) <= l.max_bytes, "media byte budget exceeded");
        span_meta(s, count, b.tokens.size(), end, l);
        base = add(base, s.start - end);
        uint64_t extent = 0;
        for (size_t j = 0; j < s.positions.size(); ++j) {
            const auto& pos = s.positions[j];
            for (int32_t coord : pos) {
                need(coord >= 0 && uint64_t(coord) <= uint64_t(l.max_position), "invalid relative position");
                need(add(base, uint64_t(coord)) <= uint64_t(l.max_position), "absolute position budget exceeded");
                extent = std::max(extent, uint64_t(coord));
            }
            if (s.kind == MediaKind::Image)
                need(pos == MediaPosition{0, int32_t(j / s.nx), int32_t(j % s.nx)}, "image positions do not match grid");
            need(b.tokens[size_of(s.start) + j] == s.pad_id, "span pad tokens do not match");
        }
        need(s.advance > extent, "position advance does not cover span");
        base = add(base, s.advance);
        end = add(s.start, count);
    }
    need(wire_size(b.tokens.size(), b.spans.size(), rows, b.width) <= l.max_bytes, "media byte budget exceeded");
    need(add(base, b.tokens.size() - end) <= uint64_t(l.max_position) + 1, "text position budget exceeded");
}

}  // namespace

void validate_media(const MediaBundle& b, const MediaLimits& l) {
    validate_structure(b, l);
    for (const auto& s : b.spans) floats_ok(s.embeddings, mul(s.positions.size(), b.width));
}

MediaBundle read_media(std::istream& in, const MediaLimits& l) {
    check_limits(l);
    const uint64_t magic = read_u(in, 4), version = read_u(in, 2), header = read_u(in, 2), flags = read_u(in, 4);
    need(magic == kV2 && version == 2 && header == kHeader && flags == 0, "unsupported media header");
    MediaBundle b;
    b.width = uint32_t(read_u(in, 4));
    width_ok(b.width, l);
    const uint64_t nt = read_u(in, 8), ns = read_u(in, 8), nr = read_u(in, 8);
    const uint64_t pb = read_u(in, 8), eb = read_u(in, 8), total = read_u(in, 8);
    need(nt <= l.max_tokens && ns <= l.max_spans && nr <= l.max_rows, "media count budget exceeded");
    need(ns <= nr && (ns != 0 || nr == 0), "invalid span/row counts");
    need(pb == mul(nr, 12) && eb == mul(mul(nr, b.width), 4) && total == wire_size(nt, ns, nr, b.width),
         "media byte counts mismatch");
    need(total <= l.max_bytes, "media byte budget exceeded");
    b.tokens.reserve(size_of(nt));
    for (uint64_t i = 0; i < nt; ++i) {
        const int32_t token = read_i(in);
        token_ok(token, l);
        b.tokens.push_back(token);
    }
    b.spans.reserve(size_of(ns));
    std::vector<uint64_t> counts;
    uint64_t po = 0, eo = 0, end = 0;
    for (uint64_t i = 0; i < ns; ++i) {
        VisualSpan s;
        s.start = read_u(in, 8);
        const uint64_t count = read_u(in, 8);
        s.kind = MediaKind(read_u(in, 4));
        s.pad_id = read_i(in);
        s.advance = read_u(in, 8);
        s.nx = uint32_t(read_u(in, 4)); s.ny = uint32_t(read_u(in, 4));
        const uint64_t pos_off = read_u(in, 8), emb_off = read_u(in, 8), reserved = read_u(in, 8);
        span_meta(s, count, nt, end, l);
        need(count <= nr && pos_off == po && emb_off == eo && reserved == 0, "noncanonical span offsets or flags");
        po = add(po, mul(count, 12)); eo = add(eo, mul(mul(count, b.width), 4));
        need(po <= pb && eo <= eb, "span payload exceeds budget");
        counts.push_back(count);
        b.spans.push_back(std::move(s));
        end = add(b.spans.back().start, count);
    }
    need(po == pb && eo == eb, "unused media payload");
    for (size_t i = 0; i < b.spans.size(); ++i) {
        auto& pos = b.spans[i].positions;
        pos.reserve(size_of(counts[i]));
        for (uint64_t row = 0; row < counts[i]; ++row) pos.push_back({read_i(in), read_i(in), read_i(in)});
    }
    validate_structure(b, l);
    for (size_t i = 0; i < b.spans.size(); ++i)
        b.spans[i].embeddings = read_floats(in, mul(counts[i], b.width));
    eof(in);
    validate_media(b, l);
    return b;
}

void write_media(std::ostream& out, const MediaBundle& b, const MediaLimits& l) {
    validate_media(b, l);
    uint64_t nr = 0;
    for (const auto& s : b.spans) nr = add(nr, s.positions.size());
    write_u(out, kV2, 4); write_u(out, 2, 2); write_u(out, kHeader, 2); write_u(out, 0, 4);
    write_u(out, b.width, 4); write_u(out, b.tokens.size(), 8); write_u(out, b.spans.size(), 8);
    write_u(out, nr, 8); write_u(out, mul(nr, 12), 8); write_u(out, mul(mul(nr, b.width), 4), 8);
    write_u(out, wire_size(b.tokens.size(), b.spans.size(), nr, b.width), 8);
    for (int32_t token : b.tokens) write_u(out, uint32_t(token), 4);
    uint64_t po = 0, eo = 0;
    for (const auto& s : b.spans) {
        write_u(out, s.start, 8); write_u(out, s.positions.size(), 8); write_u(out, uint32_t(s.kind), 4);
        write_u(out, uint32_t(s.pad_id), 4); write_u(out, s.advance, 8); write_u(out, s.nx, 4); write_u(out, s.ny, 4);
        write_u(out, po, 8); write_u(out, eo, 8); write_u(out, 0, 8);
        po = add(po, mul(s.positions.size(), 12)); eo = add(eo, mul(mul(s.positions.size(), b.width), 4));
    }
    for (const auto& s : b.spans) for (const auto& pos : s.positions) for (int32_t p : pos) write_u(out, uint32_t(p), 4);
    for (const auto& s : b.spans) write_floats(out, s.embeddings);
}

MediaPositionPlan media_positions(const MediaBundle& b, uint64_t capacity, const MediaLimits& l) {
    validate_media(b, l);
    need(capacity >= b.tokens.size() && capacity <= l.max_tokens, "invalid position capacity");
    uint64_t next = b.tokens.size();
    for (const auto& s : b.spans) next = add(next - s.positions.size(), s.advance);
    need(add(next, capacity - b.tokens.size()) <= uint64_t(l.max_position) + 1, "generation position budget exceeded");
    MediaPositionPlan plan;
    plan.positions.reserve(size_of(capacity)); plan.rows.reserve(size_of(capacity));
    uint64_t cell = 0, base = 0;
    auto text = [&]() {
        const auto p = int32_t(base++);
        plan.positions.push_back({p, p, p}); plan.rows.push_back({}); ++cell;
    };
    for (size_t i = 0; i < b.spans.size(); ++i) {
        const auto& s = b.spans[i];
        while (cell < s.start) text();
        need(i < MediaRow::text, "too many row references");
        for (size_t j = 0; j < s.positions.size(); ++j) {
            const auto& pos = s.positions[j];
            plan.positions.push_back({int32_t(base + uint64_t(pos[0])), int32_t(base + uint64_t(pos[1])),
                                      int32_t(base + uint64_t(pos[2]))});
            plan.rows.push_back({uint32_t(i), uint64_t(j)});
        }
        cell += s.positions.size(); base += s.advance;
    }
    while (cell < capacity) text();
    return plan;
}

std::vector<LegacyImage> read_legacy_images(std::istream& in, const MediaLimits& l) {
    check_limits(l);
    std::vector<LegacyImage> images;
    uint64_t rows = 0, used = 0;
    while (in.peek() != std::char_traits<char>::eof()) {
        need(read_u(in, 4) == kV1, "invalid legacy image header");
        const int32_t count = read_i(in), nx = read_i(in), ny = read_i(in), width = read_i(in);
        need(nx > 0 && ny > 0 && count > 0 && mul(uint64_t(nx), uint64_t(ny)) == uint64_t(count), "invalid legacy image grid");
        need(width > 0, "invalid embedding width");
        width_ok(uint32_t(width), l);
        rows = add(rows, uint64_t(count));
        const uint64_t n = mul(uint64_t(count), uint64_t(width));
        used = add(used, add(20, mul(n, 4)));
        need(rows <= l.max_rows && images.size() < l.max_spans, "image count budget exceeded");
        need(used <= l.max_bytes, "image byte budget exceeded");
        LegacyImage im;
        im.width = uint32_t(width); im.nx = uint32_t(nx); im.ny = uint32_t(ny);
        im.embeddings = read_floats(in, n);
        images.push_back(std::move(im));
    }
    eof(in);
    return images;
}

void write_legacy_images(std::ostream& out, const std::vector<LegacyImage>& images, const MediaLimits& l) {
    check_limits(l);
    need(images.size() <= l.max_spans, "image count budget exceeded");
    uint64_t rows = 0, used = 0;
    for (const auto& im : images) {
        legacy_ok(im, l);
        rows = add(rows, mul(im.nx, im.ny)); used = add(used, add(20, mul(im.embeddings.size(), 4)));
    }
    need(rows <= l.max_rows && used <= l.max_bytes, "image budget exceeded");
    for (const auto& im : images) {
        write_u(out, kV1, 4); write_u(out, mul(im.nx, im.ny), 4);
        write_u(out, im.nx, 4); write_u(out, im.ny, 4); write_u(out, im.width, 4);
        write_floats(out, im.embeddings);
    }
}

MediaBundle adapt_legacy_images(const std::vector<LegacyImage>& images, const std::vector<int32_t>& tokens,
                              int32_t pad_id, const MediaLimits& l) {
    check_limits(l);
    token_ok(pad_id, l);
    need(!images.empty() && images.size() <= l.max_spans, "invalid legacy image count");
    need(tokens.size() <= l.max_tokens, "token budget exceeded");
    MediaBundle b;
    b.width = images[0].width; b.tokens = tokens;
    size_t cell = 0, index = 0;
    while (cell < tokens.size()) {
        if (tokens[cell] != pad_id) { ++cell; continue; }
        need(index < images.size(), "more image tokens than records");
        const auto& im = images[index++];
        legacy_ok(im, l);
        need(im.width == b.width, "image widths differ");
        const uint64_t count = mul(im.nx, im.ny);
        need(count <= tokens.size() - cell, "image extends past tokens");
        VisualSpan s;
        s.start = cell; s.pad_id = pad_id; s.kind = MediaKind::Image;
        s.advance = std::max(im.nx, im.ny); s.nx = im.nx; s.ny = im.ny; s.embeddings = im.embeddings;
        for (uint64_t j = 0; j < count; ++j) s.positions.push_back({0, int32_t(j / im.nx), int32_t(j % im.nx)});
        b.spans.push_back(std::move(s));
        cell += size_of(count);
    }
    need(index == images.size(), "more image records than tokens");
    need(tokens.empty() || tokens.back() != pad_id, "prompt cannot end in an image");
    validate_media(b, l);
    return b;
}

uint64_t media_span_fingerprint(const MediaBundle& b, size_t index, const MediaLimits& l) {
    validate_media(b, l);
    need(index < b.spans.size(), "invalid span index");
    const auto& s = b.spans[index];
    uint64_t h = 1469598103934665603ull;  // Existing image-cache seed.
    if (s.kind == MediaKind::Image) {
        hash_u(h, s.positions.size(), 8); hash_u(h, s.nx, 8); hash_u(h, s.ny, 8);
    } else {
        hash_u(h, kV2, 4); hash_u(h, b.width, 4); hash_u(h, uint32_t(s.kind), 4); hash_u(h, uint32_t(s.pad_id), 4);
        hash_u(h, s.start, 8); hash_u(h, s.positions.size(), 8); hash_u(h, s.advance, 8);
        for (const auto& pos : s.positions) for (int32_t p : pos) hash_u(h, uint32_t(p), 4);
    }
    for (float f : s.embeddings) hash_u(h, float_bits(f), 4);
    return h;
}

}  // namespace strata::program
