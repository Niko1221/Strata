#include "strata/program/media_embeddings.hpp"
#include "strata/program/video_limits.hpp"

#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#ifdef _WIN32
#include <fcntl.h>
#include <io.h>
#endif

using namespace strata::program;
namespace {
int checks = 0;
void check(bool ok, const char* message) {
    ++checks;
    if (!ok) { std::cerr << "FAIL: " << message << '\n'; std::exit(1); }
}
MediaBundle sample() {
    MediaBundle b;
    b.width = 2; b.tokens = {11, 20, 248056, 248056, 21, 248057, 248057, 22};
    VisualSpan image;
    image.start = 2; image.pad_id = 248056; image.kind = MediaKind::Image; image.advance = 2;
    image.nx = 2; image.ny = 1; image.positions = {{0, 0, 0}, {0, 0, 1}};
    image.embeddings = {.25f, -0.f, .5f, -1.f};
    VisualSpan video;
    video.start = 5; video.pad_id = 248057; video.kind = MediaKind::Video; video.advance = 4;
    video.positions = {{0, 0, 0}, {3, 1, 2}}; video.embeddings = {2, 3, 4, 5};
    b.spans = {image, video};
    return b;
}
std::string bytes(const MediaBundle& b, const MediaLimits& limits = {}) {
    std::ostringstream out(std::ios::binary);
    write_media(out, b, limits);
    return out.str();
}
MediaBundle parse(const std::string& data, const MediaLimits& limits = {}) {
    std::istringstream in(data, std::ios::binary);
    return read_media(in, limits);
}
template<class F> void refused(F operation, const char* message) {
    bool caught = false;
    try { operation(); } catch (const MediaError&) { caught = true; }
    check(caught, message);
}
std::string hex_file(const char* path) {
    std::ifstream in(path);
    check(bool(in), "open shared golden");
    std::string result;
    int high = -1;
    for (int c; (c = in.get()) != std::char_traits<char>::eof();) {
        if (std::isspace(static_cast<unsigned char>(c))) continue;
        int n = c >= '0' && c <= '9' ? c - '0' : c >= 'a' && c <= 'f' ? c - 'a' + 10 : -1;
        check(n >= 0, "valid fixture hex");
        if (high == -1) high = n;
        else { result.push_back(char(high * 16 + n)); high = -1; }
    }
    check(high == -1, "complete fixture hex");
    return result;
}
void tests(const char* fixture) {
    const auto b = sample();
    const auto raw = bytes(b);
    check(raw == hex_file(fixture), "C++ writer matches shared golden");
    check(bytes(parse(raw)) == raw, "C++ wire roundtrip preserves bits");
    auto plan = media_positions(b, 10);
    const std::vector<MediaPosition> expected = {{0,0,0}, {1,1,1}, {2,2,2}, {2,2,3}, {4,4,4},
                                                {5,5,5}, {8,6,7}, {9,9,9}, {10,10,10}, {11,11,11}};
    check(plan.positions == expected, "position mapping and generation tail");
    check(plan.rows[2].span == 0 && plan.rows[2].row == 0 && plan.rows[6].span == 1 && plan.rows[6].row == 1,
          "visual row ownership");
    check(plan.rows[0].span == MediaRow::text && plan.rows[9].span == MediaRow::text, "text rows unbound");
    refused([&] { media_positions(b, 7); }, "capacity below prompt");
    refused([&] { media_positions(b, (1u << 20) + 1); }, "capacity budget");
    for (size_t n = 0; n < raw.size(); ++n) refused([&] { parse(raw.substr(0, n)); }, "all truncation boundaries");
    refused([&] { parse(raw + "x"); }, "trailing garbage");
    for (size_t at : {size_t(4), size_t(6), size_t(8), size_t(40), size_t(48), size_t(56),
                      size_t(96+40), size_t(96+48), size_t(96+56)}) {
        auto changed = raw; changed[at] ^= 1;
        refused([&] { parse(changed); }, "header/offset/reserved mutation");
    }
    MediaLimits l;
    l.max_tokens = 7; refused([&] { parse(raw, l); }, "token budget");
    l = {}; l.max_spans = 1; refused([&] { parse(raw, l); }, "span budget");
    l = {}; l.max_rows = 3; refused([&] { parse(raw, l); }, "row budget");
    l = {}; l.max_width = 1; refused([&] { parse(raw, l); }, "width budget");
    l = {}; l.max_bytes = raw.size() - 1; refused([&] { parse(raw, l); }, "byte budget");
    l = {}; l.max_position = 7; refused([&] { parse(raw, l); }, "position budget");
    l = {}; l.max_position = 10; refused([&] { media_positions(b, 10, l); }, "tail position budget");
    l = {}; l.expected_width = 3; refused([&] { parse(raw, l); }, "model width binding");
    l = {}; l.vocab_size = 248057; refused([&] { parse(raw, l); }, "vocab binding");
    l = {}; l.allowed_pad_ids = {248056}; refused([&] { parse(raw, l); }, "pad binding");
    l = {}; l.max_rows = 0; refused([&] { parse(raw, l); }, "invalid limits");
    for (int mutation = 0; mutation < 10; ++mutation) {
        auto changed = b;
        switch (mutation) {
        case 0: changed.spans[0].start = 8; break;
        case 1: changed.spans[0].positions[1] = {0,1,0}; break;
        case 2: changed.spans[1].advance = 3; break;
        case 3: changed.spans[1].positions[1][0] = -1; break;
        case 4: changed.spans[1].kind = MediaKind(3); break;
        case 5: changed.spans[0].embeddings[0] = std::numeric_limits<float>::quiet_NaN(); break;
        case 6: changed.tokens[2] = 21; break;
        case 7: changed.spans[1].start = 3; break;
        case 8: changed.spans[1].nx = 2; break;
        case 9: changed.spans[0].embeddings.pop_back(); break;
        }
        std::ostringstream out;
        refused([&] { write_media(out, changed); }, "semantic mutations");
        check(out.str().empty(), "invalid input writes nothing");
    }
    const uint64_t original = media_span_fingerprint(b, 1);
    auto changed = b; changed.spans[1].advance++;
    check(original != media_span_fingerprint(changed, 1), "fingerprint covers position advance");
    changed = b; changed.spans[1].positions[1][1]++;
    check(original != media_span_fingerprint(changed, 1), "fingerprint covers coordinates");
    changed = b; changed.spans[1].embeddings.back()++;
    check(original != media_span_fingerprint(changed, 1), "fingerprint covers embedding payload");
    LegacyImage im; im.width = 2; im.nx = 2; im.ny = 1; im.embeddings = b.spans[0].embeddings;
    std::ostringstream old(std::ios::binary);
    write_legacy_images(old, {im});
    std::istringstream in(old.str(), std::ios::binary);
    auto images = read_legacy_images(in);
    const auto adapted = adapt_legacy_images(images, {11,248056,248056,12}, 248056);
    check(media_positions(adapted, 6).positions == std::vector<MediaPosition>{{0,0,0},{1,1,1},{1,1,2},{3,3,3},{4,4,4},{5,5,5}},
          "legacy positions match image host loop");
    check(media_span_fingerprint(b, 0) == media_span_fingerprint(adapted, 0), "legacy fingerprint unchanged");
    for (size_t n = 1; n < old.str().size(); ++n) {
        std::istringstream cut(old.str().substr(0, n), std::ios::binary);
        refused([&] { read_legacy_images(cut); }, "legacy truncation boundaries");
    }
    refused([&] { adapt_legacy_images(images, {11,248056,12}, 248056); }, "legacy short pad run");
    refused([&] { adapt_legacy_images(images, {11,248056,248056}, 248056); }, "legacy terminal image refused");
    check(media_fingerprints(b) == std::vector<uint64_t>{media_span_fingerprint(b, 0),
                                                       media_span_fingerprint(b, 1)}, "batch fingerprints preserve legacy/video identity");
    MediaBundle q;
    q.width = 2560; q.tokens = {11, 248053, 248057, 248057, 248054, 12};
    VisualSpan group;
    group.kind = MediaKind::Video; group.pad_id = 248057; group.start = 2; group.advance = 2;
    group.positions = {{0,0,0}, {0,0,1}}; group.embeddings.resize(2 * q.width);
    q.spans = {group};
    validate_qwen4_media(q);
    const auto qr = bytes(q);
    std::istringstream qi(qr, std::ios::binary);
    check(bytes(read_media(qi, {}, true)) == qr, "Qwen4 profile read/roundtrip");
    MediaBundle many;
    many.width = 2560;
    for (size_t i = 0; i < 129; ++i) {
        auto span = group;
        span.start = many.tokens.size() + 2;
        many.tokens.insert(many.tokens.end(), q.tokens.begin(), q.tokens.end());
        many.spans.push_back(std::move(span));
    }
    MediaLimits video_limits;
    video_limits.max_spans = video_limits.max_rows;
    const auto many_wire = bytes(many, video_limits);
    refused([&] { parse(many_wire); }, "generic span cap remains 128");
    std::istringstream many_input(many_wire, std::ios::binary);
    auto decoded = read_media(many_input, video_limits, true);
    check(decoded.spans.size() == 129 && decoded.tokens == many.tokens, "video span budget accepts 129 groups");
    auto wrong_profile = q;
    wrong_profile.tokens[1] = 17;
    const auto short_wrong = bytes(wrong_profile).substr(0, 64 + q.tokens.size() * 4 + 64 + 24);
    std::istringstream early(short_wrong, std::ios::binary);
    try { read_media(early, {}, true); check(false, "invalid profile accepted"); }
    catch (const MediaError& e) {
        check(std::string(e.what()).find("delimiters") != std::string::npos, "profile rejection precedes embedding reads");
    }
    const std::vector<int64_t> other_request = {99, 248053, 248057, 248057, 248054, 12};
    std::istringstream binding(qr.substr(0, 64 + q.tokens.size() * 4), std::ios::binary);
    try { read_media(binding, {}, true, &other_request); check(false, "unbound request accepted"); }
    catch (const MediaError& e) {
        check(std::string(e.what()).find("match the request") != std::string::npos, "request binding precedes payload reads");
    }
    wrong_profile = q; wrong_profile.spans[0].positions[0][0] = 1;
    refused([&] { validate_qwen4_media(wrong_profile); }, "Qwen4 rejects nonzero relative video time");
    wrong_profile = q; wrong_profile.spans[0].advance = 3;
    refused([&] { validate_qwen4_media(wrong_profile); }, "Qwen4 rejects a wrong group advance");
    wrong_profile = q; wrong_profile.tokens.push_back(248057);
    refused([&] { validate_qwen4_media(wrong_profile); }, "Qwen4 rejects an unbound visual pad");
    uint32_t rng = 77;
    for (int i = 0; i < 4000; ++i) {
        auto fuzz = raw;
        rng = rng * 1664525u + 1013904223u;
        const size_t at = rng % fuzz.size();
        rng = rng * 1664525u + 1013904223u;
        fuzz[at] ^= char(1u << (rng % 8));
        try { check(bytes(parse(fuzz)) == fuzz, "accepted mutation remains canonical"); }
        catch (const MediaError&) { ++checks; }
    }
    std::cout << "media_embeddings_test: " << checks << " checks passed\n";
}
}  // namespace

int main(int argc, char** argv) {
    try {
        if (argc == 2) { tests(argv[1]); return 0; }
        if (argc < 3) throw MediaError("expected fixture.hex, or --roundtrip/--positions/--hashes/--legacy file");
        const std::string mode = argv[1];
        std::ifstream in(argv[2], std::ios::binary);
        if (!in) throw MediaError("cannot open media test input");
#ifdef _WIN32
        if (mode == "--roundtrip" || mode == "--legacy") _setmode(_fileno(stdout), _O_BINARY);
#endif
        if (mode == "--legacy" && argc == 4) {
            auto images = read_legacy_images(in);
            const int32_t pad = int32_t(std::stol(argv[3]));
            std::vector<int32_t> ids = {11};
            for (const auto& im : images) { ids.insert(ids.end(), size_t(im.nx) * im.ny, pad); ids.push_back(12); }
            write_media(std::cout, adapt_legacy_images(images, ids, pad));
            return 0;
        }
        if (mode == "--video-check" && argc == 3) {
            MediaLimits limits;
            limits.max_rows = video_limits::max_rows;
            limits.max_spans = limits.max_rows;
            limits.max_bytes = video_limits::max_wire_bytes;
            limits.expected_width = 2560;
            limits.allowed_pad_ids = {248057};
            const auto bundle = read_media(in, limits, true);
            const auto plan = media_positions(bundle, bundle.tokens.size() + 8, limits);
            uint64_t rows = 0;
            for (const auto& span : bundle.spans) rows += span.positions.size();
            check(plan.rows.size() == bundle.tokens.size() + 8, "long-video position plan");
            std::cout << "video groups=" << bundle.spans.size() << " rows=" << rows << '\n';
            return 0;
        }
        auto bundle = read_media(in);
        if (mode == "--roundtrip" && argc == 3) write_media(std::cout, bundle);
        else if (mode == "--hashes" && argc == 3) {
            for (size_t i = 0; i < bundle.spans.size(); ++i) std::cout << media_span_fingerprint(bundle, i) << '\n';
        } else if (mode == "--positions" && argc == 4) {
            const auto plan = media_positions(bundle, std::stoull(argv[3]));
            for (size_t i = 0; i < plan.positions.size(); ++i) {
                const auto& pos = plan.positions[i]; const auto& row = plan.rows[i];
                std::cout << pos[0] << ' ' << pos[1] << ' ' << pos[2] << ' '
                          << (row.span == MediaRow::text ? int64_t(-1) : int64_t(row.span)) << ' ' << row.row << '\n';
            }
        } else throw MediaError("unknown media test mode");
        return 0;
    } catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
}
