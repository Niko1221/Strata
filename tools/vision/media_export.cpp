#include "media_export.hpp"

#include "gguf.h"
#include "llama.h"
#include "mtmd.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <locale>
#include <memory>
#include <sstream>
#include <vector>

namespace strata::vision {
namespace {
using program::MediaBundle;
using program::MediaError;
using program::MediaKind;
using program::VisualSpan;
constexpr int32_t image_pad = 248056, video_pad = 248057, vision_start = 248053, vision_end = 248054;
using Bitmap = std::unique_ptr<mtmd_bitmap, decltype(&mtmd_bitmap_free)>;
using Chunks = std::unique_ptr<mtmd_input_chunks, decltype(&mtmd_input_chunks_free)>;

void need(bool ok, const char* what) { if (!ok) throw MediaError(what); }
void read(std::istream& in, void* data, size_t n) {
    in.read(static_cast<char*>(data), static_cast<std::streamsize>(n));
    need(bool(in), "truncated RGB frame spool");
}
uint64_t read_le(std::istream& in, size_t bytes) {
    unsigned char data[8]{};
    read(in, data, bytes);
    uint64_t value = 0;
    for (size_t i = 0; i < bytes; ++i) value |= uint64_t(data[i]) << (8 * i);
    return value;
}
double read_double(std::istream& in) {
    const uint64_t bits = read_le(in, 8);
    double value;
    static_assert(sizeof(value) == sizeof(bits) && std::numeric_limits<double>::is_iec559);
    std::memcpy(&value, &bits, sizeof(value));
    return value;
}

program::MediaLimits video_media_limits(uint32_t width, const VideoExportLimits& limits) {
    program::MediaLimits media;
    media.expected_width = width;
    media.max_rows = limits.rows;
    media.max_spans = limits.rows;
    media.max_bytes = limits.embedding_bytes;
    media.allowed_pad_ids = {video_pad};
    return media;
}

void append_group(MediaBundle& bundle, mtmd_context* ctx, uint32_t w, uint32_t h,
                  const std::vector<unsigned char>& first, const std::vector<unsigned char>& second,
                  double timestamp, const VideoExportLimits& limits) {
    Bitmap a(mtmd_bitmap_init(w, h, first.data()), mtmd_bitmap_free);
    Bitmap b(mtmd_bitmap_init(w, h, second.data()), mtmd_bitmap_free);
    Chunks chunks(mtmd_input_chunks_init(), mtmd_input_chunks_free);
    need(a && b && chunks, "cannot allocate a temporal frame group");
    mtmd_bitmap_set_mergeable(a.get(), true);
    mtmd_bitmap_set_mergeable(b.get(), true);
    // The bundled helper's 4 FPS / five-second markers are NOT the model's
    // processor convention. Every pair gets the pinned processor's wrapper.
    std::ostringstream stamp;
    stamp.imbue(std::locale::classic());
    stamp << '<' << std::fixed << std::setprecision(1) << timestamp << " seconds>";
    const std::string text = stamp.str() + mtmd_default_marker() + mtmd_default_marker();
    mtmd_input_text input{text.c_str(), text.size(), false, true};
    const mtmd_bitmap* bitmaps[2] = {a.get(), b.get()};
    need(mtmd_tokenize(ctx, chunks.get(), &input, bitmaps, 2) == 0, "cannot preprocess the temporal frame pair");
    size_t visual = 0;
    for (size_t c = 0; c < mtmd_input_chunks_size(chunks.get()); ++c) {
        const mtmd_input_chunk* chunk = mtmd_input_chunks_get(chunks.get(), c);
        if (mtmd_input_chunk_get_type(chunk) == MTMD_INPUT_CHUNK_TYPE_TEXT) {
            size_t n = 0;
            const llama_token* tokens = mtmd_input_chunk_get_tokens_text(chunk, &n);
            need(n <= 256, "unexpected frame-wrapper token count");
            if (n > 0) bundle.tokens.insert(bundle.tokens.end(), tokens, tokens + n);
            continue;
        }
        need(mtmd_input_chunk_get_type(chunk) == MTMD_INPUT_CHUNK_TYPE_IMAGE && ++visual == 1,
             "the model did not merge exactly two frames into one visual group");
        const size_t rows = mtmd_input_chunk_get_n_tokens(chunk);
        const uint64_t expected = uint64_t(w / 32) * (h / 32);
        need(rows == expected && rows <= limits.group_tokens, "mtmd changed the video resolution policy");
        const uint64_t floats = uint64_t(rows) * bundle.width;
        uint64_t previous = 0;
        for (const auto& span : bundle.spans) previous += span.positions.size();
        need(previous + rows <= limits.rows && (previous + rows) * bundle.width * 4 <= limits.embedding_bytes,
             "video visual row/embedding budget exceeded");
        VisualSpan span;
        span.start = bundle.tokens.size();
        span.kind = MediaKind::Video;
        span.pad_id = video_pad;
        span.advance = uint64_t(mtmd_input_chunk_get_n_pos(chunk));
        const mtmd_image_tokens* image = mtmd_input_chunk_get_tokens_image(chunk);
        for (size_t i = 0; i < rows; ++i) {
            const mtmd_decoder_pos pos = mtmd_image_tokens_get_decoder_pos(image, 0, i);
            need(pos.t == 0 && pos.y == i / (w / 32) && pos.x == i % (w / 32) && pos.z == 0,
                 "mtmd video positions disagree with the Qwen processor profile");
            span.positions.push_back({int32_t(pos.t), int32_t(pos.y), int32_t(pos.x)});
        }
        need(span.advance == std::max(w / 32, h / 32), "mtmd video position advance disagrees with the profile");
        need(mtmd_encode_chunk(ctx, chunk) == 0, "the temporal vision encoder failed");
        const float* embedding = mtmd_get_output_embd(ctx);
        need(embedding != nullptr, "the temporal vision encoder returned no embeddings");
        // mtmd owns/reuses the output: COPY before processing any later group.
        span.embeddings.assign(embedding, embedding + size_t(floats));
        bundle.tokens.insert(bundle.tokens.end(), rows, video_pad);
        bundle.spans.push_back(std::move(span));
    }
    need(visual == 1, "video has no temporal visual group");
}

bool token_is(const llama_model* model, const char* text, int32_t expected) {
    llama_token token = -1;
    return llama_tokenize(llama_model_get_vocab(model), text, int32_t(std::strlen(text)), &token, 1, false, true) == 1 &&
           token == expected;
}
}  // namespace

bool supports_video_profile(const std::string& projector, const llama_model* model, int width) {
    if (width != 2560 || !model) return false;
    char arch[64]{};
    if (llama_model_meta_val_str(model, "general.architecture", arch, sizeof(arch)) <= 0 ||
        std::strcmp(arch, "qwen4exp") != 0) return false;
    if (!token_is(model, "<|image_pad|>", image_pad) || !token_is(model, "<|video_pad|>", video_pad) ||
        !token_is(model, "<|vision_start|>", vision_start) || !token_is(model, "<|vision_end|>", vision_end)) return false;
    gguf_init_params params{true, nullptr};
    std::unique_ptr<gguf_context, decltype(&gguf_free)> gg(gguf_init_from_file(projector.c_str(), params), gguf_free);
    if (!gg) return false;
    const int64_t p = gguf_find_key(gg.get(), "clip.projector_type");
    const int64_t patch = gguf_find_key(gg.get(), "clip.vision.patch_size");
    const int64_t merge = gguf_find_key(gg.get(), "clip.vision.spatial_merge_size");
    return p >= 0 && patch >= 0 && merge >= 0 && gguf_get_kv_type(gg.get(), p) == GGUF_TYPE_STRING &&
           std::strcmp(gguf_get_val_str(gg.get(), p), "qwen3vl_merger") == 0 &&
           gguf_get_kv_type(gg.get(), patch) == GGUF_TYPE_UINT32 && gguf_get_val_u32(gg.get(), patch) == 16 &&
           gguf_get_kv_type(gg.get(), merge) == GGUF_TYPE_UINT32 && gguf_get_val_u32(gg.get(), merge) == 2 &&
           gguf_find_tensor(gg.get(), "v.patch_embd.weight.1") >= 0;
}

MediaBundle export_video(mtmd_context* ctx, const std::string& packet, uint32_t width, const VideoExportLimits& limits) {
    need(std::filesystem::is_regular_file(packet), "RGB frame spool must be a regular owned file");
    std::ifstream in(packet, std::ios::binary);
    need(bool(in), "cannot open the RGB frame spool");
    char magic[4];
    read(in, magic, 4);
    const uint64_t flags = read_le(in, 4), frames = read_le(in, 4), w = read_le(in, 4), h = read_le(in, 4);
    const double duration = read_double(in);
    const uint64_t rgb = read_le(in, 8);
    need(std::memcmp(magic, "SVF1", 4) == 0 && flags == 0, "bad RGB frame spool version/flags");
    need(width == 2560 && frames >= 1 && frames <= limits.frames && w >= 32 && h >= 32 &&
         w % 32 == 0 && h % 32 == 0 && w <= 32768 && h <= 32768, "bad RGB frame spool dimensions/count");
    need(std::isfinite(duration) && duration > 0 && duration <= limits.duration_s, "bad RGB frame spool duration");
    const uint64_t per_frame = w * h * 3, group_rows = w / 32 * (h / 32), groups = (frames + 1) / 2;
    need(group_rows <= limits.group_tokens && rgb == per_frame * frames && rgb <= limits.rgb_bytes &&
         groups * group_rows <= limits.rows && groups * group_rows * width * 4 <= limits.embedding_bytes,
         "RGB frame spool exceeds the video budget");
    need(std::filesystem::file_size(packet) == 36 + rgb + frames * 8, "RGB frame spool length does not match header");
    // Validate every timestamp BEFORE any model encode, then rewind the finite spool.
    double prior = -1;
    for (uint64_t i = 0; i < frames; ++i) {
        const double t = read_double(in);
        need(std::isfinite(t) && t >= 0 && t >= prior && t <= duration, "bad frame timestamp");
        prior = t;
        in.seekg(std::streamoff(per_frame), std::ios::cur);
        need(bool(in), "cannot seek the RGB frame spool");
    }
    in.seekg(36, std::ios::beg);
    need(bool(in), "cannot rewind the RGB frame spool");
    MediaBundle bundle;
    bundle.width = width;
    std::vector<unsigned char> a(size_t(per_frame), 0), b(size_t(per_frame), 0);
    double previous = -1;
    for (uint64_t i = 0; i < frames; i += 2) {
        const double first = read_double(in);
        need(std::isfinite(first) && first >= 0 && first >= previous && first <= duration, "bad frame timestamp");
        read(in, a.data(), a.size());
        double last = first;
        if (i + 1 < frames) {
            last = read_double(in);
            need(std::isfinite(last) && last >= first && last <= duration, "bad frame timestamp");
            read(in, b.data(), b.size());
        } else b = a; // last frame repeated, NEVER dropped, for an odd temporal group
        previous = last;
        append_group(bundle, ctx, uint32_t(w), uint32_t(h), a, b, (first + last) / 2, limits);
    }
    need(in.peek() == std::char_traits<char>::eof() && in.eof() && !in.bad(), "trailing RGB spool bytes or read error");
    program::validate_qwen4_media(bundle, video_media_limits(width, limits));
    return bundle;
}

void publish_media(const std::string& output, const MediaBundle& bundle, const VideoExportLimits& limits) {
    const std::string partial = output + ".partial";
    try {
        std::ofstream out(partial, std::ios::binary | std::ios::trunc);
        need(bool(out), "cannot create the media partial file");
        program::write_media(out, bundle, video_media_limits(bundle.width, limits));
        out.close();
        need(bool(out), "cannot finish the media partial file");
        // Owned callers create an empty destination; replace is atomic on POSIX.
        // On Windows remove that empty reservation first, never a cache entry.
#if defined(_WIN32)
        std::filesystem::remove(output);
#endif
        std::filesystem::rename(partial, output);
    } catch (...) {
        std::remove(partial.c_str());
        throw;
    }
}

void warm_video(mtmd_context* ctx, uint32_t width, const VideoExportLimits& limits) {
    // Same loaded projector/weights as ENC; no second ~907 MB projector context.
    const uint32_t rows = uint32_t(std::min({uint64_t(limits.group_tokens), limits.rows,
        limits.embedding_bytes / (uint64_t(width) * 4), limits.rgb_bytes / (32 * 32 * 3)}));
    need(rows > 0, "video budgets leave no room for a warm-up group");
    // Maximum row count AND maximum spatial extent. Reuse the same weights,
    // buffers and mtmd context as images; no hidden second projector allocation.
    const uint32_t w = rows * 32, h = 32;
    std::vector<unsigned char> first(size_t(w) * h * 3, 127), second(first.size(), 129);
    MediaBundle bundle;
    bundle.width = width;
    append_group(bundle, ctx, w, h, first, second, 0.0, limits);
    program::validate_media(bundle);
}

}  // namespace strata::vision
