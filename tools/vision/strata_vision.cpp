// tools/vision/strata_vision.cpp - the image half of Strata's multimodal path.
//
// Turns an image into the embeddings the text model reads in place of its <|image_pad|> tokens, with llama.cpp's
// mtmd library and the model's mmproj file (vision encoder + projector).  The engine does the rest: it places the
// rows at the image's pad tokens and gives them their 2-D M-RoPE positions (see --serve GENI in generate.cpp).
//
//   strata-vision --mmproj <mmproj.gguf> --model <text model .gguf, first split> [--gpu] [--threads N]
//                 [--max-tokens N] [--min-tokens N] [--flash-attn on|off|auto] [--ffmpeg-dir DIR]
// Flash attention defaults to auto, and to off on the CPU when ggml is built with AVX-512 (see main).
//
// Resident: prints "READY <n_embd>", then per stdin line
//   ENC <image path> <output path>   ->  "OK <n_tokens> <nx> <ny> <ms>"  or  "ERR <message>"
//   ENCV <fps> <max_frames> <max_side> <pair_tokens> <total_tokens> <min_pair_tokens> <video path> <output path>
//                                    ->  "OK <n_tokens> <n_groups> <n_frames> <ms>" then "LAYOUT <layout>"
//   QUIT
// The output file is  int32 {0x31455653 'SVE1', n_tokens, nx, ny, n_embd}  then float32 [n_tokens][n_embd],
// row i at grid position (x = i % nx, y = i / nx).  The text model is opened vocab-only (no weights).
//
// A video (ENCV, needs ffmpeg/ffprobe) is laid out as the Qwen3-VL processor does it (transformers
// processing_qwen3_vl.py): frames sampled at <fps> by ffmpeg, two consecutive frames merged into one temporal patch
// (pairs 0+1, 2+3, ...; an odd last frame doubled), and each pair preceded by its time as text, "<1.2 seconds>" (the
// pair's mean), then <|vision_start|> image <|vision_end|>.  The output file holds one SVE1 record per frame pair;
// LAYOUT is the sequence the prompt takes in place of the video, ';'-separated: "T<id>,<id>,..." text tokens or "I<n>"
// one record's n image cells.  <max_frames> lowers the rate for a long video (0 = no cap); <pair_tokens> sizes each
// frame so a pair is about that many tokens (0 = as decoded, up to --max-tokens), <max_side> caps the longer side
// (0 = no cap).  Small text needs a large <pair_tokens>: a token covers 32x32 pixels of the sized frame.
// <total_tokens> is the whole video's budget (0 = none): a long video first gets smaller frames, down to
// <min_pair_tokens> per pair, then a lower rate.  Encoding runs 32 pairs at a time, so memory does not grow with length.
#include "ggml-cpu.h"
#include "gguf.h"
#include "llama.h"
#include "mtmd.h"
#include "mtmd-helper.h"

#include <algorithm>
#if !defined(_WIN32)
#include <fcntl.h>
#include <unistd.h>
#endif
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

namespace {

void quiet_log(ggml_log_level level, const char* text, void*) {
    if (level >= GGML_LOG_LEVEL_WARN) std::fputs(text, stderr);
}

// "ENC <image> <out>": paths may contain spaces, so the image path ends at the last " " before the output path;
// the server always writes paths without spaces, but a user calling the tool by hand might not
bool parse_enc(const std::string& line, std::string& img, std::string& out) {
    if (line.rfind("ENC ", 0) != 0) return false;
    const std::string rest = line.substr(4);
    const size_t sp = rest.rfind(' ');
    if (sp == std::string::npos || sp == 0) return false;
    img = rest.substr(0, sp);
    out = rest.substr(sp + 1);
    return !img.empty() && !out.empty();
}

// "ENCV <fps> <max_frames> <max_side> <pair_tokens> <total_tokens> <min_pair_tokens> <video> <out>": the numbers
// first, then the paths as for ENC
struct EncvArgs {
    float fps = 0;
    int max_frames = 0, max_side = 0, pair_tokens = 0, total_tokens = 0, min_pair_tokens = 0;
    std::string video, out;
};
bool parse_encv(const std::string& line, EncvArgs& a) {
    if (line.rfind("ENCV ", 0) != 0) return false;
    std::istringstream is(line.substr(5));
    if (!(is >> a.fps >> a.max_frames >> a.max_side >> a.pair_tokens >> a.total_tokens >> a.min_pair_tokens))
        return false;
    std::string rest;
    std::getline(is, rest);
    rest.erase(0, rest.find_first_not_of(' '));
    return parse_enc("ENC " + rest, a.video, a.out);
}

// Area-average downscale of an RGB frame to the size a frame pair of about pair_tokens tokens takes (one token per
// 32x32 pixels: 16-pixel patches, 2x2 merged), its longer side at most max_side; nullptr when it already fits.  The
// encoder's cost follows the pixel count, and mtmd would otherwise feed a 1080p frame up to its token cap.  A new
// mergeable bitmap.
mtmd_bitmap* fit_frame(const mtmd_bitmap* src, int max_side, int pair_tokens) {
    const int sx = (int) mtmd_bitmap_get_nx(src), sy = (int) mtmd_bitmap_get_ny(src);
    double scale = 1.0;
    if (max_side > 0) scale = std::min(scale, (double) max_side / std::max(sx, sy));
    if (pair_tokens > 0) scale = std::min(scale, std::sqrt((double) pair_tokens * 32.0 * 32.0 / ((double) sx * sy)));
    if (scale >= 1.0) return nullptr;
    const int dx = std::max(32, (int) (sx * scale)), dy = std::max(32, (int) (sy * scale));
    const unsigned char* in = mtmd_bitmap_get_data(src);
    std::vector<unsigned char> outp((size_t) dx * dy * 3);
    for (int y = 0; y < dy; ++y) {
        const int y0 = (int) ((int64_t) y * sy / dy), y1 = std::max(y0 + 1, (int) ((int64_t) (y + 1) * sy / dy));
        for (int x = 0; x < dx; ++x) {
            const int x0 = (int) ((int64_t) x * sx / dx), x1 = std::max(x0 + 1, (int) ((int64_t) (x + 1) * sx / dx));
            uint32_t acc[3] = {0, 0, 0};
            for (int yy = y0; yy < y1; ++yy)
                for (int xx = x0; xx < x1; ++xx)
                    for (int c = 0; c < 3; ++c) acc[c] += in[((size_t) yy * sx + xx) * 3 + c];
            const uint32_t cnt = (uint32_t) ((y1 - y0) * (x1 - x0));
            for (int c = 0; c < 3; ++c) outp[((size_t) y * dx + x) * 3 + c] = (unsigned char) ((acc[c] + cnt / 2) / cnt);
        }
    }
    mtmd_bitmap* bm = mtmd_bitmap_init((uint32_t) dx, (uint32_t) dy, outp.data());
    if (bm) mtmd_bitmap_set_mergeable(bm, true);
    return bm;
}

// The video's frames from mtmd's ffmpeg reader (its text items - a label, timestamps - are skipped: ENCV writes the
// Qwen3-VL timestamps itself), each sized by fit_frame and mergeable, at most max_frames; false on a read error.
bool read_frames(mtmd_helper_video* v, int max_frames, int max_side, int pair_tokens, std::vector<mtmd_bitmap*>& out) {
    while (max_frames <= 0 || (int) out.size() < max_frames) {
        mtmd_bitmap* bm = nullptr;
        char* text = nullptr;
        const int32_t r = mtmd_helper_video_read_next(v, &bm, &text);
        std::free(text);
        if (r == -1) break;
        if (r != 0) return false;
        if (!bm) continue;
        if (mtmd_bitmap* small = fit_frame(bm, max_side, pair_tokens)) {
            mtmd_bitmap_free(bm);
            bm = small;
        }
        mtmd_bitmap_set_mergeable(bm, true);
        out.push_back(bm);
    }
    return true;
}

// One encoded image chunk as an SVE1 record; "" or the error.
std::string write_record(mtmd_context* ctx, const mtmd_input_chunk* ch, int n_embd, std::FILE* f, int& n, int& nx, int& ny) {
    if (mtmd_encode_chunk(ctx, ch) != 0) return "the vision encoder failed";
    const mtmd_image_tokens* it = mtmd_input_chunk_get_tokens_image(ch);
    n = (int) mtmd_input_chunk_get_n_tokens(ch);
    // the grid from the decoder positions (nx/ny getters are deprecated): x and y of the last token
    const mtmd_decoder_pos last = mtmd_image_tokens_get_decoder_pos(it, 0, (size_t) n - 1);
    nx = (int) last.x + 1;
    ny = (int) last.y + 1;
    if (nx * ny != n) return "the image grid is not rectangular (" + std::to_string(n) + " tokens)";
    const int32_t hdr[5] = {0x31455653, n, nx, ny, n_embd};
    const float* embd = mtmd_get_output_embd(ctx);
    if (std::fwrite(hdr, sizeof hdr, 1, f) != 1 ||
        std::fwrite(embd, sizeof(float) * (size_t) n_embd, (size_t) n, f) != (size_t) n)
        return "short write";
    return "";
}

}  // namespace

int main(int argc, char** argv) {
    std::string mmproj, model, ffmpeg_dir;
    bool gpu = false;
    int threads = 0, max_tokens = 0, min_tokens = 0;
    llama_flash_attn_type fa = LLAMA_FLASH_ATTN_TYPE_AUTO;
    bool fa_given = false;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&]() -> std::string {
            if (i + 1 >= argc) { std::fprintf(stderr, "%s needs a value\n", a.c_str()); std::exit(2); }
            return argv[++i];
        };
        if (a == "--mmproj") mmproj = next();
        else if (a == "--model") model = next();
        else if (a == "--gpu") gpu = true;
        else if (a == "--threads") threads = std::atoi(next().c_str());
        else if (a == "--max-tokens") max_tokens = std::atoi(next().c_str());
        else if (a == "--min-tokens") min_tokens = std::atoi(next().c_str());   // mtmd image_min_tokens (#767)
        else if (a == "--ffmpeg-dir") ffmpeg_dir = next();   // ffmpeg and ffprobe for ENCV; default: PATH
        else if (a == "--flash-attn") {   // FA keeps K and V in FP16; off = the attention in FP32
            const std::string v = next();
            fa = v == "on" ? LLAMA_FLASH_ATTN_TYPE_ENABLED : v == "off" ? LLAMA_FLASH_ATTN_TYPE_DISABLED
                                                                        : LLAMA_FLASH_ATTN_TYPE_AUTO;
            fa_given = true;
        }
        else { std::fprintf(stderr, "unknown argument %s\n", a.c_str()); return 2; }
    }
    if (mmproj.empty() || model.empty()) {
        std::fprintf(stderr, "usage: strata-vision --mmproj <mmproj.gguf> --model <model.gguf> [--gpu] [--threads N] "
                             "[--max-tokens N] [--min-tokens N] [--flash-attn on|off|auto] [--ffmpeg-dir DIR]\n");
        return 2;
    }
    // On the CPU the GPU stays unseen: a CUDA build otherwise opens a context there (measured: 0.4-0.7 GB of VRAM,
    // 150-260 expert slots less for the engine beside it).  Before anything reaches the CUDA runtime.
    if (!gpu) {
#ifdef _WIN32
        _putenv_s("CUDA_VISIBLE_DEVICES", "-1");
#else
        setenv("CUDA_VISIBLE_DEVICES", "-1", 1);
#endif
    }
    llama_log_set(quiet_log, nullptr);
    mtmd_helper_log_set(quiet_log, nullptr);
    llama_backend_init();

    const auto load_t0 = std::chrono::steady_clock::now();
#if !defined(_WIN32)
    // ask the OS for the projector file up front (128 KiB steps) so the loader's sequential reads find it cached;
    // STRATA_READ_AHEAD=0 turns it off, as in the engine
    if (const char* ra = std::getenv("STRATA_READ_AHEAD"); ra == nullptr || std::atoi(ra) != 0) {
        const int fd = open(mmproj.c_str(), O_RDONLY);
        if (fd >= 0) {
            const off_t n = lseek(fd, 0, SEEK_END);
            for (off_t at = 0; at < n; at += (off_t) (128 << 10))
                (void) posix_fadvise(fd, at, std::min<off_t>((off_t) (128 << 10), n - at), POSIX_FADV_WILLNEED);
            close(fd);
        }
    }
#endif
    llama_model_params mp = llama_model_default_params();
    mp.vocab_only = true;
    llama_model* text = llama_model_load_from_file(model.c_str(), mp);
    if (!text) { std::printf("ERR cannot open the text model %s\n", model.c_str()); std::fflush(stdout); return 1; }

    mtmd_context_params cp = mtmd_context_params_default();
    cp.use_gpu = gpu;
    cp.print_timings = false;
    cp.warmup = false;
    // On the CPU "auto" turns flash attention on.  ggml's fast (tiled) CPU kernel for it needs the head size, 72 in this
    // encoder, to be a multiple of the vector width: 8 floats with AVX2 (the release builds), 16 with AVX-512 (a build
    // from source on a CPU that has it), where ggml falls back to a kernel that is several times slower and accumulates
    // in FP16.  A 1024x1024 picture on a Ryzen 7 7700X, 8 threads: AVX2 8-10 s with it, 14-15 s without; AVX-512 44 s
    // with it (26% off the FP32 attention's output), 13 s without.
    if (!gpu && !fa_given && ggml_cpu_has_avx512()) fa = LLAMA_FLASH_ATTN_TYPE_DISABLED;
    cp.flash_attn_type = fa;
    // on the CPU without --threads: one per core (mtmd's own default is 4 threads)
    if (threads <= 0 && !gpu) threads = std::max(1u, std::thread::hardware_concurrency() / 2);
    if (threads > 0) cp.n_threads = threads;
    if (max_tokens > 0) cp.image_max_tokens = max_tokens;
    if (min_tokens > 0) cp.image_min_tokens = min_tokens;
    mtmd_context* ctx = mtmd_init_from_file(mmproj.c_str(), text, cp);
    if (!ctx || !mtmd_support_vision(ctx)) {
        std::printf("ERR cannot load the vision encoder %s\n", mmproj.c_str());
        std::fflush(stdout);
        return 1;
    }
    std::fprintf(stderr, "strata-vision: model files loaded in %.1f s\n",
                 std::chrono::duration<double>(std::chrono::steady_clock::now() - load_t0).count());
    // a vocab-only model reports no hparams, so the width comes from the projector: its output is the model's input
    int n_embd = 0;
    {
        gguf_init_params gp{true, nullptr};
        gguf_context* gg = gguf_init_from_file(mmproj.c_str(), gp);
        const int64_t k = gg ? gguf_find_key(gg, "clip.vision.projection_dim") : -1;
        if (k >= 0) n_embd = (int) gguf_get_val_u32(gg, k);
        if (gg) gguf_free(gg);
    }
    if (n_embd <= 0) { std::printf("ERR the vision encoder has no projection_dim\n"); std::fflush(stdout); return 1; }
    // Warm up at the LARGEST picture before READY: the encoder's GPU work buffers are allocated now, not at the first
    // real picture.  The server starts this process before the engine, so the engine sizes its expert cache from
    // what is really left; allocating ~1 GB later, on a GPU the engine has filled, made Windows page GPU memory and
    // the engine crawl to a standstill.  (A square image well above any cap; mtmd scales it to the token limit.)
    // On the CPU there is no VRAM to reserve, and the warm-up would only delay the engine's start by one encode
    // (~6 s at 1,024 tokens).
    if (!gpu) std::fprintf(stderr, "strata-vision: on the CPU, %d threads, no warm-up\n", threads);
    else {
        const auto warm_t0 = std::chrono::steady_clock::now();
        const uint32_t side = 2048;
        std::vector<unsigned char> rgb((size_t) side * side * 3, 128);
        mtmd_bitmap* bm = mtmd_bitmap_init(side, side, rgb.data());
        mtmd_input_chunks* chunks = mtmd_input_chunks_init();
        const std::string marker = mtmd_default_marker();
        mtmd_input_text txt{marker.c_str(), marker.size(), false, true};
        const mtmd_bitmap* bms[1] = {bm};
        int warm_tokens = 0;
        if (bm && mtmd_tokenize(ctx, chunks, &txt, bms, 1) == 0) {
            for (size_t c = 0; c < mtmd_input_chunks_size(chunks); ++c) {
                const mtmd_input_chunk* ch = mtmd_input_chunks_get(chunks, c);
                if (mtmd_input_chunk_get_type(ch) == MTMD_INPUT_CHUNK_TYPE_IMAGE && mtmd_encode_chunk(ctx, ch) == 0)
                    warm_tokens = (int) mtmd_input_chunk_get_n_tokens(ch);
            }
        }
        std::fprintf(stderr, "strata-vision: warmed up at %d image tokens in %.1f s\n", warm_tokens,
                     std::chrono::duration<double>(std::chrono::steady_clock::now() - warm_t0).count());
        mtmd_input_chunks_free(chunks);
        if (bm) mtmd_bitmap_free(bm);
    }
    std::printf("READY %d\n", n_embd);
    std::fflush(stdout);

    std::string line;
    while (std::getline(std::cin, line)) {
        while (!line.empty() && (line.back() == '\r' || line.back() == '\n')) line.pop_back();
        if (line == "QUIT") break;
        EncvArgs va;
        if (line.rfind("ENCV ", 0) == 0) {
            if (!parse_encv(line, va)) {
                std::printf("ERR expected: ENCV <fps> <max_frames> <max_side> <pair_tokens> <total_tokens> "
                            "<min_pair_tokens> <video> <output>\n");
                std::fflush(stdout);
                continue;
            }
            const auto t0 = std::chrono::steady_clock::now();
            mtmd_helper_video_init_params vp = mtmd_helper_video_init_params_default();
            if (va.fps > 0) vp.fps_target = va.fps;
            vp.timestamp_interval_ms = 0;                     // the timestamps are written below, one per pair
            vp.ffmpeg_bin_dir = ffmpeg_dir.empty() ? nullptr : ffmpeg_dir.c_str();
            mtmd_helper_video* video = mtmd_helper_video_init(ctx, va.video.c_str(), vp);
            if (!video) {
                std::printf("ERR cannot read the video %s (is ffmpeg/ffprobe installed?)\n", va.video.c_str());
                std::fflush(stdout);
                continue;
            }
            // The plan, from the probed length (n_frames at the asked rate): every frame while the pairs fit the
            // whole-video budget at <pair_tokens>; past that smaller frames, down to <min_pair_tokens> per pair;
            // past that fewer frames - a lower rate spreads them over the whole video (a cap alone would keep only
            // its start).  The hard cap in read_frames stays for a length ffprobe estimated short.
            const mtmd_helper_video_info vi0 = mtmd_helper_video_get_info(video);
            int want = vi0.n_frames > 0 ? vi0.n_frames : (va.max_frames > 0 ? va.max_frames : 64);
            int frames_plan = va.max_frames > 0 ? std::min(want, va.max_frames) : want;
            int pair_tok = va.pair_tokens;
            if (va.total_tokens > 0) {
                // 3% under the budget: frames are sized to a 32-pixel grid, so a pair can come out a little over
                const int budget = (int) ((int64_t) va.total_tokens * 97 / 100);
                const int pairs = std::max(1, (frames_plan + 1) / 2);
                const int fit = budget / pairs;
                if (pair_tok <= 0 || fit < pair_tok) pair_tok = fit;
                if (pair_tok < va.min_pair_tokens) {
                    pair_tok = va.min_pair_tokens;
                    frames_plan = std::max(2, 2 * (budget / std::max(1, va.min_pair_tokens)));
                }
            }
            if (vi0.n_frames > frames_plan && vi0.fps > 0) {
                vp.fps_target = vi0.fps * (float) frames_plan / (float) vi0.n_frames;
                mtmd_helper_video_free(video);
                video = mtmd_helper_video_init(ctx, va.video.c_str(), vp);
            }
            const float fps = video ? mtmd_helper_video_get_info(video).fps : 0.0f;
            std::string err, layout;
            if (!video || fps <= 0) err = "the video could not be read";
            std::FILE* f = nullptr;
            if (err.empty() && !(f = std::fopen(va.out.c_str(), "wb"))) err = "cannot write " + va.out;
            // In batches of 32 pairs: read, tokenize, encode, write, free - an hour-scale video never holds more
            // than one batch of preprocessed frames.  Per pair "<t seconds>" (the pair's mean time) and two markers
            // with nothing between them, so mtmd merges exactly 0+1, 2+3, ... (and wraps each merged pair in
            // <|vision_start|> ... <|vision_end|> itself); an odd last frame is doubled, as the reference pads it.
            constexpr int kBatchFrames = 64;
            const std::string marker = mtmd_default_marker();
            int total = 0, groups = 0, n_frames = 0;
            bool done = false;
            while (err.empty() && !done) {
                std::vector<mtmd_bitmap*> frames;
                const int room = frames_plan - n_frames;
                if (room <= 0 || !read_frames(video, std::min(kBatchFrames, room), va.max_side, pair_tok, frames)) {
                    if (room > 0) err = "the video could not be read";
                    break;
                }
                done = (int) frames.size() < std::min(kBatchFrames, room) || (int) frames.size() == room;
                if (frames.empty()) break;
                const int base = n_frames, got = (int) frames.size();
                n_frames += got;
                if (got % 2 == 1) {
                    const mtmd_bitmap* last = frames.back();
                    mtmd_bitmap* dup = mtmd_bitmap_init(mtmd_bitmap_get_nx(last), mtmd_bitmap_get_ny(last),
                                                        mtmd_bitmap_get_data(last));
                    mtmd_bitmap_set_mergeable(dup, true);
                    frames.push_back(dup);
                    done = true;
                }
                std::string prompt;
                for (int i = 0; i + 1 < (int) frames.size(); i += 2) {
                    const int a = base + i, b = base + std::min(i + 1, got - 1);
                    char ts[48];
                    // ffmpeg's fps filter (round=near) returns, for output frame k, the source frame at about (k + 0.5) / fps
                    std::snprintf(ts, sizeof ts, "<%.1f seconds>", (a + b + 1) / 2.0 / fps);
                    prompt += ts + marker + marker;
                }
                mtmd_input_chunks* chunks = mtmd_input_chunks_init();
                mtmd_input_text txt{prompt.c_str(), prompt.size(), false, true};
                if (mtmd_tokenize(ctx, chunks, &txt, frames.data(), frames.size()) != 0)
                    err = "the video could not be preprocessed";
                for (size_t c = 0; err.empty() && c < mtmd_input_chunks_size(chunks); ++c) {
                    const mtmd_input_chunk* ch = mtmd_input_chunks_get(chunks, c);
                    if (!layout.empty()) layout += ';';
                    if (mtmd_input_chunk_get_type(ch) == MTMD_INPUT_CHUNK_TYPE_TEXT) {
                        size_t nt = 0;
                        const llama_token* tk = mtmd_input_chunk_get_tokens_text(ch, &nt);
                        layout += 'T';
                        for (size_t j = 0; j < nt; ++j) layout += (j ? "," : "") + std::to_string(tk[j]);
                    } else if (mtmd_input_chunk_get_type(ch) == MTMD_INPUT_CHUNK_TYPE_IMAGE) {
                        int n = 0, nx = 0, ny = 0;
                        err = write_record(ctx, ch, n_embd, f, n, nx, ny);
                        layout += 'I' + std::to_string(n);
                        total += n;
                        ++groups;
                    } else {
                        err = "the video gave a chunk that is neither text nor image";
                    }
                }
                mtmd_input_chunks_free(chunks);
                for (mtmd_bitmap* bm : frames) mtmd_bitmap_free(bm);
            }
            if (f) std::fclose(f);
            if (err.empty() && groups == 0) err = "the video has no frames";
            const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
            if (err.empty()) std::printf("OK %d %d %d %.0f\nLAYOUT %s\n", total, groups, n_frames, ms, layout.c_str());
            else std::printf("ERR %s\n", err.c_str());
            std::fflush(stdout);
            if (video) mtmd_helper_video_free(video);
            continue;
        }
        std::string img, out;
        if (!parse_enc(line, img, out)) { std::printf("ERR expected: ENC <image> <output>\n"); std::fflush(stdout); continue; }
        const auto t0 = std::chrono::steady_clock::now();
        mtmd_helper_bitmap_wrapper bw = mtmd_helper_bitmap_init_from_file(ctx, img.c_str(), false,
                                                                            mtmd_helper_init_opt_default());
        if (!bw.bitmap) { std::printf("ERR cannot read the image %s\n", img.c_str()); std::fflush(stdout); continue; }
        mtmd_input_chunks* chunks = mtmd_input_chunks_init();
        const std::string prompt = mtmd_default_marker();
        mtmd_input_text txt{prompt.c_str(), prompt.size(), false, true};
        const mtmd_bitmap* bm[1] = {bw.bitmap};
        std::string err;
        const mtmd_input_chunk* ichunk = nullptr;
        if (mtmd_tokenize(ctx, chunks, &txt, bm, 1) != 0) err = "the image could not be preprocessed";
        for (size_t c = 0; err.empty() && c < mtmd_input_chunks_size(chunks); ++c) {
            const mtmd_input_chunk* ch = mtmd_input_chunks_get(chunks, c);
            if (mtmd_input_chunk_get_type(ch) == MTMD_INPUT_CHUNK_TYPE_IMAGE) ichunk = ch;
        }
        if (err.empty() && !ichunk) err = "no image chunk";
        if (err.empty() && mtmd_encode_chunk(ctx, ichunk) != 0) err = "the vision encoder failed";
        if (err.empty()) {
            const mtmd_image_tokens* it = mtmd_input_chunk_get_tokens_image(ichunk);
            const int n = (int) mtmd_input_chunk_get_n_tokens(ichunk);
            // the grid from the decoder positions (nx/ny getters are deprecated): x and y of the last token
            const mtmd_decoder_pos last = mtmd_image_tokens_get_decoder_pos(it, 0, (size_t) n - 1);
            const int nx = (int) last.x + 1, ny = (int) last.y + 1;
            if (nx * ny != n) err = "the image grid is not rectangular (" + std::to_string(n) + " tokens)";
            const float* embd = mtmd_get_output_embd(ctx);
            FILE* f = err.empty() ? std::fopen(out.c_str(), "wb") : nullptr;
            if (err.empty() && !f) err = "cannot write " + out;
            if (f) {
                const int32_t hdr[5] = {0x31455653, n, nx, ny, n_embd};
                const bool ok = std::fwrite(hdr, sizeof hdr, 1, f) == 1 &&
                                std::fwrite(embd, sizeof(float) * (size_t) n_embd, (size_t) n, f) == (size_t) n;
                std::fclose(f);
                if (!ok) err = "short write to " + out;
                const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
                if (ok) std::printf("OK %d %d %d %.0f\n", n, nx, ny, ms);
            }
        }
        if (!err.empty()) std::printf("ERR %s\n", err.c_str());
        std::fflush(stdout);
        mtmd_input_chunks_free(chunks);
        mtmd_bitmap_free(bw.bitmap);
        if (bw.video_ctx) mtmd_helper_video_free(bw.video_ctx);
    }
    mtmd_free(ctx);
    llama_model_free(text);
    llama_backend_free();
    return 0;
}
