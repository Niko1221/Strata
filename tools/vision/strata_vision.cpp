// tools/vision/strata_vision.cpp - the image half of Strata's multimodal path.
//
// Turns an image into the embeddings the text model reads in place of its <|image_pad|> tokens, with llama.cpp's
// mtmd library and the model's mmproj file (vision encoder + projector).  The engine does the rest: it places the
// rows at the image's pad tokens and gives them their 2-D M-RoPE positions (see --serve GENI in generate.cpp).
//
//   strata-vision --mmproj <mmproj.gguf> --model <text model .gguf, first split> [--gpu] [--threads N]
//                 [--max-tokens N] [--min-tokens N] [--flash-attn on|off|auto]
// Flash attention defaults to auto, and to off on the CPU when ggml is built with AVX-512 (see main).
//
// Resident: prints "READY <n_embd>", then per stdin line
//   ENC <image path> <output path>        ->  "OK <n_tokens> <ms>"  or  "ERR <message>"
//   ENCV <video path> <output path> [fps] ->  "OK <n_tokens> <ms>"  or  "ERR <message>"
//   QUIT
// An image is one picture; a clip is read and encoded natively by mtmd (it samples the frames at the fps -
// the clip's own rate when unset - and merges them the model's way).  The output file is one or more records
// (a clip has one per merged frame group), each  int32 {0x31455653 'SVE1', n_tokens, nx, ny, n_embd}  then
// float32 [n_tokens][n_embd], row i at grid position (x = i % nx, y = i / nx).  <n_tokens> in the OK line is
// the total over the file's records.  The text model is opened vocab-only (no weights).
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

// "ENCV <video> <out> [fps]": a clip, read and encoded natively by mtmd (it samples the frames at `fps` and
// merges them the model's way).  The optional fps is a trailing number; without it the last token is the output
// path and mtmd uses the clip's own rate.  As with ENC, a path may contain spaces, so the tokens are taken from
// the right.
bool parse_encv(const std::string& line, std::string& video, std::string& out, float& fps) {
    if (line.rfind("ENCV ", 0) != 0) return false;
    const std::string rest = line.substr(5);
    const size_t sp = rest.rfind(' ');
    if (sp == std::string::npos || sp == 0) return false;
    const std::string last = rest.substr(sp + 1);
    const std::string mid = rest.substr(0, sp);
    char* end = nullptr;
    const float f = std::strtof(last.c_str(), &end);
    if (!last.empty() && end != last.c_str() && *end == '\0') {          // a trailing number: the fps
        fps = f;
        const size_t sp2 = mid.rfind(' ');
        if (sp2 == std::string::npos || sp2 == 0) return false;
        video = mid.substr(0, sp2);
        out = mid.substr(sp2 + 1);
    } else {                                                              // no fps: the last token is the output
        fps = 0.0f;
        video = mid;
        out = last;
    }
    return !video.empty() && !out.empty();
}

}  // namespace

int main(int argc, char** argv) {
    std::string mmproj, model;
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
                             "[--max-tokens N] [--min-tokens N] [--flash-attn on|off|auto]\n");
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
#if !defined(_WIN32) && !defined(__APPLE__)   // macOS has no posix_fadvise
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
        std::string media, out;
        float fps = 0.0f;
        const bool clip = parse_encv(line, media, out, fps);
        if (!clip && !parse_enc(line, media, out)) {
            std::printf("ERR expected: ENC <image> <output> or ENCV <video> <output> [fps]\n");
            std::fflush(stdout);
            continue;
        }
        const char* kind = clip ? "video" : "image";
        const auto t0 = std::chrono::steady_clock::now();
        mtmd_helper_init_opt opt = mtmd_helper_init_opt_default();
        if (clip) {
            if (fps > 0.0f) opt.video_params.fps_target = fps;   // 0 = the clip's own rate (mtmd's default)
            opt.video_params.timestamp_interval_ms = 0;          // no per-frame timestamp text: the engine takes pictures only
        }
        mtmd_helper_bitmap_wrapper bw = mtmd_helper_bitmap_init_from_file(ctx, media.c_str(), false, opt);
        if (!bw.bitmap) { std::printf("ERR cannot read the %s %s\n", kind, media.c_str()); std::fflush(stdout); continue; }
        mtmd_input_chunks* chunks = mtmd_input_chunks_init();
        const std::string prompt = mtmd_default_marker();
        mtmd_input_text txt{prompt.c_str(), prompt.size(), false, true};
        const mtmd_bitmap* bm[1] = {bw.bitmap};
        std::string err;
        int total_n = 0;
        if (mtmd_tokenize(ctx, chunks, &txt, bm, 1) != 0) err = "the " + std::string(kind) + " could not be preprocessed";
        FILE* f = nullptr;
        if (err.empty()) {
            f = std::fopen(out.c_str(), "wb");
            if (!f) err = "cannot write " + out;
        }
        // one record per image chunk: a picture is one, a clip one per merged frame group, in order - the engine
        // reads them back in that order at the clip's pad tokens
        for (size_t c = 0; err.empty() && f && c < mtmd_input_chunks_size(chunks); ++c) {
            const mtmd_input_chunk* ch = mtmd_input_chunks_get(chunks, c);
            if (mtmd_input_chunk_get_type(ch) != MTMD_INPUT_CHUNK_TYPE_IMAGE) continue;   // skip a clip's text
            if (mtmd_encode_chunk(ctx, ch) != 0) { err = "the vision encoder failed"; break; }
            const mtmd_image_tokens* it = mtmd_input_chunk_get_tokens_image(ch);
            const int n = (int) mtmd_input_chunk_get_n_tokens(ch);
            // the grid from the decoder positions (nx/ny getters are deprecated): x and y of the last token
            const mtmd_decoder_pos last = mtmd_image_tokens_get_decoder_pos(it, 0, (size_t) n - 1);
            const int nx = (int) last.x + 1, ny = (int) last.y + 1;
            if (nx * ny != n) { err = "a " + std::string(kind) + " grid is not rectangular (" + std::to_string(n) + " tokens)"; break; }
            const float* embd = mtmd_get_output_embd(ctx);
            const int32_t hdr[5] = {0x31455653, n, nx, ny, n_embd};
            const bool ok = std::fwrite(hdr, sizeof hdr, 1, f) == 1 &&
                            std::fwrite(embd, sizeof(float) * (size_t) n_embd, (size_t) n, f) == (size_t) n;
            if (!ok) { err = "short write to " + out; break; }
            total_n += n;
        }
        if (f) std::fclose(f);
        if (err.empty() && total_n < 1) err = "no image chunks";
        if (err.empty()) {
            const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
            std::printf("OK %d %.0f\n", total_n, ms);
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
