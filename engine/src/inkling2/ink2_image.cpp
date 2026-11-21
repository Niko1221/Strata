// src/inkling2/ink2_image.cpp - see ink2_image.hpp.
#include "ink2_image.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#define STB_IMAGE_IMPLEMENTATION
#define STBI_ONLY_JPEG
#define STBI_ONLY_PNG
#define STBI_ONLY_BMP
#define STBI_ONLY_GIF
#define STBI_NO_LINEAR
#define STBI_NO_HDR
#include "../../third_party/stb/stb_image.h"

namespace ink2img {
namespace {

int tokens_for(int w, int h) { return ((h + TILE - 1) / TILE) * ((w + TILE - 1) / TILE); }

// area average (each output pixel the mean of the source area it covers, partial pixels weighted)
// CBA: the engine's own downscale for pictures over the token cap (the reference processor never resizes, so there is
// nothing to match); a PIL-style antialiased bicubic would only matter if the cap is hit often
void area_resize(const std::vector<uint8_t>& src, int sw, int sh, std::vector<uint8_t>& dst, int dw, int dh) {
    dst.assign((size_t) dw * dh * 3, 0);
    const double fx = (double) sw / dw, fy = (double) sh / dh;
    for (int oy = 0; oy < dh; ++oy) {
        const double y0 = oy * fy, y1 = (oy + 1) * fy;
        for (int ox = 0; ox < dw; ++ox) {
            const double x0 = ox * fx, x1 = (ox + 1) * fx;
            double acc[3] = {0, 0, 0}, area = 0;
            for (int iy = (int) y0; iy < std::min(sh, (int) std::ceil(y1)); ++iy) {
                const double wy = std::min(y1, iy + 1.0) - std::max(y0, (double) iy);
                for (int ix = (int) x0; ix < std::min(sw, (int) std::ceil(x1)); ++ix) {
                    const double wgt = wy * (std::min(x1, ix + 1.0) - std::max(x0, (double) ix));
                    const uint8_t* s = &src[((size_t) iy * sw + ix) * 3];
                    for (int c = 0; c < 3; ++c) acc[c] += wgt * s[c];
                    area += wgt;
                }
            }
            for (int c = 0; c < 3; ++c)
                dst[((size_t) oy * dw + ox) * 3 + c] = (uint8_t) std::min(255.0, std::max(0.0, std::round(acc[c] / area)));
        }
    }
}

// patches from a tile accessor px(tile, y, x, c) (y, x within the 40 x 40 tile)
template <class F>
void cut(Picture& p, F px) {
    const int nt = p.rows * p.cols;
    p.patches.assign((size_t) nt * 4 * PIN, 0.0f);
    for (int t = 0; t < nt; ++t)
        for (int ir = 0; ir < 2; ++ir)
            for (int ic = 0; ic < 2; ++ic) {
                float* o = &p.patches[((size_t) t * 4 + ir * 2 + ic) * PIN];
                for (int y = 0; y < PATCH; ++y)
                    for (int x = 0; x < PATCH; ++x)
                        for (int c = 0; c < 3; ++c) *o++ = px(t, ir * PATCH + y, ic * PATCH + x, c);
            }
}

}  // namespace

std::string load_picture(const std::string& path, int max_tokens, Picture& p) {
    int w = 0, h = 0, comp = 0;
    unsigned char* px = stbi_load(path.c_str(), &w, &h, &comp, 4);
    if (!px) return "cannot decode " + path + ": " + stbi_failure_reason();
    if (w < 1 || h < 1) { stbi_image_free(px); return "empty picture " + path; }
    // the processor's convert_to_rgb: alpha over a checkerboard of 8 px squares, 204 where (row/8 + col/8) is even, else
    // 153, integer rounding as (rgb * a + bg * (255 - a) + 127) / 255 (an opaque pixel stays as it is)
    std::vector<uint8_t> rgb((size_t) w * h * 3);
    for (int y = 0; y < h; ++y)
        for (int x = 0; x < w; ++x) {
            const unsigned char* s = px + ((size_t) y * w + x) * 4;
            const int a = s[3], bg = ((y / 8 + x / 8) % 2 == 0) ? 204 : 153;
            for (int c = 0; c < 3; ++c) rgb[((size_t) y * w + x) * 3 + c] = (uint8_t) ((s[c] * a + bg * (255 - a) + 127) / 255);
        }
    stbi_image_free(px);
    p.src_w = w;
    p.src_h = h;
    if (max_tokens > 0 && tokens_for(w, h) > max_tokens) {
        double s = std::sqrt((double) max_tokens * TILE * TILE / ((double) w * h));
        int nw, nh;
        for (;; s *= 0.995) {
            nw = std::max(1, (int) (w * s));
            nh = std::max(1, (int) (h * s));
            if (tokens_for(nw, nh) <= max_tokens) break;
        }
        std::vector<uint8_t> small;
        area_resize(rgb, w, h, small, nw, nh);
        rgb.swap(small);
        w = nw;
        h = nh;
    }
    p.w = w;
    p.h = h;
    p.rows = (h + TILE - 1) / TILE;
    p.cols = (w + TILE - 1) / TILE;
    uint64_t k = 1469598103934665603ull;
    auto mix = [&k](const void* d, size_t n) {
        for (size_t i = 0; i < n; ++i) { k ^= ((const uint8_t*) d)[i]; k *= 1099511628211ull; }
    };
    mix(&w, sizeof w);
    mix(&h, sizeof h);
    mix(rgb.data(), rgb.size());
    p.key = k;
    const float pad = (-1.0f - 127.5f) / 127.5f;   // the processor pads with -1 on the 0-255 scale, then normalizes
    cut(p, [&](int t, int y, int x, int c) {
        const int gy = (t / p.cols) * TILE + y, gx = (t % p.cols) * TILE + x;
        if (gy >= h || gx >= w) return pad;
        return ((float) rgb[((size_t) gy * w + gx) * 3 + c] - 127.5f) / 127.5f;
    });
    return "";
}

std::string load_pixels(const std::string& path, Picture& p) {
    FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) return "cannot open " + path;
    int32_t rc[2] = {0, 0};
    std::vector<float> t;
    std::string err;
    if (std::fread(rc, 4, 2, f) != 2 || rc[0] < 1 || rc[1] < 1) err = "bad pixel file " + path;
    else {
        t.resize((size_t) rc[0] * rc[1] * TILE * TILE * 3);
        if (std::fread(t.data(), 4, t.size(), f) != t.size()) err = "short pixel file " + path;
    }
    std::fclose(f);
    if (!err.empty()) return err;
    p.rows = rc[0];
    p.cols = rc[1];
    p.w = p.src_w = p.cols * TILE;
    p.h = p.src_h = p.rows * TILE;
    p.key = 0;
    cut(p, [&](int tile, int y, int x, int c) { return t[(((size_t) tile * TILE + y) * TILE + x) * 3 + c]; });
    return "";
}

std::string pct_decode(const std::string& s) {
    std::string o;
    for (size_t i = 0; i < s.size(); ++i) {
        if (s[i] == '%' && i + 2 < s.size()) {
            const std::string hx = s.substr(i + 1, 2);
            char* e = nullptr;
            const long v = std::strtol(hx.c_str(), &e, 16);
            if (e && *e == '\0') { o += (char) v; i += 2; continue; }
        }
        o += s[i];
    }
    return o;
}

}  // namespace ink2img
