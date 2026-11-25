// engine/metal/src/hy4flash_gguf/vis_bundle.cpp - see vis_bundle.h.
#include "vis_bundle.h"

#include <cstdio>
#include <cstring>

namespace strata::hy4_flash {
namespace {

template <class T>
bool take(std::FILE* f, T& v) {
    return std::fread(&v, sizeof(T), 1, f) == 1;
}

std::string at(int i, const std::string& why) {
    return "picture " + std::to_string(i) + ": " + why;
}

}  // namespace

std::string read_bundle(std::vector<BundlePicture>& out, const std::string& path, const int32_t* ids,
                        int64_t n_ids, const VisionGeometry& g) {
    out.clear();
    const int32_t cell = g.patch * g.merge;
    if (cell <= 0) return "the vision shard's patch and merge sizes are not usable";

    std::FILE* f = std::fopen(path.c_str(), "rb");
    if (f == nullptr) return "cannot open the picture bundle " + path;
    struct Closer {
        std::FILE* f;
        ~Closer() { std::fclose(f); }
    } closer{f};

    char magic[4] = {};
    int32_t count = 0;
    if (std::fread(magic, 1, 4, f) != 4 || std::memcmp(magic, "DSI1", 4) != 0 || !take(f, count))
        return "the picture bundle does not start with a DSI1 header";
    if (count < 1 || (int64_t) count > n_ids)
        return "the picture bundle declares " + std::to_string(count) + " pictures, which cannot fit a " +
               std::to_string(n_ids) + "-token prompt";

    int64_t end = 0;   // one past the previous picture's run, so the runs must not overlap or go backwards
    for (int i = 0; i < count; ++i) {
        BundlePicture p;
        int64_t start = 0;
        if (!take(f, start) || !take(f, p.w) || !take(f, p.h)) return at(i, "the header is truncated");
        if (p.w < cell || p.h < cell)
            return at(i, std::to_string(p.w) + "x" + std::to_string(p.h) + " has a side below " +
                             std::to_string(cell));
        if (p.w % cell != 0 || p.h % cell != 0)
            return at(i, std::to_string(p.w) + "x" + std::to_string(p.h) +
                             " is not a multiple of patch x merge (" + std::to_string(cell) + ")");
        // `>` and not `>=`: every bound in the stack accepts an exact max_pixels (terminal/serve/hy4_flash_pictures.py,
        // src/hy4flash_gguf/vision.cu), and 2048 x 2048 is exactly 4194304 - the size the authority's own clamp
        // produces.  A `>=` here would refuse precisely that.
        if ((int64_t) p.w * p.h > g.max_pixels)
            return at(i, std::to_string(p.w) + "x" + std::to_string(p.h) + " is " +
                             std::to_string((int64_t) p.w * p.h) + " pixels, over the shard's limit of " +
                             std::to_string(g.max_pixels));
        p.start = start;
        p.count = image_token_count(g, p.h, p.w);     // from the header's own dimensions, never recomputed
        if (start < end)
            return at(i, "its run starts at " + std::to_string(start) + ", inside the previous picture's");
        if (start < 0 || start >= n_ids || start + p.count > n_ids)
            return at(i, "its run [" + std::to_string(start) + ", " + std::to_string(start + p.count) +
                             ") does not fit the " + std::to_string(n_ids) + "-token prompt");
        for (int64_t k = start; k < start + p.count; ++k)
            if (ids[k] != g.image_token)
                return at(i, "the prompt holds " + std::to_string(ids[k]) + " at " + std::to_string(k) +
                                 ", where its " + std::to_string(p.count) + "-token run needs " +
                                 std::to_string(g.image_token));
        end = start + p.count;
        p.rgb.resize((size_t) p.w * (size_t) p.h * 3);
        if (std::fread(p.rgb.data(), 1, p.rgb.size(), f) != p.rgb.size())
            return at(i, "its pixels are truncated (" + std::to_string(p.rgb.size()) + " bytes needed)");
        out.push_back(std::move(p));
    }
    if (std::fgetc(f) != EOF) return "the picture bundle has data after its last picture";
    return {};
}

}  // namespace strata::hy4_flash
