// metal/tests/engine/rope_table_parity.cpp - the decode window's RoPE table (src/engine/rope_window.h) against the loop
// it replaced and against ggml's rope_yarn.  CPU only: no GPU, no model.  Exit 0 = every check passed.
//
//   rope_table_parity [MAX_CONTEXT] [--hashes]     (default 1048832, the 1M probe's allocation; what ctest runs)
//
// The window's table has MAXC + 16 rows (Window::init).  A row depends only on its position, so the table of every
// smaller context is a prefix of this one.  ctest runs the 1M allocation: with setup_host.cpp built without
// -ffp-contract=off, Apple clang 21 changes two entries of the yarn x2 table, at positions 589,449 and 910,519, past a
// 512K table.
//
//   1. UNSCALED, the default path: the table from build_rope_table is the window's earlier loop (window.cpp before
//      rope scaling reached it; its body is copied below) byte for byte, at every row.  And with --rope-freq-base 5e6,
//      which that loop never saw (it read the model file's key): the same loop given that base.
//   2. SCALED: the configurations below against a float64 transcription of ggml's rope_yarn (llama.cpp 3cf03257, the
//      commit setup.py pins: ggml/src/ggml-cpu/ops.cpp:5958-5973) written here, with every operation rounded on its
//      own and cos and sin called separately: bit-exact.  As in src/kernels/rope_parity.cpp's check 4d, the
//      correction dims and the ramp come from the shared helpers (RopeScaling::corr_dims and rope_yarn_ramp,
//      rope_scaling.hpp), so this check cannot catch an error in them; check 3's corr [14, 22] anchors the dims.
//   3. STRUCTURE (src/kernels/rope_parity.cpp's check 4e): row 0 is (mscale, 0) in every pair; under YaRN the
//      correction window is pairs [14, 22] for this model, and at the last row the first pair extrapolates and the
//      last pair interpolates (the two answers differ there, so the check can fail).
//   --hashes, information: the FNV-1a of each table as a PC's session builds it (layer.cpp:
//      build_rope_table(64, scaling, max_cells), cos [pos][32] then sin), to compare with a PC's.
#include "../../src/engine/rope_window.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <initializer_list>
#include <string>
#include <vector>

using strata::kernels::RopeScaling;
using strata::kernels::RopeScalingType;

namespace {

double (*volatile vcos)(double) = &::cos;   // called through a volatile pointer: cos and sin stay two calls
double (*volatile vsin)(double) = &::sin;

double secs(std::chrono::steady_clock::time_point t0) {
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
}

// window.cpp's loop before this change (base = the model file's qwen4exp.rope.freq_base, 1e7), one row at a time: the
// body is verbatim, `cs` points at row p's 64 floats
void legacy_window_row(double base, int p, float* cs) {
    for (int i = 0; i < 32; ++i) {
        const double th = (double) p * std::pow(base, -2.0 * i / 64.0);
        cs[i * 2] = (float) std::cos(th);
        cs[i * 2 + 1] = (float) std::sin(th);
    }
}

// ggml's rope_yarn in float64, each operation rounded on its own: theta_interp = freq_scale * theta_extrap; with
// ext_factor != 0, theta = theta_interp * (1 - ramp_mix) + theta_extrap * ramp_mix and
// mscale = attn_factor * (1 + 0.1 * log(1 / freq_scale)); cos and sin times mscale
struct Spec {
    bool scaled = false, correct = false;
    double fs = 1, ms = 1, inv[32] = {}, ramp[32] = {}, keep[32] = {};
    explicit Spec(const RopeScaling& sc) {
        scaled = sc.type != RopeScalingType::None;
        correct = scaled && sc.ext_factor != 0;
        fs = sc.freq_scale_in > 0 ? sc.freq_scale_in : 1.0 / sc.factor;
        ms = sc.attn_factor;
        if (correct) {
            const volatile double t = 0.1 * std::log(1.0 / fs);
            ms = sc.attn_factor * (1.0 + t);
        }
        if (!scaled) ms = 1.0;
        double cd[2];
        sc.corr_dims(64, cd);
        for (int i = 0; i < 32; ++i) {
            inv[i] = std::pow(sc.freq_base, -2.0 * (double) i / 64.0);
            ramp[i] = (double) strata::kernels::rope_yarn_ramp((float) cd[0], (float) cd[1], i) * sc.ext_factor;
            keep[i] = 1.0 - ramp[i];
        }
    }
    void entry(int p, int i, float& c, float& s) const {
        const volatile double extrap = (double) p * inv[i];
        double ang = extrap;
        if (scaled) {
            const volatile double interp = fs * extrap;
            ang = interp;
            if (correct) {
                const volatile double a = interp * keep[i];
                const volatile double b = extrap * ramp[i];
                ang = a + b;
            }
        }
        c = (float) (vcos(ang) * ms);
        s = (float) (vsin(ang) * ms);
    }
};

uint64_t fnv1a(const float* p, size_t n, uint64_t h = 1469598103934665603ull) {
    const unsigned char* b = (const unsigned char*) p;
    for (size_t k = 0; k < n * 4; ++k) {
        h ^= b[k];
        h *= 1099511628211ull;
    }
    return h;
}

// a configuration as generate.cpp resolves it: --rope-scaling TYPE --rope-scale FACTOR, --yarn-attn-factor ATTN and
// --rope-freq-scale FS (0: 1 / FACTOR), with the automatic --yarn-ext-factor (1 for yarn, else 0)
RopeScaling scaled(RopeScalingType type, double factor, double attn = 1.0, double freq_scale_in = 0.0) {
    RopeScaling sc;
    sc.type = type;
    sc.factor = factor;
    sc.attn_factor = attn;
    sc.freq_scale_in = freq_scale_in;
    sc.ext_factor = type == RopeScalingType::YaRN ? 1.0 : 0.0;
    return sc;
}

}  // namespace

int main(int argc, char** argv) {
    long long max_cells = 1048832;
    bool hashes = false;
    for (int a = 1; a < argc; ++a) {
        if (std::string(argv[a]) == "--hashes") hashes = true;
        else max_cells = std::atoll(argv[a]);
    }
    if (max_cells < 1 || max_cells > 33554176) {   // the earlier loop's int index (p * 32 + i) * 2 stays below 2^31
        std::fprintf(stderr, "usage: rope_table_parity [MAX_CONTEXT 1..33554176] [--hashes]\n");
        return 2;
    }
    const int MAXC = (int) std::max<long long>(4096, (max_cells + 64) / 4 * 4 + 64);   // Window::init
    const int rows = MAXC + 16;
    const size_t n = (size_t) rows * 64;
    std::vector<float> got(n);
    int bad = 0;

    // ---- 1. unscaled: the default path (freq_base 1e7, what generate.cpp resolves for the shipped models), then
    // --rope-freq-base 5e6 (under none generate.cpp keeps only the base)
    for (const double base : {1e7, 5e6}) {
        RopeScaling none;
        none.freq_base = base;
        auto t0 = std::chrono::steady_clock::now();
        strata::metal_engine::window_rope_table(none, rows, got.data());
        const double t_new = secs(t0);
        long long diff_rows = 0;
        float row[64];
        t0 = std::chrono::steady_clock::now();
        for (int p = 0; p < rows; ++p) {
            legacy_window_row(base, p, row);
            diff_rows += std::memcmp(row, &got[(size_t) p * 64], sizeof row) != 0;
        }
        const double t_old = secs(t0);
        if (base == 1e7)
            std::printf("unscaled, %d rows (--max-context %lld): build_rope_table vs the earlier loop   %s  "
                        "(the table built in %.2f s; the copied loop with its row compare %.2f s)\n", rows, max_cells,
                        diff_rows == 0 ? "byte-identical" : "*** DIFFERENT ***", t_new, t_old);
        else
            std::printf("unscaled, --rope-freq-base %g: build_rope_table vs the earlier loop given that base   %s\n",
                        base, diff_rows == 0 ? "byte-identical" : "*** DIFFERENT ***");
        if (diff_rows) std::printf("  %lld rows differ\n", diff_rows);
        bad += diff_rows != 0;
    }

    // ---- 2. + 3. scaled
    struct Case { const char* name; RopeScaling sc; };
    const Case cases[] = {{"yarn x2", scaled(RopeScalingType::YaRN, 2.0)},
                          {"yarn x3", scaled(RopeScalingType::YaRN, 3.0)},
                          {"yarn x4", scaled(RopeScalingType::YaRN, 4.0)},
                          {"linear x4", scaled(RopeScalingType::Linear, 4.0)},
                          {"yarn x4 --yarn-attn-factor 0.9", scaled(RopeScalingType::YaRN, 4.0, 0.9)},
                          {"linear x4 --yarn-attn-factor 1.25", scaled(RopeScalingType::Linear, 4.0, 1.25)},
                          {"yarn --rope-freq-scale 0.3", scaled(RopeScalingType::YaRN, 1.0, 1.0, 0.3)}};
    for (const Case& k : cases) {
        const RopeScaling& sc = k.sc;
        const Spec spec(sc);
        strata::metal_engine::window_rope_table(sc, rows, got.data());
        long long diff = 0;
        for (int p = 0; p < rows; ++p)
            for (int i = 0; i < 32; ++i) {
                float c, s;
                spec.entry(p, i, c, s);
                diff += std::memcmp(&c, &got[((size_t) p * 32 + i) * 2], 4) != 0;
                diff += std::memcmp(&s, &got[((size_t) p * 32 + i) * 2 + 1], 4) != 0;
            }
        // 3. row 0: every pair's angle is 0, so cos is mscale (the spare key's scale) and sin is 0
        const float msf = (float) spec.ms;
        bool row0 = true;
        for (int i = 0; i < 32; ++i) row0 = row0 && std::memcmp(&got[2 * i], &msf, 4) == 0 && got[2 * i + 1] == 0.0f;
        double cd[2];
        sc.corr_dims(64, cd);
        bool shape = true;
        if (sc.type == RopeScalingType::YaRN) {
            // the last row: pair 0 keeps the trained angle (extrapolates), pair 31 takes the scaled one (interpolates)
            const int far = rows - 1;
            const double ext0 = (double) far, ext31 = (double) far * spec.inv[31];
            const float c0_ext = (float) (std::cos(ext0) * spec.ms), c0_int = (float) (std::cos(spec.fs * ext0) * spec.ms);
            const float s31_ext = (float) (std::sin(ext31) * spec.ms), s31_int = (float) (std::sin(spec.fs * ext31) * spec.ms);
            const float g0 = got[(size_t) far * 64], g31 = got[(size_t) far * 64 + 63];
            const bool apart = std::fabs((double) c0_ext - (double) c0_int) > 1e-3 && std::fabs((double) s31_ext - (double) s31_int) > 1e-8;
            shape = cd[0] == 14 && cd[1] == 22 && std::memcmp(&g0, &c0_ext, 4) == 0 && std::memcmp(&g31, &s31_int, 4) == 0 && apart;
        }
        std::printf("%s: freq_scale %.17g mscale %.17g: %lld of %zu floats differ from the spec, row 0 %s",
                    k.name, sc.freq_scale(), spec.ms, diff, n, row0 ? "(mscale, 0)" : "*** WRONG ***");
        if (sc.type == RopeScalingType::YaRN)
            std::printf(", corr [%g, %g], structure %s", cd[0], cd[1], shape ? "ok" : "*** WRONG ***");
        std::printf("\n");
        bad += diff != 0 || !row0 || !shape;
    }

    // ---- --hashes: the tables as a PC's session builds them, cos then sin, at the same --max-context
    if (hashes) {
        const int max_pos = (int) max_cells;
        float* cos_tab = got.data();
        float* sin_tab = got.data() + (size_t) max_pos * 32;   // got holds rows * 64 >= 2 * max_pos * 32 floats
        std::printf("build_rope_table(64, none, %d): fnv1a ", max_pos);
        strata::kernels::build_rope_table(64, RopeScaling{}, max_pos, cos_tab, sin_tab);
        std::printf("%016llx\n", (unsigned long long) fnv1a(got.data(), (size_t) max_pos * 64));
        for (const Case& k : cases) {
            strata::kernels::build_rope_table(64, k.sc, max_pos, cos_tab, sin_tab);
            std::printf("build_rope_table(64, %s, %d): fnv1a %016llx\n", k.name, max_pos,
                        (unsigned long long) fnv1a(got.data(), (size_t) max_pos * 64));
        }
    }
    std::printf("%s\n", bad ? "FAILED" : "PASS");
    return bad ? 1 : 0;
}
