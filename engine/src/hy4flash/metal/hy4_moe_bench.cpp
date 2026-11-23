// hy4_moe_bench - kernel-level throughput of the Hy4-Flash prefill MoE GEMMs (hy4_mppmoe.metal) with the routing
// fixed by an explicit tiles[] array, so tile shape and quant format are the only variables.
// usage: hy4_moe_bench <kernel-dir>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "mtl_rt.h"

struct ExpertDev {  // = ExpertDev of hy4.metal (16 B)
    uint64_t body_off;
    uint32_t dn_off;
    uint8_t gu_code, dn_code;
    uint16_t pad;
};
struct Int2 { int x, y; };

static double median(std::vector<double> v) {
    std::sort(v.begin(), v.end());
    return v[v.size() / 2];
}
static int sb_bytes(int code) { return code == 2 ? 70 : code == 3 ? 102 : code == 4 ? 134 : code == 5 ? 166 : code == 6 ? 198 : 262; }

int main(int argc, char** argv) {
    const std::string dir = argc > 1 ? argv[1] : ".";
    mtl::Context cx;
    const std::vector<std::string> names = {"hy4.metal", "hy4_split.metal", "hy4_hcp.metal", "hy4_flash.metal",
                                            "hy4_kvb.metal", "hy4_dfly2.metal", "hy4_lin2.metal", "hy4_moe2.metal",
                                            "hy4_lin3.metal", "hy4_dfly2b.metal", "hy4_mpp.metal", "hy4_mppq.metal",
                                            "hy4_mppmoe.metal", "hy4_mppflash.metal"};
    std::vector<std::string> files;
    for (auto& n : names) files.push_back(dir + "/" + n);
    std::string why;
    const int lib = cx.compile_try(files, "#define NO_GGML_COMMON 1\n", false, &why);
    if (lib < 0) {
        printf("compile failed: %s\n", why.c_str());
        return 1;
    }

    const int NEXP = 512, K = 1536, N = 1536;
    const int ROWS_SMALL = 256, ROWS_BIG = 131072;  // an 8K chunk gathers 8192*16 = 131072 token-expert rows
    const size_t slot = 4u << 20;                    // room for gu + dn of one expert
    mtl::Buffer sec4 = cx.alloc(slot * NEXP);
    std::memset(sec4.ptr, 0x5A, slot * NEXP);
    for (int e = 0; e < NEXP; ++e) {  // sqx (per-row scale) = 1.0 for every row
        __fp16* sq = (__fp16*) ((char*) sec4.ptr + (size_t) e * slot);
        for (int i = 0; i < 1536; ++i) sq[i] = (__fp16) 1.0;
    }
    mtl::Buffer ex = cx.alloc((size_t) NEXP * sizeof(ExpertDev));
    mtl::Buffer xs_small = cx.alloc((size_t) ROWS_SMALL * K * 2), hs_small = cx.alloc((size_t) ROWS_SMALL * K * 2);
    mtl::Buffer xs_big = cx.alloc((size_t) ROWS_BIG * K * 2), hs_big = cx.alloc((size_t) ROWS_BIG * K * 2);
    mtl::Buffer ye = cx.alloc((size_t) ROWS_BIG * K * 4);
    for (auto* b : {&xs_small, &xs_big}) {
        __fp16* x = (__fp16*) b->ptr;
        const size_t n = b->bytes / 2;
        for (size_t i = 0; i < n; ++i) x[i] = (__fp16) 0.1;
    }
    mtl::Buffer tiles = cx.alloc((size_t) 4 * NEXP * sizeof(Int2));

    struct Cfg { const char* name; int TM, TN, SG, mul, tiles, code, dn, roff; };
    // mul: rows one tile covers = mul*TM (gu2/dn2 do two M tiles, gu/dn one)
    const Cfg cfgs[] = {
        {"moe_mm_gu2_64_64_4", 64, 64, 4, 2, 2, 3, 0, 0},
        {"moe_mm_gu2_64_64_4", 64, 64, 4, 2, 2, 4, 0, 0},
        {"moe_mm_gu2_64_64_4", 64, 64, 4, 2, 2, 5, 0, 0},
        {"moe_mm_gu2_64_64_4", 64, 64, 4, 2, 2, 6, 0, 0},
        {"moe_mm_dn2_64_64_4", 64, 64, 4, 2, 2, 4, 1, 0},
        {"moe_mm_dn2_64_64_4", 64, 64, 4, 2, 2, 5, 1, 0},
        {"moe_mm_dn2_64_64_4", 64, 64, 4, 2, 2, 6, 1, 0},
    };
    printf("512 experts x 256 token-expert rows per layer; kernel-only times (no plan/gather/combine)\n");
    printf("%-20s %4s %3s %3s %5s %6s | %8s %9s %9s\n", "kernel", "TM", "TN", "code", "buf", "", "ms", "TFLOP/s", "wGB/s");
    for (int big = 0; big < 2; ++big)
        for (const Cfg& c : cfgs) {
            const int sb = sb_bytes(c.code);
            const uint32_t dn_off = (uint32_t) (3072 + (size_t) 3072 * 6 * sb);
            for (int e = 0; e < NEXP; ++e) {
                ExpertDev* d = (ExpertDev*) ((char*) ex.ptr + (size_t) e * sizeof(ExpertDev));
                d->body_off = (uint64_t) e * slot;
                d->dn_off = dn_off;
                d->gu_code = (uint8_t) c.code;
                d->dn_code = (uint8_t) c.code;
                d->pad = 0;
            }
            mtl::Pipeline p = cx.pipeline(lib, c.name);
            const int ntiles = NEXP * c.tiles;
            Int2* t = (Int2*) tiles.ptr;
            for (int e = 0; e < NEXP; ++e)
                for (int j = 0; j < c.tiles; ++j)
                    t[e * c.tiles + j] = Int2{e, c.roff + e * (c.tiles * c.mul * c.TM) + j * (c.mul * c.TM)};
            mtl::Buffer& in = big ? xs_big : xs_small;
            mtl::Buffer& out = c.dn ? ye : (big ? hs_big : hs_small);
            mtl::Dispatch d;
            d.p = &p;
            d.args = {mtl::Arg::B(sec4), mtl::Arg::B(ex), mtl::Arg::B(in), mtl::Arg::B(out), mtl::Arg::B(tiles)};
            d.grid[0] = (uint32_t) (N / c.TN);
            d.grid[1] = (uint32_t) ntiles;
            d.grid[2] = 1;
            d.tg[0] = (uint32_t) (c.SG * 32);
            d.tg[1] = d.tg[2] = 1;
            cx.run({d});  // warm
            std::vector<double> ts;
            for (int r = 0; r < 7; ++r) ts.push_back(cx.run({d}, 2) / 2.0);
            const double s = median(ts);
            const double rows = (double) ntiles * c.mul * c.TM;
            const int nm = c.dn ? 1 : 2;
            const double flop = 2.0 * rows * N * K * nm;
            const double wbytes = (double) ntiles * (double) (N / c.TN) * (double) (nm * c.TN) * K * (double) sb / 256.0;
            printf("%-20s %4d %3d %5d %6s %6s | %8.3f %9.2f %9.1f\n", c.name, c.TM, c.TN, c.code, big ? "8K" : "fit", "",
                   s * 1e3, flop / s / 1e12, wbytes / s / 1e9);
        }
    return 0;
}
