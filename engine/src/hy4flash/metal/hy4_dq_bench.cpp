// hy4_dq_bench - cost of the MPP prefill dequant (dequant_h_<fmt>) alone and paired with mpp_mm_64_64_4, on the
// engine's real dense-GEMM shapes. The engine runs dequant_h then mpp_mm for every lin() at M >= mpp_min; this tells us
// whether a few-ms GEMM is actually paying most of its time in the dequant.
// usage: hy4_dq_bench <kernel dir> [fmt]
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#include "mtl_rt.h"

struct MmArgs { int M, N, K; };

static int sb_bytes(int f) { return f == 2 ? 70 : f == 3 ? 102 : f == 4 ? 134 : f == 5 ? 166 : f == 6 ? 198 : f == 8 ? 262 : 0; }

int main(int argc, char** argv) {
    mtl::Context cx;
    const std::string dir = argc > 1 ? argv[1] : ".";
    const int fmt = argc > 2 ? atoi(argv[2]) : 6;
    std::string why;
    const int lib = cx.compile_try({dir + "/hy4.metal", dir + "/hy4_mpp.metal", dir + "/hy4_mppq.metal"}, "#define NO_GGML_COMMON 1\n", false, &why);
    if (lib < 0) { printf("compile failed: %s\n", why.c_str()); return 1; }
    const int SB = sb_bytes(fmt);
    struct Shape { int M, N, K; const char* what; };
    const Shape shapes[] = {{8192, 3072, 8192, "o_proj"}, {8192, 8192, 3072, "linear_gate"}, {8192, 1024, 8192, "q_b"}, {512, 16384, 1024, "q_b (512 tok)"}};
    std::mt19937 rng(7);
    for (const Shape& s : shapes) {
        const size_t nw = (size_t) s.N * s.K;
        mtl::Buffer W = cx.alloc(nw / 256 * SB + 64);            // quantized weights (random bytes)
        mtl::Buffer WS = cx.alloc(nw * 2);                        // fp16 scratch
        mtl::Buffer A = cx.alloc((size_t) s.M * s.K * 2);
        mtl::Buffer C = cx.alloc((size_t) s.M * s.N * 4);
        std::memset(W.ptr, 0x5a, nw / 256 * SB);                  // any bit pattern; only the byte count matters
        std::memset(A.ptr, 0x11, (size_t) s.M * s.K * 2);
        mtl::Pipeline pdq = cx.pipeline(lib, "dequant_h_" + std::to_string(fmt));
        mtl::Pipeline pmm = cx.pipeline(lib, "mpp_mm_64_64_4");
        mtl::Dispatch dq;
        dq.p = &pdq;
        dq.args = {mtl::Arg::B(W), mtl::Arg::B(WS), mtl::Arg::V<int32_t>(s.N), mtl::Arg::V<int32_t>(s.K)};
        dq.grid[0] = (uint32_t) (s.K / 256); dq.grid[1] = (uint32_t) s.N; dq.tg[0] = 32;
        MmArgs ma{s.M, s.N, s.K};
        mtl::Dispatch mm;
        mm.p = &pmm;
        mm.args = {mtl::Arg::B(A), mtl::Arg::B(WS), mtl::Arg::B(C), mtl::Arg::V<MmArgs>(ma)};
        mm.grid[0] = (uint32_t) ((s.N + 63) / 64); mm.grid[1] = (uint32_t) ((s.M + 63) / 64); mm.tg[0] = 128;
        auto med = [&](const std::vector<mtl::Dispatch>& g) {
            std::vector<double> ts;
            for (int r = 0; r < 7; ++r) ts.push_back(cx.run(g, 3) / 3);
            std::sort(ts.begin(), ts.end());
            return ts[ts.size() / 2];
        };
        const double t_dq = med({dq});
        const double t_mm = med({mm});
        const double t_both = med({dq, mm});
        printf("%-18s M %5d N %5d K %5d | dequant %.3f ms | mm %.3f ms | both %.3f ms (overhead %.3f ms)\n",
               s.what, s.M, s.N, s.K, t_dq * 1e3, t_mm * 1e3, t_both * 1e3, (t_both - t_dq - t_mm) * 1e3);
        cx.free(W); cx.free(WS); cx.free(A); cx.free(C);
    }
    return 0;
}
