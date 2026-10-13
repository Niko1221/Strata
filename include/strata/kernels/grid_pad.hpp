// include/strata/kernels/grid_pad.hpp - empty workgroups past the RDNA4 "fill stall".
//
// On a gfx12 card a launch whose waves land at, or just below, a whole multiple of the card's wave slots (WGPs x 64:
// 2 CUs x 2 SIMD32 x 16 waves; 2048 on the 32-WGP R9700 / 9070 XT) runs ~8-14 us longer than one a few waves
// bigger (ROCm legacy-rocm-build#6689; reproduced on an R9700 with ROCm 7.14.1 and 7.2.4: 125-128 blocks of 512
// threads 16.4 us, 124 or 129 blocks 2.2 us). Measured on Strata's decode matvecs (one launch, hipGraph):
// attn_gate 2560->6144 at 2-4 columns 41-44 -> 33 us, ssm_out/attn_output 6144->2560 at 1 column 41 -> 31 us,
// the 512-row projections at 1 column 16.6 -> 5.7 us. A few extra workgroups that return at once take it away.
//
// grid_pad(blocks, threads) is the grid to launch: `blocks` plus that padding where the rule asks for it; the
// kernel returns at once in a block past its real grid, so every output is the unpadded launch's, bit for bit.
// Only gfx12 (other cards: `blocks`, unchanged). STRATA_GRID_PAD=0 turns it off, =1 applies it on any HIP card.
#pragma once

#include "strata/kernels/gfx_arch.hpp"

#include <algorithm>
#include <cstdlib>

namespace strata::kernels {

/// The rule alone (host, no device query): `waves_per_block` waves a block, `slots` wave slots on the card. A launch
/// within the last max(16, 4 x waves a block) waves below a multiple of `slots`, or within 16 x k waves past the k-th
/// multiple (at most 256), gets empty blocks up to 16 x k waves past it. Fitted to the measurements above: the
/// 12288-wave attn_gate launch needed 64 waves past (+8 blocks of 4 waves took back only half), 10240 waves 16,
/// 2048 waves 4; ROCm's report has 6144 one-wave blocks needing 32-64.
inline unsigned grid_pad_rule(unsigned blocks, unsigned waves_per_block, unsigned slots) {
    if (slots == 0 || waves_per_block == 0 || blocks == 0) return blocks;
    const unsigned long long waves = (unsigned long long) blocks * waves_per_block;
    unsigned long long k = waves / slots;
    long long past = (long long) (waves % slots);
    if (past + (long long) std::max(16u, 4 * waves_per_block) > (long long) slots) {   // just below the next multiple
        ++k;
        past -= (long long) slots;
    }
    if (k == 0) return blocks;
    const long long want = (long long) std::min<unsigned long long>(256, 16 * k);
    if (past >= want) return blocks;
    return blocks + unsigned((want - past + waves_per_block - 1) / waves_per_block);
}

#if defined(__HIPCC__)
/// The current device's wave slots and wave size, 0 slots where the padding is off (cached per device ordinal).
struct GridPadDevice { unsigned slots = 0, wave = 32; };
inline GridPadDevice grid_pad_device() {
    static const int mode = [] { const char* v = std::getenv("STRATA_GRID_PAD"); return v && *v ? std::atoi(v) : -1; }();
    if (mode == 0) return {};
    static GridPadDevice cache[16];
    static bool known[16] = {};
    int dev = 0;
    if (hipGetDevice(&dev) != hipSuccess || dev < 0 || dev >= 16) return {};
    if (!known[dev]) {
        hipDeviceProp_t p;
        GridPadDevice d;
        if (hipGetDeviceProperties(&p, dev) == hipSuccess && (mode == 1 || gfx_arch_is_gfx12(p.gcnArchName)) &&
            p.multiProcessorCount > 0 && p.warpSize > 0) {
            d.slots = unsigned(p.multiProcessorCount) * 64u;   // HIP counts WGPs on gfx12: 32 on the R9700 -> 2048
            d.wave = unsigned(p.warpSize);
        }
        cache[dev] = d;
        known[dev] = true;
    }
    return cache[dev];
}

inline unsigned grid_pad(unsigned blocks, unsigned threads_per_block) {
    const GridPadDevice d = grid_pad_device();
    return d.slots ? grid_pad_rule(blocks, (threads_per_block + d.wave - 1) / d.wave, d.slots) : blocks;
}
#else
inline unsigned grid_pad(unsigned blocks, unsigned) { return blocks; }
#endif

}  // namespace strata::kernels
