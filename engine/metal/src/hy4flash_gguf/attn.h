// engine/metal/src/hy4flash_gguf/attn.h - how the host encodes the MLA attention (msl/hy4_flash_attn.metal): three dispatches, the score
// row, the softmax and PV, each with its own grid.  One helper for forward.cpp (trunk blocks) and mtp.cpp (the MTP
// layer) and the kernel tests in tmp/tests/hy4_flash/attn, so the grids and the shape checks cannot drift between them.
//
// The three dispatches keep their memory barriers (Dispatch::barrier defaults to true): qk writes the score row
// that the softmax reads and rewrites, and PV reads the softmax's weights.  Never relax_after() over them.
#pragma once

#include "mtl_rt.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

namespace strata::hy4_flash {

// The tiling the engine runs: hy4_flash_mla_qk_t<qk_hb, qk_tg, qk_u> and hy4_flash_mla_pv_t<pv_hb, pv_vec, pv_tg, pv_u>, the
// instantiations at the end of hy4_flash_attn.metal.  Their host names carry the tile (qk_name / pv_name below), so this
// struct and the compiled library cannot drift: a tile that was not instantiated fails at pipeline lookup.  A tiling
// moves no bit, it only groups independent chains (hy4_flash_attn.metal's note); tmp/tests/hy4_flash/attn sweeps the
// alternatives bitwise.  U is how many loads a thread issues ahead of the arithmetic; it does not enter the grid.
struct AttnTile {
    uint32_t qk_hb, qk_tg, qk_u;           // heads a thread carries / keys a group covers / half4 loads ahead (qk)
    uint32_t pv_hb, pv_vec, pv_tg, pv_u;   // heads x consecutive dims a thread carries / threads / keys ahead (pv)
    std::string qk_name() const {
        return "hy4_flash_mla_qk_h" + std::to_string(qk_hb) + "_t" + std::to_string(qk_tg) + "_u" + std::to_string(qk_u);
    }
    std::string pv_name() const {
        return "hy4_flash_mla_pv_h" + std::to_string(pv_hb) + "_v" + std::to_string(pv_vec) + "_t" + std::to_string(pv_tg) +
               "_u" + std::to_string(pv_u);
    }
};
inline constexpr AttnTile kAttnTile{4, 128, 2, 2, 2, 64, 16};
inline constexpr uint32_t kAttnDqkMax = 576;   // HY4_FLASH_ATTN_DQK_MAX
inline constexpr uint32_t kAttnSoftmaxTg = 256;   // HY4_FLASH_TG: the softmax's reduction tree is a function of it

struct AttnPipes {
    mtl::Pipeline qk, sm, pv;
};

// `sel`: function constant 43 (hy4_flash_attn_sel) - the indexer's selection instead of the contiguous range.
// (The tests pass other instantiations' names; the engine always uses kAttnTile's.)
inline AttnPipes attn_pipes(mtl::Context* cx, int lib, bool sel, const std::string& qk = kAttnTile.qk_name(),
                            const std::string& pv = kAttnTile.pv_name()) {
    AttnPipes p;
    p.qk = cx->pipeline(lib, qk, mtl::Constants().b(43, sel));
    p.sm = cx->pipeline(lib, "hy4_flash_mla_softmax", mtl::Constants().b(43, sel));
    p.pv = cx->pipeline(lib, pv, mtl::Constants().b(43, sel));
    return p;
}

// Appends the three dispatches.  `args` are hy4_flash_attn.metal's nine (q, kv, sinks, sel, n_sel, pos, sc, o, the
// Hy4FlashAttnArgs bytes), bound identically to each kernel.  `n_cap` is an upper bound on every row's key count in this
// dispatch - the window for a sliding block, the scratch row (kSelCap) otherwise: the score pass's grid covers
// n_cap keys and a row with more would silently lose keys, so the caller must pass a true bound.  `sc_stride` is the
// score scratch row the same rows are written to; n_cap may not exceed it, and that too is refused here.
inline void append_attn(std::vector<mtl::Dispatch>& ds, const AttnPipes& p, const std::vector<mtl::Arg>& args,
                        uint32_t n_head, uint32_t n_tok, uint32_t n_cap, uint32_t sc_stride, uint32_t d_qk,
                        uint32_t d_v, uint32_t kv_stride, const char* where, const AttnTile& t = kAttnTile) {
    // the kernels' shape requirements: qk stages d_qk floats a head and reads K as half4, PV reads V as pv_vec halves
    const bool ok = args.size() == 9 && d_qk <= kAttnDqkMax && kv_stride % 4 == 0 && args[1].off % 8 == 0 &&
                    d_v % t.pv_vec == 0 && n_head > 0 && n_tok > 0 && n_cap <= sc_stride;
    if (!ok) {
        std::fprintf(stderr, "%s: hy4_flash attention shape not supported (args %zu, d_qk %u > %u, kv_stride %u %% 4, "
                             "kv offset %zu %% 8, d_v %u %% %u, n_cap %u > sc_stride %u)\n",
                     where, args.size(), d_qk, kAttnDqkMax, kv_stride, args.size() > 1 ? args[1].off : 0, d_v,
                     t.pv_vec, n_cap, sc_stride);
        std::abort();
    }
    mtl::Dispatch d;
    d.args = args;
    d.p = &p.qk;
    d.grid[0] = (n_cap + t.qk_tg - 1) / t.qk_tg;
    d.grid[1] = (n_head + t.qk_hb - 1) / t.qk_hb;
    d.grid[2] = n_tok;
    d.tg[0] = t.qk_tg;
    if (d.grid[0] == 0) d.grid[0] = 1;
    mtl::require_fit(d, where);
    ds.push_back(d);

    d.p = &p.sm;
    d.grid[0] = n_head;
    d.grid[1] = n_tok;
    d.grid[2] = 1;
    d.tg[0] = kAttnSoftmaxTg;
    mtl::require_fit(d, where);
    ds.push_back(d);

    d.p = &p.pv;
    const uint32_t dims_per_tg = t.pv_tg * t.pv_vec;
    d.grid[0] = (d_v + dims_per_tg - 1) / dims_per_tg;
    d.grid[1] = (n_head + t.pv_hb - 1) / t.pv_hb;
    d.grid[2] = n_tok;
    d.tg[0] = t.pv_tg;
    mtl::require_fit(d, where);
    ds.push_back(d);
}

}  // namespace strata::hy4_flash
