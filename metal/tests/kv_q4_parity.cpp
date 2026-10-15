// metal/tests/kv_q4_parity.cpp - bit-level parity of the kv_q4.metal kernels (M0-p) against the CPU reference of
// kv_q4_ref.h, which replicates the DEVICE arithmetic exactly (the same fwht register butterfly, the same
// group-maximum tie rule, the same int(fma(x, id, 8.5f)) contraction and the same software round-to-nearest-even
// f32 -> f16 bits).
//
// Cases: fwht256 (whole-buffer equality, plus the H * H = I sanity at 1e-5), the single-step append with tie
// blocks in the rotated domain (x = {v, -v, ...} across and within lanes, one +v among -v, all equal), an
// all-zero group (d = -0.0 -> scale bits 0x8000, id = 0 -> q = trunc(8.5) = 8) and a permuted page table, a
// 3-step append (grid.z, K and V pools apart) over the amplitude classes 1e-3 / 40 / 1 of
// src/kernels/kv_steps_parity.cpp:80, the prompt batch kernel with the host and stage pools, the composite raw
// -> device fwht -> device quantize chain (the input is NOT pre-rotated), and the gather into fp16 scratch.
// Fixed RNG (ref::Rng); this Mac is its own reference - nothing compares against another machine's bytes.
#include "mtl_rt.h"
#include "ref_quant.h"

#include "../../include/strata/kernels/f16_bits.hpp"
#include "kv_q4_ref.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

namespace k = strata::kernels;
using namespace kvq4ref;

int g_fail = 0;

void cmp_blocks(const char* what, const mtl::Buffer& got, const std::vector<uint8_t>& ref) {
    long bad = 0;
    size_t first = (size_t) -1;
    const size_t n = std::min(got.bytes, ref.size());
    for (size_t off = 0; off + 18 <= n; off += 18)
        if (std::memcmp((const uint8_t*) got.ptr + off, ref.data() + off, 18) != 0) {
            if (first == (size_t) -1) first = off;
            ++bad;
        }
    if (got.bytes != ref.size()) ++bad;
    std::printf("  %-38s %6ld 18-byte blocks differ of %zu  %s\n", what, bad, n / 18, bad ? "FAIL" : "PASS");
    if (first != (size_t) -1) std::printf("    first differing block %zu (byte offset %zu)\n", first / 18, first);
    g_fail += bad != 0;
}

int main() {
    mtl::Context cx;
    const int lib = cx.compile({"kv_q4.metal"}, "#define NO_GGML_COMMON 1\n");
    const mtl::Pipeline pfw = cx.pipeline(lib, "fwht256"), papp = cx.pipeline(lib, "kv_append_q4_kernel"),
                         pbat = cx.pipeline(lib, "kv_append_q4_batch_kernel"),
                         pgat = cx.pipeline(lib, "kv_gather_q4_kernel");
    static const mtl::Buffer kNone;   // a nil buffer: the kernels' "no host / stage pool"
    ref::Rng r(23);
    std::printf("device %s\n", cx.device_name().c_str());

    // ---- 1. fwht256: whole-buffer equality with the device-order reference
    {
        const int rows = 3 * H + 5;
        mtl::Buffer src = cx.alloc(4 * (size_t) rows * 256), dst = cx.alloc(4 * (size_t) rows * 256);
        for (size_t i = 0; i < src.bytes / 4; ++i) src.as<float>()[i] = r.sym() * 3.0f;
        mtl::Dispatch d;
        d.p = &pfw;
        d.args = {mtl::Arg::B(src), mtl::Arg::B(dst), mtl::Arg::V((uint64_t) rows), mtl::Arg::V(1.0f / 16.0f)};
        d.grid[0] = (uint32_t) ((rows + 3) / 4);
        d.tg[0] = 32;
        d.tg[1] = 4;
        cx.run({d});
        std::vector<float> ref(rows * 256);
        fwht256_ref(src.as<float>(), ref.data(), rows);
        long bad = 0;
        for (int64_t i = 0; i < (int64_t) rows * 256; ++i)
            if (std::memcmp(&dst.as<float>()[i], &ref[i], 4) != 0) ++bad;
        std::printf("  %-38s %ld of %d floats differ (bits)     %s\n", "fwht256 vs device-order ref", bad, rows * 256,
                    bad ? "FAIL" : "PASS");
        g_fail += bad != 0;
        // H is its own inverse: a 1e-5 sanity only (the float butterflies do not cancel bitwise)
        mtl::Dispatch d2 = d;
        d2.args = {mtl::Arg::B(dst), mtl::Arg::B(dst), mtl::Arg::V((uint64_t) rows), mtl::Arg::V(1.0f / 16.0f)};
        cx.run({d2});
        double worst = 0;
        for (int64_t i = 0; i < (int64_t) rows * 256; ++i)
            worst = std::max(worst, (double) std::fabs(dst.as<float>()[i] - src.as<float>()[i]));
        std::printf("  %-38s max |H*H*x - x| = %.2e          %s\n", "fwht256 self-inverse (sanity)", worst,
                    worst < 1e-5 ? "PASS" : "FAIL");
        g_fail += worst >= 1e-5;
    }

    // ---- 2. the single-step append: tie blocks and an all-zero group in the ROTATED domain, a permuted page
    // table, the host copy
    {
        const int pages = 8, cells = pages * P;
        std::vector<int32_t> table(pages);
        for (int i = 0; i < pages; ++i) table[i] = (i * 3 + 5) % pages;   // a permutation
        mtl::Buffer table_b = cx.alloc(4 * pages), step_b = cx.alloc(16), kcur = cx.alloc(4 * H * D),
                    vcur = cx.alloc(4 * H * D), kpool = cx.alloc((size_t) cells * H * ROW_BYTES),
                    vpool = cx.alloc((size_t) cells * H * ROW_BYTES), hpool_k = cx.alloc((size_t) cells * H * ROW_BYTES),
                    hpool_v = cx.alloc((size_t) cells * H * ROW_BYTES);
        std::memcpy(table_b.as<int32_t>(), table.data(), 4 * pages);
        std::vector<float> kk(H * D), vv(H * D);
        for (size_t i = 0; i < kk.size(); ++i) {
            kk[i] = r.sym() * 2.0f;
            vv[i] = r.sym();
        }
        fwht256_ref(kk.data(), kk.data(), H);   // the contract's input is already rotated
        fwht256_ref(vv.data(), vv.data(), H);
        // the tie fixtures, written into the rotated rows the quantizer actually sees
        for (int i = 0; i < 32; ++i) kk[i] = (i & 1) ? -1.5f : 1.5f;        // +/-v alternating: cross-lane ties
        for (int i = 32; i < 64; ++i) kk[i] = (i == 49) ? 0.75f : -0.75f;   // one +v among -v
        for (int i = 64; i < 96; ++i) kk[i] = 0.0f;                         // an all-zero group (head 0, block 2)
        for (int i = 0; i < 32; ++i) vv[i] = -2.25f;                        // all equal (a negative winner)
        for (int i = 32; i < 64; ++i) vv[i] = 2.25f;                        // all equal, positive
        std::memcpy(kcur.as<float>(), kk.data(), kk.size() * 4);
        std::memcpy(vcur.as<float>(), vv.data(), vv.size() * 4);
        const int32_t pos = 5 * P + 17;
        const int32_t step[4] = {pos, pos + 1, 0, 0};
        std::memcpy(step_b.as<int32_t>(), step, 16);
        mtl::Dispatch d;
        d.p = &papp;
        d.args = {mtl::Arg::B(kpool),       mtl::Arg::B(vpool),     mtl::Arg::B(table_b),  mtl::Arg::B(step_b),
                  mtl::Arg::B(kcur),        mtl::Arg::B(vcur),      mtl::Arg::B(hpool_k),  mtl::Arg::B(hpool_v),
                  mtl::Arg::V((int32_t) 4), mtl::Arg::V((int32_t) 2), mtl::Arg::V((int32_t) H), mtl::Arg::V((int32_t) D),
                  mtl::Arg::V((int32_t) P)};
        d.grid[0] = H;
        d.grid[1] = BPH;
        d.grid[2] = 2;   // planes: K and V pools apart
        d.tg[0] = 32;
        cx.run({d});
        std::vector<uint8_t> rk((size_t) cells * H * ROW_BYTES, 0), rv = rk, hk = rk, hv = rk;
        append_cell_ref(kk.data(), vv.data(), table, rk, rv, false, pos);
        append_cell_ref(kk.data(), vv.data(), table, hk, hv, true, pos);
        cmp_blocks("kv_append_q4_kernel (ties, table)", kpool, rk);
        cmp_blocks("kv_append_q4_kernel V (ties, table)", vpool, rv);
        cmp_blocks("kv_append_q4_kernel host copy K", hpool_k, hk);
        cmp_blocks("kv_append_q4_kernel host copy V", hpool_v, hv);
        // the all-zero group's documented bits, off the device pool: d = 0 / -8 = -0.0 -> scale bits 0x8000,
        // id = 0 -> q = trunc(8.5) = 8 -> every qs byte 0x88
        const int64_t zrow = ((int64_t) table[pos / P] * H + 0) * P + pos % P;
        const block_q4_0* zb = (const block_q4_0*) (kpool.as<const uint8_t>() + zrow * ROW_BYTES) + 2;
        int zbad = zb->d != 0x8000;
        for (int t = 0; t < 16; ++t) zbad += zb->qs[t] != 0x88;
        std::printf("  %-38s d = 0x%04x, qs bytes 0x88           %s\n", "kv_append_q4 all-zero group", zb->d,
                    zbad ? "FAIL" : "PASS");
        g_fail += zbad != 0;
    }

    // ---- 3. a 3-step append (grid.z) over the amplitude classes 1e-3 / 40 / 1, then the gather over that pool;
    // a PERMUTED table, so the gather's page-table lookup is exercised too (the engine passes the identity on a
    // Mac)
    {
        const int pages = 4, cells = pages * P, n_steps = 3;
        std::vector<int32_t> table(pages);
        for (int i = 0; i < pages; ++i) table[i] = (i * 3 + 1) % pages;   // a permutation
        mtl::Buffer table_b = cx.alloc(4 * pages), step_b = cx.alloc(4 * 4 * n_steps),
                    kcur = cx.alloc(4 * (size_t) n_steps * H * D), vcur = cx.alloc(4 * (size_t) n_steps * H * D),
                    kpool = cx.alloc((size_t) cells * H * ROW_BYTES), vpool = cx.alloc((size_t) cells * H * ROW_BYTES);
        std::memcpy(table_b.as<int32_t>(), table.data(), 4 * pages);
        std::vector<float> kk((size_t) n_steps * H * D), vv((size_t) n_steps * H * D);
        const std::vector<int32_t> poss{1 * P + 3, 2 * P + 33, 0 * P + 61};
        for (int s = 0; s < n_steps; ++s) {
            const float scale = s == 0 ? 1e-3f : (s == 1 ? 40.0f : 1.0f);   // kv_steps_parity.cpp:80's classes
            for (size_t i = 0; i < (size_t) H * D; ++i) {
                kk[s * H * D + i] = r.sym() * 2.0f * scale;
                vv[s * H * D + i] = r.sym() * scale;
            }
            fwht256_ref(kk.data() + s * H * D, kk.data() + s * H * D, H);
            fwht256_ref(vv.data() + s * H * D, vv.data() + s * H * D, H);
            const int32_t step[4] = {poss[s], poss[s] + 1, 0, 0};
            std::memcpy(step_b.as<int32_t>() + s * 4, step, 16);
        }
        std::memcpy(kcur.as<float>(), kk.data(), kk.size() * 4);
        std::memcpy(vcur.as<float>(), vv.data(), vv.size() * 4);
        mtl::Dispatch d;
        d.p = &papp;
        d.args = {mtl::Arg::B(kpool),       mtl::Arg::B(vpool),     mtl::Arg::B(table_b),  mtl::Arg::B(step_b),
                  mtl::Arg::B(kcur),        mtl::Arg::B(vcur),      mtl::Arg::B(kNone),    mtl::Arg::B(kNone),
                  mtl::Arg::V((int32_t) 4), mtl::Arg::V((int32_t) 2), mtl::Arg::V((int32_t) H), mtl::Arg::V((int32_t) D),
                  mtl::Arg::V((int32_t) P)};
        d.grid[0] = H;
        d.grid[1] = BPH;
        d.grid[2] = n_steps * 2;
        d.tg[0] = 32;
        cx.run({d});
        std::vector<uint8_t> rk((size_t) cells * H * ROW_BYTES, 0), rv = rk;
        for (int s = 0; s < n_steps; ++s)
            append_cell_ref(kk.data() + (size_t) s * H * D, vv.data() + (size_t) s * H * D, table, rk, rv, false, poss[s]);
        cmp_blocks("kv_append_q4_kernel 3 steps (K)", kpool, rk);
        cmp_blocks("kv_append_q4_kernel 3 steps (V)", vpool, rv);

        // the gather: ids with repeats, fp16 scratch bits from the reference blocks
        const int max_ids = 40;
        mtl::Buffer ids = cx.alloc(4 * max_ids), ksc = cx.alloc(2 * (size_t) max_ids * H * D),
                    vsc = cx.alloc(2 * (size_t) max_ids * H * D), gstep_b = cx.alloc(16);
        std::vector<int32_t> idsv(max_ids);
        for (int i = 0; i < max_ids; ++i) idsv[i] = poss[r.next() % n_steps];
        std::memcpy(ids.as<int32_t>(), idsv.data(), 4 * max_ids);
        const int32_t gstep[4] = {0, 0, 0, max_ids};
        std::memcpy(gstep_b.as<int32_t>(), gstep, 16);
        mtl::Dispatch g;
        g.p = &pgat;
        g.args = {mtl::Arg::B(kpool), mtl::Arg::B(vpool), mtl::Arg::B(table_b), mtl::Arg::B(ids), mtl::Arg::B(gstep_b),
                  mtl::Arg::B(ksc),   mtl::Arg::B(vsc),   mtl::Arg::V((int32_t) H), mtl::Arg::V((int32_t) D),
                  mtl::Arg::V((int32_t) P)};
        const int total_blocks = max_ids * H * BPH;
        g.grid[0] = (uint32_t) ((total_blocks + 3) / 4);
        g.tg[0] = 32;
        g.tg[1] = 4;
        cx.run({g});
        long bad = 0;
        for (int i = 0; i < max_ids; ++i)
            for (int h = 0; h < H; ++h)
                for (int t = 0; t < D; ++t) {
                    const int64_t row = ((int64_t) table[idsv[i] / P] * H + h) * P + idsv[i] % P;   // through the table
                    const block_q4_0* kb = (const block_q4_0*) (rk.data() + row * ROW_BYTES) + t / 32;
                    const block_q4_0* vb = (const block_q4_0*) (rv.data() + row * ROW_BYTES) + t / 32;
                    const int j = t % 32;
                    const int kq = (j < 16) ? ((kb->qs[j] & 0x0F) - 8) : ((kb->qs[j - 16] >> 4) - 8);
                    const int vq = (j < 16) ? ((vb->qs[j] & 0x0F) - 8) : ((vb->qs[j - 16] >> 4) - 8);
                    const int64_t off = ((int64_t) i * H + h) * D + t;
                    bad += ksc.as<uint16_t>()[off] != k::f16_from_f32((float) kq * k::f32_from_f16(kb->d));
                    bad += vsc.as<uint16_t>()[off] != k::f16_from_f32((float) vq * k::f32_from_f16(vb->d));
                }
        std::printf("  %-38s %ld of %d scratch values differ  %s\n", "kv_gather_q4_kernel (fp16 bits)", bad,
                    max_ids * H * D, bad ? "FAIL" : "PASS");
        g_fail += bad != 0;
    }

    // ---- 4. the prompt batch kernel: T cells from pos0, into the pool, the host copy and the stage pool
    {
        const int pages = 8, cells = pages * P, T = 17, pos0 = P - 8;   // crosses page 0 -> 1 inside one batch
        std::vector<int32_t> table(pages);
        for (int i = 0; i < pages; ++i) table[i] = (i * 3 + 5) % pages;
        mtl::Buffer table_b = cx.alloc(4 * pages), K = cx.alloc(4 * (size_t) T * H * D),
                    V = cx.alloc(4 * (size_t) T * H * D), kpool = cx.alloc((size_t) cells * H * ROW_BYTES),
                    vpool = cx.alloc((size_t) cells * H * ROW_BYTES), hpool_k = cx.alloc((size_t) cells * H * ROW_BYTES),
                    hpool_v = cx.alloc((size_t) cells * H * ROW_BYTES), spool_k = cx.alloc((size_t) cells * H * ROW_BYTES),
                    spool_v = cx.alloc((size_t) cells * H * ROW_BYTES);
        std::memcpy(table_b.as<int32_t>(), table.data(), 4 * pages);
        std::vector<float> kk((size_t) T * H * D), vv((size_t) T * H * D);
        for (size_t i = 0; i < kk.size(); ++i) {
            kk[i] = r.sym() * 2.0f;
            vv[i] = r.sym();
        }
        for (int64_t row = 0; row < (int64_t) T * H; ++row) {
            fwht256_ref(kk.data() + row * D, kk.data() + row * D, 1);
            fwht256_ref(vv.data() + row * D, vv.data() + row * D, 1);
        }
        std::memcpy(K.as<float>(), kk.data(), kk.size() * 4);
        std::memcpy(V.as<float>(), vv.data(), vv.size() * 4);
        std::vector<mtl::Dispatch> ds;
        for (int is_v = 0; is_v < 2; ++is_v) {
            mtl::Dispatch d;
            d.p = &pbat;
            d.args = {mtl::Arg::B(kpool), mtl::Arg::B(vpool), mtl::Arg::B(table_b), mtl::Arg::B(K), mtl::Arg::B(V),
                      mtl::Arg::B(hpool_k), mtl::Arg::B(hpool_v), mtl::Arg::B(spool_k), mtl::Arg::B(spool_v),
                      mtl::Arg::V((uint64_t) pos0), mtl::Arg::V((int32_t) H), mtl::Arg::V((int32_t) D),
                      mtl::Arg::V((int32_t) P), mtl::Arg::V((int32_t) is_v)};
            d.grid[0] = T;
            d.grid[1] = H;
            d.grid[2] = BPH;
            d.tg[0] = 32;
            ds.push_back(d);
        }
        cx.run(ds);
        std::vector<uint8_t> rk((size_t) cells * H * ROW_BYTES, 0), rv = rk, hk = rk, hv = rk, sk = rk, sv = rk;
        for (int t = 0; t < T; ++t) {
            const float* kc = kk.data() + (size_t) t * H * D, * vc = vv.data() + (size_t) t * H * D;
            append_cell_ref(kc, vc, table, rk, rv, false, pos0 + t);
            append_cell_ref(kc, vc, table, hk, hv, true, pos0 + t);
            append_cell_ref(kc, vc, table, sk, sv, true, pos0 + t);
        }
        cmp_blocks("kv_append_q4_batch_kernel (table)", kpool, rk);
        cmp_blocks("kv_append_q4_batch_kernel V (table)", vpool, rv);
        cmp_blocks("kv_append_q4_batch_kernel host K", hpool_k, hk);
        cmp_blocks("kv_append_q4_batch_kernel host V", hpool_v, hv);
        cmp_blocks("kv_append_q4_batch_kernel stage K", spool_k, sk);
        cmp_blocks("kv_append_q4_batch_kernel stage V", spool_v, sv);
    }

    // ---- 5. the composite chain: raw (unrotated) K and V -> one device fwht -> device quantize -> compare,
    // through a permuted table like the cases above
    {
        const int pages = 2, cells = pages * P;
        std::vector<int32_t> table(pages);
        for (int i = 0; i < pages; ++i) table[i] = (i * 3 + 1) % pages;   // a permutation (here: 1, 0)
        mtl::Buffer table_b = cx.alloc(4 * pages), step_b = cx.alloc(16), raw = cx.alloc(4 * 2 * H * D),
                    rot = cx.alloc(4 * 2 * H * D), kpool = cx.alloc((size_t) cells * H * ROW_BYTES),
                    vpool = cx.alloc((size_t) cells * H * ROW_BYTES);
        std::memcpy(table_b.as<int32_t>(), table.data(), 4 * pages);
        std::vector<float> x(2 * H * D);   // K's H rows, then V's
        for (size_t i = 0; i < x.size(); ++i) x[i] = r.sym() * 3.0f;
        std::memcpy(raw.as<float>(), x.data(), x.size() * 4);
        const int32_t pos = P + 7;
        const int32_t step[4] = {pos, pos + 1, 0, 0};
        std::memcpy(step_b.as<int32_t>(), step, 16);
        mtl::Dispatch fw;
        fw.p = &pfw;
        fw.args = {mtl::Arg::B(raw), mtl::Arg::B(rot), mtl::Arg::V((uint64_t) (2 * H)), mtl::Arg::V(1.0f / 16.0f)};
        fw.grid[0] = (uint32_t) ((2 * H + 3) / 4);
        fw.tg[0] = 32;
        fw.tg[1] = 4;
        mtl::Dispatch ap;
        ap.p = &papp;
        ap.args = {mtl::Arg::B(kpool),     mtl::Arg::B(vpool), mtl::Arg::B(table_b), mtl::Arg::B(step_b),
                   mtl::Arg::B(rot),       mtl::Arg::B(rot, 4 * (size_t) H * D), mtl::Arg::B(kNone), mtl::Arg::B(kNone),
                   mtl::Arg::V((int32_t) 4), mtl::Arg::V((int32_t) 2), mtl::Arg::V((int32_t) H), mtl::Arg::V((int32_t) D),
                   mtl::Arg::V((int32_t) P)};
        ap.grid[0] = H;
        ap.grid[1] = BPH;
        ap.grid[2] = 2;
        ap.tg[0] = 32;
        cx.run({fw, ap});   // one command buffer: the append reads the fwht's writes (ordered dispatches)
        std::vector<float> xrot(2 * H * D);
        fwht256_ref(x.data(), xrot.data(), 2 * H);   // the host runs the same chain in the same order
        std::vector<uint8_t> rk((size_t) cells * H * ROW_BYTES, 0), rv = rk;
        append_cell_ref(xrot.data(), xrot.data() + (size_t) H * D, table, rk, rv, false, pos);
        cmp_blocks("composite: device fwht -> device q4 K", kpool, rk);
        cmp_blocks("composite: device fwht -> device q4 V", vpool, rv);
    }

    std::printf("%s\n", g_fail ? "SOME FAILED" : "ALL PASS");
    return g_fail ? 1 : 0;
}
