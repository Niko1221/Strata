// metal/tests/kv_format_test.cpp - the engine's K/V format wiring (src/engine/kv_format.h, M1-b: --kv q4_0 real on
// Metal; M2-b: the prompt chains format-aware) pinned at dispatch level: the pool sizes by format, the startup
// format check (k8v4's explicit error), the qsa pipeline key / Constants composition (decode, MTP and both prompt
// chains), the prompt-path choice - and, with a real context, the STALE-KEY TRAP
// behaviorally: Window::P() looks the key up first, so a key that left the format out would hand back the cached
// int8 specialization of the same function to run over q4 pools.  The engine's own pools cannot be allocated in
// tree (Window::init needs the model), so this drives the one source both use - kv_format.h - under the engine's
// exact lookup and dispatch discipline.  qsa_q4_parity.cpp stays the numerics gate of the fc-25 kernels.
#include "../src/engine/kv_format.h"

#include <cstdio>
#include <cstring>
#include <map>
#include <string>
#include <vector>

using namespace strata::metal_engine;

namespace {
int fails = 0;
void check(bool ok, const char* what) {
    std::printf("%s %s\n", ok ? "ok  " : "FAIL", what);
    fails += !ok;
}
bool has(const mtl::Constants& c, int idx, int type, uint32_t v) {
    for (const auto& e : c.vals)
        if (e.index == idx && e.type == type && e.v == v) return true;
    return false;
}

// the engine's qsa geometry (window.cpp's N/HC constants of the QSA layers)
constexpr int QH = 24, QKV = 2, QD = 256, IH = 4, ID = 128, MAXC = 64;
struct QsaArgs { uint32_t pos0, pos_base, max_cells, n_tok; float eps; };
}  // namespace

int main() {
    std::printf("device-free checks\n");
    // ---- 1. the pool sizes by format (what Window::init allocates per QSA layer, K and V each)
    {
        const KvPoolBytes i8 = kv_pool_bytes(false), q4 = kv_pool_bytes(true);
        check(i8.pool == 2 * 256 && i8.scale == 2 * 4 * 2, "int8/fp16 pools: [cell][2 heads][256] + [cell][2 heads][4] f16 scales (today's)");
        check(q4.pool == 2 * 144 && q4.scale == 0, "q4 pools: [cell][2 heads][144] (8 x 18-byte blocks, the scale inside), no scale pool");
        // 576 B/cell across K + V, as the pool layout of qsa.metal's fc-25 mode
        check(2 * q4.pool == 576, "q4 pools: K + V together 576 B/cell");
    }

    // ---- 2. the startup format check (Window::init): k8v4 is refused, was silently int8 before
    {
        check(kv_format_check(false, false).empty(), "int8/fp16: no error");
        check(kv_format_check(true, false).empty(), "q4_0: no error (this PR's path)");
        const std::string e = kv_format_check(false, true);
        check(!e.empty() && e.find("k8v4") != std::string::npos && e.find("not implemented") != std::string::npos,
              "k8v4: an explicit startup error naming it");
        check(!kv_format_check(true, true).empty(), "hybrid refused whatever the q4 flag is (the flags are exclusive in generate.cpp)");
    }

    // ---- 3. the key / Constants composition: every function constant in BOTH, and the keys pairwise apart
    {
        check(qsa_prep_key(false, false) == "qsa_prep" && qsa_prep_key(true, false) == "qsa_prep#mrope",
              "prep keys without q4 are today's literals (the default path's caches unchanged)");
        const std::string k00 = qsa_prep_key(false, false), k10 = qsa_prep_key(true, false),
                          k01 = qsa_prep_key(false, true), k11 = qsa_prep_key(true, true);
        check(k00 != k01 && k00 != k10 && k00 != k11 && k01 != k10 && k01 != k11 && k10 != k11,
              "the four prep keys (mrope x q4) are pairwise distinct");
        check(qsa_prep_constants(false, false).vals.empty(), "prep constants without mrope/q4 are empty (today's call)");
        check(has(qsa_prep_constants(true, false), 24, 1, 1) && qsa_prep_constants(true, false).vals.size() == 1, "mrope: fc 24 only");
        check(has(qsa_prep_constants(false, true), 25, 1, 1) && qsa_prep_constants(false, true).vals.size() == 1, "q4: fc 25 only");
        check(has(qsa_prep_constants(true, true), 24, 1, 1) && has(qsa_prep_constants(true, true), 25, 1, 1) &&
                  qsa_prep_constants(true, true).vals.size() == 2, "mrope + q4: fc 24 and fc 25");
        check(qsa_q4_key("qsa_attn", false) == "qsa_attn" && qsa_q4_key("qsa_attn", true) == "qsa_attn#q4" &&
                  qsa_q4_key("qsa_merge", true) == "qsa_merge#q4", "attn/merge keys carry #q4 when on");
        check(qsa_q4_constants(false).vals.empty() && has(qsa_q4_constants(true), 25, 1, 1) &&
                  qsa_q4_constants(true).vals.size() == 1, "attn/merge constants: fc 25 when on, empty when off");
        // the prompt chains' mm sites (M2-b): today's literals plus #q4, the layout's fc 7 always bound (its MSL
        // default is TRUE when left out, so a !k4 flash must set it false, not drop it)
        check(qsa_prompt_mm_key("qsa_attn_mm#k4", false) == "qsa_attn_mm#k4" && qsa_prompt_mm_key("qsa_flash", false) == "qsa_flash",
              "the mm keys without q4 are today's literals (both chains' caches unchanged)");
        check(qsa_prompt_mm_key("qsa_attn_mm#k4", true) == "qsa_attn_mm#k4#q4" && qsa_prompt_mm_key("qsa_flash", true) == "qsa_flash#q4",
              "the mm keys carry #q4 when on");
        check(has(qsa_prompt_mm_constants(true, false), 7, 1, 1) && qsa_prompt_mm_constants(true, false).vals.size() == 1,
              "attn_mm's constants without q4: fc 7 true only (today's call)");
        check(has(qsa_prompt_mm_constants(false, false), 7, 1, 0) && qsa_prompt_mm_constants(false, false).vals.size() == 1,
              "flash without the k4 layout or q4: fc 7 FALSE, still bound (today's b(7, p.k4) call)");
        check(has(qsa_prompt_mm_constants(true, true), 7, 1, 1) && has(qsa_prompt_mm_constants(true, true), 25, 1, 1) &&
                  qsa_prompt_mm_constants(true, true).vals.size() == 2, "attn_mm + q4: fc 7 and fc 25 in one set");
        check(has(qsa_prompt_mm_constants(false, true), 7, 1, 0) && has(qsa_prompt_mm_constants(false, true), 25, 1, 1) &&
                  qsa_prompt_mm_constants(false, true).vals.size() == 2, "flash + q4 without the k4 layout: fc 7 false and fc 25");
    }

    // ---- 4. the prompt-path choice (M2-b): every format runs the fast prompt chains (format-aware keys), the env
    //         hatches (STRATA_METAL_PROMPT=window and friends) still divert to the decode window
    {
        check(prompt_fast_path(false), "int8/fp16: the fast prompt paths as today");
        check(prompt_fast_path(true), "q4: the fast prompt paths too (the chains' dispatches carry the format)");
    }

    // ---- 5. the stale-key trap, behaviorally: the engine's lookup discipline over the real library
    std::printf("device checks\n");
    {
        mtl::Context cx;
        // the engine's lqsa_ library: kv_q4.metal's shared prelude in front of qsa.metal (window.cpp's compile)
        const int lib = cx.compile({"kv_q4.metal", "qsa.metal"}, "#define NO_GGML_COMMON 1\n");
        // Window::P() verbatim in shape: the key lookup FIRST, the pipeline (with its Constants) only on a miss
        std::map<std::string, mtl::Pipeline> cache;
        auto P = [&](const std::string& fn, const mtl::Constants& c, const std::string& key) -> const mtl::Pipeline* {
            auto it = cache.find(key);
            if (it == cache.end()) it = cache.emplace(key, cx.pipeline(lib, fn, c)).first;
            return &it->second;
        };
        const mtl::Pipeline* p8 = P("qsa_prep", qsa_prep_constants(false, false), qsa_prep_key(false, false));
        const mtl::Pipeline* p4 = P("qsa_prep", qsa_prep_constants(false, true), qsa_prep_key(false, true));
        check(p4 != p8 && p4->handle != p8->handle && cache.size() == 2,
              "the q4 key MISSES the cached int8 pipeline: a second specialization, not the same object");
        check(P("qsa_prep", qsa_prep_constants(false, false), qsa_prep_key(false, false)) == p8,
              "the int8 entry is still the int8 pipeline after the q4 insert");
        // the counterfactual this key guards against: the pre-M1-b key (no q4 term) WOULD have returned the int8
        // pipeline object for the q4 constants - the silent-wrong-pipeline trap, here shown, not run
        check(P("qsa_prep", qsa_prep_constants(false, true), qsa_prep_key(false, false)) == p8,
              "counterfactual: a key without the format hands back the int8 pipeline (why the key carries q4)");

        // both pipelines over an all-zero fixture (v rows raw, no norm/RoPE): the pools' bytes say which
        // specialization ran.  The engine's dispatch sites bind these same buffers in this order.
        auto A = [&](size_t bytes) { return cx.alloc(bytes); };
        mtl::Buffer qfull = A(4 * (size_t) QH * 2 * QD), kcur = A(4 * (size_t) QKV * QD), vcur = A(4 * (size_t) QKV * QD),
                    iqr = A(4 * (size_t) IH * ID), ikr = A(4 * (size_t) ID), qn = A(4 * QD), kn = A(4 * QD),
                    iqn = A(4 * ID), ikn = A(4 * ID), cs = A(8 * (size_t) (MAXC + 16) * 32), qcur = A(4 * (size_t) QH * QD),
                    qidx = A(4 * (size_t) IH * ID), tail = A(4 * 4 * ID), dead = A(4 * ID),
                    pooled = A(4 * (size_t) (MAXC / 4 + 2) * ID);
        for (int i = 0; i < QD; ++i) { qn.as<float>()[i] = 1.0f; kn.as<float>()[i] = 1.0f; }
        for (int i = 0; i < ID; ++i) { iqn.as<float>()[i] = 1.0f; ikn.as<float>()[i] = 1.0f; }
        const KvPoolBytes i8b = kv_pool_bytes(false), q4b = kv_pool_bytes(true);
        mtl::Buffer kq8 = A((size_t) MAXC * i8b.pool), vq8 = A((size_t) MAXC * i8b.pool),
                    ksc8 = A((size_t) MAXC * i8b.scale), vsc8 = A((size_t) MAXC * i8b.scale),
                    kq4 = A((size_t) MAXC * q4b.pool), vq4 = A((size_t) MAXC * q4b.pool),
                    sc_none = A(16);   // the engine's placeholder for the unallocated scale pools
        auto run = [&](const mtl::Pipeline* p, const mtl::Buffer& kq, const mtl::Buffer& vq, const mtl::Buffer& ksc,
                       const mtl::Buffer& vsc, const mtl::Buffer& mtab) {
            QsaArgs a{0, 0, (uint32_t) MAXC, 1, 1e-6f};
            mtl::Dispatch d;
            d.p = p;
            d.args = {mtl::Arg::B(qfull), mtl::Arg::B(kcur), mtl::Arg::B(vcur), mtl::Arg::B(iqr), mtl::Arg::B(ikr),
                      mtl::Arg::B(qn), mtl::Arg::B(kn), mtl::Arg::B(iqn), mtl::Arg::B(ikn), mtl::Arg::B(cs),
                      mtl::Arg::B(qcur), mtl::Arg::B(qidx), mtl::Arg::B(kq), mtl::Arg::B(vq), mtl::Arg::B(ksc),
                      mtl::Arg::B(vsc), mtl::Arg::B(tail), mtl::Arg::B(dead), mtl::Arg::B(pooled), mtl::Arg::V(a),
                      mtl::Arg::B(mtab)};
            d.grid[0] = 33;
            d.grid[1] = 1;
            d.tg[0] = 256;
            cx.run({d});
        };
        // q4 of an all-zero row: d = -0.0 (bits 0x8000), every qs byte 0x88 - both heads, all 8 blocks
        auto q4_zero_bits = [&](const mtl::Buffer& pool) {
            long bad = 0;
            for (int h = 0; h < QKV; ++h)
                for (int b = 0; b < 8; ++b) {
                    const uint16_t* d = (const uint16_t*) ((const uint8_t*) pool.ptr + (size_t) h * 144) + b * 9;
                    bad += *d != 0x8000;
                    for (int j = 0; j < 16; ++j) bad += ((const uint8_t*) pool.ptr + (size_t) h * 144)[b * 18 + 2 + j] != 0x88;
                }
            return bad;
        };
        run(p8, kq8, vq8, ksc8, vsc8, sc_none);
        {
            // int8 of an all-zero row: every code byte 0 and every fp16 scale 0
            long bad = 0;
            for (int i = 0; i < 2 * QD; ++i) bad += vq8.as<uint8_t>()[i] != 0;
            for (int i = 0; i < (int) (MAXC * i8b.scale / 2); ++i) bad += vsc8.as<uint16_t>()[i] != 0;
            check(bad == 0, "the int8-keyed pipeline writes the int8 pools (zero codes, zero scales)");
        }
        run(p4, kq4, vq4, sc_none, sc_none, sc_none);
        check(q4_zero_bits(vq4) == 0, "the q4-keyed pipeline writes the q4 pools (d = 0x8000, qs = 0x88) at the 144-byte rows");
        // the combined specialization (M-RoPE + q4: a vision prompt under --kv q4_0), compiled and dispatched in
        // tree so the pair is not first exercised at runtime.  fc 24 makes prep read the M-RoPE table (buffer
        // 20); with every axis's position the cell's own, the rotation is the plain one - and the v rows (raw,
        // never RoPE'd) keep the same all-zero q4 bits.  The pool is scribbled first, so the check proves the
        // dispatch wrote it, not that an old result survived.  The identity table cannot tell "reads mtab" from
        // "ignores it" - that discrimination (a random table, the K pool's numbers) is qsa_q4_parity's pass 5.
        mtl::Buffer mtab = A(4 * 3 * (size_t) (MAXC + 16));
        for (int c = 0; c < MAXC + 16; ++c)
            for (int i = 0; i < 3; ++i) mtab.as<int32_t>()[(size_t) c * 3 + i] = c;
        const mtl::Pipeline* pmq = P("qsa_prep", qsa_prep_constants(true, true), qsa_prep_key(true, true));
        check(cache.size() == 3 && pmq != p8 && pmq != p4, "mrope+q4: a third key, a third specialization");
        std::memset(vq4.ptr, 0xAA, (size_t) MAXC * q4b.pool);
        run(pmq, kq4, vq4, sc_none, sc_none, mtab);
        check(q4_zero_bits(vq4) == 0, "the mrope+q4 pipeline compiles and writes the same q4 pool bits");

        // the PROMPT chains' own library and keys (M2-b): the same trap one file later.  The chains dispatch
        // qsa_attn / qsa_merge (both) and qsa_attn_mm / qsa_flash (the matrix-unit sites) through these helpers,
        // and a key without #q4 would hand them the cached int8 specialization over q4 pools.  Each pipeline
        // creation compiles the exact constant set an engine site now binds; the fc-25 arms' NUMERICS over a full
        // prompt geometry are qsa_prompt_parity's q4 cases (dispatched there, not here).
        const int plib = cx.compile({"kv_q4.metal", "qsa.metal", "qsa_prompt.metal"}, "#define NO_GGML_COMMON 1\n");
        std::map<std::string, mtl::Pipeline> ppc;
        auto PP = [&](const std::string& fn, const mtl::Constants& c, const std::string& key) -> const mtl::Pipeline* {
            auto it = ppc.find(key);
            if (it == ppc.end()) it = ppc.emplace(key, cx.pipeline(plib, fn, c)).first;
            return &it->second;
        };
        const mtl::Pipeline* ca8 = PP("qsa_attn", qsa_q4_constants(false), qsa_q4_key("qsa_attn", false));
        const mtl::Pipeline* ca4 = PP("qsa_attn", qsa_q4_constants(true), qsa_q4_key("qsa_attn", true));
        check(ca4 != ca8 && ca4->handle != ca8->handle && ppc.size() == 2,
              "a prompt chain's q4 attn key misses the cached int8 pipeline: a second specialization");
        check(PP("qsa_attn", qsa_q4_constants(true), qsa_q4_key("qsa_attn", false)) == ca8,
              "counterfactual: a prompt key without the format hands back the int8 pipeline (why every prompt key carries q4)");
        const mtl::Pipeline* cm8 = PP("qsa_attn_mm", qsa_prompt_mm_constants(true, false), qsa_prompt_mm_key("qsa_attn_mm#k4", false));
        const mtl::Pipeline* cm4 = PP("qsa_attn_mm", qsa_prompt_mm_constants(true, true), qsa_prompt_mm_key("qsa_attn_mm#k4", true));
        check(cm4 != cm8 && cm4 != ca4 && cm4 != ca8 && ppc.size() == 4,
              "attn_mm: the k4 key and the k4#q4 key are two specializations of the one function");
        const mtl::Pipeline* cf4 = PP("qsa_flash", qsa_prompt_mm_constants(false, true), qsa_prompt_mm_key("qsa_flash", true));
        check(cf4 != cm4 && ppc.size() == 5, "flash: its q4 key compiles the !k4 layout's constant set (fc 7 false) as its own");
        // flash's k4-hardware site (an M5's probe) binds {fc 7 true, fc 25} under the SAME qsa_prompt_mm_key -
        // one process probes one layout, so the engine's cache never holds both sets under it (that is why the
        // key carries no fc 7); compiled here BESIDE the cache, still the constant set that site binds
        const mtl::Pipeline cfk4 = cx.pipeline(plib, "qsa_flash", qsa_prompt_mm_constants(true, true));
        check(cfk4.handle != cf4->handle && cfk4.handle != cm4->handle && ppc.size() == 5,
              "flash's k4-hardware constant set (fc 7 true beside fc 25): its own specialization under the shared key");
        const mtl::Pipeline* cg4 = PP("qsa_merge", qsa_q4_constants(true), qsa_q4_key("qsa_merge", true));
        check(cg4 != ca4 && ppc.size() == 6, "merge: its q4 key is its own specialization");
    }

    std::printf("%s\n", fails ? "SOME FAILED" : "ALL PASS");
    return fails ? 1 : 0;
}
