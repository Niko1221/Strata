// engine/metal/src/hy4flash_gguf/msl/hy4_flash_attn.metal - MLA attention, the sampler and the small shape kernels the block needs
// (semantics.md §8).
//
// MLA, ABSORBED.  The projections wk_b and wv_b are folded into the query and the output by the host as plain GEMVs
// (one dispatch a head: wk_b is [n_head][512][192], wv_b [n_head][256][512]), so what is left here is the attention
// over ONE shared key per position:
//
//   Q[t][h] = [ wk_b[h] . q_nope[t][h] | rope(q[t][h][192:256]) ]      576 floats
//   K[s]    = [ rms_norm(kv_a) | rope(k_pe) ]                          576 halves, the cache row
//   V[s]    = K[s][0:512]                                              the same halves
//   sc      = Q[t][h] . K[s] * 1/sqrt(256)
//   m       = max(max_s sc, sink[h])          the sink joins the MAXIMUM as well as the denominator
//   e       = exp(sc - m);  z = sum_s e + exp(sink - m)
//   o[t][h] = sum_s (e[s] / z) * V[s]                                  512 floats
//
// TWO PASSES OVER A STORED SCORE ROW, not an online softmax.  The reference computes the whole score row, takes its
// maximum and only then exponentiates (hy4_block.py attention), and an online pass with a running rescale is a
// DIFFERENT arithmetic - every term picks up the rounding of however many rescales followed it.  The scores fit: a
// sparse block's query attends at most top_k + kpool - 1 = 2051 cells and a sliding block's at most 512, so the
// scratch is n_tok * n_head * 2051 floats = 4.2 MB for a decode window, and keeping it buys the reference's own
// order.
//
// T INVARIANCE.  The softmax is one thread group per (head, token) with a compile-time thread count, so neither
// the maximum nor the denominator can see how many tokens the dispatch carries; the visible set is a function of
// the absolute position (the sliding window's lower bound, or the indexer's selection, which is itself
// position-only).  The dot product runs over k ascending in one thread, the sum over keys runs over a fixed
// butterfly, and the PV sum runs over the keys in ascending order in the thread that owns the dimension - wherever
// in the grid that thread sits (the three-dispatch note below).

#define HY4_FLASH_ATTN_SINKMAX 1   // documentation only: the sink takes part in the maximum (hy4_block.py line 356)

struct Hy4FlashAttnArgs {
    uint n_head;
    uint d_qk;         // 576 = kv_lora_rank + rope_dim
    uint d_v;          // 512 = kv_lora_rank
    uint kv_stride;    // halves between cache rows
    uint q_stride;     // floats between tokens of q (n_head * d_qk)
    uint o_stride;     // floats between tokens of o (n_head * d_v)
    uint sel_stride;   // ints between tokens of sel
    uint sc_stride;    // floats between (token, head) rows of the score scratch
    uint window;       // sliding window; 0 = the whole causal prefix
    float kq_scale;
};

// function constant 43 (hy4_flash_norm.metal explains the range): true = the attended cells come from the indexer's
// selection buffer, false = the contiguous range the position and the window give.
constant bool HY4_FLASH_ATTN_SEL [[function_constant(43)]];
constant bool hy4_flash_attn_sel = is_function_constant_defined(HY4_FLASH_ATTN_SEL) ? HY4_FLASH_ATTN_SEL : false;


static inline float hy4_flash_tg_max(float v, threadgroup float* part, ushort sg, ushort lane) {
    for (ushort o = 16; o > 0; o >>= 1) v = max(v, simd_shuffle_xor(v, o));
    if (lane == 0) part[sg] = v;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float m = part[0];
    for (ushort k = 1; k < HY4_FLASH_TG_SG; ++k) m = max(m, part[k]);
    return m;
}

// ---- the attention: THREE DISPATCHES, the same arithmetic ------------------------------------------------------------
//
// The attention is split at its two natural seams, one dispatch each:
//
//   hy4_flash_mla_qk       the score row:     sc[t][h][i] = (Q[t][h] . K[s_i]) * kq_scale
//   hy4_flash_mla_softmax  the softmax:       sc[t][h][i] = exp(sc - m) / z,   one thread group per (head, token)
//   hy4_flash_mla_pv       the output:        o[t][h][d]  = sum_i sc[t][h][i] * V[s_i][d]
//
// WHY.  One group per (head, token) is 32 groups for 16 cores on a sparse block (64 on a sliding one), and inside a
// group a single thread would walk a key's whole 576-long dot and then 2 output dimensions over every key, dividing
// `scr[i] / z` again for every dimension.  The score of a key and the sum of an output dimension are each
// INDEPENDENT of every other key's score and every other dimension's sum; only the softmax couples a row.  So the
// two big passes are spread over the keys (qk) and the output dimensions (pv) of many thread groups, and one thread
// carries SEVERAL independent chains - HB heads of the same key in qk (the key row is loaded once for all of them),
// HB heads x VEC dimensions in pv - for ILP.  Nothing here splits a sum.
//
// BITWISE THE ONE-KERNEL VERSION, by construction (tmp/tests/hy4_flash/attn runs that kernel verbatim against these):
//   * each dot is still ONE accumulator over k = 0 .. d_qk-1 ascending, `acc += q[k] * float(K[k])`, contraction
//     off for the whole library (hy4_flash_fp.metal), then `* kq_scale` - its loop's operations in its order.  Loading K
//     four halves at a time changes no value (half -> float is exact) and no order;
//   * the softmax is that code moved, not rewritten: HY4_FLASH_TG threads, thread `tid` owns keys tid, tid + HY4_FLASH_TG,
//     ... in ascending order, the same per-thread maximum and the same hy4_flash_tg_max, the same precise::exp, the same
//     per-thread sum and hy4_flash_tg_sum tree, the sink's term last.  The maximum is recomputed from the stored scores
//     (bit-for-bit the values it kept in a register), in the same per-thread order;
//   * the weight is the same FP32 division `e / z` (not a reciprocal multiply), done ONCE per key instead of once
//     per key per output dimension; it is stored as a float, so the PV product sees exactly the value the
//     `scr[i] / z * v` expression computed and rounded;
//   * each output dimension is ONE accumulator over the keys in ascending order (`acc += w * float(V[d])`), the key
//     chunks taken in order with the accumulator carried across them.
// T INVARIANCE holds either way: every value is a function of (head, token) and its visible set alone; the grid's
// token axis only says how many such rows run.
//
// The score scratch `sc` ends holding the normalised weights, not the exponentials.  Nothing reads it
// after the attention (it is scratch: forward.cpp / mtp.cpp bind it to these three dispatches only).

#define HY4_FLASH_ATTN_DQK_MAX 576   // the widest q / k row qk stages in threadgroup memory (d_qk = 512 + 64 here)
#define HY4_FLASH_ATTN_PV_CH 256     // keys whose weights and cache rows pv stages in threadgroup memory at a time

struct Hy4FlashAttnRow {
    int lo;   // the first cell of a range row
    int n;    // keys in the row
};
static inline Hy4FlashAttnRow hy4_flash_attn_row(device const int* n_sel, device const int* pos, constant Hy4FlashAttnArgs& a,
                                      uint t) {
    const int p = pos[t];
    Hy4FlashAttnRow r;
    r.lo = hy4_flash_attn_sel ? 0 : (a.window != 0 ? max(0, p - int(a.window) + 1) : 0);
    r.n = hy4_flash_attn_sel ? n_sel[t] : p - r.lo + 1;
    return r;
}

// The three kernels share one argument list, so the host binds the same nine arguments to each.
#define HY4_FLASH_ATTN_ARGS                                                                                                \
    device const float *q [[buffer(0)]], device const half *kv [[buffer(1)]], device const float *sinks [[buffer(2)]], \
        device const int *sel [[buffer(3)]], device const int *n_sel [[buffer(4)]], device const int *pos [[buffer(5)]], \
        device float *sc [[buffer(6)]], device float *o [[buffer(7)]], constant Hy4FlashAttnArgs &a [[buffer(8)]],         \
        uint3 tgp [[threadgroup_position_in_grid]], ushort tid [[thread_index_in_threadgroup]],                     \
        ushort sg [[simdgroup_index_in_threadgroup]], ushort lane [[thread_index_in_simdgroup]]

// ---- the score row
//
// Staging d through threadgroup memory so that consecutive lanes load consecutive d of one key buys nothing here: a
// lane walking ITS key's row sequentially already consumes every fetched line over its next iterations.  What this
// split attacks instead is the ILP - a 576-long dependent chain per key, one chain per thread, and 32 thread groups
// for 16 cores on a sparse block.  Here a thread still walks its key's row in order, but for HB heads at once, over
// n_cap / QK_TG x n_head / HB groups.
//
// grid (ceil(n_cap / QK_TG), ceil(n_head / HB), n_tok), threads QK_TG.  Thread `tid` of group x owns key
// i = x * QK_TG + tid for HB consecutive heads; the HB queries are staged in threadgroup memory ([k][j]) and the key
// row is read once, four halves at a time, for all of them.  Needs d_qk <= HY4_FLASH_ATTN_DQK_MAX and kv rows 8-byte
// aligned (kv_stride % 4 == 0, the binding at a multiple of 8 bytes) - the host checks both.
template <int HB, int QK_TG, int U>
kernel void hy4_flash_mla_qk_t(HY4_FLASH_ATTN_ARGS) {
    const uint t = tgp.z, h0 = tgp.y * HB;
    const Hy4FlashAttnRow r = hy4_flash_attn_row(n_sel, pos, a, t);
    const int i0 = int(tgp.x) * QK_TG;
    if (i0 >= r.n) return;   // the whole group is past this row's keys: uniform, and before the barrier
    threadgroup float qs[HY4_FLASH_ATTN_DQK_MAX * HB];
    device const float* qt = q + (ulong)t * a.q_stride;
    for (uint x = tid; x < a.d_qk * HB; x += QK_TG) {
        const uint j = x / a.d_qk, k = x - j * a.d_qk, h = h0 + j;
        qs[k * HB + j] = h < a.n_head ? qt[(ulong)h * a.d_qk + k] : 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const int i = i0 + int(tid);
    if (i >= r.n) return;
    const int s = hy4_flash_attn_sel ? sel[(ulong)t * a.sel_stride + i] : r.lo + i;
    device const half* kr = kv + (ulong)s * a.kv_stride;
    float acc[HB];
    for (int j = 0; j < HB; ++j) acc[j] = 0.0f;
    // U half4 loads are issued before the arithmetic that consumes them, so a thread waits for one memory latency
    // per 4U elements instead of one per 4 (the loop is latency-bound, not ALU-bound); the arithmetic is still the
    // k-ascending chain, element by element.
    uint k = 0;
    for (; k + 4 * U <= a.d_qk; k += 4 * U) {
        float4 kf[U];
        for (int u = 0; u < U; ++u) kf[u] = float4(*(device const half4*)(kr + k + 4 * u));
        for (int u = 0; u < U; ++u)
            for (int c = 0; c < 4; ++c)
                for (int j = 0; j < HB; ++j) acc[j] += qs[(k + 4 * u + c) * HB + j] * kf[u][c];
    }
    for (; k + 4 <= a.d_qk; k += 4) {
        const float4 kf = float4(*(device const half4*)(kr + k));
        for (int c = 0; c < 4; ++c)
            for (int j = 0; j < HB; ++j) acc[j] += qs[(k + c) * HB + j] * kf[c];
    }
    for (; k < a.d_qk; ++k) {
        const float kf = float(kr[k]);
        for (int j = 0; j < HB; ++j) acc[j] += qs[k * HB + j] * kf;
    }
    for (int j = 0; j < HB; ++j) {
        const uint h = h0 + j;
        if (h < a.n_head) sc[(ulong)(t * a.n_head + h) * a.sc_stride + i] = acc[j] * a.kq_scale;
    }
}

// ---- the softmax over a stored score row (the sink joins the maximum and, last, the denominator)
// grid (n_head, n_tok), threads HY4_FLASH_TG.  The old kernel's middle, op for op (see the note above).
kernel void hy4_flash_mla_softmax(HY4_FLASH_ATTN_ARGS) {
    const uint h = tgp.x, t = tgp.y;
    const Hy4FlashAttnRow r = hy4_flash_attn_row(n_sel, pos, a, t);
    const int n = r.n;
    device float* scr = sc + (ulong)(t * a.n_head + h) * a.sc_stride;
    threadgroup float part[HY4_FLASH_TG_SG];
    float mx = -INFINITY;
    for (int i = tid; i < n; i += HY4_FLASH_TG) mx = max(mx, scr[i]);
    mx = hy4_flash_tg_max(mx, part, sg, lane);
    const float sk = sinks[h];
    const float m = max(mx, sk);
    float s1 = 0.0f;
    for (int i = tid; i < n; i += HY4_FLASH_TG) {
        const float e = precise::exp(scr[i] - m);
        scr[i] = e;
        s1 += e;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);   // every reader of part[] from the maximum is done
    const float z = hy4_flash_tg_sum(s1, part, sg, lane) + precise::exp(sk - m);
    // The reference divides before the matmul ((e / z) @ V), and so did the old PV loop, once per dimension.
    for (int i = tid; i < n; i += HY4_FLASH_TG) scr[i] = scr[i] / z;
}

// ---- PV
// VEC consecutive halves of a cache row, as floats (half -> float is exact)
template <int VEC>
static inline void hy4_flash_attn_load_v(device const half* vr, thread float* v) {
    if (VEC == 4) {
        const float4 v4 = float4(*(device const half4*)vr);
        for (int c = 0; c < 4; ++c) v[c] = v4[c];
    } else if (VEC == 2) {
        const float2 v2 = float2(*(device const half2*)vr);
        for (int c = 0; c < 2; ++c) v[c] = v2[c];
    } else {
        for (int c = 0; c < VEC; ++c) v[c] = float(vr[c]);
    }
}

// grid (ceil(d_v / (PV_TG * VEC)), ceil(n_head / HB), n_tok), threads PV_TG.  Thread `tid` of group x owns the VEC
// consecutive dimensions from d0 = (x * PV_TG + tid) * VEC for HB consecutive heads: HB x VEC independent
// accumulators, each over the keys in ascending order.  A chunk of HY4_FLASH_ATTN_PV_CH keys has its HB weights and its
// cache row staged in threadgroup memory per round.  Needs d_v % VEC == 0 and (for VEC > 1) the rows aligned as qk.
template <int HB, int VEC, int PV_TG, int U>
kernel void hy4_flash_mla_pv_t(HY4_FLASH_ATTN_ARGS) {
    const uint t = tgp.z, h0 = tgp.y * HB;
    const Hy4FlashAttnRow r = hy4_flash_attn_row(n_sel, pos, a, t);
    const uint d0 = (tgp.x * PV_TG + tid) * VEC;
    threadgroup float ws[HY4_FLASH_ATTN_PV_CH * HB];   // [i][j]
    threadgroup int ss[HY4_FLASH_ATTN_PV_CH];
    device const float* scr = sc + (ulong)(t * a.n_head + h0) * a.sc_stride;
    float acc[HB][VEC];
    for (int j = 0; j < HB; ++j)
        for (int c = 0; c < VEC; ++c) acc[j][c] = 0.0f;
    // (the chunk loop's trip count is the same in every thread of the group - the barriers inside need that)
    for (int c0 = 0; c0 < r.n; c0 += HY4_FLASH_ATTN_PV_CH) {
        const int cn = min(HY4_FLASH_ATTN_PV_CH, r.n - c0);
        threadgroup_barrier(mem_flags::mem_threadgroup);   // the previous chunk has been consumed
        for (int x = tid; x < cn * HB; x += PV_TG) {
            const int j = x / cn, i = x - j * cn;
            ws[i * HB + j] = h0 + uint(j) < a.n_head ? scr[(ulong)j * a.sc_stride + c0 + i] : 0.0f;
        }
        for (int x = tid; x < cn; x += PV_TG)
            ss[x] = hy4_flash_attn_sel ? sel[(ulong)t * a.sel_stride + c0 + x] : r.lo + c0 + x;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (d0 < a.d_v) {
            // U keys' cache rows are loaded before the arithmetic that consumes them (one memory latency per U keys
            // instead of per key: the loop is latency-bound); the sums still take the keys one by one, ascending.
            int i = 0;
            for (; i + U <= cn; i += U) {
                float v[U][VEC];
                for (int u = 0; u < U; ++u) hy4_flash_attn_load_v<VEC>(kv + (ulong)ss[i + u] * a.kv_stride + d0, v[u]);
                for (int u = 0; u < U; ++u)
                    for (int j = 0; j < HB; ++j) {
                        const float w = ws[(i + u) * HB + j];
                        for (int c = 0; c < VEC; ++c) acc[j][c] += w * v[u][c];
                    }
            }
            for (; i < cn; ++i) {
                float v[VEC];
                hy4_flash_attn_load_v<VEC>(kv + (ulong)ss[i] * a.kv_stride + d0, v);
                for (int j = 0; j < HB; ++j) {
                    const float w = ws[i * HB + j];
                    for (int c = 0; c < VEC; ++c) acc[j][c] += w * v[c];
                }
            }
        }
    }
    if (d0 >= a.d_v) return;
    for (int j = 0; j < HB; ++j) {
        const uint h = h0 + j;
        if (h >= a.n_head) continue;
        device float* orow = o + (ulong)t * a.o_stride + (ulong)h * a.d_v + d0;
        for (int c = 0; c < VEC; ++c) orow[c] = acc[j][c];
    }
}

// The tiling the engine runs.  The host name CARRIES THE TILE (attn.h builds it from kAttnTile), so a host grid and
// a compiled tile that disagree fail at pipeline lookup instead of silently dropping keys or dimensions.  Changing
// one moves no bit - every choice above is a tiling of independent chains - only the speed.
#define HY4_FLASH_ATTN_QK_INST(HB, TG, U, NAME) \
    template [[host_name(NAME)]] kernel void hy4_flash_mla_qk_t<HB, TG, U>(HY4_FLASH_ATTN_ARGS);
#define HY4_FLASH_ATTN_PV_INST(HB, VEC, TG, U, NAME) \
    template [[host_name(NAME)]] kernel void hy4_flash_mla_pv_t<HB, VEC, TG, U>(HY4_FLASH_ATTN_ARGS);
HY4_FLASH_ATTN_QK_INST(4, 128, 2, "hy4_flash_mla_qk_h4_t128_u2")
HY4_FLASH_ATTN_PV_INST(2, 2, 64, 16, "hy4_flash_mla_pv_h2_v2_t64_u16")

// ---- the sampler ---------------------------------------------------------------------------------------------------
// argmax over a row, ties to the SMALLER id.  One thread group per token; the lane sweep is ascending and
// the fold keeps the smaller id on a tie, the same shape hy4_flash_router.metal uses for its rank loop.
struct Hy4FlashArgmaxArgs {
    uint n;
    uint stride;
};

// grid (n_tok), threads 32 * k for k in 1..32.  Each thread sweeps its stride ascending (so a strict > keeps its
// smallest id), then the simdgroups fold by butterfly and the simdgroup winners fold once more.  Every fold uses the
// one total order "larger value, then smaller id", so the winner is THE row's maximum with the smallest id whatever
// the thread count: 32 threads and 1024 give the same int.  A NaN never wins (v > best is false), and a row of
// -inf / NaN keeps the sentinel.
kernel void hy4_flash_argmax(device const float* x [[buffer(0)]], device int* out [[buffer(1)]],
                       constant Hy4FlashArgmaxArgs& a [[buffer(2)]], uint3 tgp [[threadgroup_position_in_grid]],
                       ushort tid [[thread_index_in_threadgroup]], ushort lane [[thread_index_in_simdgroup]],
                       ushort sg [[simdgroup_index_in_threadgroup]], uint3 ntg3 [[threads_per_threadgroup]]) {
    const ushort ntg = ushort(ntg3.x);
    device const float* r = x + (ulong)tgp.x * a.stride;
    float best = -INFINITY;
    int bi = 0x7FFFFFFF;
    for (uint i = tid; i < a.n; i += ntg) {
        const float v = r[i];
        if (v > best) {
            best = v;
            bi = int(i);
        }
    }
    for (ushort o = 16; o > 0; o >>= 1) {
        const float ob = simd_shuffle_xor(best, o);
        const int oi = simd_shuffle_xor(bi, o);
        if (ob > best || (ob == best && oi < bi)) {
            best = ob;
            bi = oi;
        }
    }
    threadgroup float tb[32];
    threadgroup int ti[32];
    const ushort nsg = (ntg + 31) / 32;
    if (nsg > 1) {
        if (lane == 0) { tb[sg] = best; ti[sg] = bi; }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (sg == 0) {
            best = lane < nsg ? tb[lane] : -INFINITY;
            bi = lane < nsg ? ti[lane] : 0x7FFFFFFF;
            for (ushort o = 16; o > 0; o >>= 1) {
                const float ob = simd_shuffle_xor(best, o);
                const int oi = simd_shuffle_xor(bi, o);
                if (ob > best || (ob == best && oi < bi)) {
                    best = ob;
                    bi = oi;
                }
            }
        }
    }
    if (tid == 0) out[tgp.x] = bi;
}

// ---- the shape kernels ---------------------------------------------------------------------------------------------
// Strided f32 copy, f32 -> f16 cache store, row sum, bias add and the attention gate.  They carry no arithmetic
// choice of their own except the gate's sigmoid, which is hy4_flash_common.metal's.
struct Hy4FlashElemArgs {
    uint n;
    uint a_stride;
    uint b_stride;
    uint y_stride;
};

// grid (ceil(n / HY4_FLASH_TG) * HY4_FLASH_TG, rows)
kernel void hy4_flash_copy_f32(device const float* x [[buffer(0)]], device float* y [[buffer(1)]],
                         constant Hy4FlashElemArgs& a [[buffer(2)]], uint2 gid [[thread_position_in_grid]]) {
    if (gid.x >= a.n) return;
    y[(ulong)gid.y * a.y_stride + gid.x] = x[(ulong)gid.y * a.a_stride + gid.x];
}

// The f16 KV / indexer cache store.  A non-finite value MAY NOT ENTER THE CACHE.
//
// So a value that is not finite, or whose magnitude f16 cannot hold (|v| > 65504, which `half()` would turn into
// an inf), is COUNTED and **written as zero**.  Writing `half(v)` and only counting it is a silent inf, which is
// forbidden: one inf in the KV cache is read by every later position through the attention mask, so a single bad
// value quietly poisons the whole rest of the sequence while the request runs to completion.  Zero is not a correct
// answer either, and it is not meant to be: the host reads this counter after every window and ends the request
// with `nonfinite`.  Zero only bounds the damage between the bad store and that check.
//
// Note the two are not the same test: `isfinite` rejects inf and nan, `fabs(v) > 65504` rejects the finite f32
// values that f16 cannot represent.  Both must be caught, or an overflow becomes an inf one dispatch later.
kernel void hy4_flash_store_half(device const float* x [[buffer(0)]], device half* y [[buffer(1)]],
                           constant Hy4FlashElemArgs& a [[buffer(2)]], device atomic_uint* bad [[buffer(3)]],
                           uint2 gid [[thread_position_in_grid]]) {
    if (gid.x >= a.n) return;
    const float v = x[(ulong)gid.y * a.a_stride + gid.x];
    const bool ok = isfinite(v) && fabs(v) <= 65504.0f;
    if (!ok) atomic_fetch_add_explicit(bad, 1u, memory_order_relaxed);
    y[(ulong)gid.y * a.y_stride + gid.x] = ok ? half(v) : half(0.0f);
}

// y = x + b, with b a single row broadcast over the rows (engram_kv_b, which the reference adds to the [hc+1, n_embd]
// reshape, so one b row covers the whole token).
kernel void hy4_flash_add_row(device const float* x [[buffer(0)]], device const float* b [[buffer(1)]],
                        device float* y [[buffer(2)]], constant Hy4FlashElemArgs& a [[buffer(3)]],
                        uint2 gid [[thread_position_in_grid]]) {
    if (gid.x >= a.n) return;
    y[(ulong)gid.y * a.y_stride + gid.x] = x[(ulong)gid.y * a.a_stride + gid.x] + b[gid.x];
}

// y = x1 + x2 (the routed-up and the shared expert, in the reference's order: `up + sh`)
kernel void hy4_flash_add_f32(device const float* x1 [[buffer(0)]], device const float* x2 [[buffer(1)]],
                        device float* y [[buffer(2)]], constant Hy4FlashElemArgs& a [[buffer(3)]],
                        uint2 gid [[thread_position_in_grid]]) {
    if (gid.x >= a.n) return;
    y[(ulong)gid.y * a.y_stride + gid.x] =
        x1[(ulong)gid.y * a.a_stride + gid.x] + x2[(ulong)gid.y * a.b_stride + gid.x];
}

// The attention gate: y = kqv * sigmoid(g) (hy4_block.py attention, applied BEFORE attn_output)
kernel void hy4_flash_gate_mul(device const float* x [[buffer(0)]], device const float* g [[buffer(1)]],
                         device float* y [[buffer(2)]], constant Hy4FlashElemArgs& a [[buffer(3)]],
                         uint2 gid [[thread_position_in_grid]]) {
    if (gid.x >= a.n) return;
    y[(ulong)gid.y * a.y_stride + gid.x] =
        x[(ulong)gid.y * a.a_stride + gid.x] * hy4_flash_sigmoid(g[(ulong)gid.y * a.b_stride + gid.x]);
}

// ---- the shared key's tail rotated, its latent normed, both stored into the f16 cache: ONE dispatch that is
// bitwise the four it replaces (hy4_flash_rope_norm on kvp[512:576], hy4_flash_rmsnorm kvp[0:512] -> kvn, hy4_flash_store_half x2):
//   * threads 0 .. n_rot/2 - 1 rotate pair k exactly as hy4_flash_rope_norm's thread k does (the same f32 angle
//     recurrence from the position, precise cos / sin, the same two expressions) and write it back to kvp;
//   * all HY4_FLASH_TG threads norm kvp[0:n] as hy4_flash_rmsnorm does (thread tid takes i = tid, tid + HY4_FLASH_TG, ..., the same
//     hy4_flash_tg_sum tree, precise::rsqrt, `v * scale` then `* w`) into kvn;
//   * the stores are hy4_flash_store_half's: a value that is not finite or does not fit f16 is counted and written as 0.
// The two halves touch disjoint elements of kvp, so doing them in one group changes no input of either.
// grid (n_tok), HY4_FLASH_TG threads.
struct Hy4FlashKvFinishArgs {
    uint n;            // the latent (kv_lora, 512): normed
    uint n_rot;        // the tail (64): rotated
    uint row;          // floats between tokens of kvp (576) and of kvn (n), halves between cache rows (576)
    uint kvn_stride;
    float eps;
    float theta_scale;
    // M-RoPE only (hy4_flash_kv_finish_mrope): `pos` is [3][pos_stride]; mlen1 / mlen2 are
    // 3 * mrope_section[1] and 3 * mrope_section[2].  Zero on the text path.
    uint pos_stride;
    uint mlen1, mlen2;
};
static inline half hy4_flash_store_guard(float v, device atomic_uint* bad) {
    const bool ok = isfinite(v) && fabs(v) <= 65504.0f;
    if (!ok) atomic_fetch_add_explicit(bad, 1u, memory_order_relaxed);
    return ok ? half(v) : half(0.0f);
}
template <bool MROPE>
kernel void hy4_flash_kv_finish_t(device float* kvp [[buffer(0)]], device const float* w [[buffer(1)]],
                          device float* kvn [[buffer(2)]], device half* cache [[buffer(3)]],
                          device const int* pos [[buffer(4)]], device atomic_uint* bad [[buffer(5)]],
                          constant Hy4FlashKvFinishArgs& a [[buffer(6)]], uint3 tgp [[threadgroup_position_in_grid]],
                          ushort tid [[thread_index_in_threadgroup]], ushort sg [[simdgroup_index_in_threadgroup]],
                          ushort lane [[thread_index_in_simdgroup]]) {
    const uint t = tgp.x;
    device float* xr = kvp + (ulong)t * a.row;
    device half* cr = cache + (ulong)t * a.row;
    // the tail (hy4_flash_rope_norm, one pair a thread; the loop STRIDES, so a tail wider than 2 * HY4_FLASH_TG is covered
    // instead of silently truncated.  Each pair rebuilds its own angle from pos[t], so what a pair computes does
    // not depend on the stride or on how many pairs a thread takes.)
    for (uint k = tid; 2 * k < a.n_rot; k += HY4_FLASH_TG) {
        float th = float(MROPE ? pos[hy4_flash_mrope_axis(k, a.mlen1, a.mlen2) * a.pos_stride + t] : pos[t]);
        for (uint j = 0; j < k; ++j) th *= a.theta_scale;
        const float c = precise::cos(th), s = precise::sin(th);
        device float* p = xr + a.n + 2 * k;
        const float x0 = p[0], x1 = p[1];
        const float y0 = x0 * c - x1 * s;
        const float y1 = x0 * s + x1 * c;
        p[0] = y0;
        p[1] = y1;
        cr[a.n + 2 * k] = hy4_flash_store_guard(y0, bad);
        cr[a.n + 2 * k + 1] = hy4_flash_store_guard(y1, bad);
    }
    // the latent (hy4_flash_rmsnorm with the weight)
    threadgroup float part[HY4_FLASH_TG_SG];
    float s2 = 0.0f;
    for (uint i = tid; i < a.n; i += HY4_FLASH_TG) {
        const float v = xr[i];
        s2 += v * v;
    }
    const float scale = precise::rsqrt(hy4_flash_tg_sum(s2, part, sg, lane) / float(a.n) + a.eps);
    device float* yr = kvn + (ulong)t * a.kvn_stride;
    for (uint i = tid; i < a.n; i += HY4_FLASH_TG) {
        const float v = xr[i] * scale;
        const float y = v * w[i];
        yr[i] = y;
        cr[i] = hy4_flash_store_guard(y, bad);
    }
}

template [[host_name("hy4_flash_kv_finish")]] kernel decltype(hy4_flash_kv_finish_t<false>) hy4_flash_kv_finish_t<false>;
template [[host_name("hy4_flash_kv_finish_mrope")]] kernel decltype(hy4_flash_kv_finish_t<true>) hy4_flash_kv_finish_t<true>;
