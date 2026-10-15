// metal/kernels/qsa_prompt.metal - QSA attention for a PROMPT chunk on the matrix units (after qsa.metal: QH, QKV, QD,
// QBUDGET, QCH, QsaArgs).  The same split-K partials as qsa_attn (part [t][chunk][QH][QD + 2]: acc, then m, l), so
// qsa_merge is unchanged; per (chunk of QCH = 64 selected cells, kv head g, token t) one simdgroup:
//   S[12 x 64] = Qh[12 x 256] . K^T / 16      single-simdgroup matmul2d, K (int8 * fp16 scale per 64) decoded into
//                                             the right-input cooperative tensor, 32 dims a step
//   per head: m = max, p = exp(s - m), l = sum p (fp32, through threadgroup memory), P -> fp16
//   acc[12 x 256] = P[12 x 64] . V[64 x 256]  four 64-dim matmuls, V decoded into the right input
// The operands are fp16 (q rounded from fp32; k, v = int8 * scale rounded): to rounding of qsa_attn's fp32.
//
// QSA_KV_Q4 (function constant 25, qsa.metal's q4 mode; this file compiles with kv_q4.metal and qsa.metal in
// front, so the prelude and Q4_ROW are in scope): the q4 pools [cell][kv head][Q4_ROW] decode straight into the
// right inputs as half(float(nibble - 8) * d), d inside each 18-byte block - under the constant the scale caches
// stay unread (the host binds a placeholder).  Both cooperative layouts decode their own way: !k4 one element a
// thread (block k >> 5, element (k & 31) the low nibble of byte (k & 15) below 16, the high above); k4 one 4-run -
// its k .. k + 3 (k % 4 = 0) sit in one block, four consecutive bytes on one nibble side.  qsa_attn_mm's partials
// stay in the rotated domain: qsa_merge rotates the merged row back under the same constant, so nothing else
// moves.  qsa_flash has no merge: its epilogue rotates the accumulated row back FIRST (orthonormal,
// self-inverse - the 1/16 scale), then divides by l and applies sigmoid(gate), which does not commute with the
// rotation; the row's 256 dims span four simdgroups, so the butterflies cross them through sp as the staging
// (dead since the chunk loop ended).
// Cross-platform note, as qsa.metal:35-38 states for decode: the PC's prompt kernels apply the q4 scale in fp32
// (prefill's mode 4); this file rounds the decoded operand to fp16, being a matmul2d right input - the same
// half-rounding the int8 arm above has always had against qsa_attn's fp32.
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace mpp::tensor_ops;
#ifndef STRATA_COOP_K4
#define STRATA_COOP_K4
// function constant 7: the right-input cooperative tensor holds runs of 4 consecutive k at one n per lane (the M5's
// layout; tests/coop_layout).  false (the M4: runs of 2 n at one k): every element decoded on its own.
constant bool COOP_K4 [[function_constant(7)]];
constant bool coop_k4 = is_function_constant_defined(COOP_K4) ? COOP_K4 : true;
#endif

// a bisection probe (function constant 8): stop at stage n and write m = n (1: after the ids, 2: after S, 3: after
// the softmax); 0 = normal
constant int QSA_STOP [[function_constant(8)]];
constant int qsa_stop = is_function_constant_defined(QSA_STOP) ? QSA_STOP : 0;

// grid (chunks, QKV, n_tok), 32 threads.  qh: the q heads as fp16 [t][QH][QD] (qcur rounded).
kernel void qsa_attn_mm(device const half* qh [[buffer(0)]], device const char* kq [[buffer(1)]],
                        device const char* vq [[buffer(2)]], device const half* ksc [[buffer(3)]],
                        device const half* vsc [[buffer(4)]], device const int* ids [[buffer(5)]],
                        device float* part [[buffer(6)]], constant QsaArgs& a [[buffer(7)]],
                        uint3 tg [[threadgroup_position_in_grid]], ushort lane [[thread_index_in_simdgroup]]) {
    constexpr int HPG = QH / QKV, NCH = (QBUDGET + QCH - 1) / QCH;
    threadgroup float s[16 * QCH];
    threadgroup half p[16 * QCH];
    threadgroup int cid[QCH];
    threadgroup half vs[QCH][4];
    const int chunk = int(tg.x), g = int(tg.y), t = int(tg.z);
    const uint n_kv = a.pos0 + uint(t) + 1, width = min(n_kv, uint(QBUDGET));
    const int c0 = chunk * QCH;
    device float* pp = part + ((uint(t) * NCH + chunk) * QH + g * HPG) * (QD + 2);
    if (uint(c0) >= width) {
        if (lane < HPG) { pp[lane * (QD + 2) + QD] = -INFINITY; pp[lane * (QD + 2) + QD + 1] = 0.0f; }
        return;
    }
    const int nc = min(QCH, int(width) - c0);
    for (int c = lane; c < QCH; c += 32) {
        const int id = c < nc ? ids[t * QBUDGET + c0 + c] : -1;
        cid[c] = id;
        // q4 has no scale pool (d rides inside each 18-byte block): the host binds a placeholder, never read
        if (!qsa_kv_q4)
            for (int j = 0; j < 4; ++j) vs[c][j] = id >= 0 ? vsc[(uint(id) * QKV + g) * 4 + j] : half(0.0f);
    }
    simdgroup_barrier(mem_flags::mem_threadgroup);
    if (qsa_stop == 1) { if (lane < HPG) { pp[lane * (QD + 2) + QD] = 1.0f; pp[lane * (QD + 2) + QD + 1] = 1.0f; } return; }
    // ---- S = Q K^T
    {
        constexpr auto desc = matmul2d_descriptor(16, QCH, 32, false, true, false, matmul2d_descriptor::mode::multiply_accumulate);
        matmul2d<desc, execution_simdgroup> op;
        tensor<device half, dextents<int32_t, 2>, tensor_inline> tQ((device half*)qh + (uint(t) * QH + g * HPG) * QD,
                                                                   dextents<int32_t, 2>(QD, HPG));
        auto rk = op.template get_right_input_cooperative_tensor<half, half, float>();
        auto tA = tQ.template slice<32, 16>(0, 0);
        auto acc = op.template get_destination_cooperative_tensor<__tensor_ops_detail::__remove_addrspace_t<decltype(tA)>, __tensor_ops_detail::__remove_addrspace_t<decltype(rk)>, float>();
        for (uint16_t i = 0; i < acc.get_capacity(); ++i)
            if (acc.is_valid_element(i)) acc[i] = 0.0f;
        for (int k0 = 0; k0 < QD; k0 += 32) {
            if (!coop_k4) {
                for (uint16_t i = 0; i < rk.get_capacity(); ++i) {
                    const auto ix = rk.get_multidimensional_index(i);
                    const int k = k0 + ix[0], id = cid[ix[1]];
                    const uint row = uint(id) * QKV + g;
                    if (qsa_kv_q4) {
                        // block (k >> 5) of the Q4_ROW-byte row; element (k & 31) is the low nibble of byte
                        // (k & 15) below 16, the high above - qsa_attn's decode, the fp32 product rounded to half
                        half w = half(0.0f);
                        if (id >= 0) {
                            const device kv_q4_blk* blk =
                                (const device kv_q4_blk*)((device const uchar*)kq + (size_t)row * Q4_ROW) + (k >> 5);
                            const uchar qb = blk->qs[k & 15];
                            w = half(float(((k & 31) < 16 ? (qb & 0x0F) : (qb >> 4)) - 8) * f32_from_f16(blk->d));
                        }
                        rk[i] = w;
                    } else {
                        rk[i] = id >= 0 ? half(float(kq[row * QD + k]) * float(ksc[row * 4 + (k >> 6)])) : half(0.0f);
                    }
                }
            } else
#pragma clang loop unroll(full)
            for (uint16_t i = 0; i < rk.get_capacity(); i += 4) {
                const auto ix = rk.get_multidimensional_index(i);
                const int k = k0 + ix[0], c = ix[1], id = cid[c];
                half4 w = half4(0.0h);
                if (id >= 0) {
                    const uint row = uint(id) * QKV + g;
                    if (qsa_kv_q4) {
                        // a run's k .. k + 3 (k % 4 = 0) sit in one block, four consecutive bytes on one nibble
                        // side - one shift and mask for the whole run, the fp32 products rounded to half
                        const device kv_q4_blk* blk =
                            (const device kv_q4_blk*)((device const uchar*)kq + (size_t)row * Q4_ROW) + (k >> 5);
                        const device uchar* qs = blk->qs + (k & 15);
                        const uint4 nib = (uint4(qs[0], qs[1], qs[2], qs[3]) >> uint4((k & 31) < 16 ? 0u : 4u)) & uint4(0x0Fu);
                        w = half4((float4(nib) - 8.0f) * f32_from_f16(blk->d));
                    } else {
                        const char4 q4 = *(device const char4*)(kq + row * QD + k);
                        w = half4(float4(q4) * float(ksc[row * 4 + (k >> 6)]));
                    }
                }
                rk[i] = w.x; rk[i + 1] = w.y; rk[i + 2] = w.z; rk[i + 3] = w.w;
            }
            tA = tQ.template slice<32, 16>(k0, 0);
            op.run(tA, rk, acc);
        }
        tensor<threadgroup float, dextents<int32_t, 2>, tensor_inline> tS(s, dextents<int32_t, 2>(QCH, 16));
        acc.store(tS);
    }
    simdgroup_barrier(mem_flags::mem_threadgroup);
    if (qsa_stop == 2) { if (lane < HPG) { pp[lane * (QD + 2) + QD] = 2.0f; pp[lane * (QD + 2) + QD + 1] = 1.0f; } return; }
    // ---- softmax pieces per head (lane = head for the reductions; 12 heads)
    float mh = -INFINITY, lh = 0.0f;
    if (lane < HPG) {
        for (int c = 0; c < nc; ++c) mh = max(mh, s[lane * QCH + c] * 0.0625f);
        for (int c = 0; c < QCH; ++c) {
            const float e = c < nc ? exp(s[lane * QCH + c] * 0.0625f - mh) : 0.0f;
            lh += e;
            p[lane * QCH + c] = half(e);
        }
    } else if (lane < 16) {
        for (int c = 0; c < QCH; ++c) p[lane * QCH + c] = half(0.0f);
    }
    simdgroup_barrier(mem_flags::mem_threadgroup);
    if (qsa_stop == 3) { if (lane < HPG) { pp[lane * (QD + 2) + QD] = 3.0f; pp[lane * (QD + 2) + QD + 1] = 1.0f; } return; }
    // ---- acc = P V, 64 dims at a time
    {
        // K = 32 cells a run (two runs a 64-cell chunk): K = 64 runs compute garbage on the M4
        constexpr auto desc = matmul2d_descriptor(16, 64, 32, false, true, false, matmul2d_descriptor::mode::multiply_accumulate);
        matmul2d<desc, execution_simdgroup> op;
        tensor<threadgroup half, dextents<int32_t, 2>, tensor_inline> tP(p, dextents<int32_t, 2>(QCH, 16));
        tensor<device float, dextents<int32_t, 2>, tensor_inline> tO(pp, dextents<int32_t, 2>(QD + 2, HPG));
        for (int d0 = 0; d0 < QD; d0 += 64) {
            auto rv = op.template get_right_input_cooperative_tensor<half, half, float>();
            auto tPh = tP.template slice<32, 16>(0, 0);
            auto acc = op.template get_destination_cooperative_tensor<__tensor_ops_detail::__remove_addrspace_t<decltype(tPh)>, __tensor_ops_detail::__remove_addrspace_t<decltype(rv)>, float>();
            for (uint16_t i = 0; i < acc.get_capacity(); ++i)
                if (acc.is_valid_element(i)) acc[i] = 0.0f;
            for (int c0 = 0; c0 < QCH; c0 += 32) {
#pragma clang loop unroll(full)
                for (uint16_t i = 0; i < rv.get_capacity(); ++i) {
                    const auto ix = rv.get_multidimensional_index(i);   // (cell, dim)
                    const int c = c0 + ix[0], d = d0 + ix[1], id = cid[c];
                    if (qsa_kv_q4) {
                        // the V rows decode like the K ones (d rides in the block; the vs scale cache is not read)
                        half w = half(0.0f);
                        if (id >= 0) {
                            const device kv_q4_blk* blk =
                                (const device kv_q4_blk*)((device const uchar*)vq + (size_t)(uint(id) * QKV + g) * Q4_ROW) + (d >> 5);
                            const uchar qb = blk->qs[d & 15];
                            w = half(float(((d & 31) < 16 ? (qb & 0x0F) : (qb >> 4)) - 8) * f32_from_f16(blk->d));
                        }
                        rv[i] = w;
                    } else {
                        rv[i] = id >= 0 ? half(float(vq[(uint(id) * QKV + g) * QD + d]) * float(vs[c][d >> 6])) : half(0.0f);
                    }
                }
                tPh = tP.template slice<32, 16>(c0, 0);
                op.run(tPh, rv, acc);
            }
            if (qsa_stop == 4) {   // the bisection: the old direct store into the 258-float rows
                auto mO = tO.slice(d0, 0);
                acc.store(mO);
                continue;
            }
            // through threadgroup memory (s is free now): a matmul2d store into rows of QD + 2 floats loses the
            // whole threadgroup on the M4
            tensor<threadgroup float, dextents<int32_t, 2>, tensor_inline> tT(s, dextents<int32_t, 2>(64, 16));
            simdgroup_barrier(mem_flags::mem_threadgroup);
            acc.store(tT);
            simdgroup_barrier(mem_flags::mem_threadgroup);
            for (int it = lane; it < HPG * 64; it += 32) pp[(it >> 6) * (QD + 2) + d0 + (it & 63)] = s[it];
        }
    }
    if (lane < HPG) { pp[lane * (QD + 2) + QD] = mh; pp[lane * (QD + 2) + QD + 1] = lh; }
}

// qcur (fp32) -> fp16 for qsa_attn_mm
kernel void qsa_q_f16(device const float* q [[buffer(0)]], device half* qh [[buffer(1)]], constant uint& n [[buffer(2)]],
                      uint gid [[thread_position_in_grid]]) {
    if (gid < n) qh[gid] = half(q[gid]);
}

// qsa_score as a GEMM (fp32 operands, so the selection sees qsa_score's values to fp32 rounding): S[(t, h)][b] =
// qidx[t][h] . pooled[b] for 16 tokens x IH heads (64 rows) by 64 blocks a threadgroup; the epilogue sums relu over
// the heads.  The tail block n_bid (the `dead` key, + 1e9 when it has cells) and blocks past it are left to the
// scalar path: the tail is scored here by one simdgroup a token; blocks > n_bid are not written (qsa_topk reads
// b <= n_bid).  grid (ceil(max blocks / 64), ceil(T / 16)), 128 threads.
kernel void qsa_score_mm(device const float* qidx [[buffer(0)]], device const float* pooled [[buffer(1)]],
                         device const float* dead [[buffer(2)]], device float* scores [[buffer(3)]],
                         constant QsaArgs& a [[buffer(4)]],
                         uint3 tg [[threadgroup_position_in_grid]], ushort tid [[thread_index_in_threadgroup]],
                         ushort sg [[simdgroup_index_in_threadgroup]], ushort lane [[thread_index_in_simdgroup]]) {
    threadgroup float ot[64 * 64];
    const int b0 = int(tg.x) * 64, t0 = int(tg.y) * 16;
    const uint nb_all = (a.pos0 + a.n_tok) / 4 + 1;   // blocks any token of the chunk can score
    tensor<device float, dextents<int32_t, 2>, tensor_inline> tQ((device float*)qidx, dextents<int32_t, 2>(ID, int(a.n_tok) * IH));
    tensor<device float, dextents<int32_t, 2>, tensor_inline> tK((device float*)pooled, dextents<int32_t, 2>(ID, int(nb_all)));
    tensor<threadgroup float, dextents<int32_t, 2>, tensor_inline> tO(ot, dextents<int32_t, 2>(64, 64));
    constexpr auto desc = matmul2d_descriptor(64, 64, dynamic_length_v<int>, false, true, false);
    matmul2d<desc, execution_simdgroups<4>> op;
    auto mA = tQ.slice(0, t0 * IH);
    auto mB = tK.slice(0, b0);
    op.run(mA, mB, tO);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const uint stride = a.max_cells / 4 + 2;
    for (int idx = tid; idx < 16 * 64; idx += 128) {
        const int tt = idx >> 6, j = idx & 63, t = t0 + tt, b = b0 + j;
        if (t >= int(a.n_tok)) continue;
        const uint n_kv = a.pos0 + uint(t) + 1, n_bid = n_kv >> 2;
        if (uint(b) >= n_bid) continue;
        float s = 0.0f;
        for (int h = 0; h < IH; ++h) s += max(ot[(tt * IH + h) * 64 + j], 0.0f);
        scores[uint(t) * stride + b] = s;
    }
    // the tail block of each token in this threadgroup's block range (the first column threadgroup does it)
    if (tg.x == 0)
        for (int tt = sg; tt < 16; tt += 4) {
            const int t = t0 + tt;
            if (t >= int(a.n_tok)) break;
            const uint n_kv = a.pos0 + uint(t) + 1, n_bid = n_kv >> 2;
            float k[4];
            for (int i = 0; i < 4; ++i) k[i] = dead[lane + 32 * i];
            float s = 0.0f;
            for (int h = 0; h < IH; ++h) {
                float d = 0.0f;
                for (int i = 0; i < 4; ++i) d = fma(qidx[(t * IH + h) * ID + lane + 32 * i], k[i], d);
                d = simd_sum(d);
                s += max(d, 0.0f);
            }
            if ((n_kv & 3) != 0) s += 1e9f;
            if (lane == 0) scores[uint(t) * stride + n_bid] = s;
        }
}

// qsa_flash: the whole attention of a (kv head g, token t) in one threadgroup (no split-K partials, no merge): the
// width selected cells in chunks of 64 with an online softmax; 4 simdgroups, simdgroup s owns dims 64 s .. 64 s + 63
// (its slice of S's K sum and its 64 output dims).  Out: xq = (acc / l) * sigmoid(gate) as fp16 (attn_output's GEMM
// operand).  grid (QKV, n_tok), 128 threads.
kernel void qsa_flash(device const half* qh [[buffer(0)]], device const char* kq [[buffer(1)]],
                      device const char* vq [[buffer(2)]], device const half* ksc [[buffer(3)]],
                      device const half* vsc [[buffer(4)]], device const int* ids [[buffer(5)]],
                      device const float* qfull [[buffer(6)]], device half* xqh [[buffer(7)]], constant QsaArgs& a [[buffer(8)]],
                      uint3 tg [[threadgroup_position_in_grid]], ushort tid [[thread_index_in_threadgroup]],
                      ushort sg [[simdgroup_index_in_threadgroup]], ushort lane [[thread_index_in_simdgroup]]) {
    constexpr int HPG = QH / QKV;
    threadgroup float sp[4][16 * QCH];
    threadgroup half p[16 * QCH];
    threadgroup int cid[QCH];
    threadgroup float vs[QCH];
    threadgroup float fac[16], lsum[16], mrun[16];
    const int g = int(tg.x), t = int(tg.y);
    const uint n_kv = a.pos0 + uint(t) + 1, width = min(n_kv, uint(QBUDGET));
    const int nch = int((width + QCH - 1) / QCH);
    if (tid < 16) { mrun[tid] = -INFINITY; lsum[tid] = 0.0f; }
    constexpr auto dS = matmul2d_descriptor(16, QCH, 32, false, true, false, matmul2d_descriptor::mode::multiply_accumulate);
    constexpr auto dO = matmul2d_descriptor(16, 64, 32, false, true, false, matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<dS, execution_simdgroup> opS;
    matmul2d<dO, execution_simdgroup> opO;
    tensor<device half, dextents<int32_t, 2>, tensor_inline> tQ((device half*)qh + (uint(t) * QH + g * HPG) * QD,
                                                               dextents<int32_t, 2>(QD, HPG));
    tensor<threadgroup half, dextents<int32_t, 2>, tensor_inline> tP(p, dextents<int32_t, 2>(QCH, 16));
    auto rv = opO.template get_right_input_cooperative_tensor<half, half, float>();
    auto tPh = tP.template slice<32, 16>(0, 0);
    auto acc = opO.template get_destination_cooperative_tensor<__tensor_ops_detail::__remove_addrspace_t<decltype(tPh)>,
                                                               __tensor_ops_detail::__remove_addrspace_t<decltype(rv)>, float>();
    // (a second destination tensor, not a copy of acc: the MetalPerformancePrimitives headers of macOS 26.3 cannot copy a
    // cooperative tensor - "cannot initialize a parameter of type 'void *' with an rvalue of type 'const void *'", #2323;
    // tot is zeroed right below, so the copy was only there for its type)
    auto tot = opO.template get_destination_cooperative_tensor<__tensor_ops_detail::__remove_addrspace_t<decltype(tPh)>,
                                                               __tensor_ops_detail::__remove_addrspace_t<decltype(rv)>, float>();   // the running output; acc takes one chunk (fresh each chunk: on the M4 a run does not accumulate
                      // into elements rescaled in place)
    for (uint16_t i = 0; i < acc.get_capacity(); ++i)
        if (acc.is_valid_element(i)) tot[i] = 0.0f;
    const int d0 = sg * 64;
    for (int ch = 0; ch < nch; ++ch) {
        const int c0 = ch * QCH, nc = min(QCH, int(width) - c0);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid < QCH) {
            const int id = tid < nc ? ids[t * QBUDGET + c0 + tid] : -1;
            cid[tid] = id;
            // q4 has no scale pool (d rides inside each 18-byte block): the host binds a placeholder, never read
            if (!qsa_kv_q4)
                vs[tid] = id >= 0 ? float(vsc[(uint(id) * QKV + g) * 4 + sg * 0]) : 0.0f;   // (written but never read: the V arm below reads vsc directly)
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        // S partial over this simdgroup's 64 dims
        {
            auto rk = opS.template get_right_input_cooperative_tensor<half, half, float>();
            auto tA = tQ.template slice<32, 16>(d0, 0);
            auto s = opS.template get_destination_cooperative_tensor<__tensor_ops_detail::__remove_addrspace_t<decltype(tA)>,
                                                                     __tensor_ops_detail::__remove_addrspace_t<decltype(rk)>, float>();
            for (uint16_t i = 0; i < s.get_capacity(); ++i)
                if (s.is_valid_element(i)) s[i] = 0.0f;
            for (int k0 = d0; k0 < d0 + 64; k0 += 32) {
                if (!coop_k4) {
                    for (uint16_t i = 0; i < rk.get_capacity(); ++i) {
                        const auto ix = rk.get_multidimensional_index(i);
                        const int k = k0 + ix[0], id = cid[ix[1]];
                        const uint row = uint(id) * QKV + g;
                        if (qsa_kv_q4) {
                            // block (k >> 5) of the Q4_ROW-byte row; element (k & 31) is the low nibble of byte
                            // (k & 15) below 16, the high above - qsa_attn's decode, the fp32 product to half
                            half w = half(0.0f);
                            if (id >= 0) {
                                const device kv_q4_blk* blk =
                                    (const device kv_q4_blk*)((device const uchar*)kq + (size_t)row * Q4_ROW) + (k >> 5);
                                const uchar qb = blk->qs[k & 15];
                                w = half(float(((k & 31) < 16 ? (qb & 0x0F) : (qb >> 4)) - 8) * f32_from_f16(blk->d));
                            }
                            rk[i] = w;
                        } else {
                            rk[i] = id >= 0 ? half(float(kq[row * QD + k]) * float(ksc[row * 4 + (k >> 6)])) : half(0.0f);
                        }
                    }
                } else
#pragma clang loop unroll(full)
                for (uint16_t i = 0; i < rk.get_capacity(); i += 4) {
                    const auto ix = rk.get_multidimensional_index(i);
                    const int k = k0 + ix[0], c = ix[1], id = cid[c];
                    half4 w = half4(0.0h);
                    if (id >= 0) {
                        const uint row = uint(id) * QKV + g;
                        if (qsa_kv_q4) {
                            // a run's k .. k + 3 (k % 4 = 0) sit in one block, four consecutive bytes on one
                            // nibble side - one shift and mask for the whole run, the fp32 products to half
                            const device kv_q4_blk* blk =
                                (const device kv_q4_blk*)((device const uchar*)kq + (size_t)row * Q4_ROW) + (k >> 5);
                            const device uchar* qs = blk->qs + (k & 15);
                            const uint4 nib = (uint4(qs[0], qs[1], qs[2], qs[3]) >> uint4((k & 31) < 16 ? 0u : 4u)) & uint4(0x0Fu);
                            w = half4((float4(nib) - 8.0f) * f32_from_f16(blk->d));
                        } else {
                            const char4 q4 = *(device const char4*)(kq + row * QD + k);
                            w = half4(float4(q4) * float(ksc[row * 4 + (k >> 6)]));
                        }
                    }
                    rk[i] = w.x; rk[i + 1] = w.y; rk[i + 2] = w.z; rk[i + 3] = w.w;
                }
                tA = tQ.template slice<32, 16>(k0, 0);
                opS.run(tA, rk, s);
            }
            tensor<threadgroup float, dextents<int32_t, 2>, tensor_inline> tS(sp[sg], dextents<int32_t, 2>(QCH, 16));
            s.store(tS);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        // online softmax per head row: simdgroup 0, lane = row
        if (sg == 0 && lane < 16) {
            if (lane < HPG) {
                float m = mrun[lane];
                for (int c = 0; c < nc; ++c) {
                    const int q = lane * QCH + c;
                    m = max(m, (((sp[0][q] + sp[1][q]) + (sp[2][q] + sp[3][q])) * 0.0625f));
                }
                const float f = exp(mrun[lane] - m);
                float l = lsum[lane] * f;
                for (int c = 0; c < QCH; ++c) {
                    const int q = lane * QCH + c;
                    const float e = c < nc ? exp(((sp[0][q] + sp[1][q]) + (sp[2][q] + sp[3][q])) * 0.0625f - m) : 0.0f;
                    l += e;
                    p[q] = half(e);
                }
                mrun[lane] = m;
                lsum[lane] = l;
                fac[lane] = f;
            } else {
                for (int c = 0; c < QCH; ++c) p[lane * QCH + c] = half(0.0f);
                fac[lane] = 0.0f;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        // rescale this simdgroup's accumulator rows, then acc += P V over its 64 dims
        for (uint16_t i = 0; i < acc.get_capacity(); ++i)
            if (acc.is_valid_element(i)) acc[i] = 0.0f;
        for (int c0 = 0; c0 < QCH; c0 += 32) {   // K = 32 cells a run (K = 64 is wrong on the M4)
#pragma clang loop unroll(full)
            for (uint16_t i = 0; i < rv.get_capacity(); ++i) {
                const auto ix = rv.get_multidimensional_index(i);   // (cell, dim)
                const int c = c0 + ix[0], d = d0 + ix[1], id = cid[c];
                if (qsa_kv_q4) {
                    // the V rows decode like the K ones; d rides in the block, so vsc (a placeholder) is not read
                    half w = half(0.0f);
                    if (id >= 0) {
                        const device kv_q4_blk* blk =
                            (const device kv_q4_blk*)((device const uchar*)vq + (size_t)(uint(id) * QKV + g) * Q4_ROW) + (d >> 5);
                        const uchar qb = blk->qs[d & 15];
                        w = half(float(((d & 31) < 16 ? (qb & 0x0F) : (qb >> 4)) - 8) * f32_from_f16(blk->d));
                    }
                    rv[i] = w;
                } else {
                    rv[i] = id >= 0 ? half(float(vq[(uint(id) * QKV + g) * QD + d]) * float(vsc[(uint(id) * QKV + g) * 4 + sg])) : half(0.0f);
                }
            }
            tPh = tP.template slice<32, 16>(c0, 0);
            opO.run(tPh, rv, acc);
        }
        for (uint16_t i = 0; i < acc.get_capacity(); ++i)
            if (acc.is_valid_element(i)) tot[i] = fma(tot[i], fac[acc.get_multidimensional_index(i)[1]], acc[i]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (qsa_kv_q4) {
        // the V pool is rotated, so tot is R(o): rotate back FIRST (the orthonormal FWHT is its own inverse),
        // then / lsum, then the gate - qsa_merge's placement (the elementwise gate does not commute with it).
        // A row's 256 dims span four simdgroups: sp is dead since the chunk loop ended and becomes the staging.
        threadgroup float* rot = &sp[0][0];
        for (uint16_t i = 0; i < acc.get_capacity(); ++i)
            if (acc.is_valid_element(i)) {
                const auto ix = acc.get_multidimensional_index(i);   // (dim, row)
                if (ix[1] < HPG) rot[ix[1] * 256 + d0 + ix[0]] = tot[i];
            }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        // one row over the 128 threads, two elements each (tid and tid + 128): the scale, the stage order and
        // the butterfly arithmetic are fwht_row256's - the low five index bits inside each simdgroup (both
        // elements' partners share the xor offset), 32 / 64 across the simdgroups through the row, and 128
        // inside the thread, whose two elements are the pair
        for (int h = 0; h < HPG; ++h) {
            threadgroup float* row = rot + h * 256;
            float x0 = row[tid] * (1.0f / 16.0f), x1 = row[tid + 128] * (1.0f / 16.0f);
            for (ushort b = 1; b < 32; b *= 2) {
                const bool low = (lane & b) == 0;
                const float p0 = simd_shuffle_xor(x0, b), p1 = simd_shuffle_xor(x1, b);
                x0 = low ? x0 + p0 : p0 - x0;
                x1 = low ? x1 + p1 : p1 - x1;
            }
            for (ushort s = 1; s < 4; s *= 2) {   // index bit 32 s: partner simdgroup sg ^ s, same lane, same half
                row[tid] = x0; row[tid + 128] = x1;
                threadgroup_barrier(mem_flags::mem_threadgroup);   // the pre-step values staged first...
                const float p0 = row[(sg ^ s) * 32 + lane], p1 = row[(sg ^ s) * 32 + lane + 128];
                threadgroup_barrier(mem_flags::mem_threadgroup);   // ...every read done before the rewrite
                const bool low = (sg & s) == 0;
                x0 = low ? x0 + p0 : p0 - x0;
                x1 = low ? x1 + p1 : p1 - x1;
            }
            row[tid] = x0 + x1;   // index bit 128: the thread's own two elements are the pair
            row[tid + 128] = x0 - x1;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint16_t i = 0; i < acc.get_capacity(); ++i)
            if (acc.is_valid_element(i)) {
                const auto ix = acc.get_multidimensional_index(i);   // (dim, row)
                const int h = ix[1], d = d0 + ix[0];
                if (h < HPG) {
                    const int hh = g * HPG + h;
                    const float gate = qfull[(uint(t) * QH + hh) * 2 * QD + QD + d];
                    xqh[(uint(t) * QH + hh) * QD + d] = half(rot[h * 256 + d] / lsum[h] * (1.0f / (1.0f + exp(-gate))));
                }
            }
    } else
        for (uint16_t i = 0; i < acc.get_capacity(); ++i)
            if (acc.is_valid_element(i)) {
                const auto ix = acc.get_multidimensional_index(i);   // (dim, row)
                const int h = ix[1], d = d0 + ix[0];
                if (h < HPG) {
                    const int hh = g * HPG + h;
                    const float gate = qfull[(uint(t) * QH + hh) * 2 * QD + QD + d];
                    xqh[(uint(t) * QH + hh) * QD + d] = half(tot[i] / lsum[h] * (1.0f / (1.0f + exp(-gate))));
                }
            }
}
