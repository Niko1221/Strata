// engine/metal/src/hy4flash_gguf/msl/hy4_flash_indexer.metal - the compressed indexer of a hy4-flash sparse block (indexer.hpp:11-67,
// semantics.md §5).
//
// Four kernels here; the other two nodes of the chain are compositions of batch 1's kernels and are NOT rewritten:
//
//   indexer_k   = hy4_flash_rmsnorm(gemv(indexer.attn_k, cur), eps, k_norm)                        [128]
//   indexer_q   = hy4_flash_rope_norm(gemv(indexer.attn_q_b, q_resid), pos)                        [16 x 128], tail 64
//   hy4_flash_indexer_weights   wgt = (gemv(indexer.proj, cur)) / sqrt(key_dim * n_head)           [16]
//   hy4_flash_indexer_kpool     one pooled key per complete group of kpool positions               [128]
//   hy4_flash_indexer_score     score[t][g] = sum_h relu(iq[t][h] . pooled[g]) * wgt[t][h]         [n_pool]
//   hy4_flash_indexer_topk      the attended cell indices of each query, OFFICIAL semantics
//
// THE SELECTION IS THE OFFICIAL ONE, not llama.cpp's (the two official implementations agree, and llama's
// n_sel = 2051 exceeds the top_k it declares).  For a query at position p:
//
//   nv  = p / kpool                     pools that CONTEST - the query's own group is excluded, complete or not
//   cap = top_k / kpool - 1 = 511       the budget, one group short because the tail spends it
//   tail = cells kpool*(p/kpool) .. p   the query's own group, FORCE-RECALLED (kpool slots, p%kpool+1 valid)
//
// so n_sel = 511*4 + p%4 + 1, never above top_k, and below the budget (nv <= cap) the selection is the IDENTITY on
// the causal set.  Ties go to the SMALLER pool index.  tmp/tests/hy4_flash/pyref/test_indexer_select.py pins all of this
// on the numpy side and is the oracle these kernels are tested against.
//
// T INVARIANCE.  Every kernel here is one thread group per output row (a group, or a (token, pool), or a query), so
// no reduction and no selection can see how many tokens the dispatch carries.  The pooled key is rotated to its
// group's FIRST position and the score's visibility test is `kpool*g + kpool - 1 <= pos[t]`, both of which are
// functions of absolute positions only - never of the window.

#define HY4_FLASH_KPOOL_MAX 8     // members of a pool group (4 here)
#define HY4_FLASH_IDX_MAXH 64     // indexer heads (16 here)

// ---- the weights' scale --------------------------------------------------------------------------------------------
// wgt = raw / sqrt(key_dim * n_head).  Its own kernel because folding the divide into the score would change the
// arithmetic: relu(dot) * (raw / s) is not relu(dot) * raw / s in f32.  `denom` is computed on the host.
struct Hy4FlashIdxWeightArgs {
    uint n;          // values in a row (n_head)
    uint x_stride;
    uint y_stride;
    float denom;     // sqrt(key_dim * n_head), as f32
};

// grid (ceil(n / HY4_FLASH_TG) * HY4_FLASH_TG, n_tok)
kernel void hy4_flash_indexer_weights(device const float* x [[buffer(0)]], device float* y [[buffer(1)]],
                                constant Hy4FlashIdxWeightArgs& a [[buffer(2)]], uint2 gid [[thread_position_in_grid]]) {
    if (gid.x >= a.n) return;
    y[(ulong)gid.y * a.y_stride + gid.x] = x[(ulong)gid.y * a.x_stride + gid.x] / a.denom;
}

// ---- the kpool compressor ------------------------------------------------------------------------------------------
// For every complete group g of kpool positions (hy-v4-flash.cpp:824-832, indexer.hpp:26-33):
//
//   p[m][d] = softmax over the kpool MEMBERS, PER DIMENSION, of (ig[first+m][d] + ape[m][d])
//   v[d]    = sum_m p[m][d] * iv[first+m][d]                   added in member order
//   pooled  = rope( rms_norm(v, compressor_norm), pos = first )    pos is the group's FIRST position
//
// The softmax is over the four members and not over the 128 dimensions - `ape` is the position-in-group embedding,
// so the members are weighted by where they sit in the group.  Getting that axis wrong is the easy mistake.
//
// iv / ig come from the indexer cache, where they are f16 (a cache row is value | gate | pooled, 3 x 128 halves),
// so they are read as half and widened.  pooled is written twice: f32 for the parity comparison and for a score in
// the same dispatch, half into the cache.  The f32 copy is 128 floats a group - nothing next to the cache.
struct Hy4FlashKpoolArgs {
    uint dim;          // key_length (128)
    uint kpool;        // positions per group (4)
    uint g0;           // absolute index of the first group in this dispatch
    uint n_group;      // groups in this dispatch
    uint iv_stride;    // halves between cache rows of value
    uint ig_stride;    // halves between cache rows of gate
    uint ape_stride;   // floats between members of ape
    uint out_stride;   // floats between groups of the f32 output
    uint out16_stride; // halves between groups of the cache output
    uint n_rot;        // rotated dims at the end of a pooled key (64)
    float eps;
    float theta_scale; // powf(indexer base, -2 / n_rot), as ggml builds it
};

// Test hook: 1 = the pooled key is rotated to its group's LAST position instead of its FIRST.  That is the
// `pool_rope_last` mutation of the parity gate; production never sets it.
constant int HY4_FLASH_KPOOL_PROBE [[function_constant(53)]];
constant int hy4_flash_kpool_probe = is_function_constant_defined(HY4_FLASH_KPOOL_PROBE) ? HY4_FLASH_KPOOL_PROBE : 0;

// grid (n_group), threads HY4_FLASH_TG
kernel void hy4_flash_indexer_kpool(device const half* iv [[buffer(0)]], device const half* ig [[buffer(1)]],
                              device const float* ape [[buffer(2)]], device const float* wnorm [[buffer(3)]],
                              device float* pooled [[buffer(4)]], device half* pooled16 [[buffer(5)]],
                              constant Hy4FlashKpoolArgs& a [[buffer(6)]], uint3 tgp [[threadgroup_position_in_grid]],
                              ushort tid [[thread_index_in_threadgroup]], ushort sg [[simdgroup_index_in_threadgroup]],
                              ushort lane [[thread_index_in_simdgroup]]) {
    threadgroup float part[HY4_FLASH_TG_SG];
    threadgroup float buf[1024];        // the pooled key before the norm and the rotation (dim <= 1024)
    const uint first = a.kpool * (a.g0 + tgp.x);
    device const half* ivr = iv + (ulong)first * a.iv_stride;
    device const half* igr = ig + (ulong)first * a.ig_stride;
    float ss = 0.0f;
    for (uint d = tid; d < a.dim; d += HY4_FLASH_TG) {
        float gl[HY4_FLASH_KPOOL_MAX];
        float mx = -INFINITY;
        for (uint m = 0; m < a.kpool; ++m) {
            gl[m] = float(igr[(ulong)m * a.ig_stride + d]) + ape[(ulong)m * a.ape_stride + d];
            mx = max(mx, gl[m]);
        }
        float sum = 0.0f;
        for (uint m = 0; m < a.kpool; ++m) {
            gl[m] = precise::exp(gl[m] - mx);
            sum += gl[m];
        }
        float v = 0.0f;
        for (uint m = 0; m < a.kpool; ++m) v += gl[m] / sum * float(ivr[(ulong)m * a.iv_stride + d]);
        buf[d] = v;
        ss += v * v;
    }
    const float scale = precise::rsqrt(hy4_flash_tg_sum(ss, part, sg, lane) / float(a.dim) + a.eps);
    for (uint d = tid; d < a.dim; d += HY4_FLASH_TG) buf[d] = buf[d] * scale * wnorm[d];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    device float* o = pooled + (ulong)tgp.x * a.out_stride;
    device half* o16 = pooled16 + (ulong)tgp.x * a.out16_stride;
    const uint off = a.dim - a.n_rot;
    for (uint d = tid; d < off; d += HY4_FLASH_TG) {
        o[d] = buf[d];
        o16[d] = half(buf[d]);
    }
    // the rotation, the same recurrence hy4_flash_rope_norm uses (repeated per thread from the position, so the value
    // does not depend on how the pairs are spread over threads)
    const uint rope_pos = hy4_flash_kpool_probe == 1 ? first + a.kpool - 1 : first;
    for (uint k = tid; 2 * k < a.n_rot; k += HY4_FLASH_TG) {
        float th = float(rope_pos);
        for (uint j = 0; j < k; ++j) th *= a.theta_scale;
        const float c = precise::cos(th), s = precise::sin(th);
        const float x0 = buf[off + 2 * k], x1 = buf[off + 2 * k + 1];
        const float y0 = x0 * c - x1 * s, y1 = x0 * s + x1 * c;
        o[off + 2 * k] = y0;
        o[off + 2 * k + 1] = y1;
        o16[off + 2 * k] = half(y0);
        o16[off + 2 * k + 1] = half(y1);
    }
}

// ---- the score -----------------------------------------------------------------------------------------------------
// score[t][g] = sum_h relu(iq[t][h] . pooled[g]) * wgt[t][h], added in HEAD order; -inf where the pool is not
// visible, i.e. where it does not END at or before the query's position (hy4_block.py kpool_select).
struct Hy4FlashScoreArgs {
    uint dim;          // key_length (128)
    uint n_head;       // indexer heads (16)
    uint n_pool;
    uint n_tok;
    uint kpool;
    uint q_stride;     // floats between tokens of iq (n_head * dim)
    uint pool_stride;  // halves between pools
    uint w_stride;     // floats between tokens of wgt
    uint s_stride;     // floats between tokens of score
};

// grid (n_pool, n_tok), threads HY4_FLASH_TG
kernel void hy4_flash_indexer_score(device const float* iq [[buffer(0)]], device const half* pooled [[buffer(1)]],
                              device const float* wgt [[buffer(2)]], device const int* pos [[buffer(3)]],
                              device float* score [[buffer(4)]], constant Hy4FlashScoreArgs& a [[buffer(5)]],
                              uint3 tgp [[threadgroup_position_in_grid]], ushort tid [[thread_index_in_threadgroup]],
                              ushort sg [[simdgroup_index_in_threadgroup]], ushort lane [[thread_index_in_simdgroup]]) {
    threadgroup float hv[HY4_FLASH_IDX_MAXH];
    device const float* qr = iq + (ulong)tgp.y * a.q_stride;
    device const half* pr = pooled + (ulong)tgp.x * a.pool_stride;
    device const float* wr = wgt + (ulong)tgp.y * a.w_stride;
    for (uint h = sg; h < a.n_head; h += HY4_FLASH_TG_SG) {
        float s = 0.0f;
        for (uint d = lane; d < a.dim; d += 32) s += qr[(ulong)h * a.dim + d] * float(pr[d]);
        s = hy4_flash_simd_sum(s);
        if (lane == 0) hv[h] = max(s, 0.0f) * wr[h];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
        float t = 0.0f;
        for (uint h = 0; h < a.n_head; ++h) t += hv[h];
        const int pool_end = int(a.kpool) * int(tgp.x) + int(a.kpool) - 1;
        score[(ulong)tgp.y * a.s_stride + tgp.x] = pool_end <= pos[tgp.y] ? t : -INFINITY;
    }
}

// ---- the selection -------------------------------------------------------------------------------------------------
// The threshold K is the cap-th largest order-preserving uint key of the scores, and the pools are emitted in
// ASCENDING order so that the tie quota is spent on the smallest indices - the tie rule above.
//
// K BY A RADIX SELECT.  "The largest K with #{key >= K} >= cap" is exactly the cap-th largest key (any larger K
// counts only the keys above it, fewer than cap), and an MSB-first radix select finds that key in 4 passes of 8
// bits: per pass, a histogram of the next digit over the keys that match the digits fixed so far, then the digit
// where the count from the top reaches what is still needed.  Integer counting, so neither the threadgroup
// atomics' order nor the window can move it.  What is still needed after the last digit is the tie quota
// cap - #{key > K}, and the last histogram bin is #{key == K}.
// THE EMIT IS A PARALLEL PREFIX (it assumes HY4_FLASH_TG threads in simdgroups of 32: psum has HY4_FLASH_TG_SG slots), with
// these decisions: pool g is taken iff key > K, or key == K and its rank among the equal pools (how many equal
// pools have a smaller index) is below the quota; its cells go to 4 x (pools taken before g).  Both ranks are
// integer prefix sums over g ascending, chunk by chunk.  The output - which cells, in which order, n_sel - is
// checked bit for bit against the serial reference by tmp/tests/hy4_flash/topk.
//
// The recalled pools cover cells 0 .. kpool*nv - 1 and the tail starts at kpool*nv, so the two never overlap and
// the concatenation is already sorted: no unique pass is needed (the numpy oracle's np.unique is a no-op here).
struct Hy4FlashTopkArgs {
    uint kpool;
    uint top_k;        // the cell budget (2048)
    uint n_pool;       // pools the score buffer holds
    uint n_tok;
    uint s_stride;     // floats between tokens of score
    uint sel_stride;   // ints between tokens of sel
};

// Test hooks: 1 = ties go to the LARGER pool index (the tie test's negative control), 2 = the budget shrinks when
// the dispatch carries more than one token, which is a selection that follows the window (the T-invariance tests'
// negative control), 3 = NO FORCED TAIL, the `no_tail` mutation the parity gate must catch.  Unset in production.
constant int HY4_FLASH_IDX_PROBE [[function_constant(51)]];
constant int hy4_flash_idx_probe = is_function_constant_defined(HY4_FLASH_IDX_PROBE) ? HY4_FLASH_IDX_PROBE : 0;

// bits -> a uint that orders like the float (larger key == larger value); -inf maps to the smallest.  NaN has no
// place in the order and is not expected: a score is a finite sum of products, or exactly -inf from the mask.
static inline uint hy4_flash_fkey(float f) {
    const uint b = as_type<uint>(f);
    return (b & 0x80000000u) ? ~b : (b | 0x80000000u);
}

// grid (n_tok), threads HY4_FLASH_TG.  sel holds the attended CELL indices, n_sel their count.
kernel void hy4_flash_indexer_topk(device const float* score [[buffer(0)]], device const int* pos [[buffer(1)]],
                             device int* sel [[buffer(2)]], device int* n_sel [[buffer(3)]],
                             constant Hy4FlashTopkArgs& a [[buffer(4)]], uint3 tgp [[threadgroup_position_in_grid]],
                             ushort tid [[thread_index_in_threadgroup]], ushort sg [[simdgroup_index_in_threadgroup]],
                             ushort lane [[thread_index_in_simdgroup]]) {
    threadgroup atomic_uint hist[256];   // the radix select's digit histogram
    threadgroup uint st[3];              // prefix, need, #{key == prefix}: written by thread 0 between two barriers
    threadgroup uint psum[2][HY4_FLASH_TG_SG]; // the emit's per-simdgroup totals
    const int p = pos[tgp.x];
    const int kp = int(a.kpool);
    const int nv = p / kp;                               // the query's own group does NOT contest
    int cap = int(a.top_k) / kp - 1;                     // ... and is force-recalled instead
    if (hy4_flash_idx_probe == 2 && a.n_tok > 1) cap -= 1;
    const int first = kp * (p / kp);
    const int ntail = hy4_flash_idx_probe == 3 ? 0 : p - first + 1;   // kpool slots, p%kpool+1 of them valid
    device const float* sr = score + (ulong)tgp.x * a.s_stride;
    device int* out = sel + (ulong)tgp.x * a.sel_stride;
    if (nv <= cap) {
        // below the budget the union of the recalled cells and the tail is the whole causal set 0 .. p
        for (int i = tid; i < nv * kp; i += HY4_FLASH_TG) out[i] = i;
        for (int i = tid; i < ntail; i += HY4_FLASH_TG) out[nv * kp + i] = first + i;
        if (tid == 0) n_sel[tgp.x] = nv * kp + ntail;
        return;
    }
    // ---- K = the cap-th largest key (nv > cap here)
    uint prefix = 0, need = (uint)cap;
    for (int pass = 0; pass < 4; ++pass) {
        const uint shift = 24u - 8u * uint(pass);
        const uint hmask = pass == 0 ? 0u : (0xFFFFFFFFu << (shift + 8u));   // the digits fixed so far
        for (int b = tid; b < 256; b += HY4_FLASH_TG) atomic_store_explicit(&hist[b], 0u, memory_order_relaxed);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (int g = tid; g < nv; g += HY4_FLASH_TG) {
            const uint k = hy4_flash_fkey(sr[g]);
            if ((k & hmask) == prefix) atomic_fetch_add_explicit(&hist[(k >> shift) & 255u], 1u, memory_order_relaxed);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid == 0) {
            // from the top digit down: the first digit whose count reaches what is still needed (digit 0 must,
            // since every key that matches the prefix is counted somewhere and their total is >= need)
            uint above = 0;
            int d = 255;
            for (; d > 0; --d) {
                const uint h = atomic_load_explicit(&hist[d], memory_order_relaxed);
                if (above + h >= need) break;
                above += h;
            }
            st[0] = prefix | (uint(d) << shift);
            st[1] = need - above;
            st[2] = atomic_load_explicit(&hist[d], memory_order_relaxed);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        prefix = st[0];
        need = st[1];
    }
    const uint K = prefix;
    const uint quota = need;   // cap - #{key > K}: how many of the equal-key pools get in
    const uint ce = st[2];     // #{key == K}
    // the negative control takes the LAST quota of the equal pools instead of the first
    const uint skip = hy4_flash_idx_probe == 1 ? ce - quota : 0u;

    // ---- the emit: ascending g, chunks of HY4_FLASH_TG pools, two integer prefix sums a chunk
    uint eq_base = 0, take_base = 0;   // equal pools / taken pools in the chunks before this one
    for (int c0 = 0; c0 < nv; c0 += HY4_FLASH_TG) {   // (the trip count is the same in every thread: barriers inside)
        const int g = c0 + int(tid);
        const uint k = g < nv ? hy4_flash_fkey(sr[g]) : 0u;
        const uint eq = g < nv && k == K ? 1u : 0u;
        const uint gt = g < nv && k > K ? 1u : 0u;
        const uint eq_ex = simd_prefix_exclusive_sum(eq);
        if (lane == 31) psum[0][sg] = eq_ex + eq;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        uint eq_sg = 0, eq_all = 0;
        for (ushort q = 0; q < HY4_FLASH_TG_SG; ++q) {
            eq_sg += q < sg ? psum[0][q] : 0u;
            eq_all += psum[0][q];
        }
        const uint rank = eq_base + eq_sg + eq_ex;   // equal pools with a smaller index
        const uint take = gt != 0u || (eq != 0u && rank >= skip && rank < skip + quota) ? 1u : 0u;
        const uint tk_ex = simd_prefix_exclusive_sum(take);
        if (lane == 31) psum[1][sg] = tk_ex + take;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        uint tk_sg = 0, tk_all = 0;
        for (ushort q = 0; q < HY4_FLASH_TG_SG; ++q) {
            tk_sg += q < sg ? psum[1][q] : 0u;
            tk_all += psum[1][q];
        }
        if (take != 0u) {
            const int at = int(take_base + tk_sg + tk_ex) * kp;
            for (int m = 0; m < kp; ++m) out[at + m] = g * kp + m;
        }
        eq_base += eq_all;
        take_base += tk_all;
        threadgroup_barrier(mem_flags::mem_threadgroup);   // (psum is double-buffered, so this one is belt and braces)
    }
    const int n = int(take_base) * kp;
    for (int i = tid; i < ntail; i += HY4_FLASH_TG) out[n + i] = first + i;
    if (tid == 0) n_sel[tgp.x] = n + ntail;
}
