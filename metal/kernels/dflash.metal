// metal/kernels/dflash.metal - the native DFlash2 drafter (metal/src/engine/dflash.cpp), after quant/units/gemv.metal.
//
// The projections are gemv.metal's Q8_0 tiles on plain [row][n] activations; what is here is the rest of the graph
// (llama.cpp src/models/dflash.cpp, native-spec/graph.md): the rms norms, the two-tap dynamic conv, q/k norm + RoPE,
// the sliding-window attention over a K/V ring plus the block's own rows, and the candidate selector.  Every
// reduction runs in a fixed order (no atomics), so a draft is bitwise repeatable.
//
//   df_rms             y = x * rsqrt(mean(x^2) + eps) * w
//   df_conv            side-0 conv of a sublayer's input
//   df_conv_res_rms    side-1 conv of its output + residual, then the next norm (one pass)
//   df_swiglu          h = silu(g) * u
//   df_qkv_decode      the block's q/k/v: per-head norm, RoPE; k, v to f16 (the block's own K/V buffer)
//   df_qkv_inject      the injected rows' k/v: norm, RoPE, f16 into the ring slot of their position
//   df_attn / _merge   split-K attention (ring chunks + the block) and its log-sum-exp merge
//   df_topk            the 16 best draft-vocab logits of a row (ties: lowest index), mapped through d2t
//   df_selector        the candidate lattice, packed as llama.cpp's nextn rows

constant constexpr int DF_D = 2560;       // model width
constant constexpr int DF_HD = 128;       // head dim
constant constexpr int DF_RING = 2064;    // slots of a layer's K/V ring (2048 window + 16 spare)
constant constexpr int DF_CH = 258;       // ring slots one attention chunk walks (8 x 258 = 2064)
constant constexpr int DF_NCH = 9;        // attention chunks: 8 of the ring + the block's own keys
constant constexpr int DF_TOPK = 16;
constant constexpr float DF_EPS = 1e-6f;

// grid (rows), 256 threads; in place is fine (a thread reads and writes its own elements, a barrier between)
kernel void df_rms(device const float* x [[buffer(0)]], device const float* w [[buffer(1)]], device float* y [[buffer(2)]],
                   uint3 tg [[threadgroup_position_in_grid]], ushort tid [[thread_index_in_threadgroup]],
                   ushort sg [[simdgroup_index_in_threadgroup]], ushort lane [[thread_index_in_simdgroup]]) {
    threadgroup float part[8];
    device const float* xr = x + tg.x * DF_D;
    float ss = 0.0f;
    for (int d = tid; d < DF_D; d += 256) ss = fma(xr[d], xr[d], ss);
    ss = simd_sum(ss);
    if (lane == 0) part[sg] = ss;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float tot = 0.0f;
    for (int i = 0; i < 8; ++i) tot += part[i];
    const float rs = 1.0f / sqrt(tot / float(DF_D) + DF_EPS);
    device float* yr = y + tg.x * DF_D;
    for (int d = tid; d < DF_D; d += 256) yr[d] = xr[d] * rs * w[d];
}

// y[ch] = (base[ch, 0, side] + dyn[g + 320 side]) x[t][ch] + (base[ch, 1, side] + dyn[g + 160 + 320 side]) x[t-1][ch]
// (the second tap is 0 at block row 0); g = ch / 16.  base: [ch + 2560 tap + 5120 side], dyn rows of 640.
static inline float df_conv_at(device const float* base, device const float* dyn, device const float* x, int t, int ch, int side) {
    const int g = ch >> 4;
    const float w0 = base[ch + 5120 * side] + dyn[t * 640 + g + 320 * side];
    const float w1 = base[ch + 2560 + 5120 * side] + dyn[t * 640 + g + 160 + 320 * side];
    float y = w0 * x[t * DF_D + ch];
    if (t > 0) y += w1 * x[(t - 1) * DF_D + ch];
    return y;
}

// grid (10, rows), 256 threads
kernel void df_conv(device const float* x [[buffer(0)]], device const float* dyn [[buffer(1)]],
                    device const float* base [[buffer(2)]], device float* y [[buffer(3)]],
                    uint3 tg [[threadgroup_position_in_grid]], ushort tid [[thread_index_in_threadgroup]]) {
    const int ch = int(tg.x) * 256 + tid, t = int(tg.y);
    y[t * DF_D + ch] = df_conv_at(base, dyn, x, t, ch, 0);
}

// grid (rows), 256 threads, 10 channels each: r = conv(x, side 1) + res -> rout; yn = rmsnorm(r, w)
kernel void df_conv_res_rms(device const float* x [[buffer(0)]], device const float* dyn [[buffer(1)]],
                            device const float* base [[buffer(2)]], device const float* res [[buffer(3)]],
                            device float* rout [[buffer(4)]], device const float* w [[buffer(5)]],
                            device float* yn [[buffer(6)]], uint3 tg [[threadgroup_position_in_grid]],
                            ushort tid [[thread_index_in_threadgroup]], ushort sg [[simdgroup_index_in_threadgroup]],
                            ushort lane [[thread_index_in_simdgroup]]) {
    threadgroup float part[8];
    const int t = int(tg.x);
    float r[10];
    float ss = 0.0f;
    for (int i = 0; i < 10; ++i) {
        const int ch = tid + 256 * i;
        r[i] = df_conv_at(base, dyn, x, t, ch, 1) + res[t * DF_D + ch];
        rout[t * DF_D + ch] = r[i];
        ss = fma(r[i], r[i], ss);
    }
    ss = simd_sum(ss);
    if (lane == 0) part[sg] = ss;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float tot = 0.0f;
    for (int i = 0; i < 8; ++i) tot += part[i];
    const float rs = 1.0f / sqrt(tot / float(DF_D) + DF_EPS);
    for (int i = 0; i < 10; ++i) {
        const int ch = tid + 256 * i;
        yn[t * DF_D + ch] = r[i] * rs * w[ch];
    }
}

// h[t][i] = silu(g) * u, rows of n; grid (ceil(n * rows / 256)), 256 threads
kernel void df_swiglu(device const float* g [[buffer(0)]], device const float* u [[buffer(1)]], device float* h [[buffer(2)]],
                      constant uint& total [[buffer(3)]], uint gid [[thread_position_in_grid]]) {
    if (gid >= total) return;
    const float gv = g[gid];
    h[gid] = gv / (1.0f + exp(-gv)) * u[gid];
}

// One head's 128 values: rmsnorm with w (the head norm), then RoPE at position p (NEOX pairs (k, k + 64); theta as
// ggml-metal's kernel_rope_multi computes it).  128 threads, thread d returns output element d.
static inline float df_head_norm_rope(float v, device const float* w, int p, threadgroup float* xs, threadgroup float* part,
                                      ushort d, ushort sg, ushort lane) {
    float ss = simd_sum(v * v);
    if (lane == 0) part[sg] = ss;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float tot = part[0] + part[1] + part[2] + part[3];
    xs[d] = v * (1.0f / sqrt(tot / float(DF_HD) + DF_EPS)) * w[d];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const int k = d & 63;
    const float theta = float(p) * pow(10000.0f, (-1.0f / float(DF_HD)) * float(2 * k));
    const float c = cos(theta), s = sin(theta);
    const float x0 = xs[k], x1 = xs[k + 64];
    return d < 64 ? x0 * c - x1 * s : x0 * s + x1 * c;
}

struct DfQkvArgs {
    int pos0;   // position of the block's / group's first row
    int nb;     // rows
};

// grid (48, rows), 128 threads: heads 0..31 q (-> qo f32), 32..39 k (norm, RoPE -> nk f16 [kvh][row][128]), 40..47 v (-> nv f16)
kernel void df_qkv_decode(device const float* q [[buffer(0)]], device const float* k [[buffer(1)]],
                          device const float* v [[buffer(2)]], device const float* qw [[buffer(3)]],
                          device const float* kw [[buffer(4)]], device float* qo [[buffer(5)]],
                          device half* nk [[buffer(6)]], device half* nv [[buffer(7)]], constant DfQkvArgs& a [[buffer(8)]],
                          uint3 tg [[threadgroup_position_in_grid]], ushort d [[thread_index_in_threadgroup]],
                          ushort sg [[simdgroup_index_in_threadgroup]], ushort lane [[thread_index_in_simdgroup]]) {
    threadgroup float xs[DF_HD];
    threadgroup float part[4];
    const int h = int(tg.x), r = int(tg.y), p = a.pos0 + r;
    if (h < 32) {
        const float o = df_head_norm_rope(q[r * 4096 + h * DF_HD + d], qw, p, xs, part, d, sg, lane);
        qo[r * 4096 + h * DF_HD + d] = o;
    } else if (h < 40) {
        const int kh = h - 32;
        const float o = df_head_norm_rope(k[r * 1024 + kh * DF_HD + d], kw, p, xs, part, d, sg, lane);
        nk[(kh * 8 + r) * DF_HD + d] = half(o);
    } else {
        const int kh = h - 40;
        nv[(kh * 8 + r) * DF_HD + d] = half(v[r * 1024 + kh * DF_HD + d]);
    }
}

// grid (16, rows), 128 threads: heads 0..7 k, 8..15 v of one layer's injected rows -> ring slot (pos0 + r) % 2064
// rk/rv: the layer's ring [kvh][slot][128]
kernel void df_qkv_inject(device const float* k [[buffer(0)]], device const float* v [[buffer(1)]],
                          device const float* kw [[buffer(2)]], device half* rk [[buffer(3)]], device half* rv [[buffer(4)]],
                          constant DfQkvArgs& a [[buffer(5)]], uint3 tg [[threadgroup_position_in_grid]],
                          ushort d [[thread_index_in_threadgroup]], ushort sg [[simdgroup_index_in_threadgroup]],
                          ushort lane [[thread_index_in_simdgroup]]) {
    threadgroup float xs[DF_HD];
    threadgroup float part[4];
    const int h = int(tg.x), r = int(tg.y), p = a.pos0 + r, slot = p % DF_RING;
    if (h < 8) {
        const float o = df_head_norm_rope(k[r * 1024 + h * DF_HD + d], kw, p, xs, part, d, sg, lane);
        rk[(h * DF_RING + slot) * DF_HD + d] = half(o);
    } else {
        rv[((h - 8) * DF_RING + slot) * DF_HD + d] = half(v[r * 1024 + (h - 8) * DF_HD + d]);
    }
}

struct DfAttnArgs {
    int pos0;
    int nb;
};

// Split-K attention: a threadgroup is (kv head, chunk); chunks 0..7 walk 258 ring slots each, chunk 8 the block's
// own rows (always visible: the block is bidirectional).  A ring slot is visible iff it holds a position
// sp >= 0 (SP: the slot's position, -1 empty) and the row's window reaches it: sp > p1 - 2048.  Simdgroup = query
// row; in it 8 lanes share one of the group's 4 q heads and 16 of the 128 dims each.  Out: part[(row*32 + qh)*9 + c]
// = (acc[128], m, l).
kernel void df_attn(device const float* Q [[buffer(0)]], device const half* rk [[buffer(1)]], device const half* rv [[buffer(2)]],
                    device const half* nk [[buffer(3)]], device const half* nv [[buffer(4)]],
                    device const int* SP [[buffer(5)]], device float* part [[buffer(6)]], constant DfAttnArgs& a [[buffer(7)]],
                    uint3 tg [[threadgroup_position_in_grid]], ushort sg [[simdgroup_index_in_threadgroup]],
                    ushort lane [[thread_index_in_simdgroup]]) {
    const int row = sg;
    if (row >= a.nb) return;   // a whole simdgroup: the shuffles below stay uniform
    const int kh = int(tg.x), c = int(tg.y), g = lane >> 3, l8 = lane & 7, qh = kh * 4 + g;
    const int p1 = a.pos0 + row, lo = p1 - 2047;
    float q[16], acc[16];
    for (int i = 0; i < 16; ++i) { q[i] = Q[row * 4096 + qh * DF_HD + l8 * 16 + i]; acc[i] = 0.0f; }
    const float scale = 1.0f / sqrt(float(DF_HD));
    float m = -INFINITY, l = 0.0f;
    const int n = c < 8 ? DF_CH : a.nb;
    for (int j = 0; j < n; ++j) {
        device const half* kr;
        device const half* vr;
        if (c < 8) {
            const int s = c * DF_CH + j, sp = SP[s];
            if (sp < 0 || sp < lo) continue;
            kr = rk + (kh * DF_RING + s) * DF_HD + l8 * 16;
            vr = rv + (kh * DF_RING + s) * DF_HD + l8 * 16;
        } else {
            kr = nk + (kh * 8 + j) * DF_HD + l8 * 16;
            vr = nv + (kh * 8 + j) * DF_HD + l8 * 16;
        }
        float dot = 0.0f;
        for (int i = 0; i < 16; ++i) dot = fma(q[i], float(kr[i]), dot);
        dot += simd_shuffle_xor(dot, 1);
        dot += simd_shuffle_xor(dot, 2);
        dot += simd_shuffle_xor(dot, 4);
        const float sc = dot * scale, mn = max(m, sc), e = exp(sc - mn), corr = exp(m - mn);
        for (int i = 0; i < 16; ++i) acc[i] = fma(acc[i], corr, e * float(vr[i]));
        l = fma(l, corr, e);
        m = mn;
    }
    device float* o = part + ((row * 32 + qh) * DF_NCH + c) * 130;
    for (int i = 0; i < 16; ++i) o[l8 * 16 + i] = acc[i];
    if (l8 == 0) { o[128] = m; o[129] = l; }
}

// grid (rows * 32), 128 threads: Attn[row][qh*128 + d] = sum_c w_c acc_c / sum_c w_c l_c, w_c = exp(m_c - max m)
kernel void df_attn_merge(device const float* part [[buffer(0)]], device float* out [[buffer(1)]],
                          uint3 tg [[threadgroup_position_in_grid]], ushort d [[thread_index_in_threadgroup]]) {
    device const float* p = part + int(tg.x) * DF_NCH * 130;
    float mx = -INFINITY;
    for (int c = 0; c < DF_NCH; ++c) mx = max(mx, p[c * 130 + 128]);
    float l = 0.0f, acc = 0.0f;
    for (int c = 0; c < DF_NCH; ++c) {
        const float w = exp(p[c * 130 + 128] - mx);   // an empty chunk: m = -inf, w = 0
        l = fma(p[c * 130 + 129], w, l);
        acc = fma(p[c * 130 + d], w, acc);
    }
    out[int(tg.x) * DF_HD + d] = acc / l;
}

struct DfTopkArgs {
    uint nv;   // draft vocab rows
};

// grid (rows), 1024 threads (thread t holds logits t, t + 1024, ...): 16 rounds of "the largest not yet taken, the
// lowest index on a tie".  lg: [row][nv], row r of the grid is block row r + 1.  cand/unary: [block row][16].
kernel void df_topk(device const float* lg [[buffer(0)]], device const int* d2t [[buffer(1)]], device int* cand [[buffer(2)]],
                    device float* unary [[buffer(3)]], constant DfTopkArgs& a [[buffer(4)]],
                    uint3 tg [[threadgroup_position_in_grid]], ushort tid [[thread_index_in_threadgroup]],
                    ushort sg [[simdgroup_index_in_threadgroup]], ushort lane [[thread_index_in_simdgroup]]) {
    constexpr int VPT = 40;   // 1024 x 40 >= 40525
    threadgroup float sv[32];
    threadgroup int si[32];
    threadgroup int win[2];
    float v[VPT];
    device const float* row = lg + ulong(tg.x) * a.nv;
    for (int k = 0; k < VPT; ++k) {
        const uint i = uint(tid) + 1024u * k;
        v[k] = i < a.nv ? row[i] : -INFINITY;
    }
    for (int round = 0; round < DF_TOPK; ++round) {
        float bv = -INFINITY;
        int bi = 0x7fffffff;
        for (int k = 0; k < VPT; ++k) {
            const int i = int(tid) + 1024 * k;
            if (v[k] > bv) { bv = v[k]; bi = i; }   // ascending i: the first of equal values stays
        }
        for (int off = 16; off > 0; off >>= 1) {
            const float ov = simd_shuffle_down(bv, off);
            const int oi = simd_shuffle_down(bi, off);
            if (ov > bv || (ov == bv && oi < bi)) { bv = ov; bi = oi; }
        }
        if (lane == 0) { sv[sg] = bv; si[sg] = bi; }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (sg == 0) {
            bv = sv[lane];
            bi = si[lane];
            for (int off = 16; off > 0; off >>= 1) {
                const float ov = simd_shuffle_down(bv, off);
                const int oi = simd_shuffle_down(bi, off);
                if (ov > bv || (ov == bv && oi < bi)) { bv = ov; bi = oi; }
            }
            if (lane == 0) {
                win[0] = bi;
                cand[(tg.x + 1) * DF_TOPK + round] = d2t[bi];
                unary[(tg.x + 1) * DF_TOPK + round] = bv;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        const int wi = win[0];
        if ((wi & 1023) == int(tid))
            for (int k = 0; k < VPT; ++k)
                if (k == (wi >> 10)) v[k] = -INFINITY;
        threadgroup_barrier(mem_flags::mem_threadgroup);   // win is rewritten next round
    }
}

struct DfSelArgs {
    int anchor;       // the block's first token (position 1's predecessor)
    uint off_pre;     // byte offsets of selector_predecessor / selector_successor in W
    uint off_next;
};

// grid (rows - 1), 256 threads: thread (a, s) scores successor s after predecessor a at block row p = grid + 1:
//   score = unary[p][s] + sum_r succ[cand[p][s]][r] * pre[pid][r] * gate[p][r]   (pid: the anchor at p = 1, else cand[p-1][a])
// rows are Q8_0 256-vectors (8 blocks of 34 bytes) gathered by token id.  gate: [row - 1][256] (the selector_hidden
// gemv of block rows 1..).  lat: llama.cpp's nextn rows, 2560 wide: ids at 0..15, score[a][s] at 16 + 16 a + s.
kernel void df_selector(device const uchar* W [[buffer(0)]], device const int* cand [[buffer(1)]],
                        device const float* unary [[buffer(2)]], device const float* gate [[buffer(3)]],
                        device float* lat [[buffer(4)]], constant DfSelArgs& a [[buffer(5)]],
                        uint3 tg [[threadgroup_position_in_grid]], ushort tid [[thread_index_in_threadgroup]]) {
    threadgroup float gs[256];
    const int p = int(tg.x) + 1, aa = tid >> 4, s = tid & 15;
    gs[tid] = gate[(p - 1) * 256 + tid];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const int pid = p == 1 ? a.anchor : cand[(p - 1) * DF_TOPK + aa];
    const int sid = cand[p * DF_TOPK + s];
    device const uchar* pr = W + a.off_pre + ulong(pid) * 272;
    device const uchar* sr = W + a.off_next + ulong(sid) * 272;
    float acc = 0.0f;
    for (int b = 0; b < 8; ++b) {
        const float dp = float(*(device const half*)(pr + b * 34)), ds = float(*(device const half*)(sr + b * 34));
        device const char* pq = (device const char*)(pr + b * 34 + 2);
        device const char* sq = (device const char*)(sr + b * 34 + 2);
        for (int j = 0; j < 32; ++j) {
            const float cond = (dp * float(pq[j])) * gs[b * 32 + j];
            acc = fma(ds * float(sq[j]), cond, acc);
        }
    }
    device float* row = lat + p * DF_D;
    row[16 + 16 * aa + s] = acc + unary[p * DF_TOPK + s];
    if (tid < 16) row[tid] = float(cand[p * DF_TOPK + tid]);
}
