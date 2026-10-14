// metal/kernels/kv_q4.metal - Q4_0 KV with Walsh-Hadamard rotation (M0-p), the standalone kernels of
// src/kernels/cuda/kv_q4.cu on Metal.  Contract: include/strata/kernels/kv_q4.hpp.  Bit-level port of the CUDA
// reference (the PC's parity fixture is the same arithmetic), with the numerics pinned:
//
//   * the codeword math runs in fp32 (d = mval / -8, id = 1 / d), and the fp16 scale is the software
//     round-to-nearest-even of include/strata/kernels/f16_bits.hpp's f16_from_f32 (bit-exact on every input;
//     MSL's half() conversion is not used);
//   * the codeword is int(fma(x, id, 8.5f)): the CUDA build compiles x * id + 8.5f with --fmad=true, i.e. ONE
//     multiply-add with a single rounding; the explicit fma() is that contraction, whatever Metal's math mode is.
//     int() truncates toward zero, as __float2int_rz does (rint() is NOT used);
//   * the group maximum's tie rule is the device one, for FINITE inputs: (a > amax) || (a == amax && v > mval),
//     butterflies over xor offsets 16, 8, 4, 2, 1, so every lane converges to the same (|max|, value) pair - a tie
//     in |x| resolves to the LARGER value (x = {v, -v, ...} quantizes with d = v / -8), never to whichever lane
//     happened to win;
//   * fwht256 is orthonormal (scale 1/16, passed in) and so its own inverse; one simdgroup per row, 8 values per
//     lane in registers, the low 5 index bits across lanes and the high 3 within them - the CUDA register order.
//
// Mac has no paging (unified memory): the page table is the identity where the engine passes one.
//
// Two binding forms of every kernel, one body: the CUDA-shaped standalone kernels (each pointer/scalar its own
// buffer - what tests/kv_q4_parity.cpp drives through mtl_rt) and the *_e twins that read ONE argument struct
// from buffer 0 (the engine's dispatch: strata/gpu/metal.hpp's dispatch() binds one bytes blob at index 0, and
// the struct's pointer fields carry the gpuAddress values).  The twins' structs are mirrored in
// metal/src/engine/kv_q4_host.cpp - keep the field order and types in step.

#define QK4 32

struct kv_q4_blk {          // include/strata/kernels/kv_q4.hpp: 2-byte scale + 16 packed bytes, 18 total
    ushort d;
    uchar qs[16];
};

// ---- the software fp16 conversions of include/strata/kernels/f16_bits.hpp, as bits (no half() anywhere)
static inline ushort f16_from_f32(float f) {
    const uint x = as_type<uint>(f);
    const uint sign = (x >> 16) & 0x8000u;
    const uint rawexp = (x >> 23) & 0xFFu;
    const int ex = (int)rawexp - 127 + 15;
    uint man = x & 0x7FFFFFu;
    if (rawexp == 0xFFu) return (ushort)(sign | 0x7C00u | (man ? 0x200u : 0u));   // inf, or a quiet/NaN
    if (ex >= 31) return (ushort)(sign | 0x7C00u);                                // finite overflow -> inf
    if (ex <= 0) {
        if (ex < -10) return (ushort)sign;                                        // underflows to a signed zero
        man |= 0x800000u;                                                         // restore the implicit 1
        const uint sh = (uint)(14 - ex);
        uint h = (man >> sh) & 0x3FFu;
        const uint rem = man & ((1u << sh) - 1u);
        if (rem > (1u << (sh - 1)) || (rem == (1u << (sh - 1)) && (h & 1u))) ++h;
        return (ushort)(sign | h);
    }
    ushort h = (ushort)(sign | ((uint)ex << 10) | (man >> 13));
    const uint rem = man & 0x1FFFu;
    if (rem > 0x1000u || (rem == 0x1000u && (h & 1u))) ++h;                       // the carry may bump the exponent
    return h;
}

static inline float f32_from_f16(ushort h) {
    const uint sign = (uint)(h & 0x8000u) << 16;
    const uint ex = (h >> 10) & 0x1Fu, man = h & 0x3FFu;
    uint out;
    if (ex == 0) {
        if (man == 0) {
            out = sign;
        } else {   // a subnormal is man * 2^-24: shift the leading 1 to bit 10, exponent 113 - s
            uint m = man;
            int s = 0;
            while ((m & 0x400u) == 0) { m <<= 1; ++s; }
            out = sign | ((uint)(113 - s) << 23) | ((m & 0x3FFu) << 13);
        }
    } else if (ex == 31) {
        out = sign | 0x7F800000u | (man << 13);
    } else {
        out = sign | ((ex - 15 + 127) << 23) | (man << 13);
    }
    return as_type<float>(out);
}

// ---- one 32-value group in one simdgroup (lane = element), ggml's q4_0: d = (the value of largest |x|) / -8,
// q = clamp(trunc(fma(x, 1/d, 8.5)), 0, 15).  Returns the scale bits; `byte` is lane t's packed byte for t < 16
// (element t in the low nibble, t + 16 in the high one).  The tie rule keeps every lane on the same scale.
static inline ushort q4_group(float x, ushort lane, thread uchar& byte) {
    float amax = fabs(x), mval = x;
    for (ushort o = 16; o > 0; o >>= 1) {
        const float a = simd_shuffle_xor(amax, o);
        const float v = simd_shuffle_xor(mval, o);
        if (a > amax || (a == amax && v > mval)) { amax = a; mval = v; }
    }
    const float d = mval / -8.0f;
    const float id = d != 0.0f ? 1.0f / d : 0.0f;
    const int q = int(fma(x, id, 8.5f));   // CUDA --fmad=true: the contracted multiply-add; int() truncates
    const uchar qc = (uchar)(q < 0 ? 0 : (q > 15 ? 15 : q));
    const uchar qhi = simd_shuffle(qc, min(ushort(lane + 16), ushort(31)));
    byte = (uchar)(qc | (qhi << 4));
    return f16_from_f32(d);
}

// pool row = ((page * kv_heads + h) * page_size + cell), 8 blocks of head_dim 256 -> 144 B (see kv_q4.cu)
static inline void q4_store(device uchar* pool, ulong row, int b, ushort lane, ushort d, uchar byte) {
    device kv_q4_blk* blk = (device kv_q4_blk*)(pool + row * sizeof(kv_q4_blk) * 8) + b;
    if (lane == 0) blk->d = d;
    if (lane < 16) blk->qs[lane] = byte;
}

// ---- fwht256: one row per simdgroup, 8 values per lane in registers (kv_q4.cu's fwht256_kernel)
static inline void fwht256_lane(device const float* src, device float* dst, ulong n_rows, float scale,
                                ulong r, ushort lane) {
    if (r >= n_rows) return;
    device const float* row_src = src + r * 256;
    device float* row_dst = dst + r * 256;
    constexpr int el_w = 8;

    float reg[el_w];
    for (int i = 0; i < el_w; ++i) reg[i] = row_src[i * 32 + lane] * scale;

    for (ushort h = 1; h < 32; h *= 2) {   // the low 5 index bits live across lanes
        for (int j = 0; j < el_w; ++j) {
            const float val = reg[j];
            const float val2 = simd_shuffle_xor(val, h);
            reg[j] = (lane & h) == 0 ? val + val2 : val2 - val;
        }
    }
    for (int h = 32; h < 256; h *= 2) {    // the high 3 bits across each lane's registers
        const int step = h / 32;
        for (int j = 0; j < el_w; j += 2 * step) {
            for (int k = 0; k < step; ++k) {
                const float x = reg[j + k];
                const float y = reg[j + k + step];
                reg[j + k] = x + y;
                reg[j + k + step] = x - y;
            }
        }
    }
    for (int i = 0; i < el_w; ++i) row_dst[i * 32 + lane] = reg[i];
}

// ---- one block = one 32-value group of one KV head of K (plane 0) or V (plane 1) for token step_idx
// (kv_q4.cu's kv_append_q4_kernel; the host copy's row uses the identity layout, pos / page_size)
static inline void kv_append_q4_thread(device uchar* k_q4, device uchar* v_q4,
                                       device const int* table, device const int* step, int step_stride, int planes,
                                       device const float* kcur, device const float* vcur,
                                       int kv_heads, int head_dim, int page_size,
                                       device uchar* host_k, device uchar* host_v,
                                       uint3 tg, ushort lane) {
    const uint step_idx = tg.z / (uint)planes;
    const bool is_v = (tg.z % (uint)planes) == 1u;
    const ulong pos = (ulong)step[(ulong)step_idx * (ulong)step_stride + 0 /* kStepPos */];
    const uint h = tg.x, b = tg.y;
    const ulong step_off = (ulong)step_idx * (ulong)kv_heads * (ulong)head_dim;
    const float x = (is_v ? vcur : kcur)[step_off + (ulong)h * head_dim + (ulong)b * QK4 + lane];
    uchar byte;
    const ushort d = q4_group(x, lane, byte);
    const int page = table[pos / (ulong)page_size];
    if (page >= 0)
        q4_store(is_v ? v_q4 : k_q4, ((ulong)page * kv_heads + h) * (ulong)page_size + (pos % (ulong)page_size),
                 (int)b, lane, d, byte);
    if (host_k != nullptr)
        q4_store(is_v ? host_v : host_k, ((pos / (ulong)page_size) * kv_heads + h) * (ulong)page_size + (pos % (ulong)page_size),
                 (int)b, lane, d, byte);
}

// ---- the prompt path: grid (T, kv_heads, groups), K then V (is_v_grid); also into the staging pool (identity
// layout) when given (kv_q4.cu's kv_append_q4_batch_kernel)
static inline void kv_append_q4_batch_thread(device uchar* k_q4, device uchar* v_q4,
                                             device const int* table, ulong pos0,
                                             device const float* K, device const float* V,
                                             int kv_heads, int head_dim, int page_size, int is_v_grid,
                                             device uchar* host_k, device uchar* host_v,
                                             device uchar* stage_k, device uchar* stage_v,
                                             uint3 tg, ushort lane) {
    const ulong t = tg.x;
    const ulong pos = pos0 + t;
    const uint h = tg.y, b = tg.z;
    const bool is_v = is_v_grid != 0;
    const float x = (is_v ? V : K)[t * (ulong)(kv_heads * head_dim) + (ulong)h * head_dim + (ulong)b * QK4 + lane];
    uchar byte;
    const ushort d = q4_group(x, lane, byte);
    const int page = table[pos / (ulong)page_size];
    const ulong row_id = ((pos / (ulong)page_size) * kv_heads + h) * (ulong)page_size + (pos % (ulong)page_size);
    if (page >= 0)
        q4_store(is_v ? v_q4 : k_q4, ((ulong)page * kv_heads + h) * (ulong)page_size + (pos % (ulong)page_size),
                 (int)b, lane, d, byte);
    if (host_k != nullptr) q4_store(is_v ? host_v : host_k, row_id, (int)b, lane, d, byte);
    if (stage_k != nullptr) q4_store(is_v ? stage_v : stage_k, row_id, (int)b, lane, d, byte);
}

// ---- gather step[kStepWidth = 3] cells named by `ids` into FP16 scratch [id][kv_head][head_dim]
// (kv_q4.cu's kv_gather_q4_kernel): one 32-thread row per block, 4 rows per threadgroup
static inline void kv_gather_q4_thread(device const uchar* k_q4, device const uchar* v_q4,
                                       device const int* table, device const int* ids, device const int* step,
                                       int kv_heads, int head_dim, int page_size,
                                       device ushort* k_scratch, device ushort* v_scratch,
                                       ulong blk_idx, uint t) {
    const int blocks_per_head = head_dim / QK4;                       // 8
    const int bytes_per_head = blocks_per_head * (int)sizeof(kv_q4_blk);   // 144
    const ulong total_blocks = (ulong)step[3 /* kStepWidth */] * (ulong)kv_heads * (ulong)blocks_per_head;
    if (blk_idx >= total_blocks) return;

    const ulong id = blk_idx / (ulong)(kv_heads * blocks_per_head);
    const int rem = (int)(blk_idx % (ulong)(kv_heads * blocks_per_head));
    const int h = rem / blocks_per_head;
    const int b = rem % blocks_per_head;

    const int cell = ids[id];
    const ulong page = (ulong)table[cell / (ulong)page_size];
    const ulong row = (page * kv_heads + h) * (ulong)page_size + (cell % (ulong)page_size);

    const device kv_q4_blk* k_blk = (const device kv_q4_blk*)(k_q4 + row * (ulong)bytes_per_head) + b;
    const device kv_q4_blk* v_blk = (const device kv_q4_blk*)(v_q4 + row * (ulong)bytes_per_head) + b;
    const float kd = f32_from_f16(k_blk->d);
    const float vd = f32_from_f16(v_blk->d);
    const int j = t < 16 ? (int)t : (int)(t - 16);
    const uchar k_byte = k_blk->qs[j];
    const uchar v_byte = v_blk->qs[j];
    const int kq = (t < 16) ? ((k_byte & 0x0F) - 8) : ((k_byte >> 4) - 8);
    const int vq = (t < 16) ? ((v_byte & 0x0F) - 8) : ((v_byte >> 4) - 8);
    const ulong dst = (id * (ulong)kv_heads + (ulong)h) * (ulong)head_dim + ((ulong)b * QK4 + t);
    k_scratch[dst] = f16_from_f32((float)kq * kd);
    v_scratch[dst] = f16_from_f32((float)vq * vd);
}

// ---- the standalone (CUDA-shaped) kernels: every pointer and scalar its own buffer, as the parity test binds
// them through mtl_rt.  Grids are the CUDA ones: fwht256 and kv_gather_q4_kernel use (32, 4) threads; the appends
// one 32-thread group per (head, block, step/token).
kernel void fwht256(device const float* src [[buffer(0)]], device float* dst [[buffer(1)]],
                    constant ulong& n_rows [[buffer(2)]], constant float& scale [[buffer(3)]],
                    uint3 tg [[threadgroup_position_in_grid]], uint3 tp [[thread_position_in_threadgroup]],
                    uint3 tptg [[threads_per_threadgroup]], ushort lane [[thread_index_in_simdgroup]]) {
    fwht256_lane(src, dst, n_rows, scale, (ulong)tg.x * tptg.y + tp.y, lane);
}

kernel void kv_append_q4_kernel(device uchar* k_q4 [[buffer(0)]], device uchar* v_q4 [[buffer(1)]],
                                device const int* table [[buffer(2)]], device const int* step [[buffer(3)]],
                                device const float* kcur [[buffer(4)]], device const float* vcur [[buffer(5)]],
                                device uchar* host_k [[buffer(6)]], device uchar* host_v [[buffer(7)]],
                                constant int& step_stride [[buffer(8)]], constant int& planes [[buffer(9)]],
                                constant int& kv_heads [[buffer(10)]], constant int& head_dim [[buffer(11)]],
                                constant int& page_size [[buffer(12)]],
                                uint3 tg [[threadgroup_position_in_grid]], ushort lane [[thread_index_in_simdgroup]]) {
    kv_append_q4_thread(k_q4, v_q4, table, step, step_stride, planes, kcur, vcur, kv_heads, head_dim, page_size,
                        host_k, host_v, tg, lane);
}

kernel void kv_append_q4_batch_kernel(device uchar* k_q4 [[buffer(0)]], device uchar* v_q4 [[buffer(1)]],
                                      device const int* table [[buffer(2)]], device const float* K [[buffer(3)]],
                                      device const float* V [[buffer(4)]],
                                      device uchar* host_k [[buffer(5)]], device uchar* host_v [[buffer(6)]],
                                      device uchar* stage_k [[buffer(7)]], device uchar* stage_v [[buffer(8)]],
                                      constant ulong& pos0 [[buffer(9)]], constant int& kv_heads [[buffer(10)]],
                                      constant int& head_dim [[buffer(11)]], constant int& page_size [[buffer(12)]],
                                      constant int& is_v_grid [[buffer(13)]],
                                      uint3 tg [[threadgroup_position_in_grid]],
                                      ushort lane [[thread_index_in_simdgroup]]) {
    kv_append_q4_batch_thread(k_q4, v_q4, table, pos0, K, V, kv_heads, head_dim, page_size, is_v_grid,
                              host_k, host_v, stage_k, stage_v, tg, lane);
}

kernel void kv_gather_q4_kernel(device const uchar* k_q4 [[buffer(0)]], device const uchar* v_q4 [[buffer(1)]],
                                device const int* table [[buffer(2)]], device const int* ids [[buffer(3)]],
                                device const int* step [[buffer(4)]],
                                device ushort* k_scratch [[buffer(5)]], device ushort* v_scratch [[buffer(6)]],
                                constant int& kv_heads [[buffer(7)]], constant int& head_dim [[buffer(8)]],
                                constant int& page_size [[buffer(9)]],
                                uint3 tg [[threadgroup_position_in_grid]], uint3 tp [[thread_position_in_threadgroup]],
                                uint3 tptg [[threads_per_threadgroup]]) {
    kv_gather_q4_thread(k_q4, v_q4, table, ids, step, kv_heads, head_dim, page_size, k_scratch, v_scratch,
                        (ulong)tg.x * tptg.y + tp.y, tp.x);
}

// ---- the engine twins: one argument struct at buffer 0 (strata/gpu/metal.hpp's dispatch(); the pointer fields
// carry gpuAddress values).  Mirrored in metal/src/engine/kv_q4_host.cpp - keep field order and types in step.
struct Fwht256Args {
    device const float* src;
    device float* dst;
    ulong n_rows;
    float scale;
};

struct KvAppendArgs {
    device uchar* k_q4;
    device uchar* v_q4;
    device const int* table;
    device const int* step;
    device const float* kcur;
    device const float* vcur;
    device uchar* host_k;
    device uchar* host_v;
    int step_stride;
    int planes;
    int kv_heads;
    int head_dim;
    int page_size;
};

struct KvAppendBatchArgs {
    device uchar* k_q4;
    device uchar* v_q4;
    device const int* table;
    device const float* K;
    device const float* V;
    device uchar* host_k;
    device uchar* host_v;
    device uchar* stage_k;
    device uchar* stage_v;
    ulong pos0;
    int kv_heads;
    int head_dim;
    int page_size;
    int is_v_grid;
};

struct KvGatherArgs {
    device const uchar* k_q4;
    device const uchar* v_q4;
    device const int* table;
    device const int* ids;
    device const int* step;
    device ushort* k_scratch;
    device ushort* v_scratch;
    int kv_heads;
    int head_dim;
    int page_size;
};

kernel void fwht256_e(constant Fwht256Args& a [[buffer(0)]],
                      uint3 tg [[threadgroup_position_in_grid]], uint3 tp [[thread_position_in_threadgroup]],
                      uint3 tptg [[threads_per_threadgroup]], ushort lane [[thread_index_in_simdgroup]]) {
    fwht256_lane(a.src, a.dst, a.n_rows, a.scale, (ulong)tg.x * tptg.y + tp.y, lane);
}

kernel void kv_append_q4_e(constant KvAppendArgs& a [[buffer(0)]],
                           uint3 tg [[threadgroup_position_in_grid]], ushort lane [[thread_index_in_simdgroup]]) {
    kv_append_q4_thread(a.k_q4, a.v_q4, a.table, a.step, a.step_stride, a.planes, a.kcur, a.vcur, a.kv_heads,
                        a.head_dim, a.page_size, a.host_k, a.host_v, tg, lane);
}

kernel void kv_append_q4_batch_e(constant KvAppendBatchArgs& a [[buffer(0)]],
                                 uint3 tg [[threadgroup_position_in_grid]],
                                 ushort lane [[thread_index_in_simdgroup]]) {
    kv_append_q4_batch_thread(a.k_q4, a.v_q4, a.table, a.pos0, a.K, a.V, a.kv_heads, a.head_dim, a.page_size,
                              a.is_v_grid, a.host_k, a.host_v, a.stage_k, a.stage_v, tg, lane);
}

kernel void kv_gather_q4_e(constant KvGatherArgs& a [[buffer(0)]],
                           uint3 tg [[threadgroup_position_in_grid]], uint3 tp [[thread_position_in_threadgroup]],
                           uint3 tptg [[threads_per_threadgroup]]) {
    kv_gather_q4_thread(a.k_q4, a.v_q4, a.table, a.ids, a.step, a.kv_heads, a.head_dim, a.page_size,
                        a.k_scratch, a.v_scratch, (ulong)tg.x * tptg.y + tp.y, tp.x);
}
