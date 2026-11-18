// metal/kernels/gemv_wide.metal - M1 / M2 only: the K-quant and IQ4_XS decoders with wider loads, bitwise the stock
// ones, through gemv.metal's templates.  Needs 16-byte aligned tensors (window.cpp checks).

#define T_Q4_KW 212
#define T_Q5_KW 213
#define T_Q6_KW 214
#define T_IQ4_XSW 223

// FW reads as F
#define WIDE_AS(FW, F)                                                                                             \
    template <> struct fmt<FW> { enum { QK = fmt<F>::QK, BB = fmt<F>::BB, UNIT = fmt<F>::UNIT, TAB = fmt<F>::TAB }; }; \
    template <> struct X<FW> {                                                                                     \
        template <typename P> static inline void load(P xp, P s16, int u, thread float* xs, thread float* xsum) { \
            X<F>::load(xp, s16, u, xs, xsum);                                                                      \
        }                                                                                                          \
    };                                                                                                             \
    template <> inline void load_tab<FW>(threadgroup uint* tab, ushort tid, ushort nth) { load_tab<F>(tab, tid, nth); }
WIDE_AS(T_Q4_KW, T_Q4_K) WIDE_AS(T_Q5_KW, T_Q5_K) WIDE_AS(T_Q6_KW, T_Q6_K) WIDE_AS(T_IQ4_XSW, T_IQ4_XS)

// scale_min_k4 on the header as one uint4
static inline void scale_min_k4_w(int j, uint4 hd, thread float& d, thread float& m) {
    const uint sh = 8 * uint(j & 3);
    const uint y = (hd.y >> sh) & 255u, z = (hd.z >> sh) & 255u, w = (hd.w >> sh) & 255u;
    d = float(j < 4 ? (y & 63u) : ((w & 0xFu) | ((y >> 6) << 4)));
    m = float(j < 4 ? (z & 63u) : ((w >> 4) | ((z >> 6) << 4)));
}

template <> struct Dec<T_Q4_KW> {
    static inline void run(device const uchar* row, int u, thread float* q, thread float4& sc, threadgroup const uint*) {
        device const uchar* b = row + (u >> 3) * 144;
        const int c = (u >> 1) & 3, m = u & 1;
        const uint4 hd = *(device const uint4*)b;
        const uint4 v4 = *(device const uint4*)(b + 16 + 32 * c + 16 * m);
        const half2 dd = as_type<half2>(hd.x);
        float s1, m1, s2, m2;
        scale_min_k4_w(2 * c, hd, s1, m1);
        scale_min_k4_w(2 * c + 1, hd, s2, m2);
        for (int w = 0; w < 4; ++w) {
            const uint v = v4[w];
            bytes4(v & 0x0F0F0F0Fu, q + 4 * w);
            bytes4((v >> 4) & 0x0F0F0F0Fu, q + 16 + 4 * w);
        }
        const float d = float(dd.x), dm = float(dd.y);
        sc = float4(d * s1, d * s2, dm * m1, dm * m2);
    }
};

template <> struct Dec<T_Q5_KW> {
    static inline void run(device const uchar* row, int u, thread float* q, thread float4& sc, threadgroup const uint*) {
        device const uchar* b = row + (u >> 3) * 176;
        const int c = (u >> 1) & 3, m = u & 1;
        const uint4 hd = *(device const uint4*)b;
        const uint4 h4 = *(device const uint4*)(b + 16 + 16 * m);
        const uint4 v4 = *(device const uint4*)(b + 48 + 32 * c + 16 * m);
        const half2 dd = as_type<half2>(hd.x);
        float s1, m1, s2, m2;
        scale_min_k4_w(2 * c, hd, s1, m1);
        scale_min_k4_w(2 * c + 1, hd, s2, m2);
        for (int w = 0; w < 4; ++w) {
            const uint v = v4[w], h = h4[w];
            bytes4((v & 0x0F0F0F0Fu) | (((h >> (2 * c)) & 0x01010101u) << 4), q + 4 * w);
            bytes4(((v >> 4) & 0x0F0F0F0Fu) | (((h >> (2 * c + 1)) & 0x01010101u) << 4), q + 16 + 4 * w);
        }
        const float d = float(dd.x), dm = float(dd.y);
        sc = float4(d * s1, d * s2, dm * m1, dm * m2);
    }
};

template <> struct Dec<T_Q6_KW> {
    static inline void run(device const uchar* row, int u, thread float* q, thread float4& sc, threadgroup const uint*) {
        device const uchar* b = row + (u >> 3) * 210;
        const int n = (u >> 2) & 1, m = (u >> 1) & 1, p = u & 1;
        // 210-byte blocks are only 2-aligned
        device const packed_ushort4* lv = (device const packed_ushort4*)(b + 64 * n + 32 * p + 16 * m);
        device const packed_ushort4* hv = (device const packed_ushort4*)(b + 128 + 32 * n + 16 * m);
        device const char* scs = (device const char*)(b + 192);
        const float d = rdh(b + 208);
        ushort lw[8], hw[8];
        for (int v = 0; v < 2; ++v) {
            const ushort4 a4 = ushort4(lv[v]), h4 = ushort4(hv[v]);
            lw[4 * v] = a4.x; lw[4 * v + 1] = a4.y; lw[4 * v + 2] = a4.z; lw[4 * v + 3] = a4.w;
            hw[4 * v] = h4.x; hw[4 * v + 1] = h4.y; hw[4 * v + 2] = h4.z; hw[4 * v + 3] = h4.w;
        }
        for (int w = 0; w < 8; ++w) {
            const ushort l = lw[w], h = hw[w];
            bytes2((l & 0x0F0F) | (((h >> (2 * p)) & 0x0303) << 4), q + 2 * w);
            bytes2(((l >> 4) & 0x0F0F) | (((h >> (2 * p + 4)) & 0x0303) << 4), q + 16 + 2 * w);
        }
        const float d1 = d * float(scs[8 * n + m + 2 * p]), d2 = d * float(scs[8 * n + m + 2 * p + 4]);
        sc = float4(d1, d2, 32.0f * d1, 32.0f * d2);
    }
};

// IQ4_XS: header and codes as uint2s
template <> inline float deq<T_IQ4_XSW>(device const uchar* b, int ui, thread float* q, threadgroup const uint* tab) {
    const uint2 hd = *(device const uint2*)b;
    device const uint2* qv = (device const uint2*)(b + 8 + 16 * ui);
    const uint2 a0 = qv[0], a1 = qv[1];
    const uint w[4] = {a0.x, a0.y, a1.x, a1.y};
    for (int j = 0; j < 8; ++j) {
        const uint v = (w[j >> 1] >> (16 * (j & 1))) & 0xFFFFu;
        const half2 w0 = as_type<half2>(tab[v & 255]), w1 = as_type<half2>(tab[v >> 8]);
        q[2 * j] = float(w0.x);
        q[2 * j + 16] = float(w0.y);
        q[2 * j + 1] = float(w1.x);
        q[2 * j + 17] = float(w1.y);
    }
    const int ls = int((hd.y >> (4 * ui)) & 15) | int(((hd.x >> (16 + 2 * ui)) & 3) << 4);
    return float(as_type<half2>(hd.x).x) * float(ls - 32);
}

// instances: F the type, FW its wide pseudo type
#define GVW_(F, FW, L, R, S, K) template [[host_name("gemvw_" #F "_" #L "_" #R "_" #S "_" #K)]] kernel decltype(gemv<FW, L, R, S, K>) gemv<FW, L, R, S, K>;
#define GVW(F, FW, L, R, S, K) GVW_(F, FW, L, R, S, K)
#define GVXW_(F, FW, L, R, S) template [[host_name("gemvxw_" #F "_" #L "_" #R "_" #S)]] kernel decltype(gemv_x<FW, L, R, S>) gemv_x<FW, L, R, S>;
#define GVXW(F, FW, L, R, S) GVXW_(F, FW, L, R, S)
// the tiles gemv_tile picks on M1 / M2
#define GVW_SET(F, FW) GVW(F, FW, 16, 1, 4, 1) GVW(F, FW, 16, 2, 4, 1) GVW(F, FW, 32, 1, 4, 1) GVW(F, FW, 32, 2, 4, 1) \
                       GVW(F, FW, 32, 2, 2, 1) GVW(F, FW, 32, 1, 8, 2) GVW(F, FW, 32, 2, 4, 2) GVXW(F, FW, 16, 1, 8)
GVW_SET(T_Q4_K, T_Q4_KW) GVW_SET(T_Q5_K, T_Q5_KW) GVW_SET(T_Q6_K, T_Q6_KW) GVW_SET(T_IQ4_XS, T_IQ4_XSW)

// BF16 2,2 (stock decoder)
#define GV1(F, L, R, S, K) GV(F, L, R, S, K)
GV1(T_BF16, 32, 2, 2, 1)
