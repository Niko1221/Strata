// metal/src/engine/msl/ds4_mm.metal - the DeepSeek prompt chunk's expert GEMMs: moe_mm.metal's mm_gu3 / mm_down3 (matmul2d on dequantized
// weight tiles) over the engine's own expert slots and its UNPADDED sort order.  Compiled after quant/units/mma/moe_mm.metal.
//   tiles[i] = (slot, first row, valid rows): a tile is TM rows of one expert; rows past `valid` (the next expert's) are computed and
//   dropped (the operand rows exist: the scratch carries TM rows of slack), the swiglu is DeepSeek's clamped one.

struct DsMmArgs {
    ulong gate_off;
    ulong up_off;
    ulong down_off;
    ulong stride;    // bytes between slots
    uint gu_row;     // bytes per gate/up row (n_embd values)
    uint d_row;      // bytes per down row (n_ff values)
    uint n_embd;
    uint n_ff;
    float limit;
    uint pad0;
};

// MXFP4 (39) for the tiles: a low-bit layer's MXFP4 down (UD-Q2_K_XL's 7 mixed layers).  Kept here, not in quant.metal, so the
// Qwen kernels' sources stay as they were.  A 32-value block of 17 bytes: the E8M0 scale byte e, then 16 bytes whose low nibble is
// value j and whose high nibble value j + 16 (ggml's kvalues_mxfp4, the E2M1 values doubled); the weight is code * 2^(e - 128).
#define T_MXFP4 39
template <> struct fmt<T_MXFP4> { enum { QK = 32, BB = 17, UNIT = 32, TAB = 1 }; };
template <> inline void load_tab<T_MXFP4>(threadgroup uint*, ushort, ushort) {}
constant half ds4_mm_mx_code[16] = {0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12};
template <> struct W4<T_MXFP4> {   // values k .. k + 3 of a row (k % 4 == 0)
    static inline half4 run(device const uchar* row, int k, threadgroup const uint*) {
        device const uchar* b = row + (k >> 5) * 17;
        const uint e = b[0];
        const float d = as_type<float>(e < 2u ? (0x00200000u << e) : ((e - 1u) << 23));
        const int j = k & 31, sh = (j >> 4) * 4;
        device const uchar* q = b + 1 + (j & 15);
        const float4 w = float4(float(ds4_mm_mx_code[(q[0] >> sh) & 15]), float(ds4_mm_mx_code[(q[1] >> sh) & 15]),
                                float(ds4_mm_mx_code[(q[2] >> sh) & 15]), float(ds4_mm_mx_code[(q[3] >> sh) & 15]));
        return half4(w * d);
    }
};

// xs[r][c] = half(x[idx[r]][c]) for r < rows
kernel void ds4_mm_gather(device const float* x [[buffer(0)]], constant ulong& ldx [[buffer(1)]], device const int* idx [[buffer(2)]],
                          constant uint& n_embd [[buffer(3)]], device half* xs [[buffer(4)]],
                          uint2 tgp [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {
    const uint c = tgp.x * 256u + tid;
    if (c >= n_embd) return;
    xs[(ulong) tgp.y * n_embd + c] = half(x[(ulong) idx[tgp.y] * ldx + c]);
}

template <int F, int TM, int TN, int TK, int SG>
kernel void ds4_mm_gu(device const uchar* W [[buffer(0)]], device const half* XS [[buffer(1)]], device half* HS [[buffer(2)]],
                      device const int4* tiles [[buffer(3)]], constant DsMmArgs& a [[buffer(4)]], constant uint& n_tiles [[buffer(5)]],
                      uint3 tg [[threadgroup_position_in_grid]], ushort sg [[simdgroup_index_in_threadgroup]],
                      ushort lane [[thread_index_in_simdgroup]], ushort tid [[thread_index_in_threadgroup]]) {
    if (tg.y >= n_tiles) return;
    threadgroup float ot[SG][TM * TN];
    threadgroup uint tab[fmt<F>::TAB];
    load_tab<F>(tab, tid, SG * 32);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const int4 tl = tiles[tg.y];
    const int e = tl.x, row0 = tl.y, valid = tl.z, n0 = (int(tg.x) * SG + sg) * TN;
    device const uchar* E = W + ulong(e) * a.stride;
    device const uchar* g0 = E + a.gate_off + ulong(n0) * a.gu_row;
    device const uchar* u0 = E + a.up_off + ulong(n0) * a.gu_row;
    constexpr auto desc = matmul2d_descriptor(TM, TN, TK, false, true, false, matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<desc, execution_simdgroup> op;
    tensor<device half, dextents<int32_t, 2>, tensor_inline> tX((device half*)XS, dextents<int32_t, 2>(a.n_embd, 1 << 30));
    auto rg = op.template get_right_input_cooperative_tensor<half, half, float>();
    auto ru = op.template get_right_input_cooperative_tensor<half, half, float>();
    auto tA = tX.template slice<TK, TM>(0, row0);
    auto accg = op.template get_destination_cooperative_tensor<__tensor_ops_detail::__remove_addrspace_t<decltype(tA)>, __tensor_ops_detail::__remove_addrspace_t<decltype(rg)>, float>();
    auto accu = op.template get_destination_cooperative_tensor<__tensor_ops_detail::__remove_addrspace_t<decltype(tA)>, __tensor_ops_detail::__remove_addrspace_t<decltype(rg)>, float>();
    for (uint16_t i = 0; i < accg.get_capacity(); ++i)
        if (accg.is_valid_element(i)) { accg[i] = 0.0f; accu[i] = 0.0f; }
    for (int k0 = 0; k0 < int(a.n_embd); k0 += TK) {
        const uint16_t cap = rg.get_capacity(), half_cap = cap / 2;
#pragma clang loop unroll(full)
        for (uint16_t i = 0; i < half_cap; i += 4) {
            const auto ix = rg.get_multidimensional_index(i);
            const int k = k0 + ix[0], n = ix[1];
            device const uchar* rgw = g0 + ulong(n) * a.gu_row;
            device const uchar* ruw = u0 + ulong(n) * a.gu_row;
            half4 wg, wu, wg2, wu2;
            W8<F>::run(rgw, k, tab, wg, wg2);
            W8<F>::run(ruw, k, tab, wu, wu2);
            rg[i] = wg.x; rg[i + 1] = wg.y; rg[i + 2] = wg.z; rg[i + 3] = wg.w;
            ru[i] = wu.x; ru[i + 1] = wu.y; ru[i + 2] = wu.z; ru[i + 3] = wu.w;
            const uint16_t j = i + half_cap;
            rg[j] = wg2.x; rg[j + 1] = wg2.y; rg[j + 2] = wg2.z; rg[j + 3] = wg2.w;
            ru[j] = wu2.x; ru[j + 1] = wu2.y; ru[j + 2] = wu2.z; ru[j + 3] = wu2.w;
        }
        tA = tX.template slice<TK, TM>(k0, row0);
        op.run(tA, rg, accg);
        op.run(tA, ru, accu);
    }
    for (uint16_t i = 0; i < accg.get_capacity(); ++i)
        if (accg.is_valid_element(i)) {
            const float g = fmin(accg[i], a.limit), u = fmin(fmax(accu[i], -a.limit), a.limit);
            accg[i] = g / (1.0f + exp(-g)) * u;
        }
    tensor<threadgroup float, dextents<int32_t, 2>, tensor_inline> tO(ot[sg], dextents<int32_t, 2>(TN, TM));
    accg.store(tO);
    simdgroup_barrier(mem_flags::mem_threadgroup);
    for (int it = lane; it < TM * TN; it += 32) {
        const int m = it / TN, f = it - m * TN;
        if (m < valid) HS[ulong(row0 + m) * a.n_ff + n0 + f] = half(ot[sg][it]);
    }
}

template <int F, int TM, int TN, int TK, int SG>
kernel void ds4_mm_dn(device const uchar* W [[buffer(0)]], device half* YEH [[buffer(1)]], device const half* HS [[buffer(2)]],
                      device const int4* tiles [[buffer(3)]], constant DsMmArgs& a [[buffer(4)]], constant uint& n_tiles [[buffer(5)]],
                      uint3 tg [[threadgroup_position_in_grid]], ushort sg [[simdgroup_index_in_threadgroup]],
                      ushort lane [[thread_index_in_simdgroup]], ushort tid [[thread_index_in_threadgroup]]) {
    if (tg.y >= n_tiles || a.pad0 == 3u) return;
    threadgroup float ot[SG][TM * TN];
    threadgroup uint tab[fmt<F>::TAB];
    load_tab<F>(tab, tid, SG * 32);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const int4 tl = tiles[tg.y];
    const int e = tl.x, row0 = tl.y, valid = tl.z, n0 = (int(tg.x) * SG + sg) * TN;
    device const uchar* w0 = W + ulong(e) * a.stride + a.down_off + ulong(n0) * a.d_row;
    constexpr auto desc = matmul2d_descriptor(TM, TN, TK, false, true, false, matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<desc, execution_simdgroup> op;
    tensor<device half, dextents<int32_t, 2>, tensor_inline> tH((device half*)HS, dextents<int32_t, 2>(a.n_ff, 1 << 30));
    auto rw = op.template get_right_input_cooperative_tensor<half, half, float>();
    auto tA = tH.template slice<TK, TM>(0, row0);
    auto acc = op.template get_destination_cooperative_tensor<__tensor_ops_detail::__remove_addrspace_t<decltype(tA)>, __tensor_ops_detail::__remove_addrspace_t<decltype(rw)>, float>();
    for (uint16_t i = 0; i < acc.get_capacity(); ++i)
        if (acc.is_valid_element(i)) acc[i] = 0.0f;
    for (int k0 = 0; k0 < int(a.n_ff); k0 += TK) {
        const uint16_t half_cap = rw.get_capacity() / 2;
#pragma clang loop unroll(full)
        for (uint16_t i = 0; i < half_cap; i += 4) {
            const auto ix = rw.get_multidimensional_index(i);
            half4 w, w2;
            if (a.pad0 == 1u) { w = half4(0); w2 = half4(0); } else if (a.pad0 == 2u) { w = half4(1); w2 = half4(1); W8<F>::run(w0, 0, tab, w, w2); w = half4(0); } else
            W8<F>::run(w0 + ulong(ix[1]) * a.d_row, k0 + ix[0], tab, w, w2);
            rw[i] = w.x; rw[i + 1] = w.y; rw[i + 2] = w.z; rw[i + 3] = w.w;
            const uint16_t j = i + half_cap;
            rw[j] = w2.x; rw[j + 1] = w2.y; rw[j + 2] = w2.z; rw[j + 3] = w2.w;
        }
        tA = tH.template slice<TK, TM>(k0, row0);
        op.run(tA, rw, acc);
    }
    tensor<threadgroup float, dextents<int32_t, 2>, tensor_inline> tO(ot[sg], dextents<int32_t, 2>(TN, TM));
    acc.store(tO);
    simdgroup_barrier(mem_flags::mem_threadgroup);
    for (int it = lane; it < TM * TN; it += 32) {
        const int m = it / TN, f = it - m * TN;
        if (m < valid) ((device float*) YEH)[ulong(row0 + m) * a.n_embd + n0 + f] = ot[sg][it];
    }
}

// ---- the same two GEMMs for a GPU whose matmul2d right-input layout is not the M5's (ds4_finish.cpp's probe; the M2 Max holds
// runs of 2 n at one k, where the paired fill above puts weights at the wrong (k, n) and every tile's result is wrong).  Each lane
// decodes one weight row's TK values in k order with the type's W8 into threadgroup memory (the simdgroup's own part of `ot`, free
// until the epilogue), and the matmul reads that tile as a (TK, TN) threadgroup tensor (moe_mm.metal's mm_gu does the same): no
// assumption about the layout, each weight decoded once.  TN = 32 (a lane a row) and TK = 32 (four W8 a row).
template <int TN, int TK, int F>
static inline void ds4_mm_stage(device const uchar* w0, uint row_bytes, int k0, threadgroup half* st, threadgroup const uint* tab,
                                ushort lane) {
    device const uchar* r = w0 + ulong(lane) * row_bytes;
    threadgroup half* d = st + int(lane) * TK;
    for (int kk = 0; kk < 16; kk += 4) {
        half4 a, b;
        W8<F>::run(r, k0 + kk, tab, a, b);
        d[kk] = a.x; d[kk + 1] = a.y; d[kk + 2] = a.z; d[kk + 3] = a.w;
        d[16 + kk] = b.x; d[16 + kk + 1] = b.y; d[16 + kk + 2] = b.z; d[16 + kk + 3] = b.w;
    }
}

template <int F, int TM, int TN, int TK, int SG>
kernel void ds4_mm_gus(device const uchar* W [[buffer(0)]], device const half* XS [[buffer(1)]], device half* HS [[buffer(2)]],
                       device const int4* tiles [[buffer(3)]], constant DsMmArgs& a [[buffer(4)]], constant uint& n_tiles [[buffer(5)]],
                       uint3 tg [[threadgroup_position_in_grid]], ushort sg [[simdgroup_index_in_threadgroup]],
                       ushort lane [[thread_index_in_simdgroup]], ushort tid [[thread_index_in_threadgroup]]) {
    static_assert(TN == 32 && TK == 32 && TM * TN >= TN * TK, "the staged tiles: a lane a row, and gate + up fit in ot");
    if (tg.y >= n_tiles) return;
    threadgroup float ot[SG][TM * TN];
    threadgroup uint tab[fmt<F>::TAB];
    load_tab<F>(tab, tid, SG * 32);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const int4 tl = tiles[tg.y];
    const int e = tl.x, row0 = tl.y, valid = tl.z, n0 = (int(tg.x) * SG + sg) * TN;
    device const uchar* E = W + ulong(e) * a.stride;
    device const uchar* g0 = E + a.gate_off + ulong(n0) * a.gu_row;
    device const uchar* u0 = E + a.up_off + ulong(n0) * a.gu_row;
    threadgroup half* sgt = (threadgroup half*) ot[sg];   // gate's TN x TK, then up's
    threadgroup half* sut = sgt + TN * TK;
    constexpr auto desc = matmul2d_descriptor(TM, TN, TK, false, true, false, matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<desc, execution_simdgroup> op;
    tensor<device half, dextents<int32_t, 2>, tensor_inline> tX((device half*)XS, dextents<int32_t, 2>(a.n_embd, 1 << 30));
    tensor<threadgroup half, dextents<int32_t, 2>, tensor_inline> tG(sgt, dextents<int32_t, 2>(TK, TN));
    tensor<threadgroup half, dextents<int32_t, 2>, tensor_inline> tU(sut, dextents<int32_t, 2>(TK, TN));
    auto tA = tX.template slice<TK, TM>(0, row0);
    auto accg = op.template get_destination_cooperative_tensor<decltype(tA), decltype(tG), float>();
    auto accu = op.template get_destination_cooperative_tensor<decltype(tA), decltype(tG), float>();
    for (uint16_t i = 0; i < accg.get_capacity(); ++i)
        if (accg.is_valid_element(i)) { accg[i] = 0.0f; accu[i] = 0.0f; }
    for (int k0 = 0; k0 < int(a.n_embd); k0 += TK) {
        ds4_mm_stage<TN, TK, F>(g0, a.gu_row, k0, sgt, tab, lane);
        ds4_mm_stage<TN, TK, F>(u0, a.gu_row, k0, sut, tab, lane);
        simdgroup_barrier(mem_flags::mem_threadgroup);
        tA = tX.template slice<TK, TM>(k0, row0);
        op.run(tA, tG, accg);
        op.run(tA, tU, accu);
        simdgroup_barrier(mem_flags::mem_threadgroup);   // (the next step overwrites the tiles)
    }
    for (uint16_t i = 0; i < accg.get_capacity(); ++i)
        if (accg.is_valid_element(i)) {
            const float g = fmin(accg[i], a.limit), u = fmin(fmax(accu[i], -a.limit), a.limit);
            accg[i] = g / (1.0f + exp(-g)) * u;
        }
    tensor<threadgroup float, dextents<int32_t, 2>, tensor_inline> tO(ot[sg], dextents<int32_t, 2>(TN, TM));
    accg.store(tO);
    simdgroup_barrier(mem_flags::mem_threadgroup);
    for (int it = lane; it < TM * TN; it += 32) {
        const int m = it / TN, f = it - m * TN;
        if (m < valid) HS[ulong(row0 + m) * a.n_ff + n0 + f] = half(ot[sg][it]);
    }
}

template <int F, int TM, int TN, int TK, int SG>
kernel void ds4_mm_dns(device const uchar* W [[buffer(0)]], device half* YEH [[buffer(1)]], device const half* HS [[buffer(2)]],
                       device const int4* tiles [[buffer(3)]], constant DsMmArgs& a [[buffer(4)]], constant uint& n_tiles [[buffer(5)]],
                       uint3 tg [[threadgroup_position_in_grid]], ushort sg [[simdgroup_index_in_threadgroup]],
                       ushort lane [[thread_index_in_simdgroup]], ushort tid [[thread_index_in_threadgroup]]) {
    static_assert(TN == 32 && TK == 32 && TM * TN * 2 >= TN * TK, "the staged tiles: a lane a row, and the down fits in ot");
    if (tg.y >= n_tiles) return;
    threadgroup float ot[SG][TM * TN];
    threadgroup uint tab[fmt<F>::TAB];
    load_tab<F>(tab, tid, SG * 32);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const int4 tl = tiles[tg.y];
    const int e = tl.x, row0 = tl.y, valid = tl.z, n0 = (int(tg.x) * SG + sg) * TN;
    device const uchar* w0 = W + ulong(e) * a.stride + a.down_off + ulong(n0) * a.d_row;
    threadgroup half* swt = (threadgroup half*) ot[sg];
    constexpr auto desc = matmul2d_descriptor(TM, TN, TK, false, true, false, matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<desc, execution_simdgroup> op;
    tensor<device half, dextents<int32_t, 2>, tensor_inline> tH((device half*)HS, dextents<int32_t, 2>(a.n_ff, 1 << 30));
    tensor<threadgroup half, dextents<int32_t, 2>, tensor_inline> tW(swt, dextents<int32_t, 2>(TK, TN));
    auto tA = tH.template slice<TK, TM>(0, row0);
    auto acc = op.template get_destination_cooperative_tensor<decltype(tA), decltype(tW), float>();
    for (uint16_t i = 0; i < acc.get_capacity(); ++i)
        if (acc.is_valid_element(i)) acc[i] = 0.0f;
    for (int k0 = 0; k0 < int(a.n_ff); k0 += TK) {
        ds4_mm_stage<TN, TK, F>(w0, a.d_row, k0, swt, tab, lane);
        simdgroup_barrier(mem_flags::mem_threadgroup);
        tA = tH.template slice<TK, TM>(k0, row0);
        op.run(tA, tW, acc);
        simdgroup_barrier(mem_flags::mem_threadgroup);
    }
    tensor<threadgroup float, dextents<int32_t, 2>, tensor_inline> tO(ot[sg], dextents<int32_t, 2>(TN, TM));
    acc.store(tO);
    simdgroup_barrier(mem_flags::mem_threadgroup);
    for (int it = lane; it < TM * TN; it += 32) {
        const int m = it / TN, f = it - m * TN;
        if (m < valid) ((device float*) YEH)[ulong(row0 + m) * a.n_embd + n0 + f] = ot[sg][it];
    }
}

#define DS4_MM_INST(F, T)                                                                                                          template [[host_name("ds4_mm_gu_" #T)]] kernel decltype(ds4_mm_gu<F, 32, 32, 32, 4>) ds4_mm_gu<F, 32, 32, 32, 4>;               template [[host_name("ds4_mm_dn_" #T)]] kernel decltype(ds4_mm_dn<F, 32, 32, 32, 4>) ds4_mm_dn<F, 32, 32, 32, 4>; \
    template [[host_name("ds4_mm_gus_" #T)]] kernel decltype(ds4_mm_gus<F, 32, 32, 32, 4>) ds4_mm_gus<F, 32, 32, 32, 4>; \
    template [[host_name("ds4_mm_dns_" #T)]] kernel decltype(ds4_mm_dns<F, 32, 32, 32, 4>) ds4_mm_dns<F, 32, 32, 32, 4>;
DS4_MM_INST(T_IQ1_M, 29)
DS4_MM_INST(T_IQ1_S, 19)
DS4_MM_INST(T_IQ2_XXS, 16)
DS4_MM_INST(T_IQ2_XS, 17)
DS4_MM_INST(T_IQ2_S, 22)
DS4_MM_INST(T_IQ3_XXS, 18)
DS4_MM_INST(T_IQ3_S, 21)
DS4_MM_INST(T_MXFP4, 39)
