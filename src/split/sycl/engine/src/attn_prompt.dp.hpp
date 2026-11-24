#define DPCT_PROFILING_ENABLED
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
// Attention over a prompt chunk (included by kernels.cu after the decode
// attention): one block per (row, KV head) walks the row's attended positions
// on this card, APC at a time, for the KV head's ATT_QPK query heads (rows of a
// 16-row tile) on the tensor cores: S = q k^T and O += p v as f16 x f16 with
// f32 accumulation, q and p rounded to f16 as llama.cpp's CUDA attention takes
// them. Online softmax per head. The partial (acc, max, sum) of every head goes
// to part[owner card][row][head of ATT_OWN][HD + 2]; the merge across cards
// (k_pf_attn_merge, after the exchange) applies the gate.
namespace {

constexpr int APC = 32;                  // positions per tile
constexpr int AP_NPT = APC / 16;         // their 16-position tiles
constexpr int AP_THREADS = 256;
constexpr int AP_LD = HD + 8;            // halves per Q / K / V row in shared memory (a multiple of 8, for wmma)
constexpr int AP_LDF = HD + 4;           // floats per row of the f32 staging tile
constexpr int AP_PLD = APC + 8;          // halves per P row
constexpr int AP_W = HD + 2;             // floats per partial: acc, max, sum

constexpr size_t AP_SMEM =
    sizeof(sycl::half) * (16 * AP_LD + APC * AP_LD + 16 * AP_PLD) +
    sizeof(float) * (16 * AP_LDF + 3 * 16);
static_assert(ATT_QPK <= 16, "a KV head's query heads in one 16-row tile");
static_assert(AP_THREADS / 32 * 2 * 16 == HD, "eight warps of two 16-column O tiles");
static_assert(AP_THREADS / 32 * APC <= AP_LDF && APC % 16 == 0, "the S partials of a tile side by side in st");


void k_pf_attn(const float *__restrict__ qg, const sycl::half *__restrict__ kc,
               const sycl::half *__restrict__ vc, KvPages g,
               const int *__restrict__ pos0p, const int *__restrict__ cells,
               const int *__restrict__ n_cells, int card, int C, int r0,
               float *__restrict__ part, uint8_t *dpct_local) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();

    auto ap_smem = (unsigned char *)dpct_local;
    sycl::half *qh = reinterpret_cast<sycl::half *>(ap_smem); // [16][AP_LD]
    sycl::half *kv = qh + 16 * AP_LD;  // [APC][AP_LD]: K, then V
    sycl::half *ph = kv + APC * AP_LD; // [16][AP_PLD]
    float* st = reinterpret_cast<float*>(ph + 16 * AP_PLD);     // [16][AP_LDF]: S partials, then the P V tile
    float* sm = st + 16 * AP_LDF;                               // [16]: running max
    float* sz = sm + 16;                                        // [16]: running sum
    float* ssc = sz + 16;                                       // [16]: this tile's rescale
    const int t = r0 + item_ct1.get_group(2), kvh = item_ct1.get_group(1),
              tid = item_ct1.get_local_id(2), warp = tid >> 5;
    const int* list = cells && n_cells[t] >= 0 ? cells + (size_t) t * QSA_MAXSEL : nullptr;
    const int ctx = list ? n_cells[t] : kv_owned(*pos0p + t, card);
    // q rows (heads kvh * QPK + j) as f16; rows past QPK are zero
    for (int e = tid; e < 16 * HD; e += AP_THREADS) {
        const int r = e / HD, d = e % HD;
        const float x = r < ATT_QPK ? qg[((size_t) t * NQH + kvh * ATT_QPK + r) * 2 * HD + d] : 0.f;
        qh[r * AP_LD + d] =
            sycl::vec<float, 1>(x)
                .convert<sycl::half, sycl::rounding_mode::rte>()[0];
    }
    if (tid < 16) {
        sm[tid] = -INFINITY;
        sz[tid] = 0.f;
    }
    // this thread's O elements: row orow, columns ocol + 16 j
    const int orow = tid / 16, ocol = tid % 16;
    float o[16];
#pragma unroll
    for (int j = 0; j < 16; ++j) o[j] = 0.f;
    const float scale = sycl::rsqrt((float)HD);
    // a tile's K or V rows, 8 halves a load, gathered into registers ahead of their use: V during S and the softmax,
    // the next tile's K during P V (each a load of the position's index, then of the row)
    constexpr int AP_GL = APC * HD / 8 / AP_THREADS;
    static_assert(APC * HD / 8 % AP_THREADS == 0, "a tile's loads spread evenly");
    auto gather = [&](const sycl::half *base, int c, int nr,
                      sycl::uint4(&r)[AP_GL]) {
#pragma unroll
        for (int i = 0; i < AP_GL; ++i) {
            const int e = tid + i * AP_THREADS, pp = e / (HD / 8), d8 = e % (HD / 8);
            r[i] = sycl::uint4(0, 0, 0, 0);
            if (pp < nr) {
                const int pos = list ? list[c + pp] : c + pp;
                if (g.q8) {   // 8 int8 values and their group's scale, as 8 halves
                    const size_t at = kvq_row(g, pos) + (size_t) kvh * HD + d8 * 8;
                    const sycl::uint2 q =
                        *reinterpret_cast<const sycl::uint2 *>(
                            reinterpret_cast<const int8_t *>(base) + at);
                    const float d =
                        sycl::vec<sycl::half, 1>(
                            base[kvq_scale(g, pos) +
                                 ((size_t)kvh * HD + d8 * 8) / KV8_GROUP])
                            .convert<float,
                                     sycl::rounding_mode::automatic>()[0];
                    const int8_t* qb = reinterpret_cast<const int8_t*>(&q);
                    __dpct_align__(16) sycl::half h[8];
#pragma unroll
                    for (int k = 0; k < 8; ++k) h[k] =
                        sycl::vec<float, 1>(qb[k] * d)
                            .convert<sycl::half, sycl::rounding_mode::rte>()[0];
                    r[i] = *reinterpret_cast<const sycl::uint4 *>(h);
                } else {
                    r[i] = *reinterpret_cast<const sycl::uint4 *>(
                        base + kvp_row(g, pos) + (size_t)kvh * HD + d8 * 8);
                }
            }
        }
    };
    auto place = [&](const sycl::uint4(&r)[AP_GL]) {
#pragma unroll
        for (int i = 0; i < AP_GL; ++i) {
            const int e = tid + i * AP_THREADS, pp = e / (HD / 8), d8 = e % (HD / 8);
            *reinterpret_cast<sycl::uint4 *>(kv + pp * AP_LD + d8 * 8) = r[i];
        }
    };
    sycl::uint4 kr[AP_GL], vr[AP_GL];
    if (ctx > 0) gather(kc, 0, sycl::min(APC, ctx), kr);
    for (int c0 = 0; c0 < ctx; c0 += APC) {
        const int np = sycl::min(APC, ctx - c0);
        
        
        item_ct1.barrier(); // the last tile's V and P reads are done
        place(kr);
        gather(vc, c0, np, vr);
        
        
        item_ct1.barrier();
        // S partials: warp w takes dimensions [32 w, 32 w + 32) (two k-steps) for each 16-position tile
        {
            
            dpct::experimental::matrix::joint_matrix<
                dpct::experimental::matrix::a, 16, 16, 16, sycl::half,
                dpct::experimental::matrix::row_major>
                fa[2];
            
            dpct::experimental::matrix::joint_matrix<
                dpct::experimental::matrix::b, 16, 16, 16, sycl::half,
                dpct::experimental::matrix::col_major>
                fb;
            
            dpct::experimental::matrix::joint_matrix<
                dpct::experimental::matrix::accumulator, 16, 16, 16, float>
                fs[AP_NPT];
#pragma unroll
            for (int ks = 0; ks < 2; ++ks)
                sycl::ext::oneapi::experimental::matrix::joint_matrix_load(
                    sycl::ext::oneapi::this_work_item::get_sub_group(),
                    fa[ks].get(),
                    sycl::address_space_cast<
                        sycl::access::address_space::generic_space,
                        sycl::access::decorated::no, const sycl::half>(
                        qh + warp * 32 + ks * 16),
                    AP_LD);
#pragma unroll
            for (int pt = 0; pt < AP_NPT; ++pt) {
                sycl::ext::oneapi::experimental::matrix::joint_matrix_fill(
                    sycl::ext::oneapi::this_work_item::get_sub_group(),
                    fs[pt].get(), 0.f);
#pragma unroll
                for (int ks = 0; ks < 2; ++ks) {
                    sycl::ext::oneapi::experimental::matrix::joint_matrix_load(
                        sycl::ext::oneapi::this_work_item::get_sub_group(),
                        fb.get(),
                        sycl::address_space_cast<
                            sycl::access::address_space::generic_space,
                            sycl::access::decorated::no, const sycl::half>(
                            kv + pt * 16 * AP_LD + warp * 32 + ks * 16),
                        AP_LD);
                    sycl::ext::oneapi::experimental::matrix::joint_matrix_mad(
                        sycl::ext::oneapi::this_work_item::get_sub_group(),
                        fs[pt].get(), fa[ks].get(), fb.get(), fs[pt].get());
                }
                // the eight warps' partials side by side in st: warp w at columns [APC w, APC w + APC)
                sycl::ext::oneapi::experimental::matrix::joint_matrix_store(
                    sycl::ext::oneapi::this_work_item::get_sub_group(),
                    fs[pt].get(),
                    sycl::address_space_cast<
                        sycl::access::address_space::generic_space,
                        sycl::access::decorated::no, float>(st + warp * APC +
                                                            pt * 16),
                    AP_LDF,
                    sycl::ext::oneapi::experimental::matrix::layout::row_major);
            }
        }
        
        
        item_ct1.barrier();
        // the scores: thread (row r, positions c + 16 u) sums the eight partials; the row's online-softmax step (16
        // lanes a row)
        {
            const int r = tid / 16, c = tid % 16;
            float sv[AP_NPT], mx = -INFINITY;
#pragma unroll
            for (int u = 0; u < AP_NPT; ++u) {
                float sacc = 0.f;
#pragma unroll
                for (int w = 0; w < AP_THREADS / 32; ++w) sacc += st[r * AP_LDF + w * APC + u * 16 + c];
                sv[u] = c + 16 * u < np ? sacc * scale : -INFINITY;
                mx = sycl::fmax(mx, sv[u]);
            }
#pragma unroll
            for (int off = 8; off; off >>= 1) mx = sycl::fmax(
                mx, dpct::permute_sub_group_by_xor(
                        sycl::ext::oneapi::this_work_item::get_sub_group(), mx,
                        off, 16));
            const float mo = sm[r], mn = sycl::fmax(mo, mx);
            float ps = 0.f;
#pragma unroll
            for (int u = 0; u < AP_NPT; ++u) {
                const float p =
                    sv[u] == -INFINITY ? 0.f : sycl::native::exp(sv[u] - mn);
                ps += p;
                ph[r * AP_PLD + u * 16 + c] =
                    sycl::vec<float, 1>(p)
                        .convert<sycl::half, sycl::rounding_mode::rte>()[0];
            }
#pragma unroll
            for (int off = 8; off; off >>= 1) ps +=
                dpct::permute_sub_group_by_xor(
                    sycl::ext::oneapi::this_work_item::get_sub_group(), ps, off,
                    16);
            sycl::group_barrier(
                sycl::ext::oneapi::this_work_item::get_sub_group());
            if (c == 0) {
                const float sc =
                    mo == -INFINITY ? 0.f : sycl::native::exp(mo - mn);
                ssc[r] = sc;
                sz[r] = sz[r] * sc + ps;
                sm[r] = mn;
            }
        }
        // V rows of the tile into the K buffer (its S reads are done: the barrier orders them)
        
        
        item_ct1.barrier();
        place(vr);
        
        
        item_ct1.barrier();
        if (c0 + APC < ctx)
            gather(kc, c0 + APC, sycl::min(APC, ctx - c0 - APC), kr);
        // P V: warp w takes O columns [32 w, 32 w + 32) (two 16-column tiles), the tile's positions in k-steps of 16
        {
            
            dpct::experimental::matrix::joint_matrix<
                dpct::experimental::matrix::a, 16, 16, 16, sycl::half,
                dpct::experimental::matrix::row_major>
                fp[AP_NPT];
            
            dpct::experimental::matrix::joint_matrix<
                dpct::experimental::matrix::b, 16, 16, 16, sycl::half,
                dpct::experimental::matrix::row_major>
                fv;
            
            dpct::experimental::matrix::joint_matrix<
                dpct::experimental::matrix::accumulator, 16, 16, 16, float>
                fo;
#pragma unroll
            for (int ks = 0; ks < AP_NPT; ++ks)
                sycl::ext::oneapi::experimental::matrix::joint_matrix_load(
                    sycl::ext::oneapi::this_work_item::get_sub_group(),
                    fp[ks].get(),
                    sycl::address_space_cast<
                        sycl::access::address_space::generic_space,
                        sycl::access::decorated::no, const sycl::half>(ph +
                                                                       ks * 16),
                    AP_PLD);
#pragma unroll
            for (int ct = 0; ct < 2; ++ct) {
                const int d0 = warp * 32 + ct * 16;
                sycl::ext::oneapi::experimental::matrix::joint_matrix_fill(
                    sycl::ext::oneapi::this_work_item::get_sub_group(),
                    fo.get(), 0.f);
#pragma unroll
                for (int ks = 0; ks < AP_NPT; ++ks) {
                    sycl::ext::oneapi::experimental::matrix::joint_matrix_load(
                        sycl::ext::oneapi::this_work_item::get_sub_group(),
                        fv.get(),
                        sycl::address_space_cast<
                            sycl::access::address_space::generic_space,
                            sycl::access::decorated::no, const sycl::half>(
                            kv + ks * 16 * AP_LD + d0),
                        AP_LD);
                    sycl::ext::oneapi::experimental::matrix::joint_matrix_mad(
                        sycl::ext::oneapi::this_work_item::get_sub_group(),
                        fo.get(), fp[ks].get(), fv.get(), fo.get());
                }
                sycl::ext::oneapi::experimental::matrix::joint_matrix_store(
                    sycl::ext::oneapi::this_work_item::get_sub_group(),
                    fo.get(),
                    sycl::address_space_cast<
                        sycl::access::address_space::generic_space,
                        sycl::access::decorated::no, float>(st + d0),
                    AP_LDF,
                    sycl::ext::oneapi::experimental::matrix::layout::row_major);
            }
        }
        
        
        item_ct1.barrier();
        const float sc = ssc[orow];
#pragma unroll
        for (int j = 0; j < 16; ++j) o[j] =
            sycl::fma(o[j], (float)sc, st[orow * AP_LDF + ocol + 16 * j]);
    }
    
    item_ct1.barrier();
    if (orow < ATT_QPK) {
        const int h = kvh * ATT_QPK + orow, k = h / ATT_OWN, hl = h % ATT_OWN;
        float* dst = part + (((size_t) k * C + t) * ATT_OWN + hl) * AP_W;
#pragma unroll
        for (int j = 0; j < 16; ++j) dst[ocol + 16 * j] = o[j];
        if (ocol == 0) {
            dst[HD] = sm[orow];
            dst[HD + 1] = sz[orow];
        }
    }
}

// this card's heads: its own partial and the others' (from card g + 1 in slot 0, g + 2 in slot 1, ...), merged; the gate
void k_pf_attn_merge(const float* __restrict__ own, const float* __restrict__ recv,
                                                      const float* __restrict__ qg, float* __restrict__ out, int C, int card) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const int t = item_ct1.get_group(2), hl = item_ct1.get_group(1),
              i = item_ct1.get_local_id(2);
    const size_t at = ((size_t) t * ATT_OWN + hl) * AP_W;
    float m = own[at + HD], z = own[at + HD + 1], a = own[at + i];
#pragma unroll
    for (int k = 0; k < NCARD - 1; ++k) {
        const float* src = recv + (size_t) k * C * ATT_OWN * AP_W + at;
        att_merge(m, z, a, src[HD], src[HD + 1], src[i]);
    }
    const int hg = card * ATT_OWN + hl;
    out[((size_t) t * ATT_OWN + hl) * HD + i] = a / z * sigm(qg[((size_t) t * NQH + hg) * 2 * HD + HD + i]);
}

}  // namespace
