#define DPCT_PROFILING_ENABLED
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
// The sampler's candidates (included by kernels.cu): per verify row, this
// card's slice of the head's logits with the row's penalties applied
// (llama.cpp's penalties stage, as Strata applies it: repeat multiplies a
// non-positive logit and divides a positive one, then frequency x count and
// presence once), then its TOPK_N highest, ordered by logit (descending) and
// token id (ascending) on ties. A block per row: a radix select of the
// TOPK_N-th highest key, then everything at or above it sorted. More than
// TOPK_CAP at or above it (a tie at the threshold that wide: a row of equal
// logits) takes a second pass that picks the same TOPK_N, so the result never
// depends on how many tie. A NaN or inf among the logits sets err bit 16.
namespace {

constexpr int TOPK_THREADS = 1024, TOPK_CAP = 256;

__dpct_inline__ unsigned
tk_key(float v) { // order-preserving: a larger float, a larger key
    const unsigned u = sycl::bit_cast<unsigned int>(v);
    return (u & 0x80000000u) ? ~u : (u | 0x80000000u);
}


void k_topk_rows(float *__restrict__ logits, int n, int v0,
                 const int *__restrict__ pen_tok,
                 const int *__restrict__ pen_cnt, const int *__restrict__ pen_n,
                 int pen_stride, const float *__restrict__ pen_par,
                 float *__restrict__ out_v, int *__restrict__ out_i,
                 unsigned *err, unsigned *hist, unsigned &prefix_s,
                 unsigned &want_s, unsigned long long *cand, unsigned &n_cand,
                 unsigned *warp_n) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const int row = item_ct1.get_group(2), tid = item_ct1.get_local_id(2),
              lane = tid & 31;
    float* L = logits + (size_t) row * n;
    // a NaN or inf among the head's logits (before the penalties, which a request sets): err bit 16
    bool bad = false;
    for (int j = tid; j < n; j += TOPK_THREADS)
        bad = bad || !sycl::isfinite(L[j]);
    if (bad) dpct::atomic_fetch_or<sycl::access::address_space::generic_space>(
        err, 16u);
    
    item_ct1.barrier(); // every logit checked before any penalty changes one
    // the row's penalties on this card's tokens (each token once in the list)
    const float repeat = pen_par[row * 3], freq = pen_par[row * 3 + 1], present = pen_par[row * 3 + 2];   // the row's penalty parameters
    for (int i = tid; i < pen_n[row]; i += TOPK_THREADS) {
        const int tok = pen_tok[row * pen_stride + i] - v0, cnt = pen_cnt[row * pen_stride + i];
        if (tok < 0 || tok >= n || cnt <= 0) continue;
        float l = L[tok];
        l = l <= 0.f ? l * repeat : l / repeat;
        L[tok] = l - (float) cnt * freq - present;
    }
    
    item_ct1.barrier();
    // the TOPK_N-th highest key: four passes of 8 bits

    unsigned prefix = 0, mask = 0, want = TOPK_N;
    for (int shift = 24; shift >= 0; shift -= 8) {
        for (int b = tid; b < 256; b += TOPK_THREADS) hist[b] = 0;
        
        
        item_ct1.barrier();
        for (int jb = 0; jb < n; jb += TOPK_THREADS) {
            const int j = jb + tid;
            const unsigned k = j < n ? tk_key(L[j]) : 0u;
            const bool in = j < n && (k & mask) == prefix;
            const unsigned bin = in ? (k >> shift) & 255u : 256u;
#if FNX_HIP
            const unsigned peers = (unsigned) __match_any_sync(0xffffffffull, bin);   // HIP wants a 64-bit mask (wave32: the high half is empty)
#else
            const unsigned peers = dpct::match_any_over_sub_group(
                sycl::ext::oneapi::this_work_item::get_sub_group(), 0xffffffffu,
                bin);
#endif
            if (in && lane == dpct::ffs<int>(peers) - 1) dpct::atomic_fetch_add<
                sycl::access::address_space::generic_space>(
                &hist[bin], (unsigned)sycl::popcount(peers));
        }
        
        
        item_ct1.barrier();
        if (tid == 0) {
            unsigned acc = 0;
            int b = 255;
            for (; b > 0; --b) {
                if (acc + hist[b] >= want) break;
                acc += hist[b];
            }
            prefix_s = prefix | ((unsigned) b << shift);
            want_s = want - acc;
        }
        
        
        item_ct1.barrier();
        prefix = prefix_s;
        want = want_s;
        mask |= 255u << shift;
        
        
        item_ct1.barrier();
    }
    // everything at or above the threshold, then sorted: (key desc, id asc) as one 64-bit key, descending

    for (int i = tid; i < TOPK_CAP; i += TOPK_THREADS) cand[i] = 0ull;
    if (tid == 0) n_cand = 0;
    
    item_ct1.barrier();
    for (int j = tid; j < n; j += TOPK_THREADS) {
        const unsigned k = tk_key(L[j]);
        if (k >= prefix) {
            const unsigned s = dpct::atomic_fetch_add<
                sycl::access::address_space::generic_space>(&n_cand, 1u);
            if (s < TOPK_CAP) cand[s] = ((unsigned long long) k << 32) | (0xffffffffu - (unsigned) j);
        }
    }
    
    item_ct1.barrier();
    const bool wide = n_cand > TOPK_CAP;
    
    item_ct1.barrier(); // every thread has read n_cand before the wide pass
                        // resets it
    if (wide) {   // block-uniform. The first TOPK_N of the sorted set are every key above the threshold
        // (TOPK_N - want of them) and the `want` lowest ids at it: those exactly, the ties taken in id order a chunk at a
        // time (a block-wide rank from each warp's ballot), stopping once `want` are in

        const int warp = tid >> 5;
        for (int i = tid; i < TOPK_CAP; i += TOPK_THREADS) cand[i] = 0ull;
        if (tid == 0) n_cand = 0;
        
        
        item_ct1.barrier();
        for (int j = tid; j < n; j += TOPK_THREADS) {
            const unsigned k = tk_key(L[j]);
            if (k > prefix) cand[dpct::atomic_fetch_add<
                sycl::access::address_space::generic_space>(&n_cand, 1u)] =
                ((unsigned long long)k << 32) | (0xffffffffu - (unsigned)j);
        }
        const unsigned base = TOPK_N - want;
        unsigned taken = 0;
        for (int jb = 0; jb < n && taken < want; jb += TOPK_THREADS) {
            const int j = jb + tid;
            const bool eq = j < n && tk_key(L[j]) == prefix;
            const unsigned bal = sycl::reduce_over_group(
                sycl::ext::oneapi::this_work_item::get_sub_group(),
                (0xffffffffu &
                 (0x1 << sycl::ext::oneapi::this_work_item::get_sub_group()
                             .get_local_linear_id())) &&
                        eq
                    ? (0x1 << sycl::ext::oneapi::this_work_item::get_sub_group()
                                  .get_local_linear_id())
                    : 0,
                sycl::ext::oneapi::plus<>());
            if (lane == 0) warp_n[warp] = (unsigned)sycl::popcount(bal);
            
            
            item_ct1.barrier();
            unsigned before =
                         (unsigned)sycl::popcount(bal & ((1u << lane) - 1u)),
                     total = 0;
            for (int w = 0; w < TOPK_THREADS / 32; ++w) {
                before += w < warp ? warp_n[w] : 0u;
                total += warp_n[w];
            }
            if (eq && taken + before < want) cand[base + taken + before] = ((unsigned long long) prefix << 32) | (0xffffffffu - (unsigned) j);
            taken += total;
            
            
            item_ct1.barrier(); // warp_n read by every thread before the next
                                // chunk writes it
        }
        
        
        item_ct1.barrier();
    }
    for (int size = 2; size <= TOPK_CAP; size <<= 1)
        for (int stride = size >> 1; stride > 0; stride >>= 1) {
            if (tid < TOPK_CAP) {
                const int o = tid ^ stride;
                if (o > tid) {
                    const bool desc = (tid & size) == 0;
                    const unsigned long long a = cand[tid], b = cand[o];
                    if (desc ? a < b : a > b) {
                        cand[tid] = b;
                        cand[o] = a;
                    }
                }
            }
            
            
            item_ct1.barrier();
        }
    if (tid < TOPK_N) {
        const unsigned long long c = cand[tid];
        const int j = (int) (0xffffffffu - (unsigned) (c & 0xffffffffu));
        out_i[row * TOPK_N + tid] = v0 + j;
        out_v[row * TOPK_N + tid] = L[j];
    }
}

}  // namespace
