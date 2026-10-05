// GDN WY-recurrence kernel using rocWMMA tensor cores (gfx1100/RDNA3).
// Included inside namespace strata::prefill { namespace { ... } } after gdn_rec_wy_kernel.
// Assumes S, HK, HV, C, CB, RG, RPG, NCB, WY are defined in the enclosing scope.

// rocwmma must be included at GLOBAL scope (see gdn_rec_wy_wmma_rocwmma.hpp, which defines
// STRATA_GDN_WY_HAVE_ROCWMMA and runs the macro lift/restore dance). This header is included
// INSIDE namespace strata::prefill { namespace { ... } }, so the include here is only the
// guarded no-op that makes this header self-contained for reviewers.
#if !defined(STRATA_GDN_WY_HAVE_ROCWMMA)
#if defined(__HIPCC__) && defined(__has_include)
#if __has_include(<rocwmma/rocwmma.hpp>)
#define STRATA_GDN_WY_HAVE_ROCWMMA 1
#endif
#endif
#endif
#if defined(STRATA_GDN_WY_HAVE_ROCWMMA)
#include <rocwmma/rocwmma.hpp>
#endif

static_assert(WY % 16 == 0 && CB % 16 == 0 && S % 16 == 0, "GDN WY WMMA requires multiples of 16");

#if defined(STRATA_GDN_WY_HAVE_ROCWMMA)

__global__ void __launch_bounds__(CB * RG) gdn_rec_wy_wmma_kernel(
    float* __restrict__ state, const float* __restrict__ h,
    const float* __restrict__ gate, const float* __restrict__ beta,
    float* __restrict__ oc_out, int64_t T) {

    // LDS Budget (~61.4 KiB):
    // kt16 16384 B + qt16 16384 B (overlaid with P16 cols [0..64) after site 3) +
    // Sin16 8192 B (overlaid with pk32 [WY][CB] fp32 after site 3) +
    // U16 4096 B + A 16384 B (o32 [WY][CB] in the lower half and pq32 [WY][CB] in the upper
    // half after the solve, then S_out [S][CB] over the whole array after the oc write)
    // + dec ~260 B = ~61.4 KiB < 64 KiB.
    // rocWMMA 7.1.1 store_matrix_sync static-asserts fragment/pointer type equality (no
    // converting stores), so the fold results stay fp32 in LDS; the only fp16 tiles are GEMM
    // operand tiles (kt16, qt16/P16, Sin16, U16) - the recorded bound counts those roundings.
    __shared__ __half kt16[WY][S];
    __shared__ __half qt16[WY][S];
    __shared__ __half Sin16[S][CB];
    __shared__ __half U16[WY][CB];
    __shared__ float A[WY * WY]; // o32 lower half + pq32 upper half, then S_out over all of it
    __shared__ float dec[WY + 1];
    // Sin16 storage alias: pk32 lives here after site 3, and it doubles as the warp-private
    // f32 conversion scratch (4 x 256 floats) at site 5 - both dead by then
    float* const pf = (float*)Sin16;

    const int head = blockIdx.x / NCB, cb = blockIdx.x % NCB;
    const int c = threadIdx.x, rg = threadIdx.y, col = cb * CB + c, tid = rg * CB + c;
    const int qh = head % HK;
    float s[RPG];
    float* base = state + ((size_t) (rg * RPG) * HV + head) * S + col;
    const size_t rs = (size_t) HV * S;

    #pragma unroll
    for (int r = 0; r < RPG; ++r) s[r] = base[r * rs];

    for (int64_t t0 = 0; t0 < T; t0 += WY) {
        const int w = (int) min((int64_t) WY, T - t0);
        
        // Compute dec array
        if (threadIdx.x == 0) {
            dec[0] = 1.0f;
            for (int t = 0; t < w; ++t) dec[t + 1] = dec[t] * __expf(gate[(t0 + t) * HV + head]);
        }
        __syncthreads();

        // Load k and q into fp16 LDS, zero-padding rows >= w
        for (int t = 0; t < WY; ++t) {
            if (t < w) {
                const float* src_k = h + HK * S + qh * S + (t0 + t) * C;
                const float* src_q = h + (t0 + t) * C + qh * S;
                for (int i = tid; i < S; i += CB * RG) {
                    kt16[t][i] = __float2half_rn(src_k[i]);
                    qt16[t][i] = __float2half_rn(src_q[i]);
                }
            } else {
                for (int i = tid; i < S; i += CB * RG) {
                    kt16[t][i] = __float2half_rn(0.0f);
                    qt16[t][i] = __float2half_rn(0.0f);
                }
            }
        }
        
        // Build Sin16 from register state s
        // F1: Each thread writes ONLY its own column to avoid races
        for (int r = 0; r < RPG; ++r) {
            Sin16[rg * RPG + r][c] = __float2half_rn(s[r]);
        }
        __syncthreads();

        // Site 1: A-build GEMM (WY x WY, K=S)
        // Warp assignment: 4 warps, each handles 16 rows of A (M dim).
        // Fragment: A (row_major), B (col_major).
        {
            int warpId = tid >> 5;
            int row_start = warpId * 16;
            
            #pragma unroll
            for (int nt = 0; nt < WY / 16; ++nt) {
                rocwmma::fragment<rocwmma::accumulator, 16, 16, 16, float> acc_n;
                rocwmma::fill_fragment(acc_n, 0.0f);
                
                #pragma unroll
                for (int ks = 0; ks < S / 16; ++ks) {
                    rocwmma::fragment<rocwmma::matrix_a, 16, 16, 16, __half, rocwmma::row_major> a_frag;
                    rocwmma::fragment<rocwmma::matrix_b, 16, 16, 16, __half, rocwmma::col_major> b_frag;
                    
                    rocwmma::load_matrix_sync(a_frag, &kt16[row_start][ks * 16], S);
                    rocwmma::load_matrix_sync(b_frag, &kt16[nt * 16][ks * 16], S);
                    
                    rocwmma::mma_sync(acc_n, a_frag, b_frag, acc_n);
                }
                
                // Store to A LDS
                // A is float[WY*WY]. Row-major.
                // We store to A[row_start .. row_start+15][nt*16 .. nt*16+15]
                rocwmma::store_matrix_sync(&A[row_start * WY + nt * 16], acc_n, WY, rocwmma::mem_row_major);
            }
        }
        __syncthreads();

        // Site 1 Scale: A[t][s2] = beta * prod_gate * Araw
        for (int p = tid; p < WY * WY; p += CB * RG) {
            int t = p / WY, s2 = p % WY;
            if (t >= w) { A[p] = 0.0f; continue; }
            if (s2 > t) { A[p] = 0.0f; continue; }
            if (s2 == t) { A[p] = 1.0f; continue; }
            
            float b = beta[(t0 + t) * HV + head];
            float dec_ratio = 1.0f;
            for (int j = s2 + 1; j <= t; ++j) dec_ratio *= __expf(gate[(t0 + j) * HV + head]);
            A[p] = b * dec_ratio * A[p];
        }
        __syncthreads();

        // Site 2: rhs fold pk (WY x CB, K=S)
        // Warp assignment: 4 warps, each handles 16 rows of pk (M dim).
        // F4: Compute into per-nt accumulator fragments but DO NOT store yet.
        rocwmma::fragment<rocwmma::accumulator, 16, 16, 16, float> acc_pk[CB / 16];
        {
            int warpId = tid >> 5;
            int row_start = warpId * 16;
            
            #pragma unroll
            for (int nt = 0; nt < CB / 16; ++nt) {
                rocwmma::fill_fragment(acc_pk[nt], 0.0f);
                
                #pragma unroll
                for (int ks = 0; ks < S / 16; ++ks) {
                    rocwmma::fragment<rocwmma::matrix_a, 16, 16, 16, __half, rocwmma::row_major> a_frag;
                    // F7: matrix_b row_major for Sin16 (row-major [K][N] tile)
                    rocwmma::fragment<rocwmma::matrix_b, 16, 16, 16, __half, rocwmma::row_major> b_frag;
                    
                    rocwmma::load_matrix_sync(a_frag, &kt16[row_start][ks * 16], S);
                    rocwmma::load_matrix_sync(b_frag, &Sin16[ks * 16][nt * 16], CB);
                    
                    rocwmma::mma_sync(acc_pk[nt], a_frag, b_frag, acc_pk[nt]);
                }
            }
        }
        // No __syncthreads() here, Sin16 is still needed for Site 3

        // Site 3: pq fold (WY x CB, K=S)
        // Warp assignment: 4 warps, each handles 16 rows of pq (M dim).
        // F5: Accumulators in registers, no store.
        rocwmma::fragment<rocwmma::accumulator, 16, 16, 16, float> acc_pq[CB / 16];
        {
            int warpId = tid >> 5;
            int row_start = warpId * 16;
            
            #pragma unroll
            for (int nt = 0; nt < CB / 16; ++nt) {
                rocwmma::fill_fragment(acc_pq[nt], 0.0f);
                
                #pragma unroll
                for (int ks = 0; ks < S / 16; ++ks) {
                    rocwmma::fragment<rocwmma::matrix_a, 16, 16, 16, __half, rocwmma::row_major> a_frag;
                    // F7: matrix_b row_major for Sin16
                    rocwmma::fragment<rocwmma::matrix_b, 16, 16, 16, __half, rocwmma::row_major> b_frag;
                    
                    rocwmma::load_matrix_sync(a_frag, &qt16[row_start][ks * 16], S);
                    rocwmma::load_matrix_sync(b_frag, &Sin16[ks * 16][nt * 16], CB);
                    
                    rocwmma::mma_sync(acc_pq[nt], a_frag, b_frag, acc_pq[nt]);
                }
            }
        }
        __syncthreads();

        // F6: store pk32 into Sin16's storage (Sin16 is dead after site 3); acc_pq stays in
        // registers - pq32 lands in A's upper half only after the solve has finished reading A
        {
            int warpId = tid >> 5;
            int row_start = warpId * 16;

            #pragma unroll
            for (int nt = 0; nt < CB / 16; ++nt) {
                rocwmma::store_matrix_sync(&pf[row_start * CB + nt * 16], acc_pk[nt], CB, rocwmma::mem_row_major);
            }
        }
        __syncthreads();

        // Site 4: Forward substitution
        float uj[WY];
        for (int t = 0; t < w; ++t) {
            float vj = h[2 * HK * S + head * S + (t0 + t) * C + col];
            float b = beta[(t0 + t) * HV + head];
            float g = __expf(gate[(t0 + t) * HV + head]);
            // F8: rhs reads pk32 from the Sin16 storage overlay
            float rhs = b * (vj - g * dec[t] * pf[t * CB + c]);
            
            float acc = rhs;
            for (int s2 = 0; s2 < t; ++s2) {
                acc -= A[t * WY + s2] * uj[s2];
            }
            uj[t] = acc;
        }
        
        // Build U16
        // F9: Single writer guard
        if (rg == 0) {
            for (int t = 0; t < WY; ++t) {
                U16[t][c] = (t < w) ? __float2half_rn(uj[t]) : __float2half_rn(0.0f);
            }
        }
        __syncthreads();
        // pq32 -> A's upper half (the solve is done reading A; o32 occupies the lower half later)
        {
            int warpId = tid >> 5;
            int row_start = warpId * 16;

            #pragma unroll
            for (int nt = 0; nt < CB / 16; ++nt) {
                rocwmma::store_matrix_sync(&A[WY * CB + row_start * CB + nt * 16], acc_pq[nt], CB, rocwmma::mem_row_major);
            }
        }
        __syncthreads();

        // Site 5: P GEMM (WY x WY, K=S)
        // Warp assignment: 4 warps, each handles 16 rows of P (M dim).
        // Overlay qt16 with P16. qt16 is dead after Site 3 - BUT the mma loads read the whole
        // qt16 row (ld=S) every nt iteration, so ALL K-loops must finish before ANY P16 store
        // lands: K outer, nt inner, one accumulator fragment per nt tile (the per-nt-then-K
        // order self-clobbered a_frag with freshly stored P16 rows - the T=1 oracle FAIL class).
        {
            int warpId = tid >> 5;
            int row_start = warpId * 16;

            rocwmma::fragment<rocwmma::accumulator, 16, 16, 16, float> acc[WY / 16];
            #pragma unroll
            for (int nt = 0; nt < WY / 16; ++nt) rocwmma::fill_fragment(acc[nt], 0.0f);

            #pragma unroll
            for (int ks = 0; ks < S / 16; ++ks) {
                rocwmma::fragment<rocwmma::matrix_a, 16, 16, 16, __half, rocwmma::row_major> a_frag;
                rocwmma::load_matrix_sync(a_frag, &qt16[row_start][ks * 16], S);
                #pragma unroll
                for (int nt = 0; nt < WY / 16; ++nt) {
                    // F7: col_major is correct for kt16-transposed trick here
                    rocwmma::fragment<rocwmma::matrix_b, 16, 16, 16, __half, rocwmma::col_major> b_frag;
                    rocwmma::load_matrix_sync(b_frag, &kt16[nt * 16][ks * 16], S);
                    rocwmma::mma_sync(acc[nt], a_frag, b_frag, acc[nt]);
                }
            }

            // Store P16: rocWMMA 7.1.1 has no converting store, so each f32 fragment lands in
            // a warp-private 16x16 region of the dead pk32 scratch and is hand-converted into
            // the fp16 tile (same row-major layout, ld=S).
            #pragma unroll
            for (int nt = 0; nt < WY / 16; ++nt) {
                __syncthreads();   // scratch free (previous nt) + everyone past the mma loads
                rocwmma::store_matrix_sync(&pf[warpId * 256], acc[nt], 16, rocwmma::mem_row_major);
                __syncthreads();
                for (int i = (tid & 31); i < 256; i += 32)   // this warp's own 16x16 scratch region
                    qt16[row_start + i / 16][nt * 16 + i % 16] = __float2half_rn(pf[warpId * 256 + i]);
                __syncthreads();
            }
        }
        __syncthreads();

        // Site 5 Scale: P16[t][s2] = ratio * P16[t][s2]
        // P16 is in qt16.
        for (int p = tid; p < WY * WY; p += CB * RG) {
            int t = p / WY, s2 = p % WY;
            if (t >= w) { qt16[t][s2] = __float2half_rn(0.0f); continue; }
            if (s2 > t) { qt16[t][s2] = __float2half_rn(0.0f); continue; }
            
            float ratio = 1.0f;
            if (s2 < t) {
                for (int j = s2 + 1; j <= t; ++j) ratio *= __expf(gate[(t0 + j) * HV + head]);
            }
            // s2 == t is included with ratio 1
            float val = ratio * (float)qt16[t][s2];
            qt16[t][s2] = __float2half_rn(val);
        }
        __syncthreads();

        // Site 6: O_intra GEMM (WY x CB, K=WY)
        // Warp assignment: 4 warps, each handles 16 rows of O (M dim).
        // Overlay A with o32. A is dead after Site 4.
        {
            int warpId = tid >> 5;
            int row_start = warpId * 16;
            
            #pragma unroll
            for (int nt = 0; nt < CB / 16; ++nt) {
                rocwmma::fragment<rocwmma::accumulator, 16, 16, 16, float> acc;
                rocwmma::fill_fragment(acc, 0.0f);
                
                #pragma unroll
                for (int ks = 0; ks < WY / 16; ++ks) {
                    rocwmma::fragment<rocwmma::matrix_a, 16, 16, 16, __half, rocwmma::row_major> a_frag;
                    // F7: matrix_b row_major for U16 (row-major [K][N] tile)
                    rocwmma::fragment<rocwmma::matrix_b, 16, 16, 16, __half, rocwmma::row_major> b_frag;
                    
                    // A is P16 (qt16). B is U16.
                    rocwmma::load_matrix_sync(a_frag, &qt16[row_start][ks * 16], S);
                    rocwmma::load_matrix_sync(b_frag, &U16[ks * 16][nt * 16], CB);
                    
                    rocwmma::mma_sync(acc, a_frag, b_frag, acc);
                }
                rocwmma::store_matrix_sync(&A[row_start * CB + nt * 16], acc, CB, rocwmma::mem_row_major);
            }
        }
        __syncthreads();

        // Write oc
        if (rg == 0) {
            for (int t = 0; t < w; ++t) {
                // F10: pq32 lives in A's upper half
                float o = dec[t + 1] * A[WY * CB + t * CB + c] + A[t * CB + c];
                oc_out[(t0 + t) * HV * S + head * S + col] = o * rsqrtf((float) S);
            }
        }
        // F11: NEW WAR barrier
        __syncthreads();

        // State fold
        // Rescale U16 in place
        // F13: Single writer guard
        if (rg == 0) {
            for (int t = 0; t < WY; ++t) {
                if (t < w) {
                    float dec_ratio = 1.0f;
                    for (int j = t + 1; j < w; ++j) dec_ratio *= __expf(gate[(t0 + j) * HV + head]);
                    U16[t][c] = __float2half_rn(dec_ratio * (float)U16[t][c]);
                }
            }
        }
        __syncthreads();

        // State GEMM (S x CB, K=WY)
        // Warp assignment: 4 warps x TWO 16-row M-tiles each = all 128 state rows
        // (a single 16-row tile per warp covered only half the state - the oracle caught it).
        // Matrix A is kt16^T (S x WY). Matrix B is U16 (WY x CB).
        {
            int warpId = tid >> 5;
            int row_start = warpId * 32; // 4 warps * 32 rows = 128 rows

            #pragma unroll
            for (int nt = 0; nt < CB / 16; ++nt) {
                rocwmma::fragment<rocwmma::accumulator, 16, 16, 16, float> acc[2];
                rocwmma::fill_fragment(acc[0], 0.0f);
                rocwmma::fill_fragment(acc[1], 0.0f);

                #pragma unroll
                for (int ks = 0; ks < WY / 16; ++ks) {
                    // F7: matrix_b row_major for U16; matrix_a col_major = the kt16^T view
                    rocwmma::fragment<rocwmma::matrix_a, 16, 16, 16, __half, rocwmma::col_major> a_frag;
                    rocwmma::fragment<rocwmma::matrix_b, 16, 16, 16, __half, rocwmma::row_major> b_frag;

                    // A[m][k] = kt16[ks*16 + k][row_start + m] (col_major, ld=S)
                    rocwmma::load_matrix_sync(a_frag, &kt16[ks * 16][row_start], S);
                    rocwmma::load_matrix_sync(b_frag, &U16[ks * 16][nt * 16], CB);
                    rocwmma::mma_sync(acc[0], a_frag, b_frag, acc[0]);
                    rocwmma::load_matrix_sync(a_frag, &kt16[ks * 16][row_start + 16], S);
                    rocwmma::mma_sync(acc[1], a_frag, b_frag, acc[1]);
                }

                // F12: Store S_out into A's storage (o32 is dead after the oc write)
                // A's storage holds: o32 [WY][CB] then S_out [S][CB] - both fit, barrier-separated
                rocwmma::store_matrix_sync(&A[row_start * CB + nt * 16], acc[0], CB, rocwmma::mem_row_major);
                rocwmma::store_matrix_sync(&A[(row_start + 16) * CB + nt * 16], acc[1], CB, rocwmma::mem_row_major);
            }
        }
        __syncthreads();

        // Update state registers
        // F12: s[r] = dec[w] * s[r] + A[(rg * RPG + r) * CB + c];
        for (int r = 0; r < RPG; ++r) {
            int i = rg * RPG + r;
            s[r] = dec[w] * s[r] + A[i * CB + c];
        }
        __syncthreads();
    }

    #pragma unroll
    for (int r = 0; r < RPG; ++r) base[r * rs] = s[r];
}

bool gdn_wy_wmma_supported() {
    static int arch[64] = {};
    int dev = 0;
    if (hipGetDevice(&dev) != hipSuccess || dev < 0 || dev >= 64) { hipGetLastError(); return false; }
    if (arch[dev] == 0) {
        hipDeviceProp_t p{};
        if (hipGetDeviceProperties(&p, dev) != hipSuccess) { hipGetLastError(); return false; }
        arch[dev] = (p.gcnArchName[0] == 'g' && p.gcnArchName[1] == 'f' && p.gcnArchName[2] == 'x' &&
                     p.gcnArchName[3] == '1' && p.gcnArchName[4] == '1' && p.gcnArchName[5] == '0' &&
                     p.gcnArchName[6] == '0') ? 1 : -1;
    }
    return arch[dev] == 1;
}

#else

__global__ void __launch_bounds__(CB * RG) gdn_rec_wy_wmma_kernel(
    float* __restrict__ state, const float* __restrict__ h,
    const float* __restrict__ gate, const float* __restrict__ beta,
    float* __restrict__ oc_out, int64_t T) {
    // Stub
}

bool gdn_wy_wmma_supported() {
    return false;
}

#endif
