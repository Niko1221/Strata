// DFlash2 equations, independently expressed as direct CUDA/HIP kernels.
// Reference: llama.cpp src/models/dflash.cpp (a9d27ac), z-lab/dflash (MIT, 07ebd93).
#include "strata/kernels/dflash2.hpp"
#include "strata/kernels/bf16_bits.hpp"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <stdexcept>

namespace strata::kernels {
namespace {
template<int Epilogue = 0>
__global__ void conv_kernel(const float* x, const float* dynamic, const float* base, float* out,
                            int rows, int hidden, int kernel, int group, int side, int block_rows, uint16_t* bf16 = nullptr) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= rows * hidden) return;
    const int row = i / hidden, channel = i % hidden, groups = hidden / group;
    float value = 0;
    const int limit = min(kernel, row % block_rows + 1);
    for (int tap = 0; tap < limit; ++tap) {
        const float weight = base[(side * kernel + tap) * hidden + channel] +
            dynamic[((row * 2 + side) * kernel + tap) * groups + channel / group];
        value += weight * x[(row - tap) * hidden + channel];
    }
    if constexpr (Epilogue == 1) bf16[i] = bf16_from_f32(value);
    else if constexpr (Epilogue == 2) {
#if defined(__HIPCC__) || defined(STRATA_USE_HIP)
        out[i] = out[i] + value;
#else
        out[i] = __fadd_rn(out[i], value);
#endif
    } else out[i] = value;
}
// Retained for same-input timing and parity (STRATA_DFLASH2_SERIAL_TOPK=1).
__global__ void topk_kernel(const float* logits, int32_t* ids, float* unary, int vocab, int top_k) {
    if (threadIdx.x != 0) return;
    const int row = blockIdx.x;
    ids += row * top_k; unary += row * top_k; logits += (size_t) row * vocab;
    for (int k = 0; k < top_k; ++k) { ids[k] = -1; unary[k] = -INFINITY; }
    for (int token = 0; token < vocab; ++token) {
        const float score = logits[token];
        // Non-finite model outputs must not manufacture a token ID.
        if (!isfinite(score)) continue;
        int at = top_k;
        while (at > 0 && score > unary[at - 1]) --at;
        if (at == top_k) continue;
        for (int k = top_k - 1; k > at; --k) { ids[k] = ids[k - 1]; unary[k] = unary[k - 1]; }
        ids[at] = token; unary[at] = score;
    }
}
// One CTA per row/tile. A second pass merges the tiles' winners using the same
// reduction; only tile_count * K candidates need temporary storage. IDs break ties.
constexpr int kTopkTile = 4096;
// Keep a tile's logits in registers rather than reloading them for every
// winner. Ordered float bits plus the reverse token ID encode the same
// score/ID comparison as the reference reduction. Inspired by llama.cpp #27342.
__global__ void topk_register_kernel(const float* logits, int32_t* ids, float* unary,
                                     int vocab, int top_k, int tiles) {
    const int row = blockIdx.x, tile = blockIdx.y, t = threadIdx.x;
    logits += (size_t) row * vocab;
    const int begin = tile * kTopkTile;
    unsigned long long keys[kTopkTile / 256];
    __shared__ unsigned long long winners[8];
#pragma unroll
    for (int i = 0; i < kTopkTile / 256; ++i) {
        const int token = begin + t + i * 256;
        unsigned long long key = 0;
        if (token < vocab) {
            const float score = logits[token];
            if (isfinite(score)) {
                // Both signs of zero compare equal in the reference kernel.
                unsigned bits = score == 0.0f ? 0 : __float_as_uint(score);
                bits = bits & 0x80000000u ? ~bits : bits | 0x80000000u;
                key = ((unsigned long long) bits << 32) | (unsigned) (vocab - 1 - token);
            }
        }
        keys[i] = key;
    }
    const size_t at = ((size_t) row * tiles + tile) * top_k;
    for (int k = 0; k < top_k; ++k) {
        unsigned long long best = 0;
#pragma unroll
        for (int i = 0; i < kTopkTile / 256; ++i) best = max(best, keys[i]);
        // Reduce within logical 32-lane warps first, then only eight warp
        // winners. Integer max is associative, so score/tie bits are exact.
        for (int offset = 16; offset; offset >>= 1)
            best = max(best, __shfl_down_sync(0xffffffffu, best, offset));
        if ((t & 31) == 0) winners[t / 32] = best;
        __syncthreads();
        if (t < 32) {
            best = t < 8 ? winners[t] : 0;
            for (int offset = 16; offset; offset >>= 1)
                best = max(best, __shfl_down_sync(0xffffffffu, best, offset));
            if (!t) winners[0] = best;
        }
        __syncthreads();
        best = winners[0];
        if (!t) {
            const int token = best ? vocab - 1 - (int) (unsigned) best : -1;
            ids[at + k] = token;
            unary[at + k] = token >= 0 ? logits[token] : -INFINITY;
        }
#pragma unroll
        for (int i = 0; i < kTopkTile / 256; ++i)
            if (keys[i] == best) keys[i] = 0;
        __syncthreads();
    }
}
__global__ void topk_parallel_kernel(const float* logits, const int32_t* candidates, int32_t* ids, float* unary,
                                    int vocab, int top_k, int tile_width, int tiles) {
    const int row = blockIdx.x, tile = blockIdx.y, t = threadIdx.x;
    logits += (size_t)row * vocab;
    if (candidates) candidates += (size_t) row * vocab;
    ids += ((size_t) row * tiles + tile) * top_k;
    unary += ((size_t) row * tiles + tile) * top_k;
    const int begin = tile * tile_width, end = begin + min(vocab - begin, tile_width);
    __shared__ float score[256];
    __shared__ int token[256];
    float upper = INFINITY;
    int upper_id = -1;
    for (int k = 0; k < top_k; ++k) {
        float best = -INFINITY;
        int id = -1;
        for (int64_t i = (int64_t) begin + t; i < end; i += 256) {
            const float v = logits[i];
            const int candidate = candidates ? candidates[i] : i;
            if (candidate < 0 || !isfinite(v) || v > upper || (v == upper && candidate <= upper_id)) continue;
            if (id < 0 || v > best || (v == best && candidate < id)) { best = v; id = candidate; }
        }
        score[t] = best; token[t] = id;
        __syncthreads();
        for (int offset = 128; offset; offset >>= 1) {
            if (t < offset) {
                const int other = token[t + offset];
                const float value = score[t + offset];
                if (other >= 0 && (token[t] < 0 || value > score[t] ||
                                  (value == score[t] && other < token[t]))) {
                    score[t] = value; token[t] = other;
                }
            }
            __syncthreads();
        }
        upper = score[0]; upper_id = token[0];
        if (!t) { ids[k] = upper_id; unary[k] = upper; }
        __syncthreads();
    }
}
__device__ float codebook_value(const float* p, size_t i) { return p[i]; }
__device__ float codebook_value(const uint16_t* p, size_t i) { return f32_from_bf16(p[i]); }

template<class T>
__global__ void select_kernel(const int32_t* ids, const float* unary, const float* hidden,
                              const T* prev, const T* next, const int32_t* anchor,
                              int32_t* out, float* confidence, int rows, int top_k, int rank) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    int predecessor = anchor[0];
    float scores[64];
    bool valid = predecessor >= 0;
    for (int row = 1; row < rows; ++row) {
        int best = -1;
        bool row_valid = valid;
        for (int k = 0; k < top_k; ++k) {
            const int token = ids[row * top_k + k];
            float score = 0;
            if (!valid || token < 0) { scores[k] = -INFINITY; continue; }
            for (int j = 0; j < rank; ++j)
                score += (codebook_value(prev, (size_t) predecessor * rank + j) * hidden[row * rank + j]) *
                         codebook_value(next, (size_t) token * rank + j);
            score += unary[row * top_k + k];
            scores[k] = score;
            if (!isfinite(score)) row_valid = false;
            if (isfinite(score) && (best < 0 || score > scores[best])) best = k;
        }
        valid = row_valid && best >= 0;
        if (!valid) { out[row - 1] = -1; if (confidence) confidence[row - 1] = 0; continue; }
        predecessor = ids[row * top_k + best];
        out[row - 1] = predecessor;
        if (confidence) {
            float sum = 0;
            for (int k = 0; k < top_k; ++k) sum += expf(scores[k] - scores[best]);
            confidence[row - 1] = 1.0f / sum;
        }
    }
}

// Candidates are independent within a row. Keep each candidate's rank sum in
// its original serial order, and let lane 0 retain the original tie handling
// and confidence sum. Only the work between candidates is parallelized.
__global__ void select_bf16_parallel(const int32_t* ids, const float* unary, const float* hidden,
                                    const uint16_t* prev, const uint16_t* next, const int32_t* anchor,
                                    int32_t* out, float* confidence, int rows, int top_k, int rank) {
    const int t = threadIdx.x;
    __shared__ float scores[64];
    __shared__ int predecessor, valid;
    if (!t) { predecessor = anchor[0]; valid = predecessor >= 0; }
    __syncthreads();
    for (int row = 1; row < rows; ++row) {
        if (t < top_k) {
            const int token = ids[row * top_k + t];
            float score = -INFINITY;
            if (valid && token >= 0) {
                score = 0;
                for (int j = 0; j < rank; ++j)
                    score += (codebook_value(prev, (size_t) predecessor * rank + j) * hidden[row * rank + j]) *
                             codebook_value(next, (size_t) token * rank + j);
                score += unary[row * top_k + t];
            }
            scores[t] = score;
        }
        __syncthreads();
        if (!t) {
            bool row_valid = valid;
            int best = -1;
            for (int k = 0; k < top_k; ++k) {
                if (!valid || ids[row * top_k + k] < 0) continue;
                const float score = scores[k];
                if (!isfinite(score)) row_valid = false;
                if (isfinite(score) && (best < 0 || score > scores[best])) best = k;
            }
            valid = row_valid && best >= 0;
            if (!valid) { out[row - 1] = -1; if (confidence) confidence[row - 1] = 0; }
            else {
                predecessor = ids[row * top_k + best];
                out[row - 1] = predecessor;
                if (confidence) {
                    float sum = 0;
                    for (int k = 0; k < top_k; ++k) sum += expf(scores[k] - scores[best]);
                    confidence[row - 1] = 1.0f / sum;
                }
            }
        }
        __syncthreads();
    }
}
// A row after the first can follow any candidate in the preceding row.
// Precompute that small lattice in parallel, retaining each rank sum's order.
__global__ void selector_lattice(const int32_t* ids,const float* unary,const float* hidden,
    const uint16_t* prev,const uint16_t* next,const int32_t* anchor,float* scores,int top_k,int rank) {
    const int row=blockIdx.x+1, predecessor_slot=blockIdx.y, candidate=threadIdx.x;
    if(candidate>=top_k || (row==1 && predecessor_slot!=0))return;
    const int predecessor=row==1?anchor[0]:ids[(row-1)*top_k+predecessor_slot];
    const int token=ids[row*top_k+candidate];
    float score=-INFINITY;
    if(predecessor>=0 && token>=0) {
        score=0;
        for(int j=0;j<rank;++j)
            score+=(codebook_value(prev,(size_t)predecessor*rank+j)*hidden[row*rank+j])*
                    codebook_value(next,(size_t)token*rank+j);
        score+=unary[row*top_k+candidate];
    }
    scores[((row-1)*top_k+predecessor_slot)*top_k+candidate]=score;
}
__global__ void selector_lattice_walk(const int32_t* ids,const float* scores,const int32_t* anchor,
    int32_t* out,float* confidence,int rows,int top_k) {
    if(threadIdx.x || blockIdx.x)return;
    bool valid=anchor[0]>=0;int predecessor_slot=0;
    for(int row=1;row<rows;++row) {
        const float* s=scores+((row-1)*top_k+predecessor_slot)*top_k;
        int best=-1;bool row_valid=valid;
        for(int k=0;k<top_k;++k) {
            if(!valid || ids[row*top_k+k]<0)continue;
            if(!isfinite(s[k]))row_valid=false;
            if(isfinite(s[k]) && (best<0 || s[k]>s[best]))best=k;
        }
        valid=row_valid && best>=0;
        if(!valid) {out[row-1]=-1;if(confidence)confidence[row-1]=0;}
        else {
            out[row-1]=ids[row*top_k+best];predecessor_slot=best;
            if(confidence) {
                float sum=0;for(int k=0;k<top_k;++k)sum+=expf(s[k]-s[best]);
                confidence[row-1]=1.f/sum;
            }
        }
    }
}
}  // namespace

void dflash2_convolve(const float* x, const float* dynamic, const float* base, float* out,
                     int rows, int hidden, int kernel, int group, int side, int block_rows, void* stream) {
    if (!x || !dynamic || !base || !out || rows <= 0 || (int64_t)rows * hidden > INT32_MAX - 255 || hidden <= 0 || kernel <= 0 || kernel > 8 || group <= 0 || hidden % group ||
        (int64_t)rows * (hidden / group) * 2 * kernel > INT32_MAX ||
        block_rows <= 0 || block_rows > 8 || rows % block_rows || side < 0 || side > 1 ||
        x == out || dynamic == out || base == out)
        throw std::invalid_argument("dflash2_convolve: invalid geometry or aliased output");
    conv_kernel<0><<<(rows * hidden + 255) / 256, 256, 0, (cudaStream_t) stream>>>(
        x, dynamic, base, out, rows, hidden, kernel, group, side, block_rows);
}
void dflash2_convolve_epilogue(const float* x, const float* dynamic, const float* base,
                               uint16_t* bf16, float* residual, int rows, int hidden,
                               int kernel, int group, int side, int block_rows, void* stream) {
    if (!x || !dynamic || !base || (bool(bf16) == bool(residual)) || rows <= 0 || hidden <= 0 ||
        (int64_t)rows * hidden > INT32_MAX - 255 || kernel <= 0 || kernel > 8 || group <= 0 || hidden % group ||
        (int64_t)rows * (hidden / group) * 2 * kernel > INT32_MAX || block_rows <= 0 || block_rows > 8 ||
        rows % block_rows || side < 0 || side > 1 || residual == x || residual == dynamic || residual == base ||
        (const void*)bf16 == x || (const void*)bf16 == dynamic || (const void*)bf16 == base)
        throw std::invalid_argument("dflash2_convolve_epilogue: invalid geometry or alias");
    if (bf16) conv_kernel<1><<<(rows * hidden + 255) / 256,256,0,(cudaStream_t)stream>>>(
        x,dynamic,base,nullptr,rows,hidden,kernel,group,side,block_rows,bf16);
    else conv_kernel<2><<<(rows * hidden + 255) / 256,256,0,(cudaStream_t)stream>>>(
        x,dynamic,base,residual,rows,hidden,kernel,group,side,block_rows);
}
std::size_t dflash2_topk_scratch_bytes(int rows, int vocab, int top_k) {
    if (rows <= 0 || vocab <= 0 || (int64_t) rows * vocab > INT32_MAX ||
        top_k < 1 || top_k > 64 || top_k > vocab)
        throw std::invalid_argument("dflash2_topk: invalid workspace geometry");
    return (std::size_t) rows * ((vocab + kTopkTile - 1LL) / kTopkTile) * top_k *
           (sizeof(int32_t) + sizeof(float));
}
void dflash2_topk(const float* logits, int32_t* ids, float* unary, int rows, int vocab, int top_k, void* stream,
                  void* scratch, std::size_t scratch_bytes) {
    if (!logits || !ids || !unary || rows <= 0 || (int64_t)rows * vocab > INT32_MAX || top_k < 1 || top_k > 64 || top_k > vocab)
        throw std::invalid_argument("dflash2_topk: invalid geometry");
    if (scratch && scratch_bytes < dflash2_topk_scratch_bytes(rows, vocab, top_k))
        throw std::invalid_argument("dflash2_topk: insufficient workspace");
    static const bool serial = [] {
        const char* value = std::getenv("STRATA_DFLASH2_SERIAL_TOPK");
        return value && value[0] == '1';
    }();
    static const bool tiled = [] {
        const char* value = std::getenv("STRATA_DFLASH2_TILED_TOPK");
        return !value || value[0] != '0';
    }();
    if (serial) topk_kernel<<<rows, 1, 0, (cudaStream_t) stream>>>(logits, ids, unary, vocab, top_k);
    else if (tiled && scratch && vocab > kTopkTile && rows <= 65535 &&
             (vocab + kTopkTile - 1LL) / kTopkTile <= 65535) {
        const int tiles = (vocab + kTopkTile - 1LL) / kTopkTile;
        int32_t* local_ids = (int32_t*) scratch;
        float* local_scores = (float*) (local_ids + (size_t) rows * tiles * top_k);
        static const bool register_topk = [] {
            const char* value = std::getenv("STRATA_DFLASH2_REGISTER_TOPK");
            return value && value[0] == '1';
        }();
        if (register_topk)
            topk_register_kernel<<<dim3(rows, tiles), 256, 0, (cudaStream_t) stream>>>(
                logits, local_ids, local_scores, vocab, top_k, tiles);
        else
            topk_parallel_kernel<<<dim3(rows, tiles), 256, 0, (cudaStream_t) stream>>>(
                logits, nullptr, local_ids, local_scores, vocab, top_k, kTopkTile, tiles);
        topk_parallel_kernel<<<rows, 256, 0, (cudaStream_t) stream>>>(
            local_scores, local_ids, ids, unary, tiles * top_k, top_k, tiles * top_k, 1);
    } else topk_parallel_kernel<<<rows, 256, 0, (cudaStream_t) stream>>>(
        logits, nullptr, ids, unary, vocab, top_k, vocab, 1);
}
void dflash2_select(const int32_t* ids, const float* unary, const float* hidden,
                    const float* prev, const float* next, const int32_t* anchor,
                    int32_t* out, float* confidence, int rows, int top_k, int rank, void* stream) {
    if (!ids || !unary || !hidden || !prev || !next || !anchor || !out ||
        rows < 2 || rows > 8 || top_k < 1 || top_k > 64 || rank < 1 || rank > 1024)
        throw std::invalid_argument("dflash2_select: invalid geometry");
    select_kernel<<<1, 1, 0, (cudaStream_t) stream>>>(
        ids, unary, hidden, prev, next, anchor, out, confidence, rows, top_k, rank);
}
void dflash2_select_bf16(const int32_t* ids, const float* unary, const float* hidden,
                         const uint16_t* prev, const uint16_t* next, const int32_t* anchor,
                         int32_t* out, float* confidence, int rows, int top_k, int rank, void* stream, bool reference, void* scratch, std::size_t scratch_bytes) {
    if (!ids || !unary || !hidden || !prev || !next || !anchor || !out ||
        rows < 2 || rows > 8 || top_k < 1 || top_k > 64 || rank < 1 || rank > 1024)
        throw std::invalid_argument("dflash2_select_bf16: invalid geometry");
    static const bool serial = [] {
        const char* value = std::getenv("STRATA_DFLASH2_SERIAL_SELECT");
        return value && value[0] == '1';
    }();
    static const bool lattice = [] {
        const char* value=std::getenv("STRATA_DFLASH2_LATTICE_SELECTOR");
        return value && value[0]=='1';
    }();
    const size_t needed=(size_t)(rows-1)*top_k*top_k*sizeof(float);
    if(lattice && !serial && !reference && scratch && scratch_bytes>=needed) {
        selector_lattice<<<dim3(rows-1,top_k),64,0,(cudaStream_t)stream>>>(
            ids,unary,hidden,prev,next,anchor,(float*)scratch,top_k,rank);
        selector_lattice_walk<<<1,1,0,(cudaStream_t)stream>>>(
            ids,(float*)scratch,anchor,out,confidence,rows,top_k);
        return;
    }
    if (serial || reference) select_kernel<<<1, 1, 0, (cudaStream_t)stream>>>(
        ids,unary,hidden,prev,next,anchor,out,confidence,rows,top_k,rank);
    else select_bf16_parallel<<<1, 64, 0, (cudaStream_t)stream>>>(
        ids,unary,hidden,prev,next,anchor,out,confidence,rows,top_k,rank);
}
namespace {
__global__ void hc_mean_kernel(const float* residual, float* out, uint16_t* out_bf16,
                               int rows, int hidden, int hc) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= rows * hidden) return;
    const int row = i / hidden, c = i % hidden;
    float sum = 0;
    for (int h = 0; h < hc; ++h) sum += residual[(row * hc + h) * hidden + c];
    sum /= hc;
    if (out) out[i] = sum;
    if (out_bf16) out_bf16[i] = bf16_from_f32(sum);
}
template<bool Canonical, bool Paired>
__global__ void attention128_kernel(const float* q, const uint16_t* k, const uint16_t* v,
                                    const int32_t* table, const int32_t* steps, float* scratch,
                                    int rows, int heads, int kv_heads, int page_size, int chunks, int window) {
    const int ch = blockIdx.x, head = blockIdx.y, row = blockIdx.z, t = threadIdx.x;
    const int lane = t & 31, warp = t / 32;
    const int end = steps[row * 4 + 1], position = end - rows + row;
    const int first = window ? max(0, end - rows - window + 1) : 0;
    const int begin = first + ch * 64;
    __shared__ float scores[64];
    __shared__ float probabilities[64];
    __shared__ size_t poolrows[64];
    const int kv_head = Canonical ? head >> 2 : head / (heads / kv_heads);
    for (int c = warp; c < 64; c += (Paired ? 2 : 4)) {
        const int cell = begin + c;
        float dot = 0;
        const bool visible = cell < end && (!window || abs(cell - position) < window);
        size_t poolrow = 0;
        if (visible) {
            const int page = Canonical ? cell >> 6 : cell / page_size;
            const int offset = Canonical ? cell & 63 : cell % page_size;
            poolrow = ((size_t)table[page] * (Canonical ? 8 : kv_heads) + kv_head) *
                      (Canonical ? 64 : page_size) + offset;
            for (int d = lane; d < 128; d += 32)
                dot += q[(row * heads + head) * 128 + d] * __half2float(__ushort_as_half(k[poolrow * 128 + d]));
            for (int offset = 16; offset; offset >>= 1) dot += __shfl_xor_sync(0xffffffffu, dot, offset);
        }
        if (!lane) {
            scores[c] = visible ? dot * 0.08838834764831845f : -INFINITY;
            poolrows[c] = poolrow;
        }
    }
    __syncthreads();
    float m = -INFINITY;
    for (int c = 0; c < 64; ++c) m = fmaxf(m, scores[c]);
    // Value components share their softmax weights and the address staged
    // by QK. Compute expf once per cell, keeping the original ordered sums
    // and masked-cell skip.
    if (t < 64 && scores[t] != -INFINITY) {
        probabilities[t] = expf(scores[t] - m);
    }
    __syncthreads();
    float sum = 0, acc = 0, acc1 = 0;
    for (int c = 0; c < 64; ++c) {
        if (scores[c] == -INFINITY) continue;
        const float p = probabilities[c];
        const size_t poolrow = poolrows[c];
        sum += p;
        if constexpr (Paired) {
            const float2 value = __half22float2(reinterpret_cast<const __half2*>(v + poolrow * 128)[t]);
            acc += p * value.x;
            acc1 += p * value.y;
        } else {
            acc += p * __half2float(__ushort_as_half(v[poolrow * 128 + t]));
        }
    }
    float* dst = scratch + ((row * heads + head) * chunks + ch) * 130;
    if constexpr (Paired) { dst[2 * t] = acc; dst[2 * t + 1] = acc1; }
    else dst[t] = acc;
    if (!t) { dst[128] = m; dst[129] = sum; }
}
__global__ void attention128_merge(const float* scratch, float* out, int chunks) {
    const int h = blockIdx.x, d = threadIdx.x;
    const float* src = scratch + h * chunks * 130;
    float m = -INFINITY;
    for (int c = 0; c < chunks; ++c) m = fmaxf(m, src[c * 130 + 128]);
    float sum = 0, acc = 0;
    for (int c = 0; c < chunks; ++c) {
        const float l = src[c * 130 + 129];
        if (!l) continue;
        const float scale = expf(src[c * 130 + 128] - m);
        sum += scale * l; acc += scale * src[c * 130 + d];
    }
    out[h * 128 + d] = sum ? acc / sum : 0;
}
}
void dflash2_hc_mean(const float* residual, float* out, uint16_t* out_bf16,
                     int rows, int hidden, int hc, void* stream) {
    if (!residual || (!out && !out_bf16) || rows < 1 || hidden < 1 || hc < 1 ||
        (int64_t)rows * hidden > INT32_MAX - 255 || (int64_t)rows * hidden * hc > INT32_MAX)
        throw std::invalid_argument("dflash2_hc_mean: invalid geometry or missing buffers");
    hc_mean_kernel<<<(rows * hidden + 255) / 256, 256, 0, (cudaStream_t)stream>>>(residual, out, out_bf16, rows, hidden, hc);
}
void dflash2_attention(const float* q, const uint16_t* k, const uint16_t* v, const int32_t* table,
                       const int32_t* steps, float* out, float* scratch, int rows, int heads,
                       int kv_heads, int page_size, int active_cells, int window, void* stream) {
    if (!q || !k || !v || !table || !steps || !out || !scratch || rows < 1 || rows > 8 ||
        heads < 1 || heads > 65535 || kv_heads < 1 || heads % kv_heads || page_size < 1 || active_cells < 1 || active_cells > INT32_MAX - 63 || window < 0 ||
        (int64_t)rows * heads * ((active_cells + 63LL) / 64) * 130 > INT32_MAX)
        throw std::invalid_argument("dflash2_attention: invalid geometry or missing buffers");
    const int chunks = (active_cells + 63) / 64;
    // The official geometry uses power-of-two pages and four Q heads per KV
    // head. Its addresses need shifts, not runtime integer divisions. Other
    // geometries keep the general indexing path.
    if (heads == 32 && kv_heads == 8 && page_size == 64) {
        // Two adjacent values share their denominator and use one aligned
        // half2 load. Keep the original ordered sum for each component. Small
        // windows retain four warps; long windows benefit from smaller CTAs.
        if (active_cells >= 512 && reinterpret_cast<std::uintptr_t>(v) % alignof(__half2) == 0)
            attention128_kernel<true, true><<<dim3(chunks, heads, rows), 64, 0, (cudaStream_t)stream>>>(q,k,v,table,steps,scratch,rows,heads,kv_heads,page_size,chunks,window);
        else
            attention128_kernel<true, false><<<dim3(chunks, heads, rows), 128, 0, (cudaStream_t)stream>>>(q,k,v,table,steps,scratch,rows,heads,kv_heads,page_size,chunks,window);
    } else
        attention128_kernel<false, false><<<dim3(chunks, heads, rows), 128, 0, (cudaStream_t)stream>>>(q,k,v,table,steps,scratch,rows,heads,kv_heads,page_size,chunks,window);
    attention128_merge<<<rows * heads,128,0,(cudaStream_t)stream>>>(scratch,out,chunks);
}

}  // namespace strata::kernels
