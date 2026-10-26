// DFlash2 equations, independently expressed as direct CUDA/HIP kernels.
// Reference: llama.cpp src/models/dflash.cpp (a9d27ac), z-lab/dflash (MIT, 07ebd93).
#include "strata/kernels/dflash2.hpp"
#include "strata/kernels/bf16_bits.hpp"
#include "strata/kernels/bf16_gemv.hpp"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <stdexcept>

namespace strata::kernels {
namespace {
template<int Tile, int Warps>
__global__ void bf16_project_tiles(const uint16_t* __restrict__ x, const uint16_t* __restrict__ weights,
                                   float* __restrict__ out, int width, int outputs, int rows) {
    const int output = blockIdx.x * Warps + threadIdx.x / 32;
    if (output >= outputs) return;
    const int lane = threadIdx.x & 31, row0 = blockIdx.y * Tile;
    const uint16_t* w = weights + (size_t)output * width;
    float acc[Tile] = {};
    for (int i = lane; i < width; i += 32) {
        const float value = f32_from_bf16(w[i]);
#pragma unroll
        for (int b = 0; b < Tile; ++b)
            if (row0 + b < rows) acc[b] += f32_from_bf16(x[(size_t)(row0 + b) * width + i]) * value;
    }
#pragma unroll
    for (int b = 0; b < Tile; ++b) {
        float value = acc[b];
        for (int offset = 16; offset; offset >>= 1)
            value += __shfl_down_sync(0xffffffffu, value, offset);
        if (lane == 0 && row0 + b < rows) out[(size_t)(row0 + b) * outputs + output] = value;
    }
}
template<int Tile>
void launch_bf16_tiles(const uint16_t* x, const uint16_t* weights, float* out,
                        int width, int outputs, int rows, int warps, void* stream) {
    const dim3 grid((outputs + warps - 1) / warps, (rows + Tile - 1) / Tile);
    switch (warps) {
    case 1: bf16_project_tiles<Tile,1><<<grid,32,0,(cudaStream_t)stream>>>(x,weights,out,width,outputs,rows); break;
    case 2: bf16_project_tiles<Tile,2><<<grid,64,0,(cudaStream_t)stream>>>(x,weights,out,width,outputs,rows); break;
    case 4: bf16_project_tiles<Tile,4><<<grid,128,0,(cudaStream_t)stream>>>(x,weights,out,width,outputs,rows); break;
    case 8: bf16_project_tiles<Tile,8><<<grid,256,0,(cudaStream_t)stream>>>(x,weights,out,width,outputs,rows); break;
    }
}
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
template<bool WarpReduce = false>
__global__ void topk_parallel_kernel(const float* logits, const int32_t* candidates, int32_t* ids, float* unary,
                                    int vocab, int top_k, int tile_width, int tiles) {
    const int row = blockIdx.x, tile = blockIdx.y, t = threadIdx.x;
    logits += (size_t)row * vocab;
    if (candidates) candidates += (size_t) row * vocab;
    ids += ((size_t) row * tiles + tile) * top_k;
    unary += ((size_t) row * tiles + tile) * top_k;
    const int begin = tile * tile_width, end = begin + min(vocab - begin, tile_width);
    __shared__ float score[WarpReduce ? 8 : 256];
    __shared__ int token[WarpReduce ? 8 : 256];
    __shared__ float winner_score;
    __shared__ int winner_id;
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
        if constexpr (WarpReduce) {
            for (int offset = 16; offset; offset >>= 1) {
                const float value = __shfl_down_sync(0xffffffffu, best, offset);
                const int other = __shfl_down_sync(0xffffffffu, id, offset);
                if (other >= 0 && (id < 0 || value > best || (value == best && other < id))) {
                    best = value; id = other;
                }
            }
            if ((t & 31) == 0) { score[t / 32] = best; token[t / 32] = id; }
            __syncthreads();
            if (t < 32) {
                best = t < 8 ? score[t] : -INFINITY;
                id = t < 8 ? token[t] : -1;
                for (int offset = 16; offset; offset >>= 1) {
                    const float value = __shfl_down_sync(0xffffffffu, best, offset);
                    const int other = __shfl_down_sync(0xffffffffu, id, offset);
                    if (other >= 0 && (id < 0 || value > best || (value == best && other < id))) {
                        best = value; id = other;
                    }
                }
                if (!t) { winner_score = best; winner_id = id; }
            }
            __syncthreads();
            // Separate winner storage is not overwritten by the next warp
            // reduction. Its first barrier protects every reader of this winner.
            upper = winner_score; upper_id = winner_id;
            if (!t) { ids[k] = upper_id; unary[k] = upper; }
        } else {
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
}
// Each vocabulary tile supplies a sorted list. Only its head can be the next
// global winner, so keep two list heads per lane instead of rescanning all K.
__global__ void topk_merge_heads(const int32_t* candidates, const float* scores,
                                 int32_t* ids, float* unary, int tiles, int top_k) {
    const int row = blockIdx.x, lane = threadIdx.x;
    int cursor[2] = {}, head[2];
    float value[2];
#pragma unroll
    for (int part = 0; part < 2; ++part) {
        const int tile = lane + part * 32;
        const size_t at = ((size_t)row * tiles + tile) * top_k;
        head[part] = tile < tiles ? candidates[at] : -1;
        value[part] = tile < tiles ? scores[at] : -INFINITY;
    }
    for (int k = 0; k < top_k; ++k) {
        int id = -1;
        float best = -INFINITY;
#pragma unroll
        for (int part = 0; part < 2; ++part) {
            const int other = head[part]; const float score = value[part];
            if (other >= 0 && isfinite(score) && (id < 0 || score > best || (score == best && other < id))) {
                best = score; id = other;
            }
        }
        for (int offset = 16; offset; offset >>= 1) {
            const float score = __shfl_down_sync(0xffffffffu,best,offset);
            const int other = __shfl_down_sync(0xffffffffu,id,offset);
            if (other >= 0 && (id < 0 || score > best || (score == best && other < id))) {
                best = score; id = other;
            }
        }
        best = __shfl_sync(0xffffffffu,best,0);
        id = __shfl_sync(0xffffffffu,id,0);
        if (!lane) { ids[(size_t)row*top_k+k]=id; unary[(size_t)row*top_k+k]=best; }
#pragma unroll
        for (int part = 0; part < 2; ++part) {
            if (id >= 0 && head[part] == id) {
                ++cursor[part];
                const int tile = lane + part * 32;
                const size_t at = ((size_t)row * tiles + tile) * top_k + cursor[part];
                head[part] = cursor[part] < top_k ? candidates[at] : -1;
                value[part] = cursor[part] < top_k ? scores[at] : -INFINITY;
            }
        }
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

// Load four rank values at once, preserving the four serial FMA updates.
// Dispatch requires whole aligned vectors; other shapes retain scalar reads.
__device__ float selector_vector_sum(const uint16_t* p, const uint16_t* n,
                                     const float* h, int rank) {
    float score = 0;
    for (int j = 0; j < rank; j += 4) {
        const uint2 pv = *(const uint2*)(p + j), nv = *(const uint2*)(n + j);
        const float4 hv = *(const float4*)(h + j);
        score += (f32_from_bf16(uint16_t(pv.x)) * hv.x) * f32_from_bf16(uint16_t(nv.x));
        score += (f32_from_bf16(uint16_t(pv.x >> 16)) * hv.y) * f32_from_bf16(uint16_t(nv.x >> 16));
        score += (f32_from_bf16(uint16_t(pv.y)) * hv.z) * f32_from_bf16(uint16_t(nv.y));
        score += (f32_from_bf16(uint16_t(pv.y >> 16)) * hv.w) * f32_from_bf16(uint16_t(nv.y >> 16));
    }
    return score;
}
// Candidates are independent within a row. Keep each candidate's rank sum in
// its original serial order, and let lane 0 retain the original tie handling
// and confidence sum. Only the work between candidates is parallelized.
template<bool Vector = false>
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
                if constexpr (Vector) score = selector_vector_sum(prev + (size_t)predecessor * rank,
                    next + (size_t)token * rank, hidden + row * rank, rank);
                else {
                for (int j = 0; j < rank; ++j)
                    score += (codebook_value(prev, (size_t) predecessor * rank + j) * hidden[row * rank + j]) *
                             codebook_value(next, (size_t) token * rank + j);
                }
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
template<bool Vector = false>
__global__ void selector_lattice(const int32_t* ids,const float* unary,const float* hidden,
    const uint16_t* prev,const uint16_t* next,const int32_t* anchor,float* scores,int top_k,int rank) {
    const int row=blockIdx.x+1, predecessor_slot=blockIdx.y, candidate=threadIdx.x;
    if(candidate>=top_k || (row==1 && predecessor_slot!=0))return;
    const int predecessor=row==1?anchor[0]:ids[(row-1)*top_k+predecessor_slot];
    const int token=ids[row*top_k+candidate];
    float score=-INFINITY;
    if(predecessor>=0 && token>=0) {
        score=0;
        if constexpr (Vector) score = selector_vector_sum(prev + (size_t)predecessor * rank,
            next + (size_t)token * rank, hidden + row * rank, rank);
        else {
        for(int j=0;j<rank;++j)
            score+=(codebook_value(prev,(size_t)predecessor*rank+j)*hidden[row*rank+j])*
                    codebook_value(next,(size_t)token*rank+j);
        }
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

void dflash2_bf16_project(const uint16_t* x, const uint16_t* weights, float* out,
                          int width, int outputs, int rows, int tile, void* stream, int warps) {
    if (!x || !weights || !out || width < 1 || outputs < 1 || outputs > INT32_MAX - 7 ||
        rows < 1 || rows > 8 || (tile != 1 && tile != 2 && tile != 4) ||
        (warps != 1 && warps != 2 && warps != 4 && warps != 8))
        throw std::invalid_argument("dflash2_bf16_project: invalid buffers or geometry");
    if (rows <= tile) {
        bf16_gemv_batch(x, weights, out, width, outputs, rows, stream);
        return;
    }
    switch (tile) {
    case 1: launch_bf16_tiles<1>(x,weights,out,width,outputs,rows,warps,stream); break;
    case 2: launch_bf16_tiles<2>(x,weights,out,width,outputs,rows,warps,stream); break;
    case 4: launch_bf16_tiles<4>(x,weights,out,width,outputs,rows,warps,stream); break;
    }
}
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
    static const bool warp_reduce = [] {
        const char* value = std::getenv("STRATA_DFLASH2_WARP_TOPK");
        return value && value[0] == '1';
    }();
    auto parallel = [&](const float* source, const int32_t* candidates, int32_t* output_ids,
                        float* output_scores, int width, int tile_width, int tiles, dim3 grid) {
        if (warp_reduce)
            topk_parallel_kernel<true><<<grid,256,0,(cudaStream_t)stream>>>(
                source,candidates,output_ids,output_scores,width,top_k,tile_width,tiles);
        else
            topk_parallel_kernel<false><<<grid,256,0,(cudaStream_t)stream>>>(
                source,candidates,output_ids,output_scores,width,top_k,tile_width,tiles);
    };
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
            parallel(logits,nullptr,local_ids,local_scores,vocab,kTopkTile,tiles,dim3(rows,tiles));
        static const bool merge_heads = [] {
            const char* value = std::getenv("STRATA_DFLASH2_MERGE_TOPK");
            return value && value[0] == '1';
        }();
        if (merge_heads && tiles <= 64)
            topk_merge_heads<<<rows,32,0,(cudaStream_t)stream>>>(local_ids,local_scores,ids,unary,tiles,top_k);
        else parallel(local_scores,local_ids,ids,unary,tiles*top_k,tiles*top_k,1,dim3(rows));
    } else parallel(logits,nullptr,ids,unary,vocab,vocab,1,dim3(rows));
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
    static const bool vector_enabled = [] {
        const char* value = std::getenv("STRATA_DFLASH2_VECTOR_SELECTOR");
        return value && value[0] == '1';
    }();
    const bool vector = !reference && vector_enabled && rank % 4 == 0 &&
        ((uintptr_t(prev) | uintptr_t(next)) & 7) == 0 && (uintptr_t(hidden) & 15) == 0;
    const size_t needed=(size_t)(rows-1)*top_k*top_k*sizeof(float);
    if(lattice && !serial && !reference && scratch && scratch_bytes>=needed) {
        if (vector) selector_lattice<true><<<dim3(rows-1,top_k),64,0,(cudaStream_t)stream>>>(
            ids,unary,hidden,prev,next,anchor,(float*)scratch,top_k,rank);
        else selector_lattice<false><<<dim3(rows-1,top_k),64,0,(cudaStream_t)stream>>>(
            ids,unary,hidden,prev,next,anchor,(float*)scratch,top_k,rank);
        selector_lattice_walk<<<1,1,0,(cudaStream_t)stream>>>(
            ids,(float*)scratch,anchor,out,confidence,rows,top_k);
        return;
    }
    if (serial || reference) select_kernel<<<1, 1, 0, (cudaStream_t)stream>>>(
        ids,unary,hidden,prev,next,anchor,out,confidence,rows,top_k,rank);
    else if (vector) select_bf16_parallel<true><<<1, 64, 0, (cudaStream_t)stream>>>(
        ids,unary,hidden,prev,next,anchor,out,confidence,rows,top_k,rank);
    else select_bf16_parallel<false><<<1, 64, 0, (cudaStream_t)stream>>>(
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
template<bool Canonical, bool Paired, int PageShift = 6, bool CacheQuery = false>
__global__ void attention128_kernel(const float* q, const uint16_t* k, const uint16_t* v,
                                    const int32_t* table, const int32_t* steps, float* scratch,
                                    int rows, int heads, int kv_heads, int page_size, int chunks, int window) {
    const int ch = blockIdx.x, head = blockIdx.y, row = blockIdx.z, t = threadIdx.x;
    const int lane = t & 31, warp = t / 32;
    float cached_q[CacheQuery ? 4 : 1];
    if constexpr (CacheQuery) {
#pragma unroll
        for (int j=0;j<4;++j) cached_q[j]=q[(row*heads+head)*128+lane+32*j];
    }
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
            const int page = Canonical ? cell >> PageShift : cell / page_size;
            const int offset = Canonical ? cell & ((1 << PageShift) - 1) : cell % page_size;
            poolrow = ((size_t)table[page] * (Canonical ? 8 : kv_heads) + kv_head) *
                      (Canonical ? (1 << PageShift) : page_size) + offset;
            if constexpr (CacheQuery) {
                for (int d = lane; d < 128; d += 32)
                    dot += cached_q[d / 32] * __half2float(__ushort_as_half(k[poolrow * 128 + d]));
            } else {
                for (int d = lane; d < 128; d += 32)
                    dot += q[(row * heads + head) * 128 + d] * __half2float(__ushort_as_half(k[poolrow * 128 + d]));
            }
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
    static const bool small_page = [] {
        const char* value = std::getenv("STRATA_DFLASH2_SMALL_PAGE_ATTENTION");
        return value && value[0] == '1';
    }();
    static const bool cached_query = [] {
        const char* v=std::getenv("STRATA_DFLASH2_CACHE_QUERY"); return v && v[0]=='1';
    }();
    if (cached_query) {
        if (small_page && heads==32 && kv_heads==8 && page_size==4) {
            if (active_cells>=512 && reinterpret_cast<std::uintptr_t>(v)%alignof(__half2)==0)
                attention128_kernel<true,true,2,true><<<dim3(chunks,heads,rows),64,0,(cudaStream_t)stream>>>(q,k,v,table,steps,scratch,rows,heads,kv_heads,page_size,chunks,window);
            else attention128_kernel<true,false,2,true><<<dim3(chunks,heads,rows),128,0,(cudaStream_t)stream>>>(q,k,v,table,steps,scratch,rows,heads,kv_heads,page_size,chunks,window);
        } else if (heads==32 && kv_heads==8 && page_size==64) {
            if (active_cells>=512 && reinterpret_cast<std::uintptr_t>(v)%alignof(__half2)==0)
                attention128_kernel<true,true,6,true><<<dim3(chunks,heads,rows),64,0,(cudaStream_t)stream>>>(q,k,v,table,steps,scratch,rows,heads,kv_heads,page_size,chunks,window);
            else attention128_kernel<true,false,6,true><<<dim3(chunks,heads,rows),128,0,(cudaStream_t)stream>>>(q,k,v,table,steps,scratch,rows,heads,kv_heads,page_size,chunks,window);
        } else attention128_kernel<false,false,6,true><<<dim3(chunks,heads,rows),128,0,(cudaStream_t)stream>>>(q,k,v,table,steps,scratch,rows,heads,kv_heads,page_size,chunks,window);
        attention128_merge<<<rows*heads,128,0,(cudaStream_t)stream>>>(scratch,out,chunks);
        return;
    }
    // The official geometry uses power-of-two pages and four Q heads per KV
    // head. Its addresses need shifts, not runtime integer divisions. Other
    // geometries keep the general indexing path.
    if (small_page && heads == 32 && kv_heads == 8 && page_size == 4) {
        if (active_cells >= 512 && reinterpret_cast<std::uintptr_t>(v) % alignof(__half2) == 0)
            attention128_kernel<true, true, 2><<<dim3(chunks, heads, rows), 64, 0, (cudaStream_t)stream>>>(q,k,v,table,steps,scratch,rows,heads,kv_heads,page_size,chunks,window);
        else
            attention128_kernel<true, false, 2><<<dim3(chunks, heads, rows), 128, 0, (cudaStream_t)stream>>>(q,k,v,table,steps,scratch,rows,heads,kv_heads,page_size,chunks,window);
    } else if (heads == 32 && kv_heads == 8 && page_size == 64) {
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
