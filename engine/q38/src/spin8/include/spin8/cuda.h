#pragma once
#include "spin8/spin8.h"
#include <cuda_runtime_api.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif
/* All tensor pointers and error_flag are device pointers on the stream's
 * device. Matrices and weights are host arrays, copied into launch parameters;
 * no retained host ownership, allocation, global state, synchronization, or
 * default-stream use. Strides are in elements. Full blocks of eight are
 * transformed; tails are identity. Transform calls reset error_flag on the
 * supplied stream. Attention resets it only if reset_error=1; use 0 to
 * accumulate multiple heads after caller memset. After stream completion, flag
 * bit 1 means nonfinite input/result, bit 2 means an all-masked attention row.
 * Outputs must be discarded on any nonzero flag. CUDA runtime failures are
 * returned immediately where detectable; asynchronous errors require stream
 * synchronization. Tensor allocation sizes and stream/device ownership remain
 * the caller's contract.
 */
enum {
  SPIN8_CUDA_NONFINITE = 1,
  SPIN8_CUDA_ALL_MASKED = 2,
  SPIN8_CUDA_INVALID_COUNT = 4
};
typedef enum {
  SPIN8_CUDA_ATTENTION_DIRECT = 0,
  SPIN8_CUDA_ATTENTION_THREE_VIEW = 1,
  SPIN8_CUDA_ATTENTION_HADAMARD = 2
} spin8_cuda_attention_mode;
cudaError_t spin8_cuda_transform_f32(const float *src, size_t src_stride,
                                     float *dst, size_t dst_stride, size_t rows,
                                     size_t cols, const double matrix[64],
                                     int inverse, int *error_flag,
                                     cudaStream_t stream);
cudaError_t spin8_cuda_hadamard_f32(const float *src, size_t src_stride,
                                    float *dst, size_t dst_stride, size_t rows,
                                    size_t cols, uint8_t sign_mask, int inverse,
                                    int *error_flag, cudaStream_t stream);
/* Outputs must be disjoint; an output may alias src only at exactly the same
 * address and row stride. All source values for a block are read before writes.
 */
cudaError_t spin8_cuda_transform3_f32(const float *src, size_t src_stride,
                                      float *dst0, float *dst1, float *dst2,
                                      size_t dst_stride, size_t rows,
                                      size_t cols, const double matrices[192],
                                      int *error_flag, cudaStream_t stream);
/* Undo each view with its transpose, then sum in the original vector space.
 * weights=null selects 1/3; otherwise finite nonnegative sum=1 (+/-1e-12).
 * Destination and all inputs must be disjoint. */
cudaError_t spin8_cuda_reduce3_f32(const float *src0, const float *src1,
                                   const float *src2, size_t src_stride,
                                   float *dst, size_t dst_stride, size_t rows,
                                   size_t cols, const double matrices[192],
                                   const double weights[3], int *error_flag,
                                   cudaStream_t stream);
/* Single-head/GQA building block: q[nq,q_stride], k/v[nk,kv_stride],
 * out[nq,out_stride], all with head_dim active columns; K and V share dtype.
 * mask is contiguous [nq,nk], 0=excluded, any other byte=allowed, null=all.
 * nk is key capacity. Optional nk_device points to an int active key count in
 * [1,nk], consumed on GPU for graph replay; invalid count sets bit 4 and zeroes
 * output. Unused capacity is ignored. Mask/scores row stride stays nk capacity.
 * scores is caller-owned contiguous float[nq,nk] scratch, overwritten with the
 * one common softmax distribution. All-masked rows write zero and set bit 2.
 * mode 0=ordinary dot, 1=weighted three transformed-view dots BEFORE one
 * softmax, 2=signed normalized Hadamard HD on each complete eight-block.
 * mode 1 matrices are finite orthogonal row-major [3,8,8]; modes 0/2 ignore
 * matrices and weights. F32 and F16 KV are supported, other dtypes rejected.
 * All tensors/scratch/error_flag must be non-overlapping; q/k/v may overlap.
 */
cudaError_t spin8_cuda_attention_f32(
    const float *q, size_t q_stride, const void *k, const void *v,
    size_t kv_stride, spin8_dtype kv_dtype, float *out, size_t out_stride,
    size_t nq, size_t nk, size_t head_dim, const uint8_t *mask,
    const double matrices[192], const double weights[3], int mode,
    uint8_t sign_mask, float *scores, const int *nk_device, int *error_flag,
    int reset_error, cudaStream_t stream);
#ifdef __cplusplus
}
#endif
