#pragma once

#include <cuda_runtime.h>

namespace strata::kernels {

// Record both edges into the same capture. The origin stream must rejoin this
// branch before using its output, reusing its inputs, or ending capture.
inline cudaError_t begin_captured_branch(cudaStream_t origin, cudaStream_t branch,
                                         cudaEvent_t ready) {
    const cudaError_t recorded = cudaEventRecord(ready, origin);
    return recorded == cudaSuccess ? cudaStreamWaitEvent(branch, ready, 0) : recorded;
}

inline cudaError_t join_captured_branch(cudaStream_t origin, cudaStream_t branch,
                                        cudaEvent_t done) {
    const cudaError_t recorded = cudaEventRecord(done, branch);
    return recorded == cudaSuccess ? cudaStreamWaitEvent(origin, done, 0) : recorded;
}

}  // namespace strata::kernels
