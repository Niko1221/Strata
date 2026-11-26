// What an engine says when its binary holds no GPU code for the card it was started on (#6759).
//
// Every .cu of an engine is compiled for the same CMAKE_CUDA_ARCHITECTURES, so asking CUDA for the attributes of one probe kernel
// (cudaFuncGetAttributes) answers for the whole binary: it fails with "no kernel image is available for execution on the device"
// when there is no SASS for the card and no PTX old enough to JIT. Without this check such a binary loads its weights, every kernel
// launch then fails without a message, the logits stay 0 and the answer is token id 0 forever (a CUDA 13 package on a V100).
#pragma once

#include <string>

namespace strata::core {

// `engine` = "strata-inkling" / "strata-hy4", `card` = the device name, `cuda_error` = cudaGetErrorString of the probe.
inline std::string no_gpu_code_message(const std::string& engine, const std::string& card, int cc_major, int cc_minor,
                                       const std::string& cuda_error) {
    const int cc = cc_major * 10 + cc_minor;
    std::string m = "this " + engine + " has no GPU code for " + card + " (compute capability " + std::to_string(cc_major) + "." +
                    std::to_string(cc_minor) + "): " + cuda_error + ". ";
    if (cc < 75) {
        m += "The CUDA 13 engine packages hold sm_75 and newer; a Pascal or Volta card needs the CUDA 12 engine package "
             "(setup: --cuda 12), or a build with CUDA 12.x and -DSTRATA_EXPERIMENTAL_SM60=ON -DCMAKE_CUDA_ARCHITECTURES=" +
             std::to_string(cc);
    } else {
        m += "Use the engine package made for this card (strata update; setup picks it), or rebuild with -DCMAKE_CUDA_ARCHITECTURES=" +
             std::to_string(cc);
    }
    return m;
}

}  // namespace strata::core
