#pragma once

#include <string_view>

namespace strata::kernels::detail {

// The multi read's variants, the same numbers `fused_gr.cu` latches per card.
constexpr int kGrVariantPlain = 1, kGrVariantSplit = 2, kGrVariantStaged = 3, kGrVariantRegisterHalf = 7;

// What `STRATA_HC_SPLIT` asks for.  Unset names the staged/upstream path: it never asks for the P4 candidate.
constexpr int fused_gr_env_variant(std::string_view value) noexcept {
    if (value == "0") return kGrVariantPlain;
    if (value == "1") return kGrVariantSplit;
    if (value == "7") return kGrVariantRegisterHalf;
    return kGrVariantStaged;
}

// The P4 self-test runs only for an explicit `STRATA_HC_SPLIT=7` on gfx1200/gfx1201.
constexpr bool fused_gr_p4_selftest_requested(int want, bool gfx12) noexcept {
    return gfx12 && want == kGrVariantRegisterHalf;
}

// The variant to latch once the self-test ran.  P4 is selected only when the request was explicit and its parity
// gate passed; every other request keeps the staged/split/plain fallback, and unset never leaves staged.
constexpr int fused_gr_select(int want, bool split_ok, bool staged_ok, bool p4_gate_ok) noexcept {
    if (want == kGrVariantSplit && split_ok) return kGrVariantSplit;
    if (want == kGrVariantRegisterHalf && p4_gate_ok) return kGrVariantRegisterHalf;
    if ((want == kGrVariantStaged || want == kGrVariantRegisterHalf) && staged_ok) return kGrVariantStaged;
    if (split_ok) return kGrVariantSplit;
    return kGrVariantPlain;
}

// P4 may only be promoted when the architecture and both sides of its staged-parity oracle pass.
constexpr bool fused_gr_p4_parity_gate(bool gfx12, bool staged_ok, bool p4_ok) noexcept {
    return gfx12 && staged_ok && p4_ok;
}

// Emit the q32 parity diagnostic only when the complete self-test succeeded and this path was checked.
constexpr bool fused_gr_p4_q32_parity_marker_ok(bool selftest_ok, bool qfuse_hc, bool staged_ok, bool p4_ok) noexcept {
    return selftest_ok && qfuse_hc && staged_ok && p4_ok;
}

// Which backend's runtime returned the code.  The two headers share the numbers of the faults they both define,
// and each has codes the other does not: NVIDIA's `driver_types.h` has the device-fault and containment codes
// (226, 714-718, 721, 810, 911) that HIP's `hip_runtime_api.h` never returns - HIP uses 911 for
// `hipErrorInvalidChannelDescriptor`, a plain API error - and HIP has the HSA runtime codes 1052/1053.  A code
// is therefore classified per backend, never by its number alone.
enum class GrBackend : unsigned char { Cuda, Hip };

// Runtime launch/sync error codes.  `fused_gr.cu` static_asserts every one against its own header's name in the
// backend it compiles, so a header that renames or renumbers one breaks the build instead of the classification.
// `kGrErr*` are the numbers both headers give the same meaning; `kGrCudaErr*` / `kGrHipErr*` exist in one header.
constexpr int kGrErrSuccess = 0;
constexpr int kGrErrInvalidValue = 1;
constexpr int kGrErrMemoryAllocation = 2;
constexpr int kGrErrInvalidDevicePointer = 17;
constexpr int kGrErrInvalidMemcpyDirection = 21;
constexpr int kGrErrInvalidConfiguration = 9;
constexpr int kGrErrInvalidDeviceFunction = 98;
constexpr int kGrErrNoDevice = 100;
constexpr int kGrErrInvalidDevice = 101;
constexpr int kGrErrInvalidKernelImage = 200;      // HIP names it hipErrorInvalidImage
constexpr int kGrErrNoKernelImageForDevice = 209;  // HIP names it hipErrorNoBinaryForGpu
constexpr int kGrErrECCNotCorrectable = 214;       // CUDA cudaErrorECCUncorrectable / HIP hipErrorECCNotCorrectable, both fatal
constexpr int kGrErrUnsupportedLimit = 215;
constexpr int kGrErrInvalidResourceHandle = 400;
constexpr int kGrErrNotReady = 600;

constexpr int kGrErrIllegalAddress = 700;
constexpr int kGrErrLaunchOutOfResources = 701;
constexpr int kGrErrLaunchTimeout = 702;           // HIP names it hipErrorLaunchTimeOut
constexpr int kGrCudaErrLaunchIncompatibleTexturing = 703;
constexpr int kGrErrContextIsDestroyed = 709;
constexpr int kGrErrAssert = 710;
constexpr int kGrErrLaunchFailure = 719;
constexpr int kGrErrCooperativeLaunchTooLarge = 720;
constexpr int kGrErrNotSupported = 801;
constexpr int kGrErrStreamCaptureUnsupported = 900;
constexpr int kGrErrUnknown = 999;
constexpr int kGrCudaErrContained = 226;
constexpr int kGrCudaErrHardwareStackError = 714;
constexpr int kGrCudaErrIllegalInstruction = 715;
constexpr int kGrCudaErrMisalignedAddress = 716;
constexpr int kGrCudaErrInvalidAddressSpace = 717;
constexpr int kGrCudaErrInvalidPc = 718;
constexpr int kGrCudaErrTensorMemoryLeak = 721;
constexpr int kGrCudaErrMpsClientTerminated = 810;
constexpr int kGrCudaErrExternalDevice = 911;      // HIP's 911 is hipErrorInvalidChannelDescriptor
constexpr int kGrCudaErrInvalidClusterSize = 912;  // HIP's 912 is hipErrorInvalidTexture
constexpr int kGrHipErrRuntimeMemory = 1052;       // the HSA runtime's own memory call failed
constexpr int kGrHipErrRuntimeOther = 1053;        // the HSA runtime's own other call failed

// Return true when it is unsafe to issue more GPU work after a runtime error. The runtime headers identify the
// context-fatal device faults below; uncorrectable ECC is also fatal because the device reported corrupted
// execution state. Only explicitly enumerated, documented API/launch-refusal statuses are recoverable. Unknown or
// unclassified values fail closed: the fallback self-test must not assume the context is usable without evidence.
// Codes are checked against the selected backend's runtime header in fused_gr.cu; CUDA's 911 is fatal, while HIP's
// 911/912 name ordinary invalid-channel/texture errors and are explicitly recoverable for HIP.
constexpr bool fused_gr_context_fatal(int err, GrBackend backend) noexcept {
    if (backend == GrBackend::Cuda) {
        switch (err) {
            case kGrErrECCNotCorrectable:
            case kGrErrIllegalAddress:
            case kGrErrLaunchTimeout:
            case kGrErrContextIsDestroyed:
            case kGrErrAssert:
            case kGrErrLaunchFailure:
            case kGrCudaErrContained:
            case kGrCudaErrHardwareStackError:
            case kGrCudaErrIllegalInstruction:
            case kGrCudaErrMisalignedAddress:
            case kGrCudaErrInvalidAddressSpace:
            case kGrCudaErrInvalidPc:
            case kGrCudaErrTensorMemoryLeak:
            case kGrCudaErrMpsClientTerminated:
            case kGrCudaErrExternalDevice:
                return true;
            case kGrErrSuccess:
            case kGrErrInvalidValue:
            case kGrErrMemoryAllocation:
            case kGrErrInvalidDevicePointer:
            case kGrErrInvalidMemcpyDirection:
            case kGrErrInvalidConfiguration:
            case kGrErrInvalidDeviceFunction:
            case kGrErrNoDevice:
            case kGrErrInvalidDevice:
            case kGrErrUnsupportedLimit:
            case kGrErrNotSupported:
            case kGrErrInvalidKernelImage:
            case kGrErrNoKernelImageForDevice:
            case kGrErrInvalidResourceHandle:
            case kGrErrNotReady:
            case kGrErrLaunchOutOfResources:
            case kGrCudaErrLaunchIncompatibleTexturing:
            case kGrErrCooperativeLaunchTooLarge:
            case kGrErrStreamCaptureUnsupported:
            case kGrCudaErrInvalidClusterSize:
                return false;
            default:
                return true;
        }
    }

    switch (err) {
        case kGrErrECCNotCorrectable:
        case kGrErrIllegalAddress:
        case kGrErrLaunchTimeout:
        case kGrErrContextIsDestroyed:
        case kGrErrAssert:
        case kGrErrLaunchFailure:
            return true;
        case kGrErrSuccess:
        case kGrErrInvalidValue:
        case kGrErrMemoryAllocation:
        case kGrErrInvalidDevicePointer:
        case kGrErrInvalidMemcpyDirection:
        case kGrErrInvalidConfiguration:
        case kGrErrInvalidDeviceFunction:
        case kGrErrNoDevice:
        case kGrErrInvalidDevice:
        case kGrErrUnsupportedLimit:
        case kGrErrNotSupported:
        case kGrErrInvalidKernelImage:
        case kGrErrNoKernelImageForDevice:
        case kGrErrInvalidResourceHandle:
        case kGrErrNotReady:
        case kGrErrLaunchOutOfResources:
        case kGrErrCooperativeLaunchTooLarge:
        case kGrErrStreamCaptureUnsupported:
        case kGrCudaErrExternalDevice:     // HIP's 911 is hipErrorInvalidChannelDescriptor
        case kGrCudaErrInvalidClusterSize: // HIP's 912 is hipErrorInvalidTexture
        case kGrHipErrRuntimeMemory:
        case kGrHipErrRuntimeOther:
            return false;
        default:
            return true;
    }
}

struct GrCleanupPolicy {
    bool selftest_ok;
    bool p4_ok;
};

// Candidate-stream cleanup failures disable P4 only; shared-stream/allocation cleanup failures invalidate the test.
constexpr GrCleanupPolicy fused_gr_cleanup_policy(bool shared_cleanup_ok, bool p4_check_ok,
                                                  bool p4_cleanup_ok) noexcept {
    return {shared_cleanup_ok, shared_cleanup_ok && p4_check_ok && p4_cleanup_ok};
}

}  // namespace strata::kernels::detail
