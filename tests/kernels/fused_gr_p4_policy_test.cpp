#include "strata/kernels/fused_gr_policy.hpp"

#include <cstdio>
#include <string_view>

namespace d = strata::kernels::detail;

namespace {

int failures = 0;

void expect(bool got, bool expected, const char* name) {
    if (got != expected) {
        ++failures;
        std::fprintf(stderr, "FAIL: %s expected=%d got=%d\n", name, expected, got);
    }
}

void expect_eq(int got, int expected, const char* name) {
    if (got != expected) {
        ++failures;
        std::fprintf(stderr, "FAIL: %s expected=%d got=%d\n", name, expected, got);
    }
}

}  // namespace

int main() {
    struct TestCase {
        bool gfx12;
        bool staged_ok;
        bool p4_ok;
        bool expected;
        const char* name;
    };
    const TestCase cases[] = {
        {true, true, true, true, "gfx12 with staged and P4 parity"},
        {true, false, true, false, "reject P4 when staged reference fails"},
        {true, true, false, false, "reject P4 when candidate parity fails"},
        {false, true, true, false, "reject P4 outside gfx12"},
        {false, false, false, false, "reject all failed gates"},
    };
    for (const auto& test : cases) {
        expect(d::fused_gr_p4_parity_gate(test.gfx12, test.staged_ok, test.p4_ok), test.expected, test.name);
    }

    struct MarkerCase {
        bool selftest_ok;
        bool qfuse_hc;
        bool staged_ok;
        bool p4_ok;
        bool expected;
        const char* name;
    };
    const MarkerCase marker_cases[] = {
        {true, true, true, true, true, "report q32 parity after full self-test"},
        {false, true, true, true, false, "suppress q32 marker after unrelated self-test failure"},
        {true, false, true, true, false, "suppress q32 marker when QFUSE_HC is off"},
        {true, true, false, true, false, "suppress q32 marker when staged parity fails"},
        {true, true, true, false, false, "suppress q32 marker when P4 parity fails"},
    };
    for (const auto& test : marker_cases) {
        expect(d::fused_gr_p4_q32_parity_marker_ok(test.selftest_ok, test.qfuse_hc, test.staged_ok, test.p4_ok),
               test.expected, test.name);
    }

    // STRATA_HC_SPLIT is opt-in and the selector must be the exact value, not merely start with 7.
    struct EnvCase {
        std::string_view value;
        int expected;
        const char* name;
    };
    const EnvCase env_cases[] = {
        {"", d::kGrVariantStaged, "unset/empty value requests staged"},
        {"2", d::kGrVariantStaged, "=2 requests staged"},
        {"1", d::kGrVariantSplit, "=1 requests split"},
        {"0", d::kGrVariantPlain, "=0 requests plain"},
        {"7", d::kGrVariantRegisterHalf, "exact =7 requests P4"},
        {"9", d::kGrVariantStaged, "an unlisted value requests staged"},
        {"70", d::kGrVariantStaged, "=70 does not request P4"},
        {"7suffix", d::kGrVariantStaged, "a 7-prefixed suffix does not request P4"},
        {"07", d::kGrVariantStaged, "a zero-prefixed 7 does not request P4"},
    };
    for (const auto& test : env_cases) {
        expect_eq(d::fused_gr_env_variant(test.value), test.expected, test.name);
    }

    // Only the exact explicit =7 value on gfx1200/gfx1201 runs the P4 self-test.
    struct OptInCase {
        std::string_view value;
        bool gfx12;
        bool expected;
        const char* name;
    };
    const OptInCase optin_cases[] = {
        {"", true, false, "unset never self-tests P4 on gfx12"},
        {"", false, false, "unset never self-tests P4 off gfx12"},
        {"7", true, true, "=7 self-tests P4 on gfx12"},
        {"7", false, false, "=7 self-tests nothing off gfx12"},
        {"2", true, false, "=2 never self-tests P4 on gfx12"},
        {"1", true, false, "=1 never self-tests P4"},
        {"0", true, false, "=0 never self-tests P4"},
        {"70", true, false, "=70 never self-tests P4"},
        {"7suffix", true, false, "7-prefixed suffix never self-tests P4"},
    };
    for (const auto& test : optin_cases) {
        expect(d::fused_gr_p4_selftest_requested(d::fused_gr_env_variant(test.value), test.gfx12),
               test.expected, test.name);
    }

    // The latched variant: unset stays staged even when P4's parity passes; P4 needs the exact explicit request.
    struct SelectCase {
        std::string_view value;
        bool split_ok;
        bool staged_ok;
        bool p4_gate_ok;
        int expected;
        const char* name;
    };
    const SelectCase select_cases[] = {
        {"", true, true, true, d::kGrVariantStaged, "unset with every gate passing stays staged"},
        {"", true, true, false, d::kGrVariantStaged, "unset with P4 parity failing stays staged"},
        {"", true, false, false, d::kGrVariantSplit, "unset with staged failing falls back to split"},
        {"", false, false, false, d::kGrVariantPlain, "unset with nothing checked falls back to plain"},
        {"7", true, true, true, d::kGrVariantRegisterHalf, "=7 with both parity gates passes selects P4"},
        {"7", true, true, false, d::kGrVariantStaged, "=7 with P4 parity failing selects staged"},
        {"7", true, false, false, d::kGrVariantSplit, "=7 with staged failing falls back to split"},
        {"2", true, true, true, d::kGrVariantStaged, "=2 never selects P4"},
        {"1", true, true, true, d::kGrVariantSplit, "=1 selects split"},
        {"1", false, true, true, d::kGrVariantPlain, "=1 with split unchecked falls back to plain"},
        {"70", true, true, true, d::kGrVariantStaged, "=70 cannot select P4"},
        {"7suffix", true, true, true, d::kGrVariantStaged, "7-prefixed suffix cannot select P4"},
    };
    for (const auto& test : select_cases) {
        expect_eq(d::fused_gr_select(d::fused_gr_env_variant(test.value), test.split_ok, test.staged_ok, test.p4_gate_ok),
                  test.expected, test.name);
    }

    // A device-side fault poisons the shared context for the whole process, and the two headers do not share
    // every code: the faults both headers name are fatal in both backends, CUDA's device-fault and containment
    // codes are fatal for CUDA alone (HIP never returns them, and uses 911/912 for its own API errors), and the
    // launch refusals / API errors stay recoverable in both.
    using B = d::GrBackend;
    struct ErrCase {
        int err;
        B backend;
        bool expected;
        const char* name;
    };
    const ErrCase err_cases[] = {
        {d::kGrErrSuccess, B::Cuda, false, "success is not an error (CUDA)"},
        {d::kGrErrSuccess, B::Hip, false, "success is not an error (HIP)"},
        // the faults both headers name with the same number
        {d::kGrErrIllegalAddress, B::Cuda, true, "illegal address is context-fatal (CUDA)"},
        {d::kGrErrIllegalAddress, B::Hip, true, "illegal address is context-fatal (HIP)"},
        {d::kGrErrLaunchTimeout, B::Cuda, true, "launch timeout is context-fatal (CUDA)"},
        {d::kGrErrLaunchTimeout, B::Hip, true, "launch timeout is context-fatal (HIP)"},
        {d::kGrErrContextIsDestroyed, B::Cuda, true, "destroyed context is context-fatal (CUDA)"},
        {d::kGrErrContextIsDestroyed, B::Hip, true, "destroyed context is context-fatal (HIP)"},
        {d::kGrErrAssert, B::Cuda, true, "device assert is context-fatal (CUDA)"},
        {d::kGrErrAssert, B::Hip, true, "device assert is context-fatal (HIP)"},
        {d::kGrErrLaunchFailure, B::Cuda, true, "unspecified launch failure is context-fatal (CUDA)"},
        {d::kGrErrLaunchFailure, B::Hip, true, "device exception is context-fatal (HIP)"},
        // CUDA's device-fault and containment codes: fatal for CUDA, not this backend's codes for HIP
        {d::kGrCudaErrContained, B::Cuda, true, "contained device exception is context-fatal (CUDA)"},
        {d::kGrCudaErrContained, B::Hip, false, "HIP has no 226, so it is not fatal there"},
        {d::kGrCudaErrHardwareStackError, B::Cuda, true, "hardware stack error is context-fatal (CUDA)"},
        {d::kGrCudaErrHardwareStackError, B::Hip, false, "HIP has no 714, so it is not fatal there"},
        {d::kGrCudaErrIllegalInstruction, B::Cuda, true, "illegal instruction is context-fatal (CUDA)"},
        {d::kGrCudaErrIllegalInstruction, B::Hip, false, "HIP has no 715, so it is not fatal there"},
        {d::kGrCudaErrMisalignedAddress, B::Cuda, true, "misaligned address is context-fatal (CUDA)"},
        {d::kGrCudaErrMisalignedAddress, B::Hip, false, "HIP has no 716, so it is not fatal there"},
        {d::kGrCudaErrInvalidAddressSpace, B::Cuda, true, "invalid address space is context-fatal (CUDA)"},
        {d::kGrCudaErrInvalidAddressSpace, B::Hip, false, "HIP has no 717, so it is not fatal there"},
        {d::kGrCudaErrInvalidPc, B::Cuda, true, "invalid program counter is context-fatal (CUDA)"},
        {d::kGrCudaErrInvalidPc, B::Hip, false, "HIP has no 718, so it is not fatal there"},
        {d::kGrCudaErrTensorMemoryLeak, B::Cuda, true, "tensor memory leak is context-fatal (CUDA)"},
        {d::kGrCudaErrTensorMemoryLeak, B::Hip, false, "HIP has no 721, so it is not fatal there"},
        {d::kGrCudaErrMpsClientTerminated, B::Cuda, true, "MPS client termination is context-fatal (CUDA)"},
        {d::kGrCudaErrMpsClientTerminated, B::Hip, false, "HIP has no 810, so it is not fatal there"},
        {d::kGrCudaErrExternalDevice, B::Cuda, true, "external device error is context-fatal (CUDA)"},
        {d::kGrCudaErrExternalDevice, B::Hip, false, "HIP's 911 is hipErrorInvalidChannelDescriptor: recoverable"},
        // HIP's HSA runtime call failures: its header documents a call failing, not the context going bad
        {d::kGrHipErrRuntimeMemory, B::Hip, false, "HIP's HSA runtime memory failure is recoverable"},
        {d::kGrHipErrRuntimeOther, B::Hip, false, "HIP's HSA runtime failure is recoverable"},
        {d::kGrHipErrRuntimeMemory, B::Cuda, false, "CUDA has no 1052, so it is not fatal there"},
        {d::kGrHipErrRuntimeOther, B::Cuda, false, "CUDA has no 1053, so it is not fatal there"},
        // ordinary API and launch refusals: the context stays usable, so the fallback self-test can run
        {d::kGrErrInvalidValue, B::Cuda, false, "invalid value is recoverable (CUDA)"},
        {d::kGrErrInvalidValue, B::Hip, false, "invalid value is recoverable (HIP)"},
        {d::kGrErrMemoryAllocation, B::Cuda, false, "allocation failure is recoverable (CUDA)"},
        {d::kGrErrMemoryAllocation, B::Hip, false, "allocation failure is recoverable (HIP)"},
        {d::kGrErrInvalidConfiguration, B::Cuda, false, "invalid launch configuration is recoverable (CUDA)"},
        {d::kGrErrInvalidConfiguration, B::Hip, false, "invalid launch configuration is recoverable (HIP)"},
        {d::kGrErrLaunchOutOfResources, B::Cuda, false, "launch out of resources is recoverable (CUDA)"},
        {d::kGrErrLaunchOutOfResources, B::Hip, false, "launch out of resources is recoverable (HIP)"},
        {d::kGrErrInvalidDeviceFunction, B::Cuda, false, "invalid device function is recoverable (CUDA)"},
        {d::kGrErrInvalidDeviceFunction, B::Hip, false, "invalid device function is recoverable (HIP)"},
        {d::kGrErrInvalidKernelImage, B::Cuda, false, "invalid kernel image is recoverable (CUDA)"},
        {d::kGrErrInvalidKernelImage, B::Hip, false, "invalid image is recoverable (HIP)"},
        {d::kGrErrNoKernelImageForDevice, B::Cuda, false, "missing kernel image for device is recoverable (CUDA)"},
        {d::kGrErrNoKernelImageForDevice, B::Hip, false, "no binary for GPU is recoverable (HIP)"},
        {d::kGrErrInvalidResourceHandle, B::Cuda, false, "invalid resource handle is recoverable (CUDA)"},
        {d::kGrErrInvalidResourceHandle, B::Hip, false, "invalid resource handle is recoverable (HIP)"},
        {d::kGrErrNotReady, B::Cuda, false, "not ready is recoverable (CUDA)"},
        {d::kGrErrNotReady, B::Hip, false, "not ready is recoverable (HIP)"},
        {d::kGrErrLaunchIncompatibleTexturing, B::Cuda, false, "incompatible texturing mode is recoverable (CUDA)"},
        {d::kGrErrCooperativeLaunchTooLarge, B::Cuda, false, "cooperative launch too large is recoverable (CUDA)"},
        {d::kGrErrCooperativeLaunchTooLarge, B::Hip, false, "cooperative launch too large is recoverable (HIP)"},
        {d::kGrErrStreamCaptureUnsupported, B::Cuda, false, "stream capture refusal is recoverable (CUDA)"},
        {d::kGrErrStreamCaptureUnsupported, B::Hip, false, "stream capture refusal is recoverable (HIP)"},
        {d::kGrCudaErrInvalidClusterSize, B::Cuda, false, "invalid cluster size is recoverable (CUDA)"},
        {d::kGrCudaErrInvalidClusterSize, B::Hip, false, "HIP's 912 is hipErrorInvalidTexture: recoverable"},
        {d::kGrErrUnknown, B::Cuda, false, "unknown is not a documented context-fatal code (CUDA)"},
        {d::kGrErrUnknown, B::Hip, false, "unknown is not a documented context-fatal code (HIP)"},
    };
    for (const auto& test : err_cases) {
        expect(d::fused_gr_context_fatal(test.err, test.backend), test.expected, test.name);
    }

    std::printf("fused_gr_p4_policy_test: %zu gate + %zu marker + %zu env + %zu opt-in + %zu select + %zu fatal cases, %d failures\n",
                sizeof(cases) / sizeof(cases[0]), sizeof(marker_cases) / sizeof(marker_cases[0]),
                sizeof(env_cases) / sizeof(env_cases[0]), sizeof(optin_cases) / sizeof(optin_cases[0]),
                sizeof(select_cases) / sizeof(select_cases[0]), sizeof(err_cases) / sizeof(err_cases[0]), failures);
    return failures == 0 ? 0 : 1;
}
