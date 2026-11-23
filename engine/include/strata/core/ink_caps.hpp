// Per-device capability answers for strata-inkling (#6796, #6808).
//
// A layer split or expert split runs one stage per card, and the MoE launcher asks "may this card use the INT8 / sm_120 FP4 / WMMA kernels?"
// on every stage. Those answers used to be cached once per process (a function-local `static int cached`), filled on the first call, which
// happens on stage 0. On a mixed split (sm_120 + sm_86, or sm_86 + sm_60) the later stage then ran stage 0's kernel path: the sm_86 card ran
// the sm_120 FP4 kernels (corrupted prompt states, malformed tool calls), the sm_60 card the INT8 tensor-core kernels (wrong logits).
//
// PerDeviceFlag keeps one answer per device index. The rules are pure functions of the compute capability so a test can feed two different
// capability sets without a second card.
#pragma once

namespace strata::core {

constexpr int kInkCapDevices = 64;

inline bool ink_cap_int8_tc(int major) { return major >= 8; }
inline bool ink_cap_wmma_f16(int major) { return major == 7; }  // Volta sm_70 (V100) and Turing sm_75 (RTX 2080 Ti)
inline bool ink_cap_sm120(int major, int minor) { return major == 12 && (minor == 0 || minor == 1); }

// One cached boolean per device. get(dev, query) runs `query` once per device and keeps its answer; a negative device index (the
// device could not be read) answers false and caches nothing.
class PerDeviceFlag {
public:
    PerDeviceFlag() { for (int& c : v_) c = -1; }
    template <class Query>
    bool get(int dev, Query&& query) {
        if (dev < 0) return false;
        if (dev >= kInkCapDevices) dev = kInkCapDevices - 1;
        if (v_[dev] >= 0) return v_[dev] != 0;
        const bool r = query();
        v_[dev] = r ? 1 : 0;
        return r;
    }
private:
    int v_[kInkCapDevices];
};

}  // namespace strata::core
