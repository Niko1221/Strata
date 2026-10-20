// include/strata/spec/dflash_backend.hpp - what DflashDrafter (dflash_drafter.hpp) forwards to when it is not the
// llama.cpp path: the native fused-Metal drafter (metal/src/engine/dflash.cpp, kernels/dflash.metal).  Same contract
// as DflashDrafter's methods; the drafter front keeps the timers.
#pragma once

#include <cstdint>
#include <memory>
#include <string>

namespace strata::spec {

struct DflashBackend {
    virtual ~DflashBackend() = default;
    virtual bool load(const std::string& path, std::string& err) = 0;
    virtual int target_hidden() const = 0;
    virtual int block() const = 0;
    virtual bool inject(const float* feats, int n, int pos0, std::string& err) = 0;
    virtual int draft(int32_t id_last, int pos0, int max_n, float p_min, int32_t* out, float* conf) = 0;
    virtual void trim(int pos) = 0;
    virtual int pos_max() const = 0;
    virtual const float* lattice() const = 0;
};

/// The native backend (defined by the Metal engine's dflash.cpp); load() it before use.
std::unique_ptr<DflashBackend> make_dflash_native();

}  // namespace strata::spec
