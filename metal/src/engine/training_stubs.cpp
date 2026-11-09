// metal/src/engine/training_stubs.cpp - training entry points with no Metal implementation.
//
// WHY THIS FILE EXISTS. The Metal build reuses the engine core through
// STRATA_METAL_HOST_SOURCES (src/core/layer.cpp, src/prefill/prefill.cpp and
// src/program/generate.cpp). The training work added --lora and SCORE to those
// shared files, but their implementations are CUDA-only: src/kernels/cuda/lora.cu
// launches CUDA kernels and src/prefill/score.cpp includes <cuda_runtime.h> and
// allocates device buffers. Neither can be compiled into a Metal target, so the
// link of `strata` ends with undefined strata::kernels::lora_* / Scorer symbols.
//
// WHY NOT gen_stubs.py. The generated stubs (stubs.cpp -> missing.cpp) stop the
// engine with the entry point's name. That is right for a kernel whose absence
// would silently produce wrong numbers, but wrong here: lora_apply and
// lora_apply_f16 are called on every layer unconditionally and are gated by a
// DEVICE FLAG (see include/strata/kernels/lora.hpp). With no adapter loaded an
// abort-on-call stub would kill a Metal engine on its first token, even though
// the intended behaviour is bit-identical to the unadapted model.
//
// SEMANTICS ON METAL. No adapter can be loaded (load_lora_adapters fails with an
// explicit message), SCORE is unavailable (Scorer::attach fails with an explicit
// message), the per-request flags are inert, and both apply calls are exact
// no-ops. That is exactly the state of a CUDA engine started without --lora.
// Training is therefore opt-in on CUDA only; inference on Metal is unchanged.

#include "strata/kernels/lora.hpp"
#include "strata/kernels/lora_adapters.hpp"
#include "strata/prefill/score.hpp"

namespace {

// Visible from both namespaces below without qualification.
const char* const kNoTraining =
    "LoRA adapters require a CUDA build (the Metal backend has no LoRA implementation)";
const char* const kNoScore =
    "SCORE requires a CUDA build (the Metal backend has no scoring head)";

}  // namespace

namespace strata::kernels {
namespace {

const Lora kNoAdapter{};

}  // namespace

const Lora& lora() { return kNoAdapter; }

bool lora_upload(const std::vector<LoraLayerHost>&, std::string& err) {
    err = kNoTraining;
    return false;
}

bool lora_update(const std::vector<LoraLayerHost>&, std::string& err) {
    err = kNoTraining;
    return false;
}

// No adapter exists on this backend, so there is nothing to put on another device.
bool lora_replicate(std::string&) { return true; }

void lora_set_enabled(bool) {}
bool lora_enabled() { return false; }

void lora_set_polarity(int) {}
int lora_polarity() { return 0; }

uint64_t lora_device_bytes() { return 0; }
uint64_t lora_device_bytes(const std::vector<LoraLayerHost>&) { return 0; }

// Gated by the device flag in the CUDA implementation; here the flag is always off.
void lora_apply(int64_t, const float*, int64_t, int64_t, float*, int64_t, void*) {}
void lora_apply_f16(int64_t, const uint16_t*, int64_t, int64_t, float*, int64_t, void*) {}

bool load_lora_adapters(const LoraAdapterFiles&, const core::ModelGeometry&, std::string& summary,
                        uint64_t& digest, std::string& err, bool) {
    summary = "0";
    digest = 0;
    err = kNoTraining;
    return false;
}

}  // namespace strata::kernels

namespace strata::prefill {

// Scorer's pimpl is private to the class; defining it here is what makes the
// unique_ptr destructor instantiable without bringing in the CUDA-only .cpp.
struct Scorer::Impl {};

Scorer::Scorer() : impl_(nullptr) {}
Scorer::~Scorer() = default;

bool Scorer::attach(Prefill&, const core::WeightTable&, const core::ModelGeometry&,
                    const core::NativeHead*, void*, std::string& err) {
    err = kNoScore;
    return false;
}

bool Scorer::run(const int64_t*, int64_t, const int32_t*, float*, std::string& err) {
    err = kNoScore;
    return false;
}

int64_t Scorer::scored() const { return 0; }
double Scorer::ms_head() const { return 0.0; }
int64_t Scorer::rows_per_block() const { return 0; }

const std::string& Scorer::note() const {
    static const std::string note = "SCORE unavailable on Metal";
    return note;
}

}  // namespace strata::prefill
