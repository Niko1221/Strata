// include/strata/kernels/lora.hpp - a LoRA adapter on the mixer's output projections: the engine's side of
// llama.cpp's `llama_adapter_lora` (`--lora` / `--lora-scaled`), for the two projections whose input is the
// mixer's output and whose output is the layer's residual write:
//
//   GDN layers:  ssm_out.weight       (ssm_value_dim -> n_embd)
//   QSA layers:  attn_output.weight   (n_head * head_dim -> n_embd)
//
// For layer l with an adapter there, the projection's output y = W x becomes
//
//   y <- y + s B (A x)        A = lora_a (rank x n_in), B = lora_b (n_out x rank), s = scale * alpha / rank
//
// exactly llama.cpp's `llm_build_lora_mm` (alpha 0: s = scale).  Several files are summed: their ranks are stacked,
// each B already multiplied by its own s.  A and B stay fp32 on every device that runs the layer (tens of MB), the
// products are fp32.  Off unless an adapter is loaded; a loaded one is switched per request through a device flag,
// so the captured graphs are the same either way.  With the flag off the projection's output is untouched: the
// result is bit-identical to the engine without an adapter.  The MTP draft layer never has one (it drafts; the
// verified tokens are the adapted model's).
#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace strata::kernels {

/// One layer's adapter as the loader builds it (host): `a` rank x n_in, `b` n_out x rank, B already scaled.
struct LoraLayerHost {
    int64_t rank = 0, n_in = 0, n_out = 0;
    std::vector<float> a, b;
};

/// Largest summed rank of one layer (a block holds a chunk of tokens' A x in shared memory).
constexpr int64_t kLoraMaxRank = 512;
/// Tokens a call may pass while its stream is being captured (the decode and verify windows); a longer call (the
/// prompt path, eager) uses a scratch grown on demand.
constexpr int64_t kLoraCaptureTokens = 64;

struct Lora {
    bool loaded = false;
    int64_t rank_max = 0;          ///< the largest summed rank of any layer
    std::vector<bool> adapted;     ///< per layer: an adapter on its output projection
    /// Layer l's output projection is followed by the adapter.  Decided at load time, never per request: it adds
    /// kernels to the graphs, so the graphs depend on it.
    bool covers(int64_t l) const { return loaded && l >= 0 && l < (int64_t) adapted.size() && adapted[(size_t) l]; }
};

/// The loaded adapter (empty until `lora_upload`).
const Lora& lora();

/// Upload the adapter built by the loader (`layers` has n_layers entries; rank 0 = none there).  Starts ON.  Once,
/// before any graph is captured (the graphs hold the tables' addresses).
bool lora_upload(const std::vector<LoraLayerHost>& layers, std::string& err);

/// A layer split: put the loaded adapter's tables on the CURRENT device too (lora_apply uses the tables of the
/// device it runs on).  No-op without an adapter or when this device has them already.
bool lora_replicate(std::string& err);

/// The per-request switch, on every device that holds the adapter.  Synchronizes them when it changes, so call
/// it between requests.
void lora_set_enabled(bool on);
bool lora_enabled();

/// Device bytes of the tables on one device (the VRAM a layer split's stage pays for them).
uint64_t lora_device_bytes();
/// The same for tables the loader built, before anything is uploaded (a layer split's VRAM estimate).
uint64_t lora_device_bytes(const std::vector<LoraLayerHost>& layers);

/// y[t * ldy + o] += s B (A x[t * ldx + :]) for t < n, o < n_out, gated by the device flag (off: nothing written).
/// No-op for a layer the adapter does not cover.  ldx / ldy 0 = dense rows.  Capturable for n <= kLoraCaptureTokens
/// (two kernels per 8 tokens through the layer's own scratch, no host sync); a longer eager call runs two tiled
/// products through a per-stream scratch grown on demand.
void lora_apply(int64_t layer, const float* x, int64_t ldx, int64_t n, float* y, int64_t ldy, void* stream);
/// The same with x as FP16 bits (the prompt path's gated attention, `attn_h`).
void lora_apply_f16(int64_t layer, const uint16_t* x, int64_t ldx, int64_t n, float* y, int64_t ldy, void* stream);

}  // namespace strata::kernels
