// helper_prefill.hpp — strata-mi50, 2026-10-27
// The helper GPU computing a share of the prompt's experts (STRATA_PREFILL_HELPER, plugin ABI 5): the work space and the per-layer
// pipeline that strata_helper_prefill_* run in the plugin's own HIP runtime.  PIANO_MI50_PROMPT_2026-10-27.md.
#pragma once
#include <cstdint>
#include <string>
#include <vector>

namespace strata::core { class RemoteExperts; }

namespace strata::plugin {

struct PrefillWs;

/// Frees the work space (device memory, stream, pinned host buffers).  Safe on nullptr.
void prefill_destroy(PrefillWs* ws);
/// Allocates (or re-sizes) the work space for chunks of up to `t_max` tokens on `device`; `layer_ok[l]` = 1 for the layers it can compute.
bool prefill_reserve(PrefillWs*& ws, int device, int64_t t_max, std::vector<uint8_t>& layer_ok, uint16_t** x16, uint16_t** sum16,
                     std::string& err);
/// Queues one layer; the host inputs are read before return.
bool prefill_begin(PrefillWs* ws, const strata::core::RemoteExperts& remote, int64_t layer, int64_t t, int64_t k,
                   const uint16_t* x16, const int32_t* ids, const float* w, const uint8_t* take, std::string& err);
/// Waits for the queued layer and copies its sums (t x n_embd fp16) to `sum16`.
bool prefill_finish(PrefillWs* ws, uint16_t* sum16, double* ms_wait, std::string& err);

}  // namespace strata::plugin
