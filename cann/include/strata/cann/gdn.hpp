// cann/include/strata/cann/gdn.hpp - gated delta net (the op qwen35moe spends 3 of 4 layers in).
//
// Contract matches ggml's GGML_OP_GATED_DELTA_NET for one head and one sequence:
//   state is stored transposed, row j of the buffer is column j of the matrix,
//   decay multiplies column i by exp(g[i]) (or the whole matrix by exp(g[0])),
//   delta[j] = (v[j] - <row_j, k>) * beta,  row_j += delta[j] * k,  attn[j] = <row_j, q> / sqrt(S).
#pragma once

#include <cstdint>

#include "strata/cann/device.hpp"
#include "strata/cann/ops.hpp"

namespace strata::cann {

struct GdnShape {
    int64_t S = 0;    // head dimension (S_v)
    int64_t T = 0;    // tokens
    bool kda = true;  // true: g carries one value per channel; false: one value per token
};

// CPU reference in float, same layout as the device version.
//   q,k,v: [T][S]   g: kda ? [T][S] : [T][1]   beta: [T][1]
//   state: [S][S] (transposed, see above)      attn: [T][S]
void gdn_ref(const float * q, const float * k, const float * v, const float * g, const float * beta, float * state,
             float * attn, const GdnShape & shape);

// fp16 device version. `state` is updated in place; `attn` receives the per-token output.
void gdn(Stream & stream, const Tensor & q, const Tensor & k, const Tensor & v, const Tensor & g,
         const Tensor & beta, Tensor & state, Tensor & attn, const GdnShape & shape);

// The same recurrence for a whole head batch in one pass: 8 operator calls per token instead of 8
// per head. q/k/v/attn are [T][H*S] with the heads contiguous inside a token, `decay` and `beta`
// are [T][H] (the decay already exponentiated), and state is [H][S][S], updated in place.
void gdn_batched(Stream & stream, const Tensor & q, const Tensor & k, const Tensor & v, const Tensor & decay,
                 const Tensor & beta, Tensor & state, Tensor & attn, int64_t S, int64_t H, int64_t T);

} // namespace strata::cann
