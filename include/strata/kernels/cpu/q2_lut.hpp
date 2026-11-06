// include/strata/kernels/cpu/q2_lut.hpp - the table-lookup Q2_0 rows for CPUs with AVX but no AVX2 (Sandy / Ivy
// Bridge): an opt-in rung in front of q2_rows_any, STRATA_Q2_LUT=1.
//
// Why: on Sandy / Ivy Bridge pmaddubsw, pmaddwd, every vector shift and mulps issue on port 0 only, so the
// multiply-add rows of q2_avx1.cpp are port-0 bound (~2.4 weight x activation products per cycle on a Xeon
// E5-2603).  This rung replaces the multiply with pshufb lookups, which issue on ports 1 and 5 (~7.4 per cycle on
// the same core).  Per token and per pair of activations a 16-entry table holds the four codes' sums; the weights
// are repacked so one 16-byte vector holds a column pair's 4-bit index for 16 rows (src/kernels/cpu/q2_lut.cpp
// has the layout).  The integer part per 32-value activation chunk is exact - the same sums as q2_avx1.cpp - and
// only the float summation order differs (last bits), the same for a row whatever the row range or window width.
//
// The repacked copy (a "mirror", the same size as the GGUF rows) is made the first time a matrix is used, and
// only for weights inside a range registered as stable: memory that keeps the same expert at the same address
// while registered (ArenaExpertSource registers its arena).  Anything else - a reused staging buffer, a range that
// is not registered - is declined (false) and takes q2_rows_any as before.
#pragma once

#include "strata/kernels/cpu/expert.hpp"

#include <cstddef>
#include <cstdint>

namespace strata::kernels::cpu {

/// STRATA_Q2_LUT=1 on a CPU with AVX but no AVX2; STRATA_Q2_LUT=force on any AVX CPU (tests, A/B on newer CPUs).
bool q2_lut_enabled();

/// [base, base + bytes) holds weights that stay where they are until unregistered.  Unregistering drops the
/// mirrors made from it; it must not run while rows from that range are being computed.
void q2_lut_register_stable(const void* base, uint64_t bytes);
void q2_lut_unregister_stable(const void* base);

/// Rows [r0, r1) of the `rows`-row Q2_0 GGUF matrix at `w` (row_bytes = nblocks * 18) for `nt` tokens, like
/// q2_rows_any.  False - nothing written - when the rung is off or `w` is not in a registered range; the caller
/// then takes q2_rows_any.
bool q2_lut_rows(const uint8_t* w, size_t row_bytes, int nblocks, int rows, const ActQ* const* a, int nt,
                 float* const* out, int r0, int r1);

/// Bytes of mirrors made so far (for the load report and tests).
uint64_t q2_lut_mirror_bytes();

}  // namespace strata::kernels::cpu
