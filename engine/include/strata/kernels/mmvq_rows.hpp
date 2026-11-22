#pragma once
// STRATA_MTP_HEAD_INDEX=1 (opt-in): which weight block an MMVQ output row reads.  native_mmvq_rows reads the MTP draft
// head's rows from the main head by index instead of from gather_rows' copy of them.  Header-only and host-callable, so
// tests/core/mmvq_rows_test.cpp checks the kernel's own addressing against such a copy on the CPU.

#include <cstddef>
#include <cstdint>

#if defined(__CUDACC__) || defined(__HIPCC__)
#define STRATA_MMVQ_ROWS_HD __host__ __device__
#else
#define STRATA_MMVQ_ROWS_HD
#endif

namespace strata::kernels {

// The block of output row `r` at reduction block `kbx`: row r of the matrix, or (Indexed) row rows[r].  Row rows[r] of
// the head is the bytes gather_rows(head, rows) puts in row r of its copy, so the indexed read takes the same operands
// as the plain read of the copy, and the kernel's arithmetic, which does not depend on the address, gives the same bits.
template<bool Indexed>
STRATA_MMVQ_ROWS_HD inline std::size_t mmvq_block(const int32_t* rows, int r, int blocks_per_row, int kbx) {
    return std::size_t(Indexed ? rows[r] : r) * std::size_t(blocks_per_row) + std::size_t(kbx);
}

}  // namespace strata::kernels
