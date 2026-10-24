// include/strata/kernels/ep_kernels.hpp - the SYCL port's expert parallelism.  Card 0 runs every layer and pushes each
// layer's routed ids and q8_1 rows into card 1's inbox; card 1 computes the rows of the experts card 0 does not hold
// and writes the rows themselves (not their weighted sum) into card 0's parts_, so card 0's combine is the single-card
// one.  A flag holds the epoch its card's window graph advances in device memory (a host value would freeze at
// capture), and every spin is bounded.
#pragma once
#include <cstddef>
#include <cstdint>

namespace strata::kernels {

/// *ctr += 1: the first node of each card's window graph, so the flags this window raises carry a new epoch.
void ep_bump(uint32_t* ctr, void* stream);
/// Copy the routed ids and q8_1 rows into the peer's inbox and raise its flag to *ctr.  peer_ids, peer_xq and
/// peer_flag are the peer's memory as this card's kernels address it.
void ep_push(const int32_t* ids, int n_ids, const uint8_t* xq, size_t xq_bytes, int32_t* peer_ids, uint8_t* peer_xq,
             uint32_t* peer_flag, const uint32_t* ctr, void* stream);
/// Spin until *flag (this card's memory, written by the other card) reaches *ctr (this card's).  On the bound, *err
/// (mapped host memory, may be null) gets `code` and the wait gives up.
void ep_wait(const uint32_t* flag, const uint32_t* ctr, uint32_t spin_max, uint32_t* err, uint32_t code, void* stream);
/// The peer: copy the rows of the entries card 0 does not own (`res0`) into card 0's parts_ and raise *peer_flag to
/// *ctr, in one kernel (each work-group fences at system scope after its row; the last to finish raises the flag;
/// `done` is zero between calls).  With the flag in a separate later kernel, card 0 sometimes read a row before all of
/// it had arrived.
void ep_send_rows(float* peer_rows, const float* rows, const int32_t* ids, const int32_t* res0, int n, int64_t row,
                  uint32_t* done, uint32_t* peer_flag, const uint32_t* ctr, void* stream);

/// Copy `bytes` read past this card's caches, for data the other card wrote into this card's memory: an inbound PCIe
/// write does not invalidate this card's caches, so a plain load can return the previous window's data.  `bytes` is a
/// multiple of 4; both pointers are 4-byte aligned.
void ep_copy_uncached(void* dst, const void* src, size_t bytes, void* stream);

/// STRATA_EP_VERIFY=1: add to *mismatch (host) the peer-owned rows where peer_rows (read past the caches, as
/// ep_copy_uncached) differs from this card's parts.
void ep_compare(const float* parts, const float* peer_rows, const int32_t* ids, const int32_t* res0, int n, int64_t row,
                uint32_t* mismatch, void* stream);

/// resident_plan for one card of the pair.  An entry belongs to card 0 when its expert is resident there (`res0`),
/// else to card 1 (`res1`); an expert on neither is a plan error.  Role 1 is card 0, role 2 card 1: each plans only
/// its own entries, and counts[1] is their number.  res0 / res1 are this card's copies of both tables for the layer.
void resident_plan_ep(const int32_t* ids, int n_entries, int k, const int32_t* res0, const int32_t* res1, int role,
                      int n_expert, const uint8_t* cache_base, const unsigned long long* slot_off, long long blob,
                      int32_t* plan, long long capx, void* stream, uint32_t* plan_err);

}  // namespace strata::kernels
