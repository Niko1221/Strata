// The 27B engine's per-card buffers (s27_engine.cu, s27_prefill.cu): the Engine of engine.hpp built for FNX_MODEL=27b.
#pragma once

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <functional>
#include <vector>

#include "kernels.hpp"
#include "model.hpp"
#include "owned.hpp"
#include "s27_attn.cuh"
#include "s27_ring3.cuh"

namespace fnx {

// a whole block's sum of v (in every thread): warps folded through shuffles, the warp sums through sh, broadcast from
// thread 0. This model's engine and prompt path share it for their block reductions (norms and the softmax
// denominator); sh holds blockDim.x / 32 floats.
static __device__ float s27_block_sum(float v, float* sh) {
    for (int o = 16; o; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    if ((threadIdx.x & 31) == 0) sh[threadIdx.x >> 5] = v;
    __syncthreads();
    v = threadIdx.x < (blockDim.x >> 5) ? sh[threadIdx.x] : 0.f;
    if (threadIdx.x < 32)
        for (int o = 16; o; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    if (threadIdx.x == 0) sh[0] = v;
    __syncthreads();
    const float r = sh[0];
    __syncthreads();
    return r;
}

// a lane's state on one card: its page table, recurrent states and conv histories; the engine's current-lane pointers
// in CardBuf are a copy of the lane's (Engine::use_lane)
struct LaneDev {
    int* pt = nullptr;
    std::vector<float*> S, chist;
    int* ctl = nullptr;               // [2]: the lane's position and tag in a batched window
    int* mrope_buf = nullptr;         // the lane's rotary table ([max_ctx][3]), made at its first image
    int* mrope = nullptr;             // mrope_buf while its sequence holds an image, else null (rotary position = cell)
};

// the widest q8_1 row a window quantizes: the draft layer's [e | h], the FFN's slice, the attention's heads
constexpr int S27_XQ_COLS = 2 * N > FFD_CARD_MAX ? (2 * N > S27_NQ * HD ? 2 * N : S27_NQ * HD) : FFD_CARD_MAX;
static_assert(S27_XQ_COLS >= NVH / NCARD * HS && S27_XQ_COLS >= S27_NQ * HD, "xq holds every row a window quantizes");

struct CardBuf {
    int dev = 0;
    cudaStream_t st = nullptr;
    cudaEvent_t ev0 = nullptr, ev1 = nullptr;
    float *R = nullptr, *mixed = nullptr, *bo = nullptr;   // the residual, a half's normed input, its sum across the cards
    float* hfin = nullptr;  // the last window's rows after the output norm: the draft layer's source (draft from_main)
    void* xq = nullptr;     // q8_1 rows of up to S27_XQ_COLS, with a split twin (q8_twin) add_norm writes directly
    int8_t* xtq = nullptr;
    float* xtd = nullptr;
    void* hq = nullptr;     // the FFN's swiglu as q8_1 rows of the card's FF slice, with its twin
    float *qkvz = nullptr, *ab = nullptr, *gdn_scratch = nullptr;
    float* qkv = nullptr;   // [T][S27_QKV]: the card's q | gate heads, K heads, V heads
    float *att = nullptr, *apart = nullptr;
    float* gu = nullptr;
    float* head_logits = nullptr;
    int *tk_pen_tok = nullptr, *tk_pen_cnt = nullptr, *tk_pen_n = nullptr, *tk_i = nullptr;   // top_tokens()
    float* tk_v = nullptr;
    unsigned* tk_err = nullptr;
    float* tk_par = nullptr;
    float* best_v = nullptr;
    int* best_i = nullptr;
    unsigned long long* am_keys = nullptr;
    unsigned* err = nullptr;          // the window's error bits (ring timeouts and range, NaN logits)
    // state: the current lane's
    std::vector<float*> S, chist;
    __half* pool = nullptr;           // the paged K/V pool (s27_attn.cuh): n_pages pages of S27_PAGE halves
    int n_pages = 0;
    int* pt = nullptr;                // the current lane's page table (device)
    // what commit() cuts a window back to: the state after each row, each kept count's histories
    std::vector<float*> snap, hsnap;
    float* snap_block = nullptr;      // every recurrent layer's snap (the prompt path borrows it between windows)
    size_t snap_block_bytes = 0;
    R3 r3;
    Ring ring;                        // FNX_RING_F32=1: the sums as fp32 words (kernels.cu's ring) instead of r3's
    bool ring32 = false;
    int* ctl = nullptr;               // [0] window position, [1] ring tag (as unsigned)
    int* mrope = nullptr;             // the current lane's rotary table
    char* stage[2] = {nullptr, nullptr};   // a parked state's copies pass through these page-locked halves
    cudaEvent_t stage_ev[2] = {nullptr, nullptr};
    std::vector<LaneDev> lanes;
    struct GraphSlot {
        cudaGraphExec_t exec = nullptr;
        std::vector<int> key;
    };
    std::vector<GraphSlot> win_graphs = std::vector<GraphSlot>(MAXT + 1), mtp_graphs = std::vector<GraphSlot>(MAXT + 1),
                           batch_graphs = std::vector<GraphSlot>(MAXT + 1);
    // batch_graphs[T]'s rows (LaneRows): [MAXT] each row's position word, then [MAXT] its page table; and their
    // recurrent state then conv history per layer (GdnLanes): [NL][MAXT] S, then [NL][MAXT] hist
    std::vector<const int**> lane_rows = std::vector<const int**>(MAXT + 1, nullptr);
    std::vector<float**> lane_gdn = std::vector<float**>(MAXT + 1, nullptr);
    int* row_pos = nullptr;
    std::vector<const int**> seg_ctl = std::vector<const int**>(MAXT + 1, nullptr);
    std::vector<int*> seg_row0 = std::vector<int*>(MAXT + 1, nullptr);
    CopyJob* rows_commit_jobs = nullptr;
    CopyJob* commit_jobs = nullptr;
    cudaEvent_t commit_ev = nullptr, rows_commit_ev = nullptr;
    // the draft layer: its token rows, source rows, [e | h] normed, residual, outputs after the head norm, head logits
    float *mtp_emb = nullptr, *mtp_src = nullptr, *mtp_cat = nullptr, *mtp_R = nullptr, *mtp_out = nullptr,
          *mtp_logits = nullptr, *mtp_bv = nullptr;
    int *mtp_bi = nullptr, *mtp_ctl = nullptr;
    int* mtp_pen = nullptr;           // the last row's penalties (penalize_row's pen, this card's draft-vocabulary indexes)
    float* mtp_pen_par = nullptr;
    float* mtp_tk_v = nullptr;
    int *mtp_tk_i = nullptr, *mtp_zero = nullptr;
    unsigned* mtp_tk_err = nullptr;
};

// what engine_common.cu reads of this model (engine_parts.hpp says what each is)
constexpr int ROW_W = N;
constexpr int LANE_ROW_COLS = 3;   // and the row's lane's rotary table
inline void lane_row_extra(const int** rows, int r, const LaneDev& L) { rows[2 * MAXT + r] = L.mrope; }
inline void card_lane(CardBuf& b, const LaneDev& L) {
    b.pt = L.pt;
    b.S = L.S;
    b.chist = L.chist;
    b.mrope = L.mrope;
}
inline cudaError_t reset_hists(CardBuf&) { return cudaSuccess; }   // the recurrent layers' are all it has
constexpr int HIST_JOBS = 0;
inline int hist_jobs(CopyJob*, const CardBuf&, const LaneDev*, size_t) { return 0; }
inline const float* draft_src(const CardBuf& b, bool from_main) { return from_main ? b.hfin : b.mtp_out; }
constexpr bool MTP_HEAD_EVERY_ROW = false;   // mtp_enqueue: the last row through the head
inline unsigned* err_word(CardBuf& b) { return b.err; }
inline const char* window_error(unsigned err, bool draft) {
    if (err & 1u) return "a ring sum timed out";
    if (err & 2u) return "a ring sum met a NaN or inf (or a value past fp16's range)";
    if (err & 16u) return draft ? "the draft layer's logits hold a NaN or inf" : "the head's logits hold a NaN or inf";
    if (err) return "an error bit no message names";
    return nullptr;
}
// mtp: unused (the draft layer's K/V is a slot of every page, so the pages carry it); hist_half: the conv history half
// the next window reads
std::vector<std::pair<char*, size_t>> state_pieces(CardBuf& b, const CardW& w, bool mtp, int pos, bool kv, unsigned hist_half,
                                                   const std::vector<int>& pages);

}  // namespace fnx
