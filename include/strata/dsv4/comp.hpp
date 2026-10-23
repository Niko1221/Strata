// include/strata/dsv4/comp.hpp - DeepSeek V4-Flash compressor block + compress-layer attention
// (dsv4-mvp slice S4).
//
// GPU port of src/dsv4/dsv4ref.cpp's compressor_forward() (:623) and the ratio>0 arm of
// attention_forward() (:967), FP32 semantics with the reference's double-precision pooling softmax:
//   kv/score = x @ compressor_{kv,gate}.T   (BF16 weights dequantized in-kernel, double-accumulated)
//   compressed token j: per-feature softmax over the pooling rows of (score + ape[(tok%ratio)])
//     dotted with kv; overlap (coff=2, r=4) pools rows [i<r: tok=j*r+i-r, dims 0:d] and
//     [i>=r: same tok, dims d:2d]; tok < 0 is skipped (-inf)
//   rmsnorm(compressor_norm) -> rope at position j*ratio with the YaRN compress tables
//     (theta 160000, factor 16, orig 65536, beta 32/1) -> cache slot j (prefill) / pos/ratio (decode)
// The compress-layer attention block adds the compressed tokens behind the sliding-window KV
// (offset s prefill / win decode, compress_topk's causal t >= (j+1)/ratio mask) and runs the same
// MQA-512 + inverse rope + grouped O projection as the hash layers (strata/dsv4/attn.hpp, S3).
#pragma once

#include <cstdint>
#include <string>

#include <cuda_runtime.h>

#include "strata/dsv4/attn.hpp"

namespace strata::dsv4 {

struct Dsv4Geometry;

// One layer's compressor weights, device pointers into the dense.bin arena.  kv/gate are RAW BF16 rows
// [W][n_embd] (pack kind 4, 2 B/elem - convert in the kernel, never dereference as f32); ape is F32
// [ratio][W] (GGUF shape {W, ratio}); norm F32 [kv_lora_rank].
struct Dsv4CompWeights {
    const uint16_t* wkv = nullptr;    // blk.L.attn_compressor_kv.weight   BF16 [W][4096]
    const uint16_t* wgate = nullptr;  // blk.L.attn_compressor_gate.weight BF16 [W][4096]
    const float* ape = nullptr;       // blk.L.attn_compressor_ape.weight  F32  [ratio][W]
    const float* norm = nullptr;      // blk.L.attn_compressor_norm.weight F32  [512]
};

// The compressor's pooling carry state (dsv4ref CompState): [coff*ratio][W] each, kv_state zeros /
// score_state -inf at init.  Device-resident; one per compress layer, alive across prefill->decode.
//
// raw_state is the pre-ape score mirror and exists for chunked prefill only (dsv4_comp_forward's
// chunk_prefill): when a pass boundary splits a ratio block, the block's pooling must still run in
// the reference's prefill order, which adds ape in double before the softmax, while the decode path
// adds ape in fp32 to the stored score and divides at the very end.  Carrying the un-added score lets
// the continuation pool the spanning block exactly as the single pass would have.  It is allocated
// with the other two (a few MB over the whole model) but written and read only under chunk_prefill,
// so the single-pass and decode paths never touch it.
struct Dsv4CompState {
    float* kv_state = nullptr;
    float* score_state = nullptr;
    float* raw_state = nullptr;
    int64_t ratio = 0, d = 0, coff = 0;
};

bool dsv4_comp_state_init(Dsv4CompState& st, int64_t ratio, int64_t d, int64_t coff,
                          cudaStream_t stream);
void dsv4_comp_state_free(Dsv4CompState& st);

// Compressor + compress-attention scratch.  W is the widest compressor width needed (1024 covers r=4
// and r=128); max_comp bounds the compressed tokens per call (max_s/4 + 1 covers both ratios).
struct Dsv4CompWorkspace {
    float* kv = nullptr;        // [max_s][W]
    float* score = nullptr;     // [max_s][W]
    float* kv_c = nullptr;      // [max_comp][d] compressed tokens this call
    float* shift = nullptr;     // [coff*ratio][W] decode overlap state-shift staging
    float* kv_all = nullptr;    // [max_s + max_comp][d] prefill [kv_new | kv_comp] concat
    int32_t* topk = nullptr;    // [max_s][win + max_comp] combined window+compress topk
    float* y = nullptr;         // [max_s][n_embd] attention-half hc_pre output / block input
    float* post = nullptr;      // [max_s][4]
    float* mixes = nullptr;     // [max_s][24] hc_pre dot results (mix -> fin kernel)
    float* comb = nullptr;      // [max_s][16]
    float* bout = nullptr;      // [max_s][n_embd] attention block output
    float* h2 = nullptr;        // [max_s][4*n_embd] hc_post staging
    int max_s = 0;
    int max_comp = 0;
    int64_t W = 0;
    int64_t state_rows = 0;     // coff*ratio, sized for the shift staging
};

bool dsv4_comp_workspace_init(Dsv4CompWorkspace& ws, int max_s, int64_t W, int max_comp, int win,
                              int64_t n_embd, int64_t state_rows);
void dsv4_comp_workspace_free(Dsv4CompWorkspace& ws);

// The compressor, dsv4ref.cpp:623 compressor_forward's exact semantics.  x is the attention block input
// [s][n_embd] (already hc_pre'd + rmsnormed).  Compressed tokens are rmsnorm(norm)+rope(pos=j*ratio,
// the YaRN compress table) written into out_cache slot j (prefill) / start_pos/ratio (decode); kv_out
// receives the [ncomp][d] rows when non-null.  Returns ncomp (host-deterministic, no sync).
// chunk_prefill marks a pass that continues a chunked prefill (start_pos > 0 and the pass boundary
// split a ratio block): pooling that spanning block keeps the prefill arithmetic through st.raw_state
// instead of the decode path's fp32 ape add plus trailing divide.  false = the unchanged path.
int64_t dsv4_comp_forward(const Dsv4CompWeights& w, Dsv4CompState& st, Dsv4CompWorkspace& ws,
                          const float* x, int64_t s, int64_t start_pos, const Dsv4RopeTables& rope,
                          double eps, float* out_cache, float* kv_out, cudaStream_t stream,
                          bool chunk_prefill = false);

// Host-side compressed-slot topk (dsv4ref compress_topk :875): prefill row j gets slots t + offset for
// t < (j+1)/ratio (-1 past the causal bound); decode gets slots offset..offset+(start_pos+1)/ratio-1.
// Writes into topk (row-major [s][nc], decode [1][n]); returns nc (0 when s < ratio).
int64_t dsv4_compress_topk(int64_t ratio, int64_t s, int64_t start_pos, int64_t offset,
                           int32_t* topk);

// The attention block of a compress layer (dsv4ref attention_forward, ratio > 0): the S3 q/kv front
// half, the compressor feeding the cache's compressed region, attention over the [window | compressed]
// topk against kv_all (prefill) / the cache (decode), inverse rope, grouped O projection.  rope must be
// the YaRN compress table (theta 160000).  kv_cache layout: [win][d] window ring then the compressed
// slots (win * d floats in).  ratio is 128 in S4 (the r=4 layers' sparse arm is S5's indexer).
bool dsv4_comp_attn_forward(const Dsv4AttnWeights& w, const Dsv4RopeTables& rope,
                            Dsv4AttnWorkspace& aws, const Dsv4CompWeights& cw, Dsv4CompState& cst,
                            Dsv4CompWorkspace& cws, const float* x, float* out, float* kv_cache,
                            int64_t ratio, int64_t s, int64_t start_pos, int64_t sliding_window,
                            cudaStream_t stream, bool chunk_prefill = false);

// =========================================================================================== [S5] indexer

// One r=4 layer's DSA indexer weights (dsv4ref load_layer :586): the query-side projection pair and the
// indexer's own compressor (ratio 4, d = idx_key_dim 128, coff 2, W = 256).  proj is RAW BF16 [64][4096]
// (pack kind 4 - errata #9: convert, never read as f32); wq_b is the pack's native Q8_0 [8192][1024].
struct Dsv4IndexerWeights {
    const uint16_t* proj = nullptr;   // blk.L.indexer.proj.weight          BF16 [64][4096]
    const uint8_t* wq_b = nullptr;    // blk.L.indexer.attn_q_b.weight      Q8_0 [8192][1024]
    Dsv4CompWeights comp;             // blk.L.indexer_compressor_{kv,gate,ape,norm}
};

// Indexer scratch.  q [max_s][64*128] and wgt [max_s][64] are downloaded to the host once per layer-pass:
// the EXACT top-512 selection contract (fp32 keys, stable sort, ties ascending) runs on the host, mirroring
// dsv4ref indexer_forward (:895) bit-for-bit in structure.  nc <= 4 at golden sizes, so the D2H is tiny.
struct Dsv4IndexerWorkspace {
    float* q = nullptr;      // [max_s][idx_heads * idx_key_dim]
    float* wgt = nullptr;    // [max_s][idx_heads]
    // [S7c] device-side selection scratch, grown on demand (dsv4_indexer_workspace_reserve):
    float* dp = nullptr;     // [s * nc][idx_heads]  q[j][h] . keys[t]
    float* sc = nullptr;     // [s * nc]             indexer score, fp32 (host contract)
    int32_t* tk = nullptr;   // [s][512]             selected indices, host contract's K = min(512, nc)
    int64_t cap = 0;         // entries reserved in dp / sc   (s * nc)
    int64_t cap_s = 0;       // rows reserved in tk
    int64_t max_s = 0;
};

/// Grow the device selection scratch to cover s*nc score slots and s top-k rows (2x headroom; a no-op
/// once the decode context stops growing).  False on allocation failure.
bool dsv4_indexer_workspace_reserve(Dsv4IndexerWorkspace& ws, int64_t s, int64_t nc);

bool dsv4_indexer_workspace_init(Dsv4IndexerWorkspace& ws, int64_t max_s);
void dsv4_indexer_workspace_free(Dsv4IndexerWorkspace& ws);

// The attention block of an r=4 layer (dsv4ref attention_forward, ratio == 4, :967): the S3 q/kv front
// half, the attention compressor feeding the cache's compressed region, then the indexer - q from the
// SHARED qr (indexer.attn_q_b, rope at the token position), the indexer compressor's fp32 keys
// (indexer_compressor_*, own 128-wide state), weights from indexer.proj (raw BF16) scaled by
// 128^-0.5 * 64^-0.5, score = sum_h relu(q.k) * wgt, EXACT top-512 (host stable_sort, prefill causal
// cutoff t < (j+1)/4, offset s prefill / win decode).  Attention runs over the [window | indexer topk]
// list against kv_all (prefill) / the cache (decode) through the same MQA-512 + inverse rope + grouped O
// projection as dsv4_comp_attn_forward.  idx_cache is the layer's indexer key arena slice
// ([ctx/4][128] fp32); ist is its compressor state.  rope must be the YaRN compress table.
bool dsv4_idx_attn_forward(const Dsv4AttnWeights& w, const Dsv4RopeTables& rope,
                           Dsv4AttnWorkspace& aws, const Dsv4CompWeights& cw, Dsv4CompState& cst,
                           const Dsv4IndexerWeights& iw, Dsv4CompState& ist, float* idx_cache,
                           Dsv4CompWorkspace& cws, Dsv4IndexerWorkspace& iws, const float* x, float* out,
                           float* kv_cache, int64_t s, int64_t start_pos, int64_t sliding_window,
                           cudaStream_t stream, bool chunk_prefill = false);

// The per-layer attention-half weights the engine gathers: block norm + hc_pre/hc_post parameters,
// then the S3 attention tensors, (compress layers) the compressor tensors, and (r=4 layers) the indexer.
struct Dsv4AttnHalfWeights {
    const float* attn_norm = nullptr;  // blk.L.attn_norm.weight, F32 [n_embd]
    const float* hc_fn = nullptr;      // blk.L.hc_attn_fn.weight,   F32 [24][4*n_embd]
    const float* hc_base = nullptr;    // blk.L.hc_attn_base.weight, F32 [24]
    const float* hc_scale = nullptr;   // blk.L.hc_attn_scale.weight, F32 [3]
    Dsv4AttnWeights attn;
    Dsv4CompWeights comp;              // comp.wkv == nullptr on the pure hash layers 0/1
    Dsv4IndexerWeights idx;            // idx.proj == nullptr except on the r=4 layers (even 2..42)
};

// One layer's attention half on the [s][4][n_embd] HC state h (dsv4ref run_pass's attn arm):
//   hash layers 0/1:  hc_pre -> rmsnorm(attn_norm) -> dense MLA (plain rope) -> hc_post
//   r=128 layers:     hc_pre -> rmsnorm -> compressor + MLA-over-compressed (YaRN rope) -> hc_post
//   r=4 layers (S5, even 2..42, L02 the hash+DSA hybrid included): hc_pre -> rmsnorm -> compressor +
//                     indexer top-512 sparse attention (YaRN rope) -> hc_post.
// comp_state may be null on hash layers; idx_state/idx_cache are referenced only on r=4 layers.
// Returns false + err on a hard failure.
bool dsv4_attn_half(const Dsv4Geometry& g, int64_t layer, const Dsv4AttnHalfWeights& w,
                    const Dsv4RopeTables& rope_plain, const Dsv4RopeTables& rope_yarn,
                    float* d_h, float* kv_cache, Dsv4CompState* comp_state, Dsv4CompState* idx_state,
                    float* idx_cache, Dsv4AttnWorkspace& aws, Dsv4CompWorkspace& cws,
                    Dsv4IndexerWorkspace& iws, int64_t s, int64_t start_pos, cudaStream_t stream,
                    std::string& err, bool chunk_prefill = false);

}  // namespace strata::dsv4
