// src/core/verify.cpp - see include/strata/core/verify.hpp.
#include "strata/core/verify.hpp"
#if defined(_WIN32)
#include <intrin.h>
#endif

#include "strata/core/native_head.hpp"
#include "strata/kernels/iq_kernels.hpp"
#include "strata/kernels/cpu/expert_layout.hpp"
#include "strata/kernels/bf16_gemv.hpp"
#include "strata/kernels/cpu/expert.hpp"
#include "strata/kernels/elementwise.hpp"
#include "strata/kernels/fused_gr.hpp"
#include "strata/kernels/gr.hpp"
#include "strata/kernels/kv_q8.hpp"
#include "strata/kernels/native_mmvq.hpp"
#include "strata/kernels/native_moe.hpp"
#include "strata/kernels/native_qsa.hpp"
#include "strata/kernels/native_qsa_indexer.hpp"
#include "strata/kernels/native_router.hpp"
#include "strata/kernels/native_rope.hpp"
#include "strata/kernels/ngram.hpp"
#include "strata/kernels/ple.hpp"
#include "strata/kernels/qsa.hpp"
#include "strata/kernels/qsa_decode_attn.hpp"
#include "strata/kernels/qsa_select.hpp"
#include "strata/kernels/quantize_act.hpp"
#include "strata/kernels/rope.hpp"
#include "strata/kernels/s2_expert_grouped.hpp"
#include "strata/kernels/sampler.hpp"
#include "strata/kernels/shared_expert.hpp"
#include "strata/kernels/verify_kernels.hpp"

#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <immintrin.h>

namespace strata::core {
namespace {

constexpr float EPS = 1e-6f;
using Clock = std::chrono::steady_clock;
double ms_since(Clock::time_point t) { return std::chrono::duration<double, std::milli>(Clock::now() - t).count(); }
const bool g_dbg = std::getenv("STRATA_VERIFY_DEBUG") != nullptr;
#define VDBG(...) do { if (g_dbg) { std::fprintf(stderr, "verify dbg: " __VA_ARGS__); std::fflush(stderr); } } while (0)

struct Bump {
    uint8_t* base = nullptr;
    uint64_t used = 0;
    template <typename T> T* take(uint64_t n) {
        T* p = base ? (T*) (base + used) : nullptr;
        used += (n * sizeof(T) + 255) & ~255ull;
        return p;
    }
};

bool mapped(size_t bytes, void** h, void** d) {
    if (cudaHostAlloc(h, bytes, cudaHostAllocMapped) != cudaSuccess) return false;
    std::memset(*h, 0, bytes);
    return cudaHostGetDevicePointer(d, *h, 0) == cudaSuccess;
}

strata::kernels::QsaShapes shapes_of(const ModelGeometry& g) {
    strata::kernels::QsaShapes s = strata::kernels::qsa_real_shapes();
    s.n_head = g.n_head;
    s.n_head_kv = g.n_head_kv;
    s.head_dim = g.head_dim;
    s.idx_n_head = g.idx_q_heads;
    s.idx_dim = g.idx_key_dim;
    return s;
}

const WeightRef* need(const LayerView& v, const char* suffix, std::string& err) {
    const WeightRef* r = v.get(suffix);
    if (r == nullptr && err.empty()) err = v.name(suffix) + " is missing";
    return r;
}

bool native_of(const WeightRef* w, const std::string& name, std::string& err) {
    if (w == nullptr) return false;
    if (w->native_data == nullptr) {
        err = "verify: " + name + " is not served natively (run with --native)";
        return false;
    }
    return true;
}

}  // namespace

Verifier::~Verifier() {
    if (cs_) cudaStreamSynchronize(cs_);
    for (auto& e : exec_)
        if (e) cudaGraphExecDestroy(e);
    if (commit_exec_) cudaGraphExecDestroy(commit_exec_);
    if (cs_) cudaStreamDestroy(cs_);
    if (copy_) { cudaStreamSynchronize(copy_); cudaStreamDestroy(copy_); }
    if (side_) cudaStreamDestroy(side_);
    if (shs_) cudaStreamDestroy(shs_);
    for (cudaEvent_t e : {fork_, join_, abfork_, abjoin_, shfork_, shjoin_, res_ready_})
        if (e) cudaEventDestroy(e);
    if (arena_) cudaFree(arena_);
    if (stamps_) cudaFree(stamps_);
    void* hosts[] = {h_tok_, h_step_, h_pos_, h_commit_, h_ple_, h_out_, h_x_, h_ids_, h_w_, h_seq_, h_flag_, h_ymiss_,
                     h_flagA_, h_plan_, h_flagB_, h_pids_, h_pw_, h_pseq_, h_pleflag_, h_rows_, h_res_};
    for (void* h : hosts)
        if (h) cudaFreeHost(h);
}

bool Verifier::init(const WeightTable& wt, const ModelGeometry& g, SessionState& ss, const VerifyHits& hits,
                    const NativeHead* head, int max_t, std::string& err) {
    wt_ = &wt;
    g_ = &g;
    ss_ = &ss;
    hits_ = hits;
    head_ = head;
    max_t_ = max_t;
    if (max_t < 2 || max_t > strata::kernels::kVerifyMaxT || max_t > strata::kernels::cpu::MAXT) {
        err = "verify: the window must hold 2.." + std::to_string(strata::kernels::kVerifyMaxT) + " tokens";
        return false;
    }
    if (hits.res == nullptr || hits.cache_base == nullptr || hits.blob <= 0 || hits.slots <= 0) {
        err = "verify: needs the profile-filled VRAM expert tier (--expert-profile and --expert-cache)";
        return false;
    }
    std::string why;
    if (!layer_verify_compatible(why)) {
        err = "verify: " + why + " (the verify window reproduces the default native decode path)";
        return false;
    }
    if (!strata::kernels::fused_gr_supported(g.n_embd, g.hc, g.hc_lr) || ss.k != 10 || g.ssm_state_size != 128 ||
        g.ssm_d_conv != 4) {
        err = "verify: geometry differs from the artifact's";
        return false;
    }
    const WeightRef* wo = wt.find("output.weight");
    if (wo == nullptr) { err = "verify: output.weight is missing"; return false; }
    n_vocab_ = wo->ne1;

    const strata::kernels::QsaShapes s = shapes_of(g);
    cap_ = strata::kernels::qsa_selection_width(strata::kernels::kTopkMaxCells, s);
    max_blocks_ = ss.qsa_states[0].max_cells / s.idx_block + 2;
    attn_scratch_floats_ = (int64_t) strata::kernels::qsa_decode_attn_scratch_floats(cap_, s);

    const uint64_t T = (uint64_t) max_t, N = (uint64_t) g.n_embd, HC = (uint64_t) g.hc, K = (uint64_t) ss.k;
    const uint64_t C = (uint64_t) g.ssm_conv_channels, ZV = (uint64_t) g.ssm_value_dim, HV = (uint64_t) g.ssm_v_heads;
    const uint64_t NH = (uint64_t) g.n_head, HD = (uint64_t) g.head_dim, NKV = (uint64_t) g.n_head_kv;
    const uint64_t IQ = (uint64_t) g.idx_q_heads, ID = (uint64_t) g.idx_key_dim;
    const uint64_t nG = (uint64_t) g.n_gdn_layers(), nQ = (uint64_t) g.n_qsa_layers();
    const uint64_t HS = (uint64_t) strata::kernels::NG_HIST * strata::kernels::NG_HC_DIM;
    const uint64_t TS = (uint64_t) (s.idx_block - 1) * ID;
    const int max_in = (int) std::max<uint64_t>(std::max<uint64_t>(N, ZV), NH * HD);

    // ---- mapped staging
    bool ok = mapped(T * 4, (void**) &h_tok_, (void**) &m_tok_) &&
              mapped(T * strata::kernels::kStepCount * 4, (void**) &h_step_, (void**) &m_step_) &&
              mapped(T * NH * 4, (void**) &h_pos_, (void**) &m_pos_) &&
              mapped((2 + T) * 4 + 16, (void**) &h_commit_, (void**) &m_commit_) &&
              mapped(T * N * 4, (void**) &h_ple_, (void**) &m_ple_) &&
              mapped(T * 4 + 16, (void**) &h_out_, (void**) &m_out_) &&
              mapped(T * N * 4, (void**) &h_x_, (void**) &m_x_) &&
              mapped(T * K * 4, (void**) &h_ids_, (void**) &m_ids_) &&
              mapped(T * K * 4, (void**) &h_w_, (void**) &m_w_) &&
              mapped(64, (void**) &h_seq_, (void**) &m_seq_) &&
              mapped(64, (void**) &h_flag_, (void**) &m_flag_) &&
              mapped(64, (void**) &h_flagA_, (void**) &m_flagA_) &&
              mapped(64, (void**) &h_flagB_, (void**) &m_flagB_) &&
              mapped(T * K * N * 4, (void**) &h_ymiss_, (void**) &m_ymiss_) &&
              mapped(2 * T * K * 4, (void**) &h_pids_, (void**) &m_pids_) &&
              mapped(2 * T * K * 4, (void**) &h_pw_, (void**) &m_pw_) &&
              mapped(64, (void**) &h_pseq_, (void**) &m_pseq_) &&
              mapped(64, (void**) &h_pleflag_, (void**) &m_pleflag_) &&
              mapped(T * HC * N * 4, (void**) &h_rows_, (void**) &m_rows_);
    if (!ok) { err = "verify: mapped staging allocation failed"; return false; }
    // A plan block, one per token group: counts(4) | start(cap+1) | dst(cap) | tok(cap) | pad | ptr(cap u64) |
    // ptr2(cap u64) | start2(cap+1).  The main GPU's hits (`verify_hit_plan`, device) fill counts [groups, entries],
    // start, dst, tok and ptr; the pool's PCIe share (mapped, `sink_`) counts [-, entries, groups], dst, tok, ptr2 and
    // start2.
    {
        const int64_t cap = (int64_t) (T * K);
        plan_ptr_off_ = (4 + (cap + 1) + cap + cap + 1) & ~1ll;
        plan_i32_ = (plan_ptr_off_ + 4 * cap + (cap + 1) + 2) & ~1ll;
        plan_stride_ = plan_i32_ + 16;
        if (!mapped((size_t) plan_i32_ * 4 * 2 + 64, (void**) &h_plan_, (void**) &m_plan_)) {
            err = "verify: mapped plan allocation failed";
            return false;
        }
        sink_.counts = h_plan_;
        sink_.dst = h_plan_ + 4 + cap + 1;
        sink_.tok = sink_.dst + cap;
        sink_.ptr2 = (unsigned long long*) (h_plan_ + plan_ptr_off_) + cap;
        sink_.start2 = h_plan_ + plan_ptr_off_ + 4 * cap;
        sink_.cap = cap;
        sink_.publish = &Verifier::publish_plan;
        sink_.host_rows_only = strata::kernels::native_moe_combine_enabled();   // the gather-combine (post)
        sink_.fetch = &Verifier::fetch_dma;
        sink_.ctx = this;
    }
    // the residency snapshot, rounded up to whole 16-byte copies
    res_words_ = (g.n_layers * g.n_expert + 3) & ~3ll;
    if (!mapped((size_t) res_words_ * 4, (void**) &h_res_, (void**) &m_res_)) {
        err = "verify: mapped residency allocation failed";
        return false;
    }
    std::memcpy(h_res_, hits.res, (size_t) (g.n_layers * g.n_expert) * sizeof(int32_t));

    // ---- the device arena: the same sequence counted, then carved
    auto carve = [&](Bump& b) {
        tok_ = b.take<int32_t>(T); step_ = b.take<int32_t>(T * strata::kernels::kStepCount);
        pos_ = b.take<int32_t>(T * NH); commit_ = b.take<int32_t>(2 + T);
        ple_ = b.take<float>(T * N); emb_ = b.take<float>(T * N); R_ = b.take<float>(T * HC * N);
        mixed_ = b.take<float>(T * N); bo_ = b.take<float>(T * N);
        inj_ = b.take<float>(T * HC); inj2_ = b.take<float>(T * HC);
        lo_ = b.take<float>(T * (uint64_t) g.hc_lr); rs_ = b.take<float>(T * HC); xn_ = b.take<float>(T * HC * N);
        xq_ = b.take<uint8_t>(strata::kernels::native_q8_1_bytes(max_in, (int) T));
        xil_ = b.take<uint8_t>(strata::kernels::native_q8_1_il_bytes(max_in, (int) T));
        qkv_L_ = b.take<float>(nG * T * C); h_L_ = b.take<float>(nG * T * C);
        gate_L_ = b.take<float>(nG * T * HV); beta_L_ = b.take<float>(nG * T * HV);
        z_ = b.take<float>(T * ZV); y_ = b.take<float>(T * ZV); y_dummy_ = b.take<float>(T * ZV);
        qfull_ = b.take<float>(T * NH * 2 * HD); qcur_ = b.take<float>(T * NH * HD);
        kcur_ = b.take<float>(T * NKV * HD); vcur_ = b.take<float>(T * NKV * HD);
        idx_raw_L_ = b.take<float>(nQ * T * ID); qidx_ = b.take<float>(T * IQ * ID);
        scores_ = b.take<float>(T * (uint64_t) max_blocks_); sel_ = b.take<int32_t>(T * (uint64_t) cap_);
        attn_ = b.take<float>(T * NH * HD); attn32_ = b.take<float>(T * NH * HD);
        attn_scratch_ = b.take<float>(T * (uint64_t) attn_scratch_floats_);
        tail_snap_ = b.take<float>(nQ * TS);
        logits_ = b.take<float>(T * (uint64_t) g.n_expert); w_ = b.take<float>(T * K); ids_ = b.take<int32_t>(T * K);
        plogits_ = b.take<float>(T * (uint64_t) g.n_expert); pw_ = b.take<float>(T * K); pids_ = b.take<int32_t>(T * K);
        shared_ = b.take<float>(T * N); parts_ = b.take<float>(T * K * N); hit_out_ = b.take<float>(T * K * N);
        hit_slot_ = b.take<int32_t>(T * K); hit_dst_ = b.take<int32_t>(T * K); hit_count_ = b.take<int32_t>(4);
        ring_count_ = b.take<unsigned>(2);
        plan_ = b.take<int32_t>(2 * (uint64_t) plan_stride_); dplan_ = b.take<int32_t>(2 * (uint64_t) plan_stride_);
        res_ = b.take<int32_t>((uint64_t) res_words_); slot_ptr_ = b.take<unsigned long long>((uint64_t) hits.slots);
        staging_ = b.take<uint8_t>((uint64_t) kStagingBlobs * strata::kernels::cpu::expert_layout().max_blob);
        hit_xq_ = b.take<uint8_t>(T * (N / 32) * 34); hit_xs_ = b.take<float>(T * (N / 32));
        nat_xq_ = b.take<uint8_t>(T * (N / 32) * 36);
        hit_scratch_ = b.take<uint8_t>(std::max<uint64_t>(
            strata::kernels::moe_hit_grouped_scratch_bytes((int64_t) (T * K), g.n_embd, g.n_ff),
            strata::kernels::native_expert_scratch_bytes((int64_t) (T * K), g.n_ff)));
        head_mixed_ = b.take<float>(T * N); head_inj_ = b.take<float>(HC);
        sh_bf16_ = b.take<uint16_t>(T * N); sh_gate_ = b.take<float>(T * (uint64_t) g.n_ff);
        sh_up_ = b.take<float>(T * (uint64_t) g.n_ff); sh_g_ = b.take<float>(T + 4);
        head_logits_ = b.take<float>(T * (uint64_t) n_vocab_);
        hist_snap_ = b.take<float>(T * HS);
    };
    Bump count;
    carve(count);
    if (cudaMalloc(&arena_, count.used) != cudaSuccess) {
        err = "verify: the device arena (" + std::to_string(count.used >> 20) + " MiB) does not fit";
        return false;
    }
    cudaMemset(arena_, 0, count.used);
    Bump real;
    real.base = (uint8_t*) arena_;
    carve(real);
    sink_.staging = (unsigned long long) staging_;
    sink_.staging_cap = kStagingBlobs;
    (void) TS;
    {
        std::vector<unsigned long long> sp((size_t) hits.slots);
        for (int64_t i = 0; i < hits.slots; ++i)
            sp[(size_t) i] = (unsigned long long) (hits.cache_base + (hits.slot_off != nullptr ? hits.slot_off[i]
                                                                                              : (uint64_t) i * (uint64_t) hits.blob));
        if (cudaMemcpy(slot_ptr_, sp.data(), sp.size() * sizeof(unsigned long long), cudaMemcpyHostToDevice) != cudaSuccess) {
            err = "verify: the slot table could not be uploaded";
            return false;
        }
    }
    if (cudaStreamCreateWithFlags(&copy_, cudaStreamNonBlocking) != cudaSuccess) {
        err = "verify: copy stream create failed";
        return false;
    }
    if (cudaStreamCreateWithFlags(&cs_, cudaStreamNonBlocking) != cudaSuccess ||
        cudaStreamCreateWithFlags(&side_, cudaStreamNonBlocking) != cudaSuccess ||
        cudaStreamCreateWithFlags(&shs_, cudaStreamNonBlocking) != cudaSuccess ||
        cudaEventCreateWithFlags(&fork_, cudaEventDisableTiming) != cudaSuccess ||
        cudaEventCreateWithFlags(&join_, cudaEventDisableTiming) != cudaSuccess ||
        cudaEventCreateWithFlags(&abfork_, cudaEventDisableTiming) != cudaSuccess ||
        cudaEventCreateWithFlags(&abjoin_, cudaEventDisableTiming) != cudaSuccess ||
        cudaEventCreateWithFlags(&shfork_, cudaEventDisableTiming) != cudaSuccess ||
        cudaEventCreateWithFlags(&shjoin_, cudaEventDisableTiming) != cudaSuccess ||
        cudaEventCreateWithFlags(&res_ready_, cudaEventDisableTiming) != cudaSuccess) {
        err = "verify: stream create failed";
        return false;
    }
    std::fprintf(stderr, "strata verify: window up to %d tokens, %.1f MiB of device buffers\n", max_t,
                 (double) count.used / 1048576.0);
    return true;
}

const float* Verifier::final_R(int t) const { return R_ + (size_t) t * (size_t) (g_->hc * g_->n_embd); }

// ================================ THE WINDOW, AS CAPTURED ================================
//
// Plan v0.3 P6 (split window): with `groups_ == 2` the window's tokens are cut into two groups A = [0, T/2 up) and
// B = the rest, and the stream is ordered
//
//     pre(0,A) pre(0,B) | post(0,A) pre(1,A) | post(0,B) pre(1,B) | post(1,A) pre(2,A) | ...
//
// so the CPU computes A's experts of layer l while the GPU runs B's mixer and router of layer l, and B's experts
// while the GPU combines A and runs A's layer l+1.  B's mixer only needs A's mixer of the same layer (K/V, GDN
// state), never A's experts, so nothing waits that did not wait before.  Every token's arithmetic is unchanged.
bool Verifier::record_window(int T, cudaStream_t cs, std::string& err) {
    using namespace strata::kernels;
    const ModelGeometry& g = *g_;
    const WeightTable& wt = *wt_;
    SessionState& ss = *ss_;
    const int64_t N = g.n_embd, HC = g.hc, K = ss.k, C = g.ssm_conv_channels, ZV = g.ssm_value_dim;
    const int64_t HV = g.ssm_v_heads, HK = g.ssm_k_heads, NH = g.n_head, HD = g.head_dim, NKV = g.n_head_kv;
    const int64_t IQ = g.idx_q_heads, ID = g.idx_key_dim, NE = g.n_expert, MT = max_t_;
    const QsaShapes s = shapes_of(g);
    const GrShapes gs{g.n_embd, g.hc, g.hc_lr};
    const uint64_t gdn_floats = (uint64_t) g.ssm_state_size * g.ssm_v_heads * g.ssm_state_size +
                                (uint64_t) g.ssm_conv_channels * (g.ssm_d_conv - 1);
    const int64_t HS = (int64_t) NG_HIST * NG_HC_DIM;
    const int64_t TS = (s.idx_block - 1) * ID;
    const bool ple_on = ss.ple.ready();
    auto Rt = [&](int t) { return R_ + (size_t) t * HC * N; };
    const int G = (split_ && T >= 2) ? 2 : 1;
    const int tb_[2] = {0, (T + 1) / 2}, te_[2] = {G == 2 ? (T + 1) / 2 : T, T};
    groups_[T] = G;
    // --window-profile: row l < n_layers holds layer l's stages, row n_layers the window's own stamps
    auto stamp = [&](int64_t row, int i, cudaStream_t st) {
        if (stamps_ != nullptr && G == 1)
            gpu_stamp(stamps_ + ((size_t) T * (size_t) (g.n_layers + 1) + (size_t) row) * kStamps + i, st);
    };
    stamp(g.n_layers, 0, cs);

    // ---- the window's inputs, from mapped staging; the residency snapshot on the shared expert's branch (the first
    // hit plan waits for it)
    copy_i32_from_mapped(tok_, m_tok_, T, cs);
    copy_i32_from_mapped(step_, m_step_, (int64_t) T * kStepCount, cs);
    copy_i32_from_mapped(pos_, m_pos_, (int64_t) T * NH, cs);
    if (cudaEventRecord(shfork_, cs) != cudaSuccess || cudaStreamWaitEvent(shs_, shfork_, 0) != cudaSuccess) {
        err = "verify: the residency copy could not fork";
        return false;
    }
    copy_from_mapped((float*) res_, (const float*) m_res_, res_words_, shs_);   // int32 words, copied as they are
    if (cudaEventRecord(res_ready_, shs_) != cudaSuccess) {
        err = "verify: the residency copy could not join";
        return false;
    }

    // ---- the embeddings, broadcast to the hc streams
    if (const NativeEmbed* ne = native_embed()) {       // plan v0.3 P6: the GGUF-form table
        ne->gather_dev(tok_, T, emb_, cs);
        broadcast_streams(emb_, R_, N, (int) HC, T, cs);
    } else {
        const WeightRef* w = wt.find("token_embd.weight");
        if (w == nullptr || w->codebook_iq4nl || (w->code_bits != 2 && w->code_bits != 4 && w->code_bits != 8)) {
            err = "verify: token_embd.weight is missing or not an S2/S4/S8 tensor";
            return false;
        }
        const auto* codes = (const uint8_t*) w->data;
        const auto* scales = (const float*) (codes + w->codes_bytes);
        const auto* offsets = w->has_offset ? (const float*) (codes + w->codes_bytes + w->scales_bytes) : nullptr;
        const uint64_t row_codes = (uint64_t) (w->ne0 / (8 / w->code_bits));
        const uint64_t row_groups = (uint64_t) (w->ne0 / w->group_elems);
        embedding_gather_dev(codes, scales, offsets, tok_, T, w->ne0, w->code_bits, w->code_bias, w->group_elems,
                             row_codes, row_groups, emb_, cs);
        broadcast_streams(emb_, R_, N, (int) HC, T, cs);
    }

    // set_predict: the next layer's router on this layer's FFN input, on a side branch forked after the doorbell and
    // joined before the next layer rewrites that input
    auto predicts = [&](int64_t l) {
        return predict_ != nullptr && G == 1 && l + 1 < g.n_layers && native_router_enabled();
    };

    // per-layer state indices (GDN and QSA layers are numbered separately)
    std::vector<int64_t> gdn_idx((size_t) g.n_layers, -1), qsa_idx((size_t) g.n_layers, -1);
    {
        int64_t qi = 0, gi = 0;
        for (int64_t l = 0; l < g.n_layers; ++l) {
            if (is_qsa_layer(g, l)) qsa_idx[(size_t) l] = qi++;
            else gdn_idx[(size_t) l] = gi++;
        }
    }

    // the shared expert: in a window of one token group on a branch of its own that forks at the doorbell, beside the
    // routed experts (joined before the combine); in a split window inline
    const bool beside = G == 1;
    auto shared = [&](int64_t l, int grp) -> bool {
        const int tb = tb_[grp], te = te_[grp], n = te - tb;
        const LayerView v(wt, l);
        const WeightRef *wgi = need(v, "ffn_gate_inp_shexp.weight", err), *wsg = need(v, "ffn_gate_shexp.weight", err),
                        *wsu = need(v, "ffn_up_shexp.weight", err), *wsd = need(v, "ffn_down_shexp.weight", err);
        if (!wgi || !wsg || !wsu || !wsd) return false;
        if (!native_of(wsg, v.name("ffn_gate_shexp.weight"), err) || !native_of(wsu, v.name("ffn_up_shexp.weight"), err) ||
            !native_of(wsd, v.name("ffn_down_shexp.weight"), err))
            return false;
        NativeSharedWeights nsw;
        nsw.gate_type = wsg->native_type; nsw.gate_data = wsg->native_data;
        nsw.up_type = wsu->native_type; nsw.up_data = wsu->native_data;
        nsw.down_type = wsd->native_type; nsw.down_data = wsd->native_data;
        nsw.q8_1 = xq_;                                   // the main stream's next use is the next layer's mixer
        cudaStream_t ws = cs;
        if (beside) {
            if (cudaEventRecord(shfork_, cs) != cudaSuccess || cudaStreamWaitEvent(shs_, shfork_, 0) != cudaSuccess) {
                err = "verify: the shared expert's branch could not fork";
                return false;
            }
            ws = shs_;
        }
        if (!shared_expert_native_bf16())                 // the scalar gate's BF16 input (the native gate reads x)
            for (int t = tb; t < te; ++t) f32_to_bf16_bulk(mixed_ + t * N, sh_bf16_ + t * N, N, ws);
        try {
            shared_expert_multi(n, mixed_ + tb * N, sh_bf16_ + tb * N, nsw, (const uint16_t*) wgi->data,
                                sh_gate_ + (size_t) tb * g.n_ff, sh_up_ + (size_t) tb * g.n_ff, sh_g_ + tb, shared_ + tb * N,
                                N, g.n_ff, ws);
        } catch (const std::exception& e) {
            err = std::string("verify shared expert: ") + e.what();
            return false;
        }
        if (beside) {
            stamp(l, kLayerStamps, shs_);
            if (cudaEventRecord(shjoin_, shs_) != cudaSuccess) {
                err = "verify: the shared expert's branch could not join";
                return false;
            }
        }
        return true;
    };

    // ---------------------------------------------------------------- pre(l, group): up to the ring
    auto pre = [&](int64_t l, int grp) -> bool {
        const int tb = tb_[grp], te = te_[grp], n = te - tb;
        stamp(l, 0, cs);
        const LayerView v(wt, l);
        const char* pfx[2] = {"hc_attn_", "hc_ffn_"};
        const WeightRef *wn[2], *wd[2], *wu[2], *wi[2];
        for (int h = 0; h < 2; ++h) {
            wn[h] = need(v, (std::string(pfx[h]) + "norm.weight").c_str(), err);
            wd[h] = need(v, (std::string(pfx[h]) + "down.weight").c_str(), err);
            wu[h] = need(v, (std::string(pfx[h]) + "up.weight").c_str(), err);
            wi[h] = need(v, (std::string(pfx[h]) + "inject.weight").c_str(), err);
            if (!wn[h] || !wd[h] || !wu[h] || !wi[h]) return false;
        }
        bool pending = l > 0;   // the previous layer's FFN write, folded into this layer's first read
        if (l == 1 && ple_on) {
            if (grp == 0) {                                // the host reads the rows while layer 0 runs
                wait_flag_ge(m_pleflag_, 1, cs);
                copy_from_mapped(ple_, m_ple_, (int64_t) T * N, cs);
            }
            float* normalized = (float*) ((uint8_t*) ss.ple.scratch + ple_block_scratch_bytes());
            for (int t = tb; t < te; ++t) {
                gr_write(Rt(t), bo_ + t * N, inj2_ + t * HC, gs, Rt(t), cs);
                PleOut po;
                po.normalized = normalized;
                po.result = Rt(t);
                try {
                    ple_block(ple_ + t * N, Rt(t), ss.ple.hist, ss.ple.w, po, ss.ple.scratch, cs);
                    ple_history_advance(ss.ple.hist, normalized, cs);
                } catch (const std::exception& e) {
                    err = std::string("verify PLE: ") + e.what();
                    return false;
                }
                copy_from_mapped(hist_snap_ + (size_t) t * HS, ss.ple.hist, HS, cs);
            }
            pending = false;
        }
        auto gr_read_group = [&](int half, bool apply, float* inj_prev, float* inj_out) {
            FusedGrArgs fa[kFusedGrMaxT];
            for (int t = tb; t < te; ++t) {
                FusedGrArgs& a = fa[t - tb];
                a.R = Rt(t); a.R_out = Rt(t); a.apply = apply;
                a.bo_prev = bo_ + t * N; a.inj_prev = inj_prev + t * HC;
                a.w_norm = (const float*) wn[half]->data; a.w_down = (const uint16_t*) wd[half]->data;
                a.w_up = (const uint16_t*) wu[half]->data; a.w_inject = (const uint16_t*) wi[half]->data;
                a.eps = EPS; a.lo = lo_ + t * g.hc_lr; a.rs = rs_ + t * HC;
                a.inject_out = inj_out + t * HC; a.mixed = mixed_ + t * N;
            }
            fused_gr_read_multi(fa, n, xn_ + (size_t) tb * HC * N, cs);
        };
        gr_read_group(0, pending, inj2_, inj_);
        stamp(l, 1, cs);
        float* xm = mixed_ + tb * N;
        // the mixer's projections: a group of 2+ tokens is quantized with the interleaved copy the multi-token kernels
        // read
        auto quant = [&](const float* x, int64_t n_in) {
            if (n >= 2) native_quantize_q8_1_il(x, xq_, xil_, (int) n_in, n, cs);
            else native_quantize_q8_1(x, xq_, (int) n_in, n, cs);
        };
        auto proj = [&](const WeightRef* w, float* y, int64_t n_in, int64_t n_out) {
            if (n >= 2) native_mmvq_il(w->native_type, w->native_data, xq_, xil_, y, (int) n_in, (int) n_out, n, cs);
            else native_mmvq(w->native_type, w->native_data, xq_, y, (int) n_in, (int) n_out, n, cs);
        };
        try {
            if (!is_qsa_layer(g, l)) {
                // ======================= GDN =======================
                const WeightRef *wqkv = need(v, "attn_qkv.weight", err), *wg = need(v, "attn_gate.weight", err),
                                *wout = need(v, "ssm_out.weight", err), *wa = need(v, "ssm_alpha.weight", err),
                                *wb = need(v, "ssm_beta.weight", err), *wc = need(v, "ssm_conv1d.weight", err),
                                *wnm = need(v, "ssm_norm.weight", err), *wdt = need(v, "ssm_dt.bias", err),
                                *wsa = need(v, "ssm_a", err);
                if (!wqkv || !wg || !wout || !wa || !wb || !wc || !wnm || !wdt || !wsa) return false;
                if (!native_of(wqkv, v.name("attn_qkv.weight"), err) || !native_of(wg, v.name("attn_gate.weight"), err) ||
                    !native_of(wout, v.name("ssm_out.weight"), err))
                    return false;
                const int64_t gi = gdn_idx[(size_t) l];
                float* state = ss.gdn_state + (size_t) gi * gdn_floats;
                float* conv = state + (uint64_t) g.ssm_state_size * g.ssm_v_heads * g.ssm_state_size;
                float* qkv = qkv_L_ + (size_t) gi * MT * C;
                float* hb = h_L_ + (size_t) gi * MT * C;
                float* gate = gate_L_ + (size_t) gi * MT * HV;
                float* beta = beta_L_ + (size_t) gi * MT * HV;
                // alpha/beta beside the projections (a few blocks, latency-bound), joined before the recurrence
                if (cudaEventRecord(abfork_, cs) != cudaSuccess || cudaStreamWaitEvent(side_, abfork_, 0) != cudaSuccess) {
                    err = "verify: the GDN gates' branch could not fork";
                    return false;
                }
                gdn_ab_multi(xm, (const uint16_t*) wa->data, (const uint16_t*) wb->data, (const float*) wdt->data,
                             (const float*) wsa->data, gate + (size_t) tb * HV, beta + (size_t) tb * HV, (int) N, (int) HV,
                             n, side_);
                if (cudaEventRecord(abjoin_, side_) != cudaSuccess) {
                    err = "verify: the GDN gates' branch could not join";
                    return false;
                }
                quant(xm, N);
                proj(wqkv, qkv + (size_t) tb * C, N, C);
                gdn_conv_l2_multi(conv, qkv, (const float*) wc->data, hb, (int) C, (int) (2 * HK), EPS, n, cs, tb);
                proj(wg, z_ + (size_t) tb * ZV, N, ZV);
                if (cudaStreamWaitEvent(cs, abjoin_, 0) != cudaSuccess) {
                    err = "verify: the GDN gates' branch could not join";
                    return false;
                }
                // the recurrence from the untouched state over tokens [0, te); outputs only for this group's
                gdn_step_norm_multi(state, hb, (int) C, gate, beta, z_, (const float*) wnm->data, EPS, y_, (int) HK,
                                    (int) HV, te, nullptr, cs, tb);
                quant(y_ + (size_t) tb * ZV, ZV);
                proj(wout, bo_ + tb * N, ZV, N);
            } else {
                // ======================= QSA =======================
                const int64_t qi = qsa_idx[(size_t) l];
                const QsaState& st = ss.qsa_states[qi];
                const WeightRef *wik = need(v, "indexer.k_proj.weight", err), *wq = need(v, "attn_q.weight", err),
                                *wk = need(v, "attn_k.weight", err), *wv = need(v, "attn_v.weight", err),
                                *wo = need(v, "attn_output.weight", err), *wiq = need(v, "indexer.q_proj.weight", err),
                                *wqn = need(v, "attn_q_norm.weight", err), *wkn = need(v, "attn_k_norm.weight", err),
                                *wiqn = need(v, "indexer.q_norm.weight", err), *wikn = need(v, "indexer.k_norm.weight", err);
                if (!wik || !wq || !wk || !wv || !wo || !wiq || !wqn || !wkn || !wiqn || !wikn) return false;
                if (!native_of(wq, v.name("attn_q.weight"), err) || !native_of(wk, v.name("attn_k.weight"), err) ||
                    !native_of(wv, v.name("attn_v.weight"), err) || !native_of(wo, v.name("attn_output.weight"), err))
                    return false;
                auto norm_rope = [&](float* data, const WeightRef* norm, int rows, int cols, const int32_t* pos,
                                     cudaStream_t ns) {
                    if (native_qsa_enabled()) native_qsa_rms_norm_weighted(data, (const float*) norm->data, data, cols, rows, EPS, ns);
                    else rms_norm_weighted(data, (const float*) norm->data, rows, cols, EPS, ns);
                    if (native_rope_enabled()) native_rope_apply(data, data, rows, cols, (int) s.n_rot, (float) qsa_freq_base(), pos, ns);
                    else rope_neox_apply(data, data, rows, cols, (int) s.n_rot, st.cos_tab, st.sin_tab, pos, ns);
                };
                // the same for the n tokens' heads in one launch each: token t's positions at pos + t * NH
                const bool batched = native_qsa_enabled() && native_rope_enabled();
                auto norm_rope_tokens = [&](float* data, const WeightRef* norm, int heads, int cols, const int32_t* pos,
                                            cudaStream_t ns) {
                    native_qsa_rms_norm_weighted(data, (const float*) norm->data, data, cols, n * heads, EPS, ns);
                    native_rope_apply_tokens(data, data, n * heads, cols, (int) s.n_rot, (float) qsa_freq_base(), pos, heads,
                                             (int) NH, ns);
                };
                // the indexer - its keys appended, the queries' block scores and top-k - on a branch beside the
                // attention's projections, joined before the attention
                if (cudaEventRecord(abfork_, cs) != cudaSuccess || cudaStreamWaitEvent(side_, abfork_, 0) != cudaSuccess) {
                    err = "verify: the indexer's branch could not fork";
                    return false;
                }
                float* idx_raw = idx_raw_L_ + (size_t) qi * MT * ID;
                bf16_gemv_fp32_mmvf_multi(xm, (const uint16_t*) wik->data, idx_raw + tb * ID, N, ID, n, side_);
                if (grp == 0) copy_from_mapped(tail_snap_ + (size_t) qi * TS, st.idx_tail, TS, side_);
                const QsaIndexerBuffers ib{st.idx_tail, st.idx_dead, st.idx_pooled, st.idx_block_pos};
                native_qsa_indexer_append_multi(idx_raw + tb * ID, step_ + tb * kStepCount + kStepPos, (int) kStepCount, n,
                                                0, (const float*) wikn->data, EPS, ib, s, st.max_cells,
                                                (float) qsa_freq_base(), side_);
                bf16_gemv_fp32_mmvf_multi(xm, (const uint16_t*) wiq->data, qidx_ + tb * IQ * ID, N, IQ * ID, n, side_);
                if (batched) norm_rope_tokens(qidx_ + tb * IQ * ID, wiqn, (int) IQ, (int) ID, pos_ + tb * NH, side_);
                else for (int t = tb; t < te; ++t) norm_rope(qidx_ + t * IQ * ID, wiqn, (int) IQ, (int) ID, pos_ + t * NH, side_);
                qsa_block_scores(st.idx_pooled, st.idx_dead, qidx_ + tb * IQ * ID, step_ + tb * kStepCount, n, max_blocks_,
                                 s, scores_ + (size_t) tb * max_blocks_, side_, qsa_score_grid_blocks);
                qsa_block_topk(scores_ + (size_t) tb * max_blocks_, step_ + tb * kStepCount, n, max_blocks_, cap_, s,
                               sel_ + (size_t) tb * cap_, side_);
                if (cudaEventRecord(abjoin_, side_) != cudaSuccess) {
                    err = "verify: the indexer's branch could not join";
                    return false;
                }
                quant(xm, N);
                proj(wk, kcur_ + tb * NKV * HD, N, NKV * HD);
                proj(wv, vcur_ + tb * NKV * HD, N, NKV * HD);
                if (batched) norm_rope_tokens(kcur_ + tb * NKV * HD, wkn, (int) NKV, (int) HD, pos_ + tb * NH, cs);
                else for (int t = tb; t < te; ++t) norm_rope(kcur_ + t * NKV * HD, wkn, (int) NKV, (int) HD, pos_ + t * NH, cs);
                if (st.kv_int8)
                    kv_append_q8_steps(st.k_q, st.v_q, st.k_scale, st.v_scale, st.page_table, step_ + tb * kStepCount,
                                       (int) kStepCount, kcur_ + tb * NKV * HD, vcur_ + tb * NKV * HD, n, s, cs);
                else
                    for (int t = tb; t < te; ++t)
                        kv_append_step(st.k_pool, st.v_pool, st.page_table, step_ + t * kStepCount, kcur_ + t * NKV * HD,
                                       vcur_ + t * NKV * HD, s, cs);
                proj(wq, qfull_ + tb * NH * 2 * HD, N, NH * 2 * HD);
                // the query halves of every token's heads (rows of 2 * HD: query, then gate)
                if (cudaMemcpy2DAsync(qcur_ + tb * NH * HD, (size_t) HD * 4, qfull_ + tb * NH * 2 * HD, (size_t) HD * 2 * 4,
                                      (size_t) HD * 4, (size_t) (n * NH), cudaMemcpyDeviceToDevice, cs) != cudaSuccess) {
                    err = "verify: the q/gate split failed";
                    return false;
                }
                if (batched) norm_rope_tokens(qcur_ + tb * NH * HD, wqn, (int) NH, (int) HD, pos_ + tb * NH, cs);
                else for (int t = tb; t < te; ++t) norm_rope(qcur_ + t * NH * HD, wqn, (int) NH, (int) HD, pos_ + t * NH, cs);
                if (cudaStreamWaitEvent(cs, abjoin_, 0) != cudaSuccess) {
                    err = "verify: the indexer's branch could not join";
                    return false;
                }
                QsaAttnPools pools;
                pools.page_table = st.page_table;
                if (st.kv_int8) { pools.k_q = st.k_q; pools.v_q = st.v_q; pools.k_scale = st.k_scale; pools.v_scale = st.v_scale; }
                else { pools.k_pool = st.k_pool; pools.v_pool = st.v_pool; }
                qsa_decode_attn_batch(qcur_ + tb * NH * HD, pools, sel_ + (size_t) tb * cap_, step_ + tb * kStepCount, cap_,
                                      s, attn_scratch_ + (size_t) tb * attn_scratch_floats_, attn_ + tb * NH * HD, n, cs);
                if (native_qsa_enabled())
                    native_qsa_gate_apply(attn_ + tb * NH * HD, qfull_ + tb * NH * 2 * HD, attn32_ + tb * NH * HD,
                                          (int) (n * NH), (int) HD, cs);
                else
                    for (int t = tb; t < te; ++t)
                        qsa_gate_apply_f32(attn_ + t * NH * HD, qfull_ + t * NH * 2 * HD, s, attn32_ + t * NH * HD, cs);
                quant(attn32_ + tb * NH * HD, NH * HD);
                proj(wo, bo_ + tb * N, NH * HD, N);
            }
        } catch (const std::exception& e) {
            err = "verify layer " + std::to_string(l) + ": " + e.what();
            return false;
        }
        stamp(l, 2, cs);
        gr_read_group(1, true, inj_, inj2_);
        stamp(l, 3, cs);
        // the router, the doorbell and the main GPU's routed experts - the ones its VRAM tier holds, decided here from
        // the residency snapshot the pool reads too (it leaves their rows to the GPU)
        if (l == 0 && grp == 0 && cudaStreamWaitEvent(cs, res_ready_, 0) != cudaSuccess) {
            err = "verify: the residency copy could not join";
            return false;
        }
        int32_t* hit_plan = dplan_ + (size_t) grp * (size_t) plan_stride_;
        if (native_router_enabled()) {             // one kernel: logits, top 10, ring, hit plan
            const WeightRef* wr = need(v, "ffn_gate_inp.weight", err);
            if (!wr) return false;
            VerifyRouterArgs ra;
            ra.x = xm; ra.w = (const uint16_t*) wr->data; ra.logits = logits_ + tb * NE;
            ra.ids = ids_ + tb * K; ra.weights = w_ + tb * K;
            ra.x_out = m_x_ + tb * N; ra.ids_out = m_ids_ + tb * K; ra.w_out = m_w_ + tb * K; ra.seq = m_seq_;
            ra.ring = (uint32_t) (l * G + grp + 1);   // the rings so far (the host waits for this one)
            ra.res = res_ + (size_t) l * NE; ra.slot_ptr = slot_ptr_; ra.plan = hit_plan;
            ra.cap = (int) (MT * K); ra.ptr_off = (int) plan_ptr_off_;
            ra.counter = ring_count_; ra.n_tok = n; ra.n_embd = (int) N; ra.n_expert = (int) NE;
            try {
                verify_router(ra, cs);
            } catch (const std::exception& e) {
                err = "verify layer " + std::to_string(l) + " router: " + e.what();
                return false;
            }
        } else {
            for (int t = tb; t < te; ++t) {
                MoEBuffers mb = ss.moe;
                mb.logits = logits_ + t * NE; mb.ids = ids_ + t * K; mb.weights = w_ + t * K;
                if (!moe_route(wt, g, l, K, mb, mixed_ + t * N, cs, err, nullptr)) return false;
            }
            doorbell_publish(xm, ids_ + tb * K, w_ + tb * K, (int64_t) n * N, (int64_t) n * K, m_x_ + tb * N,
                             m_ids_ + tb * K, m_w_ + tb * K, m_seq_, cs);
            verify_hit_plan(ids_ + tb * K, (int) (n * K), (int) K, res_ + (size_t) l * NE, (int) NE, slot_ptr_, hit_plan,
                            MT * K, plan_ptr_off_, cs);
        }
        stamp(l, 4, cs);
        // after the doorbell: the shared expert's branch first (the longest), the routed experts' activations, and the
        // next layer's prediction
        if (beside && !shared(l, grp)) return false;
        if (strata::kernels::cpu::expert_layout().native)
            quantize_q8_1_rows(xm, n, N, nat_xq_ + (size_t) tb * (N / 32) * 36, cs);
        else
            quantize_q8_0_scaled(xm, hit_xq_ + (size_t) tb * (N / 32) * 34, hit_xs_ + (size_t) tb * (N / 32), (int64_t) n * N, cs);
        stamp(l, 5, cs);
        if (predicts(l)) {
            const WeightRef* wr1 = need(LayerView(wt, l + 1), "ffn_gate_inp.weight", err);
            if (!wr1) return false;
            if (cudaEventRecord(fork_, cs) != cudaSuccess || cudaStreamWaitEvent(side_, fork_, 0) != cudaSuccess) {
                err = "verify: the prediction branch could not fork";
                return false;
            }
            const size_t half = (size_t) ((l + 1) & 1) * (size_t) (MT * K);
            VerifyRouterArgs ra;
            ra.x = xm; ra.w = (const uint16_t*) wr1->data; ra.logits = plogits_; ra.ids = pids_; ra.weights = pw_;
            ra.ids_out = m_pids_ + half; ra.w_out = m_pw_ + half; ra.seq = m_pseq_;
            ra.ring = (uint32_t) (l + 1);              // the predictions published so far
            ra.counter = ring_count_ + 1; ra.n_tok = n; ra.n_embd = (int) N; ra.n_expert = (int) NE;
            try {
                verify_router(ra, side_);
            } catch (const std::exception& e) {
                err = "verify layer " + std::to_string(l) + " prediction: " + e.what();
                return false;
            }
            if (cudaEventRecord(join_, side_) != cudaSuccess) {
                err = "verify: the prediction branch could not join";
                return false;
            }
        }
        return beside || shared(l, grp);
    };

    // ---------------------------------------------------------------- post(l, group): experts, combine
    auto post = [&](int64_t l, int grp) -> bool {
        const int tb = tb_[grp], te = te_[grp], n = te - tb;
        const uint32_t ring = (uint32_t) (l * G + grp + 1);
        const int64_t cap = (int64_t) n * K, capx = (int64_t) max_t_ * K;
        struct Plan {   // a plan block's fields (`init`)
            const int32_t *counts = nullptr, *start = nullptr, *dst = nullptr, *tok = nullptr, *start2 = nullptr;
            const unsigned long long *ptr = nullptr, *ptr2 = nullptr;
        };
        auto plan_at = [&](const int32_t* pl) {
            Plan p;
            p.counts = pl;
            p.start = pl + 4;
            p.dst = p.start + capx + 1;
            p.tok = p.dst + capx;
            p.ptr = (const unsigned long long*) (pl + plan_ptr_off_);
            p.ptr2 = p.ptr + capx;
            p.start2 = pl + plan_ptr_off_ + 4 * capx;
            return p;
        };
        float* hit_out = hit_out_ + (size_t) tb * K * N;
        const auto& lay = strata::kernels::cpu::expert_layout();
        auto grouped = [&](const Plan& p, const unsigned long long* gp, const int32_t* gs, const int32_t* gn) {
            if (lay.native) {
                // the layer's GGUF formats (i-quant gate/up, Q2_0 / IQ4_NL down)
                const auto& f = lay.fmt[(size_t) l];
                const NativeExpertLayout L = native_expert_layout(f.gu_type, f.d_type, f.n_embd, f.n_ff);
                native_expert_grouped(L, gp, gs, gn, p.dst, p.tok, cap, cap,
                                      nat_xq_ + (size_t) tb * (N / 32) * 36, hit_scratch_, hit_out, cs);
            } else {
                moe_grouped_s2(gp, gs, gn, p.dst, p.tok, cap, cap, hit_xq_ + (size_t) tb * (N / 32) * 34,
                               hit_xs_ + (size_t) tb * (N / 32), hit_scratch_, hit_out, cs);
            }
        };
        // plan v0.3 P6: the VRAM groups at once; the PCIe groups once the pool has published them and the copy engine
        // has landed them in staging
        const Plan vram = plan_at(dplan_ + (size_t) grp * (size_t) plan_stride_);
        grouped(vram, vram.ptr, vram.start, vram.counts);
        stamp(l, 6, cs);
        Plan pcie;
        if (pcie_share_) {
            wait_flag_ge(m_flagA_, ring, cs);                  // the pool published this group's PCIe share
            stamp(l, 7, cs);
            int32_t* pl = plan_ + (size_t) grp * (size_t) plan_stride_;
            copy_i32_from_mapped(pl, m_plan_ + (size_t) grp * (size_t) plan_i32_, plan_i32_, cs);
            pcie = plan_at(pl);
            wait_flag_ge(m_flagB_, ring, cs);                  // it is in staging (DMA) or mapped
            if (sink_.pcie_mode == 2) {                        // stage it with a copy kernel, then point at staging
                const int64_t per = G == 2 ? kStagingBlobs / 2 : kStagingBlobs;
                uint8_t* stage = staging_ + (size_t) (grp * per) * lay.max_blob;
                fetch_blobs(pcie.ptr2, pcie.counts + 2, stage, (int64_t) lay.blob_bytes(l), (int) per, cs);
                rebase_ptrs((unsigned long long*) pcie.ptr2, pcie.counts + 2, stage, (int64_t) lay.blob_bytes(l), cs);
            }
            grouped(pcie, pcie.ptr2, pcie.start2, pcie.counts + 2);
        } else {
            stamp(l, 7, cs);
        }
        stamp(l, 8, cs);
        wait_flag_ge(m_flag_, ring, cs);                       // the CPU's share is in the mapped rows
        stamp(l, 9, cs);
        if (beside && cudaStreamWaitEvent(cs, shjoin_, 0) != cudaSuccess) {
            err = "verify: the shared expert's branch could not join";
            return false;
        }
        if (native_moe_combine_enabled()) {
            // only the host's rows cross PCIe; the GPU's come from `hit_out` (the pool leaves their rows unwritten)
            try {
                native_moe_gather_combine(hit_out, m_ymiss_ + (size_t) tb * K * N, vram.dst, vram.counts + 1, pcie.dst,
                                          pcie.counts != nullptr ? pcie.counts + 1 : nullptr, w_ + tb * K, shared_ + tb * N,
                                          bo_ + tb * N, N, K, n, cs);
            } catch (const std::exception& e) {
                err = "verify layer " + std::to_string(l) + " combine: " + e.what();
                return false;
            }
        } else {
            copy_from_mapped(parts_ + (size_t) tb * K * N, m_ymiss_ + (size_t) tb * K * N, (int64_t) n * K * N, cs);
            moe_hit_add(parts_ + (size_t) tb * K * N, hit_out, vram.dst, vram.counts + 1, cap, N, cs);
            if (pcie_share_) moe_hit_add(parts_ + (size_t) tb * K * N, hit_out, pcie.dst, pcie.counts + 1, cap, N, cs);
            for (int t = tb; t < te; ++t) {
                MoEBuffers mb = ss.moe;
                mb.weights = w_ + t * K; mb.shared = shared_ + t * N;
                if (!moe_combine_parts(g, l, K, mb, parts_ + (size_t) t * K * N, bo_ + t * N, cs, err)) return false;
            }
        }
        stamp(l, 10, cs);
        if (predicts(l) && cudaStreamWaitEvent(cs, join_, 0) != cudaSuccess) {
            err = "verify: the prediction branch could not join";
            return false;
        }
        if (l == g.n_layers - 1)
            for (int t = tb; t < te; ++t) gr_write(Rt(t), bo_ + t * N, inj2_ + t * HC, gs, Rt(t), cs);
        stamp(l, 11, cs);
        return true;
    };

    stamp(g.n_layers, 1, cs);
    for (int grp = 0; grp < G; ++grp)
        if (!pre(0, grp)) return false;
    for (int64_t l = 0; l < g.n_layers; ++l)
        for (int grp = 0; grp < G; ++grp) {
            if (!post(l, grp)) return false;
            if (l + 1 < g.n_layers && !pre(l + 1, grp)) return false;
        }
    stamp(g.n_layers, 2, cs);

    // ---- the head, T columns, and the argmax of each
    {
        const WeightRef *hn = wt.find("output_hc_norm.weight"), *hd = wt.find("output_hc_down.weight"),
                        *hu = wt.find("output_hc_up.weight");
        if (!hn || !hd || !hu) { err = "verify: an output_hc_* weight is missing"; return false; }
        for (int t = 0; t < T; ++t) {
            BlockBuffers bb = ss.block;
            bb.R = Rt(t);
            bb.mixed = head_mixed_ + t * N;
            if (head_ != nullptr && head_->loaded()) {
                if (!lm_head_mix(wt, g, bb, cs, err)) return false;
            } else if (!lm_head(wt, g, bb, head_logits_ + (size_t) t * n_vocab_, cs, err)) {
                return false;
            }
        }
        if (head_ != nullptr && head_->loaded()) {
            try {
                if (T >= 2) {
                    native_quantize_q8_1_il(head_mixed_, xq_, xil_, (int) N, T, cs);
                    native_mmvq_il(head_->type(), head_->weights(), xq_, xil_, head_logits_, (int) N, (int) n_vocab_, T, cs);
                } else {
                    native_quantize_q8_1(head_mixed_, xq_, (int) N, T, cs);
                    native_mmvq(head_->type(), head_->weights(), xq_, head_logits_, (int) N, (int) n_vocab_, T, cs);
                }
            } catch (const std::exception& e) {
                err = std::string("verify head: ") + e.what();
                return false;
            }
        }
        SamplerParams sp;
        sp.greedy = true;
        sp.temperature = 0.0f;
        sample_tokens(head_logits_, T, (int) n_vocab_, nullptr, 0, sp, m_out_, cs);
    }
    if (host_rows_) copy_from_mapped(m_rows_, R_, (int64_t) T * HC * N, cs);   // a plain copy kernel, here to host
    stamp(g.n_layers, 3, cs);
    stamp(g.n_layers, 4, cs);   // right after the last: one stamp's own cost
    return true;
}

bool Verifier::capture(int T, std::string& err) {
    if (exec_[T] != nullptr) return true;
    if (profile_ && stamps_ == nullptr &&
        cudaMalloc(&stamps_, (size_t) (strata::kernels::kVerifyMaxT + 1) * (size_t) (g_->n_layers + 1) * kStamps *
                                 sizeof(unsigned long long)) != cudaSuccess) {
        err = "verify: the profile's stamp buffer does not fit";
        return false;
    }
    if (cudaStreamBeginCapture(cs_, cudaStreamCaptureModeThreadLocal) != cudaSuccess) {
        err = "verify: begin capture failed";
        return false;
    }
    std::string rerr;
    const bool ok = record_window(T, cs_, rerr);
    cudaGraph_t graph = nullptr;
    const cudaError_t ce = cudaStreamEndCapture(cs_, &graph);
    if (!ok) {
        if (graph) cudaGraphDestroy(graph);
        err = rerr;
        return false;
    }
    if (ce != cudaSuccess) {
        err = std::string("verify: end capture: ") + cudaGetErrorString(ce);
        return false;
    }
    const cudaError_t ie = cudaGraphInstantiate(&exec_[T], graph, 0);
    cudaGraphDestroy(graph);
    if (ie != cudaSuccess) {
        err = std::string("verify: instantiate: ") + cudaGetErrorString(ie);
        return false;
    }
    const cudaError_t ue = cudaGraphUpload(exec_[T], cs_);
    const cudaError_t us = cudaStreamSynchronize(cs_);
    std::fprintf(stderr, "strata verify: captured the %d-token window (upload %s, sync %s)\n", T,
                 cudaGetErrorString(ue), cudaGetErrorString(us));
    return true;
}

bool Verifier::capture_commit(std::string& err) {
    if (commit_exec_ != nullptr) return true;
    using namespace strata::kernels;
    const ModelGeometry& g = *g_;
    SessionState& ss = *ss_;
    const QsaShapes s = shapes_of(g);
    const int64_t C = g.ssm_conv_channels, HV = g.ssm_v_heads, ID = g.idx_key_dim, MT = max_t_;
    const uint64_t gdn_floats = (uint64_t) g.ssm_state_size * g.ssm_v_heads * g.ssm_state_size +
                                (uint64_t) g.ssm_conv_channels * (g.ssm_d_conv - 1);
    const int64_t TS = (s.idx_block - 1) * ID;
    const int64_t HS = (int64_t) NG_HIST * NG_HC_DIM;
    if (cudaStreamBeginCapture(cs_, cudaStreamCaptureModeThreadLocal) != cudaSuccess) {
        err = "verify: begin commit capture failed";
        return false;
    }
    bool ok = true;
    try {
        copy_i32_from_mapped(commit_, m_commit_, 2 + MT, cs_);
        int64_t qsa_index = 0, gdn_index = 0;
        for (int64_t l = 0; l < g.n_layers && ok; ++l) {
            const LayerView v(*wt_, l);
            if (!is_qsa_layer(g, l)) {
                const WeightRef* wnm = need(v, "ssm_norm.weight", err);
                if (!wnm) { ok = false; break; }
                float* state = ss.gdn_state + (size_t) gdn_index * gdn_floats;
                float* conv = state + (uint64_t) g.ssm_state_size * g.ssm_v_heads * g.ssm_state_size;
                const float* qkv = qkv_L_ + (size_t) gdn_index * MT * C;
                gdn_conv_commit(conv, qkv, (int) C, commit_, cs_);
                gdn_step_norm_multi(state, h_L_ + (size_t) gdn_index * MT * C, (int) C, gate_L_ + (size_t) gdn_index * MT * HV,
                                    beta_L_ + (size_t) gdn_index * MT * HV, z_, (const float*) wnm->data, EPS, y_dummy_,
                                    (int) g.ssm_k_heads, (int) HV, (int) MT, commit_, cs_);
                ++gdn_index;
            } else {
                const QsaState& st = ss.qsa_states[qsa_index];
                const WeightRef* wikn = need(v, "indexer.k_norm.weight", err);
                if (!wikn) { ok = false; break; }
                copy_from_mapped(st.idx_tail, tail_snap_ + (size_t) qsa_index * TS, TS, cs_);
                const QsaIndexerBuffers ib{st.idx_tail, st.idx_dead, st.idx_pooled, st.idx_block_pos};
                native_qsa_indexer_append_multi(idx_raw_L_ + (size_t) (qsa_index * MT) * ID, commit_ + 2, 1, (int) MT, 0,
                                                (const float*) wikn->data, EPS, ib, s, st.max_cells,
                                                (float) qsa_freq_base(), cs_);
                ++qsa_index;
            }
        }
        if (ok && ss.ple.ready()) copy_indexed(ss.ple.hist, hist_snap_, HS, commit_ + 1, HS, cs_);
    } catch (const std::exception& e) {
        err = std::string("verify commit: ") + e.what();
        ok = false;
    }
    cudaGraph_t graph = nullptr;
    const cudaError_t ce = cudaStreamEndCapture(cs_, &graph);
    if (!ok) {
        if (graph) cudaGraphDestroy(graph);
        return false;
    }
    if (ce != cudaSuccess || cudaGraphInstantiate(&commit_exec_, graph, 0) != cudaSuccess) {
        if (graph) cudaGraphDestroy(graph);
        err = std::string("verify: commit capture: ") + cudaGetErrorString(ce);
        return false;
    }
    cudaGraphDestroy(graph);
    return true;
}

bool Verifier::run(int T, const int32_t* tokens, int64_t pos0, PoolMultiFn pool, void* user, int32_t* out,
                   std::string& err) {
    using namespace strata::kernels;
    if (T < 1 || T > max_t_) { err = "verify: window size out of range"; return false; }
    const ModelGeometry& g = *g_;
    SessionState& ss = *ss_;
    if (pos0 + T > ss.qsa_states[0].max_cells) { err = "verify: the window runs past the context"; return false; }
    if (!capture(T, err) || !capture_commit(err)) return false;
    VDBG("captured; staging\n");
    const Clock::time_point t0 = Clock::now();
    const QsaShapes s = shapes_of(g);
    for (int t = 0; t < T; ++t) {
        h_tok_[t] = tokens[t];
        qsa_step_fill(h_step_ + t * kStepCount, pos0 + t, s);
        for (int64_t h = 0; h < g.n_head; ++h) h_pos_[t * g.n_head + h] = (int32_t) (pos0 + t);
    }
    const bool ple_on = ss.ple.ready();
    if (ple_on) {   // the reads start here unless ple_ahead started them
        int32_t prev[2] = {ss.ple_prev[0], ss.ple_prev[1]};
        for (int t = 0; t < T; ++t) {
            uint32_t rows[PLE_N_HEADS];
            ngram_rows(&tokens[t], prev, 1, ss.ple.consts, rows);
            if (!ss.ple.table->ahead_holds(t, rows)) ss.ple.table->ahead(t, rows);
            prev[0] = prev[1];
            prev[1] = tokens[t];
        }
    }
    // the residency this window decides the main GPU's hits by, and the pool reads (`residency`)
    std::memcpy(h_res_, hits_.res, (size_t) (g.n_layers * g.n_expert) * sizeof(int32_t));
    *(volatile uint32_t*) h_seq_ = 0;
    *(volatile uint32_t*) h_flag_ = 0;
    *(volatile uint32_t*) h_flagA_ = 0;
    *(volatile uint32_t*) h_flagB_ = 0;
    *(volatile uint32_t*) h_pseq_ = 0;
    *(volatile uint32_t*) h_pleflag_ = 0;
    std::atomic_thread_fence(std::memory_order_seq_cst);
    last_t_ = T;
    last_pos0_ = pos0;
    for (int t = 0; t < T; ++t) last_tokens_[t] = tokens[t];
    ms_host += ms_since(t0);
    VDBG("staged; launching\n");
    const cudaError_t le = cudaGraphLaunch(exec_[T], cs_);
    if (le != cudaSuccess) { err = std::string("verify: launch: ") + cudaGetErrorString(le); return false; }
    (void) cudaStreamQuery(cs_);
    VDBG("launched\n");
    if (ple_on) {   // before layer 1: the graph waits for the flag, raised even when a read failed
        const Clock::time_point tp = Clock::now();
        bool ok = true;
        for (int t = 0; t < T && ok; ++t) ok = ss.ple.table->ahead_collect(t, h_ple_ + (size_t) t * g.n_embd, err);
        std::atomic_thread_fence(std::memory_order_seq_cst);
        _mm_sfence();
        *(volatile uint32_t*) h_pleflag_ = 1;
        ms_host += ms_since(tp);
        if (!ok) return false;
    }
    volatile uint32_t* const seq = h_seq_;
    volatile uint32_t* const flag = h_flag_;
    const int G = groups_[T] > 0 ? groups_[T] : 1;
    const int gtb[2] = {0, (T + 1) / 2}, gte[2] = {G == 2 ? (T + 1) / 2 : T, T};
    for (int64_t k = 0; k < g.n_layers * G; ++k) {
        const int64_t l = k / G;
        const int grp = (int) (k % G);
        const uint32_t want = (uint32_t) (k + 1);
        const Clock::time_point a = Clock::now();
        auto last_flush = a;
        uint32_t spins = 0;
        while (*seq < want) {
            _mm_pause();
            if ((++spins & 1023u) != 0) continue;
            const auto now = Clock::now();
            if (now - last_flush > std::chrono::microseconds(2000)) {
                last_flush = now;
                const cudaError_t q = cudaStreamQuery(cs_);
                if (q != cudaErrorNotReady && *seq < want) {
                    err = "verify: layer " + std::to_string(l) + " never rang (" +
                          (q == cudaSuccess ? std::string("graph finished") : std::string(cudaGetErrorString(q))) + ")";
                    return false;
                }
            }
            if (now - a > std::chrono::seconds(20)) { err = "verify: timed out at layer " + std::to_string(l); return false; }
        }
        const Clock::time_point b = Clock::now();
        VDBG("layer %lld rang\n", (long long) l);
        cur_layer_ = want - 1;
        set_plan_slot(grp);
        const int tb = gtb[grp], n = gte[grp] - gtb[grp];
        if (pool != nullptr)
            pool(user, h_x_ + (size_t) tb * g.n_embd, h_ids_ + (size_t) tb * ss.k, n, ss.k,
                 h_ymiss_ + (size_t) tb * ss.k * g.n_embd, l);
        VDBG("layer %lld served\n", (long long) l);
        std::atomic_thread_fence(std::memory_order_seq_cst);
        _mm_sfence();
        if (pcie_share_ && *(volatile uint32_t*) h_flagA_ != want) {   // no PCIe share published: an empty one
            sink_.counts[0] = 0;
            sink_.counts[1] = 0;
            sink_.counts[2] = 0;
            sink_.start2[0] = 0;
            std::atomic_thread_fence(std::memory_order_seq_cst);
            *(volatile uint32_t*) h_flagA_ = want;
            raise_flag(h_flagB_, want);
        }
        *flag = want;
        ms_wait += std::chrono::duration<double, std::milli>(b - a).count();
        ms_pool += ms_since(b);
        // the next layer's prediction, published during this layer's pool, else microseconds later (its branch is
        // joined before the next layer): waited for, so the experts the second GPU takes never depend on timing
        if (predict_ != nullptr && G == 1 && l + 1 < g.n_layers && native_router_enabled()) {
            const Clock::time_point c = Clock::now();
            if (*(volatile uint32_t*) h_pseq_ < (uint32_t) (l + 1)) {
                ++predict_late;
                while (*(volatile uint32_t*) h_pseq_ < (uint32_t) (l + 1) && Clock::now() - c < std::chrono::milliseconds(50))
                    _mm_pause();
            }
            if (*(volatile uint32_t*) h_pseq_ >= (uint32_t) (l + 1)) {
                std::atomic_thread_fence(std::memory_order_acquire);
                const size_t half = (size_t) ((l + 1) & 1) * (size_t) max_t_ * (size_t) ss.k;
                predict_(predict_user_, l + 1, h_pids_ + half, h_pw_ + half, T, ss.k);
            }
            ms_predict += ms_since(c);
        }
    }
    const cudaError_t se = cudaStreamSynchronize(cs_);
    if (se != cudaSuccess) { err = std::string("verify: ") + cudaGetErrorString(se); return false; }
    cudaStreamSynchronize(copy_);   // no host function of this window may raise flag B in the next one
    for (int t = 0; t < T; ++t) out[t] = ((volatile int32_t*) h_out_)[t];
    if (stamps_ != nullptr && G == 1) {
        const int64_t NL = g.n_layers;
        h_stamps_.resize((size_t) (NL + 1) * kStamps);
        cudaMemcpy(h_stamps_.data(), stamps_ + (size_t) T * (size_t) (NL + 1) * kStamps,
                   h_stamps_.size() * sizeof(unsigned long long), cudaMemcpyDeviceToHost);
        if (prof_ns_.empty()) prof_ns_.assign(2 * kStamps, 0.0);
        const unsigned long long* w = h_stamps_.data() + (size_t) NL * kStamps;
        for (int64_t l = 0; l < NL; ++l) {
            const unsigned long long* st = h_stamps_.data() + (size_t) l * kStamps;
            const int kind = is_qsa_layer(g, l) ? 1 : 0;
            const unsigned long long before = l == 0 ? w[1] : (st - kStamps)[kLayerStamps - 1];
            prof_ns_[(size_t) kind * kStamps] += (double) (st[0] - before);
            for (int i = 1; i < kLayerStamps; ++i) prof_ns_[(size_t) kind * kStamps + i] += (double) (st[i] - st[i - 1]);
            // the shared expert's branch, from the doorbell (it forks there) to its end
            prof_ns_[(size_t) kind * kStamps + kLayerStamps] += (double) (st[kLayerStamps] - st[4]);
            {   // the wait for the CPU's rows, by bucket
                const double us = (double) (st[9] - st[8]) / 1e3;
                int b = 0;
                while (b < kWaitBuckets - 1 && us >= kWaitEdges[b]) ++b;
                ++prof_wait_hist_[kind][b];
            }
            ++prof_layers_[kind];
        }
        prof_window_ns_[0] += (double) (w[1] - w[0]);
        prof_window_ns_[1] += (double) (w[3] - w[2]);
        prof_window_ns_[2] += (double) (w[3] - w[0]);
        prof_window_ns_[3] += (double) (w[4] - w[3]);
        ++prof_windows_;
    }
    VDBG("window done\n");
    ++windows;
    return true;
}

void Verifier::print_profile() const {
    if (prof_windows_ == 0) return;
    static const char* const names[kStamps] = {"stamp gap",      "HC read, mixer",   "mixer",          "HC read, FFN",
                                               "router, ring, hits", "expert inputs", "VRAM experts",
                                               "wait for PCIe plan", "PCIe experts", "wait for CPU",   "rows + combine",
                                               "last HC write",  "shared expert, beside"};
    const double nw = (double) prof_windows_;
    std::printf("%-24s GPU time over %lld windows, ms per window: inputs %.3f, head %.3f, whole window %.3f; "
                "one stamp %.2f us (each stage below includes one; the last runs from the ring beside the "
                "five before \"rows + combine\")\n", "window profile", (long long) prof_windows_,
                prof_window_ns_[0] / nw / 1e6, prof_window_ns_[1] / nw / 1e6, prof_window_ns_[2] / nw / 1e6,
                prof_window_ns_[3] / nw / 1e3);
    std::printf("  %-22s %12s %12s %14s\n", "stage", "GDN us/layer", "QSA us/layer", "ms per window");
    for (int i = 0; i < kStamps; ++i) {
        const double a = prof_ns_[(size_t) i], b = prof_ns_[(size_t) kStamps + i];
        std::printf("  %-22s %12.1f %12.1f %14.3f\n", names[i],
                    prof_layers_[0] ? a / (double) prof_layers_[0] / 1e3 : 0.0,
                    prof_layers_[1] ? b / (double) prof_layers_[1] / 1e3 : 0.0, (a + b) / nw / 1e6);
    }
    std::printf("  %-22s", "wait for CPU, layers");
    for (int b = 0; b < kWaitBuckets; ++b) {
        const double f = 100.0 * (double) (prof_wait_hist_[0][b] + prof_wait_hist_[1][b]) /
                         (double) std::max<int64_t>(1, prof_layers_[0] + prof_layers_[1]);
        if (b < kWaitBuckets - 1) std::printf(" <%.0f us %.0f%%,", kWaitEdges[b], f);
        else std::printf(" more %.0f%%\n", f);
    }
}

void Verifier::ple_ahead(int t, int32_t token) {
    using namespace strata::kernels;
    SessionState& ss = *ss_;
    if (!ss.ple.ready() || t < 0 || t >= max_t_) return;
    ahead_tok_[t] = token;
    int32_t prev[2] = {t >= 2 ? ahead_tok_[t - 2] : ss.ple_prev[t], t >= 1 ? ahead_tok_[t - 1] : ss.ple_prev[1]};
    uint32_t rows[PLE_N_HEADS];
    ngram_rows(&token, prev, 1, ss.ple.consts, rows);
    ss.ple.table->ahead(t, rows);
}

void Verifier::set_plan_slot(int grp) {
    const int64_t cap = sink_.cap;
    int32_t* base = h_plan_ + (size_t) grp * (size_t) plan_i32_;
    sink_.counts = base;
    sink_.dst = base + 4 + cap + 1;
    sink_.tok = sink_.dst + cap;
    sink_.ptr2 = (unsigned long long*) (base + plan_ptr_off_) + cap;
    sink_.start2 = base + plan_ptr_off_ + 4 * cap;
    const int G = groups_[last_t_] > 0 ? groups_[last_t_] : 1;
    const int64_t per = G == 2 ? kStagingBlobs / 2 : kStagingBlobs;
    sink_.staging = (unsigned long long) (staging_ + (size_t) (grp * per) * strata::kernels::cpu::expert_layout().max_blob);
    sink_.staging_cap = per;
}

// Flag B only rises: a host function of an earlier layer may run after a later layer already raised it directly.
void Verifier::raise_flag(uint32_t* flag, uint32_t value) {
    volatile long* f = (volatile long*) flag;
#if defined(_WIN32)
    long cur = *f;
    while ((uint32_t) cur < value) {
        const long prev = _InterlockedCompareExchange(f, (long) value, cur);
        if (prev == cur) break;
        cur = prev;
    }
#else
    uint32_t cur = __atomic_load_n((uint32_t*) flag, __ATOMIC_SEQ_CST);
    while (cur < value && !__atomic_compare_exchange_n((uint32_t*) flag, &cur, value, false, __ATOMIC_SEQ_CST, __ATOMIC_SEQ_CST)) {}
#endif
}

// Plan v0.3 P6: the PCIe share by DMA.  The copy engine moves the blobs while the CPU computes its own share and the
// GPU its VRAM experts; a host function raises flag B when they have landed (the graph waits for it before the PCIe
// groups).  Staging is split between the two token groups of a split window.
void Verifier::fetch_dma(void* ctx, const uint8_t* const* src, int n, size_t bytes) {
    Verifier* v = (Verifier*) ctx;
    const uint32_t want = v->cur_layer_ + 1;
    if (n <= 0) { raise_flag(v->h_flagB_, want); return; }
    uint8_t* stage = (uint8_t*) v->sink_.staging;                  // this group's half in a split window
    for (int i = 0; i < n; ++i) cudaMemcpyAsync(stage + (size_t) i * bytes, src[i], bytes, cudaMemcpyHostToDevice, v->copy_);
    FlagSet& fs = v->flag_sets_[v->cur_layer_ % (sizeof v->flag_sets_ / sizeof v->flag_sets_[0])];
    fs.flag = v->h_flagB_;
    fs.value = want;
    cudaLaunchHostFunc(v->copy_, [](void* p) { FlagSet* s = (FlagSet*) p; raise_flag(s->flag, s->value); }, &fs);
}

void Verifier::publish_plan(void* ctx) {
    Verifier* v = (Verifier*) ctx;
    _mm_sfence();
    *(volatile uint32_t*) v->h_flagA_ = v->cur_layer_ + 1;
}

bool Verifier::commit(int n_keep, std::string& err, bool wait) {
    if (n_keep < 1 || n_keep > last_t_) { err = "verify: commit count out of range"; return false; }
    const Clock::time_point t0 = Clock::now();
    h_commit_[0] = n_keep;
    h_commit_[1] = n_keep - 1;
    for (int t = 0; t < max_t_; ++t) h_commit_[2 + t] = t < n_keep ? (int32_t) (last_pos0_ + t) : -1;
    std::atomic_thread_fence(std::memory_order_seq_cst);
    const cudaError_t le = cudaGraphLaunch(commit_exec_, cs_);
    if (le != cudaSuccess) { err = std::string("verify: commit launch: ") + cudaGetErrorString(le); return false; }
    (void) cudaStreamQuery(cs_);   // submit now, not at the next driver call
    commit_pending_ = true;
    for (int t = 0; t < n_keep; ++t) {
        ss_->ple_prev[0] = ss_->ple_prev[1];
        ss_->ple_prev[1] = last_tokens_[t];
    }
    ms_commit += ms_since(t0);
    return !wait || wait_commit(err);
}

bool Verifier::wait_commit(std::string& err) {
    if (!commit_pending_) return true;
    const Clock::time_point t0 = Clock::now();
    commit_pending_ = false;
    const cudaError_t se = cudaStreamSynchronize(cs_);
    ms_commit += ms_since(t0);
    if (se != cudaSuccess) { err = std::string("verify: commit: ") + cudaGetErrorString(se); return false; }
    return true;
}

}  // namespace strata::core
