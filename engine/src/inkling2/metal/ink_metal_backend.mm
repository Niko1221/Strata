// engine/src/inkling2/metal/ink_metal_backend.mm - InkBackend plugin for Apple Silicon Metal (strata-inkling)
// MIT License - Copyright (c) 2026 Strata contributors
#include "../ink_backend.h"
#include "../ink_vision.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <memory>
#include <numeric>
#include <string>
#include <vector>

#define INK_METAL_WITH_MODEL_FILE 1
#include "ink_metal_runner.h"

#ifdef INK_METAL_EMBEDDED_SHADERS
#include "ink_metal_embedded_shaders.h"
#endif

namespace ink_metal {
inline bool InkMetalRunner::compile_shaders(const std::string& kernel_dir, std::string& err) {
#ifdef INK_METAL_EMBEDDED_SHADERS
    (void)kernel_dir;
    std::string why;
    std::string codecs_src = std::string(kInkMetalIqGridsSrc) + "\n" + kInkMetalCodecsSrc;
    lib_codecs = cx->compile_try({}, codecs_src, false, &why);
    if (lib_codecs < 0) { err = "codecs MSL: " + why; return false; }
    lib_ops = cx->compile_try({}, kInkMetalOpsSrc, false, &why);
    if (lib_ops < 0) { err = "ops MSL: " + why; return false; }
    ops_src_text = kInkMetalOpsSrc;
    lib_dspark = cx->compile_try({}, kInkMetalDsparkSrc, false, &why);
    if (lib_dspark < 0) { err = "dspark MSL: " + why; return false; }
    lib_vis = cx->compile_try({}, kInkMetalVisionSrc, false, &why);
    if (lib_vis < 0) { err = "vision MSL: " + why; return false; }
    lib_dflash2 = cx->compile_try({}, kInkMetalDflash2Src, false, &why);
    if (lib_dflash2 < 0) fprintf(stderr, "[ink_metal] DFlash 2 optional compile skipped: %s\n", why.c_str());
    lib_mpp = -1;
#ifdef INK_METAL_HAS_MPP_SRC
    if (cx->supports_metal4()) {
        std::string mwhy;
        std::string mpp_src = std::string(kInkMetalIqGridsSrc) + "\n" + kInkMetalMppSrc;
        lib_mpp = cx->compile_try({}, mpp_src, false, &mwhy);
        if (lib_mpp < 0) fprintf(stderr, "[ink_metal] Metal 4 MPP optional compile skipped: %s\n", mwhy.c_str());
    }
#endif
    return true;
#else
    std::string why;
    lib_codecs = cx->compile_try({kernel_dir + "/ink_metal_iq_grids.metal", kernel_dir + "/ink_metal_codecs.metal"}, "", false, &why);
    if (lib_codecs < 0) { err = "codecs MSL: " + why; return false; }
    lib_ops = cx->compile_try({kernel_dir + "/ink_metal_ops.metal"}, "", false, &why);
    if (lib_ops < 0) { err = "ops MSL: " + why; return false; }
    {
        FILE* fo = std::fopen((kernel_dir + "/ink_metal_ops.metal").c_str(), "rb");
        if (fo) { char b[65536]; size_t n; while ((n = std::fread(b, 1, sizeof b, fo)) > 0) ops_src_text.append(b, n); std::fclose(fo); }
    }
    lib_dspark = cx->compile_try({kernel_dir + "/ink_metal_dspark.metal"}, "", false, &why);
    if (lib_dspark < 0) { err = "dspark MSL: " + why; return false; }
    lib_vis = cx->compile_try({kernel_dir + "/ink_metal_vision.metal"}, "", false, &why);
    if (lib_vis < 0) { err = "vision MSL: " + why; return false; }
    lib_dflash2 = cx->compile_try({kernel_dir + "/ink_metal_dflash2.metal"}, "", false, &why);
    if (lib_dflash2 < 0) fprintf(stderr, "[ink_metal] DFlash 2 optional compile skipped: %s\n", why.c_str());
    lib_mpp = -1;
    if (cx->supports_metal4()) {
        std::string mwhy;
        lib_mpp = cx->compile_try({kernel_dir + "/ink_metal_iq_grids.metal", kernel_dir + "/ink_metal_mpp.metal"}, "", false, &mwhy);
        if (lib_mpp < 0) fprintf(stderr, "[ink_metal] Metal 4 MPP optional compile skipped: %s\n", mwhy.c_str());
    }
    return true;
#endif
}
}  // namespace ink_metal

namespace ink2 {

static inline uint64_t m5_splitmix64(uint64_t& x) noexcept {
    uint64_t z = (x += 0x9e3779b97f4a7c15ull);
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ull;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebull;
    return z ^ (z >> 31);
}

static inline double m5_u01(uint64_t& state) noexcept {
    return (double)(m5_splitmix64(state) >> 11) * (1.0 / 9007199254740992.0);
}

// Compute normalized target probabilities on Top-K candidate row under temperature / top_k / top_p / min_p
struct M5SampleOpts {
    float temperature = 0.0f;
    int top_k = 64;
    float top_p = 1.0f;
    float min_p = 0.0f;
};

static void m5_cand_probs(const int32_t* ids, const float* vals, int K, const M5SampleOpts& s,
                          std::vector<int32_t>& out_ids, std::vector<float>& out_p) {
    out_ids.clear();
    out_p.clear();
    const int lim = (s.top_k > 0) ? std::min(K, s.top_k) : K;
    if (lim <= 0) return;
    const float inv_t = 1.0f / std::max(s.temperature, 1e-6f);
    float mx = -1e30f;
    for (int i = 0; i < lim; ++i) {
        if (ids[i] >= 0 && vals[i] > mx) mx = vals[i];
    }
    out_ids.reserve((size_t)lim);
    out_p.reserve((size_t)lim);
    double sum = 0.0;
    for (int i = 0; i < lim; ++i) {
        if (ids[i] < 0) continue;
        float p = std::exp((vals[i] - mx) * inv_t);
        out_ids.push_back(ids[i]);
        out_p.push_back(p);
        sum += (double)p;
    }
    if (sum <= 0.0 || out_p.empty()) return;
    const float inv_s = (float)(1.0 / sum);
    for (float& p : out_p) p *= inv_s;

    // min_p filter (candidates are already sorted descending!)
    if (s.min_p > 0.0f && !out_p.empty()) {
        const float thr = s.min_p * out_p[0];
        size_t keep = 1;
        while (keep < out_p.size() && out_p[keep] >= thr) ++keep;
        out_ids.resize(keep);
        out_p.resize(keep);
    }
    // top_p nucleus filter
    if (s.top_p > 0.0f && s.top_p < 1.0f && out_p.size() > 1) {
        double cum = 0.0;
        size_t keep = 0;
        for (size_t i = 0; i < out_p.size(); ++i) {
            cum += (double)out_p[i];
            keep = i + 1;
            if (cum >= (double)s.top_p) break;
        }
        out_ids.resize(keep);
        out_p.resize(keep);
    }
    double rsum = 0.0;
    for (float p : out_p) rsum += (double)p;
    if (rsum > 0.0) {
        const float inv = (float)(1.0 / rsum);
        for (float& p : out_p) p *= inv;
    }
}

class MetalInkBackend final : public InkBackend {
public:
    const char* backend_name() const noexcept override { return "metal"; }

    int effective_max_ctx() const override { return runner_.max_ctx; }

    int effective_chunk() const override {
        return runner_.m5.active ? std::min(runner_.m5.prefill_chunk, runner_.max_batch) : 0;   // 0 = the --chunk value (default 512), as before the M5 Max work
    }

    bool supports_candidates() const noexcept override {
        return runner_.m5.active && runner_.m5.use_candidates;
    }

    bool prefers_candidate_rows() const noexcept override {
        return supports_candidates();
    }

    bool prefers_async_prompt_ckpt() const noexcept override {
        return runner_.m5.active && runner_.m5.use_post_step_ckpt;
    }

    int preferred_prefill_cap_with_decoders() const noexcept override {
        return runner_.m5.active ? runner_.m5.prefill_cap_with_decoders : 0;
    }

    bool supports_sampled_spec() const noexcept override {
        return runner_.m5.active && runner_.use_dflash2;
    }

    bool dspark_ready() const override { return runner_.use_dflash2 || runner_.use_dspark; }

    // The one drafter rule, principle 2 (strata/core/ink_draft_rule.hpp): on a Mac that pages its model a draft step reads more pages, so from 2048 tokens on the step is plain.
    bool draft_plain_now(int ctx_pos) const override {
        if (!runner_.use_dflash2) return false;
        if (ctx_pos >= ink_metal::kMetalDflash2CtxCut) return true;
        return strata::core::ink_draft_plain_step(runner_.draft_hybrid, ctx_pos, runner_.draft_user_on);
    }

    // A long prompt never feeds the drafter (its steps are plain from kMetalDflash2CtxCut tokens on, and the prompt-speed cost of the ingest is skipped).
    void hint_prompt_len(int slot, int prompt_len) override {
        if (slot >= 0 && slot < (int)runner_.df2_slot_off.size()) runner_.df2_slot_off[(size_t)slot] = (prompt_len >= ink_metal::kMetalDflash2CtxCut) ? 1 : 0;
    }

    std::string device_info() const override {
        return cx_.device_name() + " (workSet=" +
               std::to_string((int)(cx_.recommended_working_set() / (1024ull * 1024ull * 1024ull))) + " GiB)";
    }

    BackendStats stats() const override {
        BackendStats s;
        s.total_experts = (size_t)mod_.man.c.n_phys * (size_t)mod_.man.c.n_experts;
        s.vram_experts_loaded = s.total_experts;  // Unified zero-copy Apple Silicon memory!
        s.ram_experts_loaded = 0;
        s.vram_used_bytes = cx_.allocated();
        s.vram_total_bytes = cx_.recommended_working_set();
        s.active_slots = active_slots_;
        s.max_slots = runner_.max_slots;
        s.kv_dtype = runner_.kv_is_fp8 ? KvDtype::FP8_E4M3 : kv_dtype_;
        return s;
    }

    bool init(const InkModelFile& model, const BackendOpts& opts, std::string& err) override {
        if (kv_dtype_is_int_quant(opts.kv_dtype)) {
            err = "metal: --kv k8v4/k8v8 is not supported on the Metal backend (allocates fp16-sized pages); use --kv fp8 or fp16";
            return false;
        }
        kv_dtype_ = opts.kv_dtype;
        strata::ink::df2_model_path() = model.path;
        strata::ink::df2_parallel() = std::max(1, std::min(64, opts.parallel));

        const char* force_mf = std::getenv("INK_METAL_FORCE_MODEL_FILE");
        if (model.kind == ModelSourceKind::Strata && !(force_mf && force_mf[0] == '1')) {
            if (!mod_.load(cx_, model.path, err)) return false;
        } else {
            if (!mod_.load_from_model_file(cx_, model, err)) return false;
        }
        const int par = std::max(1, std::min(64, opts.parallel));
        // Probe M5 Max config early to size batch_cap for chunk=8192 when STRATA_INK_M5MAX=1
        const auto m5_pre = ink_metal::M5MaxCfg::detect(cx_);
        int want_chunk = opts.chunk;
        if (m5_pre.active) {
            want_chunk = std::max(want_chunk, m5_pre.prefill_chunk);
            if (par >= 8) want_chunk = std::min(want_chunk, 4096);  // Keep C16 RSS comfortably <= 40 GiB!
            if (cx_.recommended_working_set() < (48ull << 30)) want_chunk = std::min(want_chunk, 2048);
        }
        const int batch_cap = std::max(256, std::max(want_chunk, par));
        // DFlash 2 on Metal (kMetalDflash2Available): loaded only where it is validated, see the gate in ink_metal_runner.h (16 KiB-aligned SQ-20 pack, Apple M5 Max, not in the exact mode).
        // With the switch false the drafter is never loaded and plain decode is the tip's.
        const bool enable_draft = opts.enable_dspark &&
                                  ((ink_metal::kMetalDflash2Available && strata::ink::df2_requested()) || ink2::dormant_dspark_allowed());
        {   // asked for by name: --draft on, or STRATA_INK_DRAFTER=dflash2
            const char* de = std::getenv("STRATA_INK_DRAFTER");
            const std::string dl = de ? std::string(de) : std::string();
            runner_.draft_user_on = opts.draft_forced || dl == "dflash2" || dl == "DFLASH2" || dl == "df2";
            if (runner_.draft_user_on && !ink_metal::kMetalDflash2Available)
                fprintf(stderr, "[ink_metal] DFlash 2 is not available on Metal in this release; decoding without a drafter.\n");
        }
        if (!runner_.init(cx_, mod_, par, std::max(256, opts.max_ctx), batch_cap,
                          enable_draft, opts.vision_path, "", err)) {
            return false;
        }
        snaps_.resize((size_t)par);
        frnn_snaps_.resize((size_t)par);
        ckpts_.assign((size_t)par, PromptCkpt{});
        slot_used_.assign((size_t)par, 0);
        slot_beam_depth_.assign((size_t)par, runner_.is_apple10 ? 5 : 4);
        slot_ema_gain_.assign((size_t)par, 1.15);
        slot_cooldown_.assign((size_t)par, 0);
        slot_cd_len_.assign((size_t)par, 8);
        return true;
    }

    bool alloc_slot(int slot, int /*max_ctx*/, KvDtype /*kv_dtype*/, std::string& err) override {
        if (slot < 0 || slot >= runner_.max_slots) {
            err = "metal slot out of range";
            return false;
        }
        if (!slot_used_[slot]) { slot_used_[slot] = 1; ++active_slots_; }
        runner_.reset_slot(slot);
        drop_ckpt(slot);
        slot_beam_depth_[slot] = runner_.is_apple10 ? 5 : 4;
        slot_ema_gain_[slot] = 1.15;
        slot_cooldown_[slot] = 0;
        slot_cd_len_[slot] = 8;
        if (last_ver_slot_ == slot) last_ver_slot_ = -1;
        if (last_tree_slot_ == slot) last_tree_slot_ = -1;
        return true;
    }

    void free_slot(int slot) override {
        if (slot >= 0 && slot < runner_.max_slots && slot_used_[slot]) {
            slot_used_[slot] = 0;
            --active_slots_;
            runner_.reset_slot(slot);
            drop_ckpt(slot);
            if (last_ver_slot_ == slot) last_ver_slot_ = -1;
            if (last_tree_slot_ == slot) last_tree_slot_ = -1;
        }
    }

    void reset_slot(int slot) override {
        runner_.reset_slot(slot);
        drop_ckpt(slot);
        if (slot >= 0 && slot < (int)slot_beam_depth_.size()) {
            slot_beam_depth_[slot] = runner_.is_apple10 ? 5 : 4;
            slot_ema_gain_[slot] = 1.15;
            slot_cooldown_[slot] = 0;
            slot_cd_len_[slot] = 8;
        }
        if (last_ver_slot_ == slot) last_ver_slot_ = -1;
        if (last_tree_slot_ == slot) last_tree_slot_ = -1;
    }

    void snapshot_slot(int slot) override {
        if (slot >= 0 && slot < (int)snaps_.size()) {
            runner_.snapshot_slot(slot, snaps_[slot]);
            if (runner_.use_dspark) {
                const size_t D = (size_t)mod_.man.c.hidden;
                frnn_snaps_[slot].resize(D);
                std::memcpy(frnn_snaps_[slot].data(), runner_.dsp_frnn_slot.as<float>() + (size_t)slot * D, D * 4);
            }
        }
    }

    void rollback_slot(int slot, int keep_len) override {
        if (slot < 0 || slot >= (int)snaps_.size()) return;
        if (runner_.use_dflash2) runner_.df2.rollback_slot(slot, keep_len);
        // O(1) ZERO-REPLAY partial-accept rollback via ShortConv verify ring (linear verify!)
        if (slot == last_ver_slot_ && keep_len > last_ver_pos0_ && keep_len <= last_ver_pos0_ + last_ver_m_) {
            runner_.restore_verify_step(slot, keep_len - last_ver_pos0_ - 1);
            return;
        }
        // O(1) ZERO-REPLAY early-stop rollback for multi-branch tree commit!
        if (slot == last_tree_slot_ && keep_len >= last_tree_anchor_ &&
            keep_len < last_tree_anchor_ + (int)last_tree_path_.size()) {
            runner_.restore_slot(slot, snaps_[slot]);
            const int kept = keep_len - last_tree_anchor_;
            if (kept > 0) {
                std::vector<int32_t> sub(last_tree_path_.begin(), last_tree_path_.begin() + kept);
                runner_.commit_tree_path(slot, last_tree_anchor_, sub);
            } else if (runner_.use_dspark && !frnn_snaps_[slot].empty()) {
                const size_t D = (size_t)mod_.man.c.hidden;
                std::memcpy(runner_.dsp_frnn_slot.as<float>() + (size_t)slot * D, frnn_snaps_[slot].data(), D * 4);
            }
            return;
        }
        runner_.restore_slot(slot, snaps_[slot]);
        if (runner_.use_dspark && !frnn_snaps_[slot].empty()) {
            const size_t D = (size_t)mod_.man.c.hidden;
            std::memcpy(runner_.dsp_frnn_slot.as<float>() + (size_t)slot * D, frnn_snaps_[slot].data(), D * 4);
        }
    }

    bool save_prompt_checkpoint(int slot, int seq_len) override {
        if (slot < 0 || slot >= (int)ckpts_.size() || seq_len <= 0 || seq_len > runner_.max_ctx) return false;
        const auto& c = mod_.man.c;
        const size_t row = (size_t)c.n_kv * (size_t)c.head_dim;
        const size_t eb = runner_.kv_is_fp8 ? 1ul : 2ul;
        PromptCkpt& ck = ckpts_[slot];
        ck.valid = false;
        runner_.snapshot_slot(slot, ck.sconv);
        ck.swa_k.resize((size_t)c.n_visits);
        ck.swa_v.resize((size_t)c.n_visits);
        for (int v = 0; v < c.n_visits; ++v) {
            if (!c.is_swa[c.phys_for_visit(v)]) { ck.swa_k[v].clear(); ck.swa_v[v].clear(); continue; }
            const size_t stride = (size_t)runner_.visit_kv_stride[v];
            const size_t nb = std::min<size_t>((size_t)seq_len, stride) * row * eb;
            ck.swa_k[v].resize(nb);
            ck.swa_v[v].resize(nb);
            std::memcpy(ck.swa_k[v].data(), runner_.kv_k[v].as<uint8_t>() + (size_t)slot * stride * row * eb, nb);
            std::memcpy(ck.swa_v[v].data(), runner_.kv_v[v].as<uint8_t>() + (size_t)slot * stride * row * eb, nb);
        }
        for (int l = 0; l < 3; ++l) { ck.dsp_k[l].clear(); ck.dsp_v[l].clear(); }
        ck.frnn.clear();
        if (runner_.use_dflash2) {
            runner_.df2.save_slot_ring(slot, ck.df2_ring);
        }
        if (runner_.use_dspark) {
            const size_t D = (size_t)c.hidden;
            ck.frnn.assign(runner_.dsp_frnn_slot.as<float>() + (size_t)slot * D,
                           runner_.dsp_frnn_slot.as<float>() + (size_t)(slot + 1) * D);
            for (int l = 0; l < 3; ++l) {
                if (!c.dspark_is_swa[l]) continue;
                const size_t stride = (size_t)runner_.dsp_kv_stride[l];
                const size_t n = std::min<size_t>((size_t)seq_len, stride) * row;
                ck.dsp_k[l].resize(n);
                ck.dsp_v[l].resize(n);
                std::memcpy(ck.dsp_k[l].data(), runner_.dsp_ctx_k[l].as<uint16_t>() + (size_t)slot * stride * row, n * 2);
                std::memcpy(ck.dsp_v[l].data(), runner_.dsp_ctx_v[l].as<uint16_t>() + (size_t)slot * stride * row, n * 2);
            }
        }
        ck.seq_len = seq_len;
        ck.valid = true;
        return true;
    }

    bool restore_prompt_checkpoint(int slot, int seq_len) override {
        if (slot < 0 || slot >= (int)ckpts_.size()) return false;
        PromptCkpt& ck = ckpts_[slot];
        if (!ck.valid || ck.seq_len != seq_len) return false;
        const auto& c = mod_.man.c;
        const size_t row = (size_t)c.n_kv * (size_t)c.head_dim;
        const size_t eb = runner_.kv_is_fp8 ? 1ul : 2ul;
        runner_.restore_slot(slot, ck.sconv);
        for (int v = 0; v < c.n_visits; ++v) {
            if (ck.swa_k[v].empty()) continue;
            const size_t stride = (size_t)runner_.visit_kv_stride[v];
            std::memcpy(runner_.kv_k[v].as<uint8_t>() + (size_t)slot * stride * row * eb, ck.swa_k[v].data(), ck.swa_k[v].size());
            std::memcpy(runner_.kv_v[v].as<uint8_t>() + (size_t)slot * stride * row * eb, ck.swa_v[v].data(), ck.swa_v[v].size());
        }
        if (runner_.use_dflash2) {
            runner_.df2.restore_slot_ring(slot, ck.df2_ring);
        }
        if (runner_.use_dspark && !ck.frnn.empty()) {
            const size_t D = (size_t)c.hidden;
            std::memcpy(runner_.dsp_frnn_slot.as<float>() + (size_t)slot * D, ck.frnn.data(), D * 4);
            for (int l = 0; l < 3; ++l) {
                if (ck.dsp_k[l].empty()) continue;
                const size_t stride = (size_t)runner_.dsp_kv_stride[l];
                std::memcpy(runner_.dsp_ctx_k[l].as<uint16_t>() + (size_t)slot * stride * row, ck.dsp_k[l].data(), ck.dsp_k[l].size() * 2);
                std::memcpy(runner_.dsp_ctx_v[l].as<uint16_t>() + (size_t)slot * stride * row, ck.dsp_v[l].data(), ck.dsp_v[l].size() * 2);
            }
        }
        snapshot_slot(slot);
        if (last_ver_slot_ == slot) last_ver_slot_ = -1;
        if (last_tree_slot_ == slot) last_tree_slot_ = -1;
        slot_cooldown_[slot] = 0;
        return true;
    }

    bool forward_batch(const StepBatch& batch, float* logits_out, TapCapture* taps_out, std::string& err) override {
        if (batch.n_tokens <= 0) return true;
        if (batch.n_tokens > runner_.max_batch) {
            const int V = mod_.man.c.vocab_unpadded;
            int out_row = 0;
            for (int off = 0; off < batch.n_tokens; ) {
                // Split cleanly at slot boundaries when a multi-slot prefill batch exceeds max_batch
                // so each single-slot run stays contiguous and hits Metal 4 NAX FlashAttention-2!
                int take = std::min(runner_.max_batch, batch.n_tokens - off);
                if (runner_.m5.active && take >= 64) {
                    const int s0 = batch.slot_ids[off];
                    int run_end = off + 1;
                    while (run_end < off + take && batch.slot_ids[run_end] == s0 &&
                           batch.pos[run_end] == batch.pos[run_end - 1] + 1) {
                        ++run_end;
                    }
                    if (run_end - off >= 64 && run_end < off + take) {
                        take = run_end - off;
                    }
                }
                StepBatch sub;
                sub.capture_taps = batch.capture_taps;
                sub.img_embeds = batch.img_embeds;
                int sub_want = 0;
                for (int i = 0; i < take; ++i) {
                    const int k = off + i;
                    const bool wl = batch.want_logits[k] != 0;
                    if (wl) ++sub_want;
                    sub.push(batch.slot_ids[k], batch.pos[k], batch.token_ids[k], wl, batch.img_embed_idx[k]);
                }
                float* sub_lg = (logits_out && sub_want > 0) ? (logits_out + (size_t)out_row * V) : nullptr;
                if (!forward_batch(sub, sub_lg, taps_out, err)) return false;
                out_row += sub_want;
                off += take;
            }
            return true;
        }
        ink_metal::InkBatchStep st;
        build_step(batch, st);
        if (taps_out) st.capture_taps = true;
        const int n_want = batch.count_logits();
        st.compute_logits = (n_want > 0 && logits_out != nullptr);
        if (st.compute_logits) {
            if (n_want == 1) {
                for (int m = 0; m < st.M; ++m) {
                    if (batch.want_logits[m]) { st.logits_last_only_idx = m; break; }
                }
                if (!runner_.forward_batch(st, logits_out, nullptr, err)) return false;
            } else if (st.M > 64) {
                std::vector<int> want_rows;
                want_rows.reserve((size_t)n_want);
                for (int m = 0; m < st.M; ++m) if (batch.want_logits[m]) want_rows.push_back(m);
                if (n_want <= 64) {
                    st.logits_last_only_idx = -1;
                    st.logits_rows = std::move(want_rows);
                    if (!runner_.forward_batch(st, logits_out, nullptr, err)) return false;
                } else {
                    st.compute_logits = false;
                    if (!runner_.forward_batch(st, nullptr, nullptr, err)) return false;
                    runner_.unembed_selected_rows(want_rows, logits_out);
                }
            } else {
                const int V = mod_.man.c.vocab_unpadded;
                std::vector<float> all_lg((size_t)st.M * V);
                st.logits_last_only_idx = -1;
                if (!runner_.forward_batch(st, all_lg.data(), nullptr, err)) return false;
                int dst_row = 0;
                for (int m = 0; m < st.M; ++m) {
                    if (batch.want_logits[m]) {
                        std::memcpy(logits_out + (size_t)dst_row * V,
                                    all_lg.data() + (size_t)m * V,
                                    (size_t)V * sizeof(float));
                        ++dst_row;
                    }
                }
            }
        } else {
            if (!runner_.forward_batch(st, nullptr, nullptr, err)) return false;
        }
        if (taps_out) {
            const size_t D = (size_t)mod_.man.c.hidden;
            const size_t row_f = (size_t)st.M * D;
            const float* tp = runner_.b_taps4.as<float>();
            for (int t = 0; t < 4; ++t) {
                taps_out->taps[t].assign(tp + (size_t)t * row_f, tp + (size_t)(t + 1) * row_f);
            }
        }
        return true;
    }

    bool forward_batch_cand(const StepBatch& batch, CandRows& out, std::string& err) override {
        if (!supports_candidates()) return InkBackend::forward_batch_cand(batch, out, err);
        if (batch.n_tokens <= 0) { out.k = 128; out.rows = 0; out.idx.clear(); out.val.clear(); out.finite.clear(); return true; }
        if (batch.n_tokens > runner_.max_batch) {
            out.k = 128; out.rows = 0; out.idx.clear(); out.val.clear(); out.finite.clear();
            for (int off = 0; off < batch.n_tokens; off += runner_.max_batch) {
                const int take = std::min(runner_.max_batch, batch.n_tokens - off);
                StepBatch sub;
                sub.capture_taps = batch.capture_taps;
                sub.img_embeds = batch.img_embeds;
                for (int i = 0; i < take; ++i) {
                    const int k = off + i;
                    sub.push(batch.slot_ids[k], batch.pos[k], batch.token_ids[k], batch.want_logits[k] != 0, batch.img_embed_idx[k]);
                }
                CandRows sub_out;
                if (!forward_batch_cand(sub, sub_out, err)) return false;
                if (sub_out.rows > 0) {
                    out.rows += sub_out.rows;
                    out.idx.insert(out.idx.end(), sub_out.idx.begin(), sub_out.idx.end());
                    out.val.insert(out.val.end(), sub_out.val.begin(), sub_out.val.end());
                    out.finite.insert(out.finite.end(), sub_out.finite.begin(), sub_out.finite.end());
                }
            }
            return true;
        }
        ink_metal::InkBatchStep st;
        build_step(batch, st);
        const int n_want = batch.count_logits();
        st.compute_logits = (n_want > 0);
        if (!st.compute_logits) {
            out.k = 128; out.rows = 0; out.idx.clear(); out.val.clear(); out.finite.clear();
            return runner_.forward_batch(st, nullptr, nullptr, err);
        }
        if (n_want > 64) { err = "metal: more than 64 candidate rows in one batch"; return false; }
        if (n_want == 1) {
            for (int m = 0; m < st.M; ++m) if (batch.want_logits[m]) { st.logits_last_only_idx = m; break; }
        } else {
            st.logits_last_only_idx = -1;
            for (int m = 0; m < st.M; ++m) if (batch.want_logits[m]) st.logits_rows.push_back(m);
        }
        ink_metal::InkMetalRunner::Top128Out top;
        top.K = 128;
        if (!runner_.forward_batch(st, nullptr, nullptr, err, &top)) return false;
        out.k = top.K;
        out.rows = top.n_rows;
        out.idx = std::move(top.ids);
        out.val = std::move(top.vals);
        out.finite = std::move(top.finite);
        return true;
    }

    int dspark_draft(int slot, int32_t anchor_tok, int anchor_pos, int max_draft,
                     int32_t* out_draft_ids, std::string& /*err*/) override {
        if (max_draft <= 0) return 0;
        if (runner_.use_dflash2) {
            int32_t tmp[7]{};
            float conf[7]{};
            int n = runner_.df2.draft_block8(slot, anchor_tok, anchor_pos, tmp, conf);
            const int take = std::min(n, max_draft);
            for (int i = 0; i < take; ++i) out_draft_ids[i] = tmp[i];
            return take;
        }
        if (!runner_.use_dspark) return 0;
        const int req = std::min(7, max_draft);
        int32_t tmp[7]{};
        int n = runner_.dspark_draft_block8(slot, anchor_tok, anchor_pos, tmp, req);
        for (int i = 0; i < n; ++i) out_draft_ids[i] = tmp[i];
        return n;
    }

    int dspark_spec_step_greedy(
        int slot,
        int32_t anchor_tok,
        int anchor_pos,
        int max_emit,
        int32_t* out_emitted,
        int* out_drafted,
        std::string& err) override {
        if (runner_.use_dflash2) {
            return dflash2_spec_step_greedy(slot, anchor_tok, anchor_pos, max_emit, out_emitted, out_drafted, err);
        }
        if (!runner_.use_dspark || !runner_.use_dspark_meff || slot < 0 || slot >= runner_.max_slots ||
            max_emit < 2 || anchor_pos <= 0 || anchor_pos + 12 >= runner_.max_ctx) {
            return 0;
        }

        if (slot_cooldown_[slot] > 0) {
            --slot_cooldown_[slot];
            ink_metal::InkBatchStep st;
            st.M = 1;
            st.slot_ids = {slot};
            st.pos = {anchor_pos};
            st.tok_ids = {anchor_tok};
            st.capture_taps = true;
            st.compute_logits = true;
            int32_t tok = 0;
            if (!runner_.forward_batch(st, nullptr, &tok, err)) return -1;
            out_emitted[0] = tok;
            if (out_drafted) *out_drafted = 0;
            last_ver_slot_ = -1;
            last_tree_slot_ = -1;
            return 1;
        }

        const double t_step0 = mtl::now_s();
        const bool is_bf16_trunk = (mod_.man.get("model.llm.layers.0.attn.wq_du.weight").fmt == ink_metal::FMT_BF16);
        const bool is_sq10 = (mod_.man.build.find("10") != std::string::npos);
        const double t_plain_ms = runner_.is_apple10 ? (is_bf16_trunk ? 17.2 : (is_sq10 ? 13.5 : 14.8)) : 41.0;

        auto update_gain = [&](int n_emit) {
            const double step_ms = std::max(1.0, (mtl::now_s() - t_step0) * 1e3);
            const double step_gain = (double)n_emit * t_plain_ms / step_ms;
            slot_ema_gain_[slot] = 0.72 * slot_ema_gain_[slot] + 0.28 * step_gain;
            if (step_gain >= 1.05) {
                slot_ema_gain_[slot] = std::max(slot_ema_gain_[slot], 1.14);
                slot_cd_len_[slot] = 8;
            } else if (slot_ema_gain_[slot] < 0.90) {
                slot_cooldown_[slot] = slot_cd_len_[slot];
                slot_cd_len_[slot] = std::min(16, slot_cd_len_[slot] + 4);
                slot_ema_gain_[slot] = 0.96;
            }
        };

        ink_metal::InkDraftTree tree;
        const int max_b = (max_emit <= 4) ? std::min(10, max_emit + 2) : 10;
        const int dcap = std::max(4, std::min(slot_beam_depth_[slot], std::min(7, max_emit - 1)));
        const int tm = runner_.dspark_draft_beam_tree(slot, anchor_tok, anchor_pos, max_b, true, tree, dcap);
        if (tm <= 1) return 0;

        if (tree.is_linear) {
            ink_metal::InkBatchStep st;
            st.M = tm;
            st.slot_ids.assign((size_t)tm, slot);
            st.pos.resize((size_t)tm);
            for (int i = 0; i < tm; ++i) st.pos[i] = anchor_pos + i;
            st.tok_ids = tree.tok_ids;
            st.capture_taps = true;
            st.compute_logits = true;
            st.record_verify_ring = true;
            int32_t ver_ids[32]{};
            if (!runner_.forward_batch(st, nullptr, ver_ids, err)) return -1;

            const int rem = std::min(tm - 1, max_emit - 1);
            int acc = 0;
            for (int i = 0; i < rem; ++i) {
                if (ver_ids[i] == tree.tok_ids[i + 1]) out_emitted[acc++] = ver_ids[i];
                else break;
            }
            out_emitted[acc] = ver_ids[acc];
            if (acc < tm - 1) runner_.restore_verify_step(slot, acc);

            last_ver_slot_ = slot;
            last_ver_pos0_ = anchor_pos;
            last_ver_m_ = tm;
            last_tree_slot_ = -1;
            slot_beam_depth_[slot] = std::max(4, std::min(7, acc + 3));
            if (out_drafted) *out_drafted = rem;
            update_gain(1 + acc);
            return 1 + acc;
        }

        snapshot_slot(slot);
        ink_metal::InkBatchStep st;
        st.M = tm;
        st.slot_ids.assign((size_t)tm, slot);
        st.pos.resize((size_t)tm);
        int max_d = 1;
        for (int i = 0; i < tm; ++i) {
            st.pos[i] = anchor_pos + tree.depths[i];
            if (tree.depths[i] > max_d) max_d = tree.depths[i];
        }
        st.tok_ids = tree.tok_ids;
        st.capture_taps = true;
        st.compute_logits = true;
        st.is_tree = true;
        st.tree_anchor_pos = anchor_pos;
        st.tree_parent = tree.parents;
        st.tree_anc_mask = tree.anc_masks;
        int32_t ver_ids[32]{};
        if (!runner_.forward_batch(st, nullptr, ver_ids, err)) return -1;

        std::vector<int32_t> path;
        path.reserve(8);
        path.push_back(0);
        int cur_node = 0;
        int acc = 0;
        while (acc + 1 < max_emit) {
            const int32_t tn = ver_ids[cur_node];
            int mc = -1;
            for (int ci = cur_node + 1; ci < tm; ++ci) {
                if (tree.parents[ci] == cur_node && tree.tok_ids[ci] == tn) { mc = ci; break; }
            }
            if (mc < 0) break;
            out_emitted[acc++] = tn;
            path.push_back(mc);
            cur_node = mc;
        }
        out_emitted[acc] = ver_ids[cur_node];
        runner_.commit_tree_path(slot, anchor_pos, path);

        last_tree_slot_ = slot;
        last_tree_anchor_ = anchor_pos;
        last_tree_path_ = std::move(path);
        last_ver_slot_ = -1;
        slot_beam_depth_[slot] = std::max(4, std::min(7, acc + 3));
        if (out_drafted) *out_drafted = max_d;
        update_gain(1 + acc);
        return 1 + acc;
    }

    int dspark_spec_step_sampled(
        int slot,
        int32_t anchor_tok,
        int anchor_pos,
        int max_emit,
        float temperature,
        int top_k,
        float top_p,
        float min_p,
        uint64_t rng_seed,
        uint64_t rng_offset,
        int32_t* out_emitted,
        int* out_drafted,
        std::string& err) override {
        // The shared hook of the integration tip passes the sampling knobs one by one and a (seed, offset) pair; M5SampleOpts / step_seed rebuild what the sampler below uses.
        M5SampleOpts sopts;
        sopts.temperature = temperature; sopts.top_k = top_k; sopts.top_p = top_p; sopts.min_p = min_p;
        uint64_t step_seed = rng_seed ^ (rng_offset * 0x9e3779b97f4a7c15ull);
        if (!supports_sampled_spec() || slot < 0 || slot >= runner_.max_slots ||
            max_emit < 2 || anchor_pos <= 0 || anchor_pos + 12 >= runner_.max_ctx) {
            return 0;
        }

        // Draft with DFlash 2 only (DFlash 2 is the one drafter; the n-gram suffix fallback of the owner's first version is not used)
        int32_t draft_ids[7]{};
        float draft_conf[7]{};
        int n_avail = runner_.df2.draft_block8(slot, anchor_tok, anchor_pos, draft_ids, draft_conf);
        int n_draft = choose_df2_draft_len(draft_conf, std::min(n_avail, max_emit - 1));

        const int tm = 1 + n_draft;
        ink_metal::InkBatchStep st;
        st.M = tm;
        st.slot_ids.assign((size_t)tm, slot);
        st.pos.resize((size_t)tm);
        st.tok_ids.resize((size_t)tm);
        st.tok_ids[0] = anchor_tok;
        st.pos[0] = anchor_pos;
        for (int i = 0; i < n_draft; ++i) {
            st.tok_ids[i + 1] = draft_ids[i];
            st.pos[i + 1] = anchor_pos + 1 + i;
        }
        st.capture_taps = true;
        st.compute_logits = true;
        st.record_verify_ring = (tm > 1);
        ink_metal::InkMetalRunner::Top128Out top;
        top.K = 128;
        if (!runner_.forward_batch(st, nullptr, nullptr, err, &top)) return -1;

        uint64_t rng = step_seed ? step_seed : 0x9e3779b97f4a7c15ull;
        std::vector<int32_t> c_ids;
        std::vector<float> c_p;
        int acc = 0;
        for (int i = 0; i < n_draft; ++i) {
            m5_cand_probs(top.ids.data() + (size_t)i * top.K, top.vals.data() + (size_t)i * top.K, top.K, sopts, c_ids, c_p);
            const int32_t dt = draft_ids[i];
            float p_t = 0.0f;
            int dt_pos = -1;
            for (size_t j = 0; j < c_ids.size(); ++j) {
                if (c_ids[j] == dt) { p_t = c_p[j]; dt_pos = (int)j; break; }
            }
            // Exact rejection sampling against Dirac-1 draft proposal q(dt) = 1:
            // Accept dt with probability p_t; on reject sample from residual p_target \ {dt} (exact target distribution!)
            const double u = m5_u01(rng);
            if (u < (double)p_t) {
                out_emitted[acc++] = dt;
            } else {
                double rem_sum = 0.0;
                for (size_t j = 0; j < c_p.size(); ++j) {
                    if ((int)j != dt_pos) rem_sum += (double)c_p[j];
                }
                int32_t chosen = c_ids.empty() ? dt : c_ids[0];
                if (rem_sum > 1e-12) {
                    double r = m5_u01(rng) * rem_sum;
                    double cum = 0.0;
                    for (size_t j = 0; j < c_p.size(); ++j) {
                        if ((int)j == dt_pos) continue;
                        cum += (double)c_p[j];
                        chosen = c_ids[j];
                        if (r <= cum) break;
                    }
                }
                out_emitted[acc] = chosen;
                if (tm > 1) runner_.restore_verify_step(slot, acc);
                runner_.df2.rollback_slot(slot, anchor_pos + acc + 1);
                last_ver_slot_ = (tm > 1) ? slot : -1;
                last_ver_pos0_ = anchor_pos;
                last_ver_m_ = tm;
                last_tree_slot_ = -1;
                if (out_drafted) *out_drafted = n_draft;
                return 1 + acc;
            }
        }

        // All n_draft accepted -> sample bonus token from row n_draft
        m5_cand_probs(top.ids.data() + (size_t)n_draft * top.K, top.vals.data() + (size_t)n_draft * top.K, top.K, sopts, c_ids, c_p);
        int32_t bonus = c_ids.empty() ? 0 : c_ids[0];
        if (!c_p.empty()) {
            double r = m5_u01(rng);
            double cum = 0.0;
            for (size_t j = 0; j < c_p.size(); ++j) {
                cum += (double)c_p[j];
                bonus = c_ids[j];
                if (r <= cum) break;
            }
        }
        out_emitted[acc] = bonus;
        last_ver_slot_ = (tm > 1) ? slot : -1;
        last_ver_pos0_ = anchor_pos;
        last_ver_m_ = tm;
        last_tree_slot_ = -1;
        if (out_drafted) *out_drafted = n_draft;
        return 1 + acc;
    }

    bool encode_image(const ImageRec& img, std::vector<float>& out_embeds, std::string& err) override {
        if (img.file.empty()) return false;
        if (!runner_.ensure_vision(err)) return false;
        int ntok = 0;
        return runner_.vis.encode_image_file(cx_, img.file, out_embeds, ntok, err);
    }

    bool selfcheck(std::string& err) override {
        if (runner_.mod == nullptr) {
            return selfcheck_device(err);
        }
        StepBatch b;
        b.push(0, 0, 1, true);
        std::vector<float> lg((size_t)mod_.man.c.vocab_unpadded);
        runner_.reset_slot(0);
        if (!forward_batch(b, lg.data(), nullptr, err)) return false;
        for (float v : lg) {
            if (!std::isfinite(v)) { err = "metal selfcheck produced non-finite logits"; return false; }
        }
        runner_.reset_slot(0);
        return true;
    }

    bool selfcheck_device(std::string& err) {
        runner_.cx = &cx_;
        if (runner_.lib_codecs < 0) {
            if (!runner_.compile_shaders("", err)) return false;
        }
        auto p_f32 = cx_.pipeline(runner_.lib_codecs, "ink_linear_f32");
        (void)cx_.pipeline(runner_.lib_codecs, "ink_moe_gu_situ");
        (void)cx_.pipeline(runner_.lib_codecs, "ink_moe_gu_sk23");
        (void)cx_.pipeline(runner_.lib_codecs, "ink_moe_gu_sk3");
        (void)cx_.pipeline(runner_.lib_codecs, "ink_moe_gu_sk5");
        auto p_gdot = cx_.pipeline(runner_.lib_codecs, "ink_gguf_row_dot_test");

        // 1. Verify non-matching card refusal guard works cleanly
        const auto ref_chk = ink_metal::M5MaxCfg::probe("Apple M3 Pro", false, false, 18, 36ull << 30, true, false, true);
        if (!ref_chk.refused || ref_chk.active) {
            err = "metal selfcheck: M5MaxCfg failed to refuse non-matching Apple M3 Pro";
            return false;
        }

        // 2. Verify F32 linear parity
        constexpr int N = 32, K = 64, M = 2;
        mtl::Buffer bw = cx_.alloc((size_t)N * K * sizeof(float));
        mtl::Buffer bx = cx_.alloc((size_t)M * K * sizeof(float));
        mtl::Buffer by = cx_.alloc((size_t)M * N * sizeof(float));
        float* wp = bw.as<float>();
        float* xp = bx.as<float>();
        float* yp = by.as<float>();
        for (size_t i = 0; i < (size_t)N * K; ++i) wp[i] = std::sin((float)(i % 97)) * 0.5f - 0.25f;
        for (size_t i = 0; i < (size_t)M * K; ++i) xp[i] = std::cos((float)(i % 89)) * 0.5f;
        std::memset(yp, 0, (size_t)M * N * sizeof(float));

        mtl::Dispatch d;
        d.p = &p_f32;
        d.grid[0] = (uint32_t)((N + 3) / 4);
        d.grid[1] = (uint32_t)((M + 3) / 4);
        d.tg[0] = 128;
        d.args = {mtl::Arg::B(bw), mtl::Arg::B(bx), mtl::Arg::B(by),
                  mtl::Arg::V(N), mtl::Arg::V(K), mtl::Arg::V(M)};
        cx_.run({d});

        bool ok = true;
        for (int m = 0; m < M && ok; ++m) {
            for (int n = 0; n < N; ++n) {
                float ref = 0.0f;
                for (int k = 0; k < K; ++k) ref += wp[(size_t)n * K + k] * xp[(size_t)m * K + k];
                const float got = yp[(size_t)m * N + n];
                if (!std::isfinite(got) || std::fabs(ref - got) > 1e-4f * std::max(1.0f, std::fabs(ref))) {
                    err = "metal selfcheck F32 linear mismatch at (" + std::to_string(m) + "," + std::to_string(n) +
                          "): ref=" + std::to_string(ref) + " got=" + std::to_string(got);
                    ok = false;
                    break;
                }
            }
        }
        cx_.free(bw);
        cx_.free(bx);
        cx_.free(by);
        if (!ok) return false;

        // 3. Verify ALL 6 GSQ-RCO GGUF codec kernels {16, 17, 19, 21, 22, 42} against exact CPU dequant (< 1e-6!)
        const uint32_t rco_types[6] = {16u, 17u, 19u, 21u, 22u, 42u};
        const size_t rco_bpb[6] = {66, 74, 50, 110, 82, 18};
        const int rco_epb[6] = {256, 256, 256, 256, 256, 64};
        constexpr int GK = 256, GN = 4;
        mtl::Buffer g_x = cx_.alloc((size_t)GK * sizeof(float));
        mtl::Buffer g_y = cx_.alloc((size_t)GN * sizeof(float));
        float* gx = g_x.as<float>();
        for (int k = 0; k < GK; ++k) gx[k] = std::cos((float)(k * 13 + 7) * 0.031f) * 0.5f;

        uint64_t seed = 0x123456789abcdef0ull;
        for (int ti = 0; ti < 6 && ok; ++ti) {
            const uint32_t gt = rco_types[ti];
            const size_t rb = (size_t)(GK / rco_epb[ti]) * rco_bpb[ti];
            mtl::Buffer g_w = cx_.alloc((size_t)GN * rb);
            uint8_t* raw = g_w.as<uint8_t>();
            for (size_t b = 0; b < (size_t)GN * rb; ++b) raw[b] = (uint8_t)(m5_splitmix64(seed) & 0xFFu);
            // Set clean finite fp16 block scales (d = 0.25 -> 0x3400, 0.5 -> 0x3800) so no random NaN/Inf half scales!
            for (int r = 0; r < GN; ++r) {
                for (int bi = 0; bi < GK / rco_epb[ti]; ++bi) {
                    uint8_t* blk = raw + (size_t)r * rb + (size_t)bi * rco_bpb[ti];
                    const uint16_t d16 = (uint16_t)(0x3400u + ((r + bi) & 3u) * 0x0100u);
                    blk[0] = (uint8_t)(d16 & 0xFFu);
                    blk[1] = (uint8_t)(d16 >> 8);
                }
            }
            std::memset(g_y.ptr, 0, (size_t)GN * sizeof(float));
            mtl::Dispatch gd;
            gd.p = &p_gdot;
            gd.grid[0] = (uint32_t)GN;
            gd.tg[0] = 32;
            gd.args = {mtl::Arg::B(g_w), mtl::Arg::B(g_x), mtl::Arg::B(g_y), mtl::Arg::V(gt), mtl::Arg::V(GK)};
            cx_.run({gd});

            std::vector<float> cpu_row(GK);
            const float* gy = g_y.as<float>();
            for (int r = 0; r < GN; ++r) {
                if (!ink2::gguf_detail::gguf_dequant_row_f32(gt, raw + (size_t)r * rb, cpu_row.data(), GK)) {
                    err = "metal selfcheck: CPU dequant_row failed for GGUF type " + std::to_string(gt);
                    ok = false; break;
                }
                double ref = 0.0;
                for (int k = 0; k < GK; ++k) ref += (double)cpu_row[k] * (double)gx[k];
                const double diff = std::fabs((double)gy[r] - ref);
                if (!std::isfinite(gy[r]) || diff > 1e-5 * std::max(1.0, std::fabs(ref))) {
                    err = "metal selfcheck GGUF type " + std::to_string(gt) + " row " + std::to_string(r) +
                          " mismatch: ref=" + std::to_string(ref) + " got=" + std::to_string(gy[r]) +
                          " diff=" + std::to_string(diff);
                    ok = false; break;
                }
            }
            cx_.free(g_w);
        }
        cx_.free(g_x);
        cx_.free(g_y);
        return ok;
    }

private:
    void build_step(const StepBatch& batch, ink_metal::InkBatchStep& st) {
        st.M = batch.n_tokens;
        st.slot_ids = batch.slot_ids;
        st.pos = batch.pos;
        st.tok_ids = batch.token_ids;
        st.capture_taps = batch.capture_taps || runner_.use_dflash2;
        if (st.M > 1 && st.M <= 8) {
            bool same_slot = true;
            for (int m = 1; m < st.M; ++m) if (st.slot_ids[m] != st.slot_ids[0]) { same_slot = false; break; }
            if (same_slot) {
                st.record_verify_ring = true;
                last_ver_slot_ = st.slot_ids[0];
                last_ver_pos0_ = st.pos[0];
                last_ver_m_ = st.M;
            }
        }
        const int D = mod_.man.c.hidden;
        bool any_img = false;
        for (int idx : batch.img_embed_idx) if (idx >= 0) { any_img = true; break; }
        if (any_img && batch.img_embeds) {
            st.is_img.assign((size_t)st.M, 0);
            st.img_embeds.assign((size_t)st.M * D, 0.0f);
            for (int m = 0; m < st.M; ++m) {
                if (batch.img_embed_idx[m] >= 0) {
                    st.is_img[m] = 1;
                    std::memcpy(st.img_embeds.data() + (size_t)m * D,
                                batch.img_embeds + (size_t)batch.img_embed_idx[m] * D,
                                (size_t)D * sizeof(float));
                }
            }
        }
    }

    int choose_df2_draft_len(const float conf[7], int max_d) const {
        if (!runner_.m5.active) return std::min(7, max_d);
        const bool is_gguf = (mod_.man.build.find("gguf") != std::string::npos ||
                              mod_.man.build.find("GGUF") != std::string::npos);
        // M5 Max measured T(M)/T(1) = 1.0 + slope * (E_union(M) - 8.0), draft overhead ~0.14
        const double slope = is_gguf ? 0.009 : 0.011;
        const double draft_ovh = 0.14;
        return runner_.df2.choose_verify_drafts(conf, max_d, slope, draft_ovh);
    }

    double default_plain_ms() const {
        const bool is_gguf = (mod_.man.build.find("gguf") != std::string::npos ||
                              mod_.man.build.find("GGUF") != std::string::npos);
        if (is_gguf) return runner_.use_mpp_trunk_i8 ? 22.5 : 25.0;
        return runner_.use_mpp_trunk_i8 ? 16.5 : 19.0;
    }

    int dflash2_spec_step_greedy(int slot, int32_t anchor_tok, int anchor_pos, int max_emit,
                                 int32_t* out_emitted, int* out_drafted, std::string& err) {
        if (slot < 0 || slot >= runner_.max_slots || max_emit < 2 || anchor_pos <= 0 ||
            anchor_pos + 12 >= runner_.max_ctx) {
            return 0;
        }
        if (slot_cooldown_[slot] > 0) {
            --slot_cooldown_[slot];
            const double t0 = mtl::now_s();
            ink_metal::InkBatchStep st;
            st.M = 1;
            st.slot_ids = {slot};
            st.pos = {anchor_pos};
            st.tok_ids = {anchor_tok};
            st.capture_taps = true;
            st.compute_logits = true;
            int32_t tok = 0;
            if (!runner_.forward_batch(st, nullptr, &tok, err)) return -1;
            const double ms = (mtl::now_s() - t0) * 1e3;
            if (ms > 2.0 && ms < 200.0) plain_ms_ema_ = (plain_ms_ema_ <= 0.0) ? ms : (0.8 * plain_ms_ema_ + 0.2 * ms);
            out_emitted[0] = tok;
            if (out_drafted) *out_drafted = 0;
            last_ver_slot_ = -1;
            last_tree_slot_ = -1;
            return 1;
        }

        const double t_step0 = mtl::now_s();
        int32_t draft_ids[7]{};
        float draft_conf[7]{};
        const int n_avail = runner_.df2.draft_block8(slot, anchor_tok, anchor_pos, draft_ids, draft_conf);
        const int n_draft = choose_df2_draft_len(draft_conf, std::min(n_avail, max_emit - 1));

        if (n_draft <= 0) {
            const double t0 = mtl::now_s();
            ink_metal::InkBatchStep st;
            st.M = 1;
            st.slot_ids = {slot};
            st.pos = {anchor_pos};
            st.tok_ids = {anchor_tok};
            st.capture_taps = true;
            st.compute_logits = true;
            int32_t tok = 0;
            if (!runner_.forward_batch(st, nullptr, &tok, err)) return -1;
            const double ms = (mtl::now_s() - t0) * 1e3;
            if (ms > 2.0 && ms < 200.0) plain_ms_ema_ = (plain_ms_ema_ <= 0.0) ? ms : (0.8 * plain_ms_ema_ + 0.2 * ms);
            out_emitted[0] = tok;
            if (out_drafted) *out_drafted = 0;
            last_ver_slot_ = -1;
            last_tree_slot_ = -1;
            return 1;
        }

        const int tm = 1 + n_draft;
        ink_metal::InkBatchStep st;
        st.M = tm;
        st.slot_ids.assign((size_t)tm, slot);
        st.pos.resize((size_t)tm);
        st.tok_ids.resize((size_t)tm);
        st.tok_ids[0] = anchor_tok;
        st.pos[0] = anchor_pos;
        for (int i = 0; i < n_draft; ++i) {
            st.tok_ids[i + 1] = draft_ids[i];
            st.pos[i + 1] = anchor_pos + 1 + i;
        }
        st.capture_taps = true;
        st.compute_logits = true;
        st.record_verify_ring = true;
        int32_t ver_ids[32]{};
        if (!runner_.forward_batch(st, nullptr, ver_ids, err)) return -1;

        int acc = 0;
        for (int i = 0; i < n_draft; ++i) {
            if (ver_ids[i] == draft_ids[i]) out_emitted[acc++] = ver_ids[i];
            else break;
        }
        out_emitted[acc] = ver_ids[acc];
        if (acc < n_draft) {
            runner_.restore_verify_step(slot, acc);
            runner_.df2.rollback_slot(slot, anchor_pos + acc + 1);
        }

        last_ver_slot_ = slot;
        last_ver_pos0_ = anchor_pos;
        last_ver_m_ = tm;
        last_tree_slot_ = -1;
        if (out_drafted) *out_drafted = n_draft;

        // EWMA realized-gain guard using real measured plain_ms_ema_ (prevents false cooldowns AND protects prose!)
        const double t_plain_ms = (plain_ms_ema_ > 0.0) ? plain_ms_ema_ : default_plain_ms();
        const double step_ms = std::max(1.0, (mtl::now_s() - t_step0) * 1e3);
        const double step_gain = (double)(1 + acc) * t_plain_ms / step_ms;
        slot_ema_gain_[slot] = 0.72 * slot_ema_gain_[slot] + 0.28 * step_gain;
        if (step_gain >= 1.05) {
            slot_ema_gain_[slot] = std::max(slot_ema_gain_[slot], 1.12);
            slot_cd_len_[slot] = 6;
        } else if (slot_ema_gain_[slot] < 0.85) {
            slot_cooldown_[slot] = slot_cd_len_[slot];
            slot_cd_len_[slot] = std::min(16, slot_cd_len_[slot] + 4);
            slot_ema_gain_[slot] = 0.96;
        }
        return 1 + acc;
    }

    struct PromptCkpt {
        bool valid = false;
        int seq_len = 0;
        ink_metal::InkMetalRunner::SlotSconvSnap sconv;
        ink_metal::InkMetalDFlash2::RingSnap df2_ring;
        std::vector<float> frnn;
        std::vector<std::vector<uint8_t>> swa_k, swa_v;
        std::vector<uint16_t> dsp_k[3], dsp_v[3];
    };
    void drop_ckpt(int slot) {
        if (slot >= 0 && slot < (int)ckpts_.size()) ckpts_[slot].valid = false;
    }
    std::vector<PromptCkpt> ckpts_;
    mtl::Context cx_;
    ink_metal::InkMetalModel mod_;
    ink_metal::InkMetalRunner runner_;
    std::vector<ink_metal::InkMetalRunner::SlotSconvSnap> snaps_;
    std::vector<std::vector<float>> frnn_snaps_;
    std::vector<uint8_t> slot_used_;
    std::vector<int> slot_beam_depth_;
    std::vector<double> slot_ema_gain_;
    std::vector<int> slot_cooldown_;
    std::vector<int> slot_cd_len_;
    int last_ver_slot_ = -1, last_ver_pos0_ = -1, last_ver_m_ = 0;
    int last_tree_slot_ = -1, last_tree_anchor_ = -1;
    std::vector<int32_t> last_tree_path_;
    int active_slots_ = 0;
    double plain_ms_ema_ = 0.0;
    KvDtype kv_dtype_ = KvDtype::FP16;
};

std::unique_ptr<InkBackend> make_metal_backend() {
    return std::make_unique<MetalInkBackend>();
}

}  // namespace ink2
