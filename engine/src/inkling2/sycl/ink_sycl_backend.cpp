// engine/src/inkling2/sycl/ink_sycl_backend.cpp - InkBackend plugin for Intel Arc SYCL (dg2 + bmg_g31 B70)
// CRITICAL BATTLEMAGE B70 RULE: Zero reqd_sub_group_size(8); strictly SG16!
// MIT License - Copyright (c) 2026 Strata contributors
#include "../ink_backend.h"
#include "../ink_vision.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

#define INK_SYCL_WITH_MODEL_FILE 1
#include "ink_sy_fit.h"
#include "ink_sy_runner.h"

namespace {
#if defined(__GNUC__) || defined(__clang__)
__attribute__((constructor(101)))
#endif
void ink_sycl_early_l0_v1_init() {
    ink_sycl::ensure_l0_v1_default();
}
struct InkSyclEarlyInitGuard {
    InkSyclEarlyInitGuard() { ink_sycl::ensure_l0_v1_default(); }
};
static InkSyclEarlyInitGuard g_ink_sycl_early_init_guard;
}  // namespace

namespace ink2 {

class SyclInkBackend final : public InkBackend {
public:
    const char* backend_name() const noexcept override { return mod_.ep2_active ? "sycl-ep2" : "sycl"; }

    bool dspark_ready() const override { return runner_.use_dflash2 || runner_.use_dspark; }

    // The one drafter rule, principle 2 (strata/core/ink_draft_rule.hpp): on a hybrid load (experts not all in VRAM) a draft step costs extra expert reads, so from 2048 tokens on the step is plain.
    bool draft_plain_now(int ctx_pos) const override {
        if (!runner_.use_dflash2) return false;
        return strata::core::ink_draft_plain_step(mod_.n_host_experts > 0, ctx_pos, draft_user_on_);
    }

    int effective_chunk() const override { return effective_chunk_; }

    bool supports_candidates() const noexcept override { return ink_sycl::arc_cfg().active; }

    bool supports_sampled_spec() const noexcept override { return runner_.use_dflash2 && ink_sycl::arc_cfg().active; }

    std::string device_info() const override {
        const auto& d = ink_sycl::dev_caps();
        std::string info = d.name + " (" + d.platform + ", " +
               std::to_string((int)(d.global_mem / (1024ull * 1024ull * 1024ull))) + " GiB, SG16)";
        if (mod_.ep2_active) {
            info += " + " + mod_.ep2_dev1_name + " (" +
                    std::to_string((int)(mod_.ep2_dev1_mem / (1024ull * 1024ull * 1024ull))) +
                    " GiB) [EP-2GPU split=" + std::to_string(mod_.ep2_split) + ", " +
                    (mod_.ep2_p2p ? "p2p" : "pinned") + "]";
        }
        return info;
    }

    // The main card's memory alone (the picture-reader rule, ink_vision_policy.h; stats() adds the second card's in ep2).
    size_t primary_device_bytes() const override { return (size_t)ink_sycl::dev_caps().global_mem; }

    BackendStats stats() const override {
        BackendStats s;
        s.total_experts = (size_t)mod_.man.c.n_phys * (size_t)mod_.man.c.n_experts;
        s.vram_experts_loaded = (size_t)mod_.n_vram_experts;
        s.ram_experts_loaded = (size_t)mod_.n_host_experts;
        s.vram_used_bytes = ink_sycl::dev_bytes_allocated() + mod_.vram_expert_bytes_g1;
        s.vram_total_bytes = ink_sycl::dev_caps().global_mem + (mod_.ep2_active ? mod_.ep2_dev1_mem : 0);
        s.active_slots = active_slots_;
        s.max_slots = runner_.max_slots;
        s.kv_dtype = kv_dtype_;
        return s;
    }

    bool init(const InkModelFile& model, const BackendOpts& opts, std::string& err) override {
        if (kv_dtype_is_int_quant(opts.kv_dtype)) {
            err = "sycl: --kv k8v4/k8v8 is not supported on the SYCL backend (allocates fp16-sized pages); use --kv fp8 or fp16";
            return false;
        }
        ink_sycl::apply_gpu_index(opts.gpu_index, opts.gpus);
        const auto& acfg = ink_sycl::arc_cfg();
        kv_dtype_ = opts.kv_dtype;
        if (acfg.fast && acfg.use_fp8_kv_auto && opts.kv_dtype_auto) {
            kv_dtype_ = KvDtype::FP8_E4M3;
        }
        const bool want_fp8_kv = (kv_dtype_ == KvDtype::FP8_E4M3);
        const int par = std::max(1, std::min(64, opts.parallel));
        const int ctx = std::max(256, opts.max_ctx);
        effective_chunk_ = (acfg.active && opts.chunk_auto) ? acfg.effective_chunk : 0;
        const int run_chunk = (effective_chunk_ > 0) ? effective_chunk_ : opts.chunk;
        const int batch_cap = std::max(256, std::max(run_chunk, par));
        if (opts.draft_forced && !std::getenv("STRATA_INK_DRAFTER")) ink_sycl::sy_setenv("STRATA_INK_DRAFTER", "dflash2", 0);   // --draft on
        std::string df2_path = opts.enable_dspark ? ink_sycl::df2_resolve_path(model.path, opts.dspark_path) : "";
        {   // asked for by name: --draft on, or STRATA_INK_DRAFTER=dflash2 (never vetoed, never cut to plain steps)
            const char* de = std::getenv("STRATA_INK_DRAFTER");
            const std::string dl = de ? ink_sycl::to_lower_ascii(de) : "";
            draft_user_on_ = opts.draft_forced || dl == "dflash2" || dl == "df2";
        }
        // Exact runner + KV + DFlash2/DSpark + Vision + XMX/Hybrid reserve (ink_sy_fit.h), so an 8-16 GB Arc card is told BEFORE the runner allocates
        // that the --parallel / --max-context asked for does not fit (a start used to die with "dalloc ...: out of device memory").
        const auto& mc = model.m.c;
        ink_sycl::InkSyFitGeom fit;
        const int core_len = mc.core_end - mc.core_start;
        for (int v = 0; v < mc.n_visits; ++v) {
            const int ip = (v < mc.core_start) ? v
                         : (v < mc.core_start + core_len * mc.num_passes)
                             ? (mc.core_start + (v - mc.core_start) % core_len)
                             : (mc.core_end + (v - mc.core_start - core_len * mc.num_passes));
            fit.visit_is_swa.push_back(mc.is_swa[ip] ? 1 : 0);
        }
        fit.swa_win = mc.swa_win;
        fit.n_kv = mc.n_kv;
        fit.head_dim = mc.head_dim;
        fit.hidden = mc.hidden;
        fit.dflash2 = !df2_path.empty();
        fit.dspark = !fit.dflash2 && opts.enable_dspark && ink2::dormant_dspark_allowed() && model.m.dspark.present;
        fit.vision = !opts.vision_path.empty();
        fit.kv_elem_bytes = want_fp8_kv ? 1 : 2;
        fit.sc_elem_bytes = (acfg.fast && acfg.use_bf16_sconv) ? 2 : 4;
        fit.intelarc_active = acfg.active;
        const size_t extra_reserve = ink_sycl::ink_sy_runner_reserve(fit, par, ctx, run_chunk);
        ink_sycl::ensure_l0_v1_default();
        mod_.draft_probe = {};
        mod_.draft_probe.bytes = fit.dflash2 ? ink_sycl::ink_sy_df2_bytes(par) : 0;
        mod_.draft_probe.user_on = draft_user_on_;
        if (model.kind == ModelSourceKind::Strata) {
            if (!mod_.load(model.path, opts.vram_experts_gb, err, extra_reserve, opts.gpus, opts.gpu_ids, opts.ep_split)) return false;
        } else {
            if (!mod_.load_from_model_file(model, opts.vram_experts_gb, err, extra_reserve)) return false;
        }
        if (mod_.draft_probe.vetoed) {   // the one drafter rule: it would move the model off full residency or cut the expert tier
            fprintf(stderr, "%s\n", mod_.draft_probe.line.c_str());
            df2_path.clear();
            fit.dflash2 = false;
        }
        {
            const size_t gpu_mem = ink_sycl::dev_caps().global_mem;
            const size_t used = ink_sycl::dev_bytes_allocated();
            const size_t avail = gpu_mem > used ? gpu_mem - used : 0;
            const std::string why = ink_sycl::ink_sy_fit_message(fit, par, ctx, run_chunk, avail);
            if (!why.empty()) {
                err = why;
                return false;
            }
        }
        if (!runner_.init(mod_, par, ctx, batch_cap, !fit.dflash2 && opts.enable_dspark && ink2::dormant_dspark_allowed(),
                          err, want_fp8_kv, df2_path)) {
            return false;
        }
        if (!opts.vision_path.empty()) {
            if (!runner_.vis.load(opts.vision_path, err)) return false;
        }
        slot_used_.assign((size_t)par, 0);
        return true;
    }

    bool alloc_slot(int slot, int /*max_ctx*/, KvDtype /*kv_dtype*/, std::string& err) override {
        if (slot < 0 || slot >= runner_.max_slots) { err = "sycl slot out of range"; return false; }
        if (!slot_used_[slot]) { slot_used_[slot] = 1; ++active_slots_; }
        runner_.reset_slot(slot);
        return true;
    }

    void free_slot(int slot) override {
        if (slot >= 0 && slot < runner_.max_slots && slot_used_[slot]) {
            slot_used_[slot] = 0;
            --active_slots_;
            runner_.reset_slot(slot);
        }
    }

    void reset_slot(int slot) override { runner_.reset_slot(slot); }
    void snapshot_slot(int slot) override { runner_.snapshot_slot(slot); }
    void rollback_slot(int slot, int keep_len) override { runner_.rollback_slot(slot, keep_len); }

    bool forward_batch(const StepBatch& batch, float* logits_out, TapCapture* taps_out, std::string& err) override {
        if (batch.n_tokens <= 0) return true;
        const int M_tot = batch.n_tokens;
        const int D = mod_.man.c.hidden;
        const int V = mod_.man.c.vocab_unpadded;
        const int cap = runner_.max_batch;

        if (taps_out != nullptr) {
            for (int t = 0; t < 4; ++t) taps_out->taps[t].assign((size_t)M_tot * D, 0.0f);
        }
        std::vector<float> chunk_taps;
        int dst_logit_row = 0;
        for (int off = 0; off < M_tot; off += cap) {
            const int M = std::min(cap, M_tot - off);
            if (taps_out != nullptr) chunk_taps.resize((size_t)4 * M * D);
            std::vector<uint8_t> is_img;
            std::vector<float> img_emb;
            bool any_img = false;
            for (int m = 0; m < M; ++m) if (batch.img_embed_idx[off + m] >= 0) { any_img = true; break; }
            if (any_img && batch.img_embeds) {
                is_img.assign((size_t)M, 0);
                img_emb.assign((size_t)M * D, 0.0f);
                for (int m = 0; m < M; ++m) {
                    const int idx = batch.img_embed_idx[off + m];
                    if (idx >= 0) {
                        is_img[m] = 1;
                        std::memcpy(img_emb.data() + (size_t)m * D,
                                    batch.img_embeds + (size_t)idx * D,
                                    (size_t)D * sizeof(float));
                    }
                }
            }

            std::vector<int> want_idx;
            if (logits_out != nullptr) {
                for (int m = 0; m < M; ++m) if (batch.want_logits[off + m] != 0) want_idx.push_back(m);
            }
            const int n_want = (int)want_idx.size();
            float* chunk_lg = (n_want > 0 && logits_out != nullptr) ? (logits_out + (size_t)dst_logit_row * V) : nullptr;
            int mode = 0;
            if (n_want == 1 && want_idx[0] == M - 1) mode = 1;
            else if (n_want > 0) mode = 2;

            if (!runner_.forward_batch(batch.token_ids.data() + off, batch.slot_ids.data() + off, batch.pos.data() + off, M,
                                       any_img ? is_img.data() : nullptr, any_img ? img_emb.data() : nullptr,
                                       mode, chunk_lg, nullptr, batch.capture_taps,
                                       (mode == 2) ? want_idx.data() : nullptr, n_want,
                                       nullptr, nullptr, nullptr,
                                       taps_out ? chunk_taps.data() : nullptr)) {
                err = "sycl forward_batch failed"; return false;
            }
            if (taps_out != nullptr) {
                for (int t = 0; t < 4; ++t) {
                    std::memcpy(taps_out->taps[t].data() + (size_t)off * D,
                                chunk_taps.data() + (size_t)t * M * D,
                                (size_t)M * D * sizeof(float));
                }
            }
            dst_logit_row += n_want;
        }
        return true;
    }

    bool forward_batch_cand(const StepBatch& batch, CandRows& out, std::string& err) override {
        if (batch.n_tokens <= 0) { out.rows = 0; out.k = 0; return true; }
        const int M_tot = batch.n_tokens;
        const int D = mod_.man.c.hidden;
        const int V = mod_.man.c.vocab_unpadded;
        const int cap = runner_.max_batch;
        const int n_tot_want = batch.count_logits();
        out.rows = n_tot_want;
        out.k = std::min(kInkCandidates, V);
        out.val.assign((size_t)n_tot_want * kInkCandidates, -1e38f);
        out.idx.assign((size_t)n_tot_want * kInkCandidates, 0);
        out.finite.assign((size_t)n_tot_want, 1);

        int dst_row = 0;
        for (int off = 0; off < M_tot; off += cap) {
            const int M = std::min(cap, M_tot - off);
            std::vector<uint8_t> is_img;
            std::vector<float> img_emb;
            bool any_img = false;
            for (int m = 0; m < M; ++m) if (batch.img_embed_idx[off + m] >= 0) { any_img = true; break; }
            if (any_img && batch.img_embeds) {
                is_img.assign((size_t)M, 0);
                img_emb.assign((size_t)M * D, 0.0f);
                for (int m = 0; m < M; ++m) {
                    const int idx = batch.img_embed_idx[off + m];
                    if (idx >= 0) {
                        is_img[m] = 1;
                        std::memcpy(img_emb.data() + (size_t)m * D,
                                    batch.img_embeds + (size_t)idx * D,
                                    (size_t)D * sizeof(float));
                    }
                }
            }
            std::vector<int> want_idx;
            for (int m = 0; m < M; ++m) if (batch.want_logits[off + m] != 0) want_idx.push_back(m);
            const int n_want = (int)want_idx.size();
            const int mode = (n_want > 0) ? 2 : 0;
            if (!runner_.forward_batch(batch.token_ids.data() + off, batch.slot_ids.data() + off, batch.pos.data() + off, M,
                                       any_img ? is_img.data() : nullptr, any_img ? img_emb.data() : nullptr,
                                       mode, nullptr, nullptr, batch.capture_taps,
                                       (mode == 2) ? want_idx.data() : nullptr, n_want,
                                       (mode == 2) ? (out.val.data() + (size_t)dst_row * kInkCandidates) : nullptr,
                                       (mode == 2) ? (out.idx.data() + (size_t)dst_row * kInkCandidates) : nullptr,
                                       (mode == 2) ? (out.finite.data() + dst_row) : nullptr)) {
                err = "sycl forward_batch_cand failed";
                return false;
            }
            dst_row += n_want;
        }
        return true;
    }

    int dspark_draft(int slot, int32_t anchor_tok, int anchor_pos, int max_draft,
                     int32_t* out_draft_ids, std::string& /*err*/) override {
        if (runner_.use_dflash2 && max_draft > 0) {
            return runner_.dflash2_draft(slot, anchor_tok, anchor_pos, max_draft, out_draft_ids);
        }
        if (!runner_.use_dspark || max_draft <= 0) return 0;
        int tmp[7]{};
        int n = runner_.dspark_draft_block8(slot, anchor_tok, anchor_pos, tmp);
        int ret = std::min(n, max_draft);
        for (int i = 0; i < ret; ++i) out_draft_ids[i] = tmp[i];
        return ret;
    }

    int dspark_spec_step_greedy(int slot, int32_t anchor_tok, int anchor_pos, int max_emit,
                                int32_t* out_emitted, int* out_drafted, std::string& err) override {
        if (!runner_.use_dflash2) return 0;
        const int r = runner_.dflash2_spec_step_greedy(slot, anchor_tok, anchor_pos, max_emit, out_emitted, out_drafted);
        if (r < 0) err = "sycl dflash2 greedy spec step failed";
        return r;
    }

    int dspark_spec_step_sampled(int slot, int32_t anchor_tok, int anchor_pos, int max_emit,
                                 float temperature, int top_k, float top_p, float min_p,
                                 uint64_t rng_seed, uint64_t rng_offset,
                                 int32_t* out_emitted, int* out_drafted, std::string& err) override {
        if (!runner_.use_dflash2 || !ink_sycl::arc_cfg().active) return 0;
        // the shared hook passes a (seed, offset) pair; the runner takes one 64-bit step seed
        const uint64_t step_seed = rng_seed ^ (rng_offset * 0x9e3779b97f4a7c15ull);
        const int r = runner_.dflash2_spec_step_sampled(
            slot, anchor_tok, anchor_pos, max_emit,
            temperature, top_k, top_p, min_p, step_seed,
            out_emitted, out_drafted);
        if (r < 0) err = "sycl dflash2 sampled spec step failed";
        return r;
    }

    bool encode_image(const ImageRec& img, std::vector<float>& out_embeds, std::string& err) override {
        if (!runner_.vis.loaded || img.file.empty()) return false;
        int ntok = 0;
        return runner_.vis.encode_image_file(img.file, out_embeds, ntok, err);
    }

    bool selfcheck(std::string& err) override {
        ink_sycl::apply_gpu_index(pending_gpu_index_, pending_gpus_);
        if (mod_.path.empty() && runner_.mod == nullptr) {
            return selfcheck_device(err);
        }
        StepBatch b;
        b.push(0, 0, 1, true);
        std::vector<float> lg((size_t)mod_.man.c.vocab_unpadded);
        runner_.reset_slot(0);
        if (!forward_batch(b, lg.data(), nullptr, err)) return false;
        for (float v : lg) {
            if (!std::isfinite(v)) { err = "sycl selfcheck produced non-finite logits"; return false; }
        }
        runner_.reset_slot(0);
        return true;
    }

    bool selfcheck_device(std::string& err) {
        constexpr int N = 32, K = 64, M = 2;
        std::vector<float> w((size_t)N * K), x((size_t)M * K);
        for (size_t i = 0; i < w.size(); ++i) w[i] = std::sin((float)(i % 97)) * 0.5f - 0.25f;
        for (size_t i = 0; i < x.size(); ++i) x[i] = std::cos((float)(i % 89)) * 0.5f;
        float* wd = sycl::malloc_device<float>(w.size(), ink_sycl::Q());
        float* xd = sycl::malloc_device<float>(x.size(), ink_sycl::Q());
        float* yd = sycl::malloc_device<float>((size_t)M * N, ink_sycl::Q());
        if (!wd || !xd || !yd) {
            if (wd) sycl::free(wd, ink_sycl::Q());
            if (xd) sycl::free(xd, ink_sycl::Q());
            if (yd) sycl::free(yd, ink_sycl::Q());
            err = "sycl selfcheck: device USM allocation failed";
            return false;
        }
        ink_sycl::h2d_sync(wd, w.data(), w.size() * sizeof(float));
        ink_sycl::h2d_sync(xd, x.data(), x.size() * sizeof(float));
        ink_sycl::dzero(yd, (size_t)M * N * sizeof(float));
        ink_sycl::k_linear_f32_sg16(wd, N, K, xd, yd, M);
        std::vector<float> y((size_t)M * N, 0.0f);
        ink_sycl::d2h(y.data(), yd, y.size() * sizeof(float));
        ink_sycl::wait();
        sycl::free(wd, ink_sycl::Q());
        sycl::free(xd, ink_sycl::Q());
        sycl::free(yd, ink_sycl::Q());
        for (int m = 0; m < M; ++m) {
            for (int n = 0; n < N; ++n) {
                float ref = 0.0f;
                for (int k = 0; k < K; ++k) ref += w[(size_t)n * K + k] * x[(size_t)m * K + k];
                const float got = y[(size_t)m * N + n];
                if (!std::isfinite(got) || std::fabs(ref - got) > 1e-4f * std::max(1.0f, std::fabs(ref))) {
                    err = "sycl selfcheck F32 SG16 linear mismatch at (" + std::to_string(m) + "," + std::to_string(n) +
                          "): ref=" + std::to_string(ref) + " got=" + std::to_string(got);
                    return false;
                }
            }
        }
        return true;
    }

private:
    friend std::unique_ptr<InkBackend> make_sycl_backend(const BackendOpts* opts);
    ink_sycl::InkSyModel mod_;
    ink_sycl::InkSyRunner runner_;
    std::vector<uint8_t> slot_used_;
    int active_slots_ = 0;
    int effective_chunk_ = 0;
    bool draft_user_on_ = false;   // --draft on / STRATA_INK_DRAFTER=dflash2
    KvDtype kv_dtype_ = KvDtype::FP16;
    int pending_gpu_index_ = 0;
    int pending_gpus_ = 1;
};

std::unique_ptr<InkBackend> make_sycl_backend(const BackendOpts* opts) {
    auto b = std::make_unique<SyclInkBackend>();
    if (opts) {
        b->pending_gpu_index_ = opts->gpu_index;
        b->pending_gpus_ = opts->gpus;
    }
    return b;
}

}  // namespace ink2
