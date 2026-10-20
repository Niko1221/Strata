// src/spec/dflash_drafter.cpp - see include/strata/spec/dflash_drafter.hpp.
// The selector walk and the injection batches follow llama.cpp's common_speculative_impl_draft_dflash
// (third_party/llama.cpp/common/speculative.cpp), for one sequence.
#include "strata/spec/dflash_drafter.hpp"

#include "strata/spec/dflash_backend.hpp"

#include "llama.h"
#include "llama-ext.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace strata::spec {

struct DflashDrafter::Impl {
    llama_model* model = nullptr;
    llama_context* ctx = nullptr;
    llama_batch batch{};
    llama_batch inject{};
    int n_tgt = 0, n_dec = 0, n_enc = 0, block = 8, top_k = 0, n_ubatch = 64;   // 64: the compute buffer stays ~0.2 GB
    bool mrope = false;
    llama_token mask = 0;
    int pos_max = -1;
    const float* lat = nullptr;   // the last draft's lattice rows (llama.cpp's buffer: valid until the next decode)

    ~Impl() {
        if (batch.token) llama_batch_free(batch);
        if (inject.embd) {
            if (mrope) { std::free(inject.pos); inject.pos = nullptr; }
            llama_batch_free(inject);
        }
        if (ctx) llama_free(ctx);
        if (model) llama_model_free(model);
    }
};

DflashDrafter::DflashDrafter() = default;
DflashDrafter::~DflashDrafter() = default;

bool DflashDrafter::loaded() const { return nat_ || (d_ && d_->ctx); }
int DflashDrafter::target_hidden() const { return nat_ ? nat_->target_hidden() : d_ ? d_->n_tgt : 0; }
int DflashDrafter::block() const { return nat_ ? nat_->block() : d_ ? d_->block : 0; }
int DflashDrafter::pos_max() const { return nat_ ? nat_->pos_max() : d_ ? d_->pos_max : -1; }
const float* DflashDrafter::lattice() const { return nat_ ? nat_->lattice() : d_ ? d_->lat : nullptr; }

static void quiet_log(ggml_log_level level, const char* text, void*) {
    if (level >= GGML_LOG_LEVEL_WARN || std::getenv("STRATA_DFLASH_LOG")) std::fputs(text, stderr);
}

bool DflashDrafter::load(const std::string& path, std::string& err, Backend backend) {
    if (backend == Backend::kAuto) {
        const char* e = std::getenv("STRATA_DFLASH_NATIVE");
        backend = e && e[0] && e[0] != '0' ? Backend::kNative : Backend::kLlama;
    }
    if (backend == Backend::kNative) {
        auto nat = make_dflash_native();
        if (!nat->load(path, err)) return false;
        nat_ = std::move(nat);
        return true;
    }
    llama_log_set(quiet_log, nullptr);
    llama_backend_init();
    auto d = std::make_unique<Impl>();
    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = 999;
    d->model = llama_model_load_from_file(path.c_str(), mp);
    if (!d->model) { err = "--dflash: cannot load " + path; return false; }
    if (llama_model_target_layer_ids_n(d->model) != (uint32_t) kNumLayers) {
        err = "--dflash: the draft does not read 5 target layers";
        return false;
    }
    const int32_t* ids = llama_model_target_layer_ids(d->model);
    for (int k = 0; k < kNumLayers; ++k)
        if (ids[k] != kLayers[k] + 1) {   // llama.cpp names a layer's input: the output of the layer before
            err = "--dflash: the draft's target layers are not 5/15/25/35/45";
            return false;
        }
    d->n_dec = llama_model_n_embd(d->model);
    d->n_tgt = d->n_dec;   // DFlash2 for Flash-Next: draft and target hidden are both 2560
    d->n_enc = kNumLayers * d->n_tgt;
    d->top_k = llama_model_dflash_selector_top_k(d->model);
    if (d->top_k <= 0) { err = "--dflash: not a DFlash2 draft (no candidate selector)"; return false; }
    char buf[32] = {};
    if (llama_model_meta_val_str(d->model, "dflash.block_size", buf, sizeof buf) >= 0) d->block = std::atoi(buf);
    d->mask = llama_vocab_mask(llama_model_get_vocab(d->model));
    d->mrope = llama_model_rope_type(d->model) == LLAMA_ROPE_TYPE_MROPE;

    llama_context_params cp = llama_context_default_params();
    cp.n_ctx = 0;   // the model's (262144); every layer is a 2048 sliding window, so the cache stays small
    cp.n_batch = cp.n_ubatch = (uint32_t) d->n_ubatch;
    cp.n_seq_max = 1;
    cp.swa_full = false;
    cp.embeddings = false;
    d->ctx = llama_init_from_model(d->model, cp);
    if (!d->ctx) { err = "--dflash: cannot create the draft context"; return false; }
    llama_set_n_threads(d->ctx, 1, 1);   // the CPU part is token_embd's row gather: a pool per decode costs more than it does
    llama_set_embeddings_nextn(d->ctx, true, /*masked*/ false);   // the selector lattice comes out of h_nextn
    llama_set_causal_attn(d->ctx, false);
    d->batch = llama_batch_init(d->block, 0, 1);
    d->inject = llama_batch_init(d->n_ubatch, d->n_enc, 1);
    if (d->mrope) {
        std::free(d->inject.pos);
        d->inject.pos = (llama_pos*) std::malloc(sizeof(llama_pos) * 4 * (size_t) d->n_ubatch);
    }
    d_ = std::move(d);
    return true;
}

void DflashDrafter::trim(int pos) {
    if (nat_) { nat_->trim(pos); return; }
    if (!loaded()) return;
    llama_memory_seq_rm(llama_get_memory(d_->ctx), 0, pos <= 0 ? -1 : pos, -1);
    if (pos <= 0) llama_memory_seq_rm(llama_get_memory(d_->ctx), 0, -1, -1);
    d_->pos_max = std::min(d_->pos_max, pos - 1);
    if (pos <= 0) d_->pos_max = -1;
}

namespace {
struct Timer {
    double& ms;
    std::chrono::steady_clock::time_point t0 = std::chrono::steady_clock::now();
    ~Timer() { ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count(); }
};
}  // namespace

bool DflashDrafter::inject(const float* feats, int n, int pos0, std::string& err) {
    if (!loaded() || n <= 0) return true;
    Timer tm{ms_inject};
    ++n_inject;
    if (nat_) return nat_->inject(feats, n, pos0, err);
    Impl& d = *d_;
    if (pos0 <= d.pos_max) trim(pos0);
    else if (d.pos_max >= 0 && pos0 != d.pos_max + 1) trim(0);   // a gap: positions it never saw
    for (int off = 0; off < n; off += d.n_ubatch) {
        const int c = std::min(d.n_ubatch, n - off);
        d.inject.n_tokens = c;
        std::memcpy(d.inject.embd, feats + (size_t) off * d.n_enc, (size_t) c * d.n_enc * sizeof(float));
        for (int i = 0; i < c; ++i) {
            const llama_pos p = pos0 + off + i;
            d.inject.pos[i] = p;
            if (d.mrope) {
                d.inject.pos[1 * c + i] = p;
                d.inject.pos[2 * c + i] = p;
                d.inject.pos[3 * c + i] = 0;
            }
            d.inject.n_seq_id[i] = 1;
            d.inject.seq_id[i][0] = 0;
            d.inject.logits[i] = false;
        }
        const int rc = llama_decode(d.ctx, d.inject);
        if (rc != 0) {
            err = "--dflash: injecting the target features failed (" + std::to_string(rc) + ")";
            trim(0);
            return false;
        }
        d.pos_max = pos0 + off + c - 1;
    }
    return true;
}

int DflashDrafter::draft(int32_t id_last, int pos0, int max_n, float p_min, int32_t* out, float* conf) {
    if (!loaded() || max_n <= 0) return 0;
    Timer tm{ms_draft};
    ++n_draft;
    if (nat_) return nat_->draft(id_last, pos0, max_n, p_min, out, conf);
    Impl& d = *d_;
    if (d.pos_max != pos0 - 1) return 0;   // the features up to id_last's position are not all in
    const int nb = std::min(d.block, max_n + 1);
    d.batch.n_tokens = 0;
    for (int i = 0; i < nb; ++i) {
        d.batch.token[i] = i == 0 ? id_last : d.mask;
        d.batch.pos[i] = pos0 + i;
        d.batch.n_seq_id[i] = 1;
        d.batch.seq_id[i][0] = 0;
        d.batch.logits[i] = false;
        ++d.batch.n_tokens;
    }
    const int rc = llama_decode(d.ctx, d.batch);
    int n = 0;
    if (rc == 0) {
        const float* lattice = llama_get_embeddings_nextn(d.ctx);
        if (lattice) {
            d.lat = lattice;
            int pred = 0;
            for (int i = 1; i < nb; ++i) {
                const float* row = lattice + (size_t) i * d.n_dec;
                const float* sc = row + d.top_k + (size_t) pred * d.top_k;
                pred = (int) (std::max_element(sc, sc + d.top_k) - sc);
                // softmax(scores) at the argmax, i.e. 1 / sum(exp(s_k - s_max))
                float sum = 0.0f;
                for (int k = 0; k < d.top_k; ++k) sum += std::exp(sc[k] - sc[pred]);
                if (1.0f / sum < p_min) break;
                if (conf) conf[n] = 1.0f / sum;
                out[n++] = (int32_t) row[pred];
            }
        }
    }
    llama_memory_seq_rm(llama_get_memory(d.ctx), 0, pos0, -1);   // the noise rows are not context
    d.pos_max = pos0 - 1;
    return n;
}

}  // namespace strata::spec
