// Owned context leases. Exported memory excludes the separately owned frontier.
#include "llama.h"
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>

static thread_local std::string error_text;
struct Lease { llama_context * ctx; uint64_t consumed_work = 0; };

extern "C" {
const char * ctt_error() { return error_text.c_str(); }
void * ctt_load(const char * path) {
    ggml_backend_load_all();
    auto p = llama_model_default_params();
    p.n_gpu_layers = 99;
    auto * m = llama_model_load_from_file(path, p);
    if (!m) { error_text = "Model load failed"; return nullptr; }
    if (llama_model_is_hybrid(m) || llama_model_is_recurrent(m) || llama_model_has_encoder(m)) {
        llama_model_free(m);
        error_text = "Unsupported hybrid/recurrent/encoder model";
        return nullptr;
    }
    return m;
}
void ctt_unload(void * model) { llama_model_free(static_cast<llama_model *>(model)); }
int ctt_vocab(void * model) { return llama_vocab_n_tokens(llama_model_get_vocab(static_cast<llama_model *>(model))); }
int ctt_piece(void * model, int token, char * target, int size) {
    return llama_token_to_piece(llama_model_get_vocab(static_cast<llama_model *>(model)),token,target,size,0,true);
}
bool ctt_eog(void * model, int token) {
    return llama_vocab_is_eog(llama_model_get_vocab(static_cast<llama_model *>(model)),token);
}
int ctt_meta(void * model, const char * key, char * value, size_t size) {
    return llama_model_meta_val_str(static_cast<llama_model *>(model), key, value, size);
}
void * ctt_new(void * model) {
    auto p = llama_context_default_params();
    p.n_ctx = 2048; p.n_batch = 2048; p.n_ubatch = 2048; p.n_seq_max = 1;
    p.n_threads = p.n_threads_batch = 2;
    p.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
    p.type_k = p.type_v = GGML_TYPE_F32;
    p.no_perf = false;
    auto * ctx = llama_init_from_model(static_cast<llama_model *>(model), p);
    if (!ctx) { error_text = "Context allocation failed"; return nullptr; }
    return new Lease{ctx};
}
void ctt_free(void * lease) { auto * l = static_cast<Lease *>(lease); llama_free(l->ctx); delete l; }
int ctt_position(void * lease) {
    return llama_memory_seq_pos_max(llama_get_memory(static_cast<Lease *>(lease)->ctx), 0);
}
int ctt_consume(void * lease, const int32_t * tokens, int count) {
    auto * l = static_cast<Lease *>(lease);
    const int start = ctt_position(lease)+1;
    auto b = llama_batch_init(count, 0, 1);
    b.n_tokens = count;
    for (int i=0; i<count; ++i) {
        b.token[i]=tokens[i]; b.pos[i]=start+i; b.n_seq_id[i]=1; b.seq_id[i][0]=0; b.logits[i]=(i==count-1);
    }
    const int result = llama_decode(l->ctx, b);
    llama_batch_free(b);
    if (!result) l->consumed_work += count;
    else error_text = "Backend decode failed; discard lease";
    return result;
}
const float * ctt_logits(void * lease) { return llama_get_logits_ith(static_cast<Lease *>(lease)->ctx, -1); }
uint64_t ctt_work(void * lease) { return static_cast<Lease *>(lease)->consumed_work; }
size_t ctt_state_size(void * lease) { return llama_state_get_size(static_cast<Lease *>(lease)->ctx); }
size_t ctt_export(void * lease, void * target, size_t size) {
    return llama_state_get_data(static_cast<Lease *>(lease)->ctx, static_cast<uint8_t *>(target), size);
}
size_t ctt_import(void * lease, const void * source, size_t size) {
    return llama_state_set_data(static_cast<Lease *>(lease)->ctx, static_cast<const uint8_t *>(source), size);
}
}
