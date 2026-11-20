#pragma once
#include "ops.cuh"
#include <array>
#include <vector>

namespace strata::qwen38 {
void preload_kv();
enum class KvMode { f16, k8v4, spin8_k8v4, f16_turbo2, f16_turbo3, f16_turbo4 };
// Head geometry of the attention layers. Defaults are the validated 27B shape.
struct KvShape {
    int q_heads=24, kv_heads=4, head_dim=256;
    int q_elements()const{return q_heads*head_dim;}
    int kv_elements()const{return kv_heads*head_dim;}
};
// Immutable CPU admission. Invalid profiles fail before any CUDA allocation.
struct KvConfig {
    KvMode mode=KvMode::f16;
    bool automatic=false;
    int view=0;
    std::array<float,64> matrix{};
    std::string profile_sha="none",matrix_sha="none";
    static KvConfig admit(const std::string& mode,const std::string& view="",const std::string& profile="");
    const char* name() const;
    const char* key_format() const;
    const char* value_format() const;
    const char* view_name() const;
    size_t key_row_bytes() const;
    size_t value_row_bytes() const;
    int turbo() const;
    size_t scratch_bytes(const KvShape& shape={}, uint64_t capacity=0) const;
    size_t coefficient_bytes() const;
};
struct KvBudget {uint64_t weights=0,recurrent=0,scratch=0,upload_scratch=0,allocator_allowance=0;};
struct KvSelection {
    KvConfig config;
    std::string requested,reason;
    uint64_t available=0,reserve=0,required=0,payload=0;
    uint64_t allocator_slack=0;
};
uint64_t kv_storage_bytes(const KvConfig&,uint64_t capacity,uint64_t layers,const KvShape& shape={});
// Bounded score staging for the batched f16 attention path (one allocation).
uint64_t kv_batched_score_bytes(const KvShape& shape,uint64_t capacity);
uint64_t kv_allocator_slack(const KvConfig&,uint64_t capacity,uint64_t layers,const KvShape& shape={});
KvSelection select_kv(const KvConfig& requested,uint64_t capacity,uint64_t layers,const KvBudget&,uint64_t available,uint64_t reserve,const KvShape& shape={});
struct KvSpan {void* data=nullptr;size_t bytes=0;template<class T=float>T* as()const{return static_cast<T*>(data);}};
struct KvLayer { KvSpan key,value; int length=0; };
// One serialized session owns all history and bounded transform scratch.
// Inputs are post-normalization/post-RoPE Q/K, ordinary V and nonuniform gate.
// No prefix restore, shift, speculative overwrite or cross-session sharing.
class KvCache {
    KvConfig config_;
    KvShape shape_;
    int capacity_;
    cudaStream_t stream_;
    std::vector<KvLayer> layers_;
    Buffer keys_,values_;
    Buffer matrix_,q_,k_,v_,aggregate_,status_;
    // Bounded per-token score staging for the batched f16 attention path.
    // Capacity is capped so admission can account for it conservatively.
    Buffer batch_scores_;
    int batch_token_limit_=1;
    bool poisoned_=false;
    bool conservative_=false;
    uint64_t fences_=0;
public:
    KvCache(const KvConfig&,int capacity,int layers,cudaStream_t,bool conservative=false,const KvShape& shape={});
    void reset();
    void rewind(int new_length);
    void step(int layer,int position,const float* q,const float* k,const float* v,
              const float* gate,float* output,float* scores);
    // enqueue leaves errors sticky until check/reset; downstream cache kernels
    // cannot consume failed storage. Caller must check before publishing output.
    void enqueue(int layer,int position,const float* q,const float* k,const float* v,
                 const float* gate,float* output,float* scores);
    void enqueue_batch(int layer,int pos_start,int n_tokens,const float* q_batch,const float* k_batch,
                       const float* v_batch,const float* gate_batch,float* output_batch,float* scores);
    void check();
    uint64_t fences()const{return fences_;}
    const KvConfig& config()const{return config_;}
    size_t persistent_bytes()const;
    size_t scratch_bytes()const;
    size_t batch_score_bytes()const{return batch_scores_.bytes;}
    int batch_token_limit()const{return batch_token_limit_;}
    size_t coefficient_bytes()const{return matrix_.bytes;}
    const KvLayer& layer(int index)const{return layers_.at(index);}
};
// Production transform exposed to focused numerical tests; no history storage.
void kv_spin_transform(const float* input,float* output,const float* matrix,int rows,bool inverse,cudaStream_t);
}
