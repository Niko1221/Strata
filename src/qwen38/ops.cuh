#pragma once
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <stdexcept>
#include <string>
#include <cstdint>

namespace strata::qwen38 {
inline void ck(cudaError_t e) { if(e!=cudaSuccess) throw std::runtime_error(cudaGetErrorString(e)); }
struct Buffer {
    void* data=nullptr; size_t bytes=0;
    Buffer()=default; explicit Buffer(size_t n){allocate(n);}
    Buffer(const Buffer&)=delete; Buffer& operator=(const Buffer&)=delete;
    Buffer(Buffer&& b) noexcept:data(b.data),bytes(b.bytes){b.data=nullptr;b.bytes=0;}
    Buffer& operator=(Buffer&& b) noexcept { if(this!=&b){if(data)cudaFree(data);data=b.data;bytes=b.bytes;b.data=nullptr;b.bytes=0;}return *this; }
    ~Buffer(){if(data)cudaFree(data);}
    void allocate(size_t n){if(data)throw std::runtime_error("buffer already allocated");ck(cudaMalloc(&data,n));bytes=n;}
    template<class T=float>T* as()const{return static_cast<T*>(data);}
    void zero(cudaStream_t s){ck(cudaMemsetAsync(data,0,bytes,s));}
};
void decode(int type,const void* input,float* output,int n,cudaStream_t stream);
void preload_ops();
void matvec(int type,const void* weights,const float* input,float* output,int n,int rows,cudaStream_t stream);
void rms(const float* x,const float* weight,float* out,int n,float eps,cudaStream_t s);
void add(float* x,const float* y,int n,cudaStream_t s);
void swiglu(float* gate,const float* up,int n,cudaStream_t s);
void gdn_gates(float* alpha,float* beta,const float* dt,const float* a,int n,cudaStream_t s);
void finite(const float* x,size_t n,int* error,cudaStream_t s);
void hadamard(const float* x,const float* signs,float* y,int n,bool inverse,bool grouped_gdn,cudaStream_t s);
// Text positions use identical t/h/w coordinates in IMRoPE sections 11/11/10.
// Cache layout: [KV head][context cell][256], no hidden fallback or sliding.
void attention_prepare(const float* qg,const float* k,const float* v,const float* qw,const float* kw,
                       float* q,float* gate,half* kc,half* vc,int pos,int ctx,float eps,float base,cudaStream_t s);
void attention_apply(const float* q,const float* gate,const half* kc,const half* vc,float* out,
                     float* scores,int pos,int ctx,cudaStream_t s,const int* status=nullptr);
void attention_normalize(const float* qg,const float* k,const float* qw,const float* kw,
                         float* q,float* gate,float* normalized_k,int pos,float eps,float base,cudaStream_t s);
}
