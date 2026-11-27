// GGML reference adapter for hy4-flash attention. Use FP32 latent KV for the reference input.

// Set STRATA_GGML_DIR to a built llama.cpp checkout.
//   nvcc -O2 -std=c++17 -shared -Xcompiler -fPIC tools/hy4_flash/ggml_diag_adapter.cpp \
//        -o build/libhy4_flash_diag.so -I"$STRATA_GGML_DIR/ggml/include" \
//        -L"$STRATA_GGML_DIR/build/bin" -lggml -lggml-base -lggml-cuda

#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cuda.h"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <vector>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <stdexcept>
template<class T>std::vector<T> get(const T*p,size_t n){std::vector<T>x(n);if(cudaMemcpy(x.data(),p,n*sizeof(T),cudaMemcpyDeviceToHost))throw std::runtime_error("copy");return x;}
extern "C" void hy4_flash_diag_fa(const float*qa,const float*qr,const float*kl,const float*kr,const float*sinks,int L,int R,int H,int T,int N,const int*qp,const int*kp,int window,float scale,float*out,void*stream){
 cudaStreamSynchronize((cudaStream_t)stream);
 auto a=get(qa,(size_t)L*T*H),r=get(qr,(size_t)R*T*H),k=get(kl,(size_t)L*N),p=get(kr,(size_t)R*N);auto qpos=get(qp,T),kpos=get(kp,N);
 int D=L+R,NP=(N+255)/256*256;std::vector<float>q((size_t)D*T*H);std::vector<half>kv((size_t)D*NP),mask((size_t)NP*T);
 for(int h=0;h<H;h++)for(int t=0;t<T;t++){for(int d=0;d<L;d++)q[(h*T+t)*D+d]=a[(h*T+t)*L+d];for(int d=0;d<R;d++)q[(h*T+t)*D+L+d]=r[(t*H+h)*R+d];}
 for(int j=0;j<N;j++){for(int d=0;d<L;d++)kv[j*D+d]=__float2half_rn(k[j*L+d]);for(int d=0;d<R;d++)kv[j*D+L+d]=__float2half_rn(p[j*R+d]);}
 int maxvalid=0;for(int t=0;t<T;t++){int valid=0;for(int j=0;j<NP;j++){bool ok=j<N&&kpos[j]<=qpos[t]&&(window==0||kpos[j]>qpos[t]-window);valid+=ok;mask[t*NP+j]=__float2half_rn(ok?0.f:-INFINITY);}maxvalid=std::max(maxvalid,valid);}
 static auto be=ggml_backend_cuda_init(0);ggml_init_params ip={4*1024*1024,nullptr,true};auto ctx=ggml_init(ip);
 auto Q=ggml_new_tensor_3d(ctx,GGML_TYPE_F32,D,T,H),K=ggml_new_tensor_3d(ctx,GGML_TYPE_F16,D,NP,1);auto V=ggml_view_3d(ctx,K,L,NP,1,K->nb[1],K->nb[2],0);auto M=ggml_new_tensor_2d(ctx,GGML_TYPE_F16,NP,T);
 auto Y=ggml_flash_attn_ext(ctx,Q,K,V,M,scale,0,0);ggml_tensor*S=nullptr;std::vector<float>sk;if(sinks){sk=get(sinks,H);S=ggml_new_tensor_1d(ctx,GGML_TYPE_F32,H);ggml_flash_attn_ext_add_sinks(Y,S);}ggml_prec_set_acc(Y,GGML_PREC_F32);ggml_flash_attn_ext_set_n_kv_max(Y,maxvalid);
 auto g=ggml_new_graph(ctx);ggml_build_forward_expand(g,Y);auto buf=ggml_backend_alloc_ctx_tensors(ctx,be);if(!buf)throw std::runtime_error("ggml alloc");
 ggml_backend_tensor_set(Q,q.data(),0,q.size()*4);ggml_backend_tensor_set(K,kv.data(),0,kv.size()*2);ggml_backend_tensor_set(M,mask.data(),0,mask.size()*2);if(S)ggml_backend_tensor_set(S,sk.data(),0,sk.size()*4);
 if(ggml_backend_graph_compute(be,g)!=GGML_STATUS_SUCCESS)throw std::runtime_error("ggml compute");std::vector<float>y((size_t)L*H*T),z(y.size());ggml_backend_tensor_get(Y,y.data(),0,y.size()*4);
 for(int h=0;h<H;h++)for(int t=0;t<T;t++)std::copy_n(y.data()+(t*H+h)*L,L,z.data()+(h*T+t)*L);
 cudaMemcpy(out,z.data(),z.size()*4,cudaMemcpyHostToDevice);ggml_backend_buffer_free(buf);ggml_free(ctx);
}

extern "C" void hy4_flash_diag_norm(float*x,const float*w,long long rows,long long cols,float eps,void*stream){cudaStreamSynchronize((cudaStream_t)stream);auto in=get(x,rows*cols);std::vector<float>wt;if(w)wt=get(w,cols);static auto be=ggml_backend_cuda_init(0);ggml_init_params ip={4*1024*1024,nullptr,true};auto ctx=ggml_init(ip);auto X=ggml_new_tensor_2d(ctx,GGML_TYPE_F32,cols,rows);auto Y=ggml_rms_norm(ctx,X,eps);ggml_tensor*W=nullptr;if(w){W=ggml_new_tensor_1d(ctx,GGML_TYPE_F32,cols);Y=ggml_mul(ctx,Y,W);}auto g=ggml_new_graph(ctx);ggml_build_forward_expand(g,Y);auto buf=ggml_backend_alloc_ctx_tensors(ctx,be);if(!buf)throw std::runtime_error("norm alloc");ggml_backend_tensor_set(X,in.data(),0,in.size()*4);if(w)ggml_backend_tensor_set(W,wt.data(),0,wt.size()*4);if(ggml_backend_graph_compute(be,g)!=GGML_STATUS_SUCCESS)throw std::runtime_error("norm compute");ggml_backend_tensor_get(Y,in.data(),0,in.size()*4);cudaMemcpy(x,in.data(),in.size()*4,cudaMemcpyHostToDevice);ggml_backend_buffer_free(buf);ggml_free(ctx);}

extern "C" void hy4_flash_diag_mm_f32(const float*w,const float*x,float*y,int ni,int no,int nt,void*stream){cudaStreamSynchronize((cudaStream_t)stream);auto wh=get(w,(size_t)ni*no),xh=get(x,(size_t)ni*nt);static auto be=ggml_backend_cuda_init(0);ggml_init_params ip={4*1024*1024,nullptr,true};auto ctx=ggml_init(ip);auto W=ggml_new_tensor_2d(ctx,GGML_TYPE_F32,ni,no),X=ggml_new_tensor_2d(ctx,GGML_TYPE_F32,ni,nt);auto Y=ggml_mul_mat(ctx,W,X);auto g=ggml_new_graph(ctx);ggml_build_forward_expand(g,Y);auto buf=ggml_backend_alloc_ctx_tensors(ctx,be);if(!buf)throw std::runtime_error("mm alloc");ggml_backend_tensor_set(W,wh.data(),0,wh.size()*4);ggml_backend_tensor_set(X,xh.data(),0,xh.size()*4);if(ggml_backend_graph_compute(be,g)!=GGML_STATUS_SUCCESS)throw std::runtime_error("mm compute");std::vector<float>z((size_t)no*nt);ggml_backend_tensor_get(Y,z.data(),0,z.size()*4);cudaMemcpy(y,z.data(),z.size()*4,cudaMemcpyHostToDevice);ggml_backend_buffer_free(buf);ggml_free(ctx);}
