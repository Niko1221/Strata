#pragma once
#include "kv.hpp"
#include "resources.hpp"
#include "strata/kernels/fused_gdn.hpp"

namespace strata::qwen38 {
// Load kernel code and exercise the actual launch paths before admitting large
// allocations. Tiny zero inputs contain no checkpoint or user data. This is
// startup residency, not a performance/quality qualification or GPU setting.
inline void warmup_runtime(cudaStream_t stream){
    host_resources();size_t free=0,total=0;ck(cudaMemGetInfo(&free,&total));
    if(free<GPU_RESERVE+128*1024*1024)throw ResourceError("CUDA warmup requires reserve plus 128 MiB");
    preload_ops();preload_kv();
    Buffer q(6144*4),k(1024*4),v(1024*4),gate(6144*4),out(6144*4),scores(24*4);
    for(auto b:{&q,&k,&v,&gate,&out,&scores})b->zero(stream);
    for(const char* mode:{"f16","k8v4","spin8_k8v4","f16_turbo2","f16_turbo3","f16_turbo4"}){
        KvCache cache(KvConfig::admit(mode),1,1,stream);
        cache.step(0,0,q.as(),k.as(),v.as(),gate.as(),out.as(),scores.as());
    }
    Buffer state(128*128*4),history(128*3*4),conv(128*4*4),error(4);
    for(auto b:{&state,&history,&conv,&error})b->zero(stream);
    strata::kernels::fused_gdn_conv_l2(history.as(),q.as(),conv.as(),out.as(),128,1,1e-6f,stream);
    strata::kernels::fused_gdn_step_norm_silu(state.as(),k.as(),k.as(),v.as(),gate.as(),gate.as(),v.as(),k.as(),1e-6f,out.as(),1,1,stream);
    finite(state.as(),128*128,error.as<int>(),stream);finite(out.as(),128,error.as<int>(),stream);
    int bad=0;ck(cudaMemcpyAsync(&bad,error.data,4,cudaMemcpyDeviceToHost,stream));ck(cudaStreamSynchronize(stream));
    if(bad)throw std::runtime_error("non-finite native startup warmup");
    host_resources();
}
}
