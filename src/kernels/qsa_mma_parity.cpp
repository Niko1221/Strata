// Standalone opt-in qualification: same inputs, all four KV modes, capture/replay.
// This bounded random fixture is not a model-quality or performance qualification.
#include "strata/kernels/qsa_decode_attn.hpp"
#include "strata/kernels/f16_bits.hpp"
#include "strata/kernels/kv_q4.hpp"
#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>
namespace k = strata::kernels;
static void ck(cudaError_t e) {
    if (e != cudaSuccess) { std::fprintf(stderr, "%s\n", cudaGetErrorString(e)); std::exit(2); }
}
template<class T> static T* upload(const std::vector<T>& v) {
    T* p; ck(cudaMalloc(&p, v.size()*sizeof(T)));
    ck(cudaMemcpy(p,v.data(),v.size()*sizeof(T),cudaMemcpyHostToDevice)); return p;
}
int main() {
    if (!std::getenv("STRATA_QSA_SM75_MMA")) {
        std::fprintf(stderr,"Set STRATA_QSA_SM75_MMA=1 for qualification\n"); return 2;
    }
    std::mt19937 rng(1654); std::uniform_real_distribution<float> random(-1,1);
    int failures=0, cases=0; double worst=0, worst_rms=0;
    cudaStream_t stream; ck(cudaStreamCreate(&stream));
    for(int cap : {1,65,129,2051}) for(int mode=0;mode<4;++mode) {
        k::QsaShapes s{}; s.head_dim=256; s.n_head=24; s.n_head_kv=2; s.page_size=16;
        const int pages=(cap+15)/16, rows=pages*2*16;
        std::vector<uint16_t> h(rows*256), scales(rows*4,k::f16_from_f32(0.01f));
        std::vector<int8_t> codes(rows*256);
        std::vector<k::block_q4_0> q4(rows*8);
        for(auto& v:h) v=k::f16_from_f32(random(rng));
        for(auto& v:codes) v=(int8_t)(random(rng)*80);
        for(auto& v:q4) { v.d=k::f16_from_f32(0.1f); for(auto& b:v.qs) b=(uint8_t)rng(); }
        auto dh=upload(h); auto ds=upload(scales); auto dc=upload(codes); auto d4=upload(q4);
        for(int mask=0;mask<3;++mask) for(int nq : {1,2,3,4,5}) {
            std::vector<int32_t> pages_h(pages);
            for(int p=0;p<pages;++p) pages_h[p]=(mask==2 || (mask==1 && p%3==0)) ? -1 : pages-1-p;
            auto dp=upload(pages_h); k::QsaAttnPools pools{}; pools.page_table=dp;
            if(mode==0) pools.k_pool=pools.v_pool=dh;
            else if(mode==1) { pools.k_q=pools.v_q=dc; pools.k_scale=pools.v_scale=ds; }
            else if(mode==2) pools.k_q4=pools.v_q4=(uint8_t*)d4;
            else { pools.k_q=dc; pools.k_scale=ds; pools.v_q4=(uint8_t*)d4; }
            std::vector<float> q(nq*24*256); for(auto& v:q) v=random(rng);
            std::vector<int32_t> ids(nq*cap), steps(nq*k::kStepCount,0);
            for(int z=0;z<nq;++z) { steps[z*k::kStepCount+k::kStepWidth]=std::max(0,cap-z);
                for(int i=0;i<cap;++i) ids[z*cap+i]=cap-1-i; }
            auto dq=upload(q); auto di=upload(ids); auto dt=upload(steps);
            std::vector<float> zero(q.size(),0), scr(k::qsa_decode_attn_scratch_floats(cap,s)*nq,0);
            auto scratch=upload(scr); auto out=upload(zero);
            std::vector<float> ref(q.size()), got(q.size()), replay(q.size());
            k::qsa_decode_attn_batch(dq,pools,di,dt,cap,s,scratch,out,nq,stream,false);
            ck(cudaStreamSynchronize(stream)); ck(cudaMemcpy(ref.data(),out,q.size()*4,cudaMemcpyDeviceToHost));
            // First candidate invocation occurs during capture for the first mode/case.
            ck(cudaStreamBeginCapture(stream,cudaStreamCaptureModeGlobal));
            k::qsa_decode_attn_batch(dq,pools,di,dt,cap,s,scratch,out,nq,stream,true);
            cudaGraph_t graph; ck(cudaStreamEndCapture(stream,&graph));
            cudaGraphExec_t exec; ck(cudaGraphInstantiate(&exec,graph,nullptr,nullptr,0));
            ck(cudaGraphLaunch(exec,stream)); ck(cudaStreamSynchronize(stream));
            ck(cudaMemcpy(got.data(),out,q.size()*4,cudaMemcpyDeviceToHost));
            ck(cudaGraphLaunch(exec,stream)); ck(cudaStreamSynchronize(stream));
            ck(cudaMemcpy(replay.data(),out,q.size()*4,cudaMemcpyDeviceToHost));
            double err=0,power=0,maxerr=0; bool finite=true;
            for(size_t i=0;i<q.size();++i) { double d=got[i]-ref[i]; err+=d*d; power+=(double)ref[i]*ref[i];
                maxerr=std::max(maxerr,std::fabs(d)); finite &= std::isfinite(got[i]); }
            double rms=std::sqrt(err/std::max(power,1e-30));
            bool exact_fallback=(nq==1||nq==5);
            // Preset tolerance for bounded [-1,1] random inputs, not a quality claim.
            bool pass=finite && maxerr<0.005 && rms<0.005 &&
                !std::memcmp(got.data(),replay.data(),q.size()*4) &&
                (!exact_fallback || !std::memcmp(ref.data(),got.data(),q.size()*4));
            failures+=!pass; ++cases; worst=std::max(worst,maxerr); worst_rms=std::max(worst_rms,rms);
            if(!pass) std::printf("FAIL cap=%d mode=%d mask=%d T=%d abs=%g nrmse=%g\n",cap,mode,mask,nq,maxerr,rms);
            ck(cudaGraphExecDestroy(exec)); ck(cudaGraphDestroy(graph));
            for(void* p : { (void*)dp,(void*)dq,(void*)di,(void*)dt,(void*)scratch,(void*)out }) ck(cudaFree(p));
        }
        for(void* p : { (void*)dh,(void*)ds,(void*)dc,(void*)d4 }) ck(cudaFree(p));
    }
    ck(cudaStreamDestroy(stream));
    std::printf("cases=%d failures=%d max_abs=%g worst_nrmse=%g\n",cases,failures,worst,worst_rms);
    return failures?1:0;
}
