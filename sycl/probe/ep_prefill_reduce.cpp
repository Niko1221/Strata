// Independent FP32 router-order oracle and fresh IPC reduction test; run with the pair lease.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <vector>
#include "strata/core/ep_mem.hpp"
#include "strata/kernels/ep_kernels.hpp"

static float sample(size_t i, unsigned salt) {
    return float(int((uint32_t(i)*131u + salt*17u) % 2047u)-1023)/256.0f;
}

int main() {
    using namespace strata::core;
    constexpr int max_tokens=4096, width=2560, k=10;
    if(dpct::device_count()!=2) return 2;
    auto& q0=dpct::dev_mgr::instance().get_device(0).in_order_queue();
    auto& q1=dpct::dev_mgr::instance().get_device(1).in_order_queue();
    dpct::select_device(0);
    EpRegion peer_rows, peer_sum;
    std::string err;
    if(!ep_region_alloc(1,0,size_t(max_tokens)*k*width*4,peer_rows,err) ||
       !ep_region_alloc(1,0,size_t(max_tokens)*width*4,peer_sum,err)) {
        std::printf("FAIL %s\n",err.c_str());return 3;
    }
    auto* local=sycl::malloc_device<float>(size_t(max_tokens)*k*width,q0);
    auto* shared=sycl::malloc_device<float>(size_t(max_tokens)*width,q0);
    auto* out=sycl::malloc_device<float>(size_t(max_tokens)*width,q0);
    auto* gate=sycl::malloc_device<float>(max_tokens,q0);
    auto* slot0=sycl::malloc_device<int32_t>(max_tokens*k,q0);
    auto* slot1=sycl::malloc_device<int32_t>(max_tokens*k,q1);
    auto* w0=sycl::malloc_device<float>(max_tokens*k,q0);
    auto* w1=sycl::malloc_device<float>(max_tokens*k,q1);
    auto* pr=static_cast<float*>(peer_rows.local);
    auto* ps=static_cast<float*>(peer_sum.local);
    std::vector<int32_t> slots(max_tokens*k);
    std::vector<float> weights(max_tokens*k), gates(max_tokens), host(size_t(max_tokens)*width);
    double largest_abs=0,largest_scaled=0;
    int cases=0;
    for(int tokens : {1,17,256,4096}) for(int percent : {0,25,60,100}) {
        if(tokens==4096 && percent!=60) continue;
        for(unsigned epoch=1;epoch<=2;++epoch) {
            const int rows=tokens*k;
            auto owned=[=](int i){return (i*37+int(epoch)*11)%100>=percent;};
            int local_rows=0;
            for(int i=0;i<rows;++i) local_rows+=owned(i);
            int a=0,b=local_rows;
            for(int t=0;t<tokens;++t) {
                float norm=0;
                for(int j=0;j<k;++j) norm+=float((t*k+j)*13%17+1);
                for(int j=0;j<k;++j) {
                    int i=t*k+j;
                    slots[i]=owned(i)?a++:b++;
                    weights[i]=float(i*13%17+1)/norm;
                }
                gates[t]=sample(t,epoch);
            }
            const size_t local_elements=size_t(local_rows)*width;
            const size_t peer_elements=size_t(rows-local_rows)*width;
            if(local_elements) q0.parallel_for(sycl::range<1>(local_elements),[=](sycl::id<1> i){local[i]=sample(i,epoch);});
            if(peer_elements) q1.parallel_for(sycl::range<1>(peer_elements),[=](sycl::id<1> i){pr[i]=sample(i[0]+size_t(local_rows)*width,epoch);});
            q0.parallel_for(sycl::range<1>(size_t(tokens)*width),[=](sycl::id<1> i){shared[i]=sample(i,epoch+97);});
            q0.memcpy(slot0,slots.data(),rows*4);q1.memcpy(slot1,slots.data(),rows*4);
            q0.memcpy(w0,weights.data(),rows*4);q1.memcpy(w1,weights.data(),rows*4);
            q0.memcpy(gate,gates.data(),tokens*4);
            strata::kernels::ep_prefill_reduce(pr,slot1,w1,ps,tokens,local_rows,width,k,&q1);
            q1.wait();
            strata::kernels::ep_copy_uncached(out,peer_sum.remote,size_t(tokens)*width*4,&q0);
            strata::kernels::ep_prefill_finish(local,slot0,w0,shared,gate,out,tokens,local_rows,width,k,&q0);
            q0.memcpy(host.data(),out,size_t(tokens)*width*4).wait();
            size_t bad=0;
            double max_abs=0,max_scaled=0;
            for(int t=0;t<tokens;++t) for(int d=0;d<width;++d) {
                float reference=0;
                double magnitude=0;
                for(int j=0;j<k;++j) {
                    int i=t*k+j;
                    float v=sample(size_t(slots[i])*width+d,epoch);
                    reference=std::fma(weights[i],v,reference);
                    magnitude+=std::abs(double(weights[i])*v);
                }
                float sh=sample(size_t(t)*width+d,epoch+97)/(1.0f+std::exp(-gates[t]));
                reference+=sh;
                double error=std::abs(double(host[size_t(t)*width+d])-reference);
                double tolerance=2e-6*(1+magnitude+std::abs(sh));
                bad+=!std::isfinite(error) || error>tolerance;
                max_abs=std::max(max_abs,error);max_scaled=std::max(max_scaled,error/tolerance);
            }
            std::printf("REDUCE tokens=%d peer_percent=%d epoch=%u local_rows=%d max_abs=%.9g tolerance_ratio=%.6g bad=%zu\n",
                        tokens,percent,epoch,local_rows,max_abs,max_scaled,bad);
            std::fflush(stdout);
            if(bad) return 4;
            largest_abs=std::max(largest_abs,max_abs);largest_scaled=std::max(largest_scaled,max_scaled);++cases;
        }
    }
    q0.wait();q1.wait();
    sycl::free(local,q0);sycl::free(shared,q0);sycl::free(out,q0);sycl::free(gate,q0);
    sycl::free(slot0,q0);sycl::free(slot1,q1);sycl::free(w0,q0);sycl::free(w1,q1);
    ep_region_free(peer_sum);ep_region_free(peer_rows);
    std::printf("PASS cases=%d max_abs=%.9g tolerance_ratio=%.6g\n",cases,largest_abs,largest_scaled);
}
