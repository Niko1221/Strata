// CPU oracle: c++ -std=c++17 -O2 -DSTRATA_Q36_ROUTER_CPU_ONLY sycl/probe/q36_router_parity.cpp -o /tmp/q36-router-cpu
// GPU parity: link this source with sycl/src/kernels/cuda/native_router.dp.cpp using icpx -fsycl.
// GPU execution is explicit: the default GPU selector fails if no GPU is available; there is no CPU fallback.
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>
#ifndef STRATA_Q36_ROUTER_CPU_ONLY
#include <sycl/sycl.hpp>
#include "strata/kernels/native_router.hpp"
#include "../src/qwen35/q36_kernels.dp.hpp"
#endif
namespace {
struct Answer { std::vector<int32_t> ids; std::vector<float> weights; };
void require(bool ok, const std::string& what) { if (!ok) throw std::runtime_error(what); }
// Independent scalar oracle: rank finite logits, then normalize only the selected
// exponents in long double. For finite 256/top8 the selected softmax mass is >=8/256,
// so the router's 2^-14 clamp cannot be active. Inputs deliberately include exact
// ties, but avoid ambiguous near-ties introduced solely by native-exp rounding.
Answer reference(const float* x, int ne, int k) {
    require(ne >= k && k > 0, "oracle shape");
    Answer a; a.ids.resize(ne); std::iota(a.ids.begin(), a.ids.end(), 0);
    for (int e=0;e<ne;++e) require(std::isfinite(x[e]), "oracle requires finite logits");
    std::stable_sort(a.ids.begin(),a.ids.end(),[&](int l,int r) { return x[l] > x[r]; });
    a.ids.resize(k); a.weights.resize(k);
    std::vector<long double> p(k); long double sum=0;
    for(int j=0;j<k;++j) sum += p[j]=std::exp((long double)x[a.ids[j]]-x[a.ids[0]]);
    for(int j=0;j<k;++j) a.weights[j]=static_cast<float>(p[j]/sum);
    return a;
}
std::vector<std::vector<float>> fixtures(int ne=256) {
    std::vector<std::vector<float>> v;
    v.emplace_back(ne,0.0f);                         // all tie: low ID first, exactly 1/8
    v.emplace_back(ne,-19.0f);                     // shifted all-tie
    v.emplace_back(ne); for(int e=0;e<ne;++e) v.back()[e]=(e-ne)*0.125f;
    v.emplace_back(ne,-1000.0f);                   // top8 entirely in lane 0
    for(int j=0;j<8;++j) v.back()[32*j]=1000.0f-j*0.25f;
    v.emplace_back(ne,-80.0f);                     // interleaved ties across lanes
    for(int j=0;j<16;++j) v.back()[(j*47)%ne]=4.0f;
    v.emplace_back(ne,-1000.0f); v.back()[ne-1]=1000.0f; // underflowing losers
    std::mt19937 rng(0x36a3b);
    for(int t=0;t<31;++t) {
        v.emplace_back(ne);
        for(auto& x:v.back()) x=(int(rng()%2049)-1024)*0.0625f;
    }
    return v;
}
void check(const Answer& expected, const int32_t* ids,const float* w,const std::string& label) {
    float sum=0;
    for(size_t j=0;j<expected.ids.size();++j) {
        require(ids[j]==expected.ids[j], label+" ID mismatch at "+std::to_string(j));
        require(std::isfinite(w[j]) && w[j]>=0, label+" invalid weight");
        require(std::abs(w[j]-expected.weights[j])<=2e-5f,label+" weight mismatch at "+std::to_string(j));
        sum+=w[j];
    }
    require(std::abs(sum-1.0f)<3e-5f,label+" weights do not sum to one");
}
void cpu_checks() {
    auto f=fixtures();
    auto a=reference(f[0].data(),256,8);
    for(int j=0;j<8;++j) require(a.ids[j]==j && a.weights[j]==0.125f,"all-tie fixture");
    a=reference(f[2].data(),256,8);
    for(int j=0;j<8;++j) require(a.ids[j]==255-j,"ascending fixture");
    a=reference(f[3].data(),256,8);
    for(int j=0;j<8;++j) require(a.ids[j]==32*j,"owner-lane fixture");
    a=reference(f[5].data(),256,8);
    require(a.ids[0]==255 && a.weights[0]==1.0f,"extreme fixture");
    for(auto& row:f) { auto x=reference(row.data(),256,8); check(x,x.ids.data(),x.weights.data(),"oracle invariant"); }
    std::cout<<"CPU oracle: "<<f.size()<<" fixtures validated (GPU kernels not executed)\n";
}
#ifndef STRATA_Q36_ROUTER_CPU_ONLY
template<class T> struct Buffer {
    sycl::queue& q; T* p;
    Buffer(sycl::queue& q,size_t n):q(q),p(sycl::malloc_shared<T>(n,q)) { require(p!=nullptr,"USM allocation failed"); }
    ~Buffer(){sycl::free(p,q);} Buffer(const Buffer&)=delete;
};
void gpu_checks() {
    sycl::queue q(sycl::gpu_selector_v,sycl::property::queue::in_order{});
    const auto groups=q.get_device().get_info<sycl::info::device::sub_group_sizes>();
    require(std::find(groups.begin(),groups.end(),32)!=groups.end(),"router requires 32-lane subgroups");
    auto f=fixtures(); const int nt=static_cast<int>(f.size());
    Buffer<float> input(q,size_t(nt)*256), weights(q,size_t(nt)*8+2);
    Buffer<int32_t> ids(q,size_t(nt)*8+2);
    for(int t=0;t<nt;++t) std::copy(f[t].begin(),f[t].end(),input.p+256*t);
    ids.p[0]=ids.p[nt*8+1]=-9999; weights.p[0]=weights.p[nt*8+1]=-9999.0f;
    strata::kernels::native_router_topk(input.p,ids.p+1,weights.p+1,256,8,nt,&q);q.wait_and_throw();
    require(ids.p[0]==-9999&&ids.p[nt*8+1]==-9999&&weights.p[0]==-9999&&weights.p[nt*8+1]==-9999,"top8 wrote outside exact output span");
    for(int t=0;t<nt;++t) check(reference(f[t].data(),256,8),ids.p+1+8*t,weights.p+1+8*t,"top8 token "+std::to_string(t));
    // Test fused route+group windows of 1,2 and8 tokens, including same-lane
    // winners and tie-heavy cases. Standalone calls use one work-group per token;
    // the fused function uses one subgroup per token in a single work-group.
    Buffer<int32_t> fused_ids(q,64),starts(q,65),ng(q,1),dst(q,64),tok(q,64);
    Buffer<float> fused_w(q,64); Buffer<unsigned long long> ptr(q,64); Buffer<uint8_t> blobs(q,256);
    for(int n:{1,2,8}) for(int base:{0,3,15}) {
        q.submit([&](sycl::handler& h) {
            sycl::local_accessor<int,1> sids(sycl::range<1>(64),h),sid(sycl::range<1>(64),h);
            auto* lg=input.p+base*256; auto* fi=fused_ids.p;auto* fw=fused_w.p;
            auto* b=blobs.p;auto* gp=ptr.p;auto* gs=starts.p;auto* gn=ng.p;auto* ed=dst.p;auto* et=tok.p;
            h.parallel_for(sycl::nd_range<3>(sycl::range<3>(1,8,32),sycl::range<3>(1,8,32)),
                [=](sycl::nd_item<3>) [[sycl::reqd_sub_group_size(32)]] {
                    q36k::k_route_group<false>(lg,fi,fw,n,256,nullptr,b,1,gp,gs,gn,ed,et,
                        sids.get_multi_ptr<sycl::access::decorated::no>().get(),sid.get_multi_ptr<sycl::access::decorated::no>().get());
                });
        });q.wait_and_throw();
        require(std::memcmp(fused_ids.p,ids.p+1+base*8,size_t(n)*8*sizeof(int32_t))==0,"fused route IDs differ bitwise");
        require(std::memcmp(fused_w.p,weights.p+1+base*8,size_t(n)*8*sizeof(float))==0,"fused route weights differ bitwise");
        std::vector<int> order(n*8);std::iota(order.begin(),order.end(),0);
        std::stable_sort(order.begin(),order.end(),[&](int l,int r){return fused_ids.p[l]<fused_ids.p[r];});
        int g=0;
        for(int e=0;e<n*8;++e) {
            require(dst.p[e]==order[e]&&tok.p[e]==order[e]/8,"fused grouping entry mismatch");
            if(!e||fused_ids.p[order[e-1]]!=fused_ids.p[order[e]]) {
                require(starts.p[g]==e&&ptr.p[g]==reinterpret_cast<unsigned long long>(blobs.p+fused_ids.p[order[e]]),"fused group boundary mismatch");++g;
            }
        }
        require(*ng.p==g&&starts.p[g]==n*8,"fused final boundary mismatch");
    }
    // The generic 512/top10 entry must dispatch to the untouched legacy path.
    auto f512=fixtures(512); Buffer<float> in512(q,17*512),w1(q,170),w2(q,170);Buffer<int32_t> i1(q,170),i2(q,170);
    for(int t=0;t<17;++t)std::copy(f512[t].begin(),f512[t].end(),in512.p+512*t);
    for(int n:{1,17}) {
        strata::kernels::native_router_topk(in512.p,i1.p,w1.p,512,10,n,&q);
        if(n==1)strata::kernels::native_router_top10(in512.p,i2.p,w2.p,&q);
        else strata::kernels::native_router_top10_multi(in512.p,i2.p,w2.p,n,&q);
        q.wait_and_throw();
        require(std::memcmp(i1.p,i2.p,n*10*sizeof(int32_t))==0&&std::memcmp(w1.p,w2.p,n*10*sizeof(float))==0,"legacy top10 dispatcher changed bits");
    }
    bool rejected=false;
    try{strata::kernels::native_router_topk(input.p,ids.p,weights.p,256,9,1,&q);}catch(const std::invalid_argument&){rejected=true;}
    require(rejected,"unsupported router shape was not rejected");
    std::cout<<"GPU parity: "<<nt<<" top8 rows, 9 fused route/group windows, legacy top10 1/17 rows passed on "
             <<q.get_device().get_info<sycl::info::device::name>()<<"\n";
}
#endif
}
int main() {
    try { cpu_checks();
#ifndef STRATA_Q36_ROUTER_CPU_ONLY
        gpu_checks();
#endif
        return 0;
    } catch(const std::exception& e) { std::cerr<<e.what()<<"\n";return 1; }
}
