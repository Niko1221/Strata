// Actual Engine::add_rms_post host-path parity, without loading a model.
// Link strata_qwen35 compiled with STRATA_Q36_TEST_HOOKS and q36_addrms_hooks.inc.
// --cpu-only checks fixture geometry/signal without selecting a SYCL device.
#include "../src/qwen35/q36_test_hooks.hpp"
#include <dpct/dpct.hpp>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>
namespace D=strata::qwen35::test;
namespace {
constexpr int N=2048, Guard=16;
constexpr float Canary=-123456.75f, Eps=1e-6f;
enum class Alias { Distinct, YEqualsX, HEqualsY };
const char* alias_name(Alias a) {
    return a==Alias::Distinct?"distinct":a==Alias::YEqualsX?"y_equals_x":"h_equals_y";
}
void require(bool yes,const std::string& why) { if (!yes) throw std::runtime_error(why); }
struct Fixture {
    std::vector<float> x,y,w,expected_x;
    int T;
    std::string name;
};
Fixture fixture(int T,int pattern,Alias alias) {
    Fixture f;f.T=T;f.name=std::string(pattern==0?"non_dyadic":"mixed_row_scales")+"/"+alias_name(alias);
    f.x.resize(size_t(T)*N);f.y.resize(f.x.size());f.w.resize(N);f.expected_x.resize(f.x.size());
    const float scales[]={0.000137f,0.0173f,0.391f,2.731f,0.000731f,0.113f,1.371f,0.0371f};
    for (int j=0;j<N;++j)f.w[j]=0.731f+float((j*13+7)%101)/137.0f;
    for (int t=0;t<T;++t)for (int j=0;j<N;++j) {
        const size_t i=size_t(t)*N+j;
        const float scale=pattern==0?0.137f:scales[t];
        f.x[i]=float(int((size_t(t)*43+j*73+19)%257)-128)*(scale/131.0f);
        f.y[i]=float(int((size_t(t)*17+j*41+11)%251)-125)*(scale/197.0f);
        if (pattern==1 && j%113==0)f.y[i]=-0.917f*f.x[i]+scale*0.00137f;
        if (alias==Alias::YEqualsX)f.y[i]=f.x[i];
        f.expected_x[i]=f.x[i]+f.y[i];
    }
    return f;
}
void validate_fixture(const Fixture& f) {
    require(f.T>=1&&f.T<=8&&f.x.size()==size_t(f.T)*N,"fixture dimensions");
    for(int t=0;t<f.T;++t) {
        double signal=0;
        for(int j=0;j<N;++j) {
            const size_t i=size_t(t)*N+j;
            require(std::isfinite(f.x[i])&&std::isfinite(f.y[i])&&std::isfinite(f.expected_x[i]),"nonfinite fixture");
            signal+=std::abs(double(f.expected_x[i]));
        }
        require(signal>1e-6,"fixture is vacuous");
    }
}
#ifndef STRATA_Q36_ADDRMS_CPU_ONLY
template<class T>struct Guarded {
    sycl::queue& q; T* allocation; T* p; size_t n;
    Guarded(sycl::queue& q,size_t n,T canary):q(q),allocation(sycl::malloc_shared<T>(n+2*Guard,q)),p(allocation?allocation+Guard:nullptr),n(n),canary(canary) {
        require(allocation!=nullptr,"USM allocation failed");std::fill(allocation,allocation+n+2*Guard,canary);
    }
    ~Guarded(){sycl::free(allocation,q);}Guarded(const Guarded&)=delete;
    void copy(const std::vector<T>& values){require(values.size()==n,"fixture upload size");std::copy(values.begin(),values.end(),p);}
    void guard(const char* name) const {
        for(int j=0;j<Guard;++j)require(allocation[j]==canary&&p[n+j]==canary,std::string(name)+" canary damaged");
    }
    T canary;
};
void same_bits(const float* a,const float* b,size_t n,const std::string& what) {
    for(size_t i=0;i<n;++i) {
        require(std::isfinite(a[i])&&std::isfinite(b[i]),what+" nonfinite");
        if(std::memcmp(a+i,b+i,sizeof(float))!=0)
            throw std::runtime_error(what+" differs at "+std::to_string(i)+" ("+std::to_string(a[i])+" / "+std::to_string(b[i])+")");
    }
}
void check_cpu_norm(const float* got,const Fixture& f) {
    // Independent numerical check in addition to actual old/new bit equality.
    for(int t=0;t<f.T;++t) {
        double square=0;for(int j=0;j<N;++j){const double v=f.expected_x[size_t(t)*N+j];square+=v*v;}
        const double inverse=1/std::sqrt(square/N+Eps);
        for(int j=0;j<N;++j) {
            const size_t i=size_t(t)*N+j;const double expected=f.expected_x[i]*inverse*f.w[j];
            require(std::isfinite(got[i])&&std::abs(got[i]-expected)<=2e-6+2e-5*std::abs(expected),"actual old norm exceeds independent CPU tolerance");
        }
    }
}
void run_case(sycl::queue& q,int T,int pattern,Alias alias) {
    const auto f=fixture(T,pattern,alias);validate_fixture(f);const size_t count=f.x.size();
    Guarded<float> x(q,count,Canary),y(q,count,Canary),h(q,count,Canary),w(q,N,Canary);
    // Native q8_1 layout is 4 scale bytes +32 quantized values per32 inputs.
    Guarded<uint8_t> xq(q,count/32*36,uint8_t(0xa5));
    x.copy(f.x);y.copy(f.y);w.copy(f.w);
    const float* input_y=alias==Alias::YEqualsX?x.p:y.p;
    float* output_h=alias==Alias::HEqualsY?y.p:h.p;
    std::vector<float> old_x(count),old_h(count);
    const auto result=D::add_rms_host(&q,x.p,input_y,w.p,output_h,xq.p,N,T,Eps,old_x.data(),old_h.data());
    q.wait_and_throw();
    require(result.comparisons==3&&result.quantized,"actual host path did not execute all three old/new comparisons");
    same_bits(old_x.data(),f.expected_x.data(),count,"actual old residual vs CPU");
    same_bits(x.p,f.expected_x.data(),count,"fused residual vs CPU");
    same_bits(output_h,old_h.data(),count,"actual old/fused normalized h");
    check_cpu_norm(old_h.data(),f);
    if(alias!=Alias::HEqualsY)same_bits(y.p,f.y.data(),count,"read-only y");
    same_bits(w.p,f.w.data(),N,"read-only norm weights");
    x.guard("x");y.guard("y");h.guard("h");w.guard("w");xq.guard("xq");
    require(result.mismatches==0,"actual Engine::add_rms_post VERIFY reported "+std::to_string(result.mismatches)+" mismatches");
    std::cout<<"actual host add_rms T="<<T<<" case="<<f.name<<" comparisons="<<result.comparisons<<" mismatches="<<result.mismatches<<" old_vx_cpu=exact canaries=ok\n";
}
#endif
}
int main(int argc,char** argv) {
    try {
        const bool cpu_arg=argc==2&&std::string(argv[1])=="--cpu-only";
#ifdef STRATA_Q36_ADDRMS_CPU_ONLY
        const bool cpu_only=true;
#else
        const bool cpu_only=cpu_arg;
#endif
        require(argc==1||cpu_arg,"usage: q36_addrms_parity [--cpu-only]");
        int fixtures=0;
        for(int T:{1,2,4,8})for(int pattern:{0,1})for(Alias a:{Alias::Distinct,Alias::YEqualsX,Alias::HEqualsY}) {
            validate_fixture(fixture(T,pattern,a));++fixtures;
        }
        if(cpu_only){std::cout<<"CPU: "<<fixtures<<" actual-host fixtures validated; no GPU was selected\n";return 0;}
#ifndef STRATA_Q36_ADDRMS_CPU_ONLY
        auto& q=dpct::get_in_order_queue();require(q.get_device().is_gpu(),"requires a GPU SYCL device");
        std::cout<<"GPU: "<<q.get_device().get_info<sycl::info::device::name>()<<"\n";
        for(int T:{1,2,4,8})for(int pattern:{0,1})for(Alias a:{Alias::Distinct,Alias::YEqualsX,Alias::HEqualsY})run_case(q,T,pattern,a);
        std::cout<<"Actual Engine::add_rms_post: "<<fixtures<<" old/new host-path cases passed\n";return 0;
#else
        return 0;
#endif
    }catch(const std::exception& e){std::cerr<<"q36_addrms_parity: "<<e.what()<<"\n";return 1;}
}
