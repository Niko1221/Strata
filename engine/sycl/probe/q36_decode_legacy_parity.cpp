// True legacy parity: link against strata_kernels, not a reimplementation of it.
// This deliberately uses random, non-dyadic half scales and float activations.
#include "../src/qwen35/q36_kernels.dp.hpp"
#include "strata/kernels/native_mmvq.hpp"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>
namespace {
namespace K=strata::kernels;
using QB=q36k::QB;
void require(bool ok,const std::string& s){if(!ok)throw std::runtime_error(s);}
uint32_t rng_state=0x36a3b123;
uint32_t random32(){auto x=rng_state;x^=x<<13;x^=x>>17;x^=x<<5;return rng_state=x;}
float random_float(){return (int(random32()%2000001)-1000000)/327731.0f;}
uint32_t bits(float x){uint32_t b;std::memcpy(&b,&x,4);return b;}
template<class T>struct Buffer {
    sycl::queue& q;T* p;size_t n;
    Buffer(sycl::queue& q,size_t n):q(q),p(sycl::malloc_shared<T>(n,q)),n(n){require(p!=nullptr,"USM allocation");}
    ~Buffer(){sycl::free(p,q);}Buffer(const Buffer&)=delete;
};
struct Weights {
    int ni,no,nb;bool packed;
    Buffer<uint8_t> raw,planes;
    Weights(sycl::queue& q,int ni,int no,bool packed):ni(ni),no(no),nb(ni/32),packed(packed),raw(q,size_t(nb)*no*34),planes(q,size_t(nb)*no*34){
        std::vector<uint8_t> host(raw.n);
        for(size_t b=0;b<size_t(nb)*no;++b){
            // Finite non-dyadic model-like scales: half exponent 4..8, random mantissa.
            uint16_t scale=uint16_t(((4+random32()%5)<<10)|(random32()&1023));
            std::memcpy(host.data()+b*34,&scale,2);
            for(int j=0;j<32;++j)host[b*34+2+j]=uint8_t(int8_t(int(random32()%255)-127));
        }
        std::copy(host.begin(),host.end(),raw.p);
        K::native_q8_0_pack_host(host.data(),planes.p,ni,no);
        if(packed){require(K::native_q8_0_packed_eligible(ni,no),"fixture packed shape");K::native_q8_0_packed_register(raw.p,planes.p,ni,no);}
    }
    ~Weights(){if(packed)K::native_q8_0_packed_unregister(raw.p);}
    const uint8_t* data()const{return packed?planes.p:raw.p;}
    const sycl::half* scales()const{return packed?reinterpret_cast<const sycl::half*>(planes.p+size_t(ni)*no):nullptr;}
};
size_t float_diff(const float* a,const float* b,size_t n,const std::string& label){
    size_t count=0;double max_abs=0;uint32_t max_ulp=0;
    for(size_t i=0;i<n;++i)if(bits(a[i])!=bits(b[i])){
        if(count<2)std::cout<<label<<" at "<<i<<" new="<<std::setprecision(9)<<a[i]<<" old="<<b[i]<<" bits="<<std::hex<<bits(a[i])<<"/"<<bits(b[i])<<std::dec<<"\n";
        ++count;max_abs=std::max(max_abs,std::abs(double(a[i])-b[i]));
        auto order=[](uint32_t u){return u&0x80000000u?~u:u|0x80000000u;};auto ua=order(bits(a[i])),ub=order(bits(b[i]));max_ulp=std::max(max_ulp,ua>ub?ua-ub:ub-ua);
    }
    std::cout<<label<<" differences="<<count<<"/"<<n<<" max_abs="<<max_abs<<" max_ulp="<<max_ulp<<"\n";return count;
}
size_t qb_diff(const QB* a,const QB* b,size_t n,const std::string& label){
    size_t blocks=0,scales=0,sums=0,quants=0;
    for(size_t i=0;i<n;++i){if(std::memcmp(a+i,b+i,sizeof(QB))!=0)++blocks;
        uint16_t ad[2],bd[2];std::memcpy(ad,&a[i].ds,4);std::memcpy(bd,&b[i].ds,4);scales+=ad[0]!=bd[0];sums+=ad[1]!=bd[1];
        for(int j=0;j<32;++j)quants+=a[i].qs[j]!=b[i].qs[j];}
    std::cout<<label<<" blocks="<<blocks<<" scale(d)="<<scales<<" sum(s)="<<sums<<" quant bytes="<<quants<<"\n";return blocks;
}
void q8_store_launch(sycl::queue& q,const float* x,QB* out,int n){
    q.parallel_for(sycl::nd_range<3>(sycl::range<3>(1,1,(n+255)/256*256),sycl::range<3>(1,1,256)),
        [=](sycl::nd_item<3> it)[[sycl::reqd_sub_group_size(32)]]{int i=it.get_global_id(2);if(i<n)q36k::q8_store(out,i,x[i],i&31);});
}
template<int NC>void make_activations(sycl::queue& q,Buffer<float>& f,Buffer<QB>& x,bool from_float){
    for(size_t i=0;i<f.n;++i)f.p[i]=random_float();
    if(from_float)K::native_quantize_q8_1(f.p,x.p,int(f.n)/NC,NC,&q);
    else for(size_t i=0;i<x.n;++i){uint16_t dh[2]={uint16_t(((5+random32()%4)<<10)|(random32()&1023)),uint16_t(((10+random32()%4)<<10)|(random32()&1023))};std::memcpy(&x.p[i].ds,dh,4);for(int j=0;j<32;++j)x.p[i].qs[j]=int8_t(int(random32()%255)-127);}
}
template<int NC,int NJ,bool PK>size_t rows_case(sycl::queue& q,int ni,int no,bool from_float){
    Weights w(q,ni,no,PK);Buffer<float>f(q,size_t(NC)*ni),a(q,size_t(NC)*no),b(q,size_t(NC)*no);Buffer<QB>x(q,size_t(NC)*ni/32),qx(q,size_t(NC)*ni/32);
    make_activations<NC>(q,f,x,from_float);
    K::native_mmvq(8,w.raw.p,x.p,b.p,ni,no,NC,&q);
    q36k::launch_q8_rows<NC,NJ,PK>(w.data(),w.scales(),x.p,a.p,ni/32,no,&q);q.wait_and_throw();
    std::string label="rows ni="+std::to_string(ni)+" no="+std::to_string(no)+" NC="+std::to_string(NC)+" PK="+std::to_string(PK)+" float_act="+std::to_string(from_float);
    size_t bad=float_diff(a.p,b.p,a.n,label);
    if(from_float){q8_store_launch(q,f.p,qx.p,int(f.n));q.wait_and_throw();bad+=qb_diff(qx.p,x.p,x.n,"standalone q8_store vs true native quantizer");}
    return bad;
}
template<int NC,int NJ,bool PK>size_t shared_case(sycl::queue& q,int ni,int no){
    Weights g(q,ni,no,PK),u(q,ni,no,PK);Buffer<float>f(q,size_t(NC)*ni),oldg(q,size_t(NC)*no),oldu(q,size_t(NC)*no),newg(q,size_t(NC)*no),newu(q,size_t(NC)*no);Buffer<QB>x(q,size_t(NC)*ni/32),oldq(q,size_t(NC)*no/32),sameq(q,size_t(NC)*no/32),fused(q,size_t(NC)*no/32);
    make_activations<NC>(q,f,x,true);
    K::native_mmvq(8,g.raw.p,x.p,oldg.p,ni,no,NC,&q);K::native_mmvq(8,u.raw.p,x.p,oldu.p,ni,no,NC,&q);
    K::native_swiglu_quantize_q8_1(oldg.p,oldu.p,oldq.p,no,NC,&q);
    q36k::launch_q8_rows<NC,NJ,PK>(g.data(),g.scales(),x.p,newg.p,ni/32,no,&q);q36k::launch_q8_rows<NC,NJ,PK>(u.data(),u.scales(),x.p,newu.p,ni/32,no,&q);
    auto*gg=oldg.p;auto*uu=oldu.p;auto*sq=sameq.p;int total=NC*no;
    q.parallel_for(sycl::nd_range<3>(sycl::range<3>(1,1,(total+255)/256*256),sycl::range<3>(1,1,256)),[=](sycl::nd_item<3>it)[[sycl::reqd_sub_group_size(32)]]{int i=it.get_global_id(2);if(i<total){float gi=gg[i];q36k::q8_store(sq,i,gi/(1.0f+sycl::native::exp(-gi))*uu[i],i&31);}});
    auto*wg=g.data();auto*wu=u.data();auto*dg=g.scales();auto*du=u.scales();auto*xx=x.p;auto*fo=fused.p;int nb=ni/32;
    q.submit([&](sycl::handler&h){sycl::local_accessor<float[NC][32],0>lg(h),lu(h);h.parallel_for(sycl::nd_range<3>(sycl::range<3>(1,1,no/32*1024),sycl::range<3>(1,1,1024)),[=](sycl::nd_item<3>)[[sycl::reqd_sub_group_size(32)]]{q36k::k_sh_gu<NC,NJ,PK>(wg,dg,wu,du,xx,fo,nb,no,lg,lu);});});q.wait_and_throw();
    std::string label="shared ni="+std::to_string(ni)+" no="+std::to_string(no)+" NC="+std::to_string(NC)+" PK="+std::to_string(PK);
    size_t bad=float_diff(newg.p,oldg.p,newg.n,label+" gate");bad+=float_diff(newu.p,oldu.p,newu.n,label+" up");bad+=qb_diff(sameq.p,oldq.p,oldq.n,label+" same input quant epilogue");bad+=qb_diff(fused.p,oldq.p,oldq.n,label+" fused vs true legacy");return bad;
}
template<int NC>size_t all_rows(sycl::queue& q){size_t bad=0;for(bool ff:{false,true}){bad+=rows_case<NC,1,false>(q,512,2048,ff);bad+=rows_case<NC,2,false>(q,2048,512,ff);bad+=rows_case<NC,2,true>(q,2048,2048,ff);}return bad;}
}
int main(){try{
    sycl::queue q(sycl::gpu_selector_v,[](sycl::exception_list es){for(auto&e:es)std::rethrow_exception(e);},sycl::property::queue::in_order{});
    std::cout<<"GPU "<<q.get_device().get_info<sycl::info::device::name>()<<" legacy STRATA_MMVQ_WIDE_32="<<(std::getenv("STRATA_MMVQ_WIDE_32")?std::getenv("STRATA_MMVQ_WIDE_32"):"default(1)")<<"\n";
    size_t bad=all_rows<1>(q)+all_rows<2>(q)+all_rows<3>(q)+all_rows<4>(q)+all_rows<8>(q);
    bad+=shared_case<1,1,false>(q,512,512);bad+=shared_case<4,1,false>(q,512,512);bad+=shared_case<1,2,false>(q,2048,512);bad+=shared_case<4,2,false>(q,2048,512);bad+=shared_case<1,2,true>(q,2048,2048);bad+=shared_case<4,2,true>(q,2048,2048);
    std::cout<<"TOTAL differing output entries/blocks="<<bad<<"\n";return bad?1:0;
}catch(const std::exception&e){std::cerr<<e.what()<<"\n";return 2;}}
