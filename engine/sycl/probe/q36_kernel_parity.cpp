// Standalone header GPU probe: icpx -fsycl -std=c++20 -O2 -fp-model=precise
// -I sycl/include -I include -I $DPCT/include sycl/probe/q36_kernel_parity.cpp -o q36-kernel-parity
// --cpu-only validates the independent Q8 oracle without selecting any GPU.
// Add STRATA_Q36_TEST_DRIVER_HOOKS and link the engine compiled with
// STRATA_Q36_TEST_HOOKS for real conv/GDN/attention kernels. Never include engine.cpp here.
#include "../src/qwen35/q36_kernels.dp.hpp"
#ifdef STRATA_Q36_TEST_DRIVER_HOOKS
#include "../src/qwen35/q36_test_hooks.hpp"
#endif
#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>
namespace {
using QB=q36k::QB;
void require(bool yes,const std::string& s){if(!yes)throw std::runtime_error(s);}
float value(size_t i,float scale=1.0f){return (int((i*73+19)%257)-128)*scale;}
template<class T>struct Buf {
    sycl::queue& q;T* p;size_t n;
    Buf(sycl::queue& q,size_t n):q(q),p(sycl::malloc_shared<T>(n,q)),n(n){require(p!=nullptr,"USM allocation");}
    ~Buf(){sycl::free(p,q);}Buf(const Buf&)=delete;
    void copy(const std::vector<T>& x){require(x.size()<=n,"upload span");std::copy(x.begin(),x.end(),p);}
};
template<class T>void bits(const T* a,const T* b,size_t n,const std::string& s){
    if(std::memcmp(a,b,n*sizeof(T))==0)return;
    size_t i=0;for(;i<n&&std::memcmp(a+i,b+i,sizeof(T))==0;++i){}
    throw std::runtime_error(s+" bit mismatch at "+std::to_string(i));
}
void near(const float* a,const std::vector<double>& b,const std::string& s,double abs=1e-4,double rel_l2=1e-4){
    double mx=0,se=0,sr=0;
    for(size_t i=0;i<b.size();++i){require(std::isfinite(a[i])&&std::isfinite(b[i]),s+" nonfinite");
        const double d=double(a[i])-b[i];mx=std::max(mx,std::abs(d));se+=d*d;sr+=b[i]*b[i];}
    const double rl=std::sqrt(se/std::max(sr,1e-30));
    std::cout<<s<<" max_abs="<<mx<<" relative_l2="<<rl<<"\n";
    require(mx<=abs&&rl<=rel_l2,s+" exceeds fixed numerical tolerance");
}
float xor_sum(std::array<float,32> x){for(int m=16;m;m>>=1){auto old=x;for(int l=0;l<32;++l)x[l]=old[l]+old[l^m];}return x[0];}
QB quant_cpu(const float* x){
    QB q{};std::array<float,32> sum;float a=0;
    for(int l=0;l<32;++l){a=std::max(a,std::fabs(x[l]));sum[l]=x[l];}
    const float d=std::min(a/127.0f,65504.0f),s=xor_sum(sum);
    q.ds=sycl::half2(sycl::half(d),sycl::half(std::max(-65504.0f,std::min(65504.0f,s))));
    for(int l=0;l<32;++l)q.qs[l]=a==0?0:int8_t(std::max(-127.0f,std::min(127.0f,std::round(x[l]/d))));
    return q;
}
void quant_gpu(sycl::queue& q,const float* x,QB* out,size_t n){
    require(n%32==0,"quant shape");
    q.parallel_for(sycl::nd_range<3>(sycl::range<3>(1,1,(n+255)/256*256),sycl::range<3>(1,1,256)),
        [=](sycl::nd_item<3> it) [[sycl::reqd_sub_group_size(32)]] {
            const size_t i=it.get_global_id(2);if(i<n)q36k::q8_store(out,int(i),x[i],int(i&31));
        });
}
struct Q8Fixture {
    int nc,nb,rows;std::vector<uint8_t> raw,packed;std::vector<sycl::half> scales;std::vector<QB> x;
    Q8Fixture(int nc,int nb,int rows):nc(nc),nb(nb),rows(rows),raw(size_t(rows)*nb*34),packed(size_t(rows)*nb*32),scales(size_t(rows)*nb),x(size_t(nc)*nb){
        for(size_t b=0;b<scales.size();++b){scales[b]=sycl::half(std::ldexp(1.0f,-8-int(b%4)));std::memcpy(raw.data()+34*b,&scales[b],2);
            for(int j=0;j<32;++j){int8_t v=int8_t(int((b*37+j*11)%255)-127);raw[b*34+2+j]=uint8_t(v);packed[b*32+j]=uint8_t(v);}}
        for(size_t b=0;b<x.size();++b){x[b].ds=sycl::half2(sycl::half(std::ldexp(1.0f,-9-int(b%4))),sycl::half(0));
            for(int j=0;j<32;++j)x[b].qs[j]=int8_t(int((b*17+j*23)%255)-127);}
    }
    std::vector<float> ordered()const {
        std::vector<float> y(size_t(nc)*rows);
        for(int c=0;c<nc;++c)for(int r=0;r<rows;++r){std::array<float,32> lane{};
            for(int l=0;l<32;++l){float p[4]={};for(int z=0;z<4;++z)for(int j=0;j<(nb+31)/32;++j){const int b=l/4+8*z+32*j;if(b>=nb)continue;
                int dot=0;for(int e=0;e<8;++e)dot+=int(int8_t(raw[(size_t(r)*nb+b)*34+2+8*(l%4)+e]))*int(x[size_t(c)*nb+b].qs[8*(l%4)+e]);
                p[z]+=float(scales[size_t(r)*nb+b])*float(x[size_t(c)*nb+b].ds[0])*float(dot);}
                lane[l]=p[0];lane[l]+=p[1];lane[l]+=p[2];lane[l]+=p[3];}
            y[size_t(c)*rows+r]=xor_sum(lane);}
        return y;
    }
    std::vector<double> scalar()const {
        std::vector<double> y(size_t(nc)*rows);
        for(int c=0;c<nc;++c)for(int r=0;r<rows;++r)for(int b=0;b<nb;++b)for(int e=0;e<32;++e)
            y[size_t(c)*rows+r]+=double(float(scales[size_t(r)*nb+b]))*double(float(x[size_t(c)*nb+b].ds[0]))*int(int8_t(raw[(size_t(r)*nb+b)*34+2+e]))*int(x[size_t(c)*nb+b].qs[e]);
        return y;
    }
};
void cpu_checks(){for(int nb:{1,8,33,64}){Q8Fixture f(2,nb,5);auto y=f.ordered();near(y.data(),f.scalar(),"Q8 CPU oracle nb="+std::to_string(nb));}
    float x[32]={};auto q=quant_cpu(x);for(int i=0;i<32;++i)require(q.qs[i]==0,"zero quant oracle");
    std::cout<<"CPU oracle checks passed; GPU has not been selected\n";
}
void quant_checks(sycl::queue& q){const size_t n=16*32;Buf<float>x(q,n);Buf<QB>out(q,n/32+2);std::vector<QB>gold(n/32);
    for(size_t i=0;i<n;++i)x.p[i]=value(i,1.0f/64);
    std::fill(x.p,x.p+32,0);for(int j=0;j<32;++j){x.p[32+j]=(j&1)?1e9f:-1e9f;x.p[64+j]=1e9f;x.p[96+j]=value(j,1e-10f);}
    for(size_t b=0;b<n/32;++b)gold[b]=quant_cpu(x.p+32*b);
    std::memset(out.p,0xa5,(n/32+2)*sizeof(QB));QB left=out.p[0],right=out.p[n/32+1];
    quant_gpu(q,x.p,out.p+1,n);q.wait_and_throw();bits(out.p+1,gold.data(),gold.size(),"q8_store CPU/GPU");bits(out.p,&left,1,"q8 left guard");bits(out.p+n/32+1,&right,1,"q8 right guard");
    std::cout<<"q8_store: zero, signed/sum overflow, tiny and ordinary blocks bitwise passed\n";
}
template<int NC,int NJ>void rows_case(sycl::queue& q,int nb,int rows){Q8Fixture f(NC,nb,rows);Buf<uint8_t>w(q,f.raw.size()),pk(q,f.packed.size());Buf<sycl::half>d(q,f.scales.size());Buf<QB>x(q,f.x.size());Buf<float>a(q,size_t(NC)*rows),b(q,size_t(NC)*rows);w.copy(f.raw);pk.copy(f.packed);d.copy(f.scales);x.copy(f.x);
    q36k::launch_q8_rows<NC,NJ,false>(w.p,nullptr,x.p,a.p,nb,rows,&q);q36k::launch_q8_rows<NC,NJ,true>(pk.p,d.p,x.p,b.p,nb,rows,&q);q.wait_and_throw();
    auto gold=f.ordered();bits(a.p,gold.data(),gold.size(),"Q8 ordered CPU/GPU");bits(a.p,b.p,gold.size(),"Q8 packed/unpacked");near(a.p,f.scalar(),"Q8 nc="+std::to_string(NC)+" nb="+std::to_string(nb)+" rows="+std::to_string(rows));
}
template<int NC,int NJ,bool PK>void shared_case(sycl::queue& q,int nb){const int rows=64;Q8Fixture g(NC,nb,rows),u(NC,nb,rows);
    // Independent up matrix, both signs and magnitudes differ from the gate.
    for(size_t b=0;b<u.scales.size();++b)for(int j=0;j<32;++j){int8_t v=int8_t(int((b*53+j*7)%239)-119);u.raw[b*34+2+j]=uint8_t(v);u.packed[b*32+j]=uint8_t(v);}
    Buf<uint8_t>gw(q,PK?g.packed.size():g.raw.size()),uw(q,PK?u.packed.size():u.raw.size());Buf<sycl::half>gd(q,g.scales.size()),ud(q,u.scales.size());Buf<QB>x(q,g.x.size()),out(q,NC*rows/32),ref(q,NC*rows/32);Buf<float>ga(q,NC*rows),up(q,NC*rows),prod(q,NC*rows);
    gw.copy(PK?g.packed:g.raw);uw.copy(PK?u.packed:u.raw);gd.copy(g.scales);ud.copy(u.scales);x.copy(g.x);
    auto*wg=gw.p;auto*wu=uw.p;auto*dg=PK?gd.p:nullptr;auto*du=PK?ud.p:nullptr;auto*xx=x.p;auto*oo=out.p;
    q.submit([&](sycl::handler& h){sycl::local_accessor<float[NC][32],0>lg(h),lu(h);
        h.parallel_for(sycl::nd_range<3>(sycl::range<3>(1,1,rows/32*1024),sycl::range<3>(1,1,1024)),[=](sycl::nd_item<3>) [[sycl::reqd_sub_group_size(32)]] {q36k::k_sh_gu<NC,NJ,PK>(wg,dg,wu,du,xx,oo,nb,rows,lg,lu);});});
    q36k::launch_q8_rows<NC,NJ,PK>(gw.p,dg,x.p,ga.p,nb,rows,&q);q36k::launch_q8_rows<NC,NJ,PK>(uw.p,du,x.p,up.p,nb,rows,&q);
    const float*gp=ga.p;const float*uu=up.p;float*pp=prod.p;q.parallel_for(sycl::range<1>(NC*rows),[=](sycl::id<1>i){const float v=gp[i];pp[i]=v/(1+sycl::native::exp(-v))*uu[i];});quant_gpu(q,prod.p,ref.p,NC*rows);q.wait_and_throw();bits(out.p,ref.p,NC*rows/32,"shared_gu fused/reference");
    std::cout<<"shared_gu bitwise nc="<<NC<<" nb="<<nb<<" packed="<<PK<<" passed\n";
}
// A 256-thread, separate-launch baseline checks the small fused MoE's changed
// work distribution, RMS summation order and materialize-before-quantize rule.
void epilogue_reference(sycl::queue& q,float* x,const float* y,const float* wn,float* out,int n,int T,const float*parts=nullptr,const int32_t*prow=nullptr,const float*wts=nullptr,const float*shared=nullptr,const float*sgate=nullptr){
    q.submit([&](sycl::handler& h){sycl::local_accessor<float,1>sum(sycl::range<1>(256),h);
        h.parallel_for(sycl::nd_range<3>(sycl::range<3>(1,1,T*256),sycl::range<3>(1,1,256)),[=](sycl::nd_item<3>it){int t=it.get_group(2),l=it.get_local_id(2);float s=0;
            for(int i=l;i<n;i+=256){float v;if(parts){float acc=wts[t*8]*parts[size_t(prow?prow[t*8]:t*8)*n+i];for(int k=1;k<8;++k)acc=sycl::fma(wts[t*8+k],parts[size_t(prow?prow[t*8+k]:t*8+k)*n+i],acc);acc+=1.0f/(1+sycl::native::exp(-sgate[t]))*shared[t*n+i];v=x[t*n+i]+acc;}else v=x[t*n+i]+y[t*n+i];x[t*n+i]=v;s+=v*v;}
            if(!wn)return;sum[l]=s;it.barrier(sycl::access::fence_space::local_space);for(int k=128;k;k>>=1){if(l<k)sum[l]+=sum[l+k];it.barrier(sycl::access::fence_space::local_space);}float r=sycl::rsqrt(sum[0]/n+1e-6f);for(int i=l;i<n;i+=256)out[t*n+i]=x[t*n+i]*r*wn[i];});});
}
void epilogue_case(sycl::queue& q,int n,int T,bool moe,bool norm,bool remap){const size_t nn=size_t(n)*T;Buf<float>x(q,nn),refx(q,nn),y(q,nn),w(q,n),out(q,nn),refout(q,nn),parts(q,nn*8),weights(q,T*8),shared(q,nn),sg(q,T);Buf<int32_t>prow(q,T*8);Buf<QB>qx(q,nn/32),rq(q,nn/32);
    for(size_t i=0;i<nn;++i){x.p[i]=refx.p[i]=value(i,1.0f/128);y.p[i]=value(i+7,1.0f/256);shared.p[i]=value(i+3,1.0f/512);}for(int i=0;i<n;++i)w.p[i]=1+value(i,1.0f/1024);for(size_t i=0;i<nn*8;++i)parts.p[i]=value(i+53,1.0f/512);for(int t=0;t<T;++t){sg.p[t]=value(t,1.0f/256);for(int k=0;k<8;++k){weights.p[t*8+k]=(k+1)/36.0f;prow.p[t*8+k]=T*8-1-(t*8+k);}}
    auto*xx=x.p;auto*yy=y.p;auto*ww=norm?w.p:nullptr;auto*oo=out.p;auto*qq=qx.p;auto*pa=parts.p;auto*pr=remap?prow.p:nullptr;auto*wt=weights.p;auto*ss=shared.p;auto*sgp=sg.p;
    q.submit([&](sycl::handler& h){sycl::local_accessor<float,1>sh(sycl::range<1>(256),h),vv(sycl::range<1>(2048),h);sycl::local_accessor<const float*,1>pk(sycl::range<1>(16),h);
        const int local=moe?1024:256;
        h.parallel_for(sycl::nd_range<3>(sycl::range<3>(1,1,T*local),sycl::range<3>(1,1,local)),[=](sycl::nd_item<3>) [[sycl::reqd_sub_group_size(32)]] {
            if(moe)q36k::k_moe_out_small(xx,pa,pr,wt,8,ss,sgp,1,ww,oo,n,1e-6f,vv.get_multi_ptr<sycl::access::decorated::no>().get(),sh.get_multi_ptr<sycl::access::decorated::no>().get(),pk.get_multi_ptr<sycl::access::decorated::no>().get(),norm?qq:nullptr);
            else q36k::k_add_rms_q(xx,yy,ww,oo,qq,n,1e-6f,sh.get_multi_ptr<sycl::access::decorated::no>().get());});});
    epilogue_reference(q,refx.p,y.p,ww,refout.p,n,T,moe?parts.p:nullptr,pr,wt,ss,sgp);if(norm)quant_gpu(q,refout.p,rq.p,nn);q.wait_and_throw();bits(x.p,refx.p,nn,"epilogue x");if(norm){bits(out.p,refout.p,nn,"epilogue normalized h");bits(qx.p,rq.p,nn/32,"epilogue q8 image");}
    std::cout<<(moe?"moe_out_small":"add_rms_q")<<" bitwise n="<<n<<" T="<<T<<" norm="<<norm<<" remap="<<remap<<" passed\n";
}
void group_checks(sycl::queue& q){Buf<int32_t>ids(q,64),slots(q,256),dst(q,64),tok(q,64),start(q,65),ng(q,1);Buf<uint8_t>blobs(q,2048);Buf<unsigned long long>ptr(q,64);for(int e=0;e<256;++e)slots.p[e]=e%3==0?-1:255-e;for(int i=0;i<64;++i)ids.p[i]=(i*47)%17;
    for(int n:{0,1,8,64})for(bool hy:{false,true}){auto*ii=ids.p;auto*sl=slots.p;auto*bb=blobs.p;auto*pp=ptr.p;auto*st=start.p;auto*gg=ng.p;auto*dd=dst.p;auto*tt=tok.p;
        q.submit([&](sycl::handler& h){sycl::local_accessor<int,1>sid(sycl::range<1>(64),h);h.parallel_for(sycl::nd_range<3>(sycl::range<3>(1,1,64),sycl::range<3>(1,1,64)),[=](sycl::nd_item<3>){auto*sp=sid.get_multi_ptr<sycl::access::decorated::no>().get();if(hy)q36k::k_group_small<true>(ii,n,8,sl,bb,8,pp,st,gg,dd,tt,sp);else q36k::k_group_small<false>(ii,n,8,nullptr,bb,8,pp,st,gg,dd,tt,sp);});});q.wait_and_throw();
        std::vector<int>order;for(int e=0;e<n;++e)if(!hy||slots.p[ids.p[e]]>=0)order.push_back(e);std::stable_sort(order.begin(),order.end(),[&](int a,int b){return ids.p[a]<ids.p[b];});int g=0;for(size_t e=0;e<order.size();++e){require(dst.p[e]==order[e]&&tok.p[e]==order[e]/8,"group entries");if(!e||ids.p[order[e-1]]!=ids.p[order[e]]){require(start.p[g]==int(e)&&ptr.p[g]==reinterpret_cast<unsigned long long>(blobs.p+(hy?slots.p[ids.p[order[e]]]:ids.p[order[e]])*8),"group boundary/pointer");++g;}}require(*ng.p==g&&start.p[g]==int(order.size()),"group sentinel");}
    std::cout<<"group_small CPU/GPU HY/non-HY empty/1/8/64 passed\n";
}
#ifdef STRATA_Q36_TEST_DRIVER_HOOKS
namespace D=strata::qwen35::test;
void gdn_oracle(const float* input,int C,const float* a,const float* b,const float* z,const float* gamma,int HK,int HV,int T,std::vector<double>& state,std::vector<double>& out){
    const int S=128;out.resize(size_t(T)*HV*S);
    for(int t=0;t<T;++t)for(int h=0;h<HV;++h){const int qh=h%HK;double raw[128],g=std::exp(double(a[t*HV+h]));
        for(int c=0;c<S;++c){double kv=0;for(int r=0;r<S;++r)kv+=state[(size_t(r)*HV+h)*S+c]*input[size_t(t)*C+HK*S+qh*S+r];double delta=(input[size_t(t)*C+2*HK*S+h*S+c]-g*kv)*b[t*HV+h];raw[c]=0;
            for(int r=0;r<S;++r){auto& v=state[(size_t(r)*HV+h)*S+c];v=g*v+input[size_t(t)*C+HK*S+qh*S+r]*delta;raw[c]+=v*input[size_t(t)*C+qh*S+r];}raw[c]/=std::sqrt(128.0);}
        double ss=0;for(double v:raw)ss+=v*v;const double inv=1/std::sqrt(ss/128+1e-6);for(int c=0;c<S;++c){size_t i=(size_t(t)*HV+h)*S+c;out[i]=raw[c]*inv*gamma[c]*(z[i]/(1+std::exp(-double(z[i]))));}}
}
void driver_gdn(sycl::queue& q,int HK,int HV,int T){const int C=(2*HK+HV)*128;const size_t sn=size_t(128)*HV*128,yn=size_t(T)*HV*128;Buf<float>state(q,sn),seq(q,sn),held(q,sn),cols(q,sn),oc(q,yn),input(q,size_t(T)*C),alpha(q,T*HV),beta(q,T*HV),z(q,yn),gamma(q,128),out(q,yn),os(q,yn),oh(q,yn);std::vector<double>goldstate(sn),gold;
    for(size_t i=0;i<sn;++i)state.p[i]=seq.p[i]=held.p[i]=cols.p[i]=float(goldstate[i]=value(i,1.0f/16384));for(size_t i=0;i<size_t(T)*C;++i)input.p[i]=value(i,1.0f/1024);for(int i=0;i<T*HV;++i){alpha.p[i]=-0.1f+(i%17)*0.005f;beta.p[i]=0.1f+(i%17)*0.04f;}for(size_t i=0;i<yn;++i)z.p[i]=value(i+17,1.0f/128);for(int i=0;i<128;++i)gamma.p[i]=1+value(i,1.0f/512);
    D::gdn_sequence(&q,state.p,input.p,C,alpha.p,beta.p,z.p,gamma.p,1e-6f,out.p,HK,HV,T,true);
    D::gdn_sequence(&q,held.p,input.p,C,alpha.p,beta.p,z.p,gamma.p,1e-6f,oh.p,HK,HV,T,false);
    for(int t=0;t<T;++t)D::gdn_sequence(&q,seq.p,input.p+size_t(t)*C,C,alpha.p+t*HV,beta.p+t*HV,z.p+size_t(t)*HV*128,gamma.p,1e-6f,os.p+size_t(t)*HV*128,HK,HV,1,true);
    D::gdn_sequence_columns(&q,cols.p,input.p,C,alpha.p,beta.p,z.p,gamma.p,1e-6f,oc.p,HK,HV,T);
    q.wait_and_throw();bits(state.p,seq.p,sn,"GDN window/step state");bits(out.p,os.p,yn,"GDN window/step output");bits(out.p,oh.p,yn,"GDN verify output");for(size_t i=0;i<sn;++i)require(held.p[i]==float(goldstate[i]),"GDN write=false changed state");
    gdn_oracle(input.p,C,alpha.p,beta.p,z.p,gamma.p,HK,HV,T,goldstate,gold);near(state.p,goldstate,"GDN state HK="+std::to_string(HK)+" HV="+std::to_string(HV)+" T="+std::to_string(T));near(out.p,gold,"GDN output");near(cols.p,goldstate,"GDN columns state");near(oc.p,gold,"GDN columns output");
}
void driver_conv(sycl::queue& q,int T){const int HK=2,HV=4,C=(2*HK+HV)*128;Buf<float>hist(q,C*3),seq(q,C*3),held(q,C*3),input(q,T*C),w(q,C*4),out(q,T*C),os(q,T*C),oh(q,T*C);std::vector<double>hs(C*3),gold(size_t(T)*C);
    for(int i=0;i<C*3;++i)hist.p[i]=seq.p[i]=held.p[i]=float(hs[i]=value(i,1.0f/1024));for(int i=0;i<T*C;++i)input.p[i]=value(i+7,1.0f/512);for(int i=0;i<C*4;++i)w.p[i]=value(i+11,1.0f/1024);
    D::conv_sequence(&q,hist.p,input.p,w.p,out.p,C,2*HK,1e-6f,T,true);D::conv_sequence(&q,held.p,input.p,w.p,oh.p,C,2*HK,1e-6f,T,false);for(int t=0;t<T;++t)D::conv_sequence(&q,seq.p,input.p+t*C,w.p,os.p+t*C,C,2*HK,1e-6f,1,true);q.wait_and_throw();bits(hist.p,seq.p,C*3,"conv window/step history");bits(out.p,os.p,T*C,"conv window/step output");bits(out.p,oh.p,T*C,"conv verify output");for(int i=0;i<C*3;++i)require(held.p[i]==float(hs[i]),"conv verify changed history");
    for(int t=0;t<T;++t){for(int c=0;c<C;++c){double v=hs[c*3]*w.p[c*4]+hs[c*3+1]*w.p[c*4+1]+hs[c*3+2]*w.p[c*4+2]+input.p[t*C+c]*double(w.p[c*4+3]);hs[c*3]=hs[c*3+1];hs[c*3+1]=hs[c*3+2];hs[c*3+2]=input.p[t*C+c];gold[t*C+c]=v/(1+std::exp(-v));}for(int h=0;h<2*HK;++h){double ss=1e-6;for(int c=0;c<128;++c)ss+=gold[t*C+h*128+c]*gold[t*C+h*128+c];for(int c=0;c<128;++c)gold[t*C+h*128+c]/=std::sqrt(ss);}}
    near(out.p,gold,"conv T="+std::to_string(T));near(hist.p,hs,"conv history");
}
void driver_prep(sycl::queue& q,bool vision){const int T=3,H=16,HK=2,HD=256,ctx=16,pos=5;const size_t nq=size_t(T)*H*HD,nk=size_t(HK)*ctx*HD;Buf<float>qg(q,nq*2),kx(q,T*HK*HD),vx(q,T*HK*HD),qw(q,HD),kw(q,HD),query(q,nq),gate(q,nq),kc(q,nk),vc(q,nk);Buf<int32_t>pp(q,1),delta(q,1),tab(q,ctx*3);*pp.p=pos;*delta.p=7;
    for(size_t i=0;i<nq*2;++i)qg.p[i]=value(i,1.0f/256);for(int i=0;i<T*HK*HD;++i){kx.p[i]=value(i+17,1.0f/256);vx.p[i]=value(i+31,1.0f/256);}for(int i=0;i<HD;++i){qw.p[i]=1+value(i,1.0f/1024);kw.p[i]=1+value(i+41,1.0f/1024);}for(int p=0;p<ctx;++p){tab.p[p*3]=p*2;tab.p[p*3+1]=p+11;tab.p[p*3+2]=p+23;}std::fill(kc.p,kc.p+nk,-999);std::fill(vc.p,vc.p+nk,-999);
    D::attention_prepare(&q,qg.p,kx.p,vx.p,qw.p,kw.p,query.p,gate.p,kc.p,vc.p,T,H,HK,pp.p,ctx,10000000.0f,1e-6f,delta.p,vision?tab.p:nullptr);q.wait_and_throw();std::vector<double>gq(nq),gk(nk,-999),gv(nk,-999);
    for(int t=0;t<T;++t)for(int h=0;h<H+HK;++h){bool isq=h<H;const float*src=isq?qg.p+(size_t(t)*H+h)*2*HD:kx.p+(size_t(t)*HK+h-H)*HD;const float*wn=isq?qw.p:kw.p;double norm=0;for(int d=0;d<HD;++d)norm+=double(src[d])*src[d];norm=1/std::sqrt(norm/HD+1e-6);double buf[HD];for(int d=0;d<HD;++d)buf[d]=src[d]*norm*wn[d];
        for(int d=0;d<HD;++d){double v=buf[d];if(d<64){int pair=d<32?d:d-32,rp=vision?tab.p[(pos+t)*3+pair%3]:pos+t+7;double a=rp*std::pow(10000000.0,-2.0*pair/64);v=d<32?buf[d]*std::cos(a)-buf[d+32]*std::sin(a):buf[d-32]*std::sin(a)+buf[d]*std::cos(a);}if(isq){gq[(size_t(t)*H+h)*HD+d]=v;require(gate.p[(size_t(t)*H+h)*HD+d]==qg.p[(size_t(t)*H+h)*2*HD+HD+d],"attention prep gate copy");}else{size_t i=(size_t(h-H)*ctx+pos+t)*HD+d;gk[i]=v;gv[i]=vx.p[(size_t(t)*HK+h-H)*HD+d];}}}
    near(query.p,gq,vision?"M-RoPE query":"text RoPE query");near(kc.p,gk,vision?"M-RoPE key cache":"text RoPE key cache");near(vc.p,gv,"attention V cache",0,0);
}
void driver_attention(sycl::queue& q,int T,int pos){const int H=16,HK=2,HD=256,ctx=pos+T+1,chunks=(ctx+255)/256;const size_t n=size_t(T)*H*HD,kn=size_t(HK)*ctx*HD;Buf<float>query(q,n),k(q,kn),v(q,kn),gate(q,n),out(q,n),seq(q,n),part(q,size_t(T)*H*chunks*HD),pm(q,T*H*chunks),pl(q,T*H*chunks);Buf<int32_t>pp(q,T);for(size_t i=0;i<n;++i){query.p[i]=value(i,1.0f/256);gate.p[i]=value(i+17,1.0f/128);}for(size_t i=0;i<kn;++i){k.p[i]=value(i+37,1.0f/256);v.p[i]=value(i+71,1.0f/256);}for(int t=0;t<T;++t)pp.p[t]=pos+t;
    D::attention_split(&q,query.p,k.p,v.p,gate.p,pp.p,T,H,HK,ctx,part.p,pm.p,pl.p,out.p);for(int t=0;t<T;++t)D::attention_split(&q,query.p+size_t(t)*H*HD,k.p,v.p,gate.p+size_t(t)*H*HD,pp.p+t,1,H,HK,ctx,part.p,pm.p,pl.p,seq.p+size_t(t)*H*HD);q.wait_and_throw();bits(out.p,seq.p,n,"attention window/step");
    std::vector<double>gold(n);for(int t=0;t<T;++t)for(int h=0;h<H;++h){int kv=h/(H/HK),npos=pos+t+1;std::vector<double>scores(npos);double mx=-INFINITY;for(int p=0;p<npos;++p){double s=0;for(int d=0;d<HD;++d)s+=double(query.p[(size_t(t)*H+h)*HD+d])*k.p[(size_t(kv)*ctx+p)*HD+d];mx=std::max(mx,scores[p]=s/16);}double sum=0;for(double& s:scores)sum+=s=std::exp(s-mx);for(int d=0;d<HD;++d){double num=0;for(int p=0;p<npos;++p)num+=scores[p]*v.p[(size_t(kv)*ctx+p)*HD+d];size_t i=(size_t(t)*H+h)*HD+d;gold[i]=num/sum/(1+std::exp(-double(gate.p[i])));}}
    near(out.p,gold,"attention H16/HKV2 T="+std::to_string(T)+" pos="+std::to_string(pos));
}
#endif
void gpu_checks(){sycl::queue q(sycl::gpu_selector_v,[](sycl::exception_list es){for(auto& e:es)std::rethrow_exception(e);},sycl::property::queue::in_order{});auto d=q.get_device();auto sg=d.get_info<sycl::info::device::sub_group_sizes>();require(d.has(sycl::aspect::fp16)&&std::find(sg.begin(),sg.end(),32)!=sg.end()&&d.get_info<sycl::info::device::max_work_group_size>()>=1024,"probe requires fp16, subgroup32, workgroup1024 GPU");std::cout<<"GPU: "<<d.get_info<sycl::info::device::name>()<<"\n";
    quant_checks(q);rows_case<1,1>(q,1,17);rows_case<2,1>(q,8,65);rows_case<4,2>(q,33,17);rows_case<8,2>(q,64,33);rows_case<1,1>(q,1,1033);shared_case<1,1,false>(q,8);shared_case<4,2,true>(q,64);
    for(int T:{1,8})for(int n:{256,2048}){epilogue_case(q,n,T,false,true,false);epilogue_case(q,n,T,true,true,true);}epilogue_case(q,2048,3,true,false,false);group_checks(q);
#ifdef STRATA_Q36_TEST_DRIVER_HOOKS
    driver_prep(q,false);driver_prep(q,true);for(int T:{1,3,8}){driver_conv(q,T);driver_gdn(q,2,4,T);driver_attention(q,T,255);}driver_conv(q,17);driver_conv(q,65);driver_gdn(q,16,32,3);driver_attention(q,1,0);driver_attention(q,3,256);
#else
    std::cout<<"Driver hooks disabled: conv/GDN/attention NOT tested in this executable\n";
#endif
    std::cout<<"All enabled GPU kernel parity checks passed\n";
}
}
int main(int argc,char**argv){try{cpu_checks();if(argc==2&&std::string(argv[1])=="--cpu-only")return 0;require(argc==1,"usage: q36-kernel-parity [--cpu-only]");gpu_checks();return 0;}catch(const std::exception& e){std::cerr<<"FAIL: "<<e.what()<<"\n";return 1;}}
