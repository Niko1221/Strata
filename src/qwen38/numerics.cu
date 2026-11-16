// Independent FP64 host references for the production CUDA operations.
// Tolerances are fixed before running: see check_close() call sites and README.md.
#include "ops.cuh"
#include "resources.hpp"
#include "strata/kernels/fused_gdn.hpp"
#include <vector>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <algorithm>
namespace Q=strata::qwen38;
namespace K=strata::kernels;
static void demand(bool b,const char* m){if(!b)throw std::runtime_error(m);}
static void admission(){Q::host_resources();size_t f,t;Q::ck(cudaMemGetInfo(&f,&t));demand(f>Q::GPU_RESERVE+64*1024*1024,"fixture CUDA reserve");}
static double hf(uint16_t u){int sign=(u>>15)?-1:1,ex=(u>>10)&31,frac=u&1023;if(!ex)return sign*std::ldexp(double(frac),-24);if(ex==31)return frac?NAN:sign*INFINITY;return sign*std::ldexp(double(1024+frac),ex-25);}
static uint16_t bits(float x){return __half_as_ushort(__float2half(x));}
static double loadh(const uint8_t* p){return hf(uint16_t(p[0])|(uint16_t(p[1])<<8));}
template<class T>static Q::Buffer upload(const std::vector<T>& v){Q::Buffer b(v.size()*sizeof(T));Q::ck(cudaMemcpy(b.data,v.data(),b.bytes,cudaMemcpyHostToDevice));return b;}
static std::vector<float> read(const Q::Buffer& b){std::vector<float> v(b.bytes/4);Q::ck(cudaMemcpy(v.data(),b.data,b.bytes,cudaMemcpyDeviceToHost));return v;}
static void check_close(const std::vector<float>& got,const std::vector<double>& expected,double atol,double rtol,const char* name){
    demand(got.size()==expected.size(),"test shape mismatch");double mx=0;for(size_t i=0;i<got.size();++i){double err=std::abs(double(got[i])-expected[i]);mx=std::max(mx,err);if(!std::isfinite(got[i])||err>atol+rtol*std::abs(expected[i])){std::fprintf(stderr,"%s index=%zu got=%.9g expected=%.12g error=%g\n",name,i,got[i],expected[i],err);throw std::runtime_error("numeric tolerance exceeded");}}
    std::printf("PASS %s n=%zu max_abs=%.9g atol=%g rtol=%g\n",name,got.size(),mx,atol,rtol);
}
// CPU byte decoder transcribed from immutable Prism0.4.1 ggml-quants.c.
// Separate block loops intentionally differ from the CUDA scalar index paths.
static std::vector<double> reference(int ty,const std::vector<uint8_t>& bytes,int n){
    std::vector<double> y(n);const uint8_t* p=bytes.data();
    if(ty==0||ty==1||ty==30){for(int i=0;i<n;++i){if(ty==0){float f;std::memcpy(&f,p+4*i,4);y[i]=f;}else{uint16_t u=uint16_t(p[2*i])|uint16_t(p[2*i+1])<<8;if(ty==1)y[i]=hf(u);else{uint32_t v=uint32_t(u)<<16;float f;std::memcpy(&f,&v,4);y[i]=f;}}}return y;}
    int block=ty==142?128:ty==8?32:256,sz=ty==142||ty==8?34:ty==23?136:ty==12?144:ty==13?176:210;
    for(int b=0;b<n/block;++b,p+=sz){double* out=y.data()+b*block;double d=loadh(p);
        if(ty==142){for(int j=0;j<128;++j){int code=(p[2+j/4]>>(2*(j%4)))&3;out[j]=(code-1)*d;}}
        else if(ty==8){for(int j=0;j<32;++j)out[j]=d*int8_t(p[2+j]);}
        else if(ty==23){const int table[]={-127,-104,-83,-65,-49,-35,-22,-10,1,13,25,38,53,69,89,113};uint16_t hi=p[2]|uint16_t(p[3])<<8;
            for(int ib=0;ib<8;++ib){int sc=((p[4+ib/2]>>(4*(ib%2)))&15)|(((hi>>(2*ib))&3)<<4);for(int j=0;j<16;++j){uint8_t q=p[8+ib*16+j];out[ib*32+j]=d*(sc-32)*table[q&15];out[ib*32+j+16]=d*(sc-32)*table[q>>4];}}}
        else if(ty==12||ty==13){const uint8_t* sc=p+4;const uint8_t* qs=p+(ty==13?48:16);double dm=loadh(p+2);for(int group=0;group<4;++group)for(int half=0;half<2;++half){int i=group*2+half;int scale=i<4?sc[i]&63:(sc[i+4]&15)|((sc[i-4]>>6)<<4);int min=i<4?sc[i+4]&63:(sc[i+4]>>4)|((sc[i]>>6)<<4);
            for(int j=0;j<32;++j){int code=half?qs[group*32+j]>>4:qs[group*32+j]&15;if(ty==13&&(p[16+j]&(1<<i)))code+=16;out[i*32+j]=d*scale*code-dm*min;}}}
        else {double scale=loadh(p+208);for(int ip=0;ip<2;++ip)for(int l=0;l<32;++l){const uint8_t* ql=p+ip*64;uint8_t qh=p[128+ip*32+l];const int8_t* sc=reinterpret_cast<const int8_t*>(p+192)+ip*8;
            out[ip*128+l]=scale*sc[l/16]*(((ql[l]&15)|((qh&3)<<4))-32);
            out[ip*128+l+32]=scale*sc[l/16+2]*(((ql[l+32]&15)|(((qh>>2)&3)<<4))-32);
            out[ip*128+l+64]=scale*sc[l/16+4]*(((ql[l]>>4)|(((qh>>4)&3)<<4))-32);
            out[ip*128+l+96]=scale*sc[l/16+6]*(((ql[l+32]>>4)|(((qh>>6)&3)<<4))-32);}}
    }return y;
}
static void packed(cudaStream_t s){admission();constexpr int n=512,rows=7;
    for(int ty:{142,23,12,13,14,8,0,1,30}){
        int block=ty==142?128:ty==8?32:ty==0||ty==1||ty==30?1:256;
        int size=ty==142||ty==8?34:ty==23?136:ty==12?144:ty==13?176:ty==14?210:ty==0?4:2;
        std::vector<uint8_t> bytes(n*rows/block*size);for(size_t i=0;i<bytes.size();++i)bytes[i]=uint8_t(i*73+19);
        auto seth=[&](int offset,float x){uint16_t h=bits(x);bytes[offset]=uint8_t(h);bytes[offset+1]=uint8_t(h>>8);};
        if(block>1){for(int b=0;b<n*rows/block;++b){seth(b*size+(ty==14?208:0),b%3==0?0.f:b%3==1?0.03125f:-0.0625f);if(ty==12||ty==13)seth(b*size+2,0.015625f);}}
        else for(int i=0;i<n*rows;++i){float v=float(std::sin(i*0.7));if(ty==0)std::memcpy(bytes.data()+i*4,&v,4);else if(ty==1)seth(i*2,v);else{uint32_t u;std::memcpy(&u,&v,4);bytes[i*2]=uint8_t(u>>16);bytes[i*2+1]=uint8_t(u>>24);}}
        auto ref=reference(ty,bytes,n*rows);auto w=upload(bytes);Q::Buffer out(n*rows*4);Q::decode(ty,w.data,out.as(),n*rows,s);Q::ck(cudaStreamSynchronize(s));
        check_close(read(out),ref,0,0,("decode_type_"+std::to_string(ty)).c_str());
        std::vector<float> x(n);for(int i=0;i<n;++i)x[i]=float(std::sin(i*0.11)*0.2);auto dx=upload(x);Q::Buffer dy(rows*4);std::vector<double> y(rows);
        for(int r=0;r<rows;++r)for(int i=0;i<n;++i)y[r]+=ref[r*n+i]*x[i];
        Q::matvec(ty,w.data,dx.as(),dy.as(),n,rows,s);Q::ck(cudaStreamSynchronize(s));check_close(read(dy),y,2e-4,2e-5,("matvec_type_"+std::to_string(ty)).c_str());
    }
    Q::Buffer b(4096);for(int ty:{142,23,13,14}){bool refused=false;try{Q::matvec(ty,b.data,b.as(),b.as(),129,1,s);}catch(const std::exception&){refused=true;}demand(refused,"bad shape accepted");}
    bool refused=false;try{Q::matvec(142,b.data,b.as(),b.as(),128,0,s);}catch(...){refused=true;}demand(refused,"zero output shape accepted");
    std::puts("PASS packed invalid input/output shapes");
}
static double sig(double x){return 1/(1+std::exp(-x));}
static void hadamard(cudaStream_t s){admission();
    for(int n:{5120,6144,17408})for(int mode=0;mode<3;++mode){if(mode==2&&n!=6144)continue;
        std::vector<float> x(n),signs(n);for(int i=0;i<n;++i){x[i]=float(std::sin(i*.17)*.7);signs[i]=(i*13%7)<3?-1.f:1.f;}auto dx=upload(x),ds=upload(signs);Q::Buffer dy(n*4);
        Q::hadamard(dx.as(),ds.as(),dy.as(),n,mode==1,mode==2,s);std::vector<double> want(n);
        for(int i=0;i<n;++i)for(int j=0;j<1024;++j){int col=i/1024*1024+j,src=col;if(mode==2){int h=col/128;src=((h%3)*16+h/3)*128+col%128;}
            unsigned parity=(i%1024)&j;parity^=parity>>8;parity^=parity>>4;parity^=parity>>2;parity^=parity>>1;
            want[i]+=(parity&1?-1.:1.)/32*x[src]*(mode==1?signs[i]:signs[col]);}
        Q::ck(cudaStreamSynchronize(s));check_close(read(dy),want,3e-6,2e-5,("Hadamard_width_"+std::to_string(n)+"_mode_"+std::to_string(mode)).c_str());
    }
}
static void gdn(cudaStream_t s){admission();constexpr int C=10240,HV=48,HK=16,D=128;std::vector<double> hist(C*3),state(D*HV*D);std::vector<float> conv(C*4),gamma(D),dt(HV),a(HV);
    for(size_t i=0;i<conv.size();++i)conv[i]=float(std::sin(i*.17)*.35);for(int i=0;i<D;++i)gamma[i]=float(0.8+std::sin(i*.3)*.2);for(int i=0;i<HV;++i){dt[i]=float(std::cos(i)*.2);a[i]=float(-.1-.02*(i%7));}
    auto cw=upload(conv),gw=upload(gamma),dw=upload(dt),aw=upload(a);Q::Buffer dh(C*3*4),ds(D*HV*D*4),hc(C*4),dy(HV*D*4);dh.zero(s);ds.zero(s);
    for(int t=0;t<6;++t){std::vector<float> qkv(C),z(HV*D),al(HV),be(HV);for(int i=0;i<C;++i)qkv[i]=float(std::sin(i*.13+t*.37)*.7);for(int i=0;i<HV*D;++i)z[i]=float(std::cos(i*.07+t)*1.5);for(int i=0;i<HV;++i){al[i]=float(std::sin(i*.31+t));be[i]=float(std::cos(i*.29-t));}
        auto dq=upload(qkv),dz=upload(z),da=upload(al),db=upload(be);Q::gdn_gates(da.as(),db.as(),dw.as(),aw.as(),HV,s);K::fused_gdn_conv_l2(dh.as(),dq.as(),cw.as(),hc.as(),C,HK*2,1e-6f,s);K::fused_gdn_step_norm_silu(ds.as(),hc.as(),hc.as()+HK*D,hc.as()+2*HK*D,da.as(),db.as(),dz.as(),gw.as(),1e-6f,dy.as(),HK,HV,s);
        std::vector<double> h(C),y(HV*D);for(int i=0;i<C;++i){double u=hist[3*i]*conv[4*i]+hist[3*i+1]*conv[4*i+1]+hist[3*i+2]*conv[4*i+2]+qkv[i]*conv[4*i+3];hist[3*i]=hist[3*i+1];hist[3*i+1]=hist[3*i+2];hist[3*i+2]=qkv[i];h[i]=u*sig(u);}
        for(int head=0;head<HK*2;++head){double ss=1e-6;for(int j=0;j<D;++j)ss+=h[head*D+j]*h[head*D+j];for(int j=0;j<D;++j)h[head*D+j]/=std::sqrt(ss);}
        for(int head=0;head<HV;++head){int qh=head%HK;double decay=std::exp(std::log1p(std::exp(double(al[head])+dt[head]))*a[head]),beta=sig(be[head]);
            for(int col=0;col<D;++col){double pred=0;for(int row=0;row<D;++row)pred+=decay*state[(row*HV+head)*D+col]*h[HK*D+qh*D+row];double delta=beta*(h[2*HK*D+head*D+col]-pred),val=0;
                for(int row=0;row<D;++row){auto& st=state[(row*HV+head)*D+col];st=decay*st+h[HK*D+qh*D+row]*delta;val+=st*h[qh*D+row];}y[head*D+col]=val/std::sqrt(double(D));}
            double ss=0;for(int col=0;col<D;++col)ss+=y[head*D+col]*y[head*D+col];double norm=std::sqrt(ss/D+1e-6);for(int col=0;col<D;++col)y[head*D+col]=y[head*D+col]/norm*gamma[col]*z[head*D+col]*sig(z[head*D+col]);}
        Q::ck(cudaStreamSynchronize(s));check_close(read(dy),y,5e-5,5e-4,("GDN_full_sequence_"+std::to_string(t)).c_str());check_close(read(ds),state,5e-6,5e-4,"GDN_state_FP32");
    }
    ds.zero(s);dh.zero(s);Q::ck(cudaStreamSynchronize(s));for(float v:read(ds))demand(v==0,"GDN reset");std::puts("PASS GDN reset");
}
static void attention(cudaStream_t s){admission();constexpr int ctx=9,H=24,KV=4,D=256;std::vector<float> qw(D),kw(D);for(int i=0;i<D;++i){qw[i]=float(.9+.1*std::cos(i));kw[i]=float(1.1+.1*std::sin(i));}auto dqw=upload(qw),dkw=upload(kw);Q::Buffer kc(KV*ctx*D*2),vc(KV*ctx*D*2),q(H*D*4),gate(H*D*4),out(H*D*4),scores(H*ctx*4);kc.zero(s);vc.zero(s);std::vector<double> keys(KV*ctx*D),vals(KV*ctx*D);
    for(int pos=0;pos<ctx;++pos){std::vector<float> qg(H*D*2),k(KV*D),v(KV*D);for(size_t i=0;i<qg.size();++i)qg[i]=float(std::sin(i*.07+pos*.6));for(int i=0;i<KV*D;++i){k[i]=float(std::cos(i*.13+pos));v[i]=float(std::sin(i*.17-pos*.9));}auto dqg=upload(qg),dk=upload(k),dv=upload(v);Q::attention_prepare(dqg.as(),dk.as(),dv.as(),dqw.as(),dkw.as(),q.as(),gate.as(),kc.as<half>(),vc.as<half>(),pos,ctx,1e-6f,1e7f,s);Q::attention_apply(q.as(),gate.as(),kc.as<half>(),vc.as<half>(),out.as(),scores.as(),pos,ctx,s);
        std::vector<double> query(H*D),want(H*D);
        for(int head=0;head<H+KV;++head){bool isq=head<H;const float* x=isq?qg.data()+head*512:k.data()+(head-H)*D;double ss=0;for(int i=0;i<D;++i)ss+=double(x[i])*x[i];double scale=1/std::sqrt(ss/D+1e-6);std::vector<double> r(D);for(int i=0;i<D;++i)r[i]=x[i]*scale*(isq?qw[i]:kw[i]);
            for(int i=0;i<32;++i){double theta=pos*std::pow(1e7,-double(i)/32),u=r[i],w=r[i+32];r[i]=u*std::cos(theta)-w*std::sin(theta);r[i+32]=u*std::sin(theta)+w*std::cos(theta);}
            for(int i=0;i<D;++i)if(isq)query[head*D+i]=r[i];else{keys[((head-H)*ctx+pos)*D+i]=hf(bits(float(r[i])));vals[((head-H)*ctx+pos)*D+i]=hf(bits(v[(head-H)*D+i]));}}
        for(int head=0;head<H;++head){std::vector<double> p(pos+1);for(int t=0;t<=pos;++t)for(int i=0;i<D;++i)p[t]+=query[head*D+i]*keys[((head/6)*ctx+t)*D+i]/16;double mx=*std::max_element(p.begin(),p.end()),sum=0;for(auto& z:p)sum+=(z=std::exp(z-mx));for(int i=0;i<D;++i){double y=0;for(int t=0;t<=pos;++t)y+=p[t]/sum*vals[((head/6)*ctx+t)*D+i];want[head*D+i]=y*sig(qg[head*512+D+i]);}}
        Q::ck(cudaStreamSynchronize(s));check_close(read(q),query,2e-5,2e-5,"partial_RoPE_Q");check_close(read(out),want,4e-4,3e-4,("GQA_gated_sequence_"+std::to_string(pos)).c_str());
    }
    bool rejected=false;try{Q::attention_apply(q.as(),gate.as(),kc.as<half>(),vc.as<half>(),out.as(),scores.as(),ctx,ctx,s);}catch(...){rejected=true;}demand(rejected,"context overrun accepted");kc.zero(s);vc.zero(s);Q::ck(cudaStreamSynchronize(s));std::puts("PASS attention capacity/reset");
}
static void failures(cudaStream_t s){admission();std::vector<float> x={0,INFINITY,NAN,-INFINITY};auto b=upload(x);Q::Buffer e(4);e.zero(s);Q::finite(b.as(),x.size(),e.as<int>(),s);int flag=0;Q::ck(cudaMemcpy(&flag,e.data,4,cudaMemcpyDeviceToHost));demand(flag==1,"nonfinite not caught");std::puts("PASS nonfinite failure detection");}
int main(){try{Q::host_resources();Q::ck(cudaSetDevice(0));cudaStream_t s;Q::ck(cudaStreamCreate(&s));try{packed(s);hadamard(s);gdn(s);attention(s);failures(s);}catch(...){cudaStreamDestroy(s);throw;}Q::ck(cudaStreamDestroy(s));std::puts("PASS production CUDA numerical fixtures");return 0;}catch(const std::exception& e){std::fprintf(stderr,"FAIL %s\n",e.what());return 1;}}
