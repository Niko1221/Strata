// Independent FP64 host references for the production CUDA operations.
// Tolerances are fixed before running: see check_close() call sites and README.md.
#define Q38_IQ_HOST_TABLE
#include "ops.cuh"
#include "iq_tables.hpp"
#include "iq_decode.hpp"
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

// Exercise the multi-token shared-memory path of the fused GDN conv/L2 kernel.
static void gdn_batch(cudaStream_t s){
    admission();
    constexpr int C=10240, QK2=32, T=6;
    std::vector<float> conv(C*4), qkv(size_t(C)*T);
    for(size_t i=0;i<conv.size();++i)conv[i]=float(std::sin(i*.17)*.35);
    for(size_t i=0;i<qkv.size();++i)qkv[i]=float(std::sin(i*.13)*.7);
    auto dcw=upload(conv), dq=upload(qkv);
    Q::Buffer hb_seq(size_t(C)*T*4), hb_bat(size_t(C)*T*4);
    Q::Buffer hst_seq(C*3*4), hst_bat(C*3*4);
    hst_seq.zero(s); hst_bat.zero(s);
    // Sequential reference: one single-token kernel per token.
    for(int t=0;t<T;++t)
        K::fused_gdn_conv_l2(hst_seq.as(), dq.as()+size_t(t)*C, dcw.as(),
                             hb_seq.as()+size_t(t)*C, C, QK2, 1e-6f, s);
    // Batch path under test: one kernel, shared part[] reused across tokens.
    K::fused_gdn_conv_l2_batch(hst_bat.as(), dq.as(), dcw.as(), hb_bat.as(),
                               C, QK2, 1e-6f, T, s);
    Q::ck(cudaStreamSynchronize(s));
    const auto seq_out = read(hb_seq);
    const auto seq_hist = read(hst_seq);
    check_close(read(hb_bat), std::vector<double>(seq_out.begin(), seq_out.end()),
                3e-5, 3e-4, "GDN_conv_l2_batch_output_T6");
    check_close(read(hst_bat), std::vector<double>(seq_hist.begin(), seq_hist.end()),
                0, 0, "GDN_conv_l2_batch_history_T6");
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
static std::vector<double> reference_ext(int ty, const std::vector<uint8_t>& bytes, int n) {
    if (ty == 21 || ty == 22) {
        std::vector<double> y(n);
        const int sz = (ty == 21) ? 110 : 82;
        for (int b = 0; b < n / 256; ++b) {
            const uint8_t* p = bytes.data() + size_t(b) * sz;
            const float d = float(loadh(p));
            for (int j = 0; j < 256; ++j) {
                y[size_t(b) * 256 + j] = Q::iq_value(ty, p, j, d, q38_iq2s_grid, q38_iq3s_grid);
            }
        }
        return y;
    }
    return reference(ty, bytes, n);
}

static std::vector<Q::BlockQ8_1> read_q8(const Q::Buffer& b) {
    std::vector<Q::BlockQ8_1> v(b.bytes / sizeof(Q::BlockQ8_1));
    Q::ck(cudaMemcpy(v.data(), b.data, b.bytes, cudaMemcpyDeviceToHost));
    return v;
}

static void check_q8_blocks_equal(const std::vector<Q::BlockQ8_1>& got, const std::vector<Q::BlockQ8_1>& want, const char* name) {
    demand(got.size() == want.size(), "q8 block count mismatch");
    for (size_t i = 0; i < got.size(); ++i) {
        demand(got[i].d == want[i].d && got[i].s == want[i].s, name);
        for (int k = 0; k < 8; ++k) demand(got[i].qs[k] == want[i].qs[k], name);
    }
}

static std::vector<double> dequant_q8_1_host(const std::vector<Q::BlockQ8_1>& blks) {
    std::vector<double> out(blks.size() * 32);
    for (size_t b = 0; b < blks.size(); ++b) {
        const double d = blks[b].d;
        for (int k = 0; k < 8; ++k) {
            const uint32_t w = uint32_t(blks[b].qs[k]);
            for (int j = 0; j < 4; ++j) {
                out[b * 32 + k * 4 + j] = d * double(int8_t((w >> (8 * j)) & 0xFFu));
            }
        }
    }
    return out;
}

static std::vector<uint8_t> make_packed_weights(int ty, int n, int rows, uint32_t seed_add) {
    const int block = (ty == 142) ? 128 : (ty == 8) ? 32 : 256;
    const int size = (ty == 142 || ty == 8) ? 34 : (ty == 12) ? 144 : (ty == 13) ? 176 : (ty == 14) ? 210 : (ty == 21) ? 110 : 82;
    std::vector<uint8_t> bytes(size_t(n) * rows / block * size);
    uint32_t st = 0x5A17C3u + seed_add;
    for (size_t i = 0; i < bytes.size(); ++i) {
        st = 1664525u * st + 1013904223u;
        bytes[i] = uint8_t(st >> 24);
    }
    auto seth = [&](size_t offset, float x) {
        uint16_t h = bits(x);
        bytes[offset] = uint8_t(h);
        bytes[offset + 1] = uint8_t(h >> 8);
    };
    const size_t nblk = size_t(n) * rows / block;
    for (size_t b = 0; b < nblk; ++b) {
        const float sc = (b % 5 == 0) ? 0.f : (b % 2 == 0) ? 0.015625f : -0.03125f;
        // Synthetic weights must stay near real-model amplitude so the Q8_1
        // input quantisation error fits the 6e-3 vs_fp64 budget.  The strict
        // _exact_q8 tolerances are unchanged.
        const float dscale = (ty == 8) ? 0.03125f : 0.015625f;
        seth(b * size + (ty == 14 ? 208 : 0), sc * dscale);
        if (ty == 12 || ty == 13) {
            seth(b * size + 2, 0.0078125f * dscale);
            for (int j = 0; j < 8; ++j) bytes[b * size + 4 + j] = 1;
        }
        if (ty == 14) {
            for (int j = 0; j < 16; ++j) bytes[b * size + 192 + j] = 1;
        }
    }
    return bytes;
}

static void q8_1_ops(cudaStream_t s) {
    admission();
    // 1. rms_quantize_q8_1
    for (int n : {512, 5120, 17408}) {
        std::vector<float> x(n), w(n);
        for (int i = 0; i < n; ++i) {
            x[i] = float(std::sin(i * 0.13 + 0.2) * 0.8);
            w[i] = float(0.85 + 0.25 * std::cos(i * 0.07));
        }
        auto dx = upload(x), dw = upload(w);
        Q::Buffer out_f32(size_t(n) * 4), ref_f32(size_t(n) * 4);
        Q::Buffer out_q8(size_t(n / 32) * sizeof(Q::BlockQ8_1)), ref_q8(size_t(n / 32) * sizeof(Q::BlockQ8_1));
        Q::rms_quantize_q8_1(dx.as(), dw.as(), out_f32.as(), out_q8.as<Q::BlockQ8_1>(), n, 1e-6f, s);
        Q::rms(dx.as(), dw.as(), ref_f32.as(), n, 1e-6f, s);
        Q::quantize_q8_1(ref_f32.as(), ref_q8.as<Q::BlockQ8_1>(), n, s);
        Q::ck(cudaStreamSynchronize(s));
        double ss = 0;
        for (int i = 0; i < n; ++i) ss += double(x[i]) * x[i];
        const double inv = 1.0 / std::sqrt(ss / n + 1e-6);
        std::vector<double> want(n);
        for (int i = 0; i < n; ++i) want[i] = x[i] * inv * w[i];
        check_close(read(out_f32), want, 1e-6, 1e-5, ("rms_quantize_f32_" + std::to_string(n)).c_str());
        auto ref_host = read(ref_f32);
        std::vector<double> ref_d(ref_host.begin(), ref_host.end());
        check_close(read(out_f32), ref_d, 0, 0, ("rms_quantize_exact_rms_" + std::to_string(n)).c_str());
        check_q8_blocks_equal(read_q8(out_q8), read_q8(ref_q8), "rms_quantize_q8_blocks");
    }
    // 2. swiglu_quantize_q8_1
    for (int n : {512, 17408}) {
        std::vector<float> gate(n), up(n);
        for (int i = 0; i < n; ++i) {
            gate[i] = float(std::sin(i * 0.19 - 0.4) * 1.3);
            up[i] = float(std::cos(i * 0.11 + 0.1) * 0.9);
        }
        auto dg = upload(gate), dg_ref = upload(gate), du = upload(up);
        Q::Buffer out_q8(size_t(n / 32) * sizeof(Q::BlockQ8_1)), ref_q8(size_t(n / 32) * sizeof(Q::BlockQ8_1));
        Q::swiglu_quantize_q8_1(dg.as(), du.as(), out_q8.as<Q::BlockQ8_1>(), n, s);
        Q::swiglu(dg_ref.as(), du.as(), n, s);
        Q::quantize_q8_1(dg_ref.as(), ref_q8.as<Q::BlockQ8_1>(), n, s);
        Q::ck(cudaStreamSynchronize(s));
        std::vector<double> want(n);
        for (int i = 0; i < n; ++i) want[i] = (double(gate[i]) * sig(gate[i])) * double(up[i]);
        check_close(read(dg), want, 1e-6, 1e-5, ("swiglu_quantize_f32_" + std::to_string(n)).c_str());
        check_q8_blocks_equal(read_q8(out_q8), read_q8(ref_q8), "swiglu_quantize_q8_blocks");
    }
    // 3. matvec_q8_1 across types {8, 12, 13, 14, 21, 22, 142}, 4-block remainders (K=256,512,768),
    // real widths (K=5120,6144,17408), small/odd rows (1,7,17,48,257,520,521), and exact allocation end bounds
    struct ShapeCase { int n; int rows; };
    const ShapeCase shapes[] = {
        {256, 1}, {512, 7}, {768, 17}, {512, 48},
        {256, 521}, {512, 520}, {768, 257}, {1024, 521},
        {5120, 17}, {5120, 257}, {6144, 7}, {17408, 1}
    };
    for (ShapeCase sc : shapes) {
        std::vector<float> x(sc.n);
        for (int i = 0; i < sc.n; ++i) x[i] = float(std::sin(i * 0.09 + 0.3) * 0.25);
        auto dx = upload(x);
        Q::Buffer dx_q8(size_t(sc.n / 32) * sizeof(Q::BlockQ8_1));
        Q::quantize_q8_1(dx.as(), dx_q8.as<Q::BlockQ8_1>(), sc.n, s);
        Q::ck(cudaStreamSynchronize(s));
        auto x_q8_deq = dequant_q8_1_host(read_q8(dx_q8));

        for (int ty : {8, 12, 13, 14, 21, 22, 142}) {
            auto bytes = make_packed_weights(ty, sc.n, sc.rows, uint32_t(ty * 31 + sc.rows + sc.n));
            auto ref_w = reference_ext(ty, bytes, sc.n * sc.rows);
            auto dw = upload(bytes); // exact byte size allocation to verify zero out-of-bounds reads
            Q::Buffer dy_q8(size_t(sc.rows) * 4), dy_f32(size_t(sc.rows) * 4);
            Q::matvec_q8_1(ty, dw.data, dx_q8.as<Q::BlockQ8_1>(), dx.as(), dy_q8.as(), sc.n, sc.rows, s);
            Q::matvec(ty, dw.data, dx.as(), dy_f32.as(), sc.n, sc.rows, s);
            Q::ck(cudaStreamSynchronize(s));
            std::vector<double> want_q8(sc.rows), want_f32(sc.rows);
            for (int r = 0; r < sc.rows; ++r) {
                for (int i = 0; i < sc.n; ++i) {
                    const double wv = ref_w[size_t(r) * sc.n + i];
                    want_q8[r] += wv * x_q8_deq[i];
                    want_f32[r] += wv * x[i];
                }
            }
            const std::string tag = "matvec_q8_1_t" + std::to_string(ty) + "_n" + std::to_string(sc.n) + "_r" + std::to_string(sc.rows);
            if (ty != 142) check_close(read(dy_q8), want_q8, 2e-4, 5e-5, (tag + "_exact_q8").c_str());
            check_close(read(dy_q8), want_f32, 6e-3, 6e-3, (tag + "_vs_fp64").c_str());
            auto f32_out = read(dy_f32);
            std::vector<double> f32_d(f32_out.begin(), f32_out.end());
            check_close(read(dy_q8), f32_d, 6e-3, 6e-3, (tag + "_vs_matvec").c_str());
        }
    }
    // Q8_0 1..3 block remainders (K=32, 64, 96 -> nb=1, 2, 3) at both 1row (r=17) and 8rows (r=257)
    for (int n_q8 : {32, 64, 96}) {
        for (int rows_q8 : {17, 257}) {
            std::vector<float> x(n_q8);
            for (int i = 0; i < n_q8; ++i) x[i] = float(std::sin(i * 0.17 - 0.1) * 0.25);
            auto dx = upload(x);
            Q::Buffer dx_q8(size_t(n_q8 / 32) * sizeof(Q::BlockQ8_1));
            Q::quantize_q8_1(dx.as(), dx_q8.as<Q::BlockQ8_1>(), n_q8, s);
            Q::ck(cudaStreamSynchronize(s));
            auto x_q8_deq = dequant_q8_1_host(read_q8(dx_q8));
            auto bytes = make_packed_weights(8, n_q8, rows_q8, uint32_t(n_q8 * 13 + rows_q8));
            auto ref_w = reference_ext(8, bytes, n_q8 * rows_q8);
            auto dw = upload(bytes);
            Q::Buffer dy(size_t(rows_q8) * 4);
            Q::matvec_q8_1(8, dw.data, dx_q8.as<Q::BlockQ8_1>(), dx.as(), dy.as(), n_q8, rows_q8, s);
            Q::ck(cudaStreamSynchronize(s));
            std::vector<double> want_q8(rows_q8);
            for (int r = 0; r < rows_q8; ++r)
                for (int i = 0; i < n_q8; ++i) want_q8[r] += ref_w[size_t(r) * n_q8 + i] * x_q8_deq[i];
            check_close(read(dy), want_q8, 2e-4, 5e-5, ("matvec_q8_0_tail_n" + std::to_string(n_q8) + "_r" + std::to_string(rows_q8)).c_str());
        }
    }
    // Unaligned 2B-offset and 1B-offset fallback tests across packed types
    for (int off_b : {2, 1}) {
        for (int ty : {8, 12, 13, 14, 21, 22}) {
            constexpr int n = 512, rows = 257;
            std::vector<float> x(n);
            for (int i = 0; i < n; ++i) x[i] = float(std::cos(i * 0.07) * 0.2);
            auto dx = upload(x);
            Q::Buffer dx_q8(size_t(n / 32) * sizeof(Q::BlockQ8_1));
            Q::quantize_q8_1(dx.as(), dx_q8.as<Q::BlockQ8_1>(), n, s);
            Q::ck(cudaStreamSynchronize(s));
            auto x_q8_deq = dequant_q8_1_host(read_q8(dx_q8));
            auto bytes = make_packed_weights(ty, n, rows, uint32_t(ty * 97 + off_b));
            auto ref_w = reference_ext(ty, bytes, n * rows);
            std::vector<uint8_t> padded(bytes.size() + size_t(off_b), 0);
            std::memcpy(padded.data() + off_b, bytes.data(), bytes.size());
            auto dw_pad = upload(padded);
            Q::Buffer dy(size_t(rows) * 4);
            Q::matvec_q8_1(ty, dw_pad.as<uint8_t>() + off_b, dx_q8.as<Q::BlockQ8_1>(), dx.as(), dy.as(), n, rows, s);
            Q::ck(cudaStreamSynchronize(s));
            std::vector<double> want(rows);
            for (int r = 0; r < rows; ++r)
                for (int i = 0; i < n; ++i) want[r] += ref_w[size_t(r) * n + i] * (off_b == 1 ? double(x[i]) : x_q8_deq[i]);
            check_close(read(dy), want, 2e-4, 5e-5, ("matvec_q8_1_off" + std::to_string(off_b) + "_t" + std::to_string(ty)).c_str());
        }
    }
    // 4. matvec_q8_1_dual & matvec_q8_1_triple (small + odd boundary splits)
    for (int ty : {8, 12, 13, 14, 21, 22, 142}) {
        for (int small_case : {1, 0}) {
            const int n = small_case ? 512 : 1024;
            const int r0 = small_case ? 48 : 259;
            const int r1 = small_case ? 48 : 263;
            std::vector<float> x(n);
            for (int i = 0; i < n; ++i) x[i] = float(std::sin(i * 0.05 + 0.1) * 0.2);
            auto dx = upload(x);
            Q::Buffer dx_q8(size_t(n / 32) * sizeof(Q::BlockQ8_1));
            Q::quantize_q8_1(dx.as(), dx_q8.as<Q::BlockQ8_1>(), n, s);
            Q::ck(cudaStreamSynchronize(s));
            auto x_q8_deq = dequant_q8_1_host(read_q8(dx_q8));
            auto b0 = make_packed_weights(ty, n, r0, uint32_t(ty * 11 + 1));
            auto b1 = make_packed_weights(ty, n, r1, uint32_t(ty * 11 + 2));
            auto rw0 = reference_ext(ty, b0, n * r0), rw1 = reference_ext(ty, b1, n * r1);
            auto dw0 = upload(b0), dw1 = upload(b1);
            Q::Buffer dy0(size_t(r0) * 4), dy1(size_t(r1) * 4);
            Q::matvec_q8_1_dual(ty, dw0.data, dy0.as(), r0, dw1.data, dy1.as(), r1, dx_q8.as<Q::BlockQ8_1>(), dx.as(), n, s);
            Q::ck(cudaStreamSynchronize(s));
            std::vector<double> w0_ref(r0), w1_ref(r1);
            for (int r = 0; r < r0; ++r) for (int i = 0; i < n; ++i) w0_ref[r] += rw0[size_t(r) * n + i] * (ty == 142 ? double(x[i]) : x_q8_deq[i]);
            for (int r = 0; r < r1; ++r) for (int i = 0; i < n; ++i) w1_ref[r] += rw1[size_t(r) * n + i] * (ty == 142 ? double(x[i]) : x_q8_deq[i]);
            check_close(read(dy0), w0_ref, 2e-4, 5e-5, ("dual_y0_t" + std::to_string(ty) + "_r" + std::to_string(r0)).c_str());
            check_close(read(dy1), w1_ref, 2e-4, 5e-5, ("dual_y1_t" + std::to_string(ty) + "_r" + std::to_string(r1)).c_str());
        }
        {
            constexpr int n = 1024, r0 = 257, r1 = 131, r2 = 135;
            std::vector<float> x(n);
            for (int i = 0; i < n; ++i) x[i] = float(std::cos(i * 0.06 - 0.2) * 0.2);
            auto dx = upload(x);
            Q::Buffer dx_q8(size_t(n / 32) * sizeof(Q::BlockQ8_1));
            Q::quantize_q8_1(dx.as(), dx_q8.as<Q::BlockQ8_1>(), n, s);
            Q::ck(cudaStreamSynchronize(s));
            auto x_q8_deq = dequant_q8_1_host(read_q8(dx_q8));
            auto b0 = make_packed_weights(ty, n, r0, uint32_t(ty * 17 + 1));
            auto b1 = make_packed_weights(ty, n, r1, uint32_t(ty * 17 + 2));
            auto b2 = make_packed_weights(ty, n, r2, uint32_t(ty * 17 + 3));
            auto rw0 = reference_ext(ty, b0, n * r0), rw1 = reference_ext(ty, b1, n * r1), rw2 = reference_ext(ty, b2, n * r2);
            auto dw0 = upload(b0), dw1 = upload(b1), dw2 = upload(b2);
            Q::Buffer dy0(size_t(r0) * 4), dy1(size_t(r1) * 4), dy2(size_t(r2) * 4);
            Q::matvec_q8_1_triple(ty, dw0.data, dy0.as(), r0, dw1.data, dy1.as(), r1, dw2.data, dy2.as(), r2,
                                  dx_q8.as<Q::BlockQ8_1>(), dx.as(), n, s);
            Q::ck(cudaStreamSynchronize(s));
            std::vector<double> w0_ref(r0), w1_ref(r1), w2_ref(r2);
            for (int r = 0; r < r0; ++r) for (int i = 0; i < n; ++i) w0_ref[r] += rw0[size_t(r) * n + i] * (ty == 142 ? double(x[i]) : x_q8_deq[i]);
            for (int r = 0; r < r1; ++r) for (int i = 0; i < n; ++i) w1_ref[r] += rw1[size_t(r) * n + i] * (ty == 142 ? double(x[i]) : x_q8_deq[i]);
            for (int r = 0; r < r2; ++r) for (int i = 0; i < n; ++i) w2_ref[r] += rw2[size_t(r) * n + i] * (ty == 142 ? double(x[i]) : x_q8_deq[i]);
            check_close(read(dy0), w0_ref, 2e-4, 5e-5, ("triple_y0_t" + std::to_string(ty)).c_str());
            check_close(read(dy1), w1_ref, 2e-4, 5e-5, ("triple_y1_t" + std::to_string(ty)).c_str());
            check_close(read(dy2), w2_ref, 2e-4, 5e-5, ("triple_y2_t" + std::to_string(ty)).c_str());
        }
    }
    // 5. Validation boundary checks
    Q::Buffer buf(4096);
    auto expect_throw = [&](auto&& fn, const char* msg) {
        bool threw = false;
        try { fn(); } catch (const std::exception&) { threw = true; }
        demand(threw, msg);
    };
    expect_throw([&]{ Q::rms_quantize_q8_1(buf.as(), buf.as(), buf.as(), buf.as<Q::BlockQ8_1>(), 0, 1e-6f, s); }, "rms_quantize n=0");
    expect_throw([&]{ Q::rms_quantize_q8_1(buf.as(), buf.as(), buf.as(), buf.as<Q::BlockQ8_1>(), 33, 1e-6f, s); }, "rms_quantize n%32");
    expect_throw([&]{ Q::rms_quantize_q8_1(buf.as(), buf.as(), buf.as(), buf.as<Q::BlockQ8_1>(), 17440, 1e-6f, s); }, "rms_quantize n>17408");
    expect_throw([&]{ Q::swiglu_quantize_q8_1(buf.as(), buf.as(), buf.as<Q::BlockQ8_1>(), 0, s); }, "swiglu_quantize n=0");
    expect_throw([&]{ Q::swiglu_quantize_q8_1(buf.as(), buf.as(), buf.as<Q::BlockQ8_1>(), 33, s); }, "swiglu_quantize n%32");
    expect_throw([&]{ Q::matvec_q8_1_dual(12, buf.data, buf.as(), 0, buf.data, buf.as(), 16, buf.as<Q::BlockQ8_1>(), buf.as(), 256, s); }, "dual rows0=0");
    expect_throw([&]{ Q::matvec_q8_1_triple(12, buf.data, buf.as(), 16, nullptr, buf.as(), 16, buf.data, buf.as(), 16, buf.as<Q::BlockQ8_1>(), buf.as(), 256, s); }, "triple null w1");
    std::puts("PASS q8_1_ops fused quantizers and single/dual/triple matvec");
}
static void failures(cudaStream_t s){admission();std::vector<float> x={0,INFINITY,NAN,-INFINITY};auto b=upload(x);Q::Buffer e(4);e.zero(s);Q::finite(b.as(),x.size(),e.as<int>(),s);int flag=0;Q::ck(cudaMemcpy(&flag,e.data,4,cudaMemcpyDeviceToHost));demand(flag==1,"nonfinite not caught");std::puts("PASS nonfinite failure detection");}
int main(){try{Q::host_resources();Q::ck(cudaSetDevice(0));cudaStream_t s;Q::ck(cudaStreamCreate(&s));try{packed(s);q8_1_ops(s);hadamard(s);gdn(s);gdn_batch(s);attention(s);failures(s);}catch(...){cudaStreamDestroy(s);throw;}Q::ck(cudaStreamDestroy(s));std::puts("PASS production CUDA numerical fixtures");return 0;}catch(const std::exception& e){std::fprintf(stderr,"FAIL %s\n",e.what());return 1;}}
