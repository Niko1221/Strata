#include "strata/hy4flash_gguf/cpu/native_q81.hpp"
#include "strata/kernels/cpu/expert.hpp"
#include "strata/hy4flash_gguf/cpu/expert_layout.hpp"
#include "ggml.h"
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
namespace strata::hy4_flash::cpu {
using namespace ::strata::kernels::cpu;
namespace {
float half_at(const uint8_t* p) { ggml_fp16_t h; std::memcpy(&h,p,2); return ggml_fp16_to_fp32(h); }
void scale_min(const uint8_t* s,int j,int& sc,int& mn) {
    if(j<4){sc=s[j]&63;mn=s[j+4]&63;}
    else {sc=(s[j+4]&15)|((s[j-4]>>6)<<4);mn=(s[j+4]>>4)|((s[j]>>6)<<4);}
}
}
void native_quant_q81_scalar(const float* x, void* dst, int n) {
    auto* y=static_cast<uint8_t*>(dst);
    for(int b=0;b<n/32;++b) {
        float amax=0.f, sum[32];
        for(int i=0;i<32;++i){sum[i]=x[b*32+i];amax=std::fmax(amax,std::fabs(sum[i]));}
        for(int step=16;step>0;step/=2)for(int i=0;i<step;++i)sum[i]+=sum[i+step];
        float d=amax/127.f;
        if(std::fabs(d)>65504.f)d=std::copysign(65504.f,d);
        const float raw=std::fabs(sum[0])>65504.f?std::copysign(65504.f,sum[0]):sum[0];
        const ggml_fp16_t ds[2]={ggml_fp32_to_fp16(d),ggml_fp32_to_fp16(raw)};
        std::memcpy(y+b*36,ds,4);
        auto* q=reinterpret_cast<int8_t*>(y+b*36+4);
        for(int i=0;i<32;++i) {
            const float v=amax==0.f?0.f:std::round(x[b*32+i]/d);
            q[i]=(int8_t)std::clamp(v,-127.f,127.f);
        }
    }
}
void native_quant_q81_compatible(const float* x, void* dst, int n) {
#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
    static const bool vector = [] {
        const char* opt = std::getenv("STRATA_CPU_Q81_VECTOR_QUANT");
        return opt && std::strcmp(opt, "1") == 0 && cpu_avx2_ok();
    }();
    if (vector) { native_quant_q81_avx2(x, dst, n); return; }
#endif
    native_quant_q81_scalar(x, dst, n);
}
float native_dot_q81_scalar(int type,const uint8_t* row,const void* act,int n) {
    const auto* a=static_cast<const uint8_t*>(act);
    double total=0;
    if(type!=12 && type!=14)throw std::invalid_argument("Q8_1 CPU dot needs Q4_K/Q6_K");
    for(int b=0;b<n/256;++b) {
        const uint8_t* w=row+b*(type==12?144:210);
        const float dw=half_at(w+(type==12?0:208)),dm=type==12?half_at(w+2):0.f;
        for(int g=0;g<8;++g) {
            const uint8_t* q8=a+(b*8+g)*36;const auto* aq=reinterpret_cast<const int8_t*>(q8+4);
            int dot0=0,dot1=0,suma=0,sc=0,mn=0;
            if(type==12)scale_min(w+4,g,sc,mn);
            for(int i=0;i<32;++i) {
                int q;
                if(type==12){q=(w[16+(g/2)*32+i]>>(4*(g&1)))&15;dot0+=q*aq[i];suma+=aq[i];}
                else {
                    const int h=g/4,p=g%4;
                    q=((w[h*64+(p&1)*32+i]>>(p>=2?4:0))&15) |
                      (((w[128+h*32+i]>>(2*p))&3)<<4);
                    q-=32;(i<16?dot0:dot1)+=q*aq[i];
                }
            }
            const float da=half_at(q8);
            if(type==12)total+=(double)dw*da*(dot0*sc)-(double)dm*da*(suma*mn);
            else total+=(double)dw*da*(dot0*(int8_t)w[192+g*2]+dot1*(int8_t)w[193+g*2]);
        }
    }
    return (float)total;
}
float native_dot_q81_compatible(int type,const uint8_t* row,const void* act,int n) {
#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
    static const bool avx=cpu_avx2_ok() && std::getenv("STRATA_CPU_Q81_SCALAR")==nullptr;
    if(avx)return native_dot_q81_avx2(type,row,act,n);
#endif
    return native_dot_q81_scalar(type,row,act,n);
}
bool native_dot_q81_pair_compatible(int type, const uint8_t* gate, const uint8_t* up,
                                    const void* act, int n, float& g, float& u) {
#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
    static const bool enabled = [] {
        const char* opt = std::getenv("STRATA_CPU_Q81_PAIR_GU");
        return opt && std::strcmp(opt, "1") == 0 && cpu_avx2_ok() &&
               std::getenv("STRATA_CPU_Q81_SCALAR") == nullptr;
    }();
    if (enabled && type == 12) {
        native_dot_q81_pair_q4_avx2(gate, up, act, n, g, u);
        return true;
    }
#endif
    return false;
}
} // namespace strata::hy4_flash::cpu
