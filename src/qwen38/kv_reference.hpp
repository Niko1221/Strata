#pragma once
// Independent CPU block layouts and FP64 attention/transform reference.
#include "kv.hpp"
#include "tqcodec/dialects.h"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <vector>
#include <cassert>
namespace kv_reference {
inline void demand(bool b,const char* why){if(!b)throw std::runtime_error(why);}
inline double half_value(uint16_t u){int sign=u>>15?-1:1,e=(u>>10)&31,f=u&1023;return e==31?(f?NAN:sign*INFINITY):sign*std::ldexp(double(e?1024+f:f),e?e-25:-24);}
inline uint16_t half_bits(float x){return __half_as_ushort(__float2half_rn(x));}
namespace tables {
inline uint16_t half_encode(float x){return half_bits(x);}
inline float half_decode(uint16_t x){return float(half_value(x));}
#include "dialect_tables.inc"
}
inline void put_half(uint8_t* p,float x){auto h=half_bits(x);p[0]=uint8_t(h);p[1]=uint8_t(h>>8);}
inline double get_half(const uint8_t* p){return half_value(uint16_t(p[0])|uint16_t(p[1])<<8);}
inline void cpu_dialect(int id,int op,const void* in,void* out,size_t rows,size_t cols,size_t src_stride,size_t dst_stride){
    tqc_dialect_v1_request q{1,sizeof(q),uint32_t(id),uint32_t(op),TQC_D_CPU,TQC_D_F32,nullptr,rows,cols,
        {const_cast<void*>(in),rows*src_stride,src_stride},{out,rows*dst_stride,dst_stride},{}};
    tqc_dialect_v1_receipt receipt{1,sizeof(receipt),0,0,0};demand(tqc_dialect_v1_execute(&q,&receipt)==0,"CPU Turbo oracle failed");
}
inline std::vector<float> spin_f32(const std::vector<float>& x,const strata::qwen38::KvConfig& c){
    std::vector<float> out(x.size());for(size_t b=0;b<x.size();b+=8)for(int i=0;i<8;++i){float v=0;for(int j=0;j<8;++j)v=std::fma(c.matrix[i*8+j],x[b+j],v);out[b+i]=v;}return out;
}
inline std::vector<double> inverse(const std::vector<double>& x,const strata::qwen38::KvConfig& c){
    auto out=x;
    if(c.mode==strata::qwen38::KvMode::spin8_k8v4){
        std::fill(out.begin(),out.end(),0);for(size_t b=0;b<x.size();b+=8)for(int i=0;i<8;++i)for(int j=0;j<8;++j)out[b+i]+=double(c.matrix[j*8+i])*x[b+j];
    }else if(c.turbo()){
        // Dense parity matrix, no butterfly: F^-1 = D1 H D2 / sqrt128.
        std::fill(out.begin(),out.end(),0);for(size_t b=0;b<x.size();b+=128)for(int i=0;i<128;++i)for(int j=0;j<128;++j){
            unsigned p=i&j;p^=p>>4;p^=p>>2;p^=p>>1;
            out[b+i]+=(p&1?-1.:1.)/std::sqrt(128.)*tables::turbo_cpu_s1[i]*tables::turbo_cpu_s2[j]*x[b+j];
        }
    }return out;
}
inline std::vector<uint8_t> pack(const std::vector<float>& x,const strata::qwen38::KvConfig& c,bool key){
    size_t stride=key?c.key_row_bytes():c.value_row_bytes();std::vector<uint8_t> out(x.size()/256*stride);
    if(!key&&c.turbo()){cpu_dialect(c.turbo(),TQC_D_ENCODE,x.data(),out.data(),x.size()/256,256,1024,stride);return out;}
    if(stride==512){for(size_t i=0;i<x.size();++i)put_half(out.data()+i*2,x[i]);return out;}
    for(size_t b=0;b<x.size()/32;++b){const float* src=x.data()+b*32;uint8_t* dst=out.data()+b*(key?34:18);
        float largest=0,max_value=0;for(int j=0;j<32;++j)if(std::abs(src[j])>largest){largest=std::abs(src[j]);max_value=src[j];}
        float scale=key?largest/127.f:max_value/-8.f,reciprocal=scale?1.f/scale:0;put_half(dst,scale);
        if(key){for(int j=0;j<32;++j)dst[2+j]=uint8_t(int8_t(std::round(src[j]*reciprocal)));}
        else{for(int j=0;j<16;++j)dst[2+j]=uint8_t(std::min(15,int(src[j]*reciprocal+8.5f))|(std::min(15,int(src[j+16]*reciprocal+8.5f))<<4));}
    }return out;
}
inline std::vector<double> unpack(const std::vector<uint8_t>& bytes,const strata::qwen38::KvConfig& c,bool key){
    size_t stride=key?c.key_row_bytes():c.value_row_bytes();std::vector<double> out(bytes.size()/stride*256);
    for(size_t i=0;i<out.size();++i){
        if(stride==512){out[i]=get_half(bytes.data()+2*i);continue;}
        if(!key&&c.turbo()){
            auto* p=bytes.data()+i/128*(18+16*c.turbo());size_t j=i%128;int code;
            if(c.turbo()==3)code=(p[2+j/2]>>(4*(j%2)))&15;
            else{code=(p[2+j/4]>>(2*(j%4)))&3;if(c.turbo()==2)code|=((p[34+j/8]>>(j%8))&1)<<2;}
            const float* table=c.turbo()==1?tables::CENTROIDS_2BIT:c.turbo()==2?tables::CENTROIDS_3BIT:tables::CENTROIDS_4BIT;
            out[i]=float(float(get_half(p))*table[code]);
        }else{auto* p=bytes.data()+i/32*(key?34:18);int j=int(i%32);out[i]=get_half(p)*(key?int(int8_t(p[2+j])):int((p[2+j%16]>>(j<16?0:4))&15)-8);}
    }return out;
}
inline std::vector<double> attention(const std::vector<float>& q,const std::vector<float>& gate,const std::vector<std::vector<double>>& keys,const std::vector<std::vector<double>>& values,const strata::qwen38::KvConfig& c){
    size_t count=keys.size();std::vector<double> out(6144);
    for(int head=0;head<24;++head){std::vector<double> probabilities(count);for(size_t t=0;t<count;++t)for(int j=0;j<256;++j)probabilities[t]+=double(q[head*256+j])*keys[t][head/6*256+j]/16;
        double maximum=*std::max_element(probabilities.begin(),probabilities.end()),denominator=0;for(auto& p:probabilities)denominator+=(p=std::exp(p-maximum));
        for(int j=0;j<256;++j)for(size_t t=0;t<count;++t)out[head*256+j]+=probabilities[t]/denominator*values[t][head/6*256+j];
    }
    out=inverse(out,c);for(int i=0;i<6144;++i)out[i]/=1+std::exp(-double(gate[i]));return out;
}
}
