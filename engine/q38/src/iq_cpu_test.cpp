#include "iq_tables.hpp"
#include "iq_decode.hpp"
#include "strata/qwen38/admission.hpp"
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <stdexcept>
using strata::qwen38::iq_value;
// Reference loops follow dequantize_row_iq2_s / dequantize_row_iq3_s at
// ggml commit 643956891dfdf51aecdbd05ca508dd21e21e8195, MIT.
static void reference(int type,const uint8_t* p,float d,float* out){
    if(type==22){for(int ib=0;ib<8;++ib){float db[2]={d*(.5f+(p[74+ib]&15))*.25f,d*(.5f+(p[74+ib]>>4))*.25f};
        for(int l=0;l<4;++l){unsigned idx=p[2+4*ib+l]|((p[66+ib]<<(8-2*l))&0x300);uint64_t grid=q38_iq2s_grid[idx];
            for(int j=0;j<8;++j)*out++=db[l/2]*float((grid>>(8*j))&255)*(p[34+4*ib+l]&(1<<j)?-1.f:1.f);}}}
    else {for(int ib=0;ib<8;ib+=2){for(int k=0;k<2;++k){float db=d*(1+2*((p[106+ib/2]>>(4*k))&15));
        for(int l=0;l<4;++l){uint8_t hi=p[66+ib+k],sign=p[74+(ib+k)*4+l];unsigned q0=p[2+(ib+k)*8+2*l]|((hi<<(8-2*l))&256),q1=p[3+(ib+k)*8+2*l]|((hi<<(7-2*l))&256);
            uint32_t g0=q38_iq3s_grid[q0],g1=q38_iq3s_grid[q1];for(int j=0;j<8;++j){uint32_t g=j<4?g0:g1;*out++=db*float((g>>(8*(j%4)))&255)*(sign&(1<<j)?-1.f:1.f);}}}}}
}
int main(){try{uint32_t rng=0x5aa53117;std::array<uint8_t,110> b{};float expected[256];
    for(int type:{21,22}){strata::TensorInfo t;t.type=type;t.shape={5120,17408};
        if(!strata::qwen38::matrix_type(type)||strata::qwen38::payload(t)!=uint64_t(20)*17408*(type==21?110:82))throw std::runtime_error("IQ row geometry");
        t.shape={257,256};if(strata::qwen38::payload(t))throw std::runtime_error("partial packed row admitted");
        t.shape={5120,UINT64_MAX};if(strata::qwen38::payload(t))throw std::runtime_error("overflow extent admitted");}
    {strata::TensorInfo t;t.type=8;t.shape={5120,48};t.name="blk.0.ssm_alpha.weight";
        if(!strata::qwen38::matrix_type(t.type)||strata::qwen38::convert_auxiliary(t)||strata::qwen38::payload(t)!=uint64_t(160)*48*34)throw std::runtime_error("Q8 projection path");
        t.name="blk.0.ssm_beta.weight";if(strata::qwen38::convert_auxiliary(t))throw std::runtime_error("Q8 beta projection path");}
    for(int type:{21,22})for(int iteration=0;iteration<2048;++iteration){for(auto& x:b){rng=rng*1664525u+1013904223u;x=uint8_t(rng>>24);}
        // Exhaust each codebook index in every subblock and group; random signs/scales remain.
        for(int sub=0;sub<8;++sub){b[66+sub]=uint8_t(iteration>>8)*(type==21?255:85);for(int k=0;k<(type==21?8:4);++k)b[2+sub*(type==21?8:4)+k]=uint8_t(iteration);
            for(int group=0;group<4;++group)b[(type==21?74:34)+4*sub+group]=uint8_t(iteration>>2);}
        for(float d:{0.f,0.03125f,-0.0625f}){reference(type,b.data(),d,expected);for(int i=0;i<256;++i){float got=iq_value(type,b.data(),i,d,q38_iq2s_grid,q38_iq3s_grid);if(std::memcmp(&got,&expected[i],sizeof(float))!=0)throw std::runtime_error("scalar decode differs");}}}
    std::puts("PASS IQ2_S and IQ3_S scalar decode versus independent pinned block loops");return 0;
}catch(const std::exception& e){std::fprintf(stderr,"FAIL %s\n",e.what());return 1;}}
