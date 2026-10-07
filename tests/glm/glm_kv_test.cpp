#include "strata/glm/kv.hpp"
#include <cstdio>
#include <limits>
using namespace strata::glm;
int main() {
    int bad=0;
    // Every finite FP8 value must round-trip, including sign, subnormals and rounding boundaries.
    for(int i=0;i<256;++i) if((i&127)!=127 && to_fp8(from_fp8((uint8_t)i))!=i)++bad;
    if(to_fp8(0.0009765625f)!=0 || to_fp8(448)!=126 || to_fp8(1000)!=126)++bad;
    if(to_bf16(1.00390625f)!=0x3f80 || !std::isnan(from_bf16(to_bf16(std::numeric_limits<float>::quiet_NaN()))))++bad;
    for(auto fmt:{KvFormat::F32,KvFormat::BF16,KvFormat::FP8}) {
        KvCache kv; kv.reset(2,4,128,64,fmt);
        std::vector<float> x(192),y(192*3);
        for(int i=0;i<192;++i)x[i]=(i<128?1.0f:0.01f)*std::sin((float)i);
        kv.write(1,2,x.data()); kv.read_layer(1,3,y.data());
        for(int i=0;i<192;++i) {
            const float tol=fmt==KvFormat::F32?0:fmt==KvFormat::BF16?0.004f:0.063f;
            if(std::fabs(y[384+i]-x[i])>tol*std::max(0.01f,std::fabs(x[i])))++bad;
        }
        for(int i=0;i<384;++i)if(y[i]!=0)++bad;
        const size_t expected=8*(fmt==KvFormat::F32?192*4:fmt==KvFormat::BF16?192*2:192+8);
        if(kv.bytes()!=expected)++bad;
        std::printf("KV %s: %llu bytes\n",kv_name(fmt),(unsigned long long)kv.bytes());
    }
    std::printf("compact KV: %d failures\n",bad); return bad?1:0;
}
