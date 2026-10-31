// Dispatch/control-flow fixture, NOT a numerical kernel test. Links the real native_expert.cpp
// with instrumented CPU probes and row backends; no model, GPU, or wide-ISA execution.
#include "strata/kernels/cpu/native_expert.hpp"
#include "strata/kernels/cpu/tq_cpu.hpp"
#include "ggml-cpu.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <string>
namespace cpu = strata::kernels::cpu;
static bool avx2=true, avx512=false;
static int gg_avx2=1, gg_avx512=0, iq_calls=0, dots=0;
static const uint8_t* expected_weights=nullptr;
static const void* const* expected_acts=nullptr;
static float* const* expected_out=nullptr;
static size_t expected_stride=0;
static int expected_type=0, expected_nt=0, expected_r0=0, expected_r1=0;
static bool forwarded=true;
static void dot(int, float* s, size_t, const void*, size_t, const void*, size_t, int) { ++dots; *s=7; }
extern "C" {
const ggml_type_traits_cpu* ggml_get_type_traits_cpu(ggml_type) {
    static ggml_type_traits_cpu t{}; t.vec_dot=dot; return &t;
}
int ggml_cpu_has_avx2() { return gg_avx2; }
int ggml_cpu_has_avx512() { return gg_avx512; }
}
namespace strata::kernels::cpu {
bool cpu_avx2_ok() { return avx2; }
bool cpu_avx512_ok() { return avx512; }
bool cpu_avx1_ok() { return false; }
bool cpu_gather_fast() { return false; }
int iq256_gather_setting() { return 0; }
bool tq_supported(int t) noexcept { return t>=144 && t<=146; }
bool tq_kernels_enabled() { return true; }
TqIsa tq_isa() { return TqIsa::scalar; }
void tq_rows(TqIsa,int,const uint8_t*,size_t,int,const void* const*,int,float* const*,int,int) { std::abort(); }
void iq256x_rows(int type,const uint8_t* w,size_t stride,int n,const void* const* acts,int nt,float* const* out,int r0,int r1) {
    ++iq_calls;
    forwarded = forwarded && type==expected_type && w==expected_weights && stride==expected_stride && n==512 &&
                acts==expected_acts && nt==expected_nt && out==expected_out && r0==expected_r0 && r1==expected_r1;
}
void kq256_rows(int,const uint8_t*,size_t,int,const void* const*,int,float* const*,int,int) { std::abort(); }
void iq4nl256_down_rows(const uint8_t*,size_t,int,const void* const*,int,float* const*,int,int) { std::abort(); }
bool iq128_down_supported(int) noexcept { return false; }
void iq128_down_rows(int,const uint8_t*,size_t,int,const void* const*,int,float* const*,int,int) { std::abort(); }
bool nvfp4_512_fits(int) { return false; }
bool nvfp4_256_fits(int) { return false; }
void nvfp4_512_rows(const uint8_t*,size_t,int,const void* const*,int,float* const*,int,int,float) { std::abort(); }
void nvfp4_256_rows(const uint8_t*,size_t,int,const void* const*,int,float* const*,int,int,float) { std::abort(); }
}
static cpu::NativeFmt fmt(int d) {
    cpu::NativeFmt f;
    f.gu_type=22; f.d_type=d; f.gu_act=f.d_act=15; f.n_ff=512; f.n_embd=2048;
    f.gu_row=656; f.d_row=d==22?164:220; f.act_bytes=2336; f.h_bytes=584;
    f.up_off=656*512; f.down_off=2*f.up_off; f.bytes=f.down_off+f.d_row*2048;
    return f;
}
int main(int argc,char** argv) {
    const std::string mode=argc>1?argv[1]:"default";
    if(mode!="off" && mode!="zero" && mode!="invalid") setenv("STRATA_IQ_DOWN_EXACT","1",1);
    if(mode=="zero") setenv("STRATA_IQ_DOWN_EXACT","0",1);
    if(mode=="invalid") setenv("STRATA_IQ_DOWN_EXACT","yes",1);
    if(mode=="no-iq") setenv("STRATA_NO_IQ256","1",1);
    if(mode=="mt9") setenv("STRATA_IQ_MT_MIN","9",1);
    const bool enabled=mode=="on";
    int checked=0,failures=0;
    auto check=[&](const char* label,cpu::NativeFmt f,int nt,bool expect,int r0=0,int r1=1) {
        // All mutated formats retain allocated backing. Invalid row ranges only use empty loops.
        std::vector<uint8_t> blob(2000000,0); const void* a[9]{};
        if(f.tail_off) { const float scale=1; std::memcpy(blob.data()+f.tail_off+8,&scale,4); }
        float data[9][2049]{}; float* out[9]; for(int t=0;t<9;++t)out[t]=data[t];
        iq_calls=dots=0; forwarded=true;
        expected_type=f.d_type; expected_weights=blob.data()+f.down_off; expected_stride=f.d_row;
        expected_acts=a; expected_out=out; expected_nt=nt; expected_r0=r0; expected_r1=r1;
        cpu::native_down_rows(f,blob.data(),a,nt,out,r0,r1);
        bool ok=(iq_calls==(expect?1:0)) && forwarded;
        if(!expect && r0<r1) ok=ok && dots==nt*(r1-r0) && data[0][r0]==7;
        ++checked; if(!ok) { ++failures; std::printf("FAIL %s d=%d nt=%d iq=%d dots=%d\n",label,f.d_type,nt,iq_calls,dots); }
    };
    for(int d:{21,22}) {
        const auto f=fmt(d);
        for(int nt=1;nt<=9;++nt) check("width",f,nt,enabled && nt>=2 && nt<=8);
        gg_avx2=0; check("ggml-no-avx2",f,2,false); gg_avx2=1;
        gg_avx512=1; check("ggml-avx512-forced-avx2",f,2,false); gg_avx512=0;
        avx2=false; check("cpu-no-avx2",f,2,false); avx2=true;
        avx512=true; check("cpu-avx512",f,2,false); avx512=false;
        auto g=f; g.had_gu=true; check("had-gu",g,2,false);
        g=f; g.had_d=true; check("had-down",g,2,false);
        g=f; g.gu_type=21; check("gu-type",g,2,false);
        g=f; g.gu_act=8; check("gu-act",g,2,false);
        g=f; g.d_act=8; check("d-act",g,2,false);
        g=f; g.n_ff=256; check("ff",g,2,false);
        g=f; g.n_embd=2560; check("embd",g,2,false);
        g=f; g.d_type=23; check("unsupported-down",g,2,false);
        g=f; ++g.gu_row; check("gu-row",g,2,false);
        g=f; ++g.d_row; check("d-row",g,2,false);
        g=f; ++g.act_bytes; check("act-size",g,2,false);
        g=f; ++g.h_bytes; check("hidden-size",g,2,false);
        g=f; ++g.up_off; check("up-offset",g,2,false);
        g=f; ++g.down_off; check("down-offset",g,2,false);
        g=f; ++g.bytes; check("blob-size",g,2,false);
        g=f; g.tail_off=1; check("tail",g,2,false);
        check("empty",f,2,enabled,5,5);
        check("reverse",f,2,false,5,4);
        check("last-row",f,2,enabled,2047,2048);
    }
    std::printf("dispatch mode=%s checked=%d failures=%d (instrumented backends, not numerics)\n",mode.c_str(),checked,failures);
    return failures?1:0;
}
