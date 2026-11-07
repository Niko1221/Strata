// CPU-only synthetic IQ3_S/IQ2_S down-row regression. No model or GPU.
// Calls public iq256x_rows and native_down_rows with real GGML AVX2 arithmetic.
// Only CPU/ISA probes and unrelated backends are stand-ins; no model or GPU.
#include "strata/kernels/cpu/iq_avx2.hpp"
#include "strata/kernels/cpu/native_expert.hpp"
#include "strata/kernels/cpu/tq_cpu.hpp"
#include "ggml-cpu.h"
#include <cstdlib>
#include <string>
#include "ggml-quants.h"
#include "ggml-cpu/quants.h"
#include "ggml-impl.h"
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>
extern "C" { float ggml_table_f32_f16[1 << 16]; }
namespace cpu = strata::kernels::cpu;
static bool avx2=true, avx512=false;
static int gg_avx2=1, gg_avx512=0, gather=0;
extern "C" {
const ggml_type_traits_cpu* ggml_get_type_traits_cpu(ggml_type type) {
    static ggml_type_traits_cpu iq3{}, iq2{};
    iq3.vec_dot=ggml_vec_dot_iq3_s_q8_K; iq2.vec_dot=ggml_vec_dot_iq2_s_q8_K;
    if(type==GGML_TYPE_IQ3_S) return &iq3;
    if(type==GGML_TYPE_IQ2_S) return &iq2;
    std::abort();
}
int ggml_cpu_has_avx2() { return gg_avx2; }
int ggml_cpu_has_avx512() { return gg_avx512; }
}
namespace strata::kernels::cpu {
bool cpu_avx2_ok() { return avx2; }
bool cpu_avx512_ok() { return avx512; }
bool cpu_avx1_ok() { return false; }
bool cpu_gather_fast() { return false; }
int iq256_gather_setting() { return gather; }
bool cpu_gather_fast_here() { return false; }
bool tq_supported(int t) noexcept { return t>=144 && t<=146; }
bool tq_kernels_enabled() { return true; }
TqIsa tq_isa() { return TqIsa::scalar; }
void tq_rows(TqIsa,int,const uint8_t*,size_t,int,const void* const*,int,float* const*,int,int) { std::abort(); }
void kq256_rows(int,const uint8_t*,size_t,int,const void* const*,int,float* const*,int,int) { std::abort(); }
bool iq128_down_supported(int) noexcept { return false; }
void iq128_down_rows(int,const uint8_t*,size_t,int,const void* const*,int,float* const*,int,int) { std::abort(); }
bool nvfp4_512_fits(int) { return false; }
bool nvfp4_256_fits(int) { return false; }
void nvfp4_512_rows(const uint8_t*,size_t,int,const void* const*,int,float* const*,int,int,float) { std::abort(); }
void nvfp4_256_rows(const uint8_t*,size_t,int,const void* const*,int,float* const*,int,int,float) { std::abort(); }
}

static uint64_t next(uint64_t& x) {
    uint64_t z = (x += 0x9e3779b97f4a7c15ULL);
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
    return z ^ (z >> 31);
}
static uint32_t bits(float x) { uint32_t n; std::memcpy(&n, &x, sizeof(n)); return n; }
int main(int argc, char** argv) {
    const bool native=argc>1 && std::string(argv[1])=="native";
    if (!__builtin_cpu_supports("avx2") || !__builtin_cpu_supports("fma") || !__builtin_cpu_supports("f16c")) {
        std::fprintf(stderr, "SKIP: AVX2/FMA/F16C required\n"); return 77;
    }
    for (int i=0;i<65536;++i) ggml_table_f32_f16[i]=GGML_COMPUTE_FP16_TO_FP32((ggml_fp16_t)i);
    constexpr int N=512, T=8;
    const int R=native?2048:9;
    uint64_t checked=0, mismatches=0, preservation=0;
    for (int type : {21,22}) for (uint64_t seed : {1ULL,42ULL,871ULL,1097ULL}) for (int pad : {0,16}) {
        uint64_t state=seed;
        if(native && pad) continue;
        const size_t block=type==21?110:82, stride=2*block+pad;
        cpu::NativeFmt f;
        f.gu_type=22; f.d_type=type; f.gu_act=f.d_act=15; f.n_ff=N; f.n_embd=2048;
        f.gu_row=656; f.d_row=stride; f.act_bytes=2336; f.h_bytes=584;
        f.up_off=656*512; f.down_off=2*f.up_off; f.bytes=f.down_off+stride*2048;
        const size_t offset=native?f.down_off:0;
        std::vector<uint8_t> blob(offset+R*stride+4096,0xa5);
        uint8_t* w=blob.data()+offset;
        for (int r=0;r<R;++r) for (int b=0;b<2;++b) {
            uint8_t* p=w+r*stride+b*block;
            for (size_t k=0;k<block;++k) p[k]=uint8_t(next(state));
            const uint16_t scale=uint16_t(0x2400+(next(state)%0x1000));
            std::memcpy(p,&scale,2);
        }
        std::vector<block_q8_K> acts(T*2);
        std::vector<float> input(N);
        for (int t=0;t<T;++t) {
            for (auto& v:input) v=float(int(next(state)%20001)-10000)/1024.0f;
            quantize_row_q8_K_ref(input.data(),acts.data()+t*2,N);
        }
        const auto wb=blob; const auto ab=acts;
        const void* ap[T]; float* out[T];
        std::vector<float> expected(T*R), got(T*R,123456.0f);
        for (int t=0;t<T;++t) {
            ap[t]=acts.data()+t*2; out[t]=got.data()+t*R;
            for (int r=0;r<R;++r) {
                if (type==21) ggml_vec_dot_iq3_s_q8_K(N,&expected[t*R+r],0,w+r*stride,0,ap[t],0,1);
                else ggml_vec_dot_iq2_s_q8_K(N,&expected[t*R+r],0,w+r*stride,0,ap[t],0,1);
            }
        }
        for (int variant : {0,1}) for (int nt=1;nt<=T;++nt) {
            gather=variant;
            auto rows=[&](int r0,int r1) {
                if(native) cpu::native_down_rows(f,blob.data(),ap,nt,out,r0,r1);
                else cpu::iq256x_rows(type,w,stride,N,ap,nt,out,r0,r1);
            };
            std::fill(got.begin(),got.end(),123456.0f);
            rows(0,0);
            for (float x:got) preservation+=bits(x)!=bits(123456.0f);
            rows(R/2,R);
            for (int t=0;t<T;++t) for (int r=0;r<R/2;++r) preservation+=bits(got[t*R+r])!=bits(123456.0f);
            rows(0,R/2);
            for (int t=0;t<T;++t) for (int r=0;r<R;++r) {
                if(t<nt) { ++checked; mismatches+=bits(got[t*R+r])!=bits(expected[t*R+r]); }
                else preservation+=bits(got[t*R+r])!=bits(123456.0f);
            }
        }
        preservation+=blob!=wb;
        preservation+=std::memcmp(acts.data(),ab.data(),acts.size()*sizeof(block_q8_K))!=0;
    }
    std::printf("synthetic=1 entry=%s checked=%llu bit_mismatches=%llu preservation_failures=%llu\n",
        native?"native_down_rows":"iq256x_rows",(unsigned long long)checked,(unsigned long long)mismatches,(unsigned long long)preservation);
    return mismatches || preservation ? 1 : 0;
}
