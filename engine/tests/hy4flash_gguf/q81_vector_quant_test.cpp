#include "strata/hy4flash_gguf/cpu/native_q81.hpp"
#include "strata/hy4flash_gguf/cpu/expert_layout.hpp"
#include "ggml.h"
#include <array>
#include <vector>
#include <random>
#include <bit>
#include <cmath>
#include <limits>
#include <cstdio>
#include <cstring>
#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
#include <immintrin.h>
namespace c=strata::hy4_flash::cpu;
int main(){
 if(!strata::kernels::cpu::cpu_avx2_ok())return 77;
 std::mt19937 rng(20261108);unsigned long long cases=0;int fail=0;
 auto test=[&](const float*x,int n,const char*kind){std::vector<unsigned char>a(n/32*36),b(a.size());c::native_quant_q81_scalar(x,a.data(),n);c::native_quant_q81_avx2(x,b.data(),n);++cases;if(a!=b){if(fail++<5){size_t i=0;while(a[i]==b[i])++i;std::printf("FAIL kind=%s case=%llu byte=%zu scalar=%u simd=%u\n",kind,cases,i,a[i],b[i]);}}};
 std::array<float,32>x{};
 const unsigned saved=_mm_getcsr();
 for(unsigned extra:{0u,0x8040u}){
  _mm_setcsr((saved&~0x8040u)|extra);
  for(int k=0;k<150000;++k){for(float&v:x){uint32_t z=rng();if((z&0x7f800000u)==0x7f800000u)z^=0x00800000u;v=std::bit_cast<float>(z);}test(x.data(),32,"random-finite-bits");}
  for(int k=0;k<50000;++k){const float scale=std::ldexp(1.f,int(rng()%240)-120);for(float&v:x)v=(float(int(rng()%200001)-100000)/100000.f)*scale;test(x.data(),32,"scaled");}
  for(float v:{0.f,-0.f,std::numeric_limits<float>::denorm_min(),-std::numeric_limits<float>::denorm_min(),std::numeric_limits<float>::min(),std::numeric_limits<float>::max(),65504.f,65504.f*127.f,1.f,-1.f}){x.fill(v);test(x.data(),32,"constant");}
  // Force a known block maximum; inspect both neighbors of every half-integer code boundary.
  for(int e=-120;e<120;e+=3)for(int j=-126;j<=126;++j){float d=std::ldexp(1.f,e),v=(j+.5f)*d;for(float q:{std::nextafter(v,-INFINITY),v,std::nextafter(v,INFINITY)}){x.fill(q);x[31]=127.f*d;test(x.data(),32,"round-threshold");}}
  // Scale and sum half-rounding boundary neighborhoods, including subnormal half scales.
  for(unsigned h=1;h<0x7bfe;h+=3){float lo=ggml_fp16_to_fp32(h),hi=ggml_fp16_to_fp32(h+1),mid=(lo+hi)*.5f;for(float d:{std::nextafter(mid,0.f),mid,std::nextafter(mid,INFINITY)}){x.fill(0);x[0]=d*127.f;x[1]=-x[0];test(x.data(),32,"half-scale");x.fill(d);test(x.data(),32,"half-sum");}}
  for(int k=0;k<256;++k){for(int j=0;j<32;++j)x[j]=std::bit_cast<float>(uint32_t((rng()%0x007fffffu)+1u))*((j&1)?-1:1);test(x.data(),32,"subnormal");}
  x.fill(0);x[0]=x[16]=std::numeric_limits<float>::max();x[8]=x[24]=-std::numeric_limits<float>::max();test(x.data(),32,"finite-tree-NaN");
  x.fill(std::numeric_limits<float>::max());test(x.data(),32,"finite-tree-Inf");
  for(int j=0;j<32;++j)x[j]=(j&1)?-0.f:0.f;test(x.data(),32,"signed-zeros");
  for(float v:{INFINITY,-INFINITY,std::numeric_limits<float>::quiet_NaN()}){x.fill(1.f);x[7]=v;test(x.data(),32,"nonfinite-fallback");}
 }
 _mm_setcsr(saved);
 std::vector<float>v(1536);for(int i=0;i<1536;++i)v[i]=std::sin(i*.31f)*4.f;test(v.data(),1536,"1536");
 std::printf("SUMMARY cases=%llu fail=%d\n",cases,fail);return fail?1:0;
}

#else
int main() { return 77; } // Runtime test applies only to the AVX2 quantizer.
#endif
