// Compare portable Q2 rows with pinned ggml ARM dots; activation contracts differ.
#include "strata/kernels/cpu/expert.hpp"
#include "ggml.h"
#include "ggml-cpu.h"
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>
namespace c=strata::kernels::cpu;
int main(){
 ggml_cpu_init();constexpr int rows=256,blocks=40,n=blocks*64,nt=3,iterations=20;
 std::mt19937 rng(87126);std::vector<unsigned char>w(rows*blocks*18);
 for(int r=0;r<rows;++r)for(int b=0;b<blocks;++b){auto*p=w.data()+(r*blocks+b)*18;uint16_t scale=0x2400;std::memcpy(p,&scale,2);for(int j=0;j<16;++j)p[2+j]=rng();}
 auto*traits=ggml_get_type_traits_cpu(GGML_TYPE_Q2_0);auto*act_traits=ggml_get_type_traits_cpu(traits->vec_dot_type);
 c::ActQ acts[nt];const c::ActQ*input[nt];float*dst[nt];std::vector<float>a(rows*nt),b(rows*nt),x(n);
 std::vector<unsigned char>qa[nt];
 for(int t=0;t<nt;++t){for(auto&v:x)v=float(int(rng()%2001)-1000)/317;c::act_quant_q8_1(x.data(),n,acts[t]);input[t]=&acts[t];dst[t]=a.data()+t*rows;qa[t].resize(ggml_row_size(traits->vec_dot_type,n));act_traits->from_float(x.data(),qa[t].data(),n);}
 auto scalar=[&](){c::q2_0_gguf_rows_multi(w.data(),blocks*18,blocks,input,nt,dst,0,rows);};
 auto ggml=[&](){for(int t=0;t<nt;++t)for(int r=0;r<rows;++r)traits->vec_dot(n,b.data()+t*rows+r,0,w.data()+r*blocks*18,0,qa[t].data(),0,1);};
 scalar();ggml();double squared=0,mag=0;
 for(size_t i=0;i<a.size();++i){if(!std::isfinite(a[i])||!std::isfinite(b[i]))return 1;squared+=std::pow(double(a[i])-b[i],2);mag+=double(b[i])*b[i];}
 for(int path=0;path<2;++path){auto start=std::chrono::steady_clock::now();for(int i=0;i<iterations;++i){if(path==0)scalar();else ggml();}double ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count()/iterations;std::printf("%s rows=%d width=%d tokens=%d iterations=%d mean_ms=%.6f\n",path==0?"portable_q8_1":"ggml_arm_q8_0",rows,n,nt,iterations,ms);}
 std::printf("activation contracts differ; output relative RMS=%.9f; finite=PASS\n",std::sqrt(squared/mag));
}
