
#include "strata/hy4flash_gguf/cpu_experts.hpp"
#include "strata/hy4flash_gguf/cpu/native_expert.hpp"
#include "strata/hy4flash_gguf/cpu/native_q81.hpp"
#include "strata/hy4flash_gguf/cpu/expert_layout.hpp"
#include "ggml.h"
#include <vector>
#include <array>
#include <cstdio>
#include <cstring>
#include <cmath>
#include <limits>
namespace c=strata::hy4_flash::cpu;
using strata::hy4_flash::CpuExperts;
// Compare the production SwiGLU row kernel to the scalar formula on both sides of each bound.
static bool swiglu_rows_match() {
    constexpr int width = 64, rows = 32;
    c::NativeFmt fmt{};
    std::string err;
    if (!c::native_fmt(GGML_TYPE_Q8_0, GGML_TYPE_Q8_0, width, rows, fmt, err)) {
        std::puts(err.c_str());
        return false;
    }
    std::vector<uint8_t> weights(fmt.bytes), act(fmt.act_bytes);
    // Each Q8_0 block has unit scale. The 127 entries occupy different lanes.
    std::array<float, width> row{};
    row[0] = 1.f;
    row[2] = 127.f;
    c::native_quant_act(fmt, row.data(), act.data());
    constexpr float dots[] = {-32.f, -10.f, 4.f, 10.f, 32.f};
    row[1] = 127.f;
    row[2] = 0.f;
    for (int r = 0; r < rows; ++r) {
        row[0] = dots[r % 5];
        ggml_quantize_chunk(GGML_TYPE_Q8_0, row.data(), weights.data() + r * fmt.gu_row, 0, 1, width, nullptr);
        row[0] = dots[(r / 5) % 5];
        ggml_quantize_chunk(GGML_TYPE_Q8_0, row.data(), weights.data() + fmt.up_off + r * fmt.gu_row, 0, 1, width, nullptr);
    }
    const void* input[] = {act.data()};
    std::array<float, rows> out{};
    float* output[] = {out.data()};
    for (float limit : {0.f, 10.f}) {
        fmt.swiglu_clamp = limit;
        c::native_gu_rows(fmt, weights.data(), input, 1, output, 0, rows);
        for (int r = 0; r < rows; ++r) {
            float gate = dots[r % 5], up = dots[(r / 5) % 5];
            if (limit > 0.f) {
                gate = std::fmin(gate, limit);
                up = std::fmax(-limit, std::fmin(up, limit));
            }
            const float expected = gate / (1.f + std::exp(-gate)) * up;
            if (!(std::fabs(out[r] - expected) <= 1e-5f * std::fabs(expected) + 1e-12f)) {
                std::fprintf(stderr, "SwiGLU row %d limit %g: got %g, expected %g\n", r, limit, out[r], expected);
                return false;
            }
        }
    }
    return true;
}

int main(){
 int fail=0,cases=0; auto check=[&](bool ok,const char* msg){std::printf("%s %s\n",ok?"PASS":"FAIL",msg);if(!ok)++fail;};
 check(swiglu_rows_match(),"Q8_0 SwiGLU rows match the scalar formula");
 CpuExperts pool(4); std::string err;
 std::printf("CPU_AVX2_AVAILABLE=%d\n",int(strata::kernels::cpu::cpu_avx2_ok()));
 const int dim=1536,ff=1536;std::vector<float>x(dim);for(int i=0;i<dim;++i)x[i]=std::sin(i*.051f)*1.7f;
 for(bool compat:{false,true})for(int gt:{12,14})for(int dt:{12,14})for(float clamp:{0.f,10.f}){
  c::NativeFmt f{};if(!CpuExperts::make_format(gt,dt,dim,ff,clamp,f,err,compat)){std::puts(err.c_str());return 2;}
  std::vector<uint8_t>blob(f.bytes),reverse(f.bytes);
  std::vector<float>row(dim);uint32_t seed=42;
  for(int plane=0;plane<3;++plane)for(int r=0;r<ff;++r){
   for(int i=0;i<dim;++i){seed=1664525u*seed+1013904223u;row[i]=float(int(seed>>8)%2001-1000)*.008f;}
   const int type=plane==2?dt:gt;size_t off=plane==0?0:plane==1?f.up_off:f.down_off;
   size_t rb=plane==2?f.d_row:f.gu_row;
   ggml_quantize_chunk((ggml_type)type,row.data(),blob.data()+off+r*rb,0,1,dim,nullptr);
  }
  // Explicit planes deliberately reverse address ordering: down < up < gate.
  const size_t down_bytes=f.d_row*dim,gu_bytes=f.gu_row*ff;
  std::memcpy(reverse.data(),blob.data()+f.down_off,down_bytes);
  std::memcpy(reverse.data()+down_bytes,blob.data()+f.up_off,gu_bytes);
  std::memcpy(reverse.data()+down_bytes+gu_bytes,blob.data(),gu_bytes);
  std::array<const uint8_t*,16>g{},u{},d{};int ranks[3]={15,2,7};
  for(int e=0;e<3;++e){g[e]=reverse.data()+down_bytes+gu_bytes;u[e]=reverse.data()+down_bytes;d[e]=reverse.data();}
  std::vector<float>out(16*dim,12345.f),want(dim),hidden(ff);
  std::array<uint8_t,c::kNativeActBytes>act{};std::array<uint8_t,c::kNativeHBytesMax>hq{};
  c::native_quant_act(f,x.data(),act.data());
  if(compat){
   bool equal=true;
   for(int plane=0;plane<3;++plane)for(int r=0;r<64;++r){
    int type=plane==2?dt:gt;size_t off=plane==0?0:plane==1?f.up_off:f.down_off;
    const uint8_t* w=blob.data()+off+r*(plane==2?f.d_row:f.gu_row);
    float a=c::native_dot_q81_scalar(type,w,act.data(),dim);
    float b=c::native_dot_q81_compatible(type,w,act.data(),dim);
    equal &= std::memcmp(&a,&b,4)==0;
#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
    if(strata::kernels::cpu::cpu_avx2_ok()){float v=c::native_dot_q81_avx2(type,w,act.data(),dim);equal &= std::memcmp(&a,&v,4)==0;}
#endif
   }
   check(equal,"Q8_1 scalar and runtime SIMD integer-dot paths bitwise");
  }
  const void* ap[]={act.data()};float* hp[]={hidden.data()};
  c::native_gu_rows_invariant(f,blob.data(),ap,1,hp,0,ff);c::native_quant_h(f,hidden.data(),hq.data());
  const void* qp[]={hq.data()};float* op[]={want.data()};c::native_down_rows_invariant(f,blob.data(),qp,1,op,0,dim);
  bool ok=pool.compute(f,g.data(),u.data(),d.data(),ranks,3,x.data(),out.data(),err);
  for(int r=0;r<16;++r){bool selected=r==15||r==2||r==7;
   if(selected)ok &= std::memcmp(out.data()+r*dim,want.data(),dim*4)==0;
   else for(int i=0;i<dim;++i)ok &= out[r*dim+i]==12345.f;
  }
  if(clamp>0.f){bool bounded=true;for(float h:hidden)bounded &= std::fabs(h)<=100.f;check(bounded,"clamped hidden values stay within bound");}
  // Full16 reuses the same planes without copying weights; every rank must be overwritten.
  for(int e=0;e<16;++e){g[e]=reverse.data()+down_bytes+gu_bytes;u[e]=reverse.data()+down_bytes;d[e]=reverse.data();}
  int allr[16];for(int e=0;e<16;++e)allr[e]=15-e;
  std::vector<float>full(16*dim,-999.f);
  bool fullok=pool.compute(f,g.data(),u.data(),d.data(),allr,16,x.data(),full.data(),err);
  for(int r=0;r<16;++r)fullok &= std::memcmp(full.data()+r*dim,want.data(),dim*4)==0;
  check(fullok,"all16 reversed ranks match contiguous CPU oracle");
  char msg[100];std::snprintf(msg,sizeof(msg),"Q%d/Q%d clamp%.0f compat%d reverse planes CPU bitwise",gt,dt,clamp,int(compat));check(ok,msg);++cases;
  // The four positions for three drafts have different activations and route the
  // same expert at different ranks. Unselected parts must survive the scatter.
  constexpr int nt=4;
  std::vector<float> input(nt*dim),reference(nt*16*dim,12345.f),window=reference;
  CpuExperts::WindowJob window_jobs[2]{};
  // Give the second expert distinct weights so exchanging experts fails the oracle too.
  std::swap(g[1],u[1]);
  for(int j=0;j<2;++j) {
   window_jobs[j].plane[0]=g[j]; window_jobs[j].plane[1]=u[j]; window_jobs[j].plane[2]=d[j];
  }
  bool window_ok=true;
  for(int t=0;t<nt;++t) {
   for(int i=0;i<dim;++i)input[t*dim+i]=x[i]+float(t)*0.1f;
   int row_ranks[2]={2+t,15-t};
   int count=(t==1?1:2);
   for(int j=0;j<count;++j)window_jobs[j].entries[window_jobs[j].count++]=t*16+row_ranks[j];
   window_ok &= pool.compute(f,g.data(),u.data(),d.data(),row_ranks,count,
                             input.data()+t*dim,reference.data()+t*16*dim,err);
  }
  window_ok &= pool.compute_window(f,window_jobs,2,nt,input.data(),window.data(),err);
  check(window_ok && std::memcmp(window.data(),reference.data(),window.size()*sizeof(float))==0,
        "four-row CPU window matches four independent rows, including untouched ranks");
  if(cases==1) {
   auto rotated=f;
   rotated.had_gu=true;
   check(!pool.compute_window(rotated,window_jobs,2,nt,input.data(),window.data(),err),
         "unsupported window layout rejected");
  }
  auto saved=out;int dup[2]={2,2};check(!pool.compute(f,g.data(),u.data(),d.data(),dup,2,x.data(),out.data(),err)&&out==saved,"duplicate rank rejected before write");
  check(pool.compute(f,nullptr,nullptr,nullptr,nullptr,0,nullptr,nullptr,err),"zero jobs is a no-op");
 }
 c::NativeFmt bad{};check(!CpuExperts::make_format(12,14,1536,1792,10,bad,err),"oversized native FF rejected");
 std::printf("SUMMARY cases=%d fail=%d legacy_Q8K_and_compatible_Q81_CPU_oracles\n",cases,fail);return fail?1:0;
}
