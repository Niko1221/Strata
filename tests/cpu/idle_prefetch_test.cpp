#include "strata/kernels/cpu/idle_prefetch.hpp"
#include "strata/kernels/cpu/expert_layout.hpp"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <thread>
using namespace strata::kernels::cpu;

static void put_env(const char* name, const char* value, int) {
#ifdef _WIN32
    _putenv_s(name, value);
#else
    setenv(name, value, 1);
#endif
}

namespace strata::core {void release_gpu_waits(FILE*) {}}
int main(){
 if(!cpu_avx2_ok())return 77;
 put_env("STRATA_NUMA_POOL","0",1);
 const int N=32,NE=8;std::vector<std::vector<uint16_t>> router(2,std::vector<uint16_t>(N*NE));
 for(auto& layer:router)for(int e=0;e<NE;e++)for(int i=0;i<N;i++)layer[e*N+i]=uint16_t(0x3f80+e*4); // positive, increasing BF16
 std::vector<std::vector<uint8_t>> mem(NE,std::vector<uint8_t>(10001,7));
 std::vector<WarmBlob> blobs(NE);for(int e=0;e<NE;e++)blobs[e]={mem[e].data()+1,10000,4101};
 blobs[7]={};blobs[5]={}; // two GPU residents
 float x[N];for(float&v:x)v=1;
 for(int n:{1,2,4,8}){
  ExpertPool pool(3,false,true,PoolAffinity::All,96);IdleExpertPrefetch warm(pool,n);std::string err;
  if(!warm.init(router,N,NE,8,err))return 1;
  for(int j=0;j<20;j++){
   for(float&v:x)v=1;
   const auto before=warm.read_lines();
   warm.start(1,x,1,blobs);
   // start owns the quantized activation; later host writes must not change the prediction.
   for(float&v:x)v=-1;
   const auto deadline=std::chrono::steady_clock::now()+std::chrono::seconds(3);
   while(warm.read_lines()==before&&std::chrono::steady_clock::now()<deadline)
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
   warm.stop();
   if(warm.read_lines()==before){std::fprintf(stderr,"prefetch made no progress\n");return 7;}
   const int count=std::min(n,4);
   auto ids=warm.selected();if(ids.size()!=size_t(count))return 2;
   const int expected[]={6,4,3,2};for(int i=0;i<count;i++)if(ids[i]!=expected[i])return 3;
  }
  if(!warm.read_lines()||warm.predictions()!=20)return 4;
 }
 for(const auto& bytes:mem)for(uint8_t value:bytes)if(value!=7)return 6;
 {ExpertPool p(3,false,true);IdleExpertPrefetch w(p,4);std::string err;if(w.init(router,31,NE,8,err))return 5;}
 puts("PASS q4 predictor order, GPU exclusions, 1/2/4 selection, mixed unaligned blobs and cancellable warm reads");
}
