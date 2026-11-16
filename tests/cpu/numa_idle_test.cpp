#include "strata/kernels/cpu/idle_prefetch.hpp"
#include "strata/kernels/cpu/expert_layout.hpp"
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <thread>
#include <vector>
using namespace strata::kernels::cpu;

static void put_env(const char* name, const char* value, int) {
#ifdef _WIN32
    _putenv_s(name, value);
#else
    setenv(name, value, 1);
#endif
}

namespace strata::core { void release_gpu_waits(FILE*) {} }
struct Context {std::atomic<int> entered{0},left{0};};
void idle(void* p,int,int,const std::atomic<bool>& stop) {
 auto& c=*static_cast<Context*>(p);++c.entered;
 while(!stop.load()) std::this_thread::yield();
 ++c.left;
}
int main(int argc,char** argv){
 const bool prediction=argc>1;
 if(prediction&&!cpu_avx2_ok())return 77;
 put_env("STRATA_NUMA_POOL","1",1);put_env("STRATA_NUMA_SPLIT","test",1);
 put_env("STRATA_POOL_SPIN_US","0",1);
 for(int workers:{2,3,16}) for(bool host:{false,true}) {
  ExpertPool pool(workers,false,host,PoolAffinity::All,96);
  if(pool.numa_slices()!=2)return 1;
  if(!prediction) {
   for(int j=0;j<150;++j){
    Context c;pool.idle_start(idle,&c);
    if(j%3==0){const auto end=std::chrono::steady_clock::now()+std::chrono::seconds(1);
     while(c.entered.load()!=workers&&std::chrono::steady_clock::now()<end)std::this_thread::yield();
     if(c.entered!=workers){std::fprintf(stderr,"NUMA idle workers never ran: %d/%d\n",int(c.entered),workers);pool.idle_stop();return 2;}}
    pool.idle_stop();if(c.entered!=c.left)return 3;
   }
  } else {
   const int N=32,NE=8;
   std::vector<std::vector<uint16_t>> router(1,std::vector<uint16_t>(N*NE));
   for(int e=0;e<NE;e++)for(int i=0;i<N;i++)router[0][e*N+i]=uint16_t(0x3f80+e*4);
   std::vector<std::vector<uint8_t>> mem(NE,std::vector<uint8_t>(12288,7));
   std::vector<WarmBlob> blobs(NE);
   for(int e=0;e<NE;e++)blobs[e]={mem[e].data(),12288,8192,4096};
   IdleExpertPrefetch prefetch(pool,4);std::string error;
   if(!prefetch.init(router,N,NE,8,error)){std::fprintf(stderr,"NUMA predictor rejected: %s\n",error.c_str());return 4;}
   float x[N];for(auto& v:x)v=1;
   for(int i=0;i<30;i++){
    prefetch.start(0,x,1,blobs);
    const auto deadline=std::chrono::steady_clock::now()+std::chrono::seconds(3);
    while(prefetch.read_lines()<uint64_t(i+1)*4*192&&std::chrono::steady_clock::now()<deadline)
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
    prefetch.stop();
    auto ids=prefetch.selected();if(ids!=std::vector<int>({7,6,5,4}))return 5;
   }
   if(prefetch.predictions()!=30||prefetch.read_lines()!=30ULL*4*192)return 6;
  }
 }
 std::puts(prediction?"PASS NUMA prediction and warm reads":"PASS 900 NUMA idle start/cancel/join cycles");
}
