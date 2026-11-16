#include "strata/kernels/cpu/pool.hpp"
#include <atomic>
#include <thread>
#include <chrono>
#include <cstdio>
#include <cstdlib>
using namespace strata::kernels::cpu;

static void put_env(const char* name, const char* value, int) {
#ifdef _WIN32
    _putenv_s(name, value);
#else
    setenv(name, value, 1);
#endif
}

namespace strata::core { void release_gpu_waits(FILE*) {} }
struct C { std::atomic<int> entered{0},left{0}; };
void work(void* p,int,int,const std::atomic<bool>& stop) {
 auto& c=*(C*)p;c.entered.fetch_add(1);
 while(!stop.load(std::memory_order_acquire))std::this_thread::yield();
 c.left.fetch_add(1);
}
int main(){
 put_env("STRATA_POOL_SPIN_US","0",1);
 put_env("STRATA_NUMA_POOL","0",1);
 for(int n:{1,3,16}){
  ExpertPool p(n,false,true,PoolAffinity::All,96);
  for(int t=0;t<300;t++){
   C c;p.idle_start(work,&c);
   if(n>1&&t%4==0){auto end=std::chrono::steady_clock::now()+std::chrono::seconds(2);while(!c.entered.load()&&std::chrono::steady_clock::now()<end)std::this_thread::yield();if(!c.entered.load())return 2;}
   // All public foreground entries must join the old idle context, even for an empty batch.
   NativeFmt fmt{};
   switch(t%5){
    case 0:p.run(nullptr,0);break;
    case 1:p.run_split(nullptr,0);break;
    case 2:p.run_split_multi(nullptr,0);break;
    case 3:p.run_split_multi_native(fmt,nullptr,0);break;
    default:p.idle_stop();break;
   }
   int a=c.entered.load(),b=c.left.load();
   if(a!=b){fprintf(stderr,"unjoined background job %d/%d\n",a,b);return 1;}
   p.idle_stop();
  }
 }
 C c;{ExpertPool p(3,false,true);p.idle_start(work,&c);std::this_thread::sleep_for(std::chrono::milliseconds(3));}
 if(c.entered.load()!=c.left.load())return 3;
 puts("PASS 900 start/cancel/restart cycles, sleep/wake, 1/3/16 workers and destructor join");
}
