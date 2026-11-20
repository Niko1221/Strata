#include "weight_arena.hpp"
#include "kv.hpp"
#include "layer_fence.hpp"
#include "resources.hpp"
#include "startup.cuh"
#include <cmath>
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <limits>
#include <memory>
#include <vector>
namespace Q=strata::qwen38;
static void demand(bool b,const char* text){if(!b)throw std::runtime_error(text);}
template<class F>void rejects(F f){bool bad=false;try{f();}catch(const std::exception&){bad=true;}demand(bad,"expected rejection");}
struct Stream{cudaStream_t s;Stream(){Q::ck(cudaStreamCreate(&s));}~Stream(){cudaStreamDestroy(s);}};
struct Event{cudaEvent_t e;Event(){Q::ck(cudaEventCreate(&e));}~Event(){cudaEventDestroy(e);}};
static size_t free_bytes(){size_t free,total;Q::ck(cudaMemGetInfo(&free,&total));return free;}
static void reserve(){Q::host_resources();demand(free_bytes()>Q::GPU_RESERVE+512*1024*1024,"hardware fixture reserve");}
static void layout_tests(){
    Q::WeightLayout a;a.append(17);a.append(513);a.append(4);
    demand(a.entries()[1].offset==256&&a.entries()[2].offset==1024&&a.bytes()==1028&&a.payload()==534,"aligned exact spans");
    rejects([]{Q::WeightLayout x;x.append(0);});
    rejects([]{Q::WeightLayout x;x.append(SIZE_MAX);x.append(1);});
    rejects([]{Q::WeightLayout::f32_bytes(SIZE_MAX/4+1);});
    demand(Q::WeightLayout::f32_bytes(257)==1028,"F32 conversion span");
    rejects([]{Q::WeightArena empty(Q::WeightLayout{});});
    auto cfg=Q::KvConfig::admit("f16");Q::KvBudget budget{a.bytes(),4096,16384,1024,144*1024*1024};
    auto fit=Q::select_kv(cfg,33,2,budget,UINT64_MAX,Q::GPU_RESERVE);
    demand(Q::select_kv(cfg,33,2,budget,fit.required,Q::GPU_RESERVE).required==fit.required,"arena exact-fit budget");
    rejects([&]{Q::select_kv(cfg,33,2,budget,fit.required-1,Q::GPU_RESERVE);});
    std::atomic<bool> stop{false};int fenced=0,enqueued=0;Q::LayerFence fences(8);
    try{for(int i=0;i<64;++i){fences.before(stop);++enqueued;fences.after(i,stop,[&]{++fenced;stop=true;});}}catch(const Q::Cancelled&){}
    demand(enqueued==8&&fenced==1,"cancel at first group boundary");
    bool cancelled=false;try{fences.before(stop);}catch(const Q::Cancelled&){cancelled=true;}demand(cancelled,"cancel before enqueue");
    rejects([]{Q::LayerFence f(9);});rejects([]{Q::LayerFence f(0);});
    std::puts("PASS checked weight layout, overflow, exact F32 spans and 8-layer cancellation boundary");
}
static void allocations(){
    // Hundreds of irregular spans exercise backing overhead without large models.
    for(int count:{31,129,513}){
        reserve();Q::WeightLayout layout;for(int i=0;i<count;++i)layout.append((count==129?1048576:32768)+(i%7)*173);
        size_t before=free_bytes(),individual=0,arena_delta=0;
        {std::vector<Q::Buffer> old;for(const auto& e:layout.entries())old.emplace_back(e.bytes);individual=before-free_bytes();}
        before=free_bytes();
        {Q::WeightArena arena(layout);arena_delta=before-free_bytes();
            demand(arena.bytes()==layout.bytes(),"arena allocation extent");
            for(size_t i=0;i<layout.entries().size();++i){auto v=arena.view(i);demand(uintptr_t(v.data)%256==0&&v.bytes==layout.entries()[i].bytes,"arena view alignment/extent");
                Q::ck(cudaMemset(v.data,int(i%251),v.bytes));}
            for(size_t i=0;i<layout.entries().size();++i){auto v=arena.view(i);std::vector<uint8_t> bytes(v.bytes);Q::ck(cudaMemcpy(bytes.data(),v.data,v.bytes,cudaMemcpyDeviceToHost));
                demand(std::all_of(bytes.begin(),bytes.end(),[&](uint8_t b){return b==i%251;}),"arena overlapping spans");}
            rejects([&]{arena.view(layout.entries().size());});
        }
        demand(arena_delta<=layout.bytes()+4*1024*1024,"arena unexplained backing allocation");
        // Exception after allocation must unwind the sole owner. Small pool
        // residency may persist, so record deltas and bound it, not exact equality.
        before=free_bytes();rejects([&]{Q::WeightArena failing(layout);throw std::runtime_error("injected upload failure");});
        demand(free_bytes()+4*1024*1024>=before,"failed upload leaked arena");
        std::printf("ALLOC count=%d payload=%zu arena_bytes=%zu padding=%zu individual_delta=%zu arena_delta=%zu\n",count,layout.payload(),layout.bytes(),layout.bytes()-layout.payload(),individual,arena_delta);
    }
}
static void conversion(cudaStream_t s){
    Q::WeightLayout layout;layout.append(256*2);layout.append(Q::WeightLayout::f32_bytes(256));layout.append(17);
    Q::WeightArena arena(layout);auto raw=arena.view(0),converted=arena.view(1),tail=arena.view(2);
    std::vector<uint16_t> input(256);for(int i=0;i<256;++i)input[i]=i%2?0x4000:0x3c00;
    Q::ck(cudaMemcpy(raw.data,input.data(),raw.bytes,cudaMemcpyHostToDevice));Q::ck(cudaMemset(tail.data,0x5a,tail.bytes));
    Q::decode(1,raw.data,converted.as(),256,s);Q::ck(cudaStreamSynchronize(s));
    std::vector<float> result(256);Q::ck(cudaMemcpy(result.data(),converted.data,converted.bytes,cudaMemcpyDeviceToHost));
    for(int i=0;i<256;++i)demand(result[i]==(i%2?2.f:1.f),"arena F16 auxiliary conversion");
    std::vector<uint8_t> guard(tail.bytes);Q::ck(cudaMemcpy(guard.data(),tail.data,tail.bytes,cudaMemcpyDeviceToHost));
    demand(std::all_of(guard.begin(),guard.end(),[](uint8_t b){return b==0x5a;}),"conversion exceeds exact span");
    Q::Buffer x(256*4),y(4);std::vector<float> ones(256,1.f);Q::ck(cudaMemcpy(x.data,ones.data(),x.bytes,cudaMemcpyHostToDevice));
    Q::matvec(0,converted.data,x.as(),y.as(),256,1,s);float sum=0;Q::ck(cudaMemcpyAsync(&sum,y.data,4,cudaMemcpyDeviceToHost,s));Q::ck(cudaStreamSynchronize(s));demand(sum==384.f,"arena converted matvec");
    std::puts("PASS arena production F16-to-F32 auxiliary decode/matvec and adjacent-span sentinel");
}
struct Inputs{
    Q::Buffer q{6144*4},k{1024*4},v{1024*4},gate{6144*4},out{6144*4},scores;
    Inputs(int capacity):scores(size_t(24)*capacity*4){}
    void fill(int position){
        for(auto b:{&q,&k,&v,&gate}){std::vector<float> h(b->bytes/4);for(size_t i=0;i<h.size();++i)h[i]=float(std::sin(i*.019+position*.13));Q::ck(cudaMemcpy(b->data,h.data(),b->bytes,cudaMemcpyHostToDevice));}
    }
    std::vector<float> output(){std::vector<float> h(6144);Q::ck(cudaMemcpy(h.data(),out.data,out.bytes,cudaMemcpyDeviceToHost));return h;}
    void enqueue(Q::KvCache& c,int layer,int pos){c.enqueue(layer,pos,q.as(),k.as(),v.as(),gate.as(),out.as(),scores.as());}
};
static void differential(const Q::KvConfig& cfg,cudaStream_t s){
    reserve();
    for(int capacity:{9,33}){
        Q::KvCache fast(cfg,capacity,2,s),reference(cfg,capacity,2,s,true);Inputs a(capacity),b(capacity);
        for(int pos=0;pos<capacity;++pos){a.fill(pos);b.fill(pos);
            for(int layer=0;layer<2;++layer){a.enqueue(fast,layer,pos);b.enqueue(reference,layer,pos);}
            fast.check();reference.check();demand(a.output()==b.output(),"conservative/grouped output is not bit exact");}
        auto bad=[&](float value){fast.reset();a.fill(0);Q::ck(cudaMemcpy(a.v.data,&value,4,cudaMemcpyHostToDevice));
            a.enqueue(fast,0,0);a.enqueue(fast,1,0);rejects([&]{fast.check();});rejects([&]{a.enqueue(fast,0,1);});
            fast.reset();a.fill(0);a.enqueue(fast,0,0);fast.check();};
        bad(std::numeric_limits<float>::quiet_NaN());bad(1e20f);
        fast.reset();reference.reset();a.fill(0);b.fill(0);a.enqueue(fast,0,0);b.enqueue(reference,0,0);fast.check();reference.check();demand(a.output()==b.output(),"reset reuse not exact");
        std::printf("PASS grouped differential mode=%s view=%s ctx=%d exact_output=true invalid/overflow/sticky/reset=pass\n",cfg.name(),cfg.view_name(),capacity);
    }
}
static void benchmark(const Q::KvConfig& cfg,cudaStream_t s){
    reserve();Inputs in(33);in.fill(2);
    for(bool conservative:{true,false}){
        Q::KvCache cache(cfg,33,16,s,conservative);std::vector<float> times;std::vector<double> walls;
        uint64_t fences=0;Event begin,end;
        for(int rep=0;rep<9;++rep){cache.reset();uint64_t before=cache.fences();auto start=std::chrono::steady_clock::now();Q::ck(cudaEventRecord(begin.e,s));
            for(int pos=0;pos<8;++pos){for(int layer=0;layer<16;++layer){in.enqueue(cache,layer,pos);if(conservative||(layer+1)%2==0)cache.check();}}
            Q::ck(cudaEventRecord(end.e,s));Q::ck(cudaEventSynchronize(end.e));float ms=0;Q::ck(cudaEventElapsedTime(&ms,begin.e,end.e));
            double wall=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count();
            fences=cache.fences()-before;if(rep>=2){times.push_back(ms);walls.push_back(wall);}}
        std::sort(times.begin(),times.end());std::sort(walls.begin(),walls.end());
        std::printf("TIMING mode=%s conservative=%d warmup=2 runs=7 positions=8 layers=16 host_fences=%llu cuda_ms_min=%.6f median=%.6f max=%.6f wall_median_ms=%.6f\n",cfg.name(),conservative,(unsigned long long)fences,times.front(),times[3],times.back(),walls[3]);
    }
}
int main(){try{
    layout_tests();Q::host_resources();Q::ck(cudaSetDevice(0));Stream stream;size_t before=free_bytes();Q::warmup_runtime(stream.s);size_t after=free_bytes();
    std::printf("PRELOAD free_before=%zu free_after=%zu\n",before,after);allocations();conversion(stream.s);
    for(const char* mode:{"f16","k8v4","spin8_k8v4","f16_turbo2","f16_turbo3","f16_turbo4"}){
        for(const char* view:{"vector","plus","minus"}){bool spin=std::string(mode)=="spin8_k8v4";if(!spin&&std::string(view)!="vector")continue;
            auto cfg=Q::KvConfig::admit(mode,spin?view:"");differential(cfg,stream.s);if(std::string(view)=="vector")benchmark(cfg,stream.s);}}
    std::puts("PASS hardware allocation and grouped production KV suite");return 0;
}catch(const std::exception& e){std::fprintf(stderr,"FAIL %s\n",e.what());return 1;}}
