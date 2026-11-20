// Extracted production functions with CPU fake APIs. No CUDA DLL/model is loaded.
#include "strata/perf131k/blob_failure.hpp"
#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <memory>
#include <mutex>
#include <new>
#include <string>
#include <system_error>
#include <thread>
#include <unordered_map>
#include <vector>
long fail_at = -1, allocations = 0;
bool reject_after_read = false;
unsigned issued = 0, waited = 0, events = 0, event_closes = 0;
int arrays_live = 0, pins_live = 0;
void allocation_point() {
    if (reject_after_read && issued > waited) throw std::bad_alloc();
    if (fail_at >= 0 && allocations++ == fail_at) throw std::bad_alloc();
}
void* operator new(size_t n) { allocation_point(); if (auto p=std::malloc(n ? n:1)) return p; throw std::bad_alloc(); }
void operator delete(void* p) noexcept { std::free(p); }
void operator delete(void* p,size_t) noexcept { std::free(p); }
void* operator new[](size_t n) { allocation_point(); if(auto p=std::malloc(n ? n:1)) { ++arrays_live; return p; } throw std::bad_alloc(); }
void operator delete[](void* p) noexcept { if(p) --arrays_live; std::free(p); }
void operator delete[](void* p,size_t) noexcept { if(p) --arrays_live; std::free(p); }
using DWORD=unsigned long; using HANDLE=void*;
struct OVERLAPPED { DWORD Offset=0,OffsetHigh=0; HANDLE hEvent=nullptr; };
constexpr int TRUE=1,FALSE=0,ERROR_IO_PENDING=997,MEM_RELEASE=1,MEM_COMMIT=2,MEM_RESERVE=4,PAGE_READWRITE=8;
DWORD GetLastError() { return ERROR_IO_PENDING; }
void* VirtualAlloc(void*,size_t n,int,int) { return std::calloc(1,n); }
int VirtualFree(void* p,int,int) { std::free(p); return 1; }
HANDLE CreateEventW(void*,int,int,void*) { ++events; return reinterpret_cast<HANDLE>(uintptr_t(3)); }
int CloseHandle(HANDLE) { ++event_closes; return 1; }
int ReadFile(HANDLE,void*,DWORD,void*,OVERLAPPED*) { ++issued; return 0; }
int GetOverlappedResult(HANDLE,OVERLAPPED*,DWORD* got,int) { ++waited; *got=4096; return 1; }
using cudaError_t=int; using cudaEvent_t=void*;
constexpr int cudaSuccess=0,cudaHostAllocDefault=0,cudaErrorNotReady=34;
int cudaHostAlloc(void** p,size_t n,int) { *p=std::malloc(n); if(*p)++pins_live; return *p?0:1; }
int cudaFreeHost(void* p) { if(p)--pins_live; std::free(p); return 0; }
int cudaGetLastError() { return 0; }
const char* cudaGetErrorString(int) { return "fake"; }
int cudaEventQuery(void*) { return 0; }
int cudaEventSynchronize(void*) { return 0; }
bool available_memory_bytes(uint64_t& n) { n=UINT64_MAX; return true; }
constexpr bool STAGE_PIN_DEFAULT=true;
constexpr uint64_t kStagePinFloor=1ull<<30, kStageAge=3;
namespace strata::perf131k {
enum class Stage { expert_direct_read_host };
struct Scope { explicit Scope(Stage) {} };
}
namespace strata::kernels::cpu {
struct Layout { bool native=true; struct Fmt { int tail_off=0; }; std::vector<Fmt> fmt{2}; };
const Layout& expert_layout() { static Layout value; return value; }
}
struct FileExpertSource {
    struct Fill { size_t v; int64_t layer,e; uint8_t* dst; };
    struct Map { uint8_t* base=nullptr; };
    struct StageBufFree { bool pinned=false; void operator()(uint8_t* p) const noexcept; };
    int64_t n_layers_=2,n_expert_=512;
    uint64_t stage_blob_=16, stage_seq_=0,epoch_=0;
    std::vector<uint8_t*> role_ptr_;
    std::vector<uint64_t> layer_offsets_{0,8ull<<20},layer_blob_bytes_{16,16},role_bytes_;
    std::vector<int> role_file_; std::vector<Map> maps_;
    std::vector<void*> direct_{reinterpret_cast<void*>(uintptr_t(1))};
    std::vector<std::unique_ptr<uint8_t[],StageBufFree>> stage_buf_;
    std::vector<int64_t> stage_key_;
    std::vector<uint64_t> stage_epoch_,stage_used_;
    std::vector<char> stage_busy_,stage_pf_,stage_ev_live_;
    std::vector<void*> stage_ev_;
    std::unordered_map<int64_t,size_t> stage_of_;
    std::unordered_map<const uint8_t*,size_t> stage_idx_;
    std::atomic<uint64_t> io_pf_used_{0},io_pf_used_bytes_{0},io_pf_unused_{0},stage_fence_waits_{0};
    bool stage_pin_said_=false,stage_pin_failed_said_=false,stage_grew_=false;
    bool unmapped_=false, io_pf_=false,complement_ready_=false;
    std::mutex stage_mu_;
    std::condition_variable stage_cv_;
    std::atomic<uint64_t> file_us_{0},file_read_bytes_{0},file_blob_bytes_{0};
    std::vector<int64_t> zc_keys_;
    std::vector<uint8_t*> override_;
    std::unique_ptr<std::atomic<uint32_t>[]> warm_stamp_;
    std::atomic<uint64_t> warm_hits_{0};
    const uint8_t* resident_blob(size_t) const { return nullptr; }
    uint64_t cached_bytes(int64_t,int64_t,uint64_t& total,bool) const { total=16;return 0; }
    void claim_todo(int64_t,const int64_t*,int64_t,std::vector<Fill>&,uint8_t*,bool);
    bool read_direct(const Fill*,size_t) const;
    bool copy_from_files(int64_t,int64_t,uint8_t*) { throw std::bad_alloc(); }
    bool fill_stage(size_t,int64_t,int64_t,uint8_t*);
    void publish_stage(size_t,int64_t,bool,double);
    bool claim_stage(int64_t,size_t&,bool&,bool=false);
    bool coherent() const {
        size_t n=stage_buf_.size();
        if(stage_ev_.size()!=n || stage_ev_live_.size()!=n || stage_key_.size()!=n || stage_epoch_.size()!=n ||
           stage_used_.size()!=n || stage_busy_.size()!=n || stage_pf_.size()!=n || stage_idx_.size()!=n)return false;
        for(auto& [k,v]:stage_of_) if(v>=n || stage_key_[v]!=k)return false;
        for(size_t i=0;i<n;++i) { auto it=stage_idx_.find(stage_buf_[i].get()); if(it==stage_idx_.end() || it->second!=i)return false; }
        return true;
    }
};
// @WORKERS@
// @COMP_WORKERS@
// @COMP_FAILURE@
// @FILL@
// @PUBLISH@
// @STAGE_FREE@
// @TODO@
// @CLAIM@
// @DIRECT@
struct GrowCache {
    bool success=false;int64_t mapped=128;
    bool grow(int64_t,std::string& error) { error="injected";return success; }
    int64_t mapped_bytes() const { return mapped; }
};
unsigned uploads=0;
bool grow_consumer(GrowCache& xcache,std::string& e) {
    int64_t mapped=0,free_b=128,want_free=0;std::string gerr,note;
    // @VRAM_GROW@
    ++uploads;
    return true;
}
int terminal_consumer(GrowCache& xcache) {
    // @VRAM_TERMINAL@
    return 0;
}
#define CHECK(x) do { if(!(x)) { std::fprintf(stderr,"line %d: %s\n",__LINE__,#x);return 1; } } while(0)
int main(int argc,char** argv) {
    if(argc<2)return 2;
    std::string mode=argv[1];
    if(mode=="stage") {
        unsigned failures=0,successes=0;
        for(int point=0;point<32;++point) {
            {
                FileExpertSource src; size_t v=99; bool fill=false;
                allocations=0;fail_at=point;
                bool hit=src.claim_stage(0,v,fill);
                fail_at=-1;
                CHECK(!hit && src.coherent());
                if(fill) { ++successes;CHECK(src.stage_buf_.size()==1 && src.stage_busy_[v]); }
                else { ++failures;CHECK(src.stage_buf_.empty() && src.stage_of_.empty()); }
            }
            CHECK(arrays_live==0 && pins_live==0);
        }
        FileExpertSource src; size_t v=0;bool fill=false;
        CHECK(!src.claim_stage(0,v,fill) && fill);
        src.stage_busy_[v]=0;src.epoch_=10;
        fail_at=0;allocations=0;
        CHECK(!src.claim_stage(1,v,fill));fail_at=-1;
        CHECK(!fill && src.coherent() && src.stage_of_.count(0)==1 && src.stage_of_.count(1)==0);
        CHECK(!src.claim_stage(1,v,fill) && fill && src.coherent());
        CHECK(src.stage_buf_.size()==1 && src.stage_of_.count(0)==0);
        CHECK(failures>=10 && successes>0);
        std::printf("PASS stage allocation_failures=%u successes=%u reuse_preserved=1\n",failures,successes);
    } else if(mode=="retention") {
        const char* pin_env=std::getenv("STRATA_STAGE_PIN");
        CHECK(pin_env!=nullptr);
        const bool expect_pin=std::atoi(pin_env)!=0;
        {
            FileExpertSource src;
            for(unsigned i=0;i<1024;++i) {
                size_t v=0;bool fill=false;
                CHECK(!src.claim_stage(0,v,fill) && fill);
                CHECK(!src.fill_stage(v,0,0,src.stage_buf_[v].get()));
                CHECK(src.stage_of_.empty() && src.coherent());
                CHECK(src.stage_buf_.size()==std::min<size_t>(i+1,256));
            }
            CHECK(src.stage_buf_.size()==256);
            CHECK(arrays_live==(expect_pin?0:256) && pins_live==(expect_pin?256:0));
        }
        CHECK(arrays_live==0 && pins_live==0);
        {
            FileExpertSource src;
            for(int64_t key=0;key<600;++key) {
                size_t v=0;bool fill=false;
                CHECK(!src.claim_stage(key,v,fill) && fill);
                CHECK(src.stage_busy_[v] && src.coherent());
            }
            CHECK(src.stage_buf_.size()==600);
            for(size_t v=0;v<src.stage_buf_.size();++v)
                src.publish_stage(v,src.stage_key_[v]/src.n_expert_,false,0);
            CHECK(src.stage_of_.empty() && src.coherent());
            size_t v=0;bool fill=false;
            CHECK(!src.claim_stage(0,v,fill) && fill);
            CHECK(src.stage_buf_.size()==600 && src.coherent());
            CHECK(arrays_live==(expect_pin?0:600) && pins_live==(expect_pin?600:0));
        }
        CHECK(arrays_live==0 && pins_live==0);
        std::puts("PASS repeated_failed_fill_retained=256 busy_retained=600 reuse_retains_capacity=1 fixture_owner_releases_blobs=1");
    } else if(mode=="vram") {
        GrowCache cache;std::string error;
        cache.mapped=-1;
        CHECK(!grow_consumer(cache,error) && uploads==0 && error.find("cleanup pending")!=std::string::npos);
        CHECK(terminal_consumer(cache)==1);
        cache.mapped=128;error.clear();
        CHECK(grow_consumer(cache,error) && uploads==1 && terminal_consumer(cache)==0);
        cache.success=true;
        CHECK(grow_consumer(cache,error) && uploads==2);
        std::puts("PASS VRAM cleanup_pending stops refill/publication and command loop; ordinary capacity failure keeps prefix");
    } else if(mode=="todo") {
        (void) strata::kernels::cpu::expert_layout();
        for(int zero_copy=0;zero_copy<2;++zero_copy) for(int point=0;point<3;++point) {
            FileExpertSource src;src.direct_.clear();
            if(zero_copy)src.role_ptr_.push_back(reinterpret_cast<uint8_t*>(uintptr_t(1)));
            else src.io_pf_=true;
            std::vector<FileExpertSource::Fill> todo;int64_t experts[]={0,1};uint8_t pending[]={0,0};
            fail_at=point;allocations=0;
            src.claim_todo(0,experts,2,todo,pending,zero_copy!=0);
            fail_at=-1;
            CHECK(todo.empty() && src.stage_of_.empty() && src.coherent());
            CHECK(pending[0]==0 && pending[1]==0);
            src.claim_todo(0,experts,2,todo,pending,zero_copy!=0);
            CHECK(todo.size()==2 && src.coherent() && pending[0]==1 && pending[1]==1);
        }
        std::puts("PASS todo preclaim allocation_failures=6 demand_retry=1");
    } else if(mode=="workers") {
        for(int point=0;point<12;++point) {
            std::atomic<unsigned> next{0}, done{0}, active{0};
            auto work = [&] { ++active; for(unsigned i;(i=next.fetch_add(1))<100;) ++done; --active; };
            fail_at=point;allocations=0;
            run_fill_workers(4,work);
            fail_at=-1;
            CHECK(done==100 && active==0);
        }
        FileExpertSource src;size_t v=0;bool fill=false;
        CHECK(!src.claim_stage(0,v,fill) && fill);
        CHECK(!src.fill_stage(v,0,0,src.stage_buf_[v].get()));
        CHECK(!src.stage_busy_[v] && src.stage_of_.empty() && src.coherent());
        std::printf("PASS worker allocation_failures_joined=1 failed_fill_published=1\n");
    } else if(mode=="complement") {
        std::atomic<bool> failed{false};
        std::atomic<int> active{0}, completed{0};
        struct Active { std::atomic<int>& count; ~Active() { --count; } };
        auto failing = [&] {
            ++active;
            Active guard{active};
            throw std::bad_alloc();
        };
        run_complement_workers(6,failing,failed);
        CHECK(failed && active==0);
        failed=false;
        auto succeeding = [&] { ++active; Active guard{active}; ++completed; };
        run_complement_workers(6,succeeding,failed);
        CHECK(!failed && active==0 && completed==6);
        int released=0;
        std::string err, message="mapped read failed";
        fail_at=0; allocations=0;
        CHECK(!complement_copy_failed(err,message,[&] { ++released; }));
        fail_at=-1;
        CHECK(released==1 && err=="mapped read failed");
        err.clear(); message.clear();
        fail_at=0; allocations=0;
        CHECK(!complement_copy_failed(err,message,[&] { ++released; }));
        fail_at=-1;
        CHECK(released==2 && err.empty());
        std::puts("PASS complement_worker_exception=joined arena_release_before_error_format=1");
    } else if(mode=="direct") {
        FileExpertSource src;uint8_t a[16]{},b[16]{};
        FileExpertSource::Fill fills[]={{0,0,0,a},{0,1,0,b}};
        // Every process gets fresh thread-local scratch. Sweep all preparatory allocations.
        fail_at=argc>2?std::atoi(argv[2]):-1;allocations=0;reject_after_read=true;
        bool ok=src.read_direct(fills,2);fail_at=-1;reject_after_read=false;
        CHECK(issued==waited);
        CHECK(!ok || issued==2);
        CHECK(ok || issued==0);
        std::printf("PASS direct ok=%u issued=%u waited=%u events=%u\n",ok,issued,waited,events);
    } else return 2;
    return 0;
}
