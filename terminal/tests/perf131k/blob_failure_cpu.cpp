// CPU fake APIs. Consumer functions are inserted verbatim; this is not a product build.
#include "strata/perf131k/blob_failure.hpp"
#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <cstring>
#include <memory>
#include <mutex>
#include <string>
#include <vector>
#include <thread>
#include <new>
namespace bf = strata::perf131k::blob_failure;
using DWORD=unsigned long; using HANDLE=void*;
struct OVERLAPPED { DWORD Offset=0,OffsetHigh=0; HANDLE hEvent=nullptr; };
constexpr int TRUE=1,FALSE=0,ERROR_IO_PENDING=997,MEM_RELEASE=1,MEM_COMMIT=2,MEM_RESERVE=4,PAGE_READWRITE=8;
std::string mode;
std::vector<std::string> order;
unsigned error_value=0,error_reads=0,issued=0,waited=0,freed=0;
DWORD GetLastError() { ++error_reads; return error_value; }
void* VirtualAlloc(void*,size_t bytes,int,int) {
    order.push_back("alloc"); if(mode=="alloc") { error_value=1455; return nullptr; }
    return std::calloc(1,bytes);
}
int VirtualFree(void* p,int,int) { ++freed; std::free(p); return 1; }
HANDLE CreateEventW(void*,int,int,void*) {
    order.push_back("event"); if(mode=="event") { error_value=8; return nullptr; }
    return reinterpret_cast<HANDLE>(uintptr_t(3));
}
int CloseHandle(HANDLE) { return 1; }
int ReadFile(HANDLE,void*,DWORD,void*,OVERLAPPED*) {
    order.push_back("read");
    if(mode=="issue" && issued==1) { error_value=5; return 0; }
    ++issued; error_value=ERROR_IO_PENDING; return 0;
}
int GetOverlappedResult(HANDLE,OVERLAPPED*,DWORD* got,int) {
    order.push_back("wait"); ++waited;
    *got=mode=="short" ? 0 : 4096;
    error_value=mode=="wait" ? 23 : 999; // A later API overwrites the earlier issue error.
    return mode=="wait" ? 0 : 1;
}
namespace strata::perf131k {
enum class Stage { expert_direct_read_host };
struct Scope { explicit Scope(Stage) {} };
}
struct ExpertSource { virtual const uint8_t* blob_stable(int,int)=0; };
struct FileExpertSource: ExpertSource {
    struct Fill { size_t v; int64_t layer,e; uint8_t* dst; };
    struct Map { uint8_t* base=nullptr; };
    int64_t n_layers_=2,n_expert_=512,stage_blob_=16;
    std::vector<uint8_t*> role_ptr_;
    std::vector<uint64_t> layer_offsets_{0,8ull<<20},layer_blob_bytes_{16,16},role_bytes_;
    std::vector<int> role_file_;
    std::vector<Map> maps_;
    std::vector<void*> direct_{reinterpret_cast<void*>(uintptr_t(1))};
    std::mutex stage_mu_;
    std::condition_variable stage_cv_;
    bool io_pf_=false;
    std::atomic<uint64_t> io_crit_us_{0},io_crit_n_{0};
    std::vector<std::unique_ptr<uint8_t[]>> stage_buf_;
    std::vector<int64_t> stage_key_;
    std::vector<int> stage_busy_;
    FileExpertSource() { stage_buf_.emplace_back(new uint8_t[16]); stage_key_.push_back(0); stage_busy_.push_back(0); }
    bool read_direct(const Fill*,size_t) const;
    const uint8_t* staged_blob(int64_t,int64_t);
    bool claim_stage(int64_t key,size_t& v,bool& fill) {
        v=0;fill=mode!="claim" && mode!="key";
        stage_key_[0]=mode=="key" ? -1 : key;
        return mode=="key";
    }
    bool fill_stage(size_t,int64_t,int64_t,uint8_t*) { return mode!="fill"; }
    const uint8_t* blob_stable(int l,int e) override { return staged_blob(l,e); }
};
struct ArenaExpertSource: ExpertSource { const uint8_t* blob_stable(int,int) override { return nullptr; } };
// @DIRECT_FUNCTION@
// @STAGED_FUNCTION@
namespace strata::kernels::cpu { struct Layout { int blob_bytes(int)const{return 16;} }; Layout expert_layout(){return{};} }
unsigned transfers=0;
struct Cache { bool fill_slot_blocking(int,const uint8_t*,std::string&,int64_t){++transfers;return true;} };
int profile() {
    FileExpertSource src; ArenaExpertSource arena_src; ExpertSource* srcp=&src; Cache xcache;
    std::vector<std::pair<int,int>> profile{{0,0}};size_t i=0;int slot=0;std::string err;
    // @PROFILE_BLOCK@
    return 0;
}
bool heap_fail=false;
void* operator new[](size_t n,const std::nothrow_t&) noexcept { return heap_fail ? nullptr : std::malloc(n); }
void operator delete[](void* p) noexcept { std::free(p); }
using cudaError_t=int;constexpr int cudaSuccess=0,cudaHostAllocDefault=0;
int cudaHostAlloc(void** p,size_t n,int) { if(mode=="pin")return 101;*p=std::malloc(n);return 0; }
int cudaGetLastError(){error_value=999;return 0;}
bool stage_alloc(bool want_pin) {
    int64_t stage_blob_=16;
    // @ALLOC_BLOCK@
    std::free(raw);return true;
}
int main(int argc,char**argv) {
    if(argc!=2)return 2;mode=argv[1];
    bool ok=false;
    if(mode=="claim" || mode=="key" || mode=="fill") {
        ok=profile()==0;
    } else if(mode=="pin" || mode=="heap") {
        bf::Scope scope(bf::Source::file,45,168);
        heap_fail=mode=="heap";ok=stage_alloc(mode=="pin");heap_fail=false;
        if(!ok)bf::flush_terminal(16);
    } else if(mode=="flood") {
        bf::Scope scope(bf::Source::file,45,168);
        for(unsigned n=0;n<100;++n)bf::record(bf::Step::stage_fill,16);
        bf::flush_terminal(16);
    } else if(mode=="threads") {
        std::vector<std::thread> threads;
        for(unsigned i=0;i<4;++i) threads.emplace_back([]{for(unsigned j=0;j<20;++j)bf::record(bf::Step::stage_fill,16);});
        for(auto& t:threads)t.join();
        bf::Scope scope(bf::Source::file,45,168);bf::flush_terminal(16);
    } else {
        FileExpertSource src;uint8_t a[16]{},b[16]{};
        FileExpertSource::Fill fills[]={{0,0,0,a},{0,1,0,b}};
        bf::Scope scope(bf::Source::file,45,168);
        ok=src.read_direct(fills,2);
        if(!ok)bf::flush_terminal(16);
    }
    std::printf("STATE ok=%u issued=%u waited=%u error_reads=%u transfers=%u order=",ok,issued,waited,error_reads,transfers);
    for(const auto&s:order)std::printf("%s,",s.c_str());std::puts("");
    return 0;
}
