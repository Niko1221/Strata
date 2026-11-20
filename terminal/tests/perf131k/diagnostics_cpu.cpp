// CPU fault injection only: functions under test are extracted verbatim from the consumers.
#include "strata/perf131k/blob_failure.hpp"
#include "strata/core/expert_cache_refuse.hpp"
#include "strata/core/residency_trim.hpp"
#include <algorithm>
#include <cassert>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>
#undef assert
#define assert(expr) do { if (!(expr)) { std::fprintf(stderr,"failed line %d: %s\n",__LINE__,#expr); return 1; } } while (0)
using CUdeviceptr = uintptr_t;
using CUdevice = int;
using CUresult = int;
using CUmemGenericAllocationHandle = unsigned long long;
struct CUmemAllocationProp {};
struct CUmemAccessDesc { struct { int type, id; } location; int flags; };
constexpr int CUDA_SUCCESS=0, cudaSuccess=0, CU_MEM_LOCATION_TYPE_DEVICE=1,
              CU_MEM_ACCESS_FLAGS_PROT_READWRITE=3,
              CU_DEVICE_ATTRIBUTE_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED=1,
              CU_MEM_ALLOC_GRANULARITY_RECOMMENDED=1;
int failure=0, cleanup_failure=0, reserves=0, unmaps=0, releases=0, address_frees=0;
unsigned long long next_handle=0;
namespace strata::core::ec_test {
bool (*refuse_segment_call)(const char*, int64_t) = nullptr;
}
namespace ec_test = strata::core::ec_test;
std::vector<std::string> driver_calls;
using cudaError_t=int;
int cudaDeviceSynchronize() { return cleanup_failure==4 ? 997 : 0; }
std::vector<std::string> calls;
int cudaGetDevice(int* p) { *p=0; return 0; }
int cudaGetLastError() { return 0; }
const char* cudaGetErrorString(int) { return "fake"; }
CUmemAllocationProp device_prop(int) { return {}; }
struct Vmm {
    bool ok=true;
    int create(CUmemGenericAllocationHandle* h, size_t, const CUmemAllocationProp*, int) const {
        calls.push_back("create"); *h=++next_handle; return failure==1 ? 701 : 0;
    }
    int map(CUdeviceptr,size_t,int,CUmemGenericAllocationHandle,int) const {
        calls.push_back("map"); return failure==2 ? 702 : 0;
    }
    int set_access(CUdeviceptr,size_t,const CUmemAccessDesc*,int) const {
        calls.push_back("access"); return failure==3 ? 703 : 0;
    }
    int unmap(CUdeviceptr p,size_t) const {
        ++unmaps; calls.push_back("unmap"); driver_calls.push_back("unmap:"+std::to_string((p-4096)/64));
        return cleanup_failure==1 ? 999 : 0;
    }
    int release(CUmemGenericAllocationHandle h) const {
        ++releases; calls.push_back("release"); driver_calls.push_back("release:"+std::to_string(h-1));
        return cleanup_failure==2 ? 998 : 0;
    }
    int address_free(CUdeviceptr,size_t) const { ++address_frees; return cleanup_failure==3 ? 996 : 0; }
    int device_get(CUdevice* d,int) const { *d=0; return 0; }
    int attribute(int* p,int,CUdevice) const { *p=1; return 0; }
    int granularity(size_t* p,const CUmemAllocationProp*,int) const { *p=64; return 0; }
    int reserve(CUdeviceptr* p,size_t,int,int,int) const { ++reserves; *p=4096; return 0; }
};
const Vmm& vmm() { static Vmm v; return v; }
struct ExpertCache {
    int64_t seg_=0,seg_req_=64,mapped_segs_=0,live_slots_=0;
    int64_t slots_=24,blob_=32;
    std::vector<uint64_t> off_;
    uint8_t* base_=nullptr;
    uint64_t reserved_=0;
    std::vector<unsigned long long> segs_;
    std::vector<int64_t> seg_size_;
    std::vector<char> seg_mapped_;
    bool cleanup_pending_=false;
    // @CACHE_VALID@
    int64_t full_bytes() const { return slots_*blob_; }
    int64_t slots() const { return live_slots_; }
    uint8_t* device_slot(int32_t);
    const uint8_t* device_slot(int32_t) const;
    bool open_segmented(uint64_t,std::string&);
    bool grow(int64_t,std::string&);
    int64_t mapped_bytes() const;
    bool shrink(int64_t,std::string&);
    int64_t slots_within(int64_t bytes) const { return bytes/32; }
    void release_segmented();
};
// @CACHE_FUNCTIONS@
// Execute the caller's shrink/error and next-request guard verbatim, but stub
// GPU residency upload/loan callbacks. This is not full generate-loop execution.
bool shrink_command(ExpertCache& xcache, std::vector<int32_t>& host_res,
                    std::vector<std::pair<int32_t,int32_t>>& vram_evicted,
                    std::vector<std::string>& events, std::string& e) {
    const int64_t keep=4*64;
    auto res_upload=[&]() { events.push_back("upload:"+std::to_string(xcache.slots())); };
    auto relend=[&]() { events.push_back("relend:"+std::to_string(xcache.slots())); };
    // @SHRINK_CALLER@
    return true;
}
int next_request_guard(const ExpertCache& xcache) {
    // @REQUEST_GUARD@
    return 0;
}
bool blob_present=true, transfer_ok=true;
int transfers=0;
struct Source { const uint8_t* blob_stable(int,int) { static uint8_t b; return blob_present ? &b : nullptr; } };
struct Cache {
    bool fill_slot_blocking(int,const uint8_t*,std::string& err,int64_t) {
        ++transfers; if (!transfer_ok) err="transfer failure 77"; return transfer_ok;
    }
};
// This fixture exercises the ordinary blob transfer path, not nibble packing.
namespace nib { struct Layer { bool on=false; }; Layer layer(int) { return {}; } }
struct NibbleCache {
    bool active() const { return false; }
    bool add(int,int,int,const uint8_t*,std::string&) { std::abort(); }
};
namespace strata { namespace kernels { namespace cpu {
struct Layout { int blob_bytes(int) const { return 128; } };
Layout expert_layout() { return {}; }
}}}
int fill_profile(std::string& err) {
    Source source; Source* srcp=&source; Cache xcache;
    NibbleCache nib_x;
    std::vector<std::pair<int,int>> profile{{2,3}};
    int i=0,slot=0;
    // @GENERATE_BLOCK@
    return 0;
}
int main(int argc, char**) {
    if (argc==1) {
    for (failure=1; failure<=3; ++failure) {
        calls.clear(); cleanup_failure=0;
        ExpertCache cache; std::string err="stale";
        assert(!cache.open_segmented(128,err));
        assert(err.find("cudaMalloc failed")!=std::string::npos);
        const char* api=failure==1 ? "cuMemCreate" : failure==2 ? "cuMemMap" : "cuMemSetAccess";
        assert(err.find(api)!=std::string::npos && err.find(std::to_string(700+failure))!=std::string::npos);
        assert(!cache.cleanup_pending_ && cache.base_==nullptr && cache.mapped_bytes()==0);
        assert(cache.segs_.empty() && cache.seg_mapped_.empty());
    }
    failure=0;
    for (int refusal=0;refusal<2;++refusal) {
        cleanup_failure=0;next_handle=0;driver_calls.clear();
        ExpertCache c;std::string err;
        assert(c.open_segmented(12*64,err));
        auto release9=+[](const char* call,int64_t seg) { return seg==9 && std::string(call)=="release"; };
        auto unmap6=+[](const char* call,int64_t seg) { return seg==6 && std::string(call)=="unmap"; };
        ec_test::refuse_segment_call=refusal==0 ? release9 : unmap6;
        std::vector<int32_t> res(24);
        for(int i=0;i<24;++i) res[i]=i;
        std::vector<std::pair<int32_t,int32_t>> ev;
        std::vector<std::string> events;
        assert(!shrink_command(c,res,ev,events,err));
        assert(err.find("cleanup pending")!=std::string::npos);
        assert(!c.valid() && c.live_slots_==0 && c.mapped_bytes()==-1);
        const auto& read_only=c;
        assert(c.device_slot(0)==nullptr && read_only.device_slot(0)==nullptr);
        assert(ev.size()==24 && std::all_of(res.begin(),res.end(),[](int x) { return x==strata::core::kNotResident; }));
        assert((events==std::vector<std::string>{"upload:0","relend:0"}));
        assert(next_request_guard(c)==1);
        const int stopped=refusal==0 ? 9 : 6;
        assert(c.mapped_segs_==(refusal==0 ? 9 : 7));
        assert(c.seg_mapped_[stopped]==(refusal==0 ? 0 : 1) && c.segs_[stopped]==(unsigned)stopped+1);
        for(int i=0;i<stopped;++i) assert(c.seg_mapped_[i] && c.segs_[i]==(unsigned)i+1);
        for(int i=stopped+1;i<12;++i) assert(!c.seg_mapped_[i] && c.segs_[i]==0);
        std::vector<std::string> expected;
        for(int i=11;i>stopped;--i) {
            expected.push_back("unmap:"+std::to_string(i));
            expected.push_back("release:"+std::to_string(i));
        }
        if(refusal==0) expected.push_back("unmap:9");
        assert(driver_calls==expected); // skipped unmap 6 must never release handle 6
        const auto before_calls=calls;
        assert(!c.grow(12*64,err) && !c.shrink(0,err) && calls==before_calls);
        ec_test::refuse_segment_call=nullptr;
        // A failed close retains handles, then retries only releases, never a
        // second unmap of segments already unmapped by shrink or the first close.
        cleanup_failure=2;driver_calls.clear();
        c.release_segmented();
        assert(c.cleanup_pending_ && !c.valid());
        const int unmapped=unmaps;
        for(const auto& call:driver_calls)
            if(call.rfind("unmap:",0)==0) assert(std::stoi(call.substr(6))<(refusal==0 ? 9 : 7));
        c.release_segmented();
        assert(unmaps==unmapped && c.cleanup_pending_);
        cleanup_failure=0;c.release_segmented();
        assert(unmaps==unmapped && !c.cleanup_pending_ && c.base_==nullptr);
        assert(c.open_segmented(12*64,err) && c.valid() && next_request_guard(c)==0);
        assert(c.shrink(4*64,err) && c.grow(12*64,err) && c.mapped_bytes()==12*64);
        c.release_segmented();
    }
    for (int kind=1;kind<=4;++kind) {
        cleanup_failure=0;
        ExpertCache cache; std::string err;
        assert(cache.open_segmented(128,err) && cache.mapped_bytes()==128 && cache.valid());
        const int before_reserves=reserves;
        cleanup_failure=kind;
        cache.release_segmented();
        assert(cache.cleanup_pending_ && cache.base_!=nullptr && cache.reserved_==128);
        assert(cache.mapped_bytes()==-1 && !cache.valid() && cache.live_slots_==0);
        assert(!cache.open_segmented(64,err) && reserves==before_reserves);
        assert(err.find("ownership")!=std::string::npos);
        assert(!cache.grow(128,err) && !cache.shrink(0,err));
        const int before_unmaps=unmaps;
        cache.release_segmented();
        if(kind==2 || kind==3) assert(unmaps==before_unmaps); // unmapped segments are not unmapped twice
        assert(cache.cleanup_pending_ && cache.mapped_bytes()==-1);
        cleanup_failure=0;
        cache.release_segmented();
        assert(!cache.cleanup_pending_ && cache.base_==nullptr && cache.reserved_==0 && cache.mapped_bytes()==0);
        assert(cache.open_segmented(64,err) && cache.valid());
        cache.release_segmented();
    }
    // Rollback of a segment whose access setup failed retains its handle/mapping.
    for(int kind=1;kind<=2;++kind) {
        failure=3;cleanup_failure=kind;
        ExpertCache cache;std::string err;
        assert(!cache.open_segmented(128,err));
        assert(cache.cleanup_pending_ && cache.segs_[0]!=0 && cache.mapped_bytes()==-1);
        assert(cache.seg_mapped_[0]==(kind==1));
        assert(err.find("CUresult=703")!=std::string::npos);
        assert(err.find("cudaMalloc failed")==std::string::npos); // unknown ownership cannot enter the smaller retry
        int before=reserves;
        assert(!cache.open_segmented(64,err) && reserves==before);
        cleanup_failure=0;cache.release_segmented();
        assert(!cache.cleanup_pending_ && cache.base_==nullptr);
    }
    failure=0;cleanup_failure=0;
    ExpertCache cache;std::string err;
    assert(cache.open_segmented(128,err));
    assert(cache.shrink(0,err) && cache.mapped_bytes()==0);
    failure=3;cleanup_failure=1;
    assert(!cache.grow(128,err) && cache.cleanup_pending_ && cache.segs_[0]!=0 && cache.mapped_bytes()==-1);
    assert(!cache.grow(128,err));
    failure=0;cleanup_failure=0;cache.release_segmented();
    assert(!cache.cleanup_pending_ && cache.mapped_bytes()==0);
    }
    std::string err;
    blob_present=false; err="stale cudaMalloc failure";
    assert(fill_profile(err)==1 && transfers==0);
    assert(err.find("blob acquisition failed")!=std::string::npos && err.find("unknown")!=std::string::npos);
    assert(err.find("stale")==std::string::npos);
    blob_present=true; transfer_ok=false; err="stale";
    assert(fill_profile(err)==1 && err=="transfer failure 77" && transfers==1);
    transfer_ok=true; assert(fill_profile(err)==0 && transfers==2);
    std::puts("CPU injected VMM/blob/transfer paths PASS; CUDA product compile/run NOT_RUN");
}
