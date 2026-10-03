#pragma once
// Opt-in CUDA diagnostics for the private Q8 placement experiment. No sampler,
// residency or synchronization changes when STRATA_FLEET_PROFILE is unset/0.
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <utility>
#if defined(STRATA_FLEET_CUDA_TRACE)
#include <cuda_profiler_api.h>
#include <cuda_runtime.h>
#include <nvtx3/nvToolsExt.h>
#endif

namespace strata::program {
class FleetDecodeCapture {
    bool enabled_ = false, active_ = false;
    int64_t skip_ = 32, count_ = 16;
public:
    FleetDecodeCapture() {
#if defined(STRATA_FLEET_CUDA_TRACE)
        const char* flag=std::getenv("STRATA_FLEET_PROFILE");
        enabled_=flag && flag[0]=='1';
        if (const char* p=std::getenv("STRATA_FLEET_PROFILE_SKIP")) skip_=std::max<int64_t>(0,std::atoll(p));
        if (const char* p=std::getenv("STRATA_FLEET_PROFILE_COUNT")) count_=std::clamp<int64_t>(std::atoll(p),1,1024);
#endif
    }
    ~FleetDecodeCapture() { stop(); }
    bool active() const { return active_; }
    void stop() {
#if defined(STRATA_FLEET_CUDA_TRACE)
        if (active_) { cudaDeviceSynchronize(); cudaProfilerStop(); active_=false; }
#endif
    }
    struct Range {
        bool on;
        Range(bool active,const char* name):on(active) {
#if defined(STRATA_FLEET_CUDA_TRACE)
            if(on) nvtxRangePushA(name);
#else
            (void)name;
#endif
        }
        ~Range() {
#if defined(STRATA_FLEET_CUDA_TRACE)
            if(on) nvtxRangePop();
#endif
        }
    };
    struct Window {
        FleetDecodeCapture& owner;
        int64_t index;
        const int64_t& produced;
        bool range_on=false;
        Window(FleetDecodeCapture& c,int64_t i,const int64_t& n):owner(c),index(i),produced(n) {
#if defined(STRATA_FLEET_CUDA_TRACE)
            if(owner.enabled_ && i==owner.skip_) {
                cudaDeviceSynchronize(); cudaProfilerStart(); owner.active_=true;
                std::fprintf(stderr,"fleet profile start window=%lld committed=%lld\n",(long long)i,(long long)n);
            }
            range_on=owner.active_;
            if(range_on) nvtxRangePushA("decode window");
#endif
        }
        ~Window() {
#if defined(STRATA_FLEET_CUDA_TRACE)
            if(range_on) nvtxRangePop();
            if(owner.active_ && index+1==owner.skip_+owner.count_) {
                std::fprintf(stderr,"fleet profile stop window=%lld committed=%lld\n",(long long)(index+1),(long long)produced);
                owner.stop();
            }
#endif
        }
    };
    template<class F> auto phase(const char* name,F&& fn)->decltype(fn()) {
        Range range(active_,name); return fn();
    }
};
}
