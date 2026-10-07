#include "strata/core/readonly_cache_snapshot.hpp"
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#define CU(call) do { const cudaError_t e=(call); if(e!=cudaSuccess) { \
    std::fprintf(stderr,"line %d: %s\n",__LINE__,cudaGetErrorString(e)); std::exit(2); } } while(0)
#define REQUIRE(x) do { if(!(x)) { std::fprintf(stderr,"line %d: %s\n",__LINE__,#x);std::exit(3); } } while(0)

// The production boundary protocol: snapshot after a completed window, enqueue
// D2D reads on the refill stream, and fence the next cache mutation on compute.
// Host fallback fills follow the D2D work on the same ordered refill stream.
int main() {
    constexpr size_t bytes=5222400;
    cudaStream_t compute=nullptr,refill=nullptr;
    cudaEvent_t read_done=nullptr;
    CU(cudaStreamCreateWithFlags(&compute,cudaStreamNonBlocking));
    CU(cudaStreamCreateWithFlags(&refill,cudaStreamNonBlocking));
    CU(cudaEventCreateWithFlags(&read_done,cudaEventDisableTiming));
    uint8_t *cache=nullptr,*primary=nullptr,*host=nullptr;
    int32_t* tags=nullptr;
    CU(cudaMalloc(&cache,2*bytes));CU(cudaMalloc(&primary,2*bytes));CU(cudaMalloc(&tags,2*sizeof(int32_t)));
    CU(cudaHostAlloc(&host,bytes,cudaHostAllocDefault));
    strata::core::ReadonlyCacheSnapshot snapshot;
    CU(snapshot.open(1,8,2,bytes,tags,cache));
    std::vector<uint8_t> got(2*bytes);
    for(int round=0;round<8;++round) {
        const uint8_t first=(uint8_t)(31+round),second=(uint8_t)(81+round);
        const int32_t ids[2]={2,5};std::memset(host,second,bytes);
        CU(cudaMemsetAsync(cache,first,bytes,compute));
        CU(cudaMemcpyAsync(tags,ids,sizeof(ids),cudaMemcpyHostToDevice,compute));
        CU(snapshot.enqueue(compute));REQUIRE(!snapshot.find(0,2));
        REQUIRE(snapshot.complete_after_stream_sync(cudaStreamSynchronize(compute)));
        const uint8_t* source=snapshot.find(0,2);REQUIRE(source==cache);
        REQUIRE(!snapshot.find(0,1));REQUIRE(!snapshot.find(1,2));
        CU(cudaMemcpyAsync(primary,source,bytes,cudaMemcpyDeviceToDevice,refill));
        CU(cudaEventRecord(read_done,refill));
        CU(cudaStreamWaitEvent(compute,read_done,0));
        REQUIRE(snapshot.invalidate());REQUIRE(!snapshot.find(0,2));
        // A later window may overwrite the cache immediately after the fence.
        CU(cudaMemsetAsync(cache,0xee,bytes,compute));
        CU(cudaMemcpyAsync(primary+bytes,host,bytes,cudaMemcpyHostToDevice,refill));
        CU(cudaStreamSynchronize(refill));CU(cudaStreamSynchronize(compute));
        CU(cudaMemcpy(got.data(),primary,got.size(),cudaMemcpyDeviceToHost));
        for(size_t i=0;i<bytes;++i) REQUIRE(got[i]==first && got[bytes+i]==second);
    }
    // Pending metadata on shutdown is drained before pinned observation memory
    // is freed. Invalidated observations never authorize another refill.
    CU(snapshot.enqueue(compute));snapshot.close();REQUIRE(!snapshot.enabled());
    CU(cudaStreamSynchronize(refill));CU(cudaStreamSynchronize(compute));
    CU(cudaFreeHost(host));CU(cudaFree(tags));CU(cudaFree(primary));CU(cudaFree(cache));
    CU(cudaEventDestroy(read_done));CU(cudaStreamDestroy(refill));CU(cudaStreamDestroy(compute));
    std::puts("PASS secondary refill: completed snapshot, D2D/fallback, mutation fence, invalidation and pending close");
}
