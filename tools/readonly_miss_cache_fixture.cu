#include "strata/kernels/readonly_miss_cache.hpp"
#include "strata/kernels/captured_branch.hpp"
#include "strata/kernels/verify_kernels.hpp"
#include <cuda_runtime.h>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <random>
#include <vector>

using namespace strata::kernels;
#define CU(call) do { auto e = (call); if (e != cudaSuccess) { \
    std::fprintf(stderr, "%s:%d %s: %s\n", __FILE__, __LINE__, #call, cudaGetErrorString(e)); std::exit(2); } } while (0)
#define REQUIRE(x) do { if (!(x)) { std::fprintf(stderr, "require failed at %d: %s\n", __LINE__, #x); std::exit(3); } } while (0)

template<class T> struct Device {
    T* p = nullptr;
    explicit Device(size_t count) { CU(cudaMalloc((void**)&p, count * sizeof(T))); }
    ~Device() { if (p) cudaFree(p); }
};

__global__ void consume(const unsigned long long* ptrs, const int32_t* n,
                        uint4* output, int64_t per) {
    for (int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x; i < *n * per;
         i += (int64_t)gridDim.x * blockDim.x)
        output[i] = ((const uint4*)ptrs[i/per])[i%per];
}

__global__ void resident_work(const int32_t* count, uint32_t* out) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t v = (uint32_t)(*count * 103 + i);
    for (int j=0;j<512;++j) v = (v ^ (v >> 13)) * 1664525u + 1013904223u;
    out[i] = v;
}

void exercise(size_t bytes, int ways, int iterations, bool overlap=false, int cap=16) {
    constexpr int layers = 3, experts = 71;
    REQUIRE(cap >= 1 && cap <= kMissCacheMaxGroups);
    const size_t stride = bytes + 32;  // guard each cache slot, not only the allocation
    uint8_t *host = nullptr, *alias = nullptr;
    CU(cudaHostAlloc((void**)&host, layers * experts * bytes, cudaHostAllocMapped));
    CU(cudaHostGetDevicePointer((void**)&alias, host, 0));
    for (int l = 0; l < layers; ++l)
        for (int e = 0; e < experts; ++e)
            for (size_t i = 0; i < bytes; ++i)
                host[((size_t)l * experts + e) * bytes + i] = (uint8_t)(i*37 + (i>>8) + e*101 + l*19);
    const int storage_ways = std::max(ways,1);
    Device<uint8_t> cache(layers * storage_ways * stride), stage(cap * bytes + 64), output(cap * bytes);
    Device<int32_t> tags(layers * storage_ways), count(1), starts(cap+1), dst(cap*2), ids(64);
    Device<uint32_t> resident_out(256);
    Device<uint64_t> ages(layers*storage_ways), clocks(layers), counters(layers*4);
    Device<unsigned long long> ptrs(cap);
    Device<ReadonlyMissCachePlan> plan(1);
    CU(cudaMemset(cache.p, 0xcd, layers * storage_ways * stride));
    CU(cudaMemset(stage.p, 0xcd, cap * bytes + 64));
    CU(cudaMemset(tags.p, 0xff, layers * storage_ways * sizeof(int32_t)));
    CU(cudaMemset(ages.p, 0, layers * storage_ways * sizeof(uint64_t)));
    CU(cudaMemset(clocks.p, 0, layers * sizeof(uint64_t)));
    CU(cudaMemset(counters.p, 0, layers * 4 * sizeof(uint64_t)));
    CU(cudaDeviceSynchronize()); // the nonblocking streams do not inherit default-stream initialization
    cudaStream_t stream, branch;
    cudaEvent_t ready, done;
    CU(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    CU(cudaStreamCreateWithFlags(&branch, cudaStreamNonBlocking));
    CU(cudaEventCreateWithFlags(&ready, cudaEventDisableTiming));
    CU(cudaEventCreateWithFlags(&done, cudaEventDisableTiming));
    ReadonlyMissCacheBank banks[layers];
    cudaGraphExec_t graph[layers] = {};
    for (int l = 0; l < layers; ++l) {
        banks[l] = {cache.p + (size_t)l*ways*stride, tags.p+l*ways, ages.p+l*ways,
                    clocks.p+l, counters.p+l*4, ways, (int64_t)stride};
        CU(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
        if (overlap) CU(begin_captured_branch(stream, branch, ready));
        const cudaStream_t fetch = overlap ? branch : stream;
        if (ways) fetch_readonly_misses(ptrs.p, count.p, starts.p, dst.p, ids.p, stage.p,
                                       bytes, cap, banks[l], plan.p, fetch);
        else {
            fetch_blobs(ptrs.p, count.p, stage.p, bytes, cap, fetch);
            rebase_ptrs(ptrs.p, count.p, stage.p, bytes, fetch);
        }
        resident_work<<<1,256,0,stream>>>(count.p,resident_out.p);
        CU(cudaGetLastError());
        if (overlap) CU(join_captured_branch(stream, branch, done));
        consume<<<64,256,0,stream>>>(ptrs.p,count.p,(uint4*)output.p,bytes/16);
        CU(cudaGetLastError());
        cudaGraph_t g;
        CU(cudaStreamEndCapture(stream, &g));
        CU(cudaGraphInstantiate(&graph[l], g, nullptr, nullptr, 0));
        CU(cudaGraphDestroy(g));
    }
    std::mt19937 rng(35119);
    std::vector<uint8_t> got(cap*bytes), guard(layers*ways*stride), stage_guard(64);
    uint64_t requested = 0, abandoned = 0, null_hit_sources = 0;
    for (int it = 0; it < iterations; ++it) {
        const int layer = it < 8 ? 0 : (int)(rng()%layers);
        std::vector<int> chosen(experts);
        std::iota(chosen.begin(), chosen.end(), 0);
        std::shuffle(chosen.begin(), chosen.end(), rng);
        chosen.resize(rng()%(cap+1));
        if (it == 0) chosen = {};
        if (it == 1) chosen = {0,1,2,3};
        if (it == 2) chosen = {3,2,1,0};
        if (it == 3) chosen = {4,3,2,1};  // miss before hits: must reserve every hit
        if (it == 4) chosen = {0,4,3,2};
        if (it == 5) { chosen.resize(cap); std::iota(chosen.begin(),chosen.end(),0); }
        const int n = (int)chosen.size();
        std::vector<int32_t> htags(ways), hstarts(cap+1,0), hdst(cap*2,0), hids(64,-1);
        std::vector<unsigned long long> hptrs(cap,0);
        if (ways) CU(cudaMemcpy(htags.data(),banks[layer].tags,ways*sizeof(int32_t),cudaMemcpyDeviceToHost));
        for (int q = 0; q < n; ++q) {
            hstarts[q] = q*2; hdst[q*2] = (q*7)%64; hids[hdst[q*2]] = chosen[q];
            bool hit = std::find(htags.begin(),htags.end(),chosen[q]) != htags.end();
            // A hit must not even read its RAM source. A wrongly labelled hit
            // still fails the independent full-byte check of the consumer.
            hptrs[q] = hit ? 0 : (unsigned long long)(alias + ((size_t)layer*experts+chosen[q])*bytes);
            if (hit) ++null_hit_sources;
        }
        hstarts[n] = 2*n;
        CU(cudaMemcpyAsync(count.p,&n,sizeof(n),cudaMemcpyHostToDevice,stream));
        CU(cudaMemcpyAsync(starts.p,hstarts.data(),hstarts.size()*4,cudaMemcpyHostToDevice,stream));
        CU(cudaMemcpyAsync(dst.p,hdst.data(),hdst.size()*4,cudaMemcpyHostToDevice,stream));
        CU(cudaMemcpyAsync(ids.p,hids.data(),hids.size()*4,cudaMemcpyHostToDevice,stream));
        CU(cudaMemcpyAsync(ptrs.p,hptrs.data(),hptrs.size()*8,cudaMemcpyHostToDevice,stream));
        if (ways && it > 8 && it % 11 == 0) {
            plan_readonly_misses(count.p, starts.p, dst.p, ids.p, stage.p, bytes, cap, banks[layer],plan.p,stream);
            if (it % 22 == 0) fill_readonly_misses(ptrs.p,bytes,plan.p,stream);
            CU(cudaStreamSynchronize(stream));
            std::vector<int32_t> after(ways);
            CU(cudaMemcpy(after.data(),banks[layer].tags,ways*4,cudaMemcpyDeviceToHost));
            for (int e : chosen)
                if (std::find(htags.begin(),htags.end(),e) == htags.end())
                    REQUIRE(std::find(after.begin(),after.end(),e) == after.end());
            ++abandoned; // before or after fill, without publication: next graph must recover
            continue;
        }
        CU(cudaGraphLaunch(graph[layer],stream));
        CU(cudaMemcpyAsync(got.data(),output.p,n*bytes,cudaMemcpyDeviceToHost,stream));
        CU(cudaStreamSynchronize(stream));
        if (ways) {
            const char* compact = std::getenv("STRATA_Q8_COMPACT_MISS_FILL");
            if (compact && std::strcmp(compact, "1") == 0) {
                ReadonlyMissCachePlan copied;
                CU(cudaMemcpy(&copied, plan.p, sizeof(copied), cudaMemcpyDeviceToHost));
                std::vector<int> expected;
                for (int q = 0; q < n; ++q)
                    if (std::find(htags.begin(), htags.end(), chosen[q]) == htags.end()) expected.push_back(q);
                REQUIRE(copied.count == n && copied.fill_count == (int)expected.size());
                for (int q = 0; q < copied.fill_count; ++q) REQUIRE(copied.fill_group[q] == expected[q]);
            }
        }
        uint32_t resident[256];
        CU(cudaMemcpy(resident,resident_out.p,sizeof(resident),cudaMemcpyDeviceToHost));
        for (int i=0;i<256;++i) {
            uint32_t v=(uint32_t)(n*103+i);
            for (int j=0;j<512;++j) v=(v^(v>>13))*1664525u+1013904223u;
            REQUIRE(resident[i]==v);
        }
        for (int q = 0; q < n; ++q)
            REQUIRE(std::memcmp(got.data()+q*bytes,host+((size_t)layer*experts+chosen[q])*bytes,bytes)==0);
        requested += n;
    }
    uint64_t stats[layers*4];
    CU(cudaMemcpy(stats,counters.p,sizeof(stats),cudaMemcpyDeviceToHost));
    uint64_t total=0,hits=0,fills=0,bypasses=0;
    for(int l=0;l<layers;++l) {total+=stats[l*4];hits+=stats[l*4+1];fills+=stats[l*4+2];bypasses+=stats[l*4+3];}
    if (ways) {
        REQUIRE(total==requested && hits+fills==total && bypasses<=fills && hits>0 && null_hit_sources>0);
        if (ways < cap) REQUIRE(bypasses>0);
        REQUIRE(abandoned>0);
    } else {
        REQUIRE(total==0 && hits==0 && fills==0 && bypasses==0);
        total=requested;fills=requested;
    }
    if (ways) CU(cudaMemcpy(guard.data(),cache.p,guard.size(),cudaMemcpyDeviceToHost));
    for(int s=0;s<layers*ways;++s)
        for(size_t i=bytes;i<stride;++i) REQUIRE(guard[(size_t)s*stride+i]==0xcd);
    CU(cudaMemcpy(stage_guard.data(),stage.p+cap*bytes,64,cudaMemcpyDeviceToHost));
    for(auto v:stage_guard) REQUIRE(v==0xcd);
    // Cache use never mutates authoritative host weights.
    for(int l=0;l<layers;++l) for(int e=0;e<experts;++e) for(size_t i=0;i<bytes;++i)
        REQUIRE(host[((size_t)l*experts+e)*bytes+i]==(uint8_t)(i*37+(i>>8)+e*101+l*19));
    std::printf("PASS bytes=%zu ways=%d overlap=%d requests=%llu hits=%llu fills=%llu bypasses=%llu abandoned=%llu\n",
        bytes,ways,(int)overlap,(unsigned long long)total,(unsigned long long)hits,(unsigned long long)fills,
        (unsigned long long)bypasses,(unsigned long long)abandoned);
    for(auto g:graph) CU(cudaGraphExecDestroy(g));
    CU(cudaStreamDestroy(stream));
    CU(cudaStreamSynchronize(branch));
    CU(cudaStreamDestroy(branch));
    CU(cudaEventDestroy(ready));CU(cudaEventDestroy(done));
    CU(cudaFreeHost(host));
}

int main(int argc,char** argv) {
    bool quick = argc>1 && std::strcmp(argv[1],"--quick")==0;
    for(bool overlap : {false,true}) {
        for(int ways : {0,1,4,16}) {
            exercise(16,ways,quick?32:160,overlap);
            exercise(144,ways,quick?32:160,overlap);
        }
        for(int ways : {0,4}) exercise(5222400,ways,quick?24:80,overlap);
        for(int ways : {4,16}) exercise(144,ways,quick?32:160,overlap,64);
    }
    std::puts("PASS staged/cache miss fetch: fork/join, complete bytes, independent resident outputs, source immutability, graph replay, guards, abandoned fills");
}
