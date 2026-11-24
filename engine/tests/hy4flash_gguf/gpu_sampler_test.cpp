#include "strata/hy4flash_gguf/gpu_sampler.hpp"
#include "strata/hy4flash_gguf/indexer.hpp"
#include <cuda_runtime.h>
#include <cmath>
#include <cstdio>
#include <limits>
#include <stdexcept>
#include <vector>
#include <algorithm>
static void check(cudaError_t e){if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));}
int main() try {
    using namespace strata::hy4_flash;
    GpuSampler gpu;
    float* device=nullptr;check(cudaMalloc(&device,120832*4));
    uint32_t rng=12345;int checked_tokens=0;
    for(int n:{1,3,65,1024,120832}) {
        std::vector<float> values(n);
        for(auto& v:values){rng=rng*1664525+1013904223;v=(int(rng>>20)-2048)/256.f;}
        // Ties, signed zeros and a vocabulary wider than Qwen's shortlist.
        for(int i=0;i<n;i+=7)values[i]=i%2?0.f:-0.f;
        check(cudaMemcpy(device,values.data(),n*4,cudaMemcpyHostToDevice));
        for(uint64_t seed:{1ull,42ull,987654321ull})for(float temp:{0.f,.7f,1.f})for(int k:{0,1,50,1000000})for(float p:{.1f,.9f,1.f}) {
            SamplerConfig c;c.seed=seed;c.temperature=temp;c.top_k=k;c.top_p=p;c.repeat_penalty=1.2f;c.repeat_last=9;
            Sampler cpu(c);gpu.configure(c);std::vector<int32_t> history={0,0,n-1,3};
            for(int step=0;step<5;++step) {
                int64_t a=cpu.pick(values.data(),n,history),b=gpu.pick(device,n,history);
                if(a!=b){std::fprintf(stderr,"parity n=%d temp=%g k=%d p=%g step=%d cpu=%ld gpu=%ld\n",n,temp,k,p,step,(long)a,(long)b);return 1;}
                history.push_back(a);++checked_tokens;
            }
        }
    }
    for(float bad:{std::numeric_limits<float>::infinity(),-std::numeric_limits<float>::infinity(),std::numeric_limits<float>::quiet_NaN()}) {
        float x[]={0,bad,1};check(cudaMemcpy(device,x,sizeof x,cudaMemcpyHostToDevice));
        for(float t:{0.f,.7f}){SamplerConfig c;c.seed=42;c.temperature=t;gpu.configure(c);if(gpu.pick(device,3,{})!=-1)return 2;}
    }
    // Device acceptance must preserve sequential picks and the RNG state after
    // early rejection, EOS, truncation, penalties and an unused nonfinite row.
    int round_cases=0;
    for(float temp:{0.f,.7f})for(float penalty:{1.f,1.2f})for(int mismatch=0;mismatch<4;++mismatch)
    for(int limit=1;limit<=4;++limit)for(int stop_at:{-1,0,2}) {
        constexpr int n=65,T=4;std::vector<float> values(n*T);
        for(int i=0;i<n*T;++i)values[i]=std::sin(i*.137f)*4.f;
        SamplerConfig c;c.seed=42;c.temperature=temp;c.repeat_penalty=penalty;c.repeat_last=20;c.top_p=.9f;
        GpuSampler ref;ref.configure(c);gpu.configure(c);
        check(cudaMemcpy(device,values.data(),values.size()*4,cudaMemcpyHostToDevice));
        std::vector<int32_t> history={1,3,1},draft,picks,stop;
        auto hist=history;
        for(int row=0;row<T;++row) {
            int id=ref.pick(device+row*n,n,hist);picks.push_back(id);
            if(row<T-1){int d=row==mismatch?(id+1)%n:id;draft.push_back(d);hist.push_back(d);}
        }
        if(stop_at>=0)stop.push_back(picks[stop_at]);
        ref.configure(c);hist=history;
        RoundVerdict expected{};expected.next_position=7;
        for(int row=0;row<T && row<limit;++row) {
            int id=ref.pick(device+row*n,n,hist);expected.tokens[expected.count++]=id;
            expected.next_token=id;hist.push_back(id);
            const bool match=row<3 && id==draft[row];expected.accepted+=match;
            expected.stopped=std::find(stop.begin(),stop.end(),id)!=stop.end();
            if(expected.stopped||!match)break;
        }
        expected.next_position+=expected.count;
        gpu.begin_verify(device,n,draft,history,7,limit,stop);
        const auto got=gpu.finish_verify();
        if(got.count!=expected.count||got.accepted!=expected.accepted||got.next_position!=expected.next_position||
           got.stopped!=expected.stopped||got.nonfinite||got.next_token!=expected.next_token)return 5;
        for(int i=0;i<got.count;++i)if(got.tokens[i]!=expected.tokens[i])return 6;
        for(int i=0;i<3;++i)if(gpu.pick(device,n,hist)!=ref.pick(device,n,hist))return 7;
        ++round_cases;
    }
    // Cache commit uses the selected input prefix, not output-token identities.
    // Exercise every incoming carry and every keep length, across pool boundaries.
    IndexerCache indexer;std::string error;
    if(!indexer_cache_init(indexer,{1},128,4,128,error))throw std::runtime_error(error);
    indexer.verify_capacity=8;indexer.verify_vg.resize(1,nullptr);
    check(cudaMalloc(&indexer.verify_vg[0],2*128*12*4));
    std::vector<float> saved(2*128*12);for(size_t i=0;i<saved.size();++i)saved[i]=(float)i;
    check(cudaMemcpy(indexer.verify_vg[0],saved.data(),saved.size()*4,cudaMemcpyHostToDevice));
    for(int base=4;base<8;++base)for(int keep=1;keep<=4;++keep) {
        const int n=65;std::vector<float> values(4*n,0);for(int r=0;r<4;++r)values[r*n+1]=2;
        check(cudaMemcpy(device,values.data(),values.size()*4,cudaMemcpyHostToDevice));
        indexer.verify_base=base;indexer.verify_tokens=4;indexer.valid_pos=base+4;
        indexer.carry=(base+4)%4;indexer.n_pool=(base+4)/4;
        std::vector<float> vg(2*128*4,-7.f);
        check(cudaMemcpy(indexer.vg[0],vg.data(),vg.size()*4,cudaMemcpyHostToDevice));
        int cnt[2]={indexer.carry,indexer.n_pool};check(cudaMemcpy(indexer.counters,cnt,8,cudaMemcpyHostToDevice));
        SamplerConfig c;c.seed=42;gpu.configure(c);
        const auto* dv=gpu.begin_verify(device,n,{1,1,1},{},base,keep,{});
        indexer_commit_device(indexer,dv,gpu.stream());const auto verdict=gpu.finish_verify();
        check(cudaMemcpy(cnt,indexer.counters,8,cudaMemcpyDeviceToHost));
        check(cudaMemcpy(vg.data(),indexer.vg[0],vg.size()*4,cudaMemcpyDeviceToHost));
        const int pos=base+keep,carry=pos%4;
        if(verdict.count!=keep||cnt[0]!=carry||cnt[1]!=pos/4)return 8;
        for(int i=0;i<carry*2*128;++i) {
            const float expected=pos/4!=(base+4)/4?saved[(pos/4-base/4)*4*2*128+i]:-7.f;
            if(vg[i]!=expected)return 10;
        }
    }
    std::printf("PASS device prefix/RNG cases=%d; indexer commit cases=16\n",round_cases);
    for(int nan_row:{0,1,3}) {
        std::vector<float> values(4*65,0.f);for(int r=0;r<4;++r)values[r*65+1]=2;
        values[nan_row*65]=std::numeric_limits<float>::quiet_NaN();
        check(cudaMemcpy(device,values.data(),values.size()*4,cudaMemcpyHostToDevice));
        SamplerConfig c;c.seed=42;c.temperature=.7f;GpuSampler ref;ref.configure(c);gpu.configure(c);
        // Force rejection in row zero whenever it is finite, making later NaNs unreachable.
        int id=nan_row?ref.pick(device,65,{}):-1;
        gpu.begin_verify(device,65,{(id+1)%65,1,1},{},0,4,{});const auto got=gpu.finish_verify();
        if(nan_row==0 ? (!got.nonfinite||got.count) : (got.nonfinite||got.count!=1||got.tokens[0]!=id))return 11;
        std::vector<float> finite(65,1.f);check(cudaMemcpy(device,finite.data(),65*4,cudaMemcpyHostToDevice));
        for(int i=0;i<3;++i)if(ref.pick(device,65,{})!=gpu.pick(device,65,{}))return 12;
    }
    check(cudaFree(device));std::printf("PASS GPU/CPU picks=%d; nonfinite guard\n",checked_tokens);return 0;
}catch(const std::exception& e){std::fprintf(stderr,"%s\n",e.what());return 9;}
