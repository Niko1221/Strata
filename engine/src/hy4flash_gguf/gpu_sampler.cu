#include "strata/hy4flash_gguf/gpu_sampler.hpp"
#include <cuda_runtime.h>
#include <math_constants.h>
#include <cub/cub.cuh>
#include <algorithm>
#include <chrono>
#include <stdexcept>
#include <climits>

namespace strata::hy4_flash {
namespace {
#include "gpu_pick.cuh"
void checked(cudaError_t e) {if(e!=cudaSuccess)throw std::runtime_error(std::string("hy4-flash GPU sampler: ")+cudaGetErrorString(e));}
struct Buf {
    void* p=nullptr;size_t bytes=0;
    ~Buf(){if(p)cudaFree(p);}
    void reserve(size_t n){if(n<=bytes)return;if(p)checked(cudaFree(p));p=nullptr;bytes=0;checked(cudaMalloc(&p,n));bytes=n;}
    template<class T>T* as(){return static_cast<T*>(p);}
};
__global__ void prepare_k(const float* logits,float* keys,int* ids,int n,const int* history,int hn,SamplerConfig c,int* bad) {
    for(int i=blockIdx.x*blockDim.x+threadIdx.x;i<n;i+=blockDim.x*gridDim.x) {
        float x=logits[i];if(!isfinite(x))atomicExch(bad,1);
        if(c.repeat_penalty!=1.f)for(int j=0;j<hn;++j)if(history[j]==i) {
            x=x>0.f?__fdiv_rn(x,c.repeat_penalty):__fmul_rn(x,c.repeat_penalty);break;
        }
        if(c.temperature>0.f && c.temperature!=1.f)x=__fdiv_rn(x,c.temperature);
        keys[i]=(i==c.excluded_token)?-CUDART_INF_F:x;ids[i]=i;
    }
}
__global__ void greedy_k(const float* logits,int n,const int* history,int hn,SamplerConfig c,int* out) {
    __shared__ float score[256];__shared__ int ids[256],invalid[256];
    float best=-CUDART_INF_F;int id=n,bad=0;
    for(int i=threadIdx.x;i<n;i+=blockDim.x) {
        float v=logits[i];if(!isfinite(v))bad=1;
        if(i==c.excluded_token)continue;
        if(c.repeat_penalty!=1.f)for(int j=0;j<hn;++j)if(history[j]==i) {
            v=v>0.f?__fdiv_rn(v,c.repeat_penalty):__fmul_rn(v,c.repeat_penalty);break;
        }
        if(v>best || (v==best && i<id)){best=v;id=i;}
    }
    score[threadIdx.x]=best;ids[threadIdx.x]=id;invalid[threadIdx.x]=bad;__syncthreads();
    for(int d=128;d;d/=2) {
        if(threadIdx.x<d) {
            int j=threadIdx.x+d;invalid[threadIdx.x]|=invalid[j];
            if(score[j]>score[threadIdx.x] || (score[j]==score[threadIdx.x]&&ids[j]<ids[threadIdx.x])) {
                score[threadIdx.x]=score[j];ids[threadIdx.x]=ids[j];
            }
        }
        __syncthreads();
    }
    if(!threadIdx.x)*out=invalid[0]?-1:ids[0];
}
__global__ void exp_k(const float* sorted,float* probs,double* scan_in,int k) {
    for(int i=blockIdx.x*blockDim.x+threadIdx.x;i<k;i+=blockDim.x*gridDim.x)
        scan_in[i]=(double)(probs[i]=expf(__fsub_rn(sorted[i],sorted[0])));
}
__global__ void normalize_k(float* probs,double* input,const double* prefix,int k,const int* keep) {
    const int n=keep?*keep:k;const double sum=prefix[n-1];
    for(int i=blockIdx.x*blockDim.x+threadIdx.x;i<k;i+=blockDim.x*gridDim.x)
        input[i]=i<n?(double)(probs[i]=(float)((double)probs[i]/sum)):0.;
}
__global__ void cutoff_k(const double* prefix,int k,float top_p,int* keep) {
    if(threadIdx.x||blockIdx.x)return;
    int lo=0,hi=k;while(lo<hi){int mid=(lo+hi)/2;if(prefix[mid]<(double)top_p)lo=mid+1;else hi=mid;}
    *keep=min(k,lo+1);
}
// Parallel double scans avoid a single GPU thread walking the full vocabulary.
// Reduction grouping can change last bits vs the CPU; this path is opt-in.
__global__ void draw_k(const double* prefix,const int* ids,int k,const int* keep,uint64_t* rng,const int* bad,int* out) {
    if(threadIdx.x||blockIdx.x)return;
    if(*bad){*out=-1;return;}
    if(keep)k=*keep;
    *rng+=0x9e3779b97f4a7c15ull;uint64_t z=*rng;
    z=(z^(z>>30))*0xbf58476d1ce4e5b9ull;z=(z^(z>>27))*0x94d049bb133111ebull;z^=z>>31;
    const float u=(float)((double)(z>>11)*(1.0/9007199254740992.0));
    int lo=0,hi=k;while(lo<hi){int mid=(lo+hi)/2;if(prefix[mid]<=(double)u)lo=mid+1;else hi=mid;}
    *out=ids[min(lo,k-1)];
}
__global__ void accept_prefix_k(const int* picks,const int* drafts,int rows,int limit,int base,
    const int* stop,int nstop,const uint64_t* snapshots,uint64_t* rng,RoundVerdict* result,const DeviceLoopState* loop=nullptr) {
    if(threadIdx.x||blockIdx.x)return;
    if(loop){base=loop->base;limit=loop->remaining;}
    RoundVerdict v{};v.next_position=base;
    for(int i=0;i<rows && i<limit;++i) {
        const int token=picks[i];
        if(token<0){v.nonfinite=1;break;}
        v.tokens[v.count++]=token;v.next_token=token;
        const bool match=i<rows-1 && token==drafts[i];
        if(match)++v.accepted;
        for(int j=0;j<nstop;++j)if(token==stop[j])v.stopped=1;
        if(v.stopped||!match)break;
    }
    // Discard all speculative random draws beyond the output prefix, including
    // later rows already sampled for an ultimately rejected branch.
    *rng=snapshots[v.count];v.next_position=base+v.count;*result=v;
}
}
struct GpuSampler::Impl {
    cudaStream_t stream=nullptr;SamplerConfig config;
    Buf keys[2],ids[2],probs,sort_tmp,scan_tmp,scan_in,prefix,keep,history,rng,result,bad,pick_parts;
    Buf row_picks,drafts,stops,snapshots,verdict;
    std::vector<int32_t> staged_history;
    bool pending=false;
    int loop_stops=0;
    Impl(){checked(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));rng.reserve(8);result.reserve(4);bad.reserve(4);}
    ~Impl(){cudaStreamSynchronize(stream);cudaStreamDestroy(stream);}
    void reserve(int n,int history_count) {
        for(int i=0;i<2;++i){keys[i].reserve(n*4);ids[i].reserve(n*4);}
        probs.reserve(n*4);scan_in.reserve(n*8);prefix.reserve(n*8);keep.reserve(4);
        pick_parts.reserve(64*sizeof(PickPart));history.reserve(std::max(1,history_count)*4);
        size_t bytes=0;
        checked(cub::DeviceRadixSort::SortPairsDescending(nullptr,bytes,keys[0].as<float>(),keys[1].as<float>(),ids[0].as<int>(),ids[1].as<int>(),n,0,32,stream));
        sort_tmp.reserve(bytes);
        checked(cub::DeviceScan::InclusiveSum(nullptr,bytes,scan_in.as<double>(),prefix.as<double>(),n,stream));
        scan_tmp.reserve(bytes);
    }
    void enqueue(const float* logits,int n,const int32_t* hist,int hn);
};
GpuSampler::GpuSampler():impl_(new Impl){}
GpuSampler::~GpuSampler()=default;
void GpuSampler::configure(const SamplerConfig& c) {
    if(impl_->pending)throw std::runtime_error("GPU sampler: finish pending verification before reconfiguration");
    impl_->config=c;uint64_t seed=c.seed;
    if(!seed)seed=(uint64_t)std::chrono::steady_clock::now().time_since_epoch().count()*0x2545f491ull;
    checked(cudaMemcpyAsync(impl_->rng.p,&seed,8,cudaMemcpyHostToDevice,impl_->stream));
    checked(cudaStreamSynchronize(impl_->stream));
}
int32_t GpuSampler::pick(const float* logits,int n,const std::vector<int32_t>& h) {
    auto& s=*impl_;if(!logits||n<=0)return -1;
    if(s.pending)throw std::runtime_error("GPU sampler: finish pending verification first");
    const int hn=s.config.repeat_penalty!=1.f ? std::min<int>(std::max(0,s.config.repeat_last),h.size()):0;
    if(hn){s.history.reserve(hn*4);checked(cudaMemcpyAsync(s.history.p,h.data()+h.size()-hn,hn*4,cudaMemcpyHostToDevice,s.stream));}
    s.enqueue(logits,n,s.history.as<int>(),hn);
    checked(cudaGetLastError());int32_t out=-1;
    checked(cudaMemcpyAsync(&out,s.result.p,4,cudaMemcpyDeviceToHost,s.stream));checked(cudaStreamSynchronize(s.stream));return out;
}
void GpuSampler::Impl::enqueue(const float* logits,int n,const int32_t* hist,int hn) {
    auto& s=*this;
    if(s.config.temperature<=0.f && s.config.repeat_penalty==1.f) {
        s.pick_parts.reserve(64*sizeof(PickPart));device_pick(logits,n,s.pick_parts.as<PickPart>(),nullptr,s.result.as<int>(),s.stream,s.config.excluded_token);
    } else if(s.config.temperature<=0.f)greedy_k<<<1,256,0,s.stream>>>(logits,n,hist,hn,s.config,s.result.as<int>());
    else {
        for(int i=0;i<2;++i){s.keys[i].reserve(n*4);s.ids[i].reserve(n*4);}s.probs.reserve(n*4);
        checked(cudaMemsetAsync(s.bad.p,0,4,s.stream));
        prepare_k<<<(n+255)/256,256,0,s.stream>>>(logits,s.keys[0].as<float>(),s.ids[0].as<int>(),n,hist,hn,s.config,s.bad.as<int>());
        size_t bytes=0;
        checked(cub::DeviceRadixSort::SortPairsDescending(nullptr,bytes,s.keys[0].as<float>(),s.keys[1].as<float>(),s.ids[0].as<int>(),s.ids[1].as<int>(),n,0,32,s.stream));
        s.sort_tmp.reserve(bytes);
        // CUB stable sort preserves ascending token ID among equal finite keys.
        checked(cub::DeviceRadixSort::SortPairsDescending(s.sort_tmp.p,bytes,s.keys[0].as<float>(),s.keys[1].as<float>(),s.ids[0].as<int>(),s.ids[1].as<int>(),n,0,32,s.stream));
        const int k=s.config.top_k>0?std::min(n,s.config.top_k):n;
        s.scan_in.reserve(k*8);s.prefix.reserve(k*8);s.keep.reserve(4);
        size_t scan_bytes=0;
        checked(cub::DeviceScan::InclusiveSum(nullptr,scan_bytes,s.scan_in.as<double>(),s.prefix.as<double>(),k,s.stream));
        s.scan_tmp.reserve(scan_bytes);
        auto scan=[&]{checked(cub::DeviceScan::InclusiveSum(s.scan_tmp.p,scan_bytes,s.scan_in.as<double>(),s.prefix.as<double>(),k,s.stream));};
        exp_k<<<(k+255)/256,256,0,s.stream>>>(s.keys[1].as<float>(),s.probs.as<float>(),s.scan_in.as<double>(),k);scan();
        normalize_k<<<(k+255)/256,256,0,s.stream>>>(s.probs.as<float>(),s.scan_in.as<double>(),s.prefix.as<double>(),k,nullptr);scan();
        if(s.config.top_p<1.f) {
            cutoff_k<<<1,1,0,s.stream>>>(s.prefix.as<double>(),k,s.config.top_p,s.keep.as<int>());
            normalize_k<<<(k+255)/256,256,0,s.stream>>>(s.probs.as<float>(),s.scan_in.as<double>(),s.prefix.as<double>(),k,s.keep.as<int>());scan();
        }
        draw_k<<<1,1,0,s.stream>>>(s.prefix.as<double>(),s.ids[1].as<int>(),k,s.config.top_p<1.f?s.keep.as<int>():nullptr,s.rng.as<uint64_t>(),s.bad.as<int>(),s.result.as<int>());
    }
    checked(cudaGetLastError());
}
const RoundVerdict* GpuSampler::begin_verify(const float* logits,int n,const std::vector<int32_t>& draft,
    const std::vector<int32_t>& h,int base,int remaining,const std::vector<int32_t>& stop) {
    auto& s=*impl_;const int rows=(int)draft.size()+1;
    if(s.pending||!logits||n<=0||rows>8||remaining<1||base<0)throw std::runtime_error("GPU sampler: invalid verify input/state");
    const int stride=std::max(1,s.config.repeat_penalty!=1.f?std::max(0,s.config.repeat_last):0);
    s.reserve(n,rows*stride);s.row_picks.reserve(8*4);s.drafts.reserve(7*4);
    s.stops.reserve(std::max<size_t>(1,stop.size())*4);s.snapshots.reserve(9*8);s.verdict.reserve(sizeof(RoundVerdict));
    s.staged_history.assign(rows*stride,-1);
    for(int r=0;r<rows;++r)for(int j=0;j<stride;++j) {
        const long at=(long)h.size()+r-stride+j;
        if(at>=0)s.staged_history[r*stride+j]=at<(long)h.size()?h[at]:draft[at-h.size()];
    }
    checked(cudaMemcpyAsync(s.history.p,s.staged_history.data(),s.staged_history.size()*4,cudaMemcpyHostToDevice,s.stream));
    if(!draft.empty())checked(cudaMemcpyAsync(s.drafts.p,draft.data(),draft.size()*4,cudaMemcpyHostToDevice,s.stream));
    if(!stop.empty())checked(cudaMemcpyAsync(s.stops.p,stop.data(),stop.size()*4,cudaMemcpyHostToDevice,s.stream));
    checked(cudaMemcpyAsync(s.snapshots.p,s.rng.p,8,cudaMemcpyDeviceToDevice,s.stream));
    for(int r=0;r<rows;++r) {
        const int hn=s.config.repeat_penalty!=1.f?std::min<int>(stride,h.size()+r):0;
        s.enqueue(logits+(size_t)r*n,n,s.history.as<int>()+(r+1)*stride-hn,hn);
        checked(cudaMemcpyAsync(s.row_picks.as<int>()+r,s.result.p,4,cudaMemcpyDeviceToDevice,s.stream));
        checked(cudaMemcpyAsync(s.snapshots.as<uint64_t>()+r+1,s.rng.p,8,cudaMemcpyDeviceToDevice,s.stream));
    }
    accept_prefix_k<<<1,1,0,s.stream>>>(s.row_picks.as<int>(),s.drafts.as<int>(),rows,remaining,base,s.stops.as<int>(),stop.size(),
        s.snapshots.as<uint64_t>(),s.rng.as<uint64_t>(),s.verdict.as<RoundVerdict>());
    checked(cudaGetLastError());s.pending=true;return s.verdict.as<RoundVerdict>();
}
void* GpuSampler::stream()const{return impl_->stream;}
RoundVerdict GpuSampler::finish_verify() {
    auto& s=*impl_;if(!s.pending)throw std::runtime_error("GPU sampler: no pending verification");
    RoundVerdict out;checked(cudaMemcpyAsync(&out,s.verdict.p,sizeof out,cudaMemcpyDeviceToHost,s.stream));
    checked(cudaStreamSynchronize(s.stream));s.pending=false;return out;
}
void GpuSampler::prepare_loop(int n,int rows,const std::vector<int32_t>& stop) {
    auto& s=*impl_;
    if(s.pending||s.config.repeat_penalty!=1.f)throw std::runtime_error("GPU loop sampler currently requires neutral repetition penalty");
    s.reserve(n,1);s.row_picks.reserve(8*4);s.snapshots.reserve(9*8);s.verdict.reserve(sizeof(RoundVerdict));
    s.stops.reserve(std::max<size_t>(1,stop.size())*4);s.loop_stops=stop.size();
    if(!stop.empty())checked(cudaMemcpyAsync(s.stops.p,stop.data(),stop.size()*4,cudaMemcpyHostToDevice,s.stream));
    checked(cudaStreamSynchronize(s.stream));
}
const RoundVerdict* GpuSampler::record_loop(const float* logits,int n,int rows,const int32_t* window,
    const int32_t*,DeviceLoopState* state,void* stream) {
    auto& s=*impl_;const auto saved=s.stream;s.stream=static_cast<cudaStream_t>(stream);
    try {
        checked(cudaMemcpyAsync(s.snapshots.p,s.rng.p,8,cudaMemcpyDeviceToDevice,s.stream));
        for(int r=0;r<rows;++r){s.enqueue(logits+(size_t)r*n,n,nullptr,0);
            checked(cudaMemcpyAsync(s.row_picks.as<int>()+r,s.result.p,4,cudaMemcpyDeviceToDevice,s.stream));
            checked(cudaMemcpyAsync(s.snapshots.as<uint64_t>()+r+1,s.rng.p,8,cudaMemcpyDeviceToDevice,s.stream));}
        accept_prefix_k<<<1,1,0,s.stream>>>(s.row_picks.as<int>(),window+1,rows,rows,0,s.stops.as<int>(),s.loop_stops,
            s.snapshots.as<uint64_t>(),s.rng.as<uint64_t>(),s.verdict.as<RoundVerdict>(),state);
        checked(cudaGetLastError());
    }catch(...){s.stream=saved;throw;}
    s.stream=saved;return s.verdict.as<RoundVerdict>();
}
}
