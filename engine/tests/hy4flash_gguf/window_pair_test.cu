#include "strata/kernels/native_mmvq.hpp"
#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <utility>
#include <vector>

static void check(cudaError_t e){if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));}
struct Buffer {void* p=nullptr;explicit Buffer(size_t n){check(cudaMalloc(&p,n));}~Buffer(){cudaFree(p);}};
int main() try {
    using namespace strata::kernels;
    cudaStream_t s;check(cudaStreamCreateWithFlags(&s,cudaStreamNonBlocking));
    native_mmvq_set_rows(2);
    const int ni=1536,no=1536,slots=32;
    const size_t stride=(size_t)ni*no/256*144;
    std::vector<uint8_t> weights(2*slots*stride);
    uint32_t rng=12345;for(auto& b:weights){rng=1664525*rng+1013904223;b=rng>>24;}
    for(size_t i=0;i<weights.size();i+=144){uint16_t d=0x1800,m=0x1000;std::memcpy(weights.data()+i,&d,2);std::memcpy(weights.data()+i+2,&m,2);}
    Buffer w(weights.size());check(cudaMemcpyAsync(w.p,weights.data(),weights.size(),cudaMemcpyHostToDevice,s));
    const auto xb=native_q8_1_bytes(ni,1);
    for(int t:{1,2,3,4,7,8})for(int mode=0;mode<3;++mode) {
        const int n=t*16;std::vector<int32_t> ids(n);std::vector<float> x((size_t)n*ni);
        for(int i=0;i<n;++i)ids[i]=mode==0?-1:(mode==1?i%16:((i%5==0)?-1:((i%16)+(i/16)%2*16)));
        for(auto& v:x){rng=1664525*rng+1013904223;v=(int(rng>>16)-32768)/32768.f;}
        Buffer dx(x.size()*4),q(n*xb),di(n*4),a(n*no*4),b(n*no*4),c(n*no*4),d(n*no*4);
        check(cudaMemcpyAsync(dx.p,x.data(),x.size()*4,cudaMemcpyHostToDevice,s));check(cudaMemcpyAsync(di.p,ids.data(),n*4,cudaMemcpyHostToDevice,s));
        for(void* p:{a.p,b.p,c.p,d.p})check(cudaMemsetAsync(p,0x42,n*no*4,s));
        native_quantize_q8_1_wide((float*)dx.p,q.p,ni,n,s);
        if(!native_mmvq_grouped_pair(12,w.p,stride,(char*)w.p+slots*stride,stride,q.p,xb,(float*)a.p,(float*)b.p,ni,no,n,s,(int32_t*)di.p,1))return 2;
        if(!native_mmvq_window_pair(12,w.p,stride,(char*)w.p+slots*stride,stride,q.p,xb,(float*)c.p,(float*)d.p,ni,no,n,(int32_t*)di.p,t,s))return 2;
        check(cudaStreamSynchronize(s));std::vector<float> ref(n*no),got(n*no);
        for(auto pair:{std::make_pair(a.p,c.p),std::make_pair(b.p,d.p)}) {
            check(cudaMemcpy(ref.data(),pair.first,ref.size()*4,cudaMemcpyDeviceToHost));
            check(cudaMemcpy(got.data(),pair.second,got.size()*4,cudaMemcpyDeviceToHost));
            if(std::memcmp(ref.data(),got.data(),ref.size()*4)){std::fprintf(stderr,"mismatch T=%d mode=%d\n",t,mode);return 1;}
        }
        for(int stripes:{1,2,4,8,16,32}) {
            check(cudaMemsetAsync(c.p,0x42,n*no*4,s));check(cudaMemsetAsync(d.p,0x42,n*no*4,s));
            if(!native_mmvq_window_striped_pair(12,w.p,stride,(char*)w.p+slots*stride,stride,
                q.p,xb,(float*)c.p,(float*)d.p,ni,no,n,s,(int32_t*)di.p,stripes))return 2;
            check(cudaStreamSynchronize(s));
            for(auto pair:{std::make_pair(a.p,c.p),std::make_pair(b.p,d.p)}) {
                check(cudaMemcpy(ref.data(),pair.first,ref.size()*4,cudaMemcpyDeviceToHost));
                check(cudaMemcpy(got.data(),pair.second,got.size()*4,cudaMemcpyDeviceToHost));
                if(std::memcmp(ref.data(),got.data(),ref.size()*4)){std::fprintf(stderr,"stripe GU mismatch T=%d mode=%d stripes=%d\n",t,mode,stripes);return 1;}
            }
        }
        // Both down types, including masked entries, against the original grouped kernel.
        for(int type:{12,14}) {
            const size_t ds=(size_t)ni*no/256*(type==12?144:210);
            std::vector<uint8_t> dw(slots*ds);
            for(auto& v:dw){rng=1664525*rng+1013904223;v=rng>>24;}
            for(size_t i=0;i<dw.size();i+=(type==12?144:210)) {
                uint16_t scale=0x1800;std::memcpy(dw.data()+i+(type==12?0:208),&scale,2);
                if(type==12){uint16_t m=0x1000;std::memcpy(dw.data()+i+2,&m,2);}
            }
            Buffer wd(dw.size());check(cudaMemcpyAsync(wd.p,dw.data(),dw.size(),cudaMemcpyHostToDevice,s));
            native_mmvq_set_q6_c50(true);check(cudaMemsetAsync(a.p,0x42,n*no*4,s));
            native_mmvq_grouped_context(type,wd.p,ds,q.p,xb,(float*)a.p,ni,no,n,s,(int32_t*)di.p,1);
            check(cudaStreamSynchronize(s));check(cudaMemcpy(ref.data(),a.p,ref.size()*4,cudaMemcpyDeviceToHost));
            for(int stripes:{1,2,4,8,16,32}) {
                check(cudaMemsetAsync(c.p,0x42,n*no*4,s));
                if(!native_mmvq_window_striped(type,wd.p,ds,q.p,xb,(float*)c.p,ni,no,n,s,(int32_t*)di.p,stripes))return 2;
                check(cudaStreamSynchronize(s));check(cudaMemcpy(got.data(),c.p,got.size()*4,cudaMemcpyDeviceToHost));
                if(std::memcmp(ref.data(),got.data(),ref.size()*4)){std::fprintf(stderr,"stripe down mismatch type=%d T=%d mode=%d stripes=%d\n",type,t,mode,stripes);return 1;}
            }
        }
        std::printf("T=%d mode=%d exact\n",t,mode);
    }
    check(cudaStreamDestroy(s));return 0;
}catch(const std::exception& e){std::fprintf(stderr,"%s\n",e.what());return 1;}
