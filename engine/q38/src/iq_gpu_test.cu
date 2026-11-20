// Fixed before execution: exact scalar decode; matvec absolute tolerance 0.005,
// relative 0.0001 (FP32 warp-tree reduction versus FP64 serial oracle).
#define Q38_IQ_HOST_TABLE
#define main iq_cpu_reference_check
#include "iq_cpu_test.cpp"
#undef main
#include "ops.cuh"
#include "resources.hpp"
#include <vector>
#include <cmath>
#include <algorithm>
namespace Q=strata::qwen38;
int main(){try{
    if(iq_cpu_reference_check()!=0)throw std::runtime_error("CPU reference failed");
    Q::host_resources();size_t free_bytes,total;Q::ck(cudaMemGetInfo(&free_bytes,&total));
    if(free_bytes<Q::GPU_RESERVE+64*1024*1024)throw std::runtime_error("GPU reserve");
    constexpr int rows=9;
    for(int type:{21,22})for(int n:{5120,17408}){
        const int size=type==21?110:82,blocks=n/256;std::vector<uint8_t> w(size*blocks*rows);uint32_t state=0x9147635;
        for(auto& v:w){state=1664525u*state+1013904223u;v=uint8_t(state>>24);}
        for(int b=0;b<blocks*rows;++b){w[b*size]=0;w[b*size+1]=b%3==0?0:b%3==1?0x28:0xac;} // fp16 0, 1/32, -1/16
        std::vector<float> x(n),actual(n*rows),decoded(n*rows),y(rows);float tmp[256];
        for(int i=0;i<n;++i)x[i]=float(std::sin(i*.013)*.125);
        for(int b=0;b<blocks*rows;++b){float d=b%3==0?0.f:b%3==1?.03125f:-.0625f;reference(type,w.data()+b*size,d,tmp);std::copy(tmp,tmp+256,decoded.begin()+b*256);}
        Q::Buffer dw(w.size()),dx(x.size()*4),dy(rows*4),dd(decoded.size()*4);
        Q::ck(cudaMemcpy(dw.data,w.data(),w.size(),cudaMemcpyHostToDevice));Q::ck(cudaMemcpy(dx.data,x.data(),x.size()*4,cudaMemcpyHostToDevice));
        Q::decode(type,dw.data,dd.as(),n*rows,0);Q::matvec(type,dw.data,dx.as(),dy.as(),n,rows,0);Q::ck(cudaDeviceSynchronize());
        Q::ck(cudaMemcpy(actual.data(),dd.data,dd.bytes,cudaMemcpyDeviceToHost));Q::ck(cudaMemcpy(y.data(),dy.data,dy.bytes,cudaMemcpyDeviceToHost));
        for(size_t i=0;i<actual.size();++i)if(actual[i]!=decoded[i])throw std::runtime_error("decode mismatch");
        for(int r=0;r<rows;++r){double expected=0;for(int i=0;i<n;++i)expected+=double(decoded[r*n+i])*x[i];
            if(!std::isfinite(y[r])||std::abs(y[r]-expected)>.005+.0001*std::abs(expected))throw std::runtime_error("matvec mismatch");}
        std::printf("PASS IQ type=%d n=%d rows=%d\n",type,n,rows);
    }return 0;
}catch(const std::exception& e){std::fprintf(stderr,"FAIL %s\n",e.what());return 1;}}
