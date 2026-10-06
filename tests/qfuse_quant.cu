// Quantization-boundary regression: ordinary random data rarely lands near an
// integer half-step, where precise and native fast division choose other codes.
#include "strata/kernels/native_mmvq.hpp"
#include "strata/kernels/q8_1_finite.hpp"
#include <cuda_runtime.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>
using namespace strata::kernels;
struct Block { half2 ds; int8_t qs[32]; };
__global__ void quant(const float* x, Block* out, int count, bool legacy) {
    const int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=count) return;
    const float xi=x[i]; float maximum=fabsf(xi), sum=xi;
    for(int o=16;o;o>>=1) maximum=fmaxf(maximum,__shfl_xor_sync(0xffffffffu,maximum,o));
    for(int o=16;o;o>>=1) sum+=__shfl_xor_sync(0xffffffffu,sum,o);
    const float d=legacy?q8_1_finite(maximum/127.f):q8_1_native_scale(maximum);
    out[i/32].qs[i%32]=legacy?q8_1_quant(xi,d,maximum):q8_1_native_quant(xi,d,maximum);
    if(i%32==0) out[i/32].ds=q8_1_ds(d,sum);
}
static void check(cudaError_t e) {
    if(e!=cudaSuccess) { std::fprintf(stderr,"%s\n",cudaGetErrorString(e)); std::exit(2); }
}
int main() {
    constexpr int blocks=8192, n=blocks*32;
    std::mt19937 rng(1987); std::uniform_real_distribution<float> scale(.1f,10.f);
    std::vector<float> input(n);
    for(int b=0;b<blocks;++b) {
        const float maximum=scale(rng)*std::pow(10.f,(float)(b%12-5)), d=maximum/127.f;
        for(int j=0;j<31;++j) {
            const int q=(int)(rng()%253)-126;
            input[b*32+j]=d*(q+.5f);
        }
        input[b*32+31]=maximum;
    }
    for(int i=0;i<32;++i) input[i]=0;
    float* x; Block *a,*b,*c;
    check(cudaMalloc((void**)&x,n*4));
    check(cudaMalloc((void**)&a,blocks*sizeof(Block)));
    check(cudaMalloc((void**)&b,blocks*sizeof(Block)));
    check(cudaMalloc((void**)&c,blocks*sizeof(Block)));
    check(cudaMemcpy(x,input.data(),n*4,cudaMemcpyHostToDevice));
    cudaStream_t stream; check(cudaStreamCreate(&stream));
    quant<<<n/256,256,0,stream>>>(x,a,n,false);
    quant<<<n/256,256,0,stream>>>(x,b,n,true);
    native_quantize_q8_1(x,c,n,1,stream);
    check(cudaDeviceSynchronize());
    std::vector<unsigned char> ha(blocks*sizeof(Block)),hb(ha.size()),hc(ha.size());
    check(cudaMemcpy(ha.data(),a,ha.size(),cudaMemcpyDeviceToHost));
    check(cudaMemcpy(hb.data(),b,hb.size(),cudaMemcpyDeviceToHost));
    check(cudaMemcpy(hc.data(),c,hc.size(),cudaMemcpyDeviceToHost));
    int bad=0,old=0;
    for(size_t i=0;i<ha.size();++i) { bad+=ha[i]!=hc[i]; old+=hb[i]!=hc[i]; }
    std::printf("QFUSE boundary: %d blocks, new differing bytes %d, legacy differing bytes %d\n",blocks,bad,old);
    cudaFree(x);cudaFree(a);cudaFree(b);cudaFree(c);cudaStreamDestroy(stream);
    return bad || old==0;
}
