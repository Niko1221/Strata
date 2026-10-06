// QFUSE must reproduce the separate quantizer, including its finite-half guard.
#include "strata/kernels/verify_kernels.hpp"
#include "strata/kernels/native_mmvq.hpp"
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

static void check(cudaError_t e) {
    if (e != cudaSuccess) { std::fprintf(stderr, "%s\n", cudaGetErrorString(e)); std::exit(2); }
}
struct Buffer {
    float* p = nullptr;
    size_t n;
    explicit Buffer(size_t count): n(count) { check(cudaMalloc((void**)&p, n * 4)); }
    ~Buffer() { cudaFree(p); }
    void random(std::mt19937& rng, float scale) {
        std::normal_distribution<float> d(0.f, scale);
        std::vector<float> h(n); for (auto& v:h) v=d(rng);
        check(cudaMemcpy(p,h.data(),n*4,cudaMemcpyHostToDevice));
    }
};
static int differences(const void* a, const void* b, size_t bytes) {
    std::vector<unsigned char> x(bytes), y(bytes);
    check(cudaMemcpy(x.data(),a,bytes,cudaMemcpyDeviceToHost));
    check(cudaMemcpy(y.data(),b,bytes,cudaMemcpyDeviceToHost));
    int n=0; for(size_t i=0;i<bytes;++i) n+=x[i]!=y[i]; return n;
}
int main() {
    using namespace strata::kernels;
    constexpr int HK=16, HV=48, Z=128*HV, C=(2*HK+HV)*128, ST=128*HV*128;
    Buffer state(ST), h(8*C), gate(8*HV), beta(8*HV), z(8*Z), gamma(128), y(8*Z), ref(8*Z);
    Buffer q(8*Z/32*9), rq(8*Z/32*9);
    cudaStream_t stream; check(cudaStreamCreate(&stream));
    int bad=0, cases=0;
    std::mt19937 rng(42);
    state.random(rng,.01f); h.random(rng,.03f); gate.random(rng,.02f); beta.random(rng,.02f); z.random(rng,1.f);
    for(float scale:{1.f,100000.f}) {
        gamma.random(rng,scale);
        for(int width=1;width<=8;++width) for(int begin:{0,width-1}) {
            for(int replay=0;replay<2;++replay) {
                check(cudaMemsetAsync(q.p,0x5a,q.n*4,stream));
                cudaGraph_t graph=nullptr; cudaGraphExec_t exec=nullptr;
                if(replay) check(cudaStreamBeginCapture(stream,cudaStreamCaptureModeThreadLocal));
                gdn_step_norm_multi(state.p,h.p,C,gate.p,beta.p,z.p,gamma.p,1e-6f,y.p,HK,HV,width,nullptr,stream,begin,q.p);
                if(replay) {
                    check(cudaStreamEndCapture(stream,&graph));
                    check(cudaGraphInstantiate(&exec,graph,nullptr,nullptr,0));
                    check(cudaGraphLaunch(exec,stream)); check(cudaGraphLaunch(exec,stream));
                }
                gdn_step_norm_multi(state.p,h.p,C,gate.p,beta.p,z.p,gamma.p,1e-6f,ref.p,HK,HV,width,nullptr,stream,begin);
                native_quantize_q8_1(y.p+begin*Z,rq.p,Z,width-begin,stream);
                check(cudaStreamSynchronize(stream));
                const int yd=differences(y.p+begin*Z,ref.p+begin*Z,(width-begin)*Z*4);
                const int qd=differences(q.p,rq.p,(width-begin)*Z/32*36);
                if(yd||qd) { ++bad; std::printf("FAIL scale=%g width=%d begin=%d graph=%d float_bytes=%d q8_bytes=%d\n",scale,width,begin,replay,yd,qd); }
                if(exec) { check(cudaGraphExecDestroy(exec)); check(cudaGraphDestroy(graph)); }
                ++cases;
            }
        }
    }
    check(cudaStreamDestroy(stream));
    std::printf("QFUSE GDN: %d cases, %d failures\n",cases,bad);
    return bad?1:0;
}
