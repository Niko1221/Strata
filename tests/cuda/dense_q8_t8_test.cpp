#include "strata/prefill/dense_q8_t8.hpp"
#include "strata/kernels/native_mmvq.hpp"
#include <cuda_runtime.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <vector>

#define CU(call) do {auto e=(call);if(e!=cudaSuccess){std::fprintf(stderr,"%d: %s\n",__LINE__,cudaGetErrorString(e));std::exit(2);}}while(0)
#define REQUIRE(x) do {if(!(x)){std::fprintf(stderr,"%d: %s\n",__LINE__,#x);std::exit(3);}}while(0)
constexpr int K=2560,N=10240,T=8;
struct Case {
    cudaStream_t stream=nullptr;
    void *w=nullptr,*q=nullptr;
    float *x=nullptr,*control=nullptr,*candidate=nullptr;
    strata::prefill::DenseQ8T8 pilot;
    cudaGraph_t graph=nullptr;cudaGraphExec_t exec=nullptr;
    std::vector<unsigned char> weights;
    std::vector<float> input,got,ref;
    explicit Case(int seed):weights(N*(K/32)*34+4096,0),input(T*K),got(T*N),ref(T*N) {
        for(int row=0;row<N;++row)for(int b=0;b<K/32;++b){
            auto* block=weights.data()+(row*(K/32)+b)*34;
            const uint16_t scale=0x2400;std::memcpy(block,&scale,2); // exact1/64
            for(int j=0;j<32;++j)block[2+j]=(unsigned char)(int8_t)(((row*7+b*13+j*31+seed)%255)-127);
        }
        for(int i=0;i<T*K;++i)input[i]=float((i*17+seed)%251-125)*.001f;
        for(int i=(T-1)*K;i<T*K;++i)input[i]=0; // exactzero row
        CU(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));
        CU(cudaMalloc(&w,weights.size()));CU(cudaMalloc(&q,T*(K/32)*36));
        CU(cudaMalloc(&x,input.size()*sizeof(float)));
        CU(cudaMalloc(&control,ref.size()*sizeof(float)));CU(cudaMalloc(&candidate,got.size()*sizeof(float)));
        CU(cudaMemcpyAsync(w,weights.data(),weights.size(),cudaMemcpyHostToDevice,stream));
        CU(cudaMemcpyAsync(x,input.data(),input.size()*sizeof(float),cudaMemcpyHostToDevice,stream));
        std::string error;REQUIRE(pilot.open(stream,error));
        REQUIRE(!pilot.open(stream,error));
        REQUIRE(!pilot.run(8,w,q,candidate,K,N,4,stream));
        REQUIRE(!pilot.run(8,w,q,candidate,K,N,T,nullptr));
        REQUIRE(!pilot.run(12,w,q,candidate,K,N,T,stream));
        launch();CU(cudaStreamSynchronize(stream));check();
        CU(cudaStreamBeginCapture(stream,cudaStreamCaptureModeThreadLocal));
        launch();CU(cudaStreamEndCapture(stream,&graph));
        CU(cudaGraphInstantiate(&exec,graph,0));
    }
    void launch(){
        strata::kernels::native_quantize_q8_1(x,q,K,T,stream);
        strata::kernels::native_q8_0_mmvq(w,q,control,K,N,T,stream);
        REQUIRE(pilot.run(8,w,q,candidate,K,N,T,stream));
    }
    void mutate(){
        for(int i=0;i<(T-1)*K;++i)input[i]=input[i]*.8f+.015625f;
        CU(cudaMemcpyAsync(x,input.data(),input.size()*sizeof(float),cudaMemcpyHostToDevice,stream));
        CU(cudaGraphLaunch(exec,stream));
    }
    void check(){
        CU(cudaMemcpy(got.data(),candidate,got.size()*sizeof(float),cudaMemcpyDeviceToHost));
        CU(cudaMemcpy(ref.data(),control,ref.size()*sizeof(float),cudaMemcpyDeviceToHost));
        double squared=0,mass=0,max_error=0;
        for(size_t i=0;i<got.size();++i){
            REQUIRE(std::isfinite(got[i])&&std::isfinite(ref[i]));
            double diff=double(got[i])-ref[i];squared+=diff*diff;mass+=double(ref[i])*ref[i];
            max_error=std::fmax(max_error,std::fabs(diff));
        }
        double rms=std::sqrt(squared/std::fmax(mass,1e-30));REQUIRE(rms<1e-5);
        for(int row=0;row<N;++row)REQUIRE(got[(T-1)*N+row]==0&&ref[(T-1)*N+row]==0);
        // Independent scalar interpretation of original activation and weight
        // blocks for64output rows per nonzero token. No floating quantizer oracle.
        std::vector<unsigned char> activation(T*(K/32)*36);
        CU(cudaMemcpy(activation.data(),q,activation.size(),cudaMemcpyDeviceToHost));
        for(int token=0;token<T-1;++token)for(int pick=0;pick<64;++pick){
            int row=(pick*149)%N;double sum=0,sum_abs=0;
            for(int b=0;b<K/32;++b){
                const auto* wb=weights.data()+(row*(K/32)+b)*34;
                const auto* ab=activation.data()+(token*(K/32)+b)*36;
                uint16_t h=0;std::memcpy(&h,ab,2);
                int exponent=(h>>10)&31,frac=h&1023;
                double scale=exponent?std::ldexp(1.0+frac/1024.0,exponent-15):std::ldexp(double(frac),-24);
                if(h&0x8000)scale=-scale;
                int dot=0;for(int j=0;j<32;++j)dot+=int((int8_t)wb[2+j])*int((int8_t)ab[4+j]);
                double part=double(dot)*scale/64;sum+=part;sum_abs+=std::fabs(part);
            }
            const double epsilon=std::ldexp(1.0,-24),ops=2*(K/32)+16;
            const double bound=(ops*epsilon/(1-ops*epsilon))*sum_abs+1e-7;
            REQUIRE(std::fabs(double(got[token*N+row])-sum)<=bound);
            REQUIRE(std::fabs(double(ref[token*N+row])-sum)<=bound);
        }
        std::printf("PASS dense T8: relative RMS %.9g maxabs %.9g; scalarbounds,zero row\n",rms,max_error);
    }
    ~Case(){
        CU(cudaStreamSynchronize(stream));
        if(exec)CU(cudaGraphExecDestroy(exec));if(graph)CU(cudaGraphDestroy(graph));
        pilot.close();
        CU(cudaFree(candidate));CU(cudaFree(control));CU(cudaFree(x));CU(cudaFree(q));CU(cudaFree(w));
        CU(cudaStreamDestroy(stream));
    }
};
int main(){
    int device=0;CU(cudaGetDevice(&device));cudaDeviceProp prop{};CU(cudaGetDeviceProperties(&prop,device));
    if(prop.major!=12||prop.minor!=0)return 77;
    Case a(71),b(131);
    for(int i=0;i<4;++i){a.mutate();b.mutate();CU(cudaStreamSynchronize(a.stream));CU(cudaStreamSynchronize(b.stream));a.check();b.check();}
    // Early close after a captured launch must complete its scratch users.
    a.mutate();a.pilot.close();a.check();
    std::puts("PASS dense T8 projection: capture, mutation, independent owners/streams, scalar format, unsupported shapes and early close");
}
