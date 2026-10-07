// Reuse the independently checked synthetic Q8 tensors and original layout.
// Timings include activation quantization, repacking when enabled, and QKV.
#define main dense_q8_t8_parity_entry
#include "../tests/cuda/dense_q8_t8_test.cpp"
#undef main

int main() {
    int device=0;CU(cudaGetDevice(&device));cudaDeviceProp prop{};
    CU(cudaGetDeviceProperties(&prop,device));
    if(prop.major!=12 || prop.minor!=0) return 77;
    Case c(71);
    for(bool mmq : {false,true}) {
        cudaGraph_t graph=nullptr;cudaGraphExec_t exec=nullptr;
        CU(cudaStreamBeginCapture(c.stream,cudaStreamCaptureModeThreadLocal));
        strata::kernels::native_quantize_q8_1(c.x,c.q,K,T,c.stream);
        if(mmq) REQUIRE(c.pilot.run(8,c.w,c.q,c.candidate,K,N,T,c.stream));
        else strata::kernels::native_q8_0_mmvq(c.w,c.q,c.control,K,N,T,c.stream);
        CU(cudaStreamEndCapture(c.stream,&graph));CU(cudaGraphInstantiate(&exec,graph,0));
        for(int i=0;i<32;++i) CU(cudaGraphLaunch(exec,c.stream));
        CU(cudaStreamSynchronize(c.stream));
        cudaEvent_t begin=nullptr,end=nullptr;CU(cudaEventCreate(&begin));CU(cudaEventCreate(&end));
        CU(cudaEventRecord(begin,c.stream));
        constexpr int replays=256;
        for(int i=0;i<replays;++i) CU(cudaGraphLaunch(exec,c.stream));
        CU(cudaEventRecord(end,c.stream));CU(cudaEventSynchronize(end));
        float ms=0;CU(cudaEventElapsedTime(&ms,begin,end));
        std::printf("RESULT mmq=%d k=%d n=%d t=%d replays=%d stage_us=%.6f\n",int(mmq),K,N,T,replays,ms*1000/replays);
        CU(cudaEventDestroy(begin));CU(cudaEventDestroy(end));CU(cudaGraphExecDestroy(exec));CU(cudaGraphDestroy(graph));
    }
    c.check();
}
