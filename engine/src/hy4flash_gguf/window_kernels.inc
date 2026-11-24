// Included in attention.cu's anonymous namespace. Masked phases preserve the
// original token/rank output layout and the exact per-expert reduction kernels.
__global__ void window_publish_k(WindowMailbox* m,const int32_t* ids,int n) {
    const int i=threadIdx.x;if(i<n)m->ids[i]=ids[i];
    if(i==0){m->stall[0]=0;m->stall[1]=0;}
    __syncthreads();__threadfence_system();__syncthreads();
    if(i==0){const uint32_t e=m->request+1;if(!e)m->error=1;else m->request=e;}
}
__global__ void window_plan_k(WindowMailbox* m,int32_t* indices,int n) {
    if(threadIdx.x==0)while(m->plan_ready<m->request&&!m->error)__nanosleep(128);
    __syncthreads();__threadfence_system();
    for(int i=threadIdx.x;i<n;i+=blockDim.x)indices[i]=m->error?-1:((volatile int32_t*)m->index)[i];
}
__global__ void window_wait_k(WindowMailbox* m,int ph) {
    if(threadIdx.x==0) {
        const long long t0=clock64();
        while(m->ready[ph]<m->request&&!m->error)__nanosleep(128);
        m->stall[0]+=(unsigned long long)(clock64()-t0);
    }
    __threadfence_system();
}
__global__ void window_consumed_k(WindowMailbox* m,int ph) {
    __threadfence_system();if(threadIdx.x==0)m->consumed[ph]=m->request;
}
__global__ void window_done_k(WindowMailbox* m) {
    __threadfence_system();if(threadIdx.x==0)m->done=m->request;
}
// These nodes are captured only in the CPU-enabled graph. Layer completion protects
// both mapped buffers from reuse until this scatter and the normal weighted reduce finish.
__global__ void window_cpu_input_k(WindowMailbox* m,const float* input,float* host,int size) {
    for(int i=threadIdx.x;i<size;i+=blockDim.x)host[i]=input[i];
    __threadfence_system();__syncthreads();
    if(threadIdx.x==0)m->cpu_input_ready=m->request;
}
__global__ void window_cpu_parts_k(WindowMailbox* m,const float* host,float* parts,int dim) {
    const int entry=blockIdx.x;
    const bool mine=(m->cpu_mask[entry/kWindowTopK] & (1u<<(entry%kWindowTopK)))!=0;
    // Entry 0 stays behind to time the wait even when it owns no row - but ONLY when some row is the
    // CPU's. A layer the CPU sat out never publishes `cpu_ready` for this epoch, and the per-block early
    // return is what used to keep this kernel from waiting on a flag that will never arrive.
    bool any=mine;
    if(!entry)for(int t=0,nt=gridDim.x/kWindowTopK;t<nt;++t)any|=m->cpu_mask[t]!=0;
    if(!any)return;
    if(threadIdx.x==0) {
        const long long t0=clock64();
        while(m->cpu_ready<m->request&&!m->error)__nanosleep(128);
        if(!entry)m->stall[1]+=(unsigned long long)(clock64()-t0);
    }
    __syncthreads();__threadfence_system();
    if(m->error||!mine)return;
    for(int i=threadIdx.x;i<dim;i+=blockDim.x)
        parts[entry*dim+i]=host[entry*dim+i];
}
__global__ void window_repeat_k(const float* x,float* y,int dim,int n) {
    for(int i=blockIdx.x*blockDim.x+threadIdx.x;i<n*dim;i+=gridDim.x*blockDim.x)
        y[i]=x[(i/dim/kWindowTopK)*dim+i%dim];
}
// The draft block's whole top-k goes to the CPU: publish the routed ids and the latent input, then
// spin for the answer. One block each - the payloads are 16 ids, 1536 floats in and 16x1536 out.
__global__ void draft_cpu_publish_k(DraftMailbox* m,const int32_t* ids,const float* input,
                                    float* host,int dim,int k) {
    for(int i=threadIdx.x;i<k;i+=blockDim.x)m->ids[i]=ids[i];
    for(int i=threadIdx.x;i<dim;i+=blockDim.x)host[i]=input[i];
    __syncthreads();__threadfence_system();__syncthreads();
    if(!threadIdx.x){const uint32_t e=m->request+1;if(!e)m->error=1;else m->request=e;}
}
__global__ void draft_cpu_parts_k(DraftMailbox* m,const float* host,float* parts,int n) {
    if(!threadIdx.x)while(m->done<m->request&&!m->error)__nanosleep(128);
    __syncthreads();__threadfence_system();
    if(m->error)return;
    for(int i=threadIdx.x;i<n;i+=blockDim.x)parts[i]=host[i];
}
#if defined(CUDART_VERSION) && CUDART_VERSION >= 13000 && !defined(STRATA_USE_HIP)
__global__ void window_condition_k(const int* index,int n,cudaGraphConditionalHandle handle) {
    int any=0;for(int i=threadIdx.x;i<n;i+=blockDim.x)any|=index[i]>=0;
    any=__any_sync(0xffffffffu,any);if(!threadIdx.x)cudaGraphSetConditional(handle,any);
}
struct WindowConditional {
    cudaStream_t main=nullptr,body=nullptr;cudaGraphNode_t node=nullptr;
    cudaStream_t begin(cudaStream_t stream,const int* index,int n) {
        static const bool on=[] {const char* e=std::getenv("STRATA_HY4_FLASH_WINDOW_SKIP_EMPTY");const char* l=std::getenv("STRATA_HY4_FLASH_DEVICE_LOOP");return e&&std::atoi(e)!=0&&!(l&&std::atoi(l)!=0);}();
        if(!on)return stream;
        auto checked=[](cudaError_t e){if(e!=cudaSuccess)throw std::runtime_error(std::string("window conditional: ")+cudaGetErrorString(e));};
        // Create the side stream during eager warmup, never lazily inside capture.
        static thread_local cudaStream_t side=nullptr;
        if(!side)checked(cudaStreamCreateWithFlags(&side,cudaStreamNonBlocking));
        cudaStreamCaptureStatus status;cudaGraph_t graph=nullptr;
        checked(cudaStreamGetCaptureInfo(stream,&status,nullptr,&graph));
        if(status!=cudaStreamCaptureStatusActive)return stream;
        cudaGraphConditionalHandle handle;
        checked(cudaGraphConditionalHandleCreate(&handle,graph,0,0));
        window_condition_k<<<1,32,0,stream>>>(index,n,handle);
        const cudaGraphNode_t* deps=nullptr;const cudaGraphEdgeData* edges=nullptr;size_t count=0;
        checked(cudaStreamGetCaptureInfo(stream,&status,nullptr,&graph,&deps,&edges,&count));
        cudaGraphNodeParams params{};params.type=cudaGraphNodeTypeConditional;
        params.conditional.handle=handle;params.conditional.type=cudaGraphCondTypeIf;params.conditional.size=1;
        checked(cudaGraphAddNode(&node,graph,deps,edges,count,&params));
        checked(cudaStreamUpdateCaptureDependencies(stream,&node,nullptr,1,cudaStreamSetCaptureDependencies));
        checked(cudaStreamBeginCaptureToGraph(side,params.conditional.phGraph_out[0],nullptr,nullptr,0,cudaStreamCaptureModeThreadLocal));
        main=stream;body=side;return body;
    }
    void finish() {
        if(!body)return;
        cudaGraph_t graph=nullptr;const auto e=cudaStreamEndCapture(body,&graph);body=nullptr;
        if(e!=cudaSuccess)throw std::runtime_error("window conditional body capture failed");
    }
    ~WindowConditional(){if(body){cudaGraph_t g=nullptr;cudaStreamEndCapture(body,&g);}}
};
#else
struct WindowConditional {cudaStream_t begin(cudaStream_t s,const int*,int){return s;}void finish(){}};
#endif
void window_experts(const float* input,float* parts,const LatentMoeWeights& w,
                    int dim,int ff,int nt,float clamp,void* stream) {
    auto& d=*w.window;const int n=nt*kWindowTopK;
    const int batches=(n+kWindowStage-1)/kWindowStage;
    auto s=static_cast<cudaStream_t>(stream);
    char* p=static_cast<char*>(d.scratch);
    auto take=[&](std::size_t b){void* q=p;p+=(b+15)&~std::size_t(15);return q;};
    float* x=(float*)take((std::size_t)n*dim*4);
    float* gate=(float*)take((std::size_t)n*ff*4);
    float* up=(float*)take((std::size_t)n*ff*4);
    const auto xb=kernels::native_q8_1_bytes(dim,1),hb=kernels::native_q8_1_bytes(ff,1);
    void* xq=take(n*xb);void* hq=take(n*hb);
    if(d.cpu_input)window_cpu_input_k<<<1,256,0,s>>>(d.mailbox,input,d.cpu_input,nt*dim);
    window_repeat_k<<<grid_for((long long)n*dim),kBlock,0,s>>>(input,x,dim,n);
    kernels::native_quantize_q8_1_wide(x,xq,dim,n,stream);
    window_plan_k<<<1,128,0,s>>>(d.mailbox,d.index,(batches+1)*kWindowEntries);
    static const bool pair=[] {const char* e=std::getenv("STRATA_HY4_FLASH_WINDOW_PAIR");return !e||std::atoi(e)!=0;}();
    static const bool grouped=[] {const char* e=std::getenv("STRATA_HY4_FLASH_WINDOW_GROUPED");return e&&std::atoi(e)!=0;}();
    static const bool compact=[] {const char* e=std::getenv("STRATA_HY4_FLASH_WINDOW_COMPACT");return e&&std::atoi(e)!=0;}();
    static const bool compact_rows=[] {const char* e=std::getenv("STRATA_HY4_FLASH_WINDOW_COMPACT_ROWS");return e&&std::atoi(e)!=0;}();
    static const int stripes=[] {const char* e=std::getenv("STRATA_HY4_FLASH_WINDOW_STRIPES");return e?std::max(0,std::min(32,std::atoi(e))):0;}();
    for(int ph=0;ph<=batches;++ph) {
        if(ph)window_wait_k<<<1,32,0,s>>>(d.mailbox,ph-1);
        const int32_t* index=d.index+ph*kWindowEntries;
        WindowConditional conditional;
        const auto main=s;
        if(ph){s=conditional.begin(s,index,n);stream=s;}
        if(compact_rows && ph) {
            for(int t=0;t<nt;++t) {
                const int off=t*kWindowTopK;const auto* qi=(char*)xq+off*xb;
                auto* gh=gate+off*ff;auto* uh=up+off*ff;void* h=(char*)hq+off*hb;
                const bool fused=pair && w.gate_exps_type==w.up_exps_type &&
                    kernels::native_mmvq_grouped_compact_pair(w.gate_exps_type,d.gate,d.gate_stride,d.up,d.up_stride,
                        qi,xb,gh,uh,dim,ff,kWindowTopK,stream,index+off,1);
                if(!fused) {
                    kernels::native_mmvq_grouped_compact_miss(w.gate_exps_type,d.gate,d.gate_stride,qi,xb,gh,dim,ff,kWindowTopK,stream,index+off,1);
                    kernels::native_mmvq_grouped_compact_miss(w.up_exps_type,d.up,d.up_stride,qi,xb,uh,dim,ff,kWindowTopK,stream,index+off,1);
                }
                kernels::native_swiglu_clamp_quantize_q8_1_masked(gh,uh,h,ff,kWindowTopK,clamp,index+off,stream);
                kernels::native_mmvq_grouped_compact_miss(w.down_exps_type,d.down,d.down_stride,h,hb,parts+off*dim,ff,dim,kWindowTopK,stream,index+off,1);
            }
            conditional.finish();s=main;stream=main;
            window_consumed_k<<<1,32,0,s>>>(d.mailbox,ph-1);
            continue;
        }
        const bool striped=stripes && !ph && w.gate_exps_type==w.up_exps_type &&
            kernels::native_mmvq_window_striped_pair(w.gate_exps_type,d.gate,d.gate_stride,d.up,d.up_stride,
                xq,xb,gate,up,dim,ff,n,stream,index,stripes);
        const bool fused=striped || (grouped && w.gate_exps_type==w.up_exps_type &&
            kernels::native_mmvq_window_pair(w.gate_exps_type,d.gate,d.gate_stride,d.up,d.up_stride,
                xq,xb,gate,up,dim,ff,n,index,nt,stream));
        const bool paired=fused || (pair && w.gate_exps_type==w.up_exps_type &&
            kernels::native_mmvq_grouped_pair(w.gate_exps_type,d.gate,d.gate_stride,d.up,d.up_stride,
                xq,xb,gate,up,dim,ff,n,stream,index,1));
        if(!paired) {
            kernels::native_mmvq_grouped_context(w.gate_exps_type,d.gate,d.gate_stride,xq,xb,gate,dim,ff,n,stream,index,1);
            kernels::native_mmvq_grouped_context(w.up_exps_type,d.up,d.up_stride,xq,xb,up,dim,ff,n,stream,index,1);
        }
        kernels::native_swiglu_clamp_quantize_q8_1_masked(gate,up,hq,ff,n,clamp,index,stream);
        const bool striped_down=stripes && !ph && kernels::native_mmvq_window_striped(
            w.down_exps_type,d.down,d.down_stride,hq,hb,parts,ff,dim,n,stream,index,stripes);
        if(!striped_down)(compact && ph ? kernels::native_mmvq_grouped_compact_miss : kernels::native_mmvq_grouped_context)(
            w.down_exps_type,d.down,d.down_stride,hq,hb,parts,ff,dim,n,stream,index,1);
        conditional.finish();s=main;stream=main;
        if(ph)window_consumed_k<<<1,32,0,s>>>(d.mailbox,ph-1);
    }
    if(d.cpu_parts)window_cpu_parts_k<<<n,128,0,s>>>(d.mailbox,d.cpu_parts,parts,dim);
}
