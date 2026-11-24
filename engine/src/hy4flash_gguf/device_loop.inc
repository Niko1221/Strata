// CUDA13 fixed-window loop. Host services data requests, never launches rounds.
#if CUDART_VERSION >= 13000 && defined(STRATA_HY4_FLASH_GPU_SAMPLER)
namespace {
struct alignas(64) LoopEngram {
    volatile uint32_t request=0,ready=0,error=0;
    int base=0,token[8]={},previous[16]={};
};
__global__ void loop_target_setup_k(DeviceLoopState* s,const int* tokens,const int* history,
    int* pos,int* len,float* residual,const uint8_t* embedding,int dim,int hc,int T,int vocab,LoopEngram* service) {
    for(int i=blockIdx.x*blockDim.x+threadIdx.x;i<T*dim*hc;i+=gridDim.x*blockDim.x){
        const int t=i/(dim*hc),d=i%dim,id=tokens[t];
        if(id<0||id>=vocab){residual[i]=0.f;if(i==t*dim*hc)atomicExch((unsigned int*)&service->error,1u);continue;}
        const auto* b=embedding+((size_t)id*(dim/32)+d/32)*34;
        const uint16_t bits=uint16_t(b[0])|(uint16_t(b[1])<<8);
        residual[i]=__half2float(*reinterpret_cast<const __half*>(&bits))*(float)(int8_t)b[2+d%32];
    }
    if(!blockIdx.x&&!threadIdx.x){
        len[0]=s->base;len[1]=s->base+T;service->base=s->base;
        for(int i=0;i<T;++i){pos[i]=s->base+i;service->token[i]=tokens[i];
            for(int back=1;back<=2;++back){const int at=s->base+i-back;
                service->previous[i*2+2-back]=at>=s->base?tokens[i-back]:(at>=0?history[at]:0);}}
        __threadfence_system();service->request=service->request+1;
    }
}
__global__ void loop_engram_wait_k(LoopEngram* service) {
    if(!threadIdx.x)while(service->ready<service->request&&!service->error)__nanosleep(128);
    __threadfence_system();
}
__global__ void loop_hidden_k(const DeviceLoopState* s,const float* source,float* target,int dim,int T){
    for(int i=blockIdx.x*blockDim.x+threadIdx.x;i<T*dim;i+=gridDim.x*blockDim.x)
        target[(size_t)(s->base+1)*dim+i]=source[i];
}
__global__ void loop_switch_k(const DeviceLoopState* s,cudaGraphConditionalHandle handle){if(!threadIdx.x)cudaGraphSetConditional(handle,s->base%4);}
__global__ void loop_publish_k(DeviceLoopState* s,const RoundVerdict* v,int* history,LoopOutput* out,
    LoopEngram* service,int T,int max_base,cudaGraphConditionalHandle repeat) {
    if(threadIdx.x||blockIdx.x)return;
    if(service->error){s->nonfinite=1;cudaGraphSetConditional(repeat,0);return;}
    for(int i=0;i<v->count;++i){history[s->base+1+i]=v->tokens[i];out->tokens[s->produced+i]=v->tokens[i];}
    s->base=v->next_position;s->pending=v->next_token;s->produced+=v->count;s->remaining-=v->count;
    s->accepted+=v->accepted;s->offered+=T-1;++s->windows;s->stopped=v->stopped;s->nonfinite=v->nonfinite;
    __threadfence_system();out->count=s->produced;
    cudaGraphSetConditional(repeat,!s->stopped&&!s->nonfinite&&!out->cancel&&s->remaining>=T&&s->base+T<=max_base);
}
struct LoopResources {
    cudaStream_t stream=nullptr,body=nullptr,copy=nullptr;
    cudaGraph_t graph=nullptr;cudaGraphExec_t exec=nullptr;
    LoopEngram* host=nullptr;LoopEngram* dev=nullptr;
    LoopOutput* output=nullptr;LoopOutput* dout=nullptr;
    float* pinned=nullptr;
    ~LoopResources(){
        if(host)__atomic_store_n(&host->error,1u,__ATOMIC_RELEASE);
        if(output)__atomic_store_n(&output->cancel,1u,__ATOMIC_RELEASE);
        if(body){cudaStreamCaptureStatus status;
            if(cudaStreamIsCapturing(body,&status)==cudaSuccess&&status==cudaStreamCaptureStatusActive){cudaGraph_t g=nullptr;cudaStreamEndCapture(body,&g);}}
        if(stream)cudaStreamSynchronize(stream);if(copy)cudaStreamSynchronize(copy);
        if(exec)cudaGraphExecDestroy(exec);if(graph)cudaGraphDestroy(graph);
        for(auto st:{stream,body,copy})if(st)cudaStreamDestroy(st);
        if(host)cudaFreeHost(host);if(output)cudaFreeHost(output);if(pinned)cudaFreeHost(pinned);
    }
};
}
#endif
bool forward_device_loop(DenseResidency& dense,KvCache& cache,const Geometry& g,Mtp& mtp,GpuSampler& sampler,
    const std::vector<int32_t>& history,float* hidden,int remaining,int draft_tokens,const std::vector<int32_t>& stops,
    const std::function<bool()>& cancel,const std::function<void(int32_t)>& emit,DeviceLoopState& result,std::string& error) {
#if CUDART_VERSION >= 13000 && defined(STRATA_HY4_FLASH_GPU_SAMPLER)
    const int T=draft_tokens+1,base=cache.len;
    if(!dense.work||T<2||T>8||g.idx_kpool!=4||g.engram_ngram_size!=3||remaining<T||remaining>8192)return false;
    auto& w=*dense.work;auto& gg=dense.file->gg;
    if(!w.idx||w.window_generation!=w.sc.generation||!w.slices.drain_window()||!w.slices.drain_async())return false;
    for(int c=0;c<4;++c)if(!w.window_graph[T][c].source)return false;
    if(!mtp.loop_ready(T,w.idx_engaged))return false;
    // The last request may have ended with a T1/T2/T3 tail. Replaying a T4
    // target graph while the service still plans that tail deadlocks on phase B.
    if(!w.slices.prepare_window(gg,T,g.expert_dim,g.expert_ff,g.swiglu_clamp))return false;
    const auto* te=gg.info("token_embd.weight");if(!te||te->type!=8)return false;
    const auto* table=gg.info("per_layer_token_embd.weight");if(!table||table->type!=43)return false;
    const int max_base=w.idx_engaged?cache.max_tokens:std::min(cache.max_tokens,g.idx_top_k-g.idx_kpool);
    if(base+T>max_base)return false;
    auto ck=[](cudaError_t e){if(e!=cudaSuccess)throw std::runtime_error(std::string("hy4-flash device loop: ")+cudaGetErrorString(e));};
    Buf state(sizeof(DeviceLoopState)),tokens(8*4),hist((cache.max_tokens+1)*4);
    LoopResources r; // releases GPU waiters before the buffers above can retire
    try {
        sampler.prepare_loop(g.vocab_size,T,stops);
        for(auto p:{&r.stream,&r.body,&r.copy})ck(cudaStreamCreateWithFlags(p,cudaStreamNonBlocking));
        ck(cudaHostAlloc((void**)&r.host,sizeof(LoopEngram),cudaHostAllocMapped));new(r.host)LoopEngram{};
        ck(cudaHostGetDevicePointer((void**)&r.dev,r.host,0));
        ck(cudaHostAlloc((void**)&r.output,sizeof(LoopOutput),cudaHostAllocMapped));new(r.output)LoopOutput{};
        ck(cudaHostGetDevicePointer((void**)&r.dout,r.output,0));
        const int cols=g.engram_heads,dim=g.engram_head_dim,nb=g.trunk_blocks;
        const size_t row_count=(size_t)nb*T*cols*dim,valid_count=(size_t)nb*T*cols;
        ck(cudaHostAlloc((void**)&r.pinned,(row_count+valid_count)*4,cudaHostAllocDefault));
        result={base,remaining,history.back(),0,0,0,0,0,0};
        ck(cudaMemcpy(state.p,&result,sizeof result,cudaMemcpyHostToDevice));
        ck(cudaMemcpy(hist.p,history.data(),history.size()*4,cudaMemcpyHostToDevice));
        int embedding_type=0;const void* embedding=forward_weight(dense,"token_embd.weight",embedding_type);
        ck(cudaGraphCreate(&r.graph,0));
        cudaGraphConditionalHandle repeat;ck(cudaGraphConditionalHandleCreate(&repeat,r.graph,1,cudaGraphCondAssignDefault));
        cudaGraphNodeParams outer{};outer.type=cudaGraphNodeTypeConditional;outer.conditional.handle=repeat;
        outer.conditional.type=cudaGraphCondTypeWhile;outer.conditional.size=1;cudaGraphNode_t loop;
        ck(cudaGraphAddNode(&loop,r.graph,nullptr,nullptr,0,&outer));
        const auto body_graph=outer.conditional.phGraph_out[0];
        ck(cudaStreamBeginCaptureToGraph(r.body,body_graph,nullptr,nullptr,0,cudaStreamCaptureModeThreadLocal));
        mtp.record_loop_draft(static_cast<DeviceLoopState*>(state.p),static_cast<int*>(tokens.p),hidden,T,w.idx_engaged,r.body);
        loop_target_setup_k<<<(T*g.hc_width()+255)/256,256,0,r.body>>>(static_cast<DeviceLoopState*>(state.p),static_cast<int*>(tokens.p),static_cast<int*>(hist.p),
            static_cast<int*>(w.sc.buf[Scratch::kPos].p),static_cast<int*>(w.sc.buf[Scratch::kLen].p),static_cast<float*>(w.sc.buf[Scratch::kState].p),
            static_cast<const uint8_t*>(embedding),g.hidden_dim,g.hc_mult,T,g.vocab_size,r.dev);
        loop_engram_wait_k<<<1,1,0,r.body>>>(r.dev);
        cudaGraphConditionalHandle carry;ck(cudaGraphConditionalHandleCreate(&carry,body_graph,0,0));
        loop_switch_k<<<1,1,0,r.body>>>(static_cast<DeviceLoopState*>(state.p),carry);
        cudaStreamCaptureStatus status;const cudaGraphNode_t* deps;const cudaGraphEdgeData* edges;size_t count;
        ck(cudaStreamGetCaptureInfo(r.body,&status,nullptr,nullptr,&deps,&edges,&count));
        cudaGraphNodeParams sw{};sw.type=cudaGraphNodeTypeConditional;sw.conditional.handle=carry;sw.conditional.type=cudaGraphCondTypeSwitch;sw.conditional.size=4;
        cudaGraphNode_t choose;ck(cudaGraphAddNode(&choose,body_graph,deps,edges,count,&sw));
        for(int c=0;c<4;++c){cudaGraphNode_t target;ck(cudaGraphAddChildGraphNode(&target,sw.conditional.phGraph_out[c],nullptr,0,w.window_graph[T][c].source));}
        ck(cudaStreamUpdateCaptureDependencies(r.body,&choose,nullptr,1,cudaStreamSetCaptureDependencies));
        loop_hidden_k<<<(T*g.hidden_dim+255)/256,256,0,r.body>>>(static_cast<DeviceLoopState*>(state.p),static_cast<float*>(w.window_hidden.p),hidden,g.hidden_dim,T);
        const auto* v=sampler.record_loop(static_cast<float*>(w.sc.buf[Scratch::kLogits].p),g.vocab_size,T,static_cast<int*>(tokens.p),static_cast<int*>(hist.p),static_cast<DeviceLoopState*>(state.p),r.body);
        indexer_commit_loop(*w.idx,v,static_cast<DeviceLoopState*>(state.p),T,r.body);
        mtp.record_loop_commit(static_cast<DeviceLoopState*>(state.p),v,static_cast<int*>(tokens.p),hidden,T,r.body);
        loop_publish_k<<<1,1,0,r.body>>>(static_cast<DeviceLoopState*>(state.p),v,static_cast<int*>(hist.p),r.dout,r.dev,T,max_base,repeat);
        cudaGraph_t captured;ck(cudaStreamEndCapture(r.body,&captured));ck(cudaGraphInstantiate(&r.exec,r.graph,nullptr,nullptr,0));
        int device=0;ck(cudaGetDevice(&device));
        // CUPTI-instrumented GraphLaunch can block: a dedicated launch thread
        // leaves this thread available to serve Engram requests from the graph.
        strata::core::HostGraphLaunch launch(r.exec,r.stream,device,true,[&]{
            __atomic_store_n(&r.host->error,1u,__ATOMIC_RELEASE);
            __atomic_store_n(&r.output->cancel,1u,__ATOMIC_RELEASE);
            w.slices.window_fail("device loop aborted");
        });
        ck(launch.status());std::fprintf(stderr,"HY4_FLASH_DEVICE_LOOP launched T=%d base=%d\n",T,base);
        uint32_t served=0,emitted=0;auto progress=std::chrono::steady_clock::now();
        for(;;) {
            if(cancel())__atomic_store_n(&r.output->cancel,1u,__ATOMIC_RELEASE);
            const auto request=__atomic_load_n(&r.host->request,__ATOMIC_ACQUIRE);
            if(request!=served) {
                int pos[8];for(int t=0;t<T;++t)pos[t]=r.host->base+t;
                for(int il=0;il<nb;++il)if(g.layers[il].engram_table>=0) {
                    const auto& et=g.engram_tables[g.layers[il].engram_table];
                    EngramHashParams hp;hp.ngram_size=g.engram_ngram_size;hp.heads_per_ngram=g.engram_heads_per_ngram;
                    hp.columns=cols;hp.pad_token=g.engram_pad_token;hp.multipliers=et.multipliers;hp.primes=et.head_bucket_counts;hp.offsets=et.head_offsets;
                    hp.token_map=dense.token_map.data();hp.token_map_len=dense.token_map.size();
                    std::vector<int64_t> ids;std::vector<float> valid;
                    if(!engram_row_ids(hp,r.host->token,pos,r.host->previous,nullptr,T,ids,valid,error))throw std::runtime_error(error);
                    auto* rows=r.pinned+(size_t)il*T*cols*dim;auto* val=r.pinned+row_count+(size_t)il*T*cols;
                    std::copy(valid.begin(),valid.end(),val);
                    for(int j=0;j<T*cols;++j){const auto* raw=static_cast<const uint8_t*>(gg.at(gg.data_start()+table->offset+(uint64_t)ids[j]*dim,dim));
                        if(!raw)throw std::runtime_error("loop Engram mapping unavailable");dequant_f8_e4m3_row(raw,dim,rows+j*dim);}
                    ck(cudaMemcpyAsync(w.async_engram_rows[il].p,rows,(size_t)T*cols*dim*4,cudaMemcpyHostToDevice,r.copy));
                    ck(cudaMemcpyAsync(w.async_engram_valid[il].p,val,(size_t)T*cols*4,cudaMemcpyHostToDevice,r.copy));
                }
                ck(cudaStreamSynchronize(r.copy));__atomic_store_n(&r.host->ready,request,__ATOMIC_RELEASE);served=request;progress=std::chrono::steady_clock::now();
            }
            const auto available=__atomic_load_n(&r.output->count,__ATOMIC_ACQUIRE);
            while(emitted<available)emit(r.output->tokens[emitted++]);
            if(w.slices.window_exec.failed.load())throw std::runtime_error("loop expert service failed");
            const auto status=launch.query();if(status==cudaSuccess)break;if(status!=cudaErrorNotReady)ck(status);
            if(std::chrono::steady_clock::now()-progress>std::chrono::seconds(60))throw std::runtime_error("loop progress timeout");
            std::this_thread::yield();
        }
        ck(launch.finish());ck(cudaMemcpy(&result,state.p,sizeof result,cudaMemcpyDeviceToHost));
        while(emitted<r.output->count)emit(r.output->tokens[emitted++]);
        if(!w.slices.drain_window())throw std::runtime_error("loop drain failed");
        cache.len=result.base;w.idx->valid_pos=result.base;w.idx->carry=result.base%4;w.idx->n_pool=result.base/4;mtp.loop_mirror(result.base);
        std::fprintf(stderr,"HY4_FLASH_DEVICE_LOOP finished windows=%d outputs=%d accepted=%d offered=%d\n",result.windows,result.produced,result.accepted,result.offered);
        return true;
    }catch(const std::exception& e){error=e.what();w.slices.window_fail(error.c_str());return false;}
#else
    error="device loop requires CUDA13";return false;
#endif
}
