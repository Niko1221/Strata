// Included inside ExpertSlices: one owner of the existing expert cache metadata.
// The T1 worker is drained before a window is submitted, and vice versa.
struct WindowLayer {
    WindowMailbox* host=nullptr;
    WindowExperts device;
    cpu::NativeFmt cpu_fmt{};
    uint64_t base[3]={}; std::size_t stride[3]={};
    int block=-1, pool=-1;
    std::atomic<uint32_t> served{0};
    ~WindowLayer() {
        if(device.index)cudaFree(device.index);
        if(host)cudaFreeHost(host);
    }
};
struct WindowCpu {
    std::thread thread;
    std::mutex mutex;
    std::condition_variable cv;
    WindowLayer* pending=nullptr;
    uint32_t epoch=0;
    int count=0, tokens=0;
    struct Balance { int limit=0,samples=0,late=0; } balance[kWindowTokens+1];
    // Balance measured CPU and GPU branch times when assigning cache misses.
    strata::hy4_flash::PcieModel cost{};
    double cpu_ms=0;                 ///< the CPU branch of the layer in flight, written by the worker
    unsigned long long fits=0, solves=0;
    double cpu_total=0, pcie_total=0;        ///< the two branches' host-side durations, reported
    double cpu_stall=0, pcie_stall=0;        ///< and what the GPU actually waited out, which drives the fit
    CpuExperts::WindowJob jobs[kWindowEntries];
    float* host=nullptr;
    float* device=nullptr;
    size_t input_elements=0;
    unsigned long long experts=0, entries=0, mixed_windows=0;
    bool late=false;
    ~WindowCpu(){if(host)cudaFreeHost(host);}
};
struct Hy4FlashWindowExecutor {
    std::vector<std::unique_ptr<WindowLayer>> layers;
    Buf scratch;
    std::unique_ptr<WindowCpu> cpu;
    uint8_t* staging=nullptr;  // bounded host copy when file-page registration is unavailable
    std::thread worker;
    std::atomic<bool> stop{false}, active{false};
    std::atomic<bool> started{false};
    std::atomic<uint32_t> failed{0};
    int device=0, tokens=0;
    double cycles_per_ms=0;   ///< the device clock, for the mailbox's stall counters
    unsigned long long windows=0, entries=0, unique=0, transfers=0, bytes=0;
    ~Hy4FlashWindowExecutor() { if(worker.joinable())std::terminate(); if(staging)cudaFreeHost(staging); }
} window_exec;

void window_fail(const char* why) {
    if(!window_exec.failed.exchange(1))std::fprintf(stderr,"HY4_FLASH_WINDOW_ERROR %s\n",why);
    for(auto& a:window_exec.layers)if(a)flag_store(&a->host->error,1);
}
bool drain_window() {
    if(!window_exec.worker.joinable())return true;
    const auto until=std::chrono::steady_clock::now()+std::chrono::seconds(30);
    for(;;) {
        if(window_exec.failed.load())return false;
        bool pending=window_exec.active.load();
        for(auto& a:window_exec.layers)if(a && flag_load(&a->host->request)!=a->served.load())pending=true;
        if(!pending)return true;
        if(std::chrono::steady_clock::now()>until){window_fail("drain timeout");return false;}
        std::this_thread::yield();
    }
}
bool wait_window_layer(int il,uint32_t epoch) {
    const auto until=std::chrono::steady_clock::now()+std::chrono::seconds(30);
    auto& a=*window_exec.layers[il];
    while(a.served.load()<epoch) {
        if(window_exec.failed.load())return false;
        if(std::chrono::steady_clock::now()>until){window_fail("request completion timeout");return false;}
        std::this_thread::yield();
    }
    return !window_exec.failed.load();
}
void stop_window() {
    if(!window_exec.worker.joinable() && !(window_exec.cpu && window_exec.cpu->thread.joinable()))return;
    if(!drain_window())window_fail("shutdown after failed window");
    window_exec.stop.store(true);
    for(auto& a:window_exec.layers)if(a)flag_store(&a->host->error,1);
    if (auto* c=window_exec.cpu.get()) {
        // Pair the stop predicate with the enqueue mutex to avoid a lost wakeup.
        { std::lock_guard<std::mutex> lock(c->mutex); c->cv.notify_all(); }
        if(c->thread.joinable())c->thread.join();
    }
    if(window_exec.worker.joinable())window_exec.worker.join();
    if(const auto* c=window_exec.cpu.get()) {
        std::fprintf(stderr,"HY4_FLASH_WINDOW_CPU_STATS layers=%llu unique_experts=%llu entries=%llu result_bytes=%llu\n",
                     c->mixed_windows,c->experts,c->entries,c->entries*c->input_elements/kWindowTokens*sizeof(float));
        std::fprintf(stderr,"HY4_FLASH_WINDOW_SPLIT fits=%llu solves=%llu ready=%d cpu_ms=%.1f pcie_ms=%.1f "
                     "cpu_stall_ms=%.1f pcie_stall_ms=%.1f fit a=%.4f c=%.4f d=%.4f g0=%.4f g=%.4f\n",
                     c->fits,c->solves,(int)c->cost.ready(),c->cpu_total,c->pcie_total,
                     c->cpu_stall,c->pcie_stall,c->cost.a,c->cost.c,c->cost.d,c->cost.g0,c->cost.g);
    }
    cudaStreamSynchronize(copy);cudaDeviceSynchronize();
    std::fprintf(stderr,"HY4_FLASH_WINDOW_STATS layers=%llu entries=%llu unique=%llu transfers=%llu h2d_bytes=%llu\n",
        window_exec.windows,window_exec.entries,window_exec.unique,window_exec.transfers,window_exec.bytes);
}
void window_cpu_loop() {
    auto& c=*window_exec.cpu;
    try {
        for (;;) {
            WindowLayer* layer;
            uint32_t epoch;
            int tokens,count;
            {
                std::unique_lock<std::mutex> lock(c.mutex);
                c.cv.wait(lock,[&]{return c.pending || window_exec.stop.load();});
                if(window_exec.stop.load())return;
                layer=c.pending; epoch=c.epoch; tokens=c.tokens; count=c.count;
                c.pending=nullptr;
            }
            auto& m=*layer->host;
            const auto until=std::chrono::steady_clock::now()+std::chrono::seconds(30);
            while(flag_load(&m.cpu_input_ready)!=epoch) {
                if(window_exec.stop.load() || window_exec.failed.load())return;
                if(std::chrono::steady_clock::now()>until){window_fail("CPU window input timeout");return;}
                std::this_thread::yield();
            }
            std::string error;
            // The wait above is the GPU's time, not this branch's: the clock starts at the compute.
            const auto started=std::chrono::steady_clock::now();
            if(!cpu_experts->compute_window(layer->cpu_fmt,c.jobs,count,tokens,c.host,c.host+c.input_elements,error)) {
                window_fail(error.c_str()); return;
            }
            c.cpu_ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-started).count();
            const int phases=(tokens*kWindowTopK+kWindowStage-1)/kWindowStage;
            c.late=flag_load(&m.consumed[phases-1])==epoch;
            flag_store(&m.cpu_ready,epoch);
        }
    } catch(const std::exception& e){window_fail(e.what());}
}

// Plan on distinct experts before assigning GPU victims. Each CPU expert stays in its
// existing RAM slot until the GPU has consumed every output belonging to this layer.
int plan_window_cpu(WindowLayer& a,int tokens,const int* distinct,int nd,const int* owner) {
    auto& c=*window_exec.cpu;
    auto& m=*a.host;
    const auto& q=pools[a.pool];
    const int free=std::max(0,q.n-q.overflow-q.next_free);
    int missing=0,misses_per_row[kWindowTokens]={},room[kWindowTokens]={};
    for(int j=0;j<nd;++j)if(m.index[0][distinct[j]]<0)++missing;
    for(int i=0;i<tokens*kWindowTopK;++i)if(m.index[0][i]<0)++misses_per_row[i/kWindowTopK];
    auto& balance=c.balance[tokens];
    if(!balance.limit)balance.limit=std::min(cpu_max,std::max(1,kWindowTopK-int(kWindowTopK*cpu_gpu_frac)));
    // Fit both branches before solving for the split that balances layer time.
    static const bool balance_on=[] {
        const char* e=std::getenv("STRATA_HY4_FLASH_WINDOW_BALANCE");
        return e==nullptr || std::atoi(e)!=0;
    }();
    const bool modeled=balance_on && cpu_split==CpuSplit::Auto && c.cost.ready();
    int layer_room=INT_MAX;
    if(modeled)layer_room=std::max(0,missing-c.cost.pick(missing,0));
    for(int t=0;t<tokens;++t) {
        room[t]=cpu_max;
        if(cpu_split==CpuSplit::Ratio)room[t]=std::min(cpu_max,misses_per_row[t]-int(misses_per_row[t]*cpu_gpu_frac));
        if(cpu_split==CpuSplit::Auto)room[t]=modeled?cpu_max:balance.limit;
        m.cpu_mask[t]=0;
    }
    const int allowed=cpu_cache_first?std::max(0,missing-free):missing;
    int n=0;
    for(int j=nd-1;j>=0 && n<allowed && layer_room>0;--j) {
        const int i=distinct[j];
        if(m.index[0][i]>=0)continue;
        auto& job=c.jobs[n];
        job.count=0;
        bool fits=true;
        for(int r=i;r<tokens*kWindowTopK;++r)if(owner[r]==i) {
            if(room[r/kWindowTopK]==0){fits=false;break;}
            if(job.count==tokens)throw std::runtime_error("duplicate expert in window row");
            job.entries[job.count++]=r;
        }
        if(!fits)continue;
        const int id=m.ids[i],key=a.block*n_expert+id;
        for(int p=0;p<3;++p) {
            job.plane[p]=exclusive?ram_plane(key,p):static_cast<const uint8_t*>(gg->at(a.base[p]+uint64_t(id)*a.stride[p],a.stride[p]));
            if(!job.plane[p])throw std::runtime_error("missing CPU window expert plane");
        }
        for(int r=0;r<job.count;++r) {
            const int entry=job.entries[r],t=entry/kWindowTopK;
            m.cpu_mask[t]|=1u<<(entry%kWindowTopK);
            --room[t];
        }
        c.entries+=job.count;
        ++n;
        --layer_room;
    }
    c.experts+=n;
    if(n)++c.mixed_windows;
    return n;
}
void window_loop() {
    if(cudaSetDevice(window_exec.device)!=cudaSuccess){window_fail("worker device");window_exec.started.store(true);return;}
    window_exec.started.store(true);
    // Keep the service thread responsive to per-layer expert requests.
    std::size_t cursor=0;
    unsigned spins=0;
    static const unsigned spin_cap=[] {
        const char* e=std::getenv("STRATA_HY4_FLASH_WINDOW_SPIN");
        const int v=e?std::atoi(e):4096;
        return v<0?0u:(unsigned)v;
    }();
    try {
    while(!window_exec.stop.load()) {
        bool found=false;
        const std::size_t n=window_exec.layers.size();
        for(std::size_t t=0;t<n;++t) {
            auto& ptr=window_exec.layers[(cursor+t)%n];
            if(!ptr)continue;
            auto& a=*ptr; auto& m=*a.host;
            const uint32_t epoch=flag_load(&m.request);
            if(epoch==a.served.load())continue;
            found=true;window_exec.active.store(true);
            // This branch's own clock: planning plus the staged copies, never the waits for the GPU.
            const auto branch_t0=std::chrono::steady_clock::now();
            double pcie_ms=0;
            cursor=(cursor+t+1)%n;
            static const bool trace=std::getenv("HY4_FLASH_WINDOW_TRACE")!=nullptr;
            if(trace)std::fprintf(stderr,"WINDOW request layer=%d epoch=%u\n",a.block,epoch);
            if(epoch!=a.served.load()+1 || !epoch){window_fail("invalid epoch");return;}
            static const int abort_layer=[] {const char* e=std::getenv("HY4_FLASH_WINDOW_ABORT_LAYER");return e?std::atoi(e):-1;}();
            if(a.block==abort_layer){window_fail("injected supply abort");return;}
            const int count=window_exec.tokens*kWindowTopK;
            const int phases=(count+kWindowStage-1)/kWindowStage;
            auto& q=pools[a.pool];const int permanent=q.n-q.overflow;
            int ids[kWindowEntries],owner[kWindowEntries],distinct[kWindowEntries],nd=0;
            for(int ph=0;ph<=kWindowBatches;++ph)std::fill_n(m.index[ph],kWindowEntries,-1);
            if(++group_mark==0){std::fill(claimed.begin(),claimed.end(),0);group_mark=1;}
            // Claim every hit before selecting any victim. Duplicates share ownership.
            for(int i=0;i<count;++i) {
                const int id=m.ids[i];if(id<0||id>=n_expert){window_fail("invalid expert ID");return;}
                ids[i]=id;owner[i]=i;
                for(int j=0;j<i;++j)if(ids[j]==id){owner[i]=owner[j];break;}
                const int key=a.block*n_expert+id,slot=slot_of[key];
                if(hot_cache)heat.observe(key);
                if(owner[i]==i)distinct[nd++]=i;
                if(slot>=0) {
                    if(slot>=permanent||key_of[q.g0+slot]!=key){window_fail("hit owner mismatch");return;}
                    claimed[q.g0+slot]=group_mark;used[q.g0+slot]=1;m.index[0][i]=slot;++hits;
                } else ++misses;
            }
            auto* cpu=window_exec.cpu.get();
            const int tokens=window_exec.tokens;
            const int cpu_count=cpu?plan_window_cpu(a,tokens,distinct,nd,owner):0;
            if(cpu_count) {
                std::lock_guard<std::mutex> lock(cpu->mutex);
                cpu->pending=&a; cpu->epoch=epoch; cpu->tokens=tokens; cpu->count=cpu_count;
                cpu->cv.notify_one();
            }
            std::vector<Miss> jobs; jobs.reserve(nd);
            for(int j=0;j<nd;++j) {
                const int i=distinct[j];if(m.index[0][i]>=0)continue;
                if(cpu_count && (m.cpu_mask[i/kWindowTopK] & (1u<<(i%kWindowTopK))))continue;
                const int key=a.block*n_expert+ids[i];int slot=-1;
                if(q.next_free<permanent)slot=q.next_free++;
                else if(hot_cache) {
                    slot=heat.victim(key,q.g0,permanent,q.hot_hand,key_of,claimed,group_mark);
                    if(slot<0)++hot_rejected;else ++hot_admitted;
                }
                int old=-1;
                if(slot>=0) {
                    old=key_of[q.g0+slot];if(old>=0){slot_of[old]=-1;++evictions;}
                    slot_of[key]=slot;key_of[q.g0+slot]=key;claimed[q.g0+slot]=group_mark;used[q.g0+slot]=1;
                } else slot=permanent+(int)(jobs.size()%kWindowStage);
                const int ph=1+(int)jobs.size()/kWindowStage;
                for(int r=i;r<count;++r)if(owner[r]==i)m.index[ph][r]=slot;
                jobs.push_back({ids[i],slot,0,old});
            }
            async_map_dirty=true;
            flag_store(&m.plan_ready,epoch);
            pcie_ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-branch_t0).count();
            if(trace)std::fprintf(stderr,"WINDOW plan layer=%d jobs=%zu\n",a.block,jobs.size());
            for(int ph=0;ph<phases;++ph) {
                if(ph) {
                    const auto until=std::chrono::steady_clock::now()+std::chrono::seconds(30);
                    while(flag_load(&m.consumed[ph-1])!=epoch) {
                        if(window_exec.stop.load()||window_exec.failed.load())return;
                        if(std::chrono::steady_clock::now()>until){window_fail("staging consumer timeout");return;}
                        std::this_thread::yield();
                    }
                }
                const auto copy_t0=std::chrono::steady_clock::now();
                { OffloadNvtx range("hy4-flash.window_h2d");
                const int first=ph*kWindowStage, end=std::min((ph+1)*kWindowStage,(int)jobs.size());
                if (first<end) {
                    if (exclusive) {
                        const std::vector<Miss> batch(jobs.begin()+first,jobs.begin()+end);
                        if (!transfer_exclusive(a.block,a.stride,batch)) {window_fail("expert exchange");return;}
                    } else {
                        // When the mmap cannot be pinned, copy only this phase into page-locked RAM.
                        // The copy-stream fence below also protects reuse of the staging buffer.
                        auto source = [&](int j,int r) {
                            return gg->at(a.base[r]+uint64_t(jobs[j].expert)*a.stride[r],a.stride[r]);
                        };
                        auto stage = [&](int j,int r) {
                            size_t offset=(j-first)*q.slot_bytes;
                            for(int k=0;k<r;++k)offset+=q.stride_r[k];
                            return window_exec.staging+offset;
                        };
                        if(window_exec.staging) {
                            std::atomic<bool> bad{false};
                            pool->run(end-first,[&](int i) {
                                for(int r=0;r<3;++r) {
                                    const void* src=source(first+i,r);
                                    if(!src){bad.store(true);return;}
                                    std::memcpy(stage(first+i,r),src,a.stride[r]);
                                }
                            });
                            if(bad.load()){window_fail("expert staging");return;}
                        }
                        for(int j=first;j<end;++j)for(int r=0;r<3;++r) {
                            const void* src=window_exec.staging?stage(j,r):source(j,r);
                            char* dst=static_cast<char*>(tier.p)+q.plane[r]+jobs[j].slot*q.stride_r[r];
                            if(!src||gg->copy_expert_async(dst,src,a.stride[r],copy)!=cudaSuccess){
                                window_fail("expert DMA");return;
                            }
                        }
                        if(cudaStreamSynchronize(copy)!=cudaSuccess){window_fail("DMA completion");return;}
                        n_fetch+=end-first;
                        h2d_bytes+=(end-first)*(a.stride[0]+a.stride[1]+a.stride[2]);
                    }
                    window_exec.transfers+=end-first;
                    window_exec.bytes+=(end-first)*(a.stride[0]+a.stride[1]+a.stride[2]);
                }
                }
                pcie_ms+=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-copy_t0).count();
                flag_store(&m.ready[ph],epoch);
                if(trace)std::fprintf(stderr,"WINDOW ready layer=%d phase=%d\n",a.block,ph);
            }
            const auto until=std::chrono::steady_clock::now()+std::chrono::seconds(30);
            while(flag_load(&m.done)!=epoch) {
                if(window_exec.stop.load()||window_exec.failed.load())return;
                if(std::chrono::steady_clock::now()>until){window_fail("layer completion timeout");return;}
                std::this_thread::yield();
            }
            if(cpu_count) {
                // Acquire the CPU worker's last publication before reusing its descriptors.
                if(flag_load(&m.cpu_ready)!=epoch){window_fail("CPU window epoch mismatch");return;}
                cpu->cpu_total+=cpu->cpu_ms;
                if(cpu_split==CpuSplit::Auto) {
                    auto& b=cpu->balance[tokens];
                    b.late+=cpu->late;
                    if(++b.samples==64) {
                        if(b.late>48)b.limit=std::max(1,b.limit-1);
                        if(b.late<16)b.limit=std::min(cpu_max,b.limit+1);
                        b.samples=b.late=0;
                    }
                }
            }
            if(cpu && cpu_split==CpuSplit::Auto) {
                // Charge only transfers that stall GPU progress when estimating offload cost.
                const double k=window_exec.cycles_per_ms>0?window_exec.cycles_per_ms:1.0;
                const double pcie_stall=double(m.stall[0])/k, cpu_stall=double(m.stall[1])/k;
                cpu->cost.on=true;
                cpu->cost.sample(cpu_count,(int)jobs.size(),cpu_stall,pcie_stall);
                cpu->pcie_total+=pcie_ms;
                cpu->cpu_stall+=cpu_stall; cpu->pcie_stall+=pcie_stall;
                if(++cpu->fits%64==0){cpu->cost.solve();++cpu->solves;}
            }
            ++window_exec.windows;window_exec.entries+=count;window_exec.unique+=nd;
            a.served.store(epoch);window_exec.active.store(false);
        }
        if(window_exec.failed.load())return;
        if(found)spins=0;
        else if(spins<spin_cap){++spins;
#if defined(__x86_64__) || defined(__i386__)
            __builtin_ia32_pause();
#endif
        } else {spins=0;std::this_thread::yield();}
    }
    }catch(const std::exception& e){window_fail(e.what());}
}
bool prepare_window(Gguf& file,int tokens,int expert_dim,int expert_ff,float clamp) {
    if(!drain_async()||!drain_window())return false;
    for(int pi=0;pi<n_pools;++pi)if(pools[pi].overflow<kWindowStage)return false;
    if(window_exec.layers.empty()) {
        // A returned prefill loan can enter verify before any ordinary open_block call.
        gg=&file;
        if(!ensure_copy())return false;
        if(cudaStreamSynchronize(copy)!=cudaSuccess)return false;
        cudaStreamDestroy(copy);
        if(cudaStreamCreateWithFlags(&copy,cudaStreamNonBlocking)!=cudaSuccess)return false;
        if(!exclusive && !file.expert_pin_ready() && !window_exec.staging) {
            if(!pool || cudaHostAlloc((void**)&window_exec.staging,kWindowStage*slot_bytes,cudaHostAllocDefault)!=cudaSuccess)
                return false;
        }
        if (!configure_cpu_experts()) return false;
        std::unique_ptr<WindowCpu> cpu;
        if (cpu_max) {
            cpu=std::make_unique<WindowCpu>();
            auto& c=*cpu;
            c.input_elements=size_t(kWindowTokens)*expert_dim;
            if(cudaHostAlloc((void**)&c.host,c.input_elements*(1+kWindowTopK)*sizeof(float),cudaHostAllocMapped)!=cudaSuccess ||
               cudaHostGetDevicePointer((void**)&c.device,c.host,0)!=cudaSuccess)return false;
            std::fprintf(stderr,"HY4_FLASH_WINDOW_CPU enabled cap_per_token=%d\n",cpu_max);
        }
        std::vector<std::unique_ptr<WindowLayer>> layers(pool_of.size());
        // Repeated input/Q8 and per-entry GU/quantized intermediate, shared by all layers.
        const std::size_t bytes=(std::size_t)kWindowEntries*(expert_dim*4+2*expert_ff*4+
            kernels::native_q8_1_bytes(expert_dim,1)+kernels::native_q8_1_bytes(expert_ff,1))+1024;
        window_exec.scratch=Buf(bytes);if(!window_exec.scratch.p)return false;
        for(std::size_t il=0;il<pool_of.size();++il)if(pool_of[il]>=0) {
            auto a=std::make_unique<WindowLayer>();a->block=(int)il;a->pool=pool_of[il];
            const char* roles[]={"ffn_gate_exps.weight","ffn_up_exps.weight","ffn_down_exps.weight"};
            for(int r=0;r<3;++r){const auto* ti=file.info(blk(il)+roles[r]);if(!ti)return false;
                a->base[r]=file.data_start()+ti->offset;a->stride[r]=tensor_payload_bytes(*ti)/n_expert;}
            if(auto* c=cpu.get()) {
                const auto* gate=file.info(blk(il)+roles[0]);
                const auto* up=file.info(blk(il)+roles[1]);
                const auto* down=file.info(blk(il)+roles[2]);
                std::string why;
                if(gate->type!=up->type || !CpuExperts::make_format(gate->type,down->type,expert_dim,expert_ff,clamp,a->cpu_fmt,why,
                                                                     cpu_gpu_quant))return false;
                a->device.cpu_input=c->device;
                a->device.cpu_parts=c->device+c->input_elements;
            }
            if(cudaHostAlloc((void**)&a->host,sizeof(WindowMailbox),cudaHostAllocMapped)!=cudaSuccess)return false;
            std::memset(a->host,0,sizeof(WindowMailbox));
            if(cudaHostGetDevicePointer((void**)&a->device.mailbox,a->host,0)!=cudaSuccess ||
               cudaMalloc((void**)&a->device.index,sizeof(a->host->index))!=cudaSuccess)return false;
            auto& q=pools[a->pool];
            a->device.gate=static_cast<char*>(tier.p)+q.plane[0];a->device.gate_stride=q.stride_r[0];
            a->device.up=static_cast<char*>(tier.p)+q.plane[1];a->device.up_stride=q.stride_r[1];
            a->device.down=static_cast<char*>(tier.p)+q.plane[2];a->device.down_stride=q.stride_r[2];
            a->device.scratch=window_exec.scratch.p;
            layers[il]=std::move(a);
        }
        if(cudaGetDevice(&window_exec.device)!=cudaSuccess)return false;
        int khz=0;   // cudaDevAttrClockRate is kHz, which is cycles a millisecond
        if(cudaDeviceGetAttribute(&khz,cudaDevAttrClockRate,window_exec.device)==cudaSuccess&&khz>0)
            window_exec.cycles_per_ms=khz;
        window_exec.layers=std::move(layers);
        window_exec.cpu=std::move(cpu);
        std::fprintf(stderr,"HY4_FLASH_WINDOW_SUPPLY initialized scratch_bytes=%zu staging=existing-overflow source=%s\n",bytes,
            exclusive?"gpu-ram":window_exec.staging?"staged-mmap":"pinned-mmap");
    }
    window_exec.tokens=tokens;
    for(auto& a:window_exec.layers)if(a)a->device.tokens=tokens;
    if(!window_exec.worker.joinable()) {
        if(window_exec.cpu)window_exec.cpu->thread=std::thread([this]{window_cpu_loop();});
        window_exec.worker=std::thread([this]{window_loop();});
        while(!window_exec.started.load())std::this_thread::yield();
    }
    return !window_exec.failed.load();
}
