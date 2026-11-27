#include "strata/hy4flash_gguf/mtp.hpp"
#include "strata/hy4flash_gguf/model_file.hpp"
#include "strata/hy4flash_gguf/geometry.hpp"
#include "strata/hy4flash_gguf/forward.hpp"
#include "strata/hy4flash_gguf/attention.hpp"
#include "strata/hy4flash_gguf/cpu_experts.hpp"
#include "strata/hy4flash_gguf/window.hpp"
#include "strata/hy4flash_gguf/indexer.hpp"
#include "strata/hy4flash_gguf/kv_cache.hpp"
#include "strata/hy4flash_gguf/round_state.hpp"
#include "strata/kernels/bf16_gemv.hpp"
#include "strata/kernels/elementwise.hpp"
#include "strata/kernels/native_mmvq.hpp"
#include "strata/kernels/native_router.hpp"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <map>
#include <fstream>
#include <stdexcept>
#include <thread>
#include <new>
#include <climits>

namespace strata::hy4_flash {
namespace {
#include "gpu_pick.cuh"
void check(cudaError_t error, const char* operation = "hy4-flash MTP") {
    if (error != cudaSuccess) throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(error));
}
struct Buffer {
    void* p = nullptr;
    std::size_t bytes = 0;
    ~Buffer() { if (p) cudaFree(p); }
    void reserve(std::size_t n) {
        if (n <= bytes) return;
        if (p) check(cudaFree(p));
        p = nullptr; bytes = 0;
        check(cudaMalloc(&p, n), "allocate MTP device buffer"); bytes = n;
    }
    template<class T = float> T* as() { return static_cast<T*>(p); }
};
__device__ float bf16(uint16_t bits) { return __uint_as_float(unsigned(bits) << 16); }
__device__ float sum_warp(float v) {
    for (int d = 16; d; d >>= 1) v += __shfl_xor_sync(0xffffffff, v, d);
    return v;
}
__global__ void draft_embedding_k(const uint8_t* table,const int32_t* token,float* out,int dim,int vocab,int nt=1) {
    for(int at=blockIdx.x*blockDim.x+threadIdx.x;at<dim*nt;at+=blockDim.x*gridDim.x) {
        const int id=token[at/dim],i=at%dim;
        if(id<0||id>=vocab){out[at]=0;continue;}
        const uint8_t* b=table+((size_t)id*(dim/32)+i/32)*34;
        out[at]=__half2float(*reinterpret_cast<const __half*>(b))*(float)(int8_t)b[2+i%32];
    }
}
__global__ void draft_advance_k(int* pos,int* len){if(!threadIdx.x){++*pos;++len[0];++len[1];}}
#if CUDART_VERSION >= 13000 && !defined(STRATA_USE_HIP)
__global__ void continue_choice_k(const RoundVerdict* v,const int* meta,cudaGraphConditionalHandle handle) {
    if(!threadIdx.x)cudaGraphSetConditional(handle, v->count>0&&!v->stopped&&!v->nonfinite&&v->count+1<meta[1]?v->count-1:8);
}
#endif
__global__ void continue_setup_k(const RoundVerdict* v,const int* input,int* tokens,int* pos,int* len,
    int* counters,float* previous,const float* target,int dim,const int* meta,int nt,bool draft) {
    const int base=meta[0];
    for(int i=blockIdx.x*blockDim.x+threadIdx.x;i<nt*dim;i+=gridDim.x*blockDim.x)
        previous[i]=target[(size_t)(draft?v->next_position:base)*dim+i];
    if(!blockIdx.x&&!threadIdx.x) {
        const int at=draft?v->next_position:base;
        for(int i=0;i<nt;++i){tokens[i]=draft?v->next_token:input[i];pos[i]=at+i;}
        len[0]=at;len[1]=at+nt;counters[0]=at%4;counters[1]=at/4;
    }
}

__global__ void loop_mtp_setup_k(const DeviceLoopState* state,const int* input,int* tokens,int* pos,int* len,
    int* counters,float* previous,const float* target,int dim,int nt,bool draft) {
    const int base=state->base;
    for(int i=blockIdx.x*blockDim.x+threadIdx.x;i<nt*dim;i+=gridDim.x*blockDim.x)
        previous[i]=target[(size_t)base*dim+i];
    if(!blockIdx.x&&!threadIdx.x){for(int i=0;i<nt;++i){tokens[i]=draft?state->pending:input[i];pos[i]=base+i;}
        len[0]=base;len[1]=base+nt;counters[0]=base%4;counters[1]=base/4;}
}
#if CUDART_VERSION >= 13000 && !defined(STRATA_USE_HIP)
__global__ void loop_keep_choice_k(const RoundVerdict* v,cudaGraphConditionalHandle handle) {
    if(!threadIdx.x)cudaGraphSetConditional(handle,v->count>0?v->count-1:8);
}
#endif

// Indexed, all-resident BF16 experts. Two accumulators share an input for GU;
// the down pass reads each rank's activation. No host routing or weight DMA.
template<bool Pair, bool Down>
__global__ void expert_mm(const float* x, const uint16_t* w, const uint16_t* w2,
                          const int32_t* ids, float* y, float* y2,
                          int n_in, int n_out, int k) {
    const int row = blockIdx.x, rank = blockIdx.y, token = blockIdx.z;
    const int expert = ids[token * k + rank];
    const float* input = x + (std::size_t) (Down ? token * k + rank : token) * n_in;
    const std::size_t offset = ((std::size_t) expert * n_out + row) * n_in;
    float a = 0.f, b = 0.f;
    for (int i = 2 * threadIdx.x; i < n_in; i += 2 * blockDim.x) {
        a += bf16(w[offset + i]) * input[i];
        a += bf16(w[offset + i + 1]) * input[i + 1];
        if constexpr (Pair) {
            b += bf16(w2[offset + i]) * input[i];
            b += bf16(w2[offset + i + 1]) * input[i + 1];
        }
    }
    a = sum_warp(a); b = sum_warp(b);
    __shared__ float sa[8], sb[8];
    if ((threadIdx.x & 31) == 0) { sa[threadIdx.x / 32] = a; sb[threadIdx.x / 32] = b; }
    __syncthreads();
    if (threadIdx.x < 32) {
        a = threadIdx.x < blockDim.x / 32 ? sa[threadIdx.x] : 0.f;
        b = threadIdx.x < blockDim.x / 32 ? sb[threadIdx.x] : 0.f;
        a = sum_warp(a); b = sum_warp(b);
        if (threadIdx.x == 0) {
            const std::size_t dst = ((std::size_t) token * k + rank) * n_out + row;
            y[dst] = a;
            if constexpr (Pair) y2[dst] = b;
        }
    }
}
__global__ void router_mm(const float* x, const float* w, float* y, int n_in, int n_out) {
    const int row = blockIdx.x, token = blockIdx.y;
    float acc = 0.f;
    for (int i = 2 * threadIdx.x; i < n_in; i += 512) {
        acc += w[(std::size_t) row * n_in + i] * x[(std::size_t) token * n_in + i];
        acc += w[(std::size_t) row * n_in + i + 1] * x[(std::size_t) token * n_in + i + 1];
    }
    acc = sum_warp(acc);
    __shared__ float sums[8];
    if ((threadIdx.x & 31) == 0) sums[threadIdx.x / 32] = acc;
    __syncthreads();
    if (threadIdx.x < 32) {
        acc = sum_warp(threadIdx.x < 8 ? sums[threadIdx.x] : 0.f);
        if (!threadIdx.x) y[token * n_out + row] = acc;
    }
}
__global__ void concat_rows(const float* a, const float* b, float* out, int width, int count) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < width * count; i += blockDim.x * gridDim.x) {
        const int token = i / width, d = i % width;
        out[token * 2 * width + d] = a[i];
        out[token * 2 * width + width + d] = b[i];
    }
}
__global__ void swiglu(float* gate, const float* up, int n, float limit) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
        float g = gate[i], u = up[i];
        if (limit > 0) { g = fminf(g, limit); u = fminf(fmaxf(u, -limit), limit); }
        gate[i] = __fdividef(g, 1.f + __expf(-g)) * u;
    }
}
__global__ void combine(const float* parts, const float* weights, float* out, int dim, int k, int nt) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < dim * nt; i += blockDim.x * gridDim.x) {
        const int t = i / dim, d = i % dim;
        float sum = 0;
        for (int rank = 0; rank < k; ++rank)
            sum += parts[((std::size_t)t * k + rank) * dim + d] * weights[t * k + rank];
        out[i] = sum;
    }
}
} // namespace

struct Mtp::Impl {
    Geometry g;
    std::string prefix;
    cudaStream_t stream = nullptr;
    struct Weight : Buffer {
        void* host = nullptr;
        static constexpr std::size_t chunk_bytes = 64u << 20;
        static constexpr std::align_val_t alignment{65536};
        std::size_t registered = 0;
        bool chunked = false;
        ~Weight() {
            if (!host) return;
            if (chunked) {
                for (std::size_t off = 0; off < registered; off += chunk_bytes)
                    cudaHostUnregister(static_cast<char*>(host) + off);
                ::operator delete(host, alignment);
            } else {
                cudaFreeHost(host);
            }
            p = nullptr;
        }
        void map_host(std::size_t n) {
            const auto status = cudaHostAlloc(&host, n, cudaHostAllocMapped);
            if (status == cudaErrorInvalidValue && n > chunk_bytes) {
                // Some drivers reject a single large pinned allocation under MPOL_BIND.
                // Keep the same NUMA policy and a contiguous plane for CPU and GPU readers.
                (void) cudaGetLastError();
                host = ::operator new(n, alignment);
                chunked = true;
                while (registered < n) {
                    const std::size_t off = registered;
                    const std::size_t size = std::min(chunk_bytes, n - off);
                    auto* part = static_cast<char*>(host) + off;
                    check(cudaHostRegister(part, size, cudaHostRegisterMapped), "register MTP weight chunk");
                    registered += size;
                    void* device = nullptr;
                    check(cudaHostGetDevicePointer(&device, part, 0), "map MTP weight chunk");
                    if (off == 0) p = device;
                    else if (reinterpret_cast<uintptr_t>(p) + off != reinterpret_cast<uintptr_t>(device))
                        throw std::runtime_error("MTP weight chunks have noncontiguous GPU addresses");
                }
            } else {
                check(status, "allocate mapped MTP weights");
                check(cudaHostGetDevicePointer(&p, host, 0), "map MTP weights to device");
            }
            bytes = n;
        }
    };
    std::map<std::string, Weight> weights;
    std::map<std::string, Buffer> scratch;
    std::map<std::string, int> types;
    std::size_t weights_size = 0;
    // Execute draft experts on the CPU when their weights are placed in host RAM.
    struct DraftCpu {
        DraftMailbox* host = nullptr;          ///< the mapped mailbox; `device` holds its GPU alias
        DraftExperts device{};
        float* io = nullptr;                   ///< [expert_dim] input followed by [k, expert_dim] parts
        const uint8_t* plane[3] = {};          ///< host bases of gate / up / down
        std::size_t stride = 0;                ///< bytes of one expert in one plane
        cpu::NativeFmt fmt{};
        std::unique_ptr<CpuExperts> pool;
        std::thread thread;
        std::atomic<bool> stop{false};
        uint32_t served = 0;
        ~DraftCpu() {
            stop.store(true);
            if (host) host->error = 1;         // release a worker parked on the request flag
            if (thread.joinable()) thread.join();
            if (io) cudaFreeHost(io);
            if (host) cudaFreeHost(host);
        }
    };
    std::unique_ptr<DraftCpu> draft_cpu;
    KvCache kv;
    IndexerCache idx;
    const void* embedding = nullptr;
    const void* output = nullptr;
    int embedding_type = 0, output_type = 0;
    static constexpr int capacity = 8;
    int checkpoint_pos = -1;
    int diagnostic_calls = 0;
    bool device_draft=false,device_cache=false,recording=false;
    int* chain_output=nullptr;
    uint64_t scratch_generation=0;
    cudaGraphExec_t draft_graph[2][4]={};
    bool draft_warm[2]={};
    struct ContinueGraph {cudaGraphExec_t exec=nullptr;const RoundVerdict* verdict=nullptr;const float* hidden=nullptr;int phase=-1,rows=-1,sparse=-1;};
    ContinueGraph continuation[4];
    cudaStream_t continuation_body=nullptr;
    cudaEvent_t verdict_ready=nullptr;
    bool continuation_pending=false,have_first=false;
    int cache_warm_tokens=0;
    void invalidate_graphs(){for(auto& row:draft_graph)for(auto& e:row)if(e){cudaGraphExecDestroy(e);e=nullptr;}draft_warm[0]=draft_warm[1]=false;
        for(auto& c:continuation)if(c.exec){cudaGraphExecDestroy(c.exec);c.exec=nullptr;}}

    Impl(const std::string& path, const Geometry& geometry, int context) : g(geometry) {
        if (g.n_mtp_blocks != 1) throw std::runtime_error("hy4-flash MTP requires exactly one nextn layer");
        prefix = "blk." + std::to_string(g.trunk_blocks) + ".";
        check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "create MTP stream");
        ModelFile source(path);
        GgufModel model = GgufModel::open(path);
        std::vector<uint8_t> staging(16u << 20);
        const char* host_option=std::getenv("STRATA_HY4_FLASH_MTP_HOST_EXPERTS");
        const bool host_experts=host_option && std::atoi(host_option)!=0;
        std::size_t mapped_weight_bytes=0;
        const uint8_t* mapped_plane[3]={};
        std::size_t mapped_stride[3]={};
        for (std::size_t shard = 0; shard < model.size(); ++shard)
            for (const auto& tensor : model.shard(shard).tensors()) {
                if (tensor.name.rfind(prefix, 0) != 0) continue;
                if (tensor.type != 0 && tensor.type != 30 && tensor.type != 8)
                    throw std::runtime_error("hy4-flash MTP expects Q8_0/BF16/F32 draft weights: " + tensor.name);
                const auto* t = source.info(tensor.name);
                const std::size_t bytes = tensor_payload_bytes(*t);
                const auto name = tensor.name.substr(prefix.size());
                auto& buffer = weights[name];
                types[name] = tensor.type;
                const int role = name == "ffn_gate_exps.weight" ? 0 : name == "ffn_up_exps.weight" ? 1
                               : name == "ffn_down_exps.weight" ? 2 : -1;
                if (host_experts && role >= 0) {
                    buffer.map_host(bytes);
                    if (!source.pread_at(source.data_start()+t->offset,buffer.host,bytes))
                        throw std::runtime_error("MTP host expert read failed: "+tensor.name);
                    mapped_plane[role]=static_cast<const uint8_t*>(buffer.host);
                    mapped_stride[role]=bytes/(std::size_t)g.n_experts;
                    mapped_weight_bytes+=bytes;
                    continue;
                }
                buffer.reserve(bytes);
                for (std::size_t off = 0; off < bytes;) {
                    const auto n = std::min(staging.size(), bytes - off);
                    if (!source.read_at(source.data_start()+t->offset+off, staging.data(), n))
                        throw std::runtime_error("MTP weight read failed");
                    check(cudaMemcpy(static_cast<char*>(buffer.p)+off, staging.data(), n, cudaMemcpyHostToDevice), tensor.name.c_str());
                    off += n;
                }
                weights_size += bytes;
            }
        std::string err;
        if (!kv_cache_init(kv, 1, g.kv_lora_rank, g.rope_dim, context, err) ||
            !indexer_cache_init(idx, {1}, g.idx_key_dim, g.idx_kpool, context, err))
            throw std::runtime_error(err);
        idx.verify_capacity = capacity;
        idx.verify_vg.assign(1, nullptr);
        check(cudaMalloc(&idx.verify_vg[0], (std::size_t) 2 * idx.key_dim * (capacity + idx.kpool) * sizeof(float)), "allocate MTP verify index");
        auto* positions = buffer<int32_t>("kv_pos", context);
        std::vector<int32_t> host_positions(context);
        for (int i = 0; i < context; ++i) host_positions[i] = i;
        check(cudaMemcpy(positions, host_positions.data(), context * sizeof(int32_t), cudaMemcpyHostToDevice), "initialize MTP positions");
        if (mapped_weight_bytes) std::fprintf(stderr,"HY4_FLASH_MTP host_expert_bytes=%zu mapped=1\n",mapped_weight_bytes);
        if (mapped_weight_bytes) start_draft_cpu(mapped_plane,mapped_stride);
        std::fprintf(stderr,"HY4_FLASH_MTP resident_weight_bytes=%zu shared_embedding=1 shared_output_head=1\n",weights_size);
    }
    /// Hand the mapped expert rows to the CPU pool instead of to the GPU's matvecs. Declines (and
    /// leaves the matvecs reading host memory) when the rows are a format the CPU path does not take.
    void start_draft_cpu(const uint8_t* const plane[3], const std::size_t stride[3]) {
        const char* opt = std::getenv("STRATA_HY4_FLASH_MTP_CPU_EXPERTS");
        if (opt && std::atoi(opt) == 0) return;
        for (int r = 0; r < 3; ++r)
            if (!plane[r] || !stride[r] || stride[r] != stride[0])
                throw std::runtime_error("MTP CPU experts: the three mapped planes disagree");
        auto c = std::make_unique<DraftCpu>();
        std::string err;
        const int gu = types.at("ffn_gate_exps.weight"), dn = types.at("ffn_down_exps.weight");
        if (!CpuExperts::make_format(gu, dn, g.expert_dim, g.expert_ff, g.swiglu_clamp, c->fmt, err, false)) {
            std::fprintf(stderr, "HY4_FLASH_MTP_CPU_EXPERTS declined: %s\n", err.c_str());
            return;
        }
        const char* workers = std::getenv("STRATA_HY4_FLASH_CPU_WORKERS");
        c->pool = std::make_unique<CpuExperts>(workers ? std::atoi(workers) : 8);
        check(cudaHostAlloc((void**)&c->host, sizeof(DraftMailbox), cudaHostAllocMapped));
        new (c->host) DraftMailbox{};
        check(cudaHostGetDevicePointer((void**)&c->device.mailbox, c->host, 0));
        const std::size_t io_floats = (std::size_t)(1 + g.top_k) * g.expert_dim;
        check(cudaHostAlloc((void**)&c->io, io_floats * sizeof(float), cudaHostAllocMapped));
        std::fill_n(c->io, io_floats, 0.f);
        float* io_device = nullptr;
        check(cudaHostGetDevicePointer((void**)&io_device, c->io, 0));
        c->device.input = io_device;
        c->device.parts = io_device + g.expert_dim;
        for (int r = 0; r < 3; ++r) c->plane[r] = plane[r];
        c->stride = stride[0];
        draft_cpu = std::move(c);
        draft_cpu->thread = std::thread([this] { draft_cpu_loop(); });
        std::fprintf(stderr, "HY4_FLASH_MTP_CPU_EXPERTS enabled experts=%d k=%d stride=%zu workers=%d\n",
                     g.n_experts, g.top_k, draft_cpu->stride, workers ? std::atoi(workers) : 8);
    }
    /// One request a draft step: the whole top-k, every rank a host row. Errors are published to the
    /// waiting kernel through the mailbox rather than left for it to spin on.
    void draft_cpu_loop() {
        auto& c = *draft_cpu;
        auto* m = c.host;
        try {
            for (;;) {
                const uint32_t want = c.served + 1;
                while (*const_cast<volatile uint32_t*>(&m->request) < want) {
                    if (c.stop.load() || *const_cast<volatile uint32_t*>(&m->error)) return;
                    std::this_thread::yield();
                }
                std::atomic_thread_fence(std::memory_order_acquire);
                const uint8_t* gates[kWindowTopK]; const uint8_t* ups[kWindowTopK];
                const uint8_t* downs[kWindowTopK]; int ranks[kWindowTopK];
                for (int j = 0; j < g.top_k; ++j) {
                    const int id = m->ids[j];
                    if (id < 0 || id >= g.n_experts) throw std::runtime_error("expert id outside the block");
                    gates[j] = c.plane[0] + (std::size_t) id * c.stride;
                    ups[j] = c.plane[1] + (std::size_t) id * c.stride;
                    downs[j] = c.plane[2] + (std::size_t) id * c.stride;
                    ranks[j] = j;
                }
                std::string err;
                if (!c.pool->compute(c.fmt, gates, ups, downs, ranks, g.top_k, c.io,
                                     c.io + g.expert_dim, err))
                    throw std::runtime_error(err);
                std::atomic_thread_fence(std::memory_order_release);
                c.served = want;
                m->done = want;
            }
        } catch (const std::exception& e) {
            std::fprintf(stderr, "HY4_FLASH_MTP_CPU_ERROR %s\n", e.what());
            m->error = 1;
        }
    }
    ~Impl() { if (stream) { cudaStreamSynchronize(stream);invalidate_graphs();
        if(continuation_body)cudaStreamDestroy(continuation_body);if(verdict_ready)cudaEventDestroy(verdict_ready);
        cudaStreamDestroy(stream); } }
    template<class T = float> T* buffer(const std::string& name, std::size_t count) {
        auto& b = scratch[name];if(count*sizeof(T)>b.bytes){invalidate_graphs();++scratch_generation;}
        b.reserve(count * sizeof(T)); return b.as<T>();
    }
    template<class T = float> T* weight(const char* name) {
        const auto it = weights.find(name);
        if (it == weights.end()) throw std::runtime_error(std::string("MTP missing weight: ")+name);
        return it->second.as<T>();
    }
    void mm(const float* x, const char* name, float* y, int in, int out, int nt) {
        const int type = types.at(name);
        if (type == 30) kernels::bf16_gemv_fp32_mmvf_cols(x, weight<uint16_t>(name), y, in, out, nt, stream);
        else {
            auto* q = buffer<uint8_t>("mm_q8", kernels::native_q8_1_bytes_wide(in,nt));
            mat_apply(mat_stage(nullptr,x,q,in,nt,stream),type,weight(name),y,in,out,nt,stream);
        }
    }
    void norm(float* x, const char* name, int width, int nt) {
        kernels::rms_norm_weighted_wide(x, weight(name), nt, width, g.rms_norm_eps, stream);
    }
    const float* run(const std::vector<int32_t>& tokens, const float* previous, std::vector<float>* logits,bool cache_only=false) {
        const bool device=device_draft||device_cache;
        const int nt = (int)tokens.size(), d = g.hidden_dim, ed = g.expert_dim, ff = g.expert_ff;
        const int nh = g.layers[g.trunk_blocks].n_heads, base = kv.len, kp = g.idx_kpool;
        if (nt < 1 || nt > capacity || base + nt > kv.max_tokens || !embedding || !output)
            throw std::runtime_error("MTP invalid batch/context or unbound target weights");
        for (int32_t token : tokens)
            if (token < 0 || token >= g.vocab_size) throw std::runtime_error("MTP token outside vocabulary");
        const char* dump_dir=std::getenv("STRATA_HY4_FLASH_MTP_DUMP");
        const bool dump=!device && dump_dir && diagnostic_calls<16;
        std::string stem;
        if(dump) {
            stem=std::string(dump_dir)+"/"+std::to_string(diagnostic_calls++);
            std::ofstream meta(stem+".meta");meta<<base<<' '<<nt<<' '<<d<<' '<<g.vocab_size<<'\n';
            for(auto id:tokens)meta<<id<<' ';
            std::vector<float> host((std::size_t)d*nt);
            check(cudaMemcpy(host.data(),previous,host.size()*4,cudaMemcpyDeviceToHost));
            std::ofstream input(stem+".input",std::ios::binary);input.write((char*)host.data(),host.size()*4);
        }
        if (!device && !indexer_cache_rewind(idx, base)) throw std::runtime_error("MTP indexer cannot rewind");
        idx.verify_base = base; idx.verify_tokens = nt;
        float* e = buffer("embedding", d*nt), *h = buffer("previous", d*nt);
        void* es = buffer<uint8_t>("embed_sc", embed_tokens_scratch_bytes(d,nt));
        if(device)draft_embedding_k<<<(d*nt+255)/256,256,0,stream>>>(static_cast<const uint8_t*>(embedding),scratch.at("chain_token").as<int>(),e,d,g.vocab_size,nt);
        else embed_tokens(embedding, embedding_type, tokens.data(),d,1,nt,e,es,stream);
        check(cudaMemcpyAsync(h,previous,(std::size_t)d*nt*4,cudaMemcpyDeviceToDevice,stream));
        norm(e,"nextn.enorm.weight",d,nt); norm(h,"nextn.hnorm.weight",d,nt);
        float* joined = buffer("joined",2*d*nt), *state=buffer("state",d*nt), *cur=buffer("cur",d*nt);
        concat_rows<<<(d*nt+255)/256,256,0,stream>>>(e,h,joined,d,nt);
        mm(joined,"nextn.eh_proj.weight",state,2*d,d,nt);
        check(cudaMemcpyAsync(cur,state,(std::size_t)d*nt*4,cudaMemcpyDeviceToDevice,stream));
        norm(cur,"attn_norm.weight",d,nt);
        auto* pos=buffer<int32_t>("pos",nt);
        std::vector<int32_t> positions(nt);
        for(int i=0;i<nt;++i) positions[i]=base+i;
        if(!device)check(cudaMemcpyAsync(pos,positions.data(),nt*4,cudaMemcpyHostToDevice,stream));
        void* q8=buffer<uint8_t>("cur_q8",kernels::native_q8_1_bytes_wide(d,nt));
        MatIn current=mat_stage(nullptr,cur,q8,d,nt,stream);
        void* psc=buffer<uint8_t>("project",mla_project_scratch_bytes(d,g.q_lora_rank,g.kv_lora_rank,g.qk_head_dim,g.rope_dim,nh,nt));
        MlaProjected pj{};
        mla_project(cur,weight("attn_q_a.weight"),types.at("attn_q_a.weight"),weight("attn_q_a_norm.weight"),weight("attn_q_b.weight"),types.at("attn_q_b.weight"),
            weight("attn_kv_a_mqa.weight"),types.at("attn_kv_a_mqa.weight"),weight("attn_kv_a_norm.weight"),d,g.q_lora_rank,g.kv_lora_rank,
            g.qk_head_dim,g.rope_dim,nh,nt,pos,g.rms_norm_eps,g.rope_theta_sparse,psc,pj,current,stream);
        const int nope=g.qk_head_dim-g.rope_dim;
        float* absorbed=buffer("absorbed",g.kv_lora_rank*nh*nt);
        if(!cache_only)mla_absorb(pj.q,weight("attn_k_b.weight"),types.at("attn_k_b.weight"),g.qk_head_dim,nope,g.kv_lora_rank,nh,nt,absorbed,
            buffer<uint8_t>("absorb",mla_absorb_scratch_bytes(nope,nt,nh)),stream);
        std::string err;
        if(device) {
            if(!kv_cache_append_dev(kv,0,scratch.at("chain_len").as<int>(),pj.kv_cmpr,pj.k_pe,nt,err,stream))throw std::runtime_error(err);
        } else if(!kv_cache_append(kv,0,base,pj.kv_cmpr,pj.k_pe,nt,err,stream))throw std::runtime_error(err);
        IndexerWeights iw{};
        iw.attn_k=weight("indexer.attn_k.weight");iw.attn_k_type=types.at("indexer.attn_k.weight");
        iw.attn_q_b=weight("indexer.attn_q_b.weight");iw.attn_q_b_type=types.at("indexer.attn_q_b.weight");
        iw.proj=weight("indexer.proj.weight");iw.proj_type=types.at("indexer.proj.weight");
        iw.value=weight("indexer_compressor_value.weight");iw.value_type=types.at("indexer_compressor_value.weight");
        iw.gate=weight("indexer_compressor_gate.weight");iw.gate_type=types.at("indexer_compressor_gate.weight");
        iw.k_norm=weight("indexer.k_norm.weight");iw.compressor_norm=weight("indexer_compressor_norm.weight");
        iw.ape=weight("indexer_compressor_ape.weight");
        auto plan=indexer_plan(g.idx_top_k,kp,g.idx_select_tail,base,nt);
        if(cache_only)plan.engaged=false; // pool/append still execute; no query selection is consumed.
        auto* sel=buffer<int32_t>("selection",std::max(1,plan.n_sel)*nt);
        indexer_select(idx,0,iw,plan,current.q8,cur,pj.q_resid_q8,pj.q_resid,nullptr,d,g.q_lora_rank,g.idx_heads,
            g.rope_dim,nt,base,pos,g.rope_theta_sparse,g.rms_norm_eps,sel,
            buffer<uint8_t>("indexer",indexer_scratch_bytes(g.idx_heads,g.idx_key_dim,kp,nt,idx.max_pool,plan.n_sel)),stream);
        if(cache_only) {
            indexer_advance(idx,nt,stream);
            if(!recording)checked_cache_commit(base,nt,kp);
            return nullptr;
        }
        float* kqv=buffer("kqv",g.v_head_dim*nh*nt);
        const int max_sel=g.idx_top_k+kp;
        mla_attention(absorbed,pj.q_pe,kv.kv_cmpr[0],kv.k_pe[0],kv_format(kv),weight("attn_sinks.weight"),
            weight("attn_v_b.weight"),types.at("attn_v_b.weight"),g.kv_lora_rank,g.rope_dim,g.v_head_dim,nh,nt,base+nt,pos,
            buffer<int32_t>("kv_pos",kv.max_tokens),0,1.f/std::sqrt(float(g.qk_head_dim)),kqv,
            buffer<uint8_t>("attention",mla_attention_scratch_bytes(g.kv_lora_rank,g.v_head_dim,nh,nt,kv.max_tokens,max_sel)),
            stream,kv.max_tokens,device_draft?scratch.at("chain_len").as<int>()+1:nullptr,plan.engaged?sel:nullptr,plan.engaged?plan.n_sel:0,max_sel);
        float* out=buffer("attn_out",d*nt);
        mla_output(kqv,cur,weight("attn_gate.weight"),types.at("attn_gate.weight"),weight("attn_output.weight"),types.at("attn_output.weight"),
            weight("post_attention_norm.weight"),d,g.v_head_dim*nh,nt,g.rms_norm_eps,out,
            buffer<uint8_t>("attn_output",mla_output_scratch_bytes(d,g.v_head_dim*nh,nt)),stream,current);
        kernels::add_inplace(state,out,d*nt,stream);
        check(cudaMemcpyAsync(cur,state,(std::size_t)d*nt*4,cudaMemcpyDeviceToDevice,stream));
        norm(cur,"ffn_norm.weight",d,nt);
        if (types.at("ffn_gate_exps.weight") == 8) {
            LatentMoeWeights w{};
            w.gate_inp=weight("ffn_gate_inp.weight");w.exp_probs_b=weight("exp_probs_b.bias");
            w.routed_down=weight("ffn_routed_down.weight");w.routed_down_type=types.at("ffn_routed_down.weight");
            w.routed_up=weight("ffn_routed_up.weight");w.routed_up_type=types.at("ffn_routed_up.weight");
            w.gate_exps=weight("ffn_gate_exps.weight");w.gate_exps_type=8;
            w.up_exps=weight("ffn_up_exps.weight");w.up_exps_type=8;
            w.down_exps=weight("ffn_down_exps.weight");w.down_exps_type=8;
            w.gate_shexp=weight("ffn_gate_shexp.weight");w.gate_shexp_type=types.at("ffn_gate_shexp.weight");
            w.up_shexp=weight("ffn_up_shexp.weight");w.up_shexp_type=types.at("ffn_up_shexp.weight");
            w.down_shexp=weight("ffn_down_shexp.weight");w.down_shexp_type=types.at("ffn_down_shexp.weight");
            w.post_norm=weight("post_ffw_norm.weight");
            if (draft_cpu) w.draft_cpu=&draft_cpu->device;
            latent_moe(cur,w,d,ed,ff,g.n_experts,g.top_k,nt,g.routed_scale,g.swiglu_clamp,g.rms_norm_eps,out,
                buffer<uint8_t>("moe",latent_moe_scratch_bytes(d,ed,ff,g.n_experts,g.top_k,nt,8,8,8)),stream);
        } else {
        float* scores=buffer("scores",g.n_experts*nt), *rw=buffer("router_weights",g.top_k*nt);
        auto* ids=buffer<int32_t>("router_ids",g.top_k*nt);
        router_mm<<<dim3(g.n_experts,nt),256,0,stream>>>(cur,weight("ffn_gate_inp.weight"),scores,d,g.n_experts);
        kernels::native_router_sigmoid(scores,weight("exp_probs_b.bias"),ids,rw,g.n_experts,g.top_k,nt,g.routed_scale,stream);
        float* latent=buffer("latent",ed*nt), *gate=buffer("expert_gate",ff*g.top_k*nt), *up=buffer("expert_up",ff*g.top_k*nt);
        mm(cur,"ffn_routed_down.weight",latent,d,ed,nt);
        // Same adaptive block rule as llama.cpp MMVF and the shared BF16 kernel.
        auto block=[](int width){int best=32,steps=(width/2+31)/32;for(int n=64;n<=256;n+=32){int s=(width/2+n-1)/n;if(s<steps){best=n;steps=s;}}return best;};
        expert_mm<true,false><<<dim3(ff,g.top_k,nt),block(ed),0,stream>>>(latent,weight<uint16_t>("ffn_gate_exps.weight"),
            weight<uint16_t>("ffn_up_exps.weight"),ids,gate,up,ed,ff,g.top_k);
        swiglu<<<(ff*g.top_k*nt+255)/256,256,0,stream>>>(gate,up,ff*g.top_k*nt,g.swiglu_clamp);
        float* parts=buffer("parts",ed*g.top_k*nt);
        expert_mm<false,true><<<dim3(ed,g.top_k,nt),block(ff),0,stream>>>(gate,weight<uint16_t>("ffn_down_exps.weight"),
            nullptr,ids,parts,nullptr,ff,ed,g.top_k);
        combine<<<(ed*nt+255)/256,256,0,stream>>>(parts,rw,latent,ed,g.top_k,nt);
        mm(latent,"ffn_routed_up.weight",out,ed,d,nt);
        float* shg=buffer("shared_gate",ff*nt), *shu=buffer("shared_up",ff*nt), *sho=buffer("shared_out",d*nt);
        mm(cur,"ffn_gate_shexp.weight",shg,d,ff,nt);mm(cur,"ffn_up_shexp.weight",shu,d,ff,nt);
        swiglu<<<(ff*nt+255)/256,256,0,stream>>>(shg,shu,ff*nt,0);
        mm(shg,"ffn_down_shexp.weight",sho,ff,d,nt);
        kernels::add_inplace(out,sho,d*nt,stream);
        norm(out,"post_ffw_norm.weight",d,nt);
        }
        kernels::add_inplace(state,out,d*nt,stream);
        norm(state,"nextn.shared_head_norm.weight",d,nt);
        indexer_advance(idx,nt,stream);
        if(!device_draft){idx.n_pool=(base+nt)/kp;idx.carry=(base+nt)%kp;idx.valid_pos=kv.len=base+nt;}
        if(logits || device_draft) {
            float* logit=buffer("logits",g.vocab_size);
            const float* last=state+(nt-1)*d;
            if(output_type==30)kernels::bf16_gemv_fp32_mmvf(last,static_cast<const uint16_t*>(output),logit,d,g.vocab_size,stream);
            else {void* q=buffer<uint8_t>("head_q8",kernels::native_q8_1_bytes(d,1));
                kernels::native_quantize_q8_1(last,q,d,1,stream);kernels::native_mmvq(output_type,output,q,logit,d,g.vocab_size,1,stream);}
            if(device_draft){device_pick(logit,g.vocab_size,buffer<PickPart>("pick_parts",64),scratch.at("chain_token").as<int>(),chain_output,stream);
                draft_advance_k<<<1,1,0,stream>>>(pos,scratch.at("chain_len").as<int>());}
            else {logits->resize(g.vocab_size);check(cudaMemcpyAsync(logits->data(),logit,(std::size_t)g.vocab_size*4,cudaMemcpyDeviceToHost,stream));}
        }
        check(cudaGetLastError());if(!recording)check(cudaStreamSynchronize(stream));
        if(dump) {
            std::vector<float> host((std::size_t)d*nt);
            check(cudaMemcpy(host.data(),state,host.size()*4,cudaMemcpyDeviceToHost));
            std::ofstream output_file(stem+".hidden",std::ios::binary);output_file.write((char*)host.data(),host.size()*4);
        }
        return state;
    }
    void checked_cache_commit(int base,int nt,int kp) {
        check(cudaGetLastError());check(cudaStreamSynchronize(stream));
        idx.n_pool=(base+nt)/kp;idx.carry=(base+nt)%kp;idx.valid_pos=kv.len=base+nt;
    }
    std::vector<int32_t> draft(int32_t token,const float* previous,int count) {
        if(count<0||count>7||kv.len+count-(have_first?1:0)>kv.max_tokens)throw std::runtime_error("MTP draft extent");
        if(token<0||token>=g.vocab_size||g.idx_kpool!=4||!previous)throw std::runtime_error("MTP draft inputs/kpool");
        if(!count)return {};
        if(embedding_type!=8)throw std::runtime_error("MTP device draft currently requires Q8_0 embedding");
        int* tok=buffer<int>("chain_token",8);int* len=buffer<int>("chain_len",2);
        int* pos=buffer<int>("pos",8);int* output_ids=buffer<int>("chain_ids",8);
        chain_output=buffer<int>("chain_pick",1);float* hidden=buffer("chain_hidden",g.hidden_dim);
        buffer<int32_t>("selection",(g.idx_top_k+g.idx_kpool)*capacity);
        const bool first_ready=have_first;have_first=false;
        if(!first_ready && !indexer_cache_rewind(idx,kv.len))throw std::runtime_error("MTP chain rewind");
        const int meta[2]={kv.len,kv.len+1},base=kv.len;
        if(!first_ready) {
            check(cudaMemcpyAsync(tok,&token,4,cudaMemcpyHostToDevice,stream));
            check(cudaMemcpyAsync(len,meta,8,cudaMemcpyHostToDevice,stream));
            check(cudaMemcpyAsync(pos,&base,4,cudaMemcpyHostToDevice,stream));
            check(cudaMemcpyAsync(hidden,previous,g.hidden_dim*4,cudaMemcpyDeviceToDevice,stream));
        }
        struct Mode {Impl& i;~Mode(){i.device_draft=false;i.recording=false;}} mode{*this};
        device_draft=true;
        for(int j=first_ready?1:0;j<count;++j) {
            const int at=kv.len,kp=g.idx_kpool;
            const int engaged=indexer_plan(g.idx_top_k,kp,g.idx_select_tail,at,1).engaged?1:0;
            auto& graph=draft_graph[engaged][at%kp];
            if(!draft_warm[engaged]) {
                run({0},hidden,nullptr);draft_warm[engaged]=true;
            } else {
                if(!graph) {
                    const auto start = std::chrono::steady_clock::now();
                    recording=true;check(cudaStreamBeginCapture(stream,cudaStreamCaptureModeThreadLocal));
                    try {run({0},hidden,nullptr);}catch(...) {
                        cudaGraph_t abandoned=nullptr;cudaStreamEndCapture(stream,&abandoned);
                        if(abandoned)cudaGraphDestroy(abandoned);throw;
                    }
                    cudaGraph_t captured=nullptr;check(cudaStreamEndCapture(stream,&captured));recording=false;
                    const auto captured_at = std::chrono::steady_clock::now();
                    const auto rc=cudaGraphInstantiate(&graph,captured,nullptr,nullptr,0);
                    const auto instantiated_at = std::chrono::steady_clock::now();
                    cudaGraphDestroy(captured);check(rc);
                    std::fprintf(stderr,"HY4_FLASH_MTP_GRAPH captured carry=%d sparse=%d capture_ms=%.3f instantiate_ms=%.3f\n",at%kp,engaged,
                        std::chrono::duration<double,std::milli>(captured_at-start).count(),
                        std::chrono::duration<double,std::milli>(instantiated_at-captured_at).count());
                }
                check(cudaGraphLaunch(graph,stream));
            }
            check(cudaMemcpyAsync(output_ids+j,chain_output,4,cudaMemcpyDeviceToDevice,stream));
            check(cudaMemcpyAsync(hidden,scratch.at("state").p,g.hidden_dim*4,cudaMemcpyDeviceToDevice,stream));
            idx.n_pool=(at+1)/kp;idx.carry=(at+1)%kp;idx.valid_pos=kv.len=at+1;
        }
        std::vector<int32_t> ids(count);check(cudaMemcpyAsync(ids.data(),output_ids,count*4,cudaMemcpyDeviceToHost,stream));
        check(cudaStreamSynchronize(stream));
        for(int id:ids)if(id<0)throw std::runtime_error("MTP draft logits nonfinite");
        return ids;
    }
    bool continue_device(const RoundVerdict* verdict,void* verdict_stream,const std::vector<int32_t>& window,
                         const float* target,int base,int remaining) {
#if CUDART_VERSION >= 13000 && !defined(STRATA_USE_HIP)
        const int T=window.size(),kp=g.idx_kpool;
        if(continuation_pending||have_first)throw std::runtime_error("MTP continuation already pending");
        if(T<1||T>8||kp!=4||embedding_type!=8||cache_warm_tokens<T||checkpoint_pos!=base)return false;
        const int sparse=indexer_plan(g.idx_top_k,kp,g.idx_select_tail,base,1).engaged;
        if(sparse!=indexer_plan(g.idx_top_k,kp,g.idx_select_tail,base+T,1).engaged)return false;
        const auto generation=scratch_generation;
        auto* input=buffer<int>("continue_input",8);auto* meta=buffer<int>("continue_meta",2);
        buffer<int>("chain_token",8);buffer<int>("chain_len",2);buffer<int>("pos",8);
        buffer<int>("chain_ids",8);chain_output=buffer<int>("chain_pick",1);
        buffer("chain_hidden",g.hidden_dim*8);
        if(generation!=scratch_generation||!draft_warm[sparse])return false;
        if(!continuation_body)check(cudaStreamCreateWithFlags(&continuation_body,cudaStreamNonBlocking));
        if(!verdict_ready)check(cudaEventCreateWithFlags(&verdict_ready,cudaEventDisableTiming));
        auto& cg=continuation[base%4];
        if(cg.exec && (cg.rows!=T||cg.sparse!=sparse||cg.verdict!=verdict||cg.hidden!=target)) {
            cudaGraphExecDestroy(cg.exec);cg.exec=nullptr;
        }
        if(!cg.exec) {
            // Graph topology depends on carry and shape, never the live base.
            const int old_len=kv.len,old_pool=idx.n_pool,old_carry=idx.carry,old_valid=idx.valid_pos;
            const int old_verify=idx.verify_base,old_tokens=idx.verify_tokens;
            cudaStream_t outer=stream;cudaGraph_t graph=nullptr;
            check(cudaStreamBeginCapture(outer,cudaStreamCaptureModeThreadLocal));
            try {
                cudaStreamCaptureStatus status;check(cudaStreamGetCaptureInfo(outer,&status,nullptr,&graph));
                cudaGraphConditionalHandle handle;check(cudaGraphConditionalHandleCreate(&handle,graph,0,0));
                continue_choice_k<<<1,1,0,outer>>>(verdict,meta,handle);
                const cudaGraphNode_t* deps=nullptr;const cudaGraphEdgeData* edges=nullptr;size_t count=0;
                check(cudaStreamGetCaptureInfo(outer,&status,nullptr,&graph,&deps,&edges,&count));
                cudaGraphNodeParams params{};params.type=cudaGraphNodeTypeConditional;
                params.conditional.handle=handle;params.conditional.type=cudaGraphCondTypeSwitch;params.conditional.size=T;
                cudaGraphNode_t node;check(cudaGraphAddNode(&node,graph,deps,edges,count,&params));
                check(cudaStreamUpdateCaptureDependencies(outer,&node,nullptr,1,cudaStreamSetCaptureDependencies));
                for(int n=1;n<=T;++n) {
                    stream=continuation_body;check(cudaStreamBeginCaptureToGraph(stream,params.conditional.phGraph_out[n-1],nullptr,nullptr,0,cudaStreamCaptureModeThreadLocal));
                    recording=true;device_cache=true;device_draft=false;kv.len=base;
                    check(cudaMemcpyAsync(idx.vg[0],scratch.at("checkpoint_vg").p,2*idx.key_dim*kp*4,cudaMemcpyDeviceToDevice,stream));
                    auto setup=[&](int nt,bool drafting) {continue_setup_k<<<(nt*g.hidden_dim+255)/256,256,0,stream>>>(
                        verdict,input,scratch.at("chain_token").as<int>(),scratch.at("pos").as<int>(),scratch.at("chain_len").as<int>(),
                        static_cast<int*>(idx.counters),scratch.at("chain_hidden").as<float>(),target,g.hidden_dim,meta,nt,drafting);};
                    setup(n,false);run(std::vector<int32_t>(n,0),scratch.at("chain_hidden").as<float>(),nullptr,true);
                    // The next checkpoint precedes its first draft, not after it.
                    check(cudaMemcpyAsync(scratch.at("checkpoint_vg").p,idx.vg[0],2*idx.key_dim*kp*4,cudaMemcpyDeviceToDevice,stream));
                    device_cache=false;device_draft=true;kv.len=base+n;
                    setup(1,true);run({0},scratch.at("chain_hidden").as<float>(),nullptr);
                    check(cudaMemcpyAsync(scratch.at("chain_ids").p,chain_output,4,cudaMemcpyDeviceToDevice,stream));
                    check(cudaMemcpyAsync(scratch.at("chain_hidden").p,scratch.at("state").p,g.hidden_dim*4,cudaMemcpyDeviceToDevice,stream));
                    cudaGraph_t body=nullptr;check(cudaStreamEndCapture(stream,&body));
                }
                stream=outer;recording=device_draft=device_cache=false;
                check(cudaStreamEndCapture(outer,&graph));const auto rc=cudaGraphInstantiate(&cg.exec,graph,nullptr,nullptr,0);cudaGraphDestroy(graph);graph=nullptr;check(rc);
            } catch(...) {
                cudaStreamCaptureStatus status;
                if(cudaStreamIsCapturing(stream,&status)==cudaSuccess&&status==cudaStreamCaptureStatusActive){cudaGraph_t abandoned=nullptr;cudaStreamEndCapture(stream,&abandoned);}
                stream=outer;
                if(cudaStreamIsCapturing(outer,&status)==cudaSuccess&&status==cudaStreamCaptureStatusActive){cudaGraph_t abandoned=nullptr;cudaStreamEndCapture(outer,&abandoned);if(abandoned)cudaGraphDestroy(abandoned);}
                recording=device_draft=device_cache=false;kv.len=old_len;idx.n_pool=old_pool;idx.carry=old_carry;idx.valid_pos=old_valid;
                idx.verify_base=old_verify;idx.verify_tokens=old_tokens;throw;
            }
            kv.len=old_len;idx.n_pool=old_pool;idx.carry=old_carry;idx.valid_pos=old_valid;idx.verify_base=old_verify;idx.verify_tokens=old_tokens;
            cg.rows=T;cg.sparse=sparse;cg.verdict=verdict;cg.hidden=target;
            std::fprintf(stderr,"HY4_FLASH_CONTINUE_GRAPH captured T=%d carry=%d sparse=%d\n",T,base%4,sparse);
        }
        const int host_meta[2]={base,remaining};
        check(cudaMemcpyAsync(input,window.data(),T*4,cudaMemcpyHostToDevice,stream));
        check(cudaMemcpyAsync(meta,host_meta,8,cudaMemcpyHostToDevice,stream));
        check(cudaEventRecord(verdict_ready,static_cast<cudaStream_t>(verdict_stream)));
        check(cudaStreamWaitEvent(stream,verdict_ready,0));check(cudaGraphLaunch(cg.exec,stream));
        continuation_pending=true;return true;
#else
        return false;
#endif
    }
};

Mtp::Mtp(const std::string& path,const Geometry& g,int context):impl_(new Impl(path,g,context)){}
Mtp::~Mtp()=default;
void Mtp::bind(DenseResidency& target) {
    impl_->embedding=forward_weight(target,"token_embd.weight",impl_->embedding_type);
    impl_->output=forward_weight(target,"output.weight",impl_->output_type);
}
void Mtp::rewind(int position) {
    if(impl_->have_first){check(cudaStreamSynchronize(impl_->stream));impl_->have_first=false;}
    if(position<0 || position>impl_->kv.len) throw std::runtime_error("MTP invalid rollback position");
    if (position == impl_->checkpoint_pos) {
        const std::size_t bytes = (std::size_t) 2 * impl_->idx.key_dim * impl_->idx.kpool * sizeof(float);
        check(cudaMemcpy(impl_->idx.vg[0], impl_->scratch.at("checkpoint_vg").p, bytes, cudaMemcpyDeviceToDevice));
        impl_->idx.valid_pos = position;
        impl_->idx.n_pool = position / impl_->idx.kpool;
        impl_->idx.carry = position % impl_->idx.kpool;
        const int counters[2] = {impl_->idx.carry, impl_->idx.n_pool};
        check(cudaMemcpy(impl_->idx.counters, counters, sizeof counters, cudaMemcpyHostToDevice));
    }
    if(position<0 || position>impl_->kv.len || !indexer_cache_rewind(impl_->idx,position))
        throw std::runtime_error("MTP invalid rollback position");
    impl_->kv.len=position;
}
void Mtp::checkpoint() {
    auto& state = *impl_;
    if(state.have_first)return; // continuation saved the checkpoint before precomputing draft 0
    const std::size_t n = (std::size_t) 2 * state.idx.key_dim * state.idx.kpool;
    float* saved = state.buffer("checkpoint_vg", n);
    check(cudaMemcpy(saved, state.idx.vg[0], n * sizeof(float), cudaMemcpyDeviceToDevice));
    state.checkpoint_pos = state.kv.len;
}
int Mtp::position()const{return impl_->kv.len;}
KvCache& Mtp::kv_cache() { return impl_->kv; }
IndexerCache& Mtp::indexer_cache() { return impl_->idx; }
void Mtp::forget_speculation() {
    impl_->checkpoint_pos = -1;
    impl_->have_first = false;
    impl_->continuation_pending = false;
}
std::size_t Mtp::weight_bytes()const{return impl_->weights_size;}
const float* Mtp::forward(const std::vector<int32_t>& ids,const float* hidden,std::vector<float>* logits){return impl_->run(ids,hidden,logits);}
void Mtp::catchup(const std::vector<int32_t>& ids,const float* hidden) {
    static const bool on=[] {const char* e=std::getenv("STRATA_HY4_FLASH_MTP_CACHE_ONLY");return e&&std::atoi(e)!=0;}();
    impl_->run(ids,hidden,nullptr,on);
    if(on)impl_->cache_warm_tokens=std::max(impl_->cache_warm_tokens,(int)ids.size());
}
std::vector<int32_t> Mtp::draft(int32_t token,const float* hidden,int count){return impl_->draft(token,hidden,count);}
bool Mtp::continue_device(const RoundVerdict* v,void* s,const std::vector<int32_t>& input,const float* h,int base,int remaining){return impl_->continue_device(v,s,input,h,base,remaining);}
void Mtp::finish_continue(const RoundVerdict& v,bool ran) {
    auto& i=*impl_;if(!i.continuation_pending)throw std::runtime_error("MTP no pending continuation");
    i.continuation_pending=false;
    if(!ran){check(cudaStreamSynchronize(i.stream));return;}
    i.have_first=true;i.checkpoint_pos=v.next_position;
    i.kv.len=i.idx.valid_pos=v.next_position+1;i.idx.carry=i.kv.len%i.idx.kpool;i.idx.n_pool=i.kv.len/i.idx.kpool;
}
bool Mtp::loop_ready(int tokens,bool sparse) {
    auto& i=*impl_;
    if(i.have_first||i.continuation_pending||tokens<2||tokens>8||i.cache_warm_tokens<tokens||!i.draft_warm[sparse]||i.embedding_type!=8)return false;
    const auto gen=i.scratch_generation;
    i.buffer<int>("chain_token",8);i.buffer<int>("chain_len",2);i.buffer<int>("pos",8);
    i.buffer<int>("chain_ids",8);i.buffer<int>("chain_pick",1);i.buffer("chain_hidden",i.g.hidden_dim*8);
    if(!i.continuation_body)check(cudaStreamCreateWithFlags(&i.continuation_body,cudaStreamNonBlocking));
    return gen==i.scratch_generation;
}
void Mtp::record_loop_draft(DeviceLoopState* state,int32_t* window,const float* target,int T,bool sparse,void* stream) {
    auto& i=*impl_;const auto saved=i.stream;const int old=i.kv.len;
    i.stream=static_cast<cudaStream_t>(stream);i.recording=true;i.device_draft=true;
    auto* hidden=i.scratch.at("chain_hidden").as<float>();auto* token=i.scratch.at("chain_token").as<int>();
    loop_mtp_setup_k<<<(i.g.hidden_dim+255)/256,256,0,i.stream>>>(state,window,token,i.scratch.at("pos").as<int>(),
        i.scratch.at("chain_len").as<int>(),static_cast<int*>(i.idx.counters),hidden,target,i.g.hidden_dim,1,true);
    check(cudaMemcpyAsync(window,token,4,cudaMemcpyDeviceToDevice,i.stream));
    check(cudaMemcpyAsync(i.scratch.at("checkpoint_vg").p,i.idx.vg[0],2*i.idx.key_dim*4*4,cudaMemcpyDeviceToDevice,i.stream));
    try {for(int j=0;j<T-1;++j){i.kv.len=(sparse?4096:0)+j;i.run({0},hidden,nullptr);
        check(cudaMemcpyAsync(window+j+1,i.chain_output,4,cudaMemcpyDeviceToDevice,i.stream));
        check(cudaMemcpyAsync(hidden,i.scratch.at("state").p,i.g.hidden_dim*4,cudaMemcpyDeviceToDevice,i.stream));}
    }catch(...){i.kv.len=old;i.stream=saved;i.recording=i.device_draft=false;throw;}
    i.kv.len=old;i.stream=saved;i.recording=i.device_draft=false;
}
void Mtp::record_loop_commit(const DeviceLoopState* state,const RoundVerdict* verdict,const int32_t* window,
    const float* target,int T,void* stream) {
#if CUDART_VERSION >= 13000 && !defined(STRATA_USE_HIP)
    auto& i=*impl_;const auto saved=i.stream;const int old=i.kv.len;auto outer=static_cast<cudaStream_t>(stream);
    cudaStreamCaptureStatus status;cudaGraph_t graph;check(cudaStreamGetCaptureInfo(outer,&status,nullptr,&graph));
    cudaGraphConditionalHandle handle;check(cudaGraphConditionalHandleCreate(&handle,graph,0,0));
    loop_keep_choice_k<<<1,1,0,outer>>>(verdict,handle);
    const cudaGraphNode_t* deps;const cudaGraphEdgeData* edges;size_t count;
    check(cudaStreamGetCaptureInfo(outer,&status,nullptr,&graph,&deps,&edges,&count));
    cudaGraphNodeParams params{};params.type=cudaGraphNodeTypeConditional;params.conditional.handle=handle;
    params.conditional.type=cudaGraphCondTypeSwitch;params.conditional.size=T;
    cudaGraphNode_t node;check(cudaGraphAddNode(&node,graph,deps,edges,count,&params));
    check(cudaStreamUpdateCaptureDependencies(outer,&node,nullptr,1,cudaStreamSetCaptureDependencies));
    if(!i.continuation_body)throw std::runtime_error("loop MTP body stream not prepared");
    try {for(int n=1;n<=T;++n){i.stream=i.continuation_body;i.recording=true;i.device_cache=true;i.kv.len=0;
        check(cudaStreamBeginCaptureToGraph(i.stream,params.conditional.phGraph_out[n-1],nullptr,nullptr,0,cudaStreamCaptureModeThreadLocal));
        check(cudaMemcpyAsync(i.idx.vg[0],i.scratch.at("checkpoint_vg").p,2*i.idx.key_dim*4*4,cudaMemcpyDeviceToDevice,i.stream));
        loop_mtp_setup_k<<<(n*i.g.hidden_dim+255)/256,256,0,i.stream>>>(state,window,i.scratch.at("chain_token").as<int>(),i.scratch.at("pos").as<int>(),
            i.scratch.at("chain_len").as<int>(),static_cast<int*>(i.idx.counters),i.scratch.at("chain_hidden").as<float>(),target,i.g.hidden_dim,n,false);
        i.run(std::vector<int32_t>(n,0),i.scratch.at("chain_hidden").as<float>(),nullptr,true);
        cudaGraph_t body;check(cudaStreamEndCapture(i.stream,&body));}
    }catch(...){i.kv.len=old;i.stream=saved;i.recording=i.device_cache=false;throw;}
    i.kv.len=old;i.stream=saved;i.recording=i.device_cache=false;
#else
    throw std::runtime_error("device loop requires CUDA13");
#endif
}
void Mtp::loop_mirror(int base) {auto& i=*impl_;i.kv.len=i.idx.valid_pos=base;i.idx.carry=base%4;i.idx.n_pool=base/4;i.checkpoint_pos=-1;}
} // namespace strata::hy4_flash
