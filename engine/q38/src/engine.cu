// Strata-owned dense Qwen35/Q38 inference. No external inference runtime.
#include "strata/qwen38/admission.hpp"
#include "strata/qwen38/universal_layer.hpp"
#include "strata/kernels/fused_gdn.hpp"
#include "strata/qwen35/sampling.hpp"
#include "ops.cuh"
#include "prefill_gemm.cuh"
#include "resources.hpp"
#include "protocol.hpp"
#include "input_reader.hpp"
#include "protocol_output.hpp"
#include "kv.hpp"
#include "upload_reader.hpp"
#include "session_state.hpp"
#include "weight_arena.hpp"
#include "layer_fence.hpp"
#include "cvec_kernels.cuh"
#include "startup.cuh"
#include "sampler.hpp"
#include <memory>
#include <cstdio>
#include <fstream>
#include <sstream>
#include <thread>
#include <random>
#include <chrono>
#include <iostream>
#include <numeric>

namespace Q=strata::qwen38;
namespace K=strata::kernels;
using Clock=std::chrono::steady_clock;
using strata::qwen35::Sampling;
using Q::Cancelled;
static int integer(const std::string& s,int lo,int hi);
static void gpu_resources(size_t need=0,const std::string& stage="execution",const std::string& tensor="none"){
    size_t free=0,total=0;Q::ck(cudaMemGetInfo(&free,&total));
    if(free<Q::GPU_RESERVE||need>free-Q::GPU_RESERVE)
        throw Q::ResourceError("CUDA reserve: stage="+stage+" tensor="+tensor+" free="+std::to_string(free)+" needed="+std::to_string(need)+" reserve="+std::to_string(Q::GPU_RESERVE));
}
struct Stream {cudaStream_t s=nullptr;Stream(){Q::ck(cudaStreamCreate(&s));}~Stream(){if(s)cudaStreamDestroy(s);}};
struct Weight {
    Q::WeightSpan b;int type=0,n=0,rows=0;
    Q::StrataQuantTensor quant()const{return Q::make_quant_tensor(b.data,type,n,rows);}
};
static bool is_q8_matvec_type(int type){return type==8||type==12||type==13||type==14||type==21||type==22;}
struct Engine {
    int prefill_chunk=512;
    size_t prefill_tile_bytes=20ull*1024*1024;
    std::unique_ptr<Q::Orthography> orthography;
    bool japanese=false;
    strata::GgufFile file;Q::Admission admission;const Q::Geometry& G;int ctx;Stream stream;
    // Prefill scratch sub-buffer offsets (floats per token), derived from geometry.
    size_t o_qkv=0,o_z=0,o_gdn=0,o_al=0,o_be=0,o_qg=0,o_bk=0,o_bv=0,o_bq=0,o_bgate=0,o_batt=0,o_bfg=0,o_bfu=0,o_sub=0;
    std::unique_ptr<Q::WeightArena> weight_arena;
    std::unique_ptr<Q::WeightArena> mtp_arena;
    std::map<std::string,Weight> weights;std::vector<Q::RecurrentState> states;
    std::vector<Q::RecurrentState> snap_states;
    std::vector<Q::UniversalTransformerLayer> layers;
    Q::PrefillGemmWorkspace prefill_ws;
    std::unique_ptr<Q::KvCache> kv;
    std::unique_ptr<Q::KvCache> mtp_kv;
    Q::KvSelection kv_selection;
    std::map<int,Q::Buffer> signs;Q::Buffer rotated;
    Q::Buffer x,h,y,qkv,z,al,be,hc,gdn,qg,k,v,q,gate,att,fg,fu,scores,logits,embrow,error,xq8;
    Q::Buffer bscratch;
    Q::Buffer mtp_hidden,mtp_eh,mtp_logits;
    uint64_t cache_bytes=0,scratch_bytes=0;size_t emb_stride=0;const strata::TensorInfo* embedding=nullptr;
    int position=0;
    bool conservative=false;
    bool fp32_decode=false;
    bool fp32_prefill=false;
    bool fp16_gemv=false;  // policy-B opt-in: FP16 dequant + cuBLAS GEMM for decode-sized batches
    int spec_draft_n_max=0;
    bool mtp_present=false;
    bool mtp_ready=false;
    cudaGraphExec_t gdn_graphs[16]={};
    cudaGraphExec_t ffn_graphs[16]={};
    cudaGraphExec_t attn_proj_graphs[16]={};
    cudaGraphExec_t head_graph=nullptr;
    bool graphs_ready=false;
    bool h_q8_valid=false;
    float* bx_ptr()const{return bscratch.as<float>();}
    float* bh_ptr()const{return bscratch.as<float>()+size_t(prefill_chunk)*size_t(G.hidden);}
    float* by_ptr()const{return bscratch.as<float>()+size_t(prefill_chunk)*(size_t(G.hidden)+size_t(G.hidden));}
    float* bsub_ptr()const{return bscratch.as<float>()+size_t(prefill_chunk)*size_t(G.hidden)*3;}
    float* brot_ptr()const{return bscratch.as<float>()+size_t(prefill_chunk)*(size_t(G.hidden)*3+o_sub);}
    // Cached Hadamard of bh_p for the common hidden-width, ungrouped case.
    float* brot_hidden_ptr()const{return bscratch.as<float>()+size_t(prefill_chunk)*(size_t(G.hidden)*3+o_sub+size_t(G.ffn));}
    bool rot_hidden_valid=false;
    Q::CvecRuntime cvec;
    std::string mtp_name(const char* role)const{return "blk."+std::to_string(G.mtp_layer())+"."+role;}
    ~Engine(){
        for(int g=0;g<16;++g){if(gdn_graphs[g])cudaGraphExecDestroy(gdn_graphs[g]);if(ffn_graphs[g])cudaGraphExecDestroy(ffn_graphs[g]);if(attn_proj_graphs[g])cudaGraphExecDestroy(attn_proj_graphs[g]);}
        if(head_graph)cudaGraphExecDestroy(head_graph);
    }
    Engine(const std::string& path,int context,const Q::KvConfig& config,bool reference=false,bool ja=false,int draft_n_max=0,
           const std::string& cvec_path="",float cvec_scale=1.f,int cvec_first=1,int cvec_last=-1,int cvec_mode=0):
        file(path,Q::retained_metadata(),true,65536,64*1024*1024),admission(Q::inspect(file)),G(admission.geometry),ctx(context),conservative(reference),
        fp32_decode(reference||std::getenv("STRATA_Q38_FP32_DECODE")!=nullptr),fp32_prefill(reference||std::getenv("STRATA_Q38_FP32_PREFILL")!=nullptr),spec_draft_n_max(draft_n_max){
        if(!cvec_path.empty()){
            std::string cvec_err;
            const int last=cvec_last>0?cvec_last:G.trunk_layers;
            if(!Q::cvec_load(cvec,cvec_path,cvec_scale,cvec_first,last,cvec_mode,G.hidden,cvec_err))
                throw std::runtime_error("control vector: "+cvec_err);
            std::fprintf(stderr,"control_vector=loaded path=%s layers=%d..%d mode=%s hidden=%d\n",
                cvec_path.c_str(),cvec_first,last,cvec_mode?"add":"project",G.hidden);
        }
        japanese=ja;
        const Q::KvShape shape{G.q_heads,G.kv_heads,G.head_dim};
        if(ja){strata::GgufFile vocabulary(path,{"tokenizer.ggml.tokens","tokenizer.ggml.token_type"},true,G.vocab);
            const auto* ts=vocabulary.get("tokenizer.ggml.tokens");const auto* ty=vocabulary.get("tokenizer.ggml.token_type");
            Q::require(ts&&ty&&ts->items.size()==size_t(G.vocab)&&ty->items.size()==size_t(G.vocab),"language profile requires complete GGUF tokens/types");
            std::vector<std::string> tokens;std::vector<int> types;for(size_t i=0;i<ts->items.size();++i){tokens.push_back(ts->items[i].s);types.push_back(int(ty->items[i].u));}
            orthography=std::make_unique<Q::Orthography>(tokens,types);
        }
        Q::host_resources();Q::ck(cudaSetDevice(0));cudaDeviceProp prop{};Q::ck(cudaGetDeviceProperties(&prop,0));
        Q::require(prop.major==12&&prop.minor==0,"this build requires CUDA SM120 device 0");
        size_t before_preload=0,after_preload=0,preload_total=0;
        Q::ck(cudaMemGetInfo(&before_preload,&preload_total));Q::warmup_runtime(stream.s);
        Q::ck(cudaMemGetInfo(&after_preload,&preload_total));
        std::fprintf(stderr,"kernel_preload_free_before=%zu free_after=%zu\n",before_preload,after_preload);
        const uint64_t recurrent_bytes=uint64_t(G.rank)*(uint64_t(G.gdn_qkv_channels())*3+uint64_t(G.rank)*uint64_t(G.state)*uint64_t(G.state))*4;
        embedding=file.find("token_embd.weight");emb_stride=Q::payload(*embedding)/uint64_t(G.vocab);
        const size_t scores_elems=(!config.automatic&&config.mode==Q::KvMode::spin8_k8v4)
            ?std::min<size_t>(size_t(G.q_heads)*ctx,size_t(G.q_heads)*size_t(G.attention_layers())*258)
            :size_t(G.q_heads)*ctx;
        o_qkv=0;
        o_z=size_t(G.gdn_qkv_channels());
        o_gdn=o_z+size_t(G.inner);
        o_al=o_gdn+size_t(G.inner);
        o_be=o_al+size_t(G.rank);
        o_qg=0;
        o_bk=size_t(G.attn_q_rows());
        o_bv=o_bk+size_t(G.attn_kv_rows());
        o_bq=o_bv+size_t(G.attn_kv_rows());
        o_bgate=o_bq+size_t(G.attn_out_rows());
        o_batt=o_bgate+size_t(G.attn_out_rows());
        o_bfg=0;
        o_bfu=size_t(G.ffn);
        o_sub=std::max({o_be+size_t(G.rank),o_batt+size_t(G.attn_out_rows()),o_bfu+size_t(G.ffn)});
        uint64_t base_scratch=(uint64_t(G.hidden)*3+uint64_t(G.gdn_qkv_channels())*2+uint64_t(G.inner)*5+uint64_t(G.rank)*2+uint64_t(G.attn_q_rows())+uint64_t(G.attn_kv_rows())*2+uint64_t(G.ffn)*2+uint64_t(G.vocab))*4
            +uint64_t(scores_elems)*4+4+uint64_t(G.ffn)*4+emb_stride;
        for(const auto& entry:admission.signs)base_scratch+=entry.second.size()*4;
        // Include measured 2 MiB backing slack for large cudaMalloc buffers.
        // The retained 16 MiB safety allowance also covers small-buffer pools.
        auto large_slack=[](uint64_t bytes)->uint64_t {constexpr uint64_t granule=2*1024*1024;return bytes>=1024*1024?(granule-bytes%granule)%granule:0;};
        // Retain the small-pool allowance and another 128 MiB for launch-size
        // dependent residency beyond the tiny pre-admission startup warmup.
        constexpr uint64_t first_use_allowance=128ull*1024*1024;
        Q::WeightLayout layout;
        uint64_t conversion_peak=0;for(const auto& name:admission.required){const auto& t=*file.find(name);if(name=="token_embd.weight")continue;
            layout.append(Q::convert_auxiliary(t)?Q::WeightLayout::f32_bytes(t.elements()):Q::payload(t));
            if(Q::convert_auxiliary(t)&&t.type!=0)conversion_peak=std::max(conversion_peak,Q::payload(t));}
        Q::require(layout.payload()==admission.gpu_weights,"weight arena/admission payload mismatch");
        const uint64_t base_allowance=16*1024*1024+first_use_allowance
            +uint64_t(G.rank)*large_slack(uint64_t(G.rank)*uint64_t(G.state)*uint64_t(G.state)*4)
            +large_slack(uint64_t(scores_elems)*4)+large_slack(layout.bytes());
        const size_t bscratch_floats_per_tok=size_t(G.hidden)*3+o_sub+(admission.folded.empty()?0:size_t(G.ffn)+size_t(G.hidden));
        size_t available=0,total=0;Q::ck(cudaMemGetInfo(&available,&total));
        uint64_t allocator_allowance=base_allowance;
        if(reference){
            scratch_bytes=base_scratch;
            kv_selection=Q::select_kv(config,ctx,G.attention_layers(),{layout.bytes(),recurrent_bytes,scratch_bytes,conversion_peak,allocator_allowance},available,Q::GPU_RESERVE,shape);
        }else{
            struct PrefillCand{int chunk;size_t tile;};
            constexpr PrefillCand kCands[]={
                {1024,20ull*1024*1024},{512,20ull*1024*1024},{256,16ull*1024*1024},
                {128,16ull*1024*1024},{64,8ull*1024*1024},{32,8ull*1024*1024}
            };
            auto eval_cand=[&](int ch,size_t tb,uint64_t& out_scratch,uint64_t& out_allow)->Q::KvBudget{
                const uint64_t ws_b=Q::PrefillGemmWorkspace::estimate_bytes(ch,G.ffn,tb);
                const uint64_t bs_b=uint64_t(ch)*bscratch_floats_per_tok*4;
                const uint64_t q8_b=(uint64_t(G.ffn)/32)*sizeof(Q::BlockQ8_1);
                const uint64_t act16_b=uint64_t(ch)*uint64_t(G.ffn)*2;
                const uint64_t y16_b=uint64_t(ch)*std::max<uint64_t>(16,tb/(uint64_t(G.hidden)*2))*2;
                out_scratch=base_scratch+ws_b+bs_b+q8_b;
                out_allow=base_allowance+large_slack(bs_b)+large_slack(tb)+large_slack(act16_b)+large_slack(y16_b);
                return {layout.bytes(),recurrent_bytes,out_scratch,conversion_peak,out_allow};
            };
            // Opt-in tuning overrides. Both values go through the same admission
            // evaluation as the built-in candidates, so scratch stays accounted.
            const char* chunk_env=std::getenv("STRATA_Q38_PREFILL_CHUNK");
            const char* tile_env=std::getenv("STRATA_Q38_PREFILL_TILE_MB");
            int env_chunk=chunk_env?integer(chunk_env,32,4096):0;
            size_t env_tile=tile_env?size_t(integer(tile_env,1,1024))*1024*1024:0;
            if(env_chunk&&env_tile){
                uint64_t sc=0,al=0;
                const auto sel=Q::select_kv(config,ctx,G.attention_layers(),eval_cand(env_chunk,env_tile,sc,al),available,Q::GPU_RESERVE,shape);
                prefill_chunk=env_chunk;prefill_tile_bytes=env_tile;scratch_bytes=sc;allocator_allowance=al;kv_selection=sel;
                std::fprintf(stderr,"prefill_tuning=env chunk=%d tile_bytes=%zu\n",prefill_chunk,prefill_tile_bytes);
            }else{
            uint64_t min_sc=0,min_al=0;
            const auto min_sel=Q::select_kv(config,ctx,G.attention_layers(),eval_cand(32,8ull*1024*1024,min_sc,min_al),available,Q::GPU_RESERVE,shape);
            const auto best_mode=min_sel.config.mode;
            prefill_chunk=32;prefill_tile_bytes=8ull*1024*1024;scratch_bytes=min_sc;allocator_allowance=min_al;kv_selection=min_sel;
            for(const auto& cand:kCands){
                uint64_t sc=0,al_b=0;
                try{
                    auto sel=Q::select_kv(config,ctx,G.attention_layers(),eval_cand(cand.chunk,cand.tile,sc,al_b),available,Q::GPU_RESERVE,shape);
                    if(sel.config.mode==best_mode){
                        prefill_chunk=cand.chunk;prefill_tile_bytes=cand.tile;
                        scratch_bytes=sc;allocator_allowance=al_b;kv_selection=sel;
                        break;
                    }
                }catch(const std::exception&){}
            }
            }
        }
        const auto& effective=kv_selection.config;cache_bytes=recurrent_bytes+kv_selection.payload;
        std::fprintf(stderr,"kv_requested=%s kv_effective=%s kv_k=%s kv_v=%s reason=%s required_with_reserve=%llu available=%llu reserve=%llu\n",kv_selection.requested.c_str(),effective.name(),effective.key_format(),effective.value_format(),kv_selection.reason.c_str(),(unsigned long long)kv_selection.required,(unsigned long long)kv_selection.available,(unsigned long long)kv_selection.reserve);
        std::fprintf(stderr,"kv_payload_bytes=%llu kv_allocator_slack=%llu nonkv_allocator_allowance=%llu recurrent_bytes=%llu scratch_bytes=%llu auxiliary_peak_bytes=%llu\n",(unsigned long long)kv_selection.payload,(unsigned long long)kv_selection.allocator_slack,(unsigned long long)allocator_allowance,(unsigned long long)recurrent_bytes,(unsigned long long)scratch_bytes,(unsigned long long)conversion_peak);
        gpu_resources(kv_selection.required-Q::GPU_RESERVE,"admission");
        gpu_resources(layout.bytes(),"weight_arena");
        size_t arena_before=0,arena_after=0;Q::ck(cudaMemGetInfo(&arena_before,&total));
        weight_arena=std::make_unique<Q::WeightArena>(layout);Q::ck(cudaMemGetInfo(&arena_after,&total));
        std::fprintf(stderr,"weight_arena_payload=%zu weight_arena_bytes=%zu weight_arena_padding=%zu weight_arena_allocations=1 free_before=%zu free_after=%zu first_use_allowance=%llu\n",layout.payload(),layout.bytes(),layout.bytes()-layout.payload(),arena_before,arena_after,(unsigned long long)first_use_allowance);
        kv=std::make_unique<Q::KvCache>(effective,ctx,G.attention_layers(),stream.s,conservative,shape);
        rotated.allocate(uint64_t(G.ffn)*4);
        for(const auto& entry:admission.signs){Q::Buffer b(entry.second.size()*4);Q::ck(cudaMemcpy(b.data,entry.second.data(),b.bytes,cudaMemcpyHostToDevice));signs.emplace(entry.first,std::move(b));}
        std::fprintf(stderr,"own-native CUDA SM120 weights=%llu cache=%llu scratch_upper=%llu host_embedding=%llu ignored_mtp=%llu\n",
            (unsigned long long)admission.gpu_weights,(unsigned long long)cache_bytes,(unsigned long long)scratch_bytes,
            (unsigned long long)admission.embedding_bytes,(unsigned long long)admission.ignored_bytes);
        {
        Q::UploadReader reader(path,file.file_size());
        std::fprintf(stderr,"weight_upload=bounded_file_read host_staging_bytes=%zu\n",reader.staging_bytes());
        auto upload=[&](const strata::TensorInfo& t,Q::WeightSpan dest,const std::string& name){
            reader.read(file.data_start()+t.offset,dest.bytes,[&]{Q::host_resources();gpu_resources(0,"weight_upload",name);},
                [&](uint64_t offset,const uint8_t* bytes,size_t n){Q::ck(cudaMemcpy(dest.as<uint8_t>()+offset,bytes,n,cudaMemcpyHostToDevice));});
        };
        size_t weight_index=0;for(const auto& name:admission.required){
            if(name=="token_embd.weight")continue;Q::host_resources();const auto& t=*file.find(name);Weight w;
            w.b=weight_arena->view(weight_index++);
            w.n=int(t.shape[0]);w.rows=t.shape.size()==2?int(t.shape[1]):1;w.type=t.type;
            if(Q::convert_auxiliary(t)&&t.type!=0){
                // Conversion scratch never exceeds one auxiliary tensor (alpha/beta < 1 MiB).
                gpu_resources(Q::payload(t),"auxiliary_conversion",name);Q::Buffer raw(Q::payload(t));upload(t,{raw.data,raw.bytes},name);
                Q::decode(t.type,raw.data,w.b.as(),int(t.elements()),stream.s);Q::ck(cudaStreamSynchronize(stream.s));w.type=0;
            }else{upload(t,w.b,name);}
            weights.emplace(name,std::move(w));gpu_resources(0,"weight_uploaded",name);
        }
        }
        states.resize(G.trunk_layers);
        for(int i=0;i<G.trunk_layers;++i)if((i+1)%G.interval){states[i].conv.allocate(uint64_t(G.gdn_qkv_channels())*3*4);states[i].gdn.allocate(uint64_t(G.rank)*uint64_t(G.state)*uint64_t(G.state)*4);}
        auto a=[](Q::Buffer& b,size_t n){b.allocate(n*4);};
        a(x,G.hidden);a(h,G.hidden);a(y,G.hidden);a(qkv,G.gdn_qkv_channels());a(hc,G.gdn_qkv_channels());a(z,G.inner);a(gdn,G.inner);a(al,G.rank);a(be,G.rank);
        a(qg,G.attn_q_rows());a(k,G.attn_kv_rows());a(v,G.attn_kv_rows());a(q,G.attn_out_rows());a(gate,G.attn_out_rows());a(att,G.attn_out_rows());a(fg,G.ffn);a(fu,G.ffn);
        a(scores,scores_elems);a(logits,G.vocab);a(error,1);
        embrow.allocate(emb_stride);
        if(!conservative){
            prefill_ws.init(prefill_chunk,G.ffn,prefill_tile_bytes);
            a(bscratch,size_t(prefill_chunk)*bscratch_floats_per_tok);
            xq8.allocate((uint64_t(G.ffn)/32)*sizeof(Q::BlockQ8_1));
        }
        layers.resize(G.trunk_layers);
        for(int i=0;i<G.trunk_layers;++i){
            auto& L=layers[i];
            L.layer_index=i;
            L.mixer_type=((i+1)%G.interval)?Q::SequenceMixerType::GATED_DELTANET_GROUPED:Q::SequenceMixerType::GATED_ATTENTION_PARTIAL_ROPE;
            L.ffn_type=Q::FfnBlockType::DENSE_SWIGLU;
            auto n=[i](const char* r){return Q::layer_name(i,r);};
            L.attn_norm=weights.at(n("attn_norm.weight")).quant();
            L.post_attn_norm=weights.at(n("post_attention_norm.weight")).quant();
            if(L.mixer_type==Q::SequenceMixerType::GATED_DELTANET_GROUPED){
                L.w_qkv=weights.at(n("attn_qkv.weight")).quant();
                L.w_out_gate=weights.at(n("attn_gate.weight")).quant();
                L.w_out_proj=weights.at(n("ssm_out.weight")).quant();
                L.w_ssm_conv1d=weights.at(n("ssm_conv1d.weight")).quant();
                L.w_ssm_alpha=weights.at(n("ssm_alpha.weight")).quant();
                L.w_ssm_beta=weights.at(n("ssm_beta.weight")).quant();
                L.w_ssm_dt_bias=weights.at(n("ssm_dt.bias")).quant();
                L.w_ssm_a=weights.at(n("ssm_a")).quant();
                L.w_ssm_norm=weights.at(n("ssm_norm.weight")).quant();
            }else{
                L.w_q=weights.at(n("attn_q.weight")).quant();
                L.w_k=weights.at(n("attn_k.weight")).quant();
                L.w_v=weights.at(n("attn_v.weight")).quant();
                L.w_q_norm=weights.at(n("attn_q_norm.weight")).quant();
                L.w_k_norm=weights.at(n("attn_k_norm.weight")).quant();
                L.w_out_proj=weights.at(n("attn_output.weight")).quant();
            }
            L.w_ffn_gate=weights.at(n("ffn_gate.weight")).quant();
            L.w_ffn_up=weights.at(n("ffn_up.weight")).quant();
            L.w_ffn_down=weights.at(n("ffn_down.weight")).quant();
        }
        mtp_present=(admission.nextn==1 && file.find(mtp_name("nextn.eh_proj.weight"))!=nullptr);
        if(mtp_present && spec_draft_n_max>0 && !conservative){
            Q::WeightLayout mtp_layout;
            for(const auto& name:admission.mtp_required){
                const auto& t=*file.find(name);
                mtp_layout.append(Q::convert_auxiliary(t)?Q::WeightLayout::f32_bytes(t.elements()):Q::payload(t));
            }
            const uint64_t mtp_kv_bytes=Q::kv_storage_bytes(effective,ctx,1,shape);
            const uint64_t mtp_need=mtp_layout.bytes()+mtp_kv_bytes+recurrent_bytes+uint64_t(4)*uint64_t(G.vocab)*4+64ull*1024*1024;
            size_t avail_mtp=0,tot_mtp=0;Q::ck(cudaMemGetInfo(&avail_mtp,&tot_mtp));
            if(avail_mtp>Q::GPU_RESERVE && mtp_need<=avail_mtp-Q::GPU_RESERVE){
                mtp_arena=std::make_unique<Q::WeightArena>(mtp_layout);
                Q::UploadReader mtp_reader(path,file.file_size());
                size_t wi=0;
                for(const auto& name:admission.mtp_required){
                    const auto& t=*file.find(name);Weight w;
                    w.b=mtp_arena->view(wi++);w.n=int(t.shape[0]);w.rows=t.shape.size()==2?int(t.shape[1]):1;w.type=t.type;
                    auto up=[&](const strata::TensorInfo& ti,Q::WeightSpan dst){
                        mtp_reader.read(file.data_start()+ti.offset,dst.bytes,[&]{Q::host_resources();gpu_resources(0,"mtp_upload",name);},
                            [&](uint64_t off,const uint8_t* bytes,size_t n){Q::ck(cudaMemcpy(dst.as<uint8_t>()+off,bytes,n,cudaMemcpyHostToDevice));});
                    };
                    if(Q::convert_auxiliary(t)&&t.type!=0){
                        Q::Buffer raw(Q::payload(t));up(t,{raw.data,raw.bytes});
                        Q::decode(t.type,raw.data,w.b.as(),int(t.elements()),stream.s);Q::ck(cudaStreamSynchronize(stream.s));w.type=0;
                    }else{up(t,w.b);}
                    weights.emplace(name,std::move(w));
                }
                mtp_kv=std::make_unique<Q::KvCache>(effective,ctx,1,stream.s,false,shape);
                snap_states.resize(G.trunk_layers);
                for(int i=0;i<G.trunk_layers;++i)if((i+1)%G.interval){snap_states[i].conv.allocate(uint64_t(G.gdn_qkv_channels())*3*4);snap_states[i].gdn.allocate(uint64_t(G.rank)*uint64_t(G.state)*uint64_t(G.state)*4);}
                a(mtp_hidden,G.hidden);a(mtp_eh,size_t(2)*G.hidden);a(mtp_logits,size_t(4)*G.vocab);
                mtp_ready=true;
            }
        }
        reset();if(!conservative)ensure_decode_graphs();gpu_resources();Q::host_resources();
        std::fprintf(stderr,"native_execution=%s layer_fence_interval=%d prefill_gemm=%s prefill_chunk=%d prefill_tile_bytes=%zu mtp_present=%d mtp_ready=%d spec_draft_n_max=%d fp16_gemv=%d\n",conservative?"conservative":"grouped",conservative?1:8,conservative?"off":"cublas_tensor_op",conservative?1:prefill_chunk,conservative?size_t(0):prefill_tile_bytes,mtp_present?1:0,mtp_ready?1:0,spec_draft_n_max,fp16_gemv?1:0);
    }
    const float* f(const std::string& name){return weights.at(name).b.as();}
    void rms_h(const float* src,const std::string& wname){
        if(!fp32_decode){
            Q::rms_quantize_q8_1(src,f(wname),h.as(),xq8.as<Q::BlockQ8_1>(),G.hidden,admission.eps,stream.s);
            h_q8_valid=true;
        }else{
            Q::rms(src,f(wname),h.as(),G.hidden,admission.eps,stream.s);
            h_q8_valid=false;
        }
    }
    void mm(const std::string& name,const Q::Buffer& in,Q::Buffer& out){const auto& w=weights.at(name);const float* input=in.as();
        const bool folded=admission.folded.count(name)>0;
        if(folded){Q::hadamard(input,signs.at(w.n).as(),rotated.as(),w.n,false,admission.grouped_gdn&&name.find(".ssm_out.")!=std::string::npos,stream.s);input=rotated.as();h_q8_valid=false;}
        if(!fp32_decode&&is_q8_matvec_type(w.type)){
            if(folded||in.data!=h.data||!h_q8_valid){
                Q::quantize_q8_1(input,xq8.as<Q::BlockQ8_1>(),w.n,stream.s);
                h_q8_valid=(!folded&&in.data==h.data);
            }
            Q::matvec_q8_1(w.type,w.b.data,xq8.as<Q::BlockQ8_1>(),input,out.as(),w.n,w.rows,stream.s);
        }else{
            Q::matvec(w.type,w.b.data,input,out.as(),w.n,w.rows,stream.s);
        }}
    void mm_dual_h(const std::string& n0,Q::Buffer& out0,const std::string& n1,Q::Buffer& out1){
        const auto& w0=weights.at(n0);const auto& w1=weights.at(n1);
        if(!fp32_decode&&w0.type==w1.type&&is_q8_matvec_type(w0.type)&&!admission.folded.count(n0)&&!admission.folded.count(n1)){
            if(!h_q8_valid){Q::quantize_q8_1(h.as(),xq8.as<Q::BlockQ8_1>(),w0.n,stream.s);h_q8_valid=true;}
            Q::matvec_q8_1_dual(w0.type,w0.b.data,out0.as(),w0.rows,w1.b.data,out1.as(),w1.rows,xq8.as<Q::BlockQ8_1>(),h.as(),w0.n,stream.s);
        }else{
            mm(n0,h,out0);mm(n1,h,out1);
        }
    }
    void mm_triple_h(const std::string& n0,Q::Buffer& out0,const std::string& n1,Q::Buffer& out1,const std::string& n2,Q::Buffer& out2){
        const auto& w0=weights.at(n0);const auto& w1=weights.at(n1);const auto& w2=weights.at(n2);
        if(!fp32_decode&&w0.type==w1.type&&w1.type==w2.type&&is_q8_matvec_type(w0.type)&&!admission.folded.count(n0)&&!admission.folded.count(n1)&&!admission.folded.count(n2)){
            if(!h_q8_valid){Q::quantize_q8_1(h.as(),xq8.as<Q::BlockQ8_1>(),w0.n,stream.s);h_q8_valid=true;}
            Q::matvec_q8_1_triple(w0.type,w0.b.data,out0.as(),w0.rows,w1.b.data,out1.as(),w1.rows,w2.b.data,out2.as(),w2.rows,xq8.as<Q::BlockQ8_1>(),h.as(),w0.n,stream.s);
        }else{
            mm(n0,h,out0);mm_dual_h(n1,out1,n2,out2);
        }
    }
    void emit_ffn_sublayer(const std::string& gate_name,const std::string& up_name,const std::string& down_name){
        if(fp16_gemv&&!conservative){
            // Policy-B opt-in: reuse the shipped tiled FP16 dequant + cuBLAS GEMM
            // path for decode-sized (n=1) batches. Reduction order differs from
            // the packed DP4A matvec; this is deliberately not byte-identical.
            prefill_ws.prepare_activation(h.as(),1,weights.at(gate_name).n,stream.s);
            prefill_ws.matmul_batch_prepared(weights.at(gate_name).quant(),fg.as(),1,stream.s);
            prefill_ws.matmul_batch_prepared(weights.at(up_name).quant(),fu.as(),1,stream.s);
            Q::swiglu(fg.as(),fu.as(),17408,stream.s);
            prefill_ws.prepare_activation(fg.as(),1,weights.at(down_name).n,stream.s);
            prefill_ws.matmul_batch_prepared(weights.at(down_name).quant(),y.as(),1,stream.s);
            return;
        }
        mm_dual_h(gate_name,fg,up_name,fu);
        const auto& wd=weights.at(down_name);
        if(!fp32_decode&&!admission.folded.count(down_name)&&is_q8_matvec_type(wd.type)){
            Q::swiglu_quantize_q8_1(fg.as(),fu.as(),xq8.as<Q::BlockQ8_1>(),G.ffn,stream.s);
            h_q8_valid=false;
            Q::matvec_q8_1(wd.type,wd.b.data,xq8.as<Q::BlockQ8_1>(),fg.as(),y.as(),wd.n,wd.rows,stream.s);
        }else{
            Q::swiglu(fg.as(),fu.as(),G.ffn,stream.s);
            mm(down_name,fg,y);
        }
    }
    void mm_batch(const std::string& name,const Q::StrataQuantTensor& qt,const float* in,float* out,int n_tokens,const float* pre_rot=nullptr){
        const float* input=in;
        if(admission.folded.count(name)&&!pre_rot){
            const bool grp=admission.grouped_gdn&&name.find(".ssm_out.")!=std::string::npos;
            const float* sg=signs.at(qt.ne0).as();
            float* dst=brot_ptr();
            for(int t=0;t<n_tokens;++t)
                Q::hadamard(in+size_t(t)*qt.ne0,sg,dst+size_t(t)*qt.ne0,qt.ne0,false,grp,stream.s);
            input=dst;
        }else if(admission.folded.count(name)&&pre_rot)input=pre_rot;
        prefill_ws.matmul_batch(qt,input,out,n_tokens,stream.s);
    }
    void reset(){
        fp32_decode=conservative||(std::getenv("STRATA_Q38_FP32_DECODE")!=nullptr);
        fp16_gemv=std::getenv("STRATA_Q38_FP16_GEMV")!=nullptr&&std::getenv("STRATA_Q38_FP16_GEMV")[0]=='1';
        if(!conservative)prefill_ws.reset_l2_window(stream.s);
        Q::reset_session(states,*kv,error,position,stream.s);
        if(mtp_kv)mtp_kv->reset();
    }
    void snapshot_gdn(){
        for(int i=0;i<G.trunk_layers;++i)if((i+1)%G.interval){
            Q::ck(cudaMemcpyAsync(snap_states[i].conv.data,states[i].conv.data,states[i].conv.bytes,cudaMemcpyDeviceToDevice,stream.s));
            Q::ck(cudaMemcpyAsync(snap_states[i].gdn.data,states[i].gdn.data,states[i].gdn.bytes,cudaMemcpyDeviceToDevice,stream.s));
        }
    }
    void restore_gdn(){
        for(int i=0;i<G.trunk_layers;++i)if((i+1)%G.interval){
            Q::ck(cudaMemcpyAsync(states[i].conv.data,snap_states[i].conv.data,states[i].conv.bytes,cudaMemcpyDeviceToDevice,stream.s));
            Q::ck(cudaMemcpyAsync(states[i].gdn.data,snap_states[i].gdn.data,states[i].gdn.bytes,cudaMemcpyDeviceToDevice,stream.s));
        }
    }
    int mtp_draft_step(int token,int mtp_pos){
        Q::require(token>=0&&token<G.vocab&&mtp_pos<ctx,"MTP token or context out of range");
        Q::ck(cudaMemcpyAsync(embrow.data,file.tensor_data(*embedding)+size_t(token)*emb_stride,emb_stride,cudaMemcpyHostToDevice,stream.s));
        Q::decode(embedding->type,embrow.data,x.as(),G.hidden,stream.s);
        if(admission.inverse_embedding){Q::hadamard(x.as(),signs.at(G.hidden).as(),rotated.as(),G.hidden,true,false,stream.s);Q::ck(cudaMemcpyAsync(x.data,rotated.data,size_t(G.hidden)*4,cudaMemcpyDeviceToDevice,stream.s));}
        Q::rms(x.as(),f(mtp_name("nextn.enorm.weight")),mtp_eh.as(),G.hidden,admission.eps,stream.s);
        Q::rms(mtp_hidden.as(),f(mtp_name("nextn.hnorm.weight")),mtp_eh.as()+size_t(G.hidden),G.hidden,admission.eps,stream.s);
        mm(mtp_name("nextn.eh_proj.weight"),mtp_eh,x);
        rms_h(x.as(),mtp_name("attn_norm.weight"));
        mm_triple_h(mtp_name("attn_q.weight"),qg,mtp_name("attn_k.weight"),k,mtp_name("attn_v.weight"),v);
        Q::attention_normalize(qg.as(),k.as(),f(mtp_name("attn_q_norm.weight")),f(mtp_name("attn_k_norm.weight")),q.as(),gate.as(),k.as(),mtp_pos,G.q_heads,G.kv_heads,admission.eps,admission.rope_base,stream.s);
        mtp_kv->enqueue(0,mtp_pos,q.as(),k.as(),v.as(),gate.as(),att.as(),scores.as());
        mm(mtp_name("attn_output.weight"),att,y);
        Q::add(x.as(),y.as(),G.hidden,stream.s);
        rms_h(x.as(),mtp_name("post_attention_norm.weight"));
        emit_ffn_sublayer(mtp_name("ffn_gate.weight"),mtp_name("ffn_up.weight"),mtp_name("ffn_down.weight"));
        Q::add(x.as(),y.as(),G.hidden,stream.s);
        Q::ck(cudaMemcpyAsync(mtp_hidden.data,x.data,size_t(G.hidden)*4,cudaMemcpyDeviceToDevice,stream.s));
        rms_h(x.as(),mtp_name("nextn.shared_head_norm.weight"));
        mm("output.weight",h,logits);
        auto lg=read_logits();
        return int(std::max_element(lg.begin(),lg.end())-lg.begin());
    }
    void emit_gdn_layer(int i){
        auto n=[i](const char* r){return Q::layer_name(i,r);};auto& st=states[i];
        rms_h(x.as(),n("attn_norm.weight"));
        mm_dual_h(n("attn_qkv.weight"),qkv,n("attn_gate.weight"),z);
        mm_dual_h(n("ssm_alpha.weight"),al,n("ssm_beta.weight"),be);
        Q::gdn_gates(al.as(),be.as(),f(n("ssm_dt.bias")),f(n("ssm_a")),G.rank,stream.s);
        K::fused_gdn_conv_l2(st.conv.as(),qkv.as(),f(n("ssm_conv1d.weight")),hc.as(),G.gdn_qkv_channels(),2*G.group,admission.eps,stream.s);
        if(conservative){
            K::fused_gdn_step_norm_silu(st.gdn.as(),hc.as(),hc.as()+2048,hc.as()+4096,al.as(),be.as(),z.as(),f(n("ssm_norm.weight")),admission.eps,gdn.as(),G.group,2*G.q_heads,stream.s);
            Q::finite(st.gdn.as(),size_t(G.rank)*size_t(G.state)*size_t(G.state),error.as<int>(),stream.s);
        }else{
            K::fused_gdn_step_norm_silu(st.gdn.as(),hc.as(),hc.as()+2048,hc.as()+4096,al.as(),be.as(),z.as(),f(n("ssm_norm.weight")),admission.eps,gdn.as(),G.group,2*G.q_heads,stream.s,error.as<int>());
        }
        mm(n("ssm_out.weight"),gdn,y);
        Q::add(x.as(),y.as(),G.hidden,stream.s);rms_h(x.as(),n("post_attention_norm.weight"));
        emit_ffn_sublayer(n("ffn_gate.weight"),n("ffn_up.weight"),n("ffn_down.weight"));
        Q::add(x.as(),y.as(),G.hidden,stream.s);Q::finite(x.as(),G.hidden,error.as<int>(),stream.s);
        cvec.apply(x.as(),1,G.hidden,i,stream.s);
    }
    void emit_attn_post_ffn(int i){
        auto n=[i](const char* r){return Q::layer_name(i,r);};
        mm(n("attn_output.weight"),att,y);
        Q::add(x.as(),y.as(),G.hidden,stream.s);rms_h(x.as(),n("post_attention_norm.weight"));
        emit_ffn_sublayer(n("ffn_gate.weight"),n("ffn_up.weight"),n("ffn_down.weight"));
        Q::add(x.as(),y.as(),G.hidden,stream.s);Q::finite(x.as(),G.hidden,error.as<int>(),stream.s);
        cvec.apply(x.as(),1,G.hidden,i,stream.s);
    }
    template<class Fn> cudaGraphExec_t capture_graph(Fn&& fn){
        cudaGraph_t graph=nullptr;cudaGraphExec_t exec=nullptr;
        Q::ck(cudaStreamBeginCapture(stream.s,cudaStreamCaptureModeThreadLocal));
        fn();
        Q::ck(cudaStreamEndCapture(stream.s,&graph));
        Q::ck(cudaGraphInstantiate(&exec,graph,nullptr,nullptr,0));
        cudaGraphDestroy(graph);
        return exec;
    }
    void ensure_decode_graphs(){
        if(graphs_ready||conservative||fp32_decode)return;
        Q::ck(cudaStreamSynchronize(stream.s));
        for(int g=0;g<G.attention_layers();++g){
            gdn_graphs[g]=capture_graph([&]{emit_gdn_layer(g*G.interval);emit_gdn_layer(g*G.interval+1);emit_gdn_layer(g*G.interval+2);});
            ffn_graphs[g]=capture_graph([&]{emit_attn_post_ffn(g*G.interval+G.interval-1);});
            // Position-dependent RoPE/KV stay outside the graph; only the
            // attention-layer norm and Q/K/V projections are capturable.
            attn_proj_graphs[g]=capture_graph([&,g]{
                const int i=g*G.interval+G.interval-1;
                auto n=[i](const char* r){return Q::layer_name(i,r);};
                rms_h(x.as(),n("attn_norm.weight"));
                mm_triple_h(n("attn_q.weight"),qg,n("attn_k.weight"),k,n("attn_v.weight"),v);
            });
        }
        head_graph=capture_graph([&]{rms_h(x.as(),"output_norm.weight");mm("output.weight",h,logits);Q::finite(logits.as(),G.vocab,error.as<int>(),stream.s);});
        graphs_ready=true;
    }
    void step(int token,std::atomic<bool>& stop,bool head){
        Q::require(token>=0&&token<G.vocab&&position<ctx,"token or context out of range");
        if(!conservative&&!fp32_decode&&!graphs_ready)ensure_decode_graphs();
        Q::host_resources();gpu_resources();error.zero(stream.s);
        Q::ck(cudaMemcpyAsync(embrow.data,file.tensor_data(*embedding)+size_t(token)*emb_stride,emb_stride,cudaMemcpyHostToDevice,stream.s));
        Q::decode(embedding->type,embrow.data,x.as(),G.hidden,stream.s);
        if(admission.inverse_embedding){Q::hadamard(x.as(),signs.at(G.hidden).as(),rotated.as(),G.hidden,true,false,stream.s);Q::ck(cudaMemcpyAsync(x.data,rotated.data,size_t(G.hidden)*4,cudaMemcpyDeviceToDevice,stream.s));}
        const bool prof=(std::getenv("STRATA_Q38_PROFILE")!=nullptr);
        double prof_fence_ms=0,prof_gdn_ms=0,prof_attn_mm_ms=0,prof_kv_ms=0,prof_ffn_ms=0,prof_head_ms=0;
        const Q::LayerFence fences(conservative?1:8);
        int fence_bad=0;
        auto check=[&]{
            auto t0=prof?Clock::now():Clock::time_point{};
            Q::host_resources();gpu_resources();
            fence_bad=0;Q::ck(cudaMemcpyAsync(&fence_bad,error.data,4,cudaMemcpyDeviceToHost,stream.s));
            kv->check();Q::require(!fence_bad,"non-finite activation/state/logit");
            if(prof)prof_fence_ms+=std::chrono::duration<double,std::milli>(Clock::now()-t0).count();
        };
        if(!conservative&&!fp32_decode&&graphs_ready){
            for(int g=0;g<G.attention_layers();++g){
                fences.before(stop);
                auto t0=prof?Clock::now():Clock::time_point{};
                Q::ck(cudaGraphLaunch(gdn_graphs[g],stream.s));h_q8_valid=false;
                if(prof){Q::ck(cudaStreamSynchronize(stream.s));prof_gdn_ms+=std::chrono::duration<double,std::milli>(Clock::now()-t0).count();t0=Clock::now();}
                const int i=g*G.interval+G.interval-1;auto n=[i](const char* r){return Q::layer_name(i,r);};
                Q::ck(cudaGraphLaunch(attn_proj_graphs[g],stream.s));h_q8_valid=false;
                Q::attention_normalize(qg.as(),k.as(),f(n("attn_q_norm.weight")),f(n("attn_k_norm.weight")),q.as(),gate.as(),k.as(),position,G.q_heads,G.kv_heads,admission.eps,admission.rope_base,stream.s);
                if(prof){Q::ck(cudaStreamSynchronize(stream.s));prof_attn_mm_ms+=std::chrono::duration<double,std::milli>(Clock::now()-t0).count();t0=Clock::now();}
                kv->enqueue(i/G.interval,position,q.as(),k.as(),v.as(),gate.as(),att.as(),scores.as());
                if(prof){Q::ck(cudaStreamSynchronize(stream.s));prof_kv_ms+=std::chrono::duration<double,std::milli>(Clock::now()-t0).count();t0=Clock::now();}
                Q::ck(cudaGraphLaunch(ffn_graphs[g],stream.s));h_q8_valid=false;
                if(prof){Q::ck(cudaStreamSynchronize(stream.s));prof_ffn_ms+=std::chrono::duration<double,std::milli>(Clock::now()-t0).count();}
                fences.after(i,stop,check);
            }
            if(head){
                auto t0=prof?Clock::now():Clock::time_point{};
                Q::ck(cudaGraphLaunch(head_graph,stream.s));h_q8_valid=false;
                if(prof){Q::ck(cudaStreamSynchronize(stream.s));prof_head_ms+=std::chrono::duration<double,std::milli>(Clock::now()-t0).count();}
            }
        }else{
            for(int i=0;i<G.trunk_layers;++i){
                fences.before(stop);
                auto n=[i](const char* r){return Q::layer_name(i,r);};
                if((i+1)%G.interval){
                    emit_gdn_layer(i);
                }else{
                    rms_h(x.as(),n("attn_norm.weight"));
                    mm_triple_h(n("attn_q.weight"),qg,n("attn_k.weight"),k,n("attn_v.weight"),v);
                    Q::attention_normalize(qg.as(),k.as(),f(n("attn_q_norm.weight")),f(n("attn_k_norm.weight")),q.as(),gate.as(),k.as(),position,G.q_heads,G.kv_heads,admission.eps,admission.rope_base,stream.s);
                    kv->enqueue(i/G.interval,position,q.as(),k.as(),v.as(),gate.as(),att.as(),scores.as());
                    emit_attn_post_ffn(i);
                }
                // Bounded synchronization makes STOP/resource checks effective during a long token.
                fences.after(i,stop,check);
            }
            if(head){rms_h(x.as(),"output_norm.weight");mm("output.weight",h,logits);Q::finite(logits.as(),G.vocab,error.as<int>(),stream.s);}
        }
        int bad=0;Q::ck(cudaMemcpyAsync(&bad,error.data,4,cudaMemcpyDeviceToHost,stream.s));Q::ck(cudaStreamSynchronize(stream.s));
        if(prof)std::fprintf(stderr,"PROF_STEP pos=%d gdn=%.2f attn_mm=%.2f kv=%.2f ffn=%.2f head=%.2f fence=%.2f\n",position,prof_gdn_ms,prof_attn_mm_ms,prof_kv_ms,prof_ffn_ms,prof_head_ms,prof_fence_ms);
        Q::require(!bad,"non-finite activation/state/logit");fences.before(stop);++position;
    }
    void run_prefill_chunk(const int32_t* tokens,int n_tokens,std::atomic<bool>& stop,bool head){
        Q::require(n_tokens>0&&n_tokens<=prefill_chunk&&position+n_tokens<=ctx,"prefill chunk out of range");
        Q::host_resources();gpu_resources();error.zero(stream.s);
        float* const bx_p=bx_ptr();
        float* const bh_p=bh_ptr();
        float* const by_p=by_ptr();
        float* const sub_p=bsub_ptr();
        const size_t C=size_t(prefill_chunk);
        for(int t=0;t<n_tokens;++t){
            const int tok=tokens[t];Q::require(tok>=0&&tok<G.vocab,"token out of range");
            Q::ck(cudaMemcpyAsync(embrow.data,file.tensor_data(*embedding)+size_t(tok)*emb_stride,emb_stride,cudaMemcpyHostToDevice,stream.s));
            Q::decode(embedding->type,embrow.data,bx_p+size_t(t)*size_t(G.hidden),G.hidden,stream.s);
            if(admission.inverse_embedding){
                Q::hadamard(bx_p+size_t(t)*size_t(G.hidden),signs.at(G.hidden).as(),rotated.as(),G.hidden,true,false,stream.s);
                Q::ck(cudaMemcpyAsync(bx_p+size_t(t)*size_t(G.hidden),rotated.data,size_t(G.hidden)*4,cudaMemcpyDeviceToDevice,stream.s));
            }
        }
        const bool prof=(std::getenv("STRATA_Q38_PROFILE")!=nullptr);
        const char* fp8_env=std::getenv("STRATA_FP8_PREFILL");
        const bool use_fp8=(fp8_env&&fp8_env[0]=='1');
        const bool can_share_bh=(n_tokens>=4&&!use_fp8
#if defined(STRATA_Q38_FUSED_WMMA)
            &&false
#endif
        );
        double prof_fence=0,prof_gdn_mm=0,prof_gdn_core=0,prof_attn_mm=0,prof_attn_kv=0,prof_ffn_mm=0;
        const Q::LayerFence fences(8);
        int fence_bad=0;
        auto check=[&]{
            auto t0=prof?Clock::now():Clock::time_point{};
            Q::host_resources();gpu_resources();
            fence_bad=0;Q::ck(cudaMemcpyAsync(&fence_bad,error.data,4,cudaMemcpyDeviceToHost,stream.s));
            kv->check();Q::require(!fence_bad,"non-finite activation/state/logit");
            if(prof)prof_fence+=std::chrono::duration<double,std::milli>(Clock::now()-t0).count();
        };
        bool bh_prepared=false;
        auto mm_bh=[&](const std::string& name,const Q::StrataQuantTensor& qt,float* out){
            if(!can_share_bh||admission.folded.count(name)){
                bh_prepared=false;
                // All input-side folded projections share the same hidden-width
                // signs and the same bh_p. ssm_out uses the grouped transform
                // when grouped_gdn is set, so it keeps its own path.
                const bool reusable=admission.folded.count(name)&&qt.ne0==G.hidden&&
                    !(admission.grouped_gdn&&name.find(".ssm_out.")!=std::string::npos);
                if(reusable&&!rot_hidden_valid){
                    const float* sg=signs.at(qt.ne0).as();
                    float* dst=brot_hidden_ptr();
                    for(int t=0;t<n_tokens;++t)
                        Q::hadamard(bh_p+size_t(t)*qt.ne0,sg,dst+size_t(t)*qt.ne0,qt.ne0,false,false,stream.s);
                    rot_hidden_valid=true;
                }
                mm_batch(name,qt,bh_p,out,n_tokens,reusable?brot_hidden_ptr():nullptr);
            }else{
                if(!bh_prepared){
                    prefill_ws.prepare_activation(bh_p,n_tokens,G.hidden,stream.s);
                    bh_prepared=true;
                }
                prefill_ws.matmul_batch_prepared(qt,out,n_tokens,stream.s);
            }
        };
        for(int i=0;i<G.trunk_layers;++i){
            fences.before(stop);
            const auto& L=layers[i];auto n=[i](const char* r){return Q::layer_name(i,r);};auto& st=states[i];
            auto tp=prof?Clock::now():Clock::time_point{};
            Q::rms_batch(bx_p,static_cast<const float*>(L.attn_norm.d_data),bh_p,n_tokens,G.hidden,admission.eps,stream.s);
            bh_prepared=false;
            rot_hidden_valid=false;
            if(L.mixer_type==Q::SequenceMixerType::GATED_DELTANET_GROUPED){
                float* const bqkv_p=sub_p;
                float* const bz_p=sub_p+C*o_z;
                float* const bgdn_p=sub_p+C*o_gdn;
                float* const bal_p=sub_p+C*o_al;
                float* const bbe_p=sub_p+C*o_be;
                mm_bh(n("attn_qkv.weight"),L.w_qkv,bqkv_p);
                mm_bh(n("attn_gate.weight"),L.w_out_gate,bz_p);
                mm_bh(n("ssm_alpha.weight"),L.w_ssm_alpha,bal_p);
                mm_bh(n("ssm_beta.weight"),L.w_ssm_beta,bbe_p);
                if(prof){Q::ck(cudaStreamSynchronize(stream.s));prof_gdn_mm+=std::chrono::duration<double,std::milli>(Clock::now()-tp).count();tp=Clock::now();}
                Q::gdn_gates_batch(bal_p,bbe_p,static_cast<const float*>(L.w_ssm_dt_bias.d_data),static_cast<const float*>(L.w_ssm_a.d_data),n_tokens,G.rank,stream.s);
                K::fused_gdn_conv_l2_batch(st.conv.as(),bqkv_p,static_cast<const float*>(L.w_ssm_conv1d.d_data),bqkv_p,G.gdn_qkv_channels(),2*G.group,admission.eps,n_tokens,stream.s);
                K::fused_gdn_step_norm_silu_batch(st.gdn.as(),bqkv_p,G.gdn_qkv_channels(),bal_p,bbe_p,bz_p,static_cast<const float*>(L.w_ssm_norm.d_data),admission.eps,bgdn_p,G.group,2*G.q_heads,n_tokens,stream.s,error.as<int>());
                if(prof){Q::ck(cudaStreamSynchronize(stream.s));prof_gdn_core+=std::chrono::duration<double,std::milli>(Clock::now()-tp).count();tp=Clock::now();}
                mm_batch(n("ssm_out.weight"),L.w_out_proj,bgdn_p,by_p,n_tokens);
                if(prof){Q::ck(cudaStreamSynchronize(stream.s));prof_gdn_mm+=std::chrono::duration<double,std::milli>(Clock::now()-tp).count();}
            }else{
                float* const bqg_p=sub_p;
                float* const bk_p=sub_p+C*o_bk;
                float* const bv_p=sub_p+C*o_bv;
                float* const bq_p=sub_p+C*o_bq;
                float* const bgate_p=sub_p+C*o_bgate;
                float* const batt_p=sub_p+C*o_batt;
                mm_bh(n("attn_q.weight"),L.w_q,bqg_p);
                mm_bh(n("attn_k.weight"),L.w_k,bk_p);
                mm_bh(n("attn_v.weight"),L.w_v,bv_p);
                Q::attention_normalize_batch(bqg_p,bk_p,static_cast<const float*>(L.w_q_norm.d_data),static_cast<const float*>(L.w_k_norm.d_data),bq_p,bgate_p,bk_p,position,n_tokens,G.q_heads,G.kv_heads,admission.eps,admission.rope_base,stream.s);
                if(prof){Q::ck(cudaStreamSynchronize(stream.s));prof_attn_mm+=std::chrono::duration<double,std::milli>(Clock::now()-tp).count();tp=Clock::now();}
                kv->enqueue_batch(i/G.interval,position,n_tokens,bq_p,bk_p,bv_p,bgate_p,batt_p,scores.as());
                if(prof){Q::ck(cudaStreamSynchronize(stream.s));prof_attn_kv+=std::chrono::duration<double,std::milli>(Clock::now()-tp).count();tp=Clock::now();}
                mm_batch(n("attn_output.weight"),L.w_out_proj,batt_p,by_p,n_tokens);
                if(prof){Q::ck(cudaStreamSynchronize(stream.s));prof_attn_mm+=std::chrono::duration<double,std::milli>(Clock::now()-tp).count();}
            }
            if(prof)tp=Clock::now();
            Q::add(bx_p,by_p,size_t(n_tokens)*G.hidden,stream.s);
            Q::rms_batch(bx_p,static_cast<const float*>(L.post_attn_norm.d_data),bh_p,n_tokens,G.hidden,admission.eps,stream.s);
            bh_prepared=false;
            float* const bfg_p=sub_p;
            float* const bfu_p=sub_p+C*o_bfu;
            mm_bh(n("ffn_gate.weight"),L.w_ffn_gate,bfg_p);
            mm_bh(n("ffn_up.weight"),L.w_ffn_up,bfu_p);
            Q::swiglu(bfg_p,bfu_p,size_t(n_tokens)*G.ffn,stream.s);
            mm_batch(n("ffn_down.weight"),L.w_ffn_down,bfg_p,by_p,n_tokens);
            Q::add(bx_p,by_p,size_t(n_tokens)*G.hidden,stream.s);
            cvec.apply(bx_p,n_tokens,G.hidden,i,stream.s);
            Q::finite(bx_p,size_t(n_tokens)*G.hidden,error.as<int>(),stream.s);
            if(prof){Q::ck(cudaStreamSynchronize(stream.s));prof_ffn_mm+=std::chrono::duration<double,std::milli>(Clock::now()-tp).count();}
            fences.after(i,stop,check);
        }
        if(head){
            prefill_ws.reset_l2_window(stream.s);
            rms_h(bx_p+size_t(n_tokens-1)*size_t(G.hidden),"output_norm.weight");
            mm("output.weight",h,logits);
            Q::finite(logits.as(),G.vocab,error.as<int>(),stream.s);
        }
        int bad=0;Q::ck(cudaMemcpyAsync(&bad,error.data,4,cudaMemcpyDeviceToHost,stream.s));Q::ck(cudaStreamSynchronize(stream.s));
        if(prof)std::fprintf(stderr,"PROF_PREFILL n=%d gdn_mm=%.2f gdn_core=%.2f attn_mm=%.2f attn_kv=%.2f ffn=%.2f fence=%.2f\n",n_tokens,prof_gdn_mm,prof_gdn_core,prof_attn_mm,prof_attn_kv,prof_ffn_mm,prof_fence);
        Q::require(!bad,"non-finite activation/state/logit");fences.before(stop);position+=n_tokens;
    }
    void prefill_chunk_all_logits(const int32_t* tokens,int n_tokens,std::atomic<bool>& stop){
        run_prefill_chunk(tokens,n_tokens,stop,false);
        Q::rms_batch(bx_ptr(),f("output_norm.weight"),bh_ptr(),n_tokens,G.hidden,admission.eps,stream.s);
        mm_batch("output.weight",weights.at("output.weight").quant(),bh_ptr(),mtp_logits.as(),n_tokens);
        prefill_ws.reset_l2_window(stream.s);
        Q::finite(mtp_logits.as(),size_t(n_tokens)*G.vocab,error.as<int>(),stream.s);
        int bad=0;Q::ck(cudaMemcpyAsync(&bad,error.data,4,cudaMemcpyDeviceToHost,stream.s));Q::ck(cudaStreamSynchronize(stream.s));
        Q::require(!bad,"non-finite MTP verify logits");
    }
    std::vector<float> read_mtp_logits(int row){
        std::vector<float> v(G.vocab);
        Q::ck(cudaMemcpy(v.data(),mtp_logits.as()+size_t(row)*G.vocab,v.size()*4,cudaMemcpyDeviceToHost));
        return v;
    }
    std::vector<float> read_logits(){std::vector<float> v(G.vocab);Q::ck(cudaMemcpy(v.data(),logits.data,v.size()*4,cudaMemcpyDeviceToHost));return v;}
};
static int integer(const std::string& s,int lo,int hi){size_t n=0;long long v;try{v=std::stoll(s,&n);}catch(...){throw std::runtime_error("invalid integer");}Q::require(n==s.size()&&v>=lo&&v<=hi,"integer out of range");return int(v);}
static float real(const std::string& s,float lo,float hi){size_t n=0;float v;try{v=std::stof(s,&n);}catch(...){throw std::runtime_error("invalid real");}Q::require(n==s.size()&&std::isfinite(v)&&v>=lo&&v<=hi,"real out of range");return v;}
static std::vector<int32_t> ids(const std::string& s){Q::require(!s.empty()&&s.back()!=',',"empty token ID");std::istringstream in(s);std::string t;std::vector<int32_t> v;while(std::getline(in,t,','))v.push_back(integer(t,0,248319));return v;}
static void option(Sampling& s,const std::string& key,const std::string& v,int ctx){
    if(key=="temperature")s.temperature=real(v,0,100);else if(key=="top_p")s.top_p=real(v,0,1);else if(key=="min_p")s.min_p=real(v,0,1);
    else if(key=="top_k")s.top_k=integer(v,0,248320);else if(key=="penalty_repeat"||key=="repeat_penalty")s.repeat=real(v,0.0001f,100);
    else if(key=="penalty_last_n"||key=="repeat_last_n")s.last_n=integer(v,-1,ctx);else if(key=="penalty_freq")s.freq=real(v,-100,100);else if(key=="penalty_present")s.present=real(v,-100,100);
    else if(key=="seed"){size_t n=0;Q::require(!v.empty()&&v[0]!='-',"invalid seed");s.seed=std::stoull(v,&n);Q::require(n==v.size()&&s.seed<=UINT32_MAX,"invalid seed");}
    else throw std::runtime_error("unsupported GEN key: "+key);
}
static void run(Engine& e,const std::vector<int32_t>& prompt,int count,Sampling sampling,std::atomic<bool>& stop,
                Q38Protocol* protocol=nullptr,const std::vector<int32_t>& force={},const std::string& dump="",uint64_t exceptions=0){
    Q::require(!prompt.empty()&&prompt.size()<=size_t(e.ctx)&&size_t(count)<=size_t(e.ctx)-prompt.size(),"request exceeds explicit context capacity");
    Q::require(force.empty()||force.size()+1==size_t(count),"force requires one more logits row than forced tokens");
    if(e.orthography)e.orthography->reset(e.japanese,exceptions);
    e.reset();e.prefill_ws.reset_stats();
    auto start=Clock::now(),prefill=start;int produced=0;size_t read=0;const char* finish="length";
    int decode_steps=0;double decode_forward_ms=0.0,sampling_ms=0.0,reset_ms=0.0;
    std::ofstream out;if(!dump.empty()){out.open(dump,std::ios::binary|std::ios::trunc);Q::require(bool(out),"cannot open logits output");}
    std::mt19937_64 rng(sampling.seed);auto hist=strata::qwen35::penalty_history_start(prompt,false);
    try{
        int last_chunk_len=0;
        if(!e.conservative&&!e.fp32_prefill&&prompt.size()>=8){
            for(size_t off=0;off<prompt.size();){
                if(stop.load())throw Cancelled{};
                const int chunk=int(std::min<size_t>(size_t(e.prefill_chunk),prompt.size()-off));
                e.run_prefill_chunk(prompt.data()+off,chunk,stop,off+size_t(chunk)==prompt.size());
                last_chunk_len=chunk;
                off+=size_t(chunk);read=off;
                if(protocol){std::printf("PP %zu %zu\n",read,prompt.size());std::fflush(stdout);}
            }
            e.prefill_ws.reset_l2_window(e.stream.s);
        }else{
            const bool saved_fp32_decode=e.fp32_decode;
            if(e.fp32_prefill)e.fp32_decode=true;
            for(size_t i=0;i<prompt.size();++i){if(stop.load())throw Cancelled{};e.step(prompt[i],stop,i+1==prompt.size());++read;if(protocol){std::printf("PP %zu %zu\n",read,prompt.size());std::fflush(stdout);}}
            e.fp32_decode=saved_fp32_decode;
        }
        prefill=Clock::now();
        auto emit=[&](int token){
            if(e.orthography)e.orthography->accept(token);
            ++produced;hist.push_back(token);
            if(protocol)std::printf("T %d\n",token);else std::printf("%s%d",produced>1?",":"",token);
            std::fflush(stdout);
            if(force.empty()&&(token==e.admission.eos||token==248044)){finish="eos";return false;}
            return true;
        };
        if(e.mtp_ready&&e.spec_draft_n_max>0&&force.empty()&&!out.is_open()&&count>0){
            auto ts0=Clock::now();auto lg0=e.read_logits();
            int tok0=Q::sample(lg0,sampling,hist,rng,e.orthography.get());
            sampling_ms+=std::chrono::duration<double,std::milli>(Clock::now()-ts0).count();
            if(emit(tok0)&&produced<count){
                const float* h_last=(last_chunk_len>0)
                    ?(e.bx_ptr()+size_t(last_chunk_len-1)*size_t(e.G.hidden))
                    :e.x.as();
                Q::ck(cudaMemcpyAsync(e.mtp_hidden.data,h_last,size_t(e.G.hidden)*4,cudaMemcpyDeviceToDevice,e.stream.s));
                int mtp_pos=0;
                while(produced<count&&!stop.load()){
                    const int k=std::min(e.spec_draft_n_max,count-produced);
                    if(k<=0||e.position+k+1>e.ctx||mtp_pos+k>e.ctx){
                        auto td0=Clock::now();e.step(tok0,stop,true);decode_forward_ms+=std::chrono::duration<double,std::milli>(Clock::now()-td0).count();++decode_steps;
                        auto ts1=Clock::now();auto lg=e.read_logits();tok0=Q::sample(lg,sampling,hist,rng,e.orthography.get());sampling_ms+=std::chrono::duration<double,std::milli>(Clock::now()-ts1).count();
                        if(!emit(tok0))break;continue;
                    }
                    std::vector<int32_t> batch(1+k);
                    batch[0]=tok0;
                    const int mtp_pos_before=mtp_pos;
                    auto td_draft=Clock::now();
                    for(int j=0;j<k;++j){batch[1+j]=e.mtp_draft_step(batch[j],mtp_pos+j);}
                    const int pos_before=e.position;
                    e.snapshot_gdn();
                    e.prefill_chunk_all_logits(batch.data(),1+k,stop);
                    decode_forward_ms+=std::chrono::duration<double,std::milli>(Clock::now()-td_draft).count();
                    decode_steps+=1+k;
                    int accepted=0;
                    int next_tok=0;
                    bool ended=false;
                    for(int j=0;j<k;++j){
                        auto ts1=Clock::now();auto row_lg=e.read_mtp_logits(j);
                        int target=Q::sample(row_lg,sampling,hist,rng,e.orthography.get());
                        sampling_ms+=std::chrono::duration<double,std::milli>(Clock::now()-ts1).count();
                        if(!emit(target)){ended=true;break;}
                        if(target==batch[1+j]&&produced<count){
                            ++accepted;
                        }else{
                            next_tok=target;
                            break;
                        }
                    }
                    if(ended)break;
                    if(accepted==k&&produced<count){
                        auto ts1=Clock::now();auto row_lg=e.read_mtp_logits(k);
                        next_tok=Q::sample(row_lg,sampling,hist,rng,e.orthography.get());
                        sampling_ms+=std::chrono::duration<double,std::milli>(Clock::now()-ts1).count();
                        if(!emit(next_tok))break;
                        mtp_pos+=k;
                        Q::ck(cudaMemcpyAsync(e.mtp_hidden.data,e.bx_ptr()+size_t(k)*size_t(e.G.hidden),size_t(e.G.hidden)*4,cudaMemcpyDeviceToDevice,e.stream.s));
                    }else if(accepted<k){
                        auto td_rb=Clock::now();
                        e.restore_gdn();
                        e.kv->rewind(pos_before);
                        e.mtp_kv->rewind(mtp_pos_before+accepted);
                        e.position=pos_before;
                        for(int j=0;j<=accepted;++j){e.step(batch[j],stop,j==accepted);++decode_steps;}
                        decode_forward_ms+=std::chrono::duration<double,std::milli>(Clock::now()-td_rb).count();
                        Q::ck(cudaMemcpyAsync(e.mtp_hidden.data,e.x.data,size_t(e.G.hidden)*4,cudaMemcpyDeviceToDevice,e.stream.s));
                        mtp_pos=mtp_pos_before+accepted;
                    }
                    tok0=next_tok;
                }
            }
        }else{
            for(int i=0;i<count;++i){if(stop.load())throw Cancelled{};
                auto ts0=Clock::now();
                auto lg=e.read_logits();if(out.is_open()){out.write(reinterpret_cast<const char*>(lg.data()),lg.size()*4);Q::require(bool(out),"logits write failed");}
                int token=size_t(i)<force.size()?force[i]:Q::sample(lg,sampling,hist,rng,e.orthography.get());
                sampling_ms+=std::chrono::duration<double,std::milli>(Clock::now()-ts0).count();
                if(e.orthography)e.orthography->accept(token);++produced;hist.push_back(token);
                if(protocol)std::printf("T %d\n",token);else std::printf("%s%d",i?",":"",token);std::fflush(stdout);
                if(force.empty()&&(token==e.admission.eos||token==248044)){finish="eos";break;}
                if(i+1<count){auto td0=Clock::now();e.step(token,stop,true);decode_forward_ms+=std::chrono::duration<double,std::milli>(Clock::now()-td0).count();++decode_steps;}
            }
        }
    }catch(const Cancelled&){finish="cancel";if(read<prompt.size())prefill=Clock::now();}
    catch(...){if(e.orthography)e.orthography->reset(false);e.reset();throw;}
    if(e.orthography)e.orthography->reset(false);
    auto gen_end=Clock::now();
    e.reset();auto end=Clock::now();
    reset_ms=std::chrono::duration<double,std::milli>(end-gen_end).count();
    double pp=std::chrono::duration<double,std::milli>(prefill-start).count(),tg=std::chrono::duration<double,std::milli>(gen_end-prefill).count(),e2e_ms=std::chrono::duration<double,std::milli>(end-start).count();
    const auto gs=e.prefill_ws.stats();
    std::fprintf(stderr,"PREFILL_CANDIDATE fused_wmma_calls=%llu\n",(unsigned long long)gs.fused_wmma_calls);
    auto publish=[&](bool cancelled){if(cancelled)finish="cancel";
        if(protocol){
            std::printf("TIMING prefill_forward_ms=%.3f decode_forward_ms=%.3f decode_steps=%d sampling_ms=%.3f reset_ms=%.3f e2e_ms=%.3f\n",pp,decode_forward_ms,decode_steps,sampling_ms,reset_ms,e2e_ms);
            std::printf("GEMM_STATS fp16=%llu fp8=%llu fp8_fb_shape=%llu fp8_fb_algo=%llu mv_fb=%llu\n",(unsigned long long)gs.fp16_gemm_calls,(unsigned long long)gs.fp8_gemm_calls,(unsigned long long)gs.fp8_fallback_shape,(unsigned long long)gs.fp8_fallback_no_algo,(unsigned long long)gs.matvec_fallback_calls);
            // Official Strata 0.2.16 --serve completion line.  The resident
            // server parses ENGINE_REQUEST from terminal/serve/engines.py;
            // without this exact shape it cannot attribute a finished turn.
            std::printf("prompt %zu tokens = 0 reused + %zu read in %.3f ms (%.2f tok/s), %d generated in %.3f ms (%.2f tok/s)\n",
                        prompt.size(), read, pp, pp>0.0? double(prompt.size())*1000.0/pp : 0.0,
                        produced, tg, tg>0.0? double(produced)*1000.0/tg : 0.0);
            auto line=q38_format_done(produced,prompt.size(),pp,tg,finish,read);std::fwrite(line.data(),1,line.size(),stdout);
        }else{
            std::puts("");
            std::fprintf(stderr,"TIMING prefill_forward_ms=%.3f decode_forward_ms=%.3f decode_steps=%d sampling_ms=%.3f reset_ms=%.3f e2e_ms=%.3f\n",pp,decode_forward_ms,decode_steps,sampling_ms,reset_ms,e2e_ms);
            std::fprintf(stderr,"GEMM_STATS fp16=%llu fp8=%llu fp8_fb_shape=%llu fp8_fb_algo=%llu mv_fb=%llu\n",(unsigned long long)gs.fp16_gemm_calls,(unsigned long long)gs.fp8_gemm_calls,(unsigned long long)gs.fp8_fallback_shape,(unsigned long long)gs.fp8_fallback_no_algo,(unsigned long long)gs.matvec_fallback_calls);
        }
        std::fflush(stdout);
    };
    if(protocol)protocol->finish(publish);else publish(stop.load());
}
static void serve(Engine& e){
    Q38Protocol state;
    Q38InputReader reader(GetStdHandle(STD_INPUT_HANDLE),state);
    // All INFO values are fixed ASCII tokens or numbers, never file metadata.
    const auto& c=e.kv->config();
    std::printf("INFO output_language=%s orthography=%s session_reset=every_request private_kv=unsupported\n",e.japanese?"ja":"model-default",e.japanese?Q::orthography_version:"off");
    std::printf("INFO backend=strata_native_cuda engine=qwen38_dense weight_runtime=strata cuda_arch=120 context=%d sampling=strata_cpu weights=original_gpu embedding=host_mmap_row_upload kv=%s kv_k=%s kv_v=%s kv_requested=%s kv_selection_reason=%s kv_required_with_reserve=%llu kv_available_at_admission=%llu kv_reserve=%llu spin8_view=%s spin8_profile_sha256=%s spin8_matrix_sha256=%s recurrent=f32 weight_bytes=%llu cache_bytes=%llu kv_payload_bytes=%llu kv_scratch_bytes=%llu kv_coefficient_bytes=%llu prefix=0 graph=0 batch=1 drafting=off\nREADY %d stop\n",e.ctx,c.name(),c.key_format(),c.value_format(),e.kv_selection.requested.c_str(),e.kv_selection.reason.c_str(),(unsigned long long)e.kv_selection.required,(unsigned long long)e.kv_selection.available,(unsigned long long)e.kv_selection.reserve,c.view_name(),c.profile_sha.c_str(),c.matrix_sha.c_str(),(unsigned long long)e.admission.gpu_weights,(unsigned long long)e.cache_bytes,(unsigned long long)e.kv->persistent_bytes(),(unsigned long long)e.kv->scratch_bytes(),(unsigned long long)e.kv->coefficient_bytes(),e.ctx);std::fflush(stdout);
    Q38Protocol::Event ev{};
    while(state.next(ev)){
        if(ev.kind!=Q38Protocol::Kind::command){state.finish_reply([&]{std::puts(ev.kind==Q38Protocol::Kind::busy?"ERR single session is busy":"ERR input queue full");std::fflush(stdout);});continue;}
        if(ev.line=="QUIT")break;bool generation=ev.line.rfind("GEN ",0)==0;bool fatal=false;
        try{std::istringstream in(ev.line);std::string cmd,num,part;in>>cmd>>num;Q::require(cmd=="GEN"&&!num.empty(),"expected GEN max_new token_ids; unsupported feature");int count=integer(num,0,e.ctx);Sampling s;
            uint64_t exceptions=0;bool seen_exceptions=false;
            while(in>>part&&part.find('=')!=std::string::npos){size_t eq=part.find('=');if(part.substr(0,eq)=="ja_allow"){Q::require(e.japanese&&!seen_exceptions,"ja_allow requires Japanese profile and must occur once");exceptions=Q::language_exceptions(part.substr(eq+1));seen_exceptions=true;}else option(s,part.substr(0,eq),part.substr(eq+1),e.ctx);part.clear();}
            Q::require(!part.empty(),"missing token IDs");std::string extra;Q::require(!(in>>extra),"extra GEN argument");
            if(s.last_n==0&&(s.repeat!=1||s.freq!=0||s.present!=0))s.last_n=std::min(64,e.ctx);
            run(e,ids(part),count,s,state.stop_flag(),&state,{},"",exceptions);
        }catch(const std::exception& ex){try{e.reset();}catch(...){fatal=true;}if(dynamic_cast<const Q::ResourceError*>(&ex))fatal=true;
            auto reply=[&]{std::string msg=ex.what();for(char& c:msg)if(c=='\n'||c=='\r')c=' ';std::printf("ERR %s\n",msg.c_str());std::fflush(stdout);};
            if(generation)state.finish([&](bool){reply();});else state.finish_reply(reply);
        }
        if(fatal)break;
    }
    reader.stop_and_join();
}
namespace strata { namespace qwen38 {
int run(int argc,char** argv){try{
    std::string model,tokens,forced,dump,kv_mode="auto",spin_view,spin_profile,language="model-default",cvec_path;
    float cvec_scale=1.f;int ctx=4096,count=32,draft_n_max=0,cvec_first=1,cvec_last=-1,cvec_mode=0;
    bool resident=false,inspection=false,conservative=false;Sampling sampling;
    for(int i=1;i<argc;++i){std::string a=argv[i];auto value=[&](){Q::require(i+1<argc,"missing option value");return std::string(argv[++i]);};
        if(a=="--help"||a=="-h"){std::puts("strata-q38 own-native CUDA SM120\n--model FILE --ctx-size N (4096) --serve\n--tokens IDS --n N [--force IDS] [--dump-full-logits FILE]\n--kv auto|f16|k8v4|spin8_k8v4|f16_turbo2|f16_turbo3|f16_turbo4 (auto default)\nAuto order: f16, f16_turbo4, spin8_k8v4; context and weights unchanged.\n--spin8-view vector|plus|minus --spin8-profile FILE (compiled reference default)\n--output-language ja|model-default (standalone default: model-default)\n--spec-draft-n-max N (0..3, MTP draft cap when declared MTP tensors are present)\n--control-vector-scaled FILE:SCALE --control-vector-layer-range A B --cvec-mode project|add --cvec-dir per-layer\n--conservative-sync (diagnostic reference execution)\n--inspect (CPU only) --version");return 0;}
        if(a=="--version"){std::printf("strata-q38 own-native 0.6.3 CUDA13 SM120 cudart=%d cublas=%d\n",CUDART_VERSION,Q::PrefillGemmWorkspace::cublas_version());return 0;}
        if(a=="--model")model=value();else if(a=="--ctx-size")ctx=integer(value(),1,262144);else if(a=="--tokens")tokens=value();else if(a=="--n")count=integer(value(),0,262144);
        else if(a=="--force")forced=value();else if(a=="--dump-full-logits")dump=value();else if(a=="--serve")resident=true;else if(a=="--inspect")inspection=true;
        else if(a=="--kv")kv_mode=value();else if(a=="--spin8-view")spin_view=value();else if(a=="--spin8-profile")spin_profile=value();
        else if(a=="--conservative-sync")conservative=true;else if(a=="--spec-draft-n-max")draft_n_max=integer(value(),0,3);
        else if(a=="--output-language")language=value();
        else if(a=="--control-vector-scaled"){
            const std::string spec=value();const auto colon=spec.rfind(':');
            Q::require(colon!=std::string::npos,"--control-vector-scaled expects FILE:SCALE");
            cvec_path=spec.substr(0,colon);cvec_scale=float(std::atof(spec.c_str()+colon+1));
            Q::require(!cvec_path.empty()&&std::isfinite(cvec_scale)&&cvec_scale!=0.f,"invalid control vector scale");
        }
        else if(a=="--control-vector-layer-range"){cvec_first=integer(value(),1,262144);cvec_last=integer(value(),1,262144);Q::require(cvec_last>=cvec_first,"invalid control vector layer range");}
        else if(a=="--cvec-mode"){const std::string m=value();Q::require(m=="project"||m=="add","--cvec-mode expects project|add");cvec_mode=m=="add"?1:0;}
        else if(a=="--cvec-dir"){const std::string d=value();Q::require(d=="per-layer"||d.rfind("single:",0)==0,"--cvec-dir expects per-layer|single:L");}
        else if(a=="--seed")option(sampling,"seed",value(),ctx);else if(a=="--temperature")option(sampling,"temperature",value(),ctx);else throw std::runtime_error("unsupported option: "+a);
    }
    Q::require(!model.empty(),"--model is required");
    bool japanese=Q::japanese_profile(language);Q::require(!japanese||(forced.empty()&&dump.empty()),"forced/logit diagnostics require --output-language model-default (constraints off)");
    const auto kv_config=Q::KvConfig::admit(kv_mode,spin_view,spin_profile);
    if(inspection){strata::GgufFile f(model,Q::retained_metadata(),true);auto a=Q::inspect(f);const auto& g=a.geometry;
        std::printf("GGUF own-native qwen35 dense profile=%s layers=%d tensors=%zu hidden=%d ffn=%d vocab=%d q_heads=%d kv_heads=%d nextn=%d mtp_tensors=%zu gpu_weight_bytes=%llu host_embedding_bytes=%llu folded=%zu inverse_embedding=%d grouped_gdn=%d\n",
            g.trunk_layers==64?"q38-27b":"q38-9b",g.trunk_layers,f.tensors().size(),g.hidden,g.ffn,g.vocab,g.q_heads,g.kv_heads,a.nextn,a.mtp_required.size(),
            (unsigned long long)a.gpu_weights,(unsigned long long)a.embedding_bytes,a.folded.size(),a.inverse_embedding,a.grouped_gdn);return 0;}
    Q::require(resident!=!tokens.empty(),"choose exactly one of --serve or --tokens");Q::require(!resident||(forced.empty()&&dump.empty()),"force/logits are standalone options");
    auto prompt=tokens.empty()?std::vector<int32_t>{}:ids(tokens);auto force=forced.empty()?std::vector<int32_t>{}:ids(forced);if(!force.empty())count=int(force.size()+1);
    if(!forced.empty()||!dump.empty())std::fprintf(stderr,"diagnostic output_language=model-default orthography=off\n");
    Q::host_resources();Engine e(model,ctx,kv_config,conservative,japanese,draft_n_max,cvec_path,cvec_scale,cvec_first,cvec_last,cvec_mode);if(resident)serve(e);else{std::atomic<bool> stop{false};run(e,prompt,count,sampling,stop,nullptr,force,dump);}return 0;
}catch(const std::exception& e){std::fprintf(stderr,"strata-q38: %s\n",e.what());return 1;}}
}  // namespace qwen38
}  // namespace strata

#ifdef STRATA_Q38_STANDALONE
int main(int argc,char** argv){ return strata::qwen38::run(argc,argv); }
#endif
