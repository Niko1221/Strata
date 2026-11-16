// Strata-owned dense Qwen35/Q38 inference. No external inference runtime.
#include "strata/qwen38/admission.hpp"
#include "strata/kernels/fused_gdn.hpp"
#include "strata/qwen35/sampling.hpp"
#include "ops.cuh"
#include "resources.hpp"
#include "protocol.hpp"
#include "protocol_output.hpp"
#include "kv.hpp"
#include "upload_reader.hpp"
#include "session_state.hpp"
#include "weight_arena.hpp"
#include "layer_fence.hpp"
#include "startup.cuh"
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
static void gpu_resources(size_t need=0,const std::string& stage="execution",const std::string& tensor="none"){
    size_t free=0,total=0;Q::ck(cudaMemGetInfo(&free,&total));
    if(free<Q::GPU_RESERVE||need>free-Q::GPU_RESERVE)
        throw Q::ResourceError("CUDA reserve: stage="+stage+" tensor="+tensor+" free="+std::to_string(free)+" needed="+std::to_string(need)+" reserve="+std::to_string(Q::GPU_RESERVE));
}
struct Stream {cudaStream_t s=nullptr;Stream(){Q::ck(cudaStreamCreate(&s));}~Stream(){if(s)cudaStreamDestroy(s);}};
struct Weight {Q::WeightSpan b;int type=0,n=0,rows=0;};
struct Engine {
    strata::GgufFile file;Q::Admission admission;int ctx;Stream stream;
    std::unique_ptr<Q::WeightArena> weight_arena;
    std::map<std::string,Weight> weights;std::vector<Q::RecurrentState> states;
    std::unique_ptr<Q::KvCache> kv;
    Q::KvSelection kv_selection;
    std::map<int,Q::Buffer> signs;Q::Buffer rotated;
    Q::Buffer x,h,y,qkv,z,al,be,hc,gdn,qg,k,v,q,gate,att,fg,fu,scores,logits,embrow,error;
    uint64_t cache_bytes=0,scratch_bytes=0;size_t emb_stride=0;const strata::TensorInfo* embedding=nullptr;
    int position=0;
    bool conservative=false;
    Engine(const std::string& path,int context,const Q::KvConfig& config,bool reference=false):file(path,Q::retained_metadata(),true),admission(Q::inspect(file)),ctx(context),conservative(reference){
        Q::host_resources();Q::ck(cudaSetDevice(0));cudaDeviceProp prop{};Q::ck(cudaGetDeviceProperties(&prop,0));
        Q::require(prop.major==12&&prop.minor==0,"this build requires CUDA SM120 device 0");
        size_t before_preload=0,after_preload=0,preload_total=0;
        Q::ck(cudaMemGetInfo(&before_preload,&preload_total));Q::warmup_runtime(stream.s);
        Q::ck(cudaMemGetInfo(&after_preload,&preload_total));
        std::fprintf(stderr,"kernel_preload_free_before=%zu free_after=%zu\n",before_preload,after_preload);
        const uint64_t recurrent_bytes=48ull*(10240*3+48*128*128)*4;
        embedding=file.find("token_embd.weight");emb_stride=Q::payload(*embedding)/248320;
        // Exact owned activation/score/logit buffers, weight-transform scratch,
        // embedding row and retained signs. Auxiliary conversion peak is separate.
        scratch_bytes=(5120ull*3+10240*2+6144*5+48*2+12288+1024*2+17408*2+24ull*ctx+248320)*4+4+17408*4+emb_stride;
        for(const auto& entry:admission.signs)scratch_bytes+=entry.second.size()*4;
        // Include measured 2 MiB backing slack for large cudaMalloc buffers.
        // The retained 16 MiB safety allowance also covers small-buffer pools.
        auto large_slack=[](uint64_t bytes)->uint64_t {constexpr uint64_t granule=2*1024*1024;return bytes>=1024*1024?(granule-bytes%granule)%granule:0;};
        // Retain the small-pool allowance and another 128 MiB for launch-size
        // dependent residency beyond the tiny pre-admission startup warmup.
        constexpr uint64_t first_use_allowance=128ull*1024*1024;
        uint64_t allocator_allowance=16*1024*1024+first_use_allowance+48*large_slack(48*128*128*4)+large_slack(24ull*ctx*4);
        Q::WeightLayout layout;
        uint64_t conversion_peak=0;for(const auto& name:admission.required){const auto& t=*file.find(name);if(name=="token_embd.weight")continue;
            layout.append(Q::convert_auxiliary(t)?Q::WeightLayout::f32_bytes(t.elements()):Q::payload(t));
            if(Q::convert_auxiliary(t)&&t.type!=0)conversion_peak=std::max(conversion_peak,Q::payload(t));}
        Q::require(layout.payload()==admission.gpu_weights,"weight arena/admission payload mismatch");
        allocator_allowance+=large_slack(layout.bytes());
        size_t available=0,total=0;Q::ck(cudaMemGetInfo(&available,&total));
        kv_selection=Q::select_kv(config,ctx,16,{layout.bytes(),recurrent_bytes,scratch_bytes,conversion_peak,allocator_allowance},available,Q::GPU_RESERVE);
        const auto& effective=kv_selection.config;cache_bytes=recurrent_bytes+kv_selection.payload;
        std::fprintf(stderr,"kv_requested=%s kv_effective=%s kv_k=%s kv_v=%s reason=%s required_with_reserve=%llu available=%llu reserve=%llu\n",kv_selection.requested.c_str(),effective.name(),effective.key_format(),effective.value_format(),kv_selection.reason.c_str(),(unsigned long long)kv_selection.required,(unsigned long long)kv_selection.available,(unsigned long long)kv_selection.reserve);
        std::fprintf(stderr,"kv_payload_bytes=%llu kv_allocator_slack=%llu nonkv_allocator_allowance=%llu recurrent_bytes=%llu scratch_bytes=%llu auxiliary_peak_bytes=%llu\n",(unsigned long long)kv_selection.payload,(unsigned long long)kv_selection.allocator_slack,(unsigned long long)allocator_allowance,(unsigned long long)recurrent_bytes,(unsigned long long)scratch_bytes,(unsigned long long)conversion_peak);
        gpu_resources(kv_selection.required-Q::GPU_RESERVE,"admission");
        gpu_resources(layout.bytes(),"weight_arena");
        size_t arena_before=0,arena_after=0;Q::ck(cudaMemGetInfo(&arena_before,&total));
        weight_arena=std::make_unique<Q::WeightArena>(layout);Q::ck(cudaMemGetInfo(&arena_after,&total));
        std::fprintf(stderr,"weight_arena_payload=%zu weight_arena_bytes=%zu weight_arena_padding=%zu weight_arena_allocations=1 free_before=%zu free_after=%zu first_use_allowance=%llu\n",layout.payload(),layout.bytes(),layout.bytes()-layout.payload(),arena_before,arena_after,(unsigned long long)first_use_allowance);
        kv=std::make_unique<Q::KvCache>(effective,ctx,16,stream.s,conservative);
        rotated.allocate(17408*4);
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
        states.resize(64);
        for(int i=0;i<64;++i)if((i+1)%4){states[i].conv.allocate(10240*3*4);states[i].gdn.allocate(48*128*128*4);}
        auto a=[](Q::Buffer& b,size_t n){b.allocate(n*4);};
        a(x,5120);a(h,5120);a(y,5120);a(qkv,10240);a(hc,10240);a(z,6144);a(gdn,6144);a(al,48);a(be,48);
        a(qg,12288);a(k,1024);a(v,1024);a(q,6144);a(gate,6144);a(att,6144);a(fg,17408);a(fu,17408);
        a(scores,size_t(24)*ctx);a(logits,248320);a(error,1);
        embrow.allocate(emb_stride);
        reset();gpu_resources();Q::host_resources();
        std::fprintf(stderr,"native_execution=%s layer_fence_interval=%d\n",conservative?"conservative":"grouped",conservative?1:8);
    }
    const float* f(const std::string& name){return weights.at(name).b.as();}
    void mm(const std::string& name,const Q::Buffer& in,Q::Buffer& out){const auto& w=weights.at(name);const float* input=in.as();
        if(admission.folded.count(name)){Q::hadamard(input,signs.at(w.n).as(),rotated.as(),w.n,false,admission.grouped_gdn&&name.find(".ssm_out.")!=std::string::npos,stream.s);input=rotated.as();}
        Q::matvec(w.type,w.b.data,input,out.as(),w.n,w.rows,stream.s);}
    void reset(){
        Q::reset_session(states,*kv,error,position,stream.s);
    }
    void step(int token,std::atomic<bool>& stop,bool head){
        Q::require(token>=0&&token<248320&&position<ctx,"token or context out of range");
        Q::host_resources();gpu_resources();error.zero(stream.s);
        Q::ck(cudaMemcpyAsync(embrow.data,file.tensor_data(*embedding)+size_t(token)*emb_stride,emb_stride,cudaMemcpyHostToDevice,stream.s));
        Q::decode(embedding->type,embrow.data,x.as(),5120,stream.s);
        if(admission.inverse_embedding){Q::hadamard(x.as(),signs.at(5120).as(),rotated.as(),5120,true,false,stream.s);Q::ck(cudaMemcpyAsync(x.data,rotated.data,5120*4,cudaMemcpyDeviceToDevice,stream.s));}
        const Q::LayerFence fences(conservative?1:8);
        auto check=[&]{
            int bad=0;Q::ck(cudaMemcpyAsync(&bad,error.data,4,cudaMemcpyDeviceToHost,stream.s));
            kv->check();Q::require(!bad,"non-finite activation/state/logit");Q::host_resources();gpu_resources();
        };
        for(int i=0;i<64;++i){
            fences.before(stop);
            auto n=[i](const char* r){return Q::layer_name(i,r);};auto& st=states[i];
            Q::rms(x.as(),f(n("attn_norm.weight")),h.as(),5120,admission.eps,stream.s);
            if((i+1)%4){
                mm(n("attn_qkv.weight"),h,qkv);mm(n("attn_gate.weight"),h,z);mm(n("ssm_alpha.weight"),h,al);mm(n("ssm_beta.weight"),h,be);
                Q::gdn_gates(al.as(),be.as(),f(n("ssm_dt.bias")),f(n("ssm_a")),48,stream.s);
                K::fused_gdn_conv_l2(st.conv.as(),qkv.as(),f(n("ssm_conv1d.weight")),hc.as(),10240,32,admission.eps,stream.s);
                K::fused_gdn_step_norm_silu(st.gdn.as(),hc.as(),hc.as()+2048,hc.as()+4096,al.as(),be.as(),z.as(),f(n("ssm_norm.weight")),admission.eps,gdn.as(),16,48,stream.s);
                Q::finite(st.gdn.as(),48*128*128,error.as<int>(),stream.s);mm(n("ssm_out.weight"),gdn,y);
            }else{
                mm(n("attn_q.weight"),h,qg);mm(n("attn_k.weight"),h,k);mm(n("attn_v.weight"),h,v);
                Q::attention_normalize(qg.as(),k.as(),f(n("attn_q_norm.weight")),f(n("attn_k_norm.weight")),q.as(),gate.as(),k.as(),position,admission.eps,admission.rope_base,stream.s);
                kv->enqueue(i/4,position,q.as(),k.as(),v.as(),gate.as(),att.as(),scores.as());
                mm(n("attn_output.weight"),att,y);
            }
            Q::add(x.as(),y.as(),5120,stream.s);Q::rms(x.as(),f(n("post_attention_norm.weight")),h.as(),5120,admission.eps,stream.s);
            mm(n("ffn_gate.weight"),h,fg);mm(n("ffn_up.weight"),h,fu);Q::swiglu(fg.as(),fu.as(),17408,stream.s);mm(n("ffn_down.weight"),fg,y);
            Q::add(x.as(),y.as(),5120,stream.s);Q::finite(x.as(),5120,error.as<int>(),stream.s);
            // Bounded synchronization makes STOP/resource checks effective during a long token.
            fences.after(i,stop,check);
        }
        if(head){Q::rms(x.as(),f("output_norm.weight"),h.as(),5120,admission.eps,stream.s);mm("output.weight",h,logits);Q::finite(logits.as(),248320,error.as<int>(),stream.s);}
        int bad=0;Q::ck(cudaMemcpyAsync(&bad,error.data,4,cudaMemcpyDeviceToHost,stream.s));Q::ck(cudaStreamSynchronize(stream.s));
        Q::require(!bad,"non-finite activation/state/logit");fences.before(stop);++position;
    }
    std::vector<float> read_logits(){std::vector<float> v(248320);Q::ck(cudaMemcpy(v.data(),logits.data,v.size()*4,cudaMemcpyDeviceToHost));return v;}
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
static int sample(std::vector<float> lg,const Sampling& s,const std::vector<int32_t>& hist,std::mt19937_64& rng){
    strata::qwen35::apply_penalties(lg,s,hist);if(s.temperature<=0)return int(std::max_element(lg.begin(),lg.end())-lg.begin());
    // top_k=0 retains the entire vocabulary; never silently replace it by 64.
    std::vector<int> ix(lg.size());std::iota(ix.begin(),ix.end(),0);size_t count=s.top_k?std::min(size_t(s.top_k),ix.size()):ix.size();
    std::partial_sort(ix.begin(),ix.begin()+count,ix.end(),[&](int a,int b){return lg[a]>lg[b]||(lg[a]==lg[b]&&a<b);});ix.resize(count);
    std::vector<double> p(count);double total=0;for(size_t i=0;i<count;++i)total+=(p[i]=std::exp((double(lg[ix[i]])-lg[ix[0]])/s.temperature));
    size_t keep=count;double c=0;for(size_t i=0;i<count;++i){if(p[i]<s.min_p*p[0]){keep=i;break;}c+=p[i]/total;if(c>=s.top_p){keep=i+1;break;}}
    keep=std::max(size_t(1),keep);double mass=std::accumulate(p.begin(),p.begin()+keep,0.0);double r=std::uniform_real_distribution<double>(0,mass)(rng);
    for(size_t i=0;i<keep;++i)if((r-=p[i])<=0)return ix[i];return ix[keep-1];
}
static void run(Engine& e,const std::vector<int32_t>& prompt,int count,Sampling sampling,std::atomic<bool>& stop,
                Q38Protocol* protocol=nullptr,const std::vector<int32_t>& force={},const std::string& dump=""){
    Q::require(!prompt.empty()&&prompt.size()<=size_t(e.ctx)&&size_t(count)<=size_t(e.ctx)-prompt.size(),"request exceeds explicit context capacity");
    Q::require(force.empty()||force.size()+1==size_t(count),"force requires one more logits row than forced tokens");
    e.reset();auto start=Clock::now(),prefill=start;int produced=0;size_t read=0;const char* finish="length";
    std::ofstream out;if(!dump.empty()){out.open(dump,std::ios::binary|std::ios::trunc);Q::require(bool(out),"cannot open logits output");}
    std::mt19937_64 rng(sampling.seed);auto hist=strata::qwen35::penalty_history_start(prompt,false);
    try{
        for(size_t i=0;i<prompt.size();++i){if(stop.load())throw Cancelled{};e.step(prompt[i],stop,i+1==prompt.size());++read;if(protocol){std::printf("PP %zu %zu\n",read,prompt.size());std::fflush(stdout);}}
        prefill=Clock::now();
        for(int i=0;i<count;++i){if(stop.load())throw Cancelled{};auto lg=e.read_logits();if(out.is_open()){out.write(reinterpret_cast<const char*>(lg.data()),lg.size()*4);Q::require(bool(out),"logits write failed");}
            int token=size_t(i)<force.size()?force[i]:sample(lg,sampling,hist,rng);++produced;hist.push_back(token);
            if(protocol)std::printf("T %d\n",token);else std::printf("%s%d",i?",":"",token);std::fflush(stdout);
            if(force.empty()&&(token==e.admission.eos||token==248044)){finish="eos";break;}
            if(i+1<count)e.step(token,stop,true);
        }
    }catch(const Cancelled&){finish="cancel";if(read<prompt.size())prefill=Clock::now();}
    catch(...){e.reset();throw;}
    e.reset();auto end=Clock::now();double pp=std::chrono::duration<double,std::milli>(prefill-start).count(),tg=std::chrono::duration<double,std::milli>(end-prefill).count();
    auto publish=[&](bool cancelled){if(cancelled)finish="cancel";if(protocol){auto line=q38_format_done(produced,prompt.size(),pp,tg,finish,read);std::fwrite(line.data(),1,line.size(),stdout);}else std::puts("");std::fflush(stdout);};
    if(protocol)protocol->finish(publish);else publish(stop.load());
}
static void serve(Engine& e){
    Q38Protocol state;
    std::thread reader([&]{std::string line;bool overflow=false;char c;while(std::cin.get(c)){
        if(c=='\n'){if(!overflow){if(!line.empty()&&line.back()=='\r')line.pop_back();state.submit(line);}else state.submit("INVALID_LINE_TOO_LONG");line.clear();overflow=false;}
        else if(!overflow){if(line.size()>=4*1024*1024){line.clear();overflow=true;}else line+=c;}
    }state.end_input();});
    // All INFO values are fixed ASCII tokens or numbers, never file metadata.
    const auto& c=e.kv->config();
    std::printf("INFO backend=strata_native_cuda engine=qwen38_dense weight_runtime=strata cuda_arch=120 context=%d sampling=strata_cpu weights=original_gpu embedding=host_mmap_row_upload kv=%s kv_k=%s kv_v=%s kv_requested=%s kv_selection_reason=%s kv_required_with_reserve=%llu kv_available_at_admission=%llu kv_reserve=%llu spin8_view=%s spin8_profile_sha256=%s spin8_matrix_sha256=%s recurrent=f32 weight_bytes=%llu cache_bytes=%llu kv_payload_bytes=%llu kv_scratch_bytes=%llu kv_coefficient_bytes=%llu prefix=0 graph=0 batch=1 drafting=off\nREADY %d stop\n",e.ctx,c.name(),c.key_format(),c.value_format(),e.kv_selection.requested.c_str(),e.kv_selection.reason.c_str(),(unsigned long long)e.kv_selection.required,(unsigned long long)e.kv_selection.available,(unsigned long long)e.kv_selection.reserve,c.view_name(),c.profile_sha.c_str(),c.matrix_sha.c_str(),(unsigned long long)e.admission.gpu_weights,(unsigned long long)e.cache_bytes,(unsigned long long)e.kv->persistent_bytes(),(unsigned long long)e.kv->scratch_bytes(),(unsigned long long)e.kv->coefficient_bytes(),e.ctx);std::fflush(stdout);
    Q38Protocol::Event ev{};
    while(state.next(ev)){
        if(ev.kind!=Q38Protocol::Kind::command){state.finish_reply([&]{std::puts(ev.kind==Q38Protocol::Kind::busy?"ERR single session is busy":"ERR input queue full");std::fflush(stdout);});continue;}
        if(ev.line=="QUIT")break;bool generation=ev.line.rfind("GEN ",0)==0;bool fatal=false;
        try{std::istringstream in(ev.line);std::string cmd,num,part;in>>cmd>>num;Q::require(cmd=="GEN"&&!num.empty(),"expected GEN max_new token_ids; unsupported feature");int count=integer(num,0,e.ctx);Sampling s;
            while(in>>part&&part.find('=')!=std::string::npos){size_t eq=part.find('=');option(s,part.substr(0,eq),part.substr(eq+1),e.ctx);part.clear();}
            Q::require(!part.empty(),"missing token IDs");std::string extra;Q::require(!(in>>extra),"extra GEN argument");
            if(s.last_n==0&&(s.repeat!=1||s.freq!=0||s.present!=0))s.last_n=std::min(64,e.ctx);
            run(e,ids(part),count,s,state.stop_flag(),&state);
        }catch(const std::exception& ex){try{e.reset();}catch(...){fatal=true;}if(dynamic_cast<const Q::ResourceError*>(&ex))fatal=true;
            auto reply=[&]{std::string msg=ex.what();for(char& c:msg)if(c=='\n'||c=='\r')c=' ';std::printf("ERR %s\n",msg.c_str());std::fflush(stdout);};
            if(generation)state.finish([&](bool){reply();});else state.finish_reply(reply);
        }
        if(fatal)break;
    }
    if(reader.joinable()){if(!state.input_ended())CancelSynchronousIo(reader.native_handle());reader.join();}
}
int main(int argc,char** argv){try{
    std::string model,tokens,forced,dump,kv_mode="auto",spin_view,spin_profile;int ctx=4096,count=32;bool resident=false,inspection=false,conservative=false;Sampling sampling;
    for(int i=1;i<argc;++i){std::string a=argv[i];auto value=[&](){Q::require(i+1<argc,"missing option value");return std::string(argv[++i]);};
        if(a=="--help"||a=="-h"){std::puts("strata-q38 own-native CUDA SM120\n--model FILE --ctx-size N (4096) --serve\n--tokens IDS --n N [--force IDS] [--dump-full-logits FILE]\n--kv auto|f16|k8v4|spin8_k8v4|f16_turbo2|f16_turbo3|f16_turbo4 (auto default)\nAuto order: f16, f16_turbo4, spin8_k8v4; context and weights unchanged.\n--spin8-view vector|plus|minus --spin8-profile FILE (compiled reference default)\n--conservative-sync (diagnostic reference execution)\n--inspect (CPU only) --version\nFP32 recurrent state, sequential prefill; no vision/prefix/shift/speculation/graph.");return 0;}
        if(a=="--version"){std::puts("strata-q38 own-native 0.6.0-dev CUDA13 SM120");return 0;}
        if(a=="--model")model=value();else if(a=="--ctx-size")ctx=integer(value(),1,262144);else if(a=="--tokens")tokens=value();else if(a=="--n")count=integer(value(),0,262144);
        else if(a=="--force")forced=value();else if(a=="--dump-full-logits")dump=value();else if(a=="--serve")resident=true;else if(a=="--inspect")inspection=true;
        else if(a=="--kv")kv_mode=value();else if(a=="--spin8-view")spin_view=value();else if(a=="--spin8-profile")spin_profile=value();
        else if(a=="--conservative-sync")conservative=true;
        else if(a=="--seed")option(sampling,"seed",value(),ctx);else if(a=="--temperature")option(sampling,"temperature",value(),ctx);else throw std::runtime_error("unsupported option: "+a);
    }
    Q::require(!model.empty(),"--model is required");
    const auto kv_config=Q::KvConfig::admit(kv_mode,spin_view,spin_profile);
    if(inspection){strata::GgufFile f(model,Q::retained_metadata(),true);auto a=Q::inspect(f);std::printf("GGUF own-native qwen35 dense layers=64 tensors=%zu nextn_ignored=%d gpu_weight_bytes=%llu host_embedding_bytes=%llu folded=%zu inverse_embedding=%d grouped_gdn=%d\n",f.tensors().size(),a.nextn,(unsigned long long)a.gpu_weights,(unsigned long long)a.embedding_bytes,a.folded.size(),a.inverse_embedding,a.grouped_gdn);return 0;}
    Q::require(resident!=!tokens.empty(),"choose exactly one of --serve or --tokens");Q::require(!resident||(forced.empty()&&dump.empty()),"force/logits are standalone options");
    auto prompt=tokens.empty()?std::vector<int32_t>{}:ids(tokens);auto force=forced.empty()?std::vector<int32_t>{}:ids(forced);if(!force.empty())count=int(force.size()+1);
    Q::host_resources();Engine e(model,ctx,kv_config,conservative);if(resident)serve(e);else{std::atomic<bool> stop{false};run(e,prompt,count,sampling,stop,nullptr,force,dump);}return 0;
}catch(const std::exception& e){std::fprintf(stderr,"strata-q38: %s\n",e.what());return 1;}}
