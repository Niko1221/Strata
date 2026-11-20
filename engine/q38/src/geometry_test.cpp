// Geometry admission tests: synthetic GGUF metadata only, no weights, no CUDA.
// Verifies that both supported dense profiles are admitted from metadata,
// that derived sizes are correct, and that a hybrid shape is rejected.
#include "strata/qwen38/admission.hpp"
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace Q = strata::qwen38;
template<class T> static void put(std::ofstream& f, T x){f.write(reinterpret_cast<const char*>(&x),sizeof(x));}
static void str(std::ofstream& f,const std::string& x){put<uint64_t>(f,x.size());f.write(x.data(),x.size());}

static void write(const std::filesystem::path& p,const Q::Geometry& g){
    std::ofstream f(p,std::ios::binary);
    // GGUF v3 header: magic, version, zero tensors, then patched metadata count.
    put<uint32_t>(f,0x46554747);put<uint32_t>(f,3);put<uint64_t>(f,0);put<uint64_t>(f,0);
    std::vector<std::function<void(std::ofstream&)>> meta;
    auto u32=[&](const char* k,uint64_t v){meta.push_back([k,v](std::ofstream& o){str(o,k);put<uint32_t>(o,4);put<uint32_t>(o,uint32_t(v));});};
    auto f32=[&](const char* k,float v){meta.push_back([k,v](std::ofstream& o){str(o,k);put<uint32_t>(o,6);put<float>(o,v);});};
    auto s=[&](const char* k,const char* v){meta.push_back([k,v](std::ofstream& o){str(o,k);put<uint32_t>(o,8);str(o,v);});};
    u32("qwen35.expert_count",0);u32("qwen35.nextn_predict_layers",uint64_t(g.nextn));
    u32("qwen35.block_count",uint64_t(g.trunk_layers+g.nextn));
    u32("qwen35.embedding_length",uint64_t(g.hidden));u32("qwen35.feed_forward_length",uint64_t(g.ffn));
    u32("qwen35.attention.head_count",uint64_t(g.q_heads));u32("qwen35.attention.head_count_kv",uint64_t(g.kv_heads));
    u32("qwen35.attention.key_length",uint64_t(g.head_dim));u32("qwen35.attention.value_length",uint64_t(g.head_dim));
    u32("qwen35.rope.dimension_count",64);u32("qwen35.ssm.conv_kernel",uint64_t(g.conv));
    u32("qwen35.ssm.state_size",uint64_t(g.state));u32("qwen35.ssm.group_count",uint64_t(g.group));
    u32("qwen35.ssm.time_step_rank",uint64_t(g.rank));u32("qwen35.ssm.inner_size",uint64_t(g.inner));
    u32("qwen35.full_attention_interval",uint64_t(g.interval));
    f32("qwen35.attention.layer_norm_rms_epsilon",g.eps);f32("qwen35.rope.freq_base",g.rope_base);
    s("general.architecture","qwen35");
    meta.push_back([](std::ofstream& o){str(o,"qwen35.rope.dimension_sections");put<uint32_t>(o,9);put<uint32_t>(o,5);put<uint64_t>(o,4);for(int x:{11,11,10,0})put<int32_t>(o,x);});
    std::vector<bool> recurrent(g.trunk_layers+g.nextn,false);
    for(int i=0;i<g.trunk_layers;++i)recurrent[i]=(i+1)%g.interval!=0;
    meta.push_back([recurrent](std::ofstream& o){str(o,"qwen35.attention.recurrent_layers");put<uint32_t>(o,9);put<uint32_t>(o,7);put<uint64_t>(o,recurrent.size());for(bool b:recurrent)put<uint8_t>(o,b?1:0);});
    meta.push_back([n=g.vocab](std::ofstream& o){str(o,"tokenizer.ggml.tokens");put<uint32_t>(o,9);put<uint32_t>(o,8);put<uint64_t>(o,n);for(int i=0;i<n;++i)str(o,std::string());});
    u32("tokenizer.ggml.eos_token_id",uint64_t(g.eos));
    const std::streampos after=f.tellp();
    f.seekp(16);put<uint64_t>(f,meta.size());f.seekp(after);
    for(auto& m:meta)m(f);
    const std::streampos table=f.tellp();
    const uint64_t data_start=(uint64_t(table)+31)/32*32;
    f.seekp(std::streamoff(data_start+32));f.put(0);
}

static Q::Geometry profile27(){Q::Geometry g;g.trunk_layers=64;g.nextn=1;g.hidden=5120;g.ffn=17408;g.vocab=248320;g.eos=248046;g.q_heads=24;g.kv_heads=4;g.head_dim=256;g.group=16;g.state=128;g.rank=48;g.inner=6144;g.conv=4;g.interval=4;g.rope_sections={11,11,10,0};g.eps=1e-6f;g.rope_base=1e7f;return g;}
static Q::Geometry profile9(){Q::Geometry g=profile27();g.trunk_layers=32;g.hidden=4096;g.ffn=12288;g.q_heads=16;g.rank=32;g.inner=4096;return g;}
static strata::GgufFile open(const std::filesystem::path& p){return strata::GgufFile(p.string(),Q::retained_metadata(),true,248320,64ull*1024*1024);}

int main(){
    const auto dir=std::filesystem::temp_directory_path()/("q38-geometry-"+std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    try{
        std::filesystem::create_directories(dir);
        const auto nine=dir/"nine.gguf",twentyseven=dir/"twentyseven.gguf",hybrid=dir/"hybrid.gguf";
        write(nine,profile9());write(twentyseven,profile27());
        {
            auto f=open(nine);const auto g=Q::read_geometry(f);
            if(g.trunk_layers!=32||g.hidden!=4096||g.ffn!=12288||g.q_heads!=16||g.vocab!=248320||g.kv_heads!=4)throw std::runtime_error("9B geometry wrong");
            if(g.attention_layers()!=8||g.gdn_layers()!=24||g.gdn_qkv_channels()!=8192||g.attn_q_rows()!=8192||g.attn_out_rows()!=4096)throw std::runtime_error("9B derived geometry wrong");
        }
        {
            auto f=open(twentyseven);const auto g=Q::read_geometry(f);
            if(g.trunk_layers!=64||g.hidden!=5120||g.ffn!=17408||g.q_heads!=24||g.gdn_qkv_channels()!=10240)throw std::runtime_error("27B geometry wrong");
            if(g.attention_layers()!=16||g.gdn_layers()!=48||g.attn_q_rows()!=12288||g.attn_out_rows()!=6144)throw std::runtime_error("27B derived geometry wrong");
        }
        {
            auto h=profile9();h.hidden=5120;write(hybrid,h);auto f=open(hybrid);
            bool refused=false;try{Q::read_geometry(f);}catch(const std::exception&){refused=true;}
            if(!refused)throw std::runtime_error("hybrid geometry accepted");
        }
        std::filesystem::remove_all(dir);
        std::cout<<"PASS 9B/27B geometry admission, derived sizes, hybrid rejection\n";
        return 0;
    }catch(const std::exception& e){std::filesystem::remove_all(dir);std::cerr<<e.what()<<"\n";return 1;}
}
