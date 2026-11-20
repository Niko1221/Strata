#pragma once
#include "strata/artifact/gguf_reader.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <map>
#include <set>
#include <string>
#include <vector>

namespace strata::qwen38 {
inline std::set<std::string> retained_metadata(){return {"prism.hadamard.weight_names","prism.hadamard.inverse_weight_names","prism.hadamard.sign_widths","prism.hadamard.sign_values","qwen35.attention.recurrent_layers"};}
inline void require(bool ok, const std::string& why) { if (!ok) throw std::runtime_error(why); }
inline bool auxiliary(int type) { return type == 0 || type == 1 || type == 30; }
// 39 = MXFP4 (32 values / 17 bytes), 40 = NVFP4 (64 values / 36 bytes).
// FP8 (CUDA_R_8F_E4M3) is an internal compute format, not a GGUF tensor type.
inline bool matrix_type(int type) { return auxiliary(type) || type == 8 || type == 11 || type == 12 || type == 13 || type == 14 || type == 20 || type == 21 || type == 22 || type == 23 || type == 39 || type == 40 || type == 142; }
inline bool convert_auxiliary(const TensorInfo& t){return auxiliary(t.type)&&(t.shape.size()==1||t.name.find(".ssm_conv1d.")!=std::string::npos||t.name.find(".ssm_alpha.")!=std::string::npos||t.name.find(".ssm_beta.")!=std::string::npos);}
// PQ2_0 is deliberately local: the MoE reader's type 42 remains 64/18.
inline uint64_t payload(const TensorInfo& t) {
    if (t.type != 142) return tensor_payload_bytes(t);
    if (t.shape.empty() || t.shape[0] % 128) return 0;
    uint64_t n = 1;
    for (auto d : t.shape) { if (!d || n > UINT64_MAX / d) return 0; n *= d; }
    return n / 128 <= UINT64_MAX / 34 ? n / 128 * 34 : 0;
}
inline std::string layer_name(int i, const char* role) { return "blk." + std::to_string(i) + "." + role; }

// Geometry is derived only from GGUF metadata and tensor extents. Execution
// dispatch uses these values, never general.name, the file name or a publisher.
struct Geometry {
    int trunk_layers=0, nextn=0;
    int hidden=0, ffn=0, vocab=0, eos=0;
    int q_heads=0, kv_heads=0, head_dim=0;
    int group=0, state=0, rank=0, inner=0, conv=0;
    int interval=0;
    std::array<int,4> rope_sections{};
    float eps=0.f, rope_base=0.f;
    int attention_layers()const{return interval>0?trunk_layers/interval:0;}
    int gdn_layers()const{return trunk_layers-attention_layers();}
    // GDN qkv packs (2*group) q/k heads and (2*q_heads) value heads of `state` channels.
    int gdn_qkv_channels()const{return (2*group+2*q_heads)*state;}
    int attn_q_rows()const{return 2*q_heads*head_dim;}
    int attn_kv_rows()const{return kv_heads*head_dim;}
    int attn_out_rows()const{return q_heads*head_dim;}
    int mtp_layer()const{return trunk_layers;}
};

struct Admission {
    std::set<std::string> required;
    std::set<std::string> mtp_required;
    uint64_t gpu_weights = 0, mtp_weights = 0, embedding_bytes = 0, ignored_bytes = 0;
    Geometry geometry;
    int nextn = 0, eos = 248046;
    float eps = 1e-6f, rope_base = 1e7f;
    std::set<std::string> folded;
    std::map<int,std::vector<float>> signs;
    bool inverse_embedding=false,grouped_gdn=false;
    int hadamard_block=0;
};
inline uint64_t integer_meta(const GgufFile& f, const std::string& key, bool optional = false) {
    const auto* m = f.get(key);
    if (!m && optional) return 0;
    require(m && m->type == MetaType::U32, "missing or non-U32 metadata: " + key);
    return m->u;
}
inline Geometry read_geometry(const GgufFile& f) {
    Geometry g;
    auto arch = f.get("general.architecture");
    require(arch && arch->type == MetaType::STRING && arch->s == "qwen35", "requires dense qwen35 architecture");
    for (const auto& kv : f.metadata()) require(kv.first.rfind("split.",0) != 0, "split GGUF is unsupported");
    require(integer_meta(f,"qwen35.expert_count",true) == 0, "experts are unsupported");
    const uint64_t nextn = integer_meta(f,"qwen35.nextn_predict_layers",true);
    require(nextn <= 1, "requires zero or one declared MTP layer");
    g.nextn = int(nextn);
    const uint64_t blocks = integer_meta(f,"qwen35.block_count");
    require(blocks > 0 && blocks <= 1024, "invalid block_count");
    g.trunk_layers = int(blocks) - g.nextn;
    require(g.trunk_layers > 0, "block_count smaller than declared MTP layers");
    auto meta_int=[&](const char* name)->int {
        const uint64_t v=integer_meta(f,std::string("qwen35.")+name);
        require(v>0 && v<=INT32_MAX,std::string("invalid qwen35.")+name);
        return int(v);
    };
    g.hidden=meta_int("embedding_length");
    g.ffn=meta_int("feed_forward_length");
    g.q_heads=meta_int("attention.head_count");
    g.kv_heads=meta_int("attention.head_count_kv");
    g.head_dim=meta_int("attention.key_length");
    require(uint64_t(meta_int("attention.value_length"))==uint64_t(g.head_dim),"asymmetric attention head dimensions are unsupported");
    g.group=meta_int("ssm.group_count");
    g.state=meta_int("ssm.state_size");
    g.rank=meta_int("ssm.time_step_rank");
    g.inner=meta_int("ssm.inner_size");
    g.conv=meta_int("ssm.conv_kernel");
    g.interval=meta_int("full_attention_interval");
    require(meta_int("rope.dimension_count")==64,"incompatible rope.dimension_count");
    require(g.trunk_layers%g.interval==0,"block_count is not a whole number of attention intervals");
    require(g.kv_heads<=g.q_heads && g.q_heads%g.kv_heads==0,"incompatible attention head grouping");
    require(g.inner==g.q_heads*g.head_dim,"incompatible GDN inner size");
    require(g.state==128 && g.conv==4,"incompatible GDN state/convolution geometry");
    const auto* vocab=f.get("tokenizer.ggml.tokens");
    require(vocab && vocab->type==MetaType::ARRAY && vocab->elem==MetaType::STRING,"invalid vocabulary");
    g.vocab=int(vocab->count);
    require(g.vocab>0 && g.vocab<=INT32_MAX,"invalid vocabulary size");
    if(f.get("tokenizer.ggml.eos_token_id")) g.eos=int(integer_meta(f,"tokenizer.ggml.eos_token_id"));
    require(g.eos>=0 && g.eos<g.vocab,"invalid EOS");
    const auto* sections=f.get("qwen35.rope.dimension_sections");
    require(sections && sections->type==MetaType::ARRAY && sections->elem==MetaType::I32 && sections->count==4,"invalid RoPE sections");
    for(size_t i=0;i<4;++i){
        require(sections->items[i].u<=INT32_MAX,"invalid RoPE section");
        g.rope_sections[i]=int(sections->items[i].u);
    }
    auto scalar=[&](const char* name)->float {
        auto m=f.get(std::string("qwen35.")+name);
        require(m && m->type==MetaType::F32 && std::isfinite(m->f) && m->f>0.f,"invalid scalar metadata: "+std::string(name));
        return float(m->f);
    };
    g.eps=scalar("attention.layer_norm_rms_epsilon");
    g.rope_base=scalar("rope.freq_base");

    // Supported profiles are explicit. Adding a geometry requires reviewing
    // scratch sizing, GDN projections, attention grouping and graph loops.
    bool is27 = g.trunk_layers==64 && g.hidden==5120 && g.ffn==17408 && g.q_heads==24 &&
        g.kv_heads==4 && g.head_dim==256 && g.group==16 && g.state==128 && g.rank==48 &&
        g.inner==6144 && g.conv==4 && g.interval==4 && g.vocab==248320 &&
        g.rope_sections==std::array<int,4>{11,11,10,0} && std::abs(g.eps-1e-6f)<=1e-12f && std::abs(g.rope_base-1e7f)<=1.f && g.eos==248046;
    bool is9 = g.trunk_layers==32 && g.hidden==4096 && g.ffn==12288 && g.q_heads==16 &&
        g.kv_heads==4 && g.head_dim==256 && g.group==16 && g.state==128 && g.rank==32 &&
        g.inner==4096 && g.conv==4 && g.interval==4 && g.vocab==248320 &&
        g.rope_sections==std::array<int,4>{11,11,10,0} && std::abs(g.eps-1e-6f)<=1e-12f && std::abs(g.rope_base-1e7f)<=1.f && g.eos==248046;
    require(is27||is9,"unsupported qwen35 dense geometry (supported: 5120/17408/64 layers, 4096/12288/32 layers)");
    return g;
}
inline Admission inspect(const GgufFile& f) {
    Admission a;
    a.geometry=read_geometry(f);
    const Geometry& g=a.geometry;
    a.nextn=g.nextn; a.eos=g.eos; a.eps=g.eps; a.rope_base=g.rope_base;
    if(const auto* r=f.get("qwen35.attention.recurrent_layers")) {
        require(r->type==MetaType::ARRAY && r->elem==MetaType::BOOL && r->count==uint64_t(g.trunk_layers+g.nextn) && r->items.size()==r->count,"bad recurrent map");
        for(int i=0;i<g.trunk_layers;++i) require(bool(r->items[i].u)==((i+1)%g.interval!=0),"wrong recurrent ordering");
        if(a.nextn)require(r->items.size()==uint64_t(g.trunk_layers+1)&&!r->items[g.trunk_layers].u,"MTP must be attention");
    }
    auto need=[&](const std::string& name,std::vector<uint64_t> dims,bool aux=false) {
        auto t=f.find(name); require(t && t->shape==dims,"missing or wrong tensor geometry: "+name);
        require(aux ? auxiliary(t->type) : matrix_type(t->type),"unsupported operation type: "+name);
        a.required.insert(name);
    };
    need("token_embd.weight",{uint64_t(g.hidden),uint64_t(g.vocab)});
    need("output.weight",{uint64_t(g.hidden),uint64_t(g.vocab)});
    need("output_norm.weight",{uint64_t(g.hidden)},true);
    for(int i=0;i<g.trunk_layers;++i) {
        auto n=[&](const char* role,std::vector<uint64_t> dims,bool aux=false){need(layer_name(i,role),dims,aux);};
        n("attn_norm.weight",{uint64_t(g.hidden)},true);
        n("post_attention_norm.weight",{uint64_t(g.hidden)},true);
        n("ffn_gate.weight",{uint64_t(g.hidden),uint64_t(g.ffn)});
        n("ffn_up.weight",{uint64_t(g.hidden),uint64_t(g.ffn)});
        n("ffn_down.weight",{uint64_t(g.ffn),uint64_t(g.hidden)});
        if((i+1)%g.interval) {
            n("attn_qkv.weight",{uint64_t(g.hidden),uint64_t(g.gdn_qkv_channels())});
            n("attn_gate.weight",{uint64_t(g.hidden),uint64_t(g.inner)});
            n("ssm_out.weight",{uint64_t(g.inner),uint64_t(g.hidden)});
            n("ssm_alpha.weight",{uint64_t(g.hidden),uint64_t(g.rank)});
            n("ssm_beta.weight",{uint64_t(g.hidden),uint64_t(g.rank)});
            n("ssm_a",{uint64_t(g.rank)},true);
            n("ssm_dt.bias",{uint64_t(g.rank)},true);
            n("ssm_conv1d.weight",{uint64_t(g.conv),uint64_t(g.gdn_qkv_channels())},true);
            n("ssm_norm.weight",{uint64_t(g.state)},true);
        } else {
            n("attn_q.weight",{uint64_t(g.hidden),uint64_t(g.attn_q_rows())});
            n("attn_k.weight",{uint64_t(g.hidden),uint64_t(g.attn_kv_rows())});
            n("attn_v.weight",{uint64_t(g.hidden),uint64_t(g.attn_kv_rows())});
            n("attn_output.weight",{uint64_t(g.attn_out_rows()),uint64_t(g.hidden)});
            n("attn_q_norm.weight",{uint64_t(g.head_dim)},true);
            n("attn_k_norm.weight",{uint64_t(g.head_dim)},true);
        }
    }
    if(f.get("prism.hadamard.version")) {
        require(integer_meta(f,"prism.hadamard.version")==1,"unsupported Hadamard version");
        require(integer_meta(f,"prism.hadamard.block_size")==1024,"requires supported Hadamard block 1024");
        a.hadamard_block=1024;
        auto string_is=[&](const char* key,const char* value){auto m=f.get(key);require(m&&m->type==MetaType::STRING&&m->s==value,std::string("unsupported ")+key);};
        string_is("prism.hadamard.transform","normalized-sylvester-walsh-hadamard");
        string_is("prism.hadamard.axis","input-last-dimension");
        string_is("prism.hadamard.sign_mode","explicit");
        if(auto m=f.get("prism.hadamard.tied_output"))require(m->type==MetaType::BOOL&&!m->u,"tied Hadamard output unsupported");
        auto array=[&](const char* key,MetaType type)->const MetaValue&{auto m=f.get(key);require(m&&m->type==MetaType::ARRAY&&m->elem==type&&m->items.size()==m->count,std::string("incomplete transform array: ")+key);return *m;};
        const auto& widths=array("prism.hadamard.sign_widths",MetaType::I32);
        const auto& values=array("prism.hadamard.sign_values",MetaType::I32);
        const std::set<int> allowed{g.hidden,g.inner,g.ffn};
        size_t off=0;
        for(const auto& m:widths.items){
            int w=int(m.u);
            require(allowed.count(w)&&!a.signs.count(w)&&off+size_t(w)<=values.count,"invalid sign vector width");
            auto& sign=a.signs[w];
            for(int i=0;i<w;++i){int64_t v=int64_t(values.items[off++].u);require(v==1||v==-1,"sign must be +/-1");sign.push_back(float(v));}
        }
        require(off==values.count&&!a.signs.empty(),"sign vector count mismatch");
        for(const auto& m:array("prism.hadamard.weight_names",MetaType::STRING).items){
            auto t=f.find(m.s);
            bool role=m.s=="output.weight"||m.s.find("attn_q.weight")!=std::string::npos||m.s.find("attn_k.weight")!=std::string::npos||m.s.find("attn_v.weight")!=std::string::npos||m.s.find("attn_qkv.weight")!=std::string::npos||m.s.find("attn_gate.weight")!=std::string::npos||m.s.find("attn_output.weight")!=std::string::npos||m.s.find("ssm_out.weight")!=std::string::npos||m.s.find("ffn_gate.weight")!=std::string::npos||m.s.find("ffn_up.weight")!=std::string::npos||m.s.find("ffn_down.weight")!=std::string::npos;
            require(role&&a.required.count(m.s)&&t&&t->shape.size()==2&&a.signs.count(int(t->shape[0]))&&a.folded.insert(m.s).second,"invalid folded tensor role: "+m.s);
        }
        require(!a.folded.empty(),"empty folded weight set");
        if(f.get("prism.hadamard.inverse_weight_names")){
            const auto& inv=array("prism.hadamard.inverse_weight_names",MetaType::STRING);
            require(inv.count<=1,"unsupported inverse tables");
            if(inv.count){require(inv.items[0].s=="token_embd.weight"&&a.signs.count(g.hidden),"unsupported inverse table");a.inverse_embedding=true;}
        }
        if(auto m=f.get("prism.hadamard.gdn_v_grouped")){require(m->type==MetaType::BOOL,"invalid GDN grouping");a.grouped_gdn=bool(m->u);}
    } else for(const auto& kv:f.metadata())require(kv.first.rfind("prism.hadamard.",0)!=0,"Hadamard metadata without version");
    std::vector<std::pair<uint64_t,uint64_t>> extents;
    const std::string mtp_prefix="blk."+std::to_string(g.mtp_layer())+".";
    for(const auto& t:f.tensors()) {
        require(t.name.find("_exps")==std::string::npos && t.name.find("expert")==std::string::npos && t.name.find("ffn_gate_inp")==std::string::npos,"expert tensor rejected");
        const uint64_t bytes=payload(t), room=f.file_size()-f.data_start();
        require(bytes && matrix_type(t.type) && t.offset<=room && bytes<=room-t.offset,"invalid tensor type or extent: "+t.name);
        require(t.offset%f.alignment()==0,"unaligned tensor offset");
        extents.emplace_back(t.offset,t.offset+bytes);
        if(t.name=="token_embd.weight") a.embedding_bytes=bytes;
        else if(a.required.count(t.name)) a.gpu_weights+=convert_auxiliary(t)?t.elements()*4:bytes;
        else if(t.name.find(".indexer.")!=std::string::npos || t.name.size()>6 && t.name.compare(t.name.size()-6,6,".scale")==0){
            // QSA checkpoints carry per-weight global scale metadata and an
            // optional indexer projection set on attention layers. NVFP4 block
            // scales are already embedded in the 36-byte payload, so these are
            // admitted but not loaded. Their extents were validated above.
            a.ignored_bytes+=bytes;
        }
        else { require(a.nextn==1 && t.name.rfind(mtp_prefix,0)==0,"unknown tensor role: "+t.name);a.ignored_bytes+=bytes;a.mtp_required.insert(t.name);a.mtp_weights+=convert_auxiliary(t)?t.elements()*4:bytes; }
    }
    std::sort(extents.begin(),extents.end());
    for(size_t i=1;i<extents.size();++i)require(extents[i].first>=extents[i-1].second,"overlapping tensor payloads");
    return a;
}
}
