#pragma once
#include "strata/artifact/gguf_reader.hpp"
#include <set>
#include <array>

namespace strata::qwen38 {
inline std::set<std::string> retained_metadata(){return {"prism.hadamard.weight_names","prism.hadamard.inverse_weight_names","prism.hadamard.sign_widths","prism.hadamard.sign_values","qwen35.attention.recurrent_layers"};}
inline void require(bool ok, const std::string& why) { if (!ok) throw std::runtime_error(why); }
inline bool auxiliary(int type) { return type == 0 || type == 1 || type == 30; }
inline bool matrix_type(int type) { return auxiliary(type) || type == 8 || type == 12 || type == 13 || type == 14 || type == 23 || type == 142; }
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
struct Admission {
    std::set<std::string> required;
    uint64_t gpu_weights = 0, embedding_bytes = 0, ignored_bytes = 0;
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
inline Admission inspect(const GgufFile& f) {
    Admission a;
    auto arch = f.get("general.architecture");
    require(arch && arch->type == MetaType::STRING && arch->s == "qwen35", "requires dense qwen35 architecture");
    for (const auto& kv : f.metadata()) require(kv.first.rfind("split.",0) != 0, "split GGUF is unsupported");
    require(integer_meta(f,"qwen35.expert_count",true) == 0, "experts are unsupported");
    const uint64_t nextn = integer_meta(f,"qwen35.nextn_predict_layers",true);
    require(nextn <= 1, "requires zero or one declared MTP layer");
    a.nextn = int(nextn);
    require(integer_meta(f,"qwen35.block_count") == 64u+nextn, "requires 64 trunk layers plus optional declared MTP");
    for (const auto& kv : std::initializer_list<std::pair<const char*,uint64_t>>{
        {"embedding_length",5120},{"feed_forward_length",17408},{"attention.head_count",24},
        {"attention.head_count_kv",4},{"attention.key_length",256},{"attention.value_length",256},
        {"rope.dimension_count",64},{"ssm.conv_kernel",4},{"ssm.state_size",128},
        {"ssm.group_count",16},{"ssm.time_step_rank",48},{"ssm.inner_size",6144}})
        require(integer_meta(f,std::string("qwen35.")+kv.first)==kv.second, std::string("incompatible ")+kv.first);
    const auto* vocab=f.get("tokenizer.ggml.tokens");
    require(vocab && vocab->type==MetaType::ARRAY && vocab->elem==MetaType::STRING && vocab->count==248320,"invalid vocabulary");
    if(f.get("tokenizer.ggml.eos_token_id")) a.eos=int(integer_meta(f,"tokenizer.ggml.eos_token_id"));
    require(a.eos>=0 && a.eos<248320,"invalid EOS");
    const auto* sections=f.get("qwen35.rope.dimension_sections");
    require(sections && sections->type==MetaType::ARRAY && sections->elem==MetaType::I32 && sections->count==4,"invalid RoPE sections");
    for(size_t i=0;i<4;++i) require(sections->items[i].u==std::array<uint64_t,4>{11,11,10,0}[i],"incompatible RoPE sections");
    auto scalar=[&](const char* name,float expected) {
        auto m=f.get(std::string("qwen35.")+name);
        require(m && m->type==MetaType::F32 && std::isfinite(m->f) && std::abs(m->f-expected)<=std::abs(expected)*1e-6,"incompatible scalar metadata");
        return float(m->f);
    };
    a.eps=scalar("attention.layer_norm_rms_epsilon",1e-6f);
    a.rope_base=scalar("rope.freq_base",1e7f);
    if(const auto* r=f.get("qwen35.attention.recurrent_layers")) {
        require(r->type==MetaType::ARRAY && r->elem==MetaType::BOOL && r->count==64u+nextn && r->items.size()==r->count,"bad recurrent map");
        for(int i=0;i<64;++i) require(bool(r->items[i].u)==((i+1)%4!=0),"wrong recurrent ordering");
        if(a.nextn)require(r->items.size()==65&&!r->items[64].u,"MTP must be attention");
    }
    if(f.get("qwen35.full_attention_interval")) require(integer_meta(f,"qwen35.full_attention_interval")==4,"wrong attention interval");
    auto need=[&](const std::string& name,std::vector<uint64_t> dims,bool aux=false) {
        auto t=f.find(name); require(t && t->shape==dims,"missing or wrong tensor geometry: "+name);
        require(aux ? auxiliary(t->type) : matrix_type(t->type),"unsupported operation type: "+name);
        a.required.insert(name);
    };
    need("token_embd.weight",{5120,248320}); need("output.weight",{5120,248320}); need("output_norm.weight",{5120},true);
    for(int i=0;i<64;++i) {
        auto n=[&](const char* role,std::vector<uint64_t> dims,bool aux=false){need(layer_name(i,role),dims,aux);};
        n("attn_norm.weight",{5120},true); n("post_attention_norm.weight",{5120},true);
        n("ffn_gate.weight",{5120,17408});n("ffn_up.weight",{5120,17408});n("ffn_down.weight",{17408,5120});
        if((i+1)%4) {
            n("attn_qkv.weight",{5120,10240}); n("attn_gate.weight",{5120,6144}); n("ssm_out.weight",{6144,5120});
            n("ssm_alpha.weight",{5120,48}); n("ssm_beta.weight",{5120,48});
            n("ssm_a",{48},true);n("ssm_dt.bias",{48},true);n("ssm_conv1d.weight",{4,10240},true);n("ssm_norm.weight",{128},true);
        } else {
            n("attn_q.weight",{5120,12288});n("attn_k.weight",{5120,1024});n("attn_v.weight",{5120,1024});n("attn_output.weight",{6144,5120});
            n("attn_q_norm.weight",{256},true);n("attn_k_norm.weight",{256},true);
        }
    }
    if(f.get("prism.hadamard.version")) {
        require(integer_meta(f,"prism.hadamard.version")==1,"unsupported Hadamard version");
        require(integer_meta(f,"prism.hadamard.block_size")==1024,"requires supported Hadamard block 1024");a.hadamard_block=1024;
        auto string_is=[&](const char* key,const char* value){auto m=f.get(key);require(m&&m->type==MetaType::STRING&&m->s==value,std::string("unsupported ")+key);};
        string_is("prism.hadamard.transform","normalized-sylvester-walsh-hadamard");string_is("prism.hadamard.axis","input-last-dimension");string_is("prism.hadamard.sign_mode","explicit");
        if(auto m=f.get("prism.hadamard.tied_output"))require(m->type==MetaType::BOOL&&!m->u,"tied Hadamard output unsupported");
        auto array=[&](const char* key,MetaType type)->const MetaValue&{auto m=f.get(key);require(m&&m->type==MetaType::ARRAY&&m->elem==type&&m->items.size()==m->count,std::string("incomplete transform array: ")+key);return *m;};
        const auto& widths=array("prism.hadamard.sign_widths",MetaType::I32);const auto& values=array("prism.hadamard.sign_values",MetaType::I32);size_t off=0;
        for(const auto& m:widths.items){int w=int(m.u);require((w==5120||w==6144||w==17408)&&!a.signs.count(w)&&off+size_t(w)<=values.count,"invalid sign vector width");auto& sign=a.signs[w];
            for(int i=0;i<w;++i){int64_t v=int64_t(values.items[off++].u);require(v==1||v==-1,"sign must be +/-1");sign.push_back(float(v));}}
        require(off==values.count&&!a.signs.empty(),"sign vector count mismatch");
        for(const auto& m:array("prism.hadamard.weight_names",MetaType::STRING).items){auto t=f.find(m.s);
            bool role=m.s=="output.weight"||m.s.find("attn_q.weight")!=std::string::npos||m.s.find("attn_k.weight")!=std::string::npos||m.s.find("attn_v.weight")!=std::string::npos||m.s.find("attn_qkv.weight")!=std::string::npos||m.s.find("attn_gate.weight")!=std::string::npos||m.s.find("attn_output.weight")!=std::string::npos||m.s.find("ssm_out.weight")!=std::string::npos||m.s.find("ffn_gate.weight")!=std::string::npos||m.s.find("ffn_up.weight")!=std::string::npos||m.s.find("ffn_down.weight")!=std::string::npos;
            require(role&&a.required.count(m.s)&&t&&t->shape.size()==2&&a.signs.count(int(t->shape[0]))&&a.folded.insert(m.s).second,"invalid folded tensor role: "+m.s);}
        require(!a.folded.empty(),"empty folded weight set");
        if(f.get("prism.hadamard.inverse_weight_names")){const auto& inv=array("prism.hadamard.inverse_weight_names",MetaType::STRING);require(inv.count<=1,"unsupported inverse tables");if(inv.count){require(inv.items[0].s=="token_embd.weight"&&a.signs.count(5120),"unsupported inverse table");a.inverse_embedding=true;}}
        if(auto m=f.get("prism.hadamard.gdn_v_grouped")){require(m->type==MetaType::BOOL,"invalid GDN grouping");a.grouped_gdn=bool(m->u);}
    } else for(const auto& kv:f.metadata())require(kv.first.rfind("prism.hadamard.",0)!=0,"Hadamard metadata without version");
    std::vector<std::pair<uint64_t,uint64_t>> extents;
    for(const auto& t:f.tensors()) {
        require(t.name.find("_exps")==std::string::npos && t.name.find("expert")==std::string::npos && t.name.find("ffn_gate_inp")==std::string::npos,"expert tensor rejected");
        const uint64_t bytes=payload(t), room=f.file_size()-f.data_start();
        require(bytes && matrix_type(t.type) && t.offset<=room && bytes<=room-t.offset,"invalid tensor type or extent: "+t.name);
        require(t.offset%f.alignment()==0,"unaligned tensor offset");
        extents.emplace_back(t.offset,t.offset+bytes);
        if(t.name=="token_embd.weight") a.embedding_bytes=bytes;
        else if(a.required.count(t.name)) a.gpu_weights+=convert_auxiliary(t)?t.elements()*4:bytes;
        else { require(a.nextn==1 && t.name.rfind("blk.64.",0)==0,"unknown tensor role: "+t.name);a.ignored_bytes+=bytes; }
    }
    std::sort(extents.begin(),extents.end());
    for(size_t i=1;i<extents.size();++i)require(extents[i].first>=extents[i-1].second,"overlapping tensor payloads");
    return a;
}
}
