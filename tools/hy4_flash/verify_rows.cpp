// Fixed-input target logits: compare single-token decode and speculative windows.
#include "strata/hy4flash_gguf/forward.hpp"
#include "strata/hy4flash_gguf/geometry.hpp"
#include "strata/hy4flash_gguf/kv_cache.hpp"
#include "strata/hy4flash_gguf/model_file.hpp"
#include "strata/kernels/native_mmvq.hpp"
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <fstream>
#include <string>
#include <vector>

int main(int argc,char** argv) {
    if(argc<7 || argc>9){std::fprintf(stderr,"verify_rows MODEL PACK TOKENS OUTPUT PREFIX WINDOW [EXPERT_GiB [--resident-experts]]\n");return 2;}
    try {
        strata::GgufModel model=strata::GgufModel::open(argv[1]);
        strata::hy4_flash::Geometry g;
        std::string err=strata::hy4_flash::read_geometry(model.meta(),g,8192,model.find("per_layer_token_embd.weight"));
        if(!err.empty())throw std::runtime_error(err);
        std::ifstream input(argv[3]);std::vector<int32_t> tokens;int32_t value;
        while(input>>value)tokens.push_back(value);
        const int prefix=std::atoi(argv[5]),window=std::atoi(argv[6]);
        if(prefix<1 || prefix>=(int)tokens.size() || window<1 || window>8)return 2;
        strata::kernels::native_mmvq_set_q6_c50(true);
        strata::kernels::native_mmvq_set_multi_exact(false);
        strata::kernels::native_mmvq_set_rows(2);
        strata::hy4_flash::KvCache kv;
        if(!strata::hy4_flash::kv_cache_init(kv,g.trunk_blocks,g.kv_lora_rank,g.rope_dim,8192,err))throw std::runtime_error(err);
        strata::hy4_flash::DenseResidency dense;
        strata::hy4_flash::ForwardOptions opt;
        opt.cache=&kv;opt.dense=&dense;opt.history=&tokens;
        const std::string budget = argc>=8 ? argv[7] : "48";
        std::size_t consumed=0;
        const double bytes=std::stod(budget,&consumed)*(1ull<<30);
        if(consumed!=budget.size() || !std::isfinite(bytes) || bytes<0 || bytes>=double(SIZE_MAX))return 2;
        opt.expert_vram_bytes=static_cast<std::size_t>(bytes);
        if(argc==9) {
            if(std::string(argv[8])!="--resident-experts")return 2;
            opt.resident_experts=true;
        }
        if (!strata::hy4_flash::prepare_experts(argv[1],g,opt,err)) throw std::runtime_error(err);
        opt.want_logits=false;
        std::vector<float> logits;
        // The first T1 after prefill primes async expert execution. Consume it in
        // the common prefix so the first compared row already uses the CPU pool.
        const char* cap=std::getenv("STRATA_HY4_FLASH_CPU_MISS_MAX");
        const bool prime_cpu=cap && std::atoi(cap)>0;
        if(prime_cpu && prefix<2)throw std::runtime_error("CPU oracle needs at least two prefix tokens");
        const auto prompt_end=tokens.begin()+prefix-int(prime_cpu);
        if(!strata::hy4_flash::forward_prefill(argv[1],g,argv[2],{tokens.begin(),prompt_end},logits,err,opt))throw std::runtime_error(err);
        if(prime_cpu && !strata::hy4_flash::forward_prefill(argv[1],g,argv[2],{*prompt_end},logits,err,opt))throw std::runtime_error(err);
        opt.want_logits=true;opt.speculative=window>1;opt.decode_equivalent=window>1;
        std::ofstream out(argv[4],std::ios::binary);
        for(std::size_t at=prefix;at<tokens.size();at+=window) {
            const auto end=std::min(tokens.size(),at+window);
            if(!strata::hy4_flash::forward_prefill(argv[1],g,argv[2],{tokens.begin()+(long)at,tokens.begin()+(long)end},logits,err,opt))throw std::runtime_error(err);
            out.write((char*)logits.data(),logits.size()*sizeof(float));
        }
        std::printf("wrote %zu rows x %d logits\n",tokens.size()-prefix,g.vocab_size);
    }catch(const std::exception& e){std::fprintf(stderr,"%s\n",e.what());return 1;}
}
