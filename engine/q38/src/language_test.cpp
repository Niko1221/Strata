#include "sampler.hpp"
#include <iostream>
#include <functional>
namespace Q = strata::qwen38;
static void check(bool ok,const char* why){if(!ok)throw std::runtime_error(why);}
static std::string utf8(uint32_t cp){std::string s;if(cp<128)s+=char(cp);else if(cp<2048){s+=char(0xc0|(cp>>6));s+=char(0x80|(cp&63));}else if(cp<65536){s+=char(0xe0|(cp>>12));s+=char(0x80|((cp>>6)&63));s+=char(0x80|(cp&63));}else{s+=char(0xf0|(cp>>18));s+=char(0x80|((cp>>12)&63));s+=char(0x80|((cp>>6)&63));s+=char(0x80|(cp&63));}return s;}
static std::string bpe(const std::string& bytes){std::string out;for(unsigned char b:bytes){unsigned extra=256,cp=0;for(unsigned j=0;j<=b;++j){bool direct=(j>=33&&j<=126)||(j>=161&&j<=172)||j>=174;cp=direct?j:extra++;}out+=utf8(cp);}return out;}
static void refuses(const std::function<void()>& fn){bool refused=false;try{fn();}catch(const std::exception&){refused=true;}check(refused,"expected refusal");}
int main(){try{
    // Each blocked spelling, fused and split at both UTF-8 boundaries, reaches the real sampler.
    for(auto cp:Q::suppressed_codepoints())for(size_t split=0;split<3;++split){auto text=utf8(cp);
        std::vector<std::string> tokens={bpe(split?text.substr(split):text),bpe("ok"),bpe(text.substr(0,split)),"<|im_end|>"};
        Q::Orthography language(tokens,{1,1,1,3});Q::Sampling s;s.temperature=0;std::mt19937_64 rng(42);
        language.reset(true);if(split)language.accept(2);
        check(Q::sample({100,3,-100,-100},s,{},rng,&language)==1,"greedy leaked suppressed token");
        s.temperature=0.8f;s.top_k=0;s.top_p=1;s.min_p=0;
        for(int i=0;i<32;++i)check(Q::sample({100,3,-100,-100},s,{},rng,&language)==1,"seeded selection leaked");
        language.reset(false);s.temperature=0;check(Q::sample({100,3,-100,-100},s,{},rng,&language)==Q::sample({100,3,-100,-100},s,{},rng),"disabled sampler changed");
        if(split){language.reset(true);check(Q::sample({100,3,-100,-100},s,{},rng,&language)==0,"new GEN state retained old prefix");}
        size_t index=std::find(Q::suppressed_codepoints().begin(),Q::suppressed_codepoints().end(),cp)-Q::suppressed_codepoints().begin();
        language.reset(true,uint64_t(1)<<index);if(split)language.accept(2);check(Q::sample({100,3,-100,-100},s,{},rng,&language)==0,"literal exception not preserved");
    }
    std::string shared;for(uint32_t cp:{0x65e5,0x672c,0x8a9e,0x554f,0x984c,0x6771,0x4eac,0x6f22,0x5b57,0x70ba,0x6642,0x7684,0x88cf,0x20bb7,0x3042,0x30a2})shared+=utf8(cp);
    Q::Orthography allowed({bpe(shared),bpe("{\"code\":\"x = 1;\"}"),"<tool_call>","<think>","</think>","<|im_end|>"},{1,1,4,4,4,3});
    Q::Sampling s;s.temperature=0;std::mt19937_64 rng(1);allowed.reset(true);
    for(int i=0;i<6;++i){std::vector<float> lg(6,-5);lg[i]=100;check(Q::sample(lg,s,{},rng,&allowed)==i,"allowed text/control changed");allowed.accept(i);}
    Q::Orthography empty({bpe(utf8(0x570b))},{1});empty.reset(true);refuses([&]{Q::sample({100},s,{},rng,&empty);});
    Q::Orthography fused({bpe("a"+utf8(0x570b)+"z"),bpe("Japan")},{1,1});fused.reset(true);check(Q::sample({100,3},s,{},rng,&fused)==1,"fused token leaked");
    for(auto v:{"","-1","1000000","0\nGEN","FFFFFFFF","00000000000000000"})refuses([&]{Q::language_exceptions(v);});
    check(Q::language_exceptions("ffffff")==0xffffff,"mask bound");refuses([]{Q::japanese_profile("zh");});
    for(unsigned b=0;b<256;++b)check(Q::bpe_bytes(bpe(std::string(1,char(b))))==std::string(1,char(b)),"GPT2 byte mapping");
    std::cout<<"PASS real greedy/seeded sampler, 24 spellings fused and split, reset, literal exceptions, shared Japanese/supplementary names, ASCII/control tokens, disabled equivalence, empty mask and malformed options\n";return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<"\n";return 1;}}
