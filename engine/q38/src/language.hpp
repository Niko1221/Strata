#pragma once
// CPU-only, finite orthography preference. This is not a language classifier.
#include <array>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>
#include <limits>

namespace strata::qwen38 {
inline constexpr const char* orthography_version = "ja-finite-v1";
// UTF-8 hex escapes keep the policy independent of the compiler source codepage.
inline const std::vector<uint32_t>& suppressed_codepoints() {
    static const std::vector<uint32_t> cps={0x9019,0x5011,0x8B93,0x55CE,0x8207,0x6703,0x8AAA,0x570B,0x9AD4,0x5B78,0x767C,0x5920,0x9EBC,0x908A,0x5F9E,0x4F86,0x61C9,0x5C0D,0x95DC,0x5BE6,0x9EDE,0x8655,0x81FA,0x7063};
    return cps;
}
inline uint64_t language_exceptions(const std::string& value) {
    if(value=="none")return 0;
    if(value.empty()||value.size()>16)throw std::runtime_error("invalid ja exception mask");
    uint64_t mask=0;
    for(char c:value){unsigned n=c>='0'&&c<='9'?c-'0':c>='a'&&c<='f'?c-'a'+10:16;
        if(n>15)throw std::runtime_error("invalid ja exception mask");mask=(mask<<4)|n;}
    if(mask>>suppressed_codepoints().size())throw std::runtime_error("unknown ja exception bits");
    return mask;
}
inline bool japanese_profile(const std::string& value){
    if(value=="ja")return true;
    if(value=="model-default")return false;
    throw std::runtime_error("output language must be ja or model-default");
}
inline std::vector<uint32_t> utf8_scalars(const std::string& value){
    std::vector<uint32_t> out;
    for(size_t i=0;i<value.size();){unsigned b=(unsigned char)value[i++],n;uint32_t cp,min;
        if(b<128){out.push_back(b);continue;}
        if(b>=0xc2&&b<=0xdf){n=1;cp=b&31;min=0x80;}
        else if(b>=0xe0&&b<=0xef){n=2;cp=b&15;min=0x800;}
        else if(b>=0xf0&&b<=0xf4){n=3;cp=b&7;min=0x10000;}
        else throw std::runtime_error("invalid tokenizer UTF-8");
        if(i+n>value.size())throw std::runtime_error("truncated tokenizer UTF-8");
        while(n--){b=(unsigned char)value[i++];if((b&0xc0)!=0x80)throw std::runtime_error("invalid tokenizer UTF-8");cp=(cp<<6)|(b&63);}
        if(cp<min||cp>0x10ffff||(cp>=0xd800&&cp<=0xdfff))throw std::runtime_error("invalid tokenizer scalar");out.push_back(cp);
    }return out;
}
// Exact GPT-2 bytes_to_unicode inverse, including bytes represented by U+0100+.
inline std::string bpe_bytes(const std::string& token){
    static const auto inverse=[](){std::map<uint32_t,unsigned char> m;unsigned extra=256;
        for(unsigned b=0;b<256;++b){bool direct=(b>=33&&b<=126)||(b>=161&&b<=172)||(b>=174);
            m[direct?b:extra++]=(unsigned char)b;}return m;}();
    std::string out;for(auto cp:utf8_scalars(token)){auto it=inverse.find(cp);if(it==inverse.end())throw std::runtime_error("token outside GPT-2 byte alphabet");out+=(char)it->second;}return out;
}
class Orthography {
    struct Effect {uint64_t hits=0;uint16_t state=0;};
    std::vector<std::string> prefixes{""};
    std::vector<std::string> patterns;
    std::vector<std::array<Effect,256>> transitions;
    std::vector<std::string> bytes;
    std::vector<bool> special;
    std::map<uint16_t,std::vector<Effect>> cached;
    uint16_t state=0;
    uint64_t exceptions=0;
    bool active=false;
    const std::vector<Effect>& effects(){
        auto it=cached.find(state);if(it!=cached.end())return it->second;
        std::vector<Effect> row(bytes.size());
        for(size_t i=0;i<bytes.size();++i){Effect e;e.state=state;
            if(special[i])e.state=0; // controls are preserved and separate textual spans
            else for(unsigned char b:bytes[i]){const auto& next=transitions[e.state][b];e.hits|=next.hits;e.state=next.state;}
            row[i]=e;
        }return cached.emplace(state,std::move(row)).first->second;
    }
public:
    Orthography(const std::vector<std::string>& tokens,const std::vector<int>& types){
        if(tokens.size()!=types.size())throw std::runtime_error("token types missing for language profile");
        for(auto cp:suppressed_codepoints()){
            std::string p;p+=(char)(0xe0|(cp>>12));p+=(char)(0x80|((cp>>6)&63));p+=(char)(0x80|(cp&63));patterns.push_back(p);
            for(size_t n=1;n<p.size();++n)if(std::find(prefixes.begin(),prefixes.end(),p.substr(0,n))==prefixes.end())prefixes.push_back(p.substr(0,n));
        }
        transitions.resize(prefixes.size());
        for(size_t s=0;s<prefixes.size();++s)for(unsigned b=0;b<256;++b){auto text=prefixes[s]+char(b);Effect e;
            for(size_t j=0;j<patterns.size();++j)if(text.size()>=3&&text.compare(text.size()-3,3,patterns[j])==0)e.hits|=uint64_t(1)<<j;
            size_t longest=0;for(size_t j=1;j<prefixes.size();++j)if(prefixes[j].size()>longest&&text.size()>=prefixes[j].size()&&text.compare(text.size()-prefixes[j].size(),prefixes[j].size(),prefixes[j])==0){longest=prefixes[j].size();e.state=uint16_t(j);}
            transitions[s][b]=e;
        }
        for(size_t i=0;i<tokens.size();++i){
            if(types[i]<1||types[i]>6)throw std::runtime_error("unsupported tokenizer token type");
            special.push_back(types[i]==3||types[i]==4);bytes.push_back(special.back()?"":bpe_bytes(tokens[i]));
        }
    }
    void reset(bool enabled,uint64_t allowed=0){state=0;active=enabled;exceptions=allowed;}
    void mask(std::vector<float>& logits){if(!active)return;if(logits.size()!=bytes.size())throw std::runtime_error("language vocabulary mismatch");
        const auto& row=effects();bool finite=false;for(size_t i=0;i<logits.size();++i){if(row[i].hits&~exceptions)logits[i]=-std::numeric_limits<float>::infinity();finite|=std::isfinite(logits[i]);}
        if(!finite)throw std::runtime_error("ja constraint has no permitted finite candidate");
    }
    void accept(int token){if(!active)return;if(token<0||size_t(token)>=bytes.size())throw std::runtime_error("invalid language token");const auto e=effects()[token];if(e.hits&~exceptions)throw std::runtime_error("ja selected suppressed token");state=e.state;}
};
}
