#pragma once
#include "language.hpp"
#include "strata/qwen35/sampling.hpp"
#include <numeric>
#include <random>
namespace strata::qwen38 {
using strata::qwen35::Sampling;
inline int sample(std::vector<float> lg,const Sampling& s,const std::vector<int32_t>& hist,std::mt19937_64& rng,Orthography* orthography=nullptr){
    strata::qwen35::apply_penalties(lg,s,hist);if(orthography)orthography->mask(lg);if(s.temperature<=0)return int(std::max_element(lg.begin(),lg.end())-lg.begin());
    // top_k=0 retains the entire vocabulary; never silently replace it by 64.
    std::vector<int> ix(lg.size());std::iota(ix.begin(),ix.end(),0);size_t count=s.top_k?std::min(size_t(s.top_k),ix.size()):ix.size();
    std::partial_sort(ix.begin(),ix.begin()+count,ix.end(),[&](int a,int b){return lg[a]>lg[b]||(lg[a]==lg[b]&&a<b);});ix.resize(count);
    std::vector<double> p(count);double total=0;for(size_t i=0;i<count;++i)total+=(p[i]=std::exp((double(lg[ix[i]])-lg[ix[0]])/s.temperature));
    size_t keep=count;double c=0;for(size_t i=0;i<count;++i){if(p[i]<s.min_p*p[0]){keep=i;break;}c+=p[i]/total;if(c>=s.top_p){keep=i+1;break;}}
    keep=std::max(size_t(1),keep);double mass=std::accumulate(p.begin(),p.begin()+keep,0.0);double r=std::uniform_real_distribution<double>(0,mass)(rng);
    for(size_t i=0;i<keep;++i)if((r-=p[i])<=0)return ix[i];return ix[keep-1];
}
}
