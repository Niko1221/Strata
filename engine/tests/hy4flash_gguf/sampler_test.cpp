#include "strata/hy4flash_gguf/sampler.hpp"
#include <cmath>
#include <cstdio>
#include <set>
#include <vector>
#include <cstdint>

int main() {
    using namespace strata::hy4_flash;
    const float logits[]={0.f,0.1f,0.2f};
    SamplerConfig c;c.seed=42;c.temperature=1.f;
    Sampler random(c);std::set<int64_t> seen;
    for(int i=0;i<100;++i)seen.insert(random.pick(logits,3,{}));
    if(seen.size()!=3)return 1;
    c.temperature=0;c.top_p=0.9f;
    Sampler greedy(c);
    for(int i=0;i<100;++i)if(greedy.pick(logits,3,{})!=2)return 2;
    SamplerConfig masked = c;
    masked.excluded_token = 2;
    if (Sampler(masked).pick(logits, 3, {}) != 1) return 7;
    masked.temperature = 1;
    Sampler masked_draw(masked);
    for (int i = 0; i < 100; ++i) if (masked_draw.pick(logits, 3, {}) == 2) return 8;
    c.repeat_penalty=4;
    Sampler penalty(c);
    if(penalty.pick(logits,3,{2})!=1)return 3;

    // Speculative sample-and-match must draw exactly once per committed row,
    // retaining the rejection token. No extra RNG draw for drafts or rejection.
    c.temperature=.7f;c.repeat_penalty=1;c.top_p=1;
    Sampler plain(c), speculative(c);
    int accepted=0,rejected=0;
    for(int window=0;window<1000;++window) {
        for(int row=0;row<4;++row) {
            auto a=plain.pick(logits,3,{}), b=speculative.pick(logits,3,{});
            if(a!=b)return 4;
            const int draft=(window+row)%3;
            if(row==3)break;
            if(b!=draft){++rejected;break;}
            ++accepted;
        }
    }
    if(!accepted||!rejected)return 5;
    std::puts("PASS temperature, greedy penalties, speculative RNG continuity");
}
