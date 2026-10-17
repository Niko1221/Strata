// Enumerate proposal and acceptance branches exactly; no Monte Carlo or GPU.
#include "strata/core/spec_prob.hpp"
#include "strata/spec/draft_policy.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstring>
using namespace strata::core;
using Dist = std::array<double, 2>;
int failures = 0;
void check(bool ok, const char* message) {
    if (!ok) { ++failures; std::fprintf(stderr, "FAIL: %s\n", message); }
}
Dist judged(const Dist& p, const std::array<float,2>& q, int d) {
    const int32_t ids[2] = {0,1};
    const SpecP P{ids,p.data(),2}; const SpecQ Q{ids,q.data(),2};
    const double a=std::min(1.0,p[d]/q[d]); Dist result{};
    if (a>0) result[spec_verify_row(P,d,Q,(float)(a/2),0.5f).token]+=a;
    if (a<1) result[spec_verify_row(P,d,Q,(float)((1+a)/2),0.5f).token]+=1-a;
    return result;
}
Dist gated(const Dist& p, const std::array<float,2>& q, bool own_draw, float threshold) {
    Dist result{};
    for (int d=0;d<2;++d) {
        const float gate=own_draw?q[d]:std::max(q[0],q[1]);
        const auto row=gate>=threshold?judged(p,q,d):p;
        for (int i=0;i<2;++i) result[i]+=q[d]*row[i];
    }
    return result;
}
bool equal(const Dist& a,const Dist& b) {
    return std::abs(a[0]-b[0])<1e-12 && std::abs(a[1]-b[1])<1e-12;
}
int main(int argc,char** argv) {
    const bool approximate=argc>1 && std::strcmp(argv[1],"pick")==0;
    check(spec_gate_pick()==approximate,"top is default; pick requires explicit opt-in");
    const Dist p{0.5,0.5}; const std::array<float,2> q{0.875f,0.125f};
    check(equal(gated(p,q,true,0.5f),Dist{0.5625,0.4375}),"own-draw gate reproduces the exact bias");
    check(equal(gated(p,q,spec_gate_pick(),0.5f),approximate?Dist{0.5625,0.4375}:p),"configured gate follows its stated law");
    for(int a=1;a<16;++a) for(int b=1;b<16;++b) {
        const Dist P{a/16.0,1-a/16.0}; const std::array<float,2> Q{b/16.0f,1-b/16.0f};
        for(float threshold:{0.0f,0.5f,0.9f,1.1f})
            check(equal(gated(P,Q,false,threshold),P),"distribution-based admission preserves p");
    }
    // A planner must not learn from draws it may select, including discarded position-keyed lookahead.
    strata::spec::DraftPolicy live(6);
    for(int i=0;i<5;++i) { live.observe(false,4,3,24,1.0);live.observe(false,6,5,24,1.5); }
    const auto request_plan=live;
    auto short_chain=live,long_chain=live;
    short_chain.observe(false,4,3,24,0.2);long_chain.observe(false,4,3,24,8.0);
    const bool short_lookup=short_chain.choose(4,5,24).lookup,long_lookup=long_chain.choose(4,5,24).lookup;
    check(!short_lookup && long_lookup,"draw-dependent chain time can change the actual policy choice");
    Dist conditional{},planned{};
    for(int d=0;d<2;++d) {
        const auto selected=(d==0?long_lookup:short_lookup)?p:judged(p,q,d);
        const auto fixed=request_plan.choose(4,5,24).lookup?p:judged(p,q,d);
        for(int i=0;i<2;++i) { conditional[i]+=q[d]*selected[i];planned[i]+=q[d]*fixed[i]; }
    }
    check(equal(conditional,Dist{0.4375,0.5625}),"conditioning the source changes the proposal law");
    check(equal(planned,p),"request plan preserves p despite subsequent timing observations");
    std::printf("spec_admission_test: 900 exact distribution cases and source/gate controls; %d failures\n",failures);
    return failures?1:0;
}
