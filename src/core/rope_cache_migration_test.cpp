#include "strata/core/rope_cache_migration.hpp"
#include "strata/artifact/dequant.hpp"
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <random>
#include <numeric>
#include <limits>

using namespace strata::core;
using namespace strata::kernels;
namespace {
int checks = 0;
void check(bool value, const char* why) {
    ++checks;
    if (!value) { std::fprintf(stderr, "FAIL: %s\n", why); std::exit(1); }
}
void near(double a, double b, double tolerance, const char* why) {
    check(std::isfinite(a) && std::isfinite(b) && std::abs(a-b) <= tolerance, why);
}
// Independent FP64 oracle: no RopeScaling helpers/table/pair converter invoked.
std::pair<double,double> oracle(double x, double y, int64_t pos, int pair, bool yarn) {
    constexpr double pi = 3.14159265358979323846, base = 1e7, context = 262144;
    const double inv = std::exp(-2.0*pair/64*std::log(base));
    double angle = pos*inv, gain = 1;
    if (yarn) {
        const double lo = std::max(0.0, std::floor(64*std::log(context/(32*2*pi))/(2*std::log(base))));
        const double hi = std::min(63.0, std::ceil(64*std::log(context/(2*pi))/(2*std::log(base))));
        const double mix = 1-std::clamp((pair-lo)/std::max(0.001,hi-lo),0.0,1.0);
        angle *= mix + (1-mix)/4;
        gain = 1+0.1*std::log(4.0);
    }
    return {gain*(x*std::cos(angle)-y*std::sin(angle)),gain*(x*std::sin(angle)+y*std::cos(angle))};
}
std::vector<double> softmax(std::vector<double> x) {
    double m=*std::max_element(x.begin(),x.end());
    double sum=0; for(auto& v:x) sum+=(v=std::exp(v-m));
    for(auto& v:x) v/=sum;
    return x;
}
SessionRope wire(const RopeScaling& r) {
    return {(int64_t)r.type,r.freq_base,r.factor,r.freq_scale(),r.orig_ctx,r.ext_factor,r.attn_factor,r.beta_fast,r.beta_slow};
}
RopeCacheProfile profile(bool yarn) {
    RopeCacheProfile p;
    p.model_identity=123; p.full_model_identity=true;
    p.config.engine_version="test";p.config.backend="cuda";p.config.kv="fp16";
    p.config.mtp_window=-1;p.config.max_context=1048576;
    p.config.switches.emplace_back("migration_rope_table", 1);
    if(yarn) {p.rope.type=RopeScalingType::YaRN;p.rope.factor=4;p.rope.ext_factor=1;}
    p.config.rope=wire(p.rope);
    return p;
}
void test_ordinary_rope_to_yarn4_key_conversion() {
    auto a=profile(false),b=profile(true);
    std::mt19937 random(741);
    std::uniform_real_distribution<double> dist(-0.4,0.4);
    const int64_t positions[]={0,1,2,127,4096,65535,131072,262143};
    int groups[3]={};
    std::vector<double> expected_scores, converted_scores, values;
    for (int64_t offset : {0,73}) for(auto p:positions) {
        double expected_score=0,converted_score=0;
        for(int pair=0;pair<32;++pair) {
            const double x=dist(random),y=dist(random);
            auto old=oracle(x,y,p+offset,pair,false), want=oracle(x,y,p+offset,pair,true);
            double cx=old.first,cy=old.second;
            convert_rope_pair(cx,cy,p+offset,pair,64,a.rope,b.rope);
            near(cx,want.first,2e-6,"FP64 oracle X incl high positions/offset");
            near(cy,want.second,2e-6,"FP64 oracle Y incl high positions/offset");
            // Tests the actual half-rounding boundaries independently using the compiler's IEEE half.
#if defined(__GNUC__) && !defined(__CUDACC__)
            double hx=(float)(_Float16)old.first,hy=(float)(_Float16)old.second;
            convert_rope_pair(hx,hy,p+offset,pair,64,a.rope,b.rope);
            near((float)(_Float16)hx,(float)(_Float16)want.first,0.001,"FP16 oracle X");
            near((float)(_Float16)hy,(float)(_Float16)want.second,0.001,"FP16 oracle Y");
#endif
            auto q=oracle(dist(random),dist(random),262144+offset,pair,true);
            expected_score+=(q.first*want.first+q.second*want.second)/16;
            converted_score+=(q.first*cx+q.second*cy)/16;
            // Known correction-window regions, independently calculated above: 14..22.
            ++groups[pair<14?0:pair>22?2:1];
        }
        near(converted_score,expected_score,2e-6,"attention score with target Q and score scaling once");
        expected_scores.push_back(expected_score);converted_scores.push_back(converted_score);values.push_back(dist(random));
    }
    auto w=softmax(expected_scores),got=softmax(converted_scores);
    near(std::inner_product(w.begin(),w.end(),values.begin(),0.0),
         std::inner_product(got.begin(),got.end(),values.begin(),0.0),2e-6,"attention output");
    check(groups[0]&&groups[1]&&groups[2],"all three frequency regions exercised");
}
SavedConversation fixture() {
    SavedConversation c;
    c.geometry={2048,4,4,128,16,32,4,4096,4096,24,2,256,4,128,4,1,512,512};
    c.layer_hi=4;c.live.ids={1,2,3,4,5};
    c.live.gdn.assign(4096,71);c.live.ple.assign(32,83);
    c.live.dead.resize(128*4);c.live.tails.resize(3*128*4);c.live.block_pos.resize(4);
    float row[128]; std::fill(std::begin(row),std::end(row),0.25f);
    std::memcpy(c.live.dead.data(),row,sizeof(row));
    c.checkpoints.push_back(c.live);c.checkpoints.back().ids.resize(3);
    ConversationKv kv;kv.format=0;kv.cells=8;kv.heads=2;kv.head_dim=256;kv.page_size=4;kv.pooled_rows=2;kv.idx_dim=128;
    kv.k.resize(8*1024);kv.v.resize(8*1024,71);kv.pooled.resize(2*128*4);
    kv.k.visit(0,kv.k.size(),[](uint8_t* data,size_t n,size_t){
        for(size_t i=0;i<n;i+=2) {uint16_t h=0x3400;std::memcpy(data+i,&h,2);}return true;
    });
    kv.pooled.visit(0,kv.pooled.size(),[&](uint8_t* data,size_t n,size_t){for(size_t i=0;i<n;i+=4)std::memcpy(data+i,row,4);return true;});
    c.kv.push_back(std::move(kv));return c;
}
void test_session_conversion() {
    auto a=profile(false),b=profile(true);auto c=fixture(),old=c;
    std::string e;
    check(migrate_rope_cache_to_yarn4(c,a,b,e),e.c_str());
    check(c.kv[0].v==old.kv[0].v && c.live.gdn==old.live.gdn && c.live.tails==old.live.tails &&
          c.live.ple==old.live.ple && c.live.ids==old.live.ids,"values recurrent state raw tails canonical IDs unchanged");
    for(size_t p=0;p<5;++p) for(size_t head=0;head<2;++head) {
        size_t row=((p/4*2+head)*4+p%4)*512;
        uint16_t before[256],after[256];old.kv[0].k.read(before,row,512);c.kv[0].k.read(after,row,512);
        check(std::memcmp(before+64,after+64,192*2)==0,"nonrotary bytes unchanged, paged two KV heads");
        for(int pair=0;pair<32;++pair) {
            auto source_raw=oracle(0.25,0.25,-(int64_t)p,pair,false);
            auto want=oracle(source_raw.first,source_raw.second,p,pair,true);
            near(strata::fp16_to_fp32(after[pair]),want.first,0.001,"full session FP16 X");
            near(strata::fp16_to_fp32(after[pair+32]),want.second,0.001,"full session FP16 Y");
        }
    }
    check(!migrate_rope_cache_to_yarn4(c,a,b,e),"duplicate rejected");
    for(int invalid=0;invalid<9;++invalid) {
        auto bad=old;auto s=a,t=b;
        if(invalid==0)t.model_identity++;
        if(invalid==1)s.rope.type=RopeScalingType::YaRN;
        if(invalid==2)bad.kv[0].format=1;
        if(invalid==3)bad.kv[0].page_size=0;
        if(invalid==4)t.config.cvec=7;
        if(invalid==5)s.config.kv=t.config.kv="bf16";
        if(invalid==6)s.full_model_identity=false;
        if(invalid==7)bad.live.imgs.push_back({0,7});
        if(invalid==8)s.config.switches.clear();
        auto unchanged=bad;
        check(!migrate_rope_cache_to_yarn4(bad,s,t,e),"invalid metadata/layout/format refused");
        check(bad.kv[0].k==unchanged.kv[0].k&&bad.live.dead==unchanged.live.dead,"failure atomic");
    }
    for(int format : {1,2,3,17}) {
        auto quantized=old; quantized.kv[0].format=format;
        check(!migrate_rope_cache_to_yarn4(quantized,a,b,e),"each native quantized KV representation explicitly rejected");
    }
    // Fail after main keys have been staged: no converted bytes may escape.
    auto nonfinite = old;
    const float nan = std::numeric_limits<float>::quiet_NaN();
    nonfinite.kv[0].pooled.visit(0, sizeof(nan), [&](uint8_t* p, size_t n, size_t) {
        std::memcpy(p, &nan, n); return true;
    });
    const auto unchanged = nonfinite;
    check(!migrate_rope_cache_to_yarn4(nonfinite,a,b,e), "nonfinite late index conversion refused");
    check(nonfinite.kv[0].k==unchanged.kv[0].k && nonfinite.kv[0].pooled==unchanged.kv[0].pooled &&
          nonfinite.live.dead==unchanged.live.dead && nonfinite.rope_migration==unchanged.rope_migration,
          "failure after staging leaves original keys and provenance intact");
    auto path=std::filesystem::temp_directory_path()/"strata-rope-migration-test.sess";
    size_t bytes=0;SavedConversation loaded;SessionFileIdentity id{123,session_config_fingerprint(b.config)};
    check(session_file_write(path.string(),c,id,bytes,e),e.c_str());
    check(session_file_read(path.string(),id,loaded,bytes,e),e.c_str());
    check(loaded.rope_migration==c.rope_migration&&loaded.kv[0].k==c.kv[0].k,"saved migration provenance and keys roundtrip");
    check(!migrate_rope_cache_to_yarn4(loaded,a,b,e),"duplicate after reload rejected");
    std::filesystem::remove(path);
}
void test_mtp_session_conversion() {
    for (int64_t window : {0, 32768}) {
        auto s=profile(false),t=profile(true);
        s.config.mtp_window=t.config.mtp_window=window;
        auto c=fixture();
        auto draft=c.kv.front();
        draft.pooled.resize(0);draft.pooled_rows=0;
        c.kv.push_back(std::move(draft));
        auto old=c;
        std::string e;
        check(migrate_rope_cache_to_yarn4(c,s,t,e),e.c_str());
        check(c.kv.back().v==old.kv.back().v,"MTP values unchanged");
        check(c.live.dead.size()==old.live.dead.size() && c.kv.back().pooled.empty(),
              "dense MTP has no extra main indexer state");
        for (size_t p=0;p<c.kv.back().cells;++p) for(size_t h=0;h<2;++h) {
            const size_t row=((p/4*2+h)*4+p%4)*512;
            uint16_t before[256],after[256];
            old.kv.back().k.read(before,row,512);c.kv.back().k.read(after,row,512);
            if(p>=old.live.ids.size()-1) {
                check(std::memcmp(before,after,512)==0,"uncomputed MTP boundary and padding untouched");
                continue;
            }
            check(std::memcmp(before+64,after+64,384)==0,"MTP nonrotary bytes unchanged");
            for(int pair=0;pair<32;++pair) {
                auto raw=oracle(0.25,0.25,-(int64_t)p,pair,false);
                auto want=oracle(raw.first,raw.second,p,pair,true);
                near(strata::fp16_to_fp32(after[pair]),want.first,0.001,"MTP absolute-position X oracle");
                near(strata::fp16_to_fp32(after[pair+32]),want.second,0.001,"MTP absolute-position Y oracle");
            }
        }
        auto bad=old;
        uint16_t nan=0x7e00;
        bad.kv.back().k.visit(0,2,[&](uint8_t* p,size_t n,size_t){std::memcpy(p,&nan,n);return true;});
        const auto original=bad;
        check(!migrate_rope_cache_to_yarn4(bad,s,t,e),"nonfinite draft key fails conversion");
        check(bad.kv.front().k==original.kv.front().k && bad.kv.back().k==original.kv.back().k &&
              bad.rope_migration==original.rope_migration,"draft failure leaves main and draft state atomic");
        bad=old;bad.kv.pop_back();
        check(!migrate_rope_cache_to_yarn4(bad,s,t,e),"MTP configuration requires saved draft state");
        bad=old;bad.kv.back().format=1;
        check(!migrate_rope_cache_to_yarn4(bad,s,t,e),"quantized draft refused");
        auto changed=t;changed.config.mtp_window=window+1;bad=old;
        check(!migrate_rope_cache_to_yarn4(bad,s,changed,e),"draft window mismatch refused");
        const auto path=std::filesystem::temp_directory_path()/"strata-rope-mtp-test.sess";
        size_t bytes=0;SavedConversation loaded;SessionFileIdentity id{123,session_config_fingerprint(t.config)};
        check(session_file_write(path.string(),c,id,bytes,e),e.c_str());
        check(session_file_read(path.string(),id,loaded,bytes,e),e.c_str());
        check(loaded.kv.back().k==c.kv.back().k && loaded.rope_migration==c.rope_migration,
              "MTP migration provenance and draft keys survive reload");
        check(!migrate_rope_cache_to_yarn4(loaded,s,t,e),"reloaded MTP migration cannot repeat");
        std::filesystem::remove(path);
    }
}
}
int main() {
    test_ordinary_rope_to_yarn4_key_conversion();test_session_conversion();test_mtp_session_conversion();
    std::printf("rope_cache_migration_test: %d checks passed\n",checks);
}
