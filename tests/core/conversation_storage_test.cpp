#include "strata/core/conversation_cache.hpp"
#include <cstdio>
#include <cstdlib>
#include <new>
#include <stdexcept>
#include <random>
#include <chrono>
#include <utility>
using namespace strata::core;
static long fail_after = -1;
void* operator new(std::size_t n) {
    if (fail_after == 0) throw std::bad_alloc();
    if (fail_after > 0) --fail_after;
    if (void* p = std::malloc(n ? n : 1)) return p;
    throw std::bad_alloc();
}
void operator delete(void* p) noexcept { std::free(p); }
void operator delete(void* p, std::size_t) noexcept { std::free(p); }
static int checks;
void check(bool b, const char* why) { ++checks; if (!b) {std::fprintf(stderr,"FAIL: %s\n",why); std::exit(1);} }
void fill(ConversationBuffer& b, size_t n, uint8_t salt) {
    b.resize(n);
    b.visit(0,n,[&](uint8_t* p,size_t m,size_t at){for(size_t i=0;i<m;++i) p[i]=uint8_t((at+i)*31+(at+i)/257+salt);return true;});
}
std::vector<uint8_t> read(const ConversationBuffer& b) {std::vector<uint8_t> v(b.size());check(b.read(v.data(),0,v.size()),"read");return v;}
SavedConversation image(int id) {
    SavedConversation s; s.live.ids={1,2,id,4}; s.live.gdn.assign(2<<20,uint8_t(id));
    s.kv.resize(1);fill(s.kv[0].k,(4<<20)+17,42);
    s.kv[0].k.visit(4<<20,17,[&](uint8_t*p,size_t n,size_t){std::memset(p,id,n);return true;});
    ConversationCheckpoint c;c.ids={1,2};c.gdn.assign(2<<20,9); c.freeze();s.checkpoints.push_back(c); c.ids={1,2,id}; s.checkpoints.push_back(std::move(c));
    s.freeze();return s;
}
int main(){
    const bool on=conversation_dedup_enabled();
    {
        ConversationBytes probe; probe.reserve(4096);
        const size_t peak = conversation_storage_stats().peak.load();
        probe.resize(4096);
        check(conversation_storage_stats().peak.load() == peak, "resize within capacity does not inflate peak");
    }
    {
        ConversationBytes a; a.assign(12345,17);auto* escaped=a.data(); auto b=a;
        escaped[0]=22;check(std::as_const(b)[0]==17,"mutable copy protects escaped pointer");
        a[0]=17;a.freeze();b.freeze();
        check((std::as_const(a).data()==std::as_const(b).data())==on,"independent byte-equal capture shares only on");
        auto c=a;c[100]=19;check(std::as_const(a)[100]==17,"write detach");
        c.resize(6);c.resize(200,3);check(std::as_const(a).size()==12345,"shrink regrow isolation");
        ConversationBytes different;different.assign(12345,18);different.freeze();
        check(std::as_const(different)[0]==18&&std::as_const(a)[0]==17,"hash collision exact comparison");
    }
    check(conversation_storage_stats().payload==0,"final byte ref release");
    {
        ConversationBuffer a;fill(a,(2<<20)+13,7);a.freeze();auto expected=read(a);
        auto b=a; b.visit((1<<20)-3,19,[](uint8_t*p,size_t n,size_t){std::memset(p,222,n);return true;});
        check(read(a)==expected,"cross-block partial-page mutation isolation");
        b.resize((1<<20)+3);b.resize((3<<20)+9,42);check(read(a)==expected,"rewind append preserves donor");
        a={};check(b.size()==(3<<20)+9,"donor eviction holds survivor");
        std::mt19937 rng(42);std::vector<uint8_t> oracle;ConversationBuffer random;
        for(int i=0;i<180;++i){size_t n=rng()%((3<<20)+19);oracle.resize(n,23);random.resize(n,23);
            if(n){size_t at=rng()%n;size_t m=std::min<size_t>(71,n-at);std::fill(oracle.begin()+at,oracle.begin()+at+m,91);
                random.visit(at,m,[](uint8_t*p,size_t c,size_t){std::memset(p,91,c);return true;});}
            random.freeze();auto copy=random;check(read(copy)==oracle,"random append rewind private suffix");}
    }
    check(conversation_storage_stats().payload==0,"buffer final ref release");
    {
        std::vector<SavedConversation> siblings;
        auto t=std::chrono::steady_clock::now();
        for(int i=0;i<20;++i)siblings.push_back(image(i+10));
        size_t logical=0;for(const auto&s:siblings)logical+=s.bytes();
        size_t physical=on?conversation_storage_stats().payload.load():logical;
        std::printf("siblings=20 logical=%zu physical=%zu peak=%zu saved=%zu capture_ms=%.3f\n",logical,physical,
                    conversation_storage_stats().peak.load(),logical>physical?logical-physical:0,
                    std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-t).count());
        if(on)check(physical<logical/2,"physical savings >50 percent");
        auto cp=siblings[0].checkpoints[0];siblings.clear();check(std::as_const(cp.gdn)[0]==9,"checkpoint donor eviction");
        auto cp2=cp;cp2.gdn.data()[0]=77;check(std::as_const(cp.gdn)[0]==9,"checkpoint reuse pool detach");
    }
    check(conversation_storage_stats().payload==0,"siblings final references released");
    if(on){
        ConversationCache cache(28<<20,100);
        for(int i=0;i<22;++i){auto s=image(i+30);check(cache.put(std::move(s)),"pressure insert fits after eviction");check(cache.bytes()<=(28<<20),"unique allocation budget bound");}
        check(cache.evictions()>0,"pressure eviction exercised");
        auto m=cache.best(std::vector<int>{1,2,7},{},true);check(m.tokens==2,"match checkpoint");
        check(cache.best(std::vector<int>{1,2,7},{{0,99}},true).tokens==0,"image identity isolation");
        check(cache.best(std::vector<int>{1,2,7},{},false).tokens==0,"steering identity isolation");
        cache.clear();check(conversation_storage_stats().payload==0,"cache clear releases all payload");
    }
    if(on) {
        auto external=image(7); ConversationCache tiny(1<<20,4);
        check(!tiny.make_room(0), "external retained references prevent false admission");
        check(!tiny.put(std::move(external)), "oversized physical state refused");
    }
    {
        // Copies, const reads and same-size resizes never detach shared storage.
        ConversationBuffer x;fill(x,(3<<20)+5,3);x.freeze();
        const size_t cow=conversation_storage_stats().cow.load();
        auto y=x;std::as_const(y).visit(0,y.size(),[](const uint8_t*,size_t,size_t){return true;});y.resize(y.size());
        check(conversation_storage_stats().cow.load()==cow,"inspection performs no copy-on-write");
        ConversationBytes a;a.assign(1<<20,3);a.freeze();auto b=a;b.resize(100);
        check(std::as_const(a).size()==(1<<20)&&std::as_const(b)[99]==3,"shrinking a shared payload keeps the donor");
        if(on)check(b.capacity()<4096,"shrinking a shared payload copies only the kept prefix");
        check(x.allocation_peak((3<<20)+105,(3<<20)+5)==(on?105+4*sizeof(ConversationBytes)+x.directory_bytes():x.allocation_peak((3<<20)+105)),
              "retained prefix estimate: dedup counts only rewritten blocks");
    }
    if(on){
        // Reproduces the park order in generate.cpp with unique payloads:
        // a taken image is held, the outgoing one carries retained K/V.
        auto uniq=[](ConversationBuffer& b,size_t n,uint64_t seed){
            b.resize(n);
            b.visit(0,n,[&](uint8_t*p,size_t m,size_t at){for(size_t i=0;i<m;++i){uint64_t z=(at+i+1)*0x9e3779b97f4a7c15ULL^seed*0xbf58476d1ce4e5b9ULL;
                z^=z>>29;z*=0x94d049bb133111ebULL;p[i]=uint8_t(z>>56);}return true;});
        };
        auto conv=[&](int id,size_t n,uint64_t seed){SavedConversation s;s.live.ids={1,id};s.kv.resize(1);uniq(s.kv[0].k,n,seed);s.freeze();return s;};
        constexpr size_t M=1<<20;
        for(auto [x,others]:{std::pair<size_t,int>{8,4},{12,4},{16,2}}){
            ConversationCache cache(64*M,64);
            for(int i=0;i<others;++i)check(cache.put(conv(10+i,8*M,10+i)),"fixture entry");
            check(cache.put(conv(1,x*M,1)),"fixture taken entry");
            auto incoming=cache.take(cache.size()-1);const size_t held=incoming.bytes();
            auto outgoing=conv(2,x*M,2);ConversationKvReuse reuse{std::move(outgoing.kv),3,3,{}};
            const size_t estimate=reuse.kv[0].k.allocation_peak(x*M+65536,x*M);
            const size_t before=cache.size();
            check(cache.make_room(estimate,held)&&cache.size()==size_t(before),"held and retained storage are not charged twice");
            SavedConversation parked;parked.live.ids={1,2};parked.kv=std::move(reuse.kv);
            parked.kv[0].k.resize(x*M+65536);parked.freeze(); // the capture appends to the retained K/V
            check(cache.put(std::move(parked),held)&&cache.size()==size_t(before+1),"put does not charge held storage twice");
        }
        {
            ConversationCache cache(64*M,64);
            for(int i=0;i<4;++i)check(cache.put(conv(20+i,8*M,20+i)),"fixture entry");
            {
                ConversationBuffer active;uniq(active,60*M,99);active.freeze();
                check(!cache.make_room(8*M)&&cache.size()==4,"an image that cannot fit evicts nothing");
            }
            ConversationBuffer active;uniq(active,30*M,98);active.freeze();
            check(cache.make_room(8*M)&&cache.size()==3&&cache.evictions()==1,"only the needed eviction runs");
        }
        {
            // Two entries share one payload: evicting one releases nothing.
            ConversationCache cache(64*M,64);
            check(cache.put(conv(30,16*M,7))&&cache.put(conv(31,16*M,7))&&cache.put(conv(32,8*M,8)),"shared fixture");
            ConversationBuffer active;uniq(active,40*M,97);active.freeze();
            check(cache.make_room(8*M)&&cache.size()==1&&cache.best(std::vector<int>{1,32,0},{},true).tokens==2,
                  "shared payload is released only with its last owner");
        }
        {
            ConversationCache cache(64*M,64);
            check(cache.put(conv(40,24*M,40))&&cache.put(conv(41,8*M,41)),"guard fixture");
            ConversationBuffer active;uniq(active,30*M,96);active.freeze();
            cache.guard(cache.best(std::vector<int>{1,40,0},{},true).serial);
            check(!cache.make_room(16*M)&&cache.guard_blocked()&&cache.size()==2,"guarded donor blocks without evicting");
            cache.unguard();
            check(cache.make_room(16*M)&&cache.size()==1&&cache.best(std::vector<int>{1,41,0},{},true).tokens==2,"unguarded: the oldest entry alone is evicted");
        }
        {
            // A refused replacement keeps the superseded copy; an admitted one replaces it.
            auto chain=[&](int id,size_t n){SavedConversation s=conv(id,n,id);ConversationCheckpoint c;c.ids={1};s.checkpoints.push_back(std::move(c));return s;};
            ConversationCache cache(3*M,8);
            check(cache.put(chain(50,M)),"superseded fixture");
            check(!cache.put(chain(51,4*M))&&cache.size()==1&&cache.superseded()==0,"refused replacement keeps the superseded copy");
            check(cache.put(chain(52,2*M))&&cache.size()==1&&cache.superseded()==1&&cache.bytes()<=3*M,
                  "admitted replacement drops the superseded copy within budget");
        }
        {
            // The plan counts wrapper and index-node allowances eviction releases.
            const size_t base=ConversationCache(1,1).bytes(); // retained index buckets
            ConversationCache cache(base+4096,8);
            check(cache.put([]{SavedConversation s;s.live.ids={1,60};s.live.gdn.assign(1024,60);return s;}()),"boundary fixture");
            check(cache.make_room(4096-64)&&cache.size()==0&&cache.bytes()<=base+64,"boundary image fits after eviction");
        }
        check(ConversationBytes::index_entries()==0,"index holds no released payloads");
    }
    for(int budget=0;budget<45;++budget){
        ConversationBuffer donor;fill(donor,(1<<20)+3,7);donor.freeze();auto original=read(donor);
        try{auto copy=donor;fail_after=budget;copy.resize((3<<20)+7);copy.visit(0,copy.size(),[](uint8_t*p,size_t n,size_t){std::memset(p,5,n);return true;});copy.freeze();}
        catch(const std::bad_alloc&){}fail_after=-1;
        check(read(donor)==original,"allocation failure leaves donor usable");
    }
    check(conversation_storage_stats().payload==0,"no failure payload leak");
    check(conversation_storage_stats().objects==0,"no failure wrapper leak");
    check(conversation_storage_stats().logical==0,"no logical counter leak");
    check(ConversationBytes::index_entries()==0,"no index entry leak");
    std::printf("storage_test mode=%d checks=%d PASS\n",on,checks);
}
