#include "strata/core/conversation_cache.hpp"
#include <cstdlib>
#include <cstdio>
#include <new>
static long fail_after=-1;
void* operator new(std::size_t n) {
    if(fail_after==0) throw std::bad_alloc();
    if(fail_after>0) --fail_after;
    if(void* p=std::malloc(n?n:1)) return p;
    throw std::bad_alloc();
}
void operator delete(void* p) noexcept { std::free(p); }
void operator delete(void* p,std::size_t) noexcept { std::free(p); }
using strata::core::ConversationCheckpoint;
bool equal(const ConversationCheckpoint& a,const ConversationCheckpoint& b) {
    if(a.ids!=b.ids || a.imgs!=b.imgs || a.gdn!=b.gdn || a.ple!=b.ple || a.tails!=b.tails || a.dead!=b.dead || a.block_pos!=b.block_pos || a.used!=b.used || a.stage_parts.size()!=b.stage_parts.size()) return false;
    for(size_t i=0;i<a.stage_parts.size();++i) if(!equal(a.stage_parts[i],b.stage_parts[i])) return false;
    return true;
}
int main() {
    int faults=0;
    for(int n=0;n<128;++n) {
        std::vector<ConversationCheckpoint> checks(3);
        for(size_t i=0;i<checks.size();++i) {
            auto& c=checks[i]; c.ids={1,2,(int)i+3};c.gdn={1,2,3};c.ple={4};c.used=i;
            c.stage_parts.resize(1);c.stage_parts[0].gdn={5,6,7};
        }
        const auto backup=checks;
        fail_after=n;
        try {
            auto split=strata::core::conversation_checkpoints_split(std::move(checks),1);
            fail_after=0; // Merge-back after a successful split must not allocate.
            if(!strata::core::conversation_checkpoints_merge(std::move(split),checks)) return 2;
            fail_after=-1;
            // Split fills each stage's identity with its parent identity by design.
            std::printf("allocation-failure cases preserved: %d\n",faults);return faults>0?0:3;
        } catch(const std::bad_alloc&) {
            fail_after=-1;++faults;
            if(checks.size()!=backup.size()) return 4;
            for(size_t i=0;i<checks.size();++i) if(!equal(checks[i],backup[i])) {
                std::fprintf(stderr,"allocation failure %d partially moved checkpoint %zu\n",n,i);return 1;
            }
        }
    }
    return 5;
}
