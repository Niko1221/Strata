"""A consumed canonical frontier requires an exact native cache match."""
import unittest
import test_r03_asan


class ExactNativeCacheTests(unittest.TestCase):
    build_and_run = test_r03_asan.HostViewSafetyTests.build_and_run

    def test_consumed_frontier_exact_cache_identity(self):
        result = self.build_and_run(r'''
#include "strata/core/conversation_cache.hpp"
using namespace strata::core;
template<class Cache>
auto exact(const Cache& cache, const std::vector<int32_t>& ids,
           const std::vector<ConversationImageKey>& images, bool cvec, int)
    -> decltype(cache.exact(ids, images, cvec)) { return cache.exact(ids, images, cvec); }
template<class Cache>
const SavedConversation* exact(const Cache&, const std::vector<int32_t>&,
                              const std::vector<ConversationImageKey>&, bool, long) { return nullptr; }
int main() {
    ConversationCache cache(1<<20, 4);
    SavedConversation image; image.live.ids={41,7,9,13}; image.cvec=false;
    image.live.imgs={{1,99}}; image.kv.resize(1); image.kv[0].k.resize(64,17);
    image.kv[0].k.freeze();
    if (!cache.put(std::move(image))) return 1;
    const std::vector<int32_t> ids{41,7,9,13};
    const std::vector<ConversationImageKey> images{{1,99}};
    const auto before=cache.storage_metrics();
    const auto* hit=exact(cache,ids,images,false,0);
    if (!hit || hit->live.ids!=ids || hit->live.imgs!=images) return 2;
    if (exact(cache,{88,7,9,13},images,false,0)) return 3;
    if (exact(cache,{41,7,9},images,false,0)) return 4;
    if (exact(cache,{41,7,9,13,20},images,false,0)) return 5;
    if (exact(cache,ids,{{1,98}},false,0) || exact(cache,ids,images,true,0)) return 6;
    if (cache.best(ids,images,false).tokens!=0) return 7; // legacy request boundary unchanged
    const auto after=cache.storage_metrics();
    if (after.payload_bytes!=before.payload_bytes || after.copied_bytes!=before.copied_bytes) return 8;
}
''')
        self.assertEqual(result.returncode, 0,
                         "Native cache must match the complete consumed frontier without suffix/prefix aliasing")


if __name__ == "__main__":
    unittest.main(verbosity=2)
