"""Repeated consumed checkpoints occupy one unpinned cache entry."""
import unittest
import test_r03_asan


class DuplicateConsumedCacheTests(unittest.TestCase):
    build_and_run=test_r03_asan.HostViewSafetyTests.build_and_run

    def test_duplicate_consumed_history_replaces_unpinned_entry(self):
        result=self.build_and_run(r'''
#include "strata/core/conversation_cache.hpp"
int main() {
    using namespace strata::core;
    ConversationCache cache(1<<20,4);
    SavedConversation image; image.live.ids={1,2,3}; image.cvec=false;
    image.kv.resize(1); image.kv[0].k.resize(64,7); image.kv[0].k.freeze();
    image.next_logits={1.0f,2.0f};
    auto duplicate=image;
    duplicate.next_logits={3.0f,4.0f};
    if (!cache.put(std::move(image)) || !cache.put(std::move(duplicate))) return 1;
    if (cache.size()!=1) return 2;
    const auto* hit=cache.exact(std::vector<int32_t>{1,2,3},{},false);
    if (!hit || hit->next_logits[0]!=3.0f) return 3;
    if (cache.storage_metrics().allocations!=1 || cache.storage_metrics().owners!=1) return 4;
}
''')
        self.assertEqual(result.returncode,0,'Duplicate consumed histories must replace their unpinned checkpoint')


if __name__=='__main__': unittest.main(verbosity=2)
