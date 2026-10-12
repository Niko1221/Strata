"""Admission must cover the actual allocations of published host-buffer writes."""
import unittest

import test_r03_asan


class HostAdmissionTests(unittest.TestCase):
    build_and_run = test_r03_asan.HostViewSafetyTests.build_and_run
    def test_readonly_visit_keeps_published_payload(self):
        result = self.build_and_run(r'''
#include "strata/core/conversation_buffer.hpp"
int main() {
    strata::core::ConversationBuffer root;
    root.resize(4096, 7); root.freeze();
    auto child=root;
    const uint8_t* before=nullptr;
    const auto copies=strata::core::ConversationBuffer::copied_bytes();
    child.visit(0, 1, [&](const uint8_t* p, size_t, size_t) {before=p; return true;});
    const uint8_t* ancestor=nullptr;
    root.visit(0, 1, [&](const uint8_t* p, size_t, size_t) {ancestor=p; return true;});
    if (ancestor != before || strata::core::ConversationBuffer::copied_bytes() != copies) return 1;
}
''')
        self.assertEqual(result.returncode, 0, "Read-only visit must not detach published payload")

    def test_standalone_admission_and_cache_unique_storage(self):
        result = self.build_and_run(r'''
#include "strata/core/conversation_cache.hpp"
int main() {
    using namespace strata::core;
    SavedConversation image; image.live.ids={1};
    ConversationKv kv; kv.k.resize(4096, 7); kv.v=kv.k;
    const size_t charge=kv.k.bytes()+kv.v.bytes();
    if (kv.bytes()!=charge) return 1;
    image.kv.push_back(kv);
    const size_t standalone=image.bytes();
    ConversationCache cache(1<<20, 4);
    if (!cache.put(std::move(image))) return 2;
    const auto metrics=cache.storage_metrics();
    if (metrics.payload_bytes!=4096 || metrics.allocations!=1) return 3;
    if (cache.bytes()+4096!=standalone) return 4;
}
''')
        self.assertEqual(result.returncode, 0,
                         "Standalone capacity admission must preserve its contract; cache storage counts actual aliases once")

    def test_published_rewrite_fits_admission(self):
        result = self.build_and_run(r'''
#include "strata/core/conversation_buffer.hpp"
int main() {
    for (size_t target : {size_t(2048), size_t(4096), size_t(8192)}) {
        strata::core::ConversationBuffer root;
        root.resize(4096, 7);
        root.freeze();
        auto child = root;
        const size_t admitted = child.allocation_peak(target);
        child.resize(target);
        child.visit(17, target-17, [](uint8_t* p, size_t n, size_t) {
            for (size_t i=0; i<n; ++i) p[i]=9;
            return true;
        });
        if (child.bytes() > admitted) return 1;
        uint8_t retained=0;
        if (!root.read(&retained, 17, 1) || retained != 7) return 2;
    }
}
''')
        self.assertEqual(result.returncode, 0,
                         "Published growth/shrink/rewrite exceeded admission: " + result.stderr)


if __name__ == "__main__":
    unittest.main(verbosity=2)
