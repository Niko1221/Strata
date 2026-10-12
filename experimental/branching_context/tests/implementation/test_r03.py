"""Unique shared-storage budget and eviction checks for R03."""
import shutil
import subprocess
import tempfile
from pathlib import Path
import unittest


class SharedStorageAccountingTests(unittest.TestCase):
    def test_r03_invariant(self):
        compiler = shutil.which("g++")
        self.assertIsNotNone(compiler, "the native cache accounting probe requires g++")
        repository = Path(__file__).resolve().parents[4]
        source = r'''
#include "strata/core/conversation_cache.hpp"
#include <cstdint>
#include <iostream>
#include <utility>
using namespace strata::core;

struct EmptyMetrics {
    size_t payload_bytes = 0, allocations = 0, owners = 0, copied_bytes = 0;
};
template<class Cache>
auto metrics(const Cache& cache, int) -> decltype(cache.storage_metrics()) {
    return cache.storage_metrics();
}
template<class Cache>
EmptyMetrics metrics(const Cache&, long) { return {}; }

int main() {
    SavedConversation root;
    root.live.ids = {1, 2, 3};
    ConversationKv kv;
    kv.k.resize(4096, 5);
    kv.k.freeze();
    root.kv.push_back(std::move(kv));
    SavedConversation child = root;
    child.live.ids.push_back(4);

    const size_t root_bytes = root.bytes();
    const size_t child_bytes = child.bytes();
    const size_t shared_payload = child.kv[0].k.shared_bytes();
    if (shared_payload != 4096) return 1;
    const size_t unique_total = root_bytes + child_bytes - shared_payload;

    ConversationCache cache(1u << 20, 4);
    if (!cache.put(std::move(root)) || !cache.put(std::move(child))) return 2;
    if (cache.bytes() != unique_total) return 3;
    const auto both = metrics(cache, 0);
    if (both.payload_bytes != 4096 || both.allocations != 1 || both.owners != 2) return 6;
    if (!cache.evict_oldest() || cache.size() != 1 || cache.bytes() != child_bytes) return 4;
    const auto one = metrics(cache, 0);
    if (one.payload_bytes != 4096 || one.allocations != 1 || one.owners != 1) return 7;
    if (cache.clear() != 1 || cache.bytes() != 0) return 5;
    const auto none = metrics(cache, 0);
    if (none.payload_bytes || none.allocations || none.owners) return 8;
    std::cout << "R03_METRIC shared_payload_bytes=" << shared_payload
              << " deduplicated_cache_bytes=" << unique_total
              << " owners_before_evict=" << both.owners
              << " owners_after_evict=" << one.owners
              << " bytes_after_evict=" << child_bytes
              << " bytes_after_clear=0\n";
    return 0;
}
'''
        with tempfile.TemporaryDirectory() as temporary:
            executable = Path(temporary) / "conversation-unique-accounting"
            built = subprocess.run(
                [compiler, "-std=c++20", "-I", str(repository / "include"), "-x", "c++", "-", "-o", str(executable)],
                input=source,
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertEqual(built.returncode, 0, built.stderr)
            result = subprocess.run([str(executable)], capture_output=True, check=False)
            self.assertEqual(
                result.returncode,
                0,
                result.stdout.decode("utf-8", errors="replace")
                or "shared payload must be charged once and released when the final owner is evicted",
            )
            print(result.stdout.decode("utf-8", errors="replace").strip())


if __name__ == "__main__":
    unittest.main()
