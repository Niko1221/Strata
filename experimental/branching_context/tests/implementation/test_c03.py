"""Adversarial native-cache retention and canonical-history eviction checks."""
import shutil
import subprocess
import tempfile
from pathlib import Path
import unittest

from context_time_travel.checkpoint_cache import CheckpointCache
from context_time_travel.store import Store


class BackgroundForkRetentionTests(unittest.TestCase):
    def test_c03_invariant(self):
        compiler = shutil.which("g++")
        self.assertIsNotNone(compiler, "the native cache policy test requires the Linux C++ compiler")
        repository = Path(__file__).resolve().parents[4]
        source = r'''
#include "strata/core/conversation_cache.hpp"
#include <utility>
using namespace strata::core;
int main() {
    ConversationCache cache(1u << 20, 4);
    SavedConversation root;
    ConversationCheckpoint pinned;
    pinned.ids = {1, 2, 3};
    pinned.pinned = true;
    root.live.ids = {1, 2, 3};
    root.checkpoints.push_back(pinned);
    if (!cache.put(std::move(root))) return 1;

    SavedConversation child;
    ConversationCheckpoint inherited = pinned;
    inherited.pinned = false;
    child.live.ids = {1, 2, 3, 9};
    child.checkpoints.push_back(inherited);
    if (!cache.put(std::move(child))) return 2;
    if (cache.size() != 2 || cache.superseded() != 0) return 3;
    if (!cache.evict_oldest() || cache.size() != 1) return 4;
    if (cache.evict_oldest() || cache.size() != 1) return 5;
    return 0;
}
'''
        with tempfile.TemporaryDirectory() as temporary:
            executable = Path(temporary) / "conversation-cache-retention"
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
                "a background child must not supersede or evict the pinned main-prefix image",
            )

        with tempfile.TemporaryDirectory() as temporary:
            store = Store(temporary)
            root = store.create_root("model-contract", [1, 2], {"status": "idle"})
            child = store.append(root, [3], {"status": "idle"})
            cache = CheckpointCache(1024, 4)
            cache.clear()
            self.assertEqual(store.lineage(root)[1], [1, 2])
            self.assertEqual(store.lineage(child)[1], [1, 2, 3])

            before = {path.name for path in store.objects.iterdir()}

            def cancel_before_publish(step):
                if step == "before_snapshot_publish":
                    raise RuntimeError("cancelled child before canonical publication")

            store.fault_hook = cancel_before_publish
            with self.assertRaisesRegex(RuntimeError, "cancelled child"):
                store.append(child, [4], {"status": "idle"})
            self.assertEqual({path.name for path in store.objects.iterdir()}, before)
            self.assertEqual(store.lineage(root)[1], [1, 2])
            self.assertEqual(store.lineage(child)[1], [1, 2, 3])


if __name__ == "__main__":
    unittest.main()
