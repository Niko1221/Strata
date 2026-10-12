"""Direct host-buffer ownership and prefix-identity checks for R01."""
import shutil
import subprocess
import tempfile
from pathlib import Path
import unittest


class HostPrefixSharingTests(unittest.TestCase):
    def test_r01_invariant(self):
        compiler = shutil.which("g++")
        self.assertIsNotNone(compiler, "the native host ownership probe requires g++")
        repository = Path(__file__).resolve().parents[4]
        source = r'''
#include "strata/core/conversation_cache.hpp"
#include <cstdint>
#include <iostream>
#include <utility>
#include <vector>
using namespace strata::core;

template<class Buffer>
const uint8_t* first_byte(const Buffer& value) {
    const uint8_t* result = nullptr;
    if (!value.visit(0, 1, [&](const uint8_t* data, size_t, size_t) {
        result = data;
        return true;
    })) return nullptr;
    return result;
}

template<class Buffer>
auto shared_bytes(const Buffer& value, int) -> decltype(value.shared_bytes()) {
    return value.shared_bytes();
}
template<class Buffer>
size_t shared_bytes(const Buffer&, long) { return 0; }
template<class Buffer>
auto copied_bytes(int) -> decltype(Buffer::copied_bytes()) {
    return Buffer::copied_bytes();
}
template<class Buffer>
size_t copied_bytes(long) { return 0; }

int main() {
    ConversationKv root;
    root.k.resize(4096, 7);
    ConversationKv a = root;
    ConversationKv b = root;
    if (first_byte(root.k) != first_byte(a.k) || first_byte(root.k) != first_byte(b.k)) return 1;
    const size_t shared_before = shared_bytes(root.k, 0);
    const size_t copied_before = copied_bytes<ConversationBuffer>(0);
    if (shared_before != 4096) return 8;

    // A writable fork must detach its changed page and leave both ancestors intact.
    if (!a.k.visit(0, 1, [](uint8_t* data, size_t, size_t) { data[0] = 9; return true; })) return 2;
    if (first_byte(root.k) == first_byte(a.k) || first_byte(root.k) != first_byte(b.k)) return 3;
    const size_t copied_delta = copied_bytes<ConversationBuffer>(0) - copied_before;
    if (copied_delta != 4096 || shared_bytes(a.k, 0) != 0 || shared_bytes(root.k, 0) != 4096) return 9;
    std::cout << "R01_METRIC shared_prefix_bytes=" << shared_before
              << " cow_copied_bytes=" << copied_delta << '\n';
    uint8_t original = 0, changed = 0;
    if (!root.k.read(&original, 0, 1) || !a.k.read(&changed, 0, 1)) return 4;
    if (original != 7 || changed != 9) return 5;

    // Prefix identity, not just a matching final block, controls cache reuse.
    ConversationCache cache(1u << 20, 4);
    SavedConversation image;
    image.live.ids = {10, 11, 13};
    image.checkpoints.push_back(image.live);
    if (!cache.put(std::move(image))) return 6;
    auto same_prefix = cache.best(std::vector<int32_t>{10, 11, 13, 20}, {}, true);
    auto other_prefix = cache.best(std::vector<int32_t>{99, 11, 13, 20}, {}, true);
    if (same_prefix.tokens != 3 || other_prefix.tokens != 0) return 7;
    return 0;
}
'''
        with tempfile.TemporaryDirectory() as temporary:
            executable = Path(temporary) / "conversation-host-sharing"
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
                or "derived host prefixes must share immutable storage, detach on write, and match exact prefix identity",
            )
            print(result.stdout.decode("utf-8", errors="replace").strip())


if __name__ == "__main__":
    unittest.main()
