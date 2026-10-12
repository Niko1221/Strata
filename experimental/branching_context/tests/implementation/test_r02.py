"""Published-page immutability and bounded COW checks for R02."""
import shutil
import subprocess
import tempfile
from pathlib import Path
import unittest


class PublishedTailIsolationTests(unittest.TestCase):
    def test_r02_invariant(self):
        compiler = shutil.which("g++")
        self.assertIsNotNone(compiler, "the native host ownership probe requires g++")
        repository = Path(__file__).resolve().parents[4]
        source = r'''
#include "strata/core/conversation_buffer.hpp"
#include <cstdint>
#include <iostream>
using namespace strata::core;

template<class Buffer>
auto publish(Buffer& value, int) -> decltype(value.freeze(), bool()) {
    value.freeze();
    return true;
}
template<class Buffer>
bool publish(Buffer&, long) { return false; }
template<class Buffer>
auto copied_bytes(int) -> decltype(Buffer::copied_bytes()) {
    return Buffer::copied_bytes();
}
template<class Buffer>
size_t copied_bytes(long) { return 0; }

int main() {
    ConversationBuffer root;
    root.resize(1u << 20, 3);
    if (!publish(root, 0)) return 1;

    // A published segment must detach even when this buffer is its only owner.
    size_t before = copied_bytes<ConversationBuffer>(0);
    if (!root.visit(4093, 17, [](uint8_t* p, size_t n, size_t) {
        for (size_t i = 0; i < n; ++i) p[i] = 9;
        return true;
    })) return 2;
    const size_t unique_copy = copied_bytes<ConversationBuffer>(0) - before;
    if (unique_copy != 17) return 3;

    ConversationBuffer child = root;
    before = copied_bytes<ConversationBuffer>(0);
    if (!child.visit(65531, 23, [](uint8_t* p, size_t n, size_t) {
        for (size_t i = 0; i < n; ++i) p[i] = 11;
        return true;
    })) return 4;
    const size_t tail_copy = copied_bytes<ConversationBuffer>(0) - before;
    if (tail_copy != 23) return 5;

    uint8_t parent_byte = 0, child_byte = 0;
    if (!root.read(&parent_byte, 65531, 1) || !child.read(&child_byte, 65531, 1)) return 6;
    if (parent_byte != 3 || child_byte != 11) return 7;
    std::cout << "R02_METRIC immutable_unique_copy=" << unique_copy
              << " private_tail_copy=" << tail_copy << '\n';
    return 0;
}
'''
        with tempfile.TemporaryDirectory() as temporary:
            executable = Path(temporary) / "conversation-published-tail"
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
                or "published pages must remain immutable and COW bytes must equal the modified tail",
            )
            print(result.stdout.decode("utf-8", errors="replace").strip())


if __name__ == "__main__":
    unittest.main()
