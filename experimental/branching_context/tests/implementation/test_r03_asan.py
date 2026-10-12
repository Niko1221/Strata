"""Actual native sanitizer and allocation-failure checks for shared host views."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest


class HostViewSafetyTests(unittest.TestCase):
    def build_and_run(self, source, sanitizer=False):
        compiler = shutil.which("g++")
        self.assertIsNotNone(compiler, "A real native C++ compiler is required")
        repository = Path(__file__).resolve().parents[4]
        with tempfile.TemporaryDirectory() as directory:
            cpp = Path(directory) / "probe.cpp"
            executable = Path(directory) / "probe"
            cpp.write_text(source, encoding="utf-8")
            flags = ["-std=c++20", "-g", "-O0"]
            if sanitizer:
                flags += ["-fsanitize=address,undefined", "-fno-omit-frame-pointer"]
            command = [compiler, *flags, "-I", str(repository / "include"), str(cpp), "-o", str(executable)]
            built = subprocess.run(command, capture_output=True, text=True, timeout=60)
            self.assertEqual(built.returncode, 0, built.stderr)
            print(json.dumps({"kind": "native-build", "flags": flags,
                "source_sha256": hashlib.sha256(cpp.read_bytes()).hexdigest(),
                "header_sha256": hashlib.sha256((repository / "include/strata/core/conversation_buffer.hpp").read_bytes()).hexdigest(),
                "binary_sha256": hashlib.sha256(executable.read_bytes()).hexdigest()}))
            environment = dict(os.environ, ASAN_OPTIONS="detect_leaks=1", UBSAN_OPTIONS="halt_on_error=1")
            return subprocess.run([str(executable)], capture_output=True, text=True,
                                  env=environment, timeout=30)

    def test_r03_published_write_after_directory_growth(self):
        result = self.build_and_run(r'''
#include "strata/core/conversation_buffer.hpp"
#include <cstdint>
int main() {
    strata::core::ConversationBuffer root;
    root.resize(4096, 7); root.freeze();
    auto child = root;
    if (!child.visit(1, 17, [](uint8_t* p, size_t n, size_t) {
            for (size_t i=0; i<n; ++i) p[i]=9;
            return true;
        })) return 1;
    uint8_t original=0, changed=0;
    if (!root.read(&original, 1, 1) || !child.read(&changed, 1, 1) || original!=7 || changed!=9) return 2;
    // Repeat split/write and shrink/grow; the ancestor retains its bytes.
    if (!child.visit(34, 23, [](uint8_t* p, size_t n, size_t) {
            for (size_t i=0; i<n; ++i) p[i]=11;
            return true;
        })) return 3;
    child.resize(2048); child.resize(4096, 13);
    if (!root.read(&original, 34, 1) || original!=7) return 4;
    return 0;
}
''', sanitizer=True)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertNotIn("runtime error:", result.stderr)
        self.assertNotIn("ERROR: AddressSanitizer", result.stderr)

    def test_r03_failed_cow_keeps_payload_and_copy_counter(self):
        result = self.build_and_run(r'''
#include "strata/core/conversation_buffer.hpp"
#include <cstdlib>
#include <new>
#include <cstdio>
static int fail_after=-1;
void* operator new(std::size_t n) {
    if (fail_after==0) throw std::bad_alloc();
    if (fail_after>0) --fail_after;
    if (void* p=std::malloc(n)) return p;
    throw std::bad_alloc();
}
void operator delete(void* p) noexcept { std::free(p); }
void operator delete(void* p,std::size_t) noexcept { std::free(p); }
int main() {
    using strata::core::ConversationBuffer;
    for (int failure=0; failure<4; ++failure) {
        ConversationBuffer root;root.resize(4096,7);root.freeze();auto child=root;
        const auto before=ConversationBuffer::copied_bytes();
        bool rejected=false; fail_after=failure;
        try {child.visit(1,17,[](uint8_t* p,size_t n,size_t){for(size_t i=0;i<n;++i)p[i]=9;return true;});}
        catch (const std::bad_alloc&) {rejected=true;}
        fail_after=-1;
        uint8_t ancestor=0, sibling=0;
        if (!rejected || ConversationBuffer::copied_bytes()!=before || child.size()!=4096 ||
            !root.read(&ancestor,1,1) || !child.read(&sibling,1,1) || ancestor!=7 || sibling!=7) return failure+1;
    }
    std::puts("R03_FAILURES allocation_points=4 completed_copy_bytes=0 retained_payload_unchanged=true");
    return 0;
}
''')
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        print(result.stdout.strip())


if __name__ == "__main__":
    unittest.main(verbosity=2)
