"""Exact Qwen-Image 2.1 RoPE comparison with independently pinned sd.cpp source.

Both translation units use the same compiler, flags and libm. The seven historical
glibc hashes remain an additional check on glibc, not a cross-libm expectation.
"""
import hashlib
import json
import math
import os
import shutil
import struct
import subprocess
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "tools/image/gate/data"
ORACLE = DATA / "sdcpp_rope_2988060"
CXX = shutil.which("g++") or shutil.which("c++") or shutil.which("clang++")


def upstream_source():
    provenance = json.loads((ORACLE / "PROVENANCE.json").read_text(encoding="utf-8"))
    originals = {}
    for item in provenance["files"]:
        raw = (ORACLE / item["file"]).read_bytes()
        assert hashlib.sha256(raw).hexdigest() == item["sha256"], item["file"]
        blob = b"blob " + str(len(raw)).encode() + b"\0" + raw
        assert hashlib.sha1(blob).hexdigest() == item["git_blob"], item["file"]
        originals[item["file"]] = raw.decode("utf-8")
    chunks = []
    for item in provenance["extracts"]:
        source = originals[item["file"]]
        first = source.index(item["start"])
        last = source.index(item["end"], first)
        chunk = source[first:last]
        assert hashlib.sha256(chunk.encode()).hexdigest() == item["sha256"]
        assert source[:first].count("\n") + 1 == item["first_line"]
        assert source[:last].count("\n") == item["last_line"]
        chunks.append(chunk)
    source = originals["qwen_image_2_1.hpp"]
    assert "std::vector<int> axes_dim = {16, 56, 56};" in source
    assert "Rope::embed_nd(embedding.ids, 1, 10000.f, config.axes_dim" in source
    return r'''
#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <utility>
#include <vector>
#define __STATIC_INLINE__ static inline
namespace sd { template<class T> struct Tensor {
    bool empty() const { return true; }
    int64_t numel() const { throw std::logic_error("oracle: unexpected tensor size access"); }
    T operator[](int64_t) const { throw std::logic_error("oracle: unexpected image slot access"); }
}; }
namespace Rope {
''' + chunks[0] + chunks[1] + "\n}\nnamespace Qwen {\n" + chunks[2] + r'''
}
std::vector<float> oracle_positions(int64_t L, int64_t H, int64_t W) {
    return Rope::flatten(Qwen::QwenImage21Layout::build(L, sd::Tensor<int32_t>{}, {{H,W}}).positions);
}
std::vector<float> oracle_table(int64_t L, int64_t H, int64_t W) {
    const auto layout = Qwen::QwenImage21Layout::build(L, sd::Tensor<int32_t>{}, {{H,W}});
    return Rope::embed_nd(layout.positions, 1, 10000.f, {16,56,56});
}
'''


HARNESS = r'''
#include "strata/image/qi21_rope.hpp"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#ifdef __GLIBC__
#include <gnu/libc-version.h>
#endif
std::vector<float> oracle_positions(int64_t, int64_t, int64_t);
std::vector<float> oracle_table(int64_t, int64_t, int64_t);
int main(int argc, char** argv) {
    if (argc == 2 && std::strcmp(argv[1], "--provenance") == 0) {
#ifdef __GLIBC__
        std::printf("libc=glibc %s\n", gnu_get_libc_version());
#else
        std::puts("libc=non-glibc");
#endif
        std::printf("compiler=%s; flags=-std=c++17 -O2\n", __VERSION__);
        return 0;
    }
    if (argc != 6) return 2;
    const bool reference = std::strcmp(argv[1], "ref") == 0;
    const int64_t L = std::atoll(argv[2]), H = std::atoll(argv[3]), W = std::atoll(argv[4]);
    const auto p = reference ? oracle_positions(L,H,W) : strata::image::qi21_positions(L,H,W);
    const auto t = reference ? oracle_table(L,H,W) : strata::image::qi21_rope_table(p);
    FILE* f = std::fopen(argv[5], "wb");
    if (!f) return 3;
    const bool ok = std::fwrite(p.data(), sizeof(float), p.size(), f) == p.size()
        && std::fwrite(t.data(), sizeof(float), t.size(), f) == t.size();
    return std::fclose(f) == 0 && ok ? 0 : 4;
}
'''


def bit_difference(actual, expected):
    """Exact comparison; diagnostics never supply a numerical acceptance tolerance."""
    if actual == expected:
        return None
    if len(actual) != len(expected):
        return f"byte lengths actual={len(actual)} reference={len(expected)}"
    count = len(actual) // 4
    a, b = struct.unpack(f"<{count}I", actual), struct.unpack(f"<{count}I", expected)
    changed = [i for i in range(count) if a[i] != b[i]]
    max_abs, max_ulp, nonfinite = 0.0, 0, 0
    for i in changed:
        av, bv = struct.unpack("<ff", struct.pack("<II", a[i], b[i]))
        if not (math.isfinite(av) and math.isfinite(bv)):
            nonfinite += 1
            continue
        ordered = lambda u: (~u & 0xffffffff) if u & 0x80000000 else u | 0x80000000
        max_abs = max(max_abs, abs(av - bv))
        max_ulp = max(max_ulp, abs(ordered(a[i]) - ordered(b[i])))
    i = changed[0]
    return (f"different={len(changed)}/{count}; first={i}; actual=0x{a[i]:08x} "
            f"reference=0x{b[i]:08x}; max_abs={max_abs:.17g}; max_ulp={max_ulp}; nonfinite={nonfinite}")


class OracleIntegrity(unittest.TestCase):
    def test_pinned_originals_and_extraction(self):
        self.assertIn("QwenImage21Layout::build", upstream_source())

    def test_exact_comparison_rejects_bit_mutation_and_permutation(self):
        original = struct.pack("<3f", 1.0, 2.0, 3.0)
        self.assertIsNone(bit_difference(original, original))
        mutated = struct.pack("<I", 0x3f800001) + original[4:]
        self.assertIn("max_ulp=1;", bit_difference(mutated, original))
        self.assertIn("different=2/3;", bit_difference(struct.pack("<3f", 2.0, 1.0, 3.0), original))


@unittest.skipUnless(CXX, "no C++ compiler")
class RopeTableMatchesSdcpp(unittest.TestCase):
    def test_tables(self):
        cases = json.loads((DATA / "qi21_rope_sha.json").read_text())
        self.assertEqual(len(cases), 7)
        with tempfile.TemporaryDirectory() as td:
            work = Path(td)
            (work / "oracle.cpp").write_text(upstream_source(), encoding="utf-8")
            (work / "main.cpp").write_text(HARNESS, encoding="utf-8")
            exe = work / ("rope.exe" if os.name == "nt" else "rope")
            cmd = [CXX, "-std=c++17", "-O2", "-I", str(ROOT / "include"),
                   str(ROOT / "src/image/dit/qi21_rope.cpp"), str(work / "oracle.cpp"),
                   str(work / "main.cpp"), "-o", str(exe)]
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            self.assertEqual(r.returncode, 0, r.stderr[-6000:])
            provenance = subprocess.check_output([str(exe), "--provenance"], text=True)
            print(provenance.strip())
            for n, case in enumerate(cases):
                with self.subTest(case=case["from"]):
                    outputs = []
                    for mode in ("own", "ref"):
                        dest = work / (mode + ".bin")
                        subprocess.run([str(exe), mode, str(case["text_tokens"]), str(case["H"]),
                                        str(case["W"]), str(dest)], check=True, timeout=60)
                        outputs.append(dest.read_bytes())
                    count = case["text_tokens"] + case["H"] * case["W"]
                    self.assertEqual(len(outputs[0]), count * (3 + 256) * 4)
                    split = count * 3 * 4
                    differences = [bit_difference(outputs[0][:split], outputs[1][:split]),
                                   bit_difference(outputs[0][split:], outputs[1][split:])]
                    if any(differences):
                        saved = Path(tempfile.mkdtemp(prefix=f"strata-rope-delta-{n}-"))
                        for mode, raw in zip(("own", "ref"), outputs):
                            (saved / (mode + ".bin")).write_bytes(raw)
                        (saved / "delta.json").write_text(json.dumps(dict(case=case, compiler=provenance,
                            positions=differences[0], table=differences[1]), indent=2), encoding="utf-8")
                        print(f"RoPE exact mismatch retained at {saved}")
                    self.assertEqual(differences, [None, None], f"positions/table: {differences}")
                    if provenance.startswith("libc=glibc "):
                        self.assertEqual(hashlib.sha256(outputs[0][split:]).hexdigest(), case["sha256"])


if __name__ == "__main__":
    unittest.main()
