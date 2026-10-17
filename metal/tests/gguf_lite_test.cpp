// metal/tests/gguf_lite_test.cpp - the GGUF reader (src/gguf_lite.cpp) on crafted files: every malformed header is
// refused with a message (no crash, no exit), valid tiny files load.  No GPU.  Exit 0 = every check passed.
#include "gguf_lite.h"
#include "ref_quant.h"

#include <unistd.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

namespace {

int failures = 0;
std::string dir;

// a GGUF v3 file built field by field
struct G {
    std::vector<uint8_t> b;
    template <class T> G& v(T x) {
        const uint8_t* p = (const uint8_t*)&x;
        b.insert(b.end(), p, p + sizeof(T));
        return *this;
    }
    G& s(const std::string& x) { v<uint64_t>(x.size()); b.insert(b.end(), x.begin(), x.end()); return *this; }
    G& head(uint64_t n_t, uint64_t n_kv) { return v<uint32_t>(0x46554747u).v<uint32_t>(3).v(n_t).v(n_kv); }
    template <class T> G& kv(const std::string& k, uint32_t type, T x) { return s(k).v(type).v(x); }
    G& kv_u32(const std::string& k, uint32_t x) { return kv(k, 4, x); }
    G& tensor(const std::string& name, std::vector<uint64_t> shape, uint32_t type, uint64_t off) {
        s(name).v<uint32_t>((uint32_t)shape.size());
        for (uint64_t d : shape) v(d);
        return v(type).v(off);
    }
    G& pad(size_t align) { b.resize((b.size() + align - 1) / align * align); return *this; }
    G& data(size_t n) { for (size_t i = 0; i < n; ++i) b.push_back((uint8_t)i); return *this; }
};

std::string write(const std::vector<uint8_t>& b, const std::string& name) {
    const std::string p = dir + "/" + name + ".gguf";
    FILE* f = std::fopen(p.c_str(), "wb");
    if (!f) { std::perror(p.c_str()); std::exit(1); }
    if (!b.empty()) std::fwrite(b.data(), 1, b.size(), f);
    std::fclose(f);
    return p;
}

void refused(const std::vector<uint8_t>& b, const std::string& what, const char* want) {
    gguf::File f;
    std::string err;
    const bool ok = !f.open(write(b, "case"), err) && err.find(want) != std::string::npos;
    std::printf("%s refused: %-44s %s\n", ok ? "ok  " : "FAIL", what.c_str(), err.empty() ? "(no error)" : err.c_str());
    failures += !ok;
}

void loads(bool ok, const std::string& what, const std::string& err) {
    std::printf("%s loads:   %-44s %s\n", ok ? "ok  " : "FAIL", what.c_str(), err.c_str());
    failures += !ok;
}

// a valid file: an F32 [8 x 2] at 0 and a Q8_0 [32] at `align`
G valid(uint32_t align) {
    G g;
    g.head(2, 1).kv_u32("general.alignment", align);
    g.tensor("a", {8, 2}, ref::F32, 0).tensor("b", {32}, ref::Q8_0, align);
    return g.pad(align).data(align + 34);
}

}  // namespace

int main() {
    const char* t = std::getenv("TMPDIR");
    std::string base = std::string(t && *t ? t : "/tmp") + "/gguf_lite_test.XXXXXX";
    if (!mkdtemp(base.data())) { std::perror("mkdtemp"); return 1; }
    dir = base;

    for (uint32_t al : {64u, 16384u, 131072u}) {   // the right offsets and sizes; no upper bound on the alignment
        const G g = valid(al);
        gguf::File f;
        std::string err;
        const bool open = f.open(write(g.b, "valid"), err);
        const gguf::Tensor* a = f.find("a");
        const gguf::Tensor* b = f.find("b");
        const size_t data = g.b.size() - al - 34;
        const bool ok = open && data % al == 0 && a && b && a->type == ref::F32 && a->offset == data &&
                        a->bytes == 64 && b->type == ref::Q8_0 && b->offset == data + al && b->bytes == 34 &&
                        f.meta_num["general.alignment"] == al;
        loads(ok, "a valid file, alignment " + std::to_string(al), err);
    }
    {   // a shard with no tensors may end before its aligned data start
        gguf::File f;
        std::string err;
        const bool open = f.open(write(G().head(0, 1).kv_u32("general.alignment", 4096).b, "valid"), err);
        loads(open && f.tensors.empty(), "no tensors, ends before the data start", err);
    }
    {   // every truncation of the valid file (0 bytes, inside the header, inside the data)
        const G g = valid(64);
        int bad = 0;
        for (size_t n = 0; n < g.b.size(); ++n) {
            gguf::File f;
            std::string err;
            if (f.open(write(std::vector<uint8_t>(g.b.begin(), g.b.begin() + n), "cut"), err) || err.empty()) ++bad;
        }
        std::printf("%s refused: %-44s %zu prefixes\n", bad ? "FAIL" : "ok  ", "every truncation of the valid file",
                    g.b.size());
        failures += bad != 0;
    }
    refused(G().v<uint32_t>(0x12345678u).v<uint32_t>(3).v<uint64_t>(0).v<uint64_t>(0).b, "not GGUF magic",
            "not a GGUF");
    refused(G().v<uint32_t>(0x46554747u).v<uint32_t>(2).v<uint64_t>(0).v<uint64_t>(0).b, "version 2", "version");
    refused(G().head(0, 1).v<uint64_t>(1ull << 62).data(64).b, "key length past the end", "truncated");
    refused(G().head(0, 1).s("k").v<uint32_t>(8).v<uint64_t>(~0ull).data(64).b, "string length past the end",
            "truncated");
    refused(G().head(0, 1).s("k").v<uint32_t>(9).v<uint32_t>(4).v<uint64_t>(1ull << 61).data(64).b,
            "u32 array past the end", "truncated");
    refused(G().head(0, 1).s("k").v<uint32_t>(9).v<uint32_t>(8).v<uint64_t>(~0ull).data(64).b,
            "string array past the end", "truncated");
    refused(G().head(1, 1).s("k").v<uint32_t>(9).v<uint32_t>(0).v<uint64_t>(0ull - (1ull << 32)).data(64).b,
            "u8 array count that wraps the read pointer", "truncated");
    refused(G().head(0, 1).s("k").v<uint32_t>(9).v<uint32_t>(9).v<uint64_t>(1).data(64).b, "nested array",
            "GGUF array of type 9: k");
    refused(G().head(0, 1).s("k").v<uint32_t>(9).v<uint32_t>(77).v<uint64_t>(1).data(64).b, "array of unknown type 77",
            "GGUF array of type 77: k");
    refused(G().head(0, 1).s("k").v<uint32_t>(77).data(64).b, "unknown value type 77", "unknown GGUF value type 77: k");
    refused(G().head(1, 0).tensor("a", {1, 1, 1, 1, 1}, ref::F32, 0).pad(32).data(32).b, "5 dimensions",
            "tensor has 5 dimensions: a");
    refused(G().head(~0ull >> 1, 0).data(64).b, "huge tensor count", "truncated");
    for (uint32_t al : {0u, 48u, 0x80000001u})
        refused(G().head(1, 1).kv_u32("general.alignment", al).tensor("a", {8}, ref::F32, 0).pad(256).data(32).b,
                "general.alignment " + std::to_string(al), "power of two");
    refused(G().head(1, 1).kv<uint64_t>("general.alignment", 10, 1ull << 32).tensor("a", {8}, ref::F32, 0).pad(256)
                .data(32).b, "general.alignment 2^32 (u64)", "power of two");
    refused(G().head(1, 1).kv<int32_t>("general.alignment", 5, -64).tensor("a", {8}, ref::F32, 0).pad(256).data(32).b,
            "general.alignment -64 (i32)", "power of two");
    refused(G().head(1, 0).tensor("a", {8}, ref::F32, 1ull << 40).pad(32).data(32).b, "offset past the file end",
            "outside");
    refused(G().head(1, 0).tensor("a", {16}, ref::F32, 0).pad(32).data(32).b, "offset + size past the file end",
            "outside");
    refused(G().head(1, 0).tensor("a", {8}, ref::F32, ~0ull - 7).pad(32).data(32).b, "offset wraps around", "outside");
    refused(G().head(1, 0).tensor("a", {1ull << 32, 1ull << 32, 2}, ref::F32, 0).pad(32).data(32).b,
            "shape product overflows", "overflows");
    refused(G().head(1, 0).tensor("a", {1ull << 62}, ref::F32, 0).pad(32).data(32).b, "byte size overflows",
            "overflows");
    refused(G().head(1, 0).tensor("a", {256}, 99, 0).pad(32).data(256).b, "unknown tensor type 99",
            "unknown tensor type 99: a");
    refused(G().head(1, 0).tensor("a", {32}, 3, 0).pad(32).data(64).b, "unknown tensor type 3 (Q4_1)",
            "unknown tensor type 3: a");
    refused(G().head(1, 0).tensor("a", {32}, 0x80000000u, 0).pad(32).data(64).b, "negative tensor type",
            "unknown tensor type");

    std::remove((dir + "/valid.gguf").c_str());
    std::remove((dir + "/cut.gguf").c_str());
    std::remove((dir + "/case.gguf").c_str());
    rmdir(dir.c_str());
    std::printf(failures ? "%d FAILED\n" : "ALL PASS\n", failures);
    return failures ? 1 : 0;
}
