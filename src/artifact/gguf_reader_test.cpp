// src/artifact/gguf_reader_test.cpp - the reader refuses what it cannot arbitrate (synthetic GGUF, no model,
// no GPU).
//
//   1. two tensors with distinct names: the file opens and find() sees both;
//   2. the same name twice: refused at open, and the error names the tensor and the file - GGUF has no
//      index to say which of the two a name means, and find() is first-match, so the alternative is a
//      lookup that silently picks by position.
//   3. no tensors, and the file ends at its header, before the 32-byte-aligned data start: the file opens,
//      with data_start() clamped to the file size.  A split model's first shard can hold metadata only and
//      end unpadded (unsloth's UD-Q5_K_XL shard 1 ends 6 bytes before its aligned data start; llama.cpp's
//      gguf-py reads it); nothing is ever read from its data section.
//   4. one tensor, and the file ends at its header: still refused at open ("data section starts past EOF").
//
// The fixture is a minimal GGUF v3 written here (header, no metadata, two F32[8] tensors, 32-byte
// alignment), so the test needs neither gguf-py nor a shard.  The file is closed before it is removed:
// on Windows an open mapping keeps it.
#include "strata/artifact/gguf_reader.hpp"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

namespace {
int g_fail = 0;
void check(bool ok, const char* what) {
    std::printf("  %-66s %s\n", what, ok ? "ok" : "FAIL");
    if (!ok) ++g_fail;
}

template <class T> void put(std::vector<uint8_t>& b, T v) {
    const auto n = b.size();
    b.resize(n + sizeof(T));
    std::memcpy(b.data() + n, &v, sizeof(T));
}
void put_str(std::vector<uint8_t>& b, const std::string& s) {
    put<uint64_t>(b, s.size());
    b.insert(b.end(), s.begin(), s.end());
}

// A GGUF v3 file of F32[8] tensors named `names`, laid out 32 bytes apart from a 32-byte-aligned data start.
// `header_only`: the file ends right after the tensor directory - no alignment padding, no data.
std::filesystem::path write_gguf(const std::vector<std::string>& names, bool header_only = false) {
    std::vector<uint8_t> b;
    put<uint32_t>(b, 0x46554747u);   // "GGUF"
    put<uint32_t>(b, 3);
    put<uint64_t>(b, names.size());  // n_tensors
    put<uint64_t>(b, 0);             // n_kv
    for (size_t i = 0; i < names.size(); ++i) {
        put_str(b, names[i]);
        put<uint32_t>(b, 1);         // n_dims
        put<uint64_t>(b, 8);
        put<uint32_t>(b, 0);         // F32
        put<uint64_t>(b, 32 * i);    // offset from data_start
    }
    if (!header_only) b.resize((b.size() + 31) / 32 * 32 + 32 * names.size(), 0);
    const auto path = std::filesystem::temp_directory_path() / "strata_gguf_reader_test.gguf";
    std::ofstream(path, std::ios::binary).write(reinterpret_cast<const char*>(b.data()), (std::streamsize)b.size());
    return path;
}

// One F32[4] tensor whose four floats are 1..4, at the 32-byte-aligned data start - known bytes, so the test can
// tell a read that came back from the file after release_pages() from a stale pointer.
std::filesystem::path write_gguf_with_data() {
    std::vector<uint8_t> b;
    put<uint32_t>(b, 0x46554747u);   // "GGUF"
    put<uint32_t>(b, 3);
    put<uint64_t>(b, 1);             // one tensor
    put<uint64_t>(b, 0);             // no metadata
    put_str(b, "blk.0.attn_q.weight");
    put<uint32_t>(b, 1);             // n_dims
    put<uint64_t>(b, 4);             // shape[0]
    put<uint32_t>(b, 0);             // F32
    put<uint64_t>(b, 0);             // offset from data_start
    b.resize((b.size() + 31) / 32 * 32, 0);   // pad to the aligned data start
    const float vals[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    for (float v : vals) put<float>(b, v);
    const auto path = std::filesystem::temp_directory_path() / "strata_gguf_reader_data.gguf";
    std::ofstream(path, std::ios::binary).write(reinterpret_cast<const char*>(b.data()), (std::streamsize)b.size());
    return path;
}
}  // namespace

int main() {
    std::printf("gguf_reader_test\n");
    {
        const auto path = write_gguf({"blk.0.attn_q.weight", "blk.0.attn_k.weight"});
        std::string err;
        size_t n = 0;
        bool both = false;
        try {
            strata::GgufFile g(path.string());
            n = g.tensors().size();
            both = g.find("blk.0.attn_q.weight") && g.find("blk.0.attn_k.weight");
        } catch (const std::exception& e) {
            err = e.what();
        }
        std::filesystem::remove(path);
        check(err.empty() && n == 2 && both, "distinct names: the file opens and find() sees both tensors");
    }
    {
        const auto path = write_gguf({"blk.0.attn_q.weight", "blk.0.attn_q.weight"});
        std::string err;
        try {
            strata::GgufFile g(path.string());
        } catch (const std::exception& e) {
            err = e.what();
        }
        std::filesystem::remove(path);
        check(!err.empty(), "the same name twice: refused at open");
        check(err.find("blk.0.attn_q.weight") != std::string::npos, "  the error names the tensor");
        check(err.find(path.filename().string()) != std::string::npos, "  the error names the file");
        if (!err.empty()) std::printf("  (%s)\n", err.c_str());
    }
    {
        const auto path = write_gguf({}, true);   // 24 bytes: the aligned data start (32) is past EOF
        std::string err;
        uint64_t size = 0, start = 1;
        size_t n = 1;
        try {
            strata::GgufFile g(path.string());
            size = g.file_size();
            start = g.data_start();
            n = g.tensors().size();
        } catch (const std::exception& e) {
            err = e.what();
        }
        std::filesystem::remove(path);
        check(err.empty() && n == 0, "no tensors, ends before the aligned data start: the file opens");
        check(err.empty() && start == size, "  data_start() is the file size (nothing to read past it)");
        if (!err.empty()) std::printf("  (%s)\n", err.c_str());
    }
    {
        const auto path = write_gguf({"blk.0.attn_q.weight"}, true);
        std::string err;
        try {
            strata::GgufFile g(path.string());
        } catch (const std::exception& e) {
            err = e.what();
        }
        std::filesystem::remove(path);
        check(err.find("data section starts past EOF") != std::string::npos,
              "one tensor, ends before the aligned data start: refused at open");
    }
    {
        // Q36 plan Phase 1: release_pages() drops the mapping's pages; the bytes still read back through the
        // mapping (they fault in from the file), released() says so, and the object still destructs cleanly.
        const auto path = write_gguf_with_data();
        std::string err;
        bool before_ok = false, after_ok = false, released = false;
        try {
            strata::GgufFile g(path.string());
            const strata::TensorInfo* t = g.find("blk.0.attn_q.weight");
            const float* p = (const float*) g.tensor_data(*t);
            before_ok = t && p[0] == 1.0f && p[1] == 2.0f && p[2] == 3.0f && p[3] == 4.0f;
            g.release_pages();
            released = g.released();
            g.release_pages();   // a second call is a no-op
            const float* q = (const float*) g.tensor_data(*t);
            after_ok = q[0] == 1.0f && q[1] == 2.0f && q[2] == 3.0f && q[3] == 4.0f;
        } catch (const std::exception& e) {
            err = e.what();
        }
        std::filesystem::remove(path);
        check(err.empty() && before_ok, "release_pages: the tensor reads its bytes before the release");
        check(err.empty() && released, "  released() is true after release_pages()");
        check(err.empty() && after_ok, "  the same bytes read back after release (faulted in from the file)");
        if (!err.empty()) std::printf("  (%s)\n", err.c_str());
    }
    {
        const auto path = write_gguf_with_data();
        std::string err;
        bool mok = false;
        try {
            strata::GgufModel m = strata::GgufModel::open(path.string());
            m.release_pages();
            mok = m.released();
        } catch (const std::exception& e) {
            err = e.what();
        }
        std::filesystem::remove(path);
        check(err.empty() && mok, "GgufModel: release_pages() releases every shard and released() agrees");
        if (!err.empty()) std::printf("  (%s)\n", err.c_str());
    }
    std::printf(g_fail ? "gguf_reader_test: %d FAILED\n" : "gguf_reader_test: all passed\n", g_fail);
    return g_fail ? 1 : 0;
}
