// engine/src/qwen35/memt_snapshot_test.cpp - the C++ `memt-memory-v1` codec.
//
// tools/test_memt_snapshot.py pins the same bytes from the Python side.  The
// FIXTURE below is `memt_snapshot.encode()`'s output for two known 2x2x2 states,
// so a byte-equal encode here proves the two implementations interoperate.
#include "strata/qwen35/memt_snapshot.hpp"

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <exception>
#include <functional>
#include <limits>
#include <string>
#include <vector>

namespace fs = std::filesystem;
namespace M = strata::qwen35::memt;

namespace {

int g_checks = 0;
int g_fail = 0;

void check(bool ok, const std::string& what) {
    ++g_checks;
    if (!ok) {
        std::fprintf(stderr, "FAIL: %s\n", what.c_str());
        ++g_fail;
    }
}

const char* kIdentity = "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef";

const char* kFixtureHex =
    "20010000000000007b225f5f6d657461646174615f5f223a207b22666f726d6174223a20226d656d742d6d656d6f7279"
    "2d7631222c20226964656e74697479223a20223031323334353637383961626364656630313233343536373839616263"
    "6465663031323334353637383961626364656630313233343536373839616263646566227d2c20226d656d6f72792e33"
    "223a207b22646174615f6f666673657473223a205b302c2033325d2c20226474797065223a2022463332222c20227368"
    "617065223a205b312c20322c20322c20325d7d2c20226d656d6f72792e37223a207b22646174615f6f66667365747322"
    "3a205b33322c2036345d2c20226474797065223a2022463332222c20227368617065223a205b312c20322c20322c2032"
    "5d7d7d2020202020000000000000003e0000803e0000c03e0000003f0000203f0000403f0000603f0000903f0000a03f"
    "0000b03f0000c03f0000d03f0000e03f0000f03f00000040";

std::string unhex(const std::string& hex) {
    std::string out;
    out.reserve(hex.size() / 2);
    auto nibble = [](char c) -> int {
        if (c >= '0' && c <= '9') return c - '0';
        if (c >= 'a' && c <= 'f') return c - 'a' + 10;
        return -1;
    };
    for (std::size_t i = 0; i + 1 < hex.size(); i += 2) {
        const int hi = nibble(hex[i]), lo = nibble(hex[i + 1]);
        if (hi < 0 || lo < 0) return {};
        out.push_back((char) (hi * 16 + lo));
    }
    return out;
}

std::string fixture() { return unhex(kFixtureHex); }

std::vector<float> state3() {
    std::vector<float> v(8);
    for (int i = 0; i < 8; ++i) v[(std::size_t) i] = (float) i / 8.0f;
    return v;
}

std::vector<float> state7() {
    std::vector<float> v(8);
    for (int i = 0; i < 8; ++i) v[(std::size_t) i] = (float) (i + 9) / 8.0f;
    return v;
}

bool rejects(const std::function<void()>& fn) {
    try {
        fn();
    } catch (const M::SnapshotError&) {
        return true;
    }
    return false;
}

struct TempDir {
    fs::path path;
    TempDir() {
        path = fs::temp_directory_path() /
               ("strata-memt-snapshot-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
        fs::create_directories(path);
    }
    ~TempDir() {
        std::error_code ignored;
        fs::remove_all(path, ignored);
    }
};

}  // namespace

int main() {
    std::set_terminate([] {
        if (auto e = std::current_exception()) {
            try { std::rethrow_exception(e); }
            catch (const std::exception& x) { std::fprintf(stderr, "terminate: %s\n", x.what()); }
        }
        std::abort();
    });
    const std::vector<float> a = state3(), b = state7();

    // 1. The C++ encoder writes exactly memt_snapshot.py's bytes.
    const M::Snapshot blob = M::encode_snapshot(kIdentity, {3, 7}, {a.data(), b.data()}, 2, 2);
    check(blob == fixture(), "encode_snapshot is byte-identical to the Python codec");
    // The order the caller passes layers in does not change the file.
    check(M::encode_snapshot(kIdentity, {7, 3}, {b.data(), a.data()}, 2, 2) == blob,
          "layer order does not change the snapshot");

    // 2. Decode round-trips the values and sorts the layers.
    {
        const auto layers = M::decode_snapshot(fixture(), kIdentity, 2, 2);
        check(layers.size() == 2 && layers[0].layer == 3 && layers[1].layer == 7, "decode keeps both layers");
        check(layers.size() == 2 && layers[0].values == a && layers[1].values == b, "decode round-trips the values");
    }

    // 3. Identity, format, shape and dtype are enforced.
    check(rejects([] { M::decode_snapshot(fixture(), "other", 2, 2); }), "a wrong identity is refused");
    check(rejects([] { M::decode_snapshot(fixture(), kIdentity, 2, 4); }), "a wrong geometry is refused");
    {
        std::string wrong_format = fixture();
        const std::size_t at = wrong_format.find("memt-memory-v1");
        wrong_format.replace(at, 14, "memt-memory-v2");
        check(rejects([&] { M::decode_snapshot(wrong_format, kIdentity, 2, 2); }), "a wrong format is refused");
    }
    {
        std::string wrong_dtype = fixture();
        const std::size_t at = wrong_dtype.find("F32");
        wrong_dtype.replace(at, 3, "BF6");
        check(rejects([&] { M::decode_snapshot(wrong_dtype, kIdentity, 2, 2); }), "another dtype is refused");
    }
    {
        std::string wrong_shape = fixture();
        const std::size_t at = wrong_shape.find("[1, 2, 2, 2]");
        wrong_shape.replace(at, 12, "[1, 2, 2, 3]");
        check(rejects([&] { M::decode_snapshot(wrong_shape, kIdentity, 2, 2); }), "another shape is refused");
    }
    {
        std::string huge = fixture();
        const std::size_t at = huge.find("[0, 32]");
        huge.replace(at, 7, "[0, 33]");
        check(rejects([&] { M::decode_snapshot(huge, kIdentity, 2, 2); }), "a byte length that disagrees is refused");
    }

    // 4. Corruption is refused, not silently read.
    check(rejects([&] { M::decode_snapshot(fixture().substr(0, 100), kIdentity, 2, 2); }),
          "a truncated file is refused");
    {
        std::string overlap = fixture();
        overlap.replace(overlap.find("[32, 64]"), 8, "[16, 48]");
        check(rejects([&] { M::decode_snapshot(overlap, kIdentity, 2, 2); }), "overlapping offsets are refused");
    }
    {
        std::string duplicate = fixture();
        duplicate.replace(duplicate.find("\"shape\""), 7, "\"dtype\"");
        check(rejects([&] { M::decode_snapshot(duplicate, kIdentity, 2, 2); }), "duplicate header keys are refused");
    }
    check(rejects([&] { M::decode_snapshot(fixture() + '\0', kIdentity, 2, 2); }), "trailing bytes are refused");
    {
        std::string nan = fixture();
        const uint32_t quiet_nan = 0x7fc00000u;
        std::memcpy(&nan[8 + 288], &quiet_nan, 4);
        check(rejects([&] { M::decode_snapshot(nan, kIdentity, 2, 2); }), "a non-finite state is refused");
    }
    {
        std::vector<float> bad = a;
        bad[3] = std::numeric_limits<float>::infinity();
        check(rejects([&] { M::encode_snapshot(kIdentity, {3}, {bad.data()}, 2, 2); }),
              "encoding a non-finite state is refused");
    }
    check(rejects([&] { M::encode_snapshot("", {3}, {a.data()}, 2, 2); }), "an empty identity is refused");
    check(rejects([&] { M::encode_snapshot(kIdentity, {3, 3}, {a.data(), a.data()}, 2, 2); }),
          "a duplicate layer is refused");
    check(rejects([&] { M::encode_snapshot(kIdentity, {3}, {}, 2, 2); }), "a missing state is refused");

    // 5. Files: load what we saved, refuse an existing name, and allow an explicit overwrite.
    {
        TempDir d;
        const fs::path path = d.path / "state.safetensors";
#if !defined(_WIN32)
        const mode_t old_umask = ::umask(0022);
#endif
        M::save_snapshot(path.string(), blob, false);
#if !defined(_WIN32)
        ::umask(old_umask);
        check(fs::status(path).permissions() == (fs::perms::owner_read | fs::perms::owner_write),
              "snapshot is readable and writable by its owner only");
#endif
        check(M::load_snapshot(path.string()) == blob, "save_snapshot / load_snapshot round-trip");
        check(rejects([&] { M::save_snapshot(path.string(), blob, false); }), "an existing snapshot is not replaced");
        M::save_snapshot(path.string(), blob, true);
        check(M::load_snapshot(path.string()) == blob, "an explicit overwrite replaces it");
        check(M::decode_snapshot(M::load_snapshot(path.string()), kIdentity, 2, 2).size() == 2,
              "the saved file decodes");
        bool stray = false;
        for (const auto& entry : fs::directory_iterator(d.path))
            stray |= entry.path().filename().string().rfind("state.safetensors.tmp", 0) == 0;
        check(!stray, "no temporary file is left behind");
        check(rejects([&] { M::save_snapshot((d.path / "state.bin").string(), blob, false); }),
              "snapshot names must end in .safetensors");
#if !defined(_WIN32)
        {
            fs::create_symlink(path, d.path / "link.safetensors");
            check(rejects([&] { M::save_snapshot((d.path / "link.safetensors").string(), blob, true); }),
                  "a symlink target is refused");
        }
#endif
    }

    // 6. An empty set is a valid snapshot (the Python codec writes one for a model
    //    whose memory layers are disabled).
    {
        const M::Snapshot empty = M::encode_snapshot(kIdentity, {}, {}, 16, 256);
        check(M::decode_snapshot(empty, kIdentity, 16, 256).empty(), "an empty snapshot round-trips");
    }

    if (g_fail == 0) std::printf("memt_snapshot_test: %d checks passed\n", g_checks);
    return g_fail == 0 ? 0 : 1;
}
