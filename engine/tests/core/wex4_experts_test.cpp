// --experts-wex4: the reader's rules (wex4_experts.cpp) on small containers written by tests/core/wex4_writer.hpp.
// CPU only, no GPU, no model.  Every refusal is checked by a fragment of its message, so a user sees what failed.
#include "strata/core/wex4_experts.hpp"
#include "strata/kernels/cpu/expert_layout.hpp"
#include "wex4_writer.hpp"

#include <cctype>
#include <chrono>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace fs = std::filesystem;
using strata::kernels::cpu::ExpertLayout;

namespace {

void require(bool ok, const std::string& what) {
    if (!ok) throw std::runtime_error(what);
}

std::string sha_hex(const std::string& s) {
    uint8_t d[32];
    strata::core::detail::wex4_sha256((const uint8_t*) s.data(), s.size(), d);
    char buf[65];
    for (int i = 0; i < 32; ++i) std::snprintf(buf + 2 * i, 3, "%02x", d[i]);
    return buf;
}

// Two layers of 4 experts: layer 0's blob (5000 B) is not a multiple of 4096 (records 8192 B apart), layer 1's is.
ExpertLayout layout() {
    ExpertLayout lay;
    lay.native = true;
    lay.n_layers = 2;
    lay.n_expert = 4;
    lay.fmt.resize(2);
    lay.fmt[0].gu_type = 19; lay.fmt[0].d_type = 20; lay.fmt[0].up_off = 2000; lay.fmt[0].down_off = 4000; lay.fmt[0].bytes = 5000;
    lay.fmt[1].gu_type = 16; lay.fmt[1].d_type = 20; lay.fmt[1].up_off = 1024; lay.fmt[1].down_off = 2048; lay.fmt[1].bytes = 4096;
    lay.bytes = {5000, 4096};
    lay.offset = {0, 5000 * 4};
    lay.total = 5000 * 4 + 4096 * 4;
    return lay;
}

std::vector<uint8_t> blob(int64_t l, int64_t e) {
    std::vector<uint8_t> b(l == 0 ? 5000 : 4096);
    uint64_t x = 0x9E3779B97F4A7C15ull ^ (uint64_t) (l * 131 + e);
    for (uint8_t& v : b) { x = x * 6364136223846793005ull + 1442695040888963407ull; v = (uint8_t) (x >> 56); }
    return b;
}

struct Case {
    fs::path dir;
    ExpertLayout lay = layout();
    std::string path() const { return (dir / "experts.wex4").string(); }
    // write the container (after `edit` changed its records or bytes), open it, return the error ("" = opened)
    std::string open(const std::function<void(std::vector<wex4_test::Record>&)>& edit_records = nullptr,
                     const std::function<void(std::vector<uint8_t>&)>& edit_bytes = nullptr, const std::string& root = "",
                     strata::core::Wex4Experts* got = nullptr) {
        auto recs = wex4_test::expert_records(lay, blob);
        if (edit_records) edit_records(recs);
        auto bytes = wex4_test::file(recs);
        if (edit_bytes) edit_bytes(bytes);
        wex4_test::save(path(), bytes);
        strata::core::Wex4Experts w;
        std::string err;
        const bool ok = strata::core::wex4_experts_open(path(), lay, root, w, err);
        if (got) *got = w;
        require(ok == err.empty(), "open's result and its message disagree: " + err);
        return err;
    }
    void refused(const std::string& err, const std::string& fragment) {
        require(!err.empty(), "accepted a container that should be refused (" + fragment + ")");
        require(err.find(fragment) != std::string::npos, "the refusal does not say '" + fragment + "': " + err);
        require(err.find(path()) != std::string::npos, "the refusal does not name the file: " + err);
    }
};

}  // namespace

int main() {
    try {
        // the SHA-256 against FIPS 180-4's examples: one block, and a tail that needs a second block
        require(sha_hex("abc") == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad", "SHA-256 of 'abc'");
        require(sha_hex("abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq") ==
                    "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1", "SHA-256 of the 56-byte example");

        Case c;
        c.dir = fs::temp_directory_path() / ("strata-wex4-test-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
        fs::create_directories(c.dir);

        // a valid container: the table, and every expert's bytes where the table says
        strata::core::Wex4Experts w;
        require(c.open(nullptr, nullptr, "", &w).empty(), "a valid container was refused");
        require(w.stride == std::vector<uint64_t>{8192, 4096}, "the per-layer record stride is wrong");
        require(w.root.size() == 64 && strata::core::wex4_root_ok(w.root), "the root is not 64 hex digits");
        require(w.file_bytes == fs::file_size(c.path()), "the file size is wrong");
        {
            std::ifstream f(c.path(), std::ios::binary);
            for (int64_t l = 0; l < 2; ++l)
                for (int64_t e = 0; e < 4; ++e) {
                    const auto want = blob(l, e);
                    std::vector<uint8_t> got(want.size());
                    f.seekg((std::streamoff) (w.first[(size_t) l] + (uint64_t) e * w.stride[(size_t) l]));
                    f.read((char*) got.data(), (std::streamsize) got.size());
                    require(got == want, "layer " + std::to_string(l) + " expert " + std::to_string(e) + " is not at first + e * stride");
                }
        }

        // the root pin: the right one (upper case too) opens, another is refused with both roots
        std::string upper = w.root;
        for (char& ch : upper) ch = (char) std::toupper((unsigned char) ch);
        require(c.open(nullptr, nullptr, upper).empty(), "the right root was refused");
        std::string other = w.root;
        other[0] = other[0] == '0' ? '1' : '0';
        c.refused(c.open(nullptr, nullptr, other), "--experts-wex4-root asks for " + other);
        c.refused(c.open(nullptr, nullptr, "abc"), "64 hex digits");

        // damage: one header byte, one index byte (the hashes catch them), a truncated file
        c.refused(c.open(nullptr, [](std::vector<uint8_t>& b) { b[100] ^= 1; }), "the header page is damaged");
        c.refused(c.open(nullptr, [](std::vector<uint8_t>& b) { b[4096 + 10] ^= 1; }), "the index is damaged");
        c.refused(c.open(nullptr, [](std::vector<uint8_t>& b) { b.resize(b.size() - 4096); }), "truncated or extended");
        // a payload byte is NOT checked at open (the reader reads the header and the index only): it opens
        require(c.open(nullptr, [&](std::vector<uint8_t>& b) { b[w.first[1] + 5] ^= 1; }).empty(),
                "open hashed an expert payload; it should read the header and the index only");

        // the record rules
        c.refused(c.open([](auto& r) { r[6].codec = 5; }), "layer 1 expert 2: stored with codec WNR3 (5); this engine reads RAW records only");
        c.refused(c.open([](auto& r) { std::swap(r[1].tensors[1], r[1].tensors[2]); }),
                  "layer 0 expert 1: tensor 1 of its record is blk.0.ffn_down_exps.weight#1");
        c.refused(c.open([](auto& r) { r[0].tensors[2].dtype = 128 + 21; }), "expected blk.0.ffn_down_exps.weight#0 (dtype 148");
        c.refused(c.open([](auto& r) { r[2].gap_before = 4096; }), "layer 0 expert 2: its record is at byte");
        c.refused(c.open([](auto& r) { r.erase(r.begin() + 5); }), "layer 1 expert 1: tensor blk.1.ffn_gate_exps.weight#1 is not in the file");
        c.refused(c.open([](auto& r) {
                      wex4_test::Record x = r[3];
                      for (auto& t : x.tensors) t.name.back() = '4';   // blk.0.ffn_*_exps.weight#4: a fifth expert
                      r.insert(r.begin() + 4, x);
                  }),
                  "the file has more experts than this model (4)");

        // what the engine does not read from WEX4: a canonical (Q2_0) pack, NVFP4 layers with their scale tail
        c.lay.native = false;
        c.refused(c.open(), "needs a native pack");
        c.lay = layout();
        c.lay.fmt[1].tail_off = 4080;
        c.refused(c.open(), "layer 1 is NVFP4 with a scale tail");
        c.lay = layout();
        c.lay.fmt[0].gu_type = 144;   // Gyro's trellis types do not fit the one-byte dtype code
        c.refused(c.open(), "layer 0 has ggml type 144");
        c.lay = layout();

        std::error_code ec;
        fs::remove_all(c.dir, ec);
        std::cout << "wex4_experts_test: PASS\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "wex4_experts_test: " << e.what() << '\n';
        return 1;
    }
}
