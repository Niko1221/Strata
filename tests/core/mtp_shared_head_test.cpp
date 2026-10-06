// Exercise the real MtpDrafter::bind shared-head branch with tiny synthetic weights, not a model.
// Reverting the metadata copy in mtp.cpp must fail the checks after the real bind call.
#include "gguf_fixture.hpp"
#include "strata/core/mtp.hpp"
#include "strata/core/native_head.hpp"
#include "strata/kernels/native_mmvq.hpp"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <stdexcept>

namespace strata::core {
struct MtpBindTestAccess {
    static void init(MtpDrafter& d, const ModelGeometry& g) {
        d.g_ = &g;
        d.max_t_ = 2;
        cudaGetDevice(&d.device_);
    }
    static void check_shared(const MtpDrafter& d, const MtpDrafter& owner, int expected) {
        if (d.dhead_ != owner.dhead_ || d.dvocab_ != owner.dvocab_ ||
            d.n_dvocab_ != owner.n_dvocab_ || d.owns_draft_head_)
            throw std::runtime_error("shared head pointers/count/ownership differ");
        if (d.dhead_type_ != expected)
            throw std::runtime_error("shared draft head type is " + std::to_string(d.dhead_type_) +
                                     ", expected " + std::to_string(expected));
        if (d.dvocab_host_ != owner.dvocab_host_)
            throw std::runtime_error("shared head lost the host token map (top2 diagnostics)");
        if (d.dhead_) {
            // Validate the bound format with the native weight-size dispatcher (no MMVQ launch here).
            (void) strata::kernels::native_mmvq_weight_bytes(d.dhead_type_, 256, (int) d.n_dvocab_);
        }
    }
    static void check_alive(const MtpDrafter& d) {
        if (!d.dhead_) return;
        unsigned char byte = 0;
        if (cudaMemcpy(&byte, d.dhead_, 1, cudaMemcpyDeviceToHost) != cudaSuccess)
            throw std::runtime_error("destroying a slot freed the owner's shared head");
    }
};
}  // namespace strata::core

namespace fs = std::filesystem;
using namespace strata::core;
namespace {
void require(bool ok, const std::string& why) { if (!ok) throw std::runtime_error(why); }
struct TempDir {
    fs::path path = fs::current_path() / ("mtp-shared-test-" +
        std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    TempDir() { fs::create_directories(path); }
    ~TempDir() { std::error_code ec; fs::remove_all(path, ec); }
};
std::vector<fixture::Kv> arch_keys() {
    return {fixture::str("general.architecture", "qwen4exp"), fixture::u32("qwen4exp.block_count", 48),
        fixture::u32("qwen4exp.embedding_length", 2560), fixture::u32("qwen4exp.expert_count", 512),
        fixture::u32("qwen4exp.expert_used_count", 10), fixture::u32("qwen4exp.attention.head_count", 24),
        fixture::u32("qwen4exp.attention.head_count_kv", 2)};
}
void run_case(const fs::path& path, int head_type, bool q4, bool subset) {
    const auto gguf = path / "head.gguf";
    const auto written = fixture::write(gguf, arch_keys(), {{"output.weight", {256, 4}, (uint32_t) head_type, 1}});
    // Zero blocks give finite values when the Q4 override dequantizes/requantizes the native head.
    {
        std::fstream f(gguf, std::ios::binary | std::ios::in | std::ios::out);
        f.seekp((std::streamoff) written.data_start);
        std::vector<char> zero((size_t) (written.size - written.data_start));
        f.write(zero.data(), (std::streamsize) zero.size());
        require(bool(f), "writing zero head blocks");
    }
    NativeHead head;
    std::string err;
    require(head.load({gguf.string()}, 256, 4, err), "native head: " + err);
    // Metadata-only output.weight: bind needs its vocabulary size, not a full model arena.
    std::ofstream(path / "index.txt") << "# align 256 pool 256 tensors 1\n"
        "output.weight 0 0 0 0 0 0 256 4 8 0 32 0 0 0 0 0 0 0\n";
    WeightTable wt;
    const std::set<std::string> skip{"output.weight"};
    require(wt.load(path.string(), nullptr, 0, err, &skip), "weight metadata: " + err);
    const auto vocab = path / "vocab.bin";
    if (subset) {
        const int32_t ids[] = {3, 1};
        std::ofstream(vocab, std::ios::binary).write((const char*) ids, sizeof(ids));
    }
    ModelGeometry g;
    g.n_embd = 256;
    MtpDrafter owner;
    MtpBindTestAccess::init(owner, g);
    owner.set_draft_vocab(vocab.string());
    owner.set_q4(false, q4);
    require(owner.bind(wt, &head, nullptr, err), "owner bind: " + err);
    const int expected = subset ? (q4 ? 2 : head_type) : -1;
    for (int i = 0; i < 2; ++i) {
        {
            MtpDrafter slot;
            MtpBindTestAccess::init(slot, g);
            require(slot.bind(wt, &head, nullptr, err, &owner), "slot bind: " + err);
            MtpBindTestAccess::check_shared(slot, owner, expected);
        }
        MtpBindTestAccess::check_alive(owner);
    }
    // A different head must still be refused without taking ownership of any source buffers.
    NativeHead other;
    require(other.load({gguf.string()}, 256, 4, err), "second head: " + err);
    MtpDrafter rejected;
    MtpBindTestAccess::init(rejected, g);
    require(!rejected.bind(wt, &other, nullptr, err, &owner) && err == "mtp: incompatible shared draft head",
            "incompatible shared head must be rejected");
    std::printf("PASS head=%d subset=%d q4=%d: both slots retain metadata and leave owner alive\n",
                head_type, subset, q4);
}
}  // namespace
int main() {
    // Do not inherit diagnostic switches that suppress the subset or enable unrelated draft sampling.
#if defined(_WIN32)
    _putenv_s("STRATA_MTP_FULL_HEAD", "0");
    _putenv_s("STRATA_SPEC_COUPLED", "0");
#else
    setenv("STRATA_MTP_FULL_HEAD", "0", 1);
    setenv("STRATA_SPEC_COUPLED", "0", 1);
#endif
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) {
        std::puts("mtp_shared_head_test: no GPU, skipped");
        return 77;
    }
    try {
        TempDir tmp;
        std::vector<std::vector<int>> cases{{13, 0, 1}, {23, 0, 1}, {13, 0, 0}};
#ifdef STRATA_NATIVE_EXPERTS
        cases.push_back({13, 1, 1});
#endif
        for (const auto& c : cases) {
            const auto path = tmp.path / (std::to_string(c[0]) + "-" + std::to_string(c[1]) + "-" + std::to_string(c[2]));
            fs::create_directories(path);
            run_case(path, c[0], c[1] != 0, c[2] != 0);
        }
        std::puts("mtp_shared_head_test: all passed");
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "mtp_shared_head_test: FAIL: %s\n", e.what());
        return 1;
    }
}
