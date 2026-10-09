// CPU-only exercise of the real verify-window planner: synthetic native layout, host pointers and callbacks.
// No CUDA API, GPU allocation, model, numerical GPU parity claim or download. A null pool in the quality
// cases also proves that no CPU expert job is run. GPU kernel/address parity remains a separate validation.
#include "strata/core/expert_source.hpp"
#include "strata/core/verify.hpp"
#include "strata/kernels/cpu/expert_layout.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <numeric>
#include <string>
#include <vector>

namespace core = strata::core;
namespace cpu = strata::kernels::cpu;
namespace fs = std::filesystem;
namespace {
constexpr int E = 128, H = 2048, K = 8, CAP = 128;
int checks = 0;
void check(bool ok, const char* what) {
    ++checks;
    if (!ok) { std::fprintf(stderr, "FAIL: %s\n", what); std::exit(1); }
}

struct LayoutFixture {
    fs::path path = fs::temp_directory_path() /
        ("strata-cap-quality-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    LayoutFixture() {
        cpu::NativeFmt fmt;
        std::string err;
        check(cpu::native_fmt(12, 14, H, 512, fmt, err), "synthetic Q4_K/Q6_K format");
        fs::create_directories(path);
        { std::ofstream out(path / "native_experts.txt"); out << "0 12 14 0 " << fmt.bytes << '\n'; }
        check(cpu::expert_layout_load(path.string(), 1, E, err, H, 512), "synthetic native layout loads");
    }
    ~LayoutFixture() { std::error_code ignored; fs::remove_all(path, ignored); }
};

struct Source : core::ExpertSource {
    std::vector<uint8_t> bytes = std::vector<uint8_t>((size_t) cpu::expert_layout().max_blob, 0);
    bool layer_ok = true;
    int unpinned = -1, unmapped = -1, missing = -1, begins = 0, blobs = 0;
    const uint8_t* blob(int64_t, int64_t expert) override { ++blobs; return expert == missing ? nullptr : bytes.data(); }
    bool pinned(int64_t, int64_t expert) const override { return expert != unpinned; }
    const uint8_t* device_alias(int64_t, int64_t expert) const override {
        return expert == unmapped ? nullptr : bytes.data() + expert; // host-only address fingerprint; never dereferenced
    }
    bool pcie_layer(int64_t) const override { return layer_ok; }
    void begin_layer(int64_t, const int32_t*, int64_t) override { ++begins; }
};

struct Plan {
    core::GpuPlanSink sink;
    std::array<int32_t, 4> counts{};
    std::array<int32_t, CAP + 1> start{}, start2{};
    std::array<int32_t, CAP> dst{}, tok{};
    std::array<unsigned long long, CAP> ptr{}, ptr2{};
    int published = 0, fetched = 0, dma_blobs = -1;
    Plan() {
        sink.counts = counts.data(); sink.start = start.data(); sink.start2 = start2.data();
        sink.dst = dst.data(); sink.tok = tok.data(); sink.ptr = ptr.data(); sink.ptr2 = ptr2.data();
        sink.cap = CAP; sink.staging_cap = core::Verifier::pcie_staging_capacity(2, K, false);
        sink.pcie_mode = 2; sink.staging = 0x10000000ull;
        sink.ctx = this;
        sink.publish = [](void* p) { ++static_cast<Plan*>(p)->published; };
        sink.fetch = [](void* p, const uint8_t* const* src, int n, size_t bytes) {
            auto& self = *static_cast<Plan*>(p);
            check(self.published == 1, "plan published before fetch callback");
            check(bytes == cpu::expert_layout().blob_bytes(0), "fetch carries native blob size");
            for (int i = 0; i < n; ++i) check(src[i] != nullptr, "DMA source exists");
            ++self.fetched; self.dma_blobs = n;
        };
    }
};

struct Case {
    Source source;
    Plan plan;
    core::ExpertDispatch d;
    std::array<int32_t, E> res;
    std::array<uint64_t, E> offsets;
    std::array<uint8_t, E * 256> cache{};
    std::vector<int32_t> ids;
    std::vector<float> x, out;
    int tokens, width;
    Case(int nt, int k = K, bool quality = true) : tokens(nt), width(k) {
        res.fill(-1);
        for (int e = 0; e < E; ++e) offsets[(size_t) e] = (uint64_t) e * 256;
        ids.resize((size_t) nt * k); std::iota(ids.begin(), ids.end(), 0);
        x.assign((size_t) nt * H, 1.0f); out.assign(ids.size() * H, 7.0f);
        d.src = &source; d.n_expert = E; d.host_res = res.data(); d.plan = &plan.sink;
        d.cache_base = cache.data(); d.cache_blob = 256; d.cache_slot_off = offsets.data();
        d.require_gpu_experts = quality; d.pcie_num = 256;
        plan.sink.cap = (int64_t) nt * k;
        plan.sink.staging_cap = core::Verifier::pcie_staging_capacity(nt, k, quality);
        // d.pool remains nullptr: any unguarded CPU fallback in a quality case fails the test.
    }
    void run() { core::expert_pool_dispatch_multi(d, x.data(), ids.data(), tokens, width, out.data()); }
    void check_success(int hits, int misses) {
        check(!d.failed, "quality planner succeeds");
        check(plan.published == 1 && plan.fetched == 1, "exactly one publish/fetch");
        check(plan.counts[0] == hits && plan.counts[2] == misses, "hit and PCIe group counts");
        check(plan.counts[1] == (int) ids.size(), "every routed entry belongs to the GPU plan");
        check(d.multi_misses == 0 && d.multi_entries == 0, "no CPU expert work");
        std::vector<int> seen(ids.size(), 0);
        for (int i = 0; i < plan.counts[1]; ++i) {
            const int at = plan.dst[(size_t) i];
            check(at >= 0 && at < (int) ids.size(), "planned destination in bounds");
            check(plan.tok[(size_t) i] == at / width, "planned token matches routed row");
            ++seen[(size_t) at];
        }
        check(std::all_of(seen.begin(), seen.end(), [](int n) { return n == 1; }), "each row planned exactly once");
        check(std::all_of(out.begin(), out.end(), [](float f) { return f == 0; }), "host contributions zeroed");
    }
    void check_failure() {
        check(d.failed && d.fail && std::string(d.fail).find("quality") != std::string::npos, "quality refusal latched");
        check(plan.published == 0 && plan.fetched == 0, "failed plan not published (verifier drains empty plan)");
        check(d.multi_misses == 0 && d.multi_entries == 0, "refusal cannot run a CPU fallback");
        check(std::all_of(out.begin(), out.end(), [](float f) { return f == 0; }), "refusal zeroes host contributions");
    }
};

void same_plan(const Case& a, const Case& b) {
    check(a.plan.counts == b.plan.counts && a.plan.start == b.plan.start && a.plan.start2 == b.plan.start2 &&
          a.plan.dst == b.plan.dst && a.plan.tok == b.plan.tok, "quality metadata matches --pcie-frac 1");
    // Pointers refer to separate fixtures, so compare their offsets instead of their absolute addresses.
    for (int q = 0; q < a.plan.counts[0]; ++q)
        check(a.plan.ptr[(size_t) q] - (unsigned long long) a.cache.data() ==
              b.plan.ptr[(size_t) q] - (unsigned long long) b.cache.data(), "hit address offsets unchanged");
    for (int q = 0; q < a.plan.counts[2]; ++q)
        check(a.plan.ptr2[(size_t) q] - (unsigned long long) a.source.bytes.data() ==
              b.plan.ptr2[(size_t) q] - (unsigned long long) b.source.bytes.data(), "PCIe address offsets unchanged");
}
} // namespace

int main() {
    // Exercise host zeroing without a GPU; these switches are read once by the real dispatch.
#ifdef _WIN32
    _putenv_s("STRATA_DEC_BATCH", "0"); _putenv_s("STRATA_VERIFY_DEVICE_PLAN", "0");
#else
    setenv("STRATA_DEC_BATCH", "0", 1); setenv("STRATA_VERIFY_DEVICE_PLAN", "0", 1);
#endif
    LayoutFixture layout;
    for (int nt : {2, 3, 4, 5, 6, 7, 8})
        for (int k : {8, 10}) {
            const int64_t capacity = core::Verifier::pcie_staging_capacity(nt, k, true);
            check(core::Verifier::pcie_staging_capacity(nt, k, false) == 16, "default staging remains 16 blobs");
            check(capacity >= nt * k, "quality staging covers every possible entry");
            check(capacity / 2 >= ((nt + 1) / 2) * k, "each split-window half covers the larger token group");
        }
    for (int nt : {1, 2, 4, 8}) {
        for (int placement : {0, 1, 2}) { // all misses, alternating hits, all hits
            Case quality(nt), reference(nt, K, false);
            // Compare the legacy pcie_num=256 planner with sufficient staging, not its default 16-blob CPU spill.
            reference.plan.sink.staging_cap = quality.plan.sink.staging_cap;
            for (int e : quality.ids)
                if (placement == 2 || (placement == 1 && e % 2)) quality.res[(size_t) e] = reference.res[(size_t) e] = e;
            quality.d.pcie_num = 0; // defense in depth against a lower CLI/request share
            quality.run(); reference.run();
            const int hits = placement == 0 ? 0 : placement == 1 ? nt * K / 2 : nt * K;
            quality.check_success(hits, nt * K - hits); reference.check_success(hits, nt * K - hits);
            same_plan(quality, reference);
        }
    }
    { Case c(8); for (size_t i = 0; i < c.ids.size(); ++i) c.ids[i] = (int32_t) (i % K);
      c.run(); c.check_success(0, K); check(c.d.pcie_experts == K, "repeated experts staged once per group"); }
    { Case c(2); c.plan.sink.pcie_mode = 0; c.run(); c.check_success(0, 2 * K);
      check(c.plan.dma_blobs == 2 * K, "DMA path stages every miss"); }
    { Case c(2); c.plan.sink.pcie_mode = 1; c.run(); c.check_success(0, 2 * K);
      check(c.plan.dma_blobs == 0, "direct path uses GPU-visible aliases"); }
    { Case c(1); c.d.plan = nullptr; c.run(); c.check_failure(); }
    { Case c(1); c.plan.sink.cap = K - 1; c.run(); c.check_failure(); }
    { Case c(1); c.d.host_res = nullptr; c.run(); c.check_failure(); }
    { Case c(1); c.source.layer_ok = false; c.run(); c.check_failure(); }
    { Case c(1); c.source.unpinned = 0; c.run(); c.check_failure(); }
    { Case c(1); c.source.unmapped = 0; c.run(); c.check_failure(); }
    { Case c(1); c.source.missing = 0; c.run(); c.check_failure(); }
    { Case c(1); c.ids[0] = E; c.run(); c.check_failure(); }
    { Case c(1); c.plan.sink.staging_cap = K - 1; c.run(); c.check_failure(); }
    { Case c(8, 10); c.run(); c.check_success(0, 80); } // quality lifts both the 16-blob staging and 64-entry host limit
    { Case c(8, 10); c.plan.sink.staging_cap = 64; c.run(); c.check_failure(); }
    // Default-off negative controls: the CPU contract and the existing 16-blob spill stay as before.
    { Case c(1, 1, false); c.source.layer_ok = false; c.d.pcie_num = 0;
      cpu::ExpertPool pool(1, false, false); c.d.pool = &pool; c.run();
      check(!c.d.failed && c.d.multi_entries == 1 && c.d.multi_misses == 1, "flag absent retains CPU fallback");
      check(c.plan.counts[1] == 0, "CPU fallback is not in GPU plan");
      check(std::all_of(c.out.begin(), c.out.end(), [](float f) { return f == 0; }), "zero native CPU expert result"); }
    { Case c(4, K, false);
      cpu::ExpertPool pool(1, false, false); c.d.pool = &pool; c.run();
      check(!c.d.failed && c.d.multi_entries == 16 && c.d.multi_misses == 16, "flag absent: --pcie-frac 1 still spills past 16");
      check(c.plan.counts[1] == 16 && c.plan.counts[2] == 16, "default staging plan unchanged");
      check(std::all_of(c.out.begin(), c.out.end(), [](float f) { return f == 0; }), "default spill uses zero native CPU experts"); }
    std::printf("vram_cap_quality_test: %d CPU checks passed (no GPU calls)\n", checks);
    return 0;
}
