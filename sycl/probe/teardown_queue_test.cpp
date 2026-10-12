// No model files: destructor-owned queues must never delete DPCT's borrowed
// default queue. Keep a tiny USM allocation alive across each teardown, as the
// MTP weights are alive when the CLI's later-created Prefill is destroyed.
#define DPCT_PROFILING_ENABLED
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include "strata/core/mtp.hpp"
#include "strata/core/weights.hpp"
#include "strata/prefill/prefill.hpp"
#include <array>
#include <cstdio>
#include <functional>
#include <stdexcept>
#include <string>

static void require(bool ok, const char* reason) {
    if (!ok) throw std::runtime_error(reason);
}
static void survives(const char* label, const std::function<void()>& teardown) {
    std::fprintf(stderr, "teardown queue test: %s\n", label);
    auto& before = dpct::get_in_order_queue();
    const auto context = before.get_context();
    const auto device = before.get_device();
    constexpr uint32_t value = 0x51A7C0DEu;
    auto* allocation = sycl::malloc_device<uint32_t>(16, before);
    require(allocation != nullptr, "tiny USM allocation failed");
    try {
        before.fill(allocation, value, 16).wait_and_throw();
        teardown();
        auto& after = dpct::get_in_order_queue();
        require(&after == &before, "default queue was replaced by teardown");
        require(after.get_context() == context && after.get_device() == device,
                "default queue context/device invalid after teardown");
        std::array<uint32_t, 16> result{};
        after.memcpy(result.data(), allocation, sizeof result).wait_and_throw();
        for (auto v : result) require(v == value, "live USM contents changed during teardown");
        auto* to_free = allocation;
        allocation = nullptr;
        sycl::free(to_free, after); // The same default-queue free as ~MtpDrafter.
    } catch (...) {
        if (allocation) sycl::free(allocation, context);
        throw;
    }
}
int main() {
    try {
        for (int request = 0; request < 8; ++request) {
            survives("unloaded MTP", [] { strata::core::MtpDrafter mtp; });
            survives("MTP rejected before queue creation", [] {
                strata::core::ModelGeometry geometry;
                strata::core::SessionState session;
                std::string error;
                strata::core::MtpDrafter mtp;
                require(!mtp.load("unused-no-model-files", geometry, session, 0, error), "invalid max_t accepted");
                require(error == "mtp: max_t out of range", "unexpected failed-load path");
            });
            survives("uninitialized prefill", [] { strata::prefill::Prefill prefill; });
            survives("prefill reset/reuse", [] {
                strata::prefill::Prefill prefill;
                for (int i = 0; i < 4; ++i) prefill.reset();
            });
            survives("prefill rejected before queue creation", [] {
                strata::core::WeightTable weights;
                strata::core::ModelGeometry geometry;
                geometry.n_embd = 1;
                strata::core::SessionState session;
                std::string error;
                strata::prefill::Prefill prefill;
                require(!prefill.init(weights, geometry, session, nullptr, nullptr, nullptr, 1, nullptr, error, nullptr, 0),
                        "unsupported geometry accepted");
                require(error == "prefill: geometry differs from the artifact's", "unexpected failed-init path");
            });
        }
        std::puts("teardown queue PASS: 40 lifecycles, default queue and live USM preserved");
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "teardown queue FAIL: %s\n", e.what());
        return 1;
    }
}
