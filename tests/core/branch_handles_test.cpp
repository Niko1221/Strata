// Host ownership metadata only. No GPU or model is simulated by this test.
#include "strata/core/branch_handles.hpp"
#include <cstdio>
#include <limits>

using namespace strata::core;
static int checks = 0, failures = 0;
#define CHECK(x) do { ++checks; if (!(x)) { ++failures; std::fprintf(stderr,"FAIL line %d: %s\n",__LINE__,#x); } } while (0)
struct Payload {
    static int live;
    int value;
    explicit Payload(int value) : value(value) { ++live; }
    ~Payload() { --live; }
};
int Payload::live = 0;

int main() {
    BranchHandleCounters counters;
    uint64_t final_id = 0;
    {
        BranchHandles<std::shared_ptr<Payload>> handles(counters);
        const auto root = handles.insert(std::make_shared<Payload>(17));
        const auto baseline = counters;
        for (int cycle = 0; cycle < 4096; ++cycle) {
            auto value = std::make_shared<Payload>(cycle);
            std::weak_ptr<Payload> weak = value;
            const auto id = handles.insert(std::move(value));
            CHECK(id > final_id);
            final_id = id;
            CHECK(handles.find(id) && (*handles.find(id))->value == cycle);
            CHECK(handles.drop(id));
            CHECK(weak.expired());
            CHECK(handles.find(id) == nullptr);
            CHECK(Payload::live == 1);
            if (cycle == 0 || cycle == 15 || cycle == 255 || cycle == 4095) {
                CHECK(counters.live_entries == baseline.live_entries);
                CHECK(counters.live_bytes == baseline.live_bytes);
                CHECK(counters.allocations - baseline.allocations == counters.releases - baseline.releases);
                std::printf("{\"cycle\":%d,\"live_entries\":%llu,\"live_bytes\":%llu,\"baseline_bytes\":%llu}\n",
                            cycle, (unsigned long long)counters.live_entries,
                            (unsigned long long)counters.live_bytes, (unsigned long long)baseline.live_bytes);
            }
        }
        CHECK(handles.find(root) && (*handles.find(root))->value == 17);
        CHECK(handles.drop(final_id)); // Release retries must be harmless.
        CHECK(!handles.drop(std::numeric_limits<uint64_t>::max()));
        CHECK(handles.find(std::numeric_limits<uint64_t>::max()) == nullptr);
        handles.clear();
        CHECK(Payload::live == 0);
        CHECK(counters.live_entries == 0);
        CHECK(counters.live_bytes == 0);
        CHECK(counters.allocations == counters.releases);
        const auto replacement = handles.insert(std::make_shared<Payload>(29));
        CHECK(replacement > final_id);
        CHECK(handles.find(root) == nullptr && handles.find(final_id) == nullptr);
        CHECK(handles.find(replacement) && (*handles.find(replacement))->value == 29);
        CHECK(handles.drop(root));
        CHECK(handles.find(replacement) != nullptr);
        const auto issued = counters.issued;
        bool refused = false;
        try { handles.insert({}); } catch (const std::invalid_argument&) { refused = true; }
        CHECK(refused && counters.issued == issued && counters.live_entries == 1);
    }
    CHECK(Payload::live == 0);
    CHECK(counters.live_entries == 0 && counters.live_bytes == 0);
    CHECK(counters.allocations == counters.releases);
    std::printf("{\"status\":\"%s\",\"checks\":%d,\"failures\":%d,\"cycles\":4096,"
                "\"allocations\":%llu,\"releases\":%llu,\"peak_bytes\":%llu,\"issued\":%llu,"
                "\"live_bytes\":%llu,\"model_forward\":false,\"gpu_used\":false}\n",
                failures ? "FAIL" : "PASS", checks, failures,
                (unsigned long long)counters.allocations, (unsigned long long)counters.releases,
                (unsigned long long)counters.peak_bytes, (unsigned long long)counters.issued,
                (unsigned long long)counters.live_bytes);
    return failures ? 1 : 0;
}
