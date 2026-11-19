#include "strata/qwen35/bounded_experts.hpp"
#include "strata/kernels/cpu/pool.hpp"

#include <array>
#include <cstdio>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <thread>

using namespace strata::qwen35;
using strata::kernels::cpu::ExpertJobMulti;
static int failures = 0;
#define CHECK(x) do { if (!(x)) { std::fprintf(stderr, "FAIL line %d: %s\n", __LINE__, #x); ++failures; } } while (0)

int main() {
    CHECK(q36_memory_mode_error("legacy", "legacy", 0).empty());
    CHECK(q36_memory_mode_error("legacy", "mapped-pack", 0).empty());
    CHECK(q36_memory_mode_error("bounded-experts-v1", "mapped-pack", 5).empty());
    CHECK(!q36_memory_mode_error("bounded-experts-v1", "legacy", 5).empty());
    CHECK(!q36_memory_mode_error("bounded-experts-v1", "mapped-pack", 0).empty());
    CHECK(!q36_memory_mode_error("bounded-experts-v2", "mapped-pack", 5).empty());
    CHECK(q36_explicit_slots("5") == 5);
    for (const char* value : {"", "0", "-1", "5junk", "1.5", "2147483648"}) {
        bool rejected = false;
        try { (void) q36_explicit_slots(value); } catch (const std::exception&) { rejected = true; }
        CHECK(rejected);
    }
    constexpr uint64_t GiB = 1ull << 30;
    CHECK(q36_device_headroom(GiB, GiB + (768u << 20)));
    CHECK(!q36_device_headroom(GiB + 1, GiB + (768u << 20)));
    CHECK(!q36_device_headroom(std::numeric_limits<uint64_t>::max(), 2 * GiB));
    CHECK(q36_host_headroom(64, 3 * GiB + 64, 4 * GiB + 64));
    CHECK(!q36_host_headroom(65, 3 * GiB + 64, 4 * GiB + 64));
    CHECK(!q36_host_headroom(0, 3 * GiB - 1, 5 * GiB));
    CHECK(!q36_host_headroom(0, 5 * GiB, 4 * GiB - 1));
    CHECK(!q36_host_headroom(std::numeric_limits<uint64_t>::max(), 5 * GiB, 5 * GiB));
    for (size_t bytes : {size_t(0), std::numeric_limits<size_t>::max(), size_t(64ull << 20)}) {
        bool rejected = false;
        try { ExpertCpuStaging stage(bytes); } catch (const std::exception&) { rejected = true; }
        CHECK(rejected);
    }
    ExpertCpuStaging stage(64);
    CHECK(stage.committed_bytes() == ExpertCpuStaging::planned_bytes(64));
    CHECK(stage.committed_bytes() <= ExpertCpuStaging::kBudgetBytes);

    // More jobs than slots, including a repeated expert with a separate token
    // group. Use the real job descriptor and keep every output/activation/nt.
    std::array<std::array<uint8_t, 64>, 19> source{};
    std::array<ExpertJobMulti, 19> jobs{};
    std::array<float, 19> output{};
    for (size_t i = 0; i < jobs.size(); ++i) {
        source[i].fill((uint8_t) (i + 1));
        jobs[i].blob = source[i].data();
        jobs[i].nt = int(i % 8 + 1);
        jobs[i].out[0] = &output[i];
        jobs[i].nact[0] = &jobs[i];
    }
    jobs.back().blob = jobs[0].blob;
    size_t reads = 0, consumed = 0, waves = 0;
    bool reading = false;
    auto read = [&](const uint8_t* src, uint8_t* dst, size_t n) {
        CHECK(!reading);
        ++reads;
        std::memcpy(dst, src, n);
        return true;
    };
    auto consume = [&](ExpertJobMulti* batch, size_t count) {
        CHECK(count <= ExpertCpuStaging::kSlots);
        CHECK(reads == consumed + count);
        reading = true;
        // A synchronous consumer may have workers; their storage must survive
        // until every reader has joined, not merely until work is submitted.
        std::thread worker([&] {
            for (size_t j = 0; j < count; ++j) {
                const auto& original = jobs[consumed + j];
                CHECK(batch[j].blob != original.blob);
                CHECK(std::memcmp(batch[j].blob, original.blob, 64) == 0);
                CHECK(batch[j].nt == original.nt && batch[j].nact[0] == original.nact[0]);
                CHECK(batch[j].out[0] == original.out[0]);
                *batch[j].out[0] = float(batch[j].blob[0]);
            }
        });
        worker.join();
        reading = false;
        consumed += count;
        ++waves;
    };
    CHECK(stage.run(jobs.data(), jobs.size(), 64, read, consume));
    CHECK(consumed == jobs.size() && waves == 3);
    CHECK(output[0] == 1 && output[18] == 1 && output[17] == 18);
    for (size_t i = 0; i < jobs.size(); ++i) CHECK(jobs[i].blob == source[i == 18 ? 0 : i].data());

    // No partial wave reaches the pool, and a failed wave leaves staging reusable.
    size_t bad_reads = 0, bad_consumes = 0;
    CHECK(!stage.run(jobs.data(), jobs.size(), 64,
        [&](const uint8_t* src, uint8_t* dst, size_t n) {
            std::memcpy(dst, src, n);
            return ++bad_reads != 2;
        }, [&](ExpertJobMulti*, size_t) { ++bad_consumes; }));
    CHECK(bad_consumes == 0);
    CHECK(!stage.run(jobs.data(), 1, 65, read, consume));
    bool threw = false;
    try { stage.run(jobs.data(), 1, 64, read, [](ExpertJobMulti*, size_t) { throw std::runtime_error("consumer"); }); }
    catch (const std::runtime_error&) { threw = true; }
    CHECK(threw);
    CHECK(stage.run(jobs.data(), 1, 64, read, [&](ExpertJobMulti*, size_t) {
        CHECK(!stage.run(jobs.data(), 1, 64, read, [](ExpertJobMulti*, size_t) {}));
    }));
    CHECK(stage.run<ExpertJobMulti>(nullptr, 0, 64, read, consume));
    std::fprintf(stderr, "q36_bounded_experts_test: %d failures\n", failures);
    return failures ? 1 : 0;
}
