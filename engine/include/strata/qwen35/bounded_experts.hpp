#pragma once
#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace strata::qwen35 {

inline int q36_explicit_slots(const std::string& value) {
    int slots = 0;
    for (char c : value) {
        if (c < '0' || c > '9' || slots > (std::numeric_limits<int>::max() - (c - '0')) / 10)
            throw std::invalid_argument("bounded-experts-v1 requires a positive decimal --expert-slots count");
        slots = slots * 10 + (c - '0');
    }
    if (!slots) throw std::invalid_argument("bounded-experts-v1 requires explicit positive --expert-slots");
    return slots;
}

inline std::string q36_memory_mode_error(const std::string& mode, const std::string& store, int slots) {
    if (mode == "legacy") return {};
    if (mode != "bounded-experts-v1") return "--q36-memory takes legacy or bounded-experts-v1";
    if (store != "mapped-pack" || slots <= 0)
        return "bounded-experts-v1 requires --q36-expert-store mapped-pack and explicit positive --expert-slots";
    return {};
}

// Admission for additional host allocations, not an estimate of total model fit.
inline bool q36_host_headroom(uint64_t additional, uint64_t physical, uint64_t commit) {
    constexpr uint64_t physical_floor = 3ull << 30, commit_floor = 4ull << 30;
    return physical >= physical_floor && commit >= commit_floor &&
           additional <= physical - physical_floor && additional <= commit - commit_floor;
}
bool q36_require_host_headroom(uint64_t additional, std::string& err);

inline bool q36_device_headroom(uint64_t additional, uint64_t available) {
    constexpr uint64_t reserve = 768ull << 20;
    return available >= reserve && additional <= available - reserve;
}

// One synchronous CPU wave, independent of the H2D pin slots and their events.
// Jobs retain their token grouping, arithmetic selection and output addresses.
// The consumer must join all readers before returning (ExpertPool does this).
class ExpertCpuStaging {
public:
    static constexpr size_t kSlots = 8;
    static constexpr size_t kBudgetBytes = 64u << 20;
    static constexpr size_t kTailBytes = 4096;

    static size_t planned_bytes(size_t max_blob_bytes) {
        // Check the cap before arithmetic/allocation, including alignment/tail.
        if (!max_blob_bytes || max_blob_bytes > kBudgetBytes / kSlots - kTailBytes - 63)
            throw std::invalid_argument("Q36 CPU expert staging exceeds the 64 MiB budget");
        return ((max_blob_bytes + kTailBytes + 63) & ~size_t(63)) * kSlots;
    }
    explicit ExpertCpuStaging(size_t max_blob_bytes)
        : storage_(planned_bytes(max_blob_bytes)), max_blob_bytes_(max_blob_bytes),
          slot_bytes_(storage_.size() / kSlots) {}
    ExpertCpuStaging(const ExpertCpuStaging&) = delete;
    ExpertCpuStaging& operator=(const ExpertCpuStaging&) = delete;
    size_t committed_bytes() const { return storage_.size(); }

    // A failed read never publishes its partial wave. Earlier completed waves
    // may have written outputs; the engine must fail the request, not merge them.
    // Input job descriptors are never rewritten, even on consumer exceptions.
    template <class Job, class Read, class Consume>
    bool run(const Job* jobs, size_t count, size_t blob_bytes, Read&& read, Consume&& consume) {
        if (busy_ || !blob_bytes || blob_bytes > max_blob_bytes_ || (count && !jobs)) return false;
        busy_ = true;
        struct Reset { bool& busy; ~Reset() { busy = false; } } reset{busy_};
        std::array<Job, kSlots> wave{};
        for (size_t base = 0; base < count;) {
            const size_t take = std::min(kSlots, count - base);
            for (size_t i = 0; i < take; ++i) {
                wave[i] = jobs[base + i];
                uint8_t* dst = storage_.data() + i * slot_bytes_;
                if (!read(wave[i].blob, dst, blob_bytes)) return false;
                std::memset(dst + blob_bytes, 0, slot_bytes_ - blob_bytes);
                wave[i].blob = dst;
            }
            consume(wave.data(), take);
            base += take;
        }
        return true;
    }

private:
    std::vector<uint8_t> storage_;
    size_t max_blob_bytes_, slot_bytes_;
    bool busy_ = false;  // single owning engine thread; reentrant use is rejected
};
}  // namespace strata::qwen35
