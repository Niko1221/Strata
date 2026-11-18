#include "strata/core/expert_profile_memory.hpp"

#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

namespace {
int fails = 0;
void check(bool ok, const char* what) {
    if (!ok) {
        std::fprintf(stderr, "FAIL: %s\n", what);
        ++fails;
    }
}
bool near(double a, double b) { return std::abs(a - b) < 1e-12; }
}  // namespace

int main() {
    using strata::core::ExpertProfileMemory;
    std::string err;
    ExpertProfileMemory memory;
    memory.reset(2, 2, 1000.0);

    check(memory.consolidate({10, 0, 0, 0}, 1000.0, 12.0, 0.75, err), "first consolidation");
    check(near(memory.cell()[0], 1.0) && near(memory.scores()[0], 1.0), "an empty cell accepts the first input");
    const auto first_cell = memory.cell();
    check(memory.consolidate({10, 0, 0, 0}, 1060.0, 1.0 / 60.0, 0.75, err), "unchanged heat is accepted");
    check(memory.cell() == first_cell && near(memory.updated_at(), 1000.0),
          "periodic saving does not count the same heat again");

    check(memory.consolidate({10, 10, 0, 0}, 1060.0, 1.0 / 60.0, 0.75, err), "a task switch");
    check(near(memory.cell()[0], 0.5) && near(memory.cell()[1], 0.5), "half a half-life gives equal old and new input");
    check(memory.scores()[1] > memory.scores()[0] && memory.scores()[0] > 0.0,
          "the new task leads without erasing the previous task");

    check(memory.consolidate({10, 10, 10, 0}, 1120.0, 1.0 / 60.0, 0.75, err), "another task switch");
    check(near(memory.cell()[0], 0.25) && near(memory.cell()[1], 0.25) && near(memory.cell()[2], 0.5),
          "old evidence decays instead of disappearing");
    check(memory.scores()[0] > 0.0 && memory.scores()[1] > 0.0, "both earlier experts retain output scores");

    ExpertProfileMemory immediate;
    immediate.reset(1, 2, 50.0);
    check(immediate.consolidate({1, 0}, 50.0, 12.0, 0.65, err) &&
          immediate.consolidate({1, 1}, 50.0, 12.0, 0.65, err), "same-time input is valid");
    check(near(immediate.cell()[0], 1.0) && near(immediate.cell()[1], 0.0),
          "zero elapsed time opens the input gate rather than manufacturing a forget interval");

    ExpertProfileMemory recent_only;
    recent_only.reset(1, 2, 100.0);
    check(recent_only.consolidate({1, 0}, 100.0, 12.0, 0.0, err) &&
          recent_only.consolidate({1, 1}, 1060.0, 1.0 / 60.0, 0.0, err), "zero long-term weight is valid");
    check(near(recent_only.scores()[0], 0.0) && near(recent_only.scores()[1], 1.0),
          "zero output weight selects the recent distribution");

    const auto before_bad_input = immediate.cell();
    check(!immediate.consolidate({0, 1}, 50.0, 12.0, 0.65, err), "cumulative heat cannot go backwards");
    check(immediate.cell() == before_bad_input, "invalid input leaves the state unchanged");
    check(!immediate.consolidate({1, 1}, 50.0, 0.0, 0.65, err), "zero half-life is refused");
    check(!immediate.consolidate({1, 1}, 50.0, 12.0, 1.1, err), "output weight above one is refused");
    check(!immediate.consolidate({1}, 50.0, 12.0, 0.65, err), "wrong heat geometry is refused");

    const std::filesystem::path dir = std::filesystem::temp_directory_path() / "strata_profile_memory_test";
    std::filesystem::remove_all(dir);
    std::filesystem::create_directories(dir);
    const std::string path = (dir / "memory.strm").string();
    check(memory.save(path, err), "state saves");
    check(!std::filesystem::exists(path + ".tmp"), "atomic save leaves no temporary file");
    ExpertProfileMemory restored;
    check(restored.load(path, 2, 2, err), "state loads");
    check(restored.cell() == memory.cell() && restored.scores() == memory.cell() &&
          near(restored.updated_at(), memory.updated_at()), "a restart restores long-term scores and time");
    check(restored.consolidate({1, 0, 0, 0}, 1120.0, 12.0, 0.75, err),
          "heat is per-process and starts at zero after a restart");

    check(!restored.load(path, 3, 2, err), "another model geometry is refused");
    {
        std::ofstream out(path, std::ios::binary | std::ios::trunc);
        out << "broken";
    }
    check(!restored.load(path, 2, 2, err), "a truncated state is refused");
    check(!std::filesystem::exists(path + ".tmp"), "a refused load creates no temporary state");

    std::filesystem::remove_all(dir);
    if (fails == 0) std::puts("expert_profile_memory_test: OK");
    return fails == 0 ? 0 : 1;
}
