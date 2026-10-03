// TuningTable::load header gate, synthetic tables: no model, no GPU.  gfx1200 and gfx1201 are the same RDNA4
// (gfx12) family and share hipBLASLt solution IDs, so a table calibrated on one must load on the other: a card
// reporting the sibling architecture - an R9700 under HSA_OVERRIDE_GFX_VERSION=12.0.0 reports gfx1200 - otherwise
// rejects its own calibration and loses the tuned prefill route entirely.  The per-solution runtime gate
// (getAlgosFromIndex + matmulIsAlgoSupported, hipBLASEx fallback) is what keeps a wrong solution ID from running,
// so the header gate can be family-wide: RDNA3 siblings (gfx1100/gfx1101) share it, and every cross-family pair
// and every version mismatch stays rejected.
#include "hipblaslt_tuning.hpp"

#include <cstdio>
#include <fstream>
#include <string>

#include <unistd.h>

using strata::prefill::hipblaslt::TuningTable;

namespace {

std::string case_path() {
    return "/tmp/strata_tuning_header_case_" + std::to_string(getpid()) + ".txt";
}

bool load(const std::string& file_arch, int file_version, const std::string& runtime_arch, int runtime_version,
          std::string& err, size_t& rows) {
    const std::string path = case_path();
    {
        std::ofstream out(path);
        out << "# solution IDs are scoped to this hipBLASLt version and device architecture\n"
            << "STRATA_HIPBLASLT_TUNING_V1 " << file_arch << " " << file_version << "\n"
            << "bf16 1 2560 1 4096 110508\n"
            << "bf16 2560 320 2560 8192 135699\n";
    }
    TuningTable table;
    const bool ok = table.load(path, runtime_arch, runtime_version, err);
    rows = table.rows().size();
    std::remove(path.c_str());
    return ok;
}

struct Expect {
    const char* file_arch;
    int file_version;
    const char* runtime_arch;
    int runtime_version;
    bool accept;
    const char* why;
};

}  // namespace

int main() {
    const Expect cases[] = {
        {"gfx1201", 100500, "gfx1201", 100500, true, "exact match"},
        {"gfx1200", 100500, "gfx1200", 100500, true, "exact match"},
        {"gfx1201", 100500, "gfx1200", 100500, true, "RDNA4 sibling: gfx1201 table on a card reporting gfx1200"},
        {"gfx1200", 100500, "gfx1201", 100500, true, "RDNA4 sibling, symmetric direction"},
        {"gfx1201", 100202, "gfx1201", 100500, false, "the version gate rejects on its own"},
        {"gfx1200", 100202, "gfx1201", 100500, false, "cross-family and wrong version"},
        {"gfx1100", 100202, "gfx1201", 100202, false, "RDNA3 table on RDNA4"},
        {"gfx1201", 100202, "gfx1100", 100202, false, "RDNA4 table on RDNA3"},
        {"gfx1100", 100202, "gfx1101", 100202, true, "RDNA3 siblings share the family rule"},
        {"gfx1030", 100202, "gfx1201", 100202, false, "RDNA2 table on RDNA4"},
    };
    int failures = 0;
    for (const auto& c : cases) {
        std::string err;
        size_t rows = 0;
        const bool ok = load(c.file_arch, c.file_version, c.runtime_arch, c.runtime_version, err, rows);
        const bool good = ok == c.accept && (!ok || rows == 2);
        if (!good) ++failures;
        std::printf("%s %s/%d vs %s/%d rows=%zu  %s\n", good ? "ok  " : "FAIL", c.file_arch, c.file_version,
                    c.runtime_arch, c.runtime_version, rows, c.why);
        if (!ok) std::printf("      err: %s\n", err.c_str());
    }
    const int total = int(sizeof(cases) / sizeof(cases[0]));
    std::printf("tuning_header_arch_family_test: %d cases, %d failures\n", total, failures);
    return failures == 0 ? 0 : 1;
}
