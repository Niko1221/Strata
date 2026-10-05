// Real-artifact Q8_0 PLE rows against ggml's reference dequantizer.
// The PLE table in Unsloth's UD-Q6_K_XL GGUF ships as Q8_0 (170-B rows) rather than IQ4_NL; this
// mirrors ple_q5_parity.cpp so the Q8_0 read path is checked the same way, through BOTH I/O modes
// (Mmap and Direct) like ple_fp8_parity does.
#define NOMINMAX
#include "strata/artifact/gguf_reader.hpp"
#include "strata/kernels/ngram.hpp"
#include "ggml.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

int main(int argc, char** argv) {
    if (argc != 2) {
        std::fprintf(stderr, "usage: ple_q8_parity <gguf-containing-ple>\n");
        return 2;
    }
    strata::GgufFile gguf(argv[1]);
    const auto* tensor = gguf.find("per_layer_token_embd.weight");
    if (!tensor || tensor->type != 8 || tensor->shape.size() != 2 || tensor->shape[0] != 160) {
        std::fprintf(stderr, "expected Q8_0 PLE [160, N]\n");
        return 2;
    }
    const auto* traits = ggml_get_type_traits(GGML_TYPE_Q8_0);
    const uint8_t* bytes = gguf.tensor_data(*tensor);
    double max_abs_all = 0.0;
    size_t rows_checked = 0;
    for (const int direct : {0, 1}) {
        strata::kernels::PleIoOptions options;
        options.mode = direct ? strata::kernels::PleIo::Direct : strata::kernels::PleIo::Mmap;
        strata::kernels::PleTable table;
        std::string err;
        if (!table.open(argv[1], err, options)) {
            std::fprintf(stderr, "%s\n", err.c_str());
            return 1;
        }
        if (std::strcmp(table.format(), "Q8_0") != 0) {
            std::fprintf(stderr, "format() = %s, expected Q8_0\n", table.format());
            return 1;
        }
        rows_checked = (size_t) table.rows();
        // The real table's row 20000003 (the first head's vocab edge) only exists in a table that big; a
        // small synthetic fixture stands in for it so the same binary checks both.
        const uint32_t probes[] = {0, 1, 12345,
                                   table.rows() > 20000004 ? 20000003u : (uint32_t) (table.rows() / 2),
                                   (uint32_t) (table.rows() - 1)};
        double max_abs = 0.0;
        for (uint32_t row : probes) {
            float got[160], want[160];
            table.read_row(row, got);
            traits->to_float(bytes + (size_t) row * 170, want, 160);
            for (int i = 0; i < 160; ++i)
                max_abs = std::max(max_abs, (double) std::fabs(got[i] - want[i]));
        }
        uint32_t rows[16];
        for (int i = 0; i < 16; ++i) rows[i] = probes[i % 5];
        float batch[16 * 160];
        if (!table.issue(rows) || !table.collect(batch, err)) {
            std::fprintf(stderr, "collect: %s\n", err.c_str());
            return 1;
        }
        for (int h = 0; h < 16; ++h) {
            float want[160];
            traits->to_float(bytes + (size_t) rows[h] * 170, want, 160);
            for (int i = 0; i < 160; ++i)
                max_abs = std::max(max_abs, (double) std::fabs(batch[h * 160 + i] - want[i]));
        }
        if (direct) std::printf("Direct: ");
        else        std::printf("Mmap:   ");
        std::printf("%zu rows, max_abs %.3e\n", rows_checked, max_abs);
        max_abs_all = std::max(max_abs_all, max_abs);
    }
    std::printf("Q8_0 PLE: max_abs %.3e %s\n", max_abs_all, max_abs_all <= 1e-6 ? "PASS" : "FAIL");
    return max_abs_all <= 1e-6 ? 0 : 1;
}