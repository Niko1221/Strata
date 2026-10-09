// src/kernels/ple_exl3_parity.cpp - PleTable::open_exl3 against the numpy n-gram codec (tools/exl3).
//
// Reads the synthetic `ngram_embedding.safetensors` from emit_ngram_table.py and checks every row against
// the expected decoded rows.  CPU only, no model.
#include "strata/kernels/ngram.hpp"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <string>
#include <vector>

int main(int argc, char** argv) {
    if (argc < 2) { std::fprintf(stderr, "usage: %s <fixture_dir>\n", argv[0]); return 2; }
    std::string dir = argv[1];
    strata::kernels::PleTable table;
    strata::kernels::PleIoOptions io;
    io.mode = (argc > 2 && std::string(argv[2]) == "direct") ? strata::kernels::PleIo::Direct
                                                            : strata::kernels::PleIo::Mmap;
    io.io_thread = false;                        // a tiny fixture; no worker thread needed
    std::string err;
    if (!table.open_exl3(dir + "/ngram_embedding.safetensors", err, io)) {
        std::fprintf(stderr, "open_exl3 failed: %s\n", err.c_str());
        return 1;
    }
    const int rows = (int) table.rows();
    const int dim = strata::kernels::PLE_HEAD_DIM;
    std::printf("format=%s mode=%s rows=%d\n", table.format(),
                io.mode == strata::kernels::PleIo::Direct ? "direct" : "mmap", rows);

    std::ifstream ef(dir + "/expected.f32", std::ios::binary);
    std::vector<float> exp((size_t) rows * dim);
    ef.read(reinterpret_cast<char*>(exp.data()), (std::streamsize) (exp.size() * 4));
    if (!ef) { std::fprintf(stderr, "cannot read expected.f32\n"); return 2; }

    double num = 0, den = 0, maxd = 0;
    std::vector<float> got(dim);
    for (int r = 0; r < rows; ++r) {
        table.read_row((uint32_t) r, got.data());
        for (int j = 0; j < dim; ++j) {
            double d = (double) got[j] - exp[(size_t) r * dim + j];
            num += d * d;
            den += (double) exp[(size_t) r * dim + j] * exp[(size_t) r * dim + j];
            maxd = std::max(maxd, std::fabs(d));
        }
    }
    double rel = std::sqrt(num) / (std::sqrt(den) + 1e-30);
    std::printf("ngram rows rel_err: %.3e max_abs: %.3e (limit 1e-4)\n", rel, maxd);
    std::printf("%s\n", rel < 1e-4 ? "OK" : "FAIL");
    return rel < 1e-4 ? 0 : 1;
}
