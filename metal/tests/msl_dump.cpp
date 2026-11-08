// metal/tests/msl_dump.cpp - compile every kernel library the engine compiles (engine/msl_libs.h), without starting an engine.
//
//   msl_dump [--kq] [--f16] [--engine-msl DIR]        STRATA_METAL_MSL_DUMP=<dir> msl_dump   writes the text of each library
//
// The engine builds its libraries from the same lists (msl_libs.h), so this is "the MSL text the engine compiles": with the dump
// directory set, mtl_rt.mm writes each library's full string (prelude, <metal_stdlib>, ggml-common, the kernel files) and an index;
// tools/msl_text_check.py compares two dumps.  It also proves that the libraries compile at the geometry this binary was built for
// (-DSTRATA_GEOMETRY / -DSTRATA_GEOMETRY_FILE): a compile error ends the run with the compiler's message.
//   --kq    the prelude of a model with K-quant / IQ down experts (kq_down_prelude)
//   --f16   STRATA_METAL_F16
#include "mtl_rt.h"
#include "engine/msl_libs.h"

#include <chrono>
#include <cstdio>
#include <cstring>
#include <string>

int main(int argc, char** argv) {
    bool kq = false, f16 = false;
    std::string edir = "../src/engine/msl";
    for (int i = 1; i < argc; ++i) {
        if (!std::strcmp(argv[i], "--kq")) kq = true;
        else if (!std::strcmp(argv[i], "--f16")) f16 = true;
        else if (!std::strcmp(argv[i], "--engine-msl") && i + 1 < argc) edir = argv[++i];
    }
    using namespace strata_metal;
    const std::string kqp = kq ? "#define STRATA_KQ_DOWN 1\n" : "";
    const std::pair<const char*, MslLib> libs[] = {
        {"core", lib_core(f16, kqp)},          {"gdn", lib_gdn()},
        {"qsa", lib_qsa()},                    {"sample", lib_sample()},
        {"ple", lib_ple(edir)},                {"prompt", lib_prompt(kqp)},
        {"prompt_gdn", lib_prompt_gdn()},      {"prompt_qsa", lib_prompt_qsa()},
        {"mtp", lib_mtp(edir)},                {"mtp_chain", lib_mtp_chain(edir)},
        {"dflash2", lib_dflash2(edir)},        {"coop_probe", lib_coop_probe()},
    };
    mtl::Context cx;
    int bad = 0;
    for (const auto& [name, L] : libs) {
        const auto t0 = std::chrono::steady_clock::now();
        std::string why;
        const int id = cx.compile_try(L.files, L.prelude, false, &why);
        const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        if (id < 0) {
            std::printf("%-11s FAIL (%.0f ms): %s\n", name, ms, why.c_str());
            ++bad;
        } else {
            std::printf("%-11s ok (%.0f ms)\n", name, ms);
        }
    }
    std::printf("%s\n", bad ? "msl_dump: FAILED" : "msl_dump: all libraries compile");
    return bad ? 1 : 0;
}
