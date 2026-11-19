// CUDA standalone of the sampler statistics harness. Build: cmake --build <dir> --target ds4_sampler_stat (optional, off by default; -DSTRATA_BUILD_DSV41=ON),
// or by hand: nvcc -include ds4_launch.h -I include -I src/dsv41 sampler_stat_main.cu.  Usage: ds4_sampler_stat ROWS.bin N_VOCAB N_SAMPLES OUTPREFIX (src/dsv41/sampler_stat.inc).
#include "sampler_stat.inc"
#include "vendor/sampler.cu"
int main(int argc, char** argv) { return sampler_stat::run(argc, argv); }
