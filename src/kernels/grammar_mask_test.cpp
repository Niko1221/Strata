#include "strata/core/grammar_mask.hpp"
#include "strata/kernels/sampler.hpp"
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <vector>

void check(bool ok) { if (!ok) { std::fprintf(stderr, "grammar mask check failed\n"); std::exit(1); } }
int main() {
    std::string err;
    std::vector<uint32_t> mask;
    check(strata::core::decode_token_mask("MASK 02000000", 3, mask, err) && mask[0] == 2);
    check(!strata::core::decode_token_mask("MASK 00000000", 3, mask, err));
    check(!strata::core::decode_token_mask("MASK 00000080", 3, mask, err));
    check(!strata::core::decode_token_mask("MASK zz000000", 3, mask, err));
    check(!strata::core::decode_token_mask("MASK 02", 3, mask, err));
    float* logits = nullptr;
    int* pick = nullptr;
    uint32_t* bits = nullptr;
    check(cudaMalloc(&logits, 3 * sizeof(float)) == cudaSuccess);
    check(cudaMalloc(&pick, sizeof(int)) == cudaSuccess);
    check(cudaMalloc(&bits, sizeof(uint32_t)) == cudaSuccess);
    uint32_t allow = 2; // token 1 is far less likely than the two forbidden tokens
    check(cudaMemcpy(bits, &allow, sizeof(allow), cudaMemcpyHostToDevice) == cudaSuccess);
    for (bool greedy : {true, false}) {
        const float input[] = {1000.0f, -1000.0f, 999.0f};
        check(cudaMemcpy(logits, input, sizeof(input), cudaMemcpyHostToDevice) == cudaSuccess);
        strata::kernels::apply_token_mask(logits, 3, bits, nullptr);
        strata::kernels::SamplerParams sp;
        sp.greedy = greedy; sp.top_k = 3; sp.top_p = 0.9f; sp.min_p = 0.1f;
        strata::kernels::sample_tokens(logits, 1, 3, nullptr, 0, sp, pick, nullptr);
        int result = -1;
        check(cudaMemcpy(&result, pick, sizeof(int), cudaMemcpyDeviceToHost) == cudaSuccess && result == 1);
    }
    cudaFree(bits); cudaFree(pick); cudaFree(logits);
    std::puts("grammar_mask_test OK: forbidden high-probability tokens cannot be sampled");
}
