// compile probe: ggml's quantized dot products used standalone
#include "vecdotq.cuh"
__global__ void probe(const void* w, const block_q8_1* x, float* y) {
    y[0] = vec_dot_q4_K_q8_1(w, x, 0, 0) + vec_dot_q5_K_q8_1(w, x, 0, 0) + vec_dot_q8_0_q8_1(w, x, 0, 0) +
           vec_dot_q5_1_q8_1(w, x, 0, 0);
}
int main() { return 0; }
