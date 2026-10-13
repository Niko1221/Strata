extern "C" __global__ void k(const float* a, const float* b, const float* c, float* out, int n) {
    float acc = 0.0f;
    for (int i = 0; i < n; ++i) {
        const float d = a[i] * b[i];      // dw * dx
        acc += d * c[i];                  // += sumi   <-- the question
    }
    out[0] = acc;
}
