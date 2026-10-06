// q5k_preunpack_parity: P0 #5 (exp 30) correctness gate.
// Proves the one-time Q5_K->Q5U pre-unpack plus the routed decode reproduce the packed Q5_K path:
//   oracle  = native_q5_k_mmvq on the packed Q5K blocks          (production kernel, trusted)
//   unpack  = native_mmvq_q5k_unpacked on the pre-unpacked Q5U    (the no-bit-unpack decode)
//   routed  = native_q5_k_mmvq again with the packed pointer registered & routing enabled
// The Q5U path must match the oracle within fp32 summation-order rounding; the min-offset (d*sc*code - mn*m)
// affine form is folded into Q5UBlock{dsc,mn1,qs[32]} so the decode is load + two dp4a + two scales.
//   q5k_preunpack_parity [n_in=512] [n_out=32] [ncols=4] [tol=1e-4]
#include "strata/kernels/native_mmvq.hpp"
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <vector>

int main(int argc, char** argv) {
    const int n_in = argc > 1 ? std::atoi(argv[1]) : 512;
    const int n_out = argc > 2 ? std::atoi(argv[2]) : 32;
    const int ncols = argc > 3 ? std::atoi(argv[3]) : 4;
    const double tol = argc > 4 ? std::atof(argv[4]) : 1e-4;
    if (n_in % 256) return 2;
    sycl::queue* q = &dpct::get_in_order_queue();

    const size_t wbytes = (size_t) n_out * (n_in / 256) * 176;   // Q5KBlock = 176 bytes
    std::vector<uint8_t> hw(wbytes);
    for (size_t i = 0; i < wbytes; ++i) hw[i] = (uint8_t) (i * 2654435761u >> 13);
    for (size_t b = 0; b + 175 < wbytes; b += 176) { hw[b] = 0x00; hw[b + 1] = 0x3c; hw[b + 2] = 0x00; hw[b + 3] = 0x3c; }   // dm = {1.0, 1.0}
    void* dw = sycl::malloc_device(wbytes, *q);
    q->memcpy(dw, hw.data(), wbytes).wait();

    const size_t ubytes = strata::kernels::native_mmvq_q5k_preunpack_bytes(n_in, n_out);
    void* du = sycl::malloc_device(ubytes, *q);
    strata::kernels::native_q5k_preunpack(dw, du, n_in, n_out, q);
    q->wait();

    std::vector<float> xf((size_t) ncols * n_in);
    for (size_t i = 0; i < xf.size(); ++i) xf[i] = (float) ((int) (i % 1000) - 500) / 250.f;
    void* xq = sycl::malloc_device((size_t) ncols * (n_in / 32) * 36, *q);
    strata::kernels::native_quantize_q8_1(xf.data(), xq, n_in, ncols, q);
    q->wait();

    float* yo = sycl::malloc_device<float>((size_t) ncols * n_out, *q);
    float* yu = sycl::malloc_device<float>((size_t) ncols * n_out, *q);
    float* yr = sycl::malloc_device<float>((size_t) ncols * n_out, *q);
    strata::kernels::native_q5_k_mmvq(dw, xq, yo, n_in, n_out, ncols, q);         // oracle (routing off by default)
    strata::kernels::native_mmvq_q5k_unpacked(du, xq, yu, n_in, n_out, ncols, q); // no-bit-unpack decode
    q->wait();
    strata::kernels::native_mmvq_register_q6k_preunpack(dw, du);                  // shared dense-K-quant registry
    strata::kernels::native_mmvq_set_q6k_preunpack(true);
    strata::kernels::native_q5_k_mmvq(dw, xq, yr, n_in, n_out, ncols, q);          // routed through the Q5U path
    q->wait();
    strata::kernels::native_mmvq_set_q6k_preunpack(false);
    strata::kernels::native_mmvq_clear_q6k_preunpack();

    std::vector<float> o((size_t) ncols * n_out), u((size_t) ncols * n_out), r((size_t) ncols * n_out);
    q->memcpy(o.data(), yo, o.size() * 4).wait();
    q->memcpy(u.data(), yu, u.size() * 4).wait();
    q->memcpy(r.data(), yr, r.size() * 4).wait();
    double den = 0, du_diff = 0, dr_diff = 0;
    for (size_t i = 0; i < o.size(); ++i) {
        den = std::max(den, (double) std::fabs(o[i]));
        du_diff = std::max(du_diff, std::fabs((double) o[i] - u[i]));
        dr_diff = std::max(dr_diff, std::fabs((double) o[i] - r[i]));
    }
    const double du_rel = du_diff / (den ? den : 1.0), dr_rel = dr_diff / (den ? den : 1.0);
    std::printf("q5k preunpack parity n_in=%d n_out=%d ncols=%d: max-rel oracle-vs-unpack %.3g, oracle-vs-routed %.3g (tol %.3g)\n",
                n_in, n_out, ncols, du_rel, dr_rel, tol);
    const bool ok = du_rel < tol && dr_rel < tol;
    std::printf(ok ? "PASS\n" : "FAIL\n");
    return ok ? 0 : 1;
}