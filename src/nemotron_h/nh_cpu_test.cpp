// src/nemotron_h/nh_cpu_test.cpp - the CPU experts (nh_cpu.cpp) against a dequantized double-precision reference,
// on a real Nemotron-H GGUF's expert tensors (no GPU): for every MoE layer of the file (or --layers a,b,...), a few
// experts and tokens, out = down(relu(up x)^2) from CpuExperts::run vs the same from the weights dequantized
// (ggml's to_float) and an f64 product.  The activations are rounded to the dot products' formats on the CPU path,
// so the reference error is that rounding: printed per layer, the test fails above --tol (relative, per output row).
//
//   nh_cpu_test --model <first shard> [--layers 1,3] [--experts 4] [--tol 0.02]
#include "strata/artifact/gguf_reader.hpp"
#include "strata/nemotron_h/cpu_experts.hpp"

#include "ggml.h"
#include "ggml-cpu.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <string>
#include <vector>

int main(int argc, char** argv) {
    std::string model, layers_s;
    int n_exp = 4;
    double tol = 0.02;
    for (int i = 1; i + 1 < argc; i += 2) {
        const std::string a = argv[i], v = argv[i + 1];
        if (a == "--model") model = v;
        else if (a == "--layers") layers_s = v;
        else if (a == "--experts") n_exp = std::atoi(v.c_str());
        else if (a == "--tol") tol = std::atof(v.c_str());
    }
    if (model.empty()) { std::fprintf(stderr, "usage: nh_cpu_test --model GGUF [--layers a,b] [--experts 4] [--tol 0.02]\n"); return 2; }
    const strata::GgufModel gm = strata::GgufModel::open(model);
    std::vector<int> layers;
    if (!layers_s.empty()) {
        for (const char* p = layers_s.c_str(); *p;) {
            layers.push_back(std::atoi(p));
            while (*p && *p != ',') ++p;
            if (*p) ++p;
        }
    } else {
        for (int l = 0; l < 89; ++l)
            if (gm.find("blk." + std::to_string(l) + ".ffn_up_exps.weight")) layers.push_back(l);
    }
    strata::nemotron_h::CpuExperts pool(4);
    std::mt19937 rng(7);
    std::normal_distribution<float> nd(0.f, 1.f);
    double worst = 0;
    for (int l : layers) {
        size_t su = 0, sd = 0;
        const strata::TensorInfo* tu = gm.find("blk." + std::to_string(l) + ".ffn_up_exps.weight", &su);
        const strata::TensorInfo* td = gm.find("blk." + std::to_string(l) + ".ffn_down_exps.weight", &sd);
        if (!tu || !td) continue;
        const int LAT = (int) tu->shape[0], FF = (int) tu->shape[1], E = (int) tu->shape[2];
        strata::nemotron_h::CpuFmt f;
        std::string err;
        if (!strata::nemotron_h::cpu_fmt((int) tu->type, (int) td->type, LAT, FF, f, err)) {
            std::fprintf(stderr, "layer %d: %s\n", l, err.c_str());
            return 1;
        }
        const uint8_t* up = gm.shard(su).tensor_data(*tu);
        const uint8_t* dn = gm.shard(sd).tensor_data(*td);
        const size_t up_b = (size_t) FF * f.up_row, d_b = (size_t) LAT * f.d_row;
        const int T = 3;
        std::vector<float> x((size_t) T * LAT);
        for (auto& v : x) v = nd(rng) * 0.5f;
        std::vector<strata::nemotron_h::CpuJob> jobs;
        std::vector<float> out((size_t) n_exp * T * LAT);
        std::vector<int> ex;
        for (int k = 0; k < n_exp; ++k) ex.push_back((int) (rng() % (unsigned) E));
        for (int k = 0; k < n_exp; ++k)
            for (int t = 0; t < T; ++t) {
                strata::nemotron_h::CpuJob j;
                j.up = up + (size_t) ex[(size_t) k] * up_b;
                j.down = dn + (size_t) ex[(size_t) k] * d_b;
                j.tok = t;
                j.out = out.data() + ((size_t) k * T + t) * LAT;
                jobs.push_back(j);
            }
        pool.run(f, x.data(), T, jobs.data(), (int) jobs.size());
        // the reference: dequantized weights, f64 products
        const auto* tr_u = ggml_get_type_traits((ggml_type) tu->type);
        const auto* tr_d = ggml_get_type_traits((ggml_type) td->type);
        std::vector<float> Wu((size_t) FF * LAT), Wd((size_t) LAT * FF);
        double lmax = 0, lmean = 0;
        int nrow = 0;
        for (int k = 0; k < n_exp; ++k) {
            tr_u->to_float(up + (size_t) ex[(size_t) k] * up_b, Wu.data(), (int64_t) FF * LAT);
            tr_d->to_float(dn + (size_t) ex[(size_t) k] * d_b, Wd.data(), (int64_t) LAT * FF);
            for (int t = 0; t < T; ++t) {
                std::vector<double> hh((size_t) FF);
                for (int r = 0; r < FF; ++r) {
                    double s = 0;
                    for (int i = 0; i < LAT; ++i) s += (double) Wu[(size_t) r * LAT + i] * x[(size_t) t * LAT + i];
                    const double u = s > 0 ? s : 0;
                    hh[(size_t) r] = u * u;
                }
                double num = 0, den = 0;
                const float* o = out.data() + ((size_t) k * T + t) * LAT;
                for (int r = 0; r < LAT; ++r) {
                    double s = 0;
                    for (int i = 0; i < FF; ++i) s += (double) Wd[(size_t) r * FF + i] * hh[(size_t) i];
                    num += (o[r] - s) * (o[r] - s);
                    den += s * s;
                }
                const double rel = std::sqrt(num / std::max(den, 1e-30));
                lmax = std::max(lmax, rel);
                lmean += rel;
                ++nrow;
            }
        }
        std::printf("layer %2d: up %s, down %s: rel. error mean %.5f max %.5f over %d rows\n", l, ggml_type_name((ggml_type) tu->type),
                    ggml_type_name((ggml_type) td->type), lmean / nrow, lmax, nrow);
        worst = std::max(worst, lmax);
    }
    std::printf("%s: worst %.5f (tolerance %.3f)\n", worst <= tol ? "PASS" : "FAIL", worst, tol);
    return worst <= tol ? 0 : 1;
}
