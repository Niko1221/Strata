// cann/tests/gguf_parity.cpp - read the real model file with the port's own code.
//
// Opens the 22.85 GB Qwen3.6-35B-A3B GGUF, prints what the header says and what the file is
// made of, then runs two checks on real bytes:
//   - a Q4_K and a Q6_K tensor decoded by this port against ggml's `to_float` (bit-exact),
//   - y = W x for one expert of blk.0.ffn_gate_exps on the NPU against the CPU.
#include <acl/acl.h>

#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include "strata/cann/device.hpp"
#include "strata/cann/gguf.hpp"
#include "strata/cann/ops.hpp"
#include "strata/cann/quants.hpp"

#ifdef STRATA_CANN_HAVE_GGML
#include <ggml.h>
#endif

using namespace strata::cann;

namespace {

const char * kDefaultPath = "/data/strata-qwen36/Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf";

double checksum(const std::vector<float> & v) {
    double s = 0.0;
    for (size_t i = 0; i < v.size(); i += 97) { // sparse sum: enough to catch a layout error
        s += v[i] * (double) (i % 13 + 1);
    }
    return s;
}

void decoder_check(const Gguf & g, const GgufTensor & t, uint64_t blocks) {
    const int block_bytes = quant_type_size((int) t.type);
    const int block_elems = quant_block_elements((int) t.type);
    if (block_bytes == 0 || block_elems == 0) {
        return;
    }
    const uint64_t bytes = blocks * (uint64_t) block_bytes;
    const int64_t n = (int64_t) blocks * block_elems;
    const std::vector<uint8_t> raw = g.tensor_bytes(t, 0, bytes);
    std::vector<float> ours((size_t) n);
    dequantize((int) t.type, raw.data(), ours.data(), n);

    const char * type_name = "?";
    switch (t.type) {
        case 8: type_name = "Q8_0"; break;
        case 12: type_name = "Q4_K"; break;
        case 13: type_name = "Q5_K"; break;
        case 14: type_name = "Q6_K"; break;
        default: break;
    }
    printf("%-34s %-5s dims=[", t.name.c_str(), type_name);
    for (size_t i = 0; i < t.dims.size(); ++i) {
        printf("%s%llu", i ? "," : "", (unsigned long long) t.dims[i]);
    }
    printf("] blocks=%llu  sum=%.6f", (unsigned long long) blocks, checksum(ours));

#ifdef STRATA_CANN_HAVE_GGML
    std::vector<float> ref((size_t) n);
    const ggml_type_traits * traits = ggml_get_type_traits((enum ggml_type) t.type);
    if (traits && traits->to_float) {
        traits->to_float(raw.data(), ref.data(), n);
        const bool same = std::memcmp(ours.data(), ref.data(), ours.size() * sizeof(float)) == 0;
        printf("  vs ggml: %s\n", same ? "BIT-EXACT" : "DIFFERS");
        if (!same) {
            throw std::runtime_error("decoder differs from ggml on " + t.name);
        }
        return;
    }
    printf("  (ggml has no to_float for this type)\n");
#else
    printf("  (built without the ggml oracle)\n");
#endif
}

void weight_mix(const Gguf & g) {
    struct Row {
        uint64_t count = 0;
        uint64_t bytes = 0;
        const char * name = nullptr;
    };
    std::vector<Row> rows(32);
    for (const auto & t : g.tensors()) {
        if (t.type >= 32) {
            continue;
        }
        Row & row = rows[t.type];
        row.count += 1;
#ifdef STRATA_CANN_HAVE_GGML
        // the oracle knows every ggml type, so the mix covers the whole file
        const size_t ts = ggml_type_size((enum ggml_type) t.type);
        const int64_t bs = ggml_blck_size((enum ggml_type) t.type);
        if (ts && bs > 0) {
            row.bytes += t.elements() / (uint64_t) bs * ts;
        }
        row.name = ggml_type_name((enum ggml_type) t.type);
#else
        const int bs = quant_type_size((int) t.type);
        if (bs) {
            row.bytes += t.elements() / QK_K * (uint64_t) bs;
        }
#endif
    }
    printf("weight mix :\n");
    uint64_t total = 0;
    for (const Row & r : rows) {
        if (r.count && r.bytes) {
            printf("  %-5s %4llu tensors  %8.2f GB\n", r.name ? r.name : "?",
                   (unsigned long long) r.count, (double) r.bytes / 1e9);
            total += r.bytes;
        }
    }
    printf("  %-5s               %8.2f GB\n\n", "sum", (double) total / 1e9);
}

void real_gemv(const Gguf & g, const char * name) {
    const GgufTensor * w = g.find(name);
    if (!w || w->dims.size() != 3 || w->type != 12) {
        printf("real GEMV : %s not a Q4_K expert tensor, skipped\n", name);
        return;
    }
    const uint64_t n_rows = w->dims[1]; // 512 rows in the expert slab
    const uint64_t n_cols = w->dims[0]; // 2048 values per row
    const uint64_t blocks = n_rows * n_cols / QK_K;
    const std::vector<uint8_t> raw = g.tensor_bytes(*w, 0, blocks * 144);
    std::vector<float> wf((size_t) n_rows * n_cols);
    dequantize(12, raw.data(), wf.data(), (int64_t) (n_rows * n_cols));

    std::mt19937 rng(3);
    std::uniform_real_distribution<float> dist(-0.5f, 0.5f);
    std::vector<float> x((size_t) n_cols);
    for (auto & v : x) {
        v = dist(rng);
    }

    std::vector<uint16_t> hw(wf.size()), hx(x.size());
    f32_to_f16(wf.data(), hw.data(), hw.size());
    f32_to_f16(x.data(), hx.data(), hx.size());
    std::vector<float> w16(wf.size()), x16(x.size());
    f16_to_f32(hw.data(), w16.data(), w16.size());
    f16_to_f32(hx.data(), x16.data(), x16.size());

    const std::vector<float> ref = matmul_ref(w16, x16, (int64_t) n_rows, (int64_t) n_cols, 1);

    Device device(0);
    Stream stream(device);
    Buffer bw(w16.size() * 2), bx(x16.size() * 2), by((size_t) n_rows * 2);
    bw.copy_from_host(hw.data(), hw.size() * 2);
    bx.copy_from_host(hx.data(), hx.size() * 2);
    Tensor tw{ bw.data(), DType::F16, (int64_t) n_cols, (int64_t) n_rows };
    Tensor tx{ bx.data(), DType::F16, 1, (int64_t) n_cols };
    Tensor ty{ by.data(), DType::F16, 1, (int64_t) n_rows };
    matmul(stream, tw, tx, ty);

    std::vector<uint16_t> hy((size_t) n_rows);
    by.copy_to_host(hy.data(), hy.size() * 2);
    stream.synchronize();
    std::vector<float> got((size_t) n_rows);
    f16_to_f32(hy.data(), got.data(), got.size());

    double mx = 0.0;
    for (size_t i = 0; i < got.size(); ++i) {
        mx = std::max(mx, std::fabs((double) got[i] - ref[i]));
    }
    printf("real GEMV  %s [%llu x %llu] on the NPU: max|diff| = %.6f  %s\n\n", name,
           (unsigned long long) n_rows, (unsigned long long) n_cols, mx, mx <= 0.5 ? "PASS" : "FAIL");
    if (mx > 0.5) {
        throw std::runtime_error("real-weight GEMV differs from CPU");
    }
}

} // namespace

int main(int argc, char ** argv) {
    setvbuf(stdout, nullptr, _IONBF, 0);
    const std::string path = argc > 1 ? argv[1] : kDefaultPath;
    try {
        const Gguf g(path);
        printf("file       : %s\n", path.c_str());
        printf("gguf       : v%u, %zu tensors, data at %llu\n", g.version(), g.tensors().size(),
               (unsigned long long) g.data_offset());
        printf("arch       : %s\n", g.arch().c_str());
        printf("layers     : %lld\n", (long long) g.metadata_int(g.arch() + ".block_count"));
        printf("experts    : %lld (top %lld)\n", (long long) g.metadata_int(g.arch() + ".expert_count"),
               (long long) g.metadata_int(g.arch() + ".expert_used_count"));
        printf("embedding  : %lld\n\n", (long long) g.metadata_int(g.arch() + ".embedding_length"));

        weight_mix(g);

        const GgufTensor * q4 = nullptr;
        const GgufTensor * q5 = nullptr;
        const GgufTensor * q6 = nullptr;
        const GgufTensor * q8 = nullptr;
        for (const auto & t : g.tensors()) {
            if (!q4 && t.type == 12 && t.elements() >= 4096) {
                q4 = &t;
            }
            if (!q5 && t.type == 13 && t.elements() >= 4096) {
                q5 = &t;
            }
            if (!q6 && t.type == 14 && t.elements() >= 4096) {
                q6 = &t;
            }
            if (!q8 && t.type == 8 && t.elements() >= 4096) {
                q8 = &t;
            }
        }
        if (!q4 || !q6) {
            printf("no Q4_K / Q6_K tensor with enough elements found\n");
            return 2;
        }

        real_gemv(g, "blk.0.ffn_gate_exps.weight");
        decoder_check(g, *q4, 64);
        if (q5) {
            decoder_check(g, *q5, 64);
        }
        decoder_check(g, *q6, 64);
        if (q8) {
            decoder_check(g, *q8, 256);
        }
        printf("\nmodel file read and decoded by the port\n");
        shutdown();
        return 0;
    } catch (const std::exception & e) {
        printf("FAILED: %s\n", e.what());
        return 1;
    }
}
