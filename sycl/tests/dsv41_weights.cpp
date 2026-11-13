// Compare the port against ggml's CPU dequantizers on rows from a real pack.
#include "ds4_gguf.dp.hpp"
#include "ds4_iq.dp.hpp"
#include "ds4_kernels.dp.hpp"
#include <dpct/blas_utils.hpp>
#include <ggml.h>
#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <map>
#include <set>
#include <sstream>
#include <vector>

namespace iq = strata::ds4::gguf::iq;
namespace gf = strata::ds4::gguf;
namespace x3 = strata::dsv41k::exl3;

int main(int argc, char** argv) try {
    if (argc != 3) {
        std::fprintf(stderr, "usage: dsv41_weights_parity PACK MODEL_DIRECTORY\n");
        return 2;
    }
    const std::set<int> wanted{8, 13, 14, 16, 17, 19, 20, 22, 30, 39};
    std::map<std::string, int> types;
    for (int t : wanted) {
        std::string name = ggml_type_name(ggml_type(t));
        std::transform(name.begin(), name.end(), name.begin(), ::toupper);
        types[name] = t;
    }
    auto& queue = dpct::get_in_order_queue();
    std::map<int, std::filesystem::path> files;
    std::set<int> seen;
    size_t failures = 0;
    std::ifstream manifest(std::filesystem::path(argv[1]) / "tensors.txt");
    if (!manifest) throw std::runtime_error("cannot read tensors.txt");
    for (std::string line; std::getline(manifest, line);) {
        std::istringstream fields(line);
        std::string kind, name, type;
        fields >> kind;
        if (kind == "file") {
            int id; fields >> id >> name;
            files[id] = std::filesystem::path(argv[2]) / name;
            continue;
        }
        int file;
        uint64_t offset, bytes;
        int64_t height, width;
        if (kind != "t" || !(fields >> name >> type >> file >> offset >> bytes >> height >> width)) continue;
        if (!types.count(type) || seen.count(types[type]) || width < 256 || width % 256 || height < 17) continue;
        const int t = types[type], k = int(width), rows = 17;
        const size_t stride = ggml_row_size(ggml_type(t), k);
        std::vector<uint8_t> encoded(stride * rows);
        std::ifstream weights(files.at(file), std::ios::binary);
        weights.seekg(offset);
        if (!weights.read(reinterpret_cast<char*>(encoded.data()), encoded.size())) throw std::runtime_error("weight read failed");
        const auto* trait = ggml_get_type_traits(ggml_type(t));
        if (!trait->to_float) throw std::runtime_error("missing CPU dequantizer");
        std::vector<float> reference(size_t(rows) * k);
        for (int r = 0; r < rows; ++r) trait->to_float(encoded.data() + r * stride, reference.data() + r * k, k);
        auto* raw = sycl::malloc_device<uint8_t>(encoded.size(), queue);
        auto* decoded = sycl::malloc_shared<uint16_t>(size_t(rows) * k, queue);
        queue.memcpy(raw, encoded.data(), encoded.size()).wait_and_throw();
        size_t bad_decode = 0;
        if (t == 20) {
            // Engram stores compressed rows in hd-byte slots (padding after
            // the IQ4_NL blocks). Exercise the engine's actual lookup decoder.
            std::vector<uint8_t> padded(size_t(rows) * k);
            for (int r = 0; r < rows; ++r)
                std::copy_n(encoded.data() + r * stride, stride, padded.data() + r * k);
            auto* packed = sycl::malloc_device<uint8_t>(padded.size(), queue);
            auto* lookup = sycl::malloc_shared<float>(reference.size(), queue);
            queue.memcpy(packed, padded.data(), padded.size()).wait_and_throw();
            strata::ds4::engram_dequant(packed, nullptr, rows, k, lookup, &queue, 2);
            queue.wait_and_throw();
            for (size_t i = 0; i < reference.size(); ++i)
                if (lookup[i] != reference[i]) ++bad_decode;
            sycl::free(packed, queue); sycl::free(lookup, queue);
        } else if (iq::supported(t)) {
            if (!iq::dequant_rows(t, raw, stride, k, rows, decoded, false, &queue)) throw std::runtime_error("dequant rejected type");
            queue.wait_and_throw();
            for (size_t i = 0; i < reference.size(); ++i) {
                const auto expected = std::bit_cast<uint16_t>(sycl::half(reference[i]));
                if (decoded[i] != expected) {
                    if (bad_decode++ < 3) std::fprintf(stderr, "%s decode[%zu]: got %04x expected %04x (%a)\n", type.c_str(), i, decoded[i], expected, reference[i]);
                }
            }
        }
        auto* qx = sycl::malloc_shared<int8_t>(size_t(k) * 8, queue);
        auto* scales = sycl::malloc_shared<float>(size_t(k / 32) * 8, queue);
        auto* fx = sycl::malloc_shared<float>(size_t(k) * 8, queue);
        auto* output = sycl::malloc_shared<float>(size_t(rows) * 8, queue);
        for (int c = 0; c < 8; ++c) for (int j = 0; j < k; ++j) {
            qx[c * k + j] = int8_t((j * 13 + c * 31) % 255 - 127);
            scales[c * (k / 32) + j / 32] = std::ldexp(1.0f, -10 + ((j / 32 + c) % 4));
            fx[c * k + j] = qx[c * k + j] * scales[c * (k / 32) + j / 32];
        }
        size_t bad_mv = 0;
        double max_error = 0;
        for (int columns : {1, 4, 8}) {
            if (iq::supported(t)) {
                iq::MvJob job{};
                job.w = raw; job.ld = stride; job.type = t; job.k = k; job.n = rows;
                job.ncol = columns; job.xq = qx; job.xd = scales; job.y = output; job.ycs = rows;
                if (!iq::mv(&job, 1, &queue)) throw std::runtime_error("IQ mv rejected geometry");
            } else {
                x3::Mat matrix{};
                matrix.fmt = t == 39 ? gf::kFmtMXFP4 : t == 8 ? gf::kFmtQ8_0 : gf::kFmtBF16;
                matrix.raw = raw; matrix.ld = t == 30 ? k : stride; matrix.k = k; matrix.n = rows;
                x3::MvJob job{};
                job.m = &matrix; job.xf = fx; job.xf_cs = k; job.ncol = columns; job.y = output; job.y_cs = rows;
                if (!gf::mv(&job, 1, &queue)) throw std::runtime_error("GGUF mv rejected geometry");
            }
            queue.wait_and_throw();
            for (int c = 0; c < columns; ++c) for (int r = 0; r < rows; ++r) {
                double dot = 0, absolute_sum = 0;
                for (int j = 0; j < k; ++j) {
                    float activation = fx[c * k + j];
                    // The Q8_0 GEMV quantizes its F32 activations per 32.
                    // Reproduce that documented ABI with an independent CPU loop.
                    if (t == 8) {
                        float maximum = 0;
                        for (int z = j / 32 * 32; z < j / 32 * 32 + 32; ++z)
                            maximum = std::max(maximum, std::fabs(fx[c * k + z]));
                        const float scale = maximum / 127;
                        activation = maximum == 0 ? 0 : std::nearbyint(activation / scale) * scale;
                    }
                    const double product = double(reference[r * k + j]) * activation;
                    dot += product; absolute_sum += std::fabs(product);
                }
                const double error = std::fabs(output[c * rows + r] - dot);
                max_error = std::max(max_error, error);
                if (!std::isfinite(output[c * rows + r]) || error > 2e-5 * absolute_sum + 1e-6) {
                    if (bad_mv++ < 3) std::fprintf(stderr, "%s mv cols=%d row=%d col=%d: got %g expected %.10g error %.3g\n", type.c_str(), columns, r, c, output[c * rows + r], dot, error);
                }
            }
        }
        // The busy-expert path decodes whole matrices to FP16, then calls
        // oneMKL with FP16 operands and FP32 accumulation. Check that ABI
        // against ggml plus an independent double-precision CPU sum.
        size_t bad_f16 = 0, bad_gemm = 0;
        double max_gemm_error = 0;
        if (iq::f16_rows_ok(t)) {
            if (!iq::f16_rows(t, raw, k, rows, decoded, k, &queue))
                throw std::runtime_error("busy-expert decoder rejected type");
            queue.wait_and_throw();
            for (size_t i = 0; i < reference.size(); ++i)
                if (decoded[i] != std::bit_cast<uint16_t>(sycl::half(reference[i]))) ++bad_f16;
            constexpr int columns = 64;
            auto* hx = sycl::malloc_shared<uint16_t>(size_t(k) * columns, queue);
            auto* gy = sycl::malloc_shared<float>(size_t(rows) * columns, queue);
            for (int c = 0; c < columns; ++c) for (int j = 0; j < k; ++j)
                hx[size_t(c) * k + j] = std::bit_cast<uint16_t>(sycl::half(
                    std::ldexp(float((j * 13 + c * 31) % 255 - 127), -9 + (c % 3))));
            dpct::blas::descriptor blas;
            blas.set_queue(&queue);
            const float one = 1.0f, zero = 0.0f;
            dpct::blas::gemm(&blas, oneapi::mkl::transpose::trans,
                            oneapi::mkl::transpose::nontrans, rows, columns, k, &one,
                            decoded, dpct::library_data_t::real_half, k,
                            hx, dpct::library_data_t::real_half, k, &zero,
                            gy, dpct::library_data_t::real_float, rows, dpct::compute_type::f32);
            queue.wait_and_throw();
            for (int c = 0; c < columns; ++c) for (int r = 0; r < rows; ++r) {
                double dot = 0, absolute_sum = 0;
                for (int j = 0; j < k; ++j) {
                    const double product = double(float(sycl::half(reference[r * k + j]))) *
                        float(std::bit_cast<sycl::half>(hx[size_t(c) * k + j]));
                    dot += product; absolute_sum += std::fabs(product);
                }
                const double error = std::fabs(gy[size_t(c) * rows + r] - dot);
                max_gemm_error = std::max(max_gemm_error, error);
                if (!std::isfinite(gy[size_t(c) * rows + r]) || error > 2e-5 * absolute_sum + 1e-6)
                    ++bad_gemm;
            }
            sycl::free(hx, queue); sycl::free(gy, queue);
        }
        failures += bad_decode + bad_mv + bad_f16 + bad_gemm;
        std::printf("%s %s: %dx%d, decode mismatches %zu, mv mismatches %zu, max abs error %.6g; "
                    "busy FP16 %s: decode mismatches %zu, GEMM mismatches %zu, max abs error %.6g\n",
                    type.c_str(), name.c_str(), rows, k, bad_decode, bad_mv, max_error,
                    iq::f16_rows_ok(t) ? "tested" : "unsupported", bad_f16, bad_gemm, max_gemm_error);
        std::fflush(stdout);
        sycl::free(raw, queue); sycl::free(decoded, queue); sycl::free(qx, queue);
        sycl::free(scales, queue); sycl::free(fx, queue); sycl::free(output, queue);
        seen.insert(t);
    }
    std::printf("Real-weight parity: %zu types, %zu failures\n", seen.size(), failures);
    return failures || seen != wanted ? 1 : 0;
} catch (const std::exception& error) {
    std::fprintf(stderr, "%s\n", error.what());
    return 2;
}
