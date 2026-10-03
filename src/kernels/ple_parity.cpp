// src/kernels/ple_parity.cpp - P2.S4's test: the n-gram hash, the IQ4_NL table read, and the PLE block.
//
// THREE PARTS, THREE DIFFERENT ORACLES; A and B use none of this project's own code, C a reference written
// independently of the kernel, from the formulas in `ple.hpp`:
//
//   A. THE HASH against `ref/ngram.py::ngram_rows`, via `ple_oracle_vectors.inc` (generated).  The properties
//      are asserted OBSERVABLE first - XOR vs sum, `%` vs `&`, the cut DIRECTION, the NULL sentinel - because
//      each rival reading produces a perfectly valid index in range.
//   B. THE TABLE against numpy reading the ORIGINAL GGUF at the offset the validated `gguf_reader` reports.
//   C. THE BLOCK against a host reference computed here, in double precision, from the formulas in `ple.hpp`
//      and on the artifact's real weights (`ple_key`, `ple_value`, the three norms and the conv kernel, read from
//      the pack at the offsets its own `index.txt` gives) and real gathered n-gram rows.  Only `hidden` and the
//      conv history are synthetic (a fixed-seed generator), because they are the residual stream and the state,
//      which no file holds.  The reference records EVERY intermediate (key, value, gate, gated, normalized,
//      conv_out, result), so a mismatch can be attributed to a stage instead of guessed at.
#include "strata/kernels/ple.hpp"
#include "strata/kernels/ngram.hpp"
#include "strata/kernels/f16_bits.hpp"
#include "strata/kernels/native_mmvq.hpp"
#include "ple_oracle_vectors.inc"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <random>
#include <sstream>
#include <string>
#include <stdexcept>
#include <vector>

namespace k = strata::kernels;
namespace o = strata::kernels::ple_oracle;

namespace {

void ck(cudaError_t e, const char* what) {
    if (e != cudaSuccess) {
        std::fprintf(stderr, "%s: %s\n", what, cudaGetErrorString(e));
        std::exit(1);
    }
}

/// TOLERANCE CHECKS MUST BE NaN-SAFE, and `x <= t` is not.
///
/// `x <= t` AND `x > t` are BOTH false for NaN, so a check written either way can silently pass - round 197
/// found `shared_expert_parity` printing a perfect score for an entirely-NaN fixture because the accumulator
/// was `if (rel > worst) worst = rel;` and NaN is never greater.  Every comparison in this file goes through
/// `le`/`gt`, which require the value to be FINITE first, and every compared pair is counted for non-finite
/// entries and the count printed.  A NaN fixture is then a loud failure rather than a green line.
bool le(double x, double t) { return std::isfinite(x) && x <= t; }
bool gt(double x, double t) { return std::isfinite(x) && x > t; }

/// How many entries of `a` are not finite.  Printed next to every comparison.
long long nonfinite(const float* a, size_t n) {
    long long k = 0;
    for (size_t i = 0; i < n; ++i)
        if (!std::isfinite(a[i])) ++k;
    return k;
}

/// Normalised L1 with the reference magnitude returned.  For the STAGE comparisons a plain |ref| denominator
/// is right: these are dense vectors of O(1) values with no cancellation between them.
///
/// Returns NaN if EITHER side has a non-finite entry, so a caller that forgets `le`/`gt` still cannot read a
/// finite-looking number out of a broken fixture.
template <class Ref, class Got>
double rel_l1(const Ref* a, const Got* b, size_t n, double* mag_out = nullptr,
              long long* nf_out = nullptr) {
    double d = 0, m = 0;
    long long nf = 0;
    for (size_t i = 0; i < n; ++i) {
        if (!std::isfinite(a[i]) || !std::isfinite(b[i])) { ++nf; continue; }
        d += std::fabs((double) a[i] - (double) b[i]);
        m += std::fabs((double) a[i]);
    }
    if (mag_out) *mag_out = m / (double) (n ? n : 1);
    if (nf_out) *nf_out = nf;
    if (nf) return std::nan("");
    return d / (m > 1e-30 ? m : 1e-30);
}

/// The file's size, WITHOUT reading it.
///
/// The first version of this test called `read_file(gguf)` - which reads the WHOLE file into a vector - three
/// times, on a 28.8 GB GGUF.  That is 27 GB of RSS, three full passes over the file, and a 132-second ctest
/// entry in a suite where every other test is under a second.  A test that slow stops being run, which is how
/// a gate stops being a gate.  Everything below now reads only the bytes it actually compares.
long long file_size(const char* path) {
    std::FILE* f = std::fopen(path, "rb");
    if (!f) return -1;
#if defined(_MSC_VER)
    _fseeki64(f, 0, SEEK_END);
    const long long n = _ftelli64(f);
#else
    std::fseek(f, 0, SEEK_END);
    const long long n = std::ftell(f);
#endif
    std::fclose(f);
    return n;
}

/// `n` bytes at `off`, or an empty vector.
std::vector<uint8_t> read_at(const char* path, long long off, size_t n) {
    std::vector<uint8_t> v(n);
    std::FILE* f = std::fopen(path, "rb");
    if (!f) return {};
#if defined(_MSC_VER)
    if (_fseeki64(f, off, SEEK_SET) != 0) { std::fclose(f); return {}; }
#else
    if (std::fseek(f, (long) off, SEEK_SET) != 0) { std::fclose(f); return {}; }
#endif
    const size_t got = std::fread(v.data(), 1, n, f);
    std::fclose(f);
    if (got != n) return {};
    return v;
}

/// Dequantize one IQ4_NL row with an INTERLEAVED nibble order, to show the correct split-half order is
/// observable.  Deliberately shares nothing with the kernel's decoder.
void deq_interleaved(const uint8_t* row, float* out160) {
    for (int b = 0; b < k::PLE_HEAD_DIM / 32; ++b) {
        const uint8_t* blk = row + (size_t) b * 18;
        uint16_t db;
        std::memcpy(&db, blk, 2);
        const float d = k::f32_from_f16(db);
        for (int j = 0; j < 16; ++j) {
            out160[b * 32 + 2 * j] = d * (float) k::iq4nl_code(blk[2 + j] & 0x0F);
            out160[b * 32 + 2 * j + 1] = d * (float) k::iq4nl_code(blk[2 + j] >> 4);
        }
    }
}

/// Three tokens, so the conv history slides twice and the native-key loop (which runs two) has two inputs.
constexpr int kBlockTokens = 3;

/// Per-stage bound on the normalized L1 distance between the kernel and the reference, which applies the same
/// activation contract.  What is left is f32 against double arithmetic and summation order: the GEMVs add 2560
/// products in f32 lanes, whose error with mixed-sign terms is about sqrt(2560) * 6e-8 = 3e-6 (the all-same-sign
/// worst case, n * eps, is 1.5e-4), and the norms and the gate round through f32 once.  The bound sits above
/// that typical error and far below both the activation contract (see the visibility check, which requires the
/// contract to be at least ten times the bound) and any structural error - a transposed conv kernel, a
/// channel-slow history, a head-fastest gather, a wrong tap order - which is O(1).
constexpr double kStageTolerance = 1e-5;

/// The six PLE tensors as the pack stores them, plus the views the kernel and the reference each want.
struct PleHostWeights {
    std::vector<uint8_t> key_codes, raw_scales;   // ple_key: 2-bit codes, then the fp16 group scales as stored
    std::vector<float> key_scales;                // the same scales widened to f32, which every S-form kernel takes
    std::vector<uint16_t> value_bf16;             // ple_value BF16 bits
    std::vector<uint16_t> conv1d_f16;             // ple_conv1d F16 bits, ggml order k + kern*c
    std::vector<float> norm_key, norm_query, norm_conv;
};

/// Offset and size in `dense.bin` of a tensor's source bytes, from the pack's own index: the rows of
/// `<pack>/index.txt` are `name file kind src_off src_bytes ...`, and file 0 is `dense.bin`.
bool pack_tensor_span(const std::string& pack, const std::string& name, uint64_t& off, uint64_t& bytes) {
    std::ifstream index(pack + "/index.txt");
    std::string line;
    while (std::getline(index, line)) {
        if (line.empty() || line[0] == '#') continue;
        std::istringstream row(line);
        std::string n;
        int file = -1, kind = -1;
        uint64_t src_off = 0, src_bytes = 0;
        if (!(row >> n >> file >> kind >> src_off >> src_bytes) || n != name || file != 0) continue;
        off = src_off;
        bytes = src_bytes;
        return true;
    }
    return false;
}

/// Reads the PLE tensors of layer 1 from `<pack>/dense.bin` at the spans the pack's index gives.  The spans of
/// `ple_key` and `ple_value` are also required to equal the generated offsets, so those two cannot drift apart
/// silently.  Returns false, with `err` set, when an entry is missing or its size is not the expected one.
bool load_ple_weights(const std::string& pack, PleHostWeights& w, std::string& err) {
    static_assert(o::kKeyScalesOffset == o::kKeyCodesOffset + o::kKeyCodesBytes,
                  "the fp16 key scales follow the key codes in the pack");
    const uint64_t hcd = (uint64_t) k::NG_HC_DIM;
    auto tensor = [&](const char* name, uint64_t want_bytes, uint64_t want_off) -> std::vector<uint8_t> {
        uint64_t off = 0, bytes = 0;
        if (!pack_tensor_span(pack, name, off, bytes) || bytes != want_bytes || (want_off && off != want_off)) {
            err = std::string("the pack index has no ") + name + " span of " + std::to_string(want_bytes) +
                  " bytes at the expected offset";
            return {};
        }
        std::vector<uint8_t> v = read_at((pack + "/dense.bin").c_str(), (long long) off, (size_t) bytes);
        if (v.empty()) err = std::string("cannot read ") + name + " from " + pack + "/dense.bin";
        return v;
    };
    const std::vector<uint8_t> key = tensor("blk.1.ple_key.weight", o::kKeyCodesBytes + o::kKeyScalesBytes,
                                            o::kKeyCodesOffset);
    const std::vector<uint8_t> value = tensor("blk.1.ple_value.weight", o::kValueBytes, o::kValueOffset);
    const std::vector<uint8_t> conv = tensor("blk.1.ple_conv1d.weight", (uint64_t) k::PLE_CONV_KERNEL * hcd * 2, 0);
    const std::vector<uint8_t> nk = tensor("blk.1.ple_norm_key.weight", hcd * 4, 0);
    const std::vector<uint8_t> nq = tensor("blk.1.ple_norm_query.weight", hcd * 4, 0);
    const std::vector<uint8_t> nc = tensor("blk.1.ple_norm_conv.weight", hcd * 4, 0);
    if (!err.empty()) return false;

    w.key_codes.assign(key.begin(), key.begin() + (long long) o::kKeyCodesBytes);
    w.raw_scales.assign(key.begin() + (long long) o::kKeyCodesBytes, key.end());
    // fp16 -> f32 for the scales, which is what every S-form kernel in this project expects
    w.key_scales.resize(w.raw_scales.size() / 2);
    for (size_t i = 0; i < w.key_scales.size(); ++i) {
        uint16_t h;
        std::memcpy(&h, w.raw_scales.data() + i * 2, 2);
        w.key_scales[i] = k::f32_from_f16(h);
    }
    // ple_value is BF16 promoted to 32 bits in the pack; take the high half, which is exact
    w.value_bf16.resize(value.size() / 4);
    for (size_t i = 0; i < w.value_bf16.size(); ++i) {
        uint32_t u;
        std::memcpy(&u, value.data() + i * 4, 4);
        w.value_bf16[i] = (uint16_t) (u >> 16);
    }
    w.conv1d_f16.resize(conv.size() / 2);
    std::memcpy(w.conv1d_f16.data(), conv.data(), conv.size());
    for (auto* norm : {&nk, &nq, &nc}) {
        std::vector<float>& dst = norm == &nk ? w.norm_key : norm == &nq ? w.norm_query : w.norm_conv;
        dst.resize(norm->size() / 4);
        std::memcpy(dst.data(), norm->data(), norm->size());
    }
    return true;
}

/// Fixed-seed uniform values in [-amp, amp), built from the raw generator bits so the fixture is the same on
/// every standard library (`std::uniform_real_distribution` is not).
void fill_uniform(float* v, size_t n, float amp, std::mt19937& rng) {
    for (size_t i = 0; i < n; ++i) v[i] = amp * ((float) (rng() >> 8) * (2.0f / 16777216.0f) - 1.0f);
}

/// What one token of the block produces, in double, stage by stage.
struct BlockRef {
    std::vector<double> key, value, gate, gated, normalized, conv, result;
};

/// The Q8_0 activation the `ple_key` GEMV multiplies by, as `quantize_q8_0` builds it: per 32-element block,
/// `d = amax / 127` in f32 and stored as fp16, then `q = rint(x / d)` in double clamped to int8.
std::vector<double> q8_0_activation(const float* x, size_t n) {
    std::vector<double> out(n, 0.0);
    for (size_t b = 0; b < n; b += 32) {
        float amax = 0.0f;
        for (size_t i = 0; i < 32; ++i) amax = std::fmax(amax, std::fabs(x[b + i]));
        if (amax == 0.0f) continue;
        const float d32 = amax / 127.0f;
        const double dx = (double) k::f32_from_f16(k::f16_from_f32(d32));
        for (size_t i = 0; i < 32; ++i)
            out[b + i] = std::fmin(127.0, std::fmax(-128.0, std::rint((double) x[b + i] / (double) d32))) * dx;
    }
    return out;
}

/// `f` rounded to BF16 (round to nearest even, on the bits), which is the activation contract of a BF16 weight.
double bf16_round(float f) {
    uint32_t i;
    std::memcpy(&i, &f, 4);
    i = (i + ((i >> 16) & 1u) + 0x7FFFu) & 0xFFFF0000u;
    std::memcpy(&f, &i, 4);
    return (double) f;
}

/// `rms_norm` per stream of `n_embd`, then the `[hc_dim]` gamma: `x * gamma / sqrt(mean(x^2) + eps)`.
std::vector<double> grouped_norm(const std::vector<double>& x, const float* gamma) {
    std::vector<double> y(x.size());
    for (size_t c = 0; c < (size_t) k::NG_HC; ++c) {
        const size_t base = c * (size_t) k::NG_N_EMBD;
        double sum = 0.0;
        for (size_t d = 0; d < (size_t) k::NG_N_EMBD; ++d) sum += x[base + d] * x[base + d];
        const double scale = 1.0 / std::sqrt(sum / (double) k::NG_N_EMBD + (double) k::NG_RMS_EPS);
        for (size_t d = 0; d < (size_t) k::NG_N_EMBD; ++d) y[base + d] = x[base + d] * scale * (double) gamma[base + d];
    }
    return y;
}

/// The PLE block for ONE token, on the host, in double precision, from the formulas at the top of `ple.hpp`.
///
/// `contract` selects the activation the two projections multiply by.  True is what the kernel does and what
/// ggml does for these weight types: the embedding is rounded to Q8_0 for the Q2_0 `ple_key` and to BF16 for the
/// BF16 `ple_value`.  False multiplies the exact embedding, which is what lets the test show the contract is
/// visible to it.  Everything after the two projections is exact either way.  `hist` is row-fastest,
/// `hist[row + NG_HIST * channel]`, and oldest row first.
BlockRef ple_block_reference(const float* emb, const float* hidden, const float* hist, const PleHostWeights& w,
                             bool contract) {
    const size_t nd = (size_t) k::NG_N_EMBD, hcd = (size_t) k::NG_HC_DIM, hc = (size_t) k::NG_HC;
    std::vector<double> act_key(emb, emb + nd), act_value(emb, emb + nd);
    if (contract) {
        act_key = q8_0_activation(emb, nd);
        for (size_t i = 0; i < nd; ++i) act_value[i] = bf16_round(emb[i]);
    }
    BlockRef r;
    // key = grouped_norm(ple_key @ emb): weight (row, i) is code (row*nd + i) of the row-major [hc_dim, n_embd]
    // tensor, four 2-bit codes per byte, code bias -1, one scale per 64 weights
    std::vector<double> projected(hcd);
    for (size_t row = 0; row < hcd; ++row) {
        double acc = 0.0;
        for (size_t i = 0; i < nd; ++i) {
            const size_t flat = row * nd + i;
            const int code = (w.key_codes[flat >> 2] >> ((flat & 3) * 2)) & 3;
            acc += (double) (code - 1) * (double) w.key_scales[flat >> 6] * act_key[i];
        }
        projected[row] = acc;
    }
    r.key = grouped_norm(projected, w.norm_key.data());
    const std::vector<double> query = grouped_norm(std::vector<double>(hidden, hidden + hcd), w.norm_query.data());
    // value = ple_value @ emb, BF16 weights
    r.value.resize(nd);
    for (size_t row = 0; row < nd; ++row) {
        double acc = 0.0;
        for (size_t i = 0; i < nd; ++i) {
            const uint32_t bits = (uint32_t) w.value_bf16[row * nd + i] << 16;
            float weight;
            std::memcpy(&weight, &bits, 4);
            acc += (double) weight * act_value[i];
        }
        r.value[row] = acc;
    }
    // gate[c] = sigmoid(sign(s) * sqrt(max(|s|, 1e-6))), s = sum_d key*query / sqrt(n_embd)
    r.gate.resize(hc);
    for (size_t c = 0; c < hc; ++c) {
        double dot = 0.0;
        for (size_t d = 0; d < nd; ++d) dot += r.key[c * nd + d] * query[c * nd + d];
        const double s = dot / std::sqrt((double) nd);
        const double signed_root = (s > 0 ? 1.0 : s < 0 ? -1.0 : 0.0) * std::sqrt(std::fmax(std::fabs(s), 1e-6));
        r.gate[c] = 1.0 / (1.0 + std::exp(-signed_root));
    }
    r.gated.resize(hcd);
    for (size_t i = 0; i < hcd; ++i) r.gated[i] = r.value[i % nd] * r.gate[i / nd];
    r.normalized = grouped_norm(r.gated, w.norm_conv.data());
    // conv[c] = silu(sum_k kW[k + kern*c] * padded[row_k][c]); tap k reads row NG_HIST - (kern-1-k)*dil of
    // [history | this token's normalized row], so tap 0 is the oldest and the last tap is the new row.
    r.conv.resize(hcd);
    r.result.resize(hcd);
    for (size_t c = 0; c < hcd; ++c) {
        double acc = 0.0;
        for (int kk = 0; kk < k::PLE_CONV_KERNEL; ++kk) {
            const int row = k::NG_HIST - (k::PLE_CONV_KERNEL - 1 - kk) * k::NGRAM_SIZE;
            const double v = row == k::NG_HIST ? r.normalized[c] : (double) hist[(size_t) row + (size_t) k::NG_HIST * c];
            acc += (double) k::f32_from_f16(w.conv1d_f16[(size_t) kk + (size_t) k::PLE_CONV_KERNEL * c]) * v;
        }
        r.conv[c] = acc / (1.0 + std::exp(-acc));
        r.result[c] = (double) hidden[c] + r.gated[c] + r.conv[c];
    }
    return r;
}

int history_advance_regression() {
    const size_t channels = (size_t) k::NG_HC_DIM, rows = (size_t) k::NG_HIST, guard = 16;
    const size_t count = channels * rows;
    std::vector<float> expected(count), normalized(channels), actual(count);
    for (size_t c = 0; c < channels; ++c)
        for (size_t r = 0; r < rows; ++r) expected[c * rows + r] = -float(c * 16 + r + 1);
    float *history_storage = nullptr, *norm_storage = nullptr;
    ck(cudaMalloc(&history_storage, (count + 2 * guard) * sizeof(float)), "history regression allocation");
    ck(cudaMalloc(&norm_storage, (channels + 2 * guard) * sizeof(float)), "history norm allocation");
    ck(cudaMemset(history_storage, 0xa5, (count + 2 * guard) * sizeof(float)), "history guard init");
    ck(cudaMemset(norm_storage, 0xa5, (channels + 2 * guard) * sizeof(float)), "history norm guard init");
    float* history = history_storage + guard;
    float* norm = norm_storage + guard;
    ck(cudaMemcpy(history, expected.data(), count * sizeof(float), cudaMemcpyHostToDevice), "history initial values");
    // the guard memsets and this upload run on the legacy stream, which the non-blocking `stream` does not wait for
    ck(cudaDeviceSynchronize(), "history setup");
    cudaStream_t stream;
    cudaGraph_t graph;
    cudaGraphExec_t executable;
    ck(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "history regression stream");
    ck(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal), "history regression capture");
    k::ple_history_advance(history, norm, stream);
    ck(cudaStreamEndCapture(stream, &graph), "history regression capture end");
    ck(cudaGraphInstantiate(&executable, graph, nullptr, nullptr, 0), "history regression instantiate");
    int bad = 0;
    for (int token = 0; token < 12; ++token) {
        for (size_t c = 0; c < channels; ++c) {
            normalized[c] = float((token + 1) * 1000000 + c); // distinct exactly represented integers
            for (size_t r = 0; r + 1 < rows; ++r) expected[c * rows + r] = expected[c * rows + r + 1];
            expected[c * rows + rows - 1] = normalized[c];
        }
        ck(cudaMemcpyAsync(norm, normalized.data(), channels * sizeof(float), cudaMemcpyHostToDevice, stream), "history new normalized row");
        ck(cudaGraphLaunch(executable, stream), "history captured advance");
        ck(cudaStreamSynchronize(stream), "history captured advance sync");
        ck(cudaMemcpy(actual.data(), history, count * sizeof(float), cudaMemcpyDeviceToHost), "history readback");
        if (std::memcmp(actual.data(), expected.data(), count * sizeof(float)) != 0) {
            std::printf("  history advance mismatch after token %d\n", token);
            ++bad;
        }
    }
    bool overlap_refused = false, null_refused = false;
    try { k::ple_history_advance(history, history + 1, stream); }
    catch (const std::invalid_argument&) { overlap_refused = true; }
    try { k::ple_history_advance(history, nullptr, stream); }
    catch (const std::invalid_argument&) { null_refused = true; }
    if (!overlap_refused || !null_refused) ++bad;
    auto guard_ok = [&](float* storage, size_t payload) {
        uint8_t before[64], after[64];
        ck(cudaMemcpy(before, storage, sizeof(before), cudaMemcpyDeviceToHost), "history prefix guard");
        ck(cudaMemcpy(after, storage + guard + payload, sizeof(after), cudaMemcpyDeviceToHost), "history suffix guard");
        return std::all_of(before, before + 64, [](uint8_t b) { return b == 0xa5; }) &&
               std::all_of(after, after + 64, [](uint8_t b) { return b == 0xa5; });
    };
    if (!guard_ok(history_storage, count) || !guard_ok(norm_storage, channels)) ++bad;
    std::printf("  PLE history 12 captured steps, row-fastest state and guards: %s\n", bad == 0 ? "pass" : "FAIL");
    ck(cudaGraphExecDestroy(executable), "history graph exec destroy");
    ck(cudaGraphDestroy(graph), "history graph destroy");
    ck(cudaStreamDestroy(stream), "history stream destroy");
    cudaFree(history_storage); cudaFree(norm_storage);
    return bad;
}

}  // namespace

int main(int argc, char** argv) {
    bool selftest = false;
    std::string pack = "pack/full";
    // --gguf, else $STRATA_PLE_GGUF, else the development layout (run from the engine root)
    std::string gguf = std::getenv("STRATA_PLE_GGUF") ? std::getenv("STRATA_PLE_GGUF")
                                                      : "../../Q2_0/Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00002-of-00002.gguf";
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--selftest") selftest = true;
        else if (a == "--pack" && i + 1 < argc) pack = argv[++i];
        else if (a == "--gguf" && i + 1 < argc) gguf = argv[++i];
        else { std::fprintf(stderr, "usage: ple_parity [--selftest] [--pack D] [--gguf F]\n");
               return 2; }
    }
    // The required inputs (the PLE table and the pack's PLE tensors) are checked before the first CUDA call, so
    // missing files cannot become an apparent pass or a misleading GPU error.
    k::PleTable table;
    std::string err;
    PleHostWeights weights;
    bool preflight_done = false;
    if (selftest) {
        if (!table.open(gguf, err) || table.rows() != o::kTableRows ||
            file_size(gguf.c_str()) != (long long) o::kTableDataStart + (long long) o::kTableRows * k::PLE_ROW_BYTES) {
            std::fprintf(stderr, "ple_parity: required PLE table is missing or incompatible: %s (%s)\n", gguf.c_str(), err.c_str());
            return 2;
        }
        if (!load_ple_weights(pack, weights, err)) {
            std::fprintf(stderr, "ple_parity: required PLE weights are missing or incompatible: %s\n", err.c_str());
            return 2;
        }
        preflight_done = true;
        if (!k::ple_block_available()) {
            std::fprintf(stderr, "ple_parity: --selftest requires a CUDA device for block checks\n");
            return 2;
        }
    }
    int bad = history_advance_regression();
    const k::PleConsts C = k::ple_artifact_consts();

    // ============================ A. THE HASH ============================
    std::printf("A. the n-gram hash, against ref/ngram.py with the artifact's own constants\n");

    // ---- the constants must be internally consistent BEFORE they are used: `head_offsets` is its own array
    //      in the metadata, so deriving it from `vocab_sizes` would hide a disagreement between them.
    {
        uint64_t run = 0;
        bool ok = true;
        for (int h = 0; h < k::PLE_N_HEADS; ++h) {
            if (C.offset[h] != run) ok = false;
            run += C.vocab[h];
        }
        const uint64_t total = run;
        const bool fits = total <= k::PLE_TABLE_ROWS;
        std::printf("  %-46s %s (sum %llu <= %llu rows; %llu spare)\n",
                    "head_offsets == the running sum of vocab_sizes", ok && fits ? "yes" : "*** NO ***",
                    (unsigned long long) total, (unsigned long long) k::PLE_TABLE_ROWS,
                    (unsigned long long) (k::PLE_TABLE_ROWS - total));
        if (!ok || !fits) ++bad;
    }

    for (int ci = 0; ci < o::kHashCaseCount; ++ci) {
        const o::HashCase& hc = o::kHashCases[ci];
        std::vector<uint32_t> got((size_t) hc.n_tokens * k::PLE_N_HEADS);
        k::ngram_rows(hc.tokens, hc.prev, hc.n_tokens, C, got.data());
        long long diff = 0;
        for (size_t i = 0; i < got.size(); ++i)
            if (got[i] != hc.rows[i]) ++diff;
        std::printf("  %-46s %s (%lld of %zu differ)\n", hc.name, diff ? "*** WRONG ***" : "matches", diff,
                    got.size());
        if (diff) {
            for (int t = 0; t < hc.n_tokens && diff; ++t) {
                for (int h = 0; h < k::PLE_N_HEADS; ++h) {
                    const uint32_t g = got[(size_t) t * 16 + h], w = hc.rows[(size_t) t * 16 + h];
                    if (g != w) {
                        std::printf("      first: t=%d h=%d got %u want %u\n", t, h, g, w);
                        t = hc.n_tokens;
                        break;
                    }
                }
            }
            ++bad;
        }
    }

    // ---- the rival readings, each computed and required to DIFFER from the oracle -------------------
    //
    // Every one of these produces a 16-element vector of valid indices.  That is the whole reason the hash is
    // transcribed property by property: no range check, no shape check and no summing test can see any of
    // them, only an oracle can - and only a DELIBERATELY-WRONG fixture can show that the oracle comparison
    // has the power to catch them.
    {
        const o::HashCase& hc = o::kHashCases[0];   // the window-direction case, 4 tokens
        const int T = hc.n_tokens, np = k::NGRAM_SIZE - 1, n_heads = k::PLE_N_HEADS;
        std::vector<uint32_t> xor_v((size_t) T * n_heads), sum_v((size_t) T * n_heads),
            mask_v((size_t) T * n_heads), back_v((size_t) T * n_heads), zero_v((size_t) T * n_heads);
        for (int i = 0; i < T; ++i) {
            int64_t ctx[k::NGRAM_SIZE];
            ctx[0] = hc.tokens[i];
            for (int s = 1; s < k::NGRAM_SIZE; ++s) ctx[s] = hc.prev[i * np + (np - s)];
            for (int n = 2; n <= k::NGRAM_SIZE; ++n) {
                // (1) SUM instead of XOR
                uint64_t sum = 0;
                for (int j = 0; j < n; ++j) sum += (uint64_t) ctx[j] * C.mult[j];
                // (2) mask instead of modulo: the vocab sizes are NOT powers of two
                uint64_t xr = k::ngram_mixed(ctx, C.mult, n);
                // (4) token 0 treated as "missing" the way -1 is
                int64_t ctx0[k::NGRAM_SIZE];
                for (int j = 0; j < k::NGRAM_SIZE; ++j)
                    ctx0[j] = (ctx[j] == k::TOKEN_NULL || ctx[j] == 0) ? k::PLE_EOS_TOKEN_ID : ctx[j];
                uint64_t xr0 = k::ngram_mixed(ctx0, C.mult, n);
                // (3) the window read NEWEST-first instead of oldest-first: prev index (s-1)
                int64_t ctxb[k::NGRAM_SIZE];
                ctxb[0] = hc.tokens[i];
                for (int s = 1; s < k::NGRAM_SIZE; ++s) ctxb[s] = hc.prev[i * np + (s - 1)];
                uint64_t xrb = k::ngram_mixed(ctxb, C.mult, n);
                const int base = (n - 2) * k::HEADS_PER_NGRAM;
                for (int g = 0; g < k::HEADS_PER_NGRAM; ++g) {
                    const int h = base + g;
                    xor_v[(size_t) i * n_heads + h] = (uint32_t) (xr % C.vocab[h] + C.offset[h]);
                    sum_v[(size_t) i * n_heads + h] = (uint32_t) (sum % C.vocab[h] + C.offset[h]);
                    mask_v[(size_t) i * n_heads + h] =
                        (uint32_t) ((xr & (C.vocab[h] - 1)) + C.offset[h]);
                    back_v[(size_t) i * n_heads + h] = (uint32_t) (xrb % C.vocab[h] + C.offset[h]);
                    zero_v[(size_t) i * n_heads + h] = (uint32_t) (xr0 % C.vocab[h] + C.offset[h]);
                }
            }
        }
        struct Rival { const char* name; const std::vector<uint32_t>* v; };
        const Rival rivals[] = {{"XOR vs SUM", &sum_v},
                                {"% vs & (vocab sizes are not powers of two)", &mask_v},
                                {"window read newest-first, not oldest-first", &back_v},
                                {"token id 0 treated as the NULL sentinel", &zero_v}};
        for (const Rival& r : rivals) {
            // A rival is observable if it disagrees with the ORACLE, which is what the test above compares.
            long long d = 0;
            for (size_t i = 0; i < r.v->size(); ++i)
                if ((*r.v)[i] != hc.rows[i]) ++d;
            const bool visible = d > 0;
            std::printf("  %-46s %s (%lld of %zu rows differ from the oracle)\n", r.name,
                        visible ? "yes" : "*** NO - the fixture cannot see this ***", d, r.v->size());
            if (!visible) ++bad;
        }
        (void) xor_v;
    }

    // ---- the EOS cut DIRECTION, which needs a case where the two directions disagree ----------------
    {
        const o::HashCase& hc = o::kHashCases[1];   // [[N,N],[7,EOS],[EOS,7]]
        const int T = hc.n_tokens, np = k::NGRAM_SIZE - 1, n_heads = k::PLE_N_HEADS;
        std::vector<uint32_t> fwd((size_t) T * n_heads), bwd((size_t) T * n_heads);
        for (int i = 0; i < T; ++i) {
            int64_t f[k::NGRAM_SIZE], b[k::NGRAM_SIZE];
            f[0] = b[0] = hc.tokens[i];
            bool cf = false, cb = false;
            for (int s = 1; s < k::NGRAM_SIZE; ++s) {
                const int32_t fv = hc.prev[i * np + (np - s)];
                cf = cf || fv < 0 || fv == k::PLE_EOS_TOKEN_ID;
                f[s] = cf ? k::PLE_EOS_TOKEN_ID : fv;
                // the BACKWARD reading: only the position that IS EOS becomes EOS
                const int32_t bv = hc.prev[i * np + (np - s)];
                b[s] = (bv < 0 || bv == k::PLE_EOS_TOKEN_ID) ? k::PLE_EOS_TOKEN_ID : bv;
                (void) cb;
            }
            for (int n = 2; n <= k::NGRAM_SIZE; ++n) {
                const uint64_t mf = k::ngram_mixed(f, C.mult, n), mb = k::ngram_mixed(b, C.mult, n);
                const int base = (n - 2) * k::HEADS_PER_NGRAM;
                for (int g = 0; g < k::HEADS_PER_NGRAM; ++g) {
                    const int h = base + g;
                    fwd[(size_t) i * n_heads + h] = (uint32_t) (mf % C.vocab[h] + C.offset[h]);
                    bwd[(size_t) i * n_heads + h] = (uint32_t) (mb % C.vocab[h] + C.offset[h]);
                }
            }
        }
        long long d = 0;
        for (size_t i = 0; i < fwd.size(); ++i)
            if (fwd[i] != bwd[i]) ++d;
        std::printf("  %-46s %s (%lld of %zu rows differ)\n",
                    "cut propagates FORWARD vs only at the EOS position",
                    d > 0 ? "yes" : "*** NO ***", d, fwd.size());
        if (!d) ++bad;
    }

    // ============================ B. THE TABLE ============================
    std::printf("\nB. the IQ4_NL table, against numpy reading the original GGUF\n");
    if (!preflight_done && !table.open(gguf, err)) {
        std::printf("  cannot open the PLE table: %s\n", err.c_str());
        std::printf("\nple_parity: %d failures, TABLE AND BLOCK SKIPPED; partial diagnostic only\n", bad);
        return selftest || bad ? 1 : 0;
    }
    std::printf("  %-46s %llu (expected %llu)\n", "rows", (unsigned long long) table.rows(),
                (unsigned long long) o::kTableRows);
    if (table.rows() != o::kTableRows) ++bad;

    // The size identity is what makes the data offset FALSIFIABLE: the tensor must exactly fill the file from
    // data_start.  Deriving the row count from the file size instead gives 320001538 rows and data_start 12,
    // which is self-consistent and wrong.
    {
        const long long sz = file_size(gguf.c_str());
        const long long need = (long long) o::kTableDataStart + (long long) table.rows() * k::PLE_ROW_BYTES;
        const bool ok = sz == need;
        std::printf("  %-46s %s (%lld vs %lld; data_start %llu)\n",
                    "data_start + rows*90 == file size", ok ? "yes" : "*** NO ***", sz, need,
                    (unsigned long long) o::kTableDataStart);
        if (!ok) ++bad;
    }

    // Read the raw bytes through the mapped table by comparing against numpy's decode of the SAME offsets.
    // 90 BYTES AT A TIME, not the whole file - see `file_size`'s note.
    {
        int probe_bad = 0;
        for (int p = 0; p < o::kProbeCount; ++p) {
            const uint32_t row = o::kProbeRows[p];
            std::vector<float> got(k::PLE_HEAD_DIM);
            table.read_row(row, got.data());
            const std::vector<uint8_t> raw =
                read_at(gguf.c_str(), (long long) o::kTableDataStart + (long long) row * k::PLE_ROW_BYTES,
                        (size_t) k::PLE_ROW_BYTES);
            if (raw.size() != (size_t) k::PLE_ROW_BYTES) { std::printf("  short read at row %u\n", row); ++probe_bad; continue; }
            std::vector<float> ref(k::PLE_HEAD_DIM);
            k::iq4nl_dequant_row(raw.data(), ref.data());
            // the sampled values from the generator (numpy), which is the oracle proper.
            // NOT `fmax`: it DISCARDS NaN, so a NaN element would leave these at 0 and the row would report
            // a perfect match.  A plain comparison propagates the NaN into `ok`.
            double head = 0, tail = 0, sum = 0;
            for (int i = 0; i < 8; ++i) {
                const double dh = std::fabs((double) got[i] - (double) o::kProbeHead[p][i]);
                const double dt = std::fabs((double) got[k::PLE_HEAD_DIM - 8 + i] -
                                            (double) o::kProbeTail[p][i]);
                if (p == 0 && i == 0) { head = dh; tail = dt; }
                else { head = (dh > head) ? dh : head; tail = (dt > tail) ? dt : tail; }
            }
            for (int i = 0; i < k::PLE_HEAD_DIM; ++i) sum += std::fabs((double) got[i]);
            const long long nfg = nonfinite(got.data(), k::PLE_HEAD_DIM);
            const bool ok = le(head, 0.0) && le(tail, 0.0) && le(std::fabs(sum - o::kProbeSum[p]), 1e-9) &&
                            nfg == 0;
            std::printf("  %-46s %s (row %u: head %.3e tail %.3e sum d %.3e, non-finite %lld)\n",
                        p == 0 ? "row matches numpy's decode of the same bytes" : "  (same, another row)",
                        ok ? "yes" : "*** NO ***", row, head, tail, std::fabs(sum - o::kProbeSum[p]), nfg);
            if (!ok) ++probe_bad;
            // and the two decoders must agree with each other bit for bit
            for (int i = 0; i < k::PLE_HEAD_DIM; ++i)
                if (got[i] != ref[i]) { std::printf("      byte-level decode differs at %d\n", i); ++probe_bad; break; }
        }
        bad += probe_bad;
    }

    // ---- the split-half nibble order must be observable ---------------------------------------------
    {
        const std::vector<uint8_t> raw =
            read_at(gguf.c_str(), (long long) o::kTableDataStart +
                                      (long long) o::kProbeRows[0] * k::PLE_ROW_BYTES,
                    (size_t) k::PLE_ROW_BYTES);
        std::vector<float> want(k::PLE_HEAD_DIM), inter(k::PLE_HEAD_DIM);
        k::iq4nl_dequant_row(raw.data(), want.data());
        deq_interleaved(raw.data(), inter.data());
        const double rel = rel_l1(want.data(), inter.data(), k::PLE_HEAD_DIM);
        const bool visible = gt(rel, 0.05);
        std::printf("  %-46s %s (%.2f%% apart; generator says %.4f)\n",
                    "split-half vs INTERLEAVED nibbles observable",
                    visible ? "yes" : "*** NO ***", rel * 100, (double) o::kInterleavedSeparation);
        if (!visible) ++bad;
    }

    // ---- and reading at file offset 0 instead of data_start must be observable ----------------------
    {
        const std::vector<uint8_t> at_ds = read_at(gguf.c_str(), (long long) o::kTableDataStart,
                                                   (size_t) k::PLE_ROW_BYTES);
        const std::vector<uint8_t> at_0 = read_at(gguf.c_str(), 0, (size_t) k::PLE_ROW_BYTES);
        std::vector<float> ok_v(k::PLE_HEAD_DIM), at0(k::PLE_HEAD_DIM);
        k::iq4nl_dequant_row(at_ds.data(), ok_v.data());
        k::iq4nl_dequant_row(at_0.data(), at0.data());
        const double rel = rel_l1(ok_v.data(), at0.data(), k::PLE_HEAD_DIM);
        const bool visible = gt(rel, 0.05);
        std::printf("  %-46s %s (%.2f%% apart)\n",
                    "data_start 192 vs file offset 0 observable", visible ? "yes" : "*** NO ***", rel * 100);
        if (!visible) ++bad;
    }

    // ---- head-slowest flatten ----------------------------------------------------------------------
    {
        const uint32_t rows16[16] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15};
        std::vector<float> emb(k::NG_N_EMBD), one(k::PLE_HEAD_DIM);
        table.gather(rows16, emb.data());
        bool blockwise = true;
        for (int h = 0; h < k::PLE_N_HEADS && blockwise; ++h) {
            table.read_row((uint32_t) h, one.data());
            for (int d = 0; d < k::PLE_HEAD_DIM; ++d)
                if (emb[h * k::PLE_HEAD_DIM + d] != one[d]) { blockwise = false; break; }
        }
        // the rival: head-fastest, element (d,h) at d*16+h - a real transpose, not a reshape
        std::vector<float> fast(k::NG_N_EMBD);
        for (int d = 0; d < k::PLE_HEAD_DIM; ++d)
            for (int h = 0; h < k::PLE_N_HEADS; ++h) fast[d * k::PLE_N_HEADS + h] = emb[h * k::PLE_HEAD_DIM + d];
        long long diff = 0;
        for (int i = 0; i < k::NG_N_EMBD; ++i)
            if (fast[i] != emb[i]) ++diff;
        std::printf("  %-46s %s (head h occupies [h*160,(h+1)*160))\n", "gather is head-slowest",
                    blockwise ? "yes" : "*** NO ***");
        std::printf("  %-46s %s (%lld of %d elements differ)\n", "head-fastest would differ",
                    diff > 0 ? "yes" : "*** NO ***", diff, k::NG_N_EMBD);
        if (!blockwise || !diff) ++bad;
    }

    // ============================ C. THE BLOCK ============================
    std::printf("\nC. the PLE block, against a host reference in double precision\n");
    if (!preflight_done && !load_ple_weights(pack, weights, err)) {
        std::printf("  cannot load the PLE weights: %s\n", err.c_str());
        std::printf("\nple_parity: %d failures, BLOCK SKIPPED; partial diagnostic only\n", bad);
        return selftest || bad ? 1 : 0;
    }
    std::printf("  weights: ple_key, ple_value, the three norms and the conv kernel, from %s/dense.bin\n",
                pack.c_str());

    if (!k::ple_block_available()) {
        std::printf("  no CUDA device; the block SKIPPED, not passed.\n");
        std::printf("\nple_parity: %d failures, BLOCK SKIPPED\n", bad);
        return selftest || bad ? 1 : 0;
    }

    // The kernel takes the two quantized weights as CODES AND SCALES and the rest in the pack's own forms, which
    // is exactly what `load_ple_weights` returns.
    const std::vector<uint8_t>& key_codes = weights.key_codes;
    const std::vector<uint8_t>& raw_scales = weights.raw_scales;
    const std::vector<float>& key_scales = weights.key_scales;
    const std::vector<uint16_t>& value_bf16 = weights.value_bf16;
    const std::vector<uint16_t>& conv1d_f16 = weights.conv1d_f16;
    const size_t n_scales = key_scales.size();

    // ---- the inputs: real gathered n-gram rows (the "real vocabulary ids" hash case, whose rows section A has
    //      just verified) for `emb`; a fixed-seed generator for the residual stream and the conv history, which
    //      no file holds.  Amplitudes are O(1) for the stream and O(0.7) mean |.| for the history, the size of a
    //      grouped-normalized row.
    const size_t hcd = (size_t) k::NG_HC_DIM, nd = (size_t) k::NG_N_EMBD;
    const o::HashCase& rows_case = o::kHashCases[5];
    if (rows_case.n_tokens < kBlockTokens) {
        std::printf("  *** the hash case has %d tokens, the block check needs %d ***\n", rows_case.n_tokens, kBlockTokens);
        return 1;
    }
    std::vector<float> emb((size_t) kBlockTokens * nd), hidden((size_t) kBlockTokens * hcd),
        hist0((size_t) k::NG_HIST * hcd);
    for (int t = 0; t < kBlockTokens; ++t)
        table.gather(rows_case.rows + (size_t) t * k::PLE_N_HEADS, emb.data() + (size_t) t * nd);
    std::mt19937 rng(20261001);
    fill_uniform(hidden.data(), hidden.size(), 1.0f, rng);
    fill_uniform(hist0.data(), hist0.size(), 1.5f, rng);

    // ---- the reference, token by token.
    //
    // THE CONV HISTORY ADVANCES BETWEEN TOKENS.  Token t's window is the history `[hist(9) | normalized(0..t)]`
    // read at rows `t+0, t+3, t+6, t+9`, so the state slides by one NORMALIZED row per token.  Feeding every
    // token the original history would make the kernel and any reference written with the same slip agree with
    // each other, so the state each token sees is built once, here, and both sides read it.  It is the
    // reference's own normalized row, rounded to f32, that enters the state: a kernel error in one token then
    // cannot leak into the next token's comparison.
    //
    // The state is row-fastest, `hist[row + NG_HIST * channel]`, because it is ggml's
    // `reshape_3d(state, d_conv - 1, conv_channels, n_seqs)` with the row as ne0.
    std::vector<std::vector<float>> hist_at;
    std::vector<BlockRef> ref;
    {
        std::vector<float> state = hist0;
        for (int t = 0; t < kBlockTokens; ++t) {
            hist_at.push_back(state);
            ref.push_back(ple_block_reference(emb.data() + (size_t) t * nd, hidden.data() + (size_t) t * hcd,
                                              state.data(), weights, true));
            for (size_t c = 0; c < hcd; ++c) {
                for (size_t r = 0; r + 1 < (size_t) k::NG_HIST; ++r)
                    state[r + (size_t) k::NG_HIST * c] = state[(r + 1) + (size_t) k::NG_HIST * c];
                state[((size_t) k::NG_HIST - 1) + (size_t) k::NG_HIST * c] = (float) ref.back().normalized[c];
            }
        }
    }

    // ---- the activation contract must be visible to this test: the exact-embedding reference and the
    //      contract reference differ by at least ten times the stage tolerance, on both projections.  Without
    //      this line a tolerance that swallowed the contract, or a kernel that dropped it, would look the same
    //      as one that honoured it.
    {
        const BlockRef exact = ple_block_reference(emb.data(), hidden.data(), hist_at[0].data(), weights, false);
        const double rel_key = rel_l1(exact.key.data(), ref[0].key.data(), hcd);
        const double rel_value = rel_l1(exact.value.data(), ref[0].value.data(), nd);
        const bool ok = gt(rel_key, 10 * kStageTolerance) && gt(rel_value, 10 * kStageTolerance);
        std::printf("  %-46s %s (key %.3e, value %.3e)\n", "Q8_0 / BF16 activation contract observable",
                    ok ? "yes" : "*** NO ***", rel_key, rel_value);
        if (!ok) ++bad;
    }

    // ---- run the block for every token ---------------------------------------------------------------
    std::vector<float> dev_key(hcd), dev_value(nd), dev_gate(k::NG_HC), dev_gated(hcd), dev_norm(hcd),
        dev_conv(hcd), dev_res(hcd);
    float *d_emb = nullptr, *d_hid = nullptr, *d_hist = nullptr, *d_nk = nullptr, *d_nq = nullptr,
          *d_nc = nullptr, *d_ck = nullptr, *d_cv = nullptr, *d_cn = nullptr, *d_cr = nullptr;
    uint8_t* d_kc = nullptr;
    uint16_t *d_vb = nullptr, *d_c1 = nullptr;
    float *d_g = nullptr, *d_v = nullptr, *d_gd = nullptr, *d_nm = nullptr, *d_co = nullptr;
    ck(cudaMalloc(&d_emb, nd * 4), "emb");
    ck(cudaMalloc(&d_hid, hcd * 4), "hid");
    ck(cudaMalloc(&d_hist, (size_t) k::NG_HIST * hcd * 4), "hist");
    ck(cudaMalloc(&d_nk, hcd * 4), "nk");
    ck(cudaMalloc(&d_nq, hcd * 4), "nq");
    ck(cudaMalloc(&d_nc, hcd * 4), "nc");
    ck(cudaMalloc(&d_kc, key_codes.size()), "kc");
    ck(cudaMalloc(&d_vb, value_bf16.size() * 2), "vb");
    ck(cudaMalloc(&d_c1, conv1d_f16.size() * 2), "c1");
    ck(cudaMalloc(&d_g, k::NG_HC * 4), "g");
    ck(cudaMalloc(&d_v, nd * 4), "v");
    ck(cudaMalloc(&d_gd, hcd * 4), "gd");
    ck(cudaMalloc(&d_nm, hcd * 4), "nm");
    ck(cudaMalloc(&d_co, hcd * 4), "co");
    ck(cudaMalloc(&d_ck, hcd * 4), "ck");
    ck(cudaMalloc(&d_cv, nd * 4), "cv");
    ck(cudaMalloc(&d_cn, hcd * 4), "cn");
    ck(cudaMalloc(&d_cr, hcd * 4), "cr");
    ck(cudaMemcpy(d_nk, weights.norm_key.data(), hcd * 4, cudaMemcpyHostToDevice), "cnk");
    ck(cudaMemcpy(d_nq, weights.norm_query.data(), hcd * 4, cudaMemcpyHostToDevice), "cnq");
    ck(cudaMemcpy(d_nc, weights.norm_conv.data(), hcd * 4, cudaMemcpyHostToDevice), "cnc");
    ck(cudaMemcpy(d_kc, key_codes.data(), key_codes.size(), cudaMemcpyHostToDevice), "ckc");
    ck(cudaMemcpy(d_vb, value_bf16.data(), value_bf16.size() * 2, cudaMemcpyHostToDevice), "cvb");
    ck(cudaMemcpy(d_c1, conv1d_f16.data(), conv1d_f16.size() * 2, cudaMemcpyHostToDevice), "cc1");

    k::PleWeights w{};
    w.key_codes = d_kc;
    w.key_scales = key_scales.data();   // uploaded per call below so the pointer is a device one
    w.value_bf16 = d_vb;
    w.norm_key = d_nk;
    w.norm_query = d_nq;
    w.norm_conv = d_nc;
    w.conv1d_f16 = d_c1;
    float* d_ks = nullptr;
    ck(cudaMalloc(&d_ks, key_scales.size() * 4), "ks");
    ck(cudaMemcpy(d_ks, key_scales.data(), key_scales.size() * 4, cudaMemcpyHostToDevice), "cks");
    w.key_scales = d_ks;

    // Per-stage comparison.  Each stage the reference records is a separate line, so a mismatch says WHICH part
    // of the block is wrong rather than only that the sum is.
    struct Stage { const char* name; const std::vector<float>* got; const std::vector<double>* want; };
    double worst_all = 0;

    for (int t = 0; t < kBlockTokens; ++t) {
        ck(cudaMemcpy(d_emb, emb.data() + (size_t) t * nd, nd * 4, cudaMemcpyHostToDevice), "cemb");
        ck(cudaMemcpy(d_hid, hidden.data() + (size_t) t * hcd, hcd * 4, cudaMemcpyHostToDevice), "chid");
        ck(cudaMemcpy(d_hist, hist_at[(size_t) t].data(), (size_t) k::NG_HIST * hcd * 4, cudaMemcpyHostToDevice),
           "chist");
        k::PleOut out{};
        out.key = d_ck; out.value = d_cv; out.gate = d_g; out.gated = d_gd;
        out.normalized = d_nm; out.conv = d_co; out.result = d_cr;
        // the workspace is the caller's, and the sync the block used to do is now the caller's too
    void* ple_ws = nullptr;
    ck(cudaMalloc(&ple_ws, k::ple_block_scratch_bytes()), "ple_block scratch");
    if (t == 0) {
        const size_t bytes = (size_t) k::ple_block_scratch_bytes();
        ck(cudaMemset(ple_ws, 0xa5, bytes), "PLE prelaunch guard sentinel");
        const struct AliasCase { float* k::PleOut::* field; size_t offset; } cases[] = {
            {&k::PleOut::key, 0}, {&k::PleOut::value, hcd}, {&k::PleOut::gate, hcd + nd},
            {&k::PleOut::gated, hcd + nd + (size_t) k::NG_HC},
            {&k::PleOut::normalized, 2 * hcd + nd + (size_t) k::NG_HC},
            {&k::PleOut::conv, 3 * hcd + nd + (size_t) k::NG_HC}, {&k::PleOut::result, 0}
        };
        int refused = 0;
        for (const auto& item : cases) {
            k::PleOut invalid = out;
            invalid.*(item.field) = static_cast<float*>(ple_ws) + item.offset;
            try { k::ple_block(d_emb, d_hid, d_hist, w, invalid, ple_ws, nullptr); }
            catch (const std::invalid_argument&) { ++refused; }
        }
        ck(cudaDeviceSynchronize(), "PLE guard no-launch sync");
        std::vector<uint8_t> sentinel(bytes);
        ck(cudaMemcpy(sentinel.data(), ple_ws, bytes, cudaMemcpyDeviceToHost), "PLE guard sentinel readback");
        const bool unchanged = std::all_of(sentinel.begin(), sentinel.end(), [](uint8_t b) { return b == 0xa5; });
        const bool rejected = refused == 7 && unchanged;
        std::printf("  PLE old caller output offsets rejected before launch: %s (%d/7, scratch %s)\n",
                    rejected ? "pass" : "FAIL", refused, unchanged ? "unchanged" : "CORRUPTED");
        if (!rejected) ++bad;
    }
    k::ple_block(d_emb, d_hid, d_hist, w, out, ple_ws, nullptr);
    ck(cudaDeviceSynchronize(), "ple_block sync");
        ck(cudaMemcpy(dev_key.data(), d_ck, hcd * 4, cudaMemcpyDeviceToHost), "rck");
        ck(cudaMemcpy(dev_value.data(), d_cv, nd * 4, cudaMemcpyDeviceToHost), "rcv");
        ck(cudaMemcpy(dev_gate.data(), d_g, k::NG_HC * 4, cudaMemcpyDeviceToHost), "rg");
        ck(cudaMemcpy(dev_gated.data(), d_gd, hcd * 4, cudaMemcpyDeviceToHost), "rgd");
        ck(cudaMemcpy(dev_norm.data(), d_nm, hcd * 4, cudaMemcpyDeviceToHost), "rnm");
        ck(cudaMemcpy(dev_conv.data(), d_co, hcd * 4, cudaMemcpyDeviceToHost), "rco");
        ck(cudaMemcpy(dev_res.data(), d_cr, hcd * 4, cudaMemcpyDeviceToHost), "rcr");

        const BlockRef& want = ref[(size_t) t];
        const Stage stages[] = {
            {"key        (grouped_norm of ple_key @ emb)", &dev_key, &want.key},
            {"value      (ple_value @ emb, BF16)", &dev_value, &want.value},
            {"gate       (signed sqrt, sigmoid)", &dev_gate, &want.gate},
            {"gated      (value broadcast * gate)", &dev_gated, &want.gated},
            {"normalized (grouped_norm(gated, ple_norm_conv))", &dev_norm, &want.normalized},
            {"conv_out   (dilated conv, then SiLU)", &dev_conv, &want.conv},
            {"result     (hidden + gated + conv)", &dev_res, &want.result},
        };
        for (const Stage& s : stages) {
            double mag = 0;
            long long nf = 0;
            const double rel = rel_l1(s.want->data(), s.got->data(), s.want->size(), &mag, &nf);
            // NaN-SAFE: `rel > worst_all` is false for NaN, so `fmax` here would silently report a perfect
            // score for a fixture full of NaNs - round 197's exact failure.
            worst_all = (rel > worst_all) ? rel : worst_all;
            if (nf) worst_all = std::nan("");
            const bool ok = le(rel, kStageTolerance) && nf == 0;
            std::printf("  token %d %-46s rel %.3e   (mean |ref| %.4f, non-finite %lld)%s\n", t, s.name, rel,
                        mag, nf, ok ? "" : "   *** FAIL ***");
            if (!ok) ++bad;
        }

        if (t == 0) {
            // Deliberate BF16 halfway values make the value projection's activation precision visible.
            // The scalar reference uses the actual packed BF16 weight bits and unrounded F32 inputs.
            std::vector<float> witness(nd), reference_value(nd), native_value(nd), native_result(hcd), replay(hcd);
            for (size_t i = 0; i < nd; ++i) witness[i] = i % 7 == 0 ? 1.00390625f : -0.501953125f;
            for (size_t row = 0; row < nd; ++row) {
                double sum = 0.0;
                for (size_t col = 0; col < nd; ++col) {
                    const uint32_t bits = uint32_t(value_bf16[row * nd + col]) << 16;
                    float weight;
                    std::memcpy(&weight, &bits, sizeof(weight));
                    sum += double(weight) * witness[col];
                }
                reference_value[row] = float(sum);
            }
            ck(cudaMemcpy(d_emb, witness.data(), nd * sizeof(float), cudaMemcpyHostToDevice), "native PLE witness");
            k::ple_block(d_emb, d_hid, d_hist, w, out, ple_ws, nullptr);
            ck(cudaDeviceSynchronize(), "legacy PLE witness sync");
            std::vector<float> legacy_value(nd);
            ck(cudaMemcpy(legacy_value.data(), d_cv, nd * sizeof(float), cudaMemcpyDeviceToHost), "legacy PLE witness value");
            cudaStream_t stream;
            ck(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "native PLE stream");
            k::ple_set_native_bf16(true);
            k::ple_block(d_emb, d_hid, d_hist, w, out, ple_ws, stream);
            ck(cudaStreamSynchronize(stream), "native PLE sync");
            ck(cudaMemcpy(native_value.data(), d_cv, nd * sizeof(float), cudaMemcpyDeviceToHost), "native PLE value");
            ck(cudaMemcpy(native_result.data(), d_cr, hcd * sizeof(float), cudaMemcpyDeviceToHost), "native PLE result");
            long long nf = 0;
            const double rel = rel_l1(reference_value.data(), native_value.data(), nd, nullptr, &nf);
            const double separation = rel_l1(native_value.data(), legacy_value.data(), nd);
            const bool correct = le(rel, 2e-6) && nf == 0 && gt(separation, 1e-5);
            std::printf("  native PLE value: %s (ref rel %.3e, BF16 separation %.3e)\n",
                        correct ? "pass" : "FAIL", rel, separation);
            if (!correct) ++bad;
            cudaGraph_t graph;
            cudaGraphExec_t executable;
            ck(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal), "native PLE capture");
            k::ple_block(d_emb, d_hid, d_hist, w, out, ple_ws, stream);
            ck(cudaStreamEndCapture(stream, &graph), "native PLE capture end");
            ck(cudaGraphInstantiate(&executable, graph, nullptr, nullptr, 0), "native PLE instantiate");
            k::ple_set_native_bf16(false);
            ck(cudaGraphLaunch(executable, stream), "native PLE replay");
            ck(cudaStreamSynchronize(stream), "native PLE replay sync");
            ck(cudaMemcpy(replay.data(), d_cr, hcd * sizeof(float), cudaMemcpyDeviceToHost), "native PLE replay result");
            const bool captured = std::memcmp(native_result.data(), replay.data(), hcd * sizeof(float)) == 0;
            std::printf("  native PLE captured selection: %s\n", captured ? "byte-identical" : "FAIL");
            if (!captured) ++bad;
            ck(cudaGraphExecDestroy(executable), "native PLE graph exec destroy");
            ck(cudaGraphDestroy(graph), "native PLE graph destroy");
            ck(cudaStreamDestroy(stream), "native PLE stream destroy");
            ck(cudaMemcpy(d_emb, emb.data(), nd * sizeof(float), cudaMemcpyHostToDevice), "restore PLE embedding");
            k::ple_block(d_emb, d_hid, d_hist, w, out, ple_ws, nullptr);
            ck(cudaDeviceSynchronize(), "restored PLE sync");
            ck(cudaMemcpy(replay.data(), d_cr, hcd * sizeof(float), cudaMemcpyDeviceToHost), "restored PLE result");
            const bool restored = std::memcmp(dev_res.data(), replay.data(), hcd * sizeof(float)) == 0;
            std::printf("  restored PLE default: %s\n", restored ? "byte-identical" : "FAIL");
            if (!restored) ++bad;
        }

        if (t == 0) {
            // This checks native-projection wiring and graph lifetime, independently
            // of the legacy Q8_0-contract reference above. Actual Q2_0 arithmetic is
            // checked separately by native_mmvq_parity against the pinned CUDA DLL.
            const size_t blocks = n_scales, native_bytes = blocks * 18, guard = 64;
            const size_t qbytes = k::native_q8_1_bytes(k::NG_N_EMBD);
            std::vector<uint8_t> native_key(native_bytes);
            for (size_t b = 0; b < blocks; ++b) {
                std::memcpy(native_key.data() + b * 18, raw_scales.data() + b * 2, 2);
                std::memcpy(native_key.data() + b * 18 + 2, key_codes.data() + b * 16, 16);
            }
            void *native_storage = nullptr, *q_storage = nullptr;
            float* raw_projection = nullptr;
            ck(cudaMalloc(&native_storage, native_bytes + 2 * guard), "native PLE weights");
            ck(cudaMalloc(&q_storage, qbytes + 2 * guard), "native PLE q8 scratch");
            ck(cudaMalloc(&raw_projection, hcd * 4), "native PLE raw projection");
            ck(cudaMemset(native_storage, 0xa5, native_bytes + 2 * guard), "native PLE weight guards");
            ck(cudaMemset(q_storage, 0xa5, qbytes + 2 * guard), "native PLE q8 guards");
            void* native_data = static_cast<uint8_t*>(native_storage) + guard;
            void* native_q = static_cast<uint8_t*>(q_storage) + guard;
            ck(cudaMemcpy(native_data, native_key.data(), native_bytes, cudaMemcpyHostToDevice), "native PLE weights upload");
            k::PleWeights nw = w;
            nw.key_native_data = native_data; nw.key_native_type = 42; nw.key_native_q8_1 = native_q;
            cudaStream_t stream;
            ck(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "native PLE key stream");
            const size_t workspace_bytes = (size_t) k::ple_block_scratch_bytes();
            ck(cudaMemset(ple_ws, 0xa5, workspace_bytes), "native PLE no-launch sentinel");
            // the guards, the weight upload and the sentinel run on the legacy stream, which the non-blocking
            // `stream` does not wait for
            ck(cudaDeviceSynchronize(), "native PLE key setup");
            int refused = 0;
            for (int c = 0; c < 9; ++c) {
                auto invalid = nw;
                if (c == 0) invalid.key_native_type = 11;
                if (c == 1) invalid.key_native_q8_1 = nullptr;
                if (c == 3) invalid.key_native_q8_1 = ple_ws;
                if (c == 4) invalid.key_native_q8_1 = d_emb;
                if (c == 5) invalid.key_native_q8_1 = out.key;
                if (c == 6) invalid.key_native_q8_1 = static_cast<uint8_t*>(native_q) + 1;
                if (c == 7) invalid.key_native_data = static_cast<uint8_t*>(native_data) + 1;
                if (c == 8) invalid.key_native_q8_1 = native_data;
                try { k::ple_block(d_emb, d_hid, d_hist, invalid, out, ple_ws, c == 2 ? nullptr : stream); }
                catch (const std::invalid_argument&) { ++refused; }
            }
            ck(cudaStreamSynchronize(stream), "native PLE refusal sync");
            std::vector<uint8_t> sentinel(workspace_bytes);
            ck(cudaMemcpy(sentinel.data(), ple_ws, workspace_bytes, cudaMemcpyDeviceToHost), "native PLE refusal sentinel");
            const bool untouched = std::all_of(sentinel.begin(), sentinel.end(), [](uint8_t x) { return x == 0xa5; });
            std::printf("  native PLE key prelaunch guards: %s (%d/9, workspace %s)\n",
                        refused == 9 && untouched ? "pass" : "FAIL", refused, untouched ? "unchanged" : "changed");
            if (refused != 9 || !untouched) ++bad;
            cudaGraph_t graph = nullptr; cudaGraphExec_t executable = nullptr;
            std::vector<float> projected(hcd), normalized(hcd), actual_key(hcd), actual_result(hcd), replay(hcd);
            for (int p = 0; p < std::min(kBlockTokens, 2); ++p) {
                ck(cudaMemcpy(d_emb, emb.data() + (size_t) p * nd, nd * 4, cudaMemcpyHostToDevice), "native PLE key input");
                ck(cudaDeviceSynchronize(), "native PLE key input sync");
                k::native_q2_0_f32(native_data, d_emb, native_q, raw_projection, k::NG_N_EMBD, k::NG_HC_DIM, 1, stream);
                ck(cudaStreamSynchronize(stream), "native PLE raw key sync");
                ck(cudaMemcpy(projected.data(), raw_projection, hcd * 4, cudaMemcpyDeviceToHost), "native PLE raw key read");
                for (int c = 0; c < k::NG_HC; ++c) {
                    double sum = 0;
                    for (size_t j = 0; j < nd; ++j) { const float v = projected[(size_t) c * nd + j]; sum += double(v * v); }
                    const float scale = 1.0f / std::sqrt(float(sum / double(nd)) + k::NG_RMS_EPS);
                    for (size_t j = 0; j < nd; ++j) { const size_t i = (size_t) c * nd + j; normalized[i] = projected[i] * scale * weights.norm_key[i]; }
                }
                k::ple_block(d_emb, d_hid, d_hist, nw, out, ple_ws, stream);
                ck(cudaStreamSynchronize(stream), "native PLE key direct sync");
                ck(cudaMemcpy(actual_key.data(), d_ck, hcd * 4, cudaMemcpyDeviceToHost), "native PLE key read");
                ck(cudaMemcpy(actual_result.data(), d_cr, hcd * 4, cudaMemcpyDeviceToHost), "native PLE result read");
                const double rel = rel_l1(normalized.data(), actual_key.data(), hcd);
                const bool wired = le(rel, 2e-6) && nonfinite(actual_result.data(), hcd) == 0;
                std::printf("  native PLE key projection->existing norm input %d: %s (rel %.3e)\n", p, wired ? "pass" : "FAIL", rel);
                if (!wired) ++bad;
                if (p == 0) {
                    ck(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal), "native PLE key capture");
                    k::ple_block(d_emb, d_hid, d_hist, nw, out, ple_ws, stream);
                    ck(cudaStreamEndCapture(stream, &graph), "native PLE key capture end");
                    ck(cudaGraphInstantiate(&executable, graph, nullptr, nullptr, 0), "native PLE key instantiate");
                }
                nw.key_native_data = nullptr; // graph selection must survive descriptor changes
                for (int repeat = 0; repeat < 2; ++repeat) {
                    ck(cudaGraphLaunch(executable, stream), "native PLE key replay");
                    ck(cudaStreamSynchronize(stream), "native PLE key replay sync");
                    ck(cudaMemcpy(replay.data(), d_cr, hcd * 4, cudaMemcpyDeviceToHost), "native PLE key replay read");
                    if (std::memcmp(actual_result.data(), replay.data(), hcd * 4) != 0) ++bad;
                }
                nw.key_native_data = native_data;
            }
            ck(cudaGraphExecDestroy(executable), "native PLE key executable destroy");
            ck(cudaGraphDestroy(graph), "native PLE key graph destroy");
            for (int b = 0; b < 2; ++b) {
                const void* storage = b == 0 ? native_storage : q_storage;
                const size_t payload = b == 0 ? native_bytes : qbytes;
                std::vector<uint8_t> ends(2 * guard);
                ck(cudaMemcpy(ends.data(), storage, guard, cudaMemcpyDeviceToHost), "native PLE prefix guard");
                ck(cudaMemcpy(ends.data() + guard, static_cast<const uint8_t*>(storage) + guard + payload, guard, cudaMemcpyDeviceToHost), "native PLE suffix guard");
                if (!std::all_of(ends.begin(), ends.end(), [](uint8_t x) { return x == 0xa5; })) ++bad;
            }
            ck(cudaStreamDestroy(stream), "native PLE key stream destroy");
            cudaFree(raw_projection); cudaFree(q_storage); cudaFree(native_storage);
            ck(cudaMemcpy(d_emb, emb.data(), nd * 4, cudaMemcpyHostToDevice), "native PLE restore input");
            k::ple_block(d_emb, d_hid, d_hist, w, out, ple_ws, nullptr);
            ck(cudaDeviceSynchronize(), "native PLE default restore sync");
            ck(cudaMemcpy(replay.data(), d_cr, hcd * 4, cudaMemcpyDeviceToHost), "native PLE default restore read");
            const bool restored = std::memcmp(dev_res.data(), replay.data(), hcd * 4) == 0;
            std::printf("  native PLE key graph/repeat/buffer guards checked; restored default: %s\n", restored ? "byte-identical" : "FAIL");
            if (!restored) ++bad;
        }

        if (t == 0) {
            // The engine only needs result and normalized. Export normalized after the complete private
            // workspace; result may overwrite hidden once its original values are no longer needed.
            const size_t workspace_bytes = (size_t) k::ple_block_scratch_bytes();
            void* compact_workspace = nullptr;
            ck(cudaMalloc(&compact_workspace, workspace_bytes + hcd * sizeof(float)), "compact PLE workspace");
            k::PleOut compact{};
            compact.normalized = reinterpret_cast<float*>(static_cast<uint8_t*>(compact_workspace) + workspace_bytes);
            compact.result = d_hid;
            k::ple_block(d_emb, d_hid, d_hist, w, compact, compact_workspace, nullptr);
            ck(cudaDeviceSynchronize(), "compact PLE sync");
            std::vector<float> compact_norm(hcd), compact_result(hcd);
            ck(cudaMemcpy(compact_norm.data(), compact.normalized, hcd * sizeof(float), cudaMemcpyDeviceToHost), "compact PLE normalized");
            ck(cudaMemcpy(compact_result.data(), d_hid, hcd * sizeof(float), cudaMemcpyDeviceToHost), "compact PLE result");
            const bool equal = std::memcmp(compact_norm.data(), dev_norm.data(), hcd * sizeof(float)) == 0 &&
                               std::memcmp(compact_result.data(), dev_res.data(), hcd * sizeof(float)) == 0;
            std::printf("  PLE separate exports and in-place hidden result: %s\n", equal ? "byte-identical" : "FAIL");
            if (!equal) ++bad;
            ck(cudaMemcpy(d_hid, hidden.data(), hcd * sizeof(float), cudaMemcpyHostToDevice), "restore PLE hidden input");
            cudaFree(compact_workspace);
        }

        cudaFree(ple_ws);
    }

    // ---- the trap that matters most: gated vs normalized as the conv input --------------------------
    // `ref/ngram.py`'s docstring calls `terms` "the padded gated values"; the source pads `normalized`.
    // Feeding the conv the gated values instead is a one-word change with every shape intact, so it is
    // computed here and required to differ.
    {
        const double rel_norm_gated = rel_l1(ref[0].gated.data(), ref[0].normalized.data(), hcd);
        std::printf("  %-52s %s (%.2f%% apart)\n", "normalized vs gated as the conv input is observable",
                    rel_norm_gated > 0.05 ? "yes" : "*** NO ***", rel_norm_gated * 100);
        if (!(rel_norm_gated > 0.05)) ++bad;
    }

    // ---- the conv TAP ORDER: tap 0 reads the furthest back, not the current row ---------------------
    {
        // Recompute the conv on the host with the taps reversed, from the reference's own `normalized`, and
        // require the result to differ from the reference's conv_out.
        std::vector<double> rev(hcd, 0.0);
        for (size_t c = 0; c < hcd; ++c) {
            double acc = 0;
            for (int kk = 0; kk < k::PLE_CONV_KERNEL; ++kk) {
                // THE RIVAL READING OF THE TAP CONVENTION.  ggml's is `t - (K-1-k)*d`, i.e. tap 0 reads the
                // FURTHEST back; the natural misreading is `t - k*d`, i.e. tap 0 reads the CURRENT row.
                //
                // The first version of this trap wrote `row = kk*dil` for the rival - which is ALGEBRAICALLY
                // THE SAME as the correct `hist - (K-1-kk)*dil` whenever `hist == (K-1)*dil`, because
                // `9-(3-kk)*3 == 3kk` identically.  It reported "0.00% apart" and was measuring nothing.
                // The rival has to REVERSE the row order: `(K-1-kk)*dil` gives 9,6,3,0 against the correct
                // 0,3,6,9, so tap 0 pairs the kernel's first weight with the NEWEST row instead of the oldest.
                const int row = (k::PLE_CONV_KERNEL - 1 - kk) * k::NGRAM_SIZE;
                // ROW-FASTEST: the history is ggml's `ne=(hist, hc_dim)`, flat = row + hist*channel
                const double v = (row == k::NG_HIST) ? ref[0].normalized[c]
                                                     : (double) hist_at[0][(size_t) row + (size_t) k::NG_HIST * c];
                // GGML-NATIVE: `kW[k + kern*c]`, because the tensor is ne=(kern, hc_dim) with ne0 = kern fast.
                acc += (double) k::f32_from_f16(conv1d_f16[(size_t) kk + (size_t) k::PLE_CONV_KERNEL * c]) * v;
            }
            rev[c] = acc / (1.0 + std::exp(-acc));
        }
        const double rel = rel_l1(ref[0].conv.data(), rev.data(), hcd);
        std::printf("  %-52s %s (%.2f%% apart)\n", "reversed conv tap order is observable",
                    gt(rel, 0.05) ? "yes" : "*** NO ***", rel * 100);
        if (!gt(rel, 0.05)) ++bad;
    }

    std::printf("\nple_parity: %d failures (worst stage rel %.3e)\n", bad, worst_all);
    if (bad) return 1;
    if (selftest) std::printf("ple_parity OK\n");
    return 0;
}
