// Isolated device-event measurements of existing BF16/F32 MMVF paths.
// Queue spans include scheduling gaps; these are not graph-replay node times.
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include "strata/kernels/bf16_gemv.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

// The BF16 kernel GRF size is a build option in some trees; report 0 (default) when the
// bench is built standalone.
#ifndef STRATA_SYCL_BF16_GRF
#define STRATA_SYCL_BF16_GRF 0
#endif

namespace {
using Clock = std::chrono::steady_clock;
constexpr size_t guard = 64;
constexpr float sentinel = -1234567.0f;

std::string quote(const std::string& text) {
    std::string out = "\"";
    for (char c : text) {
        if (c == '\\' || c == '"') out += '\\';
        if (c == '\n') out += "\\n";
        else out += c;
    }
    return out + '"';
}
uint32_t bits(float value) {
    uint32_t out;
    std::memcpy(&out, &value, sizeof(out));
    return out;
}
float from_bf16(uint16_t value) {
    uint32_t word = uint32_t(value) << 16;
    float out;
    std::memcpy(&out, &word, sizeof(out));
    return out;
}
uint16_t to_bf16(float value) { return uint16_t(bits(value) >> 16); }
struct DeviceBuffer {
    sycl::queue& q;
    void* p;
    DeviceBuffer(sycl::queue& queue, size_t bytes) : q(queue), p(sycl::malloc_device(bytes, q)) {
        if (!p) throw std::bad_alloc();
    }
    ~DeviceBuffer() {
        try { q.wait_and_throw(); } catch (...) {}
        sycl::free(p, q);
    }
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
};

double median(std::vector<double> values) {
    std::sort(values.begin(), values.end());
    return values[values.size()/2];
}
struct Timing {
    double median_us, min_us, wall_us;
};
Timing measure(sycl::queue& q, const std::function<void()>& launch, int reps, int samples) {
    std::vector<std::pair<sycl::event, sycl::event>> marks;
    const auto t0 = Clock::now();
    for (int s = 0; s < samples; ++s) {
        auto before = q.single_task([] {});
        for (int r = 0; r < reps; ++r) launch();
        auto after = q.single_task([] {});
        marks.emplace_back(before, after);
    }
    q.wait_and_throw();
    const double wall = std::chrono::duration<double, std::micro>(Clock::now()-t0).count();
    std::vector<double> us;
    for (const auto& [before, after] : marks) {
        const auto start = before.get_profiling_info<sycl::info::event_profiling::command_end>();
        const auto end = after.get_profiling_info<sycl::info::event_profiling::command_start>();
        if (!start || !end || end < start)
            throw std::runtime_error("device event timestamps missing or non-monotonic");
        us.push_back(double(end-start)/1000.0/reps);
    }
    return {median(us), *std::min_element(us.begin(), us.end()), wall/(reps*samples)};
}

struct Shape { int n, k; };
int run(sycl::queue& q, Shape shape, int tokens, bool padded, bool extremes,
        int rows_mode, int reps, int samples, const std::string& revision, bool zeros = false) {
    using namespace strata::kernels;
    const int n = shape.n, k = shape.k;
    const int ldx = k + (padded ? 6 : 0), ldy = n + (padded ? 7 : 0);
    std::vector<uint16_t> weights(size_t(n)*k);
    std::vector<float> x(size_t(tokens)*ldx, sentinel);
    uint32_t random = 1234567;
    auto next = [&] {
        random = random*1664525u + 1013904223u;
        return float(int((random >> 8) & 0xffff)-32768)*0.0000314159f;
    };
    for (size_t i = 0; i < weights.size(); ++i) {
        float scale = extremes ? ((i/size_t(k)) % 2 ? 1e-20f : 1e20f) : 0.125f;
        weights[i] = to_bf16(next()*scale);
    }
    for (int t = 0; t < tokens; ++t)
        for (int j = 0; j < k; ++j) x[size_t(t)*ldx+j] = zeros ? 0.0f : extremes ? next()*1e-10f : next();
    const size_t out_size = guard + size_t(tokens)*ldy + guard;
    std::vector<float> initial(out_size, sentinel), got(out_size), reference(out_size);
    DeviceBuffer wb(q, weights.size()*sizeof(uint16_t)), xb(q, x.size()*sizeof(float));
    DeviceBuffer yb(q, out_size*sizeof(float)), rb(q, out_size*sizeof(float));
    auto* w = static_cast<uint16_t*>(wb.p);
    auto* dx = static_cast<float*>(xb.p);
    auto* y = static_cast<float*>(yb.p) + guard;
    auto* ref = static_cast<float*>(rb.p) + guard;
    q.memcpy(w, weights.data(), weights.size()*sizeof(uint16_t));
    q.memcpy(dx, x.data(), x.size()*sizeof(float));
    q.memcpy(yb.p, initial.data(), out_size*sizeof(float));
    q.memcpy(rb.p, initial.data(), out_size*sizeof(float));
    auto launch = [&] { bf16_gemv_fp32_mmvf_multi(dx, ldx, w, y, ldy, k, n, tokens, &q); };
    for (int t = 0; t < tokens; ++t)
        bf16_gemv_fp32_mmvf(dx+size_t(t)*ldx, w, ref+size_t(t)*ldy, k, n, &q);
    launch();
    q.memcpy(reference.data(), rb.p, out_size*sizeof(float));
    q.memcpy(got.data(), yb.p, out_size*sizeof(float));
    q.wait_and_throw();
    size_t mismatches = 0, guards_bad = 0, nonfinite = 0;
    uint64_t hash = 1469598103934665603ull;
    double worst_oracle_normalized = 0;
    for (size_t i = 0; i < out_size; ++i) {
        bool output = i >= guard && i < guard + size_t(tokens)*ldy && ((i-guard)%ldy) < size_t(n);
        if (!output) { guards_bad += bits(got[i]) != bits(sentinel); continue; }
        mismatches += bits(got[i]) != bits(reference[i]);
        nonfinite += !std::isfinite(got[i]);
        hash = (hash ^ bits(got[i])) * 1099511628211ull;
    }
    for (int t = 0; t < tokens; ++t)
        for (int r = 0; r < std::min(n, 8); ++r) {
            double oracle = 0, abs_sum = 0;
            for (int j = 0; j < k; ++j) {
                double product = double(from_bf16(weights[size_t(r)*k+j]))*x[size_t(t)*ldx+j];
                oracle += product; abs_sum += std::abs(product);
            }
            worst_oracle_normalized = std::max(worst_oracle_normalized,
                std::abs(double(got[guard+size_t(t)*ldy+r])-oracle)/std::max(abs_sum, 1e-30));
        }
    constexpr int warmup = 8;
    for (int i = 0; i < warmup; ++i) launch();
    q.wait_and_throw();
    const Timing empty = measure(q, [] {}, reps, samples);
    const Timing timing = measure(q, launch, reps, samples);
    q.memcpy(got.data(), yb.p, out_size*sizeof(float)).wait_and_throw();
    if (std::memcmp(got.data(), reference.data(), out_size*sizeof(float))) ++mismatches;
    const bool pass = !mismatches && !guards_bad && !nonfinite && worst_oracle_normalized <= 1e-5;
    std::ostringstream hs; hs << std::hex << hash;
    std::cout << std::setprecision(9)
        << "{\"revision\":" << quote(revision) << ",\"device\":"
        << quote(q.get_device().get_info<sycl::info::device::name>())
        << ",\"driver\":" << quote(q.get_device().get_info<sycl::info::device::driver_version>())
        << ",\"compiler\":" << quote(__VERSION__)
        << ",\"bf16_grf\":" << STRATA_SYCL_BF16_GRF
        << ",\"family\":\"bf16_f32_mmvf\",\"rows_mode\":" << rows_mode
        << ",\"n\":" << n << ",\"k\":" << k << ",\"tokens\":" << tokens
        << ",\"ldx\":" << ldx << ",\"ldy\":" << ldy << ",\"extremes\":" << (extremes ? "true" : "false")
        << ",\"zero_input\":" << (zeros ? "true" : "false")
        << ",\"repeats\":" << reps << ",\"samples\":" << samples << ",\"warmup\":" << warmup
        << ",\"queue\":\"in_order+enable_profiling\",\"measurement\":\"device_queue_span_per_call\""
        << ",\"device_median_us\":" << timing.median_us << ",\"device_min_us\":" << timing.min_us
        << ",\"host_wall_us_per_call\":" << timing.wall_us << ",\"empty_marker_us_per_call\":" << empty.median_us
        << ",\"weight_gbps\":" << double(weights.size()*2)/timing.median_us/1000
        << ",\"bit_mismatches\":" << mismatches << ",\"guard_failures\":" << guards_bad
        << ",\"nonfinite\":" << nonfinite << ",\"oracle_normalized_error\":" << worst_oracle_normalized
        << ",\"output_hash\":" << quote(hs.str()) << ",\"pass\":" << (pass ? "true" : "false") << "}\n";
    std::cout.flush();
    return !pass;
}
}

int main(int argc, char** argv) try {
    int rows_mode = 0, reps = 16, samples = 5;
    bool quick = false;
    std::string revision = "unrecorded-working-tree";
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        auto value = [&] { if (++i >= argc) throw std::invalid_argument("missing value for " + arg); return std::string(argv[i]); };
        if (arg == "--rows") rows_mode = std::stoi(value());
        else if (arg == "--repeats") reps = std::stoi(value());
        else if (arg == "--samples") samples = std::stoi(value());
        else if (arg == "--revision") revision = value();
        else if (arg == "--quick") quick = true;
        else throw std::invalid_argument("unknown argument: " + arg);
    }
    if ((rows_mode != 0 && rows_mode != 1) || reps < 1 || samples < 1 || samples > 100)
        throw std::invalid_argument("rows must be 0/1; repeats and samples must be positive (samples <=100)");
    setenv("STRATA_MMVF_ROWS", rows_mode ? "1" : "0", 1);
    auto& dq = dpct::get_in_order_queue();
    sycl::queue q(dq.get_context(), dq.get_device(), sycl::property_list{
        sycl::property::queue::in_order{}, sycl::property::queue::enable_profiling{}});
    std::vector<Shape> shapes = quick ? std::vector<Shape>{{4, 10240}, {65, 258}, {256, 2560}}
        : std::vector<Shape>{{4, 10240}, {320, 10240}, {10240, 320}, {256, 2560}, {2560, 2560}, {10240, 2560}};
    const std::vector<int> tokens = quick ? std::vector<int>{1, 3, 8} : std::vector<int>{1, 2, 3, 4, 6, 8};
    int failures = 0;
    for (Shape shape : shapes)
        for (int t : tokens) failures += run(q, shape, t, false, false, rows_mode, reps, samples, revision);
    for (int t : {3, 8}) failures += run(q, {65, 258}, t, true, false, rows_mode, reps, samples, revision);
    failures += run(q, {65, 258}, 3, true, true, rows_mode, reps, samples, revision);
    failures += run(q, {4, 2560}, 3, false, false, rows_mode, reps, samples, revision, true);
    DeviceBuffer dummy(q, 1024);
    auto* x = static_cast<float*>(dummy.p);
    auto* w = static_cast<uint16_t*>(dummy.p);
    int rejected = 0;
    const std::vector<std::function<void()>> invalid{
        [&] { strata::kernels::bf16_gemv_fp32_mmvf_multi(x, 257, w, x, 65, 258, 65, 2, &q); },
        [&] { strata::kernels::bf16_gemv_fp32_mmvf_multi(x, 258, w, x, 65, 258, 0, 1, &q); },
        [&] { strata::kernels::bf16_gemv_fp32_mmvf_multi(x, 258, w, x, 65, 258, 65, 9, &q); },
        [&] { strata::kernels::bf16_gemv_fp32_mmvf_multi(x, 258, w, x, 65, 257, 65, 2, &q); },
        [&] { strata::kernels::bf16_gemv_fp32_mmvf_multi(nullptr, 258, w, x, 65, 258, 65, 2, &q); }
    };
    for (const auto& call : invalid) {
        try { call(); } catch (const std::invalid_argument&) { ++rejected; }
    }
    failures += rejected != int(invalid.size());
    std::cout << "{\"invalid_shape_rejections\":" << rejected << ",\"expected\":" << invalid.size() << "}\n";
    std::cout << "{\"summary\":true,\"failures\":" << failures << "}\n";
    return failures != 0;
} catch (const std::exception& exc) {
    std::cerr << "bf16 event benchmark: " << exc.what() << '\n';
    return 2;
}
