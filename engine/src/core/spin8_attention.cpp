#include "strata/core/spin8_attention.hpp"
#include "strata/kernels/qsa.hpp"
#include "spin8/spin8.h"
#include "spin8/cuda.h"
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <sstream>
#include <vector>

namespace strata::core {
namespace {
bool active = false;
std::array<double, 192> matrices{};
std::array<double, 3> weights{1.0/3, 1.0/3, 1.0/3};
uint64_t completed = 0;
spin8_monitor monitor{};
const auto epoch = std::chrono::steady_clock::now();
}
bool spin8_attention_enabled() { return active; }

bool spin8_attention_configure(const std::string& mode, const std::string& profile, std::string& error) {
    if (mode == "off") { active = false; return true; }
    if (active) { error = "spin8_profile_change_requires_new_process"; return false; }
    if (mode != "vector" && mode != "spinor-plus" && mode != "spinor-minus" && mode != "consensus") {
        error = "spin8_invalid_mode"; return false;
    }
    std::ifstream input(profile);
    std::string line;
    if (!std::getline(input, line) || line != "spin8-profile-v1") {
        error = "spin8_invalid_profile_version"; return false;
    }
    std::vector<spin8_plane> planes;
    size_t bytes = line.size();
    while (std::getline(input, line)) {
        bytes += line.size();
        if (bytes > 65536 || planes.size() >= 4096) { error = "spin8_profile_too_large"; return false; }
        if (line.empty() || line[0] == '#') continue;
        spin8_plane plane{}; std::string extra; std::istringstream row(line);
        if (!(row >> plane.a >> plane.b >> plane.angle) || (row >> extra)) {
            error = "spin8_invalid_plane"; return false;
        }
        planes.push_back(plane);
    }
    if (input.bad() || planes.empty()) { error = "spin8_empty_or_unreadable_profile"; return false; }
    spin8_plan* plan = nullptr;
    spin8_status result = spin8_plan_create(planes.data(), planes.size(), &plan);
    if (result != SPIN8_OK) { error = spin8_status_string(result); return false; }
    spin8_audit_report audit{};
    result = spin8_plan_audit(plan, &audit);
    auto candidate = matrices;
    for (int view = 0; view < 3 && result == SPIN8_OK; ++view)
        result = spin8_plan_matrix(plan, static_cast<spin8_view>(view), candidate.data() + view * 64);
    spin8_plan_free(plan);
    if (result != SPIN8_OK) { error = "transform_invariant_violation"; return false; }
    matrices = candidate;
    weights = {1.0/3, 1.0/3, 1.0/3};
    if (mode != "consensus") { weights = {0,0,0}; weights[mode == "vector" ? 0 : mode == "spinor-plus" ? 1 : 2] = 1; }
    active = true;
    completed = 0;
    spin8_monitor_init(&monitor);
    return true;
}

bool spin8_attention_check(std::string& error) {
    if (!active) return true;
    const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - epoch).count();
    int due = 0;
    auto result = spin8_monitor_due(&monitor, ++completed, seconds, &due);
    if (result != SPIN8_OK) { error = "transform_invariant_violation"; return false; }
    if (!due) return true;
    for (size_t view = 0; view < 3; ++view) {
        spin8_matrix_audit report{};
        result = spin8_audit_matrix(matrices.data() + view * 64, &report);
        if (result == SPIN8_OK && (report.det_sign != 1 || report.ef_normalized > 1e-12 ||
            report.max_error > 1e-12 || std::abs(report.log_abs_det) > 8e-12)) result = SPIN8_AUDIT_FAILED;
        if (result != SPIN8_OK) break;
        std::fprintf(stderr, "{\"event\":\"spin8_audit\",\"domain\":\"coefficient-f64\",\"view\":%zu,\"ef\":%.17g,\"max\":%.17g,\"det_sign\":%d,\"log_abs_det\":%.17g}\n",
            view, report.ef_normalized, report.max_error, report.det_sign, report.log_abs_det);
        std::array<double, 64> deployed{};
        for (size_t i = 0; i < deployed.size(); ++i) deployed[i] = static_cast<float>(matrices[view * 64 + i]);
        result = spin8_audit_matrix(deployed.data(), &report);
        if (result == SPIN8_OK && (report.det_sign != 1 || report.ef_normalized > 1e-5 ||
            report.max_error > 1e-5 || std::abs(report.log_abs_det) > 8e-5)) result = SPIN8_AUDIT_FAILED;
        if (result != SPIN8_OK) break;
        std::fprintf(stderr, "{\"event\":\"spin8_audit\",\"domain\":\"coefficient-f32\",\"view\":%zu,\"ef\":%.17g,\"max\":%.17g,\"det_sign\":%d,\"log_abs_det\":%.17g}\n",
            view, report.ef_normalized, report.max_error, report.det_sign, report.log_abs_det);
    }
    spin8_monitor_record(&monitor, completed, seconds, result);
    if (result != SPIN8_OK) { error = "transform_invariant_violation"; return false; }
    return true;
}

bool spin8_attention_step(const float* query, const uint16_t* key, const uint16_t* value,
    const int32_t* step, size_t capacity, size_t heads, size_t kv_heads, size_t dimension,
    float* scores, float* output, int32_t* status, int32_t* host_status, void* stream, std::string& error) {
    if (!active || !stream || !status || !host_status || !kv_heads || heads % kv_heads) {
        error = "spin8_invalid_native_attention"; return false;
    }
    const auto st = static_cast<cudaStream_t>(stream);
    cudaError_t result = cudaMemsetAsync(status, 0, sizeof(int32_t), st);
    for (size_t head = 0; head < heads && result == cudaSuccess; ++head) {
        const size_t kv_head = head / (heads / kv_heads);
        result = spin8_cuda_attention_f32(query + head * dimension, dimension,
            key + kv_head * dimension, value + kv_head * dimension, kv_heads * dimension, SPIN8_F16,
            output + head * dimension, dimension, 1, capacity, dimension, nullptr,
            matrices.data(), weights.data(), 1, 0, scores, step + strata::kernels::kStepWidth, status, 0, st);
    }
    if (result == cudaSuccess) result = cudaMemcpyAsync(host_status, status, sizeof(int32_t), cudaMemcpyDeviceToHost, st);
    if (result != cudaSuccess) { error = std::string("spin8_cuda: ") + cudaGetErrorString(result); return false; }
    return true;
}
}
