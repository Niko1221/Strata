// src/core/steer.cpp - load and apply per-layer residual control vectors.
#include "strata/core/steer.hpp"
#include "strata/artifact/gguf_reader.hpp"
#include "strata/kernels/elementwise.hpp"
#include "strata/kernels/bf16_bits.hpp"
#include "strata/kernels/f16_bits.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cerrno>
#include <cmath>
#include <cstring>
#include <cstdlib>
#include <map>
#include <sstream>

namespace strata::core {
namespace {

const SteeringPlan* g_plan = nullptr;

struct Spec {
    std::string path;
    float alpha = 1.0f;  ///< the strength after the colon
};

/// `path.gguf[:scale]`; scale controls how much of each normalized layer direction is projected out.
bool parse_spec(const std::string& spec, Spec& out, std::string& err) {
    const size_t colon = spec.rfind(':');
    if (colon == std::string::npos) {
        out.path = spec;
        out.alpha = 1.0f;
        return true;
    }
    out.path = spec.substr(0, colon);
    const std::string tail = spec.substr(colon + 1);
    if (tail.empty()) {
        err = "--control-vector-scaled: empty scale in " + spec;
        return false;
    }
    char* end = nullptr;
    out.alpha = std::strtof(tail.c_str(), &end);
    if (end == tail.c_str() || *end != '\0' || !std::isfinite(out.alpha)) {
        err = "--control-vector-scaled: bad scale in " + spec;
        return false;
    }
    return true;
}

/// One `direction.<L>` tensor, dequantized to host f32.  Only the three float encodings make
/// sense for a direction; an i-quant here would be a quantizer accident, not a format.
bool read_direction(const strata::GgufFile& gguf, const strata::TensorInfo& t, int64_t n_embd,
                    std::vector<float>& out, std::string& err) {
    if (t.shape.size() != 1 || (int64_t) t.shape[0] != n_embd) {
        err = t.name + " is " + std::to_string(t.shape.size()) + "-D of " +
              std::to_string(t.shape.empty() ? 0 : (int64_t) t.shape[0]) + " values, the model's "
              "branch output is " + std::to_string(n_embd) + "-wide";
        return false;
    }
    const uint8_t* base = gguf.tensor_data(t);
    out.resize((size_t) n_embd);
    if (t.type == 0) {  // F32
        std::memcpy(out.data(), base, (size_t) n_embd * 4);
    } else if (t.type == 1) {  // F16
        for (int64_t i = 0; i < n_embd; ++i) {
            uint16_t h;
            std::memcpy(&h, base + (size_t) i * 2, 2);
            out[(size_t) i] = strata::kernels::f32_from_f16(h);
        }
    } else if (t.type == 30) {  // BF16
        for (int64_t i = 0; i < n_embd; ++i) {
            uint16_t h;
            std::memcpy(&h, base + (size_t) i * 2, 2);
            out[(size_t) i] = strata::kernels::f32_from_bf16(h);
        }
    } else {
        err = t.name + " is " + t.type_name() + "; directions are F32, F16 or BF16";
        return false;
    }
    return true;
}

}  // namespace

void steer_set_active(const SteeringPlan* plan) { g_plan = plan; }
const SteeringPlan* steer_active() { return g_plan; }

bool SteeringPlan::load(const std::vector<std::string>& specs, const ModelGeometry& g,
                        std::string& err, int64_t layer_first, int64_t layer_last) {
    if (layer_first >= 0 && layer_last < layer_first) {
        err = "steering: the layer range is empty (" + std::to_string(layer_first) + ".." +
              std::to_string(layer_last) + ")";
        return false;
    }
    free();
    entries_.clear();
    arena_ = nullptr;
    arena_bytes_ = 0;

    // Every file's directions, resolved to (layer, floats).  A file may be named by several specs
    // at different layers; that is the multi-vector case and it just works.
    std::vector<SteeringVector> resolved;
    try {
        for (const std::string& spec : specs) {
            Spec s;
            if (!parse_spec(spec, s, err)) return false;
            strata::GgufFile gguf(s.path);
            std::map<int64_t, const strata::TensorInfo*> directions;
            for (const auto& t : gguf.tensors()) {
                if (t.name.rfind("direction.", 0) != 0) continue;
                const char* first = t.name.c_str() + 10;
                char* end = nullptr;
                errno = 0;
                const long long parsed = std::strtoll(first, &end, 10);
                if (first == end || *end != '\0' || errno == ERANGE || parsed < 0 || parsed >= g.n_layers) {
                    err = s.path + " has an invalid control-vector tensor name: " + t.name;
                    return false;
                }
                const int64_t layer = (int64_t) parsed;
                if (!directions.emplace(layer, &t).second) {
                    err = s.path + " has duplicate direction tensor for layer " + std::to_string(layer);
                    return false;
                }
            }
            if (directions.empty()) {
                err = s.path + " carries no direction.<L> tensors";
                return false;
            }
            for (const auto& [layer, tensor] : directions) {
                if (layer_first >= 0 && (layer < layer_first || layer > layer_last)) continue;
                SteeringVector v;
                v.layer = layer;
                v.alpha = s.alpha;
                v.spec = spec;
                std::vector<float> host;
                if (!read_direction(gguf, *tensor, g.n_embd, host, err)) return false;
                double norm2 = 0.0;
                for (float x : host) norm2 += (double) x * x;
                const double norm = std::sqrt(norm2);
                if (!(norm > 0.0) || !std::isfinite(norm)) {
                    err = tensor->name + " has a zero or non-finite direction norm";
                    return false;
                }
                v.alpha = (float) (v.alpha * norm);
                for (float& x : host) x = (float) (x / norm);
                resolved.push_back(v);
                host_.push_back(std::move(host));
            }
        }
    } catch (const std::exception& e) {
        err = std::string("steering: ") + e.what();
        free();
        return false;
    }
    if (resolved.empty()) return true;

    // one device arena, vectors back to back, then fix up the pointers
    arena_bytes_ = (int64_t) resolved.size() * g.n_embd * 4;
    void* arena = nullptr;
    if (cudaMalloc(&arena, (size_t) arena_bytes_) != cudaSuccess) {
        err = "steering: the direction arena allocation failed";
        free();
        return false;
    }
    arena_ = arena;
    for (size_t i = 0; i < resolved.size(); ++i) {
        void* dst = (uint8_t*) arena + (size_t) i * (size_t) g.n_embd * 4;
        if (cudaMemcpy(dst, host_[i].data(), (size_t) g.n_embd * 4, cudaMemcpyHostToDevice) !=
            cudaSuccess) {
            err = "steering: the direction upload failed";
            free();
            return false;
        }
        resolved[(size_t) i].dev = (const float*) dst;
    }
    entries_ = std::move(resolved);
    host_.clear();
    return true;
}

void SteeringPlan::free() {
    if (arena_) cudaFree(arena_);
    arena_ = nullptr;
    arena_bytes_ = 0;
    entries_.clear();
    host_.clear();
}

bool cvec_residual(float* residual, int64_t tokens, int64_t hc, int64_t layer, int64_t n_embd,
                   void* stream, std::string& err) {
    const SteeringPlan* plan = g_plan;
    if (!plan) return true;
    for (const SteeringVector& v : plan->entries()) {
        if (v.layer != layer || !v.dev) continue;
        const int64_t rows = tokens * hc;
        strata::kernels::project_unit_broadcast(residual, v.dev, v.alpha, rows, n_embd, stream);
        const cudaError_t status = cudaPeekAtLastError();
        if (status != cudaSuccess) {
            (void) cudaGetLastError();
            err = "cvec_residual (layer " + std::to_string(layer) + "): " + cudaGetErrorString(status);
            return false;
        }
    }
    return true;
}

}  // namespace strata::core
