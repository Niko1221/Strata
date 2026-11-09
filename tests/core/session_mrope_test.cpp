#include "strata/core/session.hpp"
#include "strata/kernels/elementwise.hpp"
#include "strata/kernels/mrope.hpp"
#include "strata/kernels/native_rope.hpp"
#include "strata/kernels/native_qsa_indexer.hpp"
#include "strata/kernels/rope.hpp"
#include "strata/prefill/kernels.hpp"

#include <cuda_runtime.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

namespace {
void ck(cudaError_t e) {
    if (e != cudaSuccess) { std::fprintf(stderr, "%s\n", cudaGetErrorString(e)); std::exit(1); }
}
void require(bool ok, const std::string& why) {
    if (!ok) { std::fprintf(stderr, "%s\n", why.c_str()); std::exit(1); }
}
constexpr int cells = 8, hd = 128, heads = 2, tokens = 4, rows = tokens * heads;
constexpr double theta = 10000000.0;
// Literal layouts: a 2x2 image vs a 1x4 image, then text. Cell 4 is the first following text cell.
const int32_t layout_a[cells * 3] = {0,0,0, 0,0,1, 0,1,0, 0,1,1, 2,2,2, 3,3,3, 4,4,4, 5,5,5};
const int32_t layout_b[cells * 3] = {0,0,0, 0,0,1, 0,0,2, 0,0,3, 4,4,4, 5,5,5, 6,6,6, 7,7,7};
const int32_t text[cells * 3] = {0,0,0, 1,1,1, 2,2,2, 3,3,3, 4,4,4, 5,5,5, 6,6,6, 7,7,7};

// Independent double-precision reference, not build_rope_table/mrope_pos/rope_neox_pair.
int compare(const float* device, const std::vector<const int32_t*>& layouts,
            const std::vector<int>& positions, int nh, double magnitude, const char* label) {
    std::vector<float> got(positions.size() * hd);
    ck(cudaMemcpy(got.data(), device, got.size() * sizeof(float), cudaMemcpyDeviceToHost));
    int bad = 0;
    for (size_t r = 0; r < positions.size(); ++r) {
        const int32_t* tab = layouts[r / nh];
        for (int d = 0; d < hd; ++d) {
            double expected = magnitude;
            if (d < 64) {
                const int pair = d % 32;
                const int p = tab[positions[r] * 3 + pair % 3];
                const double angle = p * std::pow(theta, -double(pair) / 32.0);
                expected *= d < 32 ? std::cos(angle) - std::sin(angle) : std::sin(angle) + std::cos(angle);
            }
            if (!(std::fabs(got[r * hd + d] - expected) < 2e-5)) ++bad;
        }
    }
    if (bad) std::fprintf(stderr, "%s: %d wrong elements\n", label, bad);
    return bad;
}
}

int main() {
    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess || count == 0) return 77;
    using namespace strata::kernels;
    strata::core::SessionState a, b;
    a.max_cells = b.max_cells = cells;
    std::string err;
    require(strata::core::session_mrope_init(a, cells, err), err);
    require(strata::core::session_mrope_init(b, cells, err), err);
    require(a.mrope != b.mrope, "sessions share a rotary allocation");
    require(strata::core::session_mrope_upload(a, layout_a, cells, err), err);
    require(strata::core::session_mrope_upload(b, layout_b, cells, err), err);
    const auto original_b = b.mrope;

    cudaStream_t stream = nullptr;
    ck(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    float *x = nullptr, *out = nullptr, *gamma = nullptr, *ct = nullptr, *st = nullptr;
    int32_t* pos = nullptr;
    const int32_t **owner = nullptr, **h_owner = nullptr, **m_owner = nullptr;
    const size_t bytes = rows * hd * sizeof(float);
    ck(cudaMalloc((void**) &x, bytes)); ck(cudaMalloc((void**) &out, bytes));
    ck(cudaMalloc((void**) &gamma, hd * sizeof(float)));
    ck(cudaMalloc((void**) &pos, rows * sizeof(int32_t)));
    ck(cudaMalloc((void**) &owner, tokens * sizeof(int32_t*)));
    ck(cudaHostAlloc((void**) &h_owner, tokens * sizeof(int32_t*), cudaHostAllocMapped));
    ck(cudaHostGetDevicePointer((void**) &m_owner, h_owner, 0));
    std::vector<float> ones(rows * hd, 1.0f), trig(cells * 32), tsin(cells * 32);
    ck(cudaMemcpy(x, ones.data(), bytes, cudaMemcpyHostToDevice));
    ck(cudaMemcpy(gamma, ones.data(), hd * sizeof(float), cudaMemcpyHostToDevice));
    RopeScaling scaling;
    scaling.freq_base = theta;
    build_rope_table(64, scaling, cells, trig.data(), tsin.data());
    ck(cudaMalloc((void**) &ct, trig.size() * sizeof(float)));
    ck(cudaMalloc((void**) &st, tsin.size() * sizeof(float)));
    ck(cudaMemcpy(ct, trig.data(), trig.size() * sizeof(float), cudaMemcpyHostToDevice));
    ck(cudaMemcpy(st, tsin.data(), tsin.size() * sizeof(float), cudaMemcpyHostToDevice));
    std::vector<int> positions{2,2, 2,2, 4,4, 4,4};
    ck(cudaMemcpy(pos, positions.data(), rows * sizeof(int32_t), cudaMemcpyHostToDevice));

    int bad = 0;
    // A captured single-owner graph must ignore later host bindings and changes to another session.
    cudaGraph_t graph = nullptr;
    cudaGraphExec_t solo = nullptr;
    ck(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
    {
        MropeScope bind(a.mrope);
        native_rope_apply(x, out, rows, hd, 64, scaling, pos, stream);
    }
    ck(cudaStreamEndCapture(stream, &graph));
    ck(cudaGraphInstantiate(&solo, graph, 0)); ck(cudaGraphDestroy(graph));
    for (const int32_t* new_b : {text, layout_b}) {
        require(strata::core::session_mrope_upload(b, new_b, cells, err), err);
        require(b.mrope == original_b, "rotary upload changed captured address");
        mrope_table_set(b.mrope); // deliberately wrong legacy/current owner
        ck(cudaGraphLaunch(solo, stream)); ck(cudaStreamSynchronize(stream));
        bad += compare(out, {layout_a}, positions, rows, 1.0, "single-owner replay");
    }

    // Batched production kernels: two proposal rows for each owner, same cell indexes, swapped row order.
    // The owner array is staged by the same production copy used by Verifier::record_window.
    for (int path = 0; path < 3; ++path) {
        cudaGraphExec_t batch = nullptr;
        ck(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
        const MappedCopy owners{owner, m_owner, tokens * (int64_t) sizeof(int32_t*) / 4};
        copy_from_mapped_multi(&owners, 1, stream);
        {
            MropeScope bind(a.mrope, {owner, heads});
            if (path == 0) native_rope_apply(x, out, rows, hd, 64, scaling, pos, stream);
            else if (path == 1) rope_neox_apply(x, out, rows, hd, 64, ct, st, pos, stream);
            else native_qsa_rms_norm_rope(x, hd, gamma, out, rows, hd, 64, 1e-6f, scaling, pos, stream);
        }
        ck(cudaStreamEndCapture(stream, &graph));
        ck(cudaGraphInstantiate(&batch, graph, 0)); ck(cudaGraphDestroy(graph));
        for (int order = 0; order < 3; ++order) {
            const bool swapped = order == 1;
            const int32_t* b_layout = order == 2 ? text : layout_b;
            require(order == 2 ? strata::core::session_mrope_reset(b, err)
                               : strata::core::session_mrope_upload(b, b_layout, cells, err), err);
            // Repeated owner rows model the two-proposal-per-slot verifier layout.
            h_owner[0] = h_owner[1] = swapped ? b.mrope : a.mrope;
            h_owner[2] = h_owner[3] = swapped ? a.mrope : b.mrope;
            const std::vector<const int32_t*> ref = swapped
                ? std::vector<const int32_t*>{b_layout, b_layout, layout_a, layout_a}
                : std::vector<const int32_t*>{layout_a, layout_a, b_layout, b_layout};
            ck(cudaGraphLaunch(batch, stream)); ck(cudaStreamSynchronize(stream));
            bad += compare(out, ref, positions, heads, path == 2 ? 1.0 / std::sqrt(1.0 + 1e-6) : 1.0,
                           "batch owner/row replay");
        }
        ck(cudaGraphExecDestroy(batch));
    }

    // Prompt kernels and pooled-indexer kernels use their bound session, not the last global upload.
    for (auto* session : {&a, &b}) {
        const int32_t* layout = session == &a ? layout_a : layout_b;
        require(strata::core::session_mrope_upload(*session, layout, cells, err), err);
        MropeScope bind(session->mrope);
        ck(cudaMemcpyAsync(out, x, bytes, cudaMemcpyDeviceToDevice, stream));
        strata::prefill::rope(out, 4, heads, hd, heads * hd, 2, scaling, stream);
        ck(cudaStreamSynchronize(stream));
        const std::vector<int> prompt_pos{2,2, 3,3, 4,4, 5,5};
        bad += compare(out, {layout}, prompt_pos, rows, 1.0, "prompt rotary");
        strata::prefill::rms_rope_rows(x, heads * hd, hd, out, heads * hd, gamma, 4, heads, hd,
                                      1e-6f, 2, scaling, stream);
        ck(cudaStreamSynchronize(stream));
        bad += compare(out, {layout}, prompt_pos, rows, 1.0 / std::sqrt(1.0 + 1e-6), "prompt fused rotary");

        float *tail = nullptr, *dead = nullptr, *pooled = nullptr;
        int32_t *block_pos = nullptr, *steps = nullptr;
        ck(cudaMalloc((void**) &tail, 3 * hd * sizeof(float)));
        ck(cudaMalloc((void**) &dead, hd * sizeof(float)));
        ck(cudaMalloc((void**) &pooled, 3 * hd * sizeof(float)));
        ck(cudaMalloc((void**) &block_pos, sizeof(int32_t)));
        ck(cudaMalloc((void**) &steps, cells * sizeof(int32_t)));
        const int32_t index_steps[cells] = {0,1,2,3,4,5,6,7};
        ck(cudaMemcpy(steps, index_steps, sizeof index_steps, cudaMemcpyHostToDevice));
        auto shape = qsa_real_shapes(); shape.idx_dim = hd; shape.idx_block = 4; shape.n_rot = 64;
        const QsaIndexerBuffers buffers{tail, dead, pooled, block_pos};
        for (int index_path = 0; index_path < 3; ++index_path) {
            ck(cudaMemsetAsync(tail, 0, 3 * hd * sizeof(float), stream));
            if (index_path == 0)
                native_qsa_indexer_append_batch(x, cells, 0, 0, gamma, 1e-6f, buffers, shape, cells, scaling, stream);
            else if (index_path == 1)
                native_qsa_indexer_append_steps(x, steps, 1, cells, 0, gamma, 1e-6f, buffers, shape, cells, scaling, stream);
            else
                for (int c = 0; c < cells; ++c)
                    indexer_key_append(x + c * hd, steps + c, 0, gamma, 1e-6f, buffers, shape, ct, st, stream);
            ck(cudaStreamSynchronize(stream));
            bad += compare(pooled + hd, {layout}, {4}, 1, 1.0 / std::sqrt(1.0 + 1e-6), "indexer owner");
        }
        cudaFree(tail); cudaFree(dead); cudaFree(pooled); cudaFree(block_pos); cudaFree(steps);
    }

    // No session binding: the legacy table, or null identity, still owns the old path.
    mrope_table_set(nullptr);
    native_rope_apply(x, out, rows, hd, 64, scaling, pos, stream);
    ck(cudaStreamSynchronize(stream));
    bad += compare(out, {text}, positions, rows, 1.0, "legacy text identity");
    ck(cudaGraphExecDestroy(solo));
    strata::core::session_release(a); strata::core::session_release(b);
    cudaFree(x); cudaFree(out); cudaFree(gamma); cudaFree(ct); cudaFree(st); cudaFree(pos); cudaFree(owner);
    cudaFreeHost(h_owner); cudaStreamDestroy(stream);
    std::printf("session_mrope: %d failures\n", bad);
    return bad ? 1 : 0;
}
