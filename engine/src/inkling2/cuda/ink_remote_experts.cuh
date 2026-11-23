// Two-host Inkling expert execution. The native scheduler, routing, KV,
// ShortConv and image embeddings remain on the coordinator.
#pragma once

#include "ink_cuda.cuh"
#include "strata/tp/nccl_session.hpp"

#include <array>
#include <chrono>
#include <cstdint>

namespace strata::ink {

__global__ void k_ink_collect_remote_rows(float* out, const float* peer, const int32_t* routes) {
    const int slot = blockIdx.y;
    if ((routes[slot] & 1) == 0) return;
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col < kInkHidden) {
        const size_t i = size_t(slot) * kInkHidden + col;
        out[i] = peer[i];
    }
}

class InkRemoteExperts {
public:
    using Message = std::array<uint32_t, 3>; // rows (0 = stop), layer, exact decode
    InkRemoteExperts(tp::MpiRuntime& mpi, int rows)
        : mpi_(mpi), network_(mpi, std::chrono::seconds(90)), capacity_(rows) {
        INK_CK(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking));
        INK_CK(cudaEventCreateWithFlags(&inputs_, cudaEventDisableTiming));
        if (mpi_.rank() == 0)
            INK_CK(cudaMalloc(&result_, slot_values(rows) * sizeof(float)));
    }

    ~InkRemoteExperts() {
        network_.abort();
        cudaFree(result_);
        cudaEventDestroy(inputs_);
        cudaStreamDestroy(stream_);
    }

    template <typename Workspace>
    void submit(int layer, int rows, bool exact, float* input, Workspace& ws, cudaStream_t compute) try {
        Message msg{uint32_t(rows), uint32_t(layer), uint32_t(exact)};
        validate(msg);
        INK_CK(cudaEventRecord(inputs_, compute));
        INK_CK(cudaStreamWaitEvent(stream_, inputs_, 0));
        mpi_.broadcast_control(msg);
        transfer(input, ws, rows);
        network_.receive(result_, slot_values(rows), 1, stream_);
    } catch (const std::exception& e) { abort(e); }

    template <typename Workspace>
    void collect(int rows, Workspace& ws, cudaStream_t compute) try {
        network_.synchronize(stream_);
        const dim3 grid((kInkHidden + 255) / 256, rows * kInkTopK);
        k_ink_collect_remote_rows<<<grid, 256, 0, compute>>>(ws.y_pair, result_, ws.topk_idx);
        INK_CK(cudaGetLastError());
    } catch (const std::exception& e) { abort(e); }

    void stop() {
        Message msg{};
        mpi_.broadcast_control(msg);
    }

    template <typename Workspace, typename Model>
    void serve_peer(Model& model) try {
        Workspace ws{};
        ws.alloc(capacity_);
        float* input = dalloc<float>(size_t(capacity_) * kInkHidden);
        mpi_.minimum(1); // Both workspaces must exist before the coordinator announces READY.
        while (true) {
            Message msg{};
            mpi_.broadcast_control(msg);
            if (msg[0] == 0) break;
            validate(msg);
            const int rows = int(msg[0]), layer = int(msg[1]);
            transfer(input, ws, rows);
            // Unowned slots are ignored by the expert kernels. Clear them before transfer.
            INK_CK(cudaMemsetAsync(ws.y_pair, 0, slot_values(rows) * sizeof(float), stream_));
            launch_ink_moe_dispatch_only(stream_, model.d_sec4, model.d_experts[layer],
                model.moe_info[layer], input, rows, ws, msg[2] != 0);
            network_.send(ws.y_pair, slot_values(rows), 0, stream_);
            network_.synchronize(stream_);
        }
        cudaFree(input);
        ws.free_all();
    } catch (const std::exception& e) { abort(e); }

private:
    static size_t slot_values(int rows) { return size_t(rows) * kInkTopK * kInkHidden; }

    void validate(const Message& msg) {
        if (!msg[0] || msg[0] > uint32_t(capacity_) || msg[1] >= kInkPhysLayers || msg[2] > 1)
            mpi_.fail("invalid Inkling expert work message");
    }
    template <typename Workspace>
    void transfer(float* input, Workspace& ws, int rows) {
        network_.broadcast_bytes(input, size_t(rows) * kInkHidden * sizeof(float), stream_);
        network_.broadcast_bytes(ws.topk_idx, size_t(rows) * kInkTopK * sizeof(int32_t), stream_);
        network_.broadcast_bytes(ws.topk_w, size_t(rows) * kInkTopK * sizeof(float), stream_);
    }
    [[noreturn]] void abort(const std::exception& e) {
        network_.abort();
        mpi_.fail(e.what());
    }

    tp::MpiRuntime& mpi_;
    tp::NcclSession network_;
    int capacity_;
    cudaStream_t stream_ = nullptr;
    cudaEvent_t inputs_ = nullptr;
    float* result_ = nullptr;
};

} // namespace strata::ink
