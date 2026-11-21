// src/inkling2/ink2_cpu.hpp - Inkling-2's cold experts on the CPU (strata-ink2 --cpu-threads N).
//
// The decode graph hands the experts of a layer visit that are not in VRAM to the CPU through a doorbell in mapped
// pinned memory (InkBell): the GPU writes the visit's input x, the jobs and a sequence number; one pool of pinned
// threads per NUMA node computes the jobs whose expert lives on that node (gate_up, softcapped SwiGLU, down: the same
// math as the GPU kernels, fp32 activations and sums) and writes each expert's output row and its node's ack; the
// graph waits for both acks, then the MoE tail weights all eight outputs as usual.
//
// Experts live in host memory placed per node (ink2cpu::node_alloc), each expert whole on one node, so a node's
// threads only read local memory (on a two-socket box the inter-socket link is the slow path).
#pragma once
#include <cstddef>
#include <cstdint>

namespace ink2cpu {

constexpr int BELL_H = 2304, BELL_K = 8;
struct InkBell {   // mapped pinned host memory; GPU writes ring/job/x, CPU pools write y/ack
    volatile unsigned ring;            // sequence number of the last visit with CPU jobs
    unsigned pad0[15];
    volatile unsigned ack[2];          // per node: the last sequence number done
    unsigned pad1[14];
    int layer, n;                      // the visit's layer, number of jobs
    int k[BELL_K], e[BELL_K];          // job j: top-k slot k[j], expert e[j]
    unsigned pad2[14];
    float x[BELL_H];                   // the visit's MoE input
    float y[BELL_K][BELL_H];           // job j's expert output (factor applied, not the router weight)
};

struct Setup {
    int nl = 0, ne = 0, mx_first = 0, h = 0, ff = 0;
    const uint8_t* const* ex = nullptr;    // [nl * ne] host address of expert (l, e)'s slot
    const uint8_t* home = nullptr;          // [nl * ne] its node
    const float* f13 = nullptr;             // [nl * ne] NVFP4 per-expert factors (layers < mx_first)
    const float* f2 = nullptr;
    size_t nv_partA = 0, mx_partA = 0;      // offset of part B (w2 | s2) in a slot
    float gate_cap = 0, up_cap = 0;
    InkBell* bell = nullptr;
    int threads = 0;                        // per node
};

int nodes();                                // NUMA nodes with CPUs (1 when there is no NUMA information)
// bytes of anonymous memory on `node`, its pages touched by a thread pinned there (so they are placed there)
void* node_alloc(size_t bytes, int node);
void node_free(void* p, size_t bytes);
const char* isa();                          // "avx512" | "avx2" | nullptr (no usable kernel: AVX2 + FMA + F16C needed)

void start(const Setup& s);                 // starts the pools; they spin on the bell (and nap when it is idle)
void stop();
// one job on the calling thread, single-threaded (tests: the CPU path against the GPU kernels)
void run_one(int l, int e, const float* x, float* y);

}  // namespace ink2cpu
