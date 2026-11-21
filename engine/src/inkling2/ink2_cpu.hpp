// src/inkling2/ink2_cpu.hpp - Inkling-2's cold experts on the CPU (strata-ink2 --cpu-threads N).
//
// The decode graph hands the experts of a layer visit that are not in VRAM to the CPU through one doorbell per node
// (InkBell, in that node's memory, registered mapped): the GPU writes the visit's input x, the jobs and a sequence
// number into each; one pool of pinned threads per NUMA node computes the jobs whose expert lives on that node
// (gate_up, softcapped SwiGLU, down: the same math as the GPU kernels; fp32 activations, int8 opt-in) and writes each
// expert's output row and its ack into its own doorbell; the graph waits for both acks, then the MoE tail weights all
// eight outputs as usual.
//
// Experts live in host memory placed per node (ink2cpu::node_alloc), each expert whole on one node, and so do the
// doorbells and the pools' buffers: a node's threads only touch local memory.  On a two-socket box the inter-socket
// link is the slow path; only the GPU's traffic (doorbells, expert copies) crosses it.  STRATA_NUMA_POOL=0 turns the
// placement off: one pool over all cores, one doorbell, expert memory unplaced (--cpu-threads is then the total).
#pragma once
#include <cstddef>
#include <cstdint>

namespace ink2cpu {

constexpr int BELL_H = 2304, BELL_K = 8;
struct InkBell {   // one per node in its memory, registered mapped; GPU writes ring/job/x, the node's pool y/ack
    volatile unsigned ring;            // sequence number of the last visit with CPU jobs
    unsigned pad0[15];
    volatile unsigned ack[2];          // [node]: the last sequence number done (written in that node's bell)
    unsigned pad1[14];
    int layer, n;                      // the visit's layer, number of jobs
    int k[BELL_K], e[BELL_K];          // job j: top-k slot k[j], expert e[j]
    unsigned pad2[14];
    float x[BELL_H];                   // the visit's MoE input
    float y[BELL_K][BELL_H];           // job j's expert output (factor applied, not the router weight), in the bell
                                       // of its expert's node
};

struct Setup {
    int nl = 0, ne = 0, mx_first = 0, h = 0, ff = 0;
    const uint8_t* const* ex = nullptr;    // [nl * ne] host address of expert (l, e)'s slot
    const uint8_t* home = nullptr;          // [nl * ne] its node
    const float* f13 = nullptr;             // [nl * ne] NVFP4 per-expert factors (layers < mx_first)
    const float* f2 = nullptr;
    size_t nv_partA = 0, mx_partA = 0;      // offset of part B (w2 | s2) in a slot
    float gate_cap = 0, up_cap = 0;
    InkBell* bell[2] = {};                  // [node]
    int threads = 0;                        // per node
};

int nodes();                                // NUMA nodes with CPUs (1 when there is no NUMA information)
// bytes of anonymous memory on `node`, its pages touched by a thread pinned there (so they are placed there)
void* node_alloc(size_t bytes, int node);
void node_free(void* p, size_t bytes);
const char* isa();                          // "avx512" | "avx2" | nullptr (no usable kernel: AVX2 + FMA + F16C needed)

void start(const Setup& s);                 // starts the pools; they spin on the bell (and nap when it is idle)
void stop();

}  // namespace ink2cpu
