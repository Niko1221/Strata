// metal/src/engine/kv_q4_host.cpp - the include/strata/kernels/kv_q4.hpp entry points on Metal: the standalone
// kernels of metal/kernels/kv_q4.metal, dispatched by name through the device runtime (strata/gpu/metal.hpp) on
// the caller's stream.  The *_e twins of that file read one argument struct from buffer 0; the structs are
// mirrored below - keep the field order and types in step (the static_asserts pin the sizes both sides compute).
// Mac has no paging (unified memory): the engine passes the identity page table the caller hands us unchanged.
#include "strata/gpu/metal.hpp"
#include "strata/kernels/kv_q4.hpp"
#include "strata/kernels/kv_stream.hpp"

#include "mtl_rt.h"

#include <cstdio>
#include <cstdlib>
#include <string>

namespace strata::kernels {
namespace {

// metal/kernels/kv_q4.metal's engine-side argument structs, field for field.
struct KvQ4Pools {
    uint8_t* k_q4;
    uint8_t* v_q4;
};
struct Fwht256Args {
    const float* src;
    float* dst;
    unsigned long n_rows;
    float scale;
};
struct KvAppendArgs {
    uint8_t* k_q4;
    uint8_t* v_q4;
    const int32_t* table;
    const int32_t* step;
    const float* kcur;
    const float* vcur;
    uint8_t* host_k;
    uint8_t* host_v;
    int step_stride;
    int planes;
    int kv_heads;
    int head_dim;
    int page_size;
};
struct KvAppendBatchArgs {
    uint8_t* k_q4;
    uint8_t* v_q4;
    const int32_t* table;
    const float* K;
    const float* V;
    uint8_t* host_k;
    uint8_t* host_v;
    uint8_t* stage_k;
    uint8_t* stage_v;
    unsigned long pos0;
    int kv_heads;
    int head_dim;
    int page_size;
    int is_v_grid;
};
struct KvGatherArgs {
    const uint8_t* k_q4;
    const uint8_t* v_q4;
    const int32_t* table;
    const int32_t* ids;
    const int32_t* step;
    uint16_t* k_scratch;
    uint16_t* v_scratch;
    int kv_heads;
    int head_dim;
    int page_size;
};
static_assert(sizeof(Fwht256Args) == 32 && sizeof(KvAppendArgs) == 88 && sizeof(KvAppendBatchArgs) == 96 &&
                  sizeof(KvGatherArgs) == 72,
              "the mirrored MSL argument structs of kv_q4.metal moved");

const char* kSourceName = "kv_q4.metal";
const char* kSource = "#include <metal_stdlib>\nusing namespace metal;\n";

MTL::ComputePipelineState* pipe(const char* function) {
    static const std::string src = std::string(kSource) +
                                   mtl::read_file(std::string(STRATA_METAL_KERNELS_DEFAULT) + "/" + kSourceName);
    std::string err;
    MTL::ComputePipelineState* p = gpu::metal::pipeline(kSourceName, src.c_str(), function, err);
    if (!p) {
        std::fprintf(stderr, "kv_q4: %s\n", err.c_str());
        std::exit(1);
    }
    return p;
}

void check(gpu::Error e, const char* what) {
    if (e != gpu::kSuccess) {
        std::fprintf(stderr, "kv_q4: %s: %s\n", what, gpu::error_string(e));
        std::exit(1);
    }
}

KvQ4Pools pools(const KvHostPools* p) { return p ? KvQ4Pools{p->k_q4, p->v_q4} : KvQ4Pools{}; }

void need_256(const QsaShapes& s, const char* what) {
    if (s.head_dim != 256) {
        std::fprintf(stderr, "%s: head_dim must be 256 (the Hadamard transform's size)\n", what);
        std::exit(1);
    }
}

}  // namespace

void fwht256_cuda(const float* src, float* dst, int64_t n_rows, void* stream) {
    if (n_rows <= 0) return;
    Fwht256Args a{src, dst, (unsigned long) n_rows, 1.0f / 16.0f};
    const gpu::metal::Grid groups{(uint32_t) ((n_rows + 3) / 4), 1, 1}, threads{32, 4, 1};
    check(gpu::metal::dispatch((gpu::Stream) stream, pipe("fwht256_e"), &a, sizeof a, groups, threads),
          "fwht256 launch");
}

void kv_append_q4_steps(uint8_t* k_q4, uint8_t* v_q4, const int32_t* page_table, const int32_t* step,
                        int step_stride, int n_steps, const float* kcur, const float* vcur, const QsaShapes& s,
                        void* stream, const KvHostPools* host) {
    if (n_steps <= 0) return;
    need_256(s, "kv_append_q4");
    const int planes = (v_q4 == nullptr || (v_q4 == k_q4 && vcur == kcur)) ? 1 : 2;
    const KvQ4Pools h = pools(host);
    KvAppendArgs a{k_q4,        v_q4,        page_table,  step,       kcur,       vcur,
                   h.k_q4,      h.v_q4,      step_stride, planes,     (int) s.n_head_kv,
                   (int) s.head_dim, (int) s.page_size};
    const gpu::metal::Grid groups{(uint32_t) s.n_head_kv, (uint32_t) (s.head_dim / QK4_0),
                                  (uint32_t) (n_steps * planes)},
        threads{32, 1, 1};
    check(gpu::metal::dispatch((gpu::Stream) stream, pipe("kv_append_q4_e"), &a, sizeof a, groups, threads),
          "kv_append_q4 launch");
}

void kv_append_q4_step(uint8_t* k_q4, uint8_t* v_q4, const int32_t* page_table, const int32_t* step,
                       const float* kcur, const float* vcur, const QsaShapes& s, void* stream,
                       const KvHostPools* host) {
    kv_append_q4_steps(k_q4, v_q4, page_table, step, 0, 1, kcur, vcur, s, stream, host);
}

void kv_append_q4(uint8_t* k_q4, uint8_t* v_q4, const int32_t* page_table, int64_t pos0, int64_t T, const float* K,
                  const float* V, const QsaShapes& s, void* stream, const KvHostPools* host,
                  const KvHostPools* stage) {
    if (T <= 0) return;
    need_256(s, "kv_append_q4");
    const KvQ4Pools h = pools(host), st = pools(stage);
    const gpu::metal::Grid groups{(uint32_t) T, (uint32_t) s.n_head_kv, (uint32_t) (s.head_dim / QK4_0)};
    const gpu::metal::Grid threads{32, 1, 1};
    for (int is_v = 0; is_v < 2; ++is_v) {
        KvAppendBatchArgs a{k_q4,        v_q4,        page_table, K,          V,
                            h.k_q4,      h.v_q4,      st.k_q4,    st.v_q4,    (unsigned long) pos0,
                            (int) s.n_head_kv, (int) s.head_dim, (int) s.page_size, is_v};
        check(gpu::metal::dispatch((gpu::Stream) stream, pipe("kv_append_q4_batch_e"), &a, sizeof a, groups,
                                   threads),
              "kv_append_q4 batch launch");
    }
}

void kv_gather_q4_step(const uint8_t* k_q4, const uint8_t* v_q4, const int32_t* page_table, const int32_t* ids,
                       const int32_t* step, int64_t max_ids, const QsaShapes& s, uint16_t* k_scratch,
                       uint16_t* v_scratch, void* stream) {
    if (max_ids <= 0) return;
    const int blocks_per_head = (int) (s.head_dim / QK4_0);
    const int64_t total_blocks = max_ids * s.n_head_kv * blocks_per_head;
    KvGatherArgs a{k_q4,        v_q4,       page_table, ids,         step,       k_scratch, v_scratch,
                   (int) s.n_head_kv, (int) s.head_dim, (int) s.page_size};
    const gpu::metal::Grid groups{(uint32_t) ((total_blocks + 3) / 4), 1, 1}, threads{32, 4, 1};
    check(gpu::metal::dispatch((gpu::Stream) stream, pipe("kv_gather_q4_e"), &a, sizeof a, groups, threads),
          "kv_gather_q4 launch");
}

}  // namespace strata::kernels
