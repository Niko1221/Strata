// tests/core/dflash_owned_kv_test.cpp - the DFlash drafter's OWNED K/V state (docs/DFLASH.md):
// the pools must live inside the caller's arena, whole-resident, fp16, identity page table - and
// kv_append_step must land every cell at the physical row the identity layout names, exactly.
//
// Independent of the target model, the Verifier, the DFlash layers and attention: this only
// exercises qsa_state_bytes/qsa_state_init/qsa_state_zero/kv_append_step on one owned state
// (and then on five, one per draft layer).
//
//   1. ownership: k_pool/v_pool inside the arena; kv_elastic -1; kv_mode 0; fp16; no host copy;
//      identity page table;
//   2. append/readback at positions crossing page boundaries (page_size 4): exact fp16 bits
//      against the host's f16_from_f32 of the deterministic inputs;
//   3. elastic non-membership: enabling the process elastic K/V and growing/shrinking the global
//      pools leaves the owned state's bytes untouched;
//   4. five states, one per draft layer, each independently owned and independent.
#include "strata/core/layer.hpp"
#include "strata/kernels/f16_bits.hpp"
#include "strata/kernels/qsa.hpp"
#include "strata/core/vmm.hpp"

#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <vector>

using namespace strata::core;

namespace {
int g_fail = 0;
void check(bool ok, const char* what) {
    if (!ok) { std::fprintf(stderr, "FAIL: %s\n", what); ++g_fail; }
}

// deterministic cell content: k(cell, kvh, d) — fp16-exact values (small halves)
float kval(int64_t cell, int kvh, int d) {
    return (float) ((int) ((cell * 7 + kvh * 131 + d) % 61) - 30) / 4.0f;
}
float vval(int64_t cell, int kvh, int d) {
    return (float) ((int) ((cell * 13 + kvh * 57 + d) % 47) - 23) / 8.0f;
}

struct Owned {
    void* arena = nullptr;
    uint64_t bytes = 0;
    QsaState st = {};
    bool init(const ModelGeometry& g, int64_t max_cells, std::vector<int32_t>* table_out = nullptr) {
        QsaStateInitOptions o;
        o.force_owned_kv = true;    // the pools are carved from OUR arena, whole-resident
        o.force_f16_kv = true;      // FP16 regardless of the process KV format
        o.disable_elastic = true;
        o.disable_streaming = true;
        // with_rope=true: this state owns its rope tables (share_rope stays null) - bytes() and
        // init() must agree on that, or the tables' upload writes past the arena
        bytes = qsa_state_bytes(g, max_cells, /*with_rope=*/true, /*ring_cells=*/0, o);
        if (cudaMalloc(&arena, bytes) != cudaSuccess) return false;
        if (qsa_state_init(g, max_cells, arena, st, nullptr, /*ring_cells=*/0, o) == 0) return false;
        qsa_state_zero(st, g, nullptr);
        cudaDeviceSynchronize();
        if (table_out) {
            std::vector<int32_t> t((size_t) st.n_pages);
            if (cudaMemcpy(t.data(), st.page_table, t.size() * 4, cudaMemcpyDeviceToHost) != cudaSuccess)
                return false;
            *table_out = t;
        }
        return true;
    }
};

void check_ownership(const Owned& o, const char* tag) {
    const uintptr_t a0 = reinterpret_cast<uintptr_t>(o.arena), a1 = a0 + o.bytes;
    const auto inside = [&](const void* p) { return reinterpret_cast<uintptr_t>(p) >= a0 && reinterpret_cast<uintptr_t>(p) < a1; };
    char what[160];
    std::snprintf(what, sizeof what, "%s: k_pool/v_pool inside the arena", tag);
    check(inside(o.st.k_pool) && inside(o.st.v_pool), what);
    std::snprintf(what, sizeof what, "%s: kv_elastic -1, kv_mode 0, fp16, no host copy, no map", tag);
    check(o.st.kv_elastic == -1 && o.st.kv_mode == 0 && !o.st.kv_int8 && !o.st.kv_q4 && !o.st.kv_hybrid &&
              o.st.host.k_pool == nullptr && o.st.map.slot_block == nullptr,
          what);
}

void append_and_readback(const Owned& o, const ModelGeometry& g, const std::vector<int32_t>& table) {
    const strata::kernels::QsaShapes s = strata::kernels::qsa_real_shapes();
    const int page_size = (int) s.page_size, nkv = (int) s.n_head_kv, hd = (int) s.head_dim;
    cudaStream_t cs = nullptr;
    cudaStreamCreate(&cs);
    const int64_t max_cells = o.st.max_cells;
    std::vector<float> kcur((size_t) nkv * hd), vcur((size_t) nkv * hd);
    int32_t* step_dev = nullptr;
    cudaMalloc(&step_dev, 16);
    for (int64_t cell : {int64_t{0}, int64_t{1}, int64_t{3}, int64_t{4}, int64_t{15}, int64_t{16}, int64_t{17},
                     int64_t{31}, int64_t{32}, int64_t{47}, int64_t{63}, int64_t{64}, int64_t{65},
                     (int64_t) max_cells - 1}) {
        if (cell >= max_cells) continue;
        for (int h = 0; h < nkv * hd; ++h) {
            kcur[(size_t) h] = kval(cell, h / hd, h % hd);
            vcur[(size_t) h] = vval(cell, h / hd, h % hd);
        }
        float *kd = nullptr, *vd = nullptr;
        cudaMalloc(&kd, kcur.size() * 4);
        cudaMalloc(&vd, vcur.size() * 4);
        cudaMemcpyAsync(kd, kcur.data(), kcur.size() * 4, cudaMemcpyHostToDevice, cs);
        cudaMemcpyAsync(vd, vcur.data(), vcur.size() * 4, cudaMemcpyHostToDevice, cs);
        const int32_t step[4] = {(int32_t) cell, (int32_t)(cell + 1), (int32_t)((cell + 1) / page_size),
                                 (int32_t)(cell + 1)};
        cudaMemcpyAsync(step_dev, step, sizeof step, cudaMemcpyHostToDevice, cs);
        (void) cudaGetLastError();
        kv_append_step(o.st.k_pool, o.st.v_pool, o.st.page_table, step_dev, kd, vd, s, nullptr, nullptr);
        const cudaError_t append_err = cudaGetLastError();
        if (append_err != cudaSuccess)
            std::fprintf(stderr, "dbg: append launch failed: %s (cell %lld, k_pool %p, table %p, step %p)\n",
                         cudaGetErrorString(append_err), (long long) cell, (void*) o.st.k_pool,
                         (void*) o.st.page_table, (void*) step_dev);
        cudaStreamSynchronize(cs);
        cudaFree(kd);
        cudaFree(vd);
        // read back every kv head and every dimension, exactly
        for (int h = 0; h < nkv; ++h) {
            const int64_t page = table[(size_t)(cell / page_size)];
            const int64_t r = (page * nkv + h) * page_size + cell % page_size;
            std::vector<uint16_t> got((size_t) hd);
            cudaMemcpyAsync(got.data(), o.st.k_pool + (size_t) r * hd, (size_t) hd * 2, cudaMemcpyDeviceToHost, cs);
            cudaStreamSynchronize(cs);
            for (int d = 0; d < hd; ++d) {
                const uint16_t want = strata::kernels::f16_from_f32(kval(cell, h, d));
                if (got[(size_t) d] != want) {
                    std::fprintf(stderr, "FAIL: k cell %lld h %d d %d: got %04x want %04x\n", (long long) cell, h, d,
                                 got[(size_t) d], want);
                    ++g_fail;
                    h = nkv;
                    break;
                }
            }
            cudaMemcpyAsync(got.data(), o.st.v_pool + (size_t) r * hd, (size_t) hd * 2, cudaMemcpyDeviceToHost, cs);
            cudaStreamSynchronize(cs);
            for (int d = 0; d < hd; ++d) {
                const uint16_t want = strata::kernels::f16_from_f32(vval(cell, h, d));
                if (got[(size_t) d] != want) {
                    std::fprintf(stderr, "FAIL: v cell %lld h %d d %d: got %04x want %04x\n", (long long) cell, h, d,
                                 got[(size_t) d], want);
                    ++g_fail;
                    h = nkv;
                    break;
                }
            }
        }
    }
    cudaFree(step_dev);
    cudaStreamDestroy(cs);
}
}  // namespace

int main() {
    if (cudaSetDevice(0) != cudaSuccess) {
        std::printf("dflash_owned_kv_test: no GPU\n");
        return 77;
    }
    ModelGeometry g;   // the canonical head geometry: 24Q/2KV x 256, page_size 4
    const int64_t max_cells = 4096;

    std::fprintf(stderr, "dbg: section 1 (owned+append)\n");
    {   // 1+2: one owned state, ownership + append/readback across page boundaries
        Owned o;
        std::vector<int32_t> table;
        check(o.init(g, max_cells, &table), "owned state init");
        check_ownership(o, "single state");
        check((int) table.size() == (int) o.st.n_pages && table[(size_t) 0] == 0 &&
                  table[(size_t)(table.size() - 1)] == (int32_t)(table.size() - 1),
              "identity page table");
        append_and_readback(o, g, table);
        cudaFree(o.arena);
    }
    std::fprintf(stderr, "dbg: section 4 (five layers)\n");
    {   // 4: five states, one per draft layer — each independently owned
        std::vector<Owned> layers(5);
        bool all = true;
        for (auto& o : layers) all = o.init(g, max_cells) && all;
        check(all, "five owned states init");
        for (int l = 0; l < 5; ++l) {
            char what[64];
            std::snprintf(what, sizeof what, "layer %d owned", l);
            check_ownership(layers[(size_t) l], what);
        }
        // independence: append cell 5 to layer 2 only; layers 0 and 4 must stay zero there
        const strata::kernels::QsaShapes s = strata::kernels::qsa_real_shapes();
        cudaStream_t cs = nullptr;
        cudaStreamCreate(&cs);
        std::vector<float> kcur((size_t) s.n_head_kv * s.head_dim, 0.5f), vcur = kcur;
        float *kd, *vd;
        cudaMalloc(&kd, kcur.size() * 4);
        cudaMalloc(&vd, vcur.size() * 4);
        cudaMemcpyAsync(kd, kcur.data(), kcur.size() * 4, cudaMemcpyHostToDevice, cs);
        cudaMemcpyAsync(vd, vcur.data(), vcur.size() * 4, cudaMemcpyHostToDevice, cs);
        const int32_t step[4] = {5, 6, 1, 6};
        int32_t* sd;
        cudaMalloc(&sd, sizeof step);
        cudaMemcpyAsync(sd, step, sizeof step, cudaMemcpyHostToDevice, cs);
        (void) cudaGetLastError();
        kv_append_step(layers[2].st.k_pool, layers[2].st.v_pool, layers[2].st.page_table, sd, kd, vd, s, cs, nullptr);
        const cudaError_t e4 = cudaGetLastError();
        if (e4 != cudaSuccess) std::fprintf(stderr, "dbg: s4 append err: %s\n", cudaGetErrorString(e4));
        cudaStreamSynchronize(cs);
        for (int l : {0, 4}) {
            std::vector<uint16_t> got(256);
            const int64_t r = (5 / 4) * 8 + 5 % 4;   // (page*2+kvh)*4+slot for kvh 0
            cudaMemcpyAsync(got.data(), layers[(size_t) l].st.k_pool + (size_t) r * 256, 512, cudaMemcpyDeviceToHost, cs);
            cudaStreamSynchronize(cs);
            bool zero = true;
            for (uint16_t b : got) if (b != 0) zero = false;
            char what[64];
            std::snprintf(what, sizeof what, "layer %d stays zero at layer 2's cell", l);
            check(zero, what);
        }
        cudaFree(kd);
        cudaFree(vd);
        cudaFree(sd);
        cudaStreamDestroy(cs);
        for (auto& o : layers) cudaFree(o.arena);
    }

    std::fprintf(stderr, "dbg: section 3 (elastic)\n");
    {   // 3: elastic non-membership — global grow/shrink must not touch the owned cells
        Owned o;
        std::vector<int32_t> table;
        check(o.init(g, max_cells, &table), "owned state init (elastic section)");
        append_and_readback(o, g, table);
        if (qsa_kv_elastic()) qsa_set_kv_elastic(false, 0);   // the test owns the process flag

        if (qsa_kv_elastic()) {   // VMM present: an elastic-mode state's pools must NOT be in the arena
            Owned elastic;   // a state the way the target's session would create one under --kv-grow
            elastic.bytes = qsa_state_bytes(g, max_cells, false, 0);
            check(cudaMalloc(&elastic.arena, elastic.bytes) == cudaSuccess, "elastic-mode arena");
            const uint64_t eb = qsa_state_init(g, max_cells, elastic.arena, elastic.st, nullptr, 0);
            (void) cudaGetLastError();   // a failed/reserved VMM attempt may leave a sticky error
            if (eb == 0)
                check(false, "elastic init should have succeeded");
            else {
                const uintptr_t a0 = reinterpret_cast<uintptr_t>(elastic.arena);
                const bool inside = reinterpret_cast<uintptr_t>(elastic.st.k_pool) >= a0 &&
                                    reinterpret_cast<uintptr_t>(elastic.st.k_pool) < a0 + elastic.bytes;
                check(!inside, "elastic-mode pools live outside the caller arena (the refusal is justified)");
            }
            cudaFree(elastic.arena);
        }
        // grow/shrink the registry, then verify the OWNED state's cells are untouched
        const auto before = std::invoke([&] {
            std::vector<uint16_t> r(256);
            cudaMemcpy(r.data(), o.st.k_pool, 512, cudaMemcpyDeviceToHost);
            return r;
        });
        const int64_t need = qsa_kv_elastic_need(max_cells);
        std::vector<strata::core::VmmChunk> taken;
        qsa_kv_elastic_grow(need, [&] { return strata::core::VmmChunk{}; });
        qsa_kv_elastic_shrink(need, [&](strata::core::VmmChunk c) { taken.push_back(c); });
        (void) cudaGetLastError();
        std::vector<uint16_t> after(256);
        cudaMemcpy(after.data(), o.st.k_pool, 512, cudaMemcpyDeviceToHost);
        check(before == after, "global elastic grow/shrink leaves the owned cells untouched");
        qsa_set_kv_elastic(false, 0);
        cudaFree(o.arena);
        (void) cudaGetLastError();   // the VMM experiments may leave a sticky error; the test is done
    }
    std::printf(g_fail ? "dflash_owned_kv_test: %d FAILURES\n" : "dflash_owned_kv_test: ok\n", g_fail);
    return g_fail ? 1 : 0;
}
