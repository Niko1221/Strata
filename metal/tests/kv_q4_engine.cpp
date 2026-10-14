// metal/tests/kv_q4_engine.cpp - the ENGINE dispatch path of the kv_q4 kernels, in tree: the five
// include/strata/kernels/kv_q4.hpp entry points (metal/src/engine/kv_q4_host.cpp), each driving its *_e twin of
// kv_q4.metal with ONE argument struct at buffer 0 whose pointer fields carry gpuAddress values.  What the
// kernel-path parity test cannot see, this pins bit-level: the twins' C++ -> MSL marshalling (the
// offset-pinned mirrors of kv_q4_host.cpp), the grids the host computes, and the planes=1 folding of a hybrid
// layer's call.  Not in the tests glob: it links the device runtime (strata_gpu_metal), which no globbed test
// links - CMake wires it next to runtime_hello.
#include "strata/gpu/runtime.hpp"
#include "strata/kernels/kv_q4.hpp"
#include "strata/kernels/kv_stream.hpp"

#include "kv_q4_ref.h"
#include "ref_quant.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace gpu = strata::gpu;
namespace k = strata::kernels;
using namespace kvq4ref;

int g_fail = 0;

template <class T> T* dalloc(size_t bytes) {
    T* p = nullptr;
    if (gpu::malloc_device(&p, bytes) != gpu::kSuccess) {
        std::printf("FAIL: malloc_device(%zu)\n", bytes);
        std::exit(1);
    }
    return p;
}

// whole-pool memcmp: the append twins either produce every block's 18 bytes or the pool is wrong somewhere
void cmp_pool(const char* what, const void* got, const std::vector<uint8_t>& ref) {
    const long bad = std::memcmp(got, ref.data(), ref.size()) != 0;
    std::printf("  %-38s %6ld 18-byte blocks differ of %zu  %s\n", what, bad, ref.size() / 18, bad ? "FAIL" : "PASS");
    g_fail += bad != 0;
}

k::QsaShapes shapes() {
    k::QsaShapes s;
    s.n_head = 24; s.n_head_kv = H; s.head_dim = D; s.n_rot = 64;
    s.idx_n_head = 4; s.idx_dim = 128; s.idx_block = 4; s.idx_top_k = 2048; s.page_size = P;
    return s;
}

int main() {
    gpu::DeviceProp prop;
    gpu::get_device_properties(&prop, 0);
    std::printf("device %s\n", prop.name);
    ref::Rng r(409);

    // ---- 1. fwht256_cuda
    {
        const int rows = 9;
        float* src = dalloc<float>(4 * (size_t) rows * 256);
        float* dst = dalloc<float>(4 * (size_t) rows * 256);
        std::vector<float> in(rows * 256);
        for (auto& f : in) f = r.sym() * 3.0f;
        std::memcpy(gpu::host_view(src), in.data(), in.size() * 4);
        k::fwht256_cuda(src, dst, rows, nullptr);
        gpu::device_synchronize();
        std::vector<float> ref(rows * 256);
        fwht256_ref(in.data(), ref.data(), rows);
        long bad = 0;
        for (int64_t i = 0; i < (int64_t) rows * 256; ++i)
            bad += std::memcmp((const uint8_t*) gpu::host_view(dst) + 4 * i, &ref[i], 4) != 0;
        std::printf("  %-38s %ld of %d floats differ (bits)     %s\n", "engine fwht256_cuda", bad, rows * 256,
                    bad ? "FAIL" : "PASS");
        g_fail += bad != 0;
    }

    // ---- 2. kv_append_q4_step + kv_gather_q4_step: the round trip (identity table, no host copy), with a tie
    // block and an all-zero group through the twins
    {
        const int pages = 3, cells = pages * P;
        const k::QsaShapes s = shapes();
        std::vector<int32_t> htable(pages);
        for (int i = 0; i < pages; ++i) htable[i] = i;   // the identity the engine passes on a Mac
        uint8_t* kpool = dalloc<uint8_t>((size_t) cells * H * ROW_BYTES);
        uint8_t* vpool = dalloc<uint8_t>((size_t) cells * H * ROW_BYTES);
        int32_t* table = dalloc<int32_t>(4 * pages);
        int32_t* step = dalloc<int32_t>(16);
        float* kcur = dalloc<float>(4 * H * D);
        float* vcur = dalloc<float>(4 * H * D);
        gpu::memset_device(kpool, 0, (size_t) cells * H * ROW_BYTES);
        gpu::memset_device(vpool, 0, (size_t) cells * H * ROW_BYTES);
        std::memcpy(gpu::host_view(table), htable.data(), 4 * pages);
        std::vector<float> kk(H * D), vv(H * D);
        for (auto& f : kk) f = r.sym() * 2.0f;
        for (auto& f : vv) f = r.sym();
        fwht256_ref(kk.data(), kk.data(), H);   // the contract's input is already rotated
        fwht256_ref(vv.data(), vv.data(), H);
        for (int i = 0; i < 32; ++i) kk[i] = (i & 1) ? -1.5f : 1.5f;   // a tie block, written into the rotated rows
        for (int i = 64; i < 96; ++i) vv[i] = 0.0f;                    // an all-zero group (d = -0.0, id = 0)
        std::memcpy(gpu::host_view(kcur), kk.data(), kk.size() * 4);
        std::memcpy(gpu::host_view(vcur), vv.data(), vv.size() * 4);
        const int32_t pos = P + 9;
        const int32_t st[4] = {pos, pos + 1, 0, 0};
        std::memcpy(gpu::host_view(step), st, 16);
        k::kv_append_q4_step(kpool, vpool, table, step, kcur, vcur, s, nullptr);
        gpu::device_synchronize();
        std::vector<uint8_t> rk((size_t) cells * H * ROW_BYTES, 0), rv = rk;
        append_cell_ref(kk.data(), vv.data(), htable, rk, rv, false, pos);
        cmp_pool("engine kv_append_q4_step (K)", gpu::host_view(kpool), rk);
        cmp_pool("engine kv_append_q4_step (V)", gpu::host_view(vpool), rv);

        // the gather half of the round trip: ids with repeats, fp16 scratch bits off the reference blocks
        const int max_ids = 33;
        int32_t* ids = dalloc<int32_t>(4 * max_ids);
        uint16_t* ksc = dalloc<uint16_t>(2 * (size_t) max_ids * H * D);
        uint16_t* vsc = dalloc<uint16_t>(2 * (size_t) max_ids * H * D);
        int32_t* iv = (int32_t*) gpu::host_view(ids);
        for (int i = 0; i < max_ids; ++i) iv[i] = pos;
        const int32_t gstep[4] = {0, 0, 0, max_ids};
        std::memcpy(gpu::host_view(step), gstep, 16);
        k::kv_gather_q4_step(kpool, vpool, table, ids, step, max_ids, s, ksc, vsc, nullptr);
        gpu::device_synchronize();
        long bad = 0;
        for (int i = 0; i < max_ids; ++i)
            for (int h = 0; h < H; ++h)
                for (int t = 0; t < D; ++t) {
                    const int64_t row = ((int64_t) htable[pos / P] * H + h) * P + pos % P;
                    const block_q4_0* kb = (const block_q4_0*) (rk.data() + row * ROW_BYTES) + t / 32;
                    const block_q4_0* vb = (const block_q4_0*) (rv.data() + row * ROW_BYTES) + t / 32;
                    const int j = t % 32;
                    const int kq = (j < 16) ? ((kb->qs[j] & 0x0F) - 8) : ((kb->qs[j - 16] >> 4) - 8);
                    const int vq = (j < 16) ? ((vb->qs[j] & 0x0F) - 8) : ((vb->qs[j - 16] >> 4) - 8);
                    const int64_t off = ((int64_t) i * H + h) * D + t;
                    bad += ((const uint16_t*) gpu::host_view(ksc))[off] != k::f16_from_f32((float) kq * k::f32_from_f16(kb->d));
                    bad += ((const uint16_t*) gpu::host_view(vsc))[off] != k::f16_from_f32((float) vq * k::f32_from_f16(vb->d));
                }
        std::printf("  %-38s %ld of %d scratch values differ  %s\n", "engine kv_gather_q4_step (fp16 bits)", bad,
                    max_ids * H * D, bad ? "FAIL" : "PASS");
        g_fail += bad != 0;
    }

    // ---- 3. kv_append_q4_steps, folded: a hybrid layer's call (src/kernels/kv_steps_parity.cpp:104) passes
    // k_q4 == v_q4 and kcur == vcur, the host folds the planes to 1 - grid.z is n_steps, the kernel reads kcur
    // only, and the one pool receives K's blocks
    {
        const int pages = 3, cells = pages * P, n_steps = 2;
        const k::QsaShapes s = shapes();
        std::vector<int32_t> htable(pages);
        for (int i = 0; i < pages; ++i) htable[i] = i;
        uint8_t* pool = dalloc<uint8_t>((size_t) cells * H * ROW_BYTES);
        int32_t* table = dalloc<int32_t>(4 * pages);
        int32_t* step = dalloc<int32_t>(4 * (size_t) k::kStepCount * n_steps);
        float* cur = dalloc<float>(4 * (size_t) n_steps * H * D);
        gpu::memset_device(pool, 0, (size_t) cells * H * ROW_BYTES);
        std::memcpy(gpu::host_view(table), htable.data(), 4 * pages);
        const std::vector<int32_t> poss{P + 3, 2 * P + 40};
        std::vector<float> kk((size_t) n_steps * H * D);
        for (int t = 0; t < n_steps; ++t) {
            for (size_t i = 0; i < (size_t) H * D; ++i) kk[(size_t) t * H * D + i] = r.sym() * 2.0f;
            fwht256_ref(kk.data() + (size_t) t * H * D, kk.data() + (size_t) t * H * D, H);
            const int32_t st[4] = {poss[t], poss[t] + 1, 0, 0};
            std::memcpy((int32_t*) gpu::host_view(step) + (size_t) t * k::kStepCount, st, 16);
        }
        for (int i = 0; i < 32; ++i) kk[i] = (i & 1) ? -1.5f : 1.5f;   // a tie block in step 0's rotated row
        std::memcpy(gpu::host_view(cur), kk.data(), kk.size() * 4);
        k::kv_append_q4_steps(pool, pool, table, step, k::kStepCount, n_steps, cur, cur, s, nullptr);
        gpu::device_synchronize();
        std::vector<uint8_t> rk((size_t) cells * H * ROW_BYTES, 0);
        for (int t = 0; t < n_steps; ++t)
            append_cell_ref(kk.data() + (size_t) t * H * D, nullptr, htable, rk, rk, false, poss[t]);
        cmp_pool("engine kv_append_q4_steps (folded)", gpu::host_view(pool), rk);
    }

    // ---- 4. kv_append_q4, the batch: T cells from pos0 into the pool, the host copy AND the stage pool
    {
        const int pages = 2, cells = pages * P, T = 13, pos0 = P - 5;   // crosses page 0 -> 1 inside one batch
        const k::QsaShapes s = shapes();
        std::vector<int32_t> htable(pages);
        for (int i = 0; i < pages; ++i) htable[i] = i;
        uint8_t* kpool = dalloc<uint8_t>((size_t) cells * H * ROW_BYTES);
        uint8_t* vpool = dalloc<uint8_t>((size_t) cells * H * ROW_BYTES);
        uint8_t* hpk = dalloc<uint8_t>((size_t) cells * H * ROW_BYTES);
        uint8_t* hpv = dalloc<uint8_t>((size_t) cells * H * ROW_BYTES);
        uint8_t* spk = dalloc<uint8_t>((size_t) cells * H * ROW_BYTES);
        uint8_t* spv = dalloc<uint8_t>((size_t) cells * H * ROW_BYTES);
        int32_t* table = dalloc<int32_t>(4 * pages);
        float* K = dalloc<float>(4 * (size_t) T * H * D);
        float* V = dalloc<float>(4 * (size_t) T * H * D);
        for (uint8_t* p : {kpool, vpool, hpk, hpv, spk, spv}) gpu::memset_device(p, 0, (size_t) cells * H * ROW_BYTES);
        std::memcpy(gpu::host_view(table), htable.data(), 4 * pages);
        std::vector<float> kk((size_t) T * H * D), vv((size_t) T * H * D);
        for (size_t i = 0; i < kk.size(); ++i) {
            kk[i] = r.sym() * 2.0f;
            vv[i] = r.sym();
        }
        for (int64_t row = 0; row < (int64_t) T * H; ++row) {
            fwht256_ref(kk.data() + row * D, kk.data() + row * D, 1);
            fwht256_ref(vv.data() + row * D, vv.data() + row * D, 1);
        }
        std::memcpy(gpu::host_view(K), kk.data(), kk.size() * 4);
        std::memcpy(gpu::host_view(V), vv.data(), vv.size() * 4);
        k::KvHostPools host{}, stage{};
        host.k_q4 = hpk; host.v_q4 = hpv;
        stage.k_q4 = spk; stage.v_q4 = spv;   // identity-layout rows over the whole pool: K and V each need a full-size buffer
        k::kv_append_q4(kpool, vpool, table, pos0, T, K, V, s, nullptr, &host, &stage);
        gpu::device_synchronize();
        std::vector<uint8_t> rk((size_t) cells * H * ROW_BYTES, 0), rv = rk, hk = rk, hv = rk, sk = rk, sv = rk;
        for (int t = 0; t < T; ++t) {
            const float* kc = kk.data() + (size_t) t * H * D, * vc = vv.data() + (size_t) t * H * D;
            append_cell_ref(kc, vc, htable, rk, rv, false, pos0 + t);
            append_cell_ref(kc, vc, htable, hk, hv, true, pos0 + t);
            append_cell_ref(kc, vc, htable, sk, sv, true, pos0 + t);
        }
        cmp_pool("engine kv_append_q4 batch (table)", gpu::host_view(kpool), rk);
        cmp_pool("engine kv_append_q4 batch V (table)", gpu::host_view(vpool), rv);
        cmp_pool("engine kv_append_q4 batch host K", gpu::host_view(hpk), hk);
        cmp_pool("engine kv_append_q4 batch host V", gpu::host_view(hpv), hv);
        cmp_pool("engine kv_append_q4 batch stage K", gpu::host_view(spk), sk);
        cmp_pool("engine kv_append_q4 batch stage V", gpu::host_view(spv), sv);
    }

    std::printf("%s\n", g_fail ? "SOME FAILED" : "ALL PASS");
    return g_fail ? 1 : 0;
}
