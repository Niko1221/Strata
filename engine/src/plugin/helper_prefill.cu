// helper_prefill.cu — strata-mi50, 2026-10-27
// The helper GPU (MI50, gfx906, HIP 5.x in the plugin) computing a share of the prompt's experts while the engine's own GPU computes the
// rest (STRATA_PREFILL_HELPER, plugin ABI 5, PIANO_MI50_PROMPT_2026-10-27.md).  Per layer: fp16 activations in, the marked experts' rows
// grouped by expert (16 experts a launch, as the engine's peer path does), llama.cpp's MMQ for gate/up and down from the blobs of this
// tier's VRAM cache, SwiGLU between them, and one weighted sum per token out (fp16).  Everything is queued on one stream of its own.
#include "helper_prefill.hpp"

#include "strata/core/remote_experts.hpp"
#include "strata/kernels/cpu/expert_layout.hpp"
#include "strata/prefill/moe_mmq.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>

namespace mmq = strata::prefill::mmq;

namespace strata::plugin {
namespace {
constexpr int64_t N = strata::kernels::cpu::H;    // n_embd
constexpr int64_t FF = strata::kernels::cpu::FF;  // expert hidden (640)
constexpr int GROUP = 16;                          // experts a launch (mmq::kGatherGroupMax)
constexpr size_t TAIL = 4096;                      // zeroed bytes after a group's last expert (the MMQ over-read)

// sum[t] += sum over the token's rows r of w[r] * dm[(r - r0)], in the order of the list (deterministic)
__global__ void gather_add_kernel(float* __restrict__ sum, const float* __restrict__ dm, int64_t r0, const float* __restrict__ roww,
                                  const int32_t* __restrict__ tok, const int32_t* __restrict__ start, const int32_t* __restrict__ list) {
    const int64_t d = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (d >= N) return;
    const int b = blockIdx.y;
    float* o = sum + (int64_t) tok[b] * N + d;
    float s = *o;
    for (int i = start[b]; i < start[b + 1]; ++i) {
        const int r = list[i];
        s = fmaf(roww[r], dm[(int64_t) (r - r0) * N + d], s);
    }
    *o = s;
}

__global__ void to_f16_kernel(const float* __restrict__ x, unsigned short* __restrict__ y, int64_t n) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float v = x[i];
    v = fminf(fmaxf(v, -65504.0f), 65504.0f);   // saturate (the engine's sums_to_f16 does the same)
    y[i] = __half_as_ushort(__float2half_rn(v));
}

bool ck(cudaError_t e, const char* what, std::string& err) {
    if (e == cudaSuccess) return true;
    err = std::string("helper prefill: ") + what + ": " + cudaGetErrorString(e);
    (void) cudaGetLastError();
    return false;
}
}  // namespace

struct PrefillWs {
    int device = 0;
    int64_t t_max = 0, cap = 0, n_layers = 0;
    cudaStream_t s = nullptr;
    cudaEvent_t done = nullptr;
    mmq::Context* ctx = nullptr;
    uint16_t* h_x16 = nullptr;      // pinned: the engine's activations (it may write them here directly)
    uint16_t* h_sum16 = nullptr;    // pinned: the sums coming back
    int32_t* h_meta = nullptr;      // pinned: the layer's tables
    size_t meta_ints = 0;
    uint16_t* d_x16 = nullptr;      // the activations in; the same bytes hold the sums out (the conversion is done by then)
    float* d_x = nullptr;           // T x N fp32
    float* d_sum = nullptr;         // T x N fp32
    int32_t* d_meta = nullptr;
    int32_t* d_ident = nullptr;
    float *d_gu = nullptr, *d_h = nullptr, *d_dm = nullptr;
    void *d_xq = nullptr, *d_hq = nullptr;
    uint8_t *d_grp_gu = nullptr, *d_grp_d = nullptr;
    size_t gub_max = 0, db_max = 0;
    int64_t pending_t = -1;         // tokens of the queued layer (-1: none)
    std::vector<uint8_t> layer_ok;
    // per-layer scratch (host)
    std::vector<int32_t> cnt, off, fill, slot, gid, lasttok;
};

void prefill_destroy(PrefillWs* w) {
    if (w == nullptr) return;
    int prev = 0;
    cudaGetDevice(&prev);
    cudaSetDevice(w->device);
    if (w->s) cudaStreamSynchronize(w->s);
    delete w->ctx;
    for (void* p : {(void*) w->d_x16, (void*) w->d_x, (void*) w->d_sum, (void*) w->d_meta, (void*) w->d_ident, (void*) w->d_gu,
                    (void*) w->d_h, (void*) w->d_dm, w->d_xq, w->d_hq, (void*) w->d_grp_gu, (void*) w->d_grp_d})
        if (p) cudaFree(p);
    for (void* p : {(void*) w->h_x16, (void*) w->h_sum16, (void*) w->h_meta})
        if (p) cudaFreeHost(p);
    if (w->done) cudaEventDestroy(w->done);
    if (w->s) cudaStreamDestroy(w->s);
    cudaSetDevice(prev);
    delete w;
}

bool prefill_reserve(PrefillWs*& wsp, int device, int64_t t_max, std::vector<uint8_t>& layer_ok, uint16_t** x16, uint16_t** sum16,
                     std::string& err) {
    if (wsp != nullptr && wsp->t_max >= t_max) {   // already big enough: only report the layers and the buffers
        for (size_t l = 0; l < layer_ok.size(); ++l) layer_ok[l] = l < wsp->layer_ok.size() ? wsp->layer_ok[l] : 0;
        *x16 = wsp->h_x16;
        *sum16 = wsp->h_sum16;
        return true;
    }
    prefill_destroy(wsp);
    wsp = nullptr;
    if (!mmq::built()) { err = "helper prefill: this plugin build has no MMQ"; return false; }
    const auto& lay = strata::kernels::cpu::expert_layout();
    if (!lay.native) { err = "helper prefill: only native packs (the Q2_0 pack has no helper path)"; return false; }
    auto w = new PrefillWs();
    w->device = device;
    w->t_max = t_max;
    w->cap = t_max;   // rows of one launch group: at least one expert's rows (an expert takes at most t_max)
    w->n_layers = (int64_t) layer_ok.size();
    int prev = 0;
    cudaGetDevice(&prev);
    struct Restore { int p; ~Restore() { cudaSetDevice(p); } } restore{prev};
    if (!ck(cudaSetDevice(device), "cudaSetDevice", err)) { delete w; return false; }
    w->layer_ok.assign((size_t) w->n_layers, 0);
    int ok_layers = 0;
    for (int64_t l = 0; l < w->n_layers && l < (int64_t) lay.fmt.size(); ++l) {
        const auto& f = lay.fmt[(size_t) l];
        // same cases the engine's MMQ path takes: types MMQ covers and fits, no Hadamard fold, no NVFP4 tail
        if (f.tail_off != 0 || f.had_block != 0 || f.had_gu || f.had_d) continue;
        if (!mmq::supported(f.gu_type) || !mmq::supported(f.d_type)) continue;
        if (!mmq::fits(f.gu_type, 1280) || !mmq::fits(f.d_type, N)) continue;
        if (f.n_embd != N || f.n_ff != FF) continue;
        w->layer_ok[(size_t) l] = 1;
        ++ok_layers;
        w->gub_max = std::max(w->gub_max, mmq::matrix_bytes(f.gu_type, 1280, N));
        w->db_max = std::max(w->db_max, mmq::matrix_bytes(f.d_type, N, FF));
    }
    if (ok_layers == 0) { err = "helper prefill: no layer of this pack is covered"; delete w; return false; }
    auto fail = [&](const std::string& what) {
        if (err.empty()) err = "helper prefill: " + what;
        prefill_destroy(w);
        return false;
    };
    const size_t T = (size_t) t_max, G = (size_t) w->cap;
    w->meta_ints = 5 * T * 10 + 16384;
    if (!ck(cudaStreamCreateWithFlags(&w->s, cudaStreamNonBlocking), "stream", err)) return fail("");
    if (!ck(cudaEventCreateWithFlags(&w->done, cudaEventDisableTiming), "event", err)) return fail("");
    if (!ck(cudaHostAlloc((void**) &w->h_x16, T * N * 2, cudaHostAllocDefault), "pinned x", err)) return fail("");
    if (!ck(cudaHostAlloc((void**) &w->h_sum16, T * N * 2, cudaHostAllocDefault), "pinned sums", err)) return fail("");
    if (!ck(cudaHostAlloc((void**) &w->h_meta, w->meta_ints * 4, cudaHostAllocDefault), "pinned tables", err)) return fail("");
    size_t free_b = 0, total_b = 0;
    cudaMemGetInfo(&free_b, &total_b);
    const size_t need = T * N * 2 + 2 * T * N * 4 + w->meta_ints * 4 + G * 4 + G * 1280 * 4 + G * FF * 4 + G * N * 4 +
                        mmq::q8_bytes((int64_t) G, N) + mmq::q8_bytes((int64_t) G, FF) + GROUP * w->gub_max + GROUP * w->db_max +
                        2 * TAIL + (64u << 20);   // + room for the MMQ launches' scratch
    if (need > free_b) {
        char b[200];
        std::snprintf(b, sizeof b, "the work space needs %.0f MiB and the helper has %.0f MiB free", (double) need / 1048576.0,
                      (double) free_b / 1048576.0);
        err = std::string("helper prefill: ") + b;
        prefill_destroy(w);
        return false;
    }
    if (!ck(cudaMalloc((void**) &w->d_x16, T * N * 2), "x16", err) || !ck(cudaMalloc((void**) &w->d_x, T * N * 4), "x", err) ||
        !ck(cudaMalloc((void**) &w->d_sum, T * N * 4), "sum", err) || !ck(cudaMalloc((void**) &w->d_meta, w->meta_ints * 4), "tables", err) ||
        !ck(cudaMalloc((void**) &w->d_ident, G * 4), "ident", err) || !ck(cudaMalloc((void**) &w->d_gu, G * 1280 * 4), "gu", err) ||
        !ck(cudaMalloc((void**) &w->d_h, G * FF * 4), "h", err) || !ck(cudaMalloc((void**) &w->d_dm, G * N * 4), "dm", err) ||
        !ck(cudaMalloc(&w->d_xq, mmq::q8_bytes((int64_t) G, N)), "xq", err) ||
        !ck(cudaMalloc(&w->d_hq, mmq::q8_bytes((int64_t) G, FF)), "hq", err) ||
        !ck(cudaMalloc((void**) &w->d_grp_gu, GROUP * w->gub_max + TAIL), "group gate/up", err) ||
        !ck(cudaMalloc((void**) &w->d_grp_d, GROUP * w->db_max + TAIL), "group down", err))
        return fail("");
    mmq::iota(w->d_ident, (int64_t) G, w->s);
    w->ctx = new mmq::Context();
    if (!ck(cudaStreamSynchronize(w->s), "init sync", err)) return fail("");
    const int ne = (int) strata::kernels::cpu::expert_layout().n_expert;
    w->cnt.assign((size_t) ne, 0);
    w->off.assign((size_t) ne + 1, 0);
    w->fill.assign((size_t) ne, 0);
    w->gid.assign((size_t) ne, -1);
    w->lasttok.assign((size_t) ne, -1);
    for (size_t l = 0; l < layer_ok.size(); ++l) layer_ok[l] = w->layer_ok[l];
    *x16 = w->h_x16;
    *sum16 = w->h_sum16;
    wsp = w;
    return true;
}

bool prefill_begin(PrefillWs* w, const strata::core::RemoteExperts& remote, int64_t layer, int64_t t, int64_t k, const uint16_t* x16,
                   const int32_t* ids, const float* wts, const uint8_t* take, std::string& err) {
    if (w == nullptr || w->s == nullptr) { err = "helper prefill: begin before reserve"; return false; }
    if (w->pending_t >= 0) { err = "helper prefill: begin with a layer still queued"; return false; }
    if (t < 1 || t > w->t_max || k < 1 || k > 10 || layer < 0 || layer >= w->n_layers || !w->layer_ok[(size_t) layer]) {
        err = "helper prefill: begin outside what was reserved (layer " + std::to_string(layer) + ", " + std::to_string(t) + " tokens)";
        return false;
    }
    const auto& lay = strata::kernels::cpu::expert_layout();
    const auto& f = lay.fmt[(size_t) layer];
    const int ne = (int) lay.n_expert;
    const int64_t n = t * k;
    // ---- host: which rows go where
    std::fill(w->cnt.begin(), w->cnt.end(), 0);
    for (int64_t i = 0; i < n; ++i) {
        const int32_t e = ids[i];
        if ((unsigned) e < (unsigned) ne && take[e]) ++w->cnt[(size_t) e];
    }
    struct Grp { int e0 = 0, e1 = 0; int64_t r0 = 0, nr = 0, maxr = 0; };
    std::vector<int32_t> order;   // taken experts with rows, ascending
    for (int32_t e = 0; e < ne; ++e)
        if (w->cnt[(size_t) e] > 0) {
            if (remote.slot_ptr(layer, e) == nullptr) {
                err = "helper prefill: expert " + std::to_string(e) + " of layer " + std::to_string(layer) + " is not in the helper's cache";
                return false;
            }
            order.push_back(e);
        }
    int64_t R = 0;
    for (int32_t e : order) { w->off[(size_t) e] = (int32_t) R; R += w->cnt[(size_t) e]; }
    std::vector<Grp> groups;
    for (size_t j = 0; j < order.size(); ++j) {
        const int64_t c = w->cnt[(size_t) order[j]];
        if (groups.empty() || groups.back().e1 - groups.back().e0 >= GROUP || groups.back().nr + c > w->cap) {
            Grp g;
            g.e0 = (int) j;
            g.e1 = (int) j;
            g.r0 = w->off[(size_t) order[j]];
            groups.push_back(g);
        }
        Grp& g = groups.back();
        g.e1 = (int) j + 1;
        g.nr += c;
        g.maxr = std::max(g.maxr, c);
        w->gid[(size_t) order[j]] = (int) groups.size() - 1;
    }
    std::vector<std::vector<int32_t>> tok(groups.size()), start(groups.size()), list(groups.size());
    for (size_t g = 0; g < groups.size(); ++g) list[g].reserve((size_t) groups[g].nr);
    std::vector<int32_t> last(groups.size(), -1);
    for (int32_t e : order) w->fill[(size_t) e] = w->off[(size_t) e];
    w->slot.assign((size_t) n, -1);
    // meta layout: [src R][roww R][per group: bounds n+1 | tok nt | start nt+1 | list nr]
    int32_t* m = w->h_meta;
    int32_t* src = m;
    float* roww = reinterpret_cast<float*>(m + R);
    for (int64_t i = 0; i < n; ++i) {
        const int32_t e = ids[i];
        if ((unsigned) e >= (unsigned) ne || !take[e] || w->cnt[(size_t) e] == 0) continue;
        const int32_t r = w->fill[(size_t) e]++;
        w->slot[(size_t) i] = r;
        src[r] = (int32_t) (i / k);
        roww[r] = wts[i];
        const int g = w->gid[(size_t) e];
        const int32_t tt = (int32_t) (i / k);
        if (last[(size_t) g] != tt) {
            last[(size_t) g] = tt;
            tok[(size_t) g].push_back(tt);
            start[(size_t) g].push_back((int32_t) list[(size_t) g].size());
        }
        list[(size_t) g].push_back(r);
    }
    struct Off { size_t bounds, tok, start, list; };
    std::vector<Off> offs(groups.size());
    size_t at = (size_t) (2 * R);
    for (size_t g = 0; g < groups.size(); ++g) {
        const int nexp = groups[g].e1 - groups[g].e0;
        const size_t nt = tok[g].size();
        start[g].push_back((int32_t) list[g].size());
        offs[g] = {at, at + (size_t) nexp + 1, at + (size_t) nexp + 1 + nt, at + (size_t) nexp + 1 + 2 * nt + 1};
        at = offs[g].list + (size_t) groups[g].nr;
    }
    if (at > w->meta_ints) { err = "helper prefill: the layer's tables overflow"; return false; }
    for (size_t g = 0; g < groups.size(); ++g) {
        const Grp& gr = groups[g];
        for (int j = gr.e0; j < gr.e1; ++j) m[offs[g].bounds + (size_t) (j - gr.e0)] = w->off[(size_t) order[(size_t) j]] - (int32_t) gr.r0;
        m[offs[g].bounds + (size_t) (gr.e1 - gr.e0)] = (int32_t) gr.nr;
        std::memcpy(m + offs[g].tok, tok[g].data(), tok[g].size() * 4);
        std::memcpy(m + offs[g].start, start[g].data(), start[g].size() * 4);
        std::memcpy(m + offs[g].list, list[g].data(), list[g].size() * 4);
    }
    for (int32_t e : order) w->gid[(size_t) e] = -1;
    // ---- device
    int prev = 0;
    cudaGetDevice(&prev);
    struct Restore { int p; ~Restore() { cudaSetDevice(p); } } restore{prev};
    if (!ck(cudaSetDevice(w->device), "cudaSetDevice", err)) return false;
    const cudaStream_t s = w->s;
    if (x16 != w->h_x16) std::memcpy(w->h_x16, x16, (size_t) t * N * 2);
    if (!ck(cudaMemcpyAsync(w->d_x16, w->h_x16, (size_t) t * N * 2, cudaMemcpyHostToDevice, s), "x copy", err)) return false;
    if (at > 0 && !ck(cudaMemcpyAsync(w->d_meta, m, at * 4, cudaMemcpyHostToDevice, s), "tables copy", err)) return false;
    mmq::f16_to_f32(w->d_x16, w->d_x, t * N, s);
    cudaMemsetAsync(w->d_sum, 0, (size_t) t * N * 4, s);
    const size_t gub = mmq::matrix_bytes(f.gu_type, 1280, N), db = mmq::matrix_bytes(f.d_type, N, FF);
    const int32_t* d_src = w->d_meta;
    const float* d_roww = reinterpret_cast<const float*>(w->d_meta + R);
    for (size_t g = 0; g < groups.size(); ++g) {
        const Grp& gr = groups[g];
        const int nexp = gr.e1 - gr.e0;
        for (int j = 0; j < nexp; ++j) {
            const uint8_t* bd = remote.slot_ptr(layer, order[(size_t) (gr.e0 + j)]);
            mmq::gather_native(bd, bd + f.up_off, gub / 2, bd + f.down_off, db, w->d_grp_gu + (size_t) j * gub,
                               w->d_grp_d + (size_t) j * db, s);
        }
        cudaMemsetAsync(w->d_grp_gu + (size_t) nexp * gub, 0, TAIL, s);
        cudaMemsetAsync(w->d_grp_d + (size_t) nexp * db, 0, TAIL, s);
        const int32_t* d_bounds = w->d_meta + offs[g].bounds;
        mmq::quantize(w->d_x, d_src + gr.r0, w->d_xq, f.gu_type, N, N, gr.nr, s);
        mmq::Product gu;
        gu.w = w->d_grp_gu; gu.type = f.gu_type; gu.w_rows = 1280; gu.w_cols = N; gu.expert_bytes = gub;
        gu.n = nexp; gu.xq = w->d_xq; gu.bounds = d_bounds; gu.ids = w->d_ident;
        gu.total_rows = gr.nr; gu.max_rows = gr.maxr; gu.dst = w->d_gu; gu.ld_dst = 1280;
        w->ctx->run(gu, s);
        mmq::swiglu(w->d_gu, w->d_h, gr.nr, FF, !lay.native, s);
        mmq::quantize(w->d_h, nullptr, w->d_hq, f.d_type, FF, FF, gr.nr, s);
        mmq::Product dn;
        dn.w = w->d_grp_d; dn.type = f.d_type; dn.w_rows = N; dn.w_cols = FF; dn.expert_bytes = db;
        dn.n = nexp; dn.xq = w->d_hq; dn.bounds = d_bounds; dn.ids = w->d_ident;
        dn.total_rows = gr.nr; dn.max_rows = gr.maxr; dn.dst = w->d_dm; dn.ld_dst = N;
        w->ctx->run(dn, s);
        const int64_t nt = (int64_t) tok[g].size();
        gather_add_kernel<<<dim3((unsigned) ((N + 255) / 256), (unsigned) nt), 256, 0, s>>>(
            w->d_sum, w->d_dm, gr.r0, d_roww, w->d_meta + offs[g].tok, w->d_meta + offs[g].start, w->d_meta + offs[g].list);
    }
    // the sums as fp16, over the activations' bytes (the conversion above is done: same stream), and back to the host
    to_f16_kernel<<<(unsigned) ((t * N + 255) / 256), 256, 0, s>>>(w->d_sum, reinterpret_cast<unsigned short*>(w->d_x16), t * N);
    if (!ck(cudaGetLastError(), "kernels", err)) return false;
    if (!ck(cudaMemcpyAsync(w->h_sum16, w->d_x16, (size_t) t * N * 2, cudaMemcpyDeviceToHost, s), "sums copy", err)) return false;
    if (!ck(cudaEventRecord(w->done, s), "event record", err)) return false;
    w->pending_t = t;
    return true;
}

bool prefill_finish(PrefillWs* w, uint16_t* sum16, double* ms_wait, std::string& err) {
    if (w == nullptr || w->pending_t < 0) { err = "helper prefill: finish without begin"; return false; }
    const auto t0 = std::chrono::steady_clock::now();
    int prev = 0;
    cudaGetDevice(&prev);
    struct Restore { int p; ~Restore() { cudaSetDevice(p); } } restore{prev};
    cudaSetDevice(w->device);
    const int64_t t = w->pending_t;
    w->pending_t = -1;
    if (!ck(cudaEventSynchronize(w->done), "wait", err)) return false;
    if (sum16 != w->h_sum16) std::memcpy(sum16, w->h_sum16, (size_t) t * N * 2);
    if (ms_wait != nullptr) *ms_wait += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    return true;
}

}  // namespace strata::plugin
