// opencl/probe/ocl_peer_parity.cpp - the OpenCL peer tier against the CUDA kernels it has to reproduce.
//
// This is the check that decides whether the MI50 can carry expert rows for the engine: the peer's whole
// claim is that an expert gives the SAME row on either card, because the reference path for a VRAM hit is
// `quantize_q8_1_rows` + `native_expert_grouped` on the RTX 5060 Ti (src/core/peer_experts.cpp).  So this
// runs both, on identical blobs, identical activations and the identical entry plan, and compares the
// output floats one by one.
//
// The seven format pairs are the iq3_s pack's own (native_experts.txt): every layer of the model is one of
// them, at n_embd 2560 and n_ff 640.  The blob bytes they imply (2176000, 1510400, 2329600, ...) are
// exactly what that file states, which is the first thing the peer's open() re-derives for itself.
//
//     ocl_peer_parity            the checks
//     ocl_peer_parity --bench    the checks, then the tier's experts/ms on a window of 48 layers
//
// Random bytes are valid codes for every format here (every grid index is in range); only the fp16 block
// scales are chosen, small enough that the SwiGLU outputs keep a finite fp16 q8_1 scale.
#include "strata/core/ocl_peer.hpp"
#include "strata/kernels/iq_kernels.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

namespace k = strata::kernels;
using strata::core::OclLayerInfo;
using strata::core::OclPeer;

namespace {

int g_fail = 0;

void ck(cudaError_t e, const char* what) {
    if (e != cudaSuccess) {
        std::fprintf(stderr, "CUDA error in %s: %s\n", what, cudaGetErrorString(e));
        std::exit(2);
    }
}

constexpr int64_t H = 2560;      // n_embd
constexpr int64_t FF = 640;      // expert intermediate width
constexpr int NE = 8;            // experts per layer in this probe
constexpr int NT = 4;            // tokens
constexpr int KK = 6;            // entries per token

const char* name_of(int t) {
    switch (t) {
        case 18: return "IQ3_XXS";
        case 20: return "IQ4_NL";
        case 21: return "IQ3_S";
        case 22: return "IQ2_S";
        case 23: return "IQ4_XS";
        case 42: return "Q2_0";
        default: return "?";
    }
}
int block_values(int t) { return t == 20 ? 32 : t == 42 ? 64 : 256; }

/// `rows` rows of `n` values of format t: random bytes, then a finite fp16 scale in every block.  Copied
/// from src/kernels/native_grouped_parity.cpp so the reference sees blobs of the same character.
std::vector<uint8_t> random_rows(int t, int64_t rows, int64_t n, std::mt19937& rng) {
    const size_t rb = k::iq_row_bytes(t, n), bs = k::iq_row_bytes(t, block_values(t));
    std::vector<uint8_t> w((size_t) rows * rb);
    std::uniform_int_distribution<int> byte(0, 255), ex(2, 8), man(0, 1023), sgn(0, 1);
    for (auto& b : w) b = (uint8_t) byte(rng);
    for (size_t o = 0; o < w.size(); o += bs) {
        const int e = ex(rng);
        const uint16_t h = (uint16_t) ((sgn(rng) ? 0x8000 : 0) | (e << 10) | man(rng));
        std::memcpy(&w[o], &h, 2);
    }
    return w;
}

/// The seven (gate/up, down) pairs the iq3_s pack uses, with how many layers of each.
struct Case { int gu, d, layers; };
const Case kCases[] = {
    {22, 20, 2}, {18, 20, 2}, {21, 20, 2}, {22, 42, 1}, {18, 42, 1}, {23, 20, 1}, {21, 42, 1},
};

/// One expert blob, per (layer, expert).  The peer reads through this.
struct Pack {
    std::vector<OclLayerInfo> layers;
    std::vector<std::vector<std::vector<uint8_t>>> blobs;   // [layer][expert]
    std::vector<int> gu, d;
    int64_t n_layers = 0;

    const uint8_t* at(int64_t l, int64_t e) const {
        if (l < 0 || l >= n_layers || e < 0 || e >= NE) return nullptr;
        return blobs[(size_t) l][(size_t) e].data();
    }
    static const uint8_t* blob_fn(void* ctx, int64_t l, int64_t e) { return ((const Pack*) ctx)->at(l, e); }

    explicit Pack(std::mt19937& rng) {
        for (const Case& c : kCases)
            for (int i = 0; i < c.layers; ++i) {
                const k::NativeExpertLayout L = k::native_expert_layout(c.gu, c.d, H, FF);
                OclLayerInfo li;
                li.gu_type = c.gu;
                li.d_type = c.d;
                li.n_embd = H;
                li.n_ff = FF;
                li.blob_bytes = L.bytes;
                layers.push_back(li);
                gu.push_back(c.gu);
                d.push_back(c.d);
            }
        n_layers = (int64_t) layers.size();
        blobs.resize((size_t) n_layers);
        for (int64_t l = 0; l < n_layers; ++l) {
            blobs[(size_t) l].resize(NE);
            for (int e = 0; e < NE; ++e) {
                auto b = random_rows(gu[(size_t) l], 2 * FF, H, rng);
                const auto dn = random_rows(d[(size_t) l], H, FF, rng);
                b.insert(b.end(), dn.begin(), dn.end());
                if (b.size() != layers[(size_t) l].blob_bytes) {
                    std::fprintf(stderr, "blob %zu != %llu bytes\n", b.size(),
                                 (unsigned long long) layers[(size_t) l].blob_bytes);
                    std::exit(2);
                }
                blobs[(size_t) l][(size_t) e] = std::move(b);
            }
        }
    }
};

template<typename T>
T* dalloc(size_t n) {
    T* p = nullptr;
    ck(cudaMalloc(&p, n * sizeof(T) + 256), "malloc");
    ck(cudaMemset(p, 0, n * sizeof(T) + 256), "memset");
    return p;
}

/// The CUDA reference for one layer: `ent` entries, each on the blob of `ids[ent]`, reading the q8_1 of
/// `ent/KK`, written to row `ent` of `out`.
constexpr int64_t kMaxCap = 512;      // scratch and `out` are sized for this many entries

struct Ref {
    int64_t cap = 0;
    uint8_t* blobs = nullptr;          // NE blobs, 256-byte aligned slots
    size_t slot = 0;
    uint8_t* xq = nullptr;
    uint8_t* scr = nullptr;
    float* out = nullptr;
    size_t out_floats = 0;
    unsigned long long* ptr = nullptr;
    int32_t* start = nullptr;
    int32_t* n = nullptr;
    int32_t* dst = nullptr;
    int32_t* tok = nullptr;
    float* gate_dev = nullptr;         // = scr, where the gu results land (for --dump)
    std::vector<float> gate_h;
    std::vector<uint8_t*> blob_ptr;    // the host side of the slot addresses, for ptr[]
    cudaStream_t s = nullptr;

    Ref(const Pack& P, int64_t layer, const std::vector<float>& x, std::mt19937&, cudaStream_t stream) {
        s = stream;
        cap = (int64_t) NT * KK;
        const k::NativeExpertLayout L = k::native_expert_layout(P.gu[(size_t) layer], P.d[(size_t) layer], H, FF);
        slot = (L.bytes + 255) & ~(size_t) 255;
        blobs = dalloc<uint8_t>(slot * NE);
        blob_ptr.resize(NE);
        for (int e = 0; e < NE; ++e) {
            blob_ptr[(size_t) e] = blobs + slot * (size_t) e;
            ck(cudaMemcpy(blobs + slot * (size_t) e, P.blobs[(size_t) layer][(size_t) e].data(), L.bytes,
                          cudaMemcpyHostToDevice), "blob");
        }
        float* dx = dalloc<float>((size_t) x.size());
        ck(cudaMemcpy(dx, x.data(), x.size() * 4, cudaMemcpyHostToDevice), "x");
        xq = dalloc<uint8_t>((size_t) NT * (H / 32) * 36);
        k::quantize_q8_1_rows(dx, NT, H, xq, s);
        ck(cudaStreamSynchronize(s), "quantize");
        cudaFree(dx);
        scr = dalloc<uint8_t>(k::native_expert_scratch_bytes(kMaxCap, FF));
        gate_dev = (float*) scr;
        gate_h.resize((size_t) kMaxCap * FF);
        out_floats = (size_t) kMaxCap * H;
        out = dalloc<float>(out_floats);
        ptr = dalloc<unsigned long long>(cap);
        start = dalloc<int32_t>(cap + 1);
        n = dalloc<int32_t>(1);
        dst = dalloc<int32_t>(cap);
        tok = dalloc<int32_t>(cap);
    }
    ~Ref() {
        cudaFree(blobs); cudaFree(xq); cudaFree(scr); cudaFree(out);
        cudaFree(ptr); cudaFree(start); cudaFree(n); cudaFree(dst); cudaFree(tok);
    }
    /// the real entry point: `ids` and `kind` are the probe's own routing tables
    void call(const int32_t* ids, const int32_t* kind, std::vector<float>& host, int64_t cap_entries = -1) {
        if (cap_entries < 0) cap_entries = (int64_t) NT * KK;
        std::vector<unsigned long long> p;
        std::vector<int32_t> st, ds, tk;
        const int32_t n_entries = NT * KK;
        std::vector<int> ent;
        for (int32_t e = 0; e < n_entries; ++e)
            if (kind[e] == 2) ent.push_back(e);
        for (size_t i = 0; i < ent.size(); ++i) {
            const int32_t e = ent[i];
            // grp_start[g] = the first entry of group g, grp_start[ng] = the total: so the *starts* are
            // pushed here and the total at the end (seeding it with 0 and pushing starts gives {0,0})
            if (i == 0 || ids[e] != ids[ent[i - 1]]) {
                p.push_back((unsigned long long) blob_ptr[(size_t) ids[e]]);
                st.push_back((int32_t) i);
            }
            ds.push_back(e);
            tk.push_back(e / KK);
        }
        st.push_back((int32_t) ent.size());
        const int ng = (int) p.size();
        ck(cudaMemset(out, 0, out_floats * 4), "memset");
        if (ng > 0) {
            ck(cudaMemcpyAsync(ptr, p.data(), p.size() * 8, cudaMemcpyHostToDevice, s), "ptr");
            ck(cudaMemcpyAsync(start, st.data(), st.size() * 4, cudaMemcpyHostToDevice, s), "start");
            ck(cudaMemcpyAsync(n, &ng, 4, cudaMemcpyHostToDevice, s), "n");
            ck(cudaMemcpyAsync(dst, ds.data(), ds.size() * 4, cudaMemcpyHostToDevice, s), "dst");
            ck(cudaMemcpyAsync(tok, tk.data(), tk.size() * 4, cudaMemcpyHostToDevice, s), "tok");
            const k::NativeExpertLayout L = k::native_expert_layout(gu_type, d_type, H, FF);
            // cap_groups = the group slots, cap_entries = the rows of `out` (the peer's rows are addressed
            // by their entry index, so a plan that skips entries still needs the whole block declared)
            k::native_expert_grouped(L, ptr, start, n, dst, tok, (int64_t) ng, cap_entries, xq, scr, out, s);
        }
        ck(cudaStreamSynchronize(s), "reference");
        if (std::getenv("PEER_DUMP") != nullptr) {
            int32_t dn = 0, dstart[4] = {0}, ddst[4] = {0}, dtok[4] = {0};
            unsigned long long dptr = 0;
            ck(cudaMemcpy(&dn, n, 4, cudaMemcpyDeviceToHost), "n");
            ck(cudaMemcpy(&dptr, ptr, 8, cudaMemcpyDeviceToHost), "ptr");
            ck(cudaMemcpy(dstart, start, 16, cudaMemcpyDeviceToHost), "start");
            ck(cudaMemcpy(ddst, dst, 16, cudaMemcpyDeviceToHost), "dst");
            ck(cudaMemcpy(dtok, tok, 16, cudaMemcpyDeviceToHost), "tok");
            std::printf("  dev: n=%d start=%d,%d dst=%d,%d tok=%d,%d  ng(local)=%d ent=%zu blob0=%p\n", dn, dstart[0],
                        dstart[1], ddst[0], ddst[1], dtok[0], dtok[1], ng, ent.size(), (void*) blob_ptr[0]);
            double gm = 0;
            ck(cudaMemcpy(gate_h.data(), gate_dev, (size_t) std::min<int64_t>(cap, 8) * FF * 4, cudaMemcpyDeviceToHost),
               "gate");
            for (int64_t i = 0; i < (int64_t) std::min<int64_t>(cap, 8) * FF; ++i)
                gm = std::max(gm, std::fabs((double) gate_h[(size_t) i]));
            std::printf("  dev: |gate| = %.6g over the first %lld entries\n", gm,
                        (long long) std::min<int64_t>(cap, 8));
        }
        host.assign(out_floats, 0.0f);
        ck(cudaMemcpy(host.data(), out, out_floats * 4, cudaMemcpyDeviceToHost), "out");
    }

    int gu_type = 0, d_type = 0;
};

/// The whole check for one layer: reference rows and peer rows, compared bit by bit.
struct Compare {
    size_t exact = 0, rows = 0;
    double max_abs = 0, max_rel = 0;
    int64_t worst_row = -1;
    size_t nonfinite = 0;
};

Compare compare_rows(const std::vector<float>& a, const std::vector<float>& b, const std::vector<int>& ent, int64_t first) {
    Compare c;
    for (const int e : ent) {
        ++c.rows;
        const float* ra = a.data() + (size_t) e * H;
        const float* rb = b.data() + (size_t) e * H;
        bool all = true;
        for (int64_t j = 0; j < H; ++j) {
            const float x = ra[j], y = rb[j];
            if (!std::isfinite(x) || !std::isfinite(y)) {
                ++c.nonfinite;
                all = false;
                continue;
            }
            const double ad = std::fabs((double) x - (double) y);
            // against the larger of the two: a row element that happens to be ~0 would otherwise make the
            // ratio meaningless and every layer would "fail" on a rounding bit
            const double rd = ad / std::max(1e-30, std::max(std::fabs((double) x), std::fabs((double) y)));
            if (ad != 0.0) all = false;
            if (ad > c.max_abs) c.max_abs = ad;
            if (rd > c.max_rel) { c.max_rel = rd; c.worst_row = first + e; }
        }
        if (all) ++c.exact;
    }
    return c;
}

}  // namespace

int main(int argc, char** argv) {
    const bool do_bench = argc > 1 && std::string(argv[1]) == "--bench";

    std::string name, err;
    double gib = 0, max_alloc = 0;
    if (!OclPeer::probe(0, name, gib, max_alloc)) {
        std::printf("no AMD OpenCL device: the peer tier cannot be checked here\n");
        return 2;
    }
    std::printf("peer device: %s  %.2f GiB total, %.2f GiB max single allocation\n", name.c_str(), gib, max_alloc);
    std::printf("reference:   native_expert_grouped on CUDA device 0\n\n");

    cudaStream_t s = nullptr;
    ck(cudaStreamCreate(&s), "stream");
    std::mt19937 rng(316);

    Pack P(rng);
    std::printf("pack: %lld layers, %d experts each, %lld x %lld\n\n", (long long) P.n_layers, NE, (long long) H,
                (long long) FF);

    // rank every pair, and let the primary hold none of them: the tier takes all of them
    std::vector<std::pair<int32_t, int32_t>> ranked;
    for (int32_t l = 0; l < (int32_t) P.n_layers; ++l)
        for (int32_t e = 0; e < NE; ++e) ranked.emplace_back(l, e);

    OclPeer peer;
    if (!peer.open(0, P.layers, NE, &Pack::blob_fn, &P, nullptr, ranked, 256, 0, err)) {
        std::printf("open failed: %s\n", err.c_str());
        ck(cudaStreamDestroy(s), "stream");
        return 1;
    }
    std::printf("open: %lld experts resident, %.3f GiB, %s\n", (long long) peer.resident(), peer.gib(),
                peer.device_name().c_str());
    {
        int64_t have = 0;
        for (int32_t l = 0; l < (int32_t) P.n_layers; ++l)
            for (int32_t e = 0; e < NE; ++e) have += peer.has(l, e) ? 1 : 0;
        const bool ok = have == peer.resident() && have == (int64_t) ranked.size();
        std::printf("has(): %lld of %zu pairs  %s\n", (long long) have, ranked.size(), ok ? "ok" : "FAIL");
        if (!ok) ++g_fail;
        if (peer.has(0, NE) || peer.has(P.n_layers, 0) || peer.has(-1, 0)) {
            std::printf("has(): an out-of-range pair answered yes  FAIL\n");
            ++g_fail;
        }
    }
    // ---- the gate/up stage on its own, through the engine's OTHER kernel
    //
    // The peer's rows differ from the grouped reference for some formats.  Before blaming the peer's
    // decode, this asks the engine itself: `iq_mmvq` (the prompt path) computes y[r] = W[r] . x for one
    // column, and `native_expert_grouped` computes the same thing for the same blob and the same q8_1.
    // If the two disagree, the peer is entitled to match whichever one it was written against.
    {
        std::printf("\n-- the gate/up stage: iq_mmvq (prompt path) against native_expert_grouped's gate/up\n");
        std::normal_distribution<float> nd(0.f, 1.f);
        std::vector<float> x((size_t) H);
        for (auto& v : x) v = nd(rng);
        std::vector<int> done;
        for (int64_t layer = 0; layer < P.n_layers; ++layer) {
            const int gt = P.gu[(size_t) layer];
            if (std::find(done.begin(), done.end(), gt) != done.end()) continue;
            done.push_back(gt);
            const k::NativeExpertLayout L = k::native_expert_layout(gt, P.d[(size_t) layer], H, FF);
            Ref R = Ref(P, layer, x, rng, s);
            R.gu_type = gt;
            R.d_type = P.d[(size_t) layer];
            std::vector<int32_t> ids(1, 0), kind(1, 2);
            std::vector<float> ref;
            R.call(ids.data(), kind.data(), ref, 1);
            // the scratch: gate[0..n_ff) for entry 0, then up[0..n_ff), fa bytes apart
            std::vector<float> gu((size_t) 2 * FF);
            const size_t fa = ((size_t) FF * 4 + 255) & ~(size_t) 255;
            ck(cudaMemcpy(gu.data(), R.gate_dev, (size_t) FF * 4, cudaMemcpyDeviceToHost), "gate");
            ck(cudaMemcpy(gu.data() + FF, (const uint8_t*) R.gate_dev + fa, (size_t) FF * 4, cudaMemcpyDeviceToHost),
               "up");
            (void) L;
            float* dy = dalloc<float>((size_t) 2 * FF);
            k::iq_mmvq(gt, R.blob_ptr[0], R.xq, dy, (int) H, (int) (2 * FF), 1, s);
            ck(cudaStreamSynchronize(s), "mmvq");
            std::vector<float> y((size_t) 2 * FF);
            ck(cudaMemcpy(y.data(), dy, (size_t) 2 * FF * 4, cudaMemcpyDeviceToHost), "y");
            cudaFree(dy);
            double md = 0, mr = 0, mx = 0;
            for (size_t r = 0; r < y.size(); ++r) {
                const double d = std::fabs((double) gu[r] - (double) y[r]);
                md = std::max(md, d);
                mx = std::max(mx, std::fabs((double) y[r]));
                mr = std::max(mr, d / std::max(1e-30, std::max(std::fabs((double) gu[r]), std::fabs((double) y[r]))));
            }
            std::printf("  %-8s: |mmvq| %.4g  max|d| %.4g  max rel %.3g  %s\n", name_of(gt), mx, md, mr,
                        mr <= 1e-6 ? "the two engine paths agree" : "THE TWO ENGINE PATHS DISAGREE");
        }
    }

    // ---- the down stage on its own, by the same argument as the gate/up block above
    //
    // The down matrices are the half of the blob the peer gets wrong when it gets anything wrong, and they
    // have TWO engine kernels: `native_expert_grouped` takes the split path (Split<TD>::load / apply) and
    // for IQ4_NL with n_ff == 640 it is `row_dot_40_sub8`, an 8-lane tree, while `iq_mmvq` runs
    // `Fmt<TD>::dot` per column on the ordinary 32-lane warp.  They reduce to the same grouping, so if the
    // down stage's activation mapping is right the two agree - and the peer has to match them.
    {
        std::printf("\n-- the down stage: iq_mmvq against native_expert_grouped's down rows\n");
        std::normal_distribution<float> nd(0.f, 1.f);
        std::vector<float> x((size_t) H);
        for (auto& v : x) v = nd(rng);
        std::vector<int> done;
        for (int64_t layer = 0; layer < P.n_layers; ++layer) {
            const int dt = P.d[(size_t) layer];
            if (std::find(done.begin(), done.end(), dt) != done.end()) continue;
            done.push_back(dt);
            const k::NativeExpertLayout L = k::native_expert_layout(P.gu[(size_t) layer], dt, H, FF);
            Ref R = Ref(P, layer, x, rng, s);
            R.gu_type = P.gu[(size_t) layer];
            R.d_type = dt;
            std::vector<int32_t> ids(1, 0), kind(1, 2);
            std::vector<float> ref;
            R.call(ids.data(), kind.data(), ref, 1);
            // The scratch holds gate | up | h | hq, each region 256-byte aligned, and the down stage reads
            // `hq` - the q8_1 of the SwiGLU output.  Note `h` is NOT written when the fused SwiGLU+q8_1
            // kernel is the one selected (`g_grouped_v1` off), so this takes `hq` and hands the very same
            // blocks to the other kernel: both references then see one activation.
            const size_t fa = ((size_t) 1 * FF * 4 + 255) & ~(size_t) 255;      // cap_entries == 1 below
            const void* hq_dev = (const uint8_t*) R.gate_dev + 3 * fa;
            float* dy = dalloc<float>((size_t) H);
            k::iq_mmvq(dt, R.blob_ptr[0] + L.down_off, hq_dev, dy, (int) FF, (int) H, 1, s);
            ck(cudaStreamSynchronize(s), "mmvq down");
            std::vector<float> y((size_t) H);
            ck(cudaMemcpy(y.data(), dy, (size_t) H * 4, cudaMemcpyDeviceToHost), "y");
            cudaFree(dy);
            double md = 0, mr = 0, mx = 0;
            for (size_t r = 0; r < y.size(); ++r) {
                const double d = std::fabs((double) ref[r] - (double) y[r]);
                md = std::max(md, d);
                mx = std::max(mx, std::fabs((double) y[r]));
                mr = std::max(mr, d / std::max(1e-30, std::max(std::fabs((double) ref[r]), std::fabs((double) y[r]))));
            }
            std::printf("  %-8s: |mmvq| %.4g  max|d| %.4g  max rel %.3g  %s\n", name_of(dt), mx, md, mr,
                        mr <= 1e-6 ? "the two engine paths agree" : "THE TWO ENGINE PATHS DISAGREE");

            // -- and now the peer, on the same single entry and the same token.  `y` above is a fully
            // independent row of the same matrix against the same activation, so a mismatch here is the
            // peer's and nothing else; the first differing row indices say where to look.
            std::vector<float> got((size_t) H, 0.0f);
            std::vector<int32_t> ids1(1, 0), kind1(1, 2);
            if (!peer.launch(layer, x.data(), ids1.data(), 1, 1, kind1.data(), err) ||
                !peer.finish(got.data(), err)) {
                std::printf("    peer on one entry: %s  FAIL\n", err.c_str());
                ++g_fail;
                continue;
            }
            double pmd = 0;
            int64_t prows = 0;
            for (int64_t r = 0; r < H; ++r) {
                const double d = std::fabs((double) got[(size_t) r] - (double) y[(size_t) r]);
                if (d > 1e-3) {
                    if (prows < 6)
                        std::printf("    row %4lld: got %.6g  mmvq %.6g  ref %.6g  (|d| %.4g)\n", (long long) r,
                                    (double) got[(size_t) r], (double) y[(size_t) r], (double) ref[(size_t) r], d);
                    ++prows;
                }
                pmd = std::max(pmd, d);
            }
            std::printf("    peer vs mmvq on one entry: %lld of %lld rows differ, max|d| %.4g  %s\n",
                        (long long) prows, (long long) H, pmd, prows == 0 ? "ok" : "FAIL");
            if (prows != 0) ++g_fail;

            // Everything the two sides read, so a hypothesis about the decode can be re-run offline
            // against the numbers that actually came out (see iq4nl_where.py).
            if (dt == 20) {
                FILE* fh = std::fopen("iq4nl_dump.bin", "wb");
                if (fh != nullptr) {
                    const uint32_t hdr[4] = {(uint32_t) H, (uint32_t) FF, (uint32_t) L.down_off,
                                             (uint32_t) L.d_row};
                    std::fwrite(hdr, 4, 4, fh);
                    std::fwrite(P.blobs[(size_t) layer][0].data(), 1, P.blobs[(size_t) layer][0].size(), fh);
                    const size_t hq_bytes = (size_t) (FF / 32) * 36;
                    std::vector<uint8_t> hqh(hq_bytes);
                    ck(cudaMemcpy(hqh.data(), hq_dev, hq_bytes, cudaMemcpyDeviceToHost), "hq dump");
                    std::fwrite(hqh.data(), 1, hq_bytes, fh);
                    std::fwrite(got.data(), 4, (size_t) H, fh);
                    std::fwrite(y.data(), 4, (size_t) H, fh);
                    std::fwrite(ref.data(), 4, (size_t) H, fh);
                    std::fclose(fh);
                    std::printf("    dumped iq4nl_dump.bin: blob %zu B, hq %zu B\n",
                                P.blobs[(size_t) layer][0].size(), hq_bytes);
                }
            }
        }
    }

    std::printf("\n%-9s/%-7s %-34s %s\n", "gate/up", "down", "per-entry comparison", "verdict");

    std::uniform_int_distribution<int> pick_expert(0, NE - 1);
    double ms_peer = 0;
    int64_t experts_done = 0;
    int dbg = 0;

    // a one-entry plan, run against the reference at a range of cap_entries: which of the two numbers the
    // reference call needs to be told about is worth knowing before anything else
    {
        std::vector<float> x((size_t) NT * H);
        std::normal_distribution<float> nd(0.f, 1.f);
        for (auto& v : x) v = nd(rng);
        std::vector<int32_t> ids((size_t) NT * KK, 0), kind((size_t) NT * KK, 2);
        Ref R = Ref(P, 0, x, rng, s);
        R.gu_type = P.gu[0];
        R.d_type = P.d[0];
        for (const int64_t ce : {(int64_t) 1, (int64_t) 2, (int64_t) 4, (int64_t) 8, (int64_t) 24, (int64_t) 128}) {
            std::vector<float> ref;
            R.call(ids.data(), kind.data(), ref, ce);
            double m = 0;
            for (int64_t j = 0; j < H; ++j) m = std::max(m, std::fabs((double) ref[(size_t) j]));
            k::iq_set_old_kernels(true);
            std::vector<float> ref_old;
            R.call(ids.data(), kind.data(), ref_old, ce);
            k::iq_set_old_kernels(false);
            double mo = 0;
            for (int64_t j = 0; j < H; ++j) mo = std::max(mo, std::fabs((double) ref_old[(size_t) j]));
            std::printf("  one entry, cap_entries %4lld: |ref row 0| = %.6g   (old kernels: %.6g)\n", (long long) ce, m,
                        mo);
        }
        // and with a second entry on a different expert, to see whether the first one comes alive
        {
            std::vector<int32_t> ids2((size_t) 2, 0), kind2((size_t) 2, 2);
            ids2[1] = 1;
            std::vector<float> ref;
            R.call(ids2.data(), kind2.data(), ref, 2);
            for (int r = 0; r < 2; ++r) {
                double m = 0;
                for (int64_t j = 0; j < H; ++j) m = std::max(m, std::fabs((double) ref[(size_t) (r * H + j)]));
                std::printf("  two entries (experts 0,1), row %d: |ref| = %.6g\n", r, m);
            }
        }
    }

    for (int64_t layer = 0; layer < P.n_layers; ++layer) {
        // the activations: NT tokens of H floats
        std::normal_distribution<float> nd(0.f, 1.f);
        std::vector<float> x((size_t) NT * H);
        for (auto& v : x) v = nd(rng);

        // the routing: NT x KK entries, most of them the peer's, a few left to the CPU
        std::vector<int32_t> ids((size_t) NT * KK), kind((size_t) NT * KK, 0);
        std::vector<int> ent;
        for (int t = 0; t < NT; ++t)
            for (int j = 0; j < KK; ++j) {
                const int e = t * KK + j;
                ids[(size_t) e] = pick_expert(rng);
                kind[(size_t) e] = (e % 5 == 4) ? -1 : 2;       // 1 entry in 5 stays with the CPU
                if (kind[(size_t) e] == 2) ent.push_back(e);
            }

        Ref R = Ref(P, layer, x, rng, s);
        R.gu_type = P.gu[(size_t) layer];
        R.d_type = P.d[(size_t) layer];
        std::vector<float> ref;
        R.call(ids.data(), kind.data(), ref);

        std::vector<float> got((size_t) NT * KK * H, 0.0f);
        const auto t0 = std::chrono::steady_clock::now();
        if (!peer.launch(layer, x.data(), ids.data(), NT, KK, kind.data(), err)) {
            std::printf("launch failed: %s  FAIL\n", err.c_str());
            ++g_fail;
            continue;
        }
        if (!peer.finish(got.data(), err)) {
            std::printf("finish failed: %s  FAIL\n", err.c_str());
            ++g_fail;
            continue;
        }
        const auto t1 = std::chrono::steady_clock::now();
        ms_peer += std::chrono::duration<double, std::milli>(t1 - t0).count();
        experts_done += (int64_t) ent.size();

        const Compare c = compare_rows(ref, got, ent, 0);
        const bool ok = c.max_rel <= 1e-5 && c.nonfinite == 0 && c.rows == ent.size();
        char buf[160];
        std::snprintf(buf, sizeof(buf), "%zu rows, %zu exact, max|d| %.3g, max rel %.3g", c.rows, c.exact, c.max_abs,
                      c.max_rel);
        std::printf("%-9s/%-7s %-34s %s\n", name_of(P.gu[(size_t) layer]), name_of(P.d[(size_t) layer]), buf,
                    ok ? "ok" : "FAIL");
        if (!ok) ++g_fail;
        if (!ok && dbg < 2) {
            auto mx = [](const std::vector<float>& v, const std::vector<int>& ent, int64_t off) {
                double m = 0;
                for (const int e : ent)
                    for (int64_t j = 0; j < H; ++j) m = std::max(m, std::fabs((double) v[(size_t) (e * H + j + off)]));
                return m;
            };
            std::printf("  dbg: %zu peer entries in %zu entries; |ref| %.4g  |got| %.4g\n", ent.size(),
                        (size_t) NT * KK, mx(ref, ent, 0), mx(got, ent, 0));
            std::printf("  dbg: ref[%d][0..3] = %.6g %.6g %.6g %.6g   got = %.6g %.6g %.6g %.6g\n", ent[0],
                        ref[(size_t) ent[0] * H + 0], ref[(size_t) ent[0] * H + 1], ref[(size_t) ent[0] * H + 2],
                        ref[(size_t) ent[0] * H + 3], got[(size_t) ent[0] * H + 0], got[(size_t) ent[0] * H + 1],
                        got[(size_t) ent[0] * H + 2], got[(size_t) ent[0] * H + 3]);
            ++dbg;
        }

        // the test has to be able to fail: the same experts with the tokens shifted by one
        std::vector<int32_t> kind2 = kind;
        std::vector<int32_t> ids2 = ids;
        for (int e = 0; e < NT * KK; ++e)
            if (kind2[(size_t) e] == 2) ids2[(size_t) e] = (ids[(size_t) e] + 1) % NE;
        std::vector<float> got2((size_t) NT * KK * H, 0.0f);
        if (peer.launch(layer, x.data(), ids2.data(), NT, KK, kind2.data(), err) && peer.finish(got2.data(), err)) {
            const Compare d2 = compare_rows(ref, got2, ent, 0);
            if (d2.max_rel < 1e-3) {
                std::printf("  the control (every expert shifted by one) matched the reference too  FAIL\n");
                ++g_fail;
            }
        } else {
            std::printf("  the control launch failed: %s  FAIL\n", err.c_str());
            ++g_fail;
        }
    }

    std::printf("\npeer tier: %lld experts in %.2f ms (%.1f experts/ms, all waits included)\n",
                (long long) experts_done, ms_peer,
                ms_peer > 0 ? (double) experts_done / ms_peer : 0.0);
    {
        // what that means for a token: 48 layers x 10 experts, four fifths of them on the peer
        const double per_expert = ms_peer > 0 ? ms_peer / (double) experts_done : 0.0;
        const double per_token = 48.0 * 10.0 * 0.8 * per_expert;
        std::printf("           a token's ~384 peer experts would cost %.2f ms of it\n", per_token);
    }

    if (do_bench) {
        // the window the engine actually hands over: 48 calls, one per layer, in this same order.
        // STRATA_OCL_BENCH_KK overrides the entries per token: the engine's windows are small (measured
        // 1.6-6.9 entries per layer in serve), and the per-layer cost at THAT size is what decides whether
        // the tier is worth anything, so the sweep has to be able to reach it.
        const int reps = 30;
        int kk = KK;
        if (const char* v = std::getenv("STRATA_OCL_BENCH_KK")) {
            const int n = std::atoi(v);
            if (n >= 1 && n <= 64) kk = n;
        }
        const auto t0 = std::chrono::steady_clock::now();
        for (int r = 0; r < reps; ++r)
            for (int64_t layer = 0; layer < P.n_layers; ++layer) {
                std::vector<int32_t> ids((size_t) NT * kk), kind((size_t) NT * kk, 2);
                for (int t = 0; t < NT; ++t)
                    for (int j = 0; j < kk; ++j) {
                        ids[(size_t) (t * kk + j)] = ((t * 3 + j * 5) % NE);
                    }
                std::vector<float> x((size_t) NT * H, 0.01f);
                std::vector<float> got((size_t) NT * kk * H);
                if (!peer.launch(layer, x.data(), ids.data(), NT, kk, kind.data(), err) ||
                    !peer.finish(got.data(), err)) {
                    std::printf("bench launch failed: %s\n", err.c_str());
                    ++g_fail;
                    break;
                }
            }
        const auto t1 = std::chrono::steady_clock::now();
        const double ms = std::chrono::duration<double, std::milli>(t1 - t0).count() / reps;
        const double per_layer = ms / (double) P.n_layers;
        const double per_token = per_layer * 48.0 * (1.0 - 1.0 / 5.0);
        std::printf("--bench: NT=%d KK=%d (%d entries/layer): %.3f ms per layer call, %.3f ms per token at this "
                    "tier's share\n", NT, kk, NT * kk, per_layer, per_token);
    }

    peer.close();
    ck(cudaStreamDestroy(s), "stream");
    std::printf("\nocl_peer_parity: %d failures\n", g_fail);
    return g_fail ? 1 : 0;
}
