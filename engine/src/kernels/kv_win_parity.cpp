// src/kernels/kv_win_parity.cpp - WP-2a's exact window side pool against the quantized pool it twins (GPU, synthetic, no model).
//
// kv_window_parity covers the streamed block-slot rings the verify and MTP paths share. WP-2a's EXACT WINDOW
// (`--kv q4lm32w`) makes a different reader: kv_gather_q4_step asks `qsa_win_slot` for a row of the always-resident
// side pool ([R sinks][S + B recent ring] fp16, the recent rows guarded by `win_tag`) and answers quantized
// everywhere else. No other test drives that pair: this one does, on both append forms the engine makes
// (kv_append_win_steps for the n <= 6 decode windows, kv_append_win for the prompt path's contiguous pos0/T) and
// on a ring small enough to wrap: cells are appended in batches that sometimes hit the legal maximum (T = S + B),
// and after every append the whole frontier is gathered twice - window attached and detached (null window, pure
// quantized reference) - and checked, per cell and channel, against
//
//   1. EXACTNESS: a cell expected exact reads bitwise the fp16 twin the writer just stored (the host re-derives
//      the same f16_from_f32 of the same rotated values - that function is pure bit arithmetic, so host and
//      device agree); every other cell reads bitwise its memory-free detached gather. The host predicate is an
//      independent re-derivation of `qsa_win_slot`, not a call of it: exact iff c < R (sinks are cell-addressed),
//      or (win_all) the row's tag names the cell, or (horizon, win_all == 0) c >= horizon - S AND the tag names it.
//   2. UNIFORMITY: within one cell no channel may go its own way - a read cannot be partly twin and partly
//      quantized without mixing side-pool rows. (A quantized read that happens to bit-equal the twin is logged
//      as "both" and tolerated - it is a property of the values, not of the reader.)
//   3. SINKS NEVER MOVE: rows 0..R-1 still hold cells 0..R-1 after every append, whatever the ring did.
//   4. REWIND: reading cells [0, n_want) at a horizon n_want far below the last appended cell - the resume
//      pattern - must return, for each leftover cell, either its own twin (tag still names it) or its own
//      quantized values - NEVER a later cell's. The tag is what turns overwritten ring rows into quantized reads.
//   5. THE LEGAL-MAX BATCH: T = S + B lands every recent row exactly once per launch and stays correct; the
//      over-sized batch (T = S + B + 1) must fail loudly (exit 1). That refusal is its own ctest
//      (`kv_win_parity_refuse`, WILL_FAIL) because the writer exits before anything can be observed in-process.
//
// Two tag initialisations are covered (zero like a memset workspace must, and pre-invalidated 0x7FFFFFF) - neither
// may endorse a wrong cell: a zero tag can never equal a recent cell c (c >= R >= 1 for the recent rows), and a
// pre-invalidated tag can never equal anything. `STRATA_KV_WINDOW_TAG=1` (win_all) is a second ctest of this same
// binary: kv_gather_q4_step reads that env once, so the whole process runs under one setting, and the host
// predicate mirrors the same read. The gather mechanics under test are mode-independent (the window row is taken
// before the q4 mode dispatch), so this test uses plain q4_0 (kUniform); q4lm's level tables have their own parity
// (kv_q4_parity). The fused attention reader's identical env switch is not exercised here - the scratch gather is
// the path this test pins.
#include "strata/kernels/f16_bits.hpp"    // f16_from_f32 on the host: an STRATA_HD inline, pure bit arithmetic
#include "strata/kernels/kv_q4.hpp"       // the appends/gathers and QsaShapes; brings qsa.hpp (the kStep* batch consts)

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

namespace k = strata::kernels;

namespace {

void ck(cudaError_t e, const char* w) {
    if (e != cudaSuccess) { std::fprintf(stderr, "%s: %s\n", w, cudaGetErrorString(e)); std::exit(2); }
}
template <typename T> T* dalloc(size_t n) {
    T* p = nullptr;
    ck(cudaMalloc(&p, n * sizeof(T) + 64), "malloc");
    ck(cudaMemset(p, 0, n * sizeof(T) + 64), "memset");
    return p;
}

// A ring small enough to wrap a few times, big enough for the boundary shapes: batch sizes 1..6 (the drafter's
// window) and the legal maximum T = S + B. NMAX is 3 wraps and change.
constexpr int R = 32;                  // sink cells (engine default 8; cell-addressed rows 0..R-1)
constexpr int S = 96;                  // exact cells per horizon (engine default 128)
constexpr int B = 32;                  // ring rows beyond S (engine default 512)
constexpr int RING = S + B;            // the recent ring's rows
constexpr int ROWS = R + RING;         // the side pool's rows: sinks on top of the ring
constexpr int NMAX = 3 * RING + 41;

int g_bad = 0;
void fail(const char* what, ...) {
    if (g_bad++ < 12) {
        std::fputs("  FAIL ", stderr);
        va_list a; va_start(a, what); std::vfprintf(stderr, what, a); va_end(a);
        std::fputc('\n', stderr);
    }
}

// The win_all the gather wrapper will use - the same one-shot read of STRATA_KV_WINDOW_TAG its static makes.
bool win_all_env() {
    static const bool w = [] { const char* e = std::getenv("STRATA_KV_WINDOW_TAG"); return e && e[0] == '1'; }();
    return w;
}

// Independent re-derivation of qsa_win_slot for THIS test's grid (S, R, B fixed, tag array host-read back) with
// horizon n. Two implementations of one rule - the test pins the BEHAVIOR of writer + gather against this rule,
// and any drift between them fails a run rather than passing quietly.
long long win_slot_host(long long c, long long n, const std::vector<int32_t>& tag) {
    if (c < R) return c;                                          // sink row: cell-addressed, never needs a tag
    if (!win_all_env() && c < n - (long long) S) return -1;        // the horizon check (win_all skips it)
    if ((long long) tag[(size_t)(c % RING)] != c) return -1;      // the row still has to name this cell
    return R + c % RING;
}

// The rotated values the test feeds the writer. Per (cell, head, channel) unique across cells, printable small
// integers plus a sub-1 fraction: two cells at least 1 apart can never round to the same fp16 point (the fraction
// stays below 1 - 2 * 0.125), so a cross-cell read can never bit-match the expected twin.
float kval(int c, int h, int d) { return (float) c + (float) h / 32.0f + (float) d / 4096.0f; }
float vval(int c, int h, int d) { return -1.0f - (float) c - (float) h / 32.0f - (float) d / 4096.0f; }

struct Phase {
    k::QsaShapes s;
    int H = 0, D = 0;
    int64_t n_pages = 0;
    uint8_t* k_q4 = nullptr;      // the quantized pools, identity paging: [page][H][page_size][144 B per head]
    uint8_t* v_q4 = nullptr;
    uint16_t* k_win = nullptr;    // the side pool: [ROWS][H][D] fp16 per tensor
    uint16_t* v_win = nullptr;
    int32_t* tag = nullptr;       // [RING]
    int32_t* table = nullptr;
    int32_t* steps = nullptr;     // kStepCount ints
    int32_t* ids = nullptr;       // up to NMAX ids
    float* kc = nullptr;          // one append batch's rotated cells [n <= RING][H][D]
    float* vc = nullptr;
    uint16_t* kw = nullptr;       // gather scratch: windowed and detached, K and V
    uint16_t* vw = nullptr;
    uint16_t* kd = nullptr;
    uint16_t* vd = nullptr;
    cudaStream_t cs = nullptr;
    std::vector<int32_t> htag;    // the tags as the ring actually holds them
    long long appended = 0, cells_checked = 0, cells_exact = 0, cells_quant = 0, cells_both = 0, batches = 0;

    bool alloc(uint32_t tag_init) {
        s = k::qsa_real_shapes();
        H = (int) s.n_head_kv;
        D = (int) s.head_dim;
        if (D != 256) { std::fprintf(stderr, "kv_win_parity: head_dim %d, need 256 (the q4 appends' shape)\n", D); return false; }
        if (H > 29) {  // kval/vval: the fraction ((h/32) + (d/4096)) must stay below one unit so cross-cell
                       // rounding to one fp16 point stays impossible - 29/32 + 255/4096 < 1, 30/32 + ... is not
            std::fprintf(stderr, "kv_win_parity: n_head_kv %d, the value encoding assumes <= 29\n", H);
            return false;
        }
        n_pages = (NMAX + s.page_size - 1) / s.page_size + 1;
        k_q4 = dalloc<uint8_t>((size_t) n_pages * H * (size_t) s.page_size * k::kv_q4_bytes_per_head(D, k::Q4Mode::kUniform));
        v_q4 = dalloc<uint8_t>((size_t) n_pages * H * (size_t) s.page_size * k::kv_q4_bytes_per_head(D, k::Q4Mode::kUniform));
        k_win = dalloc<uint16_t>((size_t) ROWS * H * D);
        v_win = dalloc<uint16_t>((size_t) ROWS * H * D);
        tag = dalloc<int32_t>((size_t) RING);
        {   // the second initialisation: a pre-invalidated tag (no row names a cell until a writer puts it there)
            std::vector<int32_t> t((size_t) RING, (int32_t) tag_init);
            ck(cudaMemcpy(tag, t.data(), RING * 4, cudaMemcpyHostToDevice), "tag init");
        }
        table = dalloc<int32_t>((size_t) n_pages);
        steps = dalloc<int32_t>((size_t) RING * k::kStepCount);   // one append batch's step array: T * kStepCount
        ids = dalloc<int32_t>((size_t) NMAX);
        kc = dalloc<float>((size_t) RING * H * D);
        vc = dalloc<float>((size_t) RING * H * D);
        kw = dalloc<uint16_t>((size_t) NMAX * H * D);
        vw = dalloc<uint16_t>((size_t) NMAX * H * D);
        kd = dalloc<uint16_t>((size_t) NMAX * H * D);
        vd = dalloc<uint16_t>((size_t) NMAX * H * D);
        ck(cudaStreamCreate(&cs), "stream");
        std::vector<int32_t> t((size_t) n_pages);
        for (int64_t i = 0; i < n_pages; ++i) t[(size_t) i] = (int32_t) i;
        ck(cudaMemcpy(table, t.data(), n_pages * 4, cudaMemcpyHostToDevice), "table");
        ck(cudaStreamSynchronize(cs), "setup");
        return true;
    }

    void set_step(long long n, long long width) {
        const int32_t st[4] = {(int32_t)(n - 1), (int32_t) n, (int32_t)(n / 4), (int32_t) width};
        ck(cudaMemcpy(steps, st, sizeof(st), cudaMemcpyHostToDevice), "steps");
    }

    // Append T cells [pos0, pos0+T) with this phase's writer (T <= S + B: the writers refuse more).
    void append(int64_t pos0, int64_t T, bool use_steps) {
        std::vector<float> hk((size_t) T * H * D), hv((size_t) T * H * D);
        for (int64_t t = 0; t < T; ++t)
            for (int h = 0; h < H; ++h)
                for (int d = 0; d < D; ++d) {
                    const size_t i = (size_t)(t * H * D + (size_t) h * D + d);
                    hk[i] = kval((int)(pos0 + t), h, d);
                    hv[i] = vval((int)(pos0 + t), h, d);
                }
        ck(cudaMemcpy(kc, hk.data(), hk.size() * 4, cudaMemcpyHostToDevice), "kc");
        ck(cudaMemcpy(vc, hv.data(), hv.size() * 4, cudaMemcpyHostToDevice), "vc");
        if (use_steps) {
            std::vector<int32_t> st((size_t) T * k::kStepCount);
            for (int64_t t = 0; t < T; ++t) {
                const int64_t p = pos0 + t;
                st[(size_t) t * k::kStepCount + k::kStepPos] = (int32_t) p;
                st[(size_t) t * k::kStepCount + k::kStepNKv] = (int32_t)(p + 1);
                st[(size_t) t * k::kStepCount + k::kStepNBid] = (int32_t)((p + 1) / 4);
                st[(size_t) t * k::kStepCount + k::kStepWidth] = (int32_t)(p + 1);
            }
            ck(cudaMemcpy(steps, st.data(), st.size() * 4, cudaMemcpyHostToDevice), "steps[]");
            k::kv_append_q4_steps(k_q4, v_q4, table, steps, k::kStepCount, (int) T, kc, vc, s, cs, nullptr,
                                  k::Q4Mode::kUniform);
            k::kv_append_win_steps(k_win, v_win, tag, steps, k::kStepCount, (int) T, kc, vc, H, D, cs, S, R, B);
        } else {
            k::kv_append_q4(k_q4, v_q4, table, pos0, T, kc, vc, s, cs, nullptr, nullptr, k::Q4Mode::kUniform);
            k::kv_append_win(k_win, v_win, tag, pos0, T, kc, vc, H, D, cs, S, R, B);
        }
        ck(cudaStreamSynchronize(cs), "append");
        ck(cudaGetLastError(), "append (last error)");
        appended += T;
        ++batches;
    }

    // Gather [lo, hi) at horizon n, window attached and detached, and checker every cell against the predicate.
    void sweep(long long lo, long long hi, long long n) {
        const long long cnt = std::max<long long>(0, hi - lo);
        std::vector<int32_t> hid((size_t) std::max<long long>(1, cnt));
        for (long long i = 0; i < cnt; ++i) hid[(size_t) i] = (int32_t)(lo + i);
        if (cnt > 0) ck(cudaMemcpy(ids, hid.data(), cnt * 4, cudaMemcpyHostToDevice), "ids");
        set_step(n, cnt);
        k::kv_gather_q4_step(k_q4, v_q4, table, ids, steps, cnt, s, kw, vw, cs, k::Q4Mode::kUniform,
                             k_win, v_win, tag, S, R, B);
        k::kv_gather_q4_step(k_q4, v_q4, table, ids, steps, cnt, s, kd, vd, cs, k::Q4Mode::kUniform,
                             nullptr, nullptr, nullptr, 0, 0, 0);
        ck(cudaMemcpy(htag.data(), tag, RING * 4, cudaMemcpyDeviceToHost), "tag d2h");
        std::vector<uint16_t> hw(size_t(std::max<long long>(1, cnt)) * H * D), hq = hw, hd = hw, hdd = hw;
        if (cnt > 0) {
            ck(cudaMemcpy(hw.data(), kw, hw.size() * 2, cudaMemcpyDeviceToHost), "kw d2h");
            ck(cudaMemcpy(hq.data(), vw, hw.size() * 2, cudaMemcpyDeviceToHost), "vw d2h");
            ck(cudaMemcpy(hd.data(), kd, hw.size() * 2, cudaMemcpyDeviceToHost), "kd d2h");
            ck(cudaMemcpy(hdd.data(), vd, hw.size() * 2, cudaMemcpyDeviceToHost), "vd d2h");
        }
        for (long long i = 0; i < cnt; ++i) {
            const int c = (int)(lo + i);
            const long long slot = win_slot_host(c, n, htag);   // what side-pool row (or -1: quantized) this cell gets
            int chE = 0, chQ = 0, chB = 0, chN = 0;             // twin / own-quantized / both / neither, per channel
            for (int h = 0; h < H; ++h)
                for (int d = 0; d < D; ++d) {
                    const size_t w = (size_t)(i * H * D + (size_t) h * D + d);
                    const uint16_t exp_k = k::f16_from_f32(kval(c, h, d));
                    const uint16_t exp_v = k::f16_from_f32(vval(c, h, d));
                    // two sides per channel: K and V; mine = the windowed gather, ref = the detached (quantized) one
                    const uint16_t* mine[2] = {hw.data() + w, hq.data() + w};
                    const uint16_t* ref[2] = {hd.data() + w, hdd.data() + w};
                    const uint16_t* exq[2] = {&exp_k, &exp_v};
                    for (int sd = 0; sd < 2; ++sd) {
                        const uint16_t gotm = mine[sd][0], expj = *exq[sd], dnt = ref[sd][0];
                        if (gotm == expj && gotm == dnt) ++chB;      // twin and quantized agree: a property of the values
                        else if (gotm == expj) ++chE;
                        else if (gotm == dnt) ++chQ;
                        else ++chN;
                    }
                }
            ++cells_checked;
            if (chN || (chE && chQ) || (slot >= 0 ? (chQ != 0) : (chE != 0)))
                fail("cell %d at horizon %lld: slot %lld, channels exact %d quant %d both %d neither %d",
                     c, n, slot, chE, chQ, chB, chN);
            else if (chB == H * D * 2) ++cells_both;   // twin == quantized everywhere: a property of the values
            else if (slot >= 0) ++cells_exact;
            else ++cells_quant;
        }
    }

    // The sinks are cell-addressed rows that no append may move: after any append the rows still hold cells 0..R-1.
    void check_sinks() {
        std::vector<uint16_t> rows((size_t) R * H * D);
        ck(cudaMemcpy(rows.data(), k_win, rows.size() * 2, cudaMemcpyDeviceToHost), "k_win d2h");
        for (int c = 0; c < R; ++c) {
            long long bad = 0;
            for (int h = 0; h < H; ++h)
                for (int d = 0; d < D; ++d) {
                    const size_t i = (size_t)(c * H + h) * D + d;
                    bad += rows[i] != k::f16_from_f32(kval(c, h, d));
                }
            if (bad) { char w[64]; std::snprintf(w, sizeof(w), "K sink row for cell %d moved", c); fail(w); }
        }
        ck(cudaMemcpy(rows.data(), v_win, rows.size() * 2, cudaMemcpyDeviceToHost), "v_win d2h");
        for (int c = 0; c < R; ++c) {
            long long bad = 0;
            for (int h = 0; h < H; ++h)
                for (int d = 0; d < D; ++d) {
                    const size_t i = (size_t)(c * H + h) * D + d;
                    bad += rows[i] != k::f16_from_f32(vval(c, h, d));
                }
            if (bad) { char w[64]; std::snprintf(w, sizeof(w), "V sink row for cell %d moved", c); fail(w); }
        }
    }

    // The append schedule: batches of the drafter's sizes, every 13th one at the legal maximum T = S + B.
    void drive(bool steps_form) {
        htag.resize((size_t) RING);
        std::mt19937 rng(7 + 3 * R + steps_form);
        while (appended < NMAX) {
            const int64_t pos0 = appended;
            const int64_t T = (batches % 13 == 0) ? RING : 1 + (int64_t)(rng() % 6);
            append(pos0, std::min<int64_t>(T, NMAX - pos0), steps_form);
            // the frontier the append just moved: the new cells plus a ring's worth below
            sweep(std::max<int64_t>(0, pos0 - RING), appended, appended);
            check_sinks();
        }
    }

    // Every horizon one step at a time: the small ones fully, then wrap boundaries and every (2*ring)-th
    void horizon_sweeps() {
        for (long long n = 1; n <= std::min<long long>(RING + S + 8, appended); ++n) sweep(0, n, n);
        for (long long n = 2 * RING; n <= appended; n += 2 * RING) sweep(0, n, n);
        sweep(0, appended, appended);
    }

    // The rewind: the last cells a resume would read back at horizons far below the append frontier
    void rewind_sweeps() {
        for (long long n : {(long long) 17, (long long) R - 1, (long long) R, (long long) R + 1,
                            (long long) RING - 1, (long long) RING, (long long) RING + 1,
                            (long long)(NMAX - 1), (long long) NMAX})
            if (n > 0 && n <= appended) sweep(0, n, n);
    }

    void report(const char* name) {
        std::printf("  %-9s: %lld cells in %lld batches, %lld checked (exact %lld, quantized %lld, twin==quant %lld), bad %d%s\n",
                    name, appended, batches, cells_checked, cells_exact, cells_quant, cells_both, g_bad,
                    win_all_env() ? "  [win_all]" : "");
    }
};

int run_all() {
    std::printf("kv_win_parity: the window side pool vs the quantized pool it twins - R %d S %d B %d rows %d, cells %d%s\n",
                R, S, B, R + RING, NMAX, win_all_env() ? "  [STRATA_KV_WINDOW_TAG=1: the horizon check is skipped, tag-only]" : "");
    struct Named { const char* name; bool steps_form; uint32_t tag_init; bool horizon; bool rewind; };
    const Named phases[] = {
        {"steps/0", true, 0, true, false},       // the n <= 6 form, zero tags (what a memset workspace must be)
        {"batch/0", false, 0, true, false},      // the prompt path's pos0/T form
        {"steps/inv", true, 0x7FFFFFFFu, false, true},    // pre-invalidated tags, then the rewind pattern
        {"batch/inv", false, 0x7FFFFFFFu, false, true},
    };
    for (const Named& ph : phases) {
        Phase p;
        if (!p.alloc(ph.tag_init)) return 1;
        p.drive(ph.steps_form);
        if (ph.horizon) p.horizon_sweeps();
        if (ph.rewind) p.rewind_sweeps();
        p.check_sinks();
        ck(cudaStreamSynchronize(p.cs), "phase");
        ck(cudaGetLastError(), "phase (last error)");
        p.report(ph.name);
        cudaStreamDestroy(p.cs);
        cudaFree(p.k_q4); cudaFree(p.v_q4); cudaFree(p.k_win); cudaFree(p.v_win); cudaFree(p.tag);
        cudaFree(p.table); cudaFree(p.steps); cudaFree(p.ids); cudaFree(p.kc); cudaFree(p.vc);
        cudaFree(p.kw); cudaFree(p.vw); cudaFree(p.kd); cudaFree(p.vd);
    }
    if (g_bad == 0) {
        std::printf("kv_win_parity: PASS\n");
        return 0;
    }
    std::printf("kv_win_parity: FAIL (%d)\n", g_bad);
    return 1;
}

// --refuse: exactly one over-sized batch (T = S + B + 1). The writer exits(1) before the launch, so a run that
// reaches here is the bug the K2 guard exists for, and the ctest's WILL_FAIL turns the refusal into a contract.
int run_refuse() {
    std::printf("kv_win_parity: the over-sized batch (T = S + B + 1 = %d) must be refused\n", RING + 1);
    k::QsaShapes s = k::qsa_real_shapes();
    const int H = (int) s.n_head_kv, D = (int) s.head_dim;
    uint16_t* k_win = dalloc<uint16_t>((size_t)(R + RING) * H * D);
    uint16_t* v_win = dalloc<uint16_t>((size_t)(R + RING) * H * D);
    int32_t* tag = dalloc<int32_t>((size_t) RING);
    float* hc = new float[(size_t)(RING + 1) * H * D]{};
    cudaStream_t cs;
    ck(cudaStreamCreate(&cs), "stream");
    k::kv_append_win(k_win, v_win, tag, 0, RING + 1, hc, hc, H, D, cs, S, R, B);
    std::fprintf(stderr, "kv_win_parity: --refuse: the over-sized batch was NOT refused - the side pool went stale\n");
    return 2;   // unreachable if the guard works (kv_append_win exits first)
}
}  // namespace

int main(int argc, char** argv) {
    if (argc > 1 && std::strcmp(argv[1], "--refuse") == 0) return run_refuse();
    return run_all();
}
