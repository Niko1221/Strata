// metal/tests/dispatch_lat.cpp - what ONE dispatch and ONE memory barrier cost on this GPU, on dummy work.
// The engine's T=2 decode window is 1189 dispatches in one concurrent encoder with ~1100 barriers
// (window.cpp push(): barrier unless nb_): real windows measure 86-110 ms GPU, the warm kprof kernel sum
// is 48.6 ms.  If a barrier costs b us, 1100 of them cost 1.1b ms - this puts b on the machine without
// racing the model (round 15's NOBARRIER strip crashes: raced values feed host table indexes).
//
//   dispatch_lat            all rows
//   device_name, per row: total ms over the reps median and us per dispatch
#include "mtl_rt.h"

#include <algorithm>
#include <cstdio>
#include <cstdint>
#include <vector>
#include <chrono>
#include <unistd.h>

namespace {

struct XArg { uint32_t n, nt; };   // xprep's scalars, as a buffer for the ICB row (inline bytes are not allowed there)

mtl::Dispatch prep(const mtl::Pipeline& p, const mtl::Buffer& x, const mtl::Buffer& xp, uint32_t grid,
                   const mtl::Buffer* scalars) {
    mtl::Dispatch d;
    d.p = &p;
    const uint32_t n = 16, nt = 1;
    if (scalars)
        d.args = {mtl::Arg::B(x), mtl::Arg::B(xp), mtl::Arg::B(*scalars, 0), mtl::Arg::B(*scalars, 4)};
    else
        d.args = {mtl::Arg::B(x), mtl::Arg::B(xp), mtl::Arg::V(n), mtl::Arg::V(nt)};
    d.tg[0] = 256;
    d.grid[0] = grid;
    d.barrier = true;
    return d;
}

double med(std::vector<double> v) { std::sort(v.begin(), v.end()); return v[v.size() / 2]; }

mtl::Context* g_cx = nullptr;

void row(const char* what, const std::vector<mtl::Dispatch>& ds, int reps, bool icb = false) {
    const int n = (int) ds.size();
    g_cx->warm(ds, 0.2);
    std::vector<double> t;
    for (int r = 0; r < reps; ++r) t.push_back(icb ? g_cx->run_icb(ds, 1, false) : g_cx->run_graph(ds));
    const double m = med(t);
    std::printf("%-28s n %4d  %8.2f ms total  %6.2f us/dispatch\n", what, n, m * 1e3, m * 1e6 / n);
}

}  // namespace

// run_graph_seg's per-cut handshake, isolated: MTLSharedEvent signal -> host spin notices -> at_cut ->
// host store -> GPU wait releases, x N cuts, on ~zero GPU work per segment.  The engine's decode window
// is 48 cuts; the real windows sit ~600-700 us/cut of GPU idle while the stream's wait_ms accounts only
// 90-160 us, so the crossing itself is the suspect.  at_cut: nop (pure handshake) vs peek (reads a shared
// buffer the way the engine's ids_ look does).
namespace {

volatile uint32_t sink = 0;
void cut_nop(void*, int) {}
void cut_peek(void* user, int) { sink += *(const uint32_t*) user; }

void seg_row(const char* what, const std::vector<mtl::Dispatch>& ds, const std::vector<int>& cuts,
             void (*at_cut)(void*, int), void* user, int reps) {
    g_cx->warm(ds, 0.2);
    std::vector<double> wall, gpu;
    for (int r = 0; r < reps; ++r) {
        const auto t0 = std::chrono::steady_clock::now();
        const double g = g_cx->run_graph_seg(ds, cuts, at_cut, user);
        wall.push_back(std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
        gpu.push_back(g);
    }
    std::printf("%-28s n %4d cuts %3d  wall %7.2f ms  gpu %7.2f ms  %6.1f us/cut\n", what, (int) ds.size(),
                (int) cuts.size(), med(wall) * 1e3, med(gpu) * 1e3, med(wall) * 1e6 / cuts.size());
}

void cut_sleep(void*, int) { usleep(300); }   // the engine's host-turn size: does the GPU pay for it now?

// the device-flag handshake: one encoder, the queue never stops (see Context::run_graph_flag).  ctr: the
// shared buffer's int32s [seq, rel, seen].
void flag_row(const char* what, const std::vector<mtl::Dispatch>& ds, const std::vector<int>& cuts,
              const mtl::Dispatch& sd, const mtl::Dispatch& td, int* ctr,
              void (*at_cut)(void*, int), void* user, int reps) {
    g_cx->warm(ds, 0.2);
    std::vector<double> wall, gpu;
    for (int r = 0; r < reps; ++r) {
        std::memset(ctr, 0, 12);
        const auto t0 = std::chrono::steady_clock::now();
        const double g = g_cx->run_graph_flag(ds, cuts, sd, td, ctr, ctr + 1, at_cut, user);
        wall.push_back(std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
        gpu.push_back(g);
    }
    std::printf("%-28s n %4d cuts %3d  wall %7.2f ms  gpu %7.2f ms  %6.1f us/cut\n", what, (int) ds.size(),
                (int) cuts.size(), med(wall) * 1e3, med(gpu) * 1e3, med(wall) * 1e6 / cuts.size());
}

}  // namespace

int main() {
    mtl::Context cx;
    g_cx = &cx;
    const int lib = cx.compile({"quant.metal", "units.metal", "gemv.metal"});
    const mtl::Pipeline p = cx.pipeline(lib, "xprep");
    const mtl::Buffer x = cx.alloc(4096), xp = cx.alloc(4096);
    XArg xa{16, 1};
    const mtl::Buffer scal = cx.alloc(8);
    std::memcpy(scal.ptr, &xa, 8);
    std::printf("device %s, dispatches read/write 64 B buffers\n", cx.device_name().c_str());

    for (const uint32_t grid : {1u, 48u}) {
        for (const int n : {256, 1189}) {
            for (const int every : {1, 8, 0}) {   // barrier every k dispatches (0: none; run_graph skips i == 0)
                std::vector<mtl::Dispatch> ds;
                char label[96];
                std::snprintf(label, sizeof label, "run_graph grid %u every %d", grid, every);
                for (int i = 0; i < n; ++i) {
                    mtl::Dispatch d = prep(p, x, xp, grid, nullptr);
                    d.barrier = every && i % every == 0;
                    ds.push_back(d);
                }
                row(label, ds, 21);
            }
        }
    }
    {   // the same 1189-dispatch chain through the ICB replay path (setBarrier between commands)
        std::vector<mtl::Dispatch> ds;
        for (int i = 0; i < 1189; ++i) ds.push_back(prep(p, x, xp, 1, &scal));
        row("icb grid 1 (setBarrier)", ds, 11, true);
    }
    {   // serial encoder, Metal's own dispatch serialization, no explicit barriers at all
        std::vector<mtl::Dispatch> ds;
        for (int i = 0; i < 1189; ++i) ds.push_back(prep(p, x, xp, 1, nullptr));
        cx.warm(ds, 0.2);
        std::vector<double> t;
        for (int r = 0; r < 21; ++r) t.push_back(cx.run(ds, 1, false, false));
        const double m = med(t);
        std::printf("%-28s n %4d  %8.2f ms total  %6.2f us/dispatch\n", "run serial encoder", 1189, m * 1e3, m * 1e6 / 1189);
    }
    {   // the per-cut handshake itself: 1189 tiny dispatches split at N cuts (the window has 48)
        std::vector<mtl::Dispatch> ds;
        for (int i = 0; i < 1189; ++i) ds.push_back(prep(p, x, xp, 1, nullptr));
        const mtl::Buffer sb = cx.alloc(64);
        std::memset(sb.ptr, 7, 64);
        for (const int ncut : {1, 12, 48, 96}) {
            std::vector<int> cuts;
            for (int s = 1; s <= ncut; ++s) cuts.push_back((int) ((long long) 1189 * s / (ncut + 1)));
            char label[96];
            std::snprintf(label, sizeof label, "seg nop cuts %d", ncut);
            seg_row(label, ds, cuts, cut_nop, nullptr, 11);
            std::snprintf(label, sizeof label, "seg peek cuts %d", ncut);
            seg_row(label, ds, cuts, cut_peek, sb.ptr, 11);
        }
    }
    {   // does the per-cut stall scale with what the segments DO?  The engine's segments run ~1 ms of real,
        // indirect-dispatched work; round 23's nop segments are 25 x 64 B dispatches.  Engine idle is 660 us/cut
        // vs the bench's 120-230, so one of {queue stop/resume with real work, indirect re-arming} is the rest.
        std::vector<mtl::Dispatch> ds48, dsind;
        const mtl::Buffer gb = cx.alloc(12);   // [1,1,1] threadgroups: the indirect twin of the grid-1 rows
        uint32_t* g = (uint32_t*) gb.ptr;
        g[0] = g[1] = g[2] = 1;
        for (int i = 0; i < 1189; ++i) {
            mtl::Dispatch d = prep(p, x, xp, 48, nullptr);
            ds48.push_back(d);
            d = prep(p, x, xp, 1, nullptr);
            d.indirect = &gb;
            dsind.push_back(d);
        }
        std::vector<int> cuts;
        for (int s = 1; s <= 48; ++s) cuts.push_back((int) ((long long) 1189 * s / 49));
        seg_row("seg grid48 nop cuts 48", ds48, cuts, cut_nop, nullptr, 11);
        seg_row("seg ind nop cuts 48", dsind, cuts, cut_nop, nullptr, 11);
        const mtl::Buffer pb = cx.alloc(64);
        std::memset(pb.ptr, 7, 64);
        seg_row("seg ind peek cuts 48", dsind, cuts, cut_peek, pb.ptr, 11);
        row("run_graph ind every 1", dsind, 21);   // the no-cut baseline of the ind row (med of run_graph)
    }
    {   // the queue-free handshake (cut_seq/cut_spin in units.metal, run_graph_flag): same chain and cuts as
        // the seg rows but device counters instead of MTLSharedEvents.  The engine decision rides on these
        // rows: the cut idle goes if us/cut drops from 230-400 to tens.  sleep300: does the host turn now
        // still serialize behind the spin?
        const mtl::Pipeline pseq = cx.pipeline(lib, "cut_seq"), pspin = cx.pipeline(lib, "cut_spin");
        const mtl::Buffer hb = cx.alloc(12);   // seq, rel, seen
        int* ctr = hb.as<int>();
        mtl::Dispatch sd, td;
        sd.p = &pseq; sd.args = {mtl::Arg::B(hb, 0)}; sd.tg[0] = 32; sd.grid[0] = 1; sd.barrier = false;
        td.p = &pspin; td.args = {mtl::Arg::B(hb, 4), mtl::Arg::B(hb, 8)}; td.tg[0] = 32; td.grid[0] = 1; td.barrier = false;
        std::vector<mtl::Dispatch> ds;
        for (int i = 0; i < 1189; ++i) ds.push_back(prep(p, x, xp, 1, nullptr));
        const mtl::Buffer sb2 = cx.alloc(64);
        std::memset(sb2.ptr, 7, 64);
        for (const int ncut : {12, 48, 96}) {
            std::vector<int> cuts;
            for (int s = 1; s <= ncut; ++s) cuts.push_back((int) ((long long) 1189 * s / (ncut + 1)));
            char label[96];
            std::snprintf(label, sizeof label, "flag nop cuts %d", ncut);
            flag_row(label, ds, cuts, sd, td, ctr, cut_nop, nullptr, 11);
            std::snprintf(label, sizeof label, "flag peek cuts %d", ncut);
            flag_row(label, ds, cuts, sd, td, ctr, cut_peek, sb2.ptr, 11);
            if (ncut == 48) flag_row("flag sleep300 cuts 48", ds, cuts, sd, td, ctr, cut_sleep, nullptr, 11);
        }
    }
    return 0;
}
