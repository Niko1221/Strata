// sycl/probe/alloc_probe.cpp - how large may one device allocation be on this card under this driver, and one host
// USM allocation?
//
// Strata's DeviceArena makes one allocation for its whole VRAM plan (about 30 GiB on an Arc Pro B70).  On Linux one
// device allocation above a threshold that tracks host RAM plus swap is refused on a B70 (compute-runtime#998: 19.3
// GiB on a 16 GiB box), and llama.cpp caps its buffers at 60% of max_mem_alloc_size there; native-Windows reports
// kept every allocation under 4 GiB.  On the Windows B70 PC with 128 GiB of RAM every size up to the reported
// maximum (31.16 GiB) passed (2026-10-04).  The SYCL engine therefore tries one allocation first and falls back to
// chunks after a refusal (sycl/include/strata/core/device.hpp); this probe says which case the PC is, and whether
// STRATA_ARENA_CHUNK_MIB should be set in the config's "env".
//
//   alloc_probe [--device-only | --host-only] [--host-max GiB] [--no-fill]
//
// Device: one allocation of 1, 2, 4, 8, 12, 16, 20, 24, 28, 30 GiB and of max_mem_alloc_size, each filled by a kernel
// in 1 GiB pieces (a single fill of several GiB ran on the blitter and timed out on xe), one word per 64 MiB read
// back, then freed; the fill speed says whether the allocation is in VRAM (hundreds of GB/s) or spilled to host RAM
// (tens).  Host: malloc_host of 1, 2, 4, 8, 16, 24 and 31 GiB up to the RAM less 4 GiB (--host-max), touched once per
// 2 MiB.  Every wait is under a watchdog.  VERDICT ok when the whole reported maximum is one allocation, partial with
// the chunk size to set otherwise, fail when nothing above 1 GiB is granted.
#include "probe_common.hpp"

namespace {

constexpr uint64_t GiB = 1ull << 30;
constexpr double kWedgeSeconds = 60.0;

// true when the allocation was granted and filled; fill_gbps is the fill speed (0 with --no-fill)
bool try_device(sycl::queue& q, uint64_t bytes, bool fill, double& fill_gbps, std::string& why) {
    fill_gbps = 0.0;
    void* p = nullptr;
    try {
        p = sycl::malloc_device(bytes, q);
    } catch (const sycl::exception& e) {
        why = e.what();
        return false;
    } catch (const std::exception& e) {
        why = e.what();
        return false;
    }
    if (p == nullptr) { why = "malloc_device returned null"; return false; }
    bool ok = true;
    if (fill) {
        try {
            probe::Watchdog dog(kWedgeSeconds, "the fill kernel");
            const probe::clk::time_point t0 = probe::clk::now();
            uint64_t* d = (uint64_t*) p;
            const uint64_t words = bytes / 8;
            for (uint64_t off = 0; off < words; off += GiB / 8) {
                const uint64_t n = std::min<uint64_t>(GiB / 8, words - off);
                q.parallel_for(sycl::range<1>((size_t) n), [=](sycl::id<1> i) {
                    d[off + i[0]] = (off + i[0]) * 0x9E3779B97F4A7C15ull;
                });
                q.wait_and_throw();
            }
            fill_gbps = probe::gb_per_s(bytes, probe::s_since(t0));
            // one word per 64 MiB read back: the right pattern says the pages are the allocation's own
            const uint64_t step = (64ull << 20) / 8, samples = std::max<uint64_t>(1, words / step);
            std::vector<uint64_t> back((size_t) samples);
            uint64_t* dev_back = sycl::malloc_device<uint64_t>((size_t) samples, q);
            q.parallel_for(sycl::range<1>((size_t) samples), [=](sycl::id<1> i) { dev_back[i[0]] = d[i[0] * step]; });
            q.wait_and_throw();
            q.memcpy(back.data(), dev_back, samples * 8).wait_and_throw();
            sycl::free(dev_back, q);
            for (uint64_t i = 0; i < samples; ++i)
                if (back[(size_t) i] != (i * step) * 0x9E3779B97F4A7C15ull) {
                    why = "a word read back did not hold its pattern at +" + std::to_string((i * step * 8) >> 20) + " MiB";
                    ok = false;
                    break;
                }
        } catch (const sycl::exception& e) {
            why = std::string("the fill threw: ") + e.what();
            ok = false;
        }
    }
    sycl::free(p, q);
    return ok;
}

bool try_host(sycl::queue& q, uint64_t bytes, std::string& why) {
    void* p = nullptr;
    try {
        p = sycl::malloc_host(bytes, q);
    } catch (const std::exception& e) {
        why = e.what();
        return false;
    }
    if (p == nullptr) { why = "malloc_host returned null"; return false; }
    volatile uint8_t* b = (volatile uint8_t*) p;
    for (uint64_t off = 0; off < bytes; off += 2ull << 20) b[off] = (uint8_t) off;   // every 2 MiB page, once
    sycl::free(p, q);
    return true;
}

}  // namespace

int main(int argc, char** argv) {
    bool device = true, host = true, fill = true;
    double host_max_gib = 0.0;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--device-only") host = false;
        else if (a == "--host-only") device = false;
        else if (a == "--no-fill") fill = false;
        else if (a == "--host-max" && i + 1 < argc) host_max_gib = std::atof(argv[++i]);
        else { std::printf("alloc_probe [--device-only | --host-only] [--host-max GiB] [--no-fill]\n"); return 2; }
    }
    probe::Gpu gpu;
    if (!probe::start("alloc_probe", gpu)) return 2;
    probe::print_env("UR_L0_ENABLE_RELAXED_ALLOCATION_LIMITS");
    probe::print_env("STRATA_ARENA_CHUNK_MIB");
    try {
        sycl::queue q = probe::make_queue(gpu);
        uint64_t ceiling = 0;
        std::string first_refusal;
        if (device) {
            std::printf("\ndevice allocations (max_mem_alloc_size %.2f GiB, global %.2f GiB):\n", probe::gib(gpu.max_alloc),
                        probe::gib(gpu.global));
            std::vector<uint64_t> sizes;
            for (uint64_t g : {1u, 2u, 4u, 8u, 12u, 16u, 20u, 24u, 28u, 30u})
                if (g * GiB < gpu.max_alloc) sizes.push_back(g * GiB);
            sizes.push_back(gpu.max_alloc);
            if (fill) {   // a warm-up: the first allocation of a process pays the context's own set-up
                double gbps = 0; std::string why;
                try_device(q, GiB, true, gbps, why);
            }
            for (uint64_t bytes : sizes) {
                double gbps = 0; std::string why;
                const bool ok = try_device(q, bytes, fill, gbps, why);
                if (ok) {
                    ceiling = bytes;
                    if (fill) std::printf("  %6.2f GiB  ok, filled at %.0f GB/s\n", probe::gib(bytes), gbps);
                    else std::printf("  %6.2f GiB  ok\n", probe::gib(bytes));
                } else {
                    std::printf("  %6.2f GiB  REFUSED: %s\n", probe::gib(bytes), why.c_str());
                    if (first_refusal.empty()) first_refusal = why;
                    break;
                }
            }
        }
        uint64_t host_ceiling = 0;
        if (host) {
            const uint64_t ram = probe::physical_ram_bytes();
            uint64_t cap = host_max_gib > 0 ? (uint64_t) (host_max_gib * (double) GiB) : (ram > 4 * GiB ? ram - 4 * GiB : GiB);
            std::printf("\nhost USM allocations (RAM %.1f GiB, up to %.1f GiB):\n", probe::gib(ram), probe::gib(cap));
            for (uint64_t g : {1u, 2u, 4u, 8u, 16u, 24u, 31u}) {
                if (g * GiB > cap) break;
                std::string why;
                if (try_host(q, g * GiB, why)) { host_ceiling = g * GiB; std::printf("  %3u GiB  ok\n", g); }
                else { std::printf("  %3u GiB  REFUSED: %s\n", g, why.c_str()); break; }
            }
        }
        std::printf("\n");
        if (!device) {
            probe::verdict("ok", "host USM: one allocation up to %.0f GiB works", probe::gib(host_ceiling));
        } else if (ceiling >= gpu.max_alloc) {
            probe::verdict("ok", "one device allocation up to %.2f GiB (the reported maximum) works: leave "
                           "STRATA_ARENA_CHUNK_MIB unset (the engine makes one allocation first, chunks only after a "
                           "refusal); host USM: one allocation up to %.0f GiB",
                           probe::gib(ceiling), probe::gib(host_ceiling));
        } else if (ceiling > GiB) {
            probe::verdict("partial", "the largest single device allocation is %.2f GiB of a reported %.2f (%s): set "
                           "STRATA_ARENA_CHUNK_MIB=%llu in the config's \"env\" (60%% of it), and keep the VRAM plan "
                           "under it; host USM: up to %.0f GiB",
                           probe::gib(ceiling), probe::gib(gpu.max_alloc), first_refusal.c_str(),
                           (unsigned long long) (ceiling / 10 * 6 >> 20), probe::gib(host_ceiling));
        } else {
            probe::verdict("fail", "no device allocation above 1 GiB was granted (%s): the engine cannot place its arena",
                           first_refusal.c_str());
        }
    } catch (const sycl::exception& e) {
        probe::verdict("fail", "SYCL exception: %s", e.what());
        return 2;
    }
    return 0;
}
