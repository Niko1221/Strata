// sycl/probe/bw_probe.cpp - copy bandwidth for each kind of host memory, and how fast a kernel reads host memory in
// place.
//
// The engine moves expert blobs from host memory into VRAM (the streamed expert arena, --stream-experts) and reads
// host memory in place from kernels (the pinned mirror: experts the VRAM does not hold, read over PCIe).  Which host
// memory each should use is a measurement.  On Linux a B70 copied 14.4 GB/s from a mapping registered with
// prepare_for_device_copy and 4.2 GB/s from pageable memory (XeStrata, bench/results/2026-09-30-b70); kernels read
// pinned host memory at 21.9 GB/s hot and 3.1 GB/s on first touch (da3dsoul); llama.cpp's SYCL notes say a host USM
// allocation above 2 GiB serializes copies with compute.  None of that has been measured on the Windows driver, where
// upstream's VirtualAlloc memory (src/core/pinned.cu) is the arena the engine already has.
//
//   bw_probe [total MiB] [piece MiB]     default 256 MiB in 16 MiB pieces; 5 timed passes after one warm-up pass
//
// Four kinds of host memory, H2D and D2H each (GB/s: min / median / max over the passes), the bytes checked once per
// kind; then a kernel summing the whole buffer from host USM (first touch and hot) and from VRAM, for reference.
// VERDICT: the host memory for the expert arena (a copy source) and for the pinned mirror (read by kernels).
#include "probe_common.hpp"

#include <new>

namespace {

constexpr double kWaitSeconds = 60.0;
constexpr int kPasses = 5;

struct Rate {
    double min = 0, med = 0, max = 0;
    bool ok = false;
    std::string why;
};

// `bytes` from src to dst in `piece`-sized memcpys on the in-order queue, one wait per pass, the host clock around it
// (device events are not trusted for this: upstream's HIP PCIe probe read impossible speeds from them on Windows).
Rate copy_rate(sycl::queue& q, void* dst, const void* src, size_t bytes, size_t piece) {
    Rate r;
    std::vector<double> gbs;
    try {
        for (int pass = -1; pass < kPasses; ++pass) {   // pass -1 warms up: page faults, first-touch mapping
            const auto t0 = probe::clk::now();
            for (size_t off = 0; off < bytes; off += piece)
                q.memcpy((uint8_t*) dst + off, (const uint8_t*) src + off, std::min(piece, bytes - off));
            probe::Watchdog wd(kWaitSeconds, "the copies");
            q.wait();
            if (pass >= 0) gbs.push_back(probe::gb_per_s(bytes, probe::s_since(t0)));
        }
    } catch (const sycl::exception& e) {
        r.why = e.what();
        return r;
    }
    std::sort(gbs.begin(), gbs.end());
    r.min = gbs.front();
    r.med = gbs[gbs.size() / 2];
    r.max = gbs.back();
    r.ok = true;
    return r;
}

// the OS's own pages, as upstream's pinned.cu allocates them (VirtualAlloc; 4 KiB-aligned malloc elsewhere)
uint8_t* os_alloc(size_t bytes) {
#if defined(_WIN32)
    return (uint8_t*) VirtualAlloc(nullptr, bytes, MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE);
#else
    void* p = nullptr;
    return posix_memalign(&p, 4096, bytes) == 0 ? (uint8_t*) p : nullptr;
#endif
}
void os_free(uint8_t* p) {
#if defined(_WIN32)
    VirtualFree(p, 0, MEM_RELEASE);
#else
    std::free(p);
#endif
}

// prepare_for_device_copy (sycl_ext_oneapi_copy_optimize): compute-runtime's import-external-pointer, which makes an
// externally allocated range usable as a copy source or destination, not for kernel loads.  In oneAPI 2026.1.1
// (XeStrata built with it); it throws where the driver refuses.
bool register_copy(uint8_t* p, size_t bytes, const sycl::context& ctx, std::string& why) {
    try {
        probe::syclex::prepare_for_device_copy(p, bytes, ctx);
        return true;
    } catch (const sycl::exception& e) {
        why = std::string("prepare_for_device_copy threw: ") + e.what();
        return false;
    } catch (const std::exception& e) {
        why = std::string("prepare_for_device_copy threw: ") + e.what();
        return false;
    }
}
void unregister_copy(uint8_t* p, const sycl::context& ctx) {
    try {
        probe::syclex::release_from_device_copy(p, ctx);
    } catch (const std::exception& e) {
        std::printf("  release_from_device_copy threw: %s\n", e.what());
    }
}

// the bytes, once per kind: a pattern through the device and back into another host buffer
bool bytes_ok(sycl::queue& q, uint8_t* host, uint8_t* dev, std::vector<uint8_t>& scratch, size_t bytes) {
    for (size_t i = 0; i < bytes; ++i) host[i] = (uint8_t) ((i >> 12) + (i & 0xff) + 0x5a);
    std::memset(scratch.data(), 0, bytes);
    try {
        probe::Watchdog wd(kWaitSeconds, "the byte check's copies");
        q.memcpy(dev, host, bytes);
        q.memcpy(scratch.data(), dev, bytes).wait();
    } catch (const sycl::exception& e) {
        std::printf("  the byte check threw: %s\n", e.what());
        return false;
    }
    return std::memcmp(scratch.data(), host, bytes) == 0;
}

struct ReadRate {
    double first = 0, hot = 0;
    bool ok = false, correct = true;
    std::string why;
};

// A kernel summing `words` 32-bit words at `src` into one 64-bit result through a reduction; the host's sum checks it,
// so the loads cannot be dropped.  262,144 work-items in a grid-stride loop: neighbouring items read neighbouring
// words, so the loads coalesce.  Pass 0 is the first touch; the others are hot.
ReadRate kernel_read(sycl::queue& q, const uint32_t* src, uint64_t words, uint64_t expect, uint64_t* result) {
    ReadRate r;
    constexpr size_t wg = 256, items = 1024 * wg;
    std::vector<double> hot;
    try {
        for (int pass = 0; pass < 4; ++pass) {
            const auto t0 = probe::clk::now();
            {
                probe::Watchdog wd(kWaitSeconds, "the reading kernel");
                q.parallel_for(sycl::nd_range<1>(items, wg),
                               sycl::reduction(result, sycl::plus<uint64_t>(),
                                               sycl::property_list{sycl::property::reduction::initialize_to_identity{}}),
                               [=](sycl::nd_item<1> it, auto& sum) {
                                   uint64_t s = 0;
                                   for (uint64_t i = it.get_global_id(0); i < words; i += items) s += src[i];
                                   sum += s;
                               })
                    .wait();
            }
            const double gbs = probe::gb_per_s(words * 4, probe::s_since(t0));
            uint64_t got = 0;
            q.memcpy(&got, result, sizeof got).wait();
            if (got != expect) r.correct = false;
            if (pass == 0) r.first = gbs;
            else hot.push_back(gbs);
        }
        std::sort(hot.begin(), hot.end());
        r.hot = hot[hot.size() / 2];
        r.ok = true;
    } catch (const sycl::exception& e) {
        r.why = e.what();
    }
    return r;
}

struct Kind {
    const char* label = "";
    uint8_t* p = nullptr;
    bool present = false;   // allocated (and registered, where that is the kind)
    std::string note;
    Rate h2d, d2h;
    bool bytes = false;
};

void print_rate(const char* dir, const Rate& r) {
    if (r.ok) std::printf("    %s %6.2f / %6.2f / %6.2f GB/s (min / median / max)\n", dir, r.min, r.med, r.max);
    else std::printf("    %s threw: %s\n", dir, r.why.c_str());
}

}  // namespace

int main(int argc, char** argv) {
    const size_t total_mib = argc > 1 ? (size_t) std::atoll(argv[1]) : 256;
    const size_t piece_mib = argc > 2 ? (size_t) std::atoll(argv[2]) : 16;
    const size_t bytes = total_mib << 20, piece = piece_mib << 20;
    if (bytes == 0 || piece == 0 || bytes % 4 != 0) {
        std::printf("usage: bw_probe [total MiB] [piece MiB]\n");
        return 2;
    }
    probe::Gpu gpu;
    if (!probe::start("bw_probe", gpu)) return 2;
    try {
        sycl::queue q = probe::make_queue(gpu);
        const sycl::context ctx = q.get_context();
        uint8_t* dev = (uint8_t*) sycl::malloc_device(bytes, q);
        if (!dev) {
            probe::verdict("fail", "malloc_device of %zu MiB refused", total_mib);
            return 2;
        }
        std::vector<uint8_t> scratch(bytes);
        std::printf("%zu MiB in %zu MiB pieces, %d timed passes after a warm-up pass, host clock around queue.wait()\n\n",
                    total_mib, piece_mib, kPasses);

        Kind kinds[4];
        kinds[0].label = "(a) pageable new[]";
        kinds[1].label = "(b) sycl::malloc_host";
#if defined(_WIN32)
        kinds[2].label = "(c) VirtualAlloc + prepare_for_device_copy";
        kinds[3].label = "(d) VirtualAlloc, not registered";
#else
        kinds[2].label = "(c) posix_memalign + prepare_for_device_copy";
        kinds[3].label = "(d) posix_memalign, not registered";
#endif
        kinds[0].p = new (std::nothrow) uint8_t[bytes];
        kinds[0].present = kinds[0].p != nullptr;
        if (kinds[0].present) std::memset(kinds[0].p, 1, bytes);   // touched: pageable but resident
        kinds[1].p = (uint8_t*) sycl::malloc_host(bytes, q);
        kinds[1].present = kinds[1].p != nullptr;
        if (!kinds[1].present) kinds[1].note = "malloc_host refused";
        kinds[2].p = os_alloc(bytes);
        if (kinds[2].p) {
            std::memset(kinds[2].p, 1, bytes);
            kinds[2].present = register_copy(kinds[2].p, bytes, ctx, kinds[2].note);
        } else {
            kinds[2].note = "the OS allocation failed";
        }
        kinds[3].p = os_alloc(bytes);
        kinds[3].present = kinds[3].p != nullptr;
        if (kinds[3].present) std::memset(kinds[3].p, 1, bytes);

        for (Kind& k : kinds) {
            std::printf("  %s\n", k.label);
            if (!k.present) {
                std::printf("    not measured: %s\n", k.note.c_str());
                continue;
            }
            k.h2d = copy_rate(q, dev, k.p, bytes, piece);
            print_rate("H2D", k.h2d);
            k.d2h = copy_rate(q, k.p, dev, bytes, piece);
            print_rate("D2H", k.d2h);
            k.bytes = bytes_ok(q, k.p, dev, scratch, bytes);
            std::printf("    bytes through the device and back: %s\n", k.bytes ? "identical" : "DIFFER");
        }

        // a kernel reading host USM in place (the pinned mirror's access), then VRAM for the reference
        ReadRate host_read, dev_read;
        if (kinds[1].present) {
            const uint64_t words = bytes / 4;
            const uint32_t* src = (const uint32_t*) kinds[1].p;
            for (uint64_t i = 0; i < words; ++i) ((uint32_t*) kinds[1].p)[i] = (uint32_t) (i * 2654435761u);
            uint64_t expect = 0;
            for (uint64_t i = 0; i < words; ++i) expect += src[i];
            uint64_t* result = sycl::malloc_device<uint64_t>(1, q);
            std::printf("\n  a kernel summing the %zu MiB in place\n", total_mib);
            host_read = kernel_read(q, src, words, expect, result);
            if (host_read.ok)
                std::printf("    from malloc_host: first touch %.2f GB/s, hot %.2f GB/s, sum %s\n", host_read.first,
                            host_read.hot, host_read.correct ? "correct" : "WRONG");
            else
                std::printf("    from malloc_host: threw: %s\n", host_read.why.c_str());
            q.memcpy(dev, kinds[1].p, bytes).wait();
            dev_read = kernel_read(q, (const uint32_t*) dev, words, expect, result);
            if (dev_read.ok)
                std::printf("    from VRAM:        first %.2f GB/s, hot %.2f GB/s, sum %s\n", dev_read.first, dev_read.hot,
                            dev_read.correct ? "correct" : "WRONG");
            else
                std::printf("    from VRAM:        threw: %s\n", dev_read.why.c_str());
            sycl::free(result, q);
        }

        if (kinds[2].present) unregister_copy(kinds[2].p, ctx);
        if (kinds[2].p) os_free(kinds[2].p);
        if (kinds[3].p) os_free(kinds[3].p);
        if (kinds[1].p) sycl::free(kinds[1].p, q);
        delete[] kinds[0].p;
        sycl::free(dev, q);

        // the verdict: the arena is a copy source, so the fastest correct H2D wins; the mirror is read by kernels, so
        // it is host USM whatever the figure (registered memory cannot be addressed by a kernel)
        const Kind& usm = kinds[1];
        const Kind& reg = kinds[2];
        const bool usm_ok = usm.present && usm.h2d.ok && usm.bytes;
        const bool reg_ok = reg.present && reg.h2d.ok && reg.bytes;
        bool bad_bytes = false;
        for (const Kind& k : kinds) bad_bytes = bad_bytes || (k.present && !k.bytes);
        std::printf("\n");
        if (!usm_ok && !reg_ok) {
            probe::verdict("fail", "neither malloc_host nor registered memory copied correctly (%s; %s)",
                           usm.present ? (usm.h2d.ok ? "malloc_host bytes differ" : usm.h2d.why.c_str()) : usm.note.c_str(),
                           reg.present ? (reg.h2d.ok ? "registered bytes differ" : reg.h2d.why.c_str()) : reg.note.c_str());
            return 1;
        }
        const char* usm_problem = usm.present ? (usm.h2d.ok ? "bytes differ" : usm.h2d.why.c_str()) : usm.note.c_str();
        const char* reg_problem = reg.present ? (reg.h2d.ok ? "bytes differ" : reg.h2d.why.c_str()) : reg.note.c_str();
        char arena[320];
        if (reg_ok && (!usm_ok || reg.h2d.med >= 0.9 * usm.h2d.med)) {
            if (usm_ok)
                std::snprintf(arena, sizeof arena, "arena: VirtualAlloc + prepare_for_device_copy (H2D %.1f GB/s, "
                              "malloc_host %.1f)", reg.h2d.med, usm.h2d.med);
            else
                std::snprintf(arena, sizeof arena, "arena: VirtualAlloc + prepare_for_device_copy (H2D %.1f GB/s; "
                              "malloc_host: %s)", reg.h2d.med, usm_problem);
        } else if (reg_ok) {
            std::snprintf(arena, sizeof arena, "arena: malloc_host in chunks of at most 2 GiB (H2D %.1f GB/s, registered "
                          "%.1f)", usm.h2d.med, reg.h2d.med);
        } else {
            std::snprintf(arena, sizeof arena, "arena: malloc_host in chunks of at most 2 GiB (H2D %.1f GB/s; registered "
                          "memory: %s)", usm.h2d.med, reg_problem);
        }
        char mirror[160];
        if (host_read.ok)
            std::snprintf(mirror, sizeof mirror, "mirror: malloc_host, kernels read it at %.1f GB/s hot (%.1f first touch)%s",
                          host_read.hot, host_read.first, host_read.correct ? "" : ", SUM WRONG");
        else
            std::snprintf(mirror, sizeof mirror, "mirror: malloc_host, but the kernel reading it failed (%s)", host_read.why.c_str());
        const double best_h2d = std::max(reg_ok ? reg.h2d.med : 0.0, usm_ok ? usm.h2d.med : 0.0);
        const bool pageable_wins = kinds[0].present && kinds[0].h2d.ok && kinds[0].h2d.med > 1.1 * best_h2d;
        const char* state = "ok";
        std::string why;
        if (bad_bytes || (host_read.ok && !host_read.correct)) {
            state = "partial";
            why = "; some bytes or a sum came back wrong";
        } else if (pageable_wins) {
            state = "partial";
            why = "; pageable memory copied faster than both: the driver is bouncing, measure again";
        } else if (best_h2d < 5.0) {
            state = "partial";
            why = "; under 5 GB/s on the best path (PCIe 4.0 x8 gives about 14): check the slot and the link";
        }
        probe::verdict(state, "%s; %s%s", arena, mirror, why.c_str());
        return 0;
    } catch (const sycl::exception& e) {
        probe::verdict("fail", "SYCL threw: %s", e.what());
        return 2;
    }
}
