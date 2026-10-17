// The probe behind --shared-expert-arena's measurements (bench/results/2026-10-13-community-r9700-shared-expert-arena):
// maps an arena-sized region one way, registers it with HIP, measures the CPU's reads, host->device copies and the
// GPU queue stops while memory is compacted.
//   arena_probe anon SIZE_GIB [flags]            anonymous memory + MADV_HUGEPAGE (the engine's default arena)
//   arena_probe anonhuge SIZE_GIB [flags]        anonymous MAP_HUGETLB
//   arena_probe tmpfs PATH SIZE_GIB [flags]      MAP_SHARED file: the first process allocates and fills it, a second maps it
//   arena_probe hostmalloc SIZE_GIB [flags]      hipHostMalloc, and its dma-buf export to a second process
//   arena_probe vmm SIZE_GIB [flags]             hipMemCreate on the host + POSIX fd export to a second process
// flags: --mlock --collapse --register --hold SECONDS (copies while you compact: echo 1 > /proc/sys/vm/compact_memory)
#include <hip/hip_runtime.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <csetjmp>
#include <csignal>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <fstream>
#include <random>
#include <sstream>
#include <string>
#include <sys/mman.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/wait.h>
#include <thread>
#include <unistd.h>
#include <vector>

#ifndef MADV_POPULATE_READ
#define MADV_POPULATE_READ 22
#endif
#ifndef MADV_COLLAPSE
#define MADV_COLLAPSE 25
#endif

#define CK(x)                                                                         \
    do {                                                                              \
        hipError_t e_ = (x);                                                          \
        if (e_ != hipSuccess) {                                                       \
            std::printf("FAIL %s: %s\n", #x, hipGetErrorString(e_));                  \
            return 2;                                                                 \
        }                                                                             \
    } while (0)

static const size_t MiB = 1ull << 20, GiB = 1ull << 30, EXPERT = 3 * MiB;  // one expert blob ~3 MB
static const int THREADS = 12;

static double now() {
    return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}
static std::string slurp(const std::string& p) {
    std::ifstream f(p);
    std::stringstream s;
    s << f.rdbuf();
    return s.str();
}
static uint64_t stamp(uint64_t page) { return page * 0x9E3779B97F4A7C15ull ^ 0xA5A5F00DCAFE1234ull; }

static void status_line(const char* tag) {
    std::istringstream s(slurp("/proc/self/status"));
    std::string l, out;
    while (std::getline(s, l))
        for (const char* k : {"VmRSS", "RssAnon", "RssShmem", "VmLck", "VmPin"})
            if (l.rfind(k, 0) == 0 && l[strlen(k)] == ':')
                out += std::string(" ") + k + "=" + std::to_string(std::stoll(l.substr(strlen(k) + 1)) / 1024) + "M";
    std::printf("[%s] status:%s\n", tag, out.c_str());
}

static void smaps_line(const char* tag, const void* p) {
    std::istringstream s(slurp("/proc/self/smaps"));
    std::string l, out;
    bool in = false;
    const uintptr_t a = (uintptr_t) p;
    while (std::getline(s, l)) {
        const size_t dash = l.find('-'), sp = l.find(' ');
        if (dash != std::string::npos && dash < sp) {   // a VMA header: "lo-hi perms off dev inode path"
            if (in) break;
            const uintptr_t lo = std::stoull(l.substr(0, dash), nullptr, 16);
            const uintptr_t hi = std::stoull(l.substr(dash + 1, sp - dash - 1), nullptr, 16);
            in = a >= lo && a < hi;
            if (in) out = " vma=" + std::to_string((hi - lo) / MiB) + "M" + " [" + l.substr(sp + 1) + "]";
            continue;
        }
        if (!in) continue;
        for (const char* k : {"Rss:", "AnonHugePages:", "ShmemPmdMapped:", "FilePmdMapped:", "Locked:", "Swap:"})
            if (l.rfind(k, 0) == 0) out += std::string(" ") + k + std::to_string(std::stoll(l.substr(strlen(k))) / 1024) + "M";
    }
    std::printf("[%s] smaps:%s\n", tag, out.c_str());
}

static long long kfd_evicted(pid_t pid) {
    long long sum = 0;
    for (int n = 0; n < 16; ++n) {
        const std::string id = slurp("/sys/class/kfd/kfd/topology/nodes/" + std::to_string(n) + "/gpu_id");
        if (id.empty() || std::stoll(id) == 0) continue;
        const std::string v = slurp("/sys/class/kfd/kfd/proc/" + std::to_string(pid) + "/stats_" + std::to_string(std::stoll(id)) + "/evicted_ms");
        if (!v.empty()) sum += std::stoll(v);
    }
    return sum;
}

static void fill(uint8_t* p, size_t n) {
    const size_t pages = n / 4096, per = (pages + THREADS - 1) / THREADS;
    std::vector<std::thread> th;
    for (int t = 0; t < THREADS; ++t)
        th.emplace_back([=] {
            for (size_t i = t * per; i < std::min(pages, (t + 1) * per); ++i) {
                std::memset(p + i * 4096, (int) (i & 0xff), 4096);
                *(uint64_t*) (p + i * 4096) = stamp(i);
            }
        });
    for (auto& x : th) x.join();
}

static long verify(const uint8_t* p, size_t n) {
    std::mt19937_64 r(42);
    long bad = 0;
    for (int k = 0; k < 200000; ++k) {
        const size_t i = r() % (n / 4096);
        if (*(const uint64_t*) (p + i * 4096) != stamp(i) || p[i * 4096 + 100] != (uint8_t) (i & 0xff)) ++bad;
    }
    return bad;
}

// CPU pool's access pattern: whole expert-sized blocks at random places, one load per cache line.
static double cpu_read(const uint8_t* p, size_t n, double secs) {
    std::atomic<uint64_t> bytes{0}, sink{0};
    const double t0 = now();
    std::vector<std::thread> th;
    for (int t = 0; t < THREADS; ++t)
        th.emplace_back([&, t] {
            std::mt19937_64 r(t * 7919 + 1);
            const size_t blocks = (n - EXPERT) / 4096;
            uint64_t s = 0, b = 0;
            while (now() - t0 < secs) {
                const uint64_t* q = (const uint64_t*) (p + (r() % blocks) * 4096);
                for (size_t i = 0; i < EXPERT / 8; i += 8) s += q[i];
                b += EXPERT;
            }
            bytes += b;
            sink += s;
        });
    for (auto& x : th) x.join();
    return bytes / (now() - t0) / 1e9;
}

// Warm-up pass (not counted), then three 3-second passes: each, the median, and the load average beside them.
static void reads(const char* tag, const uint8_t* p, size_t n) {
    (void) cpu_read(p, n, 2.0);
    std::vector<double> v;
    for (int r = 0; r < 3; ++r) v.push_back(cpu_read(p, n, 3.0));
    std::vector<double> m = v;
    std::sort(m.begin(), m.end());
    const std::string load = slurp("/proc/loadavg");
    std::printf("[%s] cpu read %.1f %.1f %.1f -> median %.1f GB/s (loadavg %s)\n", tag, v[0], v[1], v[2], m[1],
                load.substr(0, 14).c_str());
}

static void h2d(const char* tag, const uint8_t* p, size_t n, hipStream_t st, uint8_t* dev) {
    std::mt19937_64 r(1);
    const size_t blocks = (n - EXPERT) / 4096;
    long cnt = 0;
    double t0 = now();
    while (now() - t0 < 3.0) {
        for (int i = 0; i < 16; ++i)
            (void) hipMemcpyAsync(dev + (i % 64) * EXPERT, p + (r() % blocks) * 4096, EXPERT, hipMemcpyDefault, st);
        (void) hipStreamSynchronize(st);
        cnt += 16;
    }
    const double experts = cnt * EXPERT / (now() - t0) / 1e9;
    cnt = 0;
    t0 = now();
    while (now() - t0 < 3.0) {
        (void) hipMemcpyAsync(dev, p + (r() % (n / (256 * MiB))) * 256 * MiB, 256 * MiB, hipMemcpyDefault, st);
        (void) hipStreamSynchronize(st);
        ++cnt;
    }
    std::printf("[%s] h2d: experts(3MiB random) %.2f GB/s, 256MiB %.2f GB/s\n", tag, experts,
                cnt * 256.0 * MiB / (now() - t0) / 1e9);
}

static void stall_watch(const uint8_t* p, size_t n, double secs, hipStream_t st, uint8_t* dev) {
    std::mt19937_64 r(7);
    const size_t blocks = (n - EXPERT) / 4096;
    const long long ev0 = kfd_evicted(getpid());
    std::printf("HOLD\n");
    std::fflush(stdout);
    const double t0 = now();
    double sec0 = t0, sec_max = 0, worst = 0;
    long batches = 0;
    std::vector<double> lat;
    while (now() - t0 < secs) {
        const double a = now();
        for (int i = 0; i < 16; ++i)
            (void) hipMemcpyAsync(dev + (i % 64) * EXPERT, p + (r() % blocks) * 4096, EXPERT, hipMemcpyDefault, st);
        (void) hipStreamSynchronize(st);
        const double d = now() - a;
        lat.push_back(d);
        sec_max = std::max(sec_max, d);
        worst = std::max(worst, d);
        ++batches;
        if (now() - sec0 >= 1.0) {
            std::printf("[hold] t=%4.1f max_batch=%.1fms evicted=%lldms\n", now() - t0, sec_max * 1e3,
                        kfd_evicted(getpid()) - ev0);
            sec_max = 0;
            sec0 = now();
        }
    }
    std::sort(lat.begin(), lat.end());
    std::printf("[hold] batches=%ld median=%.2fms p99=%.2fms worst=%.1fms evicted_delta=%lldms\n", batches,
                lat[lat.size() / 2] * 1e3, lat[lat.size() * 99 / 100] * 1e3, worst * 1e3, kfd_evicted(getpid()) - ev0);
}

static sigjmp_buf jb;
static void on_fault(int) { siglongjmp(jb, 1); }
static bool cpu_ok(volatile uint8_t* p) {
    struct sigaction sa{}, o1{}, o2{};
    sa.sa_handler = on_fault;
    sigaction(SIGSEGV, &sa, &o1);
    sigaction(SIGBUS, &sa, &o2);
    bool ok = false;
    if (sigsetjmp(jb, 1) == 0) {
        p[0] = 1;
        (void) p[4096];
        ok = true;
    }
    sigaction(SIGSEGV, &o1, nullptr);
    sigaction(SIGBUS, &o2, nullptr);
    return ok;
}

static void send_fd(int sock, int fd) {
    char c = 'f', buf[CMSG_SPACE(sizeof(int))] = {};
    iovec io{&c, 1};
    msghdr m{};
    m.msg_iov = &io;
    m.msg_iovlen = 1;
    m.msg_control = buf;
    m.msg_controllen = sizeof buf;
    cmsghdr* h = CMSG_FIRSTHDR(&m);
    h->cmsg_level = SOL_SOCKET;
    h->cmsg_type = SCM_RIGHTS;
    h->cmsg_len = CMSG_LEN(sizeof(int));
    std::memcpy(CMSG_DATA(h), &fd, sizeof(int));
    if (sendmsg(sock, &m, 0) < 0) perror("sendmsg");
}
static int recv_fd(int sock) {
    char c, buf[CMSG_SPACE(sizeof(int))] = {};
    iovec io{&c, 1};
    msghdr m{};
    m.msg_iov = &io;
    m.msg_iovlen = 1;
    m.msg_control = buf;
    m.msg_controllen = sizeof buf;
    if (recvmsg(sock, &m, 0) <= 0) return -1;
    int fd = -1;
    std::memcpy(&fd, CMSG_DATA(CMSG_FIRSTHDR(&m)), sizeof(int));
    return fd;
}

struct Gpu {
    hipStream_t st{};
    uint8_t* dev{};
};
static int gpu_up(Gpu& g) {
    hipDeviceProp_t prop{};
    CK(hipSetDevice(0));
    CK(hipGetDeviceProperties(&prop, 0));
    std::printf("device: %s pci %02x:%02x.0\n", prop.name, prop.pciBusID, prop.pciDeviceID);
    CK(hipStreamCreate(&g.st));
    CK(hipMalloc((void**) &g.dev, 256 * MiB));
    return 0;
}

static hipMemAllocationProp host_prop() {
    hipMemAllocationProp prop{};
    prop.type = hipMemAllocationTypePinned;
    prop.location.type = hipMemLocationTypeHost;
    prop.location.id = 0;
    prop.requestedHandleType = hipMemHandleTypePosixFileDescriptor;
    return prop;
}

// Maps an allocation for the device and tries the CPU: through the VA itself and through mmap() of the fd.
static int vmm_map(const char* tag, hipMemGenericAllocationHandle_t h, size_t n, size_t g, int fd, void*& va,
                   uint8_t*& cpu) {
    CK(hipMemAddressReserve(&va, n, g, nullptr, 0));
    CK(hipMemMap(va, n, 0, h, 0));
    hipMemAccessDesc d{};
    d.location.type = hipMemLocationTypeDevice;
    d.location.id = 0;
    d.flags = hipMemAccessFlagsProtReadWrite;
    CK(hipMemSetAccess(va, n, &d, 1));
    hipMemAccessDesc dh{};
    dh.location.type = hipMemLocationTypeHost;
    dh.location.id = 0;
    dh.flags = hipMemAccessFlagsProtReadWrite;
    const hipError_t eh = hipMemSetAccess(va, n, &dh, 1);
    const bool through_va = cpu_ok((volatile uint8_t*) va);
    std::printf("[%s] host access desc: %s; CPU through the VA: %s\n", tag, hipGetErrorString(eh), through_va ? "yes" : "no");
    cpu = through_va ? (uint8_t*) va : nullptr;
    char link[256] = {};
    const std::string fdp = "/proc/self/fd/" + std::to_string(fd);
    (void) !readlink(fdp.c_str(), link, sizeof link - 1);
    void* m = mmap(nullptr, n, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    std::printf("[%s] fd -> %s; mmap(fd): %s\n", tag, link, m == MAP_FAILED ? std::strerror(errno) : "ok");
    if (!cpu && m != MAP_FAILED) cpu = (uint8_t*) m;
    return 0;
}

static int vmm_import(int sock, size_t n) {
    Gpu g;
    if (gpu_up(g)) return 2;
    const int fd = recv_fd(sock);
    hipMemAllocationProp prop = host_prop();
    size_t gran = 0;
    CK(hipMemGetAllocationGranularity(&gran, &prop, hipMemAllocationGranularityRecommended));
    hipMemGenericAllocationHandle_t h;
    double t = now();
    CK(hipMemImportFromShareableHandle(&h, (void*) (uintptr_t) fd, hipMemHandleTypePosixFileDescriptor));
    std::printf("[import] import %.3fs\n", now() - t);
    void* va = nullptr;
    uint8_t* cpu = nullptr;
    t = now();
    if (vmm_map("import", h, n, gran, fd, va, cpu)) return 2;
    std::printf("[import] map %.3fs\n", now() - t);
    if (cpu) {
        std::printf("[import] verify: %ld bad of 200000 pages\n", verify(cpu, n));
        smaps_line("import", cpu);
        std::printf("[import] cpu read %.1f GB/s\n", cpu_read(cpu, n, 3.0));
    }
    status_line("import");
    h2d("import", (const uint8_t*) va, n, g.st, g.dev);
    std::printf("[import] kfd evicted %lldms\n", kfd_evicted(getpid()));
    return 0;
}

// A second process takes a dma-buf of the first one's hipHostMalloc memory: CPU through mmap(), GPU through import.
static int dmabuf_import(int sock, size_t n) {
    Gpu g;
    if (gpu_up(g)) return 2;
    const int fd = recv_fd(sock);
    void* m = mmap(nullptr, n, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    std::printf("[dmabuf] mmap(fd): %s\n", m == MAP_FAILED ? std::strerror(errno) : "ok");
    if (m != MAP_FAILED) {
        std::printf("[dmabuf] verify: %ld bad of 200000 pages\n", verify((const uint8_t*) m, n));
        smaps_line("dmabuf", m);
        std::printf("[dmabuf] cpu read %.1f GB/s\n", cpu_read((const uint8_t*) m, n, 3.0));
    }
    hipExternalMemoryHandleDesc d{};
    d.type = hipExternalMemoryHandleTypeOpaqueFd;
    d.handle.fd = dup(fd);
    d.size = n;
    hipExternalMemory_t em{};
    hipError_t e = hipImportExternalMemory(&em, &d);
    std::printf("[dmabuf] hipImportExternalMemory: %s\n", hipGetErrorString(e));
    if (e == hipSuccess) {
        hipExternalMemoryBufferDesc b{};
        b.size = n;
        void* dp = nullptr;
        e = hipExternalMemoryGetMappedBuffer(&dp, em, &b);
        std::printf("[dmabuf] mapped buffer: %s\n", hipGetErrorString(e));
        if (e == hipSuccess) h2d("dmabuf", (const uint8_t*) dp, n, g.st, g.dev);
    }
    status_line("dmabuf");
    std::printf("[dmabuf] kfd evicted %lldms\n", kfd_evicted(getpid()));
    return 0;
}

static void spawn_with_fd(const char* mode, int fd, size_t n) {
    int sv[2];
    socketpair(AF_UNIX, SOCK_STREAM, 0, sv);
    const pid_t c = fork();
    if (c == 0) {
        close(sv[0]);
        char a1[16], a2[32];
        std::snprintf(a1, sizeof a1, "%d", sv[1]);
        std::snprintf(a2, sizeof a2, "%zu", n);
        execl("/proc/self/exe", "arena_probe", mode, a1, a2, (char*) nullptr);
        _exit(127);
    }
    close(sv[1]);
    send_fd(sv[0], fd);
    int ws = 0;
    waitpid(c, &ws, 0);
    std::printf("[%s] child exit %d\n", mode, WEXITSTATUS(ws));
}

int main(int argc, char** argv) {
    setvbuf(stdout, nullptr, _IOLBF, 0);
    if (argc < 3) return 1;
    const std::string mode = argv[1];
    if (mode == "vmm-import") return vmm_import(std::atoi(argv[2]), std::stoull(argv[3]));
    if (mode == "dmabuf-import") return dmabuf_import(std::atoi(argv[2]), std::stoull(argv[3]));
    int ai = 2;
    const std::string path = mode == "tmpfs" ? argv[ai++] : "";
    size_t n = std::stoull(argv[ai++]) * GiB;
    bool do_mlock = false, collapse = false, reg = false;
    double hold = 0;
    for (; ai < argc; ++ai) {
        const std::string a = argv[ai];
        if (a == "--mlock") do_mlock = true;
        else if (a == "--collapse") collapse = true;
        else if (a == "--register") reg = true;
        else if (a == "--hold") hold = std::atof(argv[++ai]);
    }
    Gpu g;
    if (gpu_up(g)) return 2;
    const char* tag = mode.c_str();
    uint8_t* p = nullptr;
    bool creator = true;

    if (mode == "vmm") {
        hipMemAllocationProp prop = host_prop();
        size_t gran = 0;
        CK(hipMemGetAllocationGranularity(&gran, &prop, hipMemAllocationGranularityRecommended));
        n = (n + gran - 1) / gran * gran;
        hipMemGenericAllocationHandle_t h;
        double t = now();
        CK(hipMemCreate(&h, n, &prop, 0));
        std::printf("[vmm] granularity %zu KiB; create %.2fs\n", gran >> 10, now() - t);
        int fd = -1;
        CK(hipMemExportToShareableHandle(&fd, h, hipMemHandleTypePosixFileDescriptor, 0));
        void* va = nullptr;
        uint8_t* cpu = nullptr;
        if (vmm_map("vmm", h, n, gran, fd, va, cpu)) return 2;
        if (cpu) {
            t = now();
            fill(cpu, n);
            std::printf("[vmm] fill %.2fs\n", now() - t);
            smaps_line("vmm", cpu);
            std::printf("[vmm] cpu read %.1f GB/s\n", cpu_read(cpu, n, 3.0));
        }
        status_line("vmm");
        h2d("vmm", (const uint8_t*) va, n, g.st, g.dev);
        int sv[2];
        socketpair(AF_UNIX, SOCK_STREAM, 0, sv);
        const pid_t c = fork();
        if (c == 0) {
            close(sv[0]);
            char a1[16], a2[32];
            std::snprintf(a1, sizeof a1, "%d", sv[1]);
            std::snprintf(a2, sizeof a2, "%zu", n);
            execl("/proc/self/exe", "arena_probe", "vmm-import", a1, a2, (char*) nullptr);
            _exit(127);
        }
        close(sv[1]);
        send_fd(sv[0], fd);
        int ws = 0;
        waitpid(c, &ws, 0);
        std::printf("[vmm] importer exit %d\n", WEXITSTATUS(ws));
        if (hold > 0) stall_watch((const uint8_t*) va, n, hold, g.st, g.dev);
        return 0;
    }

    if (mode == "hostmalloc") {
        double t = now();
        CK(hipHostMalloc((void**) &p, n, hipHostMallocDefault));
        std::printf("[hostmalloc] alloc %.2fs\n", now() - t);
        t = now();
        fill(p, n);
        std::printf("[hostmalloc] fill %.2fs\n", now() - t);
        smaps_line(tag, p);
        status_line(tag);
        std::printf("[hostmalloc] cpu read %.1f GB/s\n", cpu_read(p, n, 3.0));
        h2d(tag, p, n, g.st, g.dev);
        int fd = -1;
        const hipError_t e = hipMemGetHandleForAddressRange(&fd, p, n, hipMemRangeHandleTypeDmaBufFd, 0);
        std::printf("[hostmalloc] dma-buf export: %s fd=%d\n", hipGetErrorString(e), fd);
        if (e == hipSuccess && fd >= 0) spawn_with_fd("dmabuf-import", fd, n);
        if (hold > 0) stall_watch(p, n, hold, g.st, g.dev);
        return 0;
    }

    double t = now();
    if (mode == "anonhuge") {   // Strata's own MAP_HUGETLB path for the arena
        void* m = mmap(nullptr, n, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_HUGETLB, -1, 0);
        if (m == MAP_FAILED) {
            std::printf("FAIL MAP_HUGETLB: %s\n", std::strerror(errno));
            return 2;
        }
        p = (uint8_t*) m;
    } else if (mode == "anon") {
        void* raw = mmap(nullptr, n + 2 * MiB, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        p = (uint8_t*) (((uintptr_t) raw + 2 * MiB - 1) & ~(uintptr_t) (2 * MiB - 1));
        madvise(p, n, MADV_HUGEPAGE);
    } else {
        const int fd = open(path.c_str(), O_RDWR | O_CREAT | O_CLOEXEC, 0644);
        struct stat st{};
        fstat(fd, &st);
        creator = st.st_size == 0;
        if (creator && fallocate(fd, 0, 0, n) != 0) {
            std::printf("FAIL fallocate: %s\n", std::strerror(errno));
            return 2;
        }
        if (!creator && (size_t) st.st_size != n) {
            std::printf("FAIL size %lld\n", (long long) st.st_size);
            return 2;
        }
        std::printf("[%s] %s, fallocate %.2fs\n", tag, creator ? "creator" : "attacher", now() - t);
        p = (uint8_t*) mmap(nullptr, n, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
        madvise(p, n, MADV_HUGEPAGE);
    }
    t = now();
    if (creator) {
        fill(p, n);
        std::printf("[%s] fill %.2fs\n", tag, now() - t);
    } else {
        const int rc = madvise(p, n, MADV_POPULATE_READ);
        std::printf("[%s] populate %.2fs (%s)\n", tag, now() - t, rc ? std::strerror(errno) : "ok");
        std::printf("[%s] verify: %ld bad of 200000 pages\n", tag, verify(p, n));
    }
    if (collapse) {
        t = now();
        const int rc = madvise(p, n, MADV_COLLAPSE);
        std::printf("[%s] MADV_COLLAPSE %.2fs (%s)\n", tag, now() - t, rc ? std::strerror(errno) : "ok");
    }
    if (do_mlock) {
        t = now();
        const int rc = mlock(p, n);
        std::printf("[%s] mlock %.2fs (%s)\n", tag, now() - t, rc ? std::strerror(errno) : "ok");
    }
    if (reg) {
        t = now();
        CK(hipHostRegister(p, n, hipHostRegisterPortable));
        std::printf("[%s] hipHostRegister %.2fs\n", tag, now() - t);
    }
    smaps_line(tag, p);
    status_line(tag);
    reads(tag, p, n);
    if (reg) {
        for (int r = 0; r < 3; ++r) h2d(tag, p, n, g.st, g.dev);
        if (hold > 0) stall_watch(p, n, hold, g.st, g.dev);
        (void) hipHostUnregister(p);
    }
    return 0;
}
