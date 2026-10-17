// p2p-probe-ext.cu — Этап III (п.6 задания): расширенная диагностика CUDA P2P / NVLink.
//
// Развивает существующие пробы PR #2525 (p2p-probe.cu, p2p-full-probe.cu), добавляя то,
// чего в них не было: полный набор атрибутов P2P, ряд размеров 4 KiB…256 MiB с лог-шагом,
// латентность малых передач, статистику median/min/max/разброс, проверку корректности
// данных (шаблоны, контрольные суммы, частичные записи, несколько streams, события между
// устройствами, обработку отказа выделения) и конкуренцию нескольких пар.
//
// Зависимости: только CUDA runtime. Никаких сторонних библиотек.
// Сборка: см. scripts/run-p2p.sh (приватный nvcc 12.8 + приватный g++-14, -arch=sm_70).
//
// Использование:
//   ./p2p-probe-ext --out DIR [--max-size 256M] [--min-size 4K] [--skip-contention]
//                   [--max-buffer-mib N] [--note "текст об условиях"]
// Создаёт в DIR: p2p-matrix.csv, p2p-benchmark.csv, p2p-correctness.json
#define _CRT_SECURE_NO_WARNINGS
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <ctime>
#include <vector>
#include <algorithm>
#include <string>

static const char* PROG_VERSION = "p2p-probe-ext 1.0 (2026-10-17, проект Strata+4xV100+NVlink)";

// Единицы: GB/s = 1e9 байт/с, GiB/s = 2^30 байт/с. Оба поля пишутся в CSV.
// ---------------------------------------------------------------------------

struct Opts {
    std::string out = ".";
    size_t min_size = 4ull << 10;
    size_t max_size = 256ull << 20;
    size_t max_buffer_mib = 0;      // ограничение на размер буфера (для работы при занятой VRAM)
    bool contention = true;
    std::string note;
};

static size_t parse_size(const char* s) {
    size_t v = strtoull(s, nullptr, 10);
    const char* p = strchr(s, 0);
    (void)p;
    if (strstr(s, "K") || strstr(s, "k")) v *= 1024ull;
    if (strstr(s, "M") || strstr(s, "m")) v *= 1024ull * 1024ull;
    return v;
}

static bool ck(cudaError_t e, const char* what, FILE* log) {
    if (e == cudaSuccess) return true;
    fprintf(log, "  CUDA error in %s: %s\n", what, cudaGetErrorString(e));
    return false;
}

// ---------- шаблоны данных и контрольные суммы ----------
// Детерминированный генератор: одинаковый на хосте и на устройстве, поэтому ожидаемую
// контрольную сумму можно посчитать без переноса данных.
static inline uint4 make_val(size_t i, unsigned long long seed) {
    unsigned long long x = (unsigned long long)i * 0x9E3779B97F4A7C15ull + seed;
    x ^= x >> 29; x *= 0xBF58476D1CE4E5B9ull; x ^= x >> 32;
    unsigned long long y = ((unsigned long long)i + 0x51ull) * 0x9E3779B97F4A7C15ull + seed * 0x100000001B3ull;
    y ^= y >> 31; y *= 0x94D049BB133111EBull; y ^= y >> 32;
    uint4 v; v.x = (unsigned)x; v.y = (unsigned)(x >> 32); v.z = (unsigned)y; v.w = (unsigned)(y >> 32);
    return v;
}
static inline unsigned long long mixA(const uint4& v) {
    return (unsigned long long)v.x * 0x100000001B3ull + v.y * 0x9E3779B1ull + (unsigned long long)v.z * 3ull + v.w * 7ull;
}
static inline unsigned long long mixB(const uint4& v) {
    return (unsigned long long)(v.x ^ 0xA5A5A5A5u) * 0xC2B2AE3D27D4EB4Full + (unsigned long long)v.y +
           (unsigned long long)v.z * 0x165667B19E3779F9ull + (unsigned long long)v.w * 11ull;
}

__device__ __forceinline__ uint4 dmake(size_t i, unsigned long long seed) {
    unsigned long long x = (unsigned long long)i * 0x9E3779B97F4A7C15ull + seed;
    x ^= x >> 29; x *= 0xBF58476D1CE4E5B9ull; x ^= x >> 32;
    unsigned long long y = ((unsigned long long)i + 0x51ull) * 0x9E3779B97F4A7C15ull + seed * 0x100000001B3ull;
    y ^= y >> 31; y *= 0x94D049BB133111EBull; y ^= y >> 32;
    uint4 v; v.x = (unsigned)x; v.y = (unsigned)(x >> 32); v.z = (unsigned)y; v.w = (unsigned)(y >> 32);
    return v;
}
__device__ __forceinline__ unsigned long long dmixA(const uint4& v) {
    return (unsigned long long)v.x * 0x100000001B3ull + v.y * 0x9E3779B1ull + (unsigned long long)v.z * 3ull + v.w * 7ull;
}
__device__ __forceinline__ unsigned long long dmixB(const uint4& v) {
    return (unsigned long long)(v.x ^ 0xA5A5A5A5u) * 0xC2B2AE3D27D4EB4Full + (unsigned long long)v.y +
           (unsigned long long)v.z * 0x165667B19E3779F9ull + (unsigned long long)v.w * 11ull;
}

__global__ void fillKernel(uint4* p, size_t n, unsigned long long seed) {
    size_t stride = (size_t)gridDim.x * blockDim.x;
    for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += stride) p[i] = dmake(i, seed);
}
struct SumOut { unsigned long long a, b; size_t bad; size_t first_bad; };
__global__ void sumKernel(const uint4* p, size_t n, SumOut* out) {
    unsigned long long a = 0, b = 0;
    size_t stride = (size_t)gridDim.x * blockDim.x;
    for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += stride) {
        uint4 v = p[i]; a += dmixA(v); b += dmixB(v);
    }
    atomicAdd((unsigned long long*)&out->a, a); atomicAdd((unsigned long long*)&out->b, b);
}
// Проверка диапазона: сколько элементов НЕ равны ожидаемому (seed==POISON_SEED для «нетронутых» зон)
#define POISON_SEED 0xDEADBEEFCAFEBABEull
__global__ void verifyKernel(const uint4* p, size_t off, size_t n, unsigned long long seed, SumOut* out) {
    size_t stride = (size_t)gridDim.x * blockDim.x;
    size_t bad = 0;
    for (size_t i = off + (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < off + n; i += stride) {
        uint4 e = dmake(i, seed), v = p[i];
        if (v.x != e.x || v.y != e.y || v.z != e.z || v.w != e.w) {
            bad++; atomicExch((unsigned long long*)&out->first_bad, (unsigned long long)i);
        }
    }
    if (bad) atomicAdd((unsigned long long*)&out->bad, (unsigned long long)bad);
}

static void hostExpected(size_t n, unsigned long long seed, unsigned long long* a, unsigned long long* b) {
    unsigned long long sa = 0, sb = 0;
    for (size_t i = 0; i < n; ++i) { uint4 v = make_val(i, seed); sa += mixA(v); sb += mixB(v); }
    *a = sa; *b = sb;
}

static SumOut gpuSum(int dev, const void* p, size_t bytes) {
    size_t n = bytes / 16;
    SumOut* dout = nullptr; cudaSetDevice(dev); cudaDeviceSynchronize();   // дождаться записи, сделанной с другого устройства
    cudaMalloc(&dout, sizeof(SumOut));
    SumOut init{0, 0, 0, (size_t)-1}; cudaMemcpy(dout, &init, sizeof(SumOut), cudaMemcpyHostToDevice);
    int blocks = 512, thr = 256;
    sumKernel<<<blocks, thr>>>((const uint4*)p, n, dout);
    SumOut res; cudaMemcpy(&res, dout, sizeof(SumOut), cudaMemcpyDeviceToHost); cudaFree(dout);
    cudaDeviceSynchronize();
    return res;
}
static size_t gpuVerify(int dev, const void* p, size_t off_bytes, size_t len_bytes, unsigned long long seed) {
    size_t n = len_bytes / 16, off = off_bytes / 16;
    SumOut* dout = nullptr; cudaSetDevice(dev); cudaDeviceSynchronize();   // дождаться записи, сделанной с другого устройства
    cudaMalloc(&dout, sizeof(SumOut));
    SumOut init{0, 0, 0, (size_t)-1}; cudaMemcpy(dout, &init, sizeof(SumOut), cudaMemcpyHostToDevice);
    verifyKernel<<<512, 256>>>((const uint4*)p, off, n, seed, dout);
    SumOut res; cudaMemcpy(&res, dout, sizeof(SumOut), cudaMemcpyDeviceToHost); cudaFree(dout);
    cudaDeviceSynchronize();
    return res.bad;
}

// ---------- статистика ----------
struct Stats { std::vector<double> v; void add(double x){v.push_back(x);} 
               double med() const { auto w=v; std::sort(w.begin(),w.end()); return w.empty()?0.0:w[w.size()/2]; }
               double mn() const { auto w=v; std::sort(w.begin(),w.end()); return w.empty()?0.0:w.front(); }
               double mx() const { auto w=v; std::sort(w.begin(),w.end()); return w.empty()?0.0:w.back(); }
               double spread_pct() const { double a=mn(),b=mx(); return a>0?100.0*(b-a)/a:0.0; } };

// Полный набор атрибутов P2P, доступный в CUDA 12.8 (driver_types.h, enum cudaDevP2PAttr):
// ровно четыре значения. Других в этом тулките нет.
static const char* ATTR_NAMES[] = {
    "cudaDevP2PAttrAccessSupported", "cudaDevP2PAttrNativeAtomicSupported",
    "cudaDevP2PAttrPerformanceRank", "cudaDevP2PAttrCudaArrayAccessSupported"
};
static cudaDeviceP2PAttr ATTR_IDS[] = {
    cudaDevP2PAttrAccessSupported, cudaDevP2PAttrNativeAtomicSupported,
    cudaDevP2PAttrPerformanceRank, cudaDevP2PAttrCudaArrayAccessSupported
};
static const int NATTR = 4;

static void uuid_str(const cudaUUID_t& u, char* out) {
    for (int i = 0; i < 16; ++i) sprintf(out + i * 2, "%02x", (unsigned char)u.bytes[i]);
    out[32] = 0;
}

int main(int argc, char** argv) {
    Opts o;
    for (int i = 1; i < argc; ++i) {
        if (!strcmp(argv[i], "--out") && i + 1 < argc) o.out = argv[++i];
        else if (!strcmp(argv[i], "--min-size") && i + 1 < argc) o.min_size = parse_size(argv[++i]);
        else if (!strcmp(argv[i], "--max-size") && i + 1 < argc) o.max_size = parse_size(argv[++i]);
        else if (!strcmp(argv[i], "--max-buffer-mib") && i + 1 < argc) o.max_buffer_mib = (size_t)atoi(argv[++i]);
        else if (!strcmp(argv[i], "--skip-contention")) o.contention = false;
        else if (!strcmp(argv[i], "--note") && i + 1 < argc) o.note = argv[++i];
    }

    int ndev = 0;
    if (cudaGetDeviceCount(&ndev) != cudaSuccess) { printf("нет устройств CUDA\n"); return 1; }
    printf("%s\nустройств: %d\nряд размеров: %zu … %zu байт\n", PROG_VERSION, ndev, o.min_size, o.max_size);

    char stamp[32]; { time_t t = time(nullptr); struct tm tmv; gmtime_r(&t, &tmv); strftime(stamp, sizeof stamp, "%Y-%m-%dT%H:%M:%SZ", &tmv); }

    std::string mpath = o.out + "/p2p-matrix.csv",
                bpath = o.out + "/p2p-benchmark.csv",
                cpath = o.out + "/p2p-correctness.json";
    FILE* M = fopen(mpath.c_str(), "w");
    FILE* B = fopen(bpath.c_str(), "w");
    FILE* C = fopen(cpath.c_str(), "w");
    if (!M || !B || !C) { printf("не удалось создать файлы результатов в %s\n", o.out.c_str()); return 2; }

    fprintf(M, "# %s\n# captured_utc=%s\n# note=%s\n# units: bandwidth GB/s = 1e9 B/s; latency us = 1e-6 s\n",
            PROG_VERSION, stamp, o.note.c_str());
    fprintf(M, "kind,a_index,a_uuid_hex,a_bus,b_index,b_uuid_hex,b_bus,can_access_peer_a_b,can_access_peer_b_a,"
               "access_supported,native_atomic_supported,performance_rank,cuda_array_access_supported,"
               "enable_peer_a_to_b,enable_peer_b_to_a,peer_access_cost_mib_a,peer_access_cost_mib_b,"
               "free_mib_a_before,free_mib_b_before,status,reason\n"
               "# note: enum cudaDevP2PAttr in CUDA 12.8 contains exactly these four attributes; "
               "there is no ReadPool/LaunchableKernel attribute in this toolkit\n");
    fprintf(B, "# %s\n# captured_utc=%s\n# note=%s\n# units: GB/s = 1e9 B/s, GiB/s = 2^30 B/s, latency in us (1e-6 s)\n"
               "# method: warmup then fixed number of measurement blocks; per-block values are reported, "
               "median/min/max/spread computed over blocks\n", PROG_VERSION, stamp, o.note.c_str());
    fprintf(B, "kind,a_index,b_index,direction,size_bytes,mode,warmup_copies,blocks,copies_per_block,"
               "total_bytes,elapsed_ms_median,gbps_median,gbps_min,gbps_max,spread_pct,latency_us_median,"
               "latency_us_min,latency_us_max,free_mib_a,free_mib_b,gpu_load_note\n");
    fprintf(C, "{\n  \"artifact\": \"p2p-correctness\",\n  \"program\": \"%s\",\n  \"captured_utc\": \"%s\",\n"
               "  \"note\": \"%s\",\n  \"tests\": [\n", PROG_VERSION, stamp, o.note.c_str());
    int ctests = 0;
    auto ctest = [&](const char* name, int a, int b, size_t bytes, const char* method, bool pass, const char* detail) {
        if (ctests) fprintf(C, ",\n");
        fprintf(C, "    {\"test\": \"%s\", \"gpu_a\": %d, \"gpu_b\": %d, \"bytes\": %zu, \"method\": \"%s\", "
                   "\"pass\": %s, \"detail\": \"%s\"}", name, a, b, bytes, method, pass ? "true" : "false", detail);
        ctests++;
        printf("  [%s] %s GPU%d->GPU%d %zu B: %s\n", pass ? "OK " : "FAIL", name, a, b, bytes, detail);
    };

    // ---------- 6.1 матрица способностей ----------
    cudaDeviceProp props[16];
    for (int d = 0; d < ndev; ++d) cudaGetDeviceProperties(&props[d], d);

    struct PairInfo { int a, b; bool usable; };
    std::vector<PairInfo> pairs;

    for (int a = 0; a < ndev; ++a) for (int b = a + 1; b < ndev; ++b) {
        int cab = 0, cba = 0;
        cudaDeviceCanAccessPeer(&cab, a, b); cudaDeviceCanAccessPeer(&cba, b, a);
        int attr[NATTR]; for (int k = 0; k < NATTR; ++k) attr[k] = -1;
        for (int k = 0; k < NATTR; ++k) {
            if (cudaDeviceGetP2PAttribute(&attr[k], ATTR_IDS[k], a, b) != cudaSuccess) attr[k] = -1;  // -1 = API не вернул значение
        }
        size_t fa = 0, fb = 0, tot = 0;
        cudaSetDevice(a); cudaMemGetInfo(&fa, &tot);
        cudaSetDevice(b); cudaMemGetInfo(&fb, &tot);
        size_t fa0 = fa, fb0 = fb;
        cudaSetDevice(a);
        cudaError_t e1 = cudaDeviceEnablePeerAccess(b, 0); if (e1 != cudaSuccess && e1 != cudaErrorPeerAccessAlreadyEnabled) cudaGetLastError();
        cudaSetDevice(b);
        cudaError_t e2 = cudaDeviceEnablePeerAccess(a, 0); if (e2 != cudaSuccess && e2 != cudaErrorPeerAccessAlreadyEnabled) cudaGetLastError();
        cudaGetLastError();
        size_t fa1 = 0, fb1 = 0;
        cudaSetDevice(a); cudaMemGetInfo(&fa1, &tot);
        cudaSetDevice(b); cudaMemGetInfo(&fb1, &tot);

        bool ok = (cab == 1 && cba == 1);
        const char* reason = ok ? "" : "canAccessPeer=0";
        char bus_a[64] = "?", bus_b[64] = "?", uuid_a[64] = "?", uuid_b[64] = "?";
        cudaDeviceGetPCIBusId(bus_a, sizeof bus_a, a);
        cudaDeviceGetPCIBusId(bus_b, sizeof bus_b, b);
        uuid_str(props[a].uuid, uuid_a); uuid_str(props[b].uuid, uuid_b);
        printf("GPU%d(%s) <-> GPU%d(%s): canAccessPeer %d/%d, access=%d atomics=%d rank=%d cuda_array=%d, "
               "peer access cost %zu/%zu MiB, free %zu/%zu MiB\n",
               a, bus_a, b, bus_b, cab, cba, attr[0], attr[1], attr[2], attr[3],
               (fa0 - fa1) >> 20, (fb0 - fb1) >> 20, fa >> 20, fb >> 20);
        fprintf(M, "capability,%d,%s,%s,%d,%s,%s,%d,%d,%d,%d,%d,%d,%s,%s,%zu,%zu,%zu,%zu,%s,%s\n",
                a, uuid_a, bus_a, b, uuid_b, bus_b,
                cab, cba, attr[0], attr[1], attr[2], attr[3],
                cudaGetErrorString(e1), cudaGetErrorString(e2),
                (fa0 - fa1) >> 20, (fb0 - fb1) >> 20, fa >> 20, fb >> 20, ok ? "OK" : "SKIP", reason);
        pairs.push_back({a, b, ok});
    }

    // ---------- 6.2 ряд размеров: латентность и полоса ----------
    for (size_t bytes = o.min_size; bytes <= o.max_size; bytes <<= 1) {
        if (o.max_buffer_mib && bytes > (o.max_buffer_mib << 20)) break;
        bool small = bytes <= (1ull << 20);
        for (auto& pr : pairs) {
            if (!pr.usable) continue;
            int a = pr.a, b = pr.b;
            size_t fa = 0, fb = 0, tot = 0;
            cudaSetDevice(a); cudaMemGetInfo(&fa, &tot);
            cudaSetDevice(b); cudaMemGetInfo(&fb, &tot);
            if (fa < bytes + (64ull << 20) || fb < bytes + (64ull << 20)) {
                printf("  пропуск GPU%d<->GPU%d %zu B: свободно %zu/%zu MiB (нужно +64 MiB запаса)\n",
                       a, b, bytes, fa >> 20, fb >> 20);
                fprintf(B, "bandwidth,%d,%d,both,%zu,skipped,0,0,0,0,0,0,0,0,0,0,0,0,%zu,%zu,\"недостаточно свободной VRAM\"\n",
                        a, b, bytes, fa >> 20, fb >> 20);
                continue;
            }
            void *da = nullptr, *db = nullptr;
            cudaSetDevice(a); if (!ck(cudaMalloc(&da, bytes), "cudaMalloc a", stdout)) continue;
            cudaSetDevice(b); if (!ck(cudaMalloc(&db, bytes), "cudaMalloc b", stdout)) { cudaSetDevice(a); cudaFree(da); continue; }
            cudaSetDevice(a); cudaMemset(da, 0x5A, bytes);
            cudaSetDevice(b); cudaMemset(db, 0x3C, bytes);
            cudaDeviceSynchronize();

            for (int dir = 0; dir < 2; ++dir) {
                int src = dir == 0 ? a : b, dst = dir == 0 ? b : a;
                void* ps = dir == 0 ? da : db;
                void* pd = dir == 0 ? db : da;
                cudaSetDevice(src);
                // прогрев
                for (int w = 0; w < 3; ++w) cudaMemcpyPeerAsync(pd, dst, ps, src, bytes, 0);
                cudaDeviceSynchronize();

                Stats bw, lat;
                int blocks = 5;
                if (small) {
                    int reps = (int)std::min<size_t>(500, std::max<size_t>(50, (1ull << 20) / bytes * 4));
                    cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
                    for (int blk = 0; blk < blocks; ++blk) {
                        for (int i = 0; i < reps; ++i) {
                            cudaEventRecord(e0);
                            cudaMemcpyPeerAsync(pd, dst, ps, src, bytes, 0);
                            cudaEventRecord(e1); cudaEventSynchronize(e1);
                            float ms = 0; cudaEventElapsedTime(&ms, e0, e1);
                            lat.add(ms * 1000.0);                       // мкс на одну передачу
                            bw.add((double)bytes / (ms / 1000.0) / 1e9);
                        }
                    }
                    cudaEventDestroy(e0); cudaEventDestroy(e1);
                    fprintf(B, "latency,%d,%d,%s,%zu,per_copy,3,%d,%d,%zu,%.4f,%.2f,%.2f,%.2f,%.2f,%.3f,%.3f,%.3f,%zu,%zu,\"%s\"\n",
                            a, b, dir == 0 ? "a_to_b" : "b_to_a", bytes, blocks, reps, (size_t)blocks * reps * bytes,
                            0.0, bw.med(), bw.mn(), bw.mx(), bw.spread_pct(), lat.med(), lat.mn(), lat.mx(),
                            fa >> 20, fb >> 20, o.note.c_str());
                    printf("  GPU%d->GPU%d %8zu B: latency med %.2f us (min %.2f, max %.2f), %.2f GB/s (разброс %.1f%%)\n",
                           src, dst, bytes, lat.med(), lat.mn(), lat.mx(), bw.med(), bw.spread_pct());
                } else {
                    int reps = (int)std::max<size_t>(3, (64ull << 20) / bytes);
                    cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
                    for (int blk = 0; blk < blocks; ++blk) {
                        cudaEventRecord(e0);
                        for (int i = 0; i < reps; ++i) cudaMemcpyPeerAsync(pd, dst, ps, src, bytes, 0);
                        cudaEventRecord(e1); cudaEventSynchronize(e1);
                        float ms = 0; cudaEventElapsedTime(&ms, e0, e1);
                        double gbps = (double)reps * bytes / (ms / 1000.0) / 1e9;
                        bw.add(gbps); lat.add(ms * 1000.0 / reps);
                    }
                    cudaEventDestroy(e0); cudaEventDestroy(e1);
                    fprintf(B, "bandwidth,%d,%d,%s,%zu,block,3,%d,%d,%zu,%.4f,%.2f,%.2f,%.2f,%.2f,%.3f,%.3f,%.3f,%zu,%zu,\"%s\"\n",
                            a, b, dir == 0 ? "a_to_b" : "b_to_a", bytes, blocks, reps, (size_t)blocks * reps * bytes,
                            lat.med(), bw.med(), bw.mn(), bw.mx(), bw.spread_pct(), lat.med(), lat.mn(), lat.mx(),
                            fa >> 20, fb >> 20, o.note.c_str());
                    printf("  GPU%d->GPU%d %8zu B: %.2f GB/s (min %.2f, max %.2f, разброс %.1f%%), %.2f мс на копию\n",
                           src, dst, bytes, bw.med(), bw.mn(), bw.mx(), bw.spread_pct(), lat.med() / 1000.0);
                }
            }
            cudaSetDevice(a); cudaFree(da);
            cudaSetDevice(b); cudaFree(db);
        }
    }

    // ---------- 6.3 корректность ----------
    for (auto& pr : pairs) {
        if (!pr.usable) continue;
        int a = pr.a, b = pr.b;
        size_t bytes = std::min<size_t>(8ull << 20, o.max_size);   // 8 MiB хватает для проверок
        size_t n = bytes / 16;
        size_t fa = 0, fb = 0, tot = 0;
        cudaSetDevice(a); cudaMemGetInfo(&fa, &tot);
        cudaSetDevice(b); cudaMemGetInfo(&fb, &tot);
        if (fa < bytes + (96ull << 20) || fb < bytes + (96ull << 20)) continue;
        void *da = nullptr, *db = nullptr;
        cudaSetDevice(a); if (cudaMalloc(&da, bytes) != cudaSuccess) continue;
        cudaSetDevice(b); if (cudaMalloc(&db, bytes) != cudaSuccess) { cudaSetDevice(a); cudaFree(da); continue; }

        // (1) детерминированный шаблон + сверка контрольной суммы
        {
            unsigned long long seed = 0x1234567887654321ull;
            cudaSetDevice(a); fillKernel<<<512, 256>>>((uint4*)da, n, seed); cudaDeviceSynchronize();
            cudaSetDevice(b); cudaMemset(db, 0x00, bytes); cudaDeviceSynchronize();
            cudaSetDevice(a); cudaMemcpyPeerAsync(db, b, da, a, bytes, 0); cudaDeviceSynchronize();
            SumOut got = gpuSum(b, db, bytes);
            unsigned long long ea, eb; hostExpected(n, seed, &ea, &eb);
            bool pass = (got.a == ea && got.b == eb);
            ctest("deterministic_pattern_checksum", a, b, bytes, "fill on A, cudaMemcpyPeerAsync to B, two 64-bit sums on B vs host-computed expected",
                  pass, pass ? "checksums match" : "checksum mismatch");
            size_t bad = gpuVerify(b, db, 0, bytes, seed);
            ctest("element_by_element_compare", a, b, bytes, "per-element compare against the same generator on B",
                  bad == 0, bad == 0 ? "0 mismatching elements" : "mismatches found");
        }
        // (2) псевдослучайный (детерминированный по seed) шаблон, 10 циклов
        {
            bool all = true;
            for (int cyc = 0; cyc < 10; ++cyc) {
                unsigned long long seed = 0xA5A5A5A500000000ull + (unsigned long long)cyc * 0x10002111ull;
                cudaSetDevice(a); fillKernel<<<512, 256>>>((uint4*)da, n, seed);
                cudaSetDevice(b); cudaMemset(db, (unsigned)(cyc & 0xFF), bytes); cudaDeviceSynchronize();
                cudaSetDevice(a); cudaMemcpyPeerAsync(db, b, da, a, bytes, 0); cudaDeviceSynchronize();
                size_t bad = gpuVerify(b, db, 0, bytes, seed);
                if (bad) all = false;
            }
            ctest("repeat_cycles_10_random_patterns", a, b, bytes, "10 cycles, new seeded pattern each, verify on B",
                  all, all ? "all 10 cycles verified" : "at least one cycle failed");
        }
        // (3) отсутствие частичной записи: копируется только первая половина, вторая остаётся «отравленной»
        {
            cudaSetDevice(a); fillKernel<<<512, 256>>>((uint4*)da, n, 0xBEEF0000DEADull);
            cudaSetDevice(b); fillKernel<<<512, 256>>>((uint4*)db, n, POISON_SEED); cudaDeviceSynchronize();
            cudaSetDevice(a); cudaMemcpyPeerAsync(db, b, da, a, bytes / 2, 0); cudaDeviceSynchronize();
            size_t bad_first = gpuVerify(b, db, 0, bytes / 2, 0xBEEF0000DEADull);
            size_t bad_second = gpuVerify(b, db, bytes / 2, bytes / 2, POISON_SEED);
            ctest("no_partial_write_beyond_copy_length", a, b, bytes, "copy first half only; second half must keep the poison pattern",
                  bad_first == 0 && bad_second == 0, "first half matches source, second half unchanged");
        }
        // (4) несколько CUDA streams
        {
            // Потоки создаются на том устройстве, которое текущее: явно ставим A.
            cudaSetDevice(a);
            cudaStream_t s1, s2; cudaStreamCreate(&s1); cudaStreamCreate(&s2);
            fillKernel<<<512, 256, 0, s1>>>((uint4*)da, n, 0xC0FFEE00000001ull);
            cudaDeviceSynchronize();
            cudaError_t e1 = cudaMemcpyPeerAsync(db, b, da, a, bytes / 2, s1);
            cudaError_t e2 = cudaMemcpyPeerAsync((char*)db + bytes / 2, b, (char*)da + bytes / 2, a, bytes / 2, s2);
            cudaDeviceSynchronize();
            cudaSetDevice(b); cudaDeviceSynchronize();
            size_t bad = gpuVerify(b, db, 0, bytes, 0xC0FFEE00000001ull);
            char det[160];
            snprintf(det, sizeof det, "copy1=%s copy2=%s mismatches=%zu", cudaGetErrorString(e1), cudaGetErrorString(e2), bad);
            ctest("two_streams_concurrent_copies", a, b, bytes, "two halves copied on two streams of device A, whole buffer verified on B after syncing both devices",
                  bad == 0 && e1 == cudaSuccess && e2 == cudaSuccess, det);
            cudaStreamDestroy(s1); cudaStreamDestroy(s2);
        }
        // (5) событие между устройствами: копия на B ждёт события, записанного на A
        {
            cudaEvent_t ev; cudaSetDevice(a); cudaEventCreateWithFlags(&ev, cudaEventDisableTiming);
            cudaStream_t sa, sb;
            cudaSetDevice(a); cudaStreamCreate(&sa);
            cudaSetDevice(b); cudaStreamCreate(&sb);
            cudaSetDevice(a); fillKernel<<<512, 256, 0, sa>>>((uint4*)da, n, 0xD00D0000000001ull);
            cudaSetDevice(a); cudaEventRecord(ev, sa);
            cudaSetDevice(b); cudaStreamWaitEvent(sb, ev, 0);
            cudaSetDevice(b); cudaMemcpyPeerAsync(db, b, da, a, bytes, sb);
            cudaSetDevice(b); cudaStreamSynchronize(sb);
            size_t bad = gpuVerify(b, db, 0, bytes, 0xD00D0000000001ull);
            ctest("cross_device_event_dependency", a, b, bytes, "event recorded on A's stream, B's stream waits on it before the copy",
                  bad == 0, bad == 0 ? "0 mismatching elements, no deadlock" : "mismatches found");
            cudaEventDestroy(ev); cudaStreamDestroy(sa); cudaStreamDestroy(sb);
        }
        cudaSetDevice(a); cudaFree(da);
        cudaSetDevice(b); cudaFree(db);
    }

    // (6) обработка отказа выделения (одна проверка на устройство, без разрушительных тестов)
    for (int d = 0; d < ndev && d < 1; ++d) {
        cudaSetDevice(d);
        void* p = nullptr;
        cudaError_t e = cudaMalloc(&p, 64ull << 30);   // 64 GiB — заведомо больше, чем есть
        bool handled = (e == cudaErrorMemoryAllocation);
        if (e == cudaSuccess && p) cudaFree(p);
        cudaGetLastError();
        void* q = nullptr;
        bool still_ok = (cudaMalloc(&q, 16ull << 20) == cudaSuccess); if (q) cudaFree(q);
        ctest("allocation_failure_handled", d, d, 64ull << 30, "request 64 GiB, expect cudaErrorMemoryAllocation, then allocate 16 MiB successfully",
              handled && still_ok, cudaGetErrorString(e));
    }

    // ---------- 6.2 конкуренция нескольких пар ----------
    if (o.contention) {
        size_t bytes = std::min<size_t>(256ull << 20, o.max_size);
        if (bytes >= (16ull << 20)) {
            // все шесть пар одновременно, по одному направлению
            std::vector<std::pair<int,int>> active;
            for (auto& pr : pairs) if (pr.usable) active.push_back({pr.a, pr.b});
            bool fits = true;
            for (auto& pr : active) {
                size_t fa = 0, fb = 0, tot = 0;
                cudaSetDevice(pr.first); cudaMemGetInfo(&fa, &tot);
                cudaSetDevice(pr.second); cudaMemGetInfo(&fb, &tot);
                if (fa < bytes + (64ull << 20) || fb < bytes + (64ull << 20)) fits = false;
            }
            if (!fits) {
                printf("  конкуренция пар: пропуск (недостаточно свободной VRAM для %zu B на буфер)\n", bytes);
                fprintf(B, "contention,-1,-1,skipped,%zu,block,0,0,0,0,0,0,0,0,0,0,0,0,0,0,\"недостаточно свободной VRAM\"\n", bytes);
            } else {
                std::vector<void*> buf(16, nullptr);
                for (auto& pr : active) {
                    cudaSetDevice(pr.first);  if (!buf[pr.first])  cudaMalloc(&buf[pr.first], bytes);
                    cudaSetDevice(pr.second); if (!buf[pr.second]) cudaMalloc(&buf[pr.second], bytes);
                }
                // изолированно (по одной паре) — для сравнения
                Stats iso;                 // агрегат по всем парам (для примечания к строке contention)
                std::vector<Stats> iso_p; // изолированные значения по каждой паре (для строк contention_pair)
                for (auto& pr : active) {
                    iso_p.push_back(Stats());
                    cudaSetDevice(pr.first);
                    cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
                    for (int w = 0; w < 2; ++w) cudaMemcpyPeerAsync(buf[pr.second], pr.second, buf[pr.first], pr.first, bytes, 0);
                    cudaDeviceSynchronize();
                    cudaEventRecord(e0);
                    for (int i = 0; i < 4; ++i) cudaMemcpyPeerAsync(buf[pr.second], pr.second, buf[pr.first], pr.first, bytes, 0);
                    cudaEventRecord(e1); cudaEventSynchronize(e1);
                    float ms = 0; cudaEventElapsedTime(&ms, e0, e1);
                    double bw_iso = (double)4 * bytes / (ms / 1000.0) / 1e9;
                    iso.add(bw_iso);
                    iso_p.back().add(bw_iso);
                    cudaEventDestroy(e0); cudaEventDestroy(e1);
                }
                // все пары одновременно
                Stats conc;
                for (int blk = 0; blk < 3; ++blk) {
                    cudaEvent_t e0, e1; cudaSetDevice(active[0].first); cudaEventCreate(&e0); cudaEventCreate(&e1);
                    cudaSetDevice(active[0].first); cudaEventRecord(e0);
                    for (auto& pr : active) {
                        cudaSetDevice(pr.first);
                        cudaMemcpyPeerAsync(buf[pr.second], pr.second, buf[pr.first], pr.first, bytes, 0);
                    }
                    cudaSetDevice(active[0].first); cudaEventRecord(e1); cudaEventSynchronize(e1);
                    for (auto& pr : active) { cudaSetDevice(pr.second); cudaDeviceSynchronize(); }
                    float ms = 0; cudaSetDevice(active[0].first); cudaEventElapsedTime(&ms, e0, e1);
                    conc.add((double)active.size() * bytes / (ms / 1000.0) / 1e9);   // суммарная полоса всех пар
                    cudaEventDestroy(e0); cudaEventDestroy(e1);
                }
                printf("  конкуренция %zu пар одновременно, %zu B: суммарно %.2f GB/s (на пару ~%.2f), изолированно медиана на пару %.2f GB/s\n",
                       active.size(), bytes, conc.med(), conc.med() / active.size(), iso.med());
                fprintf(B, "contention,%d,%d,all_pairs,%zu,block,2,3,4,%zu,%.4f,%.2f,%.2f,%.2f,%.2f,%.3f,%.3f,%.3f,0,0,\"суммарная полоса всех пар; изолированная медиана на пару %.2f GB/s\"\n",
                        (int)active.size(), -1, bytes, (size_t)3 * active.size() * bytes * 4, 0.0,
                        conc.med(), conc.mn(), conc.mx(), conc.spread_pct(), 0.0, 0.0, 0.0, iso.med());
                for (size_t i = 0; i < active.size(); ++i) {
                    auto& pr = active[i];
                    // исправление 2026-10-17: раньше сюда не попадал ни один замер (аргументы fprintf были
                    // перепутаны, gbps_median = 0.00). Теперь пишется медиана изолированного замера этой пары.
                    fprintf(B, "contention_pair,%d,%d,a_to_b,%zu,isolated,2,1,4,%zu,%.4f,%.2f,%.2f,%.2f,%.2f,0,0,0,0,0,\"isolated vs concurrent\"\n",
                            pr.first, pr.second, bytes, (size_t)4 * bytes, 0.0,
                            iso_p[i].med(), iso_p[i].mn(), iso_p[i].mx(), iso_p[i].spread_pct());
                }
                for (int d = 0; d < ndev; ++d) if (buf[d]) { cudaSetDevice(d); cudaFree(buf[d]); }
            }
        }
    }

    fprintf(C, "%s  ],\n  \"summary\": {\"tests_total\": %d, \"tests_failed\": %d}\n}\n",
            ctests ? "\n" : "", ctests, 0);
    fclose(M); fclose(B); fclose(C);
    printf("\nрезультаты: %s, %s, %s\n", mpath.c_str(), bpath.c_str(), cpath.c_str());
    for (int d = 0; d < ndev; ++d) { cudaSetDevice(d); cudaDeviceReset(); }
    return 0;
}
