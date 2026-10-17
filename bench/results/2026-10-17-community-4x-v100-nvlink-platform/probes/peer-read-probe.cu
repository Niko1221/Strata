// peer-read-probe.cu — Этап VIII (п. 11) / остаток Этапа III (§9.5 patch-recommendations.md):
// пропускная способность ПРЯМОГО доступа ядра к памяти, резидентной для другой карты.
//
// Зачем. Атрибут CudaArrayAccessSupported = 1 измерен (p2p-results.md), но транспорт при
// прямом доступе ядра не измерялся: измерен только cudaMemcpyPeerAsync (38.77 ГБ/с). Без
// этого числа выбор между P2a (буферы хенд-овера в VRAM) и P2b (чтение чужой VRAM ядром)
// не обоснован: P2b выгоднее по байтам (net −16.1 ГБ на запрос), но только если peer-чтение
// не деградирует сильно относительно копии.
//
// Что измеряется (одна программа, одни условия):
//   local_read      ядро на d читает СВОЮ память                     -> потолок для чтения
//   peer_read       ядро на a читает буфер на b (редукция в СВОЙ буфер) -> предмет замера
//   peer_read_write ядро на a читает b и пишет в свой буфер          -> «копия средствами ядра»
//   peer_write      ядро на a пишет в буфер на b                     -> хенд-овер записью
//   memcpy_peer     cudaMemcpyPeerAsync                              -> контрольная копия
//   peer_read_pair  встречные чтения: a читает b и одновременно b читает a
//   peer_read_vs_memcpy  встречная нагрузка: чтение ядром b->a + копия a->b на той же паре
//   first_ms        время ПЕРВОГО запуска после включения peer access (расход таблиц страниц)
//                   — отдельно от медианы: важно для одноразовых буферов хенд-овера
//
// Методика заимствована у p2p-probe-ext.cu (Этап III), дабы числа были сопоставимы:
// одиночная асинхронная операция событиями не измеряется корректно; используется прогрев,
// затем блоки из N итераций между двумя событиями (N подбирается так, чтобы блок был
// не короче ~64 МиБ данных), медиана по блокам. Отдельно фиксируется холодный запуск.
//
// Корректность: для каждого размера и вида проверяется контрольная сумма прочитанного
// (ожидание считается на хосте по тому же детерминированному генератору) и неизменность
// «отравленной» зоны за границей диапазона. Неправильная контрольная сумма = данные не
// читались с чужой карты, и полоса не имеет смысла.
//
// Зависимости: только CUDA runtime. Сборка: scripts/run-peer-read.sh (приватный nvcc 12.8 +
// приватный g++-14, -arch=sm_70). Ограничение R12: по умолчанию devices 0,1,2 — к GPU3
// внешний контекст при работающем сервисе не подключается.
//
// Использование:
//   peer-read-probe --out DIR [--devices 0,1,2] [--min-size 1M] [--max-size 64M]
//                   [--blocks 5] [--max-buffer-mib N] [--kinds list] [--note "…"]
// Создаёт: peer-read-benchmark.csv, peer-read-correctness.json, peer-read-summary.json
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

static const char* PROG_VERSION = "peer-read-probe 1.1 (2026-10-17, проект Strata+4xV100+NVlink)";

// --------------------------------------------------------------------------- параметры
struct Opts {
    std::string out = ".";
    std::vector<int> devices = {0, 1, 2};
    size_t min_size = 1ull << 20;
    size_t max_size = 64ull << 20;
    size_t max_buffer_mib = 0;
    int blocks = 5;
    std::string kinds = "local_read,peer_read,peer_read_write,peer_write,memcpy_peer,peer_read_pair,peer_read_vs_memcpy";
    std::string note;
};

static size_t parse_size(const char* s) {
    size_t v = strtoull(s, nullptr, 10);
    if (strstr(s, "K") || strstr(s, "k")) v *= 1024ull;
    if (strstr(s, "M") || strstr(s, "m")) v *= 1024ull * 1024ull;
    return v;
}

static bool ck(cudaError_t e, const char* what, FILE* log) {
    if (e == cudaSuccess) return true;
    fprintf(log, "  CUDA error in %s: %s\n", what, cudaGetErrorString(e));
    return false;
}

// ------------------------------------------------------------------ данные и суммы
// Тот же детерминированный генератор, что в p2p-probe-ext.cu: ожидание считается на хосте.
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
#define DATA_SEED   0x51EDDA7A00000000ull
#define POISON_SEED 0xDEADBEEFCAFEBABEull

struct SumOut { unsigned long long a, b; size_t bad; size_t first_bad; };

// чтение + редукция; вывод (out) всегда размещается на карте, где выполняется ядро
// ДешёвыйCombine (только сложение): редукция не должна ограничивать полосу чтения.
// Строгая сверка содержимого выполняется verifyKernel для того же буфера на его же карте.
__global__ void readReduceKernel(const uint4* __restrict__ p, size_t n, SumOut* out) {
    unsigned long long a = 0, b = 0;
    size_t stride = (size_t)gridDim.x * blockDim.x;
    for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += stride) {
        uint4 v = p[i];
        a += (unsigned long long)v.x + v.y + ((unsigned long long)v.z << 32) + v.w;
        b += (unsigned long long)v.z + v.w + ((unsigned long long)v.x << 32) + v.y;
    }
    atomicAdd((unsigned long long*)&out->a, a); atomicAdd((unsigned long long*)&out->b, b);
}
// чтение чужого + запись в свой буфер (копия средствами ядра)
__global__ void readWriteKernel(const uint4* __restrict__ src, uint4* __restrict__ dst, size_t n) {
    size_t stride = (size_t)gridDim.x * blockDim.x;
    for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += stride) dst[i] = src[i];
}
// запись в чужую память / заполнение: p[off + i] = dmake(off + i, seed)
__global__ void fillKernel(uint4* p, size_t off, size_t n, unsigned long long seed) {
    size_t stride = (size_t)gridDim.x * blockDim.x;
    for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += stride) p[i] = dmake(off + i, seed);
}
// проверка: элементы [0,n) равны ожидаемому, элементы [n, n+guard) — «отраве»
__global__ void verifyKernel(const uint4* p, size_t n, size_t guard, unsigned long long seed, SumOut* out) {
    size_t stride = (size_t)gridDim.x * blockDim.x;
    size_t bad = 0;
    for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < n + guard; i += stride) {
        uint4 e = dmake(i, i < n ? seed : POISON_SEED), v = p[i];
        if (v.x != e.x || v.y != e.y || v.z != e.z || v.w != e.w) {
            bad++; atomicExch((unsigned long long*)&out->first_bad, (unsigned long long)i);
        }
    }
    if (bad) atomicAdd((unsigned long long*)&out->bad, (unsigned long long)bad);
}

static void hostExpected(size_t n, unsigned long long seed, unsigned long long* a, unsigned long long* b) {
    unsigned long long sa = 0, sb = 0;
    for (size_t i = 0; i < n; ++i) {
        uint4 v = make_val(i, seed);
        sa += (unsigned long long)v.x + v.y + ((unsigned long long)v.z << 32) + v.w;
        sb += (unsigned long long)v.z + v.w + ((unsigned long long)v.x << 32) + v.y;
    }
    *a = sa; *b = sb;
}

// --------------------------------------------------------------------------- утилиты
static double med(std::vector<double> v) {
    if (v.empty()) return 0; std::sort(v.begin(), v.end());
    size_t k = v.size() / 2;
    return (v.size() % 2) ? v[k] : 0.5 * (v[k - 1] + v[k]);
}
static int pick_blocks(size_t n, int threads) {
    size_t need = (n + threads - 1) / threads;
    int dev = 0, sms = 0; cudaGetDevice(&dev); cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev);
    if (sms <= 0) sms = 80;
    int cap = sms * 8;                       // 8 блоков на SM: более чем достаточно для grid-stride
    return (int)std::min(need, (size_t)cap);
}

struct Row {
    std::string kind; int rd = -1, bd = -1; std::string dir;
    size_t bytes = 0; int blocks = 0, iters = 0;
    double first_ms = 0, med_ms = 0, min_ms = 0, max_ms = 0;
    double med_gbs = 0, min_gbs = 0, max_gbs = 0, med_gibs = 0;
    double spread_pct = 0;
    int ok = 0; std::string note;
};

// Один замер вида «запустить операцию N раз между двумя событиями, повторить B блоков».
// launch() выполняет ровно одну операцию в потоке stream (или в текущем, если stream == 0).
// Возвращает время одной операции: холодный запуск отдельно, далее медиана по блокам.
struct Timing { double first_ms = 0, med_ms = 0, min_ms = 0, max_ms = 0; bool ok = true; };

template <class F>
static Timing timeOp(F launch, int blocks, int iters, cudaStream_t stream, FILE* L, const char* what) {
    Timing t;
    cudaEvent_t e0, e1;
    if (!ck(cudaEventCreate(&e0), "eventCreate", L) || !ck(cudaEventCreate(&e1), "eventCreate", L)) {
        t.ok = false; return t;
    }
    // 1) холодный запуск: одна операция, измеряется одна (первый запуск после выделения буфера)
    cudaEventRecord(e0, stream);
    launch();
    cudaEventRecord(e1, stream);
    if (!ck(cudaGetLastError(), what, L)) { t.ok = false; cudaEventDestroy(e0); cudaEventDestroy(e1); return t; }
    if (!ck(cudaStreamSynchronize(stream), what, L)) { t.ok = false; cudaEventDestroy(e0); cudaEventDestroy(e1); return t; }
    float ms = 0; cudaEventElapsedTime(&ms, e0, e1);
    t.first_ms = ms;
    // 2) прогрев: два блока без измерения
    for (int w = 0; w < 2; ++w) { for (int i = 0; i < iters; ++i) launch(); }
    if (!ck(cudaStreamSynchronize(stream), what, L) || !ck(cudaGetLastError(), what, L)) t.ok = false;
    // 3) измерение
    std::vector<double> per;
    for (int b = 0; b < blocks; ++b) {
        cudaEventRecord(e0, stream);
        for (int i = 0; i < iters; ++i) launch();
        cudaEventRecord(e1, stream);
        if (!ck(cudaStreamSynchronize(stream), what, L) || !ck(cudaGetLastError(), what, L)) { t.ok = false; break; }
        float ms2 = 0; cudaEventElapsedTime(&ms2, e0, e1);
        per.push_back(ms2 / iters);
    }
    t.med_ms = med(per);
    if (!per.empty()) { t.min_ms = *min_element(per.begin(), per.end()); t.max_ms = *max_element(per.begin(), per.end()); }
    cudaEventDestroy(e0); cudaEventDestroy(e1);
    return t;
}

static void fillRow(Row& r, size_t bytes, const Timing& t, int blocks, int iters) {
    r.bytes = bytes; r.blocks = blocks; r.iters = iters;
    r.first_ms = t.first_ms; r.med_ms = t.med_ms; r.min_ms = t.min_ms; r.max_ms = t.max_ms;
    if (t.med_ms > 0) {
        r.med_gbs = bytes / (t.med_ms * 1e-3) / 1e9;
        r.min_gbs = bytes / (t.max_ms * 1e-3) / 1e9;
        r.max_gbs = bytes / (t.min_ms * 1e-3) / 1e9;
        r.med_gibs = bytes / (t.med_ms * 1e-3) / 1073741824.0;
        if (r.med_gbs > 0) r.spread_pct = 100.0 * (r.max_gbs - r.min_gbs) / r.med_gbs;
    }
}

// сколько итераций в одном измерительном блоке (чтобы блок был не короче ~64 МиБ данных)
static int iters_for(size_t bytes) { return (int)std::max<size_t>(3, (64ull << 20) / std::max<size_t>(bytes, 1)); }

// Встречная (полудуплексная) нагрузка на одной паре a/b.
// всегда: ядро на карте a читает буфер src_b, резидентный для b  -> трафик b->a.
// mode 0: одновременно ядро на b читает буфер src_a, резидентный для a -> трафик a->b
//         (встречные чтения ядром).
// mode 1: одновременно cudaMemcpyPeerAsync из src_a (на a) в dst_b (на b) -> копия a->b.
// Обе стороны измеряются собственными событиями на своих потоках; старт синхронизируется
// событием, записанным на потоке a. Контрольные суммы проверяются для обеих сторон.
static void run_contend(int a, int b, size_t bytes, int blocks, int threads, size_t guard,
                        unsigned long long ea, unsigned long long eb,
                        int mode, FILE* L, std::vector<Row>& rows, std::vector<std::string>& checks) {
    const char* kind = mode ? "peer_read_vs_memcpy" : "peer_read_pair";
    size_t n = bytes / 16;
    int iters = iters_for(bytes);
    // буферы: src_b и dst_b — на карте b, src_a — на карте a (встречный поток идёт a->b)
    cudaSetDevice(b);
    uint4 *src_b = nullptr, *dst_b = nullptr;
    if (cudaMalloc(&src_b, (n + guard) * 16) != cudaSuccess) { cudaGetLastError(); return; }
    if (mode && cudaMalloc(&dst_b, bytes) != cudaSuccess) { cudaGetLastError(); cudaFree(src_b); return; }
    fillKernel<<<pick_blocks(n, threads), threads>>>(src_b, 0, n, DATA_SEED);
    cudaDeviceSynchronize();
    cudaSetDevice(a);
    uint4* src_a = nullptr;
    if (cudaMalloc(&src_a, (n + guard) * 16) != cudaSuccess) {
        cudaGetLastError(); cudaFree(src_b); return;
    }
    fillKernel<<<pick_blocks(n, threads), threads>>>(src_a, 0, n, DATA_SEED);
    cudaDeviceSynchronize();
    cudaSetDevice(a);
    SumOut* so_a = nullptr; cudaStream_t sa = nullptr;
    cudaEvent_t e0 = nullptr, e1a = nullptr, eStart = nullptr;
    bool mem = ck(cudaMalloc(&so_a, sizeof(SumOut)), "malloc so_a", L);
    mem = ck(cudaStreamCreate(&sa), "stream a", L) && mem;
    mem = ck(cudaEventCreate(&e0), "event e0", L) && mem;
    mem = ck(cudaEventCreate(&e1a), "event e1a", L) && mem;
    mem = ck(cudaEventCreate(&eStart), "event eStart", L) && mem;
    cudaSetDevice(b);
    SumOut* so_b = nullptr; cudaStream_t sb = nullptr;
    cudaEvent_t e0b = nullptr, e1b = nullptr;
    mem = ck(cudaMalloc(&so_b, sizeof(SumOut)), "malloc so_b", L) && mem;
    mem = ck(cudaStreamCreate(&sb), "stream b", L) && mem;
    mem = ck(cudaEventCreate(&e0b), "event e0b", L) && mem;
    mem = ck(cudaEventCreate(&e1b), "event e1b", L) && mem;
    if (!mem) {
        Row r; r.kind = kind; r.rd = a; r.bd = b; r.bytes = bytes; r.note = "ошибка выделения/потока";
        rows.push_back(r); return;
    }
    int nb = pick_blocks(n, threads);
    std::vector<double> ta, tb;
    double first_a = 0; bool okc = true;
    auto oneIter = [&](void) {
        cudaSetDevice(a); cudaMemsetAsync(so_a, 0, sizeof(SumOut), sa);
        cudaSetDevice(b); cudaMemsetAsync(so_b, 0, sizeof(SumOut), sb);
        cudaSetDevice(a);
        cudaEventRecord(eStart, sa);
        readReduceKernel<<<nb, threads, 0, sa>>>(src_b, n, so_a);        // a читает b: поток b->a
        cudaStreamWaitEvent(sb, eStart, 0);
        cudaSetDevice(b);
        if (mode == 0) readReduceKernel<<<nb, threads, 0, sb>>>(src_a, n, so_b);  // b читает a: a->b
        else cudaMemcpyPeerAsync(dst_b, b, src_a, a, bytes, sb);                  // копия a->b
    };
    // 1) холодный запуск: одна операция, измеряется каждая сторона своими событиями
    cudaSetDevice(a); cudaEventRecord(e0, sa);
    cudaSetDevice(b); cudaEventRecord(e0b, sb);
    oneIter();
    cudaSetDevice(a); cudaEventRecord(e1a, sa);
    cudaSetDevice(b); cudaEventRecord(e1b, sb);
    if (!ck(cudaGetLastError(), "launch contended (cold)", L)) okc = false;
    if (!ck(cudaStreamSynchronize(sa), "sync sa", L) || !ck(cudaStreamSynchronize(sb), "sync sb", L)) okc = false;
    { float m = 0; cudaEventElapsedTime(&m, e0, e1a); first_a = m; }
    // 2) прогрев: два блока без измерения
    for (int w = 0; w < 2; ++w) for (int it = 0; it < iters; ++it) oneIter();
    if (!ck(cudaStreamSynchronize(sa), "sync sa (warm)", L) || !ck(cudaStreamSynchronize(sb), "sync sb (warm)", L)) okc = false;
    // 3) измерение: блоки из iters операций, старт синхронизируется событием на потоке a
    for (int blk = 0; blk < blocks; ++blk) {
        cudaSetDevice(a); cudaEventRecord(e0, sa);
        cudaSetDevice(b); cudaEventRecord(e0b, sb);
        for (int it = 0; it < iters; ++it) oneIter();
        cudaSetDevice(a); cudaEventRecord(e1a, sa);
        cudaSetDevice(b); cudaEventRecord(e1b, sb);
        if (!ck(cudaGetLastError(), "launch contended", L)) { okc = false; break; }
        if (!ck(cudaStreamSynchronize(sa), "sync sa", L) || !ck(cudaStreamSynchronize(sb), "sync sb", L)) { okc = false; break; }
        float ma = 0, mb = 0;
        cudaEventElapsedTime(&ma, e0, e1a);
        cudaEventElapsedTime(&mb, e0b, e1b);
        ta.push_back(ma / iters); tb.push_back(mb / iters);
    }
    SumOut ha, hb;
    cudaSetDevice(a); cudaMemcpy(&ha, so_a, sizeof ha, cudaMemcpyDeviceToHost);
    cudaSetDevice(b); cudaMemcpy(&hb, so_b, sizeof hb, cudaMemcpyDeviceToHost);
    bool oksum = (ha.a == ea && ha.b == eb && hb.a == ea && hb.b == eb);
    bool verok = true;
    if (mode == 1) {   // второй операцией была копия: проверяем содержимое приёмника dst_b
        hb.a = ea; hb.b = eb;
        cudaMemsetAsync(so_b, 0, sizeof(SumOut));
        verifyKernel<<<nb, threads>>>(dst_b, n, 0, DATA_SEED, so_b);
        cudaDeviceSynchronize();
        SumOut hv; cudaMemcpy(&hv, so_b, sizeof hv, cudaMemcpyDeviceToHost);
        verok = (hv.bad == 0);
        oksum = (ha.a == ea && ha.b == eb && verok);
    }
    double ma = med(ta), mb = med(tb);
    Row r; r.kind = kind; r.rd = a; r.bd = b;
    r.dir = mode ? "kernel_read_b_to_a_vs_memcpy_a_to_b" : "kernel_read_both_directions";
    fillRow(r, bytes, Timing{first_a, ma, ta.empty() ? 0 : *min_element(ta.begin(), ta.end()),
                             ta.empty() ? 0 : *max_element(ta.begin(), ta.end()), okc}, blocks, iters);
    double gb_a = r.med_gbs;
    double gb_b = mb > 0 ? bytes / (mb * 1e-3) / 1e9 : 0;
    if (!tb.empty()) {
        r.min_gbs = bytes / (*max_element(tb.begin(), tb.end()) * 1e-3) / 1e9;
        r.max_gbs = bytes / (*min_element(tb.begin(), tb.end()) * 1e-3) / 1e9;
    }
    r.ok = (okc && oksum) ? 1 : 0;
    char note[256];
    snprintf(note, sizeof note, "read %d<-%d %.2f GB/s; %s %d->%d %.2f GB/s; sum %.2f GB/s",
            a, b, gb_a, mode ? "memcpy" : "read", b, a, gb_b, gb_a + gb_b);
    r.note = note;
    rows.push_back(r);
    checks.push_back("{\"kind\":\"" + std::string(kind) + "\",\"reader_a\":" + std::to_string(a) +
                     ",\"reader_b\":" + std::to_string(b) + ",\"bytes\":" + std::to_string(bytes) +
                     ",\"both_checksums_ok\":" + (oksum ? "true" : "false") +
                     ",\"opposite_verify_ok\":" + (verok ? "true" : "false") +
                     ",\"gbps_read\":" + std::to_string(gb_a) + ",\"gbps_opposite\":" + std::to_string(gb_b) +
                     ",\"gbps_sum\":" + std::to_string(gb_a + gb_b) + "}");
    fprintf(L, "  %-21s dev%d<->dev%d  read %.2f GB/s  opposite %.2f GB/s  sum %.2f GB/s  correct=%d\n",
            kind, a, b, gb_a, gb_b, gb_a + gb_b, r.ok);
    cudaSetDevice(a); cudaFree(so_a); cudaFree(src_a); cudaStreamDestroy(sa);
    cudaEventDestroy(e0); cudaEventDestroy(e1a); cudaEventDestroy(eStart);
    cudaSetDevice(b); cudaFree(so_b); cudaFree(src_b); cudaFree(dst_b); cudaStreamDestroy(sb);
    cudaEventDestroy(e0b); cudaEventDestroy(e1b);
}

// --------------------------------------------------------------------------- main
int main(int argc, char** argv) {
    Opts o;
    for (int i = 1; i < argc; ++i) {
        if (!strcmp(argv[i], "--out") && i + 1 < argc) o.out = argv[++i];
        else if (!strcmp(argv[i], "--devices") && i + 1 < argc) {
            o.devices.clear();
            char* buf = strdup(argv[++i]);
            for (char* t = strtok(buf, ","); t; t = strtok(nullptr, ",")) o.devices.push_back(atoi(t));
            free(buf);
        } else if (!strcmp(argv[i], "--min-size") && i + 1 < argc) o.min_size = parse_size(argv[++i]);
        else if (!strcmp(argv[i], "--max-size") && i + 1 < argc) o.max_size = parse_size(argv[++i]);
        else if (!strcmp(argv[i], "--max-buffer-mib") && i + 1 < argc) o.max_buffer_mib = (size_t)atoi(argv[++i]) << 20;
        else if (!strcmp(argv[i], "--blocks") && i + 1 < argc) o.blocks = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--kinds") && i + 1 < argc) o.kinds = argv[++i];
        else if (!strcmp(argv[i], "--note") && i + 1 < argc) o.note = argv[++i];
        else { fprintf(stderr, "неизвестный аргумент: %s\n", argv[i]); return 2; }
    }
    if (o.blocks < 1) o.blocks = 3;
    std::vector<std::string> kinds; {
        char* buf = strdup(o.kinds.c_str());
        for (char* t = strtok(buf, ","); t; t = strtok(nullptr, ",")) kinds.push_back(t);
        free(buf);
    }
    auto want = [&](const char* k) { return std::find(kinds.begin(), kinds.end(), std::string(k)) != kinds.end(); };

    std::string csvp = o.out + "/peer-read-benchmark.csv";
    std::string jcp = o.out + "/peer-read-correctness.json";
    std::string jsp = o.out + "/peer-read-summary.json";
    FILE* C = fopen(csvp.c_str(), "w");
    FILE* L = stdout;
    if (!C) { fprintf(stderr, "не удалось открыть %s\n", csvp.c_str()); return 1; }
    fprintf(C, "kind,reader_dev,buffer_dev,direction,bytes,blocks,iters,first_ms,median_ms,min_ms,max_ms,"
               "median_GBps,min_GBps,max_GBps,median_GiBs,spread_pct,correct,note\n");

    fprintf(L, "%s\n", PROG_VERSION);
    fprintf(L, "devices:"); for (int d : o.devices) fprintf(L, " %d", d);
    fprintf(L, "\nразмеры: %zu … %zu байт, блоков: %d\n", o.min_size, o.max_size, o.blocks);
    if (!o.note.empty()) fprintf(L, "примечание: %s\n", o.note.c_str());

    for (int d : o.devices) {
        cudaSetDevice(d);
        if (!ck(cudaFree(0), "context init", L)) return 1;
        size_t freeb = 0, tot = 0; cudaMemGetInfo(&freeb, &tot);
        fprintf(L, "  dev%d: свободно %zu MiB\n", d, freeb >> 20);
    }
    for (int a : o.devices) for (int b : o.devices) if (a != b) {
        int can = 0; cudaDeviceCanAccessPeer(&can, a, b);
        if (!can) { fprintf(L, "peer access %d->%d: недоступен\n", a, b); continue; }
        cudaSetDevice(a);
        cudaError_t e = cudaDeviceEnablePeerAccess(b, 0);
        if (e != cudaSuccess && e != cudaErrorPeerAccessAlreadyEnabled) ck(e, "enablePeerAccess", L);
        cudaGetLastError();
    }

    std::vector<Row> rows;
    std::vector<std::string> checks;

    std::vector<size_t> sizes;
    for (size_t s = o.min_size; s <= o.max_size; s *= 2) {
        if (o.max_buffer_mib && s > o.max_buffer_mib) break;
        sizes.push_back(s);
    }

    const int THREADS = 256;
    const size_t GUARD = 4096;   // «отравленная» зона за границей диапазона

    for (size_t bytes : sizes) {
        size_t n = bytes / 16;
        unsigned long long ea, eb; hostExpected(n, DATA_SEED, &ea, &eb);
        int iters = iters_for(bytes);
        fprintf(L, "\n=== размер %zu байт (%.1f MiB), элементов %zu, итераций в блоке %d ===\n",
                bytes, bytes / 1048576.0, n, iters);

        // Буферы выделяем на КАЖДУЮ упорядоченную пару: src — на карте-владельце bd,
        // dst — на карте читателя rd. (Прежняя версия держала оба буфера на bd: из-за этого
        // peer_read_write и memcpy_peer мерили локальную копию, а не перенос через NVLink.)
        for (int bd : o.devices) {
            for (int rd : o.devices) {
                bool local = (rd == bd);
                const char* rk = local ? "local_read" : "peer_read";
                cudaSetDevice(bd);
                uint4* src = nullptr; SumOut* so_v = nullptr;
                size_t alloc = (n + GUARD) * 16;
                size_t fb = 0, tot = 0;
                cudaMemGetInfo(&fb, &tot);
                if (fb < alloc + (64ull << 20)) {
                    fprintf(L, "  dev%d: свободно %zu MiB — пропуск размера (нужно %.0f MiB + 64 MiB запаса)\n",
                            bd, fb >> 20, alloc / 1048576.0);
                    Row r; r.kind = "alloc"; r.bd = bd; r.rd = rd; r.bytes = bytes;
                    r.note = "недостаточно свободной VRAM на buffer_dev: " + std::to_string(fb >> 20) + " MiB";
                    rows.push_back(r);
                    continue;
                }
                if (!ck(cudaMalloc(&src, alloc), "malloc src", L)) continue;
                if (!ck(cudaMalloc(&so_v, sizeof(SumOut)), "malloc verify sum", L)) { cudaFree(src); continue; }
                fillKernel<<<pick_blocks(n, THREADS), THREADS>>>(src, 0, n, DATA_SEED);
                fillKernel<<<pick_blocks(GUARD, THREADS), THREADS>>>(src + n, n, GUARD, POISON_SEED);
                ck(cudaDeviceSynchronize(), "fill", L);
                uint4* dst = nullptr; SumOut* out_r = nullptr;
                if (!local) {
                    cudaSetDevice(rd);
                    size_t fr = 0;
                    cudaMemGetInfo(&fr, &tot);
                    if (fr < bytes + (64ull << 20)) {
                        fprintf(L, "  dev%d: свободно %zu MiB — пропуск пары %d<-%d для размера\n", rd, fr >> 20, rd, bd);
                        Row r; r.kind = "alloc"; r.bd = bd; r.rd = rd; r.bytes = bytes;
                        r.note = "недостаточно свободной VRAM на reader_dev: " + std::to_string(fr >> 20) + " MiB";
                        rows.push_back(r);
                        cudaSetDevice(bd); cudaFree(src); cudaFree(so_v);
                        continue;
                    }
                    if (!ck(cudaMalloc(&dst, bytes), "malloc dst", L)) { cudaSetDevice(bd); cudaFree(src); cudaFree(so_v); continue; }
                    if (!ck(cudaMalloc(&out_r, sizeof(SumOut)), "malloc out", L)) {
                        cudaFree(dst); cudaSetDevice(bd); cudaFree(src); cudaFree(so_v); continue;
                    }
                }
                int nb = pick_blocks(n, THREADS);

                // проверка содержимого буфера — средствами его же карты (без удалённых атомиков)
                auto verifySrc = [&]() -> bool {
                    cudaSetDevice(bd);
                    cudaMemsetAsync(so_v, 0, sizeof(SumOut));
                    verifyKernel<<<pick_blocks(n + GUARD, THREADS), THREADS>>>(src, n, GUARD, DATA_SEED, so_v);
                    cudaDeviceSynchronize();
                    SumOut hv; cudaMemcpy(&hv, so_v, sizeof hv, cudaMemcpyDeviceToHost);
                    return hv.bad == 0;
                };
                auto verifyDst = [&]() -> bool {
                    cudaSetDevice(rd);
                    cudaMemsetAsync(out_r, 0, sizeof(SumOut));
                    verifyKernel<<<nb, THREADS>>>(dst, n, 0, DATA_SEED, out_r);
                    cudaDeviceSynchronize();
                    SumOut hv; cudaMemcpy(&hv, out_r, sizeof hv, cudaMemcpyDeviceToHost);
                    return hv.bad == 0;
                };
                auto emit = [&](const char* kind, const char* dir, const Timing& t, bool ok, const char* note,
                               const char* jcheck) {
                    Row r; r.kind = kind; r.rd = rd; r.bd = bd; r.dir = dir;
                    fillRow(r, bytes, t, o.blocks, iters);
                    r.ok = (t.ok && ok) ? 1 : 0;
                    r.note = note;
                    rows.push_back(r);
                    checks.push_back("{\"kind\":\"" + std::string(kind) + "\",\"reader\":" + std::to_string(rd) +
                                     ",\"buffer\":" + std::to_string(bd) + ",\"bytes\":" + std::to_string(bytes) +
                                     ",\"" + std::string(jcheck) + "\":" + (ok ? "true" : "false") + "}");
                    fprintf(L, "  %-15s dev%d<->dev%d  cold %.3f ms  med %.4f ms  %8.2f GB/s  разброс %.1f%%  correct=%d\n",
                            kind, rd, bd, t.first_ms, t.med_ms, r.med_gbs, r.spread_pct, r.ok);
                };

                // ---- local_read / peer_read: чтение + редукция в буфер на карте читателя
                if (want(rk)) {
                    cudaSetDevice(rd);
                    Timing t = timeOp([&]{
                        cudaMemsetAsync(out_r ? out_r : so_v, 0, sizeof(SumOut));
                        readReduceKernel<<<nb, THREADS>>>(src, n, out_r ? out_r : so_v);
                    }, o.blocks, iters, 0, L, rk);
                    SumOut h;
                    cudaMemcpy(&h, out_r ? out_r : so_v, sizeof h, cudaMemcpyDeviceToHost);
                    bool sumok = (h.a == ea && h.b == eb);
                    bool verok = verifySrc();
                    emit(rk, local ? "local" : "read_b_to_a", t, sumok && verok,
                         local ? "ceiling: kernel reads own HBM" : "kernel reads peer-resident buffer",
                         "checksum_ok");
                }
                if (!local) {
                    // ---- peer_read_write: чтение чужого + запись в СВОЙ буфер (копия ядром)
                    if (want("peer_read_write")) {
                        cudaSetDevice(rd);
                        Timing t = timeOp([&]{ readWriteKernel<<<nb, THREADS>>>(src, dst, n); },
                                          o.blocks, iters, 0, L, "peer_read_write");
                        bool verok = verifyDst();
                        emit("peer_read_write", "copy_b_to_a_by_kernel", t, verok,
                             "read peer + write local (kernel copy)", "verify_dst_ok");
                    }
                    // ---- peer_write: запись в чужую память
                    if (want("peer_write")) {
                        cudaSetDevice(rd);
                        Timing t = timeOp([&]{ fillKernel<<<nb, THREADS>>>(src, 0, n, DATA_SEED); },
                                          o.blocks, iters, 0, L, "peer_write");
                        bool verok = verifySrc();
                        emit("peer_write", "write_a_to_b_by_kernel", t, verok,
                             "kernel on reader_dev writes buffer on buffer_dev", "verify_dst_ok");
                    }
                    // ---- memcpy_peer: контрольная копия через copy engine
                    if (want("memcpy_peer")) {
                        cudaSetDevice(rd);
                        Timing t = timeOp([&]{ cudaMemcpyPeerAsync(dst, rd, src, bd, bytes); },
                                          o.blocks, iters, 0, L, "memcpy_peer");
                        bool verok = verifyDst();
                        emit("memcpy_peer", "copy_b_to_a_by_memcpy", t, verok,
                             "reference copy (compare with p2p-results.md 38.77 GB/s)", "verify_dst_ok");
                    }
                }
                if (dst) { cudaSetDevice(rd); cudaFree(dst); cudaFree(out_r); }
                cudaSetDevice(bd); cudaFree(src); cudaFree(so_v);
            }
        }

                // ---- встречная (полудуплексная) нагрузка: каждая неупорядоченная пара один раз
        if (want("peer_read_pair") || want("peer_read_vs_memcpy")) {
            for (size_t i = 0; i < o.devices.size(); ++i) {
                for (size_t j = 0; j < o.devices.size(); ++j) {
                    int a = o.devices[i], b = o.devices[j];
                    if (a <= b) continue;
                    if (want("peer_read_pair"))
                        run_contend(a, b, bytes, o.blocks, THREADS, GUARD, ea, eb, 0, L, rows, checks);
                    if (want("peer_read_vs_memcpy"))
                        run_contend(a, b, bytes, o.blocks, THREADS, GUARD, ea, eb, 1, L, rows, checks);
                }
            }
        }
    }

    for (const Row& r : rows) {
        fprintf(C, "%s,%d,%d,%s,%zu,%d,%d,%.4f,%.4f,%.4f,%.4f,%.2f,%.2f,%.2f,%.2f,%.2f,%d,\"%s\"\n",
                r.kind.c_str(), r.rd, r.bd, r.dir.c_str(), r.bytes, r.blocks, r.iters,
                r.first_ms, r.med_ms, r.min_ms, r.max_ms,
                r.med_gbs, r.min_gbs, r.max_gbs, r.med_gibs, r.spread_pct, r.ok, r.note.c_str());
    }
    fclose(C);

    // сводка: отношения peer_read / memcpy и peer_read / local_read по размерам (пара 0/1 как пример)
    FILE* J = fopen(jsp.c_str(), "w");
    fprintf(J, "{\n\"tool\": \"%s\",\n\"devices\": [", PROG_VERSION);
    for (size_t i = 0; i < o.devices.size(); ++i) fprintf(J, "%s%d", i ? "," : "", o.devices[i]);
    fprintf(J, "],\n\"blocks\": %d,\n\"note\": \"%s\",\n\"by_size\": [\n", o.blocks, o.note.c_str());
    bool firstj = true;
    for (size_t bytes : sizes) {
        auto get = [&](const char* k, int rd, int bd) -> const Row* {
            for (const Row& r : rows)
                if (r.kind == k && r.rd == rd && r.bd == bd && r.bytes == bytes && r.med_gbs > 0) return &r;
            return nullptr;
        };
        const Row* pr = get("peer_read", 1, 0);
        const Row* mc = get("memcpy_peer", 1, 0);
        const Row* lr = get("local_read", 0, 0);
        const Row* pw = get("peer_write", 1, 0);
        const Row* rw = get("peer_read_write", 1, 0);
        const Row* cp = get("peer_read_pair", 1, 0);
        const Row* vs = get("peer_read_vs_memcpy", 1, 0);
        char buf[64];
        auto ratio = [&](const Row* x, const Row* y) -> const char* {
            if (!x || !y || y->med_gbs <= 0) return "null";
            snprintf(buf, sizeof buf, "%.3f", x->med_gbs / y->med_gbs);
            return buf;
        };
        if (!firstj) fprintf(J, ",\n");
        firstj = false;
        fprintf(J, "{\"bytes\": %zu, \"peer_read\": %.2f, \"memcpy_peer\": %.2f, \"local_read\": %.2f, "
                   "\"peer_write\": %.2f, \"peer_read_write\": %.2f, \"peer_read_pair\": %.2f, "
                   "\"peer_read_vs_memcpy\": %.2f, \"ratio_read_to_memcpy\": %s, \"ratio_read_to_local\": %s}",
                bytes, pr ? pr->med_gbs : -1.0, mc ? mc->med_gbs : -1.0, lr ? lr->med_gbs : -1.0,
                pw ? pw->med_gbs : -1.0, rw ? rw->med_gbs : -1.0, cp ? cp->med_gbs : -1.0,
                vs ? vs->med_gbs : -1.0, ratio(pr, mc), ratio(pr, lr));
    }
    fprintf(J, "\n],\n\"rows\": %zu}\n", rows.size());
    fclose(J);

    FILE* K = fopen(jcp.c_str(), "w");
    fprintf(K, "{\"tool\": \"%s\", \"checks\": [\n", PROG_VERSION);
    for (size_t i = 0; i < checks.size(); ++i) fprintf(K, "%s%s\n", i ? "," : "", checks[i].c_str());
    fprintf(K, "]}\n");
    fclose(K);

    int bad = 0; for (const Row& r : rows) if (r.ok == 0 && r.kind != "alloc" && r.kind != "contend") bad++;
    fprintf(L, "\nстрок: %zu, из них с ошибкой корректности: %d\n", rows.size(), bad);
    fprintf(L, "файлы: %s, %s, %s\n", csvp.c_str(), jcp.c_str(), jsp.c_str());
    return bad ? 3 : 0;
}
