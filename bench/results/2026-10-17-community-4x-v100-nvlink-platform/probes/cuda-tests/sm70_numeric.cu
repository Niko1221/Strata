/* Этап VI (п. 9.1): численные и ресурсные CUDA-тесты на Tesla V100 (sm_70).
 *
 * Покрывает: запуск специализированных ядер (тензорные ядра HMMA через wmma), численную точность
 * против эталона на CPU (double), допуски fp16/fp32, использование shared memory на уровне максимума
 * движка (41104 Б), ядро с высоким числом регистров, local memory, выделение/освобождение памяти и
 * обработку отказа выделения, обработку восстанавливаемых ошибок запуска, отсутствие зависаний при
 * синхронизации (потоки + события + grid-stride), межкарточное исполнение ядер (peer access).
 *
 * ВАЖНО: cudaDeviceReset() не вызывается — primary context устройства 0 разделяется с процессом
 * движка Strata, и сброс контекста высвободил бы его память. Поэтому тесты фатальных ошибок
 * (__trap, illegal memory access) здесь сознательно отсутствуют (пробел, требует окна остановки).
 *
 * Сборка:
 *   ~/strata185/cuda-12.8/bin/nvcc -ccbin ~/strata185/gcc185/usr/bin/g++-14 -arch=sm_70 -O2 \
 *       -Wno-deprecated-gpu-targets -o build/sm70-tests tests/cuda/sm70_numeric.cu
 */
#include <cuda_runtime.h>
#include <mma.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <string>
#include <algorithm>
using namespace nvcuda;

static int g_pass = 0, g_fail = 0;
static void row(const char *test, bool ok, const char *detail) {
    g_pass += ok; g_fail += !ok;
    printf("%s %-28s %s\n", ok ? "PASS" : "FAIL", test, detail);
}
#define CK(x) do { cudaError_t _e = (x); if (_e != cudaSuccess) { \
    printf("  cuda error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(_e)); return false; } } while (0)

/* ---------- эталон на CPU ---------- */
static void cpu_gemm(const std::vector<float> &A, const std::vector<float> &B, std::vector<double> &C, int M, int N, int K) {
    for (int i = 0; i < M; i++) for (int j = 0; j < N; j++) {
        double s = 0.0;
        for (int k = 0; k < K; k++) s += (double)A[i * K + k] * (double)B[k * N + j];
        C[i * N + j] = s;
    }
}
/* эталон для fp16-входов: те же A и B, но предварительно округлённые до half (честное сравнение) */
static void cpu_gemm_half(const std::vector<half> &A, const std::vector<half> &B, std::vector<double> &C, int M, int N, int K) {
    for (int i = 0; i < M; i++) for (int j = 0; j < N; j++) {
        double s = 0.0;
        for (int k = 0; k < K; k++) s += (double)__half2float(A[i * K + k]) * (double)__half2float(B[k * N + j]);
        C[i * N + j] = s;
    }
}

/* ---------- N1: GEMM через wmma fp16 (на sm_70 компилируется в HMMA) ---------- */
__global__ void gemm_wmma(const half *A, const half *B, float *C, int M, int N, int K) {
    int bx = blockIdx.x * 16, by = blockIdx.y * 16;
    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> b;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> c;
    wmma::fill_fragment(c, 0.0f);
    for (int k = 0; k < K; k += 16) {
        wmma::load_matrix_sync(a, A + bx * K + k, K);
        wmma::load_matrix_sync(b, B + k * N + by, N);
        wmma::mma_sync(c, a, b, c);
    }
    wmma::store_matrix_sync(C + bx * N + by, c, N, wmma::mem_row_major);
}

/* ---------- N2/N3: SIMT GEMM в fp32 и в fp16-аккумуляции ---------- */
__global__ void gemm_f32(const float *A, const float *B, float *C, int M, int N, int K) {
    int i = blockIdx.x * 32 + threadIdx.x, j = blockIdx.y * 32 + threadIdx.y;
    if (i >= M || j >= N) return;
    float s = 0.0f;
    for (int k = 0; k < K; k++) s += A[i * K + k] * B[k * N + j];
    C[i * N + j] = s;
}
__global__ void gemm_f16acc(const half *A, const half *B, half *C, int M, int N, int K) {
    int i = blockIdx.x * 32 + threadIdx.x, j = blockIdx.y * 32 + threadIdx.y;
    if (i >= M || j >= N) return;
    half s = __float2half(0.0f);
    for (int k = 0; k < K; k++) s = __hadd(s, __hmul(A[i * K + k], B[k * N + j]));
    C[i * N + j] = s;
}

static bool test_gemm(int M, int N, int K) {
    std::vector<float> hA(M * K), hB(K * N);
    std::vector<double> ref(M * N);
    srand(20261017);
    for (auto &v : hA) v = (rand() / (float)RAND_MAX) * 2.0f - 1.0f;
    for (auto &v : hB) v = (rand() / (float)RAND_MAX) * 2.0f - 1.0f;
    cpu_gemm(hA, hB, ref, M, N, K);

    float *dA32, *dB32, *dC32; half *dA16, *dB16, *dC16;
    CK(cudaMalloc(&dA32, sizeof(float) * M * K)); CK(cudaMalloc(&dB32, sizeof(float) * K * N));
    CK(cudaMalloc(&dC32, sizeof(float) * M * N));
    CK(cudaMalloc(&dA16, sizeof(half) * M * K)); CK(cudaMalloc(&dB16, sizeof(half) * K * N));
    CK(cudaMalloc(&dC16, sizeof(half) * M * N));
    std::vector<half> hA16(hA.size()), hB16(hB.size());
    for (size_t i = 0; i < hA.size(); i++) hA16[i] = __float2half(hA[i]);
    for (size_t i = 0; i < hB.size(); i++) hB16[i] = __float2half(hB[i]);
    std::vector<double> ref16(M * N);
    cpu_gemm_half(hA16, hB16, ref16, M, N, K);
    CK(cudaMemcpy(dA32, hA.data(), sizeof(float) * M * K, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dB32, hB.data(), sizeof(float) * K * N, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dA16, hA16.data(), sizeof(half) * M * K, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dB16, hB16.data(), sizeof(half) * K * N, cudaMemcpyHostToDevice));

    dim3 blk(32, 32), grid((M + 31) / 32, (N + 31) / 32);
    dim3 wblk(32), wgrid(M / 16, N / 16);
    bool ok = true;

    /* fp32 SIMT */
    CK(cudaMemset(dC32, 0, sizeof(float) * M * N));
    gemm_f32<<<grid, blk>>>(dA32, dB32, dC32, M, N, K);
    CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
    std::vector<float> got(M * N);
    CK(cudaMemcpy(got.data(), dC32, sizeof(float) * M * N, cudaMemcpyDeviceToHost));
    double max_abs32 = 0, max_rel32 = 0;
    for (int i = 0; i < M * N; i++) {
        double e = fabs((double)got[i] - ref[i]);
        max_abs32 = std::max(max_abs32, e);
        if (fabs(ref[i]) > 1e-6) max_rel32 = std::max(max_rel32, e / fabs(ref[i]));
    }
    char buf[400];
    snprintf(buf, sizeof buf, "max_abs=%.3e max_rel=%.3e (K=%d, fp32 SIMT)", max_abs32, max_rel32, K);
    bool ok32 = max_abs32 < 1e-3;
    row("N2_gemm_fp32_vs_cpu", ok32, buf); ok = ok && ok32;

    /* fp16 wmma (HMMA): эталон — то же произведение, но посчитанное на CPU по тем же half-входам */
    CK(cudaMemset(dC32, 0, sizeof(float) * M * N));
    gemm_wmma<<<wgrid, wblk>>>((const half*)dA16, (const half*)dB16, dC32, M, N, K);
    CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
    CK(cudaMemcpy(got.data(), dC32, sizeof(float) * M * N, cudaMemcpyDeviceToHost));
    double max_absw = 0, max_relw = 0;
    for (int i = 0; i < M * N; i++) {
        double e = fabs((double)got[i] - ref16[i]);
        max_absw = std::max(max_absw, e);
        if (fabs(ref16[i]) > 0.1) max_relw = std::max(max_relw, e / fabs(ref16[i]));
    }
    double tol_w = 1e-4 * K;   /* допуск: fp32-аккумуляция HMMA, ошибка растёт линейно с K */
    snprintf(buf, sizeof buf, "max_abs=%.3e max_rel=%.3e допуск=%.1e (wmma 16x16x16 fp16->fp32, эталон = CPU по half-входам)", max_absw, max_relw, tol_w);
    bool okw = max_absw < tol_w;
    row("N1_gemm_wmma_fp16", okw, buf); ok = ok && okw;

    /* fp16 с fp16-аккумуляцией: демонстрация роста ошибки (почему нужны допуски) */
    CK(cudaMemset(dC16, 0, sizeof(half) * M * N));
    gemm_f16acc<<<grid, blk>>>((const half*)dA16, (const half*)dB16, dC16, M, N, K);
    CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
    std::vector<half> got16(M * N);
    CK(cudaMemcpy(got16.data(), dC16, sizeof(half) * M * N, cudaMemcpyDeviceToHost));
    double max_abs16 = 0;
    for (int i = 0; i < M * N; i++) max_abs16 = std::max(max_abs16, fabs(__half2float(got16[i]) - ref16[i]));
    snprintf(buf, sizeof buf, "max_abs=%.3e (fp16 accumulate, K=%d) — сравнение с fp32-аккумуляцией %.3e", max_abs16, K, max_absw);
    row("N3_fp16_accum_error", max_abs16 >= max_absw * 0.5, buf);

    cudaFree(dA32); cudaFree(dB32); cudaFree(dC32); cudaFree(dA16); cudaFree(dB16); cudaFree(dC16);
    return ok;
}

/* ---------- N4: shared memory на уровне максимума движка (41104 Б) ---------- */
__global__ void shared_kernel(float *out, int n) {
    extern __shared__ float sm[];
    for (int i = threadIdx.x; i < n; i += blockDim.x) sm[i] = (float)i * 0.5f;
    __syncthreads();
    float acc = 0.0f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) acc += sm[i];
    __syncthreads();
    sm[threadIdx.x] = acc;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) sm[threadIdx.x] += sm[threadIdx.x + s];
        __syncthreads();
    }
    if (threadIdx.x == 0) out[blockIdx.x] = sm[0];
}
static bool test_shared() {
    const int BYTES = 41104;                 /* максимум статического shared в бинарнике движка */
    const int N = BYTES / sizeof(float);
    CK(cudaFuncSetAttribute(shared_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, BYTES));
    float *dOut; CK(cudaMalloc(&dOut, sizeof(float) * 4));
    shared_kernel<<<4, 1024, BYTES>>>(dOut, N);
    cudaError_t le = cudaGetLastError();
    cudaError_t se = cudaDeviceSynchronize();
    float host[4] = {-1, -1, -1, -1};
    if (le == cudaSuccess && se == cudaSuccess) CK(cudaMemcpy(host, dOut, sizeof host, cudaMemcpyDeviceToHost));
    double expect = 0.0; for (int i = 0; i < N; i++) expect += (double)i * 0.5;
    bool ok = (le == cudaSuccess && se == cudaSuccess);
    double max_rel = 0;
    for (int b = 0; b < 4; b++) {
        double rel = fabs((double)host[b] - expect) / expect;
        max_rel = std::max(max_rel, rel);
        if (rel > 1e-5) ok = false;
    }
    char buf[300];
    snprintf(buf, sizeof buf, "dynamic_shared=%d Б (opt-in лимит 98304) launch=%s sum_rel_err=%.2e (expect=%.0f got=%.0f)",
             BYTES, cudaGetErrorString(le), max_rel, expect, (double)host[0]);
    row("N4_shared_41104B", ok, buf);
    cudaFree(dOut);
    return ok;
}

/* ---------- N5: ядро с высоким числом регистров ---------- */
__global__ void highreg_kernel(float *out, const float *in) {
    float r[200];
    #pragma unroll
    for (int i = 0; i < 200; i++) r[i] = in[threadIdx.x * 200 + i];
    #pragma unroll
    for (int rep = 0; rep < 3; rep++)
        #pragma unroll
        for (int i = 0; i < 200; i++) r[i] = fmaf(r[i], 1.0f + 1e-3f * i, r[(i + 1) % 200]);
    float s = 0.0f;
    #pragma unroll
    for (int i = 0; i < 200; i++) s += r[i];
    if (threadIdx.x == 0) out[blockIdx.x] = s;
}
static bool test_highreg() {
    cudaFuncAttributes at{}; CK(cudaFuncGetAttributes(&at, highreg_kernel));
    const int NT = 256;
    std::vector<float> hin(NT * 200, 1.0f);
    float *dIn, *dOut;
    CK(cudaMalloc(&dIn, sizeof(float) * NT * 200)); CK(cudaMalloc(&dOut, sizeof(float) * 2));
    CK(cudaMemcpy(dIn, hin.data(), sizeof(float) * NT * 200, cudaMemcpyHostToDevice));
    highreg_kernel<<<2, NT>>>(dOut, dIn);
    cudaError_t le = cudaGetLastError(), se = cudaDeviceSynchronize();
    float host[2]; CK(cudaMemcpy(host, dOut, sizeof host, cudaMemcpyDeviceToHost));
    bool ok = (le == cudaSuccess && se == cudaSuccess && host[0] == host[1]);
    char buf[300];
    snprintf(buf, sizeof buf, "regs=%d max_threads_per_block=%d local_bytes=%d launch=%s identical_blocks=%s",
             at.numRegs, at.maxThreadsPerBlock, at.localSizeBytes, cudaGetErrorString(le), host[0] == host[1] ? "yes" : "no");
    row("N5_high_register_kernel", ok, buf);
    cudaFree(dIn); cudaFree(dOut);
    return ok;
}

/* ---------- N6: local memory / стек ---------- */
__global__ void localmem_kernel(float *out) {
    float big[256];
    for (int i = 0; i < 256; i++) big[i] = i * 0.25f;
    __syncthreads();
    float s = 0; for (int i = 0; i < 256; i++) s += big[i];
    if (threadIdx.x == 0) out[0] = s;
}
static bool test_localmem() {
    cudaFuncAttributes at{}; CK(cudaFuncGetAttributes(&at, localmem_kernel));
    float *dOut; CK(cudaMalloc(&dOut, sizeof(float)));
    localmem_kernel<<<1, 64>>>(dOut);
    cudaError_t le = cudaGetLastError(), se = cudaDeviceSynchronize();
    float h = -1; CK(cudaMemcpy(&h, dOut, sizeof h, cudaMemcpyDeviceToHost));
    bool ok = (le == cudaSuccess && se == cudaSuccess && fabs(h - 8160.0) < 1.0);
    char buf[300]; snprintf(buf, sizeof buf, "local_bytes=%d regs=%d result=%.1f expect=8160.0", at.localSizeBytes, at.numRegs, h);
    row("N6_local_memory", ok, buf);
    cudaFree(dOut);
    return ok;
}

/* ---------- N7: выделение/освобождение памяти и отказ выделения ---------- */
static bool test_memory() {
    size_t free0 = 0, total = 0;
    CK(cudaMemGetInfo(&free0, &total));
    bool leak_ok = true;
    for (int i = 0; i < 200; i++) {
        void *p = nullptr;
        cudaError_t e = cudaMalloc(&p, 16u << 20);
        if (e != cudaSuccess) { leak_ok = false; printf("  alloc %d: %s\n", i, cudaGetErrorString(e)); break; }
        e = cudaFree(p);
        if (e != cudaSuccess) { leak_ok = false; break; }
    }
    size_t free1 = 0; cudaMemGetInfo(&free1, &total);
    long long delta = (long long)free0 - (long long)free1;

    /* отказ выделения: запрашиваем больше, чем есть на карте */
    void *big = nullptr;
    cudaError_t e = cudaMalloc(&big, (size_t)60ull << 30);
    bool oom_handled = (e == cudaErrorMemoryAllocation);
    const char *oom_msg = cudaGetErrorString(e);
    if (big) { cudaFree(big); oom_handled = false; }
    cudaGetLastError();  /* сбрасываем состояние ошибки, контекст НЕ сбрасываем */

    /* работоспособность после отказа */
    float *d = nullptr;
    bool alive = (cudaMalloc(&d, 1u << 20) == cudaSuccess);
    if (alive) {
        localmem_kernel<<<1, 64>>>(d);
        alive = (cudaGetLastError() == cudaSuccess && cudaDeviceSynchronize() == cudaSuccess);
        cudaFree(d);
    }
    size_t free2 = 0; cudaMemGetInfo(&free2, &total);
    char buf[400];
    snprintf(buf, sizeof buf, "200x16MB alloc/free leak_delta=%lld Б (free %zu -> %zu -> %zu), 60GB alloc -> %s, usable_after=%s",
             delta, free0, free1, free2, oom_msg, alive ? "yes" : "no");
    bool ok = leak_ok && oom_handled && alive && llabs(delta) < (1 << 20);
    row("N7_alloc_free_oom", ok, buf);
    return ok;
}

/* ---------- N8: обработка восстанавливаемых ошибок запуска ---------- */
static bool test_launch_errors() {
    float *d = nullptr; CK(cudaMalloc(&d, sizeof(float) * 8));
    localmem_kernel<<<1, 2048>>>(d);                    /* > 1024 потоков в блоке */
    cudaError_t e1 = cudaGetLastError(); cudaDeviceSynchronize();
    localmem_kernel<<<0, 32>>>(d);                      /* нулевой grid */
    cudaError_t e2 = cudaGetLastError(); cudaDeviceSynchronize();
    cudaError_t e3 = cudaMemsetAsync(nullptr, 0, 4);   /* некорректный аргумент */
    cudaDeviceSynchronize();
    cudaGetLastError();   /* сбрасываем состояние: cudaErrorInvalidValue не «липкая», но остаётся в last-error */
    /* после восстанавливаемых ошибок обычный запуск обязан работать */
    localmem_kernel<<<1, 64>>>(d);
    cudaError_t e4 = cudaGetLastError();
    cudaError_t e5 = cudaDeviceSynchronize();
    bool recovered = (e4 == cudaSuccess && e5 == cudaSuccess);
    float h = -1; cudaMemcpy(&h, d, sizeof h, cudaMemcpyDeviceToHost);
    recovered = recovered && fabs(h - 8160.0) < 1.0;
    char buf[400];
    snprintf(buf, sizeof buf, "threads=2048 -> %s; grid=0 -> %s; memset(null) -> %s; after: launch=%s sync=%s value=%.1f",
             cudaGetErrorString(e1), cudaGetErrorString(e2), cudaGetErrorString(e3),
             cudaGetErrorString(e4), cudaGetErrorString(e5), h);
    bool ok = (e1 == cudaErrorInvalidValue || e1 == cudaErrorInvalidConfiguration)
           && (e2 == cudaErrorInvalidValue || e2 == cudaErrorInvalidConfiguration)
           && (e3 != cudaSuccess) && recovered;
    row("N8_recoverable_errors", ok, buf);
    cudaFree(d);
    return ok;
}

/* ---------- N9: отсутствие зависаний при синхронизации (потоки, события, grid-stride) ---------- */
__global__ void sync_heavy(float *buf, int n, int iters) {
    for (int it = 0; it < iters; it++) {
        for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x)
            buf[i] = buf[i] * 1.0001f + 1e-6f;
        __syncthreads();
    }
}
__global__ void producer(float *buf, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) buf[i] = (float)i;
}
__global__ void consumer(const float *buf, float *out, int n) {
    float s = 0.0f;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) s += buf[i];
    atomicAdd(out, s);
}
static bool test_sync() {
    const int N = 1 << 20, ITERS = 50;
    float *dBuf, *dOut;
    CK(cudaMalloc(&dBuf, sizeof(float) * N)); CK(cudaMalloc(&dOut, sizeof(float)));
    CK(cudaMemset(dBuf, 0, sizeof(float) * N)); CK(cudaMemset(dOut, 0, sizeof(float)));
    cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
    cudaStream_t s1, s2; cudaStreamCreate(&s1); cudaStreamCreate(&s2);
    cudaEventRecord(e0, 0);
    sync_heavy<<<80, 512, 0, s1>>>(dBuf, N, ITERS);
    cudaEventRecord(e1, s1);
    cudaStreamSynchronize(s1);
    float ms = 0; cudaEventElapsedTime(&ms, e0, e1);
    bool no_hang = (cudaGetLastError() == cudaSuccess && cudaDeviceSynchronize() == cudaSuccess);

    /* межпотоковая зависимость через событие: consumer обязан увидеть данные producer */
    producer<<<(N + 255) / 256, 256, 0, s1>>>(dBuf, N);
    cudaEvent_t ev; cudaEventCreate(&ev);
    cudaEventRecord(ev, s1);
    cudaStreamWaitEvent(s2, ev, 0);
    consumer<<<64, 1024, 0, s2>>>(dBuf, dOut, N);
    cudaStreamSynchronize(s2);
    float h = 0; CK(cudaMemcpy(&h, dOut, sizeof h, cudaMemcpyDeviceToHost));
    double expect = 0.0; for (int i = 0; i < N; i++) expect += i;
    bool order_ok = fabs((double)h - expect) / expect < 1e-3 && cudaGetLastError() == cudaSuccess;

    char buf[300];
    snprintf(buf, sizeof buf, "grid-stride 80x512 x %d итераций: %.1f мс, hang=%s; cross-stream order sum=%.0f expect=%.0f rel_err=%.1e (float32 атомарная аккумуляция) -> %s",
             ITERS, ms, no_hang ? "no" : "yes", h, expect, fabs((double)h - expect) / expect, order_ok ? "ok" : "fail");
    bool ok = no_hang && order_ok;
    row("N9_sync_no_hang", ok, buf);
    cudaEventDestroy(e0); cudaEventDestroy(e1); cudaEventDestroy(ev);
    cudaStreamDestroy(s1); cudaStreamDestroy(s2);
    cudaFree(dBuf); cudaFree(dOut);
    return ok;
}

/* ---------- N10: межкарточное исполнение ядер (peer access) ---------- */
__global__ void peer_read(const float *remote, float *out, int n) {
    float s = 0;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) s += remote[i];
    atomicAdd(out, s);
}
static bool test_peer(int ndev) {
    if (ndev < 2) { row("N10_peer_access_kernels", true, "пропущено: < 2 устройств"); return true; }
    const int N = 4096;
    std::vector<float> host(N);
    double expect = 0.0;
    for (int i = 0; i < N; i++) { host[i] = i * 0.5f; expect += i * 0.5; }

    /* Сначала пробуем подключить контекст процесса к каждой карте (это само по себе требует памяти). */
    std::vector<int> usable; std::vector<std::string> skip;
    for (int d = 0; d < ndev; d++) {
        cudaError_t e = cudaSetDevice(d);
        size_t free_b = 0, total = 0;
        if (e == cudaSuccess) e = cudaDeviceSynchronize();
        if (e == cudaSuccess) cudaMemGetInfo(&free_b, &total);
        char s[160];
        snprintf(s, sizeof s, "dev%d: attach=%s free=%zu MiB", d, cudaGetErrorString(e), free_b >> 20);
        if (e == cudaSuccess) usable.push_back(d); else skip.push_back(s);
    }

    int pairs_ok = 0, pairs_total = 0;
    for (int ai = 0; ai < (int)usable.size(); ai++) {
        int a = usable[ai];
        for (int bi = 0; bi < (int)usable.size(); bi++) {
            int b = usable[bi];
            if (a == b) continue;
            int can = 0;
            if (cudaDeviceCanAccessPeer(&can, a, b) != cudaSuccess || !can) continue;
            pairs_total++;
            cudaSetDevice(b);
            float *remote = nullptr, *out = nullptr;
            if (cudaMalloc(&remote, sizeof(float) * N) != cudaSuccess) continue;
            cudaMemcpy(remote, host.data(), sizeof(float) * N, cudaMemcpyHostToDevice);
            cudaDeviceEnablePeerAccess(a, 0);
            cudaGetLastError();   /* AlreadyEnabled — штатный ответ, очищаем last-error */
            cudaSetDevice(a);
            cudaDeviceEnablePeerAccess(b, 0);
            cudaGetLastError();
            if (cudaMalloc(&out, sizeof(float)) != cudaSuccess) { cudaFree(remote); continue; }
            cudaMemset(out, 0, sizeof(float));
            peer_read<<<8, 512>>>(remote, out, N);          /* ядро на карте a читает память карты b */
            cudaError_t le = cudaGetLastError(), se = cudaDeviceSynchronize();
            float h = 0; cudaMemcpy(&h, out, sizeof h, cudaMemcpyDeviceToHost);
            bool ok = (le == cudaSuccess && se == cudaSuccess && fabs((double)h - expect) < 1.0);
            if (ok) pairs_ok++;
            else printf("  pair %d->%d launch=%s sync=%s got=%.1f expect=%.1f\n", a, b,
                        cudaGetErrorString(le), cudaGetErrorString(se), h, expect);
            cudaFree(remote); cudaFree(out);
        }
    }
    char buf[600];
    std::string sk;
    for (auto &s : skip) { sk += s; sk += "; "; }
    snprintf(buf, sizeof buf, "карт с подключённым контекстом: %zu из %d (%s); упорядоченных пар %d/%d прочитали чужую память верно; пропуск по памяти: %s",
             usable.size(), ndev, usable.size() == (size_t)ndev ? "все" : "не все", pairs_ok, pairs_total, sk.empty() ? "нет" : sk.c_str());
    bool ok = pairs_ok == pairs_total && pairs_total > 0;
    row("N10_peer_access_kernels", ok, buf);
    cudaSetDevice(0);
    return ok;
}

int main(int argc, char **argv) {
    int dev = argc > 1 ? atoi(argv[1]) : 0;
    CK(cudaSetDevice(dev));
    cudaDeviceProp p{}; CK(cudaGetDeviceProperties(&p, dev));
    printf("device %d: %s cc=%d.%d SMs=%d shared/block=%zu opt-in=%d regs/block=%d\n",
           dev, p.name, p.major, p.minor, p.multiProcessorCount, p.sharedMemPerBlock,
           p.sharedMemPerBlockOptin, p.regsPerBlock);
    bool all = true;
    all &= test_gemm(128, 128, 128);
    all &= test_gemm(256, 256, 256);
    all &= test_shared();
    all &= test_highreg();
    all &= test_localmem();
    all &= test_memory();
    all &= test_launch_errors();
    all &= test_sync();
    int ndev = 0; cudaGetDeviceCount(&ndev);
    all &= test_peer(ndev);
    printf("SUMMARY pass=%d fail=%d RESULT %s\n", g_pass, g_fail, (g_fail == 0) ? "PASS" : "FAIL");
    return g_fail == 0 ? 0 : 2;
}
