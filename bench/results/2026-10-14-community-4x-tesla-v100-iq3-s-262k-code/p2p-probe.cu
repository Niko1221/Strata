// Достижимая пропускная способность cudaMemcpyPeer между парами карт на этой плате.
// Отвечает на вопрос "что даст движку P2P-маршрут вместо хост-стажинга": если копирование
// через NVLink даёт ~100 ГБ/с, а через хост - ~7 ГБ/с, разница и есть цена текущего выбора.
// Пары с недостатком свободной VRAM пропускаются и помечаются - сервер модели занят.
#include <cuda_runtime.h>
#include <cstdio>
#include <cstring>

static const size_t BYTES = 32ull << 20;   // 32 MiB - мало, чтобы влезть в остаток VRAM
static const int REPS = 20;

#define CK(e) do { cudaError_t _e = (e); if (_e != cudaSuccess) { \
    printf("    CUDA error %s at %s:%d\n", cudaGetErrorString(_e), __FILE__, __LINE__); return false; } } while (0)

bool pair(int a, int b) {
    cudaDeviceProp pa, pb;
    CK(cudaGetDeviceProperties(&pa, a)); CK(cudaGetDeviceProperties(&pb, b));
    int can_ab = 0, can_ba = 0;
    cudaDeviceCanAccessPeer(&can_ab, a, b);
    cudaDeviceCanAccessPeer(&can_ba, b, a);
    int access = 0, atomics = 0, rank = -1;
    cudaDeviceGetP2PAttribute(&access, cudaDevP2PAttrAccessSupported, a, b);
    cudaDeviceGetP2PAttribute(&atomics, cudaDevP2PAttrNativeAtomicSupported, a, b);
    cudaDeviceGetP2PAttribute(&rank, cudaDevP2PAttrPerformanceRank, a, b);
    printf("GPU%d <-> GPU%d: canAccessPeer %d/%d, accessSupported=%d, nativeAtomics=%d, perfRank=%d\n",
           a, b, can_ab, can_ba, access, atomics, rank);
    if (!can_ab || !can_ba) { printf("  -> P2P нет, пропуск\n"); return true; }

    size_t free_a = 0, free_b = 0, tot = 0;
    CK(cudaSetDevice(a)); CK(cudaMemGetInfo(&free_a, &tot));
    CK(cudaSetDevice(b)); CK(cudaMemGetInfo(&free_b, &tot));
    if (free_a < 400ull * 1024 * 1024 || free_b < 400ull * 1024 * 1024) {
        printf("  -> пропуск: свободно %zu MiB / %zu MiB (нужно >=400 на контекст и буфер)\n",
               free_a >> 20, free_b >> 20);
        return true;
    }
    CK(cudaSetDevice(a));
    void *da = nullptr, *db = nullptr;
    CK(cudaMalloc(&da, BYTES));
    CK(cudaSetDevice(b));
    CK(cudaMalloc(&db, BYTES));
    size_t fa0 = 0, fb0 = 0;
    CK(cudaSetDevice(a)); CK(cudaMemGetInfo(&fa0, &tot));
    CK(cudaSetDevice(b)); CK(cudaMemGetInfo(&fb0, &tot));
    CK(cudaSetDevice(a));
    cudaError_t e1 = cudaDeviceEnablePeerAccess(b, 0);
    if (e1 != cudaSuccess && e1 != cudaErrorPeerAccessAlreadyEnabled) { printf("  EnablePeerAccess(a->b): %s\n", cudaGetErrorString(e1)); }
    cudaGetLastError();
    CK(cudaSetDevice(b));
    cudaError_t e2 = cudaDeviceEnablePeerAccess(a, 0);
    if (e2 != cudaSuccess && e2 != cudaErrorPeerAccessAlreadyEnabled) { printf("  EnablePeerAccess(b->a): %s\n", cudaGetErrorString(e2)); }
    cudaGetLastError();
    size_t fa1 = 0, fb1 = 0;
    CK(cudaSetDevice(a)); CK(cudaMemGetInfo(&fa1, &tot));
    CK(cudaSetDevice(b)); CK(cudaMemGetInfo(&fb1, &tot));
    printf("  цена включить peer access: GPU%d -%zu MiB, GPU%d -%zu MiB (до буферов)\n",
           a, (fa0 - fa1) >> 20, b, (fb0 - fb1) >> 20);

    cudaEvent_t e0, e1v, f0, f1v;
    CK(cudaSetDevice(a)); CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1v));
    CK(cudaSetDevice(b)); CK(cudaEventCreate(&f0)); CK(cudaEventCreate(&f1v));
    CK(cudaMemcpyPeerAsync(db, b, da, a, BYTES, 0));   // прогрев
    CK(cudaDeviceSynchronize());
    CK(cudaSetDevice(a)); CK(cudaEventRecord(e0));
    for (int i = 0; i < REPS; ++i) CK(cudaMemcpyPeerAsync(db, b, da, a, BYTES, 0));
    CK(cudaEventRecord(e1v)); CK(cudaEventSynchronize(e1v));
    float ms = 0; CK(cudaEventElapsedTime(&ms, e0, e1v));
    printf("  a->b: %.1f ГБ/с (%d x %zu MiB за %.1f мс)\n",
           (double) REPS * BYTES / (ms / 1000.0) / 1e9, REPS, BYTES >> 20, ms);

    CK(cudaSetDevice(b)); CK(cudaEventRecord(f0));
    for (int i = 0; i < REPS; ++i) CK(cudaMemcpyPeerAsync(da, a, db, b, BYTES, 0));
    CK(cudaEventRecord(f1v)); CK(cudaEventSynchronize(f1v));
    CK(cudaEventElapsedTime(&ms, f0, f1v));
    printf("  b->a: %.1f ГБ/с\n", (double) REPS * BYTES / (ms / 1000.0) / 1e9);

    CK(cudaSetDevice(a)); cudaFree(da);
    CK(cudaSetDevice(b)); cudaFree(db);
    return true;
}

int main() {
    int n = 0;
    if (cudaGetDeviceCount(&n) != cudaSuccess) { printf("нет устройств CUDA\n"); return 1; }
    printf("устройств: %d\n", n);
    for (int a = 0; a < n; ++a)
        for (int b = a + 1; b < n; ++b)
            pair(a, b);
    for (int d = 0; d < n; ++d) { cudaSetDevice(d); cudaDeviceReset(); }   // отдать контексты
    return 0;
}
