// Две меры, которые нужны для разговора о межсоединении, при ОСТАНОВЛЕННОМ сервере модели
// (тогда на каждой карте свободно ~16 ГБ и буфер в 256 МиБ даёт настоящий потолок):
//   ./p2p-full-probe [MiB]           - P2P-пропускная способность по всем парам карт
//   ./p2p-full-probe [MiB] --host    - хост-путь каждой карты: поодиночке и все четыре сразу
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>

static size_t BYTES = 256ull << 20;
static const int REPS = 12;
#define CK(e) do { cudaError_t _e = (e); if (_e != cudaSuccess) { \
    printf("    CUDA error %s at line %d\n", cudaGetErrorString(_e), __LINE__); return; } } while (0)

void pair_bw(int a, int b) {
    int can = 0; cudaDeviceCanAccessPeer(&can, a, b);
    printf("GPU%d <-> GPU%d: canAccessPeer=%d\n", a, b, can);
    if (!can) return;
    CK(cudaSetDevice(a)); void *da = nullptr; CK(cudaMalloc(&da, BYTES));
    CK(cudaSetDevice(b)); void *db = nullptr; CK(cudaMalloc(&db, BYTES));
    CK(cudaSetDevice(a)); if (cudaDeviceEnablePeerAccess(b, 0) != cudaSuccess) cudaGetLastError();
    CK(cudaSetDevice(b)); if (cudaDeviceEnablePeerAccess(a, 0) != cudaSuccess) cudaGetLastError();
    for (int dir = 0; dir < 2; ++dir) {
        int src = dir ? b : a, dst = dir ? a : b;
        void *ps = dir ? db : da, *pd = dir ? da : db;
        cudaEvent_t e0, e1;
        CK(cudaSetDevice(src)); CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));
        CK(cudaMemcpyPeerAsync(pd, dst, ps, src, BYTES, 0)); CK(cudaDeviceSynchronize());
        CK(cudaEventRecord(e0));
        for (int i = 0; i < REPS; ++i) CK(cudaMemcpyPeerAsync(pd, dst, ps, src, BYTES, 0));
        CK(cudaEventRecord(e1)); CK(cudaEventSynchronize(e1));
        float ms = 0; CK(cudaEventElapsedTime(&ms, e0, e1));
        printf("  GPU%d -> GPU%d: %.1f ГБ/с\n", src, dst, (double) REPS * BYTES / (ms / 1000.0) / 1e9);
        CK(cudaEventDestroy(e0)); CK(cudaEventDestroy(e1));
    }
    CK(cudaSetDevice(a)); cudaFree(da); CK(cudaSetDevice(b)); cudaFree(db);
}

// k устройств работают одновременно; считаем каждую по её собственным событиям
void host_measure(const int *devs, int k, int dir) {
    static float *dev[8], *pin[8];
    for (int i = 0; i < k; ++i) {
        int d = devs[i];
        CK(cudaSetDevice(d));
        if (dev[i] == nullptr) { CK(cudaMalloc((void**)&dev[i], BYTES)); CK(cudaMallocHost((void**)&pin[i], BYTES)); memset(pin[i], 1, BYTES); }
    }
    cudaEvent_t st[8], en[8];
    for (int i = 0; i < k; ++i) {
        CK(cudaSetDevice(devs[i])); CK(cudaEventCreate(&st[i])); CK(cudaEventCreate(&en[i]));
        CK(cudaEventRecord(st[i]));
        for (int r = 0; r < REPS; ++r) {
            if (dir == 0) CK(cudaMemcpyAsync(dev[i], pin[i], BYTES, cudaMemcpyHostToDevice, 0));
            else          CK(cudaMemcpyAsync(pin[i], dev[i], BYTES, cudaMemcpyDeviceToHost, 0));
        }
        CK(cudaEventRecord(en[i]));
    }
    for (int i = 0; i < k; ++i) {
        CK(cudaSetDevice(devs[i])); CK(cudaEventSynchronize(en[i]));
        float ms = 0; CK(cudaEventElapsedTime(&ms, st[i], en[i]));
        printf("  %s GPU%d: %.2f ГБ/с%s\n", dir ? "D2H" : "H2D", devs[i],
               (double) REPS * BYTES / (ms / 1000.0) / 1e9, k > 1 ? " (все четыре одновременно)" : "");
        CK(cudaEventDestroy(st[i])); CK(cudaEventDestroy(en[i]));
    }
}

void host_bw(int n) {
    int all[8]; for (int i = 0; i < n; ++i) all[i] = i;
    for (int dir = 0; dir < 2; ++dir) {
        for (int d = 0; d < n; ++d) { int one = d; host_measure(&one, 1, dir); }
        host_measure(all, n, dir);
    }
}

int main(int argc, char **argv) {
    if (argc > 1) BYTES = (size_t) atoll(argv[1]) << 20;
    bool host = argc > 2 && !strcmp(argv[2], "--host");
    int n = 0;
    if (cudaGetDeviceCount(&n) != cudaSuccess) { printf("нет устройств CUDA\n"); return 1; }
    printf("устройств: %d, буфер: %zu МиБ, повторов: %d\n", n, BYTES >> 20, REPS);
    if (host) host_bw(n);
    else for (int a = 0; a < n; ++a) for (int b = a + 1; b < n; ++b) pair_bw(a, b);
    for (int d = 0; d < n; ++d) { cudaSetDevice(d); cudaDeviceReset(); }
    return 0;
}
