// Отрицательный контроль архитектуры (п. 9.1: «проверка ошибок no kernel image is available /
// invalid device function»).
//   build/negctl/negctl_sm80.cubin  — обычное ядро, собранное с целью sm_80: загрузка на V100
//     (sm_70) обязана дать CUDA_ERROR_NO_BINARY_FOR_GPU (209) = «no kernel image is available».
//   negctl_bf16_mma использует m16n8k16 bf16 mma: компилируется для sm_80 и ОТКАЗЫВАЕТСЯ
//     компилироваться для sm_70 («Feature '.m16n8k16' requires .target sm_80 or higher») —
//     это тот же гейт, который в движке закрыт ветками #if __CUDA_ARCH__ >= 750.
#include <cuda_bf16.h>
extern "C" __global__ void negctl_trivial(float *out) { out[threadIdx.x] = threadIdx.x * 2.0f; }
extern "C" __global__ void negctl_bf16_mma(float *out, const unsigned *a, const unsigned *b) {
    float c[4] = {0.f, 0.f, 0.f, 0.f};
    unsigned a0 = a[threadIdx.x], a1 = a[threadIdx.x + 32], a2 = a[threadIdx.x + 64], a3 = a[threadIdx.x + 96];
    unsigned b0 = b[threadIdx.x], b1 = b[threadIdx.x + 32];
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
                 : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
    out[threadIdx.x] = c[0] + c[1] + c[2] + c[3];
}
