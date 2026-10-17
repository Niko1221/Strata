/* Этап VI (п. 9.1): загрузка ВСЕХ ядер Strata на Tesla V100 (sm_70) через CUDA driver API.
 *
 * Проверяет: отсутствие "no kernel image is available" / "invalid device function" для каждого
 * cubin и каждого ядра бинарника движка; ресурсы каждого ядра (регистры, shared, local) против
 * лимитов устройства; корректность загрузки/выгрузки модулей.
 *
 * Сборка: ~/strata185/cuda-12.8/bin/nvcc -ccbin ~/strata185/gcc185/usr/bin/g++-14 -O2 -o build/kernel-load-audit tests/cuda/kernel-load-audit.c -lcuda
 * Запуск: build/kernel-load-audit build/cubin-symbols.csv build/kernel-load-audit.csv
 */
#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAXSYM 4096
typedef struct { char cubin[256]; char **sym; int n; } Mod;

static int cmpstr(const void *a, const void *b) {
    const char *ka = *(char * const *)a, *kb = *(char * const *)b;
    return strcmp(ka, kb);
}

static int dev_attr(CUdevice dev, CUdevice_attribute a, int *out, const char *name) {
    CUresult r = cuDeviceGetAttribute(out, a, dev);
    if (r != CUDA_SUCCESS) { fprintf(stderr, "attr %s: error %d\n", name, (int)r); return 0; }
    return 1;
}

int main(int argc, char **argv) {
    const char *manifest = argc > 1 ? argv[1] : "build/cubin-symbols.csv";
    const char *outcsv   = argc > 2 ? argv[2] : "build/kernel-load-audit.csv";
    const char *basedir  = argc > 3 ? argv[3] : "build/kernels";

    CUresult r = cuInit(0);
    if (r != CUDA_SUCCESS) { fprintf(stderr, "cuInit: %d\n", (int)r); return 1; }
    CUdevice dev; int ndev = 0;
    cuDeviceGetCount(&ndev);
    if (ndev < 1) { fprintf(stderr, "no devices\n"); return 1; }
    dev = 0;
    char name[256]; cuDeviceGetName(name, sizeof name, dev);
    int cc_maj = 0, cc_min = 0, sm = 0, thr_block = 0, shared_block = 0, regs_block = 0, shared_optin = 0;
    dev_attr(dev, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, &cc_maj, "cc_major");
    dev_attr(dev, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, &cc_min, "cc_minor");
    dev_attr(dev, CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, &sm, "sm_count");
    dev_attr(dev, CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK, &thr_block, "max_threads_per_block");
    dev_attr(dev, CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK, &shared_block, "max_shared_per_block");
    dev_attr(dev, CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_BLOCK, &regs_block, "max_regs_per_block");
    dev_attr(dev, CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN, &shared_optin, "max_shared_optin");
    cuDeviceGetName(name, sizeof name, dev);
    /* driver API требует текущего контекста: удерживаем primary context устройства (общий с движком) */
    CUcontext ctx = NULL;
    CUresult cr = cuDevicePrimaryCtxRetain(&ctx, dev);
    if (cr != CUDA_SUCCESS) { fprintf(stderr, "cuDevicePrimaryCtxRetain: %d\n", (int)cr); return 1; }
    cuCtxSetCurrent(ctx);
    printf("device 0: %s cc=%d.%d SMs=%d max_threads/block=%d max_shared/block=%d max_shared_optin=%d max_regs/block=%d\n",
           name, cc_maj, cc_min, sm, thr_block, shared_block, shared_optin, regs_block);

    /* чтение манифеста */
    FILE *f = fopen(manifest, "r");
    if (!f) { fprintf(stderr, "no manifest %s\n", manifest); return 1; }
    static Mod mods[128]; int nmod = 0;
    char line[4096];
    if (fgets(line, sizeof line, f)) { /* заголовок CSV */ }
    while (fgets(line, sizeof line, f)) {
        char *comma = strchr(line, ',');
        if (!comma) continue;
        *comma = 0;
        char *cubin = line, *sym = comma + 1;
        sym[strcspn(sym, "\r\n")] = 0;
        int idx = -1;
        for (int i = 0; i < nmod; i++) if (!strcmp(mods[i].cubin, cubin)) { idx = i; break; }
        if (idx < 0) { if (nmod >= 128) break; idx = nmod++; strcpy(mods[idx].cubin, cubin); mods[idx].sym = NULL; mods[idx].n = 0; }
        mods[idx].sym = realloc(mods[idx].sym, sizeof(char*) * (mods[idx].n + 1));
        mods[idx].sym[mods[idx].n++] = strdup(sym);
    }
    fclose(f);

    FILE *out = fopen(outcsv, "w");
    fprintf(out, "cubin,symbol,load_result,regs,shared_bytes,local_bytes,max_threads_per_block\n");

    long tot_sym = 0, ok_sym = 0, fail_sym = 0, mod_ok = 0, mod_fail = 0;
    long max_regs = 0, max_shared = 0, max_local = 0;
    int over_regs = 0, over_shared = 0, over_threads = 0;
    int first_err_code = 0; char first_err[256] = "";

    for (int i = 0; i < nmod; i++) {
        char path[512]; snprintf(path, sizeof path, "%s/%s", basedir, mods[i].cubin);
        FILE *cf = fopen(path, "rb");
        if (!cf) { fprintf(stderr, "missing %s\n", path); mod_fail++; continue; }
        fseek(cf, 0, SEEK_END); long sz = ftell(cf); fseek(cf, 0, SEEK_SET);
        char *buf = malloc(sz); if (fread(buf, 1, sz, cf) != (size_t)sz) { fclose(cf); fprintf(stderr, "short read %s\n", path); return 1; } fclose(cf);

        CUmodule m;
        CUresult lr = cuModuleLoadData(&m, buf);
        if (lr != CUDA_SUCCESS) {
            const char *es = NULL; cuGetErrorString(lr, &es);
            mod_fail++; tot_sym += mods[i].n; fail_sym += mods[i].n;
            if (!first_err_code) { first_err_code = (int)lr; snprintf(first_err, sizeof first_err, "%s: %s", mods[i].cubin, es ? es : "?"); }
            for (int k = 0; k < mods[i].n; k++)
                fprintf(out, "%s,%s,MODULE_LOAD_FAIL_%d,,,,\n", mods[i].cubin, mods[i].sym[k], (int)lr);
            free(buf);
            continue;
        }
        mod_ok++;
        qsort(mods[i].sym, mods[i].n, sizeof(char*), cmpstr);
        for (int k = 0; k < mods[i].n; k++) {
            CUfunction fn;
            CUresult gr = cuModuleGetFunction(&fn, m, mods[i].sym[k]);
            tot_sym++;
            if (gr != CUDA_SUCCESS) {
                const char *es = NULL; cuGetErrorString(gr, &es);
                fail_sym++;
                if (!first_err_code) { first_err_code = (int)gr; snprintf(first_err, sizeof first_err, "%s: %s", mods[i].sym[k], es ? es : "?"); }
                fprintf(out, "%s,%s,GET_FUNCTION_FAIL_%d,,,,\n", mods[i].cubin, mods[i].sym[k], (int)gr);
                continue;
            }
            int regs = -1, sh = -1, lo = -1, mt = -1;
            cuFuncGetAttribute(&regs, CU_FUNC_ATTRIBUTE_NUM_REGS, fn);
            cuFuncGetAttribute(&sh, CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES, fn);
            cuFuncGetAttribute(&lo, CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES, fn);
            cuFuncGetAttribute(&mt, CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK, fn);
            ok_sym++;
            if (regs > max_regs) max_regs = regs;
            if (sh > max_shared) max_shared = sh;
            if (lo > max_local) max_local = lo;
            if (regs * 1 > regs_block) over_regs++;
            if (sh > shared_optin) over_shared++;
            if (mt > thr_block) over_threads++;
            fprintf(out, "%s,%s,OK,%d,%d,%d,%d\n", mods[i].cubin, mods[i].sym[k], regs, sh, lo, mt);
        }
        cuModuleUnload(m);
        free(buf);
    }
    fclose(out);

    printf("SUMMARY modules_ok=%d modules_fail=%d symbols_total=%ld symbols_ok=%ld symbols_fail=%ld\n",
           (int)mod_ok, (int)mod_fail, tot_sym, ok_sym, fail_sym);
    printf("SUMMARY max_regs=%ld max_shared_bytes=%ld max_local_bytes=%ld over_regs=%d over_shared=%d over_threads=%d\n",
           max_regs, max_shared, max_local, over_regs, over_shared, over_threads);
    if (first_err_code) printf("FIRST_ERROR code=%d %s\n", first_err_code, first_err);
    cuDevicePrimaryCtxRelease(0);
    printf("RESULT %s\n", (fail_sym == 0 && mod_fail == 0) ? "PASS" : "FAIL");
    return (fail_sym == 0 && mod_fail == 0) ? 0 : 2;
}
