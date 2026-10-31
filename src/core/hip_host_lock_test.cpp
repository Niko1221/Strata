// src/core/hip_host_lock_test.cpp - STRATA_HIP_MLOCK's bookkeeping (strata/platform/hip_host_lock.hpp) on plain anonymous
// pages, no GPU call: what the kernel locked is VmLck in /proc/self/status.  Exit 77 (skip) when mlock is refused.
#include "strata/platform/hip_host_lock.hpp"

#include <sys/mman.h>
#include <unistd.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>

static int fails = 0;
#define CHECK(c) do { if (!(c)) { std::printf("FAIL line %d: %s\n", __LINE__, #c); ++fails; } } while (0)

static long vm_lck_kb() {
    std::ifstream f("/proc/self/status");
    std::string k;
    long v = -1;
    while (f >> k)
        if (k == "VmLck:") { f >> v; break; }
    return v;
}

int main() {
    setenv("STRATA_HIP_MLOCK", "1", 1);   // before the first lock: on() reads it once
    namespace d = strata::hip_compat::host_lock_detail;
    const long pg = sysconf(_SC_PAGESIZE), kb = pg / 1024;
    uint8_t* m = (uint8_t*) mmap(nullptr, 4 * pg, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    CHECK(m != MAP_FAILED);
    const long base = vm_lck_kb();
    d::lock(m, 2 * pg);
    if (vm_lck_kb() == base) { std::printf("SKIP: mlock refused (memlock limit)\n"); return 77; }
    CHECK(vm_lck_kb() == base + 2 * kb);
    d::unlock(m);
    CHECK(vm_lck_kb() == base);
    // two ranges that share page 1: unlocking either one keeps the shared page locked for the other
    d::lock(m, pg + pg / 2);
    d::lock(m + pg + pg / 2, pg + pg / 2);
    CHECK(vm_lck_kb() == base + 3 * kb);
    d::unlock(m);
    CHECK(vm_lck_kb() == base + 2 * kb);
    d::unlock(m + pg + pg / 2);
    CHECK(vm_lck_kb() == base + kb);   // the shared page: until it is freed
    d::unlock(nullptr);
    d::unlock(m + 1);                   // never locked here: nothing happens
    CHECK(vm_lck_kb() == base + kb);
    munmap(m, 4 * pg);
    CHECK(vm_lck_kb() == base);
    std::printf("%s (%d failures)\n", fails ? "FAILED" : "ok", fails);
    return fails ? 1 : 0;
}
