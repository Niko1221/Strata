// LD_PRELOAD diagnostic: every plain global operator new returns 64-byte aligned memory.
// If CachyOS's v3 libamdhip64 stops crashing with this preloaded, the fault is the
// under-aligned `new roc::VirtualGPU` (ReferenceCountedObject::operator new(size_t) ->
// ::operator new(size), 16-byte heap alignment), as described in ROCm/rocm-systems#14003.
//
//   g++ -O2 -shared -fPIC -o align64_new.so align64_new.cpp
//   LD_PRELOAD=./align64_new.so ./narrow mS
//
// glibc's free() accepts posix_memalign() pointers, so the matching deletes call free().
#include <cstdlib>
#include <new>

static void* alloc64(std::size_t n) {
    void* p = nullptr;
    if (posix_memalign(&p, 64, n ? n : 1) != 0) throw std::bad_alloc();
    return p;
}

void* operator new(std::size_t n) { return alloc64(n); }
void* operator new[](std::size_t n) { return alloc64(n); }
void* operator new(std::size_t n, const std::nothrow_t&) noexcept {
    void* p = nullptr;
    return posix_memalign(&p, 64, n ? n : 1) == 0 ? p : nullptr;
}
void operator delete(void* p) noexcept { std::free(p); }
void operator delete[](void* p) noexcept { std::free(p); }
void operator delete(void* p, std::size_t) noexcept { std::free(p); }
void operator delete[](void* p, std::size_t) noexcept { std::free(p); }
