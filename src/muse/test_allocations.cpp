// Test-only allocation faults. Kept in a separate translation unit so optimized
// callers do not mistake the malloc/free implementation of replacement new/delete
// for a mismatched allocation. No hooks are linked into strata-muse itself.
#include <cstddef>
#include <cstdlib>
#include <new>

namespace allocation_fault {
long countdown = -1;
size_t count = 0;
bool measuring = false;
}
void* operator new(std::size_t bytes) {
    if (allocation_fault::measuring) ++allocation_fault::count;
    if (allocation_fault::countdown == 0) {
        allocation_fault::countdown = -1;
        throw std::bad_alloc();
    }
    if (allocation_fault::countdown > 0) --allocation_fault::countdown;
    if (void* p = std::malloc(bytes ? bytes : 1)) return p;
    throw std::bad_alloc();
}
void* operator new[](std::size_t bytes) { return ::operator new(bytes); }
void operator delete(void* p) noexcept { std::free(p); }
void operator delete[](void* p) noexcept { std::free(p); }
void operator delete(void* p, std::size_t) noexcept { std::free(p); }
void operator delete[](void* p, std::size_t) noexcept { std::free(p); }
