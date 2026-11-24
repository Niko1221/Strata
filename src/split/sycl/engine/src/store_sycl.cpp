#include "store.hpp"

#include <stdexcept>

namespace fnx {
namespace {
[[noreturn]] void unsupported() {
    throw std::runtime_error("SYCL split: the shared-memory lab holder is not supported");
}
}
void* Store::dev(int, size_t) { unsupported(); }
void* Store::host(int, size_t) { unsupported(); }
void Store::free_dev(int, void*, size_t) { unsupported(); }
size_t Store::bytes(int) const { unsupported(); }
void Store::attach(const std::vector<Chunk>&) { unsupported(); }
const void* Store::move(const void*) const { unsupported(); }
void* Store::take(int, size_t) { unsupported(); }
void Store::reserve(int, size_t) { unsupported(); }
}  // namespace fnx
