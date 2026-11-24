#pragma once

#include "kernels.hpp"
#include <vector>

namespace fnx {

class NativeQsa {
public:
    NativeQsa(dpct::queue_ptr queue, int max_rows, int max_pools);
    ~NativeQsa();
    NativeQsa(const NativeQsa&) = delete;
    NativeQsa& operator=(const NativeQsa&) = delete;

    size_t bytes() const { return bytes_; }
    void scores(const float* query, const uint16_t* pooled, const KvPages& pages,
                const int* position, int rows, int active_pools, float* output);

private:
    dpct::queue_ptr queue_;
    int max_rows_, max_pools_, tile_;
    size_t bytes_ = 0;
    std::vector<void*> buffers_;
    float *keys_, *dots_, *dead_;
    int* steps_;
};

} // namespace fnx
