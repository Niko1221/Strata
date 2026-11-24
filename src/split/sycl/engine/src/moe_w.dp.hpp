#define DPCT_PROFILING_ENABLED
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
// The routed experts' matrices and the prompt path's jobs, shared by kernels.cu
// (moe.cuh) and moe_mma.cu.
struct MoeW {
    const char* wa;      // the experts' matrices, expert stride sa
    size_t sa;
    size_t row_a;        // row bytes
    int nrows, ncols;
    const char* const* ep;   // or each expert's matrix (VRAM or mapped host memory): DW::ep
};
__dpct_inline__ const char *moe_expert(const MoeW &w, int e) {
    return w.ep ? w.ep[e] : w.wa + (size_t)e * w.sa;
}

// a BigJob in shared memory (thread 0 can't copy 520 bytes alone without stalling the block)
__dpct_inline__ void load_bigjob(BigJob &jb, const BigJob *src) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    for (int k = item_ct1.get_local_id(2); k < (int)(sizeof(BigJob) / 4);
         k += item_ct1.get_local_range(2))
        reinterpret_cast<int*>(&jb)[k] = reinterpret_cast<const int*>(src)[k];
    
    item_ct1.barrier();
}
