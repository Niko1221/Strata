// Test-only actual host path. Include after q36_engine.dp.cpp's definitions.
#ifdef STRATA_Q36_TEST_HOOKS
#include "q36_test_hooks.hpp"
namespace strata::qwen35::test {
AddRmsHostResult add_rms_host(sycl::queue* q, float* x, const float* y,
                             const float* w, float* h, void* xq, int n, int T,
                             float eps, float* reference_x, float* reference_h) {
    if (!q || !q->has_property<sycl::property::queue::in_order>())
        throw std::invalid_argument("add_rms_host requires a borrowed in-order queue");
    if (q->get_context() != dpct::get_current_device().in_order_queue().get_context())
        throw std::invalid_argument("add_rms_host queue must share the current DPCT context");
    if (!x || !y || !w || !h || !xq || n != 2048 || T < 1 || T > 8 || !(eps > 0))
        throw std::invalid_argument("invalid add_rms_host input geometry/pointers");
    // Value initialization is deliberate: no full-model init and no uninitialized
    // activation pointers are used. All real allocations belong to the caller.
    Model model{}; model.g.n_embd=n; model.eps=eps;
    Layer layer{}; layer.post_norm=w;
    Engine engine{}; engine.m=&model; engine.st=q;
    engine.x=x; engine.y=const_cast<float*>(y); engine.h=h; engine.xq=xq;
    engine.fast_k=true; engine.verify_k=true; engine.use_graph=false;
    struct ScratchGuard {
        Engine& e; sycl::queue& q;
        ~ScratchGuard() noexcept {
            try { q.wait_and_throw(); } catch (...) {}
            // Only these three buffers are allocated by add_rms_post/verify_q8.
            for (void* p : {static_cast<void*>(e.vx_),static_cast<void*>(e.vh_),e.vq})
                if (p) { try { sycl::free(p,q); } catch (...) {} }
            e.vx_=e.vh_=nullptr; e.vq=nullptr;
        }
    } scratch{engine,*q};
    engine.add_rms_post(layer,T);
    q->wait_and_throw();
    if (reference_x) q->memcpy(reference_x,engine.vx_,size_t(n)*T*sizeof(float)).wait_and_throw();
    if (reference_h) q->memcpy(reference_h,engine.vh_,size_t(n)*T*sizeof(float)).wait_and_throw();
    return {engine.v_calls,engine.v_bad,engine.xq_post};
}
}
#endif
