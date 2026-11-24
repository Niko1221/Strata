// What an engine made on the cards and in page-locked memory (buffers, streams, events, handles), each with how to
// release it; ~Engine releases them, newest first, so a lab process can make and drop engines around one loaded model.
// Every model's engine registers through own(): save_state and load_state make page-locked stages on the cards'
// threads at once, which unguarded pushes onto the one list would race on (engine_common.cu holds the list).
#pragma once

#define DPCT_PROFILING_ENABLED
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>

#include <functional>
#include <vector>

namespace fnx {

void own(std::function<void()> release);
// the releases, for ~Engine to run once no other thread can register (the cards are idle)
std::vector<std::function<void()>> take_owned();
// the current card, for a release that has to run on it; and the releases of what is made most often
inline int current_card() {
    int d = -1;
    d = dpct::get_current_device_id();
    return d;
}
inline void own_dev(void* p) {
    const int d = current_card();
    
    own([d, p] {
                 dpct::select_device(d); dpct::dpct_free(p, dpct::get_in_order_queue());
    });
}
inline void own_pinned(void* p) { own([p] {
                                            sycl::free(p, dpct::get_in_order_queue()); }); }
inline void own_stream(dpct::queue_ptr s) {
    const int d = current_card();
    
    own([d, s]() mutable {
                 dpct::select_device(d); dpct::get_current_device().destroy_queue(s);
    });
}
inline void own_event(dpct::event_ptr e) {
    const int d = current_card();
    
    own([d, e] {
                 dpct::select_device(d); dpct::destroy_event(e);
    });
}

}  // namespace fnx
