#pragma once
#include <atomic>
#include <stdexcept>
namespace strata::qwen38 {
struct Cancelled{};
class LayerFence {
    int interval_;
public:
    explicit LayerFence(int interval):interval_(interval){
        if(interval<1||interval>8)throw std::runtime_error("layer fence interval must be 1..8");
    }
    void before(const std::atomic<bool>& stop)const{if(stop.load())throw Cancelled{};}
    template<class F>void after(int layer,const std::atomic<bool>& stop,F fence)const{
        if((layer+1)%interval_==0){fence();before(stop);}
    }
};
}
