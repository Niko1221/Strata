#pragma once
#include "ops.cuh"
#include <limits>
#include <vector>

namespace strata::qwen38 {
// Offsets are checked before any CUDA allocation. Views never own memory.
struct WeightSpan {
    void* data=nullptr; size_t bytes=0;
    template<class T=float>T* as()const{return static_cast<T*>(data);}
};
class WeightLayout {
public:
    struct Entry {size_t offset,bytes;};
private:
    std::vector<Entry> entries_;
    size_t bytes_=0,payload_=0;
public:
    static size_t add_checked(size_t a,size_t b){
        if(b>std::numeric_limits<size_t>::max()-a)throw std::overflow_error("weight arena size overflow");
        return a+b;
    }
    static size_t f32_bytes(size_t elements){
        if(elements>std::numeric_limits<size_t>::max()/4)throw std::overflow_error("F32 weight size overflow");
        return elements*4;
    }
    void append(size_t bytes){
        if(!bytes)throw std::runtime_error("empty weight arena span");
        const size_t offset=add_checked(bytes_,(256-bytes_%256)%256);
        const size_t end=add_checked(offset,bytes),payload=add_checked(payload_,bytes);
        entries_.push_back({offset,bytes});bytes_=end;payload_=payload;
    }
    size_t bytes()const{return bytes_;}
    size_t payload()const{return payload_;}
    const std::vector<Entry>& entries()const{return entries_;}
};
class WeightArena {
    Buffer owner_;
    std::vector<WeightLayout::Entry> entries_;
public:
    explicit WeightArena(const WeightLayout& layout):entries_(layout.entries()){
        if(!layout.bytes())throw std::runtime_error("empty weight arena");
        owner_.allocate(layout.bytes());
    }
    WeightSpan view(size_t index)const{
        const auto& e=entries_.at(index);
        if(e.offset>owner_.bytes||e.bytes>owner_.bytes-e.offset)throw std::runtime_error("weight view outside arena");
        return {owner_.as<uint8_t>()+e.offset,e.bytes};
    }
    size_t bytes()const{return owner_.bytes;}
};
}
