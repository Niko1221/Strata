#pragma once
#include <string>

namespace strata::prefill {
// Experimental SM120 Q8_0 [10240,2560] projection, exactly eight columns.
// One owner and one ordered stream; prepare before graph capture. The original
// native Q8_1 codes and half scales are preserved, with a different reduction.
class DenseQ8T8 {
public:
    DenseQ8T8() = default;
    ~DenseQ8T8();
    DenseQ8T8(const DenseQ8T8&) = delete;
    DenseQ8T8& operator=(const DenseQ8T8&) = delete;
    bool open(void* stream, std::string& error);
    // false enqueues nothing: callers retain their native vector fallback.
    bool run(int type, const void* w, const void* native_q8_1, float* y,
             int k, int n, int t, void* stream);
    void close();
private:
    void* state_ = nullptr;
};
}
