#pragma once

namespace strata::prefill {

struct DrainResult {
    int error = 0;
    int stream = -1;
};

// Do not continue submitting/waiting on GPU work after the first runtime error.
// Separate from the runtime so error paths can be tested without faulting a GPU.
template <class Sync>
DrainResult checked_stream_drain(Sync&& sync, int count) {
    for (int stream = 0; stream < count; ++stream) {
        const int error = static_cast<int>(sync(stream));
        if (error != 0) return {error, stream};
    }
    return {};
}

}  // namespace strata::prefill
