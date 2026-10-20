// metal/src/engine/cvec.cpp - the control vector (--control-vector-scaled, strata/kernels/cvec.hpp) on Metal: the
// tables the shared loader uploads and the per-request switch.  The decode window applies it (window.cpp, cvec_at;
// the kernel is msl/cvec.metal): unified memory, so `dir` and `s` are host memory the window copies into its own
// buffer.  The switch is read when a window or a prompt chunk is encoded - the window caches its encodings per
// switch state - so a request with the vector off runs exactly the stock dispatches.
//
// cvec_apply and the prompt path's fused gr_write_cvec_norm_rs are the CUDA engine's host-side hooks
// (layer.cpp, verify.cpp, prefill.cpp): Metal does not run those paths, so they stay stubs (stubs.cpp).
#include "strata/kernels/cvec.hpp"

namespace strata::kernels {
namespace {

constexpr int64_t kMaxEmbd = 4096;   // cvec.metal keeps 16 values a thread of 256

Cvec g_cvec;
std::vector<float> g_dir, g_s;
int g_on = 0;

}  // namespace

const Cvec& cvec() { return g_cvec; }

bool cvec_upload(const std::vector<float>& dir, const std::vector<float>& s, int mode, int first, int last,
                 int64_t n_embd, int64_t hc, std::string& err) {
    if (n_embd < 1 || n_embd > kMaxEmbd) { err = "control vector: unsupported n_embd"; return false; }
    if (s.empty() || dir.size() != s.size() * (size_t) n_embd) { err = "control vector: bad table sizes"; return false; }
    g_dir = dir;
    g_s = s;
    g_on = 1;
    g_cvec.dir = g_dir.data();
    g_cvec.s = g_s.data();
    g_cvec.on = &g_on;
    g_cvec.mode = mode;
    g_cvec.first = first;
    g_cvec.last = last;
    g_cvec.n_embd = n_embd;
    g_cvec.hc = hc;
    g_cvec.steered.assign(s.size(), false);
    for (size_t l = 0; l < s.size(); ++l) g_cvec.steered[l] = s[l] != 0.0f;
    return true;
}

// one GPU, one memory: nothing to copy
bool cvec_replicate(std::string&) { return true; }

void cvec_set_enabled(bool on) {
    if (g_cvec.loaded()) g_on = on ? 1 : 0;
}

bool cvec_enabled() { return g_cvec.loaded() && g_on != 0; }

// no device tables for a caller's own kernel (the CUDA prompt path's STRATA_CVEC_FUSE): the window encodes the vector
bool cvec_tables(const float**, const float**, const int**) { return false; }

}  // namespace strata::kernels
