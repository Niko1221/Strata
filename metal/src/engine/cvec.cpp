// metal/src/engine/cvec.cpp - the control vector's tables on Metal (include/strata/kernels/cvec.hpp; the CUDA side is
// src/kernels/cuda/cvec.cu).  The loader (generate.cpp's load_control_vectors) uploads the tables here; the decode
// window and the prompt chains (window.cpp) put them in their own buffers and apply them after each steered layer's
// FFN write (msl/cvec.metal).  cvec_apply / cvec_tables stay stubs: they serve the CUDA graphs' and the CUDA prefill's
// paths, which the Metal engine does not run.
#include "strata/kernels/cvec.hpp"

#include <vector>

namespace strata::metal_engine {

namespace {
std::vector<float> g_dir, g_s;
int g_on = 0;                  // the host's flag; the device copy is the window's (cvec_bind_flag)
int* g_on_dev = nullptr;
}  // namespace

const std::vector<float>& cvec_dir_host() { return g_dir; }
const std::vector<float>& cvec_s_host() { return g_s; }
int cvec_on_host() { return g_on; }
// the window's flag buffer (shared storage): cvec_set_enabled writes it from now on
void cvec_bind_flag(int* dev) {
    g_on_dev = dev;
    if (g_on_dev != nullptr) *g_on_dev = g_on;
}

}  // namespace strata::metal_engine

namespace strata::kernels {

namespace {
Cvec g_cvec;
}  // namespace

const Cvec& cvec() { return g_cvec; }

bool cvec_upload(const std::vector<float>& dir, const std::vector<float>& s, int mode, int first, int last,
                 int64_t n_embd, int64_t hc, std::string& err) {
    if (n_embd < 1 || n_embd > 256 * 16) { err = "control vector: unsupported n_embd"; return false; }
    if (s.empty() || dir.size() != s.size() * (size_t) n_embd) { err = "control vector: bad table sizes"; return false; }
    namespace me = strata::metal_engine;
    me::g_dir = dir;
    me::g_s = s;
    me::g_on = 1;
    if (me::g_on_dev != nullptr) *me::g_on_dev = 1;
    // the host tables stand for the device pointers: on Metal only `loaded()` and `covers()` read them outside window.cpp
    g_cvec.dir = me::g_dir.data();
    g_cvec.s = me::g_s.data();
    g_cvec.on = &me::g_on;
    g_cvec.mode = mode;
    g_cvec.first = first;
    g_cvec.last = last;
    g_cvec.n_embd = n_embd;
    g_cvec.hc = hc;
    g_cvec.steered.assign(s.size(), false);
    for (size_t l = 0; l < s.size(); ++l) g_cvec.steered[l] = s[l] != 0.0f;
    return true;
}

bool cvec_replicate(std::string&) { return true; }   // one device

// between requests: nothing is in flight on Metal then (every command buffer is waited for), so the shared-storage
// flag is written directly
void cvec_set_enabled(bool on) {
    namespace me = strata::metal_engine;
    if (!g_cvec.loaded()) return;
    me::g_on = on ? 1 : 0;
    if (me::g_on_dev != nullptr) *me::g_on_dev = me::g_on;
}

bool cvec_enabled() { return g_cvec.loaded() && strata::metal_engine::g_on != 0; }

}  // namespace strata::kernels
