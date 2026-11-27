// engine/src/inkling2/cuda/ink_pdl.cuh - programmatic dependent launch for the decode graphs: the host half (PERF-STEP lane)
//
// MIT License
// Copyright (c) 2026 Niko1221 and the Strata contributors
//
// What it does: after a decode graph is captured, every kernel->kernel edge whose downstream kernel is registered (it starts with INK_PDL_WAIT(), see
// ink_pdl_dev.cuh) is changed to a programmatic edge on the "all blocks of the upstream kernel have begun" port. The downstream kernel's blocks then start
// while the upstream kernel's last blocks still run, wait at INK_PDL_WAIT() until the upstream kernel has completed and its writes are visible, and go on.
// Results are exactly those of the plain graph; only the launch and the ramp-up overlap. A kernel that is not registered keeps its ordinary edge, and so does
// any edge that is not kernel->kernel (copies, memsets, events). No launch site changes: one call replaces cudaGraphInstantiate.
//
// Safety argument: a registered kernel B behind a programmatic edge from A is only ever started early relative to A. Everything B reads or writes that an
// earlier kernel touches comes after B's INK_PDL_WAIT(). B's own predecessors other than A are ordinary edges (complete before B starts). A kernel behind B
// that is not registered has an ordinary edge to B and starts after B completed; B completes after A has (B waited for it in every block).
// The switch is STRATA_INK_PDL=0|1 (default: on for the RTX 5090 only, see wanted()), the port STRATA_INK_PDL_PORT=2 (all blocks begun, default) | 1 (kernel's own trigger / exit).
// On a card below sm_90, with CUDA below 12.3 or when a kernel's loaded code is below sm_90 (no wait in it), nothing is rewritten.
#pragma once

#include "ink_pdl_dev.cuh"

#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#if defined(CUDART_VERSION) && CUDART_VERSION >= 13000
#define INK_PDL_API 1
#define INK_PDL_GET_EDGES cudaGraphGetEdges
#define INK_PDL_ADD_DEPS cudaGraphAddDependencies
#define INK_PDL_REM_DEPS cudaGraphRemoveDependencies
#elif defined(CUDART_VERSION) && CUDART_VERSION >= 12030
#define INK_PDL_API 1
#define INK_PDL_GET_EDGES cudaGraphGetEdges_v2
#define INK_PDL_ADD_DEPS cudaGraphAddDependencies_v2
#define INK_PDL_REM_DEPS cudaGraphRemoveDependencies_v2
#endif

namespace strata::ink::pdl {

// On by default only on the RTX 5090 (cc 12.x with 160..192 SMs; measured: tokens identical, plain decode +5 %, nothing slower; PERF-STEP RESULTS.md).
// STRATA_INK_PDL=1 turns it on for any sm_90+ card, STRATA_INK_PDL=0 off everywhere.
constexpr int kDefaultOnMajor = 12;

inline int device_major() {
    int dev = 0, major = 0;
    if (cudaGetDevice(&dev) != cudaSuccess) { (void)cudaGetLastError(); return 0; }
    if (cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, dev) != cudaSuccess) { (void)cudaGetLastError(); return 0; }
    return major;
}

inline bool wanted() {
    static const int env = [] {
        const char* e = std::getenv("STRATA_INK_PDL");
        return (e != nullptr && *e != '\0') ? (std::atoi(e) != 0 ? 1 : 0) : -1;
    }();
    if (env >= 0) return env == 1;
    int dev = 0, sms = 0;   // the measured card: the 5090 row (160..192 SMs); other cc 12.x cards (5060 Ti, ...) stay off until measured
    if (device_major() != kDefaultOnMajor || cudaGetDevice(&dev) != cudaSuccess || cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev) != cudaSuccess) { (void)cudaGetLastError(); return false; }
    return sms >= 160 && sms <= 192;
}

inline int port_mode() {
    static const int v = [] {
        const char* e = std::getenv("STRATA_INK_PDL_PORT");
        return (e != nullptr && std::atoi(e) == 1) ? 1 : 2;
    }();
    return v;
}

struct Registry {
    std::mutex mu;
    std::unordered_set<const void*> fns;
    std::unordered_map<const void*, int> ok_cache;   // fn -> 1 usable on this device, 0 not
    bool filled = false;
    bool logged = false;
    long graphs = 0, edges = 0, kernel_edges = 0;
};
inline Registry& reg() {
    static Registry r;
    return r;
}

/// Register a kernel (a template instance too: `add(k<1, false>)`). Call only for kernels whose first statement is INK_PDL_WAIT().
template <typename... A>
inline void add(void (*k)(A...)) {
    reg().fns.insert(reinterpret_cast<const void*>(k));
}

/// Same, with the kernel's name; STRATA_INK_PDL_SKIP=a,b skips every kernel whose name contains "a" or "b" (a bisect tool for a slowdown), STRATA_INK_PDL_ONLY=a,b keeps only those.
inline bool name_listed(const char* name, const char* list) {
    std::string l(list), tok;
    for (size_t i = 0; i <= l.size(); ++i) {
        if (i == l.size() || l[i] == ',') {
            if (!tok.empty() && std::string(name).find(tok) != std::string::npos) return true;
            tok.clear();
        } else {
            tok += l[i];
        }
    }
    return false;
}
template <typename... A>
inline void add_named(const char* name, void (*k)(A...)) {
    const char* skip = std::getenv("STRATA_INK_PDL_SKIP");
    const char* only = std::getenv("STRATA_INK_PDL_ONLY");
    if (skip != nullptr && *skip != '\0' && name_listed(name, skip)) return;
    if (only != nullptr && *only != '\0' && !name_listed(name, only)) return;
    reg().fns.insert(reinterpret_cast<const void*>(k));
}

inline void register_all();   // defined in ink_pdl_kernels.cuh, after every kernel is declared

#ifdef INK_PDL_API
/// The loaded code of the kernel is for sm_90 or above (so INK_PDL_WAIT() is in it).
inline bool code_has_wait(const void* fn) {
    cudaFuncAttributes fa{};
    if (cudaFuncGetAttributes(&fa, fn) != cudaSuccess) {
        (void)cudaGetLastError();
        return false;
    }
    return fa.binaryVersion >= 90 && fa.ptxVersion >= 90;
}

inline bool device_ok() {
    int dev = 0, major = 0;
    if (cudaGetDevice(&dev) != cudaSuccess) { (void)cudaGetLastError(); return false; }
    if (cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, dev) != cudaSuccess) { (void)cudaGetLastError(); return false; }
    return major >= 9;
}

struct RewriteStats { int kernels = 0, kernel_edges = 0, converted = 0; };

/// Convert the eligible edges of `g` (not the caller's graph: pass a clone). Returns false when the API refused (the graph is then unusable: use the original).
inline bool rewrite(cudaGraph_t g, RewriteStats& st) {
    size_t ne = 0;
    if (INK_PDL_GET_EDGES(g, nullptr, nullptr, nullptr, &ne) != cudaSuccess) { (void)cudaGetLastError(); return false; }
    if (ne == 0) return true;
    std::vector<cudaGraphNode_t> from(ne), to(ne);
    std::vector<cudaGraphEdgeData> ed(ne);
    if (INK_PDL_GET_EDGES(g, from.data(), to.data(), ed.data(), &ne) != cudaSuccess) { (void)cudaGetLastError(); return false; }
    Registry& R = reg();
    std::unordered_map<cudaGraphNode_t, int> kind;   // 0 not a kernel, 1 a kernel, 2 a kernel that may start early
    std::unordered_map<cudaGraphNode_t, long> nblk;  // blocks of a kernel node
    // measured on the 5090: an early-started kernel behind a tiny upstream kernel (1..8 blocks: top-k, bin) only spins; these two knobs bisect that
    static const long min_up = [] { const char* e = std::getenv("STRATA_INK_PDL_MINBLK"); return e ? std::atol(e) : 2L; }();      // the upstream kernel has at least this many blocks (default 2: behind a one-block kernel an early start only spins; measured on the 5090, exact 121.9 vs 139.1 tok/s)
    static const long max_down = [] { const char* e = std::getenv("STRATA_INK_PDL_MAXBLK"); return e ? std::atol(e) : 0L; }();   // the downstream kernel has at most this many (0 = no limit)
    auto classify = [&](cudaGraphNode_t n) -> int {
        auto it = kind.find(n);
        if (it != kind.end()) return it->second;
        cudaGraphNodeType t = cudaGraphNodeTypeEmpty;
        int k = 0;
        if (cudaGraphNodeGetType(n, &t) == cudaSuccess && t == cudaGraphNodeTypeKernel) {
            cudaKernelNodeParams p{};
            k = 1;
            nblk[n] = 0;
            if (cudaGraphKernelNodeGetParams(n, &p) == cudaSuccess && p.func != nullptr) {
                nblk[n] = (long)p.gridDim.x * (long)p.gridDim.y * (long)p.gridDim.z;
                std::lock_guard<std::mutex> lock(R.mu);
                if (R.fns.count(p.func) != 0) {
                    auto c = R.ok_cache.find(p.func);
                    if (c == R.ok_cache.end()) c = R.ok_cache.emplace(p.func, code_has_wait(p.func) ? 1 : 0).first;
                    if (c->second) k = 2;
                }
            } else {
                (void)cudaGetLastError();
            }
        }
        kind[n] = k;
        return k;
    };
    std::vector<cudaGraphNode_t> rf, rt;
    std::vector<cudaGraphEdgeData> olde, newe;
    const unsigned char port = (port_mode() == 1) ? (unsigned char)cudaGraphKernelNodePortProgrammatic : (unsigned char)cudaGraphKernelNodePortLaunchCompletion;
    for (size_t i = 0; i < ne; ++i) {
        const int a = classify(from[i]), b = classify(to[i]);
        if (a >= 1 && b >= 1) ++st.kernel_edges;
        if (ed[i].type != 0 || ed[i].from_port != 0 || a < 1 || b != 2) continue;
        if (nblk[from[i]] < min_up || (max_down > 0 && nblk[to[i]] > max_down)) continue;
        rf.push_back(from[i]); rt.push_back(to[i]); olde.push_back(ed[i]);
        cudaGraphEdgeData n = ed[i];
        n.type = (unsigned char)cudaGraphDependencyTypeProgrammatic;
        n.from_port = port;
        newe.push_back(n);
    }
    for (const auto& kv : kind) if (kv.second) ++st.kernels;
    st.converted = (int)rf.size();
    if (rf.empty()) return true;
    if (INK_PDL_REM_DEPS(g, rf.data(), rt.data(), olde.data(), rf.size()) != cudaSuccess) { (void)cudaGetLastError(); return false; }
    if (INK_PDL_ADD_DEPS(g, rf.data(), rt.data(), newe.data(), rf.size()) != cudaSuccess) { (void)cudaGetLastError(); return false; }
    return true;
}
#endif  // INK_PDL_API

/// Drop-in for `cudaGraphInstantiate(exec, g, nullptr, nullptr, 0)`: the graph is instantiated with programmatic edges when that is on and possible, as it is
/// otherwise. `g` is not modified (the caller destroys it as before). `tag` names the graph in the one log line.
inline cudaError_t instantiate(cudaGraphExec_t* exec, cudaGraph_t g, const char* tag = "graph") {
#ifdef INK_PDL_API
    if (wanted() && device_ok()) {
        Registry& R = reg();
        bool fill = false;
        {
            std::lock_guard<std::mutex> lock(R.mu);
            if (!R.filled) { R.filled = true; fill = true; }
        }
        if (fill) register_all();
        cudaGraph_t c = nullptr;
        if (cudaGraphClone(&c, g) == cudaSuccess && c != nullptr) {
            RewriteStats st;
            const bool ok = rewrite(c, st);
            if (ok && st.converted > 0 && cudaGraphInstantiate(exec, c, nullptr, nullptr, 0) == cudaSuccess) {
                cudaGraphDestroy(c);
                std::lock_guard<std::mutex> lock(R.mu);
                ++R.graphs; R.edges += st.converted; R.kernel_edges += st.kernel_edges;
                if (!R.logged || std::getenv("STRATA_INK_PDL_LOG") != nullptr) {
                    R.logged = true;
                    std::fprintf(stderr, "[InkPDL] %s: %d kernels, %d of %d kernel->kernel edges are programmatic (port %d)\n", tag, st.kernels, st.converted, st.kernel_edges, port_mode());
                }
                return cudaSuccess;
            }
            (void)cudaGetLastError();
            cudaGraphDestroy(c);
        } else {
            (void)cudaGetLastError();
        }
    }
#endif
    (void)tag;
    return cudaGraphInstantiate(exec, g, nullptr, nullptr, 0);
}

}  // namespace strata::ink::pdl
