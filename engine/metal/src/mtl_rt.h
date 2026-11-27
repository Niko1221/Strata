// metal/src/mtl_rt.h - a thin C++ face over Metal for the standalone kernel programs (parity, benches).
//
// The implementation (mtl_rt.mm) is the only Objective-C++ file; every program and test is plain C++ against this
// header.  Kernels are MSL source files compiled at run time (newLibraryWithSource): the Command Line Tools have no
// offline `metal` compiler, and the runtime compiler is part of the OS.  The quant header (include/strata/quant/strata-quant-common.h) is
// prepended for the i-quant grids and block structs.
#pragma once

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

namespace mtl {

struct Buffer {
    void* handle = nullptr;   // id<MTLBuffer> (retained); the first chunk of a chunked buffer
    void* ptr = nullptr;      // CPU address (shared storage)
    size_t bytes = 0;
    // a buffer longer than maxBufferLength: overlapping chunks every `stride` bytes (see Context::wrap)
    std::shared_ptr<std::vector<void*>> chunks;
    size_t stride = 0;
    size_t bias = 0;          // bytes from the MTLBuffer's start to ptr (a view onto an allocation the runtime owns)
    // or disjoint regions (Context::wrap_ranges): chunk k holds the bytes from starts[k]; the gaps are not wrapped
    std::shared_ptr<std::vector<size_t>> starts;
    template <class T> T* as() const { return static_cast<T*>(ptr); }
};

struct Pipeline {
    void* handle = nullptr;   // id<MTLComputePipelineState>
    int max_threads = 0;      // maxTotalThreadsPerThreadgroup
    int simd_width = 0;       // threadExecutionWidth
    int static_tg_mem = 0;    // staticThreadgroupMemoryLength
    std::string name;         // the function (+ its constants), for messages
};

// A function-constant value set (index -> int / bool / uint).
struct Constants {
    struct V { int index; int type; uint32_t v; };  // type 0 int, 1 bool, 2 uint, 3 short, 4 ushort
    std::vector<V> vals;
    Constants& i(int idx, int v) { vals.push_back({idx, 0, (uint32_t)v}); return *this; }
    Constants& b(int idx, bool v) { vals.push_back({idx, 1, (uint32_t)v}); return *this; }
    Constants& u(int idx, uint32_t v) { vals.push_back({idx, 2, v}); return *this; }
};

struct Arg {           // one argument of a dispatch: a buffer (+ offset) or inline bytes
    const Buffer* buf = nullptr;
    size_t off = 0;
    std::vector<uint8_t> bytes;
    static Arg B(const Buffer& b, size_t off = 0) { Arg a; a.buf = &b; a.off = off; return a; }
    template <class T> static Arg V(const T& v) {
        Arg a; a.bytes.resize(sizeof(T)); std::memcpy(a.bytes.data(), &v, sizeof(T)); return a;
    }
};

struct Dispatch {
    const Pipeline* p = nullptr;
    std::vector<Arg> args;
    uint32_t grid[3] = {1, 1, 1};     // threadgroups
    uint32_t tg[3] = {32, 1, 1};      // threads per threadgroup
    uint32_t tg_mem = 0;              // dynamic threadgroup memory at index 0 (bytes)
    const Buffer* indirect = nullptr; // grid from a GPU buffer (3 x uint32 threadgroup counts) at ind_off
    size_t ind_off = 0;
    bool barrier = true;              // run_graph: a buffer memory barrier before this dispatch
};

// A pipeline whose maxTotalThreadsPerThreadgroup is below the threadgroup size a dispatch asks for is NOT executed by
// Metal and raises no error: the dispatch simply does not run, its outputs silently keep whatever was in them, and
// every answer that reads them is garbage.  So every dispatch is checked: too few threads is fatal, with a message
// that names the kernel, the limit and the size needed.
// STRATA_METAL_MAX_THREADS_CAP=N lowers every pipeline's reported limit to N (a test hook: it proves this check).
void require_fit(const Pipeline& p, uint32_t threads, const char* where);
inline void require_fit(const Dispatch& d, const char* where = "dispatch") {
    require_fit(*d.p, d.tg[0] * d.tg[1] * d.tg[2], where);
}

class Context {
public:
    Context();
    ~Context();
    // not copyable: it owns raw pointers (impl_, fill_) and the filler's thread
    Context(const Context&) = delete;
    Context& operator=(const Context&) = delete;
    std::string device_name() const;
    bool has_family_apple(int n) const;          // MTLGPUFamilyApple<n>
    bool supports_metal4() const;
    uint64_t recommended_working_set() const;
    uint64_t max_buffer_length() const;
    // Zero-copy weights: the GPU address of host memory [p, p + bytes) (file-backed pages of the model, RAM slots) through a
    // no-copy buffer over the pages it spans, made resident.  0: not mappable (the cap on buffers is reached, or the pages are not
    // ours).  The first dispatch that reads cold pages waits for them (the buffer is wired on use).
    uint64_t zc_map(const void* p, size_t bytes);
    Pipeline pipeline_uncached(int lib, const std::string& fn, const Constants& c);
    size_t zc_count() const;
    uint64_t allocated() const { return allocated_; }   // bytes of alloc() + wrap() buffers so far

    // Pins this context to safe math: STRATA_METAL_MATH=fast then ends the process instead of silently changing the
    // reduction order, and so does any of the switches deterministic_env_violation() names.  Off by default.  hy4-flash
    // turns it on at start-up because its kernels must be bitwise the same for every window size (T invariance).
    void lock_safe_math();
    bool safe_math_locked() const { return safe_math_locked_; }

    // Compiles a library from MSL source files (paths), prepended by `prelude` (defines) and, if include_ggml,
    // ggml-common.h with GGML_COMMON_DECL_METAL/IMPL_METAL.  Returns a library id (index); aborts with the
    // compiler log on error.
    // Safe math unless STRATA_METAL_MATH=fast (the argument is ignored; kept for callers), or safe math is locked.
    int compile(const std::vector<std::string>& files, const std::string& prelude = "", bool fast_math = false);
    // compile() that returns -1 (the compiler's message in *why) on a failure instead of ending the process
    int compile_try(const std::vector<std::string>& files, const std::string& prelude, bool fast_math, std::string* why);
    Pipeline pipeline(int lib, const std::string& fn, const Constants& c = {});

    Buffer alloc(size_t bytes);                  // shared storage, zeroed
    Buffer wrap(void* page_aligned, size_t bytes);  // no-copy wrap (page-aligned, size multiple of page)
    // the same, also when bytes > maxBufferLength: then overlapping chunks, each holding any range of up to
    // max_span bytes that starts in it (max_span = the largest tensor bound at one offset)
    Buffer wrap(void* page_aligned, size_t bytes, size_t max_span);
    // only the given [begin, end) byte ranges (page-aligned, ascending, each up to maxBufferLength), bound by their
    // offsets from the start of p: what lies between them is never wrapped, so it stays out of the GPU's working set
    Buffer wrap_ranges(void* page_aligned, const std::vector<std::pair<size_t, size_t>>& ranges);
    void free(Buffer& b);

    // Encodes `ds` into ONE compute encoder (serial or concurrent dispatch type) and ONE command buffer, `reps`
    // times over; commits and waits.  Returns the GPU time (GPUEndTime - GPUStartTime) in seconds.
    void* mark();                 // split_runs() that returns a token for the buffer just committed (nullptr: nothing was queued)
    void wait_mark(void* token);  // block until that buffer has completed, then drop the token
    void drop_mark(void* token);  // drop a token without waiting (an event recorded again, or destroyed, before it was waited on)
    void split_runs();   // commit the batched dispatches as their own command buffer, without waiting
    void flush_runs();   // commit and wait for the dispatches batched since the last flush
    double run(const std::vector<Dispatch>& ds, int reps = 1, bool concurrent = false, bool barrier_between = false);
    // A dependency graph in ONE concurrent encoder: a memory barrier only before dispatches with `barrier` set, so
    // the dispatches between two barriers may overlap.  Returns the GPU time in seconds.
    double run_graph(const std::vector<Dispatch>& ds, size_t count = (size_t) -1);   // count: only the first `count` dispatches (the step profile's prefixes)
    // run_graph on a SECOND queue, committed without waiting: the GPU may overlap it with work on the main queue (the two must touch
    // different buffers).  Returns a token; wait_async(token) blocks until it has completed (returns its GPU time in seconds).
    void* run_graph_async(const std::vector<Dispatch>& ds);
    double wait_async(void* token);
    // run_graph in segments: the dispatches [cuts[s-1], cuts[s]) are one concurrent encoder; after segment s (for
    // s < cuts.size()) the GPU signals a shared event and waits until the host has run at_cut(s) - the host's turn
    // between two parts of one committed command buffer (the SSD-streaming handshake: the routing is read, the
    // missing experts are put in place).  Returns the GPU time in seconds.
    double run_graph_seg(const std::vector<Dispatch>& ds, const std::vector<int>& cuts, void (*at_cut)(void*, int),
                         void* user);
    // The same, with the GPU's WAIT for the host placed later than its SIGNAL: after segment s the GPU signals, keeps
    // computing the dispatches [cuts[s], waits[s]) - work that does not depend on what at_cut(s) writes - and only
    // then waits for the host.  So the host's turn overlaps that work instead of the GPU idling through it.
    // cuts[s] <= waits[s] <= cuts[s + 1] (ds.size() for the last); waits empty, or waits[s] == cuts[s] for every s,
    // encodes exactly what the overload above does.  The dispatches in [cuts[s], waits[s]) must neither read nor
    // write anything at_cut(s) touches.
    double run_graph_seg(const std::vector<Dispatch>& ds, const std::vector<int>& cuts, const std::vector<int>& waits,
                         void (*at_cut)(void*, int), void* user);

    // WHERE run_graph_seg's WALL TIME GOES, which `gpu` alone cannot say.  The value it returns is
    // `GPUEndTime - GPUStartTime` of the one command buffer, and that span CONTAINS the host's turns: at every
    // cut the GPU signals a shared event and stalls until the host answers, so the returned span is not the time
    // the GPU computed - part of it is the GPU waiting for us.  Three pieces are separated:
    //
    //   encode_ms  building the encoders and the dispatches, before commit (pure host, outside the GPU span)
    //   spin_ms    the host's busy-wait for the GPU to reach each cut - time the GPU was still computing, and
    //              NOT the same as the at_cut body that `host_cut_ms` measures
    //   cut_ms     the at_cut bodies themselves (the expert lookup and the reads it starts)
    //   commit_ms  [cb commit]
    //   wait_ms    waitUntilCompleted after the last segment
    //
    // Accumulated over every call until `reset_seg_timing()`; `n_cb` / `n_enc` / `n_cut` count the command
    // buffers, the encoders and the cuts, so a per-encoder or per-cut fixed cost can be divided out.
    // `n_dis` counts the dispatches actually issued, `n_dis_plain` the ones that went through run_graph /
    // run_graph_async instead of the segmented path - work that no segment timing would ever see.
    // PER SEGMENT, which is what makes the busy-wait meaningful instead of a total.  The GPU signals at the
    // end of segment s and we spin until it does, so the spin at cut s is segment s's own GPU time - and in
    // hy4-flash one segment is one block, so these are a block's times inside the real pipeline rather than those
    // of a bench that runs the block's kernels alone.  kMaxCuts is a cap, not a geometry: hy4-flash decode uses 41.
    static constexpr int kMaxCuts = 64;
    struct SegTiming {
        double encode_ms = 0, spin_ms = 0, cut_ms = 0, commit_ms = 0, wait_ms = 0, gpu_ms = 0;
        uint64_t n_cb = 0, n_enc = 0, n_cut = 0, n_dis = 0, n_dis_plain = 0;
        double spin_at[kMaxCuts] = {};   ///< spin before cut s, summed over the calls
        double cut_at[kMaxCuts] = {};    ///< the at_cut body at cut s, summed over the calls
        uint64_t n_at[kMaxCuts] = {};    ///< how many calls reached cut s (the divisor for the two above)
    };
    const SegTiming& seg_timing() const { return seg_tm_; }
    void reset_seg_timing() { seg_tm_ = SegTiming(); reset_fill_stats(); }
    // Puts the accumulators back to `t` (a caller whose command buffer is not one of the windows being profiled -
    // hy4-flash's MTP draft step - takes seg_timing() before its run_graph_seg and restores it after).
    void restore_seg_timing(const SegTiming& t) { seg_tm_ = t; }

    // THE GPU FILLER.  Decode on hy4-flash cuts each window into 41 host turns and several command buffers, and the GPU's
    // clock governor reads the gaps as idleness and drops the P-state, so the real kernels then run at a lower clock
    // than the same kernels run back to back.  The filler keeps the GPU from looking idle while the host is busy: a
    // stream of short (~250 us) buffers of ONE threadgroup running an ALU-only loop on its own queue.  It loads
    // nothing but its 4-byte step count and writes only a word nothing reads, so it cannot change a value and takes
    // no bandwidth from the real work.  It shares the GPU for timing only: a buffer started just before the host's
    // turn ends runs on for up to one buffer length (one threadgroup) next to the released real work.
    //   set_seg_fill(1)  `cut`: filler only while the host is inside a run_graph_seg cut (holds one CPU core while
    //                    armed - the caller arms it for decode and disarms it after)
    //   set_seg_fill(2)  `always`: filler while segmented runs happened in the last 50 ms
    //   set_seg_fill(0)  disarm (the thread parks on a condition variable)
    // Inert unless called: no thread, queue or compile.  STRATA_METAL_SEG_FILL=off|cut|always overrides the mode the
    // caller asks for (off disables it outright), so it can be switched without a rebuild.  Returns "" or why the
    // filler could not be built.  ~Context stops and joins the thread.
    std::string set_seg_fill(int mode);
    int seg_fill_mode() const;               ///< 0 / 1 / 2, as armed now
    double seg_fill_buffer_us() const;       ///< measured GPU time of one filler buffer, 0 before the first arm
    struct FillStats { uint64_t n = 0; double gpu_ms = 0; };
    FillStats fill_stats() const;
    void reset_fill_stats();
    // `spin_at` / `cut_at` are indexed by the cut's position WITHIN THE CALL by default, and one hy4-flash window is
    // five calls (an Engram block forces a flush), so index 0 would mix five different blocks.  Given the
    // window's cut count here, they are indexed by the cut's position within the WINDOW instead - which for
    // hy4-flash decode is the ordinal, the cuts being emitted in ordinal order.  The caller must check what it
    // assumed: `n_cut / positions` has to equal the period, or the two are not aligned and the profile is
    // meaningless.  0 = off (index within the call).
    void set_cut_period(int p) { cut_period_ = p > 0 && p <= kMaxCuts ? p : 0; }
    int cut_period() const { return cut_period_; }
    // One command buffer per `run_cb` call, empty or not: measures commit+wait round trips.
    double run_wall(const std::vector<Dispatch>& ds, int reps, double* gpu_seconds = nullptr);

    // Indirect command buffer: the dispatches recorded once, executed `reps` times in one command buffer.
    // Buffers are bound per command (setKernelBuffer); inline bytes are not allowed (use buffers).
    double run_icb(const std::vector<Dispatch>& ds, int reps, bool barrier_between);
    // Runs `ds` repeatedly for about `seconds` of wall time: lifts the GPU out of its idle clocks before a timing.
    void warm(const std::vector<Dispatch>& ds, double seconds = 0.3);

    std::string ggml_common_path;                // where strata-quant-common.h (formerly ggml-common.h) is read from
    std::string kernel_dir;                      // where relative kernel paths resolve

private:
    struct Impl;
    Impl* impl_;
    struct Filler;
    Filler* fill_ = nullptr;
    uint64_t allocated_ = 0;
    SegTiming seg_tm_;
    int cut_period_ = 0;
    bool safe_math_locked_ = false;
};

double now_s();                                  // monotonic wall clock (seconds)
std::string read_file(const std::string& path);

// The environment switches a bitwise-deterministic engine must not inherit: STRATA_METAL_MATH=fast (reassociation),
// STRATA_METAL_F16 (half activations and half accumulation), STRATA_METAL_SKIP (a timing path that returns wrong
// values) and STRATA_METAL_GEMV_NR (a tile choice that would vary with the window).  Returns the offending name
// ("STRATA_METAL_MATH=fast" for the first), or nullptr when none is set.
const char* deterministic_env_violation();

}  // namespace mtl
