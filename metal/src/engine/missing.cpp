// metal/src/engine/missing.cpp - what a kernel entry point that has no Metal implementation yet does when called.
//
// Default: say which one and stop (exit code 86), so a run never continues on outputs nobody wrote.
// STRATA_METAL_STUBS=log: say it once per entry point and return (outputs untouched) - for listing which entry points a
// code path reaches, never for answers.
#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <set>
#include <string>

namespace strata::metal_engine {

void missing(const char* name) {
    static const bool log_only = [] {
        const char* v = std::getenv("STRATA_METAL_STUBS");
        return v != nullptr && std::string(v) == "log";
    }();
    if (!log_only) {
        std::fprintf(stderr, "strata metal: %s has no Metal implementation yet\n", name);
        std::fflush(stderr);
        std::_Exit(86);
    }
    static std::mutex mu;
    static std::set<std::string> seen;
    std::lock_guard<std::mutex> lk(mu);
    if (seen.insert(name).second) std::fprintf(stderr, "strata metal: STUB %s\n", name);
}

}  // namespace strata::metal_engine

// ---- entry points the shared host loop asks every request, with nothing to do on Metal yet
#include "strata/kernels/cvec.hpp"
#include "strata/kernels/fused_gr.hpp"
namespace strata::kernels {
// no control vector on Metal (--control-vector-scaled is not ported): the empty one, `loaded()` false
const Cvec& cvec() {
    static const Cvec none{};
    return none;
}
}  // namespace strata::kernels

// the hyper-connection persist read (STRATA_HC_PERSIST) is a CUDA/HIP kernel: on Metal every read takes the
// default path, so these are the same no-ops src/kernels/cuda/fused_gr.cu compiles for non-CUDA translation units.
namespace strata::kernels {
void fused_gr_persist_init() {}
bool fused_gr_read_persist(const FusedGrArgs*, int, void*) { return false; }
}  // namespace strata::kernels

// an uncaught exception or abort on the Metal build prints where it came from (macOS gives no core dump by default and
// lldb needs a debugging permission a user's session may not have)
#include <csignal>
#include <signal.h>
#include <execinfo.h>
#include <exception>
#include <sys/ucontext.h>
#include <unistd.h>
namespace {
[[noreturn]] void metal_terminate() {
    void* fr[64];
    const int n = backtrace(fr, 64);
    const char msg[] = "strata metal: terminate - the call stack:\n";
    (void) !write(2, msg, sizeof msg - 1);
    backtrace_symbols_fd(fr, n, 2);
    std::abort();
}
const bool g_terminate_set = (std::set_terminate(metal_terminate), true);
// a crash (SIGSEGV / SIGBUS) prints its call stack too: macOS keeps no report for a command-line process by default
void metal_crash(int sig, siginfo_t* si, void* uc) {
    void* fr[64];
    const int n = backtrace(fr, 64);
    char msg[640];
    const auto* ss = uc ? &((const ucontext_t*) uc)->uc_mcontext->__ss : nullptr;
    const int m = std::snprintf(msg, sizeof msg,
                                "strata metal: crashed (signal %d code %d at address %p) pc %p x0 %p x1 %p x2 %p x8 %p x19 %p x21 %p x22 %p x23 %p x24 %p x25 %p x26 %p - the call stack:\n",
                                sig, si ? si->si_code : 0, si ? si->si_addr : nullptr,
                                ss ? (void*) ss->__pc : nullptr, ss ? (void*) ss->__x[0] : nullptr,
                                ss ? (void*) ss->__x[1] : nullptr, ss ? (void*) ss->__x[2] : nullptr,
                                ss ? (void*) ss->__x[8] : nullptr, ss ? (void*) ss->__x[19] : nullptr,
                                ss ? (void*) ss->__x[21] : nullptr, ss ? (void*) ss->__x[22] : nullptr,
                                ss ? (void*) ss->__x[23] : nullptr, ss ? (void*) ss->__x[24] : nullptr,
                                ss ? (void*) ss->__x[25] : nullptr, ss ? (void*) ss->__x[26] : nullptr);
    (void) !write(2, msg, (size_t) m);
    backtrace_symbols_fd(fr, n, 2);
    signal(sig, SIG_DFL);
    raise(sig);
}
bool set_crash() {
    struct sigaction sa {};
    sa.sa_sigaction = metal_crash;
    sa.sa_flags = SA_SIGINFO;
    sigaction(SIGSEGV, &sa, nullptr);
    sigaction(SIGBUS, &sa, nullptr);
    sigaction(SIGTRAP, &sa, nullptr);   // libmalloc's integrity trips and __builtin_trap land here
    return true;
}
const bool g_crash_set = set_crash();
}  // namespace
