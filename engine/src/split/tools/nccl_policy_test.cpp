// The NVIDIA decision (src/hip/rccl_dl.hpp nccl_decide, what engine.cu calls with FNX_PF_NCCL): off unless asked for, so the default path never loads a
// library or makes a line; asked for, a library that opens is used with a line naming it, and a missing or wrong library leaves the host exchange with one
// plain line (a library that is not NCCL takes the loader's path that rccl_policy_test covers). Also pins that the RCCL lines (the default name) are
// unchanged. CPU only, like rccl_policy_test.
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

#include "../src/hip/rccl_dl.hpp"

static int bad = 0;
#define CHECK(c) do { if (!(c)) { std::printf("FAIL line %d: %s\n", __LINE__, #c); ++bad; } } while (0)

int main() {
    using namespace fnx;
    int calls = 0;
    auto pass = [&](std::string& why) { ++calls; why = NCCL_LIB; return true; };
    // the engine's open() loads NCCL_LIB through the same loader; here pointed at a file that is not there or is not NCCL (FNX_RCCL_LIB)
    auto load = [&](std::string& why) { ++calls; RcclApi api; return rccl_load(api, NCCL_LIB, why); };

    CHECK(!NCCL_DEFAULT_ON);
    CHECK(std::strcmp(NCCL_LIB, "libnccl.so.2") == 0);

    // FNX_PF_NCCL unset or 0: nothing tried, nothing said (the default path)
    unsetenv("FNX_RCCL_LIB");
    RcclDecision d = nccl_decide(nullptr, pass);
    CHECK(!d.use && d.line.empty() && calls == 0);
    d = nccl_decide("0", pass);
    CHECK(!d.use && d.line.empty() && calls == 0);
    d = nccl_decide("0", load);
    CHECK(!d.use && d.line.empty() && calls == 0);

    // FNX_PF_NCCL=1 and a library that opens: used, the line names NCCL and the library (no peer push is needed on NVIDIA)
    d = nccl_decide("1", pass);
    CHECK(d.use && calls == 1 && d.line == "The prompt sums use NCCL (libnccl.so.2).");

    // FNX_PF_NCCL=1 and the library missing: the host exchange, one plain line naming NCCL, the library and the loader's reason
    setenv("FNX_RCCL_LIB", "/nonexistent/libnccl.so.2", 1);
    calls = 0;
    d = nccl_decide("1", load);
    CHECK(!d.use && calls == 1);
    CHECK(d.line.rfind("NCCL is not available here (cannot load /nonexistent/libnccl.so.2", 0) == 0);
    CHECK(d.line.find("): the prompt sums use the host exchange.") != std::string::npos);
    CHECK(d.line.find('\n') == std::string::npos);
    unsetenv("FNX_RCCL_LIB");

    // the AMD decision keeps its own name and lines (the name parameter defaults to RCCL)
    calls = 0;
    auto rpass = [&](std::string& why) { ++calls; why = "librccl.so.1"; return true; };
    d = rccl_decide("1", true, false, rpass);
    CHECK(d.use && d.line == "The prompt sums use RCCL (librccl.so.1).");
    d = rccl_decide("1", false, false, rpass);
    CHECK(!d.use && d.line == "RCCL is not used for the prompt sums: the peer push between the cards is off, so the host exchange stays.");

    std::printf(bad ? "nccl_policy_test: %d FAILED\n" : "nccl_policy_test: ok\n", bad);
    return bad ? 1 : 0;
}
