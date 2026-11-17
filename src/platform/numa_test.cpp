// numa_test: the NUMA staging decisions of include/strata/platform/numa.hpp on made-up machines.
// Pure functions, no OS calls - it runs anywhere, like kernels/cpu/numa_pool_test.cpp does for the pool.
#include "strata/platform/numa.hpp"

#include <cstdio>
#include <string>
#include <vector>

namespace pf = strata::platform;

static int g_fail = 0;

static void expect(bool ok, const std::string& what) {
    if (!ok) {
        ++g_fail;
        std::fprintf(stderr, "FAIL: %s\n", what.c_str());
    }
}

int main() {
    // ---- decide_numa_gpu(env, device_node, node_count): the ring's node, or -1 = the legacy path
    auto pick = [](const char* env, int dev_node, int nodes) { return pf::decide_numa_gpu(env, dev_node, nodes); };
    for (const char* env : {(const char*) nullptr, ""}) {
        const auto p = pick(env, 1, 2);
        expect(p.node == -1 && p.log.empty(), "unset/empty: legacy path, silent");
    }
    for (const char* off : {"off", "OFF", "false", "No"}) {
        const auto p = pick(off, 1, 2);
        expect(p.node == -1 && p.log.empty(), std::string("off word ") + off + ": legacy path, silent");
    }
    {   // "0" is a node number on this switch, not an off word
        const auto p = pick("0", 1, 2);
        expect(p.node == 0 && p.log.find("forced") != std::string::npos, "\"0\": node 0 forced");
        const auto q = pick("1", 0, 2);
        expect(q.node == 1, "\"1\": node 1 forced");
        const auto r = pick("2", 0, 2);
        expect(r.node == -1 && r.log.find("does not exist") != std::string::npos, "\"2\" on 2 nodes: refused with the count");
    }
    {   // auto: needs 2+ nodes and a device node the OS could name
        const auto p = pick("auto", 1, 2);
        expect(p.node == 1 && p.log.find("auto") != std::string::npos, "auto, device on n1, 2 nodes: n1");
        expect(pick("auto", -1, 2).node == -1, "auto, device node unknown: legacy path");
        expect(pick("auto", 0, 1).node == -1, "auto, single-node machine: legacy path");
        expect(pick("auto", 0, 0).node == -1, "auto, no nodes: legacy path");
        expect(pick("AUTO", 1, 2).node == 1, "auto is case-insensitive");
        expect(pick("yes", 1, 2).node == -1 && !pick("yes", 1, 2).log.empty(), "garbage: refused, logged");
    }

    // ---- decide_numa_arena_split(env, n_layers, node_count): one domain per pipeline stage
    auto split = [](const char* env, int64_t layers, int nodes) {
        return pf::decide_numa_arena_split(env, layers, nodes);
    };
    for (const char* env : {(const char*) nullptr, "", "0", "off", "no", "FALSE"}) {
        const auto p = split(env, 62, 2);
        expect(p.bounds.empty() && p.log.empty(),
               env ? std::string("unset/off \"") + env + "\": one domain, silent" : "unset: one domain, silent");
    }
    {
        const auto p = split("31", 62, 2);
        expect(p.bounds.size() == 1 && p.bounds[0] == 31 && !p.log.empty(),
               "\"31\" on 62 layers, 2 nodes: boundary at 31");
        const auto q = split("20,40", 62, 2);
        expect(q.bounds.size() == 2 && q.bounds[0] == 20 && q.bounds[1] == 40,
               "\"20,40\": two boundaries, three domains");
        const auto one = split("31", 62, 1);
        expect(one.bounds.empty() && !one.log.empty(), "single-node machine: refused, logged");
    }
    for (const char* bad : {"x", "31x", "62", "63", "40,20", "31,31", ",31", "31,", "31,,40", "-31"}) {
        const auto p = split(bad, 62, 2);
        expect(p.bounds.empty() && !p.log.empty(),
               std::string("\"") + bad + "\": refused, logged");
    }

    if (g_fail) {
        std::fprintf(stderr, "numa_test: %d failure(s)\n", g_fail);
        return 1;
    }
    std::printf("numa_test: OK\n");
    return 0;
}
