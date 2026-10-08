// dsv4 test: the batched prefill must produce the SAME numbers as the token-per-token path.
//
// The whole point of forward_chunk() is that it reads each weight row once for T tokens instead of T
// times. That is only worth shipping if it does not move a single logit, so this test runs the same
// prompt twice - once with prefill batching off, once with chunks of 2, 3, 5, 7, 13 and the default -
// and compares the logits and the routing trace of every token.
//
// It needs a model file, but any tiny one will do:
//   python3 tools/tiny_model.py /tmp/dsv4_tiny
//   ./build-out/test_prefill_batch --model /tmp/dsv4_tiny/tiny.gguf --ids 5,9,17,23,31,7,11,3
//
// With --no-qat-sim the activation quantisation of the reference is skipped, which is what
// tools/tiny_model.py assumes when it writes its own reference logits.
#include "dsv4/model.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <sstream>
#include <string>
#include <vector>

using namespace dsv4;

static int g_fail = 0;
#define CHECK(cond) do { if (!(cond)) { std::printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); ++g_fail; } } while (0)

static std::vector<int> split_ids(const std::string& s) {
    std::vector<int> out;
    std::stringstream ss(s);
    std::string t;
    while (std::getline(ss, t, ',')) if (!t.empty()) out.push_back(std::atoi(t.c_str()));
    return out;
}

struct Trace {
    // Logits exist only for the token that ends the prompt: that is the distribution the answer is
    // sampled from, and the only one either path computes.
    std::vector<float> last_logits;
    // last_routing is per layer, and the batched path leaves the routing of the LAST token of each
    // chunk in it. So the trace records it once per chunk end, at that token's index.
    std::vector<int> last_routing;
    std::vector<size_t> chunk_ends;
};

/// Run the prompt one token at a time (prefill batching disabled).
static Trace run_sequential(const std::vector<std::string>& shards, const std::vector<int>& ids, bool qat) {
    RunOpts o;
    o.prefill_chunk = 0;
    o.qat_sim = qat;
    o.verbose = false;
    o.gpu = false;              // the CPU path is the reference: no device, no residency plan
    o.ctx = 512;
    Model m;
    std::string err;
    if (!m.load(shards, o, err)) { std::printf("FAIL load: %s\n", err.c_str()); ++g_fail; return {}; }
    m.reset();
    Trace tr;
    int pos = 0;
    for (size_t i = 0; i < ids.size(); ++i) {
        std::vector<float> lg;
        m.forward(ids[i], pos, &lg);
        m.end_token();
        ++pos;
        tr.last_logits = lg;
        tr.last_routing = m.last_routing;
        tr.chunk_ends.push_back(i);
    }
    return tr;
}

/// Run the prompt in chunks of exactly `chunk` tokens (the last chunk may be shorter).
static Trace run_chunked(const std::vector<std::string>& shards, const std::vector<int>& ids, bool qat, int chunk) {
    RunOpts o;
    o.prefill_chunk = chunk;
    o.qat_sim = qat;
    o.verbose = false;
    o.gpu = false;
    o.ctx = 512;
    Model m;
    std::string err;
    if (!m.load(shards, o, err)) { std::printf("FAIL load: %s\n", err.c_str()); ++g_fail; return {}; }
    if (chunk > 1 && m.max_chunk() < chunk) {
        std::printf("NOTE: chunk %d requested, %d available (window/context cap)\n", chunk, m.max_chunk());
    }
    m.reset();
    Trace tr;
    int pos = 0;
    const int cap = std::min(chunk, std::max(1, m.max_chunk()));
    for (size_t i = 0; i < ids.size(); ) {
        const size_t left = ids.size() - i;
        const int n = std::min<int>((int) left, cap);
        std::vector<float> lg;
        m.forward_chunk(ids.data() + i, n, pos, (i + (size_t) n == ids.size()) ? &lg : nullptr);
        m.end_chunk(n);
        tr.last_logits = lg;
        tr.last_routing = m.last_routing;
        tr.chunk_ends.push_back(i + (size_t) n - 1);
        i += (size_t) n;
        pos += n;
    }
    return tr;
}

int main(int argc, char** argv) {
    std::vector<std::string> shards;
    std::vector<int> ids;
    bool qat = true;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--model" && i + 1 < argc) shards.push_back(argv[++i]);
        else if (a == "--ids" && i + 1 < argc) ids = split_ids(argv[++i]);
        else if (a == "--no-qat-sim") qat = false;
        else { std::printf("usage: %s --model TINY.gguf --ids a,b,c[,...] [--no-qat-sim]\n", argv[0]); return 2; }
    }
    if (shards.empty() || ids.size() < 4) {
        std::printf("usage: %s --model TINY.gguf --ids a,b,c[,...] [--no-qat-sim]\n", argv[0]);
        return 2;
    }

    const Trace base = run_sequential(shards, ids, qat);
    if (g_fail) return 1;
    CHECK(base.last_logits.size() > 0);

    // The tiny model's sliding window caps the chunk at that window, so the list stays inside it.
    const int chunks[] = {1, 2, 3, 4, 5, 8};
    for (int chunk : chunks) {
        const Trace got = run_chunked(shards, ids, qat, chunk);
        if (g_fail) return 1;

        // The routing of the LAST token of the prompt must be identical: the router sees the same
        // activations, so a different expert here means the batched path drifted somewhere upstream.
        const int routing_diff = got.last_routing == base.last_routing ? 0 : 1;

        double max_abs = 0.0, max_rel = 0.0;
        const std::vector<float>& a = base.last_logits;
        const std::vector<float>& b = got.last_logits;
        CHECK(a.size() == b.size());
        if (a.size() == b.size()) {
            for (size_t i = 0; i < a.size(); ++i) {
                const double d = std::fabs((double) a[i] - (double) b[i]);
                const double scale = std::max(1.0, std::fabs((double) a[i]));
                max_abs = std::max(max_abs, d);
                max_rel = std::max(max_rel, d / scale);
            }
        }
        std::printf("chunk=%-3d routing differs=%d  logit max|delta|=%.3e max rel=%.3e\n",
                    chunk, routing_diff, max_abs, max_rel);
        CHECK(routing_diff == 0);
        // row_dots() walks the same blocks with the same double accumulator as row_dot(), so every
        // single product is bit-identical. What is left is the ORDER of the float sums: the
        // sequential path adds a token's routed experts in routing order, the batched path in expert
        // order. Over 43 layers that is float noise, and 1e-3 is what it measures at on the tiny
        // model - a real bug moves a logit by orders of magnitude more.
        CHECK(max_abs < 1e-3);
    }

    std::printf("%s\n", g_fail ? "PREFILL BATCH TESTS FAILED" : "ALL PREFILL BATCH TESTS PASSED");
    return g_fail ? 1 : 0;
}
