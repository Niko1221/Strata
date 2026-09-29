// src/spec/suffix_drafter_test.cpp - plan v0.3 P6: suffix-lookup drafter tests and an offline simulation.
//
//   suffix_drafter_test                      unit tests
//   suffix_drafter_test --simulate [--k K] FILE.ids ...
//
// The simulation replays text as if a model had produced it: each step drafts up to K tokens, accepts the
// prefix that matches the true next tokens, and commits that prefix plus one model token (the verify pass's
// bonus). tokens/step is the lookup drafter's ceiling speedup on that text when a K+1-token verify costs the same
// as a 1-token step. Two regimes per file:
//   continue  history = first half of the prompt; generate its second half (natural continuation)
//   copy      history = the whole prompt; generate its middle third again (output quoting input: edits)
// No model runs; this measures the drafter on text, not on the model's own outputs.
#include "strata/spec/suffix_drafter.hpp"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

using strata::spec::SuffixDrafter;

namespace {
int g_fail = 0;
void check(bool ok, const char* what) {
    if (!ok) { std::fprintf(stderr, "FAIL: %s\n", what); ++g_fail; }
}

std::vector<int32_t> read_ids(const char* path) {
    std::ifstream f(path);
    std::string text((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    for (char& c : text) if (c == ',') c = ' ';
    std::istringstream in(text);
    std::vector<int32_t> ids;
    for (int32_t v; in >> v;) ids.push_back(v);
    return ids;
}

struct Sim { long steps = 0, tokens = 0, proposals = 0, accepted = 0; };

Sim simulate(const std::vector<int32_t>& history, const std::vector<int32_t>& target, int K) {
    SuffixDrafter d(3, 32, history.size() + target.size() + 16);
    d.append(history.data(), history.size());
    Sim s;
    std::vector<int32_t> draft(K);
    size_t i = 0;
    while (i < target.size()) {
        const int n = d.propose(K, draft.data());
        int a = 0;
        while (a < n && i + a < target.size() && draft[a] == target[i + a]) ++a;
        if (n > 0) { ++s.proposals; s.accepted += a; }
        const size_t commit = std::min(target.size() - i, (size_t) a + 1);
        d.append(&target[i], commit);
        i += commit;
        s.tokens += (long) commit;
        ++s.steps;
    }
    return s;
}

void unit_tests() {
    {   // a repeated passage is proposed verbatim
        SuffixDrafter d;
        const std::vector<int32_t> doc = {10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20};
        d.append(doc.data(), doc.size());
        const int32_t again[] = {10, 11, 12};
        d.append(again, 3);
        int32_t out[8];
        const int n = d.propose(5, out);
        check(n == 5 && out[0] == 13 && out[4] == 17, "repeat proposes the continuation");
        check(d.last_match() == 3, "match length is the shared suffix");
    }
    {   // no earlier occurrence: nothing proposed
        SuffixDrafter d;
        const std::vector<int32_t> doc = {1, 2, 3, 4, 5, 6, 7};
        d.append(doc.data(), doc.size());
        int32_t out[4];
        check(d.propose(4, out) == 0, "no match proposes nothing");
    }
    {   // two occurrences of the trigram: the longer match wins even though it is older
        SuffixDrafter d;
        const std::vector<int32_t> doc = {7, 8, 1, 2, 3, 100, 101, 9, 1, 2, 3, 200, 201, 50, 7, 8, 1, 2, 3};
        d.append(doc.data(), doc.size());
        int32_t out[2];
        const int n = d.propose(2, out);
        check(n == 2 && out[0] == 100 && out[1] == 101, "longest match preferred over most recent");
        check(d.last_match() == 5, "longest match length 5");
    }
    {   // a match shorter than min_match is ignored (only a bigram in common)
        SuffixDrafter d(4);
        const std::vector<int32_t> doc = {1, 2, 3, 9, 5, 2, 3};
        d.append(doc.data(), doc.size());
        int32_t out[2];
        check(d.propose(2, out) == 0, "min_match respected");
    }
    {   // periodic text: the continuation may run into the current suffix
        SuffixDrafter d;
        const std::vector<int32_t> doc = {1, 2, 3, 1, 2, 3, 1, 2, 3};
        d.append(doc.data(), doc.size());
        int32_t out[6];
        const int n = d.propose(6, out);
        check(n >= 3 && out[0] == 1 && out[1] == 2 && out[2] == 3, "periodic continuation");
    }
    {   // beyond the nominal capacity the table grows
        SuffixDrafter d(3, 32, 1024);
        std::vector<int32_t> doc;
        for (int i = 0; i < 5000; ++i) doc.push_back(i % 997);
        d.append(doc.data(), doc.size());
        int32_t out[4];
        check(d.propose(4, out) > 0, "propose after overflow of nominal capacity");
    }
    {   // assign: the shared prefix stays, the rest of the old text proposes nothing
        SuffixDrafter d;
        const std::vector<int32_t> a = {1, 2, 3, 4, 5, 6, 7, 8, 9, 1, 2, 3};
        d.append(a.data(), a.size());
        int32_t out[4];
        check(d.propose(4, out) > 0 && out[0] == 4, "before assign");
        const std::vector<int32_t> b = {1, 2, 3, 40, 50, 60, 70, 1, 2, 3};
        d.assign(b.data(), b.size());
        check(d.size() == b.size(), "assign replaces the history");
        const int n = d.propose(4, out);
        check(n == 4 && out[0] == 40 && out[3] == 70, "the new text's continuation, not the old one's");
        d.assign(a.data(), a.size());
        check(d.propose(4, out) == 4 && out[0] == 4 && out[3] == 7, "the old text again");
    }
    {   // many texts: the slots of undone positions are dropped when the table is rebuilt
        SuffixDrafter d(3, 32, 256);
        std::vector<int32_t> doc(200);
        for (int r = 0; r < 50; ++r) {
            for (int i = 0; i < 200; ++i) doc[(size_t) i] = r * 1000 + i % 150;
            d.assign(doc.data(), doc.size());
        }
        int32_t out[4];
        check(d.propose(4, out) == 4 && out[0] == 49000 + 50, "propose after many texts");
    }
    {   // proposals depend on the history alone: after assign undid a longer text and the table grew, as a fresh
        // drafter's (a 6-token alphabet: every trigram recurs, more often than WAYS)
        uint32_t rng = 12345;
        auto next = [&] { rng = rng * 1664525u + 1013904223u; return (int32_t) ((rng >> 16) % 6); };
        std::vector<int32_t> a(3000), b;
        for (int32_t& t : a) t = next();
        b.assign(a.begin(), a.begin() + 1700);
        for (int i = 0; i < 1300; ++i) b.push_back(next());
        SuffixDrafter x(3, 32, 16), y(3, 32, 1u << 14);
        x.append(a.data(), a.size());
        x.assign(b.data(), 1000);
        x.append(b.data() + 1000, b.size() - 1000);
        y.append(b.data(), b.size());
        bool same = x.size() == y.size();
        int32_t ox[8], oy[8];
        for (int i = 0; i < 300 && same; ++i) {
            const int nx = x.propose(8, ox), ny = y.propose(8, oy);
            same = nx == ny && x.last_match() == y.last_match() && std::equal(ox, ox + nx, oy);
            const int32_t t = next();
            x.append(t);
            y.append(t);
        }
        check(same, "proposals after assign and growth are a fresh drafter's");
    }
    std::printf("suffix_drafter unit tests: %s\n", g_fail ? "FAILED" : "OK");
}
}  // namespace

int main(int argc, char** argv) {
    if (argc > 1 && std::strcmp(argv[1], "--simulate") == 0) {
        int K = 8, first = 2;
        if (argc > 3 && std::strcmp(argv[2], "--k") == 0) { K = std::atoi(argv[3]); first = 4; }
        std::printf("%-28s %8s | %-44s | %-44s\n", "prompt", "tokens", "continue: tok/step  proposals  acc/proposal",
                    "copy: tok/step  proposals  acc/proposal");
        for (int f = first; f < argc; ++f) {
            const std::vector<int32_t> ids = read_ids(argv[f]);
            if (ids.size() < 64) continue;
            const size_t half = ids.size() / 2;
            const Sim c = simulate({ids.begin(), ids.begin() + half}, {ids.begin() + half, ids.end()}, K);
            const size_t a = ids.size() / 3, b = 2 * ids.size() / 3;
            const Sim p = simulate(ids, {ids.begin() + a, ids.begin() + b}, K);
            const char* name = std::strrchr(argv[f], '/') ? std::strrchr(argv[f], '/') + 1 : argv[f];
            std::printf("%-28s %8zu | %8.2f %10.1f%% %10.2f              | %8.2f %10.1f%% %10.2f\n", name, ids.size(),
                        (double) c.tokens / c.steps, 100.0 * c.proposals / c.steps,
                        c.proposals ? (double) c.accepted / c.proposals : 0.0, (double) p.tokens / p.steps,
                        100.0 * p.proposals / p.steps, p.proposals ? (double) p.accepted / p.proposals : 0.0);
        }
        return 0;
    }
    unit_tests();
    return g_fail ? 1 : 0;
}
