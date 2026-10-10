// src/net/stage_ckpt_store_test.cpp - a stage worker's store of conversation checkpoint parts
// (docs/remote-stage/CHECKPOINTS.md), walked by hand:
//
//   1. a part is found under its id with its length; an id never stored is not;
//   2. a store prunes to `keep` only through put / prune: the parts the main process no longer names go, the
//      others stay (also ids it dropped without a message, until the next keep);
//   3. admission: at most kStageCkptMax parts after pruning - one more is refused and the store is left as it was;
//      the same id again replaces its part and counts once;
//   4. clear (a disconnect) empties it: the next main process's ids start again.
#include "strata/net/stage_ckpt_store.hpp"

#include <cstdint>
#include <cstdio>
#include <vector>

using strata::net::StageCkptStore;
using strata::net::kStageCkptMax;

namespace {
int g_fail = 0;
void check(bool ok, const char* what) {
    std::printf("  %-74s %s\n", what, ok ? "ok" : "FAIL");
    if (!ok) ++g_fail;
}
StageCkptStore::Part part(int64_t L, uint8_t tag) {
    StageCkptStore::Part p;
    p.L = L;
    p.state.gdn.assign(16, tag);
    return p;
}
}  // namespace

int main() {
    std::printf("stage_ckpt_store_test\n");
    {
        StageCkptStore s;
        check(s.admits(1, {1}), "an empty store admits a part");
        s.put(1, part(2048, 1), {1});
        s.put(2, part(4096, 2), {1, 2});
        const StageCkptStore::Part* p = s.find(2);
        check(p != nullptr && p->L == 4096 && p->state.gdn.size() == 16 && p->state.gdn[0] == 2,
              "a stored part is found under its id, with its length and state");
        check(s.find(3) == nullptr, "an id never stored is missing");
        // the main process evicted 1 when it saved 3: keep = {2, 3}
        s.put(3, part(6000, 3), {2, 3});
        check(s.find(1) == nullptr && s.find(2) != nullptr && s.find(3) != nullptr && s.size() == 2,
              "a save prunes to keep: the part the main process evicted goes");
        // ids the main process dropped without a message (a non-prefix erase) stay until the next keep
        s.put(4, part(100, 4), {2, 3, 4});
        check(s.size() == 3, "ids dropped without a message stay until the next keep");
        s.prune({4});
        check(s.size() == 1 && s.find(4) != nullptr, "a reset / restore prunes to its keep");
        s.prune({});
        check(s.size() == 0, "a reset that keeps nothing empties the store");
    }
    {
        StageCkptStore s;
        std::vector<int64_t> keep;
        for (int64_t id = 1; id <= kStageCkptMax; ++id) {
            keep.push_back(id);
            if (!s.admits(id, keep)) break;
            s.put(id, part(id, (uint8_t) id), keep);
        }
        check(s.size() == (size_t) kStageCkptMax, "the store takes kStageCkptMax parts");
        std::vector<int64_t> over = keep;
        over.push_back(kStageCkptMax + 1);
        check(!s.admits(kStageCkptMax + 1, over), "one more after pruning is refused");
        check(s.size() == (size_t) kStageCkptMax && s.find(1) != nullptr && s.find(kStageCkptMax) != nullptr,
              "a refused part leaves the store as it was");
        std::vector<int64_t> rolled(keep.begin() + 1, keep.end());
        rolled.push_back(kStageCkptMax + 1);
        check(s.admits(kStageCkptMax + 1, rolled), "it fits when the keep drops one (pruning counts first)");
        check(s.admits(5, keep), "the same id again counts once (it replaces its part)");
        s.put(5, part(77, 9), keep);
        check(s.size() == (size_t) kStageCkptMax && s.find(5)->L == 77 && s.find(5)->state.gdn[0] == 9,
              "a part saved again under its id replaces the old one");
        s.clear();
        check(s.size() == 0 && s.find(5) == nullptr, "a disconnect clears the store");
        check(s.admits(1, {1}), "the next main process starts from an empty store");
    }
    {
        StageCkptStore s;
        s.put(1, part(10, 1), {1});
        s.put(2, part(20, 2), {1});   // a keep without the new id: the part reported stored stays
        check(s.find(2) != nullptr && s.find(1) != nullptr && s.size() == 2,
              "a put keeps its own id even when keep[] leaves it out");
        s.put(3, part(30, 3), {});
        check(s.find(3) != nullptr && s.size() == 1, "... and still prunes every other part keep[] leaves out");
    }
    {
        std::string err;
        const uint64_t mib = 1ull << 20;
        check(strata::net::stage_ckpt_ram_ok(7, 55 * mib, 1024 * mib, err), "7 parts of 55 MiB fit in half of 1 GiB");
        const bool refused = !strata::net::stage_ckpt_ram_ok(10, 55 * mib, 1024 * mib, err);
        check(refused && err.find("550 MiB") != std::string::npos && err.find("1024 MiB") != std::string::npos &&
                  err.find("--prompt-cache 8 or less") != std::string::npos,
              "10 do not: refused, naming the sizes and the --prompt-cache that fits");
        check(strata::net::stage_ckpt_ram_ok(0, 55 * mib, 1 * mib, err), "no checkpoints: no RAM check");
        check(strata::net::stage_ckpt_ram_ok(10, 55 * mib, std::nullopt, err), "RAM unknown: not refused");
        check(strata::net::stage_ckpt_ram_ok(kStageCkptMax + 1, 55 * mib, 1 * mib, err),
              "a ckpt_max the link refuses anyway is left to the link's message");
    }
    if (g_fail == 0) std::printf("stage_ckpt_store_test: all passed\n");
    else std::printf("stage_ckpt_store_test: %d FAILED\n", g_fail);
    return g_fail == 0 ? 0 : 1;
}
