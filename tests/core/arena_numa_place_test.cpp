// tests/core/arena_numa_place_test.cpp - STRATA_ARENA_NUMA_PLACE: the expert arena's per-node page ranges
// (detail::arena_numa_slice_ranges / arena_numa_gaps / arena_numa_share), the gate that predicts the pool's split
// (detail::arena_numa_place_gate) and, on a Linux host with two NUMA nodes, the page step itself on a small anonymous
// mapping (detail::arena_numa_populate, checked with move_pages).  No GPU, no CUDA call, no model.
#include "strata/core/expert_source.hpp"

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>
#include <system_error>
#include <utility>
#include <vector>

#if defined(__linux__)
#include <sys/mman.h>
#include <sys/syscall.h>
#include <unistd.h>
#include <filesystem>
#include <fstream>
#endif

namespace {

using Ranges = std::vector<std::pair<uint64_t, uint64_t>>;
constexpr uint64_t P = 4096;

void require(bool ok, const std::string& message) {
    if (!ok) throw std::runtime_error(message);
}

struct Layout {
    std::vector<uint64_t> off, stride, roles;   // per layer; roles 3 per layer
    int64_t n_expert = 0;
    uint64_t limit = 0;                         // bytes of the arena the blobs use
};

// a packed layout (experts.bin: blob after blob, layer after layer) or pin by need's padded one (stride on a page)
Layout make_layout(const std::vector<std::array<uint64_t, 3>>& layer_roles, const std::vector<uint64_t>& tails, int64_t n_expert,
                   bool padded) {
    Layout l;
    l.n_expert = n_expert;
    uint64_t at = 0;
    for (size_t i = 0; i < layer_roles.size(); ++i) {
        const uint64_t blob = layer_roles[i][0] + layer_roles[i][1] + layer_roles[i][2] + tails[i];
        const uint64_t s = padded ? (blob + P - 1) / P * P : blob;
        l.off.push_back(at);
        l.stride.push_back(s);
        for (int r = 0; r < 3; ++r) l.roles.push_back(layer_roles[i][(size_t) r]);
        at += s * (uint64_t) n_expert;
    }
    l.limit = at;
    return l;
}

// The page rule, page by page from first principles: 0 = a page holding any slice-0 byte (the first half of a role),
// 1 = a page wholly inside the second half of one role and holding no slice-0 byte, -1 = neither (tails, pads)
std::vector<int> brute_owner(const Layout& l) {
    const uint64_t pages = (l.limit + P - 1) / P;
    std::vector<int> owner((size_t) pages, -1);
    std::vector<std::pair<uint64_t, uint64_t>> first_halves, second_halves;
    for (size_t i = 0; i < l.off.size(); ++i)
        for (int64_t e = 0; e < l.n_expert; ++e) {
            uint64_t start = l.off[i] + (uint64_t) e * l.stride[i];
            for (int r = 0; r < 3; ++r) {
                const uint64_t b = l.roles[3 * i + (size_t) r], mid = start + b / 2, end = start + b;
                if (mid > start) first_halves.emplace_back(start, mid);
                if (end > mid) second_halves.emplace_back(mid, end);
                start = end;
            }
        }
    for (const auto& h : second_halves)
        for (uint64_t pg = (h.first + P - 1) / P; (pg + 1) * P <= h.second; ++pg) owner[(size_t) pg] = 1;
    for (const auto& h : first_halves)
        for (uint64_t pg = h.first / P; pg * P < h.second; ++pg) owner[(size_t) pg] = 0;
    return owner;
}

void check_ranges_shape(const Ranges& rs, uint64_t limit, const char* what) {
    for (size_t i = 0; i < rs.size(); ++i) {
        require(rs[i].first % P == 0 && rs[i].second % P == 0, std::string(what) + ": a range is not page aligned");
        require(rs[i].first < rs[i].second, std::string(what) + ": an empty or inverted range");
        require(rs[i].second <= (limit + P - 1) / P * P, std::string(what) + ": a range runs past the arena");
        if (i > 0) require(rs[i - 1].second < rs[i].first, std::string(what) + ": ranges not sorted or not coalesced");
    }
}

std::vector<int> owner_of(const Ranges& r0, const Ranges& r1, const Ranges& gaps, uint64_t limit) {
    const uint64_t pages = (limit + P - 1) / P;
    std::vector<int> owner((size_t) pages, -2);   // -2 = in no list (must not happen)
    auto mark = [&](const Ranges& rs, int v) {
        for (const auto& r : rs)
            for (uint64_t pg = r.first / P; pg < r.second / P; ++pg) {
                require(pg < pages, "a range page is past the arena");
                require(owner[(size_t) pg] == -2, "a page is in two lists (" + std::to_string(owner[(size_t) pg]) + " and " +
                                                      std::to_string(v) + ") at page " + std::to_string(pg));
                owner[(size_t) pg] = v;
            }
    };
    mark(r0, 0);
    mark(r1, 1);
    mark(gaps, -1);
    return owner;
}

size_t runs_of(const std::vector<int>& owner, int v) {
    size_t n = 0;
    for (size_t i = 0; i < owner.size(); ++i)
        if (owner[i] == v && (i == 0 || owner[i - 1] != v)) ++n;
    return n;
}

void check_layout(const Layout& l, const char* name) {
    using namespace strata::core::detail;
    const Ranges r0 = arena_numa_slice_ranges(l.off, l.stride, l.n_expert, l.roles, 0, l.limit);
    const Ranges r1 = arena_numa_slice_ranges(l.off, l.stride, l.n_expert, l.roles, 1, l.limit);
    const Ranges gaps = arena_numa_gaps(r0, r1, l.limit);
    check_ranges_shape(r0, l.limit, name);
    check_ranges_shape(r1, l.limit, name);
    check_ranges_shape(gaps, l.limit, name);
    const std::vector<int> want = brute_owner(l);
    const std::vector<int> got = owner_of(r0, r1, gaps, l.limit);
    for (size_t pg = 0; pg < want.size(); ++pg)
        require(got[pg] == want[pg], std::string(name) + ": page " + std::to_string(pg) + " is in list " +
                                         std::to_string(got[pg]) + ", the page rule gives " + std::to_string(want[pg]));
    // coalescing: one range per maximal run of a slice's pages
    require(r0.size() == runs_of(want, 0) && r1.size() == runs_of(want, 1) && gaps.size() == runs_of(want, -1),
            std::string(name) + ": the ranges are not the maximal runs");
    // the per-blob rule is numa_slice_ranges' own (the FileExpertSource complement's): the same pages per blob
    for (size_t i = 0; i < l.off.size(); ++i)
        for (int64_t e = 0; e < l.n_expert; ++e)
            for (int k = 0; k < 2; ++k)
                for (const auto& r : numa_slice_ranges(l.off[i] + (uint64_t) e * l.stride[i], &l.roles[3 * i], k))
                    for (uint64_t pg = r.first / P; pg < r.second / P && pg < got.size(); ++pg)
                        require(got[(size_t) pg] == k, std::string(name) + ": a numa_slice_ranges page is in another list");
}

void test_slice_ranges() {
    // UD-Q4_K_XL-like: gate / up 640 rows x 1440 B (q4_K over 2560), down 2560 rows x 480 B (q5_1 over 640): a blob of
    // exactly 750 pages, and its halves fall on 112.5 / 150 pages
    const std::array<uint64_t, 3> udq4 = {921600, 921600, 1228800};
    // the canonical 2-bit roles of numa_slice_ranges' own test (blob not a whole number of pages)
    const std::array<uint64_t, 3> canon = {870400, 870400, 921600};
    // small roles (under a page each) and an NVFP4-like 16 B tail
    const std::array<uint64_t, 3> tiny = {1000, 1000, 3000};
    const std::array<uint64_t, 3> odd = {123457, 123457, 99999};
    for (bool padded : {false, true}) {
        const char* name = padded ? "padded" : "packed";
        check_layout(make_layout({udq4, canon, udq4}, {0, 0, 16}, 5, padded), name);
        check_layout(make_layout({tiny, odd, canon}, {0, 16, 0}, 7, padded), name);
        check_layout(make_layout({odd}, {16}, 1, padded), name);
    }
    // the slice-0 halves of UD-Q4_K_XL's gate are 320 rows of 1440 B: the pool's row split (row0 = rows * k / 2) and the
    // byte split meet on that row
    require(udq4[0] / 2 == (640 * 1 / 2) * 1440ull && udq4[2] / 2 == (2560 * 1 / 2) * 480ull,
            "a role half is not the pool's row half");
    // clipped to the limit: nothing past it, and bad inputs give nothing
    const Layout l = make_layout({udq4}, {0}, 3, false);
    const Ranges cut = strata::core::detail::arena_numa_slice_ranges(l.off, l.stride, l.n_expert, l.roles, 1, l.limit / 2);
    for (const auto& r : cut) require(r.second <= l.limit / 2, "a range ran past the clip");
    require(strata::core::detail::arena_numa_slice_ranges(l.off, {}, l.n_expert, l.roles, 0, l.limit).empty(),
            "mismatched vectors gave ranges");
    require(strata::core::detail::arena_numa_slice_ranges(l.off, l.stride, l.n_expert, l.roles, 2, l.limit).empty(),
            "slice 2 gave ranges");
}

void test_share() {
    using strata::core::detail::arena_numa_share;
    Ranges rs;
    for (uint64_t i = 0; i < 100; ++i) rs.emplace_back(i * 10 * P, i * 10 * P + (1 + i % 7) * P);
    for (int parts : {1, 2, 3, 5}) {
        size_t next = 0;
        uint64_t total = 0, most = 0;
        for (const auto& r : rs) total += r.second - r.first;
        for (int j = 0; j < parts; ++j) {
            const auto sh = arena_numa_share(rs, parts, j);
            require(sh.first == next, "a share does not start where the previous one ended");
            require(sh.second > sh.first, "an empty share for 100 ranges");
            uint64_t b = 0;
            for (size_t i = sh.first; i < sh.second; ++i) b += rs[i].second - rs[i].first;
            most = std::max(most, b);
            next = sh.second;
        }
        require(next == rs.size(), "the shares do not cover every range");
        require(most <= total / (uint64_t) parts + 7 * P, "a share is far above its part of the bytes");
    }
    const Ranges two = {{0, P}, {2 * P, 3 * P}};
    size_t covered = 0;
    for (int j = 0; j < 3; ++j) {
        const auto sh = arena_numa_share(two, 3, j);
        covered += sh.second - sh.first;
    }
    require(covered == 2, "three shares of two ranges lost or doubled one");
    require(arena_numa_share({}, 3, 0).first == arena_numa_share({}, 3, 0).second, "a share of nothing was not empty");
}

void test_gate() {
    using strata::core::detail::arena_numa_place_gate;
    const std::vector<int> one = {0}, two = {0, 1}, three = {0, 1, 2}, sparse = {0, 2};
    auto g = arena_numa_place_gate(nullptr, nullptr, nullptr, nullptr, one);
    require(!g.on && g.why.empty(), "one node: the gate was not off and silent");
    g = arena_numa_place_gate(nullptr, nullptr, nullptr, nullptr, two);
    require(g.on && g.nodes[0] == 0 && g.nodes[1] == 1, "two nodes, nothing set: the gate was not on for nodes 0 and 1");
    g = arena_numa_place_gate(nullptr, nullptr, nullptr, nullptr, sparse);
    require(g.on && g.nodes[0] == 0 && g.nodes[1] == 2, "sparse node ids were not kept");
    for (const char* off : {"0", "off", "OFF", "no", "false"}) {
        g = arena_numa_place_gate(off, nullptr, nullptr, nullptr, two);
        require(!g.on && !g.why.empty(), std::string("STRATA_ARENA_NUMA_PLACE=") + off + " did not turn it off");
    }
    require(arena_numa_place_gate("1", nullptr, nullptr, nullptr, two).on, "STRATA_ARENA_NUMA_PLACE=1 turned it off");
    require(!arena_numa_place_gate("0", nullptr, nullptr, nullptr, one).on, "the switch turned it on with one node");
    require(!arena_numa_place_gate(nullptr, "0", nullptr, nullptr, two).on, "STRATA_NUMA_POOL=0 left it on");
    require(!arena_numa_place_gate(nullptr, nullptr, "test", nullptr, two).on, "STRATA_NUMA_SPLIT=test left it on");
    require(!arena_numa_place_gate(nullptr, nullptr, "0", nullptr, two).on, "STRATA_NUMA_SPLIT=0 left it on");
    require(arena_numa_place_gate(nullptr, nullptr, "1", nullptr, two).on, "STRATA_NUMA_SPLIT=1 turned it off");
    require(arena_numa_place_gate(nullptr, nullptr, "", nullptr, two).on, "an empty STRATA_NUMA_SPLIT turned it off");
    require(!arena_numa_place_gate(nullptr, "0", "1", nullptr, two).on, "STRATA_NUMA_POOL=0 did not win over SPLIT=1");
    require(!arena_numa_place_gate(nullptr, nullptr, nullptr, "1", two).on, "STRATA_EXCHANGE_ROTATE=1 left it on");
    require(arena_numa_place_gate(nullptr, nullptr, nullptr, "0", two).on, "STRATA_EXCHANGE_ROTATE=0 turned it off");
    g = arena_numa_place_gate(nullptr, nullptr, nullptr, nullptr, three);
    require(!g.on && !g.why.empty(), "three nodes: the gate was not off with a reason");
}

#if defined(__linux__)
// The page step on this machine: with two NUMA nodes with CPUs, a 96 MiB anonymous mapping (4 KiB pages) of a made-up
// arena is placed and every page's node is read back.  Skipped (not failed) on one node.
bool test_populate_on_this_host() {
    std::vector<int> ids;
    std::vector<std::vector<int>> cpus;
    std::error_code ec;
    for (int n = 0; n < 64; ++n) {
        const std::filesystem::path d = "/sys/devices/system/node/node" + std::to_string(n);
        if (!std::filesystem::exists(d, ec)) continue;
        std::ifstream f(d / "cpulist");
        std::string list;
        std::getline(f, list);
        if (list.find_first_of("0123456789") == std::string::npos) continue;
        ids.push_back(n);
        cpus.emplace_back();   // affinity is only for speed: none here
    }
    if (ids.size() != 2) {
        std::printf("arena_numa_place_test: %zu NUMA nodes with CPUs, the page step check needs 2: skipped\n", ids.size());
        return false;
    }
    const std::array<uint64_t, 3> udq4 = {921600, 921600, 1228800};
    const Layout l = make_layout({udq4, udq4}, {0, 16}, 16, false);
    const uint64_t bytes = l.limit + 3 * 1024 * 1024;   // a spare blob at the end, as the arena has
    void* m = mmap(nullptr, (size_t) bytes, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    require(m != MAP_FAILED, "mmap of the test arena failed");
    (void) madvise(m, (size_t) bytes, MADV_NOHUGEPAGE);
    using namespace strata::core::detail;
    const Ranges ranges[2] = {arena_numa_slice_ranges(l.off, l.stride, l.n_expert, l.roles, 0, l.limit),
                              arena_numa_slice_ranges(l.off, l.stride, l.n_expert, l.roles, 1, l.limit)};
    const int nodes[2] = {ids[0], ids[1]};
    const std::vector<int> cpu_lists[2] = {cpus[0], cpus[1]};
    const ArenaPopulateStats st = arena_numa_populate((uint8_t*) m, bytes, nodes, cpu_lists, ranges, 3);
    require(st.policy_failed == 0, "set_mempolicy failed (errno " + std::to_string(st.policy_errno) + ")");
    const uint64_t pages = (bytes + P - 1) / P;
    std::vector<void*> addr((size_t) pages);
    std::vector<int> status((size_t) pages, -1);
    for (uint64_t i = 0; i < pages; ++i) addr[(size_t) i] = (uint8_t*) m + i * P;
    require(syscall(SYS_move_pages, 0, (unsigned long) pages, addr.data(), nullptr, status.data(), 0) == 0,
            "move_pages failed");
    const std::vector<int> want = brute_owner(l);
    uint64_t right[2] = {0, 0}, total[2] = {0, 0}, absent = 0;
    for (uint64_t i = 0; i < pages; ++i) {
        if (status[(size_t) i] < 0) ++absent;   // every page must exist after the step (gaps included)
        const int k = i < want.size() ? want[(size_t) i] : -1;
        if (k < 0) continue;
        ++total[k];
        if (status[(size_t) i] == nodes[k]) ++right[k];
    }
    std::printf("arena_numa_place_test: page step on nodes %d / %d: slice 0 %llu of %llu pages on node %d, slice 1 %llu of "
                "%llu on node %d, %llu pages not present, %d ranges touched page by page\n",
                nodes[0], nodes[1], (unsigned long long) right[0], (unsigned long long) total[0], nodes[0],
                (unsigned long long) right[1], (unsigned long long) total[1], nodes[1], (unsigned long long) absent, st.touched);
    require(absent == 0, "pages were left unpopulated");
    // MPOL_PREFERRED may spill when a node is full: judged at the report's 95 %
    for (int k = 0; k < 2; ++k)
        require(right[k] * 100 >= total[k] * 95, "slice " + std::to_string(k) + " is under 95 % on its node");
    munmap(m, (size_t) bytes);
    return true;
}
#endif

}  // namespace

int main() {
    try {
        test_slice_ranges();
        test_share();
        test_gate();
#if defined(__linux__)
        test_populate_on_this_host();
#endif
    } catch (const std::exception& e) {
        std::cerr << "arena_numa_place_test: FAILED: " << e.what() << "\n";
        return 1;
    }
    std::printf("arena_numa_place_test: passed\n");
    return 0;
}
