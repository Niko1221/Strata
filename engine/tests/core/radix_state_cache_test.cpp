// MIT License - Copyright (c) 2026 Strata contributors
#include "strata/core/radix_state_cache.hpp"

#include <algorithm>
#include <array>
#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace {
using strata::core::PrefixLayout;
using strata::core::PrefixRows;
using strata::core::RadixStateCache;

void check(bool ok, const char* message) {
    if (!ok) {
        std::fprintf(stderr, "%s\n", message);
        std::exit(1);
    }
}

// Two pitched attention heads, a four-token index pool, recurrent state and
// a wrapped sliding window exercise the component layouts of the native adapters.
struct Lane {
    explicit Lane(size_t pitch) : pitch(pitch), heads(2 * pitch), pooled(pitch / 4) {}
    size_t pitch;
    std::vector<uint8_t> heads, pooled;
    std::array<uint8_t, 47> ring{};
    uint32_t recurrent = 0;
    size_t captured = 0;
    void step(int pos, int token) {
        recurrent = 33 * recurrent + token;
        heads[pos] = recurrent >> 16;
        heads[pitch + pos] = recurrent >> 24;
        ring[pos % ring.size()] = recurrent;
        if ((pos + 1) % 4 == 0) pooled[pos / 4] = recurrent >> 8;
    }
    PrefixLayout kv(int begin, int end) {
        return {{heads.data() + begin, size_t(end - begin), 2, pitch},
                {pooled.data() + begin / 4, size_t(end / 4 - begin / 4)}};
    }
    PrefixLayout state(int end) {
        const size_t rows = std::min<size_t>(31, end), start = (end - rows) % ring.size();
        const size_t head = std::min(rows, ring.size() - start);
        return {{&recurrent, sizeof(recurrent)}, {ring.data() + start, head}, {ring.data(), rows - head}};
    }
    void copy(const PrefixRows& r, uint8_t* host, bool restore) {
        if (!restore) captured += r.count * r.width;
        for (size_t i = 0; i < r.count; ++i) {
            auto* dev = static_cast<uint8_t*>(r.ptr) + (r.pitch ? r.pitch : r.width) * i;
            if (restore) std::memcpy(dev, host + r.width * i, r.width);
            else std::memcpy(host + r.width * i, dev, r.width);
        }
    }
    bool same(const Lane& other, int end) const {
        for (int i = 0; i < end; ++i) {
            if (heads[i] != other.heads[i] || heads[pitch + i] != other.heads[other.pitch + i]) return false;
        }
        for (int i = std::max(0, end - 31); i < end; ++i)
            if (ring[i % ring.size()] != other.ring[i % ring.size()]) return false;
        return recurrent == other.recurrent && std::equal(pooled.begin(), pooled.begin() + end / 4, other.pooled.begin());
    }
};

void exercise(size_t budget) {
    RadixStateCache cache(budget);
    std::vector<int32_t> first(1060), branch(1060);
    for (int i = 0; i < 1060; ++i) first[i] = (i * 17) % 127;
    branch = first;
    for (int i = 512; i < 1060; ++i) branch[i] += 8;

    auto save = [&](Lane& lane, const auto& ids, int pos, auto& cursor) {
        cache.save(ids, pos, cursor, [&](int b, int e) { return lane.kv(b, e); },
            [&](int e) { return lane.state(e); },
            [&](const auto& r, auto* h, bool read) { lane.copy(r, h, read); }, [] {});
    };
    auto restore = [&](Lane& lane, const auto& cp, auto& cursor) {
        return cache.restore(cp, cursor, [&](int b, int e) { return lane.kv(b, e); },
            [&](int e) { return lane.state(e); },
            [&](const auto& r, auto* h, bool read) { lane.copy(r, h, read); }, [] {});
    };

    Lane source(1080), dest(1120), reference(1140);
    RadixStateCache::Cursor cursor;
    for (int i = 0; i < 1059; ++i) {
        source.step(i, first[i]);
        save(source, first, i + 1, cursor);
    }
    if (!budget) {
        check(source.captured == 0 && !cache.find(branch), "oversized checkpoints copied device state");
        return;
    }
    check(source.captured == 1059 * 2 + 1059 / 4 + 3 * (4 + 31),
          "append-only rows were copied more than once");
    auto* cp = cache.find(branch);
    check(cp && cp->length == 512 && restore(dest, *cp, cursor), "branch restore failed");
    for (int i = 0; i < 512; ++i) reference.step(i, branch[i]);
    check(dest.same(reference, 512), "restored pitch, pooled row, window or recurrence differs");
    for (int i = 512; i < 1059; ++i) {
        dest.step(i, branch[i]);
        reference.step(i, branch[i]);
        save(dest, branch, i + 1, cursor);
    }
    check(dest.same(reference, 1059), "continued branch differs from cold state");
    check(dest.captured == (1059 - 512) * 2 + (1059 / 4 - 512 / 4) + 2 * (4 + 31),
          "branch duplicated a shared segment");
    cp = cache.find(first);
    check(cp && cp->length == 1059 && restore(dest, *cp, cursor) && dest.same(source, 1059),
          "sibling branch overwrote immutable rows");
    cache.clear();
    check(!cache.find(first) && cursor.expired(), "clear left cached state alive");
}
}

int main() {
    exercise(1 << 20);
    exercise(0);
    return 0;
}
