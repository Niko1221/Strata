#include "strata/core/exchange_storage.hpp"
#include <algorithm>
#include <cstring>
#include <iostream>
#include <random>
#include <stdexcept>
#include <thread>
#include <vector>

using strata::core::detail::ExchangeStorage;
static void require(bool ok, const char* message) { if (!ok) throw std::runtime_error(message); }

static void exercise(size_t bytes, size_t keep, size_t rounds, bool seeded = false) {
    constexpr size_t n = 24, ram = 16, batch_size = 4, guard = 19;
    std::vector<uint8_t> arena(guard + ram * bytes + guard, 0xcd);
    std::vector<uint8_t> spare(guard + (batch_size + keep) * bytes + guard, 0xef);
    std::vector<uint8_t> aliases(ram * bytes), spare_aliases((batch_size + keep) * bytes);
    std::vector<std::vector<uint8_t>> oracle(n, std::vector<uint8_t>(bytes));
    std::mt19937 random(7391);
    for (auto& row : oracle) for (auto& value : row) value = (uint8_t)random();
    std::vector<uint64_t> offsets(n, ~uint64_t{0});
    std::vector<bool> gpu(n, true);
    for (size_t e = 0; e < ram; ++e) {
        offsets[e] = e * bytes; gpu[e] = false;
        std::memcpy(arena.data() + guard + e * bytes, oracle[e].data(), bytes);
    }
    ExchangeStorage s;
    std::string error;
    require(s.initialize(offsets, arena.data() + guard, aliases.data(), ram * bytes,
                         spare.data() + guard, spare_aliases.data(), batch_size, bytes, error, keep), error.c_str());
    require(s.consistent(), "invalid initialization");
    std::vector<size_t> selected;
    if (seeded && keep) {
        for (size_t q = 0; q < std::min(keep, n-ram); ++q) {
            const size_t e = n-1-q;
            auto view = s.seed_buffer(q);
            require(view.host && view.device && !s.resident(e).host, "seed visible before full fill");
            std::memcpy(view.host, oracle[e].data(), bytes);
            selected.push_back(e);
        }
        require(!s.publish_seeded({n}) && s.consistent(), "out of range seed accepted");
        require(!s.publish_seeded({0}) && s.consistent(), "mandatory RAM expert seeded twice");
        if (keep > 1)
            require(!s.publish_seeded({selected[0],selected[0]}) && s.consistent(), "duplicate seed accepted");
        require(s.publish_seeded(selected) && s.consistent(), "valid seed rejected");
        require(!s.publish_seeded(selected) && !s.seed_buffer(0).host, "seed published twice");
    }
    std::atomic<bool> stop{false}, bad_view{false};
    std::atomic<size_t> observations{0};
    std::thread reader([&] {
        while (!stop.load()) {
            for (size_t e = 0; e < n; ++e) {
                const auto v = s.resident(e);
                if (!v.host) continue;
                bool found = false;
                for (size_t q = 0; q < ram; ++q)
                    found |= v.host == arena.data() + guard + q * bytes && v.device == aliases.data() + q * bytes;
                for (size_t q = 0; q < batch_size + keep; ++q)
                    found |= v.host == spare.data() + guard + q * bytes && v.device == spare_aliases.data() + q * bytes;
                if (!found) bad_view.store(true);
            }
            ++observations;
        }
    });
    struct Join {
        std::atomic<bool>& stop; std::thread& reader;
        void finish() { stop.store(true); if (reader.joinable()) reader.join(); }
        ~Join() { finish(); }
    } join{stop, reader};
    while (!observations.load()) std::this_thread::yield();
    uint64_t saved = 0;
    for (size_t round = 0; round < rounds; ++round) {
        std::vector<size_t> incoming, outgoing;
        for (size_t e = 0; e < n; ++e) (gpu[e] ? outgoing : incoming).push_back(e);
        std::shuffle(incoming.begin(), incoming.end(), random);
        std::shuffle(outgoing.begin(), outgoing.end(), random);
        std::vector<ExchangeStorage::Change> changes;
        for (size_t q = 0; q < batch_size; ++q) {
            auto input = s.resident(incoming[q]);
            require(input.host && !std::memcmp(input.host, oracle[incoming[q]].data(), bytes), "bad H2D source");
            auto old = s.resident(outgoing[q]);
            uint8_t* staged = old.host ? old.host : s.spare(q).host;
            if (old.host) {
                require(!std::memcmp(old.host, oracle[outgoing[q]].data(), bytes), "retained source differs");
                saved += bytes;
            } else {
                // Simulated D2H has completed; H2D and readers finish before commit.
                std::memcpy(staged, oracle[outgoing[q]].data(), bytes);
            }
            changes.push_back({incoming[q], outgoing[q], q, staged, bytes});
        }
        if (keep) {
            auto invalid = changes;
            invalid.back().out = invalid.front().out;
            require(!s.commit_retained(invalid) && s.consistent(), "duplicate batch mutated storage");
            invalid = changes; invalid.back().bytes++;
            require(!s.commit_retained(invalid) && s.consistent(), "bad geometry mutated storage");
            invalid = changes; invalid.back().staged = nullptr;
            require(!s.commit_retained(invalid) && s.consistent(), "missing source accepted");
            invalid = changes; invalid.back().in = n;
            require(!s.commit_retained(invalid) && s.consistent(), "bad expert accepted");
            require(s.commit_retained(changes), "valid retained batch rejected");
            require(!s.commit_retained(changes), "batch committed twice");
        } else {
            for (const auto& c : changes) require(s.commit(c.in, c.out, c.q, c.staged, c.bytes), "legacy commit failed");
        }
        for (size_t q = 0; q < batch_size; ++q) { gpu[incoming[q]] = true; gpu[outgoing[q]] = false; }
        require(s.consistent(), "buffer ownership not bijective");
        for (size_t e = 0; e < n; ++e) {
            auto view = s.resident(e);
            require(gpu[e] || view.host, "lost the sole RAM copy of a nonresident expert");
            if (seeded && gpu[e] && view.host)
                require(std::find(selected.begin(),selected.end(),e) != selected.end(), "retained an unselected GPU expert");
            if (seeded && std::find(selected.begin(),selected.end(),e) != selected.end())
                require(view.host, "lost a fixed seed's canonical bytes");
            if (!view.host) continue;
            require(!std::memcmp(view.host, oracle[e].data(), bytes), "expert bytes changed");
            bool alias = false;
            for (size_t q = 0; q < ram; ++q)
                alias |= view.host == arena.data() + guard + q * bytes && view.device == aliases.data() + q * bytes;
            for (size_t q = 0; q < batch_size + keep; ++q)
                alias |= view.host == spare.data() + guard + q * bytes && view.device == spare_aliases.data() + q * bytes;
            require(alias, "host pointer and CUDA alias separated");
        }
        require(s.avoided_d2h_bytes() == saved, "wrong saved-copy accounting");
    }
    require(s.exchanges() == rounds * batch_size, "lost exchanges");
    require(s.avoided_bytes() == rounds * batch_size * bytes, "wrong host-copy accounting");
    auto untouched = [&](const std::vector<uint8_t>& buffer, uint8_t value) {
        return std::all_of(buffer.begin(), buffer.begin() + guard, [&](auto x) { return x == value; }) &&
               std::all_of(buffer.end() - guard, buffer.end(), [&](auto x) { return x == value; });
    };
    require(untouched(arena, 0xcd) && untouched(spare, 0xef), "buffer guard overwritten");
    join.finish();
    require(!bad_view.load(), "concurrent observer saw a mismatched pointer/alias");
    s.clear();
    require(!s.active() && !s.retained_capacity() && !s.retained_count() && !s.avoided_d2h_bytes(), "clear retained state");
}

int main() {
    try {
        for (size_t bytes : {size_t(1), size_t(33), size_t(4097)})
            for (size_t keep : {size_t(0), size_t(1), size_t(2), size_t(4), size_t(8)})
                exercise(bytes, keep, 512);
        for (size_t keep : {size_t(0), size_t(3), size_t(8)}) exercise(5222400, keep, 4);
        for (size_t bytes : {size_t(1),size_t(33),size_t(4097)})
            for (size_t keep : {size_t(1),size_t(2),size_t(4),size_t(8)})
                exercise(bytes,keep,512,true);
        for (size_t keep : {size_t(1),size_t(3),size_t(8)}) exercise(5222400,keep,4,true);
        std::cout << "retained_exchange_storage_test: PASS (55392 exchanges, default/FIFO/fixed seeds, guards, aliases, ownership, immutable bytes)\n";
    } catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
}
