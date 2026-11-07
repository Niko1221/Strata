// Actual ExpertStore, simulated GPU mapping ONLY here. No model or GPU required.
// Queries deliberately run independently of slots_allocated(): that accessor's
// mutex must not accidentally serialize the formerly racy gpu_mapped() read.
#include "ds4_store.hpp"
#include <stdexcept>

using strata::ds4::ExpertStore;
#define CHECK(x) do { if (!(x)) throw std::runtime_error(#x); } while (0)
template<class F> void until(F f) {
    const auto end = std::chrono::steady_clock::now() + std::chrono::seconds(10);
    while (!f()) {
        if (std::chrono::steady_clock::now() > end) throw std::runtime_error("worker timeout");
        std::this_thread::yield();
    }
}
int main() {
    try {
        CHECK(strata::ds4::WeightLockPolicy::enabled());
        std::atomic<bool> available{false}, release_map{false}, stop{false};
        std::atomic<unsigned> maps{0}, unmaps{0}, samples{0}, queries{0};
        ExpertStore::LayerGeo geo; geo.n_exp = 32; geo.blob = 4096;
        ExpertStore::Cfg cfg; cfg.threads = 1; cfg.floor_bytes = 0;
        cfg.budget_bytes = 24 * 20480; cfg.reserve = 0;
        cfg.ram_avail = [&] {
            const uint64_t result = available.load() ? ~uint64_t{0} : uint64_t{0};
            ++samples; return result;
        };
        cfg.map_chunk = [&](void* p, size_t) -> void* {
            const unsigned n = ++maps;
            while (!release_map.load()) std::this_thread::yield();
            // Alternate identity/nonidentity so both mapping queries inspect entries.
            return n % 2 ? p : static_cast<uint8_t*>(p) + 4096;
        };
        cfg.unmap_chunk = [&](void*) { ++unmaps; };
        auto store = std::make_unique<ExpertStore>();
        struct Unblock { std::atomic<bool>& release; ~Unblock() { release = true; } } unblock{release_map};
        CHECK(store->init(cfg, {geo}, {}, {}));
        CHECK(store->slots_allocated() == 0 && !store->gpu_mapped() && store->identity_mapped());
        std::thread reader([&] {
            while (!stop.load()) {
                (void) store->gpu_mapped();
                (void) store->identity_mapped();
                ++queries;
            }
        });
        // Always join the reader on assertions/exceptions, before destroying store.
        struct Join { std::atomic<bool>& stop; std::thread& t;
            ~Join() { stop = true; if (t.joinable()) t.join(); }
        } join{stop, reader};
        auto request = [&] {
            std::vector<char> ok; std::vector<int> slots; const int expert = 0;
            store->begin(0, &expert, 1, ok, slots);
            // No backing: begin must fall back without waiting or submitting I/O.
            CHECK(slots[0] == -1);
        };
        const unsigned sampled = samples.load();
        request(); until([&] { return samples.load() > sampled; });
        CHECK(maps == 0 && !store->gpu_mapped());
        available = true;
        // Repeated empty -> publish -> shrink cycles hit first vector reserve,
        // entry publication, removal, and regrowth while the direct queries run.
        for (unsigned cycle = 1; cycle <= 32; ++cycle) {
            release_map = false;
            until([&] { request(); return maps.load() == cycle; });
            const unsigned before = queries.load();
            until([&] { return queries.load() > before + 100; }); // mapping work outside mu_
            CHECK(!store->gpu_mapped());
            release_map = true;
            until([&] { return store->gpu_mapped(); });
            CHECK(store->identity_mapped() == (cycle % 2 != 0));
            CHECK(store->dev_blob(0) != 0);
            until([&] { return store->shrink_chunks(1) == 1; });
            CHECK(!store->gpu_mapped() && store->identity_mapped());
        }
        // Join readers before ending object lifetime; shutdown itself races only
        // with the store's owned preparation worker, which must finish and join.
        stop = true; reader.join();
        release_map = false; request(); until([&] { return maps == 33; });
        std::atomic<bool> destroying{false}, destroyed{false};
        std::thread shutdown([&] { destroying = true; store.reset(); destroyed = true; });
        until([&] { return destroying.load(); });
        CHECK(!destroyed.load());
        release_map = true; shutdown.join();
        CHECK(destroyed && maps == unmaps);
        std::printf("store publication: %u direct concurrent queries; 32 shrink/regrowth cycles and pending shutdown passed\n", queries.load());
        return 0;
    } catch (const std::exception& e) { std::fprintf(stderr, "%s\n", e.what()); return 1; }
}
