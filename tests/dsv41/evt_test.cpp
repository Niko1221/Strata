// evt_test: the serve loop's event queue (ds4_evt.hpp, Project Maya #587).  No GPU, no model.
#include <atomic>
#include <cstdio>
#include <string>
#include <thread>
#include <vector>

#include "ds4_evt.hpp"

static int g_fail = 0;
#define CHECK(c) do { if (!(c)) { std::printf("FAIL line %d: %s\n", __LINE__, #c); ++g_fail; } } while (0)

int main() {
    namespace e = strata::ds4::evt;
    {   // (priority first, then the order they were added in)
        e::Queue q;
        for (uint64_t i = 1; i <= 20; ++i) q.push(e::Event{i, (int) (i % 3), nullptr});
        std::vector<std::pair<int, uint64_t>> got;
        e::Event ev;
        while (q.pop(ev)) got.push_back({ev.priority, ev.id});
        CHECK(got.size() == 20);
        for (size_t i = 1; i < got.size(); ++i)
            CHECK(got[i - 1].first > got[i].first || (got[i - 1].first == got[i].first && got[i - 1].second < got[i].second));
    }
    {   // (update moves an event; a repeated id is ignored)
        e::Queue q;
        q.push(e::Event{1, 1, nullptr});
        q.push(e::Event{2, 1, nullptr});
        q.push(e::Event{2, 5, nullptr});
        CHECK(q.size() == 2);
        CHECK(q.update(2, 7));
        CHECK(!q.update(9, 7));
        e::Event ev;
        CHECK(q.pop(ev) && ev.id == 2);
        CHECK(q.pop(ev) && ev.id == 1);
        CHECK(!q.pop(ev));
    }
    {   // (the engine: a line outranks a turn; events from another thread; stop ends run())
        e::Engine eng;
        std::string order;
        eng.add([&] { order += 't'; }, 1);
        eng.add([&] { order += 'l'; }, 2);
        eng.add([&] { order += 'T'; }, 1);
        std::thread th([&] {
            eng.add([&] { order += 'x'; eng.stop(); }, 0);
        });
        eng.run();
        th.join();
        CHECK(order == "ltTx");
        CHECK(eng.add([] {}, 1) == 0);   // stopped
    }
    {   // (a tick runs when no event comes)
        std::atomic<int> ticks{0};
        e::Engine eng([&] { if (++ticks >= 3) { } }, 1);
        std::thread th([&] {
            while (ticks.load() < 3) std::this_thread::yield();
            eng.stop();
        });
        eng.run();
        th.join();
        CHECK(ticks.load() >= 3);
    }
    std::printf("%s (%d failed)\n", g_fail ? "FAILED" : "ALL OK", g_fail);
    return g_fail ? 1 : 0;
}
