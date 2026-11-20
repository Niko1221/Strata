// Controlled CPU fixtures using the exact collector called by production hooks.
// The fake clock does not represent GPU execution or a benchmark.
#include "strata/perf131k/trace.hpp"
#include <cstring>
#include <thread>
#include <vector>

using namespace strata::perf131k;
std::atomic<uint64_t> ticks{0};
uint64_t fake_clock() { return ticks.fetch_add(10) + 10; }
int main(int argc, char **argv) {
    if (argc != 3)
        return 2;
    const char *mode = argv[1];
    if (std::strcmp(mode, "runtime") == 0) {
        Runtime runtime;
        Scope request(Stage::request, 0, 4096, -1, true);
        Scope child(Stage::prefill_host, 0, 4096);
        return 0;
    }
    auto recorder = std::make_unique<Recorder>(fake_clock);
    if (std::strcmp(mode, "off") == 0) {
        context.parent = 123;
        {
            Scope ignored(Stage::request);
        }
        if (ticks.load() != 0 || context.parent != 123)
            return 3;
        std::puts("OFF: clock untouched, TLS unchanged, no output created");
        return 0;
    }
    active.store(recorder.get());
    if (std::strcmp(mode, "normal") == 0) {
        graph_created(recorder.get());
        {
            Scope request(Stage::request, 0, 4096, -1, true);
            Scope verify(Stage::verify_host, 4095, 4096);
            {
                Scope issue(Stage::graph_launch_issue);
                issue.metadata(&ticks, recorder.get());
            }
            {
                Scope copy(Stage::h2d_issue, -1, -1, 2);
                copy.metadata(&ticks, nullptr, -1, 256);
            }
        }
        std::vector<std::thread> threads;
        for (int i = 0; i < 4; ++i)
            threads.emplace_back([] {
                for (int j = 0; j < 40; ++j) {
                    Scope read(Stage::expert_copy_host);
                    read.metadata(nullptr, nullptr, 128);
                }
            });
        for (auto &thread : threads)
            thread.join();
        graph_retired(recorder.get());
    } else if (std::strcmp(mode, "lifecycle") == 0) {
        for (int i = 0; i < 2; ++i) {
            graph_created(recorder.get());
            {
                Scope issue(Stage::graph_enqueue_host);
                issue.metadata(&ticks, recorder.get());
            }
            graph_retired(recorder.get());
        }
    } else if (std::strcmp(mode, "overflow") == 0) {
        for (size_t i = 0; i < Recorder::capacity + 3; ++i) {
            Scope scope(Stage::verify_host);
        }
    } else if (std::strcmp(mode, "unwind") == 0) {
        try {
            Scope scope(Stage::verify_host);
            throw 1;
        } catch (int) {
        }
    } else if (std::strcmp(mode, "identities") == 0) {
        std::array<int, 300> handles{};
        for (auto &handle : handles) {
            graph_created(&handle);
            Scope scope(Stage::graph_enqueue_host);
            scope.metadata(&ticks, &handle);
        }
    } else if (std::strcmp(mode, "unfinished") == 0) {
        std::FILE *file = exclusive_output(argv[2]);
        if (!file)
            return 4;
        {
            Scope scope(Stage::verify_host);
            recorder->write_and_close(file);
        }
        active.store(nullptr);
        return std::fclose(file) == 0 ? 0 : 6;
    } else
        return 2;
    active.store(nullptr);
    std::FILE *file = exclusive_output(argv[2]);
    if (!file) {
        std::puts("exclusive open rejected");
        return 4;
    }
    bool ok = recorder->write_and_close(file);
    if (std::fclose(file) != 0)
        ok = false;
    if (context.parent != 0 || context.generation != 0)
        return 5;
    std::puts(ok ? "CPU collector fixture written" : "export failed");
    return ok ? 0 : 6;
}
