#include "kernels.hpp"
#include "engine_parts.hpp"
#include "strata/kprof.hpp"

#include <algorithm>
#include <iostream>
#include <stdexcept>
#include <vector>

static void check_copy_submission() {
    constexpr size_t words = 10 * 1024 * 1024 / sizeof(uint32_t);
    constexpr uint32_t iterations = 32000000, pattern = 0x5a5a5a5a;
    auto hash = [](uint32_t count) {
        uint32_t value = 1234567;
        for (uint32_t i = 0; i < count; ++i) {
            value = value * 1664525u + 1013904223u;
            value ^= value >> 13;
        }
        return value;
    };
    const uint32_t expected = hash(iterations);
    for (int card = 0; card < 2; ++card) {
        dpct::select_device(card);
        auto& device = dpct::get_current_device();
        const auto peer_context = dpct::get_device(1 - card).get_context();
        sycl::queue compute(device.get_context(), device,
            sycl::property_list{sycl::property::queue::in_order{},
                                sycl::property::queue::enable_profiling{}});
        sycl::queue copy(device.get_context(), device, sycl::property::queue::in_order{});
        std::vector<std::pair<void*, sycl::context>> allocations;
        auto alloc = [&](bool host, const sycl::context& context) {
            auto* p = host ? sycl::malloc_host<uint32_t>(words, context)
                           : sycl::malloc_device<uint32_t>(words, device, context);
            if (!p) throw std::bad_alloc();
            allocations.emplace_back(p, context);
            return p;
        };
        try {
            auto* source = alloc(false, compute.get_context());
            auto* destination = alloc(false, compute.get_context());
            auto* local = alloc(true, compute.get_context());
            auto* peer = alloc(true, peer_context);
            auto produce = [&](uint32_t count) {
                return compute.single_task([=] {
                    uint32_t value = 1234567;
                    for (uint32_t i = 0; i < count; ++i) {
                        value = value * 1664525u + 1013904223u;
                        value ^= value >> 13;
                    }
                    source[0] = value;
                });
            };
            for (bool tail : {false, true})
                for (int kind = 0; kind < 5; ++kind)
                    for (int repeat = -1; repeat < 3; ++repeat) {
                        compute.memset(source, 0x5a, words * sizeof(uint32_t)).wait_and_throw();
                        std::fill(local, local + words, pattern);
                        std::fill(peer, peer + words, pattern);
                        const auto began = std::chrono::steady_clock::now();
                        const auto producer = produce(repeat < 0 ? 1 : iterations);
                        const auto produced = std::chrono::steady_clock::now();
                        sycl::event ready;
                        if (tail) ready = fnx::sycl_port::stream_tail(&compute);
                        else dpct::sync_barrier(&ready, &compute);
                        const auto recorded = std::chrono::steady_clock::now();
                        fnx::sycl_port::stream_barrier(&copy, {ready});
                        const auto dependent = std::chrono::steady_clock::now();
                        const uint32_t* from = kind >= 3 ? (kind == 3 ? local : peer) : source;
                        uint32_t* to = kind == 1 ? local : kind == 2 ? peer : destination;
                        auto copied = copy.memcpy(to, from, words * sizeof(uint32_t));
                        const auto submitted = std::chrono::steady_clock::now();
                        copied.wait_and_throw();
                        const auto completed = std::chrono::steady_clock::now();
                        std::vector<uint32_t> actual(words);
                        if (kind == 1 || kind == 2) std::copy(to, to + words, actual.begin());
                        else copy.memcpy(actual.data(), to, words * sizeof(uint32_t)).wait_and_throw();
                        const uint32_t first = kind >= 3 ? pattern : repeat < 0 ? hash(1) : expected;
                        if (actual[0] != first ||
                            !std::all_of(actual.begin() + 1, actual.end(),
                                         [](uint32_t value) { return value == pattern; }))
                            throw std::runtime_error("Copy submission fixture delivered stale data");
                        if (repeat < 0) continue;
                        auto ms = [](auto a, auto b) {
                            return std::chrono::duration<double, std::milli>(b - a).count();
                        };
                        const auto start = producer.get_profiling_info<sycl::info::event_profiling::command_start>();
                        const auto end = producer.get_profiling_info<sycl::info::event_profiling::command_end>();
                        const char* names[] = {"d2d", "d2h-local", "d2h-peer", "h2d-local", "h2d-peer"};
                        std::cout << "COPY_SUBMIT card=" << card << " kind=" << names[kind]
                                  << " tail=" << tail << " repeat=" << repeat
                                  << " producer_submit_ms=" << ms(began, produced)
                                  << " record_ms=" << ms(produced, recorded)
                                  << " dependency_ms=" << ms(recorded, dependent)
                                  << " copy_submit_ms=" << ms(dependent, submitted)
                                  << " total_wall_ms=" << ms(began, completed)
                                  << " producer_gpu_ms=" << double(end - start) / 1e6
                                  << " data=PASS" << std::endl;
                    }
            compute.wait_and_throw();
            copy.wait_and_throw();
        } catch (...) {
            compute.wait_and_throw();
            copy.wait_and_throw();
            for (auto& entry : allocations) sycl::free(entry.first, entry.second);
            throw;
        }
        for (auto& entry : allocations) sycl::free(entry.first, entry.second);
    }
}

static void check_event_recording() {
    auto& q = dpct::get_in_order_queue();
    sycl::queue recording(q.get_context(), q.get_device(),
                          sycl::property::queue::in_order{});
    sycl::queue dependent(q.get_context(), q.get_device(),
                          sycl::property::queue::in_order{});
    auto* flag = sycl::malloc_host<unsigned>(1, q);
    if (!flag) throw std::bad_alloc();
    try {
        sycl::event marker;
        dpct::sync_barrier(&marker, &recording);
        marker.wait_and_throw();
        *flag = 0;
        cuStreamWaitValue32(&recording, reinterpret_cast<dpct::device_ptr>(flag), 1, 0);
        std::thread release([=] {
            std::this_thread::sleep_for(std::chrono::milliseconds(200));
            __atomic_store_n(flag, 1u, __ATOMIC_RELEASE);
        });
        const auto start = std::chrono::steady_clock::now();
        try {
            dpct::sync_barrier(&marker, &recording);
            fnx::sycl_port::stream_barrier(&dependent, {marker});
            dpct::sync_barrier(&marker, &dependent);
        } catch (...) {
            release.join();
            throw;
        }
        const auto submit_time = std::chrono::steady_clock::now() - start;
        release.join();
        marker.wait_and_throw();
        fnx::sycl_port::check_errors();
        if (submit_time > std::chrono::milliseconds(100))
            throw std::runtime_error("Event recording blocked behind a pending host_task");
        std::cout << "PASS async_event_recording pending_host_task delay_ms=200" << std::endl;
    } catch (...) {
        sycl::free(flag, q);
        throw;
    }
    sycl::free(flag, q);
}

static void check_event_mailbox(bool profiling = false) {
    profiling = profiling || strata::kprof::enabled();
    auto& q = dpct::get_in_order_queue();
    const auto properties = profiling
        ? sycl::property_list{sycl::property::queue::in_order{},
                              sycl::property::queue::enable_profiling{}}
        : sycl::property_list{sycl::property::queue::in_order{}};
    auto* data = sycl::malloc_host<int>(256, q);
    if (!data) throw std::bad_alloc();
    auto* device_data = sycl::malloc_device<int>(1, q);
    if (!device_data) {
        sycl::free(data, q);
        throw std::bad_alloc();
    }
    try {
        {
            sycl::queue source(q.get_context(), q.get_device(), properties);
            sycl::queue consumer(q.get_context(), q.get_device(), properties);
            fnx::sycl_port::EventMailbox mailbox;
            unsigned key = 0;
            mailbox.signal(&source, &key, 1);
            mailbox.wait(&consumer, &key, 1);
            const int value = 917;
            {
                strata::kprof::Section section(&source, "mailbox copy producer");
                source.memcpy(device_data, &value, sizeof(value));
            }
            mailbox.signal(&source, &key, 2);
            mailbox.wait(&consumer, &key, 2);
            consumer.single_task([=] { data[1] = *device_data; }).wait_and_throw();
            if (data[1] != value)
                throw std::runtime_error("Mailbox lost its copy dependency");
            const auto start = std::chrono::steady_clock::now();
            {
                strata::kprof::Section section(&source, "mailbox delayed producer");
                source.submit([&](sycl::handler& handler) {
                    handler.host_task([=] {
                        std::this_thread::sleep_for(std::chrono::milliseconds(200));
                        data[0] = 431;
                    });
                });
            }
            mailbox.signal(&source, &key, 3);
            mailbox.wait(&consumer, &key, 3);
            const auto submitted = std::chrono::steady_clock::now() - start;
            consumer.single_task([=] { data[1] = data[0]; }).wait_and_throw();
            if (submitted > std::chrono::milliseconds(100) || data[1] != 431)
                throw std::runtime_error("Mailbox blocked or lost its pending host_task dependency");
        }
        for (int restart = 0; restart < 2; ++restart) {
            std::fill_n(data, 256, -1);
            fnx::sycl_port::EventMailbox mailbox;
            unsigned keys[4]{};
            std::exception_ptr errors[2];
            std::thread workers[2];
            for (int c = 0; c < 2; ++c)
                workers[c] = std::thread([&, c] {
                    try {
                        dpct::select_device(c);
                        auto& devq = dpct::get_in_order_queue();
                        sycl::queue stream(devq.get_context(), devq.get_device(), properties);
                        for (unsigned seq = 1; seq <= 64; ++seq) {
                            if (c == 1 && seq == 3)
                                std::this_thread::sleep_for(std::chrono::milliseconds(50));
                            const int at = c * 64 + int(seq - 1);
                            {
                                strata::kprof::Section section(&stream, "mailbox peer producer");
                                stream.single_task([=] { data[at] = int(seq) + 1000 * c; });
                            }
                            mailbox.signal(&stream, keys + 1 - c, seq);
                            mailbox.wait(&stream, keys + c, seq);
                            stream.single_task([=] {
                                data[128 + at] = data[(1 - c) * 64 + seq - 1];
                            });
                            mailbox.signal(&stream, keys + 2 + 1 - c, seq);
                            mailbox.wait(&stream, keys + 2 + c, seq);
                        }
                        stream.wait_and_throw();
                    } catch (...) {
                        errors[c] = std::current_exception();
                        mailbox.cancel(errors[c]);
                    }
                });
            for (auto& worker : workers) worker.join();
            for (const auto& error : errors)
                if (error) std::rethrow_exception(error);
            for (int c = 0; c < 2; ++c)
                for (int row = 0; row < 64; ++row)
                    if (data[128 + c * 64 + row] != row + 1 + 1000 * (1 - c))
                        throw std::runtime_error("Event mailbox delivered stale peer data");
        }
        fnx::sycl_port::EventMailbox cancelled;
        const auto failure = std::make_exception_ptr(std::runtime_error("peer failure"));
        std::exception_ptr propagated;
        std::thread waiting([&] {
            try { cancelled.wait(&q, reinterpret_cast<unsigned*>(data), 1); }
            catch (...) { propagated = std::current_exception(); }
        });
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
        cancelled.cancel(failure);
        waiting.join();
        if (!propagated) throw std::logic_error("Cancelled mailbox accepted a wait");
        try {
            std::rethrow_exception(propagated);
        } catch (const std::runtime_error& error) {
            if (std::string(error.what()) != "peer failure") throw;
        }
        std::cout << "PASS event_mailbox cards=2 rounds=64 restarts=2 delayed_peer_ms=50 cancellation=PASS empty_queue=PASS copy_dependency=PASS pending_host_task=PASS"
                  << " profiling=" << profiling
                  << std::endl;
    } catch (...) {
        sycl::free(device_data, q);
        sycl::free(data, q);
        throw;
    }
    sycl::free(device_data, q);
    sycl::free(data, q);
}

static void check_kernel_zero() {
    setenv("FNX_KERNEL_ZERO", "1", 1);
    for (int c = 0; c < 2; ++c) {
        dpct::select_device(c);
        auto& q = dpct::get_in_order_queue();
        const size_t capacity = (size_t)64 * 1024 * 1024 + 16;
        auto* data = sycl::malloc_device<unsigned char>(capacity, q);
        if (!data) throw std::bad_alloc();
        try {
            for (size_t bytes : {size_t(0), size_t(1), size_t(7), size_t(256),
                                 size_t(8) * 1024 * 1024, size_t(64) * 1024 * 1024}) {
                for (int offset : {0, 1}) {
                    q.parallel_for(sycl::range<1>(capacity), [=](sycl::id<1> id) {
                        data[id] = 0x7f;
                    });
                    fnx::sycl_port::zero_device(&q, data + offset, bytes).wait_and_throw();
                    std::vector<unsigned char> host(capacity);
                    q.memcpy(host.data(), data, capacity).wait_and_throw();
                    for (size_t i = 0; i < capacity; ++i) {
                        const unsigned char expected = i >= size_t(offset) &&
                            i - offset < bytes ? 0 : 0x7f;
                        if (host[i] != expected)
                            throw std::runtime_error("Kernel zero changed a guard or missed a byte");
                    }
                }
            }
        } catch (...) {
            sycl::free(data, q);
            throw;
        }
        sycl::free(data, q);
        const size_t large_bytes = (size_t(1) << 33) + 4;
        const size_t words = (large_bytes + 60) / sizeof(uint32_t);
        auto* large = sycl::malloc_device<uint32_t>(words, q);
        auto* bad = sycl::malloc_device<unsigned>(65536, q);
        if (!large || !bad) {
            sycl::free(large, q);
            sycl::free(bad, q);
            throw std::bad_alloc();
        }
        try {
            q.parallel_for(sycl::range<1>(65536), [=](sycl::id<1> id) {
                for (size_t i = id[0]; i < words; i += 65536) large[i] = 0x7f7f7f7f;
            });
            fnx::sycl_port::zero_device(&q, large, large_bytes).wait_and_throw();
            q.parallel_for(sycl::range<1>(65536), [=](sycl::id<1> id) {
                unsigned errors = 0;
                for (size_t i = id[0]; i < words; i += 65536)
                    errors |= large[i] != (i < large_bytes / 4 ? 0u : 0x7f7f7f7fu);
                bad[id] = errors;
            });
            std::vector<unsigned> errors(65536);
            q.memcpy(errors.data(), bad, errors.size() * sizeof(unsigned)).wait_and_throw();
            if (std::any_of(errors.begin(), errors.end(), [](unsigned v) { return v != 0; }))
                throw std::runtime_error("Large kernel zero missed data or overwrote its guard");
        } catch (...) {
            sycl::free(large, q);
            sycl::free(bad, q);
            throw;
        }
        sycl::free(large, q);
        sycl::free(bad, q);
    }
    std::cout << "PASS kernel_zero cards=2 sizes=6 aligned_and_unaligned large_bytes=8589934596 guards=PASS" << std::endl;
}

int main(int argc, char** argv) {
    std::vector<std::pair<void*, sycl::context>> allocations;
    std::vector<sycl::queue*> queues;
    auto cleanup = [&] {
        for (auto* q : queues) q->wait_and_throw();
        for (auto& entry : allocations) sycl::free(entry.first, entry.second);
        allocations.clear();
    };
    try {
        if (dpct::device_count() != 2) throw std::runtime_error("Select exactly two GPUs");
        if (argc == 2 && std::string(argv[1]) == "--copy-submit") {
            check_copy_submission();
            return 0;
        }
        if (argc == 2 && std::string(argv[1]) == "--kernel-zero") {
            check_kernel_zero();
            return 0;
        }
        if (argc == 2 && (std::string(argv[1]) == "--event-mailbox" ||
                         std::string(argv[1]) == "--profile-mailbox" ||
                         std::string(argv[1]) == "--profile-queue-mailbox")) {
            const bool profiling = std::string(argv[1]) != "--event-mailbox";
            if (std::string(argv[1]) == "--profile-mailbox") setenv("STRATA_KPROF", "1", 1);
            check_event_mailbox(profiling);
            return 0;
        }
        check_event_recording();
        if (argc == 2 && std::string(argv[1]) == "--event-recording") return 0;
        fnx::Ring rings[2]{};
        float* x[2]{};
        unsigned* tag[2]{};
        auto host = [&](size_t n, sycl::queue& q) {
            auto* p = sycl::malloc_host<unsigned long long>(n, q);
            if (!p) throw std::bad_alloc();
            allocations.emplace_back(p, q.get_context());
            return p;
        };
        auto device = [&](size_t bytes, sycl::queue& q) {
            auto* p = sycl::malloc_device(bytes, q);
            if (!p) throw std::bad_alloc();
            allocations.emplace_back(p, q.get_context());
            q.memset(p, 0, bytes);
            return p;
        };
        for (int c = 0; c < 2; ++c) {
            dpct::select_device(c);
            auto& q = dpct::get_in_order_queue();
            queues.push_back(&q);
            rings[c].card = c;
            rings[c].rows = fnx::MAXT;
            rings[c].in = host(fnx::ring_in_words(fnx::MAXT), q);
            rings[c].ag = host(fnx::ring_ag_words(fnx::MAXT), q);
            rings[c].err = static_cast<unsigned*>(device(sizeof(unsigned), q));
            x[c] = static_cast<float*>(device(fnx::N * fnx::MAXT * sizeof(float), q));
            tag[c] = static_cast<unsigned*>(device(sizeof(unsigned), q));
        }
        if (queues[0]->get_context() != queues[1]->get_context())
            throw std::runtime_error("The ring must use one shared SYCL context");
        unsigned* values[2];
        std::exception_ptr graph_errors[2];
        std::thread graph_workers[2];
        for (int c = 0; c < 2; ++c)
            values[c] = reinterpret_cast<unsigned*>(host(1, *queues[c]));
        for (int c = 0; c < 2; ++c)
            graph_workers[c] = std::thread([&, c] {
                try {
                    dpct::select_device(c);
                    auto& q = *queues[c];
                    auto capture = [&](unsigned* target) {
                        dpct::experimental::begin_recording(&q);
                        q.single_task([=] { *target += 1; });
                        dpct::experimental::command_graph_ptr graph;
                        dpct::experimental::end_recording(&q, &graph);
                        return graph;
                    };
                    fnx::CardBuf::GraphSlot slot;
                    for (int repeat = 0; repeat < 16; ++repeat) {
                        values[c][0] = values[c][1] = 0;
                        fnx::install_graph(slot, capture(values[c]), {0});
                        q.ext_oneapi_graph(*slot.exec).wait_and_throw();
                        fnx::install_graph(slot, capture(values[c] + 1), {1});
                        q.ext_oneapi_graph(*slot.exec).wait_and_throw();
                        if (values[c][0] != 1 || values[c][1] != 1)
                            throw std::runtime_error("Graph replacement kept a stale lane pointer");
                    }
                    delete slot.exec;
                } catch (...) {
                    graph_errors[c] = std::current_exception();
                }
            });
        for (auto& worker : graph_workers) worker.join();
        for (const auto& error : graph_errors)
            if (error) std::rethrow_exception(error);
        std::cout << "PASS concurrent_graph_replacement changes_destination cards=2 repeats=16" << std::endl;
        for (int c = 0; c < 2; ++c) {
            rings[c].next_in = rings[1 - c].in;
            rings[c].next_ag = rings[1 - c].ag;
        }
        fnx::sycl_port::stream_barrier(queues[0]).wait_and_throw();
        auto* flag = reinterpret_cast<unsigned*>(host(1, *queues[0]));
        *flag = 0;
        cuStreamWaitValue32(queues[0], reinterpret_cast<dpct::device_ptr>(flag), 1, 0);
        const auto started = std::chrono::steady_clock::now();
        auto waiting = fnx::sycl_port::stream_barrier(queues[0]);
        sycl::queue dependent(queues[0]->get_context(), queues[0]->get_device(),
                              sycl::property::queue::in_order{});
        auto joined = fnx::sycl_port::stream_barrier(&dependent, {waiting});
        if (std::chrono::steady_clock::now() - started > std::chrono::seconds(1))
            throw std::runtime_error("Barrier submission blocked its peer's flag write");
        if (queues[0]->ext_oneapi_empty() || dependent.ext_oneapi_empty())
            throw std::runtime_error("A pending stream wait was reported complete");
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        cuStreamWriteValue32(queues[1], reinterpret_cast<dpct::device_ptr>(flag), 1, 0);
        joined.wait_and_throw();
        fnx::sycl_port::check_errors();
        std::cout << "PASS delayed_stream_flag_and_async_barrier cards=2 delay_ms=100" << std::endl;
        int call = 1;
        for (const bool fp32 : {false, true}) {
            for (int c = 0; c < 2; ++c) {
                rings[c].f32 = fp32;
                queues[c]->wait_and_throw();
                std::fill_n(static_cast<unsigned long long*>(rings[c].in),
                            fnx::ring_in_words(fnx::MAXT), 1ull << 32);
                std::fill_n(static_cast<unsigned long long*>(rings[c].ag),
                            fnx::ring_ag_words(fnx::MAXT), 1ull << 32);
            }
            // Pre-tagged mailboxes allow each device's first JIT launch to finish alone.
            for (int c = 0; c < 2; ++c) {
                dpct::select_device(c);
                fnx::ring_sum(rings[c], x[c], fnx::MAXT, tag[c], 0, queues[c]);
                queues[c]->wait_and_throw();
            }
            for (const int tokens : {1, fnx::MAXT}) {
                const int n = tokens * fnx::N;
                std::vector<float> input[2];
                for (int c = 0; c < 2; ++c) {
                    input[c].resize(n);
                    for (int i = 0; i < n; ++i)
                        input[c][i] = (int((i * 19 + c * 7) % 97) - 48) / 8.0f;
                    queues[c]->memcpy(x[c], input[c].data(), n * sizeof(float)).wait_and_throw();
                }
                for (int c = 0; c < 2; ++c) {
                    dpct::select_device(c);
                    fnx::ring_sum(rings[c], x[c], tokens, tag[c], call, queues[c]);
                }
                ++call;
                for (int c = 0; c < 2; ++c) {
                    std::vector<float> got(n);
                    unsigned error = 0;
                    queues[c]->memcpy(got.data(), x[c], n * sizeof(float));
                    queues[c]->memcpy(&error, rings[c].err, sizeof(error)).wait_and_throw();
                    if (error) throw std::runtime_error("Ring reported error " + std::to_string(error));
                    for (int i = 0; i < n; ++i)
                        if (got[i] != input[0][i] + input[1][i])
                            throw std::runtime_error("Ring differs from the CPU sum");
                }
                std::cout << "PASS ring=" << (fp32 ? "fp32" : "fp16")
                          << " tokens=" << tokens << " cards=2" << std::endl;
            }
        }
        cleanup();
    } catch (const sycl::exception& e) {
        std::cerr << "FAIL " << e.what() << std::endl;
        std::_Exit(1);
    } catch (const std::exception& e) {
        std::cerr << "FAIL " << e.what() << std::endl;
        try { cleanup(); } catch (...) {}
        return 1;
    }
}
