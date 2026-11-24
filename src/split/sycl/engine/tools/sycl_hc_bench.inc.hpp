#include "sycl_kernel_bench.hpp"
#include "strata/prefill/gemm.hpp"
#include <oneapi/mkl.hpp>

namespace fnx::native_prefill {
void gr_norm_rs(const float*, const float*, float, float*, uint16_t*, int64_t, void*,
                uint16_t*, int64_t, bool);
void gr_silu(const float*, uint16_t*, int64_t, void*, uint16_t*);
void gr_mix_r(const float*, const float*, const float*, const float*, float*, uint16_t*,
              int64_t, void*, uint16_t*, uint16_t*);
void gr_write(float*, const float*, const float*, int64_t, int64_t, void*);
}

namespace fnx {
void check_attention_landing_reuse(bool ack, int sender, int repeats, int rows) {
    const int peer = 1 - sender;
    const size_t count = attn_prompt_block_floats(rows);
    dpct::select_device(sender);
    auto& producer = dpct::get_in_order_queue();
    sycl::queue send(producer.get_context(), producer.get_device(),
                     sycl::property::queue::in_order{});
    kernel_bench::Fixture source_fixture(producer);
    dpct::select_device(peer);
    auto& consumer = dpct::get_in_order_queue();
    if (producer.get_context() != consumer.get_context())
        throw std::runtime_error("Attention reuse fixture requires shared context");
    kernel_bench::Fixture target_fixture(consumer);
    auto* source = source_fixture.alloc<float>(NCARD * count);
    auto* target = target_fixture.alloc<float>(count);
    auto* guarded = sycl::malloc_host<float>(count + 2, consumer);
    if (!guarded) throw std::bad_alloc();
    std::fill_n(guarded, count + 2, -1234.f);
    auto* landing = guarded + 1;
    unsigned flags[ATT_PIECES_MAX]{}, consumed{};
    Engine::Pf p;
    p.attn_copy_events = true;
    p.events = std::make_shared<sycl_port::EventMailbox>();
    p.apart[0] = source;
    p.parecv[0][peer] = landing;
    p.paflags[0][peer] = flags;
    int stale = 0;
    for (int round = 0; round < repeats; ++round) {
        const unsigned seq = 2 * round + 1;
        const float before = float(16 * round + 1), after = before + 1;
        dpct::select_device(sender);
        auto ready = producer.fill(source, before, NCARD * count);
        auto sent = pf_attn_send(p, 0, sender, rows, 0, rows, 0, seq, &send, &ready);
        sent->wait_and_throw();
        dpct::select_device(peer);
        consumer.submit([&](sycl::handler& h) {
            h.host_task([] { std::this_thread::sleep_for(std::chrono::milliseconds(100)); });
        });
        const auto copied = consumer.memcpy(target, landing, count * sizeof(float));
        p.events->publish(&consumed, seq, copied);
        const auto reusable = p.events->event(&consumed, seq);
        dpct::select_device(sender);
        ready = producer.fill(source, after, NCARD * count);
        sent = pf_attn_send(p, 0, sender, rows, 0, rows, 0, seq + 1, &send, &ready,
                            ack ? &reusable : nullptr);
        sent->wait_and_throw();
        auto first = target_fixture.read(target, count);
        if (!std::all_of(first.begin(), first.end(), [=](float v) { return v == before; }))
            ++stale;
        consumer.memcpy(target, landing, count * sizeof(float));
        auto second = target_fixture.read(target, count);
        if (!std::all_of(second.begin(), second.end(), [=](float v) { return v == after; }) ||
            guarded[0] != -1234.f || guarded[count + 1] != -1234.f)
            throw std::runtime_error("Attention reuse fixture copy/guard failure");
    }
    sycl::free(guarded, consumer);
    if ((ack && stale != 0) || (!ack && stale == 0))
        throw std::runtime_error("Attention reuse causal control failed");
    std::cout << (ack ? "PASS" : "REPRODUCED")
              << " attention_reuse sender=" << sender << " ack=" << ack
              << " rows=" << rows << " rounds=" << repeats << " stale=" << stale
              << " guards=PASS" << std::endl;
}

void check_attention_copy(bool direct, int repeats, int rows) {
    constexpr int pieces = 4;
    const size_t count = attn_prompt_block_floats(rows);
    std::array<Engine::Pf, 2> cards;
    std::array<sycl::queue*, 2> compute;
    std::array<std::unique_ptr<sycl::queue>, 2> send, receive;
    std::array<std::unique_ptr<kernel_bench::Fixture>, 2> fixtures;
    auto mailbox = std::make_shared<sycl_port::EventMailbox>();
    std::vector<void*> pinned;
    float* destination[2]{};
    for (int c = 0; c < 2; ++c) {
        dpct::select_device(c);
        compute[c] = &dpct::get_in_order_queue();
        send[c] = std::make_unique<sycl::queue>(compute[c]->get_context(),
            compute[c]->get_device(), sycl::property::queue::in_order{});
        receive[c] = std::make_unique<sycl::queue>(compute[c]->get_context(),
            compute[c]->get_device(), sycl::property::queue::in_order{});
        fixtures[c] = std::make_unique<kernel_bench::Fixture>(*compute[c]);
        destination[c] = fixtures[c]->alloc<float>(count);
        auto& p = cards[c];
        p.events = mailbox;
        p.attn_copy_events = direct;
        for (int set = 0; set < 2; ++set) {
            p.apart[set] = fixtures[c]->alloc<float>(NCARD * count);
            auto* landing = sycl::malloc_host<float>(count + 2, *compute[c]);
            auto* flags = sycl::malloc_host<unsigned>(pieces, *compute[c]);
            if (!landing || !flags) throw std::bad_alloc();
            pinned.push_back(landing);
            pinned.push_back(flags);
            std::fill_n(landing, count + 2, -1234.f);
            p.arecv[set] = landing + 1;
            p.aflags[set] = flags;
        }
    }
    for (int c = 0; c < 2; ++c)
        for (int set = 0; set < 2; ++set) {
            cards[c].parecv[set][1 - c] = cards[1 - c].arecv[set];
            cards[c].paflags[set][1 - c] = cards[1 - c].aflags[set];
        }
    for (int round = 0; round < repeats; ++round) {
        std::exception_ptr errors[2];
        std::thread workers[2];
        for (int c = 0; c < 2; ++c)
            workers[c] = std::thread([&, c] {
                try {
                    dpct::select_device(c);
                    auto& p = cards[c];
                    const unsigned seq = round + 1;
                    const int set = seq & 1u;
                    auto* source = p.apart[set];
                    if (round == 0 && c == 1)
                        std::this_thread::sleep_for(std::chrono::milliseconds(100));
                    auto ready = compute[c]->parallel_for(sycl::range<1>(NCARD * count),
                        [=](sycl::id<1> id) {
                            source[id] = float((id[0] + seq * 7 + c * 11) % 97) / 8.f;
                        });
                    sycl_port::stream_barrier(send[c].get(), {ready});
                    for (int i = 0; i < pieces; ++i) {
                        const int a = rows * i / pieces, e = rows * (i + 1) / pieces;
                        pf_attn_send(p, set, c, rows, a, e - a, i, seq, send[c].get(), &ready);
                    }
                    for (int i = 0; i < pieces; ++i) {
                        const int a = rows * i / pieces, e = rows * (i + 1) / pieces;
                        const auto produced = mailbox->event(p.aflags[set] + i, seq);
                        receive[c]->submit([&](sycl::handler& h) {
                            h.depends_on(produced);
                            h.memcpy(destination[c] + attn_prompt_block_floats(a),
                                     p.arecv[set] + attn_prompt_block_floats(a),
                                     attn_prompt_block_floats(e - a) * sizeof(float));
                        });
                    }
                    std::vector<float> actual(count);
                    receive[c]->memcpy(actual.data(), destination[c], count * sizeof(float)).wait_and_throw();
                    send[c]->wait_and_throw();
                    for (size_t i = 0; i < count; ++i)
                        if (actual[i] != float((c * count + i + seq * 7 + (1 - c) * 11) % 97) / 8.f)
                            throw std::runtime_error("Attention copy delivered stale partials");
                    if (p.arecv[set][-1] != -1234.f || p.arecv[set][count] != -1234.f)
                        throw std::runtime_error("Attention copy overwrote a landing guard");
                } catch (...) {
                    errors[c] = std::current_exception();
                    mailbox->cancel(errors[c]);
                }
            });
        for (auto& worker : workers) worker.join();
        for (auto& error : errors) if (error) std::rethrow_exception(error);
    }
    for (void* p : pinned) sycl::free(p, compute[0]->get_context());
    std::cout << "PASS attention_copy direct=" << direct << " rows=" << rows
              << " rounds=" << repeats << " pieces=4 data=PASS guards=PASS" << std::endl;
}

void check_hc_swap(const std::string& mode, int repeats, int rows, int cols, int pieces) {
    constexpr int chunks = 8;
    const size_t count = size_t(rows) * cols, share = size_t((rows + 1) / 2) * cols;
    const bool serialized = mode == "own-serial";
    const bool own_events = mode == "own-events";
    const bool staged = mode == "own-stages" || mode == "own-stages-direct" || own_events;
    const bool reduce = mode == "own-ring" || mode == "own-direct" || serialized || staged;
    const bool native_recording = mode == "native", use_events = mode == "events" || reduce;
    if (reduce && cols != N) throw std::invalid_argument("Own-ring fixture requires N columns");
    const auto properties = sycl::property_list{sycl::property::queue::in_order{}};
    std::array<sycl::queue*, 2> compute;
    std::array<std::unique_ptr<sycl::queue>, 2> exchange;
    std::array<std::unique_ptr<kernel_bench::Fixture>, 2> fixtures;
    float* matrix[2][chunks]{};
    float* landing[2][2]{};
    unsigned* flags[2][2]{};
    unsigned* data_error[2]{};
    unsigned hc_consumed[2][2]{}, hc_reuse_seq[2][2]{};
    Engine::PfRing rings[2]{};
    std::vector<void*> pinned;
    sycl_port::EventMailbox mailbox;
    for (int c = 0; c < 2; ++c) {
        dpct::select_device(c);
        compute[c] = &dpct::get_in_order_queue();
        exchange[c] = std::make_unique<sycl::queue>(
            compute[c]->get_context(), compute[c]->get_device(), properties);
        fixtures[c] = std::make_unique<kernel_bench::Fixture>(*compute[c]);
        data_error[c] = fixtures[c]->alloc<unsigned>(6);
        compute[c]->memset(data_error[c], 0, 6 * sizeof(unsigned));
        if (reduce) {
            auto& r = rings[c];
            r.plan = mode == "own-direct" || mode == "own-stages-direct"
                ? pf_ring_plan_direct(2, c) : pf_ring_plan(2, c);
            r.send = fixtures[c]->alloc<sycl::half>(share);
            r.flags = sycl::malloc_host<unsigned>(2 * PF_RING_STEPS, *compute[c]);
            if (!r.flags) throw std::bad_alloc();
            pinned.push_back(r.flags);
            std::fill_n(r.flags, 2 * PF_RING_STEPS, 0u);
            for (int set = 0; set < 2; ++set) {
                r.in[0][set] = sycl::malloc_host<sycl::half>(share, *compute[c]);
                r.ag[set] = sycl::malloc_host<sycl::half>(count, *compute[c]);
                if (!r.in[0][set] || !r.ag[set]) throw std::bad_alloc();
                pinned.push_back(r.in[0][set]);
                pinned.push_back(r.ag[set]);
                r.din[0][set] = fixtures[c]->alloc<sycl::half>(share);
                r.dag[set] = fixtures[c]->alloc<sycl::half>(count);
            }
        }
        for (auto& m : matrix[c]) m = fixtures[c]->alloc<float>(count);
        for (int set = 0; set < 2; ++set) {
            auto* p = sycl::malloc_host<float>(share + 2, *compute[c]);
            auto* f = sycl::malloc_host<unsigned>(pieces, *compute[c]);
            if (!p || !f) throw std::bad_alloc();
            pinned.push_back(p);
            pinned.push_back(f);
            std::fill_n(p, share + 2, -1234.f);
            std::fill_n(f, pieces, 0u);
            landing[c][set] = p + 1;
            flags[c][set] = f;
        }
    }
    if (compute[0]->get_context() != compute[1]->get_context())
        throw std::runtime_error("HC swap requires the shared two-card context");
    if (reduce)
        for (int c = 0; c < 2; ++c) {
            rings[c].nflags = rings[1 - c].flags;
            rings[c].next_consumed = rings[1 - c].consumed;
            for (int set = 0; set < 2; ++set) {
                rings[c].nin[0][set] = rings[1 - c].in[0][set];
                rings[c].nag[set] = rings[1 - c].ag[set];
            }
        }
    auto prepare = [&](int c, int j, unsigned seq) {
        float* m = matrix[c][j];
        const int own = (c + 1) % 2, begin = own * rows / 2, end = (own + 1) * rows / 2;
        return compute[c]->parallel_for(sycl::range<1>(count), [=](sycl::id<1> id) {
            const size_t i = id[0];
            m[i] = reduce ? float(int((i + seq * 7 + c * 11) % 97) - 48) / 8.f
                : (i / cols >= size_t(begin) && i / cols < size_t(end)
                    ? float((c + 1) * 100000 + seq * 1000 + i) : -999.f);
        });
    };
    auto check_stage = [&](int c, int j, unsigned seq, unsigned stage) {
        const float* m = matrix[c][j];
        unsigned* error = data_error[c];
        const int own = 1 - c, begin = own * rows / 2, end = (own + 1) * rows / 2;
        return compute[c]->parallel_for(sycl::range<1>(count), [=](sycl::id<1> id) {
            const size_t i = id[0];
            if (stage == 2 && (i / cols < size_t(begin) || i / cols >= size_t(end)))
                return;
            const float expected = stage == 1
                ? float(int((i + seq * 7 + c * 11) % 97) - 48) / 8.f
                : (float(int((i + seq * 7) % 97) - 48) +
                   float(int((i + seq * 7 + 11) % 97) - 48)) / 8.f;
            if (m[i] != expected) {
                unsigned empty = 0;
                if (sycl::atomic_ref<unsigned, sycl::memory_order::relaxed,
                    sycl::memory_scope::device,
                    sycl::access::address_space::global_space>(*error)
                        .compare_exchange_strong(empty, 1)) {
                    error[1] = seq;
                    error[2] = unsigned(i);
                    error[3] = sycl::bit_cast<unsigned>(m[i]);
                    error[4] = sycl::bit_cast<unsigned>(expected);
                    error[5] = stage;
                }
            }
        });
    };
    for (int c = 0; c < 2; ++c) {
        dpct::select_device(c);
        prepare(c, 0, 0).wait_and_throw();
        cuStreamWriteValue32(exchange[c].get(),
                            reinterpret_cast<dpct::device_ptr>(flags[c][0]), 0, 0);
        cuStreamWaitValue32(exchange[c].get(),
                           reinterpret_cast<dpct::device_ptr>(flags[c][0]), 0, 0);
        exchange[c]->wait_and_throw();
        sycl_port::stream_barrier(exchange[c].get()).wait_and_throw();
        sycl::event warm;
        dpct::sync_barrier(&warm, exchange[c].get());
        warm.wait_and_throw();
    }
    std::cout << "BEGIN hc_swap mode=" << mode
              << " cards=2 chunks=" << chunks << " rows=" << rows
              << " cols=" << cols << " pieces=" << pieces << " repeats=" << repeats
              << " delayed_peer_ms=100 profiling=0" << std::endl;
    std::exception_ptr errors[2];
    std::thread workers[2];
    for (int c = 0; c < 2; ++c) {
        workers[c] = std::thread([&, c] {
            try {
                dpct::select_device(c);
                for (int round = 0; round < repeats; ++round) {
                    std::array<sycl::event, chunks> done;
                    for (int j = 0; j < chunks; ++j) {
                        const unsigned seq = round * chunks + j + 1;
                        const int set = seq & 1u, peer = 1 - c;
                        auto ready = prepare(c, j, seq);
                        if (staged) check_stage(c, j, seq, 1);
                        if (reduce) {
                            if (own_events)
                                ready = pf_own_copy_events(rings[c], c, matrix[c][j], rows, seq,
                                                           data_error[c], compute[c], mailbox, ready);
                            else pf_ring_run(rings[c], c, c, matrix[c][j], rows, seq,
                                             data_error[c], compute[c], true, &mailbox);
                            if (serialized) compute[c]->wait_and_throw();
                            if (!own_events) ready = sycl_port::stream_barrier(compute[c]);
                            if (staged) ready = check_stage(c, j, seq, 2);
                        }
                        sycl_port::stream_barrier(exchange[c].get(), {ready});
                        if (c == 1 && seq == 1)
                            std::this_thread::sleep_for(std::chrono::milliseconds(100));
                        if (seq == 1)
                            std::cout << "HC_SWAP card=" << c << " before_swap" << std::endl;
                        std::optional<sycl::event> landing_ready;
                        if (own_events && hc_reuse_seq[c][set])
                            landing_ready = mailbox.event(hc_consumed[peer] + set, hc_reuse_seq[c][set]);
                        auto copied = pf_swap_rows(matrix[c][j], cols * sizeof(float), rows, c, pieces,
                                     landing[c][set], flags[c][set], landing[peer][set],
                                     flags[peer][set], seq, exchange[c].get(),
                                     use_events ? &mailbox : nullptr, use_events ? &ready : nullptr,
                                     landing_ready ? &*landing_ready : nullptr);
                        if (own_events) {
                            mailbox.publish(hc_consumed[c] + set, seq, copied);
                            hc_reuse_seq[c][set] = seq;
                        }
                        if (seq == 1)
                            std::cout << "HC_SWAP card=" << c << " before_record" << std::endl;
                        if (native_recording) dpct::sync_barrier(&done[j], exchange[c].get());
                        else done[j] = use_events ? copied : sycl_port::stream_barrier(exchange[c].get());
                        if (serialized) done[j].wait_and_throw();
                    }
                    for (const auto& event : done)
                        sycl_port::stream_barrier(compute[c], {event});
                    for (int j = 0; j < chunks; ++j) {
                        const unsigned seq = round * chunks + j + 1;
                        const float* m = matrix[c][j];
                        unsigned* error = data_error[c];
                        compute[c]->parallel_for(sycl::range<1>(count), [=](sycl::id<1> id) {
                            const size_t i = id[0];
                            const int owner = i / cols < size_t(rows / 2) ? 1 : 0;
                            const float expected = reduce
                                ? (float(int((i + seq * 7) % 97) - 48) +
                                   float(int((i + seq * 7 + 11) % 97) - 48)) / 8.f
                                : float((owner + 1) * 100000 + seq * 1000 + i);
                            if (m[i] != expected) {
                                unsigned empty = 0;
                                if (sycl::atomic_ref<unsigned, sycl::memory_order::relaxed,
                                    sycl::memory_scope::device,
                                    sycl::access::address_space::global_space>(*error)
                                        .compare_exchange_strong(empty, 1)) {
                                    error[1] = seq;
                                    error[2] = unsigned(i);
                                    error[3] = sycl::bit_cast<unsigned>(m[i]);
                                    error[4] = sycl::bit_cast<unsigned>(expected);
                                    error[5] = 3;
                                }
                            }
                        });
                    }
                }
                compute[c]->wait_and_throw();
                exchange[c]->wait_and_throw();
            } catch (...) {
                errors[c] = std::current_exception();
                mailbox.cancel(errors[c]);
            }
        });
    }
    for (auto& worker : workers) worker.join();
    for (const auto& error : errors) if (error) std::rethrow_exception(error);
    sycl_port::check_errors();
    for (int c = 0; c < 2; ++c) {
        auto error = fixtures[c]->read(data_error[c], 6);
        if (error[0]) {
            std::cerr << "HC_SWAP_DATA_ERROR card=" << c << " seq=" << error[1]
                      << " element=" << error[2]
                      << " actual=" << sycl::bit_cast<float>(error[3])
                      << " expected=" << sycl::bit_cast<float>(error[4])
                      << " stage=" << error[5] << std::endl;
            throw std::runtime_error("HC swap delivered incorrect rows in an earlier round");
        }
        for (int j = 0; j < chunks; ++j) {
            const unsigned seq = (repeats - 1) * chunks + j + 1;
            auto values = fixtures[c]->read(matrix[c][j], count);
            for (size_t i = 0; i < count; ++i) {
                const int owner = i / cols < rows / 2 ? 1 : 0;
                const float expected = reduce
                    ? (float(int((i + seq * 7) % 97) - 48) +
                       float(int((i + seq * 7 + 11) % 97) - 48)) / 8.f
                    : float((owner + 1) * 100000 + seq * 1000 + i);
                if (values[i] != expected)
                    throw std::runtime_error("HC swap delivered stale or incorrect rows");
            }
        }
        for (int set = 0; set < 2; ++set)
            if (landing[c][set][-1] != -1234.f || landing[c][set][share] != -1234.f)
                throw std::runtime_error("HC swap overwrote a landing-buffer guard");
    }
    for (void* p : pinned) sycl::free(p, compute[0]->get_context());
    std::cout << "PASS hc_swap mode=" << mode
              << " exact_rows=PASS all_rounds=PASS double_buffer_reuse=PASS guards=PASS" << std::endl;
}

void benchmark_hc(sycl::queue& q, int card, int rows, int warmup, int repeats, bool update_only) {
    using namespace kernel_bench;
    Fixture f(q);
    std::cout << "SETUP HC card=" << card << " rows=" << rows
              << " N=" << N << " HC=" << HC << " LR=" << LR
              << " PP BF16 / all-cards FP16; weights exactly representable in both"
              << std::endl;
    auto* R = f.random((size_t)rows * D, 0.5f, 73);
    auto* original = f.alloc<float>((size_t)rows * D);
    q.memcpy(original, R, (size_t)rows * D * sizeof(float)).wait_and_throw();
    auto* norm = f.upload(std::vector<float>(D, 1.f));
    auto* bo = f.random((size_t)rows * N, 0.01f, 81);
    auto* inj_prev = f.random((size_t)rows * HC, 0.1f, 82);
    auto* pp_inj = f.alloc<float>((size_t)rows * HC);
    auto* split_inj = f.alloc<float>((size_t)rows * HC);
    auto* pp_mix = f.alloc<float>((size_t)rows * N);
    auto* split_mix = f.alloc<float>((size_t)rows * N);
    auto* mixh = f.alloc<sycl::half>((size_t)rows * N);
    auto* pp_mixh = f.alloc<uint16_t>((size_t)rows * N);
    auto* pp_rs = f.alloc<float>((size_t)rows * HC);
    auto* pp_x = f.alloc<uint16_t>((size_t)rows * D);
    auto* pp_lo = f.alloc<float>((size_t)rows * LR);
    auto* pp_lo16 = f.alloc<uint16_t>((size_t)rows * LR);
    auto* pp_up = f.alloc<float>((size_t)rows * D);
    std::mt19937 rng(91);
    std::vector<int8_t> down((size_t)LR * D), up((size_t)D * LR);
    std::vector<uint16_t> pp_down(down.size()), pp_upw(up.size()), iw((size_t)HC * D);
    auto bf = [](float v) { return uint16_t(sycl::bit_cast<uint32_t>(v) >> 16); };
    for (size_t i = 0; i < down.size(); ++i) {
        down[i] = int(rng() % 15) - 7;
        pp_down[i] = bf(down[i] / 512.f);
    }
    for (size_t i = 0; i < up.size(); ++i) {
        up[i] = int(rng() % 15) - 7;
        pp_upw[i] = bf(up[i] / 512.f);
    }
    for (auto& v : iw) v = bf((int(rng() % 15) - 7) / 512.f);
    const uint16_t scale = sycl::bit_cast<uint16_t>(sycl::half(1.f / 512));
    Q8S wd{f.upload(down), f.upload(std::vector<uint16_t>(down.size() / 32, scale))};
    Q8S wu{f.upload(up), f.upload(std::vector<uint16_t>(up.size() / 32, scale))};
    auto* bwd = f.upload(pp_down);
    auto* bwu = f.upload(pp_upw);
    auto* wi = f.upload(iw);
    Engine::Pf p{};
    dpct::blas::descriptor descriptor;
    descriptor.set_queue(&q);
    p.blas = &descriptor;
    p.rs = f.alloc<float>((size_t)rows * HC);
    p.xh = f.alloc<sycl::half>((size_t)rows * D);
    p.wh = f.alloc<sycl::half>((size_t)D * LR);
    p.lo32 = f.alloc<float>((size_t)rows * LR);
    p.lo16 = f.alloc<sycl::half>((size_t)rows * LR);
    p.up32 = f.alloc<float>((size_t)rows * D);
    strata::prefill::Gemm gemm;
    std::string error;
    if (!gemm.init_external(&q, reinterpret_cast<uint16_t*>(p.wh), (size_t)D * LR,
                            nullptr, 0, error))
        throw std::runtime_error(error);
    auto pp = [&] {
        native_prefill::gr_norm_rs(R, norm, 1e-6f, pp_rs, pp_x, rows, &q, nullptr, 0, false);
        gemm.bf16(pp_x, bwd, pp_lo, rows, LR, D);
        native_prefill::gr_silu(pp_lo, pp_lo16, rows, &q, nullptr);
        gemm.bf16(pp_lo16, bwu, pp_up, rows, D, LR);
        native_prefill::gr_mix_r(R, pp_rs, norm, pp_up, pp_mix, nullptr,
                                 rows, &q, pp_mixh, nullptr);
        gemm.bf16(pp_x, wi, pp_inj, rows, HC, D);
    };
    auto split = [&] {
        pf_hc_read(p, R, nullptr, nullptr, norm, wd, wu, wi, 1e-6f,
                    split_mix, split_inj, rows, &q, mixh);
    };
    auto reset = [&] {
        q.memcpy(R, original, (size_t)rows * D * sizeof(float)).wait_and_throw();
    };
    auto split_rows = [&] {
        pf_hc_read_rows(p, R, bo, inj_prev, norm, wd, wu, wi, 1e-6f,
                       split_mix, split_inj, 0, rows / 2, &q, mixh);
        pf_hc_read_rows(p, R, bo, inj_prev, norm, wd, wu, wi, 1e-6f,
                       split_mix, split_inj, rows / 2, rows - rows / 2, &q, mixh);
    };
    reset();
    pf_hc_read(p, R, bo, inj_prev, norm, wd, wu, wi, 1e-6f,
               split_mix, split_inj, rows, &q, mixh);
    q.wait_and_throw();
    const auto full_R = f.read(R, (size_t)rows * D);
    const auto full_mix = f.read(split_mix, (size_t)rows * N);
    const auto full_inj = f.read(split_inj, (size_t)rows * HC);
    reset(); split_rows(); q.wait_and_throw();
    compare(full_R, f.read(R, full_R.size()), 0.f, 0.f, "hc_rows_residual");
    compare(full_mix, f.read(split_mix, full_mix.size()), 0.00001f, 0.00001f, "hc_rows_mix");
    compare(full_inj, f.read(split_inj, full_inj.size()), 0.f, 0.f, "hc_rows_inject");
    reset();
    if (update_only) {
        auto pp_update = [&] {
            native_prefill::gr_write(R, bo, inj_prev, HC, rows, &q);
            pp();
        };
        auto split_update = [&] {
            pf_hc_read(p, R, bo, inj_prev, norm, wd, wu, wi, 1e-6f,
                        split_mix, split_inj, rows, &q, mixh);
        };
        reset(); pp_update(); q.wait_and_throw();
        auto expected_R = f.read(R, (size_t)rows * D);
        reset(); split_update(); q.wait_and_throw();
        compare(expected_R, f.read(R, expected_R.size()), 0.f, 0.f,
                "hc_fused_residual_T" + std::to_string(rows));
        compare(f.read(pp_mix, (size_t)rows * N), f.read(split_mix, (size_t)rows * N),
                0.003f, 0.01f, "hc_update_mix_T" + std::to_string(rows));
        compare(f.read(pp_inj, (size_t)rows * HC), f.read(split_inj, (size_t)rows * HC),
                0.02f, 0.01f, "hc_update_inject_T" + std::to_string(rows));
        paired(q, card, "hc_read_write_T" + std::to_string(rows),
               pp_update, split_update, warmup, repeats, reset);
        return;
    }
    pp(); split(); q.wait_and_throw();
    compare(f.read(pp_mix, (size_t)rows * N), f.read(split_mix, (size_t)rows * N),
            0.003f, 0.01f, "hc_mix_T" + std::to_string(rows));
    compare(f.read(pp_inj, (size_t)rows * HC), f.read(split_inj, (size_t)rows * HC),
            0.02f, 0.01f, "hc_inject_T" + std::to_string(rows));
    paired(q, card, "hc_read_T" + std::to_string(rows), pp, split, warmup, repeats);
    auto write_pp = [&] {
        native_prefill::gr_write(R, bo, inj_prev, HC, rows, &q);
    };
    auto write_split = [&] { hc_write(R, bo, inj_prev, rows, &q); };
    reset(); write_pp(); q.wait_and_throw();
    auto ref = f.read(R, (size_t)rows * D);
    reset(); write_split(); q.wait_and_throw();
    compare(ref, f.read(R, ref.size()), 0.000001f, 0.000001f,
            "hc_write_T" + std::to_string(rows));
    paired(q, card, "hc_write_T" + std::to_string(rows),
           write_pp, write_split, warmup, repeats, reset);
    reset();

    // Separate true oneMKL event times from the production APIs' tag intervals.
    for (const auto& shape : {std::pair<int, int>{LR, D}, {D, LR}, {HC, D}}) {
        const int n = shape.first, k = shape.second;
        std::vector<sycl::half> x((size_t)rows * k), w((size_t)n * k);
        std::vector<oneapi::mkl::bfloat16> bx(x.size()), bw(w.size());
        std::mt19937 grng(103);
        for (size_t i = 0; i < x.size(); ++i) {
            float v = (int(grng() % 15) - 7) / 128.f;
            x[i] = sycl::half(v); bx[i] = oneapi::mkl::bfloat16(v);
        }
        for (size_t i = 0; i < w.size(); ++i) {
            float v = (int(grng() % 15) - 7) / 512.f;
            w[i] = sycl::half(v); bw[i] = oneapi::mkl::bfloat16(v);
        }
        auto* hx = f.upload(x); auto* hw = f.upload(w);
        auto* dx = f.upload(bx); auto* dw = f.upload(bw);
        auto* y = f.alloc<float>((size_t)rows * n);
        auto* by = f.alloc<float>((size_t)rows * n);
        auto bcall = [&] {
            auto ev = oneapi::mkl::blas::column_major::gemm(q,
                oneapi::mkl::transpose::trans, oneapi::mkl::transpose::nontrans,
                n, rows, k, 1.f, dw, k, dx, k, 0.f, by, n);
            strata::kprof::note(&q, ev, "oneMKL.BF16");
        };
        auto hcall = [&] {
            auto ev = oneapi::mkl::blas::column_major::gemm(q,
                oneapi::mkl::transpose::trans, oneapi::mkl::transpose::nontrans,
                n, rows, k, 1.f, hw, k, hx, k, 0.f, y, n);
            strata::kprof::note(&q, ev, "oneMKL.FP16");
        };
        bcall(); hcall(); q.wait_and_throw();
        compare(f.read(by, (size_t)rows * n), f.read(y, (size_t)rows * n),
                0.000001f, 0.000001f, "hc_gemm_N" + std::to_string(n));
        paired(q, card, "hc_gemm_T" + std::to_string(rows) + "_N" +
                std::to_string(n) + "_K" + std::to_string(k),
               bcall, hcall, warmup, repeats);
    }
}
} // namespace fnx
