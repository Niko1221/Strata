#include "native_moe.hpp"
#include "strata/kernels/iq_dequant_expert.hpp"
#include "strata/sycl_queue.hpp"
#include "sycl_kernel_bench.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <iostream>
#include <memory>
#include <random>
#include <stdexcept>
#include <thread>
#include <unordered_map>
#include <vector>

namespace fnx::native_prefill {
void gather_rows16(const uint16_t*, const int32_t*, uint16_t*, int64_t, int64_t, void*);
}

namespace {
namespace i8 = strata::prefill::i8;

void check_down_tail(int card, int nf, int type, const std::vector<int>& counts) {
    std::cout << "BEGIN down card=" << card << " ff=" << nf << " type=" << type << std::endl;
    dpct::select_device(card);
    sycl::queue q(dpct::get_current_device(),
        sycl::property_list{sycl::property::queue::in_order{},
                            sycl::property::queue::enable_profiling{}});
    constexpr int ne = 2560;
    constexpr int iq4[] = {-127, -104, -83, -65, -49, -35, -22, -10,
                          1, 13, 25, 38, 53, 69, 89, 113};
    i8::Layout lo;
    i8::make_layout(lo, counts.data(), counts.size());
    const int nkb = nf / 32, rows = lo.row0[lo.n];
    const int block = type == 20 ? 32 : 64;
    std::vector<int8_t> hx(i8::xq_bytes(lo, nf), 0);
    std::vector<float> hd(i8::dx_bytes(lo, nf) / sizeof(float), 0.f);
    for (int e = 0; e < lo.n; ++e)
        for (int r = 0; r < counts[e]; ++r)
            for (int kb = 0; kb < nkb; ++kb) {
                const size_t tile = (size_t)(lo.tile0[e] + r / 8) * nkb + kb;
                hd[tile * 8 + r % 8] = (1 + r % 4) / 128.f;
                for (int k = 0; k < 32; ++k)
                    hx[tile * 256 + r % 8 * 32 + k] =
                        int8_t((e * 19 + r * 7 + kb * 13 + k * 3) % 255 - 127);
            }
    std::vector<void*> allocations;
    auto alloc = [&](size_t bytes) {
        void* p = sycl::malloc_device(bytes, q);
        if (!p) throw std::bad_alloc();
        allocations.push_back(p);
        return p;
    };
    auto upload = [&](void* dst, const void* src, size_t bytes) {
        void* pinned = sycl::malloc_host(bytes, q);
        if (!pinned) throw std::bad_alloc();
        std::memcpy(pinned, src, bytes);
        try {
            q.memcpy(dst, pinned, bytes).wait_and_throw();
        } catch (...) {
            sycl::free(pinned, q);
            throw;
        }
        sycl::free(pinned, q);
    };
    try {
        auto* dx = static_cast<int8_t*>(alloc(hx.size()));
        auto* dd = static_cast<float*>(alloc(hd.size() * sizeof(float)));
        auto* dy = static_cast<float*>(alloc((size_t)rows * ne * sizeof(float)));
        upload(dx, hx.data(), hx.size());
        upload(dd, hd.data(), hd.size() * sizeof(float));
        std::array<const uint8_t*, i8::kMaxGroup> blobs{};
        std::vector<float> expected((size_t)rows * ne, 0.f);
        std::mt19937 rng(1701 + nf + type);
        for (int e = 0; e < lo.n; ++e) {
            // Include an aligned section offset and exact matrix end, not a padded blob.
            constexpr size_t offset = 4;
            std::vector<uint8_t> w(offset + (size_t)ne * (nf / block) * 18);
            for (auto& b : w) b = uint8_t(rng());
            for (int n = 0; n < ne; ++n)
                for (int b = 0; b < nf / block; ++b) {
                    const uint16_t scale = sycl::bit_cast<uint16_t>(
                        sycl::half((1 + (b + n) % 3) / 1024.f));
                    std::memcpy(w.data() + offset + ((size_t)n * (nf / block) + b) * 18, &scale, 2);
                }
            auto* dw = static_cast<uint8_t*>(alloc(w.size()));
            upload(dw, w.data(), w.size());
            blobs[e] = dw;
            for (int r = 0; r < counts[e]; ++r)
                for (int n = 0; n < ne; ++n)
                    for (int kb = 0; kb < nkb; ++kb) {
                        const auto* b = w.data() + offset +
                            ((size_t)n * (nf / block) + kb * 32 / block) * 18;
                        uint16_t bits;
                        std::memcpy(&bits, b, 2);
                        const float sw = sycl::bit_cast<sycl::half>(bits);
                        const size_t tile = (size_t)(lo.tile0[e] + r / 8) * nkb + kb;
                        int dot = 0;
                        for (int k = 0; k < 32; ++k) {
                            int v;
                            if (type == 20)
                                v = iq4[(b[2 + k % 16] >> (k / 16 * 4)) & 15];
                            else {
                                const int kk = (kb % 2) * 32 + k;
                                v = ((b[2 + kk / 4] >> (kk % 4 * 2)) & 3) - 1;
                            }
                            dot += int(hx[tile * 256 + r % 8 * 32 + k]) * v;
                        }
                        expected[(size_t)(lo.row0[e] + r) * ne + n] +=
                            float(dot) * (sw * hd[tile * 8 + r % 8]);
                    }
        }
        auto run = [&] {
            if (!i8::down(type, blobs.data(), 4, ne, lo, nf, dx, dd, dy, &q))
                throw std::runtime_error("Down tail shape rejected");
        };
        std::cout << "RUN down inputs ready" << std::endl;
        run();
        std::vector<float> actual(expected.size());
        q.memcpy(actual.data(), dy, actual.size() * sizeof(float)).wait_and_throw();
        float worst = 0;
        for (size_t i = 0; i < actual.size(); ++i) {
            const float err = std::fabs(actual[i] - expected[i]);
            if (!std::isfinite(actual[i]) || err > 1e-5f + 1e-6f * std::fabs(expected[i]))
                throw std::runtime_error("WG/generic down disagrees with independent CPU decoded dot products");
            worst = std::max(worst, err);
        }
        for (int i = 0; i < 5; ++i) run();
        q.wait_and_throw();
        const auto start = std::chrono::steady_clock::now();
        for (int i = 0; i < 30; ++i) run();
        q.wait_and_throw();
        const double ms = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - start).count() / 30;
        std::cout << "PASS down card=" << card << " ff=" << nf << " type=" << type
                  << " rows=" << rows << " max_error=" << worst << " mean_ms=" << ms << std::endl;
    } catch (...) {
        q.wait_and_throw();
        for (void* p : allocations) sycl::free(p, q);
        throw;
    }
    for (void* p : allocations) sycl::free(p, q);
}

void check_card(int card, int nf, bool unaligned = false, bool matched = false,
                bool model_group = false, bool legacy_bench = true,
                bool full_model = false, bool staging = false,
                int gu_type = 21, int down_type = 20,
                bool repeated_sources = false, bool source_tail = false,
                bool skewed = false) {
    dpct::select_device(matched && dpct::device_count() == 1 ? 0 : card);
    sycl::device device = dpct::get_current_device();
    // Official table/workspace caches are per-device: keep one context per device.
    static std::unordered_map<sycl::device, sycl::context> contexts;
    auto context = contexts.find(device);
    if (context == contexts.end())
        context = contexts.emplace(device, sycl::context(device)).first;
    sycl::queue q(context->second, device,
        sycl::property_list{sycl::property::queue::in_order{},
                            sycl::property::queue::enable_profiling{}});
    std::unique_ptr<sycl::queue> copy;
    if (staging)
        copy = std::make_unique<sycl::queue>(context->second, device,
            sycl::property_list{sycl::property::queue::in_order{}});
    constexpr int ne = 2560;
    std::array<int, fnx::NEXP> cnt{};
    cnt[0] = 1; cnt[2] = 7; cnt[3] = 35; cnt[17] = 13; cnt[31] = 16; cnt[511] = 1;
    if (model_group) {
        cnt.fill(0);
        for (int e = 0; e < (full_model ? fnx::NEXP : 32); ++e) cnt[e] = 80;
    }
    if (skewed) {
        cnt.fill(0);
        cnt[0] = 1; cnt[2] = 7; cnt[3] = 64; cnt[17] = 128; cnt[31] = 255;
        cnt[32] = 256; cnt[33] = 257; cnt[34] = 512; cnt[63] = 1025;
        cnt[127] = 2049; cnt[511] = 3702;
    }
    int nr = 0;
    for (int rows : cnt) nr += rows;
    const int input_rows = repeated_sources ? nr / fnx::TOPK - int(source_tail) : nr;
    std::vector<void*> allocations;
    std::vector<std::vector<uint8_t>> staged_gu(fnx::NEXP), staged_dn(fnx::NEXP);
    auto alloc = [&](size_t bytes) {
        void* p = sycl::malloc_device(bytes, q);
        if (!p) throw std::bad_alloc();
        allocations.push_back(p);
        return p;
    };
    try {
        std::array<int, fnx::NEXP> row0{};
        for (int e = 1; e < fnx::NEXP; ++e) row0[e] = row0[e - 1] + cnt[e - 1];
        i8::Layout lo;
        i8::make_layout(lo, cnt.data(), i8::kMaxGroup);
        const int gu_block_bytes = gu_type == 23 ? 136 : 110;
        const int down_block_bytes = down_type == 8 ? 34 : 18;
        const size_t gate_bytes = (size_t)nf * (ne / 256) * gu_block_bytes;
        const size_t down_bytes = (size_t)ne * (nf / 32) * down_block_bytes;
        std::array<const char*, fnx::NEXP> gp{}, dp{};
        std::array<const uint8_t*, fnx::NEXP> gates{}, downs{};
        std::mt19937 rng(42);
        for (int e = 0; e < fnx::NEXP; ++e) {
            if (!cnt[e]) continue;
            std::vector<uint8_t> gu(2 * gate_bytes), down(down_bytes);
            for (auto& b : gu) b = uint8_t(rng());
            for (auto& b : down) b = uint8_t(rng());
            const uint16_t scale = sycl::bit_cast<uint16_t>(sycl::half(0.0004f));
            for (size_t b = 0; b < gu.size(); b += gu_block_bytes)
                std::memcpy(gu.data() + b, &scale, 2);
            for (size_t b = 0; b < down.size(); b += down_block_bytes)
                std::memcpy(down.data() + b, &scale, 2);
            auto* g = static_cast<uint8_t*>(alloc(gu.size()));
            auto* d = static_cast<uint8_t*>(alloc(down.size()));
            q.memcpy(g, gu.data(), gu.size()).wait_and_throw();
            q.memcpy(d, down.data(), down.size()).wait_and_throw();
            gp[e] = reinterpret_cast<const char*>(g);
            dp[e] = reinterpret_cast<const char*>(d);
            gates[e] = g; downs[e] = d;
        }
        std::vector<uint16_t> hx((size_t)nr * ne), gathered(hx.size());
        for (auto& a : hx)
            a = sycl::bit_cast<uint16_t>(sycl::half((int(rng() % 201) - 100) / 128.f));
        std::vector<fnx::BigJob> jobs;
        std::vector<int> dst(nr), source(nr), inverse_dst(nr);
        int r0 = 0;
        for (int e = 0; e < fnx::NEXP; ++e) {
            for (int r = 0; r < cnt[e]; r += fnx::PF_JOB) {
                fnx::BigJob job{};
                job.w = e; job.n = std::min(fnx::PF_JOB, cnt[e] - r);
                for (int k = 0; k < job.n; ++k) {
                    const int row = r0 + r + k;
                    job.y[k] = row * 17 % input_rows;
                    job.out[k] = row * 11 % nr;
                    dst[row] = job.out[k];
                    source[row] = job.y[k];
                    inverse_dst[job.out[k]] = row;
                    std::copy_n(hx.data() + (size_t)job.y[k] * ne, ne,
                                 gathered.data() + (size_t)row * ne);
                }
                jobs.push_back(job);
            }
            r0 += cnt[e];
        }
        auto* dc = static_cast<int*>(alloc(sizeof(cnt)));
        auto* dg = static_cast<const char**>(alloc(sizeof(gp)));
        auto* dd = static_cast<const char**>(alloc(sizeof(dp)));
        auto* dj = static_cast<fnx::BigJob*>(alloc(jobs.size() * sizeof(jobs[0])));
        auto* djd = static_cast<fnx::BigJob*>(alloc(jobs.size() * sizeof(jobs[0])));
        auto* nj = static_cast<int*>(alloc(sizeof(int)));
        auto* dx = static_cast<uint16_t*>(alloc(hx.size() * 2 + (unaligned ? 2 : 0))) + (unaligned ? 1 : 0);
        auto* xs = static_cast<uint16_t*>(alloc(hx.size() * 2));
        auto* got = static_cast<float*>(alloc((size_t)nr * ne * 4 + (unaligned ? 4 : 0))) + (unaligned ? 1 : 0);
        auto* ref = static_cast<float*>(alloc((size_t)nr * ne * 4));
        auto* gu = static_cast<float*>(alloc((size_t)nr * 2 * nf * 4));
        auto* wh = static_cast<uint16_t*>(alloc((size_t)2 * nf * ne * 2));
        auto* wd = static_cast<uint16_t*>(alloc((size_t)nf * ne * 2));
        auto* hh = static_cast<uint16_t*>(alloc((size_t)nr * nf * 2));
        q.memcpy(dc, cnt.data(), sizeof(cnt));
        q.memcpy(dg, gp.data(), sizeof(gp));
        q.memcpy(dd, dp.data(), sizeof(dp));
        q.memcpy(dj, jobs.data(), jobs.size() * sizeof(jobs[0]));
        std::vector<fnx::BigJob> down_jobs = jobs;
        for (auto& job : down_jobs)
            for (int k = 0; k < job.n; ++k) job.y[k] = job.out[k];
        const int njobs = jobs.size();
        q.memcpy(djd, down_jobs.data(), down_jobs.size() * sizeof(down_jobs[0]));
        q.memcpy(nj, &njobs, sizeof(njobs));
        q.memcpy(dx, hx.data(), hx.size() * 2);
        q.memcpy(xs, gathered.data(), gathered.size() * 2);
        q.wait_and_throw();
        fnx::DW gw, dw;
        gw.type = gu_type; gw.n_in = ne; gw.n_out = 2 * nf;
        gw.row_bytes = (ne / 256) * gu_block_bytes; gw.ep = dg;
        dw.type = down_type; dw.n_in = nf; dw.n_out = ne;
        dw.row_bytes = (nf / 32) * down_block_bytes; dw.ep = dd;
        fnx::NativeMoe adapter(&q, nr, ne, nf);
        i8::Workspace ws;
        if (!i8::available(&q) || !i8::workspace(lo, ne, nf, ws, &q))
            throw std::runtime_error("PP2 DPAS kernels unavailable on selected B70");
        strata::prefill::Gemm gemm;
        std::string error;
        if (!gemm.init_external(&q, wh, (size_t)2 * nf * ne, nullptr, 0, error))
            throw std::runtime_error(error);
        auto pp2_i8 = [&] {
            for (int first = 0; first < fnx::NEXP; first += i8::kMaxGroup) {
                i8::Layout group;
                i8::make_layout(group, cnt.data() + first, i8::kMaxGroup);
                if (!group.row0[group.n]) continue;
                i8::quantize_rows(xs + (size_t)row0[first] * ne, group, ne, ws.xq, ws.dx, &q);
                if (!i8::gate_up(21, gates.data() + first, gate_bytes, nf, group, ne,
                                 ws.xq, ws.dx, gu, &q))
                    throw std::runtime_error("PP2 reference gate/up refused");
                i8::swiglu_quantize(gu, group, nf, ws.hq, ws.dh, &q);
                if (!i8::down(20, downs.data() + first, 0, ne, group, nf, ws.hq, ws.dh,
                              ref + (size_t)row0[first] * ne, &q))
                    throw std::runtime_error("PP2 reference down refused");
            }
        };
        for (bool int8 : (matched ? std::vector<bool>{true} : std::vector<bool>{false, true})) {
            if (int8 && down_type != 8) pp2_i8();
            else {
                for (int e = 0; e < fnx::NEXP; ++e) {
                    const int r = row0[e], rows = cnt[e];
                    if (!rows) continue;
                    strata::kernels::iq_dequant_expert_f16(
                        gu_type, down_type, gates[e], gates[e] + gate_bytes, downs[e],
                        nf, ne, wh, wd, &q);
                    gemm.f16(xs + (size_t)r * ne, wh,
                              gu + (size_t)r * 2 * nf, rows, 2 * nf, ne);
                    if (down_type == 8) {
                        const auto* g = gu + (size_t)r * 2 * nf;
                        q.parallel_for(sycl::range<1>((size_t)rows * nf), [=](sycl::id<1> id) {
                            hh[id] = fnx::native_swiglu_f16(g[2 * id[0]], g[2 * id[0] + 1]);
                        });
                        gemm.f16(hh, wd, ref + (size_t)r * ne, rows, ne, nf);
                        continue;
                    }
                    std::vector<float> hg((size_t)rows * 2 * nf);
                    q.memcpy(hg.data(), gu + (size_t)r * 2 * nf, hg.size() * 4).wait_and_throw();
                    std::vector<uint16_t> hidden((size_t)rows * nf);
                    for (size_t j = 0; j < hidden.size(); ++j) {
                        const float a = hg[2 * j], b = hg[2 * j + 1];
                        const float v = a / (1.f + std::exp(-a)) * b;
                        hidden[j] = sycl::bit_cast<uint16_t>(sycl::half(
                            v > 65504.f ? 65504.f : (v < -65504.f ? -65504.f : v)));
                    }
                    q.memcpy(hh, hidden.data(), hidden.size() * 2).wait_and_throw();
                    gemm.f16(hh, wd, ref + (size_t)r * ne, rows, ne, nf);
                }
            }
            sycl::event weights_ready;
            if (staging) {
                adapter.run(gw, 2 * gate_bytes, dw, down_bytes, dc, dj, dx, got, int8, nr,
                            nullptr, input_rows);
                for (int e = 0; e < fnx::NEXP; ++e) {
                    if (!cnt[e]) continue;
                    staged_gu[e].resize(2 * gate_bytes);
                    staged_dn[e].resize(down_bytes);
                    q.memcpy(staged_gu[e].data(), gp[e], staged_gu[e].size());
                    q.memcpy(staged_dn[e].data(), dp[e], staged_dn[e].size());
                    q.memset(const_cast<char*>(gp[e]), 0, staged_gu[e].size());
                    q.memset(const_cast<char*>(dp[e]), 0, staged_dn[e].size());
                }
                q.wait_and_throw();
                copy->submit([](sycl::handler& h) {
                    h.host_task([] { std::this_thread::sleep_for(std::chrono::milliseconds(50)); });
                });
                for (int e = 0; e < fnx::NEXP; ++e) {
                    if (!cnt[e]) continue;
                    copy->memcpy(const_cast<char*>(gp[e]), staged_gu[e].data(), staged_gu[e].size());
                    weights_ready = copy->memcpy(const_cast<char*>(dp[e]), staged_dn[e].data(),
                                                 staged_dn[e].size());
                }
            }
            adapter.run(gw, 2 * gate_bytes, dw, down_bytes, dc, dj, dx, got, int8, nr,
                        staging ? &weights_ready : nullptr, input_rows);
            std::vector<float> actual((size_t)nr * ne), expected(actual.size());
            q.memcpy(actual.data(), got, actual.size() * 4);
            q.memcpy(expected.data(), ref, expected.size() * 4).wait_and_throw();
            float worst = 0, scale = 0;
            for (int r = 0; r < nr; ++r)
                for (int k = 0; k < ne; ++k) {
                    const float a = actual[(size_t)dst[r] * ne + k];
                    const float b = expected[(size_t)r * ne + k];
                    if (!std::isfinite(a)) throw std::runtime_error("Nonfinite native MoE result");
                    worst = std::max(worst, std::fabs(a - b));
                    scale = std::max(scale, std::fabs(b));
                }
            if (worst > 2e-5f * scale + (int8 && down_type != 8 ? 1e-8f : 2e-6f))
                throw std::runtime_error("Native MoE disagrees with direct PP2 kernels: max_error=" +
                    std::to_string(worst) + " scale=" + std::to_string(scale));
            std::cout << "PASS card=" << card << " ff=" << nf << " int8=" << int8
                      << " gu_type=" << gu_type << " down_type=" << down_type
                      << " rows=" << nr
                      << " input_rows=" << input_rows
                      << " unaligned=" << unaligned << " staging=" << staging
                      << " max_error=" << worst << std::endl;
        }
        if (down_type == 8) {
            const char* configured = std::getenv("FNX_PF_MOE_GROUPED");
            const std::string saved = configured ? configured : "";
            setenv("FNX_PF_MOE_GROUPED", "0", 1);
            fnx::NativeMoe serial(&q, nr, ne, nf);
            if (configured) setenv("FNX_PF_MOE_GROUPED", saved.c_str(), 1);
            else unsetenv("FNX_PF_MOE_GROUPED");
            auto direct = [&] {
                serial.run(gw, 2 * gate_bytes, dw, down_bytes, dc, dj, dx, ref, true, nr);
            };
            auto native = [&] {
                adapter.run(gw, 2 * gate_bytes, dw, down_bytes, dc, dj, dx, got, true, nr);
            };
            kernel_bench::paired(q, card, "fallback_gu" + std::to_string(gu_type) +
                "_ff" + std::to_string(nf) + "_rows" + std::to_string(nr),
                direct, native, 3, 10);
            adapter.report_profile(card);
            q.wait_and_throw();
            for (void* p : allocations) sycl::free(p, q);
            return;
        }
        if (matched) {
            auto* dsrc = static_cast<int*>(alloc((size_t)nr * sizeof(int)));
            auto* dinverse = static_cast<int*>(alloc((size_t)nr * sizeof(int)));
            auto* pp_scattered = static_cast<float*>(alloc((size_t)nr * ne * sizeof(float)));
            q.memcpy(dsrc, source.data(), (size_t)nr * sizeof(int));
            q.memcpy(dinverse, inverse_dst.data(), (size_t)nr * sizeof(int)).wait_and_throw();
            std::unique_ptr<fnx::NativeMoe> control;
            if (repeated_sources) {
                const char* configured = std::getenv("FNX_PF_MOE_DIRECT");
                const bool present = configured != nullptr;
                const std::string saved = configured ? configured : "";
                setenv("FNX_PF_MOE_DIRECT", "0", 1);
                control = std::make_unique<fnx::NativeMoe>(&q, nr, ne, nf);
                if (present) setenv("FNX_PF_MOE_DIRECT", saved.c_str(), 1);
                else unsetenv("FNX_PF_MOE_DIRECT");
            }
            auto direct = [&] {
                if (control) {
                    control->run(gw, 2 * gate_bytes, dw, down_bytes, dc, dj, dx,
                        pp_scattered, true, nr, nullptr, input_rows);
                    return;
                }
                fnx::native_prefill::gather_rows16(dx, dsrc, xs, nr, ne, &q);
                pp2_i8();
                // The PP vector row-copy primitive permutes raw float bits here.
                // This matches scatter's workload, not PP's fused weighted combine.
                fnx::native_prefill::gather_rows16(
                    reinterpret_cast<const uint16_t*>(ref), dinverse,
                    reinterpret_cast<uint16_t*>(pp_scattered), nr, 2 * ne, &q);
            };
            auto native = [&] {
                adapter.run(gw, 2 * gate_bytes, dw, down_bytes, dc, dj, dx, got, true, nr,
                            nullptr, input_rows);
            };
            direct(); native(); q.wait_and_throw();
            std::vector<float> a((size_t)nr * ne), b(a.size());
            q.memcpy(a.data(), pp_scattered, a.size() * sizeof(float));
            q.memcpy(b.data(), got, b.size() * sizeof(float)).wait_and_throw();
            kernel_bench::compare(a, b, 0.f, 0.f, "moe_full_row_permutation");
            kernel_bench::paired(q, card, "moe_ff" + std::to_string(nf) +
                "_rows" + std::to_string(nr) + "_input" + std::to_string(input_rows) +
                "_unaligned" + std::to_string(unaligned), direct, native, 5, 20);
            adapter.report_profile(card);
            q.wait_and_throw();
            for (void* p : allocations) sycl::free(p, q);
            return;
        }
        auto time = [&](auto run) {
            run(); q.wait_and_throw();
            const auto start = std::chrono::steady_clock::now();
            for (int i = 0; i < 3; ++i) run();
            q.wait_and_throw();
            return std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - start).count() / 3;
        };
        const double pp2_ms = time(pp2_i8);
        const double adapter_ms = time([&] {
            adapter.run(gw, 2 * gate_bytes, dw, down_bytes, dc, dj, dx, got, true, nr);
        });
        const double migrated_ms = unaligned || !legacy_bench ? 0 : time([&] {
            fnx::moe_gate_up_mma(gw, 2 * gate_bytes, dj, nj, njobs, dx, gu, &q);
            fnx::swiglu_h16(gu, nf, nr, hh, &q);
            fnx::moe_down_mma(dw, down_bytes, djd, nj, njobs, hh, got, &q);
        });
        std::cout << "BENCH card=" << card << " ff=" << nf << " rows=" << nr
                  << " pp2_i8_ms=" << pp2_ms << " adapter_i8_ms=" << adapter_ms
                  << " migrated_mma_ms=" << (unaligned || !legacy_bench ? std::string("skipped") : std::to_string(migrated_ms))
                  << " (adapter includes gather/scatter and host metadata)" << std::endl;
        q.wait_and_throw();
    } catch (...) {
        if (copy) copy->wait_and_throw();
        q.wait_and_throw();
        for (void* p : allocations) sycl::free(p, q);
        throw;
    }
    for (void* p : allocations) sycl::free(p, q);
}
}

int main(int argc, char** argv) {
    try {
        std::cout << "BEGIN two-card native MoE tests" << std::endl;
        if (argc == 3 && std::string(argv[1]) == "--direct-card") {
            const int card = std::stoi(argv[2]);
            if (card < 0 || card >= dpct::device_count() || card > 1)
                throw std::runtime_error("Select only GPU0/1");
            setenv("STRATA_KPROF", "1", 1);
            setenv("FNX_PF_MOE_DEVICE", "1", 1);
            setenv("FNX_PF_MOE_DIRECT", "1", 1);
            for (int nf : {320, 640}) {
                check_card(card, nf, false, true, false, false, false, false, 21, 20, true);
                check_card(card, nf, false, true, true, false, false, true, 21, 20, true);
                check_card(card, nf, true, true, true, false, false, false, 21, 20, true);
                for (bool tail : {false, true})
                    check_card(card, nf, false, true, true, false, true, false, 21, 20, true, tail);
            }
            std::cout << "PASS direct quantized input card=" << card << std::endl;
            return 0;
        }
        if (argc == 3 && (std::string(argv[1]) == "--fallback-card" ||
                          std::string(argv[1]) == "--fallback-full-card" ||
                          std::string(argv[1]) == "--fallback-skew-card")) {
            const int card = std::stoi(argv[2]);
            if (card < 0 || card >= dpct::device_count() || card > 1)
                throw std::runtime_error("Select only GPU0/1");
            setenv("STRATA_KPROF", "1", 1);
            setenv("FNX_PF_MOE_GROUPED", "1", 1);
            const bool full = std::string(argv[1]) == "--fallback-full-card";
            const bool skewed = std::string(argv[1]) == "--fallback-skew-card";
            for (int type : {21, 23})
                for (int nf : {320, 640})
                    if (skewed)
                        check_card(card, nf, true, false, false, false, false, true,
                                   type, 8, false, false, true);
                    else if (full)
                        check_card(card, nf, false, false, true, false, true, false, type, 8);
                    else
                        for (bool dense : {false, true})
                            check_card(card, nf, false, false, dense, false, false, !dense, type, 8);
            std::cout << "PASS grouped FP16 fallback card=" << card << std::endl;
            return 0;
        }
        if (argc == 3 && std::string(argv[1]) == "--staging-card") {
            const int card = std::stoi(argv[2]);
            if (card < 0 || card >= dpct::device_count() || card > 1)
                throw std::runtime_error("Select only GPU0/1");
            for (int nf : {320, 640})
                check_card(card, nf, false, false, false, false, false, true);
            std::cout << "PASS staged-weight dependency card=" << card << std::endl;
            return 0;
        }
        if (argc == 3 && (std::string(argv[1]) == "--matched-card" ||
                          std::string(argv[1]) == "--matched-full-card")) {
            int card = std::stoi(argv[2]);
            if (card < 0 || card > 1 || dpct::device_count() < 1 || dpct::device_count() > 2)
                throw std::runtime_error("Select only GPU0/1");
            setenv("STRATA_KPROF", "1", 1);
            if (std::string(argv[1]) == "--matched-full-card") {
                for (int nf : {320, 640})
                    check_card(card, nf, false, true, true, false, true);
            } else {
                for (bool model_group : {false, true})
                    for (int nf : {320, 640})
                        check_card(card, nf, false, true, model_group);
            }
            std::cout << "PASS matched MoE kernel benchmark card=" << card << std::endl;
            return 0;
        }
        const bool single = argc == 2 && std::string(argv[1]) == "--single-card";
        if (dpct::device_count() != (single ? 1 : 2))
            throw std::runtime_error("Select two B70 GPUs, or one with --single-card");
        if (argc == 2 && std::string(argv[1]) == "--down") {
            for (int card = 0; card < 2; ++card) {
                for (int nf : {32, 96, 320, 640})
                    check_down_tail(card, nf, 20, {1, 0, 7, 65});
                for (int nf : {64, 192, 320, 640})
                    check_down_tail(card, nf, 42, {1, 0, 7, 65});
                for (const auto& counts : {std::vector<int>{17, 31},
                                          std::vector<int>{65, 127},
                                          std::vector<int>{257, 193},
                                          std::vector<int>{1, 0, 7, 1}})
                    check_down_tail(card, 320, 20, counts);
                check_down_tail(card, 320, 42, {65, 127});
            }
            return 0;
        }
        for (int card = 0; card < (single ? 1 : 2); ++card) {
            std::cout << "RUN FP16 saturation card=" << card << std::endl;
            dpct::select_device(card);
            sycl::device device = dpct::get_current_device();
            sycl::queue q(sycl::context(device), device,
                sycl::property_list{sycl::property::queue::in_order{},
                                    sycl::property::queue::enable_profiling{}});
            auto* values = sycl::malloc_device<uint16_t>(3, q);
            if (!values) throw std::bad_alloc();
            q.single_task([=] {
                values[0] = fnx::native_swiglu_f16(65504.f, 65504.f);
                values[1] = fnx::native_swiglu_f16(65504.f, -65504.f);
                values[2] = fnx::native_swiglu_f16(NAN, 1.f);
            }).wait_and_throw();
            uint16_t host[3];
            q.memcpy(host, values, sizeof(host)).wait_and_throw();
            const float positive = sycl::bit_cast<sycl::half>(host[0]);
            const float negative = sycl::bit_cast<sycl::half>(host[1]);
            const float nan = sycl::bit_cast<sycl::half>(host[2]);
            sycl::free(values, q);
            if (positive != 65504.f || negative != -65504.f || !std::isnan(nan))
                throw std::runtime_error("FP16 SwiGLU saturation/NaN mismatch");
            std::cout << "PASS card=" << card << " FP16 saturation and NaN" << std::endl;
            for (int nf : {320, 640}) check_card(card, nf, false, false, false, !single);
            check_card(card, 320, true);
        }
    } catch (const std::exception& e) {
        std::cerr << "FAIL " << e.what() << std::endl;
        return 1;
    }
}
