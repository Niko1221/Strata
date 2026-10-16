// src/nemotron_h/nh_cpu.cpp - Nemotron-H's routed experts on the CPU, through ggml-cpu (see the header).
#include "strata/nemotron_h/cpu_experts.hpp"

#include "ggml.h"
#include "ggml-cpu.h"

#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <cstring>
#include <mutex>
#include <thread>
#include <vector>

namespace strata::nemotron_h {
namespace {

const ggml_type_traits_cpu* traits(int type) { return ggml_get_type_traits_cpu((ggml_type) type); }

void init_once() {
    static std::once_flag once;
    std::call_once(once, [] { ggml_cpu_init(); });
}

constexpr int kRowsPerTask = 64;

}  // namespace

bool cpu_fmt(int up_type, int d_type, int64_t n_embd, int64_t n_ff, CpuFmt& f, std::string& err) {
    init_once();
    const ggml_type_traits_cpu* tu = traits(up_type);
    const ggml_type_traits_cpu* td = traits(d_type);
    if (tu == nullptr || tu->vec_dot == nullptr || td == nullptr || td->vec_dot == nullptr) {
        err = "nemotron_h CPU experts: ggml-cpu has no dot product for type " +
              std::to_string(tu && tu->vec_dot ? d_type : up_type);
        return false;
    }
    const ggml_type_traits_cpu* au = traits(tu->vec_dot_type);
    const ggml_type_traits_cpu* ad = traits(td->vec_dot_type);
    if (au == nullptr || au->from_float == nullptr || ad == nullptr || ad->from_float == nullptr) {
        err = "nemotron_h CPU experts: ggml-cpu cannot quantize an activation for this layer";
        return false;
    }
    if (n_embd % ggml_blck_size((ggml_type) up_type) || n_ff % ggml_blck_size((ggml_type) d_type) ||
        n_embd % ggml_blck_size(tu->vec_dot_type) || n_ff % ggml_blck_size(td->vec_dot_type)) {
        err = "nemotron_h CPU experts: the expert geometry is not whole blocks";
        return false;
    }
    f.up_type = up_type;
    f.d_type = d_type;
    f.up_act = (int) tu->vec_dot_type;
    f.d_act = (int) td->vec_dot_type;
    f.n_embd = n_embd;
    f.n_ff = n_ff;
    f.up_row = ggml_row_size((ggml_type) up_type, n_embd);
    f.d_row = ggml_row_size((ggml_type) d_type, n_ff);
    f.act_bytes = ggml_row_size(tu->vec_dot_type, n_embd);
    f.h_bytes = ggml_row_size(td->vec_dot_type, n_ff);
    return true;
}

struct CpuExperts::Impl {
    int n_threads = 1;
    std::vector<std::thread> workers;
    std::mutex mu;
    std::condition_variable cv_go, cv_done;
    uint64_t gen = 0;
    int running = 0;
    bool quit = false;
    // the current phase
    std::atomic<int> next{0};
    int n_tasks = 0;
    int phase = 0;
    // the call
    const CpuFmt* f = nullptr;
    const CpuJob* jobs = nullptr;
    int n_jobs = 0;
    int tasks_up = 0, tasks_down = 0;   // per job
    std::vector<uint8_t> act;           // [T][act_bytes]
    std::vector<int> tok_q;             // tokens quantized this call
    std::vector<float> h;               // [n_jobs][n_ff]
    std::vector<uint8_t> hq;            // [n_jobs][h_bytes]
    const float* x = nullptr;

    void task(int t) {
        const CpuFmt& F = *f;
        if (phase == 0) {   // the token activations
            const int tk = tok_q[(size_t) t];
            traits(F.up_act)->from_float(x + (size_t) tk * F.n_embd, act.data() + (size_t) tk * F.act_bytes, F.n_embd);
        } else if (phase == 1) {   // up rows + relu^2
            const int j = t / tasks_up, c = t % tasks_up;
            const CpuJob& J = jobs[j];
            const int r0 = c * kRowsPerTask, r1 = std::min<int>((int) F.n_ff, r0 + kRowsPerTask);
            const auto vd = traits(F.up_type)->vec_dot;
            const uint8_t* a = act.data() + (size_t) J.tok * F.act_bytes;
            float* hr = h.data() + (size_t) j * F.n_ff;
            for (int r = r0; r < r1; ++r) {
                float s = 0.f;
                vd((int) F.n_embd, &s, 0, J.up + (size_t) r * F.up_row, 0, a, 0, 1);
                const float u = s > 0.f ? s : 0.f;
                hr[r] = u * u;
            }
        } else if (phase == 2) {   // h -> the down activation
            traits(F.d_act)->from_float(h.data() + (size_t) t * F.n_ff, hq.data() + (size_t) t * F.h_bytes, F.n_ff);
        } else {   // down rows
            const int j = t / tasks_down, c = t % tasks_down;
            const CpuJob& J = jobs[j];
            const int r0 = c * kRowsPerTask, r1 = std::min<int>((int) F.n_embd, r0 + kRowsPerTask);
            const auto vd = traits(F.d_type)->vec_dot;
            const uint8_t* a = hq.data() + (size_t) j * F.h_bytes;
            for (int r = r0; r < r1; ++r) {
                float s = 0.f;
                vd((int) F.n_ff, &s, 0, J.down + (size_t) r * F.d_row, 0, a, 0, 1);
                J.out[r] = s;
            }
        }
    }
    void drain() {
        for (;;) {
            const int t = next.fetch_add(1, std::memory_order_relaxed);
            if (t >= n_tasks) return;
            task(t);
        }
    }
    void worker() {
        uint64_t seen = 0;
        for (;;) {
            {
                std::unique_lock<std::mutex> lk(mu);
                cv_go.wait(lk, [&] { return quit || gen != seen; });
                if (quit) return;
                seen = gen;
            }
            drain();
            {
                std::lock_guard<std::mutex> lk(mu);
                if (--running == 0) cv_done.notify_one();
            }
        }
    }
    void run_phase(int ph, int n) {
        if (n <= 0) return;
        phase = ph;
        n_tasks = n;
        next.store(0, std::memory_order_relaxed);
        const int helpers = std::min<int>((int) workers.size(), n - 1);
        if (helpers <= 0) {
            drain();
            return;
        }
        {
            std::lock_guard<std::mutex> lk(mu);
            running = (int) workers.size();
            ++gen;
        }
        cv_go.notify_all();
        drain();
        std::unique_lock<std::mutex> lk(mu);
        cv_done.wait(lk, [&] { return running == 0; });
    }
};

CpuExperts::CpuExperts(int threads) : p_(std::make_unique<Impl>()) {
    init_once();
    if (threads <= 0) threads = (int) std::max(1u, std::thread::hardware_concurrency());
    p_->n_threads = threads;
    for (int i = 1; i < threads; ++i) p_->workers.emplace_back([this] { p_->worker(); });
}

CpuExperts::~CpuExperts() {
    {
        std::lock_guard<std::mutex> lk(p_->mu);
        p_->quit = true;
    }
    p_->cv_go.notify_all();
    for (auto& t : p_->workers) t.join();
}

int CpuExperts::threads() const { return p_->n_threads; }

void CpuExperts::run(const CpuFmt& f, const float* x, int T, const CpuJob* jobs, int n) {
    if (n <= 0) return;
    Impl& P = *p_;
    P.f = &f;
    P.x = x;
    P.jobs = jobs;
    P.n_jobs = n;
    P.tasks_up = (int) ((f.n_ff + kRowsPerTask - 1) / kRowsPerTask);
    P.tasks_down = (int) ((f.n_embd + kRowsPerTask - 1) / kRowsPerTask);
    if (P.act.size() < (size_t) T * f.act_bytes) P.act.resize((size_t) T * f.act_bytes);
    if (P.h.size() < (size_t) n * f.n_ff) P.h.resize((size_t) n * f.n_ff);
    if (P.hq.size() < (size_t) n * f.h_bytes) P.hq.resize((size_t) n * f.h_bytes);
    // the tokens the jobs read, each quantized once
    std::vector<char> seen((size_t) T, 0);
    P.tok_q.clear();
    for (int j = 0; j < n; ++j)
        if (!seen[(size_t) jobs[j].tok]) {
            seen[(size_t) jobs[j].tok] = 1;
            P.tok_q.push_back(jobs[j].tok);
        }
    P.run_phase(0, (int) P.tok_q.size());
    P.run_phase(1, n * P.tasks_up);
    P.run_phase(2, n);
    P.run_phase(3, n * P.tasks_down);
}

}  // namespace strata::nemotron_h
