// The production Verifier -> MtpDrafter sampling boundary, without model weights.
// Capture the same staging + coupled/prob sampler kernels that MTP's round/step graphs use.
// Mutate and free the verifier's source bias before replay: the old shallow borrow picks the wrong token.
// Also exercise the real batch verifier's masked-mu offers, NULL tails and one-shot slot borrowing.
#include "strata/core/mtp.hpp"
#include "strata/core/verify.hpp"
#include "strata/core/coupled_draft.hpp"
#include "strata/core/on_device.hpp"
#include "strata/core/spec_prob.hpp"

#include <cuda_runtime.h>

#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace {
void require(bool ok, const char* what) {
    if (!ok) { std::fprintf(stderr, "FAIL: %s\n", what); std::exit(1); }
}
void check(cudaError_t e, const char* what) {
    if (e != cudaSuccess) {
        std::fprintf(stderr, "FAIL: %s: %s\n", what, cudaGetErrorString(e));
        std::exit(1);
    }
}
constexpr int kVocab = 8;
std::vector<float> bias_for(int token) {
    std::vector<float> bias(kVocab, 0.0f);
    bias[(size_t) token] = 50.0f;
    return bias;
}
}  // namespace

namespace strata::core {
// Only supplies tiny dimensions and inspects ownership; all copies, request updates, staging and draws
// run through production implementations. No inference/model buffers are needed at this boundary.
struct MtpSamplingFixture {
    static void init(MtpDrafter& d, int device, std::string& err) {
        d.device_ = device;
        d.n_vocab_ = kVocab;
        d.max_t_ = 2;
        check(cudaStreamCreateWithFlags(&d.cs_, cudaStreamNonBlocking), "drafter stream");
        require(d.setup_coupled(err), err.c_str());
        require(d.coupled_ok_, "coupled sampler configured");
    }
    static void init_source(Verifier& v, int device, const ModelGeometry& g, Verifier* next = nullptr) {
        v.device_ = device;
        v.n_vocab_ = kVocab;
        v.g_ = &g;
        v.le_ = next ? 0 : g.n_layers;
        v.next_ = next;
        v.slot_sp_.resize(1);
        v.slot_bias_dev_.resize(1, nullptr);
    }
    static void record(MtpDrafter& d, float* logits, int32_t* step, int32_t* out, float* prob) {
        using namespace strata::kernels;
        coupled_draft_stage(d.m_cparams_, d.m_chist_, d.cparams_, d.cring_, kCoupledHistCap, d.cs_);
        if (d.h_q_)
            spec_draft_sample(logits, kVocab, nullptr, nullptr, kVocab, d.cparams_, d.cring_, kCoupledHistCap,
                              0, step, d.cscratch_, out, prob, d.m_q_, 0, 1.0f, d.cs_);
        else
            coupled_draft_sample(logits, kVocab, nullptr, nullptr, kVocab, d.cparams_, d.cring_, kCoupledHistCap,
                                 0, step, d.cscratch_, out, prob, d.cs_);
    }
    static void params(MtpDrafter& d, bool biased) {
        strata::kernels::SamplerParams device_params;
        check(cudaMemcpy(&device_params, d.cparams_, sizeof(device_params), cudaMemcpyDeviceToHost), "read params");
        require(d.h_cparams_->logit_bias == (biased ? d.cbias_ : nullptr), "host params own/clear bias");
        require(device_params.logit_bias == (biased ? d.cbias_ : nullptr), "device params own/clear bias");
        if (biased) {
            cudaPointerAttributes attr{};
            check(cudaPointerGetAttributes(&attr, device_params.logit_bias), "bias allocation device");
            require(attr.type == cudaMemoryTypeDevice && attr.device == d.device_, "bias belongs to drafter device");
        }
    }
    static const float* bias(const MtpDrafter& d) { return d.cbias_; }
    static const float* slot_bias(const Verifier& v) { return v.slot_bias_dev_[0]; }
    static bool sample(Verifier& v, const std::vector<int>& rows, const std::vector<int32_t>& tokens,
                       const std::vector<int64_t>& pos, const std::vector<float>& logits, std::string& err) {
        for (size_t t = 0; t < rows.size(); ++t) {
            v.last_rows_[t] = rows[t];
            v.last_tokens_[t] = tokens[t];
            v.last_pos_b_[t] = pos[t];
        }
        check(cudaMemcpyAsync(v.head_logits_, logits.data(), logits.size() * sizeof(float),
                              cudaMemcpyHostToDevice, v.cs_), "stage verdict logits");
        return v.sample_rows((int) rows.size(), err);
    }
    static void consumed(const Verifier& v) {
        for (size_t s = 0; s < v.slot_spec_q_.size(); ++s)
            require(v.slot_spec_q_[s] == nullptr && v.slot_spec_nq_[s] == 0 && !v.slot_spec_measure_[s],
                    "slot proposal pointer/count/measure consumed once");
    }
    static void verdicts(int device, const ModelGeometry& g, bool unavailable) {
        Verifier v;
        init_source(v, device, g);
        v.max_t_ = 5;
        v.slots_.resize(3, nullptr);
        v.slot_sp_.resize(3);
        v.slot_bias_dev_.resize(3, nullptr);
        v.slot_spec_q_.assign(3, nullptr);
        v.slot_spec_nq_.assign(3, 0);
        v.slot_spec_measure_.assign(3, false);
        check(cudaStreamCreateWithFlags(&v.cs_, cudaStreamNonBlocking), "verdict stream");
        check(cudaMalloc(&v.arena_, 5 * kVocab * sizeof(float)), "verdict logits");
        v.head_logits_ = (float*) v.arena_;
        check(cudaHostAlloc(&v.h_out_, 5 * sizeof(int32_t), cudaHostAllocMapped), "verdict mapped output");
        check(cudaHostGetDevicePointer((void**) &v.m_out_, v.h_out_, 0), "verdict device output");
        strata::kernels::SamplerParams sp;
        sp.top_k = 2;
        sp.top_p = 1.0f;
        sp.seed = 13;
        for (int s = 0; s < 3; ++s) v.set_slot_sampling(s, sp);
        // Exact half/half p. mu(0)=1/2 removes token 0, so a NULL row must return 1,
        // not the ordinary-p draw (0 at seed 13, position 1).
        const std::vector<float> logits = {
            0, 0, -INFINITY, -INFINITY, -INFINITY, -INFINITY, -INFINITY, -INFINITY,
            0, 0, -INFINITY, -INFINITY, -INFINITY, -INFINITY, -INFINITY, -INFINITY,
            -INFINITY, 0, -INFINITY, -INFINITY, -INFINITY, -INFINITY, -INFINITY, -INFINITY,
            0, -INFINITY, -INFINITY, -INFINITY, -INFINITY, -INFINITY, -INFINITY, -INFINITY,
            0, 0, -INFINITY, -INFINITY, -INFINITY, -INFINITY, -INFINITY, -INFINITY,
        };
        std::array<int32_t, 2 * kSpecQStride> accept_null, reject_bonus;
        std::array<int32_t, kSpecQStride> null;
        accept_null.fill(-1); reject_bonus.fill(-1); null.fill(-1);
        auto mu = [](int32_t* row, int id, float mass) {
            row[0] = id;
            std::memcpy(row + kSpecQEntries, &mass, sizeof(mass));
        };
        mu(accept_null.data(), 0, 0.25f);   // p(0)=1/2 >= mu(0): certain acceptance
        mu(accept_null.data() + kSpecQStride, 0, 0.5f);
        mu(reject_bonus.data(), 2, 0.25f);  // p(2)=0: certain rejection, residual is token 1
        mu(null.data(), 0, 0.5f);
        const std::vector<int> rows{0, 0, 1, 1, 2};
        const std::vector<int32_t> tokens{7, 0, 7, 2, 7};
        const std::vector<int64_t> pos{0, 1, 10, 11, 1};
        v.set_slot_spec_q(0, accept_null.data(), 2, true);
        v.set_slot_spec_q(1, reject_bonus.data(), 1, true);
        v.set_slot_spec_q(2, null.data(), 1, true);
        std::string err;
        const bool ok = sample(v, rows, tokens, pos, logits, err);
        consumed(v);
        if (unavailable) {
            require(!ok && err.find("proposal sampler unavailable") != std::string::npos,
                    "unsupported measure kernel fails instead of falling back to ordinary p");
            return;
        }
        require(ok, err.c_str());
        const int32_t want[] = {0, 1, 1, 0, 1};
        for (int t = 0; t < 5; ++t) {
            if (v.h_out_[t] != want[t])
                std::fprintf(stderr, "slot verdict row %d: want %d, got %d\n", t, want[t], v.h_out_[t]);
            require(v.h_out_[t] == want[t], "real batch sampler accepts/rejects offers and samples NULL, including T1");
        }
        require(v.spec_windows == 3, "all three slot groups use the proposal kernel");
        require(sample(v, rows, tokens, pos, logits, err), err.c_str());
        require(v.h_out_[1] == 0 && v.h_out_[4] == 0 && v.spec_windows == 3,
                "second window has plain-p bonus/T1 draws, not stale NULL proposals");
        // The default API is still raw q: its last row is a plain bonus even with n_q==T.
        v.set_slot_spec_q(0, accept_null.data(), 2);
        require(sample(v, rows, tokens, pos, logits, err), err.c_str());
        require(v.h_out_[1] == 0, "legacy raw-q tail remains ordinary p");
        consumed(v);
    }
};
}  // namespace strata::core

int main(int argc, char** argv) {
    const bool two_gpu = argc == 2 && std::strcmp(argv[1], "--two-gpu") == 0;
    require(argc == 1 || two_gpu, "usage: mtp_sampling_test [--two-gpu]");
    int count = 0;
    const cudaError_t available = cudaGetDeviceCount(&count);
    if (available != cudaSuccess || count < (two_gpu ? 2 : 1)) {
        std::printf("SKIP: needs %s CUDA device%s\n", two_gpu ? "two real" : "a", two_gpu ? "s" : "");
        return 77;
    }
    constexpr int drafter_device = 0;
    const int source_device = two_gpu ? 1 : drafter_device;
    check(cudaSetDevice(drafter_device), "select drafter device");
    // Deliberately do not enable peer access: a copy may transfer between devices, but kernels read only local bias.
    if (two_gpu) {
        const cudaError_t e = cudaDeviceDisablePeerAccess(source_device);
        check(e == cudaErrorPeerAccessNotEnabled ? cudaSuccess : e, "disable kernel peer access");
        (void) cudaGetLastError();
    }
    using strata::core::MtpDrafter;
    using strata::core::MtpSamplingFixture;
    using strata::core::OnDevice;
    using strata::core::Verifier;
    std::string err;
    require(strata::core::coupled_draft_env(), "run with STRATA_SPEC_COUPLED=1 or STRATA_SPEC_PROB=1");
    strata::kernels::SamplerParams sp;
    sp.top_k = 1;   // sampled mode, with a literal, deterministic winner after adding the explicit bias
    sp.top_p = 1.0f;
    sp.temperature = 1.0f;
    sp.seed = 13;
    {
        MtpDrafter disabled;
        require(disabled.set_draft_sampling(sp, err), err.c_str());
        require(disabled.vram_bytes() == 0 && !MtpSamplingFixture::bias(disabled), "disabled allocates nothing");
    }
    MtpDrafter d;
    MtpSamplingFixture::init(d, drafter_device, err);
    const uint64_t base_vram = d.vram_bytes();
    float *logits = nullptr, *prob = nullptr;
    int32_t *step = nullptr, *out = nullptr;
    check(cudaMalloc(&logits, kVocab * sizeof(float)), "logits");
    check(cudaMalloc(&prob, sizeof(float)), "probability");
    check(cudaMalloc(&step, sizeof(int32_t)), "step record");
    check(cudaMalloc(&out, sizeof(int32_t)), "sample output");
    check(cudaMemset(step, 0, sizeof(int32_t)), "step zero");
    check(cudaStreamBeginCapture(d.stream(), cudaStreamCaptureModeThreadLocal), "capture sampler");
    MtpSamplingFixture::record(d, logits, step, out, prob);
    cudaGraph_t graph = nullptr;
    cudaGraphExec_t exec = nullptr;
    check(cudaStreamEndCapture(d.stream(), &graph), "end capture");
    check(cudaGraphInstantiate(&exec, graph, nullptr, nullptr, 0), "instantiate sampler graph");
    const float raw[kVocab] = {8.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    auto launch = [&] {
        check(cudaMemcpyAsync(logits, raw, sizeof(raw), cudaMemcpyHostToDevice, d.stream()), "reset raw logits");
        check(cudaGraphLaunch(exec, d.stream()), "replay sampler graph");
    };
    auto result = [&](int want, const char* what) {
        int32_t got = -1;
        float p = -1.0f;
        check(cudaStreamSynchronize(d.stream()), "wait sampler");
        check(cudaMemcpy(&got, out, sizeof(got), cudaMemcpyDeviceToHost), "read sample");
        check(cudaMemcpy(&p, prob, sizeof(p), cudaMemcpyDeviceToHost), "read probability");
        if (got != want) std::fprintf(stderr, "%s: want %d, got %d\n", what, want, got);
        require(got == want && p == 1.0f, what);
    };
    auto draw = [&](int want, const char* what) { launch(); result(want, what); };
    strata::core::ModelGeometry g;
    g.n_layers = 2;
    const float* owned_bias = nullptr;
    const char* one_block = std::getenv("STRATA_SAMPLER_ONE_BLOCK");
    MtpSamplingFixture::verdicts(drafter_device, g, one_block && std::atoi(one_block) != 0);
    {
        Verifier last, first;
        MtpSamplingFixture::init_source(last, source_device, g);
        MtpSamplingFixture::init_source(first, drafter_device, g, &last);
        require(first.set_logit_bias(bias_for(2), err), err.c_str());
        first.set_sampling(sp);
        require(first.sampling().logit_bias == last.sampling().logit_bias && first.sampling().logit_bias,
                "solo source is the final stage's actual bias");
        auto greedy = first.sampling();
        greedy.greedy = true;
        require(d.set_draft_sampling(greedy, err), err.c_str());
        require(!d.coupled() && !MtpSamplingFixture::bias(d) && d.vram_bytes() == base_vram,
                "greedy does not allocate draft bias");
        MtpSamplingFixture::params(d, false);
        require(d.set_draft_sampling(first.sampling(), err), err.c_str());
        // The same source allocation now contains a different request. A shallow borrow yields token 5, not 2.
        require(first.set_logit_bias(bias_for(5), err), err.c_str());
        draw(2, "owned bias survives source mutation");
        MtpSamplingFixture::params(d, true);
        owned_bias = MtpSamplingFixture::bias(d);
        require(d.vram_bytes() == base_vram + kVocab * sizeof(float), "owned bias counted once");
        launch();   // next request updates while a previous sampler graph may still be reading the records
        require(d.set_draft_sampling(first.sampling(), err), err.c_str());
        result(2, "request update waits for previous sampled graph");
        draw(5, "changed request bias reaches captured sampler");
        require(MtpSamplingFixture::bias(d) == owned_bias, "bias address stable across requests");
        require(first.set_logit_bias({}, err), err.c_str());
        first.set_sampling(sp);
        require(d.set_draft_sampling(first.sampling(), err), err.c_str());
        MtpSamplingFixture::params(d, false);
        draw(0, "cleared request uses raw logits");
        require(first.set_logit_bias(bias_for(3), err), err.c_str());
        first.set_sampling(sp);
        require(d.set_draft_sampling(first.sampling(), err), err.c_str());
    }   // free the actual verifier-owned source allocation before replaying the drafter's captured graph
    draw(3, "owned bias survives source destruction");
    MtpSamplingFixture::params(d, true);
    require(MtpSamplingFixture::bias(d) == owned_bias, "clear/reapply does not retire graph pointers");
    {
        Verifier last, first;
        MtpSamplingFixture::init_source(last, source_device, g);
        MtpSamplingFixture::init_source(first, drafter_device, g, &last);
        first.set_slot_sampling(0, sp);
        require(first.set_slot_logit_bias(0, bias_for(6), err), err.c_str());
        require(d.set_draft_sampling(first.slot_sampling(0), err), err.c_str());
        require(first.set_slot_logit_bias(0, bias_for(7), err), err.c_str());
        draw(6, "slot bias survives source mutation");
        MtpSamplingFixture::params(d, true);
        require(d.set_draft_sampling(first.slot_sampling(0), err), err.c_str());
        draw(7, "changed slot bias reaches captured sampler");
        const float* retired = first.slot_sampling(0).logit_bias;
        require(first.set_slot_logit_bias(0, {}, err), err.c_str());
        first.set_slot_sampling(0, sp);
        require(first.slot_sampling(0).logit_bias == nullptr && MtpSamplingFixture::slot_bias(last) == nullptr,
                "slot clear removes both source pointers");
        {
            OnDevice source(source_device);
            cudaPointerAttributes attr{};
            const cudaError_t e = cudaPointerGetAttributes(&attr, retired);
            require(e == cudaErrorInvalidValue || (e == cudaSuccess && attr.type == cudaMemoryTypeUnregistered),
                    "slot clear frees the source allocation instead of losing its owner");
            (void) cudaGetLastError();
        }
        draw(7, "drafter graph survives clearing/freeing slot source");
        require(d.set_draft_sampling(first.slot_sampling(0), err), err.c_str());
        MtpSamplingFixture::params(d, false);
        draw(0, "cleared slot uses raw logits");
        require(first.set_slot_logit_bias(0, bias_for(6), err), err.c_str());
        require(d.set_draft_sampling(first.slot_sampling(0), err), err.c_str());
        draw(6, "reapplied slot bias reaches captured sampler");
    }
    draw(6, "slot bias survives verifier destruction");
    require(MtpSamplingFixture::bias(d) == owned_bias && d.vram_bytes() == base_vram + kVocab * sizeof(float),
            "solo and slot requests reuse one local bias allocation");
    {
        OnDevice caller(source_device);
        require(d.set_draft_sampling(sp, err), err.c_str());
        int current = -1;
        check(cudaGetDevice(&current), "caller device");
        require(current == source_device, "setter restores caller device");
    }
    MtpSamplingFixture::params(d, false);
    draw(0, "clear after source destruction");
    check(cudaGraphExecDestroy(exec), "destroy graph execution");
    check(cudaGraphDestroy(graph), "destroy graph");
    check(cudaFree(logits), "free logits");
    check(cudaFree(prob), "free probability");
    check(cudaFree(step), "free step");
    check(cudaFree(out), "free output");
    std::printf("mtp_sampling_test: PASS (%s, source GPU %d -> drafter GPU %d)\n",
                d.prob() ? "prob" : "coupled", source_device, drafter_device);
    return 0;
}
