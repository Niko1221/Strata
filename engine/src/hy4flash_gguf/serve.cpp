// src/hy4flash_gguf/serve.cpp - see include/strata/hy4flash_gguf/serve.hpp.
#include "strata/hy4flash_gguf/serve.hpp"
#include "strata/hy4flash_gguf/conversation_cache.hpp"
#if defined(STRATA_HY4_FLASH_GPU_SAMPLER)
#include "strata/hy4flash_gguf/gpu_sampler.hpp"
#endif

#include "strata/hy4flash_gguf/dfly_loop.hpp"
#include "strata/hy4flash_gguf/mat.hpp"
#include "strata/hy4flash_gguf/mtp.hpp"
#include <cuda_runtime.h>
#include <algorithm>
#include <stdexcept>

#include <atomic>
#include <iostream>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace strata::hy4_flash {
namespace {

/// Read stdin on a separate thread so STOP can interrupt active generation.
class Lines {
public:
    // QUIT may arrive while the client still holds stdin open. The reader owns
    // the queue until EOF, including when serve_loop has already returned.
    static std::shared_ptr<Lines> start() {
        auto lines = std::make_shared<Lines>();
        std::thread([lines] { lines->read(); }).detach();
        return lines;
    }

    /// The next line, or false when stdin has ended and nothing is queued.
    bool next(std::string& out) {
        std::unique_lock<std::mutex> lk(m_);
        cv_.wait(lk, [this] { return !q_.empty() || eof_; });
        if (q_.empty()) return false;
        out = std::move(q_.front());
        q_.pop_front();
        return true;
    }

    /// Poll control messages without consuming a queued generation request.
    bool poll_control(std::string& out) {
        std::lock_guard<std::mutex> lk(m_);
        if (q_.empty()) return false;
        if (q_.front() != "STOP" && q_.front() != "QUIT") return false;
        out = std::move(q_.front());
        q_.pop_front();
        return true;
    }

    bool eof() const { return eof_.load(); }

private:
    void read() {
        std::string line;
        while (std::getline(std::cin, line)) {
            while (!line.empty() && (line.back() == '\r' || line.back() == '\n')) line.pop_back();
            {
                std::lock_guard<std::mutex> lk(m_);
                q_.push_back(line);
            }
            cv_.notify_one();
        }
        eof_ = true;
        cv_.notify_all();
    }
    std::mutex m_;
    std::condition_variable cv_;
    std::deque<std::string> q_;
    std::atomic<bool> eof_{false};
};

double ms_since(const std::chrono::steady_clock::time_point& t) {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t).count();
}

/// `GEN <max_new> [key=value ...] <id,id,...>`. The ids are the LAST field, so the keys are everything
/// between that and max_new - which is the main engine's grammar, not a convenience of this one.
bool parse_gen(const std::string& line, int& max_new, SamplerConfig& sc, bool& keep_turn, bool& reuse_prompt, bool& ignore_eos,
               std::vector<std::int32_t>& ids,
               std::string& err) {
    std::vector<std::string> f;
    for (std::size_t i = 0; i < line.size();) {
        while (i < line.size() && line[i] == ' ') ++i;
        const std::size_t s = i;
        while (i < line.size() && line[i] != ' ') ++i;
        if (i > s) f.emplace_back(line.substr(s, i - s));
    }
    if (f.size() < 3) { err = "GEN needs a token count and ids"; return false; }
    max_new = std::atoi(f[1].c_str());
    if (max_new < 1) { err = "GEN max_new must be at least 1"; return false; }
    for (std::size_t i = 2; i + 1 < f.size(); ++i) {
        const std::size_t eq = f[i].find('=');
        if (eq == std::string::npos) continue;   // an unknown positional field: ignored, as the main engine does
        const std::string k = f[i].substr(0, eq), v = f[i].substr(eq + 1);
        if (k == "temperature") sc.temperature = (float) std::atof(v.c_str());
        else if (k == "top_p") sc.top_p = (float) std::atof(v.c_str());
        else if (k == "top_k") sc.top_k = std::atoi(v.c_str());
        else if (k == "cache_prompt") reuse_prompt = v != "0";
        else if (k == "ignore_eos") ignore_eos = v == "1";
        else if (k == "ckpt") keep_turn = std::atoi(v.c_str()) != 0;
        else if (k == "seed") sc.seed = std::strtoull(v.c_str(), nullptr, 10);
        // every other key is the main engine's and not this one's: ignored rather than refused, so a
        // server that sends one does not lose the request.
    }
    ids.clear();
    const std::string& last = f.back();
    for (std::size_t i = 0; i < last.size();) {
        const std::size_t c = last.find(',', i);
        const std::string t = last.substr(i, c == std::string::npos ? std::string::npos : c - i);
        if (!t.empty()) ids.push_back((std::int32_t) std::strtol(t.c_str(), nullptr, 10));
        if (c == std::string::npos) break;
        i = c + 1;
    }
    if (ids.empty()) { err = "GEN had no ids"; return false; }
    if (sc.top_p <= 0.0f) sc.top_p = 1.0f;   // the server omits it rather than sending 0
    return true;
}

}  // namespace

int serve_loop(const ServeState& s) {
    if (s.gguf == nullptr || s.pack == nullptr || s.g == nullptr || s.cache == nullptr || s.fo == nullptr) {
        std::printf("ERR serve: the engine did not hand over its state\n");
        std::fflush(stdout);
        return 2;
    }
    const int chunk = prefill_chunk_max();
    const bool gpu_sampling=[] {const char* e=std::getenv("STRATA_HY4_FLASH_GPU_SAMPLE");return e&&std::atoi(e)!=0;}();
    const bool device_accept=[] {const char* e=std::getenv("STRATA_HY4_FLASH_DEVICE_ACCEPT");return e&&std::atoi(e)!=0;}();
    const bool device_continue=[] {const char* e=std::getenv("STRATA_HY4_FLASH_DEVICE_CONTINUE");return e&&std::atoi(e)!=0;}();
    const bool device_loop=[] {const char* e=std::getenv("STRATA_HY4_FLASH_DEVICE_LOOP");return e&&std::atoi(e)!=0;}();
    if(device_loop && (!device_accept||device_continue)){std::printf("ERR DEVICE_LOOP requires DEVICE_ACCEPT and DEVICE_CONTINUE=0\n");return 2;}
    if(device_loop){
        const char* graph=std::getenv("STRATA_HY4_FLASH_MTP_GRAPH");const char* cached=std::getenv("STRATA_HY4_FLASH_MTP_CACHE_ONLY");
        if(!graph||!std::atoi(graph)||!cached||!std::atoi(cached)){std::printf("ERR DEVICE_LOOP requires MTP_GRAPH and MTP_CACHE_ONLY\n");return 2;}
    }
    if(device_accept && (!gpu_sampling || !s.mtp)){std::printf("ERR DEVICE_ACCEPT requires GPU sampling and MTP\n");return 2;}
    if (s.dfly != nullptr && (s.mtp != nullptr || s.dfly_model == nullptr || s.dfly_k_max < 1)) {
        std::printf("ERR serve: --draft dfly needs its drafter, k >= 1 and no --mtp\n");
        return 2;
    }
    if (s.dfly != nullptr)
        std::fprintf(stderr, "HY4_FLASH_DFLY enabled k_max=%d window=%s verification=sample_and_match sampler=%s\n",
                     s.dfly_k_max, dfly_window_name(s.dfly_window), gpu_sampling ? "gpu" : "host");
    if(device_continue) {
        const char* graph=std::getenv("STRATA_HY4_FLASH_MTP_GRAPH");const char* cache=std::getenv("STRATA_HY4_FLASH_MTP_CACHE_ONLY");
        if(!device_accept||!graph||std::atoi(graph)==0||!cache||std::atoi(cache)==0) {
            std::printf("ERR DEVICE_CONTINUE requires DEVICE_ACCEPT, MTP_GRAPH and MTP_CACHE_ONLY\n");return 2;
        }
    }
#if defined(STRATA_HY4_FLASH_GPU_SAMPLER)
    std::unique_ptr<GpuSampler> gpu_sampler;
    if(gpu_sampling) {gpu_sampler=std::make_unique<GpuSampler>();std::fprintf(stderr,"HY4_FLASH_GPU_SAMPLE enabled full_vocab=1 rng=splitmix64\n");}
#else
    if(gpu_sampling){std::printf("ERR GPU sampler requires CUDA build\n");return 2;}
#endif
    struct Hidden {
        float* p = nullptr;
        ~Hidden() { if (p) cudaFree(p); }
    } hidden;
    const int hd = s.g->hidden_dim;
    if (s.mtp) {
        if (cudaMalloc(&hidden.p, (std::size_t)(s.max_context + 1) * hd * sizeof(float)) != cudaSuccess ||
            cudaMemset(hidden.p, 0, hd * sizeof(float)) != cudaSuccess) {
            std::printf("ERR MTP target hidden allocation failed\n"); return 2;
        }
        std::fprintf(stderr,"HY4_FLASH_MTP enabled draft_tokens=%d verification=sample_and_match\n",s.draft_tokens);
    }
    ConversationCache conversations(s.conversation_cache_mib, s.conversation_cache_slots);
    if (s.conversation_cache_mib > 0 && s.conversation_cache_slots > 0)
        std::printf("INFO conversation_cache_mib=%d conversation_cache_slots=%d\n",
                    s.conversation_cache_mib, s.conversation_cache_slots);
    std::printf("INFO hy4_flash_raw_completion=1\n");
    std::printf("READY %d stop\n", s.max_context);
    std::fflush(stdout);

    auto in = Lines::start();
    std::string line;
    // Reuse only the cache prefix that matches the next request.
    std::vector<std::int32_t> cached;
    // Disable prefix reuse when measuring full prefill.
    const bool no_reuse = [] {
        const char* v = std::getenv("STRATA_HY4_FLASH_NO_REUSE");
        return v != nullptr && *v == '1';
    }();
    while (in->next(line)) {
        if (line.empty()) continue;
        if (line == "QUIT") return 0;
        if (line == "STOP") continue;   // nothing is running; the main engine ignores it too
        const bool picture_request = line.rfind("GENI ", 0) == 0;
        if (line.rfind("GEN ", 0) != 0 && !picture_request) {
            // BGEN, VRAM and the rest: refused by name, so the server fails that request instead of
            // waiting for a DONE that will never come.
            std::printf("ERR this engine serves GEN, STOP and QUIT only (got: %.64s)\n", line.c_str());
            std::fflush(stdout);
            continue;
        }

        int max_new = 1;
        SamplerConfig sc = s.sampler;
        std::vector<std::int32_t> ids;
        std::string err;
        bool keep_turn = true;
        bool reuse_prompt = true, ignore_eos = false;
        if (!parse_gen(line, max_new, sc, keep_turn, reuse_prompt, ignore_eos, ids, err)) {
            std::printf("ERR %s\n", err.c_str());
            std::fflush(stdout);
            continue;
        }
        if (std::any_of(ids.begin(),ids.end(),[&](int32_t id){return id<0 || id>=s.vocab;})) {
            std::printf("ERR token outside vocabulary\n");std::fflush(stdout);continue;
        }
        if ((int) ids.size() + max_new > s.max_context) {
            std::printf("ERR the prompt and its %d new tokens need %d of %d context positions\n", max_new,
                        (int) ids.size() + max_new, s.max_context);
            std::fflush(stdout);
            continue;
        }

        std::unique_ptr<VisionPrompt> picture;
        if (picture_request) {
            if (s.vision_shard.empty()) {
                std::printf("ERR hy4-flash vision is disabled\n");
                std::fflush(stdout);
                continue;
            }
#ifdef STRATA_HY4_FLASH_VISION
            const auto last = line.rfind(' ');
            const auto before = line.rfind(' ', last - 1);
            picture = encode_pictures(s.vision_shard, line.substr(before + 1, last - before - 1), ids, err);
            if (!picture) {
                std::printf("ERR image: %s\n", err.c_str());
                std::fflush(stdout);
                continue;
            }
#endif
            // Token IDs alone cannot distinguish two pictures of the same size.
            reuse_prompt = false;
            keep_turn = false;
        }
        // The drafter only has sequential text positions. Use target-only decode for
        // pictures, including subsequent generated tokens with the image position offset.
        Mtp* const request_mtp = picture ? nullptr : s.mtp;

        // ---- CONTINUE FROM THE LONGEST SHARED PREFIX.
        std::size_t reuse = 0;
        const std::vector<std::int32_t> no_stops;
        const auto& stop_tokens = ignore_eos ? no_stops : s.stop_tokens;
        if (!no_reuse && reuse_prompt) {
            try {
                reuse = conversations.resume(s, hidden.p, cached, ids, keep_turn);
            } catch (const std::exception& e) {
                // A CUDA copy may have replaced part of the active image.
                std::fprintf(stderr, "hy4-flash conversation transfer failed: %s\n", e.what());
                return 2;
            }
        }
        s.cache->len = (int) reuse;
        if (reuse > 0) {
            std::printf("RESUME %zu\n", reuse);
            std::fflush(stdout);
        }
        std::vector<std::int32_t> history = ids;
        std::vector<std::int32_t> step = ids;
        if (ignore_eos) sc.excluded_token = s.g->eos_token_id;
        Sampler sampler(sc);
#if defined(STRATA_HY4_FLASH_GPU_SAMPLER)
        if(gpu_sampler){auto resolved=sc;resolved.seed=sampler.seed();gpu_sampler->configure(resolved);}
#endif
        const float* device_logits=nullptr;
        const auto t_prompt = std::chrono::steady_clock::now();
        double prompt_ms = 0.0;
        const char* finish = "length";
        int produced = 0;
        bool failed = false;
        long drafts_offered = 0, drafts_accepted = 0, windows = 0;
        double draft_ms = 0, verify_ms = 0, catchup_ms = 0, device_loop_ms=0;
        static const bool offload_stats=[] {const char* e=std::getenv("STRATA_HY4_FLASH_OFFLOAD_STATS");return e&&std::atoi(e)!=0;}();
        struct Cost {OffloadCounters c;uint64_t rows=0,passes=0;double wall_ms=0;};
        Cost prompt_cost,decode_cost;
        bool quit = false;
        auto control = [&] {
            std::string ctl;
            bool stop = false;
            while (in->poll_control(ctl)) {
                stop = true;
                if (ctl == "QUIT") quit = true;
            }
            return stop;
        };
        auto catchup = [&](const std::vector<int32_t>& input, int base) {
            if (!request_mtp) return;
            const auto start = std::chrono::steady_clock::now();
            request_mtp->bind(*s.fo->dense);
            for (std::size_t at=0; at<input.size(); at+=8) {
                const auto end=std::min(input.size(),at+8);
                request_mtp->catchup({input.begin()+(long)at,input.begin()+(long)end},
                               hidden.p+(std::size_t)(base+at)*hd);
            }
            catchup_ms += ms_since(start);
        };
        auto run_target = [&](const std::vector<int32_t>& input, std::vector<float>& logits, bool want, bool last) {
            const int base = s.cache->len;
            const auto before=s.fo->dense->offload;
            const auto cost_start=std::chrono::steady_clock::now();
            ForwardOptions co=*s.fo;
            co.picture = picture.get();
            co.history=&history;co.want_logits=want;co.logits_last_only=last;
            if(gpu_sampling && want)co.device_logits_out=&device_logits;
            if(request_mtp) {
                co.hidden_out=hidden.p+(std::size_t)(base+1)*hd;
                co.hidden_all_rows=true;co.speculative=true;
                const char* exact=std::getenv("STRATA_HY4_FLASH_VERIFY_DECODE_EQUIVALENT");
                co.decode_equivalent=produced>0 && (exact==nullptr || std::atoi(exact)!=0);
            }
            if(!forward_prefill(*s.gguf,*s.g,*s.pack,input,logits,err,co))throw std::runtime_error(err);
            if(offload_stats){
                auto& cost=produced?decode_cost:prompt_cost;const auto& after=s.fo->dense->offload;
                cost.rows+=input.size();++cost.passes;cost.wall_ms+=ms_since(cost_start);
                cost.c.expert_bytes+=after.expert_bytes-before.expert_bytes;
                cost.c.transfers+=after.transfers-before.transfers;
                cost.c.hits+=after.hits-before.hits;cost.c.misses+=after.misses-before.misses;
                cost.c.window_layers+=after.window_layers-before.window_layers;
                cost.c.window_entries+=after.window_entries-before.window_entries;
                cost.c.window_unique+=after.window_unique-before.window_unique;
            }
        };
        long dfly_rounds = 0, dfly_one_row = 0, dfly_fallbacks = 0, dfly_kv0_chosen = 0;
        long dfly_drafted = 0, dfly_gated = 0;
        double dfly_propose_ms = 0, dfly_verify_ms = 0;
        std::string dfly_k_hist;
        if (s.dfly != nullptr && !picture) {
            // Run the shared DFly loop for requests with an external drafter.
            s.dfly->trusted = true;
            DflyRequest req;
            req.prompt = &ids;
            req.max_new = max_new;
            req.k_max = s.dfly_k_max;
            req.window = s.dfly_window;
            req.stop_tokens = &s.stop_tokens;
            req.sampler = sc;
            req.draft_path = *s.dfly_model;
            req.tail_last_row_only = true;   // run_target's prompt rule: the last row's logits only
#if defined(STRATA_HY4_FLASH_GPU_SAMPLER)
            req.device_logits = gpu_sampler != nullptr;
#endif
            req.pick = [&](const float* row, const std::vector<std::int32_t>& h) -> std::int64_t {
#if defined(STRATA_HY4_FLASH_GPU_SAMPLER)
                if (gpu_sampler) return gpu_sampler->pick(row, (int) s.vocab, h);
#endif
                return sampler.pick(row, s.vocab, h);
            };
            req.emit = [&](std::int32_t id, const float*) {
                if (produced == 0) prompt_ms = ms_since(t_prompt);
                std::printf("T %d\n", (int) id);
                std::fflush(stdout);
                ++produced;
                history.push_back(id);
                return true;
            };
            req.should_stop = control;
            req.on_prompt_chunk = [&](int done, int total) {
                const double elapsed_ms = ms_since(t_prompt);
                std::printf("PP %d %d %.0f %.1f\n", done, total, elapsed_ms,
                            1000.0 * (done - (double) reuse) / std::max(1.0, elapsed_ms));
                std::fflush(stdout);
            };
            std::vector<std::int32_t> out;
            DflyStats st;
            std::string derr;
            DflyFinish fin = DflyFinish::error;
            try {
                fin = dfly_generate(*s.gguf, *s.g, *s.pack, *s.fo, *s.fo->dense, *s.cache, *s.dfly, req, out, st, derr);
            } catch (const std::exception& e) {
                // Thrown from inside a pass or a pick: the device may be ahead of the host, so reset the target
                // as the loop's own failures do.
                s.dfly->trusted = false;
                std::string e2;
                const bool clean = dfly_abort_reset(*s.fo->dense, *s.cache, e2) == DflyResetResult::clean;
                derr = std::string("dfly: ") + e.what() + (clean ? " (the target was reset)" : "; the reset failed too (" + e2 + ")");
                fin = clean ? DflyFinish::error : DflyFinish::restart_worker;
            }
            drafts_offered = st.offered;
            drafts_accepted = st.accepted;
            dfly_rounds = st.rounds;
            dfly_one_row = st.ar_steps;
            dfly_fallbacks = st.fallbacks;
            dfly_propose_ms = st.propose_ms;
            dfly_verify_ms = st.verify_ms;
            dfly_kv0_chosen = st.kv0_chosen;
            dfly_drafted = st.drafted;
            dfly_gated = st.gated;
            for (std::size_t i = 0; i < st.k_rounds.size(); ++i)
                dfly_k_hist += (i ? "/" : "") + std::to_string(st.k_rounds[i]);
            switch (fin) {
            case DflyFinish::stop: finish = "stop"; break;
            case DflyFinish::length: finish = "length"; break;
            case DflyFinish::cancel: finish = "cancel"; break;
            case DflyFinish::nonfinite: finish = "nonfinite"; break;
            case DflyFinish::error:
            case DflyFinish::restart_worker:
                std::printf("ERR %s\n", derr.c_str());
                std::fflush(stdout);
                cached.clear();
                if (fin == DflyFinish::restart_worker) return 3;   // the CUDA context cannot be trusted
                failed = true;
                break;
            }
        } else
        try {
        if(request_mtp)request_mtp->rewind((int)reuse);

        // the prompt, in chunks, with only the tail producing logits
        std::size_t done = reuse;
        while (!failed && ids.size() - done > (std::size_t) chunk) {
            if (control()) { finish="cancel"; break; }
            const std::vector<std::int32_t> part(ids.begin() + (long) done,
                                                 ids.begin() + (long) done + chunk);
            std::vector<float> ignored;
            run_target(part,ignored,false,true);
            catchup(part,(int)done);
            done += (std::size_t) chunk;
            // Keep the server informed while long prompts are still in prefill.
            const double elapsed_ms = ms_since(t_prompt);
            std::printf("PP %zu %zu %.0f %.1f\n", done, ids.size(), elapsed_ms,
                        1000.0 * (done - reuse) / std::max(1.0, elapsed_ms));
            std::fflush(stdout);
        }
        if (failed) continue;
        step.assign(ids.begin() + (long) done, ids.end());
        std::printf("REUSED %zu\n", reuse);
        std::fflush(stdout);

        while (produced < max_new && std::strcmp(finish,"cancel")!=0) {
            if (control()) { finish="cancel"; break; }
#if defined(STRATA_HY4_FLASH_GPU_SAMPLER)
            if(request_mtp && device_loop && produced>0 && sc.repeat_penalty==1.f) {
                DeviceLoopState loop_result;std::string loop_error;bool cancelled=false;
                const auto loop_start=std::chrono::steady_clock::now();
                const bool ran=forward_device_loop(*s.fo->dense,*s.cache,*s.g,*request_mtp,*gpu_sampler,history,hidden.p,
                    max_new-produced,s.draft_tokens,stop_tokens,
                    [&]{cancelled=control()||cancelled;return cancelled;},
                    [&](int32_t token){std::printf("T %d\n",token);std::fflush(stdout);history.push_back(token);},loop_result,loop_error);
                if(!loop_error.empty())throw std::runtime_error(loop_error);
                if(ran){device_loop_ms+=ms_since(loop_start);produced+=loop_result.produced;windows+=loop_result.windows;
                    drafts_accepted+=loop_result.accepted;drafts_offered+=loop_result.offered;
                    step.assign(1,history.back());
                    if(cancelled){finish="cancel";break;}if(loop_result.nonfinite){finish="nonfinite";break;}
                    if(loop_result.stopped){finish="stop";break;}continue;}
            }
#endif
            std::vector<float> logits;
            const int base=s.cache->len;
            std::vector<int32_t> draft;
            const bool verifying=request_mtp && produced>0;
            if(verifying) {
                request_mtp->checkpoint();
                const auto start=std::chrono::steady_clock::now();
                const float* h=hidden.p+(std::size_t)base*hd;
                int32_t token=history.back();
                const int count=std::min(s.draft_tokens,max_new-produced-1);
                SamplerConfig greedy;
                Sampler proposer(greedy);
                static const bool draft_graph=[] {const char* e=std::getenv("STRATA_HY4_FLASH_MTP_GRAPH");return e&&std::atoi(e)!=0;}();
                if(draft_graph)draft=request_mtp->draft(token,h,count);
                else for(int j=0;j<count;++j) {
                    std::vector<float> dl;
                    h=request_mtp->forward({token},h,&dl);
                    const auto next=proposer.pick(dl.data(),s.vocab,history);
                    if(next<0)throw std::runtime_error("MTP draft logits are nonfinite");
                    token=(int32_t)next;draft.push_back(token);
                }
                draft_ms+=ms_since(start);drafts_offered+=(long)draft.size();++windows;
                step={history.back()};step.insert(step.end(),draft.begin(),draft.end());
                const auto old_size=history.size();
                history.insert(history.end(),draft.begin(),draft.end());
                const auto verify_start=std::chrono::steady_clock::now();
                run_target(step,logits,true,false);
                verify_ms+=ms_since(verify_start);
                history.resize(old_size);
            } else {
                run_target(step,logits,true,true);
                catchup(step,base);
                if(produced==0) {
                    prompt_ms=ms_since(t_prompt);
                    static const bool loan_metrics=[] {const char* e=std::getenv("STRATA_HY4_FLASH_PREFILL_BORROW");return e&&std::atoi(e)!=0;}();
                    if(loan_metrics) std::fprintf(stderr,"HY4_FLASH_PREFILL_LOAN prompt_complete\n");
                }
            }
            int committed=0;
            bool ended=false;
            bool continued=false;
#if defined(STRATA_HY4_FLASH_GPU_SAMPLER)
            if(device_accept && verifying) {
                const auto* dv=gpu_sampler->begin_verify(device_logits,s.vocab,draft,history,base,max_new-produced,stop_tokens);
                forward_commit_device(*s.fo->dense,dv,gpu_sampler->stream());
                const bool queued=device_continue && request_mtp->continue_device(dv,gpu_sampler->stream(),step,hidden.p,base,max_new-produced);
                const auto verdict=gpu_sampler->finish_verify();
                if(queued) {
                    continued=verdict.count>0&&!verdict.stopped&&!verdict.nonfinite&&verdict.count+1<max_new-produced;
                    request_mtp->finish_continue(verdict,continued);
                }
                committed=verdict.count;ended=verdict.stopped;
                if(ended)finish="stop";
                if(verdict.nonfinite)finish="nonfinite";
                if(committed)forward_commit_mirror(*s.fo->dense,*s.cache,verdict);
                for(int j=0;j<committed;++j){std::printf("T %d\n",verdict.tokens[j]);history.push_back(verdict.tokens[j]);}
                std::fflush(stdout);produced+=committed;drafts_accepted+=verdict.accepted;
                if(control())finish="cancel";
            } else
#endif
            {
            for(std::size_t j=0;j<(verifying?step.size():1);++j) {
            const std::int64_t id =
#if defined(STRATA_HY4_FLASH_GPU_SAMPLER)
                gpu_sampler?gpu_sampler->pick(device_logits+j*s.vocab,s.vocab,history):
#endif
                sampler.pick(logits.data() + j*s.vocab, s.vocab, history);
            if (id < 0) {
                finish = "nonfinite";   // #879's reason: a window's logits held a NaN or an infinity
                break;
            }
            std::printf("T %d\n", (int) id);
            std::fflush(stdout);
            ++produced;
            ++committed;
            history.push_back((std::int32_t) id);

            // Emit the token before reporting that its stop condition ended the turn.
            for (std::int32_t t : stop_tokens) ended = ended || (std::int64_t) t == id;
            const bool matched=verifying && j<draft.size() && id==draft[j];
            if(matched)++drafts_accepted;
            if (ended) { finish = "stop"; break; }
            if(control()){finish="cancel";break;}
            if(!matched)break;
            }
            }
            if(verifying) {
                s.cache->len=base+committed;
                if(!continued){request_mtp->rewind(base);
                    if(committed)catchup({step.begin(),step.begin()+committed},base);}
            }
            step.assign(1,history.back());
            if(ended || std::strcmp(finish,"nonfinite")==0)break;
        }
        } catch(const std::exception& e) {
            std::printf("ERR %s\n",e.what());std::fflush(stdout);
            cached.clear();failed=true;
        }
        if (failed) { if(quit)return 0; continue; }

        const double total_ms = ms_since(t_prompt);
        // The common server reads accepted/offered/reused from the same DONE
        // fields for every model. Speculative tokens are emitted only after verification.
        std::printf("DONE %d %d %.1f %.1f %s %ld %ld %zu\n", produced, (int) ids.size(), prompt_ms,
                    total_ms - prompt_ms, finish, drafts_accepted, drafts_offered, reuse);
        if(request_mtp)std::fprintf(stderr,"HY4_FLASH_MTP_STATS windows=%ld accepted=%ld offered=%ld draft_ms=%.3f verify_ms=%.3f catchup_ms=%.3f\n",
            windows,drafts_accepted,drafts_offered,draft_ms,verify_ms,catchup_ms);
        if (s.dfly != nullptr && !picture)
            std::fprintf(stderr, "HY4_FLASH_DFLY_STATS rounds=%ld accepted=%ld offered=%ld one_row=%ld fallbacks=%ld "
                         "propose_ms=%.3f verify_ms=%.3f kv_rounds=%s drafted=%ld gated=%ld kv0_chosen=%ld\n",
                         dfly_rounds, drafts_accepted, drafts_offered, dfly_one_row, dfly_fallbacks, dfly_propose_ms,
                         dfly_verify_ms, dfly_k_hist.empty() ? "-" : dfly_k_hist.c_str(), dfly_drafted, dfly_gated,
                         dfly_kv0_chosen);
        if(device_loop)std::fprintf(stderr,"HY4_FLASH_DEVICE_LOOP_TIME ms=%.3f (all phases of device loops; other phase counters exclude these loops)\n",device_loop_ms);
        if(offload_stats)for(auto item:{std::pair<const char*,Cost*>{"prompt",&prompt_cost},{"decode",&decode_cost}}){
            const auto& c=*item.second;
            std::fprintf(stderr,"HY4_FLASH_OFFLOAD phase=%s rows=%llu passes=%llu expert_bytes=%llu transfers=%llu hits=%llu misses=%llu window_layers=%llu entries=%llu unique=%llu target_ms=%.3f outputs=%d device_loop_excluded=%d\n",
                item.first,(unsigned long long)c.rows,(unsigned long long)c.passes,
                (unsigned long long)c.c.expert_bytes,(unsigned long long)c.c.transfers,
                (unsigned long long)c.c.hits,(unsigned long long)c.c.misses,
                (unsigned long long)c.c.window_layers,(unsigned long long)c.c.window_entries,
                (unsigned long long)c.c.window_unique,c.wall_ms,produced,device_loop?1:0);
        }
        std::fflush(stdout);
        if(quit)return 0;
        // What the cache holds now: this prompt and everything generated from it, at their positions.
        if (std::strcmp(finish, "nonfinite") == 0) {
            conversations.clear();
            cached.clear();
        } else if (keep_turn) {
            history.resize(std::min(history.size(), size_t(s.cache->len)));
            cached = std::move(history);
        } else {
            cached.clear();
        }
    }
    return 0;
}

}  // namespace strata::hy4_flash
