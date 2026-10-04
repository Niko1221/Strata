
// dsv4_run: DeepSeek-V4-Flash runner (GPU matvecs, hot experts in VRAM, MISS on CPU, profile-driven residency).
//
// Two modes:
//   one-shot   --prompt-ids ... : prints the ids and the decoded text, then exits.
//   --serve    resident over stdin/stdout, speaking Strata's line protocol (see dsv4/serve_loop.hpp) so
//              serve/server.py can put the OpenAI/Anthropic API and the web app on top of it.
#include "dsv4/gpu.hpp"
#include "dsv4/model.hpp"
#include "dsv4/sampler.hpp"
#include "dsv4/serve_loop.hpp"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <sstream>
#include <string>
#include <vector>

using namespace dsv4;

static void usage() {
    std::puts(
        "usage: dsv4_run --model FIRST_SHARD.gguf (--prompt-ids 0,123,456 | --serve) [options]\n"
        "  --model F                 repeatable; '…-00001-of-0000N.gguf' auto-loads all shards\n"
        "  --prompt-ids a,b,c        token ids (tokenise with the HF tokenizer / chat template, see README)\n"
        "  --serve                   stay resident and speak Strata's engine protocol on stdin/stdout\n"
        "                            (READY/GEN/PP/T/DONE/ERR/STOP/QUIT) - see tools/serve_dsv4.py\n"
        "  --n-predict N             tokens to generate (default 16)             --ctx N (default 2048)\n"
        "  --temperature F --top-p F --top-k N --min-p F   sampling (default: greedy)\n"
        "  --seed N                  RNG seed (0 = from the clock)\n"
        "  --eos-ids a,b             ids that end a generation (default: the GGUF's eos_token_id)\n"
        "  --cpu                     no GPU at all (reference baseline)         --threads N / --pool-workers N\n"
        "  --no-qat-sim              do not simulate the FP8/FP4 activation quantisation of the reference\n"
        "  --max-layers N            debug: run the first N layers only\n"
        "  --expert-vram-pct P|auto  share of VRAM left after fixed costs for hot experts (0 = none, 100 = all that fits)\n"
        "  --expert-cache N          absolute number of VRAM expert slots (overrides pct)\n"
        "  --expert-vram-reserve-mib M   (default 1024)     --vram-free-mib N / --vram-total-mib N  (override driver values)\n"
        "  --expert-profile F        profile .bin (tools/make_expert_profile.py or --expert-profile-save): ONLY its experts go to RAM,\n"
        "                            the hottest ones to VRAM; the rest stay on disk (mmap)\n"
        "  --expert-profile-save F   write the routing-frequency profile at the end (atomic)\n"
        "  --expert-ram all|profile  'all' copies every expert to RAM (default: profile only)\n"
        "  --ram-cache-mib M         host-RAM LRU arena for MISS experts (default 8192, 0 = read straight from disk)\n"
        "  --expert-adapt-swaps N    max VRAM swaps per token (0 = static)      --verify-slots   byte-compare every uploaded slot\n"
        "  --dump-routing F          write the routing trace (input of tools/make_expert_profile.py)\n"
        "  --dump-logits F           write float32 logits of every processed token\n"
        "  --selfcheck               dequantisation statistics of real tensors, then exit if no prompt is given\n"
        "  --quiet                   no startup report on stderr (serve/server.py keeps its own log)\n"
        "  --pcie-frac F             not implemented (must be 0)");
}

static std::vector<int> split_ids(const std::string& s) {
    std::vector<int> out;
    std::stringstream ss(s);
    std::string t;
    while (std::getline(ss, t, ',')) if (!t.empty()) out.push_back(std::atoi(t.c_str()));
    return out;
}

int main(int argc, char** argv) {
    std::vector<std::string> shards;
    std::vector<int> prompt;
    std::vector<int> eos_ids;
    RunOpts o;
    SampleOpts so;
    int n_predict = 16;
    bool serve = false;
    std::string dump_routing, dump_logits;
    bool selfcheck = false;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&]() -> const char* { if (i + 1 >= argc) { usage(); std::exit(2); } return argv[++i]; };
        if (a == "--model") {
            std::string p = next();
            const size_t of = p.rfind("-of-");
            if (of != std::string::npos && of >= 5 && p.size() >= of + 9 && p.find_first_not_of("0123456789", of + 4) == of + 9) {
                const int n = std::atoi(p.substr(of + 4, 5).c_str());
                for (int k = 1; k <= n; ++k) { char b[16]; std::snprintf(b, sizeof b, "%05d", k); shards.push_back(p.substr(0, of - 5) + b + p.substr(of)); }
            } else shards.push_back(p);
        }
        else if (a == "--prompt-ids") prompt = split_ids(next());
        else if (a == "--eos-ids") eos_ids = split_ids(next());
        else if (a == "--serve") serve = true;
        else if (a == "--n-predict") n_predict = std::atoi(next());
        else if (a == "--ctx") o.ctx = std::atoi(next());
        else if (a == "--cpu") o.gpu = false;
        else if (a == "--threads" || a == "--pool-workers") o.threads = std::atoi(next());
        else if (a == "--no-qat-sim") o.qat_sim = false;
        else if (a == "--max-layers") o.max_layers = std::atoi(next());
        else if (a == "--quiet") o.verbose = false;
        else if (a == "--temperature") so.temperature = (float) std::atof(next());
        else if (a == "--top-p") so.top_p = (float) std::atof(next());
        else if (a == "--top-k") so.top_k = std::atoi(next());
        else if (a == "--min-p") so.min_p = (float) std::atof(next());
        else if (a == "--seed") so.seed = (uint64_t) std::strtoull(next(), nullptr, 10);
        else if (a == "--expert-vram-pct") { std::string v = next(); o.vram_pct = v == "auto" ? -1.0 : std::atof(v.c_str()); }
        else if (a == "--expert-cache") o.abs_slots = std::atoll(next());
        else if (a == "--expert-vram-reserve-mib") o.reserve_mib = std::atoll(next());
        else if (a == "--vram-free-mib") o.vram_free_mib = std::atoll(next());
        else if (a == "--vram-total-mib") o.vram_total_mib = std::atoll(next());
        else if (a == "--expert-profile") o.profile_in = next();
        else if (a == "--expert-profile-save") o.profile_out = next();
        else if (a == "--expert-ram") o.ram_all = std::string(next()) == "all";
        else if (a == "--ram-cache-mib") o.ram_cache_mib = std::atoll(next());
        else if (a == "--expert-adapt-swaps") o.adapt_swaps = std::atoi(next());
        else if (a == "--expert-source") { if (std::string(next()) == "host") std::fprintf(stderr, "note: --expert-source host == --expert-ram all in this build; experts are always mmap'ed first\n"); }
        else if (a == "--verify-slots") o.verify_slots = true;
        else if (a == "--dump-routing") dump_routing = next();
        else if (a == "--dump-logits") dump_logits = next();
        else if (a == "--selfcheck") selfcheck = true;
        else if (a == "--pcie-frac") { if (std::atof(next()) != 0.0) { std::fprintf(stderr, "error: the PCIe miss path is not implemented in this build (--pcie-frac must be 0)\n"); return 2; } }
        else { usage(); return a == "--help" || a == "-h" ? 0 : 2; }
    }
    if (shards.empty()) { usage(); return 2; }
    if (!serve && prompt.empty() && !selfcheck) { usage(); return 2; }
    if (o.vram_pct < 0 && o.abs_slots < 0 && !o.gpu) o.vram_pct = 0;

    Model m; std::string err;
    auto t0 = std::chrono::steady_clock::now();
    if (!m.load(shards, o, err)) { std::fprintf(stderr, "error: %s\n", err.c_str()); return 1; }
    std::fprintf(stderr, "loaded in %.1f s (%s)\n", std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count(), o.gpu ? (gpu::is_emulated() ? "GPU EMULATED on CPU" : "CUDA") : "CPU only");
    // selfcheck prints a table on stdout; in --serve that channel is the protocol, so it stays out of it.
    if (selfcheck && serve) std::fprintf(stderr, "note: --selfcheck is skipped with --serve (stdout is the protocol)\n");
    else if (selfcheck) m.selfcheck();
    if (serve) return serve_loop(m, o, eos_ids);
    if (prompt.empty()) return selfcheck ? 0 : (usage(), 2);

    const Dsv4Config& c = m.config();
    std::vector<int> stops = eos_ids;
    if (stops.empty() && c.eos >= 0) stops.push_back(c.eos);
    FILE* tr = nullptr; FILE* lf = nullptr;
    if (!dump_routing.empty()) {
        tr = std::fopen(dump_routing.c_str(), "wb");
        if (!tr) { std::fprintf(stderr, "cannot write %s\n", dump_routing.c_str()); return 1; }
        const uint32_t ver = 1, nl = (uint32_t) c.n_layer, ne = (uint32_t) c.n_expert, k = (uint32_t) c.n_expert_used; const uint64_t mh = m.model_hash();
        std::fwrite("DSV4TRCE", 1, 8, tr); std::fwrite(&ver, 4, 1, tr); std::fwrite(&nl, 4, 1, tr); std::fwrite(&ne, 4, 1, tr); std::fwrite(&k, 4, 1, tr); std::fwrite(&mh, 8, 1, tr);
    }
    if (!dump_logits.empty()) lf = std::fopen(dump_logits.c_str(), "wb");
    if ((int) prompt.size() + n_predict > o.ctx) { std::fprintf(stderr, "error: prompt + n-predict exceeds --ctx %d\n", o.ctx); return 1; }

    std::vector<float> logits;
    int pos = 0, next_tok = -1;
    auto step = [&](int tok, bool want) {
        std::fill(m.last_routing.begin(), m.last_routing.end(), -1);
        m.forward(tok, pos, (want || lf) ? &logits : nullptr);
        if (tr) std::fwrite(m.last_routing.data(), 4, m.last_routing.size(), tr);
        if (lf && !logits.empty()) std::fwrite(logits.data(), 4, logits.size(), lf);
        m.end_token();
        ++pos;
    };
    auto tp = std::chrono::steady_clock::now();
    for (size_t i = 0; i < prompt.size(); ++i) step(prompt[i], i + 1 == prompt.size());
    const double prefill_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - tp).count();
    std::printf("prompt: %zu tokens in %.2f s (%.2f tok/s)\n", prompt.size(), prefill_s, prompt.size() / std::max(prefill_s, 1e-9));
    std::printf("generated ids:");
    std::string text;
    auto tg = std::chrono::steady_clock::now();
    int produced = 0;
    // The one-shot path uses the same sampler as --serve: temperature 0 (the default) is still greedy argmax.
    Sampler sampler(so);
    std::vector<int> history = prompt;
    const std::vector<std::string>& vocab = m.tokens();
    for (int i = 0; i < n_predict; ++i) {
        next_tok = logits.empty() ? -1 : sampler.sample(logits.data(), (int) logits.size(), history);
        if (next_tok < 0) break;
        std::printf(" %d", next_tok); std::fflush(stdout);
        if ((size_t) next_tok < vocab.size()) text += decode_token_text(vocab[(size_t) next_tok]);
        history.push_back(next_tok);
        ++produced;
        if (std::find(stops.begin(), stops.end(), next_tok) != stops.end()) break;
        if (i + 1 < n_predict) step(next_tok, true);
    }

    const double gen_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - tg).count();
    std::printf("\ntext: %s\n", text.c_str());
    std::printf("decode: %d tokens in %.2f s (%.2f tok/s)\n", produced, gen_s, produced / std::max(gen_s, 1e-9));
    const Stats& s = m.stats;
    const double tot = (double) (s.hits + s.misses);
    std::fprintf(stderr, "expert stats:\n  hits: %llu\n  misses: %llu\n  hit rate: %.2f%%\n  admits: %llu  swaps: %llu\n  RAM cache: %llu hits, %llu loads (%.2f GiB), %llu evictions\n  H2D: %.3f GiB\n  CPU miss time: %.2f s\n  GPU hit wait: %.2f s\n",
                 (unsigned long long) s.hits, (unsigned long long) s.misses, tot > 0 ? 100.0 * (double) s.hits / tot : 0.0, (unsigned long long) s.admits,
                 (unsigned long long) s.swaps, (unsigned long long) s.cache_hits, (unsigned long long) s.cache_loads,
                 s.cache_bytes / 1073741824.0, (unsigned long long) s.cache_evictions,
                 s.h2d_bytes / 1073741824.0, s.cpu_miss_s, s.gpu_hit_s);
    if (tr) std::fclose(tr);
    if (lf) std::fclose(lf);
    if (!o.profile_out.empty()) { if (!m.save_profile(err)) std::fprintf(stderr, "profile save failed: %s\n", err.c_str()); else std::fprintf(stderr, "profile saved: %s\n", o.profile_out.c_str()); }
    std::printf("\nGenerated %d tokens in %.2f s (%.2f tok/s)\n", produced, gen_s, produced / std::max(gen_s, 1e-9));
    return 0;
}
