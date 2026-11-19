// vulkan_engine.cpp - NATIVE Strata Vulkan engine for qwen35moe (35B-A3B).
//
// Token generation runs entirely on our Vulkan forward path
// (engine/vulkan/native/model.cpp): resident dense weights + streamed MoE
// experts + our router/combine/sampler. No llama.cpp inference is involved.
// The tokenizer still lives in the Python server (exported from the GGUF).
//
// Extra commands beyond the Strata protocol:
//   EXPERT <layer> <expert>   full gate -> SiLU*up -> down forward on our tier
//   SHEXP <layer>             shared expert: FFN + scalar sigmoid gate
//   VSTAT                     streaming cache stats
//
// Build: see build.sh.  Run: see README.md.

#include "strata/vulkan/expert_stream.hpp"
#include "strata/vulkan/router.hpp"
#include "model.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <limits>
#include <numeric>
#include <random>
#include <string>
#include <vector>

using namespace strata::vulkan;

namespace {

struct Options {
    std::string model;
    std::string native_shaders = "engine/vulkan/native/shaders";
    std::string expert_shaders = "engine/vulkan/shaders";
    std::string q4_spv, q6_spv, q8_spv, silu_spv, rgemv_spv, rtopk_spv;
    std::string iq3_xxs_spv, iq2_xs_spv, iq1_m_spv;
    int ctx = 32768;
    int threads = 8;
    int slots = 2048;
    std::string trace;  // --trace FILE: router top-k ids per layer
    bool debug = false;
    int lp = 0;  // --lp K: emit top-K logprobs per generated token for parity checks
};

bool take_int(const std::string& s, const std::string& key, int& out) {
    if (s.rfind(key + "=", 0) != 0) return false;
    try { out = std::stoi(s.substr(key.size() + 1)); } catch (...) { return false; }
    return true;
}
bool take_float(const std::string& s, const std::string& key, float& out) {
    if (s.rfind(key + "=", 0) != 0) return false;
    try { out = std::stof(s.substr(key.size() + 1)); } catch (...) { return false; }
    return true;
}

struct GenRequest {
    int max_new = 0;
    std::vector<int> prompt;
    float temperature = 0.8f;
    float top_p = 0.95f;
    int top_k = 40;
    uint32_t seed = 42;
    bool greedy = false;
};

bool parse_gen(const std::string& line, GenRequest& req) {
    std::vector<std::string> f;
    size_t pos = 0;
    while (pos < line.size()) {
        size_t end = line.find(' ', pos);
        if (end == std::string::npos) end = line.size();
        if (end > pos) f.push_back(line.substr(pos, end - pos));
        pos = end + 1;
    }
    if (f.size() < 2) return false;
    try { req.max_new = std::stoi(f[1]); } catch (...) { return false; }
    if (req.max_new < 0) req.max_new = 0;
    const std::string& ids = f.back();
    size_t start = 0;
    while (start <= ids.size()) {
        size_t comma = ids.find(',', start);
        std::string tok = ids.substr(start, comma == std::string::npos ? std::string::npos : comma - start);
        if (!tok.empty()) {
            try { req.prompt.push_back(std::stoi(tok)); } catch (...) { /* ignore */ }
        }
        if (comma == std::string::npos) break;
        start = comma + 1;
    }
    for (size_t i = 2; i + 1 < f.size(); ++i) {
        int k = 0; float v = 0.0f;
        if (take_float(f[i], "temperature", v)) { req.temperature = v; req.greedy = v <= 0.0001f; }
        else if (take_float(f[i], "top_p", v)) req.top_p = v;
        else if (take_int(f[i], "top_k", k)) req.top_k = k;
        else if (take_int(f[i], "seed", k)) req.seed = static_cast<uint32_t>(k);
    }
    return true;
}

void emit(const char* fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    std::vprintf(fmt, ap);
    va_end(ap);
    std::cout << std::flush;
}

// Emit top-K logprobs in Strata's LP wire format (see serve/server.py):
//   LP <logprob of the token just emitted as T> <tok>:<logprob> ...
// Logprobs are log_softmax over the full vocabulary (temperature-independent),
// so they can be compared with the llama.cpp bridge and read by Strata's server.
void emit_top_logprobs(const float* logits, int n, int k, int chosen) {
    if (k <= 0 || n <= 0) return;
    float mx = -std::numeric_limits<float>::infinity();
    for (int i = 0; i < n; ++i) if (!std::isnan(logits[i])) mx = std::max(mx, logits[i]);
    double z = 0.0;
    for (int i = 0; i < n; ++i) z += std::exp(double(logits[i] - mx));
    const double lz = std::log(z);
    const int keep = std::min(k, n);
    std::vector<int> top(n);
    std::iota(top.begin(), top.end(), 0);
    std::partial_sort(top.begin(), top.begin() + keep, top.end(), [&](int a, int b) {
        return logits[a] > logits[b];
    });
    emit("LP %.6f", double(logits[chosen] - mx) - lz);
    for (int i = 0; i < keep; ++i) {
        int t = top[i];
        emit(" %d:%.6f", t, double(logits[t] - mx) - lz);
    }
    emit("\n");
}

bool select_device(VkInstance inst, VkPhysicalDevice& phys, uint32_t& qfam) {
    uint32_t n = 0;
    vkEnumeratePhysicalDevices(inst, &n, nullptr);
    std::vector<VkPhysicalDevice> devs(n);
    vkEnumeratePhysicalDevices(inst, &n, devs.data());
    for (auto d : devs) {
        VkPhysicalDeviceProperties p{};
        vkGetPhysicalDeviceProperties(d, &p);
        if (p.vendorID != 0x8086) continue;
        uint32_t nq = 0;
        vkGetPhysicalDeviceQueueFamilyProperties(d, &nq, nullptr);
        std::vector<VkQueueFamilyProperties> qs(nq);
        vkGetPhysicalDeviceQueueFamilyProperties(d, &nq, qs.data());
        for (uint32_t i = 0; i < nq; ++i)
            if (qs[i].queueFlags & VK_QUEUE_COMPUTE_BIT) { phys = d; qfam = i; return true; }
    }
    return false;
}

// On an integrated GPU the Vulkan DEVICE_LOCAL heap is system RAM. The old
// vram_free_mib=0 was a placeholder: report what is actually free, capped by
// the heap the driver exposes. MemAvailable is the honest number here (a PCI
// BAR reading is not a memory limit). Unknown memory leaves the 0 placeholder.
static long long free_shared_mib(VkPhysicalDevice phys) {
    long long available = 0;
    if (FILE* f = std::fopen("/proc/meminfo", "r")) {
        char row[256] = {};
        while (std::fgets(row, sizeof row, f)) {
            long long kb = 0;
            if (std::sscanf(row, "MemAvailable: %lld kB", &kb) == 1) { available = kb / 1024; break; }
        }
        std::fclose(f);
    }
    if (available <= 0) return 0;
    VkPhysicalDeviceMemoryProperties mp{};
    vkGetPhysicalDeviceMemoryProperties(phys, &mp);
    long long heap = 0;
    for (uint32_t i = 0; i < mp.memoryHeapCount; ++i)
        if (mp.memoryHeaps[i].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT)
            heap += static_cast<long long>(mp.memoryHeaps[i].size >> 20);
    return heap > 0 ? std::min(available, heap) : available;
}

// Parse the comma-separated token ids of an EVAL line.
std::vector<int> parse_ids(const std::string& line) {
    std::vector<int> ids;
    size_t pos = line.find(' ');
    if (pos == std::string::npos) return ids;
    const std::string s = line.substr(pos + 1);
    size_t start = 0;
    while (start <= s.size()) {
        size_t comma = s.find(',', start);
        std::string tok = s.substr(start, comma == std::string::npos ? std::string::npos : comma - start);
        if (!tok.empty()) { try { ids.push_back(std::stoi(tok)); } catch (...) {} }
        if (comma == std::string::npos) break;
        start = comma + 1;
    }
    return ids;
}

int argmax_of(const float* x, int n) {
    int best = 0;
    for (int i = 1; i < n; ++i) if (x[i] > x[best]) best = i;
    return best;
}

// Greedy / temperature / top-k / top-p on the host (logits are one row).
int sample(const float* logits, int n, const GenRequest& req) {
    if (n <= 0) return 0;
    if (req.greedy) {
        // softmax is monotonic: argmax of the logits is argmax of the
        // probabilities. Skip the 248k exp() calls and the two scratch arrays
        // on the most common decode path. NaNs lose, ties pick the lowest id,
        // matching the old softmax + argmax behaviour.
        int best = 0;
        for (int i = 0; i < n; ++i) {
            if (std::isnan(logits[i])) continue;
            if (best < 0 || std::isnan(logits[best]) || logits[i] > logits[best]) best = i;
        }
        return best < 0 ? 0 : best;
    }
    const float temp = std::max(0.01f, req.temperature);
    std::vector<float> p(static_cast<size_t>(n));
    float mx = -std::numeric_limits<float>::infinity();
    for (int i = 0; i < n; ++i) if (!std::isnan(logits[i])) mx = std::max(mx, logits[i]);
    double sum = 0.0;
    for (int i = 0; i < n; ++i) {
        float v = std::isnan(logits[i]) ? -std::numeric_limits<float>::infinity() : (logits[i] - mx) / temp;
        p[i] = float(std::exp(double(v)));
        sum += p[i];
    }
    for (int i = 0; i < n; ++i) p[i] = float(p[i] / sum);

    if (req.greedy) {
        int best = 0;
        for (int i = 1; i < n; ++i) if (p[i] > p[best]) best = i;
        return best;
    }
    std::vector<int> cand(n);
    std::iota(cand.begin(), cand.end(), 0);
    if (req.top_k > 0 && req.top_k < n) {
        std::partial_sort(cand.begin(), cand.begin() + req.top_k, cand.end(),
                          [&](int a, int b) { return p[a] > p[b] || (p[a] == p[b] && a < b); });
        cand.resize(req.top_k);
    }
    if (req.top_p < 1.0f) {
        double acc = 0.0; size_t keep = cand.size();
        for (size_t i = 0; i < cand.size(); ++i) { acc += p[cand[i]]; if (acc >= req.top_p) { keep = i + 1; break; } }
        cand.resize(keep);
    }
    std::vector<double> w;
    w.reserve(cand.size());
    for (int id : cand) w.push_back(std::max(0.0, double(p[id])));
    double tot = std::accumulate(w.begin(), w.end(), 0.0);
    if (!(tot > 0)) return cand.empty() ? 0 : cand[0];
    for (double& x : w) x /= tot;
    std::mt19937 rng(req.seed);
    std::discrete_distribution<int> dist(w.begin(), w.end());
    return cand[dist(rng)];
}

} // namespace

int main(int argc, char** argv) {
    Options opt;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&]() -> std::string { return i + 1 < argc ? argv[++i] : std::string(); };
        if (a == "--model") opt.model = next();
        else if (a == "--native-shaders") opt.native_shaders = next();
        else if (a == "--expert-shaders") opt.expert_shaders = next();
        else if (a == "--ctx") opt.ctx = std::atoi(next().c_str());
        else if (a == "--threads") opt.threads = std::atoi(next().c_str());
        else if (a == "--slots") opt.slots = std::atoi(next().c_str());
        else if (a == "--lp") opt.lp = std::atoi(next().c_str());
        else if (a == "--debug") opt.debug = true;
        else if (a == "--trace") opt.trace = next();
        else if (a == "--q4-shader") opt.q4_spv = next();
        else if (a == "--down-shader") opt.q6_spv = next();
        else if (a == "--q8-shader") opt.q8_spv = next();
        else if (a == "--iq3-xxs-shader") opt.iq3_xxs_spv = next();
        else if (a == "--iq2-xs-shader") opt.iq2_xs_spv = next();
        else if (a == "--iq1-m-shader") opt.iq1_m_spv = next();
        else if (a == "--silu-shader") opt.silu_spv = next();
        else if (a == "--router-gemv-shader") opt.rgemv_spv = next();
        else if (a == "--router-topk-shader") opt.rtopk_spv = next();
        // --backend-dir / --serve and other config flags are accepted but unused.
    }
    if (opt.model.empty()) {
        std::fprintf(stderr,
                     "usage: %s --model FILE.gguf [--native-shaders DIR] [--expert-shaders DIR] "
                     "[--ctx N] [--threads N] [--slots N]\n", argv[0]);
        return 2;
    }
    if (opt.ctx < 512) opt.ctx = 32768;
    if (opt.threads < 1) opt.threads = 8;
    if (opt.slots < 1) opt.slots = 2048;  // ~1.8 GiB pool; keeps most hot experts resident
    if (opt.q4_spv.empty()) opt.q4_spv = opt.expert_shaders + "/expert_q4_k_gemv.comp.spv";
    if (opt.q6_spv.empty()) opt.q6_spv = opt.expert_shaders + "/expert_q6_k_gemv.comp.spv";
    if (opt.q8_spv.empty()) opt.q8_spv = opt.expert_shaders + "/expert_q8_0_gemv.comp.spv";
    if (opt.iq3_xxs_spv.empty()) opt.iq3_xxs_spv = opt.expert_shaders + "/expert_iq3_xxs_gemv.comp.spv";
    if (opt.iq2_xs_spv.empty()) opt.iq2_xs_spv = opt.expert_shaders + "/expert_iq2_xs_gemv.comp.spv";
    if (opt.iq1_m_spv.empty()) opt.iq1_m_spv = opt.expert_shaders + "/expert_iq1_m_gemv.comp.spv";
    if (opt.silu_spv.empty()) opt.silu_spv = opt.expert_shaders + "/expert_silu_mul.comp.spv";
    if (opt.rgemv_spv.empty()) opt.rgemv_spv = opt.expert_shaders + "/router_gemv.comp.spv";
    if (opt.rtopk_spv.empty()) opt.rtopk_spv = opt.expert_shaders + "/router_topk.comp.spv";

    std::string err;

    // ---- Vulkan device -------------------------------------------------------
    VkApplicationInfo app{VK_STRUCTURE_TYPE_APPLICATION_INFO};
    app.apiVersion = VK_API_VERSION_1_3;
    VkInstanceCreateInfo ici{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
    ici.pApplicationInfo = &app;
    VkInstance inst = VK_NULL_HANDLE;
    if (vkCreateInstance(&ici, nullptr, &inst) != VK_SUCCESS) {
        std::fprintf(stderr, "vkCreateInstance failed\n"); return 1;
    }
    VkPhysicalDevice phys = VK_NULL_HANDLE;
    uint32_t qfam = 0;
    if (!select_device(inst, phys, qfam)) {
        std::fprintf(stderr, "no Intel Vulkan device\n"); return 1;
    }
    VkDevice dev = VK_NULL_HANDLE;
    VkQueue queue = VK_NULL_HANDLE;
    float prio = 1.0f;
    VkDeviceQueueCreateInfo qi{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
    qi.queueFamilyIndex = qfam; qi.queueCount = 1; qi.pQueuePriorities = &prio;
    VkDeviceCreateInfo di{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
    di.queueCreateInfoCount = 1; di.pQueueCreateInfos = &qi;
    if (vkCreateDevice(phys, &di, nullptr, &dev) != VK_SUCCESS) return 1;
    vkGetDeviceQueue(dev, qfam, 0, &queue);

    // ---- Our streaming tier + router -----------------------------------------
    ExpertIndex index;
    if (!ExpertIndex::load(opt.model, index, err)) {
        std::fprintf(stderr, "index: %s\n", err.c_str()); return 1;
    }
    StreamConfig scfg;
    scfg.cache_slots = uint32_t(opt.slots);
    scfg.io_threads = uint32_t(opt.threads);
    // The router trace (check_prefetch.py) shows only ~3% overlap between the
    // experts chosen by consecutive layers, so guessing layer l+1 from layer l
    // wastes about half the NVMe reads. On-demand reads plus the 2048-slot cache
    // are faster; keep the heuristic opt-in for other models.
    scfg.prefetch_next_layer = std::getenv("STRATA_PREFETCH") != nullptr;
    auto stream = ExpertStream::create(index, phys, dev, qfam, queue,
                                       opt.q4_spv, opt.q6_spv, opt.silu_spv, opt.q8_spv,
                                       opt.iq3_xxs_spv, opt.iq2_xs_spv, opt.iq1_m_spv,
                                       scfg, err);
    if (!stream) { std::fprintf(stderr, "stream: %s\n", err.c_str()); return 1; }

    Router::Limits rlim;
    rlim.max_experts = std::max<uint32_t>(256, index.n_experts());
    rlim.max_k = 2048; rlim.max_top_k = 64; rlim.max_tokens = 64;
    auto router = Router::create(phys, dev, qfam, queue, opt.rgemv_spv, opt.rtopk_spv, rlim, err);
    if (!router) { std::fprintf(stderr, "router: %s\n", err.c_str()); return 1; }

    // ---- Native forward ------------------------------------------------------
    native::VulkanCtx vctx{phys, dev, uint32_t(qfam), queue};
    native::Geometry geom;
    geom.n_layers = 40; geom.n_embd = 2048; geom.n_head = 16; geom.n_head_kv = 2;
    geom.head_dim = 256; geom.n_experts = 256; geom.top_k = 8; geom.n_ff = 512;
    geom.n_ff_shexp = 512; geom.hk = 16; geom.hv = 32; geom.d_state = 128;
    geom.d_conv = 4; geom.conv_channels = 8192; geom.n_rot = 64;
    native::NativeForward::Deps deps{stream.get(), router.get(), &index};
    native::NativeForward::ShaderDirs dirs{opt.native_shaders, opt.expert_shaders};
    auto forward = native::NativeForward::create(vctx, opt.model, geom, deps, dirs, opt.ctx, err);
    if (!forward) { std::fprintf(stderr, "native forward: %s\n", err.c_str()); return 1; }
    if (const char* sd = std::getenv("STRATA_NATIVE_STAGE_DIR")) forward->set_stage_dump_dir(sd);
    FILE* trace = opt.trace.empty() ? nullptr : std::fopen(opt.trace.c_str(), "w");
    if (trace) forward->set_trace(trace);
    // debug knobs (temporary)
    // Note: rms_debug_ is set in model.cpp from getenv; see below.
    const int n_vocab = forward->geometry().n_vocab;
    const int eos = forward->geometry().eos;
    std::vector<float> logits(static_cast<size_t>(n_vocab));
    FILE* dump = nullptr;
    const char* dump_env = std::getenv("STRATA_NATIVE_DUMP_LAYERS");
    if (dump_env && *dump_env) dump = std::fopen(dump_env, "wb");

    emit("INFO engine=strata-vulkan-native streaming=1\n");
    emit("READY %d stop\n", opt.ctx);

    std::string line;
    while (std::getline(std::cin, line)) {
        if (line == "QUIT") break;
        if (line.rfind("VRAM", 0) == 0) {
            emit("VRAM expert_slots=%d vram_free_mib=%lld\n", opt.slots, free_shared_mib(phys));
            continue;
        }
        if (line.rfind("VSTAT", 0) == 0) {
            StreamStats st = stream->stats();
            emit("VSTAT hits=%llu misses=%llu prefetches=%llu mb=%.1f\n",
                 (unsigned long long)st.hits, (unsigned long long)st.misses,
                 (unsigned long long)st.prefetches, double(st.bytes_read) / 1e6);
            continue;
        }
        if (line.rfind("EXPERT", 0) == 0) {
            int layer = -1; long e = -1;
            if (std::sscanf(line.c_str(), "EXPERT %d %ld", &layer, &e) != 2 || e < 0) {
                emit("ERR usage: EXPERT <layer> <expert>\n"); continue;
            }
            const ExpertTensor* g = index.find(layer, ExpertKind::Gate);
            const ExpertTensor* u = index.find(layer, ExpertKind::Up);
            const ExpertTensor* d = index.find(layer, ExpertKind::Down);
            if (!g || !u || !d || uint32_t(e) >= g->n_experts) {
                emit("ERR expert out of range\n"); continue;
            }
            uint32_t sg = 0, su = 0, sd = 0;
            if (!stream->ensure({layer, ExpertKind::Gate, uint32_t(e)}, sg, err) ||
                !stream->ensure({layer, ExpertKind::Up, uint32_t(e)}, su, err) ||
                !stream->ensure({layer, ExpertKind::Down, uint32_t(e)}, sd, err)) {
                emit("ERR ensure: %s\n", err.c_str()); continue;
            }
            std::vector<float> x(size_t(g->k)), gg(size_t(g->n_rows)),
                ug(size_t(u->n_rows)), hg(size_t(d->k)), og(size_t(d->n_rows));
            for (size_t i = 0; i < x.size(); ++i) x[i] = std::sin(float(i) * 0.017f) * 0.5f;
            const auto t0 = std::chrono::steady_clock::now();
            bool ok = stream->matvec({layer, ExpertKind::Gate, uint32_t(e)}, sg, x.data(), g->k, gg.data(), g->n_rows, err) &&
                      stream->matvec({layer, ExpertKind::Up, uint32_t(e)}, su, x.data(), u->k, ug.data(), u->n_rows, err) &&
                      stream->silu_mul(gg.data(), ug.data(), hg.data(), d->k, err) &&
                      stream->matvec({layer, ExpertKind::Down, uint32_t(e)}, sd, hg.data(), d->k, og.data(), d->n_rows, err);
            if (!ok) { emit("ERR expert: %s\n", err.c_str()); continue; }
            double sum = 0; for (float v : og) sum += v;
            const double us = std::chrono::duration<double, std::micro>(
                std::chrono::steady_clock::now() - t0).count();
            emit("EXPERT %d %ld sum=%.6f first=%.6f us=%.0f\n", layer, e, sum, og[0], us);
            continue;
        }
        if (line.rfind("SHEXP", 0) == 0) {
            int layer = -1;
            if (std::sscanf(line.c_str(), "SHEXP %d", &layer) != 1) {
                emit("ERR usage: SHEXP <layer>\n"); continue;
            }
            const ExpertTensor* sg = index.find_shared(layer, ExpertKind::Gate);
            const ExpertTensor* sd = index.find_shared(layer, ExpertKind::Down);
            if (!sg || !sd) { emit("ERR no shared expert in layer\n"); continue; }
            std::vector<float> x(size_t(sg->k)), out(size_t(sd->n_rows));
            for (size_t i = 0; i < x.size(); ++i) x[i] = std::sin(float(i) * 0.017f) * 0.5f + 0.1f;
            const auto t0 = std::chrono::steady_clock::now();
            if (!stream->shared_forward(layer, x.data(), sg->k, out.data(), sd->n_rows, err)) {
                emit("ERR shared expert: %s\n", err.c_str()); continue;
            }
            double sum = 0; for (float v : out) sum += v;
            const double us = std::chrono::duration<double, std::micro>(
                std::chrono::steady_clock::now() - t0).count();
            emit("SHEXP %d sum=%.6f first=%.6f us=%.0f\n", layer, sum, out[0], us);
            continue;
        }
        if (line.rfind("EVAL", 0) == 0) {
            // Teacher forcing: run the given token sequence and emit top-K
            // logprobs after every position, without sampling. This compares
            // the forwards of two engines even when greedy decoding diverges
            // on a near-tie and cascades into different sequences.
            const std::vector<int> ids = parse_ids(line);
            if (ids.empty()) { emit("ERR bad EVAL\n"); continue; }
            forward->reset();
            bool ok = true;
            for (size_t i = 0; i < ids.size(); ++i) {
                if (ids[i] < 0 || ids[i] >= n_vocab) {
                    emit("ERR EVAL token out of range\n"); ok = false; break;
                }
                if (!forward->forward(ids[i], logits.data(), err)) {
                    emit("ERR EVAL forward: %s\n", err.c_str()); ok = false; break;
                }
                emit_top_logprobs(logits.data(), n_vocab, opt.lp, argmax_of(logits.data(), n_vocab));
            }
            emit("DONE %d %zu 0.0 0.0 %s\n", ok ? int(ids.size()) : 0, ids.size(),
                 ok ? "length" : "nonfinite");
            continue;
        }
        if (line.rfind("GEN", 0) != 0) continue;

        GenRequest req;
        if (!parse_gen(line, req) || req.prompt.empty()) {
            emit("ERR bad GEN line\n"); continue;
        }
        bool bad_id = false;
        for (int t : req.prompt) if (t < 0 || t >= n_vocab) { bad_id = true; break; }
        if (bad_id) {
            emit("ERR prompt token out of range\n");
            emit("DONE 0 %d 0.0 0.0 nonfinite\n", int(req.prompt.size()));
            continue;
        }
        const int room = opt.ctx - int(req.prompt.size());
        if (room < 0) {
            emit("ERR prompt longer than context\n");
            emit("DONE 0 %d 0.0 0.0 length\n", int(req.prompt.size()));
            continue;
        }
        req.max_new = std::min(req.max_new, room);
        forward->reset();

        const auto tp0 = std::chrono::steady_clock::now();
        bool ok = true;
        for (int t : req.prompt) {
            if (!forward->forward_dump(t, logits.data(), dump, err)) { ok = false; break; }
        }
        const double prompt_ms = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - tp0).count();
        if (opt.debug) {
            std::vector<int> top(n_vocab);
            std::iota(top.begin(), top.end(), 0);
            std::partial_sort(top.begin(), top.begin() + std::min(10, n_vocab), top.end(),
                              [&](int a, int b) { return logits[a] > logits[b]; });
            std::fprintf(stderr, "DEBUG top logits after prompt:");
            for (int i = 0; i < std::min(10, n_vocab); ++i)
                std::fprintf(stderr, " %d=%.4f", top[i], logits[top[i]]);
            std::fprintf(stderr, "  (nan=%d)\n", (int)std::count_if(logits.begin(), logits.end(),
                         [](float x) { return std::isnan(x); }));
        }
        emit("PP %zu %zu %.1f %.1f\n", req.prompt.size(), req.prompt.size(), prompt_ms,
             prompt_ms > 0 ? 1000.0 * req.prompt.size() / prompt_ms : 0.0);
        if (!ok) {
            emit("ERR forward failed: %s\n", err.c_str());
            emit("DONE 0 %zu %.1f 0.0 nonfinite\n", req.prompt.size(), prompt_ms);
            continue;
        }

        const auto tg0 = std::chrono::steady_clock::now();
        int generated = 0;
        const char* finish = "length";
        for (int i = 0; i < req.max_new; ++i) {
            int next = sample(logits.data(), n_vocab, req);
            if (next == eos) { finish = "stop"; break; }
            emit("T %d\n", next);
            emit_top_logprobs(logits.data(), n_vocab, opt.lp, next);
            ++generated;
            if (!forward->forward(next, logits.data(), err)) { finish = "nonfinite"; break; }
        }
        const double decode_ms = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - tg0).count();
        emit("DONE %d %zu %.1f %.1f %s\n", generated, req.prompt.size(), prompt_ms, decode_ms, finish);
    }

    if (dump) std::fclose(dump);
    if (trace) std::fclose(trace);
    forward.reset();
    router.reset();
    stream.reset();
    vkDestroyDevice(dev, nullptr);
    vkDestroyInstance(inst, nullptr);
    return 0;
}
