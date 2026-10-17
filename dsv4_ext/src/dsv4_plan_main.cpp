// dsv4_plan: reads the real DeepSeek-V4 GGUF shards (header only) and prints config, per-layer expert
// types/bytes, and the VRAM/expert-slot plan.  CPU only; VRAM is read from nvidia-smi unless given.
#include "dsv4/config.hpp"
#include "dsv4/gguf_header.hpp"
#include "dsv4/mem_plan.hpp"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

using namespace dsv4;

static void usage() {
    std::puts(
        "usage: dsv4_plan --model FIRST_SHARD.gguf [--model OTHER.gguf ...] [options]\n"
        "  (a name like X-00001-of-00003.gguf auto-loads all shards)\n"
        "  --expert-vram-pct P|auto     share of the VRAM left after fixed costs (default auto)\n"
        "  --expert-cache N             absolute slots (overrides pct)\n"
        "  --expert-vram-reserve-mib M  default 1024\n"
        "  --vram-total-mib N --vram-free-mib N   (default: nvidia-smi, else required)\n"
        "  --max-context N              default 4096 (KV estimate, ROUGH)\n"
        "  --kv-mib N  --act-mib N  --prefill-mib N  --scratch-mib N   override estimates\n"
        "  --no-mtp                     do not count the extra (MTP) layer's experts as fixed VRAM\n"
        "  --sweep                      table for P = 0/25/50/75/100\n"
        "  --dump-tensors               list every tensor (name, dims, type, bytes)");
}

static std::vector<std::string> expand_shards(const std::string& p) {
    const size_t of = p.rfind("-of-");
    if (of == std::string::npos || of < 5 || p.size() < of + 4 + 5 + 5) return {p};
    const std::string a = p.substr(of - 5, 5), b = p.substr(of + 4, 5);
    if (a.find_first_not_of("0123456789") != std::string::npos || b.find_first_not_of("0123456789") != std::string::npos) return {p};
    const int n = std::atoi(b.c_str());
    std::vector<std::string> out;
    for (int i = 1; i <= n; ++i) {
        char buf[16];
        std::snprintf(buf, sizeof buf, "%05d", i);
        out.push_back(p.substr(0, of - 5) + buf + p.substr(of));
    }
    return out;
}

static bool nvidia_smi(long long& total_mib, long long& free_mib) {
    FILE* f = popen("nvidia-smi --query-gpu=memory.total,memory.free --format=csv,noheader,nounits 2>/dev/null", "r");
    if (!f) return false;
    long long t = 0, fr = 0;
    const bool ok = std::fscanf(f, "%lld, %lld", &t, &fr) == 2;
    pclose(f);
    if (ok) { total_mib = t; free_mib = fr; }
    return ok;
}

int main(int argc, char** argv) {
    std::vector<std::string> models;
    std::string pct_s = "auto";
    long long abs_slots = -1, reserve_mib = 1024, vt = -1, vf = -1, max_ctx = 4096;
    long long kv_mib = -1, act_mib = -1, prefill_mib = -1, scratch_mib = -1;
    bool sweep = false, dump = false, no_mtp = false;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&]() -> const char* { if (i + 1 >= argc) { usage(); std::exit(2); } return argv[++i]; };
        if (a == "--model") { for (auto& s : expand_shards(next())) models.push_back(s); }
        else if (a == "--expert-vram-pct") pct_s = next();
        else if (a == "--expert-cache") abs_slots = std::atoll(next());
        else if (a == "--expert-vram-reserve-mib") reserve_mib = std::atoll(next());
        else if (a == "--vram-total-mib") vt = std::atoll(next());
        else if (a == "--vram-free-mib") vf = std::atoll(next());
        else if (a == "--max-context") max_ctx = std::atoll(next());
        else if (a == "--kv-mib") kv_mib = std::atoll(next());
        else if (a == "--act-mib") act_mib = std::atoll(next());
        else if (a == "--prefill-mib") prefill_mib = std::atoll(next());
        else if (a == "--scratch-mib") scratch_mib = std::atoll(next());
        else if (a == "--no-mtp") no_mtp = true;
        else if (a == "--sweep") sweep = true;
        else if (a == "--dump-tensors") dump = true;
        else { usage(); return a == "--help" || a == "-h" ? 0 : 2; }
    }
    if (models.empty()) { usage(); return 2; }

    GgufHeader h; std::string err;
    if (!gguf_read_header(models, h, err)) { std::fprintf(stderr, "error: %s\n", err.c_str()); return 1; }
    Dsv4Config c;
    if (!config_from_gguf(h, c, err)) { std::fprintf(stderr, "error: %s\n", err.c_str()); return 1; }
    std::printf("%zu shard(s), %zu tensors\n%s", h.files.size(), h.tensors.size(), config_summary(c).c_str());
    if (h.tensors.empty()) { std::fprintf(stderr, "error: no tensors found: pass ALL shards (the first one holds metadata only)\n"); return 1; }

    if (dump) {
        for (const GgufTensor& t : h.tensors) {
            std::string d;
            for (size_t k = 0; k < t.dims.size(); ++k) d += (k ? "x" : "") + std::to_string(t.dims[k]);
            std::printf("T %-48s %-18s %-8s %12llu\n", t.name.c_str(), d.c_str(), ggml_type_str(t.type).c_str(), (unsigned long long) t.nbytes);
        }
    }

    ExpertInventory inv;
    if (!inventory_from_gguf(h, c, inv, err)) { std::fprintf(stderr, "error: %s\n", err.c_str()); return 1; }
    if (inv.unknown_type_tensors) std::printf("WARNING: %d tensors have an unknown ggml type (bytes not counted)\n", inv.unknown_type_tensors);
    if (inv.layers_without_experts) std::printf("WARNING: %d layers have no routed-expert tensors (naming differs from '*_exps'?)\n", inv.layers_without_experts);

    std::puts("\nexpert bytes per layer (gate/up/down types):");
    for (int l = 0; l < c.n_layer;) {
        int e = l;
        while (e + 1 < c.n_layer && (e + 1 != c.n_hash_layers) && inv.types[(size_t) e + 1] == inv.types[(size_t) l] && inv.bytes_per_expert[(size_t) e + 1] == inv.bytes_per_expert[(size_t) l]) ++e;
        std::printf("  layers %2d-%2d  %-26s %9.3f MiB/expert  (%s)\n", l, e, inv.types[(size_t) l].c_str(),
                    inv.bytes_per_expert[(size_t) l] / 1048576.0, l < c.n_hash_layers ? "hash routing" : "learned router");
        l = e + 1;
    }
    std::printf("routed experts: %.2f GiB | extra/MTP experts: %.2f GiB | everything else: %.2f GiB\n",
                inv.total_expert_bytes / 1073741824.0, inv.extra_expert_bytes / 1073741824.0, inv.other_bytes / 1073741824.0);

    long long total = vt, fr = vf;
    if (total < 0 || fr < 0) {
        long long t2 = 0, f2 = 0;
        if (nvidia_smi(t2, f2)) { if (total < 0) total = t2; if (fr < 0) fr = f2; std::puts("VRAM read from nvidia-smi"); }
        else { std::fprintf(stderr, "error: nvidia-smi unavailable: pass --vram-total-mib and --vram-free-mib\n"); return 1; }
    }

    // ROUGH estimates (fp16 latent KV; sliding window + compressed entries + indexer keys).  Override with --kv-mib.
    double kv = 0;
    for (int l = 0; l < c.n_layer; ++l) {
        const int r = c.compress_ratios[(size_t) l];
        double entries = c.sliding_window;
        if (r > 0) entries += (double) max_ctx / r;
        kv += entries * c.key_len * 2.0;
        if (r == 4) kv += (double) max_ctx / r * c.idx_key_len * 2.0;
    }
    MemPlanInput in;
    in.vram_total = total << 20; in.vram_free = fr << 20; in.reserve = reserve_mib << 20;
    in.weights = (int64_t) inv.other_bytes;
    in.kv = kv_mib >= 0 ? kv_mib << 20 : (int64_t) kv;
    in.act = (act_mib >= 0 ? act_mib : 512) << 20;
    in.prefill = (prefill_mib >= 0 ? prefill_mib : 1024) << 20;
    in.scratch = (scratch_mib >= 0 ? scratch_mib : 512) << 20;
    in.mtp = no_mtp ? 0 : (int64_t) inv.extra_expert_bytes;
    in.n_expert = c.n_expert;
    in.bytes_per_expert = inv.bytes_per_expert;
    in.ram_expert_bytes = inv.total_expert_bytes;
    in.abs_slots = abs_slots;
    in.pct = pct_s == "auto" ? -1.0 : std::atof(pct_s.c_str());

    std::printf("\n(KV/act/prefill/scratch are ROUGH estimates, max-context %lld: override with --kv-mib etc.)\n", max_ctx);
    MemPlan p = plan_expert_memory(in);
    if (!p.ok) { std::fprintf(stderr, "error: %s\n", p.err.c_str()); return 1; }
    std::printf("%s", plan_report(in, p).c_str());

    if (sweep) {
        std::puts("\n  P   slots  slots/layer  expert MiB in VRAM   % of all experts");
        for (double pp : {0.0, 25.0, 50.0, 75.0, 100.0}) {
            MemPlanInput s = in; s.pct = pp; s.abs_slots = -1;
            MemPlan q = plan_expert_memory(s);
            if (!q.ok) { std::printf("%3.0f  %s\n", pp, q.err.c_str()); continue; }
            std::printf("%3.0f  %6lld  %4d..%-4d  %10lld         %6.2f%%\n", pp, (long long) q.slots_total,
                        *std::min_element(q.slots_layer.begin(), q.slots_layer.end()),
                        *std::max_element(q.slots_layer.begin(), q.slots_layer.end()), (long long) (q.slot_bytes_total >> 20),
                        100.0 * (double) q.slot_bytes_total / (double) q.all_expert_bytes);
        }
    }
    return 0;
}
