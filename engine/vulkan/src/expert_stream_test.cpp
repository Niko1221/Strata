// Integration test for the Vulkan expert-streaming tier.
// Streams real experts from the GGUF on disk into a bounded VRAM slot pool,
// runs the Vulkan matvec shaders (Q4_K gate/up, Q6_K down), and compares every
// output against ggml-compatible CPU references read straight from the file.
// A third phase runs the whole MoE expert forward gate -> SiLU -> up -> down
// and checks every stage.
#include "strata/vulkan/expert_stream.hpp"
#include "strata/vulkan/router.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include <fcntl.h>
#include <unistd.h>

using namespace strata::vulkan;

namespace {

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

bool make_device(VkPhysicalDevice phys, uint32_t qfam, VkDevice& dev, VkQueue& queue) {
    float prio = 1.0f;
    VkDeviceQueueCreateInfo qi{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
    qi.queueFamilyIndex = qfam; qi.queueCount = 1; qi.pQueuePriorities = &prio;
    VkDeviceCreateInfo di{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
    di.queueCreateInfoCount = 1; di.pQueueCreateInfos = &qi;
    if (vkCreateDevice(phys, &di, nullptr, &dev) != VK_SUCCESS) return false;
    vkGetDeviceQueue(dev, qfam, 0, &queue);
    return true;
}

bool read_expert(int fd, const ExpertIndex& index, const ExpertTensor& t, uint32_t e,
                 std::vector<uint8_t>& raw) {
    raw.resize(size_t(t.n_bytes));
    const uint64_t off = index.expert_offset(t, e);
    uint64_t done = 0;
    while (done < raw.size()) {
        ssize_t n = ::pread(fd, raw.data() + done, raw.size() - size_t(done), off_t(off + done));
        if (n <= 0) return false;
        done += uint64_t(n);
    }
    return true;
}

// A GGUF can store down experts as Q4_K (type 12) or Q6_K (type 14), even in
// the same file (this Qwen3.8-35B-A3B Q4_K_M does: 21 / 20 layers).
float ref_dot(const uint8_t* row, const float* x, uint32_t k, int32_t type) {
    return type == 14 ? q6_k_dot_reference(row, x, k) : q4_k_dot_reference(row, x, k);
}

inline float silu(float v) { return v / (1.0f + std::exp(-v)); }

struct Diff {
    uint32_t bad = 0;
    double max_abs = 0.0;
    double max_rel = 0.0;
    void add(const float* got, const float* ref, uint32_t n) {
        for (uint32_t i = 0; i < n; ++i) {
            const double a = std::fabs(double(got[i]) - double(ref[i]));
            const double rel = a / std::max<double>(1e-6, std::fabs(double(ref[i])));
            max_abs = std::max(max_abs, a);
            max_rel = std::max(max_rel, rel);
            if (rel > 2e-2 && a > 2e-3) ++bad;
        }
    }
    void print(const char* name, uint32_t values) const {
        std::printf("  %-8s max_abs=%.3g max_rel=%.3g bad=%u/%u\n",
                    name, max_abs, max_rel, bad, values);
    }
};

} // namespace

int main(int argc, char** argv) {
    std::string model, shader = "expert_q4_k_gemv.comp.spv";
    std::string down_shader = "expert_q6_k_gemv.comp.spv";
    std::string q8_shader = "expert_q8_0_gemv.comp.spv";
    std::string silu_shader = "expert_silu_mul.comp.spv";
    std::string router_gemv_shader = "router_gemv.comp.spv";
    std::string router_topk_shader = "router_topk.comp.spv";
    uint32_t n_experts = 32, slots = 16, threads = 4, rows = 128, forward = 8;
    int layer = 0;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&]() -> std::string { return (i + 1 < argc) ? argv[++i] : std::string(); };
        if (a == "--model") model = next();
        else if (a == "--shader") shader = next();
        else if (a == "--down-shader") down_shader = next();
        else if (a == "--q8-shader") q8_shader = next();
        else if (a == "--silu-shader") silu_shader = next();
        else if (a == "--router-gemv-shader") router_gemv_shader = next();
        else if (a == "--router-topk-shader") router_topk_shader = next();
        else if (a == "--experts") n_experts = uint32_t(std::strtoul(next().c_str(), nullptr, 10));
        else if (a == "--slots") slots = uint32_t(std::strtoul(next().c_str(), nullptr, 10));
        else if (a == "--threads") threads = uint32_t(std::strtoul(next().c_str(), nullptr, 10));
        else if (a == "--rows") rows = uint32_t(std::strtoul(next().c_str(), nullptr, 10));
        else if (a == "--forward") forward = uint32_t(std::strtoul(next().c_str(), nullptr, 10));
        else if (a == "--layer") layer = int(std::strtol(next().c_str(), nullptr, 10));
    }
    if (model.empty()) {
        std::fprintf(stderr, "usage: expert_stream_test --model <gguf> [--shader <q4.spv>] "
                             "[--down-shader <q6.spv>] [--q8-shader <q8.spv>] [--silu-shader <spv>] "
                             "[--router-gemv-shader <spv>] [--router-topk-shader <spv>] "
                             "[--experts N] [--slots N] [--threads N] [--rows N] [--forward N] [--layer N]\n");
        return 2;
    }

    std::string err;
    ExpertIndex index;
    if (!ExpertIndex::load(model, index, err)) {
        std::fprintf(stderr, "index: %s\n", err.c_str()); return 1;
    }
    std::printf("GGUF: %u layers, %u experts, %zu expert tensors\n",
                index.n_layers(), index.n_experts(), index.tensors().size());
    for (const auto& t : index.tensors())
        if (t.kind == ExpertKind::Gate) {
            std::printf("gate: layer %d  %ux%ux%u  %zu B/expert  type %d\n",
                        t.layer, t.n_rows, t.k, t.n_experts, size_t(t.n_bytes), t.ggml_type);
            break;
        }
    if (const ExpertTensor* dn = index.find(layer, ExpertKind::Down))
        std::printf("down: layer %d  %ux%ux%u  %zu B/expert  type %d\n",
                    dn->layer, dn->n_rows, dn->k, dn->n_experts, size_t(dn->n_bytes), dn->ggml_type);

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
        std::fprintf(stderr, "no Intel Vulkan device with a compute queue\n"); return 1;
    }
    VkPhysicalDeviceProperties pp{};
    vkGetPhysicalDeviceProperties(phys, &pp);
    std::printf("GPU: %s (api %u.%u)\n", pp.deviceName,
                VK_VERSION_MAJOR(pp.apiVersion), VK_VERSION_MINOR(pp.apiVersion));

    VkDevice dev = VK_NULL_HANDLE;
    VkQueue queue = VK_NULL_HANDLE;
    if (!make_device(phys, qfam, dev, queue)) {
        std::fprintf(stderr, "vkCreateDevice failed\n"); return 1;
    }

    StreamConfig cfg;
    cfg.cache_slots = slots;
    cfg.io_threads = threads;
    cfg.prefetch_next_layer = true;
    auto stream = ExpertStream::create(index, phys, dev, qfam, queue, shader, down_shader,
                                       silu_shader, q8_shader, "", "", "", cfg, err);
    if (!stream) { std::fprintf(stderr, "stream: %s\n", err.c_str()); return 1; }

    const ExpertTensor* gate = index.find(layer, ExpertKind::Gate);
    const ExpertTensor* up = index.find(layer, ExpertKind::Up);
    const ExpertTensor* down = index.find(layer, ExpertKind::Down);
    if (!gate || !up || !down) { std::fprintf(stderr, "missing gate/up/down tensor\n"); return 1; }

    int fd = ::open(model.c_str(), O_RDONLY | O_CLOEXEC);

    // ---- Phase 1: Q4_K gate matvec over n_experts ---------------------------
    uint32_t gate_rows = std::min(rows, gate->n_rows);
    const uint32_t k = gate->k;
    std::vector<float> x(k);
    for (uint32_t i = 0; i < k; ++i)
        x[i] = std::sin(float(i) * 0.017f) * 0.5f + 0.1f;

    Diff gate_diff;
    for (uint32_t e = 0; e < n_experts; ++e) {
        ExpertStream::Key key{layer, ExpertKind::Gate, e};
        uint32_t slot = 0;
        if (!stream->ensure(key, slot, err)) { std::fprintf(stderr, "ensure: %s\n", err.c_str()); return 1; }
        std::vector<float> got(gate_rows);
        if (!stream->matvec(key, slot, x.data(), k, got.data(), gate_rows, err)) {
            std::fprintf(stderr, "matvec: %s\n", err.c_str()); return 1;
        }

        std::vector<uint8_t> raw;
        if (!read_expert(fd, index, *gate, e, raw)) { std::fprintf(stderr, "short GGUF read\n"); return 1; }
        std::vector<float> ref(gate_rows);
        for (uint32_t r = 0; r < gate_rows; ++r)
            ref[r] = q4_k_dot_reference(raw.data() + size_t(r) * (size_t(k) / 256) * 144, x.data(), k);
        gate_diff.add(got.data(), ref.data(), gate_rows);

        if (e < 4 || gate_diff.bad)
            std::printf("expert %4u slot %3u  got[0]=%+.5f ref[0]=%+.5f\n",
                        e, slot, double(got[0]), double(ref[0]));

        if (const ExpertTensor* nx = index.find(layer + 1, ExpertKind::Gate))
            stream->prefetch({layer + 1, ExpertKind::Gate, e % nx->n_experts});
    }
    std::printf("\nQ4_K gate matvec: "); gate_diff.print("gate", n_experts * gate_rows);

    // ---- Phase 2: down matvec over n_experts (Q4_K or Q6_K per layer) -------
    const uint32_t down_rows = std::min(rows, down->n_rows);
    const uint32_t dk = down->k;
    std::vector<float> xd(dk);
    for (uint32_t i = 0; i < dk; ++i)
        xd[i] = std::cos(float(i) * 0.013f) * 0.4f - 0.2f;

    Diff down_diff;
    for (uint32_t e = 0; e < n_experts; ++e) {
        ExpertStream::Key key{layer, ExpertKind::Down, e};
        uint32_t slot = 0;
        if (!stream->ensure(key, slot, err)) { std::fprintf(stderr, "ensure down: %s\n", err.c_str()); return 1; }
        std::vector<float> got(down_rows);
        if (!stream->matvec(key, slot, xd.data(), dk, got.data(), down_rows, err)) {
            std::fprintf(stderr, "matvec down: %s\n", err.c_str()); return 1;
        }

        std::vector<uint8_t> raw;
        if (!read_expert(fd, index, *down, e, raw)) { std::fprintf(stderr, "short GGUF read\n"); return 1; }
        std::vector<float> ref(down_rows);
        const uint32_t stride = (dk / 256u) * (down->ggml_type == 14 ? 210u : 144u);
        for (uint32_t r = 0; r < down_rows; ++r)
            ref[r] = ref_dot(raw.data() + size_t(r) * stride, xd.data(), dk, down->ggml_type);
        down_diff.add(got.data(), ref.data(), down_rows);

        if (e < 4 || down_diff.bad)
            std::printf("down   %4u slot %3u  got[0]=%+.5f ref[0]=%+.5f\n",
                        e, slot, double(got[0]), double(ref[0]));
    }
    std::printf("down matvec (type %d): ", down->ggml_type);
    down_diff.print("down", n_experts * down_rows);

    // ---- Phase 3: full expert forward gate -> SiLU -> up -> down -----------
    uint32_t fwd = std::min(forward, std::min(gate->n_experts, down->n_experts));
    Diff fg, fu, fh, fwd_diff;
    std::vector<float> gg(gate->n_rows), ug(up->n_rows), hg(down->k), og(down->n_rows);
    std::vector<float> rg(gate->n_rows), ru(up->n_rows), rh(down->k), ro(down->n_rows);
    for (uint32_t e = 0; e < fwd; ++e) {
        const ExpertStream::Key kg{layer, ExpertKind::Gate, e};
        const ExpertStream::Key ku{layer, ExpertKind::Up, e};
        const ExpertStream::Key kd{layer, ExpertKind::Down, e};
        uint32_t sg = 0, su = 0, sd = 0;
        if (!stream->ensure(kg, sg, err) || !stream->ensure(ku, su, err) ||
            !stream->ensure(kd, sd, err)) { std::fprintf(stderr, "ensure forward: %s\n", err.c_str()); return 1; }
        if (!stream->matvec(kg, sg, x.data(), k, gg.data(), gate->n_rows, err) ||
            !stream->matvec(ku, su, x.data(), k, ug.data(), up->n_rows, err))
            { std::fprintf(stderr, "matvec gate/up: %s\n", err.c_str()); return 1; }
        // SiLU*up now runs on the Vulkan queue too (first activation kernel).
        if (!stream->silu_mul(gg.data(), ug.data(), hg.data(), down->k, err)) {
            std::fprintf(stderr, "silu_mul: %s\n", err.c_str()); return 1;
        }
        if (!stream->matvec(kd, sd, hg.data(), dk, og.data(), down->n_rows, err)) {
            std::fprintf(stderr, "matvec down: %s\n", err.c_str()); return 1;
        }

        // CPU reference of every stage, straight from the GGUF bytes.
        std::vector<uint8_t> rg_raw, ru_raw, rd_raw;
        if (!read_expert(fd, index, *gate, e, rg_raw) || !read_expert(fd, index, *up, e, ru_raw) ||
            !read_expert(fd, index, *down, e, rd_raw)) { std::fprintf(stderr, "short GGUF read\n"); return 1; }
        const uint32_t gs = (k / 256u) * (gate->ggml_type == 14 ? 210u : 144u);
        const uint32_t us = (k / 256u) * (up->ggml_type == 14 ? 210u : 144u);
        const uint32_t ds = (dk / 256u) * (down->ggml_type == 14 ? 210u : 144u);
        for (uint32_t r = 0; r < gate->n_rows; ++r)
            rg[r] = ref_dot(rg_raw.data() + size_t(r) * gs, x.data(), k, gate->ggml_type);
        for (uint32_t r = 0; r < up->n_rows; ++r)
            ru[r] = ref_dot(ru_raw.data() + size_t(r) * us, x.data(), k, up->ggml_type);
        for (uint32_t i = 0; i < down->k; ++i)
            rh[i] = silu(rg[i]) * ru[i];
        for (uint32_t r = 0; r < down->n_rows; ++r)
            ro[r] = ref_dot(rd_raw.data() + size_t(r) * ds, rh.data(), dk, down->ggml_type);

        fg.add(gg.data(), rg.data(), gate->n_rows);
        fu.add(ug.data(), ru.data(), up->n_rows);
        fh.add(hg.data(), rh.data(), down->k);
        fwd_diff.add(og.data(), ro.data(), down->n_rows);
        if (e < 2)
            std::printf("forward expert %u: out[0] got=%+.5f ref=%+.5f\n",
                        e, double(og[0]), double(ro[0]));
    }
    std::printf("\nfull expert forward (%u experts):\n", fwd);
    fg.print("gate", fwd * gate->n_rows);
    fu.print("up", fwd * up->n_rows);
    fh.print("silu*up", fwd * down->k);
    fwd_diff.print("down", fwd * down->n_rows);

    // ---- Phase 4: router-guided prefetch from the real F32 router matrix ----
    // The native engine will do this on the current hidden state; the test uses
    // a deterministic vector so the mechanism itself is verifiable without a
    // full model forward. The router weights come from the GGUF on disk.
    const RouterTensor* rt = index.find_router(layer + 1);
    if (rt && rt->k <= 2048 && rt->n_experts <= 256) {
        Router::Limits lim;
        lim.max_experts = 256; lim.max_k = 2048; lim.max_top_k = 64; lim.max_tokens = 64;
        auto router = Router::create(phys, dev, qfam, queue, router_gemv_shader,
                                     router_topk_shader, lim, err);
        if (!router) { std::fprintf(stderr, "router: %s\n", err.c_str()); return 1; }

        const size_t rw = size_t(rt->k) * rt->n_experts * sizeof(float);
        std::vector<float> W(size_t(rt->k) * rt->n_experts);
        uint64_t done = 0;
        while (done < rw) {
            ssize_t n = ::pread(fd, reinterpret_cast<uint8_t*>(W.data()) + done,
                                rw - done, off_t(rt->file_offset + done));
            if (n <= 0) break;
            done += uint64_t(n);
        }
        if (done == rw) {
            std::vector<float> hidden(rt->k);
            for (uint32_t i = 0; i < rt->k; ++i)
                hidden[i] = std::sin(float(i) * 0.013f) * 0.4f;
            std::vector<float> logits(rt->n_experts);
            int32_t ids[64] = {};
            float wt[64] = {};
            if (!router->gemv(W.data(), hidden.data(), logits.data(), rt->n_experts, rt->k, err) ||
                !router->topk(logits.data(), 1, rt->n_experts, 8, ids, wt, err)) {
                std::fprintf(stderr, "route: %s\n", err.c_str()); return 1;
            }
            std::printf("\nrouter-guided prefetch (layer %d):", layer + 1);
            for (int i = 0; i < 8; ++i) std::printf(" %d", ids[i]);
            std::printf("\n");

            const StreamStats before = stream->stats();
            stream->prefetch_route(layer + 1, ids, 8);
            uint32_t served = 0;
            for (int i = 0; i < 8; ++i) {
                uint32_t slot = 0;
                if (!stream->ensure({layer + 1, ExpertKind::Gate, uint32_t(ids[i])}, slot, err)) {
                    std::fprintf(stderr, "ensure routed: %s\n", err.c_str()); return 1;
                }
                if (stream->stats().hits > before.hits + i) ++served;
            }
            const StreamStats after = stream->stats();
            std::printf("prefetched=%llu  served-from-cache=%u/8  misses=%llu\n",
                        (unsigned long long)(after.prefetches - before.prefetches), served,
                        (unsigned long long)(after.misses - before.misses));
            if (after.prefetches == before.prefetches || served != 8) {
                std::fprintf(stderr, "FAILED: router-guided prefetch did not serve the routed experts\n");
                return 1;
            }
        }
        router.reset();
    }

    // ---- Phase 5: shared expert, including the structure traps --------------
    // The SYCL parity test pins the structure by computing the WRONG readings
    // and requiring them to differ. We do the same here on real GGUF tensors:
    // silu on gate (not up), one scalar sigmoid gate (not per-dimension), and
    // the shared output added unweighted to the routed sum (the combine itself
    // is still host-side in the engine, so only the first two are checked here).
    const ExpertTensor* sg = index.find_shared(layer, ExpertKind::Gate);
    const ExpertTensor* su = index.find_shared(layer, ExpertKind::Up);
    const ExpertTensor* sd = index.find_shared(layer, ExpertKind::Down);
    const RouterTensor* sgi = index.find_shared_gate(layer);
    if (!sg || !su || !sd || !sgi) {
        std::fprintf(stderr, "FAILED: no shared expert tensors in layer %d\n", layer);
        return 1;
    }
    std::printf("\nshared expert: gate %ux%u type %d, down %ux%u type %d, scalar gate %u F32\n",
                sg->n_rows, sg->k, sg->ggml_type, sd->n_rows, sd->k, sd->ggml_type, sgi->k);

    std::vector<float> sgot(sd->n_rows);
    if (!stream->shared_forward(layer, x.data(), k, sgot.data(), sd->n_rows, err)) {
        std::fprintf(stderr, "shared_forward: %s\n", err.c_str()); return 1;
    }

    std::vector<uint8_t> sgraw, suraw, sdraw;
    if (!read_expert(fd, index, *sg, 0, sgraw) || !read_expert(fd, index, *su, 0, suraw) ||
        !read_expert(fd, index, *sd, 0, sdraw)) {
        std::fprintf(stderr, "short GGUF read (shared)\n"); return 1;
    }
    const uint32_t sgs = (k / 256u) * (sg->ggml_type == 14 ? 210u : 144u);
    const uint32_t sus = (k / 256u) * (su->ggml_type == 14 ? 210u : 144u);
    const uint32_t sds = (sd->k / 256u) * (sd->ggml_type == 14 ? 210u : 144u);
    std::vector<float> gr(sg->n_rows), ur(su->n_rows), hr(sd->k), oraw(sd->n_rows);
    for (uint32_t r = 0; r < sg->n_rows; ++r)
        gr[r] = ref_dot(sgraw.data() + size_t(r) * sgs, x.data(), k, sg->ggml_type);
    for (uint32_t r = 0; r < su->n_rows; ++r)
        ur[r] = ref_dot(suraw.data() + size_t(r) * sus, x.data(), k, su->ggml_type);
    for (uint32_t i = 0; i < sd->k; ++i) hr[i] = silu(gr[i]) * ur[i];
    for (uint32_t r = 0; r < sd->n_rows; ++r)
        oraw[r] = ref_dot(sdraw.data() + size_t(r) * sds, hr.data(), sd->k, sd->ggml_type);

    // The scalar gate: read F32 from the GGUF and dot with the same x.
    std::vector<float> gw(sgi->k);
    {
        const uint64_t n = uint64_t(sgi->k) * sizeof(float);
        uint64_t done = 0;
        while (done < n) {
            ssize_t got = ::pread(fd, reinterpret_cast<uint8_t*>(gw.data()) + done,
                                  n - done, off_t(sgi->file_offset + done));
            if (got <= 0) break;
            done += uint64_t(got);
        }
        if (done != n) { std::fprintf(stderr, "short shared gate read\n"); return 1; }
    }
    double gdot = 0;
    for (uint32_t i = 0; i < sgi->k; ++i) gdot += double(gw[i]) * double(x[i]);
    const float sgate = float(1.0 / (1.0 + std::exp(-gdot)));
    std::vector<float> want(oraw);
    for (float& v : want) v *= sgate;

    Diff shared_diff;
    shared_diff.add(sgot.data(), want.data(), sd->n_rows);
    std::printf("shared forward: "); shared_diff.print("shexp", sd->n_rows);
    std::printf("  scalar gate=%.6f (dot=%.4f)  out[0] got=%+.5f ref=%+.5f\n",
                double(sgate), gdot, double(sgot[0]), double(want[0]));

    auto rel_l1 = [](const std::vector<float>& a, const std::vector<float>& b) {
        double d = 0, m = 0;
        for (size_t i = 0; i < a.size(); ++i) { d += std::fabs(double(a[i]) - double(b[i])); m += std::fabs(double(a[i])); }
        return d / (m > 1e-30 ? m : 1e-30);
    };
    // Trap 1: silu on up instead of gate must be distinguishable.
    std::vector<float> wrong_h(sd->k), wrong_out(sd->n_rows);
    for (uint32_t i = 0; i < sd->k; ++i) wrong_h[i] = silu(ur[i]) * gr[i];
    for (uint32_t r = 0; r < sd->n_rows; ++r)
        wrong_out[r] = float(ref_dot(sdraw.data() + size_t(r) * sds, wrong_h.data(), sd->k, sd->ggml_type) * sgate);
    const double silu_sep = rel_l1(want, wrong_out);
    // Trap 2: a per-dimension (elementwise) gate must be distinguishable.
    std::vector<float> elem(want);
    for (uint32_t i = 0; i < sd->n_rows; ++i) elem[i] *= 1.0f / (1.0f + std::exp(-x[i % k]));
    const double elem_sep = rel_l1(want, elem);
    // Trap 3: dropping the scalar gate must be distinguishable.
    const double gate_sep = rel_l1(want, oraw);
    std::printf("  traps: silu-on-up %.2f%% apart, elementwise gate %.2f%% apart, ungated %.2f%% apart\n",
                silu_sep * 100, elem_sep * 100, gate_sep * 100);
    if (!(silu_sep > 0.01) || !(elem_sep > 0.01) || !(gate_sep > 0.01)) {
        std::fprintf(stderr, "FAILED: a shared-expert structure trap is not observable\n");
        return 1;
    }

    ::close(fd);

    auto st = stream->stats();
    std::printf("\nstats: hits=%llu misses=%llu prefetches=%llu MB=%.1f io_ms=%llu\n",
                (unsigned long long)st.hits, (unsigned long long)st.misses,
                (unsigned long long)st.prefetches, double(st.bytes_read) / 1e6,
                (unsigned long long)st.io_ms);

    const uint32_t bad = gate_diff.bad + down_diff.bad + fg.bad + fu.bad + fh.bad +
                         fwd_diff.bad + shared_diff.bad;
    const uint32_t total = n_experts * gate_rows + n_experts * down_rows +
                           fwd * (gate->n_rows + up->n_rows + down->k + down->n_rows) +
                           sd->n_rows;
    std::printf("TOTAL bad_rows=%u/%u\n", bad, total);

    stream.reset();
    vkDestroyDevice(dev, nullptr);
    vkDestroyInstance(inst, nullptr);

    if (bad) { std::fprintf(stderr, "FAILED: %u rows differ from CPU reference\n", bad); return 1; }
    std::printf("PASS\n");
    return 0;
}
