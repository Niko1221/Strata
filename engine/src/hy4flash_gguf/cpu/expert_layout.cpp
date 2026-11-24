// src/kernels/cpu/expert_layout.cpp - plan v0.3 P6: the per-layer expert table.  See the header.
#include "strata/hy4flash_gguf/cpu/expert_layout.hpp"
#include "strata/kernels/cpu/hadamard.hpp"

#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#if defined(_MSC_VER)
#include <intrin.h>
#include <immintrin.h>
#elif defined(__x86_64__) || defined(__i386__)
#include <cpuid.h>
#endif
#include <fstream>
#include <sstream>

namespace strata::hy4_flash::cpu {
using namespace ::strata::kernels::cpu;
namespace {
ExpertLayout g_layout;
HadamardSpec rotation;

const float* rotation_signs(int width) {
    const auto it = rotation.signs.find(width);
    return it == rotation.signs.end() ? nullptr : it->second.data();
}
}

const ExpertLayout& expert_layout() { return g_layout; }

bool expert_layout_load(const std::string& pack_dir, int64_t n_layers, int64_t n_expert, std::string& err,
                        int64_t block_base, int64_t n_embd, int64_t n_ff, float swiglu_clamp, int64_t gpu_only_from) {
    ExpertLayout L;
    L.n_layers = n_layers;
    L.n_expert = n_expert;
    std::ifstream in(pack_dir + "/native_experts.txt");
    // The caller's block base and expert geometry (declared defaulting to 0, H and FF: the indices are the
    // block ids and the geometry is the compiled one, which is what every caller before hy4-flash needed).
    L.block_base = block_base;
    L.n_embd = n_embd;
    L.n_ff = n_ff;
    if (n_embd <= 0 || n_ff <= 0) {
        err = "expert_layout_load: an expert geometry of " + std::to_string(n_embd) + " x " +
              std::to_string(n_ff);
        return false;
    }
    if (!in) {
        L.total = (uint64_t) n_layers * (uint64_t) n_expert * (uint64_t) BLOB;
        rotation = {};
        g_layout = L;
        return true;
    }
#if !defined(STRATA_NATIVE_EXPERTS)
    err = "this pack has native (IQ) experts but the engine was built without STRATA_NATIVE_EXPERTS";
    return false;
#else
    L.native = true;
    L.fmt.resize((size_t) n_layers);
    L.offset.assign((size_t) n_layers, ~0ull);
    L.bytes.assign((size_t) n_layers, 0);
    L.max_blob = 0;
    std::string line;
    while (std::getline(in, line)) {
        if (line.empty() || line[0] == '#') {
            if (!line.empty() && line[0] == '#') {
                // "# strata native experts vN: ...".  A version this engine does not know may mean columns it
                // would misread, so it is refused rather than parsed as far as the known columns go.
                static const char tag[] = "# strata native experts v";
                if (line.compare(0, sizeof tag - 1, tag) == 0) {
                    L.version = std::atoi(line.c_str() + sizeof tag - 1);
                    if (L.version > kExpertLayoutVersion) {
                        err = "native_experts.txt is v" + std::to_string(L.version) + "; this engine reads up to v" +
                              std::to_string(kExpertLayoutVersion) + " (the pack was written by a newer tools/"
                              "iq_pack.py: update the engine, or repack with this one)";
                        return false;
                    }
                }
                // v3 packs record their expert count in the header; a pruned model (GSQ-RCO Coder) ships
                // fewer experts than the canonical geometry the caller passes, which is a compile-time
                // default, so the header wins.
                const size_t at = line.find("(n_expert ");
                if (at != std::string::npos) L.n_expert = std::atoll(line.c_str() + at + 10);
            }
            continue;
        }
        std::istringstream ss(line);
        long long l = -1, gt = -1, dt = -1;
        unsigned long long off = 0, blob = 0, go = 0, uo = 0, dox = 0;
        if (!(ss >> l >> gt >> dt >> off >> blob) || l < 0 || l >= n_layers) {
            err = "native_experts.txt: a malformed line: " + line;
            return false;
        }
        NativeFmt f;
        if (gpu_only_from >= 0 && l >= gpu_only_from && gt == 30 && dt == 30) {
            // hy4-flash's BF16 draft experts run on the GPU. Validate their stored
            // planes without claiming that they fit the quantized CPU pool.
            f.gu_type = f.d_type = 30;
            f.n_embd = L.n_embd; f.n_ff = L.n_ff;
            f.gu_row = (size_t) L.n_embd * 2;
            f.d_row = (size_t) L.n_ff * 2;
            f.up_off = f.gu_row * (size_t) L.n_ff;
            f.down_off = 2 * f.up_off;
            f.bytes = f.down_off + f.d_row * (size_t) L.n_embd;
        } else if (!native_fmt((int) gt, (int) dt, L.n_embd, L.n_ff, f, err)) return false;
        // The Q2_0 gate/up rows run on thread-local buffers sized by the COMPILED FF (pool.cpp), so a Q2_0
        // layer wider than that would write past them. Q2_0 packs are the compiled geometry's, so this cannot
        // happen today; refusing it here keeps it that way rather than leaving the overrun to be found later.
        if (q2_native_kernels(f.gu_type) && f.n_ff > FF) {
            err = "native_experts.txt: layer " + std::to_string(l) + " has Q2_0 gate/up of " +
                  std::to_string(f.n_ff) + " rows, and the Q2_0 row buffers hold " + std::to_string(FF);
            return false;
        }
        if (f.bytes != blob) {
            err = "native_experts.txt: layer " + std::to_string(l) + " blob is " + std::to_string(blob) +
                  " B but its formats make " + std::to_string(f.bytes);
            return false;
        }
        if (ss >> go >> uo >> dox) {   // v2 lines: the GGUF offsets
            if (L.gguf_off.empty()) L.gguf_off.assign((size_t) (3 * n_layers), 0);
            L.gguf_off[(size_t) (3 * l)] = go;
            L.gguf_off[(size_t) (3 * l + 1)] = uo;
            L.gguf_off[(size_t) (3 * l + 2)] = dox;
            std::string file;             // v3: the shard that holds this layer (a file name beside --native)
            if (ss >> file) {
                if (L.gguf_file.empty()) L.gguf_file.assign((size_t) (3 * n_layers), std::string());
                // One name covers all three roles.  When a shard boundary falls inside a layer the packer writes
                // "gate,up,down" instead (v4), an empty field meaning the --native shard.  A GGUF file name has no
                // comma, so the split is unambiguous; an engine older than v4 fails to open such a "file" loudly.
                // The comma form is from #255 (gopinath87607).
                std::vector<std::string> parts;
                for (size_t from = 0;;) {
                    const size_t comma = file.find(',', from);
                    parts.push_back(file.substr(from, comma == std::string::npos ? comma : comma - from));
                    if (comma == std::string::npos) break;
                    from = comma + 1;
                }
                if (parts.size() == 1) {
                    const std::string one = parts[0];   // not parts.assign(3, parts[0]): that aliases the element
                    parts.assign(3, one);                //   the assignment is about to overwrite
                }
                std::string extra;
                if (parts.size() != 3 || (ss >> extra)) {
                    err = "native_experts.txt: layer " + std::to_string(l) + ": the shard column is one name or "
                          "gate,up,down, not: " + line;
                    return false;
                }
                for (size_t r = 0; r < 3; ++r) L.gguf_file[(size_t) (3 * l) + r] = parts[r];
            }
        }
        f.swiglu_clamp = swiglu_clamp;
        L.fmt[(size_t) l] = f;
        L.offset[(size_t) l] = off;
        L.bytes[(size_t) l] = blob;
        if (blob > L.max_blob) L.max_blob = blob;
    }
    uint64_t at = 0;
    for (int64_t l = 0; l < n_layers; ++l) {
        if (L.offset[(size_t) l] != at) {
            err = "native_experts.txt: layer " + std::to_string(l) + " is missing or not contiguous";
            return false;
        }
        at += L.bytes[(size_t) l] * (uint64_t) L.n_expert;
    }
    L.total = at;
    // v5: Agention's Hadamard-folded experts (prism.hadamard, hadamard.hpp).  The pack must say which layers are
    // rotated and with what signs; a v5 pack without hadamard.txt (or the reverse) is refused, never run unrotated.
    {
        std::ifstream probe(pack_dir + "/hadamard.txt");
        const bool have = (bool) probe;
        if (L.version >= 5 && !have) {
            err = "native_experts.txt v5 declares Hadamard-folded experts, but " + pack_dir + "/hadamard.txt is missing";
            return false;
        }
        if (have && L.version < 5) {
            err = pack_dir + "/hadamard.txt exists but native_experts.txt is v" + std::to_string(L.version) +
                  ": the pack is inconsistent (repack with tools/iq_pack.py)";
            return false;
        }
        for (int64_t l = 0; l < n_layers; ++l) L.fmt[(size_t) l].layer = (int) l;
        HadamardSpec h;
        if (have && !hadamard_parse(pack_dir + "/hadamard.txt", n_layers, h, err)) return false;
        rotation = std::move(h);
        const HadamardSpec& hs = rotation;
        for (int64_t l = 0; hs.any() && l < n_layers; ++l) {
            NativeFmt& f = L.fmt[(size_t) l];
            f.had_block = hs.block;
            f.had_gu = hs.gu(l);
            f.had_d = hs.down(l);
            if ((f.had_gu && f.n_embd % hs.block) || (f.had_d && f.n_ff % hs.block)) {
                err = "hadamard.txt: block " + std::to_string(hs.block) + " does not divide layer " +
                      std::to_string(l) + "'s expert widths";
                return false;
            }
            // the Q2_0 rows quantize their activations with Strata's own act_quant_any, which does not rotate
            if ((f.had_gu && f.gu_type == 42) || (f.had_d && f.d_type == 42)) {
                err = "hadamard.txt: layer " + std::to_string(l) + " folds a Q2_0 expert, which this engine cannot rotate";
                return false;
            }
            f.gu_signs = f.had_gu ? rotation_signs((int) f.n_embd) : nullptr;
            f.d_signs = f.had_d ? rotation_signs((int) f.n_ff) : nullptr;
            // explicit signs are per width: a width without a vector would silently mean identity signs
            if (!hs.signs.empty() && ((f.had_gu && !f.gu_signs) || (f.had_d && !f.d_signs))) {
                err = "hadamard.txt: no sign vector for layer " + std::to_string(l) + "'s expert width";
                return false;
            }
        }
    }
    g_layout = L;
    return true;
#endif
}

}  // namespace strata::hy4_flash::cpu
