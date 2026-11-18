// APR (prism.hadamard) metadata of a GGUF: Agention's Gyro files store some routed-expert matrices with a Hadamard rotation folded
// into their input columns, so their inputs must be rotated the same way before the dot: x' = H_128 (s * x) / sqrt(128) per block
// of 128 columns (the normalized Sylvester-Walsh-Hadamard matrix, natural order; s the file's explicit +-1 sign vector of the
// input width). The keys (Gyro-S and Gyro-M write the same ones): prism.hadamard.{version, block_size, transform, axis, sign_mode,
// sign_widths, sign_values, weight_names}. The main engine reads the same data from the pack's hadamard.txt (tools/iq_pack.py,
// src/kernels/cpu/hadamard.cpp). Pure host code.
#pragma once

#include <set>
#include <stdexcept>
#include <string>
#include <vector>

#include "gguf.hpp"

namespace fnx {

constexpr int HAD_BLOCK = 128;   // the one rotation block the kernels run (one thread a column, 4 warps)

struct HadSpec {
    int block = 0;                         // 0: the file rotates nothing
    std::vector<float> signs_ff, signs_n;  // the sign vectors of the FF-wide (down) and N-wide (gate/up) inputs
    std::set<std::string> names;           // the tensors stored rotated
    bool rotated(const std::string& name) const { return names.count(name) != 0; }
};

// throws on what the split engine cannot run (another block, transform, axis or sign mode, or sign widths other than FF and N)
inline HadSpec had_spec(const Gguf& g, int ff, int n) {
    HadSpec h;
    if (!g.has_key("prism.hadamard.version")) return h;
    const std::string what = "prism.hadamard: ";
    if (g.i("prism.hadamard.version") != 1) throw std::runtime_error(what + "version " + std::to_string(g.i("prism.hadamard.version")) + " (this engine reads 1)");
    if (g.i("prism.hadamard.block_size") != HAD_BLOCK)
        throw std::runtime_error(what + "block " + std::to_string(g.i("prism.hadamard.block_size")) + " (the split engine rotates blocks of 128)");
    if (g.s("prism.hadamard.transform") != "normalized-sylvester-walsh-hadamard" || g.s("prism.hadamard.axis") != "input-last-dimension" ||
        g.s("prism.hadamard.sign_mode") != "explicit")
        throw std::runtime_error(what + "transform '" + g.s("prism.hadamard.transform") + "', axis '" + g.s("prism.hadamard.axis") + "', signs '" +
                                 g.s("prism.hadamard.sign_mode") + "' (the split engine runs the normalized Sylvester transform on the input "
                                 "columns with explicit signs)");
    const std::vector<int64_t> widths = g.ia("prism.hadamard.sign_widths"), vals = g.ia("prism.hadamard.sign_values");
    size_t at = 0;
    for (const int64_t w : widths) {
        if (w <= 0 || w % HAD_BLOCK || at + (size_t) w > vals.size()) throw std::runtime_error(what + "a sign width of " + std::to_string(w));
        std::vector<float> s((size_t) w);
        for (int64_t i = 0; i < w; ++i) {
            const int64_t v = vals[at + (size_t) i];
            if (v != 1 && v != -1) throw std::runtime_error(what + "a sign value of " + std::to_string(v));
            s[(size_t) i] = (float) v;
        }
        at += (size_t) w;
        if (w == ff) h.signs_ff = std::move(s);
        else if (w == n) h.signs_n = std::move(s);
        else throw std::runtime_error(what + "signs for an input " + std::to_string(w) + " wide (the experts' inputs are " + std::to_string(ff) +
                                      " and " + std::to_string(n) + ")");
    }
    if (at != vals.size()) throw std::runtime_error(what + "more sign values than the widths name");
    for (const std::string& name : g.sa("prism.hadamard.weight_names")) {
        const bool down = name.find(".ffn_down_exps.weight") != std::string::npos;
        const bool gu = name.find(".ffn_gate_exps.weight") != std::string::npos || name.find(".ffn_up_exps.weight") != std::string::npos;
        if (!down && !gu) throw std::runtime_error(what + name + " is rotated (the split engine rotates the routed experts' inputs only)");
        if ((down ? h.signs_ff : h.signs_n).empty()) throw std::runtime_error(what + name + " is rotated, but the file has no signs for its input");
        h.names.insert(name);
    }
    for (const std::string& name : h.names) {   // gate and up read one activation: both rotated or neither
        const size_t p = name.find(".ffn_gate_exps.weight");
        if (p == std::string::npos) continue;
        if (!h.rotated(name.substr(0, p) + ".ffn_up_exps.weight")) throw std::runtime_error(what + name + " is rotated and its up matrix is not");
    }
    for (const std::string& name : h.names) {
        const size_t p = name.find(".ffn_up_exps.weight");
        if (p != std::string::npos && !h.rotated(name.substr(0, p) + ".ffn_gate_exps.weight"))
            throw std::runtime_error(what + name + " is rotated and its gate matrix is not");
    }
    h.block = h.names.empty() ? 0 : HAD_BLOCK;
    return h;
}

}  // namespace fnx
