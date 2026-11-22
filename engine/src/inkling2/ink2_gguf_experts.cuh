// Expert formats of a GGUF model (included by ink2_engine.cu after the GGUF reader's Gguf / gg_row_bytes).
// A GGUF may give layers 0..MXFP8_FIRST-1 one ggml type pair (gate_up, down) and layers MXFP8_FIRST..NL-1 another, as the
// checkpoint does with NVFP4 / MXFP8 (ggml-org's IQ4_NL file keeps Q8_0 experts in layers 20-23).  One pair for every
// layer: one expert pool, as before; two pairs: the later layers in pool_mx with their own slot size.
// CBA: one type pair per layer range, the ranges fixed at MXFP8_FIRST; arbitrary per-layer types would need a pool per
// slot size (and its share of the offload budget)
#pragma once

const char* gg_type_name(uint32_t t) {   // ggml.h's enum ggml_type
    static const char* const n[] = {"F32", "F16", "Q4_0", "Q4_1", "type 4", "type 5", "Q5_0", "Q5_1", "Q8_0", "Q8_1",
                                    "Q2_K", "Q3_K", "Q4_K", "Q5_K", "Q6_K", "Q8_K", "IQ2_XXS", "IQ2_XS", "IQ3_XXS",
                                    "IQ1_S", "IQ4_NL", "IQ3_S", "IQ2_S", "IQ4_XS", "I8", "I16", "I32", "I64", "F64",
                                    "IQ1_M", "BF16", "type 31", "type 32", "type 33", "TQ1_0", "TQ2_0", "type 36",
                                    "type 37", "type 38", "MXFP4", "NVFP4", "Q1_0", "Q2_0", "F8_E4M3"};
    return t < sizeof n / sizeof n[0] ? n[t] : "unknown type";
}
bool gg_expert_type_ok(uint32_t t) { return t == GG_IQ3_S || t == GG_IQ4_NL || t == GG_Q8_0; }

// ty[l] = {gate_up type, down type} of layer l (nl layers, the second range from split): "" or why they are refused
std::string gg_expert_mix_error(const uint32_t (*ty)[2], int nl, int split) {
    char m[320];
    for (int l = 0; l < nl; ++l)
        for (int p = 0; p < 2; ++p)
            if (!gg_expert_type_ok(ty[l][p])) {
                std::snprintf(m, sizeof m, "GGUF experts: layer %d %s is %s (supported: IQ3_S, IQ4_NL, Q8_0)", l,
                              p ? "down" : "gate_up", gg_type_name(ty[l][p]));
                return m;
            }
    for (int r0 : {0, split}) {
        const int r1 = r0 ? nl : split;
        for (int l = r0 + 1; l < r1; ++l)
            if (ty[l][0] != ty[r0][0] || ty[l][1] != ty[r0][1]) {
                std::snprintf(m, sizeof m, "GGUF experts: layer %d is %s / %s (gate_up / down), layer %d %s / %s; "
                              "supported: one type pair for layers 0-%d and one for layers %d-%d", l,
                              gg_type_name(ty[l][0]), gg_type_name(ty[l][1]), r0, gg_type_name(ty[r0][0]),
                              gg_type_name(ty[r0][1]), split - 1, split, nl - 1);
                return m;
            }
    }
    return "";
}

// g_gg_a / g_gg_b (part bytes per expert, both ranges) and g_gg_two, or exit naming the layers and types
void gg_expert_formats(const Gguf& g) {
    uint32_t ty[NL][2];
    for (int l = 0; l < NL; ++l) {
        ty[l][0] = g.get("blk." + std::to_string(l) + ".ffn_gate_up_exps.weight").type;
        ty[l][1] = g.get("blk." + std::to_string(l) + ".ffn_down_exps.weight").type;
    }
    const std::string err = gg_expert_mix_error(ty, NL, MXFP8_FIRST);
    if (!err.empty()) { std::fprintf(stderr, "%s\n", err.c_str()); std::exit(1); }
    for (int r = 0; r < 2; ++r) {
        const int l = r ? MXFP8_FIRST : 0;
        g_gg_a[r] = g.get("blk." + std::to_string(l) + ".ffn_gate_up_exps.weight").bytes / NE;
        g_gg_b[r] = g.get("blk." + std::to_string(l) + ".ffn_down_exps.weight").bytes / NE;
        if (g_gg_a[r] % 16 || g_gg_b[r] % 16) {
            std::fprintf(stderr, "GGUF experts of layer %d: %zu / %zu bytes, not 16-byte multiples\n", l, g_gg_a[r], g_gg_b[r]);
            std::exit(1);
        }
    }
    g_gg_two = ty[0][0] != ty[MXFP8_FIRST][0] || ty[0][1] != ty[MXFP8_FIRST][1];
    if (g_gg_two)
        std::fprintf(stderr, "GGUF experts: layers 0-%d %s / %s, layers %d-%d %s / %s (two expert pools)\n", MXFP8_FIRST - 1,
                     gg_type_name(ty[0][0]), gg_type_name(ty[0][1]), MXFP8_FIRST, NL - 1,
                     gg_type_name(ty[MXFP8_FIRST][0]), gg_type_name(ty[MXFP8_FIRST][1]));
}
