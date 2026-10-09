#include "strata/core/rope_cache_migration.hpp"
#include "strata/artifact/dequant.hpp"
#include <bit>
#include <cstring>
#include <limits>
#include <stdexcept>

namespace strata::core {
namespace {
// IEEE binary16, round-to-nearest/even. No GPU header in this host-only code.
uint16_t half(float x) {
    const uint32_t bits = std::bit_cast<uint32_t>(x);
    const uint16_t sign = (bits >> 16) & 0x8000;
    int exp = int((bits >> 23) & 255) - 127 + 15;
    uint32_t mant = bits & 0x7fffff;
    if (!std::isfinite(x) || exp >= 31) throw std::runtime_error("nonfinite/overflow converted FP16 key");
    if (exp <= 0) {
        if (exp < -10) return sign;
        mant |= 0x800000;
        const int shift = 14 - exp;
        const uint32_t q = mant >> shift, rem = mant & ((1u << shift) - 1);
        return sign | uint16_t(q + (rem > (1u << (shift - 1)) ||
                           (rem == (1u << (shift - 1)) && (q & 1))));
    }
    mant += 0xfff + ((mant >> 13) & 1);
    if (mant & 0x800000) { mant = 0; ++exp; }
    if (exp >= 31) throw std::runtime_error("overflow converted FP16 key");
    return sign | uint16_t(exp << 10) | uint16_t(mant >> 13);
}
void write(ConversationBuffer& b, size_t at, const void* src, size_t n) {
    if (!b.visit(at, n, [&](uint8_t* p, size_t c, size_t offset) {
        std::memcpy(p, static_cast<const uint8_t*>(src) + offset - at, c); return true;
    })) throw std::runtime_error("key write outside validated buffer");
}
SessionRope session_rope(const kernels::RopeScaling& r) {
    return {(int64_t)r.type, r.freq_base, r.factor, r.freq_scale(), r.orig_ctx,
            r.ext_factor, r.attn_factor, r.beta_fast, r.beta_slow};
}
bool coherent(const RopeCacheProfile& p) {
    auto c = p.config; c.rope = session_rope(p.rope);
    return session_config_fingerprint(c) == session_config_fingerprint(p.config);
}
void row_float(float* row, int64_t pos, const RopeCacheProfile& s, const RopeCacheProfile& t) {
    for (int pair = 0; pair < 32; ++pair) {
        double x = row[pair], y = row[pair + 32];
        convert_rope_pair(x, y, pos, pair, 64, s.rope, t.rope);
        if (!std::isfinite(x) || !std::isfinite(y)) throw std::runtime_error("nonfinite index key");
        row[pair] = (float)x; row[pair + 32] = (float)y;
        if (!std::isfinite(row[pair]) || !std::isfinite(row[pair + 32]))
            throw std::runtime_error("overflow converted FP32 index key");
    }
}
}

void convert_rope_pair(double& x, double& y, int64_t position, int pair, int n_rot,
                       const kernels::RopeScaling& source, const kernels::RopeScaling& target) {
    if (position < 0 || n_rot <= 0 || n_rot % 2 || pair < 0 || pair >= n_rot / 2 ||
        kernels::rope_scaling_invalid(source) || kernels::rope_scaling_invalid(target))
        throw std::invalid_argument("invalid rotary pair/profile");
    float oc, os, nc, ns;
    kernels::rope_table_coefficients(n_rot, source, position, pair, oc, os);
    kernels::rope_table_coefficients(n_rot, target, position, pair, nc, ns);
    const double denominator = double(oc)*oc + double(os)*os;
    if (!(denominator > 0) || !std::isfinite(denominator)) throw std::invalid_argument("noninvertible RoPE gain");
    const double c = (double(nc)*oc + double(ns)*os) / denominator;
    const double s = (double(ns)*oc - double(nc)*os) / denominator;
    const double a = x;
    x = c*a - s*y;
    y = s*a + c*y;
}

bool migrate_rope_cache_to_yarn4(SavedConversation& cache, const RopeCacheProfile& source,
                                 const RopeCacheProfile& target, std::string& error) {
    error.clear();
    auto fail = [&](const char* why) { error = std::string("RoPE migration: ") + why; return false; };
    using RT = kernels::RopeScalingType;
    if (!source.full_model_identity || !target.full_model_identity || !source.model_identity ||
        source.model_identity != target.model_identity) return fail("full model identities differ or are absent");
    if (cache.rope_migration[0]) return fail("cache was already migrated");
    if (std::find(source.config.switches.begin(), source.config.switches.end(),
                  std::pair<std::string, int64_t>{"migration_rope_table", 1}) == source.config.switches.end())
        return fail("source lacks the required table-coefficient execution fingerprint");
    if (source.rope.type != RT::None || target.rope.type != RT::YaRN || target.rope.factor != 4.0 ||
        target.rope.freq_scale() != 0.25 || source.rope.freq_base != target.rope.freq_base ||
        kernels::rope_scaling_invalid(source.rope) || kernels::rope_scaling_invalid(target.rope) ||
        !coherent(source) || !coherent(target)) return fail("requires coherent ordinary -> YaRN 4x profiles");
    auto config = target.config;
    config.rope = source.config.rope;
    config.max_context = source.config.max_context;
    if (session_config_fingerprint(config) != session_config_fingerprint(source.config) ||
        target.config.max_context < source.config.max_context) return fail("non-RoPE execution settings differ");
    if (source.config.kv != "fp16" || source.config.kv_rot || source.config.mtp_window < -1)
        return fail("only unrotated FP16 KV is implemented (BF16/quantized rejected)");
    const bool draft = source.config.mtp_window >= 0;
    const auto& g = cache.geometry;
    // Actual qwen4exp QSA/GDN layout; no conventional-transformer assumption.
    if (!cache.stage_images.empty() || cache.layer_lo != 0 || cache.layer_hi != g[1] ||
        g[1] <= 0 || g[2] != 4 || g[1] % g[2] || g[9] != 24 || g[10] != 2 ||
        g[11] != 256 || g[12] != 4 || g[13] != 128 || cache.kv.size() != size_t(g[1]/4) + draft)
        return fail("unsupported architecture, layer carve, or draft layout");
    if (cache.live.ids.empty() || !cache.live.imgs.empty() || !cache.live.stage_parts.empty() ||
        cache.live.ids.size() > uint64_t(source.config.max_context)) return fail("invalid token extent or multimodal state");
    const size_t tokens = cache.live.ids.size(), layers = size_t(g[1]/4);
    auto checkpoint_valid = [&](const ConversationCheckpoint& cp) {
        return !cp.ids.empty() && cp.ids.size() <= tokens && cp.imgs.empty() && cp.stage_parts.empty() &&
            std::equal(cp.ids.begin(), cp.ids.end(), cache.live.ids.begin()) &&
            cp.dead.size() == layers*128*4 && cp.tails.size() == layers*3*128*4 && cp.block_pos.size() == layers*4;
    };
    if (!checkpoint_valid(cache.live)) return fail("invalid live indexer state");
    for (const auto& cp : cache.checkpoints) if (!checkpoint_valid(cp)) return fail("invalid ancestor checkpoint");
    for (size_t layer = 0; layer < cache.kv.size(); ++layer) {
        const auto& kv = cache.kv[layer];
        const size_t pooled_rows = layer < layers ? tokens/4+1 : 0;
        if (kv.format != 0 || kv.heads != 2 || kv.head_dim != 256 || kv.idx_dim != 128 ||
            kv.page_size <= 0 || kv.cells <= 0 || kv.cells % kv.page_size || uint64_t(kv.cells) < tokens ||
            uint64_t(kv.cells) - tokens >= uint64_t(kv.page_size) ||
            uint64_t(kv.cells) > SIZE_MAX/1024 || kv.k.size() != size_t(kv.cells)*1024 ||
            kv.v.size() != kv.k.size() || !kv.k_scale.empty() || !kv.v_scale.empty() ||
            kv.pooled_rows != int64_t(pooled_rows) || kv.pooled.size() != pooled_rows*128*4)
            return fail("unsupported KV representation or corrupt buffer extent");
    }
    try {
        // One small position/pair transform table shared across layers and heads.
        // This is not a second KV representation: paged keys stay paged.
        struct Rotation { double c, s; };
        std::vector<Rotation> rotations(tokens*32);
        for (size_t p=0;p<tokens;++p) for (int pair=0;pair<32;++pair) {
            double x=1,y=0;
            convert_rope_pair(x,y,p,pair,64,source.rope,target.rope);
            rotations[p*32+pair]={x,y};
        }
        struct Staged { ConversationBuffer k, pooled; };
        std::vector<Staged> staged(cache.kv.size());
        std::vector<std::vector<uint8_t>> dead;
        auto stage_dead = [&](const ConversationCheckpoint& cp) {
            auto bytes = cp.dead;
            for (size_t layer = 0; layer < layers; ++layer) {
                float row[128]; std::memcpy(row, bytes.data()+layer*sizeof(row), sizeof(row));
                row_float(row, 0, source, target);
                std::memcpy(bytes.data()+layer*sizeof(row), row, sizeof(row));
            }
            dead.push_back(std::move(bytes));
        };
        stage_dead(cache.live);
        for (const auto& cp : cache.checkpoints) stage_dead(cp);
        for (size_t layer = 0; layer < cache.kv.size(); ++layer) {
            const auto& kv = cache.kv[layer];
            auto& stage = staged[layer]; stage.k = kv.k; stage.pooled = kv.pooled;
            // MTP cell p pairs main residual p with token p+1, using RoPE
            // position p. Its final cell may not yet exist at an output cap;
            // ordinary continuation recomputes it via draft_first. Do not
            // interpret that unused cell (or page padding) as a valid key.
            const size_t key_tokens = layer < layers ? tokens : tokens-1;
            for (size_t p = 0; p < key_tokens; ++p) for (size_t head = 0; head < 2; ++head) {
                const size_t row = ((p/kv.page_size*2+head)*kv.page_size+p%kv.page_size)*512;
                uint16_t values[64]; stage.k.read(values, row, sizeof(values));
                for (int pair = 0; pair < 32; ++pair) {
                    double x = strata::fp16_to_fp32(values[pair]), y = strata::fp16_to_fp32(values[pair+32]);
                    const auto& rotation=rotations[p*32+pair];
                    const double a=x;
                    x=rotation.c*a-rotation.s*y;
                    y=rotation.s*a+rotation.c*y;
                    values[pair] = half((float)x); values[pair+32] = half((float)y);
                }
                write(stage.k, row, values, sizeof(values));
            }
            for (size_t p = 0; p < size_t(kv.pooled_rows); ++p) {
                float row[128]; stage.pooled.read(row, p*sizeof(row), sizeof(row));
                row_float(row, p == tokens/4 ? 0 : int64_t(p*4), source, target);
                write(stage.pooled, p*sizeof(row), row, sizeof(row));
            }
        }
        // No allocations or fallible transfers after this point. V/GDN/PLE/raw
        // tails/token IDs are untouched. Old state is not silently called exact.
        for (size_t i = 0; i < cache.kv.size(); ++i) {
            cache.kv[i].k = std::move(staged[i].k);
            cache.kv[i].pooled = std::move(staged[i].pooled);
        }
        cache.live.dead.swap(dead[0]);
        for (size_t i = 0; i < cache.checkpoints.size(); ++i) cache.checkpoints[i].dead.swap(dead[i+1]);
        cache.rope_migration = {1, session_config_fingerprint(source.config),
                                session_config_fingerprint(target.config), tokens};
        return true;
    } catch (const std::exception& e) { error = std::string("RoPE migration: ") + e.what(); return false; }
}
} // namespace strata::core
