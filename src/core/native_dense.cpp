#include "strata/core/native_dense.hpp"
#include <cstdlib>
#include "strata/core/weights.hpp"
#include "strata/artifact/gguf_reader.hpp"
#include "strata/kernels/native_mmvq.hpp"
#include "strata/platform/memory.hpp"

#include <cuda_runtime.h>
#include <algorithm>
#include <climits>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <limits>
#include <memory>
#include <set>
#include <cmath>
#include <cstring>
#include <vector>

namespace strata::core {
namespace {
// `load` makes one cudaMalloc per projection matrix, and the driver backs each call with 2 MiB granules
// (measured on the RTX 5060: 300 buffers of 6.73 MiB asked 2019 MiB and cost 2400).  So a stage's real
// footprint is the granule-rounded sum, not the payload sum.  This rule - a granule for every matrix of a
// MiB or more, the payload for the rest - prices the iq3_s pack's 300 matrices at 2255 MiB where they
// really take 2238 (+0.8%).  It over-prices, which is the safe side.  (Pricing every matrix at a granule is
// far worse, 2378 MiB: the driver packs the sub-MiB matrices into the slack of the rounded ones.)
constexpr uint64_t kAllocGranule = 2ull << 20;
constexpr uint64_t kSmallAlloc = 1ull << 20;
uint64_t alloc_bytes(uint64_t bytes) {
    return bytes < kSmallAlloc ? bytes : (bytes + kAllocGranule - 1) / kAllocGranule * kAllocGranule;
}
int g_layer_lb = -1, g_layer_le = -1;   // set_layer_range; -1: every layer
/// The draft block (`set_draft_block`): the model's OWN block past the trunk, index `n_trunk`, which is packed
/// only by `tools/iq_pack.py --mtp` and run only by `--mtp`.  Off, it is dropped exactly as it always was.
bool g_draft = false;
bool in_range(const std::string& name) {
    if (g_layer_lb < 0 || name.rfind("blk.", 0) != 0) return true;
    const int l = std::atoi(name.c_str() + 4);
    // `l == g_layer_le` is the draft block on the stage that holds the trunk's end (`g_layer_le == n_layers`
    // there, and lower on every earlier stage, so a split still gives the block to exactly one card).
    return (l >= g_layer_lb && l < g_layer_le) || (g_draft && l == g_layer_le);
}
// S23 experiment (STRATA_HC_Q8=1): the hyper-connection projections' Q8_0 bytes for the verify read
bool hc_q8_requested() {
    static const bool on = [] { const char* v = std::getenv("STRATA_HC_Q8"); return v != nullptr && v[0] == '1'; }();
    return on;
}
bool hc_q8_name(const std::string& name) {
    if (name == "output_hc_down.weight" || name == "output_hc_up.weight") return true;   // S25: the final mixer too
    if (name.rfind("blk.", 0) != 0) return false;
    static const char* suffixes[] = {".hc_attn_down.weight", ".hc_attn_up.weight", ".hc_attn_inject.weight",
                                     ".hc_ffn_down.weight", ".hc_ffn_up.weight", ".hc_ffn_inject.weight"};
    for (const char* suffix : suffixes) if (name.ends_with(suffix)) return true;
    return false;
}
// S25: the GGUF keeps the inject rows as F32 - quantized here to Q8_0 (ggml's reference rounding) so the Q8_0 read
// covers every hc projection
uint16_t half_bits(float f) {
    uint32_t x;
    std::memcpy(&x, &f, 4);
    const uint32_t sign = (x >> 16) & 0x8000u;
    int32_t e = (int32_t) ((x >> 23) & 0xff) - 127 + 15;
    uint32_t m = x & 0x7fffffu;
    if (e <= 0) {                                   // subnormal or zero
        if (e < -10) return (uint16_t) sign;
        m |= 0x800000u;
        const uint32_t shift = (uint32_t) (14 - e);
        uint32_t h = m >> shift;
        const uint32_t rem = m & ((1u << shift) - 1), half = 1u << (shift - 1);
        if (rem > half || (rem == half && (h & 1u))) ++h;
        return (uint16_t) (sign | h);
    }
    if (e >= 31) return (uint16_t) (sign | 0x7c00u);
    uint32_t h = ((uint32_t) e << 10) | (m >> 13);
    const uint32_t rem = m & 0x1fffu;
    if (rem > 0x1000u || (rem == 0x1000u && (h & 1u))) ++h;
    return (uint16_t) (sign | h);
}
std::vector<uint8_t> q8_0_of(const float* x, uint64_t n) {
    std::vector<uint8_t> out((size_t) (n / 32 * 34));
    for (uint64_t b = 0; b < n / 32; ++b) {
        float amax = 0.0f;
        for (int j = 0; j < 32; ++j) amax = (std::max)(amax, std::fabs(x[b * 32 + j]));
        const float d = amax / 127.0f, id = d != 0.0f ? 1.0f / d : 0.0f;
        uint8_t* o = out.data() + b * 34;
        const uint16_t hb = half_bits(d);
        std::memcpy(o, &hb, 2);
        for (int j = 0; j < 32; ++j) o[2 + j] = (uint8_t) (int8_t) std::lround(x[b * 32 + j] * id);
    }
    return out;
}
// Which family a shard is, and how deep its trunk runs.  glm5-next carries a next-token-prediction block PAST
// the trunk (`nextn_predict_layers` = 1, block 45 of 46) which v1 does not run, so the pack does not hold it and
// the engine must not expect the canonical table to either.  Read from the shard's own metadata, so the file
// stays the authority; `n_trunk` < 0 means "no block past the last one", which is the qwen4exp case.
Arch family_of(const strata::GgufFile& gguf, int64_t& n_trunk) {
    n_trunk = -1;
    Arch a = Arch::Unknown;
    const auto* g = gguf.get("general.architecture");
    if (!g || !arch_from_string(g->s, a)) return Arch::Unknown;
    if (a != Arch::Glm5Next) return a;
    const auto* bc = gguf.get(std::string(arch_meta_prefix(a)) + ".block_count");
    const auto* nn = gguf.get(std::string(arch_meta_prefix(a)) + ".nextn_predict_layers");
    if (bc && nn && bc->u > nn->u) n_trunk = (int64_t) (bc->u - nn->u);
    return a;
}

/// The index holds TWO dimensions and a tensor with more is written with its trailing ones folded into ne1 -
/// which is what they are: ggml makes ne0 the row length and everything after it the row count.  The packer
/// folds them (iq_pack.py) and the engine has to fold them the same way or it refuses a tensor the pack holds,
/// which is what glm5-next's MLA weights need (attn_k_b is [256, 512, 64]).
uint64_t folded_ne1(const strata::TensorInfo& t) {
    uint64_t n = 1;
    for (size_t i = 1; i < t.shape.size(); ++i) n *= t.shape[i];
    return n;
}

bool eligible(const strata::TensorInfo& tensor, bool include_ple_key, Arch arch, int64_t n_trunk) {
    const auto& name = tensor.name;
    if (name.rfind("blk.", 0) != 0) return false;
    // **THE DRAFT BLOCK IS THE ONE `blk.<n_trunk>` TENSOR THE ENGINE MAY WANT.**  Without `--mtp` this drop is
    // what it always was and the pack holds nothing there either.  With it the block is a real layer, and every
    // one of its quantized tensors is written into `index.txt` as "served from the GGUF" (a row with no bytes),
    // so dropping it here leaves the `WeightRef` with neither `data` nor `native_data` and `project` refuses it.
    if (n_trunk >= 0 && std::strtol(name.c_str() + 4, nullptr, 10) >= n_trunk &&
        !(g_draft && std::strtol(name.c_str() + 4, nullptr, 10) == n_trunk))
        return false;
    // Match the native PLE kernel: Q2_0, IQ3_XXS, IQ4_XS and Q8_0 (UD-Q4_K_XL). Other keys retain the packed BF16
    // fallback.
    if (name == "blk.1.ple_key.weight")
        return include_ple_key && (tensor.type == 42 || tensor.type == 18 || tensor.type == 23 || tensor.type == 8);
    static const char* suffixes[] = {".attn_qkv.weight", ".attn_gate.weight", ".ssm_out.weight",
        ".attn_q.weight", ".attn_k.weight", ".attn_v.weight", ".attn_output.weight",
        ".ffn_gate_shexp.weight", ".ffn_up_shexp.weight", ".ffn_down_shexp.weight"};
    static const char* glm_suffixes[] = {".attn_q_a.weight", ".attn_q_b.weight", ".attn_kv_a_mqa.weight",
        ".attn_k_b.weight", ".attn_v_b.weight", ".ssm_f_a.weight", ".ssm_f_b.weight", ".ssm_g_a.weight",
        ".ssm_g_b.weight", ".ssm_beta.weight", ".ffn_gate.weight", ".ffn_up.weight", ".ffn_down.weight",
        ".indexer.attn_q_b.weight", ".indexer.attn_k.weight", ".indexer_compressor_gate.weight",
        // the draft block's own projection.  It is the one quantized tensor of that block whose name does not
        // appear on a trunk layer, and the packer writes it as "served natively" for exactly this reason.
        ".nextn.eh_proj.weight"};
    if (arch == Arch::Glm5Next) {
        for (const char* suffix : glm_suffixes) if (name.ends_with(suffix)) return true;
    }
    for (const char* suffix : suffixes) if (name.ends_with(suffix)) return true;
    return false;
}
struct DeviceFree { void operator()(void* p) const { if (p) cudaFree(p); } };
using DevicePtr = std::unique_ptr<void, DeviceFree>;
struct Pending {
    WeightRef* ref;
    int type;
    uint64_t bytes;
    DevicePtr data;
    DevicePtr packed;   // STRATA_Q8_PACKED=1 copy, or null
};
}

bool NativeDense::served_names(const std::vector<std::string>& shards, bool include_ple_key,
                               std::set<std::string>& out, std::string& err) {
    try {
        Arch arch = Arch::Unknown;
        int64_t n_trunk = -1;
        for (const auto& path : shards) {
            strata::GgufFile gguf(path);
            int64_t trunk = -1;
            const Arch a = family_of(gguf, trunk);
            // First non-empty wins, and the two are read separately: a later shard of a split may still name its
            // family while carrying none of the model's keys (the GLM-5.3-Flash split names `glm5next` on shards
            // 2-5), so the trunk depth found on shard 1 must survive a shard that cannot restate it.
            if (arch == Arch::Unknown && a != Arch::Unknown) arch = a;
            if (trunk >= 0) n_trunk = trunk;
            for (const auto& tensor : gguf.tensors())
                if (eligible(tensor, include_ple_key, arch, n_trunk) &&
                    strata::kernels::native_mmvq_supported(tensor.type) && tensor.shape.size() >= 2)
                    out.insert(tensor.name);
        }
        return true;
    } catch (const std::exception& error) {
        err = std::string("native dense: ") + error.what();
        return false;
    }
}

bool NativeDense::served_bytes_per_layer(const std::vector<std::string>& shards, bool include_ple_key,
                                         int64_t n_layers, std::vector<uint64_t>& out, std::string& err) {
    if (n_layers < 0) { err = "native dense: served_bytes_per_layer: negative layer count"; return false; }
    out.assign((size_t) n_layers, 0);
    try {
        Arch arch = Arch::Unknown;
        int64_t n_trunk = -1;
        std::set<std::string> seen;
        uint64_t non_blk = 0;   // eligible names that are not `blk.` - none today, but not silently dropped
        for (const auto& path : shards) {
            strata::GgufFile gguf(path);
            int64_t trunk = -1;
            const Arch a = family_of(gguf, trunk);
            if (arch == Arch::Unknown && a != Arch::Unknown) arch = a;
            if (trunk >= 0) n_trunk = trunk;
            for (const auto& tensor : gguf.tensors()) {
                if (!eligible(tensor, include_ple_key, arch, n_trunk)) continue;
                if (!seen.insert(tensor.name).second) continue;
                if (!strata::kernels::native_mmvq_supported(tensor.type) || tensor.shape.size() < 2) continue;
                // `load` compares the GGUF shape against the canonical row and REFUSES a tensor where they
                // disagree, so the shape read here is the one the upload would use - and `folded_ne1` is the
                // packer's own rule for a matrix with more than two dimensions (glm5-next's `attn_k_b`).
                const uint64_t ne1 = folded_ne1(tensor);
                if (tensor.shape[0] == 0 || ne1 == 0 || tensor.shape[0] > (uint64_t) INT_MAX ||
                    ne1 > (uint64_t) INT_MAX)
                    continue;
                const uint64_t bytes =
                    strata::kernels::native_mmvq_weight_bytes(tensor.type, (int) tensor.shape[0], (int) ne1);
                // `eligible` admits only `blk.<l>.` names, so this is belt and braces - but a name that got
                // through would otherwise be priced at zero, which is the one error a placement cannot see.
                const long l = std::strtol(tensor.name.c_str() + 4, nullptr, 10);
                if (l < 0 || l >= n_layers) { non_blk += bytes; continue; }
                out[(size_t) l] += bytes;
            }
        }
        if (non_blk != 0) {
            err = "native dense: served_bytes_per_layer: " + std::to_string(non_blk) +
                  " B of eligible weights outside every layer";
            return false;
        }
        return true;
    } catch (const std::exception& error) {
        err = std::string("native dense: ") + error.what();
        return false;
    }
}

void NativeDense::set_layer_range(int lb, int le) { g_layer_lb = lb; g_layer_le = le; }
void NativeDense::set_draft_block(bool on) { g_draft = on; }
// The byte walk `load` does for the layers [lb, le), without the allocations: the same filters in the same order,
// reading the GGUF headers and the canonical table only, so it needs no device.  A later stage's projections are not
// on the card when the split search prices that stage, and `load` runs one cudaMalloc per matrix (#1238).  The two
// have to stay in step.  A repeated name is skipped rather than refused (`load` refuses it).
bool NativeDense::weight_bytes_for(const std::vector<std::string>& shards, WeightTable& table, bool include_ple_key,
                                   int64_t lb, int64_t le, uint64_t& out, std::string& err) {
    out = 0;
    try {
        std::set<std::string> seen;
        uint64_t total = 0;
        // The arch and the trunk depth, read the way `served_names` reads them and for the same reason: `eligible`
        // needs both (glm5-next's tensor names are its own, and its draft block is the one `blk.<n_trunk>` tensor
        // the engine may want).  Without them the walk prices a different set than `load` allocates.
        Arch arch = Arch::Unknown;
        int64_t n_trunk = -1;
        for (const auto& path : shards) {
            strata::GgufFile gguf(path);
            int64_t trunk = -1;
            const Arch a = family_of(gguf, trunk);
            if (arch == Arch::Unknown && a != Arch::Unknown) arch = a;
            if (trunk >= 0) n_trunk = trunk;
            for (const auto& tensor : gguf.tensors()) {
                if (!eligible(tensor, include_ple_key, arch, n_trunk)) continue;
                if (tensor.name.rfind("blk.", 0) == 0) {
                    const long l = std::strtol(tensor.name.c_str() + 4, nullptr, 10);
                    if (l < lb || l >= le) continue;
                }
                if (!seen.insert(tensor.name).second) continue;
                auto found = table.table_.find(tensor.name);
                if (found == table.table_.end()) {
                    err = "native dense: tensor absent from canonical table: " + tensor.name;
                    return false;
                }
                const auto& ref = found->second;
                if (!strata::kernels::native_mmvq_supported(tensor.type)) continue;
                if (tensor.name == "blk.1.ple_key.weight" && !ref.quantized()) continue;
                if (!ref.quantized() || tensor.shape.size() != 2 ||
                    ref.ne0 <= 0 || ref.ne0 > INT_MAX || ref.ne1 <= 0 || ref.ne1 > INT_MAX ||
                    tensor.shape[0] != (uint64_t) ref.ne0 || tensor.shape[1] != (uint64_t) ref.ne1) {
                    err = "native dense: incompatible matrix " + tensor.name;
                    return false;
                }
                total += alloc_bytes(strata::kernels::native_mmvq_weight_bytes(tensor.type, (int) ref.ne0, (int) ref.ne1));
            }
        }
        out = total;
        return true;
    } catch (const std::exception& error) {
        err = std::string("native dense: ") + error.what();
        return false;
    }
}
bool NativeDense::keep_unquantized_ple_key(const std::string& pack_dir, std::set<std::string>& skip,
                                           std::string& err) {
    const std::string key = "blk.1.ple_key.weight";
    if (!skip.count(key)) return true;
    int code_bits = -1;
    if (!WeightTable::index_code_bits(pack_dir, key, code_bits, err)) return false;
    if (code_bits == 0) skip.erase(key);
    return true;
}

NativeDense::~NativeDense() {
    for (const void* p : packed_keys_) strata::kernels::native_q8_0_packed_unregister(p);
    if (scratch_) cudaFree(scratch_);
    for (void* p : weights_) cudaFree(p);
}

bool NativeDense::load(const std::vector<std::string>& shards, WeightTable& table, std::string& err,
                       bool include_ple_key, int64_t layer_lo, int64_t layer_hi) {
    auto outside = [&](const std::string& name) {   // a blk.<l>. tensor of another stage's layers
        if (layer_hi < 0 || name.rfind("blk.", 0) != 0) return false;
        const long l = std::strtol(name.c_str() + 4, nullptr, 10);
        return l < layer_lo || l >= layer_hi;
    };
    if (scratch_ || !weights_.empty()) { err = "native dense: already loaded"; return false; }
    if (shards.empty()) { err = "native dense: at least one GGUF shard is required"; return false; }
    try {
        std::vector<Pending> pending;
        std::set<std::string> seen;
        int max_in = 0;
        uint64_t total = 0, hc_q8_bytes = 0, allocated = 0;
        uint64_t split_count = 0, split_tensors = 0;
        std::set<uint64_t> split_numbers;
        bool have_architecture = false;
        Arch arch = Arch::Unknown;
        int64_t n_trunk = -1;
        for (const auto& path : shards) {
            strata::GgufFile gguf(path);
            int64_t trunk = -1;
            const Arch a = family_of(gguf, trunk);
            if (arch == Arch::Unknown && a != Arch::Unknown) arch = a;
            if (trunk >= 0) n_trunk = trunk;
            const auto* count = gguf.get("split.count");
            const auto* number = gguf.get("split.no");
            const auto* tensors = gguf.get("split.tensors.count");
            const auto* family = gguf.get("general.architecture");
            // The shard that CARRIES the model's keys is the one the counts are checked on.  llama.cpp's splitter
            // writes them into shard 1 alone, and a later shard is allowed to name its family without restating
            // them - the GLM-5.3-Flash split does exactly that (`glm5next` on shards 2-5, block_count nowhere but
            // shard 1).  Asking such a shard for `block_count` reported a missing key of the MODEL as if the shard
            // were malformed, and refused a file the engine can read.
            const bool meta_shard = number && number->u != 0 ? false : true;   // shard 1 of a split, or a whole model
            if (meta_shard) {
                if (family) {
                    err = strata::check_architecture(gguf);
                    if (!err.empty()) return false;
                    have_architecture = true;
                    if (count && number && tensors && number->u == 0 && count->u > 1) {
                        split_count = count->u;
                        split_tensors = tensors->u;
                    }
                }
            } else {
                // A later shard that names a family still has to name THIS model's family: two files mixed into
                // one shard list is the failure the guard exists for.
                if (family) {
                    Arch said = Arch::Unknown;
                    if (!arch_from_string(family->s, said) || (arch != Arch::Unknown && said != arch)) {
                        err = "native dense: shard " + std::to_string(number->u + 1) + " is '" + family->s +
                              "', not this model's family";
                        return false;
                    }
                }
                if (!have_architecture || !split_count || !count || !number || !tensors ||
                    count->u != split_count || number->u == 0 || number->u >= split_count ||
                    tensors->u != split_tensors) {
                    err = "native dense: additional shard must match the architecture-validated first shard's "
                          "split metadata";
                    return false;
                }
            }
            if (number && !split_numbers.insert(number->u).second) {
                err = "native dense: duplicate split shard number"; return false;
            }
            std::vector<uint64_t> offsets;
            for (const auto& tensor : gguf.tensors()) offsets.push_back(tensor.offset);
            std::sort(offsets.begin(), offsets.end());
            if (std::adjacent_find(offsets.begin(), offsets.end()) != offsets.end()) {
                err = "native dense: tensor payload offsets overlap"; return false;
            }
            // Validate every directory span, including tensors we do not upload:
            // an ignored tensor must not overlap the native matrix that follows it.
            const uint64_t payload = gguf.file_size() - gguf.data_start();
            for (const auto& tensor : gguf.tensors()) {
                int block_elements = 0, block_bytes = 0;
                uint64_t elements = 1;
                if (tensor.shape.empty() || !strata::block_geometry(tensor.type, block_elements, block_bytes) ||
                    tensor.shape[0] % (uint64_t) block_elements != 0) {
                    err = "native dense: invalid block geometry " + tensor.name; return false;
                }
                for (uint64_t dimension : tensor.shape) {
                    if (!dimension || elements > (std::numeric_limits<uint64_t>::max)() / dimension) {
                        err = "native dense: invalid tensor extent " + tensor.name; return false;
                    }
                    elements *= dimension;
                }
                const uint64_t blocks = elements / (uint64_t) block_elements;
                if (blocks > (std::numeric_limits<uint64_t>::max)() / (uint64_t) block_bytes) {
                    err = "native dense: tensor byte count overflow " + tensor.name; return false;
                }
                const uint64_t bytes = blocks * (uint64_t) block_bytes;
                if (tensor.offset > payload || bytes > payload - tensor.offset) {
                    err = "native dense: truncated payload " + tensor.name; return false;
                }
                const auto next = std::upper_bound(offsets.begin(), offsets.end(), tensor.offset);
                if (next != offsets.end() && bytes > *next - tensor.offset) {
                    err = "native dense: overlapping payload " + tensor.name; return false;
                }
                // the uploads below read these: ask for them now so the reads overlap
                if (eligible(tensor, include_ple_key, arch, n_trunk) && !outside(tensor.name) &&
                    strata::kernels::native_mmvq_supported(tensor.type))
                    strata::platform::advise_willneed(gguf.tensor_data(tensor), bytes);
            }
            for (const auto& tensor : gguf.tensors()) {
                if (!eligible(tensor, include_ple_key, arch, n_trunk) || outside(tensor.name)) continue;
                if (!in_range(tensor.name) && tensor.name.find("ple") == std::string::npos) continue;
                if (!seen.insert(tensor.name).second) {
                    err = "native dense: duplicate tensor " + tensor.name; return false;
                }
                auto found = table.table_.find(tensor.name);
                if (found == table.table_.end()) {
                    err = "native dense: tensor absent from canonical table: " + tensor.name; return false;
                }
                auto& ref = found->second;
                if (ref.native_data) { err = "native dense: override already attached"; return false; }
                if (!strata::kernels::native_mmvq_supported(tensor.type)) continue;
                // #326: the pack keeps an unquantized (--compat-bf16) key, which the PLE reads from the arena
                if (tensor.name == "blk.1.ple_key.weight" && !ref.quantized()) continue;
                // `folded_ne1` is the packer's own rule, not a second opinion: a [256, 512, 64] weight is 256-wide
                // rows and the index calls that 32768 of them.  Demanding exactly two GGUF dimensions, as this
                // did, refused glm5-next's MLA weights - which the pack holds and the engine has to read.
                if (!ref.quantized() || tensor.shape.size() < 2 ||
                    ref.ne0 <= 0 || ref.ne0 > INT_MAX || ref.ne1 <= 0 || ref.ne1 > INT_MAX ||
                    tensor.shape[0] != (uint64_t) ref.ne0 || folded_ne1(tensor) != (uint64_t) ref.ne1) {
                    err = "native dense: incompatible matrix " + tensor.name; return false;
                }
                const auto bytes = strata::kernels::native_mmvq_weight_bytes(
                    tensor.type, (int) ref.ne0, (int) ref.ne1);
                void* allocation = nullptr;
                auto status = cudaMalloc(&allocation, bytes);
                DevicePtr data(allocation);
                if (status == cudaSuccess)
                    status = cudaMemcpy(data.get(), gguf.tensor_data(tensor), bytes, cudaMemcpyHostToDevice);
                if (status != cudaSuccess) {
                    // The count so far and the card's own numbers: without them a failed upload says only which
                    // tensor it stopped on, and the two questions that follow ("how much does this model want",
                    // "how much did the card have") both need a second run to answer.
                    size_t free_b = 0, total_b = 0;
                    (void) cudaMemGetInfo(&free_b, &total_b);
                    err = "native dense upload " + tensor.name + ": " + cudaGetErrorString(status) +
                          " (this tensor " + std::to_string(bytes) + " B, " + std::to_string(total) +
                          " B resident, " + std::to_string(free_b) + " B free of " + std::to_string(total_b) + ")";
                    return false;
                }
                max_in = (std::max)(max_in, (int) ref.ne0);
                total += bytes;
                allocated += alloc_bytes(bytes);
                // STRATA_Q8_PACKED=1: a second, packed copy of an eligible Q8_0 matrix for the decode MMVQ (the GGUF
                // copy stays: the prompt path's GEMMs read it).
                DevicePtr packed;
                if (tensor.type == 8 && tensor.name != "blk.1.ple_key.weight" &&   // the PLE has its own kernel
                    strata::kernels::native_q8_0_packed_enabled() &&
                    strata::kernels::native_q8_0_packed_eligible((int) ref.ne0, (int) ref.ne1)) {
                    std::vector<uint8_t> host(bytes);
                    strata::kernels::native_q8_0_pack_host(gguf.tensor_data(tensor), host.data(), (int) ref.ne0,
                                                           (int) ref.ne1);
                    void* packed_allocation = nullptr;
                    auto packed_status = cudaMalloc(&packed_allocation, bytes);
                    packed.reset(packed_allocation);
                    if (packed_status == cudaSuccess)
                        packed_status = cudaMemcpy(packed.get(), host.data(), bytes, cudaMemcpyHostToDevice);
                    if (packed_status != cudaSuccess) {
                        err = "native dense packed upload " + tensor.name + ": " + cudaGetErrorString(packed_status);
                        return false;
                    }
                }
                pending.push_back(Pending{&ref, (int) tensor.type, bytes, std::move(data), std::move(packed)});
            }
            if (hc_q8_requested())
                for (const auto& tensor : gguf.tensors()) {
                    // S25: the F32 inject rows hold BF16-exact values: the read takes the pack's BF16 rows unless
                    // STRATA_HC_Q8_INJECT=1 asks for a Q8_0 copy
                    static const bool q8_inject = [] { const char* v = std::getenv("STRATA_HC_Q8_INJECT"); return v && v[0] == '1'; }();
                    const bool f32_inject = q8_inject && tensor.type == 0 && tensor.name.ends_with("_inject.weight");
                    if ((tensor.type != 8 && !f32_inject) || !hc_q8_name(tensor.name) || tensor.shape.size() != 2) continue;
                    auto found = table.table_.find(tensor.name);
                    if (found == table.table_.end() || found->second.hc_q8 != nullptr) continue;
                    auto& ref = found->second;
                    if (tensor.shape[0] != (uint64_t) ref.ne0 || tensor.shape[1] != (uint64_t) ref.ne1 || ref.ne0 % 32 != 0) {
                        err = "native dense (STRATA_HC_Q8): incompatible matrix " + tensor.name; return false;
                    }
                    const uint64_t bytes = (uint64_t) ref.ne0 / 32 * 34 * (uint64_t) ref.ne1;
                    void* p = nullptr;
                    std::vector<uint8_t> q8;
                    if (f32_inject) q8 = q8_0_of((const float*) gguf.tensor_data(tensor), (uint64_t) ref.ne0 * (uint64_t) ref.ne1);
                    if (cudaMalloc(&p, bytes) != cudaSuccess ||
                        cudaMemcpy(p, f32_inject ? (const void*) q8.data() : gguf.tensor_data(tensor), bytes,
                                   cudaMemcpyHostToDevice) != cudaSuccess) {
                        err = "native dense (STRATA_HC_Q8) upload " + tensor.name; return false;
                    }
                    ref.hc_q8 = p;
                    weights_.push_back(p);
                    hc_q8_bytes += bytes;
                }
        }
        if (pending.empty()) { err = "native dense: no supported GDN/QSA matrices in supplied shards"; return false; }
        void* allocation = nullptr;
        // THE SCRATCH IS WIDE ENOUGH FOR A WHOLE BATCH, NOT ONE TOKEN.  Every native projection shares this one
        // buffer, and a batched prefill quantizes `ncols` activation columns into it before a single `native_mmvq`
        // -- so it has to hold the widest input at the widest column count, or the 8-column call writes past it.
        const auto status =
            cudaMalloc(&allocation, strata::kernels::native_q8_1_bytes(max_in, strata::kernels::NATIVE_MMVQ_MAX_NCOLS));
        DevicePtr scratch(allocation);
        if (status != cudaSuccess) { err = std::string("native dense scratch: ") + cudaGetErrorString(status); return false; }
        // All checks and allocations finish before publishing any reference.
        weights_.reserve(pending.size());
        uint64_t packed_bytes = 0;
        size_t packed_count = 0;
        for (auto& item : pending) {
            item.ref->native_data = item.data.get();
            item.ref->native_type = item.type;
            item.ref->native_q8_1 = scratch.get();
            if (item.packed) {
                strata::kernels::native_q8_0_packed_register(item.data.get(), item.packed.get(), (int) item.ref->ne0,
                                                             (int) item.ref->ne1);
                packed_keys_.push_back(item.data.get());
                packed_bytes += item.bytes;
                ++packed_count;
                weights_.push_back(item.packed.release());
            }
            weights_.push_back(item.data.release());
        }
        if (packed_count)
            std::fprintf(stderr, "native dense: STRATA_Q8_PACKED=1 packed %zu Q8_0 matrices for decode (+%.1f MiB)\n",
                         packed_count, packed_bytes / 1048576.0);
        scratch_ = scratch.release();
        bytes_ = total + hc_q8_bytes;
        allocated_ = allocated;
        if (hc_q8_requested())
            std::fprintf(stderr, "strata: STRATA_HC_Q8=1: %.2f GiB of Q8_0 hyper-connection projections for the verify read\n",
                         (double) hc_q8_bytes / 1073741824.0);
        return true;
    } catch (const std::exception& error) {
        err = std::string("native dense: ") + error.what();
        return false;
    }
}
} // namespace strata::core
