// include/strata/hy4flash_gguf/cpu/expert_layout.hpp - plan v0.3 P6: where each routed expert lives in experts.bin.
//
// A Q2_0 pack (tools/strata_pack.py) has one blob size for every layer, `BLOB`, in the Strata expert form.  A
// native pack (tools/iq_pack.py, the IQ2_XS / IQ3_XXS files) keeps each expert's raw GGUF slices, so the blob
// size and the formats change from layer to layer; `native_experts.txt` says how.  Everything that touches an
// expert blob - the arena, the VRAM tier, the prompt path, the CPU pool, the GPU window - asks this table.
#pragma once

#include "strata/kernels/cpu/expert.hpp"
#include "strata/kernels/cpu/expert_layout.hpp"
#include "strata/hy4flash_gguf/cpu/native_expert.hpp"

#include <cstdint>
#include <string>
#include <vector>

namespace strata::hy4_flash::cpu {
using ::strata::kernels::cpu::BLOB;
using ::strata::kernels::cpu::FF;
using ::strata::kernels::cpu::H;
using ::strata::kernels::cpu::NE;

struct ExpertLayout {
    bool native = false;
    int64_t n_layers = 0, n_expert = NE;
    std::vector<NativeFmt> fmt;           ///< per layer (native packs)
    std::vector<uint64_t> offset, bytes;  ///< per layer: where its 512 blobs start, bytes per blob
    /// Plan v0.3 P6: per layer, the absolute offsets of the gate / up / down tensors in their GGUF files, so the
    /// arena can be filled from the GGUF itself when the pack has no experts.bin (3 x n_layers, 0 = unknown).
    std::vector<uint64_t> gguf_off;
    /// Per layer PER ROLE (3 x n_layers, `3 * layer + role`, matching `gguf_off`), the GGUF file - a name beside
    /// the --native shard - that holds that role's tensor.  Empty = the --native shard itself.  Per role because a
    /// shard boundary can fall inside a layer: Unsloth's UD-Q4_K_XL has layer 11's down in shard 2 and its gate
    /// and up in shard 3 (native_experts.txt v4, the `gate,up,down` shard column).
    std::vector<std::string> gguf_file;
    int version = 0;                      ///< native_experts.txt's header version (0 = none given)
    /// Index zero corresponds to this model block. Layout indices are relative to it.
    int64_t block_base = 0;
    /// The expert geometry this layout was read for - what `native_fmt` was asked about each layer's types.
    /// The compiled-in H / FF are only right for the family they were compiled for, so a caller passes its own.
    int64_t n_embd = H, n_ff = FF;
    uint64_t max_blob = BLOB;
    uint64_t total = 0;                   ///< experts.bin size

    /// The block `index` (an expert ordinal) belongs to - what a tensor name is composed from.
    int64_t block_of(int64_t index) const { return block_base + index; }

    uint64_t blob_bytes(int64_t layer) const { return native ? bytes[(size_t) layer] : (uint64_t) BLOB; }
    uint64_t layer_offset(int64_t layer) const {
        return native ? offset[(size_t) layer] : (uint64_t) layer * (uint64_t) n_expert * (uint64_t) BLOB;
    }
    uint64_t blob_offset(int64_t layer, int64_t expert) const {
        return layer_offset(layer) + (uint64_t) expert * blob_bytes(layer);
    }
};

const ExpertLayout& expert_layout();
/// Reads `<pack_dir>/native_experts.txt` when it exists (a native pack), else sets the canonical layout.
/// Versions up to kExpertLayoutVersion are read; a newer one is refused (a newer packer wrote it).
/// Count routing blocks only. A dense prefix is excluded.
bool expert_layout_load(const std::string& pack_dir, int64_t n_layers, int64_t n_expert, std::string& err,
                        int64_t block_base = 0, int64_t n_embd = H, int64_t n_ff = FF,
                        float swiglu_clamp = 0.f, int64_t gpu_only_from = -1);
/// The newest native_experts.txt this engine reads.  v4 = v3 plus the per-role shard column `gate,up,down`,
/// written only when some layer's roles are in different shards (every other pack stays v3, byte for byte).
/// v5 = v4 plus `hadamard.txt` beside it: Hadamard-folded experts (hadamard.hpp), written only for such models,
/// so an engine that cannot rotate the activations refuses the pack instead of running it wrong.
inline constexpr int kExpertLayoutVersion = 5;

}  // namespace strata::hy4_flash::cpu
