// include/strata/kernels/cpu/expert_layout.hpp - plan v0.3 P6: where each routed expert lives in experts.bin.
//
// A Q2_0 pack (tools/strata_pack.py) has one blob size for every layer, `BLOB`, in the Strata expert form.  A
// native pack (tools/iq_pack.py, the IQ2_XS / IQ3_XXS files) keeps each expert's raw GGUF slices, so the blob
// size and the formats change from layer to layer; `native_experts.txt` says how.  Everything that touches an
// expert blob - the arena, the VRAM tier, the prompt path, the CPU pool, the GPU window - asks this table.
#pragma once

#include "strata/kernels/cpu/expert.hpp"
#include "strata/kernels/cpu/native_expert.hpp"

#include <cstdint>
#include <string>
#include <vector>

namespace strata::kernels::cpu {

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
    uint64_t max_blob = BLOB;
    uint64_t total = 0;                   ///< experts.bin size

    uint64_t blob_bytes(int64_t layer) const { return native ? bytes[(size_t) layer] : (uint64_t) BLOB; }
    uint64_t layer_offset(int64_t layer) const {
        return native ? offset[(size_t) layer] : (uint64_t) layer * (uint64_t) n_expert * (uint64_t) BLOB;
    }
    uint64_t blob_offset(int64_t layer, int64_t expert) const {
        return layer_offset(layer) + (uint64_t) expert * blob_bytes(layer);
    }
};

/// Plan v0.3 P6: whether this CPU (and its OS) runs the AVX-512 kernels (F, BW, VL, VNNI, VBMI).  Probed in a
/// file compiled without AVX-512, so asking is safe everywhere; STRATA_FORCE_AVX2=1 answers no (for tests).
bool cpu_avx512_ok();
/// Whether this CPU (and its OS) runs the AVX2 kernels (AVX, AVX2, FMA, F16C): the floor of every expert kernel
/// (q2_avx2.cpp, iq_avx2.cpp, and ggml-cpu in the portable build).  STRATA_FORCE_AVX2 does not change it.
bool cpu_avx2_ok();
/// The CPU's brand string (CPUID 0x80000002..4), for messages; "unknown" when it has none.
std::string cpu_name();
/// Whether this CPU (and its OS) runs the AVX1 kernels: AVX (256-bit float), SSSE3 and SSE4.1, and
/// deliberately NOT FMA3 or F16C, which arrived a generation later than the CPUs this is for.
/// q2_avx1.cpp does a software fp16 decode and a mul+add rather than using them.
bool cpu_avx1_ok();
/// The startup gate for the whole ladder (AVX-512, then AVX2, then AVX1).  Lives in `expert_layout.cpp`
/// rather than calling `cpu_require_expert_support()` because that one is defined in a `/arch:AVX512`
/// translation unit, where the compiler may emit AVX-512 into the error path itself.
void cpu_require_expert_support_any();

// ---- LOCAL PORT (Z620): the canonical expert path, dispatched on what the CPU has.
//
// `pool.cpp` is the CPU expert worker loop.  Every entry point it uses is defined in `expert.cpp`, an
// `/arch:AVX512` translation unit, so calling those on a pre-AVX-512 CPU traps rather than merely running
// slowly.  These wrappers pick the AVX-512 original or the AVX1 port (src/kernels/cpu/s2_expert_avx1.cpp).
// They live in `expert_layout.cpp` because that TU carries no ISA flag, so the feature test is safe to
// evaluate here, and the AVX-512 original is only ever called once the test has returned true.
//
// Each has the same contract as the `s2_expert_*` function it wraps; s2_avx1_parity.cpp checks that
// against an independent reference rather than assuming it.
void s2_expert_vnni_q_any(const uint8_t* blob, const ActQ& a1, float* out, ExpertScratch& ws);
void s2_expert_gu_rows_any(const uint8_t* blob, const ActQ& a1, float* ff, int r0, int r1);
void s2_expert_down_rows_any(const uint8_t* blob, const ActQ& a2, float* out, int r0, int r1);
void s2_expert_gu_rows_multi_any(const uint8_t* blob, const ActQ* const* a1, int n_tokens, float* const* ff, int r0,
                                 int r1);
void s2_expert_down_rows_multi_any(const uint8_t* blob, const ActQ* const* a2, int n_tokens, float* const* out,
                                   int r0, int r1);
void s2_expert_vnni_multi_any(const uint8_t* blob, const ActQ* const* a1, int n_tokens, float* const* out,
                              ExpertScratchMulti& ws);
/// Q2_0 GGUF rows / activation quantizer on the kernels this CPU has.
void q2_rows_any(const uint8_t* w, size_t row_bytes, int nblocks, const ActQ* const* a, int nt, float* const* out,
                 int r0, int r1);
void act_quant_any(const float* x, int n, ActQ& a);

/// The process-wide layout (canonical Q2_0 until `expert_layout_load` finds a native pack).
const ExpertLayout& expert_layout();
/// Reads `<pack_dir>/native_experts.txt` when it exists (a native pack), else sets the canonical layout.
/// Versions up to kExpertLayoutVersion are read; a newer one is refused (a newer packer wrote it).
bool expert_layout_load(const std::string& pack_dir, int64_t n_layers, int64_t n_expert, std::string& err);
/// The newest native_experts.txt this engine reads.  v4 = v3 plus the per-role shard column `gate,up,down`,
/// written only when some layer's roles are in different shards (every other pack stays v3, byte for byte).
inline constexpr int kExpertLayoutVersion = 4;

}  // namespace strata::kernels::cpu
