#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace strata::core {

/// Opt-in native output projection. Uploads unchanged GGUF blocks once at startup and owns its
/// activation scratch. The normal canonical head remains available for numerical A/B comparisons.
/// Plan v0.3 P6: any type the native MMVQ takes (Q5_K in the Q2_0 / IQ3_XXS files, IQ4_XS in IQ2_XS).
/// `load` takes the model's shards (strata::gguf_split_paths): the architecture is checked on the first, the
/// tensor is read from the shard that holds it.
class NativeHead {
public:
    NativeHead() = default;
    ~NativeHead();
    NativeHead(const NativeHead&) = delete;
    NativeHead& operator=(const NativeHead&) = delete;

    bool load(const std::vector<std::string>& shards, int64_t n_in, int64_t n_out, std::string& err);
    /// Route the VRAM weights through a fixed virtual address kept across unload()/load() (prefill-head-alias).  Set once
    /// before the first load; without it the head keeps its plain cudaMalloc behaviour.
    void request_stable_va() { stable_va_ = true; }

    void unload();
    /// Moves the loaded weights from VRAM to mapped pinned host memory (a card too short for it: every logits row then reads them over PCIe).  False: not loaded, already
    /// on the host, or a repacked layout that stays in VRAM.
    bool to_host(std::string& err);
    bool on_host() const { return host_ != nullptr; }
    bool run(const float* mixed, float* logits, void* stream, std::string& err) const;
    uint64_t weight_bytes() const { return bytes_; }
    bool loaded() const { return weights_ != nullptr; }
    /// Plan v0.3 P6: the GGUF blocks, for the verify window's multi-column head.
    const void* weights() const { return weights_; }
    int type() const { return type_; }
    /// The device whose memory holds the weights (the one current when `load` ran; -1 before).
    int device() const { return device_; }
    /// Bytes of one vocabulary row.
    size_t row_bytes() const { return n_out_ > 0 ? (size_t) (bytes_ / (uint64_t) n_out_) : 0; }

private:
    /// Map the physical backing for `bytes` into a fixed VA on `dev`, reserving that VA once; keeps it across unload().  False: VMM unavailable or out of memory.
    bool map_stable_weights(int dev, uint64_t bytes);

    void* weights_ = nullptr;    void* host_ = nullptr;   // the pinned host buffer weights_ aliases when the head did not fit in VRAM
    void* scratch_ = nullptr;
    uint64_t bytes_ = 0;
    int n_in_ = 0, n_out_ = 0;
    int type_ = -1;
    int device_ = -1;

    // VMM backing for the VRAM weights (prefill-head-alias): one address range reserved once on `device_` and kept across
    // unload()/load() cycles, so every captured verify graph keeps reading the same pointer.  Physically mapped while
    // loaded, unmapped+released during a prompt; base stays fixed either way.  (The shared strata::core VmmRange binds one
    // device process-wide, so the head owns its own per-device mapping instead.)
    unsigned long long va_ = 0;      // stable reserved VA; weights_ aliases it while mapped (0 before the first reserve)
    uint64_t vmm_size_ = 0;          // granularity-rounded size of that reservation
    unsigned long long phys_ = 0;    // current physical handle over [va_, va_+vmm_size_), 0 when unmapped
    bool stable_va_ = false;         // set once before the first load: VMM-back the VRAM weights at a fixed VA
};

/// Plan v0.3 P6: `token_embd.weight` in its GGUF form (the IQ model files), in mapped pinned host memory: a row
/// is read over PCIe per token, so the table costs no VRAM.  `embed_row`, the verify window and the MTP drafter
/// use it instead of the canonical table when it is set (`set_native_embed`).  `STRATA_EMBED_VRAM=1` places the
/// same bytes in VRAM instead (slow PCIe links); the gathers read `dev_` either way.
class NativeEmbed {
public:
    NativeEmbed() = default;
    ~NativeEmbed();
    NativeEmbed(const NativeEmbed&) = delete;
    NativeEmbed& operator=(const NativeEmbed&) = delete;
    bool load(const std::vector<std::string>& shards, int64_t n_embd, int64_t n_vocab, std::string& err);
    /// Rows for device token ids.
    void gather_dev(const int32_t* tokens, int64_t n_tok, float* out, void* stream) const;
    /// One row for a host token id.
    void gather_one(int64_t token, float* out, void* stream) const;
    uint64_t bytes() const { return bytes_; }
    int type() const { return type_; }
    /// The table was placed in VRAM (STRATA_EMBED_VRAM=1, or the pinned / alias fallbacks) rather than mapped.
    bool in_vram() const { return dev_ != nullptr && host_ == nullptr; }

private:
    void* host_ = nullptr;
    const void* dev_ = nullptr;
    uint64_t bytes_ = 0;
    size_t row_ = 0;
    int64_t n_embd_ = 0, n_vocab_ = 0;
    int type_ = -1;
};
void set_native_embed(const NativeEmbed* e);
const NativeEmbed* native_embed();

}  // namespace strata::core
