#pragma once
// Opt-in Qwen3.6 expert pack. The engine default stays the legacy host arena.
// This path writes the same bytes as expert_blob() into a file and maps that
// file. It does not cudaHostAlloc or malloc the full expert arena.
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace strata::qwen35 {

inline constexpr char kExpertPackMagic[8] = {'Q', '3', '6', 'E', 'P', 'K', '1', '\0'};
inline constexpr uint32_t kExpertPackVersion = 1;
inline constexpr uint32_t kExpertPackPrefixBytes = 128;
inline constexpr uint32_t kExpertPackRecordBytes = 128;
inline constexpr uint32_t kExpertPackVersionField = 8;
inline constexpr uint32_t kExpertPackPayloadOffsetField = 64;
inline constexpr uint64_t kExpertPackAlign = 4096;
inline constexpr uint64_t kQ36ExpertTail = 4096;

struct ExpertBlobLayout {
    uint32_t gu_type = 0;
    uint32_t d_type = 0;
    uint32_t n_embd = 0;
    uint32_t n_ff = 0;
    uint32_t n_expert = 0;
    uint64_t gu_row = 0;
    uint64_t d_row = 0;
    uint64_t up_off = 0;
    uint64_t down_off = 0;
    uint64_t blob_bytes = 0;
    uint64_t stride = 0;
};

struct ExpertSource {
    const uint8_t* gate = nullptr;
    const uint8_t* up = nullptr;
    const uint8_t* down = nullptr;
    uint64_t src_gsz = 0;
    uint64_t src_dsz = 0;
};

struct PackLayer {
    ExpertBlobLayout layout;
    ExpertSource source;
};

struct PackBuildRequest {
    std::string model_path;
    std::vector<PackLayer> layers;
    uint64_t tail = kQ36ExpertTail;
    // Opt-in Windows runtime reader. Requires an existing verified pack; never
    // creates/replaces/repackages model data on this path.
    bool bounded_reads = false;
};

// One expert: gate at 0, up at up_off, down at down_off, everything else in the
// stride zero. Returns false without writing when the copies would overlap or
// run past the stride.
bool pack_expert_blob(const ExpertBlobLayout& layout, const ExpertSource& src, uint32_t expert, uint8_t* dst);

class MappedExpertPack {
public:
    static constexpr std::size_t kBoundedReadScratchBytes = 1u << 20;
    MappedExpertPack();
    ~MappedExpertPack();
    MappedExpertPack(MappedExpertPack&&) noexcept;
    MappedExpertPack& operator=(MappedExpertPack&&) noexcept;
    MappedExpertPack(const MappedExpertPack&) = delete;
    MappedExpertPack& operator=(const MappedExpertPack&) = delete;

    const uint8_t* layer_data(uint32_t layer) const;
    uint64_t layer_bytes(uint32_t layer) const;
    uint32_t layer_count() const;
    const std::string& path() const;

    // Read the bytes represented by a mapped range into caller-owned staging.
    // The same open file is used; the source mapping is not touched. This keeps
    // a whole-layer GPU upload from faulting the full expert pack into the
    // process working set. Concurrent reads use independent file offsets.
    // Invalid ranges are rejected before writing. An I/O error may leave a
    // partial result, which the caller must not submit to the GPU.
    bool read_mapped(const uint8_t* src, uint8_t* dst, std::size_t bytes) const;

    // Enable before concurrent reads. Reopens the SAME file object with no
    // buffering, owns a serialized 1 MiB aligned bounce buffer, fails closed.
    // Existing mapped ranges become opaque addresses; payload is not touched.
    bool enable_bounded_reads(std::string& err);
    uint64_t bounded_read_scratch_bytes() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
    friend std::unique_ptr<MappedExpertPack> open_or_build_expert_pack(const PackBuildRequest& req, std::string& err);
};

// Reuses a pack only when the model hash, geometry, and payload hash match.
// A failed build deletes its .partial file and does not fall back to an
// anonymous arena. A bad published file is moved aside as .rejected.
std::unique_ptr<MappedExpertPack> open_or_build_expert_pack(const PackBuildRequest& req, std::string& err);

// Empty string means the file matches the model and, when expected is set, that geometry.
std::string expert_pack_verify(const std::string& pack_path, const std::string& model_path,
                               const std::vector<PackLayer>* expected, uint64_t tail, bool reuse_verified = false);
// Reuse is supported only with read-locked NTFS file identities and USNs.
// The default above always hashes every byte; STRATA_Q36_VERIFY_PACK=1 also
// forces full verification on the engine path. Geometry is checked every time.
bool expert_pack_last_verification_cached();

std::string expert_pack_sha256_file(const std::string& path, std::string& err);
uint64_t expert_pack_last_scratch_bytes();

// Bounded host staging for mapped-pack (T05-B). Default proposal is 2 x 32 MiB.
// This is separate from the 1.10 GiB GPU staging pair. The mapped-pack engine
// owns the pinned (or pageable fallback) buffers and gives this pool a
// non-owning view. A slot stays in flight until its CUDA copy completes.
enum class PinSlotState : uint8_t { Reusable = 0, HostFilling = 1, H2dInFlight = 2 };

class ExpertPinPool {
public:
    static constexpr uint32_t kDefaultSlots = 2;
    static constexpr uint64_t kDefaultSlotBytes = 32ull << 20;  // 32 MiB

    explicit ExpertPinPool(uint32_t slots = kDefaultSlots, uint64_t slot_bytes = kDefaultSlotBytes);
    // Non-owning view of externally allocated buffers (e.g. cudaHostAlloc). Does not free them.
    ExpertPinPool(uint8_t* const* external, uint32_t slots, uint64_t slot_bytes);
    ExpertPinPool(const ExpertPinPool&) = delete;
    ExpertPinPool& operator=(const ExpertPinPool&) = delete;
    ExpertPinPool(ExpertPinPool&&) noexcept;
    ExpertPinPool& operator=(ExpertPinPool&&) noexcept;
    ~ExpertPinPool();

    uint32_t slot_count() const;
    uint64_t slot_bytes() const;
    uint64_t committed_bytes() const;
    bool owns_buffers() const;
    PinSlotState state(int slot) const;
    uint8_t* data(int slot);
    const uint8_t* data(int slot) const;

    // Reusable -> HostFilling. Returns slot index, or -1 when none are free.
    int begin_fill();
    // HostFilling -> H2dInFlight. False on a bad slot or wrong state.
    bool mark_h2d(int slot);
    // HostFilling or H2dInFlight -> Reusable. False on a bad slot or Reusable.
    bool release(int slot);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace strata::qwen35
