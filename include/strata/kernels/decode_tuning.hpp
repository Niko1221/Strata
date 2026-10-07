// include/strata/kernels/decode_tuning.hpp - on-device tuning of the decode kernels' launch shapes (gfx1100).
//
// The decode hot path's GPU experts (`native_expert_grouped`: gate/up, then down) and the i-quant matrix-vector
// kernel (`iq_mmvq`) give one warp to each weight row; how many rows share a thread block is a free choice.  It
// changes occupancy and scheduling but not one floating-point operation: every row is still reduced by the same
// warp, over the same lanes, in the same order, so every variant writes bitwise the same output as the default.
// Which one is fastest depends on the GPU, the driver and the compiler, so it is measured on the machine
// (tools/hip/tune_decode.cpp writes the table, tools/hip/autotune.py runs the whole procedure) instead of guessed.
//
// The table (STRATA_DECODE_TUNING=<path>) is plain text:
//
//   # comments
//   STRATA_DECODE_TUNING_V1 <gpu arch> <HIP runtime version> <toolchain id> <selection space>
//   gu   <ggml type> <n_in> <n_out> <rows per block>      gate/up of the grouped experts (n_in = n_embd, n_out = n_ff)
//   down <ggml type> <n_in> <n_out> <rows per block>      down of the grouped experts   (n_in = n_ff,   n_out = n_embd)
//   mmvq <ggml type> <n_in> <n_out> <rows per block>      iq_mmvq
//
// The header must match the running engine exactly - the architecture (e.g. gfx1100), the HIP runtime, the
// compiler that built the kernels (the tuner checked bitwise equality with THAT build), and the kernel selection
// space (below) - or the table is refused with a message and the defaults stay.  A shape the table does not list
// keeps its default.  CUDA builds compile only the default shapes and ignore the variable (with a message), so
// their behaviour is unchanged.
//
// What a row names: the rows per block of the PLAIN one-warp-per-row kernels (mmvq_kernel / native_gu_kernel /
// native_down_kernel).  The no-table default for split types instead runs the multi decode-once kernels, whose
// rows per block is their own - so the number 16 in a table and the number 16 in the default's launch geometry
// are different mechanisms.  The tuner compares whole configurations against the engine's true default entry
// (native_expert_grouped / iq_mmvq with no table), byte-checks every candidate against it first, and a table row
// replaces the whole call's launch: the kernels the table leaves at their default run the default chain.
#pragma once

#include <cstdint>
#include <istream>
#include <string>
#include <vector>

namespace strata::kernels {

enum class DecodeKernel : int { GateUp = 0, Down = 1, Mmvq = 2 };

/// Defaults (the layouts before tuning existed): 8 rows per block for the grouped experts, 4 for iq_mmvq.
inline constexpr int kDefaultExpertRows = 8;
inline constexpr int kDefaultMmvqRows = 4;

/// Which kernel selection space this build offers the tuner.  The value rides in the table header; a build whose
/// space differs refuses the table ("run the tuner again").  Bump it when the plain kernels' arithmetic or the
/// default dispatch chain changes enough that an older table no longer describes this build - a compiler or
/// runtime bump alone does not change it, a kernel rework does (the sub-warp/EXACT_N rework is the case that
/// motivated the field: same compiler, same runtime, different kernels).
inline constexpr int kDecodeTuningSpace = 2;

/// The rows-per-block variants HIP builds compile besides the default (one list: the parser's validity set, the
/// launchers' compiled-in set and the tuner's candidate list all derive from these two macros).  CUDA builds
/// compile only the defaults.
#define STRATA_EXPERT_ROWS_VARIANTS(X) X(4) X(16) X(2)
#define STRATA_MMVQ_ROWS_VARIANTS(X)  X(2) X(8) X(1)

/// The rows-per-block values a kernel is compiled for (one warp of 32 lanes per row, at most 512 threads).
bool decode_rows_valid(DecodeKernel k, int rows) noexcept;
/// The candidate list the tuner measures for a kernel, default first.
std::vector<int> decode_rows_candidates(DecodeKernel k);
const char* decode_kernel_name(DecodeKernel k) noexcept;
int decode_default_rows(DecodeKernel k) noexcept;

struct DecodeTuningRow {
    DecodeKernel kernel = DecodeKernel::GateUp;
    int type = 0;
    int64_t n_in = 0, n_out = 0;
    int rows = 0;
};

/// Who the kernels were built and run by.  A table is only valid for exactly this identity.
struct DecodeIdentity {
    std::string arch;          ///< e.g. "gfx1100" (feature suffixes such as ":xnack-" stripped); "cuda" on CUDA
    long long runtime = 0;     ///< hipRuntimeGetVersion / cudaRuntimeGetVersion
    std::string toolchain;     ///< 16 hex digits: a hash of the device compiler's version string
    int space = 0;             ///< kDecodeTuningSpace of the build the table was measured on
};

class DecodeTuningTable {
public:
    /// Parse a table; refuses (returns false, `err` says why) a malformed line, an unknown kernel, a rows value
    /// the kernel is not compiled for, a duplicate shape, or a header that does not match `want`.
    bool parse(std::istream& in, const DecodeIdentity& want, std::string& err);
    bool load(const std::string& path, const DecodeIdentity& want, std::string& err);
    /// Rows per block for this shape, or `fallback` when the table does not list it.
    int rows(DecodeKernel k, int type, int64_t n_in, int64_t n_out, int fallback) const noexcept;
    const std::vector<DecodeTuningRow>& entries() const noexcept { return rows_; }
    bool empty() const noexcept { return rows_.empty(); }

    static std::string header(const DecodeIdentity& id);
    static std::string line(const DecodeTuningRow& r);

private:
    std::vector<DecodeTuningRow> rows_;
};

/// A stable 16-hex-digit FNV-1a hash of `s` (the toolchain id).
std::string decode_toolchain_hash(const std::string& s);

// ---- the running engine (implemented in the GPU translation unit) ----

/// This process's identity: the current device's architecture, the runtime version and the kernels' compiler.
DecodeIdentity decode_identity();
/// Load STRATA_DECODE_TUNING once (later calls are no-ops) and say on stderr what was used.  Call it before any
/// graph capture; captured graphs keep the kernels they captured.  Safe to skip: the first launch loads it lazily.
void decode_tuning_init();
/// The active table, or nullptr (none set, refused, or a CUDA build).
const DecodeTuningTable* decode_tuning_active();

}  // namespace strata::kernels
