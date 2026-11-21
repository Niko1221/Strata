#pragma once
// --experts-wex4 (docs/WEX4.md): the routed experts of a native pack read from a WEX4 container instead of experts.bin
// or the GGUF.  WEX4 is the lossless container of github.com/KairaCode/Wex-WRN2: a 4096-byte header page, an index, and
// one 4096-aligned record per expert, each with its own SHA-256; the SHA-256 of the header page (the root) names the
// file.  Decoding a RAW record returns the stored bytes exactly: nothing here changes a weight.
//
// This is a small reader of its own (RAW records only), not the Wex-WRN2 C reader: that one also carries the WNR3 and
// WBP1 codecs, which no Strata pack can use.

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace strata::kernels::cpu { struct ExpertLayout; }

namespace strata::core {

struct Wex4Experts {
    uint64_t file_bytes = 0;
    std::string root;                     ///< 64 hex digits: the SHA-256 of the header page, the container's identity
    std::vector<uint64_t> first, stride;  ///< per layer: expert e is the layer's blob at first + e * stride
    double check_ms = 0;                  ///< the time the header and index checks took
};

/// Reads the header page and the index of `path` (plain reads, no mapping) and checks what the engine relies on:
///   - the header page's SHA-256, the index's SHA-256, the format version, the file size;
///   - for every layer L and expert E of `lay` (a native layout), one RAW record that holds exactly the tensors
///     blk.L.ffn_gate_exps.weight#E, blk.L.ffn_up_exps.weight#E, blk.L.ffn_down_exps.weight#E, in that order, with
///     dtype 128 + the layout's ggml type and the layout's byte sizes, so the record is the blob [gate | up | down];
///   - a layer's records are 4096-aligned, consecutive and one size apart (the stride), and the file has no expert
///     E = n_expert in any layer (a container of a larger model is refused, not read in part);
///   - with `root_hex` (64 hex digits; empty = any), the root.
/// The expert payloads are NOT hashed here (that reads the whole file; `python -I wex4_check.py FILE` does it).
/// On failure `err` names the file, and the layer, expert and rule where one applies.
bool wex4_experts_open(const std::string& path, const strata::kernels::cpu::ExpertLayout& lay,
                       const std::string& root_hex, Wex4Experts& out, std::string& err);

/// Whether `hex` is a root as --experts-wex4-root takes it: 64 hex digits.
bool wex4_root_ok(const std::string& hex);

namespace detail {
/// SHA-256 (FIPS 180-4), portable.  Only the header page and the index are hashed with it (a few MB at open).
void wex4_sha256(const uint8_t* p, size_t n, uint8_t out[32]);
}

}  // namespace strata::core
