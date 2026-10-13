// The running state of a conversation (the GDN recurrence and conv history, the PLE history, the indexer's tail and
// spare row) is fp32 where the engine computes it. Where a conversation is STORED - a parked conversation-cache
// image, a session file - that state can be coded to fewer bytes. Nothing here touches the live state or the live
// checkpoint chain: a codec is a property of a stored copy, and every coded array is widened back to fp32 before a
// kernel can read it.
//
// bfloat16 is the top half of a float32, so it keeps fp32's exponent range: no scale to choose, no value that can
// overflow into another value, no subnormal cliff. Encode rounds to nearest even and bounds the error at 2^-8
// relative, per value (bfloat16 carries 8 significant bits); decode is a shift and is exact. Below bf16's own
// subnormal range a value flushes to a zero of its sign. The format that ships pre-quantized in the model's PLE
// table relies on the same property (`bf16_dequant_row`, strata/kernels/ngram.hpp).
//
// int8 and fp8 were considered and left for later: both need a scale chosen for data whose range is only known per
// conversation, and both change what counts as a corrupted file. `StateCodec` is the seam they would enter through.
#pragma once

#include <cstddef>
#include <cstdint>
#include <string>

namespace strata::core {

enum class StateCodec : uint8_t { Fp32 = 0, Bf16 = 1 };

// The codec an option names ("fp32", "bf16", case-insensitive). False for an unknown or empty name: the caller
// refuses it at start-up rather than silently falling back to a default.
bool state_codec_by_name(const std::string& name, StateCodec& out);
// The name to print and to accept back. nullptr for a codec this build does not know.
const char* state_codec_name(StateCodec codec);
// A codec code read from a session file. Only a code this build knows is accepted, and only as a whole byte.
bool state_codec_by_code(uint64_t code, StateCodec& out);

// The bytes `fp32_bytes` of fp32 values occupy under `codec`. `fp32_bytes` is always a whole number of floats: every
// running-state array is built from float and int32 counts. `Fp32` is the identity.
size_t state_codec_size(size_t fp32_bytes, StateCodec codec);
// The inverse: the fp32 bytes of the values a `coded_bytes` array under `codec` holds. A size that is not a whole
// number of coded values returns 0, which every caller treats as a refusal.
size_t state_codec_fp32_bytes(size_t coded_bytes, StateCodec codec);

// One array of `floats` fp32 values. `dst` holds state_codec_size(floats * 4, codec) bytes and must not overlap
// `src`. Null pointers are a failure only when there is something to convert.
bool state_encode(const float* src, size_t floats, StateCodec codec, uint8_t* dst, std::string& error);
bool state_decode(const uint8_t* src, size_t floats, StateCodec codec, float* dst, std::string& error);

// The same `floats` values, held under `from` in `src` and written under `to` in `dst`, without going through fp32
// in between. `dst` holds state_codec_size(floats * 4, to) bytes and must not overlap `src`; a `from` that is already
// `to` copies the bytes that array occupies. This is the host-to-host conversion a park and a mount do on stored
// state: state_encode/state_decode move values to or from fp32 arrays, which for these two codecs means one extra
// full-size pass over the data (a memcpy into scratch on the way in, a memcpy out on the way back). The rounding,
// NaN, signed-zero, subnormal and overflow behaviour is the same code both paths use.
bool state_transcode_bytes(const uint8_t* src, size_t floats, StateCodec from, StateCodec to, uint8_t* dst,
                           std::string& error);

// The largest array one call converts at a time when a coded array crosses the device boundary: the copy is chunked
// so that a coded save or restore needs this much host memory and no more.
constexpr size_t kStateCodecChunkFloats = 1u << 20;   // 4 MiB of fp32 values

} // namespace strata::core