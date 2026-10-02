// src/pool/protocol.cpp - the pool's frames, key=value control payloads and the row wire formats.
#include "strata/pool/protocol.hpp"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace strata::pool {

namespace {
#pragma pack(push, 1)
struct Header {
    uint32_t magic;
    uint16_t type;
    uint16_t flags;
    uint64_t length;
    uint64_t reserved;
};
#pragma pack(pop)
static_assert(sizeof(Header) == 24, "the frame header is 24 bytes");
}  // namespace

const char* msg_name(Msg m) {
    switch (m) {
    case Msg::Hello: return "HELLO";
    case Msg::Auth: return "AUTH";
    case Msg::AuthOk: return "AUTH_OK";
    case Msg::Config: return "CONFIG";
    case Msg::Ready: return "READY";
    case Msg::Verify: return "VERIFY";
    case Msg::VerifyRows: return "VERIFY_ROWS";
    case Msg::Commit: return "COMMIT";
    case Msg::Prefill: return "PREFILL";
    case Msg::PrefillRows: return "PREFILL_ROWS";
    case Msg::Reset: return "RESET";
    case Msg::CkptSave: return "CKPT_SAVE";
    case Msg::CkptRestore: return "CKPT_RESTORE";
    case Msg::CkptRetain: return "CKPT_RETAIN";
    case Msg::EndRequest: return "END_REQUEST";
    case Msg::Stats: return "STATS";
    case Msg::StatsReply: return "STATS_REPLY";
    case Msg::Ack: return "ACK";
    case Msg::Err: return "ERR";
    case Msg::Bye: return "BYE";
    }
    return "?";
}

bool parse_wire(const std::string& s, Wire& w) {
    if (s == "f32" || s == "fp32" || s == "float32") { w = Wire::F32; return true; }
    if (s == "f16" || s == "fp16" || s == "float16" || s == "half") { w = Wire::F16; return true; }
    if (s == "bf16" || s == "bfloat16") { w = Wire::BF16; return true; }
    return false;
}

const char* wire_name(Wire w) { return w == Wire::F32 ? "f32" : w == Wire::F16 ? "f16" : "bf16"; }

// ------------------------------------------------------------------------------------------------ row codecs
namespace {
inline uint32_t bits_of(float f) { uint32_t u; std::memcpy(&u, &f, 4); return u; }
inline float float_of(uint32_t u) { float f; std::memcpy(&f, &u, 4); return f; }

inline uint16_t f32_to_f16(float f) {
    const uint32_t x = bits_of(f);
    const uint32_t sign = (x >> 16) & 0x8000u;
    const uint32_t ex = (x >> 23) & 0xffu;
    uint32_t man = x & 0x7fffffu;
    if (ex == 0xffu) return (uint16_t) (sign | 0x7c00u | (man ? 0x200u : 0u));     // inf / NaN (quiet)
    int32_t e = (int32_t) ex - 127 + 15;
    if (e >= 0x1f) return (uint16_t) (sign | 0x7c00u);                             // overflow: inf
    if (e <= 0) {                                                                   // subnormal or zero
        if (e < -10) return (uint16_t) sign;
        man |= 0x800000u;
        const int shift = 14 - e;
        uint32_t h = man >> shift;
        const uint32_t rem = man & ((1u << shift) - 1u), half = 1u << (shift - 1);
        if (rem > half || (rem == half && (h & 1u))) ++h;
        return (uint16_t) (sign | h);
    }
    uint32_t h = ((uint32_t) e << 10) | (man >> 13);
    const uint32_t rem = man & 0x1fffu;
    if (rem > 0x1000u || (rem == 0x1000u && (h & 1u))) ++h;                         // may carry into the exponent
    return (uint16_t) (sign | h);
}

inline float f16_to_f32(uint16_t h) {
    const uint32_t sign = (uint32_t) (h & 0x8000u) << 16;
    uint32_t ex = (h >> 10) & 0x1fu, man = h & 0x3ffu;
    if (ex == 0x1fu) return float_of(sign | 0x7f800000u | (man << 13));
    if (ex == 0) {
        if (man == 0) return float_of(sign);
        int e = -1;
        do { ++e; man <<= 1; } while ((man & 0x400u) == 0);
        man &= 0x3ffu;
        return float_of(sign | ((uint32_t) (127 - 15 - e) << 23) | (man << 13));
    }
    return float_of(sign | ((ex + 127 - 15) << 23) | (man << 13));
}

inline uint16_t f32_to_bf16(float f) {
    uint32_t x = bits_of(f);
    if ((x & 0x7fffffffu) > 0x7f800000u) return (uint16_t) ((x >> 16) | 0x40u);   // NaN stays NaN (quiet)
    x += 0x7fffu + ((x >> 16) & 1u);
    return (uint16_t) (x >> 16);
}

inline float bf16_to_f32(uint16_t b) { return float_of((uint32_t) b << 16); }
}  // namespace

void encode_rows(Wire w, const float* in, size_t n, uint8_t* out) {
    if (w == Wire::F32) { std::memcpy(out, in, n * 4); return; }
    uint16_t* o = (uint16_t*) out;
    if (w == Wire::F16)
        for (size_t i = 0; i < n; ++i) o[i] = f32_to_f16(in[i]);
    else
        for (size_t i = 0; i < n; ++i) o[i] = f32_to_bf16(in[i]);
}

void decode_rows(Wire w, const uint8_t* in, size_t n, float* out) {
    if (w == Wire::F32) { std::memcpy(out, in, n * 4); return; }
    const uint16_t* p = (const uint16_t*) in;
    if (w == Wire::F16)
        for (size_t i = 0; i < n; ++i) out[i] = f16_to_f32(p[i]);
    else
        for (size_t i = 0; i < n; ++i) out[i] = bf16_to_f32(p[i]);
}

// ------------------------------------------------------------------------------------------------ KV
void KV::set(const std::string& k, double v) {
    char b[64];
    std::snprintf(b, sizeof b, "%.17g", v);
    m[k] = b;
}

std::string KV::str(const std::string& k, const std::string& d) const {
    const auto it = m.find(k);
    return it == m.end() ? d : it->second;
}

int64_t KV::i64(const std::string& k, int64_t d) const {
    const auto it = m.find(k);
    if (it == m.end() || it->second.empty()) return d;
    char* e = nullptr;
    const long long v = std::strtoll(it->second.c_str(), &e, 10);
    return (e && *e == '\0') ? (int64_t) v : d;
}

double KV::f64(const std::string& k, double d) const {
    const auto it = m.find(k);
    if (it == m.end() || it->second.empty()) return d;
    char* e = nullptr;
    const double v = std::strtod(it->second.c_str(), &e);
    return (e && *e == '\0') ? v : d;
}

std::string KV::encode() const {
    std::string s;
    for (const auto& [k, v] : m) {
        std::string clean = v;
        for (char& c : clean) if (c == '\n' || c == '\r') c = ' ';
        s += k + "=" + clean + "\n";
    }
    return s;
}

KV KV::decode(const std::string& s) {
    KV kv;
    size_t a = 0;
    while (a < s.size()) {
        size_t b = s.find('\n', a);
        if (b == std::string::npos) b = s.size();
        const std::string line = s.substr(a, b - a);
        const size_t eq = line.find('=');
        if (eq != std::string::npos && eq > 0) kv.m[line.substr(0, eq)] = line.substr(eq + 1);
        a = b + 1;
    }
    return kv;
}

// ------------------------------------------------------------------------------------------------ Channel
bool Channel::send(Msg type, const void* a, size_t na, const void* b, size_t nb, std::string& err) {
    Header h{kMagic, (uint16_t) type, 0, (uint64_t) (na + nb), 0};
    if (!s_.send_all(&h, sizeof h, err)) return false;
    if (na > 0 && !s_.send_all(a, na, err)) return false;
    if (nb > 0 && !s_.send_all(b, nb, err)) return false;
    bytes_out += sizeof h + na + nb;
    return true;
}

bool Channel::recv(Msg& type, std::vector<uint8_t>& payload, std::string& err, uint64_t max_bytes) {
    Header h{};
    if (!s_.recv_all(&h, sizeof h, err)) return false;
    if (h.magic != kMagic) { err = "not a Strata pool peer (" + peer() + ": bad frame magic)"; return false; }
    if (h.length > max_bytes) { err = "a frame of " + std::to_string(h.length) + " bytes from " + peer() + " is too large"; return false; }
    type = (Msg) h.type;
    payload.resize((size_t) h.length);
    if (h.length > 0 && !s_.recv_all(payload.data(), (size_t) h.length, err)) return false;
    bytes_in += sizeof h + h.length;
    return true;
}

bool Channel::recv_into(Msg& type, std::vector<uint8_t>& prefix, size_t prefix_bytes, void* rows, size_t rows_bytes,
                        std::string& err) {
    Header h{};
    if (!s_.recv_all(&h, sizeof h, err)) return false;
    if (h.magic != kMagic) { err = "not a Strata pool peer (" + peer() + ": bad frame magic)"; return false; }
    type = (Msg) h.type;
    bytes_in += sizeof h + h.length;
    if (h.length != (uint64_t) (prefix_bytes + rows_bytes)) {
        // an Err or anything unexpected: read it whole for the caller to report
        if (h.length > ((uint64_t) 1 << 20) && type != Msg::Err) {
            err = "an unexpected " + std::string(msg_name(type)) + " frame of " + std::to_string(h.length) + " bytes";
            return false;
        }
        prefix.resize((size_t) h.length);
        if (h.length > 0 && !s_.recv_all(prefix.data(), (size_t) h.length, err)) return false;
        return true;   // the caller checks type and size
    }
    prefix.resize(prefix_bytes);
    if (prefix_bytes > 0 && !s_.recv_all(prefix.data(), prefix_bytes, err)) return false;
    if (rows_bytes > 0 && !s_.recv_all(rows, rows_bytes, err)) return false;
    return true;
}

bool Channel::drain(std::string& err) {
    std::vector<uint8_t> p;
    while (pending > 0) {
        Msg t;
        if (!recv(t, p, err, 1 << 20)) return false;
        --pending;
        if (t == Msg::Err) { err = peer() + ": " + KV::decode(p).str("message", "error"); return false; }
        if (t != Msg::Ack) { err = unexpected(t, Msg::Ack, p); return false; }
        last_ack = p;
    }
    return true;
}

std::string unexpected(Msg got, Msg want, const std::vector<uint8_t>& payload) {
    if (got == Msg::Err) return KV::decode(payload).str("message", "the worker reported an error");
    return std::string("expected ") + msg_name(want) + ", got " + msg_name(got);
}

}  // namespace strata::pool
