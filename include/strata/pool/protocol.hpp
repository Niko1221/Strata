// include/strata/pool/protocol.hpp - the pool's wire protocol between a coordinator and its workers.
//
// THE SPLIT.  The model's 48 layers are cut into contiguous ranges: the coordinator runs [0, K1), worker 1 runs
// [K1, K2), ..., the last worker runs [Kn, 48).  Each node keeps ONLY its own layers' state: their experts in its
// RAM and VRAM cache, their KV / GDN state for the whole context.  So the pool holds more experts on GPUs than any
// one PC could, and the context's state is divided between the PCs.  A token's residual crosses the network once
// per stage boundary per verify window (12,804 floats per token: the hyper-connection residual, the pending write
// and its inject - what the multi-GPU split hands from card to card through pinned RAM), and once back to the
// coordinator, which runs the output head, the sampler and the draft layer (MTP) on its own GPU.
//
// SESSIONS.  One TCP connection per worker carries one engine session.  The coordinator speaks first after the
// worker's HELLO: AUTH (the HMAC of the worker's nonce under the shared secret), then CONFIG (the layer range and
// everything that must match: context, KV format, window size, rope).  The worker loads its share and answers
// READY.  After that every request message gets exactly one reply, in order; COMMIT, RESET, CKPT_SAVE,
// CKPT_RETAIN and END_REQ are acknowledged lazily (the coordinator drains their ACKs before its next request), so a
// commit costs no round trip of its own.
//
// FRAMES.  A 24-byte little-endian header (magic, type, flags, length) and `length` bytes of payload.  Control
// payloads (HELLO, AUTH, CONFIG, READY, STATS, ERR) are "key=value" lines; data payloads are packed binary.
// Row payloads are float32, or float16 / bfloat16 when the coordinator asks for a lighter wire (--pool-wire).
#pragma once

#include "strata/pool/net.hpp"

#include <cstdint>
#include <cstring>
#include <map>
#include <string>
#include <vector>

namespace strata::pool {

constexpr uint32_t kMagic = 0x4c4f5053;   // "SPOL"
constexpr int kProtocol = 1;
constexpr int kDefaultPort = 7701;

enum class Msg : uint16_t {
    Hello = 1,       // worker -> coordinator, on accept: node facts + nonce
    Auth = 2,        // coordinator -> worker: mac=HMAC(secret, worker nonce), nonce=<coordinator nonce>
    AuthOk = 3,      // worker -> coordinator: mac=HMAC(secret, coordinator nonce)
    Config = 4,      // coordinator -> worker: the range and the settings that must match
    Ready = 5,       // worker -> coordinator: loaded; its cache, chunk, free VRAM
    Verify = 10,     // i32 T, i64 pos0, i32 tokens[T], rows[T x handoff]
    VerifyRows = 11, // rows[T x handoff]
    Commit = 12,     // i32 n_keep                                  (lazy ACK)
    Prefill = 13,    // i64 p0, i64 T, i64 segment, u64 ckpt, i64 tokens[T], rows[T x D]
    PrefillRows = 14,// rows[T x D]
    Reset = 15,      // the session from position 0                 (lazy ACK)
    CkptSave = 16,   // u64 id: keep the running state under id     (lazy ACK)
    CkptRestore = 17,// u64 id: put it back                         (ACK / Err)
    CkptRetain = 18, // u32 n, u64 ids[n]: drop every other id      (lazy ACK)
    EndRequest = 19, // a request ended: refill lent slots, log     (lazy ACK)
    Stats = 20,      // -> StatsReply (key=value)
    StatsReply = 21,
    Ack = 30,
    Err = 31,        // message=<text>
    Bye = 40,        // the coordinator is leaving; the worker keeps its load for the next one
};

const char* msg_name(Msg m);

enum class Wire : uint8_t { F32 = 0, F16 = 1, BF16 = 2 };
bool parse_wire(const std::string& s, Wire& w);
const char* wire_name(Wire w);
inline size_t wire_bytes(Wire w) { return w == Wire::F32 ? 4 : 2; }

/// float32 -> the wire format (`out` holds n * wire_bytes(w) bytes) and back.  F16 rounds to nearest even and
/// saturates to +-inf like IEEE; BF16 rounds to nearest even (NaN kept quiet).
void encode_rows(Wire w, const float* in, size_t n, uint8_t* out);
void decode_rows(Wire w, const uint8_t* in, size_t n, float* out);

/// key=value lines (no '\n' in values).  Unknown keys are ignored by readers, so either side may add facts.
struct KV {
    std::map<std::string, std::string> m;
    void set(const std::string& k, const std::string& v) { m[k] = v; }
    void set(const std::string& k, int64_t v) { m[k] = std::to_string(v); }
    void set(const std::string& k, double v);
    bool has(const std::string& k) const { return m.count(k) != 0; }
    std::string str(const std::string& k, const std::string& d = {}) const;
    int64_t i64(const std::string& k, int64_t d = 0) const;
    double f64(const std::string& k, double d = 0) const;
    std::string encode() const;
    static KV decode(const std::string& s);
    static KV decode(const std::vector<uint8_t>& b) { return decode(std::string(b.begin(), b.end())); }
};

/// Little-endian packing for the binary payloads.
class Packer {
public:
    std::vector<uint8_t> b;
    void put_bytes(const void* p, size_t n) { const uint8_t* c = (const uint8_t*) p; b.insert(b.end(), c, c + n); }
    template <class T> void put(T v) { put_bytes(&v, sizeof v); }   // x86-64 / ARM64 hosts are little-endian
};
class Unpacker {
public:
    Unpacker(const uint8_t* p, size_t n) : p_(p), n_(n) {}
    template <class T> bool get(T& v) {
        if (at_ + sizeof v > n_) return false;
        std::memcpy(&v, p_ + at_, sizeof v);
        at_ += sizeof v;
        return true;
    }
    bool get_bytes(void* dst, size_t n) {
        if (at_ + n > n_) return false;
        std::memcpy(dst, p_ + at_, n);
        at_ += n;
        return true;
    }
    const uint8_t* here() const { return p_ + at_; }
    size_t left() const { return n_ - at_; }
    bool skip(size_t n) { if (at_ + n > n_) return false; at_ += n; return true; }

private:
    const uint8_t* p_;
    size_t n_, at_ = 0;
};

/// One framed connection.  `send` writes a header and up to two payload parts (so a row buffer need not be copied
/// behind its small binary prefix); `recv` reads one frame.  Not thread-safe: one thread per channel at a time.
class Channel {
public:
    Channel() = default;
    explicit Channel(Socket s) : s_(std::move(s)) {}
    bool valid() const { return s_.valid(); }
    Socket& socket() { return s_; }
    const std::string& peer() const { return s_.peer(); }
    void close() { s_.close(); }

    bool send(Msg type, const void* a, size_t na, const void* b, size_t nb, std::string& err);
    bool send(Msg type, const std::vector<uint8_t>& p, std::string& err) { return send(type, p.data(), p.size(), nullptr, 0, err); }
    bool send(Msg type, const KV& kv, std::string& err) {
        const std::string s = kv.encode();
        return send(type, s.data(), s.size(), nullptr, 0, err);
    }
    bool send(Msg type, std::string& err) { return send(type, nullptr, 0, nullptr, 0, err); }
    /// One frame into `type` / `payload`.  `max_bytes` refuses a frame longer than that (a stray connection).
    bool recv(Msg& type, std::vector<uint8_t>& payload, std::string& err, uint64_t max_bytes = (uint64_t) 4 << 30);
    /// A frame whose payload is read straight into the caller's buffer after a `prefix` of known size: rows land
    /// in pinned memory without a staging copy.  The frame must be exactly prefix + rows bytes.
    bool recv_into(Msg& type, std::vector<uint8_t>& prefix, size_t prefix_bytes, void* rows, size_t rows_bytes,
                   std::string& err);

    /// Lazily acknowledged requests in flight: `drain` reads their ACKs (an Err fails it with the worker's text).
    int pending = 0;
    bool drain(std::string& err);
    std::vector<uint8_t> last_ack;   ///< the payload of the last ACK `drain` read (most are empty)

    uint64_t bytes_out = 0, bytes_in = 0;

private:
    Socket s_;
};

/// The worker's text of an Err frame, or a description of an unexpected frame type.
std::string unexpected(Msg got, Msg want, const std::vector<uint8_t>& payload);

}  // namespace strata::pool
