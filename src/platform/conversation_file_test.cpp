#include "strata/platform/conversation_file.hpp"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <limits>
#include <sstream>

using namespace strata::core;
using namespace strata::platform;
namespace {
int checks = 0;
int heartbeats = 0;
void heartbeat() noexcept { ++heartbeats; }
void check(bool ok, const char* label) {
    ++checks;
    if (!ok) { std::fprintf(stderr, "FAIL: %s\n", label); std::exit(1); }
}

bool same(const ConversationCheckpoint& a, const ConversationCheckpoint& b) {
    return a.ids == b.ids && a.imgs == b.imgs && a.gdn == b.gdn && a.ple == b.ple &&
        a.tails == b.tails && a.dead == b.dead && a.block_pos == b.block_pos && a.used == b.used &&
        a.stage_parts.empty() && b.stage_parts.empty();
}
bool same(const SavedConversation& a, const SavedConversation& b) {
    if (a.geometry != b.geometry || a.layer_lo != b.layer_lo || a.layer_hi != b.layer_hi || a.cvec != b.cvec || !same(a.live, b.live) ||
        a.checkpoints.size() != b.checkpoints.size() || a.kv.size() != b.kv.size()) return false;
    for (size_t i = 0; i < a.checkpoints.size(); ++i)
        if (!same(a.checkpoints[i], b.checkpoints[i])) return false;
    for (size_t i = 0; i < a.kv.size(); ++i) {
        const auto& x = a.kv[i]; const auto& y = b.kv[i];
        if (x.format != y.format || x.cells != y.cells || x.heads != y.heads || x.head_dim != y.head_dim ||
            x.page_size != y.page_size || x.pooled_rows != y.pooled_rows || x.idx_dim != y.idx_dim ||
            x.k != y.k || x.v != y.v || x.k_scale != y.k_scale || x.v_scale != y.v_scale || x.pooled != y.pooled)
            return false;
    }
    return true;
}

SavedConversation fixture() {
    SavedConversation s;
    for (size_t i = 0; i < s.geometry.size(); ++i) s.geometry[i] = int64_t(i) - 8;
    s.layer_lo = 7; s.layer_hi = 48;
    s.cvec = false;
    s.live.ids = {1, -1, INT32_MIN, INT32_MAX};
    s.live.imgs = {{INT64_MIN, UINT64_MAX}, {3, 0x123456789abcdef0ULL}};
    s.live.used = UINT64_MAX;
    s.live.gdn = {0, 1, 255}; s.live.ple = {2, 3}; s.live.tails = {4};
    s.live.dead = {5, 6, 7}; s.live.block_pos = {8, 9};
    s.checkpoints = {s.live, {}};
    s.checkpoints[0].ids.pop_back();
    // Payload sizes deliberately do not describe a model. The codec preserves
    // bytes; the shared core, tested separately, validates model geometry.
    for (int format = 0; format < 4; ++format) {
        ConversationKv k;
        k.format = format; k.cells = 17; k.heads = 2; k.head_dim = 128;
        k.page_size = 16; k.pooled_rows = 3; k.idx_dim = 64;
        k.k = {uint8_t(format), 0, 255}; k.v = {7, 8}; k.k_scale = {9};
        k.v_scale = {10, 11}; k.pooled = {12, 13};
        s.kv.push_back(std::move(k));
    }
    return s;
}
std::string encode(const SavedConversation& s, const ConversationIdentity& id) {
    std::ostringstream stream(std::ios::binary);
    std::string error;
    check(conversation_file_write(stream, s, id, error), "encode fixture");
    return stream.str();
}
uint64_t integer(const std::string& bytes, size_t offset) {
    uint64_t n = 0;
    for (size_t i = 0; i < 8; ++i) n |= uint64_t(uint8_t(bytes.at(offset + i))) << (i * 8);
    return n;
}
void set_integer(std::string& bytes, size_t offset, uint64_t n) {
    for (size_t i = 0; i < 8; ++i) bytes.at(offset + i) = char(n >> (i * 8));
}
void rejected(const std::string& bytes, const ConversationIdentity& id, uint64_t budget = 1 << 20,
              std::optional<uint64_t> available = 1 << 21, uint64_t floor = 0) {
    const auto sentinel = fixture();
    auto output = sentinel;
    std::istringstream stream(bytes, std::ios::binary);
    std::string error;
    check(!conversation_file_read(stream, id, budget, available, floor, output, error), "invalid file declined");
    check(!error.empty(), "decline explains cause");
    check(same(output, sentinel), "decline preserves caller image");
}

struct TempDirectory {
    std::filesystem::path path;
    TempDirectory() {
        const auto tick = std::chrono::steady_clock::now().time_since_epoch().count();
        for (unsigned i = 0; i < 100; ++i) {
            auto candidate = std::filesystem::temp_directory_path() /
                ("strata-file-test-" + std::to_string(tick) + "-" + std::to_string(i));
            if (std::filesystem::create_directory(candidate)) { path = std::move(candidate); return; }
        }
        check(false, "create private test directory");
    }
    ~TempDirectory() { std::error_code ec; std::filesystem::remove_all(path, ec); }
};
void write(const std::filesystem::path& path, const std::string& bytes) {
    std::ofstream f(path, std::ios::binary);
    f.write(bytes.data(), bytes.size()); f.close();
    check(bool(f), "write identity fixture");
}
std::string hex(const ConversationIdentity& identity) {
    constexpr char digits[] = "0123456789abcdef";
    std::string out;
    for (uint8_t b : identity.digest) { out += digits[b >> 4]; out += digits[b & 15]; }
    return out;
}
struct CountingBuffer : std::stringbuf {
    explicit CountingBuffer(const std::string& bytes) : std::stringbuf(bytes, std::ios::in) {}
    size_t read_bytes = 0;
    std::streamsize xsgetn(char* s, std::streamsize n) override {
        const auto got = std::stringbuf::xsgetn(s, n);
        read_bytes += size_t(got);
        return got;
    }
};
} // namespace

int main() {
    ConversationIdentity id{};
    for (size_t i = 0; i < id.digest.size(); ++i) id.digest[i] = uint8_t(i);
    const auto source = fixture();
    const auto bytes = encode(source, id);
    uint64_t measured = 0;
    std::string sizing_error;
    check(conversation_file_size(source, id, measured, sizing_error) && measured == bytes.size(), "quota reservation equals encoded fixture length");
    const auto bound = integer(bytes, 48);
    check(bytes.substr(0, 8) == std::string("STRSNAP\2", 8), "versioned magic");
    check(integer(bytes, 56) == uint64_t(-8), "geometry signed bits are little endian");
    check(integer(bytes, 216) == 0, "steering field");
    check(integer(bytes, 224) == 4, "live token count");
    check(uint8_t(bytes[236]) == 255 && uint8_t(bytes[239]) == 255, "negative tokens keep all bits");
    check(bound >= source.bytes(), "portable allocation bound covers decoded payload");
    std::string error;
    SavedConversation decoded;
    std::istringstream input(bytes, std::ios::binary);
    check(conversation_file_read(input, id, bound, bound + 17, 17, decoded, error), "exact staging and floor admitted");
    check(same(source, decoded), "all formats, checkpoints and buffers survive round trip");
    {
        auto segmented = source, canonical = source;
        segmented.kv[0].k = {};
        canonical.kv[0].k = {};
        for (size_t n=1;n<=17;++n) segmented.kv[0].k.resize(n*1024*1024,71);
        canonical.kv[0].k.resize(17*1024*1024,71);
        const auto encoded = encode(segmented,id);
        check(encoded == encode(canonical,id), "disk bytes are independent of incremental buffer segmentation");
        const auto staging = integer(encoded,48);
        std::istringstream read(encoded,std::ios::binary);
        SavedConversation roundtrip;
        check(conversation_file_read(read,id,staging,staging+17,17,roundtrip,error), "read segmented snapshot within exact staging bound");
        check(same(segmented,roundtrip) && roundtrip.bytes() <= staging, "segmented snapshot round trip preserves payload and admission");
    }
    rejected(bytes, id, bound - 1);
    rejected(bytes, id, bound, bound + 16, 17);
    rejected(bytes, id, bound, std::nullopt);
    rejected(bytes, id, bound, 16, 17);
    rejected(bytes, id, bound, UINT64_MAX, UINT64_MAX);
    auto foreign = id; foreign.digest[0] ^= 1;
    rejected(bytes, foreign);
    rejected(bytes + 'x', id);
    for (size_t n = 0; n < bytes.size(); ++n) rejected(bytes.substr(0, n), id);
    for (size_t i = 0; i < bytes.size(); ++i) {
        auto damaged = bytes; damaged[i] ^= 1; rejected(damaged, id);
    }
    auto corrupt = bytes;
    set_integer(corrupt, 224, UINT64_MAX); rejected(corrupt, id);
    corrupt = bytes; set_integer(corrupt, 48, UINT64_MAX); rejected(corrupt, id);
    corrupt = bytes; set_integer(corrupt, 48, kConversationFileWorkspace); rejected(corrupt, id);
    corrupt = bytes; set_integer(corrupt, 216, 2); rejected(corrupt, id);

    auto unsupported = source; unsupported.live.stage_parts.emplace_back();
    std::ostringstream output;
    check(!conversation_file_write(output, unsupported, id, error), "layer split is refused");
    check(output.str().empty(), "unsupported image refused before writing header");
    output.setstate(std::ios::badbit);
    check(!conversation_file_write(output, source, id, error), "write failure is reported");
    std::istringstream broken(bytes);
    broken.setstate(std::ios::badbit);
    check(!conversation_file_read(broken, id, bound, bound, 0, decoded, error), "read failure is reported");
    const auto empty = encode({}, id);
    check(conversation_file_size({}, id, measured, sizing_error) && measured == empty.size(), "empty envelope reservation includes framing");
    std::istringstream empty_input(empty);
    check(conversation_file_read(empty_input, id, 1 << 20, 1 << 21, 0, decoded, error), "empty envelope round trip");
    check(same(decoded, {}), "empty image replaces earlier output");

    auto indexed = source;
    indexed.live.ids = {1, 2, 3, 4, 5, 6}; indexed.live.imgs = {{2, 17}};
    indexed.live.gdn.resize(1 << 20, 42);
    indexed.checkpoints = {indexed.live, indexed.live};
    indexed.checkpoints[0].ids.resize(2); indexed.checkpoints[0].imgs.clear();
    indexed.checkpoints[1].ids.resize(4);
    const auto indexed_bytes = encode(indexed, id);
    const auto indexed_bound = integer(indexed_bytes, 48);
    std::ostringstream progressing;
    heartbeats = 0;
    check(conversation_file_write(progressing, indexed, id, error, heartbeat) && heartbeats >= 3,
          "large writes report incremental progress before completion");
    check(progressing.str() == indexed_bytes, "progress reporting leaves file bytes unchanged");
    std::istringstream progressing_read(indexed_bytes);
    heartbeats = 0;
    check(conversation_file_read(progressing_read, id, indexed_bound, indexed_bound * 2, 0, decoded, error, heartbeat) && heartbeats >= 3,
          "large reads report incremental progress");
    std::istringstream denied_read(indexed_bytes);
    heartbeats = 0;
    check(!conversation_file_read(denied_read, id, indexed_bound, std::nullopt, 0, decoded, error, heartbeat) && heartbeats == 0,
          "denied staging does not invent progress");
    std::ostringstream failed_write;
    failed_write.setstate(std::ios::badbit);
    check(!conversation_file_write(failed_write, indexed, id, error, heartbeat) && heartbeats == 0,
          "failed initial write does not invent progress");
    ConversationCache ram(indexed_bound, 2);
    auto ram_copy = indexed;
    check(ram.put(std::move(ram_copy)), "RAM reference accepts prefix fixture");
    for (size_t length = 0; length < 9; ++length) {
        std::vector<int64_t> prompt;
        for (size_t i = 0; i < length; ++i) prompt.push_back(int32_t(i + 1));
        for (bool steering : {false, true}) {
            for (const auto& images : std::vector<std::vector<ConversationImageKey>>{
                    {}, {{2, 17}}, {{2, 18}}, {{2, 17}, {7, 19}}}) {
                CountingBuffer buffer(indexed_bytes);
                std::istream stream(&buffer);
                ConversationFileMatch match;
                check(conversation_file_match(stream, id, indexed_bound, prompt, images, steering, match, error),
                      "read disk prefix metadata");
                const auto expected = ram.best(prompt, images, steering);
                check(match.tokens == expected.tokens && (match.tokens == 0 || match.live == expected.live),
                      "disk matching agrees with RAM for live/checkpoint/image/steering cases");
                check(match.staging_bytes == indexed_bound, "candidate reports staging allocation");
                check(buffer.read_bytes < 1024, "prefix search seeks over multi-megabyte state");
            }
        }
    }
    for (size_t length = 0; length < 248; ++length) {
        std::istringstream short_file(indexed_bytes.substr(0, length));
        ConversationFileMatch match{37, true, 123};
        check(!conversation_file_match(short_file, id, indexed_bound, {1, 2, 3}, {}, false, match, error),
              "truncated prefix metadata is declined");
        check(match.tokens == 37 && match.staging_bytes == 123, "failed probe preserves output");
    }
    std::istringstream foreign_file(indexed_bytes);
    ConversationFileMatch match;
    std::istringstream wide_token_file(indexed_bytes);
    check(conversation_file_match(wide_token_file, id, indexed_bound, {INT64_MAX, 2, 3}, {}, false, match, error) &&
          match.tokens == 0, "64-bit request tokens are compared without narrowing");
    check(!conversation_file_match(foreign_file, foreign, indexed_bound, {1, 2, 3}, {}, false, match, error),
          "foreign identity is declined during selection");
    std::istringstream over_budget(indexed_bytes);
    check(!conversation_file_match(over_budget, id, indexed_bound - 1, {1, 2, 3}, {}, false, match, error),
          "over-budget candidate is declined during selection");

    TempDirectory temp;
    auto asset = temp.path / "weights";
    auto moved = temp.path / "relocated-weights";
    std::string content(200000, 'a');
    write(asset, content);
    write(moved, content);
    ConversationIdentity initial, other;
    check(conversation_identity({{"weights", asset}}, {{"settings", "kv=int8"}}, initial, error), "hash whole asset");
    check(conversation_identity({{"weights", moved}}, {{"settings", "kv=int8"}}, other, error) && other == initial,
          "relocating unchanged assets preserves identity");
    ConversationIdentity repeated, copied;
    check(conversation_identity({{"weights", asset}, {"embedding", asset}}, {{"settings", "kv=int8"}}, repeated, error) &&
          conversation_identity({{"weights", asset}, {"embedding", moved}}, {{"settings", "kv=int8"}}, copied, error) && repeated == copied,
          "deduplicated reads retain each role and match independently copied assets");
    content[100000] = 'b'; write(moved, content);
    check(conversation_identity({{"weights", moved}}, {{"settings", "kv=int8"}}, other, error) && other != initial,
          "middle content change with same size and ends invalidates identity");
    check(conversation_identity({{"tokenizer", asset}}, {{"settings", "kv=int8"}}, other, error) && other != initial,
          "asset role participates in identity");
    check(conversation_identity({{"weights", asset}}, {{"settings", "kv=k8v4"}}, other, error) && other != initial,
          "runtime settings participate in identity");
    check(conversation_identity({{"weights", asset}, {"tokenizer", moved}}, {{"settings", "kv=int8"}}, other, error) && other != initial,
          "additional assets participate in identity");
    other = initial;
    check(!conversation_identity({{"weights", temp.path / "missing"}}, {{"settings", "kv=int8"}}, other, error), "missing asset fails closed");
    check(other == initial, "failed hash preserves caller identity");
    check(conversation_identity({}, {{"settings", "fixture-settings"}}, other, error), "fixed identity fixture");
    // Independently computed with Python hashlib and struct.pack('<Q', length).
    check(hex(other) == "f163e269d671c03aadd2dd4206b143ca682709d3e2a16a407bae95a95e2b70b1", "identity uses SHA-256 with framed fields");
    // Named identity mismatches are rejected before any state payload is read.
    const ConversationSettings settings{{"engine-version", "0.1.30"}, {"geometry", "48"},
        {"expert-quant/0", "IQ3_S"}, {"kv-format", "int8"}, {"kv-resident", "32768"},
        {"split", "single"}, {"prefill-chunk", "8192"}, {"spec", "1"},
        {"steering", "none"}, {"rope-factor", "1"}};
    ConversationIdentity named;
    check(conversation_identity({{"weights", asset}}, settings, named, error), "named layout identity");
    const auto named_bytes = encode(indexed, named);
    check(conversation_file_size(indexed, named, measured, error) && measured == named_bytes.size(),
          "identity directory included in disk quota reservation");
    std::istringstream named_input(named_bytes);
    check(conversation_file_read(named_input, named, indexed_bound, indexed_bound, 0, decoded, error) && same(decoded, indexed),
          "named identity and layer range survive round trip");
    for (size_t i = 0; i < settings.size(); ++i) {
        auto changed = settings; changed[i].second += "changed";
        ConversationIdentity expected;
        check(conversation_identity({{"weights", asset}}, changed, expected, error), "changed layout fixture");
        const auto reason = "snapshot identity differs: runtime/" + settings[i].first;
        CountingBuffer buffer(named_bytes); std::istream read(&buffer);
        const auto sentinel = decoded;
        check(!conversation_file_read(read, expected, indexed_bound, indexed_bound, 0, decoded, error) && error == reason,
              "read reports first differing layout field");
        check(same(decoded, sentinel) && buffer.read_bytes < 2048, "identity refusal reads no state and preserves output");
        std::istringstream probe(named_bytes);
        check(!conversation_file_match(probe, expected, indexed_bound, {1, 2, 3}, {}, false, match, error) && error == reason,
              "prefix probe reports first differing layout field");
    }
    write(asset, "different weights");
    check(conversation_identity({{"weights", asset}}, settings, other, error), "changed weight content");
    std::istringstream weight_input(named_bytes);
    check(!conversation_file_read(weight_input, other, indexed_bound, indexed_bound, 0, decoded, error) &&
          error == "snapshot identity differs: asset/weights", "asset rejection names its role");
    auto excessive = named_bytes; set_integer(excessive, 40, UINT64_MAX); rejected(excessive, named);
    auto long_name = named_bytes; set_integer(long_name, 48, UINT64_MAX); rejected(long_name, named);
    auto legacy = named_bytes; legacy[7] = 1; rejected(legacy, named);
    check(!conversation_identity({}, {{"kv", "int8"}, {"kv", "k8v4"}}, other, error), "duplicate identity names rejected");
    check(!conversation_identity({}, {{"bad\nname", "value"}}, other, error), "identity names cannot inject log lines");
    ConversationSettings many;
    for (unsigned i = 0; i < 257; ++i) many.emplace_back(std::to_string(i), "x");
    check(!conversation_identity({}, many, other, error), "identity metadata is bounded");
    std::printf("conversation_file_test: %d checks passed\n", checks);
}
