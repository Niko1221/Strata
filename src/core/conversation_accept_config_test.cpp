// CPU-only tests for the opt-in accepted configuration: SessionReadLimits::accept_config (base files) and the
// matching argument of session_delta_apply.  A file saved under another engine CONFIGURATION is refused by default
// and read when exactly that config is accepted; the model fingerprint must always match.
// Built with -DSTRATA_BUILD_CONVERSATION_TESTS=ON; no CUDA, no model.
#include "strata/core/conversation_file.hpp"

#include <array>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <string>
#include <vector>

using namespace strata::core;
namespace fs = std::filesystem;

namespace {
int checks = 0;
void check(bool ok, const char* label) {
    ++checks;
    if (!ok) { std::fprintf(stderr, "FAIL: %s\n", label); std::exit(1); }
}

ConversationBuffer pattern(size_t n, uint8_t seed) {
    ConversationBuffer b;
    b.resize(n);
    b.visit(0, n, [&](uint8_t* p, size_t c, size_t at) {
        for (size_t i = 0; i < c; ++i) p[i] = uint8_t((at + i) * 131u + seed);
        return true;
    });
    return b;
}

ConversationCheckpoint checkpoint(size_t tokens, uint8_t seed) {
    ConversationCheckpoint c;
    for (size_t i = 0; i < tokens; ++i) c.ids.push_back(int32_t(1000 + i));
    c.imgs = {{3, 0x1234567890abcdefull ^ seed}};
    c.gdn.assign(4099, uint8_t(seed * 3u + 1u));
    c.ple.assign(77, uint8_t(seed * 5u + 2u));
    c.tails.assign(301, uint8_t(seed * 7u + 3u));
    c.dead.assign(64, uint8_t(seed * 11u + 4u));
    c.block_pos.assign(8, uint8_t(seed * 13u + 5u));
    c.used = 42 + seed;
    return c;
}

SavedConversation sample() {
    SavedConversation s;
    for (size_t i = 0; i < s.geometry.size(); ++i) s.geometry[i] = int64_t(100 + i);
    s.layer_lo = 0;
    s.layer_hi = 48;
    s.cvec = false;
    s.live = checkpoint(7, 1);
    s.checkpoints.push_back(checkpoint(3, 2));
    for (int layer = 0; layer < 2; ++layer) {
        ConversationKv kv;
        kv.format = 2 + layer;
        kv.cells = 1024;
        kv.heads = 2;
        kv.head_dim = 256;
        kv.page_size = 64;
        kv.pooled_rows = 9;
        kv.idx_dim = 128;
        kv.k = pattern(4096, uint8_t(layer));
        kv.v = pattern(4097 + layer, uint8_t(layer + 10));
        kv.k_scale = pattern(512, uint8_t(layer + 20));
        kv.v_scale = pattern(512, uint8_t(layer + 30));
        kv.pooled = pattern(9 * 128 * 4, uint8_t(layer + 40));
        s.kv.push_back(std::move(kv));
    }
    return s;
}

std::vector<SessionKvSource> delta_sources(const SavedConversation& s) {
    std::vector<SessionKvSource> out;
    for (const auto& k : s.kv) {
        SessionKvSource q;
        q.format = k.format;
        q.cells = k.cells;
        q.heads = k.heads;
        q.head_dim = k.head_dim;
        q.page_size = k.page_size;
        q.pooled_rows = k.pooled_rows;
        q.idx_dim = k.idx_dim;
        const std::array<const ConversationBuffer*, 5> parts = {&k.k, &k.v, &k.k_scale, &k.v_scale, &k.pooled};
        for (size_t i = 0; i < 5; ++i) q.sizes[i] = parts[i]->size();
        q.read = [parts](size_t part, size_t offset, void* dst, size_t n) {
            return parts[part]->read(dst, offset, n);
        };
        out.push_back(std::move(q));
    }
    return out;
}
} // namespace

int main() {
    const fs::path dir = fs::temp_directory_path() /
        ("strata-accept-config-test-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    fs::remove_all(dir);
    fs::create_directories(dir);
    const SessionFileIdentity saved{0x1111222233334444ull, 0x5555666677778888ull};
    const SessionFileIdentity other_config{saved.model, saved.config ^ 0x0f0f0f0f0f0f0f0full};
    const SessionFileIdentity other_model{saved.model ^ 0x10, saved.config};
    const SavedConversation original = sample();
    const fs::path base = dir / "base.sess";
    std::string error;
    size_t written = 0;
    check(session_file_write(base.string(), original, saved, written, error), "base: write");

    auto read_with = [&](const SessionFileIdentity& id, uint64_t accept, SavedConversation& out, std::string& why) {
        SessionReadLimits limits;
        limits.accept_config = accept;
        size_t bytes = 0;
        return session_file_read(base.string(), id, out, bytes, why, limits);
    };
    SavedConversation got;
    check(!read_with(other_config, 0, got, error) &&
              error.find("another engine configuration") != std::string::npos,
          "base: the default refuses another configuration");
    check(!read_with(other_model, saved.config, got, error) && error.find("another model") != std::string::npos,
          "base: an accepted config never accepts another model");
    check(!read_with(other_config, saved.config ^ 1ull, got, error),
          "base: an unrelated accepted config is refused");
    check(read_with(other_config, saved.config, got, error) && got.live.ids == original.live.ids &&
              got.live.imgs == original.live.imgs,
          "base: the accepted config reads the file");
    check(read_with(saved, saved.config, got, error), "base: accepting its own config changes nothing");

    // delta record and its apply pass
    const fs::path log = dir / "chain.dlog";
    SavedConversation grown = original;
    grown.live.ids.push_back(9001);
    SavedConversation meta = grown;
    meta.kv.clear();
    const std::vector<SessionKvSource> sources = delta_sources(grown);
    uint64_t wrote = 0;
    check(session_delta_write(log.string(), 0, meta, sources, (int64_t) grown.live.ids.size(), saved, wrote, error),
          "delta: write");
    const SessionReadLimits none;
    SessionDeltaRecord rec;
    uint64_t next = 0;
    check(!session_delta_read(log.string(), 0, wrote, other_config, none, rec, next, error) &&
              error.find("another model or engine configuration") != std::string::npos,
          "delta: the default refuses another configuration");
    check(!session_delta_read(log.string(), 0, wrote, other_model, none, rec, next, error),
          "delta: another model is refused");
    SessionReadLimits accepted;
    accepted.accept_config = saved.config;
    check(session_delta_read(log.string(), 0, wrote, other_config, accepted, rec, next, error) && next == wrote &&
              rec.meta.live.ids == grown.live.ids,
          "delta: the accepted config reads the record");
    auto sink = [](size_t, size_t, uint64_t, const void*, size_t) { return true; };
    check(!session_delta_apply(log.string(), 0, other_config, rec, sink, error),
          "delta apply: the default refuses another configuration");
    check(session_delta_apply(log.string(), 0, other_config, rec, sink, error, nullptr, saved.config),
          "delta apply: the accepted config applies the record");
    check(!session_delta_apply(log.string(), 0, other_model, rec, sink, error, nullptr, saved.config),
          "delta apply: another model is refused");
    check(session_delta_apply(log.string(), 0, saved, rec, sink, error),
          "delta apply: accepting its own config changes nothing");

    fs::remove_all(dir);
    std::printf("conversation_accept_config_test: %d checks passed\n", checks);
    return 0;
}
