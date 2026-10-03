// src/core/conversation_disk_test.cpp - the L3 disk tier, CPU only. No CUDA, no model.
//
// The test builds SavedConversation images by hand, writes them to a temporary directory, and
// checks the round trip, the exact prefix lookup, the atomic publication, the startup recovery,
// the LRU eviction, and every corruption path.
#include "strata/core/conversation_disk.hpp"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <string>
#include <vector>

using namespace strata::core;

namespace {

int checks = 0;
void check(bool value, const char* description) {
    ++checks;
    if (!value) {
        std::fprintf(stderr, "FAIL: %s\n", description);
        std::exit(1);
    }
}

ConversationBuffer make_buffer(size_t bytes, uint8_t value) {
    ConversationBuffer buffer;
    buffer.resize(bytes);
    buffer.visit(0, bytes, [&](uint8_t* data, size_t chunk, size_t) {
        std::memset(data, value, chunk);
        return true;
    });
    return buffer;
}

ConversationDiskRecord make_record(std::initializer_list<int32_t> ids, uint8_t fill, size_t kv_bytes,
                                   bool cvec = true, std::initializer_list<ConversationImageKey> images = {}) {
    ConversationDiskRecord record;
    SavedConversation stage;
    stage.geometry[0] = 8;
    stage.geometry[1] = 1;
    stage.geometry[2] = 64;
    stage.layer_lo = 0;
    stage.layer_hi = 4;
    stage.cvec = cvec;
    stage.live.ids.assign(ids);
    stage.live.imgs.assign(images);
    stage.live.gdn.assign(8, fill);
    stage.live.ple.assign(5, static_cast<uint8_t>(fill + 1));
    stage.live.tails.assign(3, static_cast<uint8_t>(fill + 2));
    stage.live.dead.assign(2, static_cast<uint8_t>(fill + 3));
    stage.live.block_pos.assign(4, static_cast<uint8_t>(fill + 4));
    ConversationKv kv;
    kv.format = 0;
    kv.cells = 4;
    kv.heads = 1;
    kv.head_dim = 64;
    kv.page_size = 4;
    kv.pooled_rows = 2;
    kv.idx_dim = 8;
    kv.k = make_buffer(kv_bytes, fill);
    kv.v = make_buffer(kv_bytes + 1, static_cast<uint8_t>(fill + 1));
    kv.k_scale = make_buffer(16, static_cast<uint8_t>(fill + 2));
    kv.v_scale = make_buffer(16, static_cast<uint8_t>(fill + 3));
    kv.pooled = make_buffer(64, static_cast<uint8_t>(fill + 4));
    stage.kv.push_back(std::move(kv));
    record.stages.push_back(std::move(stage));
    return record;
}

bool same_checkpoint(const ConversationCheckpoint& a, const ConversationCheckpoint& b) {
    return a.ids == b.ids && a.imgs == b.imgs && a.gdn == b.gdn && a.ple == b.ple && a.tails == b.tails &&
           a.dead == b.dead && a.block_pos == b.block_pos && a.used == b.used &&
           a.stage_parts.size() == b.stage_parts.size();
}

bool same_kv(const ConversationKv& a, const ConversationKv& b) {
    return a.format == b.format && a.cells == b.cells && a.heads == b.heads && a.head_dim == b.head_dim &&
           a.page_size == b.page_size && a.pooled_rows == b.pooled_rows && a.idx_dim == b.idx_dim && a.k == b.k &&
           a.v == b.v && a.k_scale == b.k_scale && a.v_scale == b.v_scale && a.pooled == b.pooled;
}

bool same_stage(const SavedConversation& a, const SavedConversation& b) {
    if (a.geometry != b.geometry || a.layer_lo != b.layer_lo || a.layer_hi != b.layer_hi || a.cvec != b.cvec)
        return false;
    if (!same_checkpoint(a.live, b.live) || a.checkpoints.size() != b.checkpoints.size() ||
        a.kv.size() != b.kv.size())
        return false;
    for (size_t i = 0; i < a.checkpoints.size(); ++i)
        if (!same_checkpoint(a.checkpoints[i], b.checkpoints[i])) return false;
    for (size_t i = 0; i < a.kv.size(); ++i)
        if (!same_kv(a.kv[i], b.kv[i])) return false;
    return true;
}

bool same_record(const ConversationDiskRecord& a, const ConversationDiskRecord& b) {
    if (a.stages.size() != b.stages.size()) return false;
    for (size_t i = 0; i < a.stages.size(); ++i)
        if (!same_stage(a.stages[i], b.stages[i])) return false;
    return true;
}

ConversationDiskOptions options_for(const std::filesystem::path& directory,
                                    const ConversationDiskIdentity& identity, uint64_t budget) {
    ConversationDiskOptions options;
    options.directory = directory;
    options.identity = identity;
    options.budget_bytes = budget;
    return options;
}

void patch_byte(const std::filesystem::path& path, uint64_t offset, uint8_t value) {
    std::FILE* file = std::fopen(path.string().c_str(), "r+b");
    check(file != nullptr, "a patched record opens");
    check(std::fseek(file, static_cast<long>(offset), SEEK_SET) == 0, "a patched record seeks");
    check(std::fwrite(&value, 1, 1, file) == 1, "a patched record writes");
    std::fclose(file);
}

std::vector<std::filesystem::path> temporary_files(const std::filesystem::path& directory) {
    std::vector<std::filesystem::path> found;
    for (const auto& entry : std::filesystem::directory_iterator(directory)) {
        const std::string name = entry.path().filename().string();
        if (name.size() >= 4 && name.compare(name.size() - 4, 4, ".tmp") == 0) found.push_back(entry.path());
    }
    return found;
}

void test_checksum_and_identity() {
    check(conversation_disk_checksum(nullptr, 0) == 14695981039346656037ull,
          "FNV-1a 64 of an empty input");
    const uint8_t a = 'a';
    check(conversation_disk_checksum(&a, 1) == 0xaf63dc4c8601ec8cull, "FNV-1a 64 of \"a\"");
    const uint8_t abc[3] = {'a', 'b', 'c'};
    check(conversation_disk_checksum(abc, 3) == 0xe71fa2190541574bull, "FNV-1a 64 of \"abc\"");
    const ConversationDiskIdentity first = ConversationDiskIdentity::from_string("qwen3-v100");
    const ConversationDiskIdentity again = ConversationDiskIdentity::from_string("qwen3-v100");
    const ConversationDiskIdentity other = ConversationDiskIdentity::from_string("qwen3-v100-rope");
    check(first == again, "the identity from one text key is stable");
    check(!(first == other), "two text keys give two identities");
    bool nonzero = false;
    for (const uint8_t byte : first.bytes) nonzero = nonzero || byte != 0;
    check(nonzero, "the identity is not all zero");
}

void test_name() {
    const std::vector<int32_t> ids = {1, 2, 3};
    const std::vector<ConversationImageKey> images = {{0, 7}};
    const std::string name = conversation_disk_name(ids, images, true);
    check(name == conversation_disk_name(ids, images, true), "the name is stable");
    check(name.find('/') == std::string::npos && name.find('\\') == std::string::npos,
          "the name has no separator");
    check(name.find("..") == std::string::npos, "the name has no parent reference");
    check(name != conversation_disk_name(ids, images, false), "the cvec flag changes the name");
    check(name != conversation_disk_name(ids, {{0, 8}}, true), "the image key changes the name");
    check(name != conversation_disk_name({1, 2}, images, true), "the token prefix changes the name");
}

void test_round_trip(const std::filesystem::path& directory) {
    const ConversationDiskIdentity identity = ConversationDiskIdentity::from_string("round-trip");
    ConversationDiskStore store;
    std::string error;
    check(store.open(options_for(directory, identity, 1ull << 30), error) == ConversationDiskStatus::ok,
          "the store opens");
    check(store.enabled(), "the store is enabled");
    // One buffer larger than a segment proves the segmented streaming path.
    const size_t kv_bytes = ConversationBuffer::segment_bytes + 17;
    ConversationDiskRecord record = make_record({1, 2, 3, 4}, 0x10, kv_bytes);
    const ConversationDiskRecord original = record;
    check(store.put(std::move(record), error) == ConversationDiskStatus::ok, "the record lands");
    check(store.records() == 1, "the store holds one record");
    check(store.bytes() > 0, "the store accounts for the record bytes");
    const std::string name = conversation_disk_name({1, 2, 3, 4}, {}, true);
    check(std::filesystem::exists(directory / (name + ConversationDiskStore::file_suffix)),
          "the record file exists");
    check(temporary_files(directory).empty(), "no temporary file remains");

    ConversationDiskMatch match;
    check(store.best({1, 2, 3, 4, 5}, {}, true, match), "the prefix matches");
    check(match.tokens == 4 && match.name == name, "the match has the deepest prefix and the name");
    check(!store.best({1, 2, 3, 4}, {}, true, match), "an equal-length prompt is not a match");
    check(!store.best({1, 2, 9}, {}, true, match), "a different token is not a match");
    check(!store.best({1, 2, 3, 4, 5}, {}, false, match), "a different cvec flag is not a match");

    ConversationDiskRecord loaded;
    check(store.get(name, loaded, error) == ConversationDiskStatus::ok, "the record loads");
    check(same_record(loaded, original), "the loaded record equals the stored one");
    check(loaded.stages.front().kv.front().k.size() == kv_bytes, "the segmented buffer keeps its size");
    check(store.get(conversation_disk_name({9, 9, 9}, {}, true), loaded, error) ==
              ConversationDiskStatus::miss,
          "an unknown name is a miss");
    check(store.remove(name, error) == ConversationDiskStatus::ok, "the record removes");
    check(store.records() == 0 && store.bytes() == 0, "the index forgets the removed record");
    check(store.remove(name, error) == ConversationDiskStatus::miss, "a second remove is a miss");
    check(!store.best({1, 2, 3, 4, 5}, {}, true, match), "a removed record does not match");
}

void test_exact_images(const std::filesystem::path& directory) {
    const ConversationDiskIdentity identity = ConversationDiskIdentity::from_string("images");
    ConversationDiskStore store;
    std::string error;
    check(store.open(options_for(directory, identity, 1ull << 30), error) == ConversationDiskStatus::ok,
          "the image store opens");
    ConversationDiskRecord record = make_record({5, 6, 7}, 0x20, 256, true, {{0, 0x1234}});
    check(store.put(std::move(record), error) == ConversationDiskStatus::ok, "the image record lands");
    ConversationDiskMatch match;
    check(store.best({5, 6, 7, 8}, {{0, 0x1234}}, true, match), "the same image key matches");
    check(match.tokens == 3, "the image match has the deepest prefix");
    check(!store.best({5, 6, 7, 8}, {{0, 0x1235}}, true, match), "a different image key does not match");
    check(!store.best({5, 6, 7, 8}, {}, true, match), "a missing image key does not match");
}

void test_multi_stage(const std::filesystem::path& directory) {
    const ConversationDiskIdentity identity = ConversationDiskIdentity::from_string("split");
    ConversationDiskStore store;
    std::string error;
    check(store.open(options_for(directory, identity, 1ull << 30), error) == ConversationDiskStatus::ok,
          "the split store opens");
    ConversationDiskRecord record = make_record({2, 4, 6, 8}, 0x30, 512);
    SavedConversation second = record.stages.front();
    second.layer_lo = 4;
    second.layer_hi = 8;
    second.kv.front().k = make_buffer(513, 0xAB);
    record.stages.push_back(std::move(second));
    const ConversationDiskRecord original = record;
    check(store.put(std::move(record), error) == ConversationDiskStatus::ok, "the two-stage record lands");
    ConversationDiskMatch match;
    check(store.best({2, 4, 6, 8, 10}, {}, true, match) && match.tokens == 4,
          "the two-stage record matches on its shared chain");
    ConversationDiskRecord loaded;
    check(store.get(match.name, loaded, error) == ConversationDiskStatus::ok, "the two-stage record loads");
    check(loaded.stages.size() == 2, "both stages return");
    check(loaded.stages[1].layer_lo == 4 && loaded.stages[1].layer_hi == 8, "the second carve returns");
    check(same_record(loaded, original), "both stages equal the stored images");
}

void test_recovery(const std::filesystem::path& directory) {
    const ConversationDiskIdentity identity = ConversationDiskIdentity::from_string("recovery");
    const std::string name = conversation_disk_name({3, 1, 4, 1}, {}, true);
    const ConversationDiskRecord original = make_record({3, 1, 4, 1}, 0x40, 128);
    {
        ConversationDiskStore store;
        std::string error;
        check(store.open(options_for(directory, identity, 1ull << 30), error) == ConversationDiskStatus::ok,
              "the first store opens");
        ConversationDiskRecord record = original;
        check(store.put(std::move(record), error) == ConversationDiskStatus::ok, "the record lands");
    }
    ConversationDiskStore reopened;
    std::string error;
    check(reopened.open(options_for(directory, identity, 1ull << 30), error) == ConversationDiskStatus::ok,
          "the store reopens");
    check(reopened.records() == 1 && reopened.recoveries() == 1, "the index recovers the record");
    ConversationDiskMatch match;
    check(reopened.best({3, 1, 4, 1, 5}, {}, true, match) && match.name == name,
          "the recovered record matches");
    ConversationDiskRecord loaded;
    check(reopened.get(name, loaded, error) == ConversationDiskStatus::ok, "the recovered record loads");
    check(same_record(loaded, original), "the recovered record equals the stored one");
}

void test_eviction(const std::filesystem::path& directory) {
    const ConversationDiskIdentity identity = ConversationDiskIdentity::from_string("eviction");
    uint64_t one = 0;
    {
        ConversationDiskStore probe;
        std::string error;
        check(probe.open(options_for(directory, identity, 1ull << 30), error) == ConversationDiskStatus::ok,
              "the probe store opens");
        ConversationDiskRecord record = make_record({1, 1, 1, 1}, 0x50, 128);
        check(probe.put(std::move(record), error) == ConversationDiskStatus::ok, "the probe record lands");
        one = probe.bytes();
        check(one > 0, "the probe record has a size");
        check(probe.remove(conversation_disk_name({1, 1, 1, 1}, {}, true), error) ==
                  ConversationDiskStatus::ok,
              "the probe record removes");
    }
    ConversationDiskStore store;
    std::string error;
    check(store.open(options_for(directory, identity, 2 * one), error) == ConversationDiskStatus::ok,
          "the small store opens");
    ConversationDiskRecord first = make_record({1, 1, 1, 1}, 0x50, 128);
    ConversationDiskRecord second = make_record({1, 1, 1, 2}, 0x50, 128);
    ConversationDiskRecord third = make_record({1, 1, 1, 3}, 0x50, 128);
    check(store.put(std::move(first), error) == ConversationDiskStatus::ok, "the first record lands");
    check(store.put(std::move(second), error) == ConversationDiskStatus::ok, "the second record lands");
    const std::string first_name = conversation_disk_name({1, 1, 1, 1}, {}, true);
    const std::string second_name = conversation_disk_name({1, 1, 1, 2}, {}, true);
    const std::string third_name = conversation_disk_name({1, 1, 1, 3}, {}, true);
    ConversationDiskRecord loaded;
    check(store.get(first_name, loaded, error) == ConversationDiskStatus::ok, "the first record loads");
    check(store.put(std::move(third), error) == ConversationDiskStatus::ok, "the third record lands");
    check(store.bytes() <= 2 * one, "the store stays inside the budget");
    check(store.records() == 2 && store.evictions() == 1, "one record was evicted");
    ConversationDiskMatch match;
    check(store.best({1, 1, 1, 1, 9}, {}, true, match), "the used record survives");
    check(store.best({1, 1, 1, 3, 9}, {}, true, match), "the newest record survives");
    check(!store.best({1, 1, 1, 2, 9}, {}, true, match), "the least recently used record is gone");
    check(!std::filesystem::exists(directory / (second_name + ConversationDiskStore::file_suffix)),
          "the evicted file is gone");
}

void test_caps_and_limits(const std::filesystem::path& directory) {
    const ConversationDiskIdentity identity = ConversationDiskIdentity::from_string("caps");
    ConversationDiskStore store;
    std::string error;
    ConversationDiskOptions options = options_for(directory, identity, 1ull << 30);
    options.max_records = 1;
    check(store.open(options, error) == ConversationDiskStatus::ok, "the capped store opens");
    ConversationDiskRecord first = make_record({7, 7, 7, 1}, 0x60, 128);
    ConversationDiskRecord second = make_record({7, 7, 7, 2}, 0x60, 128);
    check(store.put(std::move(first), error) == ConversationDiskStatus::ok, "the first capped record lands");
    check(store.put(std::move(second), error) == ConversationDiskStatus::ok, "the second capped record lands");
    check(store.records() == 1, "the record cap holds one record");
    ConversationDiskMatch match;
    check(!store.best({7, 7, 7, 1, 9}, {}, true, match), "the capped-out record is gone");
    check(store.best({7, 7, 7, 2, 9}, {}, true, match), "the newest record stays");
}

void test_corruption(const std::filesystem::path& directory) {
    const ConversationDiskIdentity identity = ConversationDiskIdentity::from_string("corruption");
    const std::string name = conversation_disk_name({8, 8, 8, 8}, {}, true);
    const std::filesystem::path file = directory / (name + ConversationDiskStore::file_suffix);
    auto make = [&]() {
        ConversationDiskStore store;
        std::string error;
        check(store.open(options_for(directory, identity, 1ull << 30), error) == ConversationDiskStatus::ok,
              "the corruption store opens");
        ConversationDiskRecord record = make_record({8, 8, 8, 8}, 0x70, 128);
        check(store.put(std::move(record), error) == ConversationDiskStatus::ok, "the corruption record lands");
        return store;
    };
    // A flipped payload byte.
    {
        ConversationDiskStore store = make();
        std::string error;
        patch_byte(file, std::filesystem::file_size(file) - 1, 0xFF);
        ConversationDiskRecord loaded;
        check(store.get(name, loaded, error) == ConversationDiskStatus::corrupt, "a bad payload byte is corrupt");
        check(store.records() == 0 && store.corruptions() == 1, "the store drops the corrupt record");
        check(!std::filesystem::exists(file), "the corrupt file is removed");
    }
    // A flipped metadata byte.
    {
        ConversationDiskStore store = make();
        std::string error;
        patch_byte(file, 96 + 1, 0xFF);
        ConversationDiskRecord loaded;
        check(store.get(name, loaded, error) == ConversationDiskStatus::corrupt, "a bad metadata byte is corrupt");
        check(!std::filesystem::exists(file), "the corrupt file is removed");
    }
    // A truncated file.
    {
        ConversationDiskStore store = make();
        std::string error;
        std::filesystem::resize_file(file, std::filesystem::file_size(file) - 1);
        ConversationDiskRecord loaded;
        check(store.get(name, loaded, error) == ConversationDiskStatus::corrupt, "a truncated record is corrupt");
    }
    // A bad version.
    {
        ConversationDiskStore store = make();
        std::string error;
        patch_byte(file, 8, 99);
        ConversationDiskStore reopened;
        check(reopened.open(options_for(directory, identity, 1ull << 30), error) == ConversationDiskStatus::ok,
              "the store reopens after a version change");
        check(reopened.records() == 0 && reopened.corruptions() == 1, "the bad version is removed on open");
    }
    // A different identity.
    {
        ConversationDiskStore store = make();
        std::string error;
        ConversationDiskStore reopened;
        check(reopened.open(options_for(directory, ConversationDiskIdentity::from_string("other"), 1ull << 30),
                            error) == ConversationDiskStatus::ok,
              "the store reopens under a new identity");
        check(reopened.records() == 0 && reopened.corruptions() == 1,
              "the foreign identity is removed on open");
    }
}

void test_disabled_and_invalid(const std::filesystem::path& directory) {
    ConversationDiskStore store;
    std::string error;
    check(store.open(options_for(directory, ConversationDiskIdentity::from_string("off"), 0), error) ==
              ConversationDiskStatus::disabled,
          "a zero budget disables the store");
    check(!store.enabled(), "the disabled store reports disabled");
    ConversationDiskRecord record = make_record({1, 2, 3, 4}, 0x80, 128);
    check(store.put(std::move(record), error) == ConversationDiskStatus::disabled,
          "a disabled store refuses a put");
    ConversationDiskRecord loaded;
    check(store.get("c0000000000000000-0000000000000004", loaded, error) == ConversationDiskStatus::disabled,
          "a disabled store refuses a get");
    ConversationDiskMatch match;
    check(!store.best({1, 2, 3, 4, 5}, {}, true, match), "a disabled store matches nothing");

    const ConversationDiskIdentity identity = ConversationDiskIdentity::from_string("invalid");
    ConversationDiskStore live;
    check(live.open(options_for(directory, identity, 1ull << 30), error) == ConversationDiskStatus::ok,
          "the invalid-input store opens");
    ConversationDiskRecord empty;
    check(live.put(std::move(empty), error) == ConversationDiskStatus::invalid, "an empty record is invalid");
    ConversationDiskRecord no_tokens = make_record({}, 0x90, 128);
    check(live.put(std::move(no_tokens), error) == ConversationDiskStatus::invalid,
          "a record without a live prefix is invalid");
    ConversationDiskRecord small = make_record({4, 4, 4, 4}, 0x90, 128);
    ConversationDiskOptions tight = options_for(directory, identity, 64);
    ConversationDiskStore cramped;
    check(cramped.open(tight, error) == ConversationDiskStatus::ok, "the tight store opens");
    check(cramped.put(std::move(small), error) == ConversationDiskStatus::invalid,
          "a record over the budget is invalid");
    check(cramped.get("../escape", loaded, error) == ConversationDiskStatus::invalid,
          "an unsafe name is invalid");
    check(cramped.remove("..", error) == ConversationDiskStatus::invalid, "an unsafe remove is invalid");
}

} // namespace

int main() {
    const std::filesystem::path root = std::filesystem::temp_directory_path() / "strata_conversation_disk_test";
    std::error_code error_code;
    std::filesystem::remove_all(root, error_code);
    std::filesystem::create_directories(root, error_code);

    test_checksum_and_identity();
    test_name();
    test_round_trip(root / "round_trip");
    test_exact_images(root / "images");
    test_multi_stage(root / "split");
    test_recovery(root / "recovery");
    test_eviction(root / "eviction");
    test_caps_and_limits(root / "caps");
    test_corruption(root / "corruption");
    test_disabled_and_invalid(root / "disabled");

    std::filesystem::remove_all(root, error_code);
    std::printf("conversation_disk_test: %d checks passed\n", checks);
    return 0;
}
