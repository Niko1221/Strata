#define _CRT_SECURE_NO_WARNINGS
#include "strata/qwen35/expert_store.hpp"
#include "strata/qwen35/bounded_experts.hpp"
#include "strata/qwen35/phase_trace.hpp"

#include <cstdint>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <string>
#include <thread>
#include <vector>

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <unistd.h>
#endif

namespace fs = std::filesystem;
using strata::qwen35::ExpertBlobLayout;
using strata::qwen35::ExpertSource;
using strata::qwen35::PackBuildRequest;
using strata::qwen35::PackLayer;
using strata::qwen35::expert_pack_last_scratch_bytes;
using strata::qwen35::expert_pack_sha256_file;
using strata::qwen35::expert_pack_verify;
using strata::qwen35::kExpertPackMagic;
using strata::qwen35::kExpertPackPayloadOffsetField;
using strata::qwen35::kExpertPackPrefixBytes;
using strata::qwen35::kExpertPackRecordBytes;
using strata::qwen35::kExpertPackVersionField;
using strata::qwen35::kQ36ExpertTail;
using strata::qwen35::open_or_build_expert_pack;
using strata::qwen35::pack_expert_blob;

namespace {

int g_fails = 0;

void check(bool ok, const char* text, int line) {
    if (ok) return;
    std::fprintf(stderr, "FAIL %s:%d %s\n", "q36_expert_store_test.cpp", line, text);
    ++g_fails;
}

#define CHECK(cond) check((cond), #cond, __LINE__)

std::string temp_dir() {
#if defined(_WIN32)
    const char* root = std::getenv("TEMP");
    const unsigned long id = GetCurrentProcessId();
#else
    const char* root = std::getenv("TMPDIR");
    if (!root) root = "/tmp";
    const unsigned long id = (unsigned long) ::getpid();
#endif
    fs::path dir = fs::path(root ? root : ".") / ("q36epk-" + std::to_string(id));
    fs::create_directories(dir);
    return dir.string();
}

void write_bytes(const fs::path& path, const void* data, size_t n) {
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    out.write(static_cast<const char*>(data), (std::streamsize) n);
}

PackLayer make_layer(const std::vector<uint8_t>& gate, const std::vector<uint8_t>& up, const std::vector<uint8_t>& down,
                     uint32_t n_expert, uint64_t src_gsz, uint64_t src_dsz) {
    PackLayer layer;
    layer.layout.gu_type = 12;
    layer.layout.d_type = 8;
    layer.layout.n_embd = 32;
    layer.layout.n_ff = 16;
    layer.layout.n_expert = n_expert;
    layer.layout.gu_row = 1;
    layer.layout.d_row = 1;
    layer.layout.up_off = 16;
    layer.layout.down_off = 32;
    layer.layout.blob_bytes = 48;
    layer.layout.stride = 64;
    layer.source.gate = gate.data();
    layer.source.up = up.data();
    layer.source.down = down.data();
    layer.source.src_gsz = src_gsz;
    layer.source.src_dsz = src_dsz;
    return layer;
}

std::vector<uint8_t> expected_layer(const PackLayer& layer) {
    const uint64_t bytes = layer.layout.stride * layer.layout.n_expert + kQ36ExpertTail;
    std::vector<uint8_t> out((size_t) bytes, 0x5a);
    std::memset(out.data(), 0, out.size());
    for (uint32_t e = 0; e < layer.layout.n_expert; ++e) {
        uint8_t* dst = out.data() + (size_t) e * (size_t) layer.layout.stride;
        std::memcpy(dst, layer.source.gate + (size_t) e * (size_t) layer.source.src_gsz, (size_t) layer.source.src_gsz);
        std::memcpy(dst + layer.layout.up_off, layer.source.up + (size_t) e * (size_t) layer.source.src_gsz,
                    (size_t) layer.source.src_gsz);
        std::memcpy(dst + layer.layout.down_off, layer.source.down + (size_t) e * (size_t) layer.source.src_dsz,
                    (size_t) layer.source.src_dsz);
    }
    return out;
}

void fill_pattern(std::vector<uint8_t>& buf, uint8_t mark) {
    for (size_t i = 0; i < buf.size(); ++i) buf[i] = (uint8_t) (mark + i);
}

uint32_t reference_crc32(const uint8_t* data, size_t size) {
    // Independent bitwise oracle for the pack format, not the production table.
    uint32_t crc = 0xffffffffu;
    for (size_t i = 0; i < size; ++i) {
        crc ^= data[i];
        for (unsigned bit = 0; bit < 8; ++bit) crc = (crc >> 1) ^ (0xedb88320u & (0u - (crc & 1u)));
    }
    return ~crc;
}

void check_stored_crc(const std::string& path, uint32_t layer, const std::vector<uint8_t>& payload) {
    std::ifstream file(path, std::ios::binary);
    file.seekg((std::streamoff) (kExpertPackPrefixBytes + (uint64_t) layer * kExpertPackRecordBytes + 104));
    uint8_t bytes[4]{};
    file.read(reinterpret_cast<char*>(bytes), sizeof(bytes));
    CHECK(file.gcount() == sizeof(bytes));
    const uint32_t stored =
        (uint32_t) bytes[0] | ((uint32_t) bytes[1] << 8) | ((uint32_t) bytes[2] << 16) | ((uint32_t) bytes[3] << 24);
    CHECK(stored == reference_crc32(payload.data(), payload.size()));
}

} // namespace

int main() {
    const uint8_t check_crc[] = {'1', '2', '3', '4', '5', '6', '7', '8', '9'};
    CHECK(reference_crc32(check_crc, sizeof(check_crc)) == 0xcbf43926u);
    const fs::path dir = temp_dir();
    const fs::path model = dir / "model.gguf";
    const char abc[] = "abc";
    write_bytes(model, abc, 3);
    std::string err;
    const fs::path empty_file = dir / "empty.bin";
    write_bytes(empty_file, "", 0);
    CHECK(expert_pack_sha256_file(empty_file.string(), err) ==
          "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855");
    CHECK(expert_pack_sha256_file(model.string(), err) ==
          "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad");
    // Cross the hash_range/CNG update boundary. Reference: Python hashlib.sha256.
    const fs::path multichunk = dir / "hash-multichunk.bin";
    std::vector<uint8_t> hash_bytes((1u << 20) + 257, 0x55);
    write_bytes(multichunk, hash_bytes.data(), hash_bytes.size());
    CHECK(expert_pack_sha256_file(multichunk.string(), err) ==
          "b35dfaf41fc659d90c3d70dc67c0a45bc48b04f33d3a78c346acb026d2327aac");

    constexpr uint32_t n_expert = 4;
    constexpr uint64_t src_gsz = 8;
    constexpr uint64_t src_dsz = 6;
    std::vector<uint8_t> gate(n_expert * src_gsz), up(n_expert * src_gsz), down(n_expert * src_dsz);
    fill_pattern(gate, 0x11);
    fill_pattern(up, 0x41);
    fill_pattern(down, 0x81);
    const PackLayer layer = make_layer(gate, up, down, n_expert, src_gsz, src_dsz);
    uint8_t one[64];
    CHECK(pack_expert_blob(layer.layout, layer.source, 1, one));
    const std::vector<uint8_t> expect = expected_layer(layer);
    CHECK(std::memcmp(one, expect.data() + 64, 64) == 0);
    CHECK(one[8] == 0 && one[15] == 0);
    CHECK(one[24] == 0 && one[31] == 0);
    CHECK(one[38] == 0 && one[63] == 0);
    CHECK(std::memcmp(one, layer.source.gate + src_gsz, (size_t) src_gsz) == 0);

    ExpertBlobLayout overlap = layer.layout;
    overlap.up_off = 4;
    CHECK(!pack_expert_blob(overlap, layer.source, 0, one));

    PackBuildRequest req;
    req.model_path = model.string();
    req.layers = {layer, layer};
    auto pack = open_or_build_expert_pack(req, err);
    CHECK(pack != nullptr);
    if (!pack) std::fprintf(stderr, "build: %s\n", err.c_str());
    CHECK(expert_pack_last_scratch_bytes() == layer.layout.stride);
    CHECK(expert_pack_last_scratch_bytes() != layer.layout.stride * n_expert);
    if (pack) {
        CHECK(pack->layer_count() == 2);
        CHECK(pack->layer_bytes(0) == expect.size());
        CHECK(std::memcmp(pack->layer_data(0), expect.data(), expect.size()) == 0);
        CHECK(std::memcmp(pack->layer_data(1), expect.data(), expect.size()) == 0);
        check_stored_crc(pack->path(), 0, expect);
        check_stored_crc(pack->path(), 1, expect);
        const uint64_t payload_at = (kExpertPackPrefixBytes + 2 * kExpertPackRecordBytes + 4095ull) & ~4095ull;
        CHECK(payload_at == 4096);
        std::ifstream file(pack->path(), std::ios::binary);
        file.seekg((std::streamoff) payload_at);
        std::vector<char> got(expect.size());
        file.read(got.data(), (std::streamsize) got.size());
        CHECK(std::memcmp(got.data(), expect.data(), expect.size()) == 0);
        CHECK(!fs::exists(fs::path(pack->path() + ".partial")));
        std::vector<uint8_t> staged(expect.size() + 16, 0xcd);
        FILE* metrics = std::tmpfile();
        CHECK(metrics != nullptr);
        if (metrics) {
            strata::qwen35::Q36PhaseTrace phase(strata::qwen35::Q36Phase::prefill, 1, true, metrics);
            CHECK(pack->read_mapped(pack->layer_data(0), staged.data() + 8, expect.size()));
            CHECK(phase.total().read_calls == 1);
            CHECK(phase.total().read_requested_bytes == expect.size());
            CHECK(phase.total().read_completed_bytes == expect.size());
            phase.finish();
            std::fclose(metrics);
        }
        CHECK(std::memcmp(staged.data() + 8, expect.data(), expect.size()) == 0);
        CHECK(staged.front() == 0xcd && staged.back() == 0xcd);
        const uint8_t* end = pack->layer_data(1) + pack->layer_bytes(1);
        CHECK(pack->read_mapped(end, nullptr, 0));
        std::memset(staged.data(), 0xcd, staged.size());
        CHECK(!pack->read_mapped(end - 1, staged.data() + 8, 2));
        CHECK(!pack->read_mapped(nullptr, staged.data() + 8, 1));
        CHECK(!pack->read_mapped(pack->layer_data(0), nullptr, 1));
        CHECK(!pack->read_mapped(reinterpret_cast<const uint8_t*>(std::numeric_limits<uintptr_t>::max()),
                                 staged.data() + 8, 16));
        CHECK(!pack->read_mapped(pack->layer_data(0), staged.data() + 8, std::numeric_limits<size_t>::max()));
        for (uint8_t byte : staged) CHECK(byte == 0xcd);
    }

#if defined(_WIN32)
    if (pack) {
        // Restored mtimes, changed geometry and active writers must not turn
        // the local NTFS receipt into permission to accept changed weights.
        const std::string path = pack->path();
        CHECK(expert_pack_verify(path, model.string(), &req.layers, kQ36ExpertTail, true).empty());
        const bool cache_supported = fs::exists(path + ".verified");
        CHECK(expert_pack_verify(path, model.string(), &req.layers, kQ36ExpertTail, true).empty());
        CHECK(strata::qwen35::expert_pack_last_verification_cached() == cache_supported);
        auto wrong_geometry = req.layers;
        wrong_geometry[0].layout.n_embd++;
        CHECK(!expert_pack_verify(path, model.string(), &wrong_geometry, kQ36ExpertTail, true).empty());
        CHECK(!strata::qwen35::expert_pack_last_verification_cached());
        CHECK(!expert_pack_verify(path, model.string(), &req.layers, kQ36ExpertTail + 1, true).empty());
        _putenv_s("STRATA_Q36_VERIFY_PACK", "1");
        CHECK(expert_pack_verify(path, model.string(), &req.layers, kQ36ExpertTail, true).empty());
        CHECK(!strata::qwen35::expert_pack_last_verification_cached());
        _putenv_s("STRATA_Q36_VERIFY_PACK", "");
        const auto old_time = fs::last_write_time(model);
        write_bytes(model, "abd", 3);
        fs::last_write_time(model, old_time);
        CHECK(expert_pack_verify(path, model.string(), &req.layers, kQ36ExpertTail, true).find("not for this model") != std::string::npos);
        write_bytes(model, abc, 3);
        CHECK(expert_pack_verify(path, model.string(), &req.layers, kQ36ExpertTail, true).empty());
        write_bytes(path + ".verified", "invalid receipt", 15);
        CHECK(expert_pack_verify(path, model.string(), &req.layers, kQ36ExpertTail, true).empty());
        CHECK(!strata::qwen35::expert_pack_last_verification_cached());
        CHECK(expert_pack_verify(path, model.string(), &req.layers, kQ36ExpertTail, true).empty());
        HANDLE writer = CreateFileW(model.c_str(), GENERIC_WRITE, FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE,
                                    nullptr, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
        CHECK(writer != INVALID_HANDLE_VALUE);
        CHECK(expert_pack_verify(path, model.string(), &req.layers, kQ36ExpertTail, true).empty());
        CHECK(!strata::qwen35::expert_pack_last_verification_cached());
        if (writer != INVALID_HANDLE_VALUE) CloseHandle(writer);
    }
#endif

    // Cross buffered-read chunk boundaries and expert boundaries. The expected
    // bytes are assembled from the source tensors, not read from the mapping.
    PackBuildRequest staged_req = req;
    const auto staging_id = std::chrono::steady_clock::now().time_since_epoch().count();
    staged_req.model_path = (dir / ("file-staging-" + std::to_string(staging_id) + ".gguf")).string();
    write_bytes(staged_req.model_path, "file-staging-model", 18);
    PackLayer wide = layer;
    wide.layout.stride = (1u << 20) + 64;
    staged_req.layers = {wide, wide};
    auto staged_pack = open_or_build_expert_pack(staged_req, err);
    CHECK(staged_pack != nullptr);
    if (staged_pack) {
#if defined(_WIN32)
        CHECK(staged_pack->enable_bounded_reads(err));
        CHECK(staged_pack->bounded_read_scratch_bytes() == (1u << 20));
        // No payload read is allowed to fall back to memcpy/mmap, even on an
        // unaligned range or while several callers share the bounce buffer.
        DWORD old_protect = 0;
        CHECK(VirtualProtect(const_cast<uint8_t*>(staged_pack->layer_data(0)),
                             (SIZE_T) staged_pack->layer_bytes(0), PAGE_NOACCESS, &old_protect));
#endif
        const auto expected = expected_layer(wide);
        check_stored_crc(staged_pack->path(), 0, expected);
        check_stored_crc(staged_pack->path(), 1, expected);
        const uint8_t* base = staged_pack->layer_data(0);
        std::vector<uint8_t> got(expected.size());
        CHECK(staged_pack->read_mapped(base, got.data(), got.size()));
        CHECK(got == expected);
        std::atomic<bool> independent_offsets{true};
        std::vector<std::thread> readers;
        for (size_t worker = 0; worker < 4; ++worker) {
            readers.emplace_back([&, worker] {
                for (size_t trial = 0; trial < 128; ++trial) {
                    const size_t offset = ((worker + trial) % n_expert) * (size_t) wide.layout.stride;
                    uint8_t chunk[80];
                    std::memset(chunk, 0xcd, sizeof(chunk));
                    if (!staged_pack->read_mapped(base + offset, chunk + 8, 64) ||
                        std::memcmp(chunk + 8, expected.data() + offset, 64) != 0 ||
                        chunk[0] != 0xcd || chunk[79] != 0xcd) independent_offsets = false;
                }
            });
        }
        for (auto& reader : readers) reader.join();
        CHECK(independent_offsets.load());
        uint8_t cross[16] = {};
        const size_t boundary = (size_t) wide.layout.stride - 3;
        CHECK(staged_pack->read_mapped(base + boundary, cross, sizeof(cross)));
        CHECK(std::memcmp(cross, expected.data() + boundary, sizeof(cross)) == 0);
        uint8_t tail[19];
        const auto last = staged_pack->layer_data(1) + staged_pack->layer_bytes(1);
        CHECK(staged_pack->read_mapped(last - sizeof(tail), tail, sizeof(tail)));
        for (auto value : tail) CHECK(value == 0);

        // Exercise the production CPU staging helper against the protected
        // mapping, rather than merely testing its copy seam in isolation.
        struct BlobJob { const uint8_t* blob; size_t expert; };
        std::vector<BlobJob> wave_jobs;
        for (size_t i = 0; i < 19; ++i) wave_jobs.push_back({base + (i % n_expert) * wide.layout.stride, i % n_expert});
        strata::qwen35::ExpertCpuStaging cpu_stage((size_t) wide.layout.stride);
        size_t seen = 0;
        CHECK(cpu_stage.run(wave_jobs.data(), wave_jobs.size(), (size_t) wide.layout.stride,
            [&](const uint8_t* src, uint8_t* dst, size_t bytes) { return staged_pack->read_mapped(src, dst, bytes); },
            [&](BlobJob* batch, size_t count) {
                for (size_t i = 0; i < count; ++i) {
                    CHECK(std::memcmp(batch[i].blob, expected.data() + batch[i].expert * wide.layout.stride,
                                      (size_t) wide.layout.stride) == 0);
                    ++seen;
                }
            }));
        CHECK(seen == wave_jobs.size());

        // A path replacement cannot switch staging to a newly supplied file:
        // the read is bound to the same open object as the verified mapping.
        const std::string original = staged_pack->path();
        fs::rename(original, original + ".held");
        write_bytes(original, "unverified replacement", 22);
        CHECK(staged_pack->read_mapped(base, got.data(), got.size()));
        CHECK(got == expected);
        auto moved = std::move(*staged_pack);
        CHECK(!staged_pack->read_mapped(base, got.data(), 1));
        CHECK(moved.read_mapped(base, got.data(), got.size()));
        CHECK(got == expected);
    }

    PackLayer reread = layer;
    reread.source.gate = nullptr;
    reread.source.up = nullptr;
    reread.source.down = nullptr;
    PackBuildRequest again = req;
    again.layers = {reread, reread};
    err.clear();
    auto reused = open_or_build_expert_pack(again, err);
    CHECK(reused != nullptr);
    CHECK(expert_pack_last_scratch_bytes() == 0);
    if (reused) CHECK(std::memcmp(reused->layer_data(0), expect.data(), expect.size()) == 0);

#if defined(_WIN32)
    {
        PackBuildRequest bounded = again;
        bounded.bounded_reads = true;
        auto locked_pack = open_or_build_expert_pack(bounded, err);
        CHECK(locked_pack != nullptr);
        if (locked_pack) {
            CHECK(locked_pack->bounded_read_scratch_bytes() == (1u << 20));
            std::vector<uint8_t> bytes(expect.size());
            CHECK(locked_pack->read_mapped(locked_pack->layer_data(0), bytes.data(), bytes.size()));
            CHECK(bytes == expect);
            for (const auto& path : {model, fs::path(locked_pack->path())}) {
                HANDLE writer = CreateFileW(path.c_str(), GENERIC_WRITE,
                    FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE, nullptr, OPEN_EXISTING, 0, nullptr);
                CHECK(writer == INVALID_HANDLE_VALUE);
                if (writer != INVALID_HANDLE_VALUE) CloseHandle(writer);
            }
        }
    }
#endif

    if (pack) {
        std::fstream bad(pack->path(), std::ios::binary | std::ios::in | std::ios::out);
        const auto pack_time = fs::last_write_time(pack->path());
        bad.seekg((std::streamoff) kExpertPackPayloadOffsetField);
        uint64_t payload_at = 0;
        bad.read(reinterpret_cast<char*>(&payload_at), 8);
        bad.seekp((std::streamoff) payload_at);
        char flip = 0;
        bad.seekg((std::streamoff) payload_at);
        bad.read(&flip, 1);
        flip ^= 0x01;
        bad.seekp((std::streamoff) payload_at);
        bad.write(&flip, 1);
        bad.close();
        fs::last_write_time(pack->path(), pack_time);
        const std::string mismatch = expert_pack_verify(pack->path(), model.string(), &req.layers, kQ36ExpertTail, true);
        CHECK(mismatch.find("hash") != std::string::npos);
        CHECK(!strata::qwen35::expert_pack_last_verification_cached());
    }
    pack.reset();
    reused.reset();
    {
        PackBuildRequest no_repair = req;
        no_repair.bounded_reads = true;
        const auto pack_path = req.model_path + ".q36epk";
        const auto before = expert_pack_sha256_file(pack_path, err);
        CHECK(open_or_build_expert_pack(no_repair, err) == nullptr);
        CHECK(!err.empty());
        CHECK(expert_pack_sha256_file(pack_path, err) == before);
        CHECK(!fs::exists(pack_path + ".rejected"));
        CHECK(!fs::exists(pack_path + ".partial"));
    }
    err.clear();
    auto rebuilt = open_or_build_expert_pack(req, err);
    CHECK(rebuilt != nullptr);
    if (rebuilt) {
        CHECK(std::memcmp(rebuilt->layer_data(0), expect.data(), expect.size()) == 0);
        CHECK(fs::exists(fs::path(rebuilt->path() + ".rejected")));
    }

    const fs::path other = dir / "other.gguf";
    write_bytes(other, "other-model", 11);
    std::vector<uint8_t> fake(kExpertPackPrefixBytes, 0);
    std::memcpy(fake.data(), kExpertPackMagic, 8);
    fake[kExpertPackVersionField] = 99;
    write_bytes(fs::path(other.string() + ".q36epk"), fake.data(), fake.size());
    CHECK(expert_pack_verify(other.string() + ".q36epk", other.string(), nullptr, kQ36ExpertTail).find("version") !=
          std::string::npos);
    PackBuildRequest second = req;
    second.model_path = other.string();
    err.clear();
    auto replaced = open_or_build_expert_pack(second, err);
    CHECK(replaced != nullptr);
    if (!replaced) std::fprintf(stderr, "replace: %s\n", err.c_str());

    const fs::path locked = dir / "locked.gguf";
    write_bytes(locked, "locked-model", 12);
    const std::string lock_path = locked.string() + ".q36epk.lock";
#if defined(_WIN32)
    const uint32_t pid = GetCurrentProcessId();
#else
    const uint32_t pid = (uint32_t) ::getpid();
#endif
    uint8_t pid_bytes[4] = {(uint8_t) pid, (uint8_t) (pid >> 8), (uint8_t) (pid >> 16), (uint8_t) (pid >> 24)};
    write_bytes(lock_path, pid_bytes, 4);
    PackBuildRequest held = req;
    held.model_path = locked.string();
    err.clear();
    auto blocked = open_or_build_expert_pack(held, err);
    CHECK(blocked == nullptr);
    CHECK(err.find("lock") != std::string::npos);
    CHECK(!fs::exists(fs::path(locked.string() + ".q36epk")));
    fs::remove(lock_path);

    PackBuildRequest existing_only = held;
    existing_only.bounded_reads = true;
    CHECK(open_or_build_expert_pack(existing_only, err) == nullptr);
    CHECK(!fs::exists(locked.string() + ".q36epk"));
    CHECK(!fs::exists(locked.string() + ".q36epk.partial"));

    const fs::path partial_model = dir / "partial.gguf";
    write_bytes(partial_model, "partial-model", 13);
    write_bytes(fs::path(partial_model.string() + ".q36epk.partial"), "NOPE", 4);
    PackBuildRequest partial_req = req;
    partial_req.model_path = partial_model.string();
    err.clear();
    auto published = open_or_build_expert_pack(partial_req, err);
    CHECK(published != nullptr);
    if (published) CHECK(!fs::exists(fs::path(published->path() + ".partial")));

    PackBuildRequest bad_layout = req;
    bad_layout.model_path = (dir / "overlap.gguf").string();
    write_bytes(bad_layout.model_path, "overlap", 7);
    bad_layout.layers[0].layout.up_off = 4;
    bad_layout.layers[1].layout.up_off = 4;
    err.clear();
    CHECK(open_or_build_expert_pack(bad_layout, err) == nullptr);
    CHECK(!fs::exists(bad_layout.model_path + ".q36epk"));

    std::fprintf(stderr, g_fails ? "%d checks failed\n" : "q36 expert store checks passed\n", g_fails);
    return g_fails ? 1 : 0;
}
