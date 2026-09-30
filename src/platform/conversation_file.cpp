// The streaming envelope/integrity-footer design follows @maedoc's Strata #52.
// Serialize the shared image instead of its independent CUDA segment/apply walk.
#include "strata/platform/conversation_file.hpp"
#include "strata/core/conversation_memory.hpp"

#include <openssl/evp.h>

#include <algorithm>
#include <bit>
#include <fstream>
#include <limits>
#include <map>
#include <memory>
#include <stdexcept>

namespace strata::platform {
namespace {
using core::ConversationCheckpoint;
using core::ConversationImageKey;
using core::ConversationKv;
using core::ConversationBuffer;
using core::SavedConversation;
constexpr std::array<uint8_t, 8> magic{'S','T','R','S','N','A','P',2};
// GPU state blobs retain their native scalar representation.
static_assert(std::endian::native == std::endian::little);
// These portable allowances also bound vector-object storage before resize.
constexpr uint64_t checkpoint_overhead = 512, kv_overhead = 256;
static_assert(sizeof(ConversationCheckpoint) <= checkpoint_overhead);
static_assert(sizeof(ConversationKv) <= kv_overhead);
static_assert(sizeof(SavedConversation) <= 1024);

struct Digest {
    std::unique_ptr<EVP_MD_CTX, decltype(&EVP_MD_CTX_free)> ctx{EVP_MD_CTX_new(), EVP_MD_CTX_free};
    Digest() {
        if (!ctx || EVP_DigestInit_ex(ctx.get(), EVP_sha256(), nullptr) != 1)
            throw std::runtime_error("SHA-256 initialization failed");
    }
    void update(const void* p, size_t n) {
        if (n && EVP_DigestUpdate(ctx.get(), p, n) != 1)
            throw std::runtime_error("SHA-256 update failed");
    }
    ConversationDigest finish() {
        ConversationDigest out;
        unsigned n = 0;
        if (EVP_DigestFinal_ex(ctx.get(), out.data(), &n) != 1 || n != out.size())
            throw std::runtime_error("SHA-256 finalization failed");
        return out;
    }
};

std::array<uint8_t, 8> little(uint64_t value) {
    std::array<uint8_t, 8> bytes;
    for (unsigned i = 0; i < 8; ++i) bytes[i] = uint8_t(value >> (8 * i));
    return bytes;
}
void add(uint64_t& total, uint64_t count, uint64_t width = 1) {
    if (width && count > (UINT64_MAX - total) / width)
        throw std::runtime_error("snapshot allocation count overflow");
    total += count * width;
}
uint64_t allocation_bound(const SavedConversation& image) {
    uint64_t total = kConversationFileWorkspace;
    add(total, image.checkpoints.size(), checkpoint_overhead);
    add(total, image.kv.size(), kv_overhead);
    auto checkpoint = [&](const ConversationCheckpoint& c) {
        if (!c.stage_parts.empty()) throw std::runtime_error("layer-split snapshots are unsupported");
        add(total, c.ids.size(), sizeof(int32_t));
        add(total, c.imgs.size(), sizeof(ConversationImageKey));
        for (const auto* v : {&c.gdn, &c.ple, &c.tails, &c.dead, &c.block_pos}) add(total, v->size());
    };
    checkpoint(image.live);
    for (const auto& c : image.checkpoints) checkpoint(c);
    for (const auto& kv : image.kv)
        for (const auto* v : {&kv.k, &kv.v, &kv.k_scale, &kv.v_scale, &kv.pooled}) {
            add(total, v->size());
            // The reader rebuilds canonical segments, independently of how many
            // small appends produced the in-memory image. Allow portable directory storage.
            add(total, v->size() / ConversationBuffer::segment_bytes +
                       (v->size() % ConversationBuffer::segment_bytes != 0), 64);
        }
    return total;
}

struct IoProgress {
    ConversationIoProgress callback = nullptr;
    uint64_t bytes = 0;
    void advance(uint64_t n) {
        if (!callback) return;
        if (n >= (1 << 20) - bytes) { bytes = 0; callback(); }
        else bytes += n;
    }
    void finish() { if (callback) callback(); }
};

struct Writer {
    std::ostream& stream;
    Digest digest;
    IoProgress progress;
    void bytes(const void* p, size_t n) {
        const auto* data = static_cast<const char*>(p);
        // Keep streamsize conversion bounded even for a very large snapshot.
        while (n) {
            const size_t chunk = std::min<size_t>(n, 65536);
            if (!stream.write(data, chunk)) throw std::runtime_error("snapshot write failed");
            digest.update(data, chunk);
            progress.advance(chunk);
            data += chunk; n -= chunk;
        }
    }
    void integer(uint64_t value) { const auto b = little(value); bytes(b.data(), b.size()); }
    void blob(const std::vector<uint8_t>& v) { integer(v.size()); bytes(v.data(), v.size()); }
    void blob(const ConversationBuffer& v) {
        integer(v.size());
        v.visit(0, v.size(), [&](const uint8_t* p, size_t n, size_t) { bytes(p,n); return true; });
    }
    void checkpoint(const ConversationCheckpoint& c) {
        integer(c.ids.size());
        for (int32_t id : c.ids) {
            const auto b = little(std::bit_cast<uint32_t>(id)); bytes(b.data(), 4);
        }
        integer(c.imgs.size());
        for (const auto& i : c.imgs) { integer(std::bit_cast<uint64_t>(i.start)); integer(i.hash); }
        integer(c.used);
        for (const auto* v : {&c.gdn, &c.ple, &c.tails, &c.dead, &c.block_pos}) blob(*v);
    }
};

struct Reader {
    std::istream& stream;
    Digest digest;
    uint64_t remaining = 0;
    IoProgress progress;
    void bytes(void* p, size_t n) {
        auto* data = static_cast<char*>(p);
        while (n) {
            const size_t chunk = std::min<size_t>(n, 65536);
            if (!stream.read(data, chunk)) throw std::runtime_error("truncated snapshot");
            digest.update(data, chunk);
            progress.advance(chunk);
            data += chunk; n -= chunk;
        }
    }
    uint64_t integer(size_t width = 8) {
        std::array<uint8_t, 8> b{}; bytes(b.data(), width);
        uint64_t out = 0;
        for (size_t i = 0; i < width; ++i) out |= uint64_t(b[i]) << (8 * i);
        return out;
    }
    int64_t signed_integer() { return std::bit_cast<int64_t>(integer()); }
    template<class T> void allocate(std::vector<T>& v, uint64_t n, uint64_t width = sizeof(T)) {
        if (n > v.max_size() || n > remaining / width)
            throw std::runtime_error("snapshot exceeds admitted staging bytes");
        remaining -= n * width;
        v.resize(static_cast<size_t>(n));
        // New vectors normally allocate exactly n elements; account for any excess.
        const uint64_t extra = (v.capacity() - v.size()) * sizeof(T);
        if (extra > remaining) throw std::runtime_error("snapshot vector capacity exceeds staging bound");
        remaining -= extra;
    }
    void blob(std::vector<uint8_t>& v) { allocate(v, integer()); bytes(v.data(), v.size()); }
    void blob(ConversationBuffer& v) {
        const uint64_t n = integer();
        if (n > SIZE_MAX) throw std::runtime_error("snapshot buffer size overflow");
        const size_t admitted = v.allocation_peak(size_t(n));
        if (admitted == SIZE_MAX || admitted > remaining)
            throw std::runtime_error("snapshot segmented buffer exceeds staging bound");
        remaining -= admitted;
        v.resize(size_t(n));
        if (v.bytes() > admitted) {
            const size_t extra = v.bytes() - admitted;
            if (extra > remaining) throw std::runtime_error("snapshot segment capacity exceeds staging bound");
            remaining -= extra;
        }
        v.visit(0, v.size(), [&](uint8_t* p, size_t count, size_t) { bytes(p,count); return true; });
    }
    void checkpoint(ConversationCheckpoint& c) {
        allocate(c.ids, integer());
        for (auto& id : c.ids) id = std::bit_cast<int32_t>(static_cast<uint32_t>(integer(4)));
        allocate(c.imgs, integer());
        for (auto& i : c.imgs) { i.start = signed_integer(); i.hash = integer(); }
        c.used = integer();
        for (auto* v : {&c.gdn, &c.ple, &c.tails, &c.dead, &c.block_pos}) blob(*v);
    }
};

// Candidate selection reads only prefix metadata. Full decoding checks integrity of
// the selected file before the shared core can apply any state.
struct Probe {
    std::istream& stream;
    uint64_t file_remaining, allocation_remaining;
    IoProgress progress;
    void consume(uint64_t n) {
        if (n > file_remaining) throw std::runtime_error("truncated snapshot metadata");
        file_remaining -= n;
    }
    void bytes(void* p, size_t n) {
        consume(n);
        if (!stream.read(static_cast<char*>(p), n)) throw std::runtime_error("snapshot metadata read failed");
        progress.advance(n);
    }
    void skip(uint64_t n) {
        consume(n);
        if (n > uint64_t(std::numeric_limits<std::streamoff>::max()) ||
            !stream.seekg(static_cast<std::streamoff>(n), std::ios::cur))
            throw std::runtime_error("snapshot metadata seek failed");
        progress.advance(n);
    }
    uint64_t integer(size_t width = 8) {
        std::array<uint8_t, 8> b{}; bytes(b.data(), width);
        uint64_t out = 0;
        for (size_t i = 0; i < width; ++i) out |= uint64_t(b[i]) << (8 * i);
        return out;
    }
    void account(uint64_t n, uint64_t width) {
        if (n > allocation_remaining / width)
            throw std::runtime_error("snapshot metadata exceeds staging bound");
        allocation_remaining -= n * width;
    }
    int64_t checkpoint(const std::vector<int64_t>& prompt, const std::vector<ConversationImageKey>& images) {
        const uint64_t count = integer(); account(count, sizeof(int32_t));
        bool equal = count != 0 && count < prompt.size();
        if (equal) {
            for (size_t i = 0; i < count; ++i)
                if (std::bit_cast<int32_t>(static_cast<uint32_t>(integer(4))) != prompt[i]) equal = false;
        } else skip(count * 4);
        const uint64_t image_count = integer(); account(image_count, sizeof(ConversationImageKey));
        if (equal) {
            const size_t expected = std::count_if(images.begin(), images.end(),
                [&](const auto& image) { return image.start < static_cast<int64_t>(count); });
            equal = image_count == expected;
        }
        if (equal) {
            for (const auto& image : images) {
                if (image.start >= static_cast<int64_t>(count)) continue;
                const int64_t start = std::bit_cast<int64_t>(integer());
                const uint64_t hash = integer();
                if (image.start != start || image.hash != hash) equal = false;
            }
        } else skip(image_count * 16);
        skip(8); // checkpoint retention counter
        for (unsigned i = 0; i < 5; ++i) {
            const uint64_t n = integer(); account(n, 1); skip(n);
        }
        return equal ? static_cast<int64_t>(count) : 0;
    }
};
} // namespace

bool conversation_identity(const std::vector<ConversationAsset>& assets, const ConversationSettings& settings,
                           ConversationIdentity& identity, std::string& error) {
    try {
        ConversationIdentity result;
        auto field = [&](const std::string& name, const ConversationDigest& digest) {
            if (result.fields.size() == 256 || name.empty() || name.size() > 128 ||
                name.find_first_not_of("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_./:-") != std::string::npos)
                throw std::runtime_error("invalid or excessive snapshot identity fields");
            if (std::any_of(result.fields.begin(), result.fields.end(), [&](const auto& f) { return f.name == name; }))
                throw std::runtime_error("duplicate snapshot identity field: " + name);
            result.fields.push_back({name, digest});
        };
        for (const auto& [name, value] : settings) {
            Digest hash; hash.update(value.data(), value.size());
            field("runtime/" + name, hash.finish());
        }
        std::array<char, 32768> buffer;
        std::map<std::filesystem::path, ConversationDigest> contents;
        for (const auto& asset : assets) {
            // Hash each complete file once per startup, even when several roles
            // share a GGUF. Never trust a file-stat-only cache across processes.
            const auto path = std::filesystem::canonical(asset.path);
            auto [entry, fresh] = contents.try_emplace(path);
            if (fresh) {
                std::ifstream f(path, std::ios::binary);
                if (!f) throw std::runtime_error("cannot open identity asset: " + path.string());
                Digest content;
                while (f) {
                    f.read(buffer.data(), buffer.size());
                    content.update(buffer.data(), static_cast<size_t>(f.gcount()));
                }
                if (f.bad() || !f.eof()) throw std::runtime_error("cannot read identity asset: " + path.string());
                entry->second = content.finish();
            }
            field("asset/" + asset.role, entry->second);
        }
        Digest hash;
        const std::string domain = "strata-conversation-state-v2";
        hash.update(domain.data(), domain.size());
        for (const auto& f : result.fields) {
            const auto n = little(f.name.size()); hash.update(n.data(), n.size());
            hash.update(f.name.data(), f.name.size()); hash.update(f.digest.data(), f.digest.size());
        }
        result.digest = hash.finish();
        identity = std::move(result);
        return true;
    } catch (const std::exception& e) { error = e.what(); return false; }
}

namespace {
uint64_t identity_bytes(const ConversationIdentity& identity) {
    if (identity.fields.size() > 256) throw std::runtime_error("too many snapshot identity fields");
    uint64_t bytes = 32 + 8;
    for (const auto& field : identity.fields) {
        if (field.name.empty() || field.name.size() > 128)
            throw std::runtime_error("invalid snapshot identity field name");
        add(bytes, 8 + field.name.size() + field.digest.size());
    }
    return bytes;
}
void write_identity(Writer& w, const ConversationIdentity& identity) {
    w.bytes(identity.digest.data(), identity.digest.size()); w.integer(identity.fields.size());
    for (const auto& f : identity.fields) {
        w.integer(f.name.size()); w.bytes(f.name.data(), f.name.size()); w.bytes(f.digest.data(), f.digest.size());
    }
}
template<class Input> void check_identity(Input& input, const ConversationIdentity& expected) {
    ConversationDigest stored;
    input.bytes(stored.data(), stored.size());
    const auto count = input.integer();
    if (count > 256) throw std::runtime_error("invalid snapshot identity field count");
    // Compare in bounded stack storage before admitting or decoding any state.
    for (size_t i = 0; i < count; ++i) {
        const auto length = input.integer();
        if (!length || length > 128) throw std::runtime_error("invalid snapshot identity field name");
        std::array<char, 128> name{}; ConversationDigest digest;
        input.bytes(name.data(), size_t(length)); input.bytes(digest.data(), digest.size());
        if (i >= expected.fields.size()) throw std::runtime_error("snapshot identity differs: extra field");
        const auto& field = expected.fields[i];
        if (field.name.size() != length || !std::equal(field.name.begin(), field.name.end(), name.begin()) || digest != field.digest)
            throw std::runtime_error("snapshot identity differs: " + field.name);
    }
    if (count < expected.fields.size()) throw std::runtime_error("snapshot identity differs: " + expected.fields[count].name);
    if (stored != expected.digest) throw std::runtime_error("snapshot identity differs: digest");
}
} // namespace

bool conversation_file_write(std::ostream& stream, const SavedConversation& image,
                             const ConversationIdentity& identity, std::string& error, ConversationIoProgress progress) {
    try {
        const auto bound = allocation_bound(image);
        identity_bytes(identity);
        Writer w{stream, {}, {progress}};
        w.bytes(magic.data(), magic.size()); write_identity(w, identity); w.integer(bound);
        for (int64_t n : image.geometry) w.integer(std::bit_cast<uint64_t>(n));
        w.integer(std::bit_cast<uint64_t>(image.layer_lo)); w.integer(std::bit_cast<uint64_t>(image.layer_hi));
        w.integer(image.cvec ? 1 : 0);
        w.checkpoint(image.live);
        w.integer(image.checkpoints.size());
        for (const auto& c : image.checkpoints) w.checkpoint(c);
        w.integer(image.kv.size());
        for (const auto& kv : image.kv) {
            w.integer(static_cast<uint64_t>(kv.format));
            for (int64_t n : {kv.cells, kv.heads, kv.head_dim, kv.page_size, kv.pooled_rows, kv.idx_dim})
                w.integer(std::bit_cast<uint64_t>(n));
            for (const auto* v : {&kv.k, &kv.v, &kv.k_scale, &kv.v_scale, &kv.pooled}) w.blob(*v);
        }
        const auto sum = w.digest.finish();
        if (!stream.write(reinterpret_cast<const char*>(sum.data()), sum.size()))
            throw std::runtime_error("snapshot footer write failed");
        w.progress.finish();
        return true;
    } catch (const std::exception& e) { error = e.what(); return false; }
}

bool conversation_file_size(const SavedConversation& image, const ConversationIdentity& identity, uint64_t& bytes, std::string& error) {
    try {
        // Header, geometry, steering, checkpoint/KV counts and integrity footer.
        uint64_t total = 8 + identity_bytes(identity) + 8 + 18 * 8 + 16 + 8 + 8 + 8 + 32;
        auto checkpoint = [&](const ConversationCheckpoint& c) {
            if (!c.stage_parts.empty()) throw std::runtime_error("layer-split snapshots are unsupported");
            add(total, 8 * 8); // token/image counts, retention counter, five blob lengths
            add(total, c.ids.size(), 4); add(total, c.imgs.size(), 16);
            for (const auto* v : {&c.gdn, &c.ple, &c.tails, &c.dead, &c.block_pos}) add(total, v->size());
        };
        checkpoint(image.live);
        for (const auto& c : image.checkpoints) checkpoint(c);
        for (const auto& kv : image.kv) {
            add(total, 12 * 8); // format, six dimensions, five blob lengths
            for (const auto* v : {&kv.k, &kv.v, &kv.k_scale, &kv.v_scale, &kv.pooled}) add(total, v->size());
        }
        bytes = total;
        return true;
    } catch (const std::exception& e) { error = e.what(); return false; }
}

bool conversation_file_read(std::istream& stream, const ConversationIdentity& identity,
                            uint64_t staging_limit, std::optional<uint64_t> available, uint64_t floor,
                            SavedConversation& output, std::string& error, ConversationIoProgress progress) {
    try {
        Reader r{stream, {}, 0, {progress}};
        std::array<uint8_t, 8> tag;
        r.bytes(tag.data(), tag.size());
        if (tag != magic) throw std::runtime_error("unsupported conversation snapshot format");
        check_identity(r, identity);
        const uint64_t bound = r.integer();
        if (bound < kConversationFileWorkspace || bound > staging_limit ||
            !core::conversation_memory_admit(available, bound, floor))
            throw std::runtime_error("snapshot staging admission denied");
        r.remaining = bound - kConversationFileWorkspace;
        SavedConversation image;
        for (auto& n : image.geometry) n = r.signed_integer();
        image.layer_lo = r.signed_integer(); image.layer_hi = r.signed_integer();
        const auto cvec = r.integer();
        if (cvec > 1) throw std::runtime_error("invalid snapshot steering state");
        image.cvec = cvec != 0;
        r.checkpoint(image.live);
        r.allocate(image.checkpoints, r.integer(), checkpoint_overhead);
        for (auto& c : image.checkpoints) r.checkpoint(c);
        r.allocate(image.kv, r.integer(), kv_overhead);
        for (auto& kv : image.kv) {
            const auto format = r.integer();
            if (format > 3) throw std::runtime_error("unsupported snapshot KV format");
            kv.format = static_cast<int>(format);
            kv.cells = r.signed_integer(); kv.heads = r.signed_integer(); kv.head_dim = r.signed_integer();
            kv.page_size = r.signed_integer(); kv.pooled_rows = r.signed_integer(); kv.idx_dim = r.signed_integer();
            for (auto* v : {&kv.k, &kv.v, &kv.k_scale, &kv.v_scale, &kv.pooled}) r.blob(*v);
        }
        ConversationDigest footer;
        if (!stream.read(reinterpret_cast<char*>(footer.data()), footer.size()) || footer != r.digest.finish())
            throw std::runtime_error("snapshot integrity check failed");
        if (stream.peek() != std::char_traits<char>::eof() || stream.bad())
            throw std::runtime_error("snapshot has trailing data or read error");
        output = std::move(image);
        r.progress.finish();
        return true;
    } catch (const std::exception& e) { error = e.what(); return false; }
}

bool conversation_file_match(std::istream& stream, const ConversationIdentity& identity,
                             uint64_t staging_limit, const std::vector<int64_t>& prompt,
                             const std::vector<ConversationImageKey>& images, bool cvec,
                             ConversationFileMatch& match, std::string& error, ConversationIoProgress progress) {
    try {
        if (!stream.seekg(0, std::ios::end)) throw std::runtime_error("snapshot metadata requires seekable input");
        const auto end = stream.tellg();
        if (end < std::streampos(0) || !stream.seekg(0)) throw std::runtime_error("cannot measure snapshot file");
        Probe p{stream, static_cast<uint64_t>(end), 0, {progress}};
        std::array<uint8_t, 8> tag;
        p.bytes(tag.data(), tag.size());
        if (tag != magic) throw std::runtime_error("unsupported conversation snapshot format");
        check_identity(p, identity);
        const uint64_t bound = p.integer();
        if (bound < kConversationFileWorkspace || bound > staging_limit)
            throw std::runtime_error("snapshot staging admission denied");
        p.allocation_remaining = bound - kConversationFileWorkspace;
        p.skip(18 * 8 + 16); // geometry and session layer range is validated by the shared core after selection
        const auto steering = p.integer();
        if (steering > 1) throw std::runtime_error("invalid snapshot steering state");
        ConversationFileMatch found{0, false, bound};
        const int64_t live = p.checkpoint(prompt, images);
        if (live > 0) found = {live, true, bound};
        const uint64_t checkpoints = p.integer(); p.account(checkpoints, checkpoint_overhead);
        for (uint64_t i = 0; i < checkpoints; ++i) {
            const int64_t n = p.checkpoint(prompt, images);
            if (n > found.tokens) found = {n, false, bound};
        }
        if (bool(steering) != cvec) found.tokens = 0;
        match = found;
        p.progress.finish();
        return true;
    } catch (const std::exception& e) { error = e.what(); return false; }
}
} // namespace strata::platform
