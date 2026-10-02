#include "strata/core/conversation_disk.hpp"
#include "strata/core/on_device.hpp"

#include <cuda_runtime.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <cerrno>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <type_traits>

namespace strata::core {
namespace {
namespace fs = std::filesystem;

// One staging copy and one synced, dropped page-cache window: bounds both the pinned buffer and the page cache a
// park or restore can hold at any moment.
constexpr size_t kChunk = 32u << 20;

bool fail(std::string& error, const std::string& message) {
    error = "conversation disk: " + message;
    return false;
}

bool io_fail(std::string& error, const char* what, const std::string& file) {
    return fail(error, std::string(what) + " " + file + ": " + std::strerror(errno));
}

bool cuda_fail(std::string& error, const char* what, cudaError_t status) {
    return fail(error, std::string(what) + ": " + cudaGetErrorString(status));
}

struct Fd {
    int fd = -1;
    explicit Fd(int f) : fd(f) {}
    Fd(Fd&& other) noexcept : fd(other.fd) { other.fd = -1; }
    Fd(const Fd&) = delete;
    ~Fd() { if (fd >= 0) ::close(fd); }
};

// Written bytes are on disk and out of the page cache when this returns: the next chunk cannot pile up dirty pages
// that the kernel would have to find room for by evicting the experts' cache.
bool write_at(int fd, uint64_t offset, const uint8_t* p, size_t n) {
    for (size_t done = 0; done < n;) {
        const ssize_t w = ::pwrite(fd, p + done, n - done, (off_t) (offset + done));
        if (w < 0 && errno == EINTR) continue;
        if (w <= 0) return false;
        done += (size_t) w;
    }
    if (::sync_file_range(fd, (off_t) offset, (off_t) n,
                          SYNC_FILE_RANGE_WAIT_BEFORE | SYNC_FILE_RANGE_WRITE | SYNC_FILE_RANGE_WAIT_AFTER) != 0)
        return false;
    ::posix_fadvise(fd, (off_t) offset, (off_t) n, POSIX_FADV_DONTNEED);
    return true;
}

bool read_at(int fd, uint64_t offset, uint8_t* p, size_t n) {
    for (size_t done = 0; done < n;) {
        const ssize_t r = ::pread(fd, p + done, n - done, (off_t) (offset + done));
        if (r < 0 && errno == EINTR) continue;
        if (r <= 0) return false;
        done += (size_t) r;
    }
    ::posix_fadvise(fd, (off_t) offset, (off_t) n, POSIX_FADV_DONTNEED);
    return true;
}

bool write_file(const std::string& file, const std::vector<uint8_t>& bytes, std::string& error) {
    Fd f(::open(file.c_str(), O_WRONLY | O_CREAT | O_TRUNC | O_CLOEXEC, 0600));
    if (f.fd < 0) return io_fail(error, "open", file);
    for (size_t at = 0; at < bytes.size(); at += kChunk)
        if (!write_at(f.fd, at, bytes.data() + at, std::min(kChunk, bytes.size() - at))) return io_fail(error, "write", file);
    return true;
}

bool read_file(const std::string& file, std::vector<uint8_t>& bytes, std::string& error) {
    Fd f(::open(file.c_str(), O_RDONLY | O_CLOEXEC));
    struct stat st{};
    if (f.fd < 0 || ::fstat(f.fd, &st) != 0) return io_fail(error, "open", file);
    bytes.resize((size_t) st.st_size);
    for (size_t at = 0; at < bytes.size(); at += kChunk)
        if (!read_at(f.fd, at, bytes.data() + at, std::min(kChunk, bytes.size() - at))) return io_fail(error, "read", file);
    return true;
}

// The running-state files: a checkpoint (with its stage parts) as length-prefixed fields.
struct Writer {
    std::vector<uint8_t> out;
    void raw(const void* p, size_t n) {
        const auto* b = static_cast<const uint8_t*>(p);
        out.insert(out.end(), b, b + n);
    }
    void u64(uint64_t v) { raw(&v, sizeof v); }
    template<class T> void vec(const std::vector<T>& v) {
        static_assert(std::is_trivially_copyable_v<T>);
        u64(v.size());
        raw(v.data(), v.size() * sizeof(T));
    }
    void checkpoint(const ConversationCheckpoint& c) {
        vec(c.ids); vec(c.imgs); vec(c.gdn); vec(c.ple); vec(c.tails); vec(c.dead); vec(c.block_pos);
        u64(c.used); u64((uint64_t) c.layer_lo); u64((uint64_t) c.layer_hi);
        u64(c.stage_parts.size());
        for (const auto& part : c.stage_parts) checkpoint(part);
    }
};

struct Reader {
    const std::vector<uint8_t>& in;
    size_t at = 0;
    bool raw(void* p, size_t n) {
        if (n > in.size() - at) return false;
        std::memcpy(p, in.data() + at, n);
        at += n;
        return true;
    }
    bool u64(uint64_t& v) { return raw(&v, sizeof v); }
    template<class T> bool vec(std::vector<T>& v) {
        uint64_t n = 0;
        if (!u64(n) || n > (in.size() - at) / sizeof(T)) return false;
        v.resize((size_t) n);
        return raw(v.data(), (size_t) n * sizeof(T));
    }
    bool checkpoint(ConversationCheckpoint& c, int depth = 0) {
        uint64_t lo = 0, hi = 0, parts = 0;
        if (!vec(c.ids) || !vec(c.imgs) || !vec(c.gdn) || !vec(c.ple) || !vec(c.tails) || !vec(c.dead) ||
            !vec(c.block_pos) || !u64(c.used) || !u64(lo) || !u64(hi) || !u64(parts) || (depth > 0 && parts != 0) ||
            parts > 64) return false;
        c.layer_lo = (int64_t) lo; c.layer_hi = (int64_t) hi;
        c.stage_parts.resize((size_t) parts);
        for (auto& part : c.stage_parts)
            if (!checkpoint(part, depth + 1)) return false;
        return true;
    }
};

bool read_checkpoint(const std::string& file, ConversationCheckpoint& c, std::string& error) {
    std::vector<uint8_t> bytes;
    if (!read_file(file, bytes, error)) return false;
    Reader r{bytes};
    if (!r.checkpoint(c) || r.at != bytes.size()) return fail(error, "malformed running state in " + file);
    return true;
}

std::string kv_file(const std::string& dir, size_t entry, size_t buffer) {
    return dir + "/kv-" + std::to_string(entry) + "-" + std::to_string(buffer) + ".bin";
}

std::string ck_file(const std::string& dir, int64_t tokens) { return dir + "/ck-" + std::to_string(tokens) + ".bin"; }

bool hex_name(const std::string& name) {
    return name.size() == 16 && std::all_of(name.begin(), name.end(), [](char c) {
        return (c >= '0' && c <= '9') || (c >= 'a' && c <= 'f');
    });
}
} // namespace

ConversationDisk::~ConversationDisk() {
    if (bounce_) cudaFreeHost(bounce_);
}

bool ConversationDisk::open(const std::string& dir, uint64_t budget_bytes, std::string& error) {
    std::error_code ec;
    fs::create_directories(dir, ec);
    if (ec || !fs::is_directory(dir, ec)) return fail(error, "cannot create " + dir + ": " + ec.message());
    // Only this class's own directories (16 hex digits): the option may point at a directory that holds more.
    for (const auto& child : fs::directory_iterator(dir, ec))
        if (child.is_directory() && hex_name(child.path().filename().string())) fs::remove_all(child.path(), ec);
    if (ec) return fail(error, "cannot clear " + dir + ": " + ec.message());
    dir_ = dir;
    budget_ = budget_bytes;
    return true;
}

std::string ConversationDisk::path(const std::string& id) const {
    uint64_t h = 1469598103934665603ull;
    for (unsigned char c : id) { h ^= c; h *= 1099511628211ull; }
    char name[17];
    std::snprintf(name, sizeof name, "%016llx", (unsigned long long) h);
    return dir_ + "/" + name;
}

bool ConversationDisk::staging(std::string& error) {
    if (bounce_) return true;
    const cudaError_t status = cudaHostAlloc(&bounce_, kChunk, cudaHostAllocPortable);
    if (status == cudaSuccess) return true;
    bounce_ = nullptr;
    return cuda_fail(error, "staging buffer", status);
}

void ConversationDisk::drop(const std::string& id) {
    entries_.erase(id);
    std::error_code ec;
    fs::remove_all(path(id), ec);
}

void ConversationDisk::evict(const std::string& keep, Stats& stats) {
    for (;;) {
        uint64_t total = 0;
        const std::string* oldest = nullptr;
        uint64_t oldest_used = UINT64_MAX;
        for (const auto& [id, e] : entries_) {
            total += e.bytes;
            if (id != keep && e.used < oldest_used) { oldest_used = e.used; oldest = &id; }
        }
        stats.on_disk = total;
        if (total <= budget_ || oldest == nullptr) return;
        drop(std::string(*oldest));
        ++stats.evicted;
    }
}

bool ConversationDisk::park(const std::string& id, const ConversationView& view, const ConversationStages& stages,
                            const ModelGeometry& g, const QsaState& draft, int64_t unchanged, Stats& stats,
                            std::string& error) {
    stats = {};
    auto failed = [&]() { drop(id); return false; };
    if (!staging(error)) return failed();
    SavedConversation live;
    if (!conversation_live_save(live, view, stages, g, error)) return failed();
    const auto old_it = entries_.find(id);
    const Entry* old = old_it == entries_.end() ? nullptr : &old_it->second;
    if (old == nullptr || old->geometry != live.geometry || old->layer_lo != live.layer_lo ||
        old->layer_hi != live.layer_hi) unchanged = 0;
    const std::string dir = path(id);
    std::error_code ec;
    fs::create_directories(dir, ec);
    if (ec) return fail(error, "cannot create " + dir + ": " + ec.message()), failed();

    Entry next;
    next.ids = view.ids; next.imgs = view.images; next.cvec = view.cvec;
    next.geometry = live.geometry; next.layer_lo = live.layer_lo; next.layer_hi = live.layer_hi;
    const int64_t upto = (int64_t) view.ids.size();
    const auto entries = conversation_kv_entries(stages, draft);
    next.kv.resize(entries.size());
    for (size_t k = 0; k < entries.size(); ++k) {
        const auto& e = entries[k];
        const OnDevice on(e.dev);
        if (const cudaError_t status = cudaDeviceSynchronize(); status != cudaSuccess)
            return cuda_fail(error, "synchronize", status), failed();
        if (!conversation_kv_extent(*e.state, g, upto, e.index, next.kv[k], error)) return failed();
        // the drafter's final cell may not have been computed when the output cap was reached (as in the RAM save)
        const int64_t same = e.index ? unchanged : std::max<int64_t>(0, unchanged - 1);
        const bool comparable = old != nullptr && k < old->kv.size() && old->kv[k].format == next.kv[k].format &&
                                old->kv[k].heads == next.kv[k].heads && old->kv[k].head_dim == next.kv[k].head_dim &&
                                old->kv[k].page_size == next.kv[k].page_size;
        const auto kept = conversation_kv_kept(next.kv[k], g, comparable ? same : 0, e.index);
        const auto buffers = conversation_kv_buffers(*e.state);
        for (size_t b = 0; b < buffers.size(); ++b) {
            const uint64_t size = next.kv[k].sizes[b];
            const uint64_t keep = std::min({kept[b], size, comparable ? old->kv[k].sizes[b] : 0});
            const std::string file = kv_file(dir, k, b);
            Fd f(::open(file.c_str(), O_RDWR | O_CREAT | O_CLOEXEC, 0600));
            if (f.fd < 0) return io_fail(error, "open", file), failed();
            if (size > keep && buffers[b] == nullptr) return fail(error, "missing K/V buffer"), failed();
            for (uint64_t at = keep; at < size; at += kChunk) {
                const size_t n = (size_t) std::min<uint64_t>(kChunk, size - at);
                if (const cudaError_t status = cudaMemcpy(bounce_, static_cast<const uint8_t*>(buffers[b]) + at, n,
                                                          cudaMemcpyDefault); status != cudaSuccess)
                    return cuda_fail(error, "K/V copy", status), failed();
                if (!write_at(f.fd, at, static_cast<const uint8_t*>(bounce_), n)) return io_fail(error, "write", file), failed();
            }
            if (::ftruncate(f.fd, (off_t) size) != 0) return io_fail(error, "truncate", file), failed();
            stats.written += size - keep;
            stats.kept += keep;
            next.bytes += size;
        }
    }

    Writer w;
    w.checkpoint(live.live);
    if (!write_file(dir + "/live.bin", w.out, error)) return failed();
    stats.written += w.out.size();
    next.bytes += w.out.size();
    // A checkpoint at N tokens is unchanged when the K/V before N is: then it is the very one this conversation was
    // restored with (prompt reading only adds checkpoints past where it resumed).
    for (const auto& c : view.checkpoints) {
        const int64_t n = (int64_t) c.ids.size();
        const std::string file = ck_file(dir, n);
        const bool reuse = n <= unchanged && old != nullptr &&
                           std::find(old->checkpoints.begin(), old->checkpoints.end(), n) != old->checkpoints.end();
        if (reuse) {
            stats.kept += fs::file_size(file, ec);
            next.bytes += fs::file_size(file, ec);
        } else {
            Writer ck;
            ck.checkpoint(c);
            if (!write_file(file, ck.out, error)) return failed();
            stats.written += ck.out.size();
            next.bytes += ck.out.size();
        }
        next.checkpoints.push_back(n);
    }
    if (old != nullptr)
        for (int64_t n : old->checkpoints)
            if (std::find(next.checkpoints.begin(), next.checkpoints.end(), n) == next.checkpoints.end())
                fs::remove(ck_file(dir, n), ec);
    next.used = ++clock_;
    entries_[id] = std::move(next);
    evict(id, stats);
    return true;
}

bool ConversationDisk::load(const std::string& id, const ConversationStages& stages, const ModelGeometry& g,
                            const QsaState& draft, SavedConversation& image, std::string& error) {
    const auto it = entries_.find(id);
    if (it == entries_.end()) return fail(error, "no parked conversation " + id);
    const Entry& e = it->second;
    const std::string dir = path(id);
    SavedConversation loaded;
    loaded.geometry = e.geometry; loaded.layer_lo = e.layer_lo; loaded.layer_hi = e.layer_hi; loaded.cvec = e.cvec;
    if (!read_checkpoint(dir + "/live.bin", loaded.live, error)) return false;
    if (loaded.live.ids != e.ids || loaded.live.imgs != e.imgs) return fail(error, "live tokens differ from the index");
    for (int64_t n : e.checkpoints)
        if (!read_checkpoint(ck_file(dir, n), loaded.checkpoints.emplace_back(), error)) return false;
    if (!conversation_snapshot_validate_state(loaded, stages, g, error)) return false;
    const auto entries = conversation_kv_entries(stages, draft);
    if (entries.size() != e.kv.size()) return fail(error, "K/V layer count differs");
    for (size_t k = 0; k < entries.size(); ++k) {
        ConversationKvExtent want;
        if (!conversation_kv_extent(*entries[k].state, g, (int64_t) e.ids.size(), entries[k].index, want, error))
            return false;
        if (!(want == e.kv[k])) return fail(error, "K/V extent differs");
        for (size_t b = 0; b < want.sizes.size(); ++b) {
            std::error_code ec;
            if (fs::file_size(kv_file(dir, k, b), ec) != want.sizes[b] || ec) return fail(error, "K/V file size differs");
        }
    }
    image = std::move(loaded);
    it->second.used = ++clock_;
    return true;
}

ConversationRestore ConversationDisk::restore(const std::string& id, const SavedConversation& image,
                                              const ConversationStages& stages, const ModelGeometry& g,
                                              const QsaState& draft, uint64_t& bytes, std::string& error) {
    bytes = 0;
    const auto it = entries_.find(id);
    if (it == entries_.end()) return fail(error, "no parked conversation " + id), ConversationRestore::invalid;
    if (!staging(error)) return ConversationRestore::invalid;
    const Entry& e = it->second;
    const std::string dir = path(id);
    const int64_t upto = (int64_t) e.ids.size();
    const auto entries = conversation_kv_entries(stages, draft);
    // Every file is opened before the first write, so a missing one is still a clean miss.
    std::vector<Fd> files;
    for (size_t k = 0; k < entries.size(); ++k)
        for (size_t b = 0; b < 5; ++b) {
            const std::string file = kv_file(dir, k, b);
            files.emplace_back(::open(file.c_str(), O_RDONLY | O_CLOEXEC));
            if (files.back().fd < 0) return io_fail(error, "open", file), ConversationRestore::invalid;
        }
    bool ok = true;
    for (size_t k = 0; k < entries.size() && ok; ++k) {
        const auto& en = entries[k];
        const OnDevice on(en.dev);
        if (const cudaError_t status = cudaDeviceSynchronize(); status != cudaSuccess) {
            ok = cuda_fail(error, "synchronize", status);
            break;
        }
        const auto buffers = conversation_kv_buffers(*en.state);
        for (size_t b = 0; b < buffers.size() && ok; ++b) {
            const uint64_t size = e.kv[k].sizes[b];
            if (size && buffers[b] == nullptr) { ok = fail(error, "missing target K/V buffer"); break; }
            for (uint64_t at = 0; at < size && ok; at += kChunk) {
                const size_t n = (size_t) std::min<uint64_t>(kChunk, size - at);
                if (!read_at(files[k * 5 + b].fd, at, static_cast<uint8_t*>(bounce_), n)) {
                    ok = io_fail(error, "read", kv_file(dir, k, b));
                } else if (const cudaError_t status = cudaMemcpy(static_cast<uint8_t*>(buffers[b]) + at, bounce_, n,
                                                                 cudaMemcpyDefault); status != cudaSuccess) {
                    ok = cuda_fail(error, "K/V copy", status);
                }
                bytes += n;
            }
        }
    }
    // The VRAM slots and the draft ring follow the authoritative buffers either way: after a failure they must not
    // keep serving the outgoing conversation's pages as if they were current.
    for (const auto& en : entries) {
        const OnDevice on(en.dev);
        std::string ignored;
        if (!conversation_kv_restored(*en.state, g, upto, ok ? error : ignored)) ok = false;
    }
    if (ok) ok = conversation_live_restore(image.live, stages, g, error);
    return ok ? ConversationRestore::restored : ConversationRestore::transfer_failed;
}

} // namespace strata::core
