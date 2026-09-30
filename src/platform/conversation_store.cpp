// Eviction-only persistence of the shared image, following @maedoc's #52
// temporary-file / atomic-publication approach without its separate apply walk.
#include "strata/platform/conversation_store.hpp"

#include <openssl/rand.h>

#include <algorithm>
#include <cerrno>
#include <cstdio>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <streambuf>
#include <system_error>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <fcntl.h>
#include <io.h>
#include <sys/stat.h>
#else
#include <fcntl.h>
#include <sys/file.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

namespace strata::platform {
namespace {
namespace fs = std::filesystem;
[[noreturn]] void system_failure(const char* operation) {
    throw std::system_error(errno, std::generic_category(), operation);
}
bool managed_name(const fs::path& path, const char* extension) {
    if (path.extension() != extension) return false;
    const auto stem = path.stem().string();
    return stem.size() == 32 && std::all_of(stem.begin(), stem.end(),
        [](char c) { return (c >= '0' && c <= '9') || (c >= 'a' && c <= 'f'); });
}
std::string unique_name() {
    std::array<uint8_t, 16> bytes;
    if (RAND_bytes(bytes.data(), int(bytes.size())) != 1) throw std::runtime_error("snapshot name generation failed");
    constexpr char hex[] = "0123456789abcdef";
    std::string name;
    for (auto b : bytes) { name += hex[b >> 4]; name += hex[b & 15]; }
    return name;
}
struct RemoveFile {
    fs::path path;
    bool armed = false;
    ~RemoveFile() { if (armed && !path.empty()) { std::error_code ec; fs::remove(path, ec); } }
};

// FILE buffering stays small; no whole-file staging. Enforce the reserved quota
// even if the codec's length estimator ever disagrees with its writer.
class FileBuffer : public std::streambuf {
public:
    FileBuffer(FILE* file, uint64_t limit) : file_(file), remaining_(limit) {}
    uint64_t remaining() const { return remaining_; }
protected:
    std::streamsize xsputn(const char* data, std::streamsize n) override {
        if (n < 0 || uint64_t(n) > remaining_) return 0;
        const size_t written = std::fwrite(data, 1, size_t(n), file_);
        remaining_ -= written;
        return static_cast<std::streamsize>(written);
    }
    int_type overflow(int_type c) override {
        if (traits_type::eq_int_type(c, traits_type::eof())) return traits_type::not_eof(c);
        const char value = traits_type::to_char_type(c);
        return xsputn(&value, 1) == 1 ? c : traits_type::eof();
    }
    int sync() override { return std::fflush(file_) == 0 ? 0 : -1; }
private:
    FILE* file_;
    uint64_t remaining_;
};

FILE* create_file(const fs::path& path) {
#ifdef _WIN32
    const int fd = _wopen(path.c_str(), _O_WRONLY | _O_CREAT | _O_EXCL | _O_BINARY, _S_IREAD | _S_IWRITE);
    if (fd < 0) system_failure("create snapshot");
    FILE* file = _fdopen(fd, "wb");
    if (!file) { const int saved = errno; _close(fd); errno = saved; system_failure("open snapshot stream"); }
#else
    const int fd = ::open(path.c_str(), O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC, 0600);
    if (fd < 0) system_failure("create snapshot");
    FILE* file = fdopen(fd, "wb");
    if (!file) { const int saved = errno; ::close(fd); errno = saved; system_failure("open snapshot stream"); }
#endif
    return file;
}
void sync_file(FILE* file) {
    if (std::fflush(file) != 0) system_failure("flush snapshot");
#ifdef _WIN32
    if (_commit(_fileno(file)) != 0) system_failure("sync snapshot");
#else
    if (fsync(fileno(file)) != 0) system_failure("sync snapshot");
#endif
}
void publish(const fs::path& temporary, const fs::path& destination) {
#ifdef _WIN32
    if (!MoveFileExW(temporary.c_str(), destination.c_str(), MOVEFILE_WRITE_THROUGH))
        throw std::system_error(GetLastError(), std::system_category(), "publish snapshot");
#else
    if (::rename(temporary.c_str(), destination.c_str()) != 0) system_failure("publish snapshot");
#endif
}
void sync_directory(const fs::path& directory) {
#ifndef _WIN32
    const int fd = ::open(directory.c_str(), O_RDONLY | O_DIRECTORY | O_CLOEXEC);
    if (fd < 0) system_failure("open snapshot directory for sync");
    const int result = fsync(fd), saved = errno;
    ::close(fd);
    if (result != 0) { errno = saved; system_failure("sync snapshot directory"); }
#else
    (void) directory; // MoveFileExW above requests write-through publication.
#endif
}
} // namespace

struct ConversationStore::Impl {
    fs::path directory;
    ConversationIdentity identity;
    uint64_t budget;
    size_t slots;
    ConversationIoProgress progress;
#ifdef _WIN32
    HANDLE lock = INVALID_HANDLE_VALUE;
#else
    int lock = -1;
#endif
    Impl(fs::path path, const ConversationIdentity& id, uint64_t bytes, size_t count, ConversationIoProgress heartbeat)
        : directory(std::move(path)), identity(id), budget(bytes), slots(count), progress(heartbeat) {}
    ~Impl() {
#ifdef _WIN32
        if (lock != INVALID_HANDLE_VALUE) CloseHandle(lock);
#else
        if (lock >= 0) ::close(lock);
#endif
    }
    void acquire() {
        const auto path = directory / ".lock";
        if (fs::is_symlink(fs::symlink_status(path))) throw std::runtime_error("snapshot lock is a symlink");
#ifdef _WIN32
        lock = CreateFileW(path.c_str(), GENERIC_READ | GENERIC_WRITE, 0, nullptr, OPEN_ALWAYS,
                           FILE_ATTRIBUTE_NORMAL, nullptr);
        if (lock == INVALID_HANDLE_VALUE)
            throw std::system_error(GetLastError(), std::system_category(), "lock snapshot directory");
#else
        lock = ::open(path.c_str(), O_RDWR | O_CREAT | O_CLOEXEC | O_NOFOLLOW, 0600);
        if (lock < 0) system_failure("open snapshot lock");
        if (flock(lock, LOCK_EX | LOCK_NB) != 0) system_failure("lock snapshot directory");
#endif
    }
    void prune(uint64_t incoming, bool new_entry, const fs::path& protected_path = {}) {
        // Keep only aggregate counters and one victim while scanning; opening an
        // old cache with a larger entry limit does not allocate an unbounded index.
        for (;;) {
            uint64_t total = 0;
            size_t count = 0;
            fs::path oldest;
            fs::file_time_type oldest_time{};
            for (const auto& entry : fs::directory_iterator(directory)) {
                if (progress) progress();
                if ((!managed_name(entry.path(), ".snap") && !managed_name(entry.path(), ".tmp")) ||
                    !fs::is_regular_file(entry.symlink_status())) continue;
                const uint64_t n = entry.file_size();
                if (n > UINT64_MAX - total) throw std::runtime_error("snapshot directory byte count overflow");
                total += n;
                if (count == SIZE_MAX) throw std::runtime_error("snapshot directory entry count overflow");
                ++count;
                if (entry.path() == protected_path) continue;
                const auto modified = entry.last_write_time();
                if (oldest.empty() || modified < oldest_time || (modified == oldest_time && entry.path() < oldest)) {
                    oldest = entry.path(); oldest_time = modified;
                }
            }
            if (total <= budget - incoming && count <= slots - size_t(new_entry)) return;
            if (oldest.empty() || !fs::remove(oldest)) throw std::runtime_error("cannot evict disk snapshot");
        }
    }
    bool owns(const Candidate& candidate) const {
        return candidate.path.parent_path() == directory && managed_name(candidate.path, ".snap") &&
               fs::is_regular_file(fs::symlink_status(candidate.path));
    }
};

ConversationStore::ConversationStore() = default;
ConversationStore::~ConversationStore() = default;
bool ConversationStore::is_open() const { return bool(impl_); }
void ConversationStore::close() { impl_.reset(); }

bool ConversationStore::open(const fs::path& root, const ConversationIdentity& identity,
                             uint64_t bytes, size_t slots, std::string& error, ConversationIoProgress progress) {
    close();
    if (!bytes || !slots) return true;
    try {
        if (root.empty()) throw std::runtime_error("snapshot directory is empty");
        const auto directory = fs::absolute(root / "strata-conversations-v2").lexically_normal();
        fs::create_directories(directory.parent_path());
        if (fs::create_directory(directory))
            fs::permissions(directory, fs::perms::owner_all, fs::perm_options::replace);
        if (!fs::is_directory(fs::symlink_status(directory))) throw std::runtime_error("snapshot directory is not a regular directory");
#ifndef _WIN32
        const auto access = fs::status(directory).permissions();
        if ((access & (fs::perms::group_all | fs::perms::others_all)) != fs::perms::none)
            throw std::runtime_error("snapshot directory must have owner-only permissions");
#endif
        auto store = std::make_unique<Impl>(directory, identity, bytes, slots, progress);
        store->acquire();
        for (const auto& entry : fs::directory_iterator(directory)) {
            if (managed_name(entry.path(), ".tmp") && fs::is_regular_file(entry.symlink_status()) && !fs::remove(entry.path()))
                throw std::runtime_error("cannot remove interrupted snapshot");
        }
        store->prune(0, false);
        impl_ = std::move(store);
        return true;
    } catch (const std::exception& e) { error = e.what(); return false; }
}

bool ConversationStore::encoded_size(const core::SavedConversation& image, uint64_t& bytes, std::string& error) const {
    if (!impl_) { error = "snapshot store is closed"; return false; }
    return conversation_file_size(image, impl_->identity, bytes, error);
}

bool ConversationStore::put(const core::SavedConversation& image, std::string& error, const Candidate* protected_entry) {
    if (!impl_) { error = "disk conversation cache is disabled"; return false; }
    try {
        uint64_t bytes = 0;
        if (!encoded_size(image, bytes, error)) return false;
        if (bytes > impl_->budget) { error = "snapshot exceeds disk quota"; return false; }
        if (protected_entry && !impl_->owns(*protected_entry))
            throw std::runtime_error("protected snapshot candidate is unavailable");
        if (protected_entry && (impl_->slots < 2 || fs::file_size(protected_entry->path) > impl_->budget - bytes))
            throw std::runtime_error("disk quota cannot hold protected and incoming snapshots");
        const auto name = unique_name();
        const auto target = impl_->directory / (name + ".snap");
        if (fs::exists(target)) throw std::runtime_error("snapshot name collision");
        impl_->prune(bytes, true, protected_entry ? protected_entry->path : fs::path{});
        RemoveFile temporary{impl_->directory / (name + ".tmp")};
        auto close_file = [](FILE* f) { std::fclose(f); };
        std::unique_ptr<FILE, decltype(close_file)> file(create_file(temporary.path));
        temporary.armed = true;
        FileBuffer buffer(file.get(), bytes);
        std::ostream stream(&buffer);
        if (!conversation_file_write(stream, image, impl_->identity, error, impl_->progress)) return false;
        if (buffer.remaining()) throw std::runtime_error("snapshot encoded length differs from reservation");
        sync_file(file.get());
        if (std::fclose(file.release()) != 0) system_failure("close snapshot");
        publish(temporary.path, target);
        temporary.path = target; // remove even a published file if directory sync fails
        sync_directory(impl_->directory);
        temporary.path.clear();
        return true;
    } catch (const std::exception& e) { error = e.what(); return false; }
}

bool ConversationStore::best(const std::vector<int64_t>& prompt, const std::vector<core::ConversationImageKey>& images,
                             bool cvec, uint64_t staging_limit, const std::vector<fs::path>& excluded,
                             Candidate& candidate, std::string& error) const {
    Candidate found;
    if (!impl_) { candidate = {}; return true; }
    try {
        fs::file_time_type newest{};
        for (const auto& entry : fs::directory_iterator(impl_->directory)) {
            if (!managed_name(entry.path(), ".snap") || !fs::is_regular_file(entry.symlink_status()) ||
                std::find(excluded.begin(), excluded.end(), entry.path()) != excluded.end()) continue;
            std::ifstream file(entry.path(), std::ios::binary);
            ConversationFileMatch match;
            std::string ignored;
            const bool compatible = conversation_file_match(file, impl_->identity, staging_limit, prompt, images,
                                                             cvec, match, ignored, impl_->progress);
            if (impl_->progress) impl_->progress();
            if (!compatible && ignored.rfind("snapshot identity differs:", 0) == 0)
                std::fprintf(stderr, "strata serve: disk cache: %s; re-reading prompt if no compatible entry exists\n", ignored.c_str());
            if (!compatible || match.tokens == 0) continue;
            const auto modified = entry.last_write_time();
            if (match.tokens > found.match.tokens || (match.tokens == found.match.tokens &&
                (modified > newest || (modified == newest && entry.path() > found.path)))) {
                found = {entry.path(), match}; newest = modified;
            }
        }
        candidate = std::move(found);
        return true;
    } catch (const std::exception& e) { error = e.what(); return false; }
}

bool ConversationStore::read(const Candidate& candidate, uint64_t staging_limit, std::optional<uint64_t> available,
                             uint64_t floor, core::SavedConversation& image, std::string& error) const {
    try {
        if (!impl_ || !impl_->owns(candidate)) throw std::runtime_error("snapshot candidate is unavailable");
        std::ifstream file(candidate.path, std::ios::binary);
        return conversation_file_read(file, impl_->identity, staging_limit, available, floor, image, error, impl_->progress);
    } catch (const std::exception& e) { error = e.what(); return false; }
}

bool ConversationStore::touch(const Candidate& candidate, std::string& error) {
    try {
        if (!impl_ || !impl_->owns(candidate)) throw std::runtime_error("snapshot candidate is unavailable");
        fs::last_write_time(candidate.path, fs::file_time_type::clock::now());
        return true;
    } catch (const std::exception& e) { error = e.what(); return false; }
}
} // namespace strata::platform
