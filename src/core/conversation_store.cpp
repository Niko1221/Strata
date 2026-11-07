// The two ConversationStore backends this build has: the local spill directory (the default, byte-identical to what
// the cache wrote before the store existed) and a memory store for tests and for a cache whose files need no folder.
#include "strata/core/conversation_store.hpp"

#include <atomic>
#include <chrono>
#include <cstdio>
#include <fstream>
#include <iterator>
#include <map>
#include <new>
#include <set>
#include <system_error>

#if !defined(_WIN32)
#include <fcntl.h>
#include <unistd.h>
#endif

namespace strata::core {
namespace {

// A name the store may map to a file: flat, and never a directory entry of its own ('.' , '..') or a path.
bool store_name_ok(const std::string& name) {
    if (name.empty() || name == "." || name == "..") return false;
    if (name.find('\0') != std::string::npos) return false;
    return name.find('/') == std::string::npos && name.find('\\') == std::string::npos;
}

// What is at one path: absent, present but not a regular file, or a regular file with its size.  symlink_status is
// deliberate - a store artifact must be a regular file, and accounting must never follow a link to somebody else's
// file.
struct Inspection {
    bool inspected = true;
    bool exists = false;
    bool regular = false;
    uint64_t bytes = 0;
};

Inspection inspect_local(const std::filesystem::path& path) {
    Inspection info;
    std::error_code ec;
    const auto status = std::filesystem::symlink_status(path, ec);
    if (ec == std::errc::no_such_file_or_directory || ec == std::errc::not_a_directory) {
        ec.clear();
        return info;
    }
    if (ec) { info.inspected = false; return info; }
    if (status.type() == std::filesystem::file_type::not_found) return info;
    info.exists = true;
    if (!std::filesystem::is_regular_file(status)) return info;
    info.regular = true;
    info.bytes = std::filesystem::file_size(path, ec);
    if (ec) { info.inspected = false; info.regular = false; info.bytes = 0; }
    return info;
}

// A unique, never-empty token for a staging name; two stores in one process and two processes do not collide.
std::string unique_token() {
    static std::atomic<uint64_t> serial{0};
    const uint64_t now = (uint64_t) std::chrono::steady_clock::now().time_since_epoch().count();
    const uint64_t n = serial.fetch_add(1, std::memory_order_relaxed) + 1;
    char b[64];
    std::snprintf(b, sizeof b, "%llx-%llx", (unsigned long long) now, (unsigned long long) n);
    return b;
}

bool write_bytes(const std::filesystem::path& path, const std::vector<uint8_t>& bytes, std::string& error) {
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    if (!out) { error = "cannot create " + path.string(); return false; }
    if (!bytes.empty()) out.write(reinterpret_cast<const char*>(bytes.data()), (std::streamsize) bytes.size());
    out.close();
    if (!out) { error = "cannot write " + path.string(); return false; }
    return true;
}

} // namespace

namespace {

// ---- the local spill directory: the default backend ----
class LocalConversationStore final : public ConversationStore {
public:
    explicit LocalConversationStore(std::filesystem::path directory) : directory_(std::move(directory)) {}

    bool list(std::vector<ConversationStoreFile>& files, std::string& error) override {
        files.clear();
        error.clear();
        std::error_code ec;
        for (std::filesystem::directory_iterator it(directory_, ec), end; !ec && it != end; it.increment(ec)) {
            const std::string name = it->path().filename().string();
            if (name.empty() || name[0] == '.') continue;   // the store's own temporaries and hidden names
            const Inspection info = inspect_local(it->path());
            if (!info.inspected) { error = "cannot inspect " + it->path().string(); return false; }
            if (!info.regular) continue;                     // a directory or a link is not a store file
            try {
                files.push_back(ConversationStoreFile{name, info.bytes});
            } catch (const std::bad_alloc&) {
                files.clear();
                error = "not enough RAM to list the conversation store";
                return false;
            }
        }
        if (ec) { files.clear(); error = "cannot scan " + directory_.string(); return false; }
        return true;
    }

    bool stat(const std::string& name, uint64_t& bytes, std::string& error) override {
        bytes = 0;
        error.clear();
        if (!store_name_ok(name)) { error = "invalid conversation store name: " + name; return false; }
        const std::filesystem::path path = directory_ / name;
        const Inspection info = inspect_local(path);
        if (!info.inspected) { error = "cannot inspect " + path.string(); return false; }
        if (!info.exists) return false;
        if (!info.regular) { error = "not a regular file: " + path.string(); return false; }
        bytes = info.bytes;
        return true;
    }

    bool materialize(const std::string& name, std::string& local_path, std::string& error) override {
        error.clear();
        uint64_t bytes = 0;
        if (!stat(name, bytes, error)) {
            if (error.empty()) error = "no such conversation store file: " + name;
            return false;
        }
        local_path = (directory_ / name).string();   // the file itself: no copy, nothing to release
        return true;
    }

    void release(const std::string& local_path) override { (void) local_path; }

    bool stage(const std::string& name, std::string& local_path, std::string& error) override {
        error.clear();
        if (!store_name_ok(name)) { error = "invalid conversation store name: " + name; return false; }
        local_path = (directory_ / name).string();   // the writer writes the name the cache always wrote
        fresh_.insert(local_path);                   // only what stage() handed out may abort() remove
        return true;
    }

    bool stage_existing(const std::string& name, std::string& local_path, std::string& error) override {
        uint64_t bytes = 0;
        if (!stat(name, bytes, error)) return false;   // absent (empty error) or not a regular file
        local_path = (directory_ / name).string();     // the append lands in the store itself, as before
        return true;
    }

    bool publish(const std::string& name, const std::string& local_path, uint64_t bytes,
                 std::string& error) override {
        error.clear();
        (void) bytes;
        if (!store_name_ok(name)) { error = "invalid conversation store name: " + name; return false; }
        const std::filesystem::path target = directory_ / name;
        if (local_path != target.string()) {
            // A temporary beside the target (a session file's or a sidecar's "<name>.tmp", both written by the cache's
            // own writers): rename it into place and flush the folder, as the writers used to do themselves.
            const Inspection staged = inspect_local(local_path);
            if (!staged.inspected || !staged.regular) {
                error = "the staged conversation store file is missing or not a regular file: " + local_path;
                return false;
            }
            std::error_code rename_error;
            std::filesystem::rename(local_path, target, rename_error);
            if (rename_error) { error = "cannot publish " + name + ": " + rename_error.message(); return false; }
            flush_directory();
            // stage() registered the published name, not the writer's temporary that was just renamed onto it
            fresh_.erase(target.string());
        }
        const Inspection info = inspect_local(target);
        if (!info.inspected || !info.regular) {
            error = "the published conversation store file is missing or not a regular file: " + name;
            return false;
        }
        fresh_.erase(local_path);
        return true;
    }

    void abort(const std::string& local_path) override {
        // A stage() target is ours to discard, and so is the cache writer's own "<path>.tmp" beside it.  A path from
        // stage_existing() is a file the store already held: a failed append leaves it exactly as it was, which is
        // what the chain manifest's committed length expects.  Only a regular file is removed, so a link or a
        // directory left at a temporary's name stays for the operator.
        if (fresh_.erase(local_path) != 0) {
            remove_regular(local_path);
            remove_regular(local_path + ".tmp");
            return;
        }
        const std::filesystem::path path(local_path);
        if (path.extension() != ".tmp") return;
        if (path.parent_path().lexically_normal() != directory_.lexically_normal()) return;
        remove_regular(local_path);
    }

    bool remove(const std::string& name, std::string& error) override {
        error.clear();
        if (!store_name_ok(name)) { error = "invalid conversation store name: " + name; return false; }
        const std::filesystem::path path = directory_ / name;
        const Inspection info = inspect_local(path);
        if (!info.inspected) { error = "cannot inspect " + path.string(); return false; }
        if (!info.exists) return true;
        // The name itself is removed: a link is unlinked and its target never touched, a regular file is deleted, and a
        // directory that still holds something is refused (the caller retains and reports it).
        std::error_code ec;
        std::filesystem::remove(path, ec);
        if (ec) { error = "cannot remove " + path.string() + ": " + ec.message(); return false; }
        return true;
    }

    std::string describe() const override { return "local " + directory_.string(); }
    const std::filesystem::path& local_directory() const { return directory_; }

private:
    // POSIX has no folder flush of its own: the rename's durability comes from flushing the directory, which the
    // cache's writers have always done for a session file.  Best effort - the file is in place either way.
    void flush_directory() const {
#if !defined(_WIN32)
        const int fd = ::open(directory_.c_str(), O_RDONLY | O_CLOEXEC);
        if (fd < 0) return;
        ::fsync(fd);
        ::close(fd);
#endif
    }

    static void remove_regular(const std::string& path) {
        const Inspection info = inspect_local(path);
        if (!info.inspected || !info.exists || !info.regular) return;
        std::error_code ec;
        std::filesystem::remove(path, ec);
    }

    std::filesystem::path directory_;
    std::set<std::string> fresh_;   // the paths stage() handed out; abort() removes only these
};

// ---- the memory store: files in RAM, staged through a private temporary directory ----
class MemoryConversationStore final : public ConversationStore {
public:
    MemoryConversationStore() : root_(std::filesystem::temp_directory_path() / ("strata-conv-store-" + unique_token())) {}
    ~MemoryConversationStore() override {
        std::error_code ec;
        std::filesystem::remove_all(root_, ec);
    }

    bool list(std::vector<ConversationStoreFile>& files, std::string& error) override {
        files.clear();
        error.clear();
        try {
            for (const auto& entry : files_) files.push_back(ConversationStoreFile{entry.first, (uint64_t) entry.second.size()});
        } catch (const std::bad_alloc&) {
            files.clear();
            error = "not enough RAM to list the conversation store";
            return false;
        }
        return true;
    }

    bool stat(const std::string& name, uint64_t& bytes, std::string& error) override {
        error.clear();
        bytes = 0;
        const auto it = files_.find(name);
        if (it == files_.end()) return false;
        bytes = (uint64_t) it->second.size();
        return true;
    }

    bool materialize(const std::string& name, std::string& local_path, std::string& error) override {
        error.clear();
        const auto it = files_.find(name);
        if (it == files_.end()) { error = "no such conversation store file: " + name; return false; }
        std::string path;
        if (!staging_path(name, path, error)) return false;
        if (!write_bytes(path, it->second, error)) { discard(path); return false; }
        local_path = path;
        return true;
    }

    void release(const std::string& local_path) override { discard(local_path); }

    bool stage(const std::string& name, std::string& local_path, std::string& error) override {
        error.clear();
        if (!store_name_ok(name)) { error = "invalid conversation store name: " + name; return false; }
        return staging_path(name, local_path, error);
    }

    bool stage_existing(const std::string& name, std::string& local_path, std::string& error) override {
        const auto it = files_.find(name);
        if (it == files_.end()) { error.clear(); return false; }
        return materialize(name, local_path, error);   // a copy: the append must not touch the stored bytes until
                                                       // publish()
    }

    bool publish(const std::string& name, const std::string& local_path, uint64_t bytes,
                 std::string& error) override {
        error.clear();
        (void) bytes;   // the staged file is the authority; the writer's report is the cache's own accounting
        std::ifstream in(local_path, std::ios::binary);
        if (!in) { error = "cannot read the staged conversation store file: " + local_path; return false; }
        std::vector<uint8_t> data((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
        if (in.bad()) { error = "cannot read the staged conversation store file: " + local_path; return false; }
        try {
            files_[name] = std::move(data);
        } catch (const std::bad_alloc&) {
            error = "not enough RAM to publish " + name;
            return false;
        }
        discard(local_path);
        return true;
    }

    void abort(const std::string& local_path) override { discard(local_path); }

    bool remove(const std::string& name, std::string& error) override {
        error.clear();
        files_.erase(name);
        return true;
    }

    std::string describe() const override { return "memory"; }

private:
    // A unique staging path whose BASENAME is the store name, so a caller can tell which name a staged path
    // belongs to.  The file does not exist yet: a writer creates it.
    bool staging_path(const std::string& name, std::string& out, std::string& error) {
        error.clear();
        if (!store_name_ok(name)) { error = "invalid conversation store name: " + name; return false; }
        const std::filesystem::path directory = root_ / std::to_string(++serial_);
        std::error_code ec;
        std::filesystem::create_directories(directory, ec);
        if (ec) { error = "cannot create the conversation store's staging area: " + ec.message(); return false; }
        out = (directory / name).string();
        return true;
    }
    // Removes a staging file and its private directory.  A path outside this store's staging area is left alone: the
    // store never deletes anything it did not create.
    void discard(const std::string& local_path) {
        if (local_path.empty()) return;
        const std::filesystem::path path(local_path);
        if (path.parent_path().parent_path() != root_) return;
        std::error_code ec;
        std::filesystem::remove(path, ec);
        ec.clear();
        std::filesystem::remove(local_path + ".tmp", ec);   // the writer's own temporary, when it left one
        ec.clear();
        std::filesystem::remove(path.parent_path(), ec);
    }

    std::map<std::string, std::vector<uint8_t>> files_;
    std::filesystem::path root_;
    uint64_t serial_ = 0;
};

} // namespace

std::shared_ptr<ConversationStore> make_local_conversation_store(std::filesystem::path directory) {
    return std::make_shared<LocalConversationStore>(std::move(directory));
}

std::shared_ptr<ConversationStore> make_memory_conversation_store() {
    return std::make_shared<MemoryConversationStore>();
}

std::filesystem::path local_conversation_store_directory(const ConversationStore& store) {
    const auto* local = dynamic_cast<const LocalConversationStore*>(&store);
    return local ? local->local_directory() : std::filesystem::path();
}

} // namespace strata::core
