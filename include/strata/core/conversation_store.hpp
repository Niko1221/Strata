// Where a conversation-spill entry's files live.  The cache keeps its whole format - the session file, the sidecar,
// the chain manifest, the names, the identity rules and the budget accounting - and only stops touching the
// filesystem itself, so the same logic can run over another backend later.  Names are flat ("strata-conv-5.sess"):
// a backend maps a name to whatever it uses.
//
// This is host code only (no CUDA), like the spill cache that uses it.
#pragma once

#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>

namespace strata::core {

struct ConversationStoreFile {
    std::string name;
    uint64_t bytes = 0;
};

class ConversationStore {
public:
    virtual ~ConversationStore() = default;
    // Every regular file of the store with its size.  A directory, a link, and a temporary the store keeps for itself
    // (a name beginning with '.') are not listed: only names a cache artifact can carry come back.
    virtual bool list(std::vector<ConversationStoreFile>& files, std::string& error) = 0;
    // True: 'name' is a regular file and 'bytes' is its size.  False with an empty 'error': nothing is stored under
    // that name.  False with 'error' set: something is there but it is not a regular file, or it could not be
    // inspected - a caller that must not reuse a name treats both cases as taken.
    virtual bool stat(const std::string& name, uint64_t& bytes, std::string& error) = 0;
    // A local path the existing path-based readers can open.  The local store hands out its own file; a store that
    // cannot (memory, a network tier) writes a temporary, and release() disposes of it.
    virtual bool materialize(const std::string& name, std::string& local_path, std::string& error) = 0;
    virtual void release(const std::string& local_path) = 0;
    // A local path the existing path-based writers can fill.  The local store hands out the file's own path, so a
    // writer keeps writing the name it always wrote and publish() only verifies and flushes the folder; a store that
    // cannot stages a temporary under the same basename instead, and publish() takes it.
    virtual bool stage(const std::string& name, std::string& local_path, std::string& error) = 0;
    // stage() for a target that already holds bytes (a delta append): the local store hands out the file itself, so
    // the append lands in the store exactly as it did before the store existed (the chain manifest's committed
    // length already makes a torn append harmless), while a copying store pre-populates a temporary.  False when the
    // name is absent.
    virtual bool stage_existing(const std::string& name, std::string& local_path, std::string& error) = 0;
    // Publishes the bytes at 'local_path' (a path from stage()/stage_existing(), filled by the writer) under 'name';
    // 'bytes' is the size the writer reported.  A non-regular staged file is refused.  False leaves 'name' unchanged.
    virtual bool publish(const std::string& name, const std::string& local_path, uint64_t bytes,
                         std::string& error) = 0;
    // Discards a write that will not be published: the path stage() handed out, or the cache writer's own
    // "<path>.tmp" beside it.  Never a file the store already held (a stage_existing() path), and only a regular
    // file - a link or a directory left at a temporary's name stays for the operator.
    virtual void abort(const std::string& local_path) = 0;
    // Removes 'name'.  True when it is gone (also when it was never there).  A link is unlinked and its target is
    // never followed or touched; a directory that still holds something is refused.
    virtual bool remove(const std::string& name, std::string& error) = 0;
    virtual std::string describe() const = 0;
};

// The spill cache's default backend: flat file names in 'directory', written and read with the same calls (and the
// same bytes) as before the store existed.
std::shared_ptr<ConversationStore> make_local_conversation_store(std::filesystem::path directory);
// A backend that keeps every file in RAM and stages through a private temporary directory of its own; for tests and
// for a cache whose files are not worth a folder.
std::shared_ptr<ConversationStore> make_memory_conversation_store();

// The directory a store made by make_local_conversation_store() serves, or an empty path for any other store.  The
// cache uses it for the three things only a real directory can do: create it, wipe a crashed writer's temporaries,
// and read the recompute gate's ledger.
std::filesystem::path local_conversation_store_directory(const ConversationStore& store);

} // namespace strata::core
