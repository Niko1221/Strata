// Persistent prefix snapshots (docs/DETAILS.md): the state at the end of a client's system prompt, so a new
// conversation that starts with it reads only the rest.  One file per prefix on disk; the most recently used
// ones also as RAM images.  When RAM is short, saving and restoring stream between the session and the file
// through one bounce buffer.  Single session (no layer split), not thread safe.
#pragma once

#include "strata/core/conversation_snapshot.hpp"

#include <algorithm>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace strata::core {

class PrefixStore {
public:
    struct Options {
        std::string dir;        // empty: disabled
        std::string identity;   // engine build and model; a file with another one is deleted
        size_t ram_budget = 0, disk_budget = 0, min_free = 0;
        size_t bounce = 64u << 20;
    };
    struct Match {
        uint64_t key = 0;
        int64_t tokens = 0;
    };

    // Scans the directory: files of another identity, unreadable or partial ones are deleted.
    bool open(const Options& options, std::string& log, std::string& error);
    bool enabled() const { return !o_.dir.empty(); }
    // the longest prefix of `prompt` (shorter than it: its last token is always read)
    template<class Token> Match best(const std::vector<Token>& prompt, bool cvec) const {
        Match m;
        for (const auto& e : entries_)
            if (e.cvec == cvec && (int64_t) e.ids.size() > m.tokens && e.ids.size() < prompt.size() &&
                std::equal(e.ids.begin(), e.ids.end(), prompt.begin()))
                m = {e.key, (int64_t) e.ids.size()};
        return m;
    }
    bool contains(const std::vector<int32_t>& ids, bool cvec) const;
    // Saves the session's state at `root`, a checkpoint of what the session holds, to a new file.
    bool capture(const ConversationCheckpoint& root, bool cvec, const SessionState& session, const ModelGeometry& g,
                 const QsaState& draft, std::string& log, std::string& error);
    // `source`: "ram", "disk>ram" (read into the RAM tier first) or "disk" (streamed).  An invalid file is
    // deleted.  transfer_failed leaves the session partly written.
    ConversationRestore restore(uint64_t key, SessionState& session, const ModelGeometry& g, const QsaState& draft,
                                std::string& source, std::string& error);
    size_t size() const { return entries_.size(); }
    size_t ram_bytes() const;
    size_t disk_bytes() const;

private:
    struct Entry {
        uint64_t key = 0;
        std::vector<int32_t> ids;
        bool cvec = true;
        size_t file_bytes = 0;
        std::shared_ptr<SavedConversation> ram;
        size_t ram_bytes = 0;
        uint64_t used = 0;
    };
    std::string path(uint64_t key) const;
    Entry* find(uint64_t key);
    void erase(uint64_t key);
    void trim_disk(size_t incoming);
    bool trim_ram(size_t incoming);

    Options o_;
    std::vector<Entry> entries_;
    uint64_t clock_ = 0;
};

} // namespace strata::core
