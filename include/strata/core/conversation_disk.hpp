// --conversation-disk-dir: parked conversations live on disk instead of in host RAM, keyed by the id a client
// sends with its request (`conv=` on the GEN line). A request without an id is never parked: side calls (memory
// extraction, titles) evict the live conversation but are not worth a write themselves.
//
// One directory per id:
//   kv-E-B.bin  K/V buffer B of entry E (conversation_kv_entries' order). A later park of a conversation that was
//               restored from these files rewrites only what changed since: the pages from the first rewritten
//               token on (conversation_kv_kept), plus the small running state;
//   live.bin    the live running state, one part per stage, rewritten on every park;
//   ck-N.bin    the prompt checkpoint at N tokens, written once: its tokens are a prefix of the live ones, so it
//               cannot change while the K/V before N is unchanged.
// The index (token ids, extents, sizes) stays in RAM and the directory is wiped at startup: an image only ever
// restores into the engine that wrote it. Every write and read goes through the page cache in bounded chunks that
// are synced and dropped at once, so parking never grows the page cache and never pushes experts out of it.
#pragma once

#include "strata/core/conversation_snapshot.hpp"

#include <cstdint>
#include <map>
#include <string>
#include <vector>

namespace strata::core {

class ConversationDisk {
public:
    struct Match {
        int64_t tokens = 0;
        bool live = false;
    };
    struct Stats {
        uint64_t written = 0, kept = 0, on_disk = 0;
        size_t evicted = 0;
    };

    ConversationDisk() = default;
    ConversationDisk(const ConversationDisk&) = delete;
    ConversationDisk& operator=(const ConversationDisk&) = delete;
    ~ConversationDisk();

    // Creates `dir` if needed and removes the conversation directories a previous run left there.
    bool open(const std::string& dir, uint64_t budget_bytes, std::string& error);
    bool enabled() const { return !dir_.empty(); }
    size_t size() const { return entries_.size(); }

    // The deepest point of conversation `id` this prompt can resume from: the live end or a checkpoint.
    template<class Token>
    Match best(const std::string& id, const std::vector<Token>& prompt, const std::vector<ConversationImageKey>& images,
               bool cvec) const {
        Match best;
        const auto it = entries_.find(id);
        if (it == entries_.end() || it->second.cvec != cvec) return best;
        const Entry& e = it->second;
        auto consider = [&](size_t n, bool live) {
            if (n == 0 || n >= prompt.size() || (int64_t) n <= best.tokens) return;
            for (size_t i = 0; i < n; ++i)
                if ((int64_t) prompt[i] != (int64_t) e.ids[i]) return;
            size_t j = 0;
            for (const auto& image : images) {
                if (image.start >= (int64_t) n) break;
                if (j == e.imgs.size() || !(e.imgs[j++] == image)) return;
            }
            if (j < e.imgs.size() && e.imgs[j].start < (int64_t) n) return;
            best = {(int64_t) n, live};
        };
        consider(e.ids.size(), true);
        for (int64_t n : e.checkpoints) consider((size_t) n, false);
        return best;
    }

    // Writes the live session as conversation `id`. `unchanged` is how many leading tokens of the session's K/V
    // are still byte-identical to id's files (0 = write everything). The session is only read. On failure the
    // conversation is dropped from disk and the session is untouched.
    bool park(const std::string& id, const ConversationView& view, const ConversationStages& stages,
              const ModelGeometry& g, const QsaState& draft, int64_t unchanged, Stats& stats, std::string& error);

    // Reads id's running state and checkpoints and validates them, and the K/V extents, against the stages.
    // No session writes: a failure here leaves the outgoing conversation intact.
    bool load(const std::string& id, const ConversationStages& stages, const ModelGeometry& g, const QsaState& draft,
              SavedConversation& image, std::string& error);

    // Copies id's K/V and the loaded live state into the session. transfer_failed leaves the session partly
    // written: the caller must read the prompt from token 0.
    ConversationRestore restore(const std::string& id, const SavedConversation& image, const ConversationStages& stages,
                                const ModelGeometry& g, const QsaState& draft, uint64_t& bytes, std::string& error);

    void drop(const std::string& id);

private:
    struct Entry {
        std::vector<int32_t> ids;
        std::vector<ConversationImageKey> imgs;
        std::vector<int64_t> checkpoints;   // token counts of the ck-N.bin files
        bool cvec = true;
        std::array<int64_t, 18> geometry{};
        int64_t layer_lo = 0, layer_hi = 0;
        std::vector<ConversationKvExtent> kv;
        uint64_t used = 0, bytes = 0;
    };
    std::string path(const std::string& id) const;
    bool staging(std::string& error);
    void evict(const std::string& keep, Stats& stats);

    std::string dir_;
    uint64_t budget_ = 0, clock_ = 0;
    std::map<std::string, Entry> entries_;
    void* bounce_ = nullptr;   // pinned staging buffer for copies between the session's buffers and the files
};

} // namespace strata::core
