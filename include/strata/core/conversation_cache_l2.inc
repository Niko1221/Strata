// L2 RAM PARKED control-plane extension.
// Included from ConversationCache's private section by conversation_cache.hpp.
// The gateway can inspect lifecycle metadata and change pin/order, but never
// receives or owns KV memory. All methods run in the engine command thread.
public:
    struct EntryInfo {
        uint64_t serial = 0;
        size_t tokens = 0;
        size_t bytes = 0;
        size_t checkpoints = 0;
        bool pinned = false;
        bool guarded = false;
        bool cvec = true;
        std::string source;
    };

    struct Stats {
        bool enabled = false;
        size_t budget_bytes = 0;
        size_t slots = 0;
        size_t entries = 0;
        size_t bytes = 0;
        size_t evictions = 0;
        size_t retained_bytes = 0;
        size_t pinned = 0;
        size_t guarded = 0;
    };

    Stats l2_stats() const {
        Stats s;
        s.enabled = enabled();
        s.budget_bytes = budget_;
        s.slots = slots_;
        s.entries = entries_.size();
        s.bytes = bytes();
        s.evictions = evictions_;
        s.retained_bytes = reuse_.bytes();
        for (size_t i = 0; i < entries_.size(); ++i) {
            if (l2_pinned(serials_[i])) ++s.pinned;
            if (guarded_ != 0 && serials_[i] == guarded_) ++s.guarded;
        }
        return s;
    }

    std::vector<EntryInfo> l2_list() const {
        std::vector<EntryInfo> out;
        out.reserve(entries_.size());
        for (size_t i = 0; i < entries_.size(); ++i) {
            EntryInfo x;
            x.serial = serials_[i];
            x.tokens = entries_[i].live.ids.size();
            x.bytes = entries_[i].bytes();
            x.checkpoints = entries_[i].checkpoints.size();
            x.pinned = l2_pinned(serials_[i]);
            x.guarded = guarded_ != 0 && serials_[i] == guarded_;
            x.cvec = entries_[i].cvec;
            x.source = l2_source(serials_[i]);
            out.push_back(std::move(x));
        }
        return out;
    }

    uint64_t l2_last_serial() const { return serials_.empty() ? 0 : serials_.back(); }

    void l2_set_source(uint64_t serial, std::string source) {
        if (serial == 0) return;
        for (auto& p : l2_sources_) {
            if (p.first == serial) { p.second = std::move(source); return; }
        }
        l2_sources_.push_back({serial, std::move(source)});
    }

    bool l2_set_pinned(uint64_t serial, bool pinned) {
        if (index_of(serial) == SIZE_MAX) return false;
        auto it = std::find(l2_pins_.begin(), l2_pins_.end(), serial);
        if (pinned && it == l2_pins_.end()) l2_pins_.push_back(serial);
        if (!pinned && it != l2_pins_.end()) l2_pins_.erase(it);
        return true;
    }

    bool l2_drop(uint64_t serial) {
        const size_t i = index_of(serial);
        if (i == SIZE_MAX) return false;
        if (guarded_ != 0 && serials_[i] == guarded_) return false;
        bytes_ -= entries_[i].bytes();
        erase(i);
        return true;
    }

    bool l2_touch(uint64_t serial) {
        const size_t i = index_of(serial);
        if (i == SIZE_MAX || i + 1 == entries_.size()) return i != SIZE_MAX;
        SavedConversation e = std::move(entries_[i]);
        const uint64_t s = serials_[i];
        entries_.erase(entries_.begin() + (std::ptrdiff_t) i);
        serials_.erase(serials_.begin() + (std::ptrdiff_t) i);
        entries_.push_back(std::move(e));
        serials_.push_back(s);
        return true;
    }

    void l2_clear_metadata() {
        l2_sources_.clear();
        l2_pins_.clear();
    }

private:
    bool l2_pinned(uint64_t serial) const {
        return std::find(l2_pins_.begin(), l2_pins_.end(), serial) != l2_pins_.end();
    }
    std::string l2_source(uint64_t serial) const {
        for (const auto& p : l2_sources_)
            if (p.first == serial) return p.second;
        return {};
    }
    std::vector<std::pair<uint64_t, std::string>> l2_sources_;
    std::vector<uint64_t> l2_pins_;
