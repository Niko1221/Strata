// include/strata/program/vram_ledger.hpp - one card's expert-cache lending ledger (no GPU), STRATA_VRAM_LEDGER.
//
// Every user of a card's expert cache other than the experts themselves declares what it needs - a FIXED address (baked
// into captured graphs), a RELOCATABLE one (laid out again at each use) or EXTERNAL room (VRAM given back to the driver
// for another program) - and the ledger decides WHERE, by the three rules the engine already follows:
//   - fixed regions stack from the head, in the order they are reserved at start; the head never moves, so neither do they;
//   - relocatable loans take the tail of the slots usable now (`live`); several may be out at once (two batch slots
//     reading prompts, a prompt loan beside a K/V growth, a small 8-12 GB card where everyone squeezes): each takes the
//     first gap from the tail that holds it, so the first loan is exactly today's, and a gap a released loan leaves is
//     reused; a sub-loan is carved inside its parent;
//   - external room comes off the end (shrink), only while no loan is out past the new end.
// The ledger only computes slot ranges and checks those rules; evicting, laying out and refilling stay where they are.
// The tail arithmetic is split_lend_slots (split_ring.hpp), the same function the prompt path's loan uses today, so a
// borrower moved onto the ledger gets exactly the slots it gets now.
#pragma once

#include "strata/program/split_ring.hpp"

#include <algorithm>
#include <cstdint>
#include <string>
#include <vector>

namespace strata::program::vram_ledger {

enum class Addr { Fixed, Relocatable, External };

struct Lease {
    int id = -1;
    std::string name;
    Addr addr = Addr::Relocatable;
    int parent = -1;          ///< a sub-loan: the lease it is carved from
    int64_t lo = 0, hi = 0;   ///< slot range [lo, hi)
    uint64_t bytes = 0;       ///< what was asked
    bool active = false;
};

class Card {
public:
    /// `offsets`: the cache's per-slot byte-offset prefix sum (slots + 1 entries) or empty for uniform `max_blob` slots.
    Card(int64_t slots, std::vector<uint64_t> offsets, uint64_t max_blob)
        : full_(slots), live_(slots), off_(std::move(offsets)), max_blob_(max_blob) {}

    int64_t full() const { return full_; }
    int64_t live() const { return live_; }
    /// The end of the fixed head regions (the first slot a loan may never go below).
    int64_t head_end() const { return head_end_; }
    const std::vector<Lease>& leases() const { return leases_; }

    /// The bytes of slots [0, n) (n clamped to the full cache).
    uint64_t bytes_of(int64_t n) const {
        if (n <= 0) return 0;
        if (n > full_) n = full_;
        return off_.empty() ? (uint64_t) n * max_blob_ : off_[(size_t) n];
    }
    /// Slots [0, k) whose bytes cover `need` (the smallest such k), or -1 when the whole cache does not.
    int64_t head_slots_for(uint64_t need) const {
        for (int64_t k = 0; k <= full_; ++k)
            if (bytes_of(k) >= need) return k;
        return -1;
    }

    /// A FIXED region at the head, stacked after the ones before it.  Only at start (before any loan or shrink): its
    /// address is baked into graphs.  `budget_slots`: the most slots the head regions may take together.  -1: refused.
    int reserve_head(const std::string& name, uint64_t bytes, int64_t budget_slots) {
        if (any_active(Addr::Relocatable) || live_ != full_) return -1;
        const int64_t lo = head_end_;
        int64_t hi = lo;
        while (hi < full_ && bytes_of(hi) - bytes_of(lo) < bytes) ++hi;
        if (bytes_of(hi) - bytes_of(lo) < bytes || hi > budget_slots) return -1;
        head_end_ = hi;
        const int id = add(name, Addr::Fixed, -1, lo, hi, bytes);
        leases_.back().active = false;   // placed, not occupied: experts use it until `occupy` (the slot sessions' way)
        return id;
    }
    /// A fixed region in use (its experts evicted) / free again (experts may refill it).  The loan keeps clear of the
    /// head regions either way: their place is fixed.
    bool occupy(int id) { return set_fixed(id, true); }
    bool vacate(int id) { return set_fixed(id, false); }

    /// A RELOCATABLE loan: the first gap from the tail of `live` that holds `bytes`, taken from the gap's high end (the
    /// tail slots whose bytes cover it; with no other loan out that is split_lend_slots over `live`, today's loan).
    /// Refused (-1) when no gap holds it without reaching into the head regions, or when the loans together would
    /// leave the experts fewer than `keep_min` slots past the head regions.  `available` says how much would fit.
    int lend_tail(const std::string& name, uint64_t bytes, int64_t keep_min) {
        if (bytes == 0) return -1;
        const int64_t keep = keep_min > 0 ? keep_min : 0;
        for (const auto& [glo, ghi] : gaps()) {
            const int64_t lo = take_from_top(glo, ghi, bytes);
            if (lo < 0) continue;
            if (loaned_slots() + (ghi - lo) > live_ - head_end_ - keep) return -1;   // the experts' floor
            return add(name, Addr::Relocatable, -1, lo, ghi, bytes);
        }
        return -1;
    }
    /// The most bytes one more loan could get now (the largest gap, within the experts' floor) - for a caller that can
    /// take less (a smaller prompt chunk on a short card) instead of being refused.
    uint64_t available(int64_t keep_min) const {
        const int64_t keep = keep_min > 0 ? keep_min : 0;
        const int64_t room = live_ - head_end_ - keep - loaned_slots();
        if (room <= 0) return 0;
        uint64_t best = 0;
        for (const auto& [glo, ghi] : gaps()) {
            const int64_t lo = ghi - room > glo ? ghi - room : glo;
            const uint64_t b = bytes_of(ghi) - bytes_of(lo);
            if (b > best) best = b;
        }
        return best;
    }

    /// A sub-loan inside an active relocatable loan, from its low end, after the sub-loans already in it.  -1: no room.
    int sub(int parent, const std::string& name, uint64_t bytes) {
        const Lease* p = find(parent);
        if (p == nullptr || !p->active || p->addr != Addr::Relocatable || p->parent >= 0) return -1;
        int64_t lo = p->lo;
        for (const Lease& l : leases_)
            if (l.active && l.parent == parent && l.hi > lo) lo = l.hi;
        int64_t hi = lo;
        while (hi < p->hi && bytes_of(hi) - bytes_of(lo) < bytes) ++hi;
        if (bytes_of(hi) - bytes_of(lo) < bytes) return -1;
        return add(name, Addr::Relocatable, parent, lo, hi, bytes);
    }

    /// EXTERNAL room: keep `keep_slots` usable and give the rest back.  Refused while a loan is out (the engine only
    /// shrinks between requests) or below the head regions.  Growing back (a larger `keep_slots`) is always allowed up to
    /// the full cache.
    bool shrink_to(int64_t keep_slots) {
        if (keep_slots > full_) keep_slots = full_;
        if (keep_slots < live_ && (any_active(Addr::Relocatable) || keep_slots < head_end_)) return false;
        live_ = keep_slots;
        return true;
    }

    /// Give a lease back (and the sub-loans inside it).
    void release(int id) {
        for (Lease& l : leases_)
            if (l.id == id || l.parent == id) l.active = false;
    }

    /// The slots experts may use now: [0, live) minus the occupied head regions and the active loan.
    int64_t expert_slots() const {
        int64_t n = live_;
        for (const Lease& l : leases_)
            if (l.active && l.parent < 0 && l.addr != Addr::External) n -= l.hi - l.lo;
        return n;
    }

private:
    /// Slots held by the top-level relocatable loans.
    int64_t loaned_slots() const {
        int64_t n = 0;
        for (const Lease& l : leases_)
            if (l.active && l.addr == Addr::Relocatable && l.parent < 0) n += l.hi - l.lo;
        return n;
    }
    /// The free stretches of [head_end, live) between the top-level loans, from the tail down.
    std::vector<std::pair<int64_t, int64_t>> gaps() const {
        std::vector<std::pair<int64_t, int64_t>> busy;
        for (const Lease& l : leases_)
            if (l.active && l.addr == Addr::Relocatable && l.parent < 0) busy.emplace_back(l.lo, l.hi);
        std::sort(busy.begin(), busy.end(), [](const auto& a, const auto& b) { return a.first > b.first; });
        std::vector<std::pair<int64_t, int64_t>> out;
        int64_t top = live_;
        for (const auto& [lo, hi] : busy) {
            if (hi < top) out.emplace_back(hi, top);
            if (lo < top) top = lo;
        }
        if (head_end_ < top) out.emplace_back(head_end_, top);
        return out;
    }
    /// The lowest slot of a run ending at `hi` (inside [glo, hi)) whose bytes cover `need`, or -1.
    int64_t take_from_top(int64_t glo, int64_t hi, uint64_t need) const {
        int64_t lo = hi;
        while (lo > glo && bytes_of(hi) - bytes_of(lo) < need) --lo;
        return bytes_of(hi) - bytes_of(lo) >= need ? lo : -1;
    }
    bool any_active(Addr a) const {
        for (const Lease& l : leases_)
            if (l.active && l.addr == a && l.parent < 0) return true;
        return false;
    }
    bool set_fixed(int id, bool on) {
        for (Lease& l : leases_)
            if (l.id == id && l.addr == Addr::Fixed) { l.active = on; return true; }
        return false;
    }
    const Lease* find(int id) const {
        for (const Lease& l : leases_)
            if (l.id == id) return &l;
        return nullptr;
    }
    int add(const std::string& name, Addr a, int parent, int64_t lo, int64_t hi, uint64_t bytes) {
        Lease l;
        l.id = next_id_++;
        l.name = name;
        l.addr = a;
        l.parent = parent;
        l.lo = lo;
        l.hi = hi;
        l.bytes = bytes;
        l.active = true;
        leases_.push_back(l);
        return l.id;
    }

    int64_t full_ = 0, live_ = 0, head_end_ = 0;
    std::vector<uint64_t> off_;
    uint64_t max_blob_ = 0;
    std::vector<Lease> leases_;
    int next_id_ = 0;
};

}  // namespace strata::program::vram_ledger
