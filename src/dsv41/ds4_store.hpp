// Strata. src/dsv41/ds4_store.hpp - the RAM tier of the DeepSeek-V4.1 routed experts as a cache of its own: whole experts read
// from the model's files by unbuffered requests (the OS page cache neither used nor disturbed), kept in a fixed arena of slots
// that a replacement policy owns, filled on demand (the CPU lane's picks) and ahead of time (the lookahead's guesses).
//
// A slot holds an expert in the engine's blob layout (gate | up | down at the layer's offsets tr[r], the same as a VRAM slot), so
// the CPU lane's rows read it in place and - when the arena is registered with the GPU - the device kernels read it over PCIe
// (zero-copy, 24 GB/s on a 5070) through the same plan pointers as a VRAM expert.  An expert is three matrices at three file
// offsets (the GGUF's one tensor a role and layer): each is read aligned (4096) into the slot and moved into place (the read's
// lead of up to 4095 bytes), the blob layout leaving 8192 bytes between matrices (and around the blob) for the aligned spans.
//
// Memory: slots are allocated in chunks as they are first needed (private memory), never past the budget nor past the RAM this
// process can still get less `floor_bytes` (the owner's session and the system keep theirs).  With Cfg::layer_budget the slots
// come in classes, one a blob size (a layer's experts take the class of its blob, each class its own arena, chunks and policy),
// sized from what the budget holds of each layer's hottest experts: a low-bit pack's layers differ (9.1 to 11.3 MB in the
// ISTA-DAS 2.25 set), and one slot size, the largest, left a fifth of the arena unused.
//
// Policy (the slot to give up, among the unpinned, not GPU-visible ones): lru (oldest touch), lfu (lowest decayed hit count).
// A slot whose expert a window is computing is pinned from ensure() to release(); a slot the GPU may read is `visible` until
// sync() takes it back at a window boundary (the GPU idle), which also publishes the newly ready ones.
#pragma once

#include "ds4_os.hpp"
#include "ds4_pool.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

namespace strata::ds4 {

class ExpertStore {
public:
    struct LayerGeo {
        int n_exp = 0;
        int file[3] = {-1, -1, -1};
        uint64_t off[3] = {0, 0, 0}, stride[3] = {0, 0, 0};   // matrix r of expert e: file[r], off[r] + e stride[r]
        uint64_t bytes[3] = {0, 0, 0};
        uint64_t tr[3] = {0, 0, 0};                            // matrix r's place in the blob
        uint64_t blob = 0;                                     // the blob's bytes (gaps of 8192 between matrices)
    };
    enum Policy { kLru = 0, kLfu = 1 };
    struct Stats {
        std::atomic<int64_t> hits{0}, misses{0}, prefetched{0}, prefetch_used{0}, prefetch_wasted{0}, evictions{0}, fails{0};
        std::atomic<int64_t> bytes_read{0}, wait_ns{0}, no_slot{0}, cancelled{0}, put{0};
    };
    struct Vis {                 // a slot the GPU may read from now on (dev != 0), or no longer (dev == 0)
        int layer = 0, e = 0;
        unsigned long long dev = 0;
    };

    struct Cfg {
        int threads = 8;
        size_t budget_bytes = 0;           // the arena at most (grown in chunks as slots are first needed)
        size_t floor_bytes = 8ull << 30;   // RAM left alone
        Policy policy = kLru;
        float decay = 0.97f;               // lfu: heat x decay every `decay_every` accesses of its layer
        int decay_every = 240;
        std::function<uint64_t()> ram_avail;   // the RAM this process can still get (~0: unknown)
        // a new chunk of the arena for the GPU to read in place: returns the device address of `host` (null: not visible)
        std::function<void*(void* host, size_t bytes)> map_chunk;
        std::function<void(void* host)> unmap_chunk;
        size_t reserve = 192;              // slots kept evictable (not visible) for the window's demand and guesses
        int vis_percent = 100;             // the share of experts the GPU may take (by a fixed hash): the rest stay the CPU lane's, so both lanes work
        // bytes of blobs each layer's experts may hold (the budget's share of the hottest, per layer): slot classes sized from it;
        // empty, one class of the largest blob's slots
        std::vector<uint64_t> layer_budget;
    };

    void set_pfds(const std::vector<os::fd_t>& pfds) { pfds_ = pfds; }   // (the fallback handles were reopened)
    bool init(const Cfg& c, const std::vector<LayerGeo>& layers, const std::vector<os::fd_t>& dfds, const std::vector<os::fd_t>& pfds) {
        cfg_ = c;
        geo_ = layers;
        dfds_ = dfds;
        pfds_ = pfds;
        auto slot_of = [](uint64_t blob) {
            return (size_t) ((blob + 2 * kSlack + os::kDirectAlign - 1) & ~(uint64_t) (os::kDirectAlign - 1));
        };
        size_t blob = 0;
        for (const LayerGeo& g : geo_) blob = std::max<size_t>(blob, g.blob);
        if (blob == 0 || c.budget_bytes < slot_of(blob)) return false;
        // the classes: one a slot size the layers' blobs take, or one of the largest
        const bool by_size = c.layer_budget.size() == geo_.size() &&
                             !(std::getenv("DS4_STORE_CLASSES") && std::atoi(std::getenv("DS4_STORE_CLASSES")) == 0);
        std::vector<size_t> sizes;
        for (const LayerGeo& g : geo_)
            if (g.blob > 0) sizes.push_back(by_size ? slot_of(g.blob) : slot_of(blob));
        std::sort(sizes.begin(), sizes.end());
        sizes.erase(std::unique(sizes.begin(), sizes.end()), sizes.end());
        layer_cls_.assign(geo_.size(), -1);
        for (size_t l = 0; l < geo_.size(); ++l)
            if (geo_[l].blob > 0)
                layer_cls_[l] = (int) (std::lower_bound(sizes.begin(), sizes.end(), by_size ? slot_of(geo_[l].blob) : slot_of(blob)) - sizes.begin());
        cls_.resize(sizes.size());
        for (size_t k = 0; k < sizes.size(); ++k) cls_[k].slot_bytes = sizes[k];
        if (!by_size) cls_[0].nmax = c.budget_bytes / sizes[0];
        else {   // each class: the slots its layers' budgets take, then what the budget leaves goes to them in proportion
            std::vector<double> want(cls_.size(), 0.0);
            for (size_t l = 0; l < geo_.size(); ++l)
                if (layer_cls_[l] >= 0 && geo_[l].blob > 0) want[(size_t) layer_cls_[l]] += (double) c.layer_budget[l] / (double) geo_[l].blob;
            double bytes = 0, wsum = 0;
            for (size_t k = 0; k < cls_.size(); ++k) {
                cls_[k].nmax = std::max<size_t>((size_t) want[k], kChunk);   // (a chunk at least: any layer's miss finds a slot)
                bytes += (double) cls_[k].nmax * (double) cls_[k].slot_bytes;
                wsum += want[k] * (double) cls_[k].slot_bytes;
            }
            if (bytes > (double) c.budget_bytes) {   // (the minimums pushed it over: scaled back to the budget)
                const double f = (double) c.budget_bytes / bytes;
                for (Cls& k : cls_) k.nmax = std::max<size_t>(8, (size_t) ((double) k.nmax * f));
            } else if (wsum > 0) {
                const double left = (double) c.budget_bytes - bytes;
                for (size_t k = 0; k < cls_.size(); ++k)
                    cls_[k].nmax += (size_t) (left * (want[k] * (double) cls_[k].slot_bytes / wsum) / (double) cls_[k].slot_bytes);
            }
        }
        nmax_ = 0;
        for (Cls& k : cls_) {
            k.off = nmax_;
            nmax_ += k.nmax;
        }
        for (Cls& k : cls_) {   // one reserved range a class (chunks committed as they are needed); the reserve in proportion
            k.reserve = cls_.size() == 1 ? c.reserve : std::max<size_t>(8, (size_t) ((double) c.reserve * (double) k.nmax / (double) nmax_));
            k.arena_bytes = ((k.nmax + kChunk - 1) / kChunk) * kChunk * k.slot_bytes;
            k.arena = (uint8_t*) os::range_reserve(k.arena_bytes);
            if (k.arena == nullptr) return false;
        }
        slots_.reset(new Slot[nmax_]);
        for (size_t k = 0; k < cls_.size(); ++k)
            for (size_t j = 0; j < cls_[k].nmax; ++j) slots_[cls_[k].off + j].cls = (int) k;
        where_.resize(geo_.size());
        heat_.resize(geo_.size());
        hits_in_layer_.assign(geo_.size(), 0);
        for (size_t l = 0; l < geo_.size(); ++l) {
            where_[l].assign((size_t) geo_[l].n_exp, -1);
            heat_[l].assign((size_t) geo_[l].n_exp, 0.0f);
        }
        io_.reset(new IoQueue(std::max(1, c.threads)));
        for (size_t k = 0; k < cls_.size(); ++k) grow((int) k);   // (a chunk a class now: whether the GPU can map it is known from the start)
        // DS4_STORE_PREGROW=1 (default on Linux with the GPU mapping, =0 off): the whole arena now.  cuMemHostRegister (the GPU mapping of a chunk) taken while a kernel of
        // the decode is running waited on it for good on Linux (H100, driver 580): registering all chunks before the first
        // kernel avoids it
#if defined(__linux__)
        const bool pregrow = cfg_.map_chunk && !(std::getenv("DS4_STORE_PREGROW") != nullptr && std::atoi(std::getenv("DS4_STORE_PREGROW")) == 0);   // (Linux: on with the GPU mapping)
#else
        const bool pregrow = std::getenv("DS4_STORE_PREGROW") != nullptr && std::atoi(std::getenv("DS4_STORE_PREGROW")) != 0;
#endif
        if (pregrow)
            for (size_t k = 0; k < cls_.size(); ++k)
                while (grow((int) k)) {
                }
        return true;
    }
    ~ExpertStore() {
        io_.reset();
        for (Cls& k : cls_) {
            for (size_t i = 0; i < k.chunks.size(); ++i)
                if (cfg_.unmap_chunk && k.chunk_dev[i] != nullptr) cfg_.unmap_chunk(k.chunks[i].first);
            os::range_release(k.arena, k.arena_bytes);
        }
    }

    /// The largest class's slot (the one a single class would have)
    size_t slot_bytes() const {
        size_t b = 0;
        for (const Cls& k : cls_) b = std::max(b, k.slot_bytes);
        return b;
    }
    size_t slots_allocated() const {
        size_t n = 0;
        for (const Cls& k : cls_) n += k.nalloc;
        return n;
    }
    size_t slots_max() const { return nmax_; }
    /// The slot classes: their slot sizes, slots at most and arena bytes, for the log
    std::string classes_desc() const {
        std::string s;
        char b[96];
        for (const Cls& k : cls_) {
            std::snprintf(b, sizeof b, "%s%zu x %.2f MB", s.empty() ? "" : ", ", k.nmax, (double) k.slot_bytes / 1e6);
            s += b;
        }
        return s;
    }
    size_t arena_total() const {
        size_t b = 0;
        for (const Cls& k : cls_) b += k.nmax * k.slot_bytes;
        return b;
    }
    const Stats& stats() const { return st_; }
    double busy_ms() const { return io_ ? io_->busy_ms() : 0.0; }
    static constexpr size_t kSlack = 8192;

    /// The slot holding (layer, e) ready to read, or -1.  (Any thread; a slot found stays valid while it is pinned or visible.)
    int find(int layer, int e) const {
        if (layer < 0 || layer >= (int) where_.size()) return -1;   // (a layer the store does not cover: the DSpark drafter's)
        const int s = where_[(size_t) layer][(size_t) e];
        if (s < 0) return -1;
        Slot& sp = slots_[(size_t) s];
        try_ready(sp);
        return sp.state.load(std::memory_order_acquire) == kReady && sp.layer == layer && sp.expert == e ? s : -1;
    }
    const uint8_t* mat(int slot, int r) const { return slots_[(size_t) slot].base + geo_[(size_t) slots_[(size_t) slot].layer].tr[r]; }
    const uint8_t* blob(int slot) const { return slots_[(size_t) slot].base; }
    /// The device address of the slot's blob (0: the arena is not mapped for the GPU)
    unsigned long long dev_blob(int slot) const {
        const Cls& k = cls_[(size_t) slots_[(size_t) slot].cls];
        const size_t c = ((size_t) slot - k.off) / kChunk;
        if (c >= k.chunk_dev.size() || k.chunk_dev[c] == nullptr) return 0;
        return (unsigned long long) ((uint8_t*) k.chunk_dev[c] + (slots_[(size_t) slot].base - (uint8_t*) k.chunks[c].first));
    }

    /// Makes the experts es[0..n) of `layer` ready and pinned (reading what is not here), waiting for the reads; the
    /// caller releases them.  False if a read failed or no slot could be had for one (ok[i] says which are fine; the caller
    /// reads the others another way).
    bool ensure(int layer, const int* es, int n, std::vector<char>& ok) {
        std::vector<int> sl;
        begin(layer, es, n, ok, sl);
        return wait_slots(sl, ok, 0, n);
    }
    /// The first half of ensure(): the experts' slots found or their reads started (pinned), no waiting.  sl[i] is the slot or -1.
    void begin(int layer, const int* es, int n, std::vector<char>& ok, std::vector<int>& sl) {
        ok.assign((size_t) n, 0);
        sl.assign((size_t) n, -1);
        if (layer < 0 || layer >= (int) geo_.size()) return;
        std::lock_guard<std::mutex> lk(mu_);
        for (int i = 0; i < n; ++i) {
            const int e = es[i];
            touch_heat(layer, e);
            int s = where_[(size_t) layer][(size_t) e];
            if (s >= 0) {
                Slot& sp = slots_[(size_t) s];
                try_ready(sp);
                if (sp.state.load() == kReady && !sp.prefetch) st_.hits.fetch_add(1, std::memory_order_relaxed);
                if (sp.prefetch) st_.prefetch_used.fetch_add(1, std::memory_order_relaxed);
                sp.prefetch = false;
                sp.pin++;
                sp.tick = ++clock_;
                sl[(size_t) i] = s;
                continue;
            }
            st_.misses.fetch_add(1, std::memory_order_relaxed);
            s = start_fill(layer, e, true);
            sl[(size_t) i] = s;
        }
    }
    /// True when the reads of slot sl (from begin) are all done.
    bool loaded(int s) const { return s >= 0 && slots_[(size_t) s].left.load(std::memory_order_acquire) <= 0; }
    /// The second half: waits for the slots sl[a..b) and marks ok[i] for the good ones.  False if any failed or had no slot.
    bool wait_slots(std::vector<int>& sl, std::vector<char>& ok, int a, int b) {
        const auto t0 = std::chrono::steady_clock::now();
        bool all = true;
        for (int i = a; i < b; ++i) {
            const int s = sl[(size_t) i];
            if (s < 0) {
                all = false;
                continue;
            }
            Slot& sp = slots_[(size_t) s];
            while (sp.left.load(std::memory_order_acquire) > 0) io_->wait_any(2000);
            std::lock_guard<std::mutex> lk(mu_);
            finish(sp);
            if (sp.state.load() == kReady) ok[(size_t) i] = 1;
            else {
                sp.pin--;
                sl[(size_t) i] = -1;
                all = false;
            }
        }
        st_.wait_ns.fetch_add(std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - t0).count(),
                              std::memory_order_relaxed);
        return all;
    }
    void release(int layer, const int* es, int n, const std::vector<char>& ok) {
        if (layer < 0 || layer >= (int) geo_.size()) return;
        std::lock_guard<std::mutex> lk(mu_);
        for (int i = 0; i < n; ++i)
            if (ok[(size_t) i]) {
                const int s = where_[(size_t) layer][(size_t) es[i]];
                if (s >= 0 && slots_[(size_t) s].pin > 0) slots_[(size_t) s].pin--;
            }
    }
    /// Starts reading (layer, e) if it is neither here nor on its way (a guess: unpinned, may be given up before it is used).
    void prefetch(int layer, int e) {
        if (layer < 0 || layer >= (int) geo_.size() || e < 0 || e >= geo_[(size_t) layer].n_exp) return;
        if (where_[(size_t) layer][(size_t) e] >= 0) return;
        std::lock_guard<std::mutex> lk(mu_);
        if (where_[(size_t) layer][(size_t) e] >= 0) return;
        if (start_fill(layer, e, false) >= 0) st_.prefetched.fetch_add(1, std::memory_order_relaxed);
    }
    /// Reads the experts of `list` (layer, expert; hottest first) into free slots and returns when they are in (a start that
    /// is already warm: the profile's hottest, so the first tokens find them).  The first `list.size()` fit the arena less
    /// `leave` slots; returns how many were read.
    size_t preload(const std::vector<std::pair<int, int>>& list, size_t leave) {
        size_t n = 0;
        {
            std::lock_guard<std::mutex> lk(mu_);
            // what each class takes: the list's first that fit it less its share of `leave` (hottest first), then read coldest first
            std::vector<size_t> room(cls_.size()), used(cls_.size(), 0);
            for (size_t k = 0; k < cls_.size(); ++k) {
                const size_t lv = cls_.size() == 1 ? leave : (size_t) ((double) leave * (double) cls_[k].nmax / (double) std::max<size_t>(1, nmax_));
                room[k] = cls_[k].nmax > lv ? cls_[k].nmax - lv : 0;
            }
            std::vector<size_t> take;
            for (size_t k = 0; k < list.size(); ++k) {
                const int layer = list[k].first, e = list[k].second;
                if (layer < 0 || layer >= (int) geo_.size() || e < 0 || e >= geo_[(size_t) layer].n_exp || layer_cls_[(size_t) layer] < 0) continue;
                const size_t ci = (size_t) layer_cls_[(size_t) layer];
                if (used[ci] >= room[ci]) continue;
                ++used[ci];
                take.push_back(k);
            }
            for (size_t i = take.size(); i-- > 0;) {   // (coldest first: the hottest get the newest ticks)
                const int layer = list[take[i]].first, e = list[take[i]].second;
                if (where_[(size_t) layer][(size_t) e] >= 0) continue;
                const int s = start_fill(layer, e, false);
                if (s < 0) continue;
                slots_[(size_t) s].prefetch = false;   // (warm, not a guess)
                ++n;
            }
        }
        for (size_t i = 0;; ++i) {   // until every read is done
            bool busy = false;
            {
                std::lock_guard<std::mutex> lk(mu_);
                for (size_t j = 0; j < nmax_; ++j) {
                    Slot& sp = slots_[j];
                    if (sp.base == nullptr || sp.state.load(std::memory_order_acquire) != kLoading) continue;
                    if (sp.left.load(std::memory_order_acquire) > 0) busy = true;
                    else {
                        try_ready(sp);
                        if (sp.state.load(std::memory_order_acquire) == kLoading) finish(sp);
                    }
                }
            }
            if (!busy) break;
            io_->wait_any(5000);
        }
        return n;
    }
    /// A slot for (layer, e) that the caller fills with the expert's blob itself (an expert the VRAM cache gives back to the CPU lane: its bytes
    /// are in memory already, so no read from the SSD while the lane waits).  Returns the blob's address, or nullptr (it is here already, or
    /// no slot can be had).  The slot is pinned and not findable until put_done().
    uint8_t* put_begin(int layer, int e) {
        if (layer < 0 || layer >= (int) geo_.size() || e < 0 || e >= geo_[(size_t) layer].n_exp) return nullptr;
        std::lock_guard<std::mutex> lk(mu_);
        if (where_[(size_t) layer][(size_t) e] >= 0 || layer_cls_[(size_t) layer] < 0) return nullptr;
        const int s = take_slot(layer_cls_[(size_t) layer], false);
        if (s < 0) return nullptr;
        Slot& sp = slots_[(size_t) s];
        if (sp.layer >= 0 && sp.expert >= 0) {
            where_[(size_t) sp.layer][(size_t) sp.expert] = -1;
            if (sp.prefetch) st_.prefetch_wasted.fetch_add(1, std::memory_order_relaxed);
            st_.evictions.fetch_add(1, std::memory_order_relaxed);
        }
        sp.layer = layer;
        sp.expert = e;
        sp.pin = 1;
        sp.prefetch = false;
        sp.visible = false;
        sp.tick = ++clock_;
        sp.bad.store(false);
        sp.cancel.store(false);
        sp.left.store(1, std::memory_order_release);
        sp.state.store(kLoading, std::memory_order_release);
        where_[(size_t) layer][(size_t) e] = s;
        st_.put.fetch_add(1, std::memory_order_relaxed);
        return sp.base;
    }
    void put_done(int layer, int e, bool ok) {
        std::lock_guard<std::mutex> lk(mu_);
        const int s = where_[(size_t) layer][(size_t) e];
        if (s < 0) return;
        Slot& sp = slots_[(size_t) s];
        if (sp.pin > 0) sp.pin--;
        if (ok) {
            sp.left.store(0, std::memory_order_release);
            sp.state.store(kReady, std::memory_order_release);
        } else {
            where_[(size_t) layer][(size_t) e] = -1;
            sp.layer = sp.expert = -1;
            sp.left.store(0, std::memory_order_release);
            sp.state.store(kFree, std::memory_order_release);
            Cls& k = cls_[(size_t) sp.cls];
            k.free_hint = std::min<size_t>(k.free_hint, (size_t) s - k.off);
        }
    }
    /// Guesses still waiting in the queue for a layer the token is past: dropped unread (their slots free once the queue has gone by)
    void cancel_stale(int layer_now) {
        std::lock_guard<std::mutex> lk(mu_);
        for (size_t i = 0; i < nmax_; ++i) {
            Slot& sp = slots_[i];
            if (sp.base != nullptr && sp.prefetch && sp.pin == 0 && sp.layer >= 0 && sp.layer <= layer_now && sp.state.load(std::memory_order_acquire) == kLoading)
                sp.cancel.store(true, std::memory_order_release);
        }
    }
    /// An access the host saw for an expert the GPU computed (keeps it warm in the policy's eyes).
    void touch(int layer, int e) {
        if (layer < 0 || layer >= (int) where_.size()) return;
        std::lock_guard<std::mutex> lk(mu_);
        touch_heat(layer, e);
        const int s = where_[(size_t) layer][(size_t) e];
        if (s >= 0) slots_[(size_t) s].tick = ++clock_;
    }
    bool gpu_mapped() const { return !cls_.empty() && !cls_[0].chunk_dev.empty() && cls_[0].chunk_dev[0] != nullptr; }
    bool identity_mapped() const {
        for (const Cls& k : cls_)
            for (size_t i = 0; i < k.chunks.size(); ++i)
                if (k.chunk_dev[i] != nullptr && k.chunk_dev[i] != k.chunks[i].first) return false;
        return true;
    }
    size_t max_blob() const { return slot_bytes() - 2 * kSlack; }

    /// At a window boundary (nothing reading the slots; the caller applies `ev` to the device's table before the next window):
    /// every ready slot not visible yet becomes visible (`gpu_ok(layer, e)`: false for an expert VRAM holds), and visible
    /// slots of the lowest score are taken back until `cfg.reserve` slots can be given up.  Returns the number of events.
    size_t sync(std::vector<Vis>& ev, const std::function<bool(int, int)>& gpu_ok) {
        std::lock_guard<std::mutex> lk(mu_);
        std::vector<size_t> nvis(cls_.size(), 0), nfree(cls_.size(), 0);
        for (size_t i = 0; i < nmax_; ++i) {
            Slot& sp = slots_[i];
            if (sp.base == nullptr) continue;
            const size_t ci = (size_t) sp.cls;
            if (sp.state.load(std::memory_order_acquire) == kLoading && sp.left.load(std::memory_order_acquire) == 0) {
                try_ready(sp);
                if (sp.state.load(std::memory_order_acquire) == kLoading) finish(sp);
            }
            const int st = sp.state.load(std::memory_order_acquire);
            if (st == kFree) {
                ++nfree[ci];
                continue;
            }
            if (st != kReady) continue;
            if (sp.visible) {
                ++nvis[ci];
                continue;
            }
            if (sp.pin == 0) ++nfree[ci];
            if (sp.pin == 0 && dev_blob((int) i) != 0 && gpu_ok(sp.layer, sp.expert) && !sp.prefetch && vis_hash(sp.layer, sp.expert) < cfg_.vis_percent) {
                sp.visible = true;
                ++nvis[ci];
                --nfree[ci];
                ev.push_back({sp.layer, sp.expert, dev_blob((int) i)});
            }
        }
        // give slots back until each class's reserve is there: its visible ones with the lowest score
        for (size_t ci = 0; ci < cls_.size(); ++ci) {
            const Cls& k = cls_[ci];
            if (nfree[ci] >= k.reserve || nvis[ci] == 0) continue;
            std::vector<std::pair<double, size_t>> cand;
            cand.reserve(nvis[ci]);
            for (size_t i = k.off; i < k.off + k.nalloc; ++i)
                if (slots_[i].visible && slots_[i].state.load(std::memory_order_acquire) == kReady) cand.push_back({score(slots_[i]), i});
            const size_t want = std::min(cand.size(), k.reserve - nfree[ci]);
            std::partial_sort(cand.begin(), cand.begin() + (long) want, cand.end());
            for (size_t j = 0; j < want; ++j) {
                Slot& sp = slots_[cand[j].second];
                sp.visible = false;
                ev.push_back({sp.layer, sp.expert, 0ull});
            }
        }
        return ev.size();
    }
    /// RAM is short where the GPU reads the slots in place: the GPU-visible slots of the next `want` chunks shrink_chunks would
    /// give back taken off its tables (a Vis event each, dev 0, for the caller to apply before shrink_chunks), stopping at a
    /// chunk with a slot pinned or loading.  Returns the chunks it cleared.  Only sync() makes a slot visible again, and the
    /// caller runs both.
    size_t hide_tail(size_t want, std::vector<Vis>& ev) {
        std::lock_guard<std::mutex> lk(mu_);
        size_t done = 0;
        for (const auto& pc : tail_plan(want)) {
            const Cls& k = cls_[(size_t) pc.first];
            const size_t first = k.off + pc.second * kChunk, last = k.off + std::min(k.nalloc, (pc.second + 1) * kChunk);
            for (size_t i = first; i < last; ++i) {
                const Slot& sp = slots_[i];
                if (sp.pin > 0 || sp.state.load(std::memory_order_acquire) == kLoading) return done;
            }
            for (size_t i = first; i < last; ++i) {
                Slot& sp = slots_[i];
                if (sp.visible) {
                    sp.visible = false;
                    ev.push_back({sp.layer, sp.expert, 0ull});
                }
            }
            ++done;
        }
        return done;
    }
    /// RAM is short: gives back whole chunks from the ends of the classes' arenas (their experts forgotten; the class holding the
    /// most RAM first, a chunk each kept) - up to `want`, stopping at one with a slot in use (pinned, loading, or GPU-visible).
    /// Returns the chunks given back.
    size_t shrink_chunks(size_t want) {
        std::lock_guard<std::mutex> lk(mu_);
        size_t done = 0;
        for (const auto& pc : tail_plan(want)) {
            Cls& k = cls_[(size_t) pc.first];
            if (k.chunks.size() != pc.second + 1) break;   // (not this class's last chunk any more)
            const size_t first = k.off + pc.second * kChunk, last = k.off + k.nalloc;
            bool busy = false;
            for (size_t i = first; i < last && !busy; ++i) {
                Slot& sp = slots_[i];
                if (sp.state.load(std::memory_order_acquire) == kLoading && sp.left.load(std::memory_order_acquire) == 0) {   // (read by now)
                    try_ready(sp);
                    if (sp.state.load(std::memory_order_acquire) == kLoading) finish(sp);
                }
                busy = sp.pin > 0 || sp.visible || sp.state.load(std::memory_order_acquire) == kLoading;
            }
            if (busy) break;
            for (size_t i = first; i < last; ++i) {
                Slot& sp = slots_[i];
                if (sp.layer >= 0 && sp.expert >= 0) {
                    where_[(size_t) sp.layer][(size_t) sp.expert] = -1;
                    st_.evictions.fetch_add(1, std::memory_order_relaxed);
                }
                sp.layer = sp.expert = -1;
                sp.prefetch = false;
                sp.state.store(kFree, std::memory_order_release);
                sp.base = nullptr;
            }
            const size_t c = pc.second;
            if (cfg_.unmap_chunk && k.chunk_dev[c] != nullptr) cfg_.unmap_chunk(k.chunks[c].first);
            os::range_decommit(k.chunks[c].first, k.chunks[c].second);
            k.chunks.pop_back();
            k.chunk_dev.pop_back();
            k.nalloc = c * kChunk;
            k.free_hint = std::min(k.free_hint, k.nalloc);
            ++done;
            ++shrunk_;
        }
        return done;
    }
    size_t shrunk() const { return shrunk_; }
    /// An expert VRAM took over: no longer a visible slot
    void unmark(int layer, int e) {
        if (layer < 0 || layer >= (int) where_.size()) return;
        std::lock_guard<std::mutex> lk(mu_);
        const int s = where_[(size_t) layer][(size_t) e];
        if (s >= 0) slots_[(size_t) s].visible = false;
    }
    size_t visible_count() const {
        size_t n = 0;
        for (size_t i = 0; i < nmax_; ++i) n += slots_[i].base != nullptr && slots_[i].visible ? 1 : 0;
        return n;
    }

private:
    static constexpr int kFree = 0, kLoading = 1, kReady = 2, kFailed = 3;
    // slots a chunk of the arena (the unit the GPU mapping registers; DS4_STORE_CHUNK_SLOTS: a test knob for the registration segment size)
    static size_t chunk_slots() {
        const char* e = std::getenv("DS4_STORE_CHUNK_SLOTS");
        const long v = e ? std::atol(e) : 0;
        return v >= 8 && v <= 4096 ? (size_t) v : 64;
    }
    const size_t kChunk = chunk_slots();
    struct Slot {
        uint8_t* base = nullptr;   // the blob's first byte (null: its chunk is not committed)
        int cls = 0;               // its class
        int layer = -1, expert = -1, pin = 0;
        bool prefetch = false, visible = false;
        std::atomic<int> state{kFree}, left{0};
        std::atomic<bool> bad{false}, cancel{false};
        uint64_t tick = 0;
    };

    static int vis_hash(int layer, int e) { return (int) (((unsigned) layer * 2654435761u + (unsigned) e * 40503u) >> 7) % 100; }
    double score(const Slot& sp) const {
        if (cfg_.policy == kLru) return (double) sp.tick;
        return (double) heat_[(size_t) sp.layer][(size_t) sp.expert] + 1e-9 * (double) sp.tick;
    }
    void touch_heat(int layer, int e) {
        heat_[(size_t) layer][(size_t) e] += 1.0f;
        if (cfg_.policy != kLru && ++hits_in_layer_[(size_t) layer] % (uint64_t) cfg_.decay_every == 0)
            for (float& h : heat_[(size_t) layer]) h *= cfg_.decay;   // (a linear pass over 384 floats every 240 accesses of that layer)
    }
    // a slot for (layer, e) and its reads started; -1 if there is none to be had.  mu_ held.
    int start_fill(int layer, int e, bool demand) {
        int s = layer_cls_[(size_t) layer] >= 0 ? take_slot(layer_cls_[(size_t) layer], demand) : -1;
        if (s < 0) {
            st_.no_slot.fetch_add(1, std::memory_order_relaxed);
            return -1;
        }
        Slot& sp = slots_[(size_t) s];
        if (sp.layer >= 0 && sp.expert >= 0) {
            where_[(size_t) sp.layer][(size_t) sp.expert] = -1;
            if (sp.prefetch) st_.prefetch_wasted.fetch_add(1, std::memory_order_relaxed);
            st_.evictions.fetch_add(1, std::memory_order_relaxed);
        }
        sp.layer = layer;
        sp.expert = e;
        sp.pin = demand ? 1 : 0;
        sp.prefetch = !demand;
        sp.visible = false;
        sp.tick = ++clock_;
        sp.state.store(kLoading, std::memory_order_release);
        sp.bad.store(false);
        sp.cancel.store(false);
        sp.left.store(3, std::memory_order_release);
        where_[(size_t) layer][(size_t) e] = s;
        const LayerGeo& g = geo_[(size_t) layer];
        IoQueue::Job jobs[3];
        for (int r = 0; r < 3; ++r) {
            const uint64_t off = g.off[r] + (uint64_t) e * g.stride[r];
            const uint64_t lead = off % os::kDirectAlign;
            uint8_t* fin = sp.base + g.tr[r];   // where the matrix belongs
            // the aligned read lands the file's byte `off` at dst + lead: dst is fin - lead rounded down to 4096 (its span spills
            // under 4096 bytes either way into the blob's gaps), the matrix is then moved up to fin
            uint8_t* dst = (uint8_t*) ((uintptr_t) (fin - lead) & ~(uintptr_t) (os::kDirectAlign - 1));
            IoQueue::Job& j = jobs[r];
            j.fd = pfds_[(size_t) g.file[r]];
            j.dfd = dfds_[(size_t) g.file[r]];
            j.off = off;
            j.len = g.bytes[r];
            j.dst = dst;
            j.keep = fin;
            j.left = &sp.left;
            j.bad = &sp.bad;
            j.cancel = demand ? nullptr : &sp.cancel;
        }
        st_.bytes_read.fetch_add((int64_t) (g.bytes[0] + g.bytes[1] + g.bytes[2]), std::memory_order_relaxed);
        io_->push(jobs, 3, !demand);   // (a guess waits behind every demand read)
        return s;
    }
    // loading -> ready once its reads are done and whole (lock-free; a failed one is cleaned up by finish / take_slot)
    void try_ready(Slot& sp) const {
        int want = kLoading;
        if (sp.left.load(std::memory_order_acquire) == 0 && !sp.bad.load(std::memory_order_acquire))
            sp.state.compare_exchange_strong(want, kReady, std::memory_order_acq_rel);
    }
    void finish(Slot& sp) {   // mu_ held; reads done
        if (sp.state.load() != kLoading) return;
        if (sp.bad.load()) {
            where_[(size_t) sp.layer][(size_t) sp.expert] = -1;
            sp.layer = sp.expert = -1;
            sp.state.store(kFree, std::memory_order_release);
            sp.prefetch = false;
            if (sp.cancel.load()) st_.cancelled.fetch_add(1, std::memory_order_relaxed);
            else st_.fails.fetch_add(1, std::memory_order_relaxed);
            return;
        }
        sp.state.store(kReady, std::memory_order_release);
    }
    bool grow(int ci) {
        Cls& k = cls_[(size_t) ci];
        if (k.nalloc >= k.nmax) return false;
        const size_t n = std::min(kChunk, k.nmax - k.nalloc);
        const size_t bytes = n * k.slot_bytes;
        const uint64_t avail = cfg_.ram_avail ? cfg_.ram_avail() : ~uint64_t{0};
        if (avail != ~uint64_t{0} && avail < bytes + cfg_.floor_bytes) return false;
        void* p = k.arena + k.nalloc * k.slot_bytes;
        if (!os::range_commit(p, bytes)) return false;
        k.chunks.push_back({p, bytes});
        void* dev = cfg_.map_chunk ? cfg_.map_chunk(p, bytes) : nullptr;
        k.chunk_dev.push_back(dev);
        for (size_t i = 0; i < n; ++i) slots_[k.off + k.nalloc + i].base = (uint8_t*) p + i * k.slot_bytes + kSlack;
        k.nalloc += n;
        return true;
    }
    // the chunks shrink_chunks gives back, in order: (class, chunk), the class holding the most RAM first, a chunk a class kept
    std::vector<std::pair<int, size_t>> tail_plan(size_t want) const {
        std::vector<size_t> n(cls_.size());
        for (size_t k = 0; k < cls_.size(); ++k) n[k] = cls_[k].chunks.size();
        std::vector<std::pair<int, size_t>> plan;
        while (plan.size() < want) {
            int best = -1;
            size_t bb = 0;
            for (size_t k = 0; k < cls_.size(); ++k) {
                if (n[k] <= 1) continue;
                const size_t b = n[k] * cls_[k].slot_bytes;
                if (best < 0 || b > bb) {
                    best = (int) k;
                    bb = b;
                }
            }
            if (best < 0) break;
            plan.push_back({best, --n[(size_t) best]});
        }
        return plan;
    }
    // the slot to give up (or a free one): the policy's lowest among the unpinned, not visible, ready ones.  mu_ held.
    int take_slot(int ci, bool demand) {
        Cls& k = cls_[(size_t) ci];
        for (size_t j = k.free_hint; j < k.nalloc; ++j)
            if (slots_[k.off + j].state.load(std::memory_order_acquire) == kFree) {
                k.free_hint = j + 1;
                return (int) (k.off + j);
            }
        if (grow(ci)) return take_slot(ci, demand);
        int best = -1;
        double bv = 0;
        for (size_t j = 0; j < k.nalloc; ++j) {
            const size_t i = k.off + j;
            Slot& sp = slots_[i];
            if (sp.state.load(std::memory_order_acquire) == kLoading && sp.left.load(std::memory_order_acquire) == 0) {
                try_ready(sp);
                if (sp.state.load(std::memory_order_acquire) == kLoading) finish(sp);
                if (sp.state.load(std::memory_order_acquire) == kFree) {
                    k.free_hint = std::min(k.free_hint, j);
                    return (int) i;
                }
            }
            if (sp.pin > 0 || sp.visible || sp.state.load(std::memory_order_acquire) != kReady) continue;
            const double v = score(sp);
            if (best < 0 || v < bv) {
                best = (int) i;
                bv = v;
            }
        }
        if (best < 0) return -1;
        k.free_hint = std::min<size_t>(k.free_hint, (size_t) best - k.off);
        return best;
    }

    Cfg cfg_;
    std::vector<LayerGeo> geo_;
    std::vector<os::fd_t> dfds_, pfds_;
    // a slot class: its slots [off, off + nmax) of slots_, their size, and its own arena, chunks and free hint
    struct Cls {
        size_t slot_bytes = 0, nmax = 0, nalloc = 0, free_hint = 0, off = 0, arena_bytes = 0, reserve = 0;
        uint8_t* arena = nullptr;
        std::vector<std::pair<void*, size_t>> chunks;
        std::vector<void*> chunk_dev;
    };
    std::vector<Cls> cls_;
    std::vector<int> layer_cls_;   // a layer's class (-1: none, no blob)
    size_t nmax_ = 0;              // all classes' slots
    size_t shrunk_ = 0;
    std::unique_ptr<Slot[]> slots_;
    std::vector<std::vector<int>> where_;
    std::vector<std::vector<float>> heat_;
    std::vector<uint64_t> hits_in_layer_;
    uint64_t clock_ = 0;
    std::mutex mu_;
    std::unique_ptr<IoQueue> io_;
    Stats st_;
};

}  // namespace strata::ds4
