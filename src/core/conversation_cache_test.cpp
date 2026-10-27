#include "strata/core/conversation_cache.hpp"

#include <cstdio>
#include <cstdlib>
#include <vector>

using namespace strata::core;

namespace {
int checks = 0;
void check(bool value, const char* description) {
    ++checks;
    if (!value) { std::fprintf(stderr, "FAIL: %s\n", description); std::exit(1); }
}
SavedConversation image(std::initializer_list<int32_t> ids, bool cvec = true) {
    SavedConversation s;
    s.live.ids = ids;
    s.live.gdn.resize(64, 7);
    s.cvec = cvec;
    return s;
}
}

int main() {
    {
        ConversationBuffer bytes;
        const size_t first = ConversationBuffer::segment_bytes + 17;
        const size_t peak = bytes.allocation_peak(first);
        bytes.resize(first, 7);
        check(bytes.bytes() <= peak, "segmented payload and directory fit admitted bytes");
        uint8_t* original = nullptr;
        bytes.visit(0, 1, [&](uint8_t* p, size_t, size_t) { original = p; return true; });
        const size_t grown_peak = bytes.allocation_peak(first + 99);
        bytes.resize(first + 99, 9);
        bytes.visit(0, 1, [&](uint8_t* p, size_t, size_t) {
            check(p == original, "appending preserves existing payload addresses"); return true;
        });
        check(bytes.bytes() <= grown_peak, "growth including transient directory fits admission");
        std::vector<uint8_t> tail(116);
        check(bytes.read(tail.data(), ConversationBuffer::segment_bytes, tail.size()), "read spans segment boundaries");
        check(std::all_of(tail.begin(), tail.begin()+17, [](auto v){return v==7;}) &&
              std::all_of(tail.begin()+17, tail.end(), [](auto v){return v==9;}), "growth preserves prefix and initializes only suffix");
        auto copy = bytes;
        bytes.resize(ConversationBuffer::segment_bytes + 10);
        bytes.resize(first, 7); bytes.resize(first + 99, 9);
        check(bytes == copy, "equality ignores differing segmentation after rewind and regrowth");
        check(!bytes.read(tail.data(), bytes.size()-1, 2), "range check rejects a truncated payload");
        check(bytes.allocation_peak(SIZE_MAX) == SIZE_MAX, "allocation estimate rejects overflow");
    }
    {
        ConversationBuffer bytes;
        for (size_t n=1;n<=4096;++n) {
            const size_t peak = bytes.allocation_peak(n*17);
            bytes.resize(n*17,7);
            check(bytes.bytes() <= peak,"small append stays within the predicted allocation peak");
        }
        size_t segments = 0;
        bytes.visit(0,bytes.size(),[&](const uint8_t*,size_t,size_t){++segments;return true;});
        check(segments <= 4,"thousands of small turns do not create thousands of restore transfers");
    }
    {
        // a parked conversation's K/V buffer (16 MiB, as captured) grows by one turn: the reserve for later turns is
        // an eighth of the buffer at most, not another 16 MiB segment
        ConversationBuffer bytes;
        bytes.resize(ConversationBuffer::segment_bytes, 3);
        const size_t captured = bytes.bytes();
        check(captured - bytes.size() < 1024, "a fresh capture reserves nothing");
        const size_t turn = 1 << 20;
        const size_t peak = bytes.allocation_peak(bytes.size() + turn);
        bytes.resize(bytes.size() + turn, 4);
        check(bytes.bytes() <= peak, "a turn's growth stays within the predicted allocation peak");
        check(bytes.bytes() - captured <= turn + ConversationBuffer::segment_bytes / 8 + 1024,
              "the reserve for later turns is at most an eighth of the buffer");
        const size_t reserved = bytes.bytes();
        bytes.resize(bytes.size() + turn / 2, 5);
        check(bytes.bytes() == reserved, "the next small turn fills the reserve instead of allocating");
        size_t segments = 0;
        bytes.visit(0, bytes.size(), [&](const uint8_t*, size_t, size_t) { ++segments; return true; });
        check(segments == 2, "the reserve extends the turn's segment");
    }
    {
        // many turns of 1 MiB onto a 16 MiB buffer: the reserve grows with the buffer, so the segments do not
        // multiply one per turn, and the reserve never exceeds an eighth of the buffer
        ConversationBuffer bytes;
        bytes.resize(ConversationBuffer::segment_bytes, 1);
        for (int k = 0; k < 256; ++k) {
            const size_t before = bytes.size();
            const size_t peak = bytes.allocation_peak(before + (1 << 20));
            bytes.resize(before + (1 << 20), 2);
            check(bytes.bytes() <= peak, "each turn stays within the predicted allocation peak");
            check(bytes.bytes() - bytes.size() <= before / 8 + 4096, "the reserve stays within an eighth");
        }
        size_t segments = 0;
        bytes.visit(0, bytes.size(), [&](const uint8_t*, size_t, size_t) { ++segments; return true; });
        check(segments <= 64, "256 turns of 1 MiB take far fewer than 256 segments");
    }
    {
        ConversationKv layer;
        layer.k.resize(400);
        std::vector<ConversationKv> layers;
        layers.push_back(std::move(layer));
        const size_t retained = layers.capacity()*sizeof(ConversationKv) + layers[0].bytes();
        const size_t parked = image({1,2,3}).bytes();
        ConversationCache cache(retained + parked, 4);
        check(cache.put(image({1,2,3})), "park before retaining active storage");
        cache.retain(std::move(layers), 16);
        check(cache.bytes() == retained + parked && cache.size() == 1, "active retained storage consumes bytes but no parked slot");
        cache.limit_reuse(9); cache.limit_reuse(12);
        auto reuse = cache.take_reuse();
        check(reuse.unchanged_tokens == 9, "a later continuation cannot undo a rewind's dirty boundary");
        check(cache.bytes() == parked, "taking retained buffers releases their budget accounting");
        cache.retain(std::move(reuse.kv), 16);
        check(!cache.can_fit(parked),"retained storage is counted when checking a capture without eviction");
        check(cache.retained_bytes() == retained && cache.size() == 1 && cache.evictions() == 0,
              "optional reuse admission does not evict or release anything");
        check(!cache.can_fit(SIZE_MAX) && !cache.can_fit(1,SIZE_MAX),"non-mutating reservation rejects overflow");
        check(cache.make_room(parked), "reservation can discard optional active buffers");
        check(cache.retained_bytes() == 0 && cache.size() == 1 && cache.evictions() == 0,
              "pressure drops retained storage before evicting parked conversations");
    }
    const std::vector<int64_t> a = {1, 2, 3, 4}, b = {9, 8, 7, 6};
    {
        ConversationCache cache(1024, 2);
        check(cache.put(image({1, 2, 3})), "park A");
        check(cache.put(image({9, 8, 7})), "park B");
        auto match = cache.best(a, {}, true);
        check(match.tokens == 3 && match.live, "A/B/A: recover A");
        auto restored = cache.take(match.index);
        check(restored.live.ids == std::vector<int32_t>({1, 2, 3}), "taking selected A preserves identity");
        check(cache.size() == 1 && cache.best(b, {}, true).tokens == 3, "B remains parked");
        check(cache.bytes() == image({9,8,7}).bytes(), "byte accounting after take");
        check(cache.put(std::move(restored)), "park returned A as newest");
        check(cache.put(image({5, 6})), "evict oldest by slot limit");
        check(cache.best(b, {}, true).tokens == 0 && cache.best(a, {}, true).tokens == 3, "B evicted before A");
        check(cache.evictions() == 1, "eviction counter");
    }
    {
        // evict_oldest(): the physical-RAM admission gate in generate.cpp frees parked conversations
        // one at a time and re-checks the host's free memory, so this has to be exactly the oldest-first
        // step make_room() takes - same accounting, same counters - and it has to stop at empty.
        ConversationCache cache(4096, 4);
        cache.put(image({1, 2, 3}));
        cache.put(image({9, 8, 7}));
        check(cache.size() == 2 && cache.slots() == 4, "slots() reports the configured limit");
        const size_t both = cache.bytes();
        check(cache.evict_oldest(), "evict_oldest reports an eviction");
        check(cache.size() == 1 && cache.evictions() == 1, "evict_oldest drops exactly one, oldest first");
        check(cache.best(a, {}, true).tokens == 0 && cache.best(b, {}, true).tokens == 3,
              "the oldest conversation went, the newest stayed");
        check(cache.bytes() == both - image({1, 2, 3}).bytes(), "evict_oldest releases the entry's bytes");
        cache.evict_oldest();
        check(cache.size() == 0 && cache.bytes() == 0, "evicting the last parked conversation empties the cache");
        check(!cache.evict_oldest(), "the gate's loop can reach an empty cache: false");
        check(cache.size() == 0 && cache.evictions() == 2, "evict_oldest on an empty cache is a no-op");
    }
    {   // the RAM gate's eviction skips a conversation that holds a pinned shared prefix, and says so (false)
        ConversationCache cache(1 << 20, 4);
        auto pinned = image({1, 2, 3});
        pinned.checkpoints.push_back(ConversationCheckpoint{});
        pinned.checkpoints.back().ids = {1, 2};
        pinned.checkpoints.back().pinned = true;
        pinned.checkpoints.back().gdn.resize(64, 1);
        cache.put(std::move(pinned));
        cache.put(image({9, 8, 7}));
        check(cache.evict_oldest() && cache.size() == 1, "the oldest unpinned conversation went (the pinned one is older)");
        check(!cache.evict_oldest() && cache.size() == 1, "only a pinned one is left: nothing more can go");
        // #879: after non-finite logits every parked conversation goes, the pinned one too
        cache.put(image({4, 5, 6}));
        std::vector<ConversationKv> kv(1);
        kv[0].cells = 3;
        kv[0].k.resize(64, 1);
        cache.retain(std::move(kv), 3);
        check(cache.retained_bytes() > 0, "a retained K/V before clear()");
        check(cache.clear() == 2, "clear() reports how many were parked");
        check(cache.size() == 0 && cache.bytes() == 0 && cache.retained_bytes() == 0, "clear() leaves nothing parked");
        check(cache.best(std::vector<int64_t>{1, 2, 3, 4}, {}, true).tokens == 0, "a cleared cache matches nothing");
        check(cache.clear() == 0, "clear() on an empty cache");
    }
    {
        auto s = image({1, 2, 3});
        ConversationCheckpoint cp;
        cp.ids = {1, 2};
        s.checkpoints.push_back(cp);
        ConversationCache cache(4096, 4);
        cache.put(std::move(s));
        auto match = cache.best(std::vector<int64_t>{1, 2, 9, 4}, {}, true);
        check(match.tokens == 2 && !match.live, "edited suffix falls back to parked checkpoint");
        check(cache.best(a, {}, true).tokens == 3, "live prefix beats shorter checkpoint");
        check(cache.best(a, {}, false).tokens == 0, "steering mode is isolated");
        check(cache.best(std::vector<int64_t>{1, 2}, {}, true).tokens == 0, "equal-length checkpoint cannot consume last token");
        check(cache.best(std::vector<int64_t>{1}, {}, true).tokens == 0, "short prompt cannot match");
        check(cache.best(std::vector<int64_t>{}, {}, true).tokens == 0, "empty prompt cannot match");
        cache.put(image({1, 2, 3}));
        check(cache.best(a, {}, true).index == 1, "ties prefer most recently parked");
    }
    {
        auto s = image({1, 2, 3});
        s.live.imgs = {{1, 123}};
        ConversationCache cache(1024, 3);
        cache.put(std::move(s));
        check(cache.best(a, {{1,123}}, true).tokens == 3, "same image can resume");
        check(cache.best(a, {{1,124}}, true).tokens == 0, "different image pixels invalidate same pad tokens");
        check(cache.best(a, {}, true).tokens == 0, "missing image invalidates prefix");
        check(cache.best(a, {{1,123},{2,45}}, true).tokens == 0, "additional image in cached prefix invalidates");
        check(cache.best(a, {{1,123},{3,45}}, true).tokens == 3, "image after cached prefix does not invalidate");
    }
    {
        const size_t one = image({1,2,3}).bytes();
        ConversationCache cache(one*2, 8);
        cache.put(image({1,2,3})); cache.put(image({9,8,7}));
        check(cache.bytes() == one*2, "budget holds two exact-sized images");
        auto held = cache.take(cache.best(a, {}, true).index);
        check(cache.make_room(one, held.bytes()), "count in-flight image when reserving outgoing snapshot");
        check(cache.size() == 0, "in-flight reservation evicts otherwise fitting B");
        check(cache.put(image({5,6,7}), held.bytes()), "insert with in-flight accounting");
        check(cache.bytes()+held.bytes() <= one*2, "exchange obeys byte budget");
        check(!cache.make_room(one+1, one), "oversized exchange rejected");
        check(cache.size() == 1, "oversized snapshot does not evict useful entries");
        auto huge = image({1}); huge.live.gdn.resize(one*3);
        check(!cache.put(std::move(huge)), "oversized image rejected");
        check(cache.size() == 1, "oversized put leaves cache unchanged");
        check(!cache.make_room(0, one*2+1), "held larger than budget cannot underflow");
    }
    {
        // #342: MAIN -> SUB (9 turns) -> MAIN with 4 slots.  Every SUB turn parks the previous turn's live state
        // (its reply as generated, which the next request re-rendered) with a chain that holds that turn's
        // boundary checkpoint; the copy a turn back adds only that stale tail and is dropped, so MAIN survives.
        auto cp = [](std::vector<int32_t> ids) { ConversationCheckpoint c; c.ids = std::move(ids); return c; };
        auto with = [](std::vector<int32_t> prefix, std::initializer_list<int32_t> more) {
            prefix.insert(prefix.end(), more); return prefix;
        };
        const std::vector<int32_t> root = {1, 2, 3, 4};
        SavedConversation main = image({});
        main.live.ids = with(with(root, {10, 11, 12}), {13, 14});
        main.checkpoints = {cp(root), cp(with(root, {10, 11, 12}))};
        const size_t big = 1 << 20;
        ConversationCache cache(big, 4);
        check(cache.put(std::move(main)), "park MAIN");
        std::vector<int32_t> history = with(root, {20});         // SUB's conversation so far, re-rendered
        std::vector<ConversationCheckpoint> chain = {cp(root)};
        for (int turn = 1; turn <= 9; ++turn) {
            chain.push_back(cp(history));                        // the turn boundary the next request resumes from
            SavedConversation sub = image({});
            sub.live.ids = with(history, {900, (int32_t) turn});   // + the reply with its thinking (stale tail)
            sub.checkpoints = chain;
            check(cache.put(std::move(sub)), "park SUB turn");
            check(cache.size() <= 2, "one parked copy of SUB at a time");
            history = with(history, {30, (int32_t) turn});       // the reply as the next request renders it
        }
        check(cache.evictions() == 0 && cache.superseded() == 8, "8 superseded copies dropped, nothing evicted");
        const auto m = cache.best(with(with(root, {10, 11, 12}), {13, 14, 15}), {}, true);
        check(m.tokens == 9 && m.live, "MAIN still restores in full");
        const auto s = cache.best(with(history, {40}), {}, true);
        check(s.tokens == (int64_t) history.size() - 2, "SUB resumes from its last turn boundary");
    }
    {
        // holds(): what an idle batch slot holds is parked already (or a continuation of it is)
        auto s = image({1, 2, 3, 4, 5});
        ConversationCheckpoint cp;
        cp.ids = {1, 2, 3};
        s.checkpoints.push_back(cp);
        ConversationCache cache(4096, 4);
        cache.put(std::move(s));
        check(cache.holds(std::vector<int32_t>{1, 2, 3, 4, 5}, {}, true), "the parked live state itself");
        check(cache.holds(std::vector<int32_t>{1, 2, 3, 4}, {}, true), "a prefix of the parked live state");
        check(cache.holds(std::vector<int32_t>{1, 2}, {}, true), "a prefix of a parked checkpoint");
        check(!cache.holds(std::vector<int32_t>{1, 2, 3, 4, 5, 6}, {}, true), "a longer state is not held");
        check(!cache.holds(std::vector<int32_t>{1, 2, 9}, {}, true), "a diverging state is not held");
        check(!cache.holds(std::vector<int32_t>{}, {}, true), "an empty state is never held");
        check(!cache.holds(std::vector<int32_t>{1, 2, 3}, {}, false), "steering mode is isolated");
        check(!cache.holds(std::vector<int32_t>{1, 2, 3}, {{1, 77}}, true), "a slot with pictures is not held by a text image");
        auto p = image({1, 2, 3, 4});
        p.live.imgs = {{1, 77}, {3, 78}};
        cache.put(std::move(p));
        check(cache.holds(std::vector<int32_t>{1, 2, 3}, {{1, 77}}, true), "pictures below the state must match");
        check(!cache.holds(std::vector<int32_t>{1, 2, 3, 4}, {{1, 77}}, true), "a picture inside the state that the slot lacks");
    }
    {
        // The 2026-10-14 four-conversation measurement replayed at policy level: two batch slots, four parked
        // entries, conversations A..D.  Each has a prompt P (8 tokens; the read's periodic checkpoint at 4), a first
        // reply R1 (3 tokens; the slot holds P + 2 of them), the next turn as the client renders it (P + R1 + end +
        // header: 14 tokens, the read's turn checkpoint at 12), a second reply R2 (4 tokens; the slot holds 17) and
        // a third turn (21 tokens).  The engine parks the MAIN session (the prompt as read, plus the admission's
        // token) whenever another conversation displaces it; the replies are decoded in the slots.  Measured:
        // A's third turn found 199584 of 199839 held tokens (its prompt-read state), B found nothing.
        //   slot_parks:   an idle slot parks what it holds before an admission overwrites it
        //   checks_back:  a slot that gives "all it holds" back to the main session gives its checkpoints with it,
        //                 so the chain still supersedes (#342) the parked copies a turn back
        auto replay = [&](bool slot_parks, bool checks_back) {
            std::vector<std::vector<int32_t>> full(4);
            for (int c = 0; c < 4; ++c) {
                for (int32_t t = 1; t <= 8; ++t) full[(size_t) c].push_back(c * 100 + t);
                for (int32_t t : {21, 22, 23, 9, 31, 32, 41, 42, 43, 44, 9, 51, 52}) full[(size_t) c].push_back(c * 100 + t);
            }
            auto st = [&](int c, size_t n) { return std::vector<int32_t>(full[(size_t) c].begin(), full[(size_t) c].begin() + (std::ptrdiff_t) n); };
            auto cp = [&](int c, size_t n) { ConversationCheckpoint k; k.ids = st(c, n); return k; };
            auto park = [&](ConversationCache& cache, int c, size_t n, const std::vector<size_t>& cks) {
                SavedConversation s = image({});
                s.live.ids = st(c, n);
                for (size_t k : cks) s.checkpoints.push_back(cp(c, k));
                check(cache.put(std::move(s)), "the replay's parks all fit the budget");
            };
            auto restore = [&](ConversationCache& cache, int c, size_t prompt) -> int64_t {
                const auto m = cache.best(st(c, prompt), {}, true);
                if (m.tokens > 0) cache.take(m.index);
                return m.tokens;
            };
            // the main chain after a slot gave "all it holds" back: its checkpoints, or none
            auto back = [&](const std::vector<size_t>& slot_cks) { return checks_back ? slot_cks : std::vector<size_t>{}; };
            auto slot_park = [&](ConversationCache& cache, int c, size_t n, const std::vector<size_t>& cks) {
                if (slot_parks && !cache.holds(st(c, n), {}, true)) park(cache, c, n, cks);
            };
            enum { A, B, C, D };
            ConversationCache cache(1 << 20, 4);
            // initial-AB: B solo then admitted (its prompt + the first token parked when A's request came), A
            // admitted directly, then alone: solo again from its slot (A's prompt-read state parked first)
            park(cache, B, 9, {4});
            park(cache, A, 8, {4});
            std::vector<size_t> chain_a = back({4});           // slot 1 gave back A's 10 tokens
            // resident1-AB: B's request displaces A's solo state; B's slot gives back 10; B admitted with the turn
            // checkpoint; A's request restores A's solo state from RAM and is admitted beside B
            park(cache, A, 12, chain_a);
            std::vector<size_t> chain_b = back({4});
            chain_b.push_back(12);
            // (a request takes its parked conversation before it parks the one it displaces)
            check(restore(cache, A, 14) == 12, "A's second turn resumes from its parked solo state");
            park(cache, B, 15, chain_b);
            chain_a.push_back(12);
            // both decode: slot 0 holds B's 17, slot 1 holds A's 17
            // initial-CD: C's request displaces A's main state; D admitted into slot 0 (B's reply), C into slot 1
            park(cache, A, 14, chain_a);
            slot_park(cache, B, 17, {12});
            park(cache, D, 9, {4});
            slot_park(cache, A, 17, {12});
            park(cache, C, 8, {4});                              // C alone: solo again from its slot
            std::vector<size_t> chain_c = back({4});
            // resident1-CD: C continues from the main session (no park), D's request displaces it (slot 1 held C's
            // own earlier state: covered, no slot park), D's slot gives back 10 (covered as well)
            chain_c.push_back(12);
            park(cache, C, 15, chain_c);
            std::vector<size_t> chain_d = back({4});
            chain_d.push_back(12);
            // both decode: slot 0 holds D's 17, slot 1 holds C's 17
            // exchange1-AB: A's request takes A's parked state, displaces D's main state, is admitted into slot 1
            // (C's reply); B's request displaces A's state
            const int64_t a_found = restore(cache, A, 21);
            park(cache, D, 14, chain_d);
            slot_park(cache, C, 17, {12});
            park(cache, A, 22, {12, 20});
            const int64_t b_found = restore(cache, B, 21);
            return std::pair<int64_t, int64_t>(a_found, b_found);
        };
        const auto baseline = replay(false, false);
        check(baseline.first == 14 && baseline.second == 15, "baseline: A returns to its prompt-read state, B to its "
              "parked turn state - smallest-first eviction drops the stale initial images instead of B's copy");
        const auto parks_only = replay(true, false);
        check(parks_only.first == 17 && parks_only.second == 17, "slot parks alone: both return whole");
        const auto both = replay(true, true);
        check(both.first == 17 && both.second == 17, "slot parks + checkpoints back: both return whole");
    }
    {
        // what is NOT superseded: another conversation sharing only the root, an entry without checkpoints, the
        // other steering mode, a branch whose deepest checkpoint the new chain does not hold
        auto cp = [](std::vector<int32_t> ids) { ConversationCheckpoint c; c.ids = std::move(ids); return c; };
        ConversationCache cache(1 << 20, 8);
        SavedConversation other = image({1, 2, 3, 4, 50, 51, 52});
        other.checkpoints = {cp({1, 2, 3, 4}), cp({1, 2, 3, 4, 50, 51})};
        cache.put(std::move(other));
        cache.put(image({1, 2, 3, 4, 7, 7}));                    // no checkpoints
        SavedConversation steered = image({1, 2, 3, 4, 60, 61}, false);
        steered.checkpoints = {cp({1, 2, 3, 4, 60})};
        cache.put(std::move(steered));
        SavedConversation branch = image({1, 2, 3, 4, 60, 70, 71});
        branch.checkpoints = {cp({1, 2, 3, 4}), cp({1, 2, 3, 4, 60, 70})};
        cache.put(std::move(branch));
        SavedConversation incoming = image({1, 2, 3, 4, 60, 80, 81});
        incoming.checkpoints = {cp({1, 2, 3, 4}), cp({1, 2, 3, 4, 60})};
        check(cache.put(std::move(incoming)), "park a conversation sharing roots with all of them");
        check(cache.size() == 5 && cache.superseded() == 0, "none of them is superseded");
        SavedConversation same_state = image({1, 2, 3, 4, 60, 70});   // live equal to the branch's deepest point
        same_state.checkpoints = {cp({1, 2, 3, 4})};
        check(cache.put(std::move(same_state)) && cache.superseded() == 1 && cache.size() == 5,
              "the live state equal to an entry's deepest checkpoint supersedes it");
        SavedConversation huge = image({1, 2, 3, 4, 60, 80});
        huge.checkpoints = {cp({1, 2, 3, 4}), cp({1, 2, 3, 4, 60})};
        huge.live.gdn.resize(2 << 20);
        check(!cache.put(std::move(huge)) && cache.size() == 5 && cache.superseded() == 1,
              "an oversized put drops nothing");
    }
    {
        // --conversation-cache-park-gain N: parked_prefix_tokens reports what a park now would supersede, so a
        // conversation that advanced fewer than N tokens since that copy keeps it instead of writing the whole
        // snapshot again for the next turn to drop.  It must name exactly the entries drop_superseded() would remove.
        auto cp = [](std::vector<int32_t> ids) { ConversationCheckpoint c; c.ids = std::move(ids); return c; };
        ConversationCache cache(1 << 20, 8);
        check(cache.parked_prefix_tokens({1, 2, 3}, {}, {}, true) == 0, "an empty cache supersedes nothing");

        const std::vector<int32_t> live_ids = {1, 2, 3, 4, 60, 70};
        const std::vector<ConversationCheckpoint> chain = {cp({1, 2, 3, 4}), cp({1, 2, 3, 4, 60})};
        SavedConversation same = image({1, 2, 3, 4, 60, 70});
        same.checkpoints = chain;
        check(cache.put(std::move(same)), "park a conversation with a checkpoint chain");
        check(cache.parked_prefix_tokens(live_ids, {}, chain, true) == 5,
              "the entry the outgoing chain supersedes is reported by its deepest checkpoint");

        const std::vector<ConversationCheckpoint> other_chain = {cp({1, 2, 3})};
        check(cache.parked_prefix_tokens({1, 2, 3, 40, 50}, {}, other_chain, true) == 0,
              "a conversation sharing only the root supersedes nothing");
        check(cache.parked_prefix_tokens(live_ids, {}, chain, false) == 0,
              "the other steering mode is a different entry");
        check(cache.parked_prefix_tokens({1, 2, 3, 4, 60, 70, 71}, {}, chain, true) == 5,
              "an appended tail keeps the reported depth at the parked one");
        SavedConversation no_cp = image({1, 2, 3, 9});
        check(cache.put(std::move(no_cp)) && cache.parked_prefix_tokens({1, 2, 3, 9}, {}, {}, true) == 0,
              "an entry without checkpoints is never reported");
    }
    {
        // borrowing: a NEW conversation that starts with a checkpoint inside a parked one (a sibling subagent with
        // the same system prompt and tools) restores from it and leaves it parked; the same conversation going on
        // (its live state, or its deepest checkpoint after the client re-rendered the last reply) is still taken
        auto cp = [](std::vector<int32_t> ids) { ConversationCheckpoint c; c.ids = std::move(ids); return c; };
        SavedConversation a = image({});
        a.live.ids = {1, 2, 3, 4, 10, 11, 12, 13};
        a.checkpoints = {cp({1, 2, 3, 4}), cp({1, 2, 3, 4, 10, 11})};
        ConversationCache cache(1 << 20, 2);
        check(cache.put(std::move(a)), "park subagent A");
        const auto sib = cache.best(std::vector<int64_t>{1, 2, 3, 4, 20, 21}, {}, true);
        check(sib.tokens == 4 && !sib.live && sib.serial != 0 && cache.borrows(sib), "a sibling on A's root borrows");
        const auto again = cache.best(std::vector<int64_t>{1, 2, 3, 4, 10, 11, 30}, {}, true);
        check(again.tokens == 6 && !again.live && !cache.borrows(again),
              "A re-rendered at its deepest checkpoint is taken");
        const auto cont = cache.best(std::vector<int64_t>{1, 2, 3, 4, 10, 11, 12, 13, 14}, {}, true);
        check(cont.live && !cache.borrows(cont), "A going on is taken");
        check(!cache.borrows(cache.best(std::vector<int64_t>{5, 6, 7}, {}, true)), "no match, nothing to borrow");
        // while the outgoing conversations are parked, the guarded donor is neither evicted nor dropped
        cache.guard(sib.serial);
        check(cache.put(image({7, 7, 7})) && cache.put(image({8, 8, 8})), "two outgoing conversations park");
        check(cache.find(sib.serial) != nullptr && cache.size() == 2 && cache.evictions() == 1 && !cache.guard_blocked(),
              "the slot limit evicts the other entry, not the guarded donor");
        check(cache.best(std::vector<int64_t>{7, 7, 7, 1}, {}, true).tokens == 0, "the other one went");
        SavedConversation rewritten = image({1, 2, 3, 4, 10, 11, 40});   // holds A's deepest checkpoint: supersedes A
        rewritten.checkpoints = {cp({1, 2, 3, 4}), cp({1, 2, 3, 4, 10, 11})};
        check(cache.put(std::move(rewritten)) && cache.find(sib.serial) != nullptr && cache.superseded() == 0,
              "a guarded donor is not dropped as superseded");
        cache.unguard();
        // identity survives index shifts: take an older entry, the donor's serial still finds it
        const auto donor_index = cache.index_of(sib.serial);
        check(donor_index != SIZE_MAX, "donor indexed by its serial");
        check(cache.best(std::vector<int64_t>{1, 2, 3, 4, 10, 11, 12, 13, 14}, {}, true).serial == sib.serial,
              "matches report the entry's serial");
    }
    {
        // the outgoing conversation fits only in the guarded donor's room: make_room refuses before evicting anything
        auto cp = [](std::vector<int32_t> ids) { ConversationCheckpoint c; c.ids = std::move(ids); return c; };
        SavedConversation a = image({1, 2, 3, 4, 10, 11});
        a.checkpoints = {cp({1, 2, 3, 4}), cp({1, 2, 3, 4, 10})};
        const size_t a_bytes = a.bytes();
        const size_t out_bytes = image({9, 9, 9}).bytes();
        ConversationCache tight(a_bytes + out_bytes - 1, 4);
        check(tight.put(std::move(a)), "park the donor");
        const auto m = tight.best(std::vector<int64_t>{1, 2, 3, 4, 50}, {}, true);
        check(tight.borrows(m), "a sibling would borrow");
        tight.guard(m.serial);
        check(!tight.put(image({9, 9, 9})) && tight.guard_blocked(), "no room beside the guarded donor: refused");
        check(tight.size() == 1 && tight.evictions() == 0, "nothing evicted by the refusal");
        tight.unguard();
        auto taken = tight.take(tight.index_of(m.serial));   // what the caller does then: take it, as before
        check(taken.live.ids.size() == 6 && taken.checkpoints.size() == 2 && tight.size() == 0,
              "the taken donor is whole");
        ConversationCache one_slot(1 << 20, 1);
        SavedConversation b = image({1, 2, 3, 4, 10, 11});
        b.checkpoints = {cp({1, 2, 3, 4}), cp({1, 2, 3, 4, 10})};
        check(one_slot.put(std::move(b)), "park in a single slot");
        const auto mb = one_slot.best(std::vector<int64_t>{1, 2, 3, 4, 50}, {}, true);
        one_slot.guard(mb.serial);
        check(!one_slot.put(image({9, 9, 9})) && one_slot.guard_blocked() && one_slot.size() == 1,
              "one slot: the guarded donor blocks, nothing evicted");
        one_slot.unguard();
    }
    {
        // pin=N beside a borrow: the guarded donor and a conversation holding a pinned shared prefix both stay; the
        // oldest other entry is the one evicted (the case midhatn tested on #1164)
        auto cp = [](std::vector<int32_t> ids, bool pinned = false) {
            ConversationCheckpoint c; c.ids = std::move(ids); c.pinned = pinned; return c;
        };
        SavedConversation doc = image({1, 2, 3, 4, 5, 6, 30, 31});
        doc.checkpoints = {cp({1, 2, 3, 4, 5, 6}, true), cp({1, 2, 3, 4, 5, 6, 30})};
        SavedConversation donor = image({1, 2, 10, 11, 12});
        donor.checkpoints = {cp({1, 2}), cp({1, 2, 10, 11})};
        ConversationCache cache(1 << 20, 3);
        check(cache.put(std::move(doc)) && cache.put(std::move(donor)) && cache.put(image({7, 7, 7})),
              "pinned document, donor and an ordinary conversation parked");
        const auto m = cache.best(std::vector<int64_t>{1, 2, 40, 41}, {}, true);
        check(cache.borrows(m), "a sibling on the donor's root borrows");
        cache.guard(m.serial);
        check(cache.put(image({8, 8, 8})) && !cache.guard_blocked() && cache.size() == 3 && cache.evictions() == 1,
              "the outgoing conversation parks");
        check(cache.find(m.serial) != nullptr &&
              cache.best(std::vector<int64_t>{1, 2, 3, 4, 5, 6, 9}, {}, true).tokens == 6 &&
              cache.best(std::vector<int64_t>{7, 7, 7, 1}, {}, true).tokens == 0,
              "only the ordinary entry was evicted: the donor and the pinned document stay");
        check(cache.put(image({9, 9, 9})) && cache.size() == 3 && cache.evictions() == 2 &&
              cache.find(m.serial) != nullptr && cache.best(std::vector<int64_t>{8, 8, 8, 1}, {}, true).tokens == 0,
              "the next outgoing one evicts the previous outgoing one, still not the donor or the document");
        cache.unguard();
        ConversationCache two(1 << 20, 2);
        SavedConversation doc2 = image({1, 2, 3, 4, 5, 6, 30});
        doc2.checkpoints = {cp({1, 2, 3, 4, 5, 6}, true)};
        SavedConversation donor2 = image({1, 2, 10, 11, 12});
        donor2.checkpoints = {cp({1, 2}), cp({1, 2, 10, 11})};
        check(two.put(std::move(doc2)) && two.put(std::move(donor2)), "pinned document and donor fill two slots");
        const auto m2 = two.best(std::vector<int64_t>{1, 2, 40}, {}, true);
        two.guard(m2.serial);
        check(!two.put(image({9, 9, 9})) && two.guard_blocked() && two.size() == 2 && two.evictions() == 0,
              "no slot beside the donor and the pinned document: refused before evicting anything");
        two.unguard();
        // a match that reaches the pinned prefix is taken, not borrowed: one conversation keeps holding the pin
        const auto on_pin = two.best(std::vector<int64_t>{1, 2, 3, 4, 5, 6, 77}, {}, true);
        check(on_pin.tokens == 6 && !on_pin.live && !two.borrows(on_pin), "a pinned prefix is taken, not borrowed");
        ConversationCache three(1 << 20, 3);
        SavedConversation doc3 = image({1, 2, 3, 4, 5, 6, 30, 31, 32});
        doc3.checkpoints = {cp({1, 2, 3, 4, 5, 6}, true), cp({1, 2, 3, 4, 5, 6, 30}), cp({1, 2, 3, 4, 5, 6, 30, 31})};
        check(three.put(std::move(doc3)), "a conversation past its pinned prefix is parked");
        const auto past_pin = three.best(std::vector<int64_t>{1, 2, 3, 4, 5, 6, 30, 99}, {}, true);
        check(past_pin.tokens == 7 && !three.borrows(past_pin),
              "a match past the pinned prefix (the copy would carry the pin) is taken, not borrowed");
    }
    {
        ConversationCache disabled(0,4), no_slots(1024,0);
        check(!disabled.enabled() && !no_slots.enabled(), "both disable switches");
        check(!disabled.put(image({1,2,3})) && !no_slots.put(image({1,2,3})), "disabled cache stores nothing");
        check(disabled.best(a,{},true).tokens == 0, "disabled cache has no matches");
    }
    {
        // --conversation-cache-min-tokens: a conversation shorter than the minimum is not worth one of the slots.
        // Parking is a switch-time decision, so without this a client's short side requests park themselves and,
        // once the cache is full, evict the long conversation they were interleaved with (oldest first).
        ConversationCache any(1024, 4, 0), floor(1024, 4, 12288);
        check(any.enabled() && floor.enabled(), "the minimum does not disable the cache");
        check(any.wants(0) && any.wants(1) && any.wants(1 << 20), "0 parks every length, as before the flag");
        check(!floor.wants(0) && !floor.wants(1) && !floor.wants(12287), "below the minimum is not parked");
        check(floor.wants(12288) && floor.wants(12289), "the minimum itself and above are parked");
        // the CLI rejects a negative, but the predicate must not invert for one that reaches it another way
        ConversationCache negative(1024, 4, -1);
        check(negative.wants(0) && negative.wants(1), "a negative minimum parks everything, like 0");
        // wants() answers for the length alone; the gate is enabled() && wants(), so a disabled cache parks
        // nothing whatever the length - keep the two independent rather than folding one into the other
        ConversationCache disabled_min(0, 4, 12288);
        check(!disabled_min.enabled() && disabled_min.wants(1 << 20), "a disabled cache still answers for length");
    }
    {
        // layer-split parking: checkpoints moved apart into stage parts and put back, no running state copied
        auto running = [](std::initializer_list<int32_t> ids, uint8_t salt, size_t parts) {
            ConversationCheckpoint c;
            c.ids = ids;
            c.gdn.assign(256, salt);
            c.tails.assign(32, uint8_t(salt + 1));
            c.used = salt;
            for (size_t k = 0; k < parts; ++k) {
                ConversationCheckpoint p;
                p.gdn.assign(128, uint8_t(salt + 10 + k));
                c.stage_parts.push_back(std::move(p));
            }
            return c;
        };
        std::vector<ConversationCheckpoint> checks;
        checks.push_back(running({1, 2}, 3, 2));
        checks.push_back(running({1, 2, 3}, 5, 1));       // an incomplete set: kept aside, put back in place
        checks.push_back(running({1, 2, 3, 4}, 7, 2));
        const auto original = checks;
        const uint8_t* gdn0 = checks[0].gdn.data();
        const uint8_t* part1 = checks[2].stage_parts[1].gdn.data();
        auto split = conversation_checkpoints_split(std::move(checks), 2);
        check(checks.empty() && split.stage0.size() == 2 && split.rest.size() == 1 && split.parts.size() == 2 &&
              split.parts[0].size() == 2 && split.parts[1].size() == 2, "split: complete checkpoints apart, the other aside");
        check(split.stage0[0].stage_parts.empty() && split.parts[1][1].ids == original[2].ids &&
              split.parts[1][1].used == original[2].used && split.parts[0][0].gdn == original[0].stage_parts[0].gdn,
              "split: every stage part carries its checkpoint's ids and LRU stamp");
        check(split.stage0[0].gdn.data() == gdn0 && split.parts[1][1].gdn.data() == part1,
              "split moves the running state (same buffers, no copy)");
        std::vector<ConversationCheckpoint> back;
        check(conversation_checkpoints_merge(std::move(split), back) && back.size() == 3, "merge puts them back");
        bool same = true;
        for (size_t j = 0; j < back.size(); ++j) {
            same = same && back[j].ids == original[j].ids && back[j].gdn == original[j].gdn &&
                   back[j].tails == original[j].tails && back[j].used == original[j].used &&
                   back[j].stage_parts.size() == original[j].stage_parts.size();
            for (size_t k = 0; same && k < back[j].stage_parts.size(); ++k)
                same = back[j].stage_parts[k].gdn == original[j].stage_parts[k].gdn;
        }
        check(same, "merge(split(x)) == x, in the original order");
        check(back[0].gdn.data() == gdn0 && back[2].stage_parts[1].gdn.data() == part1, "merge moves them back too");

        auto torn = conversation_checkpoints_split(std::vector<ConversationCheckpoint>(original), 2);
        torn.parts[1].pop_back();
        check(!conversation_checkpoints_merge(std::move(torn), back) && back.empty(), "a stage short of a part is refused");
        auto mixed = conversation_checkpoints_split(std::vector<ConversationCheckpoint>(original), 2);
        mixed.parts[0][0].ids = {9, 9};
        check(!conversation_checkpoints_merge(std::move(mixed), back) && back.empty(), "a part of another checkpoint is refused");
        // a restored image: stage 0's checkpoints and each stage's, with no record of an incomplete one
        ConversationCheckpointSplit restored;
        restored.stage0.push_back(running({4, 5}, 11, 0));
        restored.parts.resize(2);
        for (auto& p : restored.parts) { auto c = running({4, 5}, 12, 0); p.push_back(std::move(c)); }
        check(conversation_checkpoints_merge(std::move(restored), back) && back.size() == 1 &&
              back[0].stage_parts.size() == 2, "a restored image's parts are all whole");
        auto none = conversation_checkpoints_split({}, 3);
        check(conversation_checkpoints_merge(std::move(none), back) && back.empty(), "no checkpoints: nothing to do");
    }
    {
        // #1503, the cross-conversation leak: the retained K/V of a restored conversation must never be reused to
        // park ANOTHER conversation the main session later holds (a batch slot gave its conversation back)
        auto kv = [](size_t n) {
            std::vector<ConversationKv> v(2);
            for (auto& l : v) l.k.resize(n, 1);
            return v;
        };
        ConversationCache cache(1 << 20, 4);
        const std::vector<int32_t> a = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
        std::vector<std::vector<ConversationKv>> st;
        st.push_back(kv(100));
        cache.retain(kv(500), 16, std::move(st), a);
        std::vector<int32_t> b(16, 99);
        auto other = cache.take_reuse_for(b);
        check(other.kv.empty() && other.stages.empty() && other.unchanged_tokens == 0,
              "another conversation never gets the retained K/V");
        check(cache.retained_bytes() == 0, "the retained K/V is dropped once refused");
        cache.retain(kv(500), 16, {}, a);
        check(cache.retained_tokens() == 16, "the retained extent is reported");
        std::vector<int32_t> cont(a.begin(), a.begin() + 10);
        cont.push_back(42);
        cont.push_back(43);
        auto partly = cache.take_reuse_for(cont);
        check(partly.unchanged_tokens == 10 && !partly.kv.empty(), "a conversation that diverges reuses only the shared prefix");
        std::vector<std::vector<ConversationKv>> st2;
        st2.push_back(kv(100));
        cache.retain(kv(500), 16, std::move(st2), a);
        cache.limit_reuse(9);
        auto same = cache.take_reuse_for(a);
        check(same.unchanged_tokens == 9 && same.stages.size() == 1 && same.stages[0].unchanged_tokens == 9,
              "the same conversation keeps the first rewrite's limit, on every stage");
        cache.retain(kv(500), 16);
        check(cache.take_reuse_for(a).kv.empty(), "retained K/V of unknown identity is never reused");

        // the sequence of the report: P restored (retained), a slot hands Q to the main session, Q is parked.  Q shares
        // only the system prompt with P (4 tokens): with the slot's drop, nothing is reused; without it (the ids check
        // alone), at most the shared system prompt
        const std::vector<int32_t> p = {7, 7, 7, 7, 100, 101, 102, 103, 104, 105};
        const std::vector<int32_t> q = {7, 7, 7, 7, 200, 201, 202, 203, 204, 205, 206, 207};
        std::vector<std::vector<ConversationKv>> stp;
        stp.push_back(kv(100));
        cache.retain(kv(500), 10, std::move(stp), p);
        (void) cache.take_reuse();   // copy_from_slot
        check(cache.take_reuse_for(q).kv.empty(), "a slot that gave its conversation back drops the retained K/V");
        std::vector<std::vector<ConversationKv>> stp2;
        stp2.push_back(kv(100));
        cache.retain(kv(500), 10, std::move(stp2), p);
        auto sys_only = cache.take_reuse_for(q);
        check(sys_only.unchanged_tokens == 4 && sys_only.stages.size() == 1 && sys_only.stages[0].unchanged_tokens == 4,
              "without the drop, only the shared system prompt is reused, on every stage");
        check(cache.retained_bytes() == 0, "taken once");
        // a conversation that is a strict extension of the retained one keeps the whole extent
        cache.retain(kv(500), 10, {}, p);
        std::vector<int32_t> p_more = p;
        p_more.push_back(300);
        check(cache.take_reuse_for(p_more).unchanged_tokens == 10, "the conversation's own continuation reuses all of it");
    }
    {
        // the later stages' retained K/V, kept with the first stage's and limited together
        auto kv = [](size_t n) {
            std::vector<ConversationKv> v(2);
            for (auto& l : v) l.k.resize(n, 1);
            return v;
        };
        ConversationCache cache(1 << 20, 4);
        std::vector<std::vector<ConversationKv>> stage_kv;
        stage_kv.push_back(kv(1000));
        stage_kv.push_back(kv(2000));
        cache.retain(kv(500), 40, std::move(stage_kv));
        check(cache.retained_bytes() >= 2 * (500 + 1000 + 2000), "retained bytes count every stage");
        cache.limit_reuse(30);
        auto r = cache.take_reuse();
        check(r.kv.size() == 2 && r.stages.size() == 2 && r.unchanged_tokens == 30 &&
              r.stages[0].unchanged_tokens == 30 && r.stages[1].unchanged_tokens == 30 &&
              r.stages[1].captured_tokens == 40, "every stage's reuse is limited to the first rewrite");
        check(cache.retained_bytes() == 0, "taken once");
        std::vector<std::vector<ConversationKv>> big;
        big.push_back(kv(1 << 20));
        cache.retain(kv(500), 40, std::move(big));
        check(cache.retained_bytes() == 0, "a stage's K/V over the budget drops the whole reuse");
        std::vector<std::vector<ConversationKv>> one;
        one.push_back(kv(100));
        cache.retain(kv(500), 40, std::move(one));
        cache.limit_reuse(0);
        check(cache.retained_bytes() == 0, "a rewrite from the start drops every stage's reuse");
        cache.retain(kv(500), 40);
        r = cache.take_reuse();
        check(r.kv.size() == 2 && r.stages.empty(), "no layer split: no stage reuse, as before");
    }
    {   // pin=N: a parked conversation that holds a pinned shared prefix is not the one evicted
        const size_t one = image({1,2,3}).bytes();
        auto pinned = image({1,2,3});
        pinned.checkpoints.push_back(ConversationCheckpoint{});
        pinned.checkpoints.back().ids = {1,2};
        pinned.checkpoints.back().pinned = true;
        pinned.checkpoints.back().gdn.resize(64, 1);
        const size_t pinned_bytes = pinned.bytes();
        ConversationCache big(pinned_bytes + one * 2 + 64, 3);
        check(pinned.pinned() && !image({1,2,3}).pinned(), "a conversation is pinned when one of its checkpoints is");
        check(big.put(std::move(pinned)), "the pinned conversation is parked first (the oldest)");
        check(big.put(image({7,8,9})) && big.put(image({4,5,6})) && big.size() == 3, "two more fill the slots");
        check(big.put(image({10,11,12})) && big.size() == 3 && big.evictions() == 1,
              "a fourth evicts the oldest unpinned one");
        check(big.best(std::vector<int32_t>{1,2,9,9}, {}, true).tokens == 2, "the pinned prefix is still found");
        ConversationCache tiny(pinned_bytes + 8, 2);
        auto p2 = image({1,2,3});
        p2.checkpoints.push_back(ConversationCheckpoint{});
        p2.checkpoints.back().ids = {1,2};
        p2.checkpoints.back().pinned = true;
        p2.checkpoints.back().gdn.resize(64, 1);
        check(tiny.put(std::move(p2)), "a pinned one fits an empty cache");
        check(!tiny.put(image({7,8,9})) && tiny.size() == 1, "nothing else fits beside it: parking is refused, the pin stays");
    }
    {
        // conversation_slot_ceiling_bytes: the documented field sum of SavedConversation::bytes() with every
        // field at the largest it can get - live + kept checkpoints each carrying one full running state and
        // ids/imgs bounded at ctx, the kv at the part sizes, the vector directories, the geometry array and the
        // struct body.  Zero inputs leave only the fixed terms: the caller treats 0 as "no ceiling".
        const size_t img = sizeof(ConversationImageKey);
        auto sum = [&](size_t kv, size_t state, size_t ctx, size_t kept, size_t n_kv) {
            return (kept + 1) * (state + ctx * (sizeof(int32_t) + img)) + kv + 18 * sizeof(int64_t) +
                   sizeof(SavedConversation) + (kept + 1) * sizeof(ConversationCheckpoint) +
                   n_kv * sizeof(ConversationKv);
        };
        struct Sized { size_t kv, state, ctx, kept, n_kv; };
        const Sized cases[] = {{0, 0, 0, 0, 0},
                               {4096, 1024, 64, 0, 1}, {4096, 1024, 64, 1, 1}, {4096, 1024, 64, 6, 1},
                               {4096, 1024, 64, 64, 1}, {4096, 1024, 0, 6, 1},
                               // the with-draft box: the full-context K/V of a MoE session, 12 QSA layers + draft
                               {size_t(1) << 34, size_t(1) << 24, size_t(1) << 21, 7, 13}};
        for (const auto& c : cases)
            check(conversation_slot_ceiling_bytes(c.kv, c.state, c.ctx, c.kept, img, c.n_kv) ==
                      sum(c.kv, c.state, c.ctx, c.kept, c.n_kv),
                  "exact sum: live + kept checkpoints at the state and token bound, kv, directories, struct");
        check(conversation_slot_ceiling_bytes(0, 0, 0, 0, img, 0) ==
                  18 * sizeof(int64_t) + sizeof(SavedConversation) + sizeof(ConversationCheckpoint),
              "zero inputs: only the fixed fields remain, the caller must not clamp on them");
        // saturating: SIZE_MAX-sized inputs report the ceiling of everything, never a wrapped small number
        check(conversation_slot_ceiling_bytes(SIZE_MAX, 1, 1, 6, img, 13) == SIZE_MAX, "kv at SIZE_MAX saturates");
        check(conversation_slot_ceiling_bytes(1, SIZE_MAX, 1, 6, img, 13) == SIZE_MAX, "state at SIZE_MAX saturates");
        check(conversation_slot_ceiling_bytes(1, 1, SIZE_MAX, 6, img, 13) == SIZE_MAX, "ctx at SIZE_MAX saturates");
        check(conversation_slot_ceiling_bytes(1, 1, 1, SIZE_MAX, img, 13) == SIZE_MAX, "kept at SIZE_MAX saturates");
        check(conversation_slot_ceiling_bytes(1, 1, 1, 6, SIZE_MAX, 13) == SIZE_MAX, "img key at SIZE_MAX saturates");
        check(conversation_slot_ceiling_bytes(1, 1, 1, 6, img, SIZE_MAX) == SIZE_MAX, "n_kv at SIZE_MAX saturates");
        const size_t base_v = conversation_slot_ceiling_bytes(1 << 20, 1 << 16, 1 << 12, 6, img, 13);
        check(conversation_slot_ceiling_bytes((1 << 20) + 1, 1 << 16, 1 << 12, 6, img, 13) >= base_v &&
              conversation_slot_ceiling_bytes(1 << 20, (1 << 16) + 1, 1 << 12, 6, img, 13) >= base_v &&
              conversation_slot_ceiling_bytes(1 << 20, 1 << 16, (1 << 12) + 1, 6, img, 13) >= base_v &&
              conversation_slot_ceiling_bytes(1 << 20, 1 << 16, 1 << 12, 7, img, 13) >= base_v &&
              conversation_slot_ceiling_bytes(1 << 20, 1 << 16, 1 << 12, 6, img + 1, 13) >= base_v &&
              conversation_slot_ceiling_bytes(1 << 20, 1 << 16, 1 << 12, 6, img, 14) >= base_v,
              "the ceiling is monotone in every input");
    }
    {
        // PROPERTY: an entry built exactly at the bound the ceiling assumes - ids at ctx, one image key per token,
        // the whole running state in every checkpoint, the K/V at the part sizes in the main and draft vectors,
        // kept checkpoints - never charges bytes() beyond the ceiling: the helper cannot cut capacity a real
        // entry uses.  Vectors resize to exact capacities; a fresh ConversationBuffer allocates its payload.
        const size_t ctx = 64, state = 4096, kept = 3, img = sizeof(ConversationImageKey), kv = 5000;
        SavedConversation s;
        auto full = [&](ConversationCheckpoint& c) {
            c.ids.resize(ctx);
            c.imgs.resize(ctx);
            c.gdn.resize(state);
        };
        full(s.live);
        s.checkpoints.resize(kept);
        for (auto& c : s.checkpoints) full(c);
        s.kv.resize(2);   // the main layer's and the draft layer's vectors
        s.kv[0].k.resize(kv / 2);
        s.kv[1].k.resize(kv - kv / 2);
        const size_t ceiling = conversation_slot_ceiling_bytes(kv, state, ctx, kept, img, 2);
        check(s.bytes() <= ceiling, "property: an entry at the bound never exceeds the ceiling");
        ConversationCache fits(ceiling, 1);
        check(fits.put(std::move(s)), "and the ceiling admits such an entry as a whole slot's budget");
        // the multi-layer box: 13 K/V vectors (12 QSA layers + the draft) at a realistic token bound
        const size_t big_ctx = 8192, big_kept = 6, n_kv = 13;
        SavedConversation m;
        auto full_big = [&](ConversationCheckpoint& c) {
            c.ids.resize(big_ctx);
            c.imgs.resize(big_ctx);
            c.gdn.resize(state);
        };
        full_big(m.live);
        m.checkpoints.resize(big_kept);
        for (auto& c : m.checkpoints) full_big(c);
        m.kv.resize(n_kv);
        for (size_t j = 0; j < n_kv; ++j) m.kv[j].k.resize(1000);
        check(m.bytes() <= conversation_slot_ceiling_bytes(n_kv * 1000, state, big_ctx, big_kept, img, n_kv),
              "property: a 13-vector K/V entry at a realistic bound fits the honest n_kv ceiling");
    }
    {
        // conversation_cache_budget_bytes: the startup clamp.  Never raises; an unknown per-slot never clamps.
        check(conversation_cache_budget_bytes(1000, 4, 0) == 1000, "unknown per-slot: no clamp");
        check(conversation_cache_budget_bytes(1000, 4, 300) == 1000, "wanted below slots x per-slot: unchanged");
        check(conversation_cache_budget_bytes(1200, 4, 300) == 1200, "exactly the ceiling: unchanged");
        check(conversation_cache_budget_bytes(1201, 4, 300) == 1200, "above it: clamped to slots x per-slot");
        check(conversation_cache_budget_bytes(0, 4, 300) == 0, "nothing wanted stays nothing");
        check(conversation_cache_budget_bytes(1000, 0, 300) == 1000, "zero slots: construction disables it, no clamp");
        check(conversation_cache_budget_bytes(SIZE_MAX, 4, SIZE_MAX) == SIZE_MAX, "slots x per-slot saturates, never wraps");
        check(conversation_cache_budget_bytes(1 << 20, 4, SIZE_MAX) == (1 << 20), "a saturated ceiling cuts nothing");
    }
    {
        // shrink(): one slot per event, the budget by the slot ratio and capped at slots x per-slot,
        // oldest-first victims inside the new caps, pinned ones kept, nothing left to give at one slot.
        auto sized = [](std::initializer_list<int32_t> ids, size_t n) {
            auto s = image(ids);
            s.live.gdn.resize(n, 5);
            return s;
        };
        auto cp = [](std::initializer_list<int32_t> ids) {
            ConversationCheckpoint c; c.ids = ids; c.pinned = true; return c;
        };
        size_t sb = 0, ba = 0, fr = 0;
        ConversationCache cache(size_t(1) << 24, 4);
        check(cache.shrink(0, sb, ba, fr) && sb == 4 && ba == ((size_t(1) << 24) * 3 / 4) && fr == 0 &&
                  cache.slots() == 3 && cache.size() == 0,
              "4 to 3 slots: budget 3/4, an empty cache frees nothing");
        const size_t big = sized({1, 2, 3}, 4096).bytes();
        cache.put(sized({1, 2, 3}, 4096));
        cache.put(sized({9, 8, 7}, 2048));
        cache.put(sized({5, 6}, 64));
        check(cache.shrink(0, sb, ba, fr) && sb == 3 && cache.slots() == 2 && cache.size() == 2 && fr == big,
              "3 to 2 slots: the oldest entry leaves, freed is exact");
        check(cache.best(std::vector<int32_t>{1, 2, 3, 4}, {}, true).tokens == 0 &&
                  cache.best(std::vector<int32_t>{9, 8, 7, 6}, {}, true).tokens == 3 &&
                  cache.best(std::vector<int32_t>{5, 6, 9}, {}, true).tokens == 2,
              "the oldest went, the newer two stay");
        ConversationCache capped(size_t(1) << 24, 4);
        check(capped.shrink(1000, sb, ba, fr) && sb == 4 && ba == 3000,
              "the per-slot cap bites: the budget goes to slots x per-slot");
        const size_t one = sized({1, 2, 3}, 4096).bytes();
        ConversationCache tight(one * 3, 4);
        tight.put(sized({1, 2, 3}, 4096));
        tight.put(sized({9, 8, 7}, 4096));
        tight.put(sized({5, 6, 8}, 4096));
        check(tight.shrink(0, sb, ba, fr) && sb == 4 && tight.slots() == 3 && ba == one * 3 * 3 / 4 &&
                  tight.size() == 2 && tight.bytes() == one * 2 && fr == one,
              "byte-bound shrink evicts oldest-first back inside the scaled budget");
        // the cap and the eviction together: a per-slot cap below the held bytes (with the ratio budget far
        // above them) forces the extra evictions itself, and freed accounts for every one of them
        ConversationCache cap_evict(size_t(1) << 24, 4);
        cap_evict.put(sized({1, 2, 3}, 4096));
        cap_evict.put(sized({9, 8, 7}, 4096));
        cap_evict.put(sized({5, 6, 8}, 4096));
        const size_t cap = one * 3 / 4;
        check(cap_evict.shrink(cap, sb, ba, fr) && sb == 4 && cap_evict.slots() == 3 && ba == cap * 3 &&
                  cap_evict.size() == 2 && cap_evict.bytes() <= ba && fr == one &&
                  cap_evict.best(std::vector<int32_t>{1, 2, 3, 4}, {}, true).tokens == 0,
              "a per-slot cap under the held bytes forces extra oldest-first evictions, freed accounts for them");
        // pinned entries can never be evicted: caps that leave no room beside them are refused outright
        auto pinned = [&](std::initializer_list<int32_t> ids) {
            SavedConversation s = image(ids);
            s.checkpoints.push_back(cp(ids));
            s.checkpoints.back().gdn.resize(64, 1);
            return s;
        };
        const size_t p = pinned({1, 2, 3}).bytes();
        ConversationCache pins(p * 2 + 128, 2);
        check(pins.put(pinned({1, 2, 3})) && pins.put(pinned({9, 8, 7})), "two pinned conversations park");
        check(!pins.shrink(0, sb, ba, fr) && sb == 2 && ba == p * 2 + 128 && pins.slots() == 2 &&
                  pins.size() == 2 && fr == 0 && pins.bytes() <= ba,
              "all pinned: one slot would leave two survivors - refuse, caps and bytes untouched");
        check(pins.best(std::vector<int32_t>{1, 2, 3, 4}, {}, true).tokens == 3 &&
                  pins.best(std::vector<int32_t>{9, 8, 7, 6}, {}, true).tokens == 3,
              "both pinned conversations stand untouched");
        ConversationCache pin_tight(p + (size_t(1) << 12), 2);
        check(pin_tight.put(pinned({1, 2, 3})) && pin_tight.put(image({9, 8, 7})),
              "a pin and a plain entry fill the tight cache");
        check(!pin_tight.shrink(0, sb, ba, fr) && sb == 2 && ba == p + (size_t(1) << 12) &&
                  pin_tight.slots() == 2,
              "half the budget cannot leave room beside the pin: refuse rather than kill every future park");
        ConversationCache pin_room(p * 3 + (size_t(1) << 16), 3);
        check(pin_room.put(pinned({1, 2, 3})) && pin_room.put(image({4, 5, 6})) && pin_room.put(image({7, 8, 9})),
              "a pin and two plain entries park");
        check(pin_room.shrink(0, sb, ba, fr) && sb == 3 && pin_room.slots() == 2 &&
                  ba == (p * 3 + (size_t(1) << 16)) * 2 / 3 && pin_room.size() == 2 &&
                  pin_room.all_pinned() == false &&
                  pin_room.best(std::vector<int32_t>{1, 2, 3, 4}, {}, true).tokens == 3,
              "caps that leave room beside the pin: the oldest plain entry goes, the pin stays");
        ConversationCache one_slot(size_t(1) << 20, 1);
        check(!one_slot.shrink(1000, sb, ba, fr) && sb == 1 && ba == (size_t(1) << 20) && fr == 0 &&
                  one_slot.slots() == 1,
              "one slot: nothing left to give, the budget untouched");
        ConversationCache off(0, 4), no_slots(size_t(1) << 20, 0);
        check(!off.shrink(1000, sb, ba, fr) && !no_slots.shrink(1000, sb, ba, fr),
              "a disabled cache does not shrink");
        // regrow(): the ratchet's other half - shrink cuts the caps, free RAM puts the configured ones back.
        ConversationCache ratchet(size_t(1) << 24, 4);
        check(!ratchet.shrunk(), "an unshrunk cache has nothing to regrow");
        check(ratchet.shrink(0, sb, ba, fr) && ratchet.shrunk() && ratchet.slots() == 3, "shrink cuts the caps");
        ratchet.regrow();
        check(!ratchet.shrunk() && ratchet.slots() == 4 && ratchet.budget() == (size_t(1) << 24),
              "regrow puts the configured slots and budget back");
        // successive hopeless events give up one slot each; that march has to end somewhere.
        ConversationCache grind(size_t(1) << 24, 4);
        check(grind.shrink(0, sb, ba, fr) && grind.shrink(0, sb, ba, fr) && grind.shrink(0, sb, ba, fr) &&
                  grind.slots() == 1 && !grind.shrink(0, sb, ba, fr),
              "shrink one slot at a time down to one, then refuses - the march terminates");
        // all_pinned(): the RAM ceiling is only recorded when nothing PERMANENTLY held remains.
        ConversationCache fresh(size_t(1) << 20, 4);
        check(fresh.all_pinned(), "an empty cache can learn the ceiling");
        check(pins.all_pinned(), "only pinned entries: the ceiling stands");
        ConversationCache mixed(size_t(1) << 20, 4);
        check(mixed.put(image({5, 6, 7})), "a plain conversation parks");
        check(!mixed.all_pinned(), "a plain entry could still go");
        const uint64_t only = mixed.best(std::vector<int32_t>{5, 6, 7, 8}, {}, true).serial;
        mixed.guard(only);
        check(!mixed.all_pinned(), "a guarded donor is only held for this attempt - no ceiling beside it");
        mixed.unguard();
        check(!mixed.all_pinned(), "still a plain entry after unguard");
        // no room beside a full budget: retain() refuses instead of wrapping its headroom check
        std::vector<ConversationKv> one_kv(1);
        one_kv[0].k.resize(16, 1);
        pins.retain(std::move(one_kv), 1);
        check(pins.take_reuse().kv.empty(), "no room beside the pins: nothing retained, no wrap");
    }
    std::printf("conversation_cache_test: %d checks passed\n", checks);
}
