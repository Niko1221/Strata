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

void shared_buffer_serialization_and_mutation() {
    KvChunkMap pool;
    pool.init(2);
    const auto lane = pool.add_lane(2);
    check(pool.reserve(lane, 2), "allocate alias backing chunks");
    std::vector<uint8_t> first(4096, 1), second(4096, 2);
    auto prefix = pool.retain(lane, 2);
    std::weak_ptr<const KvChunkMap::Prefix> lifetime = prefix;
    ConversationBuffer aliased{9, 10};
    aliased.prepend_shared(prefix, {first.data(), second.data()}, 4096);
    ConversationBuffer ordinary;
    ordinary.resize(8194);
    check(ordinary.visit(0, ordinary.size(), [](uint8_t* p, size_t n, size_t at) {
        for (size_t i = 0; i < n; ++i) {
            const size_t offset = at + i;
            p[i] = offset < 4096 ? 1 : offset < 8192 ? 2 : uint8_t(offset - 8192 + 9);
        }
        return true;
    }), "construct ordinary expanded payload");
    check(aliased.size() == ordinary.size() && aliased == ordinary, "aliased prefix expands to ordinary payload");
    const size_t resident = aliased.bytes();
    check(resident >= 2 * sizeof(const uint8_t*) && resident < 4096,
          "alias accounting includes directory metadata but excludes shared payload");
    const ConversationBuffer& constant = aliased;
    // The session file writer uses this same const visit boundary to emit expanded bytes.
    auto serialized = [](const ConversationBuffer& buffer) {
        std::vector<uint8_t> out;
        check(buffer.visit(0, buffer.size(), [&](const uint8_t* p, size_t n, size_t at) {
            check(at == out.size(), "serialization emits contiguous absolute offsets");
            out.insert(out.end(), p, p + n);
            return true;
        }), "serialize complete const buffer");
        return out;
    };
    check(serialized(constant) == serialized(ordinary), "const serialization emits expanded shared bytes");
    std::vector<uint8_t> crossing(4);
    check(constant.read(crossing.data(), 4094, 4) && crossing == std::vector<uint8_t>({1, 1, 2, 2}),
          "const read spans two alias chunks");
    check(constant.visit(4096, 1, [&](const uint8_t* p, size_t, size_t) {
        return p == second.data();
    }) && aliased.bytes() == resident, "const reads and serialization preserve alias addresses");
    check(aliased.allocation_peak(aliased.size()) >= aliased.bytes() + ordinary.bytes(),
          "mutable alias materialization includes the expanded payload in peak admission");
    auto overwrite = aliased;
    overwrite.resize_for_overwrite(overwrite.size() + 1);
    check(overwrite.read(crossing.data(), 4094, 4) && crossing == std::vector<uint8_t>({1, 1, 2, 2}) &&
          aliased == ordinary && !overwrite.prefix_owner(),
          "overwrite growth materializes the retained prefix before asynchronous writes");
    auto private_copy = aliased;
    check(private_copy.visit(4095, 2, [](uint8_t* p, size_t n, size_t) {
        std::fill(p, p + n, 77); return true;
    }), "mutable visit materializes alias");
    check(first.back() == 1 && second.front() == 2 && aliased == ordinary,
          "private mutable visit cannot overwrite sibling or shared payload");
    std::vector<uint8_t> changed(2);
    check(private_copy.read(changed.data(), 4095, 2) && changed == std::vector<uint8_t>({77, 77}),
          "mutable visit changes the private expanded buffer");
    auto resized = aliased;
    const size_t resized_peak = resized.allocation_peak(8195);
    resized.resize(8195, 55);
    uint8_t last = 0;
    check(resized.read(&last, 8194, 1) && last == 55 && aliased == ordinary && resized.bytes() <= resized_peak,
          "resize materializes the prefix and grows privately within its admitted segment reserve");
    check(pool.shrink(lane, 0), "release original alias lane");
    prefix.reset();
    check(!lifetime.expired() && pool.references() == 2, "const alias retains the real Prefix owner");
    aliased = {};
    check(lifetime.expired() && pool.references() == 0 && pool.free_chunks() == 2,
          "materialized copies do not retain Prefix ownership after final alias release");
}

void shared_prefix_cache_lifecycle() {
    for (int operation = 0; operation < 8; ++operation) {
        KvChunkMap pool;
        pool.init(2);
        const auto lane = pool.add_lane(2);
        check(pool.reserve(lane, 2), "allocate cache prefix backing");
        std::vector<uint8_t> backing(64, 7);
        auto prefix = pool.retain(lane, 2);
        std::weak_ptr<const KvChunkMap::Prefix> lifetime = prefix;
        auto saved = image({1, 2, 3});
        ConversationCheckpoint checkpoint;
        checkpoint.ids = {1, 2};
        saved.checkpoints.push_back(std::move(checkpoint));
        saved.kv.resize(1);
        saved.kv[0].prefix = prefix;
        saved.kv[0].k.prepend_shared(prefix, {backing.data(), backing.data() + 32}, 32);
        prefix.reset();
        check(pool.shrink(lane, 0), "cache is the sole physical owner after lane deletion");
        ConversationCache cache(1 << 20, 4);
        check(cache.put(std::move(saved)), "park image with a real immutable Prefix");
        check(!lifetime.expired() && pool.references() == 2 && pool.free_chunks() == 0,
              "parked image holds the final physical references");
        if (operation == 0) {
            check(cache.evict_oldest(), "evict shared image");
        } else if (operation == 1) {
            check(cache.clear() == 1, "clear parked shared image");
        } else if (operation == 2) {
            check(cache.drop_superseded({1, 2}, {}, {}, true) == 1, "drop superseded shared image");
        } else {
            auto restored = cache.take(0);
            check(!lifetime.expired() && cache.size() == 0, "take transfers Prefix out of cache");
            if (operation == 3) {
                cache.clear();
                check(!lifetime.expired(), "cache clear cannot release an externally taken image");
                restored = {};
            } else {
                cache.retain(std::move(restored.kv), 3, {}, {1, 2, 3});
                check(!lifetime.expired(), "retained reuse owns Prefix after restore");
                if (operation == 4) cache.limit_reuse(0);
                if (operation == 5) cache.retain({}, 0);
                if (operation == 6) cache.clear();
                if (operation == 7) {
                    auto reuse = cache.take_reuse_for(std::vector<int32_t>{1, 2, 3});
                    cache.clear();
                    check(!lifetime.expired() && !reuse.kv.empty(), "take_reuse transfers ownership to caller");
                    reuse = {};
                }
            }
        }
        check(lifetime.expired() && pool.references() == 0 && pool.free_chunks() == 2,
              "production cache lifecycle releases the last Prefix and alias references");
    }
    ConversationCache cache(1 << 20, 4);
    std::vector<uint8_t> backing(32, 42);
    std::weak_ptr<const KvChunkMap::Prefix> lifetime;
    {
        KvChunkMap pool;
        pool.init(1);
        const auto lane = pool.add_lane(1);
        check(pool.reserve(lane, 1), "allocate prefix whose map will be destroyed");
        auto prefix = pool.retain(lane, 1);
        lifetime = prefix;
        auto saved = image({1, 2});
        saved.kv.resize(1);
        saved.kv[0].prefix = prefix;
        saved.kv[0].k.prepend_shared(prefix, {backing.data()}, backing.size());
        check(cache.put(std::move(saved)), "cache retains Prefix past map lifetime");
    }
    check(!lifetime.expired(), "destroying map leaves cache-owned Prefix alive");
    check(cache.clear() == 1 && lifetime.expired(), "final cache owner destruction is safe after map destruction");
}

void shared_prefix_eviction_planning() {
    auto shared_image = [](const std::shared_ptr<const KvChunkMap::Prefix>& prefix,
                           int32_t id, const std::vector<uint8_t>& backing) {
        auto saved = image({id, 2, 3});
        saved.kv.resize(1);
        saved.kv[0].prefix = prefix;
        for (ConversationBuffer* buffer : {&saved.kv[0].k, &saved.kv[0].v, &saved.kv[0].k_scale,
                                          &saved.kv[0].v_scale, &saved.kv[0].pooled})
            buffer->prepend_shared(prefix, {backing.data()}, backing.size());
        saved.stage_images.resize(1);
        saved.stage_images[0].kv.resize(1);
        saved.stage_images[0].kv[0].prefix = prefix;
        saved.stage_images[0].kv[0].k.prepend_shared(prefix, {backing.data()}, backing.size());
        return saved;
    };
    {
        KvChunkMap pool;
        pool.init(1);
        const auto target = pool.add_lane(2);
        check(pool.reserve(target, 1), "allocate full pool for eviction planning");
        std::vector<uint8_t> backing(32, 7);
        auto prefix = pool.retain(target, 1);
        const auto* identity = prefix.get();
        std::weak_ptr<const KvChunkMap::Prefix> lifetime = prefix;
        ConversationCache cache(1 << 20, 4);
        check(cache.put(shared_image(prefix, 10, backing)) && cache.put(shared_image(prefix, 20, backing)),
              "park two images sharing one Prefix lease");
        prefix.reset();
        const auto first = cache.best(std::vector<int32_t>{10, 2, 3, 4}, {}, true);
        const auto second = cache.best(std::vector<int32_t>{20, 2, 3, 4}, {}, true);
        const auto version = pool.version();
        const auto refs = pool.references();
        const auto bytes = cache.bytes();
        const auto* table = pool.table(target);
        const auto physical = table[0];
        std::vector<const KvChunkMap::Prefix*> released;
        check(cache.eviction_prefixes(1, released) && released.empty(),
              "one image cannot release a Prefix also owned by the other image");
        check(cache.eviction_prefixes(2, released) && released.size() == 1 && released[0] == identity,
              "full cohort accounts for every buffer and stage alias exactly once");
        check(pool.reclaimable({}, released) == 0 && pool.free_chunks() == 0,
              "evicting cache Prefix cannot reclaim the active lane's physical chunk");
        check(pool.write_need(target, 0, 2, {}, released) == 1 &&
              pool.write_need(target, 0, 2, {}, released) > pool.free_chunks() + pool.reclaimable({}, released),
              "even all cache candidates cannot cover a new growth chunk");
        check(!cache.eviction_prefixes(3, released) && released.empty(),
              "planning beyond available candidates fails without a partial release list");
        check(cache.size() == 2 && cache.bytes() == bytes && cache.evictions() == 0 &&
              cache.find(first.serial) && cache.find(second.serial),
              "failed growth planning preserves every cache entry and its accounting");
        check(pool.version() == version && pool.references() == refs && pool.owners(physical) == 2 &&
              pool.table(target) == table && table[0] == physical && pool.held(target) == 1 &&
              pool.free_chunks() == 0 && !lifetime.expired(),
              "failed growth planning preserves physical refs, table address, contents and version");

        check(cache.eviction_prefixes(2, released) && pool.write_need(target, 0, 1, {}, released) == 0,
              "cache cohort release predicts allocation-free target exclusivity");
        check(cache.evict_oldest() && !lifetime.expired() && pool.owners(physical) == 2,
              "first actual eviction retains the shared Prefix lease");
        check(cache.evict_oldest() && lifetime.expired() && pool.owners(physical) == 1 &&
              pool.free_chunks() == 0 && pool.write_need(target, 0, 1) == 0,
              "last eviction makes active target exclusive without freeing its physical chunk");
        std::vector<KvChunkMap::Copy> copies;
        bool changed = true;
        check(pool.prepare_write(target, 0, 1, copies, &changed) && !changed && copies.empty(),
              "target writes after cache release without allocation or COW");
    }
    {
        KvChunkMap pool;
        pool.init(1);
        const auto lane = pool.add_lane(1);
        check(pool.reserve(lane, 1), "allocate external-owner planning fixture");
        std::vector<uint8_t> backing(32, 8);
        auto prefix = pool.retain(lane, 1);
        ConversationCache cache(1 << 20, 4);
        auto incoming = shared_image(prefix, 30, backing);
        check(cache.put(shared_image(prefix, 10, backing)) && cache.put(shared_image(prefix, 20, backing)),
              "park cohort alongside an external incoming image");
        prefix.reset();
        std::vector<const KvChunkMap::Prefix*> released;
        check(cache.eviction_prefixes(2, released) && released.empty(),
              "external incoming image prevents planned Prefix release");
        incoming = {};
        check(cache.eviction_prefixes(2, released) && released.size() == 1,
              "dropping incoming image permits the complete cache cohort");
        auto taken = cache.take(0);
        check(cache.eviction_prefixes(1, released) && released.empty(),
              "externally taken image prevents the remaining cache image releasing Prefix");
        check(cache.put(std::move(taken)), "return external taken owner to cache");
        const auto guarded = cache.best(std::vector<int32_t>{20, 2, 3, 4}, {}, true);
        cache.guard(guarded.serial);
        check(cache.eviction_prefixes(1, released) && released.empty(),
              "guarded donor is not counted in the release cohort");
        check(!cache.eviction_prefixes(2, released) && released.empty(),
              "guarded donor is not an available eviction candidate");
        cache.unguard();
        check(cache.eviction_prefixes(2, released) && released.size() == 1,
              "unguarding restores the complete evictable cohort");
        taken = cache.take(0);
        cache.retain(std::move(taken.kv), 3);
        // The taken stage aliases are an external owner too; clear them before
        // checking that retained reuse alone blocks release.
        taken = {};
        check(cache.eviction_prefixes(1, released) && released.empty(),
              "retained reuse is not mistakenly counted as an evictable parked entry");
        cache.retain({}, 0);
        check(cache.eviction_prefixes(1, released) && released.size() == 1,
              "dropping retained reuse permits the final parked owner to release Prefix");
    }
    {
        KvChunkMap pool;
        pool.init(1);
        const auto lane = pool.add_lane(1);
        check(pool.reserve(lane, 1), "allocate pinned-owner planning fixture");
        std::vector<uint8_t> backing(32, 9);
        auto prefix = pool.retain(lane, 1);
        ConversationCache cache(1 << 20, 4);
        auto pinned = shared_image(prefix, 10, backing);
        ConversationCheckpoint checkpoint;
        checkpoint.ids = {10};
        checkpoint.pinned = true;
        pinned.checkpoints.push_back(std::move(checkpoint));
        check(cache.put(std::move(pinned)) && cache.put(shared_image(prefix, 20, backing)),
              "park pinned and unpinned images of one Prefix");
        prefix.reset();
        std::vector<const KvChunkMap::Prefix*> released;
        check(cache.eviction_prefixes(1, released) && released.empty(),
              "pinned image prevents unpinned candidate releasing the shared Prefix");
        check(!cache.eviction_prefixes(2, released) && released.empty(),
              "pinned image cannot be planned as an eviction candidate");
        check(cache.size() == 2 && cache.evictions() == 0 && pool.references() == 2,
              "pinned planning remains read-only");
    }
}

void shared_prefix_restore_admission() {
    {
        KvChunkMap pool;
        pool.init(5);
        const auto target = pool.add_lane(3), source = pool.add_lane(3), victim = pool.add_lane(1);
        check(pool.reserve(target, 2) && pool.reserve(source, 2) && pool.reserve(victim, 1),
              "fill restore-admission pool with old main, incoming prefix and victim");
        std::vector<uint8_t> backing(64, 7);
        auto pending_outgoing = image({90, 91, 92});
        pending_outgoing.kv.resize(1);
        pending_outgoing.kv[0].prefix = pool.retain(target, 2);
        pending_outgoing.kv[0].k.prepend_shared(pending_outgoing.kv[0].prefix,
                                               {backing.data(), backing.data() + 32}, 32);
        auto incoming = image({10, 11, 12});
        incoming.kv.resize(1);
        incoming.kv[0].prefix = pool.retain(source, 2);
        incoming.kv[0].k.prepend_shared(incoming.kv[0].prefix, {backing.data(), backing.data() + 32}, 32);
        auto evictable = image({20, 21, 22});
        evictable.kv.resize(1);
        evictable.kv[0].prefix = pool.retain(victim, 1);
        ConversationCache cache(1 << 20, 4);
        check(cache.put(std::move(incoming)) && cache.put(std::move(evictable)),
              "park incoming donor and genuinely reclaimable victim");
        check(pool.shrink(source, 0) && pool.shrink(victim, 0), "cache images own donor and victim chunks");
        const auto donor_match = cache.best(std::vector<int32_t>{10, 11, 12, 13}, {}, true);
        const auto victim_match = cache.best(std::vector<int32_t>{20, 21, 22, 23}, {}, true);
        const auto* donor = cache.find(donor_match.serial)->kv[0].prefix.get();
        cache.guard(donor_match.serial);
        const auto version = pool.version();
        const auto refs = pool.references();
        const auto bytes = cache.bytes();
        const auto* target_address = pool.table(target);
        const std::vector<int32_t> before_target(target_address, target_address + 3);
        const std::vector<int32_t> before_source(pool.table(source), pool.table(source) + 3);
        const std::vector<int32_t> before_victim(pool.table(victim), pool.table(victim) + 1);
        std::vector<int64_t> owners;
        for (int32_t physical = 0; physical < 5; ++physical) owners.push_back(pool.owners(physical));
        auto unchanged = [&] {
            check(pool.free_chunks() == 0 && pool.references() == refs && pool.version() == version,
                  "restore dry-run preserves free count, references and version");
            check(pool.table(target) == target_address && pool.held(target) == 2 &&
                  std::vector<int32_t>(pool.table(target), pool.table(target) + 3) == before_target &&
                  std::vector<int32_t>(pool.table(source), pool.table(source) + 3) == before_source &&
                  std::vector<int32_t>(pool.table(victim), pool.table(victim) + 1) == before_victim,
                  "restore dry-run preserves all lane tables and the captured target address");
            for (int32_t physical = 0; physical < 5; ++physical)
                check(pool.owners(physical) == owners[physical], "restore dry-run preserves each physical owner count");
            check(cache.size() == 2 && cache.bytes() == bytes && cache.evictions() == 0 &&
                  cache.find(donor_match.serial) && cache.find(victim_match.serial) &&
                  pending_outgoing.live.ids == std::vector<int32_t>({90, 91, 92}),
                  "refused admission leaves donor parked and outgoing image unparked and intact");
        };
        check(!pool.can_share(target, *donor, 2, 3),
              "retained outgoing whole chunks cannot fund incoming private partial tail in a full pool");
        unchanged();
        check(pool.can_share(target, *donor, 2, 2),
              "incoming completed chunks alone need no private tail allocation");
        check(!pool.can_share(target, *donor, 2, 2, {}, {donor}),
              "restore planner refuses to destroy its own raw source Prefix lease");
        unchanged();
        std::vector<const KvChunkMap::Prefix*> released;
        check(cache.eviction_prefixes(1, released) && released.size() == 1 &&
              released[0] != donor && pool.reclaimable({}, released) == 1,
              "guarded incoming donor leaves one genuinely reclaimable cache candidate");
        check(pool.can_share(target, *donor, 2, 3, {}, released),
              "restore dry-run admits the partial tail when a real victim can release one chunk");
        unchanged();
        check(cache.evict_oldest() && cache.find(donor_match.serial) && !cache.find(victim_match.serial) &&
              pool.free_chunks() == 1, "actual victim eviction releases exactly the planned private chunk");
        check(pool.can_share(target, *donor, 2, 3), "actual eviction makes the same restore plan feasible");
        std::vector<KvChunkMap::Copy> copies;
        bool changed = false;
        check(pool.share(target, *donor, 2, 3, copies, &changed) && changed && copies.empty(),
              "production share succeeds after the admitted eviction");
        check(pool.table(target) == target_address && pool.table(target)[0] == donor->chunks()[0] &&
              pool.table(target)[1] == donor->chunks()[1] && pool.owners(pool.table(target)[2]) == 1,
              "restore publishes shared completed chunks and an exclusive partial tail in the fixed table");
        check(pool.owners(before_target[0]) == 1 && pool.owners(before_target[1]) == 1,
              "retained outgoing image still owns the replaced old-main chunks");
    }
    {
        KvChunkMap pool;
        pool.init(5);
        const auto donor = pool.add_lane(5), borrower = pool.add_lane(2);
        check(pool.reserve(donor, 4), "capture donor longer than borrower's entire logical window");
        auto prefix = pool.retain(donor, 4);
        const auto version = pool.version();
        const auto refs = pool.references();
        constexpr int64_t chunk_tokens = 4, upto = 6;
        constexpr int64_t whole = upto / chunk_tokens, need = (upto + chunk_tokens - 1) / chunk_tokens;
        check(!pool.can_share(borrower, *prefix, 4, 5),
              "donor's full captured extent does not fit the shorter borrower table");
        check(pool.can_share(borrower, *prefix, whole, need),
              "borrow admission uses only completed chunks below upto plus its private partial tail");
        check(pool.version() == version && pool.references() == refs && pool.held(borrower) == 0 &&
              pool.free_chunks() == 1 && pool.table(borrower)[0] == pool.trash() &&
              pool.table(borrower)[1] == pool.trash(), "successful borrow dry-run is also read-only");
        std::vector<KvChunkMap::Copy> copies;
        check(pool.share(borrower, *prefix, whole, need, copies) && copies.empty(),
              "actual shorter borrow agrees with production admission plan");
        check(pool.held(borrower) == 2 && pool.table(borrower)[0] == prefix->chunks()[0] &&
              pool.owners(pool.table(borrower)[1]) == 1 && pool.free_chunks() == 0,
              "shorter borrow shares one complete chunk and privately owns its partial tail");
        for (int c = 1; c < 4; ++c)
            check(pool.table(borrower)[1] != prefix->chunks()[c], "borrow tail never aliases later donor chunks");
    }
}
}

int main() {
    shared_buffer_serialization_and_mutation();
    shared_prefix_cache_lifecycle();
    shared_prefix_eviction_planning();
    shared_prefix_restore_admission();
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
        check(baseline.first == 14 && baseline.second == 0, "baseline: A returns to its prompt-read state, B to nothing (as measured)");
        const auto parks_only = replay(true, false);
        check(parks_only.first == 17 && parks_only.second == 0, "slot parks alone: A whole, B's copy evicted by a stale initial image");
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
        // --kv-pool-tokens: the chain moves with the K/V - every checkpoint that is a prefix of the conversation
        // goes, in order; one of another conversation (or one past it) stays behind for the caller to drop
        auto ck = [](std::vector<int32_t> ids, uint64_t used) {
            ConversationCheckpoint c;
            c.ids = std::move(ids);
            c.used = used;
            c.gdn.assign(8, 7);
            return c;
        };
        std::vector<ConversationCheckpoint> from;
        from.push_back(ck({1, 2}, 1));
        from.push_back(ck({1, 2, 3, 4}, 2));
        from.push_back(ck({1, 2, 9, 9}, 3));         // another branch
        from.push_back(ck({1, 2, 3, 4, 5, 6, 7}, 4));  // past the conversation
        from.push_back(ck({1, 2, 3, 4, 5}, 5));        // the whole conversation: a prefix
        std::vector<ConversationCheckpoint> to;
        to.push_back(ck({1}, 0));                      // what the receiver already holds stays first
        check(conversation_checkpoints_carry(from, {1, 2, 3, 4, 5}, to) == 3, "carry: three checkpoints are prefixes");
        check(to.size() == 4 && to[0].used == 0 && to[1].used == 1 && to[2].used == 2 && to[3].used == 5,
              "carry: they arrive in order after what was there");
        check(from.size() == 2 && from[0].used == 3 && from[1].used == 4, "carry: the others stay");
        check(to[3].gdn.size() == 8 && to[3].gdn[0] == 7, "carry: the running state moved with it");
        ConversationCheckpoint pic = ck({1, 2}, 6);
        pic.imgs.push_back({1, 77});
        std::vector<ConversationCheckpoint> with_picture, out;
        with_picture.push_back(std::move(pic));
        check(conversation_checkpoints_carry(with_picture, {1, 2, 3}, out) == 0 && out.empty(),
              "carry: a checkpoint with a picture is not carried");
        std::vector<ConversationCheckpoint> none, sink;
        check(conversation_checkpoints_carry(none, {1, 2}, sink) == 0 && sink.empty(), "carry: nothing to move");
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
        // #4420: a batch slot's chain holds one checkpoint (its admission's): parking it must not drop (#342) the main session's copy of the
        // same conversation that also holds the root a sibling borrows from (an agent's system prompt and tools) - supersedes_richer()
        // names that case, so the slot is not parked
        auto cp = [](std::vector<int32_t> ids) { ConversationCheckpoint c; c.ids = std::move(ids); return c; };
        SavedConversation main_copy = image({1, 2, 3, 4, 10, 11, 12});
        main_copy.checkpoints = {cp({1, 2, 3, 4}), cp({1, 2, 3, 4, 10, 11})};
        ConversationCache cache(1 << 20, 4);
        check(cache.put(std::move(main_copy)), "park the main session's copy with its root and turn checkpoints");
        const std::vector<int32_t> slot_ids{1, 2, 3, 4, 10, 11, 12, 20, 21};          // the slot decoded the reply
        const std::vector<ConversationCheckpoint> slot_chain{cp({1, 2, 3, 4, 10, 11})};   // the admission's only one
        check(cache.supersedes_richer(slot_ids, {}, slot_chain, true), "the slot's park would drop the root checkpoint");
        const std::vector<ConversationCheckpoint> full_chain{cp({1, 2, 3, 4}), cp({1, 2, 3, 4, 10, 11})};
        check(!cache.supersedes_richer(slot_ids, {}, full_chain, true), "a chain that holds both loses nothing");
        check(!cache.supersedes_richer(slot_ids, {}, slot_chain, false), "steering mode is isolated");
        check(!cache.supersedes_richer(std::vector<int32_t>{1, 2, 3, 4, 30, 31}, {},
                                       std::vector<ConversationCheckpoint>{cp({1, 2, 3, 4, 30})}, true),
              "another conversation on the same root supersedes nothing");
        check(cache.size() == 1 && cache.superseded() == 0, "the predicate changes nothing");
        const auto sib = cache.best(std::vector<int64_t>{1, 2, 3, 4, 40}, {}, true);
        check(sib.tokens == 4 && cache.borrows(sib), "a sibling still borrows the root");
        // what drop_superseded does with the slot's chain (the park this rule prevents): the root goes
        check(cache.drop_superseded(slot_ids, {}, slot_chain, true) == 1 && cache.size() == 0,
              "the slot's chain supersedes the richer copy");
    }
    {
        // Equal placeholder tokens do not identify the image pixels affecting those K/V cells.
        auto kv = [] { std::vector<ConversationKv> v(1); v[0].k.resize(100, 1); return v; };
        ConversationCache cache(1 << 20, 4);
        const std::vector<int32_t> root = {1, 2, 3, 4};
        // the pictures: equal placeholder tokens over different pixels share only the cells before the picture
        const std::vector<ConversationImageKey> img_a{{2, 111}}, img_b{{2, 222}};
        cache.retain(kv(), 4, {}, root, img_a);
        auto r = cache.take_reuse_for(root, img_b);
        check(r.unchanged_tokens == 2, "the same tokens under different pixels stop at the picture");
        cache.retain(kv(), 4, {}, root, img_a);
        r = cache.take_reuse_for(root);
        check(r.unchanged_tokens == 2, "a conversation without the picture reuses nothing past its start");
        cache.retain(kv(), 4, {}, root, img_a);
        r = cache.take_reuse_for(root, img_a);
        check(r.unchanged_tokens == 4, "the same tokens and the same pixels keep everything");
        check(conversation_shared_prefix({1, 2, 3}, {}, {1, 2}, {}) == 2, "shared prefix: the shorter prefix");
        check(conversation_shared_prefix({1, 2, 3}, {}, {1, 9, 3}, {}) == 1, "shared prefix: to the first difference");
        check(conversation_shared_prefix({1, 2}, {{1, 5}}, {1, 2}, {{1, 5}}) == 2, "shared prefix: equal pictures pass");
    }
    std::printf("conversation_cache_test: %d checks passed\n", checks);
}
