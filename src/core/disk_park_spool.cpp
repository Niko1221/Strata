#include "strata/core/disk_park_spool.hpp"

#include <algorithm>
#include <cstdio>
#include <map>
#include <tuple>
#include <cstring>
#include <limits>

namespace strata::core {
namespace {
constexpr std::size_t kMinStride = 4096;
}

DiskParkSpool::DiskParkSpool(Options options)
    : options_(std::move(options)) {
    if (options_.path.empty()) return;                  // ★ 关闭
    if (options_.state_stride != 0 && options_.state_stride < kMinStride) options_.state_stride = kMinStride;
    if (options_.kv_stride    != 0 && options_.kv_stride    < kMinStride) options_.kv_stride    = kMinStride;

    DiskParkBridge::Options bo;
    bo.base_path             = options_.path;
    bo.stride[static_cast<std::size_t>(DiskParkKind::StateImage)] = options_.state_stride;
    bo.stride[static_cast<std::size_t>(DiskParkKind::QsaKv)]      = options_.kv_stride;
    bo.capacity_bytes[static_cast<std::size_t>(DiskParkKind::StateImage)] = options_.state_capacity_bytes;
    bo.capacity_bytes[static_cast<std::size_t>(DiskParkKind::QsaKv)]      = options_.kv_capacity_bytes;
    bo.verify_crc            = options_.verify_crc;
    try {
        bridge_ = std::make_unique<DiskParkBridge>(bo);
        enabled_ = bridge_->enabled();
    } catch (const std::exception& e) {
        // ★ 起不来就关掉这一层 —— 绝不能因此让引擎起不来
        std::fprintf(stderr, "strata serve: disk park spool disabled: %s\n", e.what());
        bridge_.reset();
        enabled_ = false;
    }
}

DiskParkSpool::~DiskParkSpool() = default;

DiskParkIdentity DiskParkSpool::make_id(std::int64_t frontier, const PrefixDigest& d,
                                        std::uint32_t chunk) const noexcept {
    DiskParkIdentity id;
    id.lo       = d.lo;
    id.hi       = d.hi;
    id.tag      = options_.profile.tag();
    id.frontier = static_cast<std::uint32_t>(frontier);
    id.chunk    = chunk;
    return id;
}

PrefixDigest DiskParkSpool::digest_at(std::int64_t frontier, std::span<const std::int32_t> ids,
                                      std::span<const std::uint64_t> image_hashes,
                                      std::span<const std::int64_t> image_starts) const {
    return prefix_digest(ids, image_hashes, image_starts, frontier);
}

std::size_t DiskParkSpool::store(const SavedConversation& image, std::int64_t frontier,
                                 std::span<const std::int32_t> ids,
                                 std::span<const std::uint64_t> image_hashes,
                                 std::span<const std::int64_t> image_starts) {
    if (!enabled_) return 0;
    if (frontier <= 0 || static_cast<std::size_t>(frontier) > ids.size()) return 0;

    const std::size_t stride = options_.state_stride;
    if (stride <= kChunkHeader) return 0;
    const std::size_t per_chunk = stride - kChunkHeader;

    // ★ 流式：先只【计数】总长（不分配），再边打包边填页落盘。
    //   之前的做法是先 pack 出一个完整的 vector（@256K 约 4-6 GiB），
    //   在停放路径上等于把瞬时内存需求翻倍 —— 那会让磁盘层在内存紧张时静默失效。
    const std::uint64_t total = packed_saved_conversation_bytes(image);
    if (total == 0) return 0;
    const std::size_t n_data_chunks =
        (static_cast<std::size_t>(total) + per_chunk - 1) / per_chunk;
    if (n_data_chunks == 0 || n_data_chunks > 0xFFFFFF) return 0;   // 上限保护

    const PrefixDigest d = digest_at(frontier, ids, image_hashes, image_starts);

    // ★★ 前缀去重：同一会话在盘上留【多份】快照是纯浪费 —— find() 永远只选最长的
    //    那个匹配，短的（严格前缀）永远不会被读。实测一个会话能被写 6 次
    //    （121K→129K→146K→153K→154K→155K token），白占 ~18 GiB。
    //
    //    判据：位置 i 的摘要就是"前 i 个 token"的摘要，所以盘上若存在
    //    (digest_i, frontier = i) 的条目，它就是这份新快照的严格前缀 → 丢掉。
    //    只比同一个 tag（profile）的条目：不同 profile 的字节布局不同。
    if (options_.supersede_prefixes) {
        const std::uint32_t tag = options_.profile.tag();
        std::vector<DiskParkIdentity> live = bridge_->live_identities(DiskParkKind::StateImage);
        std::map<std::tuple<std::uint64_t, std::uint64_t, std::uint32_t>,
                 std::vector<DiskParkIdentity>> groups;
        for (const auto& id : live)
            if (id.tag == tag)
                groups[{id.lo, id.hi, id.frontier}].push_back(id);
        if (!groups.empty()) {
            PrefixDigestBuilder b;
            for (std::int64_t i = 0; i < frontier && !groups.empty(); ++i) {
                b.advance(ids[static_cast<std::size_t>(i)], image_hashes, image_starts);
                const PrefixDigest di = b.value();
                auto it = groups.find({di.lo, di.hi, static_cast<std::uint32_t>(i + 1)});
                if (it == groups.end()) continue;
                for (const auto& id : it->second)
                    bridge_->drop(id, DiskParkKind::StateImage);
                stats_.superseded += it->second.size();
                groups.erase(it);
            }
        }
    }

    // chunk 0：存在性探测页（8 字节总长 + 分片数），让 find() 一次探测就能确认
    // "这个 frontier 在磁盘上有完整条目"。
    std::vector<std::byte> probe(stride, std::byte{0});
    std::memcpy(probe.data(), &total, sizeof total);
    const std::uint32_t nchunks32 = static_cast<std::uint32_t>(n_data_chunks);
    std::memcpy(probe.data() + 8, &nchunks32, sizeof nchunks32);
    if (!bridge_->spill_sync(make_id(frontier, d, 0), DiskParkKind::StateImage, probe)) {
        ++stats_.store_failures;
        return 0;
    }

    // 逐页落盘的汇
    struct PageSink final : PackSink {
        DiskParkSpool* self;
        std::int64_t frontier;
        PrefixDigest d;
        std::size_t stride, per_chunk;
        // ★ 有虚析构的基类 → 不是聚合类型，必须显式构造
        PageSink(DiskParkSpool* s_, std::int64_t f_, PrefixDigest d_, std::size_t st_, std::size_t pc_)
            : self(s_), frontier(f_), d(d_), stride(st_), per_chunk(pc_),
              page(st_, std::byte{0}), in_page(0), next_chunk(1), written(st_), ok(true) {}
        std::vector<std::byte> page;
        std::size_t in_page = 0;      // 当前页数据区已填字节
        std::uint32_t next_chunk = 1;
        std::size_t written = 0;
        bool ok = true;

        bool flush_page() {
            const std::uint32_t magic = kChunkMagic, ver = kChunkVersion;
            std::memcpy(page.data(), &magic, sizeof magic);
            std::memcpy(page.data() + 4, &ver, sizeof ver);
            if (!self->bridge_->spill_sync(self->make_id(frontier, d, next_chunk),
                                           DiskParkKind::StateImage, page)) {
                ok = false;
                return false;
            }
            ++next_chunk;
            written += stride;
            std::fill(page.begin(), page.end(), std::byte{0});
            in_page = 0;
            return true;
        }
        bool write(const void* data, std::size_t n) override {
            if (!ok) return false;
            const auto* p = static_cast<const std::byte*>(data);
            while (n > 0) {
                const std::size_t room = per_chunk - in_page;
                const std::size_t take = n < room ? n : room;
                std::memcpy(page.data() + kChunkHeader + in_page, p, take);
                in_page += take; p += take; n -= take;
                if (in_page == per_chunk && !flush_page()) return false;
            }
            return true;
        }
    };
    PageSink sink(this, frontier, d, stride, per_chunk);   // written 从 stride 起算（探测页）

    if (!pack_saved_conversation_into(image, sink) || !sink.ok) {
        drop(Found{frontier, d});       // ★ 半写失败：整条丢掉，不留"看起来完整"的残片
        ++stats_.store_failures;
        return 0;
    }
    if (sink.in_page > 0 && !sink.flush_page()) {   // 最后不满的一页
        drop(Found{frontier, d});
        ++stats_.store_failures;
        return 0;
    }
    ++stats_.stores;
    stats_.store_bytes += sink.written;
    return sink.written;
}

std::optional<DiskParkSpool::Found> DiskParkSpool::find(std::span<const std::int32_t> ids,
                                                std::span<const std::uint64_t> image_hashes,
                                                std::span<const std::int64_t> image_starts) const {
    if (!enabled_) return std::nullopt;
    const std::uint32_t tag = options_.profile.tag();

    // 磁盘上现有的 frontier → 其摘要
    std::vector<std::pair<std::int64_t, PrefixDigest>> candidates;
    for (const DiskParkIdentity& id : bridge_->live_identities(DiskParkKind::StateImage)) {
        if (id.tag != tag || id.chunk != 0) continue;      // ★ 只认本 profile 的探测页
        candidates.emplace_back(static_cast<std::int64_t>(id.frontier), PrefixDigest{id.lo, id.hi});
    }
    if (candidates.empty()) { ++stats_.find_misses; return std::nullopt; }

    // 一次遍历 prompt，在每个候选 frontier 处比对摘要 —— O(n + #候选)
    std::sort(candidates.begin(), candidates.end(),
              [](const auto& a, const auto& b) { return a.first > b.first; });   // 长的先试
    PrefixDigestBuilder b;
    std::size_t next = 0;
    // ★★ 必须【严格】短于 prompt —— 与上游的两处语义一致：
    //     conversation_cache.hpp:171  n >= prompt.size() → 不匹配
    //     generate.cpp:8305           L > n-1 → false
    //   引擎的约定是"最后一个 token 开启下一个 verify 窗口"；若 frontier == ids.size()，
    //   read_from 会等于 n，没有 token 进窗口 → 从陈旧 logits 采样。
    const std::int64_t limit = std::min<std::int64_t>(static_cast<std::int64_t>(ids.size()) - 1,
                                                      std::numeric_limits<std::int32_t>::max());
    if (limit <= 0) { ++stats_.find_misses; return std::nullopt; }
    while (next < candidates.size() && candidates[next].first > limit) ++next;   // 超过 prompt 的放弃
    for (std::int64_t i = 0; i < limit && next < candidates.size(); ++i) {
        b.advance(ids[static_cast<std::size_t>(i)], image_hashes, image_starts);
        while (next < candidates.size() && candidates[next].first == i + 1) {
            if (b.value() == candidates[next].second) {
                ++stats_.find_hits;
                return Found{candidates[next].first, candidates[next].second};
            }
            ++next;
        }
    }
    ++stats_.find_misses;
    return std::nullopt;
}

std::optional<SavedConversation> DiskParkSpool::load(const Found& found) {
    const std::int64_t frontier = found.frontier;
    if (!enabled_ || frontier <= 0) { ++stats_.load_failures; return std::nullopt; }
    const std::size_t stride = options_.state_stride;
    const std::size_t per_chunk = stride - kChunkHeader;
    const std::uint32_t tag = options_.profile.tag();

    // ★ 摘要由调用方（find）给出 —— 不能在这里"按 frontier 反查"，
    //   因为同一个 frontier 上可能有多个不同对话的条目。
    const PrefixDigest d = found.digest;
    bool present = false;
    for (const DiskParkIdentity& id : bridge_->live_identities(DiskParkKind::StateImage)) {
        if (id.tag == tag && id.frontier == static_cast<std::uint32_t>(frontier) &&
            id.chunk == 0 && id.lo == d.lo && id.hi == d.hi) {
            present = true;
            break;
        }
    }
    if (!present) { ++stats_.load_failures; return std::nullopt; }

    std::vector<std::byte> page(stride);
    if (!bridge_->restore(make_id(frontier, d, 0), DiskParkKind::StateImage, page)) {
        ++stats_.load_failures;
        return std::nullopt;
    }
    std::uint64_t total = 0;
    std::memcpy(&total, page.data(), sizeof total);
    if (total == 0 || total > (std::uint64_t)1 << 40) { ++stats_.load_failures; return std::nullopt; }

    // ★ 流式读回：按需从分片拉取，不把整份字节流先物化出来。
    //   早先的写法是"读全量 → 再解析"，峰值 = 2× 快照 ——
    //   @512K 的 7.3 GiB 快照就变成 14.6 GiB，而可用内存只有 8.9 GiB，
    //   于是【写进去的东西根本读不回来】。现在峰值 = 解析出的 SavedConversation（1×）。
    struct PageSource final : PackSource {
        DiskParkSpool* self;
        std::int64_t frontier;
        PrefixDigest d;
        std::size_t stride, per_chunk;
        std::vector<std::byte> page;
        std::uint32_t next_chunk = 1;   // 下一个要取的分片
        std::size_t in_page = 0;        // 当前分片数据区已消费字节
        std::size_t in_page_valid = 0;  // 当前分片数据区有效字节
        bool ok = true;
        PageSource(DiskParkSpool* s_, std::int64_t f_, PrefixDigest d_, std::size_t st_, std::size_t pc_)
            : self(s_), frontier(f_), d(d_), stride(st_), per_chunk(pc_), page(st_) {}

        bool fetch() {
            if (!self->bridge_->restore(self->make_id(frontier, d, next_chunk),
                                        DiskParkKind::StateImage, page)) {
                ok = false;
                return false;
            }
            if (next_chunk == 1) {
                std::uint32_t magic = 0, ver = 0;
                std::memcpy(&magic, page.data(), sizeof magic);
                std::memcpy(&ver, page.data() + 4, sizeof ver);
                if (magic != kChunkMagic || ver != kChunkVersion) { ok = false; return false; }
            }
            ++next_chunk;
            in_page = 0;
            in_page_valid = per_chunk;      // 最后一片的尾部填充由外层总长截断
            return true;
        }
        bool read(void* dst, std::size_t n) override {
            if (!ok) return false;
            auto* p = static_cast<std::byte*>(dst);
            while (n > 0) {
                if (in_page == in_page_valid && !fetch()) return false;
                const std::size_t avail = in_page_valid - in_page;
                const std::size_t take = n < avail ? n : avail;
                std::memcpy(p, page.data() + kChunkHeader + in_page, take);
                in_page += take; p += take; n -= take;
            }
            return true;
        }
    };
    PageSource src(this, frontier, d, stride, per_chunk);
    auto image = unpack_saved_conversation_from(src, total);
    if (!image || !src.ok) {
        ++stats_.load_failures;
        // ★★ 丢掉这个坏条目。否则它会永久遮住更短的完好条目 ——
        //    find() 取【最长】匹配，而这个条目每次都命中、每次都读失败，
        //    于是每次请求都：drain() 写盘 3.5 GiB×N → 读失败 → 全量重算。
        //    典型成因：store() 途中被 SIGKILL（引擎无优雅关闭），
        //    探测页(chunk 0，find 的唯一判据)已写而数据片缺失。
        drop(found);
        return std::nullopt;
    }
    ++stats_.loads;
    stats_.load_bytes += total;
    return image;
}

bool DiskParkSpool::drop(const Found& found) {
    const std::int64_t frontier = found.frontier;
    if (!enabled_ || frontier <= 0) return false;
    const std::uint32_t tag = options_.profile.tag();
    bool any = false;
    // 收集该条目（同摘要）的全部分片再删（不能在遍历中改索引）
    std::vector<DiskParkIdentity> victims;
    for (const DiskParkIdentity& id : bridge_->live_identities(DiskParkKind::StateImage)) {
        if (id.tag != tag || id.frontier != static_cast<std::uint32_t>(frontier)) continue;
        if (id.lo != found.digest.lo || id.hi != found.digest.hi) continue;   // ★ 只删这一份
        victims.push_back(id);
    }
    for (const auto& id : victims) any = bridge_->drop(id, DiskParkKind::StateImage) || any;
    return any;
}

std::vector<std::int64_t> DiskParkSpool::frontiers() const {
    std::vector<std::int64_t> out;
    if (!enabled_) return out;
    const std::uint32_t tag = options_.profile.tag();
    for (const DiskParkIdentity& id : bridge_->live_identities(DiskParkKind::StateImage))
        if (id.tag == tag && id.chunk == 0) out.push_back(static_cast<std::int64_t>(id.frontier));
    std::sort(out.begin(), out.end());
    out.erase(std::unique(out.begin(), out.end()), out.end());
    return out;
}

DiskParkSpool::Stats DiskParkSpool::stats() const { return stats_; }

void DiskParkSpool::flush() { if (bridge_) bridge_->wait_idle(); }

} // namespace strata::core
