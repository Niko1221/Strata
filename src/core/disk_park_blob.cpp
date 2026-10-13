#include "strata/core/disk_park_blob.hpp"

#include <algorithm>
#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <system_error>

#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

namespace strata::core {
namespace {

constexpr std::uint32_t kMagic       = 0x32424B53U;   // "SKB2" 小端：头与条目都带 profile 标签
constexpr std::uint32_t kMagicLegacy = 0x31424B53U;   // "SKB1"：没有版本门的老格式，永不读取

void put_u64(std::uint8_t* p, std::uint64_t v) { std::memcpy(p, &v, 8); }
void put_u32(std::uint8_t* p, std::uint32_t v) { std::memcpy(p, &v, 4); }
std::uint32_t get_u32(const std::uint8_t* p) { std::uint32_t v = 0; std::memcpy(&v, p, 4); return v; }

bool pwrite_all(int fd, const void* src, std::size_t n, std::uint64_t off) {
    const auto* p = static_cast<const std::uint8_t*>(src);
    std::size_t done = 0;
    while (done < n) {
        const ssize_t r = ::pwrite(fd, p + done, n - done, static_cast<off_t>(off + done));
        if (r < 0) { if (errno == EINTR) continue; return false; }
        if (r == 0) return false;
        done += static_cast<std::size_t>(r);
    }
    return true;
}

bool pread_all(int fd, void* dst, std::size_t n, std::uint64_t off) {
    auto* p = static_cast<std::uint8_t*>(dst);
    std::size_t done = 0;
    while (done < n) {
        const ssize_t r = ::pread(fd, p + done, n - done, static_cast<off_t>(off + done));
        if (r < 0) { if (errno == EINTR) continue; return false; }
        if (r == 0) return false;
        done += static_cast<std::size_t>(r);
    }
    return true;
}

constexpr std::size_t kChunk = 1u << 20;
constexpr std::size_t kMetaHeader = 16;         // 文件头：magic(4) + 格式版本(4) + 标签(4) + 预留(4)
// 条目头：magic(4) + version(4) + key.lo(8) + key.hi(8) + payload 长度(8) = 32
constexpr std::size_t kEntryHeader = 32;

} // namespace

DiskParkBlobStream::DiskParkBlobStream(Options options) : options_(std::move(options)) {
    if (options_.path.empty()) return;                       // ★ 关闭
    if (options_.max_tracked == 0) options_.max_tracked = 1;

    const std::filesystem::path p(options_.path);
    std::error_code ec;
    if (p.has_parent_path() && !std::filesystem::exists(p.parent_path(), ec)) {
        std::filesystem::create_directories(p.parent_path(), ec);
        if (ec) { std::fprintf(stderr, "strata serve: disk park blob: 建目录失败\n"); return; }
    }
    // 打开；没有文件 / 文件比头还短 ⇒ 建一份带头的新文件
    bool fresh = false;                                   // 这份头是我们自己刚写的吗
    auto open_or_create = [&]() -> bool {
        fd_ = ::open(options_.path.c_str(), O_RDWR | O_CREAT, 0644);
        if (fd_ < 0) {
            std::fprintf(stderr, "strata serve: disk park blob: 打不开 %s\n", options_.path.c_str());
            return false;
        }
        struct stat st {};
        if (::fstat(fd_, &st) != 0) { ::close(fd_); fd_ = -1; return false; }
        size_ = static_cast<std::uint64_t>(st.st_size);
        if (size_ >= kMetaHeader) return true;
        std::uint8_t h[kMetaHeader] = {};
        put_u32(h + 0, kMagic);
        put_u32(h + 4, 2);                          // 文件格式版本 2 = 带标签
        put_u32(h + 8, options_.tag);
        if (!pwrite_all(fd_, h, sizeof h, 0) || ::fsync(fd_) != 0) {
            std::fprintf(stderr, "strata serve: disk park blob: 写头失败\n");
            ::close(fd_); fd_ = -1; return false;
        }
        size_ = kMetaHeader;
        fresh = true;
        return true;
    };
    if (!open_or_create()) return;
    if (fresh) return;                             // 刚建的一份：没有条目可扫

    {
        std::uint8_t h[kMetaHeader] = {};
        if (!pread_all(fd_, h, sizeof h, 0)) { ::close(fd_); fd_ = -1; return; }
        const std::uint32_t magic = get_u32(h + 0);
        const std::uint32_t have  = get_u32(h + 8);
        if (magic != kMagic) {
            std::fprintf(stderr, "strata serve: disk park blob: %s 是%s（magic %08X）"
                                 " \u21d2 这一层当作没有\n", options_.path.c_str(),
                         magic == kMagicLegacy ? "没有版本门的老格式 SKB1" : "不认识的格式", magic);
            ::close(fd_); fd_ = -1; return;
        }
        if (options_.tag != 0 && have != options_.tag) {
            // ★★★ §7.2 版本门：这份 blob 是【另一套数值】写下的（另一个引擎版本）。它的
            //   GDN 递归状态是上一套路由算出来的，喂给这一套就是错。整层不读也不追加；
            //   老文件【挪开】而不是删掉（一次 rename，18 GB 也不会被动），另起当前标签的一份。
            ::close(fd_); fd_ = -1;
            const char* mode = std::getenv("STRATA_BLOB_STALE");
            if (mode && std::string(mode) == "off") {
                std::fprintf(stderr, "strata serve: disk park blob: ★ 这份的标签 %08X ≠ 本次 %08X"
                             " ⇒ 整层关闭（STRATA_BLOB_STALE=off）\n", have, options_.tag);
                return;
            }
            char aside[1200];
            std::snprintf(aside, sizeof aside, "%s.tag-%08X.stale", options_.path.c_str(), have);
            std::error_code ec;
            std::filesystem::rename(options_.path, aside, ec);
            if (ec) {
                std::fprintf(stderr, "strata serve: disk park blob: ★ 标签 %08X ≠ 本次 %08X，"
                             "且挪开失败（%s）⇒ 整层关闭\n", have, options_.tag, ec.message().c_str());
                return;
            }
            std::fprintf(stderr, "strata serve: disk park blob: ★ 这份是标签 %08X 写下的，本次是 %08X"
                         " ⇒ 已挪开成 %s（%llu 字节一个没动），另起当前标签的一份\n",
                         have, options_.tag, aside, (unsigned long long) size_);
            if (!open_or_create()) return;
            return;
        }
    }
    // ★★ 扫一遍重建索引 —— 去重必须跨重启有效，否则这一层白做。
    //    坏条目（半写）⇒ 停在它之前：后面的字节都不认，与"只追加 + 尾部可能残缺"一致。
    std::uint64_t at = kMetaHeader;
    std::uint8_t h[kEntryHeader];
    std::size_t skipped_foreign = 0;
    while (at + kEntryHeader <= size_) {
        if (!pread_all(fd_, h, sizeof h, at)) break;
        std::uint32_t m = 0; std::uint64_t len = 0, lo = 0, hi = 0;
        std::memcpy(&m, h + 0, 4);
        std::memcpy(&lo, h + 8, 8);
        std::memcpy(&hi, h + 16, 8);
        std::memcpy(&len, h + 24, 8);
        if (m != kMagic || len == 0 || at + kEntryHeader + len > size_) break;
        // ★ 条目头 @+4（老格式在那儿写 version=1）现在是【写下它的标签】：被压缩工具
        //   混进来的别套数值记录，扫到就跳过，不进索引（条目自带长度 ⇒ 跳过是安全的）。
        const std::uint32_t etag = get_u32(h + 4);
        if (options_.tag != 0 && etag != options_.tag) {
            ++skipped_foreign;
            at += kEntryHeader + len;
            continue;
        }
        tracked_[DiskParkBlobKey{lo, hi}] = Track{DiskParkBlobRef{at + kEntryHeader, len}, ++clock_};
        at += kEntryHeader + len;
    }
    if (at != size_) {
        std::fprintf(stderr, "strata serve: disk park blob: 尾部 %llu 字节不可用（半写？），"
                             "按已扫到的 %zu 个条目继续\n",
                     (unsigned long long) (size_ - at), tracked_.size());
        size_ = at;                                   // 后续追加从这个位置接上
    }
    if (skipped_foreign != 0)
        std::fprintf(stderr, "strata serve: disk park blob: ★ %zu 条记录是别的标签写下的，已跳过\n",
                     skipped_foreign);
}

DiskParkBlobStream::~DiskParkBlobStream() {
    if (fd_ >= 0) ::close(fd_);
}

std::uint32_t DiskParkBlobStream::file_tag(const std::string& path) {
    int fd = ::open(path.c_str(), O_RDONLY);
    if (fd < 0) return 0;
    std::uint8_t h[16] = {};
    const ssize_t n = ::read(fd, h, sizeof h);
    ::close(fd);
    if (n < (ssize_t) sizeof h) return 0;
    if (get_u32(h + 0) != kMagic) return 0;          // 老格式（无门）或别的文件 ⇒ 0 = 不检查
    return get_u32(h + 8);
}

DiskParkBlobRef DiskParkBlobStream::find(const DiskParkBlobKey& key) const {
    // ★★ 不可用就别撒谎（2026-10-10 实测事故）：fd_ < 0 时内存索引可能还是"只读扫描"
    //    建起来的 ⇒ find 说有、get/put 却一定失败。上层据此判断"能不能装运行状态"，
    //    撒谎会让它走"运行状态从零开始"那条路 ⇒ 输出错。返回空 = 当作没有。
    if (fd_ < 0) return DiskParkBlobRef{};
    const auto it = tracked_.find(key);
    return it == tracked_.end() ? DiskParkBlobRef{} : it->second.ref;
}

DiskParkBlobRef DiskParkBlobStream::put(const DiskParkBlobKey& key, std::uint64_t bytes,
                                        const Fill& fill) {
    if (fd_ < 0 || bytes == 0 || !fill) { ++stats_.failures; return {}; }

    // ★ 命中 ⇒ 零写入。这就是这一层存在的理由。
    const auto it = tracked_.find(key);
    if (it != tracked_.end()) {
        it->second.stamp = ++clock_;
        ++stats_.dedup_hits;
        return it->second.ref;
    }

    const std::uint64_t at = size_;
    std::uint8_t h[kEntryHeader] = {};
    // @+4：老格式在这里写 version=1，现在写【写下它的标签】（§7.2 版本门）
    put_u32(h + 0, kMagic);
    put_u32(h + 4, options_.tag);
    put_u64(h + 8, key.lo);
    put_u64(h + 16, key.hi);
    put_u64(h + 24, bytes);
    if (!pwrite_all(fd_, h, sizeof h, at)) { ++stats_.failures; return {}; }

    std::vector<std::uint8_t> buf(kChunk);
    std::uint64_t done = 0;
    while (done < bytes) {
        const std::size_t take = static_cast<std::size_t>(std::min<std::uint64_t>(buf.size(), bytes - done));
        if (!fill(done, buf.data(), take)) { ++stats_.failures; return {}; }
        if (!pwrite_all(fd_, buf.data(), take, at + kEntryHeader + done)) { ++stats_.failures; return {}; }
        done += take;
    }
    if (::fsync(fd_) != 0) { ++stats_.failures; return {}; }

    size_ = at + kEntryHeader + bytes;
    DiskParkBlobRef ref{at + kEntryHeader, bytes};
    tracked_[key] = Track{ref, ++clock_};
    ++stats_.appends;
    stats_.bytes += bytes;
    trim();
    return ref;
}

bool DiskParkBlobStream::get(const DiskParkBlobRef& ref, const Read& sink) const {
    if (fd_ < 0 || !ref.present() || !sink) return false;
    if (ref.offset < kMetaHeader || ref.offset + ref.bytes > size_) return false;
    std::vector<std::uint8_t> buf(kChunk);
    std::uint64_t done = 0;
    while (done < ref.bytes) {
        const std::size_t take = static_cast<std::size_t>(std::min<std::uint64_t>(buf.size(), ref.bytes - done));
        if (!pread_all(fd_, buf.data(), take, ref.offset + done)) return false;
        if (!sink(done, buf.data(), take)) return false;
        done += take;
    }
    return true;
}

bool DiskParkBlobStream::worth_storing(const std::string& group, std::int64_t frontier,
                                       std::int64_t min_advance,
                                       std::int64_t min_frontier) const {
    if (frontier < min_frontier) return false;               // ★ 太短，整段重读更便宜
    if (min_advance <= 0 || group.empty()) return true;      // 闸门关掉
    const auto it = last_frontier_.find(group);
    if (it == last_frontier_.end()) return true;             // 这条会话还没存过
    return frontier - it->second >= min_advance;
}

void DiskParkBlobStream::note_stored(const std::string& group, std::int64_t frontier) {
    if (group.empty() || frontier <= 0) return;
    auto it = last_frontier_.find(group);
    if (it == last_frontier_.end()) last_frontier_.emplace(group, frontier);
    else if (frontier > it->second) it->second = frontier;    // ★ 单调，不倒退
}

void DiskParkBlobStream::trim() {
    // ★ 淘汰的是【记忆】，不是文件里的字节 —— 只追加的流不回收，等整个会话目录被删掉。
    //   丢掉记忆的代价是下次同一个检查点会重写一份（多写，不会写错）。
    while (tracked_.size() > options_.max_tracked) {
        auto victim = tracked_.begin();
        for (auto it = tracked_.begin(); it != tracked_.end(); ++it)
            if (it->second.stamp < victim->second.stamp) victim = it;
        tracked_.erase(victim);
        ++stats_.evicted_keys;
    }
}

} // namespace strata::core
