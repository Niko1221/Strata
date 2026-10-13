#include "strata/core/disk_park_store.hpp"

#include <algorithm>
#include <array>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <system_error>

#include <fcntl.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

namespace strata::core {
namespace {

// 4096 对齐：让槽头落在页边界上，避免跨设备的读改写。
constexpr std::size_t kAlign = 4096;

std::size_t align_up(std::size_t n) noexcept { return (n + kAlign - 1) & ~(kAlign - 1); }

bool pread_all(int fd, void* dst, std::size_t n, std::uint64_t off) noexcept {
    auto* p = static_cast<std::byte*>(dst);
    while (n > 0) {
        const ssize_t got = ::pread(fd, p, n, static_cast<off_t>(off));
        // ★ EINTR 要重试：一次信号打断不该被当成"页不存在"。
        //   引擎自家的 I/O（conversation_file.cpp:388-417）就是处理 EINTR 的。
        if (got < 0 && errno == EINTR) continue;
        if (got <= 0) return false;                 // EOF 或真错误
        p += static_cast<std::size_t>(got);
        off += static_cast<std::uint64_t>(got);
        n -= static_cast<std::size_t>(got);
    }
    return true;
}

bool pwrite_all(int fd, const void* src, std::size_t n, std::uint64_t off) noexcept {
    const auto* p = static_cast<const std::byte*>(src);
    while (n > 0) {
        const ssize_t put = ::pwrite(fd, p, n, static_cast<off_t>(off));
        if (put < 0 && errno == EINTR) continue;    // ★ 同上
        if (put <= 0) return false;                 // ★ 磁盘满在这里返回 -1，而不是 SIGBUS
        p += static_cast<std::size_t>(put);
        off += static_cast<std::uint64_t>(put);
        n -= static_cast<std::size_t>(put);
    }
    return true;
}

// CRC-32（反射多项式 0xEDB88320，即 zlib 用的那个），切片查表 8 路。
struct Crc32Table {
    std::array<std::array<std::uint32_t, 256>, 8> t{};
    constexpr Crc32Table() {
        for (std::uint32_t i = 0; i < 256; ++i) {
            std::uint32_t c = i;
            for (int k = 0; k < 8; ++k) c = (c & 1U) ? (0xEDB88320U ^ (c >> 1U)) : (c >> 1U);
            t[0][i] = c;
        }
        for (std::uint32_t i = 0; i < 256; ++i)
            for (std::size_t s = 1; s < 8; ++s)
                t[s][i] = (t[s - 1][i] >> 8U) ^ t[0][t[s - 1][i] & 0xFFU];
    }
};
constexpr Crc32Table kCrc{};

} // namespace

std::uint32_t DiskParkStore::crc32(std::span<const std::byte> data) noexcept {
    const auto* p = reinterpret_cast<const unsigned char*>(data.data());
    std::size_t n = data.size();
    std::uint32_t c = 0xFFFFFFFFU;
    while (n >= 8) {
        c ^= static_cast<std::uint32_t>(p[0]) | (static_cast<std::uint32_t>(p[1]) << 8U) |
             (static_cast<std::uint32_t>(p[2]) << 16U) | (static_cast<std::uint32_t>(p[3]) << 24U);
        c = kCrc.t[7][c & 0xFFU] ^ kCrc.t[6][(c >> 8U) & 0xFFU] ^ kCrc.t[5][(c >> 16U) & 0xFFU] ^
            kCrc.t[4][(c >> 24U) & 0xFFU] ^ kCrc.t[3][p[4]] ^ kCrc.t[2][p[5]] ^ kCrc.t[1][p[6]] ^
            kCrc.t[0][p[7]];
        p += 8;
        n -= 8;
    }
    while (n-- > 0) c = kCrc.t[0][(c ^ *p++) & 0xFFU] ^ (c >> 8U);
    return c ^ 0xFFFFFFFFU;
}

std::size_t DiskParkStore::slot_pitch() const noexcept {
    return align_up(kSlotHeaderSize + options_.slot_size);
}

std::uint64_t DiskParkStore::payload_offset(std::uint32_t slot) const noexcept {
    return static_cast<std::uint64_t>(slot) * slot_pitch() + kSlotHeaderSize;
}

bool DiskParkStore::matches(const Header& h, const DiskParkIdentity& id) noexcept {
    return h.magic == Header::kMagic && h.lo == id.lo && h.hi == id.hi && h.tag == id.tag &&
           h.frontier == id.frontier && h.chunk == id.chunk;
}

// ---------------------------------------------------------------- 生命周期

DiskParkStore::DiskParkStore(Options options) : options_(std::move(options)) {
    if (options_.path.empty()) throw std::invalid_argument("DiskParkStore: path is empty");
    if (options_.slot_size == 0) throw std::invalid_argument("DiskParkStore: slot_size is zero");
    if (options_.max_slots == 0) {
        const std::size_t slots = options_.capacity_bytes / slot_pitch();
        if (slots == 0) throw std::invalid_argument("DiskParkStore: capacity holds no slot");
        options_.max_slots = static_cast<std::uint32_t>(
            std::min<std::size_t>(slots, std::numeric_limits<std::uint32_t>::max()));
    }
    max_slots_  = options_.max_slots;
    file_bytes_ = static_cast<std::size_t>(max_slots_) * slot_pitch();
    // ★ 价值感知淘汰：引擎的 --value-aware-demote 会设这个环境变量
    //   （选项不方便从引擎一层层传到 store，而本仓库已有多处用环境变量控制选项）。
    if (std::getenv("STRATA_VALUE_AWARE_DEMOTE") != nullptr) options_.value_aware_demote = true;
    readers_.assign(max_slots_, 0);
    free_slots_.reserve(max_slots_);
    lru_stamps_.assign(max_slots_, kNoStamp);

    open_file(true);
    if (!load_index()) rebuild_from_scan();
}
DiskParkStore::~DiskParkStore() {
    // ★ 析构不 flush：所有者必须显式 flush_index()。这里只保证不抛。
    if (file_handle_ >= 0) ::close(file_handle_);
}

void DiskParkStore::open_file(bool create) {
    const int flags = O_RDWR | (create ? O_CREAT : 0);
    file_handle_ = ::open(options_.path.c_str(), flags, 0644);
    if (file_handle_ < 0)
        throw std::runtime_error("DiskParkStore: cannot open " + options_.path + ": " +
                                 std::strerror(errno));

    struct stat st{};
    if (::fstat(file_handle_, &st) != 0)
        throw std::runtime_error("DiskParkStore: fstat failed: " + std::string(std::strerror(errno)));

    // ★ 在任何扩展之前记录：这个文件原本就有内容吗？
    file_had_content_ = st.st_size > 0;

    if (create && static_cast<std::size_t>(st.st_size) < file_bytes_) {
        // ★ 稀疏扩展：只改大小，不写零页。
        if (::ftruncate(file_handle_, static_cast<off_t>(file_bytes_)) != 0)
            throw std::runtime_error("DiskParkStore: ftruncate failed: " +
                                     std::string(std::strerror(errno)));
    } else if (static_cast<std::size_t>(st.st_size) != 0 &&
               static_cast<std::size_t>(st.st_size) != file_bytes_) {
        throw std::runtime_error("DiskParkStore: data file has a different geometry (" +
                                 std::to_string(st.st_size) + " vs " +
                                 std::to_string(file_bytes_) + ")");
    }
}

void DiskParkStore::clear_slot(std::uint32_t slot) noexcept {
    const Header zero{};
    (void)pwrite_all(file_handle_, &zero, sizeof zero, static_cast<std::uint64_t>(slot) * slot_pitch());
}

void DiskParkStore::lru_touch_locked(std::uint32_t slot, std::uint64_t stamp) {
    if (lru_stamps_[slot] != kNoStamp) lru_.erase({lru_stamps_[slot], slot});
    lru_stamps_[slot] = stamp;
    lru_.insert({stamp, slot});
}

void DiskParkStore::lru_erase_locked(std::uint32_t slot) noexcept {
    if (lru_stamps_[slot] != kNoStamp) lru_.erase({lru_stamps_[slot], slot});
    lru_stamps_[slot] = kNoStamp;
}

// ---------------------------------------------------------------- 索引

bool DiskParkStore::load_index() {
    const std::string index_path = options_.path + ".idx";
    std::FILE* f = std::fopen(index_path.c_str(), "rb");
    if (!f) return false;

    IndexHeader ih{};
    if (std::fread(&ih, sizeof ih, 1, f) != 1) { std::fclose(f); return false; }
    bool valid = ih.magic == IndexHeader::kMagic && ih.version == 1 && ih.count <= max_slots_ &&
                 ih.slot_size == static_cast<std::uint32_t>(slot_pitch()) && ih.max_slots == max_slots_;
    std::vector<IndexRow> rows;
    if (valid) {
        rows.resize(ih.count);
        valid = rows.empty() || std::fread(rows.data(), sizeof(IndexRow), rows.size(), f) == rows.size();
    }
    std::fclose(f);
    if (!valid) return false;

    std::vector<char> live(max_slots_, 0);
    index_.clear();
    for (const IndexRow& r : rows) {
        if (r.slot >= max_slots_) return false;
        const DiskParkIdentity id{r.lo, r.hi, r.tag, r.frontier, r.chunk};
        index_[id] = r.slot;
        live[r.slot] = 1;
    }
    // LRU 与时间戳仍然从槽头恢复 —— 索引里没有 last_used。
    // 逐个读头是 O(max_slots) 次 48 字节读，只在打开时发生一次。
    for (std::uint32_t slot = 0; slot < max_slots_; ++slot) {
        if (!live[slot]) continue;
        Header h{};
        if (!pread_all(file_handle_, &h, sizeof h, static_cast<std::uint64_t>(slot) * slot_pitch()) ||
            h.magic != Header::kMagic) {
            live[slot] = 0;
            index_.erase(DiskParkIdentity{h.lo, h.hi, h.tag, h.frontier, h.chunk});
            continue;
        }
        clock_ = std::max(clock_, h.last_used);
        lru_touch_locked(slot, h.last_used);
    }
    for (std::uint32_t slot = max_slots_; slot-- > 0;) {
        if (!live[slot]) free_slots_.push_back(slot);
    }
    return true;
}

void DiskParkStore::rebuild_from_scan() {
    // ★ 数据文件是权威的：扫每个槽头，CRC 对得上的才算活。
    index_.clear();
    lru_.clear();
    free_slots_.clear();
    std::fill(lru_stamps_.begin(), lru_stamps_.end(), kNoStamp);

    std::vector<std::byte> payload(options_.slot_size);
    for (std::uint32_t slot = 0; slot < max_slots_; ++slot) {
        Header h{};
        if (!pread_all(file_handle_, &h, sizeof h, static_cast<std::uint64_t>(slot) * slot_pitch())) break;
        if (h.magic != Header::kMagic) continue;
        if (!read_payload(slot, payload)) continue;
        if (options_.verify_crc && crc32(payload) != h.crc) continue;  // ★ 残缺页 = 没存
        index_[DiskParkIdentity{h.lo, h.hi, h.tag, h.frontier, h.chunk}] = slot;
        clock_ = std::max(clock_, h.last_used);
        lru_touch_locked(slot, h.last_used);
    }
    for (std::uint32_t slot = max_slots_; slot-- > 0;) {
        if (lru_stamps_[slot] == kNoStamp) free_slots_.push_back(slot);
    }
    // ★ 全新创建的空库不算"重建"。
    //   ★★ 不能用 file_had_content_：首次打开就 ftruncate 到满尺寸，
    //      之后 st_size 恒 > 0（哪怕一页都没写过），于是第二次起必然误报。
    //      判据应该是"扫描确实找回了条目"。
    rebuilt_from_scan_ = !index_.empty();
    index_dirty_       = true;
}

void DiskParkStore::persist_index_locked() {
    const std::string index_path = options_.path + ".idx";
    const std::string tmp_path   = index_path + ".tmp";
    std::FILE* f = std::fopen(tmp_path.c_str(), "wb");
    if (!f) return;                                     // ★ 索引写不出去不是致命错误

    IndexHeader ih{};
    ih.magic     = IndexHeader::kMagic;
    ih.version   = 1;
    ih.count     = static_cast<std::uint32_t>(index_.size());
    ih.slot_size = static_cast<std::uint32_t>(slot_pitch());
    ih.max_slots = max_slots_;
    ih.clock     = clock_;
    bool ok = std::fwrite(&ih, sizeof ih, 1, f) == 1;
    if (ok) {
        for (const auto& [id, slot] : index_) {
            IndexRow r{};
            r.lo = id.lo; r.hi = id.hi; r.tag = id.tag; r.frontier = id.frontier;
            r.chunk = id.chunk; r.slot = slot;
            if (std::fwrite(&r, sizeof r, 1, f) != 1) { ok = false; break; }
        }
    }
    if (std::fflush(f) != 0) ok = false;
    std::fclose(f);
    if (!ok) { std::remove(tmp_path.c_str()); return; }
    (void)std::rename(tmp_path.c_str(), index_path.c_str());   // 原子替换
    index_dirty_ = false;
}

void DiskParkStore::flush_index() {
    std::lock_guard lock(mutex_);
    if (index_dirty_) persist_index_locked();
}

// ---------------------------------------------------------------- 负载 I/O

bool DiskParkStore::write_payload(std::uint32_t slot, std::span<const std::byte> bytes) noexcept {
    return pwrite_all(file_handle_, bytes.data(), bytes.size(), payload_offset(slot));
}

bool DiskParkStore::read_payload(std::uint32_t slot, std::span<std::byte> destination) noexcept {
    return pread_all(file_handle_, destination.data(), destination.size(), payload_offset(slot));
}

// ---------------------------------------------------------------- 查询

std::uint32_t DiskParkStore::live_slots() const {
    std::lock_guard lock(mutex_);
    return static_cast<std::uint32_t>(index_.size());
}

std::size_t DiskParkStore::used_bytes() const {
    std::lock_guard lock(mutex_);
    return index_.size() * slot_pitch();
}

bool DiskParkStore::contains(const DiskParkIdentity& id) const {
    std::lock_guard lock(mutex_);
    return index_.find(id) != index_.end();
}

bool DiskParkStore::touch(const DiskParkIdentity& id) {
    std::lock_guard lock(mutex_);
    const auto it = index_.find(id);
    if (it == index_.end()) return false;
    lru_touch_locked(it->second, ++clock_);
    index_dirty_ = true;
    return true;
}

std::vector<DiskParkIdentity> DiskParkStore::live_identities() const {
    std::lock_guard lock(mutex_);
    std::vector<DiskParkIdentity> out;
    out.reserve(index_.size());
    for (const auto& [id, slot] : index_) out.push_back(id);
    return out;
}

bool DiskParkStore::evict(const DiskParkIdentity& id) {
    std::lock_guard lock(mutex_);
    const auto it = index_.find(id);
    if (it == index_.end()) return false;
    const std::uint32_t slot = it->second;
    if (readers_[slot] != 0) return false;              // 正在被读，不能动
    clear_slot(slot);
    lru_erase_locked(slot);
    index_.erase(it);
    free_slots_.push_back(slot);
    index_dirty_ = true;
    return true;
}

// ---------------------------------------------------------------- 淘汰

bool DiskParkStore::evict_lru_entry_locked(std::vector<DiskParkIdentity>* evicted) {
    // 纯 LRU：最久未用的条目（同样整条一起丢）
    for (auto it = lru_.begin(); it != lru_.end(); ++it) {
        const std::uint32_t slot = it->second;
        if (readers_[slot] != 0) continue;
        const DiskParkIdentity* id = nullptr;
        for (const auto& kv : index_) if (kv.second == slot) { id = &kv.first; break; }
        if (!id) { lru_erase_locked(slot); clear_slot(slot); free_slots_.push_back(slot); continue; }
        const DiskParkIdentity victim = *id;
        for (auto jt = index_.begin(); jt != index_.end();) {
            if (jt->first.lo == victim.lo && jt->first.hi == victim.hi &&
                jt->first.tag == victim.tag && jt->first.frontier == victim.frontier) {
                const std::uint32_t s2 = jt->second;
                jt = index_.erase(jt);
                clear_slot(s2); lru_erase_locked(s2); free_slots_.push_back(s2);
                if (evicted) evicted->push_back(DiskParkIdentity{victim.lo, victim.hi, victim.tag, victim.frontier, 0});
            } else ++jt;
        }
        index_dirty_ = true;
        return true;
    }
    return false;
}

bool DiskParkStore::evict_cheapest_locked(std::vector<DiskParkIdentity>* evicted) {
    // ★ 候选窗口：最久未用的那一批（至少 4 个，最多 1/4 条目）。
    //   只看最旧的一批，避免"一条 155K 的大条目永远留着、小的无限轮换"。
    const std::size_t window = std::max<std::size_t>(4, index_.size() / 4);
    const DiskParkIdentity* best = nullptr;
    std::uint32_t best_slot = 0;
    std::uint64_t best_stamp = 0;
    std::size_t seen = 0;
    for (auto it = lru_.begin(); it != lru_.end() && seen < window; ++it) {
        const std::uint32_t slot = it->second;
        if (readers_[slot] != 0) continue;              // ★ 跳过正在被读的
        const DiskParkIdentity* id = nullptr;
        for (const auto& kv : index_) if (kv.second == slot) { id = &kv.first; break; }
        if (!id) {                                      // 索引与 LRU 不一致（不该发生）
            lru_erase_locked(slot); clear_slot(slot); free_slots_.push_back(slot); continue;
        }
        ++seen;
        // 价值 = 重建成本 ≈ frontier；同等则取更久未用的
        if (!best || id->frontier < best->frontier ||
            (id->frontier == best->frontier && it->first < best_stamp)) {
            best = id; best_slot = slot; best_stamp = it->first;
        }
    }
    if (!best) return false;                            // 全都在被读 / 没有条目
    const DiskParkIdentity victim = *best;              // ★ 先拷贝，下面要 erase
    // ★ 整个条目一起淘汰
    for (auto it = index_.begin(); it != index_.end();) {
        if (it->first.lo == victim.lo && it->first.hi == victim.hi &&
            it->first.tag == victim.tag && it->first.frontier == victim.frontier) {
            const std::uint32_t s = it->second;
            it = index_.erase(it);
            clear_slot(s);
            lru_erase_locked(s);
            free_slots_.push_back(s);
            if (evicted) evicted->push_back(DiskParkIdentity{
                victim.lo, victim.hi, victim.tag, victim.frontier, 0});
        } else {
            ++it;
        }
    }
    index_dirty_ = true;
    (void) best_slot;
    return true;
}

// ---------------------------------------------------------------- 读写

bool DiskParkStore::upsert_page(const DiskParkIdentity& id, std::span<const std::byte> bytes,
                                std::vector<DiskParkIdentity>* evicted) {
    if (bytes.size() != options_.slot_size) return false;

    std::lock_guard lock(mutex_);
    std::uint32_t slot = 0;
    // ★★ 只有【新分配】的槽才能在写失败时回收。
    //    已存在的条目用的是活槽 —— 把它塞进 free list 会让下一个 upsert
    //    再取到同一个槽覆写，于是两个身份指向同一槽（contains() 撒谎，
    //    read_page 静默 miss），而 spool 的 find() 会一直选中这个幻影。
    bool fresh_slot = false;

    if (const auto it = index_.find(id); it != index_.end()) {
        slot = it->second;                              // 已存在：只刷新 LRU
    } else if (!free_slots_.empty()) {
        slot = free_slots_.back();
        free_slots_.pop_back();
        fresh_slot = true;
    } else {
        // ★ 满则按"价值感知 + 条目级"淘汰（可能一次腾出多个槽）
        const bool ok_evict = options_.value_aware_demote
                                 ? evict_cheapest_locked(evicted)
                                 : evict_lru_entry_locked(evicted);
        if (!ok_evict) return false;                        // 全都在被读
        // evict_one_lru_locked 已经把槽放回 free_slots_
        if (free_slots_.empty()) return false;
        slot = free_slots_.back();
        free_slots_.pop_back();
        fresh_slot = true;
    }

    Header h{};
    h.magic     = Header::kMagic;
    h.last_used = ++clock_;
    h.lo        = id.lo;
    h.hi        = id.hi;
    h.tag       = id.tag;
    h.frontier  = id.frontier;
    h.chunk     = id.chunk;
    h.crc       = options_.verify_crc ? crc32(bytes) : 0;

    // ★ 先写负载，再写头。头是提交点：头没写成功，这个槽就还不算存在。
    if (!write_payload(slot, bytes)) { if (fresh_slot) free_slots_.push_back(slot); return false; }
    if (!pwrite_all(file_handle_, &h, sizeof h, static_cast<std::uint64_t>(slot) * slot_pitch())) {
        if (fresh_slot) free_slots_.push_back(slot);
        return false;
    }
    // 防御：写失败后残留的旧头不会被误认（magic 已被覆盖为 0 或新值，
    // 但若 pwrite 部分成功，CRC 也拦得住）。
    index_[id] = slot;
    lru_touch_locked(slot, h.last_used);
    if (!options_.defer_index_updates) persist_index_locked(); else index_dirty_ = true;
    return true;
}

bool DiskParkStore::read_page(const DiskParkIdentity& id, std::span<std::byte> destination) {
    if (destination.size() != options_.slot_size) return false;

    std::uint32_t slot = 0;
    std::uint32_t want_crc = 0;
    {
        std::lock_guard lock(mutex_);
        const auto it = index_.find(id);
        if (it == index_.end()) return false;           // ★ miss
        slot = it->second;
        readers_[slot]++;                               // ★ 读期间禁止淘汰
        Header h{};
        if (!pread_all(file_handle_, &h, sizeof h, static_cast<std::uint64_t>(slot) * slot_pitch()) ||
            !matches(h, id)) {
            readers_[slot]--;
            return false;
        }
        want_crc = h.crc;
    }

    const bool io_ok = read_payload(slot, destination);   // ★ 拷贝与 CRC 在锁外
    const bool intact = io_ok && (!options_.verify_crc || crc32(destination) == want_crc);

    std::lock_guard lock(mutex_);
    readers_[slot]--;
    if (intact) lru_touch_locked(slot, ++clock_);
    return intact;
}

} // namespace strata::core
