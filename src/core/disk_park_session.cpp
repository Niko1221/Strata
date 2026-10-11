#include "strata/core/disk_park_session.hpp"
#include "strata/core/disk_park_digest.hpp"

#include <algorithm>
#include <array>
#include <cerrno>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <system_error>

#include <fcntl.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

namespace strata::core {
namespace {

// ---- 小端序列化：不依赖宿主字节序，也不依赖结构体布局 ----
void put_u32(std::uint8_t* p, std::uint32_t v) { std::memcpy(p, &v, 4); }
void put_u64(std::uint8_t* p, std::uint64_t v) { std::memcpy(p, &v, 8); }
void put_i64(std::uint8_t* p, std::int64_t v)  { std::memcpy(p, &v, 8); }
std::uint32_t get_u32(const std::uint8_t* p) { std::uint32_t v = 0; std::memcpy(&v, p, 4); return v; }
std::uint64_t get_u64(const std::uint8_t* p) { std::uint64_t v = 0; std::memcpy(&v, p, 8); return v; }
std::int64_t  get_i64(const std::uint8_t* p) { std::int64_t  v = 0; std::memcpy(&v, p, 8); return v; }

bool pread_all(int fd, void* dst, std::size_t n, std::uint64_t offset) {
    auto* p = static_cast<std::uint8_t*>(dst);
    std::size_t done = 0;
    while (done < n) {
        const ssize_t r = ::pread(fd, p + done, n - done, static_cast<off_t>(offset + done));
        if (r < 0) { if (errno == EINTR) continue; return false; }
        if (r == 0) return false;                       // 文件比预期短
        done += static_cast<std::size_t>(r);
    }
    return true;
}

bool pwrite_all(int fd, const void* src, std::size_t n, std::uint64_t offset) {
    const auto* p = static_cast<const std::uint8_t*>(src);
    std::size_t done = 0;
    while (done < n) {
        const ssize_t r = ::pwrite(fd, p + done, n - done, static_cast<off_t>(offset + done));
        if (r < 0) { if (errno == EINTR) continue; return false; }
        if (r == 0) return false;
        done += static_cast<std::size_t>(r);
    }
    return true;
}

bool fsync_dir(const std::string& dir) {
    const int dfd = ::open(dir.c_str(), O_RDONLY | O_DIRECTORY);
    if (dfd < 0) return false;
    const int rc = ::fsync(dfd);
    ::close(dfd);
    return rc == 0;
}

bool stat_size(const std::string& path, std::uint64_t& size, bool& exists) {
    struct stat st {};
    if (::stat(path.c_str(), &st) != 0) { exists = false; size = 0; return false; }
    exists = true;
    size = static_cast<std::uint64_t>(st.st_size);
    return true;
}

constexpr std::size_t kIoChunk = 16u << 20;    // 与 session_file 的写缓冲同规格

// 覆盖【全部 part 几何 + 容量】的校验和：头里放不下几何，用它代替逐字段比对。
std::uint64_t geometry_sum(const DiskParkSessionPlan& p) {
    std::uint64_t h = 1469598103934665603ull;
    auto mix = [&h](std::uint64_t v) { h ^= v; h *= 1099511628211ull; };
    mix(p.capacity_cells);
    for (const auto& q : p.parts) { mix(q.bytes_per_cell); mix(q.row_cells); mix(q.row_bytes); }
    return h;
}

} // namespace

// ------------------------------------------------------------------ plan

bool DiskParkSessionPlan::valid() const noexcept {
    if (heads <= 0 || head_dim <= 0 || page_size <= 0 || idx_dim <= 0) return false;
    bool any = false;
    for (const auto& p : parts) {
        if (p.row_cells == 0) return false;
        if (p.bytes_per_cell == 0) continue;      // 该 part 在本 profile 下不存在
        any = true;
        if (p.row_cells > 1 && p.row_bytes == 0) return false;   // 成行 part 必须给行宽
    }
    return any;
}

// ------------------------------------------------------------------ 生命周期

DiskParkSession::~DiskParkSession() { close(); }

DiskParkSession::DiskParkSession(DiskParkSession&& o) noexcept
    : dir_(std::move(o.dir_)), plan_(o.plan_), fd_(o.fd_), log_cells_(o.log_cells_) {
    o.fd_ = -1;
}

DiskParkSession& DiskParkSession::operator=(DiskParkSession&& o) noexcept {
    if (this == &o) return *this;
    close();
    dir_ = std::move(o.dir_);
    plan_ = o.plan_;
    fd_ = o.fd_;
    log_cells_ = o.log_cells_;
    o.fd_ = -1;
    return *this;
}

void DiskParkSession::close() noexcept {
    if (fd_ >= 0) { ::close(fd_); fd_ = -1; }
}

bool DiskParkSession::write_header() const {
    // ★ 头只有 64 B —— 放不下 5 个 part 的完整几何。所以：
    //   ① 逐字段写下【用于快速判等】的标量（tag / 模式 / 几何 / 两个代表性的 part 宽度）
    //   ② 再写一个覆盖【全部 part 几何 + 容量】的校验和（geometry_sum）。
    //   几何本身由调用方的 plan 给出：read_header 会用同一个公式重算校验和，
    //   对不上就拒绝打开 —— 这比把几何塞进头里更省，且不会漏项。
    //   段偏移【刻意不写进文件】：它由几何现算，写进去只会多一个可能与真相对不上的副本。
    const std::uint64_t sum = geometry_sum(plan_);
    std::uint8_t h[kHeaderBytes] = {};
    put_u32(h + 0,  kKvMagic);
    put_u32(h + 4,  kKvVersion);
    put_u32(h + 8,  plan_.tag);
    put_u32(h + 12, static_cast<std::uint32_t>(plan_.kv_format));
    put_i64(h + 16, plan_.heads);
    put_i64(h + 24, plan_.head_dim);
    put_i64(h + 32, plan_.page_size);
    put_i64(h + 40, plan_.idx_dim);
    put_u64(h + 48, sum);
    put_u32(h + 56, static_cast<std::uint32_t>(plan_.parts[0].bytes_per_cell));
    put_u32(h + 60, static_cast<std::uint32_t>(plan_.parts[4].bytes_per_cell));
    static_assert(kHeaderBytes == 64, "头长写死在布局里");
    return pwrite_all(fd_, h, sizeof h, 0) && ::fsync(fd_) == 0;
}

bool DiskParkSession::read_header() {
    std::uint8_t h[kHeaderBytes] = {};
    if (!pread_all(fd_, h, sizeof h, 0)) return false;
    if (get_u32(h + 0) != kKvMagic) return false;
    if (get_u32(h + 4) != kKvVersion) return false;
    if (get_u32(h + 8) != plan_.tag) return false;                       // ★ profile 隔离
    if (static_cast<std::int64_t>(get_u32(h + 12)) != plan_.kv_format) return false;
    if (get_i64(h + 16) != plan_.heads) return false;
    if (get_i64(h + 24) != plan_.head_dim) return false;
    if (get_i64(h + 32) != plan_.page_size) return false;
    if (get_i64(h + 40) != plan_.idx_dim) return false;
    if (get_u64(h + 48) != geometry_sum(plan_)) return false;            // ★ 几何变了就不读
    if (get_u32(h + 56) != static_cast<std::uint32_t>(plan_.parts[0].bytes_per_cell)) return false;
    if (get_u32(h + 60) != static_cast<std::uint32_t>(plan_.parts[4].bytes_per_cell)) return false;
    return true;
}

bool DiskParkSession::open(DiskParkSession& out, const std::string& dir,
                           const DiskParkSessionPlan& plan, std::uint64_t resume_cells) {
    out.close();
    out.dir_.clear();
    out.log_cells_ = 0;
    if (dir.empty() || !plan.valid()) return false;

    std::error_code ec;
    if (!std::filesystem::exists(dir, ec)) {
        std::filesystem::create_directories(dir, ec);
        if (ec) return false;
    } else if (!std::filesystem::is_directory(dir, ec)) {
        return false;
    }
    out.dir_ = dir;
    out.plan_ = plan;

    std::uint64_t sz = 0;
    bool have = false;
    stat_size(out.kv_path(), sz, have);

    int fd = -1;
    if (have && sz >= kHeaderBytes) {
        fd = ::open(out.kv_path().c_str(), O_RDWR);
        if (fd < 0) { out.dir_.clear(); return false; }
        out.fd_ = fd;
        if (!out.read_header()) {                 // 与 plan 不符 → 这一层当作没有，且不改文件
            out.close();
            out.dir_.clear();
            return false;
        }
    } else if (!have) {
        fd = ::open(out.kv_path().c_str(), O_RDWR | O_CREAT, 0644);
        if (fd < 0) { out.dir_.clear(); return false; }
        out.fd_ = fd;
        if (!out.write_header()) { out.close(); out.dir_.clear(); return false; }
    } else {
        // 有文件但连头都不完整：上一次 create 半途而死，且没有 sidecar → 当作没有。
        // ★ 不删它：删文件是调用方的决定（这一层永远只读地放弃）。
        out.dir_.clear();
        return false;
    }

    // 水位来源优先级：sidecar（提交点）> 调用方给的 resume_cells（持有者的记忆）
    std::vector<std::byte> payload;
    std::uint64_t cells = 0;
    bool have_cells = out.read_meta(payload, cells);
    if (!have_cells && resume_cells > 0) { cells = resume_cells; have_cells = true; }
    if (have_cells) {
        bool fits = cells <= (1ull << 40);
        std::uint64_t need = kHeaderBytes;
        for (std::size_t i = 0; i < 5; ++i)
            if (out.plan_.parts[i].bytes_per_cell != 0)
                need = out.part_end(static_cast<Part>(i), cells);
        std::uint64_t sz2 = 0;
        bool have2 = false;
        stat_size(out.kv_path(), sz2, have2);
        if (fits && have2 && sz2 >= need) out.log_cells_ = cells;
    }
    return true;
}

// ------------------------------------------------------------------ 几何

std::uint64_t DiskParkSession::part_bytes(Part part, std::uint64_t cells) const noexcept {
    const auto& p = plan_.parts[static_cast<std::size_t>(part)];
    if (p.bytes_per_cell == 0) return 0;
    if (p.row_cells <= 1) return cells * p.bytes_per_cell;
    const std::uint64_t rows = cells / p.row_cells;
    const std::uint64_t rem  = cells % p.row_cells;
    return rows * p.row_bytes + rem * p.bytes_per_cell;
}

std::uint64_t DiskParkSession::base_bytes() const noexcept {
    // 段偏移的基准【不含水位】：每个 part 的起点由几何决定，与写到哪无关。
    return kHeaderBytes;
}

std::uint64_t DiskParkSession::part_end(Part part, std::uint64_t cells) const noexcept {
    return part_offset(part, cells) + part_bytes(part, cells);
}

std::uint64_t DiskParkSession::part_offset(Part part, std::uint64_t cell) const noexcept {
    const auto& p = plan_.parts[static_cast<std::size_t>(part)];
    if (p.bytes_per_cell == 0) return 0;
    std::uint64_t off = base_bytes();
    for (std::size_t i = 0; i < static_cast<std::size_t>(part); ++i)
        if (plan_.parts[i].bytes_per_cell != 0) off += part_capacity_bytes(static_cast<Part>(i));
    if (p.row_cells <= 1) return off + cell * p.bytes_per_cell;
    const std::uint64_t rows = cell / p.row_cells;
    return off + rows * p.row_bytes + (cell % p.row_cells) * p.bytes_per_cell;
}

std::uint64_t DiskParkSession::part_capacity_bytes(Part part) const noexcept {
    // ★ 固定容量：段偏移绝不能依赖水位，否则落在段中间的写入会错位。
    return part_bytes(part, plan_.capacity_cells);
}

// ------------------------------------------------------------------ 写

bool DiskParkSession::truncate_to(std::uint64_t cells) {
    // 每个 part 各自截到它在 cells 处的末尾。成行 part（pooled）保留 cells 所在的
    // 【整行】—— 下一次追加会重写这一行，而行内的旧字节本来就要被覆盖。
    std::uint64_t end = base_bytes();
    for (std::size_t i = 0; i < 5; ++i) {
        const auto& p = plan_.parts[i];
        if (p.bytes_per_cell == 0) continue;
        const auto part = static_cast<Part>(i);
        std::uint64_t cut = part_offset(part, cells);
        if (p.row_cells > 1) cut += p.row_bytes;
        end = std::max(end, cut);
    }
    if (::ftruncate(fd_, static_cast<off_t>(end)) != 0) return false;
    log_cells_ = cells;
    return true;
}

std::uint64_t DiskParkSession::append_kv(std::uint64_t from_cell, std::uint64_t to_cell,
                                          const DiskParkFill& fill) {
    static bool s_ap_dbg = false;
    auto why = [&](const char* w) {
        if (!s_ap_dbg) { s_ap_dbg = true;
            std::fprintf(stderr, "strata serve: disk park stream: [append_kv] 失败于 %s "
                         "(from=%llu to=%llu cells=%llu ok=%d fd=%d errno=%d)\n", w,
                         (unsigned long long) from_cell, (unsigned long long) to_cell,
                         (unsigned long long) log_cells_, (int) ok(), fd_, errno);
        }
        return 0;
    };
    if (!ok() || to_cell <= from_cell || !fill) return why("入口检查");

    // ★★ 回退：每个 part 各自截到 from_cell 处，不能只截最后一个 ——
    //    后者会把前面所有 part 的数据一起砍掉（实测：回退到 120 时 K 被清空）。
    if (from_cell < log_cells_ && !truncate_to(from_cell)) return why("truncate_to");

    std::uint64_t written = 0;
    std::vector<std::uint8_t> buf(kIoChunk);
    for (std::size_t i = 0; i < 5; ++i) {
        const auto& p = plan_.parts[i];
        if (p.bytes_per_cell == 0) continue;
        const auto part = static_cast<Part>(i);
        const std::uint64_t at = part_offset(part, from_cell);
        const std::uint64_t n  = part_bytes(part, to_cell) - part_bytes(part, from_cell);
        // ★★★ 修（2026-10-10）：块大小必须【按该 part 的 cell/行对齐】。
        //   原来固定用 16 MiB：V 是 288 B/cell 时 `16 MiB % 288 = 64 ≠ 0`，
        //   于是第二块的 offset 不是 cell 边界 ⇒ 回调（disk_park_logs 的
        //   "必须 cell 对齐"检查 / p9_read_host_pool）返回 false ⇒ 整层放弃。
        //   实测：V 段 > 16 MiB（≈ 58,254 token）的会话【一个字节都写不进】——
        //   135k token 的真实请求就是这样被静默丢掉的，取回侧只能冷读。
        //   向下取整到对齐量：偏移因此永远是 cell（成行 part 是行）边界。
        const std::uint64_t align = p.row_cells > 1 ? p.row_bytes : p.bytes_per_cell;
        const std::size_t chunk = (align > 0 && buf.size() >= align)
                                      ? static_cast<std::size_t>(buf.size() - (buf.size() % align))
                                      : buf.size();
        std::uint64_t done = 0;
        while (done < n) {
            const std::size_t take =
                static_cast<std::size_t>(std::min<std::uint64_t>(chunk, n - done));
            // ★ 传给 fill 的是【相对该 part 段起点的偏移】（= part 内第几字节），
            //   与 read_kv 的语义一致 —— 两边都用它算 cell，才能逐字节对上。
            const std::size_t seg = static_cast<std::size_t>(part_offset(part, 0));
            if (!fill(i, static_cast<std::size_t>(at) - seg + static_cast<std::size_t>(done),
                      buf.data(), take)) return why("fill");
            if (!pwrite_all(fd_, buf.data(), take, at + done)) return why("pwrite");
            done += take;
            written += take;
        }
    }

    // 文件长到最后一段的末尾（中间没写的洞是稀疏的，不占盘）
    std::uint64_t tail = base_bytes();
    for (std::size_t i = 0; i < 5; ++i)
        if (plan_.parts[i].bytes_per_cell != 0)
            tail = part_end(static_cast<Part>(i), to_cell);
    if (::ftruncate(fd_, static_cast<off_t>(tail)) != 0) return why("ftruncate");
    log_cells_ = to_cell;
    if (::fsync(fd_) != 0) return why("fsync");
    return written;
}

bool DiskParkSession::commit(std::uint64_t to_cell, std::span<const std::byte> meta_packed,
                             std::uint64_t kv_at) {
    // ★ 允许空载荷：那就是"只把水位提交下去"。阶段 2 的追加路径正是这样用的 ——
    //   镜像元数据（检查点链等）还没接，但水位必须落盘，否则重启后不知道从哪续写。
    if (!ok()) return false;
    const std::string tmp = state_path() + ".tmp";
    const int fd = ::open(tmp.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0644);
    if (fd < 0) return false;

    std::uint8_t h[kMetaHeaderBytes] = {};
    put_u32(h + 0,  kMetaMagic);
    put_u32(h + 4,  kMetaVersion);
    put_u32(h + 8,  plan_.tag);
    if (kv_at > 0xFFFFFFFFull) return false;        // 头里只有 32 位
    put_u32(h + 12, static_cast<std::uint32_t>(kv_at));
    put_u64(h + 16, static_cast<std::uint64_t>(meta_packed.size()));
    put_u64(h + 24, to_cell);

    bool okw = pwrite_all(fd, h, sizeof h, 0) &&
               pwrite_all(fd, meta_packed.data(), meta_packed.size(), kMetaHeaderBytes);
    if (okw) okw = ::fsync(fd) == 0;
    ::close(fd);
    if (!okw) { ::unlink(tmp.c_str()); return false; }

    if (::rename(tmp.c_str(), state_path().c_str()) != 0) { ::unlink(tmp.c_str()); return false; }
    (void) fsync_dir(dir_);                      // 目录项落盘；失败不算致命
    log_cells_ = to_cell;
    return true;
}

// ------------------------------------------------------------------ 读

bool DiskParkSession::read_meta(std::vector<std::byte>& payload, std::uint64_t& log_cells,
                                std::uint64_t* kv_at) const {
    payload.clear();
    log_cells = 0;
    if (kv_at) *kv_at = 0;
    if (dir_.empty()) return false;
    std::uint64_t sz = 0;
    bool exists = false;
    stat_size(state_path(), sz, exists);
    if (!exists || sz < kMetaHeaderBytes) return false;

    const int fd = ::open(state_path().c_str(), O_RDONLY);
    if (fd < 0) return false;
    std::uint8_t h[kMetaHeaderBytes] = {};
    bool ok = pread_all(fd, h, sizeof h, 0);
    if (ok)
        ok = get_u32(h + 0) == kMetaMagic && get_u32(h + 4) == kMetaVersion &&
             get_u32(h + 8) == plan_.tag;
    const std::uint64_t plen  = ok ? get_u64(h + 16) : 0;
    const std::uint64_t cells = ok ? get_u64(h + 24) : 0;
    // plen == 0 是合法的（"只有水位"的 sidecar）；非 0 时必须 >= 8（镜像至少含 magic+version）
    ok = ok && plen <= (1ull << 32) && sz == kMetaHeaderBytes + plen && (plen == 0 || plen >= 8);
    if (ok) {
        payload.resize(static_cast<std::size_t>(plen));
        ok = pread_all(fd, payload.data(), payload.size(), kMetaHeaderBytes);
    }
    ::close(fd);
    if (!ok) { payload.clear(); return false; }
    log_cells = cells;
    if (kv_at) *kv_at = get_u32(h + 12);            // 老 sidecar 这里是 0 ⇒ 只有水位
    return true;
}

namespace {

void put_u64_le(std::uint8_t* p, std::uint64_t v) { std::memcpy(p, &v, 8); }
std::uint64_t get_u64_le(const std::uint8_t* p) { std::uint64_t v = 0; std::memcpy(&v, p, 8); return v; }

} // namespace

bool parse_ckpt_dir(std::span<const std::byte> payload,
                    std::vector<DiskParkCkptDirEntry>& out, std::uint64_t& image_bytes) {
    out.clear();
    image_bytes = payload.size();
    if (payload.size() < 12) return false;
    const auto* b = reinterpret_cast<const std::uint8_t*>(payload.data());
    std::uint32_t magic = 0;
    std::memcpy(&magic, b + payload.size() - 4, 4);
    if (magic != kCkptDirMagic) return false;                       // 没有目录
    const std::uint64_t n = get_u64_le(b + payload.size() - 12);
    if (n > (1u << 20)) return false;                                // 防御
    const std::uint64_t dir_bytes = 12 + 32 * n;
    if (dir_bytes > payload.size()) return false;
    const std::size_t base = payload.size() - static_cast<std::size_t>(dir_bytes);
    out.reserve(static_cast<std::size_t>(n));
    for (std::uint64_t i = 0; i < n; ++i) {
        const std::size_t at = base + static_cast<std::size_t>(i) * 32;
        DiskParkCkptDirEntry e;
        e.lo  = get_u64_le(b + at + 0);
        e.hi  = get_u64_le(b + at + 8);
        e.off = get_u64_le(b + at + 16);
        e.len = get_u64_le(b + at + 24);
        if (e.off + e.len > payload.size()) return false;            // 引用必须落在载荷内
        out.push_back(e);
    }
    image_bytes = base;                                              // 镜像本体到此为止
    return true;
}

void append_ckpt_dir(std::vector<std::byte>& image,
                     const std::vector<DiskParkCkptDirEntry>& entries) {
    const std::size_t at = image.size();
    image.resize(at + 32 * entries.size() + 12);
    auto* b = reinterpret_cast<std::uint8_t*>(image.data());
    for (std::size_t i = 0; i < entries.size(); ++i) {
        const std::size_t o = at + i * 32;
        put_u64_le(b + o + 0,  entries[i].lo);
        put_u64_le(b + o + 8,  entries[i].hi);
        put_u64_le(b + o + 16, entries[i].off);
        put_u64_le(b + o + 24, entries[i].len);
    }
    put_u64_le(b + at + 32 * entries.size(), entries.size());
    const std::uint32_t magic = kCkptDirMagic;
    std::memcpy(b + at + 32 * entries.size() + 8, &magic, 4);
}

bool extract_ckpt_digests(std::span<const std::byte> payload, std::size_t expected_layers,
                          std::vector<PrefixDigest>& out) {
    out.clear();
    if (payload.empty()) return false;
    const auto* base = reinterpret_cast<const std::uint8_t*>(payload.data());
    auto rd = [&](std::size_t at) -> std::uint64_t {
        std::uint64_t v = 0;
        if (at + 8 <= payload.size()) std::memcpy(&v, base + at, 8);
        return v;
    };

    // 跳过：magic/ver(8) + geometry(18×8) + layer_lo/hi(16)
    std::size_t at = 8 + 18 * 8 + 8 + 8;
    std::vector<std::int32_t> ids;
    std::vector<std::uint64_t> no_hashes;      // 摘要只需要 ids（图片键由 ids 决定的顺序折入）
    std::vector<std::int64_t>  no_starts;

    std::function<bool()> walk = [&]() -> bool {
        if (at + 8 > payload.size()) return false;
        const std::uint64_t n_ids = rd(at);
        at += 8;
        if (at + static_cast<std::size_t>(n_ids) * 4 > payload.size()) return false;
        ids.assign(static_cast<std::size_t>(n_ids), 0);
        if (n_ids) std::memcpy(ids.data(), base + at, static_cast<std::size_t>(n_ids) * 4);
        at += static_cast<std::size_t>(n_ids) * 4;
        out.push_back(prefix_digest(ids, no_hashes, no_starts, static_cast<std::int64_t>(ids.size())));
        // imgs
        if (at + 8 > payload.size()) return false;
        const std::uint64_t n_img = rd(at);
        at += 8 + static_cast<std::size_t>(n_img) * 16;
        if (at > payload.size()) return false;
        // 5 个缓冲区
        for (int p = 0; p < 5; ++p) {
            if (at + 8 > payload.size()) return false;
            const std::uint64_t n = rd(at);
            at += 8 + static_cast<std::size_t>(n);
            if (at > payload.size()) return false;
        }
        // used
        if (at + 8 > payload.size()) return false;
        at += 8;
        // stage_parts 计数 + 子检查点
        if (at + 8 > payload.size()) return false;
        const std::uint64_t n_parts = rd(at);
        at += 8;
        for (std::uint64_t k = 0; k < n_parts; ++k) if (!walk()) return false;
        return true;
    };

    if (!walk()) return false;                                   // live
    if (at + 8 > payload.size()) return false;
    const std::uint64_t n_ck = rd(at);
    at += 8;
    for (std::uint64_t k = 0; k < n_ck; ++k) if (!walk()) return false;
    // 自检：走完应当落在【KV 层数】字段上，其值 == expected_layers
    if (expected_layers != 0) {
        if (at + 8 > payload.size() || rd(at) != expected_layers) return false;
    }
    return true;
}

std::uint64_t DiskParkSession::splice_image(std::span<const std::byte> payload,
                                            std::uint64_t kv_at, std::size_t layers,
                                            std::uint64_t from_cell, std::uint64_t cells,
                                            const DiskParkFill& fill,
                                            const std::function<bool(const void*, std::size_t)>& sink,
                                            const DiskParkCkptFill& fill_ckpt,
                                            std::uint64_t binary_bytes) const {
    if (!ok() || payload.empty() || kv_at == 0 || kv_at + 16 > payload.size() ||
        layers == 0 || cells == 0 || from_cell >= cells || !fill || !sink) return 0;
    const auto* base = reinterpret_cast<const std::uint8_t*>(payload.data());
    auto rd_u64 = [&](std::size_t at) -> std::uint64_t {
        std::uint64_t v = 0;
        if (at + 8 <= payload.size()) std::memcpy(&v, base + at, 8);
        return v;
    };

    std::uint64_t total = 0;
    std::size_t cur = 0;
    auto emit = [&](const void* p, std::size_t n) -> bool {
        if (n == 0) return true;
        if (!sink(p, n)) return false;
        total += n;
        return true;
    };
    auto emit_payload = [&](std::size_t to) -> bool {
        if (to < cur || to > payload.size()) return false;
        const bool ok2 = emit(base + cur, to - cur);
        cur = to;
        return ok2;
    };

    // ★★ 步长由【plan_】给，不读载荷里的长度字段。
    //   为什么：载荷里的长度字段是【上游按 layout 算出来的】（不是缓冲区当前的 size），
    //   而拼接要的步长是【这次实际放进日志的那个范围】（= ceil(L/page)*page 那一套）。
    //   两者在"最后一个不满页"上会差几个 cell，读错位置就会一路错到底
    //   （实测：need 到 5.1 GB；也出现过精确少 8 字节）。
    //   —— step = part_bytes(part, cells) - part_bytes(part, from_cell)
    // ★★ 从 0 开始吐：live / checkpoints / KV 层 / 尾部 全都在这条流里。
    //   第一版从 kv_at+8 起步，于是 walk_ckpt 把 KV 层的数据当成了检查点
    //   （实测：n_ids 读出 1288490188803 = 0x12C00000003）。
    //   但要先把【头】吐掉：magic/version(8) + geometry + layer_lo/hi。
    //   直接 cur = 0 会让 walk_ckpt 把 magic/version 当成长度读（实测 n_ids=5121986643）。
    if (!emit_payload(8 + 18 * 8 + 8 + 8)) return 0;   // magic/ver + geometry[18] + lo/hi

    // ★★ P12：先处理 live 与检查点链 —— 它们夹在"头部"与"KV 层"之间。
    //   载荷尾部的目录指出哪些状态被搬进了日志；搬了的那些，走到它的缓冲区时
    //   跳过载荷里的 0 占位、改从日志取真值。
    std::vector<DiskParkCkptDirEntry> dir;
    std::uint64_t image_bytes = payload.size();
    const bool has_dir = parse_ckpt_dir(payload, dir, image_bytes);
    // ★ 产出只到【镜像本体】为止；本体之后可能追加了目录（那不进产出流）。
    const std::size_t body = (binary_bytes != 0)
        ? static_cast<std::size_t>(binary_bytes)
        : static_cast<std::size_t>(image_bytes);
    if (body > payload.size()) return 0;

    // 在目录里按摘要找；live 用哨兵键
    // ★ 只用两个 u64 当键 —— 不引入 disk_park_logs 的类型（那会形成循环依赖：
    //   logs 依赖 session）。目录项本身就是 {lo, hi, ...}，够用。
    auto dir_find = [&](std::uint64_t lo, std::uint64_t hi) -> const DiskParkCkptDirEntry* {
        if (!has_dir) return nullptr;
        for (const auto& e : dir)
            if (e.lo == lo && e.hi == hi) return e.present() ? &e : nullptr;
        return nullptr;
    };

    std::vector<std::uint8_t> buf(1u << 20);   // 共用的搬运缓冲（KV 与检查点都用）
    std::vector<std::uint64_t> ck_hashes;      // 本层不用图片；留给将来
    std::vector<std::int64_t>  ck_starts;
    std::vector<std::int32_t>  ck_ids;
    std::size_t ck_index = 0;                  // 0 = live，1.. = checkpoints[ck_index-1]
    // ★★ 日志里一条记录是【5 个缓冲区首尾相接】的，所以给 fill_ckpt 的 offset 必须是
    //   "记录内累计偏移"，不能在每个 part 里从 0 重来 —— 后者会让后 4 个缓冲区
    //   全都从日志开头读（症状：fill 被调用但内容不对，或长度对不上）。
    std::size_t ck_rec_off = 0;

    // 走一个检查点：ids / imgs / 5 个缓冲区 / used / stage_parts
    std::function<bool()> walk_ckpt = [&]() -> bool {
        ck_rec_off = 0;                        // 每条记录从 0 开始累计
        // ids：原样照抄（数量 + 数据），顺便收集出来算摘要
        if (cur + 8 > payload.size()) return false;
        const std::uint64_t n_ids = rd_u64(cur);
        if (cur + 8 + static_cast<std::size_t>(n_ids) * 4 > payload.size()) return false;
        ck_ids.assign(n_ids, 0);
        if (n_ids) std::memcpy(ck_ids.data(), base + cur + 8, static_cast<std::size_t>(n_ids) * 4);
        if (!emit_payload(cur + 8 + static_cast<std::size_t>(n_ids) * 4)) return false;
        // imgs：原样照抄
        if (cur + 8 > payload.size()) return false;
        const std::uint64_t n_img = rd_u64(cur);
        if (cur + 8 + static_cast<std::size_t>(n_img) * 16 > payload.size()) return false;
        if (!emit_payload(cur + 8 + static_cast<std::size_t>(n_img) * 16)) return false;

        const PrefixDigest d = prefix_digest(ck_ids, ck_hashes, ck_starts,
                                             static_cast<std::int64_t>(ck_ids.size()));
        // live 用哨兵键 {0,0}；检查点用它的 ids 前缀摘要
        const std::uint64_t key_lo = (ck_index == 0) ? 0u : d.lo;
        const std::uint64_t key_hi = (ck_index == 0) ? 0u : d.hi;
        const DiskParkCkptDirEntry* e = dir_find(key_lo, key_hi);

        for (std::size_t p2 = 0; p2 < 5; ++p2) {
            if (cur + 8 > payload.size()) return false;
            const std::uint64_t len = rd_u64(cur);
            if (cur + 8 + static_cast<std::size_t>(len) > payload.size()) return false;
            if (!emit_payload(cur + 8)) return false;              // 吐长度字段
            const std::size_t data_at = cur;
            // ★★ 语义（方案 C 定稿）：
            //   目录项 present()（len != 0）⇒ 这个状态的载荷里是【全 0 占位】、
            //   真值在日志里 ⇒ 跳过占位、从日志取（需要调用方给 fill_ckpt）。
            //   否则（没有目录项 / len == 0）⇒ 载荷里就是真值，照抄。
            //   ★ 与 KV 无关：KV【永远】留在载荷里（见 splice_image 的注释）。
            const bool from_log = (e != nullptr) && static_cast<bool>(fill_ckpt);
            if (!from_log) {
                if (!emit_payload(data_at + static_cast<std::size_t>(len))) return false;
                continue;
            }
            // ★ 载荷里这一段是【真值长度的全 0 占位】。产出里要放日志的真值
            //   ⇒ 跳过占位（不吐出去），但要【在载荷坐标上走过它】——
            //   否则后面的缓冲区会错位。
            cur = data_at + static_cast<std::size_t>(len);
            std::uint64_t done = 0;
            while (done < len) {
                const std::size_t take = static_cast<std::size_t>(
                    std::min<std::uint64_t>(buf.size(), len - done));
                // ★ 偏移用【累计值 + done】：累计值在取完这一段之后才前进。
                //   第一版先 += len 再取，于是第一个缓冲区的偏移就是 512（读到了错误位置）。
                if (!fill_ckpt(ck_index, p2, ck_rec_off + static_cast<std::size_t>(done),
                               buf.data(), take)) return false;
                if (!emit(buf.data(), take)) return false;
                done += take;
            }
            ck_rec_off += static_cast<std::size_t>(len);            // 这一段取完，累计前进
        }
        // used + stage_parts 计数 + 子检查点：原样照抄（本机 stage_parts 恒为空）
        if (cur + 8 > payload.size()) return false;
        const std::uint64_t used = rd_u64(cur);
        (void) used;
        if (!emit_payload(cur + 8)) return false;
        if (cur + 8 > payload.size()) return false;
        const std::uint64_t n_parts = rd_u64(cur);
        if (!emit_payload(cur + 8)) return false;
        const std::size_t saved = ck_index;
        for (std::uint64_t k2 = 0; k2 < n_parts; ++k2) {
            ck_index = saved + 1 + static_cast<std::size_t>(k2);    // 子检查点按顺序编号
            if (!walk_ckpt()) return false;
        }
        ck_index = saved;
        return true;
    };

    ck_index = 0;
    if (!walk_ckpt()) return 0;                                     // live
    if (cur + 8 > payload.size()) return 0;
    {
        const std::uint64_t n_ck = rd_u64(cur);
        if (!emit_payload(cur + 8)) return 0;
        for (std::uint64_t k2 = 0; k2 < n_ck; ++k2) {
            ck_index = 1 + static_cast<std::size_t>(k2);
            if (!walk_ckpt()) return 0;
        }
    }
    ck_index = 0;

    for (std::size_t layer = 0; layer < layers; ++layer) {
        if (!emit_payload(cur + 4 + 6 * 8)) return 0;          // format + 几何
        for (std::size_t p = 0; p < 5; ++p) {
            // ---- ① 长度字段：原值照抄（真值，reader 靠它决定消费多少）----
            if (cur + 8 > payload.size()) return 0;
            const std::uint64_t extent = rd_u64(cur);
            if (cur + 8 + static_cast<std::size_t>(extent) > payload.size()) return 0;
            if (!emit_payload(cur + 8)) return 0;              // 吐长度字段
            const std::size_t data_at = cur;                    // 光标现在在【数据段起点】

            // ★★★ 方案 C（2026-10-10 定稿）：**KV 永远留在载荷里，不进日志。**
            //
            //   为什么——日志与镜像的 KV 数据模型不一致：
            //     · 镜像：KV 是【每层一份】（generate.cpp: qsa_alloc 个 ConversationKv）
            //     · kv.bin：【按位置】存的，层数不参与寻址（part_offset(part, cell)）
            //   拼接若向日志索要"层数 × step"，日志只有一份 ⇒ 长度对不上、
            //   unpack 失败（实测：fill 被要求 9000 字节，日志只有 3000）。
            //
            //   要修就得二选一，代价都大：
            //     A) 日志按层分区  ⇒ 存储 × 层数（本机 12–13 层）
            //     B) 镜像只存一份  ⇒ 要改上游 conversation_kv_save/restore
            //   ⇒ 选 C：KV 整份进载荷。收益几乎不损失 —— KV 本来就必须在载荷里，
            //     真正省下来的是【检查点 GDN】（占链 91.7%），而那部分没有"层"的问题。
            //
            //   于是这个分支只做一件事：照抄载荷（含占位 0）。
            if (!emit_payload(data_at + static_cast<std::size_t>(extent))) return 0;
            (void) p;
        }
    }
    if (!emit_payload(body)) return 0;
    return total;
}

bool DiskParkSession::read_kv(Part part, std::uint64_t from_cell, std::uint64_t to_cell,
                              std::uint8_t* dst, std::size_t bytes) const {
    if (!ok() || !dst || to_cell < from_cell) return false;
    const auto& p = plan_.parts[static_cast<std::size_t>(part)];
    if (p.bytes_per_cell == 0) return false;
    const std::uint64_t want = part_bytes(part, to_cell) - part_bytes(part, from_cell);
    if (bytes < want) return false;
    if (want == 0) return true;
    return pread_all(fd_, dst, static_cast<std::size_t>(want), part_offset(part, from_cell));
}

} // namespace strata::core
