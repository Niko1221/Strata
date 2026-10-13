#pragma once
// ★★★ P13：indexer 的"池化行"（pooled）落盘 —— 长上下文正确性的最后一环。
//
// 背景：QSA 层的稀疏选择靠 `st.idx_pooled`（每 `idx_block` 个 cell 合成一行 key，
// 每行 `idx_dim` 个 float）。它**不在 host 池里**（是设备指针），所以逐层追加日志
// 一开始把它跳过了（`part_bytes_at_cap[4] = 0`），代价约 13.6% 的 KV 字节。
// 45k 上下文实测"输出与冷读逐字相同"，但更长的上下文没验证过 —— 这就是这一环。
//
// 关键事实（`include/strata/core/layer.hpp:243-248`）：
//   * `idx_tail`    = (idx_block-1, idx_dim) —— 正在填的那个块的**尾巴**，**在检查点里**；
//   * `idx_dead`    = (idx_dim,) —— 备用槽的 key，**对整段序列是常量**，也在检查点里；
//   * `idx_pooled`  = (max_cells/idx_block + 2, idx_dim) —— 完成块的行 + 一个备用槽。
//   ⇒ **完成了的池化行是不可变的历史**（行 r 只由 cell [4r,4r+4) 决定），
//     所以它可以像逐层 KV 一样【只追加增量】；备用槽那一行由引擎在下一个块完成时覆盖，
//     它的内容是常量（`idx_dead`），恢复时从检查点补上即可。
//
// 文件布局（小端）：`[magic 8B "P13POOL1"][row_bytes u64][row 0][row 1]...`
//   ⇒ 已有的完整行数 = (文件大小 − 16) / row_bytes，**自描述、无需更新计数**。
//   追加 = pwrite 到 `16 + from_row*row_bytes`（先把可能被写坏的尾巴 ftruncate 掉）。
//
// 失败语义：缺失/损坏/magic 不符/row_bytes 不符 ⇒ 返回 false 且 rows=0，
// 调用方必须当成"池化行不齐 ⇒ 不许取回"（退化成冷读），绝不能拿半份用。

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>

#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

namespace strata::core {

namespace p13_pooled_detail {

inline constexpr char kMagic[8] = {'P', '1', '3', 'P', 'O', 'O', 'L'};
inline constexpr std::uint64_t kHeader = 16;
inline constexpr std::uint64_t kMaxRows = (std::uint64_t) 1 << 32;

// 小工具：整段 pwrite / pread
inline bool write_at(int fd, const std::uint8_t* p, std::size_t n, std::uint64_t off) {
    std::size_t done = 0;
    while (done < n) {
        const ssize_t w = ::pwrite(fd, p + done, n - done, (off_t) (off + done));
        if (w <= 0) return false;
        done += (std::size_t) w;
    }
    return true;
}
inline bool read_at(int fd, std::uint8_t* p, std::size_t n, std::uint64_t off) {
    std::size_t done = 0;
    while (done < n) {
        const ssize_t r = ::pread(fd, p + done, n - done, (off_t) (off + done));
        if (r <= 0) return false;
        done += (std::size_t) r;
    }
    return true;
}

} // namespace p13_pooled_detail

// 文件里已有的【完整行】数。false = 文件缺失/头损坏/row_bytes 不符（rows 置 0）。
inline bool p13_pooled_rows(const std::string& path, std::uint64_t row_bytes, std::uint64_t& rows,
                            std::string& error) {
    using namespace p13_pooled_detail;
    rows = 0;
    if (row_bytes == 0) { error = "p13 pooled: zero row size"; return false; }
    const int fd = ::open(path.c_str(), O_RDONLY);
    if (fd < 0) { error = "p13 pooled: absent"; return false; }
    struct stat st {};
    std::uint8_t h[kHeader];
    bool ok = ::fstat(fd, &st) == 0 && st.st_size >= (off_t) kHeader &&
              read_at(fd, h, sizeof h, 0) && std::memcmp(h, kMagic, 8) == 0;
    std::uint64_t stored = 0;
    if (ok) {
        std::memcpy(&stored, h + 8, 8);
        ok = stored == row_bytes;
    }
    ::close(fd);
    if (!ok) { error = "p13 pooled: bad header or row size"; return false; }
    rows = ((std::uint64_t) st.st_size - kHeader) / row_bytes;
    if (rows > kMaxRows) { rows = 0; error = "p13 pooled: absurd row count"; return false; }
    return true;
}

// 追加行 [from_row, to_row)：`rows` 里是这些行（(to_row-from_row)*row_bytes 字节）。
// 会先把文件里可能被写坏的尾巴截到 from_row 行（半行/多余的旧行都被清掉）。
inline bool p13_pooled_append(const std::string& path, std::uint64_t row_bytes,
                              const std::uint8_t* rows_buf, std::uint64_t from_row, std::uint64_t to_row,
                              std::string& error) {
    using namespace p13_pooled_detail;
    if (row_bytes == 0 || to_row < from_row) { error = "p13 pooled: bad append range"; return false; }
    if (to_row == from_row) return true;
    const int fd = ::open(path.c_str(), O_RDWR | O_CREAT, 0644);
    if (fd < 0) { error = "p13 pooled: cannot open " + path; return false; }
    struct stat st {};
    std::uint8_t h[kHeader];
    bool ok = ::fstat(fd, &st) == 0;
    if (ok && st.st_size == 0) {                       // 新文件：写头
        std::memcpy(h, kMagic, 8);
        std::memcpy(h + 8, &row_bytes, 8);
        ok = write_at(fd, h, sizeof h, 0);
    } else if (ok) {
        ok = st.st_size >= (off_t) kHeader && read_at(fd, h, sizeof h, 0) &&
             std::memcmp(h, kMagic, 8) == 0;
        std::uint64_t stored = 0;
        if (ok) { std::memcpy(&stored, h + 8, 8); ok = stored == row_bytes; }
    }
    if (ok) ok = ::ftruncate(fd, (off_t) (kHeader + from_row * row_bytes)) == 0;
    if (ok) ok = write_at(fd, rows_buf, (std::size_t) ((to_row - from_row) * row_bytes),
                          kHeader + from_row * row_bytes);
    ::close(fd);
    if (!ok) { error = "p13 pooled: append failed"; return false; }
    return true;
}

// 把行 [0, rows) 读进 dst（rows*row_bytes 字节）
inline bool p13_pooled_read(const std::string& path, std::uint64_t row_bytes, std::uint64_t rows,
                            std::uint8_t* dst, std::string& error) {
    using namespace p13_pooled_detail;
    if (rows == 0) return true;
    std::uint64_t have = 0;
    if (!p13_pooled_rows(path, row_bytes, have, error)) return false;
    if (have < rows) { error = "p13 pooled: fewer rows on disk than needed"; return false; }
    const int fd = ::open(path.c_str(), O_RDONLY);
    if (fd < 0) { error = "p13 pooled: absent"; return false; }
    const bool ok = read_at(fd, dst, (std::size_t) (rows * row_bytes), kHeader);
    ::close(fd);
    if (!ok) { error = "p13 pooled: short read"; return false; }
    return true;
}

} // namespace strata::core
