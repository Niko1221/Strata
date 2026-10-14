#pragma once

// 停放层【流式日志的持有者】：把 DiskParkSession（一对文件 = 一个会话）按会话索引起来，
// 供引擎在「每个请求结束后追加本轮增量」那条路径上使用（设计文档 §5.1 / §7 阶段 2）。
//
// ★★ 为什么需要一个「摘要 → 文件」的映射，而不是直接拿摘要当目录名：
//    摘要描述的是【一段前缀】，而对话每轮都在变长 —— 这一轮的摘要是 D，
//    下一轮的"上一轮"也是 D。所以：写完一轮 D→D' 之后，把 D' 也记到【同一个文件】上，
//    下一轮就能顺着 D' 找到它续写。不记的话每轮都会开一个新文件，
//    盘上堆出 N 份互相是前缀的快照 —— 那正是这套改造要消灭的东西。
//
// ★ 本层只做三件事，都不碰 CUDA：
//     ① 由调用方给的【每 part 字节数】算出 DiskParkSessionPlan
//     ② 维护「摘要 → 会话目录 + 水位」这张表（有上限）
//     ③ 把「上一轮水位 → 本轮 token 数」这段增量交给 session 去追加
//
// ★★ 本层【不做】：不写 sidecar 载荷（调用方手里才有 SavedConversation）、
//    不判"该不该写"（阈值/开关在调用方）、不碰引擎状态。

#include "strata/core/disk_park_session.hpp"

#include <cstddef>
#include <cstdint>
#include <functional>
#include <cstdio>
#include <list>
#include <map>
#include <string>
#include <vector>

namespace strata::core {

// 前缀摘要（与 DiskParkIdentity 的 lo/hi 同一套算法，见 disk_park_digest.hpp）。
struct DiskParkLogDigest {
    std::uint64_t lo = 0, hi = 0;
    friend constexpr bool operator==(const DiskParkLogDigest&, const DiskParkLogDigest&) noexcept = default;
    friend constexpr bool operator<(const DiskParkLogDigest& a, const DiskParkLogDigest& b) noexcept {
        return a.lo != b.lo ? a.lo < b.lo : a.hi < b.hi;
    }
};

// ★★★ 2026-10-13（取回侧性能修复）：会话目录 → 头部指纹索引。
//   为什么需要：取回路径的主路径要为【每个】会话目录读 tokens.bin 算 LCP、读 12 个
//   pooled.bin、读 ckpt_lengths.bin（约 15 次文件操作 × 目录数）。隔离实测：
//     停靠层关闭              冷读 85.1K = 34.6 / 34.9 / 35.0 s
//     停靠层开着，目录空       冷读 85.1K = 34.6 / 35.5 / 36.2 s
//     停靠层开着，1,508 会话   冷读 85.1K = 49.3 / 49.6 / 50.2 / 50.8 / 51.1 s
//   有了这个索引：前 1024 个 token 的指纹不一致就直接跳过 —— 【零文件 I/O】。
//   ★ 指纹只是【预筛】：通过之后仍然走原来的完整校验（LCP / pooled / ckpt_lengths），
//     所以索引再旧、再缺项，也绝不会让错误的 K/V 被装进请求（fail-closed 不变）。
struct DirHead {
    std::uint64_t cells   = 0;   // 该会话写到多少 token（state.bin 里的 log_cells）
    std::uint64_t head_n  = 0;   // 指纹覆盖多少个 token（= min(1024, cells)）
    std::uint64_t fp_head = 0;   // FNV-1a over the first head_n token ids
};

// 前 head_n 个 token id 的指纹（写侧存、读侧比，必须是同一个函数）。
[[nodiscard]] inline std::uint64_t disk_park_head_fingerprint(const std::int32_t* ids,
                                                              std::size_t n) noexcept {
    std::uint64_t h = 1469598103934665603ull;              // FNV-1a 64 offset basis
    for (std::size_t i = 0; i < n; ++i) {
        const std::uint32_t v = (std::uint32_t) ids[i];
        h ^= (std::uint64_t) (v & 0xFFu);        h *= 1099511628211ull;
        h ^= (std::uint64_t) ((v >> 8) & 0xFFu); h *= 1099511628211ull;
        h ^= (std::uint64_t) ((v >> 16) & 0xFFu);h *= 1099511628211ull;
        h ^= (std::uint64_t) ((v >> 24) & 0xFFu);h *= 1099511628211ull;
    }
    return h;
}

// 指纹覆盖的长度上限（也是请求侧要预计算的长度）。
static constexpr std::size_t kDirHeadTokens = 1024;

// 前缀指纹表：out[k] = 前 (k+1) 个 token 的指纹。请求侧算一次，之后每个目录
// 只做一次比较（O(1)），不必为每个目录各自重算一遍前缀。
[[nodiscard]] inline std::vector<std::uint64_t>
disk_park_head_prefixes(const std::int32_t* ids, std::size_t n) {
    std::vector<std::uint64_t> out(n);
    std::uint64_t h = 1469598103934665603ull;
    for (std::size_t i = 0; i < n; ++i) {
        const std::uint32_t v = (std::uint32_t) ids[i];
        h ^= (std::uint64_t) (v & 0xFFu);         h *= 1099511628211ull;
        h ^= (std::uint64_t) ((v >> 8) & 0xFFu);  h *= 1099511628211ull;
        h ^= (std::uint64_t) ((v >> 16) & 0xFFu); h *= 1099511628211ull;
        h ^= (std::uint64_t) ((v >> 24) & 0xFFu); h *= 1099511628211ull;
        out[i] = h;
    }
    return out;
}

class DiskParkLogs {
public:
    struct Config {
        std::string root;                  // 空 = 整个日志关闭（行为与没有它完全一致）
        std::uint32_t tag = 0;             // DiskParkProfile::tag()
        // 与 ConversationKv 同名的元数据（进 kv.bin 头，读回时逐字段校验）
        std::int32_t kv_format = 0;
        std::int64_t heads = 0, head_dim = 0, page_size = 0, idx_dim = 0;
        std::int64_t idx_block = 0;        // 一行 pooled 覆盖多少 cell
        std::uint64_t capacity_cells = 0;  // 每个 part 预留多少 cell（通常 = max_context）
        // 每 part 在 capacity_cells 处的字节数，顺序 k, v, k_scale, v_scale, pooled。
        // pooled 的 row_bytes 由 row_cells = idx_block 与它在这里的字节数反推。
        std::uint64_t part_bytes_at_cap[5] = {0, 0, 0, 0, 0};
        std::size_t max_open = 16;         // 同时打开几个会话文件
        // ★★ P25（2026-10-13）：盘上【总占用】上限（字节，0 = 不限）。超了按 LRU 把最久未用的
        //   会话目录【连文件一起删】。这是 08a62f46 删掉的那段，现在按方案 C 恢复：
        //   引擎知道哪个会话正在用，按整会话淘汰比 park-gc 按时间删更准；park-gc 退成
        //   老目录清扫 + 高位兜底。口径必须是 st_blocks×512（见 .cpp 启动扫描处的注释）。
        std::uint64_t max_bytes = 0;
        std::size_t max_tracked = 4096;    // 跟踪多少个「摘要 → 会话」
    };

    struct Stats {
        std::uint64_t writes = 0;          // 成功追加的轮次
        std::uint64_t write_bytes = 0;     // 累计追加字节
        std::uint64_t adopted = 0;         // 顺着上一轮摘要找到已有会话的次数
        std::uint64_t fresh = 0;           // 开了新会话的次数
        std::uint64_t opens = 0, closes = 0;
        std::uint64_t failures = 0;
        std::uint64_t evicted_dirs = 0;    // ★ P25：按容量淘汰掉的会话目录数
        std::uint64_t evicted_bytes = 0;
        std::uint64_t reaped_dirs = 0;     // ★ P26：本次运行建出来、没写过数据的空壳目录
    };

    // 一次追加的输入
    struct Request {
        DiskParkLogDigest prev;            // 上一轮写完时的摘要（= 本轮要扩展的那段前缀）
        DiskParkLogDigest next;            // 本轮写完之后整段的摘要
        bool has_prev = false;             // false = 这是新会话的第一轮
        // ★★★ P14（2026-10-10）：取回路径直传。本轮如果是从某个会话目录【取回】的，
        //   调用方知道"哪个目录、恢复到第几个 token"，直接用它当 (目录, 水位)。
        //   为什么必须直传：摘要链查的是 prefix_digest(…, read_from)，而 read_from 是
        //   【挂载点/检查点长度】，永远小于上一轮 remember() 登记的 L ⇒ lookup() 恒 miss
        //   ⇒ 每轮新建目录、have=0、整份 KV 重写（实测 76 轮 26.5 GiB，50× 放大，
        //   盘上堆出 173 个互为前缀的目录）。
        std::string restore_dir;           // 空 = 没走日志取回 → 退回摘要链
        std::uint64_t restore_tokens = 0;  // 该目录的日志已被取回到的 token 数（= got）
        std::uint64_t to_tokens = 0;       // 本轮的 token 数
        // ★★★ P15（2026-10-11）：别名 frontier。摘要链只登记"本轮结束时的 L"，
        //   而下一轮的复用点是【检查点长度】（≠ L）⇒ lookup() 恒 miss ⇒ 新建目录 +
        //   整份 KV 重写。实测 37 轮里只有 9 轮命中，20 轮整份重写共 7.22 GiB。
        //   把本轮【所有可能被下一轮复用的 frontier】都登记到同一个目录。
        struct Alias {
            DiskParkLogDigest digest{};
            std::uint64_t tokens = 0;
            Alias() = default;
            Alias(const DiskParkLogDigest& d, std::uint64_t t) : digest(d), tokens(t) {}
        };
        std::vector<Alias> aliases;
        // ★ 出参：这次写进了哪个会话目录。调用方拿它去 commit 镜像元数据
        //   （水位与 sidecar 必须是同一个文件，不能各算各的）。
        std::string* out_dir = nullptr;
        // ★★★ 2026-10-13：本轮的头部指纹（调用方算一次）—— 写进 dir_head_ 索引，
        //   供取回路径零 I/O 预筛。head_n = min(kDirHeadTokens, ids.size())。
        std::uint64_t fp_head = 0;
        std::uint64_t head_n  = 0;
    };

    explicit DiskParkLogs(Config cfg);
    ~DiskParkLogs();
    DiskParkLogs(const DiskParkLogs&) = delete;
    DiskParkLogs& operator=(const DiskParkLogs&) = delete;

    [[nodiscard]] bool enabled() const noexcept { return enabled_; }

    // ★★★ 2026-10-13：会话目录 → 头部指纹。懒加载（从 sidecar 读；与盘上目录列表
    //   对账，缺的补、多的丢），之后常驻内存。取回路径用它做零 I/O 预筛。
    //   ★ 返回的是【目录名 → DirHead】；目录名就是 digest_hex。
    [[nodiscard]] const std::map<std::string, DirHead>& dir_heads();
    [[nodiscard]] const DiskParkSessionPlan& plan() const noexcept { return plan_; }
    [[nodiscard]] Stats stats() const { return stats_; }
    [[nodiscard]] std::string dir_for(const std::string& dir_name) const;
    // ★ 逐层：会话目录 / layerNN
    [[nodiscard]] std::string dir_for_layer(const std::string& dir_name, std::size_t layer) const {
        char b[16];
        std::snprintf(b, sizeof b, "layer%02zu", layer);
        return dir_for(dir_name) + "/" + b;
    }
    // ★ 摘要 → 十六进制串。
    //   ★★ 它【不是】"查这个摘要属于哪个会话" —— 会话目录名 = 该会话【第一次建立时】
    //   用的那个 hex（= 当时 next 的 hex），之后永不改变（见 remember）。
    //   ⇒ 要拿"某个摘要所在的会话目录"，必须用 append 的 Request::out_dir 出参，
    //     或者用 lookup()（查内存表）。拿任意一个摘要去 name_for 得到的目录可能不存在。
    [[nodiscard]] static std::string name_for(const DiskParkLogDigest& d);
    // ★ P12：由【整段摘要】直接算出会话目录（不查表）。
    //   取回路径需要它：恢复端只知道 prompt 的摘要，而表是内存里的（重启后为空）——
    //   目录名本身就是摘要的 hex，所以直接拼得出来。
    [[nodiscard]] std::string session_dir_for(const DiskParkLogDigest& d) const {
        char buf[40];
        std::snprintf(buf, sizeof buf, "%016llx%016llx",
                      (unsigned long long) d.lo, (unsigned long long) d.hi);
        return dir_for(std::string(buf));
    }
    // 该前缀摘要对应的会话目录名；没有就返回空串
    [[nodiscard]] std::string lookup(const DiskParkLogDigest& d) const;
    // 该前缀摘要已经写到多少 token（0 = 没记录）
    [[nodiscard]] std::uint64_t watermark(const DiskParkLogDigest& d) const;
    // ★★ P18（2026-10-11）：当前打开着的会话目录数 / 文件数。
    //   逐层模式下 open_ 的键是 "目录#层号"，一个目录 12 条 ⇒ 两个数差 ~12 倍。
    //   生产上用 open_files() 估 fd 占用（12 层 × 目录数 + 常数）；测试用它断言淘汰。
    [[nodiscard]] std::size_t open_dirs() const noexcept { return open_dir_stamp_.size(); }
    [[nodiscard]] std::size_t open_files() const noexcept { return open_.size(); }

    // 只把水位提交下去（sidecar 的载荷为空）。阶段 2 的追加路径用它。
    [[nodiscard]] bool commit_watermark(const DiskParkLogDigest& d, std::uint64_t tokens);

    // 提交【镜像元数据 + 水位】。dir_name 来自 Request::out_dir。
    //   meta_packed = pack_saved_conversation() 的输出，主层 K/V 的缓冲区已清空；
    //   kv_at       = 镜像里第一个 KV 层元数据的偏移（拼回来时要用）；
    //   kv_layers   = 有几层 K/V（镜像里自带，这里只是给读回时做一致性校验的机会）。
    [[nodiscard]] bool commit_image(const std::string& dir_name, std::uint64_t tokens,
                                    std::span<const std::byte> meta_packed, std::uint64_t kv_at);

    // 追加本轮增量。【只写 kv.bin，不提交 sidecar】—— 提交由调用方做。
    // 命中 prev ⇒ 续写那个会话；否则开新会话（从 0 写）。
    // 返回实际追加的字节数；0 表示没写或失败。
    //
    // ★ 单层语义：整份 K/V 写在会话目录下的一个 kv.bin 里。
    //   引擎的真实 K/V 是【每层一份】（layer.hpp: 每个 QsaState 有自己的 host 池），
    //   那种情况请用下面的 append_layers —— 层数不同，数据组织就不同。
    [[nodiscard]] std::uint64_t append(const Request& req, const DiskParkFill& fill);

    // ★★★ 逐层追加（引擎用这条）。每层在会话目录下有自己的子目录 layerNN/，
    //   因为"位置寻址 + 只追加"对【每一层各自】成立：
    //   第 pos 个 token 在第 k 层的字节，落在只由 (k, pos) 决定的偏移上。
    //
    //   fill(layer, part, offset, dst, n)：offset 是该层该 part 段内的字节偏移。
    //   返回所有层实际追加的字节数之和；0 表示没写或失败。
    using DiskParkLayerFill = std::function<bool(std::size_t layer, std::size_t part,
                                                 std::size_t offset, std::uint8_t* dst, std::size_t n)>;

    [[nodiscard]] std::uint64_t append_layers(const Request& req, std::size_t layers,
                                              const DiskParkLayerFill& fill);

    // ★★★ 逐层取回：把某一层的 [from_cell, to_cell) 读出来。
    //   dir_name 来自 append_layers 的 Request::out_dir（或 name_for(摘要)）。
    //   read(layer, part, offset, src, n)：offset 是该层该 part 段内的字节偏移。
    using DiskParkLayerRead = std::function<bool(std::size_t layer, std::size_t part,
                                                 std::size_t offset, const std::uint8_t* src,
                                                 std::size_t n)>;

    [[nodiscard]] bool read_layers(const std::string& dir_name, std::size_t layers,
                                   std::uint64_t from_cell, std::uint64_t to_cell,
                                   const DiskParkLayerRead& read);

    // ★★★ 同上，但【并发】读各层（2026-10-10）。各层写自己的 host 池、pread 不带文件偏移，
    //   所以可以并行；`read` 会被多个线程同时调用（每个 (layer, part, offset) 只调一次）。
    //   threads <= 1 时退回串行。先用一次 session_for_layer 把全部层开好（它会改 open_ 这个
    //   map，不能并发），再开线程读。
    [[nodiscard]] bool read_layers_parallel(const std::string& dir_name, std::size_t layers,
                                            std::uint64_t from_cell, std::uint64_t to_cell,
                                            const DiskParkLayerRead& read, int threads);

private:
    // 读【一层】的 [from_cell, to_cell)：纯函数式，不碰 open_（并发安全的那一半）。
    bool read_layer_cells(DiskParkSession& sess, std::size_t layer, std::uint64_t from_cell,
                          std::uint64_t to_cell, const DiskParkLayerRead& read);

    struct Track {
        std::string dir_name;
        std::uint64_t watermark = 0;
        // ★ 在"最近用过"链上的位置。淘汰必须按【最近用过】而不是按【水位最小】：
        //   一条对话链是 D1→D2→D3…，水位最小的 D1 恰恰【仍然会被用到】
        //   （客户端基于更早一轮重发时）。按水位淘汰会把活跃链的头掐掉，
        //   下一轮就找不到文件、只好重写一份 —— 实测就是这么暴露的。
        std::list<DiskParkLogDigest>::iterator lru;
    };
    DiskParkSession* session_for(const std::string& dir_name, std::uint64_t resume,
                                 bool* opened_now);
    // ★ 逐层的会话：键为 "目录名#层号"，路径为 会话目录/layerNN
    DiskParkSession* session_for_layer(const std::string& dir_name, std::size_t layer,
                                       std::uint64_t resume, bool* opened_now, bool layered);
    void remember(const DiskParkLogDigest& d, const std::string& dir_name, std::uint64_t watermark);
    void touch(const DiskParkLogDigest& d);
    void trim(const std::string& active = std::string());   // ★ P25：active = 正在追加的那份，绝不删
    // ★★ 2026-10-13：头部指纹索引的加载/落盘/对账（见 dir_heads 的说明）。
    void load_dir_heads();
    void save_dir_heads() const;
    void note_dir_head(const std::string& dir_name, std::uint64_t cells,
                       std::uint64_t head_n, std::uint64_t fp_head);
    // ★★ P18：逐层路径下 open_ 的键是 "目录#层号"，一个会话目录占 12 条（12+ 个 fd），
    //   而 max_open 的淘汰循环只写在单层的 append() 里 ⇒ 逐层路径无上限增长。
    //   这里按【目录】为单位淘汰：把最久没用过的那个目录的 12 层一起关掉。
    void touch_open_dir(const std::string& dir_name);
    void evict_open_dirs();

    Config cfg_;
    DiskParkSessionPlan plan_{};
    bool enabled_ = false;
    std::map<DiskParkLogDigest, Track> tracked_;
    std::list<DiskParkLogDigest> lru_;              // 最近用过的在前
    std::map<std::string, DiskParkSession> open_;   // 键：单层=目录名；逐层="目录名#层号"
    std::map<std::string, std::uint64_t> open_dir_stamp_;   // P18：目录 → 最近使用序号
    std::uint64_t open_stamp_ = 0;                          // P18：序号发生器
    mutable Stats stats_{};
    // ★★ 2026-10-13：会话目录 → 已写字节；以及所有目录之和。
    //   只用于启动时那行诊断（[启动] 盘上已有 N 个会话目录，共 X GiB）。
    //   ★ 引擎【不做】容量上限：流式层的字节预算归 tools/park-gc（--max-gib），
    //   那里能同时看见会话目录和 checkpoints.bin，并用 st_blocks 而不是表观大小记账。
    //   按【目录】而不是按【摘要】记账：一个目录会被多个摘要（别名 frontier）指向。
    std::map<std::string, std::uint64_t> dir_bytes_;
    std::uint64_t total_bytes_ = 0;
    // ★★ 2026-10-13：头部指纹索引（见 dir_heads）。loaded 之前为空。
    std::map<std::string, DirHead> dir_heads_;
    bool dir_heads_loaded_ = false;
    bool dir_heads_dirty_  = false;
};

} // namespace strata::core
