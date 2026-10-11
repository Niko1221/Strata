#pragma once

// 停放层的【流式会话文件】：一个会话 = 一对文件（kv.bin + state.bin）。
//
// 设计见 docs/停放层流式改造设计.md §4。它替换的是「每次淘汰整块重写一份快照」
// 那套（DiskParkStore/Spool 的固定槽 + 8 MiB 定长分页），改成：
//
//   kv.bin     追加流。头部 64 B 自描述，其后是 5 个【连续区】：
//              K | V | Kscale | Vscale | pooled，各自按 cell 线性寻址。
//              第 pos 个 token 的字节落在【只由 pos 决定】的偏移上 ——
//              这是它能被追加的全部理由（内核侧写入点见设计文档 §1.1 性质 1）。
//   state.bin  元数据 sidecar，【整份原子替换】（临时文件 + fsync + rename + 目录 fsync）。
//
// ★★ sidecar 的载荷就是标准的 pack_saved_conversation 镜像，只是 kv[] 的 5 个缓冲区
//   长度【全为 0】—— 真正的 K/V 字节在 kv.bin 里。读回时把 state.bin 的载荷与
//   kv.bin 的字节按 part 交错【拼接】成一条完整的镜像流，再交给上游
//   unpack_saved_conversation_from()。于是上游的序列化/反序列化器一行都不用改。
//
// ★★ 提交点只有一个：state.bin 的 rename。水位 log_cells 写在 sidecar 里，
//   所以 kv.bin 尾部多出来的字节永远不参与 —— 半写、崩溃、被截断都只表现为
//   「退回到上一次提交」，而不会读到脏数据。
//
// ★ 本文件【不碰 CUDA、不碰引擎状态】：追加的源字节由调用方以回调给出，
//   于是它能在没有 GPU 的独立测试里完整验证。
//
// 阶段：本文件是 §7 阶段 1（存储层）。容量淘汰（文件级 LRU）与检查点增量流在
// 阶段 4、由阶段 0 的实测数字决定是否本期做。

#include <cstddef>
#include <cstdint>
#include <functional>
#include <span>
#include <string>
#include <vector>

namespace strata::core {

struct PrefixDigest;   // disk_park_digest.hpp（这里只用引用，避免头文件互相包含）

// ★★ P12：把载荷里 live / 检查点的【摘要】按顺序取出来（不改动载荷、不解码数据）。
//   为什么需要：写侧把检查点状态按 prefix_digest(ids) 放进日志（内容寻址），
//   读侧必须先知道"第 i 个检查点的摘要是什么"，才能去日志里取回它的状态。
//   摘要只由 ids 决定，所以走一遍载荷的 ids 字段就够了。
//
//   输出：out[0] = live 的摘要；out[i+1] = checkpoints[i] 的摘要。
//   返回 false = 载荷结构不符（此时调用方应整体放弃，不要猜）。
bool extract_ckpt_digests(std::span<const std::byte> payload, std::size_t expected_layers,
                          std::vector<PrefixDigest>& out);

// 一个 part 的线性几何：每 cell 多少字节、按什么粒度成行。
//   bytes_per_cell 为 0 表示这个 part 在本 profile 下不存在（例如纯 q4 时没有 scale）。
// ★★ P12：侧车载荷【尾部追加的目录】—— 记录"哪些状态（检查点 / live）被搬进了日志"。
//
// 为什么不复用检查点的 used 字段：`used` 是上游 prompt-cache 的 LRU 时间戳
// （generate.cpp: `c.used = ++check_clock`，随后按它排序决定丢弃哪些检查点），
// 而且 conversation_snapshot_test 明确要求它"在快照里存活"。拿它放引用会把上游的
// 淘汰逻辑搞乱 —— 那是静默的错误行为，比崩溃更糟。
//
// 布局（小端）：
//   [镜像本体]                      ← 长度 = counted_saved(...)，unpack 只读到这里
//   [项 0] lo(8) hi(8) off(8) len(8)     ← 32 字节一项
//   [项 1] ...
//   [项数 n(8)]
//   [魔数 "SKD1" = 0x31444B53 (4)]       ← 从末尾往前 12 字节处找它
//
// 成本：12 + 32n 字节（12 个检查点 + 1 份 live ⇒ 428 字节）。
struct DiskParkCkptDirEntry {
    std::uint64_t lo = 0, hi = 0;     // 摘要：检查点 = prefix_digest(ids,...)；live = {0,0} 哨兵
    std::uint64_t off = 0, len = 0;   // 该状态在日志记录里的区间（len == 0 ⇒ 没搬进日志）
    [[nodiscard]] bool present() const noexcept { return len != 0; }
    [[nodiscard]] bool is_live() const noexcept { return lo == 0 && hi == 0; }
};
// live 的哨兵键（正常的 prefix_digest 不会给出 {0,0}）
inline constexpr DiskParkCkptDirEntry kLiveDirKey{};

inline constexpr std::uint32_t kCkptDirMagic = 0x31444B53U;   // "SKD1"

// 在载荷里找目录。false = 没有目录（老载荷 / 纯 KV 载荷）；此时 image_bytes = size。
bool parse_ckpt_dir(std::span<const std::byte> payload,
                    std::vector<DiskParkCkptDirEntry>& out, std::uint64_t& image_bytes);

// 把目录拼到镜像后面。
void append_ckpt_dir(std::vector<std::byte>& image,
                     const std::vector<DiskParkCkptDirEntry>& entries);

// 一个 part 的线性几何：每 cell 多少字节、按什么粒度成行。
//   bytes_per_cell 为 0 表示这个 part 在本 profile 下不存在（例如纯 q4 时没有 scale）。
struct DiskParkPartPlan {
    std::size_t bytes_per_cell = 0;   // 一个 cell（一个 token）占多少字节（所有 head 合计）
    std::size_t row_cells      = 1;   // 一行覆盖多少 cell（pooled 的 idx_block；其余为 1）
    std::size_t row_bytes      = 0;   // 一行的字节数（仅 pooled 用；其余等于 bytes_per_cell）
    // ★★ 这个 part 的字节【在追加日志里】还是【在 sidecar 载荷里】。
    //   为什么要区分：拼接时载荷里的长度字段是"上游按 layout 算出来的"，
    //   如果某个 part 的长度字段非 0、而我又从日志再吐一份，同一个 part 就被算了两次
    //   （实测：pooled 600 B/层 × 3 层 = 1800 B 的多余，正是"差 1800"的来源）。
    //   本机选择：K/V/Kscale/Vscale 走日志（大、且按位置只追加）；
    //   pooled 留在载荷里（小 —— 实测 171 MiB/13 层，且它有"移动的备用行"、不是纯追加）。
    bool in_log = true;
};

// 与 profile / 几何绑定的计划。★ 任一项不同 = 字节布局不同 = 必须各用一套文件。
struct DiskParkSessionPlan {
    std::uint32_t tag = 0;                       // DiskParkProfile::tag()
    DiskParkPartPlan parts[5];                   // 0=k 1=v 2=k_scale 3=v_scale 4=pooled
    // ★★ 每个 part 的【容量】（cell 数），必须 >= 本会话可能写到的最大 cell 数。
    //   段偏移 = 头 + 前面各段的【容量字节】，与"写到哪了"无关 ——
    //   若让偏移随水位变化，落在段中间的写入就会错位。这是本文件最容易错的地方。
    //   文件是稀疏的：容量大不等于占盘（未写的洞不分配）。
    std::uint64_t capacity_cells = 0;
    // 写进 kv.bin 头、读回时逐字段校验的 K/V 元数据（与 ConversationKv 同名字段一致）
    std::int32_t  kv_format = 0;
    std::int64_t  heads     = 0;
    std::int64_t  head_dim  = 0;
    std::int64_t  page_size = 0;
    std::int64_t  idx_dim   = 0;

    [[nodiscard]] bool valid() const noexcept;
};

// 读/写一段数据时的回调。
// ★ 用回调而不是「返回 vector」：真实 K/V 是几个 GiB，不能为了读它再要一份内存。
//   fn 返回 false 表示放弃（append_kv / read_kv 随之失败）。
//
// ★★ offset 是【相对该 part 段起点的字节偏移】—— 也就是"这个 part 里第几字节"，
//    与 from_cell 无关。追加与读取用同一个定义，于是
//    `fill` 只看 (part, offset) 就能算出这是第几个 cell（cell = offset / bytes_per_cell），
//    写进去的和读出来的必然一致。
using DiskParkFill = std::function<bool(std::size_t part, std::size_t offset, std::uint8_t* dst, std::size_t n)>;

class DiskParkSession {
public:
    enum class Part : std::size_t { K = 0, V = 1, KScale = 2, VScale = 3, Pooled = 4, Count = 5 };

    DiskParkSession() = default;
    ~DiskParkSession();
    DiskParkSession(const DiskParkSession&) = delete;
    DiskParkSession& operator=(const DiskParkSession&) = delete;
    DiskParkSession(DiskParkSession&&) noexcept;
    DiskParkSession& operator=(DiskParkSession&&) noexcept;

    // 打开（或新建）一个会话目录。返回 false 表示"这一层当作没有"：
    // 参数非法、I/O 失败、或已有文件与 plan 不符（magic/version/tag/几何 任一不符）。
    // ★ 不符时【不改动】任何已有文件 —— 由调用方决定是否 remove_all 后重来。
    // 调用方若在别处记着这个会话的水位（DiskParkLogs 就是），用 resume_cells 传进来：
    // 没有 sidecar 时它就是唯一的水位来源 —— 不给的话会话会当成空的从头写。
    [[nodiscard]] static bool open(DiskParkSession& out, const std::string& dir,
                                   const DiskParkSessionPlan& plan,
                                   std::uint64_t resume_cells = 0);

    [[nodiscard]] bool ok() const noexcept { return fd_ >= 0; }
    [[nodiscard]] std::uint64_t log_cells() const noexcept { return log_cells_; }
    [[nodiscard]] const std::string& dir() const noexcept { return dir_; }
    [[nodiscard]] std::string kv_path() const { return dir_ + "/kv.bin"; }
    [[nodiscard]] std::string state_path() const { return dir_ + "/state.bin"; }

    // ---- 写 ----
    // 把 [from_cell, to_cell) 的字节从 fill 追加进 kv.bin。
    //   from_cell 应取「按 part 粒度向下取整」的值（调用方算，见设计文档 §5.1 ①）。
    //   ★ 若 from_cell < log_cells()，先把该段截到 from_cell 再写（改写历史的场景）。
    // 只写 KV，不改元数据 —— 提交点是 commit()。
    // ★ 返回【实际写入的字节数】（= 各 part 增量之和，不含稀疏空洞）；0 表示没写或失败。
    [[nodiscard]] std::uint64_t append_kv(std::uint64_t from_cell, std::uint64_t to_cell,
                                          const DiskParkFill& fill);

    // 提交：把 sidecar 换成 meta_packed，并把水位推进到 to_cell。
    //   meta_packed 是 pack_saved_conversation_into() 的输出（kv[] 的 5 个缓冲区长度为 0）。
    //   ★ 唯一的提交点；失败时旧 sidecar 与旧水位原样保留。
    // kv_at = 镜像里第一个 KV 层元数据的偏移；不传 / 0 = 只落水位（阶段 2 的用法）。
    [[nodiscard]] bool commit(std::uint64_t to_cell, std::span<const std::byte> meta_packed,
                              std::uint64_t kv_at = 0);

    // ---- 读 ----
    // sidecar 里记的镜像结构。★ 只记一个数：载荷里【第一个 KV 层元数据】的偏移。
    //   为什么要记而不再算一次：算它要复刻一遍镜像的字节布局（几何、检查点、stage…），
    //   那是"第二个真相"；写的时候顺手记下来，读的时候就不必知道布局。
    //   剩下的（层数、每层各 part 的长度）从载荷里读 —— 载荷本来就是标准镜像格式，
    //   长度字段都在里面。kv_at 用 32 位足够：载荷是去掉 K/V 之后的元数据，量级是 MB。
    //   ★ kv_at == 0 表示"这份 sidecar 只有水位"（老格式或阶段 2 的用法）。
    [[nodiscard]] bool read_meta(std::vector<std::byte>& payload, std::uint64_t& log_cells,
                                 std::uint64_t* kv_at = nullptr) const;

    // 把某个 part 的 [from_cell, to_cell) 读进 dst（dst 长度必须 >= 请求字节数）。
    [[nodiscard]] bool read_kv(Part part, std::uint64_t from_cell, std::uint64_t to_cell,
                               std::uint8_t* dst, std::size_t bytes) const;

    // 供测试与诊断：某个 part 在 kv.bin 里的字节偏移 / 占多少字节
    [[nodiscard]] std::uint64_t part_offset(Part part, std::uint64_t cell) const noexcept;
    [[nodiscard]] std::uint64_t part_bytes(Part part, std::uint64_t cells) const noexcept;
    // 该 part 的字节是否在追加日志里（false = 在 sidecar 载荷里）
    [[nodiscard]] bool part_in_log(Part part) const noexcept {
        return plan_.parts[static_cast<std::size_t>(part)].in_log;
    }

    // ★★★ 取回：把 sidecar 载荷（= 标准镜像，但主层 K/V 的 5 个缓冲区长度全为 0）
    //   与 kv.bin 里的字节【按 part 交错】拼成一条完整的镜像流，交给 sink。
    //
    //   streaming 的：K/V 有 1.4 GiB 量级，不物化 —— 拼一段喂一段。
    //   kv_at 来自 read_meta() 的第三个出参（写的时候记下的），
    //   指向【KV 层数字段】；layers 由调用方给（它本来就知道）。
    //   ★ 载荷里长度为 0 的 part ⇒ 数据在 kv.bin（这是两侧唯一的约定）。
    //   返回总字节数；0 = 失败（载荷格式不符 / 读 kv.bin 失败 / fill 拒绝）。
    //   ★ layers 必须由调用方给：它本来就知道（镜像的 KV 层数）。
    //     早先的版本从 payload 的 kv_at-8 去读那个数 —— 那是把"层数字段的位置"
    //     又假设了一遍，kv_at 稍偏一点就静默少拼几层（实测只拼了 1/3 的数据）。
    //   ★★ [from_cell, cells) 必须由调用方给 —— 这是日志里【实际存着的那个区间】。
    //   不要试图从本对象推：读回时通常是新开的 session，它既不知道上一轮水位、
    //   也不知道日志里到底覆盖了哪一段。第一版从载荷里取 cells 当两端，
    //   于是 from_cell == cells ⇒ 每个 part 的增量都是 0 ⇒ 只吐元数据、一个 K/V 字节都不吐。
    //   ★ P12：检查点/live 的状态也可以搬进日志 —— 由载荷尾部的目录（parse_ckpt_dir）
    //   指出哪些搬了、以及它们在【载荷里】的占位区间。走到那些缓冲区时：
    //   跳过占位、改从日志取真值（由 fill_ckpt 提供）。
    //
    //   fill_ckpt(ckpt, part, offset, dst, n)：
    //     ckpt = 0 → live；1..n → checkpoints[ckpt-1]
    //     part = 0 gdn / 1 ple / 2 tails / 3 dead / 4 block_pos
    //   不传 = 全部照抄载荷里的占位（老行为）。
    using DiskParkCkptFill = std::function<bool(std::size_t ckpt, std::size_t part,
                                                std::size_t offset, std::uint8_t* dst, std::size_t n)>;

    //   binary_bytes = 镜像【本体】的长度（= parse_ckpt_dir 的第三个出参）。
    //   ★ 载荷可能在本体之后还追加了目录；产出只到本体为止（目录不进产出流）。
    //     传 0 = 整条载荷都是本体（没有目录）。
    [[nodiscard]] std::uint64_t splice_image(std::span<const std::byte> payload,
                                             std::uint64_t kv_at, std::size_t layers,
                                             std::uint64_t from_cell, std::uint64_t cells,
                                             const DiskParkFill& fill,
                                             const std::function<bool(const void*, std::size_t)>& sink,
                                             const DiskParkCkptFill& fill_ckpt = {},
                                             std::uint64_t binary_bytes = 0) const;

    static constexpr std::uint32_t kKvMagic     = 0x32564B53U;  // "SKV2" 小端
    static constexpr std::uint32_t kKvVersion   = 2U;
    static constexpr std::uint32_t kMetaMagic   = 0x324D4B53U;  // "SKM2" 小端
    static constexpr std::uint32_t kMetaVersion = 3U;   // 3 = 头里带镜像结构（kv_at/layers/kv_bytes）
    static constexpr std::size_t   kHeaderBytes = 64;
    static constexpr std::size_t   kMetaHeaderBytes = 32;

private:
    void close() noexcept;
    [[nodiscard]] bool write_header() const;
    [[nodiscard]] bool read_header();
    // 段偏移的基准（= 头长）；★ 不含水位
    [[nodiscard]] std::uint64_t base_bytes() const noexcept;
    // 一个 part 在【固定容量】下占多少字节 —— 段偏移必须用它，不能用当前水位
    [[nodiscard]] std::uint64_t part_capacity_bytes(Part part) const noexcept;
    // 某个 part 写到 cells 处的末尾偏移
    [[nodiscard]] std::uint64_t part_end(Part part, std::uint64_t cells) const noexcept;
    // 把各 part 截到 cells 处（每个 part 各自截；成行 part 保留整行）
    [[nodiscard]] bool truncate_to(std::uint64_t cells);

    std::string dir_;
    DiskParkSessionPlan plan_{};
    int fd_ = -1;                      // kv.bin
    std::uint64_t log_cells_ = 0;
    std::uint64_t kv_file_bytes_ = 0;  // 头 + 各段在 log_cells_ 下的总长（含稀疏未写区）
};

} // namespace strata::core
