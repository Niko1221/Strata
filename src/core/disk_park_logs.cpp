#include "strata/core/disk_park_logs.hpp"

#include <algorithm>
#include <atomic>
#include <cstdio>
#include <filesystem>
#include <system_error>
#include <thread>

namespace strata::core {
namespace {

// 目录名的前缀：16 位十六进制摘要。用摘要而不是 token 数，是因为同一段前缀
// 在任何时候算出的摘要都一样（纯函数），于是"找到同一个会话"不需要额外的索引文件。
std::string digest_hex(const DiskParkLogDigest& d) {
    char buf[40];
    std::snprintf(buf, sizeof buf, "%016llx%016llx",
                  (unsigned long long) d.lo, (unsigned long long) d.hi);
    return buf;
}

} // namespace

DiskParkLogs::DiskParkLogs(Config cfg) : cfg_(std::move(cfg)) {
    if (cfg_.root.empty()) return;                  // ★ 关闭：行为与没有这一层完全一致
    if (cfg_.tag == 0 || cfg_.heads <= 0 || cfg_.head_dim <= 0 || cfg_.page_size <= 0 ||
        cfg_.idx_dim <= 0 || cfg_.idx_block <= 0 || cfg_.capacity_cells == 0) {
        std::fprintf(stderr, "strata serve: disk park stream: OFF (不完整的 plan)\n");
        return;
    }
    if (cfg_.max_open == 0) cfg_.max_open = 1;
    if (cfg_.max_tracked == 0) cfg_.max_tracked = 1;

    plan_.tag          = cfg_.tag;
    plan_.kv_format    = cfg_.kv_format;
    plan_.heads        = cfg_.heads;
    plan_.head_dim     = cfg_.head_dim;
    plan_.page_size    = cfg_.page_size;
    plan_.idx_dim      = cfg_.idx_dim;
    plan_.capacity_cells = cfg_.capacity_cells;

    // 由「cap 处的字节数」反推每 cell 字节数：
    //   线性 part：bytes_per_cell = bytes / cap
    //   成行 part（pooled）：row_cells = idx_block，bytes = rows*row_bytes + (cap%row_cells)*bpc
    //                        ⇒ row_bytes = (bytes - (cap % row_cells) * bpc) / rows
    // ★ 这些都必须【整除】才敢用：不整除说明调用方给的几何与缓存里的不一致，
    //   宁可整层关掉，也不要按一套错的布局去写盘。
    bool ok = true;
    for (std::size_t i = 0; i < 5 && ok; ++i) {
        const std::uint64_t bytes = cfg_.part_bytes_at_cap[i];
        if (bytes == 0) { plan_.parts[i] = {0, 1, 0}; continue; }
        if (i == 4) {                               // pooled：成行
            // ★★★ 实测修正（2026-10-10）：上游 layout 对 pooled 的行数是
            //     `pooled_rows = upto / idx_block + 1`（conversation_snapshot.cpp:46），
            //   比"capacity_cells / idx_block"【多一行】。于是
            //     cap = 262144、idx_block = 4 时上游给 65537 行；
            //     而 33554944 字节不能被 65536 整除（余 512 = 正好一行）
            //   ⇒ 原来"必须整除"的假设不成立，整层会被判成"几何不自洽"而关掉
            //     （实测就是这样：`disk park stream: OFF (几何不自洽)`）。
            //
            //   改法：两种行数都试，用 part_bytes 的【真语义】自检：
            //     part_bytes(cells) = (cells/row_cells)*row_bytes + (cells%row_cells)*bpc
            //   要求 1) row_bytes == row_cells * bpc（行内自洽，也保证 part_offset 一致）
            //        2) 在 capacity_cells 处覆盖的字节数【正好】等于 bytes
            //
            //   ★ 别再犯的错：不要要求 bpc == idx_dim*sizeof(float)。
            //     bytes_per_cell 是【一个 cell】的字节数（实测 128），
            //     而 idx_dim*4 = 512 是【一整行】的宽度。两者恒不等，加了这条就必然全灭。
            const std::uint64_t row_stride = static_cast<std::uint64_t>(cfg_.idx_block);
            if (row_stride == 0) { ok = false; break; }
            bool settled = false;
            for (int attempt = 0; attempt < 2 && !settled; ++attempt) {
                const std::uint64_t rows = cfg_.capacity_cells / row_stride
                                         + (attempt == 0 ? 0u : 1u);   // 0: 严格；1: 上游 +1
                if (rows == 0) continue;
                const std::uint64_t rem = cfg_.capacity_cells % row_stride;
                if (bytes % rows != 0) continue;
                const std::uint64_t row_bytes = bytes / rows;
                const std::uint64_t bpc = row_bytes / row_stride;
                if (bpc == 0 || row_bytes % row_stride != 0) continue;
                if (row_bytes != row_stride * bpc) continue;
                const std::uint64_t covered = rows * row_bytes + rem * bpc;
                if (covered != bytes) continue;
                plan_.parts[i] = {static_cast<std::size_t>(bpc),
                                  static_cast<std::size_t>(row_stride),
                                  static_cast<std::size_t>(row_bytes)};
                settled = true;
            }
            if (!settled) { ok = false; break; }
        } else {
            if (bytes % cfg_.capacity_cells != 0) { ok = false; break; }
            plan_.parts[i] = {static_cast<std::size_t>(bytes / cfg_.capacity_cells), 1,
                              static_cast<std::size_t>(bytes / cfg_.capacity_cells)};
        }
    }
    if (!ok || !plan_.valid()) {
        std::fprintf(stderr, "strata serve: disk park stream: OFF (几何不自洽)\n");
        return;
    }
    enabled_ = true;
}

DiskParkLogs::~DiskParkLogs() = default;

std::string DiskParkLogs::name_for(const DiskParkLogDigest& d) {
    return digest_hex(d);      // append 开新会话用的就是它
}

std::string DiskParkLogs::dir_for(const std::string& dir_name) const {
    if (!enabled_) return {};
    // 每个 profile 一个子目录：换 kv 格式/上下文档位之后，绝不会读到另一套字节
    char tagbuf[16];
    std::snprintf(tagbuf, sizeof tagbuf, "%08X", cfg_.tag);
    return cfg_.root + "/tag-" + tagbuf + "/" + dir_name;
}

std::string DiskParkLogs::lookup(const DiskParkLogDigest& d) const {
    const auto it = tracked_.find(d);
    return it == tracked_.end() ? std::string{} : it->second.dir_name;
}

std::uint64_t DiskParkLogs::watermark(const DiskParkLogDigest& d) const {
    const auto it = tracked_.find(d);
    return it == tracked_.end() ? 0 : it->second.watermark;
}

DiskParkSession* DiskParkLogs::session_for(const std::string& dir_name, std::uint64_t resume,
                                            bool* opened_now) {
    return session_for_layer(dir_name, 0, resume, opened_now, /*layered=*/false);
}

DiskParkSession* DiskParkLogs::session_for_layer(const std::string& dir_name, std::size_t layer,
                                                 std::uint64_t resume, bool* opened_now,
                                                 bool layered) {
    *opened_now = false;
    // ★ 键要带层号：单层是 dir_name，逐层是 dir_name#NN —— 否则两层会共用同一个 session。
    const std::string key = layered ? (dir_name + "#" + std::to_string(layer)) : dir_name;
    const auto it = open_.find(key);
    if (it != open_.end()) { touch_open_dir(dir_name); return &it->second; }

    DiskParkSession s;
    // ★ 把记忆里的水位带进去：没有 sidecar 时它是唯一的水位来源
    const std::string path = layered ? dir_for_layer(dir_name, layer) : dir_for(dir_name);
    if (!DiskParkSession::open(s, path, plan_, resume)) {
        ++stats_.failures;
        return nullptr;
    }
    *opened_now = true;
    ++stats_.opens;
    auto ins = open_.emplace(key, std::move(s));
    touch_open_dir(dir_name);
    return &ins.first->second;
}

namespace {

// ★★★ 逐层逐 part 的搬运，【按 cell 边界分块】。
//   为什么必须分块：附着在 part 上的回调（如引擎的 p9_read_host_pool）是按
//   "一个 cell 内的一段连续字节"寻址的 —— 一次跨 cell（尤其跨 page）就会越界。
//   实测：尺度区每页只有 ps*heads*bpc = 4*2*8 = 64 字节，而调用方一次要 64 字节，
//   正好越过第一页 ⇒ 回调返回 false ⇒ 整条日志一个字节都写不进。
//   （单 cell 的字节数 = bpc；一次最多搬 bpc。）
bool transfer_cells(const DiskParkSessionPlan& plan, std::uint64_t from, std::uint64_t to,
                    std::size_t part, std::uint8_t* buf,
                    const std::function<bool(std::size_t off, std::uint8_t* dst, std::size_t n)>& op) {
    const auto& p = plan.parts[part];
    if (p.bytes_per_cell == 0) return true;                 // 该 part 不存在
    for (std::uint64_t c = from; c < to; ++c) {
        const std::size_t off = static_cast<std::size_t>(c - from) * p.bytes_per_cell;
        if (!op(off, buf, p.bytes_per_cell)) return false;
    }
    return true;
}

} // namespace

std::uint64_t DiskParkLogs::append_layers(const Request& req, std::size_t layers,
                                          const DiskParkLayerFill& fill) {
    if (!enabled_ || layers == 0 || !fill) return 0;
    if (layers == 1) {
        // 单层退回原路径，语义完全一致
        return append(req, [&fill](std::size_t part, std::size_t off, std::uint8_t* dst, std::size_t n) {
            return fill(0, part, off, dst, n);
        });
    }

    // ---- 会话目录的选定：与 append() 同一套规则，外加【取回直传】----
    // ★★★ P14（2026-10-10）：本轮是从会话目录取回的（req.restore_dir 非空）就直接用它。
    //   摘要链在这里【必然】失效：req.prev = prefix_digest(…, read_from)，read_from 是
    //   挂载点/检查点长度，而 remember() 登记的是上一轮结束时的 L —— 实测相差
    //   1,000~7,000 token（上一轮写到 61,693，下一轮只复用 57,706）。精确匹配的
    //   tracked_ 于是每轮 miss ⇒ 新建目录 + 全量重写。取回路径本来就知道日志是从哪个
    //   目录、恢复到第几个 token 的，那是【事实】，不是猜测。
    std::string dir_name;
    if (!req.restore_dir.empty())  dir_name = req.restore_dir;
    else if (req.has_prev)         dir_name = lookup(req.prev);
    if (dir_name.empty()) {
        dir_name = digest_hex(req.next);          // 新会话
        ++stats_.fresh;
    } else {
        ++stats_.adopted;
    }
    if (req.out_dir != nullptr) *req.out_dir = dir_name;

    // 起始水位 = 该摘要上一轮写到哪（没有则 0）
    // ★★★ 修（2026-10-10）：会话文件自己的水位是【权威】—— state.bin 里就写着它写到哪了。
    //   内存里的摘要表（tracked_）不落盘：重启后 lookup()/watermark() 全 miss ⇒ have = 0
    //   ⇒ 原来会把【整份日志重写一遍】。实测 135k 会话每次多写 1.23 GiB（~8 s + 磁盘磨损），
    //   而那一轮真正新增的只有 8 个 token。取回之后那条请求必然命中这里
    //   （挂载点 ≠ 上一轮 frontier ⇒ prev 摘要查不到）。
    std::uint64_t have = req.has_prev ? watermark(req.prev) : 0;
    // ★ P14：取回直传的水位。盘上那份日志【已经覆盖到】restore_tokens，而引擎的池
    //   正是从那里恢复的 ⇒ 从这里接着追加就是真正的增量。
    if (req.restore_tokens > have) have = req.restore_tokens;
    // ★ 取回直传时【跳过】read_meta 的"把水位抬到该目录完整水位"兜底：那个水位可能
    //   超过身份判据验过的 lcp，等于把 lcp 之后的陈旧字节留在链上（半成品）。
    if (req.restore_dir.empty()) {
        // ★ 注意：不能用 s0->log_cells() —— open() 只在水位持有者【自己的 kv.bin 够长】时
        //   才信 state.bin，而逐层模式下顶层 kv.bin 只有一个头（数据在 layerNN/ 里）
        //   ⇒ log_cells() 恒为 0（实测踩到）。直接读 state.bin 里的水位。
        bool opened_now = false;
        DiskParkSession* s0 = session_for(dir_name, 0, &opened_now);
        if (s0 != nullptr) {
            std::vector<std::byte> pl;
            std::uint64_t cells = 0;
            if (s0->read_meta(pl, cells) && cells > have) have = cells;
        }
    }
    // ★ P17 诊断：有前驱却拿到水位 0 ⇒ 本轮必然整份重写（那 3 轮漏网就是这个签名）。
    //   修完之后这行不该再出现；再出现就是还有别的 miss 路径。
    if (have == 0 && req.has_prev) {
        std::fprintf(stderr, "strata serve: disk park stream: [P17] 有前驱但水位=0"
                             "（摘要链 miss 且没有可用兜底目录）：dir=%s restore_dir=%s"
                             " to_tokens=%llu → 整份重写\n",
                     dir_name.c_str(), req.restore_dir.c_str(),
                     (unsigned long long) req.to_tokens);
    }
    if (req.to_tokens <= have) return 0;          // 没有新增

    std::uint64_t written = 0;
    for (std::size_t L = 0; L < layers; ++L) {
        bool opened_now = false;
        DiskParkSession* sess = session_for_layer(dir_name, L, have, &opened_now, true);
        if (sess == nullptr) return 0;
        // ★ 按 cell 边界分块：回调一次只拿"一个 cell 的字节"
        const DiskParkFill one = [&](std::size_t part, std::size_t off,
                                     std::uint8_t* dst, std::size_t n) {
            if (part >= 5) return false;
            const auto& pp = plan_.parts[part];
            if (pp.bytes_per_cell == 0) return true;
            if ((off % pp.bytes_per_cell) != 0) return false;      // 必须 cell 对齐
            const std::uint64_t cell0 = have + off / pp.bytes_per_cell;
            const std::uint64_t nc   = n / pp.bytes_per_cell;
            for (std::uint64_t k = 0; k < nc; ++k)
                if (!fill(L, part, (std::size_t) ((cell0 - have + k) * pp.bytes_per_cell),
                          dst + (std::size_t) k * pp.bytes_per_cell, pp.bytes_per_cell))
                    return false;
            return true;
        };
        const std::uint64_t w = sess->append_kv(have, req.to_tokens, one);
        if (w == 0) {
            static bool once = false;
            if (!once) { once = true;
                std::fprintf(stderr, "strata serve: disk park stream: [层] %zu 追加 0 字节"
                             " (have=%llu to=%llu pages=%zu bpc=%zu)\n", L,
                             (unsigned long long) have, (unsigned long long) req.to_tokens,
                             (std::size_t) plan_.parts[0].bytes_per_cell,
                             (std::size_t) plan_.parts[0].row_cells);
            }
            return 0;                             // 任一层失败 ⇒ 整体当没写（不做半份）
        }
        {
            static bool once2 = false;
            if (!once2) { once2 = true;
                std::fprintf(stderr, "strata serve: disk park stream: [层] 0 写了 %llu 字节, 继续下一层\n",
                             (unsigned long long) w);
            }
        }
        written += w;
        (void) opened_now;
    }
    // ★★ 把【新的摘要】也记到同一个会话上 —— 这是链能延续的关键：
    //   下一轮客户端带来的"上一轮"就是这个摘要，找不到它就会另开一份
    //   （盘上堆出 N 份互为前缀的日志，正是要消灭的东西）。
    //   第一版把它放在"第一层"的分支里，于是新摘要从没被登记。
    remember(req.next, dir_name, req.to_tokens);
    // ★★★ P15：本轮的别名 frontier 也登记到同一个目录。下一轮的 prev 用的是【检查点
    //   长度】（引擎挂在哪个检查点上，read_from 就是它），只登记 next 的话必然 miss。
    //   水位取该 frontier 的长度：盘上那份日志至少覆盖到它（都是同一条对话的前缀）。
    for (const auto& a : req.aliases)
        if (a.tokens > 0 && a.tokens < req.to_tokens) remember(a.digest, dir_name, a.tokens);
    touch(req.next);
    ++stats_.writes;
    stats_.write_bytes += written;
    evict_open_dirs();       // ★ P18：逐层路径也要有界（单层那条老循环保持不变）
    return written;
}

bool DiskParkLogs::read_layer_cells(DiskParkSession& sess, std::size_t L, std::uint64_t from_cell,
                                    std::uint64_t to_cell, const DiskParkLayerRead& read) {
    for (std::size_t part = 0; part < 5; ++part) {
        const auto p = static_cast<DiskParkSession::Part>(part);
        const auto& pp = plan_.parts[part];
        if (pp.bytes_per_cell == 0) continue;
        // ★★ 批量 pread（2026-10-10 修）：一次读 1 MiB 那么多 cell。
        //   原来【一个 cell 一次 read_kv】= 一个 cell 一次 pread 系统调用：
        //   135k token 的会话要 12 层 × 3 part × 134,696 = 485 万次，实测把
        //   引擎的「60 s 无进展」看门狗踩爆（引擎主动停掉 ⇒ HTTP 503）。
        //   回调【仍然一个 cell 一次】—— DiskParkLayerRead 的 offset 语义不变。
        const std::size_t batch_cells =
            std::max<std::size_t>(1, (static_cast<std::size_t>(1) << 20) / pp.bytes_per_cell);
        std::vector<std::uint8_t> cell_buf(batch_cells * pp.bytes_per_cell);
        for (std::uint64_t c = from_cell; c < to_cell;) {
            const std::uint64_t n_cells =
                std::min<std::uint64_t>(static_cast<std::uint64_t>(batch_cells), to_cell - c);
            const std::size_t n_bytes = static_cast<std::size_t>(n_cells) * pp.bytes_per_cell;
            if (!sess.read_kv(p, c, c + n_cells, cell_buf.data(), n_bytes)) return false;
            for (std::uint64_t k = 0; k < n_cells; ++k) {
                if (!read(L, part, static_cast<std::size_t>((c - from_cell + k) * pp.bytes_per_cell),
                          cell_buf.data() + static_cast<std::size_t>(k) * pp.bytes_per_cell,
                          pp.bytes_per_cell))
                    return false;
            }
            c += n_cells;
        }
    }
    return true;
}

bool DiskParkLogs::read_layers(const std::string& dir_name, std::size_t layers,
                               std::uint64_t from_cell, std::uint64_t to_cell,
                               const DiskParkLayerRead& read) {
    if (!enabled_ || dir_name.empty() || layers == 0 || !read || to_cell <= from_cell) return false;
    for (std::size_t L = 0; L < layers; ++L) {
        bool opened_now = false;
        DiskParkSession* sess = session_for_layer(dir_name, L, 0, &opened_now, layers > 1);
        if (sess == nullptr) return false;
        if (!read_layer_cells(*sess, L, from_cell, to_cell, read)) return false;
    }
    return true;
}

// ★★★ 并发版（2026-10-10）：12 层各读各的 —— 层与层之间没有共享状态
//   （每层自己的 session、自己的 host 池），pread 也不改文件偏移。
//   唯一不能并发的是 session_for_layer()（它改 open_ 这个 map）⇒ 先串行开好。
bool DiskParkLogs::read_layers_parallel(const std::string& dir_name, std::size_t layers,
                                        std::uint64_t from_cell, std::uint64_t to_cell,
                                        const DiskParkLayerRead& read, int threads) {
    if (!enabled_ || dir_name.empty() || layers == 0 || !read || to_cell <= from_cell) return false;
    if (threads <= 1 || layers == 1) return read_layers(dir_name, layers, from_cell, to_cell, read);
    std::vector<DiskParkSession*> sess(layers, nullptr);
    for (std::size_t L = 0; L < layers; ++L) {
        bool opened_now = false;
        sess[L] = session_for_layer(dir_name, L, 0, &opened_now, layers > 1);
        if (sess[L] == nullptr) return false;
    }
    std::atomic<std::size_t> next{0};
    std::atomic<bool> ok{true};
    const int n = std::min<int>(threads, static_cast<int>(layers));
    std::vector<std::thread> pool;
    pool.reserve(static_cast<std::size_t>(n));
    for (int t = 0; t < n; ++t) {
        pool.emplace_back([&] {
            for (;;) {
                const std::size_t L = next.fetch_add(1);
                if (L >= layers || !ok.load()) return;
                if (!read_layer_cells(*sess[L], L, from_cell, to_cell, read)) {
                    ok.store(false);
                    return;
                }
            }
        });
    }
    for (auto& th : pool) th.join();
    return ok.load();
}

void DiskParkLogs::touch(const DiskParkLogDigest& d) {
    const auto it = tracked_.find(d);
    if (it == tracked_.end()) return;
    lru_.erase(it->second.lru);
    lru_.push_front(d);
    it->second.lru = lru_.begin();
}

void DiskParkLogs::remember(const DiskParkLogDigest& d, const std::string& dir_name,
                            std::uint64_t tokens) {
    auto it = tracked_.find(d);
    if (it == tracked_.end()) {
        lru_.push_front(d);
        Track t;
        t.dir_name  = dir_name;
        t.watermark = tokens;
        t.lru       = lru_.begin();
        it = tracked_.emplace(d, std::move(t)).first;
    } else {
        // ★ 目录名一旦定下就不再改：同一个摘要必须稳定地指向同一个文件
        if (it->second.dir_name.empty()) it->second.dir_name = dir_name;
        it->second.watermark = tokens;
        touch(d);
    }
    trim();
}

void DiskParkLogs::trim() {
    // ★★ 淘汰按【最近用过】，不按水位。
    //    对话链是 D1→D2→D3…，被用到的是"每个请求带来的那个 prev" ——
    //    那是链上前一个节点，可能水位很小（早期轮次）。按水位淘汰会掐掉它，
    //    于是下一轮找不到文件、当新会话重写一份（实测：同一条对话被写成多个目录）。
    //    被淘汰的只是【记忆】：盘上的文件还在，下次遇到该摘要会当新会话重写 ——
    //    代价是多写一次，不会写错。
    while (tracked_.size() > cfg_.max_tracked && !lru_.empty()) {
        const DiskParkLogDigest victim_key = lru_.back();
        lru_.pop_back();
        const auto victim = tracked_.find(victim_key);
        if (victim == tracked_.end()) continue;
        // 还有别的摘要指向同一目录时，不关它的 session（关了要重开一遍）
        const std::string dir = victim->second.dir_name;
        bool still_referenced = false;
        for (const auto& kv : tracked_)
            if (kv.first != victim->first && kv.second.dir_name == dir) { still_referenced = true; break; }
        if (!still_referenced) {
            const auto o = open_.find(dir);
            if (o != open_.end()) { open_.erase(o); ++stats_.closes; }
        }
        tracked_.erase(victim);
    }
}

bool DiskParkLogs::commit_watermark(const DiskParkLogDigest& d, std::uint64_t tokens) {
    if (!enabled_) return false;
    const auto it = tracked_.find(d);
    if (it == tracked_.end() || it->second.dir_name.empty()) return false;
    bool opened_now = false;
    DiskParkSession* s = session_for(it->second.dir_name, 0, &opened_now);
    if (s == nullptr) return false;
    return s->commit(tokens, {});
}

bool DiskParkLogs::commit_image(const std::string& dir_name, std::uint64_t tokens,
                                std::span<const std::byte> meta_packed, std::uint64_t kv_at) {
    // ★ P12：提交【镜像元数据 + 水位】。dir_name 由调用方用摘要的 hex 给出
    //   （与 session_dir_for() 用的是同一个名字），于是恢复端按摘要就能直接找到它。
    if (!enabled_ || dir_name.empty() || meta_packed.empty()) return false;
    bool opened_now = false;
    DiskParkSession* s = session_for(dir_name, 0, &opened_now);
    if (s == nullptr) return false;
    return s->commit(tokens, meta_packed, kv_at);
}

void DiskParkLogs::touch_open_dir(const std::string& dir_name) {
    if (dir_name.empty()) return;
    open_dir_stamp_[dir_name] = ++open_stamp_;      // ★ 单调递增 = 最近用过的排最后
}

// ★★ P18：逐层路径的淘汰。为什么必须按【目录】而不是按条目：
//   open_ 的键是 "目录#层号"，max_open=16 若按条目算，等于只留 1.3 个会话目录
//   —— 那既不是本意，也会在写一轮 12 层的中途把别的层掐掉。
//   所以这里数【目录】：超过 max_open 就把最久没用过的整个目录关掉（12 层一起）。
//   代价：下次再写那个目录要重开一次（读一遍头/state.bin），换来 fd 有界。
void DiskParkLogs::evict_open_dirs() {
    const std::size_t cap = std::max<std::size_t>(cfg_.max_open, 1);
    while (open_dir_stamp_.size() > cap) {
        auto victim = open_dir_stamp_.begin();
        for (auto it = open_dir_stamp_.begin(); it != open_dir_stamp_.end(); ++it)
            if (it->second < victim->second) victim = it;
        const std::string dir = victim->first;
        open_dir_stamp_.erase(victim);
        const std::string prefix = dir + "#";
        for (auto it = open_.begin(); it != open_.end(); ) {
            if (it->first == dir || it->first.compare(0, prefix.size(), prefix) == 0) {
                it = open_.erase(it);
                ++stats_.closes;
            } else {
                ++it;
            }
        }
    }
}

std::uint64_t DiskParkLogs::append(const Request& req, const DiskParkFill& fill) {
    if (!enabled_ || !fill) return 0;

    std::string dir_name;
    std::uint64_t from = 0;
    if (req.has_prev) {
        const auto it = tracked_.find(req.prev);
        if (it != tracked_.end()) {
            dir_name = it->second.dir_name;
            from = it->second.watermark;          // 上一轮写到哪，就从哪继续
            touch(req.prev);                      // ★ 用到了就刷新，别被淘汰掉
            ++stats_.adopted;
        }
    }
    if (dir_name.empty()) {
        // 新会话（或记忆已被淘汰）：开一个新的。★ 宁可多写一份，也不猜一个水位。
        dir_name = digest_hex(req.next);
        ++stats_.fresh;
    }
    // ★ P12：会话目录名 = 本轮写完之后的整段摘要（调用方要用它去 commit_image）。
    //   早先用 Request::dir_name 让调用方指定；改成从摘要直接算，
    //   因为调用方本来就只有摘要，多传一个字段反而容易两边不一致。
    if (req.out_dir != nullptr) *req.out_dir = dir_name;

    // ★ 单调：水位不能倒退。真倒退了（客户端改写历史）就当新会话重写，
    //   因为把旧的 log 截断重写需要调用方给"从哪截"，那是阶段 3/4 的事。
    if (req.to_tokens <= from) return 0;

    bool opened_now = false;
    DiskParkSession* s = session_for(dir_name, from, &opened_now);
    if (s == nullptr) return 0;

    // 会话自己的水位可能与记忆不一致（记忆被淘汰过、或文件是刚建的）——以【文件】为准，
    // 但只有它不高于 from 才安全续写；高了说明磁盘上比我们记的多，直接信它更省。
    const std::uint64_t file_cells = s->log_cells();
    std::uint64_t start = from;
    if (file_cells > start) start = file_cells;
    if (req.to_tokens <= start) return 0;

    const std::uint64_t wrote = s->append_kv(start, req.to_tokens, fill);
    if (wrote == 0) { ++stats_.failures; return 0; }

    // ★ 旧的摘要也指向同一个文件：下一轮请求带来的"上一轮摘要"就是 req.next，
    //   但客户端若基于更早的一轮重发（且前缀没变），req.prev 也要能找到它。
    remember(req.prev, dir_name, start);
    remember(req.next, dir_name, req.to_tokens);
    ++stats_.writes;
    stats_.write_bytes += wrote;

    // 打开的会话太多就关掉最久没用的（按字典序无意义，这里按水位：最短的先关）
    while (open_.size() > cfg_.max_open) {
        auto victim = open_.begin();
        for (auto it = open_.begin(); it != open_.end(); ++it)
            if (it->second.log_cells() < victim->second.log_cells()) victim = it;
        open_dir_stamp_.erase(victim->first);   // ★ P18：别让 open_dirs() 说谎
        open_.erase(victim);
        ++stats_.closes;
    }
    (void) opened_now;
    return wrote;
}

} // namespace strata::core
