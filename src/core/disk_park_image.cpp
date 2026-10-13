#include "strata/core/disk_park_image.hpp"

#include <cstring>
#include <limits>

namespace strata::core {
namespace {

constexpr std::uint32_t kMagic   = 0x314B5053U;   // "SPK1" 小端
constexpr std::uint32_t kVersion = 1U;

// ★ 每个元素的【最小】字节数，用于 count_ok 的长度上限检查。
//   早先这几处传 elem=1，于是一个伪造的巨大 count 会申请 ~200x left 的内存
//   （每元素约 200 字节），resize 抛 bad_alloc 会穿过本文件"绝不抛"的契约。
//   用真实下界即可把这放大压掉。
constexpr std::size_t kMinSavedBytes      = 160;   // 18*8 + 2*8 + 一个 live 检查点
constexpr std::size_t kMinCheckpointBytes = 56;    // 5 个长度 + used + 1 个长度
constexpr std::size_t kMinKvBytes         = 52;    // format + 6 个 int64 + 5 个长度

// ---------------------------------------------------------------- 写

// ★ 写入器现在是"面向汇"的：默认汇进一个 vector（测试用），
//   也可以直接汇给磁盘（Spool 用），于是不需要完整中间缓冲。
struct Writer {
    PackSink* sink = nullptr;
    std::vector<std::byte> out;          // sink == nullptr 时才用
    bool ok = true;

    void raw(const void* p, std::size_t n) {
        if (!ok) return;
        if (sink) { if (!sink->write(p, n)) ok = false; return; }
        const auto* b = static_cast<const std::byte*>(p);
        out.insert(out.end(), b, b + n);
    }
    void u8(std::uint8_t v)  { raw(&v, 1); }
    void u32(std::uint32_t v) { raw(&v, sizeof v); }   // ★ 宿主是小端；格式文档写明小端
    void u64(std::uint64_t v) { raw(&v, sizeof v); }
    void i32(std::int32_t v)  { raw(&v, sizeof v); }
    void i64(std::int64_t v)  { raw(&v, sizeof v); }
    void bool_(bool v)        { u8(v ? 1 : 0); }
};

// 迭代 ConversationBuffer，把它整块追加出去（分片流式，不额外拷一份整体）
void put_buffer(Writer& w, const ConversationBuffer& buf) {
    const std::size_t n = buf.size();
    w.u64(n);
    if (n == 0) return;
    std::size_t written = 0;
    // ★ const 缓冲的 visit 传 const uint8_t*
    buf.visit(0, n, [&](const std::uint8_t* p, std::size_t len, std::size_t /*at*/) {
        w.raw(p, len);
        written += len;
        return true;
    });
    (void)written;
}

void put_ids(Writer& w, const std::vector<std::int32_t>& v) {
    w.u64(v.size());
    for (auto x : v) w.i32(x);
}
void put_u8v(Writer& w, const std::vector<std::uint8_t>& v) {
    w.u64(v.size());
    if (!v.empty()) w.raw(v.data(), v.size());
}
void put_imgs(Writer& w, const std::vector<ConversationImageKey>& v) {
    w.u64(v.size());
    for (const auto& k : v) { w.i64(k.start); w.u64(k.hash); }
}

void put_checkpoint(Writer& w, const ConversationCheckpoint& c) {
    put_ids(w, c.ids);
    put_imgs(w, c.imgs);
    put_u8v(w, c.gdn);
    put_u8v(w, c.ple);
    put_u8v(w, c.tails);
    put_u8v(w, c.dead);
    put_u8v(w, c.block_pos);
    w.u64(c.used);
    w.u64(c.stage_parts.size());
    for (const auto& p : c.stage_parts) put_checkpoint(w, p);
}

void put_kv(Writer& w, const ConversationKv& k) {
    w.i32(k.format);
    w.i64(k.cells); w.i64(k.heads); w.i64(k.head_dim);
    w.i64(k.page_size); w.i64(k.pooled_rows); w.i64(k.idx_dim);
    put_buffer(w, k.k);
    put_buffer(w, k.v);
    put_buffer(w, k.k_scale);
    put_buffer(w, k.v_scale);
    put_buffer(w, k.pooled);
}

void put_saved(Writer& w, const SavedConversation& s) {
    for (std::size_t i = 0; i < s.geometry.size(); ++i) w.i64(s.geometry[i]);
    w.i64(s.layer_lo);
    w.i64(s.layer_hi);
    put_checkpoint(w, s.live);
    w.u64(s.checkpoints.size());
    for (const auto& c : s.checkpoints) put_checkpoint(w, c);
    w.u64(s.kv.size());
    for (const auto& k : s.kv) put_kv(w, k);
    w.bool_(s.cvec);
    w.u64(s.stage_images.size());
    for (const auto& si : s.stage_images) put_saved(w, si);
}

// 纯计数（不分配），用于 packed_saved_conversation_bytes
std::uint64_t counted_checkpoint(const ConversationCheckpoint& c) {
    std::uint64_t n = 8 + c.ids.size() * 4;
    n += 8 + c.imgs.size() * 16;
    n += 8 + c.gdn.size() + 8 + c.ple.size() + 8 + c.tails.size() + 8 + c.dead.size() + 8 + c.block_pos.size();
    n += 8 + 8;
    for (const auto& p : c.stage_parts) n += counted_checkpoint(p);
    return n;
}
std::uint64_t counted_saved(const SavedConversation& s) {
    // ★ 用 s.geometry.size() 而不是硬编码 18 —— 上游一旦改 std::array 的长度，
    //   计数与字节流就会错位，而症状是每条 load 静默解析错位。
    std::uint64_t n = s.geometry.size() * 8 + 8 + 8;
    n += counted_checkpoint(s.live);
    n += 8;
    for (const auto& c : s.checkpoints) n += counted_checkpoint(c);
    n += 8;
    for (const auto& k : s.kv) {
        n += 4 + 6 * 8;
        n += (8 + k.k.size()) + (8 + k.v.size()) + (8 + k.k_scale.size()) +
             (8 + k.v_scale.size()) + (8 + k.pooled.size());
    }
    n += 1 + 8;
    for (const auto& si : s.stage_images) n += counted_saved(si);
    return n;
}

// ---------------------------------------------------------------- 读

// ★ 从 PackSource 顺序拉取（不再持有整份 span）。
struct Reader {
    PackSource* src = nullptr;
    std::uint64_t left = 0;      // 还剩多少字节（来自探测页的总长）
    bool ok = true;

    bool need(std::size_t n) {
        if (!ok || n > left) { ok = false; return false; }
        return true;
    }
    template <class T>
    T get() {
        T v{};
        if (!need(sizeof(T))) return v;
        if (!src->read(&v, sizeof(T))) { ok = false; return T{}; }
        left -= sizeof(T);
        return v;
    }
    // 读一大块到调用方的缓冲（不缓存整份）
    bool read_into(void* dst, std::size_t n) {
        if (!need(n)) return false;
        if (!src->read(dst, n)) { ok = false; return false; }
        left -= n;
        return true;
    }
    std::uint8_t  u8()  { return get<std::uint8_t>(); }
    std::uint32_t u32() { return get<std::uint32_t>(); }
    std::uint64_t u64() { return get<std::uint64_t>(); }
    std::int32_t  i32() { return get<std::int32_t>(); }
    std::int64_t  i64() { return get<std::int64_t>(); }
    bool          bool_() { return u8() != 0; }
    // ★ 长度先做上限检查，避免被一个伪造的巨大 count 拖进 OOM
    bool count_ok(std::uint64_t n, std::size_t elem) {
        if (!ok) return false;
        if (elem != 0 && n > left / elem + 1) { ok = false; return false; }
        return true;
    }
};

bool get_buffer(Reader& r, ConversationBuffer& out) {
    const std::uint64_t n = r.u64();
    if (!r.ok) return false;
    if (n > (std::uint64_t)std::numeric_limits<std::size_t>::max()) { r.ok = false; return false; }
    if (!r.need(static_cast<std::size_t>(n))) return false;
    out.resize(static_cast<std::size_t>(n));
    if (n != 0) {
        // ★ 直接读进 ConversationBuffer 的分段，不经过中间大缓冲
        bool fed = true;
        std::size_t done = 0;
        out.visit(0, static_cast<std::size_t>(n), [&](std::uint8_t* p, std::size_t len, std::size_t) {
            if (!fed) return false;
            if (!r.read_into(p, len)) { fed = false; return false; }
            done += len;
            return true;
        });
        if (!fed || done != static_cast<std::size_t>(n)) { r.ok = false; return false; }
    }
    return true;
}

bool get_ids(Reader& r, std::vector<std::int32_t>& v) {
    const std::uint64_t n = r.u64();
    if (!r.count_ok(n, 4)) return false;
    v.resize(static_cast<std::size_t>(n));
    for (std::size_t i = 0; i < v.size(); ++i) v[i] = r.i32();
    return r.ok;
}
bool get_u8v(Reader& r, std::vector<std::uint8_t>& v) {
    const std::uint64_t n = r.u64();
    if (!r.need(static_cast<std::size_t>(n))) return false;
    v.resize(static_cast<std::size_t>(n));
    if (n != 0 && !r.read_into(v.data(), static_cast<std::size_t>(n))) return false;
    return true;
}
bool get_imgs(Reader& r, std::vector<ConversationImageKey>& v) {
    const std::uint64_t n = r.u64();
    if (!r.count_ok(n, 16)) return false;
    v.resize(static_cast<std::size_t>(n));
    for (auto& k : v) { k.start = r.i64(); k.hash = r.u64(); }
    return r.ok;
}

bool get_checkpoint(Reader& r, ConversationCheckpoint& c, int depth) {
    if (depth > 8) { r.ok = false; return false; }     // ★ 递归深度上限
    if (!get_ids(r, c.ids)) return false;
    if (!get_imgs(r, c.imgs)) return false;
    if (!get_u8v(r, c.gdn)) return false;
    if (!get_u8v(r, c.ple)) return false;
    if (!get_u8v(r, c.tails)) return false;
    if (!get_u8v(r, c.dead)) return false;
    if (!get_u8v(r, c.block_pos)) return false;
    c.used = r.u64();
    const std::uint64_t n = r.u64();
    if (!r.count_ok(n, kMinCheckpointBytes)) return false;   // ★ 不用 1
    c.stage_parts.resize(static_cast<std::size_t>(n));
    for (auto& p : c.stage_parts) if (!get_checkpoint(r, p, depth + 1)) return false;
    return r.ok;
}

bool get_kv(Reader& r, ConversationKv& k) {
    k.format = r.i32();
    k.cells = r.i64(); k.heads = r.i64(); k.head_dim = r.i64();
    k.page_size = r.i64(); k.pooled_rows = r.i64(); k.idx_dim = r.i64();
    if (!r.ok) return false;
    return get_buffer(r, k.k) && get_buffer(r, k.v) && get_buffer(r, k.k_scale) &&
           get_buffer(r, k.v_scale) && get_buffer(r, k.pooled);
}

bool get_saved(Reader& r, SavedConversation& s, int depth) {
    if (depth > 8) { r.ok = false; return false; }
    for (std::size_t i = 0; i < s.geometry.size(); ++i) s.geometry[i] = r.i64();
    s.layer_lo = r.i64();
    s.layer_hi = r.i64();
    if (!r.ok) return false;
    if (!get_checkpoint(r, s.live, 0)) return false;
    const std::uint64_t nc = r.u64();
    if (!r.count_ok(nc, kMinCheckpointBytes)) return false;  // ★ 不用 1
    s.checkpoints.resize(static_cast<std::size_t>(nc));
    for (auto& c : s.checkpoints) if (!get_checkpoint(r, c, 0)) return false;
    const std::uint64_t nk = r.u64();
    if (!r.count_ok(nk, kMinKvBytes)) return false;          // ★ 不用 1
    s.kv.resize(static_cast<std::size_t>(nk));
    for (auto& k : s.kv) if (!get_kv(r, k)) return false;
    s.cvec = r.bool_();
    const std::uint64_t ns = r.u64();
    if (!r.count_ok(ns, kMinSavedBytes)) return false;       // ★ 不用 1
    s.stage_images.resize(static_cast<std::size_t>(ns));
    for (auto& si : s.stage_images) if (!get_saved(r, si, depth + 1)) return false;
    return r.ok;
}

} // namespace

namespace {

// ★ P7 只读拆解：把「一份快照的字节」按来源归类。★ 一个字节只能被归一类，
//   所以 total 必须与 packed_saved_conversation_bytes() 逐字节相等 —— 这是这个函数的自检。
// 把一个检查点【自己的长度字段】记到 parts（与 counted_checkpoint 逐字段对齐）：
//   ids/imgs 各一个长度字段（已含在 ids/imgs 里），gdn/ple/tails/dead/block_pos 各一个，
//   再加 c.used(8) 与 stage_parts.size()(8)。子检查点只【走】不记 —— 否则顶层会多算 8 字节。
void walk_checkpoint(const ConversationCheckpoint& c, bool live, ParkBreakdown& z) {
    auto& gdn = live ? z.live_gdn : z.ck_gdn;
    auto& ple = live ? z.live_ple : z.ck_ple;
    auto& tail = live ? z.live_tail : z.ck_tail;
    auto& dead = live ? z.live_dead : z.ck_dead;
    auto& pos = live ? z.live_pos : z.ck_pos;
    auto& ids = live ? z.live_ids : z.ck_ids;
    auto& imgs = live ? z.live_imgs : z.ck_imgs;
    auto& parts = live ? z.live_parts : z.ck_parts;
    ids += 8 + (std::uint64_t) c.ids.size() * 4;
    imgs += 8 + (std::uint64_t) c.imgs.size() * 16;
    gdn += 8 + c.gdn.size();
    ple += 8 + c.ple.size();
    tail += 8 + c.tails.size();
    dead += 8 + c.dead.size();
    pos += 8 + c.block_pos.size();
    // ★ 只加 c.used(8)：stage_parts.size() 的长度字段【已经】被 counted_saved() 顶层那个
    //   `n += 8`（checkpoints.size()）算过了 —— 一个检查点流里只有【一个】份数字段。
    //   （compare 实测：packed_saved_conversation_bytes 比实际字节多 8，正是这一对）
    parts += 8;
    for (const auto& p : c.stage_parts) walk_checkpoint(p, live, z);
}

// ★★ P7 顺带发现的一处【真实记账不一致】，必须量出来：
//   counted_saved() 递归进 stage_images 时只算了 `n += counted_saved(si)`；
//   而 put_saved() 写 stage 时写的是 `8 + counted_saved(si)`（那个 u64 长度字段），
//   并且 put_saved() 会连 stage 自己的 checkpoints[]、cvec、geometry 一起写。
//   ⇒ 一旦 stage_images 非空（层切分），packed_saved_conversation_bytes() 就会【少算】，
//     而 spool 是拿它当 expected_bytes 用的 —— 少算等于按短了读。
//   本机（单卡）stage_images 恒为空，所以从未暴露。两个函数把它们分开数出来：
//     counted_total(si)  = 上游 counted_saved 认为的大小（不含 stage 自己的 checkpoints）
//     written_total(si)  = put_saved 实际会写的大小（含）
std::uint64_t counted_total(const SavedConversation& s) {
    std::uint64_t n = s.geometry.size() * 8 + 8 + 8;
    n += counted_checkpoint(s.live);
    n += 8;
    for (const auto& c : s.checkpoints) n += counted_checkpoint(c);
    n += 8;
    for (const auto& k : s.kv) {
        n += 4 + 6 * 8;
        n += (8 + k.k.size()) + (8 + k.v.size()) + (8 + k.k_scale.size()) +
             (8 + k.v_scale.size()) + (8 + k.pooled.size());
    }
    n += 1 + 8;
    for (const auto& si : s.stage_images) n += counted_total(si);   // ★ 上游的写法
    return n;
}
std::uint64_t written_total(const SavedConversation& s) {
    std::uint64_t n = s.geometry.size() * 8 + 8 + 8;
    n += counted_checkpoint(s.live);
    n += 8;
    for (const auto& c : s.checkpoints) n += counted_checkpoint(c);
    n += 8;
    for (const auto& k : s.kv) {
        n += 4 + 6 * 8;
        n += (8 + k.k.size()) + (8 + k.v.size()) + (8 + k.k_scale.size()) +
             (8 + k.v_scale.size()) + (8 + k.pooled.size());
    }
    n += 1 + 8;
    for (const auto& si : s.stage_images) n += 8 + written_total(si);   // ★ put_saved 的写法
    return n;
}

} // namespace

std::uint64_t park_checkpoint_bytes(const ConversationCheckpoint& c) { return counted_checkpoint(c); }

ParkBreakdown park_breakdown(const SavedConversation& image) {
    ParkBreakdown z;
    z.checkpoints = image.checkpoints.size();
    z.tokens      = image.live.ids.size();

    // geometry + layer_lo/hi + live.kv.size() + cvec + stage_images.size +
    // live.checkpoints.size + checkpoints.size + 两个魔数 —— 见 put_saved/counted_saved
    z.metadata = (std::uint64_t) image.geometry.size() * 8 + 8 + 8 + 8 + 1 + 8 + 8 + 8 + 4 + 4;

    walk_checkpoint(image.live, true, z);
    for (const auto& c : image.checkpoints) walk_checkpoint(c, false, z);

    for (const auto& k : image.kv) {
        z.kv_k     += 8 + k.k.size();
        z.kv_v     += 8 + k.v.size();
        z.kv_scale += (8 + k.k_scale.size()) + (8 + k.v_scale.size());
        z.kv_pooled += 8 + k.pooled.size();
        z.metadata += 4 + 6 * 8;                     // format + 6 个 int64
    }

    // 每个 stage 是一份独立的 SavedConversation，整份都归 stage_images（含它自己的
    // checkpoints / kv / 嵌套 stage）—— 这样 total 才与 put_saved 实际写的字节相等。
    for (const auto& si : image.stage_images) {
        z.stage_images += 8 + written_total(si);      // ★ put_saved 会先写一个 u64 长度
        z.stage_written += written_total(si);
        z.stage_counted += counted_total(si);
    }

    z.total = z.kv_k + z.kv_v + z.kv_scale + z.kv_pooled
            + z.live_gdn + z.live_ple + z.live_tail + z.live_dead + z.live_pos
            + z.live_ids + z.live_imgs + z.live_parts
            + z.ck_gdn + z.ck_ple + z.ck_tail + z.ck_dead + z.ck_pos
            + z.ck_ids + z.ck_imgs + z.ck_parts
            + z.stage_images + z.metadata;
    return z;
}

std::size_t packed_saved_conversation_bytes(const SavedConversation& image) {
    return static_cast<std::size_t>(4 + 4 + counted_saved(image));
}

bool pack_saved_conversation_into(const SavedConversation& image, PackSink& sink) {
    Writer w;
    w.sink = &sink;
    w.u32(kMagic);
    w.u32(kVersion);
    put_saved(w, image);
    return w.ok;
}

namespace {
struct VectorSink : PackSink {
    std::vector<std::byte> out;
    bool write(const void* data, std::size_t n) override {
        const auto* b = static_cast<const std::byte*>(data);
        out.insert(out.end(), b, b + n);
        return true;
    }
};
} // namespace

std::vector<std::byte> pack_saved_conversation(const SavedConversation& image) {
    VectorSink sink;
    sink.out.reserve(packed_saved_conversation_bytes(image));
    if (!pack_saved_conversation_into(image, sink)) return {};
    return std::move(sink.out);
}

namespace {
struct SpanSource final : PackSource {
    std::span<const std::byte> in;
    std::size_t at = 0;
    // ★ 有虚析构的基类 → 不是聚合类型，必须显式构造
    SpanSource(std::span<const std::byte> b, std::size_t a) : in(b), at(a) {}
    bool read(void* dst, std::size_t n) override {
        if (n > in.size() - at) return false;
        std::memcpy(dst, in.data() + at, n);
        at += n;
        return true;
    }
};
} // namespace

std::optional<SavedConversation> unpack_saved_conversation_from(PackSource& source,
                                                               std::uint64_t expected_bytes) {
    Reader r{&source, expected_bytes, true};
    if (r.u32() != kMagic) return std::nullopt;
    if (r.u32() != kVersion) return std::nullopt;
    SavedConversation s;
    if (!get_saved(r, s, 0)) return std::nullopt;
    if (!r.ok || r.left != 0) return std::nullopt;   // ★ 尾部有多余字节也拒绝
    return s;
}

std::optional<SavedConversation> unpack_saved_conversation(std::span<const std::byte> bytes) {
    SpanSource src{bytes, 0};
    return unpack_saved_conversation_from(src, bytes.size());
}

} // namespace strata::core
