// src/core/weights.cpp - the dense-weight loader.  See the header for the engine-vs-pack distinction.
#define DPCT_PROFILING_ENABLED
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include "strata/core/weights.hpp"

#include "strata/kernels/f16_bits.hpp"
#include "strata/platform/memory.hpp"

#include <chrono>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#if defined(_WIN32)
#include <io.h>
#else
#include <unistd.h>
#endif

namespace strata::core {
namespace {

constexpr uint64_t CHUNK = 8ull << 20;   // 8 MiB of SOURCE per staging round

double now_ms() {
    using namespace std::chrono;
    return duration<double, std::milli>(steady_clock::now().time_since_epoch()).count();
}

struct IndexRow {
    std::string name;
    int file = 0;
    WeightKind kind = WeightKind::Verbatim;
    uint64_t src_off = 0, src_bytes = 0, dst_off = 0, dst_bytes = 0;
    int64_t ne0 = 0, ne1 = 0;
    int code_bits = 0, code_bias = 0, group_elems = 0, codebook = 0, has_offset = 0;
    uint64_t codes_bytes = 0, scales_bytes = 0, offset_bytes = 0;
    int scales_fp16 = 0;
    int act_kind = 0;
    bool raw16 = false;   ///< plan v0.3 P6: already 16-bit in the file (kinds 4/5)
};

/// How a segment's bytes must be transformed on the way from the pack to the arena.
enum class Conv : int {
    Copy = 0,      ///< the bytes are already the engine form
    Bf16High = 1,  ///< f32 container holding a bf16 value: take the high 16 bits
    ToF16 = 2,     ///< f32 container holding an f16 value: a real round-to-nearest-even conversion
    WidenF16 = 3,  ///< fp16 bits to be held as f32: EXACT, and the reason this path exists at all
};

/// One contiguous run of a tensor's source span with a single destination range and one transform.
///
/// THE SEGMENT LIST IS WHY THIS LOADER CAN CHANGE WIDTHS PLANE BY PLANE.  A quantized tensor used to be one
/// verbatim span, which is only correct when every plane is already in engine form - and for 90 of the 303
/// quantized tensors the scale plane is fp16 in the pack and must be f32 in the arena.
struct Seg {
    uint64_t src_off = 0;   ///< relative to the row's src_off
    uint64_t src_bytes = 0;
    uint64_t dst_off = 0;   ///< relative to the row's dst_off
    uint64_t dst_bytes = 0;
    Conv conv = Conv::Copy;
};

/// One index row's footing for THIS rank: whether to read it at all, how much of each plane, and the shape the
/// arena will hold.  Every division here is CHECKED, because a plane that is not uniform per unit of the split
/// axis would size the arena one row short - and an arena one row short decodes to plausible weights.
struct RowShard {
    bool keep = true;
    ShardAxis axis = ShardAxis::Replicated;
    int ranges = 0;
    ShardRange range[kShardMaxRanges] = {};
    int64_t units = 0;                                     ///< the split axis's full extent; 0 = not split
    uint64_t codes = 0, scales = 0, offset = 0, dst = 0;    ///< the SHARDED plane sizes
    int64_t ne0 = 0, ne1 = 0;                               ///< the SHARDED shape the row reports
};

bool row_shard(const IndexRow& r, const TensorShard* ts, RowShard& out, std::string& err) {
    out = RowShard{};
    out.ne0 = r.ne0;
    out.ne1 = r.ne1;
    if (ts != nullptr && !ts->keep) {
        out.keep = false;
        return true;
    }
    const bool split = ts != nullptr && (ts->axis == ShardAxis::Output || ts->axis == ShardAxis::Input);
    // A QUANTIZED contraction split keeps the same block range inside EVERY row, so this row's three planes
    // become one source run per row instead of three contiguous runs.  The sizing below already says so -
    // `shard_bytes(plane, units = ne0, kept)` is `plane / ne0 * kept`, which is the same bytes whether they
    // are gathered per row or taken as a block - and the segment builder's `emit` does the per-row block
    // arithmetic.  The one thing that cannot be checked here is the FORMAT's block size (it is not in the
    // index), so `emit` refuses a range that is not a whole number of blocks.
    if (!split) {
        out.codes = r.codes_bytes;
        out.scales = r.scales_bytes;
        out.offset = r.offset_bytes;
        out.dst = r.dst_bytes;
        return true;
    }
    out.axis = ts->axis;
    out.ranges = ts->ranges;
    for (int i = 0; i < ts->ranges; ++i) out.range[i] = ts->range[i];
    out.units = out.axis == ShardAxis::Output ? r.ne1 : r.ne0;
    const int64_t kept = ts->kept();
    if (kept <= 0) {
        err = r.name + ": a split tensor with no kept range";
        return false;
    }
    if (!shard_bytes(r.codes_bytes, out.units, kept, out.codes, r.name + " (codes)", err)) return false;
    if (!shard_bytes(r.scales_bytes, out.units, kept, out.scales, r.name + " (scales)", err)) return false;
    if (!shard_bytes(r.offset_bytes, out.units, kept, out.offset, r.name + " (offset)", err)) return false;
    if (!shard_bytes(r.dst_bytes, out.units, kept, out.dst, r.name + " (total)", err)) return false;
    out.ne0 = out.axis == ShardAxis::Input ? kept : r.ne0;
    out.ne1 = out.axis == ShardAxis::Output ? kept : r.ne1;
    return true;
}

const char* file_name(int id) {
    switch (id) {
        case 0: return "dense.bin";
        case 1: return "embd.bin";
        case 2: return "experts.bin";
        case 3: return "extra.bin";   // plan v0.3 P6: a native pack's float tensors that the base pack quantized
        default: return nullptr;
    }
}

/// UTF-16 path for CreateFile-free portability: the loader uses `std::fopen`, which on Windows takes an ANSI
/// path.  The pack lives beside the executable in practice, but a pack under a user directory with a
/// non-ASCII name would silently fail to open, so the caller gets the errno rather than a null pointer.
bool read_at(std::FILE* f, uint64_t off, void* dst, size_t n, std::string& err, const char* what) {
#if defined(_WIN32)
    if (_fseeki64(f, (long long) off, SEEK_SET) != 0) { err = std::string(what) + ": seek failed"; return false; }
#else
    if (fseeko(f, (off_t) off, SEEK_SET) != 0) { err = std::string(what) + ": seek failed"; return false; }
#endif
    if (std::fread(dst, 1, n, f) != n) { err = std::string(what) + ": short read"; return false; }
    return true;
}

}  // namespace

bool WeightTable::pool_bytes(const std::string& pack_dir, uint64_t& out, std::string& err,
                             const std::set<std::string>* skip, const ShardPlan* shard) {
    const std::string path = pack_dir + "/index.txt";
    std::FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) { err = "cannot open " + path; return false; }
    char line[1024];
    out = 0;
    uint64_t pool = 0, compact = 0;
    int align = 0;
    while (std::fgets(line, sizeof line, f)) {
        if (line[0] == '#') {
            unsigned long long p = 0;
            int a = 0, tensors = 0;
            if (std::sscanf(line, "# align %d pool %llu tensors %d", &a, &p, &tensors) == 3) { pool = p; align = a; }
            continue;
        }
        if (skip == nullptr && shard == nullptr) continue;
        // The shape is parsed here as well as the bytes: a sharded row's size is a fraction of the whole, and
        // which fraction depends on whether the split axis is ne0 or ne1.
        char name[256] = {0};
        unsigned long long src_off = 0, src_bytes = 0, dst_off = 0, dst_bytes = 0;
        long long ne0 = 0, ne1 = 0;
        int file = 0, kind = 0;
        if (std::sscanf(line, "%255s %d %d %llu %llu %llu %llu %lld %lld", name, &file, &kind, &src_off,
                        &src_bytes, &dst_off, &dst_bytes, &ne0, &ne1) != 9)
            continue;
        if (skip != nullptr && skip->count(name)) continue;
        uint64_t bytes = dst_bytes;
        if (shard != nullptr) {
            const auto it = shard->find(name);
            if (it != shard->end()) {
                const TensorShard& ts = it->second;
                if (!ts.keep) continue;                     // a `Rank0` tensor on the rank that does not hold it
                if (ts.axis == ShardAxis::Output || ts.axis == ShardAxis::Input) {
                    const int64_t total = ts.axis == ShardAxis::Output ? ne1 : ne0;
                    std::string e;
                    if (!shard_bytes(dst_bytes, total, ts.kept(), bytes, name, e)) {
                        err = "pool_bytes: " + e;
                        std::fclose(f);
                        return false;
                    }
                }
            }
        }
        const uint64_t a = align > 0 ? (uint64_t) align : 256;
        compact += (bytes + a - 1) / a * a;
    }
    std::fclose(f);
    if (pool == 0) { err = "no '# align ... pool ...' header in " + path; return false; }
    out = (skip != nullptr || shard != nullptr) ? compact : pool;
    return true;
}

bool WeightTable::index_code_bits(const std::string& pack_dir, const std::string& name, int& code_bits,
                                  std::string& err) {
    const std::string path = pack_dir + "/index.txt";
    std::FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) { err = "cannot open " + path; return false; }
    char line[1024];
    code_bits = -1;
    while (std::fgets(line, sizeof line, f)) {
        if (line[0] == '#') continue;
        char row[256] = {0};
        unsigned long long dummy = 0;
        long long ne = 0;
        int i1 = 0, i2 = 0, bits = 0;
        // the first ten fields of the row `load` parses: name file kind src_off src_bytes dst_off dst_bytes ne0 ne1 code_bits
        if (std::sscanf(line, "%255s %d %d %llu %llu %llu %llu %lld %lld %d", row, &i1, &i2, &dummy, &dummy, &dummy,
                        &dummy, &ne, &ne, &bits) != 10)
            continue;
        if (name == row) { code_bits = bits; break; }
    }
    std::fclose(f);
    return true;
}

bool WeightTable::load(const std::string &pack_dir, void *arena_base,
                       uint64_t arena_bytes, std::string &err,
                       const std::set<std::string> *skip, const ShardPlan *shard) try {
    shard_ = shard;
    const std::string path = pack_dir + "/index.txt";
    std::FILE* idx = std::fopen(path.c_str(), "rb");
    if (!idx) { err = "cannot open " + path + " (run tools/pack_index.py)"; return false; }

    uint64_t pool = 0;
    int align = 0;
    std::vector<IndexRow> rows;
    char line[1024];
    while (std::fgets(line, sizeof line, idx)) {
        if (line[0] == '#') {
            unsigned long long p = 0;
            int a = 0, t = 0;
            if (std::sscanf(line, "# align %d pool %llu tensors %d", &a, &p, &t) == 3) { pool = p; align = a; }
            continue;
        }
        IndexRow r;
        char name[256] = {0};
        int kind = 0;
        // The format is written by tools/pack_index.py and is space-separated with no quoting; a name with a
        // space in it would break the parse, which is why the parser REFUSES rather than taking what it got.
        const int n = std::sscanf(line,
                                  "%255s %d %d %llu %llu %llu %llu %lld %lld %d %d %d %d %d %llu %llu %llu %d %d",
                                  name, &r.file, &kind, &r.src_off, &r.src_bytes, &r.dst_off, &r.dst_bytes,
                                  &r.ne0, &r.ne1, &r.code_bits, &r.code_bias, &r.group_elems, &r.codebook,
                                  &r.has_offset, &r.codes_bytes, &r.scales_bytes, &r.offset_bytes,
                                  &r.scales_fp16, &r.act_kind);
        if (n != 19) {
            // A malformed index line is a WRONG BYTE OFFSET waiting to happen, and a wrong offset decodes to
            // a plausible weight.  Refuse loudly instead of loading the rows that happened to parse.
            err = "index.txt: could not parse a row (" + std::to_string(n) + " of 19 fields)";
            std::fclose(idx);
            return false;
        }
        r.name = name;
        // plan v0.3 P6: a standalone native pack (tools/iq_pack.py) stores 16-bit floats as they are in the GGUF:
        // index kinds 4 (BF16) and 5 (F16) are copied and take the engine forms of kinds 1 and 3
        if (kind == 4 || kind == 5) {
            r.raw16 = true;
            kind = kind == 4 ? (int) WeightKind::Bf16InF32 : (int) WeightKind::F16InF32;
        }
        r.kind = (WeightKind) kind;
        rows.push_back(r);
    }
    std::fclose(idx);

    if (pool == 0 || rows.empty()) { err = "index.txt has no header or no rows"; return false; }

    // ---- this rank's footing for every row, worked out BEFORE anything is read ------------------------
    // Whether it holds any bytes, how much of each plane, and the shape it will report.  Doing it here and not
    // inside the copy loop is what lets the compacted arena's offsets be assigned from the SHARDED sizes: a
    // sharded tensor is smaller, so every later row's offset moves up.
    std::vector<RowShard> rsh(rows.size());
    for (size_t i = 0; i < rows.size(); ++i) {
        const TensorShard* ts = nullptr;
        if (shard != nullptr) {
            const auto it = shard->find(rows[i].name);
            if (it != shard->end()) ts = &it->second;
        }
        if (!row_shard(rows[i], ts, rsh[i], err)) return false;
    }

    // Plan v0.3 P1: a skip set compacts the arena.  Kept rows are re-placed in index order at the index's
    // alignment; skipped rows keep their metadata and get no bytes.  A shard plan compacts for the same
    // reason - the rows it cuts are not the size the index records - so both go through this one path.
    std::vector<bool> skipped(rows.size(), false);
    for (size_t i = 0; i < rows.size(); ++i) {
        if (skip != nullptr && skip->count(rows[i].name)) skipped[i] = true;
        if (!rsh[i].keep) skipped[i] = true;
    }
    if (skip != nullptr || shard != nullptr) {
        const uint64_t a = align > 0 ? (uint64_t) align : 256;
        uint64_t at = 0;
        for (size_t i = 0; i < rows.size(); ++i) {
            if (skipped[i]) { rows[i].dst_off = 0; continue; }
            rows[i].dst_off = at;
            at += (rsh[i].dst + a - 1) / a * a;
        }
        pool = at;
    }
    if (arena_bytes < pool) {
        err = "arena is " + std::to_string(arena_bytes) + " B but the index needs " + std::to_string(pool);
        return false;
    }

    // Two pinned staging buffers, both reused.  Pinned because a 5 GB pageable upload spends its time in the
    // driver copying through a bounce buffer, and bounded by CHUNK rather than by the largest tensor.
    //
    // STAGE_OUT IS NOT OPTIONAL.  The first version of this loop converted the values straight into the
    // destination pointer - which is DEVICE memory - from host code.  A host write to a device address is
    // undefined behaviour, and on this driver it does not fault: it corrupts whatever the address maps to in
    // the host's own space, which is nothing at all, so the tensor would have loaded as the UNCONVERTED pack
    // bytes and every BF16 weight would have been read as 32-bit garbage.  Converted data goes to a host
    // buffer and crosses the bus once.
    //
    // STAGE_OUT IS TWICE STAGE_IN because a conversion can now GROW: widening an fp16 scale plane to f32
    // doubles it.  Sized at CHUNK it would have written 8 MiB past the end of a pinned allocation for every
    // one of the 90 widened tensors - a heap corruption that would have been blamed on whatever ran next.
    void* stage_in = nullptr;
    void* stage_out = nullptr;
    /*
    DPCT1048: The original value cudaHostAllocDefault is not meaningful in the
    migrated code and was removed or replaced with 0. You may need to check the
    migrated code.
    */
    if (DPCT_CHECK_ERROR(stage_in = (void *)sycl::malloc_host(
                             CHUNK, dpct::get_in_order_queue())) != 0 ||
        /*
        DPCT1048: The original value cudaHostAllocDefault is not meaningful in
        the migrated code and was removed or replaced with 0. You may need to
        check the migrated code.
        */
        DPCT_CHECK_ERROR(stage_out = (void *)sycl::malloc_host(
                             CHUNK * 2, dpct::get_in_order_queue())) != 0) {
        err = "cudaHostAlloc for the staging buffers failed";
        if (stage_in) sycl::free(stage_in, dpct::get_in_order_queue());
        if (stage_out) sycl::free(stage_out, dpct::get_in_order_queue());
        return false;
    }

    uint8_t* dst_base = (uint8_t*) arena_base;
    std::FILE* cur = nullptr;
    int cur_file = -1;
    table_.clear();
    report_ = LoadReport{};

    const double t_read0 = now_ms();
    double upload_ms = 0;
    for (size_t row_i = 0; row_i < rows.size(); ++row_i) {
        const IndexRow& r = rows[row_i];
        if (skipped[row_i]) {
            WeightRef wr;
            const RowShard& rs = rsh[row_i];
            wr.data = nullptr;
            wr.resident = false;
            wr.bytes = rs.dst;
            wr.ne0 = rs.ne0;
            wr.ne1 = rs.ne1;
            wr.elements = rs.ne0 * (rs.ne1 > 0 ? rs.ne1 : 1);
            wr.kind = r.kind;
            wr.code_bits = r.code_bits;
            wr.code_bias = r.code_bias;
            wr.group_elems = r.group_elems;
            wr.codebook_iq4nl = r.codebook != 0;
            wr.has_offset = r.has_offset != 0;
            wr.codes_bytes = rs.codes;
            wr.scales_bytes = rs.scales;
            wr.offset_bytes = rs.offset;
            wr.src_off = r.src_off;
            wr.src_bytes = r.src_bytes;
            wr.file_id = r.file;
            wr.scales_fp16 = r.scales_fp16 != 0;
            wr.act_kind = r.act_kind;
            table_[r.name] = wr;
            ++report_.tensors;
            continue;
        }
        if (r.code_bits != 0 && r.dst_bytes == 0) {
            // plan v0.3 P6: a native pack's row that carries a shape only - the GGUF form must serve it
            err = r.name + ": this pack holds the tensor only in its GGUF form (run with --native SHARD1)";
            sycl::free(stage_in, dpct::get_in_order_queue());
                sycl::free(stage_out, dpct::get_in_order_queue());
            return false;
        }
        if (r.file != cur_file) {
            if (cur) std::fclose(cur);
            const char* fn = file_name(r.file);
            const std::string p = pack_dir + "/" + (fn ? fn : "?");
            cur = std::fopen(p.c_str(), "rb");
            if (!cur) {
                err = "cannot open " + p;
                sycl::free(stage_in, dpct::get_in_order_queue());
                sycl::free(stage_out, dpct::get_in_order_queue());
                return false;
            }
            cur_file = r.file;
#if !defined(_WIN32)
            // ask for this file's remaining rows up front so their reads overlap
            for (size_t j = row_i; j < rows.size(); ++j)
                if (rows[j].file == r.file && !skipped[j])
                    strata::platform::advise_willneed(fileno(cur), rows[j].src_off, rows[j].src_bytes);
#endif
        }

        // ---- the segment list: what this tensor's bytes are, plane by plane, in both forms
        //
        // PLANE-MAJOR IN THE DESTINATION.  `LayerView`'s plane pointers are prefix sums of
        // codes/scales/offset, so all of a tensor's codes must land before any of its scales.  A sharded row
        // with several ranges (the GDN q|k|v) therefore emits its ranges PLANE BY PLANE and not range by
        // range - the one ordering this list cannot get wrong silently.
        //
        // The byte arithmetic is `plane_bytes / ne1 * rows_kept` per plane, and every division in it was
        // already checked by `row_shard`, which refused a plane that is not uniform per row.
        const RowShard& rs = rsh[row_i];
        std::vector<Seg> segs;
        uint64_t dst_at = 0;
        bool bad = false;
        uint64_t sum_src = 0, sum_dst = 0;
        auto push = [&](uint64_t src_off, uint64_t src_bytes, uint64_t dst_bytes, Conv c) {
            segs.push_back(Seg{src_off, src_bytes, dst_at, dst_bytes, c});
            dst_at += dst_bytes;
            sum_src += src_bytes;
            sum_dst += dst_bytes;
        };
        // The row's source planes, in the order pack_index.py writes them.
        const uint64_t src_scales = r.scales_fp16 ? r.scales_bytes / 2 : r.scales_bytes;
        if (r.codes_bytes != 0 && r.scales_fp16 && (r.scales_bytes % 2 || src_scales * 2 != r.scales_bytes)) {
            err = r.name + ": an fp16 scale plane of an odd byte count";
            bad = true;
        }
        // This rank's ranges, or the single whole-range "range" of an unsharded tensor.
        ShardRange ranges[kShardMaxRanges];
        int n_ranges = 1;
        ranges[0] = ShardRange{0, r.ne1 > 0 ? r.ne1 : 1};
        const bool input_axis = !bad && rs.axis == ShardAxis::Input;
        if (!bad && (rs.axis == ShardAxis::Output || rs.axis == ShardAxis::Input)) {
            n_ranges = rs.ranges;
            for (int i = 0; i < rs.ranges; ++i) ranges[i] = rs.range[i];
        }
        const uint64_t codes_per_row = r.ne1 > 0 ? r.codes_bytes / (uint64_t) r.ne1 : r.codes_bytes;
        const uint64_t src_scales_per_row = r.ne1 > 0 ? src_scales / (uint64_t) r.ne1 : src_scales;
        const uint64_t dst_scales_per_row = r.ne1 > 0 ? r.scales_bytes / (uint64_t) r.ne1 : r.scales_bytes;
        const uint64_t offset_per_row = r.ne1 > 0 ? r.offset_bytes / (uint64_t) r.ne1 : r.offset_bytes;
        // A CONTRACTION split keeps part of every row, so its runs are per row and its offsets are per unit of
        // the contraction axis.  `group_elems` is the format's block (0 on a plain tensor, which makes every
        // division below degenerate to one element and the range check to a no-op) - so one `emit` serves the
        // quantized planes and the single plain run alike.
        const int64_t units = r.group_elems > 0 ? r.group_elems : 1;
        const int64_t rows_full = r.ne1 > 0 ? r.ne1 : 1;
        const int64_t blocks_full = r.ne0 > 0 ? r.ne0 / units : 0;
        auto emit = [&](uint64_t plane_src, uint64_t src_per_row, uint64_t dst_per_row, uint64_t src_block,
                        uint64_t dst_block, Conv c) -> bool {
            if (!input_axis) {
                for (int i = 0; i < n_ranges; ++i)
                    push(plane_src + (uint64_t) ranges[i].begin * src_per_row,
                         (uint64_t) ranges[i].count() * src_per_row,
                         (uint64_t) ranges[i].count() * dst_per_row, c);
                return true;
            }
            if (blocks_full <= 0 || src_block == 0 || (dst_block == 0 && dst_per_row != 0)) {
                err = r.name + ": a contraction split of a plane whose block size is not whole";
                return false;
            }
            for (int64_t row = 0; row < rows_full; ++row) {
                for (int i = 0; i < n_ranges; ++i) {
                    const ShardRange& rr = ranges[i];
                    if (rr.count() <= 0 || rr.begin < 0 || rr.end > r.ne0 || rr.begin % units != 0 ||
                        rr.count() % units != 0) {
                        err = r.name + ": a kept contraction range that is not a whole number of " +
                              std::to_string(units) + "-element units inside " + std::to_string(r.ne0);
                        return false;
                    }
                    const uint64_t b0 = (uint64_t) (rr.begin / units);
                    const uint64_t nb = (uint64_t) (rr.count() / units);
                    push(plane_src + (uint64_t) row * src_per_row + b0 * src_block, nb * src_block,
                         nb * dst_block, c);
                }
            }
            return true;
        };

        if (r.codes_bytes != 0) {
            // A quantized tensor: codes, then scales, then the optional offset plane, each contiguous in the
            // source span in that order (pack_index.py verifies the contiguity and refuses otherwise).
            // `row_shard` has already refused a QUANTIZED contraction split, so `input_axis` cannot be set
            // here; the block arithmetic below is what a future version of that case would reuse.
            const uint64_t codes_block = blocks_full > 0 ? codes_per_row / (uint64_t) blocks_full : 0;
            if (!bad) bad = !emit(0, codes_per_row, codes_per_row, codes_block, codes_block, Conv::Copy);
            if (!bad && r.scales_bytes) {
                const uint64_t sb = blocks_full > 0 ? src_scales_per_row / (uint64_t) blocks_full : 0;
                const uint64_t db = blocks_full > 0 ? dst_scales_per_row / (uint64_t) blocks_full : 0;
                bad = !emit(r.codes_bytes, src_scales_per_row, dst_scales_per_row, sb, db,
                            r.scales_fp16 ? Conv::WidenF16 : Conv::Copy);
            }
            if (!bad && r.offset_bytes) {
                const uint64_t ob = blocks_full > 0 ? offset_per_row / (uint64_t) blocks_full : 0;
                bad = !emit(r.codes_bytes + src_scales, offset_per_row, offset_per_row, ob, ob, Conv::Copy);
            }
        } else {
            // Everything else is one run whose transform is the row's kind.  The two promoted types shrink
            // 4 B/elem to 2; the rest are already the engine form.
            Conv c = Conv::Copy;
            if (r.raw16) c = Conv::Copy;
            else if (r.kind == WeightKind::Bf16InF32) c = Conv::Bf16High;
            else if (r.kind == WeightKind::F16InF32) c = Conv::ToF16;
            const uint64_t elems = (uint64_t) r.ne0 * (uint64_t) (r.ne1 > 0 ? r.ne1 : 1);
            if (c != Conv::Copy && (r.src_bytes != elems * 4 || r.dst_bytes != elems * 2)) {
                // A promoted type is 4 B/elem in and 2 B/elem out, and nothing else.
                char buf[320];
                std::snprintf(buf, sizeof buf,
                              "%s: a promoted tensor of %llu elements is %llu B in and %llu B out, not %llu/%llu",
                              r.name.c_str(), (unsigned long long) elems, (unsigned long long) r.src_bytes,
                              (unsigned long long) r.dst_bytes, (unsigned long long) (elems * 4),
                              (unsigned long long) (elems * 2));
                err = buf;
                bad = true;
            }
            const uint64_t src_per_row = r.ne1 > 0 ? r.src_bytes / (uint64_t) r.ne1 : r.src_bytes;
            const uint64_t dst_per_row = r.ne1 > 0 ? r.dst_bytes / (uint64_t) r.ne1 : r.dst_bytes;
            // A plain tensor has no quant block (`units` is 1 and `blocks_full` is ne0), so a contraction split
            // of it is one run per row at element granularity - which for a 1-D vector (`ssm_a`, `ssm_dt.bias`)
            // is simply the first half of the bytes.
            const uint64_t src_unit = blocks_full > 0 ? src_per_row / (uint64_t) blocks_full : 0;
            const uint64_t dst_unit = blocks_full > 0 ? dst_per_row / (uint64_t) blocks_full : 0;
            if (!bad) bad = !emit(0, src_per_row, dst_per_row, src_unit, dst_unit, c);
        }
        // The segment list is this loop's own idea of the layout, and it is the ONLY thing standing between a
        // mis-sized plane and a wrong byte offset.  It is checked against the totals, not trusted: a sharded
        // row's destination total is what `row_shard` sized the arena for, and its source runs have to stay
        // inside the span the index records.
        if (!bad && (sum_dst != rs.dst || sum_src > r.src_bytes)) {
            char buf[420];
            std::snprintf(buf, sizeof buf,
                          "%s: the segments cover %llu B of a %llu B destination (and %llu B of a %llu B source) "
                          "- the layout this loader built is not the one the index describes",
                          r.name.c_str(), (unsigned long long) sum_dst, (unsigned long long) rs.dst,
                          (unsigned long long) sum_src, (unsigned long long) r.src_bytes);
            err = buf;
            bad = true;
        }
        for (const Seg& s : segs) {
            const bool widening = (s.conv == Conv::WidenF16);
            if (widening && (s.src_bytes & 1ull)) {
                err = r.name + ": an fp16 plane with an odd source byte count";
                sycl::free(stage_in, dpct::get_in_order_queue());
                    sycl::free(stage_out, dpct::get_in_order_queue());
                return false;
            }
            uint64_t done = 0;
            while (done < s.src_bytes) {
                uint64_t n = std::min(CHUNK, s.src_bytes - done);
                // WIDENING NEVER SPLITS AN ELEMENT.  CHUNK is even, so a full chunk cannot, but the tail of a
                // plane whose length is odd would - and half an fp16 is a plausible-looking scale.
                if (widening && (n & 1ull)) {
                    if (n == 1) {
                        err =
                            r.name + ": an fp16 plane ending on a half element";
                        sycl::free(stage_in, dpct::get_in_order_queue());
                        sycl::free(stage_out, dpct::get_in_order_queue());
                        return false;
                    }
                    n -= 1;
                }
                if (!read_at(cur, r.src_off + s.src_off + done, stage_in, (size_t) n, err, r.name.c_str())) {
                    sycl::free(stage_in, dpct::get_in_order_queue());
                        sycl::free(stage_out, dpct::get_in_order_queue());
                    return false;
                }
                const uint8_t* src = (const uint8_t*) stage_in;
                const void* host_src = src;
                uint64_t out_bytes = n;
                uint64_t out_at = s.dst_off + done;

                switch (s.conv) {
                    case Conv::Copy:
                        break;
                    case Conv::Bf16High: {
                        // THE HIGH 16 BITS ARE THE BF16 ENCODING, exactly - not a rounding.  A bf16 value is
                        // the top half of its f32 image, and the pack promoted it without changing it, so
                        // this is a copy with a stride.
                        const uint64_t elems = n / 4;
                        out_bytes = elems * 2;
                        uint8_t* o = (uint8_t*) stage_out;
                        for (uint64_t i = 0; i < elems; ++i) {
                            uint32_t v;
                            std::memcpy(&v, src + i * 4, 4);
                            const uint16_t h = (uint16_t) (v >> 16);
                            std::memcpy(o + i * 2, &h, 2);
                        }
                        host_src = stage_out;
                        out_at = s.dst_off + done / 2;
                        break;
                    }
                    case Conv::ToF16: {
                        const uint64_t elems = n / 4;
                        out_bytes = elems * 2;
                        uint8_t* o = (uint8_t*) stage_out;
                        for (uint64_t i = 0; i < elems; ++i) {
                            float v;
                            std::memcpy(&v, src + i * 4, 4);
                            const uint16_t h = strata::kernels::f16_from_f32(v);
                            std::memcpy(o + i * 2, &h, 2);
                        }
                        host_src = stage_out;
                        out_at = s.dst_off + done / 2;
                        break;
                    }
                    case Conv::WidenF16: {
                        // fp16 bits -> f32.  EXACT: every fp16 value is an f32 value, including the
                        // subnormals, so this cannot change a single weight - which is why the engine may do
                        // it at all, and why doing it costs 10.88 MiB rather than a second kernel family.
                        const uint64_t elems = n / 2;
                        out_bytes = elems * 4;
                        float* o = (float*) stage_out;
                        for (uint64_t i = 0; i < elems; ++i) {
                            uint16_t h;
                            std::memcpy(&h, src + i * 2, 2);
                            o[i] = strata::kernels::f32_from_f16(h);
                        }
                        host_src = stage_out;
                        out_at = s.dst_off + done * 2;
                        break;
                    }
                }

                const double u0 = now_ms();
                /*
                DPCT1114: cudaMemcpy is migrated to asynchronization
                memcpy, assuming in the original code the source host memory is
                pageable memory. If the memory is not pageable, call wait() on
                event return by memcpy API to ensure synchronization behavior.
                */
                if (DPCT_CHECK_ERROR(dpct::get_in_order_queue().memcpy(
                        dst_base + r.dst_off + out_at, host_src, out_bytes).wait()) !=
                    0) {
                    err = "cudaMemcpy failed for " + r.name;
                    sycl::free(stage_in, dpct::get_in_order_queue());
                        sycl::free(stage_out, dpct::get_in_order_queue());
                    return false;
                }
                upload_ms += now_ms() - u0;
                done += n;
            }
        }

        WeightRef wr;
        wr.data = dst_base + r.dst_off;
        wr.bytes = rs.dst;
        wr.ne0 = rs.ne0;
        wr.ne1 = rs.ne1;
        wr.elements = rs.ne0 * (rs.ne1 > 0 ? rs.ne1 : 1);
        wr.kind = r.kind;
        wr.code_bits = r.code_bits;
        wr.code_bias = r.code_bias;
        wr.group_elems = r.group_elems;
        wr.codebook_iq4nl = r.codebook != 0;
        wr.has_offset = r.has_offset != 0;
        wr.codes_bytes = rs.codes;
        wr.scales_bytes = rs.scales;
        wr.offset_bytes = rs.offset;
        wr.src_off = r.src_off;
        wr.src_bytes = r.src_bytes;
        wr.file_id = r.file;
        wr.scales_fp16 = r.scales_fp16 != 0;
        wr.act_kind = r.act_kind;
        table_[r.name] = wr;

        ++report_.tensors;
        if ((r.kind == WeightKind::Bf16InF32 || r.kind == WeightKind::F16InF32) && !r.raw16 &&
            rs.axis == ShardAxis::Replicated) {
            ++report_.re_rounded;
            report_.bytes_saved += r.src_bytes - r.dst_bytes;
        }
    }
    if (cur) std::fclose(cur);
    sycl::free(stage_in, dpct::get_in_order_queue());
    sycl::free(stage_out, dpct::get_in_order_queue());

    report_.arena_bytes = pool;
    report_.read_ms = now_ms() - t_read0 - upload_ms;
    report_.upload_ms = upload_ms;
    if (DPCT_CHECK_ERROR(dpct::get_current_device().queues_wait_and_throw()) !=
        0) {
        err = "cudaDeviceSynchronize after the load failed"; return false;
    }
    return true;
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

const WeightRef* WeightTable::find(const std::string& name) const {
    const auto it = table_.find(name);
    return it == table_.end() ? nullptr : &it->second;
}

}  // namespace strata::core
