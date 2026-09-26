// src/kernels/ngram.cpp - P2.S4: the PLE n-gram hash and the table read (IQ4_NL or Q8_0).
//
// See include/strata/kernels/ngram.hpp for the semantics, the rival readings, and the note on MADV_RANDOM.
#include "strata/kernels/ngram.hpp"
#include "strata/artifact/gguf_reader.hpp"
#include "strata/kernels/f16_bits.hpp"
#include "strata/ngram/ple_reader.hpp"
#include "strata/platform/direct_file.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <thread>

#include <cstdio>
#include <cstring>
#include <vector>
#include <stdexcept>

#if defined(_WIN32)
// `PrefetchVirtualMemory` (memoryapi.h, Windows 8+) is the whole point of the change in `gather` below.
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#endif

namespace strata::kernels {

namespace {
/// The A/B arm.  Host-token-path only, so it needs no atomics; see the note on `ple_prefetch_enable`.
bool g_ple_prefetch = true;
}  // namespace

void ple_prefetch_enable(bool on) { g_ple_prefetch = on; }
bool ple_prefetch_enabled() { return g_ple_prefetch; }

PleConsts ple_artifact_consts() {
    // docs/gguf-dump-shard1.txt, verbatim.  Written once here rather than derived, because `head_offsets` is
    // ALSO in the metadata as its own array - and deriving one from the other would hide a mismatch between
    // them instead of surfacing it.  The parity test checks the two agree.
    PleConsts c{};
    c.mult[0] = 23703573157769ull;
    c.mult[1] = 20109073645365ull;
    c.mult[2] = 8052911324071ull;
    const uint64_t vocab[PLE_N_HEADS] = {
        20000003, 20000023, 20000033, 20000047, 20000059, 20000063, 20000069, 20000077,
        20000081, 20000093, 20000107, 20000147, 20000153, 20000159, 20000161, 20000171};
    const uint64_t offset[PLE_N_HEADS] = {
        0,        20000003, 40000026, 60000059, 80000106, 100000165, 120000228, 140000297,
        160000374, 180000455, 200000548, 220000655, 240000802, 260000955, 280001114, 300001275};
    for (int i = 0; i < PLE_N_HEADS; ++i) {
        c.vocab[i] = vocab[i];
        c.offset[i] = offset[i];
    }
    return c;
}

uint64_t ngram_mixed(const int64_t* ctx, const uint64_t* mult, int n) {
    // The first term is an ASSIGNMENT and the rest are XORed into it, which is how the source writes it
    // (`uint64_t mixed = ctx[0]*m[0]; for j=1.. mixed ^= ctx[j]*m[j];`).  Every product wraps mod 2^64,
    // which is what the `(uint64_t)` casts in the source make explicit.
    uint64_t mixed = (uint64_t) ctx[0] * mult[0];
    for (int j = 1; j < n; ++j) mixed ^= (uint64_t) ctx[j] * mult[j];
    return mixed;
}

void ngram_rows(const int32_t* tokens, const int32_t* prev, int n_tokens, const PleConsts& c, uint32_t* out) {
    const int n_prev = NGRAM_SIZE - 1;
    for (int i = 0; i < n_tokens; ++i) {
        int64_t ctx[NGRAM_SIZE];
        ctx[0] = tokens[i];
        bool cut = false;
        for (int s = 1; s < NGRAM_SIZE; ++s) {
            // `prev` is OLDEST FIRST, so predecessor `s` positions back is entry (n_prev - s): s=1 reads the
            // NEWEST.  Reading index (s-1) instead walks the window backwards, which still produces indices
            // in range and so cannot be caught by a range check - only by an oracle.
            const int32_t t = cut ? TOKEN_NULL : prev[i * n_prev + (n_prev - s)];
            // The cut is evaluated BEFORE the value is stored, so the position whose predecessor was EOS is
            // itself EOS.  Storing first and then cutting would leave position s holding the real token while
            // position s+1 became EOS - one token of history too much.
            cut = cut || t < 0 || t == PLE_EOS_TOKEN_ID;
            ctx[s] = cut ? PLE_EOS_TOKEN_ID : t;
        }
        for (int n = 2; n <= NGRAM_SIZE; ++n) {
            const uint64_t mixed = ngram_mixed(ctx, c.mult, n);
            const int base = (n - 2) * HEADS_PER_NGRAM;
            for (int g = 0; g < HEADS_PER_NGRAM; ++g) {
                const int h = base + g;
                out[i * PLE_N_HEADS + h] = (uint32_t) (mixed % c.vocab[h] + c.offset[h]);
            }
        }
    }
}

namespace {
const int8_t kIq4Nl[16] = {-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113};
}

int iq4nl_code(int code) { return kIq4Nl[code & 15]; }

void iq4nl_dequant_row(const uint8_t* row, float* out160) {
    for (int b = 0; b < PLE_HEAD_DIM / 32; ++b) {
        const uint8_t* blk = row + (size_t) b * 18;
        uint16_t dbits;
        std::memcpy(&dbits, blk, 2);
        const float d = f32_from_f16(dbits);
        const uint8_t* qs = blk + 2;
        // SPLIT HALVES: qs[j] holds elements j and j+16, not 2j and 2j+1.
        for (int j = 0; j < 16; ++j) {
            out160[b * 32 + j] = d * (float) kIq4Nl[qs[j] & 0x0F];
            out160[b * 32 + j + 16] = d * (float) kIq4Nl[qs[j] >> 4];
        }
    }
}

void ple_dequant_row(int type, const uint8_t* row, float* out160) {
    if (type != 8) {
        iq4nl_dequant_row(row, out160);
        return;
    }
    for (int b = 0; b < PLE_HEAD_DIM / 32; ++b) {                     // Q8_0: fp16 scale + 32 int8
        const uint8_t* blk = row + (size_t) b * 34;
        uint16_t dbits;
        std::memcpy(&dbits, blk, 2);
        const float d = f32_from_f16(dbits);
        for (int j = 0; j < 32; ++j) out160[b * 32 + j] = d * (float) (int8_t) blk[2 + j];
    }
}

std::string ple_table_shard(const std::vector<std::string>& shards) {
    for (const auto& path : shards) {
        try {
            if (GgufFile(path).find("per_layer_token_embd.weight") != nullptr) return path;
        } catch (const std::exception&) {
        }
    }
    return {};
}

// ---------------------------------------------------------------------------------------------------
struct PleTable::Impl {
    GgufFile* file = nullptr;
    const uint8_t* data = nullptr;
    uint64_t n_rows = 0;
    int type = 20;                        // IQ4_NL or Q8_0
    uint32_t row_bytes = PLE_ROW_BYTES;
    mutable uint64_t bytes_read = 0;
    // Direct mode (plan v0.3 P2): the mapping above is released after the header parse and every row comes
    // from an unbuffered SSD read into `raw`.
    PleIo mode = PleIo::Mmap;
    strata::ngram::PleReader reader;
    strata::ngram::PleReader::Ticket ticket;
    bool pending = false;
    uint32_t rows[PLE_N_HEADS] = {};
    uint8_t raw[PLE_N_HEADS * PLE_ROW_BYTES_Q8_0] = {};
    // Ram mode: the table read into `ram_base` by `loader`; `ram` (the table's first row) valid once `ram_ready`
    std::string path;
    uint64_t table_offset = 0;
    uint8_t* ram_base = nullptr;
    uint64_t ram_bytes = 0;
    const uint8_t* ram = nullptr;
    std::atomic<bool> ram_ready{false}, cancel{false};
    std::thread loader;
    std::string ram_note;
    bool pending_ram = false;   // the pending token's rows come from RAM
    // the read-ahead slots: a token's rows in flight from the SSD (`via_reader`), else read at collect
    struct Ahead {
        bool pending = false, via_reader = false;
        uint32_t rows[PLE_N_HEADS] = {};
        uint8_t raw[PLE_N_HEADS * PLE_ROW_BYTES_Q8_0] = {};
        strata::ngram::PleReader::Ticket ticket;
    };
    Ahead aheads[PleTable::kAheadSlots];

    bool from_ram() const { return mode == PleIo::Ram && ram_ready.load(std::memory_order_acquire); }
    bool direct() const { return (mode == PleIo::Direct || mode == PleIo::Ram) && !from_ram(); }
    void load();
};

// The whole table in 8 MiB unbuffered reads, four in flight.  Pageable: locking it resident first (VirtualLock of
// 28.8 GB) faults in its pages under the process's working-set lock and stalled the startup's other threads ~8 s.
void PleTable::Impl::load() {
    using strata::platform::DirectFile;
    const auto t0 = std::chrono::steady_clock::now();
    const uint64_t a0 = table_offset & ~(uint64_t) (DirectFile::alignment() - 1);
    const uint64_t need = table_offset + n_rows * (uint64_t) row_bytes - a0;
    DirectFile f;
    std::string err;
    if (!f.open(path, err)) { ram_note = "PLE table in RAM: " + err; std::fprintf(stderr, "strata: %s\n", ram_note.c_str()); return; }
    constexpr uint32_t kChunk = 8u << 20;
    constexpr int kDepth = 4;
    uint64_t next = 0, got = 0;
    int inflight = 0;
    bool ok = true;
    while (ok && !cancel.load() && (next < need || inflight > 0)) {
        while (ok && inflight < kDepth && next < need) {
            const uint32_t len = (uint32_t) std::min<uint64_t>(kChunk, ram_bytes - next);
            ok = f.submit(a0 + next, ram_base + next, len, next, err);
            next += len;
            ++inflight;
        }
        strata::platform::Completion c[kDepth];
        const int k = ok ? f.wait(c, kDepth, -1) : 0;
        for (int i = 0; i < k; ++i) {
            if (c[i].tag == DirectFile::WAKE_TAG) continue;
            --inflight;
            ok = ok && c[i].ok;
            got += c[i].bytes;
        }
    }
    while (inflight > 0) {   // a failed or cancelled load still collects what it queued
        strata::platform::Completion c[kDepth];
        const int k = f.wait(c, kDepth, -1);
        for (int i = 0; i < k; ++i) if (c[i].tag != DirectFile::WAKE_TAG) --inflight;
    }
    if (cancel.load()) return;
    if (!ok || got < need) {
        ram_note = "PLE table in RAM: the load failed" + (err.empty() ? std::string() : ": " + err) + "; rows stay on the SSD";
        std::fprintf(stderr, "strata: %s\n", ram_note.c_str());
        return;
    }
    ram = ram_base + (table_offset - a0);
    const double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    char buf[120];
    std::snprintf(buf, sizeof buf, "PLE table in RAM: %.1f GB in %.1f s", (double) need / 1e9, s);
    ram_note = buf;
    ram_ready.store(true, std::memory_order_release);
    std::fprintf(stderr, "strata: %s\n", ram_note.c_str());
}

PleTable::PleTable() : impl_(new Impl) {}
PleTable::~PleTable() { close(); delete impl_; }

bool PleTable::open(const std::string& gguf_path, std::string& err) {
    return open(gguf_path, err, PleIoOptions{});
}

bool PleTable::open(const std::string& gguf_path, std::string& err, const PleIoOptions& io) {
    close();
    try {
        impl_->file = new GgufFile(gguf_path);
    } catch (const std::exception& e) {
        err = e.what();
        return false;
    }
    const TensorInfo* t = impl_->file->find("per_layer_token_embd.weight");
    if (t == nullptr) {
        err = "per_layer_token_embd.weight is not in " + gguf_path;
        close();
        return false;
    }
    // [160, 320001536]: ne0 = 160 is the FAST axis, so the ROW index is shape[1] and a row is contiguous.
    if (t->shape.size() != 2 || t->shape[0] != (uint64_t) PLE_HEAD_DIM) {
        err = "per_layer_token_embd.weight has an unexpected shape";
        close();
        return false;
    }
    if (t->type != 20 && t->type != 8) {
        err = std::string("per_layer_token_embd.weight is ") + t->type_name() + ", not IQ4_NL or Q8_0";
        close();
        return false;
    }
    impl_->type = (int) t->type;
    impl_->row_bytes = t->type == 8 ? PLE_ROW_BYTES_Q8_0 : PLE_ROW_BYTES;
    impl_->n_rows = t->shape[1];
    impl_->data = impl_->file->tensor_data(*t);

    // THE CHECK THAT MAKES THE OFFSET FALSIFIABLE.  The manifest's `shard2_tensor.offset` is 0, but that is
    // the offset within the GGUF's DATA SECTION: the file's first 192 bytes are a header, and reading at 0
    // would decode the header plus 192 bytes of shifted rows - still plausible IQ4_NL, and wrong for every
    // row.  `GgufFile` parses the header, so `tensor_data` is already correct; this asserts the tensor
    // exactly fills its span - up to the next tensor or the end of the file, less the alignment padding -
    // which is what makes the whole arrangement checkable rather than assumed.  A wrong data offset would
    // leave a different remainder.  (The table fills the ISTA shard 2 alone; the unsloth shard 2 holds other
    // tensors too.)
    const uint64_t need = impl_->n_rows * (uint64_t) impl_->row_bytes;
    uint64_t end = impl_->file->file_size() - impl_->file->data_start();
    for (const auto& o : impl_->file->tensors())
        if (o.offset > t->offset && o.offset < end) end = o.offset;
    const uint64_t have = end - t->offset;
    if (need > have || have - need >= impl_->file->alignment()) {
        char buf[256];
        std::snprintf(buf, sizeof buf,
                      "PLE table size mismatch: %llu rows x %d B = %llu, but its span in the file is %llu B "
                      "(data_start %llu)",
                      (unsigned long long) impl_->n_rows, (int) impl_->row_bytes, (unsigned long long) need,
                      (unsigned long long) have, (unsigned long long) impl_->file->data_start());
        err = buf;
        close();
        return false;
    }
    if (io.mode == PleIo::Direct || io.mode == PleIo::Ram) {
        // The parse above is the validated source of the offset; the mapping itself is not kept, so no page of
        // the table can enter this process's working set or the file cache through it.
        const uint64_t table_offset = impl_->file->data_start() + t->offset;
        impl_->path = gguf_path;
        impl_->table_offset = table_offset;
        const uint64_t n_rows = impl_->n_rows;
        delete impl_->file;
        impl_->file = nullptr;
        impl_->data = nullptr;
        if (!impl_->reader.open(gguf_path, table_offset, n_rows, io.max_inflight, io.cache_rows, err, io.io_thread,
                                impl_->row_bytes)) {
            close();
            return false;
        }
        impl_->n_rows = n_rows;
    }
    impl_->mode = io.mode;
    return true;
}

void PleTable::close() {
    if (impl_->loader.joinable()) {
        impl_->cancel = true;
        impl_->loader.join();
    }
    impl_->ram_ready = false;
    impl_->cancel = false;
    impl_->ram = nullptr;
    if (impl_->ram_base != nullptr) {
        strata::platform::DirectFile::free_aligned(impl_->ram_base);
        impl_->ram_base = nullptr;
    }
    impl_->reader.close();
    for (auto& a : impl_->aheads) a.pending = false;
    impl_->pending = false;
    impl_->mode = PleIo::Mmap;
    delete impl_->file;
    impl_->file = nullptr;
    impl_->data = nullptr;
    impl_->n_rows = 0;
}

bool PleTable::is_open() const { return impl_->data != nullptr || impl_->reader.is_open(); }

bool PleTable::start_ram_load(std::string& err) {
    using strata::platform::DirectFile;
    if (impl_->mode != PleIo::Ram || impl_->ram_base != nullptr) return true;
    const uint64_t a0 = impl_->table_offset & ~(uint64_t) (DirectFile::alignment() - 1);
    const uint64_t end = impl_->table_offset + impl_->n_rows * (uint64_t) impl_->row_bytes;
    impl_->ram_bytes = (end - a0 + DirectFile::alignment() - 1) / DirectFile::alignment() * DirectFile::alignment();
    impl_->ram_base = (uint8_t*) DirectFile::alloc_aligned((size_t) impl_->ram_bytes);
    if (impl_->ram_base == nullptr) {
        err = "PLE table in RAM: " + std::to_string(impl_->ram_bytes >> 20) + " MiB could not be allocated";
        return false;
    }
    impl_->loader = std::thread([m = impl_] { m->load(); });
    return true;
}

bool PleTable::ram_ready() const { return impl_->from_ram(); }
PleIo PleTable::mode() const { return impl_->mode; }
uint64_t PleTable::rows() const { return impl_->n_rows; }
uint64_t PleTable::bytes_read() const { return impl_->bytes_read; }

void PleTable::read_row(uint32_t row, float* out160) const {
    if (impl_->from_ram()) {
        if (row >= impl_->n_rows) std::memset(out160, 0, (size_t) PLE_HEAD_DIM * sizeof(float));
        else ple_dequant_row(impl_->type, impl_->ram + (size_t) row * impl_->row_bytes, out160);
        impl_->bytes_read += impl_->row_bytes;
        return;
    }
    if (impl_->direct() && impl_->reader.is_open()) {
        uint8_t raw[PLE_ROW_BYTES_Q8_0];
        std::string err;
        const auto t = impl_->reader.issue(&row, 1, raw);
        if (!impl_->reader.collect(t, err)) {
            std::memset(out160, 0, (size_t) PLE_HEAD_DIM * sizeof(float));
            return;
        }
        ple_dequant_row(impl_->type, raw, out160);
        impl_->bytes_read += impl_->row_bytes;
        return;
    }
    if (impl_->data == nullptr || row >= impl_->n_rows) {
        std::memset(out160, 0, (size_t) PLE_HEAD_DIM * sizeof(float));
        return;
    }
    ple_dequant_row(impl_->type, impl_->data + (size_t) row * impl_->row_bytes, out160);
    impl_->bytes_read += impl_->row_bytes;
}

bool PleTable::issue(const uint32_t* rows16) {
    std::memcpy(impl_->rows, rows16, sizeof impl_->rows);
    if (impl_->mode != PleIo::Mmap) {
        if (impl_->pending) return false;              // one token in flight per table
        impl_->pending_ram = impl_->from_ram();
        if (!impl_->pending_ram) impl_->ticket = impl_->reader.issue(impl_->rows, PLE_N_HEADS, impl_->raw);
        impl_->pending = true;
        return true;
    }
#if defined(_WIN32)
    if (impl_->data != nullptr && g_ple_prefetch) {
        WIN32_MEMORY_RANGE_ENTRY ranges[PLE_N_HEADS];
        ULONG_PTR n = 0;
        for (int h = 0; h < PLE_N_HEADS; ++h) {
            if (rows16[h] >= impl_->n_rows) continue;
            ranges[n].VirtualAddress = (PVOID) (impl_->data + (size_t) rows16[h] * impl_->row_bytes);
            ranges[n].NumberOfBytes = impl_->row_bytes;
            ++n;
        }
        if (n > 0) (void) PrefetchVirtualMemory(GetCurrentProcess(), n, ranges, 0);
    }
#endif
    impl_->pending = true;
    return true;
}

bool PleTable::collect(float* out2560, std::string& err) {
    if (!impl_->pending) { err = "PleTable::collect without issue"; return false; }
    impl_->pending = false;
    if (impl_->pending_ram) {
        for (int h = 0; h < PLE_N_HEADS; ++h) read_row(impl_->rows[h], out2560 + (size_t) h * PLE_HEAD_DIM);
        return true;
    }
    if (impl_->mode != PleIo::Mmap) {
        if (!impl_->reader.collect(impl_->ticket, err)) return false;
        for (int h = 0; h < PLE_N_HEADS; ++h)
            ple_dequant_row(impl_->type, impl_->raw + (size_t) h * impl_->row_bytes, out2560 + (size_t) h * PLE_HEAD_DIM);
        impl_->bytes_read += (uint64_t) PLE_N_HEADS * impl_->row_bytes;
        return true;
    }
    for (int h = 0; h < PLE_N_HEADS; ++h) read_row(impl_->rows[h], out2560 + (size_t) h * PLE_HEAD_DIM);
    return true;
}

bool PleTable::gather_batch(const uint32_t* rows, size_t n_tokens, float* out, std::string& err) {
    if (impl_->pending) { err = "PleTable::gather_batch while a token is in flight"; return false; }
    const size_t n = n_tokens * (size_t) PLE_N_HEADS;
    if (impl_->direct()) {
        std::vector<uint8_t> raw(n * impl_->row_bytes);
        const auto ticket = impl_->reader.issue(rows, n, raw.data());
        if (!impl_->reader.collect(ticket, err)) return false;
        for (size_t i = 0; i < n; ++i)
            ple_dequant_row(impl_->type, raw.data() + i * impl_->row_bytes, out + i * PLE_HEAD_DIM);
        impl_->bytes_read += (uint64_t) n * impl_->row_bytes;
        return true;
    }
    for (size_t i = 0; i < n; ++i) read_row(rows[i], out + i * PLE_HEAD_DIM);
    return true;
}

void PleTable::ahead(int slot, const uint32_t* rows16) {
    if (slot < 0 || slot >= kAheadSlots) return;
    Impl::Ahead& a = impl_->aheads[slot];
    if (a.pending && a.via_reader) {   // the slot's earlier rows (a draft the window left out): long since read
        std::string err;
        (void) impl_->reader.collect(a.ticket, err);
    }
    std::memcpy(a.rows, rows16, sizeof a.rows);
    a.via_reader = impl_->direct() && impl_->reader.is_open();
    if (a.via_reader) a.ticket = impl_->reader.issue(a.rows, PLE_N_HEADS, a.raw);
    a.pending = true;
}

bool PleTable::ahead_holds(int slot, const uint32_t* rows16) const {
    if (slot < 0 || slot >= kAheadSlots) return false;
    const Impl::Ahead& a = impl_->aheads[slot];
    return a.pending && std::memcmp(a.rows, rows16, sizeof a.rows) == 0;
}

bool PleTable::ahead_collect(int slot, float* out2560, std::string& err) {
    if (slot < 0 || slot >= kAheadSlots || !impl_->aheads[slot].pending) {
        err = "PleTable::ahead_collect: slot " + std::to_string(slot) + " holds no rows";
        return false;
    }
    Impl::Ahead& a = impl_->aheads[slot];
    a.pending = false;
    if (!a.via_reader) {
        for (int h = 0; h < PLE_N_HEADS; ++h) read_row(a.rows[h], out2560 + (size_t) h * PLE_HEAD_DIM);
        return true;
    }
    if (!impl_->reader.collect(a.ticket, err)) return false;
    for (int h = 0; h < PLE_N_HEADS; ++h)
        ple_dequant_row(impl_->type, a.raw + (size_t) h * impl_->row_bytes, out2560 + (size_t) h * PLE_HEAD_DIM);
    impl_->bytes_read += (uint64_t) PLE_N_HEADS * impl_->row_bytes;
    return true;
}

void PleTable::set_injected_delay_us(double us) { impl_->reader.set_injected_delay_us(us); }

std::string PleTable::io_report() const {
    if (impl_->from_ram()) return "ple io: " + impl_->ram_note;
    if (impl_->mode == PleIo::Mmap || !impl_->reader.is_open()) return {};
    const strata::ngram::ReaderStats& s = impl_->reader.stats();
    char buf[320];
    std::snprintf(buf, sizeof buf,
                  "ple io: %llu rows, %.1f%% row-cache hits, %llu SSD reads (%.1f MB), read p50 %.0f us p99 %.0f us, "
                  "blocked %.3f ms total (submit %.3f ms), cache %llu/%llu rows",
                  (unsigned long long) s.requests, s.requests ? 100.0 * (double) s.cache_hits / (double) s.requests : 0.0,
                  (unsigned long long) s.reads, (double) s.bytes / 1e6, s.percentile(0.5), s.percentile(0.99),
                  s.wait_us / 1000.0, s.submit_us / 1000.0, (unsigned long long) impl_->reader.cache_size(),
                  (unsigned long long) impl_->reader.cache_capacity());
    return buf;
}

void PleTable::gather(const uint32_t* rows16, float* out2560) const {
    if (impl_->from_ram()) {
        for (int h = 0; h < PLE_N_HEADS; ++h) read_row(rows16[h], out2560 + (size_t) h * PLE_HEAD_DIM);
        return;
    }
    if (impl_->direct()) {
        // `gather` stays const for its existing callers; the reader's state is the table's I/O state.
        PleTable* self = const_cast<PleTable*>(this);
        std::string err;
        if (!self->issue(rows16) || !self->collect(out2560, err)) {
            std::fprintf(stderr, "PleTable::gather: %s\n", err.empty() ? "a token is already in flight" : err.c_str());
            std::memset(out2560, 0, (size_t) NG_N_EMBD * sizeof(float));
        }
        return;
    }
    // ================================ SIXTEEN SERIAL PAGE FAULTS, MEASURED ================================
    //
    // **THIS COST 2.10-2.61 ms PER TOKEN AND HAD NEVER BEEN IN THE PLAN'S BUDGET AT ALL.**  The round-309
    // `token host phases` line put it second behind the layer loop among avoidable terms, and the arithmetic
    // says why: the table is 320,001,536 rows of `PLE_ROW_BYTES` = 90 B in a 26.8 GB mapping, so the sixteen
    // rows a token needs are 1,440 B - **0.5 MB/s**.  That is not bandwidth, it is latency: sixteen reads into
    // sixteen different 4 KB pages scattered across 26.8 GB, taken ONE AT A TIME, and on this machine the PLE
    // shard is 26.8 GB against 63 GB of RAM that the 31.6 GB expert arena is also competing for, so they are
    // not in the OS cache.  Sixteen serial NVMe reads at ~150 us is 2.4 ms, which is the measurement.
    //
    // `PrefetchVirtualMemory` issues all sixteen in ONE call and lets them complete in parallel.  It is a hint
    // and cannot change the answer - a range it does not fetch is simply faulted in by the read that follows -
    // so the only risk is that it does nothing.
#if defined(_WIN32)
    if (impl_->data != nullptr && g_ple_prefetch) {
        WIN32_MEMORY_RANGE_ENTRY ranges[PLE_N_HEADS];
        ULONG_PTR n = 0;
        for (int h = 0; h < PLE_N_HEADS; ++h) {
            // Out-of-range rows are handled by `read_row` as zeros and have no address to prefetch.
            if (rows16[h] >= impl_->n_rows) continue;
            ranges[n].VirtualAddress = (PVOID) (impl_->data + (size_t) rows16[h] * impl_->row_bytes);
            ranges[n].NumberOfBytes = impl_->row_bytes;
            ++n;
        }
        if (n > 0) (void) PrefetchVirtualMemory(GetCurrentProcess(), n, ranges, 0);
    }
#endif
    // HEAD-SLOWEST, which is what `ggml_get_rows` does and what the source's own comment says: head h's 160
    // values occupy [h*160, (h+1)*160).  A head-fastest layout would put element (d, h) at d*16 + h and needs
    // a real transpose - a reshape of the same flat buffer compares equal and would make the check vacuous.
    for (int h = 0; h < PLE_N_HEADS; ++h) read_row(rows16[h], out2560 + (size_t) h * PLE_HEAD_DIM);
}

}  // namespace strata::kernels
