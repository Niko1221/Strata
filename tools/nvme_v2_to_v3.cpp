// tools/nvme_v2_to_v3.cpp - convert a v2 NVMe snapshot to the v3 envelope, in place via rename.
//
// Why this exists: the v3 format (this branch, step 3) refuses v2 files outright - an older snapshot is never
// reinterpreted, because the two formats disagree about WHERE the running state comes from (v2 read dead and
// block_pos off the live device at dump time, which at a turn boundary is not the state at L).  That is the
// right refuse-don't-convert rule for the ENGINE, but it left 200 GB of stored snapshots unreachable.  The
// bytes themselves, however, are the same payload laid out with one formula difference: v2 wrote
// min(L/idx_block + 2, ...) pooled rows where v3 writes L/idx_block + 1 - one spare row per QSA layer that no
// reader can reach (qsa.cu:253 gates b > n_bid) - plus a wider header carrying the shared core's geometry key.
// So the conversion is mechanical, and every step is checkable:
//
//   1. the source must pass ITS OWN integrity footer (FNV-1a over the v2 payload);
//   2. the whole v2 layout must be explained to the byte by the header fields plus ONE unknown, the gdn+ple
//      prefix block, which is solved from the file's own size and then validated: gdn is computable from the
//      geometry key (ssm_state_size^2 * ssm_v_heads + ssm_conv_channels * (ssm_d_conv-1)) * n_gdn * 4, and the
//      only allowed variation is the PLE block (9 * 10240 * 4 bytes) for sessions with no PLE history;
//   3. the payload is copied verbatim EXCEPT the last pooled row of every QSA layer, which is dropped;
//   4. the 18-field geometry key is copied verbatim from a reference v3 file dumped by the same engine build
//      and pack (--geometry), and the v2 header's own eight derived fields must agree with it;
//   5. the output's size arithmetic and digest are re-derived and checked before the rename.
//
// The original file is NOT deleted by this tool: the caller renames it aside (same filesystem, so a rename is
// free) and keeps it until the converted file has been functionally verified.  No CUDA, no engine, no model.
#include <cstdint>
#include <cstdio>
#include <unistd.h>
#include <cstring>
#include <string>
#include <vector>

#include "strata/kernels/ngram.hpp"   // NG_HIST, NG_HC_DIM: the PLE block's size

namespace {

constexpr uint32_t kMagic = 0x5E564D45;          // "^VME"
constexpr int64_t kV2Header = 104, kV3Header = 208, kFooter = 8;
constexpr int64_t kPleBytes = (int64_t) strata::kernels::NG_HIST * strata::kernels::NG_HC_DIM * 4;
// The MTP drafter's per-page row bytes are NOT the main KV's (its arrays are narrower than kv_host_arrays
// suggests for the main layers), and the drafter's max_cells is not in either header.  Both were FITTED, not
// guessed: across all 106 store files (L 8.8k..134k) and four freshly dumped v3 references (L 12..58513) the
// solved prefix block is EXACTLY gdn + ple (118,038,528 B) with zero spread when the drafter term is
// mtp_host * ceil(L/page) * n_head_kv * page * 132.  The converter never trusts the fit alone: the final walk
// must land on both footers to the byte, and a file whose drafter was clamped by a smaller max_cells than any
// file seen here would fail that check and be refused, not mis-converted.
constexpr int64_t kMtpRowBytes = 132;

uint64_t fnv1a(const uint8_t* p, size_t n, uint64_t h = 1469598103934665603ull) {
    for (size_t i = 0; i < n; ++i) { h ^= p[i]; h *= 1099511628211ull; }
    return h;
}

int64_t rd64(const uint8_t* p) { int64_t v; std::memcpy(&v, p, 8); return v; }
int32_t rd32(const uint8_t* p) { int32_t v; std::memcpy(&v, p, 4); return v; }
void wr64(uint8_t* p, int64_t v) { std::memcpy(p, &v, 8); }
void wr32(uint8_t* p, int32_t v) { std::memcpy(p, &v, 4); }

struct Fail { std::string reason; };

// per-layer host KV bytes for one page row, from the same arithmetic kv_host_arrays uses (int8: k, v, k_scale,
// v_scale; f16: k, v; q4 is refused - this store was written by --kv int8)
int64_t kv_row_bytes(int64_t head_dim, int32_t kv_format) {
    if (kv_format == 1) return 2 * head_dim + 2 * ((head_dim / 64) * 2);   // int8 + fp16 scale per 64 channels
    if (kv_format == 0) return 2 * head_dim * 2;
    return -1;                                                             // q4: kv_q4_bytes_per_head, refused here
}

struct Layout {
    int64_t L = 0, n_imgs = 0, n_qsa = 0, n_head_kv = 0, head_dim = 0, idx_dim = 0, page_size = 0,
            idx_block = 0, max_cells = 0, mtp_host = 0;
    int32_t kv_format = 0;
    int64_t prefix_C = 0;        // gdn (+ ple when the session had a PLE history): solved, then validated
};

// every segment size, from the header fields alone - the same walk nvme_restore does, with v2's pooled formula
int64_t payload_v2(const Layout& y, int64_t prefix_C) {
    const int64_t n_pages = (y.L + y.page_size - 1) / y.page_size;
    const int64_t kv_layer = n_pages * y.n_head_kv * y.page_size * kv_row_bytes(y.head_dim, y.kv_format);
    const int64_t pooled_v2 = y.L / y.idx_block + 2;                  // v2 wrote the spare row and one more
    const int64_t tail = (y.idx_block - 1) * y.idx_dim * 4, dead = y.idx_dim * 4;
    const int64_t mL = y.L < y.max_cells ? y.L : y.max_cells;
    const int64_t mp = (mL + y.page_size - 1) / y.page_size;
    const int64_t mtp = y.mtp_host * mp * y.n_head_kv * y.page_size * kMtpRowBytes;
    return 4 * y.L + 16 * y.n_imgs + prefix_C +
           y.n_qsa * (kv_layer + pooled_v2 * y.idx_dim * 4 + tail + dead + 4) + mtp;
}

// the v3 payload: same walk, one pooled row less per layer
int64_t payload_v3(const Layout& y, int64_t prefix_C) {
    return payload_v2(y, prefix_C) - y.n_qsa * y.idx_dim * 4;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 4) {
        std::fprintf(stderr,
            "usage: %s --geometry <reference-v3-file> <v2-file> <out-file>\n"
            "       converts ONE snapshot; the caller renames the original aside and verifies the result\n",
            argv[0]);
        return 2;
    }
    try {
        // ---- the reference v3 file: the geometry key comes from the engine that will read the output ----
        std::vector<uint8_t> ref;
        { FILE* f = std::fopen(argv[2], "rb"); if (!f) throw Fail{std::string("open reference: ") + argv[2]};
          std::fseek(f, 0, SEEK_END); long n = std::ftell(f); std::fseek(f, 0, SEEK_SET);
          ref.resize((size_t) n);
          if (std::fread(ref.data(), 1, ref.size(), f) != ref.size()) { std::fclose(f); throw Fail{"read reference"}; }
          std::fclose(f); }
        if ((int64_t) ref.size() < kV3Header || rd32(ref.data()) != kMagic || rd32(ref.data() + 4) != 3)
            throw Fail{"the reference is not a v3 snapshot"};
        const uint8_t* refgeom = ref.data() + 32;

        // ---- the source ----
        std::vector<uint8_t> in;
        { FILE* f = std::fopen(argv[3], "rb"); if (!f) throw Fail{std::string("open source: ") + argv[3]};
          std::fseek(f, 0, SEEK_END); long n = std::ftell(f); std::fseek(f, 0, SEEK_SET);
          in.resize((size_t) n);
          if (std::fread(in.data(), 1, in.size(), f) != in.size()) { std::fclose(f); throw Fail{"read source"}; }
          std::fclose(f); }
        if ((int64_t) in.size() < kV2Header + kFooter) throw Fail{"file too small for a v2 snapshot"};
        if (rd32(in.data()) != kMagic) throw Fail{"bad magic"};
        if (rd32(in.data() + 4) != 2) throw Fail{"not a v2 file"};
        const uint8_t* g = in.data() + 32;   // v2's eight derived fields, in header order
        Layout y;
        y.L = rd64(in.data() + 8), y.n_imgs = rd64(in.data() + 16), y.kv_format = rd32(in.data() + 28);
        y.n_qsa = rd64(g), y.n_head_kv = rd64(g + 16), y.head_dim = rd64(g + 24), y.idx_dim = rd64(g + 32),
        y.page_size = rd64(g + 40), y.idx_block = rd64(g + 48), y.max_cells = rd64(g + 56),
        y.mtp_host = rd64(in.data() + 96);
        if (y.L < 1 || y.n_imgs < 0 || y.n_qsa < 1 || y.head_dim < 1 || y.idx_dim < 1 || y.page_size < 1 ||
            y.idx_block < 1 || y.max_cells < 1 || y.mtp_host < 0)
            throw Fail{"implausible header fields"};
        if (y.kv_format != 1) throw Fail{"only int8 stores are converted (this store was written with --kv int8)"};
        if (y.L > y.max_cells) throw Fail{"L exceeds max_cells - the pooled-row arithmetic would not hold"};

        // ---- integrity of the SOURCE, before anything else ----
        const int64_t body = (int64_t) in.size() - kV2Header - kFooter;
        const uint64_t got = fnv1a(in.data() + kV2Header, (size_t) body);
        uint64_t want; std::memcpy(&want, in.data() + in.size() - kFooter, 8);
        if (got != want) throw Fail{"source fails its own integrity footer - refusing to convert a corrupt file"};

        // ---- solve the one unknown (gdn [+ ple]) from this file's own size, then validate it ----
        const int64_t C = (int64_t) in.size() - kV2Header - kFooter - payload_v2(y, 0);
        if (C < 0) throw Fail{"layout does not fit: the file is smaller than its own header arithmetic"};
        // EMPIRICAL validation, not an analytic guess: solve the reference file's own prefix block the same way
        // (v3 arithmetic, prefix 0) and require this file's C to equal it, or differ by exactly the PLE block
        // (a session with no PLE history checkpoints no ple blob - the dump falls back to the live array only
        // when ss.ple_hist is set, so the segment is present or absent as a whole).
        { Layout r; r.L = rd64(ref.data() + 8); r.n_imgs = rd64(ref.data() + 16); r.kv_format = rd32(ref.data() + 28);
          // the key's order is [n_embd, n_layers, qsa_interval, ssm_state_size, ssm_k_heads, ssm_v_heads,
          // ssm_d_conv, ssm_conv_channels, ssm_value_dim, n_head, n_head_kv, head_dim, idx_q_heads,
          // idx_key_dim, hc, hc_lr, n_expert, n_ff] (conversation_state.cpp) - n_qsa is n_layers/qsa_interval
          r.n_qsa = rd64(refgeom + 8) / rd64(refgeom + 16); r.n_head_kv = rd64(refgeom + 80);
          r.head_dim = rd64(refgeom + 88); r.idx_dim = rd64(refgeom + 104); r.page_size = rd64(ref.data() + 176); r.idx_block = rd64(ref.data() + 184);
          r.max_cells = rd64(ref.data() + 192); r.mtp_host = rd64(ref.data() + 200);
          const int64_t C_ref = (int64_t) ref.size() - kV3Header - kFooter - payload_v3(r, 0);
          if (C_ref < 0) throw Fail{"the reference file does not fit its own v3 arithmetic"};
          const int64_t d = C - C_ref;
          if (d != 0 && d != kPleBytes && d != -kPleBytes)
              throw Fail{"the solved prefix block (" + std::to_string(C) + " B) is not the reference's (" +
                         std::to_string(C_ref) + " B) within one PLE block (" + std::to_string(kPleBytes) +
                         " B) - the layout hypothesis is wrong for this file"}; }
        y.prefix_C = C;

        // ---- the v2 header's MODEL-identity shapes must agree with the reference geometry ----
        // max_cells is deliberately NOT compared: it is a runtime shape of the engine that dumped the file
        // (it scales with --max-context), the restore accepts any file whose max_cells does not EXCEED the
        // live engine's, and this store was written by a 262144-context server while the reference here was
        // dumped at 131072.  The file's own max_cells is carried through unchanged.
        if (y.page_size != rd64(ref.data() + 176) || y.idx_block != rd64(ref.data() + 184))
            throw Fail{"page_size / idx_block disagree with the reference geometry"};

        // ---- build the v3 file ----
        const int64_t n_pages = (y.L + y.page_size - 1) / y.page_size;
        const int64_t kv_layer = n_pages * y.n_head_kv * y.page_size * kv_row_bytes(y.head_dim, y.kv_format);
        const int64_t pooled_v2 = y.L / y.idx_block + 2, pooled_v3 = y.L / y.idx_block + 1;
        const int64_t tail = (y.idx_block - 1) * y.idx_dim * 4, dead = y.idx_dim * 4;
        const int64_t per_layer_fixed = tail + dead + 4;
        const int64_t mL = y.L < y.max_cells ? y.L : y.max_cells;
        const int64_t mp = (mL + y.page_size - 1) / y.page_size;
        const int64_t mtp = y.mtp_host * mp * y.n_head_kv * y.page_size * kMtpRowBytes;
        const int64_t out_size = kV3Header + payload_v3(y, C) + kFooter;

        std::vector<uint8_t> out((size_t) out_size, 0);
        // header: same leading fields, the shared core's geometry key, the runtime shapes beside it
        std::memcpy(out.data(), in.data(), 32);                       // magic, version, L, n_imgs, cvec, kv_format
        wr32(out.data() + 4, 3);
        std::memcpy(out.data() + 32, refgeom, 144);                   // the 18-field key, verbatim
        wr64(out.data() + 176, y.page_size); wr64(out.data() + 184, y.idx_block);
        wr64(out.data() + 192, y.max_cells); wr64(out.data() + 200, y.mtp_host);
        // payload: ids + image records + the gdn/ple prefix block, verbatim
        const int64_t ids_imgs = 4 * y.L + 16 * y.n_imgs;
        std::memcpy(out.data() + kV3Header, in.data() + kV2Header, (size_t) (ids_imgs + C));
        // per QSA layer: the KV arrays verbatim, pooled WITHOUT its last row, tail/dead/block_pos verbatim
        int64_t roff = kV2Header + ids_imgs + C, woff = kV3Header + ids_imgs + C;
        for (int64_t i = 0; i < y.n_qsa; ++i) {
            std::memcpy(out.data() + woff, in.data() + roff, (size_t) kv_layer);
            roff += kv_layer; woff += kv_layer;
            const int64_t pooled_bytes_v2 = pooled_v2 * y.idx_dim * 4;
            std::memcpy(out.data() + woff, in.data() + roff, (size_t) (pooled_bytes_v2 - y.idx_dim * 4));
            roff += pooled_bytes_v2; woff += pooled_bytes_v2 - y.idx_dim * 4;
            std::memcpy(out.data() + woff, in.data() + roff, (size_t) per_layer_fixed);
            roff += per_layer_fixed; woff += per_layer_fixed;
        }
        std::memcpy(out.data() + woff, in.data() + roff, (size_t) mtp);
        roff += mtp; woff += mtp;
        if (roff != (int64_t) in.size() - kFooter || woff != out_size - kFooter)
            throw Fail{"the walk did not land on both footers - internal arithmetic error"};
        const uint64_t digest = fnv1a(out.data() + kV3Header, (size_t) (out_size - kV3Header - kFooter));
        std::memcpy(out.data() + out_size - kFooter, &digest, 8);

        FILE* f = std::fopen(argv[4], "wb");
        if (!f) throw Fail{std::string("open output: ") + argv[4]};
        if (std::fwrite(out.data(), 1, out.size(), f) != out.size() || std::fflush(f) != 0) {
            std::fclose(f); throw Fail{"write output"};
        }
        std::fflush(f); ::fsync(::fileno(f)); std::fclose(f);
        std::printf("ok %s: L=%lld imgs=%lld prefix=%lldB pooled rows %lld -> %lld, %lld -> %lld bytes\n",
                    argv[4], (long long) y.L, (long long) y.n_imgs, (long long) C,
                    (long long) pooled_v2, (long long) pooled_v3, (long long) in.size(), (long long) out_size);
        return 0;
    } catch (const Fail& e) {
        std::fprintf(stderr, "refuse %s: %s\n", argv[3], e.reason.c_str());
        return 1;
    }
}
