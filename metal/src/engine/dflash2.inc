// metal/src/engine/dflash2.inc - the DFlash2 drafter for Qwen3.8-Flash-Next on the Metal window (included at the end of window.cpp, inside
// namespace strata::metal_engine, like prompt_chain.inc).
//
// The engine's other backends run core::Dflash2Drafter (src/core/dflash2.cpp: CUDA graphs over native_q8_0_mmvq).  The Metal engine has its own
// decode window, so the same class is implemented here with the same public API (generate.cpp's `--draft dflash2 --draft-model PACK` loop is
// unchanged) and the same DF2P pack (tools/dflash2_pack.py: Q8_0 blocks, so the engine's gemv_8_* kernels read the pack as it is):
//
//   taps       the window / prompt chunk writes, after each tapped layer, the mean of the hc streams of the layer's completed output into a
//              unified-memory buffer (df2_tap, dflash2.metal); Verifier::set_taps and Prefill::set_taps reach them through window_taps /
//              prompt_taps below
//   context    taps -> fc (12800 -> 2560) -> hidden_norm -> per layer K (norm + rope) and V into the ring (slot pos % 2064)
//   draft      the same + the block: 8 rows, 5 layers (dynamic 2-tap conv, windowed attention over the ring + the block's keys), the head over
//              the draft vocabulary, the top-16 and the selector's walk.  One command buffer per round.
//
// Activations are float (the CUDA drafter quantizes them to Q8_1): the Q8_0 weights are read by the engine's own GEMVs on xprep'd rows.


// ---- taps (Window members)
bool Window::taps_enable(const std::vector<int>& layers, std::string& err) {
    if (taps_on_) {
        if (layers != tap_layers_) { err = "metal taps: already on for other layers"; return false; }
        return true;
    }
    if (layers.empty()) { err = "metal taps: no layers"; return false; }
    for (int i = 0; i < 64; ++i) tap_slot_[i] = -1;
    for (size_t i = 0; i < layers.size(); ++i) {
        if (layers[i] < 0 || layers[i] >= 47) { err = "metal taps: layer " + std::to_string(layers[i]) + " is out of range"; return false; }
        tap_slot_[layers[i]] = (int) i;
    }
    ldf2_ = cx_->compile({strata_metal::resource("STRATA_METAL_ENGINE_KERNELS", "engine-msl", STRATA_METAL_ENGINE_KERNELS_DEFAULT) + "/dflash2.metal"},
                         "#define NO_GGML_COMMON 1\n");
    tap_layers_ = layers;
    tap_dec_ = cx_->alloc((size_t) MAXT * layers.size() * N * 4);
    wcache_.clear();   // the windows encoded so far have no tap dispatches
    taps_on_ = true;
    std::fprintf(stderr, "strata metal: DFlash2 taps on (%zu layers)\n", layers.size());
    return true;
}

void Window::tap_dispatch(std::vector<mtl::Dispatch>& ds, int l, const mtl::Buffer& R, const mtl::Buffer& bo, const mtl::Buffer& inj,
                          const mtl::Buffer& out, int rows) {
    const size_t nt = tap_layers_.size();
    struct TapArgs { uint32_t n, hc, R_stride, bo_stride, inj_stride, out_stride, out_off; } a{
        (uint32_t) N, (uint32_t) HC, (uint32_t) (HC * N), (uint32_t) N, (uint32_t) HC, (uint32_t) (nt * N), (uint32_t) (tap_slot_[l] * N)};
    mtl::Dispatch d;
    d.p = P(ldf2_, "df2_tap", {}, "df2_tap");
    d.args = {mtl::Arg::B(R), mtl::Arg::B(bo), mtl::Arg::B(inj), mtl::Arg::B(out), mtl::Arg::V(a)};
    d.grid[0] = N / 256;
    d.grid[1] = rows;
    d.tg[0] = 256;
    push(ds, d);
}

// hands the rows of a window / chunk that are at or after the first wanted position to the callback
bool Window::tap_deliver(const mtl::Buffer& buf, int rows, int64_t pos0, std::string& err) {
    if (!taps_on_ || !tap_cb_) return true;
    const int64_t t0 = std::max<int64_t>(0, tap_first_ - pos0);
    if (t0 >= rows) return true;
    return tap_cb_(buf.as<float>() + (size_t) t0 * tap_layers_.size() * N, rows - t0, pos0 + t0, err);
}

mtl::Buffer& Window::tap_pro_buf(int rows) {
    if (tap_pro_rows_ < rows) {
        if (tap_pro_.handle) cx_->free(tap_pro_);
        tap_pro_ = cx_->alloc((size_t) rows * tap_layers_.size() * N * 4);
        tap_pro_rows_ = rows;
    }
    return tap_pro_;
}

// ---- the drafter
namespace {
constexpr int DF2_RING = 2064, DF2_ROWS = 8, DF2_DRAFTS = 7, DF2_CTX_MAX = 2048, DF2_TOPB = 64, DF2_MAXCH = 33;
}

struct Window::Df2 {
    // the pack
    std::string path;
    mtl::Buffer pack;
    std::map<std::string, gguf::Tensor> ten;
    uint64_t total = 0;
    int L = 5, NH = 32, NKV = 8, I = 7680, window = 2048, group = 16, rank = 256, mask_token = 0;
    float theta = 10000.0f, eps = 1e-6f;
    std::vector<int> taps;
    // the head over the draft vocabulary
    int64_t nd = 0;
    const mtl::Buffer* head_buf = nullptr;
    gguf::Tensor head_t;
    mtl::Buffer own_head, vmap;
    bool use_map = false;
    // state and activations
    mtl::Buffer kring[8], vring[8];
    mtl::Buffer ctxpos, rtctx, dpos, rtd, bpos, rtb, anchor;
    mtl::Buffer fcy, hctx, kc, vc, h, xn, xc, dyn, q, qn, k, v, kblk, ao, o, gate, up, sw, mlp, fn, hsel, logits, topval, toptok, tkv, tki, part, outbuf;
    Act A_tap, A_h, A_x, A_ao, A_sw, A_fn;
    std::vector<float> mask_emb;
    std::vector<mtl::Dispatch> list[DF2_ROWS + 1];
    bool built[DF2_ROWS + 1] = {};
    bool loaded = false, bound = false;
    uint64_t bytes = 0;
    double gpu_ms = 0;
    int64_t rounds = 0;
};

namespace {
std::string g_df2_path;

// reads the DF2P header + index; fills the config and the tensor table (offsets are from the end of the index)
bool df2_read_index(const std::string& path, std::vector<int>& taps, std::map<std::string, gguf::Tensor>& ten, uint64_t& total,
                    uint64_t& data_off, std::map<std::string, std::string>& cfg, std::string& err) {
    std::FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) { err = "dflash2: cannot open " + path; return false; }
    struct Closer { std::FILE* f; ~Closer() { std::fclose(f); } } closer{f};
    uint32_t hdr[3];
    if (std::fread(hdr, 4, 3, f) != 3 || std::memcmp(hdr, "DF2P", 4) != 0 || hdr[1] != 1) { err = "dflash2: " + path + " is not a DF2P v1 pack"; return false; }
    std::string index(hdr[2], ' ');
    if (std::fread(index.data(), 1, hdr[2], f) != hdr[2]) { err = "dflash2: the pack's index is cut short"; return false; }
    data_off = 12 + hdr[2];
    total = 0;
    for (size_t i = 0; i < index.size();) {
        const size_t j = index.find('\n', i);
        if (j == std::string::npos) break;
        const std::string line = index.substr(i, j - i);
        i = j + 1;
        char a[256], b[64], c[256];
        long long v1, v2, v3, v4;
        if (std::sscanf(line.c_str(), "cfg %63s %255s", b, c) == 2) {
            cfg[b] = c;
            if (std::string(b) == "taps") {
                taps.clear();
                for (const char* q = c; *q;) { taps.push_back(std::atoi(q)); while (*q && *q != ',') ++q; if (*q == ',') ++q; }
            }
        } else if (std::sscanf(line.c_str(), "t %255s %63s %lld %lld %lld %lld", a, b, &v1, &v2, &v3, &v4) == 6) {
            gguf::Tensor t;
            t.name = a;
            const std::string dt = b;
            t.type = dt == "q8_0" ? ref::Q8_0 : dt == "f32" ? ref::F32 : dt == "bf16" ? ref::BF16 : -1;
            t.shape = {(uint64_t) v2, (uint64_t) v1};   // ggml order: [in][out]
            t.offset = (uint64_t) v3;
            t.bytes = (uint64_t) v4;
            ten[t.name] = t;
            total = std::max(total, t.offset + t.bytes);
        }
    }
    if (ten.empty()) { err = "dflash2: the pack has no tensors"; return false; }
    return true;
}
}  // namespace

bool Window::df2_open(const std::string& path, std::string& err) {
    std::unique_ptr<Df2> d(new Df2());
    d->path = path;
    uint64_t data_off = 0;
    std::map<std::string, std::string> cfg;
    if (!df2_read_index(path, d->taps, d->ten, d->total, data_off, cfg, err)) return false;
    auto geti = [&](const char* k, int def) { auto it = cfg.find(k); return it == cfg.end() ? def : std::atoi(it->second.c_str()); };
    d->L = geti("layers", 5); d->NH = geti("heads", 32); d->NKV = geti("kv_heads", 8); d->I = geti("inter", 7680);
    d->window = geti("window", 2048); d->group = geti("conv_group", 16); d->rank = geti("rank", 256); d->mask_token = geti("mask_token", 0);
    if (cfg.count("rope_theta")) d->theta = (float) std::atof(cfg["rope_theta"].c_str());
    if (cfg.count("eps")) d->eps = (float) std::atof(cfg["eps"].c_str());
    if (geti("hidden", 0) != N || d->NH != 32 || d->NKV != 8 || geti("head_dim", 0) != 128 || d->rank != 256 || d->L > 8 || d->window + 16 > DF2_RING ||
        d->I != 7680 || d->group != 16 || d->taps.empty()) {
        err = "dflash2: the pack's shape is not the Flash-Next drafter this engine was written for";
        return false;
    }
    df2_ = d.release();
    return true;
}

bool Window::df2_load_pack(std::string& err) {
    Df2& d = *df2_;
    if (d.loaded) return true;
    uint64_t data_off = 0;
    std::vector<int> taps;
    std::map<std::string, gguf::Tensor> ten;
    std::map<std::string, std::string> cfg;
    uint64_t total = 0;
    if (!df2_read_index(d.path, taps, ten, total, data_off, cfg, err)) return false;
    d.pack = cx_->alloc(((size_t) total + 4095) / 4096 * 4096 + 4096);
    if (!d.pack.handle) { err = "dflash2: the pack does not fit in memory"; return false; }
    std::FILE* f = std::fopen(d.path.c_str(), "rb");
    if (!f) { err = "dflash2: cannot open " + d.path; return false; }
    fseeko(f, (off_t) data_off, SEEK_SET);
    const size_t got = std::fread(d.pack.ptr, 1, (size_t) total, f);
    std::fclose(f);
    if (got != (size_t) total) { err = "dflash2: the pack is shorter than its index"; return false; }
    d.bytes += d.pack.bytes;
    d.loaded = true;
    return true;
}

bool Window::df2_bind(const std::string& vocab_file, std::string& err) {
    if (df2_ == nullptr) { err = "dflash2: no pack"; return false; }
    Df2& d = *df2_;
    if (d.bound) return true;
    if (!taps_on_) { err = "dflash2: the window keeps no taps"; return false; }
    if (!df2_load_pack(err)) return false;
    const double t0 = now_ms();
    auto A = [&](size_t bytes) { d.bytes += bytes; return cx_->alloc(bytes); };
    // ---- the draft vocabulary's head rows: the MTP's when it is the same list, else gathered here
    std::vector<int32_t> vocab;
    if (!vocab_file.empty() && vocab_file != "full") {
        if (std::FILE* vf = std::fopen(vocab_file.c_str(), "rb")) {
            int32_t id;
            while (std::fread(&id, 4, 1, vf) == 1) if (id >= 0 && id < n_vocab_) vocab.push_back(id);
            std::fclose(vf);
        }
    }
    if (vocab.empty() && vocab_file != "full") vocab = dvocab_;   // the MTP's list (may be empty: the whole head)
    const gguf::Tensor& out_t = T("output.weight");
    if (!vocab.empty()) {
        if (mhead_.handle && vocab == dvocab_) {
            d.head_buf = &mhead_;
            d.head_t = mhead_t_;
        } else {
            const size_t rb = ref::row_bytes(out_t.type, N);
            d.own_head = A(vocab.size() * rb + 4096);
            for (size_t i = 0; i < vocab.size(); ++i)
                std::memcpy(d.own_head.as<uint8_t>() + i * rb, (const uint8_t*) W_.ptr + out_t.offset + (size_t) vocab[i] * rb, rb);
            d.head_buf = &d.own_head;
            d.head_t = out_t;
            d.head_t.offset = 0;
            d.head_t.shape = {(uint64_t) N, (uint64_t) vocab.size()};
            d.head_t.bytes = vocab.size() * rb;
        }
        d.nd = (int64_t) vocab.size();
        d.vmap = A(4 * vocab.size());
        std::memcpy(d.vmap.ptr, vocab.data(), 4 * vocab.size());
        d.use_map = true;
    } else {
        d.head_buf = &W_;
        d.head_t = out_t;
        d.nd = n_vocab_;
        d.vmap = A(16);
        d.use_map = false;
    }
    if ((d.nd + DF2_TOPB - 1) / DF2_TOPB > 4096) { err = "dflash2: the head has too many rows for the top-k"; return false; }
    // ---- the mask token's embedding (constant)
    {
        const gguf::Tensor& te = T("token_embd.weight");
        const size_t rb = ref::row_bytes(te.type, N);
        d.mask_emb.resize(N);
        ref::dequant(te.type, (const uint8_t*) W_.ptr + te.offset + (size_t) d.mask_token * rb, d.mask_emb.data(), N);
    }
    // ---- buffers
    for (int l = 0; l < d.L; ++l) { d.kring[l] = A((size_t) DF2_RING * 8 * 128 * 4); d.vring[l] = A((size_t) DF2_RING * 8 * 128 * 4); }
    const size_t B = DF2_ROWS;
    d.ctxpos = A(4 * (size_t) DF2_CTX_MAX); d.rtctx = A(4 * (size_t) DF2_CTX_MAX * 128);
    d.dpos = A(4 * 8); d.rtd = A(4 * 8 * 128); d.bpos = A(4 * 8); d.rtb = A(4 * 8 * 128); d.anchor = A(4 * 8);
    d.fcy = A(4 * B * N); d.hctx = A(4 * B * N); d.kc = A(4 * B * 1024); d.vc = A(4 * B * 1024);
    d.h = A(4 * B * N); d.xn = A(4 * B * N); d.xc = A(4 * B * N); d.dyn = A(4 * B * 640);
    d.q = A(4 * B * 4096); d.qn = A(4 * B * 4096); d.k = A(4 * B * 1024); d.v = A(4 * B * 1024); d.kblk = A(4 * B * 1024);
    d.ao = A(4 * B * 4096); d.o = A(4 * B * N); d.gate = A(4 * B * d.I); d.up = A(4 * B * d.I); d.sw = A(4 * B * d.I);
    d.mlp = A(4 * B * N); d.fn = A(4 * (B + 1) * N); d.hsel = A(4 * DF2_DRAFTS * 256);
    d.logits = A(4 * (size_t) DF2_DRAFTS * (size_t) d.nd);
    d.topval = A(4 * DF2_DRAFTS * 16); d.toptok = A(4 * DF2_DRAFTS * 16);
    d.tkv = A(4 * (size_t) DF2_DRAFTS * DF2_TOPB * 16); d.tki = A(4 * (size_t) DF2_DRAFTS * DF2_TOPB * 16);
    d.part = A(4 * (size_t) DF2_MAXCH * 8 * 32 * 130);
    d.outbuf = A(4 * 32);
    auto act = [&](int n) { Act x; x.n = n; x.b = A(4 * ((size_t) 4 * MAXT * n + (size_t) MAXT * n / 16)); return x; };
    d.A_tap = act((int) d.taps.size() * N); d.A_h = act(N); d.A_x = act(N); d.A_ao = act(4096); d.A_sw = act(d.I); d.A_fn = act(N);
    d.bound = true;
    std::fprintf(stderr, "strata metal: DFlash2 drafter ready (%.0f MiB: pack %.0f, %lld-token draft head %s), in %.0f ms\n", d.bytes / 1048576.0,
                 d.pack.bytes / 1048576.0, (long long) d.nd, d.use_map ? "" : "(the whole head)", now_ms() - t0);
    return true;
}

namespace {
// the rope table rows of `pos` (cos then sin of pos * theta^(-2i/128), the angle reduced in double): [rows][2][64]
void df2_rope_rows(const int32_t* pos, int rows, float theta, float* table) {
    constexpr double two_pi = 6.283185307179586476925286766559;
    for (int r = 0; r < rows; ++r)
        for (int i = 0; i < 64; ++i) {
            const double inv = std::pow((double) theta, -(double) (2 * i) / 128.0);
            double ang = (double) pos[r] * inv;
            ang -= two_pi * std::rint(ang / two_pi);
            table[(r * 2 + 0) * 64 + i] = (float) std::cos(ang);
            table[(r * 2 + 1) * 64 + i] = (float) std::sin(ang);
        }
}
}  // namespace

// the context update of `n` rows (<= 8) whose taps start at `taps` (+ off bytes): positions at pos[0..n) (int32 buffer + off), rope table rows at rt (+ off)
void Window::df2_ctx_list(std::vector<mtl::Dispatch>& ds, const mtl::Buffer& taps, size_t taps_off, int n, const mtl::Buffer& pos, size_t pos_off,
                          const mtl::Buffer& rt, size_t rt_off) {
    Df2& d = *df2_;
    const int tap_n = (int) d.taps.size() * N;
    auto xprep_at = [&](const mtl::Buffer& x, size_t off, const Act& out, int nn) {
        mtl::Dispatch dd;
        dd.p = P(lib_, "xprep", {}, "xprep");
        const uint32_t nv = nn, nt = MAXT;
        dd.args = {mtl::Arg::B(x, off), mtl::Arg::B(out.b), mtl::Arg::V(nv), mtl::Arg::V(nt)};
        dd.tg[0] = 256;
        dd.grid[0] = (nn / 16 * MAXT + 255) / 256;
        push(ds, dd);
    };
    auto rms = [&](const mtl::Buffer& x, const mtl::Buffer& y, const char* wname, int rows) {
        struct RmsArgs { uint32_t n; float eps; } a{(uint32_t) N, d.eps};
        mtl::Dispatch dd;
        dd.p = P(ldf2_, "df2_rms", {}, "df2_rms");
        dd.args = {mtl::Arg::B(x), mtl::Arg::B(d.pack, d.ten.at(wname).offset), mtl::Arg::B(y), mtl::Arg::V(a)};
        dd.grid[0] = rows;
        dd.tg[0] = 256;
        push(ds, dd);
    };
    xprep_at(taps, taps_off, d.A_tap, tap_n);
    gemv(ds, d.ten.at("fc.weight"), d.A_tap, d.fcy, N, n, 0, &d.pack);
    rms(d.fcy, d.hctx, "hidden_norm.weight", n);
    xprep_at(d.hctx, 0, d.A_h, N);
    for (int l = 0; l < d.L; ++l) {
        const std::string p = "layers." + std::to_string(l) + ".";
        gemv(ds, d.ten.at(p + "self_attn.k_proj.weight"), d.A_h, d.kc, 1024, n, 0, &d.pack);
        gemv(ds, d.ten.at(p + "self_attn.v_proj.weight"), d.A_h, d.vc, 1024, n, 0, &d.pack);
        struct NrArgs { uint32_t in_stride, slot_mod, heads; float eps; uint32_t has_rope; } na{1024, (uint32_t) DF2_RING, 8, d.eps, 1};
        mtl::Dispatch kn;
        kn.p = P(ldf2_, "df2_norm_rope", {}, "df2_norm_rope");
        kn.args = {mtl::Arg::B(d.kc), mtl::Arg::B(d.pack, d.ten.at(p + "self_attn.k_norm.weight").offset), mtl::Arg::B(pos, pos_off), mtl::Arg::B(rt, rt_off),
                   mtl::Arg::B(d.kring[l]), mtl::Arg::V(na)};
        kn.grid[0] = n;
        kn.grid[1] = 8;
        kn.tg[0] = 128;
        push(ds, kn);
        struct CopyArgs { uint32_t slot_mod, rows, cols; } ca{(uint32_t) DF2_RING, (uint32_t) n, 1024};
        mtl::Dispatch cp;
        cp.p = P(ldf2_, "df2_copy_rows", {}, "df2_copy_rows");
        cp.args = {mtl::Arg::B(d.vc), mtl::Arg::B(d.vring[l]), mtl::Arg::B(pos, pos_off), mtl::Arg::V(ca)};
        cp.grid[0] = (n * 1024 + 255) / 256;
        cp.tg[0] = 256;
        nb_ = true;   // beside the K write
        push(ds, cp);
    }
}

// the block: 8 rows (the anchor + 7 masks), h already holds their embeddings
void Window::df2_block_list(std::vector<mtl::Dispatch>& ds) {
    Df2& d = *df2_;
    const int B = DF2_ROWS;
    auto xprep_at = [&](const mtl::Buffer& x, size_t off, const Act& out, int nn) {
        mtl::Dispatch dd;
        dd.p = P(lib_, "xprep", {}, "xprep");
        const uint32_t nv = nn, nt = MAXT;
        dd.args = {mtl::Arg::B(x, off), mtl::Arg::B(out.b), mtl::Arg::V(nv), mtl::Arg::V(nt)};
        dd.tg[0] = 256;
        dd.grid[0] = (nn / 16 * MAXT + 255) / 256;
        push(ds, dd);
    };
    auto rms = [&](const mtl::Buffer& x, const mtl::Buffer& y, const char* wname, int rows) {
        struct RmsArgs { uint32_t n; float eps; } a{(uint32_t) N, d.eps};
        mtl::Dispatch dd;
        dd.p = P(ldf2_, "df2_rms", {}, "df2_rms");
        dd.args = {mtl::Arg::B(x), mtl::Arg::B(d.pack, d.ten.at(wname).offset), mtl::Arg::B(y), mtl::Arg::V(a)};
        dd.grid[0] = rows;
        dd.tg[0] = 256;
        push(ds, dd);
    };
    auto conv = [&](const mtl::Buffer& x, const mtl::Buffer& dyn, size_t dyn_off, size_t base_off, const mtl::Buffer* resid, const mtl::Buffer& out) {
        struct ConvArgs { uint32_t rows, H, group, dyn_stride, has_resid; } a{(uint32_t) B, (uint32_t) N, (uint32_t) d.group, 640, resid ? 1u : 0u};
        mtl::Dispatch dd;
        dd.p = P(ldf2_, "df2_conv", {}, "df2_conv");
        dd.args = {mtl::Arg::B(x), mtl::Arg::B(dyn, dyn_off), mtl::Arg::B(d.pack, base_off), mtl::Arg::B(resid ? *resid : x), mtl::Arg::B(out), mtl::Arg::V(a)};
        dd.grid[0] = (B * N + 255) / 256;
        dd.tg[0] = 256;
        push(ds, dd);
    };
    for (int l = 0; l < d.L; ++l) {
        const std::string p = "layers." + std::to_string(l) + ".";
        const size_t base_a = d.ten.at(p + "attention_conv.base_kernel").offset, base_m = d.ten.at(p + "mlp_conv.base_kernel").offset;
        rms(d.h, d.xn, (p + "input_layernorm.weight").c_str(), B);
        xprep_at(d.xn, 0, d.A_x, N);
        gemv(ds, d.ten.at(p + "attention_conv.kernel_projection.weight"), d.A_x, d.dyn, 640, B, 0, &d.pack);
        conv(d.xn, d.dyn, 0, base_a, nullptr, d.xc);
        xprep_at(d.xc, 0, d.A_x, N);
        gemv(ds, d.ten.at(p + "self_attn.q_proj.weight"), d.A_x, d.q, 4096, B, 0, &d.pack);
        gemv(ds, d.ten.at(p + "self_attn.k_proj.weight"), d.A_x, d.k, 1024, B, 0, &d.pack);
        gemv(ds, d.ten.at(p + "self_attn.v_proj.weight"), d.A_x, d.v, 1024, B, 0, &d.pack);
        {
            struct NrArgs { uint32_t in_stride, slot_mod, heads; float eps; uint32_t has_rope; } qa{4096, 0, 32, d.eps, 1}, ka{1024, 0, 8, d.eps, 1};
            mtl::Dispatch qd, kd;
            qd.p = kd.p = P(ldf2_, "df2_norm_rope", {}, "df2_norm_rope");
            qd.args = {mtl::Arg::B(d.q), mtl::Arg::B(d.pack, d.ten.at(p + "self_attn.q_norm.weight").offset), mtl::Arg::B(d.bpos), mtl::Arg::B(d.rtb),
                       mtl::Arg::B(d.qn), mtl::Arg::V(qa)};
            qd.grid[0] = B; qd.grid[1] = 32; qd.tg[0] = 128;
            push(ds, qd);
            kd.args = {mtl::Arg::B(d.k), mtl::Arg::B(d.pack, d.ten.at(p + "self_attn.k_norm.weight").offset), mtl::Arg::B(d.bpos), mtl::Arg::B(d.rtb),
                       mtl::Arg::B(d.kblk), mtl::Arg::V(ka)};
            kd.grid[0] = B; kd.grid[1] = 8; kd.tg[0] = 128;
            nb_ = true;
            push(ds, kd);
        }
        {
            struct AttnArgs { uint32_t ring, window; } aa{(uint32_t) DF2_RING, (uint32_t) d.window};
            mtl::Dispatch ap, am;
            ap.p = P(ldf2_, "df2_attn_part", {}, "df2_attn_part");
            ap.args = {mtl::Arg::B(d.qn), mtl::Arg::B(d.kring[l]), mtl::Arg::B(d.vring[l]), mtl::Arg::B(d.kblk), mtl::Arg::B(d.v),
                       mtl::Arg::B(d.anchor, 4), mtl::Arg::B(d.part), mtl::Arg::V(aa)};
            ap.grid[0] = 8; ap.grid[1] = DF2_MAXCH; ap.tg[0] = 64;
            push(ds, ap);
            am.p = P(ldf2_, "df2_attn_merge", {}, "df2_attn_merge");
            am.args = {mtl::Arg::B(d.part), mtl::Arg::B(d.anchor, 4), mtl::Arg::B(d.ao), mtl::Arg::V(aa)};
            am.grid[0] = 8 * 32; am.tg[0] = 128;
            push(ds, am);
        }
        xprep_at(d.ao, 0, d.A_ao, 4096);
        gemv(ds, d.ten.at(p + "self_attn.o_proj.weight"), d.A_ao, d.o, N, B, 0, &d.pack);
        conv(d.o, d.dyn, 320 * 4, base_a + 2 * N * 4, &d.h, d.h);
        rms(d.h, d.xn, (p + "post_attention_layernorm.weight").c_str(), B);
        xprep_at(d.xn, 0, d.A_x, N);
        gemv(ds, d.ten.at(p + "mlp_conv.kernel_projection.weight"), d.A_x, d.dyn, 640, B, 0, &d.pack);
        conv(d.xn, d.dyn, 0, base_m, nullptr, d.xc);
        xprep_at(d.xc, 0, d.A_x, N);
        gemv(ds, d.ten.at(p + "mlp.gate_proj.weight"), d.A_x, d.gate, d.I, B, 0, &d.pack);
        gemv(ds, d.ten.at(p + "mlp.up_proj.weight"), d.A_x, d.up, d.I, B, 0, &d.pack);
        {
            struct SwArgs { uint32_t n; } sa{(uint32_t) (B * d.I)};
            mtl::Dispatch sd;
            sd.p = P(ldf2_, "df2_swiglu", {}, "df2_swiglu");
            sd.args = {mtl::Arg::B(d.gate), mtl::Arg::B(d.up), mtl::Arg::B(d.sw), mtl::Arg::V(sa)};
            sd.grid[0] = (sa.n + 255) / 256;
            sd.tg[0] = 256;
            push(ds, sd);
        }
        xprep_at(d.sw, 0, d.A_sw, d.I);
        gemv(ds, d.ten.at(p + "mlp.down_proj.weight"), d.A_sw, d.mlp, N, B, 0, &d.pack);
        conv(d.mlp, d.dyn, 320 * 4, base_m + 2 * N * 4, &d.h, d.h);
    }
    rms(d.h, d.fn, "norm.weight", B);
    xprep_at(d.fn, (size_t) N * 4, d.A_fn, N);   // rows 1..7
    gemv(ds, d.head_t, d.A_fn, d.logits, (size_t) d.nd, DF2_DRAFTS, 0, d.head_buf);
    {
        struct TkArgs { int32_t n; } ta{(int32_t) d.nd};
        mtl::Dispatch tp, tm;
        tp.p = P(ldf2_, "df2_topk_part", {}, "df2_topk_part");
        tp.args = {mtl::Arg::B(d.logits), mtl::Arg::B(d.tkv), mtl::Arg::B(d.tki), mtl::Arg::V(ta)};
        tp.grid[0] = DF2_DRAFTS; tp.grid[1] = DF2_TOPB; tp.tg[0] = 256;
        push(ds, tp);
        struct TkMArgs { int32_t use_map; } tma{d.use_map ? 1 : 0};
        tm.p = P(ldf2_, "df2_topk_merge", {}, "df2_topk_merge");
        tm.args = {mtl::Arg::B(d.tkv), mtl::Arg::B(d.tki), mtl::Arg::B(d.vmap), mtl::Arg::B(d.topval), mtl::Arg::B(d.toptok), mtl::Arg::V(tma)};
        tm.grid[0] = DF2_DRAFTS; tm.tg[0] = 256;
        push(ds, tm);
    }
    gemv(ds, d.ten.at("candidate_selector.hidden_projection.weight"), d.A_fn, d.hsel, 256, DF2_DRAFTS, 0, &d.pack);
    {
        struct SelArgs { uint32_t rank, rows; } sa{(uint32_t) d.rank, (uint32_t) DF2_DRAFTS};
        mtl::Dispatch sd;
        sd.p = P(ldf2_, "df2_select", {}, "df2_select");
        sd.args = {mtl::Arg::B(d.hsel), mtl::Arg::B(d.topval), mtl::Arg::B(d.toptok),
                   mtl::Arg::B(d.pack, d.ten.at("candidate_selector.predecessor_codebook").offset),
                   mtl::Arg::B(d.pack, d.ten.at("candidate_selector.successor_codebook").offset), mtl::Arg::B(d.anchor, 0),
                   mtl::Arg::B(d.outbuf, 0), mtl::Arg::B(d.outbuf, 16 * 4), mtl::Arg::V(sa)};
        sd.grid[0] = 1; sd.tg[0] = 256;
        push(ds, sd);
    }
}

// context rows of the prompt: the taps are in `taps` ([n][nt * N] floats, a Window buffer or `staging`)
bool Window::df2_context(const mtl::Buffer& taps, size_t taps_off, int64_t n, int64_t pos0, std::string& err) {
    Df2& d = *df2_;
    if (!d.bound) { err = "dflash2: not bound"; return false; }
    const size_t row_bytes = d.taps.size() * N * 4;
    for (int64_t b0 = 0; b0 < n; b0 += DF2_CTX_MAX) {
        const int nb = (int) std::min<int64_t>(DF2_CTX_MAX, n - b0);
        int32_t* pos = d.ctxpos.as<int32_t>();
        for (int i = 0; i < nb; ++i) pos[i] = (int32_t) (pos0 + b0 + i);
        std::vector<mtl::Dispatch> ds;
        nb_ = false;
        for (int g0 = 0; g0 < nb; g0 += 8) {
            const int m = std::min(8, nb - g0);
            df2_rope_rows(pos + g0, m, d.theta, d.rtctx.as<float>() + (size_t) g0 * 128);
            df2_ctx_list(ds, taps, taps_off + (size_t) (b0 + g0) * row_bytes, m, d.ctxpos, (size_t) g0 * 4, d.rtctx, (size_t) g0 * 128 * 4);
        }
        cx_->run_graph(ds);
    }
    return true;
}

bool Window::df2_draft(int n_new, int64_t pos0, int32_t anchor, int32_t* drafts, float* probs, std::string& err) {
    Df2& d = *df2_;
    if (!d.bound) { err = "dflash2: not bound"; return false; }
    if (n_new < 1 || n_new > DF2_ROWS) { err = "dflash2: draft rows out of range"; return false; }
    const double t0 = now_ms();
    const int64_t s = pos0 + n_new;
    int32_t* dp = d.dpos.as<int32_t>();
    int32_t* bp = d.bpos.as<int32_t>();
    for (int i = 0; i < 8; ++i) { dp[i] = (int32_t) (pos0 + std::min(i, n_new - 1)); bp[i] = (int32_t) (s + i); }
    int32_t* an = d.anchor.as<int32_t>();
    an[0] = anchor;
    an[1] = (int32_t) s;
    df2_rope_rows(dp, 8, d.theta, d.rtd.as<float>());
    df2_rope_rows(bp, 8, d.theta, d.rtb.as<float>());
    float* h = d.h.as<float>();
    {
        const gguf::Tensor& te = T("token_embd.weight");
        const size_t rb = ref::row_bytes(te.type, N);
        if (const float* img = nullptr) (void) img;
        ref::dequant(te.type, (const uint8_t*) W_.ptr + te.offset + (size_t) anchor * rb, h, N);
        for (int i = 1; i < DF2_ROWS; ++i) std::memcpy(h + (size_t) i * N, d.mask_emb.data(), (size_t) N * 4);
    }
    if (!d.built[n_new]) {
        nb_ = false;
        df2_ctx_list(d.list[n_new], tap_dec_, 0, n_new, d.dpos, 0, d.rtd, 0);
        df2_block_list(d.list[n_new]);
        d.built[n_new] = true;
    }
    int32_t* ot = d.outbuf.as<int32_t>();
    for (int i = 0; i < DF2_DRAFTS; ++i) ot[i] = -1;
    const double g = cx_->run_graph(d.list[n_new]);
    d.gpu_ms += 1000.0 * g;
    for (int i = 0; i < DF2_DRAFTS; ++i) {
        drafts[i] = ot[i];
        if (probs) probs[i] = d.outbuf.as<float>()[16 + i];
    }
    ++d.rounds;
    (void) t0;
    return true;
}

void Window::df2_free() {
    delete df2_;
    df2_ = nullptr;
}

uint64_t Window::df2_bytes() const { return df2_ ? df2_->bytes : 0; }
double Window::df2_gpu_ms() const { return df2_ ? df2_->gpu_ms : 0.0; }

// ---- entry points for the shared host code (verify.cpp, prefill.cpp) and the drafter shim below
bool window_taps(void* w, const std::vector<int>& layers, const float** tap_dev, std::string& err) {
    Window* x = (Window*) w;
    if (!x->taps_enable(layers, err)) return false;
    *tap_dev = x->tap_dec_ptr();
    return true;
}

// Prefill::run's Metal side: taps for the prompt rows at positions >= first_pos (cb == nullptr: none)
bool prompt_taps(const std::vector<int>& layers, int64_t first_pos, const std::function<bool(const float*, int64_t, int64_t, std::string&)>& cb,
                 std::string& err) {
    if (g_window == nullptr) { err = "metal prompt: the decode window is not up yet"; return false; }
    if (cb && !g_window->taps_enable(layers, err)) return false;
    g_window->taps_set_cb(first_pos, cb);
    return true;
}

bool Window::df2_ensure(const std::string& path, std::string& err) {
    if (df2_ != nullptr) return true;
    return df2_open(path, err);
}

// context rows given as a host pointer into one of the window's tap buffers (the prompt chunk's, the decode window's) or anywhere else (copied)
bool Window::df2_context_ptr(const float* taps, int64_t n, int64_t pos0, std::string& err) {
    if (df2_ == nullptr || !df2_->bound) { err = "dflash2: not bound"; return false; }
    const size_t row_bytes = tap_layers_.size() * N * 4;
    const uint8_t* p = (const uint8_t*) taps;
    for (mtl::Buffer* b : {&tap_pro_, &tap_dec_})
        if (b->handle && p >= b->as<uint8_t>() && p + (size_t) n * row_bytes <= b->as<uint8_t>() + b->bytes)
            return df2_context(*b, (size_t) (p - b->as<uint8_t>()), n, pos0, err);
    if (df2_stage_rows_ < n + MAXT) {
        if (df2_stage_.handle) cx_->free(df2_stage_);
        df2_stage_ = cx_->alloc((size_t) (n + MAXT) * row_bytes);
        df2_stage_rows_ = n + MAXT;
    }
    std::memcpy(df2_stage_.ptr, taps, (size_t) n * row_bytes);
    return df2_context(df2_stage_, 0, n, pos0, err);
}

}  // namespace strata::metal_engine

// ---- core::Dflash2Drafter on Metal: the class of include/strata/core/dflash2.hpp, backed by the Window above.  load() reads the pack's index (the
// tap layers are asked before the window exists); bind() brings the pack and the draft head to the GPU once the window is up.
namespace strata::core {

namespace {
std::string g_df2_pack_path;
}

Dflash2Drafter::~Dflash2Drafter() {}

bool Dflash2Drafter::load(const std::string& pack_path, int device, std::string& err) {
    device_ = device;
    std::vector<int> taps;
    std::map<std::string, gguf::Tensor> ten;
    std::map<std::string, std::string> cfg;
    uint64_t total = 0, data_off = 0;
    if (!strata::metal_engine::df2_read_index(pack_path, taps, ten, total, data_off, cfg, err)) return false;
    taps_ = taps;
    if (cfg.count("mask_token")) mask_token_ = std::atoi(cfg["mask_token"].c_str());
    g_df2_pack_path = pack_path;
    return true;
}

bool Dflash2Drafter::bind(const WeightTable&, const NativeHead*, const std::string& vocab_file, std::string& err) {
    strata::metal_engine::Window* w = strata::metal_engine::g_window;
    if (w == nullptr) { err = "dflash2: the Metal decode window is not up"; return false; }
    if (bound_) return true;
    if (!w->df2_ensure(g_df2_pack_path, err) || !w->df2_bind(vocab_file, err)) return false;
    vram_ = w->df2_bytes();
    bound_ = true;
    return true;
}

bool Dflash2Drafter::bind_raw(EmbedFn, const void*, int, int64_t, const std::vector<int32_t>&, std::string& err) {
    err = "dflash2: bind_raw is a test hook of the CUDA / HIP build";
    return false;
}

bool Dflash2Drafter::context(const float* taps, int64_t n, int64_t pos0, std::string& err) {
    strata::metal_engine::Window* w = strata::metal_engine::g_window;
    if (w == nullptr) { err = "dflash2: not bound"; return false; }
    if (!bound_) {   // the prompt's taps arrive before bind() (CUDA loads the weights in load()): bring the pack up now, over the MTP's draft vocabulary
        if (!w->df2_ensure(g_df2_pack_path, err) || !w->df2_bind("", err)) return false;
        vram_ = w->df2_bytes();
        bound_ = true;
    }
    if (!w->df2_context_ptr(taps, n, pos0, err)) return false;
    have_ctx_ = std::max<int64_t>(have_ctx_, pos0 + n);
    return true;
}

bool Dflash2Drafter::draft(const float* taps, int n_new, int64_t pos0, int32_t anchor, int32_t* drafts, float* probs, std::string& err) {
    strata::metal_engine::Window* w = strata::metal_engine::g_window;
    if (w == nullptr || !bound_) { err = "dflash2: not bound"; return false; }
    if (taps != nullptr && taps != w->tap_dec_ptr()) { err = "dflash2: the taps are not the window's"; return false; }
    const auto t0 = std::chrono::steady_clock::now();
    if (!w->df2_draft(n_new, pos0, anchor, drafts, probs, err)) return false;
    ms_draft += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    ms_gpu = w->df2_gpu_ms();
    ++rounds;
    have_ctx_ = std::max<int64_t>(have_ctx_, pos0 + n_new);
    return true;
}

bool Dflash2Drafter::debug_rows(std::vector<float>& rows, std::string& err) {
    err = "dflash2: debug_rows is not on Metal";
    (void) rows;
    return false;
}

}  // namespace strata::core

namespace strata::metal_engine {
