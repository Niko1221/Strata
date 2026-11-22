// The conversation cache of --serve (--conversation-cache-mib N, off by default): agentic use (one agent over many
// turns, an agent and its sub-agents taking turns) does not read again a prompt it has already read.
//
// The slots hold one conversation (the live one, `hist` of the serve loop).  Its state after n tokens is, per layer
// visit: the full-attention K / V rows [0, n) (8 visits), the sliding-window rows of positions [n - WINDOW, n) in
// their ring (40 visits), the four causal-conv rings (khist / vhist / ahist / mhist, HR rows each, 48 visits); with
// --spec the drafter's K / V (its global layer: rows [0, n), its two local layers: their ring's window); and the
// logits after token n - 1.  Every --kv format is bytes per row here (codes and scales alike, kv_cell_bytes).  Not
// per conversation, not saved: the expert pools and the CPU experts' placement, the drafter's Markov context (the
// serve loop always reads at least the prompt's last token with --spec, which recomputes it), the sampler.
//
// A checkpoint (Ck) is the "tail" of a state at n: everything above except the full-attention rows, ~190 MiB at f16,
// in page-locked host memory.  The full rows [0, n) are free to keep: a later position never writes them.  So:
//   - at each prompt's end (before the first sampled token) a checkpoint of the live conversation: a client that
//     sends the history back re-rendered (its reasoning dropped, its tool call re-serialized) diverges from the
//     generated tokens after the prompt, and the next request restores that point instead of reading from 0;
//   - a request that does not extend the live conversation's last prompt parks it (its full rows, its checkpoints
//     and one at its history end) and restores the parked conversation, live checkpoint or branch point that covers
//     the longest prefix of the request;
//   - a request sharing a long prefix with a known conversation without a checkpoint there (sub-agents sharing a
//     system prompt and tools) stops its prefill at the branch point once and takes a checkpoint there.
// LRU over whole parked conversations under the budget.  Off: none of this runs (the serve loop is the base one).

constexpr int SPLIT_MIN = 512;     // a branch point is taken when it covers this many tokens more than the best state

struct Pinned {   // page-locked host memory (DMA only: any NUMA node)
    uint8_t* p = nullptr;
    size_t n = 0;
    explicit Pinned(size_t bytes) {
        if (bytes && cudaHostAlloc((void**) &p, bytes, cudaHostAllocDefault) == cudaSuccess) n = bytes;
        else { p = nullptr; (void) cudaGetLastError(); }
    }
    ~Pinned() { if (p) cudaFreeHost(p); }
    Pinned(const Pinned&) = delete;
    Pinned& operator=(const Pinned&) = delete;
};
using PinnedP = std::shared_ptr<Pinned>;

struct Ck {             // the tail of the state after the first n tokens
    int n = 0;
    PinnedP tail;
    vec logits;
    int kind = 0;          // CK_PROMPT a prompt's end, CK_BRANCH a branch point, CK_END a parked history's end
};
constexpr int CK_PROMPT = 0, CK_BRANCH = 1, CK_END = 2;
struct Conv {           // a parked conversation: its tokens, its full rows [0, ids.size()), its checkpoints
    std::vector<int> ids;
    PinnedP full;
    std::vector<Ck> cks;
    int prompt_n = 0;      // its last prompt's length: a request extending it continues this conversation
    uint64_t used = 0;
    bool hold = false;     // the one sess_begin restores: not evicted meanwhile
};

struct SessArr { uint8_t* d; size_t b; int cap; };   // device rows of b bytes, position q in row q % cap

struct Sess {
    bool on = false;
    size_t budget = 0;     // bytes
    int min_tokens = 512;  // --conversation-cache-min-tokens: shorter conversations are not parked
    std::vector<SessArr> full, ring;
    std::vector<std::pair<void*, size_t>> conv;
    size_t tail_bytes = 0, row_bytes = 0;
    std::vector<Ck> live;  // checkpoints of the live conversation (its full rows are the slots')
    int live_prompt = 0;   // the live conversation's last prompt length (0: none)
    std::vector<Conv> parked;
    std::vector<PinnedP> spare;   // tails no longer used (cudaHostAlloc is slow): taken again first
    uint64_t clock = 0;
    int split = -1;        // this request: the prefill stops here for a branch checkpoint
    std::string note;      // this request's log line
} g_sess;

void sess_init(int mib, int min_tokens) {
    if (mib <= 0) return;
    g_sess.min_tokens = min_tokens;
    for (int s = 0; s < NCALLS; ++s) {
        Slot& S = E.S[s];
        void* arr[4] = {S.K, S.V, S.ksc, S.vsc};
        for (int a = 0; a < 4; ++a) {
            const size_t b = kv_cell_bytes(a);
            if (!b) continue;
            (S.full ? g_sess.full : g_sess.ring).push_back({(uint8_t*) arr[a], b, S.cap});
        }
        g_sess.conv.push_back({S.khist, (size_t) HR * KVD * 4});
        g_sess.conv.push_back({S.vhist, (size_t) HR * KVD * 4});
        g_sess.conv.push_back({S.ahist, (size_t) HR * H * 4});
        g_sess.conv.push_back({S.mhist, (size_t) HR * H * 4});
    }
    if (D.on)
        for (int i = 0; i < DL; ++i)
            for (__half* x : {D.L[i].K, D.L[i].V}) (D.L[i].local ? g_sess.ring : g_sess.full).push_back({(uint8_t*) x, (size_t) KVD * 2, D.L[i].cap});
    for (auto& a : g_sess.ring) g_sess.tail_bytes += (size_t) WINDOW * a.b;
    for (auto& c : g_sess.conv) g_sess.tail_bytes += c.second;
    for (auto& a : g_sess.full) g_sess.row_bytes += a.b;
    g_sess.budget = (size_t) mib << 20;
    if (g_sess.budget < 3 * g_sess.tail_bytes) {
        std::fprintf(stderr, "--conversation-cache-mib %d: below %zu MiB (three checkpoints of %zu MiB): cache off\n", mib,
                     (3 * g_sess.tail_bytes) >> 20, g_sess.tail_bytes >> 20);
        return;
    }
    g_sess.on = true;
    std::fprintf(stderr, "conversation cache: %d MiB of host memory, checkpoints of %.0f MiB, %.1f KiB per parked token, "
                         "conversations of %d+ tokens parked\n", mib, g_sess.tail_bytes / 1048576.0, g_sess.row_bytes / 1024.0, min_tokens);
}

size_t sess_lcp(const std::vector<int>& a, const std::vector<int>& b) {
    size_t l = 0;
    while (l < a.size() && l < b.size() && a[l] == b[l]) ++l;
    return l;
}
// positions [lo, hi) of an array <-> host bytes h (position order)
void sess_rows(const SessArr& a, int lo, int hi, uint8_t* h, bool d2h) {
    for (int q = lo; q < hi;) {
        const int r = q % a.cap, m = std::min(hi - q, a.cap - r);
        uint8_t* dv = a.d + (size_t) r * a.b;
        uint8_t* hv = h + (size_t) (q - lo) * a.b;
        CK(cudaMemcpyAsync(d2h ? (void*) hv : (void*) dv, d2h ? (const void*) dv : (const void*) hv, (size_t) m * a.b,
                           d2h ? cudaMemcpyDeviceToHost : cudaMemcpyHostToDevice, cudaStreamPerThread));
        q += m;
    }
}
void sess_tail(int n, uint8_t* h, bool d2h) {   // the ring windows and conv rings of the state at n
    CK(cudaDeviceSynchronize());
    const int lo = std::max(0, n - WINDOW);
    for (auto& a : g_sess.ring) { sess_rows(a, lo, n, h, d2h); h += (size_t) WINDOW * a.b; }
    for (auto& c : g_sess.conv) {
        CK(cudaMemcpyAsync(d2h ? (void*) h : c.first, d2h ? (const void*) c.first : (const void*) h, c.second,
                           d2h ? cudaMemcpyDeviceToHost : cudaMemcpyHostToDevice, cudaStreamPerThread));
        h += c.second;
    }
    CK(cudaDeviceSynchronize());
}
void sess_full(int n, uint8_t* h, bool d2h) {   // the full rows [0, n), array after array
    CK(cudaDeviceSynchronize());
    for (auto& a : g_sess.full) { sess_rows(a, 0, n, h, d2h); h += (size_t) n * a.b; }
    CK(cudaDeviceSynchronize());
}

// host bytes held: each tail once (branch checkpoints are shared between conversations)
size_t sess_used() {
    std::vector<const Pinned*> seen;
    size_t t = 0;
    auto tail = [&](const PinnedP& p) {
        if (p && std::find(seen.begin(), seen.end(), p.get()) == seen.end()) { seen.push_back(p.get()); t += p->n; }
    };
    for (auto& c : g_sess.live) tail(c.tail);
    for (auto& v : g_sess.parked) { t += v.full ? v.full->n : 0; for (auto& c : v.cks) tail(c.tail); }
    for (auto& p : g_sess.spare) tail(p);
    return t;
}
// room for `bytes` more: spare buffers freed first, then the least recently used parked conversations (not one held)
bool sess_room(size_t bytes) {
    while (sess_used() + bytes > g_sess.budget) {
        if (!g_sess.spare.empty()) { g_sess.spare.pop_back(); continue; }
        auto lru = g_sess.parked.end();
        for (auto it = g_sess.parked.begin(); it != g_sess.parked.end(); ++it)
            if (!it->hold && (lru == g_sess.parked.end() || it->used < lru->used)) lru = it;
        if (lru == g_sess.parked.end()) return false;
        g_sess.note += " evicted " + std::to_string(lru->ids.size());
        g_sess.parked.erase(lru);
    }
    return true;
}
// CBA: page-locked buffers are kept for reuse (cudaHostAlloc pins ~0.45 ms per MiB on a PCIe 3 host: 80 ms per
// checkpoint), a spare taken for anything up to half its size; checkpoints in PIN_UNIT steps, a parked conversation's
// rows in PIN_UNIT << k (a conversation that grew a turn finds its last buffer again).  Up to half a row buffer is
// slack; a sub-allocator in one pinned arena would waste less
constexpr size_t PIN_UNIT = (size_t) 32 << 20;
PinnedP sess_alloc(size_t bytes, bool geo = false) {
    bytes = (bytes + PIN_UNIT - 1) / PIN_UNIT * PIN_UNIT;
    if (geo) { size_t g = PIN_UNIT; while (g < bytes) g *= 2; bytes = g; }
    auto& sp = g_sess.spare;
    size_t bi = sp.size();
    for (size_t i = 0; i < sp.size(); ++i)
        if (sp[i]->n >= bytes && sp[i]->n <= 2 * bytes && (bi == sp.size() || sp[i]->n < sp[bi]->n)) bi = i;
    if (bi < sp.size()) { PinnedP p = sp[bi]; sp.erase(sp.begin() + bi); return p; }
    if (!sess_room(bytes)) return nullptr;
    PinnedP p = std::make_shared<Pinned>(bytes);
    return p->p ? p : nullptr;
}
void sess_drop(PinnedP&& p) {   // a buffer nothing uses any more goes to the spares
    if (p && p.use_count() == 1) g_sess.spare.push_back(std::move(p));
    p.reset();
}
// a checkpoint of the live state at n (its logits: lg)
bool sess_take(int n, const vec& lg, int kind) {
    PinnedP t = sess_alloc(g_sess.tail_bytes);
    if (!t) { g_sess.note += " no-room"; return false; }
    sess_tail(n, t->p, true);
    g_sess.live.push_back({n, t, lg, kind});
    return true;
}

// The live conversation (hist, last_logits) to the parked set, when worth it: its full rows, its checkpoints and its
// history end.  The live checkpoints stay valid for the slots (a later restore may use them).
void sess_park(const std::vector<int>& hist, const vec& last_logits) {
    if ((int) hist.size() < g_sess.min_tokens) return;
    Conv v;
    v.ids = hist;
    v.prompt_n = g_sess.live_prompt;
    v.cks = g_sess.live;
    const bool end = std::none_of(v.cks.begin(), v.cks.end(), [&](const Ck& c) { return c.n == (int) hist.size(); });
    const size_t need = hist.size() * g_sess.row_bytes + (end ? g_sess.tail_bytes : 0);
    if (!sess_room(need)) { g_sess.note += " park-no-room"; return; }
    // in the parked set (held) while its buffers are taken, so that the budget counts them
    v.used = ++g_sess.clock;
    v.hold = true;
    const uint64_t id = v.used;
    g_sess.parked.push_back(std::move(v));
    auto me = [&]() -> Conv& { for (auto& c : g_sess.parked) if (c.used == id) return c; return g_sess.parked.back(); };
    PinnedP full = sess_alloc(hist.size() * g_sess.row_bytes, true);
    if (!full) {
        for (size_t j = 0; j < g_sess.parked.size(); ++j) if (g_sess.parked[j].used == id) g_sess.parked.erase(g_sess.parked.begin() + j);
        g_sess.note += " park-no-room";
        return;
    }
    me().full = full;
    sess_full((int) hist.size(), full->p, true);
    if (end) {
        PinnedP t = sess_alloc(g_sess.tail_bytes);
        if (t) { sess_tail((int) hist.size(), t->p, true); me().cks.push_back({(int) hist.size(), t, last_logits, CK_END}); }
    }
    me().hold = false;
    g_sess.note += " parked " + std::to_string(hist.size());
}

// Before the request's kv_trim (a parked conversation's rows are still mapped): picks the longest prefix of pr with a
// state, parks the live conversation if pr does not continue it, restores that state.  `start` = the base loop's
// choice (hist.size() when pr extends hist, else 0); returns the position to read from.  hist / last_logits follow.
size_t sess_begin(const std::vector<int>& pr, std::vector<int>& hist, vec& last_logits, size_t start) {
    g_sess.split = -1;
    g_sess.note.clear();
    const double t0 = now_ms();
    // (a picture's image tokens are ids made from its pixels' hash, ink2_vision.cuh vis_id: every comparison below
    // tells two pictures of the same size apart)
    const size_t l_live = sess_lcp(hist, pr);
    // the best state: the live history end (start), a live checkpoint, a parked conversation's checkpoint
    size_t best = start, share = l_live;
    int src = start > 0 ? 0 : -1, ck = -1, pv = -1;   // src: 0 live end, 1 live checkpoint, 2 parked, -1 none
    for (size_t i = 0; i < g_sess.live.size(); ++i)
        if (g_sess.live[i].n <= (int) l_live && g_sess.live[i].n > (int) best) { best = g_sess.live[i].n; src = 1; ck = (int) i; }
    for (size_t j = 0; j < g_sess.parked.size(); ++j) {
        const size_t l = sess_lcp(g_sess.parked[j].ids, pr);
        share = std::max(share, l);
        for (size_t i = 0; i < g_sess.parked[j].cks.size(); ++i)
            if (g_sess.parked[j].cks[i].n <= (int) l && g_sess.parked[j].cks[i].n > (int) best) { best = g_sess.parked[j].cks[i].n; src = 2; ck = (int) i; pv = (int) j; }
    }
    // the live conversation is parked unless the request extends its last prompt (the same conversation, re-rendered
    // or continued)
    const bool same = (!hist.empty() && l_live == hist.size()) || (g_sess.live_prompt > 0 && (int) l_live >= g_sess.live_prompt);
    // a branch point between conversations only (a re-render diverging deep in the generated tail is not one)
    if (!same && share >= best + SPLIT_MIN && share < pr.size()) g_sess.split = (int) share;
    if (pv >= 0) g_sess.parked[pv].hold = true;
    if (!same && !hist.empty()) sess_park(hist, last_logits);   // (may evict parked conversations, not the held one)
    for (size_t j = 0; j < g_sess.parked.size(); ++j) if (g_sess.parked[j].hold) { pv = (int) j; g_sess.parked[j].hold = false; }
    const char* how = "live";
    if (src == 1) {   // the slots' own rows [0, n): the tail only
        const Ck c = g_sess.live[ck];
        sess_tail(c.n, c.tail->p, false);
        last_logits = c.logits;
        how = "rewind";
    } else if (src == 2) {
        Conv& v = g_sess.parked[pv];
        const Ck c = v.cks[ck];
        kv_fit(c.n + 1);   // --kv-grow: a conversation longer than the K/V mapped now
        if ((int) sess_lcp(hist, v.ids) < c.n || same) {   // (same: the slots' rows past the live prompt may differ)
            // the parked rows array after array: n rows of each, from an image of ids.size() rows each
            CK(cudaDeviceSynchronize());
            const uint8_t* h = v.full->p;
            for (auto& a : g_sess.full) { sess_rows(a, 0, c.n, const_cast<uint8_t*>(h), false); h += v.ids.size() * a.b; }
            CK(cudaDeviceSynchronize());
        }
        sess_tail(c.n, c.tail->p, false);
        last_logits = c.logits;
        how = "parked";
        v.used = ++g_sess.clock;
        // the request continues that conversation (extends its last prompt): it leaves the parked set, its checkpoints
        // up to n become the live ones; else (a sibling from a shared branch point) it stays parked, its branch
        // checkpoints shared
        std::vector<Ck> keep;
        for (auto& x : v.cks) if (x.n <= c.n && (x.n == c.n || x.kind == CK_BRANCH)) keep.push_back(x);
        g_sess.live = keep;
        if (v.prompt_n > 0 && (int) best >= v.prompt_n) {   // its buffers to the spares (unless shared with the live)
            Conv dead = std::move(v);
            g_sess.parked.erase(g_sess.parked.begin() + pv);
            sess_drop(std::move(dead.full));
            for (auto& x : dead.cks) sess_drop(std::move(x.tail));
        }
    } else if (src < 0) {
        how = "none";
    }
    for (size_t i = g_sess.live.size(); i-- > 0;)   // checkpoints past the read start are no longer the slots' state
        if (g_sess.live[i].n > (int) best) { PinnedP t = g_sess.live[i].tail; g_sess.live.erase(g_sess.live.begin() + i); sess_drop(std::move(t)); }
    g_sess.live_prompt = 0;   // (sess_prompt_end sets it)
    hist.assign(pr.begin(), pr.begin() + best);
    std::fprintf(stderr, "[ink2] conversation cache: prompt %zu, read from %zu (%s)%s%s, %zu parked, %.0f of %zu MiB, %.0f ms\n",
                 pr.size(), best, how, g_sess.split > 0 ? (", branch point " + std::to_string(g_sess.split)).c_str() : "",
                 g_sess.note.c_str(), g_sess.parked.size(), sess_used() / 1048576.0, g_sess.budget >> 20, now_ms() - t0);
    return best;
}

// the prefill chunk at p0 of T tokens: ends at the branch point when it crosses it
int sess_chunk(int p0, int T) { return g_sess.on && g_sess.split > p0 && g_sess.split < p0 + T ? g_sess.split - p0 : T; }
// after the chunk ending at p: the branch checkpoint.  CBA: the live conversation's checkpoints are never evicted
// (only parked conversations are); one branch point per request at most, they leave with it when it is parked
void sess_after_chunk(int p) {
    if (!g_sess.on || p != g_sess.split) return;
    vec lg(VOCAB);
    CK(cudaMemcpy(lg.data(), E.logits, (size_t) VOCAB * 4, cudaMemcpyDeviceToHost));
    const double t0 = now_ms();
    if (sess_take(p, lg, CK_BRANCH)) std::fprintf(stderr, "[ink2] conversation cache: branch checkpoint at %d, %.0f ms\n", p, now_ms() - t0);
}
// FNV-1a of the logits: identity checks (serve mode has no logits dump)
unsigned long long sess_hash(const vec& lg) {
    uint64_t h = 1469598103934665603ull;
    for (float f : lg) { uint32_t u; std::memcpy(&u, &f, 4); h = (h ^ u) * 1099511628211ull; }
    return (unsigned long long) h;
}
// the prompt read (lg: its last logits), before the first sampled token: the live prompt-end checkpoint
void sess_prompt_end(const std::vector<int>& pr, const vec& lg) {
    if (!g_sess.on) {   // INK2_LOGITS_HASH=1: the hash without the cache too (the reference runs)
        static const bool h = std::getenv("INK2_LOGITS_HASH") != nullptr;
        if (h) std::fprintf(stderr, "[ink2] conversation cache: off, prompt end %zu, logits %016llx\n", pr.size(), sess_hash(lg));
        return;
    }
    const double t0 = now_ms();
    bool have = false;   // a checkpoint at this very point already (the same prompt again, a restored one)
    for (size_t i = g_sess.live.size(); i-- > 0;) {   // earlier prompt ends / a restored end (prefixes of this) give way
        if (g_sess.live[i].n == (int) pr.size()) { have = true; continue; }
        if (g_sess.live[i].kind != CK_BRANCH) { PinnedP t = g_sess.live[i].tail; g_sess.live.erase(g_sess.live.begin() + i); sess_drop(std::move(t)); }
    }
    g_sess.note.clear();
    g_sess.live_prompt = have || sess_take((int) pr.size(), lg, CK_PROMPT) ? (int) pr.size() : 0;
    std::fprintf(stderr, "[ink2] conversation cache: prompt end %zu checkpoint %.0f ms%s, %.0f MiB, logits %016llx\n", pr.size(),
                 now_ms() - t0, g_sess.note.c_str(), sess_used() / 1048576.0, sess_hash(lg));
}
