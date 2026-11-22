// --kv-grow with the expert offload: the K/V and the prompt buffers take their memory from the expert pools, and pools_release stops the
// process when a pool would fall under FETCH_MAX slots.  These checks answer beforehand whether a growth fits: pf_fit reads a prompt in smaller
// chunks while one does not, and the serve loop ends a request the card cannot hold with an error (a prompt) or "length" (an answer) instead of
// the process stopping.  Nothing here runs without --kv-grow.
// CBA: --spec with --kv-grow (resident only in setup) still grows through kv_fit unchecked; its verify reads the prompt path

bool g_pf_full = false;    // pf_fit: even a 256-token chunk would leave a pool under its minimum
bool g_ctx_full = false;   // context_full ended the request (the serve loop goes on to the next one)
void pf_unmap();

// would pools_release(bytes) leave every pool FETCH_MAX slots?  The same arithmetic, nothing released
bool pools_keep_min(double bytes) {
    if (bytes <= 0) return true;
    if (!E.offload) return false;
    const double deficit = bytes + 4.0 * g_vmm.gran, used = pool_bytes(true) + pool_bytes(false);
    for (bool nv : {true, false}) {
        const int ns = nv ? E.nslot_nv : E.nslot_mx;
        if (!ns) continue;
        const int drop = (int) std::ceil(deficit * pool_bytes(nv) / used / pool_slot(nv));
        if (ns - drop < FETCH_MAX) return false;
    }
    return true;
}

// can the K/V grow to `need` cells (kv_fit's step) without a pool falling under its minimum?
bool kv_room(int need) {
    if (!E.grow || need <= E.kv_cells) return true;
    const int nc = std::min(E.ctx, (need + E.grow_step - 1) / E.grow_step * E.grow_step);
    const size_t add = kv_mapped_for(nc) - kv_mapped_for(E.kv_cells), keep = vram_keep(), fr = vram_free();
    return fr >= add + keep || pools_keep_min((double) (add + keep - fr));
}

// the serve loop's end of a prompt the card cannot hold: the buffers and the K/V beyond the default step back to the pools, nothing of this
// prompt kept as conversation (the K/V holds part of it), an ERR line for the server (no DONE follows)
void context_full(std::vector<int>& hist, size_t read, size_t prompt) {
    pf_unmap();
    kv_trim(1);
    hist.clear();
    g_pf_full = false;
    g_ctx_full = true;
    std::fprintf(stderr, "kv-grow: the prompt does not fit, %zu of %zu tokens read; expert slots %d + %d\n", read, prompt, E.nslot_nv,
                 E.nslot_mx);
    std::printf("ERR the context is full: the K/V of this %zu-token prompt does not fit beside the expert pools on this card (%zu tokens read); "
                "a shorter conversation fits\n", prompt, read);
    std::fflush(stdout);
}
