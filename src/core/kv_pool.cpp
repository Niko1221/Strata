// src/core/kv_pool.cpp - see include/strata/core/kv_pool.hpp.
#include "strata/core/kv_pool.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdio>

namespace strata::core {

bool KvPool::init(const ModelGeometry& g, const SessionState& ref, int64_t tokens, std::string& error) {
    page_size_ = strata::kernels::qsa_real_shapes().page_size;
    const int64_t chunk = chunk_cells();
    const int64_t n = (tokens + chunk - 1) / chunk;
    if (n <= 0) { error = "kv pool: no tokens"; return false; }
    const int64_t pages = (n + 1) << kChunkShift;   // + the trash chunk
    qsa_ord0_ = ref.qsa_ord0;
    layers_.assign((size_t) ref.qsa_alloc, strata::kernels::KvHostPools{});
    const uint64_t before = qsa_kv_host_bytes();
    for (int64_t j = 0; j < ref.qsa_alloc; ++j) {
        const QsaState& st = ref.qsa_states[ref.qsa_ord0 + j];
        if (st.kv_mode != 1) continue;   // a layer kept whole in VRAM has no host copy
        if (!qsa_host_alloc(g, st, pages, layers_[(size_t) j])) {
            error = "kv pool: cannot pin the pool's K/V";
            return false;
        }
    }
    pinned_bytes_ = qsa_kv_host_bytes() - before;
    n_chunks_ = n;
    free_.clear();
    for (int64_t c = n - 1; c >= 0; --c) free_.push_back((int32_t) c);   // the lowest chunk is handed out first
    return true;
}

KvPool::Table* KvPool::find(const SessionState& ss) {
    for (auto& t : tables_) if (t->ss == &ss) return t.get();
    return nullptr;
}
const KvPool::Table* KvPool::find(const SessionState& ss) const {
    for (const auto& t : tables_) if (t->ss == &ss) return t.get();
    return nullptr;
}

bool KvPool::upload(const Table& t) {
    // From pageable memory the copy may still be landing when cudaMemcpy returns, and the kernels run on
    // non-blocking streams that do not wait for it: the table is complete before the next launch reads it.
    // An upload is rare: once per chunk (4096 cells) a session grows by, and when it gives chunks back.
    return cudaMemcpy(t.dev, t.host.data(), t.host.size() * sizeof(int32_t), cudaMemcpyHostToDevice) == cudaSuccess &&
           cudaDeviceSynchronize() == cudaSuccess;
}

bool KvPool::attach(SessionState& ss, std::string& error) {
    if (!active() || find(ss) != nullptr) { error = "kv pool: not initialized, or the session is attached already"; return false; }
    if (ss.qsa_ord0 != qsa_ord0_ || ss.qsa_alloc != (int64_t) layers_.size()) {
        error = "kv pool: the session's QSA layers differ from the pool's";
        return false;
    }
    auto t = std::make_unique<Table>();
    t->ss = &ss;
    const int64_t chunk = chunk_cells();
    t->host.assign((size_t) ((ss.max_cells + chunk - 1) / chunk), (int32_t) n_chunks_);
    if (cudaMalloc((void**) &t->dev, t->host.size() * sizeof(int32_t)) != cudaSuccess || !upload(*t)) {
        error = "kv pool: cannot allocate a chunk table";
        return false;
    }
    for (int64_t j = 0; j < ss.qsa_alloc; ++j) {
        QsaState& st = ss.qsa_states[ss.qsa_ord0 + j];
        if (st.kv_mode != 1) continue;
        if (!layers_[(size_t) j].present()) { error = "kv pool: a streamed layer the pool has no K/V for"; return false; }
        st.host = layers_[(size_t) j];
        st.host.chunk = t->dev;
        st.host.chunk_host = t->host.data();
        st.host.chunk_shift = kChunkShift;
    }
    tables_.push_back(std::move(t));
    return true;
}

bool KvPool::reserve(const SessionState& ss, int64_t cells) {
    Table* t = find(ss);
    if (t == nullptr) return false;
    const int64_t chunk = chunk_cells();
    const int64_t need = std::min<int64_t>((std::max<int64_t>(cells, 0) + chunk - 1) / chunk, (int64_t) t->host.size());
    if (need <= t->held) return true;
    if (need - t->held > (int64_t) free_.size()) return false;
    for (; t->held < need; ++t->held) {
        t->host[(size_t) t->held] = free_.back();
        free_.pop_back();
    }
    ++version_;
    if (!upload(*t)) {
        std::fprintf(stderr, "strata: kv pool: chunk table upload failed\n");
        std::exit(1);   // the kernels would write through a table the movers disagree with
    }
    return true;
}

void KvPool::shrink(const SessionState& ss, int64_t cells) {
    Table* t = find(ss);
    const int64_t chunk = chunk_cells();
    const int64_t keep = (std::max<int64_t>(cells, 0) + chunk - 1) / chunk;
    if (t == nullptr || t->held <= keep) return;
    for (int64_t c = t->held - 1; c >= keep; --c) {
        free_.push_back(t->host[(size_t) c]);
        t->host[(size_t) c] = (int32_t) n_chunks_;
    }
    t->held = keep;
    ++version_;
    if (!upload(*t)) {
        std::fprintf(stderr, "strata: kv pool: chunk table upload failed\n");
        std::exit(1);
    }
}

bool KvPool::swap(const SessionState& a, const SessionState& b) {
    Table* ta = find(a);
    Table* tb = find(b);
    if (ta == nullptr || tb == nullptr || ta == tb || ta->host.size() != tb->host.size()) return false;
    // element by element: the states point at these buffers (chunk_host), so they must stay where they are
    std::swap_ranges(ta->host.begin(), ta->host.end(), tb->host.begin());
    std::swap(ta->held, tb->held);
    if (!upload(*ta) || !upload(*tb)) {
        std::fprintf(stderr, "strata: kv pool: chunk table upload failed\n");
        std::exit(1);
    }
    ++version_;
    return true;
}

int64_t KvPool::reserved_cells(const SessionState& ss) const {
    const Table* t = find(ss);
    return t != nullptr ? t->held * chunk_cells() : 0;
}

}  // namespace strata::core
