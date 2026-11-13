// include/strata/program/tp_pair.hpp - ONE TENSOR-PARALLEL RANK'S STATE, and how it is built.
//
// The engine drives two ranks that each hold a slice of every split weight and run the SAME window on the same
// tokens.  A rank is therefore: its own device, its own shard of the weights (the canonical arena and the native
// upload), its own session, a nominal expert tier, and its own `Verifier` on its own stream.  The collective is
// shared and owned here too, because the two ranks meet on it.
//
// SYCL PORT: the CUDA original (include/strata/program/tp_pair.hpp in the TP prototype) used the CUDA runtime
// directly.  This copy uses the port's own device guard (`strata::core::OnDevice`), dpct queues and USM, and
// keeps the same structure so the engine wiring can stay close to the CUDA TP prototype.
#pragma once

#include "strata/core/layout.hpp"
#include "strata/core/native_dense.hpp"
#include "strata/core/native_head.hpp"
#include "strata/core/expert_cache.hpp"
#include "strata/core/shard.hpp"
#include "strata/core/tp_host_runtime.hpp"
#include "strata/core/verify.hpp"
#include "strata/core/weights.hpp"
#include "strata/core/on_device.hpp"
#include "strata/sycl_queue.hpp"
#include "strata/prefill/prefill.hpp"

#include <algorithm>
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <condition_variable>
#include <mutex>
#include <set>
#include <string>
#include <vector>

namespace strata::program {

/// Switches to a device for the DURATION OF A SCOPE and restores the previous one.  The port's `OnDevice`
/// does exactly this; the alias keeps the TP code readable.
using OnTpDevice = strata::core::OnDevice;

/// One rank of a tensor-parallel pair.
/// THE PROMPT PATH'S REDUCE (replicated dense + expert split).  A prompt chunk is thousands of rows, far past what
/// the in-kernel collective is sized for, so the ranks meet on the host: each copies its layer output down, every
/// rank's thread adds the partial sums in rank order (the same order on every rank, so the sums are bit-identical),
/// and each copies its sum back up.  One instance, shared by the ranks' threads.
struct TpBulkReduce {
    int n = 0;
    size_t cap = 0;   // floats
    std::vector<float*> part, sum;
    std::mutex mu;
    std::condition_variable cv;
    int arrived = 0;
    uint64_t gen = 0;
    bool broken = false;

    std::vector<sycl::context> ctx;
    /// `queues[r]` is rank r's queue: its two buffers are host USM of ITS context, so its copies down and up are
    /// DMA (a copy into pageable memory measured ~25 ms for a 21 MB layer output, 4x Arc Pro B60).
    bool init(const std::vector<sycl::queue*>& queues, size_t max_floats) {
        n = (int) queues.size();
        cap = max_floats;
        const size_t bytes = (max_floats * sizeof(float) + 63) / 64 * 64;
        part.assign((size_t) n, nullptr);
        sum.assign((size_t) n, nullptr);
        for (int r = 0; r < n; ++r) {
            ctx.push_back(queues[(size_t) r]->get_context());
            part[(size_t) r] = (float*) sycl::aligned_alloc_host(64, bytes, ctx.back());
            sum[(size_t) r] = (float*) sycl::aligned_alloc_host(64, bytes, ctx.back());
            if (part[(size_t) r] == nullptr || sum[(size_t) r] == nullptr) return false;
            std::memset(part[(size_t) r], 0, bytes);   // the pages are in before the first prompt
            std::memset(sum[(size_t) r], 0, bytes);
        }
        return true;
    }
    ~TpBulkReduce() {
        for (size_t r = 0; r < ctx.size(); ++r) {
            if (r < part.size() && part[r] != nullptr) sycl::free(part[r], ctx[r]);
            if (r < sum.size() && sum[r] != nullptr) sycl::free(sum[r], ctx[r]);
        }
    }
    // rank 0's thread's time in each part of the reduce, since the last `restart` (STRATA_PREFILL_TIMING prints it)
    double ms_down = 0, ms_meet1 = 0, ms_sum = 0, ms_meet2 = 0;
    int64_t calls0 = 0;
    void restart() {
        std::lock_guard<std::mutex> lk(mu);
        broken = false;
        arrived = 0;
        if (calls0 > 0 && std::getenv("STRATA_PREFILL_TIMING") != nullptr)
            std::fprintf(stderr, "strata tp: prompt reduce, rank 0, %lld calls: copy down %.0f ms, waiting for the ranks "
                                 "%.0f ms, sum %.0f ms, waiting for the sums %.0f ms\n", (long long) calls0, ms_down,
                         ms_meet1, ms_sum, ms_meet2);
        ms_down = ms_meet1 = ms_sum = ms_meet2 = 0;
        calls0 = 0;
    }
    void fail() { std::lock_guard<std::mutex> lk(mu); broken = true; cv.notify_all(); }
    /// Every rank's thread waits here until all of them have arrived.  False: a rank failed or never came.
    bool meet() {
        std::unique_lock<std::mutex> lk(mu);
        if (broken) return false;
        const uint64_t g = gen;
        if (++arrived == n) { arrived = 0; ++gen; cv.notify_all(); return true; }
        if (!cv.wait_for(lk, std::chrono::seconds(120), [&] { return gen != g || broken; })) {
            broken = true;
            cv.notify_all();
            return false;
        }
        return !broken;
    }
    bool reduce(int r, float* y, int64_t floats, sycl::queue& q, std::string& err, int64_t layer = -1) {
        if (r < 0 || r >= n || floats <= 0 || (size_t) floats > cap) {
            err = "tensor parallel: a prompt reduce of " + std::to_string(floats) + " floats against room for " +
                  std::to_string(cap);
            fail();
            return false;
        }
        using RClock = std::chrono::steady_clock;
        auto since = [](RClock::time_point t) { return std::chrono::duration<double, std::milli>(RClock::now() - t).count(); };
        auto t0 = RClock::now();
        q.memcpy(part[(size_t) r], y, (size_t) floats * sizeof(float)).wait();
        if (r == 0) { ms_down += since(t0); ++calls0; }
        t0 = RClock::now();
        if (!meet()) { err = "tensor parallel: a rank did not reach the prompt reduce"; return false; }
        if (r == 0) ms_meet1 += since(t0);
        t0 = RClock::now();
        // THIS THREAD SUMS ITS STRIPE, FOR EVERY RANK.  One pass over the partials in all (each thread a 1/n stripe)
        // instead of every thread summing the whole output; the adds are in rank order, so every rank's copy of the
        // sum holds the same bits.
        {
            const int64_t lo = floats * r / n, hi = floats * (r + 1) / n;
            float tmp[2048];
            for (int64_t at = lo; at < hi; at += 2048) {
                const int64_t m = std::min<int64_t>(2048, hi - at);
                const float* p0 = part[0] + at;
                for (int64_t i = 0; i < m; ++i) tmp[i] = p0[i];
                for (int k = 1; k < n; ++k) {
                    const float* pk = part[(size_t) k] + at;
                    for (int64_t i = 0; i < m; ++i) tmp[i] += pk[i];
                }
                for (int k = 0; k < n; ++k) std::memcpy(sum[(size_t) k] + at, tmp, (size_t) m * sizeof(float));
            }
        }
        float* s = sum[(size_t) r];
        if (r == 0) ms_sum += since(t0);
        t0 = RClock::now();
        // TPDIAG (STRATA_TP_DIAG_BULK=1): every rank's partial and the sum, per layer, for the first chunks.
        static const bool diag = std::getenv("STRATA_TP_DIAG_BULK") != nullptr;
        // nobody stages the next layer over `part` until every rank has read this one
        if (!meet()) { err = "tensor parallel: a rank did not finish the prompt reduce"; return false; }
        if (r == 0) ms_meet2 += since(t0);
        if (r == 0) {   // TPDIAG (STRATA_TP_DIAG_BULK_DUMP=<file>): the first reduce's sum, raw floats
            static bool dumped = false;
            if (const char* dp = std::getenv("STRATA_TP_DIAG_BULK_DUMP"); dp != nullptr && !dumped) {
                dumped = true;
                if (std::FILE* f = std::fopen(dp, "wb")) { std::fwrite(s, sizeof(float), (size_t) floats, f); std::fclose(f); }
            }
        }
        if (diag && r == 0) {
            static int calls = 0;
            if (calls++ < 96) {
                std::string line;
                char b[96];
                for (int k = 0; k < n; ++k) {
                    double a2 = 0.0;
                    for (int64_t i = 0; i < floats; ++i) a2 += (double) part[(size_t) k][i] * (double) part[(size_t) k][i];
                    std::snprintf(b, sizeof b, " part%d=%.9g", k, a2);
                    line += b;
                }
                double t1 = 0.0, t2 = 0.0;
                for (int64_t i = 0; i < floats; ++i) { t1 += (double) s[i]; t2 += (double) s[i] * (double) s[i]; }
                std::fprintf(stderr, "TPDIAG bulk layer=%lld n=%lld%s sum=%.9g sumsq=%.9g\n", (long long) layer,
                             (long long) floats, line.c_str(), t1, t2);
            }
        }
        q.memcpy(y, s, (size_t) floats * sizeof(float));   // in order on the stream; the next reduce's copy down waits for it
        return true;
    }
};

/// One expert a rank owns under the expert split: where its bytes are in pinned host memory.
struct TpOwnedExpert {
    int32_t layer = 0;
    int32_t expert = 0;
    const uint8_t* host = nullptr;
    int64_t bytes = 0;
};

struct TpRank {
    int dev = -1;
    int rank = 0;
    int ranks = 2;
    dpct::queue_ptr stream = nullptr;
    void* arena = nullptr;
    void* session_buf = nullptr;
    strata::core::ExpertCache xcache;  ///< rank 0's expert cache, replicated here (an ExpertCache so the prompt
                                       ///< path's lend/refill and the verify hit path both work on this card)
    int32_t* residency = nullptr;  ///< device [n_layers * n_expert], every entry -1 (nothing resident)
    uint64_t arena_bytes = 0;
    strata::core::WeightTable wt;
    strata::core::NativeDense dense;
    strata::core::SessionState ss;
    strata::core::VerifyHits vh;
    strata::core::Verifier ver;
    /// Replicated dense + expert split: this rank's own batched prompt path and the host copy of its residency table.
    strata::prefill::Prefill sp;
    bool sp_ready = false;
    std::vector<int32_t> h_res_own;

    /// OPENS THIS RANK'S DEVICE AND STREAM - early, before the engine allocates the single-card arenas it
    /// needs for its own context.  The port's `OnDevice` guard switches the current device; the queue is
    /// created on that device.
    bool open(int device, int which, int how_many, std::string& err) {
        dev = device;
        rank = which;
        ranks = how_many;
        int prev_open = -1;
        (void) DPCT_CHECK_ERROR(prev_open = dpct::get_current_device_id());
        if (DPCT_CHECK_ERROR(dpct::select_device(dev)) != 0) {
            err = "tensor parallel: select_device(" + std::to_string(dev) + ") failed";
            return false;
        }
        if (stream == nullptr) {
            stream = dpct::get_current_device().create_queue(true);
            if (stream == nullptr) {
                std::fprintf(stderr, "strata tp: rank %d on device %d: its own stream was refused, running on "
                                     "the default stream\n", rank, dev);
            }
        }
        {
            sycl::device d = dpct::get_device(dev);
            const std::string name = d.get_info<sycl::info::device::name>();
            std::fprintf(stderr, "strata tp: rank %d opened device %d (%s) (stream %s)\n", rank, dev,
                         name.c_str(), stream != nullptr ? "up" : "not yet");
        }
        if (prev_open >= 0) (void) DPCT_CHECK_ERROR(dpct::select_device(prev_open));
        return true;
    }

    /// Loads this rank's shards and builds its session, tier and verifier.  `skip` is the engine's own skip set
    /// (the tensors the native path serves) - the same one the single-card path uses.  `src_cache`, when given,
    /// is rank 0's expert cache: it is REPLICATED on this rank's card so the replicated MoE is computed
    /// identically on both ranks.
    bool build(const std::string& pack, const std::vector<std::string>& native_shards,
               const std::set<std::string>* skip, const strata::core::ModelGeometry& g, int64_t max_context,
               int64_t k, int device, int which, int how_many, const strata::core::NativeHead* head, int max_t,
               std::string& err, const strata::core::VerifyHits* src_cache = nullptr,
               int64_t src_cache_bytes = 0, const std::vector<int64_t>* slot_bytes = nullptr,
               const std::vector<TpOwnedExpert>* own = nullptr, bool whole = false) {
        // `whole` (STRATA_TP_DENSE=replicated): the rank loads the WHOLE dense model and a whole session, exactly
        // as a single card does; only its experts are its own.
        dev = device;
        rank = which;
        ranks = how_many;
        int src_dev = -1;
        (void) DPCT_CHECK_ERROR(src_dev = dpct::get_current_device_id());
        if (!open(dev, rank, ranks, err)) return false;
        const OnTpDevice on_rank(dev);   // every allocation below belongs to THIS rank's card
        if (DPCT_CHECK_ERROR(dpct::select_device(dev)) != 0) {
            err = "tensor parallel: select_device(" + std::to_string(dev) + ") failed";
            return false;
        }
        if (stream == nullptr) {
            stream = dpct::get_current_device().create_queue(true);
            if (stream == nullptr) {
                std::fprintf(stderr, "strata tp: rank %d on device %d: its own stream was refused, running on "
                                     "the default stream\n", rank, dev);
            }
        }
        // this rank's plan decides what it holds; `plan` outlives the load call below
        strata::core::ShardTable table;
        strata::core::ShardPlan plan;
        if (!whole && (!strata::core::shard_table_n(g, rank, ranks, table, err) ||
                       !strata::core::shard_plan_layers(table, g, rank, plan, err))) {
            return false;
        }
        const strata::core::ShardPlan* plan_arg = whole ? nullptr : &plan;
        if (!strata::core::WeightTable::pool_bytes(pack, arena_bytes, err, skip, plan_arg)) return false;
        sycl::queue& q = dpct::get_in_order_queue();
        arena = strata::malloc_device_guarded(arena_bytes, q, "tensor parallel rank arena");
        if (arena == nullptr) {
            err = "tensor parallel: the rank's arena failed";
            return false;
        }
        if (!wt.load(pack, arena, arena_bytes, err, skip, plan_arg)) return false;
        // the native upload takes only this rank's rows of each matrix it serves
        // include_ple_key: the rank needs its own copy of the PLE key, which the canonical loader skips
        // because the engine serves it natively on rank 0.  Without it the rank's PLE is not ready.
        if (!dense.load(native_shards, wt, err, true, 0, -1, plan_arg)) return false;
        // the session, exactly as the layer split builds one
        session_buf = sycl::malloc_device(strata::core::session_bytes(g, max_context, k, 0, g.n_layers), q);
        if (session_buf == nullptr) {
            err = "tensor parallel: the rank's session failed";
            return false;
        }
        if (strata::core::session_init(g, max_context, k, session_buf, ss, 0, g.n_layers) == 0) {
            err = "tensor parallel: session_init failed";
            return false;
        }
        // THE SESSION MUST START ZEROED.  Rank 0's is zeroed by the engine; a rank whose GDN state and KV
        // cache are uninitialized reads garbage on the first window and the ranks diverge from token one.
        strata::core::session_zero(ss, g, nullptr, stream);
        if (own != nullptr && !own->empty()) {
            // THE EXPERT SPLIT (opt-in): THIS RANK'S OWN EXPERTS.  Each expert in `own` is held by this rank and
            // by no other, so the cache is filled from the pinned host copies instead of from rank 0's cache, and
            // the residency table names this rank's slots.
            std::vector<int64_t> sizes;
            sizes.reserve(own->size());
            int64_t max_blob = 0;
            for (const TpOwnedExpert& x : *own) { sizes.push_back(x.bytes); max_blob = std::max(max_blob, x.bytes); }
            if (!xcache.open_sized(sizes, g.n_layers, g.n_expert, err)) return false;
            const uint64_t* off = xcache.slot_offsets();
            uint8_t* base = (uint8_t*) xcache.device_slot(0);
            if (off == nullptr || base == nullptr) {
                err = "tensor parallel: the rank's own expert cache has no slot table";
                return false;
            }
            const size_t nres = (size_t) (g.n_layers * g.n_expert);
            std::vector<int32_t> res(nres, -1);
            size_t total = 0;
            for (size_t i = 0; i < own->size(); ++i) {
                const TpOwnedExpert& x = (*own)[i];
                q.memcpy(base + (size_t) off[i], x.host, (size_t) x.bytes);
                res[(size_t) x.layer * (size_t) g.n_expert + (size_t) x.expert] = (int32_t) i;
                total += (size_t) x.bytes;
                if ((i & 63) == 63) q.wait();
            }
            q.wait();
            residency = sycl::malloc_device<int32_t>(nres, q);
            if (residency == nullptr) {
                err = "tensor parallel: the rank's residency table failed";
                return false;
            }
            q.memcpy(residency, res.data(), nres * sizeof(int32_t)).wait();
            h_res_own = res;
            vh.n_slots = (int64_t) own->size();
            vh.blob = src_cache != nullptr && src_cache->blob > 0 ? src_cache->blob : max_blob;
            vh.cache_base = base;
            vh.slot_off = off;
            vh.d_res = residency;
            vh.h_res = nullptr;
            std::fprintf(stderr, "strata tp: rank %d: holds %zu experts of its own (%llu MiB)\n", rank, own->size(),
                         (unsigned long long) (total >> 20));
        } else if (src_cache != nullptr && src_cache->n_slots > 0 && src_cache->cache_base != nullptr &&
            src_cache->d_res != nullptr && src_cache->blob > 0 && std::getenv("STRATA_TP_NO_CACHE") == nullptr) {
            // REPLICATE RANK 0'S EXPERT CACHE ON THIS CARD.  The MoE is replicated, not sharded, so both ranks
            // must compute it identically: the same residents through the same GPU hit kernel and the same
            // misses through the shared pool.
            vh.n_slots = src_cache->n_slots;
            vh.blob = src_cache->blob;
            const size_t cache_bytes = src_cache_bytes > 0 ? (size_t) src_cache_bytes
                                                           : (size_t) vh.n_slots * (size_t) vh.blob;
            if (slot_bytes == nullptr || slot_bytes->empty()) {
                err = "tensor parallel: the expert cache replica needs the source cache's slot sizes";
                return false;
            }
            if (!xcache.open_sized(*slot_bytes, g.n_layers, g.n_expert, err)) return false;
            // The copy is staged through the host: Arc has no peer access, so the bytes go rank 0's card ->
            // host -> this card in 64 MiB pieces.
            {
                const uint8_t* sp = src_cache->cache_base;
                const size_t chunk = 64u << 20;
                // EACH COPY ON ITS OWN CARD'S QUEUE.  `q` is THIS rank's queue and stays so after a
                // select_device: reading rank 0's device memory through it does not fault on Arc (xe), it
                // returns other bytes, and the replica - and so every expert this rank computed - was wrong
                // (measured: the ranks' residuals parted at layer 0's expert output, 2x Arc Pro B60).
                std::vector<uint8_t> stage(chunk);
                bool cok = true;
                uint8_t* dcache = (uint8_t*) xcache.device_slot(0);
                for (size_t off = 0; off < cache_bytes && cok; off += chunk) {
                    const size_t nb = (off + chunk <= cache_bytes) ? chunk : cache_bytes - off;
                    if (DPCT_CHECK_ERROR(dpct::select_device(src_dev)) != 0) { cok = false; break; }
                    dpct::get_in_order_queue().memcpy(stage.data(), sp + off, nb).wait();
                    if (DPCT_CHECK_ERROR(dpct::select_device(dev)) != 0) { cok = false; break; }
                    dpct::get_in_order_queue().memcpy(dcache + off, stage.data(), nb).wait();
                }
                (void) DPCT_CHECK_ERROR(dpct::select_device(dev));
                if (!cok) return false;
            }
            vh.cache_base = (const uint8_t*) xcache.device_slot(0);
            vh.slot_off = xcache.slot_offsets();
            const size_t nres = (size_t) (g.n_layers * g.n_expert);
            {
                std::vector<int32_t> stage(nres);
                residency = sycl::malloc_device<int32_t>(nres, q);
                if (residency == nullptr) {
                    err = "tensor parallel: the rank's residency table failed";
                    return false;
                }
                if (DPCT_CHECK_ERROR(dpct::select_device(src_dev)) != 0) {
                    err = "tensor parallel: the rank's residency table failed";
                    return false;
                }
                dpct::get_in_order_queue().memcpy(stage.data(), src_cache->d_res, nres * sizeof(int32_t)).wait();
                if (DPCT_CHECK_ERROR(dpct::select_device(dev)) != 0) {
                    err = "tensor parallel: the rank's residency table failed";
                    return false;
                }
                dpct::get_in_order_queue().memcpy(residency, stage.data(), nres * sizeof(int32_t)).wait();
            }
            vh.d_res = residency;
            vh.h_res = src_cache->h_res;
            vh.slot_off = src_cache->slot_off;
            std::fprintf(stderr, "strata tp: rank %d: the expert cache is replicated (%lld slots, %llu MiB)\n", rank,
                         (long long) vh.n_slots, (unsigned long long) (cache_bytes >> 20));
        } else {
            // A NOMINAL EXPERT TIER.  `Verifier::init` refuses a tier whose three fields are empty; every entry
            // here is -1, so no expert is resident and the pool serves all of them.
            vh.n_slots = 1;
            vh.blob = 4 * 1024 * 1024;
            {
                std::vector<int64_t> one_slot(1, (int64_t) vh.blob);
                if (!xcache.open_sized(one_slot, g.n_layers, g.n_expert, err)) {
                    err = "tensor parallel: the rank's expert slot failed: " + err;
                    return false;
                }
            }
            vh.cache_base = (const uint8_t*) xcache.device_slot(0);
            vh.slot_off = xcache.slot_offsets();
            const size_t n = (size_t) (g.n_layers * g.n_expert);
            std::vector<int32_t> none(n, -1);
            residency = sycl::malloc_device<int32_t>(n, q);
            if (residency == nullptr) {
                err = "tensor parallel: the rank's residency table failed";
                return false;
            }
            q.memcpy(residency, none.data(), n * sizeof(int32_t)).wait();
            vh.d_res = residency;
            vh.h_res = nullptr;   // the host table is a fast path for the doorbell, which this run does not use
        }
        // THE RANK BEFORE ANYTHING IS SIZED, and then PUT BACK.  The pair is per THREAD (`shard.hpp`), and this
        // call runs on the thread that drives rank 0: leaving rank 1 set here would hand rank 0's own capture and
        // windows rank 1's head ranges.  Rank 1's thread sets it again for its own window.
        const int save_rank = strata::core::tp_rank_current();
        const int save_ranks = strata::core::tp_ranks_current();
        const int save_xrank = strata::core::tp_xrank_ref(), save_xranks = strata::core::tp_xranks_ref();
        if (whole) strata::core::tp_set_expert_ranks(rank, ranks);
        else if (!strata::core::tp_set_ranks(rank, ranks, err)) return false;
        const bool inited = ver.init(wt, g, ss, vh, head, max_t, err);
        strata::core::tp_set_expert_ranks(save_xrank, save_xranks);
        std::string init_err = err;
        ver.set_stream(stream);
        ver.set_stage(0, g.n_layers, nullptr, nullptr);   // every layer, no hand-off anywhere
        std::string restore_err;
        (void) strata::core::tp_set_ranks(save_rank, save_ranks, restore_err);
        if (!inited) {
            err = "tensor parallel: rank " + std::to_string(rank) + "'s verifier: " + init_err;
            return false;
        }
        return true;
    }

    void free() {
        if (dev < 0) return;
        (void) DPCT_CHECK_ERROR(dpct::select_device(dev));
        if (sp_ready) { sp.reset(); sp_ready = false; }
        sycl::queue& q = dpct::get_in_order_queue();
        if (residency != nullptr) sycl::free(residency, q);
        xcache.close();
        if (session_buf != nullptr) sycl::free(session_buf, q);
        if (arena != nullptr) sycl::free(arena, q);
        stream = nullptr;
        residency = nullptr;
        session_buf = nullptr;
        arena = nullptr;
    }
};

}  // namespace strata::program
