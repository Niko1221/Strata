// src/core/conversation_capture_codec_test.cpp - what a RETAINED conversation checkpoint is when the engine
// captures it under --session-state-codec, and what a capture that cannot allocate does instead.
//
// The serve loop keeps a chain of conversation checkpoints and resumes later requests through them, so a checkpoint
// it captures is a STORED copy of the running state, not the working state: it sets the checkpoint's own
// `state_codec` to the session's resolved codec before calling conversation_checkpoint_save, and the snapshot is
// coded the moment it is taken - before any conversation has been parked or saved. That moves bf16's rounding to
// capture time, so it can change an ordinary in-memory prompt-cache restore even for a conversation that was never
// written to disk. The live buffers stay fp32 either way, and a capture only ever reads them.
//
// The claims this file pins down:
//   1. a capture codes the four running-state arrays and only those: `block_pos` is int32 positions and is copied
//      exactly, and the checkpoint says which codec it holds;
//   2. capture leaves every byte of the working state as it found it;
//   3. restoring the coded snapshot widens it into the session's fp32 buffers within the codec's own bound, leaves
//      the snapshot compressed, and does the same thing however many times it is restored;
//   4. a capture left at the API default writes the fp32 bytes it always wrote, bit for bit;
//   5. a capture that cannot allocate - the stored arrays, or the bounded staging slice a coded copy crosses the
//      device boundary in - is a refusal to its caller. It does not throw, it publishes no partially captured
//      checkpoint, the snapshots that were already valid stay valid, and the working state is untouched. Those two
//      allocations are different failures and a check that only counts refusals cannot tell them apart, so the
//      armed allocator also records the sizes it saw and each check names the one it expects to have failed.
//
// Host-only, and deliberately so: making this fail on a real device means either a fork whose address-space limit
// may or may not bite, or a model large enough to exhaust RAM. Here the device copies are link-wrapped to memcpy
// (the same fault injection conversation_transfer_test uses) and the allocator is armed to refuse requests of a
// given size, so "there is no RAM left to capture this" is a condition the test sets rather than hopes for. It says
// nothing about a real out-of-memory device; it says the capture's failure path is a refusal.

#include "strata/core/conversation_snapshot.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <new>
#include <string>
#include <vector>

// The device copies, replaced. `conversation_checkpoint_save` reaches the device only through
// conversation_detail::copy_nonblocking (cudaGetDevice, a private non-blocking stream, cudaMemcpyAsync,
// cudaStreamSynchronize), so wrapping these is enough to run the real capture control flow over host memory.
extern "C" cudaError_t __wrap_cudaGetDevice(int* dev) { *dev = 0; return cudaSuccess; }
extern "C" cudaError_t __wrap_cudaStreamCreateWithFlags(cudaStream_t* s, unsigned int) {
    *s = reinterpret_cast<cudaStream_t>(static_cast<uintptr_t>(1)); return cudaSuccess;
}
extern "C" cudaError_t __wrap_cudaMemcpy(void* dst, const void* src, size_t n, cudaMemcpyKind) {
    std::memcpy(dst, src, n); return cudaSuccess;
}
extern "C" cudaError_t __wrap_cudaMemcpyAsync(void* dst, const void* src, size_t n, cudaMemcpyKind, cudaStream_t) {
    return __wrap_cudaMemcpy(dst, src, n, cudaMemcpyDefault);
}
extern "C" cudaError_t __wrap_cudaStreamSynchronize(cudaStream_t) { return cudaSuccess; }
extern "C" cudaError_t __wrap_cudaDeviceSynchronize() { return cudaSuccess; }
extern "C" cudaError_t __wrap_cudaGetLastError() { return cudaSuccess; }

// The allocator, armable. While armed, a request of `fail_from_bytes` or more throws bad_alloc and is counted;
// smaller requests still succeed, so the code under test can still build its error string and the test says
// something about the capture rather than about the C++ runtime. Replacing the allocation functions is only allowed
// at file scope, so this is global: it is disarmed except for the one call each pressure check makes.
namespace {
bool failing = false;
std::size_t fail_from_bytes = 0;
std::size_t refused = 0;
// What the armed window saw, so a check can say WHICH allocation was refused and not merely that one was: a
// capture sizes five stored arrays before it reaches the staging slice, and a test that only counts refusals cannot
// tell those two apart. Fixed-size and allocation-free - recording a refusal must not itself want the heap that is
// being refused, which with `fail_from_bytes = 1` would mean recording nothing at all.
std::size_t refused_at[8] = {0};
std::size_t refused_n = 0;
std::size_t served_max = 0;   // the largest request the armed allocator said YES to
void arm(std::size_t from) { failing = true; fail_from_bytes = from; refused = 0; refused_n = 0; served_max = 0; }
void disarm() { failing = false; fail_from_bytes = 0; }
void* allocate(std::size_t n) {
    if (failing && n >= fail_from_bytes) {
        ++refused;
        if (refused_n < sizeof(refused_at) / sizeof(refused_at[0])) refused_at[refused_n] = n;
        ++refused_n;
        throw std::bad_alloc();
    }
    void* p = std::malloc(n == 0 ? 1 : n);
    if (p == nullptr) std::abort();
    if (failing && n > served_max) served_max = n;
    return p;
}
} // namespace
void* operator new(std::size_t n) { return allocate(n); }
void* operator new[](std::size_t n) { return allocate(n); }
void* operator new(std::size_t n, const std::nothrow_t&) noexcept {
    if (failing && n >= fail_from_bytes) {
        ++refused;
        if (refused_n < sizeof(refused_at) / sizeof(refused_at[0])) refused_at[refused_n] = n;
        ++refused_n;
        return nullptr;
    }
    void* p = std::malloc(n == 0 ? 1 : n);
    if (failing && n > served_max) served_max = n;
    return p;
}
void* operator new[](std::size_t n, const std::nothrow_t&) noexcept {
    return operator new(n, std::nothrow);   // every replaceable form allocates here, so every delete form frees here
}
void operator delete(void* p) noexcept { std::free(p); }
void operator delete[](void* p) noexcept { std::free(p); }
void operator delete(void* p, std::size_t) noexcept { std::free(p); }
void operator delete[](void* p, std::size_t) noexcept { std::free(p); }
void operator delete(void* p, const std::nothrow_t&) noexcept { std::free(p); }
void operator delete[](void* p, const std::nothrow_t&) noexcept { std::free(p); }

using namespace strata::core;

namespace {
int checks = 0;
void check(bool ok, const char* label) {
    ++checks;
    if (!ok) { std::fprintf(stderr, "FAIL: %s\n", label); std::exit(1); }
}

std::vector<float> widen(const std::vector<uint8_t>& fp32) {
    std::vector<float> f(fp32.size() / sizeof(float));
    if (!fp32.empty()) std::memcpy(f.data(), fp32.data(), fp32.size());
    return f;
}

/// The codec's own bound: every value within 2^-8 of the one it came from, and nothing that was not zero going to
/// zero. This is the error a restore introduces. It is not a claim that the answers are unchanged.
bool within_codec(const std::vector<uint8_t>& want, const std::vector<uint8_t>& got_fp32) {
    const std::vector<float> w = widen(want), g = widen(got_fp32);
    if (w.size() != g.size()) return false;
    for (size_t i = 0; i < w.size(); ++i) {
        const double x = w[i], y = g[i];
        if (x == 0.0) { if (y != 0.0) return false; continue; }
        if (std::fabs((y - x) / x) > 1.0 / 256) return false;
    }
    return true;
}

// One session's running state, in host memory, at a geometry the sizing code accepts. Every value is a real float
// in a range the model's own state occupies, so the widened numbers mean something.
//
// `ssm_state_size` is the one setting a pressure check is allowed to move, because it is the one that moves z.gdn -
// the fp32 bytes of recurrent state a capture crosses the device boundary in - and nothing else the test reads. At 2
// the whole running state is small enough that a capture's five stored arrays are all smaller than its staging
// slice, which is what a check about output-array allocation needs; at 256 z.gdn is 3,146,304 bytes, bigger than
// any of those arrays and still under the 4 MiB staging limit, which is what a check about the staging slice needs.
// The checks that pin down what a capture WRITES use the default and stay cheap.
struct State {
    ModelGeometry g;
    ConversationStateSizes z;
    std::array<QsaState, 2> layers{};
    std::vector<uint8_t> pooled[2], tail[2], dead[2], block_pos[2];
    std::vector<uint8_t> gdn, ple;
    SessionState ss;

    explicit State(int64_t ssm_state_size = 2) {
        g.n_layers = 8;              // qsa_interval 4: two QSA layers, six GDN ones
        g.ssm_state_size = ssm_state_size; g.ssm_v_heads = 2; g.ssm_conv_channels = 8;
        g.n_head_kv = 1; g.head_dim = 64; g.idx_key_dim = 8;
        std::string e;
        if (!conversation_state_sizes(g, z, e)) { std::fprintf(stderr, "sizing: %s\n", e.c_str()); std::exit(1); }
        for (size_t j = 0; j < layers.size(); ++j) {
            QsaState& st = layers[j];
            st.max_cells = 96; st.idx_pooled_rows = 26;
            pooled[j].assign((size_t) 26 * g.idx_key_dim * 4, 0);
            tail[j].assign(z.tail, 0); dead[j].assign(z.dead, 0); block_pos[j].assign(z.block_pos, 0);
            st.idx_pooled = (float*) pooled[j].data(); st.idx_tail = (float*) tail[j].data();
            st.idx_dead = (float*) dead[j].data(); st.idx_block_pos = (int32_t*) block_pos[j].data();
        }
        gdn.assign(z.gdn, 0); ple.assign(z.ple, 0);
        ss.max_cells = 96;
        ss.gdn_state = (float*) gdn.data(); ss.ple_hist = (float*) ple.data();
        ss.layer_hi = g.n_layers; ss.gdn_alloc = g.n_gdn_layers(); ss.qsa_alloc = g.n_qsa_layers();
        ss.qsa_states = layers.data();
        seed(1);
    }
    /// Every float of the running state written as a value of its own; positions are int32 and written as such.
    void seed(uint64_t salt) {
        auto fill_floats = [salt](void* p, size_t bytes, uint64_t k) {
            float* f = (float*) p;
            for (size_t i = 0; i < bytes / sizeof(float); ++i, ++k)
                f[i] = (float) ((double) ((int64_t) ((k * 6364136223846793005ull + salt) >> 40) - 100000) / 3.0);
        };
        fill_floats(gdn.data(), gdn.size(), 11);
        fill_floats(ple.data(), ple.size(), 7001);
        for (size_t j = 0; j < layers.size(); ++j) {
            fill_floats(tail[j].data(), tail[j].size(), 90000 + j);
            fill_floats(dead[j].data(), dead[j].size(), 500000 + j);
            int32_t* bp = (int32_t*) block_pos[j].data();
            for (size_t i = 0; i < block_pos[j].size() / sizeof(int32_t); ++i)
                bp[i] = (int32_t) (i * 37 + j * 1000 + (salt & 0x3f));   // block indices, not a bit pattern
        }
    }
    std::vector<uint8_t> tails() const {
        std::vector<uint8_t> all;
        for (auto& l : layers) all.insert(all.end(), (uint8_t*) l.idx_tail, (uint8_t*) l.idx_tail + z.tail);
        return all;
    }
    std::vector<uint8_t> deads() const {
        std::vector<uint8_t> all;
        for (auto& l : layers) all.insert(all.end(), (uint8_t*) l.idx_dead, (uint8_t*) l.idx_dead + z.dead);
        return all;
    }
    std::vector<uint8_t> positions() const {
        std::vector<uint8_t> all;
        for (auto& l : layers)
            all.insert(all.end(), (uint8_t*) l.idx_block_pos, (uint8_t*) l.idx_block_pos + z.block_pos);
        return all;
    }
    std::vector<uint8_t> bytes() const {   // the whole working state: a capture must leave it byte-identical
        std::vector<uint8_t> all = gdn;
        all.insert(all.end(), ple.begin(), ple.end());
        all.insert(all.end(), tail[0].begin(), tail[0].end()); all.insert(all.end(), tail[1].begin(), tail[1].end());
        all.insert(all.end(), dead[0].begin(), dead[0].end()); all.insert(all.end(), dead[1].begin(), dead[1].end());
        all.insert(all.end(), block_pos[0].begin(), block_pos[0].end());
        all.insert(all.end(), block_pos[1].begin(), block_pos[1].end());
        return all;
    }
};

ConversationCheckpoint checkpoint_of(const std::vector<int32_t>& ids, StateCodec codec) {
    ConversationCheckpoint c;
    c.ids = ids;
    c.state_codec = (uint8_t) codec;   // what the serve loop sets for a retained checkpoint
    return c;
}
} // namespace

int main() {
    std::printf("conversation_capture_codec_test\n");
    State st;
    std::vector<int32_t> ids(9);
    for (size_t i = 0; i < ids.size(); ++i) ids[i] = (int32_t) i + 1;
    std::string err;

    // ---- 1 + 2 + 4: the two captures of one running state, and what each one writes
    ConversationCheckpoint fp32 = checkpoint_of(ids, StateCodec::Fp32);
    check(conversation_checkpoint_save(fp32, st.ss, st.g, err), "capture: the default (fp32) capture");
    check(fp32.state_codec == 0, "capture: left at the API default the checkpoint says fp32");
    check(fp32.gdn == st.gdn && fp32.ple == st.ple && fp32.tails == st.tails() && fp32.dead == st.deads() &&
          fp32.block_pos == st.positions(),
          "capture: an fp32 capture is the running state's own bits, the size the last release wrote");

    const std::vector<uint8_t> live_before = st.bytes();
    ConversationCheckpoint bf16 = checkpoint_of(ids, StateCodec::Bf16);
    check(conversation_checkpoint_save(bf16, st.ss, st.g, err), "capture: the same point under bf16");
    check(bf16.state_codec == (uint8_t) StateCodec::Bf16,
          "capture: a retained checkpoint captured under the option names bf16");
    check(bf16.gdn.size() * 2 == fp32.gdn.size() && bf16.ple.size() * 2 == fp32.ple.size() &&
          bf16.tails.size() * 2 == fp32.tails.size() && bf16.dead.size() * 2 == fp32.dead.size(),
          "capture: all four coded arrays are half the fp32 ones");
    check(bf16.block_pos == fp32.block_pos && bf16.block_pos == st.positions() && !bf16.block_pos.empty(),
          "capture: block_pos is int32 positions and a coded capture copies it exactly");
    check(st.bytes() == live_before, "capture: a coded capture reads the working state and never writes it");

    // ---- 3: restoring it. The session's fp32 buffers get the widened values; the snapshot stays as it was.
    const std::vector<uint8_t> held_gdn = bf16.gdn, held_ple = bf16.ple, held_tails = bf16.tails,
                               held_dead = bf16.dead, held_pos = bf16.block_pos;
    st.seed(99);   // the conversation moves on
    check(conversation_checkpoint_restore(bf16, st.ss, st.g, err), "capture: restoring a freshly captured snapshot");
    ConversationCheckpoint back;
    back.ids = ids;
    check(conversation_checkpoint_save(back, st.ss, st.g, err), "capture: reading the session back after the restore");
    check(back.state_codec == 0 && within_codec(fp32.gdn, back.gdn) && within_codec(fp32.ple, back.ple) &&
          within_codec(fp32.tails, back.tails) && within_codec(fp32.dead, back.dead) &&
          back.block_pos == bf16.block_pos,
          "capture: what it restores is the stored values widened into fp32, within the codec's own 2^-8");
    check(bf16.gdn == held_gdn && bf16.ple == held_ple && bf16.tails == held_tails &&
          bf16.dead == held_dead && bf16.block_pos == held_pos &&
          bf16.state_codec == (uint8_t) StateCodec::Bf16,
          "capture: restoring it leaves the snapshot compressed, not widened in place");
    st.seed(123);
    check(conversation_checkpoint_restore(bf16, st.ss, st.g, err) &&
          conversation_checkpoint_restore(bf16, st.ss, st.g, err),
          "capture: the same snapshot restores again");
    check(bf16.gdn == held_gdn && bf16.ple == held_ple && bf16.tails == held_tails &&
          bf16.dead == held_dead && bf16.block_pos == held_pos,
          "capture: and repeated restoration does not change one byte of it");

    // ---- 5: a capture that cannot allocate. `conversation_checkpoint_save` sizes the five stored arrays first and
    // then crosses the device boundary one bounded slice at a time. Both want a heap block, and a check about one of
    // them has to prove WHICH one it refused: counting refusals cannot tell those two apart, and this file used to
    // call an output-array refusal a staging-buffer one. Every block below therefore records the sizes the armed
    // allocator saw and names the allocation it expects to have failed. The refusal is what the caller acts on, the
    // checkpoint it was handed keeps its bytes and its codec, and nothing is thrown past it.
    const std::vector<uint8_t> live = st.bytes();

    {
        // An OUTPUT array, on a checkpoint that already holds a whole snapshot. At this geometry every stored array
        // is smaller than the capture's staging slice, so with the bar at z.gdn (768 bytes) the bf16 GDN array - half
        // of that, 384 bytes - is served and the bf16 PLE array after it, half of a 368,640-byte fp32 PLE history,
        // is what the allocator says no to.
        ConversationCheckpoint keep = checkpoint_of(ids, StateCodec::Bf16);
        check(conversation_checkpoint_save(keep, st.ss, st.g, err), "capture: the snapshot the next call is given");
        const std::vector<uint8_t> was_gdn = keep.gdn, was_ple = keep.ple, was_tails = keep.tails,
                                   was_dead = keep.dead, was_pos = keep.block_pos;
        const std::size_t gdn_out = state_codec_size(st.z.gdn, StateCodec::Bf16);
        const std::size_t ple_out = state_codec_size(st.z.ple, StateCodec::Bf16);
        check(gdn_out < st.z.gdn && st.z.gdn <= ple_out,
              "fixture: the bar falls between the GDN and PLE output arrays, so one of THOSE is what fails here");
        err.clear();
        arm(st.z.gdn);
        bool ok = true, threw = false;
        try { ok = conversation_checkpoint_save(keep, st.ss, st.g, err); } catch (...) { threw = true; }
        disarm();
        check(!threw, "capture: a capture that cannot allocate its stored arrays is not thrown past its caller");
        check(!ok, "capture: and it is refused");
        check(refused > 0, "capture: the refusal really was the allocator saying no");
        check(refused_n >= 1 && refused_at[0] == ple_out && served_max == gdn_out,
              "capture: it failed at the PLE output array, the GDN one having already been served");
        check(!err.empty(), "capture: and it reports a reason");
        check(keep.gdn == was_gdn && keep.ple == was_ple && keep.tails == was_tails &&
              keep.dead == was_dead && keep.block_pos == was_pos &&
              keep.state_codec == (uint8_t) StateCodec::Bf16,
              "capture: a refused capture publishes nothing - the checkpoint holds what it held, in its own codec");
        check(st.bytes() == live, "capture: and the working state is untouched by the refusal");
        check(conversation_checkpoint_save(keep, st.ss, st.g, err) && keep.gdn == was_gdn && keep.ple == was_ple &&
              keep.tails == was_tails && keep.dead == was_dead && keep.block_pos == was_pos,
              "capture: with the RAM back the next capture works, and writes the snapshot the refused one did not");
    }

    {
        // The STAGING slice, which is the allocation this patch puts on the capture path: a coded copy crosses the
        // device boundary in slices of at most kStateCodecChunkFloats floats, so a bf16 capture wants one of those
        // on top of the stored arrays. It cannot be reached at the geometry the checks above use - there the PLE
        // array is refused first, as that block now says - so this is a separate fixture that moves only
        // ssm_state_size. At 256 the fp32 GDN state is 3,146,304 bytes: bigger than any array this capture sizes,
        // and under the 4 MiB limit, so the first slice is the whole state and the bar at z.gdn lands on it and on
        // nothing before it.
        State big(256);
        const std::size_t staging = std::min(kStateCodecChunkFloats, big.z.gdn / 4) * 4;
        const std::size_t largest_out =
            std::max({state_codec_size(big.z.gdn, StateCodec::Bf16), state_codec_size(big.z.ple, StateCodec::Bf16),
                      state_codec_size(2 * big.z.tail, StateCodec::Bf16),
                      state_codec_size(2 * big.z.dead, StateCodec::Bf16), (std::size_t) 2 * big.z.block_pos});
        check(big.z.gdn == 3146304, "fixture: the larger session's fp32 GDN state is the size this check is sized on");
        check(largest_out < big.z.gdn && staging == big.z.gdn && staging < kStateCodecChunkFloats * 4,
              "fixture: the bar is above every stored array and lands exactly on the staging slice");
        ConversationCheckpoint keep = checkpoint_of(ids, StateCodec::Bf16);
        check(conversation_checkpoint_save(keep, big.ss, big.g, err), "capture: the larger snapshot to be refused");
        const std::vector<uint8_t> was_gdn = keep.gdn, was_ple = keep.ple, was_tails = keep.tails,
                                   was_dead = keep.dead, was_pos = keep.block_pos;
        const std::vector<uint8_t> live_big = big.bytes();
        err.clear();
        arm(big.z.gdn);
        bool ok = true, threw = false;
        try { ok = conversation_checkpoint_save(keep, big.ss, big.g, err); } catch (...) { threw = true; }
        disarm();
        check(!threw, "capture: a capture that cannot allocate its staging slice is not thrown past its caller");
        check(!ok, "capture: and it is refused");
        check(refused_n == 1 && refused_at[0] == staging,
              "capture: the one refusal was the staging slice, at exactly its own size");
        check(served_max == state_codec_size(big.z.gdn, StateCodec::Bf16),
              "capture: every stored array had been served by then, so no output allocation failed instead");
        check(!err.empty(), "capture: and it reports a reason");
        check(keep.gdn == was_gdn && keep.ple == was_ple && keep.tails == was_tails &&
              keep.dead == was_dead && keep.block_pos == was_pos &&
              keep.state_codec == (uint8_t) StateCodec::Bf16,
              "capture: a refused capture publishes nothing - the checkpoint holds what it held, in its own codec");
        check(big.bytes() == live_big, "capture: and the working state is untouched by the refusal");
        check(conversation_checkpoint_save(keep, big.ss, big.g, err) && keep.gdn == was_gdn &&
              keep.ple == was_ple && keep.tails == was_tails && keep.dead == was_dead &&
              keep.block_pos == was_pos,
              "capture: with the staging slice allocatable again the next capture works, byte for byte");
    }

    {
        // the same on the first checkpoint this session would ever capture, with nothing allocatable at all
        ConversationCheckpoint fresh = checkpoint_of(ids, StateCodec::Bf16);
        arm(1);
        bool ok = true, threw = false;
        try { ok = conversation_checkpoint_save(fresh, st.ss, st.g, err); } catch (...) { threw = true; }
        disarm();
        check(!threw && !ok && fresh.gdn.empty() && fresh.ple.empty() && fresh.tails.empty() &&
              fresh.dead.empty() && fresh.block_pos.empty(),
              "capture: with nothing allocatable the checkpoint is left empty, not half captured");
        check(st.bytes() == live, "capture: the working state survives that refusal too");
        // and the snapshots that were already valid are still valid: the refused capture took nothing from them
        check(conversation_checkpoint_restore(bf16, st.ss, st.g, err) &&
              conversation_checkpoint_restore(fp32, st.ss, st.g, err),
          "capture: the checkpoints that were already in the chain still restore after a refused one");
    }
    {
        // the default path under the same pressure: refused at its first stored array, and the fp32 bytes it already
        // had are not disturbed. Fp32 has no staging slice at all - it is the single copy it has always been.
        ConversationCheckpoint a = checkpoint_of(ids, StateCodec::Fp32);
        check(conversation_checkpoint_save(a, st.ss, st.g, err), "capture: an fp32 checkpoint to compare against");
        ConversationCheckpoint b = checkpoint_of(ids, StateCodec::Fp32);
        err.clear();
        arm(st.z.gdn);   // the fp32 GDN array it would have to size, which is the first thing it asks for
        bool ok = true, threw = false;
        try { ok = conversation_checkpoint_save(b, st.ss, st.g, err); } catch (...) { threw = true; }
        disarm();
        check(!threw && !ok && b.gdn.empty() && b.state_codec == 0,
              "capture: a refused fp32 capture publishes nothing either");
        check(refused_n >= 1 && refused_at[0] == st.z.gdn && served_max == 0,
              "capture: and it failed at the fp32 GDN output array, having been served nothing");
        check(a.gdn == st.gdn && a.state_codec == 0, "capture: and the fp32 snapshot it did not touch is intact");
        // the fp32 capture is still, and still only, the bytes it has always been
        check(conversation_checkpoint_save(b, st.ss, st.g, err) && b.state_codec == 0 && b.gdn == a.gdn &&
              b.ple == a.ple && b.tails == a.tails && b.dead == a.dead && b.block_pos == a.block_pos,
              "capture: two fp32 captures of one state are the same bytes");
    }

    std::printf("conversation_capture_codec_test: %d checks passed\n", checks);
    return 0;
}