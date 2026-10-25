// Strata. Experts per word: the "faster mode" rule of the DeepSeek V4.1 engine.  Header only, no GPU code (tests/dsv41/experts_rule_test.cpp).
//
// The model routes every word to its top 6 experts.  On a PC with a small graphics card the CPU lane and the SSD set the speed, and
// each expert dropped saves their bytes: the router's top N are kept and their weights renormalised (the same selection the full
// model makes, a shorter list).  The default is 5 where the graphics memory is 16 GiB or less, or the GPU shares the PC's RAM, the model's own count elsewhere;
// --experts N says it outright (--experts 6: full quality), DS4_TOPK=N is the test override over everything.
#pragma once

#include <cstdint>
#include <cstdlib>

namespace strata::ds4 {

constexpr uint64_t kFasterModeVramMax = 16ull << 30;   // GiB of graphics memory in all, at most
constexpr int kFasterModeExperts = 5;

// requested: --experts (-1: not given, the rule decides); vram_bytes: the total memory of the cards the engine uses;
// model_k: the experts a word takes in the model; integrated: a GPU that shares the PC's RAM (Strix Halo: part of the experts live only on
// the SSD, each expert dropped is less read - measured 5 against 6 on a Radeon 8065S: decode +11..13%, prompts +5..7%).  The result is in 1 .. model_k.
inline int experts_per_word(int requested, uint64_t vram_bytes, int model_k, bool integrated = false) {
    if (model_k <= 1) return model_k;
    int n = requested;
    if (n < 1) n = integrated || (vram_bytes != 0 && vram_bytes <= kFasterModeVramMax) ? kFasterModeExperts : model_k;
    if (n > model_k) n = model_k;
    return n < 1 ? 1 : n;
}

// Prompt chunk (--prefill-chunk) default, one rule decided once at load from bytes (no timing).  The chunk's buffers come out of the VRAM the
// expert tier would have, and every chunk streams the experts the cards do not hold once, so a bigger chunk is fewer passes.  Measured (8K prompt):
//   - the GPUs hold most of the experts (4 x R9700, share 0.86): 1024 370 tok/s, 8192 214  -> small chunk;
//   - the experts are read in place and the CPU lane streams most of them (RTX 5070 / 3060 12 GB, 1 x R9700): 8192 best (5070 244 against 186
//     at 1024, 3060 228 against 175, 1 x R9700 431 against 200) -> the big chunk (the GGUF pack's long-standing default);
//   - the experts copied into RAM (--mmap-experts off): H100 32 GiB 2048 502, 4096 538, 8192 495 -> 4096 (not measured on HIP: no).
// Reference KL against the original model does not move with the chunk (1024 0.1705, 2048 0.1692, 4096 0.1688, 8192 0.1695).
// Alloc failure halves the chunk at load (4096 -> 2048 -> 1024 ...), KV and score buffers are allocated whole at the full --max-context.
constexpr uint64_t kChunk4096MinVram = 10ull << 30;
constexpr uint64_t kChunkInPlace4096Below = 16ull << 30;
// own_default: a single-card run (no --threads, tp 1); gpu_share: the share of the routed experts the cards can hold (1 = all);
// gguf_copy: a GGUF pack with --mmap-experts off; file_experts: the experts stay in the files; hip: a HIP build; min_vram_bytes: the
// smallest card; big: the chunk of the streamed case (the engine's kVlRows, 8192)
inline int prefill_chunk_default(bool own_default, double gpu_share, bool gguf_copy, bool file_experts, bool hip, uint64_t min_vram_bytes,
                                 int big = 8192) {
    if (own_default && gpu_share >= 0.5) return 1024;
    if (gguf_copy && !hip && min_vram_bytes >= kChunk4096MinVram) return 4096;
    // the experts read in place on a card under 16 GiB (the 12 GB PCs; not measured on HIP): 4096 keeps ~120 more experts resident than 8192 (RTX 5070
    // 287 against 165, RTX 3060 386 against 277) - 5 interleaved pairs, 4096 against 8192, decode: 5070 agent x1.020 (5/5) code x1.004 (3/5), 3060
    // agent x1.018 (4/5) code x1.091 (5/5); an agent turn of 3K new tokens on a 16K context takes the same time (5070 18.9-19.2 s against 19.1-19.4 s,
    // 3060 21.4 s against 20.9-21.4 s) - for 9% less speed on a cold 8K prompt (5070 223 against 244 tok/s)
    if (file_experts && !hip && min_vram_bytes != 0 && min_vram_bytes < kChunkInPlace4096Below) return 4096;
    return file_experts ? big : 2048;
}

// --mmap-experts auto copies all the CPU lane's experts into RAM at load (no demand store) only when they fit: lane_bytes of experts,
// ram_avail the RAM this process can still get (~0: unknown -> no), ram_margin left to the rest, commit_room what Windows still lets it
// commit beside its margin (1e300: no such limit, Linux).  All or none: a lane that does not fit both leaves the demand store as it was.
// Measured 1 x R9700 + 251 GB, 2.00 pack: decode +10-25% (the copies sit on huge pages, the store's page-locked arena does not).
inline bool copy_all_fits(double lane_bytes, uint64_t ram_avail, double ram_margin, double commit_room) {
    if (!(lane_bytes > 0.0) || ram_avail == ~uint64_t{0}) return false;
    return lane_bytes + ram_margin <= (double) ram_avail && lane_bytes <= commit_room;
}

// DS4_TOPK=N (a test, not for users): over the flag and the rule; returns the count to use
inline int experts_test_override(int chosen, int model_k) {
    if (const char* v = std::getenv("DS4_TOPK")) {
        const int n = std::atoi(v);
        if (n >= 1 && n <= model_k) return n;
    }
    return chosen;
}

}  // namespace strata::ds4
