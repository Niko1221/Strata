// The prompt-end checkpoint budget above 8 slots (ink_ckpt_lru.h), no GPU.
#include "ink_ckpt_lru.h"
#include <cstdio>
#include <cstdlib>

namespace {
int failures = 0;
void check(bool ok, const char* what) {
    if (!ok) { std::fprintf(stderr, "FAIL: %s\n", what); ++failures; }
}
}  // namespace

int main() {
    using ink2::ink_prompt_ckpt_slots;
    using ink2::ink_prompt_ckpt_victim;
    check(ink_prompt_ckpt_slots(1, nullptr) == 1, "1 slot keeps its checkpoint");
    check(ink_prompt_ckpt_slots(8, "0") == 8, "8 slots ignore the switch, as before");
    check(ink_prompt_ckpt_slots(16, nullptr) == 0, "16 slots: none unless asked (as before)");
    check(ink_prompt_ckpt_slots(16, "") == 0, "empty switch: none");
    check(ink_prompt_ckpt_slots(16, "0") == 0, "0: none");
    check(ink_prompt_ckpt_slots(16, "-3") == 0, "negative: none");
    check(ink_prompt_ckpt_slots(32, "12") == 12, "32 slots, 12 asked");
    check(ink_prompt_ckpt_slots(16, "64") == 16, "never more than the slots");

    // 4 slots may hold one, slots 0..5; uses: 5 is the oldest
    std::vector<bool> valid = {true, true, false, true, true, true};
    std::vector<uint64_t> use = {10, 11, 0, 7, 12, 3};
    check(ink_prompt_ckpt_victim(valid, use, 2, 4) == 5, "five held, slot 2 saves: the least recently used goes first");
    valid[5] = false;
    check(ink_prompt_ckpt_victim(valid, use, 2, 4) == 3, "four held, cap 4: the next oldest goes");
    valid[3] = false;
    check(ink_prompt_ckpt_victim(valid, use, 2, 4) == -1, "three held: nothing goes");
    check(ink_prompt_ckpt_victim(valid, use, 0, 3) == -1, "a slot replacing its own checkpoint does not count itself");
    check(ink_prompt_ckpt_victim(valid, use, 0, 2) == 1, "... and never gives up its own");
    std::vector<bool> none(6, false);
    check(ink_prompt_ckpt_victim(none, use, 0, 1) == -1, "nothing held: nothing goes");
    if (failures == 0) std::puts("ink_ckpt_lru_test: OK");
    return failures == 0 ? 0 : 1;
}
