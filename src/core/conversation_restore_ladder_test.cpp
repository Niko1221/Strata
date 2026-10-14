// The restore ladder's pure half (layer/ladder-t3): the ordering and the prefix/image filtering that decide which
// checkpoint of a rejected image is restored.  Host-only, no CUDA: the engine supplies the validation (running
// state and, under --layer-split, the stage parts) and the restore; here the choice is exercised with a predicate.
// The case that matters - the whole image does not validate, so the LARGEST checkpoint that DOES is restored and
// only the tail is re-read - is the third block below.
#include "strata/core/conversation_restore_ladder.hpp"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

using namespace strata::core;

namespace {

int checks = 0;
void check(bool ok, const char* label) {
    ++checks;
    if (!ok) { std::fprintf(stderr, "FAIL: %s\n", label); std::exit(1); }
}

std::vector<int32_t> ids_upto(size_t n) {
    std::vector<int32_t> v;
    v.reserve(n);
    for (size_t i = 0; i < n; ++i) v.push_back((int32_t) i + 1);
    return v;
}

ConversationCheckpoint checkpoint(std::vector<int32_t> ids, std::vector<ConversationImageKey> imgs = {}) {
    ConversationCheckpoint c;
    c.ids = std::move(ids);
    c.imgs = std::move(imgs);
    return c;
}

} // namespace

int main() {
    // A prompt of 100 tokens; the image carries running-state checkpoints at 30, 90 and 60 tokens.
    const std::vector<int32_t> ids = ids_upto(100);
    const std::vector<int64_t> prompt(ids.begin(), ids.end());
    const std::vector<ConversationImageKey> no_images;
    std::vector<ConversationCheckpoint> cps;
    cps.push_back(checkpoint(ids_upto(30)));
    cps.push_back(checkpoint(ids_upto(90)));
    cps.push_back(checkpoint(ids_upto(60)));

    // The rungs come out in descending token order, whatever the stored order was.
    {
        const auto rungs = restore_ladder_rungs(cps, prompt, no_images, 0);
        check(rungs.size() == 3, "three checkpoints are three rungs");
        check(rungs[0].tokens == 90 && rungs[0].index == 1, "the largest rung comes first");
        check(rungs[1].tokens == 60 && rungs[1].index == 2, "then 60");
        check(rungs[2].tokens == 30 && rungs[2].index == 0, "then 30");
    }
    // The floor is what the request already has without the image (its live state, a checkpoint, an idle slot): a
    // rung at or below it is no help.
    {
        const auto rungs = restore_ladder_rungs(cps, prompt, no_images, 60);
        check(rungs.size() == 1 && rungs[0].tokens == 90, "the floor keeps only the deeper rungs");
    }
    // A checkpoint that is not a STRICT prefix of the prompt is not a rung: the whole prompt (nothing to re-read),
    // or one that diverges before its end.
    {
        std::vector<ConversationCheckpoint> more = cps;
        more.push_back(checkpoint(ids_upto(100)));                       // the whole prompt, not a strict prefix
        std::vector<int32_t> foreign = ids_upto(50);
        foreign.back() = 999;                                            // diverges at the last token
        more.push_back(checkpoint(std::move(foreign)));
        const auto rungs = restore_ladder_rungs(more, prompt, no_images, 0);
        check(rungs.size() == 3, "a whole-prompt or divergent checkpoint is not a rung");
    }
    // Picture identity: a checkpoint whose embeddings differ from the request's is not a rung (equal ids are not
    // enough - conversation_prefix checks the images too).
    {
        const std::vector<ConversationImageKey> req = {{10, 0xabc}};
        std::vector<ConversationCheckpoint> imgs_cps;
        imgs_cps.push_back(checkpoint(ids_upto(90), req));               // the same picture
        imgs_cps.push_back(checkpoint(ids_upto(90)));                    // same ids, no picture
        const auto rungs = restore_ladder_rungs(imgs_cps, prompt, req, 0);
        check(rungs.size() == 1 && rungs[0].index == 0, "only the checkpoint with the same picture is a rung");
    }
    // The heart: the whole image did NOT validate, so the largest checkpoint that DOES validate is restored, and the
    // caller re-reads only the tail (prompt.size() - rung.tokens tokens).
    {
        auto usable = [](size_t index) { return index == 2; };           // 90 rejected (the image's live state), 60 ok
        RestoreLadderRung rung;
        const bool found = restore_ladder_pick(cps, prompt, no_images, 0, usable, rung);
        check(found, "a usable checkpoint is found when the whole image did not validate");
        check(rung.tokens == 60 && rung.index == 2, "the largest VALID checkpoint wins, not simply the largest");
    }
    // No rung validates -> false, and the caller keeps today's behaviour (the slot's own prefix).
    {
        RestoreLadderRung rung;
        const bool found = restore_ladder_pick(cps, prompt, no_images, 0, [](size_t) { return false; }, rung);
        check(!found, "no usable checkpoint reports false");
    }
    // Two checkpoints of the same length keep their stored order (the sort is stable).
    {
        std::vector<ConversationCheckpoint> dup;
        dup.push_back(checkpoint(ids_upto(90)));
        dup.push_back(checkpoint(ids_upto(90)));
        const auto rungs = restore_ladder_rungs(dup, prompt, no_images, 0);
        check(rungs.size() == 2 && rungs[0].index == 0 && rungs[1].index == 1, "a tie keeps the stored order");
    }
    // The flag: OFF is today's behaviour - no rung at all, even when one is usable, so the caller keeps its own
    // prefix.  This is the one place the switch is honoured, so it is testable without the serve loop.
    {
        RestoreLadderRung rung;
        auto usable = [](size_t) { return true; };   // a usable rung exists; only the switch decides
        check(!restore_ladder_plan(false, cps, prompt, no_images, 0, usable, rung),
              "flag OFF: no rung is chosen even when one is usable (today's behaviour)");
        check(restore_ladder_plan(true, cps, prompt, no_images, 0, usable, rung) && rung.tokens == 90,
              "flag ON: the largest usable rung is chosen");
    }
#if defined(_MSC_VER)
    {
        _putenv_s("STRATA_RESTORE_LADDER", "");   // whatever the shell had, start from off
        check(!restore_ladder_enabled(), "switch: unset is OFF");
        _putenv_s("STRATA_RESTORE_LADDER", "0");
        check(!restore_ladder_enabled(), "switch: '0' is OFF");
        _putenv_s("STRATA_RESTORE_LADDER", "1");
        check(restore_ladder_enabled(), "switch: '1' is ON");
        _putenv_s("STRATA_RESTORE_LADDER", "");   // leave the environment as it was found
    }
#endif

    std::printf("conversation_restore_ladder_test: %d host-only checks passed\n", checks);
    return 0;
}
