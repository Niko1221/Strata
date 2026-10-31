// The conversation-cache fields of the engine's startup INFO line: no GPU, no model.
//
// The INFO line's printf ends its fixed head at spec_min_p and emits this segment through one %s
// (startup_info.hpp).  A READ of 0 must render the bytes the pre-READ-gate printf printed, a READ above 0 adds
// conversation_cache_min_read_tokens after conversation_cache_min_tokens, and every field must land at its own
// name: the bug this pins was a format string and an argument list shared by two forms of different field
// counts, which shifted every field after the extra argument (varargs have no names).
//
// The references use the original printf formats before READ, and the intended formats with READ, rendered with
// snprintf and the same values as the helper. Distinctive values check both field positions and preserved fields.
#include <cstdarg>
#include <cstdio>
#include <cstring>
#include <string>

#include "strata/program/startup_info.hpp"

using strata::program::ConversationCacheInfoFields;
using strata::program::conversation_cache_info_fields;

static int failed = 0;
#define CHECK(x) do { if (!(x)) { std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #x); ++failed; } } while (0)

// distinctive values: every field a different number, so a shifted argument cannot pass as the right field
static ConversationCacheInfoFields sample() {
    ConversationCacheInfoFields f;
    f.mib = 111;
    f.slots = 22;
    f.min_free_mib = 333;
    f.min_tokens = 444;              // WRITE
    f.min_read_tokens = 555;         // READ
    f.disk_mib = 666;
    f.disk_only = true;
    f.disk_delta = true;
    f.similarity = 0.875;
    f.n_min = 777;
    f.tail_role_token = 888;
    f.vram_elastic = true;
    return f;
}

static std::string render(const char* format, ...) {
    char buf[512];
    va_list args;
    va_start(args, format);
    std::vsnprintf(buf, sizeof buf, format, args);
    va_end(args);
    return buf;
}

int main() {
    const ConversationCacheInfoFields f = sample();

    // the startup line before the READ dimension (3b37805e): the non-spill variant
    const std::string pre_no_spill = render(
        "conversation_cache_mib=%lld conversation_cache_slots=%d conversation_cache_min_free_mib=%lld "
        "conversation_cache_min_tokens=%lld tail_role_token=%lld vram_elastic=%d",
        (long long) f.mib, f.slots, (long long) f.min_free_mib, (long long) f.min_tokens,
        (long long) f.tail_role_token, f.vram_elastic ? 1 : 0);
    // the startup line before the READ dimension: the spill variant (the disk-tier fields between the thresholds
    // and tail_role_token)
    const std::string pre_spill = render(
        "conversation_cache_mib=%lld conversation_cache_slots=%d conversation_cache_min_free_mib=%lld "
        "conversation_cache_min_tokens=%lld conversation_cache_disk_mib=%lld conversation_cache_disk_only=%d "
        "conversation_cache_disk_delta=%d conversation_cache_similarity=%.3f conversation_cache_n_min=%lld "
        "tail_role_token=%lld vram_elastic=%d",
        (long long) f.mib, f.slots, (long long) f.min_free_mib, (long long) f.min_tokens,
        (long long) f.disk_mib, f.disk_only ? 1 : 0, f.disk_delta ? 1 : 0, f.similarity,
        (long long) f.n_min, (long long) f.tail_role_token, f.vram_elastic ? 1 : 0);
    // the READ-gate commit's spill variant: the READ field after the WRITE threshold
    const std::string read_spill = render(
        "conversation_cache_mib=%lld conversation_cache_slots=%d conversation_cache_min_free_mib=%lld "
        "conversation_cache_min_tokens=%lld conversation_cache_min_read_tokens=%lld "
        "conversation_cache_disk_mib=%lld conversation_cache_disk_only=%d conversation_cache_disk_delta=%d "
        "conversation_cache_similarity=%.3f conversation_cache_n_min=%lld "
        "tail_role_token=%lld vram_elastic=%d",
        (long long) f.mib, f.slots, (long long) f.min_free_mib, (long long) f.min_tokens,
        (long long) f.min_read_tokens, (long long) f.disk_mib, f.disk_only ? 1 : 0, f.disk_delta ? 1 : 0,
        f.similarity, (long long) f.n_min, (long long) f.tail_role_token, f.vram_elastic ? 1 : 0);
    // READ>0 without the disk tier keeps the unrelated fields that close the startup segment.
    const std::string read_no_spill = render(
        "conversation_cache_mib=%lld conversation_cache_slots=%d conversation_cache_min_free_mib=%lld "
        "conversation_cache_min_tokens=%lld conversation_cache_min_read_tokens=%lld "
        "tail_role_token=%lld vram_elastic=%d",
        (long long) f.mib, f.slots, (long long) f.min_free_mib, (long long) f.min_tokens,
        (long long) f.min_read_tokens, (long long) f.tail_role_token, f.vram_elastic ? 1 : 0);

    // READ=0, spill tier off: byte-identical to the pre-READ-gate line's middle
    ConversationCacheInfoFields a = f;
    a.min_read_tokens = 0;
    a.disk_tier_on = false;
    CHECK(conversation_cache_info_fields(a) == pre_no_spill);
    // READ=0, spill tier on: byte-identical to the pre-READ-gate line's middle
    ConversationCacheInfoFields b = f;
    b.min_read_tokens = 0;
    b.disk_tier_on = true;
    CHECK(conversation_cache_info_fields(b) == pre_spill);
    ConversationCacheInfoFields c = f;
    c.disk_tier_on = true;
    CHECK(conversation_cache_info_fields(c) == read_spill);
    ConversationCacheInfoFields d = f;
    d.disk_tier_on = false;
    CHECK(conversation_cache_info_fields(d) == read_no_spill);

    // READ=0 prints no READ field at all, and no field shows another field's value (the shifted-varargs bug
    // put min_read_tokens into conversation_cache_disk_mib and every later field one argument early)
    const std::string zero_off = conversation_cache_info_fields(a);
    CHECK(zero_off.find("conversation_cache_min_read_tokens") == std::string::npos);
    CHECK(zero_off.find("conversation_cache_disk_mib=666") == std::string::npos);   // tier off: not printed
    const std::string zero_on = conversation_cache_info_fields(b);
    CHECK(zero_on.find("conversation_cache_disk_mib=666") != std::string::npos);
    CHECK(zero_on.find("conversation_cache_min_tokens=444 ") != std::string::npos);
    CHECK(zero_on.find("tail_role_token=888") != std::string::npos);
    // Both READ>0 forms keep the READ value and the unrelated fields at their own names.
    CHECK(conversation_cache_info_fields(c).find("conversation_cache_min_read_tokens=555") != std::string::npos);
    CHECK(conversation_cache_info_fields(d).find("tail_role_token=888 vram_elastic=1") != std::string::npos);

    // the segment is space-separated with no leading, trailing or doubled space: the printf wraps it in
    // "spec_min_p=%.2f %s prefill_cap=%lld", so a stray space would show in the line
    for (const ConversationCacheInfoFields* v : {&a, &b, &c, &d}) {
        const std::string s = conversation_cache_info_fields(*v);
        CHECK(!s.empty() && s.front() != ' ' && s.back() != ' ');
        CHECK(s.find("  ") == std::string::npos);
    }

    // the similarity keeps its three decimals whatever the value
    ConversationCacheInfoFields e = b;
    e.similarity = 0.5;
    CHECK(conversation_cache_info_fields(e).find("conversation_cache_similarity=0.500") != std::string::npos);
    ConversationCacheInfoFields g = b;
    g.similarity = 0.0;
    CHECK(conversation_cache_info_fields(g).find("conversation_cache_similarity=0.000") != std::string::npos);

    if (failed == 0) std::printf("startup_info_test: cache fields, READ=0 byte-identical, field positions: PASS\n");
    return failed == 0 ? 0 : 1;
}
