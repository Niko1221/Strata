// The RTX 50 (sm_120) switch set and the per-card table (strata/core/ink_sm120_table.hpp): no GPU needed, devices and environments are mocked.
// What it pins: the default is the old path; a switch on a non-sm_120 card is refused with one line; a card is matched by name + SMs + VRAM (not by
// "cc 12.0"); an unmeasured card keeps the old path; the old switch names work as aliases; exact mode never turns on more than fast mode.
#include "strata/core/ink_sm120_table.hpp"

#include <cstdio>
#include <map>
#include <string>

using namespace strata::core;

namespace {
int failed = 0;
#define CHECK(x) do { if (!(x)) { std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #x); ++failed; } } while (0)

using Env = std::map<std::string, std::string>;

Sm120Decision decide(const Sm120Dev& d, const Env& e) {
    return ink_sm120_decide(d, [&](const char* n) -> const char* {
        auto it = e.find(n);
        return it == e.end() ? nullptr : it->second.c_str();
    });
}

Sm120Dev dev(const char* name, int maj, int min, int sms, int gib) {
    Sm120Dev d;
    d.name = name;
    d.cc_major = maj;
    d.cc_minor = min;
    d.sm_count = sms;
    d.vram_bytes = (uint64_t)gib << 30;
    return d;
}
}  // namespace

int main() {
    const Sm120Dev r5090 = dev("NVIDIA GeForce RTX 5090", 12, 0, 170, 32);
    const Sm120Dev r5060ti = dev("NVIDIA GeForce RTX 5060 Ti", 12, 0, 36, 16);
    const Sm120Dev r5060ti8 = dev("NVIDIA GeForce RTX 5060 Ti", 12, 0, 36, 8);
    const Sm120Dev r5070 = dev("NVIDIA GeForce RTX 5070", 12, 0, 48, 12);
    const Sm120Dev r5070ti = dev("NVIDIA GeForce RTX 5070 Ti", 12, 0, 70, 16);
    const Sm120Dev r5080 = dev("NVIDIA GeForce RTX 5080", 12, 0, 84, 16);
    const Sm120Dev pro6000 = dev("NVIDIA RTX PRO 6000 Blackwell Workstation Edition", 12, 0, 188, 96);
    const Sm120Dev a4000 = dev("NVIDIA RTX A4000", 8, 6, 48, 16);
    const Sm120Dev rtx4090 = dev("NVIDIA GeForce RTX 4090", 8, 9, 128, 24);
    const Sm120Dev spark = dev("NVIDIA GB10", 12, 1, 48, 120);
    const Sm120Dev unknown = dev("NVIDIA Something Blackwell", 12, 0, 99, 40);

    // ---- the table: matching by name, SMs and VRAM ----
    CHECK(ink_sm120_match(r5090) && std::string(ink_sm120_match(r5090)->id) == "5090" && ink_sm120_match(r5090)->measured);
    CHECK(ink_sm120_match(r5060ti) && std::string(ink_sm120_match(r5060ti)->id) == "5060ti-16" && ink_sm120_match(r5060ti)->measured);
    CHECK(ink_sm120_match(r5060ti8) && std::string(ink_sm120_match(r5060ti8)->id) == "5060ti-8" && !ink_sm120_match(r5060ti8)->measured);
    CHECK(ink_sm120_match(r5070) && std::string(ink_sm120_match(r5070)->id) == "5070" && !ink_sm120_match(r5070)->measured);
    CHECK(ink_sm120_match(r5070ti) && std::string(ink_sm120_match(r5070ti)->id) == "5070ti");   // not the 5070 row: the name is a prefix of it
    CHECK(ink_sm120_match(r5080) && std::string(ink_sm120_match(r5080)->id) == "5080");
    CHECK(ink_sm120_match(pro6000) && std::string(ink_sm120_match(pro6000)->id) == "pro6000" && !ink_sm120_match(pro6000)->measured);
    CHECK(!ink_sm120_match(a4000) && !ink_sm120_match(rtx4090));    // not cc 12.0
    CHECK(!ink_sm120_match(spark));                                  // cc 12.1 is another SASS family
    CHECK(!ink_sm120_match(unknown));
    // the right name with the wrong silicon does not match (a 5090 name on a 36-SM, 16 GB device)
    CHECK(!ink_sm120_match(dev("NVIDIA GeForce RTX 5090", 12, 0, 36, 16)));
    CHECK(ink_sm120_row_by_id("5090") && ink_sm120_row_by_id("5060ti-16") && !ink_sm120_row_by_id("nope"));
    // the two measured rows only
    {
        int n = 0, measured = 0;
        const Sm120Row* rows = ink_sm120_rows(&n);
        for (int i = 0; i < n; ++i) if (rows[i].measured) ++measured;
        CHECK(measured == 2);
        // the placeholders for the cards still to rent are in the table
        CHECK(ink_sm120_row_by_id("5070") && ink_sm120_row_by_id("5050"));
    }

    // ---- nothing set: the old path on every card, and nothing printed ----
    for (const Sm120Dev* d : {&r5090, &r5060ti, &a4000, &spark, &unknown}) {
        const Sm120Decision x = decide(*d, {});
        CHECK(x.mode == Sm120Decision::Off && x.parts == 0 && x.note.empty() && !x.asked && !x.refused);
    }
    CHECK(!kSm120FastDefault);   // today's default; the one constant the owner flips after FASTQ

    // ---- fast mode on the measured cards ----
    {
        const Sm120Decision x = decide(r5090, {{"STRATA_INK_SM120", "1"}});
        CHECK(x.mode == Sm120Decision::Fast && x.row && std::string(x.row->id) == "5090" && x.note.empty());
        CHECK((x.parts & P_FP8_TRUNK) && (x.parts & P_MOE_OPT) && (x.parts & P_DSPARK_OPT) && !(x.parts & P_COMPACT) && !(x.parts & P_PLD));
        const Sm120Decision y = decide(r5060ti, {{"STRATA_INK_SM120", "1"}});
        CHECK(y.mode == Sm120Decision::Fast && y.row && std::string(y.row->id) == "5060ti-16");
        CHECK((y.parts & P_COMPACT) && (y.parts & P_FP8_TRUNK) && (y.parts & P_C64_SPEC) && !(y.parts & P_MOE_OPT) && !(y.parts & P_PF_ATTN));
        CHECK(y.row->fp8_mode == Sm120Fp8::InPlace && x.row->fp8_mode == Sm120Fp8::Cache);
        CHECK(y.row->c64_mode == Sm120C64::K8v4 && x.row->c64_mode == Sm120C64::Bspec);
    }
    // ---- the old switch names are aliases; the new name wins ----
    {
        const Sm120Decision a = decide(r5090, {{"STRATA_INK_RTX5090", "1"}});
        CHECK(a.mode == Sm120Decision::Fast && a.row && std::string(a.row->id) == "5090");
        const Sm120Decision b = decide(r5060ti, {{"STRATA_INK_RTX5060TI", "1"}});
        CHECK(b.mode == Sm120Decision::Fast && b.row && std::string(b.row->id) == "5060ti-16");
        const Sm120Decision c = decide(r5090, {{"STRATA_INK_RTX5090", "1"}, {"STRATA_INK_SM120", "0"}});
        CHECK(c.mode == Sm120Decision::Off && c.parts == 0);
        const Sm120Decision e = decide(r5090, {{"STRATA_INK_RTX5090", "1"}, {"STRATA_INK_RTX5090_EXACT", "1"}});
        CHECK(e.mode == Sm120Decision::Exact);
        const Sm120Decision f = decide(r5060ti, {{"STRATA_INK_RTX5060TI", "1"}, {"STRATA_INK_RTX5060TI_EXACT", "1"}});
        CHECK(f.mode == Sm120Decision::Exact);
        // old sub-switch names
        const Sm120Decision g = decide(r5060ti, {{"STRATA_INK_SM120", "1"}, {"STRATA_INK_5060TI_FP8_TRUNK", "0"}});
        CHECK(g.mode == Sm120Decision::Fast && !(g.parts & P_FP8_TRUNK) && (g.parts & P_COMPACT));
        const Sm120Decision h = decide(r5060ti, {{"STRATA_INK_SM120", "1"}, {"STRATA_INK_SM120_FP8_TRUNK", "0"}, {"STRATA_INK_5060TI_FP8_TRUNK", "1"}});
        CHECK(!(h.parts & P_FP8_TRUNK));   // the new name wins
    }
    // ---- exact never turns on more than fast, and drops the parts that change bits ----
    for (const Sm120Dev* d : {&r5090, &r5060ti}) {
        const Sm120Decision fast = decide(*d, {{"STRATA_INK_SM120", "1"}});
        const Sm120Decision ex = decide(*d, {{"STRATA_INK_SM120", "1"}, {"STRATA_INK_SM120_EXACT", "1"}});
        CHECK(ex.mode == Sm120Decision::Exact && (ex.parts & ~fast.parts) == 0);
        CHECK(!(ex.parts & P_FP8_TRUNK) && !(ex.parts & P_COMPACT) && !(ex.parts & P_TC_SMALLM) && !(ex.parts & P_MOE_OPT) && !(ex.parts & P_PF_ATTN));
    }
    // ---- a switch on a card that is not sm_120 is refused with one line, the default path runs ----
    for (const Sm120Dev* d : {&a4000, &rtx4090, &spark}) {
        const Sm120Decision x = decide(*d, {{"STRATA_INK_RTX5060TI", "1"}});
        CHECK(x.mode == Sm120Decision::Off && x.parts == 0 && x.refused && !x.note.empty());
        CHECK(x.note.find("not an sm_120 card") != std::string::npos && x.note.find(d->name) != std::string::npos);
        const Sm120Decision y = decide(*d, {{"STRATA_INK_SM120_FP8_TRUNK", "1"}});   // a part switch alone is refused the same way
        CHECK(y.parts == 0 && y.refused);
        const Sm120Decision z = decide(*d, {{"STRATA_INK_SM120", "0"}});             // an explicit 0 is silent
        CHECK(z.parts == 0 && z.note.empty());
    }
    // ---- cc 12.0 cards without a measured row keep the old path, with a note; ROW forces a row at the user's risk ----
    for (const Sm120Dev* d : {&r5070, &r5080, &pro6000, &unknown, &r5060ti8}) {
        const Sm120Decision x = decide(*d, {{"STRATA_INK_SM120", "1"}});
        CHECK(x.mode == Sm120Decision::Off && x.parts == 0 && !x.note.empty() && x.refused);
    }
    {
        const Sm120Decision x = decide(r5070, {{"STRATA_INK_SM120", "1"}, {"STRATA_INK_SM120_ROW", "5060ti-16"}});
        CHECK(x.mode == Sm120Decision::Fast && x.forced_row && x.row && std::string(x.row->id) == "5060ti-16" && (x.parts & P_COMPACT) && !x.note.empty());
        const Sm120Decision y = decide(r5070, {{"STRATA_INK_SM120", "1"}, {"STRATA_INK_SM120_ROW", "bogus"}});
        CHECK(y.mode == Sm120Decision::Off && y.refused);
        // an explicit part on an unknown cc 12.0 card runs that part only
        const Sm120Decision z = decide(unknown, {{"STRATA_INK_SM120_PF_ATTN", "1"}});
        CHECK(z.parts == P_PF_ATTN);
    }
    // ---- a part switch alone turns on that part, not the whole set (the 5090 hand-over's behavior) ----
    {
        const Sm120Decision x = decide(r5090, {{"STRATA_INK_SM120_PF_ATTN", "1"}});
        CHECK(x.parts == P_PF_ATTN && x.mode == Sm120Decision::Fast);
        const Sm120Decision y = decide(r5090, {{"STRATA_INK_SM120", "1"}, {"STRATA_INK_SM120_MOE_OPT", "0"}});
        CHECK(!(y.parts & P_MOE_OPT) && (y.parts & P_PF_ATTN));
    }
    // ---- the MoE ranges: tile sizes per batch range, first range whose m_max >= M ----
    {
        const Sm120Row& b = *ink_sm120_row_by_id("5090");
        CHECK(b.moe[ink_sm120_moe_range(b, 1)].tile_rows == 16 && b.moe[ink_sm120_moe_range(b, 192)].tile_rows == 16);
        CHECK(b.moe[ink_sm120_moe_range(b, 193)].tile_rows == 32 && b.moe[ink_sm120_moe_range(b, 512)].tile_rows == 32);
        CHECK(b.moe[ink_sm120_moe_range(b, 513)].tile_rows == 64 && b.moe[ink_sm120_moe_range(b, 1023)].tile_rows == 64);
        CHECK(b.moe[ink_sm120_moe_range(b, 1024)].tile_rows == 128 && b.moe[ink_sm120_moe_range(b, 100000)].tile_rows == 128);
        const Sm120Row& s = *ink_sm120_row_by_id("5060ti-16");
        CHECK(s.moe[ink_sm120_moe_range(s, 128)].tile_rows == 16 && s.moe[ink_sm120_moe_range(s, 129)].tile_rows == 32);
        CHECK(s.moe[ink_sm120_moe_range(s, 160)].tile_rows == 32 && s.moe[ink_sm120_moe_range(s, 161)].tile_rows == 64);
        CHECK(s.moe[ink_sm120_moe_range(s, 1536)].tile_rows == 64 && s.moe[ink_sm120_moe_range(s, 1537)].tile_rows == 128);
    }

    if (failed) { std::fprintf(stderr, "ink_sm120_table_test: %d failure(s)\n", failed); return 1; }
    std::printf("ink_sm120_table_test: ok\n");
    return 0;
}
