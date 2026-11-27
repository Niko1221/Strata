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
    const Sm120Dev r4060ti16 = dev("NVIDIA GeForce RTX 4060 Ti", 8, 9, 34, 16);
    const Sm120Dev r3060 = dev("NVIDIA GeForce RTX 3060", 8, 6, 28, 12);
    const Sm120Dev a4000 = dev("NVIDIA RTX A4000", 8, 6, 48, 16);
    const Sm120Dev rtx4090 = dev("NVIDIA GeForce RTX 4090", 8, 9, 128, 24);
    const Sm120Dev turing = dev("NVIDIA GeForce RTX 2080 Ti", 7, 5, 68, 11);
    const Sm120Dev spark = dev("NVIDIA GB10", 12, 1, 48, 120);
    const Sm120Dev unknown = dev("NVIDIA Something Blackwell", 12, 0, 99, 40);
    const Sm120Dev lap5090 = dev("NVIDIA GeForce RTX 5090 Laptop GPU", 12, 0, 82, 23);      // the card as the engine sees it (23,975 MiB, 82 SMs)
    const Sm120Dev lap5090_hami = dev("NVIDIA GeForce RTX 5090 Laptop GPU", 12, 0, 82, 22);  // an Olares app: the HAMi limit (nvidia.com/gpumem 22 GiB) is the total

    // ---- the table: matching by name, SMs and VRAM ----
    CHECK(ink_sm120_match(r5090) && std::string(ink_sm120_match(r5090)->id) == "5090" && ink_sm120_match(r5090)->measured);
    CHECK(ink_sm120_match(r5060ti) && std::string(ink_sm120_match(r5060ti)->id) == "5060ti-16" && ink_sm120_match(r5060ti)->measured);
    CHECK(ink_sm120_match(r5060ti8) && std::string(ink_sm120_match(r5060ti8)->id) == "5060ti-8" && !ink_sm120_match(r5060ti8)->measured);
    CHECK(ink_sm120_match(r5070) && std::string(ink_sm120_match(r5070)->id) == "5070" && ink_sm120_match(r5070)->measured);
    CHECK(ink_sm120_match(r5070ti) && std::string(ink_sm120_match(r5070ti)->id) == "5070ti");   // not the 5070 row: the name is a prefix of it
    CHECK(ink_sm120_match(r5080) && std::string(ink_sm120_match(r5080)->id) == "5080");
    CHECK(ink_sm120_match(pro6000) && std::string(ink_sm120_match(pro6000)->id) == "pro6000" && ink_sm120_match(pro6000)->measured);
    CHECK(ink_sm120_match(a4000) && std::string(ink_sm120_match(a4000)->id) == "a4000" && !ink_sm120_match(a4000)->measured);   // the Ada / Ampere rows (NV-ADA-AMPERE)
    CHECK(ink_sm120_match(rtx4090) && std::string(ink_sm120_match(rtx4090)->id) == "4090" && ink_sm120_match(rtx4090)->cc == 89);
    CHECK(!ink_sm120_match(dev("NVIDIA GeForce RTX 4090", 12, 0, 128, 24)));   // a row matches only its own family
    CHECK(!ink_sm120_match(dev("NVIDIA GeForce RTX 5090", 8, 9, 170, 32)));
    CHECK(ink_sm120_match(spark) && std::string(ink_sm120_match(spark)->id) == "gb10" && ink_sm120_match(spark)->measured && ink_sm120_match(spark)->cc_minor == 1);   // cc 12.1 is its own row
    CHECK(!ink_sm120_match(dev("NVIDIA GB10", 12, 0, 48, 120)) && !ink_sm120_match(dev("NVIDIA GB10", 12, 1, 170, 32)));   // the GB10 row needs cc 12.1 and its silicon
    CHECK(!ink_sm120_match(dev("NVIDIA GeForce RTX 5090", 12, 1, 170, 32)));                       // and an RTX name on cc 12.1 gets no row (the RTX rows are cc 12.0)
    CHECK(!ink_sm120_match(unknown));
    // OLARES-0223: the laptop 5090 has its own row (name + 78-86 SMs + 20-25 GiB, so a HAMi-limited 22 GiB total still matches); the desktop 5090 row never takes it and it never takes a desktop 5090
    CHECK(ink_sm120_match(lap5090) && std::string(ink_sm120_match(lap5090)->id) == "5090m" && ink_sm120_match(lap5090)->measured && !ink_sm120_match(lap5090)->default_on);
    CHECK(ink_sm120_match(lap5090_hami) && std::string(ink_sm120_match(lap5090_hami)->id) == "5090m");
    CHECK(!ink_sm120_match(dev("NVIDIA GeForce RTX 5090 Laptop GPU", 12, 0, 82, 19)) && !ink_sm120_match(dev("NVIDIA GeForce RTX 5090 Laptop GPU", 12, 0, 82, 32)));
    CHECK(std::string(ink_sm120_match(r5090)->id) == "5090");
    {
        const Sm120Decision l0 = decide(lap5090, {});                                           // measured, default off: the old path, nothing printed
        CHECK(l0.mode == Sm120Decision::Off && l0.parts == 0 && l0.note.empty() && !l0.asked);
        const Sm120Decision l1 = decide(lap5090_hami, {{"STRATA_INK_SM120", "1"}});           // one switch away: the big set without MOE_OPT and without TC_ATTN
        CHECK(l1.mode == Sm120Decision::Fast && l1.row && std::string(l1.row->id) == "5090m" && l1.parts == kSm120Laptop5090Fast);
        CHECK(!(l1.parts & P_MOE_OPT) && !(l1.parts & P_TC_ATTN) && (l1.parts & P_PF_ATTN) && (l1.parts & P_FP8_TRUNK) && (l1.parts & P_DSPARK_OPT) && (l1.parts & P_C64_SPEC));
        const Sm120Decision l2 = decide(lap5090, {{"STRATA_INK_SM120_EXACT", "1"}});           // exact = the old path's tokens: the exact set is empty
        CHECK(l2.mode == Sm120Decision::Exact && l2.parts == 0);
    }
    // the right name with the wrong silicon does not match (a 5090 name on a 36-SM, 16 GB device)
    CHECK(!ink_sm120_match(dev("NVIDIA GeForce RTX 5090", 12, 0, 36, 16)));
    CHECK(ink_sm120_row_by_id("5090") && ink_sm120_row_by_id("5060ti-16") && !ink_sm120_row_by_id("nope"));
    // the two measured rows only
    {
        int n = 0, measured = 0;
        const Sm120Row* rows = ink_sm120_rows(&n);
        for (int i = 0; i < n; ++i) if (rows[i].measured) ++measured;
        // 11-26 (coordinator, Spark proof): no row's EXACT set holds a part that changes the drafting policy or the tokens (DSPARK_OPT swaps the window numbers and the verify tree;
        // exact without it equals off on every C1 prompt with DFlash 2)
        const uint32_t kNotInExact = P_DSPARK_OPT | P_PLD | P_CORE_GRAPH | P_C64_SPEC | P_TC_ATTN | P_SPEC_ADAPT | P_COMPACT | P_PACK | P_PREFILL | P_MOE_OPT | P_DENSE_OPT | P_PF_ATTN | P_FP8_TRUNK;
        for (int i = 0; i < n; ++i) CHECK((rows[i].exact_parts & kNotInExact) == 0);
        // 11-27 (NV-ADA): no row of 15 GiB or more carries COMPACT_HYBRID (it lets COMPACT run at 2..15 slots; a 16 GB card then died at the start with 3 MiB free)
        for (int i = 0; i < n; ++i) if (rows[i].vram_gib_min >= 15) CHECK((rows[i].fast_parts & P_COMPACT_HYBRID) == 0);
        CHECK(measured == 14);   // 5090, 5060 Ti 16, 5080, 5070 Ti, 5070, 5060, RTX PRO 6000 / 5000 / 4500 / 4000 (rows lane 11-26), DGX Spark GB10 (11-27), RTX 4060 Ti 16 GB and RTX 3060 12 GB (NV-ADA-AMPERE 11-27; both default_on = false)
        // the placeholders for the cards still to rent are in the table
        CHECK(ink_sm120_row_by_id("5070") && ink_sm120_row_by_id("5050"));
    }

    CHECK(kSm120FastDefault);   // the owner's 11-26 decision: fast mode is the default on the measured rows
    // ---- nothing set: the old path on every card without a measured row, and nothing printed ----
    for (const Sm120Dev* d : {&a4000, &rtx4090, &unknown, &r5080, &r5070ti, &r5060ti8}) {   /* measured rows with default_on false (5080, 5070 Ti) keep the old path too; the Ada / Ampere rows are unmeasured too */
        const Sm120Decision x = decide(*d, {});
        CHECK(x.mode == Sm120Decision::Off && x.parts == 0 && x.note.empty() && !x.asked && !x.refused);
    }
    // ---- rows lane 11-26: the 12 GB card (COMPACT_HYBRID, FP16 KV) and the PRO cards (the 5090 set + TC_ATTN) are default on; the 16 GB 5080 / 5070 Ti are measured but default off ----
    {
        const Sm120Decision h = decide(r5070, {});
        CHECK(h.mode == Sm120Decision::Fast && h.default_on_applied && (h.parts & P_COMPACT) && (h.parts & P_COMPACT_HYBRID) && !(h.parts & P_COMPACT_KV) && (h.parts & P_COMPACT_XCODE));
        CHECK(!(h.parts & (P_PACK | P_PREFILL | P_PLD | P_CORE_GRAPH)));
        const Sm120Decision h2 = decide(r5070, {{"STRATA_INK_SM120", "1"}, {"STRATA_INK_SM120_COMPACT_KV", "1"}});   // the FP8 KV only when named
        CHECK((h2.parts & P_COMPACT_KV));
        const Sm120Decision p = decide(pro6000, {});
        CHECK(p.mode == Sm120Decision::Fast && p.default_on_applied && (p.parts & P_TC_ATTN) && (p.parts & P_FP8_TRUNK) && !(p.parts & P_COMPACT));
        const Sm120Decision q = decide(r5090, {});
        CHECK((q.parts & P_TC_ATTN) && q.parts == kSm120Fast5090);                            // the 5090 row: the big set + TC_ATTN (FASTQ equal, 11-26)
        const Sm120Decision e = decide(r5080, {{"STRATA_INK_SM120", "1"}});                   // one switch away
        CHECK(e.mode == Sm120Decision::Fast && (e.parts & P_COMPACT) && !e.default_on_applied);
    }
    // ---- nothing set on the two measured rows: their fast set is the default, with one line saying so ----
    {
        const Sm120Decision a = decide(r5090, {});
        CHECK(a.mode == Sm120Decision::Fast && a.default_on_applied && a.asked && a.row && std::string(a.row->id) == "5090" && !a.note.empty());
        CHECK((a.parts & P_FP8_TRUNK) && (a.parts & P_MOE_OPT) && (a.parts & P_PF_ATTN) && (a.parts & P_DSPARK_OPT) && (a.parts & P_TC_ATTN) && a.parts == kSm120Fast5090 && !(a.parts & P_COMPACT));
        const Sm120Decision a0 = decide(r5090, {{"STRATA_INK_SM120_TC_ATTN", "0"}});   // the long-context attention can be taken out by name
        CHECK(a0.mode == Sm120Decision::Fast && !(a0.parts & P_TC_ATTN) && (a0.parts | P_TC_ATTN) == kSm120Fast5090);
        const Sm120Decision b = decide(r5060ti, {});
        CHECK(b.mode == Sm120Decision::Fast && b.default_on_applied && b.row && std::string(b.row->id) == "5060ti-16");
        CHECK((b.parts & P_COMPACT) && (b.parts & P_COMPACT_XCODE) && (b.parts & P_COMPACT_KV) && !(b.parts & P_FP8_TRUNK) && !(b.parts & P_FP8_INPLACE));
        // =0 is the old path, silently; a part off alone edits the default set; exact is the bit-identical set; ROW alone changes nothing
        const Sm120Decision c = decide(r5060ti, {{"STRATA_INK_SM120", "0"}});
        CHECK(c.mode == Sm120Decision::Off && c.parts == 0 && c.note.empty());
        const Sm120Decision c2 = decide(r5060ti, {{"STRATA_INK_RTX5060TI", "0"}});   // the old name works as an alias
        CHECK(c2.mode == Sm120Decision::Off && c2.parts == 0);
        const Sm120Decision d1 = decide(r5060ti, {{"STRATA_INK_SM120_COMPACT", "0"}});
        CHECK(d1.mode == Sm120Decision::Fast && !(d1.parts & P_COMPACT) && !(d1.parts & kSm120CompactSubParts) && (d1.parts & P_TC_SMALLM));
        const Sm120Decision e1 = decide(r5060ti, {{"STRATA_INK_SM120_EXACT", "1"}});
        CHECK(e1.mode == Sm120Decision::Exact && e1.parts == P_L2PF);
        const Sm120Decision e2 = decide(r5090, {{"STRATA_INK_RTX5090_EXACT", "1"}});
        CHECK(e2.mode == Sm120Decision::Exact && e2.parts == 0);   // exact never changes the drafting policy
        const Sm120Decision f1 = decide(r5080, {{"STRATA_INK_SM120_ROW", "5060ti-16"}});   // a row id alone does not turn anything on
        CHECK(f1.mode == Sm120Decision::Off && f1.parts == 0);
    }

    // ---- fast mode on the measured cards ----
    {
        const Sm120Decision x = decide(r5090, {{"STRATA_INK_SM120", "1"}});
        CHECK(x.mode == Sm120Decision::Fast && x.row && std::string(x.row->id) == "5090" && x.note.empty());
        CHECK((x.parts & P_FP8_TRUNK) && (x.parts & P_MOE_OPT) && (x.parts & P_DSPARK_OPT) && !(x.parts & P_COMPACT) && !(x.parts & P_PLD));
        const Sm120Decision y = decide(r5060ti, {{"STRATA_INK_SM120", "1"}});
        CHECK(y.mode == Sm120Decision::Fast && y.row && std::string(y.row->id) == "5060ti-16");
        // the 5060 Ti fast set = COMPACT (both halves) without any FP8 trunk (FASTQ: FP8_TRUNK=0 passes); the 5090 never had COMPACT
        CHECK((y.parts & P_COMPACT) && (y.parts & kSm120CompactSubParts) == kSm120CompactSubParts && !(y.parts & P_FP8_TRUNK) && (y.parts & P_C64_SPEC) && !(y.parts & P_MOE_OPT) && !(y.parts & P_PF_ATTN));
        // asking for COMPACT by name turns on both halves; each half can be named off apart
        const Sm120Decision c1 = decide(r5060ti, {{"STRATA_INK_SM120", "1"}, {"STRATA_INK_SM120_COMPACT", "1"}});
        CHECK((c1.parts & P_COMPACT) && (c1.parts & P_COMPACT_XCODE) && (c1.parts & P_COMPACT_KV));
        const Sm120Decision c2 = decide(r5060ti, {{"STRATA_INK_SM120", "1"}, {"STRATA_INK_SM120_COMPACT", "1"}, {"STRATA_INK_SM120_COMPACT_KV", "0"}});
        CHECK((c2.parts & P_COMPACT) && (c2.parts & P_COMPACT_XCODE) && !(c2.parts & P_COMPACT_KV));
        const Sm120Decision c3 = decide(r5060ti, {{"STRATA_INK_SM120", "1"}, {"STRATA_INK_5060TI_COMPACT", "1"}, {"STRATA_INK_SM120_COMPACT_XCODE", "0"}});
        CHECK((c3.parts & P_COMPACT) && !(c3.parts & P_COMPACT_XCODE) && (c3.parts & P_COMPACT_KV));
        const Sm120Decision c4 = decide(r5060ti, {{"STRATA_INK_SM120", "1"}, {"STRATA_INK_SM120_COMPACT", "0"}, {"STRATA_INK_SM120_COMPACT_KV", "1"}});   // a half without COMPACT does nothing
        CHECK(!(c4.parts & P_COMPACT) && !(c4.parts & P_COMPACT_KV));
        CHECK(y.row->fp8_mode == Sm120Fp8::Cache && x.row->fp8_mode == Sm120Fp8::Cache);   // one FP8 trunk (the 5090's) on every card
        CHECK(!(y.parts & P_FP8_INPLACE) && !(x.parts & P_FP8_INPLACE));                    // the in-place one is explicit-only
        const Sm120Decision ip = decide(r5060ti, {{"STRATA_INK_SM120", "1"}, {"STRATA_INK_SM120_FP8_INPLACE", "1"}});
        CHECK(ip.parts & P_FP8_INPLACE);
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
        CHECK(g.mode == Sm120Decision::Fast && !(g.parts & P_FP8_TRUNK) && (g.parts & P_TC_SMALLM));
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
    for (const Sm120Dev* d : {&turing}) {   // (a card of no table family; the Ada / Ampere cards have rows now)
        const Sm120Decision x = decide(*d, {{"STRATA_INK_RTX5060TI", "1"}});
        CHECK(x.mode == Sm120Decision::Off && x.parts == 0 && x.refused && !x.note.empty());
        CHECK(x.note.find("not an sm_120 card") != std::string::npos && x.note.find(d->name) != std::string::npos);
        const Sm120Decision y = decide(*d, {{"STRATA_INK_SM120_FP8_TRUNK", "1"}});   // a part switch alone is refused the same way
        CHECK(y.parts == 0 && y.refused);
        const Sm120Decision z = decide(*d, {{"STRATA_INK_SM120", "0"}});             // an explicit 0 is silent
        CHECK(z.parts == 0 && z.note.empty());
    }
    // ---- Ada / Ampere: the unmeasured rows keep the old path with a note; ROW forces one; the parts that need sm_120 (FP4 kernels) or sm_89 (cublasLt FP8) are masked ----
    {
        const Sm120Decision x = decide(rtx4090, {{"STRATA_INK_SM120", "1"}});
        CHECK(x.mode == Sm120Decision::Off && x.parts == 0 && x.refused && x.note.find("RTX 40") != std::string::npos && x.note.find("placeholder") != std::string::npos);
        const Sm120Decision y = decide(rtx4090, {{"STRATA_INK_SM120", "1"}, {"STRATA_INK_SM120_ROW", "4090"}});
        CHECK(y.mode == Sm120Decision::Fast && y.forced_row && (y.parts & P_FP8_TRUNK) && (y.parts & P_TC_ATTN) && !(y.parts & P_TC_SMALLM));
        const Sm120Decision z = decide(a4000, {{"STRATA_INK_SM120", "1"}, {"STRATA_INK_SM120_ROW", "5060ti-16"}});   // a 5060 Ti row on an A4000: no FP4 kernels, no FP8
        CHECK(z.mode == Sm120Decision::Fast && (z.parts & P_COMPACT) && !(z.parts & P_TC_SMALLM) && !(z.parts & P_PREFILL) && !(z.parts & P_L2PF) && !(z.parts & P_FP8_TRUNK));
        const Sm120Decision w = decide(a4000, {{"STRATA_INK_SM120_FP8_TRUNK", "1"}, {"STRATA_INK_SM120_ROW", "a4000"}});   // an explicit FP8 trunk on sm_86 is ignored, with a note
        CHECK(!(w.parts & P_FP8_TRUNK) && !w.note.empty());
        // NV-ADA-AMPERE 11-27: the 4060 Ti 16 GB and 3060 12 GB rows are measured but not default; their set has no PACK and no PF_ATTN; a 16 GB row has no COMPACT_HYBRID, a 12 GB row has it
        CHECK(ink_sm120_match(r4060ti16) && std::string(ink_sm120_match(r4060ti16)->id) == "4060ti-16" && ink_sm120_match(r4060ti16)->measured && !ink_sm120_match(r4060ti16)->default_on);
        CHECK(ink_sm120_match(r3060) && std::string(ink_sm120_match(r3060)->id) == "3060" && ink_sm120_match(r3060)->measured && !ink_sm120_match(r3060)->default_on);
        CHECK(decide(r4060ti16, {}).mode == Sm120Decision::Off && decide(r3060, {}).mode == Sm120Decision::Off);   // not the default
        const Sm120Decision f16 = decide(r4060ti16, {{"STRATA_INK_SM120", "1"}});    // one switch, no ROW needed (measured)
        CHECK(f16.mode == Sm120Decision::Fast && (f16.parts & P_COMPACT) && (f16.parts & P_PLD) && (f16.parts & P_CORE_GRAPH) && (f16.parts & P_TC_ATTN)
              && !(f16.parts & (P_PACK | P_PF_ATTN | P_COMPACT_HYBRID | P_DENSE_OPT | P_DSPARK_OPT | P_MOE_OPT | P_FP8_TRUNK)));
        const Sm120Decision f12 = decide(r3060, {{"STRATA_INK_SM120", "1"}});
        CHECK((f12.parts & P_COMPACT) && (f12.parts & P_COMPACT_HYBRID) && !(f12.parts & (P_PACK | P_PF_ATTN | P_FP8_TRUNK)));
        CHECK(ink_sm120_arch_parts(120) == kSm120AllPartsMask && !(ink_sm120_arch_parts(89) & P_TC_SMALLM) && (ink_sm120_arch_parts(89) & P_FP8_TRUNK) && !(ink_sm120_arch_parts(86) & P_FP8_TRUNK) && ink_sm120_arch_parts(70) == 0);
    }
    // ---- cc 12.0 cards without a measured row keep the old path, with a note; ROW forces a row at the user's risk ----
    for (const Sm120Dev* d : {&unknown, &r5060ti8}) {
        const Sm120Decision x = decide(*d, {{"STRATA_INK_SM120", "1"}});
        CHECK(x.mode == Sm120Decision::Off && x.parts == 0 && !x.note.empty() && x.refused);
    }
    {
        const Sm120Decision x = decide(r5070, {{"STRATA_INK_SM120", "1"}, {"STRATA_INK_SM120_ROW", "5060ti-16"}});
        CHECK(x.mode == Sm120Decision::Fast && x.forced_row && x.row && std::string(x.row->id) == "5060ti-16" && (x.parts & P_TC_SMALLM) && !x.note.empty());
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

    // ---- the DGX Spark (GB10, cc 12.1): its own row, measured on a GB10; never the 5090's row ----
    {
        // a switch runs the row's fast set (the big set: FP8 trunk cache, NVFP4 MoE, PF_ATTN, drafting, C64 bspec), silently (no placeholder note)
        const Sm120Decision x = decide(spark, {{"STRATA_INK_SM120", "1"}});
        CHECK(x.mode == Sm120Decision::Fast && x.row && std::string(x.row->id) == "gb10" && x.parts == kSm120BigFast && x.note.empty() && !x.refused);
        const Sm120Decision f = decide(spark, {{"STRATA_INK_SM120", "1"}, {"STRATA_INK_SM120_ROW", "gb10"}});   // naming the row is still accepted
        CHECK(f.mode == Sm120Decision::Fast && f.forced_row && f.row && std::string(f.row->id) == "gb10" && f.parts == kSm120BigFast && !f.note.empty());
        CHECK(!(f.parts & P_COMPACT) && !(f.parts & P_PACK) && !(f.parts & P_TC_ATTN) && !(f.parts & P_SPEC_ADAPT));   // no hybrid split, and the long-context parts only by name
        // the exact set is EMPTY: DSPARK_OPT changed the drafting numbers and the verify tree against off, so it is fast-only on the Spark
        const Sm120Decision e = decide(spark, {{"STRATA_INK_SM120_EXACT", "1"}});
        CHECK(e.mode == Sm120Decision::Exact && e.parts == 0 && kSm120SparkExact == 0 && !(e.parts & P_DSPARK_OPT));
        // one part off in the row's fast set
        const Sm120Decision o = decide(spark, {{"STRATA_INK_SM120", "1"}, {"STRATA_INK_SM120_MOE_OPT", "0"}});
        CHECK(o.mode == Sm120Decision::Fast && !(o.parts & P_MOE_OPT) && (o.parts & P_PF_ATTN));
        // =0 is the old path, silently
        const Sm120Decision z = decide(spark, {{"STRATA_INK_SM120", "0"}, {"STRATA_INK_SM120_ROW", "gb10"}});
        CHECK(z.mode == Sm120Decision::Off && z.parts == 0 && z.note.empty());
        // the row carries the numbers the engine reads: the FP8 cache trunk, the 5090's batched drafting, the 5090's MoE ranges
        const Sm120Row& g = *ink_sm120_row_by_id("gb10");
        CHECK(g.fp8_mode == Sm120Fp8::Cache && g.c64_mode == Sm120C64::Bspec && g.cc_minor == 1 && g.n_moe == 4);
        CHECK(g.moe[ink_sm120_moe_range(g, 1)].tile_rows == 16 && g.moe[ink_sm120_moe_range(g, 100000)].tile_rows == 128);
        // the RTX cards never take it, whatever a part switch says: the 5090 stays on its own row and an unmeasured RTX card on none
        CHECK(std::string(decide(r5090, {{"STRATA_INK_SM120", "1"}}).row->id) == "5090");
        // a placeholder that is measured later: the default applies only with measured AND default_on, so today nothing runs without a switch (the loop above)
        CHECK(g.measured);
        CHECK(g.spec_cost == 5.0f && g.spec_plain == 0.0f);   // the drafting numbers measured on the GB10 (the 5090's 8 / 1.02 lost to off on prose and chat)
        // default_on: see the FASTQ result in RESULTS.md; the unit test pins the value the table has
        CHECK(g.default_on && decide(spark, {}).mode == Sm120Decision::Fast);
        {   // no switch: the Spark's fast set is the default, with the one line; =0 is the old path; exact is the empty set
            const Sm120Decision d0 = decide(spark, {});
            CHECK(d0.default_on_applied && d0.row && std::string(d0.row->id) == "gb10" && d0.parts == kSm120BigFast && !d0.note.empty());
            CHECK(decide(spark, {{"STRATA_INK_SM120", "0"}}).mode == Sm120Decision::Off);
            CHECK(decide(spark, {{"STRATA_INK_SM120_EXACT", "1"}}).parts == 0);
        }
        // an unknown cc 12.1 card (not a GB10) gets the "no measured table row" note and the old path
        const Sm120Dev odd = dev("NVIDIA Something Else", 12, 1, 60, 24);
        const Sm120Decision u = decide(odd, {{"STRATA_INK_SM120", "1"}});
        CHECK(u.mode == Sm120Decision::Off && u.refused && !u.note.empty());
    }

    if (failed) { std::fprintf(stderr, "ink_sm120_table_test: %d failure(s)\n", failed); return 1; }
    std::printf("ink_sm120_table_test: ok\n");
    return 0;
}
