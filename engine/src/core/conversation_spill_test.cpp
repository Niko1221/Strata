// The conversation cache's SSD tier: its folder per engine process and its budget (conversation_spill.hpp).  No GPU, no model.
#include "strata/core/conversation_spill.hpp"

#include <stdexcept>
#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <mutex>
#include <thread>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <string>

using namespace strata::core;
namespace fs = std::filesystem;

namespace {
int checks = 0, failures = 0;
void check(bool ok, const char* what) {
    ++checks;
    if (!ok) { ++failures; std::fprintf(stderr, "FAIL: %s\n", what); }
}
constexpr uint64_t GiB = uint64_t(1) << 30;
void touch(const fs::path& p) { std::ofstream(p) << "x"; }
}  // namespace

int main() {
    // the budget: min(32 GiB, 25% of the free space), never into the floor
    check(spill_default_budget(400 * GiB, 4 * GiB) == 32 * GiB, "400 GiB free: 32 GiB");
    check(spill_default_budget(40 * GiB, 4 * GiB) == 10 * GiB, "40 GiB free: a quarter, 10 GiB");
    check(spill_default_budget(5 * GiB, 4 * GiB) == 1 * GiB, "5 GiB free, floor 4: the 1 GiB above the floor");
    check(spill_default_budget(4 * GiB, 4 * GiB) == 0 && spill_default_budget(3 * GiB, 4 * GiB) == 0, "at or under the floor: none");
    SpillBudget b{10 * GiB, 0, 0};
    check(b.reserve(6 * GiB) && !b.reserve(5 * GiB) && b.reserve(4 * GiB), "reservations count until the writes end");
    b.finish(6 * GiB, 5 * GiB);
    check(b.used == 5 * GiB && b.reserved == 4 * GiB && !b.over(), "a write ends: its reservation goes, its bytes stay");
    b.finish(4 * GiB, 0);
    check(b.used == 5 * GiB && b.reserved == 0, "a failed write leaves nothing");
    check(!b.reserve(6 * GiB) && b.reserve(5 * GiB), "the cap includes what is written");
    b.finish(5 * GiB, 6 * GiB);
    check(b.over(), "a file larger than its reservation shows as over the cap");
    b.drop(6 * GiB);
    check(!b.over() && b.used == 5 * GiB, "a dropped entry gives its bytes back");
    check(!SpillBudget{0, 0, 0}.reserve(1), "no budget: nothing");

    // the names a clean may touch
    check(spill_detail::is_namespace_name("123-0123456789abcdef"), "pid-16 hex");
    for (const char* n : {"123-0123456789ABCDEF", "123-0123456789abcde", "-0123456789abcdef", "12a-0123456789abcdef",
                          "notes", "123-0123456789abcdef0", "123_0123456789abcdef"})
        check(!spill_detail::is_namespace_name(n), n);

    // the folders: two live engines on one root keep each other's folder; a dead one's goes; nothing else is touched
    std::error_code ec;
    const fs::path root = fs::temp_directory_path() / ("strata-spill-test-" + std::to_string(std::random_device{}()));
    {   // --conversation-spill-persist: a stable folder per model fingerprint; entries survive a restart, LRU at the cap, a grown conversation replaces its prefix
        auto img = [](std::vector<int32_t> ids, uint8_t fill) {
            SavedConversation s;
            s.live.ids = ids;
            s.live.gdn.assign(1 << 20, fill);
            s.cvec = true;
            return s;
        };
        const fs::path proot = fs::temp_directory_path() / ("strata-spill-persist-" + std::to_string(std::random_device{}()));
        fs::remove_all(proot, ec);
        const SessionFileIdentity A{0xa1, 5}, B{0xb2, 5}, A2{0xa1, 6};
        const uint64_t cap = uint64_t(5) << 20;   // three entries (about 1 MiB each, 2 MiB reserved per write)
        std::string e;
        size_t cleaned = 0, adopted = 99;
        {   // run 0: another model leaves an entry
            ConversationSpill s;
            check(s.open(proot, cap, e, &cleaned, &B, &adopted) && s.persistent() && adopted == 0, "persist: a first run on a new root");
            check(s.put(img({90, 91}, 1), B, e) && s.size() == 1, "persist: model B's entry");
        }
        check(fs::exists(proot / spill_detail::persist_name(B.model)), "persist: the stable folder outlives the run");
        {   // run 1: model A
            ConversationSpill s;
            check(s.open(proot, cap, e, &cleaned, &A, &adopted) && s.persistent() && adopted == 0 && cleaned == 1, "persist: model A's start removes B's folder");
            check(!fs::exists(proot / spill_detail::persist_name(B.model)) && s.dir().filename() == spill_detail::persist_name(A.model), "persist: B's folder gone, A's is fp-<hex>");
            check(s.put(img({1, 2, 3, 4}, 7), A, e) && s.put(img({10, 11}, 8), A, e) && s.put(img({20, 21}, 9), A, e) && s.size() == 3, "persist: three entries written");
            ConversationSpill other;   // a second live engine of the same model: falls back to its own per-process folder
            check(other.open(proot, cap, e, nullptr, &A, &adopted) && !other.persistent() && adopted == 0 && other.dir() != s.dir(), "persist: a second engine of the model gets its own folder");
            other.close();
        }
        {   // run 2: a restart, same fingerprint
            ConversationSpill s;
            check(s.open(proot, cap, e, &cleaned, &A, &adopted) && adopted == 3 && s.size() == 3 && s.bytes() > 0, "restart: the three entries are adopted");
            auto m = s.best(std::vector<int32_t>{1, 2, 3, 4, 9}, {}, true);
            check(m.tokens == 4 && m.live, "restart: found by token prefix");
            SavedConversation back;
            check(s.take_serial(m.serial, A, {}, back, e) && back.live.ids == std::vector<int32_t>({1, 2, 3, 4}) &&
                  back.live.gdn == std::vector<uint8_t>(1 << 20, 7), "restart: read back byte for byte");
            check(s.size() == 3 && s.find(m.serial), "a read leaves the entry in place");
            // least recently USED goes at the cap: {10,11} was written after {1,2,3,4} but the read made {1,2,3,4} the freshest
            check(s.put(img({30, 31}, 4), A, e) && s.size() == 3, "LRU: a fourth at the cap");
            check(s.best(std::vector<int32_t>{10, 11, 0}, {}, true).tokens == 0, "LRU: the least recently used ({10,11}) went");
            check(s.best(std::vector<int32_t>{1, 2, 3, 4, 0}, {}, true).tokens == 4 && s.best(std::vector<int32_t>{20, 21, 0}, {}, true).tokens == 2 &&
                  s.best(std::vector<int32_t>{30, 31, 0}, {}, true).tokens == 2, "LRU: the read one and the others stay");
            // supersede: the same conversation grown replaces its prefix; an entry that is no prefix is never removed for it
            const auto old = s.best(std::vector<int32_t>{1, 2, 3, 4, 0}, {}, true);
            const fs::path oldf = s.find(old.serial)->file;
            check(s.put(img({1, 2, 3, 4, 5, 6}, 6), A, e) && s.size() == 3 && !fs::exists(oldf) && !s.find(old.serial), "supersede: P+Q replaces P");
            check(s.best(std::vector<int32_t>{20, 21, 0}, {}, true).tokens == 2 && s.best(std::vector<int32_t>{30, 31, 0}, {}, true).tokens == 2,
                  "supersede: entries that are no prefix stay");
            check(s.best(std::vector<int32_t>{1, 2, 3, 4, 5, 6, 0}, {}, true).tokens == 6, "supersede: the grown one serves the resume");
        }
        {   // a snapshot parked in RAM that extends a spilled one deletes it (file, index file, entry) at once; a restart finds the newest
            ConversationSpill s;
            std::string e2;
            size_t ad = 0;
            check(s.open(proot, cap, e2, nullptr, &A, &ad), "drop: open");
            while (s.size() > 0) s.drop(0);
            check(s.put(img({1, 2, 3}, 1), A, e2) && s.put(img({50, 51}, 2), A, e2) && s.size() == 2, "drop: two spilled");
            const auto old = s.best(std::vector<int32_t>{1, 2, 3, 0}, {}, true);
            const fs::path oldf = s.find(old.serial)->file;
            const fs::path oldm = fs::path(oldf).replace_extension(".meta");
            check(fs::exists(oldf) && fs::exists(oldm), "drop: the files are there");
            check(s.drop_superseded({1, 2}, {}, true).empty() && s.drop_superseded({1, 2, 3, 4}, {}, false).empty() && s.size() == 2,
                  "drop: a shorter or a differently steered image supersedes nothing");
            const auto gone = s.drop_superseded({1, 2, 3, 4, 5}, {}, true);
            check(gone == std::vector<size_t>({3}) && s.size() == 1 && !fs::exists(oldf) && !fs::exists(oldm) && !s.find(old.serial),
                  "drop: the superseded spill is deleted with its index file and entry");
            check(s.put(img({1, 2, 3, 4, 5, 6}, 3), A, e2) && s.size() == 2, "drop: the newest is spilled later");
        }
        {   // restart adoption: only the newest snapshot of the conversation is found
            ConversationSpill s;
            size_t ad = 0;
            check(s.open(proot, cap, e, nullptr, &A, &ad) && ad == 2 && s.size() == 2, "adopt: two entries after the drop");
            bool short_one = false;
            for (size_t k = 0; k < s.size(); ++k) if (s.entry(k).live.ids.size() == 3) short_one = true;
            check(s.best(std::vector<int32_t>{1, 2, 3, 4, 5, 6, 7}, {}, true).tokens == 6 && !short_one, "adopt: the newest serves the conversation");
        }
        {   // #6833: two agents on one engine for a long time: the folder never passes the cap, and an appending conversation keeps one entry
            ConversationSpill s;
            std::string e3;
            size_t ad = 0;
            check(s.open(proot, cap, e3, nullptr, &A, &ad), "churn: open");
            while (s.size() > 0) s.drop(0);
            auto files_on_disk = [&]() {
                size_t n = 0;
                std::error_code fec;
                for (const auto& f : fs::directory_iterator(s.dir(), fec))
                    if (f.is_regular_file() && f.path().filename() != ".lock") ++n;
                return n;
            };
            bool within_cap = true, files_bounded = true;
            uint64_t peak = 0;
            size_t most_files = 0;
            std::vector<int32_t> x = {1, 2, 3}, y = {500, 501, 502};
            for (int turn = 0; turn < 40; ++turn) {   // agent X and agent Y, each appending a turn, parked one whole snapshot per turn
                x.push_back(10 + turn);
                (void) s.drop_superseded(x, {}, true);
                SavedConversation sx = img(x, 1);
                within_cap = within_cap && s.put(std::move(sx), A, e3);
                y.push_back(600 + turn);
                (void) s.drop_superseded(y, {}, true);
                SavedConversation sy = img(y, 2);
                within_cap = within_cap && s.put(std::move(sy), A, e3);
                peak = std::max<uint64_t>(peak, s.bytes());
                most_files = std::max(most_files, files_on_disk());
                files_bounded = files_bounded && s.bytes() <= cap;
            }
            check(within_cap && files_bounded && peak <= cap, "churn: 80 parked snapshots of two appending agents never pass the cap");
            check(s.size() == 2 && files_on_disk() == 4, "churn: one entry (file and index file) per conversation is left");
            check(most_files <= 6, "churn: the folder never holds more than the cap's three entries");
            std::vector<int32_t> px = x, py = y;   // (the next turn's prompt: the parked tokens and one more)
            px.push_back(-1);
            py.push_back(-1);
            check(s.best(px, {}, true).tokens == (int64_t) x.size() && s.best(py, {}, true).tokens == (int64_t) y.size(),
                  "churn: each agent's resume finds its newest snapshot");
            // many unrelated conversations (the evictions of a busy engine): the oldest-used go at the cap, their files with them
            for (int c = 0; c < 40; ++c) {
                SavedConversation sc = img({7000 + c, 7001 + c, 7002 + c}, 3);
                check(s.put(std::move(sc), A, e3), "churn: an unrelated conversation spills");
                peak = std::max<uint64_t>(peak, s.bytes());
                most_files = std::max(most_files, files_on_disk());
            }
            check(peak <= cap && s.size() <= 3 && files_on_disk() <= 6, "churn: forty unrelated conversations: the LRU tier stays under the cap");
        }
        {   // the same model, other settings: the files are not this run's
            ConversationSpill s;
            check(s.open(proot, cap, e, nullptr, &A2, &adopted) && adopted == 0 && s.size() == 0, "restart: another config fingerprint adopts nothing");
        }
        {   // restart: another model fingerprint
            ConversationSpill s;
            check(s.open(proot, cap, e, &cleaned, &B, &adopted) && adopted == 0 && s.size() == 0 && cleaned == 1, "restart: another model fingerprint deletes the entries");
            check(!fs::exists(proot / spill_detail::persist_name(A.model)), "restart: the old folder is gone");
        }
        {   // a damaged index file: that entry is dropped, the others are adopted
            ConversationSpill s;
            check(s.open(proot, cap, e, nullptr, &A, &adopted) && s.put(img({1, 2}, 1), A, e) && s.put(img({3, 4}, 1), A, e), "damage: two entries");
            const fs::path f0 = s.entry(0).file;
            s.close();
            fs::resize_file(fs::path(f0).replace_extension(".meta"), 9, ec);
            ConversationSpill t;
            check(t.open(proot, cap, e, nullptr, &A, &adopted) && adopted == 1 && !fs::exists(f0), "damage: a cut index file drops its entry");
        }
        fs::remove_all(proot, ec);
    }
    fs::remove_all(root, ec);
    {
        SpillNamespace a, b2;
        std::string err;
        check(a.open(root, err) && a.is_open() && fs::is_directory(a.dir()), "the first engine's folder");
        touch(a.file_for(7));
        const fs::path dead = root / "4242-0123456789abcdef";
        fs::create_directories(dead);
        touch(dead / "conv-1.session");                               // a crashed engine's file, no lock file left
        const fs::path dead2 = root / "4243-fedcba9876543210";
        fs::create_directories(dead2);
        touch(dead2 / ".lock");                                       // a crashed engine's lock file, nobody holds it
        fs::create_directories(root / "keepme");
        touch(root / "keepme" / "conv-1.session");
        touch(root / "notes.txt");
        size_t cleaned = 99;
        check(b2.open(root, err, &cleaned), "a second engine on the same root");
        check(cleaned == 2 && !fs::exists(dead) && !fs::exists(dead2), "the two dead folders went");
        check(fs::exists(a.file_for(7)), "the first engine's live folder and file stay");
        check(fs::exists(root / "keepme" / "conv-1.session") && fs::exists(root / "notes.txt"), "nothing else under the root is touched");
        check(a.dir() != b2.dir() && a.file_for(7) != b2.file_for(7), "each engine writes its own files");
#if !defined(_WIN32)
        const fs::path outside = fs::temp_directory_path() / ("strata-spill-outside-" + std::to_string(std::random_device{}()));
        fs::create_directories(outside);
        touch(outside / "keep.txt");
        fs::create_directory_symlink(outside, root / "4244-0123456789abcdef", ec);
        check(!ec && SpillNamespace::clean_dead(root) == 0 && fs::exists(outside / "keep.txt"), "a link named like a folder is never followed");
        fs::remove(root / "4244-0123456789abcdef", ec);
        fs::remove_all(outside, ec);
#endif
        const fs::path adir = a.dir();
        a.close();
        check(!fs::exists(adir) && fs::exists(b2.dir()), "an orderly end removes its own folder only");
    }
    check(fs::exists(root) && !fs::exists(root / "4242-0123456789abcdef"), "both engines ended");
    std::string err;
    check(!SpillNamespace().open(root / "notes.txt" / "sub", err) && !err.empty(), "a root that cannot be a folder is refused with a reason");

    // the entries: written as session files, the oldest go first, matched like the RAM cache, read back exactly
    {
        auto img = [](std::vector<int32_t> ids, uint8_t fill, std::vector<int32_t> ckpt = {}) {
            SavedConversation s;
            s.live.ids = ids;
            s.live.gdn.assign(1 << 20, fill);                  // 1 MiB of running state
            s.cvec = true;
            if (!ckpt.empty()) { ConversationCheckpoint c; c.ids = ckpt; c.gdn.assign(4096, (uint8_t) (fill + 1)); s.checkpoints.push_back(c); }
            return s;
        };
        const SessionFileIdentity id{11, 22};
        ConversationSpill sp;
        // room for two entries of ~2 MiB each (1 MiB payload + the 1 MiB margin per write)
        check(sp.open(root, (uint64_t(7) << 19), err) && sp.enabled(), "the tier opens");
        check(sp.put(img({1, 2, 3, 4}, 7, {1, 2}), id, err) && sp.put(img({5, 6, 7}, 9), id, err) && sp.size() == 2, "two entries written");
        const fs::path first = sp.entry(0).file;
        check(fs::exists(first) && fs::exists(sp.entry(1).file), "their files exist");
        check(sp.bytes() == sp.entry(0).bytes + sp.entry(1).bytes && sp.budget().reserved == 0, "the budget counts what is written");
        check(sp.put(img({8, 9}, 3), id, err) && sp.size() == 2 && !fs::exists(first), "a third pushes out the oldest, its file deleted");
        auto m = sp.best(std::vector<int32_t>{5, 6, 7, 1}, {}, true);
        check(m.tokens == 3 && m.live && sp.entry(m.index).live.ids == std::vector<int32_t>({5, 6, 7}), "a prompt finds the entry it continues");
        check(sp.best(std::vector<int32_t>{1, 2, 3, 4}, {}, true).tokens == 0, "a dropped entry is not found");
        check(sp.best(std::vector<int32_t>{5, 6, 7, 1}, {}, false).tokens == 0, "other steering: not found");
        SavedConversation back;
        const fs::path taken = sp.entry(m.index).file;
        check(sp.take(m.index, id, {}, back, err) && back.live.ids == std::vector<int32_t>({5, 6, 7}) &&
              back.live.gdn == std::vector<uint8_t>(1 << 20, 9), "taken back byte for byte");
        check(sp.size() == 1 && !fs::exists(taken) && sp.bytes() == sp.entry(0).bytes, "a taken entry leaves the tier");
        // a checkpoint match
        check(sp.put(img({20, 21, 22, 23}, 5, {20, 21}), id, err), "an entry with a checkpoint");
        auto mc = sp.best(std::vector<int32_t>{20, 21, 99}, {}, true);
        check(mc.tokens == 2 && !mc.live, "a prompt that shares only the checkpoint finds it");
        // another model's identity: refused, the entry goes, out unchanged
        SavedConversation keep = img({1}, 1);
        check(!sp.take(mc.index, SessionFileIdentity{11, 23}, {}, keep, err) && keep.live.ids == std::vector<int32_t>({1}), "another identity: refused");
        check(sp.size() == 1, "the refused entry is gone");
        // a cut file: refused by the bounded read
        {
            const fs::path f = sp.entry(0).file;
            fs::resize_file(f, fs::file_size(f) / 2, ec);
            SavedConversation x;
            check(!ec && !sp.take(0, id, {}, x, err) && sp.size() == 0 && !fs::exists(f), "a cut file is refused and removed");
        }
        // larger than the whole tier: refused, nothing written
        ConversationSpill small;
        std::string e2;
        check(small.open(root, 1 << 20, e2) && !small.put(img({1, 2}, 4), id, e2) && !e2.empty(), "an image larger than the tier is refused");
        size_t files = 0;
        for (auto& x : fs::directory_iterator(small.dir())) if (x.path().filename() != ".lock") ++files;
        check(files == 0, "nothing of a refused put is left on the disk");
        small.close();
        // #879: clear() drops every entry and its file, the tier stays open
        check(sp.put(img({30, 31}, 2), id, err) && sp.put(img({40, 41}, 2), id, err) && sp.size() == 2, "two more entries");
        const fs::path c0 = sp.entry(0).file, c1 = sp.entry(1).file;
        check(sp.clear() == 2 && sp.size() == 0 && sp.bytes() == 0 && !fs::exists(c0) && !fs::exists(c1), "clear() drops them and their files");
        check(sp.enabled() && fs::is_directory(sp.dir()) && sp.put(img({50}, 1), id, err) && sp.size() == 1, "the tier stays open after clear()");
        sp.close();
    }
    {   // the cost rule, with l620's measured speeds (a 236 MB image written in 120 ms, read back in 220 ms; 5 prompt tokens per ms)
        SpillCost c;
        check(c.reread_tokens_per_ms == 2.0 && c.round_trip_ms(2000000) == 4.0, "the defaults: 2 tokens per ms, 1 GB/s each way");
        for (int i = 0; i < 40; ++i) {
            c.saw_prompt(8000, 1600.0);
            c.saw_write(236000000, 120.0);
            c.saw_read(236000000, 220.0);
        }
        check(c.reread_tokens_per_ms > 4.99 && c.reread_tokens_per_ms < 5.01, "the prompt speed converges to the measured one");
        check(c.round_trip_ms(236000000) > 339.0 && c.round_trip_ms(236000000) < 341.0, "the round trip converges to 120 + 220 ms");
        check(!c.worth(51, 236000000), "a 51-token conversation is not worth the disk (re-read about 10 ms of tokens)");
        check(!c.worth(2000, 264000000), "a 2K one is not either here (400 ms against about 1.25 x 380 ms)");
        check(c.worth(8000, 350000000), "an 8K one is (1600 ms against about 1.25 x 504 ms)");
        const SpillCost before = c;
        c.saw_prompt(100, 1.0);       // fewer than kMinTokens: a request's fixed costs, not the prompt speed
        c.saw_prompt(4000, 0.0);      // no time: ignored
        c.saw_write(0, 5.0);          // nothing written: ignored
        c.saw_read(1000, -1.0);       // a clock that went back: ignored
        check(c.reread_tokens_per_ms == before.reread_tokens_per_ms && c.write_bytes_per_ms == before.write_bytes_per_ms &&
              c.read_bytes_per_ms == before.read_bytes_per_ms, "samples that say nothing about the speeds are ignored");
        SpillCost slow;   // a card that reads prompts at 1 token per ms: the same 2K conversation is worth spilling
        for (int i = 0; i < 40; ++i) { slow.saw_prompt(2000, 2000.0); slow.saw_write(264000000, 130.0); slow.saw_read(264000000, 250.0); }
        check(slow.worth(2000, 264000000), "on a slower card a 2K conversation is worth the disk");
        check(!c.worth(0, 1) && !slow.worth(0, 1), "an empty conversation never is");
    }
    {   // 2c: the background writer
        using namespace std::chrono_literals;
        struct Gate {   // holds the writer inside a write until opened
            std::mutex mu; std::condition_variable cv; bool open = false; int entered = 0;
            void enter() { std::unique_lock<std::mutex> lk(mu); ++entered; cv.notify_all(); cv.wait(lk, [&] { return open; }); }
            void release() { { std::lock_guard<std::mutex> lk(mu); open = true; } cv.notify_all(); }
            void wait_entered(int n) { std::unique_lock<std::mutex> lk(mu); cv.wait_for(lk, 5s, [&] { return entered >= n; }); }
        };
        Gate gate;
        std::vector<int> order;
        std::mutex order_mu;
        {
            BackgroundWriter<int> w([&](const int& v) -> uint64_t {
                gate.enter();
                std::lock_guard<std::mutex> lk(order_mu);
                order.push_back(v);
                return v == 3 ? 0 : (uint64_t) v * 10;   // item 3: a failed write
            });
            w.push(1);
            gate.wait_entered(1);
            w.push(2);   // one waits behind the one in flight
            std::atomic<bool> third_in{false};
            std::thread t([&] { w.push(3); third_in = true; });
            std::this_thread::sleep_for(150ms);
            check(!third_in.load(), "a third push waits while one is in flight and one is queued");
            int held = 0;
            w.for_each_held([&](const int&) { ++held; });
            check(held == 2 && !w.idle(), "the queued one and the one in flight are held");
            gate.release();
            t.join();
            w.wait_idle();
            check(third_in.load() && w.idle(), "the queue drains");
            auto done = w.take_done();
            check(done.size() == 3 && done[0].item == 1 && done[0].serial == 10 && done[1].item == 2 && done[1].serial == 20 &&
                  done[2].item == 3 && done[2].serial == 0, "written in order, each handed back with its serial (0: failed)");
            check(order == std::vector<int>({1, 2, 3}) && w.take_done().empty(), "handed back once");
        }
        Gate g2;
        std::vector<int> written;
        {
            BackgroundWriter<int> w([&](const int& v) -> uint64_t { g2.enter(); written.push_back(v); return 1; });
            w.push(7);
            g2.wait_entered(1);
            w.push(8);
            auto queued = w.take_queued();
            check(queued.size() == 1 && queued[0] == 8, "take_queued gives the queued one back unwritten");
            w.push(9);   // and the writer takes more after it
            g2.release();
        }   // the destructor: 7 finishes, 9 is not written (queued at the end)
        check(written == std::vector<int>({7}) || written == std::vector<int>({7, 9}),
              "the destructor finishes the write in flight and does not hang");
        // a SavedConversation through the real SSD tier, while the serve thread matches and takes by serial
        auto img = [](std::vector<int32_t> ids, uint8_t fill) {
            SavedConversation s;
            s.live.ids = ids;
            s.live.gdn.assign(1 << 18, fill);
            s.cvec = true;
            return s;
        };
        const SessionFileIdentity id{31, 41};
        ConversationSpill sp;
        std::string err;
        check(sp.open(root, uint64_t(64) << 20, err), "a tier for the writer");
        {
            BackgroundWriter<SavedConversation> w([&](const SavedConversation& s) -> uint64_t {
                uint64_t serial = 0;
                std::string e;
                return sp.put(s, id, e, &serial) ? serial : 0;
            });
            for (int k = 0; k < 6; ++k) w.push(img({k, k + 1, k + 2}, (uint8_t) k));
            for (int k = 0; k < 200; ++k) {   // the serve thread matching while the writer puts
                const auto m = sp.best(std::vector<int32_t>{2, 3, 4, 9}, {}, true);
                if (m.tokens > 0) (void) sp.find(m.serial);   // (may be gone or not yet listed: it must only not crash or race)
            }
            w.wait_idle();
            auto done = w.take_done();
            check(done.size() == 6 && sp.size() == 6, "six written, six listed");
            const auto m = sp.best(std::vector<int32_t>{2, 3, 4, 9}, {}, true);
            check(m.tokens == 3, "the entry a prompt continues is found");
            const auto e = sp.find(m.serial);
            check(e && e->live.ids == std::vector<int32_t>({2, 3, 4}), "found by its serial");
            SavedConversation back;
            check(sp.take_serial(m.serial, id, {}, back, err) && back.live.gdn == std::vector<uint8_t>(1 << 18, 2) &&
                  !sp.find(m.serial) && sp.size() == 5, "taken by its serial, byte for byte; it leaves the list");
            check(!sp.take_serial(m.serial, id, {}, back, err) && !sp.drop_serial(m.serial), "a gone serial: nothing");
            check(sp.drop_serial(done[0].serial) && sp.size() == 4, "dropped by its serial");
            // a written image still in RAM: the prefix it gives, by the RAM cache's rule
            const auto p = conversation_image_prefix(done[5].item, std::vector<int32_t>{5, 6, 7, 1}, {}, true);
            check(p.first == 3 && p.second, "a held image's live prefix");
            check(conversation_image_prefix(done[5].item, std::vector<int32_t>{5, 6, 7}, {}, false).first == 0, "other steering: none");
        }
        sp.close();
    }
    {   // SSD context swap (A): put_sources streams the K/V from its owner; the entry reads back as put() of the captured image
        auto pattern = [](size_t n, uint8_t seed) {
            ConversationBuffer b;
            b.resize(n);
            b.visit(0, n, [&](uint8_t* p, size_t c, size_t at) {
                for (size_t i = 0; i < c; ++i) p[i] = uint8_t((at + i) * 131u + seed);
                return true;
            });
            return b;
        };
        SavedConversation full;
        for (size_t i = 0; i < full.geometry.size(); ++i) full.geometry[i] = int64_t(100 + i);
        full.layer_lo = 0; full.layer_hi = 48;
        full.cvec = true;
        full.live.ids = {7, 8, 9, 10, 11};
        full.live.gdn.assign(1 << 16, 3);
        ConversationCheckpoint ck;
        ck.ids = {7, 8, 9};
        ck.gdn.assign(1 << 16, 4);
        full.checkpoints.push_back(ck);
        for (int layer = 0; layer < 2; ++layer) {
            ConversationKv kv;
            kv.format = 2; kv.cells = 1024; kv.heads = 2; kv.head_dim = 256; kv.page_size = 64; kv.pooled_rows = 0; kv.idx_dim = 128;
            kv.k = pattern(layer == 0 ? (17u << 20) + 5 : 4096, uint8_t(layer));   // > 16 MiB: several staging-buffer rounds
            kv.v = pattern(4096 + layer, uint8_t(layer + 10));
            kv.k_scale = pattern(512, uint8_t(layer + 20));
            kv.v_scale = pattern(512, uint8_t(layer + 30));
            full.kv.push_back(std::move(kv));
        }
        SavedConversation meta = full;
        meta.kv.clear();
        std::vector<SessionKvSource> sources;
        for (const auto& k : full.kv) {
            SessionKvSource s;
            s.format = k.format; s.cells = k.cells; s.heads = k.heads; s.head_dim = k.head_dim;
            s.page_size = k.page_size; s.pooled_rows = k.pooled_rows; s.idx_dim = k.idx_dim;
            const std::array<const ConversationBuffer*, 5> parts = {&k.k, &k.v, &k.k_scale, &k.v_scale, &k.pooled};
            for (size_t i = 0; i < 5; ++i) s.sizes[i] = parts[i]->size();
            s.read = [parts](size_t part, size_t offset, void* dst, size_t n) { return parts[part]->read(dst, offset, n); };
            sources.push_back(std::move(s));
        }
        const fs::path sroot = fs::temp_directory_path() / ("strata-spill-stream-" + std::to_string(std::random_device{}()));
        fs::remove_all(sroot, ec);
        const SessionFileIdentity id{0x5a, 3};
        std::string err;
        ConversationSpill sp;
        check(sp.open(sroot, uint64_t(256) << 20, err), "stream: open");
        uint64_t s1 = 0, s2 = 0;
        check(!sp.put_sources(full, sources, id, err, &s1), "stream: meta that still carries K/V is refused");
        check(sp.put_sources(meta, sources, id, err, &s1) && s1 != 0 && sp.size() == 1, "stream: written from the sources");
        const auto m = sp.best(std::vector<int32_t>{7, 8, 9, 10, 11, 12}, {}, true);
        check(m.tokens == 5 && m.live && m.serial == s1, "stream: found by its live tokens");
        check(sp.best(std::vector<int32_t>{7, 8, 9, 99}, {}, true).tokens == 3, "stream: its checkpoint is listed too");
        const uint64_t streamed_bytes = sp.find(s1)->bytes;
        SavedConversation back;
        check(sp.take_serial(s1, id, {}, back, err), "stream: read back");
        bool same_kv = back.kv.size() == full.kv.size();
        for (size_t l = 0; same_kv && l < full.kv.size(); ++l) {
            const auto& a = full.kv[l]; const auto& b2 = back.kv[l];
            same_kv = a.k.size() == b2.k.size() && a.v.size() == b2.v.size() && a.cells == b2.cells;
            std::vector<uint8_t> x(a.k.size()), y(b2.k.size());
            same_kv = same_kv && a.k.read(x.data(), 0, x.size()) && b2.k.read(y.data(), 0, y.size()) && x == y;
        }
        check(same_kv && back.live.ids == full.live.ids && back.live.gdn == full.live.gdn &&
              back.checkpoints.size() == 1 && back.checkpoints[0].ids == ck.ids, "stream: byte for byte what was streamed");
        check(sp.put(full, id, err, &s2) && sp.find(s2)->bytes == streamed_bytes, "stream: the same file size as put() of the captured image");
        auto broken = sources;
        broken[1].read = [](size_t, size_t, void*, size_t) { return false; };
        const size_t before = sp.size();
        check(!sp.put_sources(meta, broken, id, err) && sp.size() == before, "stream: a failing source lists nothing");
        auto files_in = [](const fs::path& d) {
            size_t n = 0;
            std::error_code e2;
            for (fs::directory_iterator it(d, e2), end; !e2 && it != end; it.increment(e2)) ++n;
            return n;
        };
        {   // a throw out of the write (not one session_file_write turns into false) hands the reservation back, leaves no file
            auto throwing = sources;
            throwing[0].read = [](size_t, size_t, void*, size_t) -> bool { throw std::runtime_error("source gone"); };
            const uint64_t used_before = sp.bytes();
            const size_t files_before = files_in(sp.dir());
            bool threw = false;
            try { (void) sp.put_sources(meta, throwing, id, err); } catch (const std::runtime_error&) { threw = true; }
            check(threw && sp.budget().reserved == 0 && sp.bytes() == used_before && sp.size() == before &&
                  files_in(sp.dir()) == files_before, "stream: a throw rolls the reservation and the file back");
        }
        {   // B: a read refused for RAM keeps the entry (keep_on_memory); without it the entry goes, as before
            uint64_t s3 = 0;
            check(sp.put_sources(meta, sources, id, err, &s3), "keep: written");
            const uint64_t used = sp.bytes();
            SessionReadLimits refuse;
            refuse.admit = [](uint64_t, std::string& w) { w = "no RAM"; return false; };
            SavedConversation b3;
            check(!sp.take_serial(s3, id, refuse, b3, err, true) && sp.find(s3) && sp.bytes() == used &&
                  sp.budget().reserved == 0 && fs::exists(sp.find(s3)->file),
                  "keep: refused for RAM, still listed with its file and bytes, nothing left reserved");
            check(sp.take_serial(s3, id, {}, b3, err, true) && b3.live.ids == full.live.ids && !sp.find(s3),
                  "keep: read back on a later try");
            uint64_t s4 = 0;
            check(sp.put_sources(meta, sources, id, err, &s4), "keep: written again");
            check(!sp.take_serial(s4, id, refuse, b3, err) && !sp.find(s4), "keep: without keep_on_memory the entry goes (old behaviour)");
        }
        sp.close();
        fs::remove_all(sroot, ec);
    }
    {   // B: the entry a read makes room for is guarded - a put() that needs room evicts another one, or fails
        auto image = [](int32_t first) {
            SavedConversation s;
            s.live.ids = {first, first + 1, first + 2};
            s.live.gdn.assign(size_t(6) << 20, uint8_t(first));
            s.cvec = true;
            return s;
        };
        const fs::path groot = fs::temp_directory_path() / ("strata-spill-guard-" + std::to_string(std::random_device{}()));
        fs::remove_all(groot, ec);
        const SessionFileIdentity id{0x6b, 4};
        std::string err;
        ConversationSpill sp;
        check(sp.open(groot, uint64_t(16) << 20, err), "guard: open (room for two entries)");
        uint64_t g1 = 0, g2 = 0, g3 = 0;
        check(sp.put(image(10), id, err, &g1) && sp.put(image(20), id, err, &g2) && sp.size() == 2, "guard: two entries");
        sp.guard(g1);   // the least recently used one
        check(sp.put(image(30), id, err, &g3) && sp.find(g1) && !sp.find(g2) && sp.find(g3), "guard: the other one is evicted");
        sp.guard(0);
        sp.close();
        fs::remove_all(groot, ec);
        ConversationSpill one;
        check(one.open(groot, uint64_t(9) << 20, err), "guard: open (room for one)");
        check(one.put(image(40), id, err, &g1) && one.size() == 1, "guard: one entry");
        one.guard(g1);
        check(!one.put(image(50), id, err, &g2) && one.find(g1) && one.size() == 1 && one.budget().reserved == 0,
              "guard: no room but the guarded one: the put fails, the guarded entry stays");
        one.guard(0);
        check(one.put(image(50), id, err, &g2) && !one.find(g1) && one.find(g2), "guard: cleared, the put evicts it");
        one.close();
        fs::remove_all(groot, ec);
    }
    fs::remove_all(root, ec);
    std::printf("conversation_spill_test: %d checks, %d failed\n", checks, failures);
    return failures == 0 ? 0 : 1;
}
