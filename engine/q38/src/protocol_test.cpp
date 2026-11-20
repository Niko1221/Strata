#include "protocol.hpp"
#include "protocol_output.hpp"

#include <cstdlib>
#include <condition_variable>
#include <iostream>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#define CHECK(condition) do { if (!(condition)) { std::cerr << "FAIL line " << __LINE__ << ": " #condition "\n"; std::abort(); } } while (0)

using Event = Q38Protocol::Event;
using Kind = Q38Protocol::Kind;

static Event next(Q38Protocol &p) { Event e{}; CHECK(p.next(e)); return e; }
static void overflow_and_reuse() {
    Q38Protocol p; std::vector<std::string> output;
    p.submit("GEN 1 1"); CHECK(next(p).line == "GEN 1 1");
    p.submit("GEN 1 2"); p.submit("bad"); p.submit("more"); p.submit("again");
    CHECK(output.empty()); // Reader never emits errors during an active request.
    p.submit("STOP"); CHECK(p.stop_flag().load()); // STOP remains effective even while full.
    p.finish([&](bool) { output.push_back("DONE"); });
    CHECK(next(p).kind == Kind::busy); p.finish_reply([&] {output.push_back("ERR busy");});
    CHECK(next(p).line == "bad"); p.finish_reply([&] {output.push_back("ERR command");});
    CHECK(next(p).kind == Kind::overflow); p.finish_reply([&] {output.push_back("ERR overflow");});
    CHECK((output == std::vector<std::string>{"DONE", "ERR busy", "ERR command", "ERR overflow"}));
    p.submit("GEN 1 3"); CHECK(next(p).line == "GEN 1 3");
    CHECK(!p.stop_flag().load()); p.submit("STOP"); CHECK(p.stop_flag().load());
    p.finish([&](bool) { output.push_back("DONE cancel"); });
    p.submit("GEN 1 4"); CHECK(next(p).line == "GEN 1 4"); CHECK(!p.stop_flag().load());
    p.finish([](bool) {}); p.end_input(); Event end{}; CHECK(!p.next(end));
}

static void terminal_boundary() {
    Q38Protocol p; p.submit("GEN 1 1"); CHECK(next(p).line == "GEN 1 1");
    std::mutex mu; std::condition_variable cv; bool attempt=false, done=false;
    std::thread reader([&] {
        {std::lock_guard<std::mutex> lock(mu); attempt=true;} cv.notify_one();
        p.submit("GEN 1 2"); // Blocked until terminal publication ends.
        p.submit("STOP");    // Accepted new request is queued; ordered STOP applies.
    });
    p.finish([&](bool) {
        std::unique_lock<std::mutex> lock(mu); cv.wait(lock,[&] {return attempt;});
        CHECK(!done); done=true; // Represents writing and flushing DONE.
    });
    reader.join(); CHECK(done);
    CHECK(next(p).line == "GEN 1 2"); CHECK(p.stop_flag().load());
    p.finish([](bool) {}); p.end_input(); Event end{}; CHECK(!p.next(end));
}

static void reject_until_reply_flushed() {
    Q38Protocol p; p.submit("GEN 1 1"); next(p);
    p.submit("GEN 1 2"); p.finish([](bool) {});
    CHECK(next(p).kind == Kind::busy);
    p.submit("GEN 1 3"); // A previous rejection has not been published yet.
    p.finish_reply([] {});
    CHECK(next(p).kind == Kind::busy); p.finish_reply([] {});
    p.submit("GEN 1 4"); CHECK(next(p).line == "GEN 1 4");
    p.finish([](bool) {}); p.end_input(); Event end{}; CHECK(!p.next(end));
}

static void immediate_cancel() {
    Q38Protocol p; p.submit("GEN 1 1"); p.submit("STOP");
    CHECK(next(p).line == "GEN 1 1"); CHECK(p.stop_flag().load());
    p.finish([](bool) {}); p.submit("GEN 1 2");
    CHECK(next(p).line == "GEN 1 2"); CHECK(!p.stop_flag().load());
    p.finish([](bool) {}); p.end_input(); Event end{}; CHECK(!p.next(end));
}

static void cancellation_before_terminal_publication() {
    Q38Protocol p; p.submit("GEN 1 1"); CHECK(next(p).line == "GEN 1 1");
    // The owner completed decode and tentatively chose a normal length finish.
    std::string finish="length";
    p.submit("STOP"); // Arrives before finish() takes the terminal mutex.
    p.finish([&](bool cancelled) { if(cancelled) finish="cancel"; });
    CHECK(finish == "cancel");
    p.submit("GEN 1 2"); CHECK(next(p).line == "GEN 1 2");
    p.finish([&](bool cancelled) { CHECK(!cancelled); });
    p.end_input(); Event end{}; CHECK(!p.next(end));
}

static void eof_cases() {
    Q38Protocol empty; empty.end_input(); Event e{}; CHECK(!empty.next(e));
    Q38Protocol queued; queued.submit("GEN 1 1"); queued.end_input();
    CHECK(next(queued).line == "GEN 1 1"); CHECK(queued.stop_flag().load());
    queued.finish([](bool) {}); CHECK(!queued.next(e));
    Q38Protocol running; running.submit("GEN 1 1"); next(running);
    running.end_input(); CHECK(running.stop_flag().load());
    running.finish([](bool) {}); CHECK(!running.next(e));
}
static void quit_cancels_active() {
    Q38Protocol p;p.submit("GEN 9 1");next(p);p.submit("QUIT");CHECK(p.stop_flag().load());
    p.finish([](bool stopped){CHECK(stopped);});CHECK(next(p).line=="QUIT");p.finish_reply([]{});p.end_input();
}
static void quit_survives_full_queue() {
    for(bool overflow : {false,true}) {
        Q38Protocol p;p.submit("GEN 9 1");CHECK(next(p).line=="GEN 9 1");
        p.submit("GEN 1 2");p.submit("bad");
        if(overflow)p.submit("over-capacity");
        p.submit("QUIT");CHECK(p.stop_flag().load());
        p.submit("GEN 1 3");p.submit("QUIT"); // Cannot enqueue work after termination.
        p.finish([](bool cancelled){CHECK(cancelled);});
        CHECK(next(p).kind==Kind::busy);p.finish_reply([]{});
        CHECK(next(p).line=="bad");p.finish_reply([]{});
        if(overflow){CHECK(next(p).kind==Kind::overflow);p.finish_reply([]{});}
        CHECK(next(p).line=="QUIT");p.finish_reply([]{});
        Event end{};CHECK(!p.next(end)); // No EOF or new input needed to terminate.
    }
    Q38Protocol queued;queued.submit("GEN 9 1");queued.submit("bad");queued.submit("overflow");queued.submit("QUIT");
    CHECK(next(queued).line=="GEN 9 1");CHECK(queued.stop_flag().load());queued.finish([](bool cancelled){CHECK(cancelled);});
    CHECK(next(queued).line=="bad");queued.finish_reply([]{});
    CHECK(next(queued).kind==Kind::overflow);queued.finish_reply([]{});
    CHECK(next(queued).line=="QUIT");queued.finish_reply([]{});
    Event end{};CHECK(!queued.next(end));
}

int main(int argc, char **argv) {
    if(argc==2 && std::string(argv[1])=="--emit-done-fixtures") {
        std::cout << q38_format_done(3, 17, 12.5, 6.2, "eos", 17)
                  << q38_format_done(1, 128, 20.0, 3.5, "cancel", 64)
                  << q38_format_done(0, 8, 0.0, 0.0, "length", 8);
        return 0;
    }
    overflow_and_reuse(); terminal_boundary(); reject_until_reply_flushed(); immediate_cancel(); cancellation_before_terminal_publication(); eof_cases(); quit_cancels_active(); quit_survives_full_queue();
    std::cout << "Q38 protocol ownership, overflow, STOP, EOF and reuse: PASS\n";
}
