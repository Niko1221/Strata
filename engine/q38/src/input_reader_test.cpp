#include "input_reader.hpp"
#include <cstdio>
#include <cstdlib>

static void require(bool value, const char* message) {
    if (!value) { std::fprintf(stderr, "FAIL %s\n", message); std::exit(1); }
}
struct Event {
    HANDLE handle = CreateEventW(nullptr, TRUE, FALSE, nullptr);
    Event() { require(handle != nullptr, "create event"); }
    ~Event() { CloseHandle(handle); }
    void set() { require(SetEvent(handle) != FALSE, "set event"); }
    void wait() { require(WaitForSingleObject(handle, 3000) == WAIT_OBJECT_0, "bounded event wait"); }
};
struct Pipe {
    HANDLE input = nullptr, writer = nullptr;
    Pipe() { require(CreatePipe(&input, &writer, nullptr, 0) != FALSE, "create pipe"); }
    ~Pipe() { CloseHandle(input); if (writer) CloseHandle(writer); }
    void send(const std::string& text) {
        DWORD written = 0;
        require(WriteFile(writer, text.data(), DWORD(text.size()), &written, nullptr)
                && written == text.size(), "write pipe");
    }
    void eof() { CloseHandle(writer); writer = nullptr; }
};

// Force the exact lost-cancellation interleaving, with the writer held open.
// QUIT has been delivered before the next read; fatal/main-stop has no input.
static void pre_read_gap(bool quit) {
    Pipe pipe;
    Q38Protocol state;
    Event boundary, resume;
    std::atomic<unsigned> reads{0}, missed{0}, cancelled{0};
    Q38InputReader reader(pipe.input, state, {
        [&] {
            if (++reads == (quit ? 2u : 1u)) { boundary.set(); resume.wait(); }
        },
        [&](DWORD error) {
            if (error == ERROR_NOT_FOUND) { ++missed; resume.set(); }
            else ++cancelled;
        }
    });
    if (quit) {
        pipe.send("QUIT\r\n");
        Q38Protocol::Event ev{};
        require(state.next(ev) && ev.line == "QUIT", "QUIT delivery");
    }
    boundary.wait();
    reader.stop_and_join();
    require(missed > 0 && cancelled > 0, "missed cancellation followed by actual pending-read cancellation");
    require(state.input_ended(), "reader ended with writer open");
    DWORD flags = 0;
    require(GetHandleInformation(pipe.writer, &flags) != FALSE, "stdin writer remains open");
    std::printf("PASS %s pre-read gap: missed=%u cancelled=%u writer=open\n",
                quit ? "QUIT" : "fatal/main-stop", missed.load(), cancelled.load());
}

static void eof() {
    Pipe pipe;
    Q38Protocol state;
    Q38InputReader reader(pipe.input, state);
    pipe.send("unterminated");
    pipe.eof();
    Q38Protocol::Event ev{};
    require(!state.next(ev), "EOF discards incomplete line");
    reader.stop_and_join();
    require(state.input_ended(), "EOF reader ended");
    std::puts("PASS EOF");
}

static void stop_reset() {
    Pipe pipe;
    Q38Protocol state;
    Event after_stop;
    std::atomic<unsigned> reads{0};
    Q38InputReader reader(pipe.input, state, {
        [&] { if (++reads == 3) after_stop.set(); }, {}
    });
    pipe.send("GEN 1 42\r\n");
    Q38Protocol::Event ev{};
    require(state.next(ev) && ev.line == "GEN 1 42", "GEN/CRLF framing");
    pipe.send("STOP\n");
    after_stop.wait();
    require(state.stop_flag().load(), "STOP cancels active request");
    state.finish([](bool cancelled) { require(cancelled, "cancelled reply"); });
    pipe.send("GEN 1 43\n");
    require(state.next(ev) && ev.line == "GEN 1 43" && !state.stop_flag().load(), "next GEN resets cancellation");
    state.finish([](bool cancelled) { require(!cancelled, "normal reply"); });
    pipe.send("QUIT\n");
    require(state.next(ev) && ev.line == "QUIT", "normal QUIT");
    reader.stop_and_join();
    std::puts("PASS STOP/GEN reset and QUIT with writer open");
}

static void unwind() {
    Pipe pipe;
    Q38Protocol state;
    try {
        Q38InputReader reader(pipe.input, state);
        throw 1;
    } catch (int) {}
    require(state.input_ended(), "exception unwinding joins reader");
    std::puts("PASS exceptional main exit with writer open");
}

int main() {
    pre_read_gap(true);
    pre_read_gap(false);
    eof();
    stop_reset();
    unwind();
    std::puts("PASS 5 production input-reader cases; no CUDA/model required");
}
