// Linux selftest for --shared-expert-arena's file (PinnedArena with a shared_file): one engine loads it, the others
// wait and then attach without loading; an engine that dies while loading leaves it to the next one; a file of
// another pack is replaced only when no engine maps it.  The other engines are child processes started by exec (a
// fork that keeps using the GPU runtime is not safe), each running one role of this same binary.
#include "strata/core/pinned.hpp"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include <fcntl.h>
#include <signal.h>
#include <sys/wait.h>
#include <unistd.h>

namespace {

const uint64_t kBytes = 3u << 20;        // not a whole number of 2 MiB pages: the file rounds the arena up
const uint64_t kPack = 0x0123456789abcdefull;
const std::vector<uint64_t> kNoBounds;

pid_t spawn(const char* role, const std::string& path) {
    const pid_t pid = fork();
    if (pid == 0) {
        execl("/proc/self/exe", "pinned_shared_test", role, path.c_str(), (char*) nullptr);
        _exit(127);
    }
    return pid;
}

int wait_exit(pid_t pid) {
    int status = 0;
    if (waitpid(pid, &status, 0) != pid || !WIFEXITED(status)) return -1;
    return WEXITSTATUS(status);
}

// child roles: 0 = as expected
int role_attach(const std::string& path) {             // waits for the loader, then must attach to its bytes
    strata::core::PinnedArena a(kBytes, kNoBounds, 0, path, kPack);
    if (!a.valid() || !a.shared_loaded) {
        std::fprintf(stderr, "attach: valid=%d loaded=%d: %s\n", a.valid(), a.shared_loaded, a.note.c_str());
        return 1;
    }
    return a.data()[123] == 0x5a && a.data()[kBytes - 1] == 0xa5 ? 0 : 2;
}

int role_die_loading(const std::string& path) {        // becomes the loader and exits without publishing
    strata::core::PinnedArena a(kBytes, kNoBounds, 0, path, kPack);
    if (!a.valid() || a.shared_loaded) return 1;
    a.data()[123] = 0x11;
    _exit(0);
}

int role_hold(const std::string& path) {               // maps the published file until killed
    strata::core::PinnedArena a(kBytes, kNoBounds, 0, path, kPack);
    if (!a.valid() || !a.shared_loaded) return 1;
    std::printf("held\n");
    std::fflush(stdout);
    pause();
    return 0;
}

int check(bool ok, const char* what, const std::string& note) {
    if (!ok) std::fprintf(stderr, "FAIL %s: %s\n", what, note.c_str());
    return ok ? 0 : 1;
}

}  // namespace

int main(int argc, char** argv) {
    alarm(60);   // a lock that is never released fails the test instead of hanging it
    if (argc == 3) {
        const std::string role = argv[1];
        if (role == "attach") return role_attach(argv[2]);
        if (role == "die-loading") return role_die_loading(argv[2]);
        if (role == "hold") return role_hold(argv[2]);
        return 3;
    }
    // tmpfs where there is one: the backing the option is for.  STRATA_TEST_ARENA_DIR picks another (a hugetlbfs
    // mount, say), and the first engine's line is printed: how much of the file is on 2 MiB pages.
    const char* dir = std::getenv("STRATA_TEST_ARENA_DIR");
    std::string path = dir != nullptr && *dir ? std::string(dir) + "/strata-shared-arena-XXXXXX"
                     : access("/dev/shm", W_OK) == 0 ? "/dev/shm/strata-shared-arena-XXXXXX"
                                                      : "/tmp/strata-shared-arena-XXXXXX";
    std::vector<char> tmpl(path.begin(), path.end());
    tmpl.push_back('\0');
    const int fd = mkstemp(tmpl.data());
    if (fd < 0) {
        std::perror("mkstemp");
        return 2;
    }
    close(fd);
    path = tmpl.data();
    int fail = 0;

    {   // 1. the first engine loads; a second one waits until it publishes, then attaches to the same bytes
        strata::core::PinnedArena first(kBytes, kNoBounds, 0, path, kPack);
        fail |= check(first.valid() && !first.shared_loaded, "the first engine loads", first.note);
        std::printf("first engine: %s\n", first.note.c_str());
        fail |= check(((uintptr_t) first.data() & ((2u << 20) - 1)) == 0, "the arena is 2 MiB-aligned", first.note);
        const pid_t waiter = spawn("attach", path);
        usleep(300 * 1000);
        int status = 0;
        fail |= check(waitpid(waiter, &status, WNOHANG) == 0, "the second engine waits for the load", "");
        first.data()[123] = 0x5a;
        first.data()[kBytes - 1] = 0xa5;
        std::string err;
        fail |= check(first.shared_publish(err) && first.shared_loaded, "publish", err);
        fail |= check(wait_exit(waiter) == 0, "the second engine attached to the loaded bytes", "");

        // 2. published: another engine in this process attaches at once
        strata::core::PinnedArena second(kBytes, kNoBounds, 0, path, kPack);
        fail |= check(second.valid() && second.shared_loaded && second.data()[123] == 0x5a, "attach", second.note);

        // 3. another pack, or another size, while engines map the file: refused, the file is left alone
        strata::core::PinnedArena wrong_pack(kBytes, kNoBounds, 0, path, kPack + 1);
        fail |= check(!wrong_pack.valid() && wrong_pack.note.find("another engine maps it") != std::string::npos,
                      "another pack is refused while the file is mapped", wrong_pack.note);
        strata::core::PinnedArena wrong_size(kBytes * 4, kNoBounds, 0, path, kPack);
        fail |= check(!wrong_size.valid(), "another size is refused while the file is mapped", wrong_size.note);
    }
    {   // 4. nobody maps it: a restart attaches without loading (the file outlives the engines)
        strata::core::PinnedArena again(kBytes, kNoBounds, 0, path, kPack);
        fail |= check(again.valid() && again.shared_loaded && again.data()[123] == 0x5a, "a restart attaches",
                      again.note);
    }
    {   // 5. a held file of another pack is refused even from another process; once released, it is replaced
        const pid_t holder = spawn("hold", path);
        usleep(300 * 1000);
        strata::core::PinnedArena other(kBytes, kNoBounds, 0, path, kPack + 1);
        fail |= check(!other.valid(), "another engine's file is refused", other.note);
        kill(holder, SIGKILL);
        (void) wait_exit(holder);
        strata::core::PinnedArena replaced(kBytes, kNoBounds, 0, path, kPack + 1);
        fail |= check(replaced.valid() && !replaced.shared_loaded, "an unused file of another pack is replaced",
                      replaced.note);
    }
    {   // 6. an engine that dies while loading: the next one loads it again
        unlink(path.c_str());
        fail |= check(wait_exit(spawn("die-loading", path)) == 0, "the loader that dies", "");
        strata::core::PinnedArena next(kBytes, kNoBounds, 0, path, kPack);
        fail |= check(next.valid() && !next.shared_loaded, "the next engine loads again", next.note);
    }

    if (unlink(path.c_str()) != 0) {
        std::perror("unlink");
        fail = 1;
    }
    std::printf("pinned_shared_test: %s\n", fail ? "FAILED" : "OK");
    return fail;
}
