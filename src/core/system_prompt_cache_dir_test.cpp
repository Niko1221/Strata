// The --system-prompt-cache-dir DEFAULT's resolution rule (system_prompt_cache_dir.hpp).  Host-only, header-only,
// no CUDA and no model: the rule is pure, so the three cases the operator cares about are exercised here without the
// engine - and it is the SAME rule the spill directory already resolves, only the leaf folder differs:
//
//   (a) no flag, the Strata folder is off the system drive  -> <strata dir>\system-prompt-cache
//   (b) no flag, the Strata folder IS on the system drive    -> <user profile>\.strata\system-prompt-cache
//   (c) the flag is set                                       -> the flag wins, over both defaults
//
// The remaining checks cover the helpers (path joining, the ".strata" profile join) and the two fallbacks the engine
// uses when the Strata folder turns out to be unusable.  The drive helpers are the spill module's own and are
// covered by its test; this one checks they are reached through the same rule.
#include "strata/core/system_prompt_cache_dir.hpp"

#include <cstdio>
#include <cstdlib>
#include <string>

using namespace strata::core;

namespace {

int checks = 0;
void check(bool ok, const char* label) {
    ++checks;
    if (!ok) { std::fprintf(stderr, "FAIL: %s\n", label); std::exit(1); }
}

const char* kStrataOffC = "D:\\AI\\repos\\asistentes\\strata-windows-x64-cuda12";
const char* kStrataOnC = "C:\\Strata\\strata-windows-x64-cuda12";
const char* kHomeOnC = "C:\\Users\\Shahrokh";
const char* kSystemDrive = "C:";

} // namespace

int main() {
    // (a) No flag and the Strata folder sits off the system drive: the Strata folder's own cache wins.
    {
        const SystemPromptCacheDirChoice c = system_prompt_cache_dir_resolve("", kStrataOffC, kHomeOnC, kSystemDrive);
        check(c.source == SystemPromptCacheDirSource::strata_dir, "a: the Strata folder is the default off the system drive");
        check(c.directory == std::string(kStrataOffC) + "\\system-prompt-cache", "a: <strata dir>\\system-prompt-cache is chosen");
        check(!c.directory.empty() && c.directory.find("D:\\") == 0, "a: the chosen folder stays on D:");
    }
    // (b) No flag and the Strata folder is on the system drive: the user profile wins instead.
    {
        const SystemPromptCacheDirChoice c = system_prompt_cache_dir_resolve("", kStrataOnC, kHomeOnC, kSystemDrive);
        check(c.source == SystemPromptCacheDirSource::user_home, "b: the system drive sends the default to the user profile");
        check(c.directory == std::string(kHomeOnC) + "\\.strata\\system-prompt-cache", "b: <user profile>\\.strata\\system-prompt-cache is chosen");
        check(c.directory.find("C:\\Strata") == std::string::npos, "b: the Strata folder on C: is NOT used");
    }
    // (c) The flag is set: it wins, wherever the Strata folder is.
    {
        const SystemPromptCacheDirChoice c = system_prompt_cache_dir_resolve("E:\\spc", kStrataOnC, kHomeOnC, kSystemDrive);
        check(c.source == SystemPromptCacheDirSource::explicit_flag, "c: an explicit flag is its own source");
        check(c.directory == "E:\\spc", "c: the explicit folder wins over both defaults");
    }
    // The drive rule is case-insensitive and recognizes a forward-slash system path.
    {
        const SystemPromptCacheDirChoice c = system_prompt_cache_dir_resolve("", "c:/Strata", kHomeOnC, "c:");
        check(c.source == SystemPromptCacheDirSource::user_home, "the system-drive match is case-insensitive");
        check(c.directory == std::string(kHomeOnC) + "\\.strata\\system-prompt-cache", "a lowercase C: still routes to the profile");
    }
    // Off Windows (no system drive) the Strata folder always wins, C:-looking path or not.
    {
        const SystemPromptCacheDirChoice c = system_prompt_cache_dir_resolve("", kStrataOnC, kHomeOnC, "");
        check(c.source == SystemPromptCacheDirSource::strata_dir, "with no system drive the Strata folder is the default");
        check(c.directory == std::string(kStrataOnC) + "\\system-prompt-cache", "no system drive: <strata dir>\\system-prompt-cache is chosen");
    }
    // The engine's "the Strata folder could not be created" fallback: skip_strata_dir sends it to the profile.
    {
        const SystemPromptCacheDirChoice c = system_prompt_cache_dir_resolve("", kStrataOffC, kHomeOnC, kSystemDrive, /*skip_strata_dir=*/true);
        check(c.source == SystemPromptCacheDirSource::user_home, "skip_strata_dir: the profile is the fallback");
        check(c.directory == std::string(kHomeOnC) + "\\.strata\\system-prompt-cache", "skip_strata_dir: <user profile>\\.strata\\system-prompt-cache");
    }
    // Nothing resolvable: no flag and no usable folder -> empty, source none (the cache stays off, fail-open).
    {
        const SystemPromptCacheDirChoice c = system_prompt_cache_dir_resolve("", "", "", "");
        check(c.source == SystemPromptCacheDirSource::none, "no inputs: source none");
        check(c.directory.empty(), "no inputs: an empty directory");
        check(!c.reason.empty(), "no inputs: a reason is still reported for the log");
    }
    // No Strata folder but a home: the profile is the default with its own reason.
    {
        const SystemPromptCacheDirChoice c = system_prompt_cache_dir_resolve("", "", kHomeOnC, kSystemDrive);
        check(c.source == SystemPromptCacheDirSource::user_home, "no Strata folder: the profile is the default");
        check(c.directory == std::string(kHomeOnC) + "\\.strata\\system-prompt-cache", "no Strata folder: <user profile>\\.strata\\system-prompt-cache");
    }
    // Helper checks: a trailing separator is trimmed, and a POSIX home keeps '/' separators.
    {
        check(system_prompt_cache_strata_dir("D:\\Strata\\") == "D:\\Strata\\system-prompt-cache", "trailing '\\' is trimmed");
        check(system_prompt_cache_strata_dir("/opt/strata/") == "/opt/strata/system-prompt-cache", "trailing '/' is trimmed (POSIX)");
        check(system_prompt_cache_user_dir("/home/me") == "/home/me/.strata/system-prompt-cache", "POSIX home uses '/' separators");
        check(system_prompt_cache_strata_dir("") == "", "no Strata folder: no folder");
        check(system_prompt_cache_user_dir("") == "", "no user home: no folder");
    }

    std::printf("system_prompt_cache_dir_test: %d host-only checks passed\n", checks);
    return 0;
}
