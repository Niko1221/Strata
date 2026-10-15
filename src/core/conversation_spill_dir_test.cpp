// The --conversation-cache-spill-dir DEFAULT's resolution rule (conversation_spill_dir.hpp).  Host-only,
// header-only, no CUDA and no model: the rule is pure, so the three cases the operator cares about are
// exercised here without the engine:
//
//   (a) no flag, the Strata folder is off the system drive  -> <strata dir>\slots
//   (b) no flag, the Strata folder IS on the system drive    -> <user profile>\.strata\slots
//   (c) the flag is set                                       -> the flag wins, over both defaults
//
// The remaining checks cover the helpers (path joining, drive detection) and the two fallbacks the engine
// uses when the Strata folder turns out to be unusable.
#include "strata/core/conversation_spill_dir.hpp"

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
    // (a) No flag and the Strata folder sits off the system drive: the Strata folder's own slots win.
    {
        const SpillDirChoice c = conversation_spill_dir_resolve("", kStrataOffC, kHomeOnC, kSystemDrive);
        check(c.source == SpillDirSource::strata_dir, "a: the Strata folder is the default off the system drive");
        check(c.directory == std::string(kStrataOffC) + "\\slots", "a: <strata dir>\\slots is chosen");
        check(!c.directory.empty() && c.directory.find("D:\\") == 0, "a: the chosen folder stays on D:");
    }
    // (b) No flag and the Strata folder is on the system drive: the user profile wins instead.
    {
        const SpillDirChoice c = conversation_spill_dir_resolve("", kStrataOnC, kHomeOnC, kSystemDrive);
        check(c.source == SpillDirSource::user_home, "b: the system drive sends the default to the user profile");
        check(c.directory == std::string(kHomeOnC) + "\\.strata\\slots", "b: <user profile>\\.strata\\slots is chosen");
        check(c.directory.find("C:\\Strata") == std::string::npos, "b: the Strata folder on C: is NOT used");
    }
    // (c) The flag is set: it wins, wherever the Strata folder is.
    {
        const SpillDirChoice c = conversation_spill_dir_resolve("E:\\parking", kStrataOnC, kHomeOnC, kSystemDrive);
        check(c.source == SpillDirSource::explicit_flag, "c: an explicit flag is its own source");
        check(c.directory == "E:\\parking", "c: the explicit folder wins over both defaults");
    }
    // The drive rule is case-insensitive and recognizes a forward-slash system path.
    {
        const SpillDirChoice c = conversation_spill_dir_resolve("", "c:/Strata", kHomeOnC, "c:");
        check(c.source == SpillDirSource::user_home, "the system-drive match is case-insensitive");
        check(c.directory == std::string(kHomeOnC) + "\\.strata\\slots", "a lowercase C: still routes to the profile");
    }
    // Off Windows (no system drive) the Strata folder always wins, C:-looking path or not.
    {
        const SpillDirChoice c = conversation_spill_dir_resolve("", kStrataOnC, kHomeOnC, "");
        check(c.source == SpillDirSource::strata_dir, "with no system drive the Strata folder is the default");
        check(c.directory == std::string(kStrataOnC) + "\\slots", "no system drive: <strata dir>\\slots is chosen");
    }
    // The engine's "the Strata folder could not be created" fallback: skip_strata_dir sends it to the profile.
    {
        const SpillDirChoice c = conversation_spill_dir_resolve("", kStrataOffC, kHomeOnC, kSystemDrive, /*skip_strata_dir=*/true);
        check(c.source == SpillDirSource::user_home, "skip_strata_dir: the profile is the fallback");
        check(c.directory == std::string(kHomeOnC) + "\\.strata\\slots", "skip_strata_dir: <user profile>\\.strata\\slots");
    }
    // Nothing resolvable: no flag and no usable folder -> empty, source none (the tier stays off, fail-open).
    {
        const SpillDirChoice c = conversation_spill_dir_resolve("", "", "", "");
        check(c.source == SpillDirSource::none, "no inputs: source none");
        check(c.directory.empty(), "no inputs: an empty directory");
        check(!c.reason.empty(), "no inputs: a reason is still reported for the log");
    }
    // No Strata folder but a home: the profile is the default with its own reason.
    {
        const SpillDirChoice c = conversation_spill_dir_resolve("", "", kHomeOnC, kSystemDrive);
        check(c.source == SpillDirSource::user_home, "no Strata folder: the profile is the default");
        check(c.directory == std::string(kHomeOnC) + "\\.strata\\slots", "no Strata folder: <user profile>\\.strata\\slots");
    }
    // Helper checks: a trailing separator is trimmed, and a POSIX home keeps '/' separators.
    {
        check(conversation_spill_strata_slots_dir("D:\\Strata\\") == "D:\\Strata\\slots", "trailing '\\' is trimmed");
        check(conversation_spill_strata_slots_dir("/opt/strata/") == "/opt/strata/slots", "trailing '/' is trimmed (POSIX)");
        check(conversation_spill_user_slots_dir("/home/me") == "/home/me/.strata/slots", "POSIX home uses '/' separators");
        check(conversation_path_drive_root("C:\\x") == "C:", "drive root: C: is detected");
        check(conversation_path_drive_root("d:/x") == "D:", "drive root: a lowercase drive is uppercased");
        check(conversation_path_drive_root("\\\\srv\\share") == "", "drive root: a UNC path has none");
        check(conversation_path_drive_root("relative\\path") == "", "drive root: a relative path has none");
        check(!conversation_path_on_system_drive("C:\\x", ""), "no system drive: no path is on it (off Windows)");
        check(conversation_path_on_system_drive("C:\\x", "C:"), "C: is the C: system drive");
        check(!conversation_path_on_system_drive("D:\\x", "C:"), "D: is not the C: system drive");
    }

    std::printf("conversation_spill_dir_test: %d host-only checks passed\n", checks);
    return 0;
}
