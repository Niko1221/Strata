#pragma once

#include <string>

// --conversation-cache-spill-dir's DEFAULT, so the durable disk tier (the parked conversations written to
// disk) starts without naming a folder by hand.  ONLY the rule lives here: it is pure - no filesystem, no
// environment - so it is exercised without the engine, the way conversation_memory.hpp exercises the parking
// policy.  generate.cpp supplies the three paths and logs the choice when the tier opens.
//
// The rule (the operator's directive): a `slots` folder inside Strata's own directory, and - when that folder
// would sit on the SYSTEM DRIVE (C: on Windows) or cannot be created - `.strata\slots` inside the user's
// profile instead.  An explicit --conversation-cache-spill-dir always wins over both.
//
//   explicit flag set                                    -> the flag            (source = explicit_flag)
//   else <strata dir>\slots, and the drive is not the OS  -> <strata dir>\slots  (source = strata_dir)
//   else <user profile>\.strata\slots                     -> ...                 (source = user_home)
//   else (nothing usable)                                 -> empty; the tier stays off (fail-open, source = none)
//
// `system_drive` is the root of the operating system's volume: "C:" on Windows, empty elsewhere (no drive
// letters, so no path is "on the system drive" and the Strata folder always wins).

namespace strata::core {

enum class SpillDirSource {
    explicit_flag,   ///< the operator named the folder; it wins
    strata_dir,      ///< the default <strata dir>\slots, off the system drive
    user_home,       ///< the default <user profile>\.strata\slots (strata folder on the system drive, or unusable)
    none,            ///< nothing resolvable; the disk tier stays off
};

inline const char* spill_dir_source_name(SpillDirSource source) {
    switch (source) {
        case SpillDirSource::explicit_flag: return "explicit";
        case SpillDirSource::strata_dir: return "strata-dir";
        case SpillDirSource::user_home: return "user-home";
        case SpillDirSource::none: return "none";
    }
    return "none";
}

struct SpillDirChoice {
    std::string directory;                       ///< the resolved folder; empty when source == none
    SpillDirSource source = SpillDirSource::none;
    std::string reason;                          ///< one line: exactly what the tier logs at open time
};

/// <strata dir>\slots, keeping the separator style of `strata_dir`; empty when `strata_dir` is empty.
inline std::string conversation_spill_strata_slots_dir(const std::string& strata_dir) {
    if (strata_dir.empty()) return std::string();
    std::string trimmed = strata_dir;
    while (!trimmed.empty() && (trimmed.back() == '/' || trimmed.back() == '\\')) trimmed.pop_back();
    const char sep = strata_dir.find('\\') != std::string::npos ? '\\' : '/';
    if (trimmed.empty()) return strata_dir + "slots";   // a bare root such as "/" (Windows keeps "C:")
    return trimmed + sep + "slots";
}

/// %USERPROFILE%\.strata\slots on Windows, $HOME/.strata/slots elsewhere; empty when `user_home` is empty.
inline std::string conversation_spill_user_slots_dir(const std::string& user_home) {
    if (user_home.empty()) return std::string();
    std::string trimmed = user_home;
    while (!trimmed.empty() && (trimmed.back() == '/' || trimmed.back() == '\\')) trimmed.pop_back();
    const char sep = user_home.find('\\') != std::string::npos ? '\\' : '/';
    if (trimmed.empty()) trimmed = user_home;
    return trimmed + sep + ".strata" + sep + "slots";
}

/// The drive root of `path` ("C:" for "C:\x" or "c:/x"), uppercased; empty for a relative or UNC path.
inline std::string conversation_path_drive_root(const std::string& path) {
    if (path.size() < 2 || path[1] != ':') return std::string();
    const char c = path[0];
    if (!((c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z'))) return std::string();
    std::string drive = path.substr(0, 2);
    if (drive[0] >= 'a' && drive[0] <= 'z') drive[0] = (char) (drive[0] - 'a' + 'A');
    return drive;
}

/// Whether `path` names a volume on the system drive.  An empty `system_drive` (non-Windows) means "no",
/// so off Windows the Strata folder is always the default.
inline bool conversation_path_on_system_drive(const std::string& path, const std::string& system_drive) {
    if (system_drive.empty()) return false;
    std::string want = system_drive;
    while (!want.empty() && (want.back() == '/' || want.back() == '\\')) want.pop_back();
    if (want.size() < 2 || want[1] != ':') return false;
    if (want[0] >= 'a' && want[0] <= 'z') want[0] = (char) (want[0] - 'a' + 'A');
    const std::string drive = conversation_path_drive_root(path);
    return !drive.empty() && drive == want.substr(0, 2);
}

/// The pure resolution.  `skip_strata_dir` forces the user-profile fallback (the caller sets it after the
/// Strata folder failed to open) - the "or cannot be created" half of the rule, kept testable.
inline SpillDirChoice conversation_spill_dir_resolve(const std::string& explicit_dir,
                                                     const std::string& strata_dir,
                                                     const std::string& user_home,
                                                     const std::string& system_drive,
                                                     bool skip_strata_dir = false) {
    SpillDirChoice choice;
    // (c) The flag always wins - over both defaults, wherever the Strata folder is.
    if (!explicit_dir.empty()) {
        choice.directory = explicit_dir;
        choice.source = SpillDirSource::explicit_flag;
        choice.reason = "explicit --conversation-cache-spill-dir";
        return choice;
    }
    const std::string strata_slots = skip_strata_dir ? std::string() : conversation_spill_strata_slots_dir(strata_dir);
    // (a) No flag and the Strata folder is off the system drive: its own slots folder.
    if (!strata_slots.empty() && !conversation_path_on_system_drive(strata_slots, system_drive)) {
        choice.directory = strata_slots;
        choice.source = SpillDirSource::strata_dir;
        choice.reason = "default: the Strata folder is off the system drive";
        if (!system_drive.empty()) choice.reason += " " + system_drive;
        return choice;
    }
    // (b) Else the user profile's .strata\slots - system drive, no Strata folder, or a Strata folder the engine
    //     could not open.
    const std::string user_slots = conversation_spill_user_slots_dir(user_home);
    if (!user_slots.empty()) {
        choice.directory = user_slots;
        choice.source = SpillDirSource::user_home;
        if (skip_strata_dir)
            choice.reason = "default: the Strata folder's slots could not be opened";
        else if (strata_slots.empty())
            choice.reason = "default: no Strata folder was resolved";
        else
            choice.reason = "default: the Strata folder " + strata_dir + " sits on the system drive" +
                            (system_drive.empty() ? "" : " " + system_drive);
        return choice;
    }
    // Nothing usable: the tier stays off (fail-open, as the rest of the engine).
    choice.source = SpillDirSource::none;
    choice.reason = "no --conversation-cache-spill-dir, no Strata folder and no user profile: the disk tier stays off";
    return choice;
}

} // namespace strata::core
