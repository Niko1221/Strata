#pragma once

#include "strata/core/conversation_spill_dir.hpp"   // the drive-root helpers, single-sourced from the spill rule

#include <string>

// --system-prompt-cache-dir's DEFAULT, so the system-prompt cache starts without the operator naming a folder by
// hand - the SAME rule --conversation-cache-spill-dir already resolves (conversation_spill_dir.hpp), applied to the
// system-prompt cache's OWN folder: the cache writes inside the tree where Strata is installed (where the models
// live), and the flag overrides it.  As in the spill module ONLY the rule lives here: it is pure - no filesystem, no
// environment - so it is exercised without the engine, and generate.cpp supplies the three paths and logs the choice
// when the cache opens.
//
// The rule (the one the spill default already follows): a `system-prompt-cache` folder inside Strata's own
// directory, and - when that folder would sit on the SYSTEM DRIVE (C: on Windows) or cannot be created -
// `.strata\system-prompt-cache` inside the user's profile instead.  An explicit --system-prompt-cache-dir always
// wins over both.
//
//   explicit flag set                                    -> the flag                             (source = explicit_flag)
//   else <strata dir>\system-prompt-cache, off the OS    -> <strata dir>\system-prompt-cache      (source = strata_dir)
//   else <user profile>\.strata\system-prompt-cache      -> ...                                  (source = user_home)
//   else (nothing usable)                                -> empty; the cache stays off           (source = none)
//
// `system_drive` is the root of the operating system's volume: "C:" on Windows, empty elsewhere (no drive letters,
// so no path is "on the system drive" and the Strata folder always wins) - exactly the contract the spill module
// documents, and the drive helpers below are the spill module's own.
namespace strata::core {

enum class SystemPromptCacheDirSource {
    explicit_flag,   ///< the operator named the folder; it wins
    strata_dir,      ///< the default <strata dir>\system-prompt-cache, off the system drive
    user_home,       ///< the default <user profile>\.strata\system-prompt-cache (strata folder on the system drive, or unusable)
    none,            ///< nothing resolvable; the system-prompt cache stays off
};

inline const char* system_prompt_cache_dir_source_name(SystemPromptCacheDirSource source) {
    switch (source) {
        case SystemPromptCacheDirSource::explicit_flag: return "explicit";
        case SystemPromptCacheDirSource::strata_dir: return "strata-dir";
        case SystemPromptCacheDirSource::user_home: return "user-home";
        case SystemPromptCacheDirSource::none: return "none";
    }
    return "none";
}

struct SystemPromptCacheDirChoice {
    std::string directory;                        ///< the resolved folder; empty when source == none
    SystemPromptCacheDirSource source = SystemPromptCacheDirSource::none;
    std::string reason;                           ///< one line: exactly what the cache logs at open time
};

/// <strata dir>\system-prompt-cache, keeping the separator style of `strata_dir`; empty when `strata_dir` is empty.
inline std::string system_prompt_cache_strata_dir(const std::string& strata_dir) {
    if (strata_dir.empty()) return std::string();
    std::string trimmed = strata_dir;
    while (!trimmed.empty() && (trimmed.back() == '/' || trimmed.back() == '\\')) trimmed.pop_back();
    const char sep = strata_dir.find('\\') != std::string::npos ? '\\' : '/';
    if (trimmed.empty()) return strata_dir + "system-prompt-cache";   // a bare root such as "/" (Windows keeps "C:")
    return trimmed + sep + "system-prompt-cache";
}

/// %USERPROFILE%\.strata\system-prompt-cache on Windows, $HOME/.strata/system-prompt-cache elsewhere; empty when
/// `user_home` is empty.
inline std::string system_prompt_cache_user_dir(const std::string& user_home) {
    if (user_home.empty()) return std::string();
    std::string trimmed = user_home;
    while (!trimmed.empty() && (trimmed.back() == '/' || trimmed.back() == '\\')) trimmed.pop_back();
    const char sep = user_home.find('\\') != std::string::npos ? '\\' : '/';
    if (trimmed.empty()) trimmed = user_home;
    return trimmed + sep + ".strata" + sep + "system-prompt-cache";
}

/// The pure resolution.  `skip_strata_dir` forces the user-profile fallback (the caller sets it after the Strata
/// folder failed to open) - the "or cannot be created" half of the rule, kept testable.  Same shape and same
/// precedence as conversation_spill_dir_resolve(); only the leaf folder differs.
inline SystemPromptCacheDirChoice system_prompt_cache_dir_resolve(const std::string& explicit_dir,
                                                                 const std::string& strata_dir,
                                                                 const std::string& user_home,
                                                                 const std::string& system_drive,
                                                                 bool skip_strata_dir = false) {
    SystemPromptCacheDirChoice choice;
    // (c) The flag always wins - over both defaults, wherever the Strata folder is.
    if (!explicit_dir.empty()) {
        choice.directory = explicit_dir;
        choice.source = SystemPromptCacheDirSource::explicit_flag;
        choice.reason = "explicit --system-prompt-cache-dir";
        return choice;
    }
    const std::string strata_cache = skip_strata_dir ? std::string() : system_prompt_cache_strata_dir(strata_dir);
    // (a) No flag and the Strata folder is off the system drive: its own system-prompt-cache folder.
    if (!strata_cache.empty() && !conversation_path_on_system_drive(strata_cache, system_drive)) {
        choice.directory = strata_cache;
        choice.source = SystemPromptCacheDirSource::strata_dir;
        choice.reason = "default: the Strata folder is off the system drive";
        if (!system_drive.empty()) choice.reason += " " + system_drive;
        return choice;
    }
    // (b) Else the user profile's .strata\system-prompt-cache - system drive, no Strata folder, or a Strata folder
    //     the cache could not open.
    const std::string user_cache = system_prompt_cache_user_dir(user_home);
    if (!user_cache.empty()) {
        choice.directory = user_cache;
        choice.source = SystemPromptCacheDirSource::user_home;
        if (skip_strata_dir)
            choice.reason = "default: the Strata folder's system-prompt-cache could not be opened";
        else if (strata_cache.empty())
            choice.reason = "default: no Strata folder was resolved";
        else
            choice.reason = "default: the Strata folder " + strata_dir + " sits on the system drive" +
                            (system_drive.empty() ? "" : " " + system_drive);
        return choice;
    }
    // Nothing usable: the cache stays off (fail-open, as the rest of the engine).
    choice.source = SystemPromptCacheDirSource::none;
    choice.reason = "no --system-prompt-cache-dir, no Strata folder and no user profile: the system-prompt cache stays off";
    return choice;
}

} // namespace strata::core
