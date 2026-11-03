//! Strata Dev: the developer variant of the app, built with `--features dev-build` and `tauri.dev-build.conf.json`
//! (see docs/DEV_BUILD.md). Without the feature none of this is compiled and the app is the Live app, unchanged.
//!
//! Dev runs beside Live without touching it: its own name and identifier (the config), its own window port (47832,
//! main.rs), no app updater (updater.rs), and its own data folders. The folders are set here, in the shell's own
//! environment before anything else runs, so the desktop log, the App Host, Control and the engine all inherit them.
//! A variable that is already set is kept, so a developer can still point one place somewhere else by hand.
//!
//! Shared with Live, read only: the engine components (`%LOCALAPPDATA%\Strata\components`) and the model files.

use std::path::{Path, PathBuf};

/// The window and App Host port of Dev (Live: 47831).
pub const PREFERRED_PORT: u16 = 47832;
/// The port Dev's engine and server listen on (Live: 8080 for the llama.cpp path, 18500 and up for the Strata engine).
pub const ENGINE_PORT: u16 = 8081;
/// The window title and the name in messages.
pub const TITLE: &str = "Strata Dev";

/// `%LOCALAPPDATA%\StrataDev` (Linux and macOS: `$XDG_DATA_HOME/strata-dev`, default `~/.local/share/strata-dev`).
pub fn root() -> PathBuf {
    if cfg!(windows) {
        std::env::var_os("LOCALAPPDATA")
            .map(PathBuf::from)
            .or_else(|| std::env::var_os("USERPROFILE").map(|h| PathBuf::from(h).join("AppData").join("Local")))
            .unwrap_or_else(std::env::temp_dir)
            .join("StrataDev")
    } else {
        std::env::var_os("XDG_DATA_HOME")
            .map(PathBuf::from)
            .or_else(|| std::env::var_os("HOME").map(|h| PathBuf::from(h).join(".local").join("share")))
            .unwrap_or_else(std::env::temp_dir)
            .join("strata-dev")
    }
}

/// Every variable Dev sets, with its value under `root`. Pure, so the test can check the whole list.
pub fn variables(root: &Path, config: &Path) -> Vec<(&'static str, String)> {
    let at = |name: &str| root.join(name).display().to_string();
    vec![
        ("STRATA_DEV", "1".into()),
        ("STRATA_DEV_CONFIG", config.display().to_string()),
        // chats, host state, skills, agent
        ("STRATA_HOME", at("home")),
        // profiles and presets
        ("STRATA_PROFILE_DIR", at("profiles")),
        // tuning, Time-skip and the rest of the per-user state (runtime/, procs/)
        ("STRATA_STATE_DIR", at("state")),
        ("STRATA_TUNE_DIR", at("tuning")),
        ("STRATA_TIMESKIP_DIR", at("time-skip")),
        ("STRATA_RUNTIME_DIR", at("runtime")),
        // desktop logs and shell-prefs.json
        ("STRATA_DESKTOP_DATA_DIR", at("desktop")),
        // the app's settings file and install folder (run configs), and the Terminal settings file the app mirrors its data folder into
        ("STRATA_SETTINGS_FILE", at("app-settings.json")),
        ("STRATA_INSTALL_DIR", at("install")),
        ("STRATA_TERMINAL_SETTINGS_FILE", at("terminal-settings.json")),
        // the folder chats work in when nothing else is chosen (Live: Documents\Strata\Workspace)
        ("STRATA_WORKSPACE", at("workspace")),
        ("STRATA_ENGINE_PORT", ENGINE_PORT.to_string()),
        // the access repair at start walks Live's folders (%APPDATA%\Strata, %LOCALAPPDATA%\Strata): not from Dev
        ("STRATA_NO_REPAIR_ACCESS", "1".into()),
    ]
}

/// The repository's `dev.config.json`, found from where this build was made (a Dev build is always made from a checkout).
fn config_file() -> PathBuf {
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("..").join("..").join("..").join("dev.config.json");
    // Windows' canonical form is a verbatim path (\\?\C:\...); the host and the page show the plain one
    let path = path.canonicalize().unwrap_or(path);
    match path.to_str().and_then(|p| p.strip_prefix(r"\\?\")) {
        Some(plain) if !plain.starts_with("UNC\\") => PathBuf::from(plain),
        _ => path,
    }
}

/// Set the Dev variables in this process (called first in `main`, while the process has one thread). Kept when already set.
pub fn apply_env() {
    let root = root();
    for (name, value) in variables(&root, &config_file()) {
        if std::env::var_os(name).is_none() {
            std::env::set_var(name, value);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_dev_folder_is_under_the_dev_root_and_the_ports_are_not_lives() {
        let root = PathBuf::from("X").join("StrataDev");
        let vars = variables(&root, Path::new("dev.config.json"));
        for (name, value) in &vars {
            if name.ends_with("_DIR") || name.ends_with("_HOME") || name.ends_with("_FILE") || *name == "STRATA_WORKSPACE" {
                assert!(Path::new(value).starts_with(&root), "{name}={value}");
            }
        }
        assert!(vars.iter().any(|(n, v)| *n == "STRATA_DEV" && v == "1"));
        assert!(!vars.iter().any(|(n, _)| *n == "STRATA_COMPONENTS_DIR"), "the engine components are shared with Live");
        assert_ne!(PREFERRED_PORT, 47831);
        assert_ne!(ENGINE_PORT, 8080);
    }
}
