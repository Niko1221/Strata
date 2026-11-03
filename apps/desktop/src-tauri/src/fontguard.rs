// Linux: keep the colour-emoji font that WebKitGTK cannot draw away from the app window.
//
// The WebKitGTK inside the AppImage paints a COLRv1 colour font through Skia, and its
// `colrv1_configure_skpaint` reads past the end of a gradient's colour-stop list. The
// library is built with libstdc++ range checks on, so the read becomes `abort()` and
// WebKitWebProcess dies; the window goes grey. Fedora 44 and Ubuntu 25.10 ship
// `Noto-COLRv1.ttf` as the default colour emoji font, and a chat that contains an emoji
// then takes the window down.
//
// fontconfig reads one file when FONTCONFIG_FILE is set and that file replaces the
// normal configuration, so the file written here includes the system configuration first
// and only then rejects the one font. The emoji still arrive, from the outline face
// ("Noto Emoji") instead of the colour one; nothing else about the fonts changes.
//
// This runs in the window process only, never in the shell: the shell owns the host, the
// engine and the agent, and they must keep the fonts the person chose.

use std::path::{Path, PathBuf};

/// The rejected file names. Only the COLRv1 colour faces are listed: the old bitmap
/// (`NotoColorEmoji.ttf` with `CBDT`) and the outline (`NotoEmoji-Regular.ttf`) faces
/// draw fine, and hiding them would leave a box instead of an emoji.
const COLORED_FONT_GLOBS: &[&str] = &["*/Noto-COLRv1.ttf", "*/NotoSansCOLRv1-Regular.ttf"];

/// The families those files are known by, so the fallback picks the outline face rather
/// than any other font that happens to carry the codepoints.
const EMOJI_FALLBACK: &str = "Noto Emoji";

/// Where the system fontconfig configuration is. `FONTCONFIG_FILE` replaces it, so the
/// file written here has to pull it back in; when it is not there, this whole thing is
/// skipped rather than leaving the window without any font at all.
const SYSTEM_CONFIG: &str = "/etc/fonts/fonts.conf";

/// Opt-out, for the person who wants their own font handling: `STRATA_FONT_CONFIG=off`.
const DISABLE: &str = "off";

/// The font directories fontconfig scans, which is where a colour emoji font is found.
fn font_dirs() -> Vec<PathBuf> {
    let mut dirs = vec![
        PathBuf::from("/usr/share/fonts"),
        PathBuf::from("/usr/local/share/fonts"),
    ];
    if let Some(home) = std::env::var_os("HOME") {
        let home = PathBuf::from(home);
        dirs.push(home.join(".local/share/fonts"));
        dirs.push(home.join(".fonts"));
    }
    dirs
}

/// A `*COLRv1*` font file the person installed, if any. The scan is shallow and stops at
/// the first hit, so a window that starts on a machine without one pays four directory
/// reads and nothing else.
pub fn colrv1_font_present(dirs: &[PathBuf]) -> Option<PathBuf> {
    fn scan(dir: &Path, depth: u8, out: &mut Option<PathBuf>) {
        if out.is_some() || depth == 0 {
            return;
        }
        let Ok(entries) = std::fs::read_dir(dir) else { return };
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() {
                scan(&path, depth - 1, out);
            } else if path
                .file_name()
                .map(|n| n.to_string_lossy().to_ascii_lowercase().contains("colrv1"))
                .unwrap_or(false)
            {
                *out = Some(path);
                return;
            }
        }
    }
    let mut found = None;
    for dir in dirs {
        scan(dir, 4, &mut found);
        if found.is_some() {
            return found;
        }
    }
    found
}

/// The configuration file's contents: the system's own rules, then this one rejection.
pub fn config_text() -> String {
    // Lines, not one escaped block: a `\` continuation swallows the leading spaces of the
    // next line, which is exactly the indentation a fontconfig file should keep readable.
    let mut lines: Vec<String> = vec![
        "<?xml version=\"1.0\"?>".into(),
        "<!DOCTYPE fontconfig SYSTEM \"urn:fontconfig:fonts.dtd\">".into(),
        "<!-- Written by Strata. It hides the COLRv1 colour emoji font from the Strata app".into(),
        "     window only, because the WebKitGTK in the AppImage aborts while drawing one.".into(),
        "     Everything else is the system's own fontconfig setup, pulled in below.".into(),
        "     To stop Strata writing this file: STRATA_FONT_CONFIG=off".into(),
        "     To use your own file instead: set FONTCONFIG_FILE yourself. -->".into(),
        "<fontconfig>".into(),
        format!("  <include ignore_missing=\"yes\">{SYSTEM_CONFIG}</include>"),
        "  <alias binding=\"strong\">".into(),
        "    <family>Noto Color Emoji</family>".into(),
        format!("    <prefer><family>{EMOJI_FALLBACK}</family></prefer>"),
        "  </alias>".into(),
    ];
    // fontconfig rejects a font only inside <selectfont>.
    lines.push("  <selectfont>".into());
    for glob in COLORED_FONT_GLOBS {
        lines.push(format!("    <rejectfont>\n      <glob>{glob}</glob>\n    </rejectfont>"));
    }
    lines.push("  </selectfont>".into());
    lines.push("</fontconfig>".into());
    lines.join("\n") + "\n"
}

/// Write the file and return it, or `None` when it could not be written. The file is
/// rewritten only when its contents differ, so a normal start-up touches the disk once
/// and a leftover from an older Strata, or a half-written file, is put right by itself.
/// This is a generated file, not a place to edit: `FONTCONFIG_FILE` or
/// `STRATA_FONT_CONFIG` is how a person sets their own.
pub fn ensure_config(dir: &Path) -> Option<PathBuf> {
    if !Path::new(SYSTEM_CONFIG).exists() {
        return None;
    }
    std::fs::create_dir_all(dir).ok()?;
    let path = dir.join("fonts-no-colrv1.conf");
    let want = config_text();
    match std::fs::read_to_string(&path) {
        Ok(have) if have == want => return Some(path),
        Ok(_) => {}
        Err(_) => {}
    }
    // Written through a temporary file, then moved: a second instance can start at the
    // same moment, and fontconfig must never be handed a half-written file.
    let tmp = dir.join(format!("fonts-no-colrv1.conf.{}.tmp", std::process::id()));
    std::fs::write(&tmp, want.as_bytes()).ok()?;
    if std::fs::rename(&tmp, &path).is_err() {
        let _ = std::fs::remove_file(&tmp);
        // the other writer got there first: take its file only when it is the one we wanted
        let have = std::fs::read_to_string(&path).ok()?;
        return (have == want).then_some(path);
    }
    Some(path)
}

/// The file the window process should hand to fontconfig, or `None` to leave the
/// environment exactly as it is. Checked in this order, so a person's own choice always
/// wins and a machine without the crashing font is untouched:
///
/// 1. `FONTCONFIG_FILE` already set → theirs.
/// 2. `STRATA_FONT_CONFIG=off` → nothing.
/// 3. `STRATA_FONT_CONFIG=<path>` → theirs, when the file is there.
/// 4. No COLRv1 font installed → nothing.
pub fn config_for(env: &dyn Fn(&str) -> Option<String>, dir: &Path) -> Option<PathBuf> {
    if env("FONTCONFIG_FILE").is_some() || env("STRATA_FONT_CONFIG").as_deref() == Some(DISABLE) {
        return None;
    }
    if let Some(custom) = env("STRATA_FONT_CONFIG") {
        let p = PathBuf::from(custom);
        return p.is_file().then_some(p);
    }
    colrv1_font_present(&font_dirs())?;
    ensure_config(dir)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn env_from(pairs: &[(&str, &str)]) -> impl Fn(&str) -> Option<String> {
        let owned: Vec<(String, String)> = pairs.iter().map(|(k, v)| (k.to_string(), v.to_string())).collect();
        move |k: &str| owned.iter().find(|(a, _)| a == k).map(|(_, v)| v.clone())
    }

    fn scratch(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("strata-fontguard-{name}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    #[test]
    fn the_written_file_pulls_in_the_system_config_and_rejects_only_the_colrv1_faces() {
        let text = config_text();
        assert!(text.starts_with("<?xml"), "fontconfig wants a declared document");
        assert!(text.contains(SYSTEM_CONFIG), "must keep the system's fonts: {text}");
        assert!(!text.contains('{') && !text.contains('}'), "every placeholder must be filled in: {text}");
        // fontconfig reads a rejection only inside <selectfont>, so the nesting is the point
        let select = text.find("<selectfont>").expect("the rejections need a <selectfont>");
        let select_end = text.find("</selectfont>").expect("<selectfont> must be closed");
        assert!(select < select_end, "<selectfont> must be closed in order");
        assert_eq!(text.matches("<rejectfont>").count(), COLORED_FONT_GLOBS.len(), "one rejection each");
        for glob in COLORED_FONT_GLOBS {
            let at = text.find(glob).unwrap_or_else(|| panic!("must reject {glob}"));
            assert!(at > select && at < select_end, "{glob} must be inside <selectfont>");
        }
        // the faces that draw correctly must stay in
        assert!(!text.contains("NotoColorEmoji.ttf"), "the bitmap face must not be rejected");
        assert!(!text.contains("NotoEmoji-Regular.ttf"), "the outline face must not be rejected");
        assert!(text.contains(EMOJI_FALLBACK), "emoji need a named fallback");
    }

    #[test]
    fn a_persons_own_fontconfig_or_an_opt_out_leaves_the_environment_alone() {
        let dir = scratch("own");
        let with_fontconfig = env_from(&[("FONTCONFIG_FILE", "/home/me/fonts.conf")]);
        assert_eq!(config_for(&with_fontconfig, &dir), None, "FONTCONFIG_FILE is theirs to set");
        let off = env_from(&[("STRATA_FONT_CONFIG", "off")]);
        assert_eq!(config_for(&off, &dir), None, "STRATA_FONT_CONFIG=off must be respected");
        let their_file = dir.join("theirs.conf");
        std::fs::write(&their_file, b"<fontconfig/>").unwrap();
        let custom = env_from(&[("STRATA_FONT_CONFIG", their_file.to_str().unwrap())]);
        assert_eq!(config_for(&custom, &dir), Some(their_file), "a given file is used as it is");
        let missing = env_from(&[("STRATA_FONT_CONFIG", "/no/such/fonts.conf")]);
        assert_eq!(config_for(&missing, &dir), None, "a file that is not there must not be handed over");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_font_dir_without_a_colrv1_face_is_left_alone() {
        let dir = scratch("none");
        let fonts = dir.join("fonts");
        std::fs::create_dir_all(fonts.join("noto")).unwrap();
        std::fs::write(fonts.join("noto/NotoColorEmoji.ttf"), b"bitmap").unwrap();
        std::fs::write(fonts.join("noto/NotoEmoji-Regular.ttf"), b"outline").unwrap();
        assert_eq!(colrv1_font_present(&[fonts.clone()]), None, "the faces that draw fine are not a reason");
        std::fs::write(fonts.join("noto/Noto-COLRv1.ttf"), b"colour").unwrap();
        assert!(colrv1_font_present(&[fonts]).is_some(), "a COLRv1 face must be found");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn the_file_is_written_once_and_an_out_of_date_one_is_put_right() {
        if !Path::new(SYSTEM_CONFIG).exists() {
            return; // a machine with no fontconfig there: nothing is written, by design
        }
        let dir = scratch("write");
        let first = ensure_config(&dir).expect("the file is written where the system config is");
        assert!(first.is_file());
        assert_eq!(ensure_config(&dir).unwrap(), first, "the same path every start");
        // a file left by an older Strata is refreshed, not honoured
        std::fs::write(&first, "<fontconfig/>").unwrap();
        assert_eq!(ensure_config(&dir), Some(first.clone()), "a file from an older Strata is put right");
        assert_eq!(std::fs::read_to_string(&first).unwrap(), config_text(), "its contents are the current ones");
        let litter = std::fs::read_dir(&dir).unwrap().filter_map(|e| e.ok()).collect::<Vec<_>>();
        assert_eq!(litter.len(), 1, "one file and no temporary left behind: {:?}", litter);
        let _ = std::fs::remove_dir_all(&dir);
    }
}
