//! The shell's own choices and the two things the host asks of it besides updates (#3402, #3477).
//!
//! `<Strata data folder>/shell-prefs.json` is written by the App Host when the person changes a setting
//! (`app/host/server/shell-prefs.mjs`) and read here when Strata starts, before any page exists:
//!
//!   { "port": 0 | 1024..65535, "tray": false, "language": "en" }
//!
//! * `port`: the port the window's origin uses. 0 is the default, 47831. When the port is busy the shell takes a free one and tells the
//!   host (`STRATA_PORT_NOTICE=busy:<wanted>:<used>`, or `reserved:<wanted>:<used>` when nothing listens on it: the system keeps the port
//!   back, #7244), which shows the person a warning: the chats and settings of the page are kept per port, so a different port starts
//!   empty. The port taken is remembered in `shell-fallback-port.json` (written by the shell only) and used again at the next start while
//!   it is free, so the page keeps one origin.
//! * `tray`: closing the window hides it and keeps Strata running in the tray (Windows, macOS). The tray menu follows `language`.
//! * `/shell/notify` on the loopback bridge shows a desktop notification (an approval is waiting) whether the window is open or hidden.

use std::path::PathBuf;

use serde_json::Value;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Prefs {
    pub port: u16,
    pub tray: bool,
    pub language: String,
}

impl Default for Prefs {
    fn default() -> Self {
        Prefs { port: 0, tray: false, language: "en".into() }
    }
}

pub const LANGUAGES: [&str; 10] = ["en", "zh", "ja", "de", "fr", "es", "ru", "pt", "ko", "he"];

pub fn prefs_path() -> PathBuf {
    crate::desktoplog::data_dir().join("shell-prefs.json")
}

/// The file's text as choices; anything missing or out of range is the default.
pub fn parse_prefs(text: &str) -> Prefs {
    let value: Value = serde_json::from_str(text.trim_start_matches('\u{feff}')).unwrap_or(Value::Null);
    let port = value.get("port").and_then(Value::as_u64).filter(|p| (1024..=65535).contains(p)).unwrap_or(0) as u16;
    let tray = value.get("tray").and_then(Value::as_bool).unwrap_or(false);
    let language = value
        .get("language")
        .and_then(Value::as_str)
        .filter(|l| LANGUAGES.contains(l))
        .unwrap_or("en")
        .to_string();
    Prefs { port, tray, language }
}

pub fn read_prefs() -> Prefs {
    std::fs::read_to_string(prefs_path()).map(|t| parse_prefs(&t)).unwrap_or_default()
}

/// The port to ask for: the person's choice, else the app's usual one.
pub fn wanted_port(prefs: &Prefs, usual: u16) -> u16 {
    if prefs.port != 0 { prefs.port } else { usual }
}

/// The notice the host shows when the wanted port was taken.
pub fn busy_notice(wanted: u16, used: u16) -> String {
    format!("busy:{wanted}:{used}")
}

/// The notice when the wanted port could not be bound and NOTHING listens on it (#7244): Windows keeps blocks of ports for itself (WSL
/// mirrored networking, Hyper-V, Docker) and answers a bind there like a port that is in use, without listing the block anywhere.
pub fn reserved_notice(wanted: u16, used: u16) -> String {
    format!("reserved:{wanted}:{used}")
}

/// What stands on a port the app could not bind.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Holder {
    /// A program answered a connection: it is in use.
    Program,
    /// The connection was refused: nobody listens, so the port is reserved by the system.
    Nobody,
    /// Neither (a timeout, a firewall): not told apart, the old wording is kept.
    Unknown,
}

/// The holder from what a connection attempt to the port gave: `None` = it connected, else the error kind.
pub fn holder_from(attempt: Option<std::io::ErrorKind>) -> Holder {
    match attempt {
        None => Holder::Program,
        Some(std::io::ErrorKind::ConnectionRefused) => Holder::Nobody,
        Some(_) => Holder::Unknown,
    }
}

/// One connection attempt to 127.0.0.1:port. A closed loopback port answers "refused" at once on Linux and macOS but only after the SYN retries on
/// Windows (about 2 s, measured on the owner PC: an 800 ms limit gave TimedOut), so the limit is 4 s; it is paid only when the wanted port could not be bound.
pub fn holder_of(port: u16) -> Holder {
    let address = std::net::SocketAddr::from(([127, 0, 0, 1], port));
    holder_from(match std::net::TcpStream::connect_timeout(&address, std::time::Duration::from_millis(4000)) {
        Ok(_) => None,
        Err(e) => Some(e.kind()),
    })
}

/// `<data folder>/shell-fallback-port.json`: the port the app used the last time its wanted port could not be bound, so a computer where the usual
/// port is reserved keeps ONE port (the page's chats and settings belong to the port, #3402) instead of a new random one on every start (#7244).
/// The shell alone writes it, so it never meets the host's write of shell-prefs.json.
pub fn fallback_path() -> PathBuf {
    crate::desktoplog::data_dir().join("shell-fallback-port.json")
}

/// The remembered port for this wanted port: only one saved for the same wanted port, in range, and not the wanted port itself.
pub fn parse_fallback(text: &str, wanted: u16) -> Option<u16> {
    let value: Value = serde_json::from_str(text.trim_start_matches('\u{feff}')).ok()?;
    let saved_wanted = value.get("wanted").and_then(Value::as_u64)?;
    let port = value.get("port").and_then(Value::as_u64).filter(|p| (1024..=65535).contains(p))? as u16;
    (saved_wanted == wanted as u64 && port != wanted).then_some(port)
}

pub fn fallback_text(wanted: u16, port: u16) -> String {
    format!("{{\"wanted\":{wanted},\"port\":{port}}}\n")
}

pub fn read_fallback(wanted: u16) -> Option<u16> {
    std::fs::read_to_string(fallback_path()).ok().and_then(|t| parse_fallback(&t, wanted))
}

/// Best effort: a folder that cannot be written only means the next start picks again, as before.
pub fn write_fallback(wanted: u16, port: u16) {
    let path = fallback_path();
    if let Some(dir) = path.parent() {
        let _ = std::fs::create_dir_all(dir);
    }
    let _ = std::fs::write(path, fallback_text(wanted, port));
}

/// `%41%20b` -> "A b" (a query value from the host). Bad escapes are kept as typed.
pub fn percent_decode(text: &str) -> String {
    let bytes = text.as_bytes();
    let mut out = Vec::with_capacity(bytes.len());
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] == b'%' && i + 2 < bytes.len() {
            if let Some(v) = std::str::from_utf8(&bytes[i + 1..i + 3]).ok().and_then(|h| u8::from_str_radix(h, 16).ok()) {
                out.push(v);
                i += 3;
                continue;
            }
        }
        out.push(if bytes[i] == b'+' { b' ' } else { bytes[i] });
        i += 1;
    }
    String::from_utf8_lossy(&out).into_owned()
}

/// The value of `key` in the query of a request path (`/shell/notify?title=..&body=..`).
pub fn query_value(path: &str, key: &str) -> Option<String> {
    let query = path.split_once('?')?.1;
    query.split('&').find_map(|pair| {
        let (k, v) = pair.split_once('=')?;
        (k == key).then(|| percent_decode(v))
    })
}

/// A short single-line text for a notification: no control characters, bounded.
pub fn clean_text(text: &str, max: usize) -> String {
    text.chars().filter(|c| !c.is_control()).take(max).collect::<String>().trim().to_string()
}

/// The tray menu: (open, quit, tooltip), in the app's language.
pub fn tray_labels(language: &str) -> (&'static str, &'static str, &'static str) {
    match language {
        "zh" => ("打开 Strata", "退出 Strata", "Strata"),
        "ja" => ("Strata を開く", "Strata を終了", "Strata"),
        "de" => ("Strata öffnen", "Strata beenden", "Strata"),
        "fr" => ("Ouvrir Strata", "Quitter Strata", "Strata"),
        "es" => ("Abrir Strata", "Salir de Strata", "Strata"),
        "ru" => ("Открыть Strata", "Выйти из Strata", "Strata"),
        "pt" => ("Abrir o Strata", "Sair do Strata", "Strata"),
        "ko" => ("Strata 열기", "Strata 종료", "Strata"),
        "he" => ("פתיחת Strata", "יציאה מ-Strata", "Strata"),
        _ => ("Open Strata", "Quit Strata", "Strata"),
    }
}

#[cfg(not(target_os = "linux"))]
mod native {
    use tauri::menu::{Menu, MenuItem};
    use tauri::tray::{MouseButton, MouseButtonState, TrayIconBuilder, TrayIconEvent};
    use tauri::{AppHandle, Manager};
    use tauri_plugin_notification::NotificationExt;

    const TRAY_ID: &str = "strata-tray";

    fn show_main(app: &AppHandle) {
        if let Some(window) = app.get_webview_window("main") {
            let _ = window.unminimize();
            let _ = window.show();
            let _ = window.set_focus();
        }
    }

    /// Hide the window and keep Strata running in the tray (made on the first need, with the menu in the app's language).
    pub fn hide_to_tray(app: &AppHandle, language: &str) {
        if app.tray_by_id(TRAY_ID).is_none() {
            let (open, quit, tip) = super::tray_labels(language);
            let built = (|| -> tauri::Result<()> {
                let open_item = MenuItem::with_id(app, "open", open, true, None::<&str>)?;
                let quit_item = MenuItem::with_id(app, "quit", quit, true, None::<&str>)?;
                let menu = Menu::with_items(app, &[&open_item, &quit_item])?;
                let mut builder = TrayIconBuilder::with_id(TRAY_ID)
                    .tooltip(tip)
                    .menu(&menu)
                    .show_menu_on_left_click(false)
                    .on_menu_event(|app, event| match event.id().as_ref() {
                        "open" => show_main(app),
                        "quit" => app.exit(0),
                        _ => {}
                    })
                    .on_tray_icon_event(|tray, event| {
                        if let TrayIconEvent::Click { button: MouseButton::Left, button_state: MouseButtonState::Up, .. } = event {
                            show_main(tray.app_handle());
                        }
                    });
                if let Some(icon) = app.default_window_icon() {
                    builder = builder.icon(icon.clone());
                }
                builder.build(app)?;
                Ok(())
            })();
            if built.is_err() {
                // No tray could be made (a desktop without one): do not hide a window nobody could bring back.
                return;
            }
        }
        if let Some(window) = app.get_webview_window("main") {
            let _ = window.hide();
        }
    }

    pub fn tray_exists(app: &AppHandle) -> bool {
        app.tray_by_id(TRAY_ID).is_some()
    }

    /// Show a desktop notification; a hidden or minimised window also asks for attention (the taskbar button flashes).
    pub fn notify(app: &AppHandle, title: &str, body: &str) -> bool {
        if title.is_empty() {
            return false;
        }
        if let Some(window) = app.get_webview_window("main") {
            let hidden = !window.is_visible().unwrap_or(true) || window.is_minimized().unwrap_or(false);
            if hidden {
                let _ = window.request_user_attention(Some(tauri::UserAttentionType::Informational));
            }
        }
        app.notification().builder().title(title).body(body).show().is_ok()
    }
}

#[cfg(not(target_os = "linux"))]
pub use native::{hide_to_tray, notify, tray_exists};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_file_gives_choices_and_anything_wrong_is_the_default() {
        assert_eq!(parse_prefs(""), Prefs::default());
        assert_eq!(parse_prefs("{ not json"), Prefs::default());
        let p = parse_prefs(r#"{"port":47900,"tray":true,"language":"de"}"#);
        assert_eq!((p.port, p.tray, p.language.as_str()), (47900, true, "de"));
        assert_eq!(parse_prefs("\u{feff}{\"port\":47900}").port, 47900);
        assert_eq!(parse_prefs(r#"{"port":80}"#).port, 0, "a port below 1024 is not used");
        assert_eq!(parse_prefs(r#"{"port":70000}"#).port, 0);
        assert_eq!(parse_prefs(r#"{"port":"47831"}"#).port, 0, "a text is not a number");
        assert_eq!(parse_prefs(r#"{"tray":"yes","language":"xx"}"#), Prefs::default());
    }

    #[test]
    fn the_wanted_port_is_the_choice_or_the_usual_one() {
        assert_eq!(wanted_port(&Prefs::default(), 47831), 47831);
        assert_eq!(wanted_port(&Prefs { port: 47900, ..Prefs::default() }, 47831), 47900);
        assert_eq!(busy_notice(47831, 51234), "busy:47831:51234");
    }

    #[test]
    fn a_reserved_port_is_told_from_a_busy_one_by_what_a_connection_gives() {
        use std::io::ErrorKind;
        assert_eq!(reserved_notice(47831, 30159), "reserved:47831:30159");
        assert_eq!(holder_from(None), Holder::Program, "somebody answered: in use");
        assert_eq!(holder_from(Some(ErrorKind::ConnectionRefused)), Holder::Nobody, "nobody listens: the system keeps the port back");
        assert_eq!(holder_from(Some(ErrorKind::TimedOut)), Holder::Unknown, "a firewall that drops is not told apart");
        assert_eq!(holder_from(Some(ErrorKind::PermissionDenied)), Holder::Unknown);
    }

    #[test]
    fn a_listening_port_is_a_program_and_a_closed_one_is_nobody() {
        let listener = std::net::TcpListener::bind(("127.0.0.1", 0)).unwrap();
        let open = listener.local_addr().unwrap().port();
        assert_eq!(holder_of(open), Holder::Program);
        drop(listener);
        assert_eq!(holder_of(open), Holder::Nobody);
    }

    #[test]
    fn the_fallback_port_is_kept_for_the_port_it_replaced() {
        let text = fallback_text(47831, 30159);
        assert_eq!(parse_fallback(&text, 47831), Some(30159));
        assert_eq!(parse_fallback(&text, 47900), None, "a different wanted port (the person changed it) forgets it");
        assert_eq!(parse_fallback(&fallback_text(47831, 47831), 47831), None, "never the wanted port itself");
        assert_eq!(parse_fallback(&fallback_text(47831, 80), 47831), None, "below 1024 is not used");
        assert_eq!(parse_fallback("{ not json", 47831), None);
        assert_eq!(parse_fallback("{\"wanted\":47831}", 47831), None);
        assert_eq!(parse_fallback("\u{feff}{\"wanted\":47831,\"port\":51000}", 47831), Some(51000));
        assert_eq!(parse_fallback(r#"{"wanted":"47831","port":"51000"}"#, 47831), None, "a text is not a number");
    }

    #[test]
    fn query_values_are_decoded_and_bounded() {
        let path = "/shell/notify?title=Strata%20needs%20your%20answer&body=a%20%26%20b%3F+ok";
        assert_eq!(query_value(path, "title").as_deref(), Some("Strata needs your answer"));
        assert_eq!(query_value(path, "body").as_deref(), Some("a & b? ok"));
        assert_eq!(query_value(path, "missing"), None);
        assert_eq!(query_value("/shell/notify", "title"), None);
        assert_eq!(percent_decode("100%"), "100%");
        assert_eq!(percent_decode("%zz%41"), "%zzA");
        assert_eq!(percent_decode("%C3%A4"), "ä");
        assert_eq!(clean_text("a\nb\u{7}c", 10), "abc");
        assert_eq!(clean_text("abcdef", 3), "abc");
    }

    #[test]
    fn every_language_has_tray_labels() {
        for language in LANGUAGES {
            let (open, quit, tip) = tray_labels(language);
            assert!(!open.is_empty() && !quit.is_empty() && !tip.is_empty(), "{language}");
            if language != "en" {
                assert_ne!(open, tray_labels("en").0, "{language}");
            }
        }
        assert_eq!(tray_labels("xx"), tray_labels("en"));
    }
}
