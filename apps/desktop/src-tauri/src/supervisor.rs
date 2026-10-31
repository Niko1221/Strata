//! Linux: the process that owns the host, and what it does when the app window cannot run.
//!
//! WebKitGTK can crash the process that shows the window (a null pointer deep inside libwebkit2gtk was seen on Ubuntu 26.04 with
//! NVIDIA on Wayland, and on Ubuntu 24.04 under Xvfb). When the window and the host lived in one process, that crash took the host
//! and the engine down with it. So on Linux the work is split in two processes of this one executable:
//!
//!   shell    (this module; no GTK, no WebKit)  starts the host, writes the session file, starts the window process, and watches both
//!   window   (`--window-child`; Tauri + WebKitGTK) shows the page, registers the app updater with the host, and nothing else
//!
//! If the window process dies for any reason except the person closing the window (a signal such as SIGSEGV, an abort, a failed
//! start, a missing library), the shell keeps the host running, tells the host to open the page in the default browser and to
//! show one notification, and prints the same sentence on the terminal. This is "browser mode"; `--browser` or STRATA_UI=browser
//! starts in it on purpose. In browser mode closing the tab does not stop the engine: the page has "Quit Strata", and `--quit`
//! ends a running Strata from the command line. The host listens on 127.0.0.1 only (the shell sets STRATA_BIND itself).
//! `--lan` or STRATA_LAN=1 adds the second, key-guarded listener that answers for other devices on your network
//! (`--access lan` in the Terminal): the shell hands the host STRATA_SHARE_ON_START=1, and the app's own page stays on
//! this computer only.
//!
//! The session file (`~/.strata/app-session.json`, mode 0600: process id, port, the secret only this user can read, the mode) is how
//! a second start finds the first one: it focuses the window (window mode) or opens another tab (browser mode) instead of
//! starting a second host.

use std::io::Write;
use std::path::PathBuf;
use std::process::{Child, Command, ExitStatus, Stdio};
use std::time::{Duration, Instant};

use crate::desktoplog::{self, say, Log, Ring};

pub const BROWSER_NOTICE: &str = "Strata opened in your browser because the app window couldn't start on this computer.";

/// What the command line asked for.
#[derive(Debug, Default, PartialEq, Eq)]
pub struct Options {
    /// `--browser` or STRATA_UI=browser: no app window, the page opens in the default browser.
    pub browser: bool,
    /// `--quit`: end a running Strata and leave.
    pub quit: bool,
    /// `--window-child`: the window process (started by the shell, not by a person).
    pub window_child: bool,
    /// `--save-logs`: collect the logs into one file in Downloads and leave.
    pub save_logs: bool,
    /// `--lan` or STRATA_LAN=1: share the model with other devices on your network as soon as Strata starts.
    pub lan: bool,
    pub help: bool,
}

pub fn parse_args(args: &[String], ui_env: Option<&str>, lan_env: Option<&str>) -> Options {
    let mut o = Options { browser: matches!(ui_env.map(|v| v.trim().to_ascii_lowercase()).as_deref(), Some("browser")), lan: lan_on(lan_env), ..Options::default() };
    for a in args.iter().skip(1) {
        match a.as_str() {
            "--browser" => o.browser = true,
            "--lan" => o.lan = true,
            "--quit" => o.quit = true,
            "--window-child" => o.window_child = true,
            "--save-logs" => o.save_logs = true,
            "--help" | "-h" => o.help = true,
            _ => {}
        }
    }
    o
}

/// The environment switch: STRATA_LAN=1 (also "on", "true", "yes", "lan"), the way STRATA_UI=browser is chosen.
fn lan_on(v: Option<&str>) -> bool {
    matches!(v.map(|s| s.trim().to_ascii_lowercase()).as_deref(), Some("1") | Some("on") | Some("true") | Some("yes") | Some("lan"))
}

pub const HELP: &str = "Strata\n\
  (no option)  open the Strata app window\n\
  --browser    open Strata in your default browser instead of the app window (same as STRATA_UI=browser)\n\
  --lan        serve the model to other devices on your local network as soon as Strata opens (same as STRATA_LAN=1)\n\
  --quit       stop a running Strata (its engine too) and leave\n\
  --save-logs  put the logs (including the log of an app window crash) into one file in your Downloads folder, to send to us\n\
Closing the browser tab does not stop Strata in browser mode: use Quit Strata in the menu, or --quit.\n\
Strata's own pages and controls stay on this computer (--lan shares only the model, on its own port).\n";

/// The command a person types to run this app again, for the messages ("run: Strata.AppImage --quit").
pub fn command_name() -> String {
    if let Ok(image) = std::env::var("APPIMAGE") {
        if let Some(name) = std::path::Path::new(&image).file_name() {
            return format!("./{}", name.to_string_lossy());
        }
    }
    std::env::current_exe().ok().and_then(|p| p.file_name().map(|n| n.to_string_lossy().into_owned())).unwrap_or_else(|| "strata-desktop".into())
}

pub fn browser_chosen_line(port: u16) -> String {
    format!(
        "Strata is open in your browser at http://127.0.0.1:{port}/. Closing the tab does not stop Strata: choose Quit Strata in the menu, or run {} --quit.",
        command_name()
    )
}

pub fn open_by_hand_line(port: u16) -> String {
    format!("No browser could be opened. Open http://127.0.0.1:{port}/ in your browser.")
}

// ------------------------------------------------------------------------------------------------ the session file

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Session {
    pub pid: u32,
    pub port: u16,
    pub token: String,
    pub mode: String,
}

fn session_path() -> PathBuf {
    if let Some(p) = std::env::var_os("STRATA_APP_SESSION_FILE") {
        return PathBuf::from(p);
    }
    let home = std::env::var_os("HOME").map(PathBuf::from).unwrap_or_else(std::env::temp_dir);
    home.join(".strata").join("app-session.json")
}

pub fn write_session(s: &Session) {
    let path = session_path();
    if let Some(dir) = path.parent() {
        let _ = std::fs::create_dir_all(dir);
    }
    let text = serde_json::json!({ "pid": s.pid, "port": s.port, "token": s.token, "mode": s.mode }).to_string();
    let mut options = std::fs::OpenOptions::new();
    options.write(true).create(true).truncate(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options.mode(0o600);
    }
    match options.open(&path) {
        Ok(mut f) => {
            let _ = f.write_all(text.as_bytes());
        }
        Err(e) => eprintln!("[strata] could not write {}: {e}", path.display()),
    }
}

pub fn read_session() -> Option<Session> {
    let text = std::fs::read_to_string(session_path()).ok()?;
    let v: serde_json::Value = serde_json::from_str(&text).ok()?;
    Some(Session {
        pid: v["pid"].as_u64()? as u32,
        port: v["port"].as_u64()? as u16,
        token: v["token"].as_str()?.to_string(),
        mode: v["mode"].as_str().unwrap_or("window").to_string(),
    })
}

pub fn remove_session(only_if_pid: u32) {
    if read_session().map(|s| s.pid == only_if_pid).unwrap_or(false) {
        let _ = std::fs::remove_file(session_path());
    }
}

fn pid_alive(pid: u32) -> bool {
    #[cfg(target_os = "linux")]
    {
        std::path::Path::new(&format!("/proc/{pid}")).exists()
    }
    #[cfg(not(target_os = "linux"))]
    {
        let _ = pid;
        true
    }
}

/// A Strata that is really running: its shell process exists and its host answers.
pub fn live_session() -> Option<Session> {
    let s = read_session()?;
    if s.pid != std::process::id() && pid_alive(s.pid) && crate::ureq_get_ok(&format!("http://127.0.0.1:{}/api/health", s.port)).unwrap_or(false) {
        Some(s)
    } else {
        None
    }
}

// ------------------------------------------------------------------------------------------------ talking to the host

/// POST to the host as the shell (the secret in the header). Returns the response body.
fn host_post(port: u16, token: &str, path: &str, body: &serde_json::Value) -> Result<String, String> {
    let (status, text) = crate::http_request_body("POST", &format!("http://127.0.0.1:{port}{path}"), &[("x-strata-shutdown", token), ("content-type", "application/json")], Some(&body.to_string()))?;
    if status.contains(" 200") || status.contains(" 202") {
        Ok(text)
    } else {
        Err(format!("{status}: {text}"))
    }
}

/// Did the host manage to open the page (the "opened" flag in its answer)?
fn opened(answer: &str) -> bool {
    let body = answer.rsplit("\r\n\r\n").next().unwrap_or(answer);
    serde_json::from_str::<serde_json::Value>(body.trim()).ok().and_then(|v| v["opened"].as_bool()).unwrap_or(false)
}

/// Tell the host: everyone uses a browser tab now (and open it, and show `notice`). Prints what happened.
pub fn switch_host_to_browser(port: u16, token: &str, reason: &str, notice: &str) {
    let body = serde_json::json!({ "mode": "browser", "reason": reason, "notice": notice, "open": true });
    match host_post(port, token, "/api/control/ui", &body) {
        Ok(answer) => {
            if !opened(&answer) {
                eprintln!("[strata] {}", open_by_hand_line(port));
            }
        }
        Err(e) => eprintln!("[strata] could not tell the host to open your browser ({e}). Open http://127.0.0.1:{port}/ in your browser."),
    }
}

/// Open the page in the default browser without changing the host's mode (a second start while Strata runs).
fn open_page_only(port: u16, token: &str) {
    match host_post(port, token, "/api/control/ui", &serde_json::json!({ "open": true })) {
        Ok(answer) if opened(&answer) => {}
        _ => eprintln!("[strata] {}", open_by_hand_line(port)),
    }
}

pub fn describe_exit(status: &ExitStatus) -> String {
    #[cfg(unix)]
    {
        use std::os::unix::process::ExitStatusExt;
        if let Some(sig) = status.signal() {
            let name = match sig {
                6 => " (abort)",
                11 => " (segmentation fault)",
                7 => " (bus error)",
                9 => " (killed)",
                15 => " (terminated)",
                _ => "",
            };
            return format!("signal {sig}{name}");
        }
    }
    match status.code() {
        Some(127) => "exit code 127: a program file it needs is missing".into(),
        Some(126) => "exit code 126: a program file it needs could not be run".into(),
        Some(c) => format!("exit code {c}"),
        None => "ended without a code".into(),
    }
}

// ------------------------------------------------------------------------------------------------ the shell

fn spawn_window(port: u16, token: &str) -> std::io::Result<Child> {
    let exe = std::env::current_exe()?;
    Command::new(exe)
        .arg("--window-child")
        .env("STRATA_WINDOW_PORT", port.to_string())
        .env("STRATA_SHUTDOWN_TOKEN", token)
        .stdin(Stdio::null())
        .stdout(Stdio::inherit())
        .stderr(Stdio::inherit())
        .spawn()
}

/// `--quit`: end a running Strata (the window, the host, the engines it started) and wait for it.
pub fn quit_running() -> i32 {
    let Some(s) = live_session() else {
        println!("Strata is not running.");
        let _ = std::fs::remove_file(session_path());
        return 0;
    };
    match host_post(s.port, &s.token, "/api/control/shutdown", &serde_json::json!({})) {
        Ok(_) => {}
        Err(e) => {
            eprintln!("[strata] could not ask Strata to stop ({e}).");
            return 1;
        }
    }
    let deadline = Instant::now() + Duration::from_secs(crate::HOST_STOP_SECONDS + 10);
    while Instant::now() < deadline {
        if !pid_alive(s.pid) {
            println!("Strata has stopped.");
            return 0;
        }
        std::thread::sleep(Duration::from_millis(200));
    }
    eprintln!("[strata] Strata is still stopping its engine. Try again in a moment.");
    1
}

/// What the host says about its network share (`sharing` in `/api/serving`'s answer): the address and, when there is no
/// API key, the warning. Both are empty when the host did not answer with that object.
#[derive(Debug, Default, PartialEq, Eq)]
pub struct Share {
    pub url: Option<String>,
    pub warning: Option<String>,
}

fn share_of(answer: &str) -> Share {
    let body = answer.rsplit("\r\n\r\n").next().unwrap_or(answer);
    let Ok(value) = serde_json::from_str::<serde_json::Value>(body.trim()) else { return Share::default() };
    Share { url: value["sharing"]["url"].as_str().map(String::from), warning: value["sharing"]["warning"].as_str().map(String::from) }
}

/// The line the shell prints for a share, in the person's words and with the address to type in the other device.
pub fn share_line(s: &Share) -> String {
    match &s.url {
        Some(url) => match &s.warning {
            Some(warning) => format!("Strata shares your model on your network at {url}. {warning}"),
            None => format!("Strata shares your model on your network at {url} (an API key protects it)."),
        },
        // the host answered but did not say where: say what happened without making an address up
        None => "Strata shares your model on your network; Settings > Serving shows the address.".into(),
    }
}

/// `--lan` on a second start, while Strata already runs: ask that host for the network share (the same call the page makes,
/// `/api/serving/share`). Nothing is started here; the host owns the listener.
fn share_with_network(port: u16, token: &str) {
    match host_post(port, token, "/api/serving/share", &serde_json::json!({ "enabled": true })) {
        Ok(answer) => println!("{}", share_line(&share_of(&answer))),
        Err(e) => eprintln!("[strata] could not ask Strata to share on your network ({e}). Open Strata and use Settings > Serving."),
    }
}

/// A second start while Strata already runs: focus the window, or open another tab. Starts no second host.
fn join_running(s: &Session, opts: &Options) -> i32 {
    if opts.lan {
        share_with_network(s.port, &s.token);
    }
    if opts.browser || s.mode == "browser" {
        println!("Strata is already running. Opening it in your browser.");
        open_page_only(s.port, &s.token);
        return 0;
    }
    // window mode: the window process's single-instance check brings the open window forward and exits at once
    match spawn_window(s.port, &s.token).and_then(|mut c| c.wait()) {
        Ok(status) if status.success() => 0,
        Ok(status) => {
            eprintln!("[strata] The app window stopped ({}). Opening Strata in your browser.", describe_exit(&status));
            open_page_only(s.port, &s.token);
            0
        }
        Err(e) => {
            eprintln!("[strata] could not start the app window ({e}). Opening Strata in your browser.");
            open_page_only(s.port, &s.token);
            0
        }
    }
}

/// Start a child with its output copied to the terminal, the log and a ring (for the crash log).
fn spawn_logged(cmd: &mut Command, who: &'static str, ring: &Ring, log: &Log) -> std::io::Result<Child> {
    cmd.stdout(Stdio::piped()).stderr(Stdio::piped());
    let mut child = cmd.spawn()?;
    if let Some(out) = child.stdout.take() {
        desktoplog::tee(out, who, false, ring.clone(), log.clone());
    }
    if let Some(err) = child.stderr.take() {
        desktoplog::tee(err, who, true, ring.clone(), log.clone());
    }
    Ok(child)
}

/// The window process died without being closed: say so in plain words, write the crash log, move everyone to the browser.
fn window_failed(why: &str, window_pid: u32, window_ring: &Ring, host_ring: &Ring, log: &Log, port: u16, token: &str) {
    say(log, &format!("The app window crashed ({why})."));
    // let the copy threads read the last lines the dead process wrote
    std::thread::sleep(Duration::from_millis(300));
    match desktoplog::write_crash_log(why, window_pid, window_ring, host_ring, log) {
        Some(path) => say(log, &format!("The crash log is {}. To collect the logs for us: {} --save-logs", path.display(), command_name())),
        None => say(log, &format!("The crash log could not be written. To collect the logs for us: {} --save-logs", command_name())),
    }
    say(log, BROWSER_NOTICE);
    switch_host_to_browser(port, token, "window-failed", BROWSER_NOTICE);
}

/// The shell. Returns the process exit code.
pub fn run(opts: &Options) -> i32 {
    if opts.help {
        print!("{HELP}");
        return 0;
    }
    if opts.save_logs {
        return match desktoplog::save_logs(&desktoplog::downloads_dir()) {
            Ok(path) => {
                println!("Saved to {}. Please send this file to us.", path.display());
                0
            }
            Err(e) => {
                eprintln!("[strata] could not save the logs: {e}");
                1
            }
        };
    }
    if opts.quit {
        return quit_running();
    }
    if let Some(s) = live_session() {
        return join_running(&s, opts);
    }

    let log = Log::open();
    let (host_ring, window_ring) = (Ring::default(), Ring::default());
    let exe = std::env::current_exe().unwrap_or_default();
    let res = crate::packaged_resource_dir(&exe).unwrap_or_else(|| PathBuf::from(env!("CARGO_MANIFEST_DIR")));
    let port = match crate::pick_stable_port() {
        Ok(p) => p,
        Err(e) => {
            say(&log, &e);
            return 1;
        }
    };
    let token = crate::random_token();
    let mode = if opts.browser { "browser" } else { "window" };
    let mut cmd = match crate::host_command(&res, port, &token) {
        Ok(c) => c,
        Err(e) => {
            say(&log, &e);
            return 1;
        }
    };
    cmd.env("STRATA_UI_MODE", mode).env("STRATA_APP_VERSION", env!("CARGO_PKG_VERSION"));
    if opts.browser {
        cmd.env("STRATA_UI_REASON", "chosen");
    }
    if opts.lan {
        // the host brings its own network listener up with the page (index.mjs, local-share.mjs)
        cmd.env("STRATA_SHARE_ON_START", "1");
    }
    let mut host = match spawn_logged(&mut cmd, "host", &host_ring, &log) {
        Ok(c) => c,
        Err(e) => {
            say(&log, &format!("could not start the App Host: {e}"));
            return 1;
        }
    };
    let health = crate::wait_for_health(port, Duration::from_secs(60), || match host.try_wait() {
        Ok(Some(status)) => Some(format!("exit status {status}")),
        _ => None,
    });
    if let Err(e) = health {
        say(&log, &e);
        let _ = host.kill();
        let _ = host.wait();
        return 1;
    }
    let me = std::process::id();
    write_session(&Session { pid: me, port, token: token.clone(), mode: mode.into() });

    let mut window: Option<Child> = None;
    if opts.browser {
        say(&log, &browser_chosen_line(port));
        switch_host_to_browser(port, &token, "chosen", "");
    } else {
        let started = std::env::current_exe().and_then(|exe| {
            let mut c = Command::new(exe);
            c.arg("--window-child").env("STRATA_WINDOW_PORT", port.to_string()).env("STRATA_SHUTDOWN_TOKEN", &token).stdin(Stdio::null());
            spawn_logged(&mut c, "window", &window_ring, &log)
        });
        match started {
            Ok(c) => window = Some(c),
            Err(e) => {
                window_failed(&format!("it could not be started: {e}"), 0, &window_ring, &host_ring, &log, port, &token);
                write_session(&Session { pid: me, port, token: token.clone(), mode: "browser".into() });
            }
        }
    }

    let code = loop {
        std::thread::sleep(Duration::from_millis(200));
        // The host ended (Quit Strata, a crash, someone killed it): nothing is left to show.
        if let Ok(Some(status)) = host.try_wait() {
            if let Some(w) = window.as_mut() {
                let _ = w.kill();
                let _ = w.wait();
            }
            if status.success() {
                log.line("shell", "the host stopped (Quit Strata or the window was closed)");
                break 0;
            }
            say(&log, &format!("The App Host stopped ({}). The last lines it printed are in {}.", describe_exit(&status), log.path.as_ref().map(|p| p.display().to_string()).unwrap_or_default()));
            break 1;
        }
        let Some(w) = window.as_mut() else { continue };
        let window_pid = w.id();
        let Ok(Some(status)) = w.try_wait() else { continue };
        window = None;
        if status.success() {
            // the person closed the window: stop the host and the engines this app started, as before
            log.line("shell", "the window was closed: stopping the host");
            crate::stop_host_child(&mut host, Some(port), Some(&token));
            break 0;
        }
        // The window stopped without being closed: keep Strata running and move to the browser.
        window_failed(&describe_exit(&status), window_pid, &window_ring, &host_ring, &log, port, &token);
        write_session(&Session { pid: me, port, token: token.clone(), mode: "browser".into() });
    };
    remove_session(me);
    code
}

#[cfg(test)]
mod tests {
    use super::*;

    fn args(list: &[&str]) -> Vec<String> {
        std::iter::once("strata-desktop").chain(list.iter().copied()).map(String::from).collect()
    }

    #[test]
    fn the_command_line_picks_browser_quit_and_the_window_child() {
        assert_eq!(parse_args(&args(&[]), None, None), Options::default());
        assert!(parse_args(&args(&["--browser"]), None, None).browser);
        assert!(parse_args(&args(&[]), Some("browser"), None).browser);
        assert!(parse_args(&args(&[]), Some(" Browser "), None).browser);
        assert!(!parse_args(&args(&[]), Some("window"), None).browser);
        assert!(parse_args(&args(&["--quit"]), None, None).quit);
        assert!(parse_args(&args(&["--window-child"]), None, None).window_child);
        assert!(parse_args(&args(&["--save-logs"]), None, None).save_logs);
        assert!(parse_args(&args(&["-h"]), None, None).help);
        // anything else (a file the desktop hands over, an AppImage option) is ignored
        assert_eq!(parse_args(&args(&["--appimage-extract-and-run", "x.gguf"]), None, None), Options::default());
    }

    #[test]
    fn lan_comes_from_the_command_line_or_the_environment() {
        assert!(!parse_args(&args(&[]), None, None).lan, "the default: this computer only");
        assert!(parse_args(&args(&["--lan"]), None, None).lan);
        assert!(parse_args(&args(&[]), None, Some("1")).lan);
        assert!(parse_args(&args(&[]), None, Some(" LAN ")).lan);
        assert!(parse_args(&args(&[]), None, Some("yes")).lan);
        assert!(parse_args(&args(&[]), None, Some("true")).lan);
        assert!(!parse_args(&args(&[]), None, Some("0")).lan);
        assert!(!parse_args(&args(&[]), None, Some("browser")).lan);
        // the two together: the page in a browser tab, the model on your network
        let both = parse_args(&args(&["--lan", "--browser"]), None, None);
        assert!(both.lan && both.browser);
        // the help names the option, the environment switch and what stays on this computer
        assert!(HELP.contains("--lan") && HELP.contains("STRATA_LAN=1") && HELP.contains("this computer"));
    }

    #[test]
    fn the_share_line_names_the_address_the_host_answered_and_the_warning() {
        let body = "{\"running\":true,\"sharing\":{\"active\":true,\"url\":\"http://192.168.1.20:47832/v1\",\"warning\":\"No API key is set.\"}}";
        let share = share_of(&format!("HTTP/1.1 200 OK\r\n\r\n{body}"));
        assert_eq!(share.url.as_deref(), Some("http://192.168.1.20:47832/v1"));
        assert_eq!(share_line(&share), "Strata shares your model on your network at http://192.168.1.20:47832/v1. No API key is set.");
        // a protected share says the key protects it, and adds nothing
        let protected = share_of("HTTP/1.1 200 OK\r\n\r\n{\"sharing\":{\"active\":true,\"url\":\"http://10.0.0.9:8792/v1\",\"warning\":null}}");
        assert_eq!(share_line(&protected), "Strata shares your model on your network at http://10.0.0.9:8792/v1 (an API key protects it).");
        // no object in the answer: still a sentence, and no address made up
        assert!(!share_line(&share_of("not json")).contains("http://"));
    }

    #[test]
    fn the_browser_message_is_the_plain_sentence_and_names_the_way_out() {
        assert_eq!(BROWSER_NOTICE, "Strata opened in your browser because the app window couldn't start on this computer.");
        let line = browser_chosen_line(47831);
        assert!(line.contains("http://127.0.0.1:47831/") && line.contains("--quit") && line.contains("Quit Strata"));
        assert!(open_by_hand_line(47831).contains("127.0.0.1:47831"));
    }

    #[test]
    fn a_session_round_trips_and_is_private() {
        let dir = std::env::temp_dir().join(format!("strata-session-test-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::env::set_var("STRATA_APP_SESSION_FILE", dir.join("s.json"));
        assert!(read_session().is_none());
        let s = Session { pid: 4242, port: 47831, token: "ab".repeat(32), mode: "browser".into() };
        write_session(&s);
        assert_eq!(read_session(), Some(s.clone()));
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            assert_eq!(std::fs::metadata(dir.join("s.json")).unwrap().permissions().mode() & 0o777, 0o600);
        }
        // only the owner removes it
        remove_session(1);
        assert!(read_session().is_some());
        remove_session(4242);
        assert!(read_session().is_none());
        std::env::remove_var("STRATA_APP_SESSION_FILE");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn the_hosts_answer_tells_whether_a_browser_opened() {
        assert!(opened("HTTP/1.1 200 OK\r\n\r\n{\"mode\":\"browser\",\"opened\":true}"));
        assert!(!opened("HTTP/1.1 200 OK\r\n\r\n{\"mode\":\"browser\",\"opened\":false}"));
        assert!(!opened("not json"));
    }
}
