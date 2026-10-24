//! Opt-in Windows window lifetime. No engine, UI or updater ownership changes.
use std::sync::Arc;
use tauri::Manager;

#[cfg(windows)]
mod owner {
    use std::{ffi::c_void, path::PathBuf};
    type Handle = *mut c_void;
    #[link(name = "kernel32")]
    extern "system" {
        fn OpenProcess(access: u32, inherit: i32, pid: u32) -> Handle;
        fn QueryFullProcessImageNameW(process: Handle, flags: u32, image: *mut u16, size: *mut u32) -> i32;
        fn WaitForSingleObject(handle: Handle, timeout: u32) -> u32;
        fn CloseHandle(handle: Handle) -> i32;
    }
    pub struct Owner(Handle);
    unsafe impl Send for Owner {}
    unsafe impl Sync for Owner {}
    impl Owner {
        pub fn open(pid: u32, image: &str) -> Option<Self> {
            if pid == 0 || !PathBuf::from(image).is_absolute() { return None; }
            let handle = unsafe { OpenProcess(0x0010_0000 | 0x1000, 0, pid) };
            if handle.is_null() { return None; }
            let owner = Self(handle);
            let mut buffer = vec![0u16; 32768];
            let mut size = buffer.len() as u32;
            if unsafe { QueryFullProcessImageNameW(handle, 0, buffer.as_mut_ptr(), &mut size) } == 0 {
                return None;
            }
            let actual = String::from_utf16_lossy(&buffer[..size as usize]);
            if !actual.eq_ignore_ascii_case(image) || !owner.alive() { return None; }
            Some(owner)
        }
        pub fn alive(&self) -> bool { unsafe { WaitForSingleObject(self.0, 0) == 258 } }
        pub fn wait(&self) { unsafe { WaitForSingleObject(self.0, u32::MAX); } }
    }
    impl Drop for Owner { fn drop(&mut self) { unsafe { CloseHandle(self.0); } } }
}

pub struct Policy {
    #[cfg(windows)]
    owner: Option<Arc<owner::Owner>>,
}

impl Policy {
    pub fn from_environment() -> Self {
        #[cfg(windows)]
        {
            let enabled = std::env::var("STRATA_CLOSE_TO_TRAY").ok().as_deref() == Some("1");
            let pid = std::env::var("STRATA_TRAY_PID").ok().and_then(|v| v.parse().ok());
            let image = std::env::var("STRATA_TRAY_IMAGE").ok();
            let owner = if enabled {
                pid.zip(image).and_then(|(pid, image)| owner::Owner::open(pid, &image)).map(Arc::new)
            } else { None };
            Self { owner }
        }
        #[cfg(not(windows))]
        Self {}
    }

    pub fn can_hide(&self) -> bool {
        #[cfg(windows)]
        { self.owner.as_ref().map(|owner| owner.alive()).unwrap_or(false) }
        #[cfg(not(windows))]
        { false }
    }

    pub fn watch_owner(&self, app: tauri::AppHandle) {
        #[cfg(windows)]
        if let Some(owner) = self.owner.clone() {
            // Hold this exact process handle: PID reuse cannot change the owner.
            std::thread::spawn(move || { owner.wait(); app.exit(0); });
        }
    }
}

pub fn quit_requested(args: &[String]) -> bool {
    args.iter().any(|arg| arg == "--quit")
}

pub fn second_launch(app: &tauri::AppHandle, args: &[String]) -> bool {
    if quit_requested(args) {
        app.exit(0); // Explicit quit uses the original clean Host shutdown.
        return true;
    }
    false // Keep the existing single-instance restore path in main.rs.
}

pub fn close(window: &tauri::Window, event: &tauri::WindowEvent) -> bool {
    if let tauri::WindowEvent::CloseRequested { api, .. } = event {
        if window.label() == "main" && window.app_handle().state::<Policy>().can_hide() {
            // Only prevent destruction after hide succeeds. Without a live Tray,
            // normal close remains available and cannot strand an invisible App.
            if window.hide().is_ok() { api.prevent_close(); return true; }
        }
    }
    false
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn explicit_quit_does_not_match_other_arguments() {
        let args = |values: &[&str]| values.iter().map(|value| value.to_string()).collect::<Vec<_>>();
        assert!(quit_requested(&args(&["strata-desktop", "--quit"])));
        assert!(!quit_requested(&args(&["strata-desktop", "--quit-later"])));
        assert!(!quit_requested(&args(&["strata-desktop", "a--quit.exe"])));
        assert!(!quit_requested(&[]));
    }

    #[cfg(windows)]
    #[test]
    fn owner_requires_the_exact_live_executable() {
        let image = crate::plain_path(std::env::current_exe().unwrap()).to_string_lossy().into_owned();
        let pid = std::process::id();
        let owner = owner::Owner::open(pid, &image).expect("the current test process");
        assert!(owner.alive());
        assert!(owner::Owner::open(pid, &format!("{image}.other")).is_none());
        assert!(owner::Owner::open(pid, "relative.exe").is_none());
        assert!(owner::Owner::open(0, &image).is_none());
    }

    #[cfg(windows)]
    #[test]
    fn a_held_owner_stops_enabling_hide_when_that_process_exits() {
        use std::os::windows::process::CommandExt;
        use std::process::{Command, Stdio};
        let image = std::path::PathBuf::from(std::env::var_os("SystemRoot").unwrap())
            .join("System32").join("ping.exe");
        let mut child = Command::new(&image).args(["-n", "60", "127.0.0.1"])
            .creation_flags(0x0800_0000).stdin(Stdio::null()).stdout(Stdio::null()).stderr(Stdio::null())
            .spawn().expect("a disposable owner with no console window");
        let opened = owner::Owner::open(child.id(), &image.to_string_lossy());
        // Always end the disposable process, including if the identity assertion fails.
        let was_alive = opened.as_ref().map(|owner| owner.alive()).unwrap_or(false);
        let _ = child.kill();
        child.wait().unwrap();
        let owner = opened.expect("the exact disposable process");
        owner.wait();
        let policy = Policy { owner: Some(Arc::new(owner)) };
        assert!(was_alive);
        assert!(!policy.can_hide(), "a dead owner cannot strand an invisible App");
    }

    #[test]
    fn missing_or_unverified_tray_never_enables_hide() {
        std::env::remove_var("STRATA_CLOSE_TO_TRAY");
        assert!(!Policy::from_environment().can_hide());
        std::env::set_var("STRATA_CLOSE_TO_TRAY", "1");
        std::env::set_var("STRATA_TRAY_PID", "0");
        std::env::set_var("STRATA_TRAY_IMAGE", "relative.exe");
        assert!(!Policy::from_environment().can_hide());
        std::env::remove_var("STRATA_CLOSE_TO_TRAY");
        std::env::remove_var("STRATA_TRAY_PID");
        std::env::remove_var("STRATA_TRAY_IMAGE");
    }
}
