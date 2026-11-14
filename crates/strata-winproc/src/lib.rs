//! Windows file-lock, process-identity, and atomic-replace primitives for Strata.
//! Mirrors `strata_control/winproc.py` and `strata_control/atomic.py`.

pub mod atomic {
    use std::fs;
    use std::io::{self, ErrorKind};
    use std::path::Path;
    use std::thread;
    use std::time::{Duration, Instant};

    pub const PATIENCE: Duration = Duration::from_secs(5);
    pub const STEP: Duration = Duration::from_millis(20);
    pub const MAX_STEP: Duration = Duration::from_millis(250);

    pub fn replace(src: &Path, dst: &Path) -> io::Result<()> {
        replace_with_patience(src, dst, PATIENCE)
    }

    pub fn replace_with_patience(src: &Path, dst: &Path, patience: Duration) -> io::Result<()> {
        let deadline = Instant::now() + patience;
        let mut step = STEP;
        loop {
            match fs::rename(src, dst) {
                Ok(()) => return Ok(()),
                Err(err)
                    if cfg!(windows)
                        && err.kind() == ErrorKind::PermissionDenied
                        && Instant::now() < deadline =>
                {
                    thread::sleep(step);
                    step = Duration::from_secs_f64((step.as_secs_f64() * 1.5).min(MAX_STEP.as_secs_f64()));
                }
                Err(err) => return Err(err),
            }
        }
    }

    pub fn write_text(path: &Path, text: &str) -> io::Result<()> {
        let tmp_name = match path.file_name() {
            Some(name) => {
                let mut s = name.to_os_string();
                s.push(".tmp");
                s
            }
            None => return Err(io::Error::new(ErrorKind::InvalidInput, "missing file name")),
        };
        let tmp = path.with_file_name(tmp_name);
        fs::write(&tmp, text.as_bytes())?;
        replace(&tmp, path)
    }
}

pub mod winproc {
    use std::fs::File;
    use std::io;

    pub const LOCK_OFFSET: u64 = 1 << 16;

    #[cfg(windows)]
    mod imp {
        use super::*;
        use std::ffi::c_void;
        use std::os::windows::io::AsRawHandle;

        type Handle = *mut c_void;
        type Bool = i32;
        type Dword = u32;

        #[repr(C)]
        struct Overlapped {
            internal: usize,
            internal_high: usize,
            offset: Dword,
            offset_high: Dword,
            h_event: Handle,
        }

        #[repr(C)]
        #[derive(Default)]
        struct FileTime {
            low: Dword,
            high: Dword,
        }

        const LOCKFILE_FAIL_IMMEDIATELY: Dword = 0x1;
        const LOCKFILE_EXCLUSIVE_LOCK: Dword = 0x2;
        const ERROR_LOCK_VIOLATION: Dword = 33;
        const ERROR_IO_PENDING: Dword = 997;
        const ERROR_ACCESS_DENIED: Dword = 5;
        const PROCESS_QUERY_LIMITED_INFORMATION: Dword = 0x1000;
        const STILL_ACTIVE: Dword = 259;

        #[link(name = "kernel32")]
        extern "system" {
            fn GetLastError() -> Dword;
            fn LockFileEx(
                h_file: Handle,
                dw_flags: Dword,
                dw_reserved: Dword,
                n_bytes_low: Dword,
                n_bytes_high: Dword,
                lp_overlapped: *mut Overlapped,
            ) -> Bool;
            fn UnlockFileEx(
                h_file: Handle,
                dw_reserved: Dword,
                n_bytes_low: Dword,
                n_bytes_high: Dword,
                lp_overlapped: *mut Overlapped,
            ) -> Bool;
            fn OpenProcess(dw_desired_access: Dword, b_inherit_handle: Bool, dw_process_id: Dword)
                -> Handle;
            fn CloseHandle(h_object: Handle) -> Bool;
            fn GetExitCodeProcess(h_process: Handle, lp_exit_code: *mut Dword) -> Bool;
            fn GetProcessTimes(
                h_process: Handle,
                lp_creation_time: *mut FileTime,
                lp_exit_time: *mut FileTime,
                lp_kernel_time: *mut FileTime,
                lp_user_time: *mut FileTime,
            ) -> Bool;
            fn QueryFullProcessImageNameW(
                h_process: Handle,
                dw_flags: Dword,
                lp_exe_name: *mut u16,
                lpdw_size: *mut Dword,
            ) -> Bool;
        }

        fn overlapped() -> Overlapped {
            Overlapped {
                internal: 0,
                internal_high: 0,
                offset: (LOCK_OFFSET & 0xFFFF_FFFF) as Dword,
                offset_high: (LOCK_OFFSET >> 32) as Dword,
                h_event: std::ptr::null_mut(),
            }
        }

        pub fn try_lock(file: &File, exclusive: bool) -> io::Result<bool> {
            let handle = file.as_raw_handle() as Handle;
            let flags = LOCKFILE_FAIL_IMMEDIATELY | if exclusive { LOCKFILE_EXCLUSIVE_LOCK } else { 0 };
            let mut ov = overlapped();
            let ok = unsafe { LockFileEx(handle, flags, 0, 1, 0, &mut ov) };
            if ok != 0 {
                return Ok(true);
            }
            let code = unsafe { GetLastError() };
            if code == ERROR_LOCK_VIOLATION || code == ERROR_IO_PENDING {
                return Ok(false);
            }
            Err(io::Error::from_raw_os_error(code as i32))
        }

        pub fn unlock(file: &File) -> io::Result<()> {
            let handle = file.as_raw_handle() as Handle;
            let mut ov = overlapped();
            let ok = unsafe { UnlockFileEx(handle, 0, 1, 0, &mut ov) };
            if ok != 0 {
                Ok(())
            } else {
                Err(io::Error::last_os_error())
            }
        }

        pub fn process_started(pid: u32) -> Option<u64> {
            if pid == 0 {
                return None;
            }
            let handle = unsafe { OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, 0, pid) };
            if handle.is_null() {
                return None;
            }
            let mut code: Dword = 0;
            let got_code = unsafe { GetExitCodeProcess(handle, &mut code) };
            if got_code == 0 || code != STILL_ACTIVE {
                unsafe { CloseHandle(handle) };
                return None;
            }
            let (mut created, mut exited, mut kernel, mut user) = (
                FileTime::default(),
                FileTime::default(),
                FileTime::default(),
                FileTime::default(),
            );
            let ok = unsafe {
                GetProcessTimes(handle, &mut created, &mut exited, &mut kernel, &mut user)
            };
            unsafe { CloseHandle(handle) };
            if ok == 0 {
                return None;
            }
            Some(((created.high as u64) << 32) | (created.low as u64))
        }

        pub fn process_image(pid: u32) -> Option<String> {
            if pid == 0 {
                return None;
            }
            let handle = unsafe { OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, 0, pid) };
            if handle.is_null() {
                return None;
            }
            let mut buf = vec![0u16; 32768];
            let mut size = buf.len() as Dword;
            let ok = unsafe { QueryFullProcessImageNameW(handle, 0, buf.as_mut_ptr(), &mut size) };
            unsafe { CloseHandle(handle) };
            if ok == 0 {
                return None;
            }
            Some(String::from_utf16_lossy(&buf[..size as usize]))
        }

        pub fn pid_alive(pid: u32, started: Option<u64>) -> io::Result<bool> {
            if pid == 0 {
                return Ok(false);
            }
            if let Some(now) = process_started(pid) {
                return Ok(started.map_or(true, |s| s == now));
            }
            let handle = unsafe { OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, 0, pid) };
            if !handle.is_null() {
                unsafe { CloseHandle(handle) };
                return Ok(false);
            }
            let err = unsafe { GetLastError() };
            Ok(err == ERROR_ACCESS_DENIED)
        }
    }

    #[cfg(not(windows))]
    mod imp {
        use super::*;

        pub fn try_lock(_file: &File, _exclusive: bool) -> io::Result<bool> {
            Err(io::Error::new(io::ErrorKind::Unsupported, "Windows only"))
        }

        pub fn unlock(_file: &File) -> io::Result<()> {
            Err(io::Error::new(io::ErrorKind::Unsupported, "Windows only"))
        }

        pub fn process_started(_pid: u32) -> Option<u64> {
            None
        }

        pub fn process_image(_pid: u32) -> Option<String> {
            None
        }

        pub fn pid_alive(_pid: u32, _started: Option<u64>) -> io::Result<bool> {
            Err(io::Error::new(io::ErrorKind::Unsupported, "Windows only"))
        }
    }

    pub use imp::{pid_alive, process_image, process_started, try_lock, unlock};
}

pub mod ffi {
    use super::*;
    use std::path::PathBuf;
    use std::slice;
    use std::time::Duration;

    /// # Safety
    /// `raw_handle` must be a valid open Windows file handle that remains valid for the call.
    #[cfg(windows)]
    #[no_mangle]
    pub unsafe extern "C" fn strata_try_lock(raw_handle: isize, exclusive: i32) -> i32 {
        use std::fs::File;
        use std::os::windows::io::{FromRawHandle, IntoRawHandle, RawHandle};
        let file = File::from_raw_handle(raw_handle as RawHandle);
        let res = winproc::try_lock(&file, exclusive != 0);
        let _ = file.into_raw_handle();
        match res {
            Ok(true) => 1,
            Ok(false) => 0,
            Err(_) => -1,
        }
    }

    /// # Safety
    /// `raw_handle` must be a valid open Windows file handle that remains valid for the call.
    #[cfg(windows)]
    #[no_mangle]
    pub unsafe extern "C" fn strata_unlock(raw_handle: isize) -> i32 {
        use std::fs::File;
        use std::os::windows::io::{FromRawHandle, IntoRawHandle, RawHandle};
        let file = File::from_raw_handle(raw_handle as RawHandle);
        let res = winproc::unlock(&file);
        let _ = file.into_raw_handle();
        if res.is_ok() { 0 } else { -1 }
    }

    #[no_mangle]
    pub extern "C" fn strata_process_started(pid: u32) -> u64 {
        winproc::process_started(pid).unwrap_or(0)
    }

    #[no_mangle]
    pub extern "C" fn strata_pid_alive(pid: u32, has_started: i32, started: u64) -> i32 {
        let s = if has_started != 0 { Some(started) } else { None };
        match winproc::pid_alive(pid, s) {
            Ok(true) => 1,
            Ok(false) => 0,
            Err(_) => -1,
        }
    }

    /// # Safety
    /// `buf` must be either null or writable for at least `cap` UTF-16 code units.
    #[no_mangle]
    pub unsafe extern "C" fn strata_process_image(pid: u32, buf: *mut u16, cap: u32) -> i32 {
        let Some(img) = winproc::process_image(pid) else {
            return -1;
        };
        let utf16: Vec<u16> = img.encode_utf16().collect();
        if buf.is_null() || (cap as usize) < utf16.len() {
            return -1;
        }
        std::ptr::copy_nonoverlapping(utf16.as_ptr(), buf, utf16.len());
        utf16.len() as i32
    }

    /// # Safety
    /// `src_ptr` and `dst_ptr` must be valid for reads of `src_len` and `dst_len` bytes respectively.
    #[no_mangle]
    pub unsafe extern "C" fn strata_atomic_replace(
        src_ptr: *const u8,
        src_len: usize,
        dst_ptr: *const u8,
        dst_len: usize,
        patience_ms: u64,
    ) -> i32 {
        if src_ptr.is_null() || dst_ptr.is_null() {
            return -1;
        }
        let Ok(src) = std::str::from_utf8(slice::from_raw_parts(src_ptr, src_len)) else {
            return -1;
        };
        let Ok(dst) = std::str::from_utf8(slice::from_raw_parts(dst_ptr, dst_len)) else {
            return -1;
        };
        match atomic::replace_with_patience(
            &PathBuf::from(src),
            &PathBuf::from(dst),
            Duration::from_millis(patience_ms),
        ) {
            Ok(()) => 0,
            Err(e) if e.kind() == std::io::ErrorKind::PermissionDenied => 1,
            Err(_) => -1,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs::{self, OpenOptions};
    use std::path::PathBuf;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn temp_path(name: &str) -> PathBuf {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        std::env::temp_dir().join(format!("strata_winproc_test_{name}_{nanos}"))
    }

    #[test]
    fn atomic_write_text_replaces_cleanly() {
        let p = temp_path("atomic.json");
        atomic::write_text(&p, "{\"step\":1}").unwrap();
        assert_eq!(fs::read_to_string(&p).unwrap(), "{\"step\":1}");
        atomic::write_text(&p, "{\"step\":2}").unwrap();
        assert_eq!(fs::read_to_string(&p).unwrap(), "{\"step\":2}");
        let _ = fs::remove_file(&p);
    }

    #[cfg(windows)]
    #[test]
    fn winproc_lock_and_self_identity() {
        let p = temp_path("lock.txt");
        let f1 = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(false)
            .open(&p)
            .unwrap();
        let f2 = OpenOptions::new().read(true).write(true).open(&p).unwrap();

        assert!(winproc::try_lock(&f1, true).unwrap());
        assert!(!winproc::try_lock(&f2, true).unwrap());
        assert!(!winproc::try_lock(&f2, false).unwrap());
        winproc::unlock(&f1).unwrap();
        assert!(winproc::try_lock(&f2, false).unwrap());
        winproc::unlock(&f2).unwrap();
        drop(f1);
        drop(f2);
        let _ = fs::remove_file(&p);

        let pid = std::process::id();
        let started = winproc::process_started(pid).expect("current process must have start time");
        assert!(winproc::pid_alive(pid, Some(started)).unwrap());
        assert!(!winproc::pid_alive(pid, Some(started + 1)).unwrap());
        assert!(winproc::process_image(pid).is_some());
    }
}
