#[cfg(windows)]
mod win {
    use std::ffi::c_void;
    use std::ptr;

    const PROCESS_QUERY_LIMITED_INFORMATION: u32 = 0x1000;
    const PROCESS_BASIC_INFORMATION_CLASS: u32 = 0;
    const PROCESS_COMMAND_LINE_INFORMATION_CLASS: u32 = 60;

    #[repr(C)]
    struct FileTime {
        low: u32,
        high: u32,
    }

    #[repr(C)]
    struct UnicodeString {
        length: u16,
        maximum_length: u16,
        buffer: *mut u16,
    }

    #[repr(C)]
    struct ProcessBasicInfo {
        reserved1: usize,
        peb_base_address: usize,
        reserved2: [usize; 2],
        unique_process_id: usize,
        inherited_from_unique_process_id: usize,
    }

    #[link(name = "kernel32")]
    extern "system" {
        fn OpenProcess(dwDesiredAccess: u32, bInheritHandle: i32, dwProcessId: u32) -> *mut c_void;
        fn CloseHandle(hObject: *mut c_void) -> i32;
        fn GetProcessTimes(
            hProcess: *mut c_void,
            lpCreationTime: *mut FileTime,
            lpExitTime: *mut FileTime,
            lpKernelTime: *mut FileTime,
            lpUserTime: *mut FileTime,
        ) -> i32;
        fn GetExitCodeProcess(hProcess: *mut c_void, lpExitCode: *mut u32) -> i32;
        fn QueryFullProcessImageNameW(
            hProcess: *mut c_void,
            dwFlags: u32,
            lpExeName: *mut u16,
            lpdwSize: *mut u32,
        ) -> i32;
        fn LocalFree(hMem: *mut c_void) -> *mut c_void;
    }

    #[link(name = "ntdll")]
    extern "system" {
        fn NtQueryInformationProcess(
            ProcessHandle: *mut c_void,
            ProcessInformationClass: u32,
            ProcessInformation: *mut c_void,
            ProcessInformationLength: u32,
            ReturnLength: *mut u32,
        ) -> i32;
    }

    #[link(name = "shell32")]
    extern "system" {
        fn CommandLineToArgvW(lpCmdLine: *const u16, pNumArgs: *mut i32) -> *mut *mut u16;
    }

    struct Handle(*mut c_void);
    impl Drop for Handle {
        fn drop(&mut self) {
            if !self.0.is_null() {
                // SAFETY: `self.0` is a valid non-null handle from `OpenProcess`.
                unsafe {
                    CloseHandle(self.0);
                }
            }
        }
    }

    fn open_query(pid: u32) -> Option<Handle> {
        // SAFETY: Win32 `OpenProcess` with `PROCESS_QUERY_LIMITED_INFORMATION`.
        let h = unsafe { OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, 0, pid) };
        if h.is_null() {
            None
        } else {
            Some(Handle(h))
        }
    }

    pub fn process_started(pid: u32) -> Option<u64> {
        let h = open_query(pid)?;
        let mut exit_code: u32 = 0;
        // SAFETY: `h.0` is a valid process handle.
        if unsafe { GetExitCodeProcess(h.0, &mut exit_code) } != 0 && exit_code != 259 {
            return None; // STILL_ACTIVE == 259; exited processes still have handles until closed
        }
        let mut creation = FileTime { low: 0, high: 0 };
        let mut exit = FileTime { low: 0, high: 0 };
        let mut kernel = FileTime { low: 0, high: 0 };
        let mut user = FileTime { low: 0, high: 0 };
        // SAFETY: All 4 FILETIME pointers are valid stack locals.
        let ok = unsafe { GetProcessTimes(h.0, &mut creation, &mut exit, &mut kernel, &mut user) };
        if ok == 0 {
            return None;
        }
        Some(((creation.high as u64) << 32) | (creation.low as u64))
    }

    pub fn parent_of(pid: u32) -> Option<u32> {
        let h = open_query(pid)?;
        let mut info = ProcessBasicInfo {
            reserved1: 0,
            peb_base_address: 0,
            reserved2: [0; 2],
            unique_process_id: 0,
            inherited_from_unique_process_id: 0,
        };
        let mut ret_len: u32 = 0;
        // SAFETY: `info` matches `PROCESS_BASIC_INFORMATION` size.
        let status = unsafe {
            NtQueryInformationProcess(
                h.0,
                PROCESS_BASIC_INFORMATION_CLASS,
                (&mut info as *mut ProcessBasicInfo).cast::<c_void>(),
                std::mem::size_of::<ProcessBasicInfo>() as u32,
                &mut ret_len,
            )
        };
        if status >= 0 {
            Some(info.inherited_from_unique_process_id as u32)
        } else {
            None
        }
    }

    pub fn exe_path(pid: u32) -> Option<String> {
        let h = open_query(pid)?;
        let mut buf = vec![0u16; 32768];
        let mut len = buf.len() as u32;
        // SAFETY: `buf` has capacity `len` u16 elements.
        let ok = unsafe { QueryFullProcessImageNameW(h.0, 0, buf.as_mut_ptr(), &mut len) };
        if ok != 0 && (len as usize) <= buf.len() {
            Some(String::from_utf16_lossy(&buf[..len as usize]))
        } else {
            None
        }
    }

    pub fn command_line(pid: u32) -> Option<Vec<String>> {
        let h = open_query(pid)?;
        let mut needed: u32 = 0;
        // SAFETY: Probing required buffer size for `ProcessCommandLineInformation`.
        unsafe {
            NtQueryInformationProcess(
                h.0,
                PROCESS_COMMAND_LINE_INFORMATION_CLASS,
                ptr::null_mut(),
                0,
                &mut needed,
            );
        }
        if needed < std::mem::size_of::<UnicodeString>() as u32 || needed > 1024 * 1024 {
            return None;
        }
        let mut buf = vec![0u8; needed as usize + 64];
        let mut ret_len: u32 = 0;
        // SAFETY: `buf` is at least `needed` bytes.
        let status = unsafe {
            NtQueryInformationProcess(
                h.0,
                PROCESS_COMMAND_LINE_INFORMATION_CLASS,
                buf.as_mut_ptr().cast::<c_void>(),
                buf.len() as u32,
                &mut ret_len,
            )
        };
        if status < 0 {
            return None;
        }
        // SAFETY: `buf` begins with a `UNICODE_STRING` populated by `NtQueryInformationProcess`.
        let us = unsafe { &*(buf.as_ptr().cast::<UnicodeString>()) };
        if us.buffer.is_null() || us.length == 0 {
            return None;
        }
        let n_chars = (us.length as usize) / 2;
        // SAFETY: `us.buffer` points inside `buf` and holds `n_chars` UTF-16 code units.
        let slice = unsafe { std::slice::from_raw_parts(us.buffer, n_chars) };
        let mut wide = Vec::with_capacity(n_chars + 1);
        wide.extend_from_slice(slice);
        wide.push(0);

        let mut argc: i32 = 0;
        // SAFETY: `wide` is NUL-terminated UTF-16.
        let argv = unsafe { CommandLineToArgvW(wide.as_ptr(), &mut argc) };
        if argv.is_null() || argc <= 0 {
            return None;
        }
        let mut out = Vec::with_capacity(argc as usize);
        for i in 0..(argc as usize) {
            // SAFETY: `argv` has `argc` NUL-terminated wide string pointers.
            unsafe {
                let p = *argv.add(i);
                let mut len = 0usize;
                while *p.add(len) != 0 {
                    len += 1;
                }
                out.push(String::from_utf16_lossy(std::slice::from_raw_parts(p, len)));
            }
        }
        // SAFETY: `argv` was allocated by `CommandLineToArgvW`.
        unsafe {
            LocalFree(argv.cast::<c_void>());
        }
        if out.is_empty() { None } else { Some(out) }
    }
}

pub fn start_token(pid: u32) -> Option<String> {
    if pid == 0 {
        return None;
    }
    #[cfg(windows)]
    {
        win::process_started(pid).map(|t| format!("w:{t}"))
    }
    #[cfg(target_os = "linux")]
    {
        let raw = std::fs::read_to_string(format!("/proc/{pid}/stat")).ok()?;
        let after = &raw[raw.rfind(')')? + 2..];
        let mut parts = after.split_whitespace();
        let state = parts.next()?;
        if state == "Z" {
            return None;
        }
        let starttime = parts.nth(18)?;
        Some(format!("l:{starttime}"))
    }
    #[cfg(not(any(windows, target_os = "linux")))]
    {
        None
    }
}

pub fn command_line(pid: u32) -> Option<Vec<String>> {
    if pid == 0 {
        return None;
    }
    #[cfg(windows)]
    {
        win::command_line(pid)
    }
    #[cfg(target_os = "linux")]
    {
        let raw = std::fs::read(format!("/proc/{pid}/cmdline")).ok()?;
        let parts: Vec<String> = raw
            .split(|&b| b == 0)
            .filter(|p| !p.is_empty())
            .map(|p| String::from_utf8_lossy(p).into_owned())
            .collect();
        if parts.is_empty() { None } else { Some(parts) }
    }
    #[cfg(not(any(windows, target_os = "linux")))]
    {
        None
    }
}

pub fn parent_of(pid: u32) -> Option<u32> {
    if pid == 0 {
        return None;
    }
    #[cfg(windows)]
    {
        win::parent_of(pid)
    }
    #[cfg(target_os = "linux")]
    {
        let raw = std::fs::read_to_string(format!("/proc/{pid}/stat")).ok()?;
        let after = &raw[raw.rfind(')')? + 2..];
        after.split_whitespace().nth(1)?.parse::<u32>().ok()
    }
    #[cfg(not(any(windows, target_os = "linux")))]
    {
        None
    }
}

pub fn exe_path(pid: u32) -> Option<String> {
    if pid == 0 {
        return None;
    }
    #[cfg(windows)]
    {
        win::exe_path(pid)
    }
    #[cfg(target_os = "linux")]
    {
        std::fs::read_link(format!("/proc/{pid}/exe"))
            .ok()
            .map(|p| p.to_string_lossy().into_owned())
    }
    #[cfg(not(any(windows, target_os = "linux")))]
    {
        None
    }
}

pub fn json_escape(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 2);
    out.push('"');
    for ch in s.chars() {
        match ch {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out.push('"');
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn self_process_is_identifiable() {
        let pid = std::process::id();
        let tok = start_token(pid).expect("start_token for self");
        assert!(tok.starts_with("w:") || tok.starts_with("l:"));
        let cmd = command_line(pid).expect("command_line for self");
        assert!(!cmd.is_empty());
        assert!(exe_path(pid).is_some());
    }

    #[test]
    fn nonexistent_pid_returns_none() {
        assert!(start_token(0).is_none());
        assert!(start_token(999_999_999).is_none());
        assert!(command_line(999_999_999).is_none());
    }
}
