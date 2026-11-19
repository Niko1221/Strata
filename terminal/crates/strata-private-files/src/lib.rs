use std::io;
use std::path::Path;

#[cfg(windows)]
mod win32 {
    use std::ffi::c_void;
    use std::io;
    use std::os::windows::ffi::OsStrExt;
    use std::path::Path;
    use std::ptr;

    type Bool = i32;
    type Dword = u32;
    type Handle = *mut c_void;
    type Psid = *mut c_void;
    type Pacl = *mut c_void;
    type PsecDesc = *mut c_void;
    type Lpcwstr = *const u16;
    type Lpwstr = *mut u16;

    const TOKEN_QUERY: Dword = 0x0008;
    const TOKEN_USER: Dword = 1;
    const SE_FILE_OBJECT: Dword = 1;
    const OWNER_SECURITY_INFORMATION: Dword = 0x0000_0001;
    const DACL_SECURITY_INFORMATION: Dword = 0x0000_0004;
    const PROTECTED_DACL_SECURITY_INFORMATION: Dword = 0x8000_0000;
    const SDDL_REVISION_1: Dword = 1;

    #[repr(C)]
    struct SidAndAttributes {
        sid: Psid,
        attributes: Dword,
    }

    #[repr(C)]
    struct TokenUser {
        user: SidAndAttributes,
    }

    #[link(name = "kernel32")]
    extern "system" {
        fn GetCurrentProcess() -> Handle;
        fn CloseHandle(h: Handle) -> Bool;
        fn LocalFree(hmem: *mut c_void) -> *mut c_void;
    }

    #[link(name = "advapi32")]
    extern "system" {
        fn OpenProcessToken(process: Handle, desired_access: Dword, token: *mut Handle) -> Bool;
        fn GetTokenInformation(
            token: Handle,
            info_class: Dword,
            info: *mut c_void,
            info_len: Dword,
            return_len: *mut Dword,
        ) -> Bool;
        fn ConvertSidToStringSidW(sid: Psid, string_sid: *mut Lpwstr) -> Bool;
        fn ConvertStringSidToSidW(string_sid: Lpcwstr, sid: *mut Psid) -> Bool;
        fn ConvertStringSecurityDescriptorToSecurityDescriptorW(
            string_sd: Lpcwstr,
            revision: Dword,
            sd: *mut PsecDesc,
            sd_size: *mut Dword,
        ) -> Bool;
        fn ConvertSecurityDescriptorToStringSecurityDescriptorW(
            sd: PsecDesc,
            revision: Dword,
            sec_info: Dword,
            string_sd: *mut Lpwstr,
            string_sd_len: *mut Dword,
        ) -> Bool;
        fn GetSecurityDescriptorDacl(
            sd: PsecDesc,
            dacl_present: *mut Bool,
            dacl: *mut Pacl,
            dacl_defaulted: *mut Bool,
        ) -> Bool;
        fn GetNamedSecurityInfoW(
            object_name: Lpcwstr,
            object_type: Dword,
            sec_info: Dword,
            owner: *mut Psid,
            group: *mut Psid,
            dacl: *mut Pacl,
            sacl: *mut Pacl,
            sd: *mut PsecDesc,
        ) -> Dword;
        fn SetNamedSecurityInfoW(
            object_name: Lpwstr,
            object_type: Dword,
            sec_info: Dword,
            owner: Psid,
            group: Psid,
            dacl: Pacl,
            sacl: Pacl,
        ) -> Dword;
    }

    struct LocalMem(*mut c_void);
    impl Drop for LocalMem {
        fn drop(&mut self) {
            if !self.0.is_null() {
                unsafe {
                    LocalFree(self.0);
                }
            }
        }
    }

    struct OwnedHandle(Handle);
    impl Drop for OwnedHandle {
        fn drop(&mut self) {
            if !self.0.is_null() {
                unsafe {
                    CloseHandle(self.0);
                }
            }
        }
    }

    fn to_wide_null(s: &str) -> Vec<u16> {
        s.encode_utf16().chain(std::iter::once(0)).collect()
    }

    fn path_to_wide_null(p: &Path) -> Vec<u16> {
        p.as_os_str()
            .encode_wide()
            .chain(std::iter::once(0))
            .collect()
    }

    unsafe fn wide_ptr_to_string(ptr: Lpwstr) -> String {
        if ptr.is_null() {
            return String::new();
        }
        let mut len = 0usize;
        while *ptr.add(len) != 0 {
            len += 1;
        }
        let slice = std::slice::from_raw_parts(ptr, len);
        String::from_utf16_lossy(slice)
    }

    pub fn current_user_sid() -> io::Result<String> {
        unsafe {
            let mut token: Handle = ptr::null_mut();
            if OpenProcessToken(GetCurrentProcess(), TOKEN_QUERY, &mut token) == 0 {
                return Err(io::Error::last_os_error());
            }
            let _token_guard = OwnedHandle(token);

            let mut needed: Dword = 0;
            GetTokenInformation(token, TOKEN_USER, ptr::null_mut(), 0, &mut needed);
            if needed == 0 {
                return Err(io::Error::last_os_error());
            }
            let mut buf = vec![0u8; needed as usize];
            if GetTokenInformation(
                token,
                TOKEN_USER,
                buf.as_mut_ptr() as *mut c_void,
                needed,
                &mut needed,
            ) == 0
            {
                return Err(io::Error::last_os_error());
            }

            let token_user = &*(buf.as_ptr() as *const TokenUser);
            let mut sid_str: Lpwstr = ptr::null_mut();
            if ConvertSidToStringSidW(token_user.user.sid, &mut sid_str) == 0 {
                return Err(io::Error::last_os_error());
            }
            let _sid_guard = LocalMem(sid_str as *mut c_void);
            Ok(wide_ptr_to_string(sid_str))
        }
    }

    pub fn dacl_sddl(path: &Path) -> io::Result<String> {
        let wide_path = path_to_wide_null(path);
        unsafe {
            let mut sd: PsecDesc = ptr::null_mut();
            let err = GetNamedSecurityInfoW(
                wide_path.as_ptr(),
                SE_FILE_OBJECT,
                DACL_SECURITY_INFORMATION,
                ptr::null_mut(),
                ptr::null_mut(),
                ptr::null_mut(),
                ptr::null_mut(),
                &mut sd,
            );
            if err != 0 {
                return Err(io::Error::from_raw_os_error(err as i32));
            }
            let _sd_guard = LocalMem(sd);

            let mut sddl_ptr: Lpwstr = ptr::null_mut();
            if ConvertSecurityDescriptorToStringSecurityDescriptorW(
                sd,
                SDDL_REVISION_1,
                DACL_SECURITY_INFORMATION,
                &mut sddl_ptr,
                ptr::null_mut(),
            ) == 0
            {
                return Err(io::Error::last_os_error());
            }
            let _sddl_guard = LocalMem(sddl_ptr as *mut c_void);
            Ok(wide_ptr_to_string(sddl_ptr))
        }
    }

    pub fn restrict_to_current_user(path: &Path, directory: Option<bool>) -> io::Result<()> {
        let is_dir = directory.unwrap_or_else(|| path.is_dir());
        let sid = current_user_sid()?;
        let sddl = if is_dir {
            format!("D:PAI(A;OICI;FA;;;{sid})")
        } else {
            format!("D:PAI(A;;FA;;;{sid})")
        };

        let wide_sid = to_wide_null(&sid);
        let wide_sddl = to_wide_null(&sddl);
        let mut wide_path = path_to_wide_null(path);

        unsafe {
            let mut psid: Psid = ptr::null_mut();
            if ConvertStringSidToSidW(wide_sid.as_ptr(), &mut psid) == 0 {
                return Err(io::Error::last_os_error());
            }
            let _psid_guard = LocalMem(psid);

            let mut sd: PsecDesc = ptr::null_mut();
            if ConvertStringSecurityDescriptorToSecurityDescriptorW(
                wide_sddl.as_ptr(),
                SDDL_REVISION_1,
                &mut sd,
                ptr::null_mut(),
            ) == 0
            {
                return Err(io::Error::last_os_error());
            }
            let _sd_guard = LocalMem(sd);

            let mut present: Bool = 0;
            let mut defaulted: Bool = 0;
            let mut dacl: Pacl = ptr::null_mut();
            if GetSecurityDescriptorDacl(sd, &mut present, &mut dacl, &mut defaulted) == 0
                || present == 0
            {
                return Err(io::Error::last_os_error());
            }

            let mut err = SetNamedSecurityInfoW(
                wide_path.as_mut_ptr(),
                SE_FILE_OBJECT,
                OWNER_SECURITY_INFORMATION
                    | DACL_SECURITY_INFORMATION
                    | PROTECTED_DACL_SECURITY_INFORMATION,
                psid,
                ptr::null_mut(),
                dacl,
                ptr::null_mut(),
            );
            if err != 0 {
                err = SetNamedSecurityInfoW(
                    wide_path.as_mut_ptr(),
                    SE_FILE_OBJECT,
                    DACL_SECURITY_INFORMATION | PROTECTED_DACL_SECURITY_INFORMATION,
                    ptr::null_mut(),
                    ptr::null_mut(),
                    dacl,
                    ptr::null_mut(),
                );
            }
            if err != 0 {
                return Err(io::Error::from_raw_os_error(err as i32));
            }
            Ok(())
        }
    }

    pub fn string_sid(trustee: &str) -> String {
        let upper = trustee.to_ascii_uppercase();
        if upper.starts_with("S-1-") {
            return upper;
        }
        let wide = to_wide_null(trustee);
        unsafe {
            let mut psid: Psid = ptr::null_mut();
            if ConvertStringSidToSidW(wide.as_ptr(), &mut psid) == 0 {
                return upper;
            }
            let _psid_guard = LocalMem(psid);
            let mut text: Lpwstr = ptr::null_mut();
            if ConvertSidToStringSidW(psid, &mut text) == 0 {
                return upper;
            }
            let _text_guard = LocalMem(text as *mut c_void);
            wide_ptr_to_string(text).to_ascii_uppercase()
        }
    }
}

pub fn current_user_sid() -> io::Result<String> {
    #[cfg(windows)]
    {
        win32::current_user_sid()
    }
    #[cfg(not(windows))]
    {
        Ok(String::new())
    }
}

pub fn dacl_sddl(path: &Path) -> io::Result<String> {
    #[cfg(windows)]
    {
        win32::dacl_sddl(path)
    }
    #[cfg(not(windows))]
    {
        let _ = path;
        Ok(String::new())
    }
}

pub fn restrict_to_current_user(path: &Path, directory: Option<bool>) -> io::Result<()> {
    #[cfg(windows)]
    {
        win32::restrict_to_current_user(path, directory)
    }
    #[cfg(all(unix, not(windows)))]
    {
        use std::os::unix::fs::PermissionsExt;
        let is_dir = directory.unwrap_or_else(|| path.is_dir());
        let mode = if is_dir { 0o700 } else { 0o600 };
        std::fs::set_permissions(path, std::fs::Permissions::from_mode(mode))
    }
    #[cfg(not(any(windows, unix)))]
    {
        let _ = (path, directory);
        Ok(())
    }
}

fn is_privileged_sid(sid: &str) -> bool {
    if matches!(
        sid,
        "S-1-5-18" | "S-1-5-32-544" | "S-1-3-4" | "S-1-3-0" | "S-1-3-1"
    ) {
        return true;
    }
    if let Some(rest) = sid.strip_prefix("S-1-5-21-") {
        if let Some(prefix) = rest.strip_suffix("-500") {
            return !prefix.is_empty() && prefix.bytes().all(|b| b.is_ascii_digit() || b == b'-');
        }
    }
    false
}

pub fn unshared_offenders(sddl: &str) -> io::Result<Vec<String>> {
    let me = current_user_sid()?.to_ascii_uppercase();
    let mut out = Vec::new();
    let mut rest = sddl;
    while let Some(start) = rest.find('(') {
        let after = &rest[start + 1..];
        let Some(end) = after.find(')') else {
            break;
        };
        let entry = &after[..end];
        rest = &after[end + 1..];
        if entry.contains('(') {
            continue;
        }
        let fields: Vec<&str> = entry.split(';').collect();
        if fields.len() < 6 || !matches!(fields[0], "A" | "OA") {
            continue;
        }
        let flags = fields[1].to_ascii_uppercase();
        let has_io = flags.as_bytes().chunks(2).any(|c| c == b"IO");
        if has_io {
            continue;
        }
        let trustee = fields[5];
        let trustee_upper = trustee.to_ascii_uppercase();
        if matches!(
            trustee_upper.as_str(),
            "SY" | "BA" | "OW" | "CO" | "CG" | "LA"
        ) {
            continue;
        }
        #[cfg(windows)]
        let resolved = win32::string_sid(trustee);
        #[cfg(not(windows))]
        let resolved = trustee_upper;

        if resolved == me || is_privileged_sid(&resolved) {
            continue;
        }
        let owned = trustee.to_string();
        if !out.contains(&owned) {
            out.push(owned);
        }
    }
    Ok(out)
}

pub fn is_unshared(path: &Path) -> io::Result<bool> {
    #[cfg(windows)]
    {
        let sddl = dacl_sddl(path)?;
        Ok(unshared_offenders(&sddl)?.is_empty())
    }
    #[cfg(all(unix, not(windows)))]
    {
        use std::os::unix::fs::MetadataExt;
        let meta = std::fs::metadata(path)?;
        Ok((meta.mode() & 0o077) == 0)
    }
    #[cfg(not(any(windows, unix)))]
    {
        let _ = path;
        Ok(true)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::{SystemTime, UNIX_EPOCH};

    #[test]
    fn restrict_and_check_temp_file() {
        let stamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let dir = std::env::temp_dir().join(format!("strata-pf-test-{stamp}"));
        std::fs::create_dir_all(&dir).unwrap();
        let file = dir.join("secret.txt");
        std::fs::write(&file, b"secret").unwrap();

        restrict_to_current_user(&dir, Some(true)).unwrap();
        restrict_to_current_user(&file, Some(false)).unwrap();
        assert!(is_unshared(&dir).unwrap());
        assert!(is_unshared(&file).unwrap());

        let _ = std::fs::remove_file(&file);
        let _ = std::fs::remove_dir(&dir);
    }

    #[test]
    fn offenders_detects_everyone() {
        let offenders = unshared_offenders("D:P(A;;FA;;;WD)(A;;FA;;;SY)(A;;FA;;;BA)").unwrap();
        assert_eq!(offenders, vec!["WD".to_string()]);
    }
}
