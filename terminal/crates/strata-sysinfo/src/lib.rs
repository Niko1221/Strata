use std::env;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SysInfo {
    pub system: String,
    pub machine: String,
    pub release: String,
    pub version: String,
    pub processor: String,
    pub node: String,
    pub platform_string: String,
}

fn map_windows_arch(raw: &str) -> String {
    match raw.trim().to_ascii_lowercase().as_str() {
        "amd64" | "x86_64" => "AMD64".to_string(),
        "arm64" | "aarch64" => "ARM64".to_string(),
        "x86" | "i386" | "i686" => "x86".to_string(),
        "ia64" => "ia64".to_string(),
        _ => raw.trim().to_string(),
    }
}

pub fn windows_machine() -> String {
    for key in ["PROCESSOR_ARCHITEW6432", "PROCESSOR_ARCHITECTURE"] {
        if let Ok(val) = env::var(key) {
            let trimmed = val.trim();
            if !trimmed.is_empty() {
                return map_windows_arch(trimmed);
            }
        }
    }
    match env::consts::ARCH {
        "x86_64" => "AMD64".to_string(),
        "aarch64" => "ARM64".to_string(),
        "x86" => "x86".to_string(),
        other => other.to_string(),
    }
}

#[cfg(windows)]
mod win {
    #[repr(C)]
    #[allow(non_snake_case)]
    struct OsVersionInfoExW {
        dwOSVersionInfoSize: u32,
        dwMajorVersion: u32,
        dwMinorVersion: u32,
        dwBuildNumber: u32,
        dwPlatformId: u32,
        szCSDVersion: [u16; 128],
        wServicePackMajor: u16,
        wServicePackMinor: u16,
        wSuiteMask: u16,
        wProductType: u8,
        wReserved: u8,
    }

    #[link(name = "ntdll")]
    extern "system" {
        fn RtlGetVersion(lpVersionInformation: *mut OsVersionInfoExW) -> i32;
    }

    #[link(name = "kernel32")]
    extern "system" {
        fn GetComputerNameW(lpBuffer: *mut u16, nSize: *mut u32) -> i32;
    }

    pub fn windows_version() -> Option<(u32, u32, u32)> {
        let mut info = OsVersionInfoExW {
            dwOSVersionInfoSize: std::mem::size_of::<OsVersionInfoExW>() as u32,
            dwMajorVersion: 0,
            dwMinorVersion: 0,
            dwBuildNumber: 0,
            dwPlatformId: 0,
            szCSDVersion: [0; 128],
            wServicePackMajor: 0,
            wServicePackMinor: 0,
            wSuiteMask: 0,
            wProductType: 0,
            wReserved: 0,
        };
        // SAFETY: `info` is properly sized and initialized for `RtlGetVersion`.
        let status = unsafe { RtlGetVersion(&mut info) };
        if status == 0 {
            Some((info.dwMajorVersion, info.dwMinorVersion, info.dwBuildNumber))
        } else {
            None
        }
    }

    pub fn computer_name() -> Option<String> {
        let mut buf = [0u16; 256];
        let mut len = buf.len() as u32;
        // SAFETY: `buf` has capacity `len` u16 elements.
        let ok = unsafe { GetComputerNameW(buf.as_mut_ptr(), &mut len) };
        if ok != 0 && (len as usize) <= buf.len() {
            Some(String::from_utf16_lossy(&buf[..len as usize]))
        } else {
            None
        }
    }
}

pub fn windows_release(major: u32, build: u32) -> String {
    if major == 10 && build >= 22000 {
        "11".to_string()
    } else {
        major.to_string()
    }
}

pub fn gather() -> SysInfo {
    #[cfg(windows)]
    {
        let system = "Windows".to_string();
        let machine = windows_machine();
        let (release, version) = match win::windows_version() {
            Some((maj, min, bld)) => (windows_release(maj, bld), format!("{maj}.{min}.{bld}")),
            None => (String::new(), String::new()),
        };
        let processor = env::var("PROCESSOR_IDENTIFIER")
            .unwrap_or_default()
            .trim()
            .to_string();
        let node = win::computer_name()
            .or_else(|| env::var("COMPUTERNAME").ok())
            .unwrap_or_default();
        let platform_string = format!("Windows-{release}-{version}");
        SysInfo {
            system,
            machine,
            release,
            version,
            processor,
            node,
            platform_string,
        }
    }
    #[cfg(not(windows))]
    {
        let system = match env::consts::OS {
            "macos" => "Darwin",
            "linux" => "Linux",
            other => other,
        }
        .to_string();
        let machine = env::consts::ARCH.to_string();
        let node = env::var("HOSTNAME").unwrap_or_default();
        SysInfo {
            system: system.clone(),
            machine,
            release: String::new(),
            version: String::new(),
            processor: String::new(),
            node,
            platform_string: system,
        }
    }
}

fn json_escape(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 2);
    out.push('"');
    for ch in s.chars() {
        match ch {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => {
                out.push_str(&format!("\\u{:04x}", c as u32));
            }
            c => out.push(c),
        }
    }
    out.push('"');
    out
}

pub fn to_json(info: &SysInfo) -> String {
    format!(
        "{{\"system\":{},\"machine\":{},\"release\":{},\"version\":{},\"processor\":{},\"node\":{},\"platform_string\":{}}}",
        json_escape(&info.system),
        json_escape(&info.machine),
        json_escape(&info.release),
        json_escape(&info.version),
        json_escape(&info.processor),
        json_escape(&info.node),
        json_escape(&info.platform_string),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn windows_release_maps_build_22000_to_11() {
        assert_eq!(windows_release(10, 19045), "10");
        assert_eq!(windows_release(10, 22000), "11");
        assert_eq!(windows_release(10, 22631), "11");
    }

    #[test]
    fn arch_mapping_matches_python() {
        assert_eq!(map_windows_arch("amd64"), "AMD64");
        assert_eq!(map_windows_arch("x86_64"), "AMD64");
        assert_eq!(map_windows_arch("aarch64"), "ARM64");
        assert_eq!(map_windows_arch("i686"), "x86");
    }

    #[test]
    fn gather_populates_core_fields_on_host() {
        let info = gather();
        assert!(!info.system.is_empty());
        assert!(!info.machine.is_empty());
        let json = to_json(&info);
        assert!(json.starts_with('{') && json.ends_with('}'));
        assert!(json.contains("\"system\":"));
    }
}
