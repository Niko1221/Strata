use std::fs::File;
use std::io::Read;
use std::path::Path;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GgufHeader {
    pub bytes: u64,
    pub gguf_version: u32,
}

pub fn gguf_header(path: &Path) -> Result<GgufHeader, String> {
    let mut file = File::open(path).map_err(|_| "GGUF shard is unreadable".to_string())?;
    let meta = file
        .metadata()
        .map_err(|_| "GGUF shard is unreadable".to_string())?;
    if !meta.is_file() || meta.len() < 8 {
        return Err("GGUF shard is not a nonempty regular file".to_string());
    }
    let mut header = [0u8; 8];
    file.read_exact(&mut header)
        .map_err(|_| "GGUF shard is unreadable".to_string())?;
    if &header[..4] != b"GGUF" {
        return Err("file does not begin with a GGUF header".to_string());
    }
    let version = u32::from_le_bytes([header[4], header[5], header[6], header[7]]);
    if version != 2 && version != 3 {
        return Err("GGUF header version is not recognized".to_string());
    }
    Ok(GgufHeader {
        bytes: meta.len(),
        gguf_version: version,
    })
}

pub fn is_valid_model_id(s: &str) -> bool {
    let bytes = s.as_bytes();
    if bytes.is_empty() || bytes.len() > 128 {
        return false;
    }
    let first = bytes[0];
    if !(first.is_ascii_lowercase() || first.is_ascii_digit()) {
        return false;
    }
    let mut prev_dot = false;
    for &b in &bytes[1..] {
        if b == b'.' {
            if prev_dot {
                return false;
            }
            prev_dot = true;
        } else if b.is_ascii_lowercase() || b.is_ascii_digit() || b == b'_' || b == b'-' {
            prev_dot = false;
        } else {
            return false;
        }
    }
    true
}

pub fn parse_shard_name(name: &str) -> Option<(&str, u32, u32)> {
    let stem = name
        .strip_suffix(".gguf")
        .or_else(|| name.strip_suffix(".GGUF"))
        .or_else(|| {
            if name.len() >= 5 && name[name.len() - 5..].eq_ignore_ascii_case(".gguf") {
                Some(&name[..name.len() - 5])
            } else {
                None
            }
        })?;
    // Format: <prefix>-NNNNN-of-MMMMM
    if stem.len() < 16 {
        return None;
    }
    let total_part = &stem[stem.len() - 5..];
    let of_part = &stem[stem.len() - 9..stem.len() - 5];
    let idx_part = &stem[stem.len() - 14..stem.len() - 9];
    let dash = stem.as_bytes()[stem.len() - 15];
    if dash != b'-' || !of_part.eq_ignore_ascii_case("-of-") {
        return None;
    }
    if !idx_part.bytes().all(|b| b.is_ascii_digit())
        || !total_part.bytes().all(|b| b.is_ascii_digit())
    {
        return None;
    }
    let prefix = &stem[..stem.len() - 15];
    if prefix.is_empty() {
        return None;
    }
    let idx: u32 = idx_part.parse().ok()?;
    let total: u32 = total_part.parse().ok()?;
    Some((prefix, idx, total))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use std::time::{SystemTime, UNIX_EPOCH};

    #[test]
    fn gguf_header_accepts_v3_and_rejects_bad_magic() {
        let stamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let path = std::env::temp_dir().join(format!("strata-models-{stamp}.gguf"));
        {
            let mut f = File::create(&path).unwrap();
            f.write_all(b"GGUF\x03\x00\x00\x00extra").unwrap();
        }
        let h = gguf_header(&path).unwrap();
        assert_eq!(h.gguf_version, 3);
        assert_eq!(h.bytes, 13);

        std::fs::write(&path, b"NOPE\x03\x00\x00\x00").unwrap();
        assert!(gguf_header(&path).is_err());
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn model_id_and_shard_validation() {
        assert!(is_valid_model_id("deepseek-v4.1-flash-ist-das-2.00bpw"));
        assert!(!is_valid_model_id(".hidden"));
        assert!(!is_valid_model_id("a..b"));
        assert_eq!(
            parse_shard_name("Model-00001-of-00009.gguf"),
            Some(("Model", 1, 9))
        );
        assert_eq!(parse_shard_name("single.gguf"), None);
    }
}
