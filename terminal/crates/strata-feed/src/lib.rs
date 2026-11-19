pub mod json;

use json::Json;

pub const SCHEMA_VERSION: i64 = 1;

pub fn version_key(value: &str) -> Result<[u64; 4], String> {
    let parts: Vec<&str> = value.split('.').collect();
    if !(3..=4).contains(&parts.len()) || parts.iter().any(|p| p.is_empty() || !p.bytes().all(|b| b.is_ascii_digit())) {
        return Err(format!("{value:?} is not a version number (expected like 0.1.46 or 0.1.46.1)"));
    }
    let mut out = [0u64; 4];
    for (i, p) in parts.iter().enumerate() {
        out[i] = p.parse::<u64>().map_err(|_| format!("{value:?} is not a version number"))?;
    }
    Ok(out)
}

pub fn app_version_key(value: &str) -> Result<([u64; 4], u8), String> {
    if let Some((base, pre)) = value.split_once('-') {
        if pre.is_empty() || !pre.bytes().all(|b| b.is_ascii_alphanumeric() || b == b'.') {
            return Err(format!("{value:?} is not a version number (expected like 0.2.0 or 0.2.0-beta)"));
        }
        Ok((version_key(base)?, 0))
    } else {
        Ok((version_key(value)?, 1))
    }
}

pub fn is_hex64(s: &str) -> bool {
    s.len() == 64 && s.bytes().all(|b| b.is_ascii_hexdigit())
}

fn is_int(v: &Json) -> Option<i64> {
    match v {
        Json::Int(n) => Some(*n),
        Json::Num(n) if n.is_finite() && n.fract() == 0.0 => Some(*n as i64),
        _ => None,
    }
}

pub fn validate_feed_bytes(payload: &[u8]) -> Result<(), String> {
    let text = std::str::from_utf8(payload).map_err(|_| "the update feed is not valid JSON".to_string())?;
    let text = text.strip_prefix('\u{feff}').unwrap_or(text);
    let data = json::parse(text).map_err(|_| "the update feed is not valid JSON".to_string())?;
    let Json::Obj(ref map) = data else {
        return Err("the update feed has an unsupported schema_version".to_string());
    };
    if map.get("schema_version").and_then(is_int) != Some(SCHEMA_VERSION) {
        return Err("the update feed has an unsupported schema_version".to_string());
    }
    let Some(seq) = map.get("sequence").and_then(is_int) else {
        return Err("the update feed has no sequence number".to_string());
    };
    if seq < 0 {
        return Err("the update feed has no sequence number".to_string());
    }
    if map.get("channel").and_then(|v| v.as_str()).is_none() {
        return Err("the feed's header has no valid channel".to_string());
    }
    if map.get("generated_at").and_then(|v| v.as_str()).is_none() {
        return Err("the feed's header has no valid generated_at".to_string());
    }
    if let Some(raw_app) = map.get("app") {
        if *raw_app != Json::Null {
            let Json::Obj(ref app) = *raw_app else {
                return Err("the feed's app member is not an object".to_string());
            };
            let Some(ver) = app.get("version").and_then(|v| v.as_str()) else {
                return Err("the feed's app entry has no valid version".to_string());
            };
            app_version_key(ver)?;
            let Some(Json::Obj(ref plats)) = app.get("platforms") else {
                return Err("the feed's app entry lists no platforms".to_string());
            };
            if plats.is_empty() {
                return Err("the feed's app entry lists no platforms".to_string());
            }
        }
    }
    if let Some(raw_ctrl) = map.get("control") {
        if *raw_ctrl != Json::Null {
            let Json::Obj(ref ctrl) = *raw_ctrl else {
                return Err("the feed's control member is not an object".to_string());
            };
            let Some(cver) = ctrl.get("version").and_then(|v| v.as_str()) else {
                return Err("the feed's control entry has no valid version".to_string());
            };
            version_key(cver)?;
        }
    }
    for (member, err) in [
        ("engine", "the feed's engine member lists no packages"),
        ("dsv41", "the feed's dsv41 member lists no packages"),
        ("shape_engines", "the feed's shape_engines member lists no packages"),
        ("drafters", "the feed's drafters member lists no packages"),
    ] {
        if let Some(raw_m) = map.get(member) {
            if *raw_m != Json::Null {
                let Some(Json::Obj(ref m)) = Some(raw_m) else {
                    return Err(err.to_string());
                };
                if !matches!(m.get("packages"), Some(Json::Arr(_))) {
                    return Err(err.to_string());
                }
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn version_keys_parse_strictly() {
        assert_eq!(version_key("0.2.14").unwrap(), [0, 2, 14, 0]);
        assert_eq!(version_key("0.2.14.1").unwrap(), [0, 2, 14, 1]);
        assert!(version_key("0.2").is_err());
        assert_eq!(app_version_key("0.2.0-beta").unwrap(), ([0, 2, 0, 0], 0));
    }

    #[test]
    fn minimal_feed_validates_and_rejects_bad_schema() {
        let ok = br#"{"schema_version":1,"sequence":42,"channel":"stable","generated_at":"2026-11-14T00:00:00Z"}"#;
        assert!(validate_feed_bytes(ok).is_ok());
        let bad = br#"{"schema_version":2,"sequence":42,"channel":"stable","generated_at":"2026-11-14T00:00:00Z"}"#;
        assert!(validate_feed_bytes(bad).is_err());
        let big_seq_and_emoji = br#"{"schema_version":1,"sequence":9007199254740993,"channel":"stable-\uD83D\uDE00","generated_at":"2026-11-14T00:00:00Z"}"#;
        assert!(validate_feed_bytes(big_seq_and_emoji).is_ok());
        let dup_key = br#"{"schema_version":1,"schema_version":1,"sequence":42,"channel":"stable","generated_at":"2026-11-14T00:00:00Z"}"#;
        assert!(validate_feed_bytes(dup_key).is_err());
    }
}
