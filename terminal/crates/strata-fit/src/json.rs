use std::collections::BTreeMap;

#[derive(Debug, Clone, PartialEq)]
pub enum Json {
    Null,
    Bool(bool),
    Int(i64),
    Num(f64),
    Str(String),
    Arr(Vec<Json>),
    Obj(BTreeMap<String, Json>),
}

impl Json {
    pub fn as_str(&self) -> Option<&str> {
        match self {
            Json::Str(s) => Some(s.as_str()),
            _ => None,
        }
    }
    pub fn as_f64(&self) -> Option<f64> {
        match self {
            Json::Int(n) => Some(*n as f64),
            Json::Num(n) => Some(*n),
            _ => None,
        }
    }
    pub fn as_i64(&self) -> Option<i64> {
        match self {
            Json::Int(n) => Some(*n),
            Json::Num(n) => Some(*n as i64),
            _ => None,
        }
    }
    pub fn as_bool(&self) -> Option<bool> {
        match self {
            Json::Bool(b) => Some(*b),
            _ => None,
        }
    }
    pub fn get(&self, key: &str) -> Option<&Json> {
        match self {
            Json::Obj(m) => m.get(key),
            _ => None,
        }
    }
    pub fn as_arr(&self) -> Option<&[Json]> {
        match self {
            Json::Arr(a) => Some(a.as_slice()),
            _ => None,
        }
    }
}

pub fn parse(input: &str) -> Result<Json, String> {
    let bytes = input.as_bytes();
    let mut i = 0;
    let val = parse_val(bytes, &mut i)?;
    skip_ws(bytes, &mut i);
    if i != bytes.len() {
        return Err(format!("trailing bytes at {i}"));
    }
    Ok(val)
}

fn skip_ws(b: &[u8], i: &mut usize) {
    while *i < b.len() && matches!(b[*i], b' ' | b'\t' | b'\r' | b'\n') {
        *i += 1;
    }
}

fn parse_val(b: &[u8], i: &mut usize) -> Result<Json, String> {
    skip_ws(b, i);
    if *i >= b.len() {
        return Err("unexpected EOF".to_string());
    }
    match b[*i] {
        b'n' => {
            if b[*i..].starts_with(b"null") {
                *i += 4;
                Ok(Json::Null)
            } else {
                Err("expected null".to_string())
            }
        }
        b't' => {
            if b[*i..].starts_with(b"true") {
                *i += 4;
                Ok(Json::Bool(true))
            } else {
                Err("expected true".to_string())
            }
        }
        b'f' => {
            if b[*i..].starts_with(b"false") {
                *i += 5;
                Ok(Json::Bool(false))
            } else {
                Err("expected false".to_string())
            }
        }
        b'"' => Ok(Json::Str(parse_str(b, i)?)),
        b'[' => parse_arr(b, i),
        b'{' => parse_obj(b, i),
        b'-' | b'0'..=b'9' => parse_num(b, i),
        c => Err(format!("unexpected byte {c} at {}", *i)),
    }
}

fn parse_str(b: &[u8], i: &mut usize) -> Result<String, String> {
    *i += 1; // skip '"'
    let mut out = String::new();
    while *i < b.len() {
        let c = b[*i];
        if c == b'"' {
            *i += 1;
            return Ok(out);
        }
        if c == b'\\' {
            *i += 1;
            if *i >= b.len() {
                return Err("unterminated escape".to_string());
            }
            match b[*i] {
                b'"' => out.push('"'),
                b'\\' => out.push('\\'),
                b'/' => out.push('/'),
                b'b' => out.push('\x08'),
                b'f' => out.push('\x0c'),
                b'n' => out.push('\n'),
                b'r' => out.push('\r'),
                b't' => out.push('\t'),
                b'u' => {
                    if *i + 4 >= b.len() {
                        return Err("short unicode escape".to_string());
                    }
                    let hex = std::str::from_utf8(&b[*i + 1..*i + 5]).map_err(|e| e.to_string())?;
                    let code = u32::from_str_radix(hex, 16).map_err(|e| e.to_string())?;
                    *i += 4;
                    if (0xD800..=0xDBFF).contains(&code) {
                        if *i + 6 >= b.len() || b[*i + 1] != b'\\' || b[*i + 2] != b'u' {
                            return Err("unpaired high surrogate".to_string());
                        }
                        let lo_hex = std::str::from_utf8(&b[*i + 3..*i + 7]).map_err(|e| e.to_string())?;
                        let lo = u32::from_str_radix(lo_hex, 16).map_err(|e| e.to_string())?;
                        if !(0xDC00..=0xDFFF).contains(&lo) {
                            return Err("invalid low surrogate".to_string());
                        }
                        let cp = 0x10000 + ((code - 0xD800) << 10) + (lo - 0xDC00);
                        let ch = char::from_u32(cp).ok_or("invalid surrogate pair")?;
                        out.push(ch);
                        *i += 6;
                    } else if (0xDC00..=0xDFFF).contains(&code) {
                        return Err("unpaired low surrogate".to_string());
                    } else {
                        let ch = char::from_u32(code).ok_or("invalid code point")?;
                        out.push(ch);
                    }
                }
                other => return Err(format!("bad escape \\{other}")),
            }
            *i += 1;
        } else if c < 0x20 {
            return Err("unescaped control character in string".to_string());
        } else {
            let s = std::str::from_utf8(&b[*i..]).map_err(|e| e.to_string())?;
            let ch = s.chars().next().ok_or("eof")?;
            out.push(ch);
            *i += ch.len_utf8();
        }
    }
    Err("unterminated string".to_string())
}

fn parse_num(b: &[u8], i: &mut usize) -> Result<Json, String> {
    let start = *i;
    while *i < b.len()
        && matches!(
            b[*i],
            b'-' | b'+' | b'.' | b'e' | b'E' | b'0'..=b'9'
        )
    {
        *i += 1;
    }
    let s = std::str::from_utf8(&b[start..*i]).map_err(|e| e.to_string())?;
    if !s.contains(['.', 'e', 'E']) {
        if let Ok(n) = s.parse::<i64>() {
            return Ok(Json::Int(n));
        }
    }
    let n: f64 = s.parse().map_err(|e| format!("bad number {s}: {e}"))?;
    if !n.is_finite() {
        return Err(format!("non-finite number {s}"));
    }
    Ok(Json::Num(n))
}

fn parse_arr(b: &[u8], i: &mut usize) -> Result<Json, String> {
    *i += 1;
    let mut items = Vec::new();
    skip_ws(b, i);
    if *i < b.len() && b[*i] == b']' {
        *i += 1;
        return Ok(Json::Arr(items));
    }
    loop {
        items.push(parse_val(b, i)?);
        skip_ws(b, i);
        if *i >= b.len() {
            return Err("unterminated array".to_string());
        }
        if b[*i] == b']' {
            *i += 1;
            return Ok(Json::Arr(items));
        }
        if b[*i] != b',' {
            return Err("expected ',' in array".to_string());
        }
        *i += 1;
    }
}

fn parse_obj(b: &[u8], i: &mut usize) -> Result<Json, String> {
    *i += 1;
    let mut map = BTreeMap::new();
    skip_ws(b, i);
    if *i < b.len() && b[*i] == b'}' {
        *i += 1;
        return Ok(Json::Obj(map));
    }
    loop {
        skip_ws(b, i);
        if *i >= b.len() || b[*i] != b'"' {
            return Err("expected object key".to_string());
        }
        let k = parse_str(b, i)?;
        skip_ws(b, i);
        if *i >= b.len() || b[*i] != b':' {
            return Err("expected ':'".to_string());
        }
        *i += 1;
        let v = parse_val(b, i)?;
        if map.contains_key(&k) {
            return Err(format!("duplicate key {k}"));
        }
        map.insert(k, v);
        skip_ws(b, i);
        if *i >= b.len() {
            return Err("unterminated object".to_string());
        }
        if b[*i] == b'}' {
            *i += 1;
            return Ok(Json::Obj(map));
        }
        if b[*i] != b',' {
            return Err("expected ',' in object".to_string());
        }
        *i += 1;
    }
}

pub fn stringify(v: &Json) -> String {
    let mut out = String::new();
    write_json(v, &mut out);
    out
}

fn write_json(v: &Json, out: &mut String) {
    match v {
        Json::Null => out.push_str("null"),
        Json::Bool(b) => out.push_str(if *b { "true" } else { "false" }),
        Json::Int(n) => out.push_str(&n.to_string()),
        Json::Num(n) => {
            if !n.is_finite() {
                out.push_str("null");
            } else if (*n - (*n as i64 as f64)).abs() == 0.0 && n.abs() < 1e15 {
                out.push_str(&format!("{}", *n as i64));
            } else {
                out.push_str(&format!("{n}"));
            }
        }
        Json::Str(s) => {
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
        }
        Json::Arr(a) => {
            out.push('[');
            for (idx, item) in a.iter().enumerate() {
                if idx > 0 {
                    out.push(',');
                }
                write_json(item, out);
            }
            out.push(']');
        }
        Json::Obj(m) => {
            out.push('{');
            for (idx, (k, val)) in m.iter().enumerate() {
                if idx > 0 {
                    out.push(',');
                }
                write_json(&Json::Str(k.clone()), out);
                out.push(':');
                write_json(val, out);
            }
            out.push('}');
        }
    }
}
