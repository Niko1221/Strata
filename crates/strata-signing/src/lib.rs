//! Ed25519 signatures for the update feed, engine components, and Tauri minisign installers.
//! Mirrors `strata_control/signing.py`.

use base64::engine::general_purpose::STANDARD as B64;
use base64::Engine;
use blake2::digest::consts::U64;
use blake2::{Blake2b, Digest as _};
use ed25519_dalek::{Signature, VerifyingKey};
use serde::{Deserialize, Serialize};
use sha2::Sha256;
use std::collections::HashMap;
use std::fmt;
use std::fs;
use std::path::Path;

pub const ALG: &str = "ed25519";
pub const ENV_EXTRA_KEYS: &str = "STRATA_UPDATE_TRUSTED_KEYS";

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SignatureError(pub String);

impl fmt::Display for SignatureError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl std::error::Error for SignatureError {}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PublicKeyEntry {
    pub alg: String,
    pub key_id: String,
    pub public: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SignatureEnvelope {
    pub alg: String,
    pub key_id: String,
    pub value: String,
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
enum KeysFile {
    Wrapped { keys: Vec<serde_json::Value> },
    List(Vec<serde_json::Value>),
}

pub fn key_id_of(public: &[u8; 32]) -> String {
    let hash = Sha256::digest(public);
    let mut out = String::with_capacity(16);
    for b in &hash[..8] {
        out.push_str(&format!("{b:02x}"));
    }
    out
}

/// Strict RFC 8032 Ed25519 verification (cofactorless; non-canonical encodings and s >= L refused).
pub fn verify_raw(public: &[u8], message: &[u8], signature: &[u8]) -> bool {
    let Ok(pub_bytes): Result<[u8; 32], _> = public.try_into() else {
        return false;
    };
    let Ok(sig_bytes): Result<[u8; 64], _> = signature.try_into() else {
        return false;
    };
    let Ok(vk) = VerifyingKey::from_bytes(&pub_bytes) else {
        return false;
    };
    let sig = Signature::from_bytes(&sig_bytes);
    vk.verify_strict(message, &sig).is_ok()
}

fn decode_b64_exact<const N: usize>(value: &str, what: &str) -> Result<[u8; N], SignatureError> {
    let raw = B64
        .decode(value.as_bytes())
        .map_err(|_| SignatureError(format!("{what} is not valid base64")))?;
    raw.try_into()
        .map_err(|_| SignatureError(format!("{what} must be {N} bytes")))
}

pub fn parse_public(entry: &serde_json::Value) -> Result<(String, [u8; 32]), SignatureError> {
    let obj = entry
        .as_object()
        .ok_or_else(|| SignatureError("a public key entry needs alg ed25519".into()))?;
    if obj.get("alg").and_then(|v| v.as_str()) != Some(ALG) {
        return Err(SignatureError("a public key entry needs alg ed25519".into()));
    }
    let pub_b64 = obj
        .get("public")
        .and_then(|v| v.as_str())
        .ok_or_else(|| SignatureError("a public key is not valid base64".into()))?;
    let public = decode_b64_exact::<32>(pub_b64, "a public key")?;
    let expected_id = key_id_of(&public);
    let given_id = obj.get("key_id").and_then(|v| v.as_str());
    if given_id != Some(expected_id.as_str()) {
        return Err(SignatureError(format!(
            "the key_id {given_id:?} does not belong to that public key"
        )));
    }
    Ok((expected_id, public))
}

pub fn load_trusted_keys_from_paths(
    builtin_keys_path: Option<&Path>,
    extra_keys_path: Option<&Path>,
) -> Result<HashMap<String, [u8; 32]>, SignatureError> {
    let mut keys = HashMap::new();
    let mut paths: Vec<(&Path, bool)> = Vec::new();
    if let Some(p) = builtin_keys_path {
        paths.push((p, true));
    }
    if let Some(p) = extra_keys_path {
        paths.push((p, false));
    }
    for (p, optional_if_missing) in paths {
        if optional_if_missing && !p.exists() {
            continue;
        }
        let raw = fs::read_to_string(p).map_err(|e| {
            SignatureError(format!("the key file {} cannot be read ({e})", p.display()))
        })?;
        let text = raw.strip_prefix('\u{feff}').unwrap_or(&raw);
        let parsed: KeysFile = serde_json::from_str(text).map_err(|e| {
            SignatureError(format!("the key file {} cannot be read ({e})", p.display()))
        })?;
        let entries = match parsed {
            KeysFile::Wrapped { keys } => keys,
            KeysFile::List(entries) => entries,
        };
        for entry in &entries {
            let (id, pubkey) = parse_public(entry)?;
            keys.insert(id, pubkey);
        }
    }
    Ok(keys)
}

pub fn verify_bytes(
    trusted: &HashMap<String, [u8; 32]>,
    payload: &[u8],
    signature: &SignatureEnvelope,
) -> Result<String, SignatureError> {
    if signature.alg != ALG {
        return Err(SignatureError(
            "the signature is not an ed25519 signature".into(),
        ));
    }
    let public = trusted.get(&signature.key_id).ok_or_else(|| {
        SignatureError(format!(
            "the signature was made by a key this PC does not trust ({:?})",
            signature.key_id
        ))
    })?;
    let raw = decode_b64_exact::<64>(&signature.value, "the signature value")?;
    if !verify_raw(public, payload, &raw) {
        return Err(SignatureError(
            "the signature does not match the signed content".into(),
        ));
    }
    Ok(signature.key_id.clone())
}

pub fn minisign_public(text_or_b64: &str) -> Result<([u8; 8], [u8; 32]), SignatureError> {
    let trimmed = text_or_b64.trim();
    let decoded_storage;
    let text = if trimmed.contains("untrusted comment") {
        trimmed
    } else {
        let bytes = B64
            .decode(trimmed.as_bytes())
            .map_err(|_| SignatureError("the minisign public key is not valid".into()))?;
        decoded_storage = String::from_utf8(bytes)
            .map_err(|_| SignatureError("the minisign public key is not valid".into()))?;
        decoded_storage.as_str()
    };
    let key_line = text
        .lines()
        .map(str::trim)
        .find(|l| !l.is_empty() && !l.starts_with("untrusted comment"))
        .ok_or_else(|| SignatureError("the minisign public key has no key line".into()))?;
    let raw = decode_b64_exact::<42>(key_line, "the minisign public key")?;
    if &raw[..2] != b"Ed" {
        return Err(SignatureError(
            "the minisign public key is not an Ed25519 key".into(),
        ));
    }
    let mut key_id = [0u8; 8];
    let mut pubkey = [0u8; 32];
    key_id.copy_from_slice(&raw[2..10]);
    pubkey.copy_from_slice(&raw[10..42]);
    Ok((key_id, pubkey))
}

pub fn minisign_verify(
    public_text: &str,
    signature_text: &str,
    data: &[u8],
) -> Result<(), SignatureError> {
    let (key_id, public) = minisign_public(public_text)?;
    let trimmed = signature_text.trim();
    let decoded_storage;
    let text = if trimmed.contains("untrusted comment") {
        trimmed
    } else {
        let bytes = B64
            .decode(trimmed.as_bytes())
            .map_err(|_| SignatureError("the app signature is not valid base64".into()))?;
        decoded_storage = String::from_utf8(bytes)
            .map_err(|_| SignatureError("the app signature is not valid base64".into()))?;
        decoded_storage.as_str()
    };
    let lines: Vec<&str> = text.lines().map(str::trim).filter(|l| !l.is_empty()).collect();
    let body: Vec<&str> = lines
        .iter()
        .copied()
        .filter(|l| !l.starts_with("untrusted comment") && !l.starts_with("trusted comment"))
        .collect();
    if body.is_empty() {
        return Err(SignatureError("the app signature has no signature line".into()));
    }
    let raw = decode_b64_exact::<74>(body[0], "the app signature")?;
    let alg = &raw[..2];
    let signer = &raw[2..10];
    let sig = &raw[10..74];
    if signer != key_id {
        return Err(SignatureError(
            "the app signature was made by a different key than the one this build trusts".into(),
        ));
    }
    let digest_buf: Vec<u8>;
    let message: &[u8] = match alg {
        b"ED" => {
            let mut hasher = Blake2b::<U64>::new();
            hasher.update(data);
            digest_buf = hasher.finalize().to_vec();
            &digest_buf
        }
        b"Ed" => data,
        _ => {
            return Err(SignatureError(
                "the app signature uses an unknown algorithm".into(),
            ))
        }
    };
    if !verify_raw(&public, message, sig) {
        return Err(SignatureError(
            "the app installer does not match its signature".into(),
        ));
    }
    if let Some(comment) = lines
        .iter()
        .find_map(|l| l.strip_prefix("trusted comment: "))
    {
        if body.len() > 1 {
            let global_sig = decode_b64_exact::<64>(body[1], "the app signature's global part")?;
            let mut combined = Vec::with_capacity(sig.len() + comment.len());
            combined.extend_from_slice(sig);
            combined.extend_from_slice(comment.as_bytes());
            if !verify_raw(&public, &combined, &global_sig) {
                return Err(SignatureError(
                    "the app signature's comment was changed".into(),
                ));
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rfc8032_vector_1() {
        // RFC 8032 Section 7.1 TEST 1 (empty message)
        let pubkey = hex_bytes(
            "d75a980182b10ab7d54bfed3c964073a0ee172f3daa62325af021a68f707511a",
        );
        let sig = hex_bytes(
            "e5564300c360ac729086e2cc806e828a84877f1eb8e5d974d873e06522490155\
             5fb8821590a33bacc61e39701cf9b46bd25bf5f0595bbe24655141438e7a100b",
        );
        assert!(verify_raw(&pubkey, b"", &sig));
        assert!(!verify_raw(&pubkey, b"x", &sig));
    }

    #[test]
    fn strata_builtin_update_key_and_minisign_parse() {
        let entry: serde_json::Value = serde_json::json!({
            "alg": "ed25519",
            "key_id": "5d1bc23abbeffd0d",
            "public": "KjnXZB0T1r3dmBNot8h+c4rTicRtllg1LH9fUlTw4ZY="
        });
        let (id, _) = parse_public(&entry).unwrap();
        assert_eq!(id, "5d1bc23abbeffd0d");

        let app_minisign_b64 = "dW50cnVzdGVkIGNvbW1lbnQ6IG1pbmlzaWduIHB1YmxpYyBrZXk6IEI2QjA5OTk5MDU1MjFGNUUKUldSZUgxSUZtWm13dHNGRU40N2xOUjdZZGpvcGl0NTB2WGxnS3VTc2RlakpSbnBxUkxndHNFY0oK";
        // Verify valid base64 & Ed25519 key parsing
        let res = minisign_public(app_minisign_b64);
        assert!(res.is_ok());
    }

    fn hex_bytes(s: &str) -> Vec<u8> {
        (0..s.len())
            .step_by(2)
            .map(|i| u8::from_str_radix(&s[i..i + 2], 16).unwrap())
            .collect()
    }
}
