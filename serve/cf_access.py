"""Cloudflare Access login as an opt-in web app credential (STRATA_CF_ACCESS_TEAM).

Behind a Cloudflare Tunnel with an Access application, every request that passed the Access login carries a signed
JWT in the `Cf-Access-Jwt-Assertion` header.  A valid one means the browser already proved who it is to Cloudflare,
so the web app needs no API key there.  Checked here: the RS256 signature against the team's published keys
(`https://<team>.cloudflareaccess.com/cdn-cgi/access/certs`, cached for an hour), the issuer, the application's
audience tag and the token's time window.  Standard library only (PKCS#1 v1.5 verification with Python ints).

Configuration (environment, e.g. environment or config):
  STRATA_CF_ACCESS_TEAM  the team name, "<team>.cloudflareaccess.com" (or the full https:// URL)
  STRATA_CF_ACCESS_AUD   the Access application's Audience (AUD) tag; several may be given, comma separated
Both must be set, otherwise Access tokens are ignored and the API key stays the only credential.  With only the
team set, the server logs the audience tag of the first correctly signed token it sees (discovery) and still accepts
none, so the tag can be read from the log instead of the dashboard.
"""
from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import threading
import time
import urllib.request

CERTS_TTL = 3600          # seconds the team's keys are kept
REFETCH_MIN = 60          # an unknown key id fetches again at most this often
LEEWAY = 60               # clock skew allowed on exp / nbf / iat

# DER DigestInfo prefix of a SHA-256 hash (RFC 8017 9.2 note 1)
_SHA256_PREFIX = bytes.fromhex("3031300d060960864801650304020105000420")


def _b64url(s: str) -> bytes:
    return base64.urlsafe_b64decode(s + "=" * (-len(s) % 4))


def _int(s: str) -> int:
    return int.from_bytes(_b64url(s), "big")


def rsa_sha256_verify(n: int, e: int, data: bytes, sig: bytes) -> bool:
    """RSASSA-PKCS1-v1_5 with SHA-256."""
    k = (n.bit_length() + 7) // 8
    if len(sig) != k:
        return False
    s = int.from_bytes(sig, "big")
    if s >= n:
        return False
    em = pow(s, e, n).to_bytes(k, "big")
    t = _SHA256_PREFIX + hashlib.sha256(data).digest()
    if k < len(t) + 11:
        return False
    expected = b"\x00\x01" + b"\xff" * (k - len(t) - 3) + b"\x00" + t
    return hmac.compare_digest(em, expected)


class AccessVerifier:
    def __init__(self, team: str, auds: list[str], fetch=None, now=time.time):
        team = team.strip().rstrip("/")
        if not team.startswith("https://"):
            team = f"https://{team}" if "." in team else f"https://{team}.cloudflareaccess.com"
        self.issuer = team
        self.certs_url = f"{team}/cdn-cgi/access/certs"
        self.auds = [a.strip() for a in auds if a.strip()]
        self._fetch = fetch or self._http_fetch
        self._now = now
        self._keys: dict[str, tuple[int, int]] = {}
        self._fetched = 0.0
        self._lock = threading.Lock()

    def _http_fetch(self, url: str) -> dict:
        with urllib.request.urlopen(url, timeout=10) as r:
            return json.load(r)

    def _key(self, kid: str) -> tuple[int, int] | None:
        with self._lock:
            age = self._now() - self._fetched
            if age > CERTS_TTL or (kid not in self._keys and age > REFETCH_MIN):
                try:
                    doc = self._fetch(self.certs_url)
                    self._keys = {k["kid"]: (_int(k["n"]), _int(k["e"])) for k in doc.get("keys", [])
                                  if k.get("kty") == "RSA" and "kid" in k}
                    self._fetched = self._now()
                except Exception:
                    pass          # keep the keys we have; a later request tries again
            return self._keys.get(kid)

    def verify(self, token: str) -> dict | None:
        """The token's claims when it is valid for this application, else None."""
        try:
            h64, p64, s64 = token.split(".")
            header = json.loads(_b64url(h64))
            claims = json.loads(_b64url(p64))
            sig = _b64url(s64)
        except Exception:
            return None
        if header.get("alg") != "RS256" or not isinstance(claims, dict):
            return None
        key = self._key(str(header.get("kid", "")))
        if key is None or not rsa_sha256_verify(key[0], key[1], f"{h64}.{p64}".encode("ascii"), sig):
            return None
        now = self._now()
        if claims.get("iss") != self.issuer:
            return None
        aud = claims.get("aud")
        aud = aud if isinstance(aud, list) else [aud]
        if not self.auds and aud and not getattr(self, "_discovered", False):
            self._discovered = True
            print(f"[strata] Cloudflare Access login seen for application AUD {', '.join(map(str, aud))}: set "
                  f"STRATA_CF_ACCESS_AUD to it to accept these logins", flush=True)
        if not any(a in self.auds for a in aud):
            return None
        try:
            if float(claims["exp"]) < now - LEEWAY:
                return None
            if "nbf" in claims and float(claims["nbf"]) > now + LEEWAY:
                return None
        except (KeyError, TypeError, ValueError):
            return None
        return claims


def from_env(env=os.environ) -> AccessVerifier | None:
    team, aud = env.get("STRATA_CF_ACCESS_TEAM", "").strip(), env.get("STRATA_CF_ACCESS_AUD", "").strip()
    if not team:
        return None
    return AccessVerifier(team, aud.split(",") if aud else [])


def token_of(headers) -> str:
    """The Access JWT a request carries: the header Cloudflare adds, else the CF_Authorization cookie."""
    tok = headers.get("Cf-Access-Jwt-Assertion", "")
    if tok:
        return tok.strip()
    for part in headers.get("Cookie", "").split(";"):
        name, _, value = part.strip().partition("=")
        if name == "CF_Authorization":
            return value
    return ""
