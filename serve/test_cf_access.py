"""serve/cf_access.py: Cloudflare Access JWTs are accepted only when signed by the team's key, for this application's
audience, from the team's issuer and inside their time window."""
import base64
import hashlib
import json
import random
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from serve import cf_access, server  # noqa: E402

AUD = "a" * 64
TEAM = "example"
ISS = "https://example.cloudflareaccess.com"
NOW = 1_800_000_000


def _prime(bits, rnd):
    while True:
        n = rnd.getrandbits(bits) | (1 << (bits - 1)) | 1
        if all(n % p for p in (3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37)) and all(
                pow(a, n - 1, n) == 1 for a in (2, 3, 5, 7, 11, 13, 17, 19, 23)):
            return n


def _keypair(seed):
    rnd = random.Random(seed)
    e = 65537
    while True:
        p, q = _prime(512, rnd), _prime(512, rnd)
        phi = (p - 1) * (q - 1)
        if p != q and phi % e:
            return p * q, e, pow(e, -1, phi)


def _b64(b):
    return base64.urlsafe_b64encode(b).rstrip(b"=").decode()


def _jwk(kid, n, e):
    return {"kid": kid, "kty": "RSA", "alg": "RS256",
            "n": _b64(n.to_bytes((n.bit_length() + 7) // 8, "big")), "e": _b64(e.to_bytes(3, "big"))}


def _sign(n, d, kid, claims, alg="RS256"):
    h = _b64(json.dumps({"alg": alg, "kid": kid}).encode())
    p = _b64(json.dumps(claims).encode())
    k = (n.bit_length() + 7) // 8
    t = cf_access._SHA256_PREFIX + hashlib.sha256(f"{h}.{p}".encode()).digest()
    em = b"\x00\x01" + b"\xff" * (k - len(t) - 3) + b"\x00" + t
    return f"{h}.{p}.{_b64(pow(int.from_bytes(em, 'big'), d, n).to_bytes(k, 'big'))}"


class AccessJwt(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.n, cls.e, cls.d = _keypair(1)
        cls.n2, cls.e2, cls.d2 = _keypair(2)

    def setUp(self):
        self.fetches = 0

        def fetch(url):
            self.assertEqual(url, ISS + "/cdn-cgi/access/certs")
            self.fetches += 1
            return {"keys": [_jwk("k1", self.n, self.e)]}
        self.v = cf_access.AccessVerifier(TEAM, [AUD], fetch=fetch, now=lambda: NOW)

    def claims(self, **kw):
        c = {"aud": [AUD], "iss": ISS, "email": "me@example.com", "iat": NOW - 10, "nbf": NOW - 10, "exp": NOW + 600}
        c.update(kw)
        return c

    def test_valid_token(self):
        got = self.v.verify(_sign(self.n, self.d, "k1", self.claims()))
        self.assertEqual(got["email"], "me@example.com")

    def test_keys_are_cached(self):
        tok = _sign(self.n, self.d, "k1", self.claims())
        self.v.verify(tok)
        self.v.verify(tok)
        self.assertEqual(self.fetches, 1)

    def test_rejections(self):
        cases = {
            "other key": _sign(self.n2, self.d2, "k1", self.claims()),
            "unknown kid": _sign(self.n2, self.d2, "k2", self.claims()),
            "other audience": _sign(self.n, self.d, "k1", self.claims(aud=["b" * 64])),
            "other issuer": _sign(self.n, self.d, "k1", self.claims(iss="https://evil.cloudflareaccess.com")),
            "expired": _sign(self.n, self.d, "k1", self.claims(exp=NOW - 120)),
            "not yet valid": _sign(self.n, self.d, "k1", self.claims(nbf=NOW + 120)),
            "no exp": _sign(self.n, self.d, "k1", {k: v for k, v in self.claims().items() if k != "exp"}),
            "alg none": _sign(self.n, self.d, "k1", self.claims(), alg="none"),
            "garbage": "not.a.jwt",
            "empty": "",
        }
        for name, tok in cases.items():
            with self.subTest(name):
                self.assertIsNone(self.v.verify(tok))

    def test_tampered_payload(self):
        h, _, s = _sign(self.n, self.d, "k1", self.claims()).split(".")
        p = _b64(json.dumps(self.claims(email="attacker@example.com")).encode())
        self.assertIsNone(self.v.verify(f"{h}.{p}.{s}"))

    def test_string_aud_and_env(self):
        self.assertIsNotNone(self.v.verify(_sign(self.n, self.d, "k1", self.claims(aud=AUD))))
        self.assertIsNone(cf_access.from_env({"STRATA_CF_ACCESS_AUD": AUD}))
        discover = cf_access.from_env({"STRATA_CF_ACCESS_TEAM": TEAM})
        discover._fetch = lambda url: {"keys": [_jwk("k1", self.n, self.e)]}
        discover._now = lambda: NOW
        self.assertIsNone(discover.verify(_sign(self.n, self.d, "k1", self.claims())))   # discovery accepts nothing
        v = cf_access.from_env({"STRATA_CF_ACCESS_TEAM": TEAM, "STRATA_CF_ACCESS_AUD": f"x, {AUD}"})
        self.assertEqual((v.issuer, v.auds), (ISS, ["x", AUD]))

    def test_token_of(self):
        self.assertEqual(cf_access.token_of({"Cf-Access-Jwt-Assertion": "a.b.c"}), "a.b.c")
        self.assertEqual(cf_access.token_of({"Cookie": "x=1; CF_Authorization=d.e.f"}), "d.e.f")
        self.assertEqual(cf_access.token_of({}), "")

    def test_non_dict_header(self):
        for bad_h in (None, [1, 2, 3], "string", 123):
            h = _b64(json.dumps(bad_h).encode())
            p = _b64(json.dumps(self.claims()).encode())
            tok = f"{h}.{p}.invalidsig"
            self.assertIsNone(self.v.verify(tok))

    def test_failed_fetch_is_throttled(self):
        calls = 0

        def failing_fetch(url):
            nonlocal calls
            calls += 1
            raise OSError("network down")

        t = NOW
        v = cf_access.AccessVerifier(TEAM, [AUD], fetch=failing_fetch, now=lambda: t)
        tok = _sign(self.n, self.d, "unknown_kid", self.claims())

        # First attempt tries and fails
        self.assertIsNone(v.verify(tok))
        self.assertEqual(calls, 1)

        # Immediate next attempt within REFETCH_MIN does not call fetch again
        t += 10
        self.assertIsNone(v.verify(tok))
        self.assertEqual(calls, 1)

        # After REFETCH_MIN (60s), retry occurs
        t += 65
        self.assertIsNone(v.verify(tok))
        self.assertEqual(calls, 2)




class ServerIntegration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.n, cls.e, cls.d = _keypair(1)

    def setUp(self):
        tok = server.ByteTokenizer()
        self.svc = server.Service(server.MockEngine(tok, "</think>\n\nok", max_context=4096), tok,
                                  server.ChatTemplate(server.ROOT / "serve/chat_template.jinja"))
        self.svc.api_key = "secret-test-key"
        self.v = cf_access.AccessVerifier(TEAM, [AUD],
                                          fetch=lambda url: {"keys": [_jwk("k1", self.n, self.e)]},
                                          now=lambda: NOW)
        self.svc.cf_access = self.v
        self.httpd = server.serve(self.svc, port=0)
        self.port = self.httpd.server_address[1]

    def tearDown(self):
        if self.httpd:
            self.httpd.shutdown()
            self.httpd.server_close()

    def test_cf_access_authorizes_request_without_api_key(self):
        import http.client
        # Without key or CF token: 401
        c = http.client.HTTPConnection("127.0.0.1", self.port, timeout=10)
        c.request("POST", "/v1/chat/completions",
                  body=json.dumps({"messages": [{"role": "user", "content": "hi"}]}).encode(),
                  headers={"Content-Type": "application/json"})
        r = c.getresponse()
        self.assertEqual(r.status, 401)
        r.read()
        c.close()

        # With valid CF-Access-Jwt-Assertion: 200
        valid_tok = _sign(self.n, self.d, "k1", {
            "iss": ISS, "aud": [AUD], "exp": NOW + 3600, "nbf": NOW - 60, "sub": "user@example.com"
        })
        c = http.client.HTTPConnection("127.0.0.1", self.port, timeout=10)
        c.request("POST", "/v1/chat/completions",
                  body=json.dumps({"messages": [{"role": "user", "content": "hi"}]}).encode(),
                  headers={"Content-Type": "application/json", "Cf-Access-Jwt-Assertion": valid_tok})
        r = c.getresponse()
        self.assertEqual(r.status, 200)
        r.read()
        c.close()

    def test_untrusted_origin_with_cf_access_is_rejected(self):
        import http.client
        valid_tok = _sign(self.n, self.d, "k1", {
            "iss": ISS, "aud": [AUD], "exp": NOW + 3600, "nbf": NOW - 60, "sub": "user@example.com"
        })
        # Untrusted Origin should be blocked by CSRF / foreign_page check
        c = http.client.HTTPConnection("127.0.0.1", self.port, timeout=10)
        c.request("POST", "/v1/chat/completions",
                  body=json.dumps({"messages": [{"role": "user", "content": "hi"}]}).encode(),
                  headers={"Content-Type": "application/json",
                           "Cf-Access-Jwt-Assertion": valid_tok,
                           "Origin": "https://evil.com"})
        r = c.getresponse()
        self.assertEqual(r.status, 403)
        r.read()
        c.close()

    def test_explicit_api_key_bypasses_origin_check(self):
        import http.client
        # Valid explicit API key bypasses origin check for developer clients/extensions
        c = http.client.HTTPConnection("127.0.0.1", self.port, timeout=10)
        c.request("POST", "/v1/chat/completions",
                  body=json.dumps({"messages": [{"role": "user", "content": "hi"}]}).encode(),
                  headers={"Content-Type": "application/json",
                           "Authorization": f"Bearer {self.svc.api_key}",
                           "Origin": "https://another-client.example.com"})
        r = c.getresponse()
        self.assertEqual(r.status, 200)
        r.read()
        c.close()

if __name__ == "__main__":
    unittest.main()
