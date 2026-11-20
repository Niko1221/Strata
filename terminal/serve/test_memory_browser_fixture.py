"""Private CPU-only browser fixture; run via `python -m serve.test_memory_browser_fixture`.

It uses the production HTTP handler with a scripted MockEngine and only
temporary synthetic profiles. stdin EOF shuts it down and removes profiles.
"""
import json
import os
from pathlib import Path
import sqlite3
import sys
import tempfile
from contextlib import closing

from serve.memory_harness import Harness
from serve.frontend import ChatTemplate
from serve.server import ByteTokenizer, MockEngine, Service, serve


def main():
    with tempfile.TemporaryDirectory(prefix="strata-hakua-browser-") as directory:
        root = Path(directory)
        os.environ["STRATA_BROWSER_A"] = "synthetic-browser-a"
        os.environ["STRATA_BROWSER_B"] = "synthetic-browser-b"
        users = [{"principal": name, "key_env": "STRATA_BROWSER_" + name.upper(), "root": str(root / name)}
                 for name in ("a", "b")]
        tok = ByteTokenizer()
        svc = Service(MockEngine(tok, "synthetic visible answer", max_context=131072), tok,
                      ChatTemplate(Path(__file__).with_name("chat_template.jinja")))
        svc.api_key = ["synthetic-browser-a", "synthetic-browser-b"]
        svc.memory_harness = Harness(users, svc.api_key)
        httpd = serve(svc, port=0)
        try:
            print("PORT " + str(httpd.server_address[1]), flush=True)
            sys.stdin.readline()
            counts = {}
            for name in ("a", "b"):
                db = root / name / "ebbinghaus.db"
                counts[name] = 0
                if db.exists():
                    with closing(sqlite3.connect(db)) as conn:
                        counts[name] = conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0]
            print("COUNTS " + json.dumps(counts), flush=True)
        finally:
            httpd.shutdown()
            httpd.server_close()


if __name__ == "__main__":
    main()
